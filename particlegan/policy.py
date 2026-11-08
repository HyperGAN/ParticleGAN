"""Recipe controls for caller-owned training loops.

The policy never calls ``backward()`` or an optimizer's ``step()``.  One update
is ordered as ``begin_step(real)``; critic forward/penalty/backward/step;
``after_critic_step()``; generator forward/backward;
``after_generator_backward(loss_gan=..., loss_critic=...)``; generator and
table optimizer steps; ``after_generator_step()``; ``finish_step()``.
``loss_critic`` is the adversarial critic loss *without* its penalty.  The
generator hook reads gradients before any clipping or other gradient edits.

The same implementation drives :class:`GANTrainer`.  Caller modules always
contain fast training weights unless the trainer's compatibility serving
swap is explicitly requested.  Use ``served_snapshot()`` for independent
inference weights, and checkpoint only at completed update boundaries.
"""
from copy import copy, deepcopy
from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch
from torch import nn

from .continuous import DataDriftController, LossSurprise, OptimizerSurprise, SettledReopenGuard, StationarityLR
from .recipes import Recipe, learning_rate_scales


def input_noise_std(recipe, completed_steps):
    """Critic input-noise standard deviation for the next update."""
    if recipe.continuous_policy is not None:
        return 0.0
    end = recipe.input_noise_anneal_end * recipe.total_steps
    return float(recipe.input_noise_std * max(0.0, 1.0 - completed_steps / end))


def output_noise_std(recipe, completed_steps):
    """Generator output-noise schedule before applying the policy floor."""
    if recipe.continuous_policy is not None or recipe.output_noise_warmup == 0:
        return float(recipe.output_noise_std)
    return float(recipe.output_noise_std * min(
        1.0, completed_steps / (recipe.output_noise_warmup * recipe.total_steps)))


def _validate_optimizer_state(optimizer, state):
    """Preflight optimizer-owned controls and Adam tensors before live mutation.

    PyTorch's loader checks group counts and casts tensors but does not check
    moment shapes. A copied optimizer alone can consequently accept a state
    that fails on the next update after live weights have already been loaded.
    """
    from .recipe_schedules import validate_plain_adam_state
    validate_plain_adam_state(optimizer, state)
    from .optim.dualnorm import NormalizedOptimizer
    if isinstance(optimizer, NormalizedOptimizer):
        optimizer.validate_state_dict(state)
    ema_critic = getattr(optimizer, "ema_critic", None)
    if ema_critic is not None:
        saved_ema = state.get("regularizer", {}).get("ema") if isinstance(state, dict) else None
        expected_ema = ema_critic.state_dict()
        if not isinstance(saved_ema, dict) or saved_ema.keys() != expected_ema.keys():
            raise ValueError("incompatible checkpoint EMA critic tensors")
        for key, tensor in expected_ema.items():
            value = saved_ema[key]
            if not isinstance(value, torch.Tensor) or value.shape != tensor.shape or value.dtype != tensor.dtype:
                raise ValueError(f"checkpoint EMA critic tensor {key} does not match its dtype or shape")
    # BISPF optimizer loaders also restore a device-local thermal cache. A
    # copied optimizer is only a validation probe: neither a successful probe
    # nor a later rejected trainer checkpoint may change that live runtime state.
    from .bispf import _LAST_THERMAL_SCALE
    thermal_cache = dict(_LAST_THERMAL_SCALE)
    try:
        restored = deepcopy(optimizer)
        restored.load_state_dict(deepcopy(state))
    finally:
        _LAST_THERMAL_SCALE.clear()
        _LAST_THERMAL_SCALE.update(thermal_cache)
    if isinstance(restored, (torch.optim.Adam, torch.optim.AdamW)):
        for group in restored.param_groups:
            for parameter in group["params"]:
                values = restored.state.get(parameter, {})
                if not values:
                    continue
                required = {"step", "exp_avg", "exp_avg_sq"}
                if group.get("amsgrad", False):
                    required.add("max_exp_avg_sq")
                if not required <= values.keys():
                    raise ValueError("incomplete checkpoint Adam state")
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = values.get(key)
                    if key in values and (not isinstance(value, torch.Tensor) or value.shape != parameter.shape):
                        raise ValueError(f"checkpoint Adam {key} does not match its parameter")
                step = values["step"]
                if isinstance(step, torch.Tensor):
                    if step.numel() != 1 or not torch.isfinite(step).all() or bool(step < 0):
                        raise ValueError("invalid checkpoint Adam step")
                elif type(step) not in (int, float) or not math.isfinite(step) or step < 0:
                    raise ValueError("invalid checkpoint Adam step")


def _state_to_device(value, device, memo=None):
    """Move control tensors, preserving repeated references across a device copy."""
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in memo:
            memo[key] = value.to(device=device)
        return memo[key]
    if isinstance(value, dict):
        return {key: _state_to_device(item, device, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_device(item, device, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_device(item, device, memo) for item in value)
    return value


@dataclass(frozen=True)
class StepNoise:
    """Noise values to reuse for both halves of this update.

    ``output_sigma`` retains the learned-noise computation graph.  Compute
    critic fakes under ``torch.no_grad()`` and reuse it in generator fakes.
    """
    input_sigma: float
    output_sigma: object


class ServedModel:
    """Independent inference modules implementing the policy's served law.

    Construct with ``policy.served_model()``.  Both ``sample`` and ``generate``
    retain DV12 latent perturbation; output noise is optional and disabled by
    default, exactly as in ``GANTrainer.sample``.  The private sampling stream
    starts from the policy's evaluation RNG state and never advances a
    training stream.  Module parameters are frozen and modules are in eval.
    A ``ManipulatedPrior`` whose manipulators postprocess generator outputs
    has that postprocessing applied last (after optional output noise) in
    ``generate``/``sample`` and ``routed_forward``, as in ``GANTrainer``'s
    critic and generator paths and the birth/death row generator.
    """

    def __init__(self, models, table, controller, *, source, output_sigma,
                 completed_steps, stream, generation=None, row_semantics="independent", routing=None,
                 feature_sampler=None, backend_selection=None):
        self.models, self.table, self.controller = models, table, controller
        self.source, self.output_sigma = source, output_sigma
        self.completed_steps, self.stream, self.generation = completed_steps, stream, generation
        self.row_semantics = row_semantics
        self.routing = routing
        self.feature_sampler, self.backend_selection = feature_sampler, backend_selection
        self.generator, self.critic = models["generator"], models["critic"]
        self.prior = models.get("prior")
        self.encoder, self.router = models.get("encoder"), models.get("router")

    @torch.no_grad()
    def _generate(self, latent, stream, output_noise, rows=None):
        if self.controller is not None:
            if self.routing is None:
                prior = self.prior if self.prior is not None else SimpleNamespace(z=self.table)
            else:
                candidate = self.routing.candidate_for(self.models, self.table, averaged=self.source == "averaged")
                prior = self.controller.routed_prior(candidate.table, candidate.log_mass)
            if self.feature_sampler is not None:
                latent = self.feature_sampler.perturb_latent(latent, stream, self.controller, prior, rows=rows)
            else:
                latent = self.controller.perturb_latent(latent, stream, prior)
        y = self.generator(latent) if self.generation is None else self.generation(self.generator, latent)
        if output_noise and self.output_sigma != 0:
            y = y + self.output_sigma * torch.randn(
                y.shape, generator=stream, device=y.device, dtype=y.dtype)
        if getattr(self.prior, "postprocesses_generated", False):
            y = self.prior.postprocess_generated(y)
        return y

    def _sampling_stream(self, generator):
        stream = self.stream if generator is None else generator
        if not isinstance(stream, type(torch.default_generator)):
            raise TypeError("sampling generator must be a torch.Generator")
        device = stream.device
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self.table.device:
            raise ValueError("sampling generator must use the snapshot device")
        return stream

    @torch.no_grad()
    def generate(self, latent, *, generator=None, output_noise=False, rows=None):
        """Apply the served latent/output-noise law to caller-supplied latents."""
        if type(output_noise) is not bool:
            raise ValueError("output_noise must be a boolean")
        stream = self._sampling_stream(generator)
        device = self.table.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            return self._generate(latent, stream, output_noise, rows=rows)

    @torch.no_grad()
    def sample(self, n, *, generator=None, output_noise=False):
        """Uniform independent-row samples from the currently served model."""
        if self.row_semantics != "independent":
            raise ValueError("sample requires independent table rows; conditional and densely soft-routed "
                             "snapshots require caller-owned routing; use generate")
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        if type(output_noise) is not bool:
            raise ValueError("output_noise must be a boolean")
        stream = self._sampling_stream(generator)
        device = self.table.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.prior is None:
                indices = torch.randint(len(self.table), (n,), device=device, generator=stream)
                latent = self.table[indices]
            else:
                latent, indices = self.prior.sample(n, generator=stream)
            return self._generate(latent, stream, output_noise, rows=indices)

    @torch.no_grad()
    def routed_forward(self, context, *, perturb=False, output_noise=False, generator=None):
        """The selected conditional bank's clean deterministic forward by default.

        Routing uses unperturbed keys/values. Optional DV12 perturbation acts
        on the mixed code after routing, followed by optional output noise.
        Both options use this snapshot's private sampling stream.
        """
        if self.routing is None:
            raise ValueError("routed_forward requires a RoutedRows serving snapshot")
        if type(perturb) is not bool or type(output_noise) is not bool:
            raise ValueError("perturb and output_noise must be booleans")
        stream = self._sampling_stream(generator)
        candidate = self.routing.candidate_for(self.models, self.table, averaged=self.source == "averaged")
        prior = self.controller.routed_prior(candidate.table, candidate.log_mass) \
            if perturb and self.controller is not None else None
        perturb_fn = (lambda codes: self.controller.perturb_latent(codes, stream, prior)) \
            if perturb and self.controller is not None else None
        device = self.table.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            y = self.routing.forward(self.models, context, candidate, perturb_fn=perturb_fn)
            if output_noise and self.output_sigma != 0:
                y = y + self.output_sigma * torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=stream)
            if getattr(self.prior, "postprocesses_generated", False):
                y = self.prior.postprocess_generated(y)
            return y


class UpdatePolicy:
    """Checkpointable controls with explicit parameter and optimizer ownership.

    ``roles`` is a list of role lists, aligned with the parameter groups of
    ``generator_optimizer``, ``critic_optimizer`` and, when distinct,
    ``table_optimizer``.  Roles are ``generator``, ``encoder``, ``router``,
    ``critic``, ``table`` and ``noise``; homogeneous groups are required.
    Omit roles to infer them from the supplied modules and table.  An optional
    ``generation(model, latent)`` callback supplies the clean generator
    forward; ``critic_features(samples)`` supplies learned critic features
    for birth/death.  These callbacks must preserve row order and implement
    the declared row semantics.  Stateful callback resources are caller-owned
    and must be checkpointed separately.

    The table may be an independent tensor, ``prior.z``, or a parameter
    registered in one supplied generator/encoder/router module. An embedded
    table still needs its own optimizer group; its average and served copy
    point to the corresponding copied module tensor and are updated once.
    Supplied modules cannot share parameters or the particle table (do not
    supply both an enclosing generator and its embedded prior as separate
    owners). Generator-gradient alignment reads G's network parameters only,
    excluding an embedded table and separate encoder/router parameters.

    Every parameter and buffer must be on the table's device. Trainable
    floating tensors must use the table's dtype; frozen parameters and
    buffers may retain their own floating precision (for example BF16 frozen
    features with FP32 heads and table). The caller handles casts at module
    boundaries. Frozen parameters and buffers are copied exactly into averages;
    EMA arithmetic applies only to trainable parameters. Checkpoint restore
    requires the same per-tensor dtypes and parameter freezing.

    The default ``row_policy='independent'`` requires uniform table atoms:
    a draw chooses one row i and generates G(z_i). Conditional and densely
    blended banks require ``row_policy='routed_paired'`` and an explicit
    ``routed_rows=RoutedRows(...)`` contract for evidence and restructuring.
    That adaptation observes caller-provided fit contexts and protected guard
    contexts via ``begin_step(real, routed=RoutedBatch(...))``. Its callbacks
    route the current bank, then decode mixed codes; DV12 perturbs the mixed
    codes after routing. ``routed_generate(context, sigma=0, perturb=False)``
    and ``served_model().routed_forward(context)`` provide clean forwards.
    A complete ``model_forward`` callback can use ordered named routing
    sites, with DV12 draws applied to each site's mixed token codes. With
    both row controls disabled, a fixed frozen bank needs no ``RoutedBatch``
    observations, gradient evidence or counterfactual probes.
    Conditional loops with row controls disabled can use ``UpdatePolicy``
    without a routed row contract.

    ``row_evidence_statistic`` declares the independent-row evidence gate's
    test statistic (``particlegan.row_evidence``). The default
    ``"sphericity"`` keeps the legacy trajectory of every qualified
    row-evidence recipe; ``"hotelling"`` is a declared candidate (full
    covariance T^2, float64 n x d x d accumulators). It is a trainer option,
    not a recipe field, and the row-evidence checkpoint configuration records
    it, so a mismatched resume is rejected.
    """

    _STREAMS = ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")
    _ROLES = {"generator", "encoder", "router", "critic", "table", "noise"}

    def __init__(self, recipe, generator, critic, *, generator_optimizer,
                 critic_optimizer, prior=None, table=None, table_optimizer=None,
                 roles=None, generation=None, critic_features=None, encoder=None,
                 router=None, row_semantics="independent", seed=0, streams=None,
                 penalty=None, schedule=None, routed_rows=None,
                 allow_shared_training_streams=False, row_evidence_statistic="sphericity"):
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
        if not isinstance(generator, nn.Module) or not isinstance(critic, nn.Module):
            raise TypeError("generator and critic must be torch modules")
        self.recipe, self.G, self.D = recipe, generator, critic
        self.prior, self.encoder, self.router = prior, encoder, router
        table = getattr(prior, "z", None) if table is None else table
        if (not isinstance(table, torch.Tensor) or table.ndim != 2
                or table.shape != (recipe.num_particles, recipe.z_dim)):
            raise ValueError("table must have shape (recipe.num_particles, recipe.z_dim)")
        if prior is not None and getattr(prior, "z", None) is not table:
            raise ValueError("prior.z and table must be the same tensor")
        if prior is not None and table.requires_grad != (recipe.prior_update == "learned"):
            raise ValueError("prior learning state contradicts Recipe.prior_update")
        self.table, self.device, self.dtype = table, table.device, table.dtype
        self.num_particles = recipe.num_particles
        if not table.is_floating_point():
            raise ValueError("table must use a floating dtype")
        self.row_policy = getattr(recipe, "row_policy", "independent")
        self._routed_rows = routed_rows
        self._routed_controls_enabled = recipe.row_evidence_gate or recipe.particle_birth_death
        if self.row_policy == "routed_paired":
            if routed_rows is None:
                raise ValueError("row_policy='routed_paired' requires an explicit RoutedRows contract")
            from .routing import RoutedRows
            if not isinstance(routed_rows, RoutedRows):
                raise TypeError("routed_rows must be a RoutedRows contract")
            if self._routed_controls_enabled and not table.requires_grad:
                raise ValueError("routed row evidence and birth/death require a trainable particle table; "
                                 "a frozen table requires both row controls disabled")
            if row_semantics == "independent":
                row_semantics = "dense_soft"
        elif routed_rows is not None:
            raise ValueError("routed_rows requires recipe.row_policy='routed_paired'")
        self.row_semantics = row_semantics
        if row_semantics not in ("independent", "conditional", "dense_soft"):
            raise ValueError("row_semantics must be independent, conditional or dense_soft")
        if (self.row_policy == "independent" and (recipe.row_evidence_gate or recipe.particle_birth_death)
                and (row_semantics != "independent" or recipe.conditioning != "scalar"
                     or recipe.encoder_mode != "none")):
            raise ValueError("row evidence and birth/death require independent unconditional table atoms; "
                             "conditional and densely soft-routed banks are unsupported")
        if generation is not None and not callable(generation):
            raise TypeError("generation must be callable")
        if critic_features is not None and not callable(critic_features):
            raise TypeError("critic_features must be callable")
        self.generation, self.critic_features = generation, critic_features
        self.opt_g, self.opt_d = generator_optimizer, critic_optimizer
        self.table_optimizer = self.opt_g if table_optimizer is None else table_optimizer
        self.optimizers = [self.opt_g, self.opt_d]
        if self.table_optimizer not in self.optimizers:
            self.optimizers.append(self.table_optimizer)
        if self.opt_g is self.opt_d or self.table_optimizer is self.opt_d:
            raise ValueError("critic and generator/table optimizers must be distinct")
        self._validate_modules()
        self.roles = self._parameter_roles(roles)
        recipe._validate_frozen_prior_ownership(prior, self.G, self.D, encoder, router,
                                               optimizers=self.optimizers)
        # Adding noise last reproduces GANTrainer's optimizer group ordering.
        self.log_output_sigma = None
        self.last_output_sigma = None
        if recipe.output_noise_mode == "learnable":
            self.log_output_sigma = nn.Parameter(torch.full(
                (), math.log(recipe.output_noise_std), device=self.device, dtype=self.dtype))
            self.opt_g.add_param_group({"params": [self.log_output_sigma], "lr": recipe.lr})
            self.roles[0].append("noise")
        self.initial_lrs = [[group["lr"] for group in optimizer.param_groups]
                            for optimizer in self.optimizers]
        if any(not math.isfinite(rate) or rate <= 0 for row in self.initial_lrs for rate in row):
            raise ValueError("policy base learning rates must be positive and finite")
        self.ema_G, self.ema_prior = deepcopy(self.G).eval(), (None if prior is None else deepcopy(prior).eval())
        self.ema_encoder = None if encoder is None else deepcopy(encoder).eval()
        self.ema_router = None if router is None else deepcopy(router).eval()
        for module in self._average_modules().values():
            module.requires_grad_(False)
        self.averaged_table = (table.detach().clone() if self._table_location is None else
                               self._table_in_modules(self._average_modules()))
        self._fast = None
        streams = dict(streams or {})
        if set(streams) - set(self._STREAMS):
            raise ValueError("unknown policy RNG stream")
        for name, offset in zip(self._STREAMS, (2, 3, 4, 5)):
            setattr(self, name, self._stream(streams.get(name), seed + offset))
        # The penalty stream is reserved and performs no draws; native image
        # hosts historically share it with the latent stream. Active training
        # and evaluation streams still require independent ownership.
        active_streams = (self.latent_generator, self.noise_generator, self.eval_generator)
        if type(allow_shared_training_streams) is not bool:
            raise TypeError("allow_shared_training_streams must be a boolean")
        adaptive = (recipe.continuous_policy is not None or recipe.lr_control == "stationarity"
                    or recipe.row_evidence_gate or recipe.particle_birth_death or recipe.serve_average > 0)
        if allow_shared_training_streams and adaptive:
            raise ValueError("adaptive policies require distinct active training streams")
        if (self.eval_generator in (self.latent_generator, self.noise_generator)
                or (not allow_shared_training_streams
                    and len({id(stream) for stream in active_streams}) != len(active_streams))):
            raise ValueError("active policy RNG streams must be distinct")
        self.completed_steps = 0
        self.controller = (None if recipe.continuous_policy is None else
                           DataDriftController(recipe.continuous_policy))
        if self.controller is not None:
            self.controller.observe_prior(self._prior_view())
            # Penalties constructed after the policy bind the same controller;
            # attach_penalty handles the reverse construction order.
            self.opt_d.continuous_controller = self.controller
        self.lr_settle = (StationarityLR(self.optimizers, prior_param=table,
                                        release_rule=recipe.table_release_rule)
                          if recipe.lr_control == "stationarity" else None)
        self.routed_control = None
        self.surprise = (OptimizerSurprise() if self.lr_settle is not None and recipe.reopen_signal == "optimizer"
                         else None)
        self.loss_surprise = (LossSurprise() if self.lr_settle is not None and recipe.reopen_signal == "optimizer"
                              else None)
        self.reopen_guard = None if recipe.reopen_guard is None else SettledReopenGuard()
        if self.reopen_guard is not None:
            self.reopen_guard.observe_epoch(self._loss_epoch(), self.surprise)
        self.birth_death = None
        self.row_evidence = None
        # Declared trainer option (review finding #11): the row-evidence test statistic. "sphericity" is the
        # legacy law of every qualified row-evidence recipe; "hotelling" is the full-covariance T^2 candidate.
        if row_evidence_statistic not in ("sphericity", "hotelling"):
            raise ValueError("row_evidence_statistic must be sphericity or hotelling")
        if row_evidence_statistic != "sphericity" and (
                not recipe.row_evidence_gate or self.row_policy == "routed_paired"):
            raise ValueError("row_evidence_statistic applies only to the independent-row row_evidence_gate")
        if self.row_policy == "routed_paired":
            self.routed_control = routed_rows.bind(
                models=self._training_modules(), averaged_models=self._average_modules(),
                table=table, averaged_table=self.averaged_table, optimizers=self.optimizers,
                table_optimizer=self.table_optimizer, seed=seed + 6,
                completed_steps=lambda: self.completed_steps, controller=self.controller,
                allow_frozen_table=not self._routed_controls_enabled)
            if recipe.particle_birth_death:
                self.routed_control.validate_optimizer_transport()
                self.birth_death = self.routed_control
            if recipe.row_evidence_gate:
                self.row_evidence = self.routed_control.evidence
            row_parameters = set(self.routed_control.row_parameters.values())
            for optimizer, testers in zip(self.optimizers, self.lr_settle.testers):
                for group, tester in zip(optimizer.param_groups, testers):
                    if (tester is not None and len(group["params"]) == 1
                            and group["params"][0] in row_parameters):
                        tester.rows = len(table)
        elif recipe.particle_birth_death:
            cells = recipe.birth_death_cells if recipe.birth_death_backend != "knn" else None
            if cells is not None and len(table) <= cells:
                self.birth_death = None
            else:
                from .birth_death import ParticleRows, ParticleBirthDeath, ScalarHeadFeatures
                features = critic_features
                if recipe.birth_death_space == "critic" and features is None:
                    features = ScalarHeadFeatures(self.D)
                rows = ParticleRows(table=table, optimizer=self.table_optimizer,
                                    averaged_table=self.averaged_table,
                                    generate=lambda latent: self._postprocess_generated(
                                        self._clean_generate(self.G, latent)),
                                    critic_features=features,
                                    evaluation_modules=tuple(module for module in (self.G, encoder, router)
                                                             if module is not None),
                                    controller=self.controller, completed_steps=lambda: self.completed_steps,
                                    semantics=row_semantics)
                self.birth_death = ParticleBirthDeath(
                    rows, seed + 6, space=recipe.birth_death_space,
                    isolation=recipe.birth_death_isolation,
                    feature_scale=recipe.birth_death_feature_scale,
                    cells=cells)
        if recipe.row_evidence_gate and self.routed_control is None:
            from .row_evidence import RowEvidence
            self.row_evidence = RowEvidence(table, null=recipe.row_evidence_null,
                                            statistic=row_evidence_statistic)
        self.penalty = None
        if penalty is None and hasattr(self.opt_d, "record"):
            penalty = recipe.make_critic_penalty(self.opt_d)
        if penalty is not None:
            self.attach_penalty(penalty)
        self.schedule = learning_rate_scales if schedule is None else schedule
        self._feature_selection = None
        if recipe.birth_death_backend != "knn":
            from .feature_policy import FeatureSelection
            self._feature_selection = FeatureSelection(self, seed + 6)
        self._phase = "ready"
        self._noise = None
        self._stray_flags = self._hot = self._z_before = None
        from .k3p import _scope_reference_adam_graph_checks
        for optimizer in self.optimizers:
            _scope_reference_adam_graph_checks(optimizer)

    def _training_modules(self):
        return {name: module for name, module in (
            ("generator", self.G), ("critic", self.D), ("prior", self.prior),
            ("encoder", self.encoder), ("router", self.router)) if module is not None}

    def _average_modules(self):
        return {name: module for name, module in (
            ("generator", self.ema_G), ("prior", self.ema_prior),
            ("encoder", self.ema_encoder), ("router", self.ema_router)) if module is not None}

    def _validate_modules(self):
        seen = set()
        self._table_location = None
        table_owners = set()
        for name, module in self._training_modules().items():
            for tensor in (*module.parameters(), *module.buffers()):
                if tensor.device != self.device:
                    raise ValueError("policy modules and table must share one device")
                if tensor.requires_grad and tensor.is_floating_point() and tensor.dtype != self.dtype:
                    raise ValueError("trainable policy tensors must use the table's floating dtype; "
                                     "frozen parameters and buffers may retain their own precision")
            for key, parameter in module.named_parameters():
                if id(parameter) in seen:
                    raise ValueError("policy modules must not share parameters")
                seen.add(id(parameter))
                if parameter is self.table:
                    if name == "critic":
                        raise ValueError("the particle table cannot be a critic parameter")
                    self._table_location = (name, key)
                    table_owners.add(name)
            for key, buffer in module.named_buffers():
                if buffer is self.table:
                    if name == "critic":
                        raise ValueError("the particle table cannot be a critic buffer")
                    self._table_location = (name, key)
                    table_owners.add(name)
        if len(table_owners) > 1:
            raise ValueError("policy modules must not share the particle table")

    def _table_in_modules(self, modules):
        name, key = self._table_location
        module = modules[name]
        return dict([*module.named_parameters(), *module.named_buffers()])[key]

    def _parameter_roles(self, roles):
        owners = {}
        for name, module in self._training_modules().items():
            if name != "prior":
                owners.update({id(p): name for p in module.parameters()})
            else:
                owners.update({id(p): "router" for p in module.parameters() if id(p) != id(self.table)})
        owners[id(self.table)] = "table"
        if roles is not None and (not isinstance(roles, (list, tuple)) or len(roles) != len(self.optimizers)):
            raise ValueError("roles must match policy optimizers and parameter groups")
        result, seen, table_group = [], set(), None
        for index, optimizer in enumerate(self.optimizers):
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("policy optimizers must be torch optimizers")
            row = []
            if roles is not None and len(roles[index]) != len(optimizer.param_groups):
                raise ValueError("roles must match policy optimizers and parameter groups")
            for group_index, group in enumerate(optimizer.param_groups):
                inferred = {owners.get(id(p)) for p in group["params"]}
                role = next(iter(inferred)) if len(inferred) == 1 else None
                if roles is not None:
                    explicit = "table" if roles[index][group_index] == "prior" else roles[index][group_index]
                    if explicit not in self._ROLES or any(value != explicit for value in inferred):
                        raise ValueError("optimizer groups must contain only parameters of their explicit role")
                    role = explicit
                if role is None or role not in self._ROLES:
                    raise ValueError("optimizer groups require one known generator/encoder/router/critic/table role")
                if (index == 1) != (role == "critic"):
                    raise ValueError("critic parameters must belong to critic_optimizer only")
                for parameter in group["params"]:
                    if id(parameter) in seen:
                        raise ValueError("parameters must have exactly one optimizer owner")
                    seen.add(id(parameter))
                if role == "table":
                    if optimizer is not self.table_optimizer or len(group["params"]) != 1:
                        raise ValueError("the table requires its own group in table_optimizer")
                    table_group = (index, group_index)
                row.append(role)
            result.append(row)
        if self.table.requires_grad and table_group is None:
            raise ValueError("trainable table must be owned by table_optimizer")
        for name, module in self._training_modules().items():
            if any(p.requires_grad and id(p) not in seen for p in module.parameters()):
                raise ValueError(f"trainable {name} parameters require optimizer ownership")
        return result

    def _prior_view(self, averaged=False):
        if self.row_policy == "routed_paired":
            models = self._average_modules() if averaged else self._training_modules()
            table = self.averaged_table if averaged else self.table
            candidate = self._routed_rows.candidate_for(models, table, averaged=averaged)
            return DataDriftController.routed_prior(candidate.table, candidate.log_mass)
        module = self.ema_prior if averaged else self.prior
        return module if module is not None else SimpleNamespace(z=self.averaged_table if averaged else self.table)

    def _stream(self, generator, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed) if generator is None else generator
        if not isinstance(generator, type(torch.default_generator)):
            raise TypeError("policy streams must be torch.Generator instances")
        device = generator.device
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self.device:
            raise ValueError("random generators must use the model device")
        return generator

    def attach_penalty(self, penalty):
        """Connect the critic penalty to this policy's shared continuous controller."""
        if getattr(penalty, "optimizer", self.opt_d) is not self.opt_d:
            raise ValueError("penalty must belong to critic_optimizer")
        self.penalty = penalty
        if self.controller is not None and self.recipe.continuous_policy in (
                "dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            penalty.regularizer.continuous_controller = self.controller
        return penalty

    def begin_step(self, real, *, game_record=None, routed=None, execution_limit=None):
        """Observe critic-real rows, set LRs, take anchors, and compute noise.

        Call before any critic forward.  ``game_record`` is the KA2 optimizer's
        previous game record (defaults to ``critic_optimizer.record``).
        It is read before the real-data observation, exactly as in the native
        loop.  This observation also feeds the birth/death FIFO reservoir.
        Routed row controls require ``RoutedBatch`` fit/guard observations.
        With both row controls disabled, ``routed`` is optional and no row
        observations or reservoirs are updated; a frozen bank is supported.
        ``execution_limit`` is an external update cap; it does not change the
        recipe's schedule horizon. Omit it to use the recipe's ordinary cap.
        """
        if self.interrupted:
            raise RuntimeError("policy update was interrupted; restore a completed-update checkpoint before continuing")
        if self._phase != "ready":
            raise RuntimeError("finish_step must complete the previous policy update")
        if self.routed_control is not None:
            if self._routed_controls_enabled and routed is None:
                raise ValueError("routed_paired updates require RoutedBatch fit and separate guard contexts")
            from .routing import RoutedBatch
            if routed is not None and not isinstance(routed, RoutedBatch):
                raise TypeError("routed must be a RoutedBatch")
        elif routed is not None:
            raise ValueError("routed observations require recipe.row_policy='routed_paired'")
        self._serve_release()
        recipe = self.recipe
        limit = recipe.total_steps if execution_limit is None else execution_limit
        if execution_limit is not None and (type(execution_limit) is not int or execution_limit < 1):
            raise ValueError("execution_limit must be a positive integer or None")
        if limit is not None and self.completed_steps >= limit:
            raise RuntimeError("recipe training budget exhausted")
        if (not isinstance(real, torch.Tensor) or real.ndim < 2 or not len(real)
                or real.device != self.device or real.dtype != self.dtype):
            raise ValueError("real must be a nonempty batch on the model device and dtype")
        real = real.detach()
        # Optional observations are pure preflight when row controls are off.
        # Reject them before a started update can require checkpoint recovery.
        if self.routed_control is not None and not self._routed_controls_enabled and routed is not None:
            self.routed_control.check_batch(routed)
        self._phase = "starting"
        if self._feature_selection is not None:
            self._feature_selection.observe_shape(real)
        if self.routed_control is not None and self._routed_controls_enabled:
            self.routed_control.begin(routed)
        if self.controller is not None:
            self.controller.observe_prior(self._prior_view())
            record = getattr(self.opt_d, "record", None) if game_record is None else game_record
            if record is None:
                raise ValueError("continuous policy requires the previous critic game_record")
            self.controller.observe_game(record)
        if self.birth_death is not None and self.routed_control is None:
            self.birth_death.observe_real(real)
        from .recipe_schedules import apply_training_schedules
        apply_training_schedules(self.completed_steps, recipe, self.optimizers, self.penalty)
        if self.lr_settle is None:
            network, prior_scale = (self.schedule(self.completed_steps, recipe)
                                    if self.controller is None else self.controller.observe_real(real))
            for optimizer, rates, roles in zip(self.optimizers, self.initial_lrs, self.roles):
                for group, rate, role in zip(optimizer.param_groups, rates, roles):
                    group["lr"] = rate * (prior_scale if role == "table" else network)
        else:
            if recipe.reopen_signal == "data":
                self.controller.observe_real(real)
                reopen = self.controller.data_score > 3.
            elif recipe.reopen_signal == "optimizer":
                self.controller.observe_blind()
                loss_reopen = False
                if self.loss_surprise is not None:
                    loss_reopen = self.loss_surprise.decide(self.completed_steps, guard=self.reopen_guard)
                loss_surprised = self.loss_surprise is not None and self.loss_surprise.is_surprised
                if self.reopen_guard is None:
                    opt_reopen = self.surprise.decide(self.completed_steps)
                else:
                    opt_reopen = self.surprise.decide(self.completed_steps, guard=self.reopen_guard,
                                                      network=self._contracted_network(),
                                                      loss_surprised=loss_surprised)
                reopen = opt_reopen or loss_reopen
                if reopen:
                    self._reopen_moments()
                if recipe.reopen_anchor == "release":
                    self._anchor_release(opt_reopen)
            else:
                self.controller.observe_blind()
                reopen = False
            for group, tester in self.lr_settle.pairs(self.optimizers):
                if reopen:
                    tester.restart(group["params"], reopen=True)
                else:
                    tester.begin(group["params"])
            scales = [tester.s for index, row in enumerate(self.lr_settle.testers) if index != 1
                      for tester, role in zip(row, self.roles[index])
                      if role == "table" and tester is not None and tester.s is not None]
            prior_scale = max(scales) if scales else None
            for index, (optimizer, rates, testers) in enumerate(
                    zip(self.optimizers, self.initial_lrs, self.lr_settle.testers)):
                for group, rate, tester in zip(optimizer.param_groups, rates, testers):
                    scale = 1. if tester is None else tester.s
                    if index == 1 and prior_scale is not None:
                        scale = max(scale, 0.75 * prior_scale)
                    group["lr"] = rate * scale
        if self.controller is not None and recipe.critic_payoff_damping:
            for group in self.opt_d.param_groups:
                group["lr"] *= self.controller.critic_scale()
        flags, hold, tester = self._stray_gate()
        if tester is not None and hasattr(tester, "hold_descent"):
            tester.exclude = flags if recipe.row_evidence_exclude else None
            tester.hold_descent = hold
        if hold:
            self.row_evidence.counters["hold_steps"] = self.row_evidence.counters.get("hold_steps", 0) + 1
        self._stray_flags = flags
        sigma = self._output_sigma(output_noise_std(recipe, self.completed_steps), detach=False)
        self.last_output_sigma = float(sigma.detach() if torch.is_tensor(sigma) else sigma)
        self._noise = StepNoise(input_noise_std(recipe, self.completed_steps), sigma)
        self._phase = "critic"
        return self._noise

    before_step = begin_step

    def observe_support(self, latent):
        """Optional DV11 probe; E22/DV12 performs no probe or RNG draw here."""
        if self.controller is not None:
            sigma = self._noise.output_sigma if self._noise is not None else self.output_sigma()
            self.controller.observe_support(self.G, self.D, latent, sigma, self.noise_generator)

    def observe_critic_pair(self, real, fake):
        """Observe the detached critic pair before its loss/backward (DV8/9)."""
        if self._phase != "critic":
            raise RuntimeError("critic observations must precede after_critic_step")
        if self.controller is not None:
            self.controller.observe_pair(real, fake)

    def before_critic_backward(self):
        """Optional ordering check immediately before critic backward."""
        if self._phase != "critic":
            raise RuntimeError("critic backward must follow begin_step")

    after_critic_backward = before_critic_backward

    def after_critic_step(self):
        """Observe applied critic displacement after its optimizer's step."""
        if self._phase != "critic":
            raise RuntimeError("after_critic_step must follow the critic optimizer step")
        if self.reopen_guard is not None:
            # The completed penalty call owns the epoch, including lazy calls
            # and multiple caller roles. Rebase before changed-loss q is queued.
            self.reopen_guard.observe_epoch(self._loss_epoch(), self.surprise)
        if self.lr_settle is not None:
            self._settle_observe(1)
        self._phase = "generator"

    def before_generator_backward(self):
        """Optional ordering check immediately before generator backward."""
        if self._phase != "generator":
            raise RuntimeError("generator backward must follow after_critic_step")

    def after_generator_backward(self, *, loss_gan, loss_critic):
        """Observe unmodified generator and table gradients before their steps.

        The payoff, gradient alignment and row evidence use the same backward
        pass.  Capture the hot rows' pre-step values last; the next hook
        restores their unscaled motion before stationarity observes it.
        """
        if self._phase != "generator":
            raise RuntimeError("after_generator_backward must follow generator backward")
        if self.controller is not None:
            # Embedded table parameters retain their explicit table role and
            # do not enter the network-gradient alignment signal.
            generator = self.G
            if self._table_location is not None and self._table_location[0] == "generator":
                generator = SimpleNamespace(parameters=lambda: (p for p in self.G.parameters() if p is not self.table))
            self.controller.observe_generator(generator, loss_gan.detach(), loss_critic.detach())
        if self.loss_surprise is not None:
            self.loss_surprise.observe(float(loss_gan.detach().cpu()))
        self._hot = self._z_before = None
        tester = self._table_tester()
        if self.routed_control is not None and self._routed_controls_enabled:
            self.routed_control.observe_backward(self.table.grad)
        elif self.row_evidence is not None:
            self.row_evidence.update(self.table.grad)
        if self.row_evidence is not None:
            if (self.recipe.row_evidence_hot and self._stray_flags is not None and tester is not None
                    and tester.s < 1.0 and bool(self._stray_flags.any())):
                self._hot = self._stray_flags
                self._z_before = self.table.detach().clone()
        self._phase = "generator_step"

    def after_generator_step(self):
        """Apply hot-row correction, then observe all non-critic displacements."""
        if self._phase != "generator_step":
            raise RuntimeError("after_generator_step must follow after_generator_backward and optimizer steps")
        if self._hot is not None:
            with torch.no_grad():
                hot, before, tester = self._hot, self._z_before, self._table_tester()
                z = self.table
                z[hot] = before[hot] + (z[hot] - before[hot]) * (1.0 / tester.s)
            self.row_evidence.counters["hot_row_steps"] = self.row_evidence.counters.get("hot_row_steps", 0) + int(hot.sum())
        if self.lr_settle is not None:
            for index in range(len(self.optimizers)):
                if index != 1:
                    self._settle_observe(index)
        self._phase = "finish"

    @torch.no_grad()
    def finish_step(self):
        """Update averages, restructure particles, rebase evidence, then count.

        Averages are updated *before* birth/death.  A move clones the parent's
        fast and averaged table rows plus optimizer/history rows, then rebases
        stationarity and resets moved-row evidence.  Serving subsequently
        chooses averaged weights only when the table last decided STATIONARY.
        Routed counterfactual evidence is refreshed even when birth/death is
        disabled; that evidence-only mode never proposes or mutates rows.
        """
        if self._phase != "finish":
            raise RuntimeError("finish_step must follow after_generator_step")
        recipe = self.recipe
        rate = self._average_rate() if recipe.serve_average > 0 else None
        fast_modules = self._training_modules()
        for name, target in self._average_modules().items():
            source = fast_modules[name]
            for averaged, current in zip(target.parameters(), source.parameters()):
                if not current.requires_grad:
                    averaged.copy_(current)
                elif rate is None:
                    averaged.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
                else:
                    averaged.mul_(1.0 - rate).add_(current, alpha=rate)
            for averaged, current in zip(target.buffers(), source.buffers()):
                averaged.copy_(current)
            if name == "prior" and hasattr(target, "set_sigma"):
                target.set_sigma(source.sigma)
        if self._table_location is None:
            if not self.table.requires_grad:
                self.averaged_table.copy_(self.table)
            elif rate is None:
                self.averaged_table.mul_(recipe.ema_decay).add_(self.table, alpha=1 - recipe.ema_decay)
            else:
                self.averaged_table.mul_(1.0 - rate).add_(self.table, alpha=rate)
        event = None
        if self.routed_control is not None:
            if self._routed_controls_enabled:
                event = self.routed_control.maybe_apply(mutate=recipe.particle_birth_death)
            if event and event.get("moves") and self.lr_settle is not None:
                self._routed_rebase()
        elif self.birth_death is not None:
            if getattr(self.birth_death, "cells", None) is not None and len(self.table) <= self.birth_death.cells:
                event = None
            elif self._feature_selection is not None and self._feature_selection.state["actual_backend"] == "feature_cells":
                event = self.birth_death.maybe_apply(self._feature_selection.facade, self.last_output_sigma)
            else:
                event = self.birth_death.maybe_apply(self.last_output_sigma)
            if event and event.get("moves") and self.lr_settle is not None:
                for index, row in enumerate(self.lr_settle.testers):
                    for group, tester, role in zip(self.optimizers[index].param_groups, row, self.roles[index]):
                        if role == "table" and tester is not None:
                            tester.rebase(group["params"], self.birth_death.moved_rows)
                if self.row_evidence is not None:
                    self.row_evidence.reset(self.birth_death.moved_rows)
        self._clear_step()
        self.completed_steps += 1
        self._phase = "ready"
        return event

    def _routed_rebase(self):
        """Remove non-gradient bank moves from row-local and coupled windows."""
        moved = self.routed_control.moved_parameters
        for optimizer, testers, roles in zip(self.optimizers, self.lr_settle.testers, self.roles):
            for group, tester, role in zip(optimizer.param_groups, testers, roles):
                if tester is None:
                    continue
                affected = [parameter for parameter in group["params"] if parameter in moved]
                if affected:
                    if len(group["params"]) == 1 and tester.rows is not None:
                        tester.rebase(group["params"], moved[affected[0]])
                    else:
                        tester.restart(group["params"])
                elif getattr(self.routed_control, "restart_router", False) and role in ("encoder", "router"):
                    tester.restart(group["params"])

    @property
    def interrupted(self):
        """Whether an aborted update requires a completed-checkpoint restore."""
        return self._phase == "interrupted"

    def abort_step(self):
        """Clear pending bookkeeping and latch an unfinished update as interrupted.

        Parameters, optimizer state, controller observations and RNG draws are
        retained. Restore a completed-update checkpoint before training or
        checkpointing again. An idle boundary is not marked interrupted.
        """
        if self._phase != "ready":
            self._phase = "interrupted"
        self._clear_step()

    def _clear_step(self):
        for optimizer in self.optimizers:
            clear = getattr(optimizer, "clear_sampled_rows", None)
            if clear is not None:
                clear()
        self._noise = None
        self._stray_flags = self._hot = self._z_before = None

    def _table_tester(self):
        if self.lr_settle is not None:
            for row, roles in zip(self.lr_settle.testers, self.roles):
                for tester, role in zip(row, roles):
                    if role == "table" and tester is not None:
                        return tester
        return None

    def _stray_gate(self):
        evidence, tester = self.row_evidence, self._table_tester()
        if evidence is None or tester is None or not evidence.valid:
            return None, False, tester
        hold = evidence.fraction > evidence.Q and self.recipe.row_evidence_hold
        return (None if hold else evidence.flag), hold, tester

    def _settle_observe(self, index):
        for j, (group, rate, tester) in enumerate(zip(self.optimizers[index].param_groups,
                                                     self.initial_lrs[index], self.lr_settle.testers[index])):
            if tester is not None:
                tester.observe(group["params"], group["lr"] / rate, step=self.completed_steps + 1)
                if self.surprise is not None:
                    self.surprise.observe(f"{index}.{j}", self.optimizers[index], group)

    def _contracted_network(self):
        return {f"{i}.{j}": {"role": role, "scale": tester.s}
                for i, (testers, roles) in enumerate(zip(self.lr_settle.testers, self.roles))
                for j, (tester, role) in enumerate(zip(testers, roles))
                if (tester is not None and role in SettledReopenGuard.NETWORK_ROLES and tester.s < 1.)}

    def _loss_epoch(self, optimizer_state=None):
        from .ka2 import KA2StepRecord
        record = getattr(self.opt_d, "record", None)
        if not isinstance(record, KA2StepRecord):
            return None
        if optimizer_state is None:
            return record.anchor_started
        return optimizer_state["regularizer"]["record"]["anchor_started"]

    def _anchor_release(self, reopen):
        """reopen_anchor="release": a re-open is evidence that the game moved, so the KA2 anchor may follow KA2's own
        release rules instead of being forced on. Latched from the fire (only once KA2's blended anchor exists) until
        KA2's surprise ratio has risen above its release level and come back below its return level. While latched
        the controller's drift evidence reads 1 (set after observe_blind, so mobility and data memory are untouched;
        it is read by the next KA2 penalty call and the next observe_game)."""
        from .ka2 import REL_HI, REL_LO
        record = getattr(self.opt_d, "record", None)
        if record is None or self.controller is None:
            return
        surprise = self.surprise
        if reopen and record.last_ratio is not None:
            surprise.anchor_event = [False]
            surprise.anchor_events += 1
        if surprise.anchor_event is None:
            return
        ratio = record.last_ratio
        if ratio is not None and ratio > REL_HI:
            surprise.anchor_event[0] = True
        if surprise.anchor_event[0] and ratio is not None and ratio < REL_LO:
            surprise.anchor_event = None
            return
        self.controller.data_drive = 1.0

    @torch.no_grad()
    def _reopen_moments(self):
        """Re-open: every group's Adam second moments (and AMSGrad max) shrink by that group's own
        observed surprise ratio r squared, so its step grows by r. The memory is rescaled, not erased:
        relative per-element scales and first moments are kept."""
        for index, optimizer in enumerate(self.optimizers):
            for j, group in enumerate(optimizer.param_groups):
                r = self.surprise.last_ratios.get(f"{index}.{j}")
                if r is None or not r > 1.:
                    continue
                for p in group["params"]:
                    st = optimizer.state.get(p)
                    for key in ("exp_avg_sq", "max_exp_avg_sq"):
                        if st and key in st:
                            st[key].mul_(1. / (r * r))

    def _output_sigma(self, base, detach=True):
        mode = self.recipe.output_noise_mode
        if mode == "fixed":
            return base
        if mode == "mobility":
            return base * self.controller.mobility
        if not base:
            return 0.
        sigma = self.log_output_sigma.exp()
        settle = self.controller.mobility
        if self.lr_settle is not None:
            # The floor waits on model/table settlement, never on noise's own
            # rate. Clamped noise has zero gradient and a frozen tester; letting
            # that tester gate the floor would prevent it from ever releasing.
            # A group with no trainable parameters has no tester. Preserve the
            # historical floor fallback rather than assigning it an artificial
            # settled scale; fully trainable E22 groups use the stated floor.
            try:
                scales = [t.s for index, row in enumerate(self.lr_settle.testers) if index != 1
                          for t, role in zip(row, self.roles[index]) if role != "noise"]
                if scales and all(s is not None for s in scales):
                    settle = 1.0 if any(s > 1. / 64. for s in scales) else self.controller.mobility
            except AttributeError:
                pass
        floor = base * settle
        floor_tensor = torch.as_tensor(floor, device=sigma.device, dtype=sigma.dtype).detach()
        log_floor = torch.as_tensor(math.log(floor) if floor > 0 else -math.inf,
                                    device=sigma.device, dtype=sigma.dtype)
        # exp(log(base)) can round just below base on CUDA, suppressing the
        # learned-noise gradient at initialization. Repair only that numeric
        # inconsistency; ordinary maximum values and gradients stay intact.
        ordinary_sigma = torch.maximum(sigma, floor_tensor)
        rounded_below = (sigma < floor_tensor) & (self.log_output_sigma >= log_floor)
        differentiable_sigma = torch.maximum(self.log_output_sigma, log_floor).exp()
        # Keep the physical clamp exact and maximum's half subgradient at the
        # log-space tie. The correction and floor carry no gradient.
        repaired_sigma = ordinary_sigma.detach() + (differentiable_sigma - differentiable_sigma.detach())
        sigma = torch.where(rounded_below, repaired_sigma, ordinary_sigma)
        return float(sigma.detach()) if detach else sigma

    def output_sigma(self):
        """Current output-noise standard deviation as a host float."""
        return float(self._output_sigma(output_noise_std(self.recipe, self.completed_steps)))

    def _clean_generate(self, model, latent):
        return model(latent) if self.generation is None else self.generation(model, latent)

    def _postprocess_generated(self, generated):
        """Manipulator output postprocessing; identity for plain priors."""
        if getattr(self.prior, "postprocesses_generated", False):
            return self.prior.postprocess_generated(generated)
        return generated

    def observe_sampled_rows(self, rows):
        """Record generator-side prior draws for row-local optimizer updates.

        Caller-owned loops that bypass ``generate`` can call this during their
        generator phase. Whole-table regularizer gradients grant no ownership
        of rows that were not actually sampled.
        """
        if self._phase != "generator":
            raise RuntimeError("sampled prior rows must be observed during generator generation")
        setter = getattr(self.table_optimizer, "set_sampled_rows", None)
        if setter is not None and self.table.requires_grad:
            setter(self.table, rows)

    def generate(self, latent, *, sigma=None, stream=None, averaged=False, model=None, rows=None):
        """Generate with DV12 perturbation and optional output noise.

        Default noise is the shared value from ``begin_step``; a sampling
        caller must pass its own stream and explicitly choose ``sigma=0`` for
        clean outputs.  No output-noise draw occurs when sigma is zero.
        """
        stream = self.noise_generator if stream is None else stream
        if sigma is None:
            sigma = self._noise.output_sigma if self._noise is not None else self.output_sigma()
        model = (self.ema_G if averaged else self.G) if model is None else model
        averaged = averaged or model is self.ema_G
        if rows is not None and not averaged and self._phase == "generator":
            self.observe_sampled_rows(rows)
        if self.controller is not None:
            if (self.birth_death is not None and self._feature_selection is not None
                    and self._feature_selection.state["actual_backend"] == "feature_cells"):
                latent = self.birth_death.perturb_latent(
                    latent, stream, self.controller, prior=self._prior_view(averaged), rows=rows,
                    record=stream is self.noise_generator)
            else:
                latent = self.controller.perturb_latent(
                    latent, stream, self._prior_view(averaged), record=stream is self.noise_generator)
        y = self._clean_generate(model, latent)
        if self._feature_selection is not None:
            self._feature_selection.check_generated_shape(y)
        if sigma == 0:
            return y
        return y + sigma * torch.randn(y.shape, generator=stream, device=y.device, dtype=y.dtype)

    def routed_generate(self, context, *, sigma=None, perturb=True, stream=None, averaged=False):
        """Generate a conditional dense-bank forward with E22 training noise.

        Routing keys and values use the current selected bank. DV12, when
        enabled, perturbs ``weights @ table`` after routing, preserving both
        key and value gradient paths. ``sigma=0, perturb=False`` is the clean
        forward used by guarded proposals and serving. For paired-error GANs,
        use ``sigma=0`` here and add one shared output-noise draw to the real
        and fake error coordinates outside this helper.
        """
        if self.routed_control is None:
            raise ValueError("routed_generate requires recipe.row_policy='routed_paired' and RoutedRows")
        if type(perturb) is not bool:
            raise ValueError("perturb must be a boolean")
        stream = self.noise_generator if stream is None else self._stream(stream, 0)
        if sigma is None:
            sigma = self._noise.output_sigma if self._noise is not None else self.output_sigma()
        candidate = self.routed_control.candidate(averaged=averaged)
        prior = self.controller.routed_prior(candidate.table, candidate.log_mass) \
            if perturb and self.controller is not None else None
        perturb_fn = (lambda codes: self.controller.perturb_latent(
            codes, stream, prior, record=stream is self.noise_generator)) \
            if perturb and self.controller is not None else None
        y = self.routed_control.generate(context, candidate=candidate, perturb_fn=perturb_fn)
        if sigma == 0:
            return y
        return y + sigma * torch.randn(y.shape, generator=stream, device=y.device, dtype=y.dtype)

    def _served_parameters(self):
        values = [*self.G.parameters()]
        if self.prior is not None:
            values.extend(self.prior.parameters())
        elif self._table_location is None:
            values.append(self.table)
        for module in (self.encoder, self.router):
            if module is not None:
                values.extend(module.parameters())
        return values

    def _served_averages(self):
        values = [*self.ema_G.parameters()]
        if self.ema_prior is not None:
            values.extend(self.ema_prior.parameters())
        elif self._table_location is None:
            values.append(self.averaged_table)
        for module in (self.ema_encoder, self.ema_router):
            if module is not None:
                values.extend(module.parameters())
        return values

    @torch.no_grad()
    def _serve_release(self):
        if self._fast is not None:
            for live, fast in zip(self._served_parameters(), self._fast):
                live.copy_(fast)
            self._fast = None

    def _serve_settled(self):
        if (self.birth_death is not None and self._feature_selection is not None
                and self._feature_selection.state["actual_backend"] == "feature_cells"):
            return self.birth_death.paired_average_eligible(self.completed_steps)
        tester = self._table_tester()
        return tester is not None and getattr(tester, "last_decisive", 0) == -1

    @torch.no_grad()
    def _serve_apply(self):
        """GANTrainer compatibility swap; external loops use served_snapshot."""
        if self.recipe.serve_average <= 0:
            return
        self._serve_release()
        if self._serve_settled():
            live = self._served_parameters()
            self._fast = [p.detach().clone() for p in live]
            for parameter, average in zip(live, self._served_averages()):
                parameter.copy_(average)

    def _average_rate(self):
        if self.recipe.serve_average > 0:
            tester = self._table_tester()
            if tester is not None and tester.s is not None and tester.b:
                return min(1.0, float(tester.s) / (self.recipe.serve_average * float(tester.b)))
        return 1.0 - self.recipe.ema_decay

    def served_snapshot(self):
        """Return independent currently served state, without altering fast weights.

        ``models`` contains module state dictionaries; ``table`` is the same
        selected table independently copied.  ``source`` reports ``fast`` or
        ``averaged``.  The controller snapshot describes latent perturbation
        needed to reproduce served samples using an independent sampling RNG.
        Buffers follow the native serving path: current generator buffers and
        selected parameters (averaged buffers are copied at each update).
        Routed row buffers follow the selected table and its represented mass.
        """
        swapped = self._fast is not None
        self._serve_release()
        try:
            average = self.recipe.serve_average > 0 and self._serve_settled()
            models = {name: deepcopy(module.state_dict()) for name, module in self._training_modules().items()}
            if average:
                for name, module in self._average_modules().items():
                    for key, parameter in module.named_parameters(remove_duplicate=False):
                        models[name][key] = parameter.detach().clone()
                if self.routed_control is not None:
                    spec = self.routed_control.spec
                    buffers = dict(self.ema_router.named_buffers(remove_duplicate=False))
                    selected = {id(buffers[key]) for key in (spec.log_mass_key, *spec.row_buffers) if key in buffers}
                    for key, buffer in buffers.items():
                        if id(buffer) in selected:
                            models["router"][key] = buffer.detach().clone()
                    if self._table_location is not None:
                        owner, _ = self._table_location
                        for key, buffer in self._average_modules()[owner].named_buffers(remove_duplicate=False):
                            if buffer is self.averaged_table:
                                models[owner][key] = buffer.detach().clone()
            return deepcopy({"source": "averaged" if average else "fast",
                             "models": models,
                             "table": self.averaged_table if average else self.table,
                             "output_sigma": self.output_sigma(), "completed_steps": self.completed_steps,
                             "controller": None if self.controller is None else self.controller.state_dict(),
                             "row_semantics": self.row_semantics,
                             "routing": None if self.routed_control is None else self.routed_control.spec.to_dict(),
                             **({} if self._feature_selection is None else {
                                 "backend_selection": self._feature_selection.state_dict(),
                                 "feature_sampling": self._feature_selection.sampler_snapshot()})})
        finally:
            if swapped:
                self._serve_apply()

    def served_model(self, *, generation_factory=None):
        """Freeze independent currently served modules with exact sampling helpers.

        Custom generation callbacks must operate on the supplied model rather
        than close over live training parameters.  Auxiliary encoder/router
        module snapshots are available on the result for caller-owned routing.
        For a callback needing separate auxiliary modules, supply
        ``generation_factory(models)`` returning a callback bound to those
        frozen modules; this avoids closing over the caller's fast parameters.
        """
        snapshot = self.served_snapshot()
        models = {name: deepcopy(module).eval().requires_grad_(False)
                  for name, module in self._training_modules().items()}
        for name, module in models.items():
            module.load_state_dict(snapshot["models"][name])
        generation = self.generation
        if generation_factory is not None:
            if not callable(generation_factory):
                raise TypeError("generation_factory must be callable")
            generation = generation_factory(models)
            if not callable(generation):
                raise TypeError("generation_factory must return a callable")
        table = (snapshot["table"].detach() if self._table_location is None else
                 self._table_in_modules(models))
        controller = deepcopy(self.controller)
        if controller is not None:
            controller.load_state_dict(snapshot["controller"])
        stream = torch.Generator(device=self.device)
        stream.set_state(self.eval_generator.get_state().cpu())
        feature_sampler = None
        if snapshot.get("feature_sampling") is not None:
            from .feature_policy import FrozenFeatureSampler
            feature_sampler = FrozenFeatureSampler(snapshot["feature_sampling"])
        return ServedModel(models, table, controller, source=snapshot["source"],
                           output_sigma=snapshot["output_sigma"],
                           completed_steps=snapshot["completed_steps"], stream=stream,
                           generation=generation, row_semantics=self.row_semantics,
                           routing=None if self.routed_control is None else self.routed_control.spec,
                           feature_sampler=feature_sampler, backend_selection=snapshot.get("backend_selection"))

    def state_dict(self):
        """Save all policy, model, optimizer, averaging and RNG state independently.

        Caller data cursors and stateful callback resources are separate.
        Global PyTorch RNG is included for dropout and other module draws.
        """
        if self._phase != "ready":
            raise RuntimeError("policy checkpoints require a completed update boundary")
        swapped = self._fast is not None
        self._serve_release()
        try:
            return deepcopy({
                "schema": 1, "recipe": self.recipe.to_dict(), "roles": self.roles,
                "row_semantics": self.row_semantics, "device": str(self.device), "dtype": str(self.dtype),
                "table_requires_grad": self.table.requires_grad,
                "table_location": self._table_location,
                "models": {name: module.state_dict() for name, module in self._training_modules().items()},
                "requires_grad": {name: {key: p.requires_grad for key, p in module.named_parameters()}
                                  for name, module in self._training_modules().items()},
                "averages": {name: module.state_dict() for name, module in self._average_modules().items()},
                "table": self.table, "averaged_table": self.averaged_table,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                "initial_lrs": self.initial_lrs, "completed_steps": self.completed_steps,
                **({} if self._feature_selection is None else {
                    "backend_selection": self._feature_selection.state_dict()}),
                "last_output_sigma": self.last_output_sigma,
                "output_noise": None if self.log_output_sigma is None else self.log_output_sigma.detach(),
                "controller": None if self.controller is None else self.controller.state_dict(),
                "lr_settle": None if self.lr_settle is None else self.lr_settle.state_dict(),
                "birth_death": (None if self.birth_death is None or self.routed_control is not None
                                else self.birth_death.state_dict()),
                "row_evidence": (None if self.row_evidence is None or self.routed_control is not None
                                 else self.row_evidence.state_dict()),
                "routing": None if self.routed_control is None else self.routed_control.state_dict(),
                **({} if self.surprise is None else {"surprise": self.surprise.state_dict()}),
                **({} if self.loss_surprise is None else {"loss_surprise": self.loss_surprise.state_dict()}),
                **({} if self.reopen_guard is None else {"reopen_guard": self.reopen_guard.state_dict()}),
                "streams": {name: getattr(self, name).get_state() for name in self._STREAMS},
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                "served_source": "averaged" if self.recipe.serve_average > 0 and self._serve_settled() else "fast",
            })
        finally:
            if swapped:
                self._serve_apply()

    @staticmethod
    def _check_tensors(saved, expected, label):
        def same_metadata(value, reference):
            if type(value) is not type(reference):
                return False
            if isinstance(reference, dict):
                return value.keys() == reference.keys() and all(same_metadata(value[k], v) for k, v in reference.items())
            if isinstance(reference, (tuple, list)):
                return len(value) == len(reference) and all(same_metadata(a, b) for a, b in zip(value, reference))
            if isinstance(reference, torch.Tensor):
                return value.dtype == reference.dtype and value.shape == reference.shape and torch.equal(value, reference)
            return value == reference

        if not isinstance(saved, dict) or saved.keys() != expected.keys():
            raise ValueError(f"incompatible policy {label}")
        for key, tensor in expected.items():
            value = saved[key]
            # Module state_dicts can carry immutable constructor metadata via
            # get_extra_state (MoG stores sigma_rel and standardize this way).
            # Validate it before restoring any model or optimizer state.
            if not isinstance(tensor, torch.Tensor):
                if not key.endswith("_extra_state") or not same_metadata(value, tensor):
                    raise ValueError(f"incompatible policy {label} extra state {key}")
                continue
            if not isinstance(value, torch.Tensor) or value.shape != tensor.shape or value.dtype != tensor.dtype:
                raise ValueError(f"incompatible policy {label} tensor {key}")

    def _check_state(self, state):
        if not self.interrupted and self._feature_selection is None and self.reopen_guard is None:
            expected = self.state_dict()
        else:
            # Inspect shapes/topology without touching compatibility swaps.
            validator = copy(self)
            validator._phase = "ready"
            validator._fast = None
            expected = validator.state_dict()
        allowed = [set(expected)]
        if self.routed_control is None:
            allowed.append(set(expected) - {"routing"})
        if self.loss_surprise is not None:
            allowed.extend([s - {"loss_surprise"} for s in list(allowed)])
        if not isinstance(state, dict) or set(state) not in allowed or state.get("schema") != 1:
            raise ValueError("invalid UpdatePolicy checkpoint schema")
        saved_recipe = state.get("recipe")
        if (not isinstance(saved_recipe, dict)
                or {"row_policy": "independent", **saved_recipe} != expected["recipe"]):
            raise ValueError("checkpoint recipe does not match policy")
        for key in ("roles", "row_semantics", "device", "dtype", "requires_grad",
                    "table_requires_grad", "table_location"):
            if state[key] != expected[key]:
                raise ValueError(f"checkpoint {key} does not match policy")
        steps = state["completed_steps"]
        # total_steps is the recipe's schedule horizon. Caller-owned loops may
        # continue past it with begin_step(execution_limit=...), so it is not a
        # checkpoint budget. The host validates its own external execution cap.
        if type(steps) is not int or steps < 0:
            raise ValueError("invalid policy checkpoint step count")
        for label in ("models", "averages"):
            if not isinstance(state[label], dict) or state[label].keys() != expected[label].keys():
                raise ValueError(f"incompatible policy {label}")
            for name in expected[label]:
                self._check_tensors(state[label][name], expected[label][name], f"{label}/{name}")
        for name in ("table", "averaged_table", "output_noise"):
            value, current = state[name], expected[name]
            if current is None:
                if value is not None:
                    raise ValueError(f"incompatible policy {name}")
            elif (not isinstance(value, torch.Tensor) or value.shape != current.shape
                  or value.dtype != current.dtype or (name == "output_noise" and not torch.isfinite(value).all())):
                raise ValueError(f"incompatible policy {name}")
        if self._table_location is not None:
            owner, key = self._table_location
            for family, name in (("models", "table"), ("averages", "averaged_table")):
                table_value = state[name].detach()
                owner_value = state[family][owner][key].detach().to(table_value.device)
                if (not torch.equal(owner_value, table_value)
                        and not torch.allclose(owner_value, table_value, rtol=0, atol=0, equal_nan=True)):
                    raise ValueError(f"inconsistent policy {name} alias in {family}/{owner}/{key}")
        rates = state["initial_lrs"]
        if (not isinstance(rates, list) or len(rates) != len(self.initial_lrs)
                or any(not isinstance(a, list) or len(a) != len(b) for a, b in zip(rates, self.initial_lrs))
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for row in rates for value in row)):
            raise ValueError("invalid policy checkpoint initial learning rates")
        sigma = state["last_output_sigma"]
        if sigma is not None and (type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0):
            raise ValueError("invalid policy checkpoint output sigma")
        if not isinstance(state["optimizers"], list) or len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("incompatible policy optimizer count")
        try:
            for optimizer, values in zip(self.optimizers, state["optimizers"]):
                _validate_optimizer_state(optimizer, values)
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid policy checkpoint optimizer state") from error
        if not isinstance(state["streams"], dict) or state["streams"].keys() != expected["streams"].keys():
            raise ValueError("incompatible policy RNG streams")
        try:
            for value in state["streams"].values():
                torch.Generator(device=self.device).set_state(value.cpu())
            torch.Generator(device="cpu").set_state(state["cpu_rng"].cpu())
            if self.device.type == "cuda":
                torch.Generator(device=self.device).set_state(state["cuda_rng"].cpu())
            elif state["cuda_rng"] is not None:
                raise ValueError("CPU policy cannot load CUDA RNG state")
        except (TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid policy checkpoint RNG state") from error
        prepared = (None if self._feature_selection is None else
                    self._feature_selection.prepare_restore(state["backend_selection"], state))
        validation_birth, validation_settle = ((self.birth_death, self.lr_settle) if prepared is None
                                             else prepared["controls"])
        for key, control in (("controller", self.controller), ("lr_settle", validation_settle),
                             ("birth_death", None if self.routed_control is not None else validation_birth),
                             ("row_evidence", None if self.routed_control is not None else self.row_evidence)):
            if (control is None) != (state[key] is None):
                raise ValueError(f"incompatible policy {key}")
            if control is None:
                continue
            if key == "lr_settle":
                deepcopy(control).load_state_dict(_state_to_device(state[key], self.device), self.optimizers)
            elif key in ("birth_death", "row_evidence"):
                control.check_state(state[key])
            else:
                deepcopy(control).load_state_dict(_state_to_device(state[key], self.device))
        routing = state.get("routing")
        if (self.routed_control is None) != (routing is None):
            raise ValueError("checkpoint routing contract does not match policy")
        if self.routed_control is not None:
            self.routed_control.check_state(routing)
        if self.surprise is not None:
            deepcopy(self.surprise).load_state_dict(state["surprise"])
        if self.loss_surprise is not None and "loss_surprise" in state:
            deepcopy(self.loss_surprise).load_state_dict(state["loss_surprise"])
        if self.reopen_guard is not None:
            self.reopen_guard.check_state(state["reopen_guard"], roles=self.roles,
                                          completed_steps=state["completed_steps"],
                                          anchor_started=self._loss_epoch(state["optimizers"][1]))
        table_state = state["lr_settle"]
        settled = False
        if table_state is not None:
            settled = any(value is not None and role == "table" and value.get("last_decisive") == -1
                          for row, roles in zip(table_state, self.roles) for value, role in zip(row, roles))
        source = "averaged" if self.recipe.serve_average > 0 and settled else "fast"
        if self._feature_selection is not None:
            feature_source = self._feature_selection.saved_served_source(state)
            if feature_source is not None:
                source = feature_source
        if state["served_source"] != source:
            raise ValueError("inconsistent policy served source")
        return prepared

    def load_state_dict(self, state):
        """Validate then restore, keeping caller modules at fast training weights."""
        if self._phase not in ("ready", "interrupted"):
            raise RuntimeError("restore requires a completed update boundary")
        prepared = self._check_state(state)
        self._serve_release()
        if prepared is not None:
            self._feature_selection.commit_restore(prepared)
        for name, module in self._training_modules().items():
            module.load_state_dict(state["models"][name])
        for name, module in self._average_modules().items():
            module.load_state_dict(state["averages"][name])
        with torch.no_grad():
            self.table.copy_(state["table"])
            self.averaged_table.copy_(state["averaged_table"])
            if self.log_output_sigma is not None:
                self.log_output_sigma.copy_(state["output_noise"])
        for optimizer, values in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(deepcopy(values))
        if self.controller is not None:
            self.controller.load_state_dict(_state_to_device(state["controller"], self.device))
        if self.lr_settle is not None:
            self.lr_settle.load_state_dict(_state_to_device(state["lr_settle"], self.device), self.optimizers)
        if self.birth_death is not None and self.routed_control is None:
            self.birth_death.load_state_dict(state["birth_death"])
        if self.row_evidence is not None and self.routed_control is None:
            self.row_evidence.load_state_dict(state["row_evidence"])
        if self.routed_control is not None:
            self.routed_control.load_state_dict(state["routing"])
        if self.surprise is not None:
            self.surprise.load_state_dict(state["surprise"])
        if self.loss_surprise is not None and "loss_surprise" in state:
            self.loss_surprise.load_state_dict(state["loss_surprise"])
        if self.reopen_guard is not None:
            self.reopen_guard.load_state_dict(state["reopen_guard"], roles=self.roles,
                                              completed_steps=state["completed_steps"],
                                              anchor_started=self._loss_epoch(state["optimizers"][1]))
        self.initial_lrs, self.completed_steps = deepcopy(state["initial_lrs"]), state["completed_steps"]
        self.last_output_sigma = state["last_output_sigma"]
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        self._phase = "ready"


class E22Policy(UpdatePolicy):
    """The E22 DV12/stationarity policy for an explicit caller-owned loop.

    Task-specific recipe overrides remain authoritative.  This class requires
    DV12 and stationarity control; disabling birth/death or output noise for
    an ablation keeps the same lifecycle and checkpoint API. Independent
    E22 retains its original particle law. The named ``e22_routed`` recipe
    supplies the conditional paired-error adaptation with ``RoutedRows``
    and fit/guard ``RoutedBatch`` observations; clean served inference uses
    ``served_model().routed_forward(context)``.
    """

    def __init__(self, recipe, generator, critic, **kwargs):
        if not isinstance(recipe, Recipe) or recipe.continuous_policy != "dv12" or recipe.lr_control != "stationarity":
            raise ValueError("E22Policy requires a DV12 recipe with lr_control='stationarity'")
        super().__init__(recipe, generator, critic, **kwargs)
