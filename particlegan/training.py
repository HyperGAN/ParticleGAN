"""A checkpointable training loop using the current recipe formulation."""
from copy import deepcopy
import math

from particlegan.optim.constraint_geometry import constraint_geometry_backward

import torch
from torch import nn

from .particle_prior import MoGParticlePrior, ParticlePrior
from .priors import unwrap_prior
from .recipes import Recipe, learning_rate_scales
from .policy import UpdatePolicy, _state_to_device, _validate_optimizer_state, input_noise_std as input_noise_std, output_noise_std


def _convolution_autocast(module, enabled):
    if not enabled or not any(isinstance(child, (nn.Conv2d, nn.ConvTranspose2d))
                              and max(child.in_channels, child.out_channels) // child.groups > 32
                              for child in module.modules()):
        return False
    parameter = next(module.parameters(), None)
    if parameter is None or parameter.device.type != "cuda":
        return False
    with torch.cuda.device(parameter.device):
        return torch.cuda.is_bf16_supported(including_emulation=False)


def _float_output(value):
    if isinstance(value, torch.Tensor):
        result = value.float() if value.dtype == torch.bfloat16 else value
        if hasattr(value, "_bispf_input"):
            result._bispf_input = value._bispf_input
        if hasattr(value, "_empf_input"):
            result._empf_input = value._empf_input
        if hasattr(value, "_empf_record"):
            result._empf_record = value._empf_record
        if hasattr(value, "_empf_conditioned"):
            result._empf_conditioned = value._empf_conditioned
        if hasattr(value, "_rhbi_input"):
            result._rhbi_input = value._rhbi_input
            result._rhbi_cond = getattr(value, "_rhbi_cond", None)
            result._rhbi_cont_cond = bool(getattr(value, "_rhbi_cont_cond", False))
            result._rhbi_parts = getattr(value, "_rhbi_parts", None)
        if hasattr(value, "_rhbi_record"):
            result._rhbi_record = value._rhbi_record
        return result
    if isinstance(value, tuple):
        return tuple(_float_output(item) for item in value)
    if isinstance(value, list):
        return [_float_output(item) for item in value]
    return value


class InputNoise(nn.Module):
    """``critic(x + std * eps, ...)`` with fresh ``eps`` per call from ``generator``.

    Plain ``critic(x, ...)`` (no draw) while ``std == 0``; set ``std`` as the
    schedule moves. Pass the wrapper to the recipe's critic penalty like the
    critic itself: its EMA critic is evaluated through the same wrapper (same
    ``std`` and noise stream).
    """

    def __init__(self, critic, std=0.0, generator=None, *, bf16=False):
        super().__init__()
        self.critic, self.std, self.generator = critic, float(std), generator
        self.bf16 = _convolution_autocast(critic, bf16)

    def _forward(self, x, *args, **kwargs):
        if not self.bf16 or x.device.type != "cuda":
            return self.critic(x, *args, **kwargs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _float_output(self.critic(x.contiguous(memory_format=torch.channels_last), *args, **kwargs))

    def forward(self, x, *args, **kwargs):
        if self.std == 0:
            return self._forward(x, *args, **kwargs)
        noise = torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        out = self._forward(x + self.std * noise, *args, **kwargs)
        target = out[0] if isinstance(out, (tuple, list)) and out else out
        if isinstance(target, torch.Tensor):
            target._bispf_input = x
            target._empf_input = x
            if hasattr(target, "_rhbi_input") or hasattr(self.critic, "_rhbi_record"):
                from .rhbi import _extract_critic_inputs_parts, _record_rhbi_capture
                rec = getattr(target, "_rhbi_record", getattr(self.critic, "_rhbi_record", None))
                if rec is None or not getattr(rec, "_in_penalty", False):
                    x_clean, cond_disc, has_cont_cond, float_parts = _extract_critic_inputs_parts(
                        (x, *args), kwargs, int(target.shape[0])
                    )
                    _record_rhbi_capture(
                        out,
                        x_clean,
                        cond_disc,
                        rec,
                        has_cont_cond=has_cont_cond,
                        float_parts=float_parts,
                        overwrite=True,
                    )
        return out


# Fields added to Recipe after checkpoints were written, with the value that
# reproduces the older behaviour; a saved recipe without them is upgraded.
def _normalized_recipe(recipe):
    """A recipe dict in the form checkpoints are compared in: added fields filled with the values that
    reproduce older behaviour, removed fixed choices dropped when they hold their only value, and the
    construction-time ``initialization`` (superseded by saved weights) dropped."""
    from .recipe_compat import without_default_additions
    recipe = without_default_additions({**_ADDED_RECIPE_FIELDS, **recipe})
    for name in ("d_betas", "loss_labels"):
        if name in recipe and isinstance(recipe[name], (list, tuple)):
            recipe[name] = tuple(recipe[name])
    if isinstance(recipe.get("bispf_gate_overrides"), (list, tuple)):
        # JSON turns the stored ((gate, value), ...) pairs into nested lists.
        recipe["bispf_gate_overrides"] = tuple(
            tuple(pair) if isinstance(pair, list) else pair for pair in recipe["bispf_gate_overrides"])
    if all(recipe.get(key, value) == value for key, value in _REMOVED_RECIPE_FIELDS.items()):
        recipe = {key: value for key, value in recipe.items() if key not in _REMOVED_RECIPE_FIELDS}
    if recipe.get("initialization", None) in (None, "batch_feature_zero"):
        recipe.pop("initialization", None)
    return recipe


# Recipe fields that once named a fixed choice, with the only value they could hold; a saved
# recipe that records them with that value loads, any other value is rejected.
_REMOVED_RECIPE_FIELDS = {"loss_type": "logistic", "gan_mode": "rp",
                          "reg_method": "autograd"}
_ADDED_RECIPE_FIELDS = {"reg_anchor_weight": 1.0, "direct_particle_gain": True,
                        "reg_arm": None, "critic_formulation": "ka2",
                        "continuous_policy": None, "amsgrad": False, "critic_r1_real": True,
                        "critic_payoff_damping": True, "output_noise_mode": "fixed",
                        "lr_control": "mobility", "particle_birth_death": False,
                        "row_evidence_gate": False, "table_release_rule": "any",
                        "row_evidence_hot": True, "row_evidence_exclude": True, "row_evidence_hold": True,
                        "birth_death_space": "data", "serve_average": 0.0, "reopen_signal": "data",
                        "row_evidence_null": "theory", "birth_death_isolation": False,
                        "birth_death_feature_scale": "none", "row_policy": "independent",
                        "optimizer_family": "formulation", "eps": 1e-8,
                        "optimizer_momentum": 0.0, "optimizer_adam_lr": None,
                        "beta2_end": None, "beta2_anneal_end": 0.2,
                        "reg_coeff_end": None, "reg_coeff_anneal_end": 0.2,
                        "generator_gradient_response": "none",
                        "rhbi_weight": 0.0, "rhbi_rho": 2.0, "rhbi_tau": 2.0,
                        "rhbi_eps": 0.1, "rhbi_iters": 8, "rhbi_c_max": 20.0,
                        "rhbi_beta_ell": 0.1, "rhbi_nn_quantile": 0.25,
                        "rhbi_critic_weight": 0.0, "rhbi_self_coupling": "hollow",
                        "rhbi_relativistic_weight": 0.0, "rhbi_lambda_floor": 0.0,
                        "rhbi_gamma_gauge": 0.0, "rhbi_rapidity_screening": True,
                        "rhbi_scale_log_odds": False, "rhbi_conditional_fibre": "joint",
                        "rhbi_akt_horizon": False, "rhbi_akt_mode": "sqrt",
                        "rhbi_slack_gain": 0.0, "rhbi_chord_weight": 0.0,
                        "rhbi_thermal_tau": 0.0, "rhbi_thermal_shock_mu": 0.15,
                        "rhbi_settle_beta": 0.0, "rhbi_settle_mode": "residual",
                        "rhbi_fibre_rel_blend": 0.0, "rhbi_step_damp": 0.0,
                        "rhbi_cond_ot_weight": 0.0, "rhbi_cond_ot_mode": "kernel_ed",
                        "rhbi_simplex_weight": 0.0, "rhbi_cvx_mu_gain": 0.40,
                        "rhbi_simplex_chart_weight": 0.0, "rhbi_tail_spill_gain": 0.0,
                        "rhbi_simplex_macro_settle": False}


class GANTrainer:
    """Own the recipe's update mechanics; callers supply networks and real batches.

    Supports scalar, unconditional GAN recipes with a particle or MoG prior. Fresh real
    batches for the generator can be passed as ``generator_real`` tensors or
    zero-argument callables. Data-loader position is caller-owned and must be
    saved separately when checkpointing. Checkpoints restore global PyTorch RNG
    state as well as this trainer's sampling streams for exact continuation on
    the same device. Sampling never advances training RNG streams.

    Per update: role-wise LR schedule (``learning_rate_scales``), critic step
    with input noise and the recipe's penalty (``recipe.make_critic_penalty``;
    the critic optimizer's ``step()`` runs the spike guard and anchor EMA),
    then a generator/prior step with output noise (the generator optimizer's
    ``step()`` applies A2 latent damping). The trainer allocates the EMA
    critic ``ema_D`` (a frozen deep copy). Noise comes from a trainer stream.
    Networks train from the weights they arrive with; initialize them first
    (e.g. ``particlegan.init.deterministic_orthogonal_``). ``sample`` is clean
    by default; ``sample(..., output_noise=True)`` adds the current output
    noise (``output_sigma()``; see ``recipe.output_noise_mode``).

    ``max_steps`` bounds execution independently of the recipe's schedule;
    ``extend_execution`` raises that allowance without changing the formulation.
    Optional named RNG streams isolate input/output noise, MoG kernel noise,
    evaluation and stochastic model layers and travel with checkpoints.
    A ``ManipulatedPrior`` is classified by its storage prior
    (``particlegan.priors.unwrap_prior``): a wrapped MoG gets the same MoG
    noise stream as a plain one. Manipulator output postprocessing applies to
    critic fakes, generator fakes, strict-progress replays and ``sample``;
    manipulator penalties are reported as ``prior_manipulation`` every step;
    ``strict_progress`` rejects non-additive manipulator transforms.

    The whole update always disables autograd multithreading, including graph
    construction and higher-order critic penalties. ``serial_backward=True``
    remains a compatibility assertion; False is rejected. The checkpoint
    records this constraint, and unmarked or False historical checkpoints
    require their original source. The caller's autograd mode is restored.

    ``row_evidence_statistic`` is forwarded to ``UpdatePolicy`` (legacy
    ``"sphericity"`` by default; ``"hotelling"`` is a declared candidate).
    KA2's ``trust_mode`` candidate is selected with
    ``optimizer_options={"ka2_trust_mode": "scale"}``, which the checkpoint
    records with the other optimizer options.

    ``serve_after_step`` (default True) is the public extension point for the
    end-of-update serving swap. When a recipe serves an average
    (``serve_average > 0``) and that average has settled, each ``step`` ends by
    copying the averaged weights into the modules (the training iterate is
    restored at the start of the next update). ``serve_after_step=False``
    leaves the live training iterate in the modules after every update; the
    served state stays available from ``served_snapshot``/``served_model``.
    Only that swap changes: losses, optimizer updates and RNG consumption are
    identical, and checkpoints always store the training iterate (a resumed
    ``serve_after_step=False`` trainer also leaves the training iterate in the
    modules after ``load_state_dict``). Recipes
    with ``serve_average == 0`` behave identically either way. The option is
    execution-only and is not recorded in checkpoints; pass the same value
    when resuming.
    """

    def __init__(self, recipe, generator, discriminator, *, prior=None, seed=0,
                 latent_generator=None, penalty_generator=None,
                 noise_generator=None, input_noise_generator=None,
                 prior_noise_generator=None, eval_generator=None, model_generator=None,
                 require_latent_damping=None, max_steps=None,
                 optimizer_options=None, penalty_options=None, serial_backward=True,
                 row_evidence_statistic="sphericity", serve_after_step=True):
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
        if recipe.conditional_energy_weight:
            raise ValueError("conditional energy score requires a caller-owned paired conditional host")
        if recipe.masked_mixture_weight or recipe.masked_transport_weight or recipe.imputer_component_weight:
            raise ValueError("masked mixture/imputer objectives require a caller-owned incomplete-data host")
        if type(serial_backward) is not bool:
            raise TypeError("serial_backward must be a boolean")
        if type(serve_after_step) is not bool:
            raise TypeError("serve_after_step must be a boolean")
        self.serve_after_step = serve_after_step
        if not serial_backward:
            raise ValueError("ParticleGAN requires serial_backward=True; autograd multithreading is disabled")
        if getattr(recipe, "row_policy", "independent") != "independent":
            raise ValueError("GANTrainer requires row_policy='independent'; use E22Policy with RoutedRows "
                             "and a caller-owned loop for paired conditional contexts")
        if (recipe.model != "gan" or recipe.conditioning != "scalar"
                or recipe.encoder_mode != "none"):
            raise ValueError("GANTrainer supports unconditional scalar GANs with particle priors and no encoder")
        self.recipe, self.G, self.D = recipe, generator, discriminator
        for module in (self.G, self.D):
            if (_convolution_autocast(module, recipe.training_precision == "bf16_channels_last")
                    and next(module.parameters()).device.type == "cuda"):
                module.to(memory_format=torch.channels_last)
        if recipe.constraint_geometry_mode == "strict_progress" and (recipe.input_noise_std or recipe.output_noise_std or recipe.standardize):
            raise ValueError("strict_progress requires zero additive noise and unstandardized prior for deterministic replay")
        if recipe.constraint_geometry_mode == "strict_progress" and any(
                isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
                for module in self.G.modules()):
            raise ValueError("strict_progress requires generator forwards without stochastic layers or running buffers")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        self.max_steps = recipe.total_steps if max_steps is None else max_steps
        parameters = list(generator.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("generator must have trainable parameters")
        first_trainable = next(p for p in parameters if p.requires_grad)
        self.device, self.dtype = first_trainable.device, first_trainable.dtype
        self.prior = (recipe.make_prior().to(device=self.device, dtype=self.dtype)
                      if prior is None else prior)
        base_prior = unwrap_prior(self.prior)
        if (recipe.constraint_geometry_mode == "strict_progress"
                and not getattr(self.prior, "additive_transforms", True)):
            raise ValueError("strict_progress replay assumes additive prior transforms "
                             "(latent + (z - loc)); a prior manipulator transforms codes non-additively")
        expected_prior = MoGParticlePrior if recipe.prior_kind == "mog" else ParticlePrior
        if type(base_prior) is not expected_prior:
            raise ValueError("prior must match the recipe's ParticlePrior or MoGParticlePrior kind")
        if type(base_prior) is MoGParticlePrior and base_prior.standardize != recipe.standardize:
            raise ValueError("prior standardize must match the recipe")
        is_mog = type(base_prior) is MoGParticlePrior
        has_normal_noise = bool(getattr(self.prior, "normal_noise", False)) or any(
            getattr(m, "normal_noise", False) or type(m).__name__ == "NormalNoiseManipulator"
            for m in getattr(self.prior, "manipulators", ()))
        has_prior_noise = is_mog or has_normal_noise
        if prior_noise_generator is not None and not has_prior_noise:
            raise ValueError("prior_noise_generator requires a MoG prior or normal noise manipulator")
        if self.prior.z.shape != (recipe.num_particles, recipe.z_dim):
            raise ValueError("prior dimensions must match the recipe")
        if not any(p.requires_grad for p in discriminator.parameters()):
            raise ValueError("discriminator must have trainable parameters")
        seen = set()
        for module in (self.G, self.D, self.prior):
            for value in (*module.parameters(), *module.buffers()):
                if value.device != self.device or (value.requires_grad and value.is_floating_point()
                                                  and value.dtype != self.dtype):
                    raise ValueError("generator, discriminator and prior must share one device and trainable floating dtype")
            for parameter in module.parameters():
                if id(parameter) in seen:
                    raise ValueError("generator, discriminator and prior must not share parameters")
                seen.add(id(parameter))
        recipe._validate_frozen_prior_ownership(self.prior, self.G, self.D)
        recipe.apply_prior_policy(self.prior)
        self.optimizer_options = dict(optimizer_options or {})
        self.penalty_options = dict(penalty_options or {})
        if require_latent_damping is None:
            require_latent_damping = base_prior.z.requires_grad and recipe.latent_damping_max_rate > 0
        if type(require_latent_damping) is not bool:
            raise TypeError("require_latent_damping must be a boolean or None")
        # The recipe picks the regularization formulation; its
        # step-time work runs inside these optimizers' step().
        ema_critic = None if recipe.effective_critic_formulation in ("bispf", "empf", "rhbi") else deepcopy(self.D)
        self.opt_g, self.opt_d = recipe.make_optimizers(
            self.G, self.D, self.prior, ema_critic=ema_critic,
            require_latent_damping=require_latent_damping, **self.optimizer_options)
        self.prior_mechanisms = self.opt_g.prior_mechanisms
        self._gradient_response = None
        if getattr(recipe, "generator_gradient_response", "none") != "none":
            from .gradient_response import GradientCoherence
            self._gradient_response = GradientCoherence()
        self.loss = recipe.make_loss()
        self.prior_regularizer = recipe.make_prior_regularizer(
            weight=1.0 if recipe.prior_regularizer == "vicreg" or recipe.prior_reg > 0 else 0.,
            prior=self.prior, reference=self.prior)
        self.penalty = recipe.make_critic_penalty(self.opt_d, **self.penalty_options)
        if recipe.effective_critic_formulation == "bispf":
            # Readers of the critic thermal scale use this trainer's checkpointed
            # critic record, never another trainer's state.
            from .bispf import bind_thermal_source
            bind_thermal_source(self.opt_d.record, self.loss, self.prior_regularizer)
        elif recipe.effective_critic_formulation == "empf":
            from .empf import bind_empf_record
            bind_empf_record(self.opt_d.record, self.loss, self.prior_regularizer)
        elif recipe.effective_critic_formulation == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0:
            from .rhbi import bind_rhbi_record
            bind_rhbi_record(self.opt_d.record, self.loss, self.prior_regularizer)
        if eval_generator is not None and any(eval_generator is stream for stream in (
                latent_generator, penalty_generator, noise_generator, input_noise_generator,
                prior_noise_generator, model_generator)):
            raise ValueError("evaluation stream must be separate from training")
        self.policy = UpdatePolicy(
            recipe, self.G, self.D, prior=self.prior,
            generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
            seed=seed, streams={"latent_generator": latent_generator,
                                "penalty_generator": penalty_generator,
                                "noise_generator": noise_generator,
                                "eval_generator": eval_generator},
            penalty=self.penalty,
            schedule=lambda step, config: learning_rate_scales(step, config),
            allow_shared_training_streams=not (recipe.continuous_policy is not None
                or recipe.lr_control == "stationarity" or recipe.row_evidence_gate
                or recipe.particle_birth_death or recipe.serve_average > 0),
            row_evidence_statistic=row_evidence_statistic)
        self._STREAMS = type(self)._STREAMS
        if input_noise_generator is not None:
            self.input_noise_generator = self._stream(input_noise_generator, seed + 7)
            self._STREAMS += ("input_noise_generator",)
        else:
            self.input_noise_generator = self.noise_generator
        self.prior_noise_generator = None
        if has_prior_noise:
            self.prior_noise_generator = self._stream(prior_noise_generator, seed + 6)
            self._STREAMS += ("prior_noise_generator",)
        self.model_generator = None
        if model_generator is not None:
            self.model_generator = self._stream(model_generator, seed + 8)
            self._STREAMS += ("model_generator",)
        training_streams = [getattr(self, name) for name in self._STREAMS if name != "eval_generator"]
        if self.eval_generator in training_streams:
            raise ValueError("evaluation stream must be separate from training")
        global_stream = (torch.default_generator if self.device.type == "cpu"
                         else torch.cuda.default_generators[self.device.index])
        if self.model_generator is not None and (self.model_generator is global_stream or any(
                getattr(self, name) in (self.model_generator, global_stream)
                for name in self._STREAMS if name not in ("eval_generator", "model_generator"))):
            raise ValueError("model stream must be separate from other training streams and global draws")
        self._noisy_D = InputNoise(self.D, 0.0, self.input_noise_generator,
                                   bf16=recipe.training_precision == "bf16_channels_last")
        self._bf16_generator = _convolution_autocast(self.G, recipe.training_precision == "bf16_channels_last")
        self._in_training_step = False
        # Manipulator hooks resolved once; plain priors have neither.
        self._postprocesses = bool(getattr(self.prior, "postprocesses_generated", False))
        self._prior_penalty = bool(getattr(self.prior, "has_regularizer", False))

    @property
    def latent_damping(self):
        return self.opt_g.latent_damping

    @property
    def latent_history(self):
        return self.opt_g.latent_history

    @property
    def ema_D(self):
        """The trainer-owned EMA critic used by the anchor penalty."""
        return self.opt_d.ema_critic

    def _stream(self, generator, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed) if generator is None else generator
        device = generator.device
        # CUDA generators may expose the unindexed current-device spelling,
        # while parameter tensors always expose an explicit index.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self.device:
            raise ValueError("random generators must use the model device")
        return generator

    def _batch(self, batch, name):
        if (not isinstance(batch, torch.Tensor) or batch.ndim < 2 or not len(batch)
                or batch.device != self.device or batch.dtype != self.dtype):
            raise ValueError(f"{name} must be a nonempty batch on the model device and dtype")
        return batch.detach()

    def _output_sigma(self, base, detach=True):
        return self.policy._output_sigma(base, detach=detach)

    def output_sigma(self):
        """Current output-noise standard deviation as a float."""
        return self.policy.output_sigma()

    def _generate(self, model, latent, sigma, stream, indices=None, *, rows=None):
        if indices is not None and rows is not None:
            raise ValueError("pass sampled row IDs as indices or rows, not both")
        if self._in_training_step and self._bf16_generator and latent.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _float_output(self.policy.generate(latent, sigma=sigma, stream=stream, model=model,
                                    rows=indices if rows is None else rows))
        return self.policy.generate(latent, sigma=sigma, stream=stream, model=model,
                                    rows=indices if rows is None else rows)

    def _postprocess(self, prior, generated):
        """Apply manipulator output postprocessing (every critic/serving path)."""
        return prior.postprocess_generated(generated) if self._postprocesses else generated

    def served_snapshot(self):
        """Independent currently served generator/table state, also available in caller-owned loops."""
        return self.policy.served_snapshot()

    def served_model(self, *, generation_factory=None):
        """Independent frozen modules that reproduce currently served samples."""
        return self.policy.served_model(generation_factory=generation_factory)

    def step(self, real, *, generator_real=None, collect_stats=False):
        """Perform one D update and one G/prior update; return detached losses.

        ``step`` in the result is the completed update count. The generator
        loss pairs fakes with ``generator_real`` (a tensor or a callable;
        default: ``real``). ``collect_stats`` additionally
        returns the gradient penalty's synchronized diagnostic dictionary
        (including the penalty's blend weight ``s``).
        """
        if self.model_generator is None:
            return self._execute_step(real, generator_real=generator_real, collect_stats=collect_stats)
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.model_generator.get_state(), self.device)
            else:
                torch.set_rng_state(self.model_generator.get_state())
            try:
                return self._execute_step(real, generator_real=generator_real, collect_stats=collect_stats)
            finally:
                state = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else torch.get_rng_state()
                self.model_generator.set_state(state)

    @property
    def serial_backward(self):
        """The fixed project execution policy, retained for compatibility."""
        return True

    def _execute_step(self, real, *, generator_real=None, collect_stats=False):
        self._in_training_step = True
        try:
            # Scope the entire update: graph construction can affect backward
            # accumulation order too. Restore the caller's setting on exit.
            with torch.autograd.set_multithreading_enabled(False):
                return self._step(real, generator_real=generator_real, collect_stats=collect_stats)
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            self._in_training_step = False

    def _serve_release(self):
        return self.policy._serve_release()

    def _serve_settled(self):
        return self.policy._serve_settled()

    def _serve_apply(self):
        return self.policy._serve_apply()

    def _served_parameters(self):
        return self.policy._served_parameters()

    def _served_averages(self):
        return self.policy._served_averages()

    def _average_rate(self):
        return self.policy._average_rate()

    def _sample_training_prior(self, n):
        if self.prior_noise_generator is not None:
            return self.prior.sample(n, generator=self.latent_generator,
                                     noise_generator=self.prior_noise_generator)
        return self.prior.sample(n, generator=self.latent_generator)

    def extend_execution(self, max_total_steps):
        """Raise the external execution cap without changing recipe schedules."""
        if (type(max_total_steps) is not int or max_total_steps <= self.completed_steps
                or (self.max_steps is not None and max_total_steps <= self.max_steps)):
            raise ValueError("execution extension must exceed the current allowance and completed steps")
        self.max_steps = max_total_steps
        return self.max_steps

    def _step(self, real, *, generator_real=None, collect_stats=False):
        self._serve_release()
        recipe = self.recipe
        if self.max_steps is not None and self.completed_steps >= self.max_steps:
            raise RuntimeError("recipe training budget exhausted")
        real = self._batch(real, "real")
        if generator_real is not None and not callable(generator_real):
            generator_real = self._batch(generator_real, "generator_real")
            if generator_real.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if len(generator_real) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
        step_noise = self.policy.begin_step(real, game_record=self.penalty.regularizer.record,
                                            execution_limit=self.max_steps)
        sigma_in, sigma_out = step_noise.input_sigma, step_noise.output_sigma
        noise = self.noise_generator
        critic = self._noisy_D
        critic.std = sigma_in
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, indices_d = self._sample_training_prior(len(real))
            self.policy.observe_support(latent)
            fake = self._postprocess(self.prior, self._generate(self.G, latent, sigma_out, noise, rows=indices_d))
        self.policy.observe_critic_pair(real, fake)
        loss_d = self.loss.d_loss(critic(real), critic(fake))
        self.penalty.collect_stats = collect_stats
        penalty = self.penalty(critic, real, fake)
        penalty_stats = self.penalty.last_stats
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        self.policy.after_critic_step()

        self.D.eval()
        self.G.train()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, indices = self._sample_training_prior(len(real))
            fake_g = self._postprocess(self.prior, self._generate(self.G, latent, sigma_out, noise, rows=indices))
            if hasattr(self.opt_g, "bind_origin_sample"):
                self.opt_g.bind_origin_sample(fake_g)
            if hasattr(self.loss, "rhbi") and hasattr(self.loss.rhbi, "bind_discrete_prior_step"):
                simplicial_active = (
                    recipe.rhbi_simplex_weight > 0.0
                    and recipe.rhbi_simplex_chart_weight > 0.0
                    and recipe.rhbi_simplex_macro_settle
                    and self.prior_noise_generator is None
                    and self.prior.z.requires_grad
                    and len(self.prior.z) >= 2
                )
                self.loss.rhbi.bind_discrete_prior_step(
                    self.prior_noise_generator is None,
                    simplicial_active=simplicial_active,
                )
            fake_logits = critic(fake_g)
            real_g = generator_real() if callable(generator_real) else generator_real
            real_g = real if real_g is None else self._batch(real_g, "generator_real")
            if real_g.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if len(real_g) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
            real_logits = critic(real_g)
            loss_gan = self.loss.g_loss(fake_logits, real_logits)
            transport = (recipe.kinetic_transport_loss(fake_g, real_g)
                         if recipe.kinetic_transport_weight else loss_gan.new_zeros(()))
            transport_local = (recipe.kinetic_transport_local_loss(fake_g, real_g)
                               if recipe.kinetic_transport_local_weight else loss_gan.new_zeros(()))
            prior_reg = loss_gan.new_zeros(())
            if self.prior.z.requires_grad and (recipe.prior_regularizer == "vicreg" or recipe.prior_reg > 0):
                # Keep the historical unweighted diagnostic, even at weight 0.
                # Its contribution to the training objective is recipe.prior_reg.
                raw = self.prior.z if recipe.num_particles <= 1024 else self.prior.z[torch.unique(indices)]
                prior_reg = self.prior_regularizer(raw)
            loss_g = loss_gan + recipe.prior_reg * prior_reg
            if self.prior.z.requires_grad and recipe.prior_l2 > 0:
                loss_g = loss_g + recipe.prior_l2 * self.prior.z.square().mean()
            prior_manip_loss = None
            if self._prior_penalty:
                raw_rows = self.prior.z if recipe.num_particles <= 1024 else self.prior.z[torch.unique(indices)]
                prior_manip_loss = self.prior.regularize(
                    raw_rows,
                    sampled_codes=latent,
                    indices=indices,
                    generated=fake_g,
                    logits=fake_logits,
                )
                loss_g = loss_g + prior_manip_loss
            if recipe.kinetic_transport_weight:
                loss_g = loss_g + transport
            if recipe.kinetic_transport_local_weight:
                loss_g = loss_g + transport_local
            simplex_loss = None
            if (
                recipe.rhbi_simplex_weight > 0.0
                and self.prior_noise_generator is None
                and self.prior.z.requires_grad
                and len(self.prior.z) >= 2
                and hasattr(self.loss, "rhbi")
            ):
                base_table = unwrap_prior(self.prior).z
                simplex_loss = recipe.rhbi_simplex_weight * self.loss.rhbi.simplicial_manifold_loss(
                    base_table,
                    lambda z_q: self._postprocess(self.prior, self._generate(self.G, z_q, 0.0, noise)),
                    real_g,
                    record=getattr(self.opt_d, "record", None),
                    chart_weight=recipe.rhbi_simplex_chart_weight,
                    macro_settle=recipe.rhbi_simplex_macro_settle,
                )
                loss_g = loss_g + simplex_loss
            self.opt_g.zero_grad()
            protected_evaluator = None
            if recipe.constraint_geometry_mode == "strict_progress":
                # Same sampled indices and fixed within-component perturbation;
                # probes never call a sampler or draw training/output noise.
                locations = self.prior.z.detach().clone()
                sampled_latent = latent.detach().clone()
                def protected_evaluator():
                    rec = getattr(self.opt_d, "record", None)
                    prev_probe = getattr(rec, "_in_protected_probe", False) if rec is not None else False
                    if rec is not None:
                        rec._in_protected_probe = True
                    try:
                        replay = sampled_latent + (self.prior.z[indices] - locations[indices])
                        generated = self._postprocess(
                            self.prior, self._generate(self.G, replay, sigma_out, noise, rows=indices))
                        logits = critic(generated)
                        return (self.loss.g_loss(logits, real_logits),)
                    finally:
                        if rec is not None:
                            rec._in_protected_probe = prev_probe
            constraint_geometry_backward(loss_g, self.opt_g, (loss_gan,),
                                         protected_evaluator=protected_evaluator)
            self.policy.after_generator_backward(
                loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
            if self._gradient_response is not None:
                params = list(self.G.parameters())
                grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
                gain = self._gradient_response.observe(grads)
                # Replace noisy stochastic parameter gradients with bias-corrected EMA mean
                for p, m in zip(params, self._gradient_response.packet["means"]):
                    p.grad = (m / self._gradient_response.packet["weight"]).to(p).clone()
                # Temporarily scale generator proposal learning rate
                saved_lrs = []
                for group in self.opt_g.param_groups:
                    saved_lrs.append(group["lr"])
                    if group.get("role") == "generator":
                        group["lr"] = group["lr"] * gain
                try:
                    self.opt_g.step()
                finally:
                    for group, lr in zip(self.opt_g.param_groups, saved_lrs):
                        group["lr"] = lr
                    self._gradient_response.response["applied_steps"] += 1
            else:
                self.opt_g.step()
            self.policy.after_generator_step()
        finally:
            if hasattr(self.loss, "rhbi") and hasattr(self.loss.rhbi, "bind_discrete_prior_step"):
                self.loss.rhbi.bind_discrete_prior_step(False)
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
        self.policy.finish_step()
        if self.serve_after_step:
            self._serve_apply()
        result = {key: value.detach() for key, value in
                  dict(loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
                       prior_regularization=prior_reg, penalty=penalty).items()}
        result["step"] = self.completed_steps
        if prior_manip_loss is not None:
            # Present every step whenever a manipulator penalty is active.
            result["prior_manipulation"] = prior_manip_loss.detach()
        if recipe.kinetic_transport_weight:
            result["kinetic_transport"] = transport.detach()
        if recipe.kinetic_transport_local_weight:
            result["kinetic_transport_local"] = transport_local.detach()
        if simplex_loss is not None:
            result["rhbi_simplex"] = simplex_loss.detach()
        if collect_stats:
            result["penalty_stats"] = penalty_stats
        return result

    def _table_tester(self):
        return self.policy._table_tester()

    def _stray_gate(self):
        return self.policy._stray_gate()

    def _settle_observe(self, index):
        return self.policy._settle_observe(index)

    @torch.no_grad()
    def sample(self, n, *, ema=False, generator=None, output_noise=False,
               fixed_first_n=False, offset=0):
        """Draw live or EMA samples without changing modes or training RNGs.

        Samples are clean by default: output noise is a training regularizer.
        ``output_noise=True`` adds the current training output noise (drawn
        from the sampling stream, as before). Only the sampling stream
        (``generator`` or the trainer's evaluation stream) is consumed, so
        either choice leaves training trajectories unchanged.
        """
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        if type(output_noise) is not bool:
            raise ValueError("output_noise must be a boolean")
        stream = self.eval_generator if generator is None else self._stream(generator, 0)
        if stream in tuple(getattr(self, name) for name in self._STREAMS if name != "eval_generator"):
            raise ValueError("sampling requires a stream separate from training")
        model, prior = (self.ema_G, self.ema_prior) if ema else (self.G, self.prior)
        modes = [(module, module.training) for root in (model, prior) for module in root.modules()]
        devices = [self.device.index] if self.device.type == "cuda" else []
        try:
            model.eval()
            prior.eval()
            with torch.random.fork_rng(devices=devices):
                latent, rows = prior.sample(n, generator=stream, fixed_first_n=fixed_first_n, offset=offset)
                sigma = (self._output_sigma(output_noise_std(self.recipe, self.completed_steps))
                         if output_noise else 0.0)
                return self._postprocess(prior, self._generate(model, latent, sigma, stream, rows=rows))
        finally:
            for module, flag in modes:
                module.training = flag

    _STREAMS = ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")

    def state_dict(self):
        """Return an independent checkpoint (the training iterate; the served average is derived); save the caller's data cursor too."""
        self._check_checkpoint_boundary()
        swapped = self._fast is not None
        self._serve_release()
        try:
            return self._state_dict()
        finally:
            if swapped:
                self._serve_apply()

    def _state_dict(self):
        names = ("G", "D", "prior", "ema_G", "ema_prior")
        return deepcopy({
            **({"max_steps": self.max_steps} if self.max_steps != self.recipe.total_steps else {}),
            "serial_backward": True,
            **({"controller": self.controller.state_dict()} if self.controller is not None else {}),
            "schema": 4, "recipe": self.recipe.to_dict(),
            **self.recipe.prior_regularizer_checkpoint(self.prior_regularizer),
            "optimizer_options": self.optimizer_options, "penalty_options": self.penalty_options,
            "device": str(self.device), "dtype": str(self.dtype),
            "models": {name: getattr(self, name).state_dict() for name in names},
            "requires_grad": {name: {key: p.requires_grad for key, p in getattr(self, name).named_parameters()}
                              for name in names},
            "optimizers": [self.opt_g.state_dict(), self.opt_d.state_dict()],
            "initial_lrs": self.initial_lrs, "completed_steps": self.completed_steps,
            **({} if self.policy._feature_selection is None else {
                "backend_selection": self.policy._feature_selection.state_dict()}),
            # Compact metadata completes the reusable policy state without
            # duplicating image-sized models or optimizer tensors. Older flat
            # schema-4 checkpoints remain accepted below.
            "policy": {"last_output_sigma": self.last_output_sigma,
                       "roles": self.policy.roles, "row_semantics": self.policy.row_semantics,
                       "served_source": ("averaged" if self.recipe.serve_average > 0
                                         and self._serve_settled() else "fast")},
            "streams": {name: getattr(self, name).get_state() for name in self._STREAMS},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            **({"output_noise": {"log_sigma": self.log_output_sigma.detach()}}
               if self.log_output_sigma is not None else {}),
            **({"lr_settle": self.lr_settle.state_dict()} if self.lr_settle is not None else {}),
            **({"birth_death": self.birth_death.state_dict()} if self.birth_death is not None else {}),
            **({"row_evidence": self.row_evidence.state_dict()} if self.row_evidence is not None else {}),
            **({"surprise": self.policy.surprise.state_dict()} if self.policy.surprise is not None else {}),
            **({"loss_surprise": self.policy.loss_surprise.state_dict()}
               if self.policy.loss_surprise is not None else {}),
            **({"reopen_guard": self.policy.reopen_guard.state_dict()}
               if self.policy.reopen_guard is not None else {}),
            **({"gradient_response": self._gradient_response.state_dict()}
               if self._gradient_response is not None else {}),
        })

    def _check_checkpoint_boundary(self):
        if self._in_training_step or self.policy._phase != "ready":
            raise RuntimeError("trainer checkpoints require a completed update boundary")

    def load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state (a rejected checkpoint leaves the served model served)."""
        self._check_checkpoint_boundary()
        served = self._fast is not None
        try:
            return self._load_state_dict(state)
        except Exception:
            if served and self._fast is None:
                self._serve_apply()
            raise

    def _load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state.

        Recreate the same parameter freezing before loading. Validation of both
        optimizers (which carry the KA2 state) and all RNG states precedes any
        mutation of the live trainer. Earlier formulations cannot be resumed
        under KA2; use the release that wrote those checkpoints.
        """
        if self.policy._feature_selection is None and self.policy.reopen_guard is None:
            self._serve_release()
        if isinstance(state, dict) and state.get("schema") in (1, 2, 3):
            raise ValueError(
                f"schema-{state['schema']} GANTrainer checkpoints come from an older formulation "
                "and cannot resume under KA2; pin the release that wrote the checkpoint "
                "(0.8.0 for K3P), or start a new run")
        expected = (self.state_dict() if self.policy._feature_selection is None and self.policy.reopen_guard is None
                    else self._state_dict())
        if isinstance(state, dict) and (
                type(state.get("serial_backward", False)) is not bool
                or state.get("serial_backward", False) != self.serial_backward):
            raise ValueError("checkpoint serial_backward execution mode does not match trainer; "
                             "resume unmarked or False historical checkpoints from their pinned original source")
        if self._gradient_response is not None and isinstance(state, dict) and "gradient_response" not in state:
            raise ValueError("checkpoint lacks gradient response history; resume it with its original source "
                             "or start a new run; exact continuation cannot reconstruct omitted history")
        allowed = [set(expected), set(expected) - {"policy"}]
        if self.policy.loss_surprise is not None:
            allowed.extend([s - {"loss_surprise"} for s in list(allowed)])
        if (not isinstance(state, dict) or state.get("schema") != 4
                or set(state) not in allowed):
            raise ValueError("invalid GANTrainer checkpoint schema")
        saved_recipe = state["recipe"]
        if (isinstance(saved_recipe, dict) and "prior_update" not in saved_recipe
                and state.get("requires_grad", {}).get("prior") == {}):
            # Historical frozen tables were buffers; their saved parameter map
            # is an actual policy witness, never a candidate default inference.
            saved_recipe = {**saved_recipe, "prior_update": "frozen"}
        if not isinstance(saved_recipe, dict) or _normalized_recipe(saved_recipe) != _normalized_recipe(expected["recipe"]):
            raise ValueError("checkpoint recipe does not match trainer")
        self.recipe.restore_prior_regularizer(self.prior_regularizer, state.get("prior_regularizer"), validate_only=True)
        for key in ("optimizer_options", "penalty_options", "device", "dtype", "requires_grad"):
            if state[key] != expected[key]:
                raise ValueError(f"checkpoint {key} does not match trainer")
        saved_cap = state.get("max_steps", self.recipe.total_steps)
        if ((saved_cap is not None and (type(saved_cap) is not int or saved_cap < 1))
                or saved_cap != self.max_steps):
            raise ValueError("checkpoint execution budget does not match trainer")
        steps = state["completed_steps"]
        if type(steps) is not int or steps < 0 or (self.max_steps is not None and steps > self.max_steps):
            raise ValueError("invalid checkpoint step count")
        gradient_response = None
        if self._gradient_response is not None:
            gradient_response = deepcopy(self._gradient_response)
            gradient_response.load_state_dict(state["gradient_response"], self.G.parameters())
            if steps and (gradient_response.packet is None
                          or gradient_response.packet["count"] < steps
                          or gradient_response.response["applied_steps"] < steps):
                raise ValueError("incomplete checkpoint gradient response history")
        rates = state["initial_lrs"]
        if (not isinstance(rates, list) or len(rates) != 2
                or any(not isinstance(a, list) or len(a) != len(b) for a, b in zip(rates, self.initial_lrs))
                or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for row in rates for v in row)):
            raise ValueError("invalid checkpoint initial learning rates")
        if (not isinstance(state["models"], dict) or not isinstance(state["streams"], dict)
                or state["models"].keys() != expected["models"].keys()
                or state["streams"].keys() != expected["streams"].keys()):
            raise ValueError("invalid checkpoint model or RNG schema")
        for name, tensors in state["models"].items():
            current = expected["models"][name]
            if not isinstance(tensors, dict) or tensors.keys() != current.keys():
                raise ValueError(f"checkpoint model {name} has incompatible state")
            for key, value in current.items():
                saved = tensors[key]
                if isinstance(value, torch.Tensor):
                    if (not isinstance(saved, torch.Tensor) or saved.shape != value.shape
                            or saved.dtype != value.dtype):
                        raise ValueError(f"checkpoint model {name} has incompatible tensors")
                elif saved != value:
                    raise ValueError(f"checkpoint model {name} has incompatible state")
        if self.recipe.prior_regularizer == "coulomb" and self.recipe.prior_reg > 0 and self.recipe.prior_update == "learned":
            from .coulomb_prior import CENTER_BUFFER, RADIUS_BUFFER
            for name in ("prior", "ema_prior"):
                reference = state["models"][name]
                center = reference.get(CENTER_BUFFER, reference.get(f"base.{CENTER_BUFFER}"))
                radius = reference.get(RADIUS_BUFFER, reference.get(f"base.{RADIUS_BUFFER}"))
                if (center is None or radius is None
                        or not torch.isfinite(center).all()
                        or not torch.isfinite(radius)
                        or radius < 0):
                    raise ValueError("invalid checkpoint Coulomb initial reference")
        if not isinstance(state["optimizers"], list) or len(state["optimizers"]) != 2:
            raise ValueError("invalid checkpoint optimizer schema")
        try:
            for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
                _validate_optimizer_state(optimizer, values)
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid checkpoint optimizer state") from error
        try:
            seen_streams = {}
            for name, value in state["streams"].items():
                torch.Generator(device=self.device).set_state(value.cpu())
                owner = id(getattr(self, name))
                if owner in seen_streams and not torch.equal(value, seen_streams[owner]):
                    raise ValueError("aliased training streams have conflicting RNG states")
                seen_streams[owner] = value
                global_stream = (torch.default_generator if self.device.type == "cpu"
                                 else torch.cuda.default_generators[self.device.index])
                global_state = state["cpu_rng"] if self.device.type == "cpu" else state["cuda_rng"]
                if getattr(self, name) is global_stream and not torch.equal(value, global_state):
                    raise ValueError("global training stream has conflicting RNG state")
            torch.Generator(device="cpu").set_state(state["cpu_rng"].cpu())
            if self.device.type == "cuda":
                torch.Generator(device=self.device).set_state(state["cuda_rng"].cpu())
            elif state["cuda_rng"] is not None:
                raise ValueError("CPU trainer cannot load CUDA RNG state")
        except (TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid checkpoint RNG state") from error
        if self.log_output_sigma is not None:
            value = state["output_noise"].get("log_sigma") if isinstance(state["output_noise"], dict) else None
            if (not isinstance(value, torch.Tensor) or value.shape != self.log_output_sigma.shape
                    or value.dtype != self.log_output_sigma.dtype or not torch.isfinite(value).all()):
                raise ValueError("invalid checkpoint learnable output noise")
        if self.controller is not None:
            deepcopy(self.controller).load_state_dict(_state_to_device(state["controller"], self.device))
        prepared = (None if self.policy._feature_selection is None else
                    self.policy._feature_selection.prepare_restore(state["backend_selection"], state))
        validation_birth, validation_settle = ((self.birth_death, self.lr_settle) if prepared is None
                                             else prepared["controls"])
        if validation_settle is not None:
            deepcopy(validation_settle).load_state_dict(_state_to_device(state["lr_settle"], self.device),
                                                    (self.opt_g, self.opt_d))
        if validation_birth is not None:
            validation_birth.check_state(state["birth_death"])
        if self.row_evidence is not None:
            self.row_evidence.check_state(state["row_evidence"])
        if self.policy.surprise is not None:
            deepcopy(self.policy.surprise).load_state_dict(_state_to_device(state["surprise"], self.device))
        if self.policy.loss_surprise is not None and "loss_surprise" in state:
            deepcopy(self.policy.loss_surprise).load_state_dict(_state_to_device(state["loss_surprise"], self.device))
        if self.policy.reopen_guard is not None:
            self.policy.reopen_guard.check_state(state["reopen_guard"], roles=self.policy.roles,
                                                 completed_steps=state["completed_steps"],
                                                 anchor_started=self.policy._loss_epoch(state["optimizers"][1]))
        metadata = state.get("policy")
        if "policy" in state:
            if (not isinstance(metadata, dict) or metadata.keys() != expected["policy"].keys()
                    or metadata["roles"] != self.policy.roles
                    or metadata["row_semantics"] != self.policy.row_semantics):
                raise ValueError("checkpoint policy topology does not match trainer")
            sigma = metadata["last_output_sigma"]
            if sigma is not None and (type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0):
                raise ValueError("invalid checkpoint policy output sigma")
            settled = any(value is not None and role == "table" and value.get("last_decisive") == -1
                          for row, roles in zip(state.get("lr_settle", []), self.policy.roles)
                          for value, role in zip(row, roles))
            source = "averaged" if self.recipe.serve_average > 0 and settled else "fast"
            if self.policy._feature_selection is not None:
                feature_source = self.policy._feature_selection.saved_served_source(state)
                if feature_source is not None:
                    source = feature_source
            if metadata["served_source"] != source:
                raise ValueError("inconsistent checkpoint policy served source")
        if prepared is not None or self.policy.reopen_guard is not None:
            self._serve_release()
        if prepared is not None:
            self.policy._feature_selection.commit_restore(prepared)
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        self.recipe.restore_prior_regularizer(self.prior_regularizer, state.get("prior_regularizer"))
        if self.log_output_sigma is not None:
            with torch.no_grad():
                self.log_output_sigma.copy_(state["output_noise"]["log_sigma"])
        for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
            optimizer.load_state_dict(deepcopy(values))
        if self.controller is not None:
            self.controller.load_state_dict(_state_to_device(state["controller"], self.device))
        if self.lr_settle is not None:
            self.lr_settle.load_state_dict(_state_to_device(state["lr_settle"], self.device),
                                          (self.opt_g, self.opt_d))
        if self.birth_death is not None:
            self.birth_death.load_state_dict(state["birth_death"])
        if self.row_evidence is not None:
            self.row_evidence.load_state_dict(state["row_evidence"])
        if self.policy.surprise is not None:
            self.policy.surprise.load_state_dict(_state_to_device(state["surprise"], self.device))
        if self.policy.loss_surprise is not None and "loss_surprise" in state:
            self.policy.loss_surprise.load_state_dict(_state_to_device(state["loss_surprise"], self.device))
        if self.policy.reopen_guard is not None:
            self.policy.reopen_guard.load_state_dict(state["reopen_guard"], roles=self.policy.roles,
                                                     completed_steps=state["completed_steps"],
                                                     anchor_started=self.policy._loss_epoch(state["optimizers"][1]))
        self._gradient_response = gradient_response
        self.initial_lrs, self.completed_steps = deepcopy(rates), steps
        self.last_output_sigma = None if metadata is None else metadata["last_output_sigma"]
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        if self.serve_after_step:
            self._serve_apply()


def _policy_property(name):
    def get(trainer):
        return getattr(trainer.policy, name)

    def set_value(trainer, value):
        setattr(trainer.policy, name, value)

    return property(get, set_value)


for _name in ("log_output_sigma", "last_output_sigma", "initial_lrs", "ema_G", "ema_prior",
              "_fast", "completed_steps", "controller", "lr_settle", "birth_death", "row_evidence",
              *UpdatePolicy._STREAMS):
    setattr(GANTrainer, _name, _policy_property(_name))


def _trainer_roles(trainer):
    # Preserve the historical receipt names; the reusable API uses explicit table/noise roles.
    return [["prior" if role == "table" else "generator" if role == "noise" else role
             for role in row] for row in trainer.policy.roles]


GANTrainer.roles = property(_trainer_roles)
