"""Shared hyperparameters and small component factories; callers own control flow."""
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
try:
    from .recipe_compat import SEARCH_RECIPE_DEFAULTS
    SEARCH_RECIPE_DEFAULTS["generator_gradient_response"] = "none"
    SEARCH_RECIPE_DEFAULTS.update(
        rhbi_weight=0.0,
        rhbi_rho=2.0,
        rhbi_tau=2.0,
        rhbi_eps=0.1,
        rhbi_iters=8,
        rhbi_c_max=20.0,
        rhbi_beta_ell=0.1,
        rhbi_nn_quantile=0.25,
        rhbi_critic_weight=0.0,
        rhbi_self_coupling="hollow",
        rhbi_relativistic_weight=0.0,
        rhbi_lambda_floor=0.0,
        rhbi_gamma_gauge=0.0,
        rhbi_rapidity_screening=True,
        rhbi_scale_log_odds=False,
        rhbi_conditional_fibre="joint",
        rhbi_akt_horizon=False,
        rhbi_akt_mode="sqrt",
        rhbi_slack_gain=0.0,
        rhbi_chord_weight=0.0,
        rhbi_thermal_tau=0.0,
        rhbi_thermal_shock_mu=0.15,
        rhbi_settle_beta=0.0,
        rhbi_settle_mode="residual",
        rhbi_fibre_rel_blend=0.0,
        rhbi_step_damp=0.0,
        rhbi_cond_ot_weight=0.0,
        rhbi_cond_ot_mode="kernel_ed",
        rhbi_simplex_weight=0.0,
        rhbi_cvx_mu_gain=0.40,
        rhbi_simplex_chart_weight=0.0,
        rhbi_tail_spill_gain=0.0,
        rhbi_simplex_macro_settle=False,
    )
except (ImportError, ValueError):
    pass

# Promote a registered trainer preset here; model-family factories inherit it.
DEFAULT_RECIPE = "empf"


def _reject_table_storage_owners(table, tensors):
    """A frozen sampled table needs storage independent of mutable owners."""
    import torch
    if not table.numel() or table.layout != torch.strided:
        return
    address = table.untyped_storage().data_ptr()
    for tensor in tensors:
        if (isinstance(tensor, torch.Tensor) and tensor.numel()
                and tensor.layout == torch.strided and tensor.device == table.device
                and tensor.untyped_storage().data_ptr() == address):
            raise ValueError("frozen prior must not share storage with networks or optimizer parameters")


@dataclass(frozen=True, init=False)
class Recipe:
    """Resolved hyperparameters plus role-named factories (``make_*``) for one model.

    Fields are validated at construction; ``replace`` returns a modified copy.
    Use ``get_recipe()`` for new runs and ``Recipe.from_dict(saved_fields)``
    for restoration. Direct construction requires every field: omitted fields
    must never silently select the historical KA2 restoration defaults.
    Two fields expose KA2 ablations; their defaults retain that formulation:

    * ``reg_anchor_weight`` (1.0) scales the critic penalty's EMA-anchor term,
      which ties the critic's input gradient to its parameter EMA once the
      controller anchors the critic. 0 removes the term (no ``ema_critic`` needed).
    * ``direct_particle_gain`` (True) lets a direct sample-particle group
      (``make_generator_optimizer(direct_particles=...)``) raise its LR by up
      to 2x while successive centered gradients agree. False keeps that
      group's LR at the scheduled value (``direct_particle_betas`` still apply).
    """
    name: str = "ka2"
    critic_formulation: str = "ka2"
    model: str = "gan"
    # CUDA convolutional networks only; weights, optimizers and sampling stay FP32.
    training_precision: str = "float32"
    z_dim: int = 2
    num_particles: int = 20_000
    prior_kind: str = "particles"
    sigma_rel: float = 0.0
    standardize: bool = True
    num_classes: int | None = None
    conditioning: str = "scalar"
    ucd_target: str = "class"
    ucd_weight: float = 0.02
    alpha_bar: tuple[float, ...] = (1.0, 0.9, 0.5, 0.05, 0.0001)
    batch_size: int = 2048
    total_steps: int | None = 7_000
    continuous_policy: str | None = None
    lr: float = 0.00425
    d_lr_mult: float = 1.0
    prior_lr_mult: float = 2.0
    betas: tuple[float, float] = (0.0, 0.999)
    prior_betas: tuple[float, float] | None = None
    # None uses critic_formulation. Explicit legacy arms use the K3P optimizer;
    # fixed R1/R2 and BCap retain their released L2 units.
    reg_arm: str | None = None
    reg_coeff: float = 1.0
    # Explicit fixed-arm real/fake terms. Defaults preserve the paired kernels;
    # R1-only is (1, 0), with the original half-gamma real coefficient retained.
    reg_real_weight: float = 1.0
    reg_fake_weight: float = 1.0
    reg_kappa: float = 1.0
    reg_every: int = 1
    ebi_beta: float = 30.0
    # Optional finite same-training-panel critic step guard; none preserves history.
    critic_step_mode: str = "none"
    prior_reg: float = 0.0
    # Explicit joint inverse supervision; hosts supply paired reconstructed
    # outputs (raw logits for categorical coordinates) and their actual targets.
    joint_reconstruction_weight: float = 0.0
    # The recipe owns learning/freezing and every latent-prior penalty. Tasks
    # supply the initial distribution, capacity and sampling law.
    prior_update: str = "learned"
    prior_regularizer: str = "vicreg"
    prior_reg_target_std: float = 1.0
    prior_reg_eps: float = 1e-4
    prior_l2: float = 0.0
    # Post-optimizer proximal spring to the initial latent table; zero is exact
    # baseline behavior. Unlike a gradient penalty, restoration retains size.
    prior_proximal_strength: float = 0.0
    prior_coulomb_softening: float = .1
    prior_coulomb_confinement: float = .35
    prior_coulomb_neighbors: int = 8
    prior_sigreg_projections: int = 32
    prior_sigreg_knots: int = 17
    ema_decay: float = 0.995
    lr_anneal_start: float = 0.6
    lr_floor: float = 0.05
    # G/D follow their own cosine over min(total, horizon cap)
    # down to network_lr_floor; the prior keeps the full-budget cosine above.
    # None means "same as lr_floor"; a None horizon cap means the full budget.
    network_lr_floor: float | None = 0.01
    network_lr_horizon_cap: int | None = 1600
    reg_anchor_min_decay: float = 0.90
    # Ablation switches; see the class docstring.
    reg_anchor_weight: float = 1.0
    direct_particle_gain: bool = True
    # Critic spike guard (d_guard_ratio=0 disables it).
    d_guard_ratio: float = 5.0
    d_guard_min_steps: int = 200
    # A2 sparse latent-row damping (0 disables it).
    latent_damping_max_rate: float = 0.5
    # Direct sample-particle response betas (make_generator_optimizer).
    direct_particle_betas: tuple[float, float] = (0.0, 0.9)
    # Critic input noise: peak std at the first update, linear to 0 by
    # input_noise_anneal_end * total_steps. Generator output noise: linear
    # warmup from 0 to output_noise_std over output_noise_warmup * total_steps.
    input_noise_std: float = 0.5
    input_noise_anneal_end: float = 0.1
    output_noise_std: float = 0.029
    output_noise_warmup: float = 0.2
    encoder_mode: str = "none"
    routing_temperature: float = 0.25
    distance_reduction: str = "sum"
    observation_sigma: float = 0.03
    reconstruction_weight: float = 1.0
    conditional_energy_weight: float = 0.0
    conditional_energy_samples: int = 8
    # Incomplete-data component callers only; unsupported hosts reject these.
    masked_mixture_weight: float = 0.0
    masked_transport_weight: float = 0.0
    imputer_component_weight: float = 0.0
    # Opt-in label-free transport signal on the same G-phase target batch.
    kinetic_transport_weight: float = 0.0
    generator_gradient_response: str = "none"
    kinetic_transport_local_weight: float = 0.0
    kinetic_transport_projections: int = 32
    # AMSGrad for every recipe optimizer (G, prior and critic). Intended for a
    # G/D LR that stays high: the Adam step then shrinks with the gradient at
    # equilibrium instead of creeping up as the second moment decays. The
    # default keeps plain Adam (bit-identical to the release without it).
    amsgrad: bool = False
    # Ablation switches (appended; defaults reproduce the release bit-for-bit).
    # critic_r1_real=False drops only the real-data squared-gradient (R1) term
    # from the KA2 penalty's part A (fake cap, part-B caps and proximity stay).
    critic_r1_real: bool = True
    # critic_payoff_damping=False drops the continuous controller's D-only
    # 1/(1+payoff_error^2) LR factor (G/prior scales unchanged).
    critic_payoff_damping: bool = True
    # Generator output noise: "fixed" (output_noise_std), "learnable" (a
    # trainer-owned log-sigma trained with the generator loss through the
    # reparameterized noise, initialized at log(output_noise_std)) or
    # "mobility" (output_noise_std times the continuous controller mobility).
    output_noise_mode: str = "fixed"
    # LR control of a continuous (dv12) recipe: "mobility" (DV12: mobility and
    # game_trust scales, bit-identical to the release) or "stationarity" (per
    # optimizer group Pflug/SASA-style settling test in intrinsic time; see
    # continuous.SettleTest). Mobility is still updated (it drives
    # output_noise_mode="mobility") and game_trust still scales the KA2 anchor.
    lr_control: str = "mobility"
    # Fisher-Rao birth-death moves on the particle table (kNN density ratio,
    # sequential evidence, BH at q=.05; see birth_death.py). Off by default.
    particle_birth_death: bool = False
    # Per-row force-persistence evidence (row_evidence.py): rows whose own gradient history has a significant mean
    # keep the full table rate and do not vote in the table tester; descent is held while many rows are unbalanced.
    row_evidence_gate: bool = False
    # Table stationarity tester: which evidence may raise the table rate again ("any" | "both" | "never" | "anchor"), see continuous.py.
    table_release_rule: str = "any"
    # Parts of the row-evidence gate (ablation switches; all True is the declared mechanism).
    row_evidence_hot: bool = True
    row_evidence_exclude: bool = True
    row_evidence_hold: bool = True
    # Space in which birth-death compares the table's samples with the real reservoir: "data" (the samples themselves) or
    # "critic" (the critic's penultimate features: no data-space metric is needed). Locality uses the table's own latents.
    birth_death_space: str = "data"
    # Served model: 0 = the live iterate (the fast parameters) is what samples are drawn from. m > 0 = the averaged model is the model of
    # record: samples, evaluations and checkpoints' served parameters are an exponential average of the training iterate whose window is
    # m table-tester blocks (steps per block = b / s of the table tester, both controller state); the fast iterate continues to train.
    serve_average: float = 0.0
    # What re-opens the learning-rate ladders when the game changes under them: "data" = the base's real-batch drift
    # statistic (fast vs slow mean of random features of standardized RAW real batches, z-score > 3: data space);
    # "none" = no re-open and no statistic of the real batch is read (the ladders can still release on their own evidence);
    # "optimizer" = continuous.OptimizerSurprise: a sustained jump of the optimizers' own Adam step signal (|g| / sqrt(v_hat))
    # against its calm level re-opens every ladder once and restarts the generator-side Adam moments (hysteresis re-arm).
    reopen_signal: str = "data"
    # What an "optimizer" re-open does to the KA2 critic anchor: "hold" = nothing (a blind recipe forces the anchor on, W = 1,
    # and damps its EMA by game_trust, so after a re-open the anchor keeps pulling the critic toward the pre-event EMA critic);
    # "release" = the re-open counts as drift evidence for KA2 (evidence = 1) from the fire until KA2's own surprise ratio has
    # risen above REL_HI and fallen back below REL_LO: KA2's native release / EMA tracking / reseed rules then apply. Event-gated.
    reopen_anchor: str = "hold"
    # Optional acquisition guard for optimizer re-opens. None retains R1's
    # exact legacy law; "settled" requires a contracted network ladder at
    # the start of the surprise excursion and rebases known KA2 loss epochs.
    reopen_guard: str | None = None
    # Null law of the row-evidence gate: "theory" = the exact-law p-value of the statistic as is; "scaled" = the same law applied to t2 / c, where
    # c >= 1 is the smallest scale that makes the median p-value over the tested rows .5 (the typical row is the null: a slowly varying force
    # shared by neighbouring rows, correlated gradients or a different noise level scale every row's statistic alike; only rows that are extreme
    # relative to the bulk are flagged).
    row_evidence_null: str = "theory"
    # Birth-death also re-draws table rows that have no real support. With birth_death_space="critic": a split-conformal isolation test of every
    # row against the real reservoir in the critic's feature space (k-th neighbour distance over the local real scale; valid when the fake
    # law equals the real law, conservative for rows scored at their clean centres), Benjamini-Hochberg at the birth-death level over the table; flagged rows become clones of uniformly drawn unflagged
    # rows, and only while the flagged fraction is at most that same level (a table that is still mostly in transit is not resampled).
    birth_death_isolation: bool = False
    # Scale of the critic's feature space in critic-space birth-death and the support test: "none" = the features as the critic delivers them;
    # "std" = every feature divided by its standard deviation on the reference half (even rows) of the real reservoir. The critic function does not fix
    # the scale of a hidden unit (rescale a ReLU-type unit and its outgoing weight inversely: same function, different Euclidean distances), so the
    # raw metric depends on an arbitrary parametrisation; "std" is invariant to that symmetry and is a function of the reference half only.
    birth_death_feature_scale: str = "none"
    # Opt-in bounded feature law. Auto is conservatively scoped to a complete
    # raw-output moment frame and finite conformal/BH resolution.
    birth_death_backend: str = "knn"
    birth_death_cells: int = 64
    birth_death_metric_rank: int = 8
    birth_death_chunk: int = 256
    birth_death_parent_policy: str = "real_anchor"
    # Independent E22 uses its original equal-mass particle statistics.
    # routed_paired is a distinct conditional dense-bank adaptation, bound to
    # an explicit RoutedRows contract in the caller-owned policy API.
    row_policy: str = "independent"
    # Explicit optimizer family: the default retains KA2/K3P interventions.
    # adam uses the declared plain Adam law with only observer bookkeeping. Its fixed
    # penalties and optional moment/coefficient cosine schedules are declared
    # independently of any candidate name. Non-None endpoints interpolate from
    # betas[1]/reg_coeff over anneal_end * total_steps completed updates, then
    # hold. External execution limits do not change that declared horizon.
    optimizer_family: str = "formulation"
    # Dual-norm experiments retain the fixed loss/penalty and LR multiplier.
    # lr is their step size; isolation arms can pin native Adam groups to the
    # original baseline rate independently of the normalized-step sweep.
    optimizer_momentum: float = 0.0
    # Fixed gradient scale for smoothed polar, bias and sampled-prior updates.
    # epsilon in the smoothed-polar paper is optimizer_smoothing ** 2.
    optimizer_smoothing: float = 0.0
    # Explicit convolution adaptation; dense/default checkpoint packets stay unchanged.
    optimizer_convolution: str = "none"
    # Hybrid nonpolar rules change only CUDA channel matrices larger than 32.
    optimizer_convolution_rule: str = "polar"
    # Non-native backends are explicit numerical trainer changes. Newton-Schulz
    # accelerates annotated convolution kernels; dense matrices retain native SVD.
    # Native retains the original computation and archived checkpoint identity.
    optimizer_svd_backend: str = "native"
    # Versioned dense-only numerical profile; annotated convolution, vector
    # and sampled-prior rules remain owned by their existing settings.
    optimizer_dense_profile: str = "default"
    # Opt-in protection of existing G/encoder/prior objectives. Component
    # callers bind their protected losses before backward only when enabled.
    constraint_geometry_mode: str = "none"
    optimizer_adam_lr: float | None = None
    eps: float = 1e-8
    beta2_end: float | None = None
    beta2_anneal_end: float = 0.2
    reg_coeff_end: float | None = None
    reg_coeff_anneal_end: float = 0.2
    # Objective on raw scores; regularization is selected independently.
    loss: str = "relativistic"
    # G/E use betas/eps. D inherits unless these are explicit; prior inherits
    # G/E except for its existing prior_betas and optional prior_eps.
    d_betas: tuple[float, float] | None = None
    d_eps: float | None = None
    prior_eps: float | None = None
    loss_labels: tuple[float, float, float] = (0.0, 1.0, 1.0)
    adam_variant: str = "pytorch"
    # Exponential decay counts completed whole training updates, not role
    # applications or sampling calls. Its horizon is independent of total_steps.
    lr_schedule: str = "cosine"
    lr_decay_rate: float = 0.96
    lr_decay_steps: int = 50_000
    lr_decay_staircase: bool = False
    # BISPF shape/step gates (particlegan.bispf.BISPFGates) for the BISPF critic
    # penalty, SinkhornRelativisticLoss and the SBNP prior regularizer. The
    # defaults reproduce the historical hard-coded regimes. False is the
    # master switch for a pure global recipe that disables every
    # shape/step-conditioned branch. Overrides are (gate, value) pairs or a
    # {gate: value} mapping, stored as sorted pairs.
    bispf_shape_gates: bool = True
    bispf_gate_overrides: tuple = ()
    # Latent code law, owned with prior_update. "table" (default) samples table
    # rows. "fresh_gaussian" is the matched fixed-Gaussian latent control: it
    # requires prior_update="frozen" (checked by apply_prior_policy), keeps the task's initial table and index
    # draws, and replaces each sampled code with init_std * N(0, I) of the same
    # dimension (ParticlePrior.code_law). Opt-in; not part of any preset.
    prior_code_law: str = "table"
    kappa_radial: float = 1.0
    # Relativistic Hellinger-Born-Infeld Transport (RHBI) variational parameters.
    rhbi_weight: float = 0.0
    rhbi_rho: float = 2.0
    rhbi_tau: float = 2.0
    rhbi_eps: float = 0.1
    rhbi_iters: int = 8
    rhbi_c_max: float = 20.0
    rhbi_beta_ell: float = 0.1
    rhbi_nn_quantile: float = 0.25
    rhbi_critic_weight: float = 0.0
    rhbi_self_coupling: str = "hollow"
    rhbi_relativistic_weight: float = 0.0
    rhbi_lambda_floor: float = 0.0
    rhbi_gamma_gauge: float = 0.0
    rhbi_rapidity_screening: bool = True
    rhbi_scale_log_odds: bool = False
    rhbi_conditional_fibre: str = "joint"
    rhbi_akt_horizon: bool = False
    rhbi_akt_mode: str = "sqrt"
    rhbi_slack_gain: float = 0.0
    rhbi_chord_weight: float = 0.0
    rhbi_thermal_tau: float = 0.0
    rhbi_thermal_shock_mu: float = 0.15
    rhbi_settle_beta: float = 0.0
    rhbi_settle_mode: str = "residual"
    rhbi_fibre_rel_blend: float = 0.0
    rhbi_step_damp: float = 0.0
    rhbi_cond_ot_weight: float = 0.0
    rhbi_cond_ot_mode: str = "kernel_ed"
    rhbi_simplex_weight: float = 0.0
    rhbi_cvx_mu_gain: float = 0.40
    rhbi_simplex_chart_weight: float = 0.0
    rhbi_tail_spill_gain: float = 0.0
    rhbi_simplex_macro_settle: bool = False

    def __init__(self, **resolved_fields):
        declared = {field.name for field in fields(type(self))}
        unknown = resolved_fields.keys() - declared
        if unknown:
            raise TypeError(f"Unknown recipe field(s): {', '.join(sorted(unknown))}")
        if resolved_fields.keys() != declared:
            raise TypeError(
                "Recipe() no longer selects implicit defaults. Use get_recipe() "
                "for a new run (or get_recipe('ka2') for an explicit historical "
                "preset); use Recipe.from_dict(saved_fields) to restore a saved "
                "configuration. Direct construction requires every resolved field."
            )
        for name, value in resolved_fields.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @classmethod
    def from_dict(cls, saved_fields):
        """Restore a saved recipe without resolving today's default preset.

        Compact historical snapshots omit neutral fields added later. Their
        restoration defaults are retained here, never used for a new run.
        """
        if not isinstance(saved_fields, Mapping):
            raise TypeError("saved recipe fields must be a mapping")
        defaults = {field.name: field.default for field in fields(cls)}
        unknown = saved_fields.keys() - defaults.keys()
        if unknown:
            raise TypeError(f"Unknown recipe field(s): {', '.join(sorted(unknown))}")
        return cls(**{**defaults, **saved_fields})

    def __post_init__(self):
        if (type(self.conditional_energy_samples) is not int
                or self.conditional_energy_samples < 2):
            raise ValueError("conditional_energy_samples must be an integer >= 2")
        if type(self.bispf_shape_gates) is not bool:
            raise ValueError("bispf_shape_gates must be a boolean")
        overrides = self.bispf_gate_overrides
        try:
            pairs = overrides.items() if isinstance(overrides, Mapping) else overrides
            pairs = tuple((str(name), value) for name, value in pairs)
        except (TypeError, ValueError):
            raise ValueError("bispf_gate_overrides must be (gate, value) pairs or a mapping") from None
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError("bispf_gate_overrides names must be unique")
        pairs = tuple(sorted(pairs, key=lambda pair: pair[0]))
        object.__setattr__(self, "bispf_gate_overrides", pairs)
        if pairs or not self.bispf_shape_gates:
            self.bispf_gates()
        if self.training_precision not in ("float32", "bf16_channels_last"):
            raise ValueError("training_precision must be float32 or bf16_channels_last")
        if self.prior_update not in ("learned", "frozen"):
            raise ValueError("prior_update must be learned or frozen")
        if self.prior_code_law not in ("table", "fresh_gaussian"):
            raise ValueError("prior_code_law must be table or fresh_gaussian")
        # fresh_gaussian needs a frozen table. That is enforced when the policy is
        # bound (apply_prior_policy -> ParticlePrior.set_code_law), not here: hosts
        # construct the initial table through recipe.replace(prior_update="learned").
        from .coulomb_prior import CoulombPriorRegularizer
        CoulombPriorRegularizer(softening=self.prior_coulomb_softening,
                               confinement=self.prior_coulomb_confinement,
                               neighbors=self.prior_coulomb_neighbors)
        if self.prior_regularizer not in ("vicreg", "none", "coulomb", "sigreg", "moment_tether", "initial_mmd", "sbnp", "empf"):
            raise ValueError("prior_regularizer must be vicreg, none, coulomb, sigreg, moment_tether, initial_mmd, sbnp or empf")
        for key, minimum in (("prior_sigreg_projections", 1), ("prior_sigreg_knots", 2)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("prior_reg_target_std", "prior_reg_eps", "prior_l2", "prior_proximal_strength"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.prior_reg_eps == 0:
            raise ValueError("prior_reg_eps must be positive")
        if self.prior_regularizer == "none" and self.prior_reg != 0:
            raise ValueError("prior_regularizer='none' requires prior_reg=0")
        if self.prior_update == "frozen" and (self.row_evidence_gate or self.particle_birth_death):
            raise ValueError("frozen priors require row evidence and birth/death disabled")
        if self.prior_proximal_strength and (
                self.optimizer_family != "dualnorm" or self.optimizer_momentum
                or self.constraint_geometry_mode not in ("none", "direction_blend")
                or self.continuous_policy is not None or self.lr_control != "mobility"
                or self.row_evidence_gate or self.particle_birth_death):
            raise ValueError("prior_proximal_strength requires zero-momentum full DualNorm, "
                             "geometry none/direction_blend and no adaptive prior policies")
        from .gan_loss import GANLoss
        objective = GANLoss(self.loss, labels=self.loss_labels)
        object.__setattr__(self, "loss_labels", objective.labels)
        from .optim.dualnorm import NORMALIZED_FAMILIES
        if self.optimizer_family not in ("formulation", "adam", *NORMALIZED_FAMILIES):
            raise ValueError("unknown optimizer_family")
        if isinstance(self.optimizer_momentum, bool) or self.optimizer_momentum not in (0., .5, .9):
            raise ValueError("optimizer_momentum must be 0, 0.5 or 0.9")
        if self.optimizer_momentum != 0 and self.optimizer_family not in ("dualnorm", "dualnorm_D_only"):
            raise ValueError("optimizer_momentum requires a dualnorm optimizer family")
        if (type(self.optimizer_smoothing) not in (int, float)
                or not math.isfinite(self.optimizer_smoothing) or self.optimizer_smoothing < 0):
            raise ValueError("optimizer_smoothing must be finite and nonnegative")
        if self.optimizer_smoothing and self.optimizer_family != "dualnorm":
            raise ValueError("optimizer_smoothing requires optimizer_family='dualnorm'")
        if self.constraint_geometry_mode not in ("none", "nonascent", "strict_progress", "direction_blend"):
            raise ValueError("constraint_geometry_mode must be none, nonascent, strict_progress or direction_blend")
        if self.constraint_geometry_mode != "none" and (self.optimizer_family != "dualnorm" or self.optimizer_momentum):
            raise ValueError("constraint_geometry requires zero-momentum full DualNorm")
        if self.critic_step_mode not in ("none", "finite_cap"):
            raise ValueError("critic_step_mode must be none or finite_cap")
        if self.critic_step_mode == "finite_cap" and (
                self.optimizer_family != "dualnorm" or self.optimizer_momentum
                or self.effective_critic_formulation != "bcap" or self.reg_every != 1
                or self.input_noise_std or self.output_noise_std):
            raise ValueError("finite_cap requires zero-momentum full DualNorm, every-update BCAP, and zero input/output noise")
        if self.optimizer_convolution not in ("none", "per_offset"):
            raise ValueError("optimizer_convolution must be none or per_offset")
        if self.optimizer_convolution != "none" and self.optimizer_family != "dualnorm":
            raise ValueError("optimizer_convolution requires optimizer_family='dualnorm'")
        if self.optimizer_convolution_rule not in ("polar", "frobenius", "quintic"):
            raise ValueError("optimizer_convolution_rule must be polar, frobenius or quintic")
        if self.optimizer_convolution_rule != "polar" and (
                self.optimizer_family != "dualnorm" or self.optimizer_convolution != "per_offset"
                or self.optimizer_svd_backend not in ("native", "batched")):
            raise ValueError("nonpolar rules require full dualnorm, per_offset convolution and native or batched SVD backend")
        if self.optimizer_svd_backend not in ("native", "cpu", "newton_schulz", "batched"):
            raise ValueError("optimizer_svd_backend must be native, cpu, newton_schulz or batched")
        if self.optimizer_svd_backend != "native" and self.optimizer_family != "dualnorm":
            raise ValueError("optimizer_svd_backend requires optimizer_family='dualnorm'")
        if self.optimizer_svd_backend == "newton_schulz" and (
                self.optimizer_convolution != "per_offset" or self.optimizer_smoothing <= 0):
            raise ValueError("newton_schulz requires per_offset convolution and positive optimizer_smoothing")
        if self.optimizer_svd_backend == "batched" and self.optimizer_convolution != "per_offset":
            raise ValueError("batched SVD requires per_offset convolution")
        if (type(self.optimizer_dense_profile) is not str
                or self.optimizer_dense_profile not in ("default", "gram_fp64_guarded_v1",
                    "gram_fp64_svd_diagnostic_v2", "polar_cpu_v1", "newton_schulz_v1",
                    "newton_schulz_hybrid_v1")):
            raise ValueError("unknown optimizer_dense_profile")
        if self.optimizer_dense_profile in ("gram_fp64_guarded_v1", "gram_fp64_svd_diagnostic_v2", "polar_cpu_v1") and (
                self.optimizer_family != "dualnorm" or self.optimizer_momentum != 0
                or (self.optimizer_dense_profile != "polar_cpu_v1" and self.optimizer_smoothing != .001)
                or self.optimizer_svd_backend not in ("native", "batched")):
            raise ValueError("optimizer_dense_profile requires zero-momentum full DualNorm, "
                             "smoothing .001 and native or batched SVD backend")
        if self.optimizer_dense_profile in ("newton_schulz_v1", "newton_schulz_hybrid_v1") and (
                self.optimizer_family != "dualnorm" or self.optimizer_smoothing <= 0):
            raise ValueError("Newton-Schulz optimizer_dense_profile requires full DualNorm "
                             "and positive optimizer_smoothing")
        if self.optimizer_adam_lr is not None:
            if (isinstance(self.optimizer_adam_lr, bool) or not math.isfinite(self.optimizer_adam_lr)
                    or self.optimizer_adam_lr <= 0):
                raise ValueError("optimizer_adam_lr must be None or finite and positive")
            if self.optimizer_family not in ("dualnorm_D_only", "particle_rownorm_only"):
                raise ValueError("optimizer_adam_lr is supported only by isolation arms")
        if self.d_betas is not None:
            if (not isinstance(self.d_betas, (list, tuple)) or len(self.d_betas) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
                           for v in self.d_betas)):
                raise ValueError("d_betas must contain two finite numbers in [0, 1) or be None")
            object.__setattr__(self, "d_betas", tuple(float(v) for v in self.d_betas))
        for key in ("d_eps", "prior_eps"):
            value = getattr(self, key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive or None")
        if self.adam_variant not in ("pytorch", "tensorflow_v1"):
            raise ValueError("adam_variant must be pytorch or tensorflow_v1")
        if self.adam_variant == "tensorflow_v1" and (
                self.optimizer_family != "adam" or self.amsgrad or self.beta2_end is not None):
            raise ValueError("tensorflow_v1 requires plain Adam, no AMSGrad and constant betas")
        if self.lr_schedule not in ("cosine", "constant", "exponential"):
            raise ValueError("lr_schedule must be cosine, constant or exponential")
        if (type(self.lr_decay_steps) is not int or self.lr_decay_steps <= 0
                or type(self.lr_decay_rate) not in (int, float)
                or not math.isfinite(self.lr_decay_rate) or not 0 < self.lr_decay_rate <= 1
                or type(self.lr_decay_staircase) is not bool):
            raise ValueError("invalid exponential LR rate, steps or staircase")
        if self.lr_schedule != "cosine" and (self.continuous_policy is not None or self.network_lr_horizon_cap is not None):
            raise ValueError("explicit constant/exponential LR requires a scheduled recipe without a horizon cap")
        if self.generator_gradient_response not in ("none", "coherent_G_mean_v1", "coherent_G_damped_v2"):
            raise ValueError("unknown generator gradient response mode")
        if isinstance(self.eps, bool) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        for name in ("beta2_anneal_end", "reg_coeff_anneal_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.beta2_end is not None and (
                isinstance(self.beta2_end, bool) or not math.isfinite(self.beta2_end) or not 0 <= self.beta2_end < 1):
            raise ValueError("beta2_end must be None or in [0, 1)")
        if self.reg_coeff_end is not None and (
                isinstance(self.reg_coeff_end, bool) or not math.isfinite(self.reg_coeff_end) or self.reg_coeff_end < 0):
            raise ValueError("reg_coeff_end must be None or finite and nonnegative")
        for name in ("reg_real_weight", "reg_fake_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if ((self.reg_real_weight, self.reg_fake_weight) != (1., 1.)
                and self.reg_arm not in ("a_r1r2", "b_cap")
                and self.critic_formulation != "rhbi"):
            raise ValueError("real/fake penalty weights require an explicit fixed R1/R2, BCap, or RHBI formulation")
        if self.optimizer_family != "formulation":
            if self.reg_arm not in ("a_r1r2", "b_cap") and self.critic_formulation not in ("bispf", "empf", "rhbi"):
                raise ValueError("plain optimizers require an explicit fixed R1/R2 or BCap reg_arm")
            if (self.d_guard_ratio != 0 or self.reg_anchor_weight != 0
                    or self.latent_damping_max_rate != 0 or self.direct_particle_gain):
                raise ValueError("plain optimizers require disabled guard, anchor, latent damping and direct particle gain")
            if self.continuous_policy is not None:
                raise ValueError("plain optimizers do not implement a continuous update policy")
        if self.optimizer_family == "ada_nsgda" and (
                self.betas[0] != 0 or (self.d_betas is not None and self.d_betas[0] != 0)
                or (self.prior_betas is not None and self.prior_betas[0] != 0)):
            raise ValueError("ada_nsgda requires beta1=0 for networks and prior")
        if self.beta2_end is not None and self.optimizer_family != "adam":
            raise ValueError("scheduled beta2 requires optimizer_family='adam'")
        if self.reg_coeff_end is not None and self.reg_arm not in ("a_r1r2", "b_cap"):
            raise ValueError("scheduled reg_coeff requires a fixed R1/R2 or BCap reg_arm")
        if (self.beta2_end is not None or self.reg_coeff_end is not None) and self.total_steps is None:
            raise ValueError("recipe cosine schedules require a declared total_steps horizon")
        if self.critic_formulation not in ("ka2", "k3p", "bcap", "bispf", "empf", "rhbi"):
            raise ValueError("critic_formulation must be ka2, k3p, bcap, bispf, empf or rhbi")
        if self.critic_formulation == "bcap" and (self.optimizer_family == "formulation" or self.reg_arm != "b_cap"):
            raise ValueError("bcap formulation requires a plain optimizer and reg_arm='b_cap'")
        if self.critic_formulation in ("bispf", "empf"):
            if self.reg_arm is not None:
                raise ValueError(f"{self.critic_formulation} formulation does not accept a legacy reg_arm")
            if self.reg_anchor_weight != 0:
                raise ValueError(f"{self.critic_formulation} formulation uses a zero-mean logit gauge instead of an EMA critic anchor (require reg_anchor_weight=0)")
        if self.critic_formulation == "rhbi":
            if self.reg_arm is not None:
                raise ValueError("rhbi formulation does not accept a legacy reg_arm")
            if self.reg_anchor_weight != 0:
                raise ValueError("rhbi formulation does not use an EMA critic anchor (require reg_anchor_weight=0)")
        for rhbi_name, rhbi_pos in (
            ("rhbi_weight", False),
            ("rhbi_rho", True),
            ("rhbi_tau", False),
            ("rhbi_eps", True),
            ("rhbi_c_max", True),
            ("rhbi_beta_ell", True),
            ("rhbi_critic_weight", False),
            ("rhbi_relativistic_weight", False),
            ("rhbi_lambda_floor", False),
            ("rhbi_gamma_gauge", False),
            ("rhbi_slack_gain", False),
            ("rhbi_chord_weight", False),
            ("rhbi_thermal_tau", False),
            ("rhbi_thermal_shock_mu", False),
            ("rhbi_settle_beta", False),
            ("rhbi_fibre_rel_blend", False),
            ("rhbi_step_damp", False),
            ("rhbi_cond_ot_weight", False),
            ("rhbi_simplex_weight", False),
            ("rhbi_cvx_mu_gain", True),
            ("rhbi_simplex_chart_weight", False),
            ("rhbi_tail_spill_gain", False),
        ):
            rhbi_val = getattr(self, rhbi_name)
            if type(rhbi_val) not in (int, float) or not math.isfinite(rhbi_val) or (
                rhbi_val <= 0 if rhbi_pos else rhbi_val < 0
            ):
                raise ValueError(f"{rhbi_name} must be finite and {'positive' if rhbi_pos else 'nonnegative'}")
        if type(self.rhbi_self_coupling) is not str or self.rhbi_self_coupling not in (
            "hollow",
            "feydy",
            "feydy_unique",
            "feydy_discrete",
        ):
            raise ValueError("rhbi_self_coupling must be 'hollow', 'feydy', 'feydy_unique', or 'feydy_discrete'")
        if type(self.rhbi_conditional_fibre) is not str or self.rhbi_conditional_fibre not in ("joint", "slack", "calibrated"):
            raise ValueError("rhbi_conditional_fibre must be 'joint', 'slack', or 'calibrated'")
        if type(self.rhbi_akt_mode) is not str or self.rhbi_akt_mode not in ("sqrt", "weibull"):
            raise ValueError("rhbi_akt_mode must be 'sqrt' or 'weibull'")
        if type(self.rhbi_settle_mode) is not str or self.rhbi_settle_mode not in ("residual", "thermal"):
            raise ValueError("rhbi_settle_mode must be 'residual' or 'thermal'")
        if type(self.rhbi_cond_ot_mode) is not str or self.rhbi_cond_ot_mode not in (
            "kernel_ed",
            "hollow_sinkhorn",
            "imq_cmmd",
        ):
            raise ValueError("rhbi_cond_ot_mode must be 'kernel_ed', 'hollow_sinkhorn' or 'imq_cmmd'")
        if type(self.rhbi_rapidity_screening) is not bool:
            raise ValueError("rhbi_rapidity_screening must be a boolean")
        if type(self.rhbi_scale_log_odds) is not bool:
            raise ValueError("rhbi_scale_log_odds must be a boolean")
        if type(self.rhbi_akt_horizon) is not bool:
            raise ValueError("rhbi_akt_horizon must be a boolean")
        if type(self.rhbi_simplex_macro_settle) is not bool:
            raise ValueError("rhbi_simplex_macro_settle must be a boolean")
        if self.rhbi_beta_ell > 1.0:
            raise ValueError("rhbi_beta_ell must be in (0, 1]")
        if type(self.rhbi_nn_quantile) not in (int, float) or not (0.0 < self.rhbi_nn_quantile < 1.0):
            raise ValueError("rhbi_nn_quantile must be in (0, 1)")
        if self.rhbi_chord_weight > 1.0:
            raise ValueError("rhbi_chord_weight must be in [0, 1]")
        if self.rhbi_fibre_rel_blend > 1.0:
            raise ValueError("rhbi_fibre_rel_blend must be in [0, 1]")
        if self.rhbi_step_damp >= 1.0:
            raise ValueError("rhbi_step_damp must be in [0, 1)")
        if type(self.rhbi_iters) is not int or self.rhbi_iters < 1:
            raise ValueError("rhbi_iters must be a positive integer")
        if type(self.ebi_beta) not in (int, float) or not math.isfinite(self.ebi_beta) or self.ebi_beta <= 0:
            raise ValueError("ebi_beta must be finite and positive")
        if self.reg_arm is not None and self.critic_formulation not in ("bcap", "bispf", "empf", "rhbi"):
            # Keep resolved recipe/Forge provenance truthful about the optimizer
            # family selected by an explicit legacy arm.
            object.__setattr__(self, "critic_formulation", "k3p")
        # Mirrors drift_variants.DRIFT_VARIANTS: recipes.py must also load as a
        # standalone, torch-free file, so it cannot import package modules here.
        if self.continuous_policy not in (None, "dv1", "dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            raise ValueError("unknown continuous_policy")
        if (self.total_steps is None) != (self.continuous_policy is not None):
            raise ValueError("continuous policies require total_steps=None")
        object.__setattr__(self, "betas", tuple(self.betas))
        if self.prior_betas is not None:
            object.__setattr__(self, "prior_betas", tuple(self.prior_betas))
        object.__setattr__(self, "alpha_bar", tuple(self.alpha_bar))
        object.__setattr__(self, "direct_particle_betas", tuple(float(b) for b in self.direct_particle_betas))
        if self.network_lr_horizon_cap is not None and (
                type(self.network_lr_horizon_cap) is not int or self.network_lr_horizon_cap <= 0):
            raise ValueError("network_lr_horizon_cap must be a positive integer or None")
        if type(self.d_guard_min_steps) is not int or self.d_guard_min_steps < 0:
            raise ValueError("d_guard_min_steps must be a nonnegative integer")
        floor = self.network_lr_floor
        if floor is not None and (isinstance(floor, bool) or not math.isfinite(floor) or not 0 <= floor <= 1):
            raise ValueError("network_lr_floor must be None or in [0, 1]")
        if isinstance(self.reg_anchor_min_decay, bool) or not 0 <= self.reg_anchor_min_decay < 1:
            raise ValueError("reg_anchor_min_decay must be in [0, 1)")
        for key in ("d_guard_ratio", "input_noise_std", "output_noise_std"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("latent_damping_max_rate", "output_noise_warmup"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not math.isfinite(self.input_noise_anneal_end) or not 0 < self.input_noise_anneal_end <= 1:
            raise ValueError("input_noise_anneal_end must be in (0, 1]")
        if len(self.direct_particle_betas) != 2 or any(not 0 <= b < 1 for b in self.direct_particle_betas):
            raise ValueError("direct_particle_betas must contain two values in [0, 1)")
        for key in ("z_dim", "num_particles", "batch_size", "total_steps", "reg_every"):
            value = getattr(self, key)
            if key == "total_steps" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.encoder_mode not in ("none", "ae", "categorical", "hard"):
            raise ValueError("encoder_mode must be none, ae, categorical, or hard")
        if self.encoder_mode != "none" and self.prior_kind != "mog":
            raise ValueError("particle encoders require prior_kind='mog'")
        if self.distance_reduction not in ("sum", "mean"):
            raise ValueError("distance_reduction must be sum or mean")
        if self.model not in ("gan", "ddgan"):
            raise ValueError("model must be 'gan' or 'ddgan'")
        if self.prior_kind not in ("particles", "mog"):
            raise ValueError("prior_kind must be 'particles' or 'mog'")
        if not math.isfinite(self.sigma_rel) or self.sigma_rel < 0:
            raise ValueError("sigma_rel must be finite and nonnegative")
        if type(self.standardize) is not bool:
            raise ValueError("standardize must be a boolean")
        if self.prior_kind == "particles" and self.sigma_rel != 0:
            raise ValueError("nonzero sigma_rel requires prior_kind='mog'")
        if self.prior_kind == "mog" and self.num_particles < 2:
            raise ValueError("MoG calibration requires at least two particles")
        if self.conditioning not in ("scalar", "conditional", "ucd"):
            raise ValueError("conditioning must be scalar, conditional, or ucd")
        if self.num_classes is not None and (type(self.num_classes) is not int or self.num_classes < 1):
            raise ValueError("num_classes must be a positive integer or None")
        if self.conditioning != "scalar" and self.num_classes is None:
            raise ValueError("conditional and UCD recipes require num_classes")
        if self.ucd_target not in ("class", "time_class"):
            raise ValueError("ucd_target must be class or time_class")
        if self.ucd_target == "time_class" and (self.model != "ddgan" or self.conditioning != "ucd"):
            raise ValueError("time_class requires DDGAN + UCD")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.lr_anneal_start < 1 or not 0 <= self.lr_floor <= 1:
            raise ValueError("invalid EMA decay or LR schedule")
        for key in ("lr", "d_lr_mult", "prior_lr_mult", "routing_temperature", "observation_sigma"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if (type(self.kinetic_transport_weight) not in (int, float)
                or not math.isfinite(self.kinetic_transport_weight) or self.kinetic_transport_weight < 0):
            raise ValueError("kinetic_transport_weight must be finite and nonnegative")
        if (not isinstance(self.kinetic_transport_local_weight, (int, float))
                or not math.isfinite(self.kinetic_transport_local_weight) or self.kinetic_transport_local_weight < 0):
            raise ValueError("kinetic_transport_local_weight must be finite and nonnegative")
        if type(self.kappa_radial) not in (int, float) or isinstance(self.kappa_radial, bool):
            raise TypeError(f"kappa_radial must be a float, got {type(self.kappa_radial).__name__}")
        if self.kappa_radial <= 0:
            raise ValueError(f"kappa_radial must be positive, got {self.kappa_radial}")
        if type(self.kinetic_transport_projections) is not int or self.kinetic_transport_projections < 1:
            raise ValueError("kinetic_transport_projections must be a positive integer")
        if type(self.amsgrad) is not bool:
            raise ValueError("amsgrad must be a boolean")
        for key in ("critic_r1_real", "critic_payoff_damping"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if self.output_noise_mode not in ("fixed", "learnable", "mobility"):
            raise ValueError("output_noise_mode must be fixed, learnable or mobility")
        if self.output_noise_mode == "learnable" and not self.output_noise_std > 0:
            raise ValueError("learnable output noise needs output_noise_std > 0 (its initial value)")
        if self.output_noise_mode != "fixed" and self.continuous_policy is None:
            raise ValueError("learnable/mobility output noise requires a continuous_policy (no warmup schedule)")
        if self.lr_control not in ("mobility", "stationarity"):
            raise ValueError("lr_control must be mobility or stationarity")
        if self.lr_control == "stationarity" and self.continuous_policy != "dv12":
            raise ValueError("lr_control='stationarity' requires continuous_policy='dv12'")
        if type(self.particle_birth_death) is not bool:
            raise ValueError("particle_birth_death must be a boolean")
        if self.particle_birth_death and self.prior_kind != "particles":
            raise ValueError("particle_birth_death requires prior_kind='particles' (a trainable table)")
        if (isinstance(self.serve_average, bool) or not isinstance(self.serve_average, (int, float))
                or not math.isfinite(self.serve_average) or self.serve_average < 0):
            raise ValueError("serve_average must be a finite number >= 0")
        if self.reopen_signal not in ("data", "none", "optimizer"):
            raise ValueError("reopen_signal must be data, none or optimizer")
        if self.reopen_anchor not in ("hold", "release"):
            raise ValueError("reopen_anchor must be hold or release")
        if self.reopen_anchor != "hold" and self.reopen_signal != "optimizer":
            raise ValueError("reopen_anchor release requires reopen_signal optimizer")
        if self.reopen_guard not in (None, "settled"):
            raise ValueError("reopen_guard must be None or settled")
        if self.reopen_guard is not None and (
                self.reopen_signal != "optimizer" or self.lr_control != "stationarity"):
            raise ValueError("reopen_guard settled requires optimizer re-open and stationarity control")
        if self.row_evidence_null not in ("theory", "scaled"):
            raise ValueError("row_evidence_null must be theory or scaled")
        if self.birth_death_space not in ("data", "critic"):
            raise ValueError("birth_death_space must be data or critic")
        if self.birth_death_feature_scale not in ("none", "std"):
            raise ValueError("birth_death_feature_scale must be none or std")
        if (self.row_policy == "independent" and self.birth_death_feature_scale == "std"
                and not (self.particle_birth_death and self.birth_death_space == "critic")):
            raise ValueError("birth_death_feature_scale='std' needs particle_birth_death with birth_death_space='critic'")
        if type(self.birth_death_isolation) is not bool:
            raise ValueError("birth_death_isolation must be a boolean")
        if (self.row_policy == "independent" and self.birth_death_isolation
                and not (self.particle_birth_death and self.birth_death_space == "critic")):
            raise ValueError("birth_death_isolation needs particle_birth_death with birth_death_space='critic' (a support test on raw samples is not allowed)")
        for _name in ("row_evidence_hot", "row_evidence_exclude", "row_evidence_hold"):
            if type(getattr(self, _name)) is not bool:
                raise ValueError(f"{_name} must be a boolean")
        if self.table_release_rule not in ("any", "both", "never", "anchor"):
            raise ValueError("table_release_rule must be any, both, never or anchor")
        if self.birth_death_backend not in ("knn", "feature_cells", "auto"):
            raise ValueError("birth_death_backend must be knn, feature_cells or auto")
        for key in ("birth_death_cells", "birth_death_metric_rank", "birth_death_chunk"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.birth_death_parent_policy != "real_anchor":
            raise ValueError("birth_death_parent_policy must be real_anchor")
        if self.birth_death_backend != "knn":
            if (not self.particle_birth_death or self.birth_death_space != "critic"
                    or self.birth_death_feature_scale != "std" or self.continuous_policy != "dv12"
                    or self.lr_control != "stationarity"):
                raise ValueError("feature/auto requires standardized critic birth/death with DV12 stationarity")
        elif (self.birth_death_cells, self.birth_death_metric_rank, self.birth_death_chunk) != (64, 8, 256):
            raise ValueError("feature-cell options require feature_cells or auto")
        if type(self.row_evidence_gate) is not bool:
            raise ValueError("row_evidence_gate must be a boolean")
        if self.row_policy not in ("independent", "routed_paired"):
            raise ValueError("row_policy must be independent or routed_paired")
        if self.row_policy == "routed_paired" and (
                self.continuous_policy != "dv12" or self.lr_control != "stationarity"):
            raise ValueError("row_policy='routed_paired' requires DV12 and stationarity control")
        if self.row_evidence_gate and self.lr_control != "stationarity":
            raise ValueError("row_evidence_gate requires lr_control='stationarity'")
        if (self.row_policy == "independent" and (self.particle_birth_death or self.row_evidence_gate)
                and self.conditioning != "scalar"):
            raise ValueError("particle birth/death and row evidence require independently sampled "
                             "unconditional rows; use row_policy='routed_paired' with RoutedRows "
                             "for conditional dense banks")
        if type(self.direct_particle_gain) is not bool:
            raise ValueError("direct_particle_gain must be a boolean")
        for key in ("reg_coeff", "reg_kappa", "reg_anchor_weight", "prior_reg", "ucd_weight",
                    "reconstruction_weight", "conditional_energy_weight",
                    "masked_mixture_weight", "masked_transport_weight", "imputer_component_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if (type(self.joint_reconstruction_weight) not in (int, float)
                or not math.isfinite(self.joint_reconstruction_weight) or self.joint_reconstruction_weight < 0):
            raise ValueError("joint_reconstruction_weight must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.prior_betas is not None and (len(self.prior_betas) != 2 or any(not 0 <= b < 1 for b in self.prior_betas)):
            raise ValueError("prior_betas must contain two values in [0, 1) or be None")
        # JSON commonly writes a zero moment as 0. Adam requires homogeneous
        # floating-point moment values, even when an integer passes our range
        # checks. Normalize only after validating the original values.
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        if self.prior_betas is not None:
            object.__setattr__(self, "prior_betas", tuple(float(b) for b in self.prior_betas))
        # Validate resolved component settings at construction, not later in training.
        self._penalty_options()
        if (len(self.alpha_bar) < 2 or self.alpha_bar[0] != 1
                or any(not math.isfinite(a) or a <= 0 for a in self.alpha_bar)
                or any(b >= a for a, b in zip(self.alpha_bar, self.alpha_bar[1:]))):
            raise ValueError("alpha_bar must start at 1 and decrease strictly, remaining positive")

    def replace(self, **overrides):
        """Return a modified copy; unknown options raise TypeError."""
        return replace(self, **overrides)

    def to_dict(self):
        from .recipe_compat import without_default_additions
        result = without_default_additions(asdict(self))
        if "kinetic_transport_local_weight" in result:
            result["kinetic_transport_local_weight"] = float(result["kinetic_transport_local_weight"])
        if self.loss == "relativistic":
            # Older recipes/checkpoints had this one fixed objective. Preserve
            # their packets while recording every alternative explicitly.
            result.pop("loss")
        if self.optimizer_family == "formulation":
            result.pop("optimizer_family")
        if self.optimizer_momentum == 0:
            result.pop("optimizer_momentum")
        if self.optimizer_convolution == "none":
            result.pop("optimizer_convolution", None)
        if self.optimizer_adam_lr is None:
            result.pop("optimizer_adam_lr")
        if self.eps == 1e-8:
            result.pop("eps")
        if self.beta2_end is None:
            result.pop("beta2_end")
            if self.beta2_anneal_end == 0.2:
                result.pop("beta2_anneal_end")
        if self.reg_coeff_end is None:
            result.pop("reg_coeff_end")
            if self.reg_coeff_anneal_end == 0.2:
                result.pop("reg_coeff_anneal_end")
        if self.critic_formulation == "ka2":
            result.pop("critic_formulation")
        if self.reg_arm is None:
            result.pop("reg_arm")
        if self.reopen_guard is None:
            result.pop("reopen_guard")
        if self.birth_death_backend == "knn":
            # Keep current reference Recipe/config/checkpoint dictionaries.
            for key in ("birth_death_backend", "birth_death_cells", "birth_death_metric_rank",
                        "birth_death_chunk", "birth_death_parent_policy"):
                result.pop(key)
        if self.generator_gradient_response == "none":
            result.pop("generator_gradient_response", None)
        return result

    def make_prior(self, *, manipulators=None, **overrides):
        """Construct the prior; overrides are local to this call.

        ``prior_update`` owns learning/freezing. A conflicting ``learnable``
        override is rejected; change the recipe explicitly instead.

        Learnable tables take the ordinary random draw; use
        ``particlegan.init.deterministic_orthogonal_(prior)`` for R2 points (it
        recalibrates MoG spacing). MoG recipes calibrate spacing on the
        means (potentially expensive). Pass ``sigma=...`` to skip calibration,
        including ``sigma=0`` when restoring a checkpoint.

        ``manipulators`` (a PriorManipulator or sequence) wraps the prior in a
        ``ManipulatedPrior``. Their hyperparameters are module configuration,
        not Recipe fields: they travel in the prior's ``state_dict`` and are
        reported by ``prior.manipulator_spec()``.
        """
        from .particle_prior import MoGParticlePrior, ParticlePrior, calibrate_mog_sigma
        options = {"num_particles": self.num_particles, "z_dim": self.z_dim,
                   "sigma_rel": self.sigma_rel, "standardize": self.standardize,
                   "learnable": self.prior_update == "learned", **overrides}
        if options["learnable"] != (self.prior_update == "learned"):
            raise ValueError("learnable override contradicts Recipe.prior_update")
        kind = options.pop("prior_kind", self.prior_kind)
        if kind == "mog":
            sigma_rel = options.pop("sigma_rel")
            if "sigma" in options:
                prior = MoGParticlePrior(**options)
            else:
                prior = MoGParticlePrior(sigma=0, **options)
                sigma, d0 = calibrate_mog_sigma(prior.means(), sigma_rel)
                prior.set_sigma(sigma)
                prior.d0.copy_(d0)
                prior.sigma_rel = float(sigma_rel)
        elif kind == "particles":
            if options.pop("sigma_rel") != 0:
                raise ValueError("nonzero sigma_rel requires prior_kind='mog'")
            options.pop("standardize")
            normal_sigma = options.pop("sigma", None)
            prior = ParticlePrior(**options)
            if normal_sigma is not None and normal_sigma > 0:
                from .priors import NormalNoiseManipulator
                prior = prior.manipulate(NormalNoiseManipulator(sigma=float(normal_sigma), kappa_radial=float(self.kappa_radial)))
        else:
            raise ValueError("prior_kind must be 'particles' or 'mog'")
        if manipulators is not None:
            if isinstance(manipulators, (list, tuple)):
                prior = prior.manipulate(*manipulators)
            else:
                prior = prior.manipulate(manipulators)
        return prior

    def encode(self, query, prior, *, offset=None, draws=2, generator=None):
        """Route caller-produced queries; returns a ParticleEncoding.

        AE requires raw offsets and returns one deterministic draw. VAE rejects
        offsets because local posterior and prior must match for this recipe.

        Encoding routes queries over the storage table rows (codes are base
        locations plus prior-matching noise). A ``ManipulatedPrior`` whose
        manipulators transform sampled codes is rejected: those transforms
        would not be applied to encoded codes, so they would differ from the
        codes the decoder sees when sampling.
        """
        from .autoencoder import particle_ae, particle_vae
        from .priors import unwrap_prior
        if getattr(prior, "transforms_codes", False):
            raise ValueError("encode does not apply prior manipulator transforms; "
                             "use a prior without code-transforming manipulators")
        base = unwrap_prior(prior)
        options = dict(temperature=self.routing_temperature,
                       distance_reduction=self.distance_reduction)
        if self.encoder_mode == "ae":
            if offset is None:
                raise ValueError("AE encoding requires offset")
            return particle_ae(query, offset, base, uniformity_weight=self.ucd_weight, **options)
        if self.encoder_mode in ("categorical", "hard"):
            if offset is not None:
                raise ValueError("prior-matching VAE does not accept an offset")
            return particle_vae(query, base, draws=draws, generator=generator,
                                hard=self.encoder_mode == "hard", **options)
        raise ValueError("this recipe has no encoder")

    def make_loss(self):
        """The selected objective: ``d_loss(real, fake)``, ``g_loss(fake, real=None)``."""
        if self.effective_critic_formulation == "rhbi" or self.rhbi_weight > 0 or self.rhbi_critic_weight > 0:
            from .rhbi import RHBIGANLoss
            return RHBIGANLoss(
                self.loss,
                labels=self.loss_labels,
                weight=self.rhbi_weight,
                rho=self.rhbi_rho,
                tau=self.rhbi_tau,
                eps=self.rhbi_eps,
                iters=self.rhbi_iters,
                c_max=self.rhbi_c_max,
                beta_ell=self.rhbi_beta_ell,
                nn_quantile=self.rhbi_nn_quantile,
                critic_weight=self.rhbi_critic_weight,
                self_coupling=self.rhbi_self_coupling,
                relativistic_weight=self.rhbi_relativistic_weight,
                rapidity_screening=self.rhbi_rapidity_screening,
                scale_log_odds=self.rhbi_scale_log_odds,
                conditional_fibre=self.rhbi_conditional_fibre,
                akt_horizon=self.rhbi_akt_horizon,
                akt_mode=self.rhbi_akt_mode,
                slack_gain=self.rhbi_slack_gain,
                thermal_tau=self.rhbi_thermal_tau,
                thermal_shock_mu=self.rhbi_thermal_shock_mu,
                fibre_rel_blend=self.rhbi_fibre_rel_blend,
                cond_ot_weight=self.rhbi_cond_ot_weight,
                cond_ot_mode=self.rhbi_cond_ot_mode,
                cvx_mu_gain=self.rhbi_cvx_mu_gain,
                tail_spill_gain=self.rhbi_tail_spill_gain,
                simplex_macro_settle=self.rhbi_simplex_macro_settle,
            )
        if self.effective_critic_formulation == "bispf" and self.loss == "relativistic":
            from .bispf import SinkhornRelativisticLoss
            return SinkhornRelativisticLoss(labels=self.loss_labels, gates=self.bispf_gates())
        if self.effective_critic_formulation == "empf" and self.loss == "relativistic":
            from .empf import EMPFSinkhornRelativisticLoss
            return EMPFSinkhornRelativisticLoss(labels=self.loss_labels, loss=self.loss)
        from .gan_loss import GANLoss
        return GANLoss(self.loss, labels=self.loss_labels)

    def joint_reconstruction_loss(self, reconstruction, target, *, categorical_dim=None):
        """Weighted paired supervision for an explicitly bound joint inverse host."""
        if self.joint_reconstruction_weight == 0:
            return reconstruction[:0].sum()
        from .joint_reconstruction import joint_reconstruction_loss
        return self.joint_reconstruction_weight * joint_reconstruction_loss(
            reconstruction, target, categorical_dim=categorical_dim)

    def kinetic_transport_loss(self, fake, real):
        """Weighted empirical transport on caller-owned G-phase output batches.

        Enabling this objective requires a host consumer. It draws no samples,
        detaches real targets and retains gradients through fake outputs. Hosts
        with zero weight retain their original objective without calling it.
        """
        from .kinetic_transport import kinetic_transport_loss
        return self.kinetic_transport_weight * kinetic_transport_loss(
            fake, real, projections=self.kinetic_transport_projections)

    def kinetic_transport_local_loss(self, fake, real):
        """Weighted local-v2 residual on the same caller-owned output batches.

        Anchors, bandwidths and normalizers come from detached real targets.
        This auxiliary term does not become a protected projection objective.
        """
        weight = float(self.kinetic_transport_local_weight)
        if weight <= 0:
            return fake.new_zeros(())
        from .kinetic_transport import kinetic_transport_local_loss
        return weight * kinetic_transport_local_loss(fake, real)

    def make_critic_penalty(self, optimizer, *, output=None, collect_stats=False, **penalty_overrides):
        """The critic gradient penalty paired with one critic optimizer.

        ``optimizer`` comes from ``make_optimizers`` or ``make_critic_optimizer``;
        the penalty reads the state it needs (EMA critic, LR record, step count)
        from it. Call it like a loss: ``penalty(D, real, fake, *condition,
        **condition_kwargs)`` returns a scalar tensor; the conditioning goes to
        the critic and its EMA. ``output`` selects the logits from the critic's
        output (default: first element of a tuple/list); ``collect_stats``
        fills ``penalty.last_stats``. ``penalty_overrides`` replace the
        recipe's ``coeff``, ``kappa``, ``lazy_k``,
        ``anchor_weight`` or ``r1_real`` for this penalty.

        BISPF also accepts ``input_domain="continuous"`` for native continuous
        vectors of any width, or ``"categorical"`` for categorical matrices.
        The default ``"auto"`` preserves the historical shape/width heuristic.
        This declaration selects the existing domain law; it does not reshape
        data. Use ``GANTrainer(..., penalty_options={"input_domain": ...})`` to
        bind it to the trainer's checked checkpoint configuration.
        """
        if self.effective_critic_formulation == "bispf":
            from .bispf import BISPFCriticPenalty as CriticPenalty
        elif self.effective_critic_formulation == "empf":
            from .empf import EMPFCriticPenalty as CriticPenalty
        elif self.effective_critic_formulation == "rhbi":
            from .rhbi import RHBICriticPenalty as CriticPenalty
        elif self.effective_critic_formulation in ("k3p", "bcap"):
            from .k3p import CriticPenalty
        else:
            from .ka2 import CriticPenalty
        return CriticPenalty(self, optimizer, output=output, collect_stats=collect_stats,
                             **penalty_overrides)

    def _penalty_options(self, **overrides):
        """Resolved kernel settings for ``make_critic_penalty``."""
        if self.critic_formulation not in ("ka2", "k3p", "bcap", "bispf", "empf", "rhbi"):
            raise ValueError("critic_formulation must be ka2, k3p, bcap, bispf, empf or rhbi")
        if self.effective_critic_formulation == "rhbi":
            if not self.critic_r1_real:
                raise ValueError("critic_r1_real=False requires the KA2 formulation")
            options = {
                "coeff": self.reg_coeff,
                "kappa": self.reg_kappa,
                "lazy_k": self.reg_every,
                "anchor_weight": self.reg_anchor_weight,
                "ebi_beta": self.ebi_beta,
                "real_weight": self.reg_real_weight,
                "fake_weight": self.reg_fake_weight,
                "lambda_floor": self.rhbi_lambda_floor,
                "gamma_gauge": self.rhbi_gamma_gauge,
                "chord_weight": self.rhbi_chord_weight,
                "akt_horizon": self.rhbi_akt_horizon,
                "akt_mode": self.rhbi_akt_mode,
                "settle_beta": self.rhbi_settle_beta,
                "settle_mode": self.rhbi_settle_mode,
                **overrides,
            }
            from .rhbi import _RHBIPenaltyKernel
            _RHBIPenaltyKernel(**options)
            return options
        if self.effective_critic_formulation == "bispf":
            if not self.critic_r1_real:
                raise ValueError("critic_r1_real=False requires the KA2 formulation")
            options = {"coeff": self.reg_coeff, "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                       "anchor_weight": self.reg_anchor_weight, "ebi_beta": self.ebi_beta, **overrides}
            from .bispf import _BISPFPenaltyKernel
            _BISPFPenaltyKernel(**options)
            return options
        if self.effective_critic_formulation == "empf":
            if not self.critic_r1_real:
                raise ValueError("critic_r1_real=False requires the KA2 formulation")
            options = {"coeff": self.reg_coeff, "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                       "anchor_weight": self.reg_anchor_weight, "ebi_beta": self.ebi_beta, **overrides}
            from .empf import _EMPFPenaltyKernel
            _EMPFPenaltyKernel(**options)
            return options
        if self.effective_critic_formulation in ("k3p", "bcap"):
            if not self.critic_r1_real:
                raise ValueError("critic_r1_real=False requires the KA2 formulation")
            floor = self.resolved_network_lr_floor
            options = {"arm": self.reg_arm or "k3p", "coeff": self.reg_coeff,
                       "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                       "anchor_weight": self.reg_anchor_weight,
                       "lr_floor": floor if floor < .5 else 0., **overrides}
            if self.reg_arm in ("a_r1r2", "b_cap"):
                options = {"real_weight": self.reg_real_weight, "fake_weight": self.reg_fake_weight, **options}
            from .grad_regularizers import GradientPenalty
            GradientPenalty(**options)
            return options
        options = {"coeff": self.reg_coeff, "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                   "anchor_weight": self.reg_anchor_weight, "r1_real": self.critic_r1_real, **overrides}
        unknown = set(options) - {"coeff", "kappa", "lazy_k", "anchor_weight", "r1_real"}
        if unknown:
            raise TypeError(f"unknown critic penalty options: {sorted(unknown)}")
        from .ka2 import KA2GradientPenalty
        KA2GradientPenalty(**options)  # validate
        return options

    def make_critic_optimizer(self, critic, *, ema_critic=None, ka2_trust_mode="compound", **adam_kwargs):
        """Optimizer over ``critic``'s trainable parameters whose ``step()`` does the
        recipe's critic-side work (KA2: spike guard, moment surprise, adaptive
        EMA-critic update and guarded reseed).

        ``ema_critic`` is a caller-allocated copy of ``critic`` (e.g.
        ``copy.deepcopy(critic)``) that becomes the EMA; the critic penalty
        requires it unless ``reg_anchor_weight == 0``. Checkpoint with ``optimizer.state_dict()``: it holds the
        EMA critic and all counters. ``adam_kwargs`` override the recipe's
        ``lr * d_lr_mult``, ``betas`` and ``amsgrad`` or add options such as ``fused``.
        ``optimizer_family='adam'`` returns the declared Adam law with disabled intervention
        metadata and an observation-only checkpointed critic step counter.
        Normalized families use the same fixed penalty and step observer;
        ``lr * d_lr_mult`` is their scheduled critic step size.
        ``ka2_trust_mode`` declares how KA2's ``game_trust`` damps the anchor
        rate (``KA2StepRecord``): the default ``"compound"`` is the legacy,
        qualified law; ``"scale"`` is a declared candidate. Other critic
        formulations reject a non-default value.
        """
        if ka2_trust_mode not in ("compound", "scale"):
            raise ValueError("ka2_trust_mode must be compound or scale")
        if ka2_trust_mode != "compound" and (
                self.optimizer_family != "formulation" or self.effective_critic_formulation != "ka2"):
            raise ValueError("ka2_trust_mode applies only to the KA2 critic formulation")
        options = {"lr": self.lr * self.d_lr_mult, "betas": self.d_betas or self.betas,
                   "amsgrad": self.amsgrad, "eps": self.eps if self.d_eps is None else self.d_eps,
                   **adam_kwargs}
        if self.optimizer_family == "adam":
            from .recipe_schedules import make_plain_adam
            return make_plain_adam(self, [p for p in critic.parameters() if p.requires_grad], critic=critic, **options)
        if self.optimizer_family != "formulation":
            from .optim.dualnorm import convolution_parameter_groups, make_normalized_optimizer
            if self.optimizer_family == "particle_rownorm_only" and self.optimizer_adam_lr is not None:
                options["lr"] = self.optimizer_adam_lr * self.d_lr_mult
            params = [{"params": [p for p in critic.parameters() if p.requires_grad], "role": "critic"}]
            if self.optimizer_convolution == "per_offset":
                params = convolution_parameter_groups(params, [critic], family=self.optimizer_family,
                                                      convolution_rule=self.optimizer_convolution_rule)
            return make_normalized_optimizer(self, params, critic=critic, **options)
        if self.effective_critic_formulation == "bispf":
            from .bispf import BISPFCriticAdam
            return BISPFCriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                                   guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)
        if self.effective_critic_formulation == "empf":
            from .empf import EMPFCriticAdam
            return EMPFCriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                                  ema_critic=ema_critic,
                                  guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)
        if self.effective_critic_formulation == "rhbi":
            from .rhbi import RHBICriticAdam
            return RHBICriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                                  guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)
        if self.effective_critic_formulation == "k3p":
            from .k3p import K3PCriticAdam
            return K3PCriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                                 ema_critic=ema_critic, anchor_decay=.999,
                                 guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)
        from .ka2 import KA2CriticAdam
        return KA2CriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                             ema_critic=ema_critic, anchor_min_decay=self.reg_anchor_min_decay,
                             guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps,
                             **({} if ka2_trust_mode == "compound" else {"trust_mode": ka2_trust_mode}), **options)

    def make_generator_optimizer(self, params, *, modules=None, latent_table=None, direct_particles=None, **adam_kwargs):
        """Optimizer over ``params`` (module, tensors or param groups) whose ``step()`` does the
        recipe's generator-side work (A2 damping of the sparse
        ``latent_table``, e.g. ``prior.z`` alone in its group with beta1 == 0,
        and the direct-particle response for the param group ``direct_particles``).

        With neither, ``step()`` is exactly ``Adam.step()``. ``adam_kwargs``
        override the recipe's ``lr``, ``betas`` and ``amsgrad`` or add Adam options.
        Plain ``optimizer_family='adam'`` returns the declared Adam law without A2 or
        direct-particle response, including when role annotations are supplied.
        Normalized families accept role-named groups. Their row-normalized
        prior requires ``set_sampled_rows(prior.z, indices)`` before stepping;
        the public trainer/policy records generator-side sample IDs automatically.
        A supplied module binds Conv2d/ConvTranspose2d weights automatically when
        ``optimizer_convolution='per_offset'``. When passing custom parameter
        groups, supply their owning module(s) with ``modules=`` for the same
        binding. Only parameters in ``params`` are optimized; module traversal
        never adds parameters. This path excludes frozen parameters while
        preserving group fields and order, splitting groups only for kernels.
        Schedule every resulting group by its role instead of its position.
        Bare high-rank tensor iterables lack this layout contract and are
        rejected by DualNorm. Loaded values are never initialized or changed
        during construction. Rebuild the optimizer explicitly after unfreezing
        parameters that were excluded at construction.

        With ``constraint_geometry_mode != 'none'``, call
        ``particlegan.optim.constraint_geometry.constraint_geometry_backward``
        before ``step()`` with the host's existing protected scalar losses.
        Protection must cover the joint generator/encoder/prior optimizer.
        The default mode requires no protected-loss hook and retains the
        original optimizer type and checkpoint format.

        ``prior_proximal_strength > 0`` restores only update-owned latent rows
        toward their positions at optimizer construction, after the complete
        base update (including direction blend). Initialize tables first.
        Checkpoints carry that independent reference and the spring settings.
        Direct generated coordinates and frozen tables receive no spring.
        """
        from torch import nn
        if latent_table is not None and self.prior_update == "frozen" and latent_table.requires_grad:
            raise ValueError("trainable latent_table contradicts Recipe.prior_update='frozen'")
        module = params if isinstance(params, nn.Module) else None
        if modules is None:
            module_owners = [] if module is None else [module]
        else:
            module_owners = [modules] if isinstance(modules, nn.Module) else list(modules)
            if any(not isinstance(owner, nn.Module) for owner in module_owners):
                raise TypeError("modules must contain only torch.nn.Module instances")
            if module is not None:
                module_owners.insert(0, module)
        if module is not None:
            params = [p for p in module.parameters() if p.requires_grad]
        if latent_table is not None and self.prior_update == "frozen":
            # Materialize each iterator once; optimizer construction must retain
            # every supplied parameter/group after ownership validation.
            params = list(params)
            owners = []
            for index, value in enumerate(params):
                if isinstance(value, dict):
                    group = {**value, "params": list(value["params"])}
                    params[index] = group
                    owners.extend(group["params"])
                else:
                    owners.append(value)
            if module is not None:
                owners.extend(module.buffers())
            _reject_table_storage_owners(latent_table, owners)
            latent_table = None
        if modules is not None:
            # The explicit module binding is a new construction contract.
            # Keep historical bare-group checkpoint layouts unchanged.
            values = list(params)
            if values and isinstance(values[0], dict):
                params = []
                for value in values:
                    group = {**value, "params": [p for p in value["params"] if p.requires_grad]}
                    if group["params"]:
                        params.append(group)
            else:
                params = [p for p in values if p.requires_grad]
        options = {"lr": self.lr, "betas": self.betas, "amsgrad": self.amsgrad, "eps": self.eps, **adam_kwargs}
        if self.optimizer_family == "adam":
            from .recipe_schedules import make_plain_adam
            return make_plain_adam(self, params, **options)
        if self.optimizer_family != "formulation":
            from .optim.dualnorm import convolution_parameter_groups, make_normalized_optimizer
            params = list(params)
            if params and not isinstance(params[0], dict):
                params = [{"params": params}]
            groups = []
            direct_ids = set() if direct_particles is None else {id(p) for p in direct_particles}
            for original in params:
                group = dict(original)
                group["params"] = list(group["params"])
                group.setdefault("role", group.get("forge_role", "generator"))
                if latent_table is not None and any(p is latent_table for p in group["params"]):
                    group["role"] = "prior"
                if direct_ids and any(id(p) in direct_ids for p in group["params"]):
                    # Direct generated coordinates are the generator player;
                    # this fixture has no sampled latent table.
                    group["role"] = "generator"
                is_prior = group["role"] in ("prior", "table")
                uses_adam = (self.optimizer_family == "dualnorm_D_only"
                             or (self.optimizer_family == "particle_rownorm_only" and not is_prior))
                if uses_adam and self.optimizer_adam_lr is not None:
                    group["lr"] = self.optimizer_adam_lr * (self.prior_lr_mult if is_prior else 1.)
                groups.append(group)
            if module_owners and self.optimizer_convolution == "per_offset":
                groups = convolution_parameter_groups(groups, module_owners, family=self.optimizer_family,
                                                      convolution_rule=self.optimizer_convolution_rule)
            return make_normalized_optimizer(self, groups, **options)
        from .k3p import K3PGeneratorAdam
        return K3PGeneratorAdam(params, latent_table=latent_table, direct_particles=direct_particles,
                                latent_max_rate=self.latent_damping_max_rate,
                                direct_betas=self.direct_particle_betas,
                                direct_gain=self.direct_particle_gain,
                                saddle_escape=(self.continuous_policy == "dv12"), **options)

    def bispf_gates(self):
        """Resolved ``BISPFGates`` from ``bispf_shape_gates`` and ``bispf_gate_overrides``."""
        from .bispf import BISPFGates
        return BISPFGates.from_overrides(self.bispf_shape_gates, self.bispf_gate_overrides)

    @property
    def effective_critic_formulation(self):
        """Explicit legacy arms retain their K3P optimizer and penalty family."""
        if self.critic_formulation in ("bcap", "bispf", "empf", "rhbi"):
            return self.critic_formulation
        return "k3p" if self.reg_arm is not None else self.critic_formulation

    @property
    def resolved_network_lr_floor(self):
        """G/D LR floor: ``network_lr_floor`` or ``lr_floor``."""
        return self.lr_floor if self.network_lr_floor is None else self.network_lr_floor

    def make_prior_regularizer(self, *, prior=None, reference=None, **overrides):
        if prior is not None and reference is not None and prior is not reference:
            raise ValueError("prior and reference must identify the same initialized prior")
        if prior is None:
            prior = reference
        if reference is None:
            reference = prior
        # Regularizer references read storage buffers (initial moments,
        # Coulomb reference) that live on the unwrapped table.
        from .priors import unwrap_prior
        prior = unwrap_prior(prior)
        reference = (tuple(unwrap_prior(item) for item in reference)
                     if isinstance(reference, (list, tuple)) else unwrap_prior(reference))
        if self.prior_regularizer == "sbnp":
            from .bispf import SBNPPriorRegularizer
            options = {"weight": self.prior_reg, "target_std": self.prior_reg_target_std,
                       "eps": self.prior_reg_eps, "gates": self.bispf_gates(), **overrides}
            if self.prior_update == "frozen":
                options["weight"] = 0.
            return SBNPPriorRegularizer(reference, **options)
        if self.prior_regularizer == "empf":
            from .empf import EMPFPriorRegularizer
            options = {"weight": self.prior_reg, "target_std": self.prior_reg_target_std,
                       "eps": self.prior_reg_eps, **overrides}
            if prior is not None and hasattr(prior, "_empf_record"):
                options.setdefault("empf_record", prior._empf_record)
            if self.prior_update == "frozen":
                options["weight"] = 0.
            return EMPFPriorRegularizer(reference, **options)
        if self.prior_regularizer == "coulomb":
            from .coulomb_prior import CoulombPriorRegularizer, coulomb_reference
            options = {"weight": self.prior_reg, "softening": self.prior_coulomb_softening,
                       "confinement": self.prior_coulomb_confinement,
                       "neighbors": self.prior_coulomb_neighbors, **overrides}
            if self.prior_update == "frozen":
                options["weight"] = 0.
            if prior is not None and options["weight"] > 0:
                options["center"], options["radius"] = coulomb_reference(prior)
            return CoulombPriorRegularizer(**options)
        from .vicreg_loss import ParticleRegularizer
        if self.prior_regularizer == "sigreg":
            from .sigreg import SIGReg
            options = {"weight": self.prior_reg, "projections": self.prior_sigreg_projections,
                       "knots": self.prior_sigreg_knots, **overrides}
            if self.prior_update == "frozen":
                options["weight"] = 0.
            return SIGReg(**options)
        options = {"weight": self.prior_reg, "target_std": self.prior_reg_target_std,
                   "eps": self.prior_reg_eps, **overrides}
        if self.prior_regularizer == "moment_tether":
            unknown = set(options) - {"weight", "target_std", "eps"}
            if unknown:
                raise TypeError(f"unsupported moment_tether options: {sorted(unknown)}")
            if self.prior_update == "learned" and self.prior_reg > 0:
                from .moment_tether import MomentTetherRegularizer
                return MomentTetherRegularizer(reference, weight=options["weight"], eps=options["eps"])
            options["weight"] = 0.
        if self.prior_regularizer == "none" or self.prior_update == "frozen":
            options["weight"] = 0.
        if self.prior_regularizer == "initial_mmd" and options["weight"] > 0:
            from .initial_mmd import InitialMMDRegularizer
            if prior is None:
                raise ValueError("initial_mmd requires prior= after public initialization")
            if set(overrides) - {"weight"}:
                raise ValueError("initial_mmd has fixed panel/kernel scales; only weight can be overridden")
            return InitialMMDRegularizer.from_priors(prior, weight=options["weight"])
        return ParticleRegularizer(**options)

    def prior_regularization(self, rows, *, regularizer=None, prior=None, reference=None):
        """Return the complete weighted latent-prior objective, once.

        Callers select rows using their declared sampling law. Frozen rows and
        disabled coefficients do no variance/covariance work or optimizer work.
        The factory is the extension point for future regularizer families.
        """
        if self.prior_update == "frozen" or not rows.requires_grad:
            return rows.new_zeros(())
        # Empty rows preserve a zero gradient without reading nonfinite values
        # or doing any variance/covariance work for disabled penalties.
        result = rows[:0].sum()
        if self.prior_reg > 0:
            result = (self.make_prior_regularizer(prior=prior, reference=reference) if regularizer is None else regularizer)(rows)
        if self.prior_l2 > 0:
            result = result + self.prior_l2 * rows.square().mean()
        return result

    def prior_regularizer_checkpoint(self, regularizer):
        """Optional state for active stochastic families; old checkpoints stay flat."""
        if (self.prior_regularizer == "sigreg" and self.prior_update == "learned"
                and self.prior_reg > 0 and regularizer is not None):
            state = regularizer.state_dict()
            if state:
                return {"prior_regularizer": state}
        return {}

    def restore_prior_regularizer(self, regularizer, state, *, validate_only=False):
        """Validate an optional regularizer checkpoint before restoring its draws."""
        from copy import deepcopy
        expected = self.prior_regularizer_checkpoint(regularizer)
        if not expected:
            if state is not None:
                raise ValueError("inactive prior regularizer cannot restore active state")
            return
        if not isinstance(state, dict):
            raise ValueError("active prior regularizer requires checkpoint state")
        deepcopy(regularizer).load_state_dict(state)
        if not validate_only:
            regularizer.load_state_dict(state)

    def _validate_frozen_prior_ownership(self, prior, *modules, optimizers=()):
        """Reject aliases before freezing, including already-frozen buffers."""
        if self.prior_update != "frozen":
            return
        from .priors import unwrap_prior
        base = unwrap_prior(prior)
        if not hasattr(base, "z"):
            return
        for module in modules:
            if module is not None:
                _reject_table_storage_owners(base.z, (*module.parameters(), *module.buffers()))
        for optimizer in optimizers:
            _reject_table_storage_owners(base.z,
                (p for group in optimizer.param_groups for p in group["params"]))

    def apply_prior_policy(self, prior):
        """Bind the prior policy after initialization, before making optimizers.

        Initializing a trainable table first lets learned/frozen comparisons use
        identical deterministic locations and consume identical init streams.
        Freezing registers locations as a buffer, keeping sampling/state keys.
        A frozen prior is frozen as a whole: parameters owned by its
        manipulators (e.g. a learnable simplex router) are frozen too.
        """
        from .priors import ManipulatedPrior, unwrap_prior
        base = unwrap_prior(prior)
        if self.prior_update == "frozen" and isinstance(prior, ManipulatedPrior):
            prior.manipulators.requires_grad_(False)
        if not hasattr(base, "z"):
            return prior  # Gaussian priors have no trainable location table.
        if self.prior_update == "learned":
            if not base.z.requires_grad:
                raise ValueError("frozen prior contradicts Recipe.prior_update='learned'")
            if self.prior_regularizer in ("moment_tether", "sbnp", "empf") and self.prior_reg > 0:
                from .moment_tether import capture_initial_moments
                capture_initial_moments(base)
        elif base.z.requires_grad:
            rows = base.z.detach()
            del base.z
            base.register_buffer("z", rows)
        if self.prior_code_law != "table":
            if self.prior_update != "frozen":
                raise ValueError("prior_code_law='fresh_gaussian' requires prior_update='frozen'")
            base.set_code_law(self.prior_code_law)
        elif getattr(base, "code_law", "table") != "table":
            raise ValueError("prior code law contradicts Recipe.prior_code_law='table'")
        if self.prior_regularizer == "coulomb" and self.prior_update == "learned" and self.prior_reg > 0:
            from .coulomb_prior import capture_coulomb_reference
            capture_coulomb_reference(base)
        return prior

    def make_optimizers(self, generator, discriminator, prior=None, *, encoder=None, ema_critic=None,
                        require_latent_damping=False, separate_encoder_groups=False,
                        ka2_trust_mode="compound", **adam_kwargs):
        """Return ``(opt_g, opt_d)`` selected by ``optimizer_family``.

        ``ka2_trust_mode`` reaches only the critic optimizer
        (``make_critic_optimizer``); the default is the legacy KA2 law.

        ``opt_g`` covers G + optional E + prior (``make_generator_optimizer``;
        a learnable ``ParticlePrior`` table gets A2 damping) and ``opt_d`` the
        critic (``make_critic_optimizer``, with the caller-allocated
        ``ema_critic``). Use them like any Adam: ``zero_grad``/``step``,
        ``state_dict``/``load_state_dict`` (which carry all regularization
        state), LR schedulers. Move modules to their desired device before
        calling. Frozen parameters are excluded, and a Gaussian/frozen prior
        adds no optimizer group. Additional Adam options, such as ``fused`` or
        ``eps``, apply to both optimizers. Set learning rates, betas and
        ``amsgrad`` on the recipe (``amsgrad`` reaches every group: G/E,
        prior and critic). The generator optimizer's groups are ``[generator/encoder,
        prior]`` (either may be absent); scale the prior group by the prior
        multiplier of ``learning_rate_scales`` and everything else by the
        network one.

        ``separate_encoder_groups=True`` gives G and E distinct homogeneous
        groups, with the same network rate and optimizer settings. Use this
        when binding both modules to ``UpdatePolicy``. The default preserves
        develop's unlabelled native G/E groups and checkpoint layout. Normalized
        families already label the two roles.

        Parameters are used as supplied; initialize fresh networks first (e.g.
        ``particlegan.init.deterministic_orthogonal_``).

        ``opt_g.prior_mechanisms`` records prior capabilities and resolved A2
        status. Nonstandardized MoG locations are row-local and support A2;
        standardized reads do not. ``require_latent_damping=True`` rejects an
        unavailable or disabled hook before constructing optimizers. The default
        preserves historical component callers that did not apply A2 to MoG.
        Fixed-penalty normalized families share the caller's training loop and
        schedule; G and E have one global normalization norm, with the prior
        separate. Isolation arms use ``optimizer_adam_lr`` for native Adam
        groups and ``lr`` for normalized groups (the same role multipliers).
        """
        from .capabilities import prior_mechanisms
        if type(separate_encoder_groups) is not bool:
            raise TypeError("separate_encoder_groups must be a boolean")
        if prior is not None:
            self._validate_frozen_prior_ownership(prior, generator, discriminator, encoder, ema_critic)
            self.apply_prior_policy(prior)
            from .priors import unwrap_prior
            base_prior = unwrap_prior(prior)
            base_params = [p for p in base_prior.parameters() if p.requires_grad]
            base_ids = {id(p) for p in base_params}
            router_params = [p for p in prior.parameters() if p.requires_grad and id(p) not in base_ids]
            prior_params = base_params + router_params
            prior_ids = {id(p) for p in prior_params}
        else:
            base_prior = None
            base_params = []
            router_params = []
            prior_params = []
            prior_ids = set()
        g_params = []
        seen = set(prior_ids)
        normalized = self.optimizer_family not in ("adam", "formulation")
        groups = []
        for role, module in (("generator", generator), ("encoder", encoder)):
            role_params = []
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad and id(p) not in seen:
                        seen.add(id(p))
                        g_params.append(p)
                        role_params.append(p)
            if role_params:
                if normalized or separate_encoder_groups:
                    role_groups = [{"params": role_params, "lr": self.lr, "role": role}]
                    if normalized and self.optimizer_convolution == "per_offset":
                        from .optim.dualnorm import convolution_parameter_groups
                        role_groups = convolution_parameter_groups(role_groups, [generator, encoder],
                                                                   family=self.optimizer_family,
                                                                   convolution_rule=self.optimizer_convolution_rule)
                    groups.extend(role_groups)
                else:
                    groups.append({"params": role_params, "lr": self.lr})
        if base_params:
            groups.append({"params": base_params, "lr": self.lr * self.prior_lr_mult,
                           "betas": self.prior_betas if self.prior_betas is not None else self.betas,
                           **({"role": "prior"} if normalized or separate_encoder_groups else {}),
                           **({"eps": self.prior_eps} if self.prior_eps is not None else {})})
        if router_params:
            groups.append({"params": router_params, "lr": self.lr,
                           "betas": self.betas,
                           **({"role": "router"} if normalized or separate_encoder_groups else {})})
        mechanisms = prior_mechanisms(prior,
            latent_damping_max_rate=self.latent_damping_max_rate if self.prior_update == "learned" else 0.,
            prior_beta1=(self.prior_betas or self.betas)[0])
        if require_latent_damping and not mechanisms["a2"]["enabled"]:
            raise ValueError("required A2 latent damping unavailable: " + mechanisms["a2"]["reason"])
        # Plain and nonstandardized MoG reads have the same row-local gradient
        # ownership. Standardized MoG keeps the component API's historic policy
        # (no A2), now explicit in the optimizer receipt and optionally required.
        latent_table = base_prior.z if (base_prior is not None and hasattr(base_prior, "z") and mechanisms["prior"]["a2_eligible"]) else None
        opt_g = self.make_generator_optimizer(groups, latent_table=latent_table, **adam_kwargs)
        opt_g.prior_mechanisms = mechanisms
        # Forwarded only when declared: subclass overrides of make_critic_optimizer keep their signature.
        if ka2_trust_mode != "compound":
            adam_kwargs = {**adam_kwargs, "ka2_trust_mode": ka2_trust_mode}
        opt_d = self.make_critic_optimizer(discriminator, ema_critic=ema_critic, **adam_kwargs)
        if self.effective_critic_formulation == "empf" and hasattr(opt_d, "record"):
            opt_g._empf_record = opt_d.record
            opt_d._empf_record = opt_d.record
            if base_prior is not None:
                base_prior._empf_record = opt_d.record
        if (
            self.effective_critic_formulation == "rhbi"
            or self.rhbi_weight > 0
            or self.rhbi_critic_weight > 0
        ) and hasattr(opt_d, "record"):
            opt_g._rhbi_record = opt_d.record
            opt_d._rhbi_record = opt_d.record
        return opt_g, opt_d



def get_recipe(name="gan", *, trainer=None, **overrides):
    """Build and validate the selected recipe; see ``get_recipe_fields``.

    Restore resolved checkpoints with ``Recipe.from_dict(saved_fields)`` instead of
    resolving the current default again.
    """
    return Recipe(**get_recipe_fields(name, trainer=trainer, **overrides))


def get_recipe_fields(name="gan", *, trainer=None, **overrides):
    """Select a model family or a policy preset with explicit overrides.

    Model families share the trainer preset selected by ``DEFAULT_RECIPE``.
    ``trainer="ka2"`` pins a model family to an explicit trainer preset.
    Explicit trainer presets remain independent of that selector.
    ``"e22"`` selects the
    schedule-free DV12/stationarity policy, row evidence, critic-feature
    birth/death, learned output noise and served averaging. Its default
    dimensions are the native 100-Gaussian task's 20,000 particles, latent
    dimension 2 and batch size 2,048. Set ``num_particles``, ``z_dim``,
    ``batch_size`` and ``output_noise_std`` explicitly for another task.
    ``"e22_routed"`` selects the conditional dense-bank ``routed_paired``
    adaptation. It retains E22's controls and requires an explicit RoutedRows
    binding, paired observations and separate guard contexts. Its evidence
    and restructuring law differ from the independent-row formulation.
    ``"atlas"`` adds automatic feature-cell selection (128 cells) and the
    settled optimizer-reopen guard to E22. ``"ka2"`` retains the KA2 recipe;
    ``"k3p"`` explicitly selects the earlier critic formulation.
    ``"bcap"`` selects zero-momentum dualnorm with G/E step .012, D step
    .018 and sampled-prior row step .03, non-saturating loss, smoothing .001,
    per-offset convolution updates, fixed real/fake input-gradient caps,
    direction blending for conflicting task-owned objectives and constant
    rates. This is the selected research recipe, not a calibrated robustness
    claim. Set ``constraint_geometry_mode="none"`` for the earlier DualNorm
    recipe. ``"bcap_adam"`` retains the earlier native-Adam
    preset. Both disable the guard, anchor, latent damping, extra
    regularization, EMA serving and additive training noise.
    ``"empf"`` is the selected EMPF research default (``DEFAULT_RECIPE``):
    clock-free online ``e-means`` sufficient-statistics prototype bank,
    mass-weighted prototype-to-fake Sinkhorn OT, state-driven settling
    Born-Infeld critic penalty (``ebi_beta=6.0``), degree-of-freedom-aware EMPF
    prior regularization (``prior_reg=0.01``), zero-momentum DualNorm with
    quintic batched convolution updates, and BF16/channels-last training for
    supported CUDA convolutional networks. See
    ``reports/forge/technique-inventory.md`` for its current ordinary CUDA
    readout; selection does not add accepted calibration, Tier 3 confirmation,
    or natural-image quality credit.
    ``"bispf_optimized"`` retains the explicit optimized BISPF preset:
    zero-momentum DualNorm, non-saturating loss, the Born-Infeld chord penalty,
    SBNP prior regularization, batched convolution updates and
    BF16/channels-last training for supported CUDA convolutional networks. Its
    recorded evidence retains 18/61 fixed-suite passes (81/115 counted view
    requirements).
    ``"bispf"`` retains the original Born-Infeld Sinkhorn Particle Flow formulation
    (scale-free Sinkhorn relativistic pairing, Elastic Born-Infeld chord
    regularization, zero-mean logit gauge with no EMA critic copy, SBNP
    prior regularization, and constant learning rates).
    No research configuration file is read at runtime.

    Return every declarative field without constructing runtime components.
    This function can be loaded directly from ``recipes.py`` by configuration
    tools that do not import Torch. Preset names and field names are checked;
    call ``get_recipe`` for value validation before training. Returned fields
    are independent of the historical restoration defaults.

    Use ``Recipe.from_dict(saved_fields)`` for resolved checkpoints and
    ``recipe.replace(name=...)`` for custom report labels.
    """
    models = {
        "gan": {},
        "mog": dict(prior_kind="mog", sigma_rel=.025, num_particles=400),
        "ddgan": dict(model="ddgan", conditioning="ucd", num_classes=4),
        "ddgan_mog": dict(model="ddgan", conditioning="ucd", num_classes=4,
                          prior_kind="mog", sigma_rel=.025, num_particles=400),
        "ae_gan": dict(encoder_mode="ae", prior_kind="mog", sigma_rel=.025,
                       num_particles=400, z_dim=2),
        "vae_gan": dict(encoder_mode="hard", prior_kind="mog", sigma_rel=.025,
                        num_particles=400, z_dim=2),
        "ae_ddgan": dict(model="ddgan", encoder_mode="ae", prior_kind="mog",
                         sigma_rel=.025, num_particles=1024, z_dim=64,
                         batch_size=64, routing_temperature=.125,
                         distance_reduction="mean"),
    }
    presets = {
        "ka2": {},
        "k3p": dict(critic_formulation="k3p"),
        "bcap_adam": dict(critic_formulation="bcap", optimizer_family="adam", reg_arm="b_cap",
                     reg_coeff=1., reg_kappa=1., reg_every=1,
                     lr=.00425, d_lr_mult=1., prior_lr_mult=2., betas=(0., .999),
                     lr_floor=1., network_lr_floor=1., network_lr_horizon_cap=None,
                     beta2_end=None, reg_coeff_end=None, d_guard_ratio=0., reg_anchor_weight=0.,
                     latent_damping_max_rate=0., direct_particle_gain=False,
                     prior_reg=0., ema_decay=0., input_noise_std=0., output_noise_std=0.,
                     output_noise_warmup=0.),
        "bispf": dict(critic_formulation="bispf", optimizer_family="formulation",
                      reg_coeff=1., reg_kappa=1., reg_every=1,
                      lr=.00425, d_lr_mult=1., prior_lr_mult=2., betas=(0., .999),
                      lr_schedule="constant", lr_floor=1., network_lr_floor=1.,
                      network_lr_horizon_cap=None, beta2_end=None, reg_coeff_end=None,
                      d_guard_ratio=0., reg_anchor_weight=0.,
                      latent_damping_max_rate=0., direct_particle_gain=False,
                      prior_regularizer="sbnp", prior_reg=0.05,
                      ema_decay=0., input_noise_std=0., output_noise_std=0.,
                      output_noise_warmup=0.),
        "e22": dict(continuous_policy="dv12", total_steps=None,
                    lr_control="stationarity", amsgrad=True, reg_coeff=3.0,
                    input_noise_std=0.0, output_noise_warmup=0.0,
                    output_noise_mode="learnable", particle_birth_death=True,
                    row_evidence_gate=True, table_release_rule="anchor",
                    birth_death_space="critic", birth_death_feature_scale="std",
                    birth_death_isolation=True, row_evidence_null="scaled",
                    serve_average=4.0, reopen_signal="optimizer",
                    reopen_anchor="release"),
    }
    presets["bcap"] = {
        **presets["bcap_adam"], "optimizer_family": "dualnorm",
        "lr": .012, "d_lr_mult": 1.5, "prior_lr_mult": 2.5,
        "optimizer_momentum": 0.,
        "loss": "non_saturating", "optimizer_smoothing": .001,
        "optimizer_convolution": "per_offset",
        "constraint_geometry_mode": "direction_blend",
    }
    # This complete preset matches the selected Forge candidate without reading
    # research files at runtime. Keep the original BISPF/Adam preset and Recipe
    # dataclass defaults intact for explicit callers and saved configurations.
    presets["bispf_optimized"] = {
        **presets["bcap"], "critic_formulation": "bispf", "reg_arm": None,
        "optimizer_convolution_rule": "quintic", "prior_regularizer": "sbnp",
        "prior_reg": .01, "lr_schedule": "cosine",
        "optimizer_svd_backend": "batched", "training_precision": "bf16_channels_last",
        "routing_temperature": 0.25,
    }
    presets["empf"] = {
        **presets["bcap"], "critic_formulation": "empf", "reg_arm": None,
        "ebi_beta": 6.0, "optimizer_convolution_rule": "quintic",
        "optimizer_dense_profile": "newton_schulz_hybrid_v1",
        "prior_regularizer": "empf", "prior_reg": 0.01, "lr_schedule": "cosine",
        "optimizer_svd_backend": "batched", "training_precision": "bf16_channels_last",
        "routing_temperature": 0.25, "ucd_weight": 0.02,
    }
    presets["rhbi"] = {
        **presets["bcap"], "critic_formulation": "rhbi", "reg_arm": None,
        "optimizer_convolution_rule": "quintic",
        "optimizer_dense_profile": "polar_cpu_v1",
        "lr_schedule": "cosine", "ebi_beta": 20.0,
        "prior_regularizer": "moment_tether", "prior_reg": .01,
        "routing_temperature": 0.25, "rhbi_weight": 1.0,
    }
    presets["atlas"] = {**presets["e22"], "birth_death_backend": "auto",
                         "birth_death_cells": 128, "reopen_guard": "settled"}
    presets["e22_routed"] = {**presets["e22"], "row_policy": "routed_paired"}
    presets["halloween"] = {
        **presets["bcap_adam"], "reg_coeff": 0.0,
        "lr": 0.008020980209802098, "d_lr_mult": 0.4922016232592352,
        "betas": (0.6119661196611966, 0.5612256122561226),
        "d_betas": (0.1051710517105171, 0.7203172031720317),
        "eps": 0.3087330873308733, "d_eps": 0.007500075000750007,
        "prior_betas": (0.0, 0.999), "prior_eps": 1e-8,
        "adam_variant": "tensorflow_v1", "loss": "least_squares",
        "loss_labels": (-1.0, 1.0, 1.0), "lr_schedule": "constant",
    }
    if name in models:
        selected = DEFAULT_RECIPE if trainer is None else trainer
        if selected not in presets:
            owner = "DEFAULT_RECIPE" if trainer is None else "trainer"
            raise ValueError(f"{owner} must name a trainer preset, got {selected!r}")
        options = {"name": selected, **presets[selected], **models[name]}
    elif name in presets:
        if trainer is not None:
            raise ValueError("trainer is only supported with a model-family name")
        options = presets[name]
    else:
        raise ValueError(f"Unknown recipe {name!r}; choose {', '.join((*models, *presets))}")
    if name != "gan":
        options = {**options, "name": name}
    defaults = {field.name: field.default for field in fields(Recipe)}
    unknown = overrides.keys() - defaults.keys()
    if unknown:
        raise TypeError(f"Unknown recipe field(s): {', '.join(sorted(unknown))}")
    return {**defaults, **options, **overrides}


def learning_rate_scale(step, total_steps, start=0.6, floor=0.05):
    """Scale by completed updates: full LR for 60%, then cosine to the floor.

    Pass zero before the first optimizer update, as in the reference trainers.
    """
    if total_steps <= 0 or not 0 <= start < 1 or not 0 <= floor <= 1:
        raise ValueError("invalid learning-rate schedule")
    if floor == 1:
        return 1.0
    fraction = min(1.0, max(0.0, (step - start * total_steps) / ((1 - start) * total_steps)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction))


class NetworkLRTransition:
    """Caller-triggered network LR decay for a custom training loop.

    The caller decides when validation has plateaued and marks the number of
    completed updates. G/D then cosine-decay to the recipe's network floor over
    ``decay_steps``; the particle prior keeps its ordinary full-budget schedule.
    Save this object's state alongside the optimizers for exact continuation.
    """

    def __init__(self, decay_steps, start_step=None):
        if type(decay_steps) is not int or decay_steps <= 0:
            raise ValueError("decay_steps must be a positive integer")
        if start_step is not None and (type(start_step) is not int or start_step < 0):
            raise ValueError("start_step must be a nonnegative integer or None")
        self.decay_steps = decay_steps
        self.start_step = start_step

    def mark_plateau(self, completed_steps):
        """Start decay after this many updates; repeating the same mark is safe."""
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("completed_steps must be a nonnegative integer")
        if self.start_step is not None and self.start_step != completed_steps:
            raise ValueError("network LR transition is already marked")
        self.start_step = completed_steps

    def state_dict(self):
        return {"decay_steps": self.decay_steps, "start_step": self.start_step}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"decay_steps", "start_step"}:
            raise ValueError("invalid network LR transition state")
        restored = type(self)(**state)
        if restored.decay_steps != self.decay_steps:
            raise ValueError("network LR transition decay_steps differ from the configured schedule")
        self.start_step = restored.start_step


def learning_rate_scales(step, recipe, *, network_transition=None, controller=None):
    """Return ``(network, prior)`` LR multipliers after ``step`` completed updates.

    Generator and critic ("network") follow ``learning_rate_scale`` over
    ``min(total_steps, network_lr_horizon_cap)`` down to ``network_lr_floor``
    and then hold; the particle prior follows it over the full budget down to
    ``lr_floor``. When ``network_transition`` is supplied, G/D instead hold at
    full LR until its caller-marked plateau, then decay over its ``decay_steps``.
    KA2's critic controller is independent of these multipliers.

    A continuous recipe instead reads the current scales from its matching
    ``controller``; ``step`` is ignored. GANTrainer updates that controller
    from ordinary training signals. A custom loop must do that explicitly.
    The network value is the shared base multiplier; DV7 additionally applies
    ``controller.critic_scale()`` to D. Use ``scale_learning_rates(..., critic=)``
    to apply all three parameter roles.
    """
    if getattr(recipe, "lr_control", "mobility") == "stationarity":
        raise ValueError("lr_control='stationarity' requires per-group policy state; use "
                         "E22Policy/UpdatePolicy.begin_step() or GANTrainer instead of learning_rate_scales")
    if recipe.continuous_policy is not None:
        from .continuous import DataDriftController
        if not isinstance(controller, DataDriftController) or controller.variant != recipe.continuous_policy:
            raise ValueError("continuous LR scales require the matching controller; use GANTrainer or pass controller=")
        if network_transition is not None:
            raise ValueError("continuous policies do not accept scheduled network transitions")
        return controller.current_scales()
    if controller is not None:
        raise ValueError("controller requires a continuous recipe")
    if recipe.lr_schedule != "cosine":
        if type(step) is not int or step < 0:
            raise ValueError("LR schedule clock must be completed nonnegative training updates")
        if network_transition is not None:
            raise ValueError("explicit constant/exponential LR does not accept network transitions")
        exponent = step / recipe.lr_decay_steps
        if recipe.lr_decay_staircase:
            exponent = math.floor(exponent)
        scale = recipe.lr_decay_rate**exponent if recipe.lr_schedule == "exponential" else 1.0
        return scale, scale
    total = recipe.total_steps
    if network_transition is None:
        horizon = min(total, recipe.network_lr_horizon_cap or total)
        network = learning_rate_scale(step, horizon, recipe.lr_anneal_start, recipe.resolved_network_lr_floor)
    elif isinstance(network_transition, NetworkLRTransition):
        network = (1.0 if network_transition.start_step is None else
                   learning_rate_scale(step - network_transition.start_step,
                                       network_transition.decay_steps, 0.0,
                                       recipe.resolved_network_lr_floor))
    else:
        raise TypeError("network_transition must be a NetworkLRTransition or None")
    prior = learning_rate_scale(step, total, recipe.lr_anneal_start, recipe.lr_floor)
    return network, prior


def scale_learning_rates(step, recipe, optimizers, base_rates, prior=None, *,
                         network_transition=None, controller=None, critic=None):
    """Set every group's LR from ``learning_rate_scales(step, recipe)``.

    ``base_rates`` holds each optimizer's unscaled group LRs (read them once
    after construction). Groups whose parameters all belong to ``prior`` get
    the prior multiplier; every other group gets the network one, so a custom
    loop follows the recipe's split network/prior schedules.
    Pass a ``NetworkLRTransition`` to choose the network decay from validation
    while retaining the ordinary prior schedule. Returns ``(network, prior)``
    multipliers. Continuous recipes require their matching ``controller``.
    DV7 also requires ``critic``: its module or the exact optimized critic
    parameters. Critic and prior groups must contain only their own role.
    D receives the additional critic multiplier; the returned pair remains
    the shared network/prior multipliers. This helper never observes data or
    advances the controller: use the same observe_game/observe_real ordering
    as GANTrainer before applying rates.
    """
    network, prior_scale = learning_rate_scales(step, recipe, network_transition=network_transition, controller=controller)
    optimizers = tuple(optimizers)
    from .recipe_schedules import apply_training_schedules
    apply_training_schedules(step, recipe, optimizers)
    prior_ids = set() if prior is None else {id(p) for p in prior.parameters()}
    from .drift_variants import drift_features
    if drift_features(recipe.continuous_policy).critic_balance:
        from torch import Tensor, nn
        if critic is None:
            raise ValueError("asymmetric continuous rates require critic=module or optimized critic parameters")
        try:
            parameters = list(critic.parameters() if isinstance(critic, nn.Module) else critic)
        except TypeError as error:
            raise TypeError("critic must be a module or iterable of parameters") from error
        if not parameters or any(not isinstance(p, Tensor) for p in parameters):
            raise ValueError("critic must identify nonempty optimized parameters")
        critic_ids = {id(p) for p in parameters}
        if critic_ids & prior_ids:
            raise ValueError("critic and prior parameter roles overlap")
        optimizers, base_rates = tuple(optimizers), tuple(base_rates)
        if len(optimizers) != len(base_rates):
            raise ValueError("base_rates must match every optimizer")
        pending, found = [], set()
        critic_scale = controller.critic_scale() if recipe.critic_payoff_damping else 1.
        for optimizer, rates in zip(optimizers, base_rates):
            if len(optimizer.param_groups) != len(rates):
                raise ValueError("base_rates must match every optimizer group")
            for group, rate in zip(optimizer.param_groups, rates):
                ids = {id(p) for p in group["params"]}
                if not ids or (ids & critic_ids and not ids <= critic_ids):
                    raise ValueError("critic parameters must occupy complete nonempty groups")
                if ids & prior_ids and not ids <= prior_ids:
                    raise ValueError("prior parameters must occupy complete groups")
                if getattr(optimizer, "critic", None) is not None and not ids <= critic_ids:
                    raise ValueError("critic role does not match the critic optimizer")
                found.update(ids & critic_ids)
                value = rate * (prior_scale if ids <= prior_ids else network)
                if ids <= critic_ids:
                    value *= critic_scale
                pending.append((group, value))
        if found != critic_ids:
            raise ValueError("critic parameters must all belong to the supplied optimizers")
        for group, value in pending:
            group["lr"] = value
        return network, prior_scale
    for optimizer, rates in zip(optimizers, base_rates):
        for group, rate in zip(optimizer.param_groups, rates):
            is_prior = bool(prior_ids) and all(id(p) in prior_ids for p in group["params"])
            group["lr"] = rate * (prior_scale if is_prior else network)
    return network, prior_scale
