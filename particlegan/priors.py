"""Composable prior manipulation architecture.

This module provides base interfaces and implementations for swappable prior
manipulation, enabling custom index sampling, in-flight latent transformations,
generator post-processing autograd hooks, and intrinsic prior penalties.

Provenance contract:

* :func:`unwrap_prior` is the single way to reach the storage prior beneath a
  :class:`ManipulatedPrior`; callers never duck-type ``prior.base``.
* Every manipulator's hyperparameters are its configuration
  (:meth:`PriorManipulator.config`). They travel in ``state_dict()`` as module
  extra state and loading a state whose configuration differs raises.
  :meth:`PriorManipulator.manipulator_spec` / :meth:`ManipulatedPrior.manipulator_spec`
  return the JSON-compatible class + settings record for receipts.
* Which hooks a manipulator actually implements is resolved once, at wrap
  time, from whether its class overrides the base no-op. Changing
  ``ManipulatedPrior.manipulators`` after construction is unsupported.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import inspect
import hashlib
import math

import torch
import torch.nn as nn

from . import _qr


def unwrap_prior(prior: Any) -> Any:
    """Return the storage prior beneath any :class:`ManipulatedPrior` wrappers.

    Plain priors (and ``None``) are returned unchanged.
    """
    while isinstance(prior, ManipulatedPrior):
        prior = prior.base
    return prior


def _overrides(manipulator: "PriorManipulator", name: str) -> bool:
    return getattr(type(manipulator), name) is not getattr(PriorManipulator, name)


def _accepts(function: Any, name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


class PriorManipulator(nn.Module):
    """Base class for composable prior manipulators.

    A PriorManipulator intercepts or augments the sampling, code transformation,
    generator output postprocessing, or prior regularization of ParticlePrior
    or MoGParticlePrior modules without altering the base prior storage.

    Configuration: list hyperparameter attribute names in ``config_fields``
    (JSON-compatible values). They, plus ``penalty_weight``, form
    :meth:`config`, are saved as module extra state and validated on load.
    ``penalty_weight`` scales :meth:`regularize`'s value inside
    ``ManipulatedPrior.regularize``; it is part of the recorded configuration
    rather than a hidden constant folded into ``regularize``.

    ``additive_transform`` declares whether :meth:`transform` acts as
    ``codes + offset`` with an offset that strict-progress replay may hold
    fixed. ``None`` (default) infers True exactly when ``transform`` is not
    overridden.
    """

    row_local_gradients: bool = True
    additive_transform: Optional[bool] = None
    config_fields: Tuple[str, ...] = ()

    def __init__(self, *, penalty_weight: float = 1.0) -> None:
        super().__init__()
        if (type(penalty_weight) not in (int, float) or not math.isfinite(penalty_weight)
                or penalty_weight < 0):
            raise ValueError(f"penalty_weight must be a finite nonnegative number, got {penalty_weight!r}")
        self.penalty_weight = float(penalty_weight)

    # ------------------------------------------------------------------
    # Configuration provenance
    # ------------------------------------------------------------------

    def config(self) -> Dict[str, Any]:
        """JSON-compatible hyperparameters that define this manipulator's law."""
        return {"penalty_weight": getattr(self, "penalty_weight", 1.0),
                **{name: getattr(self, name) for name in self.config_fields}}

    def manipulator_spec(self) -> Dict[str, Any]:
        """Class identity plus :meth:`config`, suitable for receipts."""
        cls = type(self)
        return {"class": f"{cls.__module__}.{cls.__qualname__}", "config": self.config()}

    def get_extra_state(self) -> Dict[str, Any]:
        return self.manipulator_spec()

    def set_extra_state(self, state: Any) -> None:
        expected = self.manipulator_spec()
        if state != expected:
            raise ValueError(f"prior manipulator configuration mismatch: checkpoint has {state!r}, "
                             f"module is {expected!r}")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Legacy checkpoints (saved before configuration provenance existed)
        # carry no ``_extra_state``. They load without validation instead of
        # failing a strict load; parameters and buffers still load exactly.
        legacy = prefix + "_extra_state" not in state_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if legacy and prefix + "_extra_state" in missing_keys:
            missing_keys.remove(prefix + "_extra_state")

    def bind(self, base_prior: nn.Module) -> None:
        """Validate against / lazily size to the wrapped storage prior (wrap time)."""

    @property
    def is_additive_transform(self) -> bool:
        if self.additive_transform is None:
            return not _overrides(self, "transform")
        return bool(self.additive_transform)

    # ------------------------------------------------------------------
    # Hooks (all no-ops by default)
    # ------------------------------------------------------------------

    def sample_indices(
        self,
        base_prior: nn.Module,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        """Custom particle index selection.

        Args:
            base_prior: The underlying base prior module (e.g. ParticlePrior).
            batch_size: Positive integer number of particles to sample.
            generator: Optional torch.Generator for isolated RNG control.

        Returns:
            Optional 1D LongTensor of shape (batch_size,) on base_prior device,
            or None to delegate index sampling to subsequent manipulators or base.
            Returned tensors must not alias manipulator state.
        """
        return None

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Transforms sampled latent particle rows.

        Args:
            codes: Float tensor of sampled codes, shape (B, D) or intermediate transform shape.
            idx: Optional LongTensor of shape (B,) containing sampled particle indices.
            base_prior: The underlying base prior module.
            generator: Optional torch.Generator for isolated RNG control.
            training: Boolean indicating training (True) vs evaluation (False) mode.

        Returns:
            Transformed tensor, typically of shape (B, D_out).
        """
        return codes

    def postprocess_generated(
        self,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Post-processes generator outputs or attaches backward hooks.

        GANTrainer applies it on every path that feeds the critic or serving:
        critic-step fakes (under ``torch.no_grad``), generator-step fakes,
        strict-progress replays, ``GANTrainer.sample`` and
        ``ServedModel.generate``/``sample``. Implementations that attach
        autograd hooks must therefore tolerate tensors without
        ``requires_grad`` (attach only when ``generated.requires_grad``).

        Args:
            generated: Float tensor of generator outputs, typically (B, C, H, W).

        Returns:
            Post-processed or autograd-hooked tensor of identical shape and dtype.
        """
        return generated

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates the intrinsic (unweighted) manipulator penalty.

        ``ManipulatedPrior.regularize`` multiplies it by ``penalty_weight`` and
        GANTrainer adds the sum to loss_g, reporting it every step as
        ``result["prior_manipulation"]``. Manipulators that do not override
        this method are skipped entirely.

        Args:
            raw_rows: Raw particle table rows (prior.z or prior.z[unique(indices)]), shape (N_sub, D).
            sampled_codes: Optional transformed latent codes used for generation, shape (B, D_lat).
            indices: Optional particle indices drawn for this batch, shape (B,).
            generated: Optional generator outputs fake_g, shape (B, C, H, W).
            logits: Optional critic logits critic(fake_g), shape (B, ...).

        Returns:
            Scalar loss tensor () with device and dtype matching raw_rows.
        """
        return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)


def _forwarded(name: str, doc: str) -> property:
    """Explicit read-through to the storage prior; AttributeError when it lacks ``name``."""
    def get(self):
        return getattr(self.base, name)
    return property(get, doc=doc)


class ManipulatedPrior(nn.Module):
    """Wraps a base ParticlePrior or MoGParticlePrior and chains PriorManipulators.

    Forwards an explicit set of storage attributes (``z``, ``num_particles``,
    ``z_dim``, ``init_std`` and, when the base defines them, ``sigma``,
    ``standardize``, ``d0``, ``sigma_rel``, ``means``, ``set_sigma``) to
    ``self.base``. Nothing else falls through, so ``hasattr`` capability checks
    are truthful; use :func:`unwrap_prior` to reach other base state.
    """

    def __init__(
        self,
        base: nn.Module,
        manipulators: Optional[Sequence[Any]] = None,
    ) -> None:
        super().__init__()
        inherited: List[Any] = []
        if isinstance(base, ManipulatedPrior):
            inherited = list(base.manipulators)
            base = unwrap_prior(base)
        if not isinstance(base, nn.Module):
            raise TypeError(f"base prior must be an nn.Module, got {type(base)}")
        self.base = base

        flat_manipulators: List[PriorManipulator] = list(inherited)
        if manipulators is not None:
            for item in manipulators:
                if isinstance(item, (list, tuple)):
                    flat_manipulators.extend(item)
                else:
                    flat_manipulators.append(item)
        for m in flat_manipulators:
            if not isinstance(m, PriorManipulator):
                raise TypeError(f"manipulator must be an instance of PriorManipulator, got {type(m)}")
        self.manipulators = nn.ModuleList(flat_manipulators)
        for m in self.manipulators:
            m.bind(self.base)

        # Capabilities resolved once (manipulators are fixed after construction).
        self._samplers = tuple(m for m in self.manipulators if _overrides(m, "sample_indices"))
        self._transforms = tuple(m for m in self.manipulators if _overrides(m, "transform"))
        self._postprocessors = tuple(m for m in self.manipulators if _overrides(m, "postprocess_generated"))
        self._penalties = tuple(m for m in self.manipulators
                                if _overrides(m, "regularize") and m.penalty_weight != 0)
        self._base_forward_eps = _accepts(self.base.forward, "eps")
        self._base_sample_noise = _accepts(getattr(self.base, "sample", None), "noise_generator")

    def manipulate(self, *manipulators: Any) -> "ManipulatedPrior":
        """Chain additional manipulators, returning a new ManipulatedPrior."""
        return ManipulatedPrior(self.base, [*self.manipulators, *manipulators])

    def manipulator_spec(self) -> List[Dict[str, Any]]:
        """Ordered class + configuration records of the manipulator chain."""
        return [m.manipulator_spec() for m in self.manipulators]

    # ------------------------------------------------------------------
    # Resolved capabilities
    # ------------------------------------------------------------------

    @property
    def postprocesses_generated(self) -> bool:
        """True when some manipulator overrides ``postprocess_generated``."""
        return bool(self._postprocessors)

    @property
    def has_regularizer(self) -> bool:
        """True when some manipulator overrides ``regularize`` with nonzero weight."""
        return bool(self._penalties)

    @property
    def transforms_codes(self) -> bool:
        """True when some manipulator overrides ``transform``."""
        return bool(self._transforms)

    @property
    def additive_transforms(self) -> bool:
        """True when every transform is ``codes + offset`` (strict-progress replayable)."""
        return all(m.is_additive_transform for m in self.manipulators)

    @property
    def normal_noise(self) -> bool:
        """True when any manipulator adds normal noise."""
        return any(getattr(m, "normal_noise", False) or type(m).__name__ == "NormalNoiseManipulator"
                   for m in self.manipulators)

    @property
    def row_local_gradients(self) -> bool:
        """``prior_capabilities(self)["row_local_gradients"]``.

        False unless the storage prior is exactly ``ParticlePrior`` or an
        unstandardized ``MoGParticlePrior`` (and every manipulator is
        row-local). Before PR #481 this property ignored the base and
        reported True for other bases (e.g. ``FreshGaussianPrior``).
        """
        from .capabilities import prior_capabilities
        return prior_capabilities(self)["row_local_gradients"]

    # ------------------------------------------------------------------
    # Explicitly forwarded base prior attributes
    # ------------------------------------------------------------------

    @property
    def z(self) -> torch.Tensor:
        return self.base.z

    @z.setter
    def z(self, value: torch.Tensor) -> None:
        self.base.z = value

    @z.deleter
    def z(self) -> None:
        del self.base.z

    num_particles = _forwarded("num_particles", "Rows in the storage table.")
    z_dim = _forwarded("z_dim", "Width of the storage table (before transforms).")
    init_std = _forwarded("init_std", "Storage-table initialization scale.")
    sigma = _forwarded("sigma", "MoG kernel width (MoG bases only).")
    standardize = _forwarded("standardize", "MoG read standardization (MoG bases only).")
    d0 = _forwarded("d0", "MoG calibration spacing (MoG bases only).")
    means = _forwarded("means", "MoG component centers (MoG bases only).")
    set_sigma = _forwarded("set_sigma", "MoG kernel-width setter (MoG bases only).")

    @property
    def sigma_rel(self) -> float:
        return self.base.sigma_rel

    @sigma_rel.setter
    def sigma_rel(self, value: float) -> None:
        if not hasattr(self.base, "sigma_rel"):
            raise AttributeError(f"{type(self.base).__name__} has no sigma_rel")
        self.base.sigma_rel = value

    # ------------------------------------------------------------------
    # Sampling & Execution
    # ------------------------------------------------------------------

    def sample_indices(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """Sample indices by delegating through manipulators, falling back to base."""
        for m in self._samplers:
            idx = m.sample_indices(self.base, batch_size, generator=generator)
            if idx is not None:
                return idx
        if hasattr(self.base, "sample_indices"):
            return self.base.sample_indices(batch_size, generator=generator)
        return torch.randint(0, self.base.num_particles, (batch_size,),
                             device=self.base.z.device, generator=generator)

    def _transform(self, codes, idx, generator, eps=None):
        for m in self._transforms:
            if _accepts(m.transform, "eps"):
                codes = m.transform(
                    codes, idx, base_prior=self.base, generator=generator,
                    training=self.training, eps=eps
                )
            else:
                codes = m.transform(
                    codes, idx, base_prior=self.base, generator=generator,
                    training=self.training
                )
        return codes

    def forward(
        self,
        idx: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        *,
        eps: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass indexing into base prior and applying transform chain."""
        if self._base_forward_eps:
            codes = self.base(idx, generator=generator, eps=eps)
        else:
            codes = self.base(idx)
        return self._transform(codes, idx, generator, eps=eps)

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        *,
        fixed_first_n: bool = False,
        offset: int = 0,
        eps: Optional[torch.Tensor] = None,
        noise_generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Sample latent codes and particle indices."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if fixed_first_n:
            if self._base_sample_noise:
                base_codes, idx = self.base.sample(
                    batch_size, generator=generator, fixed_first_n=True,
                    offset=offset, eps=eps, noise_generator=noise_generator
                )
            else:
                base_codes, idx = self.base.sample(
                    batch_size, generator=generator, fixed_first_n=True, offset=offset
                )
            gen = noise_generator if noise_generator is not None else generator
            return self._transform(base_codes, idx, gen, eps=eps), idx

        idx = self.sample_indices(batch_size, generator=generator)
        gen = noise_generator if noise_generator is not None else generator
        codes = self.forward(idx, generator=gen, eps=eps)
        return codes, idx

    def postprocess_generated(self, generated: torch.Tensor) -> torch.Tensor:
        """Chain postprocess_generated hooks across all manipulators."""
        for m in self._postprocessors:
            generated = m.postprocess_generated(generated)
        return generated

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum ``penalty_weight * regularize(...)`` across overriding manipulators."""
        total = torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)
        for m in self._penalties:
            loss = m.regularize(
                raw_rows,
                sampled_codes=sampled_codes,
                indices=indices,
                generated=generated,
                logits=logits,
            )
            total = total + m.penalty_weight * loss
        return total


class StratifiedSamplingManipulator(PriorManipulator):
    """Epoch-stratified without-replacement permutation sampler with optional local k-NN jitter.

    Buffers:
        _perm: LongTensor of shape (num_particles,) storing the current permutation.
        _cursor: Scalar LongTensor tracking the consumption cursor in [0, num_particles].
        thermal_ratio: Scalar Float32 tensor in [0, 1] scaling optional k-NN jitter.

    Both _perm and _cursor are persisted in state_dict() for bit-exact checkpoint continuation.
    ``num_particles`` must equal the wrapped prior's table size. The jitter is
    an additive per-row offset (strict-progress replay holds it fixed).
    """

    additive_transform = True
    config_fields = ("num_particles", "jitter_std", "k_nn", "initial_cursor")

    def __init__(
        self,
        num_particles: int,
        *,
        jitter_std: float = 0.0,
        k_nn: int = 4,
        initial_cursor: int = 0,
        penalty_weight: float = 1.0,
    ) -> None:
        super().__init__(penalty_weight=penalty_weight)
        if type(num_particles) is not int or num_particles <= 0:
            raise ValueError(f"num_particles must be positive integer, got {num_particles}")
        if jitter_std < 0 or not math.isfinite(jitter_std):
            raise ValueError(f"jitter_std must be finite nonnegative float, got {jitter_std}")
        if type(k_nn) is not int or k_nn < 1:
            raise ValueError(f"k_nn must be positive integer, got {k_nn}")
        if type(initial_cursor) is not int or not 0 <= initial_cursor <= num_particles:
            raise ValueError(f"initial_cursor must be an integer in [0, {num_particles}], got {initial_cursor!r}")

        self.num_particles = num_particles
        self.jitter_std = float(jitter_std)
        self.k_nn = int(k_nn)
        self.initial_cursor = initial_cursor

        self.register_buffer("_perm", torch.arange(num_particles, dtype=torch.long))
        self.register_buffer("_cursor", torch.tensor(initial_cursor, dtype=torch.long))
        self.register_buffer("thermal_ratio", torch.tensor(1.0, dtype=torch.float32))

    def bind(self, base_prior: nn.Module) -> None:
        rows = getattr(base_prior, "num_particles", None)
        if rows is not None and rows != self.num_particles:
            raise ValueError(f"StratifiedSamplingManipulator num_particles={self.num_particles} "
                             f"does not match the base prior's {rows} rows")

    def set_thermal_ratio(self, ratio: Any) -> None:
        with torch.no_grad():
            val = float(ratio.item()) if isinstance(ratio, torch.Tensor) else float(ratio)
            self.thermal_ratio.fill_(max(0.0, min(1.0, val)))

    def sample_indices(
        self,
        base_prior: nn.Module,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        device = self._perm.device

        # In evaluation mode, draw uniform random indices without advancing the epoch cursor
        if not self.training:
            return torch.randint(0, self.num_particles, (batch_size,), device=device, generator=generator)

        # Batch exceeds total particles: generate fresh permutation with repeats
        if batch_size >= self.num_particles:
            perm = torch.randperm(self.num_particles, device=device, generator=generator)
            reps = (batch_size + self.num_particles - 1) // self.num_particles
            return perm.repeat(reps)[:batch_size]

        c = int(self._cursor.item())
        if c + batch_size >= self.num_particles:
            rem = self.num_particles - c
            tail = self._perm[c:].clone() if rem > 0 else self._perm[:0]
            self._perm.copy_(torch.randperm(self.num_particles, device=device, generator=generator))
            need = batch_size - rem
            idx = torch.cat((tail, self._perm[:need]), dim=0) if need > 0 else tail
            self._cursor.fill_(need)
            return idx
        else:
            # Clone: a later reshuffle rewrites _perm in place.
            idx = self._perm[c : c + batch_size].clone()
            self._cursor.fill_(c + batch_size)
            return idx

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if not training or self.jitter_std <= 0.0 or len(codes) <= 1:
            return codes

        t_ratio = float(self.thermal_ratio.item())
        if t_ratio <= 1e-5:
            return codes

        # Detached k-NN local distance computation keeps row_local_gradients = True
        with torch.no_grad():
            k = min(self.k_nn, len(codes) - 1)
            dists = torch.cdist(codes.detach(), codes.detach(), p=2.0)
            eye_inf = torch.eye(len(codes), device=codes.device, dtype=codes.dtype) * 1e6
            knn_dists, _ = torch.topk(dists + eye_inf, k=k, dim=1, largest=False)
            local_scale = knn_dists.median(dim=1, keepdim=True).values.clamp_min(1e-4)

        noise = torch.randn(codes.shape, device=codes.device, dtype=codes.dtype, generator=generator)
        return codes + (self.jitter_std * t_ratio) * local_scale * noise


class SimplexRoutingManipulator(PriorManipulator):
    """Maps particle codes to simplex mode weights w(z) in Delta^{K-1} and optional residuals.

    Formulation:
        w(z) = softmax(P(z) / T) in Delta^{K-1}
        where P(z) is either a 2D linear projection matrix Parameter[z_dim, num_modes]
        (mode="project") or coordinate slicing z[:, :num_modes] (mode="slice").

    All learnable parameters are 2D Parameter matrices (ndim == 2) so DualNorm
    optimizes them natively with role="router".

    Projection initialization never reads the global RNG. By default it is the
    repository's hashed deterministic semi-orthogonal matrix keyed by
    ``init_seed`` and shape (``particlegan._qr``, the same source as
    ``particlegan.init.deterministic_orthogonal_``). Passing ``generator``
    instead draws ``nn.init.orthogonal_`` from that caller-owned stream (recorded
    as ``projection_init="generator"``, with that stream's initial seed and a
    SHA-256 of its state at the draw in ``projection_init_generator``).
    Without ``z_dim`` the projection (``"project"``) or the recorded slice
    width (``"slice"``) is sized when wrapped by :class:`ManipulatedPrior`. The transform is not
    additive, so strict-progress replay rejects it.
    """

    additive_transform = False
    config_fields = ("num_modes", "temperature", "z_dim", "routing_mode", "residual_dim",
                     "learnable_projection", "init_seed", "projection_init",
                     "projection_init_generator")

    def __init__(
        self,
        num_modes: int,
        *,
        temperature: float = 1.0,
        z_dim: Optional[int] = None,
        routing_mode: str = "project",
        residual_dim: Optional[int] = None,
        learnable_projection: bool = True,
        init_seed: int = 0,
        generator: Optional[torch.Generator] = None,
        penalty_weight: float = 1.0,
    ) -> None:
        super().__init__(penalty_weight=penalty_weight)
        if type(num_modes) is not int or num_modes < 2:
            raise ValueError(f"num_modes must be integer >= 2, got {num_modes}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive finite float, got {temperature}")
        if routing_mode not in ("project", "slice"):
            raise ValueError(f"routing_mode must be 'project' or 'slice', got {routing_mode!r}")
        if z_dim is not None and (type(z_dim) is not int or z_dim < 1):
            raise ValueError(f"z_dim must be a positive integer or None, got {z_dim!r}")
        if residual_dim is not None and (type(residual_dim) is not int or residual_dim < 0):
            raise ValueError(f"residual_dim must be a nonnegative integer or None, got {residual_dim!r}")
        if type(init_seed) is not int or init_seed < 0:
            raise ValueError(f"init_seed must be a nonnegative integer, got {init_seed!r}")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")

        self.num_modes = num_modes
        self.temperature = float(temperature)
        self.routing_mode = routing_mode
        self.z_dim = z_dim
        self.residual_dim = residual_dim
        self.learnable_projection = bool(learnable_projection)
        self.init_seed = init_seed
        self.projection_init = "deterministic" if generator is None else "generator"
        self.projection_init_generator: Optional[Dict[str, Any]] = None
        self._init_generator = generator

        if self.learnable_projection:
            self.register_parameter("projection", None)
        else:
            self.register_buffer("projection", None)
        if routing_mode == "project" and z_dim is not None:
            self._initialize_projection(z_dim)

        self.last_weights: Optional[torch.Tensor] = None

    def _initialize_projection(self, z_dim: int, device: Any = None, dtype: Any = None) -> None:
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if self._init_generator is None:
            proj = _qr.semi_orthogonal(_qr.key("SimplexRoutingManipulator", self.init_seed,
                                               z_dim, self.num_modes), z_dim, self.num_modes)
            proj = proj.to(device=device, dtype=dtype)
        else:
            state = self._init_generator.get_state()
            self.projection_init_generator = {
                "initial_seed": int(self._init_generator.initial_seed()),
                "state_sha256": hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()}
            proj = torch.empty(z_dim, self.num_modes, device=self._init_generator.device, dtype=dtype)
            nn.init.orthogonal_(proj, generator=self._init_generator)
            proj = proj.to(device=device)
        self._init_generator = None  # consumed once; never retained across copies
        self.z_dim = z_dim
        if self.learnable_projection:
            self.projection = nn.Parameter(proj)
        else:
            self.projection = proj

    def _ensure_initialized(self, z_dim: int, device: Any = None, dtype: Any = None) -> None:
        if self.routing_mode == "project" and self.projection is None:
            self._initialize_projection(z_dim, device=device, dtype=dtype)

    def bind(self, base_prior: nn.Module) -> None:
        z = getattr(base_prior, "z", None)
        if z is None:
            return
        if self.z_dim is not None and self.z_dim != z.shape[1]:
            raise ValueError(f"SimplexRoutingManipulator z_dim={self.z_dim} does not match "
                             f"the base prior's {z.shape[1]}")
        self._ensure_initialized(z.shape[1], device=z.device, dtype=z.dtype)
        if self.routing_mode == "slice":
            self.z_dim = z.shape[1]  # record the bound width in the configuration

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if self.routing_mode == "project":
            if self.projection is None:
                raise RuntimeError("SimplexRoutingManipulator projection is unsized; pass z_dim "
                                   "or wrap the prior with ManipulatedPrior before sampling")
            if codes.shape[-1] != self.projection.shape[0]:
                raise ValueError(f"codes dimension {codes.shape[-1]} does not match projection "
                                 f"input {self.projection.shape[0]}")
            logits = torch.matmul(codes, self.projection.to(device=codes.device, dtype=codes.dtype))
            w = torch.softmax(logits / self.temperature, dim=-1)
            if self.residual_dim is not None and self.residual_dim > 0:
                res = codes[:, :self.residual_dim]
                out = torch.cat([w, res], dim=-1)
            else:
                out = w
        else:
            if codes.shape[-1] < self.num_modes:
                raise ValueError(
                    f"codes dimension {codes.shape[-1]} smaller than num_modes {self.num_modes}"
                )
            logits = codes[:, :self.num_modes]
            w = torch.softmax(logits / self.temperature, dim=-1)
            res = codes[:, self.num_modes:]
            out = torch.cat([w, res], dim=-1) if res.shape[-1] > 0 else w

        self.last_weights = w.detach()
        return out

    @staticmethod
    def split(codes: torch.Tensor, num_modes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Utility separating simplex weights from residual codes."""
        w = codes[:, :num_modes]
        res = codes[:, num_modes:] if codes.shape[-1] > num_modes else None
        return w, res


class NormalNoiseManipulator(PriorManipulator):
    """Adds isotropic Gaussian noise N(0, sigma^2 I) to sampled particle codes.

    Preserves row-local gradients and additive transforms for strict-progress
    replay. Noise draws consume the provided generator (or noise_generator
    routed via ManipulatedPrior).
    """

    normal_noise: bool = True
    additive_transform: bool = True
    row_local_gradients: bool = True
    config_fields: Tuple[str, ...] = ("sigma",)

    def __init__(
        self,
        sigma: float = 0.025,
        *,
        penalty_weight: float = 1.0,
        kappa_radial: float = 1.0,
    ) -> None:
        super().__init__(penalty_weight=penalty_weight)
        if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma must be a finite nonnegative number, got {sigma!r}")
        if type(kappa_radial) not in (int, float) or not math.isfinite(kappa_radial) or kappa_radial < 0:
            raise ValueError(f"kappa_radial must be a finite nonnegative number, got {kappa_radial!r}")
        self.sigma = float(sigma)
        self.kappa_radial = float(kappa_radial)

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adds isotropic normal noise to codes.

        Args:
            codes: Float tensor of base particle codes, shape (B, D).
            idx: Optional LongTensor of particle table indices, shape (B,).
            base_prior: The underlying base prior module (ParticlePrior).
            generator: Optional torch.Generator for isolated RNG control.
            training: Boolean indicating training vs evaluation mode.
            eps: Optional explicit noise tensor of shape (B, D) for reproducible
                 snapshots or unit test fixtures.

        Returns:
            Perturbed latent tensor of shape (B, D).
        """
        if self.sigma <= 0:
            return codes

        if eps is not None:
            if (eps.shape != codes.shape or eps.device != codes.device
                    or eps.dtype != codes.dtype):
                raise ValueError("eps must match sampled codes' shape, device and dtype")
            noise = eps
        else:
            noise = torch.randn(
                codes.shape,
                device=codes.device,
                dtype=codes.dtype,
                generator=generator,
            )

        if self.kappa_radial != 1.0 and codes.ndim >= 2 and codes.shape[-1] >= 2:
            codes_detached = codes.detach()
            norm = codes_detached.norm(dim=-1, keepdim=True)
            mask = norm > 1e-6
            if mask.any():
                e_r = torch.zeros_like(codes_detached)
                e_r = torch.where(mask, codes_detached / norm.clamp_min(1e-6), e_r)
                z_r = (noise * e_r).sum(dim=-1, keepdim=True) * e_r
                z_theta = noise - z_r
                noise = torch.where(
                    mask,
                    self.kappa_radial * z_r + z_theta,
                    noise,
                )

        return codes + self.sigma * noise


# Semantic alias
AdditiveNormalManipulator = NormalNoiseManipulator


def _register_initialization() -> None:
    # A learnable projection is a semi-orthogonal matrix (unit-norm columns or
    # rows): entry RMS 1/sqrt(max(z_dim, K)), so deterministic_orthogonal_ on a
    # wrapped prior reproduces the constructor's scale instead of raising.
    from .init import Normal, register
    register(SimplexRoutingManipulator, lambda module: {} if module.projection is None else {
        "projection": Normal(0.0, 1.0 / math.sqrt(max(module.projection.shape)))})


_register_initialization()


from .wavefront import (  # noqa: E402
    GhostFreeDeMixingManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
)
from .tomographic import (  # noqa: E402
    CosineRepulsionManipulator,
    MarginalEntropyManipulator,
)

__all__ = [
    "unwrap_prior",
    "NormalNoiseManipulator",
    "AdditiveNormalManipulator",
    "PriorManipulator",
    "ManipulatedPrior",
    "StratifiedSamplingManipulator",
    "SimplexRoutingManipulator",
    "GhostFreeDeMixingManipulator",
    "SobolevWavefrontManipulator",
    "SplatCarrierManipulator",
    "CosineRepulsionManipulator",
    "MarginalEntropyManipulator",
]

