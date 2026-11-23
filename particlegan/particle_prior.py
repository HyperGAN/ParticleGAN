"""
Particle prior module.

This defines a learnable latent particle cloud that can be shared across
different training scripts. It is intentionally minimal and fully
vectorized so it plays nicely with Accelerate / DDP and large particle
counts (e.g. 100k+).
"""

from typing import Any, Optional, Tuple
import math
from numbers import Real

import torch
import torch.nn as nn


PRIOR_CODE_LAWS = ("table", "fresh_gaussian")


class ParticlePrior(nn.Module):
    r"""
    Learnable latent particle cloud.

    This module holds a parameter matrix z \in R^{M x D} where each row is a
    latent "particle". Sampling is implemented as pure indexing into this
    matrix, so it is:

      * fully vectorized (no Python-side loops),
      * data-parallel / multi-GPU friendly (z is just a regular Parameter),
      * easy to plug into EP-style regularizers that operate on the full cloud.

    Pass `learnable=False` for the frozen-Gaussian control: identical interface,
    identical sampling, but the cloud is a buffer rather than a Parameter, so it
    never moves and `parameters()` comes back empty.

    Typical usage (with Accelerate):

        prior = ParticlePrior(num_particles=100_000, z_dim=256)
        prior, ... = accelerator.prepare(prior, ...)

        # later in the training loop
        idx = accelerator.unwrap_model(prior).sample_indices(batch_size)
        z = prior(idx)  # Keep the DDP forward path so gradients synchronize.

    ``code_law`` selects what ``sample`` returns. The default ``"table"`` reads
    table rows (the behaviour above). ``"fresh_gaussian"`` is the matched
    fixed-Gaussian latent control: ``sample`` still draws the same uniform row
    indices from ``generator``, then returns fresh codes ``init_std * N(0, I)``
    of the same dimension instead of table rows. The plain table has no
    separate noise stream, so those codes are also drawn from ``generator``:
    the first call's indices match the table law, but every later index draw
    sits at a different stream position. Only :class:`MoGParticlePrior` with a
    separate ``noise_generator`` (or with ``sigma > 0``, where the table law
    also draws kernel noise from ``generator``) keeps index draws aligned. Set it only through :meth:`set_code_law` on a frozen table
    (normally via ``Recipe.prior_code_law``); table reads through ``forward``
    then raise, so hosts that enumerate rows cannot silently mix the two laws.
    It is plain module configuration, not checkpoint state: the owning recipe
    re-applies it when a trainer is rebuilt.
    """

    code_law = "table"

    def __init__(
        self,
        num_particles: int = 20_000,
        z_dim: int = 4,
        init_std: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        learnable: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()

        if type(num_particles) is not int or num_particles <= 0:
            raise ValueError(f"num_particles must be positive, got {num_particles}")
        if type(z_dim) is not int or z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError("init_std must be finite and nonnegative")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.init_std = float(init_std)

        # One tensor of particles.
        # When learnable (the default) it is a single Parameter, so DDP /
        # Accelerate treat it like any other weight matrix.
        #
        # `learnable=False` registers the same tensor as a *buffer* instead: the
        # cloud is then a frozen draw from N(0, init_std^2) that no optimizer can
        # move. That is the control condition for the whole premise of this repo
        # -- a fixed Gaussian prior, with G left to do all the warping on its own
        # -- while every call site below (`sample`, `forward`, `z[idx]`) keeps
        # working unchanged. Sampling still consumes exactly the same randomness
        # either way, so flipping this flag does not shift any other RNG stream.
        z = torch.empty(num_particles, z_dim, **factory_kwargs)
        if learnable:
            self.z = nn.Parameter(z)
        else:
            self.register_buffer("z", z)
        with torch.no_grad():
            self.z.normal_(mean=0.0, std=init_std, generator=generator)

    @property
    def num_particles(self) -> int:
        return self.z.shape[0]

    @property
    def z_dim(self) -> int:
        return self.z.shape[1]

    @torch.no_grad()
    def sample_indices(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """
        Sample integer indices into the particle table.

        This is fully vectorized and runs entirely on the same device
        as `self.z`, which keeps it efficient in multi-GPU setups.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        return torch.randint(
            0,
            self.num_particles,
            (batch_size,),
            device=self.z.device,
            generator=generator,
        )

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        Index into the particle table.

        Args:
            idx: LongTensor of indices on any device; if necessary it will be
                 moved to `self.z.device` before indexing.

        Returns:
            z_batch: (B, z_dim) subset of the particle cloud.
        """
        if self.code_law != "table":
            raise ValueError("fresh_gaussian codes are not table reads; draw codes with sample()")
        if idx.device != self.z.device:
            idx = idx.to(self.z.device)
        return self.z[idx]

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        *,
        fixed_first_n: bool = False,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Convenience wrapper returning both latent codes and their indices.

        By default, this samples indices uniformly at random from the full
        particle table using `sample_indices`.

        If `fixed_first_n` is True, the call becomes deterministic and returns
        a contiguous block of particles:

            idx = [offset, offset + 1, ..., offset + batch_size - 1]

        This is handy for evaluation snapshots where you want to keep a fixed
        latent grid over the course of training (e.g. for videos).

        Returns:
            z_batch: (B, z_dim)
            idx: (B,) LongTensor of indices on the same device as `self.z`.
        """
        if self.code_law != "table":
            idx = self._sample_rows(batch_size, generator, fixed_first_n=fixed_first_n, offset=offset)
            return self._fresh_codes(len(idx), generator), idx
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if fixed_first_n:
            if offset < 0:
                raise ValueError(f"offset must be non-negative, got {offset}")
            if offset + batch_size > self.num_particles:
                raise ValueError(
                    f"Requested batch_size={batch_size} with offset={offset}, "
                    f"but num_particles={self.num_particles}"
                )
            with torch.no_grad():
                idx = torch.arange(
                    offset,
                    offset + batch_size,
                    device=self.z.device,
                    dtype=torch.long,
                )
        else:
            with torch.no_grad():
                idx = self.sample_indices(batch_size, generator=generator)
        z_batch = self.z[idx]
        return z_batch, idx

    def set_code_law(self, law: str) -> None:
        """Select table reads or fresh ``init_std * N(0, I)`` codes (see class docs).

        ``"fresh_gaussian"`` requires a frozen table: its locations are never
        sampled, so a trainable table would receive no gradient.
        """
        if law not in PRIOR_CODE_LAWS:
            raise ValueError(f"code_law must be one of {PRIOR_CODE_LAWS}, got {law!r}")
        if law != "table" and self.z.requires_grad:
            raise ValueError("fresh_gaussian codes require a frozen location table")
        self.code_law = law

    @torch.no_grad()
    def _sample_rows(self, batch_size, generator, *, fixed_first_n=False, offset=0):
        """Row indices exactly as the table law draws them (same RNG consumption)."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if fixed_first_n:
            if offset < 0:
                raise ValueError(f"offset must be non-negative, got {offset}")
            if offset + batch_size > self.num_particles:
                raise ValueError(
                    f"Requested batch_size={batch_size} with offset={offset}, "
                    f"but num_particles={self.num_particles}"
                )
            return torch.arange(offset, offset + batch_size, device=self.z.device, dtype=torch.long)
        return self.sample_indices(batch_size, generator=generator)

    def _fresh_codes(self, batch_size, generator, eps=None):
        """Fresh ``init_std * N(0, I)`` codes for the ``fresh_gaussian`` law."""
        shape = (batch_size, self.z_dim)
        if eps is None:
            eps = torch.randn(shape, device=self.z.device, dtype=self.z.dtype, generator=generator)
        elif eps.shape != shape or eps.device != self.z.device or eps.dtype != self.z.dtype:
            raise ValueError("eps must match sampled codes' shape, device and dtype")
        return eps * self.init_std

    def manipulate(self, *manipulators: Any) -> "ManipulatedPrior":
        """Wrap this prior in a ManipulatedPrior chaining the given manipulators."""
        return ManipulatedPrior(self, manipulators)


def _nonnegative_scalar(value, name):
    if isinstance(value, torch.Tensor):
        valid = value.ndim == 0 and not value.is_complex() and value.dtype != torch.bool
    else:
        valid = isinstance(value, Real) and not isinstance(value, bool)
    if not valid or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, nonnegative scalar")
    return float(value.detach() if isinstance(value, torch.Tensor) else value)


@torch.no_grad()
def calibrate_mog_sigma(centers: torch.Tensor, sigma_rel: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sigma, d0)`` from supplied read-space centers, without RNG draws.

    ``d0`` is the median nearest-neighbor distance, averaging both middle
    distances for even counts. Both detached scalar tensors use centers' device
    and dtype; d0 is rounded to that dtype *before* multiplying by sigma_rel,
    preserving historical calibration. Centers are neither modified nor
    standardized here; pass ``prior.means()`` to calibrate standardized reads.

    Potentially very expensive: copies centers to CPU float64 and runs an exact
    SciPy cKDTree search if available, otherwise chunked Torch pairwise distances
    (quadratic work, bounded temporary memory). Even SciPy can be slow for large,
    high-dimensional tables. Construction and checkpoint loading never call this.
    """
    sigma_rel = _nonnegative_scalar(sigma_rel, "sigma_rel")
    if (centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] < 1
            or not centers.is_floating_point()):
        raise ValueError("centers must be a floating-point (N, D) tensor with N >= 2 and D >= 1")
    points = centers.detach().cpu().double()
    if not torch.isfinite(points).all():
        raise ValueError("component means must be finite for calibration")
    num_particles = len(points)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        # At most ~32 MiB of distances; no full N x N matrix is retained.
        chunk_size = max(1, min(1024, 4_000_000 // num_particles))
        nearest = []
        for start in range(0, num_particles, chunk_size):
            chunk = points[start:start + chunk_size]
            distances = torch.cdist(chunk, points, compute_mode="donot_use_mm_for_euclid_dist")
            rows = torch.arange(len(chunk))
            distances[rows, rows + start] = float("inf")
            nearest.append(distances.min(dim=1).values)
        distance = torch.cat(nearest).quantile(.5).item()
    else:
        import numpy as np
        distance = float(np.median(cKDTree(points.numpy()).query(points.numpy(), k=2)[0][:, 1]))
    d0 = centers.new_tensor(distance)
    if not torch.isfinite(d0) or d0 <= 0:
        raise ValueError("median nearest-neighbor distance must be finite and positive")
    sigma = d0 * sigma_rel
    if not torch.isfinite(sigma):
        raise ValueError("calibrated sigma must be finite")
    return sigma, d0


class MoGParticlePrior(ParticlePrior):
    """Uniform Gaussian mixture with learned means and one fixed isotropic sigma.

    ``sigma`` is a required finite, nonnegative scalar. Construction only draws
    centers; it never searches neighbors or calibrates noise. Each sample is
    ``means()[idx] + sigma * eps``. Optional per-dimension read standardization
    is differentiable. Regularize raw ``z``, not noisy or standardized draws.
    Use :func:`calibrate_mog_sigma` explicitly for historical spacing-based noise.
    """

    def __init__(
        self,
        num_particles: int = 400,
        z_dim: int = 4,
        init_std: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        learnable: bool = True,
        generator: Optional[torch.Generator] = None,
        *,
        sigma: float,
        standardize: bool = True,
    ) -> None:
        sigma = _nonnegative_scalar(sigma, "sigma")
        if type(standardize) is not bool:
            raise ValueError("standardize must be a boolean")
        super().__init__(num_particles, z_dim, init_std, device, dtype, learnable, generator)
        if standardize and self.num_particles < 2:
            raise ValueError("standardized means require at least two particles")
        # Retain legacy checkpoint/metric fields; explicit sigma has no spacing.
        self.sigma_rel = 0.0
        self.standardize = standardize
        self.register_buffer("sigma", self.z.new_zeros(()))
        self.register_buffer("d0", self.z.new_zeros(()))
        self.set_sigma(sigma)

    def means(self):
        """Return all differentiable component centers, without sampling noise."""
        if not self.standardize:
            return self.z
        return (self.z - self.z.mean(0)) / (self.z.std(0) + 1e-6)

    def _cached_sigma_float(self) -> float:
        """Return ``float(self.sigma)`` with 0 CUDA syncs when the buffer has not been mutated."""
        cached = getattr(self.sigma, "_mog_sigma_cache", None)
        sig_key = (self.sigma._version, self.sigma.data_ptr(), self.sigma.dtype)
        if cached is not None and cached[:3] == sig_key:
            return cached[3]
        val = float(self.sigma.detach().item())
        self.sigma._mog_sigma_cache = (*sig_key, val)
        return val

    @torch.no_grad()
    def set_sigma(self, sigma):
        """Explicitly replace the shared fixed scale, e.g. after optional calibration.

        Use this instead of mutating the buffer so zero-noise RNG behavior stays
        in sync. Optimizers never update sigma.
        """
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim != 0 or sigma.is_complex() or sigma.dtype == torch.bool:
                raise ValueError("sigma must be a finite, nonnegative scalar")
            cached = getattr(sigma, "_mog_sigma_cache", None)
            if (
                cached is not None
                and cached[:3] == (sigma._version, sigma.data_ptr(), self.sigma.dtype)
                and sigma.dtype == self.sigma.dtype
                and math.isfinite(cached[3])
                and cached[3] >= 0.0
            ):
                rounded = cached[3]
                self_cached = getattr(self.sigma, "_mog_sigma_cache", None)
                self_key = (self.sigma._version, self.sigma.data_ptr(), self.sigma.dtype)
                if self_cached is None or self_cached[:3] != self_key or self_cached[3] != rounded:
                    self.sigma.fill_(rounded)
                    self_key = (self.sigma._version, self.sigma.data_ptr(), self.sigma.dtype)
                self._noise_enabled = rounded > 0.0
                self.sigma._mog_sigma_cache = (*self_key, rounded)
                return
            sigma_f = float(sigma.detach().item())
            if not math.isfinite(sigma_f) or sigma_f < 0.0:
                raise ValueError("sigma must be a finite, nonnegative scalar")
        else:
            sigma_f = _nonnegative_scalar(sigma, "sigma")
        cpu_val = torch.tensor(sigma_f, dtype=self.sigma.dtype)
        if not bool(torch.isfinite(cpu_val)):
            raise ValueError("sigma must be finite in the prior dtype")
        rounded = float(cpu_val.item())
        self.sigma.fill_(rounded)
        self._noise_enabled = rounded > 0.0
        self.sigma._mog_sigma_cache = (self.sigma._version, self.sigma.data_ptr(), self.sigma.dtype, rounded)

    def forward(self, idx, generator=None, *, eps=None):
        """Draw from the indexed components; use means()[idx] for centers only.

        Explicit eps must match the output shape/device/dtype and replaces the
        Gaussian RNG draw. At sigma=0 no noise RNG is consumed.
        """
        if self.code_law != "table":
            raise ValueError("fresh_gaussian codes are not table reads; draw codes with sample()")
        z = self.means()[idx.to(self.z.device)]
        if self._noise_enabled:
            if eps is None:
                eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
            elif eps.shape != z.shape or eps.device != z.device or eps.dtype != z.dtype:
                raise ValueError("eps must match sampled codes' shape, device and dtype")
            z = z + self.sigma * eps
        return z

    def sample(self, batch_size, generator=None, *, fixed_first_n=False,
               offset=0, eps=None, noise_generator=None):
        """Return noisy codes and component indices, uniformly with replacement.

        fixed_first_n fixes indices only; also supply fixed eps for reproducible
        positive-noise snapshots. By default a generator controls both indices
        and noise. ``noise_generator`` optionally isolates the Gaussian draws
        from component-index draws; sigma=0 consumes neither noise stream.
        """
        if self.code_law != "table":
            # Same index draws as the table law, then fresh codes from the same
            # Gaussian stream the MoG kernel noise would use. Later index draws
            # stay aligned with the table law only when that stream is separate
            # (noise_generator) or sigma > 0; at sigma = 0 with one generator the
            # table law draws no noise, so the code draws shift later indices.
            idx = self._sample_rows(batch_size, generator, fixed_first_n=fixed_first_n, offset=offset)
            return self._fresh_codes(len(idx), generator if noise_generator is None else noise_generator,
                                     eps), idx
        # Reuse index validation and RNG consumption exactly, including r=0.
        _, idx = super().sample(batch_size, generator,
                                fixed_first_n=fixed_first_n, offset=offset)
        return self(idx, generator=generator if noise_generator is None else noise_generator, eps=eps), idx

    def get_extra_state(self):
        return {"sigma_rel": self.sigma_rel, "standardize": self.standardize}

    def set_extra_state(self, state):
        sigma_rel, standardize = state["sigma_rel"], state["standardize"]
        if not math.isfinite(sigma_rel) or sigma_rel < 0 or type(standardize) is not bool:
            raise ValueError("invalid MoG checkpoint configuration")
        self.sigma_rel, self.standardize = float(sigma_rel), standardize

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Pre-public experiment checkpoints saved just z/sigma/d0. Their read
        # standardization must still be supplied by the constructor/config.
        key = prefix + "_extra_state"
        if key not in state_dict:
            state = self.get_extra_state()
            if (prefix + "sigma" in state_dict and prefix + "d0" in state_dict
                    and state_dict[prefix + "d0"] > 0):
                state["sigma_rel"] = float(state_dict[prefix + "sigma"] / state_dict[prefix + "d0"])
            state_dict[key] = state
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        loaded_val = float(self.sigma.detach().item())
        self._noise_enabled = loaded_val > 0.0
        if math.isfinite(loaded_val) and loaded_val >= 0.0:
            self.sigma._mog_sigma_cache = (self.sigma._version, self.sigma.data_ptr(), self.sigma.dtype, loaded_val)
        elif hasattr(self.sigma, "_mog_sigma_cache"):
            delattr(self.sigma, "_mog_sigma_cache")


class GaussianPrior(nn.Module):
    """Fresh Gaussian latent draws, with no table or trainable parameters.

    An empty buffer tracks device and dtype through normal ``.to(...)`` calls.
    ``sample`` returns ``(codes, None)`` because draws have no particle indices.
    """

    def __init__(self, z_dim=4, init_std=1.0, device=None, dtype=None):
        super().__init__()
        if type(z_dim) is not int or z_dim <= 0:
            raise ValueError("z_dim must be a positive integer")
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError("init_std must be finite and nonnegative")
        self.z_dim, self.init_std = z_dim, float(init_std)
        self.register_buffer("_anchor", torch.empty(0, device=device, dtype=dtype))

    def forward(self, batch_size, generator=None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return torch.randn((batch_size, self.z_dim), device=self._anchor.device,
                           dtype=self._anchor.dtype, generator=generator) * self.init_std

    def sample(self, batch_size, generator=None):
        return self(batch_size, generator=generator), None


class FreshGaussianPrior(ParticlePrior):
    """Fresh Gaussian noise, with a fixed reference batch for snapshots only.

    Ordinary ``sample`` calls draw from N(0, init_std**2), independently of the
    reference buffer ``z``. ``fixed_first_n=True`` instead selects that buffer
    for reproducible plots. The buffer also keeps initialization RNG consumption
    identical across prior controls. This prior has no trainable parameters;
    ordinary samples return ``None`` for indices because they are not table rows.
    """

    def __init__(self, num_particles=100_000, z_dim=256, init_std=1.0,
                 device=None, dtype=None):
        super().__init__(num_particles, z_dim, init_std, device, dtype, learnable=False)
        self.init_std = float(init_std)

    def sample(self, batch_size, generator=None, *, fixed_first_n=False, offset=0):
        if fixed_first_n:
            return super().sample(batch_size, generator, fixed_first_n=True, offset=offset)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        z = torch.randn(batch_size, self.z_dim, device=self.z.device,
                        dtype=self.z.dtype, generator=generator) * self.init_std
        return z, None


PRIOR_KINDS = ("mog", "particles", "frozen_gaussian", "fresh_gaussian", "gaussian")


def canonical_prior_kind(kind: str) -> str:
    """Keep historical ``gaussian`` configs as frozen finite-table controls."""
    kind = str(kind).lower()
    if kind not in PRIOR_KINDS:
        raise ValueError(f"Unknown prior: {kind!r} (expected one of {PRIOR_KINDS})")
    return "frozen_gaussian" if kind == "gaussian" else kind


def make_prior(kind: str, **kwargs) -> ParticlePrior:
    kind = canonical_prior_kind(kind)
    if kind == "mog":
        return MoGParticlePrior(**kwargs)
    if kind == "fresh_gaussian":
        return FreshGaussianPrior(**kwargs)
    return ParticlePrior(**kwargs, learnable=(kind == "particles"))


# Manipulator initialization registers these prior classes through init.py.
# Bind the public return annotation only after those classes are available.
from .priors import ManipulatedPrior  # noqa: E402
