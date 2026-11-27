"""Experimental horizon-free data innovation and game stability controller.

Real statistics control scalar mobility only. They never enter a loss, sample,
parameter assignment, or target fit. Fixed random nonlinear features see more
than changes of the mean. Generator gradient alignment is a distinct signal.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
from typing import NamedTuple
import torch

from ._stats import adam_moment_states
from .drift_variants import drift_features


class _LatentApplication(NamedTuple):
    radius: torch.Tensor
    displacement: torch.Tensor
    fraction: torch.Tensor
    autocast_dtype: torch.dtype | None

    def metrics(self):
        with torch.autocast(device_type=self.radius.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype is not None):
            return torch.stack((self.radius.min(), self.radius.mean(), self.radius.max(),
                self.displacement.square().mean().sqrt(), (self.fraction < 1.).float().mean()))


class DataDriftController:
    _LATENT_METRICS = ("radius_min", "radius_mean", "radius_max", "perturbation_rms", "clipped_fraction")

    def __init__(self, variant="dv1"):
        self.variant = variant
        features = drift_features(variant)
        if features.latent_bandwidth:
            self.latent_bandwidth = None
        if features.support_trust:
            self.support_mean = self.support_variance = None
            self.support_score = 0.0
            self.support_trust = 1.0
        if features.local_support:
            self.latent_applications = []
        self.mobility = 1.0
        self.game_trust = 1.0
        self.game_ratio = 1.0
        self.payoff_error = 0.0
        if features.pair:
            self.pair_mean = self.pair_variance = None
            self.pair_score = self.pair_drive = 0.0
        if features.data_memory:
            self.data_memory = 0.0
        if features.fast_reference:
            self.fast_reference = None
            self.fast_mean_variance = self.slow_mean_variance = self.mean_covariance = None
        self.alignment = 0.0
        self.data_score = 0.0
        self.data_drive = 0.0
        self.last_cosine = 0.0
        self.previous_gradient = None
        self.location = self.scale = self.projection = None
        self.reference = self.variance = None
        self.updates = 0
        self.reopens = 0
        self.closed = False

    @property
    def features(self):
        """``DriftFeatures`` of ``variant`` (derived, never checkpointed)."""
        return drift_features(self.variant)

    @property
    def latent_applications(self):
        """The last two recorded DV12 applications as ordinary float dicts.

        Recording retains detached inputs for the last two applications on
        device. Reading computes their metrics and batches one host transfer;
        repeated reads use the materialized list. Diagnostics and checkpoint
        export read it too.
        """
        records = self._latent_application_records
        pending = [(index, value) for index, value in enumerate(records) if isinstance(value, _LatentApplication)]
        if pending:
            device = pending[-1][1].radius.device
            values = torch.stack([value.metrics().to(device=device) for _, value in pending]).cpu().tolist()
            for (index, _), row in zip(pending, values):
                records[index] = dict(zip(self._LATENT_METRICS, row))
        return records

    @latent_applications.setter
    def latent_applications(self, values):
        self._latent_application_records = values

    @staticmethod
    @torch.no_grad()
    def routed_geometry(table, log_mass):
        """Support and cell width of the routed bank's represented mass.

        Exact duplicate positions form one atom whose mass is the sum of its
        rows. Coordinate spread is mass weighted and the cell count is the
        effective number of those atoms, 1 / sum(p**2). Inactive rows never
        enter either calculation. Routing usage and real observations do not
        enter this prior geometry.
        """
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or not all(table.shape)
                or not table.is_floating_point() or not isinstance(log_mass, torch.Tensor)
                or log_mass.shape != (len(table),) or log_mass.device != table.device
                or log_mass.dtype != table.dtype or bool(torch.isnan(log_mass).any())
                or bool(torch.isposinf(log_mass).any())):
            raise ValueError("routed geometry requires a floating table and matching finite or inactive log masses")
        active = torch.isfinite(log_mass)
        if not bool(active.any()):
            raise ValueError("routed geometry requires positive represented mass")
        table, log_mass = table.detach()[active], log_mass.detach()[active]
        if not bool(torch.isfinite(table).all()):
            raise ValueError("active routed geometry centers must be finite")
        centers, inverse, counts = torch.unique(table, dim=0, return_inverse=True, return_counts=True)
        # Group log masses before normalization: duplicating an atom must not
        # change its effective cell count. Sorted segments avoid atomic sums.
        ordered = log_mass[inverse.argsort(stable=True)]
        maxima = torch.segment_reduce(ordered, "max", lengths=counts)
        relative = (ordered - maxima.repeat_interleave(counts)).exp()
        masses = maxima + torch.segment_reduce(relative, "sum", lengths=counts).log()
        weights = masses.softmax(0)
        positive = weights > 0
        centers, weights = centers[positive], weights[positive]
        location = (weights.unsqueeze(1) * centers).sum(0)
        variance = (weights.unsqueeze(1) * (centers - location).square()).sum(0)
        effective_atoms = weights.square().sum().reciprocal()
        width = variance.sqrt() * effective_atoms.pow(-1. / table.shape[1])
        return centers, width

    @staticmethod
    def routed_prior(table, log_mass):
        """Bind immutable geometry to one complete fast/average candidate.

        Build a fresh view after owner updates. Reusing it within one complete
        multi-site forward avoids recomputing geometry at each routing site.
        """
        support, width = DataDriftController.routed_geometry(table, log_mass)
        return SimpleNamespace(z=table, log_mass=log_mass, _mass_support=support, _mass_width=width)

    @torch.no_grad()
    def observe_prior(self, prior):
        """Kernel width from learned latent geometry, never real-data statistics.

        N**(-1/d) is the linear size of one of N equal-volume latent cells.
        Running coordinate spread makes the width follow ordinary prior learning
        without an acquisition clock or an evaluator horizon.
        """
        if not self.features.latent_bandwidth:
            return
        if hasattr(prior, "log_mass"):
            width = getattr(prior, "_mass_width", None)
            if width is None:
                _, width = self.routed_geometry(prior.z, prior.log_mass)
        else:
            z = prior.z.detach()
            width = z.std(0, unbiased=False) * len(z) ** (-1. / z.shape[1])
        if self.latent_bandwidth is None:
            self.latent_bandwidth = width.clone()
        else:
            self.latent_bandwidth.lerp_(width, .01)

    def perturb_latent(self, latent, stream, prior=None, record=False):
        if not self.features.latent_bandwidth:
            return latent
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream)
        trust = self.support_trust if self.features.support_trust else 1.
        displacement = self.latent_bandwidth * trust * noise
        if self.features.local_support:
            if prior is None:
                raise ValueError("local support requires the corresponding prior")
            with torch.no_grad():
                nearest = torch.full((len(latent),), float("inf"), device=latent.device, dtype=latent.dtype)
                # Squared distances by broadcasting (~25x faster than cdist's no-mm path); sqrt is monotone and
                # correctly rounded, so min-then-sqrt equals the old sqrt-then-min bitwise (d == 0 iff d**2 == 0).
                # 2048 // z_dim centers per chunk keeps peak memory below the old cdist path's.
                query = latent.detach()[:, None, :]
                support = prior.z.detach()
                if hasattr(prior, "log_mass"):
                    support = getattr(prior, "_mass_support", None)
                    if support is None:
                        support, _ = self.routed_geometry(prior.z, prior.log_mass)
                for centers in support.split(max(1, 2048 // latent.shape[1])):
                    distance = query - centers[None, :, :]
                    distance = distance.mul_(distance).sum(-1)
                    distance.masked_fill_(distance == 0, float("inf"))
                    nearest = torch.minimum(nearest, distance.min(1).values)
                nearest = nearest.sqrt()
                radius = torch.where(torch.isfinite(nearest), nearest * .5, torch.zeros_like(nearest))
                norm = displacement.norm(dim=1)
                fraction = (radius / norm.clamp_min(1e-20)).clamp_max(1.)
            displacement = displacement * fraction.unsqueeze(1)
            if record:
                autocast_dtype = (torch.get_autocast_dtype(latent.device.type)
                                  if torch.is_autocast_enabled(latent.device.type) else None)
                application = _LatentApplication(radius.detach(), displacement.detach(), fraction.detach(),
                                                 autocast_dtype)
                self._latent_application_records = [*self._latent_application_records, application][-2:]
        return latent + displacement

    @torch.no_grad()
    def observe_support(self, generator, critic, latent, sigma_out, stream):
        """Probe full latent width even while its applied width is suppressed."""
        if not self.features.support_trust:
            return
        modes = [(m, m.training) for root in (generator, critic) for m in root.modules()]
        try:
            generator.eval()
            critic.eval()
            noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream)
            clean = generator(latent)
            probe = generator(latent + self.latent_bandwidth * noise)
            if sigma_out:
                epsilon = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=stream)
                clean = clean + sigma_out * epsilon
                probe = probe + sigma_out * epsilon
            difference = (critic(clean) - critic(probe)).flatten()
            mean = difference.mean()
            variance = (difference.var(unbiased=False) / len(difference)).clamp_min(1e-10)
            if self.support_mean is None:
                self.support_mean = torch.zeros_like(mean)
                self.support_variance = variance.clone()
            self.support_mean.lerp_(mean, .02)
            self.support_variance.mul_(.98 ** 2).add_(variance, alpha=.02 ** 2)
            self.support_score = float(self.support_mean / self.support_variance.clamp_min(1e-10).sqrt())
            unexplained = max(0., (self.support_score - 3.) / 3.)
            self.support_trust = 1. / (1. + unexplained * unexplained)
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def observe_real(self, real):
        x = real.flatten(1)
        if self.location is None:
            self.location = x.mean(0)
            self.scale = x.std(0, unbiased=False).clamp_min(1e-6)
            # Local stream: feature initialization never advances training RNG.
            stream = torch.Generator(device="cpu").manual_seed(1729)
            self.projection = (torch.randn(x.shape[1], 32, generator=stream, device="cpu")
                               / math.sqrt(x.shape[1])).to(x)
        u = ((x - self.location) / self.scale) @ self.projection
        features = torch.cat((u, torch.cos(u), torch.sin(u), torch.cos(2*u), torch.sin(2*u)), 1)
        mean = features.mean(0)
        var = features.var(0, unbiased=False).clamp_min(1e-6)
        if self.reference is None:
            self.reference, self.variance = mean.clone(), var.clone()
        if self.features.fast_reference:
            batch_variance = var / len(x)
            if self.fast_reference is None:
                self.fast_reference = mean.clone()
                self.fast_mean_variance = batch_variance.clone()
                self.slow_mean_variance = batch_variance.clone()
                self.mean_covariance = batch_variance.clone()
            self.fast_reference.lerp_(mean, .1)
            self.reference.lerp_(mean, .01)
            self.fast_mean_variance.mul_(.9**2).add_(batch_variance, alpha=.1**2)
            self.slow_mean_variance.mul_(.99**2).add_(batch_variance, alpha=.01**2)
            self.mean_covariance.mul_(.9*.99).add_(batch_variance, alpha=.1*.01)
            variance_of_difference = (self.fast_mean_variance + self.slow_mean_variance - 2*self.mean_covariance).clamp_min(1e-10)
            self.data_score = float(((self.fast_reference - self.reference).square() / variance_of_difference).mean().sqrt())
        else:
            variance_of_difference = (var + self.variance * (.01 / 1.99)) / len(x)
            self.data_score = float(((mean - self.reference).square() / variance_of_difference.clamp_min(1e-10)).mean().sqrt())
            self.reference.lerp_(mean, .01)
        self.variance.lerp_(var, .01)
        self.data_drive = min(1., max(0., (self.data_score - 3.) / 3.))
        return self._advance_mobility()

    @torch.no_grad()
    def observe_blind(self):
        """``observe_real`` without any statistic of the real batch: the data-drift term is 0 and the mobility follows the
        game terms (payoff error, generator-gradient alignment) only."""
        self.data_score = 0.0
        self.data_drive = 0.0
        return self._advance_mobility()

    def _advance_mobility(self):
        if self.features.data_memory:
            memory_speed = .05 if self.data_drive > self.data_memory else .005
            self.data_memory += memory_speed * (self.data_drive - self.data_memory)
        target = max(self.data_drive, min(1., max(0., self.alignment / .2)))
        if self.features.payoff:
            target = max(self.data_drive, min(1., self.payoff_error ** 2))
        if self.features.pair_target:
            target = max(target, self.pair_drive)
        speed = .05 if target > self.mobility else .005
        self.mobility += speed * (target - self.mobility)
        if self.mobility < .1:
            self.closed = True
        elif self.mobility > .3 and self.closed:
            self.reopens += 1
            self.closed = False
        self.updates += 1
        return self.current_scales()

    def current_scales(self):
        prior_mobility = max(self.mobility, self.pair_drive) if self.features.pair_prior else self.mobility
        return ((.01 + .99 * self.mobility) * self.game_trust,
                (.05 + .95 * prior_mobility) * self.game_trust)

    @torch.no_grad()
    def observe_pair(self, real, fake):
        if not self.features.pair:
            return
        def features(x):
            u = ((x.flatten(1) - self.location) / self.scale) @ self.projection
            return torch.cat((u, torch.cos(u), torch.sin(u), torch.cos(2*u), torch.sin(2*u)), 1)
        real_features, fake_features = features(real), features(fake)
        difference = real_features.mean(0) - fake_features.mean(0)
        variance = (real_features.var(0, unbiased=False) / len(real) +
                    fake_features.var(0, unbiased=False) / len(fake)).clamp_min(1e-10)
        if self.pair_mean is None:
            self.pair_mean = torch.zeros_like(difference)
            self.pair_variance = variance.clone()
        self.pair_mean.lerp_(difference, .02)
        self.pair_variance.mul_(.98 ** 2).add_(variance, alpha=.02 ** 2)
        self.pair_score = float((self.pair_mean.square() / self.pair_variance.clamp_min(1e-10)).mean().sqrt())
        self.pair_drive = min(1., max(0., (self.pair_score - 3.) / 3.))

    def critic_scale(self):
        # Separate game balance acts even before the KA2 reference exists.
        return 1. / (1. + self.payoff_error ** 2) if self.features.critic_balance else 1.

    def observe_game(self, record):
        if not self.features.game_trust:
            return
        ratio = 1.0
        if record.sur_base and record.last_sur is not None:
            ratio = record.last_sur / record.sur_base
        self.game_ratio = ratio
        evidence = self.data_drive if self.features.drive_evidence else self.data_memory
        unexplained = max(0., ratio - 1.) * (1. - evidence)
        self.game_trust = 1. / (1. + unexplained * unexplained)

    @torch.no_grad()
    def observe_generator(self, generator, loss_gan=None, loss_discriminator=None):
        if self.features.payoff:
            error = max(0., float(loss_gan - loss_discriminator) / math.log(2.))
            self.payoff_error += .02 * (error - self.payoff_error)
        values = [p.grad.detach().flatten() for p in generator.parameters() if p.grad is not None]
        if not values:
            return
        gradient = torch.cat(values)
        if self.previous_gradient is not None:
            self.last_cosine = float(torch.dot(gradient, self.previous_gradient) /
                (gradient.norm() * self.previous_gradient.norm()).clamp_min(1e-20))
            self.alignment += .02 * (self.last_cosine - self.alignment)
        self.previous_gradient = gradient.clone()

    def diagnostics(self):
        return {**({"latent_applications": self.latent_applications} if self.features.local_support else {}), **({"support_score": self.support_score, "support_trust": self.support_trust, "applied_latent_bandwidth": None if self.latent_bandwidth is None else (self.latent_bandwidth * self.support_trust).tolist()} if self.features.support_trust else {}), **({"latent_bandwidth": self.latent_bandwidth.tolist()} if self.features.latent_bandwidth and self.latent_bandwidth is not None else {}), **({"pair_score": self.pair_score, "pair_drive": self.pair_drive} if self.features.pair else {}), **({"data_memory": self.data_memory} if self.features.data_memory else {}), **{k: getattr(self, k) for k in ("payoff_error", "game_trust", "game_ratio", "variant", "mobility", "alignment", "data_score", "data_drive", "last_cosine", "updates", "reopens", "closed")}}

    def state_dict(self):
        return deepcopy({("latent_applications" if name == "_latent_application_records" else name):
                         (self.latent_applications if name == "_latent_application_records" else value)
                         for name, value in self.__dict__.items()})

    def load_state_dict(self, state):
        keys = {"latent_applications" if name == "_latent_application_records" else name for name in self.__dict__}
        if not isinstance(state, dict) or set(state) != keys or state["variant"] != self.variant:
            raise ValueError("incompatible continuous controller state")
        if self.features.latent_bandwidth:
            bandwidth = state["latent_bandwidth"]
            current = self.latent_bandwidth
            if bandwidth is not None and (not isinstance(bandwidth, torch.Tensor)
                    or bandwidth.ndim != 1 or not bandwidth.is_floating_point()
                    or not torch.isfinite(bandwidth).all() or bool((bandwidth < 0).any())
                    or (current is not None and (bandwidth.shape != current.shape
                                                or bandwidth.dtype != current.dtype
                                                or bandwidth.device != current.device))):
                raise ValueError("continuous controller latent bandwidth does not match the table")
            if current is not None and bandwidth is None:
                raise ValueError("continuous controller latent bandwidth is missing")
        restored = deepcopy(state)
        if "latent_applications" in restored:
            restored["_latent_application_records"] = restored.pop("latent_applications")
        self.__dict__.update(restored)


# One-sided alpha = .05 Student-t quantiles t_{.95}(dof), dof = 1..11 (the
# window uses K = 12 pairs at scale b -> <= 11 dof, K/2 = 6 at 2b -> <= 5).
_T95 = (6.313752, 2.919986, 2.353363, 2.131847, 2.015048, 1.943180,
        1.894579, 1.859548, 1.833113, 1.812461, 1.795885)
# Table rows can use either tested scale for a decision. Four one-sided tests
# (two directions x two scales) spend .05 per window by Bonferroni, so each
# uses alpha=.0125. These are nominal per-window levels, not anytime bounds.
_T9875 = (25.451700, 6.205347, 4.176535, 3.495406, 3.163381, 2.968687,
          2.841244, 2.751524, 2.685011, 2.633767, 2.593093)


class SettleTest:
    """Stationarity (settling) test of one optimizer param group (lr_control="stationarity").

    Intrinsic time ``tau`` advances by applied/base LR each update; a block
    ends when ``tau`` reaches ``b`` (remainder carried). Block displacement
    D_k = theta(end) - theta(start); consecutive pairs give r = cos(D1, D2)
    at scale b and, every 4 blocks, cos(D1+D2, D3+D4) at scale 2b. A window
    of 2K blocks is decided by one-sided t tests (alpha .05 each way) at both
    scales with one host sync:
      STATIONARY (v_b=-1, v_2b!=+1): s /= 2, b = max(b0, b/2)
      DRIFT      (v_b=+1, v_2b!=-1): s = min(1, 2s), b = max(b0, b/2)
      MIXED (significant, opposite) or INCONCLUSIVE (v_b=0): b *= 2
      REVERSAL (decisive, opposite to the previous decisive): apply, b *= 2.
    Pairs where either block did not move are dropped (fewer dof). A zero-
    spread window is decided by the sign of its mean. A window without any
    valid pair is frozen (no decision). Holds no parameter references.
    """

    K = 12
    B0 = 1.0
    GAMMA = .5

    def __init__(self):
        self.s, self.b, self.tau = 1.0, self.B0, 0.0
        self.anchor = None
        self.blocks = []            # displacements of the current 4-block quad
        self.r_b, self.r_2b = [], []  # device scalars; NaN marks a dropped pair
        self.blocks_in_window = 0
        self.last_decisive = 0
        self.last_decisive_scale = None
        self.last = {}
        self.counts = {k: 0 for k in ("drift", "stationary", "mixed", "inconclusive", "reversal",
                                      "frozen", "dropped_pairs", "reopens", "restarts", "rebases")}
        self.windows = 0
        self.log = []
        self.last_block = None
        self.rows = None
        self.last_row_overlap = None
        self.invalid_block_rows = None

    @staticmethod
    def flat(params):
        return torch.cat([p.detach().reshape(-1) for p in params])

    @torch.no_grad()
    def begin(self, params):
        """Take the first anchor (at the first update, after any caller setup)."""
        if self.anchor is None:
            self.anchor = self.flat(params).clone()

    @torch.no_grad()
    def restart(self, params, reopen=False):
        """Discard the partial window's current quad and re-anchor (after a
        non-gradient jump). ``reopen``: also s = 1, b = b0, clear the window."""
        self.anchor = self.flat(params).clone()
        self.blocks = []
        self.tau = 0.0
        self.invalid_block_rows = None
        if reopen:
            self.s, self.b = 1.0, self.B0
            self.r_b, self.r_2b = [], []
            self.blocks_in_window = 0
            self.last_decisive = 0      # no REVERSAL judged against pre-change evidence
            self.last_decisive_scale = None
            self.counts["reopens"] += 1
        else:
            self.counts["restarts"] += 1

    @torch.no_grad()
    def rebase(self, params, rows):
        """Remove a non-gradient jump of table rows ``rows`` (birth-death move) from
        the current block by re-anchoring only those rows; the block, the quad and
        tau are kept, so frequent moves cannot starve the tester of blocks.
        Falls back to restart() when the group is not a row table."""
        if self.anchor is None or self.rows is None:
            return self.restart(params)
        theta = self.flat(params).view(self.rows, -1)
        self.anchor.view(self.rows, -1)[rows] = theta[rows]
        # A cloned row begins a new particle lineage.  Remove its evidence
        # from every pair in the current window and from the unfinished block.
        # Otherwise old and new displacements can form a spurious cosine.
        if self.invalid_block_rows is None:
            self.invalid_block_rows = torch.zeros(self.rows, device=theta.device, dtype=torch.bool)
        self.invalid_block_rows[rows] = True
        for block in self.blocks:
            block.view(self.rows, -1)[rows] = float("nan")
        for pair in self.r_b + self.r_2b:
            pair[rows] = float("nan")
        self.counts["rebases"] += 1

    def _cos(self, a, b):
        if self.rows is not None:
            # A table row is one independently sampled particle.  The global
            # cosine weights a row by its displacement energy, so a small set
            # of migrating particles can declare DRIFT for the whole prior.
            # Average bounded row cosines instead; rows absent from either
            # block have no direction and contribute no evidence.
            aa, bb = a.view(self.rows, -1), b.view(self.rows, -1)
            na, nb = aa.norm(dim=1), bb.norm(dim=1)
            valid = (na > torch.finfo(a.dtype).tiny) & (nb > torch.finfo(b.dtype).tiny)
            self.last_row_overlap = valid.float().mean()
            cosine = (aa * bb).sum(dim=1) / (na * nb).clamp_min(torch.finfo(a.dtype).tiny)
            return torch.where(valid, cosine.clamp(-1., 1.), torch.full_like(cosine, float("nan")))
        denom = a.norm() * b.norm()
        tiny = torch.finfo(a.dtype).tiny
        return torch.where(denom > tiny, (a @ b) / denom.clamp_min(tiny), torch.full_like(denom, float("nan")))

    @torch.no_grad()
    def observe(self, params, ratio, step=None):
        """Call after the group's optimizer step with applied LR / base LR."""
        if self.anchor is None:
            raise RuntimeError("SettleTest.observe before the anchor was taken")
        self.tau += float(ratio)
        if self.tau < self.b:
            return None
        self.tau -= self.b
        theta = self.flat(params)
        d = theta - self.anchor
        if self.invalid_block_rows is not None and self.rows is not None:
            d.view(self.rows, -1)[self.invalid_block_rows] = float("nan")
            self.invalid_block_rows.zero_()
        self.anchor = theta.clone()
        self.last_block = d
        self.blocks.append(d)
        self.blocks_in_window += 1
        if len(self.blocks) % 2 == 0:
            self.r_b.append(self._cos(self.blocks[-2], self.blocks[-1]))
        if len(self.blocks) == 4:
            self.r_2b.append(self._cos(self.blocks[0] + self.blocks[1], self.blocks[2] + self.blocks[3]))
            self.blocks = []
        if self.blocks_in_window >= 2 * self.K:
            return self._decide(step)
        return None

    @staticmethod
    def _verdict(values, strict=False):
        r = values[~torch.isnan(values)].clamp(-1., 1.).double()
        n = len(r)
        if n == 0:
            return None, 0, float("nan"), float("nan")
        mean = float(r.mean())
        if n == 1:
            return 0, 1, mean, float("nan")
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return (0 if mean == 0. else (1 if mean > 0 else -1)), n, mean, math.copysign(float("inf"), mean)
        t = mean / (sd / math.sqrt(n))
        table = _T9875 if strict else _T95
        crit = table[min(n - 1, len(table)) - 1]
        return (1 if t > crit else -1 if t < -crit else 0), n, mean, t

    def _decide(self, step):
        tested_b = self.b
        values = torch.stack(self.r_b + self.r_2b)
        if self.rows is not None:
            values = torch.nanmean(values, dim=1)
        values = values.cpu()   # the window's one host sync
        nb = len(self.r_b)
        strict = self.rows is not None
        vb, nb_ok, mb, tb = self._verdict(values[:nb], strict)
        v2, n2_ok, m2, t2 = self._verdict(values[nb:], strict)
        self.counts["dropped_pairs"] += (nb - nb_ok) + (len(values) - nb - n2_ok)
        self.windows += 1
        v2 = 0 if v2 is None else v2
        if vb is None:
            decision = "frozen"
        elif strict and vb * v2 == -1:
            decision = "mixed"
        elif strict and (vb == -1 or v2 == -1):
            decision = "stationary"
        elif strict and (vb == 1 or v2 == 1):
            decision = "drift"
        elif vb == -1 and v2 != 1:
            decision = "stationary"
        elif vb == 1 and v2 != -1:
            decision = "drift"
        elif vb != 0 and v2 == -vb:
            decision = "mixed"
        else:
            decision = "inconclusive"
        self.counts[decision] += 1
        if decision in ("stationary", "drift"):
            sign = -1 if decision == "stationary" else 1
            evidence_scale = 2 * tested_b if strict and vb != sign and v2 == sign else tested_b
            at_ceiling = self.s == 1.0
            self.s = self.s * self.GAMMA if sign < 0 else min(1.0, self.s / self.GAMMA)
            if decision == "drift" and at_ceiling and self.rows is not None:
                # A DRIFT verdict cannot raise a table LR already at its cap.
                # Probe a longer scale instead of repeating the same verdict.
                # The window tested b and 2b; if both show drift, 4b is the
                # next untested dyadic scale.  This lets a table with local
                # coherent steps but long-run reversal reach its settling
                # scale without using a clock or a magnitude threshold.
                self.b *= 4 if v2 == 1 else 2
            elif self.last_decisive == -sign and (not strict or self.last_decisive_scale == evidence_scale):
                self.counts["reversal"] += 1
                self.b = 2 * evidence_scale if strict else self.b * 2
            elif strict and vb == 0 and v2 != 0:
                # The first decisive evidence was at 2b. Halve that
                # evidence scale, retaining the current b for the next test.
                self.b = max(self.B0, self.b)
            else:
                self.b = max(self.B0, self.b / 2)
            self.last_decisive = sign
            self.last_decisive_scale = evidence_scale
        elif decision in ("mixed", "inconclusive"):
            self.b *= 2
        self.last = dict(decision=decision, step=step, tested_b=tested_b, verdict_b=vb, verdict_2b=v2,
                         evidence_scale=(2 * tested_b if strict and vb == 0 and v2 != 0 else tested_b),
                         t_b=tb, t_2b=t2, mean_r_b=mb, mean_r_2b=m2,
                         n_b=nb_ok, n_2b=n2_ok)
        self.log = (self.log + [[step, decision, self.s, self.b]])[-8:]
        self.r_b, self.r_2b, self.blocks = [], [], []
        self.blocks_in_window = 0
        return decision

    def diagnostics(self):
        out = dict(s=self.s, b=self.b, tau=self.tau, windows=self.windows, last=self.last,
                   counts=dict(self.counts), log=self.log, pairs_in_window=len(self.r_b))
        out["blocks_in_window"] = self.blocks_in_window
        out["intrinsic_steps_to_next_decision"] = max(0., (2 * self.K - self.blocks_in_window) * self.b - self.tau)
        if self.rows and self.last_block is not None:
            energy = torch.nan_to_num(self.last_block.view(self.rows, -1)).square().sum(1)
            top = max(1, math.ceil(.01 * self.rows))
            total = float(energy.sum())
            out["top1pct_row_energy_share"] = float(energy.topk(top).values.sum()) / total if total > 0 else None
            out["last_row_pair_overlap"] = (float(self.last_row_overlap)
                                            if self.last_row_overlap is not None else None)
        return out

    def state_dict(self):
        return deepcopy({k: v for k, v in self.__dict__.items()})

    def load_state_dict(self, state, numel):
        if set(state) != set(self.__dict__):
            raise ValueError("incompatible settle-test state")
        anchor = state["anchor"]
        if anchor is not None and (not isinstance(anchor, torch.Tensor) or anchor.numel() != numel):
            raise ValueError("settle-test anchor does not match its parameter group")
        if any(not isinstance(d, torch.Tensor) or d.numel() != numel for d in state["blocks"]):
            raise ValueError("settle-test blocks do not match its parameter group")
        pair_numel = state["rows"] if state["rows"] is not None else 1
        if any(not isinstance(r, torch.Tensor) or r.numel() != pair_numel
               for r in state["r_b"] + state["r_2b"]):
            raise ValueError("settle-test pair evidence does not match its parameter group")
        mask = state["invalid_block_rows"]
        if mask is not None and (state["rows"] is None or not isinstance(mask, torch.Tensor)
                                 or mask.shape != (state["rows"],) or mask.dtype != torch.bool):
            raise ValueError("settle-test row lineage mask does not match its parameter group")
        self.__dict__.update(deepcopy(state))


# --- Sequential (early-stopping) table test ---------------------------------------------------
# Error accounting per decision, identical to SettleTest's nominal .05: every one of the two
# scales (b, 2b) spends .0125 on an anytime-valid early-stopping e-process (Ville's inequality,
# any number of looks) and .0125 on the window-end fixed-n test (two one-sided tests at .00625,
# the quantiles below, dof 1..11) -> .025 per scale, .05 per decision.
_T99375 = (50.923037, 8.8602, 5.391949, 4.314656, 3.810005, 3.521223,
           3.335295, 3.205955, 3.110935, 3.038243, 2.980872)
_EARLY_ALPHA = .0125
_LOG_EARLY = math.log(1. / _EARLY_ALPHA)
# Effect-size prior variances of the mixture (equal weights, fixed in advance): the mixture of
# test martingales is a test martingale, and it needs no assumed effect size.
_PRIOR_G = tuple(4. ** k for k in range(5))


def _log_t_bayes_factor(t, n):
    """log mixture Bayes factor of H1: mu/sigma ~ N(0, g) against H0: mu = 0 for a one-sample t
    statistic on n >= 2 observations, sigma unknown (right-Haar prior). Under H0 this is a test
    martingale for every sigma (Lai 1976; Perez-Ortiz et al. 2022), so P(sup_n BF_n >= 1/a) <= a."""
    nu = n - 1
    t2 = t * t
    terms = []
    for g in _PRIOR_G:
        ng = n * g
        terms.append(-.5 * math.log1p(ng)
                     + .5 * n * (math.log1p(t2 / nu) - math.log1p(t2 / (nu * (1. + ng)))))
    top = max(terms)
    return top + math.log(sum(math.exp(x - top) for x in terms) / len(terms))


class SequentialSettleTest(SettleTest):
    """SettleTest that may decide after any completed pair instead of only at the window end.

    Same statistic (row-averaged cosine of consecutive block displacements at scales b and 2b),
    same scale search, same actions and the same nominal error per decision (.05) as SettleTest.
    What changes is *when* a decision may be taken:

    * after every completed pair the evidence collected so far in the window is tested with an
      anytime-valid e-process per scale (mixture t-test Bayes factor >= 1/.0125); a crossing
      decides immediately, so strong evidence no longer waits for the 2K-block window to close;
    * if nothing has crossed when the window fills (2K blocks) the fixed-n one-sided t tests are
      applied at alpha .00625 per direction per scale (the remaining half of the .05 budget), and
      the usual STATIONARY / DRIFT / MIXED / INCONCLUSIVE actions follow.

    ``early=False`` with ``final_table=_T9875`` reproduces SettleTest's decisions exactly.
    Row-lineage handling (birth-death rebase) is inherited: the moved rows are removed from every
    pair of the running window, and the scale means are recomputed from the row vectors at each look.
    """

    def __init__(self, early=True, final_table=None, early_stationary_only=False):
        super().__init__()
        self.early = bool(early)
        self.early_stationary_only = bool(early_stationary_only)
        self.final_table = tuple(_T99375 if final_table is None else final_table)
        self.counts["early"] = 0
        self.counts["held"] = 0
        self.looks = 0
        self.last_look = {}
        # Off-support gate (set by GANTrainer each step from the birth-death support test):
        # ``exclude``: rows that do not vote (off-support rows); ``hold_descent``: a STATIONARY verdict
        # is deferred (early: evidence kept; window end: window discarded) while too much of the table
        # is off-support.
        self.exclude = None
        self.hold_descent = False
        # Table release rule (a DRIFT verdict below the ceiling raises s): "any" = the SettleTest table rule (either scale
        # positive and not opposed); "both" = both scales significantly positive; "never" = no release from this tester.
        self.release_rule = "any"
        # ANCHOR: evidence scale (in intrinsic time) of the last accepted STATIONARY decision = the scale at which the bulk
        # mean-reverts. Under a stationary bulk the block cosine is positive below it and negative near it, so a DRIFT verdict
        # whose evidence scale is below twice this anchor is what a settled table produces and is not evidence of drift.
        self.b_anchor = None
        self.counts["drift_blocked"] = 0

    def restart(self, params, reopen=False):
        super().restart(params, reopen=reopen)
        if reopen:
            self.b_anchor = None

    @torch.no_grad()
    def _scale_values(self):
        """Per-pair scalars (mean over the rows that have a direction) for the b and 2b scales;
        one host sync."""
        nb = len(self.r_b)
        if not (self.r_b or self.r_2b):
            return torch.empty(0, dtype=torch.float64, device='cpu'), torch.empty(0, dtype=torch.float64, device='cpu')
        values = torch.stack(self.r_b + self.r_2b)
        if self.rows is not None:
            if self.exclude is not None:
                values = values.masked_fill(self.exclude.to(values.device)[None, :], float("nan"))
            values = torch.nanmean(values, dim=1)
        values = values.cpu().double()
        return values[:nb], values[nb:]

    @staticmethod
    def _early_stat(values):
        """Anytime-valid look: dict(verdict, n, mean, t, log_bf); verdict 0 until the e-process crosses."""
        r = values[~torch.isnan(values)].clamp(-1., 1.)
        n = len(r)
        out = dict(verdict=0, n=n, mean=float(r.mean()) if n else float("nan"), t=float("nan"),
                   log_bf=float("-inf"))
        if n < 2:
            return out
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return out          # degenerate spread: left to the window-end rule
        t = out["mean"] / (sd / math.sqrt(n))
        out["t"] = t
        out["log_bf"] = _log_t_bayes_factor(t, n)
        if out["log_bf"] >= _LOG_EARLY:
            out["verdict"] = 1 if t > 0 else -1
        return out

    def _final_verdict(self, values):
        """Window-end fixed-n one-sided t tests (table quantiles of ``final_table``)."""
        r = values[~torch.isnan(values)].clamp(-1., 1.)
        n = len(r)
        if n == 0:
            return dict(verdict=None, n=0, mean=float("nan"), t=float("nan"), log_bf=float("nan"))
        mean = float(r.mean())
        if n == 1:
            return dict(verdict=0, n=1, mean=mean, t=float("nan"), log_bf=float("nan"))
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return dict(verdict=(0 if mean == 0. else (1 if mean > 0 else -1)), n=n, mean=mean,
                        t=math.copysign(float("inf"), mean), log_bf=float("nan"))
        t = mean / (sd / math.sqrt(n))
        crit = self.final_table[min(n - 1, len(self.final_table)) - 1]
        return dict(verdict=(1 if t > crit else -1 if t < -crit else 0), n=n, mean=mean, t=t,
                    log_bf=float("nan"))

    def _table_decision(self, sb, s2, tested_b):
        """``(vb, v2, decision)``: the SettleTest table rule, release rule and descent hold."""
        vb, v2 = sb["verdict"], s2["verdict"]
        v2 = 0 if v2 is None else v2
        if vb is None:
            decision = "frozen"
        elif vb * v2 == -1:
            decision = "mixed"
        elif vb == -1 or v2 == -1:
            decision = "stationary"
        elif vb == 1 or v2 == 1:
            decision = "drift"
            if self.s < 1.0:
                drift_scale = 2 * tested_b if vb != 1 and v2 == 1 else tested_b
                if self.release_rule == "never" or (self.release_rule == "both" and not (vb == 1 and v2 == 1)):
                    decision = "inconclusive"   # a release needs the stated evidence; otherwise search a longer scale
                elif (self.release_rule == "anchor" and self.b_anchor is not None
                      and drift_scale < 2 * self.b_anchor):
                    decision = "inconclusive"
                    self.counts["drift_blocked"] += 1
        else:
            decision = "inconclusive"
        if decision == "stationary" and self.hold_descent:
            decision = "held"           # off-support mass above the FDR budget: descent is deferred
        return vb, v2, decision

    def _apply_decisive(self, decision, vb, v2, tested_b):
        """Move s and b for an accepted STATIONARY (-1) or DRIFT (+1) decision; return its sign."""
        sign = -1 if decision == "stationary" else 1
        evidence_scale = 2 * tested_b if vb != sign and v2 == sign else tested_b
        at_ceiling = self.s == 1.0
        self.s = self.s * self.GAMMA if sign < 0 else min(1.0, self.s / self.GAMMA)
        if decision == "drift" and at_ceiling and self.rows is not None:
            self.b *= 4 if v2 == 1 else 2
        elif self.last_decisive == -sign and self.last_decisive_scale == evidence_scale:
            self.counts["reversal"] += 1
            self.b = 2 * evidence_scale
        elif vb == 0 and v2 != 0:
            self.b = max(self.B0, self.b)
        else:
            self.b = max(self.B0, self.b / 2)
        self.last_decisive = sign
        self.last_decisive_scale = evidence_scale
        if sign < 0:
            self.b_anchor = evidence_scale
        return sign

    def _record_decision(self, sb, s2, vb, v2, tested_b, step, early, decision):
        self.last = dict(decision=decision, step=step, tested_b=tested_b, verdict_b=vb, verdict_2b=v2,
                         evidence_scale=(2 * tested_b if vb == 0 and v2 != 0 else tested_b),
                         t_b=sb["t"], t_2b=s2["t"], mean_r_b=sb["mean"], mean_r_2b=s2["mean"],
                         n_b=sb["n"], n_2b=s2["n"], early=bool(early),
                         log_bf_b=sb["log_bf"], log_bf_2b=s2["log_bf"])
        self.log = (self.log + [[step, decision, self.s, self.b]])[-8:]

    def _close_window(self):
        self.r_b, self.r_2b, self.blocks = [], [], []
        self.blocks_in_window = 0

    def _conclude(self, sb, s2, tested_b, step, early):
        """Map the two scale verdicts to a decision (table rule of SettleTest) and act on it."""
        vb, v2, decision = self._table_decision(sb, s2, tested_b)
        self.counts[decision] += 1
        if early:
            self.counts["early"] += 1
        if decision in ("stationary", "drift"):
            self._apply_decisive(decision, vb, v2, tested_b)
        elif decision in ("mixed", "inconclusive"):
            self.b *= 2
        self._record_decision(sb, s2, vb, v2, tested_b, step, early, decision)
        self._close_window()
        return decision

    def _look(self, step):
        """Early-stopping look after a completed pair; None while no e-process has crossed."""
        if len(self.r_b) < 2 and len(self.r_2b) < 2:
            return None
        vb_values, v2_values = self._scale_values()
        sb, s2 = self._early_stat(vb_values), self._early_stat(v2_values)
        self.looks += 1
        self.last_look = dict(step=step, n_b=sb["n"], n_2b=s2["n"], t_b=sb["t"], t_2b=s2["t"],
                              log_bf_b=sb["log_bf"], log_bf_2b=s2["log_bf"])
        if sb["verdict"] == 0 and s2["verdict"] == 0:
            return None
        if self.hold_descent and (sb["verdict"] == -1 or s2["verdict"] == -1):
            return None                 # a descent is due but held: keep the evidence, decide when the hold lifts
        if self.early_stationary_only:
            # Only a descent (STATIONARY: cheap to undo) may be decided early, and only when the other
            # scale can speak (>= 2 values) and does not oppose it. DRIFT / release and MIXED are decided
            # at the window end with the full two-scale table, so an oscillation (r_b > 0, r_2b < 0)
            # keeps its MIXED outcome instead of becoming an early DRIFT.
            calm_b = sb["verdict"] == -1 and s2["verdict"] != 1 and s2["n"] >= 2 and s2["mean"] <= 0.
            calm_2b = s2["verdict"] == -1 and sb["verdict"] != 1 and sb["n"] >= 2 and sb["mean"] <= 0.
            if not (calm_b or calm_2b):
                return None
        self.windows += 1
        return self._conclude(sb, s2, self.b, step, early=True)

    def _decide(self, step):
        """Window end (2K blocks, nothing crossed early): fixed-n tests, then the usual actions."""
        tested_b = self.b
        vb_values, v2_values = self._scale_values()
        sb, s2 = self._final_verdict(vb_values), self._final_verdict(v2_values)
        self.counts["dropped_pairs"] += (len(vb_values) - sb["n"]) + (len(v2_values) - s2["n"])
        self.windows += 1
        return self._conclude(sb, s2, tested_b, step, early=False)

    @torch.no_grad()
    def observe(self, params, ratio, step=None):
        """Call after the group's optimizer step with applied LR / base LR."""
        if self.anchor is None:
            raise RuntimeError("SettleTest.observe before the anchor was taken")
        self.tau += float(ratio)
        if self.tau < self.b:
            return None
        self.tau -= self.b
        theta = self.flat(params)
        d = theta - self.anchor
        if self.invalid_block_rows is not None and self.rows is not None:
            d.view(self.rows, -1)[self.invalid_block_rows] = float("nan")
            self.invalid_block_rows.zero_()
        self.anchor = theta.clone()
        self.last_block = d
        self.blocks.append(d)
        self.blocks_in_window += 1
        new_pair = False
        if len(self.blocks) % 2 == 0:
            self.r_b.append(self._cos(self.blocks[-2], self.blocks[-1]))
            new_pair = True
        if len(self.blocks) == 4:
            self.r_2b.append(self._cos(self.blocks[0] + self.blocks[1], self.blocks[2] + self.blocks[3]))
            self.blocks = []
        if self.blocks_in_window >= 2 * self.K:
            return self._decide(step)
        if new_pair and self.early:
            return self._look(step)
        return None

    def diagnostics(self):
        out = super().diagnostics()
        out["sequential"] = True
        out["hold_descent"] = self.hold_descent
        out["excluded_rows"] = None if self.exclude is None else int(self.exclude.sum())
        out["b_anchor"] = self.b_anchor
        out["early_decisions"] = self.counts.get("early", 0)
        out["looks"] = self.looks
        out["last_look"] = self.last_look
        return out


class StationarityLR:
    """Per-group settling tests for explicitly owned optimizers.

    ``optimizers`` starts with generator and critic owners; a separate table
    owner may follow.  The isolated ``prior_param`` group gets the sequential
    row-lineage-aware tester, independent of any training-loop class.
    """

    def __init__(self, optimizers, prior_param=None, release_rule="any"):
        self.testers = []
        for optimizer in optimizers:
            row = []
            for group in optimizer.param_groups:
                trainable = [p for p in group["params"] if p.requires_grad]
                tester = SettleTest() if trainable else None
                if tester is not None and prior_param is not None and len(group["params"]) == 1 \
                        and group["params"][0] is prior_param and prior_param.dim() == 2:
                    tester = SequentialSettleTest(early_stationary_only=True, final_table=_T9875)     # descent may be decided early
                    tester.rows = prior_param.shape[0]
                    tester.release_rule = release_rule
                row.append(tester)
            self.testers.append(row)

    def pairs(self, optimizers):
        for optimizer, row in zip(optimizers, self.testers):
            for group, tester in zip(optimizer.param_groups, row):
                if tester is not None:
                    yield group, tester

    def diagnostics(self):
        names = ("g", "d", "table")
        return {f"{names[i] if i < len(names) else 'optimizer' + str(i)}{j}": t.diagnostics()
                for i, row in enumerate(self.testers)
                for j, t in enumerate(row) if t is not None}

    def state_dict(self):
        return [[None if t is None else t.state_dict() for t in row] for row in self.testers]

    def load_state_dict(self, state, optimizers):
        if (not isinstance(state, list) or len(state) != len(self.testers)
                or any(not isinstance(a, list) or len(a) != len(b) for a, b in zip(state, self.testers))):
            raise ValueError("settle state group count does not match the optimizers")
        for optimizer, row, saved in zip(optimizers, self.testers, state):
            for group, tester, value in zip(optimizer.param_groups, row, saved):
                if (tester is None) != (value is None):
                    raise ValueError("settle state does not match the trainable groups")
                if tester is not None:
                    deepcopy(tester).load_state_dict(value, sum(p.numel() for p in group["params"]))
        for optimizer, row, saved in zip(optimizers, self.testers, state):
            for group, tester, value in zip(optimizer.param_groups, row, saved):
                if tester is not None:
                    tester.load_state_dict(value, sum(p.numel() for p in group["params"]))


class OptimizerSurprise:
    """Re-open signal read from the optimizers' own Adam state (reopen_signal="optimizer").

    Per trainable param group, after its optimizer step: q = mean |g| / sqrt(v_hat), the
    step's gradient size in units of Adam's own second-moment memory (with beta1 = 0 this is
    Adam's per-element step / lr). q is compared, in log space, as a fast EMA (K updates)
    against a slow level. The slow level follows q only while the game is calm (ratio <
    CALM), so a jump is measured against the level before it. The step's ratio is the
    geometric mean over groups of fast / slow. RISE held for K consecutive updates fires one
    re-open. Only an abrupt rise counts: the excursion must cross RISE within 2K updates of
    the last calm update; a slow ramp is the game's own evolution (e.g. KA2 leaving its
    warm-up), not a moved target. After a fire the detector re-seeds its level and stays
    disarmed for one slow horizon (8K updates) and until the ratio is back below CALM, so its
    own re-open cannot trigger the next one. No data statistic, no clock: ratios and counts
    of the detector's own updates only.
    """

    K = 12
    RISE = 2.0
    CALM = 1.25

    def __init__(self):
        self.fast, self.slow = {}, {}
        self.pending = {}
        self.armed = True
        self.streak = 0
        self.fires = 0
        self.since_calm = 0
        self.since_fire = None
        self.last_ratio = None
        self.last_ratios = {}
        self.log = []
        # KA2 anchor release latch (reopen_anchor="release"): None = off, else [seen_high].
        self.anchor_event = None
        self.anchor_events = 0

    @staticmethod
    @torch.no_grad()
    def group_q(optimizer, group):
        # Not _stats.adam_surprise_ratio: this is the elementwise mean of
        # |g| / (sqrt(v_hat) + eps) (Adam's own step / lr at beta1 = 0), not a
        # ratio of RMS values, so the two statistics differ numerically.
        beta2 = group["betas"][1]
        total, count = None, 0
        for p, st in adam_moment_states(optimizer, group):
            t = float(st["step"])
            if t < 1:
                continue
            vhat = st["exp_avg_sq"] / (1.0 - beta2 ** t)
            q = (p.grad.abs() / (vhat.sqrt() + group.get("eps", 1e-8))).mean()
            total = q if total is None else total + q
            count += 1
        return None if count == 0 else total / count

    def observe(self, key, optimizer, group):
        q = self.group_q(optimizer, group)
        if q is not None:
            self.pending[key] = q

    def decide(self, step=None, *, guard=None, network=None, loss_surprised=False):
        """Fold this update's observations in; return True when a re-open fires."""
        if not self.pending:
            return False
        keys = sorted(self.pending)
        # float64 on the CPU: the same values whether the pending tensors live on the training device or came back
        # from a CPU-mapped checkpoint.
        values = torch.stack([self.pending[k].detach().double().cpu() for k in keys]).clamp_min(1e-30).log().tolist()
        self.pending = {}
        # A group whose gradient is exactly zero this update (e.g. a clamped scalar) carries no signal.
        kept = [(key, value) for key, value in zip(keys, values) if value > math.log(1e-30) + 1.]
        if not kept:
            return False
        keys, values = [k for k, _ in kept], [v for _, v in kept]
        logs = []
        for key, value in zip(keys, values):
            if key not in self.fast:
                self.fast[key] = self.slow[key] = value
            self.fast[key] += (value - self.fast[key]) / self.K
            logs.append(self.fast[key] - self.slow[key])
        ratio = math.exp(sum(logs) / len(logs))
        self.last_ratio = ratio
        self.last_ratios = {key: math.exp(value) for key, value in zip(keys, logs)}
        calm = ratio < self.CALM
        permitted = True if guard is None else guard.observe(ratio, self.CALM, step, network, loss_surprised=loss_surprised)
        self.since_calm = 0 if calm else self.since_calm + 1
        if self.since_fire is not None:
            self.since_fire += 1
        if calm or not self.armed or not permitted:
            for key, value in zip(keys, values):
                self.slow[key] += (value - self.slow[key]) / (8 * self.K)
        fire = False
        if self.armed:
            abrupt = self.streak > 0 or self.since_calm <= 2 * self.K
            self.streak = self.streak + 1 if permitted and ratio > self.RISE and abrupt else 0
            if self.streak >= self.K:
                fire = True
                self.fires += 1
                self.armed = False
                self.streak = 0
                self.since_fire = 0
                # The re-open rescales each group's second moment by 1/r^2, which by itself lifts that
                # group's q by r: the new reference level includes the detector's own action.
                for key, value in zip(keys, logs):
                    self.slow[key] = self.fast[key] + max(0., value)
                self.log = (self.log + [[step, round(ratio, 3)]])[-8:]
                if guard is not None:
                    guard.after_fire()
        elif calm and self.since_fire >= 8 * self.K:
            self.armed = True
        return fire

    def diagnostics(self):
        return dict(fires=self.fires, armed=self.armed, streak=self.streak, last_ratio=self.last_ratio, log=self.log,
                    anchor_event=self.anchor_event, anchor_events=self.anchor_events)

    def state_dict(self):
        # ``pending`` holds this update's observations, folded in at the next begin_step.
        return deepcopy({k: v for k, v in self.__dict__.items()})

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.__dict__):
            raise ValueError("incompatible optimizer-surprise state")
        self.__dict__.update(deepcopy(state))


class SettledReopenGuard:
    """Opt-in R1 acquisition witness; detector ratios and thresholds stay R1's.

    A contracted explicit network ladder at the last calm observation or
    excursion onset qualifies that excursion. The witness survives an ordinary
    ladder release while R1 accumulates its sustained-jump evidence. Full-rate
    excursions follow the existing slow reference without any re-open action.
    """

    NETWORK_ROLES = frozenset(("generator", "encoder", "router", "critic"))

    def __init__(self):
        self.ka2_anchor_started = None
        self.calm_network = {}
        self.excursion = None
        self.epoch_rebases = 0

    def observe(self, ratio, calm_level, step, network, loss_surprised=False):
        network = {} if network is None else network
        if ratio < calm_level:
            self.calm_network = deepcopy(network)
            self.excursion = None
            return True
        if self.excursion is None:
            witness = {**self.calm_network, **network}
            self.excursion = {"step": step, "network": deepcopy(witness)}
        return bool(self.excursion["network"]) or loss_surprised

    def after_fire(self):
        # A re-open restores every ladder to its ceiling. It supplies no
        # contraction witness for the next event.
        self.calm_network = {}
        self.excursion = None

    def observe_epoch(self, anchor_started, detector):
        if anchor_started is not None and type(anchor_started) is not bool:
            raise ValueError("KA2 loss epoch must be a boolean or None")
        changed = self.ka2_anchor_started is False and anchor_started is True
        if changed:
            detector.fast, detector.slow, detector.pending = {}, {}, {}
            detector.streak, detector.since_calm = 0, 0
            detector.last_ratio, detector.last_ratios = None, {}
            # A known objective change has no optimizer action. Preserve real
            # fires, their refractory clock, and any existing anchor latch.
            self.calm_network, self.excursion = {}, None
            self.epoch_rebases += 1
        self.ka2_anchor_started = anchor_started
        return changed

    def state_dict(self):
        return {"schema": 1, **deepcopy(self.__dict__)}

    def check_state(self, state, *, roles, completed_steps, anchor_started):
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or type(state["schema"]) is not int or state["schema"] != 1):
            raise ValueError("invalid settled re-open guard schema")
        epoch = state["ka2_anchor_started"]
        if ((epoch is not None and type(epoch) is not bool) or epoch != anchor_started):
            raise ValueError("settled re-open guard loss epoch does not match critic state")
        if type(state["epoch_rebases"]) is not int or state["epoch_rebases"] < 0:
            raise ValueError("invalid settled re-open guard epoch counter")
        topology = {f"{i}.{j}": role for i, row in enumerate(roles) for j, role in enumerate(row)}

        def check_network(network):
            if not isinstance(network, dict):
                raise ValueError("invalid settled re-open network witness")
            for key, witness in network.items():
                if (type(key) is not str or key not in topology or not isinstance(witness, dict)
                        or witness.keys() != {"role", "scale"}
                        or witness["role"] != topology[key] or witness["role"] not in self.NETWORK_ROLES):
                    raise ValueError("settled re-open witness does not match network ownership")
                scale = witness["scale"]
                if type(scale) not in (float, int) or not math.isfinite(scale) or not 0 < scale < 1:
                    raise ValueError("settled re-open witness needs a contracted network scale")

        check_network(state["calm_network"])
        excursion = state["excursion"]
        if excursion is not None:
            if (not isinstance(excursion, dict) or excursion.keys() != {"step", "network"}
                    or type(excursion["step"]) is not int
                    or not 0 <= excursion["step"] <= completed_steps):
                raise ValueError("invalid settled re-open excursion clock")
            check_network(excursion["network"])

    def load_state_dict(self, state, *, roles, completed_steps, anchor_started):
        self.check_state(state, roles=roles, completed_steps=completed_steps,
                         anchor_started=anchor_started)
        self.__dict__.update(deepcopy({key: value for key, value in state.items() if key != "schema"}))


class LossSurprise:
    """Critic loss surprise detector for abrupt distribution shift recovery.

    Tracks short-term running loss L_fast against a calm equilibrium baseline
    L_calm with variance var_calm. While calm (z < 1.5), the baseline adapts
    slowly to track the game's equilibrium loss. When an abrupt distribution
    shift occurs, critic loss jumps, producing z = (L_fast - L_calm) / sigma_calm > 3.5.
    A confirmation streak of 2 consecutive steps triggers a re-open, while the
    baseline is frozen during excursions. The detector remains quiet at equilibrium
    and stays disarmed for a cooldown period after firing.
    """

    ALPHA_FAST = 0.5
    ALPHA_SLOW = 0.01
    Z_THRESH = 3.5
    Z_CALM = 1.5
    CONFIRMATION_STREAK = 2
    COOLDOWN = 16
    MIN_VAR = 5e-6

    _STATE_KEYS = frozenset({
        "fast",
        "calm",
        "var_calm",
        "streak",
        "armed",
        "since_calm",
        "since_fire",
        "fires",
        "last_z",
        "pending_loss",
    })

    def __init__(self):
        self.reset()

    def reset(self):
        self.fast = None
        self.calm = None
        self.var_calm = 0.01
        self.streak = 0
        self.armed = True
        self.since_calm = 0
        self.since_fire = None
        self.fires = 0
        self.last_z = 0.0
        self.pending_loss = None

    def observe(self, loss: float):
        if not math.isfinite(loss):
            return
        self.pending_loss = float(loss)

    def decide(self, step=None, *, guard=None):
        if self.pending_loss is None:
            return False
        loss = self.pending_loss
        self.pending_loss = None

        if self.calm is None:
            self.calm = loss
            self.fast = loss
            self.var_calm = max(self.MIN_VAR, 0.01)
            return False

        self.fast += self.ALPHA_FAST * (loss - self.fast)
        sigma = math.sqrt(max(self.MIN_VAR, self.var_calm))
        z = (self.fast - self.calm) / sigma
        self.last_z = z

        calm = z < self.Z_CALM
        self.since_calm = 0 if calm else self.since_calm + 1
        if self.since_fire is not None:
            self.since_fire += 1

        witness_roles = set()
        if guard is not None:
            for witness in getattr(guard, "calm_network", {}).values():
                if isinstance(witness, dict) and "role" in witness:
                    witness_roles.add(witness["role"])
            excursion = getattr(guard, "excursion", None)
            if isinstance(excursion, dict) and isinstance(excursion.get("network"), dict):
                for witness in excursion["network"].values():
                    if isinstance(witness, dict) and "role" in witness:
                        witness_roles.add(witness["role"])
        permitted = not (
            guard is not None
            and getattr(guard, "ka2_anchor_started", None) is False
            and witness_roles == {"critic"}
        )

        if calm or not self.armed or not permitted:
            delta = loss - self.calm
            self.calm += self.ALPHA_SLOW * delta
            self.var_calm = (1.0 - self.ALPHA_SLOW) * self.var_calm + self.ALPHA_SLOW * (delta * delta)
            self.var_calm = max(self.MIN_VAR, self.var_calm)

        fire = False
        if self.armed:
            if permitted and z > self.Z_THRESH:
                self.streak += 1
            else:
                self.streak = 0

            if self.streak >= self.CONFIRMATION_STREAK:
                fire = True
                self.fires += 1
                self.armed = False
                self.streak = 0
                self.since_fire = 0
                self.calm = self.fast
                if guard is not None:
                    guard.after_fire()
        elif calm and self.since_fire is not None and self.since_fire >= self.COOLDOWN:
            self.armed = True

        return fire

    @property
    def is_surprised(self):
        return self.last_z > self.Z_THRESH or self.streak > 0

    def state_dict(self):
        return deepcopy({
            "fast": self.fast,
            "calm": self.calm,
            "var_calm": self.var_calm,
            "streak": self.streak,
            "armed": self.armed,
            "since_calm": self.since_calm,
            "since_fire": self.since_fire,
            "fires": self.fires,
            "last_z": self.last_z,
            "pending_loss": self.pending_loss,
        })

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state.keys()) != self._STATE_KEYS:
            raise ValueError("incompatible loss-surprise state")
        validated = {}
        for key in ("fast", "calm", "pending_loss"):
            val = state[key]
            if val is not None:
                if type(val) not in (int, float) or not math.isfinite(val):
                    raise ValueError(f"incompatible loss-surprise state: invalid {key}")
                val = float(val)
            validated[key] = val
        if (validated["fast"] is None) != (validated["calm"] is None):
            raise ValueError("incompatible loss-surprise state: fast and calm must both be set or None")
        var_calm = state["var_calm"]
        if type(var_calm) not in (int, float) or not math.isfinite(var_calm) or var_calm < 0.0:
            raise ValueError("incompatible loss-surprise state: invalid var_calm")
        validated["var_calm"] = float(var_calm)
        last_z = state["last_z"]
        if type(last_z) not in (int, float) or not math.isfinite(last_z):
            raise ValueError("incompatible loss-surprise state: invalid last_z")
        validated["last_z"] = float(last_z)
        armed = state["armed"]
        if type(armed) is not bool:
            raise ValueError("incompatible loss-surprise state: invalid armed")
        validated["armed"] = armed
        for key in ("streak", "since_calm", "fires"):
            val = state[key]
            if type(val) is not int or val < 0:
                raise ValueError(f"incompatible loss-surprise state: invalid {key}")
            validated[key] = val
        since_fire = state["since_fire"]
        if since_fire is not None and (type(since_fire) is not int or since_fire < 0):
            raise ValueError("incompatible loss-surprise state: invalid since_fire")
        validated["since_fire"] = since_fire
        for key, value in validated.items():
            setattr(self, key, value)
