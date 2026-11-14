"""Fisher-Rao birth-death moves for a particle table (recipe ``particle_birth_death``).

Spec: reports/design-birthdeath.md (rev 2). Once per full turnover of a FIFO
reservoir R of the last N D-real rows (N = table size), every particle i is
scored at its clean centre q_i = G(z_i) with the kNN log density ratio

    x_i = d log(r_k^R(q_i) / r_k^F(q_i))

against a fake pool F (an iid N-sample of the exact sampling law; see
maybe_apply for the two deviations from the spec). Under local Poisson sampling x_i = log(rho/pi)(q_i) +
logit(Beta(k, k)), sd s_k = sqrt(2 psi'(k)). Evidence S_i accumulates over a
sign-consistent, local excursion; p_i = exp(-z_i^2 / 2) (Rayleigh tail of the
meander endpoint), Benjamini-Hochberg at q over particles with n_i >= 2,
soft-thresholded exponentiated step beta_i, Bernoulli deaths (1 - e^-beta) and
single-clone births (e^-beta - 1), paired by rank. Unmatched deaths and births
clone nothing: they stay pending until a certified partner is realised (no
normalisation parents; the ``normalised`` counter is kept at 0 for checkpoint
and receipt compatibility). Row moves copy the parent's latent (+ the package's
latent jitter), EMA latent, Adam/AMSGrad rows and A2 history. All randomness
comes from a private stream; no training stream, parameter or optimizer state
is touched except by an executed move.
"""
import math
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

from ._stats import bh_rejections
from .drift_variants import drift_features
from .row_transport import prepare_copy_history


@dataclass
class ParticleRows:
    """Explicit owners and operations for an independently sampled particle bank.

    Each row of ``table`` is one equal-mass atom, sampled uniformly with
    replacement, with clean output ``generate(table[i])``. Moving a row must
    affect that atom only: conditional banks, nonuniform routers and dense
    soft blends do not satisfy this contract. Declare those models to the
    policy as a different row semantics; evidence and birth/death reject them.

    ``optimizer`` owns the table exactly once. Its full-shaped tensor states
    (Adam/AMSGrad moments) and optional ``latent_history`` move with the row;
    scalar/global states remain shared. ``averaged_table`` is a separate,
    frozen table belonging to the serving average. The generation and critic
    feature callbacks must be deterministic and consume no training RNG.
    Feature callbacks receive samples in their original, unflattened shape
    and return a two-dimensional matrix with one feature row per sample.

    Include all modules used by the generation callback in
    ``evaluation_modules``. Birth/death temporarily evaluates those modules
    without changing any of their original train/eval flags. A feature callback
    owns its own evaluation context (``ScalarHeadFeatures`` does this).
    """

    table: torch.Tensor
    optimizer: torch.optim.Optimizer
    averaged_table: torch.Tensor
    generate: Callable[[torch.Tensor], torch.Tensor]
    critic_features: Callable[[torch.Tensor], torch.Tensor] | None = None
    evaluation_modules: tuple = ()
    controller: object | None = None
    completed_steps: Callable[[], int] = field(default=lambda: 0)
    semantics: str = "independent"

    def __post_init__(self):
        if self.semantics != "independent":
            raise ValueError("particle birth/death requires independent uniformly sampled rows; "
                             "conditional and soft-routed particle banks are unsupported")
        table = self.table
        if (not isinstance(table, torch.Tensor) or table.ndim != 2
                or not all(table.shape) or table.layout != torch.strided
                or not table.is_floating_point() or not table.requires_grad or not table.is_leaf):
            raise ValueError("particle birth/death needs a trainable leaf particle table [rows, latent_dim]")
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("particle table optimizer must be a torch optimizer")
        owners = sum(p is table for group in self.optimizer.param_groups for p in group["params"])
        if owners != 1:
            raise ValueError("particle table must belong to its optimizer exactly once")
        history = getattr(self.optimizer, "latent_history", None)
        if history is not None and (not isinstance(history, torch.Tensor) or history.shape != table.shape
                                    or history.device != table.device or history.dtype != table.dtype):
            raise ValueError("optimizer latent_history must match its particle table")
        average = self.averaged_table
        if (not isinstance(average, torch.Tensor) or average.shape != table.shape
                or average.device != table.device or average.dtype != table.dtype or average.requires_grad):
            raise ValueError("averaged particle table must be frozen and match the table's shape, device and dtype")
        if average.untyped_storage().data_ptr() == table.untyped_storage().data_ptr():
            raise ValueError("averaged particle table must have separate storage from training weights")
        if not callable(self.generate) or not callable(self.completed_steps):
            raise TypeError("generation and completed_steps must be callbacks")
        if self.critic_features is not None and not callable(self.critic_features):
            raise TypeError("critic_features must be a callback")
        self.evaluation_modules = tuple(self.evaluation_modules)
        if any(not isinstance(module, torch.nn.Module) for module in self.evaluation_modules):
            raise TypeError("evaluation_modules must contain torch modules")

    @contextmanager
    def evaluating(self):
        modes = dict.fromkeys(module for root in self.evaluation_modules for module in root.modules())
        modes = [(module, module.training) for module in modes]
        try:
            for root in self.evaluation_modules:
                root.eval()
            yield
        finally:
            for module, flag in modes:
                module.training = flag


class ScalarHeadFeatures:
    """The native E22 critic-feature callback, including checkpointed head discovery.

    Concatenate learned inputs to scalar ``nn.Linear`` score heads, preserving
    firing order. Raw-sample skip heads are excluded; a critic with no learned
    feature space is rejected. Custom critics may provide another deterministic
    feature callback directly instead of using this adapter.
    """

    def __init__(self, critic):
        if not isinstance(critic, torch.nn.Module):
            raise TypeError("critic must be a torch module")
        self.critic = critic
        self._linears = [m for m in critic.modules() if isinstance(m, torch.nn.Linear)]
        if not self._linears:
            raise ValueError("birth_death_space='critic' needs a critic with an nn.Linear score head")
        self._heads = None

    @staticmethod
    def _is_raw(feature, raw):
        dx, width, raw = raw.shape[1], feature.shape[1], raw.to(feature.dtype)
        if width == dx:
            return torch.equal(feature, raw)
        return width > dx and (torch.equal(feature[:, :dx], raw) or torch.equal(feature[:, -dx:], raw))

    @torch.no_grad()
    def __call__(self, x, chunk=8192):
        fired = []
        handles = [m.register_forward_pre_hook(lambda module, inputs: fired.append((module, inputs[0].detach()))
                                               if self._heads is None or module in self._heads else None) for m in self._linears]
        # Preserve mixed module flags as well as the root's flag. Native critics
        # have uniform flags, for which this is identical to the former adapter.
        modes = [(module, module.training) for module in self.critic.modules()]
        self.critic.eval()
        captured = []
        try:
            for start in range(0, len(x), chunk):
                fired.clear()
                self.critic(x[start:start + chunk])
                if self._heads is None:
                    raw = x[start:start + chunk].flatten(1)
                    heads = [(m, f) for m, f in fired if m.out_features == 1] or fired[-1:]
                    heads = [(m, f) for m, f in heads if not self._is_raw(f, raw)]
                    if not heads:
                        raise ValueError("birth_death_space='critic': every scalar score head of the critic reads the raw sample "
                                         "(or the raw sample concatenated with something), so there is no feature space")
                    self._heads = list(dict.fromkeys(m for m, _ in heads))
                by = {id(m): f for m, f in fired}
                captured.append(torch.cat([by[id(m)] for m in self._heads], 1))
        finally:
            for handle in handles:
                handle.remove()
            for module, flag in modes:
                module.training = flag
        return torch.cat(captured).double()

    def state_dict(self):
        names = {id(module): name for name, module in self.critic.named_modules()}
        return {"heads": None if self._heads is None else [names[id(module)] for module in self._heads]}

    def check_state(self, state):
        if not isinstance(state, dict) or set(state) != {"heads"}:
            raise ValueError("invalid critic-feature state")
        heads = state["heads"]
        names = dict(self.critic.named_modules())
        if heads is not None and (not isinstance(heads, list) or not heads or any(not isinstance(name, str) for name in heads)
                                  or len(set(heads)) != len(heads)
                                  or any(name not in names or names[name] not in self._linears for name in heads)):
            raise ValueError("critic-feature heads do not match the critic")

    def load_state_dict(self, state):
        self.check_state(state)
        names = dict(self.critic.named_modules())
        self._heads = None if state["heads"] is None else [names[name] for name in state["heads"]]


def _knn(query, points, k, exclude=None, chunk=2048, shortlist_dtype=None):
    """Exact k smallest distances (ascending) and indices from each query row
    to ``points``; ``exclude[i]`` (optional) is a point index skipped for query i.
    Shortlist 2k+2 candidates with the matmul cdist (optionally in a cheaper dtype), then recompute exactly.
    Chunked and recomputed exactly, so it is not the dense ``_stats.knn_excluding_self``."""
    m = min(len(points) - (exclude is not None), 2 * k + 2)
    ds, ids = [], []
    points_short = points if shortlist_dtype is None else points.to(shortlist_dtype)
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        dist = torch.cdist(q if shortlist_dtype is None else q.to(shortlist_dtype), points_short,
                           compute_mode="use_mm_for_euclid_dist")
        if exclude is not None:
            dist.scatter_(1, exclude[start:start + chunk, None], float("inf"))
        idx = dist.topk(m, dim=1, largest=False).indices
        exact = (q[:, None, :] - points[idx]).norm(dim=2)
        exact, order = exact.sort(1)
        ds.append(exact[:, :k])
        ids.append(idx.gather(1, order)[:, :k])
    return torch.cat(ds), torch.cat(ids)


def _levina_bickel(dist):
    """MacKay-Ghahramani pooled Levina-Bickel MLE from ascending kNN distances [n, k]."""
    k = dist.shape[1]
    ok = dist[:, 0] > 0
    if not bool(ok.any()):
        return float("nan"), 0
    t = dist[ok].double()
    inverse = (t[:, -1:] / t[:, :-1]).log().sum(1) / (k - 1)
    mean = float(inverse.mean())
    return (1. / mean if mean > 0 else float("nan")), int(ok.sum())


def _normal_score(x, k):
    """Exact null probability transform of a kNN log volume ratio: under the null
    sigmoid(x) ~ Beta(k, k), so w = Phi^-1(I_sigmoid(x)(k, k)) ~ N(0, 1) exactly.
    I_u(k, k) = P(Binomial(2k-1, u) >= k), summed in log space on the smaller tail."""
    a = -x.abs()                                   # use the lower tail, restore the sign
    log_u, log_1u = torch.nn.functional.logsigmoid(a), torch.nn.functional.logsigmoid(-a)
    j = torch.arange(k, 2 * k, dtype=torch.float64, device=x.device)
    log_binom = (torch.lgamma(torch.tensor(2. * k, dtype=torch.float64, device=x.device))
                 - torch.lgamma(j + 1) - torch.lgamma(2 * k - j))
    log_cdf = torch.logsumexp(log_binom + j * log_u[:, None] + (2 * k - 1 - j) * log_1u[:, None], dim=1)
    w = torch.special.ndtri(log_cdf.exp().clamp(1e-300, .5))
    return torch.where(x > 0, -w, w)


class ParticleBirthDeath:
    """Checkpointable mass balancing on an explicit ``ParticleRows`` contract.

    Call ``observe_real`` on critic real batches, then ``maybe_apply`` after
    the generator/table optimizer step, noise update and EMA update. Moves
    copy the EMA rows as well as training rows, then re-anchor settling tests
    and reset row-gradient evidence at ``moved_rows`` before selecting served
    weights. The policy coordinates those subsequent hooks.
    """

    Q = .05          # BH false-discovery rate per evaluation; dimension-test level

    def __init__(self, rows, seed, *, space="data", isolation=False, feature_scale="none",
                 cells=None):
        if not isinstance(rows, ParticleRows):
            raise TypeError("particle birth/death requires an explicit ParticleRows contract")
        if space not in ("data", "critic") or feature_scale not in ("none", "std"):
            raise ValueError("invalid particle birth/death space or feature scale")
        if type(isolation) is not bool:
            raise ValueError("particle birth/death isolation must be boolean")
        if (isolation or feature_scale == "std") and space != "critic":
            raise ValueError("isolation and feature standardisation require critic feature space")
        if space == "critic" and rows.critic_features is None:
            raise ValueError("birth_death_space='critic' needs an explicit critic-feature callback")
        self.rows = rows
        self.N = n = rows.table.shape[0]
        d_z = rows.table.shape[1] if rows.table.ndim >= 2 else 0
        self.k = min(math.ceil((math.log2(max(n, 1) / self.Q) + 1) / 2), max(1, n - 2))
        self.capacity_bypassed = self.N <= 2 * self.k + d_z
        if not self.capacity_bypassed:
            if self.k < 4:
                raise ValueError("particle_birth_death needs at least 6 particles")
            if isolation and self.k >= (n + 1) // 2:
                raise ValueError("birth/death isolation needs enough particles for leave-one-out neighbours in its reference half")
        self.s_k = math.sqrt(2. * float(torch.polygamma(1, torch.tensor(float(max(1, self.k)), dtype=torch.float64))))
        self.dry_run = False   # test hook: evaluate, never move
        device, dtype = rows.table.device, rows.table.dtype
        self.reservoir = None  # [N, D_out], allocated on the first batch
        self.fill = 0
        self.cursor = 0
        self.rows_since_eval = 0
        self.S = torch.zeros(n, device=device, dtype=torch.float64)   # evidence in nats (step size)
        self.W = torch.zeros(n, device=device, dtype=torch.float64)   # evidence in normal scores (test)
        self.n = torch.zeros(n, device=device, dtype=torch.long)
        self.anchor = None     # [N, D_out]
        self.radius = torch.zeros(n, device=device, dtype=dtype)
        self.pending = torch.zeros(n, device=device, dtype=torch.bool)
        self.space = space
        self.isolation = isolation
        self.feature_scale = feature_scale
        self.cells = cells
        self.iso_log = []       # [step, flagged rows, acted] of the latest evaluations (diagnostics only)
        self.sample_shape = None
        if self.space == "critic":
            # locality (anchor, radius, stale resets) lives in the table's own latent space; evidence lives in critic features
            self.anchor = torch.zeros(n, rows.table.shape[1], device=device, dtype=dtype)
        self.stream = torch.Generator(device=device).manual_seed(seed)
        self.counters = {k: 0 for k in ("evals", "dim_skips", "discoveries", "realised_deaths",
                                        "realised_births", "stays", "moves", "matched", "normalised",
                                        "waited_deaths", "waited_births", "stale_resets")}
        if self.isolation:
            self.counters.update({k: 0 for k in ("iso_evals", "iso_flagged", "iso_acted", "iso_skipped", "iso_moves", "iso_dup_skips")})
        self.last = {}
        self.moved_rows = None  # rows moved by the latest maybe_apply

    # --------------------------------------------------------------- reservoir
    @torch.no_grad()
    def observe_real(self, real):
        if (not isinstance(real, torch.Tensor) or real.ndim < 2 or not len(real)
                or real.device != self.rows.table.device or real.dtype != self.rows.table.dtype):
            raise ValueError("real samples must be a nonempty batch on the particle table device and dtype")
        shape = tuple(real.shape[1:])
        if self.sample_shape is not None and shape != self.sample_shape:
            raise ValueError("real sample shape must remain unchanged for the birth/death reservoir")
        x = real.detach().flatten(1)
        if self.sample_shape is None:
            self.sample_shape = shape
        if self.reservoir is not None and self.reservoir.shape[1] != x.shape[1]:
            raise ValueError("real sample shape does not match the restored birth/death reservoir")
        if self.reservoir is None:
            self.reservoir = torch.zeros(self.N, x.shape[1], device=x.device, dtype=x.dtype)
            if self.space == "data":
                self.anchor = torch.zeros_like(self.reservoir)
        x = x[-self.N:]
        end = self.cursor + len(x)
        if end <= self.N:
            self.reservoir[self.cursor:end] = x
        else:
            split = self.N - self.cursor
            self.reservoir[self.cursor:] = x[:split]
            self.reservoir[:end - self.N] = x[split:]
        self.cursor = end % self.N
        self.fill = min(self.N, self.fill + len(x))
        self.rows_since_eval += len(real)

    def ready(self):
        return self.fill == self.N and self.rows_since_eval >= self.N

    # --------------------------------------------------------------- sampling law
    @torch.no_grad()
    def _nearest_other(self, z, points):
        """Half-NN radius as DV12's perturb_latent (zero distances masked), matmul shortlist."""
        d, _ = _knn(z, points, min(4, len(points) - 1) + 1)
        d = d.masked_fill(d == 0, float("inf"))
        nearest = d.min(1).values
        return torch.where(torch.isfinite(nearest), nearest * .5, torch.zeros_like(nearest))

    @torch.no_grad()
    def _jitter(self, latent, noise):
        c = self.rows.controller
        features = None if c is None else drift_features(c.variant)
        if features is None or not features.latent_bandwidth:
            return torch.zeros_like(latent)
        trust = c.support_trust if features.support_trust else 1.
        displacement = c.latent_bandwidth * trust * noise
        if features.local_support:
            radius = self._nearest_other(latent, self.rows.table.detach())
            norm = displacement.norm(dim=1)
            displacement = displacement * (radius / norm.clamp_min(1e-20)).clamp_max(1.).unsqueeze(1)
        return displacement

    # --------------------------------------------------------------- critic features
    @torch.no_grad()
    def _features(self, x):
        """Feature callback on unflattened sample rows; no gradients."""
        x = x.reshape(len(x), *self.sample_shape)
        feature = self.rows.critic_features(x)
        if (not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[0] != len(x)
                or not feature.shape[1] or not feature.is_floating_point() or feature.device != x.device):
            raise ValueError("critic-feature callback must return floating [samples, features] on the sample device")
        return feature.detach().double()

    @staticmethod
    def _standardise(q, F, R):
        """Divide every critic feature by its standard deviation on the reference half of the real reservoir (the even rows, the half the support test
        uses as reference): invariant to the positive rescaling of any hidden unit (the exact symmetry group of ReLU-type networks: same critic function,
        different Euclidean distances) and a function of the reference half only, so the split-conformal validity is untouched. A unit without spread on
        the reference half (dead on every real row, or constant to 1e-8 of the largest spread) has no reference scale and is dropped: keeping its raw
        scale would not be invariant, and a scale taken from the queries would leak them into the score."""
        s = R[0::2].std(0)
        s = torch.where(s > float(s.max()) * 1e-8, s, torch.full_like(s, float("inf")))
        return q / s, F / s, R / s

    # --------------------------------------------------------------- support test
    @torch.no_grad()
    def _isolated(self, q, R, k, fast):
        """Rows of the table that have no real support, in the critic's feature space: a split-conformal isolation test.

        Score of a point u: a(u) / b(u), with a(u) the distance to its k-th nearest point of the reference half R1 of the real reservoir and b(u)
        the median, over its k nearest reference points, of their own leave-one-out k-th neighbour radius (the local real scale: a sparse
        region is compared with itself, so a rare component is not flagged for being sparse). The other half R2 of the reservoir is exchangeable
        with the fake law when that law equals the real law, so p_i = (1 + #{R2 scores >= score_i}) / (1 + |R2|) is a valid p-value for row i
        (a row scored at its clean centre is, if anything, closer to the bulk than a draw: conservative). Benjamini-Hochberg at Q over the table
        (conformal p-values with a shared calibration set are positively dependent, for which BH keeps its level). The local scale is floored at 1e-3 of the
        median positive leave-one-out radius so that exact duplicates in the reservoir cannot make it zero (numerical only).
        Duplicate guard: exchangeability fails when the reservoir contains exact copies of its own rows (a finite data pool sampled with replacement: a
        copy of a reference point lands in the calibration half and looks perfectly supported, while a fresh row does not: P(p <= 1e-3) was 21 x the
        nominal level at 1,000 distinct rows); nothing is flagged when more than Q of the reservoir's feature rows are copies of another row."""
        R1, R2 = R[0::2], R[1::2]
        if 1. - len(torch.unique(R, dim=0)) / len(R) > self.Q:
            self.counters["iso_dup_skips"] += 1
            return torch.zeros(len(q), dtype=torch.bool, device=q.device)
        rho = _knn(R1, R1, k, exclude=torch.arange(len(R1), device=R.device), shortlist_dtype=fast)[0][:, -1]
        positive = rho[rho > 0]
        floor = float(positive.median()) * 1e-3 if len(positive) else 1.
        def score(u):
            d, ix = _knn(u, R1, k, shortlist_dtype=fast)
            return d[:, -1] / rho[ix].sort(dim=1).values[:, (k - 1) // 2].clamp_min(floor)      # the lower median (sort, not median: deterministic on CUDA)
        null = score(R2).sort().values
        s = score(q)
        p = (1. + (len(null) - torch.searchsorted(null, s)).double()) / (1. + len(null))
        ps = p.sort().values
        j = bh_rejections(ps, self.Q)
        if not j:
            return torch.zeros(len(p), dtype=torch.bool, device=p.device)
        return p <= ps[j - 1]

    @torch.no_grad()
    def _isolation_pick(self, flagged, child):
        """Rows to re-draw and their parents: every flagged row that the ordinary moves did not just re-draw becomes a clone of a uniformly drawn
        unflagged row among those at most twice as far from it (in the table's own latent space) as its nearest unflagged row: the mass without
        support goes back into the neighbourhood that has support nearest to it, and reaches into that neighbourhood's bulk instead of only its
        rim (the k nearest unflagged rows of a stray are the rim rows of the nearest mode; their clones sit at the rim and are flagged again).
        The candidates are also limited to the k^2 nearest unflagged rows: in a high-dimensional latent space "twice as far as the nearest" contains
        most of the table (distances concentrate) and the rule would become a uniform draw, which shuffles mass between modes; the rank cap keeps it
        local in any dimension and does not bind in the low-dimensional tables where the ball is small.
        Nothing is done while more than Q of the table is flagged (most of the table still in transit has no support to clone from; the same
        level as the test)."""
        c, n_flag = self.counters, int(flagged.sum())
        acted = 0 < n_flag <= self.Q * self.N
        c["iso_evals"] += 1
        c["iso_flagged"] += n_flag
        c["iso_acted" if acted else "iso_skipped"] += bool(n_flag)
        self.iso_log = (self.iso_log + [[self.rows.completed_steps() + 1, n_flag, int(acted)]])[-30:]
        self.last.update(iso_flagged=n_flag, iso_moves=0)
        empty = child[:0]
        if not acted:
            return empty, empty
        dead = flagged.clone()
        dead[child] = False
        keep = ~flagged
        keep[child] = False
        dead, keep = dead.nonzero().flatten(), keep.nonzero().flatten()
        if not len(dead) or not len(keep):
            return empty, empty
        z = self.rows.table.detach()
        parent, step, cap = [], max(1, (1 << 24) // len(keep)), min(self.k ** 2, len(keep))      # chunks of about 16M distances; at most k^2 candidates
        for start in range(0, len(dead), step):
            dist = torch.cdist(z[dead[start:start + step]], z[keep])                                                  # [chunk, keep]
            limit = torch.minimum(2. * dist.min(1, keepdim=True).values, dist.topk(cap, dim=1, largest=False).values[:, -1:])
            ball = dist <= limit
            draw = torch.rand(dist.shape, device=dist.device, generator=self.stream)
            parent.append(keep[torch.where(ball, draw, torch.full_like(draw, -1.)).argmax(1)])
        parent = torch.cat(parent)
        c["iso_moves"] += len(dead)
        self.last["iso_moves"] = len(dead)
        return dead, parent

    # --------------------------------------------------------------- evaluation
    def _check_generated(self, sample):
        table = self.rows.table
        if (not isinstance(sample, torch.Tensor) or sample.ndim < 2 or len(sample) != self.N
                or not sample.is_floating_point() or sample.device != table.device or sample.dtype != table.dtype):
            raise ValueError("generation callback must return one floating sample per table row on the table device and dtype")
        if self.sample_shape is None:      # legacy checkpoints did not record this shape
            shape = tuple(sample.shape[1:])
            if math.prod(shape) != self.reservoir.shape[1]:
                raise ValueError("generated sample shape does not match the birth/death reservoir")
            self.sample_shape = shape
        if tuple(sample.shape[1:]) != self.sample_shape:
            raise ValueError("generated sample shape must match the observed real sample shape")

    @torch.no_grad()
    def maybe_apply(self, sigma_out):
        """Run one evaluation (and its moves) when the reservoir has turned over."""
        self.moved_rows = None
        d_z = self.rows.table.shape[1] if self.rows.table.ndim >= 2 else 0
        if self.capacity_bypassed or self.N <= 2 * self.k + d_z:
            return None
        if self.cells is not None and self.N <= self.cells:
            return None
        if not self.ready():
            return None
        self.rows_since_eval = 0
        self.counters["evals"] += 1
        with self.rows.evaluating():
            z = self.rows.table.detach()
            # [dev from spec rev 2] F is an iid N-sample of the exact sampling law (table rows drawn
            # with replacement, as prior.sample does), not one draw per particle: R is iid from pi,
            # so under rho = pi the two pools are identically distributed and both x and the
            # dimension test are symmetric (a stratified F failed the null test: d_R 1.89 vs d_F 2.10).
            pick = torch.randint(self.N, (self.N,), device=z.device, generator=self.stream)
            latent = z[pick]
            noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=self.stream)
            jitter_delta = self._jitter(latent, noise)
            y = self.rows.generate(latent + jitter_delta)
            self._check_generated(y)
            eps = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=self.stream)
            u = torch.rand(self.N, device=z.device, dtype=torch.float64, generator=self.stream)
            if sigma_out:
                y = y + float(sigma_out) * eps
            q_raw = self.rows.generate(z)
            self._check_generated(q_raw)
            q = q_raw.flatten(1)
        F, R, n, k = y.flatten(1), self.reservoir, self.N, self.k
        own = torch.arange(n, device=q.device)
        if self.space == "critic":
            # evidence in the critic's feature space; locality (anchor, radius, stale resets) in the table's own latent space
            q_loc, zpool = z.clone(), latent + jitter_delta      # clone: the moves below overwrite the table rows in place
            rad_loc = _knn(z, zpool, k)[0][:, -1]
            q, F, R = self._features(q_raw), self._features(y), self._features(self.reservoir)
            centre = R.mean(0, keepdim=True)
            q, F, R = q - centre, F - centre, R - centre
            fast = torch.float32
            if self.feature_scale == "std":
                q, F, R = self._standardise(q, F, R)
            flagged = self._isolated(q, R, k, fast) if self.isolation else None
        else:
            q_loc, rad_loc = q, None
            fast = None
        rR, _ = _knn(q, R, k, shortlist_dtype=fast)
        # [dev from spec rev 2] no own-draw exclusion and no log(N/(N-1)): q_i is a fixed point
        # given the table and F is an iid N-sample of rho, so E[#F in a ball] = N * int rho (own
        # kernel included), exactly as R's under rho = pi. Excluding i's draw biased x toward deficit
        # by the own-kernel share of rho(q_i) (unit test (a): 84% of null evaluations had discoveries).
        rF, _ = _knn(q, F, k, shortlist_dtype=fast)
        dR, nR = _levina_bickel(_knn(R, R, k, exclude=own, shortlist_dtype=fast)[0])
        dF, nF = _levina_bickel(_knn(F, F, k, exclude=own, shortlist_dtype=fast)[0])
        D_out = q.shape[1]
        last = dict(step=self.rows.completed_steps() + 1, k=k, d_R=dR, d_F=dF)
        self.last = last
        # BD-GUARD FIX (one mechanism): the kNN-ratio null already absorbs small
        # dimension mismatches, and at N=20000 the asymptotic-variance
        # significance test fires on every evaluation (spec rev 2 R2), so it is
        # not a statistically bounded guard here. Use the data dimension d_R
        # (the reference law) directly, clamped to [1, D_out]. Skip only when
        # d_R is undefined (no bounded dimension can be formed); record d_F as
        # a diagnostic and keep counting such skips in dim_skips.
        if not math.isfinite(dR):
            self.counters["dim_skips"] += 1
            last["skip"] = "dimension undefined"
            return last
        d = min(max(dR, 1.), D_out)   # data dimension only; never skip on significance
        last["d"] = d
        rk_R, rk_F = rR[:, -1].double(), rF[:, -1].double()
        x = d * (rk_R / rk_F).log()
        finite = torch.isfinite(x)
        x = torch.where(finite, x, torch.zeros_like(x))
        # ---- sequential evidence with sign-flip / locality restarts
        # [dev from spec rev 2] the test runs on exact normal scores w of each term (x's null law
        # logit Beta(k,k) has logistic tails: Gaussian/Rayleigh on raw nats gave P(p<=.01) = .017
        # in the null unit test); S (nats) runs in lock-step and only sets the step size beta.
        w = torch.where(finite, _normal_score(x, k), torch.zeros_like(x))
        S, W, cnt = self.S, self.W, self.n
        moved_out = (q_loc - self.anchor).norm(dim=1) > self.radius
        # [dev from spec rev 2, R3 reverted] a sign flip ends the excursion and the flipping term is
        # discarded (n = 0): keeping it starts the next excursion with a term selected for
        # |w| > |W_prev|, which made the Rayleigh p anti-conservative (null test: P(p<=.01) = .017).
        # A walk restarted from 0 and conditioned on its sign is the meander the p-value assumes.
        flip = finite & (cnt > 0) & ~moved_out & (torch.sign(W + w) != torch.sign(W))
        restart = finite & ((cnt == 0) | moved_out) & ~flip
        extend = finite & ~restart & ~flip
        S.copy_(torch.where(restart, x, torch.where(extend, S + x, torch.where(flip, torch.zeros_like(S), S))))
        W.copy_(torch.where(restart, w, torch.where(extend, W + w, torch.where(flip, torch.zeros_like(W), W))))
        cnt.copy_(torch.where(restart, torch.ones_like(cnt), torch.where(extend, cnt + 1,
                                                                         torch.where(flip, torch.zeros_like(cnt), cnt))))
        self.anchor[restart] = q_loc[restart].to(self.anchor.dtype)
        self.radius[restart] = (rF[restart, -1] if rad_loc is None else rad_loc[restart]).to(self.radius.dtype)
        # ---- BH over eligible particles
        eligible = finite & (cnt >= 2)
        m = int(eligible.sum())
        zscore = W.abs() / cnt.clamp_min(1).double().sqrt()
        p = torch.exp(-.5 * zscore.square())
        discover = torch.zeros_like(eligible)
        zstar = None
        if m:
            j = bh_rejections(p[eligible].sort().values, self.Q)
            if j:
                cutoff = j * self.Q / m
                zstar = math.sqrt(-2. * math.log(cutoff))
                discover = eligible & (p <= cutoff)
        beta = torch.zeros_like(S)
        if zstar is not None:
            # soft threshold in nats (z* s_k sqrt(n)); direction from the test statistic W
            size = (S.abs() - zstar * self.s_k * cnt.double().sqrt()).clamp_min(0.)
            size = torch.where(torch.sign(S) == torch.sign(W), size, torch.zeros_like(size))
            beta = torch.where(discover, torch.sign(W) * size / cnt.clamp_min(1).double(), beta).clamp_min(-50.)
        prob = torch.where(beta > 0, 1 - torch.exp(-beta), (torch.exp(-beta) - 1).clamp_max(1.))
        discover = discover & (beta != 0)
        realised = discover & (u < prob)
        stay = discover & ~realised
        deaths = (realised & (beta > 0)).nonzero().flatten()
        births = (realised & (beta < 0)).nonzero().flatten()
        deaths = deaths[beta[deaths].argsort(descending=True, stable=True)]
        births = births[beta[births].argsort(stable=True)]
        pairs = min(len(deaths), len(births))
        # BD-PAIR FIX: evidence-certified transport only (st3 paired design).
        # The former normalisation-parent rule cloned neutral rows into dead
        # sites; native evidence shows neutral parents are ~85% of all moves
        # and precision/coverage fall with them. Unmatched deaths wait for a
        # certified deficit birth instead of taking a neutral parent.
        child, parent = deaths[:pairs], births[:pairs]
        waited_deaths = len(deaths) - len(child)
        waited_births = len(births) - pairs
        self.pending.zero_()
        self.pending[deaths[len(child):]] = True
        self.pending[births[pairs:]] = True
        spent = stay.clone()
        spent[child] = True
        spent[births[:pairs]] = True
        c = self.counters
        c["discoveries"] += int(discover.sum())
        c["realised_deaths"] += len(deaths)
        c["realised_births"] += len(births)
        c["stays"] += int(stay.sum())
        c["waited_deaths"] += waited_deaths
        c["waited_births"] += waited_births
        last.update(eligible=m, discoveries=int(discover.sum()), zstar=zstar,
                    excess=int((discover & (beta > 0)).sum()), deficit=int((discover & (beta < 0)).sum()),
                    realised_deaths=len(deaths), realised_births=len(births), matched=pairs,
                    normalised=0, waited_deaths=waited_deaths, waited_births=waited_births,
                    moves=0 if self.dry_run else len(child), mean_abs_x=float(x[finite].abs().mean()) if bool(finite.any()) else None)
        if self.dry_run:
            return last
        self.S[spent] = 0.
        self.W[spent] = 0.
        self.n[spent] = 0
        if len(child):
            self._move(child, parent)
        iso_child = iso_parent = child[:0]
        if self.isolation and self.space == "critic":
            iso_child, iso_parent = self._isolation_pick(flagged, child)
            if len(iso_child):
                self._move(iso_child, iso_parent)
                last["moves"] = len(child) + len(iso_child)      # the policy re-anchors the tester and resets row evidence of every moved row
        if len(child) or len(iso_child):
            moved = torch.cat((child, iso_child))
            self.moved_rows = moved
            sites = torch.cat((q_loc[child], q_loc[parent], q_loc[iso_child], q_loc[iso_parent]))
            stale = torch.zeros_like(self.n, dtype=torch.bool)
            step = max(1, (1 << 24) // max(1, len(sites)))              # anchors in chunks: the block is [chunk, sites], never [N, sites] (memory ~ N * moves)
            for start in range(0, len(self.anchor), step):
                block = torch.cdist(self.anchor[start:start + step], sites, compute_mode="use_mm_for_euclid_dist")
                stale[start:start + step] = (block <= self.radius[start:start + step, None]).any(1)
            stale &= self.n > 0
            c["stale_resets"] += int(stale.sum())
            self.S[stale] = 0.
            self.W[stale] = 0.
            self.n[stale] = 0
            self.S[moved] = 0.
            self.W[moved] = 0.
            self.n[moved] = 0
            c["moves"] += len(child)
            c["matched"] += pairs
        return last

    @torch.no_grad()
    def _move(self, child, parent):
        table, average = self.rows.table, self.rows.averaged_table
        history = prepare_copy_history(table, self.rows.optimizer, child, parent)
        zp = table.detach()[parent]
        noise = torch.randn(zp.shape, device=zp.device, dtype=zp.dtype, generator=self.stream)
        delta = self._jitter(zp, noise)
        table[child] = zp + delta
        average[child] = average[parent] + delta
        for value, source in history:
            value[child] = source

    # --------------------------------------------------------------- state
    _TENSORS = ("reservoir", "S", "W", "n", "anchor", "radius", "pending")

    def diagnostics(self):
        return dict(k=self.k, s_k=self.s_k, fill=self.fill, rows_since_eval=self.rows_since_eval,
                    counters=dict(self.counters), last=self.last,
                    in_excursion=int((self.n > 0).sum()), eligible_now=int((self.n >= 2).sum()),
                    **({"iso_recent": list(self.iso_log)} if self.isolation else {}))

    def state_dict(self):
        feature_callback = self.rows.critic_features
        feature_state = (feature_callback.state_dict() if self._checkpointed_features() else None)
        return dict(**{k: None if getattr(self, k) is None else getattr(self, k).detach().clone() for k in self._TENSORS},
                    fill=self.fill, cursor=self.cursor, rows_since_eval=self.rows_since_eval,
                    counters=dict(self.counters), last=dict(self.last), stream=self.stream.get_state().clone(),
                    sample_shape=self.sample_shape, feature_state=deepcopy(feature_state),
                    iso_log=deepcopy(self.iso_log), config=self._config(), dry_run=self.dry_run,
                    moved_rows=None if self.moved_rows is None else self.moved_rows.clone())

    def _config(self):
        return {"table_shape": tuple(self.rows.table.shape), "space": self.space,
                "isolation": self.isolation, "feature_scale": self.feature_scale}

    def _checkpointed_features(self):
        callback = self.rows.critic_features
        return all(callable(getattr(callback, method, None)) for method in ("state_dict", "check_state", "load_state_dict"))

    def load_state_dict(self, state):
        self.check_state(state)
        for key in self._TENSORS:
            value = state[key]
            setattr(self, key, None if value is None else value.clone().to(self.S.device))
        self.fill, self.cursor, self.rows_since_eval = int(state["fill"]), int(state["cursor"]), int(state["rows_since_eval"])
        self.counters, self.last = dict(state["counters"]), dict(state["last"])
        self.stream.set_state(state["stream"].cpu())
        self.sample_shape = state.get("sample_shape")
        self.iso_log = deepcopy(state.get("iso_log", []))
        self.moved_rows = None if state.get("moved_rows") is None else state["moved_rows"].clone().to(self.S.device)
        self.dry_run = state.get("dry_run", False)
        if "feature_state" in state and self._checkpointed_features():
            self.rows.critic_features.load_state_dict(state["feature_state"])

    def check_state(self, state):
        keys = set(self._TENSORS) | {"fill", "cursor", "rows_since_eval", "counters", "last", "stream"}
        metadata = {"sample_shape", "feature_state", "iso_log", "config", "moved_rows", "dry_run"}
        if not isinstance(state, dict) or set(state) not in (keys, keys | metadata):
            raise ValueError("invalid birth-death state")
        for key in ("S", "W", "n", "radius", "pending"):
            if (not isinstance(state[key], torch.Tensor) or state[key].shape != getattr(self, key).shape
                    or state[key].dtype != getattr(self, key).dtype):
                raise ValueError(f"birth-death {key} does not match the table")
        for key in ("reservoir", "anchor"):
            if state[key] is not None and (not isinstance(state[key], torch.Tensor) or state[key].ndim != 2
                                         or state[key].shape[0] != self.N or not state[key].is_floating_point()
                                         or state[key].dtype != self.rows.table.dtype):
                raise ValueError(f"birth-death {key} does not match the table")
        reservoir, anchor = state["reservoir"], state["anchor"]
        if self.space == "critic" and (anchor is None or anchor.shape != self.rows.table.shape):
            raise ValueError("birth-death anchor does not match the latent table")
        if self.space == "data" and ((reservoir is None) != (anchor is None)
                                      or (reservoir is not None and reservoir.shape != anchor.shape)):
            raise ValueError("birth-death data anchors do not match the reservoir")
        for key in ("fill", "cursor", "rows_since_eval"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid birth-death {key}")
        if (state["fill"] > self.N or state["cursor"] >= self.N
                or (reservoir is None and (state["fill"] or state["rows_since_eval"]))):
            raise ValueError("invalid birth-death reservoir counters")
        if not isinstance(state["counters"], dict) or not isinstance(state["last"], dict):
            raise ValueError("invalid birth-death counters or diagnostics")
        if "config" in state:
            if state["config"] != self._config():
                raise ValueError("birth-death state configuration does not match the particle rows")
            shape = state["sample_shape"]
            if shape is not None and (not isinstance(shape, tuple) or not shape
                                      or any(type(size) is not int or size <= 0 for size in shape)
                                      or (reservoir is not None and math.prod(shape) != reservoir.shape[1])):
                raise ValueError("invalid birth-death sample shape")
            if not isinstance(state["iso_log"], list):
                raise ValueError("invalid birth-death isolation log")
            moved = state["moved_rows"]
            if moved is not None and (not isinstance(moved, torch.Tensor) or moved.ndim != 1
                                       or moved.dtype != torch.long or bool((moved < 0).any()) or bool((moved >= self.N).any())):
                raise ValueError("invalid birth-death moved rows")
            if type(state["dry_run"]) is not bool:
                raise ValueError("invalid birth-death dry-run state")
            if self._checkpointed_features():
                self.rows.critic_features.check_state(state["feature_state"])
            elif state["feature_state"] is not None:
                raise ValueError("birth-death feature state needs a checkpointable critic-feature callback")
        try:
            torch.Generator(device=self.stream.device).set_state(state["stream"].cpu())
        except (TypeError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid birth-death stream state") from error
