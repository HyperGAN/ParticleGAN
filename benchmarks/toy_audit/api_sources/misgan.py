# Pure sampler/oracle excerpt from 5efc80a3dfb2baffc83f54c61cab07ccfc9826f5:lib/misgan.py
# Original Git blob b92f379bac7e0f21016242414c6ab300e4b3c277. Training builders intentionally excluded.
"""MisGAN toy: the 100-Gaussian grid lifted to 8D, missingness mechanisms,
closed-form imputation baselines (mean, kNN, Bayes-optimal) and metrics.

Data: x2 ~ 100-Gaussian grid (``sample_100gaussians``, sigma 0.03), lifted by a
fixed random orthonormal A (8x2): x_raw = A x2 + eta * e, eta = 0.01 * sigma.
Training and evaluation work in standardized coordinates (per-coordinate mean
and std of the *observed* training entries), where the fill value tau = 0 is the
observed mean. ``to2d`` maps back with A^T for the grid metrics.

Masks use 1 = observed. Every mechanism is independent of x (MCAR), as MisGAN
assumes. Nothing here consumes the global RNG.
"""
import math

import torch

from lib.toy_models import sample_100gaussians

DIM = 8
MODE_STD = 0.03
LIFT_NOISE = 0.01 * MODE_STD
BITS = 2 ** torch.arange(DIM)
# Four sensor-group dropout patterns (observed coordinates) with unequal
# probabilities. P0-P2 jointly observe every coordinate pair; P3 keeps one
# sensor, so 10% of block rows are ambiguous (a single linear view of x2).
BLOCK_PATTERNS = (((0, 1, 2, 3, 4), 0.4), ((3, 4, 5, 6, 7), 0.3),
                  ((0, 1, 2, 5, 6, 7), 0.2), ((7,), 0.1))
MECHANISMS = ("mcar_p20", "mcar_p50", "mcar_p80", "block")


def grid_centers(device=None, dtype=torch.float32):
    coords = torch.arange(10, device=device, dtype=dtype) - 4.5
    return torch.cartesian_prod(coords, coords)  # same order as lib.toy_metrics


def _block_masks():
    masks = torch.zeros(len(BLOCK_PATTERNS), DIM)
    for row, (observed, _) in enumerate(BLOCK_PATTERNS):
        masks[row, list(observed)] = 1
    return masks, torch.tensor([p for _, p in BLOCK_PATTERNS])


def pattern_probs(mechanism):
    """Exact probability of each of the 2^8 masks (index = sum m_c 2^c)."""
    all_masks = ((torch.arange(2 ** DIM)[:, None] // BITS) % 2).double()
    if mechanism.startswith("mcar_p"):
        p = int(mechanism[6:]) / 100
        return (all_masks * (1 - p) + (1 - all_masks) * p).prod(1)
    if mechanism == "block":
        masks, probs = _block_masks()
        out = torch.zeros(2 ** DIM, dtype=torch.float64)
        out[(masks.long() * BITS).sum(1)] = probs.double()
        return out
    raise ValueError(f"unknown mechanism {mechanism!r}; choose {MECHANISMS}")


def sample_masks(mechanism, n, generator, device="cpu"):
    if mechanism.startswith("mcar_p"):
        p = int(mechanism[6:]) / 100
        return (torch.rand(n, DIM, generator=generator) >= p).float().to(device)
    if mechanism == "block":
        masks, probs = _block_masks()
        return masks[torch.multinomial(probs, n, replacement=True, generator=generator)].to(device)
    raise ValueError(f"unknown mechanism {mechanism!r}; choose {MECHANISMS}")


def pattern_index(masks):
    return ((masks > 0.5).long() * BITS.to(masks.device)).sum(-1)


class Problem:
    """Fixed incomplete training pool plus a clean, masked test set (CPU generation)."""

    def __init__(self, mechanism, n_train=20_000, n_test=10_000, seed=0, device="cpu"):
        self.mechanism, self.device = mechanism, torch.device(device)
        rng = torch.Generator().manual_seed(seed)
        self.A = torch.linalg.qr(torch.randn(DIM, 2, generator=rng, dtype=torch.float64))[0]

        def draw(n):
            x2 = sample_100gaussians(n, "cpu", generator=rng, std=MODE_STD).double()
            raw = x2 @ self.A.T + LIFT_NOISE * torch.randn(n, DIM, generator=rng, dtype=torch.float64)
            return raw, torch.cdist(x2, grid_centers(dtype=torch.float64)).argmin(1)

        raw_train, _ = draw(n_train)
        m_train = sample_masks(mechanism, n_train, rng).double()
        raw_test, self.y_test = draw(n_test)
        m_test = sample_masks(mechanism, n_test, rng).double()
        raw_ref, _ = draw(n_test)  # an independent clean sample: the sliced-W1 floor
        count = m_train.sum(0).clamp_min(1)
        self.mean = (raw_train * m_train).sum(0) / count
        self.std = (((raw_train - self.mean) ** 2 * m_train).sum(0) / count).sqrt()
        f32 = dict(device=self.device, dtype=torch.float32)
        self.x_train_full = self.standardize(raw_train).to(**f32)  # oracle only
        self.m_train = m_train.to(**f32)
        self.x_train = self.x_train_full * self.m_train  # f_0: tau = 0
        self.x_test = self.standardize(raw_test).to(**f32)
        self.m_test = m_test.to(**f32)
        self.x_ref = self.standardize(raw_ref).to(**f32)
        self.y_test = self.y_test.to(self.device)
        self.probs = pattern_probs(mechanism)
        self.missing_rate = (self.probs[:, None] * (1 - (torch.arange(2 ** DIM)[:, None] // BITS) % 2)).sum(0)

    def standardize(self, raw):
        return (raw - self.mean.to(raw)) / self.std.to(raw)

    def raw(self, x):
        return x.double() * self.std.to(x.device) + self.mean.to(x.device)

    def to2d(self, x):
        return (self.raw(x) @ self.A.to(x.device)).float()

    def off_plane(self, x):
        raw, A = self.raw(x), self.A.to(x.device)  # the data plane passes through 0
        return float((raw - raw @ A @ A.T).norm(dim=1).mean())

    def train_batch(self, n, generator):
        idx = torch.randint(len(self.m_train), (n,), device=self.device, generator=generator)
        return self.x_train[idx], self.m_train[idx], self.x_train_full[idx]


# ---------------------------------------------------------------- baselines

@torch.no_grad()
def bayes_posterior(problem, x, m, draws=0, generator=None):
    """Exact posterior over the 100 modes given the observed coordinates.

    Returns ``(probs (n, 100), samples (draws, n, 8) or None)``; samples are
    exact draws of the missing coordinates from p(x_M | x_O), in standardized
    units, with observed coordinates kept.
    """
    dev = x.device
    A = problem.A.to(dev)
    mu = grid_centers(dev, torch.float64)
    s2, e2 = MODE_STD ** 2, LIFT_NOISE ** 2
    use_test_cache = (
        x is getattr(problem, "x_test", None)
        and m is getattr(problem, "m_test", None)
    )
    cached = getattr(problem, "_bayes_test_cache", None) if use_test_cache else None
    if cached is not None and cached["dev"] == dev and cached["x_ver"] == x._version and cached["m_ver"] == m._version:
        raw = cached["raw"]
        probs = cached["probs"]
        patterns = cached["patterns"]
    else:
        raw = problem.raw(x)
        probs = torch.empty(len(x), 100, dtype=torch.float64, device=dev)
        keys = pattern_index(m)
        patterns = []
        for key in keys.unique().tolist():
            rows = (keys == key).nonzero().squeeze(1)
            obs = [c for c in range(DIM) if key >> c & 1]
            miss = [c for c in range(DIM) if not key >> c & 1]
            if obs:
                Ao, y = A[obs], raw[rows][:, obs]
                S = s2 * Ao @ Ao.T + e2 * torch.eye(len(obs), dtype=torch.float64, device=dev)
                L = torch.linalg.cholesky(S)
                y_t = torch.linalg.solve_triangular(L, y.T, upper=False).T
                mu_t = torch.linalg.solve_triangular(L, Ao @ mu.T, upper=False).T
                logw = -0.5 * torch.cdist(y_t, mu_t, compute_mode="donot_use_mm_for_euclid_dist").square()
                cov = torch.linalg.inv(torch.eye(2, dtype=torch.float64, device=dev) / s2 + Ao.T @ Ao / e2)
                obs_term = (y @ Ao / e2)[None]
            else:
                logw = torch.zeros(len(rows), 100, dtype=torch.float64, device=dev)
                cov = s2 * torch.eye(2, dtype=torch.float64, device=dev)
                obs_term = None
            p = logw.softmax(1)
            probs[rows] = p
            chol = torch.linalg.cholesky(cov) if miss else None
            patterns.append((rows, miss, p.float(), cov, chol, obs_term))
        if use_test_cache:
            problem._bayes_test_cache = {
                "dev": dev,
                "x_ver": x._version,
                "m_ver": m._version,
                "raw": raw,
                "probs": probs,
                "patterns": patterns,
            }
    if not draws:
        return probs.float(), None
    out = raw.expand(draws, -1, -1).clone()
    for rows, miss, p_f32, cov, chol, obs_term in patterns:
        if miss:
            k = torch.multinomial(p_f32, draws, replacement=True, generator=generator).T  # (draws, n)
            mean = mu[k] / s2  # (draws, n, 2)
            if obs_term is not None:
                mean = mean + obs_term
            mean = mean @ cov.T
            eps = torch.randn(draws, len(rows), 2, generator=generator, device=dev, dtype=torch.float64)
            x2 = mean + eps @ chol.T
            noise = torch.randn(draws, len(rows), len(miss), generator=generator, device=dev, dtype=torch.float64)
            vals = x2 @ A[miss].T + LIFT_NOISE * noise
            block = out[:, rows]
            block[:, :, miss] = vals
            out[:, rows] = block
    out = problem.standardize(out).float()
    out = m * x + (1 - m) * out
    return probs.float(), out
