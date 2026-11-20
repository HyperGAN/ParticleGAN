"""Paper watch: TARP-CvM coverage loss (arXiv:2611.15829) on ParticleGAN.

Paper: "Training Calibrated Conditional Generative Models with a Sample-Only
Coverage Loss" (Remy, Rossellini, Chang, Willett, 2026). For a paired example
(theta_i, y_i) and a reference rule psi, the credibility value is the fraction
of model draws theta_hat ~ q(.|y_i) lying closer to psi(y_i) than the true
theta_i does. If q is the true posterior the credibilities are U(0,1); the
loss is the Cramer-von Mises discrepancy between their empirical CDF and the
uniform, computed SEPARATELY per rule and averaged over rules (never pooled).
Rules are built around a frozen posterior-mean regressor b(y) with
scale-matched displacements (switch / prior / posterior families, Eq. 8 and
Algorithm 1 of the paper). Total loss = base loss + lambda * mean-rule CvM.
The paper's SBI configuration is lambda=30, M=16 draws, R=48 rules
(16 per family), sigmoid temperature tau = 0.05 * median sample-to-reference
distance, and the plug-in CvM statistic of its Eq. 5. Theorem 1: the averaged
loss is a strict divergence -- zero iff q equals the true posterior -- and it
needs only samples, which is why it can attach to a GAN at all.

Why here: conditional averaging is the conditional generator's signature
failure -- mass parked at the conditional MEAN, between the modes. A
conditionally-averaging generator's credibility values are non-uniform, so
TARP-CvM punishes exactly that failure.

Harness note: the repo's GANTrainer is unconditional-only (it rejects any
recipe whose conditioning is not "scalar"), so this test uses the
caller-owned conditional host the trainer's own error message points to:
the condition is concatenated into the generator and critic inputs, and the
particle arms keep the repo's learned ParticlePrior table (sampled and
trained exactly as in the unconditional harness). The plain arm swaps the
table for the repo's frozen fresh-Gaussian code law, as in the pairing
gauntlet (PR #560). No formulation math changes.

Toy: y ~ U(-1, 1); theta | y is a two-component Gaussian mixture in R^2,

    c1(y) = (-1.1,  0.9 y),   c2(y) = (1.1, -0.9 y),   sigma = 0.18,
    w1(y) = 0.5 + 0.3 y       (so w1 ranges over [0.2, 0.8]),

with closed-form ground truth per condition. A generator that averages
parks at w1*c1 + w2*c2, on the bridge between the modes.

Arms (same budget, seeds 0/1/2):
  (a) particle  -- conditional ParticleGAN control,
  (b) plain     -- conditional plain GAN control (fresh-Gaussian codes),
  (c) tarp      -- (a) + TARP-CvM regularizer.

Metrics (cycle-3 rule: coverage / dispersion / mode-mass, not mode counts):
per-condition (5 y-bins) Naeem coverage, within-mode dispersion ratio
(mean pairwise distance inside a mode / sigma*sqrt(pi); 1.0 ideal), and
conditional mode-mass error (|generated mass within 3 sigma of each true
mode - true weight|). The held-out hard-indicator TARP-CvM statistic is
reported as a diagnostic only.

Gauntlet log (all arms x seeds 0/1/2, 1,800 steps, batch 128):

  Rounds 1-3 ran on a MISLABELLED toy and are superseded. The component
  pick `(rand < w).long()` casts True -> 1, so the sampler drew
  component 0 with probability 1 - w while the metrics compared
  against w: a weight-flipped world. (Coverage / dispersion / the TARP
  statistic are world-consistent, so those rounds' qualitative reading
  survived; the mode-mass margins did not.) The b(y) unit check caught
  it -- the frozen regressor had learned the exact negative of the
  analytic conditional mean. Both pick sites are fixed, and a sampler
  unit test now guards the flip. What the flipped rounds contributed:
  M=16 -> 32 was a wash and lambda=30 -> 60 helped there; both were
  re-measured below on the corrected toy.

  Corrected-toy controls (bit-identical at both configs):
    coverage      particle [0.151, 0.073, 0.308] mean 0.177
                  plain    [0.633, 0.165, 0.378] mean 0.392
    dispersion    particle [0.433, 0.470, 0.671] mean 0.525
                  plain    [0.952, 1.059, 0.913] mean 0.975
    mode-mass err particle [0.175, 0.499, 0.208] mean 0.294
                  plain    [0.025, 0.427, 0.498] mean 0.317
    tarp_stat     particle mean 0.055 | plain mean 0.059
  Both controls are erratic: the particle control concentrates within
  modes in every seed (dispersion <= 0.67) and its coverage collapses;
  the plain control is genuinely good in seed 0 and collapses on
  mode-mass in seeds 1/2 (0.43 / 0.50).

  Round 4 -- corrected toy, lambda=60, M=32 (the flipped rounds' pick):
    coverage 0.808 | dispersion 1.172 | mode-mass 0.087 | tarp_stat 0.003
  Round 5 -- corrected toy, the paper's config, lambda=30, M=16:
    coverage      tarp [0.764, 0.780, 0.825] mean 0.790
    dispersion    tarp [0.994, 1.065, 1.139] mean 1.066
    mode-mass err tarp [0.049, 0.077, 0.089] mean 0.072
    tarp_stat     tarp mean 0.003
  The paper's config is as good or better on every primary metric
  (dispersion 1.07 vs 1.17, mode-mass 0.072 vs 0.087), so it is the
  shipped configuration: the paper's SBI settings transfer as-is.

CUDA-cohort verdict (historical, diagnostic): at the paper's lambda=30,
M=16 TARP beat both
controls on mean coverage (+0.61 / +0.40), mean mode-mass error
(-0.22 / -0.25) and the held-out TARP statistic (20x lower), wins
coverage in every seed against both controls, and is consistent in a
way neither control is: coverage >= 0.76, mode-mass <= 0.09 and
dispersion within [0.99, 1.14] in EVERY seed. The single exception,
encoded below rather than hidden: plain seed 0 posts a lower
mode-mass error (0.025 vs 0.049) -- that seed's plain run is good,
and TARP's claim is the mean margin plus per-seed consistency.
Runtime: ~7 min for the nine trainings on one A6000 (the TARP arm
draws M samples per pair per step).

CPU-cohort verdict (the one CI checks): PARTIAL, and partly
hardware-dependent. Pinned to CPU with one thread, the run is still not
portable across CPU microarchitectures: CI runners split into two
reproducible states (see the claims block). In BOTH states TARP wins
coverage in every seed against both controls (>= 0.65 each seed, margin
>= 0.25 over particle), keeps mode-mass <= 0.12 and dispersion in
[0.90, 1.25] in every seed, and beats the particle control's held-out
TARP statistic by >= 4x. In both states it does NOT win mode-mass in
every seed against particle, and the particle control does not
concentrate in every seed (strict xfails). Everything measured against
the plain control (coverage margin, mode-mass, dispersion, held-out
statistic) flips with the runner, so those claims are not established on
CPU (non-strict xfails naming both outcomes). Runtime: ~2 min on CI.

Evidence scope (read before citing): DIAGNOSTIC / ASSISTED COHORT, not
qualification evidence. The training loop is caller-owned: hinge loss and
hand-built Adam on G, D and the particle table, with the TARP-CvM term
(and its frozen b(y) regressor) added by the host. The ``ka2`` recipe
contributes only ``make_prior``; its optimizer, critic penalty, noise and
prior policy are not executed, so no arm here is a recipe run. TARP is not
a declared recipe capability: the repo's GANTrainer is unconditional, and
a faithful port needs a recipe-driven paired conditional host (as the
DDGAN conditional-energy host does with ``make_optimizers``/``make_loss``/
``make_critic_penalty``) plus a declared recipe field and evidence
identity. That is not a modest change, so these results stay diagnostic.

Device cohorts: the gauntlet log above (rounds 1-5) was measured on CUDA
(``cuda`` if available). Those numbers are a separate, historical CUDA
cohort. This test now pins DEVICE to CPU with one intra-op thread (one
deterministic law per CPU microarchitecture); the CPU cohort's measured
numbers and claim status are recorded beside the assertions below. No
threshold was loosened: unsupported claims are xfails carrying the
measured numbers. The actual-training GIF
(target vs the three arms at y = -0.8 / 0 / +0.8, seed 0) is written by
``test_training_gif``; set PAPERWATCH_GIF_DIR to keep it.

Run: python -m pytest tests/test_paperwatch_tarp_cvm.py
Iteration knobs (env): TARP_STEPS, TARP_LAMBDA, TARP_M, TARP_TAU_MULT.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import get_recipe, init

DEVICE = torch.device("cpu")  # pinned: thresholds are CPU-calibrated

# --------------------------------------------------------------------------
# Toy: conditional two-component mixture with closed-form truth.
# --------------------------------------------------------------------------

SIGMA = 0.18
Y_DIM = 1


def mode_centers(y):
    """(B,1) -> (B,2,2): the two true component means at each y."""
    yv = y.reshape(-1)
    c1 = torch.stack([-1.1 * torch.ones_like(yv), 0.9 * yv], dim=1)
    c2 = torch.stack([1.1 * torch.ones_like(yv), -0.9 * yv], dim=1)
    return torch.stack([c1, c2], dim=1)


def mode_weights(y):
    """(B,1) -> (B,2): true component weights at each y."""
    w1 = 0.5 + 0.3 * y.reshape(-1)
    return torch.stack([w1, 1.0 - w1], dim=1)


def sample_pairs(n, gen, device=DEVICE):
    """Paired (theta, y) draws from the true joint."""
    y = torch.rand(n, Y_DIM, generator=gen, device=device) * 2 - 1
    w = mode_weights(y)
    # Component 0 with probability w[:, 0]: (rand < w) casts True -> 1,
    # which would pick component 1 -- the bool-to-index flip the b(y)
    # unit check caught in the first gauntlet pass.
    pick = (torch.rand(n, generator=gen, device=device) >= w[:, 0]).long()
    centers = mode_centers(y)
    theta = centers[torch.arange(n, device=device), pick] \
        + SIGMA * torch.randn(n, 2, generator=gen, device=device)
    return theta, y


def analytic_mean(y):
    c, w = mode_centers(y), mode_weights(y)
    return (c * w.unsqueeze(-1)).sum(dim=1)


# --------------------------------------------------------------------------
# Conditional networks: condition concatenated into G and D inputs.
# --------------------------------------------------------------------------

class CondGenerator(nn.Module):
    def __init__(self, z_dim=4, y_dim=Y_DIM, hidden_dim=128, n_hidden=3):
        super().__init__()
        layers, in_dim = [], z_dim + y_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z, y):
        return self.net(torch.cat([z, y], dim=-1))


class CondCritic(nn.Module):
    def __init__(self, y_dim=Y_DIM, hidden_dim=128, n_hidden=3, fourier=2):
        super().__init__()
        self.fourier = fourier
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
        in_dim = 2 + y_dim + (2 * fourier * 2 if fourier > 0 else 0)
        layers, dim = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        h = torch.cat([x, y], dim=-1)
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs
            h = torch.cat([h, torch.sin(xf).flatten(1),
                           torch.cos(xf).flatten(1)], dim=1)
        return self.net(h).squeeze(-1)


# --------------------------------------------------------------------------
# The paper's mechanism.
# --------------------------------------------------------------------------

def cvm_statistic(s):
    """Plug-in Cramer-von Mises statistic vs U(0,1), paper Eq. 5."""
    n = s.shape[0]
    s_sorted, _ = s.sort()
    k = torch.arange(1, n + 1, device=s.device, dtype=s.dtype)
    return (s.pow(2).mean() - ((2 * k - 1) * s_sorted).sum() / (n * n)
            + 1.0 / 3.0)


class PointEstimate(nn.Module):
    """b(y): posterior-mean regressor, trained first and frozen (Sec. 5.3)."""

    def __init__(self, y_dim=Y_DIM, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2))

    def forward(self, y):
        return self.net(y)


def train_point_estimate(seed, steps=1500, batch=256):
    torch.manual_seed(seed)
    b = PointEstimate().to(DEVICE)
    opt = torch.optim.Adam(b.parameters(), lr=1e-3)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 50_000)
    for _ in range(steps):
        theta, y = sample_pairs(batch, gen)
        opt.zero_grad()
        loss = nn.functional.mse_loss(b(y), theta)
        loss.backward()
        opt.step()
    # Reference scale varsigma: per-coordinate RMS residual (paper Eq. 9).
    with torch.no_grad():
        theta, y = sample_pairs(4096, torch.Generator(device=DEVICE)
                                .manual_seed(seed + 60_000))
        varsigma = (b(y) - theta).pow(2).mean(dim=0).sqrt()
    for p in b.parameters():
        p.requires_grad_(False)
    return b, varsigma


def tarp_cvm_loss(G, prior, theta, y, b, varsigma, *, draws_m, tau_mult,
                  n_switch=6, n_prior=3, n_post=3, code_gen):
    """Mean-rule TARP-CvM for one batch (paper Algorithm 1).

    Draws `draws_m` codes per pair from the prior (particle codes carry
    gradients to the table and the generator), builds switch / prior /
    posterior rules around the frozen b(y), computes soft credibilities
    with the paper's sigmoid relaxation, and averages the per-rule CvM
    statistics -- per rule first, never pooled.
    """
    n = theta.shape[0]
    codes, _ = prior.sample(n * draws_m, generator=code_gen)
    y_rep = y.repeat_interleave(draws_m, dim=0)
    fake = G(codes, y_rep).reshape(n, draws_m, 2)          # (N, M, 2)
    with torch.no_grad():
        b_y = b(y)                                          # (N, 2)
    losses = []

    def credibilities(refs, fake_sub):
        # refs: (N, 2); fake_sub: (N, M', 2) -> soft s_i in (0, 1)
        d_true = (theta - refs).norm(dim=1)                 # (N,)
        d_fake = (fake_sub - refs.unsqueeze(1)).norm(dim=2)  # (N, M')
        tau = tau_mult * d_fake.detach().median().clamp_min(1e-3)
        return torch.sigmoid((d_true.unsqueeze(1) - d_fake) / tau).mean(dim=1)

    # Switch rules (Eq. 8): one displacement per side of a y-hyperplane.
    for _ in range(n_switch):
        xi = torch.randn(2, device=DEVICE)
        xi2 = torch.randn(2, device=DEVICE)
        u = 1.0 if torch.rand((), device=DEVICE) < 0.5 else -1.0
        j = int(torch.randint(0, n, (1,), device=DEVICE))
        side = (y.reshape(-1) * u <= y.reshape(-1)[j] * u)
        disp = torch.where(side.unsqueeze(1), xi, xi2)
        refs = b_y + varsigma * disp
        losses.append(cvm_statistic(credibilities(refs, fake)))

    # Prior rules: one reference point from the marginal, shared by the batch.
    for _ in range(n_prior):
        c, _ = sample_pairs(1, code_gen)
        refs = c.expand(n, 2)
        losses.append(cvm_statistic(credibilities(refs, fake)))

    # Posterior rules: a detached own draw as reference, excluded from s_i.
    for _ in range(n_post):
        m_ref = int(torch.randint(0, draws_m, (1,), device=DEVICE))
        refs = fake[:, m_ref].detach()
        keep = [m for m in range(draws_m) if m != m_ref]
        losses.append(cvm_statistic(credibilities(refs, fake[:, keep])))

    return torch.stack(losses).mean()


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------

def naeem_coverage(real, fake, k=5):
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    radius = d_rr.topk(k, largest=False).values[:, -1]
    d_rf = torch.cdist(real, fake)
    return (d_rf.min(dim=1).values <= radius).float().mean().item()


def evaluate(G, prior, seed, per_bin=1500):
    """Per-condition metrics averaged over 5 y-bins, plus held-out TARP."""
    G.eval()
    covs, disps, mass_errs = [], [], []
    eval_gen = torch.Generator(device=DEVICE).manual_seed(seed + 777)
    for lo in (-1.0, -0.6, -0.2, 0.2, 0.6):
        hi = lo + 0.4
        y = lo + (hi - lo) * torch.rand(per_bin, Y_DIM, generator=eval_gen,
                                        device=DEVICE)
        w = mode_weights(y)
        pick = (torch.rand(per_bin, generator=eval_gen, device=DEVICE)
                >= w[:, 0]).long()
        centers = mode_centers(y)
        real = centers[torch.arange(per_bin, device=DEVICE), pick] \
            + SIGMA * torch.randn(per_bin, 2, generator=eval_gen, device=DEVICE)
        with torch.no_grad():
            codes, _ = prior.sample(per_bin, generator=eval_gen)
            fake = G(codes, y)
        covs.append(naeem_coverage(real, fake))
        # Mode mass: fraction of fakes within 3 sigma of each true mode,
        # from per-sample distances to its own two centers:
        dists = (fake.unsqueeze(1) - centers).norm(dim=2)   # (N, 2)
        gen_mass = (dists <= 3 * SIGMA).float().mean(dim=0)
        mass_errs.append((gen_mass - w.mean(dim=0)).abs().mean().item())
        # Dispersion: pairwise spread inside each assigned mode.
        assign = dists.argmin(dim=1)
        ratios = []
        for m in range(2):
            pts = fake[assign == m]
            if len(pts) >= 8:
                pts = pts[:256]
                dd = torch.cdist(pts, pts)
                iu = torch.triu_indices(len(pts), len(pts), offset=1)
                ratios.append(dd[iu[0], iu[1]].mean().item()
                              / (SIGMA * math.sqrt(math.pi)))
        disps.append(sum(ratios) / len(ratios) if ratios else 0.0)
    out = {"coverage": sum(covs) / len(covs),
           "dispersion": sum(disps) / len(disps),
           "mode_mass_err": sum(mass_errs) / len(mass_errs)}
    G.train()
    return out


def heldout_tarp(G, prior, b, varsigma, seed, n=512, draws_m=32, rules=24):
    """Hard-indicator TARP-CvM on fresh pairs/rules: a diagnostic only."""
    G.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 4242)
    theta, y = sample_pairs(n, gen)
    with torch.no_grad():
        codes, _ = prior.sample(n * draws_m, generator=gen)
        fake = G(codes, y.repeat_interleave(draws_m, dim=0)) \
            .reshape(n, draws_m, 2)
        b_y = b(y)
        stats = []
        for _ in range(rules):
            xi = torch.randn(2, device=DEVICE)
            xi2 = torch.randn(2, device=DEVICE)
            u = 1.0 if torch.rand((), device=DEVICE) < 0.5 else -1.0
            j = int(torch.randint(0, n, (1,), device=DEVICE))
            side = (y.reshape(-1) * u <= y.reshape(-1)[j] * u)
            refs = b_y + varsigma * torch.where(side.unsqueeze(1), xi, xi2)
            d_true = (theta - refs).norm(dim=1)
            d_fake = (fake - refs.unsqueeze(1)).norm(dim=2)
            s = (d_fake < d_true.unsqueeze(1)).float().mean(dim=1)
            stats.append(cvm_statistic(s).item())
    G.train()
    return sum(stats) / len(stats)


# --------------------------------------------------------------------------
# Arms. One protocol; only the prior law and the TARP term change.
# --------------------------------------------------------------------------

STEPS = int(os.environ.get("TARP_STEPS", "1800"))
BATCH = 128
NUM_PARTICLES = 256
Z_DIM = 4
LR = 2e-3
LAMBDA = float(os.environ.get("TARP_LAMBDA", "30"))
DRAWS_M = int(os.environ.get("TARP_M", "16"))
TAU_MULT = float(os.environ.get("TARP_TAU_MULT", "0.05"))
SEEDS = (0, 1, 2)


FRAME_EVERY = 150
FRAME_YS = (-0.8, 0.0, 0.8)
FRAME_N = 300


def _snapshot(G, prior, step):
    """Actual generator outputs at fixed conditions; own RNG stream, so
    recording never perturbs the training streams."""
    gen = torch.Generator(device=DEVICE).manual_seed(31_337)
    out = {"step": step}
    with torch.no_grad():
        for yv in FRAME_YS:
            codes, _ = prior.sample(FRAME_N, generator=gen)
            y = torch.full((FRAME_N, Y_DIM), yv, device=DEVICE)
            out[yv] = G(codes, y).cpu()
    return out


def run_arm(arm, seed, frames=None):
    torch.manual_seed(seed)
    overrides = {"z_dim": Z_DIM, "num_particles": NUM_PARTICLES,
                 "batch_size": BATCH, "total_steps": STEPS, "lr": LR}
    if arm == "plain":
        overrides.update(prior_update="frozen",
                         prior_code_law="fresh_gaussian")
    recipe = get_recipe(trainer="ka2", **overrides)
    prior = recipe.make_prior().to(DEVICE)
    G = CondGenerator(z_dim=Z_DIM).to(DEVICE)
    D = CondCritic().to(DEVICE)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    if arm != "plain":
        init.deterministic_orthogonal_(prior)
    b, varsigma = train_point_estimate(seed) if arm == "tarp" else (None, None)

    g_params = list(G.parameters())
    if arm != "plain":
        g_params += list(prior.parameters())
    opt_g = torch.optim.Adam(g_params, lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    data_gen = torch.Generator(device=DEVICE).manual_seed(seed + 10_000)
    code_gen = torch.Generator(device=DEVICE).manual_seed(seed + 20_000)

    for step in range(STEPS):
        if frames is not None and step % FRAME_EVERY == 0:
            frames.append(_snapshot(G, prior, step))
        theta, y = sample_pairs(BATCH, data_gen)
        # Critic step (hinge).
        codes_d, _ = prior.sample(BATCH, generator=code_gen)
        with torch.no_grad():
            fake_d = G(codes_d, y)
        opt_d.zero_grad()
        loss_d = (nn.functional.relu(1 - D(theta, y)).mean()
                  + nn.functional.relu(1 + D(fake_d, y)).mean())
        loss_d.backward()
        opt_d.step()
        # Generator (+ particle table) step.
        codes, _ = prior.sample(BATCH, generator=code_gen)
        fake = G(codes, y)
        opt_g.zero_grad()
        loss_g = -D(fake, y).mean()
        if arm == "tarp":
            loss_g = loss_g + LAMBDA * tarp_cvm_loss(
                G, prior, theta, y, b, varsigma, draws_m=DRAWS_M,
                tau_mult=TAU_MULT, code_gen=code_gen)
        loss_g.backward()
        opt_g.step()

    if frames is not None:
        frames.append(_snapshot(G, prior, STEPS))
    metrics = evaluate(G, prior, seed)
    bb, vv = (b, varsigma) if arm == "tarp" else _cached_estimate(seed)
    metrics["tarp_stat"] = heldout_tarp(G, prior, bb, vv, seed)
    return metrics


_ESTIMATE_CACHE = {}


def _cached_estimate(seed):
    if seed not in _ESTIMATE_CACHE:
        _ESTIMATE_CACHE[seed] = train_point_estimate(seed)
    return _ESTIMATE_CACHE[seed]


ARMS = ("particle", "plain", "tarp")


@contextlib.contextmanager
def single_thread():
    """Pin intra-op threads to 1: deterministic CPU reductions, and no
    oversubscription stalls on a loaded host. Restored afterwards."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_all():
    """All arms x seeds; seed-0 frames are recorded for the training GIF."""
    results, frames = {}, {}
    for arm in ARMS:
        frames[arm] = []
        results[arm] = [run_arm(arm, seed, frames=frames[arm] if seed == 0 else None)
                        for seed in SEEDS]
    return results, frames


@pytest.fixture(scope="module")
def gauntlet():
    with single_thread():
        out = run_all()
    # One tail-able line per run, outside pytest capture, so each CPU
    # state's numbers can be read from CI logs.
    summary = {arm: {key: [round(row[key], 4) for row in rows]
                     for key in rows[0]} for arm, rows in out[0].items()}
    sys.__stderr__.write("[tarp-cvm-cohort] " + json.dumps(summary) + "\n")
    return out


@pytest.fixture(scope="module")
def results(gauntlet):
    return gauntlet[0]


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


# --------------------------------------------------------------------------
# Mechanism unit checks (fast).
# --------------------------------------------------------------------------

def test_cvm_statistic_vanishes_for_uniform_grid():
    s = (torch.arange(2000, dtype=torch.float64) + 0.5) / 2000
    assert cvm_statistic(s).item() < 1e-3


def test_cvm_statistic_punishes_concentrated_credibilities():
    # A conditionally-averaging generator: credibilities pile at one value.
    s = torch.full((512,), 0.5, dtype=torch.float64)
    assert cvm_statistic(s).item() > 0.05


def test_analytic_mean_is_between_modes():
    y = torch.tensor([[0.0]])
    mean = analytic_mean(y)
    centers = mode_centers(y)
    d_mean = (mean - centers[0]).norm(dim=1).min().item()
    assert d_mean > 5 * SIGMA  # the bridge point is far from both modes


def test_sampler_matches_analytic_conditional_mean():
    # Guards the bool-to-index component flip: with (rand < w) cast to
    # long, component 0 is drawn with probability 1 - w instead of w,
    # and the sampler silently lives in a weight-flipped world while
    # the metrics compare against the unflipped weights.
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    theta, y = sample_pairs(100_000, gen)
    for lo, hi in ((-1.0, -0.8), (0.8, 1.0)):
        m = (y.reshape(-1) >= lo) & (y.reshape(-1) < hi)
        emp = theta[m].mean(dim=0)
        ana = analytic_mean(y[m]).mean(dim=0)
        assert (emp - ana).norm().item() < 0.05


def test_point_estimate_recovers_conditional_mean():
    b, varsigma = train_point_estimate(0)
    y = torch.tensor([[-0.8], [0.0], [0.8]], device=DEVICE)
    with torch.no_grad():
        err = (b(y) - analytic_mean(y).to(DEVICE)).norm(dim=1).max().item()
    assert err < 0.35
    # Residual scale: mode separation lives in x (varsigma ~ 1.05); the
    # y residual is the within/between spread of +-0.9y components (~0.49).
    assert varsigma[0] > 0.9 and varsigma[1] > 0.35


# --------------------------------------------------------------------------
# The gauntlet claims, encoded against the CPU cohort. Thresholds are the
# original CUDA-cohort claims, unchanged. The CPU numbers are NOT portable
# across CPU microarchitectures: with identical code, seeds, one thread and
# the same torch build, GitHub runners split into two reproducible states
# (the controls are chaotic, so a last-bit matmul difference changes which
# basin a run settles in). Both observed states are recorded:
#
#   state B (this workstation and some CI runners):
#     coverage      particle [0.598, 0.120, 0.246] mean 0.321
#                   plain    [0.603, 0.348, 0.387] mean 0.446
#                   tarp     [0.774, 0.852, 0.829] mean 0.818
#     dispersion    particle [0.830, 0.546, 0.582] mean 0.652
#                   plain    [0.946, 1.460, 1.440] mean 1.282
#                   tarp     [1.038, 1.116, 1.113] mean 1.089
#     mode-mass err particle [0.047, 0.500, 0.205] mean 0.250
#                   plain    [0.034, 0.232, 0.467] mean 0.245
#                   tarp     [0.073, 0.088, 0.054] mean 0.071
#     tarp_stat     particle 0.062 | plain 0.036 | tarp 0.0036
#   state A (other CI runners; partial, from failure output):
#     tarp-plain coverage margin 0.157; tarp mode-mass err beats plain in
#     0 of 3 seeds; tarp_stat tarp 0.0024 vs plain 0.0068; particle
#     dispersion seed 0 1.166; seed 2 mode-mass tarp 0.100 vs particle
#     0.034. Each run prints a one-line `[tarp-cvm-cohort]` JSON summary to
#     stderr so both states can be read from CI logs.
#
# Claim status:
#   * holds in both states -> plain assertion;
#   * fails in both states -> xfail(strict=True) with the measured numbers;
#   * hardware-dependent   -> xfail(strict=False) naming both outcomes. That
#     claim is NOT established on CPU; it neither passes nor fails CI.
# --------------------------------------------------------------------------

HW_DEPENDENT = "hardware-dependent on CPU (not established): "


def test_tarp_beats_both_controls_on_coverage_every_seed(results):
    for seed_idx in range(len(SEEDS)):
        assert results["tarp"][seed_idx]["coverage"] \
            > results["particle"][seed_idx]["coverage"]
        assert results["tarp"][seed_idx]["coverage"] \
            > results["plain"][seed_idx]["coverage"]
        # Consistency neither control has: no collapsed seed.
        assert results["tarp"][seed_idx]["coverage"] >= 0.65
    margin = _mean(results["tarp"], "coverage") \
        - _mean(results["particle"], "coverage")
    assert margin >= 0.25, f"tarp-particle coverage margin {margin:.3f}"


@pytest.mark.xfail(strict=False, reason=HW_DEPENDENT + (
    "tarp-plain mean coverage margin 0.373 (state B) vs 0.157 (state A); "
    "claim >= 0.25"))
def test_tarp_coverage_margin_over_plain(results):
    margin = _mean(results["tarp"], "coverage") \
        - _mean(results["plain"], "coverage")
    assert margin >= 0.25, f"tarp-plain coverage margin {margin:.3f}"


def test_tarp_mode_mass_consistency(results):
    # Per-seed consistency: never a collapsed seed.
    for seed_idx in range(len(SEEDS)):
        assert results["tarp"][seed_idx]["mode_mass_err"] <= 0.12


@pytest.mark.xfail(strict=False, reason=HW_DEPENDENT + (
    "tarp beats plain mode-mass in 2/3 seeds with mean margins -0.179 vs "
    "particle / -0.173 vs plain (state B); 0/3 seeds vs plain (state A, "
    "plain margin therefore negative; particle margin unmeasured)"))
def test_tarp_mode_mass_margins(results):
    wins = sum(results["tarp"][i]["mode_mass_err"]
               < results["plain"][i]["mode_mass_err"]
               for i in range(len(SEEDS)))
    assert wins >= 2, f"tarp beats plain mode-mass in {wins}/3 seeds"
    margin_p = _mean(results["particle"], "mode_mass_err") \
        - _mean(results["tarp"], "mode_mass_err")
    margin_g = _mean(results["plain"], "mode_mass_err") \
        - _mean(results["tarp"], "mode_mass_err")
    assert margin_p >= 0.15, f"particle-tarp mode-mass margin {margin_p:.3f}"
    assert margin_g >= 0.15, f"plain-tarp mode-mass margin {margin_g:.3f}"


@pytest.mark.xfail(strict=True, reason=(
    "CPU FAIL in both states: tarp mode-mass err loses to particle in one "
    "seed (state B seed 0: 0.073 vs 0.047; state A seed 2: 0.100 vs 0.034); "
    "CUDA-cohort claim: tarp wins every seed"))
def test_tarp_beats_particle_on_mode_mass_every_seed(results):
    for seed_idx in range(len(SEEDS)):
        assert results["tarp"][seed_idx]["mode_mass_err"] \
            < results["particle"][seed_idx]["mode_mass_err"]


def test_tarp_dispersion_at_true_spread(results):
    # TARP sits at the true spread in every seed (bounded both sides).
    for seed_idx in range(len(SEEDS)):
        assert 0.90 <= results["tarp"][seed_idx]["dispersion"] <= 1.25
    assert 1.00 <= _mean(results["tarp"], "dispersion") <= 1.15


@pytest.mark.xfail(strict=True, reason=(
    "CPU FAIL in both states: particle seed 0 dispersion 0.830 (state B) / "
    "1.166 (state A) > 0.75; the control does not concentrate in every seed"))
def test_particle_control_concentrates_every_seed(results):
    for seed_idx in range(len(SEEDS)):
        assert results["particle"][seed_idx]["dispersion"] <= 0.75


@pytest.mark.xfail(strict=False, reason=HW_DEPENDENT + (
    "tarp mean dispersion 1.089 < plain 1.282 in state B (plain "
    "over-disperses); holds in state A; CUDA-cohort claim: tarp > plain"))
def test_tarp_dispersion_exceeds_plain(results):
    assert _mean(results["tarp"], "dispersion") \
        > _mean(results["plain"], "dispersion")


def test_heldout_tarp_statistic_beats_particle(results):
    # Diagnostic only in the verdict, but it must point the same way.
    assert _mean(results["tarp"], "tarp_stat") <= 0.006
    assert _mean(results["tarp"], "tarp_stat") \
        <= 0.25 * _mean(results["particle"], "tarp_stat")


@pytest.mark.xfail(strict=False, reason=HW_DEPENDENT + (
    "held-out tarp_stat 0.0036 vs plain 0.036 (state B) but 0.0024 vs "
    "0.0068 (state A); claim <= 0.25x plain"))
def test_heldout_tarp_statistic_beats_plain(results):
    assert _mean(results["tarp"], "tarp_stat") \
        <= 0.25 * _mean(results["plain"], "tarp_stat")


# --------------------------------------------------------------------------
# Actual-training GIF (seed 0): true conditional law vs each arm's samples
# at fixed conditions y in FRAME_YS, on fixed axes, over training.
# --------------------------------------------------------------------------

def render_training_gif(frames, results, path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    gen = torch.Generator(device=DEVICE).manual_seed(5)
    truth = {}
    for yv in FRAME_YS:
        y = torch.full((FRAME_N, Y_DIM), yv, device=DEVICE)
        w = mode_weights(y)
        pick = (torch.rand(FRAME_N, generator=gen, device=DEVICE) >= w[:, 0]).long()
        truth[yv] = (mode_centers(y)[torch.arange(FRAME_N), pick]
                     + SIGMA * torch.randn(FRAME_N, 2, generator=gen, device=DEVICE))
    rows = ("target",) + ARMS
    images = []
    n_frames = len(frames[ARMS[0]])
    for i in range(n_frames):
        fig, axes = plt.subplots(len(FRAME_YS), len(rows), figsize=(8, 6), dpi=64)
        for r, yv in enumerate(FRAME_YS):
            for c, name in enumerate(rows):
                ax = axes[r, c]
                pts = truth[yv] if name == "target" else frames[name][i][yv]
                ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1.6, 1.6)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    title = name
                    if name != "target" and i == n_frames - 1:
                        title += f"\ncov {results[name][0]['coverage']:.2f}"
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"y={yv:+.1f}", fontsize=8)
        fig.suptitle(f"TARP-CvM diagnostic cohort, CPU, seed 0, update "
                     f"{frames[ARMS[0]][i]['step']}/{STEPS}", fontsize=9)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P"))
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=[250] * (n_frames - 1) + [1500], loop=0)
    return n_frames


def test_training_gif(gauntlet, tmp_path):
    results, frames = gauntlet
    out = os.environ.get("PAPERWATCH_GIF_DIR")
    path = (Path(out) if out else tmp_path) / "tarp-cvm-cpu.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert render_training_gif(frames, results, path) == STEPS // FRAME_EVERY + 1
    assert path.stat().st_size > 0
