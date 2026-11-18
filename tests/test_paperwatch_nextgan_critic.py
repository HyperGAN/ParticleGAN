"""Paper watch: NextGAN field-regression critic (arXiv:2611.06270) on ParticleGAN.

Paper: "NextGAN: Towards Simple and Scalable GAN Training" (2026). The critic
is a scalar potential u(x). Instead of supervising critic VALUES (real vs fake
scores), NextGAN trains it by FIELD REGRESSION: sample a real x_r and an
independently generated x_p (stop-grad), draw alpha ~ U(0, 1), form
x_alpha = alpha * x_r + (1 - alpha) * sg[x_p], and regress the critic's input
gradient onto the displacement:

    L_critic = 0.5 * || grad_x u(x_alpha) - sg[x_r - x_p] ||^2

The generator keeps the standard potential update: descend -u(x_p). In the
paper's controlled 2D study (Appendix E: 8 Gaussians on a circle, radius 2,
sigma 0.16; potential and generator 3-hidden-layer SiLU MLPs of width 128;
batch 512; Adam lr 2e-4 for both; 2 critic updates per generator update; 30K
generator updates), NextGAN reaches effective mode count N_eff = 7.995 at
captured fraction 0.928 (seed 0), while a matched non-saturating GAN baseline
collapses to N_eff = 2.770 at captured 0.933: the claim is allocation BALANCE
at equal captured fraction, not coverage.

This test asks the transfer question: does the field-regression critic help a
system that already has a learned particle prior? Three arms, identical
budget (STEPS generator updates, batch 512, seeds 0/1/2), one 2D target (8
Gaussians on a ring, radius 2, sigma 0.16, the paper's App E target):

  (a) ParticleGAN control  -- ka2 recipe, learned 256-particle prior, the
                              repo's standard (relativistic + KA2) critic,
                              driven by GANTrainer unmodified.
  (b) Plain GAN control    -- same recipe and networks, frozen fresh-Gaussian
                              prior (the repo's matched non-particle control)
                              and the non-saturating loss: the paper's
                              baseline, inside the repo's harness.
  (c) ParticleGAN + NextGAN -- (a)'s prior, generator, optimizers and prior
                              regularizer, but the critic phase is replaced by
                              field regression (2 updates per generator
                              update, plain Adam lr 2e-4 on the potential,
                              per the paper) and the generator descends
                              -u(x_p); the particle/prior update is the
                              repo's own generator optimizer step, unchanged.

Metrics (8192 eval samples per run):
  * captured fraction -- fraction of generated samples within CAPTURE_RADIUS
    (= 3 * sigma = 0.48, strictly inside the Voronoi boundary at ~0.765) of
    a mode center. Used consistently for every arm.
  * N_eff -- exp(entropy of mode shares among captured samples), the paper's
    effective mode count; 8.0 is perfect balance, 1.0 is total collapse.
  * Naeem coverage -- fraction of real samples whose 5-NN ball within the
    real set contains a generated sample (as in the cycle-1 pairing test).

Budget note: the committed budget is STEPS = 15,000 generator updates at
batch 512 for every arm (half the paper's 30K; a confirmatory gauntlet round
at the full 30K is in the PR log and shows the same pattern). Every arm sees
the same number of generator updates, the same batch size and the same
evaluation. The nine trainings take ~25 minutes on one A6000.

Verdict of the cycle-2 gauntlet (3 rounds): PARTIAL. At 3K updates nothing
separates (all arms N_eff ~ 7.9). At 15K the ParticleGAN control develops
real mode imbalance (N_eff 6.77 +- 0.27) and the field-regression critic
removes it (N_eff 7.91 +- 0.10, ahead in every seed, +1.14 mean) at a
comparable captured fraction (0.56 vs 0.65). But the plain control never
imbalances in this harness (N_eff 7.96 +- 0.02): it trains inside the same
KA2-stabilized trainer, so the paper's collapsing-baseline phenomenon does
not reproduce here and NextGAN has nothing to beat on that side (margin
-0.05, a tie). Separately, both particle-prior arms sit far below the plain
arm on Naeem coverage (~0.04-0.07 vs 0.53) at every budget -- a property of
the learned prior's diffuse within-mode spread, present with and without
the field critic, not an effect of it. The assertions below encode exactly
those measured margins, including the negative halves.

Run: python -m pytest tests/test_paperwatch_nextgan_critic.py
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init


# --------------------------------------------------------------------------
# Networks: the paper's App E architecture (3 hidden layers, width 128, SiLU)
# for both the generator and the potential/critic, in every arm.
# --------------------------------------------------------------------------

class SiLUMLP(nn.Module):
    """3-hidden-layer SiLU MLP, width 128 (arXiv:2611.06270, Appendix E)."""

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_hidden=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(dim, hidden_dim), nn.SiLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


# --------------------------------------------------------------------------
# The paper's mechanism: field-regression critic loss.
# --------------------------------------------------------------------------

def field_regression_loss(potential, real, fake_detached):
    """0.5 * || grad_x u(x_alpha) - sg[x_r - x_p] ||^2, paper eq. (Sec. 3).

    ``fake_detached`` must already be stop-grad: the field target and the
    interpolation both treat the generated sample as a constant. alpha is
    drawn per-sample from U(0, 1). The gradient is taken with
    ``create_graph=True`` so the loss can backpropagate into the potential's
    parameters (it is a second-order objective in u).
    """
    alpha = torch.rand(real.shape[0], 1, device=real.device, dtype=real.dtype)
    x_alpha = alpha * real + (1.0 - alpha) * fake_detached
    x_alpha.requires_grad_(True)
    u = potential(x_alpha)
    grad = torch.autograd.grad(u.sum(), x_alpha, create_graph=True)[0]
    target = real - fake_detached
    return 0.5 * (grad - target).pow(2).sum(dim=1).mean()


# --------------------------------------------------------------------------
# Target and metrics (paper App E target; definitions in the module docstring).
# --------------------------------------------------------------------------

N_MODES = 8
RADIUS = 2.0
SIGMA = 0.16
CAPTURE_RADIUS = 3.0 * SIGMA


def _ring_means(device="cpu"):
    angles = torch.linspace(0, 2 * math.pi, N_MODES + 1)[:-1]
    return torch.stack([angles.cos(), angles.sin()], dim=1) * RADIUS


MEANS = _ring_means()


def sample_mix(n, gen, device="cpu"):
    means = MEANS.to(device)
    idx = torch.randint(0, N_MODES, (n,), generator=gen, device=device)
    noise = torch.randn(n, 2, generator=gen, device=device)
    return means[idx] + SIGMA * noise


def mode_balance_metrics(fake):
    """Captured fraction and N_eff for generated samples ``fake`` (N, 2)."""
    means = MEANS.to(fake.device)
    dists = torch.cdist(fake, means)
    mind, nearest = dists.min(dim=1)
    captured_mask = mind <= CAPTURE_RADIUS
    captured = float(captured_mask.float().mean())
    n_captured = int(captured_mask.sum())
    if n_captured == 0:
        return {"captured": captured, "n_eff": 0.0}
    counts = torch.bincount(nearest[captured_mask], minlength=N_MODES)
    shares = counts.to(torch.float64) / counts.sum()
    shares = shares[shares > 0]
    entropy = float(-(shares * shares.log()).sum())
    return {"captured": captured, "n_eff": math.exp(entropy)}


def naeem_coverage(real, fake, k=5):
    """Fraction of real points whose kNN ball (within real) contains a fake."""
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    radius = d_rr.topk(k, largest=False).values[:, -1]
    d_rf = torch.cdist(real, fake)
    return (d_rf.min(dim=1).values <= radius).float().mean().item()


# --------------------------------------------------------------------------
# Arms. One protocol for every arm; only the prior / loss / critic phase
# changes. STEPS is the uniform budget (generator updates).
# --------------------------------------------------------------------------

STEPS = 15000
BATCH = 512
NUM_PARTICLES = 256
Z_DIM = 2
LR = 2e-4
CRITIC_STEPS = 2  # field-regression updates per generator update (paper)
SEEDS = (0, 1, 2)
N_EVAL = 8192


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_trainer(arm, seed, steps, device):
    overrides = dict(z_dim=Z_DIM, num_particles=NUM_PARTICLES,
                     batch_size=BATCH, total_steps=steps, lr=LR)
    if arm == "plain":
        overrides.update(prior_update="frozen",
                         prior_code_law="fresh_gaussian",
                         loss="non_saturating")
    recipe = get_recipe(trainer="ka2", **overrides)
    prior = recipe.make_prior()
    G = SiLUMLP(Z_DIM, 2)
    D = SiLUMLP(2, 1)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    if arm != "plain":
        init.deterministic_orthogonal_(prior)
    G, D, prior = G.to(device), D.to(device), prior.to(device)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed)
    return trainer


def _evaluate(trainer, seed, device):
    eval_gen = torch.Generator(device=device).manual_seed(seed + 999)
    with torch.no_grad():
        latent, _rows = trainer.prior.sample(
            N_EVAL, generator=eval_gen)
        fake = trainer.G(latent)
        if getattr(trainer.prior, "postprocesses_generated", False):
            fake = trainer.prior.postprocess_generated(fake)
    fake = fake.detach().cpu()
    real_eval = sample_mix(N_EVAL, torch.Generator().manual_seed(777))
    metrics = mode_balance_metrics(fake)
    metrics["coverage"] = naeem_coverage(real_eval, fake)
    return metrics


def _run_trainer_arm(arm, seed, steps, device):
    """Arms (a) and (b): unmodified GANTrainer updates."""
    trainer = _build_trainer(arm, seed, steps, device)
    data_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    for _ in range(steps):
        trainer.step(sample_mix(BATCH, data_gen, device))
    return _evaluate(trainer, seed, device)


def _run_nextgan_arm(seed, steps, device):
    """Arm (c): ParticleGAN prior/generator machinery, NextGAN critic.

    The trainer is constructed exactly as in arm (a) so the prior, the
    generator optimizer (including its latent damping) and the prior
    regularizer are the repo's own; only the update loop differs: the
    critic phase is field regression (plain Adam, paper lr) and the
    generator phase descends -u(x_p) through that same optimizer.
    """
    trainer = _build_trainer("particle", seed, steps, device)
    recipe, G, D, prior = trainer.recipe, trainer.G, trainer.D, trainer.prior
    opt_field = torch.optim.Adam(D.parameters(), lr=LR)
    data_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    latent_gen = torch.Generator(device=device).manual_seed(seed + 20_000)

    def draw_fake(n, grad):
        latent, indices = prior.sample(n, generator=latent_gen)
        fake = G(latent)
        if trainer._postprocesses:
            fake = trainer._postprocess(prior, fake)
        return fake, latent, indices

    for _ in range(steps):
        # Critic phase: CRITIC_STEPS field-regression updates.
        D.train()
        for _ in range(CRITIC_STEPS):
            real = sample_mix(BATCH, data_gen, device)
            with torch.no_grad():
                fake_detached, _lat, _idx = draw_fake(BATCH, grad=False)
            loss_c = field_regression_loss(D, real, fake_detached)
            opt_field.zero_grad()
            loss_c.backward()
            opt_field.step()
        # Generator / particle phase: descend -u(x_p), repo optimizer.
        D.eval()
        flags = [p.requires_grad for p in D.parameters()]
        try:
            D.requires_grad_(False)
            G.train()
            fake, latent, indices = draw_fake(BATCH, grad=True)
            loss_g = -D(fake).mean()
            if prior.z.requires_grad and (
                    recipe.prior_regularizer == "vicreg" or recipe.prior_reg > 0):
                raw = prior.z if recipe.num_particles <= 1024 \
                    else prior.z[torch.unique(indices)]
                loss_g = loss_g + recipe.prior_reg * trainer.prior_regularizer(raw)
            if prior.z.requires_grad and recipe.prior_l2 > 0:
                loss_g = loss_g + recipe.prior_l2 * prior.z.square().mean()
            trainer.opt_g.zero_grad()
            loss_g.backward()
            trainer.opt_g.step()
        finally:
            for parameter, flag in zip(D.parameters(), flags):
                parameter.requires_grad_(flag)
    return _evaluate(trainer, seed, device)


def run_arm(arm, seed, steps=STEPS):
    device = _device()
    torch.manual_seed(seed)
    if arm == "nextgan":
        return _run_nextgan_arm(seed, steps, device)
    return _run_trainer_arm(arm, seed, steps, device)


# --------------------------------------------------------------------------
# Mechanism unit checks (fast): the field loss is the paper's loss.
# --------------------------------------------------------------------------

def test_field_loss_matches_hand_computed_linear_potential():
    # For a linear potential u(x) = w . x, grad_x u = w everywhere, so the
    # loss is exactly 0.5 * || w - (x_r - x_p) ||^2 averaged over the batch,
    # independent of alpha.
    torch.manual_seed(0)
    potential = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        potential.weight.copy_(torch.tensor([[0.3, -0.7]]))
    real = torch.randn(64, 2)
    fake = torch.randn(64, 2)
    loss = field_regression_loss(potential, real, fake)
    w = potential.weight.detach().squeeze(0)
    expected = 0.5 * (w - (real - fake)).pow(2).sum(dim=1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_field_loss_detaches_generated_samples():
    torch.manual_seed(0)
    potential = SiLUMLP(2, 1, hidden_dim=32)
    real = torch.randn(32, 2)
    fake = torch.randn(32, 2, requires_grad=True)
    loss = field_regression_loss(potential, real, fake.detach())
    loss.backward()
    # No gradient path to the generated samples (stop-grad in the paper)...
    assert fake.grad is None
    # ...but the potential's parameters receive second-order gradients.
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in potential.parameters())


def test_field_loss_decreases_as_field_is_learned():
    # With independently paired reals/fakes most of the target's variance is
    # irreducible (the learnable part is E[x_r - x_p | x_alpha]); on this
    # fixed pair cloud a working second-order loss still fits that part and
    # drops ~15-20%, while a broken one (no create_graph, detached field)
    # cannot move at all.
    torch.manual_seed(0)
    potential = SiLUMLP(2, 1, hidden_dim=64)
    opt = torch.optim.Adam(potential.parameters(), lr=5e-3)
    gen = torch.Generator().manual_seed(3)
    real = sample_mix(256, gen)
    fake = torch.randn(256, 2)  # fixed generated cloud
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = field_regression_loss(potential, real, fake)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.88 * first


def test_mode_balance_metrics_on_known_clouds():
    gen = torch.Generator().manual_seed(0)
    balanced = sample_mix(8192, gen)
    m = mode_balance_metrics(balanced)
    assert m["captured"] > 0.9
    assert m["n_eff"] > 7.5  # true samples are near-perfectly balanced
    collapsed = MEANS[0] + SIGMA * torch.randn(8192, 2, generator=gen)
    m = mode_balance_metrics(collapsed)
    assert m["captured"] > 0.9
    assert m["n_eff"] < 1.2  # one mode only


# --------------------------------------------------------------------------
# The gauntlet result, encoded. (Assertions finalized from the measured
# margins of the final round; see the PR description for the full log.)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    return {arm: [run_arm(arm, seed) for seed in SEEDS]
            for arm in ("particle", "plain", "nextgan")}


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def test_nextgan_beats_particle_control_on_n_eff(results):
    # The positive half of the verdict: the field critic removes the
    # imbalance the ParticleGAN control develops at this budget, in every
    # seed (measured margins +1.16 / +1.38 / +0.88, mean +1.14).
    for seed_idx in range(len(SEEDS)):
        assert results["nextgan"][seed_idx]["n_eff"] \
            > results["particle"][seed_idx]["n_eff"]
    margin = _mean(results["nextgan"], "n_eff") \
        - _mean(results["particle"], "n_eff")
    assert margin >= 0.8, f"nextgan-particle N_eff margin {margin:.3f}"
    # ...and the control's imbalance is real, not a tie at the ceiling.
    assert _mean(results["particle"], "n_eff") <= 7.1


def test_nextgan_does_not_beat_plain_control_on_n_eff(results):
    # The negative half: the plain control never imbalances in this harness
    # (it shares the KA2-stabilized trainer), so there is no collapse for
    # the field critic to rescue it from. Measured margin -0.05 (a tie).
    # If a future change opens a real gap in either direction, update the
    # verdict and this assertion together.
    assert _mean(results["plain"], "n_eff") >= 7.8
    margin = _mean(results["nextgan"], "n_eff") \
        - _mean(results["plain"], "n_eff")
    assert abs(margin) <= 0.15, f"nextgan-plain N_eff margin {margin:.3f}"


def test_nextgan_captured_fraction_comparable(results):
    # The paper's claim is balance at EQUAL captured fraction: the NextGAN
    # arm must not buy N_eff by abandoning the modes' support.
    for arm in ("particle", "plain"):
        gap = abs(_mean(results["nextgan"], "captured")
                  - _mean(results[arm], "captured"))
        assert gap <= 0.15, f"captured gap vs {arm}: {gap:.3f}"


def test_particle_prior_arms_lag_on_naeem_coverage(results):
    # Both learned-prior arms -- with and without the field critic -- sit
    # far below the frozen-Gaussian arm on Naeem coverage at this budget
    # (measured ~0.04-0.07 vs 0.53): the learned prior's within-mode spread
    # is diffuse. This is a prior property, not a NextGAN effect; the field
    # critic neither causes nor fixes it.
    gap = _mean(results["plain"], "coverage") \
        - max(_mean(results["particle"], "coverage"),
              _mean(results["nextgan"], "coverage"))
    assert gap >= 0.3, f"plain-vs-particle coverage gap {gap:.3f}"
