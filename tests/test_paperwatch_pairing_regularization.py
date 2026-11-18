"""Paper watch: pairing regularization (arXiv:2604.20130) transferred to ParticleGAN.

Paper: "Pairing Regularization for Mitigating Many-to-One Collapse in GANs"
(Lin, Huang, Liu, 2026). A pairing network P(x, z) is trained with a
contrastive loss to identify, within a minibatch, which latent produced each
generated sample; the loss is added to the generator objective only, leaving
the discriminator and the adversarial loss untouched. On their 2D toys the
pairing term rescues a collapsing plain GAN (ring coverage 0.372 -> 0.851).

This test asks the transfer question: does the same term help a system that
already has a learned particle prior? Three arms, identical budget
(2,400 steps, batch 128, seeds 0/1/2), one 2D target (8 Gaussians on a ring,
sigma 0.15) where intramode collapse is directly measurable:

  (a) ParticleGAN control   -- ka2 recipe, learned 256-particle prior.
  (b) Plain GAN control     -- same recipe, frozen fresh-Gaussian prior
                               (the repo's matched non-particle control).
  (c) ParticleGAN + pairing -- (a) plus the pairing loss, delivered through
                               the repo's own PriorManipulator penalty hook
                               (``GANTrainer`` adds it to the generator loss
                               as ``prior_manipulation``); the pairing
                               network's parameters ride the generator
                               optimizer as prior router parameters, exactly
                               where the trainer puts manipulator parameters.

Metrics: Naeem coverage (fraction of real samples whose 5-NN ball within the
real set contains a generated sample) and within-mode dispersion ratio
(mean pairwise distance of generated samples assigned to each mode, divided
by the target's sigma*sqrt(pi); 1.0 is a perfect match, < 1 is intramode
concentration -- the many-to-one collapse the paper names).

Verdict of the cycle-1 gauntlet (10-iteration loop, 6 rounds run): PARTIAL.
Pairing reliably beats the ParticleGAN control on coverage (+0.044 mean at
this budget, ahead in every seed) and relieves its intramode concentration
(dispersion 0.78 -> 1.46, overshooting 1.0). It does not beat the plain GAN
control on either metric (coverage 0.64, dispersion 1.07): the paper's
headline rescue does not transfer past a plain baseline that, given this
budget, was never collapsing. The assertions below encode exactly those
measured margins -- including the negative half -- so a future change that
moves either boundary fails loudly here first.

CPU-only by design; the nine trainings take minutes, not seconds.
Run: python -m pytest tests/test_paperwatch_pairing_regularization.py
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init
from particlegan.priors import PriorManipulator
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator


# --------------------------------------------------------------------------
# The paper's mechanism, as a repo-native prior manipulator.
# --------------------------------------------------------------------------

class PairingManipulator(PriorManipulator):
    """Contrastive latent-identification penalty (arXiv:2604.20130).

    Scores every (generated sample, candidate latent) pair in the minibatch
    and returns the cross-entropy of identifying which latent produced each
    sample. Candidate codes are detached: gradients reach the generator
    (through the samples) and the pairing network (its own parameters),
    never the particle table -- in the paper the latent is exogenous.
    Candidates that repeat the same particle row as the target are masked
    out of the denominator; a repeated row is genuinely unidentifiable.
    """

    config_fields = ("hidden_dim", "z_dim", "x_dim")

    def __init__(self, *, z_dim=2, x_dim=2, hidden_dim=128, penalty_weight=1.0):
        super().__init__(penalty_weight=penalty_weight)
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def regularize(self, raw_rows, *, sampled_codes=None, indices=None,
                   generated=None, logits=None):
        if generated is None or sampled_codes is None:
            return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)
        x = generated
        z = sampled_codes.detach()
        b = x.shape[0]
        pairs = torch.cat([
            x.unsqueeze(1).expand(b, b, x.shape[-1]),
            z.unsqueeze(0).expand(b, b, z.shape[-1]),
        ], dim=-1)
        scores = self.net(pairs).squeeze(-1)  # scores[i, j] = P(x_i, z_j)
        if indices is not None:
            same = indices.unsqueeze(1) == indices.unsqueeze(0)
            eye = torch.eye(b, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(same & ~eye, float("-inf"))
        targets = torch.arange(b, device=x.device)
        return nn.functional.cross_entropy(scores, targets)


# --------------------------------------------------------------------------
# Target and metrics.
# --------------------------------------------------------------------------

N_MODES = 8
RADIUS = 2.0
SIGMA = 0.15


def _ring_means():
    angles = torch.linspace(0, 2 * math.pi, N_MODES + 1)[:-1]
    return torch.stack([angles.cos(), angles.sin()], dim=1) * RADIUS


MEANS = _ring_means()


def sample_mix(n, gen):
    idx = torch.randint(0, N_MODES, (n,), generator=gen)
    return MEANS[idx] + SIGMA * torch.randn(n, 2, generator=gen)


def naeem_coverage(real, fake, k=5):
    """Fraction of real points whose kNN ball (within real) contains a fake."""
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    radius = d_rr.topk(k, largest=False).values[:, -1]
    d_rf = torch.cdist(real, fake)
    return (d_rf.min(dim=1).values <= radius).float().mean().item()


def dispersion_ratio(fake):
    """Within-mode spread of fakes relative to the target's own spread."""
    assign = torch.cdist(fake, MEANS).argmin(dim=1)
    expected = SIGMA * math.sqrt(math.pi)
    ratios = []
    for m in range(N_MODES):
        pts = fake[assign == m]
        if len(pts) >= 8:
            d = torch.cdist(pts, pts)
            iu = torch.triu_indices(len(pts), len(pts), offset=1)
            ratios.append(d[iu[0], iu[1]].mean().item() / expected)
    return sum(ratios) / len(ratios) if ratios else 0.0


# --------------------------------------------------------------------------
# Arms. One protocol for every arm; only the prior (and the pairing
# manipulator) changes.
# --------------------------------------------------------------------------

STEPS = 2400
BATCH = 128
NUM_PARTICLES = 256
Z_DIM = 2
LR = 2e-3
LAMBDA_PAIR = 0.03
SEEDS = (0, 1, 2)


def run_arm(arm, seed):
    torch.manual_seed(seed)
    overrides = dict(z_dim=Z_DIM, num_particles=NUM_PARTICLES, batch_size=BATCH,
                     total_steps=STEPS, lr=LR)
    if arm == "plain":
        overrides.update(prior_update="frozen", prior_code_law="fresh_gaussian")
    recipe = get_recipe(trainer="ka2", **overrides)
    if arm == "pairing":
        prior = recipe.make_prior(manipulators=[
            PairingManipulator(z_dim=Z_DIM, penalty_weight=LAMBDA_PAIR)])
    else:
        prior = recipe.make_prior()
    G = SimpleMLPGenerator(z_dim=Z_DIM, hidden_dim=128, n_hidden=3)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=2)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    if arm != "plain":
        init.deterministic_orthogonal_(prior)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed)
    data_gen = torch.Generator().manual_seed(seed + 10_000)
    for _ in range(STEPS):
        trainer.step(sample_mix(BATCH, data_gen))
    eval_gen = torch.Generator().manual_seed(seed + 999)
    fake = trainer.sample(4096, generator=eval_gen, output_noise=False).cpu()
    real_eval = sample_mix(4096, torch.Generator().manual_seed(777))
    return {"coverage": naeem_coverage(real_eval, fake),
            "dispersion": dispersion_ratio(fake)}


@pytest.fixture(scope="module")
def results():
    return {arm: [run_arm(arm, seed) for seed in SEEDS]
            for arm in ("particle", "plain", "pairing")}


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


# --------------------------------------------------------------------------
# Mechanism unit checks (fast): the penalty is the paper's loss, wired the
# way the trainer promises.
# --------------------------------------------------------------------------

def test_pairing_loss_identifies_codes_and_detaches_them():
    torch.manual_seed(0)
    manip = PairingManipulator(z_dim=2, hidden_dim=32)
    codes = torch.randn(16, 2, requires_grad=True)
    # Stand-in for generator outputs: a leaf the loss can push on.
    generated = (codes.detach() * 2).clone().requires_grad_(True)
    raw = torch.zeros(4, 2)
    loss = manip.regularize(raw, sampled_codes=codes, generated=generated,
                            indices=torch.arange(16))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    # Gradient reaches the generated samples (the generator's side)...
    assert generated.grad is not None and generated.grad.abs().sum() > 0
    # ...and the pairing network...
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in manip.net.parameters())
    # ...but never the candidate codes: in the paper the latent is exogenous.
    assert codes.grad is None


def test_pairing_loss_falls_below_chance_as_identification_is_learned():
    torch.manual_seed(0)
    manip = PairingManipulator(z_dim=2, hidden_dim=32)
    codes = torch.randn(32, 2)
    generated = codes * 3.0  # a fixed injective map: identity is learnable
    raw = torch.zeros(4, 2)
    opt = torch.optim.Adam(manip.net.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = manip.regularize(raw, sampled_codes=codes, generated=generated,
                                indices=torch.arange(32))
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    # Chance is ln(32) = 3.47; a learned identifier sits far below it.
    assert loss.item() < 0.5 * first


# --------------------------------------------------------------------------
# The gauntlet result, encoded. Margins are the cycle-1 measurements with
# headroom; see the module docstring for the full iteration log summary.
# --------------------------------------------------------------------------

def test_pairing_beats_particle_control_on_coverage(results):
    for seed_idx in range(len(SEEDS)):
        assert results["pairing"][seed_idx]["coverage"] \
            > results["particle"][seed_idx]["coverage"]
    margin = _mean(results["pairing"], "coverage") \
        - _mean(results["particle"], "coverage")
    assert margin >= 0.03, f"pairing-particle coverage margin {margin:.3f}"


def test_pairing_relieves_particle_intramode_concentration(results):
    # The control concentrates within modes (many-to-one collapse)...
    assert _mean(results["particle"], "dispersion") <= 0.90
    # ...and pairing relieves it, overshooting the ideal 1.0.
    assert _mean(results["pairing"], "dispersion") >= 1.20


def test_pairing_does_not_beat_plain_control(results):
    # The negative half of the cycle-1 verdict: on this toy the plain
    # fresh-Gaussian GAN was never collapsing at this budget, and pairing
    # on ParticleGAN does not catch it. If a future change closes this
    # gap, update the verdict and this assertion together.
    margin = _mean(results["plain"], "coverage") \
        - _mean(results["pairing"], "coverage")
    assert margin >= 0.10, f"plain-pairing coverage margin {margin:.3f}"
