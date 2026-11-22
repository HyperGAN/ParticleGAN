"""Paper watch cycle 11 (directed): does the e-means EMA scaling law
transfer to EMPF's (K, B) hit-rate EMA? -- NEGATIVE, with mechanism.

INTERNAL EXPERIMENT, NOT A PAPER REPLICATION. This test was directed by
the watch (cycle 11) after cycle 10's deep pass found that EMPF's own
docstring names its ancestor: e-means (Bessa et al., arXiv:2611.09501).
The question is ours, not the paper's: EMPF already sets its prototype
bank's EMA rate by a hand-shaped rule "inspired by" e-means; does the
paper's FITTED scaling law, mapped onto EMPF's quantities, do better?

THE TWO RULES. Both set the per-batch EMA rate alpha applied to the
sufficient statistics (S, N) -- the identical slot in the identical
update (e-means Eq. 2: N <- gamma*N + (1-gamma)*n, alpha = 1 - gamma;
EMPF prototypes.py: new = (1 - alpha_eff)*old + alpha_eff*batch).

  repo rule (shipped):  alpha = clip(0.08 * (B/K)^0.45, 0.03, 0.35)
  e-means decay law (their Eq. 6, Table 2; fit on 62,410 runs,
  1e5 <= n <= 1e8, 256 <= k <= 65,536, n/k >= 8; R^2 0.95):
      alpha_0 = min(0.824 * k^-0.46 * B^0.90 * n^-0.45 * T^-0.11, 1)

THE MAPPING IS PARTIAL -- stated plainly, because it decides the result.
k <-> K and B <-> B correspond exactly, and alpha is the same quantity
in the same equation. But n (fixed dataset size) and T (epochs over
that dataset) have NO counterpart in EMPF: the bank consumes an
open-ended stream inside an adversarial loop, which is precisely why
EMPF's designers dropped the clocks ("clock-free (K, B) hit-rate EMA
scaling", empf.py). The n factor is not a detail -- at the paper's fit
range it shrinks alpha_0 by ~200x; it encodes "how much data will ever
pass by", a question EMPF never asks. What CAN be mapped is the (k, B)
shape: alpha ~ B^0.90 * K^-0.46 versus the repo's (B/K)^0.45. The
experiment therefore tests the SHAPE, with the overall constant
calibrated so both rules agree at the reference configuration
(B, K) = (128, 64), alpha = 0.1093. At fixed K the shape is expressible
through the repo's own public knobs (penalty_options alpha_base /
hit_exponent): hit_exponent 0.90, alpha_base 0.0586. The update
equations are byte-identical in both arms; only knob VALUES differ.
Not mapped, by design: the cosine step schedule (their Eq. 5) -- it
needs the horizon clock EMPF removed, and the paper's own ablation
prices it at ~0.26% MSE; the batch law (Eq. 7) -- it also needs n, T,
and B is the trainer's choice, not the bank's.

PREDICTION (if the shape transfers): the rules separate only away from
the anchor -- at K = 64, e-means prescribes alpha 0.0586 at B = 64
(repo: 0.080) and 0.2039 at B = 256 (repo: 0.1493). The e-means arm
should track better at both off-anchor batch sizes.

MEASURED (pop-os, CPU execution -- the trainer defaults to CPU in
this harness -- grid protocol; full logs in PR. Per-seed endpoints
are execution-regime-sensitive, see run_gan; the seed-mean bands
asserted below held in every regime executed):
  Iteration 1 -- GAN level, 8-Gaussian ring (radius 2.0, sigma 0.15),
  default empf recipe, K = 64, 1200 steps, seeds 0/1/2, coverage /
  dispersion means:
      B=64:   repo 0.135 / 0.487    emeans 0.140 / 0.560
      B=128:  repo 0.131 / 0.707    (anchor: rules identical --
              bit-identical metrics in the seed-0 twin run)
      B=256:  repo 0.213 / 0.691    emeans 0.192 / 0.912
      plain control: 0.488-0.561 coverage, ~1.0 dispersion, every B.
  No separation: differences sit inside seed noise at both off-anchor
  batch sizes; the B=256 emeans mean is dragged by one unstable seed
  (seed 1: 6/8 modes, coverage 0.107, dispersion 1.34) at the highest
  alpha tested (0.204).
  Iteration 2 -- bank level, the regime the law actually governs:
  standalone bank (the kernel's own update path) on a static ring
  stream, quantization MSE of the prototypes vs the target:
      update 200, means: B=64 repo 0.00995 vs emeans 0.00988;
      B=256 repo 0.01017 vs emeans 0.01015; identical at the anchor.
  The bank reaches its plateau (q_mse ~0.010) within ~25-50 updates
  under EVERY alpha in [0.059, 0.204]; residual differences are seed
  noise from the shared stratified init. On a static real stream the
  bank is converged for ~97% of a 1200-step run under both rules.

VERDICT: NEGATIVE. The e-means (k, B) shape does not transfer as a
measurable improvement at either level, and the mechanism is visible:
e-means' law optimizes a few-epoch transient that IS the whole e-means
run (their headline is convergence in 1.4-2.6 epochs); EMPF's bank
lives for thousands of steps on a stationary real distribution, where
the transient is a rounding error and the (n, T) factors that make
the law bind have no referent. The repo's (B/K)^0.45 rule is not
measurably wrong on this surface. The coverage lever for this
formulation remains particle COUNT (cycles 3/9, PR #570), not the
bank's alpha rule. Stopped at 2 iterations: both levels agree, the
null is mechanistically explained, and further budget refinements
would chase seed noise.

The assertions below encode exactly this: the anchor identity and
off-anchor separation FACTORS of the two rules (arithmetic), the bank
plateau equivalence (iteration 2 bands), and the GAN-level equivalence
bands + the standing plain-control gap (iteration 1 bands). If a
future change makes alpha bind -- a drifting target, a much larger
K, a harder assignment problem -- these bands are where it shows up.

Run: python -m pytest tests/test_paperwatch_emeans_ema_scaling.py
"""
from __future__ import annotations

import math

import pytest
import torch

from particlegan import GANTrainer, get_recipe, init
from particlegan.empf import emeans_effective_alpha, update_emeans_prototypes
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator


# --------------------------------------------------------------------------
# Target: 8 Gaussians on a ring (the PR #560 / PR #570 configuration).
# --------------------------------------------------------------------------

N_MODES = 8
RADIUS = 2.0
SIGMA = 0.15
MODE_RADIUS = 3.0 * SIGMA

_ANGLES = torch.linspace(0, 2 * math.pi, N_MODES + 1)[:-1]
MEANS = torch.stack([_ANGLES.cos(), _ANGLES.sin()], dim=1) * RADIUS


def sample_mix(n, gen):
    idx = torch.randint(0, N_MODES, (n,), generator=gen)
    return MEANS[idx] + SIGMA * torch.randn(n, 2, generator=gen)


def naeem_coverage(real, fake, k=5):
    """Naeem et al. (2020) coverage, as in PRs #560/#570."""
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    radius = d_rr.topk(k, largest=False).values[:, -1]
    d_rf = torch.cdist(real, fake)
    return (d_rf.min(dim=1).values <= radius).float().mean().item()


def dispersion_ratio(fake):
    """Within-mode spread of fakes relative to the target's own spread
    (PR #560's implementation)."""
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


def modes_reached(fake):
    dists = torch.cdist(fake, MEANS)
    return int((dists.min(dim=0).values <= MODE_RADIUS).sum().item())


# --------------------------------------------------------------------------
# The two alpha rules, expressed through the repo's own knobs.
# --------------------------------------------------------------------------

K_BANK = 64
REF_BATCH = 128
REF_ALPHA = emeans_effective_alpha(REF_BATCH, K_BANK)  # repo rule: 0.1093

# e-means (k, B) shape alpha ~ B^0.90 * K^-0.46 at fixed K, written as
# alpha_base' * (B/K)^0.90 with alpha_base' chosen so the rules agree
# at the reference batch: alpha_base' = REF_ALPHA / (REF_BATCH/K)^0.90.
EMEANS_ALPHA_BASE = REF_ALPHA / (REF_BATCH / K_BANK) ** 0.90
EMEANS_HIT_EXPONENT = 0.90


def rule_alpha(rule, batch):
    if rule == "emeans":
        return emeans_effective_alpha(
            batch, K_BANK, alpha_base=EMEANS_ALPHA_BASE,
            hit_exponent=EMEANS_HIT_EXPONENT)
    return emeans_effective_alpha(batch, K_BANK)


def rule_penalty_options(rule):
    opts = {"max_prototypes": K_BANK}
    if rule == "emeans":
        opts.update(alpha_base=EMEANS_ALPHA_BASE,
                    hit_exponent=EMEANS_HIT_EXPONENT)
    return opts


def test_rule_mapping_anchor_and_separation():
    # The rules coincide at the reference configuration by construction.
    assert abs(rule_alpha("repo", REF_BATCH)
               - rule_alpha("emeans", REF_BATCH)) < 2e-3
    # And separate off-anchor by the predicted factors: e-means steps
    # DOWN at small batch (0.0586 vs 0.080) and UP at large batch
    # (0.2039 vs 0.1493).
    assert rule_alpha("emeans", 64) < 0.85 * rule_alpha("repo", 64)
    assert rule_alpha("emeans", 256) > 1.25 * rule_alpha("repo", 256)
    # The shipped repo rule is a pure hit-rate rule: doubling B and K
    # together leaves it unchanged. The e-means shape is NOT (it keeps
    # a K^+0.44 residual) -- the structural difference under test.
    assert abs(emeans_effective_alpha(256, 128)
               - emeans_effective_alpha(128, 64)) < 1e-9


# --------------------------------------------------------------------------
# Iteration 2: bank-level transient -- the regime the law governs.
# --------------------------------------------------------------------------

BANK_UPDATES = 200
BANK_BATCHES = (64, 256)
SEEDS = (0, 1, 2)


def run_bank(rule, batch, seed):
    kw = {}
    if rule == "emeans":
        kw = {"alpha_base": EMEANS_ALPHA_BASE,
              "hit_exponent": EMEANS_HIT_EXPONENT}
    gen = torch.Generator().manual_seed(seed + 10_000)
    S = N = None
    phase, revivals = 0.0, 0
    C = None
    for _ in range(BANK_UPDATES):
        x = sample_mix(batch, gen)
        C, _w, S, N, phase, revivals = update_emeans_prototypes(
            x, S, N, phase, revivals, max_prototypes=K_BANK, **kw)
    eval_real = sample_mix(8192, torch.Generator().manual_seed(777))
    q_mse = torch.cdist(eval_real, C).min(dim=1).values.pow(2).mean().item()
    return q_mse


@pytest.fixture(scope="module")
def bank_results():
    return {rule: {b: [run_bank(rule, b, s) for s in SEEDS]
                   for b in BANK_BATCHES}
            for rule in ("repo", "emeans")}


def test_bank_reaches_plateau_under_both_rules(bank_results):
    # Measured plateau q_mse ~0.0095-0.0102 (all arms, update 200).
    for rule, by_batch in bank_results.items():
        for batch, qs in by_batch.items():
            for q in qs:
                assert q < 0.013, (
                    f"{rule} B={batch}: bank q_mse {q:.5f} above the "
                    f"measured plateau band (< 0.013)")


def test_bank_plateau_equivalence(bank_results):
    # The negative result, encoded: at update 200 the two rules' mean
    # quantization errors agree within 15% relative at both off-anchor
    # batch sizes (measured: within 1%). A future change that makes
    # the alpha rule bind at bank level breaks this band.
    for batch in BANK_BATCHES:
        repo_mean = sum(bank_results["repo"][batch]) / len(SEEDS)
        emeans_mean = sum(bank_results["emeans"][batch]) / len(SEEDS)
        rel = abs(emeans_mean - repo_mean) / repo_mean
        assert rel < 0.15, (
            f"B={batch}: bank q_mse means repo {repo_mean:.5f} vs "
            f"emeans {emeans_mean:.5f} differ by {rel:.1%} -- the "
            f"alpha rule now binds at bank level; revisit the verdict")
        print(f"\nEMEANS-BANK B={batch} repo={repo_mean:.5f} "
              f"emeans={emeans_mean:.5f} rel={rel:.3f}")


# --------------------------------------------------------------------------
# Iteration 1: GAN level -- default empf recipe, only the alpha-rule
# knobs differ between the particle arms.
# --------------------------------------------------------------------------

STEPS = 1200
GAN_BATCHES = (64, 256)
N_EVAL = 4096


def run_gan(rule, batch, seed):
    # Per-run RNG reseeding (CPU + CUDA). Note, measured during this
    # test's verification: per-seed trajectories of this 1200-step
    # protocol are NOT bit-stable across execution regimes -- CPU
    # reduction order (thread count) perturbs the trajectory
    # chaotically, so a given seed's endpoint is effectively a fresh
    # draw from the same distribution (grid regime, 2 threads: repo
    # B=64 seeds land at coverage 0.128-0.143, 8/8 modes; this file's
    # regime: the same seeds land at 0.073-0.077, 5-7/8 modes). The
    # experiment's claims therefore live in the seed-MEAN bands of the
    # tests below, which held in every regime executed; per-run
    # assertions are sanity floors only, calibrated across regimes.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    overrides = dict(z_dim=2, num_particles=256, batch_size=batch,
                     total_steps=STEPS)
    if rule == "plain":
        overrides.update(prior_update="frozen",
                         prior_code_law="fresh_gaussian")
    recipe = get_recipe("empf", **overrides)
    prior = recipe.make_prior()
    G = SimpleMLPGenerator(z_dim=2, hidden_dim=128, n_hidden=3)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=2)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    if rule != "plain":
        init.deterministic_orthogonal_(prior)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed,
                         penalty_options=rule_penalty_options(
                             "repo" if rule == "plain" else rule))
    gen = torch.Generator().manual_seed(seed + 10_000)
    finite = True
    for _ in range(STEPS):
        out = trainer.step(sample_mix(batch, gen))
        if not (torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])):
            finite = False
            break
    fake = trainer.sample(N_EVAL, generator=torch.Generator().manual_seed(seed + 999),
                          output_noise=False).cpu()
    real_eval = sample_mix(N_EVAL, torch.Generator().manual_seed(777))
    return {"rule": rule, "batch": batch, "seed": seed, "finite": finite,
            "modes": modes_reached(fake),
            "coverage": naeem_coverage(real_eval, fake),
            "dispersion": dispersion_ratio(fake)}


@pytest.fixture(scope="module")
def gan_results():
    out = {}
    for rule in ("repo", "emeans", "plain"):
        for batch in GAN_BATCHES:
            out[(rule, batch)] = [run_gan(rule, batch, s) for s in SEEDS]
    return out


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def test_gan_runs_are_finite_and_reach_modes(gan_results):
    # Sanity floor only (the claims live in the mean-band tests).
    # Observed worst cases across execution regimes (see run_gan):
    # grid regime -- emeans B=256 seed 1 at 6/8 (the unstable run at
    # alpha 0.204); this file's regime -- repo B=64 seed 1 at 5/8,
    # reproduced identically in three full verification runs. A
    # collapsed run reaches 1-3 modes; 5/8 keeps the gate meaningful.
    for (rule, batch), rows in gan_results.items():
        for r in rows:
            assert r["finite"], f"{rule} B={batch} seed {r['seed']}: non-finite"
            assert r["modes"] >= 5, (
                f"{rule} B={batch} seed {r['seed']}: reached only "
                f"{r['modes']}/8 modes")


def test_gan_alpha_rule_equivalence(gan_results):
    # The negative result at GAN level, encoded with the measured
    # bands: coverage means within 0.06 (measured deltas +0.005 at
    # B=64, -0.021 at B=256) and dispersion means within 0.30
    # (measured +0.073 / +0.221, the latter inflated by the unstable
    # seed documented above).
    for batch in GAN_BATCHES:
        repo_cov = _mean(gan_results[("repo", batch)], "coverage")
        emeans_cov = _mean(gan_results[("emeans", batch)], "coverage")
        assert abs(emeans_cov - repo_cov) < 0.06, (
            f"B={batch}: coverage repo {repo_cov:.3f} vs emeans "
            f"{emeans_cov:.3f} -- the alpha rule now binds at GAN level")
        repo_disp = _mean(gan_results[("repo", batch)], "dispersion")
        emeans_disp = _mean(gan_results[("emeans", batch)], "dispersion")
        assert abs(emeans_disp - repo_disp) < 0.30, (
            f"B={batch}: dispersion repo {repo_disp:.3f} vs emeans "
            f"{emeans_disp:.3f} -- outside the measured equivalence band")
        print(f"\nEMEANS-GAN B={batch} coverage repo={repo_cov:.3f} "
              f"emeans={emeans_cov:.3f} | dispersion repo={repo_disp:.3f} "
              f"emeans={emeans_disp:.3f}")


def test_gan_plain_gap_stands_under_both_rules(gan_results):
    # Neither alpha rule closes the standing particle-vs-plain gap
    # (measured plain means 0.488 / 0.561 vs particle 0.13-0.21):
    # the alpha rule is not the coverage lever; particle count is
    # (PR #570). Encoded as a floor on the gap so the day either rule
    # starts closing it, this test says so.
    for batch in GAN_BATCHES:
        plain_cov = _mean(gan_results[("plain", batch)], "coverage")
        for rule in ("repo", "emeans"):
            cov = _mean(gan_results[(rule, batch)], "coverage")
            assert plain_cov - cov > 0.15, (
                f"B={batch}: {rule} coverage {cov:.3f} within 0.15 of "
                f"plain {plain_cov:.3f} -- the alpha rule started "
                f"closing the plain gap; revisit the verdict")
