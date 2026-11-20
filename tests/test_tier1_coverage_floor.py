"""Tier-1 regression: ParticleGAN mode-fill coverage floor.

MODE COUNT IS NOT COVERAGE. This test guards the FILL, not the reach.

The finding (ParticleGAN paper watch, cycles 1-2, 2026-11-17/18): on
low-dimensional mixture toys, ParticleGAN's learned-prior system REACHES
every mode -- mode counts tie a plain GAN control at the ceiling -- while
covering far fewer of the real distribution's neighbourhoods. Measured:

  * 25-Gaussian grid, 6000 steps (PR #563 harness): modes 25.0 vs 25.0,
    Naeem coverage 0.128 (ParticleGAN) vs 0.507 (plain NS-GAN).
  * 8-Gaussian mixture, 2400 steps (PR #560 harness): coverage 0.201
    (ParticleGAN control) vs 0.638 (plain control).
  * 8-Gaussian ring, NextGAN harness (PR #564): both particle-prior arms
    lag the plain arm on coverage (~0.04-0.08 vs ~0.47-0.61) at every
    budget -- a diffuse-within-mode property of the learned prior.

A mode-count assertion alone certifies these runs as healthy while the
samples concentrate inside the modes instead of filling them. This test
exists so that failure mode can never pass silently: it asserts BOTH
that ParticleGAN reaches every mode AND that its Naeem coverage stays
above a floor calibrated from measured runs (below). The plain-GAN
control is trained in the same fixture and its coverage gap is RECORDED
in the test output -- parity with the plain control at the DEFAULT
particle count is the aspiration, not today's truth, so the gap is
measured and printed, never asserted. Raising the floor is the expected
direction as the formulation improves; if a change legitimately raises
coverage, raise FLOOR and update the calibration table in the same
commit.

WHY the gap exists -- and the lever that closes it (particle-count
sweep, paper watch 2026-11-18): the per-particle concentration never
moves. On the 25-Gaussian grid (BOLT harness, 6000 steps, seeds 0/1/2),
within-mode dispersion stays 0.61-0.70 at EVERY particle count while
the plain control sits at 1.32; what changes with count is how many
clusters the prior can place. Coverage climbs monotonically with
particle count -- 256: 0.128 +/- 0.020, 1,024: 0.208, 4,096: 0.347,
16,384: 0.510 +/- 0.049 -- reaching full parity with the plain control
(0.507 +/- 0.058) at 64x the default count, at flat wall time
(~3 min/run at every count). The ring fixture below reproduces the same
law at its own scale: 256 particles average 0.201 coverage, 16,384
average 0.735 -- above this fixture's plain control (0.638) in every
seed, also at flat wall time (~45 s/run at both counts). So the floor
is a statement about the DEFAULT configuration, not a ceiling on the
formulation: concentration per particle is fixed; count is the lever
that moves delivered coverage. The scaling assertion below pins that
law so a future change that breaks it -- in either direction -- fails
loudly here.

Toy choice (measured, not convenience): the 25-Gaussian grid from
PR #563 was probed on pop-os CPU: one particle-arm run at a quarter of
its shipped budget (1500 of 6000 steps, batch 256) took 110 s
(~0.073 s/step; the regime is already present there -- 25/25 modes,
coverage 0.208), so the full budget costs ~7.5 min for a single run and
~45 min for this test's six-run fixture: outside any tier-1 budget.
The 8-Gaussian ring below (the PR #560
configuration: radius 2.0, sigma 0.15, batch 128, 2400 steps) reproduces
the same regime -- every mode reached in every seed with >= 433 of 4096
samples inside the least-covered mode's 3-sigma ball, particle coverage
far below the plain control -- at roughly a sixth of the cost.

Calibration (pop-os CPU, this exact protocol, master @ 52c26c1b):

    arm       seed   modes   min-mode count   coverage   train secs*
    particle    0      8/8         451          0.1807       46
    particle    1      8/8         459          0.2129       73
    particle    2      8/8         433          0.2095      101
    plain       0      8/8         410          0.6667      250
    plain       1      8/8         424          0.6646      129
    plain       2      8/8         409          0.5815      213
    (* wall times under heavy shared-box load; ~60 s/run uncontended,
     cf. PR #560's 9-run file at 9m33s total on the same box.)

Particle coverage: mean 0.201, min 0.181, sd 0.018 -- matching PR #560's
independent 0.201 +/- 0.014. FLOOR = 0.14 = observed minimum (0.181)
minus a 0.04 safety margin (~2.3 seed-sigmas), which also absorbs the
small numeric drift that CPU thread-count differences can cause. The
floor sits far above the failure values this test exists to catch
(0.04-0.08 in the PR #564 ring runs; 0.048 for the collapsed BOLT arm).

Scaling arm calibration (same protocol, num_particles = 16,384):

    arm       seed   modes   min-mode count   coverage   train secs
    part-16k    0      8/8         444          0.7891       49
    part-16k    1      8/8         476          0.7224       34
    part-16k    2      8/8         463          0.6941       47

16k coverage: mean 0.735, min 0.694 -- the WORST 16k seed beats the
plain control's MEAN (0.638) and its best seed (0.667). The parity
assertion uses bands well inside these margins (per-seed >= plain
mean - 0.10; mean >= plain mean - 0.05; count lever >= +0.35 over the
default-count mean, measured +0.53).

Determinism: every seed is fixed (init, data, eval); the test is
CPU-pinned so CI machines without a GPU run the identical protocol.

Note: ``naeem_coverage`` is implemented in this file because master's
``lib/toy_metrics.py`` does not carry it yet -- it exists on PR #559's
unmerged branch. When #559 lands, switch to the lib version and delete
the local copy.

Actual-training GIF (toy-test contract, reports/toy_audit/api_contract):
seed 0 of each arm is snapshotted every GIF_EVERY updates with the same
separate evaluation stream as the gated endpoint (``trainer.sample`` never
consumes a training RNG, so the trajectory and calibration are unchanged).
``test_actual_training_gif`` renders the observed samples beside the
target ring, labelled with update count, live coverage and the floor/gate
status; the last frame IS the gated endpoint sample. The file is written
to ``$T1_COVERAGE_GIF`` when set, else to pytest's tmp dir; the committed
copy is docs/media/t1_coverage_floor/training.gif.

Cost: the module fixture trains nine CPU runs (~2.5-5 min on CI runners),
so the module carries the repository's ``tier1`` marker for selection
(``-m tier1`` / ``-m 'not tier1'``); it stays in the default CI suite.

Run: python -m pytest tests/test_tier1_coverage_floor.py
"""
from __future__ import annotations

import math

import pytest
import torch

from particlegan import GANTrainer, get_recipe, init
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator

pytestmark = pytest.mark.tier1


# --------------------------------------------------------------------------
# Target: 8 Gaussians on a ring (the PR #560 configuration).
# --------------------------------------------------------------------------

N_MODES = 8
RADIUS = 2.0
SIGMA = 0.15
MODE_RADIUS = 3.0 * SIGMA  # a sample counts toward a mode inside 3 sigma


def _ring_means():
    angles = torch.linspace(0, 2 * math.pi, N_MODES + 1)[:-1]
    return torch.stack([angles.cos(), angles.sin()], dim=1) * RADIUS


MEANS = _ring_means()


def sample_mix(n, gen):
    idx = torch.randint(0, N_MODES, (n,), generator=gen)
    return MEANS[idx] + SIGMA * torch.randn(n, 2, generator=gen)


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------

def real_knn_radius(real, k=5):
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    return d_rr.topk(k, largest=False).values[:, -1]


def naeem_coverage(real, fake, k=5, radius=None):
    """Fraction of real points whose kNN ball (within real) contains a fake.

    Naeem et al. (2020) coverage, as used by the paper-watch gauntlets
    (PRs #560/#563/#564). Self-contained copy: master's lib/toy_metrics.py
    does not carry this metric yet (see module docstring).
    """
    if radius is None:
        radius = real_knn_radius(real, k)
    d_rf = torch.cdist(real, fake)
    return (d_rf.min(dim=1).values <= radius).float().mean().item()


def modes_reached(fake):
    """Number of modes with >= 1 fake sample within 3 sigma of the center."""
    dists = torch.cdist(fake, MEANS)
    return int((dists.min(dim=0).values <= MODE_RADIUS).sum().item())


# --------------------------------------------------------------------------
# Arms. One protocol for both; only the prior changes. CPU-pinned: tier-1
# runs the identical protocol on machines with or without a GPU.
# --------------------------------------------------------------------------

STEPS = 2400
BATCH = 128
NUM_PARTICLES = 256
Z_DIM = 2
LR = 2e-3
SEEDS = (0, 1, 2)
N_EVAL = 4096

# Calibrated from the measured runs in the module docstring: observed
# minimum 0.181 across seeds 0/1/2 minus a 0.04 safety margin.
COVERAGE_FLOOR = 0.14

# The scaling arm: 64x the default count, where the count sweep found
# plain-control parity (grid: 0.510 vs 0.507; ring: see docstring).
NUM_PARTICLES_SCALED = 16_384

# Actual-training GIF cadence (seed 0 of each arm; observation only).
GIF_EVERY = 200
GIF_POINTS = 1024
ARM_LABELS = {"particle": f"ParticleGAN ({NUM_PARTICLES} particles)",
              "plain": "plain control (frozen Gaussian)",
              "particle_16k": f"ParticleGAN ({NUM_PARTICLES_SCALED:,} particles)"}


def run_arm(arm, seed, snapshots=False):
    torch.manual_seed(seed)
    n_particles = NUM_PARTICLES_SCALED if arm == "particle_16k" else NUM_PARTICLES
    overrides = dict(z_dim=Z_DIM, num_particles=n_particles, batch_size=BATCH,
                     total_steps=STEPS, lr=LR)
    if arm == "plain":
        # The repo's matched non-particle control: same recipe, frozen
        # fresh-Gaussian prior (PRs #560/#563).
        overrides.update(prior_update="frozen", prior_code_law="fresh_gaussian")
    recipe = get_recipe(trainer="ka2", **overrides)
    prior = recipe.make_prior()
    G = SimpleMLPGenerator(z_dim=Z_DIM, hidden_dim=128, n_hidden=3)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=2)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    if arm != "plain":
        init.deterministic_orthogonal_(prior)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed)
    data_gen = torch.Generator().manual_seed(seed + 10_000)
    real_eval = sample_mix(N_EVAL, torch.Generator().manual_seed(777))
    real_radius = real_knn_radius(real_eval)  # fixed target: compute once

    def observe():
        # Same separate evaluation stream as the gated endpoint; it never
        # touches a training RNG (GANTrainer.sample contract).
        eval_gen = torch.Generator().manual_seed(seed + 999)
        return trainer.sample(N_EVAL, generator=eval_gen, output_noise=False).cpu()

    frames = []

    def snapshot(step):
        fake = observe()
        frames.append({"step": step, "points": fake[:GIF_POINTS].clone(),
                       "modes": modes_reached(fake),
                       "coverage": naeem_coverage(real_eval, fake, radius=real_radius)})

    finite = True
    if snapshots:
        snapshot(0)
    for step in range(1, STEPS + 1):
        out = trainer.step(sample_mix(BATCH, data_gen))
        if not (torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])):
            finite = False
            break
        if snapshots and step % GIF_EVERY == 0 and step != STEPS:
            snapshot(step)
    fake = observe()
    row = {"arm": arm, "seed": seed, "finite": finite,
           "modes": modes_reached(fake),
           "coverage": naeem_coverage(real_eval, fake, radius=real_radius)}
    if snapshots:
        frames.append({"step": trainer.completed_steps, "points": fake[:GIF_POINTS].clone(),
                       "modes": row["modes"], "coverage": row["coverage"]})
        row["frames"] = frames
    return row


@pytest.fixture(scope="module")
def results():
    return {arm: [run_arm(arm, seed, snapshots=(seed == SEEDS[0])) for seed in SEEDS]
            for arm in ("particle", "plain", "particle_16k")}


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


# --------------------------------------------------------------------------
# Metric sanity (fast): the coverage implementation itself.
# --------------------------------------------------------------------------

def test_naeem_coverage_metric_sanity():
    gen = torch.Generator().manual_seed(777)
    real = sample_mix(2048, gen)
    # A generator that IS the target covers (essentially) everything.
    assert naeem_coverage(real, sample_mix(2048, torch.Generator().manual_seed(778))) > 0.9
    # A generator collapsed onto one mode covers only that mode's share.
    one_mode = MEANS[0] + SIGMA * torch.randn(2048, 2, generator=torch.Generator().manual_seed(779))
    assert naeem_coverage(real, one_mode) < 0.3
    # A generator nowhere near the target covers nothing.
    far = torch.full((2048, 2), 100.0)
    assert naeem_coverage(real, far) == 0.0


# --------------------------------------------------------------------------
# The tier-1 assertions: reach AND fill.
# --------------------------------------------------------------------------

def test_particle_reaches_every_mode(results):
    # The reach half -- necessary, and (per the finding) nowhere near
    # sufficient. Calibration: 8/8 in every seed, with >= 433 of 4096
    # samples inside the least-covered mode's 3-sigma ball.
    for row in results["particle"]:
        assert row["finite"], f"seed {row['seed']}: training went non-finite"
        assert row["modes"] == N_MODES, (
            f"seed {row['seed']}: reached {row['modes']}/{N_MODES} modes")


def test_particle_coverage_stays_above_floor(results):
    # The fill half -- the assertion this file exists for. A run that
    # reaches every mode while its samples concentrate inside them
    # fails here even though the mode count above is perfect.
    for row in results["particle"]:
        assert row["coverage"] >= COVERAGE_FLOOR, (
            f"seed {row['seed']}: coverage {row['coverage']:.3f} below the "
            f"tier-1 floor {COVERAGE_FLOOR} (calibration: min 0.181, "
            f"mean 0.201 over seeds 0/1/2; see module docstring)")


def test_particle_coverage_scales_to_plain_parity_with_count(results):
    # The mechanism assertion: per-particle concentration is fixed, so
    # delivered coverage is bought with particle COUNT. At 64x the
    # default count the particle system reaches the plain control's
    # parity band (grid sweep: 0.510 vs plain 0.507; this fixture's
    # calibration: 16k mean 0.735, worst seed 0.694, vs plain mean
    # 0.638). Bands sit well inside the measured margins. If a future
    # change makes the default count fill like 16k does today, the
    # floor test above should be raised instead of this one failing --
    # update both together, with fresh calibration.
    plain_mean = _mean(results["plain"], "coverage")
    default_mean = _mean(results["particle"], "coverage")
    scaled = results["particle_16k"]
    for row in scaled:
        assert row["finite"], f"16k seed {row['seed']}: non-finite losses"
        assert row["modes"] == N_MODES, (
            f"16k seed {row['seed']}: reached {row['modes']}/{N_MODES} modes")
        assert row["coverage"] >= plain_mean - 0.10, (
            f"16k seed {row['seed']}: coverage {row['coverage']:.3f} below "
            f"the plain parity band (plain mean {plain_mean:.3f} - 0.10)")
    scaled_mean = _mean(scaled, "coverage")
    assert scaled_mean >= plain_mean - 0.05, (
        f"16k mean coverage {scaled_mean:.3f} below the plain parity "
        f"band (plain mean {plain_mean:.3f} - 0.05)")
    assert scaled_mean - default_mean >= 0.35, (
        f"count lever too weak: 16k mean {scaled_mean:.3f} vs default "
        f"count mean {default_mean:.3f} (calibration margin +0.53)")
    print(f"\nT1-COVERAGE-SCALING default={default_mean:.3f} "
          f"16k={scaled_mean:.3f} plain={plain_mean:.3f}")


def test_plain_control_gap_is_measured_and_recorded(results):
    # The plain control is trained under the identical protocol and the
    # coverage gap AT THE DEFAULT COUNT is recorded in the test output.
    # Parity at the default count is NOT asserted: closing this gap is
    # the formulation's work, and the day it closes, this record is the
    # evidence. (Parity at 64x count IS asserted, in the scaling test
    # above.) Calibration gap: plain mean 0.638 vs particle mean 0.201
    # (gap ~0.44).
    particle_mean = _mean(results["particle"], "coverage")
    plain_mean = _mean(results["plain"], "coverage")
    gap = plain_mean - particle_mean
    for row in results["plain"]:
        assert row["finite"], f"plain seed {row['seed']}: non-finite losses"
        assert row["modes"] == N_MODES, (
            f"plain seed {row['seed']}: control reached only "
            f"{row['modes']}/{N_MODES} modes -- the control itself is broken")
    assert math.isfinite(gap)
    print(f"\nT1-COVERAGE-GAP particle={particle_mean:.3f} "
          f"plain={plain_mean:.3f} gap={gap:+.3f} "
          f"(floor={COVERAGE_FLOOR}; gap recorded, not asserted)")


# --------------------------------------------------------------------------
# Actual-training GIF (toy-test contract): observed samples vs target ring.
# --------------------------------------------------------------------------

def render_training_gif(results, path):
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    arms = ("particle", "plain", "particle_16k")
    series = {arm: results[arm][0]["frames"] for arm in arms}
    n_frames = min(len(frames) for frames in series.values())
    target = sample_mix(GIF_POINTS, torch.Generator().manual_seed(777))
    images = []
    for i in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=72)
        for ax, arm in zip(axes, arms):
            frame = series[arm][i]
            ax.scatter(target[:, 0], target[:, 1], s=3, c="#b0b0b0", label="target")
            pts = frame["points"]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, c="#d1495b", alpha=0.6, label="API samples")
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.2, 3.2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if arm == "particle":
                gate = "PASS" if frame["coverage"] >= COVERAGE_FLOOR else "below floor"
                gate = f"floor {COVERAGE_FLOOR}: {gate}"
            else:
                gate = "control" if arm == "plain" else "parity arm"
            ax.set_title(f"{ARM_LABELS[arm]}\nmodes {frame['modes']}/{N_MODES}  "
                         f"coverage {frame['coverage']:.3f}\n{gate}", fontsize=8)
        step = series["particle"][i]["step"]
        final = " (gated endpoint)" if i == n_frames - 1 else ""
        fig.suptitle(f"T1 coverage floor, seed {SEEDS[0]}: update {step}/{STEPS}{final}",
                     fontsize=10)
        axes[0].legend(loc="lower left", fontsize=6, markerscale=3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
    durations = [400] * (n_frames - 1) + [2500]
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    return n_frames


def test_actual_training_gif(results, tmp_path):
    import os
    from pathlib import Path

    path = Path(os.environ.get("T1_COVERAGE_GIF") or tmp_path / "training.gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = render_training_gif(results, path)
    assert n_frames == STEPS // GIF_EVERY + 1
    # The last frame is the gated endpoint itself, not a re-sample.
    for arm in ("particle", "plain", "particle_16k"):
        row = results[arm][0]
        assert row["frames"][-1]["step"] == STEPS
        assert row["frames"][-1]["coverage"] == row["coverage"]
    assert path.stat().st_size > 1000
    print(f"\nT1-COVERAGE-GIF {path} frames={n_frames}")
