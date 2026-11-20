"""Paper watch: BOLT-GAN critic (arXiv:2510.25609) transferred to ParticleGAN.

Paper: "Revisiting GAN with Bayes-Optimal Discrimination" (Naeini et al.,
2026). The discriminator is a bounded score h(x) = sigmoid(f(x)) in [0, 1]
trained with the BOLT loss, which is LINEAR in the score:
l_BOLT(h(x), C) = (-1)^C h(x) with C=1 real, C=2 fake. With balanced
priors the critic therefore minimizes E[h(fake)] - E[h(real)], and the
generator descends -E[h(fake)], maximizing a surrogate of the Bayes error
rate. Unconstrained, this targets total variation and diverges in
practice (the paper's Max-BER ablation, FID ~380); constraining the raw
score f to be 1-Lipschitz -- in the paper via a WGAN-GP-style gradient
penalty on the RAW score (Appendix H: "all gradient penalties are
applied to the raw score"), lambda_GP in {5, 10} robust, GP-only the
default -- bounds the discrepancy by W1 and stabilizes it ("BOLT-GAN").
On the 25-Gaussians toy (Appendix G.4: 5x5 grid, spacing 1, sigma 0.05,
modes counted with >= 1 sample within radius 0.15 = 3 sigma) BOLT-GAN
and WGAN recover 25.0 +/- 0.0 modes; the non-saturating GAN collapses
to 6.0 +/- 1.4.

This test asks the transfer question: does swapping ParticleGAN's critic
objective for BOLT-GP help a system that already has a learned particle
prior? Three arms, identical budget (steps, batch, seeds), one 2D target
(the paper's 25-Gaussians):

  (a) ParticleGAN control -- ka2 recipe, learned 256-particle prior,
      the repo's standard (relativistic) critic, untouched.
  (b) Plain NS-GAN control -- same recipe with the non-saturating loss
      and a frozen fresh-Gaussian prior (the repo's matched non-particle
      control; the paper says this family collapses to ~6/25 modes).
  (c) ParticleGAN + BOLT-GP -- (a)'s trainer, prior, optimizers and
      schedules, with ONLY the adversarial objective and critic penalty
      swapped: ``trainer.loss`` becomes the linear BOLT loss on
      sigmoid(raw score) and ``trainer.penalty`` becomes the WGAN-GP
      penalty on the raw score. lambda_GP was gauntlet-tuned from the
      paper's 10 (which collapses here, see log) to 1.0.
      The particle update is otherwise unchanged.

Protocol (explicit): a generated sample counts toward a mode if it lies
within 3 sigma (0.15) of the mode's center; modes recovered = number of
the 25 centers with at least one counting sample among 10,000 clean
samples (the paper's own counting rule, Appendix G.4). Also reported:
Naeem coverage (fraction of real samples whose 5-NN ball within the real
set contains a generated sample), as in the cycle-1 pairing test.

Gauntlet log (pop-os, RTX A6000; controls measured in round 1 at the
original 3000-step budget, BOLT arm refined one knob per round):

  R1  lambda_GP=10 (paper default), 3000 steps:
      particle 25.0 modes / 0.101 cov; plain 25.0 / 0.493;
      bolt 0.0 / 0.000 -- total collapse. Diagnosis: the GP term
      (~0.3-1.2) swamps the bounded BOLT loss (h(real) ~= h(fake) ~= 0.5
      throughout; the critic never separates), and the generator
      collapses to a point (fake std -> 0.005). A frozen-prior BOLT
      variant collapses the same way, so this is a loss/penalty balance
      failure, not a particle-prior interaction. The unconstrained
      Max-BER check (lambda_GP=0) diverges as the paper predicts: raw
      scores explode to ~19, sigmoid saturates at 1.0 for real and fake
      alike, 5 modes, max|fake| 14.2 at 3000 steps.
  R2  lambda_GP=1, 3000 steps: bolt 23.0 modes (22/22/25) / 0.044 cov.
  R3  lambda_GP=0.1, 3000 steps: bolt 15.3 modes (22/0/24) -- seed 1
      diverges outright (max|fake| 36.3); under-constrained.
  R4  lambda_GP=3, 3000 steps: bolt 10.3 modes (0/21/10) -- GP dominance
      and instability return.
  R5  lambda_GP=1, 6000 steps (budget doubled): bolt 25.0 modes
      (25/25/25) / 0.048 cov.
  R6  lambda_GP=2, 6000 steps: bolt 12.0 modes (20/16/0) -- worse;
      lambda_GP=1 is the stability sweet spot in this harness.

GPU verdict (historical): PARTIAL transfer. At the tuned lambda_GP=1 and double the
original budget, BOLT-GP matched both controls on the GPU on modes recovered
(25.0 vs 25.0 vs 25.0, means over 3 seeds) -- but the paper's headline
margin cannot appear here, because the plain NS-GAN control does NOT
collapse in this harness (25.0 modes, where the paper's toy setup gives
~6): the repo's ka2 machinery at this budget saturates the modes
metric. On Naeem coverage BOLT-GP loses clearly to both controls
(0.048 vs particle 0.128, plain 0.507, all at the final 6000-step budget): its samples are the most
dispersed of the three arms, consistent with the bounded generator
loss -E[h(fake)] giving vanishing concentration pressure once h is
moderate. At the paper's own lambda_GP in {5, 10} the transfer fails
completely in this harness (R1/R4), and near-zero lambda reproduces
the paper's Max-BER divergence (R3, lambda_GP=0 check). The assertions
below encode the CPU cohort; see the next paragraph.

CPU cohort (current evidence; CPU, one intra-op thread, seeds 0/1/2,
6000 steps, lambda_GP=1): modes particle 24.33 (25/23/25), plain 25.0,
bolt 21.0 (22/24/17); coverage particle 0.088, plain 0.461, bolt 0.046.
The GPU tie on modes does NOT reproduce on CPU: the two "BOLT matches
control" claims are recorded as strict xfails (faithful FAIL), and the
coverage loss is confirmed. CPU verdict: FAIL to transfer on modes and
coverage at the tuned setting.
Diagnostic (report-only, not a gate): the plain control's 25/25 modes and
high coverage come from a diffuse cloud over the grid square -- only 31%
of its samples lie within 3 sigma of a centre (particle 94%, bolt 91%).

Scope (declared). Arm (c) is an ASSISTED/DIAGNOSTIC cohort
(``assisted-host-swap:bolt-loss+raw-score-gp``), not recipe evidence: the
test host replaces ``trainer.loss`` and ``trainer.penalty`` by attribute
assignment after the recipe resolved, because the recipe API declares no
BOLT loss or raw-score GP capability. It cannot fill a recipe's
qualification or matched-comparison cells. Tuning budgets are unequal:
lambda_GP was tuned over 6 GPU rounds and the budget doubled once for the
BOLT arm; the controls were not tuned.

The training assertions are slow and opt-in; the mechanism checks always
run. Report (summary.json + actual-training GIF): reports/toy_audit/paper-watch/bolt-critic/.

Run: PARTICLEGAN_RUN_PAPERWATCH=1 python -m pytest tests/test_paperwatch_bolt_critic.py
Regenerate report: PYTHONPATH=.:tests python tests/test_paperwatch_bolt_critic.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from particlegan import GANTrainer, get_recipe, init
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator


# --------------------------------------------------------------------------
# The paper's mechanism: linear BOLT loss + GP on the raw score.
# --------------------------------------------------------------------------

class BOLTLoss:
    """BOLT adversarial objective on raw critic logits (arXiv:2510.25609).

    The bounded score is h = sigmoid(raw). The critic minimizes the
    empirical BOLT risk E[h(fake)] - E[h(real)] (balanced priors); the
    generator minimizes -E[h(fake)], i.e. maximizes the surrogate BER.
    Interface matches particlegan.gan_loss.GANLoss (d_loss / g_loss on
    raw scores), so GANTrainer drives it unchanged.
    """

    def d_loss(self, real_logits, fake_logits):
        return torch.sigmoid(fake_logits).mean() - torch.sigmoid(real_logits).mean()

    def g_loss(self, fake_logits, real_logits=None):
        return -torch.sigmoid(fake_logits).mean()


class BOLTGradientPenalty:
    """WGAN-GP penalty on the RAW critic score, as in the paper (App. H).

    penalty = lambda_GP * E[(||grad_x f(x_hat)||_2 - 1)^2] over
    real/fake interpolants x_hat. Wraps the trainer's KA2 penalty so the
    shared step record (``regularizer.record``) stays the object the
    trainer's policy was built with; only the penalty VALUE is replaced.
    lambda_gp = 0 is the paper's unconstrained Max-BER ablation.
    """

    def __init__(self, wrapped, lambda_gp=10.0):
        self.regularizer = wrapped.regularizer
        self.collect_stats = False
        self.last_stats = {}
        self.lambda_gp = float(lambda_gp)

    def __call__(self, critic, x_real, x_fake):
        if self.lambda_gp == 0.0:
            return x_real.new_zeros(())
        eps = torch.rand(x_real.shape[0], 1, device=x_real.device, dtype=x_real.dtype)
        x_hat = (eps * x_real + (1.0 - eps) * x_fake).detach().requires_grad_(True)
        scores = critic(x_hat)
        grads = torch.autograd.grad(scores.sum(), x_hat, create_graph=True)[0]
        return self.lambda_gp * (grads.norm(2, dim=1) - 1.0).pow(2).mean()


# --------------------------------------------------------------------------
# Target and metrics: the paper's 25-Gaussians (Appendix G.4).
# --------------------------------------------------------------------------

N_SIDE = 5
N_MODES = N_SIDE * N_SIDE
SPACING = 1.0
SIGMA = 0.05
MODE_RADIUS = 3.0 * SIGMA  # 0.15, the paper's counting radius


def _grid_means(device="cpu"):
    coords = (torch.arange(N_SIDE, dtype=torch.float32, device=device) - (N_SIDE - 1) / 2.0) * SPACING
    cx, cy = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([cx.flatten(), cy.flatten()], dim=1)


MEANS = _grid_means()


def sample_mix(n, gen, device="cpu"):
    means = MEANS.to(device)
    idx = torch.randint(0, N_MODES, (n,), generator=gen, device=device)
    return means[idx] + SIGMA * torch.randn(n, 2, generator=gen, device=device)


def modes_recovered(fake, means=None):
    """Paper's rule: # centers with >= 1 fake within 3 sigma (0.15)."""
    if means is None:
        means = MEANS.to(fake.device)
    dists = torch.cdist(fake, means)
    return int((dists.min(dim=0).values <= MODE_RADIUS).sum().item())


def in_mode_fraction(fake, means=None):
    """Diagnostic precision: share of samples within 3 sigma of any centre.

    Report-only (not a declared gate). Modes recovered and coverage both
    reward a diffuse cloud; this exposes it.
    """
    if means is None:
        means = MEANS.to(fake.device)
    return (torch.cdist(fake, means).min(dim=1).values <= MODE_RADIUS).float().mean().item()


def naeem_coverage(real, fake, k=5):
    """Fraction of real points whose kNN ball (within real) contains a fake."""
    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    radius = d_rr.topk(k, largest=False).values[:, -1]
    d_rf = torch.cdist(real, fake)
    return (d_rf.min(dim=1).values <= radius).float().mean().item()


# --------------------------------------------------------------------------
# Arms. One protocol for every arm; only the prior / objective changes.
# --------------------------------------------------------------------------

STEPS = 6000
BATCH = 256
NUM_PARTICLES = 256
Z_DIM = 2
LR = 2e-3
LAMBDA_GP = 1.0
SEEDS = (0, 1, 2)
ARMS = ("particle", "plain", "bolt")
# Pinned to CPU with one intra-op thread so the recorded numbers form one
# reproducible cohort. The gauntlet in the docstring ran on an RTX A6000;
# those GPU figures remain the historical tuning log, not this cohort.
DEVICE = "cpu"
SNAPSHOTS = 12          # evenly spaced post-update observations for the GIF
SNAPSHOT_SAMPLES = 1024
RUN_ENV = "PARTICLEGAN_RUN_PAPERWATCH"
REPORT_DIR = (Path(__file__).resolve().parents[1] / "reports" / "toy_audit" / "paper-watch"
              / "bolt-critic")

# The BOLT arm is an ASSISTED/DIAGNOSTIC cohort: the host swaps
# ``trainer.loss`` and ``trainer.penalty`` by attribute assignment after the
# recipe resolved, because the recipe API has no declared BOLT-loss or
# raw-score-GP capability. Its results are not recipe evidence and cannot
# fill a recipe's qualification or matched-comparison cells.
BOLT_COHORT = "assisted-host-swap:bolt-loss+raw-score-gp"
TUNING_BUDGET = ("BOLT arm: lambda_GP tuned over 6 GPU gauntlet rounds "
                 "(10, 1, 0.1, 3 at 3000 steps; 1, 2 at 6000 steps) plus a "
                 "lambda_GP=0 Max-BER check, and the step budget doubled once; "
                 "controls: 0 tuning rounds")



def slow(test):
    """Multi-minute CPU training evidence: opt-in, never run by default CI."""
    test = pytest.mark.skipif(os.environ.get(RUN_ENV) != "1",
                              reason=f"9 x {STEPS}-step CPU trainings; opt in with {RUN_ENV}=1")(test)
    return pytest.mark.slow(test)


def run_arm(arm, seed, *, steps=STEPS, batch=BATCH, lr=LR, lambda_gp=LAMBDA_GP,
            device=DEVICE, snapshots=SNAPSHOTS):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_arm(arm, seed, steps=steps, batch=batch, lr=lr,
                        lambda_gp=lambda_gp, device=device, snapshots=snapshots)
    finally:
        torch.set_num_threads(threads)


def _run_arm(arm, seed, *, steps, batch, lr, lambda_gp, device, snapshots):
    torch.manual_seed(seed)
    overrides = dict(z_dim=Z_DIM, num_particles=NUM_PARTICLES, batch_size=batch,
                     total_steps=steps, lr=lr)
    if arm == "plain":
        overrides.update(loss="non_saturating", prior_update="frozen",
                         prior_code_law="fresh_gaussian")
    elif arm == "bolt":
        # Placeholder objective so the recipe validates; the trainer's
        # loss/penalty are swapped for the paper's immediately below.
        overrides.update(loss="wasserstein")
    recipe = get_recipe(trainer="ka2", **overrides)
    prior = recipe.make_prior()
    G = SimpleMLPGenerator(z_dim=Z_DIM, hidden_dim=128, n_hidden=3).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=2).to(device)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    if arm != "plain":
        init.deterministic_orthogonal_(prior)
    prior = prior.to(device)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed)
    if arm == "bolt":
        trainer.loss = BOLTLoss()
        trainer.penalty = BOLTGradientPenalty(trainer.penalty, lambda_gp=lambda_gp)
    data_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    every = max(1, steps // snapshots) if snapshots else 0
    frames = []
    finite = True
    for step in range(1, steps + 1):
        out = trainer.step(sample_mix(batch, data_gen, device=device))
        if not (torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])):
            finite = False
            break
        if every and (step % every == 0 or step == steps):
            # A fixed generator so every frame shows the same latent draws.
            snap_gen = torch.Generator(device=device).manual_seed(seed + 555)
            pts = trainer.sample(SNAPSHOT_SAMPLES, generator=snap_gen,
                                 output_noise=False).cpu()
            frames.append({"step": step, "points": pts, "modes": modes_recovered(pts)})
    eval_gen = torch.Generator(device=device).manual_seed(seed + 999)
    fake = trainer.sample(10_000, generator=eval_gen, output_noise=False).cpu()
    fake_eval = fake[:4096]
    real_eval = sample_mix(4096, torch.Generator().manual_seed(777))
    return {"arm": arm, "seed": seed, "steps": steps, "lambda_gp": lambda_gp,
            "device": device, "finite": finite,
            "cohort": BOLT_COHORT if arm == "bolt" else "recipe:ka2",
            "modes": modes_recovered(fake),
            "coverage": naeem_coverage(real_eval, fake_eval),
            "precision": in_mode_fraction(fake),
            "max_abs": float(fake.abs().max()) if len(fake) else float("nan"),
            "frames": frames}


def run_all(processes=None):
    jobs = [(arm, seed) for arm in ARMS for seed in SEEDS]
    if processes and processes > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes) as pool:
            rows = pool.starmap(run_arm, jobs)
    else:
        rows = [run_arm(arm, seed) for arm, seed in jobs]
    out = {arm: [] for arm in ARMS}
    for row in rows:
        out[row["arm"]].append(row)
    return out


def summary(results):
    return {
        "question": "BOLT-GP critic (arXiv:2510.25609) transferred to ParticleGAN on 25-Gaussians",
        "device": DEVICE, "torch_threads": 1, "steps": STEPS, "batch": BATCH,
        "seeds": list(SEEDS), "lambda_gp": LAMBDA_GP,
        "bolt_cohort": BOLT_COHORT, "tuning_budget": TUNING_BUDGET,
        "arms": {arm: {"cohort": rows[0]["cohort"],
                       "modes": [r["modes"] for r in rows],
                       "coverage": [round(r["coverage"], 4) for r in rows],
                       "finite": [r["finite"] for r in rows],
                       "precision_diagnostic": [round(r["precision"], 4) for r in rows],
                       "mean_modes": round(_mean(rows, "modes"), 3),
                       "mean_coverage": round(_mean(rows, "coverage"), 4)}
                 for arm, rows in results.items()},
    }


def render_gif(results, path, seed_index=0):
    """Actual training snapshots (one seed) of every arm over the target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    rows = {arm: results[arm][seed_index] for arm in ARMS}
    n = min(len(r["frames"]) for r in rows.values())
    target = sample_mix(SNAPSHOT_SAMPLES, torch.Generator().manual_seed(4242))
    fig, axes = plt.subplots(1, len(ARMS), figsize=(3.6 * len(ARMS), 4.0), dpi=72)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=3)
    with writer.saving(fig, str(path), dpi=72):
        for i in range(n):
            for ax, arm in zip(axes, ARMS):
                fr = rows[arm]["frames"][i]
                ax.clear()
                ax.scatter(target[:, 0], target[:, 1], s=2, c="0.75")
                ax.scatter(fr["points"][:, 0], fr["points"][:, 1], s=2, c="C3")
                ax.set_xlim(-3.5, 3.5)
                ax.set_ylim(-3.5, 3.5)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{arm}  update {fr['step']}\nmodes {fr['modes']}/25 "
                             f"(goal 25/25)", fontsize=9)
            fig.suptitle(f"BOLT-GP paper watch, seed {SEEDS[seed_index]}, CPU "
                         "(grey target, red samples; bolt = assisted host-swap cohort)",
                         fontsize=8)
            writer.grab_frame()
    plt.close(fig)
    return n


@pytest.fixture(scope="module")
def results():
    # One single-threaded process per (arm, seed): same numbers as a serial
    # run, wall time of the slowest arm.
    return run_all(processes=min(len(ARMS) * len(SEEDS), os.cpu_count() or 1))


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


# --------------------------------------------------------------------------
# Mechanism unit checks (fast): the loss is the paper's loss, the penalty
# is WGAN-GP on the raw score.
# --------------------------------------------------------------------------

def test_bolt_loss_is_linear_in_bounded_score():
    real = torch.tensor([2.0, -1.0])
    fake = torch.tensor([0.5, -2.0])
    loss = BOLTLoss()
    expected_d = torch.sigmoid(fake).mean() - torch.sigmoid(real).mean()
    expected_g = -torch.sigmoid(fake).mean()
    assert torch.allclose(loss.d_loss(real, fake), expected_d)
    assert torch.allclose(loss.g_loss(fake), expected_g)
    # Bounded: the critic loss lives in [-1, 1] whatever the logits do.
    big = torch.tensor([1e4, -1e4])
    assert -1.0 <= loss.d_loss(big, -big).item() <= 1.0


def test_bolt_loss_gradients_push_scores_apart():
    real = torch.tensor([0.0], requires_grad=True)
    fake = torch.tensor([0.0], requires_grad=True)
    BOLTLoss().d_loss(real, fake).backward()
    # d/d real logit of -sigmoid is negative: minimizing raises real scores.
    assert real.grad.item() < 0
    # d/d fake logit of +sigmoid is positive: minimizing lowers fake scores.
    assert fake.grad.item() > 0


def test_bolt_gp_penalizes_raw_score_gradient_norm():
    torch.manual_seed(0)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=32, n_hidden=2, fourier=0)
    init.deterministic_orthogonal_(D, seed=1)

    class _Wrapped:  # minimal stand-in for the KA2 penalty's record holder
        regularizer = type("R", (), {"record": None})()

    pen = BOLTGradientPenalty(_Wrapped(), lambda_gp=10.0)
    real = torch.randn(16, 2)
    fake = torch.randn(16, 2)
    value = pen(D, real, fake)
    assert torch.isfinite(value) and value.item() >= 0
    value.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in D.parameters())
    # lambda_gp = 0 is the unconstrained Max-BER ablation: exactly zero.
    pen0 = BOLTGradientPenalty(_Wrapped(), lambda_gp=0.0)
    assert pen0(D, real, fake).item() == 0.0


def test_modes_recovered_counts_paper_rule():
    # One sample exactly on each center recovers all 25 modes...
    assert modes_recovered(MEANS.clone()) == 25
    # ...samples 0.2 away from every center (beyond 3 sigma) recover none.
    assert modes_recovered(MEANS.clone() + 0.2) == 0
    # Collapse onto a single mode recovers exactly 1.
    assert modes_recovered(MEANS[0].repeat(500, 1)) == 1


# --------------------------------------------------------------------------
# The gauntlet result, encoded against the pinned CPU cohort
# (reports/toy_audit/paper-watch/bolt-critic/summary.json). Measured (seeds 0/1/2):
# modes particle 25/23/25 (24.33), plain 25/25/25 (25.0), bolt 22/24/17
# (21.0); coverage particle 0.088, plain 0.461, bolt 0.046. The GPU gauntlet
# claimed a 25.0/25.0/25.0 tie; on CPU the BOLT arm does not reproduce it.
# Those claims are kept as strict xfails with the measured values rather
# than loosened thresholds.
# --------------------------------------------------------------------------

CPU_FAIL = ("faithful CPU FAIL: bolt mean modes 21.0 (22/24/17) vs particle "
            "24.33 and plain 25.0; the GPU tie at 25.0 does not reproduce")


@slow
def test_all_arms_stay_finite(results):
    for arm, rows in results.items():
        assert all(row["finite"] for row in rows), f"{arm} diverged to non-finite losses"


@slow
@pytest.mark.xfail(strict=True, reason=CPU_FAIL)
def test_bolt_matches_or_beats_particle_control(results):
    # GPU gauntlet R5 claim: bolt 25.0 vs particle 25.0 mean modes.
    assert _mean(results["bolt"], "modes") >= _mean(results["particle"], "modes")


@slow
@pytest.mark.xfail(strict=True, reason=CPU_FAIL)
def test_bolt_matches_plain_control_on_modes_at_ceiling(results):
    # GPU claim: BOLT-GP ties the (non-collapsing) plain control at 25.0.
    assert _mean(results["bolt"], "modes") >= _mean(results["plain"], "modes")


@slow
def test_plain_control_does_not_collapse_in_this_harness(results):
    # The paper's plain-GAN collapse (~6/25) does not occur here: the plain
    # ka2 control recovers all 25 modes in every seed, so there is no
    # headroom for the paper's headline margin on this metric.
    assert [row["modes"] for row in results["plain"]] == [N_MODES] * len(SEEDS)


@slow
def test_bolt_coverage_gap_is_the_documented_failure(results):
    # The honest negative half of the verdict: on Naeem coverage BOLT-GP
    # (0.046) loses to the ParticleGAN control (0.088) and the plain
    # control (0.461). This pins the measured gap so a future change that
    # closes it has to update the verdict deliberately.
    assert _mean(results["bolt"], "coverage") < _mean(results["particle"], "coverage")
    assert _mean(results["bolt"], "coverage") < _mean(results["plain"], "coverage")


@slow
def test_training_gif_shows_actual_observations(results, tmp_path):
    path = tmp_path / "training.gif"
    frames = render_gif(results, path)
    assert frames == SNAPSHOTS and path.stat().st_size > 0
    for rows in results.values():
        assert rows[0]["frames"][-1]["step"] == STEPS


def main(argv=None):
    """Regenerate the compact report: summary.json + training.gif (CPU)."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument("--processes", type=int, default=len(ARMS) * len(SEEDS))
    args = parser.parse_args(argv)
    res = run_all(processes=args.processes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary(res), indent=2) + "\n")
    render_gif(res, args.out / "training.gif")
    print(json.dumps(summary(res)["arms"], indent=1))


if __name__ == "__main__":
    main()
