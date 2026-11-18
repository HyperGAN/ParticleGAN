"""Paper-watch study: is the Gini distance correlation a good *evaluation*
metric for GAN runs?

Source paper: "EmbeddGAN" (arXiv:2609.22508). The paper's framework —
replace the discriminator with an embedding network trained against the
Gini distance correlation rho_g — is a replacement for the ParticleGAN
formulation and out of scope. The salvage tested here is rho_g itself,
computed on raw samples, as a distribution-level evaluation metric.

Claim under test: across short toy runs in a covered regime and a
deliberately mode-dropped regime, rho_g separates collapsed from covered
runs at least as well as the runs' own critic loss does, and rho_g tracks
true (Naeem) coverage across checkpoints at least as well.

Ground truth is by construction, never by the metric: the collapsed arm
is trained on a 2-of-8-mode subset of the target mixture and evaluated
against the full mixture. The critic in that arm equilibrates on its own
training distribution, which is exactly the blindness being measured.

Run directly (`python tests/test_gini_distance_metric.py`) to print the
full study table without the assertions.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.toy_metrics import gini_distance_correlation, naeem_coverage  # noqa: E402
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator

# =========================
# Toy problem: 8 Gaussians on a ring
# =========================

N_MODES = 8
MODE_STD = 0.25
COLLAPSED_MODES = (0, 1)
Z_DIM = 4
STEPS = 1600
BATCH = 256
CHECKPOINT_EVERY = 200
SEEDS = (0, 1, 2)


def _centers(device="cpu"):
    ang = torch.arange(N_MODES, dtype=torch.float64) * (2 * math.pi / N_MODES)
    return torch.stack([2.0 * torch.cos(ang), 2.0 * torch.sin(ang)], dim=1).to(device)


def sample_mog(n, modes, gen):
    centers = _centers()
    pick = torch.randint(0, len(modes), (n,), generator=gen)
    base = centers[list(modes)][pick]
    return (base + MODE_STD * torch.randn(n, 2, generator=gen)).float()


# =========================
# One training run
# =========================

def train_run(seed, modes):
    """Train a plain MLP GAN on `modes`; log metrics at each checkpoint.

    Returns a list of per-checkpoint dicts with rho_g, the run's own
    critic loss (mean over the window since the previous checkpoint),
    the same critic evaluated against the full mixture, and Naeem
    coverage of the generated samples against the full mixture.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    eval_gen = torch.Generator().manual_seed(999)
    real_eval = sample_mog(4096, tuple(range(N_MODES)), eval_gen)
    z_eval = torch.randn(4096, Z_DIM, generator=torch.Generator().manual_seed(998))

    g = SimpleMLPGenerator(z_dim=Z_DIM)
    d = SimpleMLPDiscriminator(fourier=2)
    opt_g = torch.optim.Adam(g.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.999))

    rows = []
    window_d = []
    for step in range(1, STEPS + 1):
        real = sample_mog(BATCH, modes, gen)
        z = torch.randn(BATCH, Z_DIM, generator=gen)
        fake = g(z)
        d_loss = F.softplus(-d(real)).mean() + F.softplus(d(fake.detach())).mean()
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        window_d.append(float(d_loss.detach()))

        z = torch.randn(BATCH, Z_DIM, generator=gen)
        g_loss = F.softplus(-d(g(z))).mean()
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        if step % CHECKPOINT_EVERY == 0:
            with torch.no_grad():
                fake_eval = g(z_eval)
                full_real = sample_mog(1024, tuple(range(N_MODES)), gen)
                critic_full = float(
                    F.softplus(-d(full_real)).mean() + F.softplus(d(fake_eval[:1024])).mean()
                )
            rows.append(
                {
                    "step": step,
                    "rho_g": gini_distance_correlation(
                        real_eval,
                        fake_eval,
                        max_n=512,
                        generator=torch.Generator().manual_seed(1234),
                    ),
                    "critic_loss": sum(window_d) / len(window_d),
                    "critic_loss_full": critic_full,
                    "coverage": naeem_coverage(real_eval[:2048], fake_eval[:2048], k=5),
                }
            )
            window_d = []
    return rows


def run_study(seeds=SEEDS):
    return {
        "covered": [train_run(s, tuple(range(N_MODES))) for s in seeds],
        "collapsed": [train_run(s, COLLAPSED_MODES) for s in seeds],
    }


# =========================
# Statistics helpers
# =========================

def _cohens_d(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    pooled = math.sqrt((float(a.var(unbiased=True)) + float(b.var(unbiased=True))) / 2)
    if pooled == 0:
        return float("inf") if float(a.mean()) != float(b.mean()) else 0.0
    return float(abs(a.mean() - b.mean()) / pooled)


def study_statistics(study):
    """Final-checkpoint separation and pooled checkpoint correlations."""
    final = {
        regime: [{k: run[-1][k] for k in ("rho_g", "critic_loss", "critic_loss_full", "coverage")}
                 for run in runs]
        for regime, runs in study.items()
    }
    sep = {
        key: _cohens_d(
            [r[key] for r in final["collapsed"]], [r[key] for r in final["covered"]]
        )
        for key in ("rho_g", "critic_loss", "critic_loss_full")
    }
    pooled = [row for runs in study.values() for run in runs for row in run]
    corr = {
        key: float(spearmanr([r[key] for r in pooled], [r["coverage"] for r in pooled]).statistic)
        for key in ("rho_g", "critic_loss", "critic_loss_full")
    }
    return {"final": final, "separation": sep, "correlation": corr}


# =========================
# Metric unit checks
# =========================

def test_gini_distance_correlation_basics():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(400, 2, generator=gen)
    b = torch.randn(400, 2, generator=gen)
    same = gini_distance_correlation(a, b, generator=torch.Generator().manual_seed(1))
    assert abs(same) < 0.02  # two draws from one distribution: ~0
    far = gini_distance_correlation(
        a, b + torch.tensor([6.0, 0.0]), generator=torch.Generator().manual_seed(1)
    )
    assert far > 0.5  # disjoint supports: large
    dropped = gini_distance_correlation(
        a, a[a[:, 0] > 0], generator=torch.Generator().manual_seed(1)
    )
    assert dropped > same + 0.05  # half the mass removed: clearly positive


def test_naeem_coverage_basics():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(400, 2, generator=gen)
    b = torch.randn(400, 2, generator=gen)
    assert naeem_coverage(a, b) > 0.9
    assert naeem_coverage(a, b + torch.tensor([6.0, 0.0])) < 0.05


# =========================
# The study
# =========================

@pytest.fixture(scope="module")
def study():
    return run_study()


@pytest.fixture(scope="module")
def stats(study):
    return study_statistics(study)


def test_construction_ground_truth(stats):
    cov = {r: [x["coverage"] for x in rows] for r, rows in stats["final"].items()}
    mean = {r: sum(v) / len(v) for r, v in cov.items()}
    # The collapsed arm really is collapsed, the covered arm really covers.
    assert mean["covered"] - mean["collapsed"] > 0.3


def test_rho_g_separates_at_least_as_well_as_critic_loss(stats):
    sep = stats["separation"]
    assert sep["rho_g"] >= sep["critic_loss"]
    assert sep["rho_g"] > 2.0  # and the separation is large in absolute terms


def test_rho_g_tracks_coverage_at_least_as_well_as_critic_loss(stats):
    corr = stats["correlation"]
    assert corr["rho_g"] < -0.7  # rho_g falls as true coverage rises
    assert abs(corr["rho_g"]) >= abs(corr["critic_loss"])


if __name__ == "__main__":
    result = run_study()
    stats = study_statistics(result)
    for regime, runs in result.items():
        for seed, run in zip(SEEDS, runs):
            for row in run:
                print(regime, seed, row)
    print("separation (Cohen's d, final checkpoint):", stats["separation"])
    print("correlation with coverage (Spearman, pooled):", stats["correlation"])
