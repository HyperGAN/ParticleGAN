"""Paper-watch study: CADMA capacity-aware coverage decomposition.

Source paper: "CADMA: Capacity-Aware Coverage Decomposition for
Generative Model Assessment" (Meng et al., arXiv:2611.12243, Nov 2026;
Sections 3 and 4.1 read before building).

CADMA replaces the single geometry-based coverage number with a
maximum-cardinality bipartite matching between real and generated
samples, restricted to real-induced neighbourhoods with radii
eps_i = min(rho_i, mean(rho)) (Eq. 10). With N_mat the matching size:

    C_cap      = N_mat / N                          (Eq. 7)
    D_drop     = Coverage - C_cap                   (Eq. 8)
    D_collapse = (M_cov - N_mat) / N                (Eq. 9)

where Coverage is Naeem coverage on the same eps radii and M_cov is
the number of generated samples inside at least one real neighbourhood.

Claims under test (the paper's, at toy scale):
  (i)  C_cap tracks a known reference coverage trend with lower mean
       absolute error than Naeem coverage.
  (ii) The decomposition separates the two failure modes a single
       coverage number cannot: D_collapse rises under intramode
       concentration while D_drop stays low, and D_drop rises under
       outlier-driven loss of support while D_collapse stays low.

Ground truth is by construction, never by the metric:
  * density shift -- 7-mode Gaussian mixture; generated mass is moved
    progressively from 6 modes into the remaining 1; the reference
    effective coverage decreases linearly 1 -> 1/7 (paper Sec. 4.1.1).
  * intramode concentration -- the generated set keeps the full
    geometric footprint of a fresh draw but only a fraction f of its
    points are distinct (the rest are exact duplicates); the reference
    distinct-capacity fraction is f itself. Geometry-based coverage
    should stay deceptively high while capacity falls.
  * outlier pollution -- a growing fraction p of generated samples is
    replaced by far outliers; the reference decreases linearly 1 - p.

Controls are the incumbent metrics: Naeem coverage with raw (uncapped)
kNN radii and the empirical Gini distance correlation rho_g from the
cycle-1 study (branch paper-watch/gini-distance-metric, PR #559).
Master does not carry those two functions, so equivalent local
implementations live in this module; CADMA itself is in
lib/toy_metrics.py. rho_g is a discrepancy (0 = identical), not a
coverage estimate; where an MAE is reported for it, it is for the
proxy 1 - rho_g, and the table says so.

k choice: the paper uses k=70 at N=25000 (k/N = 0.0028). This study
uses N=1400 and k=5 (k/N = 0.0036), which also matches the incumbent
Naeem control's k=5 so the comparison is on equal neighbourhood
orders. A sensitivity grid over k in {3, 5, 10, 20} was run during
development (see the PR description for the full log): density and
outlier MAE fall monotonically in k (0.060/0.091 at k=5 -> 0.007/0.011
at k=20), but concentration detection degrades monotonically (the
D_collapse rise from f=1 to f=0.2 shrinks 0.30 -> 0.22 -> 0.14 across
k = 5, 10, 20, because each distinct prototype can support ~k
matches). k=5 is the smallest k at which C_cap beats both coverage
variants on all three constructions, and the one where the
decomposition is most informative; the in-test sensitivity check
covers k in {3, 10} on the density and concentration arms.

The final section applies CADMA to real toy-GAN runs (cycle-1 style:
8 Gaussians on a ring, plain MLP GAN, short budget): a covered run
trained on all 8 modes and a collapsed run trained on 2 of 8 modes,
evaluated against the full mixture. The collapsed run's deficit must
read as collapse (high D_collapse, near-zero D_drop: its two covered
modes are over-populated and the other six are fully uncovered, which
by the paper's definition lowers C_cap rather than D_drop).

Run directly (`python tests/test_cadma_metrics.py`) to print the full
study tables without the assertions. Toy-GAN training runs on CPU
here; the development runs used the pop-os GPU box and agreed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.toy_metrics import cadma_scores  # noqa: E402
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator  # noqa: E402

# =========================
# Controls (local equivalents of the cycle-1 metrics)
# =========================

def naeem_coverage(real, fake, k=5):
    """Naeem et al. (2020) coverage with raw kNN radii (no Eq. 10 cap)."""
    a = torch.as_tensor(real, dtype=torch.float64)
    b = torch.as_tensor(fake, dtype=torch.float64)
    d_rr = torch.cdist(a, a)
    radius = d_rr.kthvalue(k + 1, dim=1).values  # +1 skips self-distance
    nearest_fake = torch.cdist(a, b).min(dim=1).values
    return float((nearest_fake <= radius).to(torch.float64).mean())


def gini_distance_correlation(real, fake, max_n=512, seed=7):
    """Empirical Gini distance correlation rho_g (cycle-1 control).

    Zero iff the two distributions agree; grows with discrepancy.
    Both sets are subsampled to at most `max_n` points.
    """
    a = torch.as_tensor(real, dtype=torch.float64)
    b = torch.as_tensor(fake, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    m = int(min(a.shape[0], b.shape[0], max_n))

    def _sub(x):
        if x.shape[0] == m:
            return x
        return x[torch.randperm(x.shape[0], generator=gen)[:m]]

    a, b = _sub(a), _sub(b)
    d_ab = torch.cdist(a, b)
    d_aa = torch.cdist(a, a)
    d_bb = torch.cdist(b, b)
    off = ~torch.eye(m, dtype=torch.bool)
    gcov = d_ab.mean() - 0.5 * (d_aa[off].mean() + d_bb[off].mean())
    x = torch.cat([a, b], dim=0)
    return float(gcov / torch.cdist(x, x).mean())


# =========================
# Constructions: 7-mode Gaussian mixture
# =========================

N_MODES = 7
MODE_RADIUS = 3.0
MODE_STD = 0.35
N_SAMPLES = 1400
K_MAIN = 5
SEEDS = (0, 1, 2)

_CENTERS = np.stack(
    [
        MODE_RADIUS * np.cos(np.arange(N_MODES) * 2 * np.pi / N_MODES),
        MODE_RADIUS * np.sin(np.arange(N_MODES) * 2 * np.pi / N_MODES),
    ],
    axis=1,
)


def _sample_counts(rng, counts):
    parts = [
        _CENTERS[i] + MODE_STD * rng.standard_normal((c, 2))
        for i, c in enumerate(counts)
        if c > 0
    ]
    x = np.concatenate(parts)
    rng.shuffle(x)
    return x


def _real_set(rng):
    return _sample_counts(rng, [N_SAMPLES // N_MODES] * N_MODES)


def _density_shift(rng, t):
    per = N_SAMPLES // N_MODES
    counts = [per - int(round(t * per))] * 6 + [per + int(round(t * 6 * per))]
    counts[6] += N_SAMPLES - sum(counts)
    return _sample_counts(rng, counts)


def _concentration(rng, f):
    full = _real_set(rng)
    n_proto = max(1, int(round(f * N_SAMPLES)))
    proto = full[:n_proto]
    out = proto[np.resize(np.arange(n_proto), N_SAMPLES)].copy()
    rng.shuffle(out)
    return out


def _outlier(rng, p):
    base = _real_set(rng)
    n_out = int(round(p * N_SAMPLES))
    if n_out:
        base[:n_out] = np.array([14.0, 14.0]) + MODE_STD * rng.standard_normal(
            (n_out, 2)
        )
        rng.shuffle(base)
    return base


CONSTRUCTIONS = {
    "density": (
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lambda t: 1.0 - t * 6.0 / 7.0,
        _density_shift,
    ),
    "concentration": ([1.0, 0.8, 0.6, 0.4, 0.2], lambda f: f, _concentration),
    "outlier": ([0.0, 0.15, 0.3, 0.45, 0.6], lambda p: 1.0 - p, _outlier),
}


def _eval_pair(real, fake, k):
    cs = cadma_scores(real, fake, k=k)
    return {
        **cs,
        "cov_raw": naeem_coverage(real, fake, k=k),
        "rho_g": gini_distance_correlation(real, fake),
    }


def run_constructions(k=K_MAIN, seeds=SEEDS, with_rho_g=True):
    """Seed-mean metric values per stage for every construction."""
    out = {}
    for name, (stages, ref_fn, gen_fn) in CONSTRUCTIONS.items():
        rows = []
        for v in stages:
            acc = []
            for seed in seeds:
                real = _real_set(np.random.default_rng(1000 + seed))
                fake = gen_fn(np.random.default_rng(5000 + seed), v)
                row = _eval_pair(real, fake, k) if with_rho_g else {
                    **cadma_scores(real, fake, k=k),
                    "cov_raw": naeem_coverage(real, fake, k=k),
                }
                acc.append(row)
            rows.append(
                {
                    "stage": v,
                    "ref": ref_fn(v),
                    **{
                        key: float(np.mean([a[key] for a in acc]))
                        for key in acc[0]
                    },
                }
            )
        out[name] = rows
    return out


def mae(rows, key):
    return float(np.mean([abs(r[key] - r["ref"]) for r in rows]))


def mae_rho_g_proxy(rows):
    return float(
        np.mean([abs((1.0 - min(max(r["rho_g"], -1.0), 1.0)) - r["ref"]) for r in rows])
    )


# =========================
# Metric unit checks
# =========================

def test_cadma_identity_and_decomposition():
    rng = np.random.default_rng(0)
    real = rng.standard_normal((300, 2))
    same = cadma_scores(real, real.copy(), k=5)
    # Identity matching: every real sample matches its own copy.
    assert same["c_cap"] == 1.0
    assert same["d_drop"] == 0.0
    assert same["d_collapse"] == 0.0
    assert same["coverage"] == 1.0

    fake = rng.standard_normal((300, 2))
    cs = cadma_scores(real, fake, k=5)
    # Eq. 8 decomposition is exact by construction.
    assert abs(cs["coverage"] - (cs["c_cap"] + cs["d_drop"])) < 1e-12
    assert 0.0 <= cs["c_cap"] <= cs["coverage"] <= 1.0
    assert cs["d_collapse"] >= 0.0


def test_cadma_disjoint_sets():
    rng = np.random.default_rng(1)
    real = rng.standard_normal((300, 2))
    fake = rng.standard_normal((300, 2)) + np.array([30.0, 0.0])
    cs = cadma_scores(real, fake, k=5)
    assert cs["c_cap"] < 0.02
    assert cs["coverage"] < 0.02


# =========================
# The construction study
# =========================

@pytest.fixture(scope="module")
def study():
    return run_constructions()


def test_c_cap_tracks_reference_better_than_naeem(study):
    for name, rows in study.items():
        assert mae(rows, "c_cap") < mae(rows, "cov_raw"), name
        # ... and better than coverage on CADMA's own regularized radii.
        assert mae(rows, "c_cap") < mae(rows, "coverage"), name
    # Absolute tracking on the two linear-reference arms.
    assert mae(study["density"], "c_cap") < 0.10
    assert mae(study["outlier"], "c_cap") < 0.12


def test_density_shift_diagnostics(study):
    rows = study["density"]
    first, mid, last = rows[0], rows[3], rows[-1]
    # Mass concentrating into one mode reads as collapse...
    assert last["d_collapse"] > 0.55
    assert last["d_collapse"] > first["d_collapse"] + 0.4
    # ...while D_drop is non-monotone: it peaks mid-shift and falls at
    # full shift, when the starved modes become fully uncovered rather
    # than locally under-supported (paper, Sec. 4.1.1).
    assert mid["d_drop"] > last["d_drop"] + 0.15


def test_concentration_separation(study):
    rows = study["concentration"]
    first, last = rows[0], rows[-1]
    # Capacity falls and the fall reads as collapse, not dropping.
    assert first["c_cap"] - last["c_cap"] > 0.2
    assert last["d_collapse"] > first["d_collapse"] + 0.2
    assert last["d_drop"] < 0.06
    assert last["d_collapse"] > 3 * last["d_drop"]
    # Geometry-based coverage overestimates effective coverage.
    assert last["cov_raw"] > last["c_cap"]


def test_outlier_separation(study):
    rows = study["outlier"]
    first, last = rows[0], rows[-1]
    # Lost support reads as dropping, not collapse (mirror image of
    # the concentration arm -- the separation claim).
    assert last["d_drop"] > first["d_drop"] + 0.15
    assert last["d_collapse"] < 0.05
    assert last["d_drop"] > 5 * last["d_collapse"]


def test_rho_g_control_blindness(study):
    # rho_g is ~0 for two draws of one distribution...
    for rows in study.values():
        assert abs(rows[0]["rho_g"]) < 0.02
    # ...and stays ~0 under pure duplication: the duplicated set has
    # (nearly) the same pairwise-distance distribution, so this
    # control cannot see the concentration failure at all.
    assert max(abs(r["rho_g"]) for r in study["concentration"]) < 0.02


# =========================
# k sensitivity (subset, no rho_g)
# =========================

@pytest.fixture(scope="module")
def sensitivity():
    return {k: run_constructions(k=k, seeds=(0, 1), with_rho_g=False) for k in (3, 10)}


def test_k_sensitivity_tradeoff(sensitivity, study):
    # Larger k tracks the density reference more closely...
    assert mae(sensitivity[10]["density"], "c_cap") < mae(study["density"], "c_cap")
    assert mae(study["density"], "c_cap") < mae(sensitivity[3]["density"], "c_cap")
    # ...but washes out concentration detection: the D_collapse rise
    # from f=1 to f=0.2 shrinks as k grows.
    def collapse_rise(rows):
        return rows[-1]["d_collapse"] - rows[0]["d_collapse"]

    assert collapse_rise(sensitivity[10]["concentration"]) < collapse_rise(
        study["concentration"]
    )


# =========================
# Real toy-GAN runs: 8 Gaussians on a ring
# =========================

RING_MODES = 8
RING_STD = 0.25
Z_DIM = 4
GAN_STEPS = 1200
GAN_BATCH = 256
GAN_SEEDS = (0, 1, 2)


def _ring_centers():
    ang = torch.arange(RING_MODES, dtype=torch.float64) * (2 * math.pi / RING_MODES)
    return torch.stack([2.0 * torch.cos(ang), 2.0 * torch.sin(ang)], dim=1)


def _sample_ring(n, modes, gen):
    centers = _ring_centers()
    pick = torch.randint(0, len(modes), (n,), generator=gen)
    base = centers[list(modes)][pick]
    return (base + RING_STD * torch.randn(n, 2, generator=gen)).float()


def _train_ring_gan(seed, modes):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    g = SimpleMLPGenerator(z_dim=Z_DIM)
    d = SimpleMLPDiscriminator(fourier=2)
    opt_g = torch.optim.Adam(g.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.999))
    for _ in range(GAN_STEPS):
        real = _sample_ring(GAN_BATCH, modes, gen)
        z = torch.randn(GAN_BATCH, Z_DIM, generator=gen)
        fake = g(z)
        d_loss = F.softplus(-d(real)).mean() + F.softplus(d(fake.detach())).mean()
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        z = torch.randn(GAN_BATCH, Z_DIM, generator=gen)
        g_loss = F.softplus(-d(g(z))).mean()
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
    with torch.no_grad():
        z_eval = torch.randn(
            N_SAMPLES, Z_DIM, generator=torch.Generator().manual_seed(998)
        )
        return g(z_eval).numpy()


@pytest.fixture(scope="module")
def gan_runs():
    real = _sample_ring(
        N_SAMPLES, tuple(range(RING_MODES)), torch.Generator().manual_seed(999)
    ).numpy()
    out = {}
    for tag, modes in (
        ("covered", tuple(range(RING_MODES))),
        ("collapsed", (0, 1)),
    ):
        out[tag] = [
            cadma_scores(real, _train_ring_gan(seed, modes), k=K_MAIN)
            for seed in GAN_SEEDS
        ]
    return out


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def test_gan_collapse_attribution(gan_runs):
    cov, col = gan_runs["covered"], gan_runs["collapsed"]
    # The collapsed run really has lost effective coverage.
    assert _mean(cov, "c_cap") > _mean(col, "c_cap") + 0.15
    # And CADMA attributes it to collapse, not dropping: the collapsed
    # generator over-populates its two modes (D_collapse high) while
    # the six missing modes are fully uncovered, which by definition
    # lowers C_cap instead of raising D_drop.
    assert _mean(col, "d_collapse") > _mean(cov, "d_collapse") + 0.15
    assert _mean(col, "d_drop") < _mean(cov, "d_drop")


if __name__ == "__main__":
    result = run_constructions()
    for name, rows in result.items():
        print(f"== {name} (k={K_MAIN}, N={N_SAMPLES}, seeds={SEEDS}) ==")
        for r in rows:
            print(
                f" stage={r['stage']:>4} ref={r['ref']:.3f} Ccap={r['c_cap']:.3f} "
                f"CovReg={r['coverage']:.3f} CovRaw={r['cov_raw']:.3f} "
                f"Ddrop={r['d_drop']:.3f} Dcoll={r['d_collapse']:.3f} "
                f"rho_g={r['rho_g']:.3f}"
            )
        print(
            f"  MAE C_cap={mae(rows, 'c_cap'):.3f} "
            f"CoverageReg={mae(rows, 'coverage'):.3f} "
            f"NaeemRaw={mae(rows, 'cov_raw'):.3f} "
            f"1-rho_g(proxy)={mae_rho_g_proxy(rows):.3f}"
        )
    real = _sample_ring(
        N_SAMPLES, tuple(range(RING_MODES)), torch.Generator().manual_seed(999)
    ).numpy()
    for tag, modes in (
        ("covered", tuple(range(RING_MODES))),
        ("collapsed", (0, 1)),
    ):
        for seed in GAN_SEEDS:
            cs = cadma_scores(real, _train_ring_gan(seed, modes), k=K_MAIN)
            print(tag, seed, {k: round(v, 3) for k, v in cs.items() if isinstance(v, float)})
