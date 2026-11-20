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

Cohorts (kept separate):
  * metric-diagnostic constructions (no training): the three fixture
    constructions above plus a k-sensitivity grid. These are
    oracle/destructive controls for the scorer, sampled with numpy
    fixture streams; they are not a ParticleGAN training test, and the
    three fixture draws per stage average sampling noise of the
    constructed sets (not training seeds).
  * public-API toy-GAN arms (training): 8 Gaussians on a ring trained
    through ``GANTrainer`` with the unmodified default ``get_recipe()``
    (only ``total_steps`` set), public deterministic initialization and
    protocol seed 0 (``lib/paper_watch_ring``, shared with PR #559). A
    covered arm trains on all 8 modes, a collapsed arm on modes {0, 1};
    both are evaluated against the full mixture with CADMA at every
    checkpoint. The committed receipt and actual-training GIF live in
    reports/paper_watch/cadma-metrics/.

Controls are the incumbent metrics: Naeem coverage with raw (uncapped)
kNN radii and the empirical Gini distance correlation rho_g from the
cycle-1 study (PR #559), imported from lib/toy_metrics.py (this PR
depends on #559). rho_g is a discrepancy (0 = identical), not a
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

Declared toy-GAN gate (terminal window = last 4 checkpoints, means):
  * construction valid: Naeem coverage (raw, k=5) covered minus
    collapsed > 0.3 (else INVALID, not FAIL);
  * C_cap(covered) > C_cap(collapsed) + 0.15;
  * D_collapse(collapsed) > D_collapse(covered) + 0.15;
  * D_drop(collapsed) < D_drop(covered).
The collapsed run's deficit must read as collapse: its two covered
modes are over-populated and the other six are fully uncovered, which
by the paper's definition lowers C_cap rather than raising D_drop.

History: the first revision of this PR trained a hand-written SimpleMLP
softplus loop over seeds {0, 1, 2}; that loop bypassed the ParticleGAN
API and is superseded (its numbers are in the PR history, not this
receipt).

Publish the toy-GAN receipt and GIF (GPU optional), and print the
construction tables:

    python tests/test_cadma_metrics.py --device cuda:0

The full toy-GAN rerun is marked ``slow`` and skipped unless
``PARTICLEGAN_RUN_SLOW=1``.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib import paper_watch_ring as ring  # noqa: E402
from lib.toy_metrics import cadma_scores, gini_distance_correlation, naeem_coverage  # noqa: E402

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
        "rho_g": gini_distance_correlation(
            real, fake, max_n=512, generator=torch.Generator().manual_seed(7)),
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
# Public-API toy-GAN arms: 8 Gaussians on a ring
# =========================

GAN_STEPS = 1200
GAN_CHECK_EVERY = 100
TERMINAL_WINDOW = 4
GAN_KEYS = ("c_cap", "coverage", "d_drop", "d_collapse", "cov_raw")
REPORT_DIR = _REPO_ROOT / "reports" / "paper_watch" / "cadma-metrics"
RECEIPT = REPORT_DIR / "receipt.json"
GIF = REPORT_DIR / "goal.gif"
ARMS = {"covered": ring.ALL_MODES, "collapsed": ring.COLLAPSED_MODES}


def _gan_observe(real_eval):
    real = real_eval.numpy()

    def observe(obs):
        fake = obs["fake"].numpy()
        cs = cadma_scores(real, fake, k=K_MAIN)
        return {**{k: cs[k] for k in ("c_cap", "coverage", "d_drop", "d_collapse")},
                "cov_raw": naeem_coverage(real, fake, k=K_MAIN)}
    return observe


def run_gan_arms(steps=GAN_STEPS, check_every=GAN_CHECK_EVERY, device="cpu", progress=None):
    real_eval = ring.eval_real(N_SAMPLES)
    return {
        arm: ring.train_ring_arm(
            modes, steps, check_every, device=device, n_eval=N_SAMPLES,
            observe=_gan_observe(real_eval),
            progress=(lambda row, arm=arm: progress(arm, row)) if progress else None)
        for arm, modes in ARMS.items()
    }


def gan_statistics(rows):
    """rows: {arm: [checkpoint dicts with GAN_KEYS]} -> terminal means, gates, verdict."""
    term = {arm: {k: float(np.mean([x[k] for x in r[-TERMINAL_WINDOW:]])) for k in GAN_KEYS}
            for arm, r in rows.items()}
    cov, col = term["covered"], term["collapsed"]
    finite = all(math.isfinite(x[k]) for r in rows.values() for x in r for k in GAN_KEYS)
    gates = {
        "construction_valid": cov["cov_raw"] - col["cov_raw"] > 0.3,
        "c_cap_gap_gt_0.15": cov["c_cap"] > col["c_cap"] + 0.15,
        "d_collapse_gap_gt_0.15": col["d_collapse"] > cov["d_collapse"] + 0.15,
        "d_drop_collapsed_lt_covered": col["d_drop"] < cov["d_drop"],
    }
    if not finite:
        verdict = "ERROR"
    elif not gates["construction_valid"]:
        verdict = "INVALID"
    else:
        verdict = "PASS" if all(gates.values()) else "FAIL"
    return {"terminal_means": term, "gates": gates, "verdict": verdict}


def _gan_rows(arms):
    return {arm: [{k: obs[k] for k in ("step",) + GAN_KEYS} for obs in res["observations"]]
            for arm, res in arms.items()}


def publish(arms, receipt=RECEIPT, gif=GIF, device="cpu"):
    rows = _gan_rows(arms)
    stats = gan_statistics(rows)
    receipt.parent.mkdir(parents=True, exist_ok=True)

    def caption(arm, obs):
        return (f"{arm} | step {obs['step']}/{GAN_STEPS}\nC_cap={obs['c_cap']:.2f} "
                f"D_drop={obs['d_drop']:.2f} D_coll={obs['d_collapse']:.2f}")

    frames = ring.render_arms_gif(
        gif, {arm: res["observations"] for arm, res in arms.items()}, caption,
        title=f"CADMA on public-API ring arms (GANTrainer, default recipe, seed 0): {stats['verdict']}")
    recipes = {arm: res["recipe"] for arm, res in arms.items()}
    if recipes["covered"] != recipes["collapsed"]:
        raise RuntimeError("arms resolved different recipes")
    payload = {
        "question": "CADMA attributes a mode-dropped public-API run's deficit to collapse (D_collapse), not D_drop",
        "source_paper": "arXiv:2611.12243 (CADMA)",
        "cohort": "public-API GANTrainer, default get_recipe(total_steps), deterministic init, protocol seed 0",
        "task": {"target": "8-Gaussian ring r=2 std=0.25", "collapsed_training_modes": list(ring.COLLAPSED_MODES),
                 "architecture": f"MLP hidden {ring.HIDDEN}x2 LeakyReLU(0.2)", "steps": GAN_STEPS,
                 "check_every": GAN_CHECK_EVERY, "terminal_window": TERMINAL_WINDOW,
                 "n_eval": N_SAMPLES, "k": K_MAIN, "sampler": "GANTrainer.sample (live, clean)"},
        "resolved_recipe": recipes["covered"],
        "device": str(device), "torch": torch.__version__,
        "rows": rows, "statistics": stats, "gif": gif.name, "gif_frames": frames,
    }
    receipt.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload


def test_short_api_run_renders_actual_training_gif(tmp_path):
    # One arm keeps the CPU smoke cheap; the published study trains both.
    res = ring.train_ring_arm(ring.COLLAPSED_MODES, 2, 1, n_eval=N_SAMPLES,
                              observe=_gan_observe(ring.eval_real(N_SAMPLES)))
    assert isinstance(res["trainer"], ring.GANTrainer)
    assert res["recipe"]["total_steps"] == 2
    assert [o["step"] for o in res["observations"]] == [1, 2]
    for o in res["observations"]:
        assert all(math.isfinite(o[k]) for k in GAN_KEYS)
        assert abs(o["coverage"] - (o["c_cap"] + o["d_drop"])) < 1e-12
    frames = ring.render_arms_gif(
        tmp_path / "goal.gif", {"collapsed": res["observations"]},
        lambda arm, obs: f"{arm} {obs['step']} C_cap={obs['c_cap']:.2f}")
    assert frames == 2 and (tmp_path / "goal.gif").stat().st_size > 1000


def test_published_gan_receipt_is_consistent():
    payload = json.loads(RECEIPT.read_text())
    assert payload["task"]["steps"] == GAN_STEPS and payload["task"]["check_every"] == GAN_CHECK_EVERY
    rows = payload["rows"]
    for arm in ARMS:
        assert [r["step"] for r in rows[arm]] == list(range(GAN_CHECK_EVERY, GAN_STEPS + 1, GAN_CHECK_EVERY))
    assert json.loads(json.dumps(gan_statistics(rows))) == payload["statistics"]
    assert payload["statistics"]["verdict"] in {"PASS", "FAIL", "INVALID", "ERROR"}
    assert payload["resolved_recipe"]["total_steps"] == GAN_STEPS
    assert payload["gif_frames"] == len(rows["covered"])
    assert (REPORT_DIR / payload["gif"]).stat().st_size > 10_000


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("PARTICLEGAN_RUN_SLOW") != "1", reason="full-budget rerun; set PARTICLEGAN_RUN_SLOW=1")
def test_full_gan_study_reproduces_receipt_verdict():
    payload = json.loads(RECEIPT.read_text())
    stats = gan_statistics(_gan_rows(run_gan_arms()))
    assert stats["verdict"] == payload["statistics"]["verdict"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_constructions()
    for name, rows in result.items():
        print(f"== {name} (k={K_MAIN}, N={N_SAMPLES}, fixture draws={SEEDS}) ==")
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
            f"1-rho_g(proxy)={mae_rho_g_proxy(rows):.3f}",
            flush=True,
        )

    def progress(arm, row):
        print(json.dumps({"arm": arm, **row}), flush=True)

    published = publish(run_gan_arms(device=args.device, progress=progress), device=args.device)
    print(json.dumps(published["statistics"], indent=1))
