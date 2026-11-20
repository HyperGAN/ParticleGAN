"""Paper-watch study: is the Gini distance correlation a good *evaluation*
metric for GAN runs?

Source paper: "EmbeddGAN" (arXiv:2609.22508). The paper's framework --
replace the discriminator with an embedding network trained against the
Gini distance correlation rho_g -- is a replacement for the ParticleGAN
formulation and out of scope. The salvage tested here is rho_g itself,
computed on raw samples, as a distribution-level evaluation metric.

Question: across a covered and a deliberately mode-dropped public-API run,
does rho_g separate the collapsed from the covered run at least as well as
the run's own critic loss, and track true (Naeem) coverage across
checkpoints at least as well?

Execution (toy-test contract, reports/toy_audit/api_contract/README.md):
both arms train through ``GANTrainer`` with the unmodified default
``get_recipe()`` (only the task budget ``total_steps`` is set), public
deterministic initialization and protocol seed 0 (``lib/paper_watch_ring``).
Ground truth is by construction, never by the metric: the collapsed arm's
training law is the 2-of-8-mode subset; both arms are evaluated against the
full mixture. No seed replicates: separation is measured over the terminal
checkpoint window of the single seed-0 trajectory per arm.

Declared gate (fixed before training; ``study_statistics``):
  * construction valid: terminal-window mean coverage, covered minus
    collapsed, > 0.3 (else INVALID, not FAIL);
  * separation: standardized terminal-window gap for rho_g > 2 and >= the
    gap of the run's own critic loss;
  * tracking: pooled Spearman(rho_g, coverage) < -0.7 and |.| >= that of
    the run's own critic loss.

History: the first revision of this PR trained a hand-written SimpleMLP
softplus loop over seeds {0, 1, 2}; that loop bypassed the ParticleGAN API
and is superseded (its numbers are in the PR history, not this receipt).

Publish the receipt and actual-training GIF (GPU optional):

    python tests/test_gini_distance_metric.py --device cuda:0

The full rerun test is marked ``slow`` (deselected by default; run with
``-m slow``). Default tests check the metric units, a short public-API
smoke run with GIF rendering, and the committed receipt's internal
consistency.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib import paper_watch_ring as ring  # noqa: E402
from lib.toy_metrics import gini_distance_correlation, naeem_coverage  # noqa: E402

STEPS = 1200
CHECK_EVERY = 100
TERMINAL_WINDOW = 4
N_EVAL = 2048
METRIC_KEYS = ("rho_g", "critic_loss", "critic_loss_full", "coverage")
REPORT_DIR = _REPO_ROOT / "reports" / "paper_watch" / "gini-distance-metric"
RECEIPT = REPORT_DIR / "receipt.json"
GIF = REPORT_DIR / "goal.gif"
ARMS = {"covered": ring.ALL_MODES, "collapsed": ring.COLLAPSED_MODES}


def _observe(real_eval):
    def observe(obs):
        fake = obs["fake"]
        return {
            "rho_g": gini_distance_correlation(
                real_eval, fake, max_n=512, generator=torch.Generator().manual_seed(1234)),
            "coverage": naeem_coverage(real_eval, fake, k=5),
        }
    return observe


def run_study(steps=STEPS, check_every=CHECK_EVERY, device="cpu", n_eval=N_EVAL, progress=None):
    real_eval = ring.eval_real(n_eval)
    out = {}
    for arm, modes in ARMS.items():
        result = ring.train_ring_arm(
            modes, steps, check_every, device=device, n_eval=n_eval,
            observe=_observe(real_eval),
            progress=(lambda row, arm=arm: progress(arm, row)) if progress else None)
        out[arm] = result
    return out


# =========================
# Statistics and gate
# =========================

def _standardized_gap(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    pooled = math.sqrt((float(a.var(unbiased=True)) + float(b.var(unbiased=True))) / 2)
    gap = abs(float(a.mean()) - float(b.mean()))
    if pooled == 0:
        return float("inf") if gap else 0.0
    return gap / pooled


def study_statistics(rows):
    """rows: {arm: [checkpoint dicts with METRIC_KEYS]} -> statistics + verdict."""
    term = {arm: r[-TERMINAL_WINDOW:] for arm, r in rows.items()}
    sep = {key: _standardized_gap([x[key] for x in term["collapsed"]],
                                  [x[key] for x in term["covered"]])
           for key in ("rho_g", "critic_loss", "critic_loss_full")}
    pooled = [x for r in rows.values() for x in r]
    corr = {key: float(spearmanr([x[key] for x in pooled], [x["coverage"] for x in pooled]).statistic)
            for key in ("rho_g", "critic_loss", "critic_loss_full")}
    cov = {arm: sum(x["coverage"] for x in r) / len(r) for arm, r in term.items()}
    finite = all(math.isfinite(x[k]) for r in rows.values() for x in r for k in METRIC_KEYS)
    gates = {
        "construction_valid": cov["covered"] - cov["collapsed"] > 0.3,
        "rho_g_separation_gt_2": sep["rho_g"] > 2.0,
        "rho_g_separation_ge_critic": sep["rho_g"] >= sep["critic_loss"],
        "rho_g_coverage_spearman_lt_-0.7": corr["rho_g"] < -0.7,
        "rho_g_tracking_ge_critic": abs(corr["rho_g"]) >= abs(corr["critic_loss"]),
    }
    if not finite:
        verdict = "ERROR"
    elif not gates["construction_valid"]:
        verdict = "INVALID"
    else:
        verdict = "PASS" if all(gates.values()) else "FAIL"
    return {"terminal_coverage": cov, "separation": sep, "correlation": corr,
            "gates": gates, "verdict": verdict}


def _rows(study):
    return {arm: [{k: obs[k] for k in ("step",) + METRIC_KEYS} for obs in res["observations"]]
            for arm, res in study.items()}


def publish(study, receipt=RECEIPT, gif=GIF, device="cpu"):
    rows = _rows(study)
    stats = study_statistics(rows)
    receipt.parent.mkdir(parents=True, exist_ok=True)

    def caption(arm, obs):
        return (f"{arm} | step {obs['step']}/{STEPS}\nrho_g={obs['rho_g']:.3f} "
                f"cov={obs['coverage']:.2f} critic={obs['critic_loss']:.3f}")

    frames = ring.render_arms_gif(
        gif, {arm: res["observations"] for arm, res in study.items()}, caption,
        title=f"Gini distance correlation study (GANTrainer, default recipe, seed 0): {stats['verdict']}")
    recipes = {arm: res["recipe"] for arm, res in study.items()}
    if recipes["covered"] != recipes["collapsed"]:
        raise RuntimeError("arms resolved different recipes")
    payload = {
        "question": "rho_g separates collapsed from covered runs and tracks coverage at least as well as critic loss",
        "source_paper": "arXiv:2609.22508 (EmbeddGAN), metric salvage only",
        "cohort": "public-API GANTrainer, default get_recipe(total_steps), deterministic init, protocol seed 0",
        "task": {"target": "8-Gaussian ring r=2 std=0.25", "collapsed_training_modes": list(ring.COLLAPSED_MODES),
                 "architecture": f"MLP hidden {ring.HIDDEN}x2 LeakyReLU(0.2)", "steps": STEPS,
                 "check_every": CHECK_EVERY, "terminal_window": TERMINAL_WINDOW, "n_eval": N_EVAL,
                 "naeem_k": 5, "rho_g_max_n": 512, "sampler": "GANTrainer.sample (live, clean)"},
        "resolved_recipe": recipes["covered"],
        "device": str(device), "torch": torch.__version__,
        "rows": rows, "statistics": stats, "gif": gif.name, "gif_frames": frames,
    }
    receipt.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload


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


def test_oracle_and_destructive_controls():
    real = ring.eval_real(1024)
    oracle = ring.sample_ring(1024, ring.ALL_MODES, torch.Generator().manual_seed(5))
    dropped = ring.sample_ring(1024, ring.COLLAPSED_MODES, torch.Generator().manual_seed(5))
    g = torch.Generator().manual_seed(1234)
    assert abs(gini_distance_correlation(real, oracle, generator=g)) < 0.02
    assert naeem_coverage(real, oracle) > 0.9
    assert gini_distance_correlation(real, dropped, generator=g) > 0.2
    assert naeem_coverage(real, dropped) < 0.4


# =========================
# Public-API execution
# =========================

def test_short_api_run_renders_actual_training_gif(tmp_path):
    # One arm keeps the CPU smoke cheap; the full study trains both.
    real_eval = ring.eval_real(256)
    res = ring.train_ring_arm(ring.COLLAPSED_MODES, 2, 1, n_eval=256, observe=_observe(real_eval))
    assert isinstance(res["trainer"], ring.GANTrainer)
    assert res["recipe"]["total_steps"] == 2
    assert [o["step"] for o in res["observations"]] == [1, 2]
    assert all(math.isfinite(o[k]) for o in res["observations"] for k in METRIC_KEYS)
    frames = ring.render_arms_gif(
        tmp_path / "goal.gif", {"collapsed": res["observations"]},
        lambda arm, obs: f"{arm} {obs['step']} rho_g={obs['rho_g']:.2f}")
    assert frames == 2 and (tmp_path / "goal.gif").stat().st_size > 1000


def test_published_receipt_is_consistent():
    payload = json.loads(RECEIPT.read_text())
    assert payload["task"]["steps"] == STEPS and payload["task"]["check_every"] == CHECK_EVERY
    rows = payload["rows"]
    for arm in ARMS:
        assert [r["step"] for r in rows[arm]] == list(range(CHECK_EVERY, STEPS + 1, CHECK_EVERY))
    recomputed = study_statistics(rows)
    assert json.loads(json.dumps(recomputed)) == payload["statistics"]
    assert payload["statistics"]["verdict"] in {"PASS", "FAIL", "INVALID", "ERROR"}
    assert payload["resolved_recipe"]["total_steps"] == STEPS
    assert payload["gif_frames"] == len(rows["covered"])
    assert (REPORT_DIR / payload["gif"]).stat().st_size > 10_000


@pytest.mark.slow
def test_full_study_reproduces_receipt_verdict():
    payload = json.loads(RECEIPT.read_text())
    stats = study_statistics(_rows(run_study()))
    assert stats["verdict"] == payload["statistics"]["verdict"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    def progress(arm, row):
        print(json.dumps({"arm": arm, **row}), flush=True)

    result = publish(run_study(device=args.device, progress=progress), device=args.device)
    print(json.dumps(result["statistics"], indent=1))
