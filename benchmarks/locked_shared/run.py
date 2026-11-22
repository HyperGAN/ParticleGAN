"""Run measured toy outcomes. No arm is rejected for changing settings."""

from __future__ import annotations

import math

import torch

from benchmarks.legacy.gan_loss import GANLoss
from benchmarks.legacy.grad_regularizers import GradientPenalty
from . import mode_hold, trajectory, two_pole


def thinned_cap():
    # Same effective cap as conceptmod's ThinnedBCap: ignores requested κ=1.
    return GradientPenalty("b_cap", coeff=1.0, kappa=100.0, norm="l2")


def no_cap():
    return GradientPenalty("f_none", coeff=0.0)


def vanilla():
    return GANLoss("logistic", "vanilla")


def experiments():
    return (
        ("two_pole", "locked_shared", lambda: two_pole.train()),
        ("two_pole", "stranger", lambda: two_pole.train(pairing="stranger")),
        ("two_pole", "thinned_cap", lambda: two_pole.train(cap_factory=thinned_cap)),
        ("trajectory", "locked_shared", lambda: trajectory.train()),
        ("trajectory", "stranger", lambda: trajectory.train(pairing="stranger")),
        ("trajectory", "nearest_stranger", lambda: trajectory.train(pairing="nearest_stranger")),
        ("ring", "locked_shared", lambda: mode_hold.train_mode_hold()),
        ("ring", "cap_off", lambda: mode_hold.train_mode_hold(cap_factory=no_cap)),
        ("ring", "vanilla", lambda: mode_hold.train_mode_hold(gan_factory=vanilla)),
        ("ring", "fm_on", lambda: mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(fm_weight=0.1))),
    )


def run_all(*, log=True):
    torch.set_num_threads(1)
    rows = []
    for toy, arm, train in experiments():
        if log:
            print(f"start toy={toy} arm={arm}", flush=True)
        row = {"toy": toy, "arm": arm, **train()}
        # A numerical blow-up cannot accidentally win a comparison.
        if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            raise ValueError(f"non-finite metrics for {toy}/{arm}: {row}")
        rows.append(row)
        if log:
            print(f"done toy={toy} arm={arm} verdict={row['verdict']} {metrics(row)}", flush=True)
    return rows


def metrics(row):
    if row["toy"] == "two_pole":
        return f"travel={row['mean_abs']:.6f}; median slope={row['grad_med']:.6f}"
    if row["toy"] == "trajectory":
        return f"identity MSE={row['identity_mse']:.6f}"
    return f"modes={row['modes']}/8; HQ={row['hq']:.2%}; effective modes={row['effective_modes']:.3f}"


def compare(rows, reference):
    """Compare every reported numeric metric and verdict, not just PASS flags."""
    expected = {(r["toy"], r["arm"]): r for r in reference}
    actual = {(r["toy"], r["arm"]): r for r in rows}
    if len(actual) != len(rows) or len(expected) != len(reference) or actual.keys() != expected.keys():
        raise ValueError("reference and benchmark must contain the same unique rows")
    checks = []
    for row in rows:
        ref = expected[row["toy"], row["arm"]]
        errors = []
        mismatches = []
        if row.keys() != ref.keys():
            mismatches.append("metric fields differ")
        for key, value in row.items():
            if key not in ref:
                continue
            if isinstance(value, (float, int)):
                errors.append(abs(value - ref[key]))
                if not math.isfinite(value) or not math.isfinite(ref[key]) or not math.isclose(value, ref[key], rel_tol=1e-6, abs_tol=1e-7):
                    mismatches.append(f"{key}: {value} != {ref[key]}")
            elif value != ref[key]:
                mismatches.append(f"{key}: {value} != {ref[key]}")
        checks.append({"toy": row["toy"], "arm": row["arm"],
                       "match": not mismatches, "max_abs_error": max(errors, default=0.0),
                       "mismatches": mismatches})
    return checks


def markdown(report):
    rows = report["rows"]
    parity = {(r["toy"], r["arm"]): r for r in report.get("parity", [])}
    locked = [r for r in rows if r["arm"] == "locked_shared"]
    passed = sum(r["verdict"] == "PASS" for r in locked)
    lines = ["# Locked shared: measured behavior", "",
             "Follow-up: [existing-formulation comparison](comparison.md) and [original locked_shared suite row](suite_locked.md).", "",
             f"**locked_shared meets {passed}/{len(locked)} behavioral targets in this run.**", "",
             "Every row trains. Verdicts use measured outputs only; no configuration gates.", "",
             "| Toy | Variant | Measurement | Result | conceptmod parity |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        check = parity.get((row["toy"], row["arm"]))
        match = ("MATCH" if check["match"] else "MISMATCH") if check else "Not run"
        lines.append(f"| {row['toy']} | {row['arm']} | {metrics(row)} | **{row['verdict']}** | {match} |")
    lines += ["", "Thresholds copied from the reference:", "",
              "- Two-pole cloud: mean absolute travel ≥ 0.30 **and** median critic slope ≤ 1.0 (80 steps).",
              "- Trajectory: same-seed identity MSE ≤ 0.02 (400 steps).",
              "- Ring: ≥ 7/8 modes and ≥ 90% of samples within 3σ of a center (1,200 steps). ≤ 2 modes is FAIL; other results are INCONCLUSIVE.", "",
              "These are different host experiments, not a complete config × toy sweep. "
              "Unrun combinations earn no result. All use the original seed 0 on CPU, one thread. "
              "VICReg 0.05 and host latent width 4 remain in trajectory/ring; "
              "cover 1.5 is a training term in trajectory and only a logged score in two-pole. "
              "Ring has no cover loss.", ""]
    if parity:
        count = sum(c["match"] for c in parity.values())
        error = max(c["max_abs_error"] for c in parity.values())
        lines += [f"**Reference parity: {count}/{len(parity)} rows match; maximum absolute metric difference {error:.3g}.**",
                  "Tolerance: relative 1e-6, absolute 1e-7. Reference runs use the same loaded ParticleGAN primitives, "
                  "but the original conceptmod training code and constructors. This verifies the extraction and PR #36 builder wiring, "
                  "not equivalence to every historical PyPI version.", ""]
        for check in parity.values():
            for mismatch in check["mismatches"]:
                lines.append(f"- {check['toy']}/{check['arm']}: {mismatch}")
    if "pypi_reference" in report:
        wheel = report["pypi_reference"]
        lines += [f"PyPI {wheel['filename']}: the four primitive source files "
                  "(`GANLoss`, gradient penalty, particle prior, VICReg) are byte-for-byte identical "
                  "to the files used for this run. Wheel and source SHA-256 hashes are recorded in `results.json`.", ""]
    if passed < len(locked):
        lines += ["The checked-in conceptmod leaderboard's all-pass claim does **not** reproduce here. "
                  "The original reference loops have the same failures. Keep the parity finding separate "
                  "from acceptance: investigate the trajectory/ring results before claiming a behavioral sweep. "
                  "Budgets, seed, thresholds and algorithms were not retuned to make this table pass.", ""]
    lines += ["Source: [conceptmod 5571213](https://github.com/HyperGAN/conceptmod/tree/5571213f5e8e129cfda45c785c3f30aad9c1d8c9/conceptmod/toys); "
              "[ParticleGAN PR #36](https://github.com/HyperGAN/ParticleGAN/pull/36).",
              "", f"Runtime: Python {report['python']}; PyTorch {report['torch']}. "
              "Full precision metrics and source hashes are in `results.json`.", "",
              "CPU toy results only; no GPU or downstream application transfer is claimed.", ""]
    return "\n".join(lines)
