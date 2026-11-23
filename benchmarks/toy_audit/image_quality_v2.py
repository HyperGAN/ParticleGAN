"""A versioned finite-template diagnostic, separate from historical toy gates.

This consumes retained exact prior enumerations and launches no training. Its
TV metrics refer to a declared nearest-template partition, with a rejection
bin for images outside the inherited RMSE tolerance. They are not pixel-law
TV, topology/recognition scores, Forge qualification, or a new default recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.locked_shared.observation import sustained
from benchmarks.transfer_suite.image_tasks import image_metrics
from .capture import write


VERSION = "finite-template-mass-v2"
MASS_TV_MAX = 0.10
PROTOCOL = {
    "version": VERSION,
    "sampling": "Every atom of the saved uniform finite prior, clean live only",
    "quality": "Inherited per-case RMSE, HQ and per-mode coverage bounds",
    "target_mass": "Uniform over the declared templates; zero rejection mass",
    "distribution_tv_max": MASS_TV_MAX,
    "finite_template_tv_max": MASS_TV_MAX,
    "sustain": "All original checkpoints, at least five passing terminal checks",
    "selection": "Fixed endpoint and full terminal suffix; no best checkpoint or EMA substitution",
    "scope": "Strengthened post-training diagnostic, not retroactive qualification",
    "threshold_basis": (
        "A declared 10% partition-mass discrepancy cap. With 32 exact atoms, "
        "at most three atoms' worth of mass discrepancy is permitted. This "
        "rejects the historical gate's accepted 25/75 two-mode law. It is not "
        "a calibrated predictor of downstream image quality."
    ),
}

# These are review groups, not claims of independent application data families.
PROPOSAL_FAMILIES = {
    170: "sparse_layout", 166: "sparse_layout", 159: "trace_location",
    154: "trace_location", 151: "orientation_layout", 150: "photometric_profile",
    131: "topology_multiplicity", 106: "orientation_layout", 80: "topology_multiplicity",
    79: "sparse_layout", 78: "orientation_layout", 77: "photometric_profile",
    76: "mirror_handedness", 75: "orientation_layout", 74: "topology_multiplicity",
    73: "mirror_handedness", 72: "mirror_handedness", 71: "photometric_profile",
    70: "photometric_profile", 69: "orientation_layout", 68: "topology_multiplicity",
    67: "photometric_profile", 66: "photometric_profile", 65: "photometric_profile",
    64: "orientation_layout", 63: "application_named_layout", 62: "photometric_profile",
    61: "application_named_layout", 59: "photometric_profile", 58: "photometric_profile",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(images, templates, thresholds):
    """Measure exact atom counts; reject ambiguous template neighborhoods.

    Invalid images and declarations raise instead of disappearing from the
    denominator. The inherited float32 partition remains exact: changing its
    reduction precision can change nearly tied assignments for poor images.
    The new reject-bin calculation extends that existing measurement.
    """
    images = np.asarray(images, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)
    for name, values in (("images", images), ("templates", templates)):
        if (values.ndim != 4 or values.shape[1:] != (1, 8, 8)
                or not len(values) or not np.isfinite(values).all()):
            raise ValueError(f"finite nonempty {name} N x 1 x 8 x 8 required")
    quality_rmse = float(thresholds["quality_rmse"])
    if not math.isfinite(quality_rmse) or quality_rmse <= 0:
        raise ValueError("quality_rmse must be finite and positive")
    if thresholds["modes"] != len(templates):
        raise ValueError("required modes must equal the declared template count")
    minimum_fraction = float(thresholds["min_mode_fraction"])
    hq_min = float(thresholds["hq_min"])
    if (not 0 < minimum_fraction <= 1 / len(templates)
            or not 0 < hq_min <= 1):
        raise ValueError("invalid inherited quality/coverage bounds")
    distances = np.sqrt(np.square(templates[:, None] - templates[None]).mean((2, 3, 4)))
    np.fill_diagonal(distances, np.inf)
    separation = float(distances.min())
    if separation <= 2 * quality_rmse:
        raise ValueError("template quality neighborhoods overlap; independent labels are ambiguous")
    inherited = image_metrics(torch.from_numpy(images.astype(np.float32)),
                              torch.from_numpy(templates.astype(np.float32)), thresholds)
    mass = np.asarray(inherited["mode_fractions"])
    valid_mass = np.asarray(inherited["quality_mode_fractions"])
    counts = np.rint(mass * len(images)).astype(np.int64)
    valid_counts = np.rint(valid_mass * len(images)).astype(np.int64)
    rejected_mass = 1 - int(valid_counts.sum()) / len(images)
    target = np.full(len(templates), 1 / len(templates))
    nearest_tv = float(np.abs(mass - target).sum() / 2)
    partition_tv = float((np.abs(valid_mass - target).sum() + rejected_mass) / 2)
    return {
        **inherited, "n": len(images), "mode_counts": counts.tolist(),
        "quality_mode_counts": valid_counts.tolist(), "mode_fractions": mass.tolist(),
        "quality_mode_fractions": valid_mass.tolist(), "rejected_mass": rejected_mass,
        "distribution_tv": nearest_tv, "finite_template_tv": partition_tv,
        "template_min_separation_rmse": separation,
    }


def requirements(spec, *, strengthened=True):
    t = spec["thresholds"]
    bounds = [("modes", ">=", t["modes"]), ("hq", ">=", t["hq_min"])]
    if strengthened:
        bounds.extend([("distribution_tv", "<=", MASS_TV_MAX),
                       ("finite_template_tv", "<=", MASS_TV_MAX)])
    return bounds


def accepted(metrics, spec, *, strengthened=True):
    return all(metrics[key] >= bound if op == ">=" else metrics[key] <= bound
               for key, op, bound in requirements(spec, strengthened=strengthened))


def verdict(curve, spec, *, strengthened=True):
    steps = [math.ceil(i * spec["steps"] / spec["thresholds"]["observations"])
             for i in range(1, spec["thresholds"]["observations"] + 1)]
    convergence = sustained(curve, requirements(spec, strengthened=strengthened),
                            expected_steps=steps,
                            minimum=spec["thresholds"]["minimum_stable_checks"])
    final_pass = bool(curve) and accepted(curve[-1], spec, strengthened=strengthened)
    passed = final_pass and convergence["confirmed_step"] is not None
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    return {"status": status, "passed": passed, "final_pass": final_pass,
            "convergence": convergence}


def controls(templates, spec):
    """Exact oracle/negative populations, never trained samples or seed runs."""
    modes, particles = len(templates), spec["particles"]
    if modes % 2 or particles % (2 * modes):
        raise ValueError("paired uniform/imbalance controls require even representable mode counts")
    count = particles // modes
    uniform = np.repeat(templates, count, axis=0)
    unequal_counts = np.array([count // 2] * (modes // 2) + [3 * count // 2] * (modes // 2))
    unequal = np.repeat(templates, unequal_counts, axis=0)
    populations = {
        "exact_uniform": uniform,
        "uniform_permuted": uniform[::-1].copy(),
        "mass_imbalance_tv_025": unequal,
        "one_mode_collapse": np.repeat(templates[:1], particles, axis=0),
        "global_template_mean": np.repeat(templates.mean(0, keepdims=True), particles, axis=0),
        # A permitted image change makes the tolerance limit observable: this
        # oracle certifies template neighborhoods, not exact pixels or topology.
        "within_tolerance_mutation": np.clip(uniform + spec["thresholds"]["quality_rmse"] / 2, 0, 1),
    }
    rows = {}
    for name, images in populations.items():
        measured = score(images, templates, spec["thresholds"])
        rows[name] = {
            "historical_accepts": accepted(measured, spec, strengthened=False),
            "v2_accepts": accepted(measured, spec),
            **{key: measured[key] for key in ("hq", "modes", "distribution_tv", "finite_template_tv")},
        }
    # These guarantees are mathematical fixtures, not result-fitted thresholds.
    assert rows["exact_uniform"]["v2_accepts"]
    assert rows["uniform_permuted"]["v2_accepts"]
    assert rows["mass_imbalance_tv_025"]["historical_accepts"]
    assert not rows["mass_imbalance_tv_025"]["v2_accepts"]
    assert not rows["one_mode_collapse"]["v2_accepts"]
    assert not rows["global_template_mean"]["v2_accepts"]
    assert rows["within_tolerance_mutation"]["v2_accepts"]
    return rows


def failure_components(final, revised):
    failures = []
    if final["hq"] < revised["hq_min"]:
        failures.append(f"quality {final['hq']:.3f}<{revised['hq_min']:.2f}")
    if final["modes"] < revised["required_modes"]:
        failures.append(f"coverage {final['modes']}/{revised['required_modes']}")
    if final["distribution_tv"] > MASS_TV_MAX:
        failures.append(f"nearest mass TV {final['distribution_tv']:.3f}>0.10")
    if final["finite_template_tv"] > MASS_TV_MAX:
        failures.append(f"valid-bin TV {final['finite_template_tv']:.3f}>0.10")
    if revised["convergence"]["passing_suffix"] < revised["convergence"]["minimum_stable_checks"]:
        failures.append(f"terminal suffix {revised['convergence']['passing_suffix']}/5")
    return failures


def host_limits(templates, spec):
    """Analytic architectural facts, separate from measured training defects."""
    flat = np.asarray(templates, dtype=np.float64).reshape(len(templates), -1)
    if spec.get("architecture") == "mean_discriminator":
        means = flat.mean(1)
        return {
            "kind": "critic_projection_nonidentifiability",
            "template_means": means.tolist(),
            "all_projected_templates_equal": bool(np.all(means == means[0])),
            "explanation": "A mean-only critic cannot identify the spatial mode law from equal projected real modes.",
        }
    if spec.get("architecture") == "uniform_generator":
        # The best constant for an image is its pixel mean, with residual
        # RMSE equal to the image's standard deviation.
        errors = np.sqrt(np.square(flat - flat.mean(1, keepdims=True)).mean(1))
        return {
            "kind": "generator_representation_impossibility",
            "best_constant_rmse_per_template": errors.tolist(),
            "all_modes_unreachable_at_quality_bound": bool(np.all(errors > spec["thresholds"]["quality_rmse"])),
            "explanation": "Every spatially uniform output is outside every target quality neighborhood.",
        }
    return {
        "kind": "optimization_cause_unidentified",
        "explanation": "Saved output clouds identify quality/mass/stability defects; they do not identify an optimizer cause.",
    }


def reanalyze(record, *, output, label):
    artifact = Path(record["artifact"])
    capture_path, result_path = artifact / "observations.npz", artifact / "result.json"
    if sha256(capture_path) != record["capture_sha256"] or sha256(result_path) != record["result_sha256"]:
        raise ValueError(f"retained artifact identity changed: {artifact}")
    original = json.loads(result_path.read_text())
    spec = record["spec"]
    with np.load(capture_path, allow_pickle=False) as arrays:
        if arrays["live"].shape[1] != spec["particles"]:
            raise ValueError("retained images do not enumerate the whole declared prior")
        curve = [dict(step=int(step), **score(images, arrays["templates"], spec["thresholds"]))
                 for step, images in zip(arrays["steps"], arrays["live"])]
        oracle = controls(arrays["templates"], spec)
        limitation = host_limits(arrays["templates"], spec)
    if len(curve) != len(original["observations"]):
        raise ValueError("recorded evaluator/capture cadence differs")
    max_difference = 0.0
    for measured, old in zip(curve, original["observations"]):
        if measured["step"] != old["step"]:
            raise ValueError("saved image checkpoints and historical metric steps differ")
        for key in ("hq", "modes", "distribution_tv", "mode_fractions", "quality_mode_fractions", "mean_rmse"):
            difference = float(np.max(np.abs(np.asarray(measured[key]) - np.asarray(old[key]))))
            max_difference = max(max_difference, difference)
            if difference > 1e-7:
                raise ValueError(f"historical metric parity fails for {label}/{measured['step']}/{key}: {difference}")
    historical = verdict(curve, spec, strengthened=False)
    if historical["status"] != record["verdict"]["status"]:
        raise ValueError(f"historical verdict parity fails: {label}")
    revised = verdict(curve, spec)
    revised.update(hq_min=spec["thresholds"]["hq_min"], required_modes=spec["modes"])
    final = curve[-1]
    compact = {
        "label": label, "spec": spec, "historical": record["verdict"], "revised": revised,
        "final": final, "failure_components": failure_components(final, revised),
        "host_limitation": limitation,
        "source_capture_sha256": record["capture_sha256"],
        "source_result_sha256": record["result_sha256"],
        "historical_metric_max_difference": max_difference,
        "min_terminal_5_hq": min(p["hq"] for p in curve[-5:]),
        "max_terminal_5_tv": max(p["finite_template_tv"] for p in curve[-5:]),
        "min_recorded_tv": min(p["finite_template_tv"] for p in curve),
    }
    write(output / f"{label}.json", {**compact, "curve": curve, "controls": oracle})
    return compact, oracle


def contract(case):
    if case.get("pr"):
        return {"family": PROPOSAL_FAMILIES[case["pr"]], "verifies": case["verifies"]}
    name = case["name"]
    family = ("known_impossible_or_nonidentifiable" if name in ("img_mean_discriminator", "img_uniform_generator")
              else "capacity_stress" if name == "img_tiny_generator"
              else "photometric_profile" if name == "img_intensity2"
              else "spatial_position" if "bars" in name or "blobs" in name
              else "orientation_layout")
    return {"family": family, "verifies": case["verifies"]}


def markdown(report):
    counts = report["counts"]
    lines = [
        "# Strengthened image toy diagnostics", "",
        f"{counts['cases']} image cases / {counts['arms']} retained training arms were re-scored without training. "
        f"The original cohort has {counts['historical_pass']} PASS / {counts['historical_fail']} FAIL; "
        f"the separate **{VERSION}** diagnostic has **{counts['v2_pass']} PASS / {counts['v2_fail']} FAIL**. "
        f"{counts['historical_pass_new_fail']} original passing arms fail the stronger definition. "
        "Historical results, ratings and GIFs remain unchanged.", "",
        "## What changed", "",
        "The historical image gate checked RMSE quality and minimum per-mode coverage, but accepted perfect "
        "two-template clouds with 25/75 mass. The new diagnostic retains those bounds and adds nearest-template "
        "mass TV ≤0.10 and quality-bin TV ≤0.10. The latter includes an explicit reject bin, so it cannot hide "
        "invalid pixels by normalizing only the good samples. Every cloud enumerates all 32 atoms; these are "
        "exact finite-prior fractions, not Monte Carlo confidence estimates. The target law is uniform over "
        "the declared templates with no rejection mass. The bound is a declared diagnostic cap, not a "
        "calibrated downstream-quality or Forge threshold.", "",
        "All retained template pairs have disjoint inherited RMSE neighborhoods. The original float32 "
        "measurement is retained because changing reduction precision can change nearly tied assignments "
        "for poor images. Reanalysis reproduces every historical checkpoint's quality and TV to 1e-7, and all "
        "historical PASS/FAIL decisions. Source hashes bind each cloud and original result. The strengthened "
        "gate still requires the complete original cadence and five passing terminal observations. EMA and "
        "best checkpoints cannot replace the live endpoint.", "",
        f"For **every one of the {counts['cases']} cases**, exact uniform/permuted oracles pass, a "
        "TV=0.25 mass imbalance passes the historical gate and fails v2, and one-mode collapse/global-mean "
        "images fail v2. A small within-tolerance pixel mutation passes, exposing the remaining tolerance "
        "limit. These are analytic evaluator controls, not trained successes.", "",
        "This improves the evaluator for every ≤3-rated image case. It does not make thirty fixed pattern "
        "pairs thirty independent application benchmarks or warrant a rating upgrade. C/O, dot-count, "
        "handedness, barcode, chirp and other names still lack held-out geometric/semantic data and independent "
        "recognition/topology/physics oracles. The inpainting, colorization and translation names still lack "
        "conditional inputs. Those broader claims require new task definitions and training, not renamed receipts.", "",
        "## Every case and its observed failure", "",
        "Arms follow the saved proposal ordering (first transpose12, second residual16); shipped cases "
        "have one frozen-reference arm. Q is exact HQ fraction; TV is the quality-bin/reject TV. "
        "Failure entries describe measured output defects and terminal stability, not an identified optimizer cause.", "",
        "All thirty proposal contrasts change both architecture and width (and therefore initialization shapes "
        "and draw order). A passing second arm demonstrates that the exact finite law is learnable by that "
        "reference; it does not isolate the cause of the first arm's failure. PR154 has no trained passing arm "
        "in this cohort. PR63's second arm has a passing endpoint but only four passing terminal checks.", "",
        "| Case / original rating | Scientific question | Historical → v2 | Final Q / TV by arm | Failed v2 components |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        origin = f"[PR{case['pr']}](https://github.com/HyperGAN/ParticleGAN/pull/{case['pr']})" if case.get("pr") else "develop"
        statuses = "; ".join(f"{i+1}: {a['historical']['status']} → {a['revised']['status']}" for i,a in enumerate(case["arms"]))
        values = "; ".join(f"{i+1}: {a['final']['hq']:.3f} / {a['final']['finite_template_tv']:.3f}" for i,a in enumerate(case["arms"]))
        failures = "; ".join(f"{i+1}: {', '.join(a['failure_components']) if a['failure_components'] else 'none'}" for i,a in enumerate(case["arms"]))
        question = case["contract"]["verifies"].replace("|", "/")
        lines.append(f"| {case['name']} · {origin} · {case['rating']}/5 | {question} | {statuses} | {values} | {failures} |")
    lines.extend(["", "## Redundancy and what remains to improve", "",
                  "All groups below remain finite uniform template laws with the same exact-prior evaluator. "
                  "Grouping prevents duplicate application names from inflating coverage. Retain specific "
                  "patterns as architecture regressions only when their frozen host/contrast adds evidence.", "",
                  "| Review group | Members | Next independent question |", "| --- | --- | --- |"])
    next_questions = {
        "photometric_profile": "Held-out intensity/phase/frequency parameterization, with an analytic conditional target rather than two stored images.",
        "orientation_layout": "Held-out positions/orientations and explicit equivariance or layout oracle; orientation coverage alone already has a shipped representative.",
        "sparse_layout": "Held-out sparse placements/quiet-zone widths and a separate feature-position/count oracle; barcode/Braille validity stays untested.",
        "trace_location": "Vary echo delay or chirp slope with held-out values and an independent signal-parameter oracle; grayscale traces do not certify DSP.",
        "topology_multiplicity": "Vary topology/count independently of positions and contrast; score connected components/holes/cardinality with an independent oracle.",
        "mirror_handedness": "Held-out mirrored shapes and an independent signed geometry feature; fixed b/d or L templates alone are template memorization.",
        "application_named_layout": "Supply observed source/mask and score the analytic conditional posterior plus observation consistency; unconditional marginals cannot validate inpainting/translation.",
        "spatial_position": "Keep the bounded spatial-coverage regression; avoid counting residual host reuse as new data-family coverage.",
        "capacity_stress": "Add a fixed representable oracle/optimization reference before treating finite-budget training failure as representational impossibility.",
        "known_impossible_or_nonidentifiable": "Keep as expected-negative diagnostics, never as mandatory qualification passes.",
    }
    groups = defaultdict(list)
    for case in report["cases"]:
        groups[case["contract"]["family"]].append(case["id"])
    for family, members in sorted(groups.items()):
        lines.append(f"| {family} | {', '.join(members)} | {next_questions[family]} |")
    lines.extend(["", "The mean-only critic cannot distinguish equal-mean blob positions: its architectural "
                  "projection makes all real modes identical. The uniform-output generator cannot represent "
                  "stripes at the inherited RMSE threshold. Both expected failures expose host limits, not "
                  "controller bugs. Bars8 loses mode mass; the tiny generator has inadequate observed coverage, "
                  "but clouds alone do not prove that its parameterization is incapable of fitting the law.", "",
                  "No critic/optimizer state was preserved with these image captures, so the report cannot "
                  "assign a causal optimizer mechanism to the remaining quality or collapse failures. The next "
                  "causal study must freeze one substantive factor and save gradients/critic states; rerunning "
                  "the same seeds or increasing budgets would not identify a cause.", "",
                  "## Reproduce", "", "```sh",
                  "/tmp/pr155-e22-venv/bin/python -m benchmarks.toy_audit.image_quality_v2 \\",
                  "  --artifacts /ml2/hypergan/toy-audit-artifacts-20261001 \\",
                  "  --catalog reports/toy_audit/catalog.json \\",
                  "  --raw-output /ml2/hypergan/toy-image-quality-v2-replay \\",
                  "  --report reports/toy_audit/image-quality-v2.json \\",
                  "  --markdown reports/toy_audit/IMAGE_QUALITY_V2.md", "```", "",
                  "Raw curves stay in `--raw-output` outside Git. The compact JSON records final measurements, "
                  "protocol, source/capture identities, gate controls and terminal aggregates. Existing "
                  "[training GIFs](PROBLEMS.md) show the unchanged source runs.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.raw_output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads(args.catalog.read_text())
    records = {"develop-" + r["name"]: [r] for r in json.loads(
        (args.artifacts / "develop-frozen-reference/index.json").read_text()) if r["kind"] == "image"}
    for path in sorted(args.artifacts.glob("pr*/capture-index.json")):
        rows = json.loads(path.read_text())
        if rows and rows[0]["kind"] == "image":
            records[path.parent.name] = rows
    image_cases = [c for c in catalog["cases"] if isinstance(c.get("spec", {}).get("thresholds"), dict)]
    if set(records) != {c["id"] for c in image_cases}:
        raise ValueError("image catalog and retained records do not cover exactly the same cases")
    results = []
    for case in sorted(image_cases, key=lambda c: (-c["rating"], c["name"])):
        arms = []
        oracle = None
        for i, record in enumerate(records[case["id"]]):
            arm, oracle = reanalyze(record, output=args.raw_output, label=f"{case['id']}-arm-{i+1}")
            arms.append(arm)
        results.append({"id": case["id"], "name": case["name"], "pr": case.get("pr"),
                        "head_sha": case.get("head_sha"), "rating": case["rating"],
                        "contract": contract(case), "arms": arms, "controls": oracle,
                        "improvement": {
                            "implemented": "Fixed exact mass/reject-bin TV bounds, six analytic controls and preserved terminal-suffix requirements",
                            "rating_changed": False,
                            "remaining_limit": "Finite templates only; held-out semantic/geometry/application oracles remain absent",
                        },
                        "contrast_changes": {key: [arms[0]["spec"].get(key), arms[1]["spec"].get(key)]
                            for key in sorted(set(arms[0]["spec"]) | set(arms[1]["spec"]))
                            if arms[0]["spec"].get(key) != arms[1]["spec"].get(key)} if len(arms) == 2 else {}})
        print(json.dumps({"case": case["id"], "historical": [a["historical"]["status"] for a in arms],
                          "v2": [a["revised"]["status"] for a in arms]}), flush=True)
    arms = [a for c in results for a in c["arms"]]
    counts = {"cases": len(results), "arms": len(arms),
              "historical_pass": sum(a["historical"]["passed"] for a in arms),
              "historical_fail": sum(not a["historical"]["passed"] for a in arms),
              "v2_pass": sum(a["revised"]["passed"] for a in arms),
              "v2_fail": sum(not a["revised"]["passed"] for a in arms),
              "historical_pass_new_fail": sum(a["historical"]["passed"] and not a["revised"]["passed"] for a in arms),
              "new_pass_historical_fail": sum(a["revised"]["passed"] and not a["historical"]["passed"] for a in arms),
              "controls": dict(Counter(name for c in results for name in c["controls"]))}
    root = Path(__file__).resolve().parents[2]
    report = {"protocol": PROTOCOL, "source_sha256": sha256(__file__), "catalog_sha256": sha256(args.catalog),
              "measurement_source_sha256": {name: sha256(root / name) for name in (
                  "benchmarks/transfer_suite/image_tasks.py", "benchmarks/locked_shared/observation.py")},
              "base_training_sha": catalog["base_sha"], "new_training_runs": 0, "counts": counts, "cases": results}
    write(args.report, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(report))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
