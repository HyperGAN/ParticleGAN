#!/usr/bin/env python3
"""Zero-training audit of an existing public-trainer BISPF measurement.

Requires the original readout; summaries cannot replace it. No model construction,
training, sampling, requalification, or leaderboard publication occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

TASK = "vector_unequal_width"
SOURCE = "d9d72d53761c898bce20102e1995fc3d899c1447"
CANDIDATE = "bispf-optimized-quintic-v1"
DEFAULT_ARCHIVE = "/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json"
ROOT = Path(__file__).resolve().parents[4]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def frozen(path):
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)


def passes(row, threshold):
    key, op, bound = threshold
    value = row[key]
    assert type(value) in (int, float) and math.isfinite(value)
    assert op in ("<=", ">=")
    return value <= bound if op == "<=" else value >= bound


def audit(archive):
    quality_path = ROOT / "reports/forge/bispf-optimized/quality.json"
    quality = json.loads(quality_path.read_bytes())
    data = archive.read_bytes()
    assert digest(data) == quality["raw_readout_archive"]["sha256"]
    assert len(data) == quality["raw_readout_archive"]["bytes"]
    raw = json.loads(data)
    assert raw["candidate_id"] == quality["candidate"] == CANDIDATE
    assert raw["candidate_revision"] == quality["candidate_revision"]
    assert quality["source_commit"] == SOURCE and quality["seed"] == 0
    records = [r for r in raw["task_results"] if r["task_id"] == TASK]
    selected = [r for r in quality["rows"] if r["task"] == TASK]
    assert len(records) == len(selected) == 1
    record, published = records[0], selected[0]
    assert record["_attempt_id"] == published["attempt"]
    assert record["status"] == published["status"] == "FAIL"
    assert record["execution_path"] == "public_trainer"
    assert record["rng"]["seed"] == 0
    for key, value in published["metrics"].items():
        assert record["metrics"][key] == value, key
    assert digest(canonical(record["metrics"])) == published["final_metrics_sha256"]
    board = json.loads((ROOT / "reports/forge/technique-inventory.json").read_bytes())
    leaders = [r for r in board["rows"] if r["candidate_id"] == CANDIDATE]
    assert len(leaders) == 1
    binding = leaders[0]["bindings"]
    assert binding["source_origin_commit"] == SOURCE
    assert binding["recipe_sha256"] == quality["global_recipe_sha256"]
    assert published["attempt"] in leaders[0]["attempt_ids"]
    task_path = f"configs/forge/tasks/{TASK}.json"
    task_bytes = frozen(task_path)
    task = json.loads(task_bytes)
    assert task["execution"]["host_definition"] == record["evidence"]["host"]["definition"]
    assert task["execution"]["prior"] == record["initial_prior"]
    assert record["prior"] == {"kind": "mog", "learnable": True, "sigma": .025, "standardize": False}
    assert record["initializer"] == "deterministic_orthogonal"
    assert published["sampling"] == {"eval_output_noise": "clean", "sampling_contract_version": 1,
                                     "sampling_law": "public_prior_without_output_noise"}
    assert record["recipe"]["prior_regularizer"] == "sbnp"
    assert record["recipe"]["loss"] == "non_saturating"
    assert record["recipe"]["optimizer_convolution_rule"] == "quintic"
    assert record["recipe"]["total_steps"] == task["execution"]["steps"] == 1200
    host_models = record["evidence"]["host"]["models"]
    assert all(len(shape) <= 2 for model in host_models.values()
               for shape in model["parameter_shapes"].values())
    assert all("SimpleMLP" in model["class"] for model in host_models.values())
    thresholds = task["evaluation"]["thresholds"]
    observations = record["evidence"]["observations"]
    assert [r["step"] for r in observations] == list(range(50, 1201, 50))
    assert observations[-1] == record["metrics"]
    flags = [all(passes(o, t) for t in thresholds) for o in observations]
    longest = suffix = 0
    for flag in flags:
        suffix = suffix + 1 if flag else 0
        longest = max(longest, suffix)
    convergence = published["evaluator_summary"]["convergence"]
    assert sum(flags) == convergence["passing_observations"]
    assert suffix == convergence["passing_suffix"] == 0
    minimum = task["evaluation"]["minimum_stable_checks"]
    assert minimum == 5 and suffix < minimum
    media_path = ROOT / f"reports/forge/bispf-optimized/quality-media/{TASK}.json"
    media = json.loads(media_path.read_bytes())
    gif_path = media_path.with_suffix(".gif")
    assert digest(gif_path.read_bytes()) == media["gif_sha256"]
    assert media["recorded_grade"] == "FAIL" and media["kind"] == "actual_training_saved_observations_gif"
    assert media["optimizer_updates_added"] == media["sampling_draws_added"] == 0
    evaluated_sources = {}
    for path, expected in task["evaluation"]["sources"].items():
        actual = digest(frozen(path))
        assert actual == expected, path
        evaluated_sources[path] = actual
    mechanism_sources = {}
    for path in ("particlegan/bispf.py", "particlegan/optim/dualnorm.py", "particlegan/recipes.py",
                 "particlegan/training.py", "particlegan/particle_prior.py", "experiments/forge/adapters.py"):
        mechanism_sources[path] = {"frozen_sha256": digest(frozen(path)),
                                   "current_sha256": digest((ROOT / path).read_bytes())}
    endpoint = record["metrics"]
    details = {key: endpoint[key] for key in ("component_counts", "component_mass", "target_mass",
        "component_covariance_errors", "component_core_covariance_errors", "component_core_eigen_ratios",
        "component_spill") if key in endpoint}
    if TASK == "vector_unequal_mass":
        rare = len(task["execution"]["host_definition"]["masses"]) - 1
        occupied = [o["step"] for o in observations if o["component_counts"][rare] >= 10]
        details.update(rare_shape_supported_observation_steps=occupied,
            rare_absent_terminal_checks=sum(o["component_counts"][rare] == 0 for o in observations if o["step"] >= 400),
            target_rare_expected_per_training_batch=.02 * 128,
            true_target_probability_zero_rare_in_4096=.98 ** 4096)
    if TASK == "vector_unequal_width":
        details["component_1_share_of_summed_full_covariance_error"] = endpoint["component_covariance_errors"][1] / sum(endpoint["component_covariance_errors"])
        details["full_to_core_covariance_error_ratio"] = endpoint["component_covariance_error"] / endpoint["component_core_covariance_error"]
    if TASK == "vector_overlap":
        details["passing_observation_steps"] = [o["step"] for o, flag in zip(observations, flags) if flag]
        details["last_two_covariance_errors"] = [o["covariance_error"] for o in observations[-2:]]
        details["covariance_error_direction"] = "Not identifiable from covariance norm alone; sample matrices unavailable in readout."
    checkpoint = record["evidence"]["provenance_checkpoint"]
    return {"schema": "bispf_existing_measurement_investigation_v1", "task": TASK,
        "audit_status": "PASS", "scientific_status_unchanged": "FAIL", "qualification_input": False,
        "updates_added": 0, "sampling_draws_added": 0, "seed": 0,
        "candidate": CANDIDATE, "candidate_revision": quality["candidate_revision"],
        "source_origin_commit": SOURCE, "source_digest": quality["source_digest"],
        "recipe_sha256": quality["global_recipe_sha256"], "task_contract_sha256": binding["task_contracts"][TASK],
        "protocol_sha256": binding["protocol_sha256"], "rng_sha256": binding["rng_sha256"],
        "attempt": published["attempt"], "original_readout": quality["raw_readout_archive"],
        "quality_publication_sha256": digest(quality_path.read_bytes()),
        "observations_sha256": digest(canonical(observations)), "observation_count": len(observations),
        "passing_checks": sum(flags), "longest_passing_run": longest, "passing_suffix": suffix,
        "required_terminal_checks": minimum, "endpoint_metrics": published["metrics"],
        "gate_audit": [{"metric": t[0], "op": t[1], "threshold": t[2], "endpoint": endpoint[t[0]],
            "endpoint_passes": passes(endpoint, t), "passing_checks": sum(passes(o, t) for o in observations)} for t in thresholds],
        "detail": details, "effective_recipe": {k: record["recipe"][k] for k in
            ("loss", "lr", "d_lr_mult", "prior_lr_mult", "prior_reg", "prior_regularizer", "optimizer_family",
             "optimizer_smoothing", "batch_size", "z_dim", "num_particles", "reg_coeff", "ema_decay")},
        "prior": record["prior"], "initialization": record["initialization"],
        "rng_stream_count": len(record["rng"]["bindings"]),
        "consumed_stream_state_hashes": checkpoint["named_stream_state_sha256"],
        "saved_samples": record["evidence"]["saved_observer_outputs"],
        "provenance_checkpoint": {k: checkpoint[k] for k in ("sha256", "state_sha256", "purpose", "prerequisite_eligible")},
        "frozen_task_sha256": digest(task_bytes), "evaluated_sources": evaluated_sources,
        "mechanism_sources": mechanism_sources, "convolution_parameters": 0,
        "actual_training_gif": {"path": str(gif_path.relative_to(ROOT)), "sha256": media["gif_sha256"]},
        "limitations": ["Readout audit verifies saved measurements, not unavailable tensor/checkpoint contents.",
                         "Frozen source differs from current source; no replay or causal attribution is inferred."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(DEFAULT_ARCHIVE))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Compare against committed compact result")
    args = parser.parse_args()
    result = audit(args.archive)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        assert json.loads(Path(__file__).with_name("result.json").read_text()) == result
        print(f"PASS {TASK}: exact original readout, all 24 saved checks, frozen sources, projection and GIF; zero updates")
    elif args.output:
        args.output.write_text(encoded)
        print(f"PASS {TASK}: wrote compact audit to {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
