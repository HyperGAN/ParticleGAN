"""Audit one frozen BISPF failure with zero optimizer updates or sampling draws."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

TASK = "staggered100"
SOURCE = "d9d72d53761c898bce20102e1995fc3d899c1447"
ARCHIVE_SHA = "50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe"
DEFAULT_ARCHIVE = Path("/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def source(path):
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)


def audit(archive):
    original_bytes = archive.read_bytes()
    assert sha(original_bytes) == ARCHIVE_SHA, "original readout digest differs"
    original = json.loads(original_bytes)
    quality_path = ROOT / "reports/forge/bispf-optimized/quality.json"
    quality = read(quality_path)
    assert quality["raw_readout_archive"]["sha256"] == ARCHIVE_SHA
    row = next(r for r in quality["rows"] if r["task"] == TASK)
    raw = next(r for r in original["task_results"] if r["task_id"] == TASK)
    summary = next(r for r in quality["attempt_summaries"] if r["attempt_id"] == row["attempt"])
    assert raw["_attempt_id"] == row["attempt"]
    assert raw["metrics"] == row["metrics"]
    assert summary["provenance"]["source_origin_commit"] == SOURCE
    selected = next(r for r in read(ROOT / "reports/forge/technique-inventory.json")["rows"]
                    if r["trainer_family"] == "bispf")
    assert selected["candidate_id"] == quality["candidate"] == "bispf-optimized-quintic-v1"
    assert selected["candidate_revision"] == quality["candidate_revision"]
    assert next(r for r in selected["tasks"] if r["task_id"] == TASK)["status"] == "FAIL"
    paths = [f"configs/forge/tasks/{TASK}.json", "lib/toy_models.py",
             "experiments/forge/adapters.py", "experiments/forge/api.py",
             "particlegan/optim/dualnorm.py", "particlegan/recipes.py",
             "particlegan/bispf.py", "benchmarks/toy100/accuracy.py",
             "benchmarks/toy100/accuracy_gate.py", "benchmarks/toy100/problems.py"]
    sources = {path: source(path) for path in paths}
    task = json.loads(sources[paths[0]])
    assert task["execution"]["model"] == {"hidden": 128, "layers": 3, "fourier": 2}
    assert task["execution"]["resources"] == {"batch_size": 2048, "num_particles": 20000, "z_dim": 2}
    assert raw["initial_prior"] == task["execution"]["prior"]
    assert raw["initializer"] == task["execution"]["initializer"] == "deterministic_orthogonal"
    assert raw["recipe"]["prior_regularizer"] == "sbnp"
    assert raw["evidence"]["prior_policy"]["update"] == "learned"
    assert raw["recipe"]["loss"] == "non_saturating"
    assert raw["recipe"]["output_noise_std"] == raw["recipe"]["ema_decay"] == 0.0
    assert raw["cost"]["completed_steps"] == task["execution"]["steps"] == 7000
    assert row["sampling"] == {"eval_output_noise": "clean", "sampling_contract_version": 1,
                               "sampling_law": "public_prior_without_output_noise"}

    # Execute the exact frozen scalar accuracy predicate, without importing a
    # trainer or drawing new samples. Coverage still belongs to its original grade.
    accuracy_tree = ast.parse(sources["benchmarks/toy100/accuracy.py"])
    predicate = next(n for n in accuracy_tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "passes_accuracy")
    namespace = {"math": math, "Mapping": Mapping, "Any": Any, "EVAL_N": 20000,
                 "LIMITS": task["evaluation"]["accuracy_limits"]}
    exec(compile(ast.Module(body=[predicate], type_ignores=[]), "frozen_accuracy.py", "exec"), namespace)
    passes_accuracy = namespace["passes_accuracy"]
    result = raw["evaluator_result"]
    terminal = result["terminal_checks"]
    assert [t["step"] for t in terminal] == [6000, 6250, 6500, 6750, 7000]
    assert not any(passes_accuracy(t["metrics"]) for t in terminal)
    assert not passes_accuracy(result["holdout_metrics"])
    assert passes_accuracy(result["oracle_metrics"])
    assert result["oracle_metrics"]["passed"] is True
    assert result["coverage_status"] == result["status"] == row["status"] == "FAIL"
    for check in terminal:
        assert check["passed"] == (check["metrics"]["frozen_pass"] and passes_accuracy(check["metrics"]))
    for key, value in row["metrics"].items():
        assert value == result["holdout_metrics"][key]

    # Inspect the exact host classes and optimizer dispatch. No weights or
    # synthetic optimizer updates are constructed as a substitute for training.
    model_tree = ast.parse(sources["lib/toy_models.py"])
    constructors = {}
    for name in ("SimpleMLPGenerator", "SimpleMLPDiscriminator"):
        cls = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        constructors[name] = sorted({n.func.attr for n in ast.walk(init) if isinstance(n, ast.Call)
                                     and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                                     and n.func.value.id == "nn"})
        assert not any("Conv" in kind for kind in constructors[name])
    optimizer_text = sources["particlegan/optim/dualnorm.py"].decode()
    assert 'torch.nn.Conv2d, torch.nn.ConvTranspose2d' in optimizer_text
    assert 'elif parameter.ndim == 2:' in optimizer_text
    recipe_text = sources["particlegan/recipes.py"].decode()
    assert 'self.effective_critic_formulation == "bispf" and self.loss == "relativistic"' in recipe_text

    # Evaluate frozen branch eligibility at the endpoint. Residual-dependent
    # branch outputs are unknown without the actual checkpoint and inputs.
    bispf_tree = ast.parse(sources["particlegan/bispf.py"])
    kernel = next(n for n in bispf_tree.body if isinstance(n, ast.ClassDef) and n.name == "_BISPFPenaltyKernel")
    penalty = next(n for n in kernel.body if isinstance(n, ast.FunctionDef) and n.name == "penalty")
    env = {"is_discrete_simplex": False, "is_image_tensor": False,
           "b_r": 2048, "b_f": 2048, "dim": 2, "call_idx": 7000}
    for name in ("is_continuous_flow", "is_dense_multimode"):
        node = next(n for n in penalty.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        env[name] = eval(compile(ast.Expression(node.value), "frozen_branch.py", "eval"), {}, env)
    capture = next(n for n in penalty.body if isinstance(n, ast.If)
                   and "call_idx >= 1650" in ast.unparse(n.test))
    harmonic_eligible = eval(compile(ast.Expression(capture.test), "frozen_branch.py", "eval"), {}, env)
    assert env["is_dense_multimode"] is True and harmonic_eligible is False

    media_path = ROOT / f"reports/forge/bispf-optimized/quality-media/{TASK}.json"
    media = read(media_path)
    gif_path = media_path.with_suffix(".gif")
    assert sha(gif_path.read_bytes()) == media["gif_sha256"]
    assert media["recorded_grade"] == "FAIL" and media["optimizer_updates_added"] == media["sampling_draws_added"] == 0
    guards = raw["evidence"]["guards"]
    assert guards["all_finite"] is True and guards["unintended_rng_deviations"] == 0
    assert guards["optimizer_updates"] == {"discriminator": 7000, "generator": 7000, "prior": 7000}
    assert len(raw["evidence"]["rng_audits"]) == 35
    assert sum(a["unintended_rng_deviations"] for a in raw["evidence"]["rng_audits"]) == 0
    checkpoint = raw["evidence"]["checkpoint"]
    artifact_root = Path(raw["evidence"]["artifact_root"])
    original_paths = {"checkpoint": artifact_root / checkpoint["path"],
                      "holdout_samples": artifact_root / TASK / "holdout_samples.npz"}
    availability = {}
    manifest = raw["evidence"]["artifact_manifest"]["files"]
    for name, path in original_paths.items():
        relative = str(path.relative_to(artifact_root))
        expected = manifest[relative]["sha256"]
        present = path.is_file()
        if present:
            assert sha(path.read_bytes()) == expected, f"{name} digest differs"
        availability[name] = {"original_path": str(path), "sha256": expected,
                              "status": "AVAILABLE" if present else "MISSING"}
    limits = task["evaluation"]["accuracy_limits"]
    holdout = result["holdout_metrics"]
    return {
        "schema": "bispf_native_failure_audit_v1", "task_id": TASK, "audit_status": "PASS",
        "scientific_status": "FAIL", "qualification_input": False, "qualification_reuse": False,
        "optimizer_updates_added": 0, "sampling_draws_added": 0,
        "candidate_id": quality["candidate"], "candidate_revision": quality["candidate_revision"],
        "source_commit": SOURCE, "source_digest": quality["source_digest"],
        "recipe_sha256": quality["global_recipe_sha256"], "attempt_id": row["attempt"],
        "input_sha256": {"quality.json": sha(quality_path.read_bytes()), "original_readout": ARCHIVE_SHA,
                         "media_receipt": sha(media_path.read_bytes())},
        "frozen_source_sha256": {path: sha(data) for path, data in sources.items()},
        "original_receipt_provenance": summary["provenance"],
        "protocol": {"seed": 0, "initializer": raw["initializer"],
                     "model": task["execution"]["model"], "resources": task["execution"]["resources"],
                     "initial_prior": raw["initial_prior"], "prior_update": "learned",
                     "prior_regularizer": "sbnp", "prior_reg": raw["recipe"]["prior_reg"],
                     "sampling": row["sampling"], "updates": 7000,
                     "evaluation": task["evaluation"]},
        "initial_model_sha256": {k: v["initial_state_sha256"] for k, v in raw["initialization"].items()},
        "named_stream_final_sha256": raw["evidence"]["provenance_checkpoint"]["named_stream_state_sha256"],
        "rng_audits": {"checks": 35, "unintended_deviations": 0},
        "terminal_summary": {"steps": [t["step"] for t in terminal], "passing": 0, "total": 5,
            "precision_range": [min(t["metrics"]["precision"] for t in terminal), max(t["metrics"]["precision"] for t in terminal)],
            "mass_tv_range": [min(t["metrics"]["mass_tv"] for t in terminal), max(t["metrics"]["mass_tv"] for t in terminal)],
            "undefined_shape_checks": sum(t["metrics"]["center_rms_sigma"] is None for t in terminal)},
        "holdout_metrics": holdout, "oracle_metrics": result["oracle_metrics"],
        "accuracy_limit_ratios": {key: holdout[key] / value if holdout[key] is not None else None
                                  for key, value in limits.items()},
        "precision_shortfall": task["evaluation"]["coverage_thresholds"]["min_precision"] - holdout["precision"],
        "host_mechanisms": {"module_constructors": constructors, "eligible_convolution_groups": 0,
            "quintic_delta_exercised": False, "selected_objective": "non_saturating",
            "dense_multimode_endpoint_eligible": env["is_dense_multimode"],
            "harmonic_capture_endpoint_eligible": harmonic_eligible,
            "actual_thermal_scale": "UNKNOWN_without_saved_state",
            "critic_penalty_applied": guards["mechanism_audit"]["mechanisms"]["critic_penalty"]["applied"]},
        "checkpoint_receipt": checkpoint, "original_artifacts": availability,
        "saved_state_probe": "BLOCKED" if not all(p.is_file() for p in original_paths.values()) else "NOT_EXECUTED",
        "actual_training_gif": {"path": str(gif_path.relative_to(ROOT)), "sha256": media["gif_sha256"],
                                "source_attempt": row["attempt"], "observations": media["observation_count"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.archive)
    destination = HERE / "audit-result.json"
    if args.write:
        destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        assert read(destination) == result, "committed audit differs; inspect source and evidence before rewriting"
    print(json.dumps({"task": TASK, "audit": result["audit_status"], "scientific_status": "FAIL",
                      "passing_terminal_checks": "0/5", "saved_state_probe": result["saved_state_probe"],
                      "updates_added": 0, "samples_added": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
