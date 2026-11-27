"""End-to-end tests for the Radford 32x32 BN-DCGAN B=256 Forge diagnostic guard."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch
from torch import nn

from benchmarks.toy_audit import api_bn_dcgan_rgb32 as guard_api
from benchmarks.toy_audit import api_run
from experiments.forge import adapters, bn_dcgan_guard
from experiments.forge.contracts import canonical, file_hash
from experiments.forge.mechanisms import NAMES as MECHANISM_NAMES
from experiments.forge.planning import load_idea, resolve_idea
from experiments.forge.preflight import task_preflight
from experiments.forge.priors import RECIPE_PRIOR_CONTRACT
from experiments.forge.sampling import PUBLIC_PRIOR_CLEAN, executed_receipt, expected_policy
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view, qualify, validate_view

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = guard_api.TASK_ID
VIEW_ID = "bn_dcgan_b256_diagnostic"


@pytest.fixture
def cpu_single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(old)


def _write_trace(
    path: Path,
    *,
    steps: int,
    warmup_updates: int,
    duration_ms: float,
    finite_losses: bool = True,
) -> dict:
    with path.open("w", encoding="utf-8") as stream:
        for step in range(1, steps + 1):
            phase = "warmup" if (step - 1) < warmup_updates else "measured"
            row = {
                "step": step,
                "phase": phase,
                "duration_ms": float(duration_ms),
                "finite_losses": finite_losses,
                "losses": {"loss_d": 0.9 if finite_losses else "nan", "loss_g": 1.1},
            }
            stream.write(canonical(row) + "\n")
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def _synthetic_complete_evidence(
    task: dict,
    tmp_path: Path,
    *,
    device_type: str = "cuda",
    duration_ms: float = 18.0,
    metric_overrides: dict | None = None,
    bad_checkpoint_index: int | None = None,
    trace_steps: int | None = None,
    trace_name: str = "updates.jsonl",
) -> dict:
    spec = task["execution"]["host_definition"]
    steps = int(task["execution"]["steps"])
    obs_count = int(task["evaluation"]["observations"])
    warmup = int(task["evaluation"]["cuda_latency_gate"]["warmup_updates"])
    ckpts = sorted({math.ceil(i * steps / obs_count) for i in range(1, obs_count + 1)})

    good_metrics = {
        "finite_fraction": 1.0,
        "feature_fid": 18.0,
        "pixel_diversity_ratio": 0.98,
        "class_coverage": 1.0,
        "class_mass_tv": 0.12,
        "spectral_l1": 0.015,
    }
    if metric_overrides:
        good_metrics.update(metric_overrides)

    bad_metrics = {
        "finite_fraction": 1.0,
        "feature_fid": 120.0,
        "pixel_diversity_ratio": 0.35,
        "class_coverage": 0.40,
        "class_mass_tv": 0.65,
        "spectral_l1": 0.12,
    }

    observations = []
    for idx, step in enumerate(ckpts):
        row_metrics = bad_metrics if (bad_checkpoint_index is not None and idx == bad_checkpoint_index) else good_metrics
        observations.append({"step": step, **row_metrics})

    g_ref, d_ref = guard_api.build_models(spec, z_dim=int(spec["resources"]["z_dim"]))
    contract = bn_dcgan_guard._contract_payload(
        spec,
        g_params=sum(p.numel() for p in g_ref.parameters()),
        d_params=sum(p.numel() for p in d_ref.parameters()),
    )
    trace_info = _write_trace(
        tmp_path / trace_name,
        steps=steps if trace_steps is None else trace_steps,
        warmup_updates=warmup,
        duration_ms=duration_ms,
    )
    return {
        **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
        "scoring_weights": "live",
        "prior_policy": {
            "contract": RECIPE_PRIOR_CONTRACT,
            "applicability": "latent_prior",
            "update": "learned",
            "trainable_locations": True,
            "regularizer": "vicreg",
            "weight": 0.0,
            "target_std": 1.0,
            "eps": 1e-4,
            "l2": 0.0,
        },
        "bn_dcgan_guard_contract": contract,
        "actual_training_gif": {"path": str((tmp_path / "goal.gif").resolve()), "sha256": "a" * 64, "frames": obs_count},
        "artifacts": {"updates": trace_info},
        "runtime": {"device": "cuda:0" if device_type == "cuda" else "cpu", "device_type": device_type},
        "validity": {
            "finite_model_state": True,
            "changed_generator": True,
            "changed_discriminator": True,
            "generator_has_batchnorm2d": True,
            "discriminator_has_batchnorm2d": True,
        },
        "guards": {
            "all_finite": True,
            "optimizer_updates": {"generator": steps, "discriminator": steps, "prior": steps},
            "hooks_exercised": True,
            "mechanism_audit": {
                "mechanisms": {
                    name: {"requested": False, "enabled": False, "calls": 0, "eligible": 0, "applied": 0}
                    for name in MECHANISM_NAMES
                }
            },
            "unintended_rng_deviations": 0,
        },
        "observations": observations,
        "live": deepcopy(observations[-1]),
    }


def test_continuous_data_regime_and_oracle_negative_controls():
    old_rng = torch.get_rng_state().clone()
    g1 = torch.Generator(device="cpu").manual_seed(0)
    g2 = torch.Generator(device="cpu").manual_seed(0)
    imgs1, lbls1 = guard_api.sample_continuous_rgb32(256, g1, return_labels=True)
    imgs2, lbls2 = guard_api.sample_continuous_rgb32(256, g2, return_labels=True)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert torch.equal(imgs1, imgs2) and torch.equal(lbls1, lbls2)
    assert imgs1.shape == (256, 3, 32, 32)
    assert float(imgs1.min()) >= -1.0 and float(imgs1.max()) <= 1.0

    # Natural-image spatial geometry: pooled 8x8 med_nn / med_all > 0.50 across batch sizes.
    for batch_n in (32, 64, 256):
        ratio = guard_api.pooled8_med_nn_ratio(imgs1[:batch_n], max_samples=batch_n)
        assert ratio > 0.50, f"expected natural-image pooled8 med_nn/med_all > 0.50 at B={batch_n}, got {ratio}"

    controls = guard_api.oracle_controls()
    assert controls["data_natural_image_regime"]["passed"] is True
    assert controls["oracle_real_holdout"]["passed"] is True
    expected_negatives = {
        "negative_constant_zero",
        "negative_single_class",
        "negative_two_classes",
        "negative_low_diversity_collapsed",
        "negative_uniform_noise",
        "negative_gaussian_noise",
        "negative_blurred_real",
        "negative_memorized_8",
        "negative_nonfinite",
    }
    assert expected_negatives <= set(controls)
    for name, entry in controls.items():
        assert entry["passed"] is entry["expected_pass"], f"control {name} unexpected outcome: {entry}"
    for name in expected_negatives:
        assert controls[name]["passed"] is False and controls[name]["failed_bounds"]


def test_radford_bn_dcgan_models_and_nonqualifying_view_scope():
    import particlegan as pg

    g_mod, d_mod = guard_api.build_models()
    assert any(isinstance(m, nn.BatchNorm2d) for m in g_mod.modules())
    assert any(isinstance(m, nn.BatchNorm2d) for m in d_mod.modules())
    pg.init.deterministic_orthogonal_(g_mod, seed=0, strict=True)
    pg.init.deterministic_orthogonal_(d_mod, seed=1, strict=True)
    with torch.no_grad():
        x = g_mod(torch.randn(8, 128))
        logits = d_mod(x)
    assert x.shape == (8, 3, 32, 32)
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert logits.shape == (8, 1)

    tasks = load_tasks(ROOT)
    assert TASK_ID in tasks
    task = tasks[TASK_ID]
    assert task["adapter"] == bn_dcgan_guard.ADAPTER
    assert task["evaluation"]["qualification_input"] is False
    assert expected_policy(task) == executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")

    view = load_view(ROOT, VIEW_ID)
    assert view["evidence_scope"] == "research_diagnostic"
    assert [a["task"] for a in view["assignments"]] == [TASK_ID]
    assert view["assignments"][0]["importance"] == "diagnostic"
    assert view["assignments"][0]["qualification_tier"] == 1

    q = qualify(view, tasks, [], candidate={})
    assert q["status"] == "DIAGNOSTIC"
    assert q["qualified_tier"] == 0
    assert q["eligible"] is False
    assert q["qualification_input"] is False

    # Cannot smuggle bn_dcgan_guard into a qualification view.
    bad_view = deepcopy(view)
    bad_view.pop("evidence_scope")
    bad_view["assignments"][0]["importance"] = "required"
    with pytest.raises(ValueError, match="cannot grant qualification"):
        validate_view(bad_view, tasks)

    # Task validation rejects tampered qualification flag or thresholds.
    tampered = deepcopy(task)
    tampered["evaluation"]["qualification_input"] = True
    with pytest.raises(ValueError, match="qualification_input"):
        bn_dcgan_guard.validate_task(tampered)


def test_preflight_across_axiom_unified_v15_bispf_and_forge_probe():
    tasks = load_tasks(ROOT)
    task = tasks[TASK_ID]

    for idea_id in ("axiom-v5", "unified-v15", "bispf-v2", "minimal-v2"):
        cand = load_idea(ROOT, idea_id)
        assert adapters.adapter_preflight(task, cand, root=ROOT) == [], f"{idea_id} adapter_preflight blocked"
        assert task_preflight(deepcopy(task), cand, {"seed": 0}, root=ROOT, tasks=tasks) == [], (
            f"{idea_id} task_preflight blocked"
        )
        req = resolve_idea(ROOT, idea_id, view_id=VIEW_ID, through_tier=1, execution_backend="cuda")
        assert req["tasks"][TASK_ID]["preflight_blockers"] == []

    for preset in ("axiom", "unified_v15", "bispf", "bcap_adam"):
        cand = {"recipe_preset": preset, "claim_contract": {"sampling_law": "task_declared"}}
        assert adapters.adapter_preflight(task, cand, root=ROOT) == [], f"preset {preset} blocked"

    strict_cand = {
        "recipe_preset": "bispf",
        "recipe_overrides": {
            "constraint_geometry_mode": "strict_progress",
            "optimizer_family": "dualnorm",
            "optimizer_convolution": "per_offset",
            "betas": (0.0, 0.999),
            "d_betas": (0.0, 0.999),
            "prior_betas": (0.0, 0.999),
            "input_noise_std": 0.0,
            "output_noise_std": 0.0,
        },
        "claim_contract": {"sampling_law": "task_declared"},
    }
    blockers = adapters.adapter_preflight(task, strict_cand, root=ROOT)
    assert any("strict_progress" in msg and "BatchNorm" in msg for msg in blockers), blockers

    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.forge",
            "probe",
            "axiom-v5",
            "--task",
            TASK_ID,
            "--view",
            VIEW_ID,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    payload = json.loads(probe.stdout)
    assert payload["candidate"] == "axiom-v5"
    assert payload["task"] == TASK_ID
    assert payload["training_launched"] is False


def test_bounded_cpu_prefix_forge_execution_gif_and_api_run(tmp_path, cpu_single_thread):
    tasks = load_tasks(ROOT)
    original = deepcopy(tasks[TASK_ID])

    prefix = deepcopy(original)
    prefix["id"] = f"{TASK_ID}-software-prefix"
    prefix["execution"]["steps"] = 2
    prefix["execution"]["host_definition"]["steps"] = 2
    prefix["execution"]["host_definition"]["observations"] = 2
    prefix["execution"]["host_definition"]["eval_samples"] = 32
    prefix["execution"]["host_definition"]["minimum_stable_checks"] = 2
    prefix["evaluation"]["observations"] = 2
    prefix["evaluation"]["eval_samples"] = 32
    prefix["evaluation"]["minimum_stable_checks"] = 2

    cand = {
        "id": "software-prefix-bispf",
        "recipe_preset": "bispf",
        "claim_contract": {"sampling_law": "task_declared"},
    }
    request = {"candidate": cand, "protocol": {"seed": 0}, "tasks": {prefix["id"]: prefix}}
    old_rng = torch.get_rng_state().clone()
    out_dir = tmp_path / "forge_prefix"
    raw = adapters.run_task(request, {"task_id": prefix["id"]}, out_dir, "cpu")
    assert torch.equal(old_rng, torch.get_rng_state())

    assert raw["cost"]["completed_steps"] == 2
    assert raw["qualification_input"] is False
    ev = raw["evidence"]
    assert ev["guards"]["all_finite"] is True
    assert ev["guards"]["hooks_exercised"] is True
    assert ev["guards"]["unintended_rng_deviations"] == 0
    assert ev["guards"]["optimizer_updates"] == {"generator": 2, "discriminator": 2, "prior": 2}
    assert ev["validity"]["finite_model_state"] is True
    assert ev["validity"]["changed_generator"] is True
    assert ev["validity"]["changed_discriminator"] is True
    assert ev["validity"]["generator_has_batchnorm2d"] is True
    assert ev["validity"]["discriminator_has_batchnorm2d"] is True
    assert ev["runtime"]["device_type"] == "cpu"
    assert all(ev[k] == v for k, v in expected_policy(original).items())

    gif_info = ev["actual_training_gif"]
    gif_path = Path(gif_info["path"])
    assert gif_path.is_file()
    assert gif_info["sha256"] == file_hash(gif_path)
    assert gif_info["frames"] == 2
    with Image.open(gif_path) as gif:
        assert gif.n_frames == 2

    trace_info = ev["artifacts"]["updates"]
    trace_path = Path(trace_info["path"])
    assert trace_path.is_file()
    assert trace_info["sha256"] == file_hash(trace_path)
    assert (out_dir / "state.pt").is_file()

    # Grading a 2-step software prefix against the full 1200-step task must fail closed as INCOMPLETE.
    assert grade_result(original, raw)["status"] == "INCOMPLETE"
    assert bn_dcgan_guard.grade(original, ev)["status"] == "INCOMPLETE"

    # Public-API runner discovery and bounded 2-step execution with goal.gif export.
    listed = subprocess.run(
        [sys.executable, "-m", "benchmarks.toy_audit.api_run", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert listed.returncode == 0, listed.stderr[-2000:]
    listed_ids = {line.split("\t", 1)[0] for line in listed.stdout.splitlines() if line.strip()}
    assert TASK_ID in listed_ids

    api_out = tmp_path / "api_run_prefix"
    rc = api_run.main([
        "--case",
        TASK_ID,
        "--steps",
        "2",
        "--eval-samples",
        "32",
        "--frames",
        "2",
        "--recipe",
        "bispf",
        "--device",
        "cpu",
        "--output",
        str(api_out),
    ])
    assert rc == 1
    case_dir = api_out / TASK_ID
    receipt = json.loads((case_dir / "receipt.json").read_text())
    assert receipt["status"] == "COMPLETE"
    assert receipt["completed_updates"] == 2
    assert receipt["default_protocol_complete"] is False
    assert receipt["verdict"] == "FAIL"
    api_gif = case_dir / "goal.gif"
    assert api_gif.is_file()
    assert receipt["artifacts"]["goal.gif"]["sha256"] == api_run.file_hash(api_gif)
    with Image.open(api_gif) as gif:
        assert gif.n_frames == 2


def test_fail_closed_grading_pass_fail_incomplete_invalid_and_cuda_latency_guard(tmp_path):
    task = deepcopy(load_tasks(ROOT)[TASK_ID])

    # 1. CUDA PASS when both quality and B=256 step latency pass.
    cuda_pass_ev = _synthetic_complete_evidence(task, tmp_path, device_type="cuda", duration_ms=18.5, trace_name="cuda_pass.jsonl")
    cuda_pass = grade_result(task, {"evidence": cuda_pass_ev})
    assert cuda_pass["status"] == "PASS"
    assert cuda_pass["qualification_input"] is False
    assert cuda_pass["evaluator_result"]["cuda_latency_passed"] is True
    assert cuda_pass["evaluator_result"]["cuda_latency_gate_status"] == "PASS"
    assert cuda_pass["metrics"]["update_ms_median"] == pytest.approx(18.5)

    # 2. CUDA FAIL on 40+ ms/step O(B^2) spatial-transport latency regression even when quality passes!
    cuda_slow_ev = _synthetic_complete_evidence(task, tmp_path, device_type="cuda", duration_ms=44.5, trace_name="cuda_slow.jsonl")
    cuda_slow = grade_result(task, {"evidence": cuda_slow_ev})
    assert cuda_slow["status"] == "FAIL"
    assert cuda_slow["evaluator_result"]["quality_passed"] is True
    assert cuda_slow["evaluator_result"]["cuda_latency_passed"] is False
    assert cuda_slow["evaluator_result"]["cuda_latency_gate_status"] == "FAIL"

    # 3. CPU run records latency honestly without false-failing the CUDA latency guard.
    cpu_ev = _synthetic_complete_evidence(task, tmp_path, device_type="cpu", duration_ms=2600.0, trace_name="cpu_pass.jsonl")
    cpu_grade = grade_result(task, {"evidence": cpu_ev})
    assert cpu_grade["status"] == "PASS"
    assert cpu_grade["evaluator_result"]["cuda_latency_passed"] is None
    assert cpu_grade["evaluator_result"]["cuda_latency_gate_status"] == "not_applicable_cpu"
    assert cpu_grade["metrics"]["update_ms_median"] == pytest.approx(2600.0)

    # 4. Quality FAIL on terminal window regression (even if final step itself looks good).
    obs_count = int(task["evaluation"]["observations"])
    transient_bad_ev = _synthetic_complete_evidence(
        task, tmp_path, device_type="cuda", duration_ms=18.0, bad_checkpoint_index=obs_count - 2, trace_name="term_bad.jsonl"
    )
    assert grade_result(task, {"evidence": transient_bad_ev})["status"] == "FAIL"

    # 5. Quality FAIL on individual metric threshold violations.
    for overrides in (
        {"feature_fid": 70.0},
        {"pixel_diversity_ratio": 0.45},
        {"pixel_diversity_ratio": 1.45},
        {"class_coverage": 0.70},
        {"class_mass_tv": 0.40},
        {"spectral_l1": 0.08},
        {"finite_fraction": 0.99},
    ):
        bad_ev = _synthetic_complete_evidence(
            task, tmp_path, device_type="cuda", duration_ms=18.0, metric_overrides=overrides, trace_name="q_fail.jsonl"
        )
        assert grade_result(task, {"evidence": bad_ev})["status"] == "FAIL"

    # 6. FAIL if networks did not change or model state is nonfinite.
    for key in ("changed_generator", "changed_discriminator", "finite_model_state"):
        mut = deepcopy(cuda_pass_ev)
        mut["validity"][key] = False
        assert grade_result(task, {"evidence": mut})["status"] == "FAIL"

    # 7. INCOMPLETE on missing contract, GIF, runtime, trace metadata, missing trace file, or truncated trace.
    for missing_key in ("bn_dcgan_guard_contract", "actual_training_gif", "runtime", "artifacts", "validity"):
        mut = deepcopy(cuda_pass_ev)
        del mut[missing_key]
        assert grade_result(task, {"evidence": mut})["status"] == "INCOMPLETE"

    trunc_ev = _synthetic_complete_evidence(
        task, tmp_path, device_type="cuda", duration_ms=18.0, trace_steps=1199, trace_name="trunc.jsonl"
    )
    assert grade_result(task, {"evidence": trunc_ev})["status"] == "INCOMPLETE"

    gone_ev = deepcopy(cuda_pass_ev)
    gone_ev["artifacts"]["updates"]["path"] = str(tmp_path / "does_not_exist.jsonl")
    assert grade_result(task, {"evidence": gone_ev})["status"] == "INCOMPLETE"

    # 8. INVALID on tampered contract, tampered trace bytes, wrong scoring_weights, or RNG deviations.
    bad_contract = deepcopy(cuda_pass_ev)
    bad_contract["bn_dcgan_guard_contract"]["batch_size"] = 64
    assert grade_result(task, {"evidence": bad_contract})["status"] == "INVALID"

    bad_weights = deepcopy(cuda_pass_ev)
    bad_weights["scoring_weights"] = "ema"
    assert grade_result(task, {"evidence": bad_weights})["status"] == "INVALID"

    bad_rng = deepcopy(cuda_pass_ev)
    bad_rng["guards"]["unintended_rng_deviations"] = 1
    assert grade_result(task, {"evidence": bad_rng})["status"] == "INVALID"

    tamper_ev = _synthetic_complete_evidence(task, tmp_path, device_type="cuda", duration_ms=18.0, trace_name="tamper.jsonl")
    with (tmp_path / "tamper.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{}\n")
    assert grade_result(task, {"evidence": tamper_ev})["status"] == "INVALID"


def test_deterministic_reproducibility_and_fresh_process_checkpoint_roundtrip(tmp_path, cpu_single_thread):
    # Two independent B=256 fixtures from seed 0 produce bit-exact identical trajectories and observations.
    fx_a = guard_api.build_case(TASK_ID, device="cpu", seed=0, recipe_name="bispf", max_steps=2)
    fx_b = guard_api.build_case(TASK_ID, device="cpu", seed=0, recipe_name="bispf", max_steps=2)

    fx_a.step(collect_stats=True)
    fx_b.step(collect_stats=True)
    assert fx_a.batch_sequence_sha256 == fx_b.batch_sequence_sha256
    assert state_digest(fx_a.state_dict()) == state_digest(fx_b.state_dict())

    # Save step-1 checkpoint before advancing reference fixture to step 2.
    ckpt_path = tmp_path / "step1_ckpt.pt"
    torch.save(fx_a.state_dict(), ckpt_path)

    # Read-only observation at step 1 must not alter step 2 trajectory.
    obs_a1 = fx_a.observe(n=32, seed=10000)
    assert obs_a1["metrics"]["finite_fraction"] == 1.0
    fx_a.step(collect_stats=True)
    fx_b.step(collect_stats=True)
    ref_state_sha = state_digest(fx_a.state_dict())
    ref_batch_sha = fx_a.batch_sequence_sha256
    assert ref_state_sha == state_digest(fx_b.state_dict())
    assert ref_batch_sha == fx_b.batch_sequence_sha256
    ref_obs2 = fx_a.observe(n=32, seed=10000)
    obs_b2 = fx_b.observe(n=32, seed=10000)
    for k in ("feature_fid", "pixel_diversity_ratio", "class_coverage", "class_mass_tv", "spectral_l1"):
        assert ref_obs2["metrics"][k] == pytest.approx(obs_b2["metrics"][k], abs=1e-6)

    # Fresh-process checkpoint restore and exact continuation from step 1 -> step 2.
    script = (
        "import json, torch\n"
        "from benchmarks.toy_audit import api_bn_dcgan_rgb32 as api\n"
        "from experiments.forge.state import state_digest\n"
        "torch.set_num_threads(1)\n"
        "with torch.autograd.set_multithreading_enabled(False):\n"
        f"    fx = api.build_case({TASK_ID!r}, device='cpu', seed=0, recipe_name='bispf', max_steps=2)\n"
        f"    fx.load_state_dict(torch.load({str(ckpt_path)!r}, map_location='cpu', weights_only=False))\n"
        "    fx.step(collect_stats=True)\n"
        "    obs = fx.observe(n=32, seed=10000)\n"
        "    print(json.dumps({'state_sha': state_digest(fx.state_dict()), 'batch_sha': fx.batch_sequence_sha256, 'fid': obs['metrics']['feature_fid']}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    restored = json.loads(proc.stdout.strip().splitlines()[-1])
    assert restored["state_sha"] == ref_state_sha
    assert restored["batch_sha"] == ref_batch_sha
    assert restored["fid"] == pytest.approx(ref_obs2["metrics"]["feature_fid"], abs=1e-6)

    # Tampered / incompatible checkpoints must be rejected strictly.
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bad_ver = deepcopy(raw_ckpt)
    bad_ver["version"] = "old-v0"
    with pytest.raises(ValueError, match="version"):
        fx_a.load_state_dict(bad_ver)

    bad_recipe = deepcopy(raw_ckpt)
    bad_recipe["recipe"]["lr"] = 999.0
    with pytest.raises(ValueError, match="recipe"):
        fx_a.load_state_dict(bad_recipe)

    missing_field = deepcopy(raw_ckpt)
    del missing_field["data_rng"]
    with pytest.raises(ValueError, match="missing required"):
        fx_a.load_state_dict(missing_field)
