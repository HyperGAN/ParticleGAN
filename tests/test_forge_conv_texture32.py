"""Semantic controls, public-API continuation, and fail-closed grading for img_conv_texture32."""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.toy_audit.api_conv_texture32 import (
    ConvTexture32Fixture,
    build_case,
    conv_texture32_oracle_controls,
    list_cases,
    sample_conv_texture32,
    score_conv_texture32,
)
from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {"recipe_preset": "empf", "claim_contract": {"sampling_law": "task_declared"}}


def _load_task() -> dict:
    return json.loads((ROOT / "configs/forge/tasks/img_conv_texture32.json").read_text())


def test_conv_texture32_preserves_view_chain_and_registers_required_tier1():
    view = load_view(ROOT, "discriminator_stability")
    v17 = json.loads((ROOT / view["previous_revision"]["archive"]).read_text())
    assert view["revision"] == 18 and view["previous_revision"]["revision"] == 17
    assert [a for a in view["assignments"] if a["task"] != "img_conv_texture32"] == v17["assignments"]

    entry = next(a for a in view["assignments"] if a["task"] == "img_conv_texture32")
    assert entry == {
        "task": "img_conv_texture32",
        "qualification_tier": 1,
        "importance": "required",
        "order": 13,
    }
    task = load_tasks(ROOT)["img_conv_texture32"]
    assert task["execution"]["prior"] == {"kind": "mog", "sigma": 0.1, "standardize": False}
    assert task["execution"]["prior_contract"] == "recipe_owned_v1"
    assert task["execution"]["steps"] == 288
    assert task["resources"]["timeout_seconds"] == 300
    assert len(list_cases()) == 1 and list_cases()[0]["id"] == "api-img-conv-texture32"


def test_oracle_200_independent_draws_and_destructive_negative_controls():
    controls = conv_texture32_oracle_controls()
    assert set(controls) == {
        "true_target_oracle",
        "zero_carrier_dc_blob_negative",
        "fixed_phase_memorization_negative",
        "single_orientation_collapse_negative",
        "off_band_white_noise_negative",
        "box_blur_damage_negative",
    }
    for name, item in controls.items():
        assert item["passed"] is item["expected_pass"], f"{name}: {item}"

    # Specific failure selectivity across all failure modes.
    assert "hf_band_ratio >= 0.55" in controls["zero_carrier_dc_blob_negative"]["failed_bounds"]
    assert "clean_texture_fraction >= 0.75" in controls["zero_carrier_dc_blob_negative"]["failed_bounds"]
    assert "orientations == 2" in controls["zero_carrier_dc_blob_negative"]["failed_bounds"]

    # Fixed-phase template memorization passes HF energy + orientations + mass_tv and fails ONLY phase diversity.
    assert controls["fixed_phase_memorization_negative"]["failed_bounds"] == [
        "pairwise_phase_diversity >= 0.4"
    ]
    assert "orientations == 2" in controls["single_orientation_collapse_negative"]["failed_bounds"]
    assert "mass_tv <= 0.18" in controls["single_orientation_collapse_negative"]["failed_bounds"]
    assert "clean_texture_fraction >= 0.75" in controls["off_band_white_noise_negative"]["failed_bounds"]
    assert "hf_band_ratio >= 0.55" in controls["box_blur_damage_negative"]["failed_bounds"]

    # Verify 200/200 independent target draws pass with substantial margin.
    ref = sample_conv_texture32(256, torch.Generator(device="cpu").manual_seed(70000), device="cpu")
    worst_hf = 99.0
    worst_clean = 99.0
    worst_tv = 0.0
    worst_pdiv = 99.0
    for idx in range(200):
        draw = sample_conv_texture32(256, torch.Generator(device="cpu").manual_seed(71000 + idx), device="cpu")
        scored = score_conv_texture32(draw, reference=ref)
        assert scored["passed"] is True, f"oracle draw {idx} failed: {scored}"
        m = scored["metrics"]
        worst_hf = min(worst_hf, m["hf_band_ratio"])
        worst_clean = min(worst_clean, m["clean_texture_fraction"])
        worst_tv = max(worst_tv, m["mass_tv"])
        worst_pdiv = min(worst_pdiv, m["pairwise_phase_diversity"])
    assert worst_hf >= 0.95
    assert worst_clean >= 0.95
    assert worst_tv <= 0.11
    assert worst_pdiv >= 0.85

    # Nonfinite and wrong-shape outputs fail closed without raising.
    bad_nan = score_conv_texture32(torch.full((256, 3, 32, 32), float("nan")), reference=ref)
    assert bad_nan["passed"] is False and "finite_fraction == 1.0" in bad_nan["failed_bounds"]
    bad_shape = score_conv_texture32(torch.zeros((256, 1, 8, 8)), reference=ref)
    assert bad_shape["passed"] is False and "finite_fraction == 1.0" in bad_shape["failed_bounds"]


def test_deliberate_model_damage_flips_passing_generator_to_fail():
    class _OracleGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blur = False

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            gen = torch.Generator(device="cpu").manual_seed(80001)
            out = sample_conv_texture32(len(z), gen, device=z.device)
            if self.blur:
                out = F.avg_pool2d(out, kernel_size=7, stride=1, padding=3)
            return out

    model = _OracleGenerator()
    z = torch.zeros((256, 8))
    healthy = score_conv_texture32(model(z))
    assert healthy["passed"] is True and healthy["failed_bounds"] == []

    model.blur = True
    damaged = score_conv_texture32(model(z))
    assert damaged["passed"] is False
    assert "hf_band_ratio >= 0.55" in damaged["failed_bounds"]
    assert "clean_texture_fraction >= 0.75" in damaged["failed_bounds"]


@pytest.mark.parametrize(
    "bad_prior",
    [
        {"kind": "particle_cloud", "sigma": 0.0, "standardize": False, "exception_reason": "zero-sigma"},
        {"kind": "mog", "sigma": 0.025, "standardize": False},
        {"kind": "mog", "sigma": 0.1, "standardize": True},
        {"kind": "mog", "sigma": 0.1, "standardize": False, "learnable": True},
    ],
)
def test_forbidden_priors_never_run_or_qualify(bad_prior):
    from experiments.forge.conv_texture32 import grade

    value = _load_task()
    value["execution"]["prior"] = bad_prior
    assert adapter_preflight(value, CANDIDATE, root=ROOT)
    assert grade(value, {})[0] == "INVALID"


def test_fresh_gaussian_and_second_order_autograd_penalties_work_cleanly():
    value = _load_task()
    fresh_candidate = {
        "recipe_preset": "empf",
        "recipe_overrides": {"prior_update": "frozen", "prior_code_law": "fresh_gaussian"},
        "claim_contract": {"sampling_law": "task_declared"},
    }
    assert adapter_preflight(value, fresh_candidate, root=ROOT) == []

    # Verify non-inplace MiniResNetCritic32 supports second-order gradient penalties across presets.
    for preset in ("empf", "rhbi_optimized", "bispf_optimized", "bcap_adam", "k3p", "ka2"):
        fix = build_case("api-img-conv-texture32", recipe_name=preset, device="cpu", seed=0, max_steps=4)
        assert isinstance(fix, ConvTexture32Fixture)
        out = fix.step()
        assert out["loss_d"] == out["loss_d"] and out["loss_g"] == out["loss_g"]


@pytest.mark.parametrize("preset", ["empf", "rhbi_optimized"])
def test_checkpoint_continuation_and_software_prefix_grading(tmp_path, preset):
    # 1. Exact checkpoint round-trip across a fresh ConvTexture32Fixture instance.
    full = build_case("api-img-conv-texture32", recipe_name=preset, device="cpu", seed=0, max_steps=6)
    for _ in range(6):
        full.step()
    full_state = state_digest(full.trainer.state_dict())
    full_obs = full.observe(n=64, seed=4242)

    part1 = build_case("api-img-conv-texture32", recipe_name=preset, device="cpu", seed=0, max_steps=6)
    for _ in range(3):
        part1.step()
    ckpt_path = tmp_path / f"{preset}_ckpt.pt"
    torch.save(part1.state_dict(), ckpt_path)

    part2 = build_case("api-img-conv-texture32", recipe_name=preset, device="cpu", seed=0, max_steps=6)
    part2.load_state_dict(torch.load(ckpt_path, weights_only=False))
    for _ in range(3):
        part2.step()
    assert state_digest(part2.trainer.state_dict()) == full_state
    assert part2.observe(n=64, seed=4242)["metrics"] == full_obs["metrics"]

    # 2. Short software prefix executes through Forge run_task, renders GIF, and grades INCOMPLETE against full 288-step task.
    full_task = _load_task()
    prefix_task = deepcopy(full_task)
    prefix_task["id"] = "img_conv_texture32-software-prefix"
    prefix_task["execution"]["steps"] = 2
    prefix_task["execution"]["host_definition"]["steps"] = 2
    prefix_task["evaluation"]["observations"] = 2

    cand = {"recipe_preset": preset, "claim_contract": {"sampling_law": "task_declared"}}
    req = {"candidate": cand, "protocol": {"seed": 0}, "tasks": {prefix_task["id"]: prefix_task}}
    out_dir = tmp_path / f"run_{preset}"
    res = run_task(req, {"task_id": prefix_task["id"]}, out_dir, "cpu")
    assert Path(res["evidence"]["actual_training_gif"]["path"]).is_file()
    # Grading against the full 288-step declaration refuses short-run qualification.
    verdict_full = grade_result(full_task, res)
    assert verdict_full["status"] == "INCOMPLETE"


def test_fresh_process_import_order_invariance():
    code_a = (
        "import json, particlegan, benchmarks.toy_audit.api_conv_texture32 as m; "
        "from experiments.forge.state import state_digest; "
        "f = m.build_case('api-img-conv-texture32', seed=0, max_steps=4); "
        "print(json.dumps({'default': particlegan.DEFAULT_RECIPE, 'g': state_digest(f.trainer.G.state_dict()), 'd': state_digest(f.trainer.D.state_dict())}))"
    )
    code_b = (
        "import json, benchmarks.toy_audit.api_conv_texture32 as m, particlegan; "
        "from experiments.forge.state import state_digest; "
        "f = m.build_case('api-img-conv-texture32', seed=0, max_steps=4); "
        "print(json.dumps({'default': particlegan.DEFAULT_RECIPE, 'g': state_digest(f.trainer.G.state_dict()), 'd': state_digest(f.trainer.D.state_dict())}))"
    )
    env = {"PYTHONPATH": str(ROOT)}
    out_a = json.loads(subprocess.check_output([sys.executable, "-c", code_a], cwd=ROOT, env=env, text=True))
    out_b = json.loads(subprocess.check_output([sys.executable, "-c", code_b], cwd=ROOT, env=env, text=True))
    assert out_a == out_b
    assert out_a["default"] == "empf"


def test_published_cuda_readout_shows_rhbi_failure_and_empf_bispf_pass_with_bound_media():
    import hashlib

    summary = json.loads((ROOT / "reports/forge/conv-texture32-tier1/summary.json").read_text())
    pub = json.loads((ROOT / "reports/toy_audit/api_contract/conv_texture32/publication.json").read_text())

    for path, expected in summary["execution_scientific_sources_sha256"].items():
        original = subprocess.check_output(["git", "show", summary["source_commit"] + ":" + path], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == expected

    by_id = {r["candidate_id"]: r for r in summary["results"]}
    assert set(by_id) == {"rhbi-v12", "empf-v8", "bispf-throughput-v1"}

    # Identical seen batch sequence and initial network/prior weights across all 3 candidates.
    assert len({r["batch_sequence_sha256"] for r in by_id.values()}) == 1
    assert len({json.dumps(r["initial_parameters"], sort_keys=True) for r in by_id.values()}) == 1

    # rhbi-v12 deterministically FAILS with zero spatial carrier energy; empf-v8 & bispf PASS 5/5.
    rhbi = by_id["rhbi-v12"]
    assert rhbi["status"] == "FAIL" and rhbi["terminal_passed"] == 0
    assert rhbi["live"]["hf_band_ratio"] < 0.02
    assert rhbi["live"]["clean_texture_fraction"] == 0.0
    assert rhbi["live"]["orientations"] == 0
    assert rhbi["live"]["pairwise_phase_diversity"] == 0.0

    for winner in ("empf-v8", "bispf-throughput-v1"):
        row = by_id[winner]
        assert row["status"] == "PASS" and row["terminal_passed"] == 5 and row["failed_bounds"] == []
        assert row["live"]["hf_band_ratio"] >= 0.55
        assert row["live"]["clean_texture_fraction"] >= 0.75
        assert row["live"]["orientations"] == 2
        assert row["live"]["mass_tv"] <= 0.18
        assert row["live"]["pairwise_phase_diversity"] >= 0.40

    # Verify all committed GIFs exist and match their recorded SHA-256 digests.
    for row in by_id.values():
        gif_bytes = (ROOT / row["gif"]).read_bytes()
        assert hashlib.sha256(gif_bytes).hexdigest() == row["gif_sha256"]

    api_run = pub["runs"][0]
    assert api_run["candidate_id"] == "rhbi-v12" and api_run["verdict"] == "FAIL"
    api_gif = ROOT / "reports/toy_audit/api_contract/conv_texture32" / api_run["gif"]
    assert hashlib.sha256(api_gif.read_bytes()).hexdigest() == api_run["gif_sha256"]

