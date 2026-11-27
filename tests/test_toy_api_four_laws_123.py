"""Comprehensive oracle, destructive control, GIF export, and public-API tests for Laws 1-3 toys."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image
import pytest
import torch

from benchmarks.toy_audit import api_contract, api_four_laws_123 as fl123, api_run
from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest
from particlegan import GANTrainer, unwrap_prior

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TASKS = (
    "fl_law1_aniso_unequal_modes",
    "fl_law1_noisy_atomic_codebook",
    "fl_law2_pure_adversarial_codim1",
    "fl_law2_groupnorm_and_vit_sprites",
    "fl_law3_highd_orphan_d32",
)


def test_interface_contract_exports_and_schema():
    assert fl123.TASKS == EXPECTED_TASKS
    assert fl123.HOST_DEFINITIONS is fl123.SPECS
    assert fl123.Fixture is fl123.FourLawsFixture
    assert set(fl123.SPECS) == set(EXPECTED_TASKS)
    assert set(fl123.THRESHOLDS) == set(EXPECTED_TASKS)

    # All specs must be strictly JSON-serializable with zero NaN/Inf.
    encoded = json.dumps(fl123.SPECS, allow_nan=False)
    assert len(encoded) > 100

    required_keys = {
        "task_id",
        "case_id",
        "kind",
        "law",
        "defects",
        "title",
        "goal",
        "target",
        "resources",
        "prior",
        "model",
        "steps",
        "eval_samples",
        "observations",
        "minimum_stable_checks",
        "sampling_law",
        "view_kind",
    }
    for tid in EXPECTED_TASKS:
        spec = fl123.SPECS[tid]
        assert required_keys <= set(spec)
        assert spec["task_id"] == spec["case_id"] == spec["kind"] == tid
        assert spec["law"] in {"law1", "law2", "law3"}
        assert isinstance(spec["defects"], list) and spec["defects"]
        assert spec["observations"] == 24
        assert spec["minimum_stable_checks"] == 5
        assert spec["steps"] >= 24
        assert spec["view_kind"] in {"vector", "image"}
        assert spec["sampling_law"] in {
            "public_prior_without_output_noise",
            "enumerated_prior_without_output_noise",
        }
        prior = spec["prior"]
        assert "learnable" not in prior
        assert prior["kind"] in {"mog", "particle_cloud"}
        if prior["kind"] == "particle_cloud" and float(prior["sigma"]) == 0.0:
            assert isinstance(prior.get("exception_reason"), str) and prior["exception_reason"].strip()

        thresholds = fl123.THRESHOLDS[tid]
        assert thresholds[0] == ["finite_fraction", "==", 1.0]
        for name, op, bound in thresholds:
            assert isinstance(name, str) and name
            assert op in {"<=", ">=", "=="}
            assert isinstance(bound, (int, float)) and not isinstance(bound, bool) and math.isfinite(float(bound))

    cases = fl123.list_cases()
    assert [c["id"] for c in cases] == list(EXPECTED_TASKS)
    for case in cases:
        validated = api_contract.validate_case(case)
        assert validated["default_recipe"] == "unified_v14"
        assert validated["protocol_seed"] == 0
        assert validated["legacy_ids"] == [f"four-laws-{case['id']}"]


def test_thresholds_are_strictly_method_agnostic_output_gates():
    """Verify THRESHOLDS grade observable outputs (plus explicit critic_grad_finite gate on ViT sprites)."""
    forbidden_substrings = (
        "stiff",
        "s_th",
        "peak",
        "mean_dz",
        "r_stable",
        "band_scale",
        "step_ms",
        "d_max_s1",
        "prior_",
    )
    allowed_grad_gates = {
        "fl_law2_groupnorm_and_vit_sprites": {"critic_grad_finite"},
        "fl_law2_groupnorm_and_vit_sprites_v2": {"critic_grad_finite"},
    }
    for tid in fl123.ALL_TASKS:
        gate_names = [row[0] for row in fl123.ALL_THRESHOLDS[tid]]
        for name in gate_names:
            lower = name.lower()
            assert not any(tok in lower for tok in forbidden_substrings), (
                f"{tid} threshold {name!r} must not gate on internal/runtime mechanics"
            )
            if "grad" in lower:
                assert name in allowed_grad_gates.get(tid, set()), (
                    f"{tid} threshold {name!r} unexpectedly gates on gradient telemetry"
                )
    assert ["critic_grad_finite", "==", 1.0] in fl123.THRESHOLDS["fl_law2_groupnorm_and_vit_sprites"]
    assert ["critic_grad_finite", "==", 1.0] in fl123.V2_THRESHOLDS["fl_law2_groupnorm_and_vit_sprites_v2"]


def test_signed_margins_semantics_and_fail_closed():
    tid = "fl_law1_aniso_unequal_modes"
    passing = {
        "finite_fraction": 1.0,
        "mode_recall": 1.0,
        "precision": 0.99,
        "mass_tv": 0.02,
        "min_cov_eig_ratio": 0.70,
        "max_cov_eig_ratio": 1.30,
    }
    m_pass = fl123.signed_margins(passing, tid)
    assert set(m_pass) == {row[0] for row in fl123.THRESHOLDS[tid]}
    assert all(math.isfinite(v) and v >= 0.0 for v in m_pass.values())
    assert m_pass["finite_fraction"] == 0.0
    assert m_pass["precision"] > 0.0
    assert m_pass["mass_tv"] > 0.0

    failing = dict(passing, precision=0.50, mass_tv=float("nan"), finite_fraction=0.5)
    m_fail = fl123.signed_margins(failing, tid)
    assert m_fail["precision"] < 0.0
    assert m_fail["mass_tv"] == -10.0
    assert m_fail["finite_fraction"] < 0.0
    assert all(math.isfinite(v) for v in m_fail.values())


def test_oracle_controls_fast_and_all_match_expected():
    t0 = time.perf_counter()
    controls = fl123.oracle_controls()
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"oracle_controls took {elapsed:.2f}s (expected fast CPU)"
    assert len(controls) >= 15

    for tid in EXPECTED_TASKS:
        task_ctrls = {k: v for k, v in controls.items() if k.startswith(f"{tid}_")}
        pos = [v for v in task_ctrls.values() if v["expected_pass"]]
        neg = [v for v in task_ctrls.values() if not v["expected_pass"]]
        assert len(pos) >= 1, f"{tid} missing positive oracle control"
        assert len(neg) >= 2, f"{tid} missing destructive negative controls"

    for name, entry in controls.items():
        assert entry["passed"] is entry["expected_pass"], f"{name}: {entry}"
        assert math.isfinite(entry["min_margin"])
        if entry["expected_pass"]:
            assert entry["min_margin"] >= 0.0
            assert entry["failed_bounds"] == []
        else:
            assert entry["min_margin"] < 0.0
            assert len(entry["failed_bounds"]) >= 1


@pytest.mark.parametrize("task_id", EXPECTED_TASKS)
def test_oracle_pass_witness_positive_inequality_margins(task_id: str):
    rng = torch.Generator(device="cpu").manual_seed(77001)
    spec = fl123.SPECS[task_id]
    n = int(spec["eval_samples"])
    oracle = fl123.sample_target_for_kind(task_id, n, rng)
    ref = fl123.sample_target_for_kind(task_id, min(n, 512), rng)

    extra: dict = {}
    if task_id == "fl_law2_pure_adversarial_codim1":
        extra["orbit_trace"] = [[float(k * 10), 0.001 * math.sin(0.2 * k)] for k in range(30)]
        extra["normal_grad_norm"] = 0.25
        extra["normal_grad_max"] = 0.45
    elif task_id == "fl_law2_groupnorm_and_vit_sprites":
        extra["critic_grad_finite"] = 1.0
        extra["critic_grad_med"] = 0.25
        extra["critic_grad_max"] = 0.50
        extra["mean_step_ms"] = 25.0

    obs = fl123.score_task(task_id, oracle, reference=ref, **extra)
    validated = api_contract.validate_observation(obs)
    assert validated["passed"] is True
    assert validated["failed_bounds"] == []
    assert validated["min_margin"] >= 0.0
    assert validated["metrics"]["min_margin"] == validated["min_margin"]

    # Every strict inequality gate must have strictly positive margin on oracle draws.
    for metric_name, op, _ in fl123.THRESHOLDS[task_id]:
        margin = validated["margins"][metric_name]
        assert math.isfinite(margin)
        if op in {"<=", ">="}:
            assert margin > 0.0 or (metric_name in {"mode_recall", "mode_coverage"} and margin == 0.0), (
                f"{task_id}.{metric_name}: expected positive margin, got {margin}"
            )
        else:
            assert margin == 0.0

    json.dumps(validated["metrics"], allow_nan=False)
    json.dumps(validated["margins"], allow_nan=False)


def test_destructive_negative_controls_all_five_tasks():
    rng = torch.Generator(device="cpu").manual_seed(88001)

    # --- Task 1: fl_law1_aniso_unequal_modes ---
    w1, m1, _, _ = fl123.aniso_ellipse_parameters("cpu")
    # 1a. Isotropic collapse (sigma=0.25 in both axes instead of 10:1)
    idx1 = torch.multinomial(w1, 4096, replacement=True, generator=rng)
    iso1 = m1[idx1] + 0.25 * torch.randn((4096, 2), generator=rng)
    r1_iso = api_contract.validate_observation(fl123.score_fl_law1_aniso_unequal_modes(iso1))
    assert not r1_iso["passed"] and r1_iso["min_margin"] < 0.0
    assert any("max_cov_eig_ratio" in b or "precision" in b for b in r1_iso["failed_bounds"])

    # 1b. Uniform-mass impostor (breaks 4:1 skew)
    unif_idx1 = torch.randint(0, 8, (4096,), generator=rng)
    _, _, covs1, _ = fl123.aniso_ellipse_parameters("cpu")
    chol1 = torch.linalg.cholesky(covs1.double()).float()
    eps1 = torch.randn((4096, 2), generator=rng)
    unif1 = m1[unif_idx1] + torch.bmm(chol1[unif_idx1], eps1.unsqueeze(-1)).squeeze(-1)
    r1_unif = api_contract.validate_observation(fl123.score_fl_law1_aniso_unequal_modes(unif1))
    assert not r1_unif["passed"] and "mass_tv <= 0.14" in r1_unif["failed_bounds"]

    # 1c. Shifted/exploded + NaN
    r1_shift = api_contract.validate_observation(
        fl123.score_fl_law1_aniso_unequal_modes(unif1 + 10.0)
    )
    assert not r1_shift["passed"] and "precision >= 0.85" in r1_shift["failed_bounds"]
    nan1 = unif1.clone()
    nan1[0, 1] = float("nan")
    r1_nan = api_contract.validate_observation(fl123.score_fl_law1_aniso_unequal_modes(nan1))
    assert not r1_nan["passed"] and "finite_fraction == 1.0" in r1_nan["failed_bounds"]

    # --- Task 2: fl_law1_noisy_atomic_codebook ---
    t2 = "fl_law1_noisy_atomic_codebook"
    _, c2, s2, _ = fl123.noisy_codebook_parameters("cpu")
    # 2a. Mode-collapsed (only 32 of 64 atoms)
    half_idx = torch.randint(0, 32, (4096,), generator=rng)
    half2 = c2[half_idx] + s2 * torch.randn((4096, 2), generator=rng)
    r2_half = api_contract.validate_observation(fl123.score_fl_law1_noisy_atomic_codebook(half2))
    assert not r2_half["passed"] and "atom_recall >= 0.96" in r2_half["failed_bounds"]

    # 2b. Shifted/exploded + Inf
    ora2 = fl123.sample_target_for_kind(t2, 4096, rng)
    r2_shift = api_contract.validate_observation(fl123.score_fl_law1_noisy_atomic_codebook(ora2 + 5.0))
    assert not r2_shift["passed"] and "outlier_rate <= 0.05" in r2_shift["failed_bounds"]
    inf2 = ora2.clone()
    inf2[2, 0] = float("inf")
    r2_inf = api_contract.validate_observation(fl123.score_fl_law1_noisy_atomic_codebook(inf2))
    assert not r2_inf["passed"] and "finite_fraction == 1.0" in r2_inf["failed_bounds"]

    # --- Task 3: fl_law2_pure_adversarial_codim1 ---
    t3 = "fl_law2_pure_adversarial_codim1"
    ora3 = fl123.sample_target_for_kind(t3, 2048, rng)
    steady = [[float(k * 10), 0.0] for k in range(30)]
    osc = [[float(k * 10), 0.20 * math.sin(0.8 * k)] for k in range(30)]
    r3_osc = api_contract.validate_observation(
        fl123.score_fl_law2_pure_adversarial_codim1(ora3, orbit_trace=osc, normal_grad_norm=0.2)
    )
    assert not r3_osc["passed"]
    assert "normal_orbit_r_tail <= 0.06" in r3_osc["failed_bounds"]
    assert "normal_orbit_ptp <= 0.12" in r3_osc["failed_bounds"]

    r3_shift = api_contract.validate_observation(
        fl123.score_fl_law2_pure_adversarial_codim1(ora3 + 2.0, orbit_trace=steady, normal_grad_norm=0.2)
    )
    assert not r3_shift["passed"] and "manifold_mae <= 0.06" in r3_shift["failed_bounds"]

    # High critic gradient norm alone does NOT fail problem-only output gates when generated manifold is exact.
    r3_diag = api_contract.validate_observation(
        fl123.score_fl_law2_pure_adversarial_codim1(ora3, orbit_trace=steady, normal_grad_norm=4.2)
    )
    assert r3_diag["passed"] is True and r3_diag["metrics"]["normal_grad_norm"] == 4.2

    # --- Task 4: fl_law2_groupnorm_and_vit_sprites ---
    t4 = "fl_law2_groupnorm_and_vit_sprites"
    tpl4 = fl123.sprite_templates("cpu")
    ora4 = fl123.sample_target_for_kind(t4, 256, rng)
    # 4a. Blurry average-of-all-templates ghost (fails sprite_hq & template_rmse)
    avg_ghost = tpl4.mean(dim=0, keepdim=True).expand(256, -1, -1, -1).clone()
    r4_blur = api_contract.validate_observation(
        fl123.score_fl_law2_groupnorm_and_vit_sprites(avg_ghost, reference=tpl4)
    )
    assert not r4_blur["passed"]
    assert "sprite_hq >= 0.85" in r4_blur["failed_bounds"]
    assert "template_rmse <= 0.12" in r4_blur["failed_bounds"]

    # 4b. Severely skewed class mass (85% class 0, 5% each of 1..3 -> fails mass_tv)
    skew4 = torch.cat(
        (
            tpl4[0:1].expand(220, -1, -1, -1),
            tpl4[1:2].expand(12, -1, -1, -1),
            tpl4[2:3].expand(12, -1, -1, -1),
            tpl4[3:4].expand(12, -1, -1, -1),
        ),
        dim=0,
    )
    r4_skew = api_contract.validate_observation(
        fl123.score_fl_law2_groupnorm_and_vit_sprites(skew4, reference=tpl4)
    )
    assert not r4_skew["passed"] and "mass_tv <= 0.14" in r4_skew["failed_bounds"]

    # critic_grad_finite == 1.0 MUST gate PASS; mean_step_ms remains non-gating telemetry.
    r4_bad_grad = api_contract.validate_observation(
        fl123.score_fl_law2_groupnorm_and_vit_sprites(ora4, critic_grad_finite=0.0, mean_step_ms=30.0, reference=tpl4)
    )
    assert r4_bad_grad["passed"] is False
    assert "critic_grad_finite == 1.0" in r4_bad_grad["failed_bounds"]

    r4_slow = api_contract.validate_observation(
        fl123.score_fl_law2_groupnorm_and_vit_sprites(ora4, critic_grad_finite=1.0, mean_step_ms=9999.0, reference=tpl4)
    )
    assert r4_slow["passed"] is True and r4_slow["metrics"]["mean_step_ms"] == 9999.0

    # --- Task 5: fl_law3_highd_orphan_d32 ---
    w5, m5, s5, q5 = fl123.highd_orphan_parameters("cpu")
    # 5a. Rank-deficient 8D subspace collapse inside R^32 (passes 2D projection visually yet fails full 32D core_eig_min!)
    idx5 = torch.multinomial(w5, 6144, replacement=True, generator=rng)
    z_lowrank = torch.zeros((6144, 32), dtype=torch.float32)
    z_lowrank[:, :8] = torch.randn((6144, 8), generator=rng)
    pts_lowrank = m5[idx5] + s5[idx5].unsqueeze(1) * (z_lowrank @ q5.T)
    r5_rank = api_contract.validate_observation(fl123.score_fl_law3_highd_orphan_d32(pts_lowrank))
    assert not r5_rank["passed"] and "core_eig_min >= 0.25" in r5_rank["failed_bounds"]

    # 5b. Missing orphan cluster
    w_no = w5[:4] / w5[:4].sum()
    idx_no = torch.multinomial(w_no, 6144, replacement=True, generator=rng)
    pts_no = m5[idx_no] + s5[idx_no].unsqueeze(1) * torch.randn((6144, 32), generator=rng)
    r5_no = api_contract.validate_observation(fl123.score_fl_law3_highd_orphan_d32(pts_no))
    assert not r5_no["passed"]
    assert "modes == 5" in r5_no["failed_bounds"]
    assert "orphan_recall >= 0.5" in r5_no["failed_bounds"]


@pytest.mark.parametrize("task_id", EXPECTED_TASKS)
def test_bounded_public_api_smoke_and_gif_export(task_id: str, tmp_path: Path, monkeypatch):
    case = next(c for c in fl123.list_cases() if c["id"] == task_id)
    built: list[fl123.FourLawsFixture] = []

    def _build_hook(c, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe_overrides=None, recipe=None):
        fix = fl123.build_case(
            c["id"],
            device=device,
            seed=seed,
            recipe_name=recipe_name,
            max_steps=max_steps,
            recipe_overrides=recipe_overrides,
            recipe=recipe,
        )
        built.append(fix)
        return fix

    monkeypatch.setattr(api_run.contract, "build", _build_hook)
    out_dir = tmp_path / task_id
    eval_n = 32 if task_id == "fl_law2_groupnorm_and_vit_sprites" else 128
    receipt = api_run.run_case(
        case,
        out_dir,
        device="cpu",
        recipe_name="unified_v14",
        steps=8,
        eval_samples=eval_n,
        frames=3,
        seed=0,
    )

    assert len(built) == 1
    fixture = built[0]
    assert isinstance(fixture.trainer, GANTrainer)
    assert fixture.aux_trainers == {}
    assert fixture.completed_steps == 8

    assert receipt["status"] == "COMPLETE"
    assert receipt["completed_updates"] == 8
    assert receipt["default_protocol_complete"] is False
    assert receipt["verdict"] == "FAIL"
    assert receipt["gif_frames"] == 3
    assert len(receipt["observations"]) >= 3

    for obs in receipt["observations"]:
        assert "margins" in obs and "min_margin" in obs
        assert math.isfinite(obs["min_margin"])
        assert all(math.isfinite(v) for v in obs["margins"].values())

    gif_path = out_dir / "goal.gif"
    assert gif_path.is_file() and gif_path.stat().st_size > 1000
    with Image.open(gif_path) as img:
        assert img.format == "GIF"
        assert img.n_frames == 3

    receipt_path = out_dir / "receipt.json"
    saved = json.loads(receipt_path.read_text())
    assert saved["artifacts"]["goal.gif"]["sha256"] == api_run.file_hash(gif_path)
    assert saved["reproducibility"]["batch_sequence_sha256"] == fixture.batch_sequence_sha256


def test_checkpoint_exact_continuation_and_render_task_gif(tmp_path: Path):
    torch.set_num_threads(1)
    # Test codim1 (which also checkpoints orbit_trace!) for exact bit-for-bit resume.
    tid = "fl_law2_pure_adversarial_codim1"
    f_direct = fl123.build_case(tid, device="cpu", seed=0, recipe_name="unified_v14", max_steps=10)
    obs_step0 = f_direct.observe(n=128, seed=10000)
    ckpt = None
    for s in range(1, 9):
        f_direct.step()
        if s == 4:
            ckpt = f_direct.state_dict()
    assert ckpt is not None

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(ckpt, ckpt_path)
    loaded = torch.load(ckpt_path, weights_only=True)

    f_resumed = fl123.build_case(tid, device="cpu", seed=99, recipe_name="unified_v14", max_steps=10)
    f_resumed.load_state_dict(loaded)
    for _ in range(4):
        f_resumed.step()

    assert f_direct.completed_steps == f_resumed.completed_steps == 8
    assert f_direct.batch_sequence_sha256 == f_resumed.batch_sequence_sha256
    assert torch.equal(unwrap_prior(f_direct.trainer.prior).z, unwrap_prior(f_resumed.trainer.prior).z)
    assert state_digest(f_direct.trainer.G.state_dict()) == state_digest(f_resumed.trainer.G.state_dict())
    assert state_digest(f_direct.trainer.D.state_dict()) == state_digest(f_resumed.trainer.D.state_dict())
    assert f_direct._orbit_trace == f_resumed._orbit_trace

    obs_a = f_direct.observe(n=128, seed=10000)
    obs_b = f_resumed.observe(n=128, seed=10000)
    assert obs_a["metrics"] == obs_b["metrics"]

    # Verify thin render_task_gif wrapper works directly on distinct observations.
    gif_out = tmp_path / "direct.gif"
    rec0 = {**obs_step0, "step": 0}
    rec8 = {**obs_a, "step": 8}
    fl123.render_task_gif(tid, [rec0, rec8], gif_out, full_budget=False, requested_steps=8, final_verdict="FAIL")
    assert gif_out.is_file() and gif_out.stat().st_size > 1000
    with Image.open(gif_out) as img:
        assert img.n_frames == 2


def test_forge_components_context_zero_unintended_rng_deviations():
    torch.set_num_threads(1)
    tid = "fl_law1_aniso_unequal_modes"
    spec = fl123.SPECS[tid]
    ctx = FormulationContext(
        recipe_preset="unified_v14",
        recipe_overrides={**spec["resources"]},
        prior=spec["prior"],
        seed=0,
        device="cpu",
        prior_contract="recipe_owned_v1",
    )
    fix = fl123.build_case(tid, device="cpu", seed=0, max_steps=8, components=ctx)
    assert fix.recipe == ctx.recipe
    for _ in range(3):
        fix.step()

    before = ctx.streams.audit()
    global_before = torch.get_rng_state().clone()
    obs = fix.observe(n=128)
    validated = api_contract.validate_observation(obs)
    assert validated["metrics"]["finite_fraction"] == 1.0

    allowed = [
        key
        for key, binding in ctx.streams.manifest()["bindings"].items()
        if binding["family"] == "eval"
    ]
    audit = ctx.streams.compare(before, ctx.streams.audit(), allowed=allowed)
    assert audit["unintended_rng_deviations"] == 0
    assert torch.equal(global_before, torch.get_rng_state())


@pytest.mark.parametrize("recipe_name", ["minimal", "axiom", "unified_v15", "bispf", "empf"])
@pytest.mark.parametrize("task_id", EXPECTED_TASKS)
def test_non_v14_recipes_run_cleanly_on_all_five_tasks(recipe_name: str, task_id: str):
    """Confirm 1st-order recipes pass all 5 tasks (incl. real SDPA ViT) while 2nd-order RoR exposes SDPA double-backward."""
    torch.set_num_threads(1)
    fix = fl123.build_case(task_id, device="cpu", seed=0, recipe_name=recipe_name, max_steps=4)
    if task_id == "fl_law2_groupnorm_and_vit_sprites" and recipe_name in ("bispf", "empf"):
        # Genuine Law 2 / L2-5 finding: 2nd-order create_graph=True RoR cannot differentiate through
        # real F.scaled_dot_product_attention flash/mem-efficient backward, whereas 1st-order recipes survive.
        with pytest.raises(RuntimeError, match="scaled_dot_product"):
            fix.step()
        return
    for _ in range(2):
        fix.step()
    eval_n = 32 if task_id == "fl_law2_groupnorm_and_vit_sprites" else 64
    obs = api_contract.validate_observation(fix.observe(n=eval_n, seed=10000))
    assert obs["metrics"]["finite_fraction"] == 1.0
    assert math.isfinite(obs["min_margin"])


def test_fresh_process_import_order_independence():
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"}
    cmd_a = (
        "import particlegan, benchmarks.toy_audit.api_four_laws_123 as m; "
        "assert all(v['passed'] is v['expected_pass'] for v in m.oracle_controls().values())"
    )
    cmd_b = (
        "import benchmarks.toy_audit.api_four_laws_123 as m, particlegan; "
        "assert all(v['passed'] is v['expected_pass'] for v in m.oracle_controls().values())"
    )
    subprocess.run([sys.executable, "-c", cmd_a], check=True, env=env)
    subprocess.run([sys.executable, "-c", cmd_b], check=True, env=env)
