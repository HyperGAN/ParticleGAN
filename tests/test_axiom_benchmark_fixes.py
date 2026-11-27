"""Regression & semantic-control tests for Track A1-A3 benchmark/scorer/v2-probe fixes."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit import api_four_laws_123 as fl123
from benchmarks.toy_audit import api_four_laws_4 as fl4
from benchmarks.toy_audit import api_interpolation as interp
from benchmarks.toy_audit import api_model_ae_ddgan_v2 as ae_ddgan_v2
from benchmarks.toy_audit import api_model_ddgan_v2 as ddgan_v2
from benchmarks.toy_audit import api_t3_isolation as t3_iso
from experiments.forge import four_laws as fl_adapter
from experiments.forge import model_toy_v2_adapter as v2_adapter
from experiments.forge.adapters import adapter_preflight
from experiments.forge.mechanisms import NAMES as MECHANISM_NAMES
from experiments.forge.priors import RECIPE_PRIOR_CONTRACT
from experiments.forge.sampling import MODEL_TOY_CLEAN, executed_receipt
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view, validate_view
from particlegan import Recipe, get_recipe

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_BISPF = json.loads((ROOT / "configs/forge/ideas/bispf-optimized-quintic-v1.json").read_text())
CANDIDATE_V14 = json.loads((ROOT / "configs/forge/ideas/unified-v14.json").read_text())


# ---------------------------------------------------------------------------
# #1 — CUDA FP32 exact finiteness & recalibrated interpolation tasks
# ---------------------------------------------------------------------------


def test_interpolation_exact_finiteness_avoids_fp32_ulp_collapse_for_n_82_and_94():
    """Exact `.all()` finiteness must never false-fail when N in {82, 94} (CUDA 1-ULP mean hazard)."""
    # Verify across CPU and CUDA (when available) that combined sizes 82 and 94 yield finite_fraction == 1.0.
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        # 1) score_img_patch_shift: 16 endpoints + (26 chords * 3 t) = 94 combined points!
        # Also test 16 endpoints + (22 chords * 3 t) = 82 combined points!
        for k_neighbors, expected_min_combined in ((2, 64), (3, 82)):
            coords = torch.linspace(1.7, 6.3, 16, device=dev)
            z = torch.stack(
                [
                    coords,
                    torch.linspace(1.7, 6.3, 16, device=dev),
                    torch.zeros(16, device=dev),
                    torch.zeros(16, device=dev),
                ],
                dim=-1,
            )
            centers = torch.stack(
                [
                    torch.linspace(1.8, 6.2, 16, device=dev),
                    torch.linspace(1.8, 6.2, 16, device=dev),
                ],
                dim=-1,
            )

            def _gen_patch(z_in: torch.Tensor) -> torch.Tensor:
                u = z_in[:, 0].clamp(1.5, 6.5)
                v = z_in[:, 1].clamp(1.5, 6.5)
                return interp.render_patch_images(torch.stack([u, v], dim=-1))

            end_imgs = interp.render_patch_images(centers)
            res = interp.score_img_patch_shift(z, end_imgs, _gen_patch, k_neighbors=k_neighbors)
            assert res["metrics"]["finite_fraction"] == 1.0
            assert "finite_fraction == 1.0" not in res["failed_bounds"]
            assert res["metrics"]["interp_peak"] > 0.5
            assert res["metrics"]["quadrant_coverage"] >= 1
            assert expected_min_combined > 0

        # Direct N=82 and N=94 tensor checks through score_sparse_arc and score_saddle_barycentric:
        # 16 endpoints on 270-deg arc with k=3 produces ~16 + 78 = 94 combined points.
        angles = torch.linspace(-math.pi / 4, 5 * math.pi / 4, 16, device=dev)
        z_arc = torch.stack([angles, torch.zeros_like(angles)], dim=-1)
        x_arc = torch.stack([2.0 * torch.cos(angles), 2.0 * torch.sin(angles)], dim=-1)

        def _gen_arc(z_in: torch.Tensor) -> torch.Tensor:
            th = z_in[:, 0]
            return torch.stack([2.0 * torch.cos(th), 2.0 * torch.sin(th)], dim=-1)

        arc_res = interp.score_sparse_arc(z_arc, x_arc, _gen_arc, k_neighbors=3)
        assert arc_res["metrics"]["finite_fraction"] == 1.0
        assert arc_res["passed"] is True

        # Directly construct exact combined sizes N=82 and N=94 inside score_img_patch_shift
        # (which previously triggered 1-ULP CUDA FP32 `float(ones(N).mean().item()) == 0.99999994 < 1.0`!).
        # Two separated 1D lines of lengths (a, b) with k=2 have (a+1)+(b+1) = M+2 undirected edges,
        # so len(combined) = M + 3*(M+2) = 4*M + 6:
        # - M=19 (lines 10 + 9) -> 21 edges -> 19 + 63 = 82 combined tensors!
        # - M=22 (lines 11 + 11) -> 24 edges -> 22 + 72 = 94 combined tensors!
        l10 = torch.linspace(0.0, 0.9, 10, device=dev)
        l9 = torch.linspace(0.0, 0.8, 9, device=dev)
        z_82 = torch.cat(
            [
                torch.stack([l10, torch.zeros_like(l10), torch.zeros_like(l10), torch.zeros_like(l10)], dim=-1),
                torch.stack([l9, torch.full_like(l9, 10.0), torch.zeros_like(l9), torch.zeros_like(l9)], dim=-1),
            ],
            dim=0,
        )
        l11 = torch.linspace(0.0, 1.0, 11, device=dev)
        z_94 = torch.cat(
            [
                torch.stack([l11, torch.zeros_like(l11), torch.zeros_like(l11), torch.zeros_like(l11)], dim=-1),
                torch.stack([l11, torch.full_like(l11, 10.0), torch.zeros_like(l11), torch.zeros_like(l11)], dim=-1),
            ],
            dim=0,
        )
        for n_hazard, z_exact in ((82, z_82), (94, z_94)):
            interp_calls = []

            def _map_uv(zi: torch.Tensor) -> torch.Tensor:
                interp_calls.append(len(zi))
                u = 2.0 + 3.5 * torch.sigmoid(zi[:, 0])
                v = 2.0 + 3.5 * torch.sigmoid(zi[:, 1] - 5.0)
                return interp.render_patch_images(torch.stack([u, v], dim=-1))

            end_exact = _map_uv(z_exact)
            interp_calls.clear()
            res_exact = interp.score_img_patch_shift(z_exact, end_exact, _map_uv, k_neighbors=2)
            assert len(z_exact) + interp_calls[0] == n_hazard
            assert res_exact["metrics"]["finite_fraction"] == 1.0
            assert "finite_fraction == 1.0" not in res_exact["failed_bounds"]
            assert res_exact["metrics"]["interp_peak"] > 0.78

        # Saddle on 4x8 = 32 grid points on [-1, 1]^2
        gx, gy = torch.meshgrid(
            torch.linspace(-0.9, 0.9, 4, device=dev),
            torch.linspace(-0.9, 0.9, 8, device=dev),
            indexing="ij",
        )
        z_sad = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

        def _gen_sad(z_in: torch.Tensor) -> torch.Tensor:
            x, y = z_in[:, 0], z_in[:, 1]
            return torch.stack([x, y, interp.saddle_height(x, y)], dim=-1)

        sad_res = interp.score_saddle_barycentric(z_sad, _gen_sad(z_sad), _gen_sad)
        assert sad_res["metrics"]["finite_fraction"] == 1.0
        assert sad_res["passed"] is True

        # Non-finite outputs MUST still fail closed across all 3 scorers!
        bad_arc = interp.score_sparse_arc(z_arc, x_arc, lambda zi: torch.full((len(zi), 2), float("nan"), device=dev))
        assert bad_arc["passed"] is False and "finite_fraction == 1.0" in bad_arc["failed_bounds"]
        bad_patch = interp.score_img_patch_shift(
            z, end_imgs, lambda zi: torch.full((len(zi), 1, 8, 8), float("inf"), device=dev)
        )
        assert bad_patch["passed"] is False and "finite_fraction == 1.0" in bad_patch["failed_bounds"]
        bad_sad = interp.score_saddle_barycentric(
            z_sad, _gen_sad(z_sad), lambda zi: torch.full((len(zi), 3), float("nan"), device=dev)
        )
        assert bad_sad["passed"] is False and "finite_fraction == 1.0" in bad_sad["failed_bounds"]


def test_recalibrated_interpolation_tasks_and_oracle_selectivity():
    tasks = load_tasks(ROOT)
    arc = tasks["interp_sparse_arc"]
    sad = tasks["interp_saddle_barycentric"]
    assert arc["execution"]["steps"] == 900
    assert arc["execution"]["host_definition"]["resources"]["num_particles"] == 16
    assert sad["execution"]["steps"] == 1000
    assert sad["execution"]["host_definition"]["resources"]["num_particles"] == 32
    assert ["barycenter_hq", ">=", 0.72] in sad["evaluation"]["thresholds"]
    for name in ("interp_sparse_arc", "interp_img_patch_shift", "interp_saddle_barycentric"):
        assert adapter_preflight(tasks[name], CANDIDATE_V14, root=ROOT) == []

    controls = interp.interpolation_oracle_controls()
    for name, item in controls.items():
        assert item["passed"] is item["expected_pass"], f"{name}: {item}"
    assert any("barycenter_hq" in b for b in controls["saddle_linear_simplex_negative"]["failed_bounds"])


# ---------------------------------------------------------------------------
# #3 — Clean oracle-free v2 DDGAN/AE-DDGAN, terminal_observations=1, timeouts, t3, grid100_10k
# ---------------------------------------------------------------------------


class _PoisonTargetProxy:
    """Allows ONLY `.sample(n, generator)`; fails immediately if training reads ground-truth target params."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def sample(self, n, generator):
        return self._wrapped.sample(n, generator)

    def __getattr__(self, name):
        raise AssertionError(f"Oracle leakage detected: training accessed ground-truth target.{name}!")


@pytest.mark.parametrize(
    "module,case_id",
    [
        (ddgan_v2, "api-model-ddgan-gaussian-chain-v2"),
        (ddgan_v2, "api-model-ddgan-bimodal-chain-v2"),
        (ae_ddgan_v2, "api-model-ae-ddgan-gaussian-chain-v2"),
        (ae_ddgan_v2, "api-model-ae-ddgan-bimodal-chain-v2"),
    ],
)
def test_v2_ddgan_and_ae_ddgan_are_strictly_oracle_free_and_checkpoint_exact(module, case_id):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture = module.build_case(case_id, device="cpu", seed=0, max_steps=6)
        assert fixture.oracle_free is True
        assert fixture.lambda_div == 0.0
        assert fixture.cond_post_std is None
        if module is ae_ddgan_v2:
            assert torch.equal(fixture.whitening, torch.eye(2, dtype=torch.float32))
            assert fixture.reconstruction_scale == 1.0

        # Replace target with poison proxy to prove ZERO reads of means/variances/weights during step().
        real_target = fixture.target
        fixture.target = _PoisonTargetProxy(real_target)
        for _ in range(3):
            out = fixture.step()
            assert out["div_loss"] == 0.0
            if module is ae_ddgan_v2:
                assert out["tether_loss"] == 0.0
        fixture.target = real_target

        # Checkpoint exact continuation across step 3 -> step 6.
        saved = deepcopy(fixture.state_dict())
        for _ in range(3):
            fixture.step()
        direct_obs = fixture.observe(n=64, seed=10000)

        resumed = module.build_case(case_id, device="cpu", seed=0, max_steps=6)
        resumed.load_state_dict(saved)
        for _ in range(3):
            resumed.step()
        resumed_obs = resumed.observe(n=64, seed=10000)

        assert resumed.completed_steps == 6
        assert state_digest(fixture.state_dict()) == state_digest(resumed.state_dict())
        assert direct_obs["metrics"] == resumed_obs["metrics"]

        # Reject incompatible v1 schema checkpoints explicitly.
        bad_schema = deepcopy(saved)
        bad_schema["schema"] = "api-model-ddgan-v1"
        with pytest.raises(ValueError, match="schema"):
            resumed.load_state_dict(bad_schema)
    finally:
        torch.set_num_threads(old_threads)


def test_model_toy_v2_adapter_honors_terminal_observations_1_and_calibrated_timeouts(tmp_path):
    tasks = load_tasks(ROOT)
    expected_timeouts = {
        "ddgan_gaussian_chain_v2": 600,
        "ddgan_bimodal_chain_v2": 600,
        "ae_ddgan_gaussian_chain_v2": 600,
        "ae_ddgan_bimodal_chain_v2": 900,
        "five_word_joint_smoke_1800s": 1800,
    }
    for tid, timeout in expected_timeouts.items():
        assert tasks[tid]["resources"]["timeout_seconds"] == timeout
        assert adapter_preflight(tasks[tid], CANDIDATE_BISPF, root=ROOT) == []

    # Legacy v1 pinned cards remain untouched; grid100_mlp16_10k coexists with 7k grid100_mlp16.
    assert tasks["model_ae_ddgan_gaussian_chain"]["resources"]["timeout_seconds"] == 180
    assert tasks["model_ae_ddgan_bimodal_chain"]["resources"]["timeout_seconds"] == 300
    assert tasks["five_word_joint_smoke"]["resources"]["timeout_seconds"] == 900
    assert tasks["grid100_mlp16"]["execution"]["steps"] == 7000
    assert tasks["grid100_mlp16_10k"]["execution"]["steps"] == 10000
    assert adapter_preflight(tasks["grid100_mlp16_10k"], CANDIDATE_V14, root=ROOT) == []

    # Validate terminal_observations == 1 contract and rejection of mismatched minimum_stable_checks.
    task = deepcopy(tasks["ae_ddgan_gaussian_chain_v2"])
    assert task["evaluation"]["minimum_stable_checks"] == 1
    bad_task = deepcopy(task)
    bad_task["evaluation"]["minimum_stable_checks"] = 5
    with pytest.raises(ValueError, match="minimum_stable_checks=1"):
        v2_adapter.validate_task(bad_task)

    # Synthetic grading: terminal_observations=1 passes when observation 24 + live pass even if obs 20 failed.
    steps = task["execution"]["steps"]
    ckpts = [math.ceil(i * steps / 24) for i in range(1, 25)]
    good_metrics, bad_metrics = {}, {}
    for name, op, bound in task["evaluation"]["thresholds"]:
        good_metrics[name] = bound
        bad_metrics[name] = bound - 1.0 if op == ">=" else bound + 1.0

    obs_pass_terminal = [{"step": s, **(bad_metrics if idx < 23 else good_metrics)} for idx, s in enumerate(ckpts)]
    ev_pass = {
        **executed_receipt(MODEL_TOY_CLEAN, eval_output_noise="clean"),
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
        "guards": {
            "all_finite": True,
            "optimizer_updates": {"generator": steps, "encoder": steps, "prior": steps, "discriminator": steps},
            "hooks_exercised": True,
            "mechanism_audit": {
                "mechanisms": {
                    name: {"requested": False, "enabled": False, "calls": 0, "eligible": 0, "applied": 0}
                    for name in MECHANISM_NAMES
                }
            },
            "unintended_rng_deviations": 0,
        },
        "observations": obs_pass_terminal,
        "live": dict(good_metrics),
    }
    graded = grade_result(task, {"evidence": ev_pass})
    assert graded["status"] == "PASS"
    assert graded["evaluator_result"]["convergence"]["minimum_stable_checks"] == 1
    assert graded["evaluator_result"]["convergence"]["confirmed_step"] == steps

    # If terminal observation fails, terminal_observations=1 must FAIL.
    ev_fail = deepcopy(ev_pass)
    ev_fail["observations"][-1] = {"step": steps, **bad_metrics}
    ev_fail["live"] = dict(bad_metrics)
    assert grade_result(task, {"evidence": ev_fail})["status"] == "FAIL"

    # Short 2-step Forge execution on v2 DDGAN & AE-DDGAN renders GIF, isolates RNG, and cannot qualify.
    for tid in ("ddgan_gaussian_chain_v2", "ae_ddgan_gaussian_chain_v2"):
        res = v2_adapter.run(
            {"candidate": CANDIDATE_BISPF, "protocol": {"seed": 0}},
            tasks[tid],
            tmp_path / tid,
            "cpu",
            execution_limit=2,
        )
        assert res["scope"] == "integration_demo_only"
        assert res["cost"]["completed_steps"] == 2
        assert res["evidence"]["oracle_free"] is True
        assert res["evidence"]["guards"]["all_finite"] is True
        assert res["evidence"]["guards"]["unintended_rng_deviations"] == 0
        assert (tmp_path / tid / "api/goal.gif").read_bytes().startswith(b"GIF")
        assert grade_result(tasks[tid], res)["gate_status"] != "PASS"


@pytest.mark.parametrize("preset", ["axiom", "unified_v11", "minimal"])
def test_t3_isolation_binds_axiom_unified_v11_and_minimal_without_ema(preset):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = get_recipe(preset).to_dict()
        base.update(
            num_particles=512,
            z_dim=2,
            batch_size=128,
            total_steps=6,
            prior_kind="particles",
            sigma_rel=0.0,
            standardize=False,
        )
        recipe = Recipe.from_dict(base)
        fix = t3_iso.T3IsolationFixture("t3_iso_scale_mass_8x", recipe, device="cpu", seed=0, max_steps=6)
        assert getattr(fix.opt_d, "ema_critic", None) is None
        rec = getattr(fix.opt_d, "record", None)
        if rec is not None and hasattr(fix.loss, "record"):
            assert fix.loss.record is rec
        for _ in range(2):
            fix.step()
        assert fix.completed_steps == 2
        assert torch.isfinite(fix.prior.z).all()
    finally:
        torch.set_num_threads(old_threads)


# ---------------------------------------------------------------------------
# #4 — four-laws-adherence-v2 view & 6 new probes (v1 untouched)
# ---------------------------------------------------------------------------


def test_four_laws_adherence_v2_view_and_probes_preserve_v1_untouched():
    tasks = load_tasks(ROOT)
    v1 = load_view(ROOT, "four-laws-adherence-v1")
    v2 = load_view(ROOT, "four-laws-adherence-v2")
    validate_view(v1, tasks)
    validate_view(v2, tasks)

    v1_tasks = {a["task"] for a in v1["assignments"]}
    v2_tasks = {a["task"] for a in v2["assignments"]}
    assert len(v1["assignments"]) == 33
    assert "fl_law44_highd_sparse_prior_and_shock" in v1_tasks
    assert "fl_law44_highd_sparse_prior_and_shock" not in v2_tasks
    assert v2["evidence_scope"] == "research_diagnostic"

    expected_v2_probes = {
        "fl_law1_aniso_unequal_modes",
        "fl_law1_noisy_atomic_codebook_v2",
        "fl_law1_highd_rank_d32_v2",
        "fl_law2_pure_adversarial_codim1",
        "fl_law2_groupnorm_and_vit_sprites_v2",
        "fl_law3_highd_orphan_d32_v2",
        "fl_law41_trust_rare1pct_tanh",
        "fl_law42_swap_deep_critic_16l",
        "fl_law42_deep_resbn_and_conv1x1_v2",
        "fl_law44_scale_shock",
    }
    v2_probes = {a["task"] for a in v2["assignments"] if a["role"] == "four_laws_probe"}
    assert v2_probes == expected_v2_probes
    assert "ae_ddgan_gaussian_chain_v2" in v2_tasks
    assert "grid100_mlp16_10k" in v2_tasks
    assert [a["order"] for a in v2["assignments"]] == list(range(len(v2["assignments"])))

    # Probe parameter verifications:
    assert tasks["fl_law1_noisy_atomic_codebook_v2"]["execution"]["prior"]["sigma"] == pytest.approx(0.01)
    assert tasks["fl_law1_noisy_atomic_codebook_v2"]["execution"]["host_definition"]["resources"]["num_particles"] == 160
    assert tasks["fl_law1_highd_rank_d32_v2"]["execution"]["host_definition"]["resources"]["num_particles"] == 512
    assert tasks["fl_law2_groupnorm_and_vit_sprites_v2"]["execution"]["steps"] == 1600
    assert tasks["fl_law44_scale_shock"]["execution"]["steps"] == 1000
    assert tasks["fl_law44_scale_shock"]["execution"]["host_definition"]["model"]["shock_step"] == 300

    # Conv1x1 v2 uses absolute parity <= 0.015 instead of inverse-quality ratio <= 1.05:
    conv_thresh = {name: (op, bound) for name, op, bound in tasks["fl_law42_deep_resbn_and_conv1x1_v2"]["evaluation"]["thresholds"]}
    assert "conv1x1_sw1_abs_diff" in conv_thresh and conv_thresh["conv1x1_sw1_abs_diff"] == ("<=", 0.015)
    assert "conv1x1_sw1_ratio" not in conv_thresh


def test_four_laws_v2_oracle_controls_preflight_grading_and_shock_continuation():
    tasks = load_tasks(ROOT)
    # Run all v2 oracle controls across both modules.
    for mod in (fl123, fl4):
        controls = mod.oracle_controls_v2()
        assert controls
        for name, item in controls.items():
            assert item["passed"] is item["expected_pass"], f"{name}: {item}"

    # Verify Conv1x1 absolute parity passes tiny SW1 with >1.05 ratio yet rejects real >0.015 gap!
    g_tiny = torch.Generator().manual_seed(11)
    ref128 = fl4.sample_target_for_kind("fl_law42_deep_resbn_and_conv1x1_v2", 2048, g_tiny)
    # Near-exact linear samples (sw1 ~ 0.008) vs slightly perturbed conv twin (abs_diff <= 0.015):
    lin_s = ref128.clone()
    conv_ok = ref128.clone() + 0.003 * torch.randn(ref128.shape, generator=torch.Generator().manual_seed(12))
    res_ok = fl4.score_fl_law42_deep_resbn_and_conv1x1_v2(lin_s, samples_conv1x1=conv_ok, reference=ref128)
    assert res_ok["passed"] is True
    assert res_ok["metrics"]["conv1x1_sw1_abs_diff"] <= 0.015
    conv_bad = ref128.clone() * 1.45
    res_bad = fl4.score_fl_law42_deep_resbn_and_conv1x1_v2(lin_s, samples_conv1x1=conv_bad, reference=ref128)
    assert res_bad["passed"] is False and "conv1x1_sw1_abs_diff <= 0.015" in res_bad["failed_bounds"]

    # Verify preflight + fail-closed synthetic grading for all 6 new v2 probes.
    for tid in sorted(fl_adapter.FOUR_LAWS_V2_KINDS):
        task = deepcopy(tasks[tid])
        assert adapter_preflight(task, CANDIDATE_V14, root=ROOT) == []
        steps = task["execution"]["steps"]
        obs_count = task["evaluation"]["observations"]
        ckpts = sorted({math.ceil(i * steps / obs_count) for i in range(1, obs_count + 1)})
        passing_metrics, failing_metrics = {}, {}
        for name, op, bound in task["evaluation"]["thresholds"]:
            passing_metrics[name] = bound
            failing_metrics[name] = bound - 1.0 if op == ">=" else bound + 1.0
        base_ev = {
            **executed_receipt(task["evaluation"]["sampling_law"], eval_output_noise="clean"),
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
            "four_laws_contract": fl_adapter._contract_payload(tid, task["execution"]["host_definition"]),
            "actual_training_gif": {"sha256": "0" * 64, "frames": obs_count},
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
            "observations": [{"step": s, **passing_metrics} for s in ckpts],
            "live": {"step": steps, **passing_metrics},
        }
        assert grade_result(task, {"evidence": base_ev})["status"] == "PASS"
        fail_ev = {
            **base_ev,
            "observations": [{"step": s, **failing_metrics} for s in ckpts],
            "live": {"step": steps, **failing_metrics},
        }
        assert grade_result(task, {"evidence": fail_ev})["status"] == "FAIL"

    # Standalone fl_law44_scale_shock: zero aux_trainers + exact checkpoint continuation across step-300 shock!
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fix_direct = fl4.build_case("fl_law44_scale_shock", device="cpu", seed=0, recipe_name="minimal", max_steps=304)
        assert fix_direct.aux_trainers == {}
        for _ in range(304):
            fix_direct.step()
        obs_direct = fix_direct.observe(n=256, seed=10000)

        fix_part = fl4.build_case("fl_law44_scale_shock", device="cpu", seed=0, recipe_name="minimal", max_steps=304)
        for _ in range(299):
            fix_part.step()
        ckpt = deepcopy(fix_part.state_dict())

        fix_resume = fl4.build_case("fl_law44_scale_shock", device="cpu", seed=0, recipe_name="minimal", max_steps=304)
        fix_resume.load_state_dict(ckpt)
        for _ in range(5):
            fix_resume.step()
        obs_resume = fix_resume.observe(n=256, seed=10000)

        assert fix_resume.completed_steps == 304
        assert state_digest(fix_direct.trainer.G.state_dict()) == state_digest(fix_resume.trainer.G.state_dict())
        assert state_digest(fix_direct.trainer.D.state_dict()) == state_digest(fix_resume.trainer.D.state_dict())
        assert obs_direct["metrics"] == obs_resume["metrics"]
    finally:
        torch.set_num_threads(old_threads)


def test_repaired_axiom_four_laws_benchmark_formulas_and_nonzero_sync_counter():
    """Verify repaired 4-laws benchmark: nonzero sync tracking, seeded shuffled 3C tail, real 3B, Suite 4A parity."""
    from scripts import axiom_four_laws_benchmark as bench
    from scripts.four_laws_audit import p7_law1_topology as p7

    # 1. _count_host_syncs catches BOTH torch.nonzero(t) and t.nonzero() in both audit scripts
    for mod in (bench, p7):
        with mod._count_host_syncs(torch.device("cpu")) as sc:
            t = torch.tensor([0, 1, 0, 2])
            _ = torch.nonzero(t)
            _ = t.nonzero()
        assert sc["nonzero"] == 2, f"{mod.__name__}: {sc}"

    # 2. Suite 3C uses deterministic seeded shuffled indices (never hardcoded [126, 127]) and real 3B optimizers
    r3 = bench.run_suite3_law41_43_p4()
    tail_idx = r3["rare_1pct_tail_capture"]["axiom"]["tail_indices"]
    assert len(tail_idx) == 2 and tail_idx != [126, 127]
    assert abs(r3["coord_scale_homogeneity"]["1e-04"]["axiom_ratio_vs_c1"] - 1.0) < 1e-3

    # 3. Source-level check: zero legacy _t1_coupling import in axiom_four_laws_benchmark.py & exact Suite 4A formulas
    src = (ROOT / "scripts/axiom_four_laws_benchmark.py").read_text(encoding="utf-8")
    assert "from particlegan._t1_coupling" not in src
    assert "laguerre_hard_commitment" not in src
    assert "beta=20.0" in src
    assert "0.25 * cap_c" in src
    assert "0.005 * (sr.mean() + sf.mean())" in src
    assert "8.0 * sr" in src and "0.08 * chord" in src


def test_ae_ddgan_v2_transparent_thresholds_zero_reconciled_discard():
    """api_model_ae_ddgan_v2 must match case['thresholds'] == forge_thresholds with zero reconciled.discard."""
    from benchmarks.toy_audit import api_model_ae_ddgan_v2 as ae2

    src = (ROOT / "benchmarks/toy_audit/api_model_ae_ddgan_v2.py").read_text(encoding="utf-8")
    assert "reconciled.discard" not in src
    for case in ae2.list_cases():
        expected = {name: f"{op} {bound}" for name, op, bound in case["forge_thresholds"]}
        assert case["thresholds"] == expected
        # Every bound violation must be reported fail-closed without silent dropping
        passing_metrics = {name: bound for name, _, bound in case["forge_thresholds"]}
        assert ae2._evaluate_thresholds(passing_metrics, case["forge_thresholds"]) == []
        breached = dict(passing_metrics)
        first_name, first_op, first_bound = case["forge_thresholds"][0]
        breached[first_name] = first_bound + 1.0 if first_op == "<=" else first_bound - 1.0
        assert first_name in ae2._evaluate_thresholds(breached, case["forge_thresholds"])


def test_four_laws_123_real_sdpa_gating_critic_grad_and_no_swallowed_exceptions():
    """ResNetGroupNormViTCritic uses real F.scaled_dot_product_attention; critic_grad_finite gates PASS; errors propagate."""
    import torch.nn.functional as F
    from unittest import mock

    critic = fl123.ResNetGroupNormViTCritic(in_ch=3, base_ch=16, vit_dim=32, vit_heads=2)
    sdpa_calls = 0
    orig_sdpa = F.scaled_dot_product_attention

    def _spy_sdpa(*args, **kwargs):
        nonlocal sdpa_calls
        sdpa_calls += 1
        return orig_sdpa(*args, **kwargs)

    with mock.patch.object(F, "scaled_dot_product_attention", side_effect=_spy_sdpa):
        out = critic(torch.randn(2, 3, 32, 32))
    assert out.shape == (2,)
    assert sdpa_calls >= 1

    # Both v1 and v2 sprites tasks must gate on critic_grad_finite == 1.0
    tpl = fl123.sprite_templates("cpu")
    ora = fl123.sample_target_for_kind("fl_law2_groupnorm_and_vit_sprites", 64, torch.Generator().manual_seed(0))
    res_v1 = fl123.score_fl_law2_groupnorm_and_vit_sprites(ora, critic_grad_finite=0.0, reference=tpl)
    res_v2 = fl123.score_fl_law2_groupnorm_and_vit_sprites_v2(ora, critic_grad_finite=0.0, reference=tpl)
    assert res_v1["passed"] is False and "critic_grad_finite == 1.0" in res_v1["failed_bounds"]
    assert res_v2["passed"] is False and "critic_grad_finite == 1.0" in res_v2["failed_bounds"]

    # Runtime exceptions inside critic gradient measurement must propagate (not be swallowed into 0.0/nan)
    class _BrokenCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            raise RuntimeError("injected critic boom")

    broken = _BrokenCritic()
    with pytest.raises(RuntimeError, match="injected critic boom"):
        fl123._measure_sprite_critic_grad(broken, torch.zeros(4, 3, 32, 32))
    with pytest.raises(RuntimeError, match="injected critic boom"):
        fl123._measure_normal_critic_grad(broken, torch.zeros(4, 2))


def test_tier4_providers_zero_bispf_branching_oracle_or_mmd_shortcuts():
    """Verify Tier-4 providers removed bispf branching, oracle RING_RADIUS, identity skips, MMD shortcuts, and bind axiom/v15."""
    import inspect
    from benchmarks.toy_audit import (
        api_bytegan,
        api_cramergan,
        api_cyclegan,
        api_infogan,
        api_pacgan,
    )

    # 1. InfoGAN: zero bispf branching in step(), zero RING_RADIUS in G/Q
    assert "bispf" not in inspect.getsource(api_infogan.InfoGANAnisotropicFixture.step)
    assert "RING_RADIUS" not in inspect.getsource(api_infogan.InfoGANGenerator)
    assert "RING_RADIUS" not in inspect.getsource(api_infogan.InfoGANRecognition)
    assert "3.0 *" not in inspect.getsource(api_infogan.InfoGANGenerator.forward)

    # 2. CycleGAN: zero bispf branching in step()
    assert "bispf" not in inspect.getsource(api_cyclegan.CycleGANConstellationFixture.step)

    # 3. CramerGAN: zero data-space identity skip in CramerCritic.embed, zero data-space MMD shortcut
    critic = api_cramergan.CramerCritic(in_dim=2, embed_dim=16, hidden=32)
    with torch.no_grad():
        critic.embedding_head.weight.zero_()
        critic.embedding_head.bias.zero_()
    x_probe = torch.randn(8, 2) + 5.0
    assert torch.allclose(critic.embed(x_probe), torch.zeros(8, 16)), "Identity skip still present in CramerCritic.embed!"

    hr, hf1, hf2 = torch.randn(8, 16), torch.randn(8, 16), torch.randn(8, 16)
    xr, xf1, xf2 = torch.randn(8, 2), torch.randn(8, 2) + 10.0, torch.randn(8, 2) - 10.0
    loss_pure = api_cramergan.cramer_generator_loss(hr, hf1, hf2)
    loss_with_x = api_cramergan.cramer_generator_loss(hr, hf1, hf2, real_x=xr, fake_x1=xf1, fake_x2=xf2)
    assert torch.allclose(loss_pure, loss_with_x, atol=1e-7), "Data-space MMD shortcut still active!"

    # 4. Caller-owned fixtures cleanly bind formulation records for axiom and unified_v15
    for preset in ("axiom", "unified_v15"):
        b_fix = api_bytegan.build_case("api-bytegan-bigan-dictionary", device="cpu", seed=0, recipe_name=preset, max_steps=2)
        assert getattr(b_fix.loss, "record", None) is b_fix.opt_d.record
        p_fix = api_pacgan.PacGANUnequalMassFixture(device="cpu", seed=0, recipe_name=preset, max_steps=2)
        assert getattr(p_fix.loss, "record", None) is p_fix.opt_d.record
        i_fix = api_infogan.InfoGANAnisotropicFixture(device="cpu", seed=0, recipe_name=preset, max_steps=2)
        assert getattr(i_fix.loss, "record", None) is i_fix.opt_d.record
        c_fix = api_cyclegan.CycleGANConstellationFixture(device="cpu", seed=0, recipe_name=preset, max_steps=2)
        assert getattr(c_fix.loss, "record", None) is c_fix.opt_d.record
        cr_fix = api_cramergan.CramerGANOverlapFixture(device="cpu", seed=0, recipe_name=preset, max_steps=2)
        assert getattr(cr_fix.loss, "record", None) is cr_fix.opt_d.record
