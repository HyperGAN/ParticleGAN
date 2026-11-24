"""Receipt integrity, comparability, code-complexity, and semantic control tests for P8 & P9 audit."""
from __future__ import annotations

import json
import pathlib
import sys
from unittest import mock

import pytest
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.four_laws_audit.p9_verify_receipts import (  # noqa: E402
    audit_code_complexity,
    audit_four_laws_links,
    audit_non_progression_readout_metrics,
    audit_p8_tuning_and_unseen,
    audit_progression_table,
    audit_unified_v12_receipts,
)


def test_four_laws_links_and_missing_receipts_detected() -> None:
    """Verify every existing readout loads and flag the exact 3 broken links in docs/four-laws.md."""
    res = audit_four_laws_links(ROOT)
    assert res["checked_local_file_links"] >= 60
    assert res["unique_readout_prefixes"] == 40
    assert res["existing_readout_count"] == 38

    broken_targets = {b["target"] for b in res["broken_links"]}
    assert broken_targets == {
        "../reports/forge/records/readout-5002641a64ddaa91f1fc3b87.json",
        "../reports/forge/records/readout-731e752ff3c4f97e29a84145.json",
        "../particlegan/unified_v12.py",
    }
    assert res["missing_readouts"] == ["readout-5002641a", "readout-731e752f"]

    # Confirm every existing readout has non-empty task_results and a candidate_id
    for item in res["readout_inventory"]:
        if item["exists"]:
            assert item["candidate_id"]
            assert item["n_tasks"] > 0


def test_link_auditor_semantic_positive_control(tmp_path: pathlib.Path) -> None:
    """Semantic control: auditor catches newly injected broken link and reports 0 broken when fixed."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "reports" / "forge" / "records").mkdir(parents=True)
    rec = tmp_path / "reports" / "forge" / "records" / "readout-deadbeef0000000000000000.json"
    rec.write_text(
        json.dumps(
            {
                "candidate_id": "ctrl",
                "study_id": "ctrl-study",
                "goal": "ctrl-view",
                "evidence_scope": "research_diagnostic",
                "task_results": [{"task_id": "t0", "gate_status": "PASS"}],
            }
        ),
        encoding="utf-8",
    )

    doc = tmp_path / "docs" / "four-laws.md"
    doc.write_text(
        "Good: [ok](../reports/forge/records/readout-deadbeef0000000000000000.json)\n"
        "Bad: [bad](../reports/forge/records/readout-111111112222222233333333.json)\n",
        encoding="utf-8",
    )
    bad_audit = audit_four_laws_links(tmp_path)
    assert len(bad_audit["broken_links"]) == 1
    assert bad_audit["missing_readouts"] == ["readout-11111111"]

    doc.write_text(
        "Good: [ok](../reports/forge/records/readout-deadbeef0000000000000000.json)\n",
        encoding="utf-8",
    )
    good_audit = audit_four_laws_links(tmp_path)
    assert good_audit["broken_links"] == []
    assert good_audit["missing_readouts"] == []
    assert good_audit["existing_readout_count"] == 1


def test_committed_unified_v12_readouts_and_three_way_r256_discrepancy() -> None:
    """Verify exact metrics in the 4 committed unified-v12 readouts and expose r256/interp discrepancies."""
    v12 = audit_unified_v12_receipts(ROOT)

    committed_files = {r["file"] for r in v12["committed_v12_readouts"]}
    assert committed_files == {
        "readout-59898e1a77c4b179c9aebbc1.json",
        "readout-624964f4e46793593bc147e4.json",
        "readout-b51ab67c4d97526186e23651.json",
        "readout-c1ee5c3bb7ac4f432532da63.json",
    }

    # No committed readout-*.json exists for unified-v12 on Tier-3 holdout or shift-recovery diagnostic
    assert "r3-tier3-long-horizon-holdout-v1" not in v12["committed_v12_goals"]
    assert "t3-shift-recovery-diagnostic-v1" not in v12["committed_v12_goals"]

    # Verified pass counts in readout-b51ab67c
    ds = v12["disc_stab_b51ab67c"]
    assert ds["pass_count"] == 46
    assert ds["total_count"] == 48
    assert ds["required_pass"] == 29
    assert ds["required_total"] == 29
    assert set(ds["failures"].keys()) == {"interp_img_patch_shift", "interp_saddle_barycentric"}

    # Crucial discrepancy: unified-v12's interp_img_patch_shift failure on RTX A6000 is genuine
    # numerical ghost failure (interp_peak ~ 0.4915 < 0.78, finite_fraction == 1.0), NOT 0.99999994 float32 edge!
    patch_fail = ds["failures"]["interp_img_patch_shift"]["metrics"]
    assert patch_fail["finite_fraction"] == 1.0
    assert patch_fail["interp_peak"] == pytest.approx(0.49147, abs=1e-3)
    assert patch_fail["endpoint_rmse"] > 0.09
    assert patch_fail["interp_rmse"] > 0.10

    # Contrast with unified-v11 (readout-fc5b67a7), which DID hit the 0.99999994 float32 edge case
    v11_disc = json.loads((ROOT / "reports/forge/records/readout-fc5b67a767e57d748f460610.json").read_text())
    v11_patch = [t for t in v11_disc["task_results"] if t["task_id"] == "interp_img_patch_shift"][0]
    assert v11_patch["metrics"]["finite_fraction"] < 1.0
    assert v11_patch["metrics"]["endpoint_hq"] == 0.0

    # Standalone t4-arch readout-c1ee5c3b is 7/7 PASS
    assert v12["t4_arch_c1ee5c3b"]["pass_count"] == 7
    assert v12["t4_arch_c1ee5c3b"]["total_count"] == 7
    assert v12["t4_arch_c1ee5c3b"]["grid100_res16_tanh_status"] == "PASS"

    # Three-way r256 latency discrepancy for unified-v12
    tw = v12["r256_three_way_discrepancy"]
    assert tw["committed_forge_standalone_ff41cf173"]["update_ms_median"] == pytest.approx(41.39, abs=0.02)
    assert tw["committed_forge_standalone_ff41cf173"]["peak_allocated_mib"] == pytest.approx(1243.44, abs=0.1)
    assert tw["uncommitted_forge_combined_table_line193"]["claimed_ms"] == pytest.approx(39.76, abs=1e-3)
    assert tw["uncommitted_forge_combined_table_line193"]["committed_readout_file"] is None
    assert tw["nvml_profiler_50step_prose_line169"]["step_ms_median"] == pytest.approx(39.22, abs=0.02)
    assert tw["nvml_profiler_50step_prose_line169"]["joules_per_step"] == pytest.approx(11.37, abs=0.02)
    assert tw["nvml_profiler_50step_prose_line169"]["dynamic_joules_per_step"] == pytest.approx(8.68, abs=0.02)
    assert tw["nvml_profiler_50step_prose_line169"]["idle_power_watts"] == pytest.approx(68.15, abs=0.05)


def test_progression_table_precision_and_contention_confounds() -> None:
    """Verify exact progression table receipts and flag FP32 vs BF16, contention, and dynamic-energy confounds."""
    rows = {r["candidate_id"]: r for r in audit_progression_table(ROOT)}

    # Verbatim matches against committed Forge readouts
    assert rows["rhbi-v12"]["forge_ms_median"] == pytest.approx(588.75, abs=0.02)
    assert rows["rhbi-v12"]["forge_peak_allocated_mib"] == pytest.approx(3750.55, abs=0.1)
    assert rows["empf-v8"]["forge_ms_median"] == pytest.approx(130.13, abs=0.02)
    assert rows["empf-v8"]["forge_peak_allocated_mib"] == pytest.approx(1435.44, abs=0.1)
    assert rows["unified-v11"]["forge_ms_median"] == pytest.approx(89.19, abs=0.02)
    assert rows["unified-v11"]["forge_peak_allocated_mib"] == pytest.approx(1372.60, abs=0.1)
    assert rows["t4-nash-v3"]["forge_ms_median"] == pytest.approx(39.68, abs=0.02)
    assert rows["minimal-v2"]["forge_ms_median"] == pytest.approx(55.50, abs=0.02)
    assert rows["law4-energy-v3"]["forge_ms_median"] == pytest.approx(45.89, abs=0.02)

    # Precision confound: rhbi-v12 is float32 while all newer champions are bf16_channels_last
    assert rows["rhbi-v12"]["training_precision"] == "float32"
    for cid in ("empf-v8", "unified-v10", "unified-v11", "unified-v12", "t4-nash-v3", "minimal-v2", "law4-energy-v3", "law4-energy-v10"):
        assert rows[cid]["training_precision"] == "bf16_channels_last"

    # Contention confound: law4-energy-v10 and law4-energy-v11 committed Forge r256 readouts are contended=True
    assert rows["law4-energy-v10"]["forge_contended"] is True
    assert rows["law4-energy-v10"]["forge_ms_median"] == pytest.approx(56.35, abs=0.02)
    assert rows["law4-energy-v10"]["profiler_phase2"]["profiler_ms_median"] == pytest.approx(45.30, abs=0.02)
    assert rows["law4-energy-v10"]["profiler_phase2"]["profiler_j_per_step"] == pytest.approx(11.68, abs=0.02)
    assert rows["law4-energy-v11"]["forge_contended"] is True

    # Dynamic energy inversion: law4-energy-v10 has LOWER dynamic J/step (8.45) than unified-v12 (8.68)
    v10_dyn = rows["law4-energy-v10"]["profiler_phase2"]["profiler_dynamic_j_per_step"]
    v12_dyn = rows["unified-v12"]["profiler_phase2"]["profiler_dynamic_j_per_step"]
    assert v10_dyn < v12_dyn


def test_code_complexity_and_surviving_occam_epicycles() -> None:
    """Verify exact LOC / branch counts and confirm Round-4 redundant epicycles survive in unified-v12."""
    cc = audit_code_complexity(ROOT)
    pm = cc["package_metrics"]

    assert pm["minimal-v2"]["loc"] == 899
    assert pm["unified-v11_current"]["loc"] > 7000  # Bloat beyond stale 4,713 claimed in docs/four-laws.md
    assert pm["unified-v12_delta_only"]["loc"] > 1400
    assert pm["unified-v12_effective_direct"]["loc"] == (
        pm["unified-v12_delta_only"]["loc"] + pm["unified-v11_current"]["loc"]
    )
    assert pm["unified-v12_effective_direct"]["loc"] > 8500
    assert pm["unified-v12_effective_direct"]["total_branches"] > 1000

    # Surviving Round-4 Occam redundant epicycles in unified-v12 (and AST-verified drops)
    ep = cc["surviving_occam_epicycles"]
    assert ep["radon_hurwitz_dz4_branch_in_prior"] is True
    assert ep["fwad_generator_probe_w_sat_pt_in_prior"] is True
    assert ep["conformal_horizon_flag_enabled_in_v12_idea"] is True
    assert ep["multiscale_horizon_flag_enabled_in_v12_idea"] is True
    assert ep["critic_secant_dropped_in_v12_optimizer"] is True
    assert ep["conv2d_ns_dropped_in_v12_optimizer"] is True

    # Exact bank.py LOC verified (745 LOC)
    assert len((ROOT / "particlegan/_unified_v11/bank.py").read_text().splitlines()) == 745


def test_non_progression_readout_metrics_and_tier3_holdout_overlap() -> None:
    """Programmatically verify key cited non-progression readout metrics and exact Tier-3 holdout overlap."""
    np_m = audit_non_progression_readout_metrics(ROOT)
    assert (np_m["t1-prior-v2"]["pass_count"], np_m["t1-prior-v2"]["total_count"]) == (29, 42)
    assert (np_m["t1-coupling-v5"]["pass_count"], np_m["t1-coupling-v5"]["total_count"]) == (31, 44)
    assert np_m["t1-coupling-v5"]["metrics"]["modes"] == 4
    assert np_m["t1-coupling-v5"]["metrics"]["hq"] == pytest.approx(0.96875, abs=1e-4)
    assert np_m["t1-coupling-v5"]["metrics"]["distribution_tv"] == pytest.approx(0.15625, abs=1e-4)
    assert (np_m["cond-fiber-v1"]["pass_count"], np_m["cond-fiber-v1"]["total_count"]) == (40, 47)
    assert (np_m["unified-v11_disc_stab"]["pass_count"], np_m["unified-v11_disc_stab"]["total_count"]) == (41, 47)
    assert np_m["unified-v11_disc_stab"]["metrics"]["t1_q0_normalized_mean"] == pytest.approx(0.1790, abs=1e-3)
    assert np_m["unified-v11_disc_stab"]["metrics"]["t1_q0_covariance_eigen_max"] == pytest.approx(1.3085, abs=1e-3)
    assert (np_m["unified-v11_t3_iso"]["pass_count"], np_m["unified-v11_t3_iso"]["total_count"]) == (2, 2)
    assert (np_m["unified-v11_t4_nash"]["pass_count"], np_m["unified-v11_t4_nash"]["total_count"]) == (3, 3)
    assert (np_m["unified-v11_t3_shift"]["pass_count"], np_m["unified-v11_t3_shift"]["total_count"]) == (5, 5)
    assert (np_m["unified-v11_t4_arch"]["pass_count"], np_m["unified-v11_t4_arch"]["total_count"]) == (6, 7)
    assert (np_m["t4-arch-v3_diag"]["pass_count"], np_m["t4-arch-v3_diag"]["total_count"]) == (7, 7)
    assert np_m["t4-arch-v3_diag"]["metrics"]["precision"] == pytest.approx(0.9814, abs=1e-3)
    assert (np_m["r4_arm5_no_laguerre"]["pass_count"], np_m["r4_arm5_no_laguerre"]["total_count"]) == (7, 15)
    assert np_m["r4_arm5_no_laguerre"]["metrics"]["precision"] == pytest.approx(0.2827, abs=1e-3)
    assert (np_m["r4_arm4_no_coalesce"]["pass_count"], np_m["r4_arm4_no_coalesce"]["total_count"]) == (9, 15)
    assert np_m["r4_arm4_no_coalesce"]["metrics"]["center_rms_sigma"] == pytest.approx(0.5312, abs=1e-3)
    assert np_m["r4_arm2_no_secant"]["metrics"]["hq"] == pytest.approx(0.2522, abs=1e-3)
    assert np_m["r4_arm1_no_ns"]["metrics"]["precision"] == pytest.approx(0.0244, abs=1e-3)
    assert np_m["r4_arm7_autograd"]["metrics"]["core_min_eigen_ratio"] == pytest.approx(0.5313, abs=1e-3)
    assert np_m["r4_arm8_no_sat_probe"]["metrics"]["mean_abs_error"] == pytest.approx(0.0026, abs=1e-3)

    # Verify exact staggered100 metric in readout-b51ab67c
    b51 = json.loads((ROOT / "reports/forge/records/readout-b51ab67c4d97526186e23651.json").read_text())
    s100 = [t for t in b51["task_results"] if t["task_id"] == "staggered100"][0]
    assert s100["metrics"]["precision"] == pytest.approx(0.9909, abs=1e-3)
    assert s100["metrics"]["center_rms_sigma"] == pytest.approx(0.0618, abs=1e-3)

    # Exact Tier-3 holdout overlap with active screening views (13/18 core, 14/18 incl. shift)
    ov = audit_p8_tuning_and_unseen(ROOT)["tier3_holdout_overlap"]
    assert ov["total_tasks"] == 18
    assert ov["overlap_core_screening_count"] == 13
    assert ov["overlap_core_screening_tasks"] == [
        "ae_gan_hold",
        "five_word_joint_hold",
        "five_word_joint_smoke",
        "grid100",
        "mode_hold",
        "ring_extension",
        "ring_hold",
        "rotated100",
        "staggered100",
        "two_pole",
        "unused_token_hold",
        "zero_tanh_rail_escape",
        "zero_tanh_smoke",
    ]
    assert ov["overlap_including_shift_count"] == 14
    assert ov["remaining_after_core_and_shift"] == [
        "clockfree_audit",
        "grid100_14k",
        "rotated100_14k",
        "staggered100_14k",
    ]


def test_p8_governance_calibration_and_unseen_failures() -> None:
    """Verify calibration BLOCKED status, governance drift, unseen ByteGAN failure, and L4 transfer failures."""
    import particlegan

    v12 = audit_unified_v12_receipts(ROOT)
    p8 = audit_p8_tuning_and_unseen(ROOT)

    # Governance gap: DEFAULT_RECIPE promoted to unified_v12 while calibration is BLOCKED & pin still points to unified-v11
    assert particlegan.DEFAULT_RECIPE == "unified_v12"
    assert v12["selection_pin_governance"]["default_adoption"] is False
    assert v12["selection_pin_governance"]["pinned_unified_candidate"] == "unified-v11"
    assert p8["calibration_adoption"] == "BLOCKED"

    # Substantial iterative tuning budget across unified-v1..v12 and law4-*
    assert p8["lineage_tuning_budgets"]["unified_v1_v12"]["declared_cards"] == 12
    assert p8["lineage_tuning_budgets"]["law4_energy_opt"]["declared_cards"] == 11
    assert p8["lineage_tuning_budgets"]["unified_v1_v12"]["concluded_readouts"] >= 60

    # Post-freeze unseen ByteGAN tasks: 0/2 PASS across unified-v12 (and all 5 families)
    v12_bg = [r for r in p8["bytegan_unseen_results"] if r["candidate_id"] == "unified-v12"]
    assert len(v12_bg) == 2
    assert all(r["gate_status"] == "FAIL" for r in v12_bg)
    assert all(r["gate_status"] == "FAIL" for r in p8["bytegan_unseen_results"])

    # Cross-GPU / host-transfer brittleness on NVIDIA L4
    l4_fail_tasks = {r["task_id"] for r in p8["unified_v12_l4_non_pass"] if r["gate_status"] == "FAIL"}
    assert "grid100_res16_tanh" in l4_fail_tasks
    assert "rotated100_affine_square_named_v1" in l4_fail_tasks
    assert "vector_unequal_width_published" in l4_fail_tasks


def test_unified_v12_zero_primitive_and_asymmetric_ns_semantic_controls() -> None:
    """Runtime semantic control on real particlegan unified_v12: 0 forbidden primitives, asymmetric NS behavior."""
    import particlegan

    def _forbid(name: str):
        def _boom(*args, **kwargs):
            raise AssertionError(f"Forbidden primitive called by unified_v12: {name}")
        return _boom

    with (
        mock.patch.object(torch.autograd, "grad", side_effect=_forbid("torch.autograd.grad")),
        mock.patch.object(torch.linalg, "svd", side_effect=_forbid("torch.linalg.svd")),
        mock.patch.object(torch.linalg, "svdvals", side_effect=_forbid("torch.linalg.svdvals")),
        mock.patch.object(torch.linalg, "eigh", side_effect=_forbid("torch.linalg.eigh")),
        mock.patch.object(torch.linalg, "eigvalsh", side_effect=_forbid("torch.linalg.eigvalsh")),
        mock.patch.object(torch.linalg, "pinv", side_effect=_forbid("torch.linalg.pinv")),
    ):
        # 1) Neural 2D MLP G + 2D MLP D: G uses 2D NS, D uses 0 Critic NS
        torch.manual_seed(0)
        g_mlp = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
        d_mlp = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        r_2d = particlegan.get_recipe("unified_v12", z_dim=2, batch_size=32, num_particles=16)
        trainer_mlp = particlegan.GANTrainer(r_2d, g_mlp, d_mlp, seed=0)
        x = torch.randn(32, 2)
        for _ in range(3):
            trainer_mlp.step(x, generator_real=x)
        assert trainer_mlp.opt_g.ns_5step_calls > 0
        assert trainer_mlp.opt_d.critic_ns_calls == 0

        # 2) Generator-free (G = nn.Identity()): D activates generator-free 2D Critic NS fallback
        torch.manual_seed(0)
        d_id = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        prior_gf = particlegan.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
        _, opt_d_gf = r_2d.make_optimizers(nn.Identity(), d_id, prior_gf)
        assert opt_d_gf._generator_free is True
        for p in d_id.parameters():
            p.grad = torch.randn_like(p)
        opt_d_gf.step()
        assert opt_d_gf.critic_ns_calls > 0

        # 3) 4D Conv2d G + D: strictly 0 Conv2d NS calls in both G and D
        torch.manual_seed(0)
        g_conv = nn.Sequential(
            nn.Unflatten(1, (8, 1, 1)),
            nn.ConvTranspose2d(8, 8, 4, 1, 0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        d_conv = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 1, bias=False),
        )
        r_conv = particlegan.get_recipe("unified_v12", z_dim=8, batch_size=8, num_particles=16)
        trainer_conv = particlegan.GANTrainer(r_conv, g_conv, d_conv, seed=0)
        img = torch.randn(8, 3, 8, 8)
        for _ in range(2):
            trainer_conv.step(img, generator_real=img)
        assert trainer_conv.opt_g.ns_5step_calls == 0
        assert trainer_conv.opt_d.critic_ns_calls == 0
