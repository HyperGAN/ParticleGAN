"""Unit, degree-0 scale-invariance, parity, edge-case, semantic-control, and fresh-process continuation tests for ``unified_v14`` prior smoothing."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from experiments.forge.techniques import recipe_field_active, technique_signature
from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    MoGParticlePrior,
    Recipe,
    get_recipe,
    init,
)
from particlegan._unified_v11 import UnifiedV11EquilibratedOptimizer, UnifiedV11StepRecord
from particlegan._unified_v12 import UnifiedV12Optimizer
from particlegan._unified_v14 import UnifiedV14Optimizer
from particlegan.optim.dualnorm import NORMALIZED_FAMILIES


def _forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("forbidden primitive or new_tensor called")


class _HighDMaskedGen(nn.Module):
    def __init__(self, z_dim: int = 8, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 32), nn.SiLU(), nn.Linear(32, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z)) * 0.1


class _RMSNormCrit(nn.Module):
    def __init__(self, in_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(nn.functional.rms_norm(x, (x.shape[-1],))).squeeze(-1)


def test_unified_v14_preset_registration_and_immutability_invariants() -> None:
    assert "unified_v14_equil" in NORMALIZED_FAMILIES
    assert DEFAULT_RECIPE == "unified_v12"
    assert get_recipe() == get_recipe("unified_v12") == get_recipe("gan")
    assert len(fields(Recipe)) == 179

    r12 = get_recipe("unified_v12")
    r13 = get_recipe("unified_v13")
    r14 = get_recipe("unified_v14")
    assert r12.optimizer_family == "unified_v12_equil"
    assert r13.optimizer_family == "unified_v12_equil"
    assert r13.rhbi_self_coupling == "feydy_atomic"
    assert r14.optimizer_family == "unified_v14_equil"
    assert r14.rhbi_self_coupling == "feydy_atomic"
    assert r14.critic_formulation == "unified_v11"
    assert r14.prior_regularizer == "unified_v11"
    assert r14.constraint_geometry_mode == "direction_blend"
    assert r14.optimizer_smoothing == pytest.approx(0.0061)
    assert Recipe.from_dict(r14.to_dict()) == r14

    sig = technique_signature(r14)
    assert sig["mechanisms"].get("unified_v14_equil_optimizer") is True
    assert recipe_field_active("optimizer_smoothing", r14) is True

    with pytest.raises(ValueError, match="direction_blend"):
        get_recipe("unified_v14", constraint_geometry_mode="nonascent")

    assert UnifiedV14Optimizer._SUPPORTS_ZEROSYNC_V4_TENSORS is True

    # Cross-family checkpoint isolation & feydy_atomic active-row hook preservation:
    p14 = nn.Parameter(torch.randn(16, 4))
    p12 = nn.Parameter(p14.detach().clone())
    p11 = nn.Parameter(p14.detach().clone())
    opt14 = UnifiedV14Optimizer([{"params": [p14], "role": "prior", "lr": 0.04}], smoothing=0.0061)
    opt12 = UnifiedV12Optimizer([{"params": [p12], "role": "prior", "lr": 0.04}], smoothing=0.0061)
    opt11 = UnifiedV11EquilibratedOptimizer(
        [{"params": [p11], "role": "prior", "algorithm": "unified_prior", "lr": 0.04}], smoothing=0.0061
    )

    rec = UnifiedV11StepRecord()
    opt14._unified_v11_record = rec
    rows = torch.tensor([5, 1, 5, 2], dtype=torch.long)
    opt14.set_sampled_rows(p14, rows)
    assert rec._active_prior_table is p14
    assert rec._active_batch_rows is not None and torch.equal(rec._active_batch_rows, rows)

    p14.grad = torch.randn_like(p14) * 1e-5
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "svdvals", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "eigvalsh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(torch.Tensor, "new_tensor", _forbid),
    ):
        opt14.step()

    sd14 = opt14.state_dict()
    assert "unified_v14_equil" in sd14
    with pytest.raises(ValueError):
        opt12.load_state_dict(sd14)
    with pytest.raises(ValueError):
        opt11.load_state_dict(sd14)
    with pytest.raises(ValueError):
        opt14.load_state_dict(opt12.state_dict())
    with pytest.raises(ValueError):
        opt14.load_state_dict(opt11.state_dict())


@pytest.mark.parametrize("path_mode", ["zerosync", "fallback_all_rows", "fallback_row_scales"])
@pytest.mark.parametrize("beta1", [0.0, 0.75])
def test_unified_v14_prior_degree0_scale_invariance_zerosync_and_fallback(
    path_mode: str, beta1: float
) -> None:
    """Prior row updates are trajectory degree-0 scale-invariant across 1e-3..1e-25 on both paths."""
    torch.manual_seed(0)
    num_particles, z_dim = 320, 16
    z_init = torch.randn(num_particles, z_dim)
    # Use >128 unique rows so the n_unique > 128 s_cap scaling branch is also exercised:
    raw_rows = torch.randint(0, num_particles, (400,), dtype=torch.long)
    uniq_rows = torch.unique(raw_rows, sorted=True)
    assert int(uniq_rows.numel()) > 128

    # 4 active steps + 1 decaying step (0.05x) to verify both active step and equilibrium damping are degree-0:
    base_grads = [torch.randn(num_particles, z_dim) for _ in range(4)]
    base_grads.append(torch.randn(num_particles, z_dim) * 0.05)
    row_scales = torch.linspace(0.5, 1.5, int(uniq_rows.numel()))

    def _run(scale: float) -> tuple[torch.Tensor, float]:
        z = nn.Parameter(z_init.clone())
        sampled_req = path_mode != "fallback_all_rows"
        opt = UnifiedV14Optimizer(
            [
                {
                    "params": [z],
                    "role": "prior",
                    "lr": 0.04,
                    "betas": (beta1, 0.999),
                    "sampled_rows_required": sampled_req,
                }
            ],
            smoothing=0.0061,
        )
        with mock.patch.object(torch.Tensor, "new_tensor", _forbid):
            for g in base_grads:
                z.grad = g * scale
                if path_mode == "zerosync":
                    opt.set_sampled_rows(z, raw_rows)
                elif path_mode == "fallback_row_scales":
                    opt.set_sampled_rows(z, uniq_rows)
                    opt.set_row_scales(z, row_scales)
                opt.step()
        return z.detach().clone(), float(opt.state[z]["peak_rms"].item())

    z_ref, pk_ref = _run(1e-5)
    assert pk_ref > 0.0
    # Verify sampled prior rows actually moved (under k=2.0, active gain 1/sqrt(5)~=0.447 => ~0.035 vs <0.001 frozen):
    eval_rows = torch.arange(num_particles) if path_mode == "fallback_all_rows" else uniq_rows
    mean_disp = float((z_ref[eval_rows] - z_init[eval_rows]).norm(dim=-1).mean().item())
    assert mean_disp > 0.025, f"Prior unexpectedly frozen on {path_mode} (beta1={beta1}): mean_disp={mean_disp:.6f}"

    # Trajectory degree-0 invariance across orders of magnitude below s_cap / k (s_cap / 2.0 ~= 1.7e-3):
    for scale in (1e-4, 1e-7, 1e-12, 1e-25):
        z_sc, pk_sc = _run(scale)
        assert pk_sc > 0.0
        assert torch.allclose(z_sc, z_ref, atol=1e-5, rtol=1e-5), (
            f"Scale invariance broken on {path_mode} beta1={beta1} at scale={scale}: "
            f"max_diff={(z_sc - z_ref).abs().max().item():.3e}"
        )

    # Normal low-D equilibrium settling preserved when peak_r * 2.0 >= s_cap (mirrors PR #606 control):
    z_eq = nn.Parameter(torch.zeros(128, 2))
    opt_eq = UnifiedV14Optimizer(
        [{"params": [z_eq], "role": "prior", "lr": 0.04, "sampled_rows_required": path_mode != "fallback_all_rows"}],
        smoothing=0.0061,
    )
    rows_128 = torch.arange(128, dtype=torch.long)
    if path_mode == "zerosync":
        opt_eq.set_sampled_rows(z_eq, rows_128)
    elif path_mode == "fallback_row_scales":
        opt_eq.set_sampled_rows(z_eq, rows_128)
        opt_eq.set_row_scales(z_eq, torch.ones(128))
    z_eq.grad = torch.full((128, 2), 0.02)  # peak_r = 0.02828 -> 2.0*peak_r = 0.05657 > 0.0061
    opt_eq.step()
    with torch.no_grad():
        z_eq.zero_()
    if path_mode == "zerosync":
        opt_eq.set_sampled_rows(z_eq, rows_128)
    elif path_mode == "fallback_row_scales":
        opt_eq.set_sampled_rows(z_eq, rows_128)
        opt_eq.set_row_scales(z_eq, torch.ones(128))
    z_eq.grad = torch.full((128, 2), 1e-4)
    opt_eq.step()
    row_disp = float(z_eq.detach().norm(dim=-1).mean().item())
    assert row_disp < 0.005, f"Prior equilibrium damping collapsed on {path_mode}: row_disp={row_disp:.5f}"

    # Bit-exact normal-scale parity with UnifiedV12Optimizer when peak_r * 2.0 >= s_cap (zero 1-ULP guard drift!):
    z_v14_n = nn.Parameter(z_init.clone())
    z_v12_n = nn.Parameter(z_init.clone())
    opt_v14_n = UnifiedV14Optimizer(
        [
            {
                "params": [z_v14_n],
                "role": "prior",
                "lr": 0.04,
                "betas": (beta1, 0.999),
                "sampled_rows_required": path_mode != "fallback_all_rows",
            }
        ],
        smoothing=0.0061,
    )
    opt_v12_n = UnifiedV12Optimizer(
        [
            {
                "params": [z_v12_n],
                "role": "prior",
                "lr": 0.04,
                "betas": (beta1, 0.999),
                "sampled_rows_required": path_mode != "fallback_all_rows",
            }
        ],
        smoothing=0.0061,
    )
    for g in base_grads:
        g_norm_scale = g * 0.05  # cur_row_rms ~= 0.20 >> s_cap / 2.0 -> s_p_row == s_cap bit-exact
        z_v14_n.grad = g_norm_scale.clone()
        z_v12_n.grad = g_norm_scale.clone()
        if path_mode == "zerosync":
            opt_v14_n.set_sampled_rows(z_v14_n, raw_rows)
            opt_v12_n.set_sampled_rows(z_v12_n, raw_rows)
        elif path_mode == "fallback_row_scales":
            opt_v14_n.set_sampled_rows(z_v14_n, uniq_rows)
            opt_v14_n.set_row_scales(z_v14_n, row_scales)
            opt_v12_n.set_sampled_rows(z_v12_n, uniq_rows)
            opt_v12_n.set_row_scales(z_v12_n, row_scales)
        opt_v14_n.step()
        opt_v12_n.step()
    assert torch.equal(z_v14_n.detach(), z_v12_n.detach()), (
        f"Normal-scale prior update drifted from bit-exact v12 parity on {path_mode} (beta1={beta1})"
    )
    assert torch.equal(opt_v14_n.state[z_v14_n]["peak_rms"], opt_v12_n.state[z_v12_n]["peak_rms"])


@pytest.mark.parametrize("scale", [1e-5, 0.05])
@pytest.mark.parametrize("n_particles,n_raw", [(32, 64), (320, 512)])
@pytest.mark.parametrize("beta1", [0.0, 0.75])
def test_unified_v14_prior_zerosync_vs_fallback_exact_parity(
    scale: float, n_particles: int, n_raw: int, beta1: float
) -> None:
    """Zero-sync (unsorted duplicate raw_rows) and fallback (unit row_scales / all-rows) match to FP32 precision."""
    torch.manual_seed(11)
    z_init = torch.randn(n_particles, 8)
    raw_idx = torch.randint(0, n_particles, (n_raw,), dtype=torch.long)
    uniq_idx = torch.unique(raw_idx, sorted=True)
    grads = [torch.randn(n_particles, 8) * scale for _ in range(3)]

    z_zs = nn.Parameter(z_init.clone())
    z_fb = nn.Parameter(z_init.clone())
    opt_zs = UnifiedV14Optimizer(
        [{"params": [z_zs], "role": "prior", "lr": 0.04, "betas": (beta1, 0.999)}], smoothing=0.0061
    )
    opt_fb = UnifiedV14Optimizer(
        [{"params": [z_fb], "role": "prior", "lr": 0.04, "betas": (beta1, 0.999)}], smoothing=0.0061
    )
    unit_scales = torch.ones(int(uniq_idx.numel()), dtype=torch.float32)

    for g in grads:
        z_zs.grad = g.clone()
        z_fb.grad = g.clone()
        opt_zs.set_sampled_rows(z_zs, raw_idx)
        opt_fb.set_sampled_rows(z_fb, uniq_idx)
        opt_fb.set_row_scales(z_fb, unit_scales)
        opt_zs.step()
        opt_fb.step()

    assert torch.allclose(z_zs.detach(), z_fb.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(opt_zs.state[z_zs]["peak_rms"], opt_fb.state[z_fb]["peak_rms"], atol=1e-6, rtol=1e-6)
    if beta1 > 0.0:
        assert torch.allclose(opt_zs.state[z_zs]["m_prior"], opt_fb.state[z_fb]["m_prior"], atol=1e-6, rtol=1e-6)

    # Also verify sampled_rows_required=False fallback matches zero-sync when all rows are sampled with duplicates:
    all_dups = torch.cat((torch.arange(n_particles, dtype=torch.long), torch.arange(min(8, n_particles), dtype=torch.long)))
    z_zs2 = nn.Parameter(z_init.clone())
    z_all = nn.Parameter(z_init.clone())
    opt_zs2 = UnifiedV14Optimizer(
        [{"params": [z_zs2], "role": "prior", "lr": 0.04, "betas": (beta1, 0.999)}], smoothing=0.0061
    )
    opt_all = UnifiedV14Optimizer(
        [
            {
                "params": [z_all],
                "role": "prior",
                "lr": 0.04,
                "betas": (beta1, 0.999),
                "sampled_rows_required": False,
            }
        ],
        smoothing=0.0061,
    )
    for g in grads:
        z_zs2.grad = g.clone()
        z_all.grad = g.clone()
        opt_zs2.set_sampled_rows(z_zs2, all_dups)
        opt_zs2.step()
        opt_all.step()
    assert torch.allclose(z_zs2.detach(), z_all.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(opt_zs2.state[z_zs2]["peak_rms"], opt_all.state[z_all]["peak_rms"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("use_fallback", [False, True])
def test_unified_v14_prior_edge_cases_zero_nan_inf_and_smoothing_zero_limit(use_fallback: bool) -> None:
    torch.manual_seed(3)
    # 1. All-zero gradient on step 1 (peak_r == 0): no 0/0 NaN, zero movement:
    z0 = nn.Parameter(torch.randn(16, 4))
    z0_orig = z0.detach().clone()
    opt0 = UnifiedV14Optimizer(
        [{"params": [z0], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=0.0061,
    )
    if not use_fallback:
        opt0.set_sampled_rows(z0, torch.tensor([0, 3, 3, 7, 15], dtype=torch.long))
    z0.grad = torch.zeros_like(z0)
    opt0.step()
    assert torch.equal(z0.detach(), z0_orig)
    assert float(opt0.state[z0]["peak_rms"].item()) == 0.0

    # 2. Mixed zero, NaN, +Inf, -Inf, huge 1e38 overflow-norm, and tiny active 1e-5 rows:
    z1 = nn.Parameter(torch.randn(16, 4))
    z1_orig = z1.detach().clone()
    opt1 = UnifiedV14Optimizer(
        [{"params": [z1], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=0.0061,
    )
    g_mix = torch.zeros(16, 4)
    g_mix[1] = torch.tensor([1e-5, -2e-5, 0.5e-5, 1.5e-5])
    g_mix[2] = float("nan")
    g_mix[3] = float("inf")
    g_mix[4] = float("-inf")
    g_mix[5] = torch.tensor([2e-5, 1e-5, -1e-5, 0.5e-5])
    if not use_fallback:
        opt1.set_sampled_rows(z1, torch.tensor([0, 1, 2, 1, 3, 4, 5, 5], dtype=torch.long))
    z1.grad = g_mix
    opt1.step()
    assert torch.isfinite(z1.detach()).all()
    assert torch.isfinite(opt1.state[z1]["peak_rms"])
    assert float(opt1.state[z1]["peak_rms"].item()) > 0.0
    # Zero / NaN / Inf rows are strictly untouched; active rows 1 & 5 moved cleanly:
    for r_dead in (0, 2, 3, 4):
        assert torch.equal(z1.detach()[r_dead], z1_orig[r_dead])
    for r_live in (1, 5):
        disp_r = float((z1.detach()[r_live] - z1_orig[r_live]).norm().item())
        assert disp_r > 0.015, f"Active row {r_live} under-moved: {disp_r}"

    # Finite 1e30 extreme-scale row updates cleanly via guarded amax rescaling without FP32 overflow:
    z_big = nn.Parameter(z1_orig.clone())
    opt_big = UnifiedV14Optimizer(
        [{"params": [z_big], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=0.0061,
    )
    g_big = torch.zeros(16, 4)
    g_big[1] = torch.tensor([1e30, -2e30, 0.5e30, 1.5e30])
    if not use_fallback:
        opt_big.set_sampled_rows(z_big, torch.tensor([0, 1, 1, 5], dtype=torch.long))
    z_big.grad = g_big
    opt_big.step()
    assert torch.isfinite(z_big.detach()).all()
    assert torch.isfinite(opt_big.state[z_big]["peak_rms"])
    assert float(opt_big.state[z_big]["peak_rms"].item()) > 1e29
    assert float((z_big.detach()[1] - z1_orig[1]).norm().item()) == pytest.approx(0.04, rel=1e-5)

    # Finite 3e38 overflow-norm row (||[3e38]*4|| = 6e38 > FLT_MAX -> inf) must not produce NaN or poison peak_rms:
    pk_before_ovf = float(opt1.state[z1]["peak_rms"].item())
    g_ovf = torch.zeros(16, 4)
    g_ovf[6] = 3e38
    if not use_fallback:
        opt1.set_sampled_rows(z1, torch.tensor([6, 6, 1], dtype=torch.long))
    z1.grad = g_ovf
    opt1.step()
    assert torch.isfinite(z1.detach()).all()
    assert torch.isfinite(opt1.state[z1]["peak_rms"])
    assert float(opt1.state[z1]["peak_rms"].item()) == pytest.approx(pk_before_ovf, rel=1e-6)

    # 3. optimizer_smoothing == 0.0 exact limit matches smoothing -> 0+:
    z_s0 = nn.Parameter(z1_orig.clone())
    z_st = nn.Parameter(z1_orig.clone())
    opt_s0 = UnifiedV14Optimizer(
        [{"params": [z_s0], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=0.0,
    )
    opt_st = UnifiedV14Optimizer(
        [{"params": [z_st], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=1e-12,
    )
    g_tiny = torch.randn(16, 4) * 1e-5
    rows_sub = torch.tensor([1, 4, 7, 4, 9], dtype=torch.long)
    if not use_fallback:
        opt_s0.set_sampled_rows(z_s0, rows_sub)
        opt_st.set_sampled_rows(z_st, rows_sub)
    z_s0.grad = g_tiny.clone()
    z_st.grad = g_tiny.clone()
    opt_s0.step()
    opt_st.step()
    assert torch.allclose(z_s0.detach(), z_st.detach(), atol=1e-6, rtol=1e-6)
    # Verify exact unit step norm == lr on active rows when smoothing == 0.0:
    active_rows = torch.arange(16) if use_fallback else torch.unique(rows_sub)
    step_norms = (z_s0.detach()[active_rows] - z1_orig[active_rows]).norm(dim=-1)
    assert torch.allclose(step_norms, torch.full_like(step_norms, 0.04), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("use_fallback", [False, True])
def test_unified_v14_prior_peak_rms_live_semantic_control(use_fallback: bool) -> None:
    """Semantic control: v14 unfreezes 1e-5 high-D prior rows (vs frozen v12) and actively uses st['peak_rms']."""
    torch.manual_seed(5)
    num_p, z_dim = 400, 32
    rows = torch.arange(256, dtype=torch.long)
    g_small = torch.randn(num_p, z_dim)
    g_small = g_small / g_small.norm(dim=-1, keepdim=True) * 1e-5  # exact 1e-5 per-row norm

    # 1. Direct v14 vs v12 comparison at 1e-5 row gradient norm (s_cap = 0.0061 * 128 / 256 = 0.00305):
    z_v14 = nn.Parameter(torch.zeros(num_p, z_dim))
    z_v12 = nn.Parameter(torch.zeros(num_p, z_dim))
    opt_v14 = UnifiedV14Optimizer(
        [{"params": [z_v14], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=0.0061,
    )
    opt_v12 = UnifiedV12Optimizer(
        [{"params": [z_v12], "role": "prior", "lr": 0.04, "sampled_rows_required": not use_fallback}],
        smoothing=0.0061,
    )
    if not use_fallback:
        opt_v14.set_sampled_rows(z_v14, rows)
        opt_v12.set_sampled_rows(z_v12, rows)
    z_v14.grad = g_small.clone()
    z_v12.grad = g_small.clone()
    opt_v14.step()
    opt_v12.step()

    eval_rows = torch.arange(num_p) if use_fallback else rows
    disp_v14 = float(z_v14.detach()[eval_rows].norm(dim=-1).mean().item())
    disp_v12 = float(z_v12.detach()[eval_rows].norm(dim=-1).mean().item())
    # v14 moves at lr / sqrt(1 + 2.0^2) ~= 0.01789, whereas v12 is crushed by s_cap (~1.3e-4):
    assert disp_v14 > 0.017, f"v14 prior crushed: {disp_v14:.6f}"
    assert disp_v12 < 0.0005, f"Expected v12 control to exhibit prior freeze: {disp_v12:.6f}"
    assert disp_v14 / max(disp_v12, 1e-12) > 80.0

    # 2. Deliberate state perturbation proves st["peak_rms"] is live (not dead code!):
    with torch.no_grad():
        z_v14.zero_()
        opt_v14.state[z_v14]["peak_rms"].fill_(1e-3)  # 100x higher peak_rms -> s_p_row = 2e-3 >> 1e-5
    if not use_fallback:
        opt_v14.set_sampled_rows(z_v14, rows)
    z_v14.grad = g_small.clone()
    opt_v14.step()
    disp_damped = float(z_v14.detach()[eval_rows].norm(dim=-1).mean().item())
    assert disp_damped < 0.02 * disp_v14, (
        f"Perturbing peak_rms failed to damp prior update (dead-code check failed): {disp_damped:.6f} vs {disp_v14:.6f}"
    )


def test_unified_v14_prior_checkpoint_tamper_and_fresh_process_continuation(tmp_path: Path) -> None:
    """Tamper rejection + bit-exact fresh-process continuation across active sub-s_cap peak_rms prior updates."""
    torch.manual_seed(0)

    # 1. Tamper rejection on prior peak_rms and m_prior:
    z_t = nn.Parameter(torch.randn(16, 4))
    opt_t = UnifiedV14Optimizer([{"params": [z_t], "role": "prior", "lr": 0.04, "betas": (0.75, 0.999)}], smoothing=0.0061)
    opt_t.set_sampled_rows(z_t, torch.tensor([0, 2, 5, 2], dtype=torch.long))
    z_t.grad = torch.randn_like(z_t) * 1e-5
    opt_t.step()
    good_sd = opt_t.state_dict()
    opt_t.validate_state_dict(good_sd)
    st_key = next(iter(good_sd["state"]))
    for bad_pk in (torch.tensor(-1.0), torch.tensor([1e-5]), torch.tensor(float("nan"))):
        bad_sd = deepcopy(good_sd)
        bad_sd["state"][st_key]["peak_rms"] = bad_pk
        with pytest.raises(ValueError, match="peak_rms"):
            opt_t.validate_state_dict(bad_sd)
    bad_mp = deepcopy(good_sd)
    bad_mp["state"][st_key]["m_prior"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="m_prior"):
        opt_t.validate_state_dict(bad_mp)

    # 2. Fresh-process bit-exact continuation of UnifiedV14Optimizer across active tiny-grad (1e-5) prior steps
    #    (both zero-sync duplicate rows and fallback row_scales):
    z_zs = nn.Parameter(torch.randn(64, 8))
    z_fb = nn.Parameter(torch.randn(64, 8))
    opt_ref = UnifiedV14Optimizer(
        [
            {"params": [z_zs], "role": "prior", "lr": 0.04, "betas": (0.75, 0.999)},
            {"params": [z_fb], "role": "prior", "lr": 0.04, "betas": (0.75, 0.999)},
        ],
        smoothing=0.0061,
    )
    step_data = []
    for i in range(8):
        gen = torch.Generator().manual_seed(500 + i)
        scale_i = 2e-5 if i < 4 else 5e-6  # drops below peak_rms across checkpoint boundary!
        g_zs = torch.randn(64, 8, generator=gen) * scale_i
        g_fb = torch.randn(64, 8, generator=gen) * scale_i
        r_zs = torch.randint(0, 64, (48,), generator=gen)
        r_fb = torch.unique(torch.randint(0, 64, (48,), generator=gen), sorted=True)
        sc_fb = torch.rand(int(r_fb.numel()), generator=gen) + 0.5
        step_data.append((g_zs, g_fb, r_zs, r_fb, sc_fb))

    for i in range(4):
        g_zs, g_fb, r_zs, r_fb, sc_fb = step_data[i]
        z_zs.grad, z_fb.grad = g_zs.clone(), g_fb.clone()
        opt_ref.set_sampled_rows(z_zs, r_zs)
        opt_ref.set_sampled_rows(z_fb, r_fb)
        opt_ref.set_row_scales(z_fb, sc_fb)
        opt_ref.step()

    assert 0.0 < float(opt_ref.state[z_zs]["peak_rms"].item()) < 1e-3
    assert 0.0 < float(opt_ref.state[z_fb]["peak_rms"].item()) < 1e-3
    ckpt_opt = tmp_path / "ckpt_v14_prior_opt.pt"
    out_opt = tmp_path / "out_v14_prior_opt.pt"
    torch.save(
        {
            "z_zs": z_zs.detach().clone(),
            "z_fb": z_fb.detach().clone(),
            "opt": opt_ref.state_dict(),
            "steps": step_data[4:],
        },
        ckpt_opt,
    )
    for i in range(4, 8):
        g_zs, g_fb, r_zs, r_fb, sc_fb = step_data[i]
        z_zs.grad, z_fb.grad = g_zs.clone(), g_fb.clone()
        opt_ref.set_sampled_rows(z_zs, r_zs)
        opt_ref.set_sampled_rows(z_fb, r_fb)
        opt_ref.set_row_scales(z_fb, sc_fb)
        opt_ref.step()

    script_opt = f"""
import torch
from torch import nn
from particlegan._unified_v14 import UnifiedV14Optimizer

pl = torch.load({str(ckpt_opt)!r}, weights_only=False)
z_zs = nn.Parameter(pl["z_zs"].clone())
z_fb = nn.Parameter(pl["z_fb"].clone())
opt = UnifiedV14Optimizer(
    [
        dict(params=[z_zs], role="prior", lr=0.04, betas=(0.75, 0.999)),
        dict(params=[z_fb], role="prior", lr=0.04, betas=(0.75, 0.999)),
    ],
    smoothing=0.0061,
)
opt.load_state_dict(pl["opt"])
for g_zs, g_fb, r_zs, r_fb, sc_fb in pl["steps"]:
    z_zs.grad, z_fb.grad = g_zs.clone(), g_fb.clone()
    opt.set_sampled_rows(z_zs, r_zs)
    opt.set_sampled_rows(z_fb, r_fb)
    opt.set_row_scales(z_fb, sc_fb)
    opt.step()
torch.save(
    {{
        "z_zs": z_zs.detach(),
        "z_fb": z_fb.detach(),
        "pk_zs": opt.state[z_zs]["peak_rms"],
        "pk_fb": opt.state[z_fb]["peak_rms"],
    }},
    {str(out_opt)!r},
)
"""
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, "-c", script_opt], check=True, env=env)
    res_opt = torch.load(out_opt, weights_only=False)
    assert torch.equal(z_zs.detach(), res_opt["z_zs"])
    assert torch.equal(z_fb.detach(), res_opt["z_fb"])
    assert torch.equal(opt_ref.state[z_zs]["peak_rms"], res_opt["pk_zs"])
    assert torch.equal(opt_ref.state[z_fb]["peak_rms"], res_opt["pk_fb"])

    # 3. End-to-end GANTrainer(get_recipe("unified_v14")) fresh-process continuation on high-D RMSNorm critic:
    rec = get_recipe(
        "unified_v14",
        prior_kind="mog",
        standardize=False,
        batch_size=16,
        num_particles=64,
        z_dim=8,
        total_steps=20,
        training_precision="float32",
    )
    g = init.deterministic_orthogonal_(_HighDMaskedGen(z_dim=8, out_dim=64), seed=10)
    d = init.deterministic_orthogonal_(_RMSNormCrit(in_dim=64), seed=11)
    p = init.deterministic_orthogonal_(MoGParticlePrior(num_particles=64, z_dim=8, sigma=0.05, standardize=False), seed=12)
    tr = GANTrainer(rec, g, d, prior=p, seed=0)
    assert isinstance(tr.opt_g, UnifiedV14Optimizer)
    batches = [torch.randn(16, 64, generator=torch.Generator().manual_seed(900 + i)) * 0.1 for i in range(8)]
    for i in range(4):
        tr.step(batches[i])
    ckpt_tr = tmp_path / "ckpt_v14_trainer.pt"
    out_tr = tmp_path / "out_v14_trainer.pt"
    torch.save({"state": tr.state_dict(), "batches": batches[4:]}, ckpt_tr)
    for i in range(4, 8):
        tr.step(batches[i])
    ref_sample = tr.sample(8)
    ref_models = tr.state_dict()["models"]
    ref_pk = tr.opt_g.state[tr.prior.z]["peak_rms"].detach().clone()

    script_tr = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v14_prior_invariance import _HighDMaskedGen, _RMSNormCrit

pl = torch.load({str(ckpt_tr)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(
    rec,
    _HighDMaskedGen(z_dim=8, out_dim=64),
    _RMSNormCrit(in_dim=64),
    prior=MoGParticlePrior(num_particles=64, z_dim=8, sigma=0.05, standardize=False),
    seed=777,
)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save(
    {{
        "sample": tr.sample(8),
        "models": tr.state_dict()["models"],
        "pk": tr.opt_g.state[tr.prior.z]["peak_rms"].detach(),
    }},
    {str(out_tr)!r},
)
"""
    subprocess.run([sys.executable, "-c", script_tr], check=True, env=env)
    res_tr = torch.load(out_tr, weights_only=False)
    assert torch.equal(ref_sample, res_tr["sample"])
    assert torch.equal(ref_pk, res_tr["pk"])
    for m_name in ("G", "D", "prior"):
        for k, v in ref_models[m_name].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, res_tr["models"][m_name][k]), f"Mismatch {m_name}.{k}"
            else:
                assert v == res_tr["models"][m_name][k]


def test_unified_v14_clockfree_audit(tmp_path: Path) -> None:
    import json
    from experiments.forge.clockfree import run_clockfree, verify_probe

    repo_p = Path(__file__).resolve().parents[1]
    task = json.loads((repo_p / "configs/forge/tasks/clockfree_audit.json").read_text())
    cand = json.loads((repo_p / "configs/forge/ideas/unified-v13.json").read_text())
    cand["id"] = "unified-v14-clockfree-test"
    cand["recipe_overrides"]["name"] = "unified-v14"
    cand["recipe_overrides"]["optimizer_family"] = "unified_v14_equil"
    audit_res = run_clockfree(
        {"candidate": cand, "protocol": {"seed": 0}},
        task,
        tmp_path / "clockfree_v14",
        "cpu",
    )
    comparisons, audit = verify_probe(task, audit_res["evidence"])
    assert audit["unexplained_clock_dependencies"] == []
    assert len(comparisons) == 4
    assert all(item["reference_sha256"] == item["changed_sha256"] for item in comparisons)

