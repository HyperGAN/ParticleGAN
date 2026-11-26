"""Comprehensive unit, invariant, host-sync, and fresh-process continuation tests for Axiom v5 (Track B)."""
from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

import particlegan as pg
from particlegan._axiom.critic import (
    AxiomPenaltyKernel,
    AxiomStepRecord,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    _eff_dim_t,
)
from particlegan._axiom.loss import AxiomGANLoss
from particlegan._axiom.optimizer import (
    AxiomOptimizer,
    _schatten6_decay_batched,
    fp32_tikhonov_ns_2d,
)
from particlegan._axiom.prior import (
    AxiomPriorRegularizer,
    central_pullback_cov,
)
from particlegan._axiom.transport import (
    AxiomTransportLoss,
    _deconflicted_routing,
    _orphan_macro_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AXIOM_DIR = REPO_ROOT / "particlegan" / "_axiom"
AXIOM_FILES = (
    AXIOM_DIR / "prior.py",
    AXIOM_DIR / "critic.py",
    AXIOM_DIR / "transport.py",
    AXIOM_DIR / "optimizer.py",
    AXIOM_DIR / "loss.py",
    AXIOM_DIR / "__init__.py",
    REPO_ROOT / "particlegan" / "axiom.py",
)
FORBIDDEN_IMPORTS = ("_unified", "_rhbi", "_t1_", "_t2_", "_t3_", "_t4_", "_minimal", "_bispf", "_empf")
FORBIDDEN_OPS = frozenset({"eigh", "eigvalsh", "svd", "svdvals", "pinv", "compile"})


# ---------------------------------------------------------------------------
# 1. Static Hygiene, LOC Caps, Recipe & Idea Contract
# ---------------------------------------------------------------------------


def test_v5_loc_caps_ast_hygiene_and_idea_contract() -> None:
    per_file_caps = {"critic.py": 380, "optimizer.py": 580, "prior.py": 470, "transport.py": 695}
    core_modules = {"prior", "critic", "transport", "optimizer"}
    total_loc = 0
    for path in AXIOM_FILES:
        src = path.read_text()
        n_lines = len(src.splitlines())
        total_loc += n_lines
        if path.name in per_file_caps:
            assert n_lines <= per_file_caps[path.name], f"{path.name} LOC={n_lines} > {per_file_caps[path.name]}"
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in FORBIDDEN_IMPORTS:
                        assert bad not in alias.name, f"{path.name} imports forbidden {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for bad in FORBIDDEN_IMPORTS:
                    assert bad not in mod, f"{path.name} imports from forbidden {mod}"
                if path.parent == AXIOM_DIR and path.stem in core_modules and node.level >= 1:
                    assert (mod.split(".")[0] if mod else "") not in core_modules
            elif isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_OPS, f"{path.name} calls forbidden .{node.attr}"
    assert total_loc <= 2500, f"Axiom total LOC={total_loc} > 2500"
    assert len(dataclasses.fields(pg.Recipe)) == 179
    assert pg.DEFAULT_RECIPE == "unified_v12"

    from experiments.forge.clockfree import source_audit
    from experiments.forge.contracts import validate_idea
    from experiments.forge.preflight import task_preflight
    from experiments.forge.views import load_tasks

    idea_v5 = json.loads((REPO_ROOT / "configs/forge/ideas/axiom-v5.json").read_text())
    idea_v4 = json.loads((REPO_ROOT / "configs/forge/ideas/axiom-v4.json").read_text())
    validate_idea(idea_v5)
    assert idea_v5["id"] == "axiom-v5"
    assert idea_v5["parent"] == "axiom-v4"
    assert idea_v4["id"] == "axiom-v4"
    tasks = load_tasks(REPO_ROOT)
    for tid in (
        "fl_law41_trust_rare1pct_tanh",
        "vector_spiral",
        "img_blobs4",
        "grid100_res16_tanh",
        "t3_iso_scale_mass_8x",
        "fl_law3_highd_orphan_d32",
        "fl_law2_groupnorm_and_vit_sprites",
        "dcgan_performance_r256",
        "bytegan_hypercube8",
    ):
        assert task_preflight(tasks[tid], idea_v5, {}, root=REPO_ROOT, tasks=tasks) == []

    rec = pg.get_recipe("axiom", total_steps=1000)
    assert source_audit(rec.to_dict(), {})["unexplained_clock_dependencies"] == []


# ---------------------------------------------------------------------------
# 2. B1: AxiomOptimizer Fixes (O1-O4)
# ---------------------------------------------------------------------------


def test_o1_equilibrium_smoothing_in_d2_and_slow_log2_ref_decay() -> None:
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(8, 16))
    rec = AxiomStepRecord()
    rec.thermal_scale = 1.0
    opt = AxiomOptimizer([w], lr=0.1, smoothing=0.05, record=rec)

    # Step 1 initializes log2_ref and prev_step_sketch
    g1 = torch.randn(8, 16)
    w.grad = g1.clone()
    opt.step()

    # Force orthogonal step (ema_step_cos ~ 0 -> s_rel == 0, g_eq == 1.0) and rat == 1.0
    opt._ema_step_cos_val, opt._ema_step_cos_t = 0.0, None
    opt._log2_damp_val, opt._log2_damp_t = 0.0, None
    g2 = torch.randn(8, 16)
    g2 = g2 * (g1.norm() / g2.norm())
    opt.state[w]["log2_ref"] = math.log2(float(g2.norm().item())) - 0.5 * math.log2(8.0)
    w_before = w.detach().clone()
    w.grad = g2.clone()
    opt.step()
    dw = w_before - w.detach()

    # O1: d2_it must include s_eq_base * g_eq = 0.05 (distinct from rs=0!)
    lr_eff = 0.1 * math.sqrt(1.0)
    dw_rs0 = fp32_tikhonov_ns_2d(g2, steps=5, rel_smooth=0.0) * lr_eff
    dw_rs005 = fp32_tikhonov_ns_2d(g2, steps=5, rel_smooth=0.05) * lr_eff
    assert float((dw - dw_rs005).abs().max().item()) < 1e-5
    assert float((dw - dw_rs0).abs().max().item()) > 3e-4

    # Slow log2_ref decay: dec <= 0.0007 (~1400-step half-life, NOT v4's 0.02!)
    ref_start = float(opt.state[w]["log2_ref"])
    for _ in range(100):
        opt._ema_step_cos_val, opt._ema_step_cos_t = 0.0, None
        w.grad = g2 * 0.01  # lt < log2_ref under g_eq == 1
        opt.step()
    drop_100 = ref_start - float(opt.state[w]["log2_ref"])
    assert 0.01 <= drop_100 <= 0.085, f"Expected slow 100-step log2_ref decay <= 0.085, got {drop_100:.4f}"


def test_o2_compound_critic_pathology_schatten6_and_dcouple() -> None:
    torch.manual_seed(1)
    # (a) Healthy shallow 3L critic -> 0 NS activations, fast foreach path
    crit3 = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
    rec3 = AxiomStepRecord()
    opt_d3 = AxiomOptimizer(crit3.parameters(), lr=0.01, critic=crit3, record=rec3)
    x = torch.randn(32, 2)
    for _ in range(6):
        opt_d3.zero_grad()
        crit3(x).square().mean().backward()
        opt_d3.step()
    assert opt_d3.ns_2d_activations == 0
    assert opt_d3.fast_path_steps == 6

    # (b) Deep 5L critic with rank-1 grads (r_stable == 1.0 < 1.05) BUT NO skew (uniform layer norms) -> 0 NS!
    crit5_unif = nn.Sequential(
        nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(),
        nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1),
    )
    opt_u = AxiomOptimizer(crit5_unif.parameters(), lr=0.01, critic=crit5_unif, record=AxiomStepRecord())
    u = torch.randn(16, 1)
    v = torch.randn(1, 16)
    r1 = (u @ v) / (u @ v).norm()
    for _ in range(4):
        opt_u.zero_grad()
        for m in crit5_unif:
            if isinstance(m, nn.Linear):
                m.weight.grad = r1.clone() if m.weight.shape == (16, 16) else torch.ones_like(m.weight) * 0.1
                m.bias.grad = torch.ones_like(m.bias) * 0.1
        opt_u.step()
    assert opt_u.ns_2d_activations == 0, "Uniform-norm rank-1 critic must NOT trigger compound NS detector"

    # (c) Deep 5L critic with BOTH >4x inter-layer contraction skew AND r_stable < 1.05 -> triggers NS!
    crit5_bad = deepcopy(crit5_unif)
    opt_b = AxiomOptimizer(crit5_bad.parameters(), lr=0.01, critic=crit5_bad, record=AxiomStepRecord())
    scales = [1.0, 0.25, 0.05, 0.005]  # 200x contraction across layers (> 4x)
    idx = 0
    for m in crit5_bad:
        if isinstance(m, nn.Linear):
            if m.weight.shape == (16, 16):
                m.weight.grad = r1 * scales[idx]
                idx += 1
            else:
                m.weight.grad = torch.ones_like(m.weight) * 0.1
            m.bias.grad = torch.ones_like(m.bias) * 0.1
    opt_b.step()
    assert opt_b.ns_2d_activations >= 4, "Contracted + rank-collapsed deep critic MUST trigger NS"
    assert opt_b._crit_skew_ema > 2.0

    # (d) Schatten-6 spectral ceiling: exact 0.0 deadband below tau=2.0, active contraction above tau=2.0
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    w_below = (q * 1.5).unsqueeze(0)
    w_above = (q * 3.5).unsqueeze(0)
    d_below = _schatten6_decay_batched(w_below, torch.tensor([0.05]), tau=2.0)
    d_above = _schatten6_decay_batched(w_above, torch.tensor([0.05]), tau=2.0)
    assert float(d_below.abs().max().item()) == 0.0
    assert float((w_above + d_above).norm().item()) < float(w_above.norm().item())

    # (e) d_couple bandpass: reduces critic step inside ema_margin in (0.28, 1.40) when g_func_gain < 0.75
    def _step_norm_at(margin: float, g_fg: float) -> float:
        torch.manual_seed(5)
        c = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        rec = AxiomStepRecord()
        rec.margin_initialized = True
        rec.ema_margin = margin
        rec.g_func_gain = g_fg
        rec.thermal_scale = 0.0  # isolate d_couple without schatten-6 decay
        opt = AxiomOptimizer(c.parameters(), lr=0.05, critic=c, record=rec)
        # Keep thermal_scale = 1.0 but zero-norm small weights so s6 is in deadband
        rec.thermal_scale = 1.0
        with torch.no_grad():
            for p in c.parameters():
                p.mul_(0.1)
        w0 = c[0].weight.detach().clone()
        c(torch.randn(16, 4)).mean().backward()
        opt.step()
        return float((c[0].weight.detach() - w0).norm().item())

    n_out_low = _step_norm_at(margin=0.0, g_fg=0.25)
    n_in_band = _step_norm_at(margin=0.80, g_fg=0.25)
    n_out_high = _step_norm_at(margin=2.0, g_fg=0.25)
    assert n_in_band < 0.35 * n_out_low
    assert math.isclose(n_out_low, n_out_high, rel_tol=1e-4)


def test_o3_batch_rms_trust_verified_wakeup_and_flip_damping() -> None:
    # (a) 1% rare-mode jump does NOT collapse global _ema_func_gain under batch-RMS trust
    class RareJumpGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w_bulk = nn.Parameter(torch.eye(2))
            self.w_rare = nn.Parameter(torch.eye(2))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            is_rare = (z.norm(dim=-1, keepdim=True) > 4.0).float()
            return (1.0 - is_rare) * (z @ self.w_bulk) + is_rare * (z @ self.w_rare)

    gen = RareJumpGen()
    rec = AxiomStepRecord()
    rec.ema_data_scale = math.sqrt(2.0)  # coord_sc = 1.0
    rec.ema_push_res = 0.0  # d_max = 0.045
    rec.margin_initialized = True
    _attach_axiom_generator_hook(gen, rec)
    opt_g = AxiomOptimizer(gen.parameters(), lr=0.02, record=None)
    opt_g._axiom_record = rec

    z = torch.zeros(32, 2)
    z[:31] = 0.2
    z[31] = torch.tensor([5.0, 0.0])  # 1/32 rare sample moves ~0.10 (>0.045), bulk moves ~0.004
    _ = gen(z)
    gen.w_bulk.grad = torch.eye(2)
    gen.w_rare.grad = torch.eye(2)
    opt_g.step()
    assert opt_g.last_probe_consumed is True
    assert opt_g.trust_backtracks_total == 0
    assert opt_g._ema_func_gain == pytest.approx(1.0)

    # (b) Verified delta_g_min wakeup: amplifies slow linear steps, backtracks superlinear cubic safely
    class ScalarMap(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.th = nn.Parameter(torch.zeros(1, 2))
            self.fn = fn

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.fn(self.th).expand(z.shape[0], -1)

    for kind, fn, expect_amp in (("linear_slow", lambda t: 0.001 * t, True), ("cubic_sup", lambda t: 0.0005 * t.pow(3), False)):
        gm = ScalarMap(fn)
        rec_w = AxiomStepRecord()
        rec_w.ema_data_scale = math.sqrt(2.0)
        rec_w.ema_push_res = 0.96875  # d_max = 0.20, d_min = 0.012
        rec_w.thermal_scale = 1.0
        _attach_axiom_generator_hook(gm, rec_w)
        opt_w = AxiomOptimizer(gm.parameters(), lr=math.sqrt(2.0))
        opt_w._axiom_record = rec_w
        z16 = torch.ones(16, 2)
        y0 = gm(z16).detach().clone()
        gm.th.grad = torch.tensor([[-1.0, -1.0]])
        opt_w.step()
        with torch.no_grad():
            rms = float((gm(z16) - y0).square().mean().sqrt().item())
        assert rms <= 0.20 * 1.005, f"{kind} violated d_max: rms={rms:.5f}"
        if expect_amp:
            assert opt_w.last_step_gain > 1.5, f"Expected wakeup amplification on {kind}, got {opt_w.last_step_gain}"

    # (c) Flip damping threshold aligned to cos < -0.45 (active at -0.55)
    p = nn.Parameter(torch.randn(8, 8))
    opt_f = AxiomOptimizer([p], lr=0.05)
    opt_f._ema_step_cos_val, opt_f._ema_step_cos_t = -0.55, None
    opt_f._log2_damp_val, opt_f._log2_damp_t = 0.0, None
    opt_f._prev_step_sketch = torch.zeros(128)
    p.grad = torch.randn(8, 8)
    opt_f.step()
    assert opt_f.log2_damp < -0.01, f"Expected flip damping at cos=-0.55 (< -0.45), got log2_damp={opt_f.log2_damp}"


def test_o4_scalar_parameter_damping_and_dead_hook_removed() -> None:
    assert not hasattr(AxiomOptimizer([nn.Parameter(torch.zeros(1))]), "_origin_norm_t")
    # 1D scalar parameter (e.g. img_blobs4 output bias) damped at fac=0.25 + reference smoothing even at cos=+1
    b_scalar = nn.Parameter(torch.zeros(1))
    b_vec = nn.Parameter(torch.zeros(4))
    opt_s = AxiomOptimizer([b_scalar], lr=0.1, smoothing=0.05)
    opt_v = AxiomOptimizer([b_vec], lr=0.1, smoothing=0.05)
    opt_s.bind_origin_sample(None)  # protocol signature preserved

    # Step 1 (no history yet): scalar moves 0.25*lr vs vector 1.0*lr
    b_scalar.grad = torch.tensor([1.0])
    b_vec.grad = torch.tensor([1.0, 0.0, 0.0, 0.0])
    opt_s.step()
    opt_v.step()
    assert math.isclose(abs(float(b_scalar.item())), 0.025, rel_tol=1e-4)
    assert math.isclose(abs(float(b_vec[0].item())), 0.10, rel_tol=1e-4)

    # Step 2 with coherent sign (cos=+1 -> g_eq == 0) & same magnitude: scalar still smoothed by 1/sqrt(1 + 0.25^2)
    s_before = float(b_scalar.item())
    b_scalar.grad = torch.tensor([1.0])
    opt_s.step()
    step2 = abs(float(b_scalar.item()) - s_before)
    expected_step2 = 0.025 / math.sqrt(1.0 + 0.25 * 0.25)
    assert math.isclose(step2, expected_step2, rel_tol=1e-3), f"got {step2}, expected {expected_step2}"

    # Degree-0 scale invariance across 1e-6 vs 1e+6 gradients on scalar parameter
    moves = []
    for scale in (1e-6, 1.0, 1e6):
        ps = nn.Parameter(torch.zeros(1))
        os_ = AxiomOptimizer([ps], lr=0.1, smoothing=0.05)
        for _ in range(5):
            ps.grad = torch.tensor([1.0 * scale])
            os_.step()
        moves.append(float(ps.item()))
    assert math.isclose(moves[0], moves[1], rel_tol=1e-4)
    assert math.isclose(moves[1], moves[2], rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 3. B2: Prior P1-P2 & 4 Scale-Equivariance Fixes (1e-5x to 1e+5x)
# ---------------------------------------------------------------------------


def test_p1_proto_hits_cooling_and_deadband_removal() -> None:
    torch.manual_seed(0)
    reg = AxiomPriorRegularizer(max_prototypes=16)
    base = torch.randn(64, 2) * 0.2
    reg.update_quantizer(base)
    assert reg.proto_hits is not None
    hits_0 = reg.proto_hits.clone()
    for _ in range(5):
        reg.update_quantizer(base + 0.01 * torch.randn_like(base))
    assert float(reg.proto_hits.mean().item()) > float(hits_0.mean().item()) + 5.0
    # Shock cools/resets proto_hits downward
    hits_pre_shock = float(reg.proto_hits.mean().item())
    reg.update_quantizer(base + 100.0)
    assert float(reg.proto_hits.mean().item()) < hits_pre_shock

    # Source check: deadband (0.15..0.27), 0.995 variance shrink, and 0.55 iso_gate removed
    src = (AXIOM_DIR / "prior.py").read_text()
    assert "0.995 - rho_mu" not in src
    assert "0.55 * ((iso_err" not in src
    assert "else 0.28) * ((iso_err_ema_t - 0.22) / 0.45)" in src


def test_p2_unbiased_two_sketch_u_stat_d32_no_jensen_shrinkage() -> None:
    # At d_z=32, D=32 (Haar-rotated), single-sketch collapses eigenvalues to 0.25*v;
    # two-independent-sketch U-statistic converges to ~1.0*v with eig_min/v >= 0.50!
    torch.manual_seed(42)
    d = 32
    q, _ = torch.linalg.qr(torch.randn(d, d))
    v_target = 0.04
    w = nn.Parameter(q * math.sqrt(v_target) * 0.35)  # start under-dispersed
    opt = torch.optim.SGD([w], lr=0.02)
    mu = torch.zeros(64, d)
    eye = torch.eye(d)[None]
    for step in range(160):
        opt.zero_grad()
        phi = (step * 0.6180339887498949) % 1.0
        sp1, sp2 = central_pullback_cov(lambda z: z @ w.T, mu, sigma_z=1.0, phase=phi, return_pair=True)
        sp_avg = 0.5 * (sp1 + sp2)
        tr_m = (torch.diagonal(sp_avg, dim1=-2, dim2=-1).sum(-1) / float(d)).clamp_min(1e-24)
        iso_rad = ((tr_m.sqrt() - math.sqrt(v_target)) / math.sqrt(v_target)).square().mean()
        iso_cov = (((sp1 - v_target * eye) / v_target) * ((sp2 - v_target * eye) / v_target)).sum((1, 2)).mean() / float(d)
        (0.45 * iso_rad + 0.55 * iso_cov).backward()
        opt.step()
    with torch.no_grad():
        cov = w @ w.T
        eigs = torch.linalg.eigvalsh(cov)
    assert float(eigs.median().item()) / v_target > 0.90, f"2-sketch median eig collapsed: {eigs.median().item()/v_target:.3f}"
    assert float(eigs.min().item()) / v_target > 0.70, f"2-sketch min eig collapsed: {eigs.min().item()/v_target:.3f}"

    # Zero-mean _box_muller_targets centering for cnt >= 2 preserves mode center c_bar
    from particlegan._axiom.prior import _box_muller_targets
    c_bar = torch.tensor([[1.5, -0.5], [0.0, 0.0]])
    v_mu = torch.ones(2)
    mc_flag = torch.tensor([True, True])
    t_pair = _box_muller_targets(torch.tensor([3, 7]), torch.tensor([0, 0]), torch.tensor([0, 1]), 2, c_bar, v_mu, mc_flag, 2)
    assert torch.allclose(t_pair.mean(0), c_bar[0], atol=1e-5), "_box_muller_targets must center cnt>=2 targets around c_bar"
    assert (t_pair[0] - t_pair[1]).norm().item() > 0.1

    # Multi-scale _update_charts with loc_ell avoids shattering broad clusters into tight-scale sub-charts
    from particlegan._axiom.transport import _update_charts
    torch.manual_seed(0)
    y_ms = torch.cat((torch.randn(128, 2) * 0.02, torch.randn(128, 2) * 0.25 + 4.0), dim=0)
    loc_e = torch.cat((torch.full((128,), 0.01), torch.full((128,), 0.12)), dim=0)
    cp_no, _, _, _, _ = _update_charts(y_ms, None, max_charts=256, allow_update=True, loc_ell=None)
    cp_ms, _, _, _, _ = _update_charts(y_ms, None, max_charts=256, allow_update=True, loc_ell=loc_e)
    assert cp_ms.shape[0] < cp_no.shape[0] // 2, f"loc_ell failed to prevent broad-cluster over-shattering: {cp_ms.shape[0]} vs {cp_no.shape[0]}"


def test_b2_scale_equivariance_across_ten_decades_and_shock_resets() -> None:
    # (a) Laguerre psi_cap exact homogeneity across c in {1e-5, 1.0, 1e+5} (no unscaled +0.5)
    psis = []
    for c in (1e-5, 1.0, 1e5):
        reg = AxiomPriorRegularizer(max_prototypes=8)
        reg.quantizer_initialized = True
        reg.proto_c = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) * c
        reg.proto_var = torch.full((4,), 0.05 * c * c)
        reg.proto_w = torch.full((4,), 0.25)
        reg.proto_hits = torch.ones(4)
        reg.w_fake_ema = torch.tensor([0.70, 0.10, 0.10, 0.10])
        reg.psi = torch.zeros(4)
        reg.v_push = reg.proto_var.clone()
        xu = torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.02, 0.02]]) * c
        reg.update_laguerre_dual(xu, reg.proto_c, torch.eye(4), torch.tensor(0.5 * c), laguerre_lr=0.15)
        psis.append(reg.psi / (c * c))
    assert torch.allclose(psis[0], psis[1], rtol=1e-4, atol=1e-6)
    assert torch.allclose(psis[1], psis[2], rtol=1e-4, atol=1e-6)

    # (b) proto_var geometric shock reset recovers 1e4x scale shift (1e8x variance) in <= 5 steps
    torch.manual_seed(0)
    reg_s = AxiomPriorRegularizer(max_prototypes=16)
    u = torch.randn(64, 2)
    for _ in range(5):
        reg_s.update_quantizer(u * 1e-1)
    v_before = float(reg_s.proto_var.median().item())
    for _ in range(5):
        reg_s.update_quantizer(u * 1e-5)
    v_after = float(reg_s.proto_var.median().item())
    assert v_after < v_before * 1e-6, f"proto_var failed shock reset: before={v_before:.2e}, after={v_after:.2e}"

    # (c) ema_data_scale: 1-step spike immunity (<= 1.04x) AND multi-step innovation jump recovers 1e4x in <= 10 steps
    rec = AxiomStepRecord()
    x1 = torch.randn(32, 2)
    z0, z1 = torch.zeros(32, 1), torch.ones(32, 1)
    rec.observe_margin_and_scale(z1, z0, x1, x1)
    s0 = float(rec.ema_data_scale)
    # 1-step transient spike must stay bounded <= 1.04x
    rec.observe_margin_and_scale(z1, z0, x1 * 1e4, x1 * 1e4)
    assert float(rec.ema_data_scale) <= s0 * 1.04
    # Persistent shift (2+ consecutive batches) triggers innovation jump and recovers in <= 10 steps
    for _ in range(9):
        rec.observe_margin_and_scale(z1, z0, x1 * 1e4, x1 * 1e4)
    assert math.isclose(float(rec.ema_data_scale), s0 * 1e4, rel_tol=0.10)

    # (d) AxiomPenaltyKernel scale-equivariance (s_coord * ||grad D|| <= kappa) over 1e-5 .. 1e+5
    torch.manual_seed(11)
    w_lin = torch.randn(4, 1)
    xr0, xf0 = torch.randn(16, 4), torch.randn(16, 4)
    pens, shats = [], []
    for c in (1e-5, 1.0, 1e5):
        rec_p = AxiomStepRecord()
        rec_p.observe_margin_and_scale(torch.ones(16, 1), -torch.ones(16, 1), xr0 * c, xf0 * c)
        kern = AxiomPenaltyKernel(record=rec_p, coeff=1.0, kappa=1.0)
        # Scale-equivariant critic D_c(x) = D_1(x / c)
        pen, st = kern.penalty(lambda x, _c=c: (x / _c) @ w_lin, xr0 * c, xf0 * c, collect_stats=True)
        pens.append(float(pen.item()))
        shats.append(float(st["s_hat"]))
    assert math.isclose(shats[0], shats[1], rel_tol=1e-3)
    assert math.isclose(shats[1], shats[2], rel_tol=1e-3)
    assert math.isclose(pens[0], pens[1], rel_tol=1e-3)
    assert math.isclose(pens[1], pens[2], rel_tol=1e-3)

    # (d2) Intrinsic effective-dimension scale s_intr = ds / sqrt(d_eff): O(D)-invariant,
    #      dimension-invariant on isotropic R^64, and recovers sparse rank ~3 on 4-blob 8x8 data.
    q_rot, _ = torch.linalg.qr(torch.randn(16, 16))
    x_sub = torch.randn(32, 3) @ torch.randn(3, 16)
    de_base = float(_eff_dim_t(x_sub, 16).item())
    de_rot = float(_eff_dim_t(x_sub @ q_rot, 16).item())
    de_scaled = float(_eff_dim_t(x_sub * 1e5 + 7.0, 16).item())
    assert math.isclose(de_base, de_rot, rel_tol=1e-4)
    assert math.isclose(de_base, de_scaled, rel_tol=1e-4)
    iso_de = sum(float(_eff_dim_t(torch.randn(16, 64), 64).item()) for _ in range(16)) / 16.0
    assert 52.0 <= iso_de <= 64.0, f"Isotropic R^64 d_eff biased: {iso_de:.2f}"
    blobs = torch.zeros(4, 8, 8)
    blobs[0, 1:3, 1:3] = 1.0; blobs[1, 1:3, 5:7] = 1.0; blobs[2, 5:7, 1:3] = 1.0; blobs[3, 5:7, 5:7] = 1.0
    xb = blobs[torch.arange(64) % 4]
    de_blobs = float(_eff_dim_t(xb, 64).item())
    assert 2.5 <= de_blobs <= 3.5, f"Sparse 4-blob 8x8 d_eff should recover centered rank ~3, got {de_blobs:.2f}"

    # (e) Pullback prior row-Adam scale equivariance across c in {1e-5, 1.0, 1e5} (no +1e-5 stall)
    dzs = []
    for c in (1e-5, 1.0, 1e5):
        torch.manual_seed(3)
        recipe = pg.get_recipe("axiom", prior_kind="mog", sigma_rel=0.05, num_particles=16, z_dim=2, batch_size=16, total_steps=50)
        class ScaledG(nn.Module):
            def __init__(self, sc: float):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2))
                self.sc = sc
            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.net(z) * self.sc
        class ScaledD(nn.Module):
            def __init__(self, sc: float):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
                self.sc = sc
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x / self.sc).squeeze(-1)
        g, d, prior = ScaledG(c), ScaledD(c), recipe.make_prior()
        for idx, m in enumerate((g, d, prior)):
            pg.init.deterministic_orthogonal_(m, seed=idx + 5)
        tr = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=50, seed=0)
        z0_tab = prior.z.detach().clone()
        rng = torch.Generator().manual_seed(7)
        for _ in range(3):
            yb = (torch.randn(16, 2, generator=rng) + 1.5) * c
            out = tr.step(yb)
            assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])
        dzs.append(float((prior.z.detach() - z0_tab).norm().item()))
    assert dzs[0] > 1e-4, f"1e-5x row-Adam stalled: dz={dzs[0]}"
    assert math.isclose(dzs[0], dzs[1], rel_tol=0.05), f"1e-5x vs 1x dz mismatch: {dzs}"
    assert math.isclose(dzs[1], dzs[2], rel_tol=0.05), f"1x vs 1e5x dz mismatch: {dzs}"


# ---------------------------------------------------------------------------
# 4. B3: Transport T1, Projected dim > 1024, and RoR Crossover
# ---------------------------------------------------------------------------


def test_b3_orphan_thresholds_weyl_tiebreak_projected_transport_and_ror_crossover() -> None:
    # (a) T1: _orphan_macro_loss silent at 2.0*rho (restored 2.80*rho) and active above 3.5*rho
    xp = torch.zeros(16, 2, requires_grad=True)
    yp = torch.cat((torch.zeros(12, 2), torch.full((4, 2), 20.0)), dim=0)
    prs = torch.full((16,), 1.0 / 16.0)
    pcs = torch.full((16,), 1.0 / 16.0)
    ch_below = torch.full((16, 16), 2.0 * 3.0)
    assert float(_orphan_macro_loss(xp, yp, ch_below, prs, pcs, rho=3.0, g_conn=0.0).item()) == 0.0
    ch_above = torch.full((16, 16), 4.5 * 3.0)
    # Give yp two distinct well-separated clusters so r_loc > 3.52 triggers
    yp2 = torch.cat((torch.zeros(8, 2) + 0.01 * torch.randn(8, 2), torch.full((8, 2), 10.0) + 0.01 * torch.randn(8, 2)), dim=0)
    xp2 = torch.zeros(16, 2, requires_grad=True)
    assert float(_orphan_macro_loss(xp2, yp2, ch_above, prs, pcs, rho=3.0, g_conn=0.0).item()) > 0.05

    # (b) T1 D-06 Weyl tie-break unbiasedness in _deconflicted_routing (equal weights & zero-distance symmetry)
    k = 16
    cf = torch.zeros(k, 2)
    xf_eq = torch.zeros(k // 2, 2)  # exact ties across all k charts -> quota rem = 8
    wr = torch.full((k,), 1.0 / k)
    rads = torch.full((k,), 0.1)
    bcnt = torch.ones(k)
    tgt, _, _, _, _ = _deconflicted_routing(
        xf_eq, cf, wr, rads, bcnt, wr.clone(), torch.tensor(0.1),
        wv_c=1.0, halo_mult=2.1, s_th=1.0, allow_update=False,
    )
    # With distinct chart coords at identical distance from origin (unit circle), check selected indices span both halves!
    ang = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / k), k)
    cf_ring = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)
    tgt_ring, _, _, _, _ = _deconflicted_routing(
        xf_eq, cf_ring, wr, rads, bcnt, wr.clone(), torch.tensor(0.1),
        wv_c=1.0, halo_mult=2.1, s_th=1.0, allow_update=False,
    )
    chosen = torch.cdist(tgt_ring, cf_ring).argmin(1)
    assert int((chosen >= k // 2).sum().item()) >= 2, f"D-06 low-index bias detected: chosen={chosen.tolist()}"

    # (c) T1 g_curve > 0.5 directed-edge sweep in simplicial_manifold_loss
    tr_loss = AxiomTransportLoss()
    t_lin = torch.linspace(0.0, 2.0, 32)
    y_curve = torch.stack((t_lin, torch.sin(t_lin)), dim=-1)
    z_tab = nn.Parameter(torch.randn(12, 2))
    g_lin = nn.Linear(2, 2)
    l_simp = tr_loss.simplicial_manifold_loss(z_tab, g_lin, y_curve, record=AxiomStepRecord())
    assert torch.isfinite(l_simp) and l_simp.requires_grad
    l_simp.backward()
    assert z_tab.grad is not None and torch.isfinite(z_tab.grad).all()

    # (d) Transport dim gate in AxiomGANLoss: 32x32x3 (3072 <= 4096) ACTIVATES transport (with no lossy thermal lock),
    #     while 64x64x3 (12288 > 4096) and flat 2D 2048 > 1024 skip transport for fast DCGAN throughput!
    rec_img = AxiomStepRecord()
    crit_img = nn.Sequential(nn.Conv2d(3, 4, 3, stride=2, padding=1), nn.Flatten(), nn.Linear(4 * 16 * 16, 1))
    _attach_axiom_critic_hook(crit_img, rec_img)
    loss_img = AxiomGANLoss(record=rec_img)
    xr_32 = torch.randn(8, 3, 32, 32)
    xf_32 = torch.randn(8, 3, 32, 32, requires_grad=True)
    dr_32, df_32 = crit_img(xr_32).squeeze(-1), crit_img(xf_32).squeeze(-1)
    lg_32 = loss_img.g_loss(df_32, dr_32)
    assert torch.isfinite(lg_32)
    assert rec_img.ell_initialized is True, "32x32x3 (raw_dim=3072 <= 4096) must activate AxiomTransportLoss"
    assert rec_img.thermal_initialized is False, "Lossy 32x32->8x8 downsampled transport must NOT trigger observe_thermal"

    rec_64 = AxiomStepRecord()
    crit_64 = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4, padding=1), nn.Flatten(), nn.Linear(4 * 16 * 16, 1))
    _attach_axiom_critic_hook(crit_64, rec_64)
    loss_64 = AxiomGANLoss(record=rec_64)
    xr_64, xf_64 = torch.randn(8, 3, 64, 64), torch.randn(8, 3, 64, 64, requires_grad=True)
    _ = loss_64.g_loss(crit_64(xf_64).squeeze(-1), crit_64(xr_64).squeeze(-1))
    assert rec_64.ell_initialized is False, "64x64x3 (raw_dim=12288 > 4096) must skip AxiomTransportLoss"

    # Flat 2D dim=2048 > 1024 still skips AxiomTransportLoss
    rec_flat = AxiomStepRecord()
    crit_flat = nn.Linear(2048, 1)
    _attach_axiom_critic_hook(crit_flat, rec_flat)
    loss_flat = AxiomGANLoss(record=rec_flat)
    xr_f, xf_f = torch.randn(8, 2048), torch.randn(8, 2048, requires_grad=True)
    _ = loss_flat.g_loss(crit_flat(xf_f).squeeze(-1), crit_flat(xr_f).squeeze(-1))
    assert rec_flat.ell_initialized is False

    # (e) AD Crossover: GroupNorm (even 128^2, with BN safety!) -> RoR; plain Conv2d (64^2 and 128^2) -> fwAD!
    c_bn64 = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4, padding=1), nn.BatchNorm2d(4), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    c_bn64.train()
    rm_before = c_bn64[1].running_mean.detach().clone()
    k_bn = AxiomPenaltyKernel(record=AxiomStepRecord())
    _, st_64 = k_bn.penalty(c_bn64, torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64), collect_stats=True, critic_module=c_bn64)
    assert st_64["regime"] == "fwad"
    assert c_bn64[1].training is True
    assert torch.equal(c_bn64[1].running_mean, rm_before), "fwAD must not mutate BatchNorm running stats"

    c_gn_bn64 = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4, padding=1), nn.GroupNorm(2, 4), nn.BatchNorm2d(4), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    c_gn_bn64.train()
    rm_gn_before = c_gn_bn64[2].running_mean.detach().clone()
    _, st_gn_bn = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(c_gn_bn64, torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64), collect_stats=True, critic_module=c_gn_bn64)
    assert st_gn_bn["regime"] == "ror" and c_gn_bn64[2].training is True
    assert torch.equal(c_gn_bn64[2].running_mean, rm_gn_before), "RoR must not mutate BatchNorm running stats"

    c_gn128 = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8, padding=1), nn.GroupNorm(2, 4), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    _, st_gn = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(c_gn128, torch.randn(4, 3, 128, 128), torch.randn(4, 3, 128, 128), collect_stats=True, critic_module=c_gn128)
    assert st_gn["regime"] == "ror"

    c_plain128 = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    _, st_fw = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(c_plain128, torch.randn(4, 3, 128, 128), torch.randn(4, 3, 128, 128), collect_stats=True, critic_module=c_plain128)
    assert st_fw["regime"] == "fwad"

    # Compact 4D image (1x8x8 = 64 <= 256) routes to exact 2B RoR!
    c_small8 = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 1))
    _, st_sm8 = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(c_small8, torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8), collect_stats=True, critic_module=c_small8)
    assert st_sm8["regime"] == "ror"


# ---------------------------------------------------------------------------
# 5. B4: Host Sync Count <= 8 Syncs/Step on grid100
# ---------------------------------------------------------------------------


def test_b4_grid100_host_syncs_at_most_8_per_step() -> None:
    from scripts.axiom_four_laws_benchmark import (
        SimpleMLPDiscriminator,
        SimpleMLPGenerator,
        _count_host_syncs,
        sample_100gaussians,
    )

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rec = pg.get_recipe("axiom", num_particles=1024, z_dim=2, prior_kind="mog", batch_size=256)
    gen = SimpleMLPGenerator(2, 128, n_hidden=3, out_dim=2).to(dev)
    crit = SimpleMLPDiscriminator(2, 128, n_hidden=3).to(dev)
    prior = pg.MoGParticlePrior(num_particles=1024, z_dim=2, sigma=0.025, device=dev)
    for m in (gen, crit, prior):
        pg.init.deterministic_orthogonal_(m, seed=0)
    trainer = pg.GANTrainer(rec, gen, crit, prior=prior, seed=0)
    rng = torch.Generator(device=dev).manual_seed(0)
    batches = [sample_100gaussians(256, dev, generator=rng) for _ in range(25)]
    for b in batches[:10]:
        trainer.step(b)
    with _count_host_syncs(dev) as sc:
        for b in batches[10:25]:
            trainer.step(b)
    # On CPU only, set_sampled_rows performs a CPU-only bounds guard (2 bool checks/step; skipped on CUDA).
    cpu_only_guard = 2.0 if dev.type == "cpu" else 0.0
    syncs_per_step = (sum(sc.values()) / 15.0) - cpu_only_guard
    assert syncs_per_step <= 8.0, f"grid100 syncs/step={syncs_per_step:.2f} exceeded <= 8.0 budget ({sc})"


# ---------------------------------------------------------------------------
# 6. Fresh-Process Exact Checkpoint Continuation & Strict Tamper Rejection
# ---------------------------------------------------------------------------


def test_v5_fresh_process_checkpoint_continuation_and_tamper_rejection(tmp_path: Path) -> None:
    # (a) Tamper & incompatible old-schema rejection on all new v5 checkpoint fields
    rec = AxiomStepRecord()
    rec.observe_margin_and_scale(torch.ones(8, 1), -torch.ones(8, 1), torch.randn(8, 2), torch.randn(8, 2))
    sd_rec = rec.state_dict()
    assert "last_data_scale" in sd_rec
    bad_rec = deepcopy(sd_rec)
    del bad_rec["last_data_scale"]
    with pytest.raises(ValueError):
        AxiomStepRecord().load_state_dict(bad_rec)
    bad_rec2 = deepcopy(sd_rec)
    bad_rec2["last_data_scale"] = -1.0
    with pytest.raises(ValueError):
        AxiomStepRecord().load_state_dict(bad_rec2)

    reg = AxiomPriorRegularizer(max_prototypes=8)
    reg.update_quantizer(torch.randn(32, 2))
    sd_reg = reg.state_dict_checkpoint()
    assert "proto_hits" in sd_reg
    bad_reg = deepcopy(sd_reg)
    del bad_reg["proto_hits"]
    with pytest.raises(ValueError):
        AxiomPriorRegularizer(max_prototypes=8).load_state_dict_checkpoint(bad_reg)
    bad_reg2 = deepcopy(sd_reg)
    bad_reg2["proto_hits"] = -torch.ones_like(bad_reg2["proto_hits"])
    with pytest.raises(ValueError):
        AxiomPriorRegularizer(max_prototypes=8).load_state_dict_checkpoint(bad_reg2)

    crit = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 1))
    opt = AxiomOptimizer(crit.parameters(), lr=0.01, critic=crit, record=AxiomStepRecord())
    crit(torch.randn(8, 8)).mean().backward()
    opt.step()
    sd_opt = opt.state_dict()
    assert "crit_skew_ema" in sd_opt["axiom"]
    bad_opt = deepcopy(sd_opt)
    del bad_opt["axiom"]["crit_skew_ema"]
    with pytest.raises(ValueError):
        opt.load_state_dict(bad_opt)
    bad_opt2 = deepcopy(sd_opt)
    bad_opt2["axiom"]["crit_skew_ema"] = -0.5
    with pytest.raises(ValueError):
        opt.load_state_dict(bad_opt2)

    # (b) Fresh-process exact continuation across quantizer/proto_hits/Laguerre/scale-shift/deep-critic-pathology
    ckpt_path = tmp_path / "v5_ckpt.pt"
    out_ref = tmp_path / "ref.pt"
    out_res = tmp_path / "res.pt"
    script = f"""
import sys, torch
from torch import nn
import particlegan as pg

mode = sys.argv[1]
def build():
    rec = pg.get_recipe("axiom", prior_kind="mog", sigma_rel=0.05, num_particles=32, z_dim=2, batch_size=32, total_steps=100)
    g = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 2))
    # 5-layer deep critic exercises compound pathology skew tracker & v_pow
    d = nn.Sequential(
        nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(),
        nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1),
    )
    prior = rec.make_prior()
    for i, m in enumerate((g, d, prior)):
        pg.init.deterministic_orthogonal_(m, seed=i * 13)
    return pg.GANTrainer(rec, g, d, prior=prior, max_steps=100, seed=0)

rng = torch.Generator().manual_seed(123)
batches = []
for t in range(10):
    sc = 1.0 if t < 4 else 50.0  # cross scale-shock & innovation-jump boundary!
    batches.append(torch.randn(32, 2, generator=rng) * sc)

tr = build()
if mode == "ref":
    for t in range(5):
        tr.step(batches[t])
    torch.save(tr.state_dict(), {str(ckpt_path)!r})
    for t in range(5, 10):
        tr.step(batches[t])
    torch.save({{
        "g": [p.detach().clone() for p in tr.G.parameters()],
        "d": [p.detach().clone() for p in tr.D.parameters()],
        "z": tr.prior.z.detach().clone(),
        "rec": tr.opt_d.record.state_dict(),
        "opt_d": tr.opt_d.state_dict()["axiom"],
        "reg": tr.prior_regularizer.state_dict_checkpoint(),
    }}, {str(out_ref)!r})
else:
    tr.load_state_dict(torch.load({str(ckpt_path)!r}, weights_only=False))
    for t in range(5, 10):
        tr.step(batches[t])
    torch.save({{
        "g": [p.detach().clone() for p in tr.G.parameters()],
        "d": [p.detach().clone() for p in tr.D.parameters()],
        "z": tr.prior.z.detach().clone(),
        "rec": tr.opt_d.record.state_dict(),
        "opt_d": tr.opt_d.state_dict()["axiom"],
        "reg": tr.prior_regularizer.state_dict_checkpoint(),
    }}, {str(out_res)!r})
"""
    for mode in ("ref", "resume"):
        proc = subprocess.run([sys.executable, "-c", script, mode], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"{mode} subprocess failed:\n{proc.stderr}\n{proc.stdout}"
    a = torch.load(out_ref, weights_only=False)
    b = torch.load(out_res, weights_only=False)
    for pa, pb in zip(a["g"], b["g"]):
        assert torch.equal(pa, pb), "Generator weights diverged after fresh-process resume"
    for pa, pb in zip(a["d"], b["d"]):
        assert torch.equal(pa, pb), "Critic weights diverged after fresh-process resume"
    assert torch.equal(a["z"], b["z"]), "Prior table diverged after fresh-process resume"
    assert a["rec"]["last_data_scale"] == b["rec"]["last_data_scale"]
    assert a["rec"]["ema_data_scale"] == b["rec"]["ema_data_scale"]
    assert a["opt_d"]["crit_skew_ema"] == b["opt_d"]["crit_skew_ema"]
    assert torch.equal(a["reg"]["proto_hits"], b["reg"]["proto_hits"])
    assert torch.equal(a["reg"]["proto_var"], b["reg"]["proto_var"])
