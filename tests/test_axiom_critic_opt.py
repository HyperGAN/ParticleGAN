"""Comprehensive unit and Four-Laws adversarial tests for Axiom Modules B (critic) & D (optimizer)."""
from __future__ import annotations

import ast
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import pytest
import torch
from torch import nn

from particlegan._axiom.critic import (
    AxiomCriticPenalty,
    AxiomPenaltyKernel,
    AxiomStepRecord,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    bind_axiom_record,
    eval_probe_forward,
)
from particlegan._axiom.optimizer import (
    AxiomOptimizer,
    collect_conv_layouts,
    fp32_tikhonov_ns_2d,
    fp32_tikhonov_ns_batched,
    mantissa_exponent_normalize,
    zero_pinv_project_nonascent,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CRITIC_PATH = REPO_ROOT / "particlegan" / "_axiom" / "critic.py"
OPTIMIZER_PATH = REPO_ROOT / "particlegan" / "_axiom" / "optimizer.py"

FORBIDDEN_IMPORT_PREFIXES = (
    "_unified",
    "_t1_",
    "_t2_",
    "_t3_",
    "_t4_",
    "_rhbi",
    "_empf",
    "_cond_fiber",
    "_minimal",
    "_bispf",
    "_nash_v3",
    "_r4_ablation",
    "bispf",
    "rhbi",
    "k3p",
    "grad_regularizers",
)
FORBIDDEN_ATTRS = frozenset({"eigh", "eigvalsh", "svd", "svdvals", "pinv", "compile"})


# ---------------------------------------------------------------------------
# 1. Static AST Hygiene, LOC Caps & Zero-RNG Consumption
# ---------------------------------------------------------------------------


def test_static_hygiene_loc_and_zero_forbidden_imports() -> None:
    critic_src = CRITIC_PATH.read_text(encoding="utf-8")
    opt_src = OPTIMIZER_PATH.read_text(encoding="utf-8")
    critic_loc = len(critic_src.splitlines())
    opt_loc = len(opt_src.splitlines())
    assert critic_loc <= 380, f"critic.py LOC={critic_loc} exceeds 380 cap"
    assert opt_loc <= 580, f"optimizer.py LOC={opt_loc} exceeds 580 cap"

    for name, src, forbidden_siblings, forbid_all_pg in (
        ("critic.py", critic_src, {"prior", "transport", "optimizer", "loss"}, False),
        ("optimizer.py", opt_src, {"prior", "critic", "transport", "loss"}, True),
    ):
        tree = ast.parse(src, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if forbid_all_pg:
                        assert not mod.startswith("particlegan"), f"{name} imports internal {mod}"
                    parts = mod.split(".")
                    for p in parts:
                        assert not any(p.startswith(fp) for fp in FORBIDDEN_IMPORT_PREFIXES), f"{name} forbidden import {mod}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if forbid_all_pg:
                    assert node.level == 0 and not mod.startswith("particlegan"), f"{name} imports internal {mod}"
                else:
                    if node.level > 0 and mod not in ("low_precision", ""):
                        assert mod not in forbidden_siblings, f"{name} sibling import {mod}"
                    if mod:
                        for p in mod.split("."):
                            assert not any(p.startswith(fp) for fp in FORBIDDEN_IMPORT_PREFIXES), f"{name} forbidden import {mod}"
                for alias in node.names:
                    if node.level == 1 and not mod:
                        assert alias.name not in forbidden_siblings, f"{name} sibling import {alias.name}"
                    assert not any(alias.name.startswith(fp) for fp in FORBIDDEN_IMPORT_PREFIXES), f"{name} forbidden import {alias.name}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_ATTRS, f"{name} references forbidden attr .{node.attr}"
                if name == "optimizer.py" and node.attr == "grad":
                    # Allow tensor `.grad` attribute access; forbid `autograd.grad` calls
                    if isinstance(node.value, ast.Attribute) and node.value.attr == "autograd":
                        raise AssertionError("optimizer.py must not call torch.autograd.grad")
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                # Ensure no step-index modulo schedules like `observed_steps % k` or `calls % k`
                left = ast.unparse(node.left)
                assert "observed_steps" not in left and "calls" not in left and "step" not in left, f"{name} clock modulo: {left}"


def test_zero_rng_consumption_in_critic_and_optimizer() -> None:
    torch.manual_seed(0)
    random.seed(0)
    critic = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(8 * 8 * 8, 1))
    rec = AxiomStepRecord()
    opt = AxiomOptimizer(critic.parameters(), lr=0.01, critic=critic, record=rec, conv_layouts=collect_conv_layouts((critic,)))
    pen_fn = AxiomCriticPenalty(opt, coeff=1.0, kappa=1.0, collect_stats=True)
    xr = torch.randn(8, 3, 16, 16)
    xf = torch.randn(8, 3, 16, 16)

    cpu_before = torch.get_rng_state()
    py_before = random.getstate()

    out_probe = eval_probe_forward(critic, (xr,), {}, rec)
    assert out_probe is not None and out_probe.shape == (8, 1)
    loss_d = critic(xr).mean() - critic(xf).mean()
    pen = pen_fn(critic, xr, xf)
    (loss_d + pen).backward()
    opt.step()

    assert torch.equal(torch.get_rng_state(), cpu_before), "Axiom critic/optimizer mutated PyTorch CPU RNG"
    assert random.getstate() == py_before, "Axiom critic/optimizer mutated Python RNG"


# ---------------------------------------------------------------------------
# 2. Law 2 (Critic Potential Geometry + Law 4.5 Normal Dissipation) Tests
# ---------------------------------------------------------------------------


def test_law2_regime1_exact_ror_matches_manual_autograd() -> None:
    torch.manual_seed(0)
    for out_dim in (1, 3):
        critic = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, out_dim))
        rec = AxiomStepRecord()
        kernel = AxiomPenaltyKernel(record=rec, coeff=1.0, kappa=1.0, lambda_floor=0.005, gamma_gauge=0.0)
        xr = torch.randn(10, 4)
        xf = torch.randn(10, 4)

        pen, stats = kernel.penalty(critic, xr, xf, collect_stats=True)
        assert stats["regime"] == "ror"
        assert math.isclose(stats["lambda_eff"], 0.005, rel_tol=1e-6)

        # Manual exact autograd calculation
        xr_m = xr.detach().requires_grad_(True)
        xf_m = xf.detach().requires_grad_(True)
        yr_m, yf_m = critic(xr_m), critic(xf_m)
        sq_r = torch.zeros(10)
        sq_f = torch.zeros(10)
        for k in range(out_dim):
            gr = torch.autograd.grad(yr_m[:, k].sum(), xr_m, create_graph=True)[0]
            gf = torch.autograd.grad(yf_m[:, k].sum(), xf_m, create_graph=True)[0]
            sq_r = sq_r + gr.square().sum(dim=-1)
            sq_f = sq_f + gf.square().sum(dim=-1)
        exact_s = 0.5 * ((sq_r / out_dim).mean() + (sq_f / out_dim).mean())
        exact_pen = 0.5 * 0.005 * exact_s

        assert torch.allclose(pen, exact_pen, rtol=1e-5, atol=1e-7)
        g_pen = torch.autograd.grad(pen, list(critic.parameters()), retain_graph=True, allow_unused=True)
        g_exact = torch.autograd.grad(exact_pen, list(critic.parameters()), allow_unused=True)
        for gp, ge in zip(g_pen, g_exact):
            if ge is None:
                assert gp is None
            else:
                assert gp is not None and torch.allclose(gp, ge, rtol=1e-5, atol=1e-6)


def test_law2_p6_zero_jensen_unbiased_below_kappa() -> None:
    """P6 Monte Carlo test: E_v[grad_theta R_fwad] matches exact grad_theta (0.5 * lambda_floor * s) below kappa."""
    torch.manual_seed(0)
    # d=24 triggers Regime II (fwAD); small weights ensure ||grad_x D|| < kappa=1.0
    critic = nn.Sequential(nn.Linear(24, 16), nn.SiLU(), nn.Linear(16, 1))
    with torch.no_grad():
        for p in critic.parameters():
            p.mul_(0.25)
    xr = torch.randn(32, 24)  # b_reg = 8 (4 real + 4 fake)
    xf = torch.randn(32, 24)

    # Exact target gradient (R1+R2 averaged over full batch, below kappa -> lambda_eff = lambda_floor)
    xr_e = xr.detach().requires_grad_(True)
    xf_e = xf.detach().requires_grad_(True)
    gr = torch.autograd.grad(critic(xr_e).sum(), xr_e, create_graph=True)[0]
    gf = torch.autograd.grad(critic(xf_e).sum(), xf_e, create_graph=True)[0]
    s_exact = 0.5 * (gr.square().sum(dim=-1).mean() + gf.square().sum(dim=-1).mean())
    assert float(s_exact.item()) < 0.5, f"expected slope^2 < kappa^2, got {float(s_exact.item())}"
    lam_floor = 0.005
    exact_grads = [
        torch.zeros_like(p) if g is None else g.detach().clone()
        for p, g in zip(critic.parameters(), torch.autograd.grad(0.5 * lam_floor * s_exact, list(critic.parameters()), allow_unused=True))
    ]

    # Average fwAD penalty gradients across 1024 Weyl phases (resetting ema_grad_sq <= kappa^2 to isolate E[.|F_{t-1}])
    rec = AxiomStepRecord()
    kernel = AxiomPenaltyKernel(record=rec, coeff=1.0, kappa=1.0, lambda_floor=lam_floor, gamma_gauge=0.0)
    acc = [torch.zeros_like(p) for p in critic.parameters()]
    s_acc, n_mc = 0.0, 1024
    for i in range(n_mc):
        rec.ema_grad_sq = 0.25
        rec.grad_sq_initialized = True
        pen, stats = kernel.penalty(critic, xr, xf, collect_stats=(i == 0))
        if i == 0:
            assert stats["regime"] == "fwad"
            assert math.isclose(stats["lambda_eff"], lam_floor, rel_tol=1e-6)
        s_acc += float((pen.detach() / (0.5 * lam_floor)).item())
        gs = torch.autograd.grad(pen, list(critic.parameters()), allow_unused=True)
        for a, g in zip(acc, gs):
            if g is not None:
                a.add_(g.detach())

    mc_vec = torch.cat([(a / n_mc).reshape(-1) for a in acc])
    ex_vec = torch.cat([g.reshape(-1) for g in exact_grads])
    rel_s = abs(s_acc / n_mc - float(s_exact.item())) / float(s_exact.item())
    rel_err = float((mc_vec - ex_vec).norm().item() / ex_vec.norm().clamp_min(1e-12).item())
    cos_g = float((mc_vec @ ex_vec).item() / (mc_vec.norm().item() * ex_vec.norm().item()))
    norm_ratio = float(mc_vec.norm().item() / ex_vec.norm().item())
    assert rel_s < 0.02, f"Scalar s_hat biased: rel_s={rel_s:.4f}"
    assert rel_err < 0.05 and cos_g > 0.998 and 0.97 <= norm_ratio <= 1.03, (
        f"Zero-Jensen unbiasedness violated: rel_err={rel_err:.4f}, cos={cos_g:.6f}, norm_ratio={norm_ratio:.4f}"
    )


def test_law2_dimension_independent_mu_d_and_adaptive_ema_ramp() -> None:
    torch.manual_seed(0)
    # 1) Low-D (dim = 2)
    c_low = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
    r_low = AxiomStepRecord()
    k_low = AxiomPenaltyKernel(record=r_low, coeff=1.0, kappa=1.0, lambda_floor=0.005)
    _, st_low = k_low.penalty(c_low, torch.randn(4, 2), torch.randn(4, 2), collect_stats=True)

    # 2) High-res 3x256x256 (dim = 196,608) lightweight strided Conv2d
    c_hi = nn.Sequential(nn.Conv2d(3, 4, 8, stride=8), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    r_hi = AxiomStepRecord()
    k_hi = AxiomPenaltyKernel(record=r_hi, coeff=1.0, kappa=1.0, lambda_floor=0.005)
    _, st_hi = k_hi.penalty(c_hi, torch.randn(4, 3, 256, 256), torch.randn(4, 3, 256, 256), collect_stats=True)

    assert st_low["regime"] == "ror" and st_hi["regime"] == "fwad"
    assert math.isclose(st_low["lambda_eff"], 0.005, rel_tol=1e-6)
    assert math.isclose(st_hi["lambda_eff"], 0.005, rel_tol=1e-6)

    # 3) Adaptive detached-EMA ramping when ||grad_x D|| > kappa
    c_steep = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        c_steep.weight.fill_(3.0)  # ||grad_x D||^2 = 4 * 9 = 36 >> kappa^2 = 1
    r_st = AxiomStepRecord()
    k_st = AxiomPenaltyKernel(record=r_st, coeff=1.0, kappa=1.0, lambda_floor=0.005, beta_ema=1.5, max_lambda=2.0)
    lams = []
    for _ in range(5):
        _, st = k_st.penalty(c_steep, torch.randn(8, 4), torch.randn(8, 4), collect_stats=True)
        lams.append(st["lambda_eff"])
    assert math.isclose(lams[0], 0.005, rel_tol=1e-6), "First call must use pre-update ema_grad_sq=1.0"
    assert lams[1] > 1.0 and lams[-1] <= 2.0 + 1e-6, f"Expected adaptive ramp into (1, 2], got {lams}"


def test_law2_clockfree_weyl_phase_invariance() -> None:
    torch.manual_seed(0)
    c1 = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4, padding=1), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(4 * 4 * 4, 1))
    c2 = deepcopy(c1)
    xr, xf = torch.randn(8, 3, 80, 80), torch.randn(8, 3, 80, 80)

    r1, r2 = AxiomStepRecord(), AxiomStepRecord()
    r2.observed_steps = 8192  # clockfree_audit step_label_offset
    k1 = AxiomPenaltyKernel(record=r1)
    k2 = AxiomPenaltyKernel(record=r2)
    for _ in range(4):
        p1, s1 = k1.penalty(c1, xr, xf, collect_stats=True)
        p2, s2 = k2.penalty(c2, xr, xf, collect_stats=True)
        assert s1["regime"] == "fwad" and s2["regime"] == "fwad"
        assert torch.equal(p1, p2)
        g1 = torch.autograd.grad(p1, list(c1.parameters()))
        g2 = torch.autograd.grad(p2, list(c2.parameters()))
        for a, b in zip(g1, g2):
            assert torch.equal(a, b)


def test_law2_swap_invariant_gauge_live_reuse_and_thermal_cooling() -> None:
    torch.manual_seed(0)
    critic = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
    rec = AxiomStepRecord()
    opt = AxiomOptimizer(critic.parameters(), lr=0.01, critic=critic, record=rec)
    pen_fn = AxiomCriticPenalty(opt, coeff=1.0, kappa=1.0, gamma_gauge=0.03, collect_stats=True)
    xr, xf = torch.randn(8, 4), torch.randn(8, 4)

    # Live forward reuse path
    dr, df = critic(xr), critic(xf)
    loss_d = (dr.mean() - df.mean())
    pen = pen_fn(critic, xr, xf)
    assert pen_fn.last_stats["live_endpoints_reused"] is True
    (loss_d + pen).backward()
    opt.step()
    assert rec.observed_steps == 1

    # Fallback when live graph was already backwarded
    opt.zero_grad(set_to_none=True)
    dr2, df2 = critic(xr), critic(xf)
    (dr2.mean() - df2.mean()).backward()
    pen2 = pen_fn(critic, xr, xf)
    assert pen_fn.last_stats["live_endpoints_reused"] is False
    pen2.backward()

    # Swap invariance in (x_real, x_fake)
    k_a = AxiomPenaltyKernel(record=AxiomStepRecord(), gamma_gauge=0.05)
    k_b = AxiomPenaltyKernel(record=AxiomStepRecord(), gamma_gauge=0.05)
    pa, sa = k_a.penalty(critic, xr, xf, collect_stats=True)
    pb, sb = k_b.penalty(critic, xf, xr, collect_stats=True)
    assert torch.allclose(pa, pb, rtol=1e-6, atol=1e-7)
    assert math.isclose(sa["s_hat"], sb["s_hat"], rel_tol=1e-6)

    # Thermal cooling straight-through + 6-channel EMA zero-Jensen oscillation cancellation + r_tail + shock reheating
    rt = AxiomStepRecord()
    for i in range(120):
        sgn = 0.08 if i % 2 == 0 else -0.08  # signed zero-mean oscillation cancels in per-channel EMA!
        s_th = rt.observe_thermal(sgn, 0.0, sgn, sgn, sgn, r_tail=0.0, tau=0.084)
    assert rt.thermal_ema < 0.01 and s_th < 0.01 and rt.thermal_locked is True
    kt = AxiomPenaltyKernel(record=rt, coeff=1.0, gamma_gauge=0.03)
    critic.zero_grad(set_to_none=True)
    pt, stt = kt.penalty(critic, xr, xf, collect_stats=True)
    assert stt["s_thermal"] == s_th
    pt.backward()
    g_cool = sum(p.grad.norm().item() for p in critic.parameters() if p.grad is not None)
    # Shock reheating restores thermal_scale -> 1.0 (and r_tail channel is tracked independently)
    s_reheat = rt.observe_thermal(0.8, 0.5, 0.2, 0.2, 0.2, r_tail=0.15, tau=0.084)
    assert s_reheat > 0.95 and rt.thermal_locked is False and math.isclose(rt.ema_tail, 0.15, rel_tol=1e-6)
    critic.zero_grad(set_to_none=True)
    kt.penalty(critic, xr, xf)[0].backward()
    g_hot = sum(p.grad.norm().item() for p in critic.parameters() if p.grad is not None)
    assert g_cool < 0.1 * g_hot


def test_law2_quantized_mxfp8_nvfp4_fwad_fallback_and_bn_safety() -> None:
    from particlegan.low_precision import EmulatedLinear

    torch.manual_seed(0)
    # 1) EmulatedLinear MXFP8 and NVFP4 (d=32 normally selects fwAD, must route to RoR safely)
    for fmt in ("mxfp8", "nvfp4"):
        q_critic = nn.Sequential(
            EmulatedLinear.from_linear(nn.Linear(32, 16), fmt),
            nn.SiLU(),
            EmulatedLinear.from_linear(nn.Linear(16, 1), fmt),
        )
        rec = AxiomStepRecord()
        kernel = AxiomPenaltyKernel(record=rec)
        pen, stats = kernel.penalty(q_critic, torch.randn(8, 32), torch.randn(8, 32), collect_stats=True)
        assert stats["regime"] == "ror", f"{fmt} must route cleanly to RoR"
        assert torch.isfinite(pen)
        pen.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm().item() > 0 for p in q_critic.parameters())

    # 2) Auto-fallback on modules raising RuntimeError under fwAD (simulating SDPBackend.MATH)
    class NoFwADCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(24, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if torch.autograd.forward_ad.unpack_dual(x).tangent is not None:
                raise RuntimeError("forward AD unsupported")
            return self.fc(x)

    nf = NoFwADCritic()
    knf = AxiomPenaltyKernel(record=AxiomStepRecord())
    p1, s1 = knf.penalty(nf, torch.randn(6, 24), torch.randn(6, 24), collect_stats=True)
    assert s1["regime"] == "ror" and id(nf) in knf._fwad_unsupported and torch.isfinite(p1)
    p1.backward()
    assert nf.fc.weight.grad is not None and torch.isfinite(nf.fc.weight.grad).all()

    # 3) BatchNorm2d eval() safety during both fwAD (plain Conv+BN) and RoR (GroupNorm+BN) (zero running_mean/var drift, .training restored)
    bn_critic = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(8 * 4 * 4, 1))
    bn_critic.train()
    bn = bn_critic[1]
    rm_before = bn.running_mean.clone()
    rv_before = bn.running_var.clone()
    kbn = AxiomPenaltyKernel(record=AxiomStepRecord())
    pbn, sbn = kbn.penalty(bn_critic, torch.randn(8, 3, 80, 80) * 5.0 + 3.0, torch.randn(8, 3, 80, 80) * 2.0 - 4.0, collect_stats=True)
    assert sbn["regime"] == "fwad" and bn.training is True
    assert torch.equal(bn.running_mean, rm_before) and torch.equal(bn.running_var, rv_before)
    pbn.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bn_critic.parameters())
    gn_bn_critic = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GroupNorm(2, 8), nn.BatchNorm2d(8), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(8 * 2 * 2, 1))
    gn_bn_critic.train()
    bn2 = gn_bn_critic[2]
    rm2_before, rv2_before = bn2.running_mean.clone(), bn2.running_var.clone()
    pbn_s, sbn_s = kbn.penalty(gn_bn_critic, torch.randn(8, 3, 16, 16) * 5.0 + 3.0, torch.randn(8, 3, 16, 16) * 2.0 - 4.0, collect_stats=True)
    assert sbn_s["regime"] == "ror" and bn2.training is True
    assert torch.equal(bn2.running_mean, rm2_before) and torch.equal(bn2.running_var, rv2_before)
    pbn_s.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gn_bn_critic.parameters())


def test_step_record_scale_homogeneity_probe_and_tamper_rejection() -> None:
    torch.manual_seed(0)
    # 1) observe_margin_and_scale is strictly scale-homogeneous (no 0.05 floor)
    xr0, xf0 = torch.randn(16, 4), torch.randn(16, 4)
    scales = []
    for c in (1e-6, 1.0, 1e6):
        r = AxiomStepRecord()
        r.observe_margin_and_scale(torch.ones(16, 1), -torch.ones(16, 1), xr0 * c, xf0 * c)
        scales.append(r.ema_data_scale / c)
    assert math.isclose(scales[0], scales[1], rel_tol=1e-5)
    assert math.isclose(scales[1], scales[2], rel_tol=1e-5)

    # 2) Stratified generator probe hook captures extreme-norm samples when B > 32
    gen = nn.Linear(4, 2)
    rec = AxiomStepRecord()
    _attach_axiom_generator_hook(gen, rec)
    z = torch.randn(64, 4)
    z[63] = 50.0  # extreme outlier sample
    _ = gen(z)
    assert rec._g_probe_cache is not None
    _, pa, _, y_pre = rec._g_probe_cache
    assert y_pre.shape == (32, 2)
    assert float(pa[0].abs().amax().item()) == 50.0, "Stratified probe must capture top-norm rail outlier"

    # 2b) Critic hook attaches all 6 _axiom_* capture attributes (incl. continuous + discrete cond)
    class CondCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(6, 1)

        def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
            return self.fc(torch.cat((x, ctx), dim=-1)) + 0.0 * t_idx.float().unsqueeze(-1)

    cc = CondCritic()
    _attach_axiom_critic_hook(cc, rec)
    xc, ctxc, tc = torch.randn(8, 4), torch.randn(8, 2), torch.arange(8, dtype=torch.long)
    out_c = cc(xc, ctxc, tc)
    assert torch.equal(out_c._axiom_input, torch.cat((xc, ctxc), dim=-1)) and out_c._axiom_record is rec
    assert out_c._axiom_cont_cond is True and out_c._axiom_cond is not None and len(out_c._axiom_parts) == 2
    assert torch.equal(out_c._axiom_parts[0], xc) and torch.equal(out_c._axiom_parts[1], ctxc)

    # 3) AxiomStepRecord checkpoint round-trip & tamper rejection
    rec.calls, rec.observed_steps, rec.chord_phase = 7, 5, 0.42
    rec.chart_state = {"cent": torch.tensor([1.0, 2.0])}
    sd = rec.state_dict()
    rec2 = AxiomStepRecord()
    rec2.load_state_dict(sd)
    assert rec2.calls == 7 and rec2.observed_steps == 5 and math.isclose(rec2.chord_phase, 0.42)
    assert torch.equal(rec2.chart_state["cent"], torch.tensor([1.0, 2.0]))

    bad_mutations = [
        lambda d: d.__setitem__("schema", 2),
        lambda d: d.__setitem__("calls", -1),
        lambda d: d.__setitem__("anchor_started", True),
        lambda d: d.__setitem__("chord_phase", 1.0),
        lambda d: d.__setitem__("ema_data_scale", 0.0),
        lambda d: d.__setitem__("thermal_scale", 1e-6),
        lambda d: d.__setitem__("chart_state", {"bad": torch.tensor([float("nan")])}),
        lambda d: d.pop("calls"),
    ]
    for mutate in bad_mutations:
        bad = deepcopy(sd)
        mutate(bad)
        with pytest.raises(ValueError):
            AxiomStepRecord().load_state_dict(bad)


# ---------------------------------------------------------------------------
# 3. Law 4 (AxiomOptimizer) Four-Laws Adversarial Tests
# ---------------------------------------------------------------------------


def test_law4_p1_all_five_regimes() -> None:
    torch.manual_seed(0)

    # Direct mantissa_exponent_normalize check across [1e-43, 3e+37] + zero/NaN safety
    g0 = torch.tensor([3.0, -4.0, 0.0, 12.0])
    u_ref, l2_ref, ok_ref = mantissa_exponent_normalize(g0)
    assert bool(ok_ref.item()) and math.isclose(float(u_ref.norm().item()), 1.0, rel_tol=1e-6)
    for exp_s in (-38, -20, 20, 35):
        sc = 10.0 ** exp_s
        u_s, l2_s, ok_s = mantissa_exponent_normalize(g0 * sc)
        assert bool(ok_s.item()) and torch.allclose(u_s, u_ref, rtol=1e-6, atol=1e-6)
        assert math.isclose(float(l2_s.item()) - float(l2_ref.item()), math.log2(sc), abs_tol=1e-4)
    u_nan, _, ok_nan = mantissa_exponent_normalize(torch.tensor([1.0, float("nan")]))
    assert not bool(ok_nan.item()) and torch.equal(u_nan, torch.zeros(2))

    # Regime 1: Rail escape from 1e-18 and 1e-12 gradients
    for scale in (1e-18, 1e-12):
        w = nn.Parameter(torch.zeros(4, 4))
        b = nn.Parameter(torch.zeros(4))
        opt = AxiomOptimizer([w, b], lr=0.02)
        for _ in range(10):
            w.grad = torch.full_like(w, scale)
            b.grad = torch.full_like(b, scale)
            opt.step()
        assert float(w.norm().item()) > 0.18, f"2D rail escape stalled at scale={scale}"
        assert float(b.norm().item()) > 0.15, f"1D rail escape stalled at scale={scale}"

    # Regime 2: Exact trajectory degree-0 invariance across c in {1e-18, 1e-12, 1, 1e12, 1e18}
    def run_traj(c: float, beta1: float = 0.0) -> list[torch.Tensor]:
        torch.manual_seed(123)
        p1d = nn.Parameter(torch.randn(6))
        p2d = nn.Parameter(torch.randn(8, 8))
        p4d = nn.Parameter(torch.randn(4, 3, 3, 3))
        ptab = nn.Parameter(torch.randn(16, 4))
        opt = AxiomOptimizer(
            [
                {"params": [p1d, p2d, p4d], "lr": 0.015, "role": "generator", "betas": (beta1, 0.999)},
                {"params": [ptab], "lr": 0.02, "role": "prior", "betas": (beta1, 0.999)},
            ]
        )
        for step_i in range(8):
            g_gen = torch.Generator().manual_seed(1000 + step_i)
            sign = 1.0 if step_i % 3 != 2 else -1.0
            p1d.grad = torch.randn(p1d.shape, generator=g_gen) * (c * sign)
            p2d.grad = torch.randn(p2d.shape, generator=g_gen) * (c * sign)
            p4d.grad = torch.randn(p4d.shape, generator=g_gen) * (c * sign)
            ptab.grad = torch.randn(ptab.shape, generator=g_gen) * (c * sign)
            opt.set_sampled_rows(ptab, torch.tensor([0, 3, 5, 3]))
            opt.step()
        return [p1d.detach().clone(), p2d.detach().clone(), p4d.detach().clone(), ptab.detach().clone()]

    for beta1 in (0.0, 0.5):
        ref = run_traj(1.0, beta1=beta1)
        for c in (1e-18, 1e-12, 1e12, 1e18):
            cur = run_traj(c, beta1=beta1)
            for a, b in zip(ref, cur):
                assert torch.allclose(a, b, rtol=1e-5, atol=1e-5), f"Degree-0 violated at c={c}, beta1={beta1}"

    # Regime 3: <= 2-step recovery from 1000x gradient spike
    x = nn.Parameter(torch.zeros(16))
    opt_sp = AxiomOptimizer([x], lr=0.05)
    for _ in range(20):
        x.grad = torch.ones_like(x)
        opt_sp.step()
    pre = x.detach().clone()
    x.grad = torch.ones_like(x)
    opt_sp.step()
    baseline_step = float((x - pre).norm().item())
    # Inject 1000x spike
    x.grad = torch.ones_like(x) * 1000.0
    opt_sp.step()
    # Two recovery steps
    for _ in range(2):
        prev = x.detach().clone()
        x.grad = torch.ones_like(x)
        opt_sp.step()
    post_step = float((x - prev).norm().item())
    assert abs(post_step - baseline_step) / baseline_step < 0.05, f"Spike poisoned step: pre={baseline_step}, post={post_step}"

    # Regime 4: Equilibrium settling under steady-amplitude sign-flipping chatter + post-shift recovery (record=None!)
    xc = nn.Parameter(torch.zeros(8))
    opt_ch = AxiomOptimizer([xc], lr=0.1)
    xc.grad = torch.ones_like(xc)
    p0 = xc.detach().clone()
    opt_ch.step()
    initial_step = float((xc - p0).norm().item())
    for t in range(1, 80):
        prev = xc.detach().clone()
        xc.grad = torch.full_like(xc, 1.0 if t % 2 == 0 else -1.0)
        opt_ch.step()
    settled_step = float((xc - prev).norm().item())
    assert opt_ch.ema_step_cos < -0.5 and opt_ch.log2_damp < -3.0
    assert settled_step < 0.1 * initial_step, f"Chatter did not settle: init={initial_step}, settled={settled_step}"
    # Post-shift coherent recovery (no one-way ratchet!)
    for _ in range(90):
        prev = xc.detach().clone()
        xc.grad = torch.ones_like(xc)
        opt_ch.step()
    recovered_step = float((xc - prev).norm().item())
    assert opt_ch.log2_damp > -0.1 and recovered_step > 0.9 * initial_step

    # Regime 5: Non-freezing under t^{-p} decaying coherent gradients
    xd = nn.Parameter(torch.zeros(8))
    opt_dc = AxiomOptimizer([xd], lr=0.05)
    for t in range(1, 81):
        prev = xd.detach().clone()
        xd.grad = torch.full_like(xd, (t ** -1.5))
        opt_dc.step()
    last_step = float((xd - prev).norm().item())
    assert opt_dc.log2_damp == 0.0 and last_step > 0.045, f"Coherent t^-p froze: last_step={last_step}"


def test_law4_p0_fp32_tikhonov_ns_zero_leakage_and_bounded_sigma1() -> None:
    torch.manual_seed(0)
    # Rank-1 128x128 + rectangular shapes
    for shape in ((128, 128), (64, 128), (128, 64)):
        u = torch.randn(shape[0], 1)
        v = torch.randn(1, shape[1])
        m_r1 = u @ v
        for rs in (0.0, 0.05):
            o = fp32_tikhonov_ns_2d(m_r1, steps=5, rel_smooth=rs)
            sv = torch.linalg.svdvals(o)
            assert float(sv[0].item()) <= 1.15 + 1e-4, f"Rank-1 sigma_1 blowup: {sv[0].item()}"
            assert float((sv[1] / sv[0].clamp_min(1e-12)).item()) < 1e-4, f"Orthogonal leakage: {sv[1]/sv[0]}"

    # Monotone contraction at large rel_smooth
    m = torch.randn(32, 32)
    s_lo = float(fp32_tikhonov_ns_2d(m, rel_smooth=0.05).norm().item())
    s_hi = float(fp32_tikhonov_ns_2d(m, rel_smooth=50.0).norm().item())
    assert s_hi < 0.25 * s_lo

    # Batched parity + safe zero/NaN handling
    m0 = torch.randn(16, 24)
    m1 = torch.zeros(16, 24)
    m2 = torch.full((16, 24), float("nan"))
    batch = torch.stack((m0, m1, m2), dim=0)
    out_b = fp32_tikhonov_ns_batched(batch, steps=5, rel_smooth=torch.tensor([0.05, 0.05, 0.05]))
    out_0 = fp32_tikhonov_ns_2d(m0, steps=5, rel_smooth=0.05)
    assert torch.allclose(out_b[0], out_0, rtol=1e-5, atol=1e-6)
    assert torch.equal(out_b[1], torch.zeros_like(m1)) and torch.equal(out_b[2], torch.zeros_like(m2))


def test_law4_p5_role_agnostic_deep_g_and_d_healthy_bounds_and_conv2d() -> None:
    torch.manual_seed(0)

    # 1) Deep 16L skip-free collapsed network: BOTH G and D trigger FP32 Tikhonov NS identically!
    def build_deep16() -> nn.Sequential:
        layers = [nn.Linear(16, 16, bias=False) for _ in range(16)]
        return nn.Sequential(*layers)

    g16 = build_deep16()
    d16 = deepcopy(g16)
    opt_g16 = AxiomOptimizer(g16.parameters(), lr=0.01)
    opt_d16 = AxiomOptimizer(d16.parameters(), lr=0.01, critic=d16)

    # Inject collapsed rank-1 gradients with deep exponential layer skew (R_skew >> 4)
    u = torch.randn(16, 1)
    v = torch.randn(1, 16)
    r1 = u @ v
    for idx, (pg, pd) in enumerate(zip(g16.parameters(), d16.parameters())):
        scale = 2.0 ** (idx * 0.5)  # 2^7.5 = 181x max/min skew
        pg.grad = r1 * scale
        pd.grad = r1 * scale
    opt_g16.step()
    opt_d16.step()
    assert opt_g16.ns_2d_activations == 16 and opt_d16.ns_2d_activations == 16
    for pg, pd in zip(g16.parameters(), d16.parameters()):
        assert torch.allclose(pg, pd, rtol=1e-6, atol=1e-6), "G and D must follow exact same role-agnostic law"

    # 2) O2 compound detector: healthy shallow 3L critic takes 0-NS fast foreach path while generator uses 2D dense NS
    d3 = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1))
    g3 = deepcopy(d3)
    opt_d3 = AxiomOptimizer(d3.parameters(), lr=0.01, critic=d3)
    opt_g3 = AxiomOptimizer(g3.parameters(), lr=0.01)
    for pd, pg in zip(d3.parameters(), g3.parameters()):
        g = ((torch.randn(pd.shape[0], 1) @ torch.randn(1, pd.shape[1])) + 0.05 * torch.randn_like(pd)) if pd.ndim == 2 else torch.randn_like(pd)
        pd.grad, pg.grad = g.clone(), g.clone()
    opt_d3.step()
    opt_g3.step()
    assert opt_d3.ns_2d_activations == 0 and opt_d3.fast_path_steps == 1
    assert opt_g3.ns_2d_activations == 2
    assert all(torch.isfinite(p).all() for p in d3.parameters())

    # 3) FP32 Tikhonov NS spectral & Frobenius bounds on 2D dense layers:
    #    - Rank-1 grad: sigma_1(Delta W) <= 1.15 * lr (exact polar, zero 2x overshoot!)
    #    - Isotropic full-rank grad: ||Delta W||_F == sqrt(64) * lr (100% Frobenius budget)
    lr = 0.01
    w_r1 = nn.Parameter(torch.zeros(64, 64))
    opt_r1 = AxiomOptimizer([w_r1], lr=lr)
    w_r1.grad = torch.randn(64, 1) @ torch.randn(1, 64)
    opt_r1.step()
    s1_r1 = float(torch.linalg.svdvals(w_r1.detach())[0].item())
    assert s1_r1 <= 1.16 * lr, f"Rank-1 spectral step exceeded 1.15*lr: {s1_r1} vs {1.15*lr}"

    w_iso = nn.Parameter(torch.zeros(64, 64))
    opt_iso = AxiomOptimizer([w_iso], lr=lr)
    w_iso.grad = torch.eye(64)
    opt_iso.step()
    frob_iso = float(w_iso.detach().norm().item())
    assert frob_iso >= 0.98 * math.sqrt(64.0) * lr, f"Isotropic layer under-stepped: {frob_iso}"

    # 4) Conv2d + ConvTranspose2d: 0 NS always, fast foreach path active
    conv_net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(8, 3, 3, padding=1))
    opt_c = AxiomOptimizer(conv_net.parameters(), lr=0.01, conv_layouts=collect_conv_layouts((conv_net,)))
    for p in conv_net.parameters():
        p.grad = torch.randn_like(p)
    opt_c.step()
    assert opt_c.ns_2d_activations == 0 and opt_c.fast_path_steps == 1


def test_law4_p4_verified_iterative_trust_and_scale_homogeneity() -> None:
    torch.manual_seed(0)

    # 1) 0 post-rescale violations across quadratic, cubic, and saturating tanh rail jump
    class NonlinearGen(nn.Module):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)
            self.mode = mode

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = self.fc(z)
            if self.mode == "quad":
                return h.square()
            if self.mode == "cubic":
                return h * h * h
            return torch.tanh(h * 10.0)

    for mode, lr_big in (("quad", 1.5), ("cubic", 1.5), ("tanh", 25.0)):
        torch.manual_seed(7)
        gen = NonlinearGen(mode)
        rec = AxiomStepRecord()
        rec.ema_data_scale = math.sqrt(2.0)  # coord_scale = 1.0
        rec.ema_push_res = 0.0  # d_max = 0.045
        _attach_axiom_generator_hook(gen, rec)
        opt_g = AxiomOptimizer(gen.parameters(), lr=lr_big)
        opt_g._axiom_record = rec

        z = torch.randn(32, 4)
        y_pre = gen(z)
        y_pre_det = y_pre.detach().clone()
        y_pre.square().mean().backward()
        opt_g.step()

        with torch.no_grad():
            y_post = gen(z)
            rms = (y_post.double() - y_pre_det.double()).square().mean(-1).sqrt()
            d_obs = float(rms.square().mean().sqrt().item())
        assert opt_g.last_probe_consumed is True
        assert opt_g.last_step_gain <= 1.0, "Upward amplification forbidden (alpha_max == 1.0)"
        assert opt_g.trust_backtracks_total > 0, f"Expected backtracking on {mode}"
        assert d_obs <= 0.045 * 1.005, f"Trust violation on {mode}: d_obs={d_obs:.5f} > 0.045"

    # 2) Scale homogeneity across c in {1e-4, 1.0, 1e4}
    gains = []
    for c in (1e-4, 1.0, 1e4):
        torch.manual_seed(99)
        class ScaledGen(nn.Module):
            def __init__(self, scale: float) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 2)
                self.scale = scale

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.fc(z) * self.scale

        gen = ScaledGen(c)
        rec = AxiomStepRecord()
        xr = torch.randn(32, 2) * c
        rec.observe_margin_and_scale(torch.ones(32, 1), -torch.ones(32, 1), xr, xr * 0.5)
        _attach_axiom_generator_hook(gen, rec)
        opt_g = AxiomOptimizer(gen.parameters(), lr=0.8)
        opt_g._axiom_record = rec

        z = torch.randn(32, 4)
        gen(z).square().mean().backward()
        opt_g.step()
        gains.append(opt_g.last_step_gain)

    assert gains[1] < 1.0, "Trust test should actively backtrack"
    assert math.isclose(gains[0], gains[1], rel_tol=1e-4)
    assert math.isclose(gains[1], gains[2], rel_tol=1e-4)


def test_law4_zero_pinv_direction_blend_prior_rows_and_record_ownership() -> None:
    torch.manual_seed(0)

    # 1) Exact closed-form zero-pinv projection tests (1-constraint, 2-constraint, collinear)
    d = torch.tensor([1.0, 1.0, 0.5])
    n1 = torch.tensor([[1.0, 0.0, 0.0]])
    p1 = zero_pinv_project_nonascent(d, n1)
    assert float((n1 @ p1).max().item()) <= 1e-10
    assert torch.allclose(p1, torch.tensor([0.0, 1.0, 0.5]), atol=1e-6)

    n2 = torch.tensor([[1.0, 0.3, 0.0], [0.2, 1.0, 0.0]])
    p2 = zero_pinv_project_nonascent(d, n2)
    assert float((n2.double() @ p2.double()).max().item()) <= 1e-10
    assert torch.allclose(p2, torch.tensor([0.0, 0.0, 0.5]), atol=1e-6)

    n_opp = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    p_opp = zero_pinv_project_nonascent(d, n_opp)
    assert float((n_opp.double() @ p_opp.double()).max().item()) <= 1e-10
    assert torch.allclose(p_opp, torch.tensor([0.0, 1.0, 0.5]), atol=1e-6)

    # 2) End-to-end AxiomOptimizer direction_blend with 2 conflicting losses + K=2 Pythagorean boost
    w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    rec = AxiomStepRecord()
    rec.cond_initialized = True
    opt_b = AxiomOptimizer([w], lr=0.1, direction_blend=True)
    opt_b._axiom_record = rec

    with pytest.raises(ValueError):
        w.grad = torch.ones_like(w)
        opt_b.step()  # must require bind_protected_losses

    l1 = (w[0] - 0.0).square() + 0.1 * (w[2] - 0.0).square()
    l2 = (w[0] - 3.0).square() + (w[1] - 0.0).square()  # conflicts on w[0]
    opt_b.bind_protected_losses([l1, l2])
    (l1 + 2.0 * l2).backward()
    opt_b.step()
    assert opt_b.constraint_geometry_stats["steps"] == 1
    assert opt_b.constraint_geometry_stats["max_derivative_after"] <= 1e-7
    assert opt_b.direction_blend_stats["conflict_steps"] == 1

    # 3) Prior table sampled-row updates (untouched rows invariant + duplicate dedup + row_scales)
    table = nn.Parameter(torch.randn(10, 4))
    table_init = table.detach().clone()
    opt_p = AxiomOptimizer([{"params": [table], "lr": 0.05, "role": "prior"}])
    table.grad = torch.randn_like(table)
    with pytest.raises(ValueError):
        opt_p.step()  # missing sampled rows
    with pytest.raises(ValueError):
        opt_p.set_sampled_rows(table, torch.tensor([99]))  # out of bounds
    opt_p.set_sampled_rows(table, torch.tensor([1, 4, 1, 7]))
    opt_p.step()
    untouched = [0, 2, 3, 5, 6, 8, 9]
    assert torch.equal(table[untouched], table_init[untouched])
    assert not torch.equal(table[[1, 4, 7]], table_init[[1, 4, 7]])

    # 4) opt_d owns record; opt_g reads record via bind_axiom_record without double-counting observed_steps
    d_mod = nn.Linear(4, 1)
    g_mod = nn.Linear(4, 4)
    rec_shared = AxiomStepRecord()
    opt_d = AxiomOptimizer(d_mod.parameters(), lr=0.01, critic=d_mod, record=rec_shared)
    opt_g = AxiomOptimizer(g_mod.parameters(), lr=0.01)
    bind_axiom_record(rec_shared, opt_g)
    assert opt_g.record is None and opt_g._axiom_record is rec_shared
    assert "regularizer" in opt_d.state_dict() and "regularizer" not in opt_g.state_dict()
    for p in d_mod.parameters():
        p.grad = torch.randn_like(p)
    for p in g_mod.parameters():
        p.grad = torch.randn_like(p)
    opt_d.step()
    opt_g.step()
    assert rec_shared.observed_steps == 1, "Only opt_d may increment observed_steps"


def test_fresh_process_exact_continuation_and_tamper_rejection(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "axiom_ckpt.pt"
    ref_path = tmp_path / "axiom_ref.pt"
    res_path = tmp_path / "axiom_res.pt"

    script = f"""
import sys, torch
from torch import nn
from particlegan._axiom.critic import AxiomCriticPenalty, AxiomStepRecord, _attach_axiom_generator_hook, bind_axiom_record
from particlegan._axiom.optimizer import AxiomOptimizer

def make_bundle():
    torch.manual_seed(42)
    critic = nn.Sequential(nn.Linear(20, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1))
    gen = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 20))
    prior = nn.Parameter(torch.randn(12, 4))
    rec = AxiomStepRecord()
    opt_d = AxiomOptimizer(critic.parameters(), lr=0.015, critic=critic, record=rec)
    opt_g = AxiomOptimizer(
        [{{"params": list(gen.parameters()), "lr": 0.01, "role": "generator"}},
         {{"params": [prior], "lr": 0.02, "role": "prior"}}],
        direction_blend=True,
    )
    bind_axiom_record(rec, opt_g)
    _attach_axiom_generator_hook(gen, rec)
    pen_fn = AxiomCriticPenalty(opt_d, coeff=1.0, kappa=1.0, collect_stats=True)
    return critic, gen, prior, rec, opt_d, opt_g, pen_fn

def run_step(step_i, critic, gen, prior, rec, opt_d, opt_g, pen_fn):
    g_rng = torch.Generator().manual_seed(5000 + step_i)
    xr = torch.randn(8, 20, generator=g_rng) * (2.0 if step_i % 2 == 0 else -2.0)
    rows = torch.tensor([(step_i + k) % 12 for k in range(8)], dtype=torch.long)
    opt_d.zero_grad(set_to_none=True)
    xf = gen(prior[rows]).detach()
    dr, df = critic(xr), critic(xf)
    rec.observe_margin_and_scale(dr, df, xr, xf)
    loss_d = (dr.mean() - df.mean()) * (1.0 if step_i % 2 == 0 else -1.0)
    pen = pen_fn(critic, xr, xf)
    (loss_d + pen).backward()
    opt_d.step()

    opt_g.zero_grad(set_to_none=True)
    opt_g.set_sampled_rows(prior, rows)
    xf2 = gen(prior[rows])
    l1 = -critic(xf2).mean()
    l2 = (xf2 - xr).square().mean()
    opt_g.bind_protected_losses([l1, l2])
    (l1 + l2).backward()
    opt_g.step()

mode = sys.argv[1]
critic, gen, prior, rec, opt_d, opt_g, pen_fn = make_bundle()
if mode == "full":
    for i in range(12):
        run_step(i, critic, gen, prior, rec, opt_d, opt_g, pen_fn)
        if i == 5:
            torch.save({{
                "critic": critic.state_dict(), "gen": gen.state_dict(), "prior": prior.detach().clone(),
                "opt_d": opt_d.state_dict(), "opt_g": opt_g.state_dict(),
            }}, r"{ckpt_path}")
    torch.save({{
        "critic": critic.state_dict(), "gen": gen.state_dict(), "prior": prior.detach().clone(),
        "rec": rec.state_dict(), "opt_d": opt_d.state_dict(), "opt_g": opt_g.state_dict(),
    }}, r"{ref_path}")
elif mode == "resume":
    ck = torch.load(r"{ckpt_path}", weights_only=False)
    critic.load_state_dict(ck["critic"])
    gen.load_state_dict(ck["gen"])
    with torch.no_grad():
        prior.copy_(ck["prior"])
    opt_d.load_state_dict(ck["opt_d"])
    opt_g.load_state_dict(ck["opt_g"])
    for i in range(6, 12):
        run_step(i, critic, gen, prior, rec, opt_d, opt_g, pen_fn)
    torch.save({{
        "critic": critic.state_dict(), "gen": gen.state_dict(), "prior": prior.detach().clone(),
        "rec": rec.state_dict(), "opt_d": opt_d.state_dict(), "opt_g": opt_g.state_dict(),
    }}, r"{res_path}")
"""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-c", script, "full"], check=True, env=env)
    subprocess.run([sys.executable, "-c", script, "resume"], check=True, env=env)

    ref = torch.load(ref_path, weights_only=False)
    res = torch.load(res_path, weights_only=False)
    for k in ref["critic"]:
        assert torch.equal(ref["critic"][k], res["critic"][k]), f"critic.{k} diverged across fresh-process resume"
    for k in ref["gen"]:
        assert torch.equal(ref["gen"][k], res["gen"][k]), f"gen.{k} diverged across fresh-process resume"
    assert torch.equal(ref["prior"], res["prior"]), "prior diverged across fresh-process resume"
    assert ref["rec"] == res["rec"], "AxiomStepRecord diverged across fresh-process resume"
    assert ref["opt_d"]["axiom"] == res["opt_d"]["axiom"], "opt_d metadata diverged"
    assert ref["opt_g"]["axiom"]["log2_damp"] == res["opt_g"]["axiom"]["log2_damp"]
    assert ref["opt_g"]["axiom"]["ema_step_cos"] == res["opt_g"]["axiom"]["ema_step_cos"]
    assert ref["opt_g"]["axiom"]["constraint_geometry"]["stats"] == res["opt_g"]["axiom"]["constraint_geometry"]["stats"]

    # Tamper rejection on AxiomOptimizer checkpoint
    p_tmp = nn.Parameter(torch.randn(4, 4))
    rec_t = AxiomStepRecord()
    opt_t = AxiomOptimizer([p_tmp], lr=0.01, record=rec_t)
    p_tmp.grad = torch.randn_like(p_tmp)
    opt_t.step()
    good_sd = opt_t.state_dict()
    for tamper in (
        lambda s: s["axiom"].__setitem__("schema", 99),
        lambda s: s["axiom"].__setitem__("log2_damp", -12.0),
        lambda s: s["axiom"].__setitem__("ema_step_cos", 1.5),
        lambda s: s["axiom"].__setitem__("prev_step_sketch", [0.0] * 10),
        lambda s: s["state"][0].__setitem__("v_pow", torch.tensor([float("nan"), 0.0, 0.0, 0.0])),
        lambda s: s["regularizer"].__setitem__("ema", {"bad": 1}),
        lambda s: s.pop("regularizer"),
    ):
        bad_sd = deepcopy(good_sd)
        tamper(bad_sd)
        with pytest.raises(ValueError):
            opt_t.validate_state_dict(bad_sd)


def test_conv2d_r256_speed_and_fast_path_sanity() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        D = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        ).to(dev, memory_format=torch.channels_last)
        D._axiom_bf16_autocast = True
        rec = AxiomStepRecord()
        opt_d = AxiomOptimizer(D.parameters(), lr=0.021, critic=D, record=rec, conv_layouts=collect_conv_layouts((D,)))
        pen_fn = AxiomCriticPenalty(opt_d, coeff=1.0, kappa=1.0, collect_stats=True)
        xr = torch.randn(16, 3, 256, 256, device=dev).to(memory_format=torch.channels_last)
        xf = torch.randn(16, 3, 256, 256, device=dev).to(memory_format=torch.channels_last)
        for _ in range(3):
            opt_d.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = D(xr).float().mean() - D(xf).float().mean() + pen_fn(D, xr, xf)
            loss.backward()
            opt_d.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            opt_d.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = D(xr).float().mean() - D(xf).float().mean() + pen_fn(D, xr, xf)
            loss.backward()
            opt_d.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / 10.0
        assert pen_fn.last_stats["regime"] == "fwad"
        assert opt_d.ns_2d_activations == 0 and opt_d.fast_path_steps == 13
        assert ms < 80.0, f"r256 D+pen+opt too slow: {ms:.2f} ms/step"
    else:
        # CPU-only fast Conv2d sanity (keeps CPU pytest < 30s)
        D = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8, 16, 4, 2, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 1, 4, 4, 0, bias=False),
        )
        rec = AxiomStepRecord()
        opt_d = AxiomOptimizer(D.parameters(), lr=0.02, critic=D, record=rec, conv_layouts=collect_conv_layouts((D,)))
        pen_fn = AxiomCriticPenalty(opt_d, coeff=1.0, kappa=1.0, collect_stats=True)
        xr, xf = torch.randn(8, 3, 32, 32), torch.randn(8, 3, 32, 32)
        for _ in range(3):
            opt_d.zero_grad(set_to_none=True)
            loss = D(xr).mean() - D(xf).mean() + pen_fn(D, xr, xf)
            loss.backward()
            opt_d.step()
        assert pen_fn.last_stats["regime"] == "fwad"
        assert opt_d.ns_2d_activations == 0 and opt_d.fast_path_steps == 3


def test_optimizer_review_regressions_large_conv_spill_dead_prefix_and_tensor_mod() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # (a) >262K 4D Conv2d (524,288 params): escapes 1e-22 / 1e20 AND stays finite/unchanged on Inf / NaN
    conv = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False).to(dev)
    assert conv.weight.numel() == 524_288
    opt_big = AxiomOptimizer(conv.parameters(), lr=0.01, conv_layouts=collect_conv_layouts((conv,)))
    for scale in (1e-22, 1e20):
        w0 = conv.weight.detach().clone()
        conv.weight.grad = torch.ones_like(conv.weight) * scale
        opt_big.step()
        delta = (conv.weight.detach() - w0).norm().item()
        assert torch.isfinite(conv.weight).all() and delta > 0.0, f"Large Conv2d deadlocked at scale={scale}"
    for bad_val in (float("inf"), float("nan")):
        w0 = conv.weight.detach().clone()
        conv.weight.grad = torch.full_like(conv.weight, bad_val)
        opt_big.step()
        assert torch.isfinite(conv.weight).all() and torch.equal(conv.weight.detach(), w0), (
            f"Large Conv2d corrupted weights on {bad_val} grad"
        )

    # (b) Dead-prefix chatter damping: rows [:2, :] dead (0), rows [2:, :] chattering +1/-1
    w_dp = nn.Parameter(torch.zeros(64, 64, device=dev))
    opt_dp = AxiomOptimizer([w_dp], lr=0.05)
    for t in range(40):
        g = torch.zeros_like(w_dp)
        g[2:, :] = 1.0 if t % 2 == 0 else -1.0
        w_dp.grad = g
        opt_dp.step()
    assert opt_dp.ema_step_cos < -0.5 and opt_dp.log2_damp < -1.0, (
        f"Dead-prefix chatter missed: cos={opt_dp.ema_step_cos}, log2_damp={opt_dp.log2_damp}"
    )

    # (c) Mixed [critic, generator] fast-path spill preserves per-group is_gr (g_scale applies ONLY to generator)
    p_crit = nn.Parameter(torch.zeros(8, device=dev))
    p_gen = nn.Parameter(torch.zeros(8, device=dev))
    opt_mix = AxiomOptimizer(
        [{"params": [p_crit], "lr": 0.05, "role": "critic"}, {"params": [p_gen], "lr": 0.05, "role": "generator"}]
    )
    opt_mix.bind_axiom_modulation(g_scale=0.25)
    p_crit.grad = torch.ones_like(p_crit) * 1e-22
    p_gen.grad = torch.ones_like(p_gen) * 1e-22
    opt_mix.step()
    d_crit = float(p_crit.detach().norm().item())
    d_gen = float(p_gen.detach().norm().item())
    assert math.isclose(d_crit, 0.05, rel_tol=1e-4), f"Critic group wrongly scaled: {d_crit}"
    assert math.isclose(d_gen, 0.05 * 0.25, rel_tol=1e-4), f"Generator group missing g_scale: {d_gen}"

    # (d) 0-D tensor g_scale & wakeup_boost via bind_axiom_modulation across fast, slow, and 2D paths
    p_f = nn.Parameter(torch.zeros(8, device=dev))
    p_s = nn.Parameter(torch.zeros(8, device=dev))
    p_2 = nn.Parameter(torch.zeros(4, 4, device=dev))
    opt_mod = AxiomOptimizer(
        [
            {"params": [p_f], "lr": 0.04, "role": "generator"},
            {"params": [p_s], "lr": 0.04, "role": "generator", "betas": (0.5, 0.999)},
            {"params": [p_2], "lr": 0.04, "role": "generator"},
        ]
    )
    opt_mod._ema_func_gain = 0.5  # ensures wakeup_boost enters pre_g = min(1.0, 0.5 * 1.5) = 0.75
    opt_mod.bind_axiom_modulation(
        g_scale=torch.tensor(0.5, device=dev),
        wakeup_boost=torch.tensor(1.5, device=dev),
    )
    p_f.grad = torch.ones_like(p_f)
    p_s.grad = torch.ones_like(p_s)
    p_2.grad = torch.eye(4, device=dev)
    opt_mod.step()
    expected_mult = 0.5 * 0.75  # g_scale * min(1, ema_func_gain * wakeup_boost)
    assert math.isclose(float(p_f.detach().norm().item()), 0.04 * expected_mult, rel_tol=1e-4)
    assert math.isclose(float(p_s.detach().norm().item()), 0.04 * expected_mult, rel_tol=1e-4)
    assert math.isclose(float(p_2.detach().norm().item()), 0.04 * expected_mult * math.sqrt(4.0), rel_tol=1e-4)


def test_axiom_v3_ror_wall_peak_rms_cooling_and_prior_momentum() -> None:
    torch.manual_seed(0)

    # 1) RoR exact-deadband C^1 wall: identically 0 below kappa, strictly active above kappa (with chord term)
    c_lin = nn.Linear(4, 1, bias=False)
    xr, xf = torch.randn(8, 4), torch.randn(8, 4)
    with torch.no_grad():
        c_lin.weight.fill_(0.2)  # ||grad_x D|| = sqrt(4 * 0.04) = 0.4 < kappa = 1.0
    r_lo = AxiomStepRecord()
    k_lo = AxiomPenaltyKernel(record=r_lo, coeff=1.0, kappa=1.0, lambda_floor=0.005, gamma_gauge=0.0)
    p_lo, s_lo = k_lo.penalty(c_lin, xr, xf, collect_stats=True)
    assert s_lo["regime"] == "ror" and math.isclose(float(p_lo.item()), 0.5 * 0.005 * 0.16, rel_tol=1e-5)

    with torch.no_grad():
        c_lin.weight.fill_(1.0)  # ||grad_x D|| = 2.0 > kappa = 1.0 -> cap_tot = (1 + 0.75 + 0.25) * (2 - 1)^2 = 2.0
    r_hi = AxiomStepRecord()
    k_hi = AxiomPenaltyKernel(record=r_hi, coeff=1.0, kappa=1.0, lambda_floor=0.005, gamma_gauge=0.0)
    p_hi, _ = k_hi.penalty(c_lin, xr, xf, collect_stats=True)
    assert math.isclose(float(p_hi.item()), 0.5 * (2.0 + 0.005 * 4.0), rel_tol=1e-4)

    # 2) Zero-deadband degree-0 Peak-RMS Tikhonov: parity between foreach fast path & scalar slow path
    p_fast = nn.Parameter(torch.zeros(4, 3, 3, 3))
    p_slow = nn.Parameter(torch.zeros(4, 3, 3, 3))
    opt_f = AxiomOptimizer([p_fast], lr=0.05)
    opt_s = AxiomOptimizer([p_slow], lr=0.05)
    opt_s._run_foreach_fast = lambda items, **kw: list(items)  # type: ignore[method-assign]
    for t in range(12):
        g = torch.randn_like(p_fast) * (1.0 if t < 4 else 0.1)
        p_fast.grad, p_slow.grad = g.clone(), g.clone()
        opt_f.step()
        opt_s.step()
    assert torch.allclose(p_fast, p_slow, rtol=1e-5, atol=1e-6)

    # 3) Calibrated thermal cooling s_cool & prior settling momentum (b1_eff >= 0.75 when prior_scale < 1)
    p_c = nn.Parameter(torch.zeros(8))
    rec_c = AxiomStepRecord()
    rec_c.thermal_scale = 0.0061  # s_cool = 0.0061 / hypot(0.0061, 0.0061) = 1 / sqrt(2)
    opt_c = AxiomOptimizer([p_c], lr=0.04, record=rec_c)
    p_c.grad = torch.ones_like(p_c)
    opt_c.step()
    assert math.isclose(float(p_c.detach().norm().item()), 0.04 / math.sqrt(2.0), rel_tol=1e-4)

    tab = nn.Parameter(torch.zeros(6, 4))
    opt_pr = AxiomOptimizer([{"params": [tab], "lr": 0.05, "role": "prior", "betas": (0.0, 0.999)}])
    tab.grad = torch.ones_like(tab)
    opt_pr.set_sampled_rows(tab, torch.tensor([0, 1]))
    opt_pr.bind_axiom_modulation(prior_scale=0.5)
    opt_pr.step()
    assert "m_prior" in opt_pr.state[tab], "Prior settling momentum must engage when prior_scale < 1.0"


def test_axiom_v4_capture_ring_detached_logits_and_vram_guard() -> None:
    torch.manual_seed(0)

    class CondCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 1)

        def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat((x, xt), dim=-1)) + 0.01 * t.float().unsqueeze(-1)

    critic = CondCritic()
    rec = AxiomStepRecord()
    _attach_axiom_critic_hook(critic, rec)

    xr = torch.randn(8, 2, requires_grad=True)
    xt = torch.randn(8, 2)
    t_idx = torch.randint(0, 4, (8,))
    raw = critic(xr, xt=xt, t=t_idx)
    det = raw.detach()
    det_view = raw.detach().reshape(8)
    assert getattr(det, "_axiom_input", None) is None

    for probe in (det, det_view):
        hit = rec.lookup_capture(probe)
        assert hit is not None
        x_in, cond, hrec, has_cc, parts = hit
        assert hrec is rec and has_cc is True and cond is not None and parts is not None and len(parts) == 2
        assert not x_in.requires_grad and not cond.requires_grad and all(not p.requires_grad for p in parts)
        assert torch.allclose(parts[0], xr.detach()) and torch.allclose(parts[1], xt)
        assert torch.equal(cond.reshape(-1), t_idx.to(torch.int64))

    # LRU ring cap <= 16 entries and cleared by clear_transient_captures()
    for _ in range(20):
        _ = critic(torch.randn(8, 2), xt=xt, t=t_idx).detach()
    assert len(rec._capture_ring) <= 16
    rec.clear_transient_captures()
    assert len(rec._capture_ring) == 0

    # High-dimensional VRAM guard (> 4096 elements/sample is never cached in _capture_ring)
    big_c = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 1))
    big_rec = AxiomStepRecord()
    _attach_axiom_critic_hook(big_c, big_rec)
    big_out = big_c(torch.randn(2, 3, 64, 64)).detach()
    assert len(big_rec._capture_ring) == 0 and big_rec.lookup_capture(big_out) is None

    # Log-space clamped ema_data_scale: exact at 1e-5x and immune to a single 1e4x shock batch
    s_rec = AxiomStepRecord()
    x_tiny = torch.ones((16, 2), dtype=torch.float32) * 1e-5
    lg = torch.zeros((16, 1), dtype=torch.float32)
    s_rec.observe_margin_and_scale(lg, lg, x_tiny, x_tiny)
    ds0 = float(s_rec.ema_data_scale)
    assert abs(ds0 - math.sqrt(2.0) * 1e-5) / (math.sqrt(2.0) * 1e-5) < 1e-5
    s_rec.observe_margin_and_scale(lg, lg, x_tiny * 1e4, x_tiny)
    ds1 = float(s_rec.ema_data_scale)
    assert ds1 <= ds0 * 1.05, f"log-space ema_data_scale jumped too far on 1e4x spike: {ds1 / ds0:.2f}x"


def test_axiom_v4_fixed_batch_closure_fallback_and_2b_kwargs_ror_parity() -> None:
    torch.manual_seed(0)

    # 1) Fixed-batch closure (expecting exact B=6) in RoR (dim=4) & fwAD (1x8x8 Conv2d)
    fixed_ctx = torch.randn(6, 4)
    c_mlp = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    k_ror = AxiomPenaltyKernel(record=AxiomStepRecord(), coeff=1.0, kappa=1.0)
    xr4, xf4 = torch.randn(6, 4), torch.randn(6, 4)
    pen_ror, st_ror = k_ror.penalty(lambda x: c_mlp(torch.cat((fixed_ctx, x), dim=-1)), xr4, xf4, collect_stats=True, critic_module=c_mlp)
    assert st_ror["regime"] == "ror" and torch.isfinite(pen_ror)
    pen_ror.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in c_mlp.parameters())

    fixed_img = torch.randn(8, 1, 80, 80)
    c_conv = nn.Sequential(nn.Conv2d(2, 4, 3, stride=4, padding=1), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(4 * 4 * 4, 1))
    k_fw = AxiomPenaltyKernel(record=AxiomStepRecord(), coeff=1.0, kappa=1.0)
    pen_fw, st_fw = k_fw.penalty(
        lambda x: c_conv(torch.cat((fixed_img, x), dim=1)),
        torch.randn(8, 1, 80, 80),
        torch.randn(8, 1, 80, 80),
        collect_stats=True,
        critic_module=c_conv,
    )
    assert st_fw["regime"] == "fwad" and torch.isfinite(pen_fw)
    pen_fw.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in c_conv.parameters())

    # 2) 2B kwargs-batched RoR matches unbatched per-batch RoR bit-for-bit
    class KwCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(6, 16)
            self.l2 = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.l2(torch.nn.functional.silu(self.l1(torch.cat((x, xt), dim=-1)))) + 0.05 * t.float().unsqueeze(-1)

    kw_c = KwCritic()
    xr, xf, xt, t_i = torch.randn(8, 3), torch.randn(8, 3), torch.randn(8, 3), torch.randint(0, 4, (8,))
    k_bat = AxiomPenaltyKernel(record=AxiomStepRecord(), coeff=1.0, kappa=0.5)
    p_bat, _ = k_bat.penalty(kw_c, xr, xf, xt=xt, t=t_i)
    k_seq = AxiomPenaltyKernel(record=AxiomStepRecord(), coeff=1.0, kappa=0.5)
    # Force fallback to unbatched by asserting B==8 inside wrapper
    def _fixed_b(x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != 8:
            raise RuntimeError("fixed batch size 8")
        return kw_c(x, xt=xt, t=t)

    p_seq, _ = k_seq.penalty(_fixed_b, xr, xf, xt=xt, t=t_i, critic_module=kw_c)
    assert torch.allclose(p_bat, p_seq, rtol=1e-5, atol=1e-6)


def test_axiom_v4_word_fixture_active_rows_and_fast_kernels_parity() -> None:
    torch.manual_seed(0)

    # 1) Instance-local param._axiom_active_rows + unbound _active_record fallback
    ptab = nn.Parameter(torch.randn(10, 4))
    opt_p = AxiomOptimizer([{"params": [ptab], "lr": 0.05, "role": "prior"}])
    opt_p.set_sampled_rows(ptab, torch.tensor([1, 3]))
    opt_p.set_sampled_rows(ptab, torch.tensor([5]))
    assert hasattr(ptab, "_axiom_active_rows") and torch.equal(ptab._axiom_active_rows, torch.tensor([1, 3, 5]))
    rec_w = AxiomStepRecord()
    rec_w.thermal_scale = 0.0061
    ptab._axiom_record = rec_w
    assert opt_p._active_record() is rec_w
    # Schema must stay unchanged (no "regularizer" key when opt_p.record is None)
    assert "regularizer" not in opt_p.state_dict()
    opt_p.clear_sampled_rows()
    assert not hasattr(ptab, "_axiom_active_rows")

    # 2) pre_normalized=True matches pre_normalized=False in fp32_tikhonov_ns_batched
    raw_m = torch.randn(4, 16, 16)
    normed = raw_m / raw_m.flatten(1).norm(dim=1).view(4, 1, 1)
    for rs in (0.0, 0.05):
        o_ref = fp32_tikhonov_ns_batched(raw_m, steps=5, rel_smooth=rs, pre_normalized=False)
        o_fast = fp32_tikhonov_ns_batched(normed, steps=5, rel_smooth=rs, pre_normalized=True)
        assert torch.allclose(o_ref, o_fast, rtol=1e-5, atol=1e-6)

    # 3) 1-sync closed-form zero_pinv_project_nonascent across 1 & 2 acute/obtuse/antiparallel constraints
    d = torch.tensor([1.0, 2.0, -0.5, 0.3], dtype=torch.float32)
    for normals in (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 0.0, 0.0], [-1.0, -2.0, 0.0, 0.0]]),  # antiparallel -> 0
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),  # 1 zero normal
    ):
        proj = zero_pinv_project_nonascent(d, normals)
        assert bool((normals.double() @ proj.double() <= 1e-9).all())
        assert float(proj.norm().item()) <= float(d.norm().item()) + 1e-6

    # 4) 4D ungrouped 1x1 Conv2d == 2D Linear exact single-step update parity on BOTH G and D
    g_rng = torch.Generator().manual_seed(601)
    w0 = torch.randn((32, 32), generator=g_rng) * 0.05
    grad0 = torch.randn((32, 32), generator=g_rng) * 0.02
    g_lin, d_lin = nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
    g_c1, d_c1 = nn.Conv2d(32, 32, 1, bias=False), nn.Conv2d(32, 32, 1, bias=False)
    with torch.no_grad():
        g_lin.weight.copy_(w0); d_lin.weight.copy_(w0)
        g_c1.weight.copy_(w0.view(32, 32, 1, 1)); d_c1.weight.copy_(w0.view(32, 32, 1, 1))
    og_l = AxiomOptimizer(g_lin.parameters(), lr=0.014)
    og_c = AxiomOptimizer(g_c1.parameters(), lr=0.014, conv_layouts=collect_conv_layouts((g_c1,)))
    od_l = AxiomOptimizer(d_lin.parameters(), lr=0.021, critic=d_lin)
    od_c = AxiomOptimizer(d_c1.parameters(), lr=0.021, critic=d_c1, conv_layouts=collect_conv_layouts((d_c1,)))
    g_lin.weight.grad, d_lin.weight.grad = grad0.clone(), grad0.clone()
    g_c1.weight.grad, d_c1.weight.grad = grad0.view(32, 32, 1, 1).clone(), grad0.view(32, 32, 1, 1).clone()
    og_l.step(); og_c.step(); od_l.step(); od_c.step()
    assert torch.allclose(g_c1.weight.detach().view(32, 32) - w0, g_lin.weight.detach() - w0, rtol=1e-5, atol=1e-6)
    assert torch.allclose(d_c1.weight.detach().view(32, 32) - w0, d_lin.weight.detach() - w0, rtol=1e-5, atol=1e-6)
    # ConvTranspose2d 1x1 must NOT be conflated with forward Conv2d 1x1 (stays on non-NS conv path)
    ct1 = nn.ConvTranspose2d(32, 32, 1, bias=False)
    o_ct = AxiomOptimizer(ct1.parameters(), lr=0.014, conv_layouts=collect_conv_layouts((ct1,)))
    ct1.weight.grad = grad0.view(32, 32, 1, 1).clone()
    o_ct.step()
    assert o_ct.ns_2d_activations == 0


def test_axiom_v4e_spectral_split_deadband_free_and_continuation() -> None:
    torch.manual_seed(0)

    # 1) V4E spectral split: on step 2 with rho_n ~ 0 (g_eq=1) & rat=1, 2D matrix has rs=0 (no s_eq floor),
    #    whereas 1D vector applies s_eq_eff = 0.05 damping.
    g_rng = torch.Generator().manual_seed(77)
    w_mat = nn.Parameter(torch.zeros(16, 16))
    v_1d = nn.Parameter(torch.zeros(16))
    opt_s = AxiomOptimizer([w_mat, v_1d], lr=0.02, smoothing=0.05)
    g1_m = torch.randn((16, 16), generator=g_rng)
    g1_v = torch.randn((16,), generator=g_rng)
    w_mat.grad, v_1d.grad = g1_m.clone(), g1_v.clone()
    opt_s.step()  # Step 1 (had_hist=False): sets sketch & log2_ref
    # Step 2 with orthogonal grad of equal norm -> rho_n ~ 0 (g_eq=1, s_rel=0), rat=1
    w_before, v_before = w_mat.detach().clone(), v_1d.detach().clone()
    g2_m = torch.randn((16, 16), generator=g_rng); g2_m = g2_m * (g1_m.norm() / g2_m.norm())
    g2_v = torch.randn((16,), generator=g_rng); g2_v = g2_v * (g1_v.norm() / g2_v.norm())
    w_mat.grad, v_1d.grad = g2_m.clone(), g2_v.clone()
    opt_s.step()
    dw_actual = -(w_mat.detach() - w_before) / 0.02
    dw_expected_s005 = fp32_tikhonov_ns_2d(g2_m, steps=5, rel_smooth=0.05)
    assert torch.allclose(dw_actual, dw_expected_s005, rtol=1e-4, atol=1e-4), "O1: 2D NS must restore s_eq_base * g_eq = 0.05 equilibrium smoothing"
    dv_norm = float((v_1d.detach() - v_before).norm().item()) / 0.02
    assert abs(dv_norm - 1.0 / math.sqrt(1.0 + 0.05 * 0.05)) < 5e-4, f"1D expected s_eq=0.05 damping, got norm={dv_norm:.6f}"

    # 2) Deadband-free log2_damp decay (< -0.75) + fast recovery (> -0.30) + bit-exact checkpoint across boundary
    p_a = nn.Parameter(torch.zeros(8, 8))
    p_b = nn.Parameter(torch.zeros(8, 8))
    oa, ob = AxiomOptimizer([p_a], lr=0.01, smoothing=0.05), AxiomOptimizer([p_b], lr=0.01, smoothing=0.05)
    g0 = torch.randn((8, 8), generator=torch.Generator().manual_seed(91))
    for k in range(35):
        sg = g0 if (k % 2 == 0) else -g0  # pure alternating chatter -> rho_n -> -1
        p_a.grad = sg.clone(); oa.step()
        if k < 20:
            p_b.grad = sg.clone(); ob.step()
        elif k == 20:
            sd = deepcopy(ob.state_dict())
            p_b = nn.Parameter(p_b.detach().clone())
            ob = AxiomOptimizer([p_b], lr=0.01, smoothing=0.05)
            ob.load_state_dict(sd)
            p_b.grad = sg.clone(); ob.step()
        else:
            p_b.grad = sg.clone(); ob.step()
    assert oa.log2_damp < -0.5, f"log2_damp must decay under alternating chatter, got {oa.log2_damp}"
    assert torch.equal(p_a.detach(), p_b.detach()) and oa.log2_damp == ob.log2_damp
    d_low = oa.log2_damp
    for _ in range(30):
        p_a.grad = g0.clone(); oa.step()
        p_b.grad = g0.clone(); ob.step()
    assert oa.log2_damp > d_low + 0.5 and oa.log2_damp == 0.0, f"deadband-free recovery failed: {d_low} -> {oa.log2_damp}"
    assert torch.equal(p_a.detach(), p_b.detach())


@pytest.mark.cuda
def test_axiom_v4_cuda_bf16_instancenorm_groupnorm_fwad() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    xr80, xf80 = torch.randn((8, 3, 80, 80), device=dev), torch.randn((8, 3, 80, 80), device=dev)
    xr16, xf16 = torch.randn((8, 3, 16, 16), device=dev), torch.randn((8, 3, 16, 16), device=dev)
    cases = (
        (nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.InstanceNorm2d(8, affine=True), nn.SiLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(8 * 4 * 4, 1)).to(dev), xr80, xf80, "fwad"),
        (nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.GroupNorm(2, 8), nn.SiLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.LayerNorm(8 * 4 * 4), nn.Linear(8 * 4 * 4, 1)).to(dev), xr80, xf80, "ror"),
        (nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.BatchNorm2d(8), nn.SiLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(8 * 4 * 4, 1)).to(dev), xr80, xf80, "fwad"),
        (nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.InstanceNorm2d(8, affine=True), nn.SiLU(), nn.Flatten(), nn.Linear(8 * 16 * 16, 1)).to(dev), xr16, xf16, "fwad"),
    )
    for mod, xr, xf, exp_regime in cases:
        kern = AxiomPenaltyKernel(record=AxiomStepRecord(), coeff=1.0, kappa=1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pen, st = kern.penalty(mod, xr, xf, collect_stats=True, critic_module=mod)
        assert st["regime"] == exp_regime and torch.isfinite(pen)
        pen.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())


def test_axiom_bn_scale_awareness_period2_stability_and_nonfreezing() -> None:
    torch.manual_seed(0)

    # (a) Active-BN scoping: _bn_net_ids tags active BN modules, excludes No-BN spatial Convs
    #     (like img_blobs4) and excludes frozen eval-mode BN backbones (like FrozenResNet18).
    nobn_net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 3, padding=1))
    opt_nobn = AxiomOptimizer(nobn_net.parameters(), lr=0.014)
    opt_nobn.bind_module_layouts((nobn_net,))
    assert len(opt_nobn._bn_net_ids) == 0

    frozen_bn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    frozen_bn.eval()
    for p in frozen_bn.parameters():
        p.requires_grad_(False)
    head = nn.Sequential(nn.GroupNorm(2, 8), nn.Conv2d(8, 1, 3, padding=1))
    hybrid = nn.Sequential(frozen_bn, head)
    opt_hyb = AxiomOptimizer(head.parameters(), lr=0.014)
    opt_hyb.bind_module_layouts((hybrid,))
    assert len(opt_hyb._bn_net_ids) == 0

    # (b) 1D BN/bias period-2 oscillation must NOT drag down global ema_step_cos / log2_damp
    #     and unit-renormalized BN momentum (b1=0.50 on 1D) tracks smooth directions cleanly.
    net = nn.Sequential(
        nn.Conv2d(4, 8, 4, stride=2, padding=1, bias=True),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 4, 4, stride=2, padding=1, bias=True),
    )
    opt = AxiomOptimizer(net.parameters(), lr=0.014)
    opt.bind_module_layouts((net,))
    assert len(opt._bn_net_ids) == len(list(net.parameters()))

    g_rng = torch.Generator().manual_seed(123)
    steady_w = {id(p): torch.randn(p.shape, generator=g_rng) for p in net.parameters() if p.ndim >= 2}
    osc_1d = {id(p): torch.ones_like(p) for p in net.parameters() if p.ndim == 1}
    for t in range(30):
        sgn = 1.0 if t % 2 == 0 else -1.0
        for p in net.parameters():
            p.grad = steady_w[id(p)].clone() if p.ndim >= 2 else (osc_1d[id(p)] * sgn)
        opt.step()
    assert opt.ema_step_cos > 0.7 and opt.log2_damp == 0.0, (
        f"1D BN/bias chatter froze global damping: cos={opt.ema_step_cos}, log2_damp={opt.log2_damp}"
    )
    for p in net.parameters():
        assert "m_u" in opt.state[p] and opt.state[p]["m_u"].dtype == torch.float32

    # (c) Odd-stride sketch detects power-of-2 channel oscillation on BN + ConvTranspose2d(128, 64, 4, 4)
    #     (where n // 128 = 1024 would alias onto a single channel!).
    #     When ema_func_gain >= 0.60, BN networks do not freeze log2_damp; when secant probe
    #     confirms overshoot (ema_func_gain < 0.60), log2_damp damps and recovers deadzone-free.
    bn_ct = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64))
    opt_ct = AxiomOptimizer(bn_ct.parameters(), lr=0.014)
    opt_ct.bind_module_layouts((bn_ct,))
    g_ch = torch.zeros_like(bn_ct[0].weight)
    g_ch[:, 1::2, :, :] = 1.0  # odd output channels oscillate; stride 1024 would only hit channel 0!
    for t in range(40):
        bn_ct[0].weight.grad = g_ch if t % 2 == 0 else -g_ch
        opt_ct.step()
    assert opt_ct.ema_step_cos < -0.5 and opt_ct.log2_damp == 0.0, (
        f"Healthy BN network should detect cosine without freezing: cos={opt_ct.ema_step_cos}, damp={opt_ct.log2_damp}"
    )
    opt_ct._ema_func_gain = 0.50  # secant-confirmed functional overshoot unlocks damping
    for t in range(30):
        bn_ct[0].weight.grad = g_ch if t % 2 == 0 else -g_ch
        opt_ct.step()
    assert opt_ct.log2_damp < -0.5, f"Secant-confirmed BN oscillation failed to damp: damp={opt_ct.log2_damp}"
    d_lo = opt_ct.log2_damp
    for _ in range(35):
        bn_ct[0].weight.grad = g_ch.clone()
        opt_ct.step()
    assert opt_ct.log2_damp > d_lo + 0.5 and opt_ct.log2_damp == 0.0, (
        f"Deadzone-free recovery failed on BN spatial Conv: {d_lo} -> {opt_ct.log2_damp}"
    )

    # (d) GANTrainer + Recipe('axiom') automatically binds BN/Conv layouts on both opt_g and opt_d
    import particlegan as pg

    g_mod = nn.Sequential(nn.Linear(4, 16, bias=False), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4))
    d_mod = nn.Sequential(nn.Linear(4, 16, bias=False), nn.BatchNorm1d(16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    rec = pg.get_recipe("axiom", z_dim=4, num_particles=64, batch_size=16)
    tr = pg.GANTrainer(rec, g_mod, d_mod, seed=0)
    assert len(tr.opt_g._bn_net_ids) > 0 and len(tr.opt_d._bn_net_ids) > 0
    for _ in range(5):
        out = tr.step(torch.randn(16, 4))
        assert math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))
    assert tr.opt_g.log2_damp == 0.0 and tr.opt_d.log2_damp == 0.0


def test_axiom_bn_momentum_scale_invariance_and_critic_dim_x() -> None:
    torch.manual_seed(0)

    def _make_bn_conv() -> nn.Sequential:
        torch.manual_seed(42)
        return nn.Sequential(
            nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 3, 4, stride=2, padding=1, bias=True),
        )

    base = _make_bn_conv()
    g_rng = torch.Generator().manual_seed(777)
    step_grads = [[torch.randn(p.shape, generator=g_rng) for p in base.parameters()] for _ in range(5)]

    ref_params: list[torch.Tensor] | None = None
    for scale in (1e-30, 1e-18, 1.0, 1e18, 1e30):
        mod = _make_bn_conv()
        opt = AxiomOptimizer(mod.parameters(), lr=0.014)
        opt.bind_module_layouts((mod,))
        for grads in step_grads:
            for p, g in zip(mod.parameters(), grads):
                p.grad = g * scale
            opt.step()
        cur = [p.detach().clone() for p in mod.parameters()]
        if ref_params is None:
            ref_params = cur
        else:
            for a, b in zip(ref_params, cur):
                assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), f"BN momentum scale invariance failed at scale={scale}"

    # Nonfinite gradients must leave BN parameters finite and completely untouched
    mod_nf = _make_bn_conv()
    opt_nf = AxiomOptimizer(mod_nf.parameters(), lr=0.014)
    opt_nf.bind_module_layouts((mod_nf,))
    for bad in (float("inf"), float("nan")):
        before = [p.detach().clone() for p in mod_nf.parameters()]
        for p in mod_nf.parameters():
            p.grad = torch.full_like(p, bad)
        opt_nf.step()
        for p, w0 in zip(mod_nf.parameters(), before):
            assert torch.isfinite(p).all() and torch.equal(p.detach(), w0)

    # High-dim 4D image critic penalty (dim_x = 3072 > 256): exact x -> c*x scale equivariance
    # AND no low-rank _eff_dim_t inflation on smooth correlated 32x32 RGB batches!
    torch.manual_seed(7)
    z = torch.randn(16, 4)
    w_proj = torch.randn(4, 3 * 32 * 32) / math.sqrt(4.0)
    xr_base = (z @ w_proj).view(16, 3, 32, 32)  # rank-4 32x32 RGB images (eff_dim << 3072)
    xf_base = ((z + 0.1 * torch.randn_like(z)) @ w_proj).view(16, 3, 32, 32)
    w_lin = torch.randn(3 * 32 * 32) / math.sqrt(3 * 32 * 32)
    pens: list[float] = []
    for c in (1e-3, 1.0, 1e3):
        rec = AxiomStepRecord()
        rec.margin_initialized = True
        rec.ema_data_scale = float(c * math.sqrt(3 * 32 * 32))
        kern = AxiomPenaltyKernel(record=rec, coeff=1.0, kappa=1.0)
        # Scale-equivariant linear critic D_c(x) = <w_lin / c, x> has ||grad D_c||_2 = ||w_lin||_2 / c
        # Under sc_t = ema_data_scale / sqrt(dim_x) = c, scaled grad norm = ||w_lin||_2 ~ 1.0 regardless of c or rank!
        crit_fn = lambda x, scale=c: (x.flatten(1) @ (w_lin / scale))
        pen, st = kern.penalty(crit_fn, xr_base * c, xf_base * c, collect_stats=True)
        assert torch.isfinite(pen)
        pens.append(float(pen.item()))
    assert abs(pens[0] - pens[1]) < 1e-4 and abs(pens[2] - pens[1]) < 1e-4, f"4D dim_x scale equivariance failed: {pens}"
    assert pens[1] < 0.05, f"Low-rank 32x32 RGB batch inflated critic penalty via eff_dim: pen={pens[1]}"


def test_axiom_bn_checkpoint_continuation_and_tamper_rejection() -> None:
    import particlegan as pg

    torch.manual_seed(0)

    def _make_pair():
        torch.manual_seed(11)
        g = nn.Sequential(nn.Linear(4, 4 * 4 * 4, bias=False), nn.Unflatten(1, (4, 4, 4)), nn.BatchNorm2d(4), nn.ReLU(), nn.ConvTranspose2d(4, 3, 4, stride=2, padding=1))
        d = nn.Sequential(nn.Conv2d(3, 4, 4, stride=2, padding=1), nn.BatchNorm2d(4), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(4 * 4 * 4, 1))
        return g, d

    rec = pg.get_recipe("axiom", z_dim=4, num_particles=64, batch_size=8)
    g_a, d_a = _make_pair()
    tr_a = pg.GANTrainer(rec, g_a, d_a, seed=0)
    g_b, d_b = _make_pair()
    tr_b = pg.GANTrainer(rec, g_b, d_b, seed=0)

    b_rng = torch.Generator().manual_seed(99)
    batches = [torch.randn((8, 3, 8, 8), generator=b_rng) for _ in range(6)]
    for t in range(3):
        tr_a.step(batches[t])
        tr_b.step(batches[t])

    # Fresh trainer & models loaded from step-3 checkpoint must match uninterrupted run exactly
    ckpt = deepcopy(tr_b.state_dict())
    g_c, d_c = _make_pair()
    tr_c = pg.GANTrainer(rec, g_c, d_c, seed=12345)
    tr_c.load_state_dict(ckpt)
    for t in range(3, 6):
        tr_a.step(batches[t])
        tr_c.step(batches[t])

    for pa, pc in zip(tr_a.G.parameters(), tr_c.G.parameters()):
        assert torch.allclose(pa.detach(), pc.detach(), rtol=0.0, atol=1e-6)
    for pa, pc in zip(tr_a.D.parameters(), tr_c.D.parameters()):
        assert torch.allclose(pa.detach(), pc.detach(), rtol=0.0, atol=1e-6)
    assert math.isclose(tr_a.opt_g.ema_step_cos, tr_c.opt_g.ema_step_cos, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(tr_a.opt_d.log2_damp, tr_c.opt_d.log2_damp, rel_tol=0.0, abs_tol=1e-6)

    # Strict tamper rejection on BN momentum state (m_u)
    base_sd = deepcopy(tr_a.opt_d.state_dict())
    pid_mu = next(k for k, v in base_sd["state"].items() if "m_u" in v)
    for mutate in (
        lambda s: s["state"][pid_mu]["m_u"].fill_(float("nan")),
        lambda s: s["state"][pid_mu]["m_u"].fill_(float("inf")),
        lambda s: s["state"][pid_mu].__setitem__("m_u", torch.ones(1)),
        lambda s: s["state"][pid_mu].__setitem__("m_u", "corrupt"),
    ):
        bad_sd = deepcopy(base_sd)
        mutate(bad_sd)
        with pytest.raises(ValueError, match="invalid AxiomOptimizer parameter state"):
            tr_a.opt_d.validate_state_dict(bad_sd)

    # Pure bfloat16 & float16 model checkpoint round-trip must preserve FP32 m_u bit-for-bit without downcast truncation
    for dt in (torch.bfloat16, torch.float16):
        def _make_lowp(dtype=dt) -> nn.Sequential:
            torch.manual_seed(314)
            return nn.Sequential(nn.Conv2d(3, 8, 4, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU()).to(dtype)

        m_ref, m_res = _make_lowp(), _make_lowp()
        o_ref, o_res = AxiomOptimizer(m_ref.parameters(), lr=0.014), AxiomOptimizer(m_res.parameters(), lr=0.014)
        o_ref.bind_module_layouts((m_ref,))
        o_res.bind_module_layouts((m_res,))
        lp_rng = torch.Generator().manual_seed(271)
        lp_batches = [torch.randn((4, 3, 8, 8), generator=lp_rng, dtype=dt) for _ in range(5)]
        for t in range(3):
            for m, o in ((m_ref, o_ref), (m_res, o_res)):
                o.zero_grad(set_to_none=True)
                m(lp_batches[t]).square().mean().backward()
                o.step()
        sd_lp = deepcopy(o_res.state_dict())
        w_lp = deepcopy(m_res.state_dict())
        m_fresh = _make_lowp()
        m_fresh.load_state_dict(w_lp)
        o_fresh = AxiomOptimizer(m_fresh.parameters(), lr=0.014)
        o_fresh.bind_module_layouts((m_fresh,))
        o_fresh.load_state_dict(sd_lp)
        for t in range(3, 5):
            for m, o in ((m_ref, o_ref), (m_fresh, o_fresh)):
                o.zero_grad(set_to_none=True)
                m(lp_batches[t]).square().mean().backward()
                o.step()
        for p1, p2 in zip(m_ref.parameters(), m_fresh.parameters()):
            assert torch.equal(p1.detach(), p2.detach()), f"{dt} parameter continuation mismatch"
            assert o_fresh.state[p2]["m_u"].dtype == torch.float32
            assert torch.equal(o_ref.state[p1]["m_u"], o_fresh.state[p2]["m_u"]), f"{dt} m_u continuation mismatch"






