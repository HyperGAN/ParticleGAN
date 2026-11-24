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
    c1 = nn.Sequential(nn.Conv2d(3, 6, 3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(6 * 8 * 8, 1))
    c2 = deepcopy(c1)
    xr, xf = torch.randn(8, 3, 16, 16), torch.randn(8, 3, 16, 16)

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

    # 3) BatchNorm2d eval() safety during fwAD (zero running_mean/var drift, .training restored)
    bn_critic = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(8 * 8 * 8, 1))
    bn_critic.train()
    bn = bn_critic[1]
    rm_before = bn.running_mean.clone()
    rv_before = bn.running_var.clone()
    kbn = AxiomPenaltyKernel(record=AxiomStepRecord())
    pbn, sbn = kbn.penalty(bn_critic, torch.randn(8, 3, 16, 16) * 5.0 + 3.0, torch.randn(8, 3, 16, 16) * 2.0 - 4.0, collect_stats=True)
    assert sbn["regime"] == "fwad" and bn.training is True
    assert torch.equal(bn.running_mean, rm_before) and torch.equal(bn.running_var, rv_before)
    pbn.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bn_critic.parameters())


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

    # 2) Role-agnostic 2D dense NS on shallow 3L critic AND generator identically (0 NS on 1x16 output & Conv2d)
    d3 = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1))
    g3 = deepcopy(d3)
    opt_d3 = AxiomOptimizer(d3.parameters(), lr=0.01, critic=d3)
    opt_g3 = AxiomOptimizer(g3.parameters(), lr=0.01)
    for pd, pg in zip(d3.parameters(), g3.parameters()):
        g = ((torch.randn(pd.shape[0], 1) @ torch.randn(1, pd.shape[1])) + 0.05 * torch.randn_like(pd)) if pd.ndim == 2 else torch.randn_like(pd)
        pd.grad, pg.grad = g.clone(), g.clone()
    opt_d3.step()
    opt_g3.step()
    assert opt_d3.ns_2d_activations == 2 and opt_g3.ns_2d_activations == 2
    for pd, pg in zip(d3.parameters(), g3.parameters()):
        assert torch.allclose(pd, pg, rtol=1e-6, atol=1e-6)

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
            d_obs = float(torch.quantile(rms, 0.95).item())
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

