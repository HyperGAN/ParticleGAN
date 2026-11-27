"""Unit tests for Axiom v6 L1 Law 2/4.5 critic penalty (`particlegan._axiom.critic`)."""

from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from particlegan._axiom import critic as axiom_critic
from particlegan._axiom.critic import (
    AxiomCriticPenalty,
    AxiomPenaltyKernel,
    AxiomStepRecord,
    _attach_axiom_critic_hook,
    _secant_sparsity,
    effective_dimension,
    probe_schedule,
)
from particlegan.low_precision import emulate_linear_training_


# ---------------------------------------------------------------------------
# 1. Static Hygiene: Zero autograd.grad, Zero Legacy Imports, Zero Sniffing
# ---------------------------------------------------------------------------


def test_critic_module_ast_hygiene_and_zero_forbidden_ops() -> None:
    source = Path(axiom_critic.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Strictly acyclic L1 imports: ONLY math_ops and state inside _axiom.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                assert node.level == 1 and mod in {"math_ops", "state"}, (
                    f"Unexpected relative import: {node.level}.{mod}"
                )
            for legacy in (
                "_unified",
                "bispf",
                "rhbi",
                "empf",
                "minimal",
                "k3p",
                "ka2",
                "grad_regularizers",
                "_t1",
                "_t2",
                "_t3",
                "_t4",
            ):
                assert legacy not in mod, f"Forbidden legacy import {mod!r}"

    # Zero autograd.grad / svd / eigh / pinv / compile.
    forbidden_attrs = {"svd", "svdvals", "eigh", "eigvalsh", "pinv", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs, f"Forbidden op {node.attr!r}"
            if (
                node.attr == "grad"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "autograd"
            ):
                raise AssertionError("Forbidden torch.autograd.grad found in critic.py")

    # Zero class-name string sniffing.
    for sniff in ("BatchDistance", "Emulated", "nvfp4", "mxfp8"):
        assert sniff not in source, (
            f"Forbidden class-name sniffing string {sniff!r} found in critic.py"
        )


# ---------------------------------------------------------------------------
# 2. Law 2.1 Scale Equivariance, Effective Dimension & Continuous Element Budget
# ---------------------------------------------------------------------------


def test_effective_dimension_and_continuous_probe_schedule() -> None:
    torch.manual_seed(0)
    # Isotropic R^64 -> d_eff ~ 64; O(D) rotation invariant; rank-3 subspace -> d_eff ~ 3.
    x_iso = torch.randn(64, 64)
    d_iso = float(effective_dimension(x_iso).item())
    assert 48.0 <= d_iso <= 64.0
    q, _ = torch.linalg.qr(torch.randn(64, 64))
    assert math.isclose(
        float(effective_dimension(x_iso @ q).item()), d_iso, rel_tol=1e-4
    )

    low_rank = torch.randn(64, 3) @ torch.randn(3, 64)
    d_lr = float(effective_dimension(low_rank).item())
    assert 1.5 <= d_lr <= 4.5

    # Scale invariance across 1e-6 .. 1e6 and constant point-mass safety.
    assert math.isclose(
        float(effective_dimension(x_iso * 1e-6).item()), d_iso, rel_tol=1e-4
    )
    assert math.isclose(
        float(effective_dimension(x_iso * 1e6).item()), d_iso, rel_tol=1e-4
    )
    assert float(effective_dimension(torch.ones(16, 4)).item()) == 4.0

    # Continuous element-budget probe/sub-batch schedule.
    assert probe_schedule(256, 2) == (2, 256)
    assert probe_schedule(256, 4) == (4, 256)
    assert probe_schedule(64, 8) == (2, 64)
    assert probe_schedule(128, 32) == (1, 128)
    assert probe_schedule(32, 3072) == (1, 10)
    assert probe_schedule(16, 3 * 256 * 256) == (1, 4)


def test_scale_equivariance_across_ten_decades_standalone_and_tracked() -> None:
    torch.manual_seed(11)
    w_lin = torch.randn(4, 1)
    xr0, xf0 = torch.randn(16, 4), torch.randn(16, 4)

    for track_margin in (False, True):
        pens, shats = [], []
        for scale in (1e-5, 1.0, 1e5):
            rec = AxiomStepRecord()
            if track_margin:
                rec.observe_margin_and_scale(
                    torch.ones(16, 1), -torch.ones(16, 1), xr0 * scale, xf0 * scale
                )
            kern = AxiomPenaltyKernel(record=rec, coeff=1.0, kappa=1.0)
            pen, stats = kern.penalty(
                lambda x, c=scale: (x / c) @ w_lin,
                xr0 * scale,
                xf0 * scale,
                collect_stats=True,
            )
            assert stats["regime"] == "fwad"
            pens.append(float(pen.item()))
            shats.append(float(stats["s_hat"]))
        assert math.isclose(shats[0], shats[1], rel_tol=1e-3)
        assert math.isclose(shats[1], shats[2], rel_tol=1e-3)
        assert math.isclose(pens[0], pens[1], rel_tol=1e-3)
        assert math.isclose(pens[1], pens[2], rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 3. Law 2.3 fwAD Accuracy Across Architectures & BN Stat Preservation
# ---------------------------------------------------------------------------


def test_fwad_accuracy_on_mlp_conv_bn_groupnorm_and_vit_plus_bn_preservation() -> None:
    torch.manual_seed(0)

    # 1) Exact analytical verification on linear critic in R^4 (K == d -> exact 0-variance s_hat).
    w = torch.tensor([[1.5], [-0.5], [0.75], [-1.25]])
    lin = nn.Linear(4, 1, bias=True)
    with torch.no_grad():
        lin.weight.copy_(w.T)
        lin.bias.zero_()
    rec = AxiomStepRecord()
    xr = torch.randn(32, 4)
    xf = torch.randn(32, 4)
    rec.observe_margin_and_scale(torch.ones(32, 1), -torch.ones(32, 1), xr, xf)
    kern = AxiomPenaltyKernel(record=rec, coeff=1.0, kappa=1.0)
    pen, st = kern.penalty(lin, xr, xf, collect_stats=True)
    d_eff = float(effective_dimension(xr).item())
    s_scale = float(rec.ema_data_scale) / math.sqrt(d_eff)
    expected_shat = (s_scale**2) * float(w.square().sum().item())
    assert st["regime"] == "fwad" and st["k_probes"] == 4
    assert math.isclose(st["s_hat"], expected_shat, rel_tol=1e-4)
    pen.backward()
    assert lin.weight.grad is not None and bool(torch.isfinite(lin.weight.grad).all())

    # 2) Conv2d + BatchNorm2d: fwAD succeeds AND preserves BN training mode + running stats!
    conv_bn = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    conv_bn.train()
    x_img = torch.randn(8, 3, 16, 16)
    _ = conv_bn(x_img * 2.0 + 1.0)
    rm0 = conv_bn[1].running_mean.clone()
    rv0 = conv_bn[1].running_var.clone()
    nb0 = conv_bn[1].num_batches_tracked.clone()

    k_bn = AxiomPenaltyKernel(record=AxiomStepRecord())
    pen_bn, st_bn = k_bn.penalty(
        conv_bn, x_img, torch.randn_like(x_img), collect_stats=True
    )
    assert st_bn["regime"] == "fwad" and torch.isfinite(pen_bn)
    assert conv_bn.training is True and conv_bn[1].training is True
    assert torch.equal(conv_bn[1].running_mean, rm0)
    assert torch.equal(conv_bn[1].running_var, rv0)
    assert torch.equal(conv_bn[1].num_batches_tracked, nb0)
    pen_bn.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in conv_bn.parameters()
    )

    # 3) GroupNorm + Conv2d runs natively on fwAD (no autograd.grad fallback!).
    gn_net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.GroupNorm(2, 8),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    pen_gn, st_gn = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(
        gn_net, x_img, torch.randn_like(x_img), collect_stats=True
    )
    assert st_gn["regime"] == "fwad" and torch.isfinite(pen_gn)
    pen_gn.backward()

    # 4) Smooth ViT critic (Patch Conv + LayerNorm + normalized attention) runs natively on fwAD,
    #    while nn.MultiheadAttention(need_weights=False) (flash SDPA without jvp) auto-falls back to secant!
    class _SmoothViTCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch = nn.Conv2d(3, 16, kernel_size=4, stride=4)
            self.ln = nn.LayerNorm(16)
            self.qkv = nn.Linear(16, 48)
            self.head = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tok = self.patch(x).flatten(2).transpose(1, 2)
            q, k, v = self.qkv(self.ln(tok)).chunk(3, dim=-1)
            w = torch.sigmoid(torch.matmul(q, k.transpose(-2, -1)) * 0.25)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return self.head((tok + torch.matmul(w, v)).mean(dim=1))

    vit_smooth = _SmoothViTCritic()
    pen_vit, st_vit = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(
        vit_smooth, x_img, torch.randn_like(x_img), collect_stats=True
    )
    assert st_vit["regime"] == "fwad" and torch.isfinite(pen_vit)
    pen_vit.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in vit_smooth.parameters()
    )

    class _FlashMHACritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch = nn.Conv2d(3, 16, kernel_size=4, stride=4)
            self.attn = nn.MultiheadAttention(16, num_heads=2, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tok = self.patch(x).flatten(2).transpose(1, 2)
            out, _ = self.attn(tok, tok, tok, need_weights=False)
            return self.head(out.mean(dim=1))

    vit_mha = _FlashMHACritic()
    pen_mha, st_mha = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(
        vit_mha, x_img, torch.randn_like(x_img), collect_stats=True
    )
    assert st_mha["regime"] == "secant" and torch.isfinite(pen_mha)
    pen_mha.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in vit_mha.parameters()
    )


# ---------------------------------------------------------------------------
# 4. Capability-Based Precision-Aware Central Secant Fallback (Zero Sniffing)
# ---------------------------------------------------------------------------


def test_automatic_secant_fallback_on_emulated_quantized_and_low_precision_ulp() -> (
    None
):
    torch.manual_seed(0)

    # 1) EmulatedLinear (MXFP8 & NVFP4) automatically triggers secant fallback and caches id.
    for fmt in ("mxfp8", "nvfp4"):
        critic = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
        emulate_linear_training_(critic, fmt)
        kern = AxiomPenaltyKernel(record=AxiomStepRecord(), coeff=1.0, kappa=1.0)
        xr, xf = torch.randn(16, 2), torch.randn(16, 2)
        pen1, st1 = kern.penalty(critic, xr, xf, collect_stats=True)
        assert (
            st1["regime"] == "secant"
            and torch.isfinite(pen1)
            and float(pen1.item()) > 0.0
        )
        assert id(critic) in kern._fwad_unsupported
        pen1.backward()
        assert any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in critic.parameters()
        )
        # Second call goes directly to secant via cached capability flag.
        pen2, st2 = kern.penalty(critic, xr, xf, collect_stats=True)
        assert st2["regime"] == "secant" and torch.isfinite(pen2)

    # 2) Precision-aware sparsity prevents ULP underflow in BF16 / FP16 on high-D critics.
    assert _secant_sparsity(1024, 0.02, torch.float32) == 1024
    assert 1 < _secant_sparsity(1024, 0.02, torch.float16) < 64
    assert _secant_sparsity(1024, 0.02, torch.bfloat16) == 1

    class _NoJvpOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x, w)
            return x @ w

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x, w = ctx.saved_tensors
            return grad_out @ w.T, x.T @ grad_out

    class _NoJvpCritic(nn.Module):
        def __init__(self, dim: int, dtype: torch.dtype) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(dim, 1, dtype=dtype) / math.sqrt(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _NoJvpOp.apply(x, self.w)

    for dt in (torch.bfloat16, torch.float16):
        c_lp = _NoJvpCritic(512, dt)
        xr_lp = torch.randn(16, 512, dtype=dt)
        xf_lp = torch.randn(16, 512, dtype=dt)
        kern_lp = AxiomPenaltyKernel(record=AxiomStepRecord())
        pen_lp, st_lp = kern_lp.penalty(c_lp, xr_lp, xf_lp, collect_stats=True)
        assert st_lp["regime"] == "secant"
        assert torch.isfinite(pen_lp) and st_lp["s_hat"] > 0.0
        pen_lp.backward()
        assert c_lp.w.grad is not None and bool(torch.isfinite(c_lp.w.grad).all())


# ---------------------------------------------------------------------------
# 5. Exact-Basis Zero-Variance vs Stochastic Unbiasedness, Live Gauge & No-RNG
# ---------------------------------------------------------------------------


def test_exact_basis_zero_variance_vs_stochastic_unbiased_and_zero_jensen() -> None:
    torch.manual_seed(0)

    # Exact basis (d=3, K=3): s_hat is deterministic across chord_phase shifts (0 variance).
    w3 = torch.tensor([[1.2], [-0.8], [0.5]])
    lin3 = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        lin3.weight.copy_(w3.T)
    xr3, xf3 = torch.randn(32, 3), torch.randn(32, 3)
    shats_3 = []
    for phase in (0.0, 0.27, 0.63):
        rec = AxiomStepRecord()
        rec.chord_phase = phase
        rec.observe_margin_and_scale(torch.ones(32, 1), -torch.ones(32, 1), xr3, xf3)
        _, st = AxiomPenaltyKernel(record=rec).penalty(
            lin3, xr3, xf3, collect_stats=True
        )
        shats_3.append(st["s_hat"])
    assert math.isclose(shats_3[0], shats_3[1], rel_tol=1e-6)
    assert math.isclose(shats_3[1], shats_3[2], rel_tol=1e-6)

    # Stochastic (d=32, K=1): average s_hat over many chord phases matches exact ||g||^2 * s_scale^2,
    # AND penalty is strictly linear in s_hat (zero Jensen bias below kappa).
    w32 = torch.randn(32, 1) * 0.15  # ||w|| < kappa -> below cap
    lin32 = nn.Linear(32, 1, bias=False)
    with torch.no_grad():
        lin32.weight.copy_(w32.T)
    xr32, xf32 = torch.randn(64, 32), torch.randn(64, 32)
    rec32 = AxiomStepRecord()
    rec32.observe_margin_and_scale(torch.ones(64, 1), -torch.ones(64, 1), xr32, xf32)
    kern32 = AxiomPenaltyKernel(record=rec32, coeff=1.0, kappa=2.0, gamma_gauge=0.0)
    assert kern32.lambda_floor == 0.005 and kern32.max_lambda == 2.0

    shat_samples = []
    for _ in range(40):
        pen, st = kern32.penalty(lin32, xr32, xf32, collect_stats=True)
        shat_samples.append(st["s_hat"])
        # Exact linearity check: pen == 0.5 * coeff * lambda_eff * s_hat!
        assert math.isclose(
            float(pen.item()), 0.5 * st["lambda_eff"] * st["s_hat"], rel_tol=1e-5
        )
    d_eff32 = float(effective_dimension(xr32).item())
    s_sc32 = float(rec32.ema_data_scale) / math.sqrt(d_eff32)
    true_shat32 = (s_sc32**2) * float(w32.square().sum().item())
    assert math.isclose(
        sum(shat_samples) / len(shat_samples), true_shat32, rel_tol=0.15
    )


def test_live_endpoint_reuse_fixed_batch_fallback_wrapper_and_zero_global_rng() -> None:
    torch.manual_seed(7)
    random.seed(13)
    np.random.seed(17)

    critic = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
    opt_d = torch.optim.SGD(critic.parameters(), lr=0.01)
    opt_d.critic = critic  # type: ignore[attr-defined]
    penalty_fn = AxiomCriticPenalty(opt_d, collect_stats=True)
    rec = opt_d.record  # type: ignore[attr-defined]
    assert not hasattr(critic, "_axiom_record") and penalty_fn.ema_critic is None

    xr, xf = torch.randn(8, 4), torch.randn(8, 4)
    # Forward through hooked critic records live endpoints for gauge reuse.
    dr, df = critic(xr), critic(xf)
    cpu_before = torch.get_rng_state().clone()
    py_before = random.getstate()
    np_before = deepcopy(np.random.get_state())

    pen = penalty_fn(critic, xr, xf)
    assert (
        torch.isfinite(pen) and penalty_fn.last_stats["live_endpoints_reused"] is True
    )
    assert penalty_fn.diagnostics()["axiom_calls"] == 1
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0] and np.array_equal(np_after[1], np_before[1])
    (dr.mean() - df.mean() + pen).backward()

    # Fixed-batch closure rejecting concatenated 2B input falls back cleanly to B without string sniffing.
    fixed_ctx = torch.randn(6, 4)
    c_fixed = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    _attach_axiom_critic_hook(c_fixed, rec)
    xr6, xf6 = torch.randn(6, 4), torch.randn(6, 4)
    pen_fb, st_fb = AxiomPenaltyKernel(record=AxiomStepRecord()).penalty(
        lambda x: c_fixed(torch.cat((fixed_ctx, x), dim=-1)),
        xr6,
        xf6,
        collect_stats=True,
        critic_module=c_fixed,
    )
    assert st_fb["regime"] == "fwad" and torch.isfinite(pen_fb)
    pen_fb.backward()


@pytest.mark.cuda
def test_fwad_bf16_autocast_conv_bn_and_groupnorm_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device("cuda:0")
    critic = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.SiLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.GroupNorm(4, 16),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1),
    ).to(dev)
    critic.train()
    xr = torch.randn(8, 3, 32, 32, device=dev)
    xf = torch.randn(8, 3, 32, 32, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _ = critic(xr)
    rm0 = critic[1].running_mean.clone()
    cuda_before = torch.cuda.get_rng_state(dev).clone()

    kern = AxiomPenaltyKernel(record=AxiomStepRecord())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pen, st = kern.penalty(critic, xr, xf, collect_stats=True)
    assert st["regime"] == "fwad" and torch.isfinite(pen)
    assert torch.equal(critic[1].running_mean, rm0)
    assert torch.equal(torch.cuda.get_rng_state(dev), cuda_before)
    pen.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in critic.parameters()
    )
