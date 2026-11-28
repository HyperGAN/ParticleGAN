"""Unit and Law-4 invariant tests for L1 ``particlegan._axiom.optimizer``."""

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

from particlegan._axiom import optimizer as opt_mod
from particlegan._axiom.optimizer import AxiomOptimizer, collect_conv_layouts
from particlegan._axiom.state import (
    AxiomStepRecord,
    _attach_axiom_generator_hook,
    bind_axiom_record,
    get_active_prior_rows,
)

OPT_PATH = Path(opt_mod.__file__).resolve()


def test_static_l1_imports_forbidden_ops_and_hygiene() -> None:
    source = OPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                assert node.module in ("math_ops", "state"), f"Disallowed relative import: {node.module}"
            else:
                assert node.module is None or not node.module.startswith("particlegan")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("particlegan")

    forbidden_attrs = {"svd", "svdvals", "eigh", "eigvalsh", "pinv", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs, f"Forbidden op {node.attr!r}"
            is_ag = node.attr == "grad" and isinstance(node.value, ast.Attribute) and node.value.attr == "autograd"
            assert not is_ag, "Forbidden torch.autograd.grad"

    for forbidden_token in (
        "_EYE_CACHE",
        "_bn_net_ids",
        "_BatchNorm",
        "BatchDistance",
        "Emulated",
    ):
        assert forbidden_token not in source, f"Legacy smell {forbidden_token!r} must not appear"

    for lineno, line in enumerate(source.splitlines(), start=1):
        assert len(line) <= 120, f"Line {lineno} exceeds 120 chars"
        assert chr(59) not in line, f"Semicolon found on line {lineno}"


def test_law4_p1_rail_escape_and_scale_invariance() -> None:
    torch.manual_seed(0)

    # Regime 1: Rail escape from 1e-18 and 1e-12 gradients.
    for scale in (1e-18, 1e-12):
        w = nn.Parameter(torch.zeros(4, 4))
        b = nn.Parameter(torch.zeros(4))
        opt = AxiomOptimizer([w, b], lr=0.02)
        for _ in range(10):
            w.grad = torch.full_like(w, scale)
            b.grad = torch.full_like(b, scale)
            opt.step()
        assert float(w.norm().item()) > 0.18
        assert float(b.norm().item()) > 0.15

    # Regime 2: Exact degree-0 trajectory invariance across 36 decades (c in [1e-18, 1e18]).
    def _run_traj(c: float, beta1: float) -> list[torch.Tensor]:
        torch.manual_seed(123)
        p1d = nn.Parameter(torch.randn(6))
        p2d = nn.Parameter(torch.randn(8, 8))
        p4d = nn.Parameter(torch.randn(4, 3, 3, 3))
        ptab = nn.Parameter(torch.randn(16, 4))
        opt = AxiomOptimizer(
            [
                {
                    "params": [p1d, p2d, p4d],
                    "lr": 0.015,
                    "role": "generator",
                    "betas": (beta1, 0.999),
                },
                {
                    "params": [ptab],
                    "lr": 0.02,
                    "role": "prior",
                    "betas": (beta1, 0.999),
                },
            ]
        )
        for step_i in range(8):
            g_gen = torch.Generator().manual_seed(1000 + step_i)
            sign = 1.0 if step_i % 3 != 2 else -1.0
            for p in (p1d, p2d, p4d, ptab):
                p.grad = torch.randn(p.shape, generator=g_gen) * (c * sign)
            opt.set_sampled_rows(ptab, torch.tensor([0, 3, 5, 3]))
            opt.step()
        return [p.detach().clone() for p in (p1d, p2d, p4d, ptab)]

    for beta1 in (0.0, 0.5):
        ref = _run_traj(1.0, beta1)
        for c in (1e-18, 1e-12, 1e12, 1e18):
            for a, b in zip(ref, _run_traj(c, beta1)):
                assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def test_law4_p1_spike_chatter_and_decay() -> None:
    torch.manual_seed(0)

    # Regime 3: <= 2-step recovery from 1e4x gradient spike.
    x = nn.Parameter(torch.zeros(16))
    opt_sp = AxiomOptimizer([x], lr=0.05)
    for _ in range(20):
        x.grad = torch.ones_like(x)
        opt_sp.step()
    pre = x.detach().clone()
    x.grad = torch.ones_like(x)
    opt_sp.step()
    base_step = float((x - pre).norm().item())
    x.grad = torch.ones_like(x) * 1e4
    opt_sp.step()
    for _ in range(2):
        prev = x.detach().clone()
        x.grad = torch.ones_like(x)
        opt_sp.step()
    post_step = float((x - prev).norm().item())
    assert abs(post_step - base_step) / base_step < 0.05

    # Regime 4: Equilibrium settling under sign-flipping chatter + post-shift recovery.
    xc = nn.Parameter(torch.zeros(8))
    opt_ch = AxiomOptimizer([xc], lr=0.1)
    xc.grad = torch.ones_like(xc)
    p0 = xc.detach().clone()
    opt_ch.step()
    init_step = float((xc - p0).norm().item())
    for t in range(1, 105):
        prev = xc.detach().clone()
        xc.grad = torch.full_like(xc, 1.0 if t % 2 == 0 else -1.0)
        opt_ch.step()
    settled = float((xc - prev).norm().item())
    assert opt_ch.ema_step_cos < -0.5 and math.isclose(opt_ch.log2_damp, -10.0, abs_tol=1e-6)
    assert settled < 0.01 * init_step
    for _ in range(60):
        prev = xc.detach().clone()
        xc.grad = torch.ones_like(xc)
        opt_ch.step()
    recovered = float((xc - prev).norm().item())
    assert opt_ch.log2_damp > -0.1 and recovered > 0.9 * init_step

    # Regime 5: Non-freezing under t^{-1.5} decaying coherent gradients (coherence stays undamped & finite).
    xd = nn.Parameter(torch.zeros(8))
    opt_dc = AxiomOptimizer([xd], lr=0.05)
    for t in range(1, 81):
        prev = xd.detach().clone()
        xd.grad = torch.full_like(xd, t**-1.5)
        opt_dc.step()
    assert opt_dc.log2_damp == 0.0 and float((xd - prev).norm().item()) > 1e-4


def test_bn_weight_norm_invariance_and_zero_byte_conv2d_state() -> None:
    gen = torch.Generator().manual_seed(17)
    w_base = torch.randn(16, 16, generator=gen)
    g_seq = [torch.randn(16, 16, generator=gen) * (0.25 if t >= 2 else 1.0) for t in range(5)]

    # Exact BN weight-norm invariance on step size under W -> alpha W, g -> g / alpha
    # (log2_ref shifts by -log2(alpha) to keep mantissa-exponent moments exact).
    for alpha in (0.01, 10.0, 100.0):
        w_a = nn.Parameter(w_base.clone())
        w_b = nn.Parameter(w_base.clone() * alpha)
        opt_a = AxiomOptimizer([w_a], lr=0.014)
        opt_b = AxiomOptimizer([w_b], lr=0.014)
        for g in g_seq:
            with torch.no_grad():
                w_b.copy_(w_a * alpha)
            w_a.grad = g.clone()
            w_b.grad = g.clone() / alpha
            pre_a, pre_b = w_a.detach().clone(), w_b.detach().clone()
            opt_a.step()
            opt_b.step()
            da = (w_a.detach() - pre_a).norm().item()
            db = (w_b.detach() - pre_b).norm().item()
            assert math.isclose(da, db, rel_tol=5e-4, abs_tol=1e-5)
            assert math.isclose(
                opt_b.state[w_b]["log2_ref"] - opt_a.state[w_a]["log2_ref"],
                -math.log2(alpha),
                abs_tol=1e-4,
            )

    # Architecture-agnostic FP32 m_unit + v_unit (8 B/param, zero m_u) on Conv2d + BatchNorm2d stack.
    conv_net = nn.Sequential(
        nn.Conv2d(3, 16, 4, 2, 1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2),
        nn.Conv2d(16, 32, 4, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 1, 4, 1, 0, bias=False),
    )
    opt_c = AxiomOptimizer(
        conv_net.parameters(),
        lr=0.02,
        critic=conv_net,
        conv_layouts=collect_conv_layouts((conv_net,)),
    )
    for _ in range(5):
        for p in conv_net.parameters():
            p.grad = torch.randn(p.shape, generator=gen)
        opt_c.step()
    expected_bytes = sum(2 * p.numel() * 4 for p in conv_net.parameters())
    tensor_bytes = sum(
        v.numel() * v.element_size() for st in opt_c.state.values() for v in st.values() if isinstance(v, torch.Tensor)
    )
    assert tensor_bytes == expected_bytes, f"Expected {expected_bytes} B FP32 ME-Adam state, got {tensor_bytes} B"
    assert all("m_u" not in st for st in opt_c.state.values())
    assert opt_c.ns_2d_activations == 0 and opt_c.fast_path_steps == 5


def test_conv1x1_linear_parity_and_spectral_bounded_critic_vs_deep16_pathology() -> None:
    gen = torch.Generator().manual_seed(33)
    w0 = torch.randn(32, 32, generator=gen) * 0.1
    g0 = torch.randn(32, 32, generator=gen) * 0.05

    # Exact 4D ungrouped 1x1 Conv2d == 2D Linear parity on both G and D.
    for is_critic in (False, True):
        m_lin = nn.Linear(32, 32, bias=False)
        m_c1 = nn.Conv2d(32, 32, 1, bias=False)
        with torch.no_grad():
            m_lin.weight.copy_(w0)
            m_c1.weight.copy_(w0.view(32, 32, 1, 1))
        opt_l = AxiomOptimizer(m_lin.parameters(), lr=0.0002, critic=(m_lin if is_critic else None))
        opt_c = AxiomOptimizer(
            m_c1.parameters(),
            lr=0.0002,
            critic=(m_c1 if is_critic else None),
            conv_layouts=collect_conv_layouts((m_c1,)),
        )
        m_lin.weight.grad = g0.clone()
        m_c1.weight.grad = g0.view(32, 32, 1, 1).clone()
        opt_l.step()
        opt_c.step()
        assert torch.allclose(m_lin.weight, m_c1.weight.view(32, 32), rtol=1e-5, atol=1e-6)

    # Coordinate-wise ME-Adam takes bounded per-entry steps (<= lr) across arbitrary scale & rank on G and D.
    lr = 0.01
    for scale in (1e-12, 1e-6, 1.0, 1e12):
        c_mod = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            c_mod.weight.zero_()
        opt_d = AxiomOptimizer(c_mod.parameters(), lr=lr, critic=c_mod)
        c_mod.weight.grad = (torch.randn(32, 1, generator=gen) @ torch.randn(1, 32, generator=gen)) * scale
        opt_d.step()
        assert opt_d.ns_2d_activations == 0 and opt_d.fast_path_steps == 1
        assert float(c_mod.weight.detach().abs().max().item()) <= lr * 1.01

    # Deep 16L G and D share identical role-agnostic ME-Adam updates without extra NS overhead.
    g16 = nn.Sequential(*[nn.Linear(16, 16, bias=False) for _ in range(16)])
    d16 = deepcopy(g16)
    opt_g16 = AxiomOptimizer(g16.parameters(), lr=0.001)
    opt_d16 = AxiomOptimizer(d16.parameters(), lr=0.001, critic=d16)
    r1 = torch.randn(16, 1, generator=gen) @ torch.randn(1, 16, generator=gen)
    for idx, (pg, pd) in enumerate(zip(g16.parameters(), d16.parameters())):
        sc = 2.0 ** (idx * 0.5)
        pg.grad, pd.grad = r1 * sc, r1 * sc
    opt_g16.step()
    opt_d16.step()
    assert opt_g16.ns_2d_activations == 0 and opt_d16.ns_2d_activations == 0
    assert opt_g16.fast_path_steps == 1 and opt_d16.fast_path_steps == 1
    for pg, pd in zip(g16.parameters(), d16.parameters()):
        assert torch.allclose(pg, pd, rtol=1e-5, atol=1e-6)


def test_direction_blend_1loss_2loss_and_backward_with_protected() -> None:
    # 1) Exact 1-loss protection is NEVER silently skipped.
    w1 = nn.Parameter(torch.tensor([1.0, 2.0, -1.0]))
    opt1 = AxiomOptimizer([w1], lr=0.1, direction_blend=True)
    l_prot = (w1[0] - 0.0).square()  # grad = [2, 0, 0]
    l_aux = (w1[0] - 10.0).square() + (w1[1] - 0.0).square()  # pushes w1[0] upward, violating l_prot!
    opt1.bind_protected_losses([l_prot])
    g_prot_ref = torch.tensor([2.0, 0.0, 0.0])
    w1_before = w1.detach().clone()
    (l_prot + l_aux).backward()
    opt1.step()
    disp1 = w1.detach() - w1_before
    assert float((g_prot_ref @ disp1).item()) <= 1e-6
    assert opt1.constraint_geometry_stats["projected_steps"] == 1
    assert opt1.constraint_geometry_stats["max_derivative_after"] <= 1e-6

    # 2) Single-backward backward_with_protected binds ONLY aux-touched params (zero G clones on prior-only aux).
    torch.manual_seed(5)
    g_a = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))
    g_b = deepcopy(g_a)
    g_c = deepcopy(g_a)
    z_a = nn.Parameter(torch.randn(10, 4))
    z_b = nn.Parameter(z_a.detach().clone())
    z_c = nn.Parameter(z_a.detach().clone())
    opt_a = AxiomOptimizer(
        [{"params": list(g_a.parameters()), "role": "generator"}, {"params": [z_a], "role": "prior"}],
        lr=0.05,
        direction_blend=True,
    )
    opt_b = AxiomOptimizer(
        [{"params": list(g_b.parameters()), "role": "generator"}, {"params": [z_b], "role": "prior"}],
        lr=0.05,
        direction_blend=True,
    )
    opt_c = AxiomOptimizer(
        [{"params": list(g_c.parameters()), "role": "generator"}, {"params": [z_c], "role": "prior"}],
        lr=0.05,
        direction_blend=True,
    )
    rows = torch.tensor([0, 2, 4, 7])
    for opt_i, z_i in ((opt_a, z_a), (opt_b, z_b), (opt_c, z_c)):
        opt_i.set_sampled_rows(z_i, rows)
    # Verify weak registry used (zero monkey-patched attrs on z_a).
    assert not hasattr(z_a, "_axiom_active_rows")
    assert get_active_prior_rows(z_a) is not None

    # 2a) Prior-only aux_loss: G is never bound/cloned/projected, z satisfies exact per-subset non-ascent.
    z_b_before = z_b.detach().clone()
    loss_gan_b = (g_b(z_b[rows]) - 1.0).square().mean()
    aux_b = 5.0 * (z_b[rows] + 3.0).square().mean()
    opt_b.backward_with_protected(loss_gan_b, aux_b)
    assert opt_b._protected is not None and len(opt_b._protected) == 1
    assert all(normal is None for normal in opt_b._protected[0][:-1])
    z_gan_normal = opt_b._protected[0][-1]
    assert z_gan_normal is not None
    z_gan_normal = z_gan_normal.detach().clone()
    opt_b.step()
    assert float((z_gan_normal * (z_b.detach() - z_b_before)).sum().item()) <= 1e-6
    assert opt_b.constraint_geometry_stats["max_derivative_after"] <= 1e-6

    # 2b) aux_loss=None skips all param clones/projection and matches G of prior-only aux_loss bit-for-bit!
    loss_gan_c = (g_c(z_c[rows]) - 1.0).square().mean()
    opt_c.backward_with_protected(loss_gan_c, None)
    assert opt_c._protected is not None and all(normal is None for normal in opt_c._protected[0])
    opt_c.step()
    assert opt_c.constraint_geometry_stats["steps"] == 1
    for pb, pc in zip(g_b.parameters(), g_c.parameters()):
        assert torch.equal(pb, pc)

    # 2c) Full-model aux_loss (touching both G and z) matches bind_protected_losses([loss_gan])!
    g_d = deepcopy(g_a)
    z_d = nn.Parameter(z_a.detach().clone())
    opt_d = AxiomOptimizer(
        [{"params": list(g_d.parameters()), "role": "generator"}, {"params": [z_d], "role": "prior"}],
        lr=0.05,
        direction_blend=True,
    )
    opt_d.set_sampled_rows(z_d, rows)
    loss_gan_a = (g_a(z_a[rows]) - 1.0).square().mean()
    aux_all_a = 5.0 * (g_a(z_a[rows]) + 3.0).square().mean()
    opt_a.bind_protected_losses([loss_gan_a])
    (loss_gan_a + aux_all_a).backward()
    opt_a.step()

    loss_gan_d = (g_d(z_d[rows]) - 1.0).square().mean()
    aux_all_d = 5.0 * (g_d(z_d[rows]) + 3.0).square().mean()
    opt_d.backward_with_protected(loss_gan_d, aux_all_d)
    opt_d.step()
    for pa, pd in zip(g_a.parameters(), g_d.parameters()):
        assert torch.allclose(pa, pd, rtol=1e-5, atol=1e-5)
    assert torch.allclose(z_a, z_d, rtol=1e-5, atol=1e-5)
    assert opt_d.constraint_geometry_stats["max_derivative_after"] <= 1e-6

    # 3) Exact 2-loss direction_blend non-ascent with 1.5x Pythagorean cap.
    w2 = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    rec = AxiomStepRecord()
    rec.cond_initialized = True
    opt2 = AxiomOptimizer([w2], lr=0.1, direction_blend=True)
    bind_axiom_record(rec, opt2)
    la = (w2[0] - 0.0).square() + 0.1 * (w2[2] - 0.0).square()
    lb = (w2[0] - 3.0).square() + (w2[1] - 0.0).square()
    opt2.bind_protected_losses([la, lb])
    (la + 2.0 * lb).backward()
    opt2.step()
    assert opt2.constraint_geometry_stats["steps"] == 1
    assert opt2.constraint_geometry_stats["max_derivative_after"] <= 1e-6
    assert opt2.direction_blend_stats["conflict_steps"] == 1

    # 4) Packed multi-param _blend_dots matches streaming dot products & fails closed with exact rollback on NaN.
    p_a = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    p_b = nn.Parameter(torch.tensor([[0.5, -1.0], [2.0, 0.25]]))
    orig_a, orig_b = p_a.detach().clone(), p_b.detach().clone()
    opt_nan = AxiomOptimizer([p_a, p_b], lr=0.05, direction_blend=True)
    bad_loss = (p_a * float("nan")).sum() + p_b.square().sum()
    opt_nan.bind_protected_losses([bad_loss])
    p_a.grad = torch.ones_like(p_a)
    p_b.grad = torch.ones_like(p_b)
    with pytest.raises(ValueError):
        opt_nan.step()
    assert torch.equal(p_a, orig_a) and torch.equal(p_b, orig_b)


def test_verified_trust_and_zero_global_rng_consumption() -> None:
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

    gen = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    rec = AxiomStepRecord()
    rec.ema_data_scale = math.sqrt(2.0)
    rec.ema_push_res = 0.0
    _attach_axiom_generator_hook(gen, rec)
    opt_g = AxiomOptimizer(gen.parameters(), lr=2.0)
    bind_axiom_record(rec, opt_g)

    z = torch.randn(16, 4)
    y_pre = gen(z)
    y_pre_det = y_pre.detach().clone()
    y_pre.square().mean().backward()

    cpu_before = torch.get_rng_state().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    opt_g.step()
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert random.getstate() == py_before
    assert np.array_equal(np_before[1], np.random.get_state()[1])

    with torch.no_grad():
        d_obs = float((gen(z).double() - y_pre_det.double()).square().mean().sqrt().item())
    assert opt_g.last_probe_consumed and opt_g.trust_backtracks_total > 0
    assert d_obs <= 0.045 * 1.005


def test_bf16_fp32_state_checkpoint_and_schema2_fail_closed() -> None:
    torch.manual_seed(9)
    # Pure BF16 parameters with FP32 momentum/prior state tensors must keep FP32 across load_state_dict!
    w_bf16 = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    tab_bf16 = nn.Parameter(torch.randn(12, 4, dtype=torch.bfloat16))
    rec = AxiomStepRecord()
    opt = AxiomOptimizer(
        [
            {
                "params": [w_bf16],
                "lr": 0.02,
                "role": "generator",
                "betas": (0.5, 0.999),
            },
            {"params": [tab_bf16], "lr": 0.02, "role": "prior", "betas": (0.5, 0.999)},
        ],
        direction_blend=True,
        record=rec,
    )
    opt.set_sampled_rows(tab_bf16, torch.tensor([0, 2, 5]))
    loss = w_bf16.float().square().sum() + tab_bf16[torch.tensor([0, 2, 5])].float().square().sum()
    opt.bind_protected_losses([loss])
    loss.backward()
    opt.step()

    # Bind a second pending loss and checkpoint BEFORE step() to verify pending direction_blend roundtrip!
    opt.zero_grad(set_to_none=True)
    opt.set_sampled_rows(tab_bf16, torch.tensor([1, 3]))
    loss2 = w_bf16.float().square().sum() + tab_bf16[torch.tensor([1, 3])].float().square().sum()
    opt.bind_protected_losses([loss2])
    loss2.backward()
    sd = opt.state_dict()
    assert sd["axiom"]["schema"] == 2
    assert sd["axiom"]["constraint_geometry"]["pending"] is not None

    # Reload into a fresh BF16 optimizer and verify FP32 buffers + exact continuation.
    w_b = nn.Parameter(w_bf16.detach().clone())
    tab_b = nn.Parameter(tab_bf16.detach().clone())
    opt_b = AxiomOptimizer(
        [
            {"params": [w_b], "lr": 0.02, "role": "generator", "betas": (0.5, 0.999)},
            {"params": [tab_b], "lr": 0.02, "role": "prior", "betas": (0.5, 0.999)},
        ],
        direction_blend=True,
        record=AxiomStepRecord(),
    )
    opt_b.load_state_dict(deepcopy(sd))
    assert opt_b.state[w_b]["m_unit"].dtype == torch.float32
    assert opt_b.state[w_b]["v_unit"].dtype == torch.float32
    assert opt_b.state[tab_b]["m_prior"].dtype == torch.float32
    w_b.grad = w_bf16.grad.detach().clone()
    tab_b.grad = tab_bf16.grad.detach().clone()
    opt.step()
    opt_b.step()
    assert torch.equal(w_bf16, w_b) and torch.equal(tab_bf16, tab_b)

    # Fail-closed rejection of schema-1, legacy m_u, out-of-range log2_damp, wrong sketch len, non-finite state.
    for tamper in (
        lambda s: s["axiom"].__setitem__("schema", 1),
        lambda s: s["axiom"].__setitem__("log2_damp", -11.0),
        lambda s: s["axiom"].__setitem__("prev_step_sketch", [0.0] * 128),
        lambda s: s["state"][0].__setitem__("m_u", torch.zeros(8, 8)),
        lambda s: s["state"][0].__setitem__("m_unit", torch.full((8, 8), float("nan"))),
        lambda s: s.pop("regularizer"),
    ):
        bad = deepcopy(sd)
        tamper(bad)
        with pytest.raises(ValueError):
            opt_b.validate_state_dict(bad)


def test_fast_path_batched_momentum_parity_and_period2_cancellation() -> None:
    torch.manual_seed(21)
    conv_fast = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8))
    conv_slow = deepcopy(conv_fast).double()
    opt_fast = AxiomOptimizer(
        conv_fast.parameters(),
        lr=0.02,
        betas=(0.5, 0.999),
        conv_layouts=collect_conv_layouts((conv_fast,)),
    )
    opt_slow = AxiomOptimizer(
        conv_slow.parameters(),
        lr=0.02,
        betas=(0.5, 0.999),
        conv_layouts=collect_conv_layouts((conv_slow,)),
    )
    for _ in range(4):
        grads = [torch.randn_like(p) for p in conv_fast.parameters()]
        for pf, ps, g in zip(conv_fast.parameters(), conv_slow.parameters(), grads):
            pf.grad = g.clone()
            ps.grad = g.double()
        opt_fast.step()
        opt_slow.step()
    assert opt_fast.fast_path_steps == 4 and opt_slow.fast_path_steps == 0
    for pf, ps in zip(conv_fast.parameters(), conv_slow.parameters()):
        assert torch.allclose(pf.float(), ps.float(), rtol=1e-4, atol=1e-4)
        assert opt_fast.state[pf]["m_unit"].dtype == torch.float32
        assert opt_fast.state[pf]["v_unit"].dtype == torch.float32

    # Period-2 transverse oscillation + small consistent drift: beta1=0.5 cancels the
    # oscillating component and steers purely along the consistent drift direction.
    w = nn.Parameter(torch.ones(4))
    opt_p2 = AxiomOptimizer([w], lr=0.1, betas=(0.5, 0.999))
    osc = torch.tensor([1.0, -1.0, 0.0, 0.0])
    drift = torch.tensor([0.0, 0.0, 0.1, 0.1])
    w.grad = (osc + drift).clone()
    opt_p2.step()
    w.grad = (-osc + drift).clone()
    opt_p2.step()
    m_after = opt_p2.state[w]["m_unit"]
    assert float(m_after[:2].abs().max().item()) < 1e-6
    assert float(m_after[2:].min().item()) > 0.0

    # Exact dyadic-norm sign reversal cancels to zero cleanly.
    w0 = nn.Parameter(torch.ones(4))
    opt_z = AxiomOptimizer([w0], lr=0.1, betas=(0.5, 0.999))
    g0 = torch.tensor([1.0, -1.0, 1.0, -1.0])  # exact L2 norm == 2.0
    w0.grad = g0.clone()
    opt_z.step()
    w0_after_1 = w0.detach().clone()
    w0.grad = -g0.clone()
    opt_z.step()
    assert torch.allclose(w0, w0_after_1, atol=1e-6)
    assert float(opt_z.state[w0]["m_unit"].norm().item()) == 0.0


@pytest.mark.cuda
def test_cuda_conv2d_fast_path_and_zero_byte_state() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 16, 4, 2, 1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2),
        nn.Conv2d(16, 32, 4, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 1, 4, 1, 0, bias=False),
    ).to(dev)
    opt = AxiomOptimizer(net.parameters(), lr=0.02, critic=net, conv_layouts=collect_conv_layouts((net,)))
    for _ in range(4):
        for p in net.parameters():
            p.grad = torch.randn(p.shape, device=dev, generator=gen)
        opt.step()
    assert opt.ns_2d_activations == 0 and opt.fast_path_steps == 4
    expected_bytes = sum(2 * p.numel() * 4 for p in net.parameters())
    tensor_bytes = sum(
        v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if isinstance(v, torch.Tensor)
    )
    assert tensor_bytes == expected_bytes

    # Verify packed host-sync budget on 4-layer MLP (opt_d + opt_g combined <= 4 syncs/step).
    mlp_d = nn.Sequential(
        nn.Linear(8, 32),
        nn.SiLU(),
        nn.Linear(32, 32),
        nn.SiLU(),
        nn.Linear(32, 32),
        nn.SiLU(),
        nn.Linear(32, 1),
    ).to(dev)
    mlp_g = nn.Sequential(
        nn.Linear(4, 32),
        nn.SiLU(),
        nn.Linear(32, 32),
        nn.SiLU(),
        nn.Linear(32, 8),
    ).to(dev)
    opt_d = AxiomOptimizer(mlp_d.parameters(), lr=0.015, critic=mlp_d)
    opt_g = AxiomOptimizer(mlp_g.parameters(), lr=0.014)
    syncs = [0]
    orig_item, orig_tolist = torch.Tensor.item, torch.Tensor.tolist

    def _w_item(self: torch.Tensor, *a: object, **kw: object) -> object:
        if self.device.type == "cuda":
            syncs[0] += 1
        return orig_item(self, *a, **kw)

    def _w_tolist(self: torch.Tensor, *a: object, **kw: object) -> object:
        if self.device.type == "cuda":
            syncs[0] += 1
        return orig_tolist(self, *a, **kw)

    torch.Tensor.item = _w_item  # type: ignore[method-assign]
    torch.Tensor.tolist = _w_tolist  # type: ignore[method-assign]
    try:
        for p in mlp_d.parameters():
            p.grad = torch.randn(p.shape, device=dev, generator=gen)
        for p in mlp_g.parameters():
            p.grad = torch.randn(p.shape, device=dev, generator=gen)
        opt_d.step()
        opt_g.step()
    finally:
        torch.Tensor.item = orig_item  # type: ignore[method-assign]
        torch.Tensor.tolist = orig_tolist  # type: ignore[method-assign]
    assert syncs[0] <= 4, f"Too many GPU host syncs for D+G MLP step: {syncs[0]}"

    # Verify deterministic CUDA mode works with backward_with_protected + duplicate sampled prior rows.
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        z_det = nn.Parameter(torch.randn(16, 4, device=dev, generator=gen))
        opt_det = AxiomOptimizer(
            [{"params": list(mlp_g.parameters()), "role": "generator"}, {"params": [z_det], "role": "prior"}],
            lr=0.014,
            direction_blend=True,
        )
        dup_rows = torch.tensor([0, 2, 2, 5, 0, 9], device=dev)
        opt_det.zero_grad()
        opt_det.set_sampled_rows(z_det, dup_rows)
        lg = mlp_g(z_det[dup_rows]).square().mean()
        aux = (z_det[dup_rows] + 2.0).square().mean()
        opt_det.backward_with_protected(lg, aux)
        opt_det.step()
        assert opt_det.constraint_geometry_stats["max_derivative_after"] <= 1e-5
    finally:
        torch.use_deterministic_algorithms(prev_det)
