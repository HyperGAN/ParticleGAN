"""Unit and mathematical-invariant tests for L0 ``particlegan._axiom.math_ops``."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from particlegan._axiom import math_ops
from particlegan._axiom.math_ops import (
    PHI1,
    PHI2,
    allocate_deficit_quotas,
    born_infeld_cost,
    fp32_tikhonov_ns_2d,
    fp32_tikhonov_ns_batched,
    hamilton_quotas,
    log_sinkhorn,
    mantissa_exponent_normalize,
    me_unit_rows,
    power_principal_axis,
    safe_cdist,
    schatten6_decay_batched,
    sinkhorn_with_slack,
    smoothstep01,
    solve_nonascent_multipliers,
    solve_spd_cholesky,
    stable_rank_power,
    sym_eigensplit_halley,
    sym_inv_sqrt_ns,
    weyl_rademacher,
    zero_pinv_project_nonascent,
)

MATH_OPS_PATH = Path(math_ops.__file__).resolve()


def _random_orthogonal(dim: int, seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=dtype))
    return q


def test_static_l0_imports_and_forbidden_ops() -> None:
    source = MATH_OPS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Zero internal _axiom imports and zero legacy sibling imports.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"Relative import forbidden: {ast.dump(node)}"
            assert node.module is None or not node.module.startswith("particlegan")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("particlegan")

    # Zero forbidden calls: autograd.grad, svd/svdvals, eigh/eigvalsh, pinv, torch.compile.
    forbidden_attrs = {"svd", "svdvals", "eigh", "eigvalsh", "pinv", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs, f"Forbidden op {node.attr!r} found in math_ops.py"
            is_autograd_grad = (
                node.attr == "grad" and isinstance(node.value, ast.Attribute) and node.value.attr == "autograd"
            )
            assert not is_autograd_grad, "Forbidden torch.autograd.grad found in math_ops.py"


def test_static_l0_no_global_caches_and_style() -> None:
    source = MATH_OPS_PATH.read_text(encoding="utf-8")

    # Zero module-global tensor caches or mutable dict/list caches.
    for name, value in vars(math_ops).items():
        if name.startswith("__"):
            continue
        assert not isinstance(value, torch.Tensor), f"Module-global tensor {name!r} forbidden"
        assert not isinstance(value, (dict, list, set)), f"Mutable module-global cache {name!r} forbidden"

    # Line length and zero semicolon-packed lines.
    for lineno, line in enumerate(source.splitlines(), start=1):
        assert len(line) <= 120, f"Line {lineno} exceeds 120 chars ({len(line)})"
        assert chr(59) not in line, f"Semicolon found on line {lineno}: {line!r}"


def test_constants_and_smoothstep01() -> None:
    assert math.isclose(PHI1 * PHI1 + PHI1, 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(PHI2**3 + PHI2**2, 1.0, rel_tol=0.0, abs_tol=1e-15)

    assert smoothstep01(-1.0) == 0.0
    assert smoothstep01(0.0) == 0.0
    assert math.isclose(smoothstep01(0.5), 0.5, abs_tol=1e-15)
    assert smoothstep01(1.0) == 1.0
    assert smoothstep01(2.5) == 1.0

    xs = torch.linspace(-0.5, 1.5, 41, dtype=torch.float64, requires_grad=True)
    ys = smoothstep01(xs)
    assert torch.all(ys >= 0.0) and torch.all(ys <= 1.0)
    assert torch.all(ys[1:] >= ys[:-1])
    for xv, yv in zip(xs.detach().tolist(), ys.detach().tolist()):
        assert math.isclose(yv, smoothstep01(xv), abs_tol=1e-14)
    ys.sum().backward()
    assert xs.grad is not None
    interior = (xs.detach() > 0.0) & (xs.detach() < 1.0)
    u = xs.detach()[interior]
    assert torch.allclose(xs.grad[interior], 6.0 * u * (1.0 - u), atol=1e-12)


def test_mantissa_exponent_normalize_and_me_unit_rows_extreme_scales() -> None:
    base = torch.tensor([3.0, -4.0, 0.0, 12.0], dtype=torch.float32)
    u_ref, l2_ref, ok_ref = mantissa_exponent_normalize(base)
    assert bool(ok_ref.item())
    assert math.isclose(float(u_ref.norm().item()), 1.0, rel_tol=1e-6)
    assert math.isclose(float(l2_ref.item()), math.log2(13.0), abs_tol=1e-5)

    for exp_10 in (-38, -30, -20, -10, 10, 20, 35):
        scale = 10.0**exp_10
        u_s, l2_s, ok_s = mantissa_exponent_normalize(base * scale)
        assert bool(ok_s.item()), f"Failed at scale 1e{exp_10}"
        assert torch.allclose(u_s, u_ref, rtol=1e-6, atol=1e-6)
        assert math.isclose(float(l2_s.item()) - float(l2_ref.item()), math.log2(scale), abs_tol=1e-4)

    # Subnormal range safety (1e-42 in float32).
    sub = torch.tensor([1e-42, -2e-42, 0.0], dtype=torch.float32)
    u_sub, l2_sub, ok_sub = mantissa_exponent_normalize(sub)
    assert bool(ok_sub.item()) and torch.isfinite(u_sub).all() and torch.isfinite(l2_sub)
    assert math.isclose(float(u_sub.norm().item()), 1.0, rel_tol=1e-4)

    # Zero, NaN, Inf, and empty handling.
    for bad in (
        torch.zeros(4),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, float("inf")]),
        torch.empty(0),
    ):
        u_b, l2_b, ok_b = mantissa_exponent_normalize(bad)
        assert not bool(ok_b.item())
        assert torch.equal(u_b, torch.zeros_like(bad))
        assert float(l2_b.item()) == 0.0

    # Batched me_unit_rows with mixed scales + zero + NaN.
    rows = torch.stack(
        (
            base * 1e-30,
            base * 1e30,
            torch.zeros_like(base),
            torch.tensor([1.0, float("nan"), 0.0, 0.0]),
        ),
        dim=0,
    )
    u_rows, l2_rows, ok_rows = me_unit_rows(rows)
    assert ok_rows.squeeze(1).tolist() == [True, True, False, False]
    assert torch.allclose(u_rows[0], u_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(u_rows[1], u_ref, rtol=1e-6, atol=1e-6)
    assert torch.equal(u_rows[2], torch.zeros_like(base))
    assert torch.equal(u_rows[3], torch.zeros_like(base))
    assert math.isclose(float(l2_rows[1].item() - l2_rows[0].item()), math.log2(1e60), abs_tol=1e-3)


def test_weyl_rademacher_properties_and_determinism() -> None:
    v1 = weyl_rademacher((8, 64), phase=0.123, salt=1)
    v2 = weyl_rademacher((8, 64), phase=0.123, salt=1)
    v3 = weyl_rademacher((8, 64), phase=0.123, salt=2)
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, v3)
    assert torch.all((v1 == 1.0) | (v1 == -1.0))

    # Sample second moment across golden-ratio stepped phases approximates identity.
    dim = 16
    n_probes = 512
    probes = torch.stack(
        [weyl_rademacher((dim,), phase=(t * PHI1) % 1.0, salt=0, dtype=torch.float64) for t in range(n_probes)],
        dim=0,
    )
    cov = (probes.T @ probes) / float(n_probes)
    assert torch.allclose(cov.diagonal(), torch.ones(dim, dtype=torch.float64), atol=1e-12)
    off_diag = cov - torch.diag(cov.diagonal())
    assert float(off_diag.abs().amax().item()) < 0.18


def test_power_principal_axis_and_ltc_split_preservation() -> None:
    for dim in (1, 2, 4, 8, 16):
        q = _random_orthogonal(dim, seed=100 + dim)
        eigs = torch.tensor([100.0] + [1.0 / (k + 1) for k in range(dim - 1)], dtype=torch.float64)
        cov = q @ torch.diag(eigs) @ q.T
        lam1, u1 = power_principal_axis(cov, iters=6, seed=0)
        assert math.isclose(float(lam1.item()), 100.0, rel_tol=1e-9)
        assert math.isclose(float(u1.norm().item()), 1.0, rel_tol=1e-12)
        assert torch.allclose(cov @ u1, lam1 * u1, rtol=1e-5, atol=1e-5)
        # Verify canonical sign (largest-magnitude entry positive).
        assert float(u1[u1.abs().argmax()].item()) > 0.0
        # Verify Law 1 LTC child covariance remains strictly PSD.
        child_cov = cov - 0.16 * lam1 * torch.outer(u1, u1)
        min_eig = float(torch.linalg.eigvalsh(child_cov).amin().item())
        assert min_eig > 0.0

    # Batched + extreme scales + zero slice.
    q4 = _random_orthogonal(4, seed=7)
    base = q4 @ torch.diag(torch.tensor([25.0, 2.0, 1.0, 0.5], dtype=torch.float64)) @ q4.T
    batch = torch.stack((base * 1e-24, base * 1e24, torch.zeros(4, 4, dtype=torch.float64)), dim=0)
    lams, us = power_principal_axis(batch, iters=6)
    assert math.isclose(float(lams[0].item()), 25e-24, rel_tol=1e-6)
    assert math.isclose(float(lams[1].item()), 25e24, rel_tol=1e-6)
    assert float(lams[2].item()) == 0.0
    assert torch.allclose(us[0], us[1], atol=1e-6)
    assert math.isclose(float(us[2].norm().item()), 1.0, rel_tol=1e-12)


def test_stable_rank_power_accuracy_and_scale_invariance() -> None:
    gen = torch.Generator().manual_seed(42)
    # Exact rank-1 -> stable_rank == 1.0 across square and rectangular shapes.
    for shape in ((16, 16), (12, 32), (32, 12)):
        u = torch.randn(shape[0], 1, generator=gen)
        v = torch.randn(1, shape[1], generator=gen)
        r1 = u @ v
        sr = stable_rank_power(r1, iters=2)
        assert math.isclose(float(sr.item()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
        sr_tiny = stable_rank_power(r1 * 1e-30, iters=2)
        sr_huge = stable_rank_power(r1 * 1e30, iters=2)
        assert math.isclose(float(sr_tiny.item()), 1.0, abs_tol=1e-5)
        assert math.isclose(float(sr_huge.item()), 1.0, abs_tol=1e-5)

    # Orthogonal matrix -> exact stable_rank == m.
    eye8 = torch.eye(8)
    assert math.isclose(float(stable_rank_power(eye8, iters=2).item()), 8.0, rel_tol=1e-5)

    # Batched 3D parity with 2D + warm-start v + zero/NaN safety.
    m0 = torch.randn(8, 12, generator=gen)
    m1 = torch.zeros(8, 12)
    m2 = torch.full((8, 12), float("nan"))
    sr_b = stable_rank_power(torch.stack((m0, m1, m2), dim=0), iters=2)
    sr_0, v_out = stable_rank_power(m0, iters=2, return_v=True)
    assert math.isclose(float(sr_b[0].item()), float(sr_0.item()), rel_tol=1e-5)
    assert float(sr_b[1].item()) == 8.0 and float(sr_b[2].item()) == 8.0
    sr_warm = stable_rank_power(m0, iters=2, init_v=v_out)
    sr_conv = stable_rank_power(m0, iters=16, init_v=v_out)
    true_sv = torch.linalg.svdvals(m0.double())
    true_sr = float((true_sv.square().sum() / true_sv[0].square()).item())
    assert float(sr_0.item()) >= float(sr_warm.item()) - 1e-5
    assert float(sr_warm.item()) >= true_sr - 1e-4
    assert abs(float(sr_conv.item()) - true_sr) < 0.05

    # Fast-path parity: pre_normalized=True (2D/3D) and frob_norm (2D mv) + me_unit_rows(need_log2=False).
    frob_m0 = float(m0.norm().item())
    sr_frob = stable_rank_power(m0, iters=2, frob_norm=frob_m0)
    sr_pre_2d = stable_rank_power(m0 / frob_m0, iters=2, pre_normalized=True)
    sr_pre_3d = stable_rank_power((m0 / frob_m0).unsqueeze(0), iters=2, pre_normalized=True).squeeze(0)
    assert math.isclose(float(sr_frob.item()), float(sr_0.item()), rel_tol=1e-5)
    assert math.isclose(float(sr_pre_2d.item()), float(sr_0.item()), rel_tol=1e-5)
    assert math.isclose(float(sr_pre_3d.item()), float(sr_0.item()), rel_tol=1e-5)
    u_full, _, v_full = me_unit_rows(m0)
    u_fast, l2_none, v_fast = me_unit_rows(m0, need_log2=False)
    assert l2_none is None and torch.equal(u_full, u_fast) and torch.equal(v_full, v_fast)


def test_sym_eigensplit_halley_all_dimensions() -> None:
    for dim in (1, 2, 3, 4, 8, 16):
        q = _random_orthogonal(dim, seed=200 + dim)
        raw_eigs = torch.linspace(-3.5, 4.5, dim, dtype=torch.float64)
        if dim >= 4:
            raw_eigs[dim // 2] = 0.0
        mat = q @ torch.diag(raw_eigs) @ q.T
        expected_pos = q @ torch.diag(raw_eigs.clamp_min(0.0)) @ q.T
        expected_neg = q @ torch.diag(raw_eigs.clamp_max(0.0)) @ q.T

        e_pos, e_neg = sym_eigensplit_halley(mat)
        assert e_pos.dtype == torch.float64 and e_neg.dtype == torch.float64
        assert torch.allclose(e_pos + e_neg, mat, atol=1e-11)
        assert torch.allclose(e_pos, e_pos.T, atol=1e-12)
        assert torch.allclose(e_neg, e_neg.T, atol=1e-12)
        assert torch.allclose(e_pos @ e_neg, torch.zeros_like(mat), atol=1e-10)
        assert torch.allclose(e_pos, expected_pos, atol=1e-8)
        assert torch.allclose(e_neg, expected_neg, atol=1e-8)

    # Batched [B, d, d] including zero slice and pure-negative slice.
    q4 = _random_orthogonal(4, seed=99)
    a_mix = q4 @ torch.diag(torch.tensor([-2.0, -0.5, 0.5, 3.0], dtype=torch.float64)) @ q4.T
    a_neg = -torch.eye(4, dtype=torch.float64) * 2.5
    a_zero = torch.zeros(4, 4, dtype=torch.float64)
    ep, en = sym_eigensplit_halley(torch.stack((a_mix, a_neg, a_zero), dim=0))
    assert torch.allclose(ep[0] + en[0], a_mix, atol=1e-11)
    assert torch.allclose(ep[1], torch.zeros(4, 4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(en[1], a_neg, atol=1e-12)
    assert torch.equal(ep[2], a_zero) and torch.equal(en[2], a_zero)


def test_sym_inv_sqrt_ns_all_dimensions_and_conditioning() -> None:
    for dim in (1, 2, 3, 4, 8, 12, 16):
        q = _random_orthogonal(dim, seed=300 + dim)
        eigs = (
            torch.logspace(-3.0, 1.0, dim, dtype=torch.float64) if dim > 1 else torch.tensor([7.5], dtype=torch.float64)
        )
        mat = q @ torch.diag(eigs) @ q.T
        w = sym_inv_sqrt_ns(mat)
        eye = torch.eye(dim, dtype=torch.float64)
        assert torch.allclose(w, w.T, atol=1e-12)
        assert torch.allclose(w @ mat @ w, eye, atol=1e-10)
        assert torch.allclose(w @ w @ mat, eye, atol=1e-10)

    # Extreme scales (1e-20 and 1e+20) and batched execution.
    q6 = _random_orthogonal(6, seed=314)
    base = q6 @ torch.diag(torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float64)) @ q6.T
    batch = torch.stack((base * 1e-20, base * 1e20), dim=0)
    w_b = sym_inv_sqrt_ns(batch)
    eye6 = torch.eye(6, dtype=torch.float64)
    assert torch.allclose(w_b[0] @ batch[0] @ w_b[0], eye6, atol=1e-10)
    assert torch.allclose(w_b[1] @ batch[1] @ w_b[1], eye6, atol=1e-10)

    # Ridge regularization on singular PSD matrix produces finite symmetric inverse square root.
    rank1 = torch.outer(
        torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
        torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
    )
    w_reg = sym_inv_sqrt_ns(rank1, eps=1e-3)
    assert torch.allclose(
        w_reg @ (rank1 + 1e-3 * torch.eye(3, dtype=torch.float64)) @ w_reg,
        torch.eye(3, dtype=torch.float64),
        atol=1e-10,
    )


def test_solve_spd_cholesky_exact_and_fallback() -> None:
    q = _random_orthogonal(6, seed=55)
    gram = q @ torch.diag(torch.tensor([0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)) @ q.T
    rhs_mat = torch.randn(6, 3, generator=torch.Generator().manual_seed(56), dtype=torch.float64)
    ridge_vec = torch.tensor([1e-3, 1e-3, 2e-2, 2e-2, 2e-2, 2e-2], dtype=torch.float64)

    sol, inv = solve_spd_cholesky(gram, rhs_mat, ridge=ridge_vec, return_inv=True)
    assert inv is not None
    full = gram + torch.diag(ridge_vec)
    assert torch.allclose(full @ sol, rhs_mat, atol=1e-11)
    assert torch.allclose(full @ inv, torch.eye(6, dtype=torch.float64), atol=1e-11)

    # 1D vector RHS without inverse.
    rhs_vec = rhs_mat[:, 0]
    sol_v, inv_none = solve_spd_cholesky(gram, rhs_vec, ridge=1e-2, return_inv=False)
    assert inv_none is None
    assert torch.allclose((gram + 1e-2 * torch.eye(6, dtype=torch.float64)) @ sol_v, rhs_vec, atol=1e-11)

    # Indefinite / non-SPD matrix triggers Halley-PSD fallback cleanly without raising.
    bad_gram = q @ torch.diag(torch.tensor([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)) @ q.T
    sol_fb, inv_fb = solve_spd_cholesky(bad_gram, rhs_vec, return_inv=True)
    assert inv_fb is not None and torch.isfinite(sol_fb).all() and torch.isfinite(inv_fb).all()


def test_fp32_tikhonov_ns_convergence_and_rank1_bounds() -> None:
    gen = torch.Generator().manual_seed(11)
    # Full-rank convergence toward polar factor O @ O^T ~= I_m.
    full = torch.randn(16, 24, generator=gen)
    orth = fp32_tikhonov_ns_2d(full, steps=5, rel_smooth=0.0)
    gram_err = float((orth @ orth.T - torch.eye(16)).abs().amax().item())
    assert gram_err < 0.25, f"Polar factor convergence too weak: max_err={gram_err}"
    sv_full = torch.linalg.svdvals(orth)
    assert float(sv_full[0].item()) <= 1.16 and float(sv_full[-1].item()) >= 0.75

    # Rank-1 matrices: bounded sigma_1 <= 1.15 and < 1e-4 orthogonal leakage.
    for shape in ((64, 64), (32, 64), (64, 32)):
        r1 = torch.randn(shape[0], 1, generator=gen) @ torch.randn(1, shape[1], generator=gen)
        for rs in (0.0, 0.05):
            out = fp32_tikhonov_ns_2d(r1, steps=5, rel_smooth=rs)
            sv = torch.linalg.svdvals(out)
            assert float(sv[0].item()) <= 1.15 + 1e-4
            assert float((sv[1] / sv[0].clamp_min(1e-12)).item()) < 1e-4

    # Monotone contraction under large rel_smooth.
    m = torch.randn(24, 24, generator=gen)
    assert float(fp32_tikhonov_ns_2d(m, rel_smooth=50.0).norm().item()) < 0.25 * float(
        fp32_tikhonov_ns_2d(m, rel_smooth=0.05).norm().item()
    )

    # Batched parity, pre_normalized parity, vector shape, and zero/NaN safety.
    m0 = torch.randn(12, 18, generator=gen)
    m1 = torch.zeros(12, 18)
    m2 = torch.full((12, 18), float("nan"))
    out_b = fp32_tikhonov_ns_batched(
        torch.stack((m0, m1, m2), dim=0),
        steps=5,
        rel_smooth=torch.tensor([0.05, 0.05, 0.05]),
    )
    out_0 = fp32_tikhonov_ns_2d(m0, steps=5, rel_smooth=0.05)
    assert torch.allclose(out_b[0], out_0, rtol=1e-5, atol=1e-6)
    assert torch.equal(out_b[1], torch.zeros_like(m1)) and torch.equal(out_b[2], torch.zeros_like(m2))

    m0_unit = m0 / m0.norm()
    out_pre = fp32_tikhonov_ns_batched(m0_unit.unsqueeze(0), steps=5, rel_smooth=0.05, pre_normalized=True).squeeze(0)
    assert torch.allclose(out_pre, out_0, rtol=1e-5, atol=1e-6)

    vec_out = fp32_tikhonov_ns_2d(torch.tensor([[3.0, 4.0]]), rel_smooth=0.0)
    assert torch.allclose(vec_out, torch.tensor([[0.6, 0.8]]), atol=1e-6)

    # Zero-H2D list/scalar rel_smooth parity against Tensor rel_smooth (2D + 1D shapes).
    pair = torch.stack((m0, torch.randn(12, 18, generator=gen)), dim=0)
    sm_list = [0.03, 0.09]
    out_t = fp32_tikhonov_ns_batched(pair, steps=5, rel_smooth=torch.tensor(sm_list))
    out_l = fp32_tikhonov_ns_batched(pair, steps=5, rel_smooth=sm_list)
    assert torch.allclose(out_l, out_t, rtol=1e-5, atol=1e-6)
    pair_1d = torch.randn(2, 1, 8, generator=gen)
    assert torch.allclose(
        fp32_tikhonov_ns_batched(pair_1d, rel_smooth=sm_list),
        fp32_tikhonov_ns_batched(pair_1d, rel_smooth=torch.tensor(sm_list)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_schatten6_decay_exact_deadband_and_contraction() -> None:
    q = _random_orthogonal(8, seed=71, dtype=torch.float32)
    # Inside deadband (sigma_1 = 1.8 < tau = 2.0): exact zero decay.
    w_safe = q @ torch.diag(torch.linspace(0.5, 1.8, 8)) @ q.T
    # Outside deadband (sigma_1 = 4.0 > tau = 2.0): strictly active contraction along top singular direction.
    w_high = q @ torch.diag(torch.tensor([0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 4.0])) @ q.T
    rates = torch.tensor([0.05, 0.05])
    decay = schatten6_decay_batched(torch.stack((w_safe, w_high), dim=0), rates, lam=0.25, tau=2.0)

    assert torch.equal(decay[0], torch.zeros_like(w_safe))
    assert float(decay[1].norm().item()) > 0.0
    s_before = float(torch.linalg.svdvals(w_high)[0].item())
    s_after = float(torch.linalg.svdvals(w_high + decay[1])[0].item())
    assert s_after < s_before

    # Scalar & Python-sequence rates parity against Tensor rates.
    stack_w = torch.stack((w_safe, w_high), dim=0)
    assert torch.allclose(schatten6_decay_batched(stack_w, 0.05, lam=0.25, tau=2.0), decay, rtol=1e-6, atol=1e-7)
    mixed_t = schatten6_decay_batched(stack_w, torch.tensor([0.02, 0.07]), lam=0.25, tau=2.0)
    mixed_l = schatten6_decay_batched(stack_w, [0.02, 0.07], lam=0.25, tau=2.0)
    assert torch.allclose(mixed_l, mixed_t, rtol=1e-6, atol=1e-7)


def test_zero_pinv_project_nonascent_kkt_dense_and_streaming() -> None:
    gen = torch.Generator().manual_seed(77)

    # Exact closed-form checks (1-loss, 2-loss acute, antiparallel, polar-cone zero).
    d = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)
    p1 = zero_pinv_project_nonascent(d, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(p1, torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64), atol=1e-12)

    n2 = torch.tensor([[1.0, 0.3, 0.0], [0.2, 1.0, 0.0]], dtype=torch.float64)
    p2 = zero_pinv_project_nonascent(d, n2)
    assert torch.allclose(p2, torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64), atol=1e-12)

    n_opp = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    p_opp = zero_pinv_project_nonascent(d, n_opp)
    assert torch.allclose(p_opp, torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64), atol=1e-12)

    # d strictly inside positive cone spanned by normals -> exact zero projection.
    n_cone = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    p_zero = zero_pinv_project_nonascent(torch.tensor([2.0, 3.0], dtype=torch.float64), n_cone)
    assert torch.equal(p_zero, torch.zeros(2, dtype=torch.float64))

    # Variational-inequality KKT check across 40 random 1- and 2-constraint instances.
    for trial in range(40):
        nc = 1 if trial < 15 else 2
        dv = torch.randn(10, generator=gen, dtype=torch.float64)
        nv = torch.randn(nc, 10, generator=gen, dtype=torch.float64)
        proj = zero_pinv_project_nonascent(dv, nv)
        # Primal feasibility: a_k . p <= 0.
        assert float((nv @ proj).max().item()) <= 1e-10
        # Cone-projection optimality: (p - d) . p == 0 and (p - d) . y >= 0 for feasible y.
        assert abs(float(((proj - dv) * proj).sum().item())) <= 1e-9
        # Verify streaming per-parameter parity (including None entries for untouched params).
        splits = (3, 4, 3)
        d_parts = list(dv.split(splits))
        n_parts = [list(nv[k].split(splits)) for k in range(nc)]
        p_stream = zero_pinv_project_nonascent(d_parts, n_parts)
        assert torch.allclose(torch.cat(p_stream), proj, atol=1e-11)

    # Streaming with None normal entries matches zero-padded dense normals.
    d_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, -1.0])]
    n_list = [[torch.tensor([1.0, 0.0]), None]]
    out_stream = zero_pinv_project_nonascent(d_list, n_list)
    assert torch.allclose(out_stream[0], torch.tensor([0.0, 2.0]), atol=1e-6)
    assert torch.allclose(out_stream[1], torch.tensor([3.0, -1.0]), atol=1e-6)

    # Direct scalar multiplier API rejects >2 constraints cleanly.
    with pytest.raises(ValueError):
        solve_nonascent_multipliers((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 5.0)


def test_safe_cdist_born_infeld_and_sinkhorn() -> None:
    gen = torch.Generator().manual_seed(88)
    x = torch.randn(8, 5, generator=gen)
    y = torch.randn(10, 5, generator=gen)

    # safe_cdist extreme scale invariance + diag_inf.
    d_ref = safe_cdist(x, y)
    assert torch.allclose(safe_cdist(x * 1e-24, y * 1e-24) * 1e24, d_ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(safe_cdist(x * 1e24, y * 1e24) * 1e-24, d_ref, rtol=1e-5, atol=1e-6)
    d_self = safe_cdist(x, diag_inf=True)
    assert torch.isinf(d_self.diagonal()).all()

    # Born-Infeld cancellation-free value & exact autograd gradient.
    xa = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    ya = torch.tensor([[1.0 + 1e-6, 2.0], [3.0, 4.0]], dtype=torch.float64)
    ell = 1.0
    cost, c_hat = born_infeld_cost(xa, ya, ell, aligned=True, fp64=True)
    expected = torch.tensor([0.5e-12, math.sqrt(1.0 + 25.0) - 1.0], dtype=torch.float64)
    assert torch.allclose(cost, expected, rtol=1e-9, atol=1e-18)
    assert torch.allclose(c_hat, expected.float(), rtol=1e-5, atol=1e-12)
    cost.sum().backward()
    assert xa.grad is not None
    diff = xa.detach() - ya
    expected_grad = diff / torch.sqrt(ell * ell + diff.square().sum(dim=-1, keepdim=True))
    assert torch.allclose(xa.grad, expected_grad, rtol=1e-9, atol=1e-12)

    # log_sinkhorn exact marginal convergence.
    c_mat = safe_cdist(x, y)
    log_a = torch.full((8,), -math.log(8))
    log_b = torch.full((10,), -math.log(10))
    pi = log_sinkhorn(-c_mat / 0.25, log_a, log_b, iters=40).exp()
    assert torch.allclose(pi.sum(dim=1), torch.full((8,), 1.0 / 8.0), atol=1e-5)
    assert torch.allclose(pi.sum(dim=0), torch.full((10,), 1.0 / 10.0), atol=1e-5)
    lk_masked = -c_mat / 0.25
    lk_masked[0, :] = float("-inf")
    pi_masked = log_sinkhorn(lk_masked, log_a, log_b, iters=10).exp()
    assert torch.isfinite(pi_masked).all() and float(pi_masked[0].abs().max().item()) == 0.0

    # sinkhorn_with_slack respects masks and produces valid marginals.
    mask = torch.ones((8, 10), dtype=torch.bool)
    mask[0, 0] = False
    pi_s = sinkhorn_with_slack(
        c_mat,
        torch.zeros(8),
        torch.zeros(10),
        rho=2.0,
        tau=0.0,
        eps=0.2,
        iters=30,
        mask=mask,
    )
    assert pi_s.shape == (9, 11)
    assert float(pi_s[0, 0].item()) == 0.0
    assert torch.allclose(pi_s[:8].sum(dim=1), torch.full((8,), 1.0 / 8.0), atol=1e-4)
    assert torch.allclose(pi_s[:, :10].sum(dim=0), torch.full((10,), 1.0 / 10.0), atol=1e-4)


def test_hamilton_and_deficit_quotas() -> None:
    # hamilton_quotas: exact sum, q_c >= 1 when total >= K, determinism.
    weights = torch.tensor([0.50, 0.25, 0.15, 0.08, 0.02], dtype=torch.float32)
    for total in (3, 5, 10, 64, 257):
        q1 = hamilton_quotas(total, weights)
        q2 = hamilton_quotas(total, weights)
        assert torch.equal(q1, q2)
        assert int(q1.sum().item()) == total
        if total >= len(weights):
            assert bool((q1 >= 1).all().item())

    # Equal weights + zero weights fallback.
    q_eq = hamilton_quotas(13, torch.ones(5))
    assert int(q_eq.sum().item()) == 13 and int(q_eq.min().item()) == 2 and int(q_eq.max().item()) == 3
    q_zero = hamilton_quotas(7, torch.zeros(4))
    assert int(q_zero.sum().item()) == 7 and bool((q_zero >= 1).all().item())

    # allocate_deficit_quotas: starved modes get priority + permutation equivariance.
    w_100 = torch.full((100,), 0.01)
    w_fake_starved = torch.cat((torch.full((64,), 1.0 / 64.0), torch.zeros(36)))
    q_starved = allocate_deficit_quotas(w_100, w_fake_starved, n_slots=64)
    assert int(q_starved.sum().item()) == 64
    assert int(q_starved[64:].sum().item()) == 36

    perm = torch.randperm(100, generator=torch.Generator().manual_seed(0))
    w_fake_dist = torch.linspace(0.002, 0.018, 100)
    w_fake_dist = w_fake_dist / w_fake_dist.sum()
    q_orig = allocate_deficit_quotas(w_100, w_fake_dist, n_slots=64)
    q_perm = allocate_deficit_quotas(w_100[perm], w_fake_dist[perm], n_slots=64)
    assert torch.equal(q_perm, q_orig[perm])

    q_large = allocate_deficit_quotas(weights, torch.full((5,), 0.2), n_slots=23)
    assert int(q_large.sum().item()) == 23


def test_zero_global_rng_consumption() -> None:
    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)

    cpu_before = torch.get_rng_state().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()

    x = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)
    y = x * 0.5 + 1.0
    sym = x.T @ x + torch.eye(8)

    _ = smoothstep01(x)
    _ = mantissa_exponent_normalize(x)
    _ = me_unit_rows(x)
    _ = weyl_rademacher((4, 8), phase=0.37, salt=5)
    _ = power_principal_axis(sym)
    _ = stable_rank_power(x)
    _ = sym_eigensplit_halley(sym)
    _ = sym_inv_sqrt_ns(sym)
    _ = solve_spd_cholesky(sym, torch.ones(8, 2), ridge=1e-2, return_inv=True)
    _ = fp32_tikhonov_ns_2d(x, rel_smooth=0.05)
    _ = schatten6_decay_batched(x.unsqueeze(0), torch.tensor([0.05]))
    _ = zero_pinv_project_nonascent(x[0], x[1:3])
    c_mat = safe_cdist(x, y)
    _ = born_infeld_cost(x, y, 1.0)
    _ = log_sinkhorn(-c_mat, torch.full((4,), -math.log(4)), torch.full((4,), -math.log(4)))
    _ = sinkhorn_with_slack(c_mat, torch.zeros(4), torch.zeros(4), rho=1.0, tau=0.1, eps=0.2)
    _ = hamilton_quotas(16, torch.ones(4))
    _ = allocate_deficit_quotas(torch.ones(4) * 0.25, torch.tensor([0.4, 0.3, 0.2, 0.1]), 6)

    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert random.getstate() == py_before
    assert np_before[0] == np.random.get_state()[0]
    assert np.array_equal(np_before[1], np.random.get_state()[1])
    assert np_before[2:] == np.random.get_state()[2:]


@pytest.mark.cuda
def test_cuda_parity_and_zero_sync_quotas() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    dev = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(9)
    q = _random_orthogonal(8, seed=9).to(dev)
    mat_eig = q @ torch.diag(torch.linspace(-2.0, 3.0, 8, device=dev, dtype=torch.float64)) @ q.T
    ep, en = sym_eigensplit_halley(mat_eig)
    assert torch.allclose(ep + en, mat_eig, atol=1e-11)

    mat_spd = q @ torch.diag(torch.logspace(-2.0, 1.0, 8, device=dev, dtype=torch.float64)) @ q.T
    w = sym_inv_sqrt_ns(mat_spd)
    assert torch.allclose(w @ mat_spd @ w, torch.eye(8, device=dev, dtype=torch.float64), atol=1e-10)

    # Verify GPU hamilton_quotas & allocate_deficit_quotas incur zero .item() host syncs.
    weights = torch.rand(64, generator=gen).to(dev)
    w_fake = torch.rand(64, generator=gen).to(dev)
    orig_item = torch.Tensor.item
    syncs = [0]

    def _count_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        if self.device.type == "cuda":
            syncs[0] += 1
        return orig_item(self, *args, **kwargs)

    torch.Tensor.item = _count_item  # type: ignore[method-assign]
    try:
        qh = hamilton_quotas(128, weights)
        qd = allocate_deficit_quotas(weights / weights.sum(), w_fake / w_fake.sum(), 32)
    finally:
        torch.Tensor.item = orig_item  # type: ignore[method-assign]

    assert syncs[0] == 0, f"Expected zero GPU host syncs in quota ops, observed {syncs[0]}"
    assert int(qh.sum().item()) == 128 and bool((qh >= 1).all().item())
    assert int(qd.sum().item()) == 32
