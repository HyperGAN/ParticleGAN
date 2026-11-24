"""Reproducible mathematical proofs, counterexamples, and empirical controls for the P0 Law 4.3 audit.

Covers:
1. Elementwise (Hadamard) scaling, Adam, Smooth SignSGD, Exact SignSGD, and ParticleGAN
   Adafactor breaking exact matrix rank on rank-1 / rank-2 inputs.
2. Invertible separable (D_L G D_R) and Kronecker (L G R) exact-rank preservation, plus the
   diagonal vs full-Kronecker (1-step Shampoo == Tikhonov polar) effective-rank dichotomy.
3. Exact-arithmetic subspace and rank invariance of polynomial Newton-Schulz and ParticleGAN's
   fused Z = [X, Y] Tikhonov NS across FP64/FP32/BF16/FP16, plus code-audit findings
   (step-varying _QUINTIC_COEFFS, smoothing=0 Frobenius rescaling norm blowup on rank-1).
4. Spectral rank metric ordering (1 <= stable_rank <= PR <= erank_RV <= exact_rank <= min(m, n))
   and mechanisms that CAN vs CANNOT introduce new singular directions (momentum, weight decay,
   minibatch accumulation, CountSketch).
5. Real-architecture empirical verification on `grid100_mlp16` vs shallow MLP vs 16-layer
   residual MLP vs 4D Conv2d DCGAN gradients.
"""
from __future__ import annotations

import math

import pytest
import torch

from particlegan._law4_energy.optimizer import _smooth_sign_unit, exact_polar_2xn
from particlegan._t4_arch.optimizer import (
    _QUINTIC_A,
    _QUINTIC_B,
    _QUINTIC_C,
    quintic_orthogonalize_2d as t4_quintic_2d,
)
from particlegan._unified_v4.optimizer import _QUINTIC_COEFFS
from particlegan._unified_v11.optimizer import (
    UnifiedV11EquilibratedOptimizer,
    quintic_orthogonalize_2d as v11_quintic_2d,
    quintic_orthogonalize_batched as v11_quintic_batched,
)
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer, _sketch_grad_67
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.convolution_quintic import COEFFICIENTS as CONV_QUINTIC_COEFFS, quintic_direction
from particlegan.optim.dualnorm import polar_factor

from scripts.four_laws_audit.p0_rank_experiments import (
    _build_res16_mlp,
    alignment_with_polar,
    ref_poly_ns,
    spectral_metrics,
    subspace_leakage,
)


# ---------------------------------------------------------------------------
# 1. Hadamard / Elementwise / Adam / SignSGD / Adafactor Rank Counterexamples
# ---------------------------------------------------------------------------

def test_hadamard_2x2_counterexample_increases_rank() -> None:
    a = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    prod = a * b
    assert int(torch.linalg.matrix_rank(a).item()) == 1
    assert int(torch.linalg.matrix_rank(prod).item()) == 2
    assert math.isclose(float(torch.linalg.det(prod).item()), 1.0, rel_tol=1e-12)


def test_adam_elementwise_scaling_breaks_exact_rank_on_rank1() -> None:
    # Analytical 2x2 closed-form determinant proof for 1-step Adam with eps > 0:
    # det([[1/(1+d), 2/(2+d)], [2/(2+d), 4/(4+d)]]) = -4*d / ((1+d)*(4+d)*(2+d)^2) != 0 for all d > 0
    u2 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    g2 = torch.outer(u2, u2)
    assert int(torch.linalg.matrix_rank(g2).item()) == 1
    for delta in (1e-8, 1e-2, 1.0):
        upd = g2 / (g2.abs() + delta)
        expected_det = (-4.0 * delta) / ((1.0 + delta) * (4.0 + delta) * ((2.0 + delta) ** 2))
        assert math.isclose(float(torch.linalg.det(upd).item()), expected_det, rel_tol=1e-6, abs_tol=1e-18)
        assert float(torch.linalg.det(upd).item()) < 0.0

    # 16x16 rank-1 matrix: step-1 Adam with eps=0 stays rank 1, eps>0 jumps to full rank
    m = 16
    g1 = torch.outer(torch.linspace(1.0, 4.0, m, dtype=torch.float64),
                     torch.linspace(1.0, 5.0, m, dtype=torch.float64))
    assert spectral_metrics(g1)["rank_tol_1e-12"] == 1
    assert spectral_metrics(g1 / g1.abs())["rank_tol_1e-12"] == 1

    # ParticleGAN peak_rms settling floor delta = 0.015 * rms + 1e-8 breaks exact rank cleanly
    delta_pg = 0.015 * float(g1.square().mean().sqrt().item()) + 1e-8
    upd_pg = g1 / (g1.abs() + delta_pg)
    assert spectral_metrics(upd_pg)["rank_tol_1e-12"] >= 4
    assert int(torch.linalg.matrix_rank(upd_pg, atol=0.0, rtol=1e-14).item()) >= 5

    # Multi-step Adam (beta1=0, beta2=0.999, EVEN WITH eps=0.0!) breaks exact rank on step 2
    rng = torch.Generator().manual_seed(0)
    g_step2 = torch.outer(torch.randn(m, generator=rng, dtype=torch.float64),
                          torch.randn(m, generator=rng, dtype=torch.float64))
    assert spectral_metrics(g_step2)["rank_tol_1e-12"] == 1
    b2 = 0.999
    v_hat = ((b2 * g1.square() + (1.0 - b2) * g_step2.square()) / (1.0 - b2 * b2)).sqrt()
    upd_step2 = g_step2 / v_hat  # strictly eps == 0.0!
    # At b2=0.999, (1-b2)/b2 = 1e-3 so Taylor terms decay as ~1e-3^k -> rank_tol_1e-12 >= 3 > 1
    assert spectral_metrics(upd_step2)["rank_tol_1e-12"] >= 3
    # At b2=0.5 (balanced EMA weights, still strictly eps == 0.0), rank jumps well above 1
    v_hat_half = ((0.5 * g1.square() + 0.5 * g_step2.square()) / 0.75).sqrt()
    upd_step2_half = g_step2 / v_hat_half
    assert spectral_metrics(upd_step2_half)["rank_tol_1e-12"] >= 6


def test_sign_sgd_and_smooth_sign_rank_behavior() -> None:
    m = 16
    g1 = torch.outer(torch.linspace(-3.0, 4.0, m, dtype=torch.float64),
                     torch.linspace(-2.0, 5.0, m, dtype=torch.float64))
    # Exact sign(u v^T) = sign(u) sign(v)^T has exact rank <= 1
    assert spectral_metrics(torch.sign(g1))["rank_tol_1e-12"] == 1

    # Exact sign on an exact rank-2 matrix G_ij = j - i + 0.5 has exact FULL rank m (det = 2^(m-1))
    idx = torch.arange(m, dtype=torch.float64)
    g2 = idx.unsqueeze(0) - idx.unsqueeze(1) + 0.5
    assert spectral_metrics(g2)["rank_tol_1e-12"] == 2
    s2 = torch.sign(g2)
    assert spectral_metrics(s2)["rank_tol_1e-12"] == m
    assert math.isclose(abs(float(torch.linalg.det(s2).item())), float(2 ** (m - 1)), rel_tol=1e-9)

    # ParticleGAN Smooth SignSGD (_smooth_sign_unit, s > 0) breaks rank even on rank-1 input!
    u_pos = torch.linspace(1.0, 4.0, 8, dtype=torch.float32)
    v_pos = torch.linspace(1.0, 5.0, 8, dtype=torch.float32)
    g1_f32 = torch.outer(u_pos, v_pos)
    sm_sign = _smooth_sign_unit(g1_f32, 0.5)
    assert spectral_metrics(sm_sign)["rank_tol_1e-6"] >= 3
    assert spectral_metrics(sm_sign)["rank_tol_1e-12"] >= 5


def test_adafactor_separable_vs_particlegan_additive_floor() -> None:
    m = 16
    g1 = torch.outer(torch.linspace(1.0, 4.0, m, dtype=torch.float64),
                     torch.linspace(1.0, 5.0, m, dtype=torch.float64))
    # Shazeer-Stern Adafactor: eps1 added BEFORE factoring -> outer product -> strictly separable D_L G D_R
    r_ss = (g1.square() + 1e-6).mean(dim=1).sqrt()
    c_ss = (g1.square() + 1e-6).mean(dim=0).sqrt()
    v_ss = torch.outer(r_ss, c_ss) / r_ss.square().mean().sqrt()
    assert spectral_metrics(g1 / v_ss)["rank_tol_1e-12"] == 1

    # ParticleGAN law4_adafactor_mp: adds delta = 0.015 * peak + 1e-8 AFTER outer product -> breaks rank-1!
    r_pg = g1.square().mean(dim=1).sqrt()
    c_pg = g1.square().mean(dim=0).sqrt()
    r_rms = r_pg.square().mean().sqrt()
    v_pg = torch.outer(r_pg, c_pg) / r_rms
    delta_pg = 0.015 * float(r_rms.item()) + 1e-8
    assert spectral_metrics(g1 / (v_pg + delta_pg))["rank_tol_1e-12"] >= 4


# ---------------------------------------------------------------------------
# 2. Separable & Kronecker Exact-Rank Preservation vs Effective-Rank Dichotomy
# ---------------------------------------------------------------------------

def test_separable_kronecker_and_frobenius_invariance() -> None:
    m = 16
    rng = torch.Generator().manual_seed(7)
    d_l = torch.diag(torch.linspace(0.5, 4.0, m, dtype=torch.float64))
    d_r = torch.diag(torch.linspace(0.3, 3.5, m, dtype=torch.float64))
    l_mat = torch.randn(m, m, generator=rng, dtype=torch.float64) + 5.0 * torch.eye(m, dtype=torch.float64)
    r_mat = torch.randn(m, m, generator=rng, dtype=torch.float64) + 5.0 * torch.eye(m, dtype=torch.float64)

    for rank in (1, 2, 4):
        u = torch.randn(m, rank, generator=rng, dtype=torch.float64)
        v = torch.randn(m, rank, generator=rng, dtype=torch.float64)
        g = u @ v.T
        assert spectral_metrics(g)["rank_tol_1e-12"] == rank
        assert spectral_metrics(d_l @ g @ d_r)["rank_tol_1e-12"] == rank
        assert spectral_metrics(l_mat @ g @ r_mat)["rank_tol_1e-12"] == rank

        # Frobenius normalization preserves every normalized spectral invariant identically
        m_raw = spectral_metrics(g)
        m_frob = spectral_metrics(g / g.norm())
        assert math.isclose(m_raw["stable_rank"], m_frob["stable_rank"], rel_tol=1e-12)
        assert math.isclose(m_raw["participation_ratio"], m_frob["participation_ratio"], rel_tol=1e-12)
        assert math.isclose(m_raw["rv_erank"], m_frob["rv_erank"], rel_tol=1e-12)


def test_diagonal_vs_kronecker_shampoo_effective_rank_dichotomy() -> None:
    m = 16
    sv = torch.tensor([1.0] + [1e-4] * (m - 1), dtype=torch.float64)
    g_axis = torch.diag(sv)
    q1, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(2), dtype=torch.float64))
    g_rot = q1 @ g_axis @ q2.T

    def adafactor_ss(mat: torch.Tensor) -> torch.Tensor:
        r_v = (mat.square() + 1e-16).mean(dim=1).sqrt()
        c_v = (mat.square() + 1e-16).mean(dim=0).sqrt()
        return mat / (torch.outer(r_v, c_v) / r_v.square().mean().sqrt())

    # Diagonal scaling lifts effective rank on axis-aligned matrices (Adafactor squares diagonal scales
    # -> 15 equalized directions out of 16, exact D_L = diag(1/sv) lifts to 16), fails on rotated matrices
    assert spectral_metrics(g_axis)["stable_rank"] < 1.01
    assert spectral_metrics(adafactor_ss(g_axis))["stable_rank"] > 14.9
    d_exact = torch.diag(1.0 / sv)
    assert spectral_metrics(d_exact @ g_axis)["stable_rank"] > 15.99
    assert spectral_metrics(adafactor_ss(g_rot))["stable_rank"] < 1.05

    # Full Kronecker 1-step Shampoo L G R == exact Tikhonov polar G (G^T G + s^2 I)^{-1/2}.
    # Forming the Gram matrix squares kappa=1e4 -> kappa^2=1e8, so FP64 eigh error is O(kappa^2 * u_fp64) ~ 1e-8.
    s = 1e-6
    el, ql = torch.linalg.eigh(g_rot @ g_rot.T + (s * s) * torch.eye(m, dtype=torch.float64))
    er, qr = torch.linalg.eigh(g_rot.T @ g_rot + (s * s) * torch.eye(m, dtype=torch.float64))
    shampoo = ((ql * el.pow(-0.25)) @ ql.T) @ g_rot @ ((qr * er.pow(-0.25)) @ qr.T)
    polar = polar_factor(g_rot, smoothing=s)
    assert float((shampoo - polar).norm().item()) < 1e-7
    assert spectral_metrics(shampoo)["stable_rank"] > 15.9

    # Confirm exact machine-precision agreement on well-conditioned rotated matrix (kappa=1e2)
    g_rot_mod = q1 @ torch.diag(torch.tensor([1.0] + [1e-2] * (m - 1), dtype=torch.float64)) @ q2.T
    el2, ql2 = torch.linalg.eigh(g_rot_mod @ g_rot_mod.T + (s * s) * torch.eye(m, dtype=torch.float64))
    er2, qr2 = torch.linalg.eigh(g_rot_mod.T @ g_rot_mod + (s * s) * torch.eye(m, dtype=torch.float64))
    shampoo_mod = ((ql2 * el2.pow(-0.25)) @ ql2.T) @ g_rot_mod @ ((qr2 * er2.pow(-0.25)) @ qr2.T)
    polar_mod = polar_factor(g_rot_mod, smoothing=s)
    assert float((shampoo_mod - polar_mod).norm().item()) < 1e-11


# ---------------------------------------------------------------------------
# 3. Newton-Schulz Subspace/Rank Invariance & ParticleGAN Implementation Audit
# ---------------------------------------------------------------------------

def test_polynomial_ns_and_fused_tikhonov_preserve_subspace_and_rank_fp64() -> None:
    m = 32
    q_u, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(10), dtype=torch.float64))
    q_v, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(11), dtype=torch.float64))
    for r_exact in (1, 2, 4):
        s = torch.tensor([1.0] * r_exact + [0.0] * (m - r_exact), dtype=torch.float64)
        g = (q_u * s) @ q_v.T
        for smooth in (0.0, 0.01, 0.2):
            y = ref_poly_ns(g, dtype=torch.float64, steps=5, smoothing=smooth)
            sm = spectral_metrics(y)
            lk = subspace_leakage(g, y, r_exact)
            assert sm["rank_tol_1e-12"] == r_exact
            assert lk["col_leak_rel"] < 1e-13
            assert lk["row_leak_rel"] < 1e-13


def test_particlegan_ns_coefficients_and_rank1_norm_blowup_audit() -> None:
    # Code audit finding 1: unified_v4/v11/v12 use 5 step-varying coefficients, NOT stationary (3.4445, -4.7750, 2.0315)
    assert len(_QUINTIC_COEFFS) == 5
    assert len(set(_QUINTIC_COEFFS)) == 5
    assert _QUINTIC_COEFFS[0] != (_QUINTIC_A, _QUINTIC_B, _QUINTIC_C)
    assert CONV_QUINTIC_COEFFS == (_QUINTIC_A, _QUINTIC_B, _QUINTIC_C)

    # Code audit finding 2: exact rank-1 preservation across all ParticleGAN polar/NS kernels (CPU FP32)
    q_u, _ = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(20), dtype=torch.float64))
    q_v, _ = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(21), dtype=torch.float64))
    g_r1 = torch.outer(q_u[:, 0], q_v[:, 0]).float()

    for out in (
        v11_quintic_2d(g_r1, steps=5, smoothing=0.0),
        v11_quintic_2d(g_r1, steps=5, smoothing=0.0061),
        v11_quintic_batched(g_r1.unsqueeze(0), steps=5, smoothing=0.0061)[0],
        t4_quintic_2d(g_r1, steps=5),
        quintic_direction(g_r1, smoothing=0.0061),
    ):
        sm = spectral_metrics(out)
        lk = subspace_leakage(g_r1, out, 1)
        assert sm["rank_tol_1e-3"] == 1
        assert sm["stable_rank"] < 1.001
        assert lk["col_leak_rel"] < 1e-4

    # Code audit finding 3: v11_quintic_2d(s=0.0) rescales rank-1 output to Frobenius norm sqrt(m) -> sigma_1 ~ sqrt(m)!
    q128_u, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(12), dtype=torch.float64))
    q128_v, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(13), dtype=torch.float64))
    g128_r1 = torch.outer(q128_u[:, 0], q128_v[:, 0]).float()
    out_s0 = v11_quintic_2d(g128_r1, steps=5, smoothing=0.0)
    out_tikh = v11_quintic_2d(g128_r1, steps=5, smoothing=0.0061)
    assert math.isclose(spectral_metrics(out_s0)["sigma_1"], math.sqrt(128.0), rel_tol=1e-3)
    assert spectral_metrics(out_tikh)["sigma_1"] <= 1.2

    # Exact 2xN Cayley-Hamilton polar preserves rank-1 in exact arithmetic; FP32 cast of inv00/01/11
    # introduces ~1.8e-6 cancellation when s>0 while keeping stable_rank == 1.000000 and rank_tol_1e-3 == 1
    g2x16 = torch.outer(torch.tensor([1.0, 2.0]), torch.linspace(1.0, 4.0, 16))
    assert spectral_metrics(exact_polar_2xn(g2x16, smoothing=0.0))["rank_tol_1e-12"] == 1
    ep_s01 = exact_polar_2xn(g2x16, smoothing=0.1)
    assert spectral_metrics(ep_s01)["rank_tol_1e-3"] == 1
    assert spectral_metrics(ep_s01)["stable_rank"] < 1.0001
    assert subspace_leakage(g2x16, ep_s01, 1)["col_leak_rel"] < 1e-5


def test_ns_five_step_spectral_gain_limit_and_tikhonov_cutoff() -> None:
    # 5-step zero-slope gain is prod(a_k) ~= 492.75: lifts 1e-2 strongly, fails to lift 1e-10
    prod_a = math.prod(a for a, _, _ in _QUINTIC_COEFFS[:5])
    assert 490.0 < prod_a < 496.0

    m = 32
    q_u, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(30), dtype=torch.float64))
    q_v, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(31), dtype=torch.float64))

    g_1e2 = (q_u * torch.tensor([1.0] + [1e-2] * (m - 1), dtype=torch.float64)) @ q_v.T
    g_1e10 = (q_u * torch.tensor([1.0] + [1e-10] * (m - 1), dtype=torch.float64)) @ q_v.T

    # On 1e-2 ratio: 5-step NS (s=0) lifts stable rank dramatically from ~1.003 to >25
    assert spectral_metrics(g_1e2)["stable_rank"] < 1.01
    assert spectral_metrics(ref_poly_ns(g_1e2, dtype=torch.float64, steps=5, smoothing=0.0))["stable_rank"] > 20.0

    # On 1e-10 ratio: 5-step NS CANNOT lift stable rank above ~1.0001 (bounded by ~493x gain)
    assert spectral_metrics(ref_poly_ns(g_1e10, dtype=torch.float64, steps=5, smoothing=0.0))["stable_rank"] < 1.001
    # Whereas exact SVD polar (with tol < 1e-10) DOES lift it to full stable rank m
    assert spectral_metrics(polar_factor(g_1e10, smoothing=0.0))["stable_rank"] > m - 0.1

    # Tikhonov smoothing s=0.1 suppresses 1e-2 modes instead of lifting them to 1
    assert spectral_metrics(ref_poly_ns(g_1e2, dtype=torch.float64, steps=5, smoothing=0.1))["stable_rank"] < 2.0


def test_ns_finite_precision_leakage_bf16_exceeds_fp64() -> None:
    m = 64
    q_u, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(40), dtype=torch.float64))
    q_v, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(41), dtype=torch.float64))
    g_r1 = torch.outer(q_u[:, 0], q_v[:, 0])

    lk_64 = subspace_leakage(g_r1, ref_poly_ns(g_r1, dtype=torch.float64, steps=5, smoothing=0.0), 1)
    lk_32 = subspace_leakage(g_r1, ref_poly_ns(g_r1, dtype=torch.float32, steps=5, smoothing=0.0), 1)
    lk_bf16 = subspace_leakage(g_r1, ref_poly_ns(g_r1, dtype=torch.bfloat16, steps=5, smoothing=0.0), 1)
    lk_fp16 = subspace_leakage(g_r1, ref_poly_ns(g_r1, dtype=torch.float16, steps=5, smoothing=0.0), 1)

    assert lk_64["col_leak_rel"] < 1e-13
    assert lk_32["col_leak_rel"] > lk_64["col_leak_rel"]
    assert lk_bf16["col_leak_rel"] > 1e-4  # BF16 rounding amplified across 5 quintic steps!
    assert lk_fp16["col_leak_rel"] > 1e-5


# ---------------------------------------------------------------------------
# 4. Metric Hierarchy & Rank-Generating Operations Audit
# ---------------------------------------------------------------------------

def test_spectral_metric_inequality_chain_and_stable_vs_rv_gap() -> None:
    spectra = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.5, 0.25],
        [1.0] + [1e-2] * 100,
        [1.0, 1e-1, 1e-2, 1e-3, 1e-4],
    ]
    for sv in spectra:
        mat = torch.diag(torch.tensor(sv, dtype=torch.float64))
        sm = spectral_metrics(mat)
        assert 1.0 - 1e-12 <= sm["stable_rank"] <= sm["participation_ratio"] + 1e-12
        assert sm["participation_ratio"] <= sm["rv_erank"] + 1e-12
        assert sm["rv_erank"] <= sm["rank_tol_1e-12"] + 1e-12

    # Stark separation: calling stable rank "erank" misreports rv_erank ~= 20 as 1.01
    mat_gap = torch.diag(torch.tensor([1.0] + [1e-2] * 100, dtype=torch.float64))
    sm_gap = spectral_metrics(mat_gap)
    assert sm_gap["stable_rank"] < 1.02
    assert sm_gap["rv_erank"] > 15.0


def test_momentum_weight_decay_countsketch_and_optimizer_contracts() -> None:
    rng = torch.Generator().manual_seed(99)
    m = 16
    stream = [torch.outer(torch.randn(m, generator=rng, dtype=torch.float64),
                          torch.randn(m, generator=rng, dtype=torch.float64)) for _ in range(5)]

    # Momentum across 5 rank-1 steps increases exact rank to 5
    mom = torch.zeros(m, m, dtype=torch.float64)
    for g_t in stream:
        mom = 0.75 * mom + 0.25 * g_t
    assert spectral_metrics(mom)["rank_tol_1e-12"] == 5

    # Coupled/decoupled weight decay adds full-rank W_t -> full rank
    w_orth, _ = torch.linalg.qr(torch.randn(m, m, generator=rng, dtype=torch.float64))
    assert spectral_metrics(stream[0] + 1e-2 * w_orth)["rank_tol_1e-12"] == m

    # CountSketch is a scalar sign-flip detector (cos == -1.0 on negated gradient)
    sk_pos = _sketch_grad_67(stream[0].float(), roll=0)
    sk_neg = _sketch_grad_67((-stream[0]).float(), roll=0)
    assert math.isclose(float((sk_pos * sk_neg).sum().item()), -1.0, abs_tol=1e-5)

    # UnifiedV11 and UnifiedV12 enforce weight_decay == 0.0
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="weight_decay"):
        UnifiedV11EquilibratedOptimizer([p], weight_decay=1e-2)
    with pytest.raises(ValueError, match="weight_decay"):
        UnifiedV12Optimizer([p], weight_decay=1e-2)


# ---------------------------------------------------------------------------
# 5. Real-Architecture Empirical Control (MLP16 vs Shallow vs ResMLP16 vs Conv2d)
# ---------------------------------------------------------------------------

def test_real_architectures_exact_vs_effective_rank_and_subdominant_alignment() -> None:
    from experiments.forge.scaling_benchmark import DCGANCritic, DCGANGenerator
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians

    dev = torch.device("cpu")
    disc = SimpleMLPDiscriminator(2, 128, 3, fourier=2)
    deterministic_orthogonal_(disc, seed=0)

    def get_hidden_grads(gen_mod: torch.nn.Module) -> list[torch.Tensor]:
        deterministic_orthogonal_(gen_mod, seed=0)
        rng = torch.Generator(device=dev).manual_seed(0)
        z = torch.randn(256, 2, generator=rng, device=dev)
        real = sample_100gaussians(256, dev, generator=rng)
        gen_mod.zero_grad(set_to_none=True)
        fake = gen_mod(z)
        loss = -disc(fake).mean() + 0.5 * torch.cdist(fake, real).min(dim=1).values.square().mean()
        loss.backward()
        return [p.grad.detach().clone() for p in gen_mod.parameters() if p.ndim == 2 and min(p.shape) >= 16 and p.grad is not None]

    grads_mlp16 = get_hidden_grads(SimpleMLPGenerator(2, 128, n_hidden=16, out_dim=2))
    grads_mlp3 = get_hidden_grads(SimpleMLPGenerator(2, 128, n_hidden=3, out_dim=2))
    grads_res16 = get_hidden_grads(_build_res16_mlp(2, 128, n_blocks=7, out_dim=2))

    # 1. Deep skip-free MLP16 gradients are NOT exact rank 1 (they have high numerical rank),
    #    YET they exhibit severe effective-rank (stable-rank) collapse relative to shallow & residual MLPs!
    mlp16_metrics = [spectral_metrics(g) for g in grads_mlp16]
    mlp3_metrics = [spectral_metrics(g) for g in grads_mlp3]
    res16_metrics = [spectral_metrics(g) for g in grads_res16]

    med_rank12_mlp16 = sorted(m["rank_tol_1e-12"] for m in mlp16_metrics)[len(mlp16_metrics) // 2]
    med_srank_mlp16 = sorted(m["stable_rank"] for m in mlp16_metrics)[len(mlp16_metrics) // 2]
    med_srank_mlp3 = sorted(m["stable_rank"] for m in mlp3_metrics)[len(mlp3_metrics) // 2]
    med_srank_res16 = sorted(m["stable_rank"] for m in res16_metrics)[len(res16_metrics) // 2]

    assert med_rank12_mlp16 >= 32  # Strictly NOT exact rank 1!
    assert med_srank_mlp16 < 1.35  # Severe effective-rank collapse
    assert med_srank_mlp3 > med_srank_mlp16
    assert med_srank_res16 > med_srank_mlp16

    # 2. On MLP16 early-mid layer (net.6, where s2/s1 ~= 0.10 > 0): FP32 NS (s=0) lifts stable rank
    #    and subdominant singular alignment far above Frobenius-MP, Adam, and Smooth SignSGD
    g_early = grads_mlp16[2]
    g_early_64 = g_early.double()
    u_ns = v11_quintic_2d(g_early, steps=5, smoothing=0.0)  # CPU runs in FP32!
    u_frob = g_early_64 / g_early_64.norm()
    u_adam = g_early_64 / (g_early_64.abs() + 1e-8)
    u_sign = _smooth_sign_unit(g_early, 0.03 * float(g_early.square().mean().sqrt().item()))

    assert spectral_metrics(u_ns)["stable_rank"] > 5.0 * spectral_metrics(u_frob)["stable_rank"]
    assert alignment_with_polar(g_early_64, u_ns, top_k_tail=16)["cos_polar_subdominant_2_to_k"] > (
        alignment_with_polar(g_early_64, u_adam, top_k_tail=16)["cos_polar_subdominant_2_to_k"] + 0.25
    )
    assert alignment_with_polar(g_early_64, u_ns, top_k_tail=16)["cos_polar_subdominant_2_to_k"] > (
        alignment_with_polar(g_early_64, u_sign, top_k_tail=16)["cos_polar_subdominant_2_to_k"] + 0.25
    )
    assert alignment_with_polar(g_early_64, u_ns, top_k_tail=16)["off_singular_basis_energy_frac"] < 0.01
    assert alignment_with_polar(g_early_64, u_adam, top_k_tail=16)["off_singular_basis_energy_frac"] > 0.30

    # 3. 4D Conv2d / ConvTranspose2d DCGAN per-offset slices have substantially higher stable rank than MLP16
    dcgan_g = DCGANGenerator(64, 128, 64, 512)
    dcgan_d = DCGANCritic(64, 64, 512)
    deterministic_orthogonal_(dcgan_g, seed=0)
    deterministic_orthogonal_(dcgan_d, seed=0)
    rng_c = torch.Generator(device=dev).manual_seed(0)
    z_c = torch.randn(8, 128, generator=rng_c, device=dev)
    real_img = torch.randn(8, 3, 64, 64, generator=rng_c, device=dev).tanh()
    fake_img = dcgan_g(z_c)
    loss_gc = -dcgan_d(fake_img).mean() + 0.5 * (fake_img - real_img).square().mean()
    loss_gc.backward()
    conv_sranks = []
    for p in dcgan_g.parameters():
        if p.ndim == 4 and p.grad is not None and min(p.shape[0], p.shape[1]) >= 16:
            c0, c1, kh, kw = p.shape
            slices = p.grad.detach().permute(2, 3, 0, 1).reshape(kh * kw, c0, c1)
            conv_sranks.append(spectral_metrics(slices[0])["stable_rank"])
    assert conv_sranks and min(conv_sranks) > med_srank_mlp16
