"""P0 Adversarial Mathematical & Empirical Audit of Law 4.3 (Matrix Rank, Scaling & Newton-Schulz).

Reproduces every formal counterexample, invariance check, dtype precision test,
and real-architecture gradient-spectrum measurement reported in:
  reports/four-laws-audit/p0-law43-rank-audit.md

Run:
  PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
      scripts/four_laws_audit/p0_rank_experiments.py
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import particlegan
from particlegan._law4_energy.optimizer import _smooth_sign_unit, exact_polar_2xn
from particlegan._t4_arch.optimizer import (
    _QUINTIC_A,
    _QUINTIC_B,
    _QUINTIC_C,
    quintic_orthogonalize_2d as t4_quintic_2d,
)
from particlegan._unified_v4.optimizer import _QUINTIC_COEFFS
from particlegan._unified_v11.optimizer import (
    quintic_orthogonalize_2d as v11_quintic_2d,
    quintic_orthogonalize_batched as v11_quintic_batched,
)
from particlegan._unified_v12.optimizer import _sketch_grad_67
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.convolution_quintic import quintic_direction
from particlegan.optim.dualnorm import polar_factor


# ---------------------------------------------------------------------------
# Spectral & Rank Metrics (rigorous distinctions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def spectral_metrics(mat: torch.Tensor) -> Dict[str, Any]:
    """Compute exact-style FP64 SVD rank and spectral diversity metrics for a 2D matrix."""
    x = mat.detach().to(device="cpu", dtype=torch.float64)
    r, c = int(x.shape[0]), int(x.shape[1])
    m = min(r, c)
    sv = torch.linalg.svdvals(x)
    s1 = float(sv[0].item()) if m > 0 else 0.0
    sr = float(sv[-1].item()) if m > 0 else 0.0
    frob_sq = float(sv.square().sum().item())
    frob = math.sqrt(frob_sq)

    if s1 <= 0.0 or frob_sq <= 0.0:
        return {
            "shape": [r, c],
            "frob": 0.0,
            "sigma_1": 0.0,
            "sigma_2": 0.0,
            "sigma_r": 0.0,
            "rank_tol_1e-12": 0,
            "rank_tol_1e-6": 0,
            "rank_tol_1e-3": 0,
            "rank_default": 0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "rv_erank": 0.0,
            "top1_energy": 0.0,
            "cond_full": float("inf"),
        }

    # Numerical ranks at explicit relative tolerances
    rank_12 = int((sv > 1e-12 * s1).sum().item())
    rank_6 = int((sv > 1e-6 * s1).sum().item())
    rank_3 = int((sv > 1e-3 * s1).sum().item())
    rank_def = int(torch.linalg.matrix_rank(x).item())

    # Stable rank: ||X||_F^2 / ||X||_2^2  (what law4-optimizer-laws.md calls "erank")
    stable_rank = frob_sq / (s1 * s1)

    # Participation ratio / Renyi-2 effective rank: (sum s_i^2)^2 / sum s_i^4
    s4_sum = float(sv.pow(4).sum().item())
    pr = (frob_sq * frob_sq) / max(s4_sum, 1e-300)

    # Roy-Vetterli (2007) entropy effective rank: exp(-sum p_i ln p_i), p_i = s_i / sum s_j
    s_sum = float(sv.sum().item())
    p = sv[sv > 0.0] / s_sum
    entropy = float(-(p * p.log()).sum().item())
    rv_erank = math.exp(entropy)

    top1_energy = (s1 * s1) / frob_sq
    cond_full = (s1 / sr) if sr > 1e-300 else float("inf")
    s2 = float(sv[1].item()) if m >= 2 else 0.0

    return {
        "shape": [r, c],
        "frob": frob,
        "sigma_1": s1,
        "sigma_2": s2,
        "sigma_r": sr,
        "rank_tol_1e-12": rank_12,
        "rank_tol_1e-6": rank_6,
        "rank_tol_1e-3": rank_3,
        "rank_default": rank_def,
        "stable_rank": stable_rank,
        "participation_ratio": pr,
        "rv_erank": rv_erank,
        "top1_energy": top1_energy,
        "cond_full": cond_full,
    }


@torch.no_grad()
def subspace_leakage(orig: torch.Tensor, transformed: torch.Tensor, exact_rank: int) -> Dict[str, float]:
    """Measure energy of `transformed` orthogonal to the true rank-r column/row subspaces of `orig`."""
    g = orig.detach().to(device="cpu", dtype=torch.float64)
    y = transformed.detach().to(device="cpu", dtype=torch.float64)
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    u_r = u[:, :exact_rank]
    v_r = vh[:exact_rank, :]
    y_norm = float(y.norm().item())
    if y_norm <= 0.0:
        return {"col_leak_rel": 0.0, "row_leak_rel": 0.0, "proj_residual_rel": 0.0}
    y_col_proj = u_r @ (u_r.T @ y)
    y_row_proj = (y @ v_r.T) @ v_r
    y_both_proj = u_r @ ((u_r.T @ y) @ v_r.T) @ v_r
    return {
        "col_leak_rel": float((y - y_col_proj).norm().item() / y_norm),
        "row_leak_rel": float((y - y_row_proj).norm().item() / y_norm),
        "proj_residual_rel": float((y - y_both_proj).norm().item() / y_norm),
    }


@torch.no_grad()
def alignment_with_polar(grad: torch.Tensor, update: torch.Tensor, top_k_tail: int = 16) -> Dict[str, float]:
    """Compare an optimizer update against exact polar U V^T on total, top-1, and subdominant modes."""
    g = grad.detach().to(device="cpu", dtype=torch.float64)
    w = update.detach().to(device="cpu", dtype=torch.float64)
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    s1 = float(s[0].item())
    keep = s > (1e-12 * s1)
    polar = (u * keep.to(torch.float64)) @ vh

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        an = float(a.norm().item())
        bn = float(b.norm().item())
        if an <= 1e-30 or bn <= 1e-30:
            return 0.0
        return float((a * b).sum().item() / (an * bn))

    # Project update into SVD coordinates: C = U^T W V
    c_mat = u.T @ w @ vh.T
    diag_c = torch.diagonal(c_mat)
    w_frob = float(w.norm().item())
    diag_norm = float(diag_c.norm().item())
    off_diag_frac = float((c_mat - torch.diag(diag_c)).norm().item() / max(w_frob, 1e-30))

    # Subdominant subspace: remove top-1 mode (u_1 v_1^T)
    u1v1 = torch.outer(u[:, 0], vh[0, :])
    w_sub = w - float((w * u1v1).sum().item()) * u1v1
    polar_sub = polar - u1v1
    k_eff = min(top_k_tail, int(s.numel()))
    polar_sub_k = (u[:, 1:k_eff] @ vh[1:k_eff, :]) if k_eff > 1 else torch.zeros_like(polar)

    # Energy fraction of update in true subdominant singular directions (diagonal k >= 2)
    tail_diag_energy = float(diag_c[1:].square().sum().item() / max(w_frob * w_frob, 1e-30))
    top1_proj_energy = float((diag_c[0] * diag_c[0]).item() / max(w_frob * w_frob, 1e-30))

    return {
        "cos_grad": _cos(w, g),
        "cos_polar_full": _cos(w, polar),
        "cos_polar_subdominant_all": _cos(w_sub, polar_sub),
        "cos_polar_subdominant_2_to_k": _cos(w_sub, polar_sub_k),
        "top1_mode_energy_frac": top1_proj_energy,
        "subdominant_diag_energy_frac": tail_diag_energy,
        "off_singular_basis_energy_frac": off_diag_frac * off_diag_frac,
        "diag_alignment_ratio": diag_norm / max(w_frob, 1e-30),
    }


# ---------------------------------------------------------------------------
# Reference Polynomial NS in arbitrary dtype (FP64 / FP32 / BF16 / FP16)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ref_poly_ns(
    mat: torch.Tensor,
    *,
    dtype: torch.dtype,
    steps: int = 5,
    smoothing: float = 0.0,
    coeffs: Tuple[Tuple[float, float, float], ...] = _QUINTIC_COEFFS,
    rescale_to_target_frob: bool = False,
) -> torch.Tensor:
    """Reference fused Z=[X, Y] or standard polynomial NS in exact requested `dtype`."""
    r, c = int(mat.shape[0]), int(mat.shape[1])
    trans = r > c
    x0 = (mat.T if trans else mat).to(dtype=torch.float64)
    m, n = int(x0.shape[0]), int(x0.shape[1])
    s_val = float(smoothing)

    if s_val > 0.0:
        scale = max(float(x0.abs().amax().item()), s_val, 1e-38)
        xs = x0 / scale
        ss = s_val / scale
        aug_frob = math.sqrt(float(xs.square().sum().item()) + m * ss * ss)
        inv_aug = 1.0 / max(aug_frob * 1.02, 1e-38)
        x_part = (xs * inv_aug).to(dtype=dtype)
        y_part = (torch.eye(m, device=mat.device, dtype=torch.float64) * (ss * inv_aug)).to(dtype=dtype)
        z = torch.cat((x_part, y_part), dim=1)
        for a, b, cc in coeffs[:steps]:
            gram = z @ z.T
            gram2 = gram @ gram
            z = a * z + (b * gram + cc * gram2) @ z
        z_out = z.to(dtype=torch.float64)
        x_out = z_out[:, :n]
        if rescale_to_target_frob:
            jf = max(float(z_out.norm().item()), 1e-12)
            x_out = x_out * min(1.15, math.sqrt(float(m)) / jf)
    else:
        scale = max(float(x0.abs().amax().item()), 1e-38)
        xs = x0 / scale
        frob = float(xs.norm().item())
        x = (xs / max(frob * 1.02, 1e-38)).to(dtype=dtype)
        for a, b, cc in coeffs[:steps]:
            gram = x @ x.T
            gram2 = gram @ gram
            x = a * x + (b * gram + cc * gram2) @ x
        x_out = x.to(dtype=torch.float64)
        if rescale_to_target_frob:
            of = max(float(x_out.norm().item()), 1e-12)
            x_out = x_out * (math.sqrt(float(m)) / of)

    return x_out.T if trans else x_out


# ---------------------------------------------------------------------------
# Experiment 1: Formal Counterexamples & Preconditioner Rank Taxonomy
# ---------------------------------------------------------------------------

def run_experiment_1() -> Dict[str, Any]:
    """Prove & verify rank behavior of Hadamard, Separable, Kronecker, Adam, SignSGD, Adafactor, Shampoo."""
    out: Dict[str, Any] = {}

    # 1a. Minimal 2x2 Hadamard counterexample
    a = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    had = a * b
    out["hadamard_2x2"] = {
        "rank_A": int(torch.linalg.matrix_rank(a).item()),
        "rank_A_hadamard_B": int(torch.linalg.matrix_rank(had).item()),
        "det_A_hadamard_B": float(torch.linalg.det(had).item()),
    }

    # 1b. Exact rank-1 matrix G = u v^T (16x16, distinct positive coordinates)
    m = 16
    u = torch.linspace(1.0, 4.0, m, dtype=torch.float64)
    v = torch.linspace(1.0, 5.0, m, dtype=torch.float64)
    g1 = torch.outer(u, v)

    # Also a second linearly independent rank-1 matrix for multi-step tests
    gen = torch.Generator().manual_seed(0)
    u2 = torch.randn(m, generator=gen, dtype=torch.float64)
    v2 = torch.randn(m, generator=gen, dtype=torch.float64)
    g2 = torch.outer(u2, v2)

    # Construct each optimizer transform on g1
    frob_sgd = g1 / g1.norm()
    d_l = torch.diag(torch.linspace(0.5, 3.0, m, dtype=torch.float64))
    d_r = torch.diag(torch.linspace(0.2, 2.5, m, dtype=torch.float64))
    separable_dl_g_dr = d_l @ g1 @ d_r

    # Ideal / Shazeer-Stern Adafactor (eps1 added before row/col mean -> still separable!)
    eps1 = 1e-6
    r_ss = (g1.square() + eps1).mean(dim=1).sqrt()
    c_ss = (g1.square() + eps1).mean(dim=0).sqrt()
    v_ss = torch.outer(r_ss, c_ss) / r_ss.square().mean().sqrt()
    adafactor_ss = g1 / v_ss

    # ParticleGAN law4_adafactor_mp (additive delta = 0.015 * peak + 1e-8 AFTER outer product!)
    r_pg = g1.square().mean(dim=1).sqrt()
    c_pg = g1.square().mean(dim=0).sqrt()
    r_rms_pg = r_pg.square().mean().sqrt()
    v_pg = torch.outer(r_pg, c_pg) / r_rms_pg
    delta_pg = 0.015 * float(r_rms_pg.item()) + 1e-8
    adafactor_pg = g1 / (v_pg + delta_pg)

    # Adam step-1 (eps=0 vs eps=1e-8 vs ParticleGAN settle_rel=0.015*rms)
    adam_s1_eps0 = g1 / g1.abs()
    adam_s1_eps1e8 = g1 / (g1.abs() + 1e-8)
    adam_s1_pg = g1 / (g1.abs() + 0.015 * float(g1.square().mean().sqrt().item()) + 1e-8)

    # Adam step-2 with beta1=0, beta2=0.999, eps=0 on rank-1 stream (g1 then g2)
    b2 = 0.999
    v_acc = (b2 * g1.square() + (1.0 - b2) * g2.square()) / (1.0 - b2 * b2)
    adam_s2_eps0_on_rank1 = g2 / v_acc.sqrt()

    # Exact SignSGD vs Smooth SignSGD on rank-1
    sign_on_rank1 = torch.sign(g1)
    smooth_sign_on_rank1 = g1 / torch.hypot(g1.abs(), torch.full_like(g1, 0.5))

    # Exact SignSGD on exact rank-2 matrix G_ij = j - i + 0.5
    idx = torch.arange(m, dtype=torch.float64)
    g_rank2 = idx.unsqueeze(0) - idx.unsqueeze(1) + 0.5  # exact rank 2!
    sign_on_rank2 = torch.sign(g_rank2)

    # 1-step Shampoo L G R = (G G^T + s^2 I)^{-1/4} G (G^T G + s^2 I)^{-1/4}
    def shampoo_1step(mat: torch.Tensor, s: float = 0.1) -> torch.Tensor:
        x = mat.to(torch.float64)
        el, ql = torch.linalg.eigh(x @ x.T + (s * s) * torch.eye(x.shape[0], dtype=torch.float64))
        er, qr = torch.linalg.eigh(x.T @ x + (s * s) * torch.eye(x.shape[1], dtype=torch.float64))
        l_mat = (ql * el.pow(-0.25)) @ ql.T
        r_mat = (qr * er.pow(-0.25)) @ qr.T
        return l_mat @ x @ r_mat

    shampoo_r1 = shampoo_1step(g1, s=0.1)
    polar_tikh_r1 = polar_factor(g1, smoothing=0.1)
    shampoo_vs_polar_err_r1 = float((shampoo_r1 - polar_tikh_r1).norm().item())

    # ParticleGAN NS on rank-1 (FP64 reference and actual v11 FP32)
    ns_fp64_s0 = ref_poly_ns(g1, dtype=torch.float64, steps=5, smoothing=0.0)
    ns_fp64_s01 = ref_poly_ns(g1, dtype=torch.float64, steps=5, smoothing=0.1)
    ns_v11_s0 = v11_quintic_2d(g1.float(), steps=5, smoothing=0.0)
    ns_v11_s01 = v11_quintic_2d(g1.float(), steps=5, smoothing=0.1)

    transforms = {
        "input_rank1": g1,
        "frob_sgd": frob_sgd,
        "separable_DL_G_DR": separable_dl_g_dr,
        "adafactor_shazeer_stern": adafactor_ss,
        "adafactor_particlegan_law4": adafactor_pg,
        "adam_step1_eps0": adam_s1_eps0,
        "adam_step1_eps1e-8": adam_s1_eps1e8,
        "adam_step1_peak_rms": adam_s1_pg,
        "adam_step2_eps0_b1_0_on_rank1_g2": adam_s2_eps0_on_rank1,
        "exact_sign_on_rank1": sign_on_rank1,
        "smooth_sign_s0.5_on_rank1": smooth_sign_on_rank1,
        "input_rank2_for_sign": g_rank2,
        "exact_sign_on_rank2": sign_on_rank2,
        "shampoo_1step_s0.1_on_rank1": shampoo_r1,
        "exact_polar_s0.1_on_rank1": polar_tikh_r1,
        "ns_fp64_s0_on_rank1": ns_fp64_s0,
        "ns_fp64_s0.1_on_rank1": ns_fp64_s01,
        "ns_v11_fp32_s0_on_rank1": ns_v11_s0,
        "ns_v11_fp32_s0.1_on_rank1": ns_v11_s01,
    }
    out["rank1_and_rank2_transforms"] = {k: spectral_metrics(t) for k, t in transforms.items()}
    out["shampoo_1step_vs_tikhonov_polar_l2_diff_rank1"] = shampoo_vs_polar_err_r1

    # 1c. Exact rank vs Effective rank under Diagonal Separable vs Full Kronecker (Shampoo)
    # Compare axis-aligned ill-conditioned vs orthogonally rotated ill-conditioned (kappa = 1e4)
    sv_ill = torch.tensor([1.0] + [1e-4] * (m - 1), dtype=torch.float64)
    g_axis = torch.diag(sv_ill)
    q1, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(2), dtype=torch.float64))
    g_rot = q1 @ g_axis @ q2.T

    def _adafactor_ss(mat: torch.Tensor) -> torch.Tensor:
        r_v = (mat.square() + 1e-16).mean(dim=1).sqrt()
        c_v = (mat.square() + 1e-16).mean(dim=0).sqrt()
        return mat / (torch.outer(r_v, c_v) / r_v.square().mean().sqrt())

    shampoo_rot = shampoo_1step(g_rot, s=1e-6)
    polar_rot = polar_factor(g_rot, smoothing=1e-6)
    out["diagonal_vs_kronecker_effective_rank"] = {
        "axis_aligned_raw": spectral_metrics(g_axis),
        "axis_aligned_adafactor_ss": spectral_metrics(_adafactor_ss(g_axis)),
        "rotated_raw": spectral_metrics(g_rot),
        "rotated_adafactor_ss": spectral_metrics(_adafactor_ss(g_rot)),
        "rotated_shampoo_1step_s1e-6": spectral_metrics(shampoo_rot),
        "rotated_polar_s1e-6": spectral_metrics(polar_rot),
        "rotated_shampoo_vs_polar_l2_diff": float((shampoo_rot - polar_rot).norm().item()),
        "rotated_ns_fp64_s0_5step": spectral_metrics(ref_poly_ns(g_rot, dtype=torch.float64, steps=5, smoothing=0.0)),
    }
    return out


# ---------------------------------------------------------------------------
# Experiment 2: Newton-Schulz Across Rank Regimes & Precision (FP64/32/BF16/16)
# ---------------------------------------------------------------------------

def run_experiment_2(device: torch.device) -> Dict[str, Any]:
    """Test NS on exact rank-deficient, nearly rank-deficient, and full-rank matrices across dtypes."""
    m = 64
    q_u, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(10), dtype=torch.float64))
    q_v, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(11), dtype=torch.float64))

    def make_mat(svs: List[float]) -> torch.Tensor:
        s = torch.tensor(svs, dtype=torch.float64)
        return (q_u * s) @ q_v.T

    # 2a. Exact rank-deficient (rank 1, 2, 4) across FP64, FP32, BF16, FP16
    dtypes = [
        ("fp64", torch.float64),
        ("fp32", torch.float32),
        ("bf16", torch.bfloat16),
        ("fp16", torch.float16),
    ]
    exact_def: Dict[str, Any] = {}
    for r_exact in (1, 2, 4):
        svs = [1.0] * r_exact + [0.0] * (m - r_exact)
        g = make_mat(svs)
        entry: Dict[str, Any] = {"input": spectral_metrics(g)}
        for dname, dt in dtypes:
            dev_run = device if (device.type == "cuda" and dt in (torch.bfloat16, torch.float16, torch.float32)) else torch.device("cpu")
            g_dev = g.to(device=dev_run)
            y_s0 = ref_poly_ns(g_dev, dtype=dt, steps=5, smoothing=0.0, rescale_to_target_frob=False)
            y_s01 = ref_poly_ns(g_dev, dtype=dt, steps=5, smoothing=0.01, rescale_to_target_frob=False)
            entry[f"ref_ns_s0_{dname}"] = {
                **spectral_metrics(y_s0),
                **subspace_leakage(g, y_s0, r_exact),
            }
            entry[f"ref_ns_tikh_s0.01_{dname}"] = {
                **spectral_metrics(y_s01),
                **subspace_leakage(g, y_s01, r_exact),
            }
        # Actual ParticleGAN v11 quintic_orthogonalize_2d on CPU & CUDA (note BF16 cutoff at m>=64 for s=0, m>256 for s>0!)
        for tag, dev_t in [("cpu", torch.device("cpu")), *([("cuda", device)] if device.type == "cuda" else [])]:
            g32 = g.to(device=dev_t, dtype=torch.float32)
            v11_s0 = v11_quintic_2d(g32, steps=5, smoothing=0.0)
            v11_s006 = v11_quintic_2d(g32, steps=5, smoothing=0.0061)
            t4_s0 = t4_quintic_2d(g32, steps=5)
            cq_s006 = quintic_direction(g32, smoothing=0.0061)
            entry[f"particlegan_v11_s0_{tag}"] = {
                **spectral_metrics(v11_s0),
                **subspace_leakage(g, v11_s0, r_exact),
            }
            entry[f"particlegan_v11_tikh_s0.0061_{tag}"] = {
                **spectral_metrics(v11_s006),
                **subspace_leakage(g, v11_s006, r_exact),
            }
            entry[f"particlegan_t4_s0_{tag}"] = {
                **spectral_metrics(t4_s0),
                **subspace_leakage(g, t4_s0, r_exact),
            }
            entry[f"particlegan_conv_quintic_s0.0061_{tag}"] = {
                **spectral_metrics(cq_s006),
                **subspace_leakage(g, cq_s006, r_exact),
            }
        exact_def[f"rank_{r_exact}"] = entry

    # Also test 128x128 rank-1 blowup check in v11_quintic_2d(s=0) vs v11_quintic_2d(s=0.0061)
    q128_u, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(12), dtype=torch.float64))
    q128_v, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(13), dtype=torch.float64))
    g128_r1 = torch.outer(q128_u[:, 0], q128_v[:, 0]).to(device=device, dtype=torch.float32)
    v11_128_s0 = v11_quintic_2d(g128_r1, steps=5, smoothing=0.0)
    v11_128_s006 = v11_quintic_2d(g128_r1, steps=5, smoothing=0.0061)
    v11_128_batched_s0 = v11_quintic_batched(g128_r1.unsqueeze(0), steps=5, smoothing=0.0)[0]

    # 2b. Nearly rank-deficient: sigma_1 = 1, sigma_{2..64} = 10^-k for k = 1,2,3,4,6,8,10,12,14
    # Plus 2-mode spectrum: sigma_1 = 1, sigma_2 = 10^-k, sigma_{3..64} = 0
    near_def: Dict[str, Any] = {}
    for exp_k in (1, 2, 3, 4, 6, 8, 10, 12, 14):
        ratio = 10.0 ** (-exp_k)
        svs = [1.0] + [ratio] * (m - 1)
        g = make_mat(svs)
        g32 = g.to(device=device, dtype=torch.float32)
        y_fp64_s0 = ref_poly_ns(g, dtype=torch.float64, steps=5, smoothing=0.0)
        y_fp32_s0 = ref_poly_ns(g32, dtype=torch.float32, steps=5, smoothing=0.0)
        y_bf16_s0 = ref_poly_ns(g32, dtype=torch.bfloat16, steps=5, smoothing=0.0)
        y_v11_s0 = v11_quintic_2d(g32, steps=5, smoothing=0.0)
        y_v11_tikh = v11_quintic_2d(g32, steps=5, smoothing=0.0061)
        y_exact_polar = polar_factor(g, smoothing=0.0)
        near_def[f"ratio_1e-{exp_k}"] = {
            "input": spectral_metrics(g),
            "ns_fp64_s0": spectral_metrics(y_fp64_s0),
            "ns_fp32_s0": spectral_metrics(y_fp32_s0),
            "ns_bf16_s0": spectral_metrics(y_bf16_s0),
            "v11_ns_s0": spectral_metrics(y_v11_s0),
            "v11_ns_tikh_s0.0061": spectral_metrics(y_v11_tikh),
            "exact_polar_fp64": spectral_metrics(y_exact_polar),
        }

    # 2c. Full-rank well-conditioned matrix
    sv_well = torch.linspace(1.0, 0.5, m, dtype=torch.float64).tolist()
    g_well = make_mat(sv_well)
    g_well_dev = g_well.to(device=device, dtype=torch.float32)
    full_rank = {
        "input": spectral_metrics(g_well),
        "ns_fp64_s0": spectral_metrics(ref_poly_ns(g_well, dtype=torch.float64, steps=5, smoothing=0.0)),
        "v11_ns_s0": spectral_metrics(v11_quintic_2d(g_well_dev, steps=5, smoothing=0.0)),
        "v11_ns_tikh_s0.0061": spectral_metrics(v11_quintic_2d(g_well_dev, steps=5, smoothing=0.0061)),
    }

    # 2d. Scalar transfer function of 5-step _QUINTIC_COEFFS vs stationary (3.4445, -4.7750, 2.0315)
    def scalar_poly_5(s0: float, coeffs: Tuple[Tuple[float, float, float], ...]) -> float:
        x = s0
        for a, b, cc in coeffs[:5]:
            x = a * x + b * (x ** 3) + cc * (x ** 5)
        return x

    stat_coeffs = ((_QUINTIC_A, _QUINTIC_B, _QUINTIC_C),) * 5
    slope_v11 = math.prod(a for a, _, _ in _QUINTIC_COEFFS[:5])
    slope_stat = _QUINTIC_A ** 5
    transfer = {
        "v11_zero_slope_prod_a": slope_v11,
        "stationary_zero_slope_a5": slope_stat,
        "samples": {
            f"{s0:.0e}": {
                "v11_5step": scalar_poly_5(s0, _QUINTIC_COEFFS),
                "stationary_5step": scalar_poly_5(s0, stat_coeffs),
            }
            for s0 in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 8e-1, 9.8e-1)
        },
    }

    # 2e. Operations that CAN vs CANNOT introduce new singular directions on rank-1 streams
    gen = torch.Generator().manual_seed(42)
    rank1_stream = [torch.outer(torch.randn(16, generator=gen, dtype=torch.float64),
                                torch.randn(16, generator=gen, dtype=torch.float64)) for _ in range(5)]
    w_orth, _ = torch.linalg.qr(torch.randn(16, 16, generator=gen, dtype=torch.float64))

    # Momentum accumulation over 5 rank-1 steps
    m_buf = torch.zeros(16, 16, dtype=torch.float64)
    b1 = 0.75
    for g_t in rank1_stream:
        m_buf = b1 * m_buf + (1.0 - b1) * g_t

    # Minibatch accumulation of B rank-1 outer products
    mb_ranks = {}
    for bsz in (1, 2, 4, 8, 16, 32):
        h = torch.randn(bsz, 16, generator=gen, dtype=torch.float64)
        d = torch.randn(bsz, 16, generator=gen, dtype=torch.float64)
        g_mb = (d.T @ h) / float(bsz)
        mb_ranks[f"B_{bsz}"] = spectral_metrics(g_mb)["rank_tol_1e-12"]

    # Weight decay added to rank-1 gradient
    g_wd = rank1_stream[0] + 1e-2 * w_orth

    # CountSketch (scalar controller only)
    sk0 = _sketch_grad_67(rank1_stream[0].float(), roll=0)
    sk1 = _sketch_grad_67((-rank1_stream[0]).float(), roll=0)
    flip_cos = float((sk0 * sk1).sum().item())

    # Exact 2xN Cayley-Hamilton polar on exact rank-1 2x16
    g2x16_r1 = torch.outer(torch.tensor([1.0, 2.0]), torch.linspace(1.0, 4.0, 16))
    ep2_s0 = exact_polar_2xn(g2x16_r1, smoothing=0.0)
    ep2_s01 = exact_polar_2xn(g2x16_r1, smoothing=0.1)

    mechanisms = {
        "momentum_5steps_of_rank1": spectral_metrics(m_buf),
        "momentum_5steps_then_ns_fp64": spectral_metrics(ref_poly_ns(m_buf, dtype=torch.float64, steps=5, smoothing=0.0)),
        "weight_decay_1e-2_plus_rank1": spectral_metrics(g_wd),
        "minibatch_outer_product_ranks": mb_ranks,
        "countsketch_sign_flip_cos": flip_cos,
        "exact_polar_2xn_rank1_s0": spectral_metrics(ep2_s0),
        "exact_polar_2xn_rank1_s0.1": spectral_metrics(ep2_s01),
    }

    return {
        "exact_rank_deficient_64x64": exact_def,
        "rank1_128x128_v11_norm_blowup": {
            "v11_s0_single": spectral_metrics(v11_128_s0),
            "v11_s0_batched": spectral_metrics(v11_128_batched_s0),
            "v11_tikh_s0.0061_single": spectral_metrics(v11_128_s006),
            "expected_sqrt_128": math.sqrt(128.0),
        },
        "nearly_rank_deficient_64x64": near_def,
        "full_rank_64x64": full_rank,
        "quintic_scalar_transfer": transfer,
        "rank_generating_mechanisms": mechanisms,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Empirical Gradients on Real ParticleGAN Architectures
# ---------------------------------------------------------------------------

def _build_res16_mlp(z_dim: int = 2, hidden: int = 128, n_blocks: int = 7, out_dim: int = 2) -> nn.Module:
    """16-layer unscaled identity-skip residual MLP matching `grid100_res16_tanh` topology."""
    class _ResBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(width, width)
            self.act = nn.LeakyReLU(0.2)
            self.fc2 = nn.Linear(width, width)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(self.act(self.fc1(x)))

    class _ResMLP16(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(z_dim, hidden)
            self.act = nn.LeakyReLU(0.2)
            self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(n_blocks)])
            self.out_proj = nn.Linear(hidden, out_dim)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = self.act(self.in_proj(z))
            for blk in self.blocks:
                h = blk(h)
            return 5.25 * torch.tanh(self.out_proj(h) / 5.25)

    return _ResMLP16()


def _compare_optimizers_on_gradient(
    grad: torch.Tensor,
    prev_grads: List[torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Apply all candidate Law-4 preconditioners to a single real 2D gradient matrix."""
    g64 = grad.detach().to(device="cpu", dtype=torch.float64)
    g32 = grad.detach().to(device=device, dtype=torch.float32)

    # 1. Raw gradient
    # 2. Frobenius-MP
    u_frob = g64 / g64.norm().clamp_min(1e-38)

    # 3. Exact SVD Polar (s=0)
    u_polar = polar_factor(g64, smoothing=0.0)

    # 4. ParticleGAN v11 Quintic NS:
    #    (a) s=0 on target device (BF16 on CUDA for m>=64),
    #    (b) s=0 pure FP32 reference,
    #    (c) actual per-layer Tikhonov s_eff = min(0.0061, 0.30 * cur_sv_rms) [v11] & 0.50 * cur_sv_rms [v12]
    m_min = min(int(g64.shape[0]), int(g64.shape[1]))
    cur_sv_rms = float(g64.norm().item()) / math.sqrt(float(max(1, m_min)))
    s_eff_v11 = min(0.0061, 0.30 * cur_sv_rms)
    s_eff_v12 = min(0.0061, 0.50 * cur_sv_rms)

    u_v11_s0_dev = v11_quintic_2d(g32, steps=5, smoothing=0.0)
    u_v11_s0_fp32 = ref_poly_ns(g64, dtype=torch.float32, steps=5, smoothing=0.0)
    u_v11_tikh_seff = v11_quintic_2d(g32, steps=5, smoothing=s_eff_v11)
    u_v12_tikh_seff = v11_quintic_2d(g32, steps=5, smoothing=s_eff_v12)

    # 5. Smooth SignSGD (s_eff = 0.03 * elem_rms matching law4_sign_mp) & Exact SignSGD
    elem_rms = float(g32.square().mean().sqrt().item())
    u_sign_exact = torch.sign(g64)
    u_sign_smooth = _smooth_sign_unit(g32, 0.03 * elem_rms)

    # 6. Adam step-1 (eps=1e-8) and multi-step Adam (beta1=0, beta2=0.999 over prev_grads + [grad])
    u_adam_s1 = g64 / (g64.abs() + 1e-8)
    v_ema = torch.zeros_like(g64)
    b2 = 0.999
    all_g = [*prev_grads, g64]
    for g_k in all_g:
        v_ema = b2 * v_ema + (1.0 - b2) * g_k.square()
    v_hat = (v_ema / (1.0 - b2 ** len(all_g))).sqrt()
    u_adam_multi = g64 / (v_hat + 1e-8)
    u_adam_pg = g64 / (v_hat + 0.015 * float(v_hat.square().mean().sqrt().item()) + 1e-8)

    # 7. Ideal Shazeer-Stern Adafactor vs ParticleGAN law4_adafactor_mp
    r_ss = (g64.square() + 1e-16).mean(dim=1).sqrt()
    c_ss = (g64.square() + 1e-16).mean(dim=0).sqrt()
    v_ss = torch.outer(r_ss, c_ss) / r_ss.square().mean().sqrt()
    u_adafactor_ss = g64 / v_ss
    u_adafactor_pg = g64 / (v_ss + 0.015 * float(r_ss.square().mean().sqrt().item()) + 1e-8)

    # 8. 1-step Shampoo (Kronecker L G R, s=1e-6 * s1)
    s1 = float(torch.linalg.svdvals(g64)[0].item())
    lam = max(1e-6 * s1, 1e-30)
    el, ql = torch.linalg.eigh(g64 @ g64.T + (lam * lam) * torch.eye(g64.shape[0], dtype=torch.float64))
    er, qr = torch.linalg.eigh(g64.T @ g64 + (lam * lam) * torch.eye(g64.shape[1], dtype=torch.float64))
    u_shampoo = ((ql * el.pow(-0.25)) @ ql.T) @ g64 @ ((qr * er.pow(-0.25)) @ qr.T)

    # 9. Momentum (beta1=0.75 over all_g) + NS (FP32)
    m_ema = torch.zeros_like(g64)
    b1 = 0.75
    for g_k in all_g:
        m_ema = b1 * m_ema + (1.0 - b1) * g_k
    m_hat = m_ema / (1.0 - b1 ** len(all_g))
    u_mom_ns_fp32 = ref_poly_ns(m_hat, dtype=torch.float32, steps=5, smoothing=0.0)

    candidates = {
        "raw_grad": g64,
        "frob_mp": u_frob,
        "exact_polar_svd": u_polar,
        "ns_5step_fp32_s0": u_v11_s0_fp32,
        "v11_quintic_ns_s0_device": u_v11_s0_dev,
        "v11_tikh_ns_seff_0.30peak": u_v11_tikh_seff,
        "v12_tikh_ns_seff_0.50peak": u_v12_tikh_seff,
        "momentum_0.75_plus_ns_fp32": u_mom_ns_fp32,
        "shampoo_1step_kronecker": u_shampoo,
        "adafactor_shazeer_stern": u_adafactor_ss,
        "adafactor_particlegan": u_adafactor_pg,
        "adam_step1_eps1e-8": u_adam_s1,
        "adam_multistep_eps1e-8": u_adam_multi,
        "adam_multistep_peak_rms": u_adam_pg,
        "sign_exact": u_sign_exact,
        "sign_smooth_law4": u_sign_smooth,
    }
    res: Dict[str, Any] = {}
    for name, upd in candidates.items():
        res[name] = {
            "spectral": spectral_metrics(upd),
            "alignment": alignment_with_polar(g64, upd, top_k_tail=16),
        }
    return res


def run_experiment_3(device: torch.device) -> Dict[str, Any]:
    """Measure real gradient spectra & optimizer responses on MLP16, Shallow MLP, ResMLP16, and DCGAN Conv2d."""
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians
    from experiments.forge.scaling_benchmark import DCGANCritic, DCGANGenerator

    results: Dict[str, Any] = {}

    # Helper to collect adversarial + transport gradients over T minibatches on 100-Gaussians
    def collect_2d_arch_grads(arch_name: str, gen_mod: nn.Module, n_batches: int = 12, batch_size: int = 256) -> Dict[str, Any]:
        disc = SimpleMLPDiscriminator(2, 128, 3, fourier=2)
        deterministic_orthogonal_(gen_mod, seed=0)
        deterministic_orthogonal_(disc, seed=0)
        gen_mod = gen_mod.to(device)
        disc = disc.to(device)
        rng = torch.Generator(device=device).manual_seed(0)

        # Collect gradients across n_batches (fixed weights for clean multi-batch Adam v_t, plus actual 1-step & 20-step)
        grad_history: Dict[str, List[torch.Tensor]] = {}
        for _ in range(n_batches):
            z = torch.randn(batch_size, 2, generator=rng, device=device)
            real = sample_100gaussians(batch_size, device, generator=rng)
            gen_mod.zero_grad(set_to_none=True)
            fake = gen_mod(z)
            # Standard GAN critic objective + weak transport pull (matching ParticleGAN G-phase signals)
            loss_adv = -disc(fake).mean()
            # Nearest-real transport term (deterministic proxy for Law 1 pullback without stateful side effects)
            dists = torch.cdist(fake, real)
            loss_ot = dists.min(dim=1).values.square().mean()
            loss = loss_adv + 0.5 * loss_ot
            loss.backward()
            for name, p in gen_mod.named_parameters():
                if p.ndim == 2 and min(p.shape) >= 16 and p.grad is not None:
                    grad_history.setdefault(name, []).append(p.grad.detach().to("cpu", dtype=torch.float64))

        # Summarize across ALL square/wide hidden 2D layers on last batch
        layer_summaries: List[Dict[str, Any]] = []
        for name, hist in grad_history.items():
            sm = spectral_metrics(hist[-1])
            layer_summaries.append({"name": name, **sm})

        # Pick representative early-mid layer (where true subdominant spectrum is visible above BF16 noise)
        # AND mid-network layer for full optimizer comparison
        keys = list(grad_history.keys())
        early_idx = min(2, len(keys) - 1)
        mid_idx = len(keys) // 2
        early_name = keys[early_idx]
        mid_name = keys[mid_idx]
        early_cmp = _compare_optimizers_on_gradient(grad_history[early_name][-1], grad_history[early_name][:-1], device)
        opt_cmp = _compare_optimizers_on_gradient(grad_history[mid_name][-1], grad_history[mid_name][:-1], device)

        # Aggregate medians across hidden layers
        def _med(vals: List[float]) -> float:
            s = sorted(vals)
            return s[len(s) // 2]

        return {
            "num_wide_2d_layers": len(layer_summaries),
            "median_rank_tol_1e-12": _med([float(x["rank_tol_1e-12"]) for x in layer_summaries]),
            "median_rank_tol_1e-6": _med([float(x["rank_tol_1e-6"]) for x in layer_summaries]),
            "median_stable_rank": _med([x["stable_rank"] for x in layer_summaries]),
            "median_participation_ratio": _med([x["participation_ratio"] for x in layer_summaries]),
            "median_rv_erank": _med([x["rv_erank"] for x in layer_summaries]),
            "median_top1_energy": _med([x["top1_energy"] for x in layer_summaries]),
            "median_sigma2_over_sigma1": _med([x["sigma_2"] / max(x["sigma_1"], 1e-30) for x in layer_summaries]),
            "layer_grad_norm_skew_max_over_min": max(x["frob"] for x in layer_summaries) / max(min(x["frob"] for x in layer_summaries), 1e-38),
            "layers": layer_summaries,
            "early_layer_name": early_name,
            "early_layer_optimizer_comparison": early_cmp,
            "mid_layer_name": mid_name,
            "mid_layer_optimizer_comparison": opt_cmp,
        }

    # 3a. 16-layer skip-free MLP (grid100_mlp16)
    results["grid100_mlp16"] = collect_2d_arch_grads(
        "grid100_mlp16", SimpleMLPGenerator(2, 128, n_hidden=16, out_dim=2)
    )

    # 3b. 3-layer shallow MLP (grid100)
    results["grid100_shallow_mlp3"] = collect_2d_arch_grads(
        "grid100_shallow_mlp3", SimpleMLPGenerator(2, 128, n_hidden=3, out_dim=2)
    )

    # 3c. 16-layer residual MLP (grid100_res16_tanh topology)
    results["grid100_res16_tanh"] = collect_2d_arch_grads(
        "grid100_res16_tanh", _build_res16_mlp(2, 128, n_blocks=7, out_dim=2)
    )

    # 3d. Also measure ACTUAL trained step-50 gradients via real ParticleGAN `unified_v12` on grid100_mlp16!
    recipe = particlegan.get_recipe("unified_v12")
    g_train = SimpleMLPGenerator(2, 128, n_hidden=16, out_dim=2)
    d_train = SimpleMLPDiscriminator(2, 128, 3, fourier=2)
    deterministic_orthogonal_(g_train, seed=0)
    deterministic_orthogonal_(d_train, seed=0)
    g_train = g_train.to(device)
    d_train = d_train.to(device)
    prior = recipe.make_prior(device=device)
    deterministic_orthogonal_(prior, seed=0)
    trainer = particlegan.GANTrainer(recipe, g_train, d_train, prior=prior, seed=0)
    rng_t = torch.Generator(device=device).manual_seed(0)
    step_snaps: Dict[str, Any] = {}
    for step_idx in range(1, 51):
        real = sample_100gaussians(256, device, generator=rng_t)
        trainer.step(real)
        if step_idx in (1, 10, 50):
            # Capture fresh generator gradient at this trained checkpoint
            g_train.zero_grad(set_to_none=True)
            z_p, _ = prior.sample(256, generator=rng_t)
            fake_p = g_train(z_p)
            loss_p = -d_train(fake_p).mean()
            loss_p.backward()
            g_specs = []
            d_specs = []
            for name, p in g_train.named_parameters():
                if p.ndim == 2 and min(p.shape) >= 16 and p.grad is not None:
                    g_specs.append({"name": name, **spectral_metrics(p.grad)})
            # Also capture critic gradients
            d_train.zero_grad(set_to_none=True)
            d_loss = -(d_train(real).mean() - d_train(fake_p.detach()).mean())
            d_loss.backward()
            for name, p in d_train.named_parameters():
                if p.ndim == 2 and min(p.shape) >= 16 and p.grad is not None:
                    d_specs.append({"name": name, **spectral_metrics(p.grad)})
            def _med(xs: List[float]) -> float:
                s = sorted(xs)
                return s[len(s) // 2]
            opt_g_ref = getattr(trainer, "opt_g", getattr(trainer, "g_opt", None))
            step_snaps[f"step_{step_idx}"] = {
                "g_ema_func_gain": float(getattr(opt_g_ref, "_ema_func_gain", 1.0)),
                "g_median_rank_tol_1e-12": _med([float(x["rank_tol_1e-12"]) for x in g_specs]),
                "g_median_rank_tol_1e-6": _med([float(x["rank_tol_1e-6"]) for x in g_specs]),
                "g_median_stable_rank": _med([x["stable_rank"] for x in g_specs]),
                "g_median_rv_erank": _med([x["rv_erank"] for x in g_specs]),
                "g_median_top1_energy": _med([x["top1_energy"] for x in g_specs]),
                "d_median_rank_tol_1e-12": _med([float(x["rank_tol_1e-12"]) for x in d_specs]),
                "d_median_stable_rank": _med([x["stable_rank"] for x in d_specs]),
                "d_median_rv_erank": _med([x["rv_erank"] for x in d_specs]),
            }
    results["grid100_mlp16_trained_unified_v12_snapshots"] = step_snaps

    # 3e. 4D Conv2d / ConvTranspose2d DCGAN (r64, matching Forge `dcgan_performance_r64`)
    dcgan_g = DCGANGenerator(64, 128, 64, 512)
    dcgan_d = DCGANCritic(64, 64, 512)
    deterministic_orthogonal_(dcgan_g, seed=0)
    deterministic_orthogonal_(dcgan_d, seed=0)
    dcgan_g = dcgan_g.to(device)
    dcgan_d = dcgan_d.to(device)
    rng_c = torch.Generator(device=device).manual_seed(0)
    z_c = torch.randn(16, 128, generator=rng_c, device=device)
    real_img = torch.randn(16, 3, 64, 64, generator=rng_c, device=device).tanh()

    # G gradient
    dcgan_g.zero_grad(set_to_none=True)
    fake_img = dcgan_g(z_c)
    loss_gc = -dcgan_d(fake_img).mean() + 0.5 * (fake_img - real_img).square().mean()
    loss_gc.backward()

    # D gradient
    dcgan_d.zero_grad(set_to_none=True)
    loss_dc = -(dcgan_d(real_img).mean() - dcgan_d(fake_img.detach()).mean())
    loss_dc.backward()

    conv_layers: List[Dict[str, Any]] = []
    for role_name, mod in (("dcgan_G", dcgan_g), ("dcgan_D", dcgan_d)):
        for name, p in mod.named_parameters():
            if p.ndim == 4 and p.grad is not None and min(p.shape[0], p.shape[1]) >= 16:
                g4 = p.grad.detach()
                c0, c1, kh, kw = (int(s) for s in g4.shape)
                # (1) Flattened 2D [c0, c1 * kh * kw]
                flat_m = spectral_metrics(g4.reshape(c0, c1 * kh * kw))
                # (2) Per-offset spatial slices [kh * kw, c0, c1] (the exact matrices NS operated on!)
                slices = g4.permute(2, 3, 0, 1).reshape(kh * kw, c0, c1)
                sl_metrics = [spectral_metrics(slices[i]) for i in range(kh * kw)]
                def _med(xs: List[float]) -> float:
                    s = sorted(xs)
                    return s[len(s) // 2]
                conv_layers.append({
                    "role": role_name,
                    "name": name,
                    "shape": [c0, c1, kh, kw],
                    "flat_2d": flat_m,
                    "per_offset_median_rank_1e-12": _med([float(x["rank_tol_1e-12"]) for x in sl_metrics]),
                    "per_offset_median_rank_1e-6": _med([float(x["rank_tol_1e-6"]) for x in sl_metrics]),
                    "per_offset_median_stable_rank": _med([x["stable_rank"] for x in sl_metrics]),
                    "per_offset_median_pr": _med([x["participation_ratio"] for x in sl_metrics]),
                    "per_offset_median_rv_erank": _med([x["rv_erank"] for x in sl_metrics]),
                    "per_offset_median_top1_energy": _med([x["top1_energy"] for x in sl_metrics]),
                })

    results["dcgan_r64_conv4d"] = {
        "layers": conv_layers,
        "median_per_offset_stable_rank": sorted(x["per_offset_median_stable_rank"] for x in conv_layers)[len(conv_layers) // 2],
        "median_per_offset_rv_erank": sorted(x["per_offset_median_rv_erank"] for x in conv_layers)[len(conv_layers) // 2],
        "median_flat_stable_rank": sorted(x["flat_2d"]["stable_rank"] for x in conv_layers)[len(conv_layers) // 2],
        "median_flat_rv_erank": sorted(x["flat_2d"]["rv_erank"] for x in conv_layers)[len(conv_layers) // 2],
    }

    return results


# ---------------------------------------------------------------------------
# Pretty Console Summary & Main Entrypoint
# ---------------------------------------------------------------------------

def print_summary(report: Dict[str, Any]) -> None:
    print("=== P0 LAW 4.3 AUDIT SUMMARY ===")
    print("particlegan:", report["meta"]["particlegan_file"])
    print("device:", report["meta"]["device"])

    print("\n--- Exp 1: Rank-1 / Rank-2 Preconditioner Counterexamples (16x16) ---")
    e1 = report["experiment_1"]["rank1_and_rank2_transforms"]
    for k, m in e1.items():
        print(
            f"  {k:34s} | rank(1e-12)={m['rank_tol_1e-12']:2d} | rank(1e-6)={m['rank_tol_1e-6']:2d} "
            f"| s_rank={m['stable_rank']:6.3f} | rv_erank={m['rv_erank']:6.3f} | s2/s1={m['sigma_2']/max(m['sigma_1'],1e-30):.2e}"
        )
    print(f"  Shampoo-1step vs Tikhonov-Polar L2 diff (rank-1): {report['experiment_1']['shampoo_1step_vs_tikhonov_polar_l2_diff_rank1']:.3e}")
    dvk = report["experiment_1"]["diagonal_vs_kronecker_effective_rank"]
    print("  Diagonal vs Kronecker on rotated kappa=1e4 (16x16):")
    for k in ("axis_aligned_raw", "axis_aligned_adafactor_ss", "rotated_raw", "rotated_adafactor_ss", "rotated_shampoo_1step_s1e-6", "rotated_ns_fp64_s0_5step"):
        m = dvk[k]
        print(f"    {k:30s} | rank={m['rank_tol_1e-12']:2d} | s_rank={m['stable_rank']:6.3f} | rv_erank={m['rv_erank']:6.3f}")
    print(f"    Rotated Shampoo vs Polar L2 diff: {dvk['rotated_shampoo_vs_polar_l2_diff']:.3e}")

    print("\n--- Exp 2: NS on Exact Rank-1 (64x64) Across Dtypes & Implementations ---")
    e2_r1 = report["experiment_2"]["exact_rank_deficient_64x64"]["rank_1"]
    for k, m in e2_r1.items():
        if k == "input":
            continue
        print(
            f"  {k:34s} | rank(1e-12)={m['rank_tol_1e-12']:2d} | rank(1e-6)={m['rank_tol_1e-6']:2d} "
            f"| s1={m['sigma_1']:7.3f} | s2/s1={m['sigma_2']/max(m['sigma_1'],1e-30):.2e} "
            f"| col_leak={m['col_leak_rel']:.2e}"
        )
    nb = report["experiment_2"]["rank1_128x128_v11_norm_blowup"]
    print(
        f"  128x128 rank-1 v11_quintic_2d(s=0) sigma_1={nb['v11_s0_single']['sigma_1']:.4f} "
        f"(sqrt(128)={nb['expected_sqrt_128']:.4f}) vs Tikhonov(s=0.0061) sigma_1={nb['v11_tikh_s0.0061_single']['sigma_1']:.4f}"
    )

    print("\n--- Exp 2b: Nearly Rank-Deficient (64x64, s1=1, s_{2..64}=10^-k) ---")
    for k, blk in report["experiment_2"]["nearly_rank_deficient_64x64"].items():
        s_in = blk["input"]["stable_rank"]
        s_64 = blk["ns_fp64_s0"]["stable_rank"]
        s_bf = blk["ns_bf16_s0"]["stable_rank"]
        s_v11 = blk["v11_ns_s0"]["stable_rank"]
        s_tk = blk["v11_ns_tikh_s0.0061"]["stable_rank"]
        rv_v11 = blk["v11_ns_s0"]["rv_erank"]
        print(
            f"  {k:12s} | in_srank={s_in:5.2f} | fp64_s0={s_64:6.2f} | bf16_s0={s_bf:6.2f} "
            f"| v11_s0={s_v11:6.2f} (rv={rv_v11:6.2f}) | v11_tikh={s_tk:6.2f}"
        )

    print("\n--- Exp 3: Real Architecture Gradient Spectra ---")
    e3 = report["experiment_3"]
    for arch in ("grid100_mlp16", "grid100_shallow_mlp3", "grid100_res16_tanh"):
        a = e3[arch]
        print(
            f"  {arch:22s} | layers={a['num_wide_2d_layers']:2d} | med_rank(1e-12)={a['median_rank_tol_1e-12']:5.1f} "
            f"| med_rank(1e-6)={a['median_rank_tol_1e-6']:5.1f} | med_srank={a['median_stable_rank']:5.2f} "
            f"| med_PR={a['median_participation_ratio']:5.2f} | med_rv_erank={a['median_rv_erank']:6.2f} "
            f"| top1%={100*a['median_top1_energy']:5.1f}% | norm_skew={a['layer_grad_norm_skew_max_over_min']:.1f}x"
        )
    dc = e3["dcgan_r64_conv4d"]
    print(
        f"  {'dcgan_r64_conv4d':22s} | layers={len(dc['layers']):2d} | per_off_srank={dc['median_per_offset_stable_rank']:5.2f} "
        f"| per_off_rv={dc['median_per_offset_rv_erank']:6.2f} | flat_srank={dc['median_flat_stable_rank']:5.2f} "
        f"| flat_rv={dc['median_flat_rv_erank']:6.2f}"
    )

    for label, key_n, key_c in (
        ("Early-Mid Layer (net.6, s2/s1~0.10)", "early_layer_name", "early_layer_optimizer_comparison"),
        ("Mid Layer (net.16, s2/s1~1.6e-4)", "mid_layer_name", "mid_layer_optimizer_comparison"),
    ):
        print(f"\n--- Exp 3 (MLP16 {label}: {e3['grid100_mlp16'][key_n]}) ---")
        cmp_m = e3["grid100_mlp16"][key_c]
        for name, blk in cmp_m.items():
            sp = blk["spectral"]
            al = blk["alignment"]
            print(
                f"  {name:30s} | rk(1e-6)={sp['rank_tol_1e-6']:3d} | s_rk={sp['stable_rank']:6.2f} "
                f"| rv={sp['rv_erank']:6.2f} | cos_pol={al['cos_polar_full']:5.3f} "
                f"| cos_sub16={al['cos_polar_subdominant_2_to_k']:5.3f} | sub_diag_E={100*al['subdominant_diag_energy_frac']:5.1f}% "
                f"| off_basis_E={100*al['off_singular_basis_energy_frac']:5.1f}%"
            )

    print("\n--- Exp 3 (Trained unified_v12 Snapshots on grid100_mlp16) ---")
    for st_k, st_v in e3["grid100_mlp16_trained_unified_v12_snapshots"].items():
        print(
            f"  {st_k:8s} | ema_fg={st_v['g_ema_func_gain']:.3f} | G: rk(1e-12)={st_v['g_median_rank_tol_1e-12']:.0f}, "
            f"rk(1e-6)={st_v['g_median_rank_tol_1e-6']:.0f}, s_rk={st_v['g_median_stable_rank']:.2f}, "
            f"rv={st_v['g_median_rv_erank']:.2f}, top1={100*st_v['g_median_top1_energy']:.1f}% | "
            f"D: s_rk={st_v['d_median_stable_rank']:.2f}, rv={st_v['d_median_rv_erank']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 Law 4.3 rank & Newton-Schulz audit experiments")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="", help="Optional JSON output path (not tracked in git)")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    report = {
        "meta": {
            "particlegan_file": str(particlegan.__file__),
            "torch_version": str(torch.__version__),
            "device": str(device),
            "seed": 0,
        },
        "experiment_1": run_experiment_1(),
        "experiment_2": run_experiment_2(device),
        "experiment_3": run_experiment_3(device),
    }
    print_summary(report)
    if args.output:
        import os
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote JSON summary to {args.output}")


if __name__ == "__main__":
    main()
