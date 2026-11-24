"""P6 Adversarial Mathematical, Autodiff & Computational Scaling Audit of Law 2.

Reproducibly verifies:
1. Exact derivative-order identity: ``fwAD.dual_level()`` + ``.backward()`` computes
   exact reverse-over-forward (RoF) mixed second-order derivatives ``(d^2 D / dtheta dx) u``
   identical to closed-form analytical mixed Hessians and reverse-over-reverse (RoR)
   ``torch.autograd.grad(..., create_graph=True)`` on both smooth (Tanh/SiLU/Fourier)
   and piecewise-linear (ReLU/LeakyReLU) critics, plus end-to-end audits of every
   ParticleGAN penalty estimator (exact canonical, Householder spatial-frame,
   Murmur3 Rademacher + k-NN Hutchinson pooling, and Conv2d image paths).
2. Exact bias, variance, and Jensen's inequality bounds for nonlinear critic penalties:
   - Strictly concave Born-Infeld ``phi_beta(s) = (sqrt(1 + beta s) - 1) / beta``
     (used in ``minimal``): strict negative loss & parameter-gradient bias.
   - Strictly convex-on-cap EBI ``[softplus_beta(sqrt(s) - kappa)]^2`` (used in
     ``t2_secant``, ``unified``, ``unified_v11``/``v12``): strict positive Jensen bias
     and false penalty triggering below the Lipschitz cap ``kappa``.
   - Separable vs. dense Rademacher variance blowup theorem (4x on 2-way rank-1
     spatial gradients; 13x + full batch correlation on ``minimal`` 3-way probes).
3. ReLU/LeakyReLU kink impulses (``O(1/delta)``), BF16 quantization underflow &
   cancellation on actual ``DCGANCritic`` (``t2_secant`` ``delta=0.02`` and ``minimal``
   ``eps=1e-5`` vs. ``unified_v11`` ``fwAD``), and transport/cyclic chord failure modes.
4. Matched RTX A6000 computational scaling regimes (latency, peak VRAM, NVML energy):
   - MLP dimension crossover (``d in {2, 4, 8, 16, 32, 64, 128, 256, 1024}``),
   - Conv2d spatial resolution scaling (``32^2``..``256^2``, FP32 & BF16),
   - Architectural compatibility matrix (BatchNorm train/eval, GroupNorm, ResNet skip,
     Flash/MemEff vs. Math SDPA attention, and ``EmulatedLinear`` MXFP8/NVFP4).

Usage:
    PYTHONPATH="$PWD" python scripts/four_laws_audit/p6_law2_autodiff.py
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from experiments.forge.scaling_benchmark import DCGANCritic
from particlegan._minimal.geometry import _unit_probes as minimal_unit_probes
from particlegan._t2_critic.penalty import (
    _build_orthonormal_frame,
    _eval_forward_tangent_sq,
    _prepare_spatial_tangent_plan,
    _separable_image_unit_perturbation,
)
from particlegan._unified.penalty import _eval_lowd_forward_grad_sq
from particlegan._unified_v11.penalty import (
    _bn_dual_bf16_hook,
    _prepare_batched_highd_tangent_plans,
    _universal_image_unit_tangent,
)
from particlegan.init import deterministic_orthogonal_
from particlegan.low_precision import EmulatedLinear


# ---------------------------------------------------------------------------
# Hardware NVML Energy Monitor (zero external dependencies)
# ---------------------------------------------------------------------------


class NVMLMonitor:
    """Minimal ctypes wrapper over libnvidia-ml.so.1 for exact GPU mJ counters."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)
        self._nvml = None
        self._handle = ctypes.c_void_p()
        self.available = False
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            if nvml.nvmlInit_v2() == 0:
                if nvml.nvmlDeviceGetHandleByIndex_v2(self.device_index, ctypes.byref(self._handle)) == 0:
                    self._nvml = nvml
                    self.available = True
        except Exception:
            self.available = False

    def total_energy_mj(self) -> Optional[int]:
        if not self.available or self._nvml is None:
            return None
        mj = ctypes.c_ulonglong()
        if self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(mj)) == 0:
            return int(mj.value)
        return None

    def close(self) -> None:
        if self.available and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self.available = False


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------


def _flatten_grads(params: List[torch.nn.Parameter]) -> torch.Tensor:
    parts = [
        (p.grad.detach().reshape(-1).double() if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64, device=p.device))
        for p in params
    ]
    return torch.cat(parts)


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def _grad_metrics(g_est: torch.Tensor, g_ref: torch.Tensor) -> Dict[str, float]:
    e = g_est.double().reshape(-1)
    r = g_ref.double().reshape(-1)
    n_e = float(e.norm().item())
    n_r = float(r.norm().item())
    diff = float((e - r).norm().item())
    max_abs = float((e - r).abs().max().item())
    cos = float(torch.dot(e, r).item() / max(n_e * n_r, 1e-30))
    rel_l2 = diff / max(n_r, 1e-30)
    norm_ratio = n_e / max(n_r, 1e-30)
    return {
        "cos": cos,
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "norm_est": n_e,
        "norm_ref": n_r,
        "norm_ratio": norm_ratio,
    }


def _phi_bi(s: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return (torch.sqrt(1.0 + beta * s) - 1.0) / beta


def _phi_ebi(s: torch.Tensor, beta: float = 12.0, kappa: float = 1.0) -> torch.Tensor:
    norm = torch.sqrt(s + 1e-12)
    return F.softplus(norm - kappa, beta=beta).square()


def _make_mlp_critic(
    in_dim: int,
    hidden_dim: int = 128,
    n_hidden: int = 3,
    activation: str = "leaky_relu",
    fourier: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    if fourier > 0 and activation == "leaky_relu":
        m = SimpleMLPDiscriminator(in_dim=in_dim, hidden_dim=hidden_dim, n_hidden=n_hidden, fourier=fourier)
        deterministic_orthogonal_(m, seed=0)
        return m.to(device=device, dtype=dtype)
    act_map: Dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=False),
        "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=False),
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    layers: List[nn.Module] = []
    cur = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(cur, hidden_dim))
        layers.append(act_map[activation]())
        cur = hidden_dim
    layers.append(nn.Linear(cur, 1))
    m = nn.Sequential(*layers)
    deterministic_orthogonal_(m, seed=0)
    return m.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Experiment 1: Exact Mixed Second-Order Derivative Identity & Path Audit
# ---------------------------------------------------------------------------


def _analytical_2layer_mixed_grad(
    W: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    act_name: str,
    outer_fn: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form analytical gradient of mean_i Phi(tau_i^2) for f(x) = v^T sigma(Wx + b).

    Explicitly includes the second derivative sigma''(z) term in d(tau)/dW and d(tau)/db,
    proving mathematically that differentiating tau = <grad_x f, u> w.r.t. theta computes
    the exact mixed second-order derivative (d^2 f / (dtheta dx)) u.
    """
    # x, u: [B, d]; W: [H, d]; b: [H]; v: [H]
    B = x.shape[0]
    z = x @ W.T + b.unsqueeze(0)  # [B, H]
    Wu = u @ W.T  # [B, H]
    if act_name == "tanh":
        s0 = torch.tanh(z)
        s1 = 1.0 - s0 * s0
        s2 = -2.0 * s0 * s1
    elif act_name == "silu":
        sig = torch.sigmoid(z)
        s1 = sig + z * sig * (1.0 - sig)
        s2 = sig * (1.0 - sig) * (2.0 + z * (1.0 - 2.0 * sig))
    else:
        raise ValueError(f"unsupported analytical activation: {act_name}")

    # Directional derivative tau_i = sum_h v_h * s1_{ih} * (Wu)_{ih}
    tau = (s1 * Wu) @ v  # [B]
    s_val = tau * tau  # [B]

    if outer_fn == "bi":
        # Phi(s) = sqrt(1 + s) - 1, Phi'(s) = 0.5 / sqrt(1 + s)
        dphi_ds = 0.5 / torch.sqrt(1.0 + s_val)
    elif outer_fn == "ebi":
        # Phi(s) = softplus_12(sqrt(s + 1e-12) - 1)^2
        r = torch.sqrt(s_val + 1e-12)
        sp = F.softplus(r - 1.0, beta=12.0)
        sig_sp = torch.sigmoid(12.0 * (r - 1.0))
        dphi_ds = (2.0 * sp * sig_sp) / (2.0 * r)
    else:
        raise ValueError(outer_fn)

    # dL / dtau_i = (1/B) * 2 * tau_i * Phi'(tau_i^2)
    w_i = (2.0 * tau * dphi_ds) / float(B)  # [B]

    # Exact mixed second-derivative contractions:
    # d(tau_i)/dv_h = s1_{ih} * (Wu)_{ih}
    grad_v = (w_i.unsqueeze(1) * s1 * Wu).sum(dim=0)  # [H]
    # d(tau_i)/db_h = v_h * s2_{ih} * (Wu)_{ih}   <-- pure sigma'' term!
    a_ih = w_i.unsqueeze(1) * v.unsqueeze(0) * s2 * Wu  # [B, H]
    grad_b = a_ih.sum(dim=0)  # [H]
    # d(tau_i)/dW_{hj} = [v_h * s2_{ih} * (Wu)_{ih}] * x_{ij} + [v_h * s1_{ih}] * u_{ij}
    term_sigma2 = a_ih.T @ x  # [H, d]
    term_bilinear = (w_i.unsqueeze(1) * v.unsqueeze(0) * s1).T @ u  # [H, d]
    grad_W = term_sigma2 + term_bilinear
    return tau, grad_W, grad_b, grad_v


def exp1_derivative_order_and_paths() -> Dict[str, Any]:
    """Verify exact mixed 2nd-order derivative identity and audit all ParticleGAN paths."""
    results: Dict[str, Any] = {}
    g_cpu = torch.Generator().manual_seed(0)

    # --- 1A: Analytical Closed-Form Mixed Hessian Verification (FP64) ---
    analytical_rows = []
    B, d, H = 32, 6, 24
    x = torch.randn(B, d, generator=g_cpu, dtype=torch.float64)
    u = torch.randn(B, d, generator=g_cpu, dtype=torch.float64)
    u = u / u.norm(dim=1, keepdim=True)
    W0 = torch.randn(H, d, generator=g_cpu, dtype=torch.float64) * 0.5
    b0 = torch.randn(H, generator=g_cpu, dtype=torch.float64) * 0.2
    v0 = torch.randn(H, generator=g_cpu, dtype=torch.float64) * 0.5

    for act_name in ("tanh", "silu"):
        act_fn = torch.tanh if act_name == "tanh" else F.silu
        for outer_fn in ("bi", "ebi"):
            loss_fn = (lambda s: _phi_bi(s, 1.0).mean()) if outer_fn == "bi" else (lambda s: _phi_ebi(s, 12.0, 1.0).mean())

            # 1) Closed-form analytical
            tau_an, gW_an, gb_an, gv_an = _analytical_2layer_mixed_grad(W0, b0, v0, x, u, act_name, outer_fn)
            g_an = torch.cat([gW_an.reshape(-1), gb_an.reshape(-1), gv_an.reshape(-1)])
            sigma2_norm = float(gb_an.norm().item())

            # 2) fwAD (RoF): dual_level + .backward()
            W_fw = W0.clone().requires_grad_(True)
            b_fw = b0.clone().requires_grad_(True)
            v_fw = v0.clone().requires_grad_(True)
            with fwAD.dual_level():
                xd = fwAD.make_dual(x, u)
                yd = act_fn(xd @ W_fw.T + b_fw.unsqueeze(0)) @ v_fw
                _, tau_fw = fwAD.unpack_dual(yd)
            loss_fw = loss_fn(tau_fw.square())
            loss_fw.backward()
            g_fw = torch.cat([W_fw.grad.reshape(-1), b_fw.grad.reshape(-1), v_fw.grad.reshape(-1)])

            # 3) RoR: autograd.grad(..., create_graph=True) + .backward()
            W_rr = W0.clone().requires_grad_(True)
            b_rr = b0.clone().requires_grad_(True)
            v_rr = v0.clone().requires_grad_(True)
            x_req = x.clone().requires_grad_(True)
            y_rr = act_fn(x_req @ W_rr.T + b_rr.unsqueeze(0)) @ v_rr
            gx = torch.autograd.grad(y_rr.sum(), x_req, create_graph=True)[0]
            tau_rr = (gx * u).sum(dim=1)
            loss_rr = loss_fn(tau_rr.square())
            loss_rr.backward()
            g_rr = torch.cat([W_rr.grad.reshape(-1), b_rr.grad.reshape(-1), v_rr.grad.reshape(-1)])

            m_fw_an = _grad_metrics(g_fw, g_an)
            m_fw_rr = _grad_metrics(g_fw, g_rr)
            tau_err = float((tau_fw.detach() - tau_an).abs().max().item())
            analytical_rows.append(
                {
                    "activation": act_name,
                    "outer_penalty": outer_fn,
                    "tau_max_abs_err": tau_err,
                    "sigma2_bias_grad_norm": sigma2_norm,
                    "fwad_vs_analytical_rel_l2": m_fw_an["rel_l2"],
                    "fwad_vs_analytical_max_abs": m_fw_an["max_abs"],
                    "fwad_vs_analytical_cos": m_fw_an["cos"],
                    "fwad_vs_ror_rel_l2": m_fw_rr["rel_l2"],
                    "fwad_vs_ror_max_abs": m_fw_rr["max_abs"],
                    "fwad_vs_ror_cos": m_fw_rr["cos"],
                }
            )
    results["analytical_mixed_hessian_fp64"] = analytical_rows

    # --- 1B: Full ||grad_x D||^2 Canonical fwAD vs RoR on Smooth & Piecewise-Linear Critics ---
    canonical_rows = []
    for act_name, fourier in (("relu", 0), ("leaky_relu", 0), ("leaky_relu", 2), ("tanh", 0), ("silu", 0)):
        for dim in (1, 2, 4):
            for dtype_name, dtype in (("fp64", torch.float64), ("fp32", torch.float32)):
                g_loc = torch.Generator().manual_seed(100 + dim)
                x_b = torch.randn(32, dim, generator=g_loc, dtype=dtype)
                m_fw = _make_mlp_critic(dim, 64, 3, activation=act_name, fourier=fourier, dtype=dtype)
                # Normalize output scale so mean ||grad_x D||_2 == 2.0 > kappa=1.0 (active cap regime)
                x_probe_s = x_b.detach().clone().requires_grad_(True)
                gn_init = torch.autograd.grad(m_fw(x_probe_s).sum(), x_probe_s)[0].norm(dim=1).mean().item()
                last_lin = [mod for mod in m_fw.modules() if isinstance(mod, nn.Linear)][-1]
                with torch.no_grad():
                    last_lin.weight.mul_(2.0 / max(gn_init, 1e-12))
                m_rr = _make_mlp_critic(dim, 64, 3, activation=act_name, fourier=fourier, dtype=dtype)
                m_rr.load_state_dict(m_fw.state_dict())

                # Exact canonical fwAD (matching _eval_lowd_forward_grad_sq)
                def live_rep(z_in: torch.Tensor, _m: int) -> torch.Tensor:
                    return m_fw(z_in)

                _, sq_fw = _eval_lowd_forward_grad_sq(live_rep, x_b, can_stack_basis=True, quantized_fprop=False)
                loss_fw = _phi_ebi(sq_fw, beta=12.0, kappa=1.0).mean()
                loss_fw.backward()
                g_fw = _flatten_grads(list(m_fw.parameters()))

                # Exact RoR
                x_req = x_b.detach().clone().requires_grad_(True)
                y_rr = m_rr(x_req)
                gx = torch.autograd.grad(y_rr.sum(), x_req, create_graph=True)[0]
                sq_rr = gx.square().sum(dim=1)
                loss_rr = _phi_ebi(sq_rr, beta=12.0, kappa=1.0).mean()
                loss_rr.backward()
                g_rr = _flatten_grads(list(m_rr.parameters()))

                # Check d^2 D / dx^2 norm on a sample for piecewise-linear vs smooth contrast
                x_one = x_b[:4].detach().clone().requires_grad_(True)
                g1 = torch.autograd.grad(m_rr(x_one).sum(), x_one, create_graph=True)[0]
                h_xx = torch.autograd.grad((g1 * torch.ones_like(g1)).sum(), x_one, create_graph=False)[0]
                hess_xx_norm = float(h_xx.detach().double().norm().item())

                gm = _grad_metrics(g_fw, g_rr)
                canonical_rows.append(
                    {
                        "activation": act_name,
                        "fourier": fourier,
                        "dim": dim,
                        "dtype": dtype_name,
                        "sq_max_abs_err": float((sq_fw.detach() - sq_rr.detach()).abs().max().item()),
                        "hess_xx_probe_norm": hess_xx_norm,
                        "mixed_grad_norm": gm["norm_ref"],
                        "rel_l2": gm["rel_l2"],
                        "max_abs": gm["max_abs"],
                        "cos": gm["cos"],
                    }
                )
    results["canonical_fwad_vs_ror"] = canonical_rows

    # --- 1C: Codebase Estimator Audit (Householder d=6,8 & Murmur3 k-NN Hutchinson d=32,170) ---
    # 1) Inspect chained Householder Q = H_2 H_1 spectrum
    householder_spectrum = []
    for dim in (5, 6, 8):
        Q = _build_orthonormal_frame(dim, dim, phase=0.23, salt=1, device=torch.device("cpu"))
        orth_err = float((Q.T @ Q - torch.eye(dim)).abs().max().item())
        det_val = float(torch.linalg.det(Q).item())
        eigs = torch.linalg.eigvals(Q)
        unit_eigs = int(((eigs - 1.0).abs() < 1e-5).sum().item())
        householder_spectrum.append(
            {
                "dim": dim,
                "orth_max_err": orth_err,
                "det": det_val,
                "unit_eigenvalues": unit_eigs,
                "expected_fixed_subspace_dim": dim - 2,
            }
        )
    results["householder_frame_geometry"] = householder_spectrum

    # 2) End-to-end parameter gradient audit for d=6, 8 (Householder spatial-frame) and d=32, 170 (k-NN Hutchinson)
    path_audit_rows = []
    for dim, regime, critic_kind in (
        (6, "householder_spatial_frame", "linear"),
        (6, "householder_spatial_frame", "nonlinear_mlp"),
        (8, "householder_spatial_frame", "linear"),
        (8, "householder_spatial_frame", "nonlinear_mlp"),
        (32, "hutchinson_knn32", "nonlinear_mlp"),
        (170, "hutchinson_knn32", "nonlinear_mlp"),
    ):
        g_loc = torch.Generator().manual_seed(200 + dim)
        B_eval = 128
        # Bimodal / clustered data for realistic spatial spacing
        centers = torch.randn(4, dim, generator=g_loc) * 1.5
        assign = torch.arange(B_eval) % 4
        x_b = centers[assign] + 0.35 * torch.randn(B_eval, dim, generator=g_loc)

        if critic_kind == "linear":
            m_ref = nn.Sequential(nn.Linear(dim, 1))
            deterministic_orthogonal_(m_ref, seed=0)
            m_ref[0].weight.data.mul_(2.5)
        else:
            m_ref = _make_mlp_critic(dim, 128, 3, activation="leaky_relu", fourier=0)
            # Scale final layer so ||grad_x D|| is around ~1.5 (active cap regime)
            with torch.no_grad():
                m_ref[-1].weight.data.mul_(4.0)

        m_est = _make_mlp_critic(dim, 128, 3, activation="leaky_relu", fourier=0) if critic_kind != "linear" else nn.Sequential(nn.Linear(dim, 1))
        m_est.load_state_dict(m_ref.state_dict())

        # Exact RoR gradient
        x_req = x_b.detach().clone().requires_grad_(True)
        gx = torch.autograd.grad(m_ref(x_req).sum(), x_req, create_graph=True)[0]
        sq_ref = gx.square().sum(dim=1)
        loss_ref = _phi_ebi(sq_ref, beta=12.0, kappa=1.0).mean()
        loss_ref.backward()
        g_ref = _flatten_grads(list(m_ref.parameters()))

        # Actual ParticleGAN _prepare_spatial_tangent_plan + _eval_forward_tangent_sq
        u_plan, recon_fn = _prepare_spatial_tangent_plan(x_b.detach(), phase=0.17, salt=1, pool_k=32)
        _, tau_list = _eval_forward_tangent_sq(lambda z: m_est(z), [x_b], [u_plan], can_concat=False)
        sq_est = recon_fn(tau_list[0])
        loss_est = _phi_ebi(sq_est, beta=12.0, kappa=1.0).mean()
        loss_est.backward()
        g_est = _flatten_grads(list(m_est.parameters()))

        # Also compare unpooled K=1 Rademacher fwAD for d > 8
        unpooled_cos = None
        k_eff_mean = None
        if dim > 8:
            m_unp = _make_mlp_critic(dim, 128, 3, activation="leaky_relu", fourier=0)
            m_unp.load_state_dict(m_ref.state_dict())
            _, tau_unp = _eval_forward_tangent_sq(lambda z: m_unp(z), [x_b], [u_plan], can_concat=False)
            sq_unp = float(dim) * tau_unp[0]
            _phi_ebi(sq_unp, beta=12.0, kappa=1.0).mean().backward()
            unpooled_cos = _grad_metrics(_flatten_grads(list(m_unp.parameters())), g_ref)["cos"]
            # Compute actual effective pooling size k_eff = 1 / sum(w_im^2) from unified_v11 batched helper
            _, _, w_bat = _prepare_batched_highd_tangent_plans(x_b.unsqueeze(0), phase=0.17, pool_k=32)
            k_eff_mean = float((1.0 / w_bat[0].square().sum(dim=1)).mean().item())

        gm = _grad_metrics(g_est, g_ref)
        path_audit_rows.append(
            {
                "dim": dim,
                "regime": regime,
                "critic_kind": critic_kind,
                "true_mean_norm": float(sq_ref.detach().sqrt().mean().item()),
                "est_mean_norm": float(sq_est.detach().sqrt().mean().item()),
                "sq_mean_ratio": float(sq_est.detach().mean().item() / max(float(sq_ref.detach().mean().item()), 1e-12)),
                "grad_cos_vs_ror": gm["cos"],
                "grad_rel_l2_vs_ror": gm["rel_l2"],
                "grad_norm_ratio": gm["norm_ratio"],
                "unpooled_1probe_grad_cos": unpooled_cos,
                "knn32_k_eff_mean": k_eff_mean,
            }
        )
    results["codebase_spatial_and_hutchinson_audit"] = path_audit_rows
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Bias, Variance & Jensen's Inequality Quantification
# ---------------------------------------------------------------------------


def exp2_bias_variance_jensen() -> Dict[str, Any]:
    """Quantify exact Jensen bias (concave BI & convex EBI), gradient bias, and separable variance."""
    results: Dict[str, Any] = {}
    rng = np.random.default_rng(0)

    # Exact 1D Gauss-Hermite reference for Z ~ N(0, 1) (large-d isotropic limit of K=1 Rademacher)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(128)
    z_nodes = np.sqrt(2.0) * gh_x
    z_weights = gh_w / np.sqrt(np.pi)
    t_nodes = z_nodes ** 2  # chi_1^2

    # --- 2A: Strictly Concave Born-Infeld phi_beta(s) = (sqrt(1 + beta*s) - 1) / beta ---
    bi_rows = []
    n_mc = 40000
    for beta in (1.0, 8.0):
        for r_true in (0.5, 1.0, 2.0, 5.0):
            s_true = r_true ** 2
            phi_true = (math.sqrt(1.0 + beta * s_true) - 1.0) / beta
            grad_coeff_true = 1.0 / math.sqrt(1.0 + beta * s_true)

            # Exact chi_1^2 limit (K=1, d -> inf)
            phi_chi1 = float(np.sum(z_weights * ((np.sqrt(1.0 + beta * s_true * t_nodes) - 1.0) / beta)))
            grad_chi1 = float(np.sum(z_weights * (t_nodes / np.sqrt(1.0 + beta * s_true * t_nodes))))

            for dim in (4, 32, 170):
                # Delocalized gradient vector g with ||g||_2 = r_true
                g_raw = rng.standard_normal(dim)
                g_vec = g_raw * (r_true / np.linalg.norm(g_raw))
                for k_probes in (1, 4, 32):
                    # i.i.d. Rademacher probes u in {+-1/sqrt(d)}^d: shape [n_mc, k_probes, d]
                    # Compute s_hat = (d / K) * sum_k (g . u_k)^2 efficiently in chunks
                    eps = rng.choice(np.array([-1.0, 1.0]), size=(n_mc, k_probes, dim))
                    dots = eps @ g_vec  # [n_mc, k_probes], since u = eps/sqrt(d), d*(g.u)^2 = (g.eps)^2
                    s_hat = np.mean(dots ** 2, axis=1)  # [n_mc]
                    phi_hat = (np.sqrt(1.0 + beta * s_hat) - 1.0) / beta
                    # Expected parameter gradient wrt g for linear f(x) = g^T x:
                    # grad_g phi(s_hat) = (1 / sqrt(1 + beta s_hat)) * (1/K) * sum_k (g.eps_k) * eps_k
                    inv_rt = 1.0 / np.sqrt(1.0 + beta * s_hat)  # [n_mc]
                    # Projection along g_vec: <grad_g phi(s_hat), g_vec> / ||g_vec||^2
                    # = inv_rt * s_hat / s_true
                    grad_proj_mean = float(np.mean(inv_rt * s_hat) / s_true)
                    # Full vector expectation E[grad_g phi(s_hat)] to test directional bias too
                    w_dots = (inv_rt[:, None] * dots) / float(k_probes)  # [n_mc, k_probes]
                    grad_vec_mean = np.einsum("mk,mkd->d", w_dots, eps) / float(n_mc)
                    grad_vec_exact = grad_coeff_true * g_vec
                    grad_dir_cos = float(
                        np.dot(grad_vec_mean, grad_vec_exact)
                        / max(np.linalg.norm(grad_vec_mean) * np.linalg.norm(grad_vec_exact), 1e-30)
                    )
                    bi_rows.append(
                        {
                            "beta": beta,
                            "r_true": r_true,
                            "dim": dim,
                            "k_probes": k_probes,
                            "s_hat_mean_ratio": float(np.mean(s_hat) / s_true),
                            "phi_true": phi_true,
                            "phi_mc_mean": float(np.mean(phi_hat)),
                            "phi_rel_jensen_bias": float((np.mean(phi_hat) - phi_true) / phi_true),
                            "phi_chi1_limit_rel_bias": float((phi_chi1 - phi_true) / phi_true),
                            "phi_rel_std": float(np.std(phi_hat) / phi_true),
                            "grad_scale_bias": float((grad_proj_mean - grad_coeff_true) / grad_coeff_true),
                            "grad_chi1_limit_scale_bias": float((grad_chi1 - grad_coeff_true) / grad_coeff_true),
                            "grad_mean_dir_cos": grad_dir_cos,
                        }
                    )
    results["concave_born_infeld_jensen"] = bi_rows

    # --- 2B: Rademacher Directional Distortion on Anisotropic / Sparse Gradient ---
    # Show that for non-uniform g (e.g. g = [2.0, 0.5, 0.5, 0.5]), Rademacher E[grad_g phi(s_hat)]
    # is NOT even collinear with g!
    g_aniso = np.array([2.0, 0.4, 0.4, 0.4], dtype=np.float64)
    # Exact enumeration of all 2^4 = 16 Rademacher sign vectors!
    signs_all = np.array([[1.0 if (m >> i) & 1 else -1.0 for i in range(4)] for m in range(16)], dtype=np.float64)
    dots_all = signs_all @ g_aniso
    s_all = dots_all ** 2
    s_exact = float(np.dot(g_aniso, g_aniso))
    grad_exact = g_aniso / math.sqrt(1.0 + s_exact)
    grad_rad_exact_mean = np.mean((dots_all / np.sqrt(1.0 + s_all))[:, None] * signs_all, axis=0)
    aniso_cos = float(
        np.dot(grad_rad_exact_mean, grad_exact) / (np.linalg.norm(grad_rad_exact_mean) * np.linalg.norm(grad_exact))
    )
    aniso_angle_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, aniso_cos)))))
    results["rademacher_directional_bias_exact_enum_d4"] = {
        "g_aniso": g_aniso.tolist(),
        "phi_exact": float(math.sqrt(1.0 + s_exact) - 1.0),
        "phi_rademacher_exact_expectation": float(np.mean(np.sqrt(1.0 + s_all) - 1.0)),
        "grad_exact": grad_exact.tolist(),
        "grad_rademacher_exact_expectation": grad_rad_exact_mean.tolist(),
        "directional_cos": aniso_cos,
        "directional_angle_deg": aniso_angle_deg,
    }

    # --- 2C: Strictly Convex-on-Cap EBI Squared Softplus Cap [softplus_beta(sqrt(s) - kappa)]^2 ---
    ebi_rows = []
    beta_ebi, kappa = 12.0, 1.0
    for r_true in (0.5, 0.8, 1.0, 1.2, 2.0):
        s_true = r_true ** 2
        r_t = torch.tensor(r_true, dtype=torch.float64)
        h_true = float(F.softplus(r_t - kappa, beta=beta_ebi).square().item())
        sp_true = float(F.softplus(r_t - kappa, beta=beta_ebi).item())
        sig_true = float(torch.sigmoid(beta_ebi * (r_t - kappa)).item())
        # grad_g h(||g||^2) = [2 * sp * sig / r] * g -> scalar multiplier = 2*sp*sig/r
        gmult_true = (2.0 * sp_true * sig_true) / r_true

        for k_probes in (1, 2, 4, 8, 32):
            # Large-d chi_k^2 / k distribution: s_hat = s_true * (chi_k^2 / k)
            chi_k = rng.chisquare(df=k_probes, size=100000) / float(k_probes)
            s_hat = s_true * chi_k
            r_hat = np.sqrt(s_hat + 1e-12)
            # stable softplus
            z_hat = beta_ebi * (r_hat - kappa)
            sp_hat = np.where(z_hat > 30.0, r_hat - kappa, np.log1p(np.exp(np.clip(z_hat, -60.0, 30.0))) / beta_ebi)
            sig_hat = 1.0 / (1.0 + np.exp(-np.clip(z_hat, -60.0, 60.0)))
            h_hat = sp_hat ** 2
            # Expected gradient projection along g: E[(2*sp_hat*sig_hat / r_hat) * (s_hat / s_true)]
            gmult_hat = np.mean((2.0 * sp_hat * sig_hat / r_hat) * chi_k)
            ebi_rows.append(
                {
                    "r_true": r_true,
                    "k_probes": k_probes,
                    "h_true": h_true,
                    "h_mc_mean": float(np.mean(h_hat)),
                    "h_bias_abs": float(np.mean(h_hat) - h_true),
                    "h_ratio": float(np.mean(h_hat) / max(h_true, 1e-12)),
                    "prob_exceed_kappa": float(np.mean(r_hat > kappa)),
                    "norm_est_mean_ratio": float(np.mean(r_hat) / r_true),
                    "grad_mult_true": gmult_true,
                    "grad_mult_mean": float(gmult_hat),
                    "grad_mult_ratio": float(gmult_hat / max(gmult_true, 1e-12)),
                }
            )
    results["convex_ebi_softplus_cap_jensen"] = ebi_rows

    # --- 2D: Separable vs Dense Rademacher Variance Theorem & Minimal Batch-Correlation Audit ---
    sep_rows = []
    C, H_im, W_im = 3, 32, 32
    dim_im = C * H_im * W_im
    n_sep = 20000
    # Build three canonical spatial gradient matrices G in R^{(C*H) x W}, normalized to ||G||_F = 1.0
    p_row = rng.standard_normal(C * H_im)
    q_col = rng.standard_normal(W_im)
    G_rank1 = np.outer(p_row, q_col)
    G_rank1 = G_rank1 / np.linalg.norm(G_rank1)

    U4 = rng.standard_normal((C * H_im, 4))
    V4 = rng.standard_normal((W_im, 4))
    G_rank4 = U4 @ V4.T
    G_rank4 = G_rank4 / np.linalg.norm(G_rank4)

    G_iso = rng.standard_normal((C * H_im, W_im))
    G_iso = G_iso / np.linalg.norm(G_iso)

    # Also a 3-way rank-1 tensor a_c * b_h * c_w for minimal's 3-way Kronecker probe
    a_c = np.array([1.0, -0.7, 0.5])
    b_h = rng.standard_normal(H_im)
    c_w = rng.standard_normal(W_im)
    G_3way = np.einsum("c,h,w->chw", a_c, b_h, c_w).reshape(C * H_im, W_im)
    G_3way = G_3way / np.linalg.norm(G_3way)

    for g_name, G_mat in (
        ("rank1_2way_stripe", G_rank1),
        ("rank1_3way_chw", G_3way),
        ("rank4_lowrank", G_rank4),
        ("fullrank_isotropic", G_iso),
    ):
        g_flat = G_mat.reshape(-1)
        # Renyi-2 effective rank of G_mat: ||G||_F^4 / ||G^T G||_F^2
        gtg_frob2 = float(np.sum((G_mat.T @ G_mat) ** 2))
        r_eff2 = 1.0 / gtg_frob2
        # Theoretical dense var = 2*(1 - sum g_i^4)
        theo_dense_var = float(2.0 * (1.0 - np.sum(g_flat ** 4)))
        # Exact closed-form 2-way separable variance for any G (||G||_F = 1):
        # E[(a^T G b)^4] = 3 E_b[||G b||_2^4] - 2 sum_i E_b[(G_{i,:} b)^4]
        # where E_b[||G b||_2^4] = ||G||_F^4 + 2||G^T G||_F^2 - 2 sum_j ||G_{:,j}||_2^4
        # and   E_b[(G_{i,:} b)^4] = 3 ||G_{i,:}||_2^4 - 2 sum_j G_{ij}^4.
        col_sq = np.sum(G_mat ** 2, axis=0)
        row_sq = np.sum(G_mat ** 2, axis=1)
        e_gb4 = 1.0 + 2.0 * gtg_frob2 - 2.0 * float(np.sum(col_sq ** 2))
        e_row4 = float(3.0 * np.sum(row_sq ** 2) - 2.0 * np.sum(G_mat ** 4))
        theo_sep2_exact = float(3.0 * e_gb4 - 2.0 * e_row4 - 1.0)
        theo_sep2_leading = float(2.0 + 6.0 / r_eff2)

        # 1) Dense Rademacher: (eps . g_flat)^2 (20k draws; low kurtosis => SE < 0.03)
        eps_d = rng.choice(np.array([-1.0, 1.0]), size=(n_sep, dim_im))
        z_dense = (eps_d @ g_flat) ** 2

        # 2) 2-way separable Rademacher (t2_secant / unified / unified_v11): (a^T G b)^2
        # Use 100k draws for heavy-tailed chi_1^2 * chi_1^2 variance accuracy
        n_sep_heavy = 100000
        s_ch = rng.choice(np.array([-1.0, 1.0]), size=(n_sep_heavy, C * H_im))
        s_w = rng.choice(np.array([-1.0, 1.0]), size=(n_sep_heavy, W_im))
        gb = s_w @ G_mat.T  # [n_sep_heavy, C*H]
        z_sep2 = np.sum(s_ch * gb, axis=1) ** 2

        # 3) 3-way separable Rademacher (minimal's C x H x W Kronecker product)
        G_t = G_mat.reshape(C, H_im, W_im)
        sc = rng.choice(np.array([-1.0, 1.0]), size=(n_sep_heavy, C))
        sh = rng.choice(np.array([-1.0, 1.0]), size=(n_sep_heavy, H_im))
        sw = rng.choice(np.array([-1.0, 1.0]), size=(n_sep_heavy, W_im))
        z_sep3 = np.einsum("nc,nh,nw,chw->n", sc, sh, sw, G_t) ** 2

        # Also evaluate concave BI (beta=1) & convex EBI (beta=12, kappa=1, r=1) under each probe
        r_test = 1.0
        sep_rows.append(
            {
                "gradient_structure": g_name,
                "renyi2_eff_rank": r_eff2,
                "dense_mean": float(np.mean(z_dense)),
                "dense_var": float(np.var(z_dense)),
                "dense_var_theory": theo_dense_var,
                "sep2_mean": float(np.mean(z_sep2)),
                "sep2_var": float(np.var(z_sep2)),
                "sep2_var_exact_theory": theo_sep2_exact,
                "sep2_var_leading_theory": theo_sep2_leading,
                "sep3_mean": float(np.mean(z_sep3)),
                "sep3_var": float(np.var(z_sep3)),
                "dense_bi_mean_at_r1": float(np.mean(np.sqrt(1.0 + z_dense) - 1.0)),
                "sep2_bi_mean_at_r1": float(np.mean(np.sqrt(1.0 + z_sep2) - 1.0)),
                "sep3_bi_mean_at_r1": float(np.mean(np.sqrt(1.0 + z_sep3) - 1.0)),
                "exact_bi_at_r1": float(math.sqrt(1.0 + r_test ** 2) - 1.0),
                "dense_norm_mean_at_r1": float(np.mean(np.sqrt(z_dense))),
                "sep2_norm_mean_at_r1": float(np.mean(np.sqrt(z_sep2))),
                "sep3_norm_mean_at_r1": float(np.mean(np.sqrt(z_sep3))),
            }
        )
    results["separable_vs_dense_variance"] = sep_rows

    # Check actual minimal _unit_probes batch correlation on 32x32x3
    x_dummy = torch.zeros(16, 3, 32, 32)
    u_min = minimal_unit_probes(x_dummy, phase=0.17, k_probes=1)[0].flatten(1).double()
    cos_mat = (u_min @ u_min.T).abs()
    off_diag = cos_mat[~torch.eye(16, dtype=torch.bool)]
    u_sep_t2 = _separable_image_unit_perturbation(x_dummy, phase=0.17, salt=3).flatten(1).double()
    cos_t2 = (u_sep_t2 @ u_sep_t2.T).abs()
    off_t2 = cos_t2[~torch.eye(16, dtype=torch.bool)]
    results["minimal_vs_t2_batch_probe_correlation"] = {
        "minimal_offdiag_abs_cos_min": float(off_diag.min().item()),
        "minimal_offdiag_abs_cos_max": float(off_diag.max().item()),
        "t2_separable_offdiag_abs_cos_mean": float(off_t2.mean().item()),
        "t2_separable_offdiag_abs_cos_max": float(off_t2.max().item()),
    }

    # --- 2E: Local k-NN Pooling Curvature-Bias vs Jensen-Bias Tradeoff (Quadratic Critic d=32) ---
    knn_tradeoff = []
    d_q = 32
    B_q = 128
    g_q = torch.Generator().manual_seed(77)
    x_q = torch.randn(B_q, d_q, generator=g_q)
    # Quadratic potential D(x) = 0.5 * c * ||x||^2 => ||grad_x D(x)||^2 = c^2 ||x||^2
    for c_curv in (0.15, 0.35, 0.70):
        sq_true = (c_curv ** 2) * x_q.square().sum(dim=1)
        phi_true_mean = float(_phi_bi(sq_true, 1.0).mean().item())
        ebi_true_mean = float(_phi_ebi(sq_true, 12.0, 1.0).mean().item())
        for k_pool in (1, 4, 8, 16, 32):
            bi_vals, ebi_vals, sq_bias_vals = [], [], []
            for trial in range(200):
                u_t, recon_t = _prepare_spatial_tangent_plan(
                    x_q,
                    phase=(trial * 0.6180339887) % 1.0,
                    salt=trial + 1,
                    pool_k=max(2, k_pool),
                )
                # Exact directional derivative for D(x) = 0.5 c ||x||^2 is c * (x . u)
                tau_sq = (c_curv * (x_q * u_t).sum(dim=1)).square()
                sq_hat = float(d_q) * tau_sq if k_pool == 1 else recon_t(tau_sq)
                bi_vals.append(float(_phi_bi(sq_hat, 1.0).mean().item()))
                ebi_vals.append(float(_phi_ebi(sq_hat, 12.0, 1.0).mean().item()))
                sq_bias_vals.append(float((sq_hat - sq_true).abs().mean().item() / sq_true.mean().item()))
            knn_tradeoff.append(
                {
                    "curvature_c": c_curv,
                    "pool_k": k_pool,
                    "bi_true": phi_true_mean,
                    "bi_est_mean": float(np.mean(bi_vals)),
                    "bi_rel_bias": float((np.mean(bi_vals) - phi_true_mean) / phi_true_mean),
                    "bi_rel_std": float(np.std(bi_vals) / phi_true_mean),
                    "ebi_true": ebi_true_mean,
                    "ebi_est_mean": float(np.mean(ebi_vals)),
                    "ebi_rel_bias": float((np.mean(ebi_vals) - ebi_true_mean) / max(ebi_true_mean, 1e-12)),
                    "pointwise_sq_rel_mae": float(np.mean(sq_bias_vals)),
                }
            )
    results["knn_pooling_tradeoff_d32"] = knn_tradeoff
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Kink Impulses, BF16 Micro-Secant Cancellation & Chord Failure
# ---------------------------------------------------------------------------


def exp3_kink_and_bf16_secant_audit(device: torch.device) -> Dict[str, Any]:
    """Audit ReLU/LeakyReLU kink impulses, DCGAN BF16 micro-secant underflow, and chord failures."""
    results: Dict[str, Any] = {}

    # --- 3A: Controlled Hyperplane Kink Crossing Impulse (O(1/delta) explosion) ---
    # 2-layer LeakyReLU(0.2) MLP in FP64 where neuron 0 has pre-activation z_0 = +eta = 1e-4
    # and perturbation u drives z_0(x + delta u) = eta - delta < 0 whenever delta > eta!
    d_k, H_k = 4, 8
    eta = 1e-4
    g_k = torch.Generator().manual_seed(303)
    W1_0 = torch.randn(H_k, d_k, generator=g_k, dtype=torch.float64) * 0.5
    W1_0[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    b1_0 = torch.randn(H_k, generator=g_k, dtype=torch.float64) * 0.5 + 1.0  # keep neurons 1..7 away from 0
    b1_0[0] = 0.0
    W2_0 = torch.randn(1, H_k, generator=g_k, dtype=torch.float64) * 0.5
    W2_0[0, 0] = 1.0

    x_k = torch.tensor([[eta, 0.2, -0.3, 0.4]], dtype=torch.float64)
    u_k = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)  # crosses z_0=0 at delta=eta

    def _run_kink_eval(mode: str, delta: float) -> Tuple[float, torch.Tensor]:
        W1 = W1_0.clone().requires_grad_(True)
        b1 = b1_0.clone().requires_grad_(True)
        W2 = W2_0.clone().requires_grad_(True)

        def fwd(z_in: torch.Tensor) -> torch.Tensor:
            return F.leaky_relu(z_in @ W1.T + b1.unsqueeze(0), 0.2) @ W2.T

        if mode == "fwad":
            with fwAD.dual_level():
                _, tau = fwAD.unpack_dual(fwd(fwAD.make_dual(x_k, u_k)))
            tau_s = tau.squeeze()
        elif mode == "ror":
            xr = x_k.clone().requires_grad_(True)
            gx = torch.autograd.grad(fwd(xr).sum(), xr, create_graph=True)[0]
            tau_s = (gx * u_k).sum()
        elif mode == "fwd_fd":
            tau_s = (fwd(x_k + delta * u_k) - fwd(x_k)).squeeze() / delta
        elif mode == "cen_fd":
            tau_s = (fwd(x_k + delta * u_k) - fwd(x_k - delta * u_k)).squeeze() / (2.0 * delta)
        else:
            raise ValueError(mode)
        (0.5 * tau_s.square()).backward()
        g_all = torch.cat([W1.grad.reshape(-1), b1.grad.reshape(-1), W2.grad.reshape(-1)])
        return float(tau_s.detach().item()), g_all

    tau_ror, g_ror = _run_kink_eval("ror", 0.0)
    tau_fw, g_fw = _run_kink_eval("fwad", 0.0)
    kink_rows = [
        {
            "method": "fwAD",
            "delta": 0.0,
            "crosses_kink": False,
            "tau": tau_fw,
            "tau_rel_err": abs(tau_fw - tau_ror) / max(abs(tau_ror), 1e-30),
            **_grad_metrics(g_fw, g_ror),
        }
    ]
    for delta in (1e-1, 2e-2, 1e-2, 1e-3, 5e-4, 2e-4, 5e-5, 1e-5):
        for mode in ("fwd_fd", "cen_fd"):
            tau_m, g_m = _run_kink_eval(mode, delta)
            gm = _grad_metrics(g_m, g_ror)
            kink_rows.append(
                {
                    "method": mode,
                    "delta": delta,
                    "crosses_kink": bool(delta > eta),
                    "tau": tau_m,
                    "tau_rel_err": abs(tau_m - tau_ror) / max(abs(tau_ror), 1e-30),
                    **gm,
                }
            )
    results["controlled_kink_impulse_fp64"] = {"eta": eta, "tau_exact": tau_ror, "rows": kink_rows}

    # --- 3B: Actual DCGANCritic BF16 Micro-Secant & Kink Audit on GPU ---
    dcgan_secant_rows = []
    for res in (32, 64, 128, 256):
        B = 16
        dim = 3 * res * res
        D = DCGANCritic(res, 64, 512).to(device)
        deterministic_orthogonal_(D, seed=0)
        # Initialize BN running stats on one forward pass in train mode, then freeze to eval()
        # so ALL methods differentiate the exact same operator!
        g_cuda = torch.Generator(device=device).manual_seed(400 + res)
        x_in = torch.empty(B, 3, res, res, device=device, dtype=torch.float32).uniform_(-1.0, 1.0, generator=g_cuda)
        with torch.no_grad():
            D.train()
            _ = D(x_in)
            D.eval()

        u_sep = _separable_image_unit_perturbation(x_in, phase=0.23, salt=3)
        params = list(D.parameters())

        # Ground truth: exact FP32 fwAD along u_sep
        _zero_grads(D)
        with fwAD.dual_level():
            y_d = D(fwAD.make_dual(x_in, u_sep))
            y_primal, tau_exact = fwAD.unpack_dual(y_d)
        tau_exact_v = tau_exact.reshape(B)
        sq_exact = float(dim) * tau_exact_v.square()
        # Use squared penalty 0.5 * mean(sq) so gradient scaling is completely clean & non-zero
        (0.5 * sq_exact.mean()).backward()
        g_exact = _flatten_grads(params)
        tau_exact_det = tau_exact_v.detach().double()

        # Verify RoR matches exact FP32 fwAD along u_sep on DCGANCritic too
        _zero_grads(D)
        x_req = x_in.detach().clone().requires_grad_(True)
        gx_ror = torch.autograd.grad(D(x_req).sum(), x_req, create_graph=True)[0]
        tau_ror_v = (gx_ror * u_sep).flatten(1).sum(dim=1)
        (0.5 * (float(dim) * tau_ror_v.square()).mean()).backward()
        g_ror_u = _flatten_grads(params)
        ror_vs_fwad = _grad_metrics(g_ror_u, g_exact)

        # Helper to count LeakyReLU sign flips between x_a and x_b
        def _count_leakyrelu_flips(x_a: torch.Tensor, x_b: torch.Tensor, use_bf16: bool) -> Tuple[int, int]:
            signs_a: List[torch.Tensor] = []
            signs_b: List[torch.Tensor] = []
            hooks = []
            target_list = signs_a

            def _pre_hook(_mod, inp):
                target_list.append((inp[0].detach() > 0).bool())

            for mod in D.modules():
                if isinstance(mod, nn.LeakyReLU):
                    hooks.append(mod.register_forward_pre_hook(_pre_hook))
            try:
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
                with torch.no_grad(), ctx:
                    _ = D(x_a)
                    target_list = signs_b
                    _ = D(x_b)
            finally:
                for h in hooks:
                    h.remove()
            flips = sum(int((a != b).sum().item()) for a, b in zip(signs_a, signs_b))
            total = sum(int(a.numel()) for a in signs_a)
            return flips, total

        # 1) unified_v11 BF16 fwAD with _bn_dual_bf16_hook
        _zero_grads(D)
        bn_hooks = [m.register_forward_hook(_bn_dual_bf16_hook) for m in D.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with fwAD.dual_level():
                    yd_bf = D(fwAD.make_dual(x_in.to(torch.bfloat16), u_sep.to(torch.bfloat16)))
                    _, tau_bf = fwAD.unpack_dual(yd_bf)
            tau_bf_v = tau_bf.float().reshape(B)
            (0.5 * (float(dim) * tau_bf_v.square()).mean()).backward()
            g_fw_bf = _flatten_grads(params)
        finally:
            for h in bn_hooks:
                h.remove()
        m_fw_bf = _grad_metrics(g_fw_bf, g_exact)
        dcgan_secant_rows.append(
            {
                "res": res,
                "dim": dim,
                "method": "unified_v11_fwAD_bf16",
                "delta": 0.0,
                "per_pixel_step": 0.0,
                "surviving_pixel_frac": 1.0,
                "relu_flip_count": 0,
                "relu_flip_frac": 0.0,
                "zero_diff_frac": 0.0,
                "tau_rel_l2": float((tau_bf_v.detach().double() - tau_exact_det).norm().item() / max(float(tau_exact_det.norm().item()), 1e-30)),
                "ror_vs_fwad_fp32_cos": ror_vs_fwad["cos"],
                "ror_vs_fwad_fp32_rel_l2": ror_vs_fwad["rel_l2"],
                **m_fw_bf,
            }
        )

        # 2) Micro-secants: t2_secant (delta=0.02) and minimal (delta=1e-5) in BOTH FP32 and BF16 autocast
        for m_label, delta, use_bf16 in (
            ("t2_micro_secant_fp32", 0.02, False),
            ("t2_micro_secant_bf16", 0.02, True),
            ("minimal_micro_secant_fp32", 1e-5, False),
            ("minimal_micro_secant_bf16", 1e-5, True),
        ):
            x_probe = (x_in + delta * u_sep).detach()
            if use_bf16:
                surv = float((x_probe.to(torch.bfloat16) != x_in.to(torch.bfloat16)).float().mean().item())
            else:
                surv = float((x_probe != x_in).float().mean().item())
            flips, total_act = _count_leakyrelu_flips(x_in, x_probe, use_bf16=use_bf16)

            _zero_grads(D)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
            with ctx:
                d_base = D(x_in).float().reshape(B)
                d_probe = D(x_probe).float().reshape(B)
            # Match t2_secant denominator (uses FP32 x_probe - x_in norm^2 + eps^2)
            step_norm = (x_probe - x_in).flatten(1).float().square().sum(dim=1).add(1e-10).sqrt()
            # Match minimal denominator (uses scalar delta)
            denom = step_norm if "t2_" in m_label else torch.full_like(step_norm, delta)
            diff_d = d_probe - d_base
            zero_diff_frac = float((diff_d.detach() == 0).float().mean().item())
            tau_hat = diff_d / denom
            (0.5 * (float(dim) * tau_hat.square()).mean()).backward()
            g_sec = _flatten_grads(params)
            gm = _grad_metrics(g_sec, g_exact)
            dcgan_secant_rows.append(
                {
                    "res": res,
                    "dim": dim,
                    "method": m_label,
                    "delta": delta,
                    "per_pixel_step": float(delta / math.sqrt(float(dim))),
                    "surviving_pixel_frac": surv,
                    "relu_flip_count": flips,
                    "relu_flip_frac": float(flips / max(1, total_act)),
                    "zero_diff_frac": zero_diff_frac,
                    "tau_rel_l2": float((tau_hat.detach().double() - tau_exact_det).norm().item() / max(float(tau_exact_det.norm().item()), 1e-30)),
                    "ror_vs_fwad_fp32_cos": ror_vs_fwad["cos"],
                    "ror_vs_fwad_fp32_rel_l2": ror_vs_fwad["rel_l2"],
                    **gm,
                }
            )
    results["dcgan_bf16_micro_secant_audit"] = dcgan_secant_rows

    # --- 3C: Transport-Chord & Cyclic-Roll Failure Modes ---
    # 1) Symmetric-flank cancellation: D(x) = -alpha * ||x||^2 on opposite pairs a = -r u, b = +r u
    sym_rows = []
    for alpha in (1.0, 10.0, 100.0):
        a = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
        b = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        u_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        D_rad = lambda z: -alpha * z.square().sum(dim=1)
        s_chord = float((D_rad(b) - D_rad(a)).abs().item() / (b - a).norm().item())
        with fwAD.dual_level():
            _, j_a = fwAD.unpack_dual(D_rad(fwAD.make_dual(a, u_dir)))
        true_slope = float(j_a.abs().item())
        sym_rows.append({"alpha": alpha, "true_grad_norm": true_slope, "s_chord": s_chord})

    # 2) High-dimensional unscaled chord attenuation on linear D(x) = w^T x, ||w||_2 = 2.0
    chord_atten_rows = []
    rng_c = np.random.default_rng(123)
    for dim in (2, 8, 32, 170, 3072, 196608):
        w = rng_c.standard_normal(dim)
        w = w * (2.0 / np.linalg.norm(w))
        xr = rng_c.standard_normal((16, dim))
        xf = rng_c.standard_normal((16, dim))
        dr = xr @ w
        df = xf @ w
        dist_rf = np.linalg.norm(xr - xf, axis=1)
        s_chord_vals = np.abs(dr - df) / dist_rf
        xf_roll = np.roll(xf, 1, axis=0)
        s_roll_vals = np.abs(np.roll(df, 1) - df) / np.linalg.norm(xf_roll - xf, axis=1)
        chord_atten_rows.append(
            {
                "dim": dim,
                "true_grad_norm": 2.0,
                "s_chord_mean": float(np.mean(s_chord_vals)),
                "s_roll_mean": float(np.mean(s_roll_vals)),
                "theoretical_2_over_sqrt_d_times_sqrt2pi": float(2.0 * math.sqrt(2.0 / math.pi) / math.sqrt(float(dim))),
                "attenuation_factor": float(2.0 / max(float(np.mean(s_chord_vals)), 1e-12)),
            }
        )

    # 3) Oscillatory mean-value averaging cancellation: D(x) = sin(20 pi x) on [0, 1]
    omega = 20.0 * math.pi
    osc_chord = abs(math.sin(omega * 1.0) - math.sin(omega * 0.0)) / 1.0
    results["chord_and_roll_failure_modes"] = {
        "symmetric_flank_cancellation": sym_rows,
        "highd_chord_attenuation": chord_atten_rows,
        "oscillatory_mean_value_cancellation": {
            "true_lipschitz_const": omega,
            "macro_chord_slope_0_to_1": osc_chord,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Experiment 4: RTX A6000 Computational Scaling Regimes & Architecture Matrix
# ---------------------------------------------------------------------------


def _bench_cuda_op(
    step_fn: Callable[[], None],
    device: torch.device,
    nvml: NVMLMonitor,
    warmup: int = 8,
    iters: int = 30,
) -> Dict[str, Any]:
    """Measure synchronized median/P95 latency, peak VRAM, and NVML mJ/iter."""
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    times_ms: List[float] = []
    e0 = nvml.total_energy_mj()
    t_wall0 = time.perf_counter()
    for _ in range(iters):
        t0 = time.perf_counter()
        step_fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_s = time.perf_counter() - t_wall0
    e1 = nvml.total_energy_mj()

    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    peak_res_mib = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
    times_sorted = sorted(times_ms)
    med_ms = statistics.median(times_sorted)
    p95_ms = times_sorted[min(len(times_sorted) - 1, int(0.95 * len(times_sorted)))]
    energy_mj_per_iter = float(e1 - e0) / float(iters) if (e0 is not None and e1 is not None and e1 >= e0) else None
    avg_power_w = (float(e1 - e0) / 1000.0) / max(wall_s, 1e-9) if (e0 is not None and e1 is not None and e1 >= e0) else None
    return {
        "median_ms": med_ms,
        "p95_ms": p95_ms,
        "peak_alloc_mib": peak_alloc_mib,
        "peak_reserved_mib": peak_res_mib,
        "energy_mj_per_iter": energy_mj_per_iter,
        "avg_power_w": avg_power_w,
    }


class _ResNetGNCritic(nn.Module):
    """64x64 ResNet critic with GroupNorm and residual skip connections."""

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, base_ch, 3, stride=2, padding=1)
        self.b1_c1 = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.b1_n1 = nn.GroupNorm(8, base_ch)
        self.b1_c2 = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.b1_n2 = nn.GroupNorm(8, base_ch)
        self.down = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.b2_c1 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.b2_n1 = nn.GroupNorm(8, base_ch * 2)
        self.b2_c2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.b2_n2 = nn.GroupNorm(8, base_ch * 2)
        self.head = nn.Linear(base_ch * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.stem(x), 0.2)
        h = h + self.b1_c2(F.leaky_relu(self.b1_n1(self.b1_c1(h)), 0.2))
        h = F.leaky_relu(self.b1_n2(h), 0.2)
        h = F.leaky_relu(self.down(h), 0.2)
        h = h + self.b2_c2(F.leaky_relu(self.b2_n1(self.b2_c1(h)), 0.2))
        h = F.leaky_relu(self.b2_n2(h), 0.2)
        return self.head(h.mean(dim=(2, 3)))


class _AttentionCritic(nn.Module):
    """Transformer-style critic using LayerNorm + F.scaled_dot_product_attention."""

    def __init__(self, d_model: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return self.head(x.mean(dim=1))


def _bench_mlp_dim_row(dim: int, B_mlp: int, device: torch.device, nvml: NVMLMonitor) -> Dict[str, Any]:
    """Benchmark all MLP derivative methods at one input dimension on CUDA."""
    model = _make_mlp_critic(dim, 256, 4, activation="leaky_relu", fourier=0, dtype=torch.float32, device=device)
    g_c = torch.Generator(device=device).manual_seed(500 + dim)
    x_b = torch.randn(B_mlp, dim, generator=g_c, device=device, dtype=torch.float32)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    u1 = torch.where(torch.randn(B_mlp, dim, generator=g_c, device=device) >= 0, 1.0, -1.0) / math.sqrt(float(dim))
    u4 = torch.where(torch.randn(4 * B_mlp, dim, generator=g_c, device=device) >= 0, 1.0, -1.0) / math.sqrt(float(dim))
    u8 = torch.where(torch.randn(8 * B_mlp, dim, generator=g_c, device=device) >= 0, 1.0, -1.0) / math.sqrt(float(dim))
    x4 = x_b.repeat(4, 1)
    x8 = x_b.repeat(8, 1)

    def step_ror() -> None:
        _zero_grads(model)
        xr = x_b.detach().clone().requires_grad_(True)
        y = model(xr)
        gx = torch.autograd.grad(y.sum(), xr, create_graph=True)[0]
        sq = gx.square().sum(dim=1)
        _phi_ebi(sq, 12.0, 1.0).mean().backward()

    def step_fwad_exact() -> None:
        _zero_grads(model)
        chunk = min(dim, 16)
        sq_acc = x_b.new_zeros((B_mlp,))
        for start in range(0, dim, chunk):
            end = min(dim, start + chunk)
            m_c = end - start
            x_rep = x_b.repeat(m_c, 1)
            u_rep = eye[start:end].repeat_interleave(B_mlp, dim=0)
            with fwAD.dual_level():
                _, jvp = fwAD.unpack_dual(model(fwAD.make_dual(x_rep, u_rep)))
            sq_acc = sq_acc + jvp.reshape(m_c, B_mlp).square().sum(dim=0)
        _phi_ebi(sq_acc, 12.0, 1.0).mean().backward()

    def step_fwad_1() -> None:
        _zero_grads(model)
        with fwAD.dual_level():
            _, jvp = fwAD.unpack_dual(model(fwAD.make_dual(x_b, u1)))
        sq = float(dim) * jvp.reshape(B_mlp).square()
        _phi_ebi(sq, 12.0, 1.0).mean().backward()

    def step_fwad_4() -> None:
        _zero_grads(model)
        with fwAD.dual_level():
            _, jvp = fwAD.unpack_dual(model(fwAD.make_dual(x4, u4)))
        sq = (float(dim) / 4.0) * jvp.reshape(4, B_mlp).square().sum(dim=0)
        _phi_ebi(sq, 12.0, 1.0).mean().backward()

    def step_fwad_8() -> None:
        _zero_grads(model)
        with fwAD.dual_level():
            _, jvp = fwAD.unpack_dual(model(fwAD.make_dual(x8, u8)))
        sq = (float(dim) / 8.0) * jvp.reshape(8, B_mlp).square().sum(dim=0)
        _phi_ebi(sq, 12.0, 1.0).mean().backward()

    def step_secant_reuse() -> None:
        _zero_grads(model)
        d0 = model(x_b).reshape(B_mlp)
        d1 = model((x_b + 0.02 * u1).detach()).reshape(B_mlp)
        sq = float(dim) * ((d1 - d0) / 0.02).square()
        (d0.mean() + _phi_ebi(sq, 12.0, 1.0).mean()).backward()

    iters_exact = 25 if dim <= 64 else (12 if dim <= 256 else 6)
    methods = [
        ("ror_exact", step_ror, 30),
        ("fwad_exact_basis", step_fwad_exact, iters_exact),
        ("fwad_1probe", step_fwad_1, 30),
        ("fwad_4probe", step_fwad_4, 30),
        ("fwad_8probe", step_fwad_8, 30),
        ("micro_secant_with_primal", step_secant_reuse, 30),
    ]
    row: Dict[str, Any] = {"dim": dim, "batch_size": B_mlp}
    for name, fn, n_it in methods:
        row[name] = _bench_cuda_op(fn, device, nvml, warmup=6, iters=n_it)
    row["exact_fwad_over_ror_latency"] = row["fwad_exact_basis"]["median_ms"] / row["ror_exact"]["median_ms"]
    row["exact_fwad_over_ror_vram"] = row["fwad_exact_basis"]["peak_alloc_mib"] / max(row["ror_exact"]["peak_alloc_mib"], 1e-6)
    row["fwad1_over_ror_latency"] = row["fwad_1probe"]["median_ms"] / row["ror_exact"]["median_ms"]
    return row


def _bench_conv_res_row(res: int, prec: str, B_img: int, device: torch.device, nvml: NVMLMonitor) -> Dict[str, Any]:
    """Benchmark all DCGANCritic penalty methods at one spatial resolution and precision on CUDA."""
    dim_im = 3 * res * res
    use_bf16 = prec == "bf16_channels_last"
    D = DCGANCritic(res, 64, 512).to(device)
    deterministic_orthogonal_(D, seed=0)
    if use_bf16:
        D = D.to(memory_format=torch.channels_last)
    g_c = torch.Generator(device=device).manual_seed(600 + res)
    x_r = torch.randn(B_img, 3, res, res, generator=g_c, device=device, dtype=torch.float32)
    x_f = torch.randn(B_img, 3, res, res, generator=g_c, device=device, dtype=torch.float32)
    if use_bf16:
        x_r = x_r.to(memory_format=torch.channels_last)
        x_f = x_f.to(memory_format=torch.channels_last)
    with torch.no_grad():
        D.train()
        _ = D(x_r)
    u_r = _universal_image_unit_tangent(x_r, phase=0.17, salt=3)
    u_f = _universal_image_unit_tangent(x_f, phase=0.17, salt=11)
    x_mix = torch.cat([x_r[: B_img // 2], x_f[: B_img // 2]], dim=0)
    u_mix = _universal_image_unit_tangent(x_mix, phase=0.17, salt=3)
    if use_bf16:
        u_r = u_r.to(memory_format=torch.channels_last)
        u_f = u_f.to(memory_format=torch.channels_last)
        x_mix = x_mix.to(memory_format=torch.channels_last)
        u_mix = u_mix.to(memory_format=torch.channels_last)

    def _set_bn_mode(train_flag: bool) -> None:
        for m in D.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.training = train_flag

    def step_ror_bn_train() -> None:
        _zero_grads(D)
        _set_bn_mode(True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
        xr = x_r.detach().clone().requires_grad_(True)
        xf = x_f.detach().clone().requires_grad_(True)
        with ctx:
            dr, df = D(xr), D(xf)
        gr, gf = torch.autograd.grad((dr.sum(), df.sum()), (xr, xf), create_graph=True)
        sq_r = gr.flatten(1).float().square().sum(dim=1)
        sq_f = gf.flatten(1).float().square().sum(dim=1)
        (_phi_ebi(sq_r, 12.0, 1.0).mean() + _phi_ebi(sq_f, 12.0, 1.0).mean()).backward()

    def step_ror_bn_eval() -> None:
        _zero_grads(D)
        _set_bn_mode(False)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
        xr = x_r.detach().clone().requires_grad_(True)
        xf = x_f.detach().clone().requires_grad_(True)
        with ctx:
            dr, df = D(xr), D(xf)
        gr, gf = torch.autograd.grad((dr.sum(), df.sum()), (xr, xf), create_graph=True)
        sq_r = gr.flatten(1).float().square().sum(dim=1)
        sq_f = gf.flatten(1).float().square().sum(dim=1)
        (_phi_ebi(sq_r, 12.0, 1.0).mean() + _phi_ebi(sq_f, 12.0, 1.0).mean()).backward()

    def step_fwad_1probe_interleaved() -> None:
        _zero_grads(D)
        _set_bn_mode(False)
        hooks = [m.register_forward_hook(_bn_dual_bf16_hook) for m in D.modules() if use_bf16 and isinstance(m, nn.modules.batchnorm._BatchNorm)]
        try:
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
            xin = x_mix.to(torch.bfloat16) if use_bf16 else x_mix
            uin = u_mix.to(torch.bfloat16) if use_bf16 else u_mix
            with ctx:
                with fwAD.dual_level():
                    _, jvp = fwAD.unpack_dual(D(fwAD.make_dual(xin, uin)))
            sq = float(dim_im) * jvp.float().reshape(B_img).square()
            (_phi_ebi(sq[: B_img // 2], 12.0, 1.0).mean() + _phi_ebi(sq[B_img // 2 :], 12.0, 1.0).mean()).backward()
        finally:
            for h in hooks:
                h.remove()

    def step_fwad_1probe_both() -> None:
        _zero_grads(D)
        _set_bn_mode(False)
        hooks = [m.register_forward_hook(_bn_dual_bf16_hook) for m in D.modules() if use_bf16 and isinstance(m, nn.modules.batchnorm._BatchNorm)]
        try:
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
            xri, uri = (x_r.to(torch.bfloat16), u_r.to(torch.bfloat16)) if use_bf16 else (x_r, u_r)
            xfi, ufi = (x_f.to(torch.bfloat16), u_f.to(torch.bfloat16)) if use_bf16 else (x_f, u_f)
            with ctx:
                with fwAD.dual_level():
                    _, jr = fwAD.unpack_dual(D(fwAD.make_dual(xri, uri)))
                    _, jf = fwAD.unpack_dual(D(fwAD.make_dual(xfi, ufi)))
            sq_r = float(dim_im) * jr.float().reshape(B_img).square()
            sq_f = float(dim_im) * jf.float().reshape(B_img).square()
            (_phi_ebi(sq_r, 12.0, 1.0).mean() + _phi_ebi(sq_f, 12.0, 1.0).mean()).backward()
        finally:
            for h in hooks:
                h.remove()

    def step_fwad_4probe_interleaved() -> None:
        _zero_grads(D)
        _set_bn_mode(False)
        hooks = [m.register_forward_hook(_bn_dual_bf16_hook) for m in D.modules() if use_bf16 and isinstance(m, nn.modules.batchnorm._BatchNorm)]
        try:
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
            xin = x_mix.to(torch.bfloat16) if use_bf16 else x_mix
            uin = u_mix.to(torch.bfloat16) if use_bf16 else u_mix
            sq_acc = x_r.new_zeros((B_img,), dtype=torch.float32)
            for _ in range(4):
                with ctx:
                    with fwAD.dual_level():
                        _, jvp = fwAD.unpack_dual(D(fwAD.make_dual(xin, uin)))
                sq_acc = sq_acc + (float(dim_im) / 4.0) * jvp.float().reshape(B_img).square()
            (_phi_ebi(sq_acc[: B_img // 2], 12.0, 1.0).mean() + _phi_ebi(sq_acc[B_img // 2 :], 12.0, 1.0).mean()).backward()
        finally:
            for h in hooks:
                h.remove()

    _set_bn_mode(True)
    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)):
        d_base_cached = D(x_mix).float().reshape(B_img).detach()

    def step_micro_secant_reuse() -> None:
        _zero_grads(D)
        _set_bn_mode(True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
        x_p = (x_mix + 0.02 * u_mix).detach()
        with ctx:
            d_probe = D(x_p).float().reshape(B_img)
        sq = float(dim_im) * ((d_probe - d_base_cached) / 0.02).square()
        (_phi_ebi(sq[: B_img // 2], 12.0, 1.0).mean() + _phi_ebi(sq[B_img // 2 :], 12.0, 1.0).mean()).backward()

    n_it = 25 if res <= 128 else 15
    row_c: Dict[str, Any] = {"res": res, "dim": dim_im, "precision": prec, "batch_size": B_img}
    for nm, fn in (
        ("ror_bn_train", step_ror_bn_train),
        ("ror_bn_eval", step_ror_bn_eval),
        ("fwad_1probe_interleaved", step_fwad_1probe_interleaved),
        ("fwad_1probe_both", step_fwad_1probe_both),
        ("fwad_4probe_interleaved", step_fwad_4probe_interleaved),
        ("micro_secant_1probe", step_micro_secant_reuse),
    ):
        row_c[nm] = _bench_cuda_op(fn, device, nvml, warmup=5, iters=n_it)
    row_c["speedup_fwad_interleaved_vs_ror_train"] = row_c["ror_bn_train"]["median_ms"] / row_c["fwad_1probe_interleaved"]["median_ms"]
    row_c["speedup_fwad_both_vs_ror_eval"] = row_c["ror_bn_eval"]["median_ms"] / row_c["fwad_1probe_both"]["median_ms"]
    row_c["speedup_secant_vs_ror_train"] = row_c["ror_bn_train"]["median_ms"] / row_c["micro_secant_1probe"]["median_ms"]
    row_c["vram_ratio_fwad_both_vs_ror_eval"] = row_c["fwad_1probe_both"]["peak_alloc_mib"] / max(row_c["ror_bn_eval"]["peak_alloc_mib"], 1e-6)
    return row_c


def exp4_scaling_regimes_gpu(device: torch.device, nvml: NVMLMonitor) -> Dict[str, Any]:
    """Run matched RTX A6000 scaling benchmarks across MLP dims, Conv2d sizes, and op types."""
    results: Dict[str, Any] = {}

    # --- 4A: MLP Dimension Scaling (d in {2, 4, 8, 16, 32, 64, 128, 256, 1024}, B=256, FP32) ---
    mlp_rows = []
    for dim in (2, 4, 8, 16, 32, 64, 128, 256, 1024):
        mlp_rows.append(_bench_mlp_dim_row(dim, 256, device, nvml))
        torch.cuda.empty_cache()
    results["mlp_dim_scaling_fp32"] = mlp_rows

    # --- 4B: Conv2d Spatial Resolution Scaling (32^2, 64^2, 128^2, 256^2; B=16; FP32 & BF16) ---
    conv_rows = []
    for res in (32, 64, 128, 256):
        for prec in ("fp32", "bf16_channels_last"):
            conv_rows.append(_bench_conv_res_row(res, prec, 16, device, nvml))
            torch.cuda.empty_cache()
    results["conv2d_resolution_scaling"] = conv_rows

    # --- 4C: Architectural Compatibility & Overhead Matrix ---
    arch_rows = []

    def _probe_arch(
        name: str,
        model: nn.Module,
        x_in: torch.Tensor,
        use_bf16: bool,
        sdpa_math_only: bool = False,
        use_bn_hook: bool = False,
    ) -> Dict[str, Any]:
        u_in = torch.randn_like(x_in)
        u_in = u_in / u_in.flatten(1).norm(dim=1).clamp_min(1e-8).reshape((-1,) + (1,) * (x_in.ndim - 1))
        entry: Dict[str, Any] = {"architecture": name, "precision": "bf16" if use_bf16 else "fp32"}

        def _make_sdpa_ctx():
            if sdpa_math_only:
                return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
            from contextlib import nullcontext
            return nullcontext()

        for mode in ("ror_exact", "fwad_1probe", "micro_secant"):
            try:
                def _run_once() -> None:
                    _zero_grads(model)
                    hooks = (
                        [m.register_forward_hook(_bn_dual_bf16_hook) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
                        if (mode == "fwad_1probe" and use_bn_hook and use_bf16)
                        else []
                    )
                    try:
                        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
                        with _make_sdpa_ctx(), amp_ctx:
                            if mode == "ror_exact":
                                xr = x_in.detach().clone().requires_grad_(True)
                                y = model(xr)
                                g = torch.autograd.grad(y.sum(), xr, create_graph=True)[0]
                                loss = g.flatten(1).float().square().sum(dim=1).mean()
                            elif mode == "fwad_1probe":
                                xin = x_in.to(torch.bfloat16) if (use_bn_hook and use_bf16) else x_in
                                uin = u_in.to(torch.bfloat16) if (use_bn_hook and use_bf16) else u_in
                                with fwAD.dual_level():
                                    _, jvp = fwAD.unpack_dual(model(fwAD.make_dual(xin, uin)))
                                if jvp is None:
                                    raise NotImplementedError("dual tangent dropped")
                                loss = jvp.float().square().mean()
                            else:
                                y0 = model(x_in).float()
                                y1 = model((x_in + 0.02 * u_in).detach()).float()
                                loss = ((y1 - y0) / 0.02).square().mean()
                        loss.backward()
                    finally:
                        for h in hooks:
                            h.remove()

                # First test once to check compatibility
                _run_once()
                torch.cuda.synchronize(device)
                bench = _bench_cuda_op(_run_once, device, nvml, warmup=4, iters=15)
                entry[mode] = {"status": "OK", "median_ms": bench["median_ms"], "peak_alloc_mib": bench["peak_alloc_mib"]}
            except Exception as ex:
                entry[mode] = {"status": "FAIL", "error": f"{type(ex).__name__}: {str(ex)[:110]}"}
        return entry

    # 1. DCGAN 64x64 BN train & eval (with and without _bn_dual_bf16_hook)
    x64 = torch.randn(16, 3, 64, 64, device=device)
    d_bn = DCGANCritic(64, 64, 512).to(device)
    deterministic_orthogonal_(d_bn, seed=0)
    with torch.no_grad():
        d_bn.train()
        _ = d_bn(x64)
    d_bn.train()
    arch_rows.append(_probe_arch("DCGAN_64_BN_train_no_hook", d_bn, x64, use_bf16=False, use_bn_hook=False))
    arch_rows.append(_probe_arch("DCGAN_64_BN_train_no_hook", d_bn, x64, use_bf16=True, use_bn_hook=False))
    d_bn.eval()
    arch_rows.append(_probe_arch("DCGAN_64_BN_eval_no_hook", d_bn, x64, use_bf16=True, use_bn_hook=False))
    arch_rows.append(_probe_arch("DCGAN_64_BN_eval_with_hook", d_bn, x64, use_bf16=True, use_bn_hook=True))
    del d_bn

    # 2. ResNet + GroupNorm (64x64)
    d_res = _ResNetGNCritic(64).to(device)
    arch_rows.append(_probe_arch("ResNet_64_Skip_GroupNorm", d_res, x64, use_bf16=False))
    arch_rows.append(_probe_arch("ResNet_64_Skip_GroupNorm", d_res, x64, use_bf16=True))
    del d_res

    # 3. Attention (SDPA default Flash/MemEff vs explicit MATH backend)
    x_seq = torch.randn(16, 64, 128, device=device)
    d_attn = _AttentionCritic(128, 4).to(device)
    arch_rows.append(_probe_arch("Attention_SDPA_default", d_attn, x_seq, use_bf16=False, sdpa_math_only=False))
    arch_rows.append(_probe_arch("Attention_SDPA_default", d_attn, x_seq, use_bf16=True, sdpa_math_only=False))
    arch_rows.append(_probe_arch("Attention_SDPA_math", d_attn, x_seq, use_bf16=False, sdpa_math_only=True))
    arch_rows.append(_probe_arch("Attention_SDPA_math", d_attn, x_seq, use_bf16=True, sdpa_math_only=True))
    del d_attn

    # 4. EmulatedLinear (MXFP8 & NVFP4)
    x_vec = torch.randn(64, 32, device=device)
    for fmt in ("mxfp8", "nvfp4"):
        base_mlp = nn.Sequential(nn.Linear(32, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1)).to(device)
        deterministic_orthogonal_(base_mlp, seed=0)
        base_mlp[0] = EmulatedLinear.from_linear(base_mlp[0], fmt, seed=0)
        base_mlp[2] = EmulatedLinear.from_linear(base_mlp[2], fmt, seed=1)
        arch_rows.append(_probe_arch(f"EmulatedLinear_{fmt}", base_mlp, x_vec, use_bf16=False))
        del base_mlp

    torch.cuda.empty_cache()
    results["architecture_compatibility_matrix"] = arch_rows
    return results


# ---------------------------------------------------------------------------
# Compact Summary Printer & CLI
# ---------------------------------------------------------------------------


def _print_summary(all_results: Dict[str, Any]) -> None:
    print("=" * 80)
    print("P6 LAW 2 AUTODIFF, BIAS-VARIANCE, KINK & SCALING AUDIT SUMMARY")
    print("=" * 80)

    print("\n[1A] Analytical Mixed 2nd-Order Derivative Identity (FP64, fwAD vs Analytical vs RoR):")
    for r in all_results["exp1"]["analytical_mixed_hessian_fp64"]:
        print(
            f"  {r['activation']:5s} {r['outer_penalty']:3s} | tau_err={r['tau_max_abs_err']:.2e} "
            f"| ||dL/db(sigma'')||={r['sigma2_bias_grad_norm']:.4f} "
            f"| fwAD-vs-Anal rel_l2={r['fwad_vs_analytical_rel_l2']:.2e} cos={r['fwad_vs_analytical_cos']:.12f} "
            f"| fwAD-vs-RoR rel_l2={r['fwad_vs_ror_rel_l2']:.2e}"
        )

    print("\n[1B] Full ||grad_x D||^2 Canonical fwAD vs RoR (FP64 & FP32):")
    for r in all_results["exp1"]["canonical_fwad_vs_ror"]:
        if r["dim"] == 4:
            print(
                f"  {r['activation']:10s} f={r['fourier']} d={r['dim']} {r['dtype']:4s} "
                f"| ||H_xx probe||={r['hess_xx_probe_norm']:.3e} ||H_tx g||={r['mixed_grad_norm']:.4f} "
                f"| rel_l2={r['rel_l2']:.2e} cos={r['cos']:.10f}"
            )

    print("\n[1C] Householder Frame & Spatial/Hutchinson Path Audit:")
    for r in all_results["exp1"]["householder_frame_geometry"]:
        print(
            f"  Householder d={r['dim']}: orth_err={r['orth_max_err']:.2e}, det={r['det']:+.6f}, "
            f"unit_eigs={r['unit_eigenvalues']}/{r['dim']} (fixes (d-2)-subspace)"
        )
    for r in all_results["exp1"]["codebase_spatial_and_hutchinson_audit"]:
        extra = ""
        if r["unpooled_1probe_grad_cos"] is not None:
            extra = f" | unpooled_1p_cos={r['unpooled_1probe_grad_cos']:.4f} | k_eff={r['knn32_k_eff_mean']:.2f}/32"
        print(
            f"  d={r['dim']:3d} {r['regime']:25s} ({r['critic_kind']:13s}) "
            f"| grad_cos={r['grad_cos_vs_ror']:.4f} rel_l2={r['grad_rel_l2_vs_ror']:.4f}{extra}"
        )

    print("\n[2A] Concave Born-Infeld Jensen Bias (d=170, beta=1.0 & 8.0):")
    for r in all_results["exp2"]["concave_born_infeld_jensen"]:
        if r["dim"] == 170 and r["r_true"] in (1.0, 2.0, 5.0):
            print(
                f"  beta={r['beta']:.0f} r={r['r_true']:.1f} K={r['k_probes']:2d} "
                f"| E[s_hat]/s={r['s_hat_mean_ratio']:.4f} "
                f"| loss_bias={r['phi_rel_jensen_bias']:+.4f} (chi1:{r['phi_chi1_limit_rel_bias']:+.4f}) "
                f"| grad_bias={r['grad_scale_bias']:+.4f} (chi1:{r['grad_chi1_limit_scale_bias']:+.4f}) "
                f"| rel_std={r['phi_rel_std']:.3f}"
            )

    print("\n[2C] Convex EBI Squared-Softplus Cap Jensen Bias (beta=12, kappa=1.0):")
    for r in all_results["exp2"]["convex_ebi_softplus_cap_jensen"]:
        if r["k_probes"] in (1, 4, 32):
            print(
                f"  r={r['r_true']:.1f} K={r['k_probes']:2d} "
                f"| h_true={r['h_true']:.5f} E[h]={r['h_mc_mean']:.5f} (ratio={r['h_ratio']:.2f}x) "
                f"| P(r_hat>1)={r['prob_exceed_kappa']:.3f} | E[r_hat]/r={r['norm_est_mean_ratio']:.4f} "
                f"| grad_ratio={r['grad_mult_ratio']:.2f}x"
            )

    print("\n[2D] Separable vs Dense Rademacher Variance Blowup (3x32x32):")
    for r in all_results["exp2"]["separable_vs_dense_variance"]:
        print(
            f"  {r['gradient_structure']:20s} r_eff2={r['renyi2_eff_rank']:5.2f} "
            f"| Var dense={r['dense_var']:.2f} (th={r['dense_var_theory']:.2f}), "
            f"2-way sep={r['sep2_var']:.2f} (exact_th={r['sep2_var_exact_theory']:.2f}), "
            f"3-way sep={r['sep3_var']:.2f}"
        )

    print("\n[3A] Controlled ReLU/LeakyReLU Kink Crossing (eta=1e-4, FP64):")
    for r in all_results["exp3"]["controlled_kink_impulse_fp64"]["rows"]:
        if r["method"] == "fwAD" or r["delta"] in (2e-2, 1e-3, 2e-4, 5e-5):
            print(
                f"  {r['method']:6s} delta={r['delta']:.1e} cross={str(r['crosses_kink']):5s} "
                f"| tau_err={r['tau_rel_err']:.3e} | grad_norm_ratio={r['norm_ratio']:.3f} | grad_cos={r['cos']:.5f}"
            )

    print("\n[3B] Actual DCGANCritic BF16 Micro-Secant & Kink Audit (cuda:0):")
    for r in all_results["exp3"]["dcgan_bf16_micro_secant_audit"]:
        print(
            f"  r{r['res']:3d} {r['method']:26s} | surv_px={r['surviving_pixel_frac']:.4f} "
            f"| flips={r['relu_flip_count']:7d} ({100*r['relu_flip_frac']:.3f}%) "
            f"| zero_diff={r['zero_diff_frac']:.2f} | tau_rel={r['tau_rel_l2']:.3f} "
            f"| grad_cos={r['cos']:+.4f} norm_ratio={r['norm_ratio']:.3f}"
        )

    print("\n[4A] MLP Dimension Scaling on RTX A6000 (B=256, 4x256 LeakyReLU, FP32):")
    for r in all_results["exp4"]["mlp_dim_scaling_fp32"]:
        print(
            f"  d={r['dim']:4d} | RoR={r['ror_exact']['median_ms']:6.3f}ms ({r['ror_exact']['peak_alloc_mib']:5.1f}MiB) "
            f"| fwAD_exact={r['fwad_exact_basis']['median_ms']:7.3f}ms ({r['fwad_exact_basis']['peak_alloc_mib']:6.1f}MiB, "
            f"{r['exact_fwad_over_ror_latency']:5.2f}x) "
            f"| fwAD_1p={r['fwad_1probe']['median_ms']:5.3f}ms ({r['fwad1_over_ror_latency']:.2f}x) "
            f"| fwAD_4p={r['fwad_4probe']['median_ms']:5.3f}ms | fwAD_8p={r['fwad_8probe']['median_ms']:5.3f}ms "
            f"| sec_re={r['micro_secant_with_primal']['median_ms']:5.3f}ms"
        )

    print("\n[4B] Conv2d DCGANCritic Resolution Scaling on RTX A6000 (B=16):")
    for r in all_results["exp4"]["conv2d_resolution_scaling"]:
        e_ror = r["ror_bn_train"]["energy_mj_per_iter"]
        e_fw = r["fwad_1probe_interleaved"]["energy_mj_per_iter"]
        e_str = f" | E(mJ): RoR={e_ror:.0f} fwAD1i={e_fw:.0f}" if (e_ror is not None and e_fw is not None) else ""
        print(
            f"  r{r['res']:3d} {r['precision']:18s} | RoR_tr={r['ror_bn_train']['median_ms']:6.2f}ms ({r['ror_bn_train']['peak_alloc_mib']:6.1f}MiB) "
            f"| RoR_ev={r['ror_bn_eval']['median_ms']:6.2f}ms ({r['ror_bn_eval']['peak_alloc_mib']:6.1f}MiB) "
            f"| fwAD1i={r['fwad_1probe_interleaved']['median_ms']:5.2f}ms ({r['fwad_1probe_interleaved']['peak_alloc_mib']:5.1f}MiB, {r['speedup_fwad_interleaved_vs_ror_train']:.2f}x) "
            f"| fwAD1b={r['fwad_1probe_both']['median_ms']:5.2f}ms ({r['speedup_fwad_both_vs_ror_eval']:.2f}x) "
            f"| fwAD4i={r['fwad_4probe_interleaved']['median_ms']:5.2f}ms "
            f"| sec1={r['micro_secant_1probe']['median_ms']:5.2f}ms{e_str}"
        )

    print("\n[4C] Architectural Compatibility Matrix (RTX A6000):")
    for r in all_results["exp4"]["architecture_compatibility_matrix"]:
        def _fmt(d: Dict[str, Any]) -> str:
            if d["status"] == "OK":
                return f"OK({d['median_ms']:.2f}ms,{d['peak_alloc_mib']:.0f}M)"
            return f"FAIL({d['error'][:36]})"
        print(
            f"  {r['architecture']:28s} {r['precision']:4s} | RoR: {_fmt(r['ror_exact']):24s} "
            f"| fwAD: {_fmt(r['fwad_1probe']):42s} | Sec: {_fmt(r['micro_secant'])}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None, help="Optional path to save full JSON metrics.")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    nvml = NVMLMonitor(device.index if device.type == "cuda" and device.index is not None else 0)
    try:
        meta = {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "nvml_available": nvml.available,
        }
        all_results: Dict[str, Any] = {"meta": meta}
        all_results["exp1"] = exp1_derivative_order_and_paths()
        all_results["exp2"] = exp2_bias_variance_jensen()
        all_results["exp3"] = exp3_kink_and_bf16_secant_audit(device)
        if device.type == "cuda":
            all_results["exp4"] = exp4_scaling_regimes_gpu(device, nvml)
        else:
            all_results["exp4"] = {"skipped": "CUDA device required for RTX A6000 GPU scaling benchmarks"}

        _print_summary(all_results)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as fh:
                json.dump(all_results, fh, indent=2)
            print(f"\nSaved full JSON results to {args.output_json}")
    finally:
        nvml.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
