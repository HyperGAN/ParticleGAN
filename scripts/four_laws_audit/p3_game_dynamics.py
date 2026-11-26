#!/usr/bin/env python3
"""P3 Adversarial Audit: Joint Minimax Game Dynamics & Four-Laws Counterexample Suite.

Investigates:
1. Linearized joint game Jacobian analysis:
   - Symmetric (potential/dissipative) vs antisymmetric (Hamiltonian/rotational)
     decomposition `J_F = S + A`.
   - Simultaneous Euler GD (`I - eta J_F`) vs 1:1 alternating D-then-G GD
     (exact symplectic identity `det(M_alt) = 1 - eta_D * mu_D`).
   - Non-normality (`||J_F J_F^T - J_F^T J_F||_F`, Jordan blocks, capacity ratio `rho`)
     and Kreiss / pseudospectral transient amplification despite `rho(M) < 1`.
   - Exact closed-form Schur-Cohn stability boundary for Heavy-Ball momentum
     (`eta_max(beta_1; gamma, omega)`) + alternating momentum 4x4 spectrum.
   - OGDA (`lam = 0.5` neutral boundary vs `lam = 1.0` contracting) and i.i.d.
     minibatch variance inflation `(1 + lam)^2 + lam^2`.
2. Exact local curvature audit of ParticleGAN's actual Law-2 critic penalties
   (`GradientPenalty(arm='b_cap')`, `UnifiedV11CriticPenalty` with/without
   `lambda_floor`, `MinimalCriticPenalty` Born-Infeld, and `R1/R2`) and Law-4
   `delta_g_max` pushforward trust band (`UnifiedV12Optimizer` /
   `UnifiedV11EquilibratedOptimizer`).
3. Explicit counterexamples satisfying all four operational laws as stated in
   `docs/four-laws.md` yet failing to converge:
   (A1) Analytic 2D bounded 1-Lipschitz Wasserstein Dirac-GAN (`||grad_x D|| < 1`,
        zero-mean gauge), testing both inactive AND actively-clamping `delta_g_max`,
        alternating/simultaneous GD, Adam (`beta_1 in {0.0, 0.9}`), actual
        `UnifiedV12Optimizer`, `OGDA (lam=0.5, 1.0)`, `lambda_floor=0.005`, and
        true Born-Infeld (`minimal`).
   (A2) Analytic 2D codimension-1 manifold-embedded Linear-Gaussian GAN under
        ParticleGAN's default `non_saturating` logistic loss (Mescheder et al. 2018
        Theorem 4.1: tangent axis converges, normal axis enters persistent limit cycle!).
   (A3) Analytic 2D full-support isotropic Gaussian GAN under ParticleGAN's public
        `hinge` loss (proves `J_DD = 0` persists even on full-dimensional support!).
   (B)  Nonlinear 2D MLP minimax game (matched capacity `rho ~ 1`, smooth `SiLU`,
        well-conditioned residual generator, exercising actual `UnifiedV12Optimizer`
        2D Quintic NS on G, 0-NS Frobenius-MP on D, `delta_g_max` secant probe, and
        actual ParticleGAN penalties) on BOTH `wasserstein`/`hinge` and codimension-1
        manifold `non_saturating` logistic loss.

Usage:
    PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
        scripts/four_laws_audit/p3_game_dynamics.py
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import particlegan
from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradientPenalty
from particlegan.recipes import get_recipe
from particlegan._minimal.geometry import MinimalCriticPenalty
from particlegan._unified_v11.critic import (
    UnifiedV11StepRecord,
    _attach_unified_v11_critic_hook,
    _attach_unified_v11_generator_hook,
)
from particlegan._unified_v11.penalty import UnifiedV11CriticPenalty
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer


# ---------------------------------------------------------------------------
# Part 1: Joint Game Jacobian, Spectral, Non-Normal & Momentum Theory
# ---------------------------------------------------------------------------


def build_damped_bilinear_jacobian(mu_d: float, omega: float = 1.0, rho: float = 1.0) -> np.ndarray:
    """Linearized Jacobian J_F for damped bilinear game with capacity gain ratio `rho`.

    F_theta = -omega * rho * psi,  F_psi = (omega / rho) * theta + mu_d * psi.
    Eigenvalues depend ONLY on (mu_d, omega), while departure from normality grows as Theta(rho^2).
    """
    return np.array([[0.0, -omega * rho], [omega / rho, mu_d]], dtype=np.float64)


def sim_gd_matrix(J: np.ndarray, eta: float) -> np.ndarray:
    """Simultaneous Euler GD update matrix M = I - eta * J."""
    return np.eye(J.shape[0], dtype=np.float64) - eta * J


def alt_gd_matrix(mu_d: float, eta_g: float, eta_d: float, omega: float = 1.0, rho: float = 1.0) -> np.ndarray:
    """Alternating 1:1 D-then-G update matrix matching `particlegan/training.py`.

    Exact determinant: det(M_alt) = 1 - eta_d * mu_d.
    """
    M_d = np.array([[1.0, 0.0], [-eta_d * (omega / rho), 1.0 - eta_d * mu_d]], dtype=np.float64)
    M_g = np.array([[1.0, eta_g * omega * rho], [0.0, 1.0]], dtype=np.float64)
    return M_g @ M_d


def sim_momentum_companion(J: np.ndarray, eta: float, beta1: float) -> np.ndarray:
    """Companion matrix for simultaneous Heavy-Ball: x_{k+1} = (1+b1) x_k - eta J x_k - b1 x_{k-1}."""
    n = J.shape[0]
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    return np.vstack([np.hstack([(1.0 + beta1) * I - eta * J, -beta1 * I]), np.hstack([I, Z])])


def alt_momentum_companion(mu_d: float, eta_g: float, eta_d: float, beta1: float, omega: float = 1.0) -> np.ndarray:
    """4x4 companion matrix for alternating 1:1 D-then-G Heavy-Ball momentum."""
    a_d = 1.0 + beta1 - eta_d * mu_d
    row_psi = np.array([-eta_d * omega, a_d, 0.0, -beta1], dtype=np.float64)
    row_theta = np.array([1.0 + beta1, 0.0, -beta1, 0.0], dtype=np.float64) + eta_g * omega * row_psi
    return np.vstack(
        [
            row_theta,
            row_psi,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        ]
    )


def ogda_companion(J: np.ndarray, eta: float, lam: float = 1.0) -> np.ndarray:
    """Companion matrix for simultaneous OGDA: x_{k+1} = x_k - eta ((1+lam) J x_k - lam J x_{k-1})."""
    n = J.shape[0]
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    return np.vstack([np.hstack([I - eta * (1.0 + lam) * J, eta * lam * J]), np.hstack([I, Z])])


def schur_cohn_eta_max(gamma: float, omega: float, beta1: float) -> float:
    """Exact Schur-Cohn stability ceiling for Heavy-Ball on lambda = gamma + i*omega.

    Derived from |a_1 - conj(a_1) a_0| < 1 - a_0^2 for z^2 - (1 + beta1 - eta*lambda) z + beta1 = 0:
        eta_max(beta1; gamma, omega) = (2 * gamma * (1 + beta1) * (1 - beta1)^2)
                                       / (gamma^2 * (1 - beta1)^2 + omega^2 * (1 + beta1)^2).
    Recovers Polyak's 2*(1+beta1)/gamma at omega = 0 and collapses as (2*gamma/omega^2)*(1-beta1)^2/(1+beta1)
    for underdamped rotational games (omega >> gamma).
    """
    if gamma <= 0.0:
        return 0.0
    one_m = 1.0 - beta1
    one_p = 1.0 + beta1
    return (2.0 * gamma * one_p * one_m * one_m) / (gamma * gamma * one_m * one_m + omega * omega * one_p * one_p)


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def estimate_kreiss_and_transient(M: np.ndarray, max_power: int = 300) -> Dict[str, float]:
    """Compute spectral radius, peak transient amplification sup_k ||M^k||_2, and Kreiss constant lower bound."""
    rho = spectral_radius(M)
    n = M.shape[0]
    P = np.eye(n, dtype=np.float64)
    max_norm = 1.0
    peak_k = 0
    for k in range(1, max_power + 1):
        P = P @ M
        nrm = float(np.linalg.norm(P, ord=2))
        if nrm > max_norm:
            max_norm = nrm
            peak_k = k
    radii = 1.0 + np.geomspace(1e-4, 2.0, 50)
    angles = np.linspace(0.0, 2.0 * math.pi, 144, endpoint=False)
    I = np.eye(n, dtype=np.complex128)
    kreiss = 1.0
    for r in radii:
        zs = r * np.exp(1j * angles)
        for z in zs:
            val = (r - 1.0) * float(np.linalg.norm(np.linalg.inv(z * I - M), ord=2))
            if val > kreiss:
                kreiss = val
    return {
        "spectral_radius": rho,
        "peak_transient_norm": max_norm,
        "peak_step": float(peak_k),
        "kreiss_constant": kreiss,
    }


def analyze_linear_game_dynamics() -> Dict[str, Any]:
    """Run all Part 1 exact linear game-Jacobian analyses."""
    out: Dict[str, Any] = {}

    sim_vs_alt: List[Dict[str, Any]] = []
    for mu_d in (0.0, 0.01, 0.1, 1.0, 2.0):
        J = build_damped_bilinear_jacobian(mu_d=mu_d, omega=1.0, rho=1.0)
        S = 0.5 * (J + J.T)
        A = 0.5 * (J - J.T)
        eigs_J = np.linalg.eigvals(J)
        for eta_g, eta_d in ((0.014, 0.014), (0.014, 0.021), (0.05, 0.05)):
            M_sim = np.eye(2) - np.diag([eta_g, eta_d]) @ J
            M_alt = alt_gd_matrix(mu_d=mu_d, eta_g=eta_g, eta_d=eta_d)
            sim_vs_alt.append(
                {
                    "mu_d": mu_d,
                    "eta_g": eta_g,
                    "eta_d": eta_d,
                    "sym_norm": float(np.linalg.norm(S, "fro")),
                    "antisym_norm": float(np.linalg.norm(A, "fro")),
                    "J_eig_real_min": float(np.min(np.real(eigs_J))),
                    "J_eig_imag_max": float(np.max(np.abs(np.imag(eigs_J)))),
                    "rho_sim": spectral_radius(M_sim),
                    "rho_alt": spectral_radius(M_alt),
                    "det_alt": float(np.linalg.det(M_alt)),
                    "det_alt_exact": 1.0 - eta_d * mu_d,
                }
            )
    out["sim_vs_alt"] = sim_vs_alt

    non_normal: List[Dict[str, Any]] = []
    for rho_cap in (1.0, 10.0, 50.0, 100.0, 500.0):
        J = build_damped_bilinear_jacobian(mu_d=0.5, omega=1.0, rho=rho_cap)
        dep_norm = float(np.linalg.norm(J @ J.T - J.T @ J, "fro"))
        M = sim_gd_matrix(J, eta=0.1)
        _, evecs = np.linalg.eig(J)
        cond_v = float(np.linalg.cond(evecs))
        kt = estimate_kreiss_and_transient(M, max_power=250)
        non_normal.append(
            {
                "rho_capacity": rho_cap,
                "departure_from_normality": dep_norm,
                "eigvec_cond": cond_v,
                **kt,
            }
        )
    J_jordan = build_damped_bilinear_jacobian(mu_d=2.0, omega=1.0, rho=1.0)
    J_shift = np.array([[0.2, 50.0], [0.0, 0.2]], dtype=np.float64)
    out["non_normal"] = non_normal
    out["jordan_critical"] = {
        "departure_from_normality": float(np.linalg.norm(J_jordan @ J_jordan.T - J_jordan.T @ J_jordan, "fro")),
        **estimate_kreiss_and_transient(sim_gd_matrix(J_jordan, eta=0.1), max_power=250),
    }
    out["jordan_shifted_50"] = {
        "departure_from_normality": float(np.linalg.norm(J_shift @ J_shift.T - J_shift.T @ J_shift, "fro")),
        **estimate_kreiss_and_transient(sim_gd_matrix(J_shift, eta=0.1), max_power=250),
    }

    momentum_table: List[Dict[str, Any]] = []
    for mu_d in (0.0, 0.01, 0.2):
        J = build_damped_bilinear_jacobian(mu_d=mu_d, omega=1.0, rho=1.0)
        eigs = np.linalg.eigvals(J)
        gamma = float(np.real(eigs[0]))
        omega_e = float(abs(np.imag(eigs[0])))
        for beta1 in (-0.5, -0.3, 0.0, 0.5, 0.85, 0.9):
            eta_max_closed = schur_cohn_eta_max(gamma, omega_e, beta1)
            rho_below = spectral_radius(sim_momentum_companion(J, 0.98 * eta_max_closed, beta1)) if eta_max_closed > 0 else float("nan")
            rho_above = spectral_radius(sim_momentum_companion(J, 1.02 * eta_max_closed, beta1)) if eta_max_closed > 0 else float("nan")
            momentum_table.append(
                {
                    "mu_d": mu_d,
                    "beta1": beta1,
                    "eta_max_closed": eta_max_closed,
                    "rho_at_0.98_eta_max": rho_below,
                    "rho_at_1.02_eta_max": rho_above,
                    "rho_sim_eta0014": spectral_radius(sim_momentum_companion(J, 0.014, beta1)),
                    "rho_alt_v12": spectral_radius(alt_momentum_companion(mu_d, 0.014, 0.021, beta1)),
                }
            )
    out["momentum_table"] = momentum_table

    ogda_table: List[Dict[str, Any]] = []
    for mu_d in (0.0, 0.01, 0.2):
        J = build_damped_bilinear_jacobian(mu_d=mu_d, omega=1.0, rho=1.0)
        for lam in (0.0, 0.5, 1.0):
            ogda_table.append(
                {
                    "mu_d": mu_d,
                    "lam": lam,
                    "rho_ogda_eta005": spectral_radius(ogda_companion(J, eta=0.05, lam=lam)),
                    "iid_noise_variance_mult": (1.0 + lam) ** 2 + lam ** 2,
                }
            )
    out["ogda_table"] = ogda_table
    return out


# ---------------------------------------------------------------------------
# Part 2: Exact Curvature Audit of ParticleGAN's Actual Law-2 Critic Penalties
# ---------------------------------------------------------------------------


class _LinearCritic(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w) + self.b


def _measure_penalty_curvature(penalty_fn, critic: _LinearCritic, x_r: torch.Tensor, x_f: torch.Tensor, eps: float = 1e-3) -> Dict[str, float]:
    """Measure exact second derivatives d^2 R / dw_0^2 and d^2 R / db^2 at (w=0, b=0)."""
    def grad_w0(val: float) -> float:
        with torch.no_grad():
            critic.w.zero_()
            critic.w[0] = val
            critic.b.zero_()
        critic.zero_grad(set_to_none=True)
        p = penalty_fn(critic, x_r, x_f)
        if not p.requires_grad:
            return 0.0
        p.backward()
        return 0.0 if critic.w.grad is None else float(critic.w.grad[0].item())

    def grad_b(val: float) -> float:
        with torch.no_grad():
            critic.w.zero_()
            critic.b.fill_(val)
        critic.zero_grad(set_to_none=True)
        p = penalty_fn(critic, x_r, x_f)
        if not p.requires_grad:
            return 0.0
        p.backward()
        return 0.0 if critic.b.grad is None else float(critic.b.grad[0].item())

    curv_w = (grad_w0(eps) - grad_w0(-eps)) / (2.0 * eps)
    curv_b = (grad_b(eps) - grad_b(-eps)) / (2.0 * eps)
    with torch.no_grad():
        critic.w.zero_()
        critic.w[0] = 0.5
        critic.b.zero_()
    critic.zero_grad(set_to_none=True)
    p_half = penalty_fn(critic, x_r, x_f)
    if p_half.requires_grad:
        p_half.backward()
        g_half = 0.0 if critic.w.grad is None else float(critic.w.grad[0].item())
    else:
        g_half = 0.0
    return {
        "d2R_dw2_at_0": curv_w,
        "d2R_db2_at_0": curv_b,
        "dR_dw_at_0.5": g_half,
        "R_at_0.5": float(p_half.detach().item()),
    }


def audit_particlegan_penalties() -> Dict[str, Any]:
    """Measure exact local Hessian curvature mu_D of ParticleGAN's actual Law-2 penalties."""
    torch.manual_seed(0)
    x_r = torch.randn(32, 2)
    x_f = torch.randn(32, 2)
    G_dummy = nn.Linear(2, 2)
    r_v12 = get_recipe("unified_v12")
    r_min = get_recipe("minimal")
    results: Dict[str, Any] = {}

    D1 = _LinearCritic(2)
    bcap = GradientPenalty(arm="b_cap", coeff=1.0, kappa=1.0)
    results["bcap_hinge"] = _measure_penalty_curvature(lambda c, xr, xf: bcap(c, xr, xf, step=1), D1, x_r, x_f)

    D2 = _LinearCritic(2)
    _, opt_d2 = r_v12.make_optimizers(G_dummy, D2)
    pen_v12_no_floor = UnifiedV11CriticPenalty(r_v12, opt_d2, lambda_floor=0.0)
    results["unified_v11_no_floor"] = _measure_penalty_curvature(pen_v12_no_floor, D2, x_r, x_f)

    D3 = _LinearCritic(2)
    _, opt_d3 = r_v12.make_optimizers(G_dummy, D3)
    pen_v12_default = UnifiedV11CriticPenalty(r_v12, opt_d3)
    results["unified_v11_default"] = _measure_penalty_curvature(pen_v12_default, D3, x_r, x_f)

    D4 = _LinearCritic(2)
    _, opt_d4 = r_min.make_optimizers(G_dummy, D4)
    pen_min = MinimalCriticPenalty(r_min, opt_d4)
    results["minimal_born_infeld"] = _measure_penalty_curvature(pen_min, D4, x_r, x_f)

    D5 = _LinearCritic(2)
    r1r2 = GradientPenalty(arm="a_r1r2", coeff=1.0)
    results["r1r2_reference"] = _measure_penalty_curvature(lambda c, xr, xf: r1r2(c, xr, xf, step=1), D5, x_r, x_f)

    return results


# ---------------------------------------------------------------------------
# Part 3A: Analytic Counterexamples Satisfying All Four Operational Laws
# ---------------------------------------------------------------------------


class BoundedLipschitzLinearCritic(nn.Module):
    """Smooth C^infty critic D_{psi, b}(x) = (tanh(||psi||)/||psi||) psi^T x + b.

    Provably satisfies ||grad_x D(x)||_2 = tanh(||psi||_2) < 1 EVERYWHERE in R^d,
    with exact expansion D(x) = psi^T x + b + O(||psi||^3) around psi = 0.
    """

    def __init__(self, dim: int = 2, init_psi: Tuple[float, ...] = (0.0, 0.0), init_b: float = 0.0) -> None:
        super().__init__()
        self.psi = nn.Parameter(torch.tensor(init_psi, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([init_b], dtype=torch.float32))

    def slope_vector(self) -> torch.Tensor:
        r_sq = self.psi.square().sum()
        r = torch.sqrt(r_sq + 1e-16)
        scale = torch.where(r_sq > 1e-10, torch.tanh(r) / r, 1.0 - r_sq / 3.0)
        return self.psi * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.slope_vector()) + self.b


class ShiftGenerator(nn.Module):
    """Exact Law-1/Law-3/Law-4 compliant generator G_theta(z) = theta + mask * z."""

    def __init__(self, init_theta: Tuple[float, ...] = (0.4, 0.0), z_mask: Tuple[float, ...] = (0.0, 0.0)) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(init_theta, dtype=torch.float32))
        self.register_buffer("z_mask", torch.tensor(z_mask, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.theta.unsqueeze(0) + z * self.z_mask.unsqueeze(0)


def run_analytic_game(
    *,
    mode: str,
    loss_name: str,
    z_mask: Tuple[float, float],
    real_mask: Tuple[float, float],
    init_theta: Tuple[float, float],
    init_psi: Tuple[float, float],
    steps: int = 3000,
    eta_g: float = 0.014,
    eta_d: float = 0.021,
    delta_g_max: float | None = None,
    ogda_lam: float = 1.0,
    gamma_gauge: float = 0.03,
    mu_d_penalty: str = "none",
    batch_size: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run an analytic 2D game satisfying all 4 operational laws."""
    gen_rng = torch.Generator().manual_seed(seed)
    G = ShiftGenerator(init_theta=init_theta, z_mask=z_mask)
    D = BoundedLipschitzLinearCritic(dim=2, init_psi=init_psi, init_b=0.0)
    gan_loss = GANLoss(loss_name)

    rec: UnifiedV11StepRecord | None = None
    opt_g: Any = None
    opt_d: Any = None

    if mode == "unified_v12":
        rec = UnifiedV11StepRecord()
        rec.ema_data_scale = 1.0
        rec.ema_push_res = 0.0
        opt_g = UnifiedV12Optimizer(
            [{"params": list(G.parameters()), "role": "generator"}],
            lr=eta_g,
            betas=(0.0, 0.999),
            smoothing=0.0061,
        )
        opt_g._unified_v11_record = rec
        opt_d = UnifiedV12Optimizer(
            [{"params": list(D.parameters()), "role": "critic"}],
            lr=eta_d,
            betas=(0.0, 0.999),
            smoothing=0.0061,
        )
        opt_d.critic = D
        opt_d.record = rec
        _attach_unified_v11_critic_hook(D, rec)
        _attach_unified_v11_generator_hook(G, rec)
    elif mode == "alt_adam_b09":
        opt_g = torch.optim.Adam(G.parameters(), lr=eta_g, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=eta_d, betas=(0.9, 0.999))
    elif mode == "alt_adam_b00":
        opt_g = torch.optim.Adam(G.parameters(), lr=eta_g, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=eta_d, betas=(0.0, 0.999))

    real_mask_t = torch.tensor(real_mask, dtype=torch.float32)

    def critic_penalty() -> torch.Tensor:
        w = D.slope_vector()
        g_sq = w.square().sum()
        g_norm = torch.sqrt(g_sq + 1e-12)
        if mu_d_penalty == "none":
            return g_sq * 0.0
        if mu_d_penalty == "bcap_hinge":
            return F.relu(g_norm - 1.0).square()
        if mu_d_penalty == "unified_softplus_no_floor":
            return F.softplus(g_norm - 1.0, beta=12.0).square()
        if mu_d_penalty == "unified_lambda_floor":
            # Exact unified_v11/v12 formula: 0.5*coeff*(cap_r+cap_f + lambda_floor*(sq_r+sq_f)) with lambda_floor=0.005
            return F.softplus(g_norm - 1.0, beta=12.0).square() + 0.005 * g_sq
        if mu_d_penalty == "born_infeld":
            return (torch.sqrt(1.0 + 8.0 * g_sq) - 1.0) / 8.0
        raise ValueError(f"unknown mu_d_penalty: {mu_d_penalty}")

    theta_hist: List[List[float]] = []
    psi_hist: List[List[float]] = []
    max_lip = 0.0
    max_abs_mean_d = 0.0
    clamp_count = 0
    prev_grad_g: torch.Tensor | None = None
    prev_grad_d: List[torch.Tensor] | None = None

    for _ in range(steps):
        if any(m > 0.0 for m in real_mask):
            z_r = torch.randn(batch_size, 2, generator=gen_rng)
            x_r = z_r * real_mask_t.unsqueeze(0)
            z_g = torch.randn(batch_size, 2, generator=gen_rng)
        else:
            x_r = torch.zeros(batch_size, 2)
            z_g = torch.zeros(batch_size, 2)

        with torch.no_grad():
            max_lip = max(max_lip, float(D.slope_vector().norm().item()))

        if mode in ("sim_gd", "sim_gd_trust", "sim_ogda"):
            G.zero_grad(set_to_none=True)
            D.zero_grad(set_to_none=True)
            x_f_det = G(z_g).detach()
            d_r = D(x_r)
            d_f = D(x_f_det)
            mean_d = 0.5 * (d_r.mean() + d_f.mean())
            max_abs_mean_d = max(max_abs_mean_d, abs(float(mean_d.detach().item())))
            loss_d = gan_loss.d_loss(d_r, d_f) + critic_penalty() + 0.5 * gamma_gauge * mean_d.square()
            loss_d.backward()
            grad_d = [p.grad.detach().clone() for p in D.parameters()]

            G.zero_grad(set_to_none=True)
            x_f = G(z_g)
            loss_g = gan_loss.g_loss(D(x_f), D(x_r).detach())
            loss_g.backward()
            grad_g = G.theta.grad.detach().clone()

            with torch.no_grad():
                if mode == "sim_ogda":
                    eff_g = grad_g if prev_grad_g is None else ((1.0 + ogda_lam) * grad_g - ogda_lam * prev_grad_g)
                    eff_d = (
                        grad_d
                        if prev_grad_d is None
                        else [(1.0 + ogda_lam) * gd - ogda_lam * pd for gd, pd in zip(grad_d, prev_grad_d)]
                    )
                    prev_grad_g = grad_g.clone()
                    prev_grad_d = [gd.clone() for gd in grad_d]
                    G.theta.sub_(eta_g * eff_g)
                    for p, gd in zip(D.parameters(), eff_d):
                        p.sub_(eta_d * gd)
                else:
                    step_g = -eta_g * grad_g
                    if mode == "sim_gd_trust" and delta_g_max is not None:
                        disp = float(step_g.norm().item()) / math.sqrt(2.0)
                        if disp > delta_g_max:
                            step_g = step_g * (delta_g_max / disp)
                            clamp_count += 1
                    G.theta.add_(step_g)
                    for p, gd in zip(D.parameters(), grad_d):
                        p.sub_(eta_d * gd)
        elif mode == "alt_gd":
            D.zero_grad(set_to_none=True)
            x_f_det = G(z_g).detach()
            d_r = D(x_r)
            d_f = D(x_f_det)
            mean_d = 0.5 * (d_r.mean() + d_f.mean())
            max_abs_mean_d = max(max_abs_mean_d, abs(float(mean_d.detach().item())))
            loss_d = gan_loss.d_loss(d_r, d_f) + critic_penalty() + 0.5 * gamma_gauge * mean_d.square()
            loss_d.backward()
            with torch.no_grad():
                for p in D.parameters():
                    p.sub_(eta_d * p.grad)

            G.zero_grad(set_to_none=True)
            x_f = G(z_g)
            loss_g = gan_loss.g_loss(D(x_f), D(x_r).detach())
            loss_g.backward()
            with torch.no_grad():
                step_g = -eta_g * G.theta.grad
                if delta_g_max is not None:
                    disp = float(step_g.norm().item()) / math.sqrt(2.0)
                    if disp > delta_g_max:
                        step_g = step_g * (delta_g_max / disp)
                        clamp_count += 1
                G.theta.add_(step_g)
        else:
            opt_d.zero_grad(set_to_none=True)
            x_f_det = G(z_g).detach()
            d_r = D(x_r)
            d_f = D(x_f_det)
            mean_d = 0.5 * (d_r.mean() + d_f.mean())
            max_abs_mean_d = max(max_abs_mean_d, abs(float(mean_d.detach().item())))
            loss_d = gan_loss.d_loss(d_r, d_f) + critic_penalty() + 0.5 * gamma_gauge * mean_d.square()
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            x_f = G(z_g)
            loss_g = gan_loss.g_loss(D(x_f), D(x_r).detach())
            loss_g.backward()
            opt_g.step()
            if mode == "unified_v12" and opt_g.last_step_gain < 0.999:
                clamp_count += 1

        theta_hist.append(G.theta.detach().tolist())
        psi_hist.append(D.slope_vector().detach().tolist())

    th = np.array(theta_hist, dtype=np.float64)
    ps = np.array(psi_hist, dtype=np.float64)
    tail = slice(steps - 500, steps)
    r_init = float(np.linalg.norm(init_theta))
    r_tail_mean = float(np.mean(np.linalg.norm(th[tail], axis=1)))
    r_tail_max = float(np.max(np.linalg.norm(th[tail], axis=1)))
    res: Dict[str, Any] = {
        "mode": mode,
        "loss": loss_name,
        "mu_d_penalty": mu_d_penalty,
        "delta_g_max": delta_g_max,
        "ogda_lam": ogda_lam if mode == "sim_ogda" else None,
        "r_init": r_init,
        "r_final": float(np.linalg.norm(th[-1])),
        "r_tail_mean": r_tail_mean,
        "r_tail_max": r_tail_max,
        "tang_coord_tail_rms": float(np.sqrt(np.mean(th[tail, 0] ** 2))),
        "norm_coord_tail_rms": float(np.sqrt(np.mean(th[tail, 1] ** 2))),
        "psi_tail_rms": float(np.sqrt(np.mean(np.sum(ps[tail] ** 2, axis=1)))),
        "max_lip_observed": max_lip,
        "max_abs_mean_d": max_abs_mean_d,
        "clamp_count": clamp_count,
    }
    if mode == "unified_v12" and opt_g is not None:
        res["final_ema_func_gain"] = float(opt_g._ema_func_gain)
        res["final_stiff_gate"] = float(opt_g.current_stiff_gate())
        res["final_ema_flip_cos"] = float(opt_g._ema_flip_cos)
        res["probe_consumptions"] = int(opt_g.probe_consumptions)
    return res


def run_analytic_counterexamples() -> Dict[str, Any]:
    """Run counterexamples A1 (2D Dirac-GAN), A2 (manifold logistic GAN), A3 (full Gaussian hinge GAN)."""
    out: Dict[str, Any] = {}

    # A1: 2D Bounded 1-Lipschitz Wasserstein Dirac-GAN (all 4 operational laws satisfied)
    a1: List[Dict[str, Any]] = []
    configs_a1 = [
        ("sim_gd", "none", None, 1.0, 0.05, 0.05),
        ("sim_gd_trust", "none", 0.004, 1.0, 0.05, 0.05),  # actively clamping delta_g_max!
        ("alt_gd", "none", None, 1.0, 0.014, 0.021),
        ("alt_gd", "none", 0.002, 1.0, 0.014, 0.021),  # actively clamping alternating GD!
        ("alt_adam_b09", "none", None, 1.0, 0.014, 0.021),
        ("alt_adam_b00", "none", None, 1.0, 0.014, 0.021),
        ("unified_v12", "none", None, 1.0, 0.014, 0.021),
        ("unified_v12", "unified_softplus_no_floor", None, 1.0, 0.014, 0.021),
        ("unified_v12", "unified_lambda_floor", None, 1.0, 0.014, 0.021),
        ("unified_v12", "born_infeld", None, 1.0, 0.014, 0.021),
        ("alt_gd", "unified_lambda_floor", None, 1.0, 0.014, 0.021),
        ("alt_gd", "born_infeld", None, 1.0, 0.014, 0.021),
        ("sim_ogda", "none", None, 0.5, 0.05, 0.05),  # lam=0.5 neutral boundary!
        ("sim_ogda", "none", None, 1.0, 0.05, 0.05),  # lam=1.0 strict contraction!
    ]
    for mode, pen, dmax, olam, eg, ed in configs_a1:
        a1.append(
            run_analytic_game(
                mode=mode,
                loss_name="wasserstein",
                z_mask=(0.0, 0.0),
                real_mask=(0.0, 0.0),
                init_theta=(0.4, 0.0),
                init_psi=(0.0, 0.0),
                steps=3000,
                eta_g=eg,
                eta_d=ed,
                delta_g_max=dmax,
                ogda_lam=olam,
                mu_d_penalty=pen,
            )
        )
    out["A1_dirac_2d_wasserstein"] = a1

    # A2: 2D Codimension-1 Manifold-Embedded Gaussian under ParticleGAN's DEFAULT `non_saturating` logistic loss
    a2: List[Dict[str, Any]] = []
    for mode, pen, dmax, olam, eg, ed in (
        ("sim_gd_trust", "none", 0.002, 1.0, 0.05, 0.05),
        ("alt_gd", "none", None, 1.0, 0.014, 0.021),
        ("unified_v12", "none", None, 1.0, 0.014, 0.021),
        ("unified_v12", "unified_softplus_no_floor", None, 1.0, 0.014, 0.021),
        ("unified_v12", "unified_lambda_floor", None, 1.0, 0.014, 0.021),
        ("unified_v12", "born_infeld", None, 1.0, 0.014, 0.021),
        ("sim_ogda", "none", None, 1.0, 0.05, 0.05),
    ):
        a2.append(
            run_analytic_game(
                mode=mode,
                loss_name="non_saturating",
                z_mask=(1.0, 0.0),
                real_mask=(1.0, 0.0),
                init_theta=(0.4, 0.4),
                init_psi=(0.0, 0.0),
                steps=3000,
                eta_g=eg,
                eta_d=ed,
                delta_g_max=dmax,
                ogda_lam=olam,
                mu_d_penalty=pen,
            )
        )
    out["A2_manifold_gaussian_nonsat"] = a2

    # A3: 2D Full-Support Isotropic Gaussian under ParticleGAN's public `hinge` loss
    a3: List[Dict[str, Any]] = []
    for mode, pen, olam, eg, ed in (
        ("alt_gd", "none", 1.0, 0.014, 0.021),
        ("unified_v12", "none", 1.0, 0.014, 0.021),
        ("unified_v12", "unified_lambda_floor", 1.0, 0.014, 0.021),
        ("unified_v12", "born_infeld", 1.0, 0.014, 0.021),
        ("sim_ogda", "none", 1.0, 0.05, 0.05),
    ):
        a3.append(
            run_analytic_game(
                mode=mode,
                loss_name="hinge",
                z_mask=(1.0, 1.0),
                real_mask=(1.0, 1.0),
                init_theta=(0.4, 0.3),
                init_psi=(0.0, 0.0),
                steps=3000,
                eta_g=eg,
                eta_d=ed,
                ogda_lam=olam,
                mu_d_penalty=pen,
            )
        )
    out["A3_full_gaussian_hinge"] = a3

    return out


# ---------------------------------------------------------------------------
# Part 3B: Nonlinear 2D MLP Minimax Game with ACTUAL ParticleGAN Components
# ---------------------------------------------------------------------------


class SmoothResidualMLPGenerator(nn.Module):
    """Balanced shallow 2D MLP generator G(z) = (z_0, 0) + mlp(z) with orthogonal init.

    Exercises UnifiedV12Optimizer's actual 2D Quintic NS polar path and delta_g_max
    secant probe while satisfying Law 1 (unimodal, ||J_G|| = O(1)), Law 3 (unit scale,
    unconditional), and Law 4 (balanced rho ~ 1, shallow, no rank collapse, no rails).
    """

    def __init__(self, width: int = 32, init_normal_offset: float = 0.5, manifold_normal_only: bool = False) -> None:
        super().__init__()
        self.manifold_normal_only = manifold_normal_only
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.15)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.net[-1].bias[1] = init_normal_offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.manifold_normal_only:
            # 1D manifold pushforward embedded in R^2: x_0 = z_0 + f_0(z_0), x_1 = f_1(z_0)
            z_1d = torch.stack((z[:, 0], torch.zeros_like(z[:, 1])), dim=1)
            return z_1d + self.net(z_1d)
        base = torch.stack((z[:, 0], torch.zeros_like(z[:, 1])), dim=1)
        return base + self.net(z)


class SmoothMLPCritic(nn.Module):
    """Balanced smooth C^infty 2D MLP critic with matched capacity (rho ~ 1)."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.25)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _eval_critic_grad_stats(D: nn.Module, x: torch.Tensor) -> Tuple[float, float, float]:
    x_req = x.detach().clone().requires_grad_(True)
    y = D(x_req)
    g = torch.autograd.grad(y.sum(), x_req, create_graph=False)[0]
    gn = g.norm(dim=1)
    return float(gn.median().item()), float(gn.max().item()), float(g[:, 1].mean().abs().item())


def run_nonlinear_mlp_game(
    *,
    penalty_kind: str,
    loss_name: str = "wasserstein",
    betas: Tuple[float, float] = (0.0, 0.999),
    simultaneous: bool = False,
    steps: int = 1200,
    tail_window: int = 300,
    batch_size: int = 64,
    lr_g: float = 0.014,
    lr_d: float = 0.021,
    seed: int = 0,
    device: str = "cpu",
    manifold_normal_only: bool = True,
) -> Dict[str, Any]:
    """Run matched 2D nonlinear MLP minimax game with actual ParticleGAN UnifiedV12Optimizer & penalties.

    All arms share identical `UnifiedV11StepRecord` scale/margin observations so the ONLY
    difference between arms is the Law-2 critic penalty (or betas / simultaneous flag).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    G = SmoothResidualMLPGenerator(width=32, init_normal_offset=0.5, manifold_normal_only=manifold_normal_only).to(dev)
    D = SmoothMLPCritic(width=32).to(dev)
    gan_loss = GANLoss(loss_name)
    r_v12 = get_recipe("unified_v12")
    r_min = get_recipe("minimal")

    rec = UnifiedV11StepRecord()
    rec.ema_data_scale = 1.0
    rec.ema_push_res = 0.0
    opt_g = UnifiedV12Optimizer(
        [{"params": list(G.parameters()), "role": "generator"}],
        lr=lr_g,
        betas=betas,
        smoothing=0.0061,
    )
    opt_g._unified_v11_record = rec
    opt_d = UnifiedV12Optimizer(
        [{"params": list(D.parameters()), "role": "critic"}],
        lr=lr_d,
        betas=betas,
        smoothing=0.0061,
    )
    opt_d.critic = D
    opt_d.record = rec
    _attach_unified_v11_critic_hook(D, rec)
    _attach_unified_v11_generator_hook(G, rec)

    if penalty_kind == "bcap_hinge_gauge":
        bcap_obj = GradientPenalty(arm="b_cap", coeff=1.0, kappa=1.0)

        def pen_fn(xr: torch.Tensor, xf: torch.Tensor, dr: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
            mean_d = 0.5 * (dr.mean() + df.mean())
            return bcap_obj(D, xr, xf, step=1) + 0.5 * 0.03 * mean_d.square()

    elif penalty_kind == "unified_v11_no_floor":
        u_pen = UnifiedV11CriticPenalty(r_v12, opt_d, lambda_floor=0.0)

        def pen_fn(xr: torch.Tensor, xf: torch.Tensor, dr: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
            return u_pen(D, xr, xf)

    elif penalty_kind == "unified_v11_default":
        u_pen = UnifiedV11CriticPenalty(r_v12, opt_d)

        def pen_fn(xr: torch.Tensor, xf: torch.Tensor, dr: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
            return u_pen(D, xr, xf)

    elif penalty_kind == "minimal_born_infeld":
        _, opt_d_min_dummy = r_min.make_optimizers(G, D)
        m_pen = MinimalCriticPenalty(r_min, opt_d_min_dummy)
        _attach_unified_v11_critic_hook(D, rec)

        def pen_fn(xr: torch.Tensor, xf: torch.Tensor, dr: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
            return m_pen(D, xr, xf)

    else:
        raise ValueError(f"unknown penalty_kind: {penalty_kind}")

    gen_rng = torch.Generator(device="cpu").manual_seed(seed + 100)
    normal_mean_hist: List[float] = []
    normal_std_hist: List[float] = []
    normal_rms_hist: List[float] = []
    lip_med_hist: List[float] = []
    lip_max_hist: List[float] = []
    mean_d_hist: List[float] = []
    slope_n_hist: List[float] = []

    for step in range(steps):
        z_r = torch.randn(batch_size, 2, generator=gen_rng).to(dev)
        x_r = torch.stack((z_r[:, 0], torch.zeros_like(z_r[:, 1])), dim=1)
        z_g = torch.randn(batch_size, 2, generator=gen_rng).to(dev)

        if simultaneous:
            opt_d.zero_grad(set_to_none=True)
            opt_g.zero_grad(set_to_none=True)
            x_f_det = G(z_g).detach()
            d_r = D(x_r)
            d_f = D(x_f_det)
            loss_d = gan_loss.d_loss(d_r, d_f) + pen_fn(x_r, x_f_det, d_r, d_f)
            loss_d.backward()
            saved_d_grads = [None if p.grad is None else p.grad.detach().clone() for p in D.parameters()]

            x_f = G(z_g)
            loss_g = gan_loss.g_loss(D(x_f), D(x_r).detach())
            loss_g.backward()
            for p, sg in zip(D.parameters(), saved_d_grads):
                p.grad = sg
            opt_d.step()
            opt_g.step()
        else:
            opt_d.zero_grad(set_to_none=True)
            x_f_det = G(z_g).detach()
            d_r = D(x_r)
            d_f = D(x_f_det)
            loss_d = gan_loss.d_loss(d_r, d_f) + pen_fn(x_r, x_f_det, d_r, d_f)
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            for p in D.parameters():
                p.requires_grad_(False)
            x_f = G(z_g)
            loss_g = gan_loss.g_loss(D(x_f), D(x_r).detach())
            loss_g.backward()
            for p in D.parameters():
                p.requires_grad_(True)
            opt_g.step()

        if step >= steps - tail_window:
            with torch.no_grad():
                z_ev = torch.randn(256, 2, generator=gen_rng).to(dev)
                x_ev = G(z_ev)
                normal_mean_hist.append(float(x_ev[:, 1].mean().item()))
                normal_std_hist.append(float(x_ev[:, 1].std().item()))
                normal_rms_hist.append(float(x_ev[:, 1].square().mean().sqrt().item()))
                mean_d_hist.append(abs(float(0.5 * (D(x_r).mean() + D(x_ev).mean()).item())))
            lmed, lmax, sln = _eval_critic_grad_stats(D, torch.cat((x_r[:32], x_ev[:32]), dim=0))
            lip_med_hist.append(lmed)
            lip_max_hist.append(lmax)
            slope_n_hist.append(sln)

    tail_m = np.array(normal_mean_hist, dtype=np.float64)
    tail_s = np.array(normal_std_hist, dtype=np.float64)
    tail_r = np.array(normal_rms_hist, dtype=np.float64)
    tail_sl = np.array(slope_n_hist, dtype=np.float64)
    signs = np.sign(tail_m)
    sign_flips = int(np.sum(signs[1:] * signs[:-1] < 0))
    n_params_g = sum(p.numel() for p in G.parameters())
    n_params_d = sum(p.numel() for p in D.parameters())
    return {
        "penalty_kind": penalty_kind,
        "loss": loss_name,
        "betas": list(betas),
        "simultaneous": simultaneous,
        "params_g": n_params_g,
        "params_d": n_params_d,
        "capacity_ratio_rho": n_params_g / max(1, n_params_d),
        "tail_normal_abs_mean": float(np.mean(np.abs(tail_m))),
        "tail_normal_std_mean": float(np.mean(tail_s)),
        "tail_normal_rms_mean": float(np.mean(tail_r)),
        "tail_normal_mean_std": float(np.std(tail_m)),
        "tail_normal_mean_ptp": float(np.ptp(tail_m)),
        "tail_sign_flips": sign_flips,
        "tail_critic_normal_slope_mean": float(np.mean(tail_sl)),
        "tail_lip_med_mean": float(np.mean(lip_med_hist)),
        "tail_lip_max_max": float(np.max(lip_max_hist)),
        "tail_abs_mean_d_mean": float(np.mean(mean_d_hist)),
        "final_ema_func_gain": float(opt_g._ema_func_gain),
        "final_stiff_gate": float(opt_g.current_stiff_gate()),
        "ns_2d_activations": int(opt_g.ns_2d_activations),
        "probe_consumptions": int(opt_g.probe_consumptions),
    }


def run_nonlinear_suite(device: str = "cpu") -> List[Dict[str, Any]]:
    """Run Part 3B nonlinear 2D MLP minimax experiments with actual ParticleGAN UnifiedV12 + penalties."""
    configs = [
        ("bcap_hinge_gauge", "wasserstein", (0.0, 0.999), False),
        ("unified_v11_no_floor", "wasserstein", (0.0, 0.999), False),
        ("unified_v11_default", "wasserstein", (0.0, 0.999), False),
        ("minimal_born_infeld", "wasserstein", (0.0, 0.999), False),
        ("bcap_hinge_gauge", "non_saturating", (0.0, 0.999), False),
        ("unified_v11_no_floor", "non_saturating", (0.0, 0.999), False),
        ("unified_v11_default", "non_saturating", (0.0, 0.999), False),
        ("minimal_born_infeld", "non_saturating", (0.0, 0.999), False),
        ("bcap_hinge_gauge", "wasserstein", (0.9, 0.999), False),
        ("bcap_hinge_gauge", "wasserstein", (0.0, 0.999), True),
    ]
    return [
        run_nonlinear_mlp_game(
            penalty_kind=pen_k,
            loss_name=loss_n,
            betas=betas,
            simultaneous=sim,
            steps=1000,
            device=device,
        )
        for pen_k, loss_n, betas, sim in configs
    ]


def main() -> None:
    t0 = time.time()
    print(f"particlegan.__file__ = {particlegan.__file__}")
    print("\n=== PART 1: Linear Game Jacobian, Non-Normality & Momentum Theory ===")
    p1 = analyze_linear_game_dynamics()
    print("Sim vs Alt (eta_g=0.014, eta_d=0.021):")
    for row in p1["sim_vs_alt"]:
        if (row["eta_g"], row["eta_d"]) == (0.014, 0.021):
            print(
                f"  mu_d={row['mu_d']:5.2f} | J_eigs=({row['J_eig_real_min']:.4f} +/- {row['J_eig_imag_max']:.4f}i) "
                f"| rho_sim={row['rho_sim']:.6f} | rho_alt={row['rho_alt']:.6f} | det_alt={row['det_alt']:.6f}"
            )
    print("\nNon-normality & Kreiss amplification (mu_d=0.5, eta=0.1, IDENTICAL eigenvalues across rho):")
    for row in p1["non_normal"]:
        print(
            f"  rho={row['rho_capacity']:5.1f} | ||JJ^T-J^TJ||_F={row['departure_from_normality']:10.2f} "
            f"| cond(V)={row['eigvec_cond']:7.2f} | spec_rad={row['spectral_radius']:.5f} "
            f"| peak||M^k||={row['peak_transient_norm']:7.2f} (@k={int(row['peak_step'])}) "
            f"| Kreiss>={row['kreiss_constant']:6.2f}"
        )
    print(
        f"  Jordan(mu_d=2): dep={p1['jordan_critical']['departure_from_normality']:.2f} "
        f"| spec_rad={p1['jordan_critical']['spectral_radius']:.5f} "
        f"| peak||M^k||={p1['jordan_critical']['peak_transient_norm']:.3f} (@k={int(p1['jordan_critical']['peak_step'])})"
    )
    print(
        f"  ShiftedJordan(50): dep={p1['jordan_shifted_50']['departure_from_normality']:.1f} "
        f"| spec_rad={p1['jordan_shifted_50']['spectral_radius']:.5f} "
        f"| peak||M^k||={p1['jordan_shifted_50']['peak_transient_norm']:.2f} (@k={int(p1['jordan_shifted_50']['peak_step'])}) "
        f"| Kreiss>={p1['jordan_shifted_50']['kreiss_constant']:.2f}"
    )
    print("\nMomentum Schur-Cohn & Alternating Spectrum:")
    for row in p1["momentum_table"]:
        print(
            f"  mu_d={row['mu_d']:4.2f} beta1={row['beta1']:+5.2f} | eta_max={row['eta_max_closed']:.6f} "
            f"| rho_sim(.014)={row['rho_sim_eta0014']:.6f} | rho_alt(v12)={row['rho_alt_v12']:.6f}"
        )
    print("\nOGDA vs Zero-Momentum (eta=0.05):")
    for row in p1["ogda_table"]:
        print(
            f"  mu_d={row['mu_d']:4.2f} lam={row['lam']:3.1f} | rho_ogda={row['rho_ogda_eta005']:.6f} "
            f"| noise_var_mult={row['iid_noise_variance_mult']:.2f}x"
        )

    print("\n=== PART 2: Exact Local Curvature Audit of ParticleGAN Law-2 Penalties ===")
    p2 = audit_particlegan_penalties()
    for name, st in p2.items():
        print(
            f"  {name:24s} | d2R/dw2(0)={st['d2R_dw2_at_0']:12.6e} | d2R/db2(0)={st['d2R_db2_at_0']:12.6e} "
            f"| dR/dw(0.5)={st['dR_dw_at_0.5']:12.6e}"
        )

    print("\n=== PART 3A: Analytic Counterexamples (All 4 Operational Laws Satisfied) ===")
    p3a = run_analytic_counterexamples()
    for group_name, rows in p3a.items():
        print(f"\n  [{group_name}]")
        for r in rows:
            tag = f"{r['mode']}"
            if r["delta_g_max"] is not None:
                tag += f"(dg<={r['delta_g_max']})"
            if r["ogda_lam"] is not None:
                tag += f"(lam={r['ogda_lam']})"
            extra = (
                f" ema_fg={r['final_ema_func_gain']:.3f} stiff={r['final_stiff_gate']:.3f}"
                if "final_ema_func_gain" in r
                else ""
            )
            print(
                f"    {tag:24s} pen={r['mu_d_penalty']:25s} | r_tail={r['r_tail_mean']:.5f} "
                f"(tang={r['tang_coord_tail_rms']:.4f}, norm={r['norm_coord_tail_rms']:.4f}) "
                f"| psi_rms={r['psi_tail_rms']:.4f} | maxLip={r['max_lip_observed']:.4f} "
                f"| clamps={r['clamp_count']}{extra}"
            )

    print("\n=== PART 3B: Nonlinear 2D MLP Minimax Game (Actual ParticleGAN UnifiedV12 + Penalties) ===")
    p3b = run_nonlinear_suite(device="cpu")
    for r in p3b:
        print(
            f"  {r['penalty_kind']:21s} {r['loss']:14s} b1={r['betas'][0]:.1f} sim={str(r['simultaneous']):5s} "
            f"| |mean_n|={r['tail_normal_abs_mean']:.4f} ptp={r['tail_normal_mean_ptp']:.4f} std_n={r['tail_normal_std_mean']:.4f} "
            f"flips={r['tail_sign_flips']:2d} | slope_n={r['tail_critic_normal_slope_mean']:.4f} "
            f"| lip_med={r['tail_lip_med_mean']:.3f} lip_max={r['tail_lip_max_max']:.3f} "
            f"| |E[D]|={r['tail_abs_mean_d_mean']:.4f} | ns2d={r['ns_2d_activations']} probes={r['probe_consumptions']}"
        )

    print(f"\nCompleted P3 audit suite in {time.time() - t0:.2f}s.")
    _ = json.dumps({"part1": p1, "part2": p2, "part3a": p3a, "part3b": p3b})


if __name__ == "__main__":
    main()
