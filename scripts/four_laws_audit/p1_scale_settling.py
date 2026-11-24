#!/usr/bin/env python3
"""P1 Adversarial Mathematical & Empirical Audit of Law 4.2 (Scale Invariance vs. Continuous Settling).

Reproducible five-regime dynamical benchmark comparing:
  (i)   Memoryless degree-0 operators (Normalized SGD, SignSGD)
  (ii)  Fixed-epsilon regularized operators (Adam eps=1e-8, Fixed-Tikhonov eps=0.0061)
  (iii) Actual ParticleGAN optimizers (UnifiedV12Optimizer, UnifiedV11EquilibratedOptimizer,
        T4EquilibratedOptimizer, MinimalGeneratorOptimizer, NashV3GeneratorAdam,
        and UnifiedV12Optimizer in stiff/coherence-gated mode)
  (iv)  Pure stateful reference mechanisms isolating each mathematical theorem:
        monotonic peak_rms (gamma=1.0), exponentially decaying peak_rms (gamma=0.9995),
        sliding-window peak_rms (W=50), and dynamic-1e-38 momentum coherence (||m||/||v||).

All experiments use protocol seed 0 and exact deterministic execution.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import particlegan
from particlegan._minimal.dynamics import MinimalGeneratorOptimizer
from particlegan._nash_v3.optim import NashV3GeneratorAdam
from particlegan._t4_arch.optimizer import T4EquilibratedOptimizer
from particlegan._unified_v11.optimizer import UnifiedV11EquilibratedOptimizer
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer


# ---------------------------------------------------------------------------
# Clean Reference Dynamical Operators (isolating Theorems 1-4)
# ---------------------------------------------------------------------------

class ReferenceDynamicalStep:
    """Minimal stateful/memoryless step operator on a list of parameter tensors."""

    def __init__(self, params: List[torch.Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.last_coherence: float = 1.0
        self.last_peak_rms: float = 0.0
        self.last_step_norm: float = 0.0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class MemorylessNormalizedSGD(ReferenceDynamicalStep):
    """Exact memoryless positive 0-homogeneous operator U(g) = g / ||g||_2 (Theorem 1)."""

    @torch.no_grad()
    def step(self) -> None:
        step_sq = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            gf = p.grad.detach().float()
            g_max = gf.abs().amax()
            if float(g_max.item()) == 0.0:
                continue
            g_u = gf / g_max
            g_n = g_u.norm()
            if float(g_n.item()) == 0.0:
                continue
            delta = (g_u / g_n) * (-self.lr)
            p.add_(delta.to(p.dtype))
            step_sq += float(delta.square().sum().item())
        self.last_step_norm = math.sqrt(step_sq)


class MemorylessSignSGD(ReferenceDynamicalStep):
    """Exact memoryless coordinatewise sign operator U(g) = sign(g) / sqrt(d)."""

    @torch.no_grad()
    def step(self) -> None:
        step_sq = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            gf = p.grad.detach().float()
            s = torch.sign(gf) / math.sqrt(float(max(1, p.numel())))
            delta = s * (-self.lr)
            p.add_(delta.to(p.dtype))
            step_sq += float(delta.square().sum().item())
        self.last_step_norm = math.sqrt(step_sq)


class FixedEpsTikhonovSGD(ReferenceDynamicalStep):
    """Fixed-epsilon Tikhonov normalized operator U_eps(g) = g / sqrt(||g||^2 + eps^2) (Theorem 2)."""

    def __init__(self, params: List[torch.Tensor], lr: float, eps: float = 0.0061) -> None:
        super().__init__(params, lr)
        self.eps = float(eps)

    @torch.no_grad()
    def step(self) -> None:
        step_sq = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            gf = p.grad.detach().float()
            g_max = gf.abs().amax()
            sc = max(float(g_max.item()), self.eps)
            if sc == 0.0:
                continue
            g_u = gf / sc
            g_n = float(g_u.norm().item())
            denom = math.hypot(g_n, self.eps / sc)
            if denom == 0.0:
                continue
            delta = (g_u / denom) * (-self.lr)
            p.add_(delta.to(p.dtype))
            step_sq += float(delta.square().sum().item())
        self.last_step_norm = math.sqrt(step_sq)


class DynamicPeakRMSSGD(ReferenceDynamicalStep):
    """Pure uncapped trajectory degree-0 peak_rms Tikhonov operator (Theorems 3a, 3b, 3c).

    Uses factored 1e-38-safe gradient norm scaling so g_t -> c * g_t cancels down to
    FP32 normal limits without any fixed s_base cap or additive eps floor.
    Supports:
      - Monotonic peak: decay=1.0, window=None (Theorem 3a)
      - Exponentially decaying peak: decay=0.9995, window=None (Theorem 3b)
      - Sliding-window peak: window=50 (Theorem 3c)
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float,
        *,
        alpha: float = 0.50,
        decay: float = 1.0,
        window: Optional[int] = None,
    ) -> None:
        super().__init__(params, lr)
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.window = window
        self._peaks: Dict[int, float] = {}
        self._windows: Dict[int, deque[float]] = {}

    @torch.no_grad()
    def step(self) -> None:
        step_sq = 0.0
        last_pk = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            gf = p.grad.detach().float()
            g_max = float(gf.abs().amax().item())
            if not math.isfinite(g_max) or g_max <= 1e-38:
                continue
            g_u = gf / g_max
            u_norm = float(g_u.norm().item())
            g_norm = u_norm * g_max  # Underflow-safe down to ~1e-38
            pid = id(p)
            if self.window is not None:
                buf = self._windows.setdefault(pid, deque(maxlen=self.window))
                buf.append(g_norm)
                peak = max(buf)
            else:
                peak_prev = self._peaks.get(pid, 0.0)
                peak = max(peak_prev * self.decay, g_norm)
                self._peaks[pid] = peak
            last_pk = peak
            # Scale denominator in g_max units to stay underflow-safe down to 1e-38:
            s_scaled = (self.alpha * peak) / g_max
            denom = math.hypot(u_norm, s_scaled)
            if denom <= 1e-30:
                continue
            delta = (g_u / denom) * (-self.lr)
            p.add_(delta.to(p.dtype))
            step_sq += float(delta.square().sum().item())
        self.last_peak_rms = last_pk
        self.last_step_norm = math.sqrt(step_sq)


class DynamicCoherenceSGD(ReferenceDynamicalStep):
    """Pure underflow-safe (1e-38) dynamic scale-normalized momentum-coherence operator (Theorem 3d).

    Matches particlegan's factored (m_unit, m_scale), (v_rms, v_scale) recurrence and
    coherence = clamp(||m_hat||_F / ||v_hat||_F, 0, 1) without s_base or fixed eps.
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float,
        *,
        beta1: float = 0.80,
        beta2: float = 0.97,
    ) -> None:
        super().__init__(params, lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self._state: Dict[int, Dict[str, Any]] = {}

    @torch.no_grad()
    def step(self) -> None:
        step_sq = 0.0
        cohs: List[float] = []
        for p in self.params:
            if p.grad is None:
                continue
            gf = p.grad.detach().float()
            g_max = gf.abs().amax()
            if not bool(torch.isfinite(g_max).item()) or float(g_max.item()) <= 1e-38:
                continue
            pid = id(p)
            st = self._state.get(pid)
            if st is None:
                st = {
                    "step": 0,
                    "m_unit": torch.zeros_like(p, dtype=torch.float32),
                    "m_scale": torch.zeros((), device=p.device, dtype=torch.float32),
                    "v_rms": torch.zeros_like(p, dtype=torch.float32),
                    "v_scale": torch.zeros((), device=p.device, dtype=torch.float32),
                }
                self._state[pid] = st
            st["step"] += 1
            t = st["step"]
            m_prev_sc = st["m_scale"]
            v_prev_sc = st["v_scale"]
            sc = torch.maximum(torch.maximum(m_prev_sc, v_prev_sc), g_max).clamp_min(1e-38)
            m_res = st["m_unit"] * (m_prev_sc / sc)
            v_res = st["v_rms"] * (v_prev_sc / sc)
            g_sc = gf / sc
            m_next = m_res.mul(self.beta1).add(g_sc, alpha=1.0 - self.beta1)
            v_next = (self.beta2 * v_res.square() + (1.0 - self.beta2) * g_sc.square()).sqrt()

            new_m_sc = m_next.abs().amax()
            safe_m = torch.where(new_m_sc > 1e-38, new_m_sc, torch.ones_like(new_m_sc))
            st["m_unit"].copy_(torch.where(new_m_sc > 1e-38, m_next / safe_m, torch.zeros_like(m_next)))
            st["m_scale"].copy_(torch.where(new_m_sc > 1e-38, new_m_sc, torch.zeros_like(new_m_sc)) * sc)

            new_v_sc = v_next.abs().amax()
            safe_v = torch.where(new_v_sc > 1e-38, new_v_sc, torch.ones_like(new_v_sc))
            st["v_rms"].copy_(torch.where(new_v_sc > 1e-38, v_next / safe_v, torch.zeros_like(v_next)))
            st["v_scale"].copy_(torch.where(new_v_sc > 1e-38, new_v_sc, torch.zeros_like(new_v_sc)) * sc)

            m_hat = m_next / max(1.0 - (self.beta1 ** t), 1e-8)
            v_hat = v_next / max(math.sqrt(1.0 - (self.beta2 ** t)), 1e-8)
            m_norm = float(m_hat.norm().item())
            v_norm = float(v_hat.norm().item())
            coh = min(1.0, max(0.0, m_norm / max(v_norm, 1e-30))) if (m_norm > 1e-30 and v_norm > 1e-30) else 0.0
            cohs.append(coh)
            if m_norm > 1e-30:
                delta = (m_hat / m_norm) * (-self.lr * coh)
                p.add_(delta.to(p.dtype))
                step_sq += float(delta.square().sum().item())
        self.last_coherence = sum(cohs) / len(cohs) if cohs else 0.0
        self.last_step_norm = math.sqrt(step_sq)


# ---------------------------------------------------------------------------
# Unified Factory for Both Reference Operators and Actual ParticleGAN Classes
# ---------------------------------------------------------------------------

def make_optimizer(
    name: str,
    params: List[torch.Tensor],
    *,
    lr: float = 0.02,
    extra_dummy_for_t4: bool = True,
) -> Any:
    """Instantiate a reference operator or an actual ParticleGAN optimizer class.

    For `pg_t4_equil`, an untouched second 2D matrix can be registered in the param group
    so `L_count == 2` activates the multi-layer momentum/coherence regime (`b1_eff = 0.80`)
    exactly as in real neural networks (without ever receiving gradients).
    """
    if name == "memless_norm_sgd":
        return MemorylessNormalizedSGD(params, lr=lr)
    if name == "memless_sign_sgd":
        return MemorylessSignSGD(params, lr=lr)
    if name == "fixed_eps_adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.0, 0.999), eps=1e-8)
    if name == "fixed_eps_tikh":
        return FixedEpsTikhonovSGD(params, lr=lr, eps=0.0061)
    if name == "pure_peak_mono":
        return DynamicPeakRMSSGD(params, lr=lr, alpha=0.50, decay=1.0, window=None)
    if name == "pure_peak_decay":
        return DynamicPeakRMSSGD(params, lr=lr, alpha=0.50, decay=0.9995, window=None)
    if name == "pure_peak_window":
        return DynamicPeakRMSSGD(params, lr=lr, alpha=0.50, decay=1.0, window=50)
    if name == "pure_coherence":
        return DynamicCoherenceSGD(params, lr=lr, beta1=0.80, beta2=0.97)
    if name == "pg_unified_v12":
        return UnifiedV12Optimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.0061,
        )
    if name == "pg_unified_v12_nosmooth":
        return UnifiedV12Optimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.0,
        )
    if name == "pg_unified_v12_stiff":
        opt = UnifiedV12Optimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.0061,
        )
        # Force stiff_gate = 1.0 (ema_fg = 0.25 <= 0.30) to directly exercise
        # UnifiedV12Optimizer's actual momentum-coherence cooling branch (b1_eff = 0.75, s_base = 0.0).
        opt._ema_func_gain = 0.25
        return opt
    if name == "pg_unified_v11":
        return UnifiedV11EquilibratedOptimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.0061,
        )
    if name == "pg_t4_equil":
        group_params = list(params)
        if extra_dummy_for_t4 and sum(1 for p in group_params if p.ndim >= 2) < 2:
            # Dummy inactive 2D tensor ensures L_count >= 2 (multi-layer b1_eff = 0.80)
            dummy = torch.zeros((2, 2), dtype=params[0].dtype, device=params[0].device)
            group_params.append(dummy)
        return T4EquilibratedOptimizer(
            [{"params": group_params, "role": "generator"}],
            lr=lr,
            betas=(0.80, 0.97),
            eps=1e-8,
        )
    if name == "pg_minimal":
        return MinimalGeneratorOptimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
        )
    if name == "pg_nash_v3":
        return NashV3GeneratorAdam(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
        )
    raise ValueError(f"unknown optimizer arm: {name!r}")


ALL_ARMS: Tuple[str, ...] = (
    "memless_norm_sgd",
    "memless_sign_sgd",
    "fixed_eps_adam",
    "fixed_eps_tikh",
    "pure_peak_mono",
    "pure_peak_decay",
    "pure_peak_window",
    "pure_coherence",
    "pg_unified_v12",
    "pg_unified_v12_stiff",
    "pg_unified_v11",
    "pg_t4_equil",
    "pg_minimal",
    "pg_nash_v3",
)


# ---------------------------------------------------------------------------
# Regime 1: Global Objective Rescaling Invariance + Underflow Breakpoints
# ---------------------------------------------------------------------------

def run_regime1_rescaling(
    arms: Tuple[str, ...] = ALL_ARMS,
    scales: Tuple[float, ...] = (1e-12, 1e-6, 1.0, 1e6, 1e12),
    underflow_scales: Tuple[float, ...] = (1e-12, 1e-18, 1e-22, 1e-25, 1e-35, 1e-37, 1e-39, 1e-40),
    steps: int = 40,
) -> Dict[str, Any]:
    """Test exact trajectory invariance under L(W) -> c * L(W) across 24+ orders of magnitude."""
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn((8, 8), generator=gen, dtype=torch.float32) * 0.5
    # Well-conditioned symmetric positive-definite curvature H (eigvals in [0.5, 2.0])
    q, _ = torch.linalg.qr(torch.randn((8, 8), generator=gen, dtype=torch.float32))
    eigs = torch.linspace(0.5, 2.0, 8, dtype=torch.float32)
    h_mat = q @ torch.diag(eigs) @ q.T
    target = torch.randn((8, 8), generator=gen, dtype=torch.float32) * 0.5

    def _run_one(arm: str, c: float, n_steps: int, shape_1d: bool = False) -> Tuple[torch.Tensor, float]:
        if shape_1d:
            p = w0[0].clone().detach().requires_grad_(True)
            t_vec = target[0]
            opt = make_optimizer(arm, [p], lr=0.02, extra_dummy_for_t4=False)
            total_move = 0.0
            for _ in range(n_steps):
                opt.zero_grad()
                # Direct analytic gradient to avoid loss-scalar underflow at c=1e-40
                g = (p.detach() - t_vec) * float(c)
                p.grad = g
                prev = p.detach().clone()
                opt.step()
                total_move += float((p.detach() - prev).norm().item())
            return p.detach().clone(), total_move
        p = w0.clone().detach().requires_grad_(True)
        opt = make_optimizer(arm, [p], lr=0.02)
        total_move = 0.0
        for _ in range(n_steps):
            opt.zero_grad()
            # Analytic gradient c * H @ (W - target) avoids intermediate squaring overflow at c=1e12
            p.grad = (h_mat @ (p.detach() - target)) * float(c)
            prev = p.detach().clone()
            opt.step()
            total_move += float((p.detach() - prev).norm().item())
        return p.detach().clone(), total_move

    matrix_results: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        finals: Dict[float, torch.Tensor] = {}
        moves: Dict[float, float] = {}
        for c in scales:
            w_end, mv = _run_one(arm, c, steps, shape_1d=False)
            finals[c] = w_end
            moves[c] = mv
        ref_1 = finals[1.0]
        ref_1e_6 = finals[1e-6]
        diff_vs_1 = {f"{c:.0e}": float((finals[c] - ref_1).abs().max().item()) for c in scales}
        diff_vs_1e_6 = {f"{c:.0e}": float((finals[c] - ref_1e_6).abs().max().item()) for c in scales}
        max_diff_all = max(diff_vs_1.values())
        max_diff_small = float((finals[1e-12] - ref_1e_6).abs().max().item())
        matrix_results[arm] = {
            "diff_vs_1": diff_vs_1,
            "diff_vs_1e-6": diff_vs_1e_6,
            "max_diff_all": max_diff_all,
            "diff_1e-12_vs_1e-6": max_diff_small,
            "moves": {f"{c:.0e}": moves[c] for c in scales},
        }

    # Extreme underflow & 1e-38 breakpoint probe on 2D matrices (single-step + 10-step displacement)
    underflow_results: Dict[str, Dict[str, Any]] = {}
    probe_arms = (
        "memless_norm_sgd",
        "fixed_eps_adam",
        "pure_peak_mono",
        "pg_unified_v12",
        "pg_unified_v11",
        "pg_t4_equil",
        "pg_minimal",
        "pg_nash_v3",
    )
    for arm in probe_arms:
        step1_moves: Dict[str, float] = {}
        peak_after: Dict[str, float] = {}
        for c in underflow_scales:
            p = w0.clone().detach().requires_grad_(True)
            opt = make_optimizer(arm, [p], lr=0.02)
            p.grad = (h_mat @ (p.detach() - target)) * float(c)
            prev = p.detach().clone()
            opt.step()
            step1_moves[f"{c:.0e}"] = float((p.detach() - prev).norm().item())
            st = getattr(opt, "state", {}).get(p, {})
            if isinstance(st, dict):
                if "peak_rms_val" in st:
                    peak_after[f"{c:.0e}"] = float(st["peak_rms_val"])
                elif "peak_rms" in st:
                    peak_after[f"{c:.0e}"] = float(st["peak_rms"].item())
        underflow_results[arm] = {
            "step1_move": step1_moves,
            "peak_after_step1": peak_after,
        }

    # 1D vector probe at c=1e-12 vs c=1e-6 (exposes unified_v11 clamp_min(eps=1e-8) on 1D params)
    vec1d_results: Dict[str, Dict[str, float]] = {}
    for arm in ("pure_peak_mono", "pg_unified_v12", "pg_unified_v11", "pg_minimal"):
        _, mv_1e_6 = _run_one(arm, 1e-6, 10, shape_1d=True)
        _, mv_1e_12 = _run_one(arm, 1e-12, 10, shape_1d=True)
        vec1d_results[arm] = {
            "move_1e-6": mv_1e_6,
            "move_1e-12": mv_1e_12,
            "ratio_1e-12_over_1e-6": mv_1e_12 / max(mv_1e_6, 1e-30),
        }

    return {
        "matrix_24decades": matrix_results,
        "underflow_probe": underflow_results,
        "vec1d_clamp_probe": vec1d_results,
    }


# ---------------------------------------------------------------------------
# Regime 2: Genuine Deterministic Convergence, Rail Escape, Near-Eq & Slow Decay
# ---------------------------------------------------------------------------

def run_regime2_deterministic(arms: Tuple[str, ...] = ALL_ARMS) -> Dict[str, Any]:
    """Test (2a) quadratic settling, (2b) 1e-12 tanh rail escape + settling,
    (2c) near-equilibrium initial kick (Obstruction 4.1), and (2d) algebraic 1/t decay (Theorem 3b).
    """
    gen = torch.Generator().manual_seed(0)
    w0_quad = torch.randn((4, 4), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(torch.randn((4, 4), generator=gen, dtype=torch.float32))
    eigs_q = torch.linspace(1.0, 10.0, 4, dtype=torch.float32)
    h_quad = q @ torch.diag(eigs_q) @ q.T

    # (2a) Quadratic convergence & terminal chatter (1000 steps)
    quad_results: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        p = w0_quad.clone().detach().requires_grad_(True)
        opt = make_optimizer(arm, [p], lr=0.02)
        tail_steps: List[float] = []
        for t in range(1000):
            opt.zero_grad()
            p.grad = h_quad @ p.detach()
            prev = p.detach().clone()
            opt.step()
            if t >= 950:
                tail_steps.append(float((p.detach() - prev).norm().item()))
        quad_results[arm] = {
            "final_err": float(p.detach().norm().item()),
            "tail_mean_step": sum(tail_steps) / len(tail_steps),
            "tail_max_step": max(tail_steps),
        }

    # (2b) Deep tanh rail escape from sech^2(14) = 2.76e-12 -> O(1) -> settle at 0
    # Tested across objective scales c in {1e-6, 1.0, 1e6} to expose s_base=0.0061 settling breakdown!
    rail_results: Dict[str, Dict[str, Any]] = {}
    a0 = 14.0
    # Verify initial gradient magnitude analytically: tanh(14) * sech^2(14) = 2.755e-12
    init_grad_val = float(math.tanh(a0) / (math.cosh(a0) ** 2))
    for arm in arms:
        per_scale: Dict[str, Dict[str, float]] = {}
        for c in (1e-6, 1.0, 1e6):
            # Isotropic 4x4 diagonal matrix W_0 = 14.0 * I_4: full-rank 4, every diagonal entry sits
            # identically on the 2.76e-12 tanh rail (off-diagonals stay identically 0, eliminating
            # asymmetric sequential-entry escape confounds while exercising 2D NS/MP/elementwise paths).
            p = (torch.eye(4, dtype=torch.float32) * a0).detach().requires_grad_(True)
            opt = make_optimizer(arm, [p], lr=0.08)
            escaped_step = -1
            max_grad_seen = 0.0
            tail_steps: List[float] = []
            for t in range(1, 1001):
                opt.zero_grad()
                x = p.detach().double()
                # Compute exact tanh(x) * sech^2(x) in float64 then cast to float32 so sech^2(14) is exact
                th = torch.tanh(x)
                sech2 = 1.0 - th * th
                g = (th * sech2 * float(c)).float()
                g_rms = float(g.square().mean().sqrt().item())
                if g_rms > max_grad_seen:
                    max_grad_seen = g_rms
                p.grad = g
                prev = p.detach().clone()
                opt.step()
                cur_max = float(p.detach().abs().max().item())
                if escaped_step < 0 and cur_max < 1.0:
                    escaped_step = t
                if t > 950:
                    tail_steps.append(float((p.detach() - prev).norm().item()))
            per_scale[f"{c:.0e}"] = {
                "escaped_step": float(escaped_step),
                "final_max_abs": float(p.detach().abs().max().item()),
                "final_rms": float(p.detach().square().mean().sqrt().item()),
                "tail_mean_step": sum(tail_steps) / len(tail_steps),
                "max_grad_seen": max_grad_seen,
            }
        rail_results[arm] = per_scale

    # (2c) Near-equilibrium start (Obstruction 4.1): W_0 = 1e-6 * unit on L(W) = 0.5 ||W||_F^2
    near_eq_results: Dict[str, Dict[str, float]] = {}
    w_unit = w0_quad / w0_quad.norm()
    w0_tiny = w_unit * 1e-6
    for arm in arms:
        p = w0_tiny.clone().detach().requires_grad_(True)
        opt = make_optimizer(arm, [p], lr=0.02)
        p.grad = p.detach().clone()
        opt.step()
        err1 = float(p.detach().norm().item())
        for _ in range(199):
            opt.zero_grad()
            p.grad = p.detach().clone()
            opt.step()
        err200 = float(p.detach().norm().item())
        near_eq_results[arm] = {
            "init_err": 1e-6,
            "step1_err": err1,
            "step1_amplification": err1 / 1e-6,
            "step200_err": err200,
        }

    # (2d) Sub-exponential algebraic gradient decay g_t = t^{-1} * W_dir (Theorem 3b & 3d)
    slow_decay_results: Dict[str, Dict[str, float]] = {}
    g_dir = w_unit.clone()
    for arm in (
        "memless_norm_sgd",
        "pure_peak_mono",
        "pure_peak_decay",
        "pure_peak_window",
        "pure_coherence",
        "pg_unified_v12",
        "pg_unified_v11",
        "pg_t4_equil",
    ):
        p = torch.zeros((4, 4), dtype=torch.float32, requires_grad=True)
        opt = make_optimizer(arm, [p], lr=0.02)
        step_norms: Dict[int, float] = {}
        for t in range(1, 5001):
            opt.zero_grad()
            # Scale by 1e-4 so unified_v12/v11 stay below s_base=0.0061 (isolating peak_rms behavior!)
            p.grad = g_dir * (1e-4 / float(t))
            prev = p.detach().clone()
            opt.step()
            if t in (1, 100, 1000, 2000, 5000):
                step_norms[t] = float((p.detach() - prev).norm().item())
        slow_decay_results[arm] = {
            f"step_{k}": v for k, v in step_norms.items()
        } | {
            "ratio_5000_over_1": step_norms[5000] / max(step_norms[1], 1e-30),
            "ratio_5000_over_2000": step_norms[5000] / max(step_norms[2000], 1e-30),
        }

    # Fast 25,000-step 1D witness proving pure_peak_decay (gamma=0.9995) reverts all the way to ratio = 1.0!
    p_long = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt_long = DynamicPeakRMSSGD([p_long], lr=0.02, alpha=0.50, decay=0.9995)
    long_steps: Dict[int, float] = {}
    for t in range(1, 25001):
        p_long.grad = torch.tensor([1.0 / float(t)], dtype=torch.float32)
        prev = p_long.detach().clone()
        opt_long.step()
        if t in (1, 2000, 10000, 20000, 25000):
            long_steps[t] = float((p_long.detach() - prev).norm().item())
    slow_decay_results["pure_peak_decay_25k_1d"] = {
        f"step_{k}": v for k, v in long_steps.items()
    } | {
        "ratio_25000_over_1": long_steps[25000] / max(long_steps[1], 1e-30),
        "ratio_2000_over_1": long_steps[2000] / max(long_steps[1], 1e-30),
    }

    return {
        "init_rail_grad": init_grad_val,
        "quadratic": quad_results,
        "tanh_rail_escape": rail_results,
        "near_equilibrium_kick": near_eq_results,
        "algebraic_decay_1_over_t": slow_decay_results,
    }


# ---------------------------------------------------------------------------
# Regime 3: Noisy Stationary Points (Zero-Mean Noise, Constant & Varying Variance)
# ---------------------------------------------------------------------------

def run_regime3_noisy_stationary(arms: Tuple[str, ...] = ALL_ARMS) -> Dict[str, Any]:
    """Test i.i.d. zero-mean Gaussian noise, constant-norm sphere noise, and variance drop."""
    d_mat = (8, 8)
    n_steps = 2500

    # Pre-generate deterministic noise sequences (seed 0) shared across all arms
    gen = torch.Generator().manual_seed(0)
    gauss_noise = torch.randn((n_steps, *d_mat), generator=gen, dtype=torch.float32)
    sphere_noise = gauss_noise / gauss_noise.flatten(1).norm(dim=1).view(n_steps, 1, 1)
    # Varying variance: sigma = 1.0 for steps 0..499, then sigma = 0.01 for steps 500..2499
    # (scaled by 1e-3 overall so unified_v12/v11 stay in the peak_rms regime < s_base)
    var_drop_noise = gauss_noise.clone() * 1e-3
    var_drop_noise[500:] *= 0.01

    def _eval_noise(seq: torch.Tensor, scale: float = 1.0) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for arm in arms:
            p = torch.zeros(d_mat, dtype=torch.float32, requires_grad=True)
            opt = make_optimizer(arm, [p], lr=0.02)
            early_steps: List[float] = []
            tail_steps: List[float] = []
            for t in range(len(seq)):
                opt.zero_grad()
                p.grad = seq[t] * scale
                prev = p.detach().clone()
                opt.step()
                dn = float((p.detach() - prev).norm().item())
                if 50 <= t < 150:
                    early_steps.append(dn)
                if t >= len(seq) - 200:
                    tail_steps.append(dn)
            entry = {
                "early_mean_step": sum(early_steps) / len(early_steps),
                "tail_mean_step": sum(tail_steps) / len(tail_steps),
                "tail_over_early": (sum(tail_steps) / len(tail_steps)) / max(sum(early_steps) / len(early_steps), 1e-30),
                "final_drift_norm": float(p.detach().norm().item()),
            }
            if hasattr(opt, "last_coherence"):
                entry["final_coherence"] = float(opt.last_coherence)
            out[arm] = entry
        return out

    # Also measure exact empirical coherence vs theoretical sqrt((1-b1)/(1+b1)) for b1 in {0.75, 0.80, 0.90}
    coh_theory_check: Dict[str, Dict[str, float]] = {}
    for b1 in (0.75, 0.80, 0.90):
        p = torch.zeros(d_mat, dtype=torch.float32, requires_grad=True)
        opt = DynamicCoherenceSGD([p], lr=0.02, beta1=b1, beta2=0.999)
        cohs: List[float] = []
        for t in range(n_steps):
            p.grad = gauss_noise[t]
            opt.step()
            if t >= n_steps - 500:
                cohs.append(opt.last_coherence)
        theo = math.sqrt((1.0 - b1) / (1.0 + b1))
        emp = sum(cohs) / len(cohs)
        coh_theory_check[f"beta1={b1:.2f}"] = {
            "theoretical_floor": theo,
            "empirical_mean": emp,
            "abs_error": abs(emp - theo),
        }

    return {
        "gaussian_const_var_scale1": _eval_noise(gauss_noise, scale=1.0),
        "gaussian_const_var_scale1e-3": _eval_noise(gauss_noise, scale=1e-3),
        "sphere_const_norm_scale1e-3": _eval_noise(sphere_noise, scale=1e-3),
        "variance_drop_100x": _eval_noise(var_drop_noise, scale=1.0),
        "coherence_iid_floor_verification": coh_theory_check,
    }


# ---------------------------------------------------------------------------
# Regime 4: Non-Convex Saddle Points, Single-Spike Freezing & Distribution Shift
# ---------------------------------------------------------------------------

def run_regime4_adversarial_shift(arms: Tuple[str, ...] = ALL_ARMS) -> Dict[str, Any]:
    """CRITICAL ADVERSARIAL TESTS:
      (4a) Post-spike permanent ratcheting (single 1e4x gradient spike at step 20)
           at both scale c=1.0 and c=1e-6 (below unified_v12's s_base cap!).
      (4b) Post-settling distribution shift with 1000x curvature/scale drop.
      (4c) Non-convex saddle escape L(x,y) = 0.5 x^2 - 0.5 y^2 + 0.25 y^4 from (1.0, 1e-8).
    """
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn((4, 4), generator=gen, dtype=torch.float32)
    target = torch.zeros((4, 4), dtype=torch.float32)
    spike_dir = torch.randn((4, 4), generator=gen, dtype=torch.float32)
    spike_dir = spike_dir / spike_dir.norm()

    # (4a) Single transient spike (1e4x) at step 20 -> resume normal quadratic for 1500 steps
    spike_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scale_label, c in (("c=1e-6", 1e-6), ("c=1e+0", 1.0)):
        per_arm: Dict[str, Dict[str, float]] = {}
        for arm in arms:
            p = w0.clone().detach().requires_grad_(True)
            opt = make_optimizer(arm, [p], lr=0.02)
            pre_spike_step = 0.0
            post_spike_step = 0.0
            step_100_post = 0.0
            for t in range(1, 1521):
                opt.zero_grad()
                g = (p.detach() - target) * float(c)
                if t == 20:
                    # Single outlier gradient spike of magnitude 1e4 * c
                    g = spike_dir * (1e4 * float(c))
                p.grad = g
                prev = p.detach().clone()
                opt.step()
                dn = float((p.detach() - prev).norm().item())
                if t == 19:
                    pre_spike_step = dn
                elif t == 21:
                    post_spike_step = dn
                elif t == 120:
                    step_100_post = dn
            per_arm[arm] = {
                "pre_spike_step": pre_spike_step,
                "post_spike_step": post_spike_step,
                "post_over_pre": post_spike_step / max(pre_spike_step, 1e-30),
                "step_100_post": step_100_post,
                "final_err": float((p.detach() - target).norm().item()),
                "init_err": float(w0.norm().item()),
                "converged_95pct": float((p.detach() - target).norm().item() < 0.05 * float(w0.norm().item())),
            }
        spike_results[scale_label] = per_arm

    # (4b) Distribution shift with 1000x curvature/scale drop after initial settling
    shift_results: Dict[str, Dict[str, float]] = {}
    t1 = torch.ones((4, 4), dtype=torch.float32) * 0.5
    t2 = -t1
    for arm in arms:
        p = torch.zeros((4, 4), dtype=torch.float32, requires_grad=True)
        opt = make_optimizer(arm, [p], lr=0.02)
        # Phase 1: settle near t1 at scale 1e-4 (inside peak_rms regime for v11/v12)
        for _ in range(400):
            opt.zero_grad()
            p.grad = (p.detach() - t1) * 1e-4
            opt.step()
        err_phase1 = float((p.detach() - t1).norm().item())
        # Phase 2: target shifts to t2 = -t1 AND gradient scale drops 1000x to 1e-7
        for _ in range(1500):
            opt.zero_grad()
            p.grad = (p.detach() - t2) * 1e-7
            opt.step()
        err_phase2 = float((p.detach() - t2).norm().item())
        shift_results[arm] = {
            "phase1_err_to_t1": err_phase1,
            "phase2_err_to_t2": err_phase2,
            "initial_dist_t1_t2": float((t1 - t2).norm().item()),
            "recovered_90pct": float(err_phase2 < 0.1 * float((t1 - t2).norm().item())),
        }

    # (4c) Non-convex saddle escape: L(x, y) = 0.5 x^2 - 0.5 y^2 + 0.25 y^4
    # Packed in a single (2, 2) matrix where row 0 = stable x (init x0=5.0), row 1 = unstable y (init 1e-8)
    # Scaled by c=1e-4 so unified_v12 operates inside its peak_rms regime (< s_base).
    # True minimizers have |y| = 1.0, x = 0.0.
    saddle_results: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        p = torch.tensor([[5.0, 5.0], [1e-8, -1e-8]], dtype=torch.float32, requires_grad=True)
        opt = make_optimizer(arm, [p], lr=0.02)
        escape_step = -1
        for t in range(1, 1001):
            opt.zero_grad()
            x = p.detach()[0]
            y = p.detach()[1]
            gx = x * 1e-4
            gy = (-y + y.pow(3)) * 1e-4
            p.grad = torch.stack([gx, gy], dim=0)
            opt.step()
            if escape_step < 0 and float(p.detach()[1].abs().mean().item()) > 0.5:
                escape_step = t
        saddle_results[arm] = {
            "escape_step": float(escape_step),
            "final_x_abs": float(p.detach()[0].abs().mean().item()),
            "final_y_abs": float(p.detach()[1].abs().mean().item()),
            "escaped": float(float(p.detach()[1].abs().mean().item()) > 0.5),
        }

    return {
        "single_spike_1e4x": spike_results,
        "shift_with_1000x_scale_drop": shift_results,
        "nonconvex_saddle_escape": saddle_results,
    }


# ---------------------------------------------------------------------------
# Regime 5: Persistent Adversarial Rotation (Bilinear Minimax Game)
# ---------------------------------------------------------------------------

def run_regime5_adversarial_rotation(arms: Tuple[str, ...] = ALL_ARMS) -> Dict[str, Any]:
    """Test bilinear minimax game min_x max_y x^T y (simultaneous & alternating)
    plus exact closed-form coherence verification on pure harmonic rotation g_t = R(t*w) g_0.
    """
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn((4, 4), generator=gen, dtype=torch.float32)
    x0 = x0 / x0.norm()
    y0 = torch.randn((4, 4), generator=gen, dtype=torch.float32)
    y0 = y0 / y0.norm()

    def _run_bilinear(mode: str, n_steps: int = 600, lr: float = 0.02) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for arm in arms:
            x = x0.clone().detach().requires_grad_(True)
            y = y0.clone().detach().requires_grad_(True)
            opt_x = make_optimizer(arm, [x], lr=lr)
            opt_y = make_optimizer(arm, [y], lr=lr)
            r0 = math.sqrt(float(x.detach().square().sum().item() + y.detach().square().sum().item()))
            max_r = r0
            min_r = r0
            tail_step_x: List[float] = []
            for t in range(n_steps):
                if mode == "simultaneous":
                    # min_x max_y <x, y>: grad_x = y, grad_y = -x (ascent on y = descent on -<x,y>)
                    gx = y.detach().clone()
                    gy = -x.detach().clone()
                    opt_x.zero_grad()
                    opt_y.zero_grad()
                    x.grad = gx
                    y.grad = gy
                    prev_x = x.detach().clone()
                    opt_x.step()
                    opt_y.step()
                else:
                    # Alternating GDA: update x, then compute y grad at updated x
                    opt_x.zero_grad()
                    x.grad = y.detach().clone()
                    prev_x = x.detach().clone()
                    opt_x.step()
                    opt_y.zero_grad()
                    y.grad = -x.detach().clone()
                    opt_y.step()
                r = math.sqrt(float(x.detach().square().sum().item() + y.detach().square().sum().item()))
                if math.isfinite(r):
                    max_r = max(max_r, r)
                    min_r = min(min_r, r)
                if t >= n_steps - 50:
                    tail_step_x.append(float((x.detach() - prev_x).norm().item()))
            r_end = math.sqrt(float(x.detach().square().sum().item() + y.detach().square().sum().item()))
            entry: Dict[str, float] = {
                "r0": r0,
                "r_end": r_end,
                "r_ratio": r_end / r0,
                "r_min": min_r,
                "r_max": max_r,
                "tail_mean_step_x": sum(tail_step_x) / len(tail_step_x),
            }
            if hasattr(opt_x, "last_coherence"):
                entry["final_coherence"] = float(opt_x.last_coherence)
            if hasattr(opt_x, "last_peak_rms"):
                entry["final_peak_rms"] = float(opt_x.last_peak_rms)
            out[arm] = entry
        return out

    # Exact closed-form harmonic rotation coherence verification:
    # g_t = (cos(t*omega), sin(t*omega)) => C_inf = (1 - b1) / sqrt(1 - 2*b1*cos(omega) + b1^2)
    harmonic_check: Dict[str, Dict[str, float]] = {}
    b1 = 0.80
    b2 = 0.97
    for label, omega in (
        ("omega=0.02 (slow game)", 0.02),
        ("omega=0.10", 0.10),
        ("omega=pi/2 (quadrature)", math.pi / 2.0),
        ("omega=pi (period-2 flip)", math.pi),
    ):
        p = torch.zeros(2, dtype=torch.float32, requires_grad=True)
        opt = DynamicCoherenceSGD([p], lr=0.02, beta1=b1, beta2=b2)
        for t in range(1, 1001):
            p.grad = torch.tensor([math.cos(t * omega), math.sin(t * omega)], dtype=torch.float32)
            opt.step()
        theo = (1.0 - b1) / math.sqrt(1.0 - 2.0 * b1 * math.cos(omega) + b1 * b1)
        harmonic_check[label] = {
            "theoretical_coh": theo,
            "empirical_coh": float(opt.last_coherence),
            "abs_error": abs(float(opt.last_coherence) - theo),
        }

    return {
        "bilinear_simultaneous": _run_bilinear("simultaneous", n_steps=600, lr=0.02),
        "bilinear_alternating": _run_bilinear("alternating", n_steps=600, lr=0.02),
        "harmonic_coherence_formula": harmonic_check,
    }


def run_all() -> Dict[str, Any]:
    return {
        "particlegan_file": particlegan.__file__,
        "regime1_rescaling": run_regime1_rescaling(),
        "regime2_deterministic": run_regime2_deterministic(),
        "regime3_noisy_stationary": run_regime3_noisy_stationary(),
        "regime4_adversarial_shift": run_regime4_adversarial_shift(),
        "regime5_adversarial_rotation": run_regime5_adversarial_rotation(),
    }


def _print_summary(res: Dict[str, Any]) -> None:
    print(f"particlegan.__file__ = {res['particlegan_file']}\n")

    print("=== REGIME 1: GLOBAL RESCALING INVARIANCE (2D 8x8 Matrix, 40 steps) ===")
    r1 = res["regime1_rescaling"]["matrix_24decades"]
    print(f"{'Arm':<22} | {'diff(1e-12 vs 1e-6)':>19} | {'diff(1e-6 vs 1)':>15} | {'diff(1e6 vs 1)':>14} | {'max_diff_all':>12}")
    print("-" * 92)
    for arm, d in r1.items():
        print(
            f"{arm:<22} | {d['diff_1e-12_vs_1e-6']:19.3e} | {d['diff_vs_1']['1e-06']:15.3e} | "
            f"{d['diff_vs_1']['1e+06']:14.3e} | {d['max_diff_all']:12.3e}"
        )

    print("\n=== REGIME 1b: EXTREME UNDERFLOW / 1e-38 BREAKPOINT (Step-1 Move Norm) ===")
    u1 = res["regime1_rescaling"]["underflow_probe"]
    u_keys = list(next(iter(u1.values()))["step1_move"].keys())
    print(f"{'Arm':<18} | " + " | ".join(f"{k:>9}" for k in u_keys))
    print("-" * (21 + 12 * len(u_keys)))
    for arm, d in u1.items():
        print(f"{arm:<18} | " + " | ".join(f"{d['step1_move'][k]:9.2e}" for k in u_keys))

    print("\n=== REGIME 1c: 1D PARAMETER CLAMP PROBE (c=1e-12 vs c=1e-6) ===")
    for arm, d in res["regime1_rescaling"]["vec1d_clamp_probe"].items():
        print(f"  {arm:<18}: move(1e-6)={d['move_1e-6']:.4e}, move(1e-12)={d['move_1e-12']:.4e}, ratio={d['ratio_1e-12_over_1e-6']:.4e}")

    print("\n=== REGIME 2: DETERMINISTIC CONVERGENCE & 1e-12 TANH RAIL ESCAPE ===")
    r2q = res["regime2_deterministic"]["quadratic"]
    r2r = res["regime2_deterministic"]["tanh_rail_escape"]
    r2n = res["regime2_deterministic"]["near_equilibrium_kick"]
    print(
        f"{'Arm':<22} | {'quad_err':>9} | {'quad_stp':>9} | {'rail_esc(1)':>11} | "
        f"{'rail_rms(1)':>11} | {'rail_rms(1e6)':>13} | {'near_eq_amp':>11}"
    )
    print("-" * 102)
    for arm in ALL_ARMS:
        print(
            f"{arm:<22} | {r2q[arm]['final_err']:9.2e} | {r2q[arm]['tail_mean_step']:9.2e} | "
            f"{int(r2r[arm]['1e+00']['escaped_step']):11d} | {r2r[arm]['1e+00']['final_rms']:11.2e} | "
            f"{r2r[arm]['1e+06']['final_rms']:13.2e} | {r2n[arm]['step1_amplification']:11.1f}"
        )

    print("\n=== REGIME 2d: ALGEBRAIC 1/t DECAY (Theorem 3b: Decaying Peak Settling Failure) ===")
    for arm, d in res["regime2_deterministic"]["algebraic_decay_1_over_t"].items():
        if arm == "pure_peak_decay_25k_1d":
            print(
                f"  {arm:<22}: step_1={d['step_1']:.4e}, step_2000={d['step_2000']:.4e}, "
                f"step_25000={d['step_25000']:.4e}, ratio_25k/1={d['ratio_25000_over_1']:.4f}"
            )
        else:
            print(
                f"  {arm:<22}: step_1={d['step_1']:.4e}, step_2000={d['step_2000']:.4e}, "
                f"step_5000={d['step_5000']:.4e}, r5k/1={d['ratio_5000_over_1']:.4f}, r5k/2k={d['ratio_5000_over_2000']:.4f}"
            )

    print("\n=== REGIME 3: NOISY STATIONARY POINTS (Gaussian vs Sphere vs Variance Drop) ===")
    r3g = res["regime3_noisy_stationary"]["gaussian_const_var_scale1e-3"]
    r3s = res["regime3_noisy_stationary"]["sphere_const_norm_scale1e-3"]
    r3v = res["regime3_noisy_stationary"]["variance_drop_100x"]
    print(f"{'Arm':<22} | {'gauss_tail/ear':>14} | {'sphere_tail/ear':>15} | {'vdrop_tail/ear':>14} | {'gauss_drift':>11}")
    print("-" * 88)
    for arm in ALL_ARMS:
        print(
            f"{arm:<22} | {r3g[arm]['tail_over_early']:14.4f} | {r3s[arm]['tail_over_early']:15.4f} | "
            f"{r3v[arm]['tail_over_early']:14.4f} | {r3g[arm]['final_drift_norm']:11.4f}"
        )
    print("  Coherence i.i.d. floor check:")
    for k, v in res["regime3_noisy_stationary"]["coherence_iid_floor_verification"].items():
        print(f"    {k}: theory={v['theoretical_floor']:.4f}, emp={v['empirical_mean']:.4f}, err={v['abs_error']:.4f}")

    print("\n=== REGIME 4: ADVERSARIAL SPIKE FREEZING, SHIFT+SCALE-DROP, & SADDLE ESCAPE ===")
    r4s_small = res["regime4_adversarial_shift"]["single_spike_1e4x"]["c=1e-6"]
    r4s_one = res["regime4_adversarial_shift"]["single_spike_1e4x"]["c=1e+0"]
    r4sh = res["regime4_adversarial_shift"]["shift_with_1000x_scale_drop"]
    r4sd = res["regime4_adversarial_shift"]["nonconvex_saddle_escape"]
    print(
        f"{'Arm':<22} | {'spk_err(1e-6)':>13} | {'spk_err(1)':>10} | {'spk_post/pre(1e-6)':>18} | "
        f"{'shift_err':>9} | {'saddle_esc':>10}"
    )
    print("-" * 96)
    for arm in ALL_ARMS:
        print(
            f"{arm:<22} | {r4s_small[arm]['final_err']:13.4f} | {r4s_one[arm]['final_err']:10.4f} | "
            f"{r4s_small[arm]['post_over_pre']:18.3e} | {r4sh[arm]['phase2_err_to_t2']:9.4f} | "
            f"{int(r4sd[arm]['escape_step']):10d}"
        )

    print("\n=== REGIME 5: BILINEAR ADVERSARIAL ROTATION ===")
    r5s = res["regime5_adversarial_rotation"]["bilinear_simultaneous"]
    r5a = res["regime5_adversarial_rotation"]["bilinear_alternating"]
    print(f"{'Arm':<22} | {'sim_r_end/r0':>12} | {'sim_tail_stp':>12} | {'alt_r_end/r0':>12} | {'alt_tail_stp':>12}")
    print("-" * 82)
    for arm in ALL_ARMS:
        print(
            f"{arm:<22} | {r5s[arm]['r_ratio']:12.4f} | {r5s[arm]['tail_mean_step_x']:12.4e} | "
            f"{r5a[arm]['r_ratio']:12.4f} | {r5a[arm]['tail_mean_step_x']:12.4e}"
        )
    print("  Harmonic rotation coherence formula check:")
    for k, v in res["regime5_adversarial_rotation"]["harmonic_coherence_formula"].items():
        print(f"    {k:<28}: theory={v['theoretical_coh']:.5f}, emp={v['empirical_coh']:.5f}, err={v['abs_error']:.2e}")


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Run P1 Law 4.2 scale-invariance vs settling benchmarks.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to dump JSON results (keep out of Git).")
    args = parser.parse_args()
    res = run_all()
    _print_summary(res)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
