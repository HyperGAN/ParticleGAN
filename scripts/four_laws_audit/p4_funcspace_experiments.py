"""P4 Adversarial Audit Experiments: Generality of Function-Space Optimization Beyond GANs.

Reproducible, deterministic (protocol seed 0) PyTorch suite evaluating whether
ParticleGAN's realized function-space pushforward trust (``delta_g_max`` /
``_band_scale`` + ``_ema_func_gain``) is a general optimization principle beyond
GANs or a lightweight one-shot secant step-size controller with known literature
equivalents and blind spots.

Experiments:
  Exp 0 — Actual ParticleGAN implementation audit (``_band_scale`` non-C^1 kink,
          one-shot secant violation on saturating & amplified superlinear maps,
          ``coord_scale`` 0.05 floor, image ``b_p=2`` budget, ``1e-38`` scaling,
          and MP-scaling formula inventory across ``t4_nash``/``minimal``/``v11``/``v12``).
  Exp 1 — Functionally equivalent reparameterizations (W1*alpha, W2/alpha for
          alpha in {1, 10, 100, 1000}) & external output-gain scaling (gamma).
  Exp 2 — Extreme Jacobian conditioning & nonlinearity gap (||J Delta theta|| vs
          ||f(theta+Delta) - f(theta)||) on 16-layer unnormalized MLP and
          16-layer saturating tanh ResNet.
  Exp 3 — Small Multi-Head Self-Attention Transformer (Q/K softmax-flip
          nonlinearity gap & training stability vs Adam/Clip/JVP/Secant/Backtrack/Armijo).
  Exp 4 — Ordinary supervised classification/regression (benign vs hostile gain)
          and autoregressive dynamical system rollout (damped pendulum & Lorenz-63).
  Exp 5 — Adversarial probe generalization & blind-spot falsification (i.i.d.
          probe-size scaling incl. b_p=2 & b_p=16, 1% rare tail mode, OOD shift,
          exact probe-Jacobian nullspace, and stratified-probe mitigation boundary).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, jvp

from particlegan._minimal.dynamics import _mp_scale as minimal_mp_scale
from particlegan._t4_nash.optim import _impedance_normalized_direction
from particlegan._unified_v4.optimizer import _QUINTIC_COEFFS, _band_scale
from particlegan._unified_v11.optimizer import quintic_orthogonalize_2d
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer
from particlegan.init import deterministic_orthogonal_


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_params(model: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _restore_params(model: nn.Module, saved: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(model.parameters(), saved):
            p.copy_(s)


def _apply_scaled_deltas(
    model: nn.Module,
    base: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for p, b, d in zip(model.parameters(), base, deltas):
            p.copy_(b + alpha * d)


def _jvp_rms_displacement(
    model: nn.Module,
    x_probe: torch.Tensor,
    deltas: Sequence[torch.Tensor],
) -> float:
    """Exact first-order predicted RMS function-space displacement ||J_theta f(x) Delta theta||_RMS."""
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    tangents = {name: d.detach() for (name, _), d in zip(model.named_parameters(), deltas)}

    def _f(p_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        merged = {**named_buffers, **p_dict}
        return functional_call(model, merged, (x_probe,))

    was_training = model.training
    model.eval()
    try:
        _, dy = jvp(_f, (named_params,), (tangents,))
        return float(dy.detach().float().square().mean().sqrt().item())
    finally:
        model.train(was_training)


@torch.no_grad()
def _realized_rms_displacement(
    model: nn.Module,
    x_probe: torch.Tensor,
    base: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    alpha: float,
    y_base: torch.Tensor | None = None,
) -> float:
    """Exact realized nonlinear RMS displacement RMS(f_{theta + alpha*Delta}(x) - f_theta(x))."""
    was_training = model.training
    model.eval()
    try:
        if y_base is None:
            _restore_params(model, base)
            y_base = model(x_probe).detach().float()
        _apply_scaled_deltas(model, base, deltas, alpha)
        y_new = model(x_probe).detach().float()
        return float((y_new - y_base).square().mean().sqrt().item())
    finally:
        _restore_params(model, base)
        model.train(was_training)


def _backtrack_realized_trust(
    model: nn.Module,
    x_probe: torch.Tensor,
    base: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    delta_max: float,
    *,
    max_halvings: int = 20,
    shrink: float = 0.5,
) -> Tuple[float, float, int]:
    """True iterative backtracking enforcing realized RMS(f(theta+alpha*d)-f(theta)) <= delta_max."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _restore_params(model, base)
            y_base = model(x_probe).detach().float()
            alpha = 1.0
            evals = 0
            last_disp = float("inf")
            for _ in range(max_halvings + 1):
                _apply_scaled_deltas(model, base, deltas, alpha)
                y_new = model(x_probe).detach().float()
                evals += 1
                disp = float((y_new - y_base).square().mean().sqrt().item())
                last_disp = disp
                if math.isfinite(disp) and disp <= delta_max:
                    return alpha, disp, evals
                if not math.isfinite(disp):
                    alpha *= shrink
                else:
                    # Conservative secant-guided shrink bounded by `shrink`
                    sec = delta_max / max(disp, 1e-12)
                    alpha *= min(shrink, max(0.05, 0.9 * sec))
            return alpha, last_disp, evals
    finally:
        _restore_params(model, base)
        model.train(was_training)


def _oneshot_secant_trust(
    model: nn.Module,
    x_probe: torch.Tensor,
    base: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    delta_max: float,
    delta_min: float = 0.0,
    alpha_max: float = 48.0,
) -> Tuple[float, float, float]:
    """ParticleGAN-style one-shot _band_scale rescaling (returns alpha, probe_disp_at_1, actual_disp_at_alpha)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _restore_params(model, base)
            y_base = model(x_probe).detach().float()
            _apply_scaled_deltas(model, base, deltas, 1.0)
            y1 = model(x_probe).detach().float()
            disp1 = float((y1 - y_base).square().mean().sqrt().item())
            alpha = _band_scale(disp1, delta_min, delta_max, alpha_max=alpha_max)
            _apply_scaled_deltas(model, base, deltas, alpha)
            ya = model(x_probe).detach().float()
            dispa = float((ya - y_base).square().mean().sqrt().item())
            return alpha, disp1, dispa
    finally:
        _restore_params(model, base)
        model.train(was_training)


def _armijo_backtrack(
    loss_fn: Callable[[], torch.Tensor],
    model: nn.Module,
    base: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    *,
    c1: float = 1e-4,
    max_halvings: int = 16,
    shrink: float = 0.5,
) -> Tuple[float, float, int]:
    """Classical Armijo backtracking line search on scalar supervised loss."""
    with torch.no_grad():
        _restore_params(model, base)
        loss0 = float(loss_fn().item())
        dir_deriv = sum(float((g.float() * d.float()).sum().item()) for g, d in zip(grads, deltas))
        if dir_deriv >= 0.0:
            return 0.0, loss0, 0
        alpha = 1.0
        evals = 0
        last_loss = loss0
        for _ in range(max_halvings + 1):
            _apply_scaled_deltas(model, base, deltas, alpha)
            loss_try = float(loss_fn().item())
            evals += 1
            last_loss = loss_try
            if math.isfinite(loss_try) and loss_try <= loss0 + c1 * alpha * dir_deriv:
                return alpha, loss_try, evals
            alpha *= shrink
        return alpha, last_loss, evals


def _compute_base_deltas(
    params: Sequence[nn.Parameter],
    grads: Sequence[torch.Tensor],
    method: str,
    lr: float,
    state: Dict[str, Any],
    step_idx: int,
) -> List[torch.Tensor]:
    """Compute raw parameter step deltas for SGD, Adam, LARS, or Muon-NS."""
    deltas: List[torch.Tensor] = []
    if method == "sgd":
        for g in grads:
            deltas.append(-lr * g)
    elif method == "adam":
        b1, b2, eps = 0.9, 0.999, 1e-8
        m_list = state.setdefault("m", [torch.zeros_like(p) for p in params])
        v_list = state.setdefault("v", [torch.zeros_like(p) for p in params])
        bc1 = 1.0 - b1 ** step_idx
        bc2 = 1.0 - b2 ** step_idx
        for i, g in enumerate(grads):
            m_list[i].mul_(b1).add_(g, alpha=1.0 - b1)
            v_list[i].mul_(b2).addcmul_(g, g, value=1.0 - b2)
            m_hat = m_list[i] / bc1
            v_hat = (v_list[i] / bc2).sqrt() + eps
            deltas.append(-lr * (m_hat / v_hat))
    elif method == "lars":
        for p, g in zip(params, grads):
            w_norm = p.detach().float().norm().clamp_min(1e-12)
            g_norm = g.float().norm().clamp_min(1e-12)
            deltas.append(-lr * (w_norm / g_norm) * g)
    elif method == "muon_ns":
        for p, g in zip(params, grads):
            if p.ndim >= 2 and min(p.shape[0], p.numel() // max(1, p.shape[0])) >= 2:
                mat = g.float().reshape(p.shape[0], -1)
                r_d, c_d = mat.shape
                aspect = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
                orth = quintic_orthogonalize_2d(mat, steps=5, smoothing=0.0).float()
                deltas.append((-lr * aspect * orth).reshape_as(p).to(p.dtype))
            else:
                g_norm = g.float().norm().clamp_min(1e-12)
                deltas.append(-lr * (g / g_norm))
    else:
        raise ValueError(f"unknown base method: {method}")
    return deltas


# ---------------------------------------------------------------------------
# Exp 0 — Actual ParticleGAN Implementation Audit
# ---------------------------------------------------------------------------
def run_exp0_code_audit(device: torch.device) -> Dict[str, Any]:
    _set_seed(0)
    # 1. Non-C^1 kinks of _band_scale at delta_max and true amplification kink delta_min * (1 - 1/alpha_max)
    d_max, d_min, a_max, h = 0.20, 0.02, 48.0, 1e-6
    d_amp_kink = d_min * (1.0 - 1.0 / a_max)
    left_dmax = (_band_scale(d_max, d_min, d_max, alpha_max=a_max) - _band_scale(d_max - h, d_min, d_max, alpha_max=a_max)) / h
    right_dmax = (_band_scale(d_max + h, d_min, d_max, alpha_max=a_max) - _band_scale(d_max, d_min, d_max, alpha_max=a_max)) / h
    left_dmin = (_band_scale(d_min, d_min, d_max, alpha_max=a_max) - _band_scale(d_min - h, d_min, d_max, alpha_max=a_max)) / h
    right_dmin = (_band_scale(d_min + h, d_min, d_max, alpha_max=a_max) - _band_scale(d_min, d_min, d_max, alpha_max=a_max)) / h
    left_kink = (_band_scale(d_amp_kink, d_min, d_max, alpha_max=a_max) - _band_scale(d_amp_kink - h, d_min, d_max, alpha_max=a_max)) / h
    right_kink = (_band_scale(d_amp_kink + h, d_min, d_max, alpha_max=a_max) - _band_scale(d_amp_kink, d_min, d_max, alpha_max=a_max)) / h

    # 2a. One-shot secant VIOLATION on saturating tanh: f(theta) = tanh(theta), theta0=0, dtheta=25
    d_sat_1 = abs(math.tanh(25.0) - math.tanh(0.0))
    alpha_sat_oneshot = _band_scale(d_sat_1, 0.0, d_max, alpha_max=48.0)
    d_sat_after_oneshot = abs(math.tanh(25.0 * alpha_sat_oneshot) - math.tanh(0.0))
    # True iterative backtracking on the same saturating scalar map
    alpha_sat_bt = 1.0
    bt_evals = 0
    while bt_evals < 30:
        bt_evals += 1
        d_try = abs(math.tanh(25.0 * alpha_sat_bt))
        if d_try <= d_max:
            break
        alpha_sat_bt *= 0.5
    d_sat_after_bt = abs(math.tanh(25.0 * alpha_sat_bt))

    # 2b. One-shot amplification OVERSHOOT on superlinear cubic f(theta) = 0.005 * theta^3, dtheta=1
    # At t=1, delta(1)=0.005 < delta_min=0.05 -> _band_scale amplifies without re-checking!
    d_sup_1 = 0.005 * (1.0 ** 3)
    alpha_sup_amp = _band_scale(d_sup_1, 0.05, d_max, alpha_max=48.0)
    d_sup_after_amp = 0.005 * (alpha_sup_amp ** 3)

    # 3. coord_scale floor distortion and probe budget b_p
    probe_budgets = {}
    coord_scale_ratios = {}
    for label, out_dim, data_scale in [
        ("vec_2d_scale1", 2, 1.0),
        ("vec_2d_tiny1e-3", 2, 1e-3),
        ("img_64x64x3", 64 * 64 * 3, 1.0),
        ("img_256x256x3", 256 * 256 * 3, 1.0),
    ]:
        b_p = min(64, max(2, min(16, 8192 // out_dim)))
        true_coord = data_scale / math.sqrt(float(out_dim))
        clamped_coord = max(0.05, true_coord)
        probe_budgets[label] = b_p
        coord_scale_ratios[label] = {
            "true_coord_scale": true_coord,
            "clamped_coord_scale": clamped_coord,
            "distortion_factor": clamped_coord / true_coord,
        }

    # 4. Dynamic 1e-38 scaling in UnifiedV12Optimizer vs torch.optim.Adam on 1e-24 gradient
    w_v12 = nn.Parameter(torch.ones((4, 4), device=device, dtype=torch.float32))
    w_adam = nn.Parameter(torch.ones((4, 4), device=device, dtype=torch.float32))
    opt_v12 = UnifiedV12Optimizer([{"params": [w_v12], "role": "generator"}], lr=0.014, smoothing=0.0)
    opt_adam = torch.optim.Adam([w_adam], lr=0.014, eps=1e-8)
    tiny_g = torch.full((4, 4), 1e-24, device=device, dtype=torch.float32)
    w_v12.grad = tiny_g.clone()
    w_adam.grad = tiny_g.clone()
    opt_v12.step()
    opt_adam.step()
    v12_step_norm = float((w_v12.detach() - 1.0).norm().item())
    adam_step_norm = float((w_adam.detach() - 1.0).norm().item())

    # 5. MP scaling formula check across t4_nash / minimal / unified_v11 / unified_v12 on 128x128 weight
    w128 = torch.ones((128, 128), device=device, dtype=torch.float32)
    u128 = torch.ones((128, 128), device=device, dtype=torch.float32)
    t4_dir, _ = _impedance_normalized_direction(w128, u128)
    t4_scale = float(t4_dir[0, 0].item())
    min_scale = float(minimal_mp_scale(w128))
    v11_aspect = math.sqrt(max(1.0, 128.0 / 128.0))
    v12_critic_factor = v11_aspect * math.sqrt(128.0)

    return {
        "band_scale_derivatives": {
            "at_delta_max_left": left_dmax,
            "at_delta_max_right": right_dmax,
            "at_delta_min_left": left_dmin,
            "at_delta_min_right": right_dmin,
            "true_amp_kink_delta": d_amp_kink,
            "at_amp_kink_left": left_kink,
            "at_amp_kink_right": right_kink,
            "is_c1_at_delta_max": abs(left_dmax - right_dmax) < 1e-3,
            "is_c1_at_amp_kink": abs(left_kink - right_kink) < 1e-3,
        },
        "oneshot_vs_backtrack_saturating_tanh": {
            "delta_max": d_max,
            "delta_at_1": d_sat_1,
            "oneshot_alpha": alpha_sat_oneshot,
            "oneshot_actual_disp": d_sat_after_oneshot,
            "oneshot_violation_ratio": d_sat_after_oneshot / d_max,
            "backtrack_alpha": alpha_sat_bt,
            "backtrack_actual_disp": d_sat_after_bt,
            "backtrack_evals": bt_evals,
        },
        "oneshot_amplification_superlinear_cubic": {
            "delta_min": 0.05,
            "delta_max": d_max,
            "delta_at_1": d_sup_1,
            "oneshot_alpha": alpha_sup_amp,
            "oneshot_actual_disp": d_sup_after_amp,
            "overshoot_vs_delta_max": d_sup_after_amp / d_max,
        },
        "probe_budgets": probe_budgets,
        "coord_scale_floor": coord_scale_ratios,
        "dynamic_1e38_underflow_check": {
            "grad_magnitude": 1e-24,
            "unified_v12_step_norm": v12_step_norm,
            "standard_adam_step_norm": adam_step_norm,
        },
        "mp_scaling_128x128": {
            "t4_nash_elem_factor": t4_scale,
            "t4_nash_expected_1_over_2sqrt128": 1.0 / (2.0 * math.sqrt(128.0)),
            "minimal_elem_factor": min_scale,
            "unified_v11_polar_aspect": v11_aspect,
            "unified_v12_critic_frob_factor": v12_critic_factor,
            "quintic_step0_coeff": list(_QUINTIC_COEFFS[0]),
        },
    }


# ---------------------------------------------------------------------------
# Exp 1 — Functionally Equivalent Reparameterizations & External Output Gain
# ---------------------------------------------------------------------------
class HomogeneousTwoLayerNet(nn.Module):
    def __init__(self, d_in: int = 8, h_dim: int = 32, d_out: int = 4, gain: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, h_dim, bias=False)
        self.fc2 = nn.Linear(h_dim, d_out, bias=False)
        self.gain = float(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * self.fc2(F.relu(self.fc1(x)))


def _init_reparam_net(
    alpha: float,
    device: torch.device,
    *,
    gain: float = 1.0,
    seed: int = 0,
) -> HomogeneousTwoLayerNet:
    net = HomogeneousTwoLayerNet(8, 32, 4, gain=gain).to(device)
    deterministic_orthogonal_(net, seed=seed)
    with torch.no_grad():
        net.fc1.weight.mul_(alpha)
        net.fc2.weight.div_(alpha)
    return net


def _layer_functional_contributions(
    net: HomogeneousTwoLayerNet,
    x: torch.Tensor,
    d_w1: torch.Tensor,
    d_w2: torch.Tensor,
) -> Tuple[float, float]:
    """Measure isolated functional RMS displacement from d_w1 alone vs d_w2 alone."""
    with torch.no_grad():
        y0 = net(x)
        y_only_w1 = net.gain * (F.relu(x @ (net.fc1.weight + d_w1).T) @ net.fc2.weight.T)
        y_only_w2 = net.gain * (F.relu(x @ net.fc1.weight.T) @ (net.fc2.weight + d_w2).T)
        c1 = float((y_only_w1 - y0).square().mean().sqrt().item())
        c2 = float((y_only_w2 - y0).square().mean().sqrt().item())
        return c1, c2


def _jacobian_matrix(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Vectorized exact output-vs-parameter Jacobian [B*d_out, P] in float64 via jacrev."""
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())

    def _f(p_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(model, {**named_buffers, **p_dict}, (x,)).reshape(-1)

    jac_dict = jacrev(_f)(named_params)
    with torch.no_grad():
        n_out = int(_f(named_params).numel())
    return torch.cat([jac_dict[k].detach().double().reshape(n_out, -1) for k in named_params], dim=1)


def _exact_gauss_newton_delta(
    net: HomogeneousTwoLayerNet,
    x: torch.Tensor,
    y_target: torch.Tensor,
    lr: float,
    damping: float = 0.0,
) -> List[torch.Tensor]:
    """Exact Gauss-Newton step -(J^T J + damping I)^+ J^T r for MSE loss."""
    params = list(net.parameters())
    with torch.no_grad():
        resid = (net(x) - y_target).reshape(-1).double()
    jac = _jacobian_matrix(net, x)
    n_out, n_par = jac.shape
    if damping > 0.0:
        jtj = (jac.T @ jac) / float(n_out) + damping * torch.eye(n_par, device=jac.device, dtype=torch.float64)
        jtr = (jac.T @ resid) / float(n_out)
        step_flat = -lr * torch.linalg.solve(jtj, jtr)
    else:
        step_flat = -lr * (torch.linalg.pinv(jac, rcond=1e-10) @ resid)
    deltas: List[torch.Tensor] = []
    off = 0
    for p in params:
        sz = p.numel()
        deltas.append(step_flat[off : off + sz].reshape_as(p).to(dtype=p.dtype))
        off += sz
    return deltas


def run_exp1_reparameterization(device: torch.device) -> Dict[str, Any]:
    _set_seed(0)
    x_train = torch.randn(64, 8, device=device)
    # Smooth teacher target generated by a distinct fixed network
    teacher = _init_reparam_net(1.0, device, seed=101)
    with torch.no_grad():
        y_train = teacher(x_train)

    # Verify exact functional equivalence across alpha in {1, 10, 100, 1000}
    alphas = [1.0, 10.0, 100.0, 1000.0]
    y_ref = _init_reparam_net(1.0, device, seed=0)(x_train).detach()
    equiv_err = {}
    for a in alphas:
        ya = _init_reparam_net(a, device, seed=0)(x_train).detach()
        equiv_err[str(int(a))] = float((ya - y_ref).abs().max().item())

    delta_max = 0.10
    configs = [
        ("sgd", "none", 0.05),
        ("adam", "none", 0.02),
        ("lars", "none", 0.05),
        ("muon_ns", "none", 0.02),
        ("exact_gn_undamped", "none", 0.10),
        ("exact_gn_damped", "none", 0.10),
        ("sgd", "jvp_trust", 0.05),
        ("sgd", "realized_trust", 0.05),
        ("adam", "realized_trust", 0.02),
        ("muon_ns", "realized_trust", 0.02),
        ("lars", "realized_trust", 0.05),
    ]

    reparam_results: Dict[str, Dict[str, Any]] = {}
    for base_method, trust_mode, lr in configs:
        arm_name = base_method if trust_mode == "none" else f"{base_method}+{trust_mode}"
        per_alpha: Dict[str, Any] = {}
        for a in alphas:
            net = _init_reparam_net(a, device, seed=0)
            state: Dict[str, Any] = {}
            step1_disp = 0.0
            step1_ratio_w1_w2 = 0.0
            diverged = False
            for step in range(1, 101):
                net.zero_grad(set_to_none=True)
                y_pred = net(x_train)
                loss = F.mse_loss(y_pred, y_train)
                if not math.isfinite(float(loss.item())) or float(loss.item()) > 1e12:
                    diverged = True
                    break
                params = list(net.parameters())
                if base_method == "exact_gn_undamped":
                    deltas = _exact_gauss_newton_delta(net, x_train, y_train, lr, damping=0.0)
                elif base_method == "exact_gn_damped":
                    deltas = _exact_gauss_newton_delta(net, x_train, y_train, lr, damping=1e-2)
                else:
                    loss.backward()
                    grads = [p.grad.detach().clone() for p in params]
                    deltas = _compute_base_deltas(params, grads, base_method, lr, state, step)

                base_w = _clone_params(net)
                if trust_mode == "none":
                    alpha_g = 1.0
                elif trust_mode == "jvp_trust":
                    d_lin = _jvp_rms_displacement(net, x_train, deltas)
                    alpha_g = min(1.0, delta_max / max(d_lin, 1e-12)) if d_lin > delta_max else 1.0
                elif trust_mode == "realized_trust":
                    alpha_g, _, _ = _backtrack_realized_trust(net, x_train, base_w, deltas, delta_max)
                else:
                    raise ValueError(trust_mode)

                scaled_deltas = [alpha_g * d for d in deltas]
                if step == 1:
                    step1_disp = _realized_rms_displacement(net, x_train, base_w, scaled_deltas, 1.0)
                    c1, c2 = _layer_functional_contributions(net, x_train, scaled_deltas[0], scaled_deltas[1])
                    step1_ratio_w1_w2 = c1 / max(c2, 1e-30)
                _apply_scaled_deltas(net, base_w, scaled_deltas, 1.0)

            with torch.no_grad():
                final_mse = float(F.mse_loss(net(x_train), y_train).item())
            if not math.isfinite(final_mse):
                diverged = True
            per_alpha[str(int(a))] = {
                "step1_delta_f": step1_disp,
                "step1_w1_over_w2_ratio": step1_ratio_w1_w2,
                "final_mse_100": final_mse if not diverged else float("inf"),
                "diverged": diverged,
            }
        reparam_results[arm_name] = per_alpha

    # Companion External Output-Gain Scaling Test: f'(x) = gamma * f(x), gamma in {1, 10, 100, 1000}
    # Target ALSO scaled by 1 (fixed target, hostile generator output gain gamma — matches overpowered G!)
    gammas = [1.0, 10.0, 100.0, 1000.0]
    external_gain_results: Dict[str, Dict[str, Any]] = {}
    for base_method, trust_mode, lr in [
        ("sgd", "none", 0.05),
        ("adam", "none", 0.02),
        ("lars", "none", 0.05),
        ("muon_ns", "none", 0.02),
        ("lars", "realized_trust", 0.05),
        ("adam", "realized_trust", 0.02),
    ]:
        arm_name = base_method if trust_mode == "none" else f"{base_method}+{trust_mode}"
        per_g: Dict[str, Any] = {}
        for g_val in gammas:
            net = _init_reparam_net(1.0, device, gain=g_val, seed=0)
            net.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(x_train), y_train)
            loss.backward()
            params = list(net.parameters())
            grads = [p.grad.detach().clone() for p in params]
            deltas = _compute_base_deltas(params, grads, base_method, lr, {}, 1)
            base_w = _clone_params(net)
            if trust_mode == "realized_trust":
                alpha_g, disp, _ = _backtrack_realized_trust(net, x_train, base_w, deltas, delta_max)
            else:
                alpha_g = 1.0
                disp = _realized_rms_displacement(net, x_train, base_w, deltas, 1.0)
            per_g[str(int(g_val))] = {"alpha": alpha_g, "step1_delta_f": disp}
        external_gain_results[arm_name] = per_g

    return {
        "max_initial_equiv_error": equiv_err,
        "homogeneous_reparam": reparam_results,
        "external_output_gain": external_gain_results,
    }


# ---------------------------------------------------------------------------
# Exp 2 — Extreme Jacobian Conditioning & Nonlinearity Gap
# ---------------------------------------------------------------------------
class DeepUnnormMLP(nn.Module):
    def __init__(self, d_in: int = 4, width: int = 64, depth: int = 16, d_out: int = 4) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [d_in] + [width] * (depth - 1) + [d_out]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = F.leaky_relu(layer(h), negative_slope=0.2)
        return self.layers[-1](h)


class SaturatingTanhResNet(nn.Module):
    def __init__(
        self,
        d_in: int = 4,
        width: int = 64,
        depth: int = 16,
        d_out: int = 4,
        pre_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out_proj = nn.Linear(width, d_out)
        self.pre_gain = float(pre_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.pre_gain * self.in_proj(x))
        for blk in self.blocks:
            h = h + torch.tanh(self.pre_gain * blk(h))
        return torch.tanh(self.out_proj(h))


def _init_orth_module(model: nn.Module, seed: int = 0, gain: float = 1.0) -> None:
    deterministic_orthogonal_(model, seed=seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if gain != 1.0:
                    m.weight.mul_(gain)
                if m.bias is not None:
                    m.bias.zero_()


def run_exp2_nonlinearity_gap(device: torch.device) -> Dict[str, Any]:
    _set_seed(0)
    x_probe = torch.randn(32, 4, device=device)
    y_target = torch.sin(x_probe) + 0.5 * torch.cos(2.0 * x_probe)

    # 2A: 16-layer skip-free unnormalized MLP — superlinear multiplicative blowup (Gamma >> 1)
    mlp = DeepUnnormMLP(4, 64, 16, 4).to(device)
    # Scale by sqrt(3/0.52) so orthogonal Uniform-RMS layers maintain O(1) forward norm across 16 layers
    _init_orth_module(mlp, seed=0, gain=math.sqrt(3.0 / 0.52))
    mlp.zero_grad(set_to_none=True)
    loss = F.mse_loss(mlp(x_probe), y_target)
    loss.backward()
    params = list(mlp.parameters())
    grads = [p.grad.detach().clone() for p in params]
    base_mlp = _clone_params(mlp)

    delta_max = 0.20
    mlp_sweep = []
    for lr in [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]:
        # Unit-RMS Adam-like sign step (step 1 of Adam with eps=1e-8)
        deltas = _compute_base_deltas(params, grads, "adam", lr, {}, 1)
        d_lin = _jvp_rms_displacement(mlp, x_probe, deltas)
        d_real = _realized_rms_displacement(mlp, x_probe, base_mlp, deltas, 1.0)
        gamma = d_real / max(d_lin, 1e-12)
        # JVP trust post-step actual realized displacement
        a_jvp = min(1.0, delta_max / max(d_lin, 1e-12)) if d_lin > delta_max else 1.0
        post_jvp = _realized_rms_displacement(mlp, x_probe, base_mlp, deltas, a_jvp)
        # One-shot secant post-step actual realized displacement
        a_sec, _, post_sec = _oneshot_secant_trust(mlp, x_probe, base_mlp, deltas, delta_max)
        # Iterative backtracking post-step actual realized displacement
        a_bt, post_bt, bt_ev = _backtrack_realized_trust(mlp, x_probe, base_mlp, deltas, delta_max)
        mlp_sweep.append(
            {
                "lr": lr,
                "delta_lin_jvp": d_lin,
                "delta_real": d_real,
                "gamma_real_over_lin": gamma,
                "jvp_trust_alpha": a_jvp,
                "jvp_trust_actual_disp": post_jvp,
                "oneshot_alpha": a_sec,
                "oneshot_actual_disp": post_sec,
                "backtrack_alpha": a_bt,
                "backtrack_actual_disp": post_bt,
                "backtrack_evals": bt_ev,
            }
        )

    # 2A Training comparison on 16-layer MLP at aggressive lr=0.05 (where Gamma >> 1!)
    mlp_train_results: Dict[str, Any] = {}
    for arm in ["adam", "adam+jvp_trust", "adam+oneshot_secant", "adam+backtrack_trust"]:
        net = DeepUnnormMLP(4, 64, 16, 4).to(device)
        _init_orth_module(net, seed=0, gain=math.sqrt(3.0 / 0.52))
        st: Dict[str, Any] = {}
        max_disp = 0.0
        violations = 0
        diverged = False
        for step in range(1, 121):
            net.zero_grad(set_to_none=True)
            pred = net(x_probe)
            lo = F.mse_loss(pred, y_target)
            if not math.isfinite(float(lo.item())) or float(lo.item()) > 1e8:
                diverged = True
                break
            lo.backward()
            ps = list(net.parameters())
            gs = [p.grad.detach().clone() for p in ps]
            ds = _compute_base_deltas(ps, gs, "adam", 0.05, st, step)
            bw = _clone_params(net)
            if arm == "adam":
                a_use = 1.0
            elif arm == "adam+jvp_trust":
                dl = _jvp_rms_displacement(net, x_probe, ds)
                a_use = min(1.0, delta_max / max(dl, 1e-12)) if dl > delta_max else 1.0
            elif arm == "adam+oneshot_secant":
                a_use, _, _ = _oneshot_secant_trust(net, x_probe, bw, ds, delta_max)
            else:
                a_use, _, _ = _backtrack_realized_trust(net, x_probe, bw, ds, delta_max)
            actual_d = _realized_rms_displacement(net, x_probe, bw, ds, a_use)
            max_disp = max(max_disp, actual_d)
            if actual_d > delta_max * 1.05:
                violations += 1
            _apply_scaled_deltas(net, bw, ds, a_use)
        with torch.no_grad():
            f_mse = float(F.mse_loss(net(x_probe), y_target).item())
        mlp_train_results[arm] = {
            "final_mse": f_mse if (not diverged and math.isfinite(f_mse)) else float("inf"),
            "max_realized_disp": max_disp,
            "trust_violations": violations,
            "diverged": diverged,
        }

    # 2B: Saturating tanh ResNet — concave saturation (Gamma << 1: one-shot secant fails!)
    # AND rail-jump saturation (pre-activation=8.0, sech^2 ~ 4.5e-7 -> Gamma >> 1e4: JVP blind!)
    res_sat = SaturatingTanhResNet(4, 64, 16, 4, pre_gain=5.0).to(device)
    _init_orth_module(res_sat, seed=0, gain=1.0)
    res_sat.zero_grad(set_to_none=True)
    F.mse_loss(res_sat(x_probe), y_target).backward()
    rs_params = list(res_sat.parameters())
    rs_grads = [p.grad.detach().clone() for p in rs_params]
    base_rs = _clone_params(res_sat)

    resnet_sweep = []
    for lr in [1e-3, 1e-2, 5e-2, 2e-1, 1.0]:
        ds = _compute_base_deltas(rs_params, rs_grads, "adam", lr, {}, 1)
        d_lin = _jvp_rms_displacement(res_sat, x_probe, ds)
        d_real = _realized_rms_displacement(res_sat, x_probe, base_rs, ds, 1.0)
        a_jvp = min(1.0, delta_max / max(d_lin, 1e-12)) if d_lin > delta_max else 1.0
        post_jvp = _realized_rms_displacement(res_sat, x_probe, base_rs, ds, a_jvp)
        a_sec, _, post_sec = _oneshot_secant_trust(res_sat, x_probe, base_rs, ds, delta_max)
        a_bt, post_bt, bt_ev = _backtrack_realized_trust(res_sat, x_probe, base_rs, ds, delta_max)
        resnet_sweep.append(
            {
                "lr": lr,
                "delta_lin_jvp": d_lin,
                "delta_real": d_real,
                "gamma_real_over_lin": d_real / max(d_lin, 1e-12),
                "jvp_trust_actual_disp": post_jvp,
                "oneshot_alpha": a_sec,
                "oneshot_actual_disp": post_sec,
                "oneshot_violation_ratio": post_sec / delta_max,
                "backtrack_alpha": a_bt,
                "backtrack_actual_disp": post_bt,
                "backtrack_evals": bt_ev,
            }
        )

    # Rail-escape jump: saturated tanh output bias = 8.0, sech^2(8) = 4.5e-7
    rail_net = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 4), nn.Tanh()).to(device)
    _init_orth_module(rail_net, seed=0, gain=0.1)
    with torch.no_grad():
        rail_net[2].bias.fill_(8.0)
    rail_net.zero_grad(set_to_none=True)
    F.mse_loss(rail_net(x_probe), torch.full_like(x_probe, -1.0)).backward()
    rp = list(rail_net.parameters())
    rg = [p.grad.detach().clone() for p in rp]
    base_rail = _clone_params(rail_net)
    # Scale-invariant step of size 10.0 on bias (jumps from +8 -> -2 across tanh(0)!)
    ds_rail = _compute_base_deltas(rp, rg, "adam", 10.0, {}, 1)
    rail_d_lin = _jvp_rms_displacement(rail_net, x_probe, ds_rail)
    rail_d_real = _realized_rms_displacement(rail_net, x_probe, base_rail, ds_rail, 1.0)

    # Bilinear small-weight saddle witness: f(x) = W2 W1 x with ||W|| ~ 1e-3.
    # JVP sees only O(1e-3) linear terms (accepts alpha=1.0!), while Delta W2 @ Delta W1 x explodes O(1)!
    bilin_net = nn.Sequential(nn.Linear(4, 16, bias=False), nn.Linear(16, 4, bias=False)).to(device)
    _init_orth_module(bilin_net, seed=0, gain=1e-3)
    base_bi = _clone_params(bilin_net)
    bi_deltas = [torch.full_like(p, 0.25) for p in bilin_net.parameters()]
    bi_d_lin = _jvp_rms_displacement(bilin_net, x_probe, bi_deltas)
    bi_d_real = _realized_rms_displacement(bilin_net, x_probe, base_bi, bi_deltas, 1.0)
    bi_jvp_alpha = min(1.0, delta_max / max(bi_d_lin, 1e-12)) if bi_d_lin > delta_max else 1.0
    bi_jvp_post = _realized_rms_displacement(bilin_net, x_probe, base_bi, bi_deltas, bi_jvp_alpha)
    bi_bt_alpha, bi_bt_post, bi_bt_evals = _backtrack_realized_trust(bilin_net, x_probe, base_bi, bi_deltas, delta_max)

    return {
        "deep_mlp16_sweep": mlp_sweep,
        "deep_mlp16_train": mlp_train_results,
        "saturating_tanh_resnet16_sweep": resnet_sweep,
        "tanh_rail_jump": {
            "delta_lin_jvp": rail_d_lin,
            "delta_real": rail_d_real,
            "gamma_real_over_lin": rail_d_real / max(rail_d_lin, 1e-12),
        },
        "bilinear_saddle_jvp_blindness": {
            "delta_max": delta_max,
            "delta_lin_jvp": bi_d_lin,
            "delta_real": bi_d_real,
            "gamma_real_over_lin": bi_d_real / max(bi_d_lin, 1e-12),
            "jvp_trust_alpha": bi_jvp_alpha,
            "jvp_trust_actual_disp": bi_jvp_post,
            "backtrack_alpha": bi_bt_alpha,
            "backtrack_actual_disp": bi_bt_post,
            "backtrack_evals": bi_bt_evals,
        },
    }


# ---------------------------------------------------------------------------
# Exp 3 — Small Multi-Head Self-Attention Transformer
# ---------------------------------------------------------------------------
class SmallAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4, qk_scale_init: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_model * 2)
        self.ff2 = nn.Linear(d_model * 2, d_model)
        self.qk_scale_init = float(qk_scale_init)
        self.last_max_logit: float = 0.0
        self.last_min_entropy: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q = self.w_q(x).view(bsz, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq, self.n_heads, self.d_head).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.d_head))
        attn = F.softmax(logits, dim=-1)
        with torch.no_grad():
            self.last_max_logit = float(logits.detach().abs().amax().item())
            ent = -(attn.detach() * (attn.detach().clamp_min(1e-12)).log()).sum(dim=-1)
            self.last_min_entropy = float(ent.amin().item())
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq, dim)
        h = x + self.w_o(out)
        h = h + self.ff2(F.gelu(self.ff1(h)))
        return h


class SmallTransformer(nn.Module):
    def __init__(
        self,
        d_in: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        d_out: int = 4,
        qk_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.block1 = SmallAttentionBlock(d_model, n_heads, qk_gain)
        self.block2 = SmallAttentionBlock(d_model, n_heads, qk_gain)
        self.head = nn.Linear(d_model, d_out)
        self.qk_gain = float(qk_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        # Predict from the final query token position
        return self.head(h[:, -1, :])

    def attn_diagnostics(self) -> Tuple[float, float]:
        max_l = max(self.block1.last_max_logit, self.block2.last_max_logit)
        min_e = min(self.block1.last_min_entropy, self.block2.last_min_entropy)
        return max_l, min_e


def _init_transformer(model: SmallTransformer, seed: int = 0, qk_gain: float = 1.0) -> None:
    deterministic_orthogonal_(model, seed=seed)
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                g = qk_gain if (name.endswith("w_q") or name.endswith("w_k")) else 0.7
                m.weight.mul_(g)
                if m.bias is not None:
                    m.bias.zero_()


def _make_associative_recall_data(
    n_samples: int,
    seq_len: int,
    device: torch.device,
    seed: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequence associative recall: tokens 0..T-2 are (key_4d, val_4d); token T-1 is (query_key, 0)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    keys = F.normalize(torch.randn(n_samples, seq_len - 1, 4, generator=gen, device=device), dim=-1)
    vals = torch.randn(n_samples, seq_len - 1, 4, generator=gen, device=device)
    target_idx = torch.randint(0, seq_len - 1, (n_samples,), generator=gen, device=device)
    batch_idx = torch.arange(n_samples, device=device)
    query_key = keys[batch_idx, target_idx]
    query_tok = torch.cat([query_key, torch.zeros_like(query_key)], dim=-1).unsqueeze(1)
    kv_toks = torch.cat([keys, vals], dim=-1)
    x = torch.cat([kv_toks, query_tok], dim=1)  # [N, seq_len, 8]
    y = vals[batch_idx, target_idx]  # [N, 4]
    return x, y


def run_exp3_transformer_attention(device: torch.device) -> Dict[str, Any]:
    x_tr, y_tr = _make_associative_recall_data(128, 10, device, seed=11)
    x_va, y_va = _make_associative_recall_data(128, 10, device, seed=12)

    # Part 3A: Direct Softmax-Flip Nonlinearity Gap Probe (saturated QK -> argmax flip!)
    # Construct a 1-head attention module where key 0 has logit +14 and key 1 has logit +12
    # so softmax is saturated (~0.88/0.12 -> or at beta=8: 1-1e-7 vs 1e-7), then take a QK step flipping winner!
    _set_seed(0)
    tf_probe = SmallTransformer(8, 32, 4, 4, qk_gain=4.0).to(device)
    _init_transformer(tf_probe, seed=0, qk_gain=4.0)
    tf_probe.zero_grad(set_to_none=True)
    F.mse_loss(tf_probe(x_tr[:16]), y_tr[:16]).backward()
    tp = list(tf_probe.parameters())
    tg = [p.grad.detach().clone() for p in tp]
    base_tf = _clone_params(tf_probe)
    delta_max = 0.25
    tf_gap_sweep = []
    for lr in [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1]:
        ds = _compute_base_deltas(tp, tg, "adam", lr, {}, 1)
        dl = _jvp_rms_displacement(tf_probe, x_tr[:16], ds)
        dr = _realized_rms_displacement(tf_probe, x_tr[:16], base_tf, ds, 1.0)
        a_jvp = min(1.0, delta_max / max(dl, 1e-12)) if dl > delta_max else 1.0
        post_jvp = _realized_rms_displacement(tf_probe, x_tr[:16], base_tf, ds, a_jvp)
        a_sec, _, post_sec = _oneshot_secant_trust(tf_probe, x_tr[:16], base_tf, ds, delta_max)
        a_bt, post_bt, bt_ev = _backtrack_realized_trust(tf_probe, x_tr[:16], base_tf, ds, delta_max)
        tf_gap_sweep.append(
            {
                "lr": lr,
                "delta_lin_jvp": dl,
                "delta_real": dr,
                "gamma_real_over_lin": dr / max(dl, 1e-12),
                "jvp_trust_actual_disp": post_jvp,
                "oneshot_actual_disp": post_sec,
                "backtrack_actual_disp": post_bt,
                "backtrack_evals": bt_ev,
            }
        )

    # Exact synthetic softmax argmax-flip witness (analytical worst case for JVP!)
    # Attention over 2 keys k0=[1,0], k1=[0,1], v0=[+2], v1=[-2]; q=[10, 0] -> p=[1-4.5e-5, 4.5e-5]
    q_param = nn.Parameter(torch.tensor([[12.0, 0.0]], device=device))
    k_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    v_vec = torch.tensor([[2.0], [-2.0]], device=device)

    def _attn_scalar(q_t: torch.Tensor) -> torch.Tensor:
        return (F.softmax(q_t @ k_mat.T, dim=-1) @ v_vec).squeeze()

    dq = torch.tensor([[-14.0, 14.0]], device=device)  # Flips logits from (12,0) to (-2,14)
    y0_sm, dy_lin_sm = jvp(_attn_scalar, (q_param,), (dq,))
    y1_sm = _attn_scalar(q_param + dq)
    sm_flip_lin = float(dy_lin_sm.abs().item())
    sm_flip_real = float((y1_sm - y0_sm).abs().item())

    # Part 3B: Training stability sweep across LRs with hostile QK gain=2.5 (prone to logit spikes)
    arms = [
        "adam",
        "adam+grad_clip",
        "adam+step_clip",
        "adam+jvp_trust",
        "adam+oneshot_secant",
        "adam+backtrack_trust",
        "adam+armijo",
    ]
    train_sweep: Dict[str, Dict[str, Any]] = {}
    for lr in [0.005, 0.02, 0.05]:
        per_arm: Dict[str, Any] = {}
        for arm in arms:
            model = SmallTransformer(8, 32, 4, 4, qk_gain=2.5).to(device)
            _init_transformer(model, seed=0, qk_gain=2.5)
            st: Dict[str, Any] = {}
            t0 = time.perf_counter()
            loss_hist: List[float] = []
            spikes = 0
            max_logit_seen = 0.0
            min_ent_seen = float("inf")
            diverged = False
            n_steps = 150
            probe_x = x_tr[:16]
            for step in range(1, n_steps + 1):
                model.zero_grad(set_to_none=True)
                pred = model(x_tr)
                lo = F.mse_loss(pred, y_tr)
                lv = float(lo.item())
                if not math.isfinite(lv) or lv > 1e6:
                    diverged = True
                    break
                if loss_hist and lv > 2.0 * loss_hist[-1]:
                    spikes += 1
                loss_hist.append(lv)
                ml, me = model.attn_diagnostics()
                max_logit_seen = max(max_logit_seen, ml)
                min_ent_seen = min(min_ent_seen, me)

                lo.backward()
                ps = list(model.parameters())
                if arm == "adam+grad_clip":
                    torch.nn.utils.clip_grad_norm_(ps, max_norm=1.0)
                gs = [p.grad.detach().clone() for p in ps]
                ds = _compute_base_deltas(ps, gs, "adam", lr, st, step)
                bw = _clone_params(model)
                if arm in ("adam", "adam+grad_clip"):
                    a_use = 1.0
                elif arm == "adam+step_clip":
                    snorm = math.sqrt(sum(float(d.float().square().sum().item()) for d in ds))
                    a_use = min(1.0, 0.15 / max(snorm, 1e-12))
                elif arm == "adam+jvp_trust":
                    dl = _jvp_rms_displacement(model, probe_x, ds)
                    a_use = min(1.0, delta_max / max(dl, 1e-12)) if dl > delta_max else 1.0
                elif arm == "adam+oneshot_secant":
                    a_use, _, _ = _oneshot_secant_trust(model, probe_x, bw, ds, delta_max)
                elif arm == "adam+backtrack_trust":
                    a_use, _, _ = _backtrack_realized_trust(model, probe_x, bw, ds, delta_max)
                elif arm == "adam+armijo":
                    a_use, _, _ = _armijo_backtrack(
                        lambda: F.mse_loss(model(x_tr), y_tr), model, bw, ds, gs
                    )
                else:
                    raise ValueError(arm)
                _apply_scaled_deltas(model, bw, ds, a_use)

            dt_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(loss_hist))
            with torch.no_grad():
                tr_mse = float(F.mse_loss(model(x_tr), y_tr).item())
                va_mse = float(F.mse_loss(model(x_va), y_va).item())
            if not (math.isfinite(tr_mse) and math.isfinite(va_mse)):
                diverged = True
            per_arm[arm] = {
                "train_mse": tr_mse if not diverged else float("inf"),
                "val_mse": va_mse if not diverged else float("inf"),
                "loss_spikes_gt_2x": spikes,
                "max_attn_logit": max_logit_seen,
                "min_attn_entropy": min_ent_seen,
                "ms_per_step": dt_ms,
                "diverged": diverged,
            }
        train_sweep[f"lr={lr}"] = per_arm

    return {
        "softmax_flip_exact_witness": {
            "delta_lin_jvp": sm_flip_lin,
            "delta_real": sm_flip_real,
            "gamma_real_over_lin": sm_flip_real / max(sm_flip_lin, 1e-12),
        },
        "transformer_gap_sweep": tf_gap_sweep,
        "transformer_training": train_sweep,
    }


# ---------------------------------------------------------------------------
# Exp 4 — Supervised Tasks + Learned Dynamical System (World Model Rollout)
# ---------------------------------------------------------------------------
def _make_spiral_classification(
    n_per_class: int,
    n_classes: int,
    device: torch.device,
    seed: int = 21,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = torch.linspace(0.15, 1.0, n_per_class, device=device)
        t = (
            torch.linspace(c * 2.0 * math.pi / n_classes, c * 2.0 * math.pi / n_classes + 3.2, n_per_class, device=device)
            + 0.15 * torch.randn(n_per_class, generator=gen, device=device)
        )
        xs.append(torch.stack([r * torch.sin(t), r * torch.cos(t)], dim=1))
        ys.append(torch.full((n_per_class,), c, dtype=torch.long, device=device))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class SupervisedMLP(nn.Module):
    def __init__(self, d_in: int, width: int, d_out: int, out_gain: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, d_out),
        )
        self.out_gain = float(out_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_gain * self.net(x)


def _pendulum_rhs(state: torch.Tensor) -> torch.Tensor:
    theta = state[..., 0]
    omega = state[..., 1]
    dtheta = omega
    domega = -0.15 * omega - torch.sin(theta)
    return torch.stack([dtheta, domega], dim=-1)


def _lorenz_rhs(state: torch.Tensor) -> torch.Tensor:
    # Scaled Lorenz-63 (u = x / 10) for O(1) coordinates with chaotic sensitivity preserved
    x, y, z = state[..., 0] * 10.0, state[..., 1] * 10.0, state[..., 2] * 10.0
    dx = 10.0 * (y - x)
    dy = x * (28.0 - z) - y
    dz = x * y - (8.0 / 3.0) * z
    return torch.stack([dx, dy, dz], dim=-1) / 10.0


def _rk4_step(rhs: Callable[[torch.Tensor], torch.Tensor], s: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = rhs(s)
    k2 = rhs(s + 0.5 * dt * k1)
    k3 = rhs(s + 0.5 * dt * k2)
    k4 = rhs(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _generate_trajectories(
    rhs: Callable[[torch.Tensor], torch.Tensor],
    s0: torch.Tensor,
    dt: float,
    horizon: int,
) -> torch.Tensor:
    traj = [s0]
    cur = s0
    for _ in range(horizon):
        cur = _rk4_step(rhs, cur, dt)
        traj.append(cur)
    return torch.stack(traj, dim=1)  # [N, H+1, D]


class ResidualDynamicsModel(nn.Module):
    def __init__(self, dim: int, width: int = 48, dt: float = 0.05, gain: float = 1.0) -> None:
        super().__init__()
        self.dt = float(dt)
        self.gain = float(gain)
        self.f = nn.Sequential(
            nn.Linear(dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * self.f(x)

    def rollout(self, x0: torch.Tensor, horizon: int) -> torch.Tensor:
        traj = [x0]
        cur = x0
        for _ in range(horizon):
            cur = cur + self.dt * self.forward(cur)
            traj.append(cur)
        return torch.stack(traj, dim=1)


def _backtrack_rollout_trust(
    model: ResidualDynamicsModel,
    x0_probe: torch.Tensor,
    horizon: int,
    base: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    delta_roll_max: float,
    *,
    max_halvings: int = 16,
) -> Tuple[float, float, int]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _restore_params(model, base)
            r_base = model.rollout(x0_probe, horizon).detach().float()
            alpha = 1.0
            evals = 0
            last_d = float("inf")
            for _ in range(max_halvings + 1):
                _apply_scaled_deltas(model, base, deltas, alpha)
                r_new = model.rollout(x0_probe, horizon).detach().float()
                evals += 1
                disp = float((r_new - r_base).square().mean().sqrt().item())
                last_d = disp
                if math.isfinite(disp) and disp <= delta_roll_max:
                    return alpha, disp, evals
                alpha *= 0.5
            return alpha, last_d, evals
    finally:
        _restore_params(model, base)
        model.train(was_training)


def run_exp4_supervised_and_dynamics(device: torch.device) -> Dict[str, Any]:
    # 4A-1: Benign Supervised Classification (4-class spiral, standard lr=0.01)
    x_sp_tr, y_sp_tr = _make_spiral_classification(64, 4, device, seed=21)
    x_sp_va, y_sp_va = _make_spiral_classification(64, 4, device, seed=22)

    sup_arms = [
        ("sgd", "none", 0.10),
        ("adam", "none", 0.01),
        ("lars", "none", 0.05),
        ("muon_ns", "none", 0.02),
        ("adam", "jvp_trust", 0.01),
        ("adam", "oneshot_secant", 0.01),
        ("adam", "backtrack_trust", 0.01),
        ("adam", "armijo", 0.01),
    ]
    benign_cls: Dict[str, Any] = {}
    for base_m, trust_m, lr in sup_arms:
        name = base_m if trust_m == "none" else f"{base_m}+{trust_m}"
        model = SupervisedMLP(2, 48, 4, out_gain=1.0).to(device)
        _init_orth_module(model, seed=0, gain=1.0)
        st: Dict[str, Any] = {}
        t0 = time.perf_counter()
        n_steps = 200
        for step in range(1, n_steps + 1):
            model.zero_grad(set_to_none=True)
            lo = F.cross_entropy(model(x_sp_tr), y_sp_tr)
            lo.backward()
            ps = list(model.parameters())
            gs = [p.grad.detach().clone() for p in ps]
            ds = _compute_base_deltas(ps, gs, base_m, lr, st, step)
            bw = _clone_params(model)
            if trust_m == "none":
                a = 1.0
            elif trust_m == "jvp_trust":
                dl = _jvp_rms_displacement(model, x_sp_tr[:32], ds)
                a = min(1.0, 0.25 / max(dl, 1e-12)) if dl > 0.25 else 1.0
            elif trust_m == "oneshot_secant":
                a, _, _ = _oneshot_secant_trust(model, x_sp_tr[:32], bw, ds, 0.25)
            elif trust_m == "backtrack_trust":
                a, _, _ = _backtrack_realized_trust(model, x_sp_tr[:32], bw, ds, 0.25)
            else:
                a, _, _ = _armijo_backtrack(
                    lambda: F.cross_entropy(model(x_sp_tr), y_sp_tr), model, bw, ds, gs
                )
            _apply_scaled_deltas(model, bw, ds, a)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / n_steps
        with torch.no_grad():
            va_logits = model(x_sp_va)
            va_loss = float(F.cross_entropy(va_logits, y_sp_va).item())
            va_acc = float((va_logits.argmax(dim=-1) == y_sp_va).float().mean().item())
        benign_cls[name] = {"val_loss": va_loss, "val_acc": va_acc, "ms_per_step": dt_ms}

    # 4A-2: Hostile-Gain Supervised Regression (out_gain=50.0, lr=0.05)
    _set_seed(31)
    x_reg = torch.randn(128, 4, device=device)
    y_reg = torch.sin(x_reg[:, :2]) + 0.3 * (x_reg[:, 2:] ** 2)
    hostile_reg: Dict[str, Any] = {}
    for base_m, trust_m, lr in [
        ("sgd", "none", 0.05),
        ("adam", "none", 0.05),
        ("lars", "none", 0.05),
        ("adam", "jvp_trust", 0.05),
        ("adam", "oneshot_secant", 0.05),
        ("adam", "backtrack_trust", 0.05),
        ("adam", "armijo", 0.05),
    ]:
        name = base_m if trust_m == "none" else f"{base_m}+{trust_m}"
        model = SupervisedMLP(4, 48, 2, out_gain=50.0).to(device)
        _init_orth_module(model, seed=0, gain=1.0)
        st = {}
        diverged = False
        for step in range(1, 151):
            model.zero_grad(set_to_none=True)
            lo = F.mse_loss(model(x_reg), y_reg)
            if not math.isfinite(float(lo.item())) or float(lo.item()) > 1e8:
                diverged = True
                break
            lo.backward()
            ps = list(model.parameters())
            gs = [p.grad.detach().clone() for p in ps]
            ds = _compute_base_deltas(ps, gs, base_m, lr, st, step)
            bw = _clone_params(model)
            if trust_m == "none":
                a = 1.0
            elif trust_m == "jvp_trust":
                dl = _jvp_rms_displacement(model, x_reg[:32], ds)
                a = min(1.0, 0.25 / max(dl, 1e-12)) if dl > 0.25 else 1.0
            elif trust_m == "oneshot_secant":
                a, _, _ = _oneshot_secant_trust(model, x_reg[:32], bw, ds, 0.25)
            elif trust_m == "backtrack_trust":
                a, _, _ = _backtrack_realized_trust(model, x_reg[:32], bw, ds, 0.25)
            else:
                a, _, _ = _armijo_backtrack(
                    lambda: F.mse_loss(model(x_reg), y_reg), model, bw, ds, gs
                )
            _apply_scaled_deltas(model, bw, ds, a)
        with torch.no_grad():
            f_mse = float(F.mse_loss(model(x_reg), y_reg).item())
        if not math.isfinite(f_mse):
            diverged = True
        hostile_reg[name] = {
            "final_mse": f_mse if not diverged else float("inf"),
            "diverged": diverged,
        }

    # 4B: Learned Dynamical System (Damped Pendulum + Lorenz-63 Rollout)
    # Part 4B-1: Damped nonlinear pendulum training under aggressive lr=0.05 & gain=10.0
    gen = torch.Generator(device=device).manual_seed(41)
    s0_tr = torch.empty(48, 2, device=device).uniform_(-2.0, 2.0, generator=gen)
    s0_va = torch.empty(32, 2, device=device).uniform_(-2.0, 2.0, generator=gen)
    dt_p = 0.05
    traj_tr = _generate_trajectories(_pendulum_rhs, s0_tr, dt_p, horizon=20)  # [48, 21, 2]
    traj_va = _generate_trajectories(_pendulum_rhs, s0_va, dt_p, horizon=40)  # [32, 41, 2]
    x_step_tr = traj_tr[:, :-1, :].reshape(-1, 2)
    dx_target_tr = (traj_tr[:, 1:, :] - traj_tr[:, :-1, :]).reshape(-1, 2) / dt_p
    x_step_va = traj_va[:, :-1, :].reshape(-1, 2)
    dx_target_va = (traj_va[:, 1:, :] - traj_va[:, :-1, :]).reshape(-1, 2) / dt_p

    pendulum_results: Dict[str, Any] = {}
    for arm in [
        "adam",
        "adam+grad_clip",
        "adam+1step_func_trust",
        "adam+rollout20_func_trust",
        "adam+armijo_1step",
    ]:
        dyn = ResidualDynamicsModel(2, 48, dt=dt_p, gain=10.0).to(device)
        _init_orth_module(dyn, seed=0, gain=0.8)
        st = {}
        for step in range(1, 181):
            dyn.zero_grad(set_to_none=True)
            lo = F.mse_loss(dyn(x_step_tr), dx_target_tr)
            if not math.isfinite(float(lo.item())) or float(lo.item()) > 1e8:
                break
            lo.backward()
            ps = list(dyn.parameters())
            if arm == "adam+grad_clip":
                torch.nn.utils.clip_grad_norm_(ps, max_norm=1.0)
            gs = [p.grad.detach().clone() for p in ps]
            ds = _compute_base_deltas(ps, gs, "adam", 0.05, st, step)
            bw = _clone_params(dyn)
            if arm in ("adam", "adam+grad_clip"):
                a = 1.0
            elif arm == "adam+1step_func_trust":
                a, _, _ = _backtrack_realized_trust(dyn, x_step_tr[:64], bw, ds, 0.30)
            elif arm == "adam+rollout20_func_trust":
                a, _, _ = _backtrack_rollout_trust(dyn, s0_tr[:16], 20, bw, ds, 0.15)
            else:
                a, _, _ = _armijo_backtrack(
                    lambda: F.mse_loss(dyn(x_step_tr), dx_target_tr), dyn, bw, ds, gs
                )
            _apply_scaled_deltas(dyn, bw, ds, a)
        with torch.no_grad():
            one_step_mse = float(F.mse_loss(dyn(x_step_va), dx_target_va).item())
            roll_pred = dyn.rollout(s0_va, 40)
            roll_err = (roll_pred - traj_va).square().mean(dim=(1, 2)).sqrt()
            roll_rmse = float(roll_err.mean().item())
            roll_max = float(roll_err.amax().item())
            div_frac = float(((roll_err > 5.0) | (~torch.isfinite(roll_err))).float().mean().item())
        pendulum_results[arm] = {
            "val_1step_mse": one_step_mse,
            "val_40step_rollout_rmse": roll_rmse,
            "val_40step_rollout_max": roll_max,
            "diverged_traj_frac": div_frac,
        }

    # Part 4B-2: Lorenz-63 Lyapunov Rollout Amplification Control
    # Shows that a TINY 1-step perturbation (delta_f_1step <= 0.02) amplifies hugely over H=40 steps!
    gen_l = torch.Generator(device=device).manual_seed(42)
    s0_l = torch.tensor([[0.1, 0.2, 2.0]], device=device) + 0.15 * torch.randn(24, 3, generator=gen_l, device=device)
    dt_l = 0.02
    traj_l_base = _generate_trajectories(_lorenz_rhs, s0_l, dt_l, horizon=80)
    # Fit a small Lorenz residual integrator cleanly, then apply a tiny 1-step perturbation delta_f <= 0.02
    lorenz_net = ResidualDynamicsModel(3, 64, dt=dt_l, gain=1.0).to(device)
    _init_orth_module(lorenz_net, seed=0, gain=0.8)
    x_l_tr = traj_l_base[:, :-1, :].reshape(-1, 3)
    dx_l_tr = (traj_l_base[:, 1:, :] - traj_l_base[:, :-1, :]).reshape(-1, 3) / dt_l
    opt_l = torch.optim.Adam(lorenz_net.parameters(), lr=0.01)
    for _ in range(250):
        opt_l.zero_grad(set_to_none=True)
        F.mse_loss(lorenz_net(x_l_tr), dx_l_tr).backward()
        opt_l.step()

    with torch.no_grad():
        base_l = _clone_params(lorenz_net)
        r_before = lorenz_net.rollout(s0_l, 80)
        # Perturb output layer to create a small constant drift with exact 1-step RMS == 0.02
        pert_deltas = [torch.zeros_like(p) for p in lorenz_net.parameters()]
        pert_deltas[-1] = torch.tensor([0.02, 0.02, 0.02], device=device)
        one_step_disp = _realized_rms_displacement(lorenz_net, x_l_tr[:64], base_l, pert_deltas, 1.0)
        _apply_scaled_deltas(lorenz_net, base_l, pert_deltas, 1.0)
        r_after = lorenz_net.rollout(s0_l, 80)
        roll_disp_80 = float((r_after - r_before).square().mean().sqrt().item())
        roll_disp_final = float((r_after[:, -1] - r_before[:, -1]).square().mean().sqrt().item())
        _restore_params(lorenz_net, base_l)

    return {
        "supervised_benign_spiral": benign_cls,
        "supervised_hostile_gain50_reg": hostile_reg,
        "pendulum_rollout_training": pendulum_results,
        "lorenz_lyapunov_amplification": {
            "one_step_delta_f": one_step_disp,
            "horizon_80_rollout_rms_disp": roll_disp_80,
            "horizon_80_terminal_rms_disp": roll_disp_final,
            "amplification_ratio_terminal_over_1step": roll_disp_final / max(one_step_disp, 1e-12),
        },
    }


# ---------------------------------------------------------------------------
# Exp 5 — Adversarial Probe Generalization & Blind-Spot Falsification
# ---------------------------------------------------------------------------
class LocalizedReLUMlp(nn.Module):
    """2-layer ReLU MLP whose hidden units can specialize to disjoint input regions."""

    def __init__(self, d_in: int = 2, n_maj: int = 16, n_rare: int = 16, d_out: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, n_maj + n_rare)
        self.fc2 = nn.Linear(n_maj + n_rare, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def run_exp5_probe_blindspots(device: torch.device) -> Dict[str, Any]:
    _set_seed(0)
    # -----------------------------------------------------------------------
    # 5A: i.i.d. Generalization vs Probe Size B_p in {2, 4, 8, 16, 64, 256}
    # Use a 2-component heteroscedastic / heavy-tailed mixture (95% core + 5% tail)
    # so displacement variance across inputs is realistic and non-uniform.
    # -----------------------------------------------------------------------
    gen = torch.Generator(device=device).manual_seed(51)
    n_test = 4096
    n_tail = int(0.05 * n_test)
    x_core = 0.4 * torch.randn(n_test - n_tail, 4, generator=gen, device=device)
    x_tail = 3.0 + 0.4 * torch.randn(n_tail, 4, generator=gen, device=device)
    x_iid_pool = torch.cat([x_core, x_tail], dim=0)
    perm = torch.randperm(n_test, generator=gen, device=device)
    x_iid_pool = x_iid_pool[perm]

    mlp_iid = SupervisedMLP(4, 48, 2, out_gain=1.0).to(device)
    _init_orth_module(mlp_iid, seed=0, gain=1.0)
    # Take a realistic random update step scaled so true test RMS == 0.25 (above delta_max=0.10)
    base_iid = _clone_params(mlp_iid)
    raw_d = [0.05 * torch.randn(p.shape, generator=gen, device=device) for p in mlp_iid.parameters()]
    # Measure per-sample squared displacement on all 4096 i.i.d. samples
    with torch.no_grad():
        y0_all = mlp_iid(x_iid_pool)
        _apply_scaled_deltas(mlp_iid, base_iid, raw_d, 1.0)
        y1_all = mlp_iid(x_iid_pool)
        _restore_params(mlp_iid, base_iid)
        # Also make a sharper localized step by perturbing a ReLU network active mainly on the 5% tail
    # ReLU net with heavy-tailed spatial sensitivity:
    relu_iid = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    _init_orth_module(relu_iid, seed=0, gain=1.0)
    with torch.no_grad():
        # Bias half the units to activate primarily on the 5% tail (x ~ +3)
        relu_iid[0].weight[16:].fill_(0.5)
        relu_iid[0].bias[16:].fill_(-3.5)
        y0_r = relu_iid(x_iid_pool)
        base_r = _clone_params(relu_iid)
        d_r = [torch.zeros_like(p) for p in relu_iid.parameters()]
        d_r[2][:, 16:] = 0.8  # Perturbs tail-active features
        d_r[2][:, :16] = 0.02  # Small perturbation on core features
        _apply_scaled_deltas(relu_iid, base_r, d_r, 1.0)
        y1_r = relu_iid(x_iid_pool)
        _restore_params(relu_iid, base_r)
        per_sample_sq_smooth = (y1_all - y0_all).square().mean(dim=-1)
        per_sample_sq_heavy = (y1_r - y0_r).square().mean(dim=-1)

    true_rms_smooth = float(per_sample_sq_smooth.mean().sqrt().item())
    true_rms_heavy = float(per_sample_sq_heavy.mean().sqrt().item())
    probe_size_stats: Dict[str, Any] = {}
    for b_p in [2, 4, 8, 16, 64, 256]:
        ratios_sm, ratios_hv = [], []
        miss_hv_15x = 0
        for trial in range(400):
            idx = torch.randint(0, n_test, (b_p,), generator=gen, device=device)
            p_sm = float(per_sample_sq_smooth[idx].mean().sqrt().item())
            p_hv = float(per_sample_sq_heavy[idx].mean().sqrt().item())
            ratios_sm.append(true_rms_smooth / max(p_sm, 1e-12))
            ratios_hv.append(true_rms_heavy / max(p_hv, 1e-12))
            if true_rms_heavy > 1.5 * p_hv:
                miss_hv_15x += 1
        rs_sm = torch.tensor(ratios_sm)
        rs_hv = torch.tensor(ratios_hv)
        probe_size_stats[f"B={b_p}"] = {
            "smooth_ratio_mean": float(rs_sm.mean().item()),
            "smooth_ratio_p95": float(torch.quantile(rs_sm, 0.95).item()),
            "smooth_ratio_max": float(rs_sm.max().item()),
            "heavy_ratio_mean": float(rs_hv.mean().item()),
            "heavy_ratio_p95": float(torch.quantile(rs_hv, 0.95).item()),
            "heavy_ratio_max": float(rs_hv.max().item()),
            "heavy_underestimate_gt_1.5x_frac": miss_hv_15x / 400.0,
        }

    # -----------------------------------------------------------------------
    # 5B: Adversarial Rare Tail Mode (1% Minority Cluster Blind Spot)
    # -----------------------------------------------------------------------
    gen5b = torch.Generator(device=device).manual_seed(52)
    n_maj_pts = 990
    n_rare_pts = 10  # Exact 1% minority cluster
    x_maj = 0.25 * torch.randn(n_maj_pts, 2, generator=gen5b, device=device)
    x_rare = torch.tensor([5.0, 5.0], device=device) + 0.25 * torch.randn(n_rare_pts, 2, generator=gen5b, device=device)
    y_maj = torch.zeros(n_maj_pts, 2, device=device)
    y_rare = torch.ones(n_rare_pts, 2, device=device)

    loc_net = LocalizedReLUMlp(2, 16, 16, 2).to(device)
    deterministic_orthogonal_(loc_net, seed=0)
    with torch.no_grad():
        # Units 0..15 cover majority cluster near (0,0); units 16..31 activate ONLY for x0+x1 > 6 (rare cluster!)
        loc_net.fc1.bias[:16].zero_()
        loc_net.fc1.weight[16:].fill_(1.0)
        loc_net.fc1.bias[16:].fill_(-6.0)
        loc_net.fc2.weight.zero_()
        loc_net.fc2.bias.zero_()
        # Calibrate fc2 so initial MSE on both majority and rare cluster is ~0
        h_rare_init = F.relu(loc_net.fc1(x_rare))
        # Solve least squares for units 16..31 to predict +1 on rare cluster
        # Since h_rare_init[:, 16:] > 0 and h_maj[:, 16:] == 0 identically:
        mean_act_rare = float(h_rare_init[:, 16:].mean().item())
        loc_net.fc2.weight[:, 16:].fill_(1.0 / (16.0 * mean_act_rare))

    with torch.no_grad():
        init_maj_mse = float(F.mse_loss(loc_net(x_maj), y_maj).item())
        init_rare_mse = float(F.mse_loss(loc_net(x_rare), y_rare).item())
        # Verify units 16..31 have strictly zero activation on majority probe x_maj[:16]
        probe_maj = x_maj[:16]
        rare_unit_act_on_probe = float(F.relu(loc_net.fc1(probe_maj))[:, 16:].abs().max().item())

    # Construct a parameter step that moves majority slightly (delta_probe = 0.005 < delta_g_min = 0.02!)
    # while destroying the rare cluster (e.g., aggressive gradient/perturbation on rare-mode weights)
    base_loc = _clone_params(loc_net)
    adv_deltas = [torch.zeros_like(p) for p in loc_net.parameters()]
    adv_deltas[3] = torch.tensor([0.005, 0.005], device=device)  # Tiny bias shift -> delta_probe = 0.005
    adv_deltas[2][:, 16:] = -0.35  # Large shift on rare-cluster weights -> invisible on majority probe!

    delta_max_5 = 0.10
    delta_min_5 = 0.02
    # Evaluate under (i) pure ceiling trust (delta_min=0), (ii) ParticleGAN band with delta_min=0.02 (amplifies!),
    # (iii) iterative backtracking on majority probe, and (iv) stratified probe containing 1 rare sample!
    a_ceil, d_probe_1, d_probe_ceil = _oneshot_secant_trust(loc_net, probe_maj, base_loc, adv_deltas, delta_max_5, 0.0)
    d_rare_ceil = _realized_rms_displacement(loc_net, x_rare, base_loc, adv_deltas, a_ceil)

    a_band, _, d_probe_band = _oneshot_secant_trust(loc_net, probe_maj, base_loc, adv_deltas, delta_max_5, delta_min_5)
    d_rare_band = _realized_rms_displacement(loc_net, x_rare, base_loc, adv_deltas, a_band)

    a_bt_maj, d_probe_bt, _ = _backtrack_realized_trust(loc_net, probe_maj, base_loc, adv_deltas, delta_max_5)
    d_rare_bt_maj = _realized_rms_displacement(loc_net, x_rare, base_loc, adv_deltas, a_bt_maj)

    # Positive mitigation: stratified probe (15 majority + 1 rare sample)
    probe_strat = torch.cat([x_maj[:15], x_rare[:1]], dim=0)
    a_bt_strat, d_probe_strat, _ = _backtrack_realized_trust(loc_net, probe_strat, base_loc, adv_deltas, delta_max_5)
    d_rare_bt_strat = _realized_rms_displacement(loc_net, x_rare, base_loc, adv_deltas, a_bt_strat)

    # Exact probability of missing a 1% minority cluster in a random slice of size b_p
    miss_prob_b16 = (1.0 - 0.01) ** 16
    miss_prob_b2 = (1.0 - 0.01) ** 2

    # -----------------------------------------------------------------------
    # 5C: Exact Probe-Jacobian Nullspace & Out-of-Distribution (OOD) Blind Spot
    # -----------------------------------------------------------------------
    gen5c = torch.Generator(device=device).manual_seed(53)
    x_id = 0.5 * torch.randn(64, 4, generator=gen5c, device=device)  # In-distribution ||x|| ~ 1
    x_ood = 4.0 + 0.5 * torch.randn(64, 4, generator=gen5c, device=device)  # Shifted OOD ||x|| ~ 8
    y_id = torch.sin(x_id[:, :2])

    ood_net = SupervisedMLP(4, 24, 2, out_gain=1.0).to(device)
    _init_orth_module(ood_net, seed=0, gain=1.0)
    # Train cleanly on ID data
    opt_ood = torch.optim.Adam(ood_net.parameters(), lr=0.01)
    for _ in range(100):
        opt_ood.zero_grad(set_to_none=True)
        F.mse_loss(ood_net(x_id), y_id).backward()
        opt_ood.step()

    # Compute exact probe Jacobian J_probe on B_p=16 ID samples: shape [32, P]
    probe_id = x_id[:16]
    params_ood = list(ood_net.parameters())
    n_par_ood = sum(p.numel() for p in params_ood)
    j_probe = _jacobian_matrix(ood_net, probe_id)  # [32, P]
    j_ood = _jacobian_matrix(ood_net, x_ood[:16])  # [32, P]

    _, _, vh = torch.linalg.svd(j_probe, full_matrices=True)
    rank_probe = int(j_probe.shape[0])
    null_basis = vh[rank_probe:].T  # [P, P - 32]
    # Project top right singular vector of J_ood onto ker(J_probe)
    _, _, vh_ood = torch.linalg.svd(j_ood, full_matrices=False)
    v_ood_top = vh_ood[0]
    v_null = null_basis @ (null_basis.T @ v_ood_top)
    v_null = v_null / v_null.norm().clamp_min(1e-12)
    # Scale v_null so realized probe displacement is tiny (<= 0.01) while OOD displacement is large!
    v_null_deltas: List[torch.Tensor] = []
    off = 0
    for p in params_ood:
        sz = p.numel()
        v_null_deltas.append((0.6 * v_null[off : off + sz]).reshape_as(p).to(dtype=p.dtype))
        off += sz

    base_ood = _clone_params(ood_net)
    jvp_null_id = _jvp_rms_displacement(ood_net, probe_id, v_null_deltas)
    real_null_id = _realized_rms_displacement(ood_net, probe_id, base_ood, v_null_deltas, 1.0)
    real_null_ood = _realized_rms_displacement(ood_net, x_ood, base_ood, v_null_deltas, 1.0)
    # Also test a natural (non-adversarial!) real ID gradient step on a polynomial-feature OOD shift
    ood_poly_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    _init_orth_module(ood_poly_net, seed=0, gain=1.0)
    ood_poly_net.zero_grad(set_to_none=True)
    F.mse_loss(ood_poly_net(x_id), y_id).backward()
    ps_p = list(ood_poly_net.parameters())
    gs_p = [p.grad.detach().clone() for p in ps_p]
    ds_p = _compute_base_deltas(ps_p, gs_p, "adam", 0.05, {}, 1)
    bw_p = _clone_params(ood_poly_net)
    a_id_bt, d_id_bt, _ = _backtrack_realized_trust(ood_poly_net, probe_id, bw_p, ds_p, delta_max_5)
    d_ood_natural = _realized_rms_displacement(ood_poly_net, x_ood, bw_p, ds_p, a_id_bt)

    return {
        "iid_probe_size_scaling": {
            "true_rms_smooth": true_rms_smooth,
            "true_rms_heavy_tail": true_rms_heavy,
            "by_probe_size": probe_size_stats,
        },
        "rare_mode_1pct_blindspot": {
            "miss_prob_b16": miss_prob_b16,
            "miss_prob_b2": miss_prob_b2,
            "init_maj_mse": init_maj_mse,
            "init_rare_mse": init_rare_mse,
            "rare_unit_act_on_maj_probe": rare_unit_act_on_probe,
            "delta_max": delta_max_5,
            "delta_min": delta_min_5,
            "raw_probe_disp": d_probe_1,
            "ceiling_only_alpha": a_ceil,
            "ceiling_only_probe_disp": d_probe_ceil,
            "ceiling_only_rare_disp": d_rare_ceil,
            "band_with_dmin_alpha": a_band,
            "band_with_dmin_probe_disp": d_probe_band,
            "band_with_dmin_rare_disp": d_rare_band,
            "backtrack_maj_probe_disp": d_probe_bt,
            "backtrack_maj_rare_disp": d_rare_bt_maj,
            "stratified_probe_alpha": a_bt_strat,
            "stratified_probe_disp": d_probe_strat,
            "stratified_rare_disp": d_rare_bt_strat,
        },
        "probe_nullspace_and_ood_blindspot": {
            "n_params": n_par_ood,
            "probe_constraints_B_times_dout": rank_probe,
            "nullspace_dim_lower_bound": n_par_ood - rank_probe,
            "nullspace_jvp_id_probe": jvp_null_id,
            "nullspace_realized_id_probe": real_null_id,
            "nullspace_realized_ood": real_null_ood,
            "nullspace_ood_over_id_ratio": real_null_ood / max(real_null_id, 1e-12),
            "natural_id_step_bt_alpha": a_id_bt,
            "natural_id_probe_disp": d_id_bt,
            "natural_ood_disp": d_ood_natural,
            "natural_ood_over_id_ratio": d_ood_natural / max(d_id_bt, 1e-12),
        },
    }


def run_all(device: torch.device) -> Dict[str, Any]:
    t_start = time.perf_counter()
    results = {
        "device": str(device),
        "torch_version": torch.__version__,
        "exp0_code_audit": run_exp0_code_audit(device),
        "exp1_reparameterization": run_exp1_reparameterization(device),
        "exp2_nonlinearity_gap": run_exp2_nonlinearity_gap(device),
        "exp3_transformer_attention": run_exp3_transformer_attention(device),
        "exp4_supervised_and_dynamics": run_exp4_supervised_and_dynamics(device),
        "exp5_probe_blindspots": run_exp5_probe_blindspots(device),
    }
    results["total_wall_seconds"] = time.perf_counter() - t_start
    return results


def _print_summary(res: Dict[str, Any]) -> None:
    print(f"=== P4 Function-Space Generality Audit ({res['device']}, {res['total_wall_seconds']:.2f}s) ===")

    e0 = res["exp0_code_audit"]
    print("\n[Exp 0: Actual ParticleGAN Code Audit]")
    bd = e0["band_scale_derivatives"]
    print(
        f"  _band_scale C^1? dmax(L={bd['at_delta_max_left']:.4f}, R={bd['at_delta_max_right']:.4f}); "
        f"true_kink={bd['true_amp_kink_delta']:.6f}(L={bd['at_amp_kink_left']:.2f}, R={bd['at_amp_kink_right']:.2f})"
    )
    os_sat = e0["oneshot_vs_backtrack_saturating_tanh"]
    print(
        f"  Saturating tanh (dmax={os_sat['delta_max']}): oneshot alpha={os_sat['oneshot_alpha']:.3f} -> "
        f"actual={os_sat['oneshot_actual_disp']:.5f} ({os_sat['oneshot_violation_ratio']:.2f}x violation!) vs "
        f"backtrack alpha={os_sat['backtrack_alpha']:.6f} -> actual={os_sat['backtrack_actual_disp']:.5f} "
        f"({os_sat['backtrack_evals']} evals)"
    )
    os_sup = e0["oneshot_amplification_superlinear_cubic"]
    print(
        f"  Superlinear cubic amplification (dmin={os_sup['delta_min']}, dmax={os_sup['delta_max']}): "
        f"d(1)={os_sup['delta_at_1']} -> alpha={os_sup['oneshot_alpha']:.2f} -> "
        f"actual={os_sup['oneshot_actual_disp']:.4f} ({os_sup['overshoot_vs_delta_max']:.2f}x dmax!)"
    )
    u38 = e0["dynamic_1e38_underflow_check"]
    print(
        f"  1e-24 grad step norm: UnifiedV12={u38['unified_v12_step_norm']:.5f} vs "
        f"Adam={u38['standard_adam_step_norm']:.5e}"
    )

    e1 = res["exp1_reparameterization"]
    print("\n[Exp 1: Functionally Equivalent Reparameterizations (alpha=1..1000)]")
    print(f"  Max init equiv error: {e1['max_initial_equiv_error']}")
    for arm, vals in e1["homogeneous_reparam"].items():
        d1, d1k = vals["1"]["step1_delta_f"], vals["1000"]["step1_delta_f"]
        r1, r1k = vals["1"]["step1_w1_over_w2_ratio"], vals["1000"]["step1_w1_over_w2_ratio"]
        m1, m1k = vals["1"]["final_mse_100"], vals["1000"]["final_mse_100"]
        print(
            f"  {arm:24s} | dF(1)={d1:.3e} dF(1k)={d1k:.3e} | "
            f"W1/W2(1)={r1:.3f} W1/W2(1k)={r1k:.3e} | mse(1)={m1:.4f} mse(1k)={m1k:.4f}"
        )
    print("  External Output Gain (gamma=1 vs 1000):")
    for arm, vals in e1["external_output_gain"].items():
        print(
            f"    {arm:22s} | dF(g=1)={vals['1']['step1_delta_f']:.4f} "
            f"dF(g=1000)={vals['1000']['step1_delta_f']:.4f}"
        )

    e2 = res["exp2_nonlinearity_gap"]
    print("\n[Exp 2: Nonlinearity Gap (JVP vs Realized)]")
    print("  16-layer unnormalized MLP:")
    for row in e2["deep_mlp16_sweep"]:
        print(
            f"    lr={row['lr']:.0e} | lin={row['delta_lin_jvp']:.3e} real={row['delta_real']:.3e} "
            f"Gamma={row['gamma_real_over_lin']:.2f} | post_jvp={row['jvp_trust_actual_disp']:.3e} "
            f"post_sec={row['oneshot_actual_disp']:.3e} post_bt={row['backtrack_actual_disp']:.3e}"
        )
    print("  16-layer MLP training (lr=0.05):")
    for arm, v in e2["deep_mlp16_train"].items():
        print(f"    {arm:22s} | mse={v['final_mse']:.5f} max_d={v['max_realized_disp']:.3e} viol={v['trust_violations']}")
    print("  16-layer saturating tanh ResNet:")
    for row in e2["saturating_tanh_resnet16_sweep"]:
        print(
            f"    lr={row['lr']:.0e} | lin={row['delta_lin_jvp']:.3e} real={row['delta_real']:.3e} "
            f"Gamma={row['gamma_real_over_lin']:.3f} | post_sec={row['oneshot_actual_disp']:.4f} "
            f"({row['oneshot_violation_ratio']:.2f}x) post_bt={row['backtrack_actual_disp']:.4f}"
        )
    rj = e2["tanh_rail_jump"]
    print(f"  Tanh rail jump: lin={rj['delta_lin_jvp']:.3e} real={rj['delta_real']:.4f} Gamma={rj['gamma_real_over_lin']:.2e}")
    bs = e2["bilinear_saddle_jvp_blindness"]
    print(
        f"  Bilinear saddle: lin={bs['delta_lin_jvp']:.3e} real={bs['delta_real']:.4f} "
        f"Gamma={bs['gamma_real_over_lin']:.1f} | jvp_alpha={bs['jvp_trust_alpha']:.2f} "
        f"jvp_post={bs['jvp_trust_actual_disp']:.4f} vs bt_post={bs['backtrack_actual_disp']:.4f}"
    )

    e3 = res["exp3_transformer_attention"]
    print("\n[Exp 3: Small Transformer Attention]")
    sw = e3["softmax_flip_exact_witness"]
    print(f"  Exact softmax flip witness: lin={sw['delta_lin_jvp']:.3e} real={sw['delta_real']:.4f} Gamma={sw['gamma_real_over_lin']:.2e}")
    for lr_k, arms in e3["transformer_training"].items():
        print(f"  Training ({lr_k}):")
        for arm, v in arms.items():
            print(
                f"    {arm:22s} | tr={v['train_mse']:.4f} va={v['val_mse']:.4f} "
                f"spikes={v['loss_spikes_gt_2x']:2d} max_logit={v['max_attn_logit']:.2f} "
                f"min_ent={v['min_attn_entropy']:.3f} ms={v['ms_per_step']:.2f}"
            )

    e4 = res["exp4_supervised_and_dynamics"]
    print("\n[Exp 4: Supervised + Dynamical System Rollout]")
    print("  Benign spiral classification:")
    for arm, v in e4["supervised_benign_spiral"].items():
        print(f"    {arm:22s} | val_acc={v['val_acc']:.4f} val_loss={v['val_loss']:.4f} ms={v['ms_per_step']:.2f}")
    print("  Hostile gain=50 regression:")
    for arm, v in e4["supervised_hostile_gain50_reg"].items():
        print(f"    {arm:22s} | final_mse={v['final_mse']:.5f} diverged={v['diverged']}")
    print("  Pendulum 40-step rollout training:")
    for arm, v in e4["pendulum_rollout_training"].items():
        print(
            f"    {arm:26s} | 1step_mse={v['val_1step_mse']:.4f} "
            f"roll40_rmse={v['val_40step_rollout_rmse']:.4f} div_frac={v['diverged_traj_frac']:.2f}"
        )
    la = e4["lorenz_lyapunov_amplification"]
    print(
        f"  Lorenz-63 Lyapunov check: 1-step dF={la['one_step_delta_f']:.4f} -> "
        f"80-step terminal disp={la['horizon_80_terminal_rms_disp']:.4f} "
        f"({la['amplification_ratio_terminal_over_1step']:.1f}x amplification)"
    )

    e5 = res["exp5_probe_blindspots"]
    print("\n[Exp 5: Adversarial Probe Blind Spots]")
    for bk, v in e5["iid_probe_size_scaling"]["by_probe_size"].items():
        print(
            f"  i.i.d. {bk:6s} | smooth p95={v['smooth_ratio_p95']:.2f} | "
            f"heavy p95={v['heavy_ratio_p95']:.2f} max={v['heavy_ratio_max']:.2f} "
            f"underest>1.5x={v['heavy_underestimate_gt_1.5x_frac']:.1%}"
        )
    rm = e5["rare_mode_1pct_blindspot"]
    print(
        f"  1% Rare Mode: P(miss B=16)={rm['miss_prob_b16']:.1%}, P(miss B=2)={rm['miss_prob_b2']:.1%} | "
        f"ceil: probe={rm['ceiling_only_probe_disp']:.4f} rare={rm['ceiling_only_rare_disp']:.3f} | "
        f"band(+dmin): alpha={rm['band_with_dmin_alpha']:.2f} probe={rm['band_with_dmin_probe_disp']:.4f} "
        f"rare={rm['band_with_dmin_rare_disp']:.3f} | "
        f"stratified: alpha={rm['stratified_probe_alpha']:.4f} rare={rm['stratified_rare_disp']:.4f}"
    )
    ns = e5["probe_nullspace_and_ood_blindspot"]
    print(
        f"  Nullspace/OOD: P={ns['n_params']}, constraints={ns['probe_constraints_B_times_dout']}, "
        f"null_dim>={ns['nullspace_dim_lower_bound']} | "
        f"null_jvp_id={ns['nullspace_jvp_id_probe']:.2e} real_id={ns['nullspace_realized_id_probe']:.4f} "
        f"real_ood={ns['nullspace_realized_ood']:.4f} ({ns['nullspace_ood_over_id_ratio']:.1f}x) | "
        f"natural_step: id={ns['natural_id_probe_disp']:.4f} ood={ns['natural_ood_disp']:.4f} "
        f"({ns['natural_ood_over_id_ratio']:.1f}x)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    res = run_all(device)
    _print_summary(res)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
