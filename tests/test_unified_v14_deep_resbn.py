"""Unit, semantic-control, zero-primitive, Schatten-6, d_couple, stiff_gate, continuation, and clockfree tests for ``unified_v14`` Bug 2 fixes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, ParticlePrior, Recipe, get_recipe, init
from particlegan._unified_v11 import UnifiedV11EquilibratedOptimizer, UnifiedV11StepRecord
from particlegan._unified_v12 import UnifiedV12Optimizer
from particlegan._unified_v14 import UnifiedV14Optimizer
from particlegan._unified_v14 import optimizer as v14_opt_mod
from particlegan.optim.dualnorm import NORMALIZED_FAMILIES


class _ResBNBlock(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.scale = 0.5 / float(max(1, depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.rms_norm(x, (x.shape[-1],))
        h = self.fc2(torch.nn.functional.silu(self.fc1(h)))
        h = torch.nn.functional.rms_norm(h, (h.shape[-1],))
        return x + self.scale * h


class _ResBNNet(nn.Module):
    def __init__(self, d_in: int, d_out: int, width: int, depth: int) -> None:
        super().__init__()
        self.stem = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([_ResBNBlock(width, depth) for _ in range(depth)])
        self.head = nn.Linear(width, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class _ConvDCGANCritic(nn.Module):
    """DCGAN-style critic: 4D Conv2d layers + 1-row Linear head (zero hidden 2D matrices with min(r, c) >= 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.head = nn.Linear(16 * 2 * 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.leaky_relu(self.conv1(x), 0.2)
        h = nn.functional.leaky_relu(self.conv2(h), 0.2).flatten(1)
        return self.head(h).squeeze(-1)


class _ConvDCGANGen(nn.Module):
    def __init__(self, z_dim: int = 8) -> None:
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim, 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.relu(self.deconv1(z[:, :, None, None]))
        return torch.tanh(self.deconv2(h))


def _forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("forbidden primitive called")


def test_unified_v14_registration_and_stiff_gate_regimes() -> None:
    assert "unified_v14_equil" in NORMALIZED_FAMILIES
    assert DEFAULT_RECIPE == "axiom"
    rec = get_recipe("unified_v14")
    assert rec.optimizer_family == "unified_v14_equil"
    assert rec.rhbi_self_coupling == "feydy_atomic"
    assert Recipe.from_dict(rec.to_dict()) == rec

    torch.manual_seed(0)
    w_g = nn.Linear(16, 16, bias=False)
    opt_g = UnifiedV14Optimizer([{"params": [w_g.weight], "role": "generator", "lr": 0.014}], smoothing=0.0061)

    # 1. Healthy regime (ema_fg >= 0.60): stiff_gate == 0.0 regardless of ema_flip_cos:
    for fg in (1.0, 0.75, 0.60):
        for fl in (0.2, 0.0, -0.4, -0.8):
            opt_g._ema_func_gain = fg
            opt_g._ema_flip_cos = fl
            assert opt_g.current_stiff_gate() == 0.0

    # 2. Normalized deep ResBN regime (ema_fg in [0.056, 0.55], ema_flip_cos >= -0.50):
    #    stiff_gate MUST remain identically 0.0 and MUST NOT allocate m_unit momentum state!
    for fg in (0.55, 0.45, 0.30, 0.24, 0.15, 0.11, 0.08, 0.06, 0.056):
        for fl in (0.30, 0.0, -0.20, -0.45, -0.50):
            opt_g._ema_func_gain = fg
            opt_g._ema_flip_cos = fl
            assert opt_g.current_stiff_gate() == 0.0, f"False stiff_gate at fg={fg}, flip={fl}"

    opt_g._ema_func_gain = 0.28
    opt_g._ema_flip_cos = 0.05
    w_g.weight.grad = torch.randn_like(w_g.weight)
    opt_g.step()
    assert "m_unit" not in opt_g.state[w_g.weight], "False stiff_gate allocated m_unit on ResBN-like trunk!"

    # 2b. Decoupled generator Tikhonov smoothing fade: at ema_fg <= 0.18 with stiff_gate == 0.0,
    #     s_base fades cleanly to 0.0 (avoiding Tikhonov double-damping on top of pre_gain = ema_fg),
    #     while at ema_fg >= 0.30 (and healthy 1.0) s_base remains full self.smoothing:
    w_sm_lo = nn.Parameter(torch.randn(16, 16))
    w_sm_zero = nn.Parameter(w_sm_lo.detach().clone())
    g_sm = torch.randn(16, 16) * 1e-4
    opt_sm_lo = UnifiedV14Optimizer([{"params": [w_sm_lo], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_sm_zero = UnifiedV14Optimizer([{"params": [w_sm_zero], "role": "generator", "lr": 0.014}], smoothing=0.0)
    opt_sm_lo._ema_func_gain = 0.16
    opt_sm_zero._ema_func_gain = 0.16
    w_sm_lo.grad = g_sm.clone()
    w_sm_zero.grad = g_sm.clone()
    opt_sm_lo.step()
    opt_sm_zero.step()
    assert opt_sm_lo.current_stiff_gate() == 0.0
    assert torch.equal(w_sm_lo.detach(), w_sm_zero.detach()), "s_base must fade to 0.0 at ema_fg <= 0.18!"

    w_sm_hi = nn.Parameter(torch.randn(16, 16))
    w_sm_hi_v12 = nn.Parameter(w_sm_hi.detach().clone())
    opt_sm_hi = UnifiedV14Optimizer([{"params": [w_sm_hi], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_sm_hi_v12 = UnifiedV12Optimizer([{"params": [w_sm_hi_v12], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_sm_hi._ema_func_gain = 1.0
    opt_sm_hi_v12._ema_func_gain = 1.0
    w_sm_hi.grad = g_sm.clone()
    w_sm_hi_v12.grad = g_sm.clone()
    opt_sm_hi.step()
    opt_sm_hi_v12.step()
    assert torch.equal(w_sm_hi.detach(), w_sm_hi_v12.detach()), "Healthy regime (ema_fg=1.0) must match v12 smoothing!"

    # 3. Sustained period-2 flipping regime (e.g., grid100_res16_tanh):
    opt_g._ema_func_gain = 0.45
    opt_g._ema_flip_cos = -0.58
    assert 0.0 < opt_g.current_stiff_gate() < 1.0
    opt_g._ema_func_gain = 0.28
    opt_g._ema_flip_cos = -0.66
    assert opt_g.current_stiff_gate() == 1.0

    # 4. Floor-overshoot regime (e.g., grid100_mlp16 / settled res16_tanh at ema_fg <= 0.050):
    opt_g._ema_func_gain = 0.05
    opt_g._ema_flip_cos = 0.0
    assert opt_g.current_stiff_gate() == 1.0
    w_g.weight.grad = torch.randn_like(w_g.weight)
    opt_g.step()
    assert "m_unit" in opt_g.state[w_g.weight], "Floor overshoot failed to engage stiff_gate momentum!"


def test_unified_v14_period2_flip_feedback_and_thermal_recovery_guard() -> None:
    torch.manual_seed(0)
    # 1. Closed-loop period-2 sign flipping drives ema_fg down to 0.05 floor -> stiff_gate == 1.0:
    w_flip = nn.Parameter(torch.randn(16, 16))
    rec = UnifiedV11StepRecord()
    rec._thermal_scale = 1.0
    opt_flip = UnifiedV14Optimizer([{"params": [w_flip], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_flip._unified_v11_record = rec
    opt_flip._ema_func_gain = 0.50
    g_pat = torch.randn(16, 16)
    for step_i in range(90):
        w_flip.grad = (1.0 if step_i % 2 == 0 else -1.0) * g_pat
        opt_flip.step()
    assert opt_flip._ema_flip_cos < -0.70
    assert opt_flip._ema_func_gain <= 0.051
    assert opt_flip.current_stiff_gate() == 1.0
    # Even if flipping later relaxes to 0.0, floor-overshoot branch keeps stiff_gate == 1.0:
    opt_flip._ema_flip_cos = 0.0
    assert opt_flip.current_stiff_gate() == 1.0

    # 2. Non-flipping ResBN-like noisy gradients at ema_fg=0.35 never trigger flip controller or stiff_gate:
    w_res = nn.Parameter(torch.randn(16, 16))
    opt_res = UnifiedV14Optimizer([{"params": [w_res], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_res._unified_v11_record = rec
    opt_res._ema_func_gain = 0.35
    for _ in range(30):
        w_res.grad = torch.randn(16, 16)
        opt_res.step()
    assert opt_res._ema_flip_cos > -0.45
    assert opt_res._ema_func_gain == 0.35
    assert opt_res.current_stiff_gate() == 0.0
    assert "m_unit" not in opt_res.state[w_res]

    # 3. PR #612 coh unmasking & s_therm < 0.5 recovery guard:
    #    When stiff_gate > 0 and s_therm >= 0.5, small delta_rel masked by low coh still pulls ema_fg down
    #    if free_ratio < ema_fg; when s_therm < 0.5 (cooled), ema_fg must not drift upward:
    g_lin = nn.Linear(8, 8, bias=False)
    opt_u = UnifiedV14Optimizer([{"params": [g_lin.weight], "role": "generator", "lr": 0.014}], smoothing=0.0)
    rec_u = UnifiedV11StepRecord()
    rec_u.ema_data_scale = 1.0
    rec_u.ema_push_res = 0.0
    rec_u._thermal_scale = 0.10  # cooled (< 0.5)
    opt_u._unified_v11_record = rec_u
    opt_u._ema_func_gain = 0.25
    x_p = torch.randn(8, 8)
    with torch.no_grad():
        y_pre = g_lin(x_p).float()
    rec_u._g_probe_cache = (g_lin, (x_p,), {}, y_pre)
    g_lin.weight.grad = torch.randn_like(g_lin.weight)
    opt_u.step()
    assert opt_u._ema_func_gain <= 0.25, "Cooled step (s_therm < 0.5) must not recover ema_func_gain upward!"


def test_unified_v14_bandpass_d_couple_and_zerosync_ema_parity() -> None:
    torch.manual_seed(0)
    grad_d = torch.randn(16, 16)

    def _critic_step_norm(g_fg: float, ema_margin: float, margin_init: bool, *, lazy_tensor: bool = False) -> float:
        w0 = torch.eye(16) * 0.8  # sigma_1 = 0.8 <= tau=2.0 -> Schatten-6 identically 0
        p = nn.Parameter(w0.clone())
        opt_d = UnifiedV14Optimizer([{"params": [p], "role": "critic", "lr": 0.02}], smoothing=0.0)
        opt_d.critic = nn.Linear(16, 16, bias=False)
        rec = UnifiedV11StepRecord()
        rec._g_func_gain = float(g_fg)
        if lazy_tensor:
            rec._ema_margin_t = torch.tensor(float(ema_margin), dtype=torch.float64)
            rec._margin_init_t = torch.tensor(1.0 if margin_init else 0.0, dtype=torch.float64)
        else:
            rec.margin_initialized = bool(margin_init)
            rec.ema_margin = float(ema_margin)
        opt_d._unified_v11_record = rec
        p.grad = grad_d.clone()
        with mock.patch.object(torch.Tensor, "new_tensor", _forbid):
            opt_d.step()
        return float((p.detach() - w0).norm().item())

    base = _critic_step_norm(1.0, 0.65, True)
    # 1. Inside critic-dominance band (0.50 <= ema_margin <= 1.10) with g_fg = 0.35 -> scaled by 0.35:
    coupled = _critic_step_norm(0.35, 0.65, True)
    assert abs(coupled / base - 0.35) < 1e-5

    # 2. Exact parity between lazy GPU-style EMA tensors and pre-flushed floats (zero new_tensor):
    coupled_lazy = _critic_step_norm(0.35, 0.65, True, lazy_tensor=True)
    assert abs(coupled_lazy - coupled) < 1e-7

    # 3. Floor at 0.12 when g_fg = 0.05 inside the band:
    floored = _critic_step_norm(0.05, 0.65, True)
    assert abs(floored / base - 0.12) < 1e-5

    # 4. Releases cleanly to 1.0 outside the band (near-equilibrium <= 0.28, inverted < 0, collapse >= 1.40, uninit):
    for ema_m, m_init in ((0.28, True), (0.15, True), (0.0, True), (-2.0, True), (1.40, True), (3.0, True), (0.65, False)):
        rel = _critic_step_norm(0.25, ema_m, m_init)
        assert abs(rel / base - 1.0) < 1e-5, f"Expected d_couple=1.0 at margin={ema_m}, init={m_init}"

    # 5. Releases cleanly to 1.0 whenever g_fg >= 0.75 (healthy or stiff-published g_fg=1.0), without touching lazy EMAs:
    p_hi = nn.Parameter(torch.eye(16) * 0.8)
    opt_hi = UnifiedV14Optimizer([{"params": [p_hi], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_hi.critic = nn.Linear(16, 16, bias=False)
    rec_hi = UnifiedV11StepRecord()
    rec_hi._g_func_gain = 1.0
    rec_hi._ema_margin_t = torch.tensor(0.50, dtype=torch.float64)
    rec_hi._margin_init_t = torch.tensor(1.0, dtype=torch.float64)
    opt_hi._unified_v11_record = rec_hi
    p_hi.grad = grad_d.clone()
    opt_hi.step()
    assert rec_hi._ema_margin_t is not None, "g_fg >= 0.75 must NOT flush pending margin EMA tensors!"


def test_unified_v14_schatten6_spectral_ceiling_deadband_invariance_and_dcgan_zero_overhead() -> None:
    torch.manual_seed(0)

    # 1. Exact deadband below tau=2.0: orthogonal weight with sigma_1 = 1.5 <= 2.0 gets ZERO Schatten-6 perturbation:
    u_mat, _, vh_mat = torch.linalg.svd(torch.randn(16, 24), full_matrices=False)
    w_small = (u_mat * 1.5) @ vh_mat  # aspect = 1.0, sigma_max = 1.5 <= tau = 2.0
    p_s6 = nn.Parameter(w_small.clone())
    g0 = torch.randn_like(w_small)
    opt_s6 = UnifiedV14Optimizer([{"params": [p_s6], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_s6.critic = nn.Linear(24, 16, bias=False)
    p_s6.grad = g0.clone()
    opt_s6.step()
    # Expected pure 0-NS Frob-MP step: factor = aspect * sqrt(m) = 1.0 * 4.0 = 4.0:
    expected = w_small - (0.02 * 4.0 / float(g0.norm().item())) * g0
    assert torch.allclose(p_s6.detach(), expected, atol=1e-6, rtol=1e-6)
    assert not any(isinstance(v, torch.Tensor) for v in opt_s6.state[p_s6].values()), "Non-stiff critic must stay 0 B/param!"

    # 2. Active ceiling above tau=2.0: contracts BOTH runaway sigma_1 AND sigma_2 < 2.8 while preserving bulk sigma_k <= 1.0:
    sv_init = torch.tensor([5.5, 4.4] + [1.0] * 14, dtype=torch.float32)
    w_hot = (u_mat * sv_init.unsqueeze(0)) @ vh_mat
    p_hot = nn.Parameter(w_hot.clone())
    opt_hot = UnifiedV14Optimizer([{"params": [p_hot], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_hot.critic = nn.Linear(24, 16, bias=False)
    g_tiny = (u_mat[:, -1:] @ vh_mat[-1:, :]) * 1e-4
    for _ in range(25):
        p_hot.grad = g_tiny.clone()
        opt_hot.step()
    sv_after = torch.linalg.svdvals(p_hot.detach())
    assert float(sv_after[0].item()) < 2.8, f"sigma_1 failed to contract: {sv_after[0].item()}"
    assert float(sv_after[1].item()) < 2.8, f"sigma_2 failed to contract: {sv_after[1].item()}"
    assert float(sv_after[5].item()) > 0.95, f"bulk singular values were damaged: {sv_after[5].item()}"

    # 3. Zero-grad and NaN-grad safety: Schatten-6 must NOT fire on zero or NaN gradients (slow path):
    p_zero = nn.Parameter(w_hot.clone())
    opt_zero = UnifiedV14Optimizer([{"params": [p_zero], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_zero.critic = nn.Linear(24, 16, bias=False)
    p_zero.grad = torch.zeros_like(p_zero)
    opt_zero.step()
    assert torch.equal(p_zero.detach(), w_hot)
    p_zero.grad = torch.full_like(p_zero, float("nan"))
    opt_zero.step()
    assert torch.equal(p_zero.detach(), w_hot)

    # 3b. Slow-path Schatten-6 coverage (b1 > 0 and tiny <1e-18 grad both contract runaway singular values cleanly):
    p_slow = nn.Parameter(w_hot.clone())
    opt_slow = UnifiedV14Optimizer([{"params": [p_slow], "role": "critic", "lr": 0.02, "betas": (0.5, 0.999)}], smoothing=0.0)
    opt_slow.critic = nn.Linear(24, 16, bias=False)
    for _ in range(25):
        p_slow.grad = g_tiny.clone()
        opt_slow.step()
    sv_slow = torch.linalg.svdvals(p_slow.detach())
    assert float(sv_slow[0].item()) < 2.8 and float(sv_slow[1].item()) < 2.8

    p_tiny = nn.Parameter(w_hot.clone())
    opt_tiny = UnifiedV14Optimizer([{"params": [p_tiny], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_tiny.critic = nn.Linear(24, 16, bias=False)
    p_tiny.grad = g_tiny * 1e-20  # forces slow path (< 1e-18) while g_active stays True (> 1e-38)
    opt_tiny.step()
    assert float(torch.linalg.svdvals(p_tiny.detach())[0].item()) < 5.5

    # 4. Degree-0 gradient scale invariance (1e-12..1e12) and exact group-split invariance:
    w_a = (u_mat * sv_init.unsqueeze(0)) @ vh_mat
    w_b = (u_mat * (sv_init * 0.9).unsqueeze(0)) @ vh_mat
    p1a, p2a = nn.Parameter(w_a.clone()), nn.Parameter(w_b.clone())
    p1b, p2b = nn.Parameter(w_a.clone()), nn.Parameter(w_b.clone())
    ga, gb = torch.randn_like(w_a), torch.randn_like(w_b)
    opt_one = UnifiedV14Optimizer([{"params": [p1a, p2a], "role": "critic", "lr": 0.02}], smoothing=0.0061)
    opt_two = UnifiedV14Optimizer(
        [{"params": [p1b], "role": "critic", "lr": 0.02}, {"params": [p2b], "role": "critic", "lr": 0.02}],
        smoothing=0.0061,
    )
    opt_one.critic = nn.Linear(24, 16, bias=False)
    opt_two.critic = nn.Linear(24, 16, bias=False)
    p1a.grad, p2a.grad = ga.clone(), gb.clone()
    p1b.grad, p2b.grad = ga.clone(), gb.clone()
    opt_one.step()
    opt_two.step()
    assert torch.equal(p1a.detach(), p1b.detach()) and torch.equal(p2a.detach(), p2b.detach())

    # Degree-0 gradient scaling invariance (with smoothing=0.0):
    p_s1, p_s2 = nn.Parameter(w_a.clone()), nn.Parameter(w_a.clone())
    o_s1 = UnifiedV14Optimizer([{"params": [p_s1], "role": "critic", "lr": 0.02}], smoothing=0.0)
    o_s2 = UnifiedV14Optimizer([{"params": [p_s2], "role": "critic", "lr": 0.02}], smoothing=0.0)
    o_s1.critic = nn.Linear(24, 16, bias=False)
    o_s2.critic = nn.Linear(24, 16, bias=False)
    p_s1.grad = ga * 1e-10
    p_s2.grad = ga * 1e10
    o_s1.step()
    o_s2.step()
    assert torch.allclose(p_s1.detach(), p_s2.detach(), atol=1e-6, rtol=1e-6)

    # 5. Zero Schatten-6 calls, zero Conv2d NS, zero Critic NS, and zero forbidden primitives on DCGAN-style Conv2d GAN:
    rec_c = get_recipe(
        "unified_v14",
        z_dim=8,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=4,
    )
    gc = init.deterministic_orthogonal_(_ConvDCGANGen(z_dim=8))
    dc = init.deterministic_orthogonal_(_ConvDCGANCritic())
    tc = GANTrainer(rec_c, gc, dc, max_steps=4)
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "svdvals", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "eigvalsh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(v14_opt_mod, "_schatten6_spectral_decay_2d", _forbid),
        mock.patch.object(v14_opt_mod, "_schatten6_spectral_decay_batched", _forbid),
        mock.patch.object(v14_opt_mod, "quintic_orthogonalize_2d", _forbid),
        mock.patch.object(v14_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        tc.step(torch.randn(8, 3, 8, 8), generator_real=torch.randn(8, 3, 8, 8))
    assert tc.opt_d.ns_2d_activations == 0
    assert tc.opt_g.ns_2d_activations == 0


def test_unified_v14_generator_free_and_cross_family_checkpoint_rejection() -> None:
    torch.manual_seed(0)
    rec_gf = get_recipe(
        "unified_v14",
        z_dim=4,
        batch_size=16,
        num_particles=32,
        prior_kind="particles",
        prior_reg=0.0,
        total_steps=6,
    )
    d_gf = init.deterministic_orthogonal_(nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 1)))
    prior_gf = ParticlePrior(num_particles=32, z_dim=4, learnable=True)
    _, opt_d_gf = rec_gf.make_optimizers(nn.Identity(), d_gf, prior_gf)
    assert isinstance(opt_d_gf, UnifiedV14Optimizer)
    assert opt_d_gf._generator_free is True
    for p in d_gf.parameters():
        p.grad = torch.randn_like(p)
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(torch.Tensor, "new_tensor", _forbid),
    ):
        opt_d_gf.step()
    assert opt_d_gf.critic_ns_calls == 1

    sd_v14 = opt_d_gf.state_dict()
    assert "unified_v14_equil" in sd_v14
    opt_v12 = UnifiedV12Optimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    opt_v11 = UnifiedV11EquilibratedOptimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError):
        opt_v12.load_state_dict(sd_v14)
    with pytest.raises(ValueError):
        opt_v11.load_state_dict(sd_v14)
    with pytest.raises(ValueError):
        opt_d_gf.load_state_dict(opt_v12.state_dict())

    # Tamper rejection on unified_v14_equil metadata fields:
    for bad_cos in (1.5, -1.5, float("nan"), True):
        bad_sd = {**sd_v14, "unified_v14_equil": {**sd_v14["unified_v14_equil"], "ema_flip_cos": bad_cos}}
        with pytest.raises(ValueError, match="ema_flip_cos"):
            opt_d_gf.validate_state_dict(bad_sd)
    for bad_fg in (0.0, -0.1, 1.5, float("nan"), True):
        bad_sd = {**sd_v14, "unified_v14_equil": {**sd_v14["unified_v14_equil"], "ema_func_gain": bad_fg}}
        with pytest.raises(ValueError, match="ema_func_gain"):
            opt_d_gf.validate_state_dict(bad_sd)
    bad_schema = {**sd_v14, "unified_v14_equil": {**sd_v14["unified_v14_equil"], "schema": 99}}
    with pytest.raises(ValueError, match="metadata"):
        opt_d_gf.validate_state_dict(bad_schema)


def test_unified_v14_fresh_process_continuation_across_boundaries(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # 1. Deep ResBN continuation across active d_couple + active Schatten-6 spectral ceiling boundary:
    dim = 64
    rec = get_recipe(
        "unified_v14",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=8,
        total_steps=16,
        training_precision="float32",
    )
    g = _ResBNNet(8, dim, width=32, depth=4)
    d = _ResBNNet(dim, 1, width=32, depth=4)
    p = MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False)
    init.deterministic_orthogonal_(g, seed=60)
    init.deterministic_orthogonal_(d, seed=61)
    init.deterministic_orthogonal_(p, seed=62)
    # Deliberately scale one hidden critic matrix above tau=2.0 so Schatten-6 is actively contracting across the checkpoint:
    with torch.no_grad():
        d.blocks[0].fc1.weight.mul_(3.2)

    trainer = GANTrainer(rec, g, d, prior=p, seed=0)
    # Put generator into ResBN multi-group ema_fg=0.35 (stiff_gate=0.0, d_couple active!):
    trainer.opt_g._ema_func_gain = 0.35
    trainer.opt_d.record._g_func_gain = 0.35
    trainer.opt_d.record.margin_initialized = True
    trainer.opt_d.record.ema_margin = 0.55

    batches = [torch.randn(8, dim, generator=torch.Generator().manual_seed(2100 + i)) for i in range(10)]
    for i in range(5):
        trainer.step(batches[i])
    assert trainer.opt_g.current_stiff_gate() == 0.0
    assert all("m_unit" not in trainer.opt_g.state.get(param, {}) for param in g.parameters())
    assert trainer.opt_d.ns_2d_activations == 0
    s1_mid = float(torch.linalg.matrix_norm(d.blocks[0].fc1.weight.detach().float(), ord=2).item())
    assert s1_mid < 3.2, "Schatten-6 must actively contract >tau weight across training steps"

    ckpt_path = tmp_path / "ckpt_v14_resbn.pt"
    out_path = tmp_path / "resumed_v14_resbn.pt"
    torch.save({"state": trainer.state_dict(), "batches": batches[5:]}, ckpt_path)
    for i in range(5, 10):
        trainer.step(batches[i])
    ref_sample = trainer.sample(4)
    ref_state = trainer.state_dict()
    ref_g_fg = trainer.opt_d.record._g_func_gain
    ref_ema_m = trainer.opt_d.record.ema_margin

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v14_deep_resbn import _ResBNNet

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(
    rec,
    _ResBNNet(8, 64, width=32, depth=4),
    _ResBNNet(64, 1, width=32, depth=4),
    prior=MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False),
    seed=321,
)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save(
    {{"sample": tr.sample(4), "models": tr.state_dict()["models"],
      "g_fg": tr.opt_d.record._g_func_gain, "ema_m": tr.opt_d.record.ema_margin}},
    {str(out_path)!r},
)
"""
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    resumed = torch.load(out_path, weights_only=False)
    assert torch.equal(ref_sample, resumed["sample"])
    assert resumed["g_fg"] == ref_g_fg
    assert resumed["ema_m"] == ref_ema_m
    for m_name in ("G", "D", "prior"):
        for k, v in ref_state["models"][m_name].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, resumed["models"][m_name][k])
            else:
                assert v == resumed["models"][m_name][k]

    # 2. Fresh-process bit-exact continuation across active period-2 flip -> stiff_gate activation boundary:
    w_a0 = torch.randn(16, 16)
    ga_pat = torch.randn(16, 16)
    w_cont = nn.Parameter(w_a0.clone())
    rec_hi = UnifiedV11StepRecord()
    rec_hi._thermal_scale = 1.0
    opt_c = UnifiedV14Optimizer([{"params": [w_cont], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_c._unified_v11_record = rec_hi
    opt_c._ema_func_gain = 0.50
    grads_seq = [(1.0 if i % 2 == 0 else -1.0) * ga_pat for i in range(60)]
    for i in range(6):
        w_cont.grad = grads_seq[i].clone()
        opt_c.step()
    assert opt_c._prev_flip_sketch is not None
    assert opt_c.current_stiff_gate() == 0.0  # pre-activation!
    ckpt_flip = tmp_path / "ckpt_v14_flip.pt"
    out_flip = tmp_path / "out_v14_flip.pt"
    torch.save({"w": w_cont.detach().clone(), "opt": opt_c.state_dict(), "grads": grads_seq[6:]}, ckpt_flip)
    for i in range(6, 60):
        w_cont.grad = grads_seq[i].clone()
        opt_c.step()
    assert opt_c.current_stiff_gate() == 1.0  # post-activation!

    script_flip = f"""
import torch
from torch import nn
from particlegan._unified_v11 import UnifiedV11StepRecord
from particlegan._unified_v14 import UnifiedV14Optimizer

pl = torch.load({str(ckpt_flip)!r}, weights_only=False)
w = nn.Parameter(pl["w"].clone())
opt = UnifiedV14Optimizer([dict(params=[w], role="generator", lr=0.014)], smoothing=0.0061)
rec = UnifiedV11StepRecord()
rec._thermal_scale = 1.0
opt._unified_v11_record = rec
opt.load_state_dict(pl["opt"])
for g in pl["grads"]:
    w.grad = g.clone()
    opt.step()
torch.save({{"w": w.detach(), "ema_fg": opt._ema_func_gain, "ema_flip": opt._ema_flip_cos, "stiff": opt.current_stiff_gate()}}, {str(out_flip)!r})
"""
    subprocess.run([sys.executable, "-c", script_flip], check=True, env=env)
    res_flip = torch.load(out_flip, weights_only=False)
    assert torch.equal(w_cont.detach(), res_flip["w"])
    assert opt_c._ema_func_gain == res_flip["ema_fg"]
    assert opt_c._ema_flip_cos == res_flip["ema_flip"]
    assert opt_c.current_stiff_gate() == res_flip["stiff"] == 1.0


def test_unified_v14_clockfree_audit(tmp_path: Path) -> None:
    from experiments.forge.clockfree import run_clockfree, verify_probe

    repo_p = Path(__file__).resolve().parents[1]
    task = json.loads((repo_p / "configs/forge/tasks/clockfree_audit.json").read_text())
    cand = json.loads((repo_p / "configs/forge/ideas/unified-v13.json").read_text())
    cand["id"] = "unified-v14"
    cand["recipe_overrides"]["optimizer_family"] = "unified_v14_equil"
    audit_res = run_clockfree(
        {"candidate": cand, "protocol": {"seed": 0}},
        task,
        tmp_path / "clockfree_v14",
        "cpu",
    )
    comparisons, audit = verify_probe(task, audit_res["evidence"])
    assert audit["unexplained_clock_dependencies"] == []
    assert len(comparisons) == 4
    assert all(item["reference_sha256"] == item["changed_sha256"] for item in comparisons)
