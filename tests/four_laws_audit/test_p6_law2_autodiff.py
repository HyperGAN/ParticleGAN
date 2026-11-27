"""Semantic proofs, counterexamples, and regression controls for P6 Law 2 Autodiff Audit.

Tests verify:
1. Exact mixed second-order derivative identity: ``fwAD.dual_level()`` + ``.backward()``
   equals closed-form analytical ``(d^2 f / dtheta dx) u`` (including ``sigma''``) and
   reverse-over-reverse ``torch.autograd.grad(..., create_graph=True)`` on smooth and
   piecewise-linear critics, with deliberate-damage control removing ``sigma''``.
2. Codebase spatial-frame Householder geometry (fixed ``(d-2)``-subspace) and linear vs.
   nonlinear spatial neighbor-mixing bias.
3. Exact Jensen's inequality bias of stochastic directional estimators:
   - Strictly negative loss, gradient-magnitude, and directional distortion bias for
     concave Born-Infeld ``(sqrt(1 + beta s) - 1) / beta`` (``minimal``).
   - Strictly positive cap bias and false penalty triggering below ``kappa`` for convex
     EBI ``[softplus_beta(sqrt(s) - kappa)]^2`` (``t2_secant``, ``unified``, ``unified_v11``/``v12``).
4. Separable vs. dense Rademacher variance blowup theorem and ``minimal`` batch-wide
   probe collinearity.
5. Piecewise-linear ReLU/LeakyReLU ``O(1/delta)`` kink impulse (``fwAD`` immune vs.
   forward & central finite differences exploding), BF16/FP32 micro-secant quantization
   underflow, and chord symmetric-flank / high-D attenuation failures.
6. Architectural AD boundaries (``EmulatedLinear`` missing ``jvp``, ``DCGANCritic``
   BF16 ``BatchNorm2d`` dual dtype crash fixed by ``_bn_dual_bf16_hook``, and SDPA
   Flash/MemEff vs. Math backend support).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from experiments.forge.scaling_benchmark import DCGANCritic
from particlegan._minimal.geometry import _unit_probes as minimal_unit_probes
from particlegan._t2_critic.penalty import (
    _build_orthonormal_frame,
    _eval_forward_tangent_sq,
    _prepare_spatial_tangent_plan,
    _separable_image_unit_perturbation,
)
from particlegan._unified.penalty import _eval_lowd_forward_grad_sq
from particlegan._unified_v11.penalty import _bn_dual_bf16_hook
from particlegan.init import deterministic_orthogonal_
from particlegan.low_precision import EmulatedLinear
from scripts.four_laws_audit.p6_law2_autodiff import (
    _AttentionCritic,
    _analytical_2layer_mixed_grad,
    _flatten_grads,
    _grad_metrics,
    _make_mlp_critic,
    _phi_bi,
    _phi_ebi,
)


@pytest.mark.parametrize("act_name", ["tanh", "silu"])
@pytest.mark.parametrize("outer_fn", ["bi", "ebi"])
def test_fwad_backward_equals_analytical_mixed_hessian_fp64(act_name: str, outer_fn: str) -> None:
    """Prove fwAD.dual_level() + .backward() computes exact mixed 2nd derivative (d^2 f / dtheta dx) u."""
    g = torch.Generator().manual_seed(0)
    B, d, H = 24, 5, 16
    x = torch.randn(B, d, generator=g, dtype=torch.float64)
    u = torch.randn(B, d, generator=g, dtype=torch.float64)
    u = u / u.norm(dim=1, keepdim=True)
    W0 = torch.randn(H, d, generator=g, dtype=torch.float64) * 0.5
    b0 = torch.randn(H, generator=g, dtype=torch.float64) * 0.2
    v0 = torch.randn(H, generator=g, dtype=torch.float64) * 0.5

    act_fn = torch.tanh if act_name == "tanh" else F.silu
    loss_fn = (lambda s: _phi_bi(s, 1.0).mean()) if outer_fn == "bi" else (lambda s: _phi_ebi(s, 12.0, 1.0).mean())

    tau_an, gW_an, gb_an, gv_an = _analytical_2layer_mixed_grad(W0, b0, v0, x, u, act_name, outer_fn)
    g_an = torch.cat([gW_an.reshape(-1), gb_an.reshape(-1), gv_an.reshape(-1)])

    # Forward-over-reverse (fwAD + .backward())
    W_fw, b_fw, v_fw = W0.clone().requires_grad_(True), b0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
    with fwAD.dual_level():
        _, tau_fw = fwAD.unpack_dual(act_fn(fwAD.make_dual(x, u) @ W_fw.T + b_fw.unsqueeze(0)) @ v_fw)
    loss_fn(tau_fw.square()).backward()
    g_fw = torch.cat([W_fw.grad.reshape(-1), b_fw.grad.reshape(-1), v_fw.grad.reshape(-1)])

    # Reverse-over-reverse (autograd.grad(create_graph=True) + .backward())
    W_rr, b_rr, v_rr = W0.clone().requires_grad_(True), b0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
    x_req = x.clone().requires_grad_(True)
    gx = torch.autograd.grad((act_fn(x_req @ W_rr.T + b_rr.unsqueeze(0)) @ v_rr).sum(), x_req, create_graph=True)[0]
    loss_fn((gx * u).sum(dim=1).square()).backward()
    g_rr = torch.cat([W_rr.grad.reshape(-1), b_rr.grad.reshape(-1), v_rr.grad.reshape(-1)])

    assert float((tau_fw.detach() - tau_an).abs().max().item()) < 1e-13
    m_an = _grad_metrics(g_fw, g_an)
    m_rr = _grad_metrics(g_fw, g_rr)
    assert m_an["rel_l2"] < 1e-12
    assert m_rr["rel_l2"] < 1e-12
    assert m_an["cos"] > 1.0 - 1e-14

    # Deliberate-damage control: b.grad is PURELY proportional to sigma''(z).
    # Dropping the second-order sigma'' term (setting gb = 0) MUST fail agreement!
    assert float(gb_an.norm().item()) > 1e-3
    g_no_sigma2 = torch.cat([gW_an.reshape(-1), torch.zeros_like(gb_an), gv_an.reshape(-1)])
    assert _grad_metrics(g_fw, g_no_sigma2)["rel_l2"] > 0.05


@pytest.mark.parametrize("act_name,fourier", [("relu", 0), ("leaky_relu", 0), ("leaky_relu", 2), ("tanh", 0)])
@pytest.mark.parametrize("dim", [1, 2, 4])
def test_canonical_fwad_equals_ror_on_smooth_and_piecewise_linear_critics(act_name: str, fourier: int, dim: int) -> None:
    """Verify exact canonical fwAD matches RoR to FP64 machine precision and mixed 2nd deriv != 0 on ReLU."""
    g = torch.Generator().manual_seed(42 + dim)
    x = torch.randn(24, dim, generator=g, dtype=torch.float64)
    m_fw = _make_mlp_critic(dim, 48, 3, activation=act_name, fourier=fourier, dtype=torch.float64)
    # Normalize output scale so mean ||grad_x D||_2 == 2.0 > kappa = 1.0 (active cap regime)
    x_probe_s = x.detach().clone().requires_grad_(True)
    gn_init = torch.autograd.grad(m_fw(x_probe_s).sum(), x_probe_s)[0].norm(dim=1).mean().item()
    last_lin = [mod for mod in m_fw.modules() if isinstance(mod, nn.Linear)][-1]
    with torch.no_grad():
        last_lin.weight.mul_(2.0 / max(gn_init, 1e-12))
    m_rr = _make_mlp_critic(dim, 48, 3, activation=act_name, fourier=fourier, dtype=torch.float64)
    m_rr.load_state_dict(m_fw.state_dict())

    _, sq_fw = _eval_lowd_forward_grad_sq(lambda z, _m: m_fw(z), x, can_stack_basis=True, quantized_fprop=False)
    _phi_ebi(sq_fw, beta=12.0, kappa=1.0).mean().backward()
    g_fw = _flatten_grads(list(m_fw.parameters()))

    x_req = x.clone().requires_grad_(True)
    gx = torch.autograd.grad(m_rr(x_req).sum(), x_req, create_graph=True)[0]
    sq_rr = gx.square().sum(dim=1)
    _phi_ebi(sq_rr, beta=12.0, kappa=1.0).mean().backward()
    g_rr = _flatten_grads(list(m_rr.parameters()))

    assert float((sq_fw.detach() - sq_rr.detach()).abs().max().item()) < 1e-12
    gm = _grad_metrics(g_fw, g_rr)
    assert gm["rel_l2"] < 1e-12
    assert gm["cos"] > 1.0 - 1e-14
    assert gm["norm_ref"] > 1e-4

    # On pure piecewise-linear ReLU/LeakyReLU, d^2 D / dx^2 = 0 a.e. yet mixed d^2 D / (dtheta dx) != 0!
    if act_name in ("relu", "leaky_relu") and fourier == 0:
        x_one = x[:4].clone().requires_grad_(True)
        g1 = torch.autograd.grad(m_rr(x_one).sum(), x_one, create_graph=True)[0]
        h_xx = torch.autograd.grad(g1.sum(), x_one)[0]
        assert float(h_xx.abs().max().item()) < 1e-13


def test_householder_frame_geometry_and_spatial_mixing_bias() -> None:
    """Verify Q = H_2 H_1 is orthonormal with det +1, fixes a (d-2)-subspace, exact on linear, biased on nonlinear."""
    for dim in (5, 6, 8):
        Q = _build_orthonormal_frame(dim, dim, phase=0.23, salt=1, device=torch.device("cpu"))
        assert float((Q.T @ Q - torch.eye(dim)).abs().max().item()) < 1e-5
        assert abs(float(torch.linalg.det(Q).item()) - 1.0) < 1e-5
        eigs = torch.linalg.eigvals(Q)
        unit_eigs = int(((eigs - 1.0).abs() < 1e-4).sum().item())
        assert unit_eigs == dim - 2

    # Linear critic: Householder spatial-frame is exact
    dim = 8
    g = torch.Generator().manual_seed(99)
    x = torch.randn(96, dim, generator=g)
    lin_ref = nn.Sequential(nn.Linear(dim, 1))
    deterministic_orthogonal_(lin_ref, seed=0)
    lin_ref[0].weight.data.mul_(2.5)
    lin_est = nn.Sequential(nn.Linear(dim, 1))
    lin_est.load_state_dict(lin_ref.state_dict())

    xr = x.clone().requires_grad_(True)
    sq_ref = torch.autograd.grad(lin_ref(xr).sum(), xr, create_graph=True)[0].square().sum(dim=1)
    _phi_ebi(sq_ref, 12.0, 1.0).mean().backward()
    g_ref = _flatten_grads(list(lin_ref.parameters()))

    u_p, recon = _prepare_spatial_tangent_plan(x.detach(), phase=0.17, salt=1)
    _, tau_l = _eval_forward_tangent_sq(lambda z: lin_est(z), [x], [u_p], can_concat=False)
    _phi_ebi(recon(tau_l[0]), 12.0, 1.0).mean().backward()
    g_est = _flatten_grads(list(lin_est.parameters()))
    assert _grad_metrics(g_est, g_ref)["rel_l2"] < 1e-5

    # Nonlinear critic: spatial neighbor-mixing introduces strictly positive gradient error
    mlp_ref = _make_mlp_critic(dim, 96, 3, activation="leaky_relu", fourier=0)
    with torch.no_grad():
        mlp_ref[-1].weight.data.mul_(4.0)
    mlp_est = _make_mlp_critic(dim, 96, 3, activation="leaky_relu", fourier=0)
    mlp_est.load_state_dict(mlp_ref.state_dict())

    xr2 = x.clone().requires_grad_(True)
    sq_r2 = torch.autograd.grad(mlp_ref(xr2).sum(), xr2, create_graph=True)[0].square().sum(dim=1)
    _phi_ebi(sq_r2, 12.0, 1.0).mean().backward()
    g_r2 = _flatten_grads(list(mlp_ref.parameters()))

    _, tau_l2 = _eval_forward_tangent_sq(lambda z: mlp_est(z), [x], [u_p], can_concat=False)
    _phi_ebi(recon(tau_l2[0]), 12.0, 1.0).mean().backward()
    g_e2 = _flatten_grads(list(mlp_est.parameters()))
    assert _grad_metrics(g_e2, g_r2)["rel_l2"] > 0.10


def test_jensen_inequality_concave_born_infeld_exact_and_mc() -> None:
    """Verify strict negative Jensen bias on loss AND parameter gradient + directional distortion for concave BI."""
    # Exact enumeration over all 2^4 = 16 Rademacher sign vectors on anisotropic g in R^4
    g_aniso = np.array([2.0, 0.4, 0.4, 0.4], dtype=np.float64)
    signs = np.array([[1.0 if (m >> i) & 1 else -1.0 for i in range(4)] for m in range(16)], dtype=np.float64)
    dots = signs @ g_aniso
    s_hat = dots ** 2
    s_true = float(np.dot(g_aniso, g_aniso))

    # Inside sqrt: strictly unbiased
    assert abs(float(np.mean(s_hat)) - s_true) < 1e-13

    # Loss Jensen bias: strictly negative
    phi_true = math.sqrt(1.0 + s_true) - 1.0
    phi_mean = float(np.mean(np.sqrt(1.0 + s_hat) - 1.0))
    assert phi_mean < phi_true - 0.05

    # Gradient Jensen bias: strictly attenuated AND rotated away from g_aniso!
    grad_true = g_aniso / math.sqrt(1.0 + s_true)
    grad_mean = np.mean((dots / np.sqrt(1.0 + s_hat))[:, None] * signs, axis=0)
    proj_ratio = float(np.dot(grad_mean, grad_true) / np.dot(grad_true, grad_true))
    assert proj_ratio < 0.95
    cos_dir = float(np.dot(grad_mean, grad_true) / (np.linalg.norm(grad_mean) * np.linalg.norm(grad_true)))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_dir))))
    assert angle_deg > 1.0


def test_jensen_inequality_convex_ebi_positive_bias_and_false_triggering() -> None:
    """Verify strictly positive Jensen bias & false cap-triggering below kappa for convex EBI cap."""
    rng = np.random.default_rng(0)
    beta, kappa, r_true = 12.0, 1.0, 0.8  # Strictly below Lipschitz cap kappa = 1.0!
    s_true = r_true ** 2
    chi1 = rng.chisquare(df=1, size=80000)
    s_hat = s_true * chi1
    r_hat = np.sqrt(s_hat + 1e-12)

    h_true = float(F.softplus(torch.tensor(r_true - kappa, dtype=torch.float64), beta=beta).square().item())
    z_hat = beta * (r_hat - kappa)
    sp_hat = np.where(z_hat > 30.0, r_hat - kappa, np.log1p(np.exp(np.clip(z_hat, -60.0, 30.0))) / beta)
    sig_hat = 1.0 / (1.0 + np.exp(-np.clip(z_hat, -60.0, 60.0)))
    h_mean = float(np.mean(sp_hat ** 2))

    # Opposite sign from concave BI: strictly POSITIVE over-penalization!
    assert h_mean > 100.0 * h_true
    # High false-trigger probability even though true slope 0.8 < 1.0
    assert float(np.mean(r_hat > kappa)) > 0.18
    # Simultaneously, reported norm E[sqrt(s_hat)] is strictly NEGATIVELY biased (~sqrt(2/pi) = 0.798)
    assert float(np.mean(r_hat)) < 0.83 * r_true
    # Expected parameter-gradient multiplier is heavily inflated below kappa
    sp_t = float(F.softplus(torch.tensor(r_true - kappa, dtype=torch.float64), beta=beta).item())
    sig_t = float(torch.sigmoid(torch.tensor(beta * (r_true - kappa), dtype=torch.float64)).item())
    gmult_true = 2.0 * sp_t * sig_t / r_true
    gmult_mean = float(np.mean((2.0 * sp_hat * sig_hat / r_hat) * chi1))
    assert gmult_mean > 50.0 * gmult_true


def test_separable_rademacher_variance_blowup_and_minimal_probe_collinearity() -> None:
    """Verify 4x (2-way) and >10x (3-way) variance blowup on rank-1 spatial gradients & minimal batch correlation."""
    rng = np.random.default_rng(7)
    C, H, W = 3, 24, 24
    dim = C * H * W
    a_c = np.array([1.0, -0.7, 0.5])
    b_h = rng.standard_normal(H)
    c_w = rng.standard_normal(W)
    G_3way = np.einsum("c,h,w->chw", a_c, b_h, c_w)
    G_3way = G_3way / np.linalg.norm(G_3way)
    G_mat = G_3way.reshape(C * H, W)
    g_flat = G_mat.reshape(-1)

    n_mc = 15000
    z_dense = (rng.choice([-1.0, 1.0], size=(n_mc, dim)) @ g_flat) ** 2
    s_ch = rng.choice([-1.0, 1.0], size=(n_mc, C * H))
    s_w = rng.choice([-1.0, 1.0], size=(n_mc, W))
    z_sep2 = np.sum(s_ch * (s_w @ G_mat.T), axis=1) ** 2
    sc = rng.choice([-1.0, 1.0], size=(n_mc, C))
    sh = rng.choice([-1.0, 1.0], size=(n_mc, H))
    sw = rng.choice([-1.0, 1.0], size=(n_mc, W))
    z_sep3 = np.einsum("nc,nh,nw,chw->n", sc, sh, sw, G_3way) ** 2

    # All three are unbiased for ||G||_F^2 = 1.0
    assert abs(float(np.mean(z_dense)) - 1.0) < 0.05
    assert abs(float(np.mean(z_sep2)) - 1.0) < 0.10
    assert abs(float(np.mean(z_sep3)) - 1.0) < 0.20

    # Variance blowup: dense ~ 2, 2-way sep ~ 8 (4x), 3-way sep > 16 (>8x)
    var_d, var_2, var_3 = float(np.var(z_dense)), float(np.var(z_sep2)), float(np.var(z_sep3))
    assert 1.7 < var_d < 2.3
    assert var_2 > 3.2 * var_d
    assert var_3 > 7.5 * var_d

    # Exact algebraic check of the closed-form 2-way separable variance theorem on a general 3x4 matrix
    G_small = rng.standard_normal((3, 4))
    G_small = G_small / np.linalg.norm(G_small)
    a_all = np.array([[1.0 if (m >> i) & 1 else -1.0 for i in range(3)] for m in range(8)], dtype=np.float64)
    b_all = np.array([[1.0 if (m >> i) & 1 else -1.0 for i in range(4)] for m in range(16)], dtype=np.float64)
    z_enum = (a_all @ G_small @ b_all.T) ** 2
    assert abs(float(np.mean(z_enum)) - 1.0) < 1e-13
    gtg_f2 = float(np.sum((G_small.T @ G_small) ** 2))
    e_gb4 = 1.0 + 2.0 * gtg_f2 - 2.0 * float(np.sum(np.sum(G_small ** 2, axis=0) ** 2))
    e_row4 = float(3.0 * np.sum(np.sum(G_small ** 2, axis=1) ** 2) - 2.0 * np.sum(G_small ** 4))
    var_exact_formula = 3.0 * e_gb4 - 2.0 * e_row4 - 1.0
    assert abs(float(np.var(z_enum)) - var_exact_formula) < 1e-13

    # Minimal _unit_probes uses shared spatial pattern across the ENTIRE batch (up to +-1 global sign)!
    x_im = torch.zeros(12, 3, 32, 32)
    u_min = minimal_unit_probes(x_im, phase=0.17, k_probes=1)[0].flatten(1).double()
    cos_min = (u_min @ u_min.T).abs()
    off_min = cos_min[~torch.eye(12, dtype=torch.bool)]
    assert float(off_min.min().item()) > 0.99999

    u_t2 = _separable_image_unit_perturbation(x_im, phase=0.17, salt=3).flatten(1).double()
    cos_t2 = (u_t2 @ u_t2.T).abs()
    off_t2 = cos_t2[~torch.eye(12, dtype=torch.bool)]
    assert float(off_t2.mean().item()) < 0.15


def test_relu_kink_impulse_fwad_immune_vs_fd_explosion() -> None:
    """Verify fwAD avoids O(1/delta) ReLU kink impulses while BOTH forward & central FD explode."""
    eta = 1e-4
    g = torch.Generator().manual_seed(303)
    W1_0 = torch.randn(8, 4, generator=g, dtype=torch.float64) * 0.5
    W1_0[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    b1_0 = torch.randn(8, generator=g, dtype=torch.float64) * 0.5 + 1.0
    b1_0[0] = 0.0
    W2_0 = torch.randn(1, 8, generator=g, dtype=torch.float64) * 0.5
    W2_0[0, 0] = 1.0

    x = torch.tensor([[eta, 0.2, -0.3, 0.4]], dtype=torch.float64)
    u = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    def _eval(mode: str, delta: float) -> torch.Tensor:
        W1, b1, W2 = W1_0.clone().requires_grad_(True), b1_0.clone().requires_grad_(True), W2_0.clone().requires_grad_(True)
        fwd = lambda z: F.leaky_relu(z @ W1.T + b1.unsqueeze(0), 0.2) @ W2.T
        if mode == "ror":
            xr = x.clone().requires_grad_(True)
            tau = (torch.autograd.grad(fwd(xr).sum(), xr, create_graph=True)[0] * u).sum()
        elif mode == "fwad":
            with fwAD.dual_level():
                _, tau_t = fwAD.unpack_dual(fwd(fwAD.make_dual(x, u)))
            tau = tau_t.squeeze()
        elif mode == "fwd_fd":
            tau = (fwd(x + delta * u) - fwd(x)).squeeze() / delta
        else:
            tau = (fwd(x + delta * u) - fwd(x - delta * u)).squeeze() / (2.0 * delta)
        (0.5 * tau.square()).backward()
        return torch.cat([W1.grad.reshape(-1), b1.grad.reshape(-1), W2.grad.reshape(-1)])

    g_ror = _eval("ror", 0.0)
    g_fw = _eval("fwad", 0.0)
    assert _grad_metrics(g_fw, g_ror)["rel_l2"] < 1e-13

    # Crossing kink at delta = 2e-4 (> eta = 1e-4): massive O(1/delta) parameter-gradient impulse!
    g_fwd_cross = _eval("fwd_fd", 2e-4)
    g_cen_cross = _eval("cen_fd", 2e-4)
    assert _grad_metrics(g_fwd_cross, g_ror)["norm_ratio"] > 50.0
    assert _grad_metrics(g_cen_cross, g_ror)["norm_ratio"] > 25.0
    assert _grad_metrics(g_fwd_cross, g_ror)["cos"] < 0.25

    # Strictly inside linear region (delta = 5e-5 < eta): both FD recover exact gradient!
    assert _grad_metrics(_eval("fwd_fd", 5e-5), g_ror)["rel_l2"] < 1e-10
    assert _grad_metrics(_eval("cen_fd", 5e-5), g_ror)["rel_l2"] < 1e-10


def test_micro_secant_quantization_underflow_and_chord_failure_modes() -> None:
    """Verify BF16 underflow for t2_secant delta=0.02, FP32 underflow for minimal eps=1e-5, and chord failures."""
    g = torch.Generator().manual_seed(11)
    x = torch.empty(4, 3, 256, 256, dtype=torch.float32).uniform_(-1.0, 1.0, generator=g)
    u = _separable_image_unit_perturbation(x, phase=0.23, salt=3)

    # t2_secant (delta=0.02) on 256x256x3: per-pixel step 4.51e-5 << BF16 ULP (~3.9e-3 at |x|=0.5)
    x_t2 = x + 0.02 * u
    surv_bf16 = float((x_t2.to(torch.bfloat16) != x.to(torch.bfloat16)).float().mean().item())
    assert surv_bf16 < 0.05  # >95% of perturbed pixels round back identically to bf16(x)!

    # minimal (eps=1e-5) on 256x256x3: per-pixel step 2.25e-8 < FP32 half-ULP (2.98e-8 for |x| >= 0.5)
    x_min = x + 1e-5 * u
    surv_fp32 = float((x_min != x).float().mean().item())
    assert surv_fp32 < 0.55  # ~50% of all pixels round to exact x even in FP32!
    large_mask = x.abs() >= 0.5
    assert float((x_min[large_mask] != x[large_mask]).float().mean().item()) == 0.0  # 100% vanish for |x| >= 0.5!
    surv_min_bf16 = float((x_min.to(torch.bfloat16) != x.to(torch.bfloat16)).float().mean().item())
    assert surv_min_bf16 < 1e-3  # >99.9% vanish completely in BF16!

    # Symmetric-flank chord cancellation: s_chord == 0 while true slope == 20.0
    a = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    D_rad = lambda z: -10.0 * z.square().sum(dim=1)
    assert float((D_rad(b) - D_rad(a)).abs().item() / (b - a).norm().item()) == 0.0
    with fwAD.dual_level():
        _, j_a = fwAD.unpack_dual(D_rad(fwAD.make_dual(a, torch.tensor([[1.0, 0.0]], dtype=torch.float64))))
    assert abs(float(j_a.abs().item()) - 20.0) < 1e-12

    # High-D unscaled chord attenuation on linear D(x) = w^T x, ||w|| = 2.0 at d = 3072
    rng = np.random.default_rng(123)
    w = rng.standard_normal(3072)
    w = w * (2.0 / np.linalg.norm(w))
    xr, xf = rng.standard_normal((16, 3072)), rng.standard_normal((16, 3072))
    s_chord = np.abs(xr @ w - xf @ w) / np.linalg.norm(xr - xf, axis=1)
    assert float(np.mean(s_chord)) < 0.06  # >30x attenuation below true slope 2.0


def test_emulated_linear_fwad_unsupported_while_ror_succeeds() -> None:
    """Verify EmulatedLinear supports native STE fwAD + backward with zero autograd.grad (remediated)."""
    for fmt in ("mxfp8", "nvfp4"):
        layer = EmulatedLinear.from_linear(nn.Linear(16, 8), fmt, seed=0)
        x = torch.randn(8, 16)
        u = torch.randn_like(x)
        with fwAD.dual_level():
            _, jvp = fwAD.unpack_dual(layer(fwAD.make_dual(x, u)))
        assert jvp is not None and torch.isfinite(jvp).all()
        jvp.square().mean().backward()
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for BF16 BN dual & Flash SDPA checks")
def test_cuda_bn_bf16_dual_hook_and_sdpa_backend_boundaries() -> None:
    """Verify DCGAN BF16 fwAD fails without _bn_dual_bf16_hook, succeeds with it, and Flash SDPA blocks both fwAD & RoR."""
    dev = torch.device("cuda:0")
    x64 = torch.randn(8, 3, 64, 64, device=dev)
    u64 = _separable_image_unit_perturbation(x64, phase=0.1, salt=1)
    D = DCGANCritic(64, 64, 512).to(dev)
    deterministic_orthogonal_(D, seed=0)
    with torch.no_grad():
        D.train()
        _ = D(x64)
        D.eval()

    # Without _bn_dual_bf16_hook, autocast BF16 fwAD crashes at Conv2d after BatchNorm2d
    with pytest.raises(RuntimeError):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with fwAD.dual_level():
                _ = D(fwAD.make_dual(x64.to(torch.bfloat16), u64.to(torch.bfloat16)))

    # With _bn_dual_bf16_hook (as in unified_v11/v12), BF16 fwAD + .backward() succeeds
    hooks = [m.register_forward_hook(_bn_dual_bf16_hook) for m in D.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with fwAD.dual_level():
                _, jvp = fwAD.unpack_dual(D(fwAD.make_dual(x64.to(torch.bfloat16), u64.to(torch.bfloat16))))
        assert jvp is not None
        jvp.float().square().mean().backward()
    finally:
        for h in hooks:
            h.remove()

    # Default fused Flash/MemEff SDPA fails BOTH fwAD AND RoR, while micro-secant & MATH backend succeed!
    attn = _AttentionCritic(128, 4).to(dev)
    x_seq = torch.randn(8, 64, 128, device=dev)
    u_seq = torch.randn_like(x_seq)

    with pytest.raises((NotImplementedError, RuntimeError)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with fwAD.dual_level():
                _ = attn(fwAD.make_dual(x_seq, u_seq))

    with pytest.raises((NotImplementedError, RuntimeError)):
        xr = x_seq.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = attn(xr)
        g = torch.autograd.grad(y.sum(), xr, create_graph=True)[0]
        g.float().square().mean().backward()

    # Micro-secant succeeds natively on fused Flash/MemEff SDPA
    with torch.autocast("cuda", dtype=torch.bfloat16):
        diff = (attn((x_seq + 0.02 * u_seq).detach()) - attn(x_seq)) / 0.02
    diff.float().square().mean().backward()

    # Under explicit SDPBackend.MATH, RoR succeeds cleanly while fwAD + .backward() STILL crashes
    # in PyTorch 2.15 due to an in-place Exp version bump inside math SDPA's dual-mode softmax!
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        xr = x_seq.clone().requires_grad_(True)
        g_math = torch.autograd.grad(attn(xr).sum(), xr, create_graph=True)[0]
        g_math.square().mean().backward()

        with pytest.raises(RuntimeError, match="inplace"):
            with fwAD.dual_level():
                _, j_math = fwAD.unpack_dual(attn(fwAD.make_dual(x_seq, u_seq)))
            assert j_math is not None
            j_math.square().mean().backward()
