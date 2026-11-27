"""Unit, mathematical-invariant, and schema-2 continuation tests for L2 ``particlegan._axiom.prior``."""

from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from torch import nn

from particlegan._axiom import prior as axiom_prior
from particlegan._axiom.prior import (
    SCHEMA_VERSION,
    AxiomPriorRegularizer,
    allocate_deficit_quotas,
    axiom_pullback_step,
    central_pullback_cov,
    coalesce_pythagorean_ltv,
    has_positive_prior_sigma,
    should_use_axiom_pullback,
    validate_axiom_prior_state,
    whitened_burg_pullback_loss,
)
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior
from particlegan.priors import ManipulatedPrior, NormalNoiseManipulator

PRIOR_PATH = Path(axiom_prior.__file__).resolve()


def _radon_bool_cc(node: ast.AST) -> int:
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp),
        ):
            cc += 1
        elif isinstance(child, ast.With):
            cc += len(child.items)
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
    return cc


def _mccabe_c901(node: ast.AST) -> int:
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp),
        ):
            cc += 1
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
    return cc


def _make_mock_trainer(
    *,
    out_shape: Tuple[int, ...] = (2,),
    latent_dim: int = 2,
    num_particles: int = 48,
    sigma_z: float = 0.20,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> Tuple[Any, torch.Generator]:
    dim = math.prod(out_shape)
    g_rng = torch.Generator(device="cpu").manual_seed(seed)

    def _lin(in_f: int, out_f: int) -> nn.Linear:
        layer = nn.Linear(in_f, out_f)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(out_f, in_f, generator=g_rng) * (0.6 / math.sqrt(in_f)))
            layer.bias.copy_(torch.randn(out_f, generator=g_rng) * 0.05)
        return layer

    class _ShapeGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(_lin(latent_dim, 32), nn.Tanh(), _lin(32, dim))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z).reshape(int(z.shape[0]), *out_shape)

    class _FlatCrit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(_lin(dim, 24), nn.Tanh(), _lin(24, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x.reshape(int(x.shape[0]), dim))

    gen_m, crit_m = _ShapeGen().to(device), _FlatCrit().to(device)
    prior = MoGParticlePrior(num_particles, latent_dim, sigma=sigma_z).to(device)
    with torch.no_grad():
        prior.z.copy_(torch.randn(num_particles, latent_dim, generator=g_rng).to(device))
    reg = AxiomPriorRegularizer(max_prototypes=32, bound_device=device, bound_shape=(num_particles, latent_dim))
    opt_g = torch.optim.Adam(list(gen_m.parameters()) + [prior.z], lr=1e-2)
    opt_d = torch.optim.Adam(crit_m.parameters(), lr=1e-2)
    opt_d.record = SimpleNamespace(  # type: ignore[attr-defined]
        thermal_scale=1.0,
        _disc_gate=1.0,
        ema_push_res=1.0,
        chord_phase=0.0,
        _g_probe_cache="stale",
        observe_margin_and_scale=lambda *a: None,
    )
    recipe = SimpleNamespace(
        prior_regularizer="axiom",
        t1_pullback_weight=1.0,
        t1_iso_weight=0.5,
        t1_laguerre_lr=0.15,
        t1_prior_row_lr=0.05,
        encoder_mode="none",
        conditioning="scalar",
        loss="non_saturating",
        num_particles=num_particles,
        prior_l2=0.0,
        prior_reg=0.0,
        kinetic_transport_weight=0.0,
        kinetic_transport_local_weight=0.0,
        kinetic_transport_loss=lambda f, r: f.new_zeros(()),
        kinetic_transport_local_loss=lambda f, r: f.new_zeros(()),
    )

    def _generate(
        gen: nn.Module,
        z: torch.Tensor,
        sigma_out: float,
        noise: torch.Generator,
        rows=None,
    ) -> torch.Tensor:
        del rows
        out = gen(z)
        if float(sigma_out) > 0.0:
            eps = torch.randn(out.shape, generator=noise, dtype=out.dtype).to(out.device)
            out = out + float(sigma_out) * eps
        return out

    trainer = SimpleNamespace(
        G=gen_m,
        D=crit_m,
        prior=prior,
        prior_regularizer=reg,
        opt_g=opt_g,
        opt_d=opt_d,
        recipe=recipe,
        policy=SimpleNamespace(
            after_generator_backward=lambda **kw: None,
            after_generator_step=lambda: None,
        ),
        _prior_penalty=False,
        _postprocess=lambda p, x: x,
        _generate=_generate,
        _batch=lambda x, nm: x,
        loss=SimpleNamespace(g_loss=lambda fl, rl: (-fl).mean()),
    )
    return trainer, torch.Generator(device="cpu").manual_seed(seed + 1000)


# ---------------------------------------------------------------------------
# 1. Static L2 Hygiene, Acyclic Imports, Style & Dual Complexity Gates
# ---------------------------------------------------------------------------


def test_static_l2_imports_and_forbidden_ops() -> None:
    source = PRIOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_l1 = {"math_ops", "state", "atlas"}
    allowed_public = {"particle_prior", "priors", "optim.constraint_geometry"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 1:
                assert mod in allowed_l1, f"L2 prior.py may only import L0/L1, got .{mod}"
            elif node.level >= 2:
                assert mod in allowed_public, f"Unexpected parent import ..{mod}"
            else:
                assert not mod.startswith("particlegan") and mod != "random"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("particlegan") and alias.name != "random"

    forbidden = {
        "svd",
        "svdvals",
        "eigh",
        "eigvalsh",
        "pinv",
        "compile",
        "randn",
        "rand",
        "randint",
        "randperm",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, f"Forbidden attribute .{node.attr}"
            is_ag = node.attr == "grad" and isinstance(node.value, ast.Attribute) and node.value.attr == "autograd"
            assert not is_ag, "Forbidden torch.autograd.grad in prior.py"
    assert "NormalNoiseManipulator" not in source, "Zero class-name string sniffing allowed"
    assert "_axiom_active_rows" not in source, "Zero monkey-patched prior.z attributes allowed"


def test_static_l2_style_and_complexity() -> None:
    source = PRIOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for lineno, line in enumerate(source.splitlines(), start=1):
        assert len(line) <= 120, f"Line {lineno} exceeds 120 chars ({len(line)})"
        assert chr(59) not in line, f"Semicolon found on line {lineno}"
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    c901 = [_mccabe_c901(fn) for fn in fns]
    bcc = [_radon_bool_cc(fn) for fn in fns]
    assert max(c901) <= 10, f"Max C901 {max(c901)} > 10"
    assert max(bcc) <= 15, f"Max BoolCC {max(bcc)} > 15"
    mean_c901 = sum(c901) / len(c901)
    assert 2.5 <= mean_c901 <= 5.5, f"Mean C901 {mean_c901:.2f} out of target band"


# ---------------------------------------------------------------------------
# 2. 10:1 Rotated Ellipse Whitened Burg Pullback Convergence (cond -> 1.0)
# ---------------------------------------------------------------------------


def test_rotated_10to1_ellipse_burg_pullback_convergence() -> None:
    rad = math.radians(37.0)
    rot = torch.tensor([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    cov_true = rot @ torch.diag(torch.tensor([1.0, 0.01])) @ rot.T  # 10:1 std (100:1 var)
    chol_true = torch.linalg.cholesky(cov_true)

    sigma_z = 0.10
    gen = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        gen.weight.copy_(torch.eye(2) * 3.0)
    opt = torch.optim.Adam(gen.parameters(), lr=0.10)
    mu = torch.zeros((4, 2))
    target_cov = cov_true.unsqueeze(0).expand(4, 2, 2)
    inv_w = torch.ones(4)

    for _ in range(500):
        opt.zero_grad()
        loss, _, _ = whitened_burg_pullback_loss(gen, mu, sigma_z, target_cov, inv_w)
        loss.backward()
        opt.step()

    with torch.no_grad():
        m_learned = sigma_z * gen.weight
        m_tilde = torch.linalg.solve_triangular(chol_true, m_learned, upper=False)
        s_tilde = m_tilde @ m_tilde.T
        eigs = torch.linalg.eigvalsh(s_tilde)
        cond = float((eigs[-1] / eigs[0]).item())
        assert cond < 1.05, f"Expected whitened condition ~1.0, got {cond}"
        assert torch.allclose(eigs, torch.ones(2), atol=0.03)
        assert torch.allclose(m_learned @ m_learned.T, cov_true, atol=0.02)


# ---------------------------------------------------------------------------
# 3. High-D (d = 32) Haar-Rotated Rank-Preserving Burg Pullback
# ---------------------------------------------------------------------------


def test_highd_d32_haar_rotated_rank_preserving_pullback() -> None:
    g = torch.Generator().manual_seed(13)
    dim = 32
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
    eig_true = torch.linspace(0.25, 2.5, dim)
    cov_true = q @ torch.diag(eig_true) @ q.T
    chol_true = torch.linalg.cholesky(cov_true)

    sigma_z = 0.05
    gen = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        # Start near-collapsed in half the coordinates to prove 1/sigma_min barrier rescues full rank!
        init_s = torch.cat([torch.full((16,), 5.0), torch.full((16,), 0.2)])
        gen.weight.copy_(torch.diag(init_s))
    opt = torch.optim.Adam(gen.parameters(), lr=0.12)
    mu = torch.zeros((2, dim))
    target_cov = cov_true.unsqueeze(0).expand(2, dim, dim)
    inv_w = torch.ones(2)

    for _ in range(450):
        opt.zero_grad()
        loss, _, _ = whitened_burg_pullback_loss(gen, mu, sigma_z, target_cov, inv_w)
        loss.backward()
        opt.step()

    with torch.no_grad():
        m_tilde = torch.linalg.solve_triangular(chol_true, sigma_z * gen.weight, upper=False)
        eigs = torch.linalg.eigvalsh(m_tilde @ m_tilde.T)
        assert float(eigs.min().item()) > 0.90, f"Rank collapse detected: min_eig={eigs.min().item()}"
        assert float(eigs.max().item()) < 1.10, f"Overshoot detected: max_eig={eigs.max().item()}"


# ---------------------------------------------------------------------------
# 4. Curvature-Compensated Tanh Variance Fidelity vs Exact 1.0 on Linear
# ---------------------------------------------------------------------------


def test_curvature_compensated_tanh_variance_fidelity() -> None:
    rng = torch.Generator().manual_seed(19)
    sigma_z = 0.55
    mu = torch.zeros((64, 2))
    target_var = 0.16
    target_cov_base = torch.eye(2) * target_var

    # 4A: On a linear generator, realized == tangent -> curvature gamma_C == 1.0000 exactly!
    reg_lin = AxiomPriorRegularizer(max_prototypes=8)
    reg_lin.update_quantizer(torch.randn(64, 2, generator=rng) * math.sqrt(target_var))
    coal_lin = reg_lin.atlas.coalesce()
    lin_gen = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lin_gen.weight.copy_(torch.eye(2) * (math.sqrt(target_var) / sigma_z))
    eps = torch.randn(64, 2, generator=rng)
    _, tang_lin, _ = whitened_burg_pullback_loss(
        lin_gen,
        mu,
        sigma_z,
        target_cov_base.unsqueeze(0).expand(64, 2, 2),
        torch.ones(64),
        eps_g=eps,
    )
    real_lin = (lin_gen(mu + sigma_z * eps) - lin_gen(mu)).detach().square().sum(dim=-1) / 2.0
    reg_lin.update_curvature_ema(torch.zeros(64, dtype=torch.long), real_lin, tang_lin)
    assert torch.allclose(
        reg_lin.curvature_gamma_per_component(coal_lin),
        torch.ones_like(coal_lin.comp_var),
        atol=1e-5,
    )

    # 4B: On a concave tanh generator, realized variance contracts vs tangent -> gamma_C > 1.05
    # and training with gamma_C * target_cov sharply reduces realized variance bias!
    def _train_tanh(use_curv: bool) -> float:
        gen_t = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Tanh())
        with torch.no_grad():
            gen_t[0].weight.copy_(torch.eye(2))
        opt = torch.optim.Adam(gen_t.parameters(), lr=0.05)
        reg = AxiomPriorRegularizer(max_prototypes=4)
        reg.atlas.initialized, reg.atlas.dim = True, 2
        reg.atlas.proto_c = torch.zeros((1, 2))
        reg.atlas.proto_cov = target_cov_base.unsqueeze(0).clone()
        reg.atlas.proto_w = torch.ones(1)
        reg.atlas.w_fake_ema = torch.ones(1)
        reg.atlas.proto_n = torch.full((1,), 50.0)
        reg.atlas.psi = torch.zeros(1)
        reg.atlas.v_push = torch.full((1,), target_var)
        g_loc = torch.Generator().manual_seed(99)
        for _ in range(120):
            coal = reg.atlas.coalesce()
            gamma = reg.curvature_gamma_per_component(coal)[0] if use_curv else torch.tensor(1.0)
            eps_b = torch.randn(64, 2, generator=g_loc)
            opt.zero_grad()
            loss, tang_v, _ = whitened_burg_pullback_loss(
                gen_t,
                mu,
                sigma_z,
                (gamma * target_cov_base).unsqueeze(0).expand(64, 2, 2),
                torch.ones(64),
                eps_g=eps_b,
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                real_v = (gen_t(mu + sigma_z * eps_b) - gen_t(mu)).square().sum(dim=-1) / 2.0
                reg.update_curvature_ema(torch.zeros(64, dtype=torch.long), real_v, tang_v)
        with torch.no_grad():
            ev_eps = torch.randn(4096, 2, generator=torch.Generator().manual_seed(7))
            ev_out = gen_t(sigma_z * ev_eps)
            realized_var = float(ev_out.var(dim=0, unbiased=False).mean().item())
            return (realized_var - target_var) / target_var

    bias_uncomp = _train_tanh(use_curv=False)
    bias_comp = _train_tanh(use_curv=True)
    assert bias_uncomp < -0.08, f"Expected concave tanh contraction bias, got {bias_uncomp}"
    assert abs(bias_comp) < abs(bias_uncomp) * 0.5, (
        f"Curvature compensation did not reduce tanh bias: uncomp={bias_uncomp:.3f}, comp={bias_comp:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. Wide-Latent (d_z > max(D, 16)) 2-Sketch U-Statistic Unbiasedness
# ---------------------------------------------------------------------------


def test_wide_latent_2sketch_ustat_unbiasedness() -> None:
    g = torch.Generator().manual_seed(29)
    dim, d_z = 4, 32  # d_z > max(4, 16) -> 2-sketch U-statistic path!
    sigma_z = 0.25
    target_cov = torch.eye(dim)
    gen = nn.Linear(d_z, dim, bias=False)
    with torch.no_grad():
        gen.weight.copy_(torch.randn(dim, d_z, generator=g) * 0.7)

    # Exact objective 2.25 * ||S_tilde - I||_F^2 / D and its exact gradient.
    m_exact = sigma_z * gen.weight
    s_exact = m_exact @ m_exact.T
    exact_loss = 2.25 * (s_exact - target_cov).square().sum() / float(dim)
    gen.zero_grad()
    exact_loss.backward()
    exact_grad = gen.weight.grad.clone()

    # Average 2-sketch U-statistic over 128 deterministic Weyl phases (zero RNG!).
    mu = torch.zeros((8, d_z))
    tc_b = target_cov.unsqueeze(0).expand(8, dim, dim)
    inv_w = torch.ones(8)
    loss_acc = 0.0
    grad_acc = torch.zeros_like(exact_grad)
    n_phases = 128
    for i in range(n_phases):
        gen.zero_grad()
        l_hat, _, _ = whitened_burg_pullback_loss(gen, mu, sigma_z, tc_b, inv_w, phase=(i + 0.5) / float(n_phases))
        l_hat.backward()
        loss_acc += float(l_hat.detach().item())
        grad_acc += gen.weight.grad.detach()
    mean_loss = loss_acc / float(n_phases)
    mean_grad = grad_acc / float(n_phases)
    cos_sim = float((mean_grad * exact_grad).sum() / (mean_grad.norm() * exact_grad.norm()).clamp_min(1e-12))
    assert math.isclose(mean_loss, float(exact_loss.item()), rel_tol=0.08)
    assert cos_sim > 0.99, f"2-sketch U-stat gradient misaligned: cos={cos_sim}"

    # At exact equilibrium S_tilde == I_D, E[loss] ~ 0 (zero Jensen inflation!).
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(d_z, dim, generator=g))
        gen.weight.copy_(q.T / sigma_z)
    eq_vals = [
        float(whitened_burg_pullback_loss(gen, mu, sigma_z, tc_b, inv_w, phase=(i + 0.5) / 64.0)[0].item())
        for i in range(64)
    ]
    assert abs(sum(eq_vals) / len(eq_vals)) < 0.05


# ---------------------------------------------------------------------------
# 6. Rank-Free 4D 8x8 Pullback Eligibility & Parity with 2D
# ---------------------------------------------------------------------------


def test_rank_free_4d_8x8_pullback_parity() -> None:
    tr_4d, noise_4d = _make_mock_trainer(out_shape=(1, 8, 8), latent_dim=4, num_particles=32, seed=5)
    tr_2d, noise_2d = _make_mock_trainer(out_shape=(64,), latent_dim=4, num_particles=32, seed=5)

    g = torch.Generator().manual_seed(55)
    real_4d = torch.randn(16, 1, 8, 8, generator=g)
    real_2d = real_4d.reshape(16, 64).clone()
    real_big = torch.randn(4, 3, 32, 32, generator=g)

    assert should_use_axiom_pullback(tr_4d, real_4d) is True
    assert should_use_axiom_pullback(tr_2d, real_2d) is True
    assert should_use_axiom_pullback(tr_4d, real_big) is False

    # ManipulatedPrior with NormalNoiseManipulator detected without class-name sniffing.
    p_base = ParticlePrior(16, 4)
    p_manip = ManipulatedPrior(p_base, (NormalNoiseManipulator(sigma=0.1),))
    assert has_positive_prior_sigma(p_manip) is True
    assert has_positive_prior_sigma(p_base) is False

    # Bit-exact parity between 4D [B, 1, 8, 8] and 2D [B, 64] pullback steps!
    z_samp, idx = tr_4d.prior.sample(16, generator=torch.Generator().manual_seed(77))
    out_4d = axiom_pullback_step(
        tr_4d,
        real_4d,
        real_4d,
        z_samp,
        idx,
        sigma_out=0.0,
        noise=noise_4d,
        critic=tr_4d.D,
        loss_d=torch.tensor(1.0),
        penalty=torch.tensor(0.1),
    )
    out_2d = axiom_pullback_step(
        tr_2d,
        real_2d,
        real_2d,
        z_samp.clone(),
        idx.clone(),
        sigma_out=0.0,
        noise=noise_2d,
        critic=tr_2d.D,
        loss_d=torch.tensor(1.0),
        penalty=torch.tensor(0.1),
    )
    assert torch.allclose(out_4d[0], out_2d[0], atol=1e-6)
    assert torch.allclose(tr_4d.prior.z, tr_2d.prior.z, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Clock-Free Boolean row_init Stationary EMA & Zero Global RNG Mutation
# ---------------------------------------------------------------------------


def test_clock_free_row_init_ema_and_zero_rng() -> None:
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()

    # Scale homogeneity across 1e-6..1e+6 and exact zero step-count dependence after first touch!
    deltas = []
    for scale in (1e-6, 1.0, 1e6):
        reg = AxiomPriorRegularizer(max_prototypes=8)
        z = torch.zeros((8, 4))
        s_idx = torch.tensor([0, 2, 5])
        is_f = torch.tensor([True, True, True])
        g1 = torch.tensor([[1.0, -2.0, 0.5, 0.2], [-0.3, 1.5, -0.8, 0.4], [0.2, 0.1, 1.0, -1.0]]) * scale
        ps = torch.ones((3, 1))
        reg.apply_prior_row_update(z, s_idx, is_f, g1, ps, 0.1, torch.tensor(1.0))
        assert reg.row_init is not None and bool(reg.row_init[s_idx].all())
        assert not bool(reg.row_init[1].item())
        z_after1 = z.clone()
        g2 = g1 * 0.5 + torch.tensor([[0.1, 0.2, -0.3, 0.4], [0.2, -0.1, 0.3, -0.2], [-0.1, 0.3, 0.1, 0.2]]) * scale
        reg.apply_prior_row_update(z, s_idx, is_f, g2, ps, 0.1, torch.tensor(1.0))
        deltas.append((z_after1, z - z_after1))

    for d1, d2 in deltas[1:]:
        assert torch.allclose(d1, deltas[0][0], atol=1e-5)
        assert torch.allclose(d2, deltas[0][1], atol=1e-5)

    # Zero step-count dependence: two regularizers with identical (row_m, row_v, row_init=True)
    # produce bit-identical updates even though one had 100 prior updates and the other had 1!
    reg_a = AxiomPriorRegularizer(max_prototypes=8)
    reg_b = AxiomPriorRegularizer(max_prototypes=8)
    za, zb = torch.zeros((4, 3)), torch.zeros((4, 3))
    idx_all = torch.arange(4)
    all_t = torch.ones(4, dtype=torch.bool)
    g_const = torch.ones((4, 3))
    for _ in range(100):
        reg_a.apply_prior_row_update(za, idx_all, all_t, g_const, torch.ones((4, 1)), 0.01, torch.tensor(1.0))
    reg_b.apply_prior_row_update(zb, idx_all, all_t, g_const, torch.ones((4, 1)), 0.01, torch.tensor(1.0))
    za.zero_()
    zb.zero_()
    g_probe = torch.tensor([[2.0, -1.0, 0.5], [0.1, 1.0, -0.2], [-1.0, 0.3, 0.7], [0.4, -0.8, 1.2]])
    reg_a.apply_prior_row_update(za, idx_all, all_t, g_probe, torch.ones((4, 1)), 0.05, torch.tensor(1.0))
    reg_b.apply_prior_row_update(zb, idx_all, all_t, g_probe, torch.ones((4, 1)), 0.05, torch.tensor(1.0))
    assert torch.allclose(za, zb, atol=1e-6)
    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)

    # Full forward + central_pullback_cov + pullback step must consume 0 global RNG state.
    tr, noise = _make_mock_trainer(out_shape=(2,), latent_dim=2, num_particles=24, seed=3)
    real = torch.randn(16, 2, generator=torch.Generator().manual_seed(4))
    z_s, idx_s = tr.prior.sample(16, generator=torch.Generator().manual_seed(5))
    py_step = random.getstate()
    np_step = np.random.get_state()
    torch_step = torch.get_rng_state()
    _ = tr.prior_regularizer(tr.prior.z)
    _ = central_pullback_cov(tr.G, tr.prior.z[:8], 0.1, return_pair=True)
    _ = axiom_pullback_step(
        tr,
        real,
        real,
        z_s,
        idx_s,
        sigma_out=0.0,
        noise=noise,
        critic=tr.D,
        loss_d=torch.tensor(1.0),
        penalty=torch.tensor(0.1),
    )
    assert random.getstate() == py_step
    assert np.array_equal(np.random.get_state()[1], np_step[1])
    assert torch.equal(torch.get_rng_state(), torch_step)


# ---------------------------------------------------------------------------
# 8. Suite 5A/5B/5C Compatibility (Mass Conservation, LTV Fixture, Quotas)
# ---------------------------------------------------------------------------


def test_suite5_compat_mass_ltv_gauge_and_quotas() -> None:
    # Suite 5B in-place reg.proto_var[0] = 20.0 fixture + exact LTV merge/split!
    k_m = 16
    centers = torch.zeros((k_m, 2))
    for i in range(k_m):
        centers[i, 0] = float(i) * 10.0
    centers[1], centers[2] = torch.tensor([10.0, 0.0]), torch.tensor([11.0, 0.0])

    reg = AxiomPriorRegularizer(max_prototypes=k_m)
    reg.quantizer_initialized = True
    reg.proto_c = centers.clone()
    reg.proto_var = torch.full((k_m,), 0.25)
    reg.proto_var[0], reg.proto_var[1], reg.proto_var[2] = 20.0, 0.10, 0.40
    reg.proto_w = torch.full((k_m,), 1.0 / float(k_m))
    reg.w_fake_ema = reg.proto_w.clone()
    reg.psi = torch.zeros(k_m)
    reg.v_push = reg.proto_var.clone()
    reg.update_quantizer(centers[3:].repeat(4, 1))
    assert math.isclose(float(reg.proto_var[1].item()), 0.375, rel_tol=1e-4)
    split_tot = 0.5 * (float(reg.proto_var[0].item()) + float(reg.proto_var[2].item())) + (
        0.25 * float((reg.proto_c[0] - reg.proto_c[2]).square().sum().item()) / 2.0
    )
    assert math.isclose(split_tot, 20.0, rel_tol=1e-4)

    # Suite 5B 10x width guard + post-coalesced psi zero-mean gauge.
    pc_10x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]])
    _, _, _, coal_ax = coalesce_pythagorean_ltv(
        pc_10x, torch.tensor([0.01, 1.0, 0.01]), torch.tensor([0.25, 0.25, 0.50])
    )
    assert not bool(coal_ax[0].item())

    reg_psi = AxiomPriorRegularizer(max_prototypes=8)
    reg_psi.quantizer_initialized = True
    reg_psi.proto_c = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
    reg_psi.proto_var = torch.full((3,), 0.05)
    reg_psi.proto_w = torch.tensor([0.4, 0.1, 0.5])
    reg_psi.w_fake_ema = torch.tensor([0.6, 0.3, 0.1])
    reg_psi.psi = torch.tensor([2.0, -2.0, 0.0])
    reg_psi.v_push = reg_psi.proto_var.clone()
    w_u = torch.tensor([[0.8, 0.2, 0.0], [0.8, 0.2, 0.0], [0.0, 0.0, 1.0]])
    reg_psi.update_laguerre_dual(reg_psi.proto_c, w_u @ reg_psi.proto_c, w_u, torch.tensor(1.0), 0.015)
    assert abs(float(reg_psi.psi.mean().item())) < 1e-6

    # Suite 5C re-exported allocate_deficit_quotas.
    q = allocate_deficit_quotas(
        torch.full((100,), 0.01),
        torch.cat([torch.full((64,), 1.0 / 64.0), torch.zeros(36)]),
        64,
    )
    assert int(q[64:].sum().item()) == 36


# ---------------------------------------------------------------------------
# 9. Schema-2 Checkpoint Roundtrip, Deepcopy, Continuation & Tamper Rejection
# ---------------------------------------------------------------------------


def test_schema2_checkpoint_roundtrip_continuation_and_tamper() -> None:
    tr1, noise1 = _make_mock_trainer(out_shape=(2,), latent_dim=2, num_particles=32, seed=11)
    g_data = torch.Generator().manual_seed(12)
    c4 = torch.tensor([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]])

    for step in range(4):
        real = c4[torch.randint(0, 4, (32,), generator=g_data)] + 0.08 * torch.randn(32, 2, generator=g_data)
        z_s, idx_s = tr1.prior.sample(32, generator=torch.Generator().manual_seed(100 + step))
        axiom_pullback_step(
            tr1,
            real,
            real,
            z_s,
            idx_s,
            sigma_out=0.0,
            noise=noise1,
            critic=tr1.D,
            loss_d=torch.tensor(1.0),
            penalty=torch.tensor(0.1),
        )

    sd = tr1.prior_regularizer.state_dict()
    assert sd["schema"] == SCHEMA_VERSION
    # Must support deepcopy(reg).load_state_dict(sd) as called by recipes.py!
    deepcopy(tr1.prior_regularizer).load_state_dict(sd)

    # Restore into a fresh trainer with cloned weights/optimizers for bit-exact continuation.
    tr2, noise2 = _make_mock_trainer(out_shape=(2,), latent_dim=2, num_particles=32, seed=11)
    tr2.G.load_state_dict(deepcopy(tr1.G.state_dict()))
    tr2.D.load_state_dict(deepcopy(tr1.D.state_dict()))
    with torch.no_grad():
        tr2.prior.z.copy_(tr1.prior.z)
    tr2.opt_g.load_state_dict(deepcopy(tr1.opt_g.state_dict()))
    tr2.prior_regularizer.load_state_dict(sd)

    real_next = c4[torch.randint(0, 4, (32,), generator=g_data)] + 0.08 * torch.randn(32, 2, generator=g_data)
    z_next, idx_next = tr1.prior.sample(32, generator=torch.Generator().manual_seed(200))
    out1 = axiom_pullback_step(
        tr1,
        real_next,
        real_next,
        z_next,
        idx_next,
        sigma_out=0.0,
        noise=noise1,
        critic=tr1.D,
        loss_d=torch.tensor(1.0),
        penalty=torch.tensor(0.1),
    )
    out2 = axiom_pullback_step(
        tr2,
        real_next.clone(),
        real_next.clone(),
        z_next.clone(),
        idx_next.clone(),
        sigma_out=0.0,
        noise=noise2,
        critic=tr2.D,
        loss_d=torch.tensor(1.0),
        penalty=torch.tensor(0.1),
    )
    assert torch.equal(out1[0].detach(), out2[0].detach())
    assert torch.equal(tr1.prior.z.detach(), tr2.prior.z.detach())
    assert torch.equal(tr1.prior_regularizer.row_m, tr2.prior_regularizer.row_m)
    assert torch.equal(tr1.prior_regularizer.row_init, tr2.prior_regularizer.row_init)

    # Fail-closed tamper rejection (incl. schema-1, bool-as-int, non-bool row_init, missing/extra keys).
    def _mut(**ov: Any) -> Dict[str, Any]:
        c = deepcopy(sd)
        c.update(ov)
        return c

    bad_list: List[Dict[str, Any]] = [
        _mut(schema=1),
        _mut(schema=True),
        {k: v for k, v in sd.items() if k != "row_init"},
        {**sd, "extra": 1},
        _mut(res_ema=-0.1),
        _mut(iso_ema=float("nan")),
        _mut(row_init=sd["row_init"].float()),
        _mut(v_real_ema=-sd["v_real_ema"]),
        _mut(atlas={**sd["atlas"], "schema": 1}),
    ]
    for bad in bad_list:
        with pytest.raises(ValueError):
            validate_axiom_prior_state(bad)
    with pytest.raises(ValueError):
        AxiomPriorRegularizer(max_prototypes=16).load_state_dict(sd)


# ---------------------------------------------------------------------------
# 10. CUDA Pullback Execution (2D + 4D 8x8) & Checkpoint Roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_cuda_pullback_step_and_checkpoint() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device("cuda:0")
    for shape in ((2,), (1, 8, 8)):
        tr, noise = _make_mock_trainer(out_shape=shape, latent_dim=4, num_particles=32, device=dev, seed=7)
        g = torch.Generator(device=dev).manual_seed(0)
        real = torch.randn(16, *shape, device=dev, generator=g)
        z_s, idx_s = tr.prior.sample(16, generator=g)
        assert should_use_axiom_pullback(tr, real) is True
        out = axiom_pullback_step(
            tr,
            real,
            real,
            z_s,
            idx_s,
            sigma_out=0.0,
            noise=noise,
            critic=tr.D,
            loss_d=torch.tensor(1.0, device=dev),
            penalty=torch.tensor(0.1, device=dev),
        )
        assert bool(torch.isfinite(out[0]).all())
        assert tr.prior_regularizer.row_m is not None and tr.prior_regularizer.row_m.device.type == "cuda"
        sd = tr.prior_regularizer.state_dict()
        restored = AxiomPriorRegularizer(max_prototypes=32, bound_device=dev, bound_shape=(32, 4))
        restored.load_state_dict(sd)
        assert restored.row_m is not None and restored.row_m.device.type == "cuda"


def test_burg_born_infeld_compression_bounds_singular_manifold_gradient() -> None:
    lin = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.diag(torch.tensor([100.0, 1.0])))
    mu = torch.randn(16, 2)
    cov = torch.eye(2).unsqueeze(0).expand(16, 2, 2).clone()
    inv_w = torch.ones(16)
    loss, _, _ = whitened_burg_pullback_loss(lin, mu, 1.0, cov, inv_w)
    # Without BI compression loss ~ 4.5 * 5000 = 22500; BI compression caps growth to O(sqrt(v)) << 2000.
    assert torch.isfinite(loss) and float(loss.item()) < 1500.0
    loss.backward()
    assert lin.weight.grad is not None and bool(torch.isfinite(lin.weight.grad).all())


def test_pullback_tight_multimode_grad_stays_scale_consistent() -> None:
    """Global ``sigma_med / sigma_out`` keeps ``loss_maha`` scale-consistent when ``sigma_med << sigma_out``."""
    grid = torch.stack(torch.meshgrid(torch.linspace(-1.8, 1.8, 4), torch.linspace(-1.8, 1.8, 4), indexing="ij"), -1)
    centers = grid.reshape(-1, 2)

    def _run_sigma(sigma: float) -> float:
        tr, noise = _make_mock_trainer(out_shape=(2,), latent_dim=2, num_particles=64, seed=19)
        tr.recipe.t1_iso_weight = 0.0
        g = torch.Generator().manual_seed(3)
        loss_g = torch.tensor(0.0)
        for _ in range(12):
            idx_c = torch.randint(0, len(centers), (64,), generator=g)
            real = centers[idx_c] + sigma * torch.randn(64, 2, generator=g)
            z_s, idx_s = tr.prior.sample(64, generator=g)
            loss_g, _, _, _, _, _ = axiom_pullback_step(
                tr,
                real,
                real,
                z_s,
                idx_s,
                sigma_out=0.0,
                noise=noise,
                critic=tr.D,
                loss_d=torch.tensor(1.0),
                penalty=torch.tensor(0.05),
            )
            assert bool(torch.isfinite(loss_g).all())
        return float(loss_g.detach().item())

    loss_wide = _run_sigma(0.15)
    loss_tight = _run_sigma(0.03)
    # Without global (sigma_med / sigma_out) normalization, 5x tighter modes inflate loss_maha by ~5x.
    assert loss_tight < 1.8 * max(loss_wide, 0.5)
    assert loss_tight < 6.0


@pytest.mark.cuda
def test_cuda_deterministic_pullback_step() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        dev = torch.device("cuda:0")
        tr, noise = _make_mock_trainer(out_shape=(2,), latent_dim=2, num_particles=32, device=dev, seed=11)
        g = torch.Generator(device=dev).manual_seed(0)
        modes = torch.tensor([[-3.0, -3.0], [3.0, 3.0]], device=dev)
        for _ in range(4):
            real = modes[torch.randint(0, 2, (16,), device=dev, generator=g)] + 0.1 * torch.randn(
                16, 2, device=dev, generator=g
            )
            z_s, idx_s = tr.prior.sample(16, generator=g)
            out = axiom_pullback_step(
                tr,
                real,
                real,
                z_s,
                idx_s,
                sigma_out=0.0,
                noise=noise,
                critic=tr.D,
                loss_d=torch.tensor(1.0, device=dev),
                penalty=torch.tensor(0.1, device=dev),
            )
            assert bool(torch.isfinite(out[0]).all())
        torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(prev)
