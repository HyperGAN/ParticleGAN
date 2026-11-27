"""Unit, mathematical-invariant, equivariance, and checkpoint tests for ``particlegan._axiom.fiber``."""

from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Tuple

import pytest
import torch

from particlegan._axiom import fiber as Fb
from particlegan._axiom.fiber import (
    SCHEMA_VERSION,
    AxiomCondFiberBank,
    conditional_fiber_loss,
    validate_fiber_bank_state,
)

_FIBER_PATH = Path(__file__).resolve().parents[1] / "particlegan" / "_axiom" / "fiber.py"


def _random_orthogonal(dim: int, gen: torch.Generator) -> torch.Tensor:
    mat = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(mat)
    signs = torch.where(torch.diagonal(r) >= 0.0, 1.0, -1.0)
    return q * signs.unsqueeze(0)


def _stub_record(
    *,
    calls: int = 0,
    observed_steps: int = 0,
    cond_fiber_bank: dict | None = None,
    cond_ema: float = 0.25,
    cond_initialized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        calls=calls,
        observed_steps=observed_steps,
        cond_fiber_bank=cond_fiber_bank,
        _cond_bank_obj=None,
        cond_ema=cond_ema,
        cond_initialized=cond_initialized,
        step_scale=1.0,
        _in_probe=False,
        _in_protected_probe=False,
    )


def _sample_bimodal_fiber(
    n: int, dim: int, gen: torch.Generator, *, sep: float = 0.85, sigma: float = 0.18
) -> Tuple[torch.Tensor, torch.Tensor]:
    c = torch.randn(n, dim, generator=gen, dtype=torch.float32)
    probs = torch.sigmoid(1.5 * c[:, 0])
    signs = torch.where(torch.rand(n, generator=gen) < probs, 1.0, -1.0).unsqueeze(1)
    axis = torch.zeros(1, dim, dtype=torch.float32)
    axis[0, 0] = sep
    y = 0.45 * c + signs * axis + sigma * torch.randn(n, dim, generator=gen)
    return c, y


def test_static_hygiene_and_acyclic_imports() -> None:
    """Enforce L1 acyclic imports, zero forbidden ops, zero semicolons, and clock-freedom."""
    src = _FIBER_PATH.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    assert all(";" not in line for line in lines), "semicolon-packed line found"
    assert max(len(line) for line in lines) <= 120

    internal_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            internal_imports.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.module and "particlegan" in node.module:
            internal_imports.append(node.module)
    assert internal_imports == ["math_ops"], f"unexpected internal imports: {internal_imports}"

    for forbidden in (
        "autograd.grad",
        "linalg.svd",
        "linalg.eigh",
        "linalg.eigvalsh",
        "linalg.pinv",
        "torch.compile",
    ):
        assert forbidden not in src, f"forbidden op {forbidden!r} found in fiber.py"
    for clock_tok in ("observed_steps", "total_steps", "completed_steps"):
        assert clock_tok not in src, f"clock token {clock_tok!r} found in fiber.py"


@pytest.mark.parametrize("dim", [1, 2, 4, 8, 16])
def test_conditional_gaussian_and_bimodal_recovery_gradients(dim: int) -> None:
    """Verify corrective mean, variance, and bimodal-mode gradients across d=1..16."""
    g = torch.Generator().manual_seed(100 + dim)
    a_mat = _random_orthogonal(dim, g).float() * 0.75
    bank = AxiomCondFiberBank()
    batch = max(32, 2 * AxiomCondFiberBank.basis_dim(dim))
    disc = torch.ones(batch, 1, dtype=torch.long)

    # Warm up on mismatched slope & over-dispersed Gaussian fiber
    for _ in range(4):
        c = torch.randn(batch, dim, generator=g)
        y = c @ a_mat + 0.25 * torch.randn(batch, dim, generator=g)
        x = (0.2 * (c @ a_mat) + 0.60 * torch.randn(batch, dim, generator=g)).requires_grad_(True)
        loss = conditional_fiber_loss(x, y, c, cond_disc=disc, bank=bank)
        loss.backward()
        assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
    assert (1,) in bank.strata

    # Probe slope=0.2 -> 1.0 and over-dispersed scale=0.60 -> 0.25
    c_p = torch.randn(batch, dim, generator=g)
    eps_x = torch.randn(batch, dim, generator=g)
    y_p = c_p @ a_mat + 0.25 * torch.randn(batch, dim, generator=g)
    slope = torch.tensor(0.2, requires_grad=True)
    scale = torch.tensor(0.60, requires_grad=True)
    x_p = slope * (c_p @ a_mat) + scale * eps_x
    loss_p = conditional_fiber_loss(x_p, y_p, c_p, cond_disc=disc, bank=bank, allow_state_update=False)
    loss_p.backward()
    assert slope.grad is not None and float(slope.grad.item()) < -0.02
    assert scale.grad is not None and float(scale.grad.item()) > 0.02

    # Under-dispersed variance (scale=0.05 -> 0.25) pushes variance outward
    bank_u = AxiomCondFiberBank()
    for _ in range(4):
        c = torch.randn(batch, dim, generator=g)
        y = c @ a_mat + 0.30 * torch.randn(batch, dim, generator=g)
        x = (c @ a_mat + 0.05 * torch.randn(batch, dim, generator=g)).requires_grad_(True)
        conditional_fiber_loss(x, y, c, cond_disc=disc, bank=bank_u).backward()
    c_u = torch.randn(batch, dim, generator=g)
    y_u = c_u @ a_mat + 0.30 * torch.randn(batch, dim, generator=g)
    scale_u = torch.tensor(0.05, requires_grad=True)
    x_u = c_u @ a_mat + scale_u * torch.randn(batch, dim, generator=g)
    conditional_fiber_loss(x_u, y_u, c_u, cond_disc=disc, bank=bank_u, allow_state_update=False).backward()
    assert scale_u.grad is not None and float(scale_u.grad.item()) < -0.01

    # Conditional bimodal fiber: collapsed unimodal generator pushed toward bimodal separation
    bank_bi = AxiomCondFiberBank()
    for _ in range(4):
        c_b, y_b = _sample_bimodal_fiber(batch, dim, g)
        x_b = (0.45 * c_b + 0.15 * torch.randn(batch, dim, generator=g)).requires_grad_(True)
        conditional_fiber_loss(x_b, y_b, c_b, bank=bank_bi).backward()
    c_bp, y_bp = _sample_bimodal_fiber(batch, dim, g)
    sep = torch.tensor(0.10, requires_grad=True)
    signs_p = torch.where(torch.rand(batch, generator=g) < torch.sigmoid(1.5 * c_bp[:, 0]), 1.0, -1.0)
    axis = torch.zeros(1, dim)
    axis[0, 0] = 1.0
    x_bp = 0.45 * c_bp + (sep * signs_p.unsqueeze(1)) * axis + 0.18 * torch.randn(batch, dim, generator=g)
    conditional_fiber_loss(x_bp, y_bp, c_bp, bank=bank_bi, allow_state_update=False).backward()
    assert sep.grad is not None and float(sep.grad.item()) < -0.01


def test_asymmetric_and_high_dim_context_no_dc32_cliff() -> None:
    """Bank works smoothly for asymmetric d_c != d_x and d_c > 32 (no legacy dc<=6/dc<=32 cliffs)."""
    g = torch.Generator().manual_seed(51)
    for d_c, d_x in ((3, 5), (7, 3), (40, 4)):
        bank = AxiomCondFiberBank()
        batch = max(32, AxiomCondFiberBank.basis_dim(d_c) + 8)
        proj = torch.randn(d_c, d_x, generator=g) * (0.6 / math.sqrt(d_c))
        for _ in range(2):
            c = torch.randn(batch, d_c, generator=g)
            y = c @ proj + 0.2 * torch.randn(batch, d_x, generator=g)
            x = (0.4 * (c @ proj) + 0.3 * torch.randn(batch, d_x, generator=g)).requires_grad_(True)
            lv = conditional_fiber_loss(x, y, c, bank=bank)
            lv.backward()
            assert torch.isfinite(lv) and x.grad is not None and torch.isfinite(x.grad).all()
        q = bank.query_stratum((0,), c[:16], x.detach()[:16], y[:16])
        assert q is not None, f"bank failed to activate for d_c={d_c}, d_x={d_x}"
        assert q["m_pool"].shape == (16, d_x) and q["w_mat"].shape == (d_x, d_x)


def test_exact_zero_loss_equilibrium_and_swap_invariance() -> None:
    """Verify exact 0 loss/grad at matched noiseless fiber, unbiasedness at X~Y|c, and X<->Y symmetry."""
    g = torch.Generator().manual_seed(202)
    # 1. Exact deterministic equilibrium (X == Y = A c): loss == 0 and grad == 0 identically (cold & warm)
    bank_det = AxiomCondFiberBank()
    a = torch.tensor([[0.8, -0.3], [0.4, 0.7]], dtype=torch.float64)
    for _ in range(4):
        c = torch.randn(32, 2, generator=g, dtype=torch.float64)
        y = c @ a
        x = y.clone().requires_grad_(True)
        loss = conditional_fiber_loss(x, y, c, bank=bank_det)
        loss.backward()
        assert abs(float(loss.item())) < 1e-12
        assert float(x.grad.abs().max().item()) < 1e-11

    # 2. Exact X <-> Y swap invariance of bank pooled drift, whitener, z_innov & anti-symmetry of dcov
    bank_xy, bank_yx = AxiomCondFiberBank(), AxiomCondFiberBank()
    for _ in range(6):
        c = torch.randn(24, 3, generator=g, dtype=torch.float64)
        x = 0.8 * c + 0.2 + 0.35 * torch.randn(24, 3, generator=g, dtype=torch.float64)
        y = 0.5 * c - 0.1 + 0.22 * torch.randn(24, 3, generator=g, dtype=torch.float64)
        bank_xy.update_stratum((1,), c, x, y)
        bank_yx.update_stratum((1,), c, y, x)
    cq = torch.randn(16, 3, generator=g, dtype=torch.float64)
    xq = 0.8 * cq + 0.35 * torch.randn(16, 3, generator=g, dtype=torch.float64)
    yq = 0.5 * cq + 0.22 * torch.randn(16, 3, generator=g, dtype=torch.float64)
    q_xy = bank_xy.query_stratum((1,), cq, xq, yq)
    q_yx = bank_yx.query_stratum((1,), cq, yq, xq)
    assert q_xy is not None and q_yx is not None
    assert torch.allclose(q_xy["m_pool"], q_yx["m_pool"], atol=1e-11)
    assert torch.allclose(q_xy["w_mat"], q_yx["w_mat"], atol=1e-11)
    assert torch.allclose(q_xy["delta_m_bank"], -q_yx["delta_m_bank"], atol=1e-11)
    assert torch.allclose(q_xy["delta_m_loo"], -q_yx["delta_m_loo"], atol=1e-11)
    assert torch.allclose(q_xy["delta_cov_loc"], -q_yx["delta_cov_loc"], atol=1e-11)
    assert abs(q_xy["z_innov"] - q_yx["z_innov"]) < 1e-11

    # 3. Statistical zero-loss & zero-gradient equilibrium when X ~ Y | c (no variance collapse!)
    rec = _stub_record()
    g_eq = torch.Generator().manual_seed(303)
    losses, g_scales, g_shifts = [], [], []
    for step in range(80):
        c_b = torch.randn(32, 2, generator=g_eq)
        t_b = torch.ones(32, 1, dtype=torch.long)
        eps_x = 0.25 * torch.randn(32, 2, generator=g_eq)
        eps_y = 0.25 * torch.randn(32, 2, generator=g_eq)
        alpha = torch.tensor(1.0, requires_grad=True)
        shift = torch.zeros(2, requires_grad=True)
        x_b = 0.7 * c_b + shift + alpha * eps_x
        y_b = 0.7 * c_b + eps_y
        lv = conditional_fiber_loss(x_b, y_b, c_b, cond_disc=t_b, record=rec)
        lv.backward()
        if step >= 20:
            losses.append(float(lv.item()))
            g_scales.append(float(alpha.grad.item()))
            g_shifts.append(float(shift.grad.mean().item()))
    for label, vals in (("loss", losses), ("g_shift", g_shifts)):
        mean_v = sum(vals) / len(vals)
        se_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals)) / math.sqrt(len(vals))
        assert abs(mean_v) < 3.5 * max(se_v, 2e-3), f"equilibrium {label} biased: mean={mean_v:.5f}, se={se_v:.5f}"
    # Bounded Lyapunov safety margin at equilibrium: no variance collapse (restores outward by alpha=0.90)
    assert abs(sum(g_scales) / len(g_scales)) < 0.04
    rec_lo = _stub_record()
    g_lo = torch.Generator().manual_seed(303)
    g_scales_lo = []
    for step in range(80):
        c_b = torch.randn(32, 2, generator=g_lo)
        t_b = torch.ones(32, 1, dtype=torch.long)
        alpha_lo = torch.tensor(0.90, requires_grad=True)
        x_b = 0.7 * c_b + alpha_lo * 0.25 * torch.randn(32, 2, generator=g_lo)
        y_b = 0.7 * c_b + 0.25 * torch.randn(32, 2, generator=g_lo)
        conditional_fiber_loss(x_b, y_b, c_b, cond_disc=t_b, record=rec_lo).backward()
        if step >= 20:
            g_scales_lo.append(float(alpha_lo.grad.item()))
    assert sum(g_scales_lo) / len(g_scales_lo) < -0.01


def test_tail_leverage_sensitivity_on_outer_5pct_context_bins() -> None:
    """Verify omega(c) tail weighting boosts corrective gradients on outer-5% context bins."""
    g = torch.Generator().manual_seed(404)
    bank_tail = AxiomCondFiberBank(tail_leverage_gain=0.90)
    bank_flat = AxiomCondFiberBank(tail_leverage_gain=0.0)
    assert bank_tail.gamma_fast == pytest.approx(0.88)

    # Create conditional target y = 0.8*c + noise; generator has outer-5% tail-only mean distortion
    r_cut = 2.45  # ~95th percentile of 2D chi radius (chi^2_2(0.95) = 5.99 -> r = 2.448)
    for _ in range(10):
        c = torch.randn(64, 2, generator=g)
        y = 0.8 * c + 0.20 * torch.randn(64, 2, generator=g)
        tail_mask = (c.norm(dim=1, keepdim=True) > r_cut).float()
        x = 0.8 * c + tail_mask * 0.90 + 0.20 * torch.randn(64, 2, generator=g)
        bank_tail.update_stratum((1,), c, x, y)
        bank_flat.update_stratum((1,), c, x, y)

    # Build probe batch containing 95% core + 5% outer-tail samples
    c_core = torch.randn(76, 2, generator=g) * 0.7
    c_outer = torch.tensor([[3.2, 0.4], [-3.1, -0.5], [0.3, 3.3], [-0.4, -3.2]])
    c_all = torch.cat((c_core, c_outer), dim=0)
    y_all = 0.8 * c_all + 0.20 * torch.randn(80, 2, generator=g)
    q_tail = bank_tail.query_stratum((1,), c_all, y_all, y_all)
    assert q_tail is not None
    assert float(q_tail["omega"][-4:].mean().item()) > 3.0 * float(q_tail["omega"][:76].mean().item())

    # Compare outer-5% tail corrective gradient magnitude with vs without tail leverage
    grads_outer = []
    for bank_i in (bank_flat, bank_tail):
        tail_bias = torch.tensor(0.90, requires_grad=True)
        mask_outer = torch.zeros(80, 1)
        mask_outer[-4:] = 1.0
        x_all = 0.8 * c_all + mask_outer * tail_bias + 0.20 * torch.randn(80, 2, generator=g)
        lv = conditional_fiber_loss(
            x_all,
            y_all,
            c_all,
            cond_disc=torch.ones(80, 1, dtype=torch.long),
            bank=bank_i,
            allow_state_update=False,
        )
        lv.backward()
        assert tail_bias.grad is not None and float(tail_bias.grad.item()) > 0.0
        grads_outer.append(float(tail_bias.grad.item()))
    assert grads_outer[1] > 1.3 * grads_outer[0], (
        f"tail_leverage_gain=0.90 did not amplify outer-5% tail gradient: {grads_outer}"
    )

    # Smoother gamma_fast=0.88 reduces outer-tail drift variance vs aggressive gamma_fast=0.50
    bank_smooth = AxiomCondFiberBank(gamma_fast=0.88)
    bank_noisy = AxiomCondFiberBank(gamma_fast=0.50)
    g_var = torch.Generator().manual_seed(405)
    drifts_s, drifts_n = [], []
    for step in range(24):
        c = torch.randn(32, 2, generator=g_var)
        y = 0.8 * c + 0.30 * torch.randn(32, 2, generator=g_var)
        x = 0.8 * c + 0.30 * torch.randn(32, 2, generator=g_var)
        qs = bank_smooth.query_stratum((1,), c_outer, 0.8 * c_outer, 0.8 * c_outer)
        qn = bank_noisy.query_stratum((1,), c_outer, 0.8 * c_outer, 0.8 * c_outer)
        if step >= 8 and qs is not None and qn is not None:
            drifts_s.append(float(qs["delta_m_bank"].square().mean().item()))
            drifts_n.append(float(qn["delta_m_bank"].square().mean().item()))
        bank_smooth.update_stratum((1,), c, x, y)
        bank_noisy.update_stratum((1,), c, x, y)
    assert sum(drifts_s) < sum(drifts_n)


@pytest.mark.parametrize("dim", [1, 2, 4, 5, 7, 8, 16])
def test_joint_od_equivariance_across_dimensions_no_solver_cliffs(dim: int) -> None:
    """Verify exact O(d_c) x O(d_x) equivariance of bank and loss across legacy cliff dims."""
    g = torch.Generator().manual_seed(500 + dim)
    q_c = _random_orthogonal(dim, g)
    q_x = _random_orthogonal(dim, g)
    aniso_c = torch.randn(dim, dim, generator=g, dtype=torch.float64) * (0.8 / math.sqrt(dim))
    slope_m = torch.randn(dim, dim, generator=g, dtype=torch.float64) * (0.7 / math.sqrt(dim))

    bank_a, bank_b = AxiomCondFiberBank(), AxiomCondFiberBank()
    batch = max(24, AxiomCondFiberBank.basis_dim(dim) + 6)
    for _ in range(3):
        c = torch.randn(batch, dim, generator=g, dtype=torch.float64) @ aniso_c
        y = c @ slope_m + 0.25 * torch.randn(batch, dim, generator=g, dtype=torch.float64)
        x = 0.5 * (c @ slope_m) + 0.40 * torch.randn(batch, dim, generator=g, dtype=torch.float64)
        bank_a.update_stratum((1,), c, x, y)
        bank_b.update_stratum((1,), c @ q_c.T, x @ q_x.T, y @ q_x.T)

    cq = torch.randn(20, dim, generator=g, dtype=torch.float64) @ aniso_c
    yq = cq @ slope_m + 0.25 * torch.randn(20, dim, generator=g, dtype=torch.float64)
    xq = 0.5 * (cq @ slope_m) + 0.40 * torch.randn(20, dim, generator=g, dtype=torch.float64)

    qa = bank_a.query_stratum((1,), cq, xq, yq)
    qb = bank_b.query_stratum((1,), cq @ q_c.T, xq @ q_x.T, yq @ q_x.T)
    assert qa is not None and qb is not None
    assert torch.allclose(qa["omega"], qb["omega"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(qa["m_pool"] @ q_x.T, qb["m_pool"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(qa["m_x"] @ q_x.T, qb["m_x"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(qa["delta_m_bank"] @ q_x.T, qb["delta_m_bank"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(qa["delta_m_loo"] @ q_x.T, qb["delta_m_loo"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(q_x @ qa["w_mat"] @ q_x.T, qb["w_mat"], atol=1e-9, rtol=1e-9)
    dcov_rot = torch.einsum("ab,nbc,dc->nad", q_x, qa["delta_cov_loc"], q_x)
    assert torch.allclose(dcov_rot, qb["delta_cov_loc"], atol=1e-10, rtol=1e-10)
    assert abs(qa["mean_err_sq"] - qb["mean_err_sq"]) < 1e-10
    assert abs(qa["z_innov"] - qb["z_innov"]) < 1e-10

    # End-to-end conditional_fiber_loss invariance and gradient equivariance in FP64
    xa = xq.clone().requires_grad_(True)
    xb = (xq @ q_x.T).clone().requires_grad_(True)
    disc = torch.ones(20, 1, dtype=torch.long)
    la = conditional_fiber_loss(xa, yq, cq, cond_disc=disc, bank=bank_a, allow_state_update=False)
    lb = conditional_fiber_loss(
        xb,
        yq @ q_x.T,
        cq @ q_c.T,
        cond_disc=disc,
        bank=bank_b,
        allow_state_update=False,
    )
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb, atol=1e-9, rtol=1e-9)
    assert torch.allclose(xa.grad @ q_x.T, xb.grad, atol=1e-9, rtol=1e-9)


def test_schema2_checkpoint_roundtrip_and_continuation() -> None:
    """Verify schema-2 serialization, innovation trim, and bit-exact continuation."""
    g = torch.Generator().manual_seed(606)
    bank = AxiomCondFiberBank(ring_capacity=64, knn_k=12)
    for _ in range(4):
        c = torch.randn(24, 2, generator=g)
        y = 1.2 * c + 0.25 * torch.randn(24, 2, generator=g)
        x = 0.9 * c + 0.40 * torch.randn(24, 2, generator=g)
        bank.update_stratum((1,), c, x, y)

    pkt = bank.state_dict()
    assert pkt["schema"] == SCHEMA_VERSION == 2
    assert len(pkt["strata"]) == 1
    st = pkt["strata"][0]
    for name in Fb._F64_KEYS:
        assert st[name].dtype == torch.float64 and st[name].device.type == "cpu"
    assert st["ring_c"].shape[0] == 64

    # Bit-exact query roundtrip via copy() and fresh load_state_dict()
    bank2 = AxiomCondFiberBank(ring_capacity=64, knn_k=12)
    bank2.load_state_dict(pkt)
    bank3 = bank.copy()
    cq, xq, yq = (
        torch.randn(16, 2, generator=g),
        torch.randn(16, 2, generator=g),
        torch.randn(16, 2, generator=g),
    )
    q1, q2, q3 = (
        bank.query_stratum((1,), cq, xq, yq),
        bank2.query_stratum((1,), cq, xq, yq),
        bank3.query_stratum((1,), cq, xq, yq),
    )
    assert q1 is not None and q2 is not None and q3 is not None
    for k, val in q1.items():
        if isinstance(val, torch.Tensor):
            assert torch.equal(val, q2[k]) and torch.equal(val, q3[k])
        else:
            assert val == q2[k] == q3[k]

    # Innovation shock trims FIFO ring and reduces n_eff
    n_before = float(bank.strata[(1,)]["n_eff"])
    bank.update_stratum((1,), cq[:8], xq[:8], yq[:8], z_innov=8.0)
    assert int(bank.strata[(1,)]["ring_c"].shape[0]) < 64
    assert float(bank.strata[(1,)]["n_eff"]) < 0.75 * n_before + 8.0

    # Bit-exact continuation through conditional_fiber_loss + lazy record hydration
    rec_a = _stub_record()
    t_disc = torch.cat(
        [torch.zeros(16, 1, dtype=torch.long), torch.ones(16, 1, dtype=torch.long)],
        dim=0,
    )
    for _ in range(3):
        c = torch.randn(32, 2, generator=g)
        y = 1.3 * c + 0.2 * torch.randn(32, 2, generator=g)
        x = (0.9 * c + 0.3 * torch.randn(32, 2, generator=g)).requires_grad_(True)
        conditional_fiber_loss(x, y, c, cond_disc=t_disc, record=rec_a).backward()
    assert rec_a.cond_initialized and 0.15 <= rec_a.step_scale <= 1.0

    saved_pkt = rec_a._cond_bank_obj.state_dict()
    rec_b = _stub_record(
        cond_fiber_bank=deepcopy(saved_pkt),
        cond_ema=rec_a.cond_ema,
        cond_initialized=True,
    )
    rec_b.step_scale = rec_a.step_scale
    cn, yn = torch.randn(32, 2, generator=g), torch.randn(32, 2, generator=g)
    xa = torch.randn(32, 2, generator=g).requires_grad_(True)
    xb = xa.detach().clone().requires_grad_(True)
    la = conditional_fiber_loss(xa, yn, cn, cond_disc=t_disc, record=rec_a)
    lb = conditional_fiber_loss(xb, yn, cn, cond_disc=t_disc, record=rec_b)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb) and torch.equal(xa.grad, xb.grad)
    assert rec_a.cond_ema == rec_b.cond_ema and rec_a.step_scale == rec_b.step_scale


def test_schema2_tamper_and_schema1_rejection() -> None:
    """Verify strict fail-closed rejection and atomic non-mutation on bad checkpoints."""
    g = torch.Generator().manual_seed(607)
    bank = AxiomCondFiberBank(ring_capacity=64, knn_k=12)
    for _ in range(2):
        c = torch.randn(24, 2, generator=g)
        bank.update_stratum((1,), c, c * 0.9, c * 1.1)
    pkt = bank.state_dict()

    bad_cases = []
    for mut in (
        lambda p: p.__setitem__("schema", 1),  # legacy v5 schema-1 rejected
        lambda p: p.__setitem__("schema", True),
        lambda p: p.__setitem__("schema", 3),
        lambda p: p.__setitem__("extra", 1),
        lambda p: p.pop("strata"),
        lambda p: p.__setitem__("d_c", 0),
        lambda p: p.__setitem__("d_x", True),
        lambda p: p.__setitem__("d_c", None),
        lambda p: (p.__setitem__("d_c", None), p.__setitem__("d_x", None)),
        lambda p: p["strata"].append(deepcopy(p["strata"][0])),
        lambda p: p["strata"][0].__setitem__("key", ()),
        lambda p: p["strata"][0].__setitem__("key", (True,)),
        lambda p: (
            p["strata"][0].pop("s_cc"),
            p["strata"][0].__setitem__("m2_c", torch.ones(2, dtype=torch.float64)),
        ),
        lambda p: p["strata"][0].__setitem__("n_eff", -1.0),
        lambda p: p["strata"][0].__setitem__("n_eff", True),
        lambda p: p["strata"][0].__setitem__("w_fast", float("nan")),
        lambda p: p["strata"][0].__setitem__("ring_c", p["strata"][0]["ring_c"].float()),
        lambda p: p["strata"][0].__setitem__("ring_x", p["strata"][0]["ring_x"][:-1]),
        lambda p: p["strata"][0].__setitem__("ring_y", torch.zeros(400, 2, dtype=torch.float64)),
        lambda p: p["strata"][0].__setitem__("mu_c", torch.zeros(3, dtype=torch.float64)),
        lambda p: p["strata"][0]["s_cc"].__setitem__((0, 0), -0.1),
        lambda p: p["strata"][0]["g_pp"].__setitem__((0, 0), -1.0),
        lambda p: p["strata"][0]["s_xx"].__setitem__((0, 0), float("nan")),
    ):
        cand = deepcopy(pkt)
        mut(cand)
        bad_cases.append(cand)
    for bad in bad_cases:
        with pytest.raises(ValueError):
            validate_fiber_bank_state(bad)
        snap = bank.state_dict()
        with pytest.raises(ValueError):
            bank.load_state_dict(bad)
        assert bank.d_c == snap["d_c"] and len(bank.strata) == len(snap["strata"])


def test_probe_guards_clock_freedom_edge_cases_and_zero_rng_consumption() -> None:
    """Verify zero global RNG mutation, probe/no_grad guards, clock-freedom, and degenerate inputs."""
    g = torch.Generator().manual_seed(707)
    c = torch.randn(32, 2, generator=g)
    y = 1.1 * c + 0.2 * torch.randn(32, 2, generator=g)
    x0 = 0.8 * c + 0.3 * torch.randn(32, 2, generator=g)
    disc = torch.zeros(32, 1, dtype=torch.long)

    # 1. Zero global RNG consumption across cold & warm forward + backward
    random.seed(12345)
    torch.manual_seed(12345)
    py_before = random.getstate()
    th_before = torch.get_rng_state()
    rec = _stub_record()
    for _ in range(3):
        x_in = x0.clone().requires_grad_(True)
        lv = conditional_fiber_loss(x_in, y, c, cond_disc=disc, record=rec)
        lv.backward()
    assert random.getstate() == py_before
    assert torch.equal(torch.get_rng_state(), th_before)

    # 2. Probe / no_grad / allow_state_update=False leave bank & record untouched
    snap_pkt = rec._cond_bank_obj.state_dict()
    snap_ema = rec.cond_ema
    with torch.no_grad():
        conditional_fiber_loss(x0.clone(), y, c, cond_disc=disc, record=rec)
    conditional_fiber_loss(
        x0.clone().requires_grad_(True),
        y,
        c,
        cond_disc=disc,
        record=rec,
        allow_state_update=False,
    )
    rec._in_probe = True
    conditional_fiber_loss(x0.clone().requires_grad_(True), y, c, cond_disc=disc, record=rec)
    rec._in_probe = False
    rec._in_protected_probe = True
    conditional_fiber_loss(x0.clone().requires_grad_(True), y, c, cond_disc=disc, record=rec)
    rec._in_protected_probe = False
    assert rec.cond_ema == snap_ema
    after_pkt = rec._cond_bank_obj.state_dict()
    assert after_pkt["strata"][0]["n_eff"] == snap_pkt["strata"][0]["n_eff"]
    assert torch.equal(after_pkt["strata"][0]["ring_c"], snap_pkt["strata"][0]["ring_c"])

    # 3. Strict clock-freedom: shifting calls & observed_steps by +10000 yields bit-identical outputs
    rec1 = _stub_record(calls=0, observed_steps=0)
    rec2 = _stub_record(calls=10000, observed_steps=10000)
    x1, x2 = x0.clone().requires_grad_(True), x0.clone().requires_grad_(True)
    l1 = conditional_fiber_loss(x1, y, c, cond_disc=disc, record=rec1)
    l2 = conditional_fiber_loss(x2, y, c, cond_disc=disc, record=rec2)
    l1.backward()
    l2.backward()
    assert torch.equal(l1, l2) and torch.equal(x1.grad, x2.grad)

    # 4. Degenerate inputs: bn <= 1, singleton strata, constant c / constant y
    assert float(conditional_fiber_loss(x0[:1], y[:1], c[:1]).item()) == 0.0
    uniq_disc = torch.arange(8).unsqueeze(1)
    assert float(conditional_fiber_loss(x0[:8], y[:8], c[:8], cond_disc=uniq_disc).item()) == 0.0
    c_const = torch.ones(24, 2)
    y_const = torch.zeros(24, 2)
    x_const = torch.randn(24, 2, generator=g, requires_grad=True)
    lv_c = conditional_fiber_loss(x_const, y_const, c_const, bank=AxiomCondFiberBank())
    lv_c.backward()
    assert torch.isfinite(lv_c) and x_const.grad is not None and torch.isfinite(x_const.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_fiber_cuda_device_restoration_and_zero_rng() -> None:
    """Verify CUDA forward/backward, exact device restoration on query/load, and zero CUDA RNG leak."""
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(808)
    torch.cuda.manual_seed(4242)
    cuda_rng_before = torch.cuda.get_rng_state(dev)

    rec = _stub_record()
    for dim in (2, 8):
        bank = AxiomCondFiberBank()
        batch = max(32, AxiomCondFiberBank.basis_dim(dim) + 8)
        disc = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        for _ in range(3):
            c = torch.randn(batch, dim, generator=g, device=dev)
            y = 1.1 * c + 0.2 * torch.randn(batch, dim, generator=g, device=dev)
            x = (0.8 * c + 0.3 * torch.randn(batch, dim, generator=g, device=dev)).requires_grad_(True)
            lv = conditional_fiber_loss(x, y, c, cond_disc=disc, bank=bank)
            lv.backward()
            assert lv.device == dev and torch.isfinite(lv)
            assert x.grad is not None and x.grad.device == dev and torch.isfinite(x.grad).all()
        q = bank.query_stratum((0,), c[:12], x.detach()[:12], y[:12])
        assert q is not None and q["m_pool"].device == dev and q["w_mat"].device == dev
        # Roundtrip through CPU state_dict and restore on CUDA
        pkt = bank.state_dict()
        assert pkt["strata"][0]["s_cc"].device.type == "cpu"
        bank_re = AxiomCondFiberBank()
        bank_re.load_state_dict(pkt)
        q_re = bank_re.query_stratum((0,), c[:12], x.detach()[:12], y[:12])
        assert q_re is not None and q_re["m_pool"].device == dev
        assert torch.equal(q["m_pool"], q_re["m_pool"])

    # Also test under record lazy hydration on CUDA
    for _ in range(3):
        c2 = torch.randn(32, 2, generator=g, device=dev)
        y2 = 1.2 * c2 + 0.2 * torch.randn(32, 2, generator=g, device=dev)
        x2 = (0.9 * c2 + 0.3 * torch.randn(32, 2, generator=g, device=dev)).requires_grad_(True)
        conditional_fiber_loss(x2, y2, c2, record=rec).backward()
    assert rec.cond_initialized
    assert torch.equal(torch.cuda.get_rng_state(dev), cuda_rng_before)


def test_quadratic_tail_curvature_and_smooth_negative_lyapunov_damping() -> None:
    """Verify O(d_c) quadratic basis detects cross-term curvature & E_- stays smoothly damped."""
    g = torch.Generator().manual_seed(909)
    bank = AxiomCondFiberBank()
    assert AxiomCondFiberBank.basis_dim(2) == 8
    assert AxiomCondFiberBank.basis_dim(4) == 19

    # Conditional target has pure quadratic curvature y = [c0*c1, c0^2 - c1^2] + noise; generator is flat
    for _ in range(8):
        c = torch.randn(64, 2, generator=g)
        quad_true = torch.stack((c[:, 0] * c[:, 1], 0.5 * (c[:, 0].square() - c[:, 1].square())), dim=1)
        y = 0.6 * c + 0.35 * quad_true + 0.15 * torch.randn(64, 2, generator=g)
        x = 0.6 * c + 0.15 * torch.randn(64, 2, generator=g)
        bank.update_stratum((1,), c, x, y)

    c_p = torch.randn(64, 2, generator=g)
    quad_p = torch.stack((c_p[:, 0] * c_p[:, 1], 0.5 * (c_p[:, 0].square() - c_p[:, 1].square())), dim=1)
    y_p = 0.6 * c_p + 0.35 * quad_p + 0.15 * torch.randn(64, 2, generator=g)
    curv = torch.tensor(0.0, requires_grad=True)
    x_p = 0.6 * c_p + curv * quad_p + 0.15 * torch.randn(64, 2, generator=g)
    lv = conditional_fiber_loss(
        x_p, y_p, c_p, cond_disc=torch.ones(64, 1, dtype=torch.long), bank=bank, allow_state_update=False
    )
    lv.backward()
    assert curv.grad is not None and float(curv.grad.item()) < -0.05

    # Smooth Lyapunov damping bounds negative-eigenvalue expansion gradient for large residuals
    w_op = torch.eye(2, dtype=torch.float64)[None]
    err_pos = torch.zeros(1, 2, 2, dtype=torch.float64)
    err_neg = -torch.eye(2, dtype=torch.float64)[None]
    zero_v = torch.zeros(1, 2, dtype=torch.float64)
    row_w = torch.ones(1, dtype=torch.float64)
    grads = []
    for rad in (1.0, 4.0, 20.0):
        xr = torch.tensor([[rad, 0.0]], dtype=torch.float64, requires_grad=True)
        op_l = Fb._fiber_operator_loss(
            xr, zero_v, w_op, (zero_v, zero_v, err_pos, err_neg, row_w), 0.0, 1.0, torch.float64
        )
        op_l.backward()
        grads.append(float(xr.grad.norm().item()))
    assert grads[0] > 0.0 and grads[2] < grads[1] < 1.0
