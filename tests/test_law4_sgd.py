"""Unit, semantic-control, primitive-audit, and fresh-process continuation tests for Law-4 SGD anatomy ladder."""
from __future__ import annotations

from dataclasses import fields
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, Recipe, get_recipe
from particlegan._law4_sgd import (
    LAW4_SGD_FAMILIES,
    Law4EquilGSGDDOptimizer,
    Law4FrobSGDMPOptimizer,
    Law4PureSGDOptimizer,
    Law4SGDGEquilDOptimizer,
)
from particlegan._unified_v11 import optimizer as u11_opt
from particlegan.optim.constraint_geometry import constraint_geometry_backward


def _build_toy_nets(dim: int = 2, hidden: int = 16, z_dim: int = 2):
    gen = nn.Sequential(nn.Linear(z_dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
    disc = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    prior = MoGParticlePrior(num_particles=16, z_dim=z_dim, sigma=0.05, standardize=False)
    return gen, disc, prior


def _v11_recipe(optimizer_family: str, **overrides) -> Recipe:
    defaults = dict(
        prior_kind="mog",
        num_particles=16,
        z_dim=2,
        standardize=False,
        optimizer_family=optimizer_family,
    )
    defaults.update(overrides)
    return get_recipe("unified_v11", **defaults)


def test_law4_sgd_contract_preserves_default_and_recipe_fields():
    assert DEFAULT_RECIPE == "unified_v11"
    assert len(fields(Recipe)) == 179
    assert len(LAW4_SGD_FAMILIES) == 6


def test_law4_pure_sgd_semantic_control_exact_unnormalized_update_and_sparse_prior():
    """Semantic control: pure SGD MUST NOT normalize gradients and MUST leave unsampled prior rows untouched."""
    torch.manual_seed(0)
    w_big = nn.Parameter(torch.randn(8, 8))
    w_small = nn.Parameter(torch.randn(8, 8))
    prior_tbl = nn.Parameter(torch.randn(12, 2))
    opt = Law4PureSGDOptimizer(
        [
            {"params": [w_big, w_small], "role": "generator", "lr": 0.02},
            {"params": [prior_tbl], "role": "prior", "lr": 0.05},
        ],
        smoothing=0.0061,
    )
    g_big = torch.randn_like(w_big) * 10.0
    g_small = torch.randn_like(w_small) * 1e-3
    g_prior = torch.randn_like(prior_tbl) * 3.0
    w_big.grad = g_big.clone()
    w_small.grad = g_small.clone()
    prior_tbl.grad = g_prior.clone()

    sampled = torch.tensor([1, 4, 9], dtype=torch.long)
    opt.set_sampled_rows(prior_tbl, sampled)

    w_big0 = w_big.detach().clone()
    w_small0 = w_small.detach().clone()
    p0 = prior_tbl.detach().clone()
    opt.step()

    # Exact unnormalized SGD: w_new == w0 - lr * grad (proves NO gradient normalization!)
    assert torch.allclose(w_big.detach(), w_big0 - 0.02 * g_big, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w_small.detach(), w_small0 - 0.02 * g_small, atol=1e-6, rtol=1e-6)
    d_big = w_big.detach() - w_big0
    d_small = w_small.detach() - w_small0
    # Ratio of step norms matches ratio of gradient norms (~1e4), proving no per-layer normalization:
    assert (d_big.norm() / d_small.norm()).item() > 1e3

    # Sparse prior masking: sampled rows get -0.05 * grad, unsampled rows get strictly 0
    dp = prior_tbl.detach() - p0
    assert torch.allclose(dp[sampled], -0.05 * g_prior[sampled], atol=1e-7, rtol=1e-6)
    unsampled = torch.tensor([i for i in range(12) if i not in {1, 4, 9}], dtype=torch.long)
    assert torch.equal(dp[unsampled], torch.zeros_like(dp[unsampled]))

    # Strictly 0 tensor states per param:
    for st in opt.state.values():
        assert not any(isinstance(v, torch.Tensor) for v in st.values())


def test_law4_asymmetric_hybrids_semantic_controls():
    """Verify G/D asymmetry: sgd_g_equil_d normalizes D+prior only; equil_g_sgd_d normalizes G+prior only."""
    torch.manual_seed(0)
    # 1. law4_sgd_g_equil_d: G unnormalized, D normalized, prior row-normalized
    wg = nn.Parameter(torch.randn(8, 8))
    pr = nn.Parameter(torch.randn(6, 2))
    opt_g = Law4SGDGEquilDOptimizer(
        [
            {"params": [wg], "role": "generator", "lr": 0.014},
            {"params": [pr], "role": "prior", "lr": 0.04},
        ],
        smoothing=0.0,
    )
    wg.grad = torch.full_like(wg, 5.0)
    pr.grad = torch.full_like(pr, 5.0)
    rows = torch.tensor([0, 2], dtype=torch.long)
    opt_g.set_sampled_rows(pr, rows)
    wg0, pr0 = wg.detach().clone(), pr.detach().clone()
    opt_g.step()
    # G is pure unnormalized SGD (-0.014 * 5.0 = -0.07 per entry):
    assert torch.allclose(wg.detach() - wg0, -0.014 * wg.grad, atol=1e-7)
    # Prior is row-normalized (each sampled row step norm == 0.04, NOT 0.04 * 5 * sqrt(2)):
    dp_rows = (pr.detach() - pr0)[rows]
    assert torch.allclose(dp_rows.norm(dim=-1), torch.full((2,), 0.04), atol=1e-5)

    wd = nn.Parameter(torch.randn(8, 8))
    opt_d = Law4SGDGEquilDOptimizer([{"params": [wd], "role": "critic", "lr": 0.02}], smoothing=0.0)
    wd.grad = torch.eye(8) * 100.0
    wd0 = wd.detach().clone()
    opt_d.step()
    # Equilibrated D step norm is bounded ~0.02 * sqrt(8), NOT 0.02 * 100 * sqrt(8):
    assert (wd.detach() - wd0).norm().item() < 0.2

    # 2. law4_equil_g_sgd_d: reverse control (G equilibrated, D pure unnormalized SGD)
    wd2 = nn.Parameter(torch.randn(8, 8))
    opt_d2 = Law4EquilGSGDDOptimizer([{"params": [wd2], "role": "critic", "lr": 0.02}], smoothing=0.0)
    wd2.grad = torch.full_like(wd2, 5.0)
    wd2_0 = wd2.detach().clone()
    opt_d2.step()
    assert torch.allclose(wd2.detach() - wd2_0, -0.02 * wd2.grad, atol=1e-7)

    wg2 = nn.Parameter(torch.randn(8, 8))
    opt_g2 = Law4EquilGSGDDOptimizer([{"params": [wg2], "role": "generator", "lr": 0.014}], smoothing=0.0)
    wg2.grad = torch.eye(8) * 100.0
    wg2_0 = wg2.detach().clone()
    opt_g2.step()
    assert (wg2.detach() - wg2_0).norm().item() < 0.2


def test_law4_frob_sgd_mp_exact_formula_zero_tensor_state_and_underflow_safe():
    """Verify delta_W = -lr * sqrt(r/c) * G / hypot(||G||_F, s_eff), 0 tensor states/param, and 1e-36 safety."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(4, 16))
    opt = Law4FrobSGDMPOptimizer([{"params": [w], "role": "generator", "lr": 0.02}], smoothing=0.0)
    g = torch.randn_like(w) * 7.5
    w.grad = g.clone()
    w0 = w.detach().clone()
    opt.step()
    expected = -0.02 * (4.0 / 16.0) ** 0.5 * (g / g.norm())
    assert torch.allclose(w.detach() - w0, expected, atol=1e-6, rtol=1e-5)

    # Strictly 0 non-scalar tensor states per parameter (only 0-D scalar peak_rms):
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                assert k == "peak_rms" and v.ndim == 0 and v.dtype == torch.float32

    # Underflow-safe down to 1e-36 gradient magnitude (sech^2 rail escape):
    w_tiny = nn.Parameter(torch.randn(8, 8))
    opt_tiny = Law4FrobSGDMPOptimizer([{"params": [w_tiny], "role": "generator", "lr": 0.01}], smoothing=0.0061)
    w_tiny.grad = torch.full_like(w_tiny, 1e-36)
    w_tiny0 = w_tiny.detach().clone()
    opt_tiny.step()
    delta_tiny = (w_tiny.detach() - w_tiny0).norm().item()
    assert delta_tiny > 0.009 and torch.isfinite(w_tiny).all()


@pytest.mark.parametrize("family", LAW4_SGD_FAMILIES)
def test_law4_sgd_zero_forbidden_primitives_and_2loss_direction_blend(monkeypatch, family):
    """All 6 families: 0 SVD/eigh/pinv/autograd.grad; 0 NS on pure-SGD & Frob-SGD arms; 2-loss direction_blend works."""
    def _forbid(name):
        def _boom(*args, **kwargs):
            raise AssertionError(f"forbidden primitive called in {family}: {name}")
        return _boom

    monkeypatch.setattr(torch.linalg, "svd", _forbid("svd"))
    monkeypatch.setattr(torch.linalg, "svdvals", _forbid("svdvals"))
    monkeypatch.setattr(torch.linalg, "eigh", _forbid("eigh"))
    monkeypatch.setattr(torch.linalg, "eigvalsh", _forbid("eigvalsh"))
    monkeypatch.setattr(torch.linalg, "pinv", _forbid("pinv"))
    monkeypatch.setattr(torch.autograd, "grad", _forbid("autograd.grad"))
    if family in ("law4_pure_sgd", "law4_sgd_secant", "law4_frob_sgd_mp"):
        monkeypatch.setattr(u11_opt, "quintic_orthogonalize_2d", _forbid("quintic_orthogonalize_2d"))
        monkeypatch.setattr(u11_opt, "quintic_orthogonalize_batched", _forbid("quintic_orthogonalize_batched"))

    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _v11_recipe(family)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    assert trainer.opt_g.family == family
    assert trainer.opt_d.family == family

    for _ in range(4):
        m = trainer.step(torch.randn(16, 2))
        assert all(torch.isfinite(torch.tensor(float(v))) for v in (m["loss_d"], m["loss_g"]))

    # Verify 2-objective zero-pinv direction_blend executes cleanly:
    x = nn.Parameter(torch.tensor([1.0, 1.0]))
    opt_db = type(trainer.opt_g)([{"params": [x], "role": "generator", "lr": 0.05}], direction_blend=True)
    l1 = (x[0] - 2.0).square()
    l2 = (x[1] + 2.0).square()
    total = l1 + l2
    opt_db.zero_grad()
    constraint_geometry_backward(total, opt_db, (l1, l2))
    opt_db.step()
    assert opt_db.constraint_geometry_stats["steps"] == 1
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("family", LAW4_SGD_FAMILIES)
def test_law4_sgd_checkpoint_tamper_rejection_and_fresh_process_continuation(tmp_path, family):
    """Verify strict checkpoint validation, tamper rejection, and bit-exact fresh-process continuation."""
    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _v11_recipe(family)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    for _ in range(3):
        trainer.step(torch.randn(16, 2))

    # Tamper rejection: family mismatch & invalid gain
    bad = trainer.opt_g.state_dict()
    bad["unified_v11_equil"]["family"] = "unified_v11_equil"
    with pytest.raises(ValueError, match="incompatible UnifiedV11EquilibratedOptimizer metadata"):
        trainer.opt_g.load_state_dict(bad)

    bad2 = trainer.opt_g.state_dict()
    bad2["unified_v11_equil"]["last_step_gain"] = -1.0
    with pytest.raises(ValueError, match="last_step_gain"):
        trainer.opt_g.load_state_dict(bad2)

    # Save checkpoint & verify bit-exact fresh-process continuation:
    ckpt_path = tmp_path / f"{family}.pt"
    saved = trainer.state_dict()
    torch.save(saved, ckpt_path)
    batches = [torch.randn(16, 2) for _ in range(3)]
    batches_path = tmp_path / f"{family}_batches.pt"
    torch.save(batches, batches_path)
    for b in batches:
        trainer.step(b)
    ref_z = trainer.prior.z.detach().clone()
    ref_g = [p.detach().clone() for p in trainer.G.parameters()]
    ref_d = [p.detach().clone() for p in trainer.D.parameters()]

    out_path = tmp_path / f"{family}_out.pt"
    script = (
        "import torch\n"
        "from torch import nn\n"
        "from particlegan import GANTrainer, MoGParticlePrior, get_recipe\n"
        "torch.manual_seed(999)\n"
        "gen = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))\n"
        "disc = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))\n"
        "prior = MoGParticlePrior(num_particles=16, z_dim=2, sigma=0.05, standardize=False)\n"
        f"recipe = get_recipe('unified_v11', prior_kind='mog', num_particles=16, z_dim=2, standardize=False, optimizer_family={family!r})\n"
        "tr = GANTrainer(recipe, gen, disc, prior=prior, seed=0)\n"
        f"tr.load_state_dict(torch.load({str(ckpt_path)!r}, weights_only=False))\n"
        f"for b in torch.load({str(batches_path)!r}, weights_only=False):\n"
        "    tr.step(b)\n"
        f"torch.save({{'z': tr.prior.z.detach(), 'g': [p.detach() for p in tr.G.parameters()], 'd': [p.detach() for p in tr.D.parameters()]}}, {str(out_path)!r})\n"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    out = torch.load(out_path, weights_only=False)
    assert torch.equal(ref_z, out["z"])
    for p1, p2 in zip(ref_g, out["g"]):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(ref_d, out["d"]):
        assert torch.equal(p1, p2)
