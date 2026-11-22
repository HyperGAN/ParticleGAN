"""Unit, semantic-control, memory-accounting, primitive-audit, and continuation tests for Law-4 alternative optimizers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import math
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, Recipe, get_recipe
from particlegan._law4_alt import (
    LAW4_ALT_FAMILIES,
    LAW4_ALT_OPTIMIZER_CLASSES,
    Law4AdafactorMPOptimizer,
    Law4AdamEquilOptimizer,
    Law4LionMPOptimizer,
    Law4MuonMOptimizer,
    Law4OptimisticEquilOptimizer,
    Law4SignMPOptimizer,
)
from particlegan._law4_alt import optimizer as law4_opt_mod
from particlegan._unified_v11 import optimizer as u11_opt_mod


def _build_toy_nets(dim: int = 2, hidden: int = 16, z_dim: int = 2):
    gen = nn.Sequential(nn.Linear(z_dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
    disc = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    prior = MoGParticlePrior(num_particles=16, z_dim=z_dim, sigma=0.05, standardize=False)
    return gen, disc, prior


def _v11_alt_recipe(family: str, **overrides) -> Recipe:
    defaults = dict(
        prior_kind="mog",
        num_particles=16,
        z_dim=2,
        standardize=False,
        optimizer_family=family,
    )
    defaults.update(overrides)
    return get_recipe("unified_v11", **defaults)


def test_law4_alt_preserves_default_and_recipe_schema():
    assert DEFAULT_RECIPE == "unified_v11"
    assert len(fields(Recipe)) == 179
    assert len(LAW4_ALT_FAMILIES) == 6
    assert set(LAW4_ALT_FAMILIES) == set(LAW4_ALT_OPTIMIZER_CLASSES)


@pytest.mark.parametrize("family", LAW4_ALT_FAMILIES)
def test_law4_alt_primitive_audit_and_underflow_safety(family: str, monkeypatch):
    """Enforce 0 SVD/eigh/pinv/autograd.grad on all 6, and 0 NS on non-NS families."""
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

    ns_families = {"law4_muon_m", "law4_optimistic_equil"}
    ns_calls = {"n": 0}
    if family not in ns_families:
        monkeypatch.setattr(law4_opt_mod, "quintic_orthogonalize_2d", _forbid("quintic_orthogonalize_2d"))
        monkeypatch.setattr(law4_opt_mod, "quintic_orthogonalize_batched", _forbid("quintic_orthogonalize_batched"))
        monkeypatch.setattr(u11_opt_mod, "quintic_orthogonalize_2d", _forbid("u11.quintic_orthogonalize_2d"))
        monkeypatch.setattr(u11_opt_mod, "quintic_orthogonalize_batched", _forbid("u11.quintic_orthogonalize_batched"))
    else:
        orig_2d = law4_opt_mod.quintic_orthogonalize_2d
        orig_b = law4_opt_mod.quintic_orthogonalize_batched

        def _wrap_2d(*args, **kwargs):
            ns_calls["n"] += 1
            return orig_2d(*args, **kwargs)

        def _wrap_b(*args, **kwargs):
            ns_calls["n"] += 1
            return orig_b(*args, **kwargs)

        monkeypatch.setattr(law4_opt_mod, "quintic_orthogonalize_2d", _wrap_2d)
        monkeypatch.setattr(law4_opt_mod, "quintic_orthogonalize_batched", _wrap_b)

    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _v11_alt_recipe(family)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    expected_cls = LAW4_ALT_OPTIMIZER_CLASSES[family]
    assert isinstance(trainer.opt_g, expected_cls)
    assert isinstance(trainer.opt_d, expected_cls)

    for _ in range(4):
        metrics = trainer.step(torch.randn(16, 2))
        assert all(torch.isfinite(torch.tensor(float(v))) for v in (metrics["loss_d"], metrics["loss_g"]))

    if family in ns_families:
        assert ns_calls["n"] > 0, f"{family} must actually invoke Quintic NS"

    # Underflow safety down to 1e-36 gradient magnitude on a 2D weight matrix:
    w = nn.Parameter(torch.randn(8, 8))
    opt = expected_cls([{"params": [w], "role": "generator"}], lr=0.01, smoothing=0.0061)
    w.grad = torch.full_like(w, 1e-36)
    w0 = w.detach().clone()
    opt.step()
    delta = (w - w0).abs().max().item()
    assert delta > 0.0 and torch.isfinite(w).all()


def test_law4_alt_state_memory_accounting():
    """Verify exact non-scalar state element counts (0 / sublinear O(m+n) / 1 / 2 states per param)."""
    rows, cols = 32, 16
    numel = rows * cols  # 512

    def _non_scalar_state_elems(opt_cls):
        torch.manual_seed(0)
        w = nn.Parameter(torch.randn(rows, cols))
        opt = opt_cls([{"params": [w], "role": "generator"}], lr=0.01, smoothing=0.0061)
        for step_idx in range(3):
            w.grad = torch.randn(rows, cols)
            opt.step()
        st = opt.state[w]
        return sum(
            v.numel() for v in st.values()
            if isinstance(v, torch.Tensor) and v.ndim > 0
        )

    assert _non_scalar_state_elems(Law4SignMPOptimizer) == 0
    assert _non_scalar_state_elems(Law4AdafactorMPOptimizer) == rows + cols  # 48 << 512
    assert _non_scalar_state_elems(Law4LionMPOptimizer) == numel  # 1 state
    assert _non_scalar_state_elems(Law4MuonMOptimizer) == numel  # 1 state
    assert _non_scalar_state_elems(Law4OptimisticEquilOptimizer) == numel  # 1 state
    assert _non_scalar_state_elems(Law4AdamEquilOptimizer) == 2 * numel  # 2 states


def test_law4_alt_exact_update_laws():
    """Semantic controls verifying each optimizer implements its exact declared mathematical geometry."""
    # 1. SignSGD-MP exact formula with smoothing=0 (hard sign) and smoothing>0 (g / hypot(|g|, s_eff)):
    w = nn.Parameter(torch.zeros(4, 8))
    g = torch.tensor(
        [
            [2.0, -0.5, 0.1, -3.0, 0.01, -0.02, 4.0, -1.0],
            [-1.0, 2.0, -0.5, 0.25, -4.0, 1.5, -0.1, 0.3],
            [0.5, -1.5, 3.0, -2.0, 0.2, -0.4, 1.0, -2.5],
            [-3.0, 0.8, -0.2, 1.2, -0.7, 2.2, -1.8, 0.6],
        ],
        dtype=torch.float32,
    )
    opt_hard = Law4SignMPOptimizer([{"params": [w], "role": "generator"}], lr=0.1, smoothing=0.0)
    w.grad = g.clone()
    opt_hard.step()
    mp_scale = math.sqrt(max(1.0, 4.0 / 8.0)) / (math.sqrt(4.0) + math.sqrt(8.0))
    expected_hard = -0.1 * mp_scale * torch.sign(g)
    assert torch.allclose(w.detach(), expected_hard, atol=1e-6)

    # Smooth sign check:
    w2 = nn.Parameter(torch.zeros(4, 8))
    opt_smooth = Law4SignMPOptimizer([{"params": [w2], "role": "generator"}], lr=0.1, smoothing=1.0)
    w2.grad = g.clone()
    opt_smooth.step()
    elem_rms = math.sqrt(float(g.square().mean().item()) + 1e-36)
    s_eff = min(1.0 / math.sqrt(8.0), 0.03 * elem_rms)
    expected_smooth = -0.1 * mp_scale * (g / torch.hypot(g.abs(), torch.tensor(s_eff)))
    assert torch.allclose(w2.detach(), expected_smooth, atol=1e-6)

    # 2. Adafactor-MP exact rank-1 Kronecker RMS reconstruction on G = a @ b^T:
    a = torch.tensor([1.0, 4.0, 9.0, 16.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    g_rank1 = torch.outer(a, b)
    w_af = nn.Parameter(torch.zeros(4, 8))
    opt_af = Law4AdafactorMPOptimizer([{"params": [w_af], "role": "generator"}], lr=0.1, smoothing=0.0, settle_rel=0.0)
    w_af.grad = g_rank1.clone()
    opt_af.step()
    # Since g_rank1 is positive rank-1, exact factored v_hat == g_rank1, so u_elem == +1 everywhere!
    expected_af = -0.1 * mp_scale * torch.ones(4, 8)
    assert torch.allclose(w_af.detach(), expected_af, atol=1e-5)

    # 3. Optimistic vs Muon opposite-sign momentum geometry on a rotating 2D gradient:
    # Step 1: g_1 = [1, 0]; Step 2: g_2 = [0, 1] (rotated +90 deg).
    # Muon (positive momentum) keeps positive x-component (lags behind g_2);
    # Optimistic (negative momentum: g_2 + 0.5*(g_2 - g_1) = [-0.5, 1.5]) has NEGATIVE x-component (leads ahead of g_2)!
    w_muon = nn.Parameter(torch.zeros(2))
    w_opt = nn.Parameter(torch.zeros(2))
    opt_muon = Law4MuonMOptimizer([{"params": [w_muon], "role": "generator"}], lr=0.1, smoothing=0.0, muon_beta=0.85)
    opt_optim = Law4OptimisticEquilOptimizer([{"params": [w_opt], "role": "generator"}], lr=0.1, smoothing=0.0, opt_lambda=0.5)
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([0.0, 1.0])
    w_muon.grad = g1.clone()
    w_opt.grad = g1.clone()
    opt_muon.step()
    opt_optim.step()
    w_muon_1 = w_muon.detach().clone()
    w_opt_1 = w_opt.detach().clone()
    # Step 1 should be identical for both:
    assert torch.allclose(w_muon_1, w_opt_1, atol=1e-6)

    w_muon.grad = g2.clone()
    w_opt.grad = g2.clone()
    opt_muon.step()
    opt_optim.step()
    d_muon_2 = w_muon.detach() - w_muon_1
    d_opt_2 = w_opt.detach() - w_opt_1
    # Update is -lr * dir, so Muon has d_x < 0 (lagging toward +x grad), Optimistic has d_x > 0 (leading toward -x grad):
    assert d_muon_2[0].item() < -0.01
    assert d_opt_2[0].item() > 0.01

    # 4. Lion step 2 retains positive memory of step 1 (unlike zero-state SignSGD):
    w_lion = nn.Parameter(torch.zeros(2, 2))
    w_sgn = nn.Parameter(torch.zeros(2, 2))
    opt_lion = Law4LionMPOptimizer([{"params": [w_lion], "role": "generator"}], lr=0.1, smoothing=0.0, lion_beta1=0.5, lion_beta2=0.9)
    opt_sgn = Law4SignMPOptimizer([{"params": [w_sgn], "role": "generator"}], lr=0.1, smoothing=0.0)
    g_a = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    g_b = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    w_lion.grad = g_a.clone()
    w_sgn.grad = g_a.clone()
    opt_lion.step()
    opt_sgn.step()
    w_lion_1 = w_lion.detach().clone()
    w_sgn_1 = w_sgn.detach().clone()
    w_lion.grad = g_b.clone()
    w_sgn.grad = g_b.clone()
    opt_lion.step()
    opt_sgn.step()
    # In col 0, 0.5*10 + 0.5*(-1) = +4.5 > 0, so Lion still moves negatively while SignSGD reverses positively:
    assert (w_lion.detach() - w_lion_1)[0, 0].item() < 0.0
    assert (w_sgn.detach() - w_sgn_1)[0, 0].item() > 0.0


@pytest.mark.parametrize("family", LAW4_ALT_FAMILIES)
def test_law4_alt_checkpoint_tamper_and_fresh_process_continuation(family: str, tmp_path):
    """Verify strict checkpoint tamper rejection and bit-exact fresh-process continuation for all 6 families."""
    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _v11_alt_recipe(family)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    for _ in range(3):
        trainer.step(torch.randn(16, 2))

    # Tamper checks (always deepcopy because Optimizer.state_dict shares inner param state dicts):
    bad_family = deepcopy(trainer.opt_g.state_dict())
    bad_family["law4_alt"]["family"] = "unified_v11_equil"
    with pytest.raises(ValueError, match="incompatible"):
        trainer.opt_g.load_state_dict(bad_family)

    bad_gain = deepcopy(trainer.opt_g.state_dict())
    bad_gain["law4_alt"]["last_step_gain"] = -1.0
    with pytest.raises(ValueError, match="last_step_gain"):
        trainer.opt_g.load_state_dict(bad_gain)

    bad_key = deepcopy(trainer.opt_g.state_dict())
    first_pid = next(iter(bad_key["state"]))
    bad_key["state"][first_pid]["bogus_buf"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected state key"):
        trainer.opt_g.load_state_dict(bad_key)

    ckpt_path = tmp_path / f"{family}.pt"
    saved_state = trainer.state_dict()
    torch.save(saved_state, ckpt_path)

    # Exact in-process dtype & bit parity check:
    gen_chk, disc_chk, prior_chk = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    tr_chk = GANTrainer(recipe, gen_chk, disc_chk, prior=prior_chk, seed=0)
    tr_chk.load_state_dict(saved_state)
    for opt_a, opt_b in ((trainer.opt_g, tr_chk.opt_g), (trainer.opt_d, tr_chk.opt_d)):
        sd_a, sd_b = opt_a.state_dict()["state"], opt_b.state_dict()["state"]
        assert set(sd_a) == set(sd_b)
        for pid, st_a in sd_a.items():
            st_b = sd_b[pid]
            for tk, tv in st_a.items():
                if isinstance(tv, torch.Tensor):
                    assert tv.dtype == st_b[tk].dtype and torch.equal(tv, st_b[tk])

    # Fresh-process bit-exact continuation:
    batches = [torch.randn(16, 2) for _ in range(3)]
    torch.save(batches, tmp_path / f"{family}_batches.pt")
    for b in batches:
        trainer.step(b)
    ref_z = trainer.prior.z.detach().clone()
    ref_g = [p.detach().clone() for p in trainer.G.parameters()]

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
        f"for b in torch.load({str(tmp_path / f'{family}_batches.pt')!r}, weights_only=False):\n"
        "    tr.step(b)\n"
        f"torch.save({{'z': tr.prior.z.detach(), 'g': [p.detach() for p in tr.G.parameters()]}}, {str(tmp_path / f'{family}_out.pt')!r})\n"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    out = torch.load(tmp_path / f"{family}_out.pt", weights_only=False)
    assert torch.equal(ref_z, out["z"])
    for p1, p2 in zip(ref_g, out["g"]):
        assert torch.equal(p1, p2)
