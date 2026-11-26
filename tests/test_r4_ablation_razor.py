"""Unit, semantic-control, primitive-audit, and fresh-process continuation tests for Round-4 Occam's Razor ablations."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, Recipe, get_recipe
from particlegan._r4_ablation import (
    R4_ABLATION_PRIOR_REGULARIZERS,
    R4AblationPriorRegularizer,
    R4AutogradCriticPenalty,
    R4MPNoNSOptimizer,
    R4NoSecantBandOptimizer,
)
from particlegan._unified_v4 import optimizer as u4_opt


def _build_toy_nets(dim: int = 2, hidden: int = 16, z_dim: int = 2):
    gen = nn.Sequential(nn.Linear(z_dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
    disc = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    prior = MoGParticlePrior(num_particles=16, z_dim=z_dim, sigma=0.05, standardize=False)
    return gen, disc, prior


def _unified_v4_recipe(**overrides) -> Recipe:
    defaults = dict(prior_kind="mog", num_particles=16, z_dim=2, standardize=False)
    defaults.update(overrides)
    return get_recipe("unified_v4", **defaults)


def test_r4_ablation_contract_preserves_default_and_recipe_fields():
    """Verify DEFAULT_RECIPE == 'axiom' and exact 179 Recipe fields."""
    assert DEFAULT_RECIPE == "axiom"
    assert len(fields(Recipe)) == 179
    view_path = Path(__file__).resolve().parents[1] / "configs/forge/views/r4-occam-ablation-v1.json"
    view = json.loads(view_path.read_text(encoding="utf-8"))
    assert view["evidence_scope"] == "research_diagnostic"
    assert len(view["assignments"]) == 16


def test_r4_mp_no_ns_strictly_zero_svd_zero_ns_zero_autograd_and_underflow_safe(monkeypatch):
    """Arm 1 (-NS): monkeypatch-enforce 0 SVD, 0 eigh, 0 pinv, 0 autograd.grad, AND 0 Quintic NS."""
    def _forbid(name):
        def _boom(*args, **kwargs):
            raise AssertionError(f"forbidden primitive called in r4_mp_no_ns: {name}")
        return _boom

    monkeypatch.setattr(torch.linalg, "svd", _forbid("svd"))
    monkeypatch.setattr(torch.linalg, "svdvals", _forbid("svdvals"))
    monkeypatch.setattr(torch.linalg, "eigh", _forbid("eigh"))
    monkeypatch.setattr(torch.linalg, "eigvalsh", _forbid("eigvalsh"))
    monkeypatch.setattr(torch.linalg, "pinv", _forbid("pinv"))
    monkeypatch.setattr(torch.autograd, "grad", _forbid("autograd.grad"))
    monkeypatch.setattr(u4_opt, "quintic_orthogonalize_2d", _forbid("quintic_orthogonalize_2d"))
    monkeypatch.setattr(u4_opt, "quintic_orthogonalize_batched", _forbid("quintic_orthogonalize_batched"))

    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _unified_v4_recipe(optimizer_family="r4_mp_no_ns", optimizer_smoothing=0.0061)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    assert isinstance(trainer.opt_g, R4MPNoNSOptimizer)
    assert isinstance(trainer.opt_d, R4MPNoNSOptimizer)

    for _ in range(4):
        real = torch.randn(16, 2)
        metrics = trainer.step(real)
        assert all(torch.isfinite(torch.tensor(float(v))) for v in (metrics["loss_d"], metrics["loss_g"]))

    # Underflow safety down to 1e-36 gradient magnitude on a 2D weight matrix:
    w = nn.Parameter(torch.randn(8, 8))
    opt = R4MPNoNSOptimizer([{"params": [w], "role": "generator"}], lr=0.01)
    w.grad = torch.full_like(w, 1e-36)
    w0 = w.detach().clone()
    opt.step()
    delta = (w - w0).abs().max().item()
    assert delta > 0.0 and torch.isfinite(w).all()


def test_r4_no_secant_band_strictly_unit_gain():
    """Arm 2 (-secant-band): verify alpha_gain is locked to 1.0 and probes are never retained."""
    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _unified_v4_recipe(optimizer_family="r4_no_secant_band")
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    assert isinstance(trainer.opt_g, R4NoSecantBandOptimizer)
    assert isinstance(trainer.opt_d, R4NoSecantBandOptimizer)
    for _ in range(4):
        trainer.step(torch.randn(16, 2) * 5.0)
        assert trainer.opt_g.last_step_gain == 1.0
        assert trainer.opt_d.last_step_gain == 1.0
        assert trainer.opt_d.record._enable_secant_probe is False


def test_r4_autograd_critic_actually_uses_second_order_autograd(monkeypatch):
    """Arm 7 (-fwad-critic): semantic control verifying torch.autograd.grad IS called."""
    calls = {"n": 0}
    orig_grad = torch.autograd.grad

    def _counting_grad(*args, **kwargs):
        calls["n"] += 1
        return orig_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _counting_grad)
    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _unified_v4_recipe(critic_formulation="r4_autograd_critic")
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    assert isinstance(trainer.penalty, R4AutogradCriticPenalty)
    for _ in range(3):
        trainer.step(torch.randn(16, 2))
    assert calls["n"] >= 3


@pytest.mark.parametrize("prior_reg", R4_ABLATION_PRIOR_REGULARIZERS)
def test_r4_prior_ablation_modes_step_and_checkpoint_roundtrip(prior_reg):
    """Arms 3, 4, 5, 8: each prior knockout constructs, steps cleanly, and rejects tampered checkpoints."""
    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    extra = {"t1c_chart_weight": 0.0} if prior_reg == "r4_no_chart_laguerre" else {}
    recipe = _unified_v4_recipe(prior_regularizer=prior_reg, **extra)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    assert isinstance(trainer.prior_regularizer, R4AblationPriorRegularizer)
    assert trainer.prior_regularizer.ablation_mode == prior_reg

    for _ in range(4):
        real = torch.zeros(16, 2) if prior_reg == "r4_no_sat_probe" else torch.randn(16, 2)
        trainer.step(real)

    ckpt = trainer.state_dict()
    gen2, disc2, prior2 = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    trainer2 = GANTrainer(recipe, gen2, disc2, prior=prior2, seed=0)
    trainer2.load_state_dict(ckpt)
    batch = torch.randn(16, 2)
    m1 = trainer.step(batch.clone())
    m2 = trainer2.step(batch.clone())
    assert abs(m1["loss_d"] - m2["loss_d"]) < 1e-6
    assert abs(m1["loss_g"] - m2["loss_g"]) < 1e-6


def test_r4_mp_no_ns_checkpoint_rejects_tampering_and_fresh_process_continuation(tmp_path):
    """Verify strict checkpoint validation and bit-exact fresh-process continuation for R4MPNoNSOptimizer."""
    torch.manual_seed(0)
    gen, disc, prior = _build_toy_nets(dim=2, hidden=16, z_dim=2)
    recipe = _unified_v4_recipe(optimizer_family="r4_mp_no_ns", optimizer_smoothing=0.0061)
    trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
    for _ in range(3):
        trainer.step(torch.randn(16, 2))
    bad = trainer.opt_g.state_dict()
    bad["unified_equil"]["family"] = "unified_equil"
    with pytest.raises(ValueError, match="incompatible R4MPNoNSOptimizer metadata"):
        trainer.opt_g.load_state_dict(bad)

    ckpt_path = tmp_path / "r4_mp_no_ns.pt"
    saved_state = trainer.state_dict()
    torch.save(saved_state, ckpt_path)
    # Verify load_state_dict preserves exact tensor dtypes/values (clockfree & gaussian1d_stability contract):
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
    batches = [torch.randn(16, 2) for _ in range(3)]
    torch.save(batches, tmp_path / "batches.pt")
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
        "recipe = get_recipe('unified_v4', prior_kind='mog', num_particles=16, z_dim=2, standardize=False, optimizer_family='r4_mp_no_ns', optimizer_smoothing=0.0061)\n"
        "tr = GANTrainer(recipe, gen, disc, prior=prior, seed=0)\n"
        f"tr.load_state_dict(torch.load({str(ckpt_path)!r}, weights_only=False))\n"
        f"for b in torch.load({str(tmp_path / 'batches.pt')!r}, weights_only=False):\n"
        "    tr.step(b)\n"
        f"torch.save({{'z': tr.prior.z.detach(), 'g': [p.detach() for p in tr.G.parameters()]}}, {str(tmp_path / 'out.pt')!r})\n"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    out = torch.load(tmp_path / "out.pt", weights_only=False)
    assert torch.equal(ref_z, out["z"])
    for p1, p2 in zip(ref_g, out["g"]):
        assert torch.equal(p1, p2)
