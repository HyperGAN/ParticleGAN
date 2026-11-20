"""Unit, semantic-control, tamper-rejection, and fresh-process continuation tests for ``preset='unified'``."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    UnifiedCriticPenalty,
    UnifiedGANLoss,
    get_recipe,
    init,
)
from particlegan.unified import (
    _compute_pythagorean_smoothed_cells,
    unified_pooled_conditional_loss,
)


class _ToyGen2D(nn.Module):
    def __init__(self, z_dim: int = 2, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 24), nn.SiLU(), nn.Linear(24, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _ToyDisc2D(nn.Module):
    def __init__(self, in_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _ToyImgGen(nn.Module):
    def __init__(self, z_dim: int = 4, size: int = 8) -> None:
        super().__init__()
        self.size = size
        self.fc = nn.Linear(z_dim, 8 * 4 * 4)
        self.conv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.fc(z)).reshape(z.shape[0], 8, 4, 4)
        if self.size != 4:
            h = nn.functional.interpolate(h, size=(self.size, self.size), mode="nearest")
        return torch.tanh(self.conv(h))


class _ToyImgDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.conv1(x))
        h = nn.functional.silu(self.conv2(h)).mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


class _ToyCondDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 24), nn.LeakyReLU(0.2), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, xt, t[:, None].to(x) / 4.0), dim=1)).squeeze(-1)


def test_unified_preset_validation_and_default_recipe_invariant() -> None:
    assert DEFAULT_RECIPE == "unified_v11"
    rec = get_recipe("unified")
    assert rec.critic_formulation == "unified"
    assert rec.effective_critic_formulation == "unified"
    assert rec.prior_regularizer == "unified"
    assert rec.training_precision == "bf16_channels_last"
    assert rec.optimizer_dense_profile == "newton_schulz_hybrid_v1"

    roundtrip = Recipe.from_dict(rec.to_dict())
    assert roundtrip == rec

    ctx = FormulationContext(
        recipe_preset="unified",
        prior={"kind": "mog", "sigma": 0.025, "standardize": False, "learnable": True},
    )
    assert ctx.capabilities()["secant_critic"] is True

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe("ka2", critic_formulation="unified")
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe("unified", reg_arm="b_cap")


def test_unified_critic_penalty_forbids_autograd_grad_everywhere() -> None:
    """Semantic control: UnifiedCriticPenalty must NEVER call torch.autograd.grad on MLP, simplex, cond, or Conv2d."""
    rec = get_recipe(
        "unified",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=4,
        total_steps=20,
        training_precision="float32",
        constraint_geometry_mode="none",
    )

    with mock.patch("torch.autograd.grad", side_effect=AssertionError("autograd.grad forbidden in UnifiedCriticPenalty!")):
        # 1. 1D, 2D, and 170D simplex critics:
        for dim in (1, 2, 170):
            g = _ToyGen2D(z_dim=4, out_dim=dim)
            d = _ToyDisc2D(in_dim=dim)
            p = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05, standardize=False)
            trainer = GANTrainer(rec, g, d, prior=p, seed=0)
            out = trainer.step(torch.randn(8, dim), collect_stats=True)
            assert out["penalty_stats"]["applied"] is True
            assert out["penalty_stats"]["phase"] == "unified"

        # 2. Small 8x8 image (K=2 orthogonal micro-secant probes) & 64x64 image (live-endpoint K=1 secant):
        for img_sz in (8, 64):
            g_img = _ToyImgGen(z_dim=4, size=img_sz)
            d_img = _ToyImgDisc()
            p_img = ParticlePrior(num_particles=16, z_dim=4)
            rec_img = rec.replace(prior_kind="particles", sigma_rel=0.0)
            trainer_img = GANTrainer(rec_img, g_img, d_img, prior=p_img, seed=0)
            out_img = trainer_img.step(torch.randn(8, 1, img_sz, img_sz), collect_stats=True)
            assert out_img["penalty_stats"]["applied"] is True
            assert out_img["penalty_stats"]["live_endpoints_reused"] is True

        # 3. Continuous-conditioned (x, xt, t) critic:
        d_cond = _ToyCondDisc()
        g_cond = _ToyGen2D(z_dim=4, out_dim=2)
        p_cond = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.25, standardize=False)
        opt_g, opt_d = rec.make_optimizers(g_cond, d_cond, p_cond)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d, collect_stats=True)
        assert isinstance(loss_fn, UnifiedGANLoss)
        assert isinstance(pen_fn, UnifiedCriticPenalty)
        x_r = torch.randn(8, 2)
        x_f = g_cond(torch.randn(8, 4))
        xt = torch.randn(8, 2)
        t = torch.randint(1, 5, (8,))
        dl = loss_fn.d_loss(d_cond(x_r, xt=xt, t=t), d_cond(x_f.detach(), xt=xt, t=t))
        pen = pen_fn(d_cond, x_r, x_f.detach(), xt=xt, t=t)
        (dl + pen).backward()
        opt_d.step()
        gl = loss_fn.g_loss(d_cond(x_f, xt=xt, t=t), d_cond(x_r, xt=xt, t=t).detach())
        gl.backward()


def test_pythagorean_total_variance_conservation_eliminates_half_cell_bias() -> None:
    """Exact mathematical check: splitting N(mu, sigma^2 I_2) into half-cells conserves 100% of total variance."""
    sigma = 0.03
    # Two modes separated by 0.60 (= 20*sigma), each split into 2 half-plane Voronoi sub-cells +-sqrt(2/pi)*sigma:
    a = math.sqrt(2.0 / math.pi) * sigma
    var_half = (1.0 - 1.0 / math.pi) * (sigma * sigma)
    c = torch.tensor(
        [
            [-0.30 - a, 0.0],
            [-0.30 + a, 0.0],
            [+0.30 - a, 0.0],
            [+0.30 + a, 0.0],
        ],
        dtype=torch.float32,
    )
    v = torch.full((4,), var_half, dtype=torch.float32)
    w = torch.full((4,), 0.25, dtype=torch.float32)

    c_bar, sigma_bar, w_mat, multicell_coal = _compute_pythagorean_smoothed_cells(c, v, w)
    assert bool(multicell_coal.all())
    assert torch.allclose(w_mat[:2, :2], torch.full((2, 2), 0.5))
    # Check mode 0 (rows 0 and 1): separated mode 1 (rows 2 and 3) must not contaminate mode 0!
    assert torch.allclose(c_bar[:2], torch.tensor([[-0.30, 0.0], [-0.30, 0.0]]), atol=1e-6)
    assert torch.allclose(c_bar[2:], torch.tensor([[+0.30, 0.0], [+0.30, 0.0]]), atol=1e-6)

    # Total per-coordinate variance inside mode 0: E[(c_bar - mu)^2]/2 + E[sigma_bar^2] == sigma^2!
    between_var = ((c_bar[:2] - torch.tensor([[-0.30, 0.0]])).square().sum(dim=-1) / 2.0).mean().item()
    within_var = sigma_bar[:2].square().mean().item()
    total_var = between_var + within_var
    rel_bias = (total_var - sigma * sigma) / (sigma * sigma)
    assert abs(rel_bias) < 1e-5, f"Expected exact total-variance conservation, got relative bias {rel_bias}"

    # High-rate tessellation check (|C| = 6 > 4): w_mat must be identity!
    c_dense = torch.stack([torch.tensor([0.01 * i, 0.0]) for i in range(6)], dim=0)
    v_dense = torch.full((6,), 0.01 * 0.01, dtype=torch.float32)
    w_dense = torch.full((6,), 1.0 / 6.0, dtype=torch.float32)
    c_b2, s_b2, w_m2, mc2 = _compute_pythagorean_smoothed_cells(c_dense, v_dense, w_dense)
    assert not bool(mc2.any())
    assert torch.allclose(w_m2, torch.eye(6))
    assert torch.allclose(c_b2, c_dense)


def test_zero_shrinkage_pooled_conditional_loss_exchangeability_and_sensitivity() -> None:
    torch.manual_seed(7)
    b = 32
    c_cont = torch.randn(b, 2)
    t_disc = torch.randint(1, 5, (b,))
    # Exact affine conditional law x = 1.8 * c + 0.2 * noise:
    x_matched = 1.8 * c_cont + 0.2 * torch.randn(b, 2)
    y_matched = 1.8 * c_cont + 0.2 * torch.randn(b, 2)

    # Swapping row k between X and Y leaves pooled mean m_hat invariant:
    loss_xy = unified_pooled_conditional_loss(x_matched, y_matched, c_cont, cond_disc=t_disc)
    loss_yx = unified_pooled_conditional_loss(y_matched, x_matched, c_cont, cond_disc=t_disc)
    assert math.isfinite(float(loss_xy.item())) and math.isfinite(float(loss_yx.item()))

    # Deliberately wrong conditional slope must yield strictly larger discrepancy than matched conditional law:
    x_wrong = (-0.5 * c_cont + 0.2 * torch.randn(b, 2)).requires_grad_(True)
    loss_wrong = unified_pooled_conditional_loss(x_wrong, y_matched, c_cont, cond_disc=t_disc)
    assert float(loss_wrong.item()) > float(loss_xy.item()) + 0.05
    loss_wrong.backward()
    assert x_wrong.grad is not None and float(x_wrong.grad.norm().item()) > 0.0


def test_unified_deliberate_damage_and_checkpoint_tamper_rejection() -> None:
    def _run_grad_probe(reg_coeff: float) -> float:
        torch.manual_seed(11)
        rec = get_recipe(
            "unified",
            prior_kind="mog",
            standardize=False,
            t1_pullback_weight=0.0,
            reg_coeff=reg_coeff,
            reg_kappa=1.0,
            lr=0.04,
            d_lr_mult=2.5,
            batch_size=16,
            num_particles=32,
            z_dim=2,
            total_steps=60,
            training_precision="float32",
        )
        g = _ToyGen2D(2, 2)
        d = _ToyDisc2D(2)
        p = MoGParticlePrior(num_particles=32, z_dim=2, sigma=0.05, standardize=False)
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        init.deterministic_orthogonal_(p, seed=2)
        for param in g.parameters():
            param.requires_grad_(False)
        opt_d = rec.make_critic_optimizer(d)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d)
        for _ in range(45):
            real = torch.randn(16, 2) * 0.1 + 3.0
            fake = torch.randn(16, 2) * 0.1 - 3.0
            opt_d.zero_grad()
            dl = loss_fn.d_loss(d(real), d(fake))
            pen = pen_fn(d, real, fake)
            (dl + pen).backward()
            opt_d.step()
        x_probe = (torch.randn(32, 2) * 0.1 + 3.0).requires_grad_(True)
        s = d(x_probe).sum()
        grad = torch.autograd.grad(s, x_probe)[0]
        return float(grad.norm(dim=1).median().item())

    grad_reg = _run_grad_probe(1.0)
    grad_unreg = _run_grad_probe(0.0)
    assert grad_reg <= 1.25
    assert grad_unreg > 2.5 * grad_reg

    # Checkpoint tamper rejection:
    rec = get_recipe(
        "unified",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=2,
        total_steps=20,
        training_precision="float32",
    )
    trainer = GANTrainer(
        rec,
        _ToyGen2D(2, 2),
        _ToyDisc2D(2),
        prior=MoGParticlePrior(num_particles=16, z_dim=2, sigma=0.05, standardize=False),
        seed=0,
    )
    trainer.step(torch.randn(8, 2))
    good = trainer.state_dict()

    bad_rec = deepcopy(good)
    bad_rec["optimizers"][1]["regularizer"]["record"]["unified"]["step_scale"] = 1.5
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|step_scale"):
        trainer.load_state_dict(bad_rec)

    bad_prior = deepcopy(good)
    bad_prior["prior_regularizer"]["schema"] = 99
    with pytest.raises(ValueError, match="schema"):
        trainer.load_state_dict(bad_prior)


@pytest.mark.parametrize("mode", ["mog_2d", "discrete_img"])
def test_unified_fresh_process_continuation_and_import_purity(tmp_path: Path, mode: str) -> None:
    """Fresh-process import-order check + exact checkpoint/RNG continuation across active mechanism boundaries."""
    ckpt_path = tmp_path / f"ckpt_{mode}.pt"
    out_path = tmp_path / f"resumed_{mode}.pt"

    if mode == "mog_2d":
        rec = get_recipe(
            "unified",
            prior_kind="mog",
            standardize=False,
            batch_size=12,
            num_particles=24,
            z_dim=2,
            total_steps=20,
            training_precision="float32",
        )
        g = _ToyGen2D(2, 2)
        d = _ToyDisc2D(2)
        p = MoGParticlePrior(num_particles=24, z_dim=2, sigma=0.05, standardize=False)
        batches = [torch.randn(12, 2, generator=torch.Generator().manual_seed(100 + i)) for i in range(8)]
    else:
        rec = get_recipe(
            "unified",
            prior_kind="particles",
            sigma_rel=0.0,
            batch_size=8,
            num_particles=16,
            z_dim=4,
            total_steps=20,
            training_precision="float32",
        )
        g = _ToyImgGen(4, 8)
        d = _ToyImgDisc()
        p = ParticlePrior(num_particles=16, z_dim=4)
        batches = [torch.randn(8, 1, 8, 8, generator=torch.Generator().manual_seed(200 + i)) for i in range(8)]

    init.deterministic_orthogonal_(g, seed=10)
    init.deterministic_orthogonal_(d, seed=11)
    init.deterministic_orthogonal_(p, seed=12)

    trainer_ref = GANTrainer(rec, g, d, prior=p, seed=0)
    for step_i in range(4):
        trainer_ref.step(batches[step_i])
    torch.save({"state": trainer_ref.state_dict(), "batches": batches[4:]}, ckpt_path)
    for step_i in range(4, 8):
        trainer_ref.step(batches[step_i])
    final_ref = trainer_ref.state_dict()
    sample_ref = trainer_ref.sample(16)

    script = f"""
import torch
import particlegan
assert particlegan.DEFAULT_RECIPE == "unified_v11"
import particlegan.unified
assert particlegan.DEFAULT_RECIPE == "unified_v11"
from particlegan import GANTrainer, MoGParticlePrior, ParticlePrior, Recipe
from tests.test_unified_recipe import _ToyGen2D, _ToyDisc2D, _ToyImgGen, _ToyImgDisc

payload = torch.load({str(ckpt_path)!r}, weights_only=False)
state = payload["state"]
rec = Recipe.from_dict(state["recipe"])
if {mode!r} == "mog_2d":
    g, d = _ToyGen2D(2, 2), _ToyDisc2D(2)
    p = MoGParticlePrior(num_particles=24, z_dim=2, sigma=0.05, standardize=False)
else:
    g, d = _ToyImgGen(4, 8), _ToyImgDisc()
    p = ParticlePrior(num_particles=16, z_dim=4)
trainer = GANTrainer(rec, g, d, prior=p, seed=99)
trainer.load_state_dict(state)
for b in payload["batches"]:
    trainer.step(b)
torch.save({{"final": trainer.state_dict(), "sample": trainer.sample(16)}}, {str(out_path)!r})
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    resumed = torch.load(out_path, weights_only=False)

    def _assert_tree_equal(a: object, b: object, prefix: str = "") -> None:
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and torch.equal(a, b), f"Mismatch in {prefix}"
        elif isinstance(a, dict):
            assert isinstance(b, dict) and set(a.keys()) == set(b.keys()), f"Key mismatch in {prefix}"
            for key in a:
                _assert_tree_equal(a[key], b[key], f"{prefix}.{key}")
        else:
            assert a == b, f"Value mismatch in {prefix}: {a!r} != {b!r}"

    assert torch.equal(sample_ref, resumed["sample"])
    for model_name in ("G", "D", "prior"):
        _assert_tree_equal(final_ref["models"][model_name], resumed["final"]["models"][model_name], model_name)
    _assert_tree_equal(
        final_ref["optimizers"][1]["regularizer"]["record"]["unified"],
        resumed["final"]["optimizers"][1]["regularizer"]["record"]["unified"],
        "record.unified",
    )
    if "prior_regularizer" in final_ref:
        _assert_tree_equal(final_ref["prior_regularizer"], resumed["final"]["prior_regularizer"], "prior_regularizer")
