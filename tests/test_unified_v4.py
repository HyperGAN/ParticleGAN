"""Unit, zero-SVD/zero-`autograd.grad` semantic-control, tamper-rejection, and fresh-process continuation tests for ``preset='unified_v4'``."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
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
    UnifiedGANLoss,
    UnifiedV4CriticPenalty,
    UnifiedV4EquilibratedOptimizer,
    UnifiedV4GANLoss,
    UnifiedV4PriorRegularizer,
    UnifiedV4StepRecord,
    get_recipe,
    init,
)
from particlegan.unified_v4 import (
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
    unified_v4_conditional_loss,
)


class _ToyGen2D(nn.Module):
    def __init__(self, z_dim: int = 2, out_dim: int = 2, tanh_out: bool = False) -> None:
        super().__init__()
        layers = [nn.Linear(z_dim, 24), nn.SiLU(), nn.Linear(24, out_dim)]
        if tanh_out:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _DeepGen16(nn.Module):
    def __init__(self, z_dim: int = 2, width: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        mods: list[nn.Module] = [nn.Linear(z_dim, width), nn.LeakyReLU(0.2)]
        for _ in range(14):
            mods.extend([nn.Linear(width, width), nn.LeakyReLU(0.2)])
        mods.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*mods)

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
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.fc(z)).reshape(z.shape[0], 8, 4, 4)
        h = nn.functional.silu(self.deconv(h))
        if self.size != 8:
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


def test_unified_v4_preset_and_frozen_baseline_invariants() -> None:
    assert DEFAULT_RECIPE == "axiom"
    assert len(fields(Recipe)) == 179

    rec = get_recipe("unified_v4")
    assert rec.critic_formulation == "unified_v4"
    assert rec.effective_critic_formulation == "unified_v4"
    assert rec.optimizer_family == "unified_equil"
    assert rec.prior_regularizer == "unified_v4"
    assert rec.constraint_geometry_mode == "none"
    assert rec.t1c_grad_deconflict == 0.0
    assert rec.rgdt_macro_slack_weight == 1.0
    assert rec.training_precision == "bf16_channels_last"

    roundtrip = Recipe.from_dict(rec.to_dict())
    assert roundtrip == rec

    # Frozen Round-2 `unified` preset remains untouched:
    rec_u3 = get_recipe("unified")
    assert rec_u3.critic_formulation == "unified"
    assert isinstance(rec_u3.make_loss(), UnifiedGANLoss)

    ctx = FormulationContext(
        recipe_preset="unified_v4",
        prior={"kind": "mog", "sigma": 0.025, "standardize": False, "learnable": True},
    )
    assert ctx.capabilities()["secant_critic"] is True

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe("ka2", critic_formulation="unified_v4")
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe("unified_v4", reg_arm="b_cap")


def test_unified_v4_strict_zero_svd_eigh_pinv_and_autograd_grad_everywhere() -> None:
    """Semantic control: default preset='unified_v4' (no constraint_geometry override!) calls 0 SVD/eigh/pinv/autograd.grad."""
    rec = get_recipe(
        "unified_v4",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=4,
        total_steps=20,
        training_precision="float32",
    )
    assert rec.constraint_geometry_mode == "none"

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("Forbidden higher-order AD or spectral decomposition called in unified_v4!")

    with (
        mock.patch("torch.autograd.grad", side_effect=_forbid),
        mock.patch("torch.linalg.svd", side_effect=_forbid),
        mock.patch("torch.linalg.svdvals", side_effect=_forbid),
        mock.patch("torch.linalg.eigh", side_effect=_forbid),
        mock.patch("torch.linalg.eigvalsh", side_effect=_forbid),
        mock.patch("torch.linalg.pinv", side_effect=_forbid),
    ):
        # 1. 1D scalar, 2D MoG pullback, and 170D simplex (64 particles exercises active_iso > 32 with 0 eigvalsh):
        rec_64 = rec.replace(num_particles=64)
        for dim in (1, 2, 170):
            g = _ToyGen2D(z_dim=4, out_dim=dim)
            d = _ToyDisc2D(in_dim=dim)
            p = MoGParticlePrior(num_particles=64, z_dim=4, sigma=0.05, standardize=False)
            trainer = GANTrainer(rec_64, g, d, prior=p, seed=0)
            assert isinstance(trainer.opt_g, UnifiedV4EquilibratedOptimizer)
            assert isinstance(trainer.opt_d, UnifiedV4EquilibratedOptimizer)
            assert isinstance(trainer.prior_regularizer, UnifiedV4PriorRegularizer)
            out = trainer.step(torch.randn(8, dim), collect_stats=True)
            assert out["penalty_stats"]["applied"] is True
            assert out["penalty_stats"]["phase"] == "unified_v4"

        # 2. 16-layer skip-free MLP (L=16 deep_gate active) & zero-sigma point-mass tanh rail escape:
        rec_deep = rec.replace(z_dim=2)
        trainer_deep = GANTrainer(
            rec_deep,
            _DeepGen16(z_dim=2, width=16, out_dim=2),
            _ToyDisc2D(in_dim=2),
            prior=MoGParticlePrior(num_particles=16, z_dim=2, sigma=0.05, standardize=False),
            seed=0,
        )
        trainer_deep.step(torch.randn(8, 2))

        rec_pt = rec_deep.replace(prior_kind="particles", sigma_rel=0.0)
        trainer_pt = GANTrainer(
            rec_pt,
            _ToyGen2D(z_dim=2, out_dim=2, tanh_out=True),
            _ToyDisc2D(in_dim=2),
            prior=ParticlePrior(num_particles=16, z_dim=2),
            seed=0,
        )
        trainer_pt.step(torch.tensor([[0.9, -0.9]]).expand(8, 2).clone())

        # 3. 2D connected zero-sigma arc manifold (simplicial + curve gate, 0 eigvalsh & 0 svdvals):
        trainer_arc = GANTrainer(
            rec_pt,
            _ToyGen2D(z_dim=2, out_dim=2),
            _ToyDisc2D(in_dim=2),
            prior=ParticlePrior(num_particles=16, z_dim=2),
            seed=0,
        )
        theta = torch.linspace(0.1, 3.0, 8)
        trainer_arc.step(torch.stack((torch.cos(theta), torch.sin(theta)), dim=1))

        # 4. Small 8x8 image & 64x64 ConvTranspose2d/Conv2d image:
        for img_sz in (8, 64):
            g_img = _ToyImgGen(z_dim=4, size=img_sz)
            d_img = _ToyImgDisc()
            p_img = ParticlePrior(num_particles=16, z_dim=4)
            rec_img = rec.replace(prior_kind="particles", sigma_rel=0.0)
            trainer_img = GANTrainer(rec_img, g_img, d_img, prior=p_img, seed=0)
            out_img = trainer_img.step(torch.randn(8, 1, img_sz, img_sz), collect_stats=True)
            assert out_img["penalty_stats"]["applied"] is True
            assert out_img["penalty_stats"]["live_endpoints_reused"] is True

        # 5. Continuous-conditioned (x, xt, t) critic + cross-batch sufficient statistics:
        d_cond = _ToyCondDisc()
        g_cond = _ToyGen2D(z_dim=4, out_dim=2)
        p_cond = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.25, standardize=False)
        opt_g, opt_d = rec.make_optimizers(g_cond, d_cond, p_cond)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d, collect_stats=True)
        assert isinstance(loss_fn, UnifiedV4GANLoss)
        assert isinstance(pen_fn, UnifiedV4CriticPenalty)
        for _ in range(3):
            x_r = torch.randn(12, 2)
            z_s, idx_s = p_cond.sample(12)
            opt_g.set_sampled_rows(p_cond.z, idx_s.flatten())
            x_f = g_cond(z_s)
            xt = torch.randn(12, 2)
            t = torch.randint(1, 5, (12,))
            opt_d.zero_grad()
            dl = loss_fn.d_loss(d_cond(x_r, xt=xt, t=t), d_cond(x_f.detach(), xt=xt, t=t))
            pen = pen_fn(d_cond, x_r, x_f.detach(), xt=xt, t=t)
            (dl + pen).backward()
            opt_d.step()
            opt_g.zero_grad()
            gl = loss_fn.g_loss(d_cond(x_f, xt=xt, t=t), d_cond(x_r, xt=xt, t=t).detach())
            gl.backward()
            opt_g.step()
        assert len(opt_d.record.cond_strata) >= 1


def test_quintic_polar_underflow_and_rank1_safety() -> None:
    torch.manual_seed(3)
    # 1e-37 underflow scale still produces finite, well-conditioned orthogonal factor:
    tiny = torch.randn(16, 8) * 1e-37
    u_tiny = quintic_orthogonalize_2d(tiny)
    assert bool(torch.isfinite(u_tiny).all())
    assert math.isclose(float(u_tiny.norm().item()), math.sqrt(8.0), rel_tol=1e-2)

    # Rank-1 [1, 32] and [32, 1] matrices use exact closed-form unit normalization:
    r1 = torch.randn(1, 32) * 1e-20
    u_r1 = quintic_orthogonalize_2d(r1)
    assert math.isclose(float(u_r1.norm().item()), 1.0, rel_tol=1e-5)

    # Zero and non-finite inputs safely return exact zeros:
    assert float(quintic_orthogonalize_2d(torch.zeros(8, 8)).abs().max().item()) == 0.0
    bad = torch.randn(3, 4, 4)
    bad[1, 0, 0] = float("nan")
    u_batch = quintic_orthogonalize_batched(bad)
    assert bool(torch.isfinite(u_batch).all())
    assert float(u_batch[1].abs().max().item()) == 0.0
    assert float(u_batch[0].norm().item()) > 0.0

    # Zero-SVD Tikhonov-smoothed Quintic polar matches exact SVD polar_factor(smoothing=0.0061):
    from particlegan.optim.dualnorm import polar_factor

    for shape, scale in (((2, 32), 0.02), ((32, 2), 0.002), ((16, 16), 0.05), ((1, 16), 0.004)):
        mat = torch.randn(shape) * scale
        u_svd = polar_factor(mat, smoothing=0.0061)
        u_q = quintic_orthogonalize_2d(mat, smoothing=0.0061)
        assert float((u_q - u_svd).norm().item()) / float(u_svd.norm().clamp_min(1e-6).item()) < 0.06
    mats_b = torch.randn(4, 8, 12) * 0.015
    u_qb = quintic_orthogonalize_batched(mats_b, smoothing=0.0061)
    for i in range(4):
        u_svdi = polar_factor(mats_b[i], smoothing=0.0061)
        assert float((u_qb[i] - u_svdi).norm().item()) / float(u_svdi.norm().clamp_min(1e-6).item()) < 0.06


def test_cross_batch_conditional_strata_and_unbiased_ustats() -> None:
    torch.manual_seed(17)
    rec = UnifiedV4StepRecord()
    # Pre-warm stratum 1 with 10 batches from y = 1.75 * c + 0.25 * eps:
    for _ in range(10):
        c_b = torch.randn(32, 2)
        y_b = 1.75 * c_b + 0.25 * torch.randn(32, 2)
        t_b = torch.ones(32, dtype=torch.long)
        x_b = (1.75 * c_b + 0.25 * torch.randn(32, 2)).requires_grad_(True)
        _ = unified_v4_conditional_loss(x_b, y_b, c_b, cond_disc=t_b, record=rec)

    assert 1 in rec.cond_strata and rec.cond_strata[1]["weight"] > 100.0

    c_test = torch.randn(32, 2)
    t_test = torch.ones(32, dtype=torch.long)
    y_test = 1.75 * c_test + 0.25 * torch.randn(32, 2)
    x_match = (1.75 * c_test + 0.25 * torch.randn(32, 2)).requires_grad_(True)
    x_wrong_slope = (-0.5 * c_test + 0.25 * torch.randn(32, 2)).requires_grad_(True)
    x_wrong_cov = (1.75 * c_test + 1.25 * torch.randn(32, 2)).requires_grad_(True)

    l_match = unified_v4_conditional_loss(x_match, y_test, c_test, cond_disc=t_test, record=rec)
    l_slope = unified_v4_conditional_loss(x_wrong_slope, y_test, c_test, cond_disc=t_test, record=rec)
    l_cov = unified_v4_conditional_loss(x_wrong_cov, y_test, c_test, cond_disc=t_test, record=rec)
    assert float(l_slope.item()) > float(l_match.item()) + 0.15
    assert float(l_cov.item()) > float(l_match.item()) + 0.05
    l_slope.backward()
    assert x_wrong_slope.grad is not None and float(x_wrong_slope.grad.norm().item()) > 0.0


def test_unified_v4_deliberate_damage_and_tamper_rejection() -> None:
    def _run_grad_probe(reg_coeff: float) -> float:
        torch.manual_seed(11)
        rec = get_recipe(
            "unified_v4",
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
    assert grad_reg <= 1.30
    assert grad_unreg > 2.5 * grad_reg

    rec = get_recipe(
        "unified_v4",
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
    bad_rec["optimizers"][1]["regularizer"]["record"]["unified_v4"]["step_scale"] = 1.5
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|step_scale"):
        trainer.load_state_dict(bad_rec)

    bad_opt = deepcopy(good)
    bad_opt["optimizers"][0]["unified_equil"]["last_step_gain"] = -1.0
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|last_step_gain"):
        trainer.load_state_dict(bad_opt)

    bad_prior = deepcopy(good)
    bad_prior["prior_regularizer"]["schema"] = 99
    with pytest.raises(ValueError, match="schema"):
        trainer.load_state_dict(bad_prior)

    bad_vpush = deepcopy(good)
    bad_vpush["prior_regularizer"]["v_push"] = -torch.ones_like(bad_vpush["prior_regularizer"]["v_push"])
    with pytest.raises(ValueError, match="v_push"):
        trainer.load_state_dict(bad_vpush)


@pytest.mark.parametrize("mode", ["mog_2d", "discrete_img"])
def test_unified_v4_fresh_process_continuation_and_import_purity(tmp_path: Path, mode: str) -> None:
    """Fresh-process import-order check + exact checkpoint/RNG continuation across active mechanism boundaries."""
    ckpt_path = tmp_path / f"ckpt_v4_{mode}.pt"
    out_path = tmp_path / f"resumed_v4_{mode}.pt"

    if mode == "mog_2d":
        rec = get_recipe(
            "unified_v4",
            prior_kind="mog",
            standardize=False,
            batch_size=12,
            num_particles=24,
            z_dim=2,
            total_steps=20,
            training_precision="float32",
        )
        g: nn.Module = _DeepGen16(2, 12, 2)
        d: nn.Module = _ToyDisc2D(2)
        p: nn.Module = MoGParticlePrior(num_particles=24, z_dim=2, sigma=0.05, standardize=False)
        batches = [torch.randn(12, 2, generator=torch.Generator().manual_seed(100 + i)) for i in range(8)]
    else:
        rec = get_recipe(
            "unified_v4",
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
assert particlegan.DEFAULT_RECIPE == "axiom"
import particlegan.unified_v4
assert particlegan.DEFAULT_RECIPE == "axiom"
from particlegan import GANTrainer, MoGParticlePrior, ParticlePrior, Recipe
from tests.test_unified_v4 import _DeepGen16, _ToyDisc2D, _ToyImgGen, _ToyImgDisc

payload = torch.load({str(ckpt_path)!r}, weights_only=False)
state = payload["state"]
rec = Recipe.from_dict(state["recipe"])
if {mode!r} == "mog_2d":
    g, d = _DeepGen16(2, 12, 2), _ToyDisc2D(2)
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
        final_ref["optimizers"][1]["regularizer"]["record"]["unified_v4"],
        resumed["final"]["optimizers"][1]["regularizer"]["record"]["unified_v4"],
        "record.unified_v4",
    )
    if "prior_regularizer" in final_ref:
        _assert_tree_equal(final_ref["prior_regularizer"], resumed["final"]["prior_regularizer"], "prior_regularizer")
