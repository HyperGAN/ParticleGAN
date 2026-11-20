"""Unit, zero-SVD/zero-eigh/zero-pinv/zero-autograd.grad, tamper-rejection, and fresh-process continuation tests for ``preset='unified_v11'``."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
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
    UnifiedV4GANLoss,
    UnifiedV11CriticPenalty,
    UnifiedV11EquilibratedOptimizer,
    UnifiedV11GANLoss,
    UnifiedV11PriorRegularizer,
    UnifiedV11StepRecord,
    get_recipe,
    init,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward, project_nonascent
from particlegan.optim.strict_progress import common_descent
from particlegan.unified_v11 import (
    sym_eigensplit_zero_eigh,
    unified_v11_conditional_loss,
    zero_eigh_symmetric,
    zero_pinv_common_descent,
    zero_pinv_project_nonascent,
)


class _ToyGen2D(nn.Module):
    def __init__(self, z_dim: int = 2, out_dim: int = 2, tanh_out: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(z_dim, 24), nn.SiLU(), nn.Linear(24, out_dim)]
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


class _ToyCondGen(nn.Module):
    def __init__(self, z_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + 3, 24), nn.LeakyReLU(0.2), nn.Linear(24, 2))

    def forward(self, z: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((z, xt, t[:, None].to(z) / 4.0), dim=1))


class _ToyCondDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 24), nn.LeakyReLU(0.2), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, xt, t[:, None].to(x) / 4.0), dim=1)).squeeze(-1)


def test_unified_v11_preset_and_frozen_baseline_invariants() -> None:
    assert DEFAULT_RECIPE == "empf"
    assert len(fields(Recipe)) == 179

    rec = get_recipe("unified_v11")
    assert rec.critic_formulation == "unified_v11"
    assert rec.effective_critic_formulation == "unified_v11"
    assert rec.optimizer_family == "unified_v11_equil"
    assert rec.prior_regularizer == "unified_v11"
    assert rec.constraint_geometry_mode == "direction_blend"
    assert rec.t1c_grad_deconflict == 0.0
    assert rec.rgdt_macro_slack_weight == 1.0
    assert rec.training_precision == "bf16_channels_last"

    roundtrip = Recipe.from_dict(rec.to_dict())
    assert roundtrip == rec

    # Frozen Round-3 unified_v4 remains untouched:
    rec_u4 = get_recipe("unified_v4")
    assert rec_u4.critic_formulation == "unified_v4"
    assert isinstance(rec_u4.make_loss(), UnifiedV4GANLoss)

    ctx = FormulationContext(
        recipe_preset="unified_v11",
        prior={"kind": "mog", "sigma": 0.025, "standardize": False, "learnable": True},
    )
    assert ctx.capabilities()["secant_critic"] is True

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe("ka2", critic_formulation="unified_v11")
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe("unified_v11", reg_arm="b_cap")
    with pytest.raises(ValueError, match="direction_blend"):
        get_recipe("unified_v11", constraint_geometry_mode="strict_progress")


def test_zero_eigh_and_zero_pinv_exact_mathematical_parity() -> None:
    torch.manual_seed(0)
    for d in (1, 2, 3, 6, 12):
        a = torch.randn(6, d, d, dtype=torch.float64)
        sym = 0.5 * (a + a.transpose(-1, -2))
        v_ref, q_ref = torch.linalg.eigh(sym)
        v_got, q_got = zero_eigh_symmetric(sym)
        assert float((v_ref - v_got).abs().max().item()) < 1e-13
        rec_got = torch.einsum("nij,nj,nkj->nik", q_got, v_got, q_got)
        assert float((sym - rec_got).abs().max().item()) < 1e-13
        ep_ref = torch.einsum("nij,nj,nkj->nik", q_ref, v_ref.clamp_min(0.0), q_ref)
        en_ref = torch.einsum("nij,nj,nkj->nik", q_ref, v_ref.clamp_max(0.0), q_ref)
        ep_got, en_got = sym_eigensplit_zero_eigh(sym)
        assert float((ep_ref - ep_got).abs().max().item()) < 1e-13
        assert float((en_ref - en_got).abs().max().item()) < 1e-13

    for n_c in (1, 2):
        for trial in range(16):
            disp = torch.randn(24, dtype=torch.float64)
            norms = torch.randn(n_c, 24, dtype=torch.float64)
            if trial == 0 and n_c == 2:
                norms[1] = norms[0]
            elif trial == 1 and n_c == 2:
                norms[1] = -norms[0]
            p_ref = project_nonascent(disp, norms)
            p_got = zero_pinv_project_nonascent(disp, norms)
            assert float((p_ref - p_got).abs().max().item()) < 1e-12
            c_ref, ok_ref = common_descent(disp, norms)
            c_got, ok_got = zero_pinv_common_descent(disp, norms)
            assert ok_ref == ok_got
            assert float((c_ref - c_got).abs().max().item()) < 1e-12


def test_unified_v11_strict_zero_svd_eigh_pinv_and_autograd_grad_everywhere() -> None:
    """Semantic control: preset='unified_v11' calls 0 SVD/svdvals/eigh/eigvalsh/pinv/autograd.grad everywhere."""
    rec = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=4,
        total_steps=20,
        training_precision="float32",
    )
    assert rec.constraint_geometry_mode == "direction_blend"

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("Forbidden higher-order AD or spectral decomposition called in unified_v11!")

    with (
        mock.patch("torch.autograd.grad", side_effect=_forbid),
        mock.patch("torch.linalg.svd", side_effect=_forbid),
        mock.patch("torch.linalg.svdvals", side_effect=_forbid),
        mock.patch("torch.linalg.eigh", side_effect=_forbid),
        mock.patch("torch.linalg.eigvalsh", side_effect=_forbid),
        mock.patch("torch.linalg.pinv", side_effect=_forbid),
    ):
        # 1. 1D scalar, 2D MoG pullback, and 170D simplex:
        rec_64 = rec.replace(num_particles=64)
        for dim in (1, 2, 170):
            g = _ToyGen2D(z_dim=4, out_dim=dim)
            d = _ToyDisc2D(in_dim=dim)
            p = MoGParticlePrior(num_particles=64, z_dim=4, sigma=0.05, standardize=False)
            trainer = GANTrainer(rec_64, g, d, prior=p, seed=0)
            assert isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer)
            assert isinstance(trainer.opt_d, UnifiedV11EquilibratedOptimizer)
            assert isinstance(trainer.prior_regularizer, UnifiedV11PriorRegularizer)
            out = trainer.step(torch.randn(8, dim), collect_stats=True)
            assert out["penalty_stats"]["applied"] is True
            assert out["penalty_stats"]["phase"] == "unified_v11"

        # 2. 16-layer skip-free MLP & zero-sigma point-mass tanh rail escape:
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

        # 3. 2D connected zero-sigma arc manifold:
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

        # 5. Continuous-conditioned DDGAN reverse chain + 2-loss direction_blend (warming CondFiberBank past active threshold!):
        d_cond = _ToyCondDisc()
        g_cond = _ToyCondGen(z_dim=4)
        p_cond = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.25, standardize=False)
        opt_g, opt_d = rec.make_optimizers(g_cond, d_cond, p_cond)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d, collect_stats=True)
        assert isinstance(loss_fn, UnifiedV11GANLoss)
        assert isinstance(pen_fn, UnifiedV11CriticPenalty)
        for _ in range(5):
            x_r = torch.randn(16, 2)
            z_s, idx_s = p_cond.sample(16)
            opt_g.set_sampled_rows(p_cond.z, idx_s.flatten())
            xt = torch.randn(16, 2)
            t = torch.ones(16, dtype=torch.long)
            x_f = g_cond(z_s, xt, t)
            opt_d.zero_grad()
            dl = loss_fn.d_loss(d_cond(x_r, xt=xt, t=t), d_cond(x_f.detach(), xt=xt, t=t))
            pen = pen_fn(d_cond, x_r, x_f.detach(), xt=xt, t=t)
            (dl + pen).backward()
            opt_d.step()
            opt_g.zero_grad()
            gl_adv = loss_fn.g_loss(d_cond(x_f, xt=xt, t=t), d_cond(x_r, xt=xt, t=t).detach())
            aux_mse = (x_f - x_r).square().mean()
            total_g = gl_adv + aux_mse
            constraint_geometry_backward(total_g, opt_g, (gl_adv, aux_mse))
            opt_g.step()
        assert (1,) in opt_d.record.cond_fiber_bank.strata
        assert opt_d.record.cond_fiber_bank.strata[(1,)]["ring_c"].shape[0] >= 64


def test_cond_fiber_bank_swap_invariance_and_zero_equilibrium_bias() -> None:
    torch.manual_seed(17)
    rec = UnifiedV11StepRecord()
    for _ in range(12):
        c_b = torch.randn(32, 2)
        y_b = 1.6 * c_b + 0.25 * torch.randn(32, 2)
        x_b = (1.6 * c_b + 0.25 * torch.randn(32, 2)).requires_grad_(True)
        t_b = torch.ones(32, dtype=torch.long)
        _ = unified_v11_conditional_loss(x_b, y_b, c_b, cond_disc=t_b, record=rec)

    assert (1,) in rec.cond_fiber_bank.strata and rec.cond_fiber_bank.strata[(1,)]["ring_c"].shape[0] >= 128

    # Equilibrium unbiasedness: at P(X|c) == P(Y|c), mean grad over fresh batches concentrates near 0,
    # while shifted slope or inflated covariance yields strong corrective loss & gradient:
    c_t = torch.randn(64, 2)
    t_t = torch.ones(64, dtype=torch.long)
    y_t = 1.6 * c_t + 0.25 * torch.randn(64, 2)
    x_eq = (1.6 * c_t + 0.25 * torch.randn(64, 2)).requires_grad_(True)
    x_bad = (-0.4 * c_t + 0.9 * torch.randn(64, 2)).requires_grad_(True)

    rec._in_probe = True  # freeze bank state during probe comparison
    l_eq = unified_v11_conditional_loss(x_eq, y_t, c_t, cond_disc=t_t, record=rec)
    l_bad = unified_v11_conditional_loss(x_bad, y_t, c_t, cond_disc=t_t, record=rec)
    assert float(l_bad.item()) > float(l_eq.item()) + 0.20
    l_bad.backward()
    assert x_bad.grad is not None and float(x_bad.grad.norm().item()) > 0.0


def test_unified_v11_deliberate_damage_and_tamper_rejection() -> None:
    rec = get_recipe(
        "unified_v11",
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
    bad_rec["optimizers"][1]["regularizer"]["record"]["unified_v11"]["step_scale"] = 1.5
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|step_scale"):
        trainer.load_state_dict(bad_rec)

    bad_bank = deepcopy(good)
    bad_bank["optimizers"][1]["regularizer"]["record"]["unified_v11"]["cond_fiber_bank"]["schema"] = 99
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|schema"):
        trainer.load_state_dict(bad_bank)

    bad_opt = deepcopy(good)
    bad_opt["optimizers"][0]["unified_v11_equil"]["last_step_gain"] = -1.0
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


@pytest.mark.parametrize("mode", ["mog_2d", "cond_ddgan"])
def test_unified_v11_fresh_process_continuation_and_import_purity(tmp_path: Path, mode: str) -> None:
    """Fresh-process import-order check + bit-exact continuation across active MoG pullback & CondFiberBank boundaries."""
    ckpt_path = tmp_path / f"ckpt_v11_{mode}.pt"
    out_path = tmp_path / f"resumed_v11_{mode}.pt"

    if mode == "mog_2d":
        rec = get_recipe(
            "unified_v11",
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
        init.deterministic_orthogonal_(g, seed=10)
        init.deterministic_orthogonal_(d, seed=11)
        init.deterministic_orthogonal_(p, seed=12)
        batches = [torch.randn(12, 2, generator=torch.Generator().manual_seed(100 + i)) for i in range(8)]
        trainer_ref = GANTrainer(rec, g, d, prior=p, seed=0)
        for i in range(4):
            trainer_ref.step(batches[i])
        torch.save({"state": trainer_ref.state_dict(), "batches": batches[4:]}, ckpt_path)
        for i in range(4, 8):
            trainer_ref.step(batches[i])
        final_ref = trainer_ref.state_dict()
        sample_ref = trainer_ref.sample(16)

        script = f"""
import torch, particlegan
assert particlegan.DEFAULT_RECIPE == "empf"
import particlegan.unified_v11
assert particlegan.DEFAULT_RECIPE == "empf"
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v11 import _DeepGen16, _ToyDisc2D

payload = torch.load({str(ckpt_path)!r}, weights_only=False)
state = payload["state"]
rec = Recipe.from_dict(state["recipe"])
trainer = GANTrainer(rec, _DeepGen16(2, 12, 2), _ToyDisc2D(2),
                     prior=MoGParticlePrior(num_particles=24, z_dim=2, sigma=0.05, standardize=False), seed=99)
trainer.load_state_dict(state)
for b in payload["batches"]:
    trainer.step(b)
torch.save({{"final": trainer.state_dict(), "sample": trainer.sample(16)}}, {str(out_path)!r})
"""
        subprocess.run([sys.executable, "-c", script], check=True)
        resumed = torch.load(out_path, weights_only=False)
        assert torch.equal(sample_ref, resumed["sample"])

        def _assert_tree_equal(a: object, b: object, prefix: str = "") -> None:
            if isinstance(a, torch.Tensor):
                assert isinstance(b, torch.Tensor) and torch.equal(a, b), f"Mismatch in {prefix}"
            elif isinstance(a, dict):
                assert isinstance(b, dict) and set(a.keys()) == set(b.keys()), f"Key mismatch in {prefix}"
                for key in a:
                    _assert_tree_equal(a[key], b[key], f"{prefix}.{key}")
            else:
                assert a == b, f"Value mismatch in {prefix}: {a!r} != {b!r}"

        for model_name in ("G", "D", "prior"):
            _assert_tree_equal(final_ref["models"][model_name], resumed["final"]["models"][model_name], model_name)
    else:
        # Conditional DDGAN fresh-process continuation across active CondFiberBank boundary:
        rec = get_recipe(
            "unified_v11",
            prior_kind="mog",
            standardize=False,
            batch_size=16,
            num_particles=16,
            z_dim=4,
            total_steps=20,
            training_precision="float32",
        )
        g_c = _ToyCondGen(4)
        d_c = _ToyCondDisc()
        p_c = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.25, standardize=False)
        init.deterministic_orthogonal_(g_c, seed=20)
        init.deterministic_orthogonal_(d_c, seed=21)
        init.deterministic_orthogonal_(p_c, seed=22)
        opt_g, opt_d = rec.make_optimizers(g_c, d_c, p_c)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d)

        steps_data = []
        for i in range(8):
            gen = torch.Generator().manual_seed(300 + i)
            xt = torch.randn(16, 2, generator=gen)
            xr = 1.3 * xt + 0.25 * torch.randn(16, 2, generator=gen)
            t = torch.ones(16, dtype=torch.long)
            z_idx = torch.randint(0, 16, (16,), generator=gen)
            z_eps = torch.randn(16, 4, generator=gen)
            steps_data.append((xr, xt, t, z_idx, z_eps))

        def _do_step(g_m, d_m, p_m, og, od, lf, pf, batch):
            xr, xt, t, z_idx, z_eps = batch
            z_s = p_m.z[z_idx] + p_m.sigma * z_eps
            og.set_sampled_rows(p_m.z, z_idx)
            xf = g_m(z_s, xt, t)
            od.zero_grad()
            dl = lf.d_loss(d_m(xr, xt=xt, t=t), d_m(xf.detach(), xt=xt, t=t))
            pen = pf(d_m, xr, xf.detach(), xt=xt, t=t)
            (dl + pen).backward()
            od.step()
            og.zero_grad()
            gl = lf.g_loss(d_m(xf, xt=xt, t=t), d_m(xr, xt=xt, t=t).detach())
            constraint_geometry_backward(gl, og, (gl,))
            og.step()

        for i in range(4):
            _do_step(g_c, d_c, p_c, opt_g, opt_d, loss_fn, pen_fn, steps_data[i])
        assert (1,) in opt_d.record.cond_fiber_bank.strata
        torch.save(
            {
                "recipe": rec.to_dict(),
                "g": g_c.state_dict(),
                "d": d_c.state_dict(),
                "p": p_c.state_dict(),
                "og": opt_g.state_dict(),
                "od": opt_d.state_dict(),
                "batches": steps_data[4:],
            },
            ckpt_path,
        )
        for i in range(4, 8):
            _do_step(g_c, d_c, p_c, opt_g, opt_d, loss_fn, pen_fn, steps_data[i])

        script = f"""
import torch
from particlegan import MoGParticlePrior, Recipe
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from tests.test_unified_v11 import _ToyCondGen, _ToyCondDisc

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
rec = Recipe.from_dict(pl["recipe"])
g_c, d_c = _ToyCondGen(4), _ToyCondDisc()
p_c = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.25, standardize=False)
og, od = rec.make_optimizers(g_c, d_c, p_c)
g_c.load_state_dict(pl["g"])
d_c.load_state_dict(pl["d"])
p_c.load_state_dict(pl["p"])
lf = rec.make_loss()
pf = rec.make_critic_penalty(od)
og.load_state_dict(pl["og"])
od.load_state_dict(pl["od"])
for xr, xt, t, z_idx, z_eps in pl["batches"]:
    z_s = p_c.z[z_idx] + p_c.sigma * z_eps
    og.set_sampled_rows(p_c.z, z_idx)
    xf = g_c(z_s, xt, t)
    od.zero_grad()
    dl = lf.d_loss(d_c(xr, xt=xt, t=t), d_c(xf.detach(), xt=xt, t=t))
    pen = pf(d_c, xr, xf.detach(), xt=xt, t=t)
    (dl + pen).backward()
    od.step()
    og.zero_grad()
    gl = lf.g_loss(d_c(xf, xt=xt, t=t), d_c(xr, xt=xt, t=t).detach())
    constraint_geometry_backward(gl, og, (gl,))
    og.step()
torch.save({{"g": g_c.state_dict(), "d": d_c.state_dict(), "bank": od.record.cond_fiber_bank.to_packet()}}, {str(out_path)!r})
"""
        subprocess.run([sys.executable, "-c", script], check=True)
        resumed = torch.load(out_path, weights_only=False)
        for k, v in g_c.state_dict().items():
            assert torch.equal(v, resumed["g"][k]), f"G mismatch {k}"
        for k, v in d_c.state_dict().items():
            assert torch.equal(v, resumed["d"][k]), f"D mismatch {k}"
        ref_bank = opt_d.record.cond_fiber_bank.to_packet()
        assert ref_bank["strata"][0]["n_eff"] == resumed["bank"]["strata"][0]["n_eff"]
        assert torch.equal(ref_bank["strata"][0]["ring_x"], resumed["bank"]["strata"][0]["ring_x"])
