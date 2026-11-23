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


class _ToyImgBNDisc(nn.Module):
    """4D image critic with a frozen eval() BatchNorm backbone + trainable BatchNorm head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.head_conv = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(8)
        self.head_fc = nn.Linear(8, 1)

    def train(self, mode: bool = True) -> "_ToyImgBNDisc":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = nn.functional.leaky_relu(self.head_bn(self.head_conv(h)), negative_slope=0.2)
        return self.head_fc(h.mean(dim=(2, 3))).squeeze(-1)


class _ToyImgBNGen(nn.Module):
    """4D 3x32x32 generator with BatchNorm2d to verify BN-safe Universal Law 4 secant probe."""

    def __init__(self, z_dim: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim, 8 * 8 * 8)
        self.bn = nn.BatchNorm2d(8)
        self.deconv1 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.bn(self.fc(z).reshape(z.shape[0], 8, 8, 8)))
        h = nn.functional.silu(self.deconv1(h))
        return torch.tanh(self.deconv2(h))


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
    assert DEFAULT_RECIPE == "unified_v11"
    assert get_recipe() == get_recipe("unified_v11") == get_recipe("gan")
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

    bad_ema_fg_missing = deepcopy(good)
    del bad_ema_fg_missing["optimizers"][0]["unified_v11_equil"]["ema_func_gain"]
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|metadata"):
        trainer.load_state_dict(bad_ema_fg_missing)

    for bad_val in (0.0, -0.2, 1.5, float("nan"), True):
        bad_ema_fg = deepcopy(good)
        bad_ema_fg["optimizers"][0]["unified_v11_equil"]["ema_func_gain"] = bad_val
        with pytest.raises(ValueError, match="invalid checkpoint optimizer state|ema_func_gain"):
            trainer.load_state_dict(bad_ema_fg)

        bad_g_fg = deepcopy(good)
        bad_g_fg["optimizers"][1]["regularizer"]["record"]["unified_v11"]["g_func_gain"] = bad_val
        with pytest.raises(ValueError, match="invalid checkpoint optimizer state|g_func_gain"):
            trainer.load_state_dict(bad_g_fg)

    bad_schema = deepcopy(good)
    bad_schema["optimizers"][0]["unified_v11_equil"]["schema"] = 99
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|metadata"):
        trainer.load_state_dict(bad_schema)

    bad_gfg_missing = deepcopy(good)
    del bad_gfg_missing["optimizers"][1]["regularizer"]["record"]["unified_v11"]["g_func_gain"]
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|record packet"):
        trainer.load_state_dict(bad_gfg_missing)

    first_state_key = next(iter(good["optimizers"][0]["state"]))
    for bad_peak in (torch.tensor(-1.0), torch.tensor([1.0]), torch.tensor(float("nan"))):
        bad_pr = deepcopy(good)
        bad_pr["optimizers"][0]["state"][first_state_key]["peak_rms"] = bad_peak
        with pytest.raises(ValueError, match="invalid checkpoint optimizer state|peak_rms"):
            trainer.load_state_dict(bad_pr)


def test_unified_v11_optimizer_group_split_asymmetry_and_conv1x1_invariance() -> None:
    """Semantic controls: zero L_count/deep_gate discontinuity, asymmetric player equilibrium, and 1x1 conv scaling."""
    torch.manual_seed(7)

    # 1. Splitting 16 matrices into 1 group vs 4 groups produces identical updates (zero L_count dependence):
    g_single = _DeepGen16(z_dim=2, width=16, out_dim=2)
    g_split = deepcopy(g_single)
    x_in = torch.randn(8, 2)
    g_single(x_in).square().mean().backward()
    g_split(x_in).square().mean().backward()

    opt_single = UnifiedV11EquilibratedOptimizer(
        [{"params": list(g_single.parameters()), "lr": 0.014, "role": "generator"}],
        direction_blend=False,
    )
    split_params = list(g_split.parameters())
    chunk = max(1, len(split_params) // 4)
    opt_split = UnifiedV11EquilibratedOptimizer(
        [
            {"params": split_params[i : i + chunk], "lr": 0.014, "role": "generator"}
            for i in range(0, len(split_params), chunk)
        ],
        direction_blend=False,
    )
    opt_single.step()
    opt_split.step()
    for p_a, p_b in zip(g_single.parameters(), g_split.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-6)

    # 2. 1x1 Conv2d (out=16, in=64) via public make_critic_optimizer uses exact RMS-to-RMS sqrt(out_ch / in_ch) = 0.5:
    conv1x1 = nn.Conv2d(64, 16, kernel_size=1, bias=False)
    nn.init.zeros_(conv1x1.weight)
    rec_c = get_recipe("unified_v11", lr=0.02, d_lr_mult=1.0, training_precision="float32")
    opt_c = rec_c.make_critic_optimizer(conv1x1)
    opt_c.smoothing = 0.0
    conv1x1.weight.grad = torch.randn_like(conv1x1.weight)
    opt_c.step()
    # Orthogonalized 16x64 matrix has Frobenius norm sqrt(16) = 4.0; scaled by lr * sqrt(16/64) = 0.02 * 0.5 = 0.01:
    expected_frob = 0.02 * (16.0 / 64.0) ** 0.5 * (16.0**0.5)
    got_frob = float(conv1x1.weight.detach().norm().item())
    assert abs(got_frob - expected_frob) / expected_frob < 0.05

    # 3. _ema_func_gain attenuates on function-space overshoot and recovers smoothly upward (no one-way ratchet):
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
    g_m = _ToyGen2D(2, 2)
    d_m = _ToyDisc2D(2)
    p_m = MoGParticlePrior(num_particles=16, z_dim=2, sigma=0.05, standardize=False)
    trainer = GANTrainer(rec, g_m, d_m, prior=p_m, seed=0)
    # Seed a depressed ema_func_gain as after transient deep stiffness, then step well-conditioned 2-layer G:
    trainer.opt_g._ema_func_gain = 0.25
    before_fg = trainer.opt_g._ema_func_gain
    for _ in range(12):
        trainer.step(torch.randn(8, 2))
    assert trainer.opt_g._ema_func_gain > before_fg


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
assert particlegan.DEFAULT_RECIPE == "unified_v11"
import particlegan.unified_v11
assert particlegan.DEFAULT_RECIPE == "unified_v11"
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


def test_unified_v11_universal_4d_laws_semantic_controls_and_continuation(tmp_path: Path) -> None:
    """Semantic controls + fresh-process continuation for Universal Laws 2, 3, and 4 on 4D 3x32x32 images."""
    torch.manual_seed(0)
    rec = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=8,
        total_steps=20,
        training_precision="float32",
    )
    g = _ToyImgBNGen(z_dim=8)
    d = _ToyImgBNDisc()
    p = MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.1, standardize=False)
    init.deterministic_orthogonal_(g, seed=30)
    init.deterministic_orthogonal_(d.head_conv, seed=31)
    init.deterministic_orthogonal_(d.head_fc, seed=32)
    init.deterministic_orthogonal_(p, seed=33)

    frozen_mean_before = d.backbone[1].running_mean.detach().clone()
    frozen_var_before = d.backbone[1].running_var.detach().clone()

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("Forbidden autograd.grad or spectral decomposition called in 4D unified_v11!")

    trainer = GANTrainer(rec, g, d, prior=p, seed=0)
    batches = [torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(500 + i)) for i in range(6)]
    with (
        mock.patch("torch.autograd.grad", side_effect=_forbid),
        mock.patch("torch.linalg.svd", side_effect=_forbid),
        mock.patch("torch.linalg.svdvals", side_effect=_forbid),
        mock.patch("torch.linalg.eigh", side_effect=_forbid),
        mock.patch("torch.linalg.eigvalsh", side_effect=_forbid),
        mock.patch("torch.linalg.pinv", side_effect=_forbid),
    ):
        out = trainer.step(batches[0], collect_stats=True)

    # Law 2: live endpoints reused, frozen BN strictly unchanged & eval, trainable BN restored after penalty():
    assert out["penalty_stats"]["applied"] is True
    assert out["penalty_stats"]["live_endpoints_reused"] is True
    assert d.backbone[1].training is False
    assert torch.equal(d.backbone[1].running_mean, frozen_mean_before)
    assert torch.equal(d.backbone[1].running_var, frozen_var_before)
    d.train()
    assert d.head_bn.training is True and d.backbone[1].training is False
    _ = trainer.penalty(d, batches[0], batches[1])
    assert d.head_bn.training is True and d.backbone[1].training is False
    assert torch.equal(d.backbone[1].running_mean, frozen_mean_before)
    assert torch.equal(d.backbone[1].running_var, frozen_var_before)

    # Law 3: margin & data-scale EMAs are live on 4D images and scalar logit coupling depends on real_logits:
    rec_d = trainer.opt_d.record
    assert rec_d.margin_initialized is True
    assert rec_d.ema_data_scale > 5.0
    loss_fn = rec.make_loss()
    r_log = torch.tensor([1.2, 0.8, -0.3, 0.5], requires_grad=True)
    f_log = torch.tensor([-0.4, 0.2, 0.1, -0.7], requires_grad=True)
    g1 = loss_fn.g_loss(f_log, r_log)
    g2 = loss_fn.g_loss(f_log, r_log + 2.0)
    assert float(g2.item()) > float(g1.item()) + 0.25
    ls_loss = rec.replace(loss="least_squares").make_loss()
    assert torch.allclose(ls_loss.g_loss(f_log, r_log), 0.5 * (f_log - 1.0).square().mean())

    # Law 4: generator forward hook captures element-budgeted 4D slice with BN-eval y_pre (zero shift at zero step):
    z_probe = torch.randn(8, 8)
    _ = g(z_probe)
    assert rec_d._g_probe_cache is not None
    _, p_args, p_kwargs, y_pre = rec_d._g_probe_cache
    assert tuple(y_pre.shape) == (2, 3, 32, 32)
    from particlegan._unified_v4.optimizer import _eval_probe_forward
    y_post_zero_step = _eval_probe_forward(g, p_args, p_kwargs, rec_d)
    assert y_post_zero_step is not None
    assert float((y_post_zero_step - y_pre).abs().max().item()) == 0.0
    rec_d._g_probe_cache = None

    # Exact fresh-process continuation across 4D image + BN + Universal Laws 2/3/4:
    for i in range(1, 3):
        trainer.step(batches[i])
    ckpt_path = tmp_path / "ckpt_v11_img4d.pt"
    out_path = tmp_path / "resumed_v11_img4d.pt"
    torch.save({"state": trainer.state_dict(), "batches": batches[3:]}, ckpt_path)
    for i in range(3, 6):
        trainer.step(batches[i])
    sample_ref = trainer.sample(4)
    d_ref = trainer.D.state_dict()

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v11 import _ToyImgBNGen, _ToyImgBNDisc

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(rec, _ToyImgBNGen(8), _ToyImgBNDisc(),
                prior=MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.1, standardize=False), seed=77)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save({{"sample": tr.sample(4), "d": tr.D.state_dict()}}, {str(out_path)!r})
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    resumed = torch.load(out_path, weights_only=False)
    assert torch.equal(sample_ref, resumed["sample"])
    for k, v in d_ref.items():
        assert torch.equal(v, resumed["d"][k]), f"4D D mismatch {k}"


def test_unified_v11_composite_generator_probe_record_replay_and_rng_preservation() -> None:
    """Public ``record_unified_v11_generator_probe`` supports multi-input composite graphs,
    nested conditioning dicts, RNG/mode/BN preservation, and direct consumption reporting."""
    import random
    import numpy as np
    from particlegan.unified_v11 import (
        UnifiedV11EquilibratedOptimizer,
        UnifiedV11StepRecord,
        bind_unified_v11_record,
        record_unified_v11_generator_probe,
    )

    class _CompositeGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU())
            self.branch_a = nn.Sequential(nn.Linear(8, 4), nn.Dropout(0.25))
            self.branch_b = nn.Linear(8 + 2, 4)
            self.replay_calls = 0

        def forward(self, z: torch.Tensor, *, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            self.replay_calls += 1
            # Deliberately draw from torch/python/numpy RNGs to verify probe replay restores them.
            _ = torch.randn(1)
            _ = random.random()
            _ = float(np.random.rand())
            h = self.shared(z)
            a = self.branch_a(h)
            b = self.branch_b(torch.cat([h, batch["cond"]], dim=-1))
            return a + b

    torch.manual_seed(11)
    gen = _CompositeGen()
    rec = UnifiedV11StepRecord()
    opt = UnifiedV11EquilibratedOptimizer([{"params": list(gen.parameters()), "role": "generator", "lr": 0.01}])
    bind_unified_v11_record(rec, opt)
    opt._unified_v11_record = rec
    opt.set_generator_probe_parameters(gen.parameters())

    z = torch.randn(8, 4)
    cond = torch.randn(8, 2)
    gen.train()
    y = gen(z, batch={"cond": cond, "ignored": torch.randn(8, 1)})
    bn = gen.shared[1]
    bn_mean_after_train = bn.running_mean.detach().clone()
    bn_var_after_train = bn.running_var.detach().clone()
    bn_tracked_after_train = bn.num_batches_tracked.detach().clone()

    torch_rng_before = torch.get_rng_state()
    py_rng_before = random.getstate()
    np_rng_before = np.random.get_state()

    captured = record_unified_v11_generator_probe(gen, (z,), {"batch": {"cond": cond}}, y, rec)
    assert captured is True
    assert rec._g_probe_cache is not None
    # Capture with BN/Dropout ran eval probe without mutating RNGs, modes, or BN stats:
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert random.getstate() == py_rng_before
    assert np.array_equal(np.random.get_state()[1], np_rng_before[1])
    assert gen.training is True and bn.training is True
    assert torch.equal(bn.running_mean, bn_mean_after_train)
    assert torch.equal(bn.running_var, bn_var_after_train)
    assert torch.equal(bn.num_batches_tracked, bn_tracked_after_train)

    # Identical weights replay reproduces y_pre exactly:
    g_mod, p_args, p_kwargs, y_pre = rec._g_probe_cache
    from particlegan._unified_v11.critic import _eval_v11_probe_forward
    y_same = _eval_v11_probe_forward(g_mod, p_args, p_kwargs, rec)
    assert y_same is not None
    assert torch.allclose(y_same, y_pre, atol=1e-6, rtol=1e-6)

    # Perturbing either upstream branch changes replayed output:
    with torch.no_grad():
        gen.branch_a[0].weight.add_(0.25)
    y_pert_a = _eval_v11_probe_forward(g_mod, p_args, p_kwargs, rec)
    assert y_pert_a is not None and not torch.allclose(y_pert_a, y_pre, atol=1e-5)
    with torch.no_grad():
        gen.branch_a[0].weight.sub_(0.25)
        gen.branch_b.weight.add_(0.25)
    y_pert_b = _eval_v11_probe_forward(g_mod, p_args, p_kwargs, rec)
    assert y_pert_b is not None and not torch.allclose(y_pert_b, y_pre, atol=1e-5)
    with torch.no_grad():
        gen.branch_b.weight.sub_(0.25)

    # Optimizer step consumes the probe directly and restores RNGs/modes/BN stats:
    y_step = gen(z, batch={"cond": cond, "ignored": torch.randn(8, 1)})
    bn_mean_after_train = bn.running_mean.detach().clone()
    bn_var_after_train = bn.running_var.detach().clone()
    bn_tracked_after_train = bn.num_batches_tracked.detach().clone()
    assert record_unified_v11_generator_probe(gen, (z,), {"batch": {"cond": cond}}, y_step, rec) is True
    torch_rng_before = torch.get_rng_state()
    py_rng_before = random.getstate()
    np_rng_before = np.random.get_state()
    calls_before_step = gen.replay_calls
    opt.zero_grad(set_to_none=True)
    y_step.square().mean().backward()
    opt.step()
    assert opt.last_probe_consumed is True
    assert opt.probe_consumptions == 1
    assert rec._g_probe_cache is None
    assert gen.replay_calls == calls_before_step + 1
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert random.getstate() == py_rng_before
    assert np.array_equal(np.random.get_state()[1], np_rng_before[1])
    assert gen.training is True and bn.training is True
    assert torch.equal(bn.running_mean, bn_mean_after_train)
    assert torch.equal(bn.running_var, bn_var_after_train)
    assert torch.equal(bn.num_batches_tracked, bn_tracked_after_train)

    # Checkpoint schema remains unchanged:
    sd = opt.state_dict()
    opt2 = UnifiedV11EquilibratedOptimizer([{"params": list(gen.parameters()), "role": "generator", "lr": 0.01}])
    opt2.load_state_dict(sd)
    assert opt2.last_probe_consumed is False
    assert opt2.probe_consumptions == 0


def test_unified_v11_t1c_confident_critic_non_silencing_and_high_d_vector_bypass() -> None:
    """Defect 1: confident critics never self-silence under T1CouplingLoss, and 2D vectors with D > 1024 take high-D bypass."""
    recipe = get_recipe("unified_v11", training_precision="float32")
    loss_fn = recipe.make_loss()
    t1c = loss_fn.t1c

    torch.manual_seed(0)
    b_sz, dim = 256, 1170
    x_real = torch.randn(b_sz, dim) * 0.35
    x_fake = torch.randn(b_sz, dim) * 0.15
    grad_norms = []
    for margin in (0.0, 0.5, 1.0, 1.8, 3.0):
        xf = x_fake.detach().clone().requires_grad_(True)
        d_real = torch.full((b_sz, 1), +0.5 * margin)
        d_fake = torch.full((b_sz, 1), -0.5 * margin, requires_grad=True)
        t1c.ell_initialized.fill_(False)
        t1c._ell_init_py = False
        lg, _ = t1c(xf, x_real, d_fake, d_real, record=None, allow_ell_update=True)
        lg.backward()
        assert d_fake.grad is not None and xf.grad is not None
        g_norm = float(d_fake.grad.norm().item())
        grad_norms.append(g_norm)
        assert g_norm > 1e-3, f"Confident critic silenced at margin={margin}: ||dL/dd_fake||={g_norm:.3e}"
        if margin >= 1.0:
            assert g_norm > 2e-3, f"Confident critic gradient too weak at margin={margin}: {g_norm:.3e}"
    # Stronger critic margin produces stronger (monotonically non-decreasing) generator critic gradient:
    assert grad_norms[-1] > grad_norms[0]

    # High-D 2D vector (dim > 1024) bypasses Euclidean t1c in UnifiedV11 critic hook and loss, while keeping scale EMAs live:
    d_net = _ToyDisc2D(in_dim=dim)
    opt_d = recipe.make_critic_optimizer(d_net)
    pen_fn = recipe.make_critic_penalty(opt_d)
    loss_hd = recipe.make_loss()
    loss_hd.record = opt_d.record
    xr_hd = torch.randn(32, dim)
    xf_hd = torch.randn(32, dim, requires_grad=True)
    dr_hd = d_net(xr_hd)
    df_hd = d_net(xf_hd)
    assert getattr(dr_hd, "_t1c_bypassed", False) is True
    assert getattr(df_hd, "_t1c_bypassed", False) is True
    with mock.patch.object(loss_hd.t1c, "forward", side_effect=AssertionError("High-D 2D vector must bypass Euclidean t1c")):
        gl_hd = loss_hd.g_loss(df_hd, dr_hd.detach())
    gl_hd.backward()
    assert xf_hd.grad is not None and float(xf_hd.grad.norm().item()) > 0.0
    _ = pen_fn(d_net, xr_hd, xf_hd.detach())
    assert opt_d.record.ema_data_scale > 5.0


def test_unified_v11_thermal_no_premature_freeze_and_cooling_interactions() -> None:
    """Defect 2: observe_t1c_thermal never freezes on collapsed masked high-D inputs or active critic margin, and cooling interactions are clean."""
    recipe = get_recipe("unified_v11", training_precision="float32")
    t1c = recipe.make_loss().t1c

    # 1. Repro 2: 100%-collapsed value channels with matched sparse binary masks must stay hot (never freeze to 1e-6):
    torch.manual_seed(0)
    b_sz, d_val, p_obs = 256, 585, 0.045
    rec_col = UnifiedV11StepRecord()
    s_col = 1.0
    for _ in range(400):
        m_r = (torch.rand(b_sz, d_val) < p_obs).float()
        m_f = (torch.rand(b_sz, d_val) < p_obs).float()
        x_r = torch.cat([torch.randn(b_sz, d_val) * 0.35 * m_r, m_r], dim=1)
        x_f = torch.cat([torch.zeros(b_sz, d_val), m_f], dim=1)
        ell = t1c._batch_ell_quantile_tensor(x_r)
        r_ed, r_mu_sq, r_spread, r_q90, r_hmax = t1c._split_two_sample_residuals(x_f, x_r, ell=ell)
        s_col = rec_col.observe_t1c_thermal(r_ed, r_mu_sq, r_spread, r_q90, r_hmax, tau=0.084)
    assert rec_col.ema_residual > 0.084
    assert s_col == 1.0

    # 2. True equilibrium (P_fake == P_real) still cleanly cools to <= 1e-4:
    torch.manual_seed(0)
    rec_eq = UnifiedV11StepRecord()
    s_eq = 1.0
    for _ in range(250):
        xr_2 = torch.randn(64, 2)
        xf_2 = torch.randn(64, 2)
        ell_2 = t1c._batch_ell_quantile_tensor(xr_2)
        vals_2 = t1c._split_two_sample_residuals(xf_2, xr_2, ell=ell_2)
        s_eq = rec_eq.observe_t1c_thermal(*vals_2, tau=0.084)
    assert s_eq <= 1e-4

    # 3. Clear critic separation (ema_margin) prevents premature cooling even if Euclidean residuals are zero:
    rec_m = UnifiedV11StepRecord()
    rec_m.observe_margin_and_scale(torch.full((16, 1), 1.0), torch.full((16, 1), -1.0))
    s_m = rec_m.observe_t1c_thermal(0.0, 0.0, 0.0, 0.0, 0.0, tau=0.084)
    assert s_m == 1.0

    # 4. Compounding interaction 1: UnifiedV11PriorRegularizer thermally scales z_dim > 4 when cooled:
    torch.manual_seed(0)
    rec_th = UnifiedV11StepRecord()
    prior_reg = UnifiedV11PriorRegularizer(weight=1.0)
    prior_reg.thermal_source = rec_th
    rows_hot = torch.randn(128, 32, requires_grad=True)
    rec_th._thermal_scale = 1.0
    prior_reg(rows_hot).backward()
    g_hot = float(rows_hot.grad.norm().item())
    rows_cold = rows_hot.detach().clone().requires_grad_(True)
    rec_th._thermal_scale = 1e-5
    prior_reg(rows_cold).backward()
    g_cold = float(rows_cold.grad.norm().item())
    assert g_hot > 0.0 and g_cold / g_hot < 1e-4

    # 5. Compounding interaction 2: gamma_gauge logit anchoring scales as sqrt(s_therm) (stays active vs s_therm):
    d_bias = nn.Sequential(nn.Linear(4, 1))
    with torch.no_grad():
        d_bias[0].weight.zero_()
        d_bias[0].bias.fill_(3.0)
    opt_db = recipe.make_critic_optimizer(d_bias)
    pen_db = recipe.make_critic_penalty(opt_db)
    xr4, xf4 = torch.randn(16, 4), torch.randn(16, 4)
    opt_db.record._thermal_scale = 1.0
    opt_db.zero_grad()
    pen_db(d_bias, xr4, xf4).backward()
    g_bias_hot = abs(float(d_bias[0].bias.grad.item()))
    opt_db.record._thermal_scale = 1e-4
    opt_db.zero_grad()
    pen_db(d_bias, xr4, xf4).backward()
    g_bias_cold = abs(float(d_bias[0].bias.grad.item()))
    # sqrt(1e-4) = 1e-2, so ratio is ~1e-2 (100x stronger than linear 1e-4):
    assert g_bias_hot > 0.0 and 0.005 < (g_bias_cold / g_bias_hot) < 0.02


def test_unified_v11_symmetric_high_d_secant_probe_and_no_ema_func_gain_lockout() -> None:
    """Defect 3: symmetric element-budgeted G/D secant probing for dim > 32 and no ema_func_gain lockout."""
    from particlegan._unified_v11.critic import _attach_unified_v11_critic_hook, record_unified_v11_generator_probe

    torch.manual_seed(0)
    dim = 1170
    d_net = nn.Linear(dim, 1)
    g_net = nn.Linear(32, dim)
    rec = UnifiedV11StepRecord()
    _attach_unified_v11_critic_hook(d_net, rec)
    _ = d_net(torch.randn(32, dim))
    z = torch.randn(32, 32)
    captured_g = record_unified_v11_generator_probe(g_net, (z,), {}, g_net(z), rec)
    assert len(rec._d_probe_list) == 1
    assert rec._d_probe_list[0][2].shape[0] == min(32, max(2, min(16, 8192 // dim)))
    assert captured_g is True and rec._g_probe_cache is not None

    # Critic optimizer consumes the high-D secant probe:
    rec_cfg = get_recipe("unified_v11", lr=0.01, training_precision="float32")
    opt_g, opt_d = rec_cfg.make_optimizers(g_net, d_net)
    x_in = torch.randn(32, dim)
    opt_d.zero_grad()
    d_net(x_in).square().mean().backward()
    opt_d.step()
    assert opt_d.last_probe_consumed is True and opt_d.probe_consumptions == 1

    # No ema_func_gain lockout when pre_gain_applied shrinks delta_rel below 0.012:
    g_small = nn.Linear(4, 4)
    d_small = _ToyDisc2D(4)
    p_small = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05, standardize=False)
    rec_l = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        lr=0.01,
        batch_size=8,
        num_particles=16,
        z_dim=4,
        total_steps=20,
        training_precision="float32",
    )
    tr = GANTrainer(rec_l, g_small, d_small, prior=p_small, seed=0)
    tr.opt_g._ema_func_gain = 0.12
    before_fg = tr.opt_g._ema_func_gain
    for _ in range(15):
        tr.step(torch.randn(8, 4))
    assert tr.opt_g._ema_func_gain > before_fg, "ema_func_gain remained locked out when depressed!"

    # Frozen equilibrium (s_therm = 1e-6, tiny delta_rel < 1e-4) must NOT falsely bounce ema_func_gain upward:
    tr.opt_g._ema_func_gain = 0.25
    tr.opt_d.record._thermal_scale = 1e-6
    tr.opt_d.record.ema_push_res = 0.0
    g_probe_in = torch.randn(8, 4)
    y_pre = g_small(g_probe_in)
    assert record_unified_v11_generator_probe(g_small, (g_probe_in,), {}, y_pre, tr.opt_d.record) is True
    tr.opt_g.zero_grad()
    (1e-7 * y_pre.square().mean()).backward()
    tr.opt_g.bind_protected_from_grads()
    tr.opt_g.set_sampled_rows(p_small.z, torch.arange(8))
    tr.opt_g.step()
    assert tr.opt_g._ema_func_gain <= 0.25 + 1e-9


def test_unified_v11_e2e_synthetic_masked_tabular_mixture_fixture() -> None:
    """End-to-end synthetic sparse masked-tabular mixture ([x * m, m], D=1170) under unified_v11."""
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    b_sz, d_val, p_obs = 256, 585, 0.05
    dim = 2 * d_val
    rng = torch.Generator(device=dev).manual_seed(42)
    mode_centers = (torch.randint(0, 2, (4, d_val), generator=rng, device=dev).float() * 2.0 - 1.0) * 0.65

    def _sample_real(bs: int) -> torch.Tensor:
        m = (torch.rand(bs, d_val, generator=rng, device=dev) < p_obs).float()
        c = torch.randint(0, 4, (bs,), generator=rng, device=dev)
        x_raw = mode_centers[c] + 0.15 * torch.randn(bs, d_val, generator=rng, device=dev)
        return torch.cat([x_raw * m, m], dim=1)

    class _TabGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(32, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dim))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            out = self.net(z)
            m_prob = torch.sigmoid(out[:, d_val:])
            return torch.cat([out[:, :d_val] * m_prob, m_prob], dim=1)

    class _TabDisc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    rec = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=b_sz,
        num_particles=400,
        z_dim=32,
        total_steps=250,
        training_precision="float32",
    )
    g = _TabGen().to(dev)
    d = _TabDisc().to(dev)
    p = MoGParticlePrior(num_particles=400, z_dim=32, sigma=0.05, standardize=False, device=dev)
    init.deterministic_orthogonal_(g, seed=1)
    init.deterministic_orthogonal_(d, seed=2)
    init.deterministic_orthogonal_(p, seed=3)
    trainer = GANTrainer(rec, g, d, prior=p, seed=0)

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("Forbidden autograd.grad or spectral decomposition called in high-D masked-tabular fixture!")

    n_steps = 220 if dev.type == "cuda" else 60
    with (
        mock.patch("torch.autograd.grad", side_effect=_forbid),
        mock.patch("torch.linalg.svd", side_effect=_forbid),
        mock.patch("torch.linalg.svdvals", side_effect=_forbid),
        mock.patch("torch.linalg.eigh", side_effect=_forbid),
        mock.patch("torch.linalg.eigvalsh", side_effect=_forbid),
        mock.patch("torch.linalg.pinv", side_effect=_forbid),
    ):
        for _ in range(n_steps):
            trainer.step(_sample_real(b_sz))

    assert trainer.opt_d.record._thermal_scale >= 0.5
    assert trainer.opt_d.probe_consumptions == n_steps
    assert trainer.opt_g.probe_consumptions == n_steps

    # Verify active critic gradient through g_loss on freshly generated batch:
    z_chk, _ = p.sample(64)
    xf_chk = g(z_chk)
    xr_chk = _sample_real(64)
    df_chk = d(xf_chk)
    df_chk.retain_grad()
    dr_chk = d(xr_chk).detach()
    trainer.loss.g_loss(df_chk, dr_chk).backward()
    assert df_chk.grad is not None and float(df_chk.grad.norm().item()) > 1e-2

    with torch.no_grad():
        sf = trainer.sample(512)
        sr = _sample_real(512)
        xv_f, m_f = sf[:, :d_val], sf[:, d_val:]
        xv_r, m_r = sr[:, :d_val], sr[:, d_val:]
        val_rms_f = float((xv_f.square().sum(dim=1) / m_f.sum(dim=1).clamp_min(1.0)).sqrt().mean().item())
        val_rms_r = float((xv_r.square().sum(dim=1) / m_r.sum(dim=1).clamp_min(1.0)).sqrt().mean().item())
        assert val_rms_f > 0.45 * val_rms_r, f"Value spread collapsed: {val_rms_f:.3f} vs {val_rms_r:.3f}"
        scores = (xv_f @ mode_centers.T) / m_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        asg = scores.argmax(dim=1)
        populated_modes = sum(int((asg == k).sum().item()) >= 25 for k in range(4))
        assert populated_modes >= 3, f"Mode collapse on masked tabular fixture: counts={[int((asg == k).sum().item()) for k in range(4)]}"
        if dev.type == "cuda":
            assert 0.015 < float(m_f.mean().item()) < 0.15


def test_unified_v11_regression_fixes_tikhonov_peak_rms_stiff_fg_and_gauge_deadband() -> None:
    """Targeted regression controls for the 3-part unified_v11 regression fix:
    1. peak_rms Tikhonov smoothing preserves full s_base=0.0061 on normal critic-scale gradients,
       never crushes tiny high-D gradients (1e-4), freezes decay under thermal cooling (s_therm < 0.5),
       and preserves prior s_cap row smoothing.
    2. _ema_func_gain does not bounce upward on stiff networks (free_ratio < 1.5) while still
       recovering smoothly on well-conditioned networks (free_ratio > 1.5) without probe lockout.
    3. s_gauge logit-drift deadband cools well-anchored equilibrium critics linearly at s_therm
       while retaining sqrt(s_therm) anchoring when |mean_logit| drifts far from zero.
    """
    torch.manual_seed(0)
    recipe = get_recipe("unified_v11", training_precision="float32")

    # 1a. Normal critic-scale gradient (sv_rms ~ 0.025) preserves full s_base=0.0061 (not 0.03*peak_rms),
    #     while genuinely tiny gradient (sv_rms ~ 1e-4) is capped by 0.50*peak_rms so it is not crushed:
    w_norm = nn.Linear(64, 64, bias=False)
    w_tiny = nn.Linear(64, 64, bias=False)
    nn.init.zeros_(w_norm.weight)
    nn.init.zeros_(w_tiny.weight)
    opt_norm = UnifiedV11EquilibratedOptimizer(
        [{"params": [w_norm.weight], "role": "critic", "lr": 0.02}], smoothing=0.0061
    )
    opt_tiny = UnifiedV11EquilibratedOptimizer(
        [{"params": [w_tiny.weight], "role": "critic", "lr": 0.02}], smoothing=0.0061
    )
    base_dir = torch.randn(64, 64)
    base_dir = base_dir / (base_dir.norm() / 8.0)  # cur_sv_rms == 1.0
    # Warm up peak_rms at normal critic scale (sv_rms=0.025), then step at cooled equilibrium (sv_rms=0.002):
    w_norm.weight.grad = base_dir * 0.025
    opt_norm.step()
    nn.init.zeros_(w_norm.weight)
    w_norm.weight.grad = base_dir * 0.002
    opt_norm.step()
    # With s_eff == s_base == 0.0061, damping is ~0.002 / hypot(0.002, 0.0061) ~= 0.31 (not ~0.94 under 0.03*peak):
    frob_norm_eq = float(w_norm.weight.detach().norm().item())
    full_step_frob = 0.02 * 8.0
    assert frob_norm_eq < 0.55 * full_step_frob, f"Equilibrium critic under-damped: {frob_norm_eq:.4f} vs {full_step_frob:.4f}"

    # Tiny high-D gradient (sv_rms=1e-4 << s_base=0.0061) must NOT be crushed (0.50*peak_rms -> ~0.89x full step):
    w_tiny.weight.grad = base_dir * 1e-4
    opt_tiny.step()
    frob_tiny = float(w_tiny.weight.detach().norm().item())
    assert frob_tiny > 0.70 * full_step_frob, f"Tiny high-D gradient crushed: {frob_tiny:.4f} vs {full_step_frob:.4f}"

    # 1b. peak_rms freezes decay during thermal cooling (s_therm < 0.5):
    rec_cool = UnifiedV11StepRecord()
    rec_cool._thermal_scale = 1e-4
    opt_norm._unified_v11_record = rec_cool
    peak_before = float(opt_norm.state[w_norm.weight]["peak_rms"].item())
    w_norm.weight.grad = base_dir * 1e-6
    opt_norm.step()
    peak_after = float(opt_norm.state[w_norm.weight]["peak_rms"].item())
    assert peak_after == peak_before, f"peak_rms decayed during thermal cooling: {peak_before} -> {peak_after}"

    # 1c. Prior row smoothing preserves equilibrium damping (0.30*peak_r reaches s_cap=0.0061 on normal prior gradients):
    z_prior = nn.Parameter(torch.zeros(128, 2))
    opt_p = UnifiedV11EquilibratedOptimizer(
        [{"params": [z_prior], "role": "prior", "algorithm": "unified_prior", "lr": 0.04}],
        smoothing=0.0061,
    )
    opt_p.set_sampled_rows(z_prior, torch.arange(128))
    z_prior.grad = torch.full((128, 2), 0.02)
    opt_p.step()
    with torch.no_grad():
        z_prior.zero_()
    opt_p.set_sampled_rows(z_prior, torch.arange(128))
    z_prior.grad = torch.full((128, 2), 1e-4)
    opt_p.step()
    # With peak_r=0.0283, 0.30*peak_r=0.0085 > s_cap=0.0061 -> s_p_row=0.0061 -> row_disp ~= 9.3e-4:
    row_disp = float(z_prior.detach().norm(dim=-1).mean().item())
    assert row_disp < 0.005, f"Prior Tikhonov row smoothing collapsed: row_disp={row_disp:.5f}"

    # 2. Stiff generator (free_ratio < 1.5, e.g. delta_rel ~= 0.5 * delta_g_max at pre_gain=0.18 -> free_ratio ~= 0.36)
    #    must NOT bounce _ema_func_gain upward out of stiff_gate:
    from particlegan._unified_v11.critic import record_unified_v11_generator_probe

    g_stiff = nn.Linear(4, 4, bias=False)
    nn.init.eye_(g_stiff.weight)
    rec_st = UnifiedV11StepRecord()
    rec_st.ema_data_scale = 1.0
    rec_st.ema_push_res = 0.0
    opt_st = UnifiedV11EquilibratedOptimizer(
        [{"params": list(g_stiff.parameters()), "role": "generator", "lr": 0.014}],
        smoothing=0.0,
    )
    opt_st._unified_v11_record = rec_st
    opt_st._ema_func_gain = 0.18
    opt_st.set_generator_probe_parameters(g_stiff.parameters())
    z_in = torch.ones(8, 4) * 2.5
    y_st = g_stiff(z_in)
    assert record_unified_v11_generator_probe(g_stiff, (z_in,), {}, y_st, rec_st) is True
    opt_st.zero_grad()
    y_st.mean().backward()
    opt_st.step()
    # At lr=0.014, pre_gain=0.18, delta_rel is ~0.015 (< delta_g_max=0.045, so alpha_gain==1.0),
    # giving free_ratio ~= (0.045 / 0.015) * 0.18 ~= 0.54 < 1.5 -> ema_func_gain must stay <= 0.18:
    assert opt_st._ema_func_gain <= 0.18 + 1e-9, f"Stiff network bounced ema_func_gain upward: {opt_st._ema_func_gain}"

    # 3. Well-anchored equilibrium critic (|mean_logit| = 0.05 <= 0.5) cools pen_gauge linearly at s_therm
    #    rather than 100x stronger sqrt(s_therm):
    d_eq = nn.Sequential(nn.Linear(4, 1))
    with torch.no_grad():
        d_eq[0].weight.zero_()
        d_eq[0].bias.fill_(0.05)
    opt_deq = recipe.make_critic_optimizer(d_eq)
    pen_deq = recipe.make_critic_penalty(opt_deq)
    xr4, xf4 = torch.randn(16, 4), torch.randn(16, 4)
    opt_deq.record._thermal_scale = 1.0
    opt_deq.zero_grad()
    pen_deq(d_eq, xr4, xf4).backward()
    g_eq_hot = abs(float(d_eq[0].bias.grad.item()))
    opt_deq.record._thermal_scale = 1e-4
    opt_deq.zero_grad()
    pen_deq(d_eq, xr4, xf4).backward()
    g_eq_cold = abs(float(d_eq[0].bias.grad.item()))
    # Linear s_therm=1e-4 scaling (ratio ~1e-4, strictly < 5e-4, NOT 1e-2 from sqrt(s_therm)):
    assert g_eq_hot > 0.0 and (g_eq_cold / g_eq_hot) < 5e-4, f"Anchored gauge over-amplified: ratio={g_eq_cold / g_eq_hot:.3e}"


def test_unified_v11_zerosync_primitives_parity_and_cuda_sync_budget() -> None:
    """Phase 3 zero-sync exact mathematical parity (CPU+CUDA) and strict CUDA sync-count regression guards."""
    from particlegan._stats import median_offdiag
    from particlegan._unified_v4.loss import _principal_axis_power
    from particlegan._unified_v11.loss import _principal_axis_power_zerosync
    from particlegan._unified_v11.prior import _median_offdiag_zerosync

    torch.manual_seed(0)
    # 1. Exact bit-for-bit parity of zero-sync off-diagonal median & power-iteration eigenvector:
    for m in (2, 3, 4, 7, 16, 64):
        mat = torch.randn(m, m).abs()
        assert torch.equal(_median_offdiag_zerosync(mat), median_offdiag(mat))
    for d in (2, 3, 8, 32):
        a = torch.randn(d, d)
        g_mat = a.T @ a
        assert torch.equal(_principal_axis_power_zerosync(g_mat, iters=6), _principal_axis_power(g_mat, iters=6))

    # 2. Exact parity of zero-sync SBNP fallback in UnifiedV11PriorRegularizer.forward:
    for use_ref in (False, True):
        for z_dim in (2, 8, 32):
            prior_ref = ParticlePrior(num_particles=64, z_dim=z_dim) if use_ref else None
            if prior_ref is not None:
                from particlegan.moment_tether import capture_initial_moments
                capture_initial_moments(prior_ref)
            reg_v11 = UnifiedV11PriorRegularizer(reference=prior_ref, weight=1.0)
            rec = UnifiedV11StepRecord()
            reg_v11.thermal_source = rec
            for s_th in (1.0, 0.25, 1e-5):
                rec._thermal_scale = s_th
                rows_a = torch.randn(64, z_dim, requires_grad=True)
                rows_b = rows_a.detach().clone().requires_grad_(True)
                val_zs = reg_v11._sbnp_fallback_zerosync(rows_a)
                val_ref = reg_v11.fallback(rows_b)
                assert torch.allclose(val_zs, val_ref, atol=1e-6, rtol=1e-6)
                val_zs.backward()
                val_ref.backward()
                assert rows_a.grad is not None and rows_b.grad is not None
                assert torch.allclose(rows_a.grad, rows_b.grad, atol=1e-6, rtol=1e-6)

    # 3. Exact parity of lazy GPU EMA record flushing & checkpoint/deepcopy round-trip:
    rec_em = UnifiedV11StepRecord()
    rl = torch.tensor([1.5, 0.5, 2.0, -0.5])
    fl = torch.tensor([-0.2, 0.1, 0.3, -0.6])
    xr = torch.randn(8, 4)
    xf = torch.randn(8, 4)
    rec_em.observe_margin_and_scale(rl, fl, xr, xf)
    rec_em.observed_steps = 1
    rec_em.observe_margin_and_scale(rl * 0.5, fl * 0.5, xr * 1.2, xf * 0.8)
    m1 = max(-50.0, min(50.0, float(rl.mean().item()) - float(fl.mean().item())))
    m2 = max(-50.0, min(50.0, float((rl * 0.5).mean().item()) - float((fl * 0.5).mean().item())))
    assert rec_em.margin_initialized is True
    assert abs(rec_em.ema_margin - (0.90 * m1 + 0.10 * m2)) < 1e-12
    dup = deepcopy(rec_em)
    assert dup.ema_margin == rec_em.ema_margin
    assert dup.ema_data_scale == rec_em.ema_data_scale
    assert dup.ema_push_res == rec_em.ema_push_res

    # 4. Zero-sync duplicate-row deduplication in UnifiedV11EquilibratedOptimizer prior group:
    z_a = nn.Parameter(torch.randn(32, 4))
    z_b = nn.Parameter(z_a.detach().clone())
    grad_z = torch.randn(32, 4)
    z_a.grad = grad_z.clone()
    z_b.grad = grad_z.clone()
    opt_a = UnifiedV11EquilibratedOptimizer(
        [{"params": [z_a], "role": "prior", "algorithm": "unified_prior", "lr": 0.04, "betas": (0.8, 0.99)}],
        smoothing=0.0061,
    )
    opt_b = UnifiedV11EquilibratedOptimizer(
        [{"params": [z_b], "role": "prior", "algorithm": "unified_prior", "lr": 0.04, "betas": (0.8, 0.99)}],
        smoothing=0.0061,
    )
    dup_idx = torch.tensor([5, 1, 5, 12, 1, 0, 12, 5], dtype=torch.long)
    uniq_idx = torch.unique(dup_idx, sorted=True)
    opt_a.set_sampled_rows(z_a, dup_idx)
    opt_b.set_sampled_rows(z_b, uniq_idx)
    assert torch.equal(opt_a.sampled_rows_for(z_a), uniq_idx)
    opt_a.step()
    opt_b.step()
    assert torch.allclose(z_a, z_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(opt_a.state[z_a]["m_prior"], opt_b.state[z_b]["m_prior"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(opt_a.state[z_a]["peak_rms"], opt_b.state[z_b]["peak_rms"], atol=1e-6, rtol=1e-6)

    # 5. Zero-sync quantizer maintenance (uncovered replacement + dead-slot Gersho split) invariants:
    reg_q = UnifiedV11PriorRegularizer(max_prototypes=16)
    reg_q.update_quantizer(torch.randn(32, 2))
    # Force dead slot + Gersho split + uncovered point:
    reg_q.proto_w[0] = 0.0
    reg_q.proto_w = reg_q.proto_w / reg_q.proto_w.sum()
    reg_q.update_quantizer(torch.randn(32, 2))
    far_batch = torch.randn(32, 2)
    far_batch[0] = torch.tensor([100.0, -100.0])
    reg_q.update_quantizer(far_batch)
    sd_q = reg_q.state_dict()
    reg_q2 = UnifiedV11PriorRegularizer(max_prototypes=16)
    reg_q2.load_state_dict(sd_q)
    assert torch.equal(reg_q2.proto_c, reg_q.proto_c)
    assert torch.equal(reg_q2.v_push, reg_q.v_push)

    # 6. CUDA sync-count regression guards (pullback <= 14 syncs/step vs ~115 before; 4D image <= 6 vs ~60 before):
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        rec_pb = get_recipe(
            "unified_v11",
            prior_kind="mog",
            standardize=False,
            batch_size=32,
            num_particles=64,
            z_dim=2,
            total_steps=40,
            training_precision="float32",
        )
        tr_pb = GANTrainer(
            rec_pb,
            _ToyGen2D(2, 2).to(dev),
            _ToyDisc2D(2).to(dev),
            prior=MoGParticlePrior(num_particles=64, z_dim=2, sigma=0.05, standardize=False, device=dev),
            seed=0,
        )
        for _ in range(12):
            tr_pb.step(torch.randn(32, 2, device=dev))
        torch.cuda.synchronize(dev)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof_pb:
            for _ in range(5):
                tr_pb.step(torch.randn(32, 2, device=dev))
        torch.cuda.synchronize(dev)
        pb_syncs = sum(
            int(e.count) for e in prof_pb.key_averages() if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize")
        ) / 5.0
        assert pb_syncs <= 16.0, f"Pullback CUDA sync regression: {pb_syncs:.1f} syncs/step (budget <= 16)"

        rec_img = get_recipe(
            "unified_v11",
            prior_kind="particles",
            sigma_rel=0.0,
            batch_size=8,
            num_particles=32,
            z_dim=8,
            total_steps=30,
            training_precision="float32",
        )
        tr_img = GANTrainer(
            rec_img,
            _ToyImgBNGen(8).to(dev),
            _ToyImgBNDisc().to(dev),
            prior=ParticlePrior(num_particles=32, z_dim=8, device=dev),
            seed=0,
        )
        for _ in range(6):
            tr_img.step(torch.randn(8, 3, 32, 32, device=dev))
        torch.cuda.synchronize(dev)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof_img:
            for _ in range(5):
                tr_img.step(torch.randn(8, 3, 32, 32, device=dev))
        torch.cuda.synchronize(dev)
        img_syncs = sum(
            int(e.count) for e in prof_img.key_averages() if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize")
        ) / 5.0
        assert img_syncs <= 6.0, f"4D image CUDA sync regression: {img_syncs:.1f} syncs/step (budget <= 6)"

    # 8. Subclass compatibility: legacy subclasses without _SUPPORTS_ZEROSYNC_V4_TENSORS keep deduplicated
    #    _sampled_rows and Python-float _origin_norm / _v4_g_scale / _v4_wakeup_boost, while base & opt-in keep tensors:
    class _LegacySubclassOpt(UnifiedV11EquilibratedOptimizer):
        pass

    class _OptInSubclassOpt(UnifiedV11EquilibratedOptimizer):
        _SUPPORTS_ZEROSYNC_V4_TENSORS = True

    p_tab = nn.Parameter(torch.randn(16, 2))
    raw_dups = torch.tensor([3, 1, 3, 5, 1, 5], dtype=torch.long)
    legacy_opt = _LegacySubclassOpt([{"params": [p_tab], "role": "prior", "algorithm": "unified_prior"}], lr=0.01)
    legacy_opt.set_sampled_rows(p_tab, raw_dups)
    assert torch.equal(legacy_opt._sampled_rows[p_tab], torch.tensor([1, 3, 5], dtype=torch.long))
    legacy_opt.bind_origin_sample(torch.tensor([[0.01, 0.0], [0.02, 0.0]]))
    assert isinstance(legacy_opt._origin_norm, float) and legacy_opt._origin_norm_t is None
    legacy_opt.bind_v4_modulation(g_scale=torch.tensor(0.25), wakeup_boost=torch.tensor(2.0), prior_scale=1.0)
    assert type(legacy_opt._v4_g_scale) is float and abs(legacy_opt._v4_g_scale - 0.25) < 1e-6
    assert type(legacy_opt._v4_wakeup_boost) is float and abs(legacy_opt._v4_wakeup_boost - 2.0) < 1e-6

    optin_opt = _OptInSubclassOpt([{"params": [p_tab], "role": "prior", "algorithm": "unified_prior"}], lr=0.01)
    optin_opt.set_sampled_rows(p_tab, raw_dups)
    assert len(optin_opt._sampled_rows[p_tab]) == len(raw_dups)
    assert torch.equal(optin_opt.sampled_rows_for(p_tab), torch.tensor([1, 3, 5], dtype=torch.long))
    optin_opt.bind_origin_sample(torch.tensor([[0.01, 0.0], [0.02, 0.0]]))
    assert optin_opt._origin_norm is None and isinstance(optin_opt._origin_norm_t, torch.Tensor)
    optin_opt.bind_v4_modulation(g_scale=torch.tensor(0.25), wakeup_boost=torch.tensor(2.0), prior_scale=1.0)
    assert isinstance(optin_opt._v4_g_scale, torch.Tensor) and isinstance(optin_opt._v4_wakeup_boost, torch.Tensor)


class _ResBNTrunkGen(nn.Module):
    """Depth-invariant branch-normalized residual MLP with constant ~2x coherent displacement."""

    def __init__(self, z_dim: int = 8, width: int = 32, out_dim: int = 8, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.stem = nn.Linear(z_dim, width)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"ffn_in": nn.Linear(width, width), "ffn_out": nn.Linear(width, width)}) for _ in range(depth)]
        )
        self.head = nn.Linear(width, out_dim)

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.leaky_relu(self.stem(z), negative_slope=0.2)
        scale = 0.5 / float(max(1, self.depth))
        for blk in self.blocks:
            b = blk["ffn_out"](nn.functional.leaky_relu(blk["ffn_in"](self._rms_norm(h)), negative_slope=0.2))
            h = h + scale * self._rms_norm(b)
        return self.head(h)


def test_unified_v11_law4_two_player_coupling_and_stiff_gate_controls(tmp_path: Path) -> None:
    """Semantic controls + fresh-process continuation for coupled Law 4 gains and overshoot/drop stiff_gate."""
    torch.manual_seed(0)

    # 1. Normalized deep residual trunk at ema_func_gain >= 0.18 (e.g. ~0.30-0.40) keeps stiff_gate == 0.0
    #    (no asymmetric beta1 momentum allocated on 2D weights), while severe overshoot (<= 0.11) activates 1.0:
    w_g = nn.Linear(16, 16, bias=False)
    opt_g = UnifiedV11EquilibratedOptimizer(
        [{"params": [w_g.weight], "role": "generator", "lr": 0.014}],
        smoothing=0.0061,
    )
    opt_g._ema_func_gain = 0.25
    assert opt_g.current_stiff_gate() == 0.0
    w_g.weight.grad = torch.randn_like(w_g.weight)
    opt_g.step()
    assert "m_unit" not in opt_g.state[w_g.weight], "False stiff_gate allocated momentum on normalized gain!"

    # Severe absolute overshoot (ema_fg <= 0.11, e.g. 16L MLP) activates stiff_gate == 1.0:
    opt_g._ema_func_gain = 0.10
    assert opt_g.current_stiff_gate() == 1.0
    w_g.weight.grad = torch.randn_like(w_g.weight)
    opt_g.step()
    assert "m_unit" in opt_g.state[w_g.weight], "Severe overshoot failed to engage stiff_gate momentum!"

    # 2. Bandpass positive-margin critic coupling scales D step by rec._g_func_gain (down to 0.12)
    #    only in the on-manifold critic-dominance band (0.30 <= ema_margin <= 1.10), releasing to 1.0
    #    at equilibrium (ema_margin <= 0.08), hostile margin (< 0), and off-manifold collapse (>= 1.40):
    grad_d = torch.randn(16, 16)

    def _critic_step_norm(g_fg: float, ema_margin: float, margin_init: bool) -> float:
        d_lin = nn.Linear(16, 16, bias=False)
        nn.init.zeros_(d_lin.weight)
        rec_d = UnifiedV11StepRecord()
        rec_d._g_func_gain = g_fg
        rec_d.ema_margin = ema_margin
        rec_d.margin_initialized = margin_init
        opt_d = UnifiedV11EquilibratedOptimizer(
            [{"params": [d_lin.weight], "role": "critic", "lr": 0.02}],
            smoothing=0.0,
        )
        opt_d.critic = d_lin
        opt_d._unified_v11_record = rec_d
        d_lin.weight.grad = grad_d.clone()
        opt_d.step()
        return float(d_lin.weight.detach().norm().item())

    n_full = _critic_step_norm(g_fg=1.0, ema_margin=0.5, margin_init=True)
    n_coupled = _critic_step_norm(g_fg=0.45, ema_margin=0.5, margin_init=True)
    n_deep_coupled = _critic_step_norm(g_fg=0.14, ema_margin=0.5, margin_init=True)
    n_floored = _critic_step_norm(g_fg=0.05, ema_margin=0.5, margin_init=True)
    n_eq_rel = _critic_step_norm(g_fg=0.20, ema_margin=0.01, margin_init=True)
    n_hostile = _critic_step_norm(g_fg=0.45, ema_margin=-2.0, margin_init=True)
    n_collapse_rel = _critic_step_norm(g_fg=0.30, ema_margin=2.0, margin_init=True)
    assert abs(n_coupled / n_full - 0.45) < 1e-4
    assert abs(n_deep_coupled / n_full - 0.14) < 1e-4
    assert abs(n_floored / n_full - 0.12) < 1e-4
    assert abs(n_eq_rel / n_full - 1.00) < 1e-4
    assert abs(n_hostile / n_full - 1.00) < 1e-4
    assert abs(n_collapse_rel / n_full - 1.00) < 1e-4

    # 3. Pathologically stiff unnormalized 16-layer MLP (at O(1) Kaiming-like scale) engages stiff_gate protection
    #    and blends published rec._g_func_gain smoothly toward 1.0 so D never starves:
    rec_16 = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=16,
        num_particles=32,
        z_dim=2,
        total_steps=40,
        training_precision="float32",
    )
    g16 = _DeepGen16(2, 32, 2)
    d16 = _ToyDisc2D(2)
    p16 = MoGParticlePrior(num_particles=32, z_dim=2, sigma=0.05, standardize=False)
    init.deterministic_orthogonal_(g16, seed=0)
    with torch.no_grad():
        for m in g16.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(2.6)
    init.deterministic_orthogonal_(d16, seed=1)
    init.deterministic_orthogonal_(p16, seed=2)
    tr16 = GANTrainer(rec_16, g16, d16, prior=p16, seed=0)
    peak_stiff_16 = 0.0
    for i in range(20):
        tr16.step(torch.randn(16, 2, generator=torch.Generator().manual_seed(700 + i)))
        peak_stiff_16 = max(peak_stiff_16, tr16.opt_g.current_stiff_gate())
    assert peak_stiff_16 > 0.5, f"Pathological 16L MLP failed to engage stiff_gate: peak={peak_stiff_16:.3f}"
    g_stiff_16 = tr16.opt_g.current_stiff_gate()
    expected_g_fg = (1.0 - g_stiff_16) * tr16.opt_g._ema_func_gain + g_stiff_16
    assert abs(tr16.opt_d.record._g_func_gain - expected_g_fg) < 1e-6

    # 4. Fresh-process bit-exact continuation across active g_func_gain coupling boundary:
    rec_bn = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=16,
        num_particles=32,
        z_dim=8,
        total_steps=40,
        training_precision="float32",
    )
    g_bn = _ResBNTrunkGen(z_dim=8, width=32, out_dim=8, depth=4)
    d_bn = _ToyDisc2D(8)
    p_bn = MoGParticlePrior(num_particles=32, z_dim=8, sigma=0.1, standardize=False)
    init.deterministic_orthogonal_(g_bn, seed=40)
    init.deterministic_orthogonal_(d_bn, seed=41)
    init.deterministic_orthogonal_(p_bn, seed=42)
    tr_bn = GANTrainer(rec_bn, g_bn, d_bn, prior=p_bn, seed=0)
    batches = [torch.randn(16, 8, generator=torch.Generator().manual_seed(800 + i)) for i in range(12)]
    for i in range(6):
        tr_bn.step(batches[i])
    # Ensure coupling state is genuinely non-trivial at the checkpoint boundary:
    tr_bn.opt_g._ema_func_gain = 0.42
    tr_bn.opt_d.record._g_func_gain = 0.42
    ckpt_path = tmp_path / "ckpt_v11_law4_couple.pt"
    out_path = tmp_path / "resumed_v11_law4_couple.pt"
    torch.save({"state": tr_bn.state_dict(), "batches": batches[6:]}, ckpt_path)
    for i in range(6, 12):
        tr_bn.step(batches[i])
    ref_state = tr_bn.state_dict()
    ref_g_fg = tr_bn.opt_d.record._g_func_gain

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v11 import _ResBNTrunkGen, _ToyDisc2D

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(
    rec,
    _ResBNTrunkGen(z_dim=8, width=32, out_dim=8, depth=4),
    _ToyDisc2D(8),
    prior=MoGParticlePrior(num_particles=32, z_dim=8, sigma=0.1, standardize=False),
    seed=123,
)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save(
    {{
        "models": tr.state_dict()["models"],
        "g_fg": tr.opt_d.record._g_func_gain,
    }},
    {str(out_path)!r},
)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    resumed = torch.load(out_path, weights_only=False)
    assert resumed["g_fg"] == ref_g_fg

    def _assert_tree_eq(a: object, b: object, prefix: str = "") -> None:
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and torch.equal(a, b), f"Coupled Law 4 continuation mismatch {prefix}"
        elif isinstance(a, dict):
            assert isinstance(b, dict) and set(a.keys()) == set(b.keys()), f"Key mismatch {prefix}"
            for key in a:
                _assert_tree_eq(a[key], b[key], f"{prefix}.{key}")
        else:
            assert a == b, f"Value mismatch {prefix}: {a!r} != {b!r}"

    for m_name in ("G", "D", "prior"):
        _assert_tree_eq(ref_state["models"][m_name], resumed["models"][m_name], m_name)


class _ToyDeepResBNDisc(nn.Module):
    """Deep Pre-RMSNorm residual critic with branch-output RMSNorm for Issue-2 depth-scaling tests."""

    def __init__(self, in_dim: int = 1088, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.depth = int(depth)
        self.inp = nn.Linear(in_dim, hidden)
        self.in_projs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out_projs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, 1)

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.leaky_relu(self.inp(x), negative_slope=0.2)
        scale = 0.5 / float(max(1, self.depth))
        for w_in, w_out in zip(self.in_projs, self.out_projs):
            br = w_out(nn.functional.leaky_relu(w_in(self._rms_norm(h)), negative_slope=0.2))
            h = h + scale * self._rms_norm(br)
        return self.head(self._rms_norm(h)).squeeze(-1)


def test_unified_v11_critic_schatten6_spectral_control_and_deep_resbn_continuation(tmp_path: Path) -> None:
    """Issue 2 semantic controls for zero-SVD exact-deadband Schatten-6 spectral decay on 2D critic hidden weights:
    1. Bounds runaway normalized spectral norm on square & rectangular critic weights under repeated
       aligned Quintic-polar steps, while leaving singular values <= tau=2.0 identically untouched.
    2. Strictly isolates generator weights, 1-row critic readouts, 1D biases, and Conv2d weights.
    3. Cools smoothly with s_therm, does nothing on zero gradients, stays contractive under large
       inverted-critic alpha_gain boosts, and preserves batched-vs-single and group-split parity.
    4. Runs with 0 autograd.grad/svd/eigh/pinv and exact fresh-process checkpoint continuation on a
       deep high-D (dim=1088) ResBN critic across active Schatten-6 steps.
    """
    from particlegan._unified_v11.optimizer import (
        _CRITIC_SCHATTEN6_TAU,
        _schatten6_spectral_decay_2d,
        _schatten6_spectral_decay_batched,
    )

    torch.manual_seed(0)

    # 1a. Repeated aligned Quintic-polar updates on square (32x32) and tall rectangular (64x16, gamma=2.0)
    #     critic hidden matrices stay spectrally bounded near tau * lam^(-1/5) ~= 2.64 (vs > 10 unbounded):
    w_sq = nn.Parameter(torch.eye(32))
    w_rect = nn.Parameter(torch.zeros(64, 16))
    with torch.no_grad():
        w_rect[:16, :16].copy_(torch.eye(16) * 2.0)  # normalized sv == 1.0 since gamma = sqrt(64/16) = 2.0
    opt_c = UnifiedV11EquilibratedOptimizer(
        [{"params": [w_sq, w_rect], "role": "critic", "lr": 0.05}],
        smoothing=0.0,
    )
    g_sq = -torch.eye(32)
    g_rect = torch.zeros(64, 16)
    g_rect[:16, :16] = -torch.eye(16)
    for _ in range(250):
        w_sq.grad = g_sq.clone()
        w_rect.grad = g_rect.clone()
        opt_c.step()
    sv_sq = float(torch.linalg.matrix_norm(w_sq.detach(), ord=2).item())
    sv_rect_norm = float(torch.linalg.matrix_norm(w_rect.detach(), ord=2).item()) / 2.0
    assert 2.0 < sv_sq < 3.0, f"Square critic spectral norm not bounded properly: {sv_sq:.3f}"
    assert 2.0 < sv_rect_norm < 3.0, f"Rect critic normalized spectral norm not bounded properly: {sv_rect_norm:.3f}"

    # 1b. Exact zero-deadband below tau=2.0: any matrix with sigma_max <= tau receives identically zero decay:
    for sig_val in (0.8, 1.5, 1.95):
        w_sub = torch.eye(16) * sig_val
        d_sub = _schatten6_spectral_decay_2d(w_sub, torch.tensor(0.02))
        assert torch.equal(d_sub, torch.zeros_like(w_sub)), f"Non-zero deadband decay at sigma={sig_val}"
    # Above tau=2.0, decay activates cleanly:
    w_over = torch.eye(16) * 2.8
    d_over = _schatten6_spectral_decay_2d(w_over, torch.tensor(0.02))
    assert float(d_over.norm().item()) > 0.0 and float((w_over + d_over).norm().item()) < float(w_over.norm().item())

    # 2. Strict role & shape isolation: generator 2D, critic 1-row readout, critic 1D bias, critic Conv2d
    #    receive zero Schatten-6 decay even at 4x initial spectral scale:
    for role, shape, is_conv in [
        ("generator", (16, 16), False),
        ("critic", (1, 16), False),
        ("critic", (16,), False),
        ("critic", (8, 8, 3, 3), True),
        ("critic", (8, 8, 1, 1), True),
    ]:
        p_small = nn.Parameter(torch.randn(*shape) * 0.1)
        p_big = nn.Parameter(p_small.detach().clone() * 40.0)
        grad_fixed = torch.randn(*shape)
        opt_s = UnifiedV11EquilibratedOptimizer([{"params": [p_small], "role": role, "lr": 0.02}], smoothing=0.0)
        opt_b = UnifiedV11EquilibratedOptimizer([{"params": [p_big], "role": role, "lr": 0.02}], smoothing=0.0)
        if is_conv:
            opt_s._conv_layouts[id(p_small)] = (False, 1)
            opt_b._conv_layouts[id(p_big)] = (False, 1)
        pre_s = p_small.detach().clone()
        pre_b = p_big.detach().clone()
        p_small.grad = grad_fixed.clone()
        p_big.grad = grad_fixed.clone()
        opt_s.step()
        opt_b.step()
        delta_s = p_small.detach() - pre_s
        delta_b = p_big.detach() - pre_b
        assert torch.allclose(delta_s, delta_b, atol=1e-5, rtol=1e-5), (
            f"Unexpected weight-dependent decay on role={role}, shape={shape}, is_conv={is_conv}"
        )

    # 3a. Thermal cooling silences Schatten-6 decay, and zero gradient applies zero decay:
    w_hot = nn.Parameter(torch.eye(16) * 3.0)
    w_cold = nn.Parameter(torch.eye(16) * 3.0)
    w_zero = nn.Parameter(torch.eye(16) * 3.0)
    rec_hot = UnifiedV11StepRecord()
    rec_cold = UnifiedV11StepRecord()
    rec_hot._thermal_scale = 1.0
    rec_cold._thermal_scale = 1e-6
    opt_hot = UnifiedV11EquilibratedOptimizer([{"params": [w_hot], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_cold = UnifiedV11EquilibratedOptimizer([{"params": [w_cold], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_zero = UnifiedV11EquilibratedOptimizer([{"params": [w_zero], "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_hot._unified_v11_record = rec_hot
    opt_cold._unified_v11_record = rec_cold
    # Tiny orthogonal-to-eye gradient so polar step norm is identical, difference is purely Schatten-6:
    g_orth = torch.zeros(16, 16)
    g_orth[0, 1] = 1.0
    w_hot.grad = g_orth.clone()
    w_cold.grad = g_orth.clone()
    w_zero.grad = torch.zeros(16, 16)
    opt_hot.step()
    opt_cold.step()
    opt_zero.step()
    assert float(w_hot.detach().diag().mean().item()) < 2.97, "Hot critic Schatten-6 did not contract above-tau weight"
    assert abs(float(w_cold.detach().diag().mean().item()) - 3.0) < 1e-5, "Cooled critic Schatten-6 drifted at equilibrium"
    assert torch.equal(w_zero.detach(), torch.eye(16) * 3.0), "Zero-gradient critic weight modified by Schatten-6"

    # 3b. Inverted-critic large alpha_gain boost (alpha_gain >> 1) keeps Schatten-6 strictly contractive:
    d_inv = nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 1, bias=False))
    with torch.no_grad():
        d_inv[0].weight.copy_(torch.eye(16) * 6.0)
        d_inv[1].weight.fill_(0.1)
    rec_inv = UnifiedV11StepRecord()
    rec_inv.margin_initialized = True
    rec_inv.ema_margin = -12.0  # triggers large delta_d_min -> alpha_gain >> 1.0
    rec_inv._thermal_scale = 1.0
    opt_inv = UnifiedV11EquilibratedOptimizer([{"params": list(d_inv.parameters()), "role": "critic", "lr": 0.02}], smoothing=0.0)
    opt_inv._unified_v11_record = rec_inv
    opt_inv.critic = d_inv
    x_probe = torch.randn(4, 16)
    with torch.no_grad():
        rec_inv._d_probe_list = [((x_probe,), {}, d_inv(x_probe))]
    d_inv[0].weight.grad = torch.randn(16, 16) * 1e-3
    d_inv[1].weight.grad = torch.randn(1, 16) * 1e-3
    sv_before_inv = float(torch.linalg.matrix_norm(d_inv[0].weight.detach(), ord=2).item())
    opt_inv.step()
    sv_after_inv = float(torch.linalg.matrix_norm(d_inv[0].weight.detach(), ord=2).item())
    assert opt_inv.last_step_gain > 5.0
    assert torch.isfinite(d_inv[0].weight).all() and sv_after_inv < sv_before_inv

    # 3c. Batched-vs-single parity & group-split invariance:
    w_a = torch.randn(16, 24) * 0.8
    w_b = torch.randn(16, 24) * 1.1
    rates = torch.tensor([0.02, 0.015])
    s6_batched = _schatten6_spectral_decay_batched(torch.stack([w_a, w_b], dim=0), rates)
    s6_a = _schatten6_spectral_decay_2d(w_a, rates[0])
    s6_b = _schatten6_spectral_decay_2d(w_b, rates[1])
    assert torch.allclose(s6_batched[0], s6_a, atol=1e-6, rtol=1e-5)
    assert torch.allclose(s6_batched[1], s6_b, atol=1e-6, rtol=1e-5)

    p1a, p2a = nn.Parameter(w_a.clone()), nn.Parameter(w_b.clone())
    p1b, p2b = nn.Parameter(w_a.clone()), nn.Parameter(w_b.clone())
    ga, gb = torch.randn_like(w_a), torch.randn_like(w_b)
    opt_one = UnifiedV11EquilibratedOptimizer([{"params": [p1a, p2a], "role": "critic", "lr": 0.02}], smoothing=0.0061)
    opt_two = UnifiedV11EquilibratedOptimizer(
        [{"params": [p1b], "role": "critic", "lr": 0.02}, {"params": [p2b], "role": "critic", "lr": 0.02}],
        smoothing=0.0061,
    )
    p1a.grad, p2a.grad = ga.clone(), gb.clone()
    p1b.grad, p2b.grad = ga.clone(), gb.clone()
    opt_one.step()
    opt_two.step()
    assert torch.equal(p1a.detach(), p1b.detach()) and torch.equal(p2a.detach(), p2b.detach())

    # 4. Zero forbidden primitives + exact fresh-process checkpoint continuation on deep ResBN critic at dim=1088:
    dim = 1088
    rec = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=8,
        total_steps=10,
        training_precision="float32",
    )
    g = _ToyGen2D(z_dim=8, out_dim=dim)
    d = _ToyDeepResBNDisc(in_dim=dim, hidden=32, depth=2)
    p = MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False)
    init.deterministic_orthogonal_(g, seed=10)
    init.deterministic_orthogonal_(d, seed=11)
    init.deterministic_orthogonal_(p, seed=12)
    # Push one critic branch weight above tau so Schatten-6 is actively contracting across the checkpoint boundary:
    with torch.no_grad():
        d.out_projs[0].weight.mul_(_CRITIC_SCHATTEN6_TAU + 0.8)

    trainer = GANTrainer(rec, g, d, prior=p, seed=0)
    batches = [torch.randn(8, dim, generator=torch.Generator().manual_seed(900 + i)) for i in range(6)]

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("Forbidden autograd.grad or spectral decomposition called during deep ResBN critic step!")

    with (
        mock.patch("torch.autograd.grad", side_effect=_forbid),
        mock.patch("torch.linalg.svd", side_effect=_forbid),
        mock.patch("torch.linalg.svdvals", side_effect=_forbid),
        mock.patch("torch.linalg.eigh", side_effect=_forbid),
        mock.patch("torch.linalg.eigvalsh", side_effect=_forbid),
        mock.patch("torch.linalg.pinv", side_effect=_forbid),
    ):
        for i in range(3):
            trainer.step(batches[i])

    ckpt_path = tmp_path / "ckpt_v11_resbn_s6.pt"
    out_path = tmp_path / "resumed_v11_resbn_s6.pt"
    torch.save({"state": trainer.state_dict(), "batches": batches[3:]}, ckpt_path)
    for i in range(3, 6):
        trainer.step(batches[i])
    sample_ref = trainer.sample(4)
    d_ref = trainer.D.state_dict()

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v11 import _ToyGen2D, _ToyDeepResBNDisc

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(
    rec,
    _ToyGen2D(z_dim=8, out_dim=1088),
    _ToyDeepResBNDisc(in_dim=1088, hidden=32, depth=2),
    prior=MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False),
    seed=99,
)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save({{"sample": tr.sample(4), "d": tr.D.state_dict()}}, {str(out_path)!r})
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    resumed = torch.load(out_path, weights_only=False)
    assert torch.equal(sample_ref, resumed["sample"])
    for k, v in d_ref.items():
        assert torch.equal(v, resumed["d"][k]), f"Deep ResBN D mismatch on {k}"


def test_unified_v11_combined_deep_resbn_law4_coupling_and_schatten6_continuation(tmp_path: Path) -> None:
    """Combined Issue 1 + Issue 2 semantic control on symmetric deep ResBN G & D at dim=1088:
    - No false stiff_gate on steady normalized deep ResBN G (current_stiff_gate == 0.0, no momentum allocated).
    - Coupled critic gain tracks rec._g_func_gain == opt_g._ema_func_gain while staying bounded.
    - Exact-deadband Schatten-6 actively contracts above-tau critic branch weights with 0 forbidden primitives.
    - Fresh-process bit-exact continuation preserves G, D, prior, and g_func_gain across
      simultaneous active Law-4 coupling and active Schatten-6 contraction.
    """
    torch.manual_seed(0)
    dim = 1088
    rec = get_recipe(
        "unified_v11",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=8,
        total_steps=16,
        training_precision="float32",
    )
    g = _ResBNTrunkGen(z_dim=8, width=32, out_dim=dim, depth=4)
    d = _ToyDeepResBNDisc(in_dim=dim, hidden=32, depth=4)
    p = MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False)
    init.deterministic_orthogonal_(g, seed=50)
    init.deterministic_orthogonal_(d, seed=51)
    init.deterministic_orthogonal_(p, seed=52)
    # Seed one critic hidden block well above the ~2.64 Schatten-6 asymptote so contraction is active alongside Law-4 coupling:
    with torch.no_grad():
        sv0 = float(torch.linalg.matrix_norm(d.out_projs[0].weight.detach(), ord=2).item())
        d.out_projs[0].weight.mul_(4.0 / max(sv0, 1e-6))
    sv_init_over = float(torch.linalg.matrix_norm(d.out_projs[0].weight.detach(), ord=2).item())

    trainer = GANTrainer(rec, g, d, prior=p, seed=0)
    # Place G at a realistic steady normalized deep-ResBN func_gain (~0.40) so coupling is active and stiff_gate == 0:
    trainer.opt_g._ema_func_gain = 0.42
    trainer.opt_d.record._g_func_gain = 0.42

    batches = [torch.randn(8, dim, generator=torch.Generator().manual_seed(1100 + i)) for i in range(10)]

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("Forbidden autograd.grad or spectral decomposition called in combined ResBN step!")

    with (
        mock.patch("torch.autograd.grad", side_effect=_forbid),
        mock.patch("torch.linalg.svd", side_effect=_forbid),
        mock.patch("torch.linalg.svdvals", side_effect=_forbid),
        mock.patch("torch.linalg.eigh", side_effect=_forbid),
        mock.patch("torch.linalg.eigvalsh", side_effect=_forbid),
        mock.patch("torch.linalg.pinv", side_effect=_forbid),
    ):
        for i in range(5):
            trainer.step(batches[i])

    # Verify Issue 1 (no false stiff_gate, active coupled g_func_gain) + Issue 2 (active spectral contraction):
    assert trainer.opt_g.current_stiff_gate() == 0.0
    assert all("m_unit" not in trainer.opt_g.state[param] for param in g.parameters())
    assert trainer.opt_d.record._g_func_gain == trainer.opt_g._ema_func_gain
    assert 0.18 < trainer.opt_d.record._g_func_gain < 0.75
    sv_mid = float(torch.linalg.matrix_norm(d.out_projs[0].weight.detach(), ord=2).item())
    assert sv_mid < sv_init_over - 0.05, f"Schatten-6 failed to contract coupled deep critic: {sv_init_over:.3f} -> {sv_mid:.3f}"

    ckpt_path = tmp_path / "ckpt_v11_combined_resbn.pt"
    out_path = tmp_path / "resumed_v11_combined_resbn.pt"
    torch.save({"state": trainer.state_dict(), "batches": batches[5:]}, ckpt_path)
    for i in range(5, 10):
        trainer.step(batches[i])
    ref_sample = trainer.sample(4)
    ref_state = trainer.state_dict()
    ref_g_fg = trainer.opt_d.record._g_func_gain
    sv_end = float(torch.linalg.matrix_norm(d.out_projs[0].weight.detach(), ord=2).item())
    assert sv_end < sv_mid, f"Schatten-6 stopped contracting after checkpoint: {sv_mid:.3f} -> {sv_end:.3f}"

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v11 import _ResBNTrunkGen, _ToyDeepResBNDisc

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(
    rec,
    _ResBNTrunkGen(z_dim=8, width=32, out_dim=1088, depth=4),
    _ToyDeepResBNDisc(in_dim=1088, hidden=32, depth=4),
    prior=MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False),
    seed=321,
)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save(
    {{
        "sample": tr.sample(4),
        "models": tr.state_dict()["models"],
        "g_fg": tr.opt_d.record._g_func_gain,
    }},
    {str(out_path)!r},
)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    resumed = torch.load(out_path, weights_only=False)
    assert torch.equal(ref_sample, resumed["sample"])
    assert resumed["g_fg"] == ref_g_fg
    for m_name in ("G", "D", "prior"):
        for k, v in ref_state["models"][m_name].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, resumed["models"][m_name][k]), f"Combined continuation mismatch {m_name}.{k}"
            else:
                assert v == resumed["models"][m_name][k], f"Combined continuation value mismatch {m_name}.{k}"


