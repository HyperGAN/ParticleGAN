"""Unit, mathematical-control, tamper-rejection, and fresh-process continuation tests for ``preset='cond_fiber'``."""
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
    CondFiberBank,
    CondFiberCriticPenalty,
    CondFiberGANLoss,
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    get_recipe,
    init,
)
from particlegan.cond_fiber import (
    CondFiberStepRecord,
    _CondFiberObserverStepRecord,
    cond_fiber_conditional_loss,
)


class _ToyGen2D(nn.Module):
    def __init__(self, z_dim: int = 4, out_dim: int = 2) -> None:
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


class _ToyCondGen(nn.Module):
    def __init__(self, z_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + z_dim + 1, 24),
            nn.LeakyReLU(0.2),
            nn.Linear(24, 24),
            nn.LeakyReLU(0.2),
            nn.Linear(24, 2),
        )

    def forward(self, xt: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((xt, z, t[:, None].to(xt) / 4.0), dim=1))


class _ToyCondDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 24),
            nn.LeakyReLU(0.2),
            nn.Linear(24, 24),
            nn.LeakyReLU(0.2),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, xt, t[:, None].to(x) / 4.0), dim=1)).squeeze(-1)


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


def test_cond_fiber_preset_validation_and_default_recipe_invariant() -> None:
    assert DEFAULT_RECIPE == "axiom"
    rec = get_recipe("cond_fiber")
    assert rec.critic_formulation == "cond_fiber"
    assert rec.effective_critic_formulation == "cond_fiber"
    assert rec.prior_regularizer == "unified"
    assert rec.training_precision == "bf16_channels_last"
    assert rec.optimizer_dense_profile == "newton_schulz_hybrid_v1"

    roundtrip = Recipe.from_dict(rec.to_dict())
    assert roundtrip == rec

    ctx = FormulationContext(
        recipe_preset="cond_fiber",
        prior={"kind": "mog", "sigma": 0.25, "standardize": False, "learnable": True},
    )
    assert ctx.capabilities()["secant_critic"] is True

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe("ka2", critic_formulation="cond_fiber")
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe("cond_fiber", reg_arm="b_cap")


def test_cond_fiber_forbids_autograd_grad_everywhere() -> None:
    """Semantic control: CondFiberCriticPenalty + CondFiberGANLoss must NEVER call torch.autograd.grad."""
    rec = get_recipe(
        "cond_fiber",
        prior_kind="mog",
        standardize=False,
        batch_size=16,
        num_particles=32,
        z_dim=4,
        total_steps=20,
        training_precision="float32",
        constraint_geometry_mode="none",
    )

    with mock.patch("torch.autograd.grad", side_effect=AssertionError("autograd.grad forbidden in CondFiber!")):
        # 1. Unconditional 2D + 8x8 image paths delegate cleanly on unified with zero autograd.grad:
        g2 = _ToyGen2D(z_dim=4, out_dim=2)
        d2 = _ToyDisc2D(in_dim=2)
        p2 = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05, standardize=False)
        trainer = GANTrainer(rec, g2, d2, prior=p2, seed=0)
        out = trainer.step(torch.randn(16, 2), collect_stats=True)
        assert out["penalty_stats"]["applied"] is True
        assert out["penalty_stats"]["phase"] == "cond_fiber"

        g_img = _ToyImgGen(z_dim=4, size=8)
        d_img = _ToyImgDisc()
        p_img = ParticlePrior(num_particles=16, z_dim=4)
        rec_img = rec.replace(prior_kind="particles", sigma_rel=0.0, num_particles=16, batch_size=8)
        trainer_img = GANTrainer(rec_img, g_img, d_img, prior=p_img, seed=0)
        out_img = trainer_img.step(torch.randn(8, 1, 8, 8), collect_stats=True)
        assert out_img["penalty_stats"]["applied"] is True
        assert out_img["penalty_stats"]["live_endpoints_reused"] is True

        # 2. Continuous-conditioned DDGAN path exercising active CondFiberBank across 6 steps:
        d_cond = _ToyCondDisc()
        g_cond = _ToyCondGen(z_dim=4)
        p_cond = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.25, standardize=False)
        opt_g, opt_d = rec.make_optimizers(g_cond, d_cond, p_cond)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d, collect_stats=True)
        assert isinstance(loss_fn, CondFiberGANLoss)
        assert isinstance(pen_fn, CondFiberCriticPenalty)
        gen = torch.Generator().manual_seed(17)
        for _ in range(6):
            xt = torch.randn(16, 2, generator=gen)
            t = torch.randint(1, 3, (16,), generator=gen)
            x_r = 0.7 * xt + 0.25 * torch.randn(16, 2, generator=gen)
            z = torch.randn(16, 4, generator=gen)
            x_f = g_cond(xt, z, t)
            opt_d.zero_grad()
            dl = loss_fn.d_loss(d_cond(x_r, xt=xt, t=t), d_cond(x_f.detach(), xt=xt, t=t))
            pen = pen_fn(d_cond, x_r, x_f.detach(), xt=xt, t=t)
            (dl + pen).backward()
            opt_d.step()
            opt_g.zero_grad()
            gl = loss_fn.g_loss(d_cond(x_f, xt=xt, t=t), d_cond(x_r, xt=xt, t=t).detach())
            gl.backward()
            opt_g.step()
        assert len(opt_d.record.cond_fiber_bank.strata) == 2


def test_cond_fiber_swap_invariance_equilibrium_unbiasedness_and_innovation_reset() -> None:
    """Verify exact X<->Y swap invariance of pooled drift/whitening/cov-cancellation, zero variance collapse,
    strict sensitivity to wrong conditional slope/variance, and clock-free innovation reset on shift."""
    # 1. Exact swap invariance of CondFiberBank pooled drift, whitening W_t, z_innov, and exact
    #    anti-symmetry of delta_cov_loc under (X, Y) <-> (Y, X):
    gen = torch.Generator().manual_seed(42)
    bank_xy = CondFiberBank()
    bank_yx = CondFiberBank()
    for _ in range(10):
        c = torch.randn(24, 2, generator=gen, dtype=torch.float64)
        x = 0.8 * c + torch.tensor([[0.3, -0.2]], dtype=torch.float64) + 0.35 * torch.randn(24, 2, generator=gen, dtype=torch.float64)
        y = 0.6 * c + torch.tensor([[-0.1, 0.1]], dtype=torch.float64) + 0.25 * torch.randn(24, 2, generator=gen, dtype=torch.float64)
        bank_xy.update_stratum((1,), c, x, y)
        bank_yx.update_stratum((1,), c, y, x)

    c_q = torch.randn(16, 2, generator=gen, dtype=torch.float64)
    x_q = 0.8 * c_q + 0.35 * torch.randn(16, 2, generator=gen, dtype=torch.float64)
    y_q = 0.6 * c_q + 0.25 * torch.randn(16, 2, generator=gen, dtype=torch.float64)
    q_xy = bank_xy.query_stratum((1,), c_q, x_q, y_q)
    q_yx = bank_yx.query_stratum((1,), c_q, y_q, x_q)
    assert q_xy is not None and q_yx is not None
    assert torch.allclose(q_xy["m_pool"], q_yx["m_pool"], atol=1e-10)
    assert torch.allclose(q_xy["w_mat"], q_yx["w_mat"], atol=1e-10)
    assert torch.allclose(q_xy["delta_cov_loc"], -q_yx["delta_cov_loc"], atol=1e-10)
    assert abs(q_xy["z_innov"] - q_yx["z_innov"]) < 1e-10

    # 2. Equilibrium unbiasedness (zero variance-collapse bias!):
    #    Under matched law p(x|c) == q(y|c) = N(0.7*c, 0.25^2 I), expected gradient w.r.t. scale alpha
    #    at alpha=1.0 must be statistically indistinguishable from 0 (never systematically positive!).
    rec = _CondFiberObserverStepRecord()
    gen_eq = torch.Generator().manual_seed(123)
    grad_scales = []
    grad_shifts = []
    for step_i in range(80):
        c_b = torch.randn(32, 2, generator=gen_eq)
        t_b = torch.ones(32, dtype=torch.long)
        eps_x = 0.25 * torch.randn(32, 2, generator=gen_eq)
        eps_y = 0.25 * torch.randn(32, 2, generator=gen_eq)
        alpha = torch.tensor(1.0, requires_grad=True)
        shift = torch.zeros(2, requires_grad=True)
        x_b = 0.7 * c_b + shift + alpha * eps_x
        y_b = 0.7 * c_b + eps_y
        loss = cond_fiber_conditional_loss(x_b, y_b, c_b, cond_disc=t_b, record=rec)
        g_alpha, g_shift = torch.autograd.grad(loss, (alpha, shift))
        if step_i >= 20:
            grad_scales.append(float(g_alpha.item()))
            grad_shifts.append(float(g_shift.mean().item()))
    mean_g_alpha = sum(grad_scales) / len(grad_scales)
    std_g_alpha = math.sqrt(sum((v - mean_g_alpha) ** 2 for v in grad_scales) / len(grad_scales))
    se_alpha = std_g_alpha / math.sqrt(len(grad_scales))
    assert abs(mean_g_alpha) < 3.5 * max(se_alpha, 1e-3), (
        f"Equilibrium variance-scale gradient biased: mean={mean_g_alpha:.5f}, se={se_alpha:.5f}"
    )
    mean_g_shift = sum(grad_shifts) / len(grad_shifts)
    assert abs(mean_g_shift) < 0.05

    # 3. Deliberate slope & variance error must produce strong corrective gradient in exact right sign:
    rec_err = _CondFiberObserverStepRecord()
    gen_err = torch.Generator().manual_seed(99)
    for _ in range(15):
        c_b = torch.randn(32, 2, generator=gen_err)
        t_b = torch.ones(32, dtype=torch.long)
        x_bad = (0.1 * c_b + 0.55 * torch.randn(32, 2, generator=gen_err)).requires_grad_(True)
        y_true = 0.7 * c_b + 0.25 * torch.randn(32, 2, generator=gen_err)
        _ = cond_fiber_conditional_loss(x_bad, y_true, c_b, cond_disc=t_b, record=rec_err)
    c_probe = torch.randn(32, 2, generator=gen_err)
    t_probe = torch.ones(32, dtype=torch.long)
    slope = torch.tensor(0.1, requires_grad=True)
    scale = torch.tensor(0.55, requires_grad=True)
    x_probe = slope * c_probe + scale * torch.randn(32, 2, generator=gen_err)
    y_probe = 0.7 * c_probe + 0.25 * torch.randn(32, 2, generator=gen_err)
    rec_err._in_protected_probe = True
    loss_err = cond_fiber_conditional_loss(x_probe, y_probe, c_probe, cond_disc=t_probe, record=rec_err)
    g_slope, g_scale = torch.autograd.grad(loss_err, (slope, scale))
    assert float(g_slope.item()) < -0.05, f"Expected negative dL/d(slope) to push 0.1 -> 0.7, got {g_slope.item()}"
    assert float(g_scale.item()) > 0.05, f"Expected positive dL/d(scale) to contract 0.55 -> 0.25, got {g_scale.item()}"

    # 4. Innovation reset under sudden target shift (strictly clock-free):
    n_before = float(bank_xy.strata[(1,)]["n_eff"])
    ring_before = int(bank_xy.strata[(1,)]["ring_c"].shape[0])
    c_sh = torch.randn(24, 2, generator=gen, dtype=torch.float64)
    x_sh = 0.8 * c_sh + 0.35 * torch.randn(24, 2, generator=gen, dtype=torch.float64)
    y_sh = 0.6 * c_sh + 6.0 + 0.25 * torch.randn(24, 2, generator=gen, dtype=torch.float64)
    q_sh = bank_xy.query_stratum((1,), c_sh, x_sh, y_sh)
    assert q_sh is not None and q_sh["z_innov"] > 5.0
    bank_xy.update_stratum((1,), c_sh, x_sh, y_sh, z_innov=q_sh["z_innov"])
    assert float(bank_xy.strata[(1,)]["n_eff"]) < 0.75 * n_before + 24.0
    assert int(bank_xy.strata[(1,)]["ring_c"].shape[0]) < ring_before + 24


def test_cond_fiber_deliberate_damage_and_checkpoint_tamper_rejection() -> None:
    """Verify critic regularization damage control and strict checkpoint tamper rejection."""
    def _run_grad_probe(reg_coeff: float) -> float:
        torch.manual_seed(11)
        rec = get_recipe(
            "cond_fiber",
            prior_kind="mog",
            standardize=False,
            t1_pullback_weight=0.0,
            reg_coeff=reg_coeff,
            reg_kappa=1.0,
            lr=0.04,
            d_lr_mult=2.5,
            batch_size=16,
            num_particles=32,
            z_dim=4,
            total_steps=60,
            training_precision="float32",
        )
        g = _ToyGen2D(4, 2)
        d = _ToyDisc2D(2)
        p = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05, standardize=False)
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

    # Tamper rejection on CondFiberStepRecord & _CondFiberObserverStepRecord with active bank:
    rec_obs = _CondFiberObserverStepRecord()
    c_b = torch.randn(24, 2, dtype=torch.float64)
    x_b = 0.6 * c_b + 0.2 * torch.randn(24, 2, dtype=torch.float64)
    y_b = 0.6 * c_b + 0.2 * torch.randn(24, 2, dtype=torch.float64)
    rec_obs.cond_fiber_bank.update_stratum((1,), c_b, x_b, y_b)
    good = rec_obs.state_dict()

    bad_schema = deepcopy(good)
    bad_schema["cond_fiber"]["cond_fiber_bank"]["schema"] = 999
    with pytest.raises(ValueError, match="schema"):
        rec_obs.load_state_dict(bad_schema)

    bad_nan = deepcopy(good)
    bad_nan["cond_fiber"]["cond_fiber_bank"]["strata"][0]["g_px"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        rec_obs.load_state_dict(bad_nan)

    bad_neg = deepcopy(good)
    bad_neg["cond_fiber"]["cond_fiber_bank"]["strata"][0]["n_eff"] = -1.0
    with pytest.raises(ValueError, match="n_eff"):
        rec_obs.load_state_dict(bad_neg)

    bad_ring = deepcopy(good)
    bad_ring["cond_fiber"]["cond_fiber_bank"]["strata"][0]["ring_x"] = bad_ring["cond_fiber"]["cond_fiber_bank"]["strata"][0]["ring_x"][:-1]
    with pytest.raises(ValueError, match="matching lengths"):
        rec_obs.load_state_dict(bad_ring)

    bad_step = deepcopy(good)
    bad_step["cond_fiber"]["unified"]["step_scale"] = 1.5
    with pytest.raises(ValueError, match="step_scale"):
        rec_obs.load_state_dict(bad_step)

    # Formulation record tamper check:
    rec_form = CondFiberStepRecord()
    rec_form.cond_fiber_bank.update_stratum((2,), c_b, x_b, y_b)
    good_f = rec_form.state_dict()
    bad_f = deepcopy(good_f)
    bad_f["formulation"] = "unified"
    with pytest.raises(ValueError, match="formulation"):
        rec_form.load_state_dict(bad_f)


def test_cond_fiber_fresh_process_ddgan_continuation_across_active_bank(tmp_path: Path) -> None:
    """Bit-exact fresh-process checkpoint + RNG continuation across active CondFiberBank boundaries."""
    from benchmarks.toy_audit import api_model_ddgan

    ckpt_path = tmp_path / "ddgan_cond_fiber_ckpt.pt"
    out_path = tmp_path / "ddgan_cond_fiber_resumed.pt"

    fix_ref = api_model_ddgan.build_case(
        "api-model-ddgan-gaussian-chain",
        device="cpu",
        seed=0,
        recipe_name="cond_fiber",
    )
    # Run 12 steps so all 4 diffusion strata are active and past the cold-start maturity gate:
    for _ in range(12):
        fix_ref.step()
    assert len(fix_ref.opt_d.record.cond_fiber_bank.strata) == 4
    torch.save(fix_ref.state_dict(), ckpt_path)
    for _ in range(8):
        fix_ref.step()
    final_ref = fix_ref.state_dict()
    obs_ref = fix_ref.observe(n=64, seed=10000)["metrics"]

    script = f"""
import torch
import particlegan
assert particlegan.DEFAULT_RECIPE == "axiom"
import particlegan.cond_fiber
assert particlegan.DEFAULT_RECIPE == "axiom"
from particlegan import Recipe
from benchmarks.toy_audit import api_model_ddgan

state = torch.load({str(ckpt_path)!r}, weights_only=False)
rec = Recipe.from_dict(state["recipe"])
fix = api_model_ddgan.build_case(
    "api-model-ddgan-gaussian-chain",
    device="cpu",
    seed=0,
    recipe=rec,
)
fix.load_state_dict(state)
for _ in range(8):
    fix.step()
torch.save({{"final": fix.state_dict(), "obs": fix.observe(n=64, seed=10000)["metrics"]}}, {str(out_path)!r})
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
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), f"List mismatch in {prefix}"
            for idx, (va, vb) in enumerate(zip(a, b)):
                _assert_tree_equal(va, vb, f"{prefix}[{idx}]")
        else:
            assert a == b, f"Value mismatch in {prefix}: {a!r} != {b!r}"

    for model_name in ("G", "D", "prior"):
        _assert_tree_equal(final_ref["models"][model_name], resumed["final"]["models"][model_name], model_name)
    _assert_tree_equal(
        final_ref["optimizers"][1]["regularizer"]["record"]["cond_fiber"],
        resumed["final"]["optimizers"][1]["regularizer"]["record"]["cond_fiber"],
        "regularizer.record.cond_fiber",
    )
    assert final_ref["batch_digest"] == resumed["final"]["batch_digest"]
    for key, val in obs_ref.items():
        assert resumed["obs"][key] == val, f"Observation metric mismatch in {key}: {val} != {resumed['obs'][key]}"
