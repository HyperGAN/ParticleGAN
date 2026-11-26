"""Unit, semantic-control, fresh-process continuation, and clock-free tests for Track 2 ``t2_secant``."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest.mock as mock

import pytest
import torch
from torch import nn

from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    InputNoise,
    T2SecantCriticAdam,
    T2SecantCriticPenalty,
    T2SecantGANLoss,
    get_recipe,
    init,
)
from experiments.forge.api import FormulationContext


def _tiny_models(z_dim: int = 2, out_dim: int = 2, hidden: int = 16):
    g = nn.Sequential(nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, out_dim))
    d = nn.Sequential(nn.Linear(out_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
    init.deterministic_orthogonal_(g)
    init.deterministic_orthogonal_(d)
    return g, d


def test_t2_secant_preset_and_recipe_validation():
    assert DEFAULT_RECIPE == "axiom"
    recipe = get_recipe("t2_secant")
    assert recipe.critic_formulation == "t2_secant"
    assert recipe.effective_critic_formulation == "t2_secant"
    assert recipe.optimizer_family == "dualnorm"
    assert recipe.reg_anchor_weight == 0.0
    assert recipe.lr_floor == 1.0 and recipe.network_lr_floor == 1.0

    ctx = FormulationContext(
        recipe_preset="t2_secant",
        prior={"kind": "mog", "sigma": 0.1, "standardize": False, "learnable": True},
        requires_capabilities=("learned_locations", "named_rng", "secant_critic"),
    )
    assert ctx.capabilities()["secant_critic"] is True

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe(
            "t2_secant",
            optimizer_family="formulation",
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_dense_profile="default",
            optimizer_svd_backend="native",
            training_precision="float32",
            constraint_geometry_mode="none",
            reg_anchor_weight=0.5,
        )
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe(
            "t2_secant",
            reg_arm="b_cap",
            optimizer_family="adam",
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_dense_profile="default",
            optimizer_svd_backend="native",
            training_precision="float32",
            constraint_geometry_mode="none",
        )


def _tiny_image_critic():
    d = nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 1),
    )
    init.deterministic_orthogonal_(d)
    return d


def test_semantic_control_zero_autograd_grad_during_training():
    """Forbidden-fallback control: t2_secant must NEVER call torch.autograd.grad during MLP or Conv training."""
    g, d = _tiny_models()
    recipe = get_recipe(
        "t2_secant",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=10,
        prior_kind="mog",
        sigma_rel=0.05,
        standardize=False,
        constraint_geometry_mode="none",
    )
    prior = recipe.make_prior()
    init.deterministic_orthogonal_(prior)
    trainer = GANTrainer(recipe, g, d, prior=prior, max_steps=10)
    data_rng = torch.Generator().manual_seed(42)

    with mock.patch("torch.autograd.grad", side_effect=AssertionError("autograd.grad forbidden in t2_secant!")):
        for _ in range(5):
            real = torch.randn(32, 2, generator=data_rng)
            out = trainer.step(real, collect_stats=True)
            assert torch.isfinite(out["loss_d"])
            assert torch.isfinite(out["penalty"])
            assert out["penalty_stats"]["applied"] is True
            assert out["penalty_stats"]["phase"] == "t2_secant"

        # Also verify 4D Conv2d image branch never calls autograd.grad and reuses live endpoints
        d_img = _tiny_image_critic()
        opt_img = recipe.make_critic_optimizer(d_img)
        pen_img = recipe.make_critic_penalty(opt_img, collect_stats=True)
        loss_fn = recipe.make_loss()
        x_r = torch.randn(8, 1, 8, 8, generator=data_rng)
        x_f = torch.randn(8, 1, 8, 8, generator=data_rng)
        ld = loss_fn.d_loss(d_img(x_r), d_img(x_f))
        p_img = pen_img(d_img, x_r, x_f)
        assert pen_img.last_stats["live_endpoints_reused"] is True
        (ld + p_img).backward()


def test_live_endpoint_reuse_matches_standalone_evaluation_bit_for_bit():
    """Reusing live d_loss logits on 4D image critics must match standalone evaluation bit-for-bit."""
    d1 = _tiny_image_critic()
    d2 = deepcopy(d1)
    recipe = get_recipe(
        "t2_secant",
        num_particles=32,
        z_dim=2,
        batch_size=8,
        total_steps=10,
        optimizer_family="formulation",
        optimizer_smoothing=0.0,
        optimizer_convolution="none",
        optimizer_convolution_rule="polar",
        optimizer_dense_profile="default",
        optimizer_svd_backend="native",
        training_precision="float32",
        constraint_geometry_mode="none",
    )
    opt1 = recipe.make_critic_optimizer(d1)
    opt2 = recipe.make_critic_optimizer(d2)
    assert isinstance(opt1, T2SecantCriticAdam)
    pen1 = recipe.make_critic_penalty(opt1, collect_stats=True)
    pen2 = recipe.make_critic_penalty(opt2, collect_stats=True)
    assert isinstance(pen1, T2SecantCriticPenalty)
    loss_fn = recipe.make_loss()
    assert isinstance(loss_fn, T2SecantGANLoss)

    rng = torch.Generator().manual_seed(7)
    x_real = torch.randn(8, 1, 8, 8, generator=rng)
    x_fake = torch.randn(8, 1, 8, 8, generator=rng) + 0.5

    # Path 1: normal training order (d_loss first -> live endpoint reuse)
    opt1.zero_grad()
    ld1 = loss_fn.d_loss(d1(x_real), d1(x_fake))
    p1 = pen1(d1, x_real, x_fake)
    assert pen1.last_stats["live_endpoints_reused"] is True
    (ld1 + p1).backward()

    # Path 2: standalone penalty evaluation + separate d_loss
    opt2.zero_grad()
    p2 = pen2(d2, x_real, x_fake)
    assert pen2.last_stats["live_endpoints_reused"] is False
    ld2 = loss_fn.d_loss(d2(x_real), d2(x_fake))
    (ld2 + p2).backward()

    assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-6)
    for param1, param2 in zip(d1.parameters(), d2.parameters()):
        assert torch.allclose(param1.grad, param2.grad, atol=1e-6, rtol=1e-6)


def test_input_noise_consistent_actual_coordinates():
    """With InputNoise(std > 0), both 4D image secants and 2D forward-tangent probes remain finite and consistent."""
    d_img = _tiny_image_critic()
    recipe = get_recipe("t2_secant", batch_size=8, num_particles=32, z_dim=2, total_steps=10)
    opt_img = recipe.make_critic_optimizer(d_img)
    pen_img = recipe.make_critic_penalty(opt_img, collect_stats=True)
    noisy_img = InputNoise(d_img, std=0.5, generator=torch.Generator().manual_seed(9))

    x_same = torch.zeros(8, 1, 8, 8)
    x_fake = x_same.clone()
    loss_d = recipe.make_loss().d_loss(noisy_img(x_same), noisy_img(x_fake))
    p_val = pen_img(noisy_img, x_same, x_fake)
    assert pen_img.last_stats["live_endpoints_reused"] is True
    assert pen_img.last_stats["max_secant_slope"] < 50.0
    (loss_d + p_val).backward()
    assert all(torch.isfinite(p.grad).all() for p in d_img.parameters() if p.grad is not None)

    _, d_mlp = _tiny_models()
    opt_mlp = recipe.make_critic_optimizer(d_mlp)
    pen_mlp = recipe.make_critic_penalty(opt_mlp, collect_stats=True)
    noisy_mlp = InputNoise(d_mlp, std=0.5, generator=torch.Generator().manual_seed(11))
    xm = torch.zeros(8, 2)
    ld_m = recipe.make_loss().d_loss(noisy_mlp(xm), noisy_mlp(xm.clone()))
    pm = pen_mlp(noisy_mlp, xm, xm.clone())
    assert pen_mlp.last_stats["max_secant_slope"] < 50.0
    (ld_m + pm).backward()
    assert all(torch.isfinite(p.grad).all() for p in d_mlp.parameters() if p.grad is not None)


def test_deliberate_damage_slope_bound_control():
    """Deliberate-damage control: t2_secant bounds actual autograd ||grad_x D||_2 <= kappa, while coeff=0 explodes."""
    from benchmarks.locked_shared import two_pole

    def run_arm(coeff: float):
        recipe = get_recipe(
            "t2_secant",
            reg_coeff=coeff,
            reg_kappa=1.0,
            optimizer_family="formulation",
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_dense_profile="default",
            optimizer_svd_backend="native",
            training_precision="float32",
            constraint_geometry_mode="none",
            lr=0.005,
            d_lr_mult=1.0,
        )
        critic = two_pole.HostCritic()
        particles = nn.Parameter(torch.zeros(64, 1))
        opt_d = recipe.make_critic_optimizer(critic)
        opt_p = torch.optim.Adam([particles], lr=0.005, betas=(0.0, 0.99))
        gan = recipe.make_loss()
        pen = recipe.make_critic_penalty(opt_d, collect_stats=True)
        real = two_pole.real_batch(64)
        for _ in range(80):
            opt_d.zero_grad()
            ld = gan.d_loss(critic(real), critic(particles.detach()))
            (ld + pen(critic, real, particles.detach())).backward()
            opt_d.step()
            opt_p.zero_grad()
            gan.g_loss(critic(particles), critic(real).detach()).backward()
            opt_p.step()
        xs = torch.cat([real, particles.detach()]).detach().requires_grad_(True)
        grad = torch.autograd.grad(critic(xs).sum(), xs, create_graph=False)[0]
        return float(particles.detach().abs().mean()), float(grad.abs().median())

    mean_active, grad_active = run_arm(1.0)
    _, grad_unreg = run_arm(0.0)
    assert mean_active >= 0.30
    assert grad_active <= 1.0, f"Expected actual critic grad_med <= 1.0 under t2_secant, got {grad_active}"
    assert grad_unreg > 1.05, f"Expected unregularized control to breach kappa=1.0, got {grad_unreg}"


def test_fresh_process_checkpoint_and_rng_continuation_and_tamper_rejection(tmp_path: Path):
    """Exact continuation across process restart + prototype/thermal activation, plus tamper rejection."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_fresh_process_checkpoint_and_tamper_test(tmp_path)
    finally:
        torch.set_num_threads(prev_threads)


def _run_fresh_process_checkpoint_and_tamper_test(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = tmp_path / "ckpt.pt"
    out_path = tmp_path / "resumed.pt"

    recipe = get_recipe(
        "t2_secant",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=20,
        prior_kind="mog",
        sigma_rel=0.05,
        standardize=False,
        constraint_geometry_mode="direction_blend",
    )

    def make_env():
        torch.manual_seed(0)
        g, d = _tiny_models()
        prior = recipe.make_prior()
        init.deterministic_orthogonal_(prior)
        tr = GANTrainer(recipe, g, d, prior=prior, seed=0, max_steps=20)
        data = torch.Generator().manual_seed(123)
        return tr, data

    # Uninterrupted 10-step reference
    tr_ref, data_ref = make_env()
    for step_idx in range(10):
        shift = 3.0 if step_idx >= 4 else -3.0
        tr_ref.step(shift + 0.1 * torch.randn(32, 2, generator=data_ref))
    ref_state = tr_ref.state_dict()

    # Run 6 steps (crossing shift + prototype revival), checkpoint to disk
    tr_a, data_a = make_env()
    for step_idx in range(6):
        shift = 3.0 if step_idx >= 4 else -3.0
        tr_a.step(shift + 0.1 * torch.randn(32, 2, generator=data_a))
    saved = {"trainer": tr_a.state_dict(), "data_rng": data_a.get_state()}
    torch.save(saved, ckpt_path)

    # Tamper rejection check
    tampered = deepcopy(saved["trainer"])
    tampered["optimizers"][1]["regularizer"]["record"]["t2_secant"]["chord_phase"] = 1.5
    tr_bad, _ = make_env()
    with pytest.raises(ValueError, match="phase"):
        tr_bad.opt_d.load_state_dict(tampered["optimizers"][1])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|phase"):
        tr_bad.load_state_dict(tampered)

    # Fresh-process continuation check (also verifies import path isolation!)
    script = f"""
import pathlib, torch
torch.set_num_threads(1)
import particlegan
from particlegan import GANTrainer, get_recipe, init
from torch import nn

expected_root = pathlib.Path({str(repo_root)!r}).resolve()
assert pathlib.Path(particlegan.__file__).resolve().parents[1] == expected_root

torch.manual_seed(999)
g = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
d = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
init.deterministic_orthogonal_(g)
init.deterministic_orthogonal_(d)
recipe = get_recipe(
    "t2_secant",
    num_particles=64, z_dim=2, batch_size=32, total_steps=20,
    prior_kind="mog", sigma_rel=0.05, standardize=False,
    constraint_geometry_mode="direction_blend",
)
prior = recipe.make_prior()
init.deterministic_orthogonal_(prior)
tr = GANTrainer(recipe, g, d, prior=prior, seed=0, max_steps=20)
payload = torch.load({str(ckpt_path)!r}, map_location="cpu", weights_only=True)
tr.load_state_dict(payload["trainer"])
data = torch.Generator()
data.set_state(payload["data_rng"])
for _ in range(4):
    tr.step(3.0 + 0.1 * torch.randn(32, 2, generator=data))
torch.save({{"trainer": tr.state_dict(), "data_rng": data.get_state()}}, {str(out_path)!r})
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    resumed = torch.load(out_path, map_location="cpu", weights_only=True)
    assert torch.equal(resumed["data_rng"], data_ref.get_state())
    for model_name in ("G", "D", "prior"):
        for k, v_ref in ref_state["models"][model_name].items():
            v_res = resumed["trainer"]["models"][model_name][k]
            if isinstance(v_ref, torch.Tensor):
                assert torch.equal(v_ref, v_res), f"Mismatch in {model_name}.{k}"
            else:
                assert v_ref == v_res, f"Mismatch in {model_name}.{k}"
    rec_ref = ref_state["optimizers"][1]["regularizer"]["record"]["t2_secant"]
    rec_res = resumed["trainer"]["optimizers"][1]["regularizer"]["record"]["t2_secant"]
    for key in ("ema_residual", "ema_data_scale", "thermal_scale", "chord_phase", "revival_phase", "revivals"):
        assert rec_ref[key] == rec_res[key], f"Record mismatch on {key}"
    assert torch.equal(rec_ref["proto_sum"], rec_res["proto_sum"])
    assert torch.equal(rec_ref["proto_count"], rec_res["proto_count"])


def test_fresh_process_continuation_across_thermal_cooling_shock_step_scale_and_sampling(tmp_path: Path):
    """Verify exact fresh-process continuation + sample parity across thermal cooling, step_scale < 1, shock reopening, and 4D live-reuse."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_fresh_process_activation_boundary_test(tmp_path)
    finally:
        torch.set_num_threads(prev_threads)


def _run_fresh_process_activation_boundary_test(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_1d = tmp_path / "ckpt_1d.pt"
    out_1d = tmp_path / "out_1d.pt"
    ckpt_img = tmp_path / "ckpt_img.pt"
    out_img = tmp_path / "out_img.pt"

    recipe_1d = get_recipe(
        "t2_secant",
        num_particles=64,
        z_dim=1,
        batch_size=64,
        total_steps=30,
        prior_kind="mog",
        sigma_rel=0.05,
        standardize=False,
        constraint_geometry_mode="direction_blend",
    )

    def make_1d():
        torch.manual_seed(0)
        g = nn.Linear(1, 1)
        d = nn.Sequential(nn.Linear(1, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        init.deterministic_orthogonal_(d)
        with torch.no_grad():
            g.weight.fill_(1.0)
            g.bias.zero_()
        prior = recipe_1d.make_prior()
        init.deterministic_orthogonal_(prior)
        tr = GANTrainer(recipe_1d, g, d, prior=prior, seed=0, max_steps=30)
        data = torch.Generator().manual_seed(321)
        return tr, data

    _, tmp_tr = make_1d()
    del tmp_tr
    init_tr, _ = make_1d()
    fixed_anchors = init_tr.prior.z.detach().clone()
    torch.save(fixed_anchors, tmp_path / "fixed_anchors.pt")
    del init_tr

    def draw_1d_batch(rng_obj, shift: float = 0.0):
        idx = torch.randint(0, fixed_anchors.shape[0], (64,), generator=rng_obj)
        return fixed_anchors[idx] + shift + 0.01 * torch.randn(64, 1, generator=rng_obj)

    # Uninterrupted 20-step 1D run: steps 0..5 match G(prior) closely (engaging _thermal_scale < 1 & _step_scale < 1),
    # steps 6..19 apply a +12.0 distribution shift (triggering shock re-opening & prototype revival).
    tr_ref, data_ref = make_1d()
    for s_idx in range(20):
        tr_ref.step(draw_1d_batch(data_ref, shift=12.0 if s_idx >= 6 else 0.0))
    ref_samples = tr_ref.sample(32)
    ref_state = tr_ref.state_dict()

    # Split run: checkpoint at step 6 while actively cooled (_thermal_scale < 1 and _step_scale < 1)
    tr_a, data_a = make_1d()
    for _ in range(6):
        tr_a.step(draw_1d_batch(data_a, shift=0.0))
    rec_mid = tr_a.opt_d.record
    assert rec_mid._thermal_scale < 1.0, f"Expected active thermal cooling < 1.0 at checkpoint, got {rec_mid._thermal_scale}"
    assert rec_mid._step_scale < 1.0, f"Expected active 1D step_scale < 1.0 at checkpoint, got {rec_mid._step_scale}"
    torch.save({"trainer": tr_a.state_dict(), "data_rng": data_a.get_state()}, ckpt_1d)

    # Also test 4D image T2SecantCriticAdam (formulation family) across checkpoint boundary
    recipe_img = get_recipe(
        "t2_secant",
        num_particles=32,
        z_dim=2,
        batch_size=8,
        total_steps=12,
        optimizer_family="formulation",
        optimizer_smoothing=0.0,
        optimizer_convolution="none",
        optimizer_convolution_rule="polar",
        optimizer_dense_profile="default",
        optimizer_svd_backend="native",
        training_precision="float32",
        constraint_geometry_mode="none",
    )

    def step_img(d_mod, opt_mod, pen_mod, loss_mod, rng_mod):
        xr = torch.randn(8, 1, 8, 8, generator=rng_mod)
        xf = torch.randn(8, 1, 8, 8, generator=rng_mod) + 0.25
        opt_mod.zero_grad(set_to_none=True)
        ld = loss_mod.d_loss(d_mod(xr), d_mod(xf))
        pv = pen_mod(d_mod, xr, xf)
        assert pen_mod.last_stats["live_endpoints_reused"] is True
        (ld + pv).backward()
        opt_mod.step()

    d_img_ref = _tiny_image_critic()
    d_img_a = deepcopy(d_img_ref)
    torch.save(d_img_ref.state_dict(), tmp_path / "init_d_img.pt")

    opt_img_ref = recipe_img.make_critic_optimizer(d_img_ref)
    pen_img_ref = recipe_img.make_critic_penalty(opt_img_ref, collect_stats=True)
    loss_img_ref = recipe_img.make_loss()
    rng_img_ref = torch.Generator().manual_seed(555)
    for _ in range(6):
        step_img(d_img_ref, opt_img_ref, pen_img_ref, loss_img_ref, rng_img_ref)

    opt_img_a = recipe_img.make_critic_optimizer(d_img_a)
    pen_img_a = recipe_img.make_critic_penalty(opt_img_a, collect_stats=True)
    loss_img_a = recipe_img.make_loss()
    rng_img_a = torch.Generator().manual_seed(555)
    for _ in range(3):
        step_img(d_img_a, opt_img_a, pen_img_a, loss_img_a, rng_img_a)
    torch.save({"d": d_img_a.state_dict(), "opt": opt_img_a.state_dict(), "rng": rng_img_a.get_state()}, ckpt_img)

    script = f"""
import pathlib, torch
torch.set_num_threads(1)
from torch import nn
import particlegan
from particlegan import GANTrainer, get_recipe, init

expected_root = pathlib.Path({str(repo_root)!r}).resolve()
assert pathlib.Path(particlegan.__file__).resolve().parents[1] == expected_root

# 1. Resume 1D cooled trainer in fresh process and cross shock re-opening + revival boundary
torch.manual_seed(777)
g = nn.Linear(1, 1)
d = nn.Sequential(nn.Linear(1, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
init.deterministic_orthogonal_(d)
with torch.no_grad():
    g.weight.fill_(1.0)
    g.bias.zero_()
recipe_1d = get_recipe(
    "t2_secant",
    num_particles=64, z_dim=1, batch_size=64, total_steps=30,
    prior_kind="mog", sigma_rel=0.05, standardize=False,
    constraint_geometry_mode="direction_blend",
)
prior = recipe_1d.make_prior()
init.deterministic_orthogonal_(prior)
tr = GANTrainer(recipe_1d, g, d, prior=prior, seed=0, max_steps=30)
p1 = torch.load({str(ckpt_1d)!r}, map_location="cpu", weights_only=True)
tr.load_state_dict(p1["trainer"])
assert tr.opt_d.record._thermal_scale < 1.0
assert tr.opt_d.record._step_scale < 1.0
data = torch.Generator()
data.set_state(p1["data_rng"])
fixed_anchors = torch.load({str(tmp_path / 'fixed_anchors.pt')!r}, map_location="cpu", weights_only=True)
shock_reopened = False
for _ in range(14):
    idx = torch.randint(0, fixed_anchors.shape[0], (64,), generator=data)
    batch = fixed_anchors[idx] + 12.0 + 0.01 * torch.randn(64, 1, generator=data)
    tr.step(batch)
    if tr.opt_d.record._thermal_scale == 1.0:
        shock_reopened = True
assert shock_reopened, "Expected shock guard to reopen _thermal_scale to 1.0"
assert tr.opt_d.record.revivals >= 1, "Expected prototype revival across distribution shift"
samples = tr.sample(32)
torch.save({{"trainer": tr.state_dict(), "samples": samples, "data_rng": data.get_state()}}, {str(out_1d)!r})

# 2. Resume 4D image T2SecantCriticAdam in fresh process
d_img = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(8 * 4 * 4, 1))
d_img.load_state_dict(torch.load({str(tmp_path / 'init_d_img.pt')!r}, map_location="cpu", weights_only=True))
recipe_img = get_recipe(
    "t2_secant",
    num_particles=32, z_dim=2, batch_size=8, total_steps=12,
    optimizer_family="formulation", optimizer_smoothing=0.0,
    optimizer_convolution="none", optimizer_convolution_rule="polar",
    optimizer_dense_profile="default", optimizer_svd_backend="native",
    training_precision="float32", constraint_geometry_mode="none",
)
opt_img = recipe_img.make_critic_optimizer(d_img)
pen_img = recipe_img.make_critic_penalty(opt_img, collect_stats=True)
loss_img = recipe_img.make_loss()
p_img = torch.load({str(ckpt_img)!r}, map_location="cpu", weights_only=True)
d_img.load_state_dict(p_img["d"])
opt_img.load_state_dict(p_img["opt"])
rng_img = torch.Generator()
rng_img.set_state(p_img["rng"])
for _ in range(3):
    xr = torch.randn(8, 1, 8, 8, generator=rng_img)
    xf = torch.randn(8, 1, 8, 8, generator=rng_img) + 0.25
    opt_img.zero_grad(set_to_none=True)
    ld = loss_img.d_loss(d_img(xr), d_img(xf))
    pv = pen_img(d_img, xr, xf)
    assert pen_img.last_stats["live_endpoints_reused"] is True
    (ld + pv).backward()
    opt_img.step()
torch.save({{"d": d_img.state_dict(), "opt": opt_img.state_dict(), "rng": rng_img.get_state()}}, {str(out_img)!r})
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    res_1d = torch.load(out_1d, map_location="cpu", weights_only=True)
    assert torch.equal(res_1d["data_rng"], data_ref.get_state())
    assert torch.equal(res_1d["samples"], ref_samples)
    for model_name in ("G", "D", "prior"):
        for k, v_ref in ref_state["models"][model_name].items():
            v_res = res_1d["trainer"]["models"][model_name][k]
            if isinstance(v_ref, torch.Tensor):
                assert torch.equal(v_ref, v_res), f"1D mismatch in {model_name}.{k}"
            else:
                assert v_ref == v_res, f"1D mismatch in {model_name}.{k}"
    rec1_ref = ref_state["optimizers"][1]["regularizer"]["record"]["t2_secant"]
    rec1_res = res_1d["trainer"]["optimizers"][1]["regularizer"]["record"]["t2_secant"]
    for key in ("ema_residual", "ema_data_scale", "thermal_scale", "step_scale", "chord_phase", "revival_phase", "revivals"):
        assert rec1_ref[key] == rec1_res[key], f"1D record mismatch on {key}"

    res_img = torch.load(out_img, map_location="cpu", weights_only=True)
    assert torch.equal(res_img["rng"], rng_img_ref.get_state())
    for k, v_ref in d_img_ref.state_dict().items():
        assert torch.equal(v_ref, res_img["d"][k]), f"4D D mismatch in {k}"
    rec_img_ref = opt_img_ref.record.state_dict()["t2_secant"]
    rec_img_res = next(
        v["record"]["t2_secant"] for v in res_img["opt"].values() if isinstance(v, dict) and "record" in v
    )
    for key in ("ema_residual", "ema_data_scale", "thermal_scale", "step_scale", "chord_phase"):
        assert rec_img_ref[key] == rec_img_res[key], f"4D record mismatch on {key}"


def test_clockfree_invariance_perturbations():
    """Same state + inputs must produce identical learning state across step_label, horizon, eval cadence."""
    recipe = get_recipe(
        "t2_secant",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=50,
        prior_kind="mog",
        sigma_rel=0.05,
        standardize=False,
    )

    def build():
        torch.manual_seed(0)
        g, d = _tiny_models()
        prior = recipe.make_prior()
        init.deterministic_orthogonal_(prior)
        return GANTrainer(recipe, g, d, prior=prior, seed=0, max_steps=200), torch.Generator().manual_seed(77)

    tr0, data0 = build()
    for _ in range(3):
        tr0.step(torch.randn(32, 2, generator=data0))
    init_ckpt = deepcopy(tr0.state_dict())
    init_data = data0.get_state()

    def run_branch(mode: str):
        tr, data = build()
        tr.load_state_dict(deepcopy(init_ckpt))
        data.set_state(init_data.clone())
        if mode == "step_label":
            tr.completed_steps += 25
        elif mode == "horizon":
            tr.recipe = tr.recipe.replace(total_steps=500)
            tr.policy.recipe = tr.recipe
        eval_rng = torch.Generator().manual_seed(999)
        for _ in range(4):
            if mode == "eval_cadence":
                tr.sample(16, generator=eval_rng)
            tr.step(torch.randn(32, 2, generator=data))
        return tr.state_dict()

    ref = run_branch("reference")
    for mode in ("step_label", "horizon", "eval_cadence"):
        cand = run_branch(mode)
        for model_name in ("G", "D", "prior"):
            for k in ref["models"][model_name]:
                v_r, v_c = ref["models"][model_name][k], cand["models"][model_name][k]
                if isinstance(v_r, torch.Tensor):
                    assert torch.equal(v_r, v_c)
                else:
                    assert v_r == v_c
