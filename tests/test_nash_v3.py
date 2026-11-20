"""Unit, semantic, tamper, 0-forbidden-op, clockfree, and fresh-process continuation tests for T4NashV3."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.api_imbalance_nash import (
    HOST_DEFINITIONS,
    build_models_for_spec,
)
from experiments.forge.adapters import adapter_preflight
from experiments.forge.state import state_digest
from experiments.forge.views import load_tasks
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init
from particlegan._nash_v3.loss import _protected_softplus_neg
from particlegan._nash_v3.optim import _adam_unit_direction
from particlegan.nash_v3 import (
    NashV3ConstraintGeneratorAdam,
    NashV3GeneratorAdam,
    pullback_covariance_isometry_loss,
    quantile_1d_monge_displacement,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward

ROOT = Path(__file__).resolve().parents[1]
IMBALANCE_TASKS = (
    "nash_overpowered_critic_ring16",
    "nash_overpowered_generator_ring16",
    "nash_unequal_width_hostile_imbalance",
)


def _load_task(name: str) -> dict:
    return json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())


def test_recipes_import_order_and_default_preserved_in_fresh_process():
    recipes_path = ROOT / "particlegan/recipes.py"
    code = (
        "import importlib.util, sys\n"
        "class BlockTorch:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "            raise AssertionError(f'Imported Torch: {fullname}')\n"
        "sys.meta_path.insert(0, BlockTorch())\n"
        f"spec = importlib.util.spec_from_file_location('standalone_recipes', {str(recipes_path)!r})\n"
        "r = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = r\n"
        "spec.loader.exec_module(r)\n"
        "assert 'torch' not in sys.modules, 'recipes.py must remain torch-free on import'\n"
        "assert r.DEFAULT_RECIPE == 'empf'\n"
        "fields = r.get_recipe_fields('t4_nash_v3')\n"
        "assert fields['critic_formulation'] == 't4_nash_v3'\n"
        "assert fields['optimizer_family'] == 'formulation'\n"
        "assert fields['reg_arm'] is None\n"
        "assert len(fields) == 179\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_strict_zero_svd_eigh_pinv_ns_autograd_grad_enforcement(monkeypatch):
    """Monkeypatch-enforce strictly 0 SVD, 0 eigh, 0 pinv, 0 Newton-Schulz, and 0 torch.autograd.grad calls."""
    import particlegan.optim.dualnorm as dualnorm_mod

    def _forbid(name):
        def _boom(*args, **kwargs):
            raise AssertionError(f"T4NashV3 called forbidden op: {name}")
        return _boom

    for attr in ("svd", "svdvals", "eigh", "eigvalsh", "pinv"):
        monkeypatch.setattr(torch.linalg, attr, _forbid(f"torch.linalg.{attr}"))
    for attr in ("svd", "pinverse"):
        if hasattr(torch, attr):
            monkeypatch.setattr(torch, attr, _forbid(f"torch.{attr}"))
    monkeypatch.setattr(torch.autograd, "grad", _forbid("torch.autograd.grad"))
    for attr in dir(dualnorm_mod):
        if "newton_schulz" in attr.lower() or "polar" in attr.lower():
            obj = getattr(dualnorm_mod, attr)
            if callable(obj):
                monkeypatch.setattr(dualnorm_mod, attr, _forbid(f"dualnorm.{attr}"))

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # 1. 1D continuous MLP path (exact 1D quantile Monge + 1D global covariance isometry + fwAD critic)
        recipe_1d = get_recipe(
            "t4_nash_v3", num_particles=32, z_dim=2, batch_size=32, total_steps=20, prior_kind="mog"
        )
        g1 = SimpleMLPGenerator(z_dim=2, hidden_dim=32, n_hidden=2, out_dim=1)
        d1 = SimpleMLPDiscriminator(in_dim=1, hidden_dim=32, n_hidden=2, fourier=2)
        p1 = MoGParticlePrior(num_particles=32, z_dim=2, sigma=0.05)
        init.deterministic_orthogonal_(g1, seed=0)
        init.deterministic_orthogonal_(d1, seed=1)
        init.deterministic_orthogonal_(p1, seed=2)
        tr1 = GANTrainer(recipe_1d, g1, d1, prior=p1, max_steps=20)
        rng = torch.Generator().manual_seed(7)
        for _ in range(3):
            out = tr1.step(2.0 + 0.5 * torch.randn((32, 1), generator=rng))
            assert math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))

        # 2. 2D continuous MLP path (streaming prototype pullback covariance isometry + fwAD critic + secant probe)
        recipe_2d = get_recipe(
            "t4_nash_v3", num_particles=32, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog"
        )
        g2 = SimpleMLPGenerator(z_dim=4, hidden_dim=32, n_hidden=2, out_dim=2)
        d2 = SimpleMLPDiscriminator(in_dim=2, hidden_dim=32, n_hidden=2, fourier=2)
        p2 = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05)
        init.deterministic_orthogonal_(g2, seed=0)
        init.deterministic_orthogonal_(d2, seed=1)
        init.deterministic_orthogonal_(p2, seed=2)
        tr2 = GANTrainer(recipe_2d, g2, d2, prior=p2, max_steps=20)
        for _ in range(3):
            out = tr2.step(torch.randn((32, 2), generator=rng))
            assert math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))

        # 3. High-D discrete simplex path (dim=170, Murmur32 Rademacher + local k-NN Hutchinson fwAD)
        recipe_disc = get_recipe(
            "t4_nash_v3", num_particles=16, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog"
        )
        gd = SimpleMLPGenerator(z_dim=4, hidden_dim=32, n_hidden=1, out_dim=170)
        dd = SimpleMLPDiscriminator(in_dim=170, hidden_dim=32, n_hidden=1, fourier=0)
        pd = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05)
        trd = GANTrainer(recipe_disc, gd, dd, prior=pd, max_steps=10)
        for _ in range(2):
            out = trd.step(torch.softmax(torch.randn((32, 170), generator=rng), dim=-1))
            assert math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))

        # 4. ConvTranspose2d + Conv2d image path (live-endpoint reuse + 1st-order micro-secant envelope)
        conv_g = nn.Sequential(
            nn.Unflatten(1, (4, 1, 1)),
            nn.ConvTranspose2d(4, 8, 4, stride=1, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
        )
        conv_d = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 4, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )
        conv_prior = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05)
        conv_trainer = GANTrainer(recipe_2d, conv_g, conv_d, prior=conv_prior, max_steps=10)
        for _ in range(2):
            out = conv_trainer.step(torch.randn((32, 1, 8, 8), generator=rng))
            assert math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))

        # 5. Multi-objective constraint_geometry_backward with ZERO torch.autograd.grad and ZERO pinv/svd
        assert not hasattr(NashV3GeneratorAdam([nn.Parameter(torch.ones(1))], lr=0.01), "bind_protected_losses")
        w = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt_g = NashV3ConstraintGeneratorAdam(
            [{"params": [w], "lr": 0.05, "role": "generator"}], betas=(0.0, 0.99), eps=1e-8
        )
        loss_a = (w[0] - 3.0).pow(2) + (w[1] + 1.0).pow(2)
        loss_b = (-2.0 * w[0] + w[1]).pow(2)
        total = loss_a + loss_b
        opt_g.zero_grad(set_to_none=True)
        constraint_geometry_backward(total, opt_g, (loss_a, loss_b))
        opt_g.step()
        assert opt_g.constraint_geometry_stats["steps"] == 1
        assert opt_g.constraint_geometry_stats["max_derivative_after"] <= 1e-6
    finally:
        torch.set_num_threads(old)


def test_pullback_covariance_isometry_zero_at_equilibrium_and_restores_pinch():
    rng = torch.Generator().manual_seed(42)
    centers = torch.tensor([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
    ids = torch.arange(128) % 4
    x_real = centers[ids] + 0.10 * torch.randn((128, 2), generator=rng)

    # Exact zero at equilibrium (permuted copy of real data) for any eps > 0!
    perm = torch.randperm(128, generator=rng)
    x_eq = x_real[perm].clone().requires_grad_(True)
    loss_eq = pullback_covariance_isometry_loss(x_eq, x_real, prototypes=centers, ema_residual=0.05)
    assert float(loss_eq.item()) < 1e-5, f"Expected exact zero at equilibrium, got {loss_eq.item()}"

    # Pinched 2D clusters (y-variance compressed to 0.075x) produce positive loss & outward y-restoring gradient!
    noise = 0.10 * torch.randn((128, 2), generator=rng)
    pinched_noise = torch.stack([noise[:, 0], 0.25 * noise[:, 1]], dim=1)
    x_pinch = (centers[ids] + pinched_noise).clone().requires_grad_(True)
    loss_pinch = pullback_covariance_isometry_loss(x_pinch, x_real, prototypes=centers, ema_residual=0.05)
    assert float(loss_pinch.item()) > 0.01
    loss_pinch.backward()
    # Gradient along pinched y-axis must point inward so -grad pushes samples OUTWARD to restore variance!
    corr_y = float((x_pinch.grad[:, 1] * pinched_noise[:, 1]).mean().item())
    assert corr_y < -1e-5, f"Expected negative grad*offset correlation to restore variance, got {corr_y}"

    # Zero-variance degenerate point mass (zero_tanh_smoke): must evaluate to exact zero.
    x_zero = torch.zeros((64, 2))
    x_f = torch.randn((64, 2), generator=rng, requires_grad=True)
    loss_zero = pullback_covariance_isometry_loss(x_f, x_zero, ema_residual=0.05)
    assert float(loss_zero.item()) == 0.0


def test_quantile_1d_monge_protected_softplus_and_peak_rms_homogeneity():
    rng = torch.Generator().manual_seed(11)
    x_r = 2.0 + 0.5 * torch.randn((64, 1), generator=rng)
    perm = torch.randperm(64, generator=rng)
    disp, sigma_ref = quantile_1d_monge_displacement(x_r, x_r[perm])
    assert float(disp.abs().max().item()) < 1e-6
    assert abs(float(sigma_ref.item()) - float(x_r.std().item())) < 1e-5

    # Protected softplus stays non-vanishing at -35 logits (hostile init).
    d_hostile = torch.full((32,), -35.0, requires_grad=True)
    loss_h = _protected_softplus_neg(d_hostile, tau_logit=3.5).mean()
    loss_h.backward()
    assert abs(float(d_hostile.grad.mean().item())) > 1e-3

    # Degree-0 homogeneity of peak_rms Adam direction under 1e-3x and 1e3x gradient scaling.
    p = torch.zeros(16, 16)
    g = torch.randn(16, 16, generator=rng)
    st1, st2 = {}, {}
    u1 = _adam_unit_direction(p, g * 1e-3, st1, (0.0, 0.99), 1e-12, False)
    u2 = _adam_unit_direction(p, g * 1e3, st2, (0.0, 0.99), 1e-12, False)
    assert torch.allclose(u1, u2, atol=1e-5, rtol=1e-4)

    # Settling contraction when gradient RMS drops to 1e-3 of peak_rms.
    u_settle = _adam_unit_direction(p, g * 1e-3, st2, (0.0, 0.99), 1e-12, False)
    assert float(u_settle.norm().item()) < 0.05 * float(u2.norm().item())


def test_bit_exact_fresh_process_continuation_across_active_mechanism_boundaries(tmp_path):
    ckpt_path = tmp_path / "nash_v3_ckpt.pt"
    out_direct = tmp_path / "nash_v3_direct.pt"
    out_resumed = tmp_path / "nash_v3_resumed.pt"

    script_part1 = f"""
import torch
from benchmarks.toy_audit.api_imbalance_nash import HOST_DEFINITIONS, build_models_for_spec, sample_target_for_kind
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior

torch.set_num_threads(1)
spec = HOST_DEFINITIONS["nash_overpowered_critic_ring16"]
recipe = get_recipe("t4_nash_v3", num_particles=64, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog")

def make_env():
    g, d = build_models_for_spec(spec, z_dim=4)
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(d, seed=1)
    p = MoGParticlePrior(num_particles=64, z_dim=4, sigma=0.10)
    init.deterministic_orthogonal_(p, seed=2)
    return GANTrainer(
        recipe, g, d, prior=p, max_steps=8,
        latent_generator=torch.Generator().manual_seed(11),
        prior_noise_generator=torch.Generator().manual_seed(12),
        noise_generator=torch.Generator().manual_seed(13),
        input_noise_generator=torch.Generator().manual_seed(14),
    )

data_rng1 = torch.Generator().manual_seed(555)
batches = [sample_target_for_kind("nash_overpowered_critic_ring16", 32, data_rng1, completed_steps=s) for s in range(8)]

direct = make_env()
for s in range(8):
    direct.step(batches[s])
torch.save({{"g": direct.G.state_dict(), "d": direct.D.state_dict(), "z": unwrap_prior(direct.prior).z.detach().clone()}}, {str(out_direct)!r})

part1 = make_env()
for s in range(4):
    part1.step(batches[s])
rec = part1.opt_d.record
assert rec.initialized and rec.margin_initialized and rec.proto_initialized
assert rec.calibration_gate < 1.0 or rec.d_trust_scale < 1.0 or part1.opt_g.last_step_gain != 1.0
torch.save(part1.state_dict(), {str(ckpt_path)!r})
"""
    p1 = subprocess.run(
        [sys.executable, "-c", script_part1],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert p1.returncode == 0, p1.stderr

    script_part2 = f"""
import torch
from benchmarks.toy_audit.api_imbalance_nash import HOST_DEFINITIONS, build_models_for_spec, sample_target_for_kind
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior

torch.set_num_threads(1)
spec = HOST_DEFINITIONS["nash_overpowered_critic_ring16"]
recipe = get_recipe("t4_nash_v3", num_particles=64, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog")
g, d = build_models_for_spec(spec, z_dim=4)
init.deterministic_orthogonal_(g, seed=99)
init.deterministic_orthogonal_(d, seed=98)
p = MoGParticlePrior(num_particles=64, z_dim=4, sigma=0.10)
init.deterministic_orthogonal_(p, seed=97)
resumed = GANTrainer(
    recipe, g, d, prior=p, max_steps=8,
    latent_generator=torch.Generator().manual_seed(901),
    prior_noise_generator=torch.Generator().manual_seed(902),
    noise_generator=torch.Generator().manual_seed(903),
    input_noise_generator=torch.Generator().manual_seed(904),
)
saved = torch.load({str(ckpt_path)!r}, weights_only=False)
resumed.load_state_dict(saved)

data_rng = torch.Generator().manual_seed(555)
batches = [sample_target_for_kind("nash_overpowered_critic_ring16", 32, data_rng, completed_steps=s) for s in range(8)]
for s in range(4, 8):
    resumed.step(batches[s])
torch.save({{"g": resumed.G.state_dict(), "d": resumed.D.state_dict(), "z": unwrap_prior(resumed.prior).z.detach().clone()}}, {str(out_resumed)!r})
"""
    p2 = subprocess.run(
        [sys.executable, "-c", script_part2],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert p2.returncode == 0, p2.stderr

    d_res = torch.load(out_direct, weights_only=False)
    r_res = torch.load(out_resumed, weights_only=False)
    assert state_digest(d_res["g"]) == state_digest(r_res["g"])
    assert state_digest(d_res["d"]) == state_digest(r_res["d"])
    assert torch.equal(d_res["z"], r_res["z"])

    # Fail-closed checkpoint tamper checks: reject corrupt/old/missing T4NashV3 state.
    spec = HOST_DEFINITIONS["nash_overpowered_critic_ring16"]
    recipe = get_recipe("t4_nash_v3", num_particles=64, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog")
    g, d = build_models_for_spec(spec, z_dim=4)
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(d, seed=1)
    p = MoGParticlePrior(num_particles=64, z_dim=4, sigma=0.10)
    init.deterministic_orthogonal_(p, seed=2)
    fresh = GANTrainer(
        recipe, g, d, prior=p, max_steps=8,
        latent_generator=torch.Generator().manual_seed(1),
        prior_noise_generator=torch.Generator().manual_seed(2),
        noise_generator=torch.Generator().manual_seed(3),
        input_noise_generator=torch.Generator().manual_seed(4),
    )
    valid_ckpt = torch.load(ckpt_path, weights_only=False)

    bad_opt = deepcopy(valid_ckpt)
    del bad_opt["optimizers"][1]["t4_nash_v3_opt"]
    with pytest.raises(ValueError, match="t4_nash_v3_opt"):
        fresh.opt_d.load_state_dict(bad_opt["optimizers"][1])

    bad_form = deepcopy(valid_ckpt)
    bad_form["optimizers"][1]["regularizer"]["record"]["formulation"] = "t4_nash"
    with pytest.raises(ValueError, match="t4_nash_v3"):
        fresh.opt_d.load_state_dict(bad_form["optimizers"][1])

    bad_peak = deepcopy(valid_ckpt)
    first_state = next(iter(bad_peak["optimizers"][0]["state"].values()))
    del first_state["peak_rms"]
    with pytest.raises(ValueError, match="peak_rms"):
        fresh.opt_g.load_state_dict(bad_peak["optimizers"][0])


def test_clockfree_step_label_invariance_and_preflight():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        r_a = get_recipe("t4_nash_v3", num_particles=32, z_dim=4, batch_size=32, total_steps=50, prior_kind="mog")
        r_b = get_recipe("t4_nash_v3", num_particles=32, z_dim=4, batch_size=32, total_steps=5000, prior_kind="mog")
        rng_data = torch.Generator().manual_seed(99)
        batches = [torch.randn((32, 2), generator=rng_data) for _ in range(4)]

        outs = []
        for rec in (r_a, r_b):
            g = SimpleMLPGenerator(z_dim=4, hidden_dim=32, n_hidden=2, out_dim=2)
            d = SimpleMLPDiscriminator(in_dim=2, hidden_dim=32, n_hidden=2, fourier=2)
            p = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05)
            init.deterministic_orthogonal_(g, seed=0)
            init.deterministic_orthogonal_(d, seed=1)
            init.deterministic_orthogonal_(p, seed=2)
            tr = GANTrainer(
                rec, g, d, prior=p, max_steps=10,
                latent_generator=torch.Generator().manual_seed(5),
                prior_noise_generator=torch.Generator().manual_seed(6),
                noise_generator=torch.Generator().manual_seed(7),
                input_noise_generator=torch.Generator().manual_seed(8),
            )
            for b in batches:
                tr.step(b)
            outs.append((state_digest(tr.G.state_dict()), state_digest(tr.D.state_dict())))
        assert outs[0] == outs[1], "T4NashV3 must be strictly horizon/clock invariant!"

        tasks = load_tasks(ROOT)
        cand = {"recipe_preset": "t4_nash_v3", "claim_contract": {"sampling_law": "task_declared"}}
        for tid in (*IMBALANCE_TASKS, "gaussian1d_smoke", "ring16_acquisition", "dcgan_performance_r256"):
            assert not adapter_preflight(tasks[tid], cand, root=ROOT), f"Preflight failed on {tid}"
    finally:
        torch.set_num_threads(old)
