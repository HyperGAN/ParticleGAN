"""Unit, semantic, tamper, and fresh-process continuation tests for Theory 4A (T4Nash)."""
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
    HostileImbalancedGenerator,
    build_models_for_spec,
    imbalance_nash_oracle_controls,
    score_imbalance_nash_kind,
)
from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.t4_nash import (
    T4NashConstraintGeneratorAdam,
    T4NashGeneratorAdam,
    T4NashStepRecord,
    born_infeld_debiased_transport_loss,
    compute_two_player_coupling,
    feydy_debiased_monge_displacement,
    intrinsic_sinkhorn_coupling,
    qmc_monge_displacement,
)
from particlegan._t4_nash.loss import _protected_softplus_neg
from particlegan._t4_nash.optim import _adam_unit_direction

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
        "assert r.DEFAULT_RECIPE == 'unified_v12'\n"
        "fields = r.get_recipe_fields('t4_nash')\n"
        "assert fields['critic_formulation'] == 't4_nash'\n"
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


def test_zero_svd_eigh_pinv_enforcement_across_mlp_conv_and_vae_projection(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("T4Nash must execute with zero SVD / eigh / pinv calls!")

    for attr in ("svd", "svdvals", "eigh", "eigvalsh", "pinv"):
        monkeypatch.setattr(torch.linalg, attr, _forbidden)

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        recipe = get_recipe(
            "t4_nash", num_particles=32, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog"
        )
        g = SimpleMLPGenerator(z_dim=4, hidden_dim=32, n_hidden=2, out_dim=2)
        d = SimpleMLPDiscriminator(in_dim=2, hidden_dim=32, n_hidden=2, fourier=2)
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        prior = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05)
        init.deterministic_orthogonal_(prior, seed=2)
        trainer = GANTrainer(recipe, g, d, prior=prior, max_steps=20)
        data_rng = torch.Generator().manual_seed(7)
        for _ in range(3):
            real = torch.randn((32, 2), generator=data_rng)
            out = trainer.step(real)
            assert math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))

        # Verify ConvTranspose2d + Conv2d path also uses zero SVD.
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
        conv_trainer = GANTrainer(recipe, conv_g, conv_d, prior=conv_prior, max_steps=10)
        for _ in range(2):
            conv_trainer.step(torch.randn((32, 1, 8, 8), generator=data_rng))

        # Verify multi-objective constraint_geometry_backward on T4NashConstraintGeneratorAdam uses zero SVD/pinv,
        # while default T4NashGeneratorAdam omits bind_protected_losses so split-optimizer hosts (ae_gan_hold) run cleanly.
        assert not hasattr(T4NashGeneratorAdam([nn.Parameter(torch.ones(1))], lr=0.01), "bind_protected_losses")
        w = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt_g = T4NashConstraintGeneratorAdam([{"params": [w], "lr": 0.05, "role": "generator"}], betas=(0.0, 0.99), eps=1e-8)
        loss_a = (w[0] - 3.0).pow(2) + (w[1] + 1.0).pow(2)
        loss_b = (-2.0 * w[0] + w[1]).pow(2)
        total = loss_a + loss_b
        opt_g.zero_grad(set_to_none=True)
        constraint_geometry_backward(total, opt_g, (loss_a, loss_b))
        w_before = w.detach().clone()
        opt_g.step()
        assert opt_g.constraint_geometry_stats["steps"] == 1
        assert opt_g.constraint_geometry_stats["max_derivative_after"] <= 1e-7
        assert not torch.equal(w, w_before)
    finally:
        torch.set_num_threads(old)


def test_peak_rms_scale_invariance_and_stationary_equilibrium_settling():
    p1 = nn.Parameter(torch.zeros(64))
    p2 = nn.Parameter(torch.zeros(64))
    st1: dict = {}
    st2: dict = {}
    g_base = torch.linspace(-2.0, 2.0, 64)

    # 1. Exact degree-0 scale invariance across 10^8x constant gradient rescaling.
    u_small = _adam_unit_direction(p1, g_base * 1e-3, st1, (0.5, 0.99), 1e-12, False)
    u_huge = _adam_unit_direction(p2, g_base * 1e5, st2, (0.5, 0.99), 1e-12, False)
    assert torch.allclose(u_small, u_huge, atol=1e-5, rtol=1e-4)

    # 2. Stationary equilibrium contraction: when ||g_t|| -> 1e-4 * G_0 after warmup,
    # peak_rms prevents Adam's 0/0 amplification limit cycle.
    p_eq = nn.Parameter(torch.zeros(64))
    st_eq: dict = {}
    for _ in range(15):
        u_hot = _adam_unit_direction(p_eq, g_base, st_eq, (0.0, 0.5), 1e-12, False)
    for _ in range(40):
        u_cool = _adam_unit_direction(p_eq, g_base * 1e-4, st_eq, (0.0, 0.5), 1e-12, False)
    assert float(u_cool.norm().item()) < 0.02 * float(u_hot.norm().item())


def test_width_and_lr_pushforward_step_impedance_invariance_and_compressed_rescue():
    z = torch.randn((32, 4), generator=torch.Generator().manual_seed(12))
    target = torch.full((32, 2), 2.5)

    def _measure_step(hidden: int, lr: float, out_scale: float = 1.0) -> tuple[float, float]:
        gen = HostileImbalancedGenerator(
            z_dim=4, hidden=hidden, layers=3, out_dim=2, output_scale=out_scale, output_offset=[0.0, 0.0]
        )
        init.deterministic_orthogonal_(gen, seed=3)
        rec = T4NashStepRecord()
        rec.observe_transport(1.2, 2.5)
        rec._enable_secant_probe = True
        from particlegan._t4_nash.record import _attach_t4_generator_hook

        _attach_t4_generator_hook(gen, rec)
        opt = T4NashGeneratorAdam(
            [{"params": list(gen.parameters()), "lr": lr, "role": "generator"}],
            generator_module=gen,
            betas=(0.0, 0.99),
            eps=1e-8,
        )
        opt._t4_record = rec
        with torch.no_grad():
            y0 = gen(z).clone()
        loss = (gen(z) - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            y1 = gen(z)
        disp = float((y1 - y0).pow(2).mean().sqrt().item())
        return disp, opt.last_step_gain

    disp_narrow, _ = _measure_step(hidden=16, lr=0.0135)
    disp_wide, _ = _measure_step(hidden=512, lr=0.0135)
    disp_huge_lr, gain_huge_lr = _measure_step(hidden=512, lr=1.35)  # 100x LR overshoot!
    disp_compressed_raw_check, gain_compressed = _measure_step(hidden=32, lr=0.0135, out_scale=0.01)

    # 32x width asymmetry & 100x LR asymmetry remain strictly bounded inside the physical trust band!
    assert 0.01 <= disp_narrow <= 0.65
    assert 0.01 <= disp_wide <= 0.65
    assert disp_huge_lr <= 0.65 and gain_huge_lr < 0.25
    assert gain_compressed > 1.5 and disp_compressed_raw_check > 0.005


def test_two_player_calibration_coupling_and_protected_rapidity_nonvanishing():
    # Extreme hostile upside-down logit margin delta = -50 must retain O(1) gradient (no FP32 underflow).
    delta = torch.tensor([-50.0, -45.0, -55.0], requires_grad=True)
    val = _protected_softplus_neg(delta).mean()
    val.backward()
    assert delta.grad is not None and bool(torch.isfinite(delta.grad).all())
    assert float(delta.grad.abs().mean().item()) > 0.15

    # Upside-down critic far from manifold -> c_d damped to floor 0.005.
    c_inv, gd_inv, gg_inv = compute_two_player_coupling(ema_margin=-25.0, ema_residual=1.4)
    assert c_inv == pytest.approx(0.005)
    assert gd_inv == pytest.approx(1.0)
    assert 0.55 <= gg_inv <= 0.60

    # Overpowered runaway critic -> gamma_d damped smoothly.
    c_run, gd_run, _ = compute_two_player_coupling(ema_margin=12.0, ema_residual=0.5)
    assert c_run == pytest.approx(1.0)
    assert gd_run < 0.20

    # Calibrated Nash saddle point (margin -> 0, residual -> 0) -> c_d == 1.0, gamma_d == 1.0.
    c_eq, gd_eq, gg_eq = compute_two_player_coupling(ema_margin=0.0, ema_residual=0.02)
    assert c_eq == pytest.approx(1.0) and gd_eq == pytest.approx(1.0) and gg_eq == pytest.approx(1.0)


def test_feydy_zero_bias_and_qmc_monge_collapsed_origin_mode_splitting():
    anchors = torch.tensor([[-3.0, 0.0], [3.0, 0.0], [0.0, -3.0], [0.0, 3.0]])
    weights = torch.full((4,), 0.25)

    # Feydy self-debiasing vanishes identically when x_fake == anchors.
    disp_zero, _ = feydy_debiased_monge_displacement(anchors, weights, anchors.clone(), sinkhorn_eps=0.20)
    assert float(disp_zero.abs().max().item()) < 1e-5

    # QMC Monge splits collapsed origin x_fake == 0 cleanly onto the 4 modes without RNG draws!
    rng_before = torch.get_rng_state()
    collapsed = torch.zeros((64, 2), requires_grad=True)
    v_qmc, _ = qmc_monge_displacement(anchors, weights, collapsed.detach(), sinkhorn_eps=0.15, qmc_phase=0.1)
    assert torch.equal(rng_before, torch.get_rng_state())
    # Every particle's displacement norm must be 3.0 (pointing to an actual mode, NEVER 0.0 centroid!).
    norms = v_qmc.norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, 3.0), atol=1e-4)

    loss_bi = born_infeld_debiased_transport_loss(
        collapsed, anchors.repeat(16, 1), prototypes=anchors, proto_weights=weights, ema_residual=1.0
    )
    loss_bi.backward()
    assert collapsed.grad is not None and float((collapsed.grad.norm(dim=1) * 64.0).min().item()) > 0.50

    # Degenerate point-mass target (rr_med == 0) shrinks residual linearly with ||x_fake||.
    pt_proto = torch.zeros((8, 3))
    pt_w = torch.full((8,), 0.125)
    _, _, res_far, _ = intrinsic_sinkhorn_coupling(pt_proto, pt_w, torch.full((32, 3), 0.5), x_real=torch.zeros((32, 3)))
    _, _, res_near, _ = intrinsic_sinkhorn_coupling(pt_proto, pt_w, torch.full((32, 3), 0.01), x_real=torch.zeros((32, 3)))
    assert float(res_near.item()) < 0.05 * float(res_far.item())


def test_fresh_process_checkpoint_continuation_across_active_mechanisms_and_tamper_rejection(tmp_path):
    ckpt_path = tmp_path / "t4_mid.pt"
    out_direct = tmp_path / "direct.pt"
    out_resumed = tmp_path / "resumed.pt"

    # Run 8 steps directly AND save checkpoint at step 4 where hostile margin, revivals,
    # secant step gain, and peak_rms are all actively engaged, then resume in a fresh Python process!
    script_part1 = f"""
import torch
from benchmarks.toy_audit.api_imbalance_nash import HOST_DEFINITIONS, build_models_for_spec, sample_target_for_kind
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior

torch.set_num_threads(1)
spec = HOST_DEFINITIONS["nash_overpowered_critic_ring16"]
recipe = get_recipe("t4_nash", num_particles=64, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog")

def make_env():
    g, d = build_models_for_spec(spec, z_dim=4)
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(d, seed=1)
    p = MoGParticlePrior(num_particles=64, z_dim=4, sigma=0.10)
    init.deterministic_orthogonal_(p, seed=2)
    tr = GANTrainer(
        recipe, g, d, prior=p, max_steps=8,
        latent_generator=torch.Generator().manual_seed(101),
        prior_noise_generator=torch.Generator().manual_seed(102),
        noise_generator=torch.Generator().manual_seed(103),
        input_noise_generator=torch.Generator().manual_seed(104),
    )
    return tr

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
recipe = get_recipe("t4_nash", num_particles=64, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog")
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

    # Fail-closed checkpoint tamper tests: missing t4_nash_opt, corrupt peak_rms, or corrupt record schema must raise ValueError.
    spec = HOST_DEFINITIONS["nash_overpowered_critic_ring16"]
    recipe = get_recipe("t4_nash", num_particles=64, z_dim=4, batch_size=32, total_steps=20, prior_kind="mog")
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
    del bad_opt["optimizers"][1]["t4_nash_opt"]
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        fresh.load_state_dict(bad_opt)
    with pytest.raises(ValueError, match="t4_nash_opt"):
        fresh.opt_d.load_state_dict(bad_opt["optimizers"][1])

    bad_rec = deepcopy(valid_ckpt)
    bad_rec["optimizers"][1]["regularizer"]["record"]["t4_nash"]["calibration_gate"] = 1.5
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        fresh.load_state_dict(bad_rec)
    with pytest.raises(ValueError, match="calibration_gate"):
        fresh.opt_d.load_state_dict(bad_rec["optimizers"][1])

    bad_peak = deepcopy(valid_ckpt)
    first_state = next(iter(bad_peak["optimizers"][0]["state"].values()))
    del first_state["peak_rms"]
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        fresh.load_state_dict(bad_peak)
    with pytest.raises(ValueError, match="peak_rms"):
        fresh.opt_g.load_state_dict(bad_peak["optimizers"][0])


def test_imbalance_nash_stretch_oracle_damage_preflight_and_grading_contracts(tmp_path):
    view = load_view(ROOT, "t4_imbalance_nash_stretch")
    assert [a["task"] for a in view["assignments"]] == list(IMBALANCE_TASKS)
    tasks = load_tasks(ROOT)
    for name in IMBALANCE_TASKS:
        assert name in tasks

    controls = imbalance_nash_oracle_controls()
    assert len(controls) == 8
    for cname, item in controls.items():
        assert item["passed"] is item["expected_pass"], f"{cname}: {item}"

    # Forbidden prior / code-law tamper checks fail preflight and grade INVALID.
    bad_task = _load_task("nash_unequal_width_hostile_imbalance")
    bad_task["execution"]["prior"] = {"kind": "mog", "sigma": 0.50, "standardize": False}
    cand = {"recipe_preset": "t4_nash", "claim_contract": {"sampling_law": "task_declared"}}
    assert adapter_preflight(bad_task, cand, root=ROOT)

    # Short software prefixes render actual-training GIFs, maintain zero RNG deviations, and grade INCOMPLETE.
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original = _load_task("nash_unequal_width_hostile_imbalance")
        prefix = deepcopy(original)
        prefix["id"] += "-software-prefix"
        prefix["execution"]["steps"] = 3
        prefix["execution"]["host_definition"]["steps"] = 3
        prefix["execution"]["host_definition"]["target_spec"]["steps"] = 3
        prefix["evaluation"]["observations"] = 3
        req = {"candidate": cand, "protocol": {"seed": 0}, "tasks": {prefix["id"]: prefix}}
        raw = run_task(req, {"task_id": prefix["id"]}, tmp_path / "prefix_run", "cpu")
        assert raw["cost"]["completed_steps"] == 3
        assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
        assert raw["evidence"]["guards"]["all_finite"] is True
        assert (tmp_path / "prefix_run" / "goal.gif").stat().st_size > 100
        assert grade_result(original, raw)["gate_status"] == "INCOMPLETE"
        assert grade_result(bad_task, raw)["gate_status"] == "INVALID"

        missing_contract = deepcopy(raw)
        del missing_contract["evidence"]["imbalance_nash_contract"]
        assert grade_result(original, missing_contract)["gate_status"] == "INCOMPLETE"

        tampered_contract = deepcopy(raw)
        tampered_contract["evidence"]["imbalance_nash_contract"]["generator_output_scale"] = 1.0
        assert grade_result(original, tampered_contract)["gate_status"] == "INVALID"

        # Deliberate model damage check on live scorer.
        damaged_pts = torch.full((4096, 2), 50.0)
        damaged = score_imbalance_nash_kind("nash_unequal_width_hostile_imbalance", damaged_pts)
        assert damaged["passed"] is False and damaged["metrics"]["hq"] == 0.0
    finally:
        torch.set_num_threads(old)
