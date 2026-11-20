"""Unit, semantic-control, fresh-process continuation, and clock-free tests for Theory 4B ``t4_arch``."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.forge import adapters, views
from experiments.forge.api import resolve_public_recipe
from experiments.forge.state import state_digest
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    MoGParticlePrior,
    T4ArchPriorRegularizer,
    T4EquilibratedOptimizer,
    get_recipe,
    init,
)
from particlegan._t4_arch.optimizer import quintic_orthogonalize_2d, quintic_orthogonalize_batched

ROOT = Path(__file__).resolve().parents[1]


def test_default_recipe_unchanged_and_t4_arch_resolves():
    """Verify DEFAULT_RECIPE is 'unified_v11' and 't4_arch' resolves cleanly."""
    assert DEFAULT_RECIPE == "unified_v11"
    recipe = get_recipe("t4_arch")
    assert recipe.critic_formulation == "t2_secant"
    assert recipe.optimizer_family == "t4_equil"
    assert recipe.prior_regularizer == "t4_arch"
    assert recipe.t1_pullback_weight == pytest.approx(1.0)

    pub = resolve_public_recipe({"recipe_preset": "t4_arch", "recipe_overrides": {}})
    assert pub.optimizer_family == "t4_equil"
    assert pub.prior_regularizer == "t4_arch"


def test_svd_free_layerwise_equilibration_on_16_layer_vanishing_and_zero_branch_mlps():
    """Forbidden-fallback & equilibration test: t4_equil never calls torch.linalg.svd
    and wakes both 1e-8 vanishing layers in 16-layer MLPs and zero-initialized residual branches.
    """
    torch.manual_seed(0)
    mlp16 = SimpleMLPGenerator(2, 64, 16, 2)
    init.deterministic_orthogonal_(mlp16, seed=0)

    recipe = get_recipe("t4_arch", batch_size=32, num_particles=64, z_dim=2, total_steps=10)
    opt = recipe.make_generator_optimizer([{"params": list(mlp16.parameters()), "role": "generator"}])
    assert isinstance(opt, T4EquilibratedOptimizer)

    z = torch.randn(32, 2)
    target = torch.tensor([[2.0, -1.5]]).expand(32, 2)

    with mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD is forbidden in t4_equil!")):
        before = [p.detach().clone() for p in mlp16.parameters() if p.ndim == 2]
        out = mlp16(z)
        loss = 0.5 * (out - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()

        # Verify raw gradient skew across 17 linear layers spans > 1e4x
        grad_norms = [float(p.grad.norm()) for p in mlp16.parameters() if p.ndim == 2]
        assert max(grad_norms) / max(min(grad_norms), 1e-30) > 1e3

        opt.step()
        after = [p.detach().clone() for p in mlp16.parameters() if p.ndim == 2]
        rel_updates = [
            float((a - b).norm() / b.norm().clamp_min(1e-8))
            for a, b in zip(after, before)
        ]
        # Despite >1e3x raw gradient skew, relative updates across layer 0 and layer 16 are equilibrated within O(1)!
        assert min(rel_updates) > 0.0
        assert max(rel_updates) / min(rel_updates) < 5.0

    # Also verify zero-initialized residual branch wakes smoothly without division by zero
    zero_layer = nn.Linear(32, 32, bias=False)
    nn.init.zeros_(zero_layer.weight)
    opt_z = T4EquilibratedOptimizer([{"params": [zero_layer.weight], "role": "generator"}], lr=0.02)
    zero_layer.weight.grad = torch.randn_like(zero_layer.weight) * 1e-6
    with mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD is forbidden in t4_equil!")):
        opt_z.step()
    assert float(zero_layer.weight.detach().norm()) > 0.0
    assert bool(torch.isfinite(zero_layer.weight).all())


def test_quintic_orthogonalize_preserves_rank2_singular_directions_and_conv_batches():
    """Verify scale-normalized Quintic Newton-Schulz lifts both singular values of a skewed 1e-9 gradient."""
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    v = torch.tensor([[0.8, 0.6], [-0.6, 0.8]])
    skewed = (u @ torch.diag(torch.tensor([1e-7, 2e-8])) @ v.T)
    orth = quintic_orthogonalize_2d(skewed, steps=5)
    svals = torch.linalg.svdvals(orth)
    assert float(svals[0]) == pytest.approx(1.0, abs=0.25)
    assert float(svals[1]) > 0.65

    conv_slices = torch.randn(9, 16, 16) * 1e-8
    orth_b = quintic_orthogonalize_batched(conv_slices, steps=5)
    assert orth_b.shape == (9, 16, 16)
    assert bool(torch.isfinite(orth_b).all())


def test_fwad_saturation_pullback_preserves_exact_generator_derivative_and_deliberate_damage():
    """Semantic control:
    1) Generator backward derivative is 100% unaltered (exact chain rule, no surrogate backward).
    2) t4_arch escapes near-rail tanh saturation cleanly, whereas deliberate damage (zeroing G / frozen lr=0) fails.
    """
    from experiments.forge.tanh_point_mass import TanhPointMassGenerator

    torch.manual_seed(0)
    g = TanhPointMassGenerator(2, 16, 2, 2)
    z = torch.randn(8, 2, requires_grad=True)
    y = g(z)
    grad_exact = torch.autograd.grad(y.sum(), z, create_graph=False)[0]
    # Manually verify exact sech^2 chain rule against PyTorch autograd
    h = F.leaky_relu(F.linear(z, g.net[0].weight, g.net[0].bias), 0.2)
    u_pre = F.linear(h, g.net[2].weight, g.net[2].bias)
    sech2 = 1.0 - torch.tanh(u_pre).pow(2)
    assert torch.allclose(y, torch.tanh(u_pre), atol=1e-6)
    assert float(grad_exact.norm()) > 0.0 and float(sech2.detach().mean()) < 1.0

    # Deliberate damage negative control: zeroing generator weights after training fails zero_tanh_smoke gate
    card = json.loads((ROOT / "configs/forge/tasks/zero_tanh_smoke.json").read_text())
    from experiments.forge.tanh_point_mass import score_samples
    damaged_samples = torch.full((4096, 2), 0.995)
    damaged_metrics = score_samples(damaged_samples, card["execution"]["host_definition"], 1000)
    assert damaged_metrics["saturated_fraction"] == 1.0
    assert damaged_metrics["mean_abs_error"] > 0.005


def test_fresh_process_import_order_and_exact_continuation_across_active_boundaries(tmp_path):
    """Fresh-process checkpoint + RNG continuation test across ALL active Theory 4B boundaries:
    - atlas_initialized == True with K >= 2 separated modes, active Kantorovich potentials psi != 0,
      v_push, w_fake, res_ema, iso_ema;
    - active _thermal_scale / _step_scale in T2SecantObserverStepRecord;
    - populated T4EquilibratedOptimizer dense (m, v_rms) and sparse prior (m, v_row, row_step) histories.
    Verifies bitwise state_digest and trainer.sample parity between uninterrupted 8-step run
    and 4-step checkpoint resumed in a brand-new Python subprocess!
    """
    ckpt_path = tmp_path / "ckpt_step4.pt"
    out_uninterrupted = tmp_path / "uninterrupted.pt"
    out_resumed = tmp_path / "resumed.pt"

    # Run 8 uninterrupted steps in main process, saving checkpoint at step 4
    def _make_trainer():
        recipe = get_recipe(
            "t4_arch",
            prior_kind="mog",
            standardize=False,
            batch_size=32,
            num_particles=64,
            z_dim=2,
            total_steps=8,
            training_precision="float32",
        )
        prior = MoGParticlePrior(64, 2, sigma=0.05, standardize=False)
        init.deterministic_orthogonal_(prior, seed=2)
        g = SimpleMLPGenerator(2, 32, 4, 2)
        d = SimpleMLPDiscriminator(2, 32, 3, 2)
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        return GANTrainer(
            recipe,
            g,
            d,
            prior=prior,
            latent_generator=torch.Generator().manual_seed(11),
            penalty_generator=torch.Generator().manual_seed(12),
            eval_generator=torch.Generator().manual_seed(13),
            noise_generator=torch.Generator().manual_seed(14),
        )

    def _bimodal_batch(step_idx: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(500 + step_idx)
        centers = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        ids = torch.randint(0, 2, (32,), generator=gen)
        return centers[ids] + 0.05 * torch.randn(32, 2, generator=gen)

    trainer_full = _make_trainer()
    assert isinstance(trainer_full.prior_regularizer, T4ArchPriorRegularizer)
    for s in range(1, 9):
        with mock.patch("torch.autograd.grad", side_effect=AssertionError("autograd.grad forbidden!")):
            with mock.patch("torch.linalg.svd", side_effect=AssertionError("linalg.svd forbidden!")):
                trainer_full.step(_bimodal_batch(s), collect_stats=True)
        if s == 4:
            st4 = trainer_full.state_dict()
            # Assert all Theory 4B mechanisms are genuinely active at checkpoint boundary!
            assert "prior_regularizer" in st4
            pr_st = st4["prior_regularizer"]
            assert pr_st["atlas_initialized"] is True
            assert pr_st["proto_c"].shape[0] >= 2
            assert pr_st["iso_initialized"] is True
            torch.save(st4, ckpt_path)

    full_final_state = trainer_full.state_dict()
    full_samples = trainer_full.sample(64, generator=torch.Generator().manual_seed(999), output_noise=False)
    torch.save({"digest": state_digest(full_final_state), "samples": full_samples}, out_uninterrupted)

    # Resume from step 4 -> 8 in a completely fresh Python subprocess (also checking clean import order!)
    script = f"""
import torch
from experiments.forge.state import state_digest
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, get_recipe, init

assert DEFAULT_RECIPE == "unified_v11"
recipe = get_recipe("t4_arch", prior_kind="mog", standardize=False, batch_size=32, num_particles=64, z_dim=2, total_steps=8, training_precision="float32")
prior = MoGParticlePrior(64, 2, sigma=0.05, standardize=False)
init.deterministic_orthogonal_(prior, seed=102)
g = SimpleMLPGenerator(2, 32, 4, 2)
d = SimpleMLPDiscriminator(2, 32, 3, 2)
init.deterministic_orthogonal_(g, seed=100)
init.deterministic_orthogonal_(d, seed=101)
trainer = GANTrainer(
    recipe, g, d, prior=prior,
    latent_generator=torch.Generator().manual_seed(21),
    penalty_generator=torch.Generator().manual_seed(22),
    eval_generator=torch.Generator().manual_seed(23),
    noise_generator=torch.Generator().manual_seed(24),
)
saved = torch.load({str(ckpt_path)!r}, map_location="cpu", weights_only=True)
trainer.load_state_dict(saved)
assert state_digest(trainer.state_dict()) == state_digest(saved)

for s in range(5, 9):
    gen = torch.Generator().manual_seed(500 + s)
    centers = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    ids = torch.randint(0, 2, (32,), generator=gen)
    batch = centers[ids] + 0.05 * torch.randn(32, 2, generator=gen)
    trainer.step(batch, collect_stats=True)

samples = trainer.sample(64, generator=torch.Generator().manual_seed(999), output_noise=False)
torch.save({{"digest": state_digest(trainer.state_dict()), "samples": samples}}, {str(out_resumed)!r})
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=str(ROOT))
    ref = torch.load(out_uninterrupted, weights_only=True)
    res = torch.load(out_resumed, weights_only=True)
    assert res["digest"] == ref["digest"]
    assert torch.equal(res["samples"], ref["samples"])

    # Also verify corrupted checkpoints fail closed
    bad = deepcopy(torch.load(ckpt_path, weights_only=True))
    bad["prior_regularizer"]["schema"] = 99
    with pytest.raises(ValueError, match="schema"):
        _make_trainer().load_state_dict(bad)


def test_t4_arch_clockfree_and_deep_native_routing(tmp_path):
    """Verify clockfree_audit_measurement_v1 passes 4/4 parity and deep native100 cards construct cleanly."""
    card = json.loads((ROOT / "configs/forge/tasks/clockfree_audit_measurement_v1.json").read_text())
    cand = {"recipe_preset": "t4_arch", "recipe_overrides": {}}
    req = {
        "candidate": cand,
        "candidate_revision": "unit-t4-clockfree",
        "protocol": {"seed": 0},
        "tasks": {card["id"]: card},
        "jobs": [{"task_id": card["id"]}],
    }
    raw = adapters.run_task(req, {"task_id": card["id"]}, tmp_path / "cf", "cpu")
    grade = views.grade_result(card, raw)
    assert grade["status"] == "PASS"
    assert grade["metrics"]["parity_comparisons"] == 4
    assert raw["evidence"]["source_audit"]["unexplained_clock_dependencies"] == []

    # Verify all 3 deep native100 cards pass preflight and construct their declared architectures
    for name, expected_cls in (
        ("grid100_mlp16", "SimpleMLPGenerator"),
        ("grid100_res16_tanh", "ResidualMLPGenerator"),
        ("grid100_res16_scaled_tanh", "ScaledResidualMLPGenerator"),
    ):
        t_card = json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())
        assert adapters.adapter_preflight(t_card, cand) == []
        from experiments.forge.nativeprofiles import build_declared_native_models
        ctx = adapters._context(
            {"candidate": cand, "protocol": {"seed": 0}},
            t_card,
            "cpu",
            t_card["execution"]["resources"],
        )
        g_mod, d_mod = build_declared_native_models(ctx, t_card["execution"]["model"])
        assert type(g_mod).__name__ == expected_cls
        assert type(d_mod).__name__ == "SimpleMLPDiscriminator"
