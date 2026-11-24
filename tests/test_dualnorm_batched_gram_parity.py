"""Numerical parity, guard-fallback, and contract tests for batched CUDA FP64 Gram polar."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.api_imbalance_nash import (
    build_models_for_spec,
    sample_target_for_kind,
)
from experiments.forge.imbalance_nash import imbalance_nash_context
from particlegan.optim import dualnorm
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import (
    NormalizedOptimizer,
    _batched_cuda_gram_polar,
    polar_factor,
)


def _fp64_svd_polar_oracle(matrix: torch.Tensor, smoothing: float) -> torch.Tensor:
    value = matrix.double()
    u, s, vh = torch.linalg.svd(value, full_matrices=False)
    cutoff = max(matrix.shape) * torch.finfo(torch.float32).eps * s[0]
    keep = s > cutoff
    if smoothing:
        weights = (s / torch.hypot(s, s.new_tensor(smoothing))) * keep
    else:
        weights = keep.to(s.dtype)
    return ((u * weights) @ vh).float()


def _make_structured_batch(
    shapes_and_ranks: list[tuple[int, int, int]],
    device: torch.device,
    seed: int = 0,
) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mats = []
    for rows, cols, rank in shapes_and_ranks:
        a = torch.randn(rows, rank, generator=gen, dtype=torch.float32)
        b = torch.randn(rank, cols, generator=gen, dtype=torch.float32)
        # Decaying spectrum spanning orders of magnitude like real Nash gradients.
        scale = torch.logspace(0.8, -3.2, rank, dtype=torch.float32)
        mats.append(((a * scale) @ b).to(device=device))
    return mats


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(256, 256), (512, 512), (512, 256), (256, 512)])
@pytest.mark.parametrize("smoothing", [0.0, 0.001, 0.0061])
def test_batched_cuda_gram_polar_matches_fp64_oracle_and_native_svd(shape, smoothing):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda:0")
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    rows, cols = shape
    min_dim = min(rows, cols)
    mats = _make_structured_batch(
        [(rows, cols, min_dim), (rows, cols, min_dim // 2), (rows, cols, min_dim // 4)],
        device=device,
        seed=17,
    )
    batch = torch.stack(mats, dim=0)
    out, ok = _batched_cuda_gram_polar(batch, smoothing=smoothing)
    assert out is not None
    assert ok.all().item() is True
    assert out.shape == batch.shape and out.dtype == torch.float32 and out.device == device

    for idx, mat in enumerate(mats):
        oracle = _fp64_svd_polar_oracle(mat, smoothing)
        native = polar_factor(mat, smoothing=smoothing)
        # Matches exact FP64 SVD oracle to ~1e-6 (far tighter than FP32 SVD's own ~1e-4 rounding).
        torch.testing.assert_close(out[idx], oracle, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(out[idx], native, rtol=2e-3, atol=5e-4)

    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.cuda
def test_batched_cuda_gram_polar_handles_zero_near_cutoff_and_nonfinite_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda:0")
    n = 256

    # 1. Exact zero matrix stays exact zero and passes guard cleanly.
    zeros = torch.zeros((2, n, n), device=device, dtype=torch.float32)
    out_z, ok_z = _batched_cuda_gram_polar(zeros, smoothing=0.001)
    assert out_z is not None and ok_z.all().item() is True
    assert torch.equal(out_z, zeros)

    # 2. Singular value sitting exactly on the FP32 rank cutoff triggers Weyl guard (ok=False).
    cutoff = n * torch.finfo(torch.float32).eps
    diag_ambig = torch.zeros((n, n), device=device, dtype=torch.float32)
    diag_ambig[0, 0] = 1.0
    diag_ambig[1, 1] = cutoff
    diag_clean = torch.eye(n, device=device, dtype=torch.float32)
    out_m, ok_m = _batched_cuda_gram_polar(torch.stack([diag_clean, diag_ambig], dim=0), smoothing=0.001)
    assert out_m is not None
    assert ok_m.tolist() == [True, False]
    torch.testing.assert_close(out_m[0], _fp64_svd_polar_oracle(diag_clean, 0.001), rtol=1e-5, atol=1e-6)

    # 3. Non-finite slice is marked ok=False without corrupting finite sibling slices:
    #    polar_cpu_v1 fallback raises LinAlgError (CPU LAPACK), and default CUDA fallback
    #    never silently skips the NaN gradient.
    nan_mat = diag_clean.clone()
    nan_mat[0, 0] = float("nan")
    out_nf, ok_nf = _batched_cuda_gram_polar(torch.stack([diag_clean, nan_mat], dim=0), smoothing=0.001)
    assert out_nf is not None
    assert ok_nf.tolist() == [True, False]
    torch.testing.assert_close(out_nf[0], _fp64_svd_polar_oracle(diag_clean, 0.001), rtol=1e-5, atol=1e-6)

    p_bad_cpu = nn.Parameter(torch.zeros((n, n), device=device, dtype=torch.float32))
    p_bad_cpu.grad = nan_mat.clone()
    opt_bad_cpu = NormalizedOptimizer([p_bad_cpu], lr=0.01, smoothing=0.001, dense_profile="polar_cpu_v1")
    with pytest.raises(torch.linalg.LinAlgError):
        opt_bad_cpu.step()

    p_bad_cuda = nn.Parameter(torch.zeros((n, n), device=device, dtype=torch.float32))
    p_bad_cuda.grad = nan_mat.clone()
    opt_bad_cuda = NormalizedOptimizer([p_bad_cuda], lr=0.01, smoothing=0.001)
    opt_bad_cuda.step()
    assert not torch.isfinite(p_bad_cuda).all()


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dense_profile", "smoothing"),
    [("default", 0.001), ("polar_cpu_v1", 0.0061)],
)
def test_normalized_optimizer_large_cuda_step_parity_and_checkpoint_identity(
    dense_profile, smoothing, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(42)

    shapes = [(512, 512), (512, 512), (512, 512), (32, 32), (512,)]
    init_tensors = [torch.randn(s, generator=gen, dtype=torch.float32).to(device) * 0.1 for s in shapes]
    grads = [
        (
            (torch.randn(s[0], min(s[0], 128), generator=gen) @ torch.randn(min(s[0], 128), s[1], generator=gen))
            .to(device)
            * 0.05
            if len(s) == 2
            else torch.randn(s, generator=gen, dtype=torch.float32).to(device) * 0.05
        )
        for s in shapes
    ]

    params_fast = [nn.Parameter(t.clone()) for t in init_tensors]
    params_ref = [nn.Parameter(t.clone()) for t in init_tensors]
    opt_fast = NormalizedOptimizer(params_fast, lr=0.014, smoothing=smoothing, dense_profile=dense_profile)
    opt_ref = NormalizedOptimizer(params_ref, lr=0.014, smoothing=smoothing, dense_profile=dense_profile)

    for pf, pr, g in zip(params_fast, params_ref, grads):
        pf.grad = g.clone()
        pr.grad = g.clone()

    # Run optimized path (batched CUDA FP64 Gram for >= 256).
    opt_fast.step()
    # Run reference unbatched path by raising the size threshold above 512.
    monkeypatch.setattr(dualnorm, "_BATCHED_GRAM_MIN_DIM", 10_000)
    opt_ref.step()

    # Large 512x512 layers match reference updates tightly (within FP32 SVD rounding).
    for idx in range(3):
        torch.testing.assert_close(params_fast[idx], params_ref[idx], rtol=1e-4, atol=2e-5)
        # Also verify exact agreement with the FP64 SVD oracle update.
        expected = init_tensors[idx] - 0.014 * math.sqrt(1.0) * _fp64_svd_polar_oracle(grads[idx], smoothing)
        torch.testing.assert_close(params_fast[idx], expected, rtol=1e-5, atol=1e-6)

    # Small 2D matrix (32x32) and 1D bias (512,) remain 100% bitwise identical.
    assert torch.equal(params_fast[3], params_ref[3])
    assert torch.equal(params_fast[4], params_ref[4])

    # Checkpoint schema and state_dict load/resume are strictly unchanged.
    saved = deepcopy(opt_fast.state_dict())
    assert saved["dualnorm"] == opt_ref.state_dict()["dualnorm"]
    restored_params = [nn.Parameter(p.detach().clone()) for p in params_fast]
    restored = NormalizedOptimizer(restored_params, lr=0.014, smoothing=smoothing, dense_profile=dense_profile)
    restored.load_state_dict(saved)
    assert restored.state_dict()["dualnorm"] == saved["dualnorm"]


@pytest.mark.cuda
def test_normalized_optimizer_large_cuda_momentum_zero_current_grad_skips():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda:0")
    p = nn.Parameter(torch.zeros((256, 256), device=device, dtype=torch.float32))
    opt = NormalizedOptimizer([p], lr=0.02, momentum=0.5, smoothing=0.001)
    p.grad = torch.eye(256, device=device, dtype=torch.float32)
    opt.step()
    after_first = p.detach().clone()
    assert not torch.equal(after_first, torch.zeros_like(after_first))
    # Zero current gradient must NOT move on stale momentum buffer.
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.equal(p.detach(), after_first)


def test_direction_blend_nonconflict_and_conflict_exact_stats():
    gen = torch.Generator(device="cpu").manual_seed(3)
    w0 = torch.randn((8, 8), generator=gen)
    b0 = torch.randn((8,), generator=gen)
    g_w = torch.randn((8, 8), generator=gen)
    g_b = torch.randn((8,), generator=gen)

    # Non-conflict case: normal aligned with gradient (displacement is -polar, so normal dot d < 0).
    w = nn.Parameter(w0.clone())
    b = nn.Parameter(b0.clone())
    opt = DirectionBlendOptimizer([w, b], lr=0.05, smoothing=0.001)
    loss = (w * g_w).sum() + (b * g_b).sum()
    opt.bind_protected_losses((loss,))
    loss.backward()
    opt.step()
    assert opt.constraint_geometry_stats["steps"] == 1
    assert opt.constraint_geometry_stats["projected_steps"] == 0
    assert opt.direction_blend_stats["conflict_steps"] == 0
    assert opt.constraint_geometry_stats["max_derivative_after"] == opt.constraint_geometry_stats["max_derivative_before"]

    # Conflict case: protected loss opposes the gradient.
    w2 = nn.Parameter(w0.clone())
    b2 = nn.Parameter(b0.clone())
    opt2 = DirectionBlendOptimizer([w2, b2], lr=0.05, smoothing=0.001)
    loss_main = (w2 * g_w).sum() + (b2 * g_b).sum()
    loss_opp = -(w2 * g_w).sum() - (b2 * g_b).sum()
    opt2.bind_protected_losses((loss_opp,))
    loss_main.backward()
    opt2.step()
    assert opt2.constraint_geometry_stats["steps"] == 1
    assert opt2.direction_blend_stats["conflict_steps"] == 1
    assert opt2.constraint_geometry_stats["projected_steps"] == 1
    assert opt2.constraint_geometry_stats["max_derivative_after"] <= 1e-12


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("candidate_id", "task_id"),
    [
        ("bispf-optimized-quintic-v1", "nash_overpowered_critic_ring16"),
        ("rhbi-v12", "nash_overpowered_critic_ring16"),
        ("rhbi-v12", "nash_overpowered_generator_ring16"),
    ],
)
def test_nash_trainer_step_parity_matches_unbatched_reference(candidate_id, task_id, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    root = Path(__file__).resolve().parents[1]
    cand = json.loads((root / f"configs/forge/ideas/{candidate_id}.json").read_text())
    task = json.loads((root / f"configs/forge/tasks/{task_id}.json").read_text())

    def _run_steps(n_steps: int):
        ctx = imbalance_nash_context({"candidate": cand, "protocol": {"seed": 0}}, task, "cuda:0", root=root)
        spec = task["execution"]["host_definition"]
        g = ctx.construct(lambda: build_models_for_spec(spec, z_dim=ctx.recipe.z_dim)[0], component="generator").to(
            ctx.device
        )
        d = ctx.construct(lambda: build_models_for_spec(spec, z_dim=ctx.recipe.z_dim)[1], component="discriminator").to(
            ctx.device
        )
        trainer = ctx.build_trainer(g, d, max_steps=int(task["execution"]["steps"]))
        data_rng = ctx.streams.generator("data", component="target", purpose="training", device="cpu")
        outs = []
        step1_params = None
        for s in range(n_steps):
            batch = sample_target_for_kind(
                task_id, ctx.recipe.batch_size, data_rng, completed_steps=s, device="cpu"
            ).to(ctx.device)
            outs.append(trainer.step(batch))
            if s == 0:
                step1_params = (
                    [p.detach().clone() for p in g.parameters()],
                    [p.detach().clone() for p in d.parameters()],
                )
        return g, d, outs, step1_params

    g_fast, d_fast, outs_fast, step1_fast = _run_steps(2)
    monkeypatch.setattr(dualnorm, "_BATCHED_GRAM_MIN_DIM", 10_000)
    g_ref, d_ref, outs_ref, step1_ref = _run_steps(2)

    # Step 1 matches exact single-step update within FP32 SVD rounding (~1e-5).
    for pf, pr in zip(step1_fast[0], step1_ref[0]):
        torch.testing.assert_close(pf, pr, rtol=1e-3, atol=5e-5)
    for pf, pr in zip(step1_fast[1], step1_ref[1]):
        torch.testing.assert_close(pf, pr, rtol=1e-3, atol=5e-5)

    # Step 2 (after feeding FP32 SVD rounding through 4-layer 512-wide R1 double-backward)
    # remains tightly coupled in both losses and parameters.
    for out_f, out_r in zip(outs_fast, outs_ref):
        assert math.isfinite(float(out_f["loss_d"])) and math.isfinite(float(out_f["loss_g"]))
        assert float(out_f["loss_d"]) == pytest.approx(float(out_r["loss_d"]), rel=1e-3, abs=2e-3)
        assert float(out_f["loss_g"]) == pytest.approx(float(out_r["loss_g"]), rel=1e-3, abs=2e-3)
    for pf, pr in zip(g_fast.parameters(), g_ref.parameters()):
        torch.testing.assert_close(pf.detach(), pr.detach(), rtol=2e-3, atol=2e-4)
    for pf, pr in zip(d_fast.parameters(), d_ref.parameters()):
        torch.testing.assert_close(pf.detach(), pr.detach(), rtol=5e-3, atol=1.5e-3)
        rel_l2 = float((pf.detach() - pr.detach()).norm() / pr.detach().norm().clamp_min(1e-8))
        assert rel_l2 < 1e-3

