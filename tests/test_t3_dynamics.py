"""Semantic, mathematical, and fresh-process continuation tests for RGDT (Theory 3)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import (
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    RGDTCriticAdam,
    RGDTCriticPenalty,
    RGDTGANLoss,
    RGDTPriorRegularizer,
    get_recipe,
    init,
)
from particlegan.t3_dynamics import (
    bipartite_deficit_macro_loss,
    born_infeld_pair,
    conformal_horizon_pair,
    pooled_tangent_conditional_loss,
    sinkhorn_slack,
)


class _TinyGen(nn.Module):
    def __init__(self, z_dim: int = 4, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 32), nn.SiLU(), nn.Linear(32, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _TinyCritic(nn.Module):
    def __init__(self, in_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def test_conformal_riemannian_symmetry_and_heteroscedastic_scale_invariance() -> None:
    """Conformal horizon normalizes 4.3x width ratios while keeping exact X == Y Feydy symmetry."""
    g = torch.Generator().manual_seed(0)
    means = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    sigmas = torch.tensor([0.07, 0.12, 0.20, 0.30])
    parts = [means[m] + sigmas[m] * torch.randn(32, 2, generator=g) for m in range(4)]
    y = torch.cat(parts, dim=0)

    ell_scalar = torch.tensor(0.022, dtype=torch.float32)
    ell_xy, ell_xx, is_conf = conformal_horizon_pair(y, y, ell_scalar, enabled=True)
    assert is_conf is True
    assert torch.allclose(ell_xy, ell_xx, atol=1e-7, rtol=1e-6)
    assert torch.allclose(ell_xy, ell_xy.T, atol=1e-7, rtol=1e-6)

    # Check intra-mode dimensionless ratio across narrowest (0.07) and broadest (0.30) modes:
    d_yy = torch.cdist(y, y)
    for m in (0, 3):
        sl = slice(32 * m, 32 * (m + 1))
        sub_d = d_yy[sl, sl]
        off = ~torch.eye(32, dtype=torch.bool)
        ratio_conf = (sub_d[off] / ell_xy[sl, sl][off]).median().item()
        assert 0.5 <= ratio_conf <= 4.5

    # Exact Feydy zero-gradient symmetry when X == Y:
    x_req = y.clone().requires_grad_(True)
    ell_xy2, ell_xx2, _ = conformal_horizon_pair(x_req, y, ell_scalar, enabled=True)
    c_xy, c_hat_xy = born_infeld_pair(x_req, y, ell_xy2)
    c_xx, c_hat_xx = born_infeld_pair(x_req, x_req, ell_xx2)
    zeros = torch.zeros(len(y))
    p_xy = sinkhorn_slack(c_hat_xy, zeros, zeros, iters=24)[: len(y), : len(y)]
    p_xx = sinkhorn_slack(c_hat_xx, zeros, zeros, iters=24)[: len(y), : len(y)]
    p_xx = 0.5 * (p_xx + p_xx.T)
    p_xx.fill_diagonal_(0.0)
    # Match active row mass exactly as in RGDTGANLoss:
    a_f = (float(len(y)) * p_xy.sum(dim=1)).clamp(0.0, 1.0)
    p_xy_off = 0.5 * (p_xy + p_xy.T)
    p_xy_off.fill_diagonal_(0.0)
    loss_sym = (p_xy_off * c_xy).sum() - 0.5 * (p_xy_off * c_xx).sum()
    grad_x = torch.autograd.grad(loss_sym, x_req)[0]
    assert float(grad_x.abs().max().item()) < 1e-6
    assert a_f.min().item() > 0.5


def test_bipartite_monge_macro_breaks_orthogonal_bar_saddle_and_ignores_poisson_noise() -> None:
    """Macro-slack splits equidistant orthogonal bars cleanly and vanishes when all modes are covered."""
    g = torch.Generator().manual_seed(0)
    bars = torch.full((4, 8, 8), -1.0)
    bars[0, :, 1:3] = 1.0  # V0
    bars[1, :, 5:7] = 1.0  # V1
    bars[2, 1:3, :] = 1.0  # H0
    bars[3, 5:7, :] = 1.0  # H1
    bars_flat = bars.reshape(4, 64)

    # Case 1: Only V0 and V1 covered by fake (16 each), real has all 4 bars (8 each):
    x_fake = torch.cat([bars_flat[0].expand(16, 64), bars_flat[1].expand(16, 64)], dim=0).clone().requires_grad_(True)
    y_real = torch.cat([bars_flat[m].expand(8, 64) for m in range(4)], dim=0) + 0.01 * torch.randn(32, 64, generator=g)

    ell = torch.tensor(0.055)
    _, c_hat = born_infeld_pair(x_fake, y_real, ell)
    zeros = torch.zeros(32)
    P_xy = sinkhorn_slack(c_hat, zeros, zeros, rho=3.0)
    loss_mac = bipartite_deficit_macro_loss(
        x_fake, y_real, c_hat, P_xy[:32, 32], P_xy[32, :32], rho=3.0, phase=0.17
    )
    assert float(loss_mac.item()) > 0.1
    grad_f = torch.autograd.grad(loss_mac, x_fake)[0]
    # Verify symmetry breaking along (H0 - H1): rows on V0 must NOT collapse to 0 projection on H0 - H1!
    h_diff = (bars_flat[2] - bars_flat[3])
    proj = (-grad_f[:16]) @ h_diff
    assert bool((proj > 1e-3).any()) and bool((proj < -1e-3).any())

    # Case 2: All 4 bars covered, with Poisson multinomial count imbalance (10, 6, 9, 7) vs (7, 9, 8, 8):
    x_cov = torch.cat(
        [bars_flat[0].expand(10, 64), bars_flat[1].expand(6, 64), bars_flat[2].expand(9, 64), bars_flat[3].expand(7, 64)],
        dim=0,
    ) + 0.01 * torch.randn(32, 64, generator=g)
    x_cov = x_cov.detach().requires_grad_(True)
    y_cov = torch.cat(
        [bars_flat[0].expand(7, 64), bars_flat[1].expand(9, 64), bars_flat[2].expand(8, 64), bars_flat[3].expand(8, 64)],
        dim=0,
    ) + 0.01 * torch.randn(32, 64, generator=g)
    _, c_hat_cov = born_infeld_pair(x_cov, y_cov, ell)
    P_cov = sinkhorn_slack(c_hat_cov, zeros, zeros, rho=3.0)
    loss_cov = bipartite_deficit_macro_loss(
        x_cov, y_cov, c_hat_cov, P_cov[:32, 32], P_cov[32, :32], rho=3.0, phase=0.17
    )
    assert float(loss_cov.item()) == 0.0


def test_pooled_tangent_conditional_discrepancy_unbiased_under_strong_drift() -> None:
    """Pooled-tangent residualization prevents variance inflation at s=1 while restoring s != 1."""
    g = torch.Generator().manual_seed(0)
    slope = torch.tensor([[0.85, 0.0], [0.0, 0.65]])
    true_std = torch.tensor([0.22, 0.18])

    def mean_scale_grad(scale_s: float) -> float:
        grads = []
        for _ in range(24):
            c = torch.randn(32, 2, generator=g)
            t_disc = torch.randint(1, 3, (32, 1), generator=g)
            eps_y = torch.randn(32, 2, generator=g) * true_std
            eps_x = torch.randn(32, 2, generator=g) * true_std
            y = c @ slope + eps_y
            s_param = torch.tensor(scale_s, requires_grad=True)
            x = c @ slope + s_param * eps_x
            loss = pooled_tangent_conditional_loss(x, y, c, cond_disc=t_disc)
            grads.append(float(torch.autograd.grad(loss, s_param)[0].item()))
        return sum(grads) / len(grads)

    g_eq = mean_scale_grad(1.0)
    g_wide = mean_scale_grad(1.8)
    g_narrow = mean_scale_grad(0.4)
    assert abs(g_eq) < 0.005
    assert g_wide > 0.004
    assert g_narrow < -0.004


def test_rgdt_prior_regularizer_deconflicts_mog_kernel_variance() -> None:
    """MoG kernel sigma > 0 softens near-coincident center repulsion while sigma == 0 repels strongly."""
    p_zero = ParticlePrior(32, 2, init_std=1.0)
    p_mog = MoGParticlePrior(32, 2, sigma=0.25, standardize=False)
    reg_zero = RGDTPriorRegularizer(p_zero, weight=1.0)
    reg_mog = RGDTPriorRegularizer(p_mog, weight=1.0)

    g = torch.Generator().manual_seed(0)
    base = torch.randn(32, 2, generator=g)
    # Place two centers very close (distance 1e-3 << sigma=0.25):
    base[1] = base[0] + torch.tensor([1e-3, 0.0])

    z0 = base.clone().requires_grad_(True)
    loss0 = reg_zero(z0)
    grad0 = torch.autograd.grad(loss0, z0)[0][0].norm().item()

    zm = base.clone().requires_grad_(True)
    lossm = reg_mog(zm)
    gradm = torch.autograd.grad(lossm, zm)[0][0].norm().item()

    assert grad0 > 5.0 * gradm


def test_rgdt_no_cross_trainer_state_leakage_and_damage_control() -> None:
    """Two interleaved RGDT trainers share zero state; damaging G strictly worsens RGDT loss."""
    recipe = get_recipe(
        "rgdt",
        prior_kind="mog",
        standardize=False,
        num_particles=32,
        z_dim=4,
        batch_size=16,
        total_steps=20,
        training_precision="float32",
    )

    def build(seed: int) -> GANTrainer:
        g = _TinyGen()
        d = _TinyCritic()
        p = MoGParticlePrior(32, 4, sigma=0.05, standardize=False, generator=torch.Generator().manual_seed(seed + 1))
        init.deterministic_orthogonal_(g, seed=seed + 2)
        init.deterministic_orthogonal_(d, seed=seed + 3)
        return GANTrainer(recipe, g, d, prior=p, seed=seed)

    data_rng1 = torch.Generator().manual_seed(101)
    batches_a = [torch.randn(16, 2, generator=data_rng1) + 1.5 for _ in range(6)]

    trainer_solo = build(0)
    for b in batches_a:
        trainer_solo.step(b)

    trainer_interleaved = build(0)
    trainer_other = build(7)
    data_rng2 = torch.Generator().manual_seed(202)
    for b in batches_a:
        trainer_other.step(torch.randn(16, 2, generator=data_rng2) * 5.0 - 3.0)
        trainer_interleaved.step(b)

    for p1, p2 in zip(trainer_solo.G.parameters(), trainer_interleaved.G.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(trainer_solo.prior.z, trainer_interleaved.prior.z)
    assert trainer_solo.opt_d.record.state_dict() == trainer_interleaved.opt_d.record.state_dict()

    # Semantic damage control: shifting fake far from real strictly increases RGDTGANLoss.g_loss:
    target = torch.full((16, 2), 2.0)
    near_fake = (target + 0.02 * torch.randn(16, 2, generator=data_rng1)).requires_grad_(True)
    damaged_fake = (target - 6.0).requires_grad_(True)
    r_log = trainer_solo.D(target)
    loss_near = trainer_solo.loss.g_loss(trainer_solo.D(near_fake), r_log)
    loss_damaged = trainer_solo.loss.g_loss(trainer_solo.D(damaged_fake), r_log)
    assert float(loss_damaged.item()) > float(loss_near.item())


@pytest.mark.parametrize("opt_family", ["dualnorm", "formulation", "adam"])
def test_rgdt_checkpoint_tamper_rejection_and_optimizer_families(opt_family: str) -> None:
    """RGDT supports dualnorm, formulation Adam, and plain Adam with strict checkpoint validation."""
    extra = (
        {}
        if opt_family == "dualnorm"
        else dict(
            optimizer_family=opt_family,
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_svd_backend="native",
            optimizer_dense_profile="default",
            constraint_geometry_mode="none",
        )
    )
    recipe = get_recipe(
        "rgdt",
        num_particles=24,
        z_dim=4,
        batch_size=12,
        total_steps=20,
        training_precision="float32",
        **extra,
    )
    g, d = _TinyGen(), _TinyCritic()
    p = ParticlePrior(24, 4)
    trainer = GANTrainer(recipe, g, d, prior=p, seed=0)
    rng = torch.Generator().manual_seed(11)
    for _ in range(3):
        trainer.step(torch.randn(12, 2, generator=rng))

    saved = deepcopy(trainer.state_dict())
    bad = deepcopy(saved)
    if opt_family == "formulation":
        bad["optimizers"][1]["regularizer"]["record"]["rgdt"]["thermal_scale"] = 2.5
    else:
        bad["optimizers"][1]["regularizer"]["record"]["rgdt"]["schema"] = 999
    with pytest.raises(ValueError):
        trainer.load_state_dict(bad)
    # Failed load must not corrupt live state:
    assert trainer.opt_d.record.observed_steps == 3


def test_rgdt_fresh_process_checkpoint_and_rng_continuation(tmp_path: Path) -> None:
    """Continuing from a serialized RGDT checkpoint in a fresh Python process matches uninterrupted execution."""
    ckpt_path = tmp_path / "ckpt_step5.pt"
    batches_path = tmp_path / "batches.pt"
    out_path = tmp_path / "continued.pt"

    rng = torch.Generator().manual_seed(42)
    batches = [torch.randn(16, 2, generator=rng) + 0.75 for _ in range(10)]
    torch.save(batches, batches_path)

    recipe = get_recipe(
        "rgdt",
        prior_kind="mog",
        standardize=False,
        num_particles=32,
        z_dim=4,
        batch_size=16,
        total_steps=30,
        training_precision="float32",
    )
    g, d = _TinyGen(), _TinyCritic()
    p = MoGParticlePrior(32, 4, sigma=0.025, standardize=False, generator=torch.Generator().manual_seed(1))
    init.deterministic_orthogonal_(g, seed=2)
    init.deterministic_orthogonal_(d, seed=3)
    trainer = GANTrainer(recipe, g, d, prior=p, seed=0)

    for step_idx in range(5):
        trainer.step(batches[step_idx])
    torch.save(trainer.state_dict(), ckpt_path)

    for step_idx in range(5, 10):
        trainer.step(batches[step_idx])
    ref_sample = trainer.sample(32)
    ref_state = trainer.state_dict()

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_t3_dynamics import _TinyCritic, _TinyGen

ckpt = torch.load({str(ckpt_path)!r}, map_location="cpu", weights_only=True)
batches = torch.load({str(batches_path)!r}, map_location="cpu", weights_only=True)
recipe = Recipe.from_dict(ckpt["recipe"])
g, d = _TinyGen(), _TinyCritic()
p = MoGParticlePrior(32, 4, sigma=0.025, standardize=False)
trainer = GANTrainer(recipe, g, d, prior=p, seed=99)
trainer.load_state_dict(ckpt)
for idx in range(5, 10):
    trainer.step(batches[idx])
sample = trainer.sample(32)
torch.save({{"state": trainer.state_dict(), "sample": sample}}, {str(out_path)!r})
"""
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        env={"PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    from experiments.forge.state import state_digest

    resumed = torch.load(out_path, map_location="cpu", weights_only=True)
    assert torch.equal(ref_sample, resumed["sample"])
    assert state_digest(ref_state) == state_digest(resumed["state"])


class _Tiny1DGen(nn.Module):
    def __init__(self, z_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return 1.0 + 0.25 * z[:, :1] + 0.01 * self.net(z)


class _Tiny1DCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def test_rgdt_fresh_process_continuation_across_thermal_step_and_shift_boundaries(tmp_path: Path) -> None:
    """Fresh-process continuation across active thermal cooling (<1), 1D step damping (<1), and shift shock reopening."""
    ckpt_path = tmp_path / "ckpt_boundary.pt"
    batches_path = tmp_path / "batches_boundary.pt"
    out_path = tmp_path / "continued_boundary.pt"

    rng = torch.Generator().manual_seed(77)
    stationary = [1.0 + 0.25 * torch.randn(128, 1, generator=rng) for _ in range(5)]
    shifted = [8.0 + 0.25 * torch.randn(128, 1, generator=rng) for _ in range(3)]
    batches = stationary + shifted
    torch.save(batches, batches_path)

    recipe = get_recipe(
        "rgdt",
        prior_kind="mog",
        standardize=True,
        num_particles=128,
        z_dim=2,
        batch_size=128,
        total_steps=20,
        training_precision="float32",
    )
    g, d = _Tiny1DGen(), _Tiny1DCritic()
    p = MoGParticlePrior(128, 2, sigma=0.05, standardize=True, generator=torch.Generator().manual_seed(5))
    init.deterministic_orthogonal_(g, seed=6)
    init.deterministic_orthogonal_(d, seed=7)
    trainer = GANTrainer(recipe, g, d, prior=p, seed=0)

    for step_idx in range(4):
        trainer.step(batches[step_idx])
    # Verify checkpoint sits strictly inside active thermal cooling and 1D DualNorm step damping:
    assert trainer.opt_d.record._thermal_scale < 1.0
    assert trainer.opt_d.record._step_scale < 1.0
    torch.save(trainer.state_dict(), ckpt_path)

    # Step 4 stays cooled, then steps 5..7 cross the abrupt +7.0 shift and trigger 1-step shock reopening:
    for step_idx in range(4, 8):
        trainer.step(batches[step_idx])
    assert trainer.opt_d.record._thermal_scale == 1.0
    assert trainer.opt_d.record._step_scale > 0.48
    ref_sample = trainer.sample(32)
    ref_state = trainer.state_dict()

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_t3_dynamics import _Tiny1DCritic, _Tiny1DGen

ckpt = torch.load({str(ckpt_path)!r}, map_location="cpu", weights_only=True)
batches = torch.load({str(batches_path)!r}, map_location="cpu", weights_only=True)
recipe = Recipe.from_dict(ckpt["recipe"])
g, d = _Tiny1DGen(), _Tiny1DCritic()
p = MoGParticlePrior(128, 2, sigma=0.05, standardize=True)
trainer = GANTrainer(recipe, g, d, prior=p, seed=123)
trainer.load_state_dict(ckpt)
assert trainer.opt_d.record._thermal_scale < 1.0
assert trainer.opt_d.record._step_scale < 1.0
for idx in range(4, 8):
    trainer.step(batches[idx])
assert trainer.opt_d.record._thermal_scale == 1.0
sample = trainer.sample(32)
torch.save({{"state": trainer.state_dict(), "sample": sample}}, {str(out_path)!r})
"""
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        env={"PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    from experiments.forge.state import state_digest

    resumed = torch.load(out_path, map_location="cpu", weights_only=True)
    assert torch.equal(ref_sample, resumed["sample"])
    assert state_digest(ref_state) == state_digest(resumed["state"])


def test_rgdt_fresh_process_continuation_across_connected_manifold_chart(tmp_path: Path) -> None:
    """Fresh-process continuation with zero-sigma ParticlePrior activating connected manifold_chart_loss."""
    ckpt_path = tmp_path / "ckpt_manifold.pt"
    batches_path = tmp_path / "batches_manifold.pt"
    out_path = tmp_path / "continued_manifold.pt"

    rng = torch.Generator().manual_seed(91)
    batches = []
    for _ in range(6):
        theta = (0.05 + 0.90 * torch.rand(24, generator=rng)) * torch.pi
        arc = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        batches.append(arc)
    torch.save(batches, batches_path)

    recipe = get_recipe(
        "rgdt",
        prior_kind="particles",
        sigma_rel=0.0,
        num_particles=16,
        z_dim=2,
        batch_size=24,
        total_steps=20,
        training_precision="float32",
    )
    g, d = _TinyGen(z_dim=2, out_dim=2), _TinyCritic(in_dim=2)
    p = ParticlePrior(16, 2, generator=torch.Generator().manual_seed(8))
    init.deterministic_orthogonal_(g, seed=9)
    init.deterministic_orthogonal_(d, seed=10)
    trainer = GANTrainer(recipe, g, d, prior=p, seed=0)

    # Confirm connected semicircle activates manifold_chart_loss (g_conn > 1e-4):
    probe_chart = trainer.loss.manifold_chart_loss(trainer.prior.z.detach(), trainer.G, batches[0])
    assert float(probe_chart.item()) > 0.0

    for step_idx in range(3):
        trainer.step(batches[step_idx])
    assert trainer.opt_d.record.rgdt_chord_phase > 0.0
    torch.save(trainer.state_dict(), ckpt_path)

    for step_idx in range(3, 6):
        trainer.step(batches[step_idx])
    ref_sample = trainer.sample(24)
    ref_state = trainer.state_dict()

    script = f"""
import torch
from particlegan import GANTrainer, ParticlePrior, Recipe
from tests.test_t3_dynamics import _TinyCritic, _TinyGen

ckpt = torch.load({str(ckpt_path)!r}, map_location="cpu", weights_only=True)
batches = torch.load({str(batches_path)!r}, map_location="cpu", weights_only=True)
recipe = Recipe.from_dict(ckpt["recipe"])
g, d = _TinyGen(z_dim=2, out_dim=2), _TinyCritic(in_dim=2)
p = ParticlePrior(16, 2)
trainer = GANTrainer(recipe, g, d, prior=p, seed=456)
trainer.load_state_dict(ckpt)
for idx in range(3, 6):
    trainer.step(batches[idx])
sample = trainer.sample(24)
torch.save({{"state": trainer.state_dict(), "sample": sample}}, {str(out_path)!r})
"""
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        env={"PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    from experiments.forge.state import state_digest

    resumed = torch.load(out_path, map_location="cpu", weights_only=True)
    assert torch.equal(ref_sample, resumed["sample"])
    assert state_digest(ref_state) == state_digest(resumed["state"])
