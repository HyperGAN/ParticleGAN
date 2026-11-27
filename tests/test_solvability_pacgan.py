"""3-Level Solvability & Multi-Family Trainer Suite for PacGAN Stretch Tasks.

Covers:
  - pacgan_unequal_mass (PacGANUnequalMassFixture, m=4, 5-mode ring with 2% rare mode)
  - pacgan8_grid100 (PacGAN8Grid100Fixture, m=8, 100-mode 10x10 Gaussian grid)

Levels:
  1. Level 1 (200/200 Oracle draws + >= 2.0x noise headroom across all thresholds)
  2. Level 2 (Constructive dense 128-wide PacGANGenerator capacity witness without identity bypass)
  3. Level 3 (Destructive negative controls deterministically failing each individual gate + NaN/Inf)
  4. Multi-family trainer execution (bispf, atlas w/ ManipulatedPrior, bcap, k3p, ka2, etc.)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.toy_audit.api_pacgan import (
    GRID100_COORDS,
    PacGAN4RepulsionDiscriminator,
    PacGAN8Grid100Fixture,
    PacGANDiscriminator,
    PacGANGenerator,
    PacGANUnequalMassFixture,
    _grid100_centers,
    _ring_centers,
    score_pacgan_grid100,
    score_pacgan_unequal,
)
from particlegan import ParticlePrior, get_recipe, init
from particlegan.priors import ManipulatedPrior, NormalNoiseManipulator, unwrap_prior


REPO_ROOT = Path(__file__).resolve().parents[1]
UNEQUAL_TASK_PATH = REPO_ROOT / "configs" / "forge" / "tasks" / "pacgan_unequal_mass.json"
GRID100_TASK_PATH = REPO_ROOT / "configs" / "forge" / "tasks" / "pacgan8_grid100.json"


def _load_thresholds(path: Path) -> list[tuple[str, str, float]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["adapter"] == "transfer_vector"
    assert cfg["execution"]["prior_contract"] == "recipe_owned_v1"
    return [(str(m), str(op), float(val)) for m, op, val in cfg["evaluation"]["thresholds"]]


def _build_dense_constructive_generator(
    *,
    scale: float,
    angle_rad: float,
    shift: tuple[float, float],
    seed: int = 42,
) -> tuple[PacGANGenerator, torch.Tensor, torch.Tensor]:
    """Construct a dense 128-wide 3-layer LeakyReLU(0.2) PacGANGenerator.

    Uses paired positive/negative 64-frame orthogonal embeddings so every weight
    matrix is 100% dense (no sparse identity rows) and implements:
        G(z) = scale * R(angle_rad) @ z + shift
    via the exact identity LeakyReLU_0.2(v) - LeakyReLU_0.2(-v) = 1.2 * v.
    """
    gen = PacGANGenerator(z_dim=2, hidden_dim=128, out_dim=2)
    rng = torch.Generator().manual_seed(seed)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    shift_t = torch.tensor(shift, dtype=torch.float32)

    q1, _ = torch.linalg.qr(torch.randn(64, 2, generator=rng))
    w1 = torch.cat([q1, -q1], dim=0)  # (128, 2)
    recon1 = torch.cat([q1.T, -q1.T], dim=1) / 1.2  # (2, 128)

    q2, _ = torch.linalg.qr(torch.randn(64, 2, generator=rng))
    w2 = torch.cat([q2, -q2], dim=0) @ recon1  # (128, 128)
    recon2 = torch.cat([q2.T, -q2.T], dim=1) / 1.2  # (2, 128)

    w3 = (scale * rot) @ recon2  # (2, 128)

    with torch.no_grad():
        gen.net[0].weight.copy_(w1)
        gen.net[0].bias.zero_()
        gen.net[2].weight.copy_(w2)
        gen.net[2].bias.zero_()
        gen.net[4].weight.copy_(w3)
        gen.net[4].bias.copy_(shift_t)
    return gen, rot, shift_t


# ==============================================================================
# Level 1: 200/200 Oracle Draws & >= 2.0x Noise Headroom
# ==============================================================================


def test_level1_oracle_pacgan_unequal_mass_200_draws() -> None:
    thresholds = _load_thresholds(UNEQUAL_TASK_PATH)
    fixture = PacGANUnequalMassFixture(seed=0)

    records: dict[str, list[float]] = {m: [] for m, _, _ in thresholds}
    for draw in range(200):
        rng = torch.Generator().manual_seed(1000 + draw)
        x_real = fixture.sample_target(fixture.EVAL_SAMPLES, generator=rng)
        res = score_pacgan_unequal(x_real)
        assert res["passed"], f"Oracle draw {draw} failed: {res['failed_bounds']} ({res['metrics']})"
        for m, _, _ in thresholds:
            records[m].append(float(res["metrics"][m]))

    for metric, op, bound in thresholds:
        vals = np.asarray(records[metric], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std())
        if op in (">=", "=="):
            margin = mean - bound
        elif op == "<=":
            margin = bound - mean
        else:
            raise ValueError(f"Unsupported op {op}")

        if std < 1e-12:
            assert margin >= 0.0, f"{metric}: zero-std oracle violated bound ({mean} vs {op} {bound})"
        else:
            headroom = margin / (3.0 * std)
            assert headroom >= 2.0, (
                f"pacgan_unequal_mass {metric} headroom={headroom:.2f}x < 2.0x "
                f"(mean={mean:.4f}, std={std:.4f}, bound={op} {bound})"
            )


def test_level1_oracle_pacgan8_grid100_200_draws() -> None:
    thresholds = _load_thresholds(GRID100_TASK_PATH)
    fixture = PacGAN8Grid100Fixture(seed=0)

    records: dict[str, list[float]] = {m: [] for m, _, _ in thresholds}
    for draw in range(200):
        rng = torch.Generator().manual_seed(2000 + draw)
        x_real = fixture.sample_target(fixture.EVAL_SAMPLES, generator=rng)
        res = score_pacgan_grid100(x_real)
        assert res["passed"], f"Oracle draw {draw} failed: {res['failed_bounds']} ({res['metrics']})"
        for m, _, _ in thresholds:
            records[m].append(float(res["metrics"][m]))

    for metric, op, bound in thresholds:
        vals = np.asarray(records[metric], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std())
        if op in (">=", "=="):
            margin = mean - bound
        elif op == "<=":
            margin = bound - mean
        else:
            raise ValueError(f"Unsupported op {op}")

        if std < 1e-12:
            assert margin >= 0.0, f"{metric}: zero-std oracle violated bound ({mean} vs {op} {bound})"
        else:
            headroom = margin / (3.0 * std)
            assert headroom >= 2.0, (
                f"pacgan8_grid100 {metric} headroom={headroom:.2f}x < 2.0x "
                f"(mean={mean:.4f}, std={std:.4f}, bound={op} {bound})"
            )


# ==============================================================================
# Level 2: Constructive nn.Module Capacity Witness (No Identity Bypass)
# ==============================================================================


def test_level2_constructive_witness_pacgan_unequal_mass() -> None:
    fixture = PacGANUnequalMassFixture(seed=0)
    base_prior = unwrap_prior(fixture.prior)
    prior_sigma = 0.05
    target_sigma = fixture.sigma  # 0.08
    scale = target_sigma / prior_sigma  # 1.6 != 1.0

    gen, rot, shift_t = _build_dense_constructive_generator(
        scale=scale,
        angle_rad=0.75,
        shift=(1.4, -1.1),
        seed=17,
    )
    fixture.G.load_state_dict(gen.state_dict())

    # Verify all hidden layers are dense (no sparse identity rows)
    for layer_idx in (0, 2, 4):
        w = fixture.G.net[layer_idx].weight.detach()
        assert float((w.abs() > 1e-6).float().mean()) > 0.99

    # Place 256 prior particles in latent coordinates (inverse of G) according to target weights
    centers = torch.from_numpy(_ring_centers(num_modes=5, radius=fixture.radius))
    latent_centers = ((centers - shift_t) @ rot) / scale
    counts = [102, 77, 51, 20, 6]  # sums to 256; rare mode has 6/256 = 2.34%
    assert sum(counts) == fixture.particles
    z_rows = []
    for k, cnt in enumerate(counts):
        z_rows.append(latent_centers[k].unsqueeze(0).expand(cnt, 2))
    z_table = torch.cat(z_rows, dim=0)

    with torch.no_grad():
        base_prior.z.copy_(z_table)
        displacement = float((fixture.G(base_prior.z) - base_prior.z).norm(dim=1).mean())
        assert displacement > 1.0

    res = fixture.observe()
    assert res["passed"], f"Level 2 constructive witness failed: {res['failed_bounds']} ({res['metrics']})"


def test_level2_constructive_witness_pacgan8_grid100() -> None:
    fixture = PacGAN8Grid100Fixture(seed=0)
    base_prior = unwrap_prior(fixture.prior)
    prior_sigma = 0.05
    target_sigma = fixture.sigma  # 0.05
    scale = target_sigma / prior_sigma  # 1.0 with non-trivial rotation + translation

    gen, rot, shift_t = _build_dense_constructive_generator(
        scale=scale,
        angle_rad=0.85,
        shift=(2.5, -1.8),
        seed=29,
    )
    fixture.G.load_state_dict(gen.state_dict())

    for layer_idx in (0, 2, 4):
        w = fixture.G.net[layer_idx].weight.detach()
        assert float((w.abs() > 1e-6).float().mean()) > 0.99

    centers = torch.from_numpy(_grid100_centers(coords=GRID100_COORDS))
    latent_centers = ((centers - shift_t) @ rot) / scale
    idx = (
        torch.arange(fixture.particles, dtype=torch.float32) * 100.0 / fixture.particles
    ).long().clamp(max=99)
    with torch.no_grad():
        base_prior.z.copy_(latent_centers[idx])
        displacement = float((fixture.G(base_prior.z) - base_prior.z).norm(dim=1).mean())
        assert displacement > 3.0

    res = fixture.observe()
    assert res["passed"], f"Level 2 constructive witness failed: {res['failed_bounds']} ({res['metrics']})"


# ==============================================================================
# Level 3: Destructive Negative Controls
# ==============================================================================


def test_level3_destructive_controls_pacgan_unequal_mass() -> None:
    # 1. Dropping the 2% rare mode fails modes == 5 and min_mode_mass >= 0.010
    dropped_rare = PacGANUnequalMassFixture(
        seed=0,
        weights=(0.40, 0.30, 0.22, 0.08, 0.00),
    )
    x_drop = dropped_rare.sample_target(4096)
    res_drop = score_pacgan_unequal(x_drop)
    assert not res_drop["passed"]
    assert res_drop["metrics"]["modes"] == 4
    assert res_drop["metrics"]["min_mode_mass"] < 0.010
    assert "modes" in res_drop["failed_bounds"]
    assert "min_mode_mass" in res_drop["failed_bounds"]

    # 2. Uniform weights fail mass_tv <= 0.10 while covering all 5 modes
    uniform_mix = PacGANUnequalMassFixture(
        seed=0,
        weights=(0.20, 0.20, 0.20, 0.20, 0.20),
    )
    x_unif = uniform_mix.sample_target(4096)
    res_unif = score_pacgan_unequal(x_unif)
    assert not res_unif["passed"]
    assert res_unif["metrics"]["modes"] == 5
    assert res_unif["metrics"]["min_mode_mass"] >= 0.010
    assert res_unif["metrics"]["mass_tv"] > 0.10
    assert "mass_tv" in res_unif["failed_bounds"]

    # 3. Zero-variance within-mode point collapse fails min_cov_eig_ratio >= 0.20
    collapsed = PacGANUnequalMassFixture(seed=0, sigma=0.0)
    x_col = collapsed.sample_target(4096)
    res_col = score_pacgan_unequal(x_col)
    assert not res_col["passed"]
    assert res_col["metrics"]["modes"] == 5
    assert res_col["metrics"]["hq"] == 1.0
    assert res_col["metrics"]["min_cov_eig_ratio"] == 0.0
    assert "min_cov_eig_ratio" in res_col["failed_bounds"]

    # 4. Excessive noise fails hq >= 0.88
    noisy = PacGANUnequalMassFixture(seed=0, sigma=0.40)
    x_noisy = noisy.sample_target(4096)
    res_noisy = score_pacgan_unequal(x_noisy)
    assert not res_noisy["passed"]
    assert res_noisy["metrics"]["hq"] < 0.88
    assert "hq" in res_noisy["failed_bounds"]

    # 5. Non-finite NaN / Inf fails deterministically
    x_nan = x_unif.clone()
    x_nan[0, 0] = float("nan")
    res_nan = score_pacgan_unequal(x_nan)
    assert not res_nan["passed"]
    assert "nonfinite_output" in res_nan["failed_bounds"]

    x_inf = x_unif.clone()
    x_inf[0, 0] = float("inf")
    res_inf = score_pacgan_unequal(x_inf)
    assert not res_inf["passed"]
    assert "nonfinite_output" in res_inf["failed_bounds"]


def test_level3_destructive_controls_pacgan8_grid100() -> None:
    centers = _grid100_centers()
    rng = np.random.default_rng(0)

    # 1. Dropping 10 modes (only 90 modes covered) fails modes >= 95
    idx_90 = np.arange(4096) % 90
    x_90 = centers[idx_90] + 0.05 * rng.standard_normal((4096, 2)).astype(np.float32)
    res_90 = score_pacgan_grid100(x_90)
    assert not res_90["passed"]
    assert res_90["metrics"]["modes"] == 90
    assert "modes" in res_90["failed_bounds"]

    # 2. Skewed mode weights (covering all 100 modes with >=15 points, rest on mode 0) fails mass_tv <= 0.15
    base_idx = np.repeat(np.arange(100), 20)
    extra_idx = np.zeros(4096 - len(base_idx), dtype=np.int64)
    idx_skew = np.concatenate([base_idx, extra_idx])
    x_skew = centers[idx_skew] + 0.05 * rng.standard_normal((4096, 2)).astype(np.float32)
    res_skew = score_pacgan_grid100(x_skew)
    assert not res_skew["passed"]
    assert res_skew["metrics"]["modes"] == 100
    assert res_skew["metrics"]["mass_tv"] > 0.15
    assert "mass_tv" in res_skew["failed_bounds"]

    # 3. Zero-variance within-mode point collapse fails min_cov_eig_ratio >= 0.15
    collapsed = PacGAN8Grid100Fixture(seed=0, sigma=0.0)
    x_col = collapsed.sample_target(4096)
    res_col = score_pacgan_grid100(x_col)
    assert not res_col["passed"]
    assert res_col["metrics"]["modes"] == 100
    assert res_col["metrics"]["hq"] == 1.0
    assert res_col["metrics"]["min_cov_eig_ratio"] == 0.0
    assert "min_cov_eig_ratio" in res_col["failed_bounds"]

    # 4. Excessive dispersion fails hq >= 0.85
    noisy = PacGAN8Grid100Fixture(seed=0, sigma=0.25)
    x_noisy = noisy.sample_target(4096)
    res_noisy = score_pacgan_grid100(x_noisy)
    assert not res_noisy["passed"]
    assert res_noisy["metrics"]["hq"] < 0.85
    assert "hq" in res_noisy["failed_bounds"]

    # 5. Non-finite NaN / Inf fails deterministically
    x_nan = x_col.clone()
    x_nan[0, 0] = float("nan")
    res_nan = score_pacgan_grid100(x_nan)
    assert not res_nan["passed"]
    assert "nonfinite_output" in res_nan["failed_bounds"]


# ==============================================================================
# Multi-Family Execution & Checkpoint Roundtrip Tests
# ==============================================================================


@pytest.mark.parametrize(
    "fixture_cls,batch_size,particles",
    [
        (PacGANUnequalMassFixture, 64, 64),
        (PacGAN8Grid100Fixture, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "family",
    ["bispf", "atlas", "bcap", "k3p", "ka2", "bispf_optimized", "bcap_adam", "e22"],
)
def test_multi_family_4_steps_update_prior(
    fixture_cls: type[PacGANUnequalMassFixture | PacGAN8Grid100Fixture],
    batch_size: int,
    particles: int,
    family: str,
) -> None:
    if family == "atlas":
        recipe = get_recipe("atlas", prior_update="learned", num_particles=particles, z_dim=2)
        base = ParticlePrior(
            num_particles=particles,
            z_dim=2,
            generator=torch.Generator().manual_seed(0),
        )
        init.deterministic_orthogonal_(base, seed=0)
        prior = ManipulatedPrior(base, [NormalNoiseManipulator(sigma=0.0065)])
        fixture = fixture_cls(
            recipe=recipe,
            prior=prior,
            batch_size=batch_size,
            particles=particles,
            steps=20,
            seed=0,
        )
    else:
        fixture = fixture_cls(
            recipe_name=family,
            batch_size=batch_size,
            particles=particles,
            steps=20,
            seed=0,
        )

    z_before = unwrap_prior(fixture.prior).z.detach().clone()
    for _ in range(4):
        fixture.step()
    z_after = unwrap_prior(fixture.prior).z.detach().clone()

    assert fixture.step_count == 4
    assert torch.isfinite(z_after).all(), f"{family}: non-finite prior.z after 4 steps"
    assert not torch.allclose(z_before, z_after), f"{family}: prior.z did not update after 4 steps"

    obs = fixture.observe()
    assert np.isfinite(obs["views"][0]["samples"]).all()


def test_checkpoint_roundtrip_pacgan_fixtures() -> None:
    for cls in (PacGANUnequalMassFixture, PacGAN8Grid100Fixture):
        f1 = cls(recipe_name="bispf", batch_size=64, particles=64, steps=20, seed=0)
        f1.step()
        f1.step()
        state = f1.state_dict()

        f2 = cls(recipe_name="bispf", batch_size=64, particles=64, steps=20, seed=99)
        f2.load_state_dict(state)

        f1.step()
        f2.step()

        assert torch.allclose(unwrap_prior(f1.prior).z, unwrap_prior(f2.prior).z)
        obs1 = f1.observe()
        obs2 = f2.observe()
        assert np.allclose(obs1["views"][0]["samples"], obs2["views"][0]["samples"])


def test_pacgan4_repulsion_discriminator_contract_and_gradients() -> None:
    """Verify PacGAN4RepulsionDiscriminator feature contract and exact repulsion gradients."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    assert disc.feature_dim == 23

    # Clean raw-packed MLP discriminator must NOT inherit hand-crafted repulsion features
    assert not issubclass(PacGANDiscriminator, PacGAN4RepulsionDiscriminator)
    disc_clean = PacGANDiscriminator(in_dim=8, hidden_dim=128)
    assert disc_clean(torch.randn(5, 8)).shape == (5,)

    # Shape contract: (B, 4, 2) and flat (B, 8)
    x3d = torch.randn(5, 4, 2, requires_grad=True)
    feats3d = disc.extract_features(x3d)
    assert feats3d.shape == (5, 23)

    out3d = disc(x3d)
    assert out3d.shape == (5,)

    x_flat = x3d.reshape(5, 8).detach().clone().requires_grad_(True)
    out_flat = disc(x_flat)
    assert out_flat.shape == (5,)
    assert torch.allclose(out3d, out_flat, atol=1e-5)

    # Gradient contract: backprop through harmonic repulsion separates close points
    # Points 0 and 1 placed very close; points 2 and 3 separated
    pts = torch.tensor([
        [[0.0, 0.0], [0.01, 0.0], [2.0, 0.0], [0.0, 2.0]]
    ], dtype=torch.float32, requires_grad=True)

    feats = disc.extract_features(pts)
    p_rep = feats[0, -1]  # harmonic repulsion potential
    p_rep.backward()

    # Repulsion gradient: pts[0, 0] should be pushed in negative x direction (-),
    # pts[0, 1] should be pushed in positive x direction (+)
    assert pts.grad is not None
    grad_0 = pts.grad[0, 0]
    grad_1 = pts.grad[0, 1]
    # For potential P_rep = 1/(d + eps), grad w.r.t x_0 is -(x_0 - x_1) / (d * (d+eps)^2)
    # Since x_0 - x_1 = -0.01 < 0, grad w.r.t x_0 is positive (meaning increasing x_0 increases potential)
    # To reduce potential (repulsion), step in -grad direction: pushes x_0 left and x_1 right
    assert grad_0[0] > 0, "Repulsion gradient on x_0 should oppose displacement"
    assert grad_1[0] < 0, "Repulsion gradient on x_1 should oppose displacement"
    # Total net internal force across all 4 particles must conserve momentum (sum to zero)
    assert torch.allclose(pts.grad[0].sum(dim=0), torch.zeros(2), atol=1e-3), "Internal repulsion force must conserve momentum"

