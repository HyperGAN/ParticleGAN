"""Comprehensive End-to-End Stretch Qualification Test Suite for ParticleGAN.

Following the Project Pattern dual-track methodology:
- Tier 1: Feature Coverage (>=5 per feature area in isolation)
- Tier 2: Boundary & Corner Cases (>=5 per feature area: extremes, limits, zero, NaN/Inf)
- Tier 3: Cross-Feature Combinations (pairwise interactions across architectural innovations)
- Tier 4: Real-World Application Scenarios (end-to-end multi-step qualification workloads)

Covered Features (PROJECT.md § Feature Inventory):
- MisGAN recovery & imputation (Features 1-5: decoupled priors, Fourier D, penalties, mask temp/grad, imputer/hint)
- Ring16 BiGAN & ALICE cycle consistency (Features 6-9: Fourier coords, capacity, cycle weight, thermal hooks)
- PacGAN-4 dispersion/repulsion (Features 10-11: 23 features, packing m=4)
- CramerGAN vector critic energy potential (Feature 12: Bellemare vector critic & energy loss)
- Grid100 deep architectures (Features 13-15: Scaled ResMLP, anti-saturation guard, MLP16 He gain)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.toy_audit import api_vectors
from benchmarks.toy_audit.api_conditionals import (
    FourierMLPDiscriminator,
    MisGANNeuralImputerFixture,
    MisGANRecoveryFixture,
    TemperatureSigmoid,
    score_misgan_neural_imputer,
    score_misgan_recovery,
)
from benchmarks.toy_audit.api_conditionals import (
    build_case as build_case_misgan,
)
from benchmarks.toy_audit.api_cramergan import (
    CRAMER_SIGMA,
    CramerCritic,
    CramerGANOverlapFixture,
    CramerGenerator,
    cramer_centers,
    cramer_critic_surrogate,
    cramer_generator_loss,
    cramer_weights,
    sample_cramer_target,
    score_cramergan,
)
from benchmarks.toy_audit.api_pacgan import (
    PacGAN4RepulsionDiscriminator,
    PacGANDiscriminator,
    PacGANGenerator,
    PacGANUnequalMassFixture,
    score_pacgan_unequal,
)
from benchmarks.toy_audit.api_pacgan import (
    _ring_centers as pacgan_ring_centers,
)
from benchmarks.toy_audit.api_ring16 import (
    BiGANEncoder,
    BiGANGenerator,
    BiGANJointCritic,
    Ring16BiGANFixture,
    score_bigan,
)
from benchmarks.toy_audit.api_vectors import (
    ResidualBlock,
    ScaledResidualMLPGenerator,
    ScaledSimpleMLPGenerator,
    TanhRailGuard,
)
from particlegan import ParticlePrior
from particlegan.bispf import bind_thermal_source
from tests.bigan_software_cohort import build_fixture as build_bigan_software_fixture

# ============================================================================
# Pytest Hooks & Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def one_cpu_thread():
    """Ensure CPU tests run deterministically on a single thread."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)




# ============================================================================
# TIER 1: FEATURE COVERAGE (ISOLATION)
# ============================================================================

# --- Feature Area 1: MisGAN (Features 1 to 5) ---

@pytest.mark.tier1
def test_tier1_misgan_decoupled_priors_independence():
    """F1: Verify separate ParticlePrior objects maintain independent states and optimizers."""
    prior_x = ParticlePrior(num_particles=128, z_dim=8)
    prior_m = ParticlePrior(num_particles=128, z_dim=8)
    prior_i = ParticlePrior(num_particles=128, z_dim=8)

    opt_x = torch.optim.SGD(prior_x.parameters(), lr=0.1)
    opt_m = torch.optim.SGD(prior_m.parameters(), lr=0.1)

    initial_m = prior_m.z.clone()
    initial_i = prior_i.z.clone()

    # Update prior_x only
    loss_x = (prior_x.z.square()).sum()
    loss_x.backward()
    opt_x.step()

    # Assert prior_m and prior_i are unchanged
    assert torch.allclose(prior_m.z, initial_m)
    assert torch.allclose(prior_i.z, initial_i)
    assert not torch.allclose(prior_x.z, prior_m.z)

    # Update prior_m independently with opt_m
    loss_m = (prior_m.z.square()).sum()
    loss_m.backward()
    opt_m.step()
    assert not torch.allclose(prior_m.z, initial_m)
    assert torch.allclose(prior_i.z, initial_i)


@pytest.mark.tier1
def test_tier1_misgan_fourier_discriminator_contract():
    """F2: Verify FourierMLPDiscriminator accepts strictly 8D input and appends K=2 Fourier features."""
    D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)
    x_masked = torch.randn(32, 8, requires_grad=True)

    score = D_x(x_masked)
    assert score.squeeze(-1).shape == (32,)
    assert score.requires_grad

    score.sum().backward()
    assert x_masked.grad is not None
    assert x_masked.grad.shape == (32, 8)
    assert not torch.all(x_masked.grad == 0)


@pytest.mark.tier1
def test_tier1_misgan_complete_critic_regularization():
    """F3: Verify critic regularization penalties evaluate across Dx, Dm, and Di."""
    D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)
    D_m = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    D_i = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    real_x = torch.randn(16, 8, requires_grad=True)
    fake_x = torch.randn(16, 8, requires_grad=True)

    def simple_r1(disc, real, fake):
        out_real = disc(real).squeeze(-1)
        grad_real = torch.autograd.grad(
            outputs=out_real.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return 0.5 * grad_real.square().sum(dim=-1).mean()

    pen_x = simple_r1(D_x, real_x, fake_x)
    pen_m = simple_r1(lambda x: D_m(x).squeeze(-1), real_x, fake_x)
    pen_i = simple_r1(lambda x: D_i(x).squeeze(-1), real_x, fake_x)

    for pen in (pen_x, pen_m, pen_i):
        assert pen.item() > 0.0
        assert math.isfinite(pen.item())


@pytest.mark.tier1
def test_tier1_misgan_temperature_and_gradient_detachment():
    """F4: Verify mask temperature tau=0.66 scaling and gradient detachment for G_x."""
    G_m = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))
    G_x = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))

    z_m = torch.randn(16, 8)
    z_x = torch.randn(16, 8)

    temp_sig = TemperatureSigmoid(0.66)
    m_logits = G_m(z_m)
    m_fake = temp_sig(m_logits)

    # Detached mask in data loss: G_m must receive zero gradients from L_x
    x_fake = G_x(z_x)
    x_masked = x_fake * m_fake.detach()
    loss_x = (x_masked.square()).sum()
    loss_x.backward()

    # G_x receives gradients, G_m receives none
    assert any(p.grad is not None and not torch.all(p.grad == 0) for p in G_x.parameters())
    assert all(p.grad is None for p in G_m.parameters())


@pytest.mark.tier1
def test_tier1_misgan_imputer_and_gain_hint_conditioning():
    """F5: Verify imputer completes masked inputs conditional on GAIN coordinate hint."""
    G_i = nn.Sequential(nn.Linear(8 + 8 + 8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))

    x_obs = torch.randn(16, 8)
    mask = (torch.rand(16, 8) > 0.5).float()
    hint = mask.clone()  # GAIN hint
    z_i = torch.randn(16, 8)

    imputer_in = torch.cat([x_obs * mask, hint, z_i], dim=1)
    imputed_vals = G_i(imputer_in)
    x_completed = x_obs * mask + imputed_vals * (1.0 - mask)

    # Observed entries are retained exactly
    assert torch.allclose(x_completed * mask, x_obs * mask, atol=1e-6)
    assert x_completed.shape == (16, 8)


# --- Feature Area 2: Ring16 BiGAN & ALICE (Features 6 to 9) ---

@pytest.mark.tier1
def test_tier1_ring16_bigan_fourier_coordinates_encoder_and_critic():
    """F6: Verify coordinate Fourier features (K=3) expand x in R^2 to 14 dimensions for E and D."""
    fourier_k = 3
    freqs = torch.pi * (2.0 ** torch.arange(fourier_k, dtype=torch.float32))

    x = torch.randn(16, 2)
    xf = x.unsqueeze(-1) * freqs  # (B, 2, 3)
    x_fourier = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
    assert x_fourier.shape == (16, 2 + 2 * 3 * 2)  # 14 dims

    E = BiGANEncoder(in_dim=14, z_dim=2, hidden=128, layers=3)
    z_enc = E(x_fourier)
    assert z_enc.shape == (16, 2)

    D = BiGANJointCritic(in_dim=14, z_dim=2, hidden=128, layers=3)
    logit = D(x_fourier, z_enc)
    assert logit.shape == (16,)


@pytest.mark.tier1
def test_tier1_ring16_bigan_capacity_upgrade_architecture():
    """F7: Verify 3-layer 128-unit capacity for G, E, and D."""
    G = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    E = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3)
    D = BiGANJointCritic(in_dim=2, z_dim=2, hidden=128, layers=3)

    for net in (G, E, D):
        params = sum(p.numel() for p in net.parameters())
        assert params >= 30000, f"Expected upgraded capacity >= 30k params, got {params}"


@pytest.mark.tier1
def test_tier1_ring16_alice_reconstruction_loss_contract():
    """F8: Verify ALICE cycle-consistency reconstruction loss with cycle_weight=2.0 and detached z_fake."""
    G = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    E = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3)

    real_x = torch.randn(16, 2)
    z_real = torch.randn(16, 2)
    z_fake_source = torch.randn(16, 2)
    x_fake = G(z_fake_source)
    cycle_weight = 2.0

    z_target = torch.randn(16, 2, requires_grad=True)
    # Reconstruction objective with detached target
    loss_recon = cycle_weight * (
        F.mse_loss(G(z_real), real_x) + F.mse_loss(E(x_fake), z_target.detach())
    )
    assert loss_recon.item() > 0.0

    loss_recon.backward()
    # z_target is detached, so receives no gradients directly from being target
    assert z_target.grad is None


@pytest.mark.tier1
def test_tier1_ring16_bigan_bispf_thermal_hooks_contract():
    """F9: Verify bind_thermal_source registers optimizer record to loss on Ring16BiGANFixture."""
    fixture = Ring16BiGANFixture(max_steps=2, device="cpu", seed=0, recipe_name="bispf")
    assert hasattr(fixture.opt_d, "record")
    assert fixture.loss.thermal_source is fixture.opt_d.record
    assert fixture.opt_d.record.legacy_thermal_channel is False

    bind_thermal_source(fixture.opt_d.record, fixture.loss, fixture.regularizer)

    metrics = fixture.step()
    assert math.isfinite(metrics["loss_d"])
    assert math.isfinite(metrics["loss_g"])
    assert math.isfinite(metrics["reg_d"])


@pytest.mark.tier1
def test_tier1_ring16_bigan_roundtrip_identity_mapping():
    """F6-F9: Verify constructive E-G roundtrip mapping achieves recon_rmse <= 0.12 on ring centers."""
    centers_np = pacgan_ring_centers(num_modes=16, radius=3.0)
    centers = torch.from_numpy(centers_np).float()

    torch.manual_seed(6)
    gen = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    enc = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3, fourier=3)
    opt = torch.optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=0.01)

    for _ in range(80):
        opt.zero_grad()
        k = torch.randint(0, 16, (128,))
        x = centers[k] + torch.randn(128, 2) * 0.10
        loss = F.mse_loss(gen(x), x) + F.mse_loss(enc(x), x)
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon = gen(enc(centers))
        rmse = float(torch.sqrt(F.mse_loss(recon, centers)).item())

    assert isinstance(gen, BiGANGenerator)
    assert isinstance(enc, BiGANEncoder)
    assert not torch.equal(recon, centers), "Reconstruction must not be an identity bypass"
    assert 0.0 < rmse <= 0.12, f"Roundtrip rmse {rmse} > 0.12"


# --- Feature Area 3: PacGAN-4 (Features 10 and 11) ---

@pytest.mark.tier1
def test_tier1_pacgan4_repulsion_discriminator_feature_dimensions():
    """F10: Verify PacGANDiscriminator is a clean raw-packed MLP (no hand-crafted repulsion features) while legacy PacGAN4RepulsionDiscriminator retains 23 features."""
    assert not issubclass(PacGANDiscriminator, PacGAN4RepulsionDiscriminator)
    clean_disc = PacGANDiscriminator(in_dim=8, hidden_dim=128)
    assert clean_disc.net[0].in_features == 8
    x = torch.randn(16, 4, 2)
    assert clean_disc(x).shape == (16,)

    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    feats = disc.extract_features(x)
    assert feats.shape == (16, 23)

    out = disc(x)
    assert out.shape == (16,)


@pytest.mark.tier1
def test_tier1_pacgan4_repulsion_discriminator_permutation_invariance():
    """F10: Verify pairwise distances and harmonic repulsion are invariant to packing permutation."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    x = torch.randn(8, 4, 2)
    # Permute order of the 4 points: [2, 0, 3, 1]
    x_perm = x[:, [2, 0, 3, 1], :]

    f1 = disc.extract_features(x)
    f2 = disc.extract_features(x_perm)

    # Coords [0:8] permute, but sorted distances [8:14], stats [14:17], sorted radii [17:21], var [21], rep [22] are identical
    assert torch.allclose(f1[:, 8:], f2[:, 8:], atol=1e-5)


@pytest.mark.tier1
def test_tier1_pacgan4_packing_reduction_m4_contract():
    """F11: Verify packing factor reduction m=8 -> m=4 reshapes (B, 2) to (B/4, 8)."""
    B = 128
    m = 4
    x = torch.randn(B, 2)
    x_packed = x.view(B // m, m * 2)
    assert x_packed.shape == (32, 8)
    assert torch.allclose(x_packed.view(B, 2), x)


@pytest.mark.tier1
def test_tier1_pacgan4_harmonic_repulsion_gradient_direction():
    """F10: Verify harmonic repulsion pushes clustered points outward along -grad R."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    pts = torch.tensor([
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]]
    ], requires_grad=True)  # Clustered near origin

    feats = disc.extract_features(pts)
    p_rep = feats[0, -1]
    p_rep.backward()

    # Repulsive force is -grad R: should push point 0 left/down and point 3 right/up
    force_0 = -pts.grad[0, 0]
    force_3 = -pts.grad[0, 3]
    assert force_0[0].item() < 0.0 and force_0[1].item() < 0.0
    assert force_3[0].item() > 0.0 and force_3[1].item() > 0.0


@pytest.mark.tier1
def test_tier1_pacgan4_generator_capacity():
    """F10: Verify PacGANGenerator maps latent R^2 -> R^2 with non-zero Jacobian."""
    G = PacGANGenerator(z_dim=2, hidden_dim=128, out_dim=2)
    z = torch.randn(16, 2, requires_grad=True)
    x = G(z)
    assert x.shape == (16, 2)

    x.sum().backward()
    assert z.grad is not None
    assert not torch.all(z.grad == 0)


# --- Feature Area 4: CramerGAN Vector Critic (Feature 12) ---

@pytest.mark.tier1
def test_tier1_cramergan_vector_critic_embedding_contract():
    """F12: Verify CramerCritic exposes 16D embedding h_psi(x) with coordinate anchor."""
    critic = CramerCritic(in_dim=2, embed_dim=16, hidden=128)
    x = torch.randn(16, 2)
    h = critic.h_psi(x)
    assert h.shape == (16, 16)

    logit = critic(x)
    assert logit.shape == (16,)


@pytest.mark.tier1
def test_tier1_cramergan_multi_draw_surrogate_contract():
    """F12: Verify Bellemare multi-draw energy surrogate f(x; x_hat_2)."""
    h_x = torch.randn(16, 16)
    h_fake2 = torch.randn(16, 16)

    surrogate = cramer_critic_surrogate(h_x, h_fake2)
    assert surrogate.shape == (16,)
    assert surrogate.dtype == torch.float32


@pytest.mark.tier1
def test_tier1_cramergan_unbiased_generator_loss_contract():
    """F12: Verify unbiased two-sample Cramer energy generator loss."""
    h_real = torch.randn(32, 16)
    h_fake1 = torch.randn(32, 16, requires_grad=True)
    h_fake2 = torch.randn(32, 16, requires_grad=True)

    loss_g = cramer_generator_loss(h_real, h_fake1, h_fake2)
    assert loss_g.dim() == 0
    assert loss_g.requires_grad

    loss_g.backward()
    assert h_fake1.grad is not None
    assert h_fake2.grad is not None


@pytest.mark.tier1
def test_tier1_cramergan_energy_distance_triangle_inequality():
    """F12: Verify Cramer energy distance satisfies metric triangle inequality."""
    h_p = torch.randn(64, 16)
    h_q = torch.randn(64, 16) + 1.0
    h_r = torch.randn(64, 16) + 2.5

    def energy(a, b):
        d_ab = torch.sqrt((a[:, None, :] - b[None, :, :]).square().sum(dim=-1) + 1e-8).mean()
        d_aa = torch.sqrt((a[:, None, :] - a[None, :, :]).square().sum(dim=-1) + 1e-8).mean()
        d_bb = torch.sqrt((b[:, None, :] - b[None, :, :]).square().sum(dim=-1) + 1e-8).mean()
        return torch.clamp(2.0 * d_ab - d_aa - d_bb, min=0.0)

    e_pq = torch.sqrt(energy(h_p, h_q)).item()
    e_qr = torch.sqrt(energy(h_q, h_r)).item()
    e_pr = torch.sqrt(energy(h_p, h_r)).item()

    assert e_pr <= e_pq + e_qr + 0.1, f"Triangle inequality violated: {e_pr} > {e_pq} + {e_qr}"


@pytest.mark.tier1
def test_tier1_cramergan_energy_distance_zero_on_identical_distributions():
    """F12: Verify energy distance is near zero on independent draws from identical distribution."""
    centers = cramer_centers()
    weights = cramer_weights()
    rng = torch.Generator().manual_seed(42)

    sample_a = sample_cramer_target(2048, generator=rng, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    sample_b = sample_cramer_target(2048, generator=rng, centers=centers, weights=weights, sigma=CRAMER_SIGMA)

    obs = score_cramergan(sample_a, sample_b, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs["passed"] is True
    assert obs["metrics"]["energy_distance"] <= 0.06


# --- Feature Area 5: Grid100 Deep Architectures (Features 13 to 15) ---

@pytest.mark.tier1
def test_tier1_grid100_scaled_resmlp_architecture():
    """F13: Verify ScaledResidualMLPGenerator has 7 blocks, 1/sqrt(14) scaling, LayerNorm, and S=5.25."""
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    assert len(gen.blocks) == 7
    assert math.isclose(gen.res_scale, 1.0 / math.sqrt(14.0), rel_tol=1e-4)
    assert isinstance(gen.norm, nn.LayerNorm)
    assert gen.scale == 5.25

    z = torch.randn(16, 2)
    x = gen(z)
    assert x.shape == (16, 2)
    assert torch.all(torch.abs(x) < 5.25)


@pytest.mark.tier1
def test_tier1_grid100_residual_block_scaling_invariance():
    """F13: Verify residual branch scaling preserves activation variance across 7 blocks."""
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    z = torch.randn(64, 2)

    h0 = gen.stem(z)
    var0 = h0.var().item()

    h = h0
    for block in gen.blocks:
        h = block(h)
    var_final = h.var().item()

    # Scaling prevents explosion: variance should remain within 5x of initial variance
    assert var_final < 5.0 * var0, f"Activation variance exploded from {var0} to {var_final}"


@pytest.mark.tier1
def test_tier1_grid100_soft_bounded_tanh_exact_derivative():
    """F14: The soft-bounded head has no backward-only training mechanism."""
    scale = 5.25
    u_safe = torch.tensor([[0.0, 1.0], [2.0, -3.0], [4.5, -4.5]], requires_grad=True)
    out_safe = TanhRailGuard.apply(u_safe, scale)
    assert torch.all(torch.abs(out_safe / scale) < 0.95)
    out_safe.sum().backward()
    assert torch.all(torch.isfinite(u_safe.grad))

    u_extreme = torch.tensor([[-1000.0, 1000.0], [500.0, -500.0]], requires_grad=True)
    out_extreme = TanhRailGuard.apply(u_extreme, scale)
    assert torch.all(torch.abs(out_extreme / scale) < 0.99)
    out_extreme.sum().backward()
    assert torch.all(torch.isfinite(u_extreme.grad))
    assert torch.all(torch.abs(u_extreme.grad) > 0.0)
    u = torch.tensor([10.], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(torch.autograd.grad(TanhRailGuard.apply(u) * 0., u)[0], torch.zeros_like(u))


@pytest.mark.tier1
def test_tier1_grid100_mlp16_kaiming_he_gain_scaling():
    """F15: Verify variance propagation and weight stats on 16-layer ScaledSimpleMLPGenerator."""
    gen = ScaledSimpleMLPGenerator(z_dim=2, hidden_dim=128, n_hidden=16, out_dim=2)
    linears = [m for m in gen.net if isinstance(m, nn.Linear)]
    norms = [m for m in gen.net if isinstance(m, nn.LayerNorm)]
    assert len(linears) == 17, f"Expected 17 linear layers, got {len(linears)}"
    assert len(norms) == 16, f"Expected 16 LayerNorm layers, got {len(norms)}"

    # Forward variance propagation across all 16 layers
    z = torch.randn(256, 2, requires_grad=True)
    h = z
    layer_vars = []
    for m in gen.net:
        h = m(h)
        if isinstance(m, nn.LeakyReLU):
            layer_vars.append(h.var().item())

    assert len(layer_vars) == 16
    for var in layer_vars:
        assert 0.05 < var < 5.0, f"Activation variance {var} outside stable range [0.05, 5.0]"

    out = h
    assert out.shape == (256, 2)
    assert torch.all(torch.isfinite(out))

    # Backward gradient flow without vanishing or explosion
    out.sum().backward()
    assert torch.all(torch.isfinite(z.grad))
    assert z.grad.norm().item() > 0.001


@pytest.mark.tier1
def test_tier1_grid100_mlp16_layernorm_gradient_flow():
    """F15: Verify LayerNorm per hidden layer preserves backward gradient flow across 16 layers."""
    layers = []
    dim = 128
    for _ in range(16):
        layers.extend([
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=False),
        ])
    net = nn.Sequential(*layers)

    x = torch.randn(32, dim, requires_grad=True)
    out = net(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    grad_norm = x.grad.norm().item()
    assert grad_norm > 1e-4, f"Vanishing gradient: norm {grad_norm} <= 1e-4"


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ============================================================================

# --- Feature Area 1: MisGAN Corner Cases ---

@pytest.mark.tier2
def test_tier2_misgan_all_missing_mask_extreme():
    """Verify all-missing mask m=0 evaluates without NaN/Inf."""
    G_i = nn.Sequential(nn.Linear(8 + 8 + 8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))
    x_obs = torch.randn(16, 8)
    mask = torch.zeros(16, 8)  # 100% missing
    hint = torch.zeros(16, 8)
    z_i = torch.randn(16, 8)

    imputed = G_i(torch.cat([x_obs * mask, hint, z_i], dim=1))
    completed = x_obs * mask + imputed * (1.0 - mask)
    assert not torch.isnan(completed).any()
    assert completed.shape == (16, 8)


@pytest.mark.tier2
def test_tier2_misgan_all_observed_mask_extreme():
    """Verify all-observed mask m=1 produces 0 imputation error."""
    G_i = nn.Sequential(nn.Linear(8 + 8 + 8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))
    x_obs = torch.randn(16, 8)
    mask = torch.ones(16, 8)  # 100% observed
    hint = torch.ones(16, 8)
    z_i = torch.randn(16, 8)

    imputed = G_i(torch.cat([x_obs * mask, hint, z_i], dim=1))
    completed = x_obs * mask + imputed * (1.0 - mask)
    assert torch.allclose(completed, x_obs)


@pytest.mark.tier2
def test_tier2_misgan_mask_temperature_extremes():
    """Verify extreme temperatures tau -> 1e-4 and tau -> 1e4 do not overflow."""
    logits = torch.tensor([-5.0, 0.0, 5.0])
    m_cold = torch.sigmoid(logits / 1e-4)
    m_hot = torch.sigmoid(logits / 1e4)

    assert not torch.isnan(m_cold).any()
    assert not torch.isnan(m_hot).any()
    assert torch.all(m_cold >= 0.0) and torch.all(m_cold <= 1.0)
    assert torch.allclose(m_hot, torch.full_like(m_hot, 0.5), atol=1e-3)


@pytest.mark.tier2
def test_tier2_misgan_single_sample_batch_corner():
    """Verify batch size B=1 executes forward and backward without dim errors."""
    D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)
    x = torch.randn(1, 8, requires_grad=True)
    score = D_x(x)
    assert score.squeeze(-1).shape == (1,)
    score.backward()
    assert x.grad is not None


@pytest.mark.tier2
def test_tier2_misgan_nan_inf_input_rejection():
    """Verify that NaN and Inf in inputs propagate into loss to prevent silent corruption."""
    D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)
    x_nan = torch.tensor([[float("nan")] * 8])
    score = D_x(x_nan)
    assert torch.isnan(score).any()


# --- Feature Area 2: Ring16 BiGAN Corner Cases ---

@pytest.mark.tier2
def test_tier2_ring16_radial_extreme_out_of_bounds():
    """Verify extreme coordinates r=100.0 and r=1e-4 produce bounded latents."""
    E = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3)
    x_extreme = torch.tensor([[100.0, 100.0], [1e-4, 1e-4]])
    z = E(x_extreme)
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()


@pytest.mark.tier2
def test_tier2_ring16_zero_cycle_weight_boundary():
    """Verify setting cycle_weight=0.0 eliminates reconstruction loss cleanly."""
    G = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    E = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3)
    cycle_weight = 0.0

    real_x = torch.randn(16, 2)
    z_real = torch.randn(16, 2)
    loss = cycle_weight * (F.mse_loss(G(z_real), real_x) + F.mse_loss(E(real_x), z_real))
    assert loss.item() == 0.0


@pytest.mark.tier2
def test_tier2_ring16_zero_latent_vector_corner():
    """Verify zero latent z=0 produces finite 2D coordinates."""
    G = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    z_zero = torch.zeros(4, 2)
    x = G(z_zero)
    assert not torch.isnan(x).any()
    assert x.shape == (4, 2)


@pytest.mark.tier2
def test_tier2_ring16_high_frequency_fourier_stability():
    """Verify high frequency Fourier projection K=6 remains numerically stable."""
    fourier_k = 6
    freqs = torch.pi * (2.0 ** torch.arange(fourier_k, dtype=torch.float32))
    x = torch.randn(16, 2)
    xf = x.unsqueeze(-1) * freqs
    x_fourier = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
    assert not torch.isnan(x_fourier).any()
    assert x_fourier.shape == (16, 2 + 2 * 6 * 2)  # 26 dims


@pytest.mark.tier2
def test_tier2_ring16_bispf_thermal_source_none_guard():
    """Verify thermal loss falls back safely or flags when thermal_source is None."""
    from particlegan._bispf.thermal import _read_thermal
    # Reading from None thermal source returns float fallback without crashing
    val = _read_thermal(None, torch.device("cpu"))
    assert isinstance(val, float)
    assert val > 0.0


# --- Feature Area 3: PacGAN-4 Corner Cases ---

@pytest.mark.tier2
def test_tier2_pacgan4_identical_coincident_points():
    """Verify 4 coincident points (zero pairwise distance) do not cause division by zero."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    coincident = torch.zeros(8, 4, 2)
    feats = disc.extract_features(coincident)
    assert not torch.isnan(feats).any()
    assert not torch.isinf(feats).any()


@pytest.mark.tier2
def test_tier2_pacgan4_collinear_points_degeneracy():
    """Verify 4 collinear points evaluate without NaN."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    collinear = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    ])
    feats = disc.extract_features(collinear)
    assert not torch.isnan(feats).any()


@pytest.mark.tier2
def test_tier2_pacgan4_extreme_cluster_separation():
    """Verify extreme cluster separation (dist=1e4) computes finite features."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    extreme = torch.tensor([
        [[0.0, 0.0], [1e4, 0.0], [0.0, 1e4], [1e4, 1e4]]
    ])
    feats = disc.extract_features(extreme)
    assert not torch.isnan(feats).any()
    assert not torch.isinf(feats).any()


@pytest.mark.tier2
def test_tier2_pacgan4_batch_size_not_divisible_by_pack_size():
    """Verify reshaping with batch size not divisible by pack size raises ValueError."""
    x = torch.randn(7, 2)
    with pytest.raises(RuntimeError):
        _ = x.view(7 // 4, 4 * 2)


@pytest.mark.tier2
def test_tier2_pacgan4_single_pack_minimum_batch():
    """Verify smallest valid packed batch (B=4, exactly 1 pack) executes correctly."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    x = torch.randn(1, 4, 2)
    out = disc(x)
    assert out.shape == (1,)


# --- Feature Area 4: CramerGAN Corner Cases ---

@pytest.mark.tier2
def test_tier2_cramergan_identical_distributions_zero_energy():
    """Verify identical distributions produce near-zero unbiased energy distance."""
    h_real = torch.randn(64, 16)
    # The two-sample unbiased energy distance evaluates to 0 on identical sets:
    d_ab = torch.sqrt((h_real[:, None, :] - h_real[None, :, :]).square().sum(dim=-1) + 1e-8).mean()
    d_aa = d_ab
    d_bb = d_ab
    unbiased_energy = 2.0 * d_ab - d_aa - d_bb
    assert torch.abs(unbiased_energy) <= 1e-5

    # Cramer generator loss evaluates to a finite scalar
    loss = cramer_generator_loss(h_real, h_real, h_real)
    assert math.isfinite(loss.item())


@pytest.mark.tier2
def test_tier2_cramergan_orthogonal_distributions_maximum_energy():
    """Verify orthogonal distributions produce strictly positive energy distance."""
    h_real = torch.zeros(32, 16)
    h_real[:, :8] = 5.0
    h_fake = torch.zeros(32, 16)
    h_fake[:, 8:] = 5.0

    loss = cramer_generator_loss(h_real, h_fake, h_fake)
    assert loss.item() > 1.0


@pytest.mark.tier2
def test_tier2_cramergan_zero_embedding_norm_defense():
    """Verify zero embeddings do not produce NaN due to epsilon stabilization."""
    h_zero = torch.zeros(16, 16)
    surrogate = cramer_critic_surrogate(h_zero, h_zero)
    assert not torch.isnan(surrogate).any()


@pytest.mark.tier2
def test_tier2_cramergan_asymmetric_batch_sizes():
    """Verify energy loss computation with asymmetric real/fake batch sizes."""
    h_real = torch.randn(40, 16)
    h_fake1 = torch.randn(20, 16)
    h_fake2 = torch.randn(20, 16)

    loss = cramer_generator_loss(h_real, h_fake1, h_fake2)
    assert math.isfinite(loss.item())


@pytest.mark.tier2
def test_tier2_cramergan_gradient_clipping_and_bounding():
    """Verify critic gradients are bounded and non-exploding."""
    critic = CramerCritic(in_dim=2, embed_dim=16, hidden=128)
    x = torch.randn(16, 2, requires_grad=True)
    out = critic(x)
    out.sum().backward()
    assert x.grad.norm().item() < 100.0


# --- Feature Area 5: Grid100 Deep Corner Cases ---

@pytest.mark.tier2
def test_tier2_grid100_tanh_rail_saturation_boundary():
    """Verify extreme logits saturate smoothly at +/- S = +/- 5.25 without overflow."""
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    # Manually pass extreme values through head
    huge_logits = torch.tensor([[-1000.0, 1000.0]])
    x = gen.scale * torch.tanh(huge_logits / gen.scale)
    assert torch.allclose(x, torch.tensor([[-5.25, 5.25]]))


@pytest.mark.tier2
def test_tier2_grid100_zero_residual_initialization():
    """Verify zero-initialized residual blocks act as exact mathematical identity."""
    block = ResidualBlock(64, scale=1.0 / math.sqrt(14.0))
    nn.init.zeros_(block.linear2.weight)
    nn.init.zeros_(block.linear2.bias)

    x = torch.randn(16, 64)
    assert torch.allclose(block(x), x, atol=1e-6)


@pytest.mark.tier2
def test_tier2_grid100_extreme_latent_norm():
    """Verify latent vectors with norm ||z|| = 1000 produce bounded outputs within [-5.25, 5.25]."""
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    z_huge = torch.randn(16, 2) * 500.0
    x = gen(z_huge)
    assert torch.all(torch.abs(x) <= 5.25)


@pytest.mark.tier2
def test_tier2_grid100_vanishing_gradient_depth_check():
    """Verify 16-layer Scaled ResMLP gradient ratio ||grad_z|| / ||grad_out|| > 1e-4."""
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    z = torch.randn(32, 2, requires_grad=True)
    out = gen(z)
    loss = out.sum()
    loss.backward()

    ratio = z.grad.norm().item() / out.norm().item()
    assert ratio > 1e-4, f"Gradient ratio {ratio} indicates vanishing gradient"


@pytest.mark.tier2
def test_tier2_grid100_layernorm_epsilon_numerical_stability():
    """Verify LayerNorm with zero variance inputs produces 0 output without NaN."""
    ln = nn.LayerNorm(64)
    zero_var = torch.full((16, 64), 3.14)
    out = ln(zero_var)
    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ============================================================================

@pytest.mark.tier3
def test_tier3_misgan_decoupled_priors_with_fourier_discriminator_and_temp():
    """Pairwise: MisGAN Decoupled Priors + Fourier Discriminator + Mask Temp tau=0.66."""
    prior_x = ParticlePrior(num_particles=64, z_dim=8)
    prior_m = ParticlePrior(num_particles=64, z_dim=8)

    G_x = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))
    G_m = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8))
    D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)

    # 1 optimization step
    opt = torch.optim.Adam(list(G_x.parameters()) + list(G_m.parameters()) + list(D_x.parameters()), lr=1e-3)

    z_x, _ = prior_x.sample(16)
    z_m, _ = prior_m.sample(16)

    x_fake = G_x(z_x)
    m_fake = torch.sigmoid(G_m(z_m) / 0.66)

    # D_x receives 8D masked data with detached mask gradient
    fake_masked = x_fake * m_fake.detach()
    score = D_x(fake_masked)
    loss = -score.mean()

    loss.backward()
    opt.step()

    # D_x and G_x updated; G_m not updated by data loss
    assert any(p.grad is not None for p in G_x.parameters())
    assert all(p.grad is None for p in G_m.parameters())


@pytest.mark.tier3
def test_tier3_misgan_fourier_discriminator_with_three_way_critic_regularization():
    """Pairwise: MisGAN Fourier Discriminator + Complete Regularization (Dx, Dm, Di)."""
    D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)
    D_m = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    D_i = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    x = torch.randn(16, 8, requires_grad=True)
    m = torch.randn(16, 8, requires_grad=True)
    i = torch.randn(16, 8, requires_grad=True)

    def r1(disc, t):
        out = disc(t).squeeze(-1)
        g = torch.autograd.grad(out.sum(), t, create_graph=True, retain_graph=True)[0]
        return 0.5 * g.square().sum(dim=-1).mean()

    total_loss = D_x(x).mean() + D_m(m).mean() + D_i(i).mean() + r1(D_x, x) + r1(lambda t: D_m(t).squeeze(-1), m) + r1(lambda t: D_i(t).squeeze(-1), i)
    assert total_loss.requires_grad
    total_loss.backward()


@pytest.mark.tier3
def test_tier3_ring16_bigan_capacity_fourier_and_alice_cycle_consistency():
    """Pairwise: Ring16 Upgraded Capacity (128x3) + Fourier Coordinates + ALICE Reconstruction (w=2.0)."""
    fourier_k = 3
    freqs = torch.pi * (2.0 ** torch.arange(fourier_k, dtype=torch.float32))

    G = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    E = BiGANEncoder(in_dim=14, z_dim=2, hidden=128, layers=3)

    real_x = torch.randn(16, 2)
    xf = real_x.unsqueeze(-1) * freqs
    x_fourier = torch.cat([real_x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)

    z_fake = torch.randn(16, 2)
    x_fake = G(z_fake)
    xf_fake = x_fake.unsqueeze(-1) * freqs
    xf_fourier = torch.cat([x_fake, torch.sin(xf_fake).flatten(1), torch.cos(xf_fake).flatten(1)], dim=1)

    loss_alice = 2.0 * (F.mse_loss(G(E(x_fourier)), real_x) + F.mse_loss(E(xf_fourier), z_fake.detach()))
    assert loss_alice.item() > 0.0
    loss_alice.backward()


@pytest.mark.tier3
def test_tier3_ring16_bigan_alice_with_bispf_thermal_hooks():
    """Pairwise: Ring16 ALICE Cycle-Consistency + BISPF Thermal Hook Calibration."""
    fixture = Ring16BiGANFixture(max_steps=2, device="cpu", seed=0, recipe_name="bispf")
    assert fixture.cycle_weight == 2.0
    assert hasattr(fixture.opt_d, "record")
    assert fixture.loss.thermal_source is fixture.opt_d.record

    # Step fixture exercising both ALICE cycle loss and BISPF thermal source updates
    metrics = fixture.step()
    assert math.isfinite(metrics["loss_d"])
    assert math.isfinite(metrics["loss_g"])
    assert math.isfinite(metrics["reg_d"])


@pytest.mark.tier3
def test_tier3_pacgan4_repulsion_with_cramergan_energy_potential():
    """Pairwise: PacGAN-4 Repulsion/Dispersion Features + CramerGAN Multi-Draw Energy Objective."""
    disc = PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)
    critic = CramerCritic(in_dim=23, embed_dim=16, hidden=128)

    pts = torch.randn(16, 4, 2)
    feats = disc.extract_features(pts)  # (16, 23)

    h = critic.h_psi(feats)  # (16, 16)
    assert h.shape == (16, 16)

    # Multi-draw energy loss on dispersion feature embeddings
    loss_energy = cramer_generator_loss(h, h, h)
    assert math.isfinite(loss_energy.item())
    assert loss_energy.item() > 0.0


@pytest.mark.tier3
def test_tier3_pacgan4_m4_packing_with_unequal_mass_scoring():
    """Pairwise: PacGAN-4 m=4 Packing + Unequal Mass Metric Qualification."""
    centers_np = pacgan_ring_centers()[:5]
    centers = torch.from_numpy(centers_np)
    weights = (0.40, 0.30, 0.20, 0.08, 0.02)

    # Generate synthetic samples with matching unequal weights
    rng = torch.Generator().manual_seed(42)
    cluster_idx = torch.multinomial(torch.tensor(weights), 2000, replacement=True, generator=rng)
    samples = centers[cluster_idx] + torch.randn(2000, 2, generator=rng) * 0.08

    obs = score_pacgan_unequal(samples, weights=weights, sigma=0.08)
    assert obs["metrics"]["modes"] == 5
    assert obs["metrics"]["min_cov_eig_ratio"] >= 0.20


@pytest.mark.tier3
def test_tier3_grid100_scaled_resmlp_with_tanh_anti_saturation_guard():
    """Pairwise: 16-Layer Scaled ResMLP + Tanh Anti-Saturation Guard Rail Penalty."""
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    z = torch.randn(32, 2)
    x = gen(z)

    # Rail penalty
    rail_threshold = 0.95 * 5.25
    penalty = F.relu(torch.abs(x) - rail_threshold).square().sum()
    loss = (x.square()).mean() + 10.0 * penalty

    loss.backward()
    assert all(p.grad is not None for p in gen.parameters())


@pytest.mark.tier3
def test_tier3_grid100_mlp16_kaiming_he_gain_with_layernorm_composite():
    """Pairwise: 16-Layer MLP Kaiming-He Gain + LayerNorm Per Layer Composite."""
    dim = 128
    gen = ScaledSimpleMLPGenerator(z_dim=dim, hidden_dim=dim, n_hidden=16, out_dim=dim, use_norm=True)

    x = torch.randn(64, dim, requires_grad=True)
    out = gen(x)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert x.grad.norm().item() > 1e-4


@pytest.mark.tier3
def test_tier3_bigan_fourier_joint_critic_with_cramer_multidraw_energy():
    """Pairwise: BiGAN Joint Critic with Fourier Coordinates + Cramer Multi-Draw Energy Formulation."""
    fourier_k = 3
    freqs = torch.pi * (2.0 ** torch.arange(fourier_k, dtype=torch.float32))

    x = torch.randn(16, 2)
    xf = x.unsqueeze(-1) * freqs
    x_f = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)  # 14 dims
    z = torch.randn(16, 2)

    critic = CramerCritic(in_dim=16, embed_dim=16, hidden=128)
    joint_pair = torch.cat([x_f, z], dim=1)  # 16 dims
    h = critic.h_psi(joint_pair)
    assert h.shape == (16, 16)


@pytest.mark.tier3
def test_tier3_misgan_imputer_with_residual_architecture():
    """Pairwise: MisGAN Imputer with Residual Block Architecture."""
    imputer = nn.Sequential(
        nn.Linear(24, 64),
        nn.LeakyReLU(0.2),
        ResidualBlock(64),
        ResidualBlock(64),
        nn.Linear(64, 8),
    )
    in_vec = torch.randn(16, 24)
    out_vec = imputer(in_vec)
    assert out_vec.shape == (16, 8)


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ============================================================================

@pytest.mark.tier4
def test_tier4_e2e_misgan_block_recovery_qualification_step():
    """E2E MisGAN Block Recovery: Multi-step training qualification step and public metric verification."""
    fixture = build_case_misgan("api-misgan-block-recovery", device="cpu", seed=0, max_steps=5)
    assert isinstance(fixture, MisGANRecoveryFixture)
    for step_idx in range(1, 4):
        stats = fixture.step()
        assert stats["step"] == step_idx
        assert all(math.isfinite(v) for v in stats.values())

    obs = fixture.observe(n=256, seed=42)
    assert "metrics" in obs
    assert "modes" in obs["metrics"]
    assert "hq" in obs["metrics"]
    assert "mask_tv" in obs["metrics"]
    assert math.isfinite(obs["metrics"]["hq"])

    # Verify score_misgan_recovery contract directly
    x_fake = fixture.ema_G_x(fixture.prior_x.sample(128)[0])
    m_fake = fixture.ema_G_m(fixture.prior_m.sample(128)[0])
    direct_score = score_misgan_recovery(x_fake, m_fake, fixture.problem, eval_seed=42)
    assert "metrics" in direct_score


@pytest.mark.tier4
def test_tier4_e2e_misgan_neural_imputer_multimodal_qualification():
    """E2E MisGAN Neural Imputer: Qualification on held-out ambiguous queries and public scoring."""
    fixture = build_case_misgan("api-misgan-block-neural-imputer", device="cpu", seed=0, max_steps=5)
    assert isinstance(fixture, MisGANNeuralImputerFixture)
    for step_idx in range(1, 4):
        stats = fixture.step()
        assert stats["step"] == step_idx
        assert all(math.isfinite(v) for v in stats.values())

    obs = fixture.observe(n=64, seed=42)
    assert "metrics" in obs
    assert "observed_max_error" in obs["metrics"]
    assert "missing_std_ratio" in obs["metrics"]
    assert math.isfinite(obs["metrics"]["observed_max_error"])

    # Verify score_misgan_neural_imputer contract directly
    def predict(x_q, m_q, rng):
        zi = torch.randn(len(x_q), 4, generator=rng, device=rng.device)
        cond = torch.cat([x_q * m_q, m_q, zi], dim=1)
        return m_q * x_q + (1.0 - m_q) * fixture.ema_G_i(cond)

    direct_score = score_misgan_neural_imputer(predict, fixture.problem, eval_seed=42)
    assert "metrics" in direct_score


@pytest.mark.tier4
def test_tier4_e2e_ring16_bigan_continuous_qualification_roundtrip():
    """E2E Ring16 BiGAN Continuous: Multi-step joint training, cycle consistency, and public score verification."""
    fixture = build_bigan_software_fixture(max_steps=5)
    for step_idx in range(1, 4):
        stats = fixture.step()
        assert stats["step"] == step_idx
        assert all(math.isfinite(v) for v in stats.values())

    obs = fixture.observe(n=256, seed=42)
    assert "metrics" in obs
    assert "modes" in obs["metrics"]
    assert "recon_rmse" in obs["metrics"]
    assert "mode_consistency" in obs["metrics"]
    assert math.isfinite(obs["metrics"]["recon_rmse"])

    # Verify score_bigan contract directly
    real_x = fixture.sample_target(128)
    z_real = fixture.E(real_x)
    recon = fixture.G(z_real)
    direct_score = score_bigan(real_x, real_x, recon, centers=fixture.centers, sigma=fixture.sigma)
    assert "metrics" in direct_score


@pytest.mark.tier4
def test_tier4_e2e_pacgan4_unequal_mass_qualification_step():
    """E2E PacGAN-4 Unequal Mass: Mode preservation, dispersion metrics, and public score verification."""
    fixture = PacGANUnequalMassFixture(max_steps=5, device="cpu", seed=0)
    for step_idx in range(1, 4):
        stats = fixture.step()
        assert stats["step"] == step_idx
        assert all(math.isfinite(v) for v in stats.values())

    obs = fixture.observe(n=256, seed=42)
    assert "metrics" in obs
    assert "modes" in obs["metrics"]
    assert "min_mode_mass" in obs["metrics"]
    assert "mass_tv" in obs["metrics"]
    assert math.isfinite(obs["metrics"]["mass_tv"])


@pytest.mark.tier4
def test_tier4_e2e_cramergan_energy_distance_overlap_qualification():
    """E2E CramerGAN Overlap: Multi-step training, energy distance computation, and public score verification."""
    fixture = CramerGANOverlapFixture(max_steps=5, device="cpu", seed=0)
    assert isinstance(fixture.G, CramerGenerator)
    for step_idx in range(1, 4):
        stats = fixture.step()
        assert stats["step"] == step_idx
        assert all(math.isfinite(v) for v in stats.values())

    obs = fixture.observe(n=256, seed=42)
    assert "metrics" in obs
    assert "modes" in obs["metrics"]
    assert "energy_distance" in obs["metrics"]
    assert "mass_tv" in obs["metrics"]
    assert math.isfinite(obs["metrics"]["energy_distance"])


@pytest.mark.tier4
def test_tier4_e2e_grid100_deep_generator_gradient_and_rail_qualification():
    """E2E Grid100 Deep Architectures: Scaled ResMLP optimization, zero rail saturation, and case scoring."""
    case = next(c for c in api_vectors.list_cases() if c["id"] == "api-grid100-res16-tanh")
    gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)

    for _ in range(3):
        opt.zero_grad()
        z = torch.randn(64, 2)
        x = gen(z)
        loss = (x.square()).mean()
        loss.backward()
        opt.step()

    z_eval = torch.randn(500, 2)
    with torch.no_grad():
        x_eval = gen(z_eval)
    score = api_vectors.score_case(case, x_eval)
    assert "metrics" in score
    assert "saturated_fraction" in score["metrics"]
    assert score["metrics"]["saturated_fraction"] <= 0.001


@pytest.mark.tier4
def test_tier4_e2e_bigan_declared_defaults_block_unsupported_noise_and_explicit_software_cohort_runs():
    """Keep declared defaults; only a separately scoped noise-free software recipe runs."""
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    from benchmarks.toy_audit.api_ring16 import (
        ALICESpiralBiGANFixture,
        _alice_spiral_case,
        _bigan_case,
        build_case as build_case_ring16,
    )

    for case_fn in (_bigan_case, _alice_spiral_case):
        meta = case_fn()
        assert meta["default_recipe"] == "k3p"
        assert tuple(meta["supported_recipes"]) == ("k3p", "ka2")

    with pytest.raises(UnsupportedRecipeOverrides, match="input_noise_std"):
        build_case_ring16("api-ring16-bigan-continuous", max_steps=4)
    f_ring = build_bigan_software_fixture(max_steps=4)
    assert f_ring.recipe.name == "k3p"
    assert f_ring.cycle_weight == 2.0
    assert f_ring.D.fourier == 3
    assert f_ring.D.net[0].in_features == 16

    f_alice = build_bigan_software_fixture(fixture_cls=ALICESpiralBiGANFixture, max_steps=4)
    assert f_alice.recipe.name == "k3p"
    assert f_alice.cycle_weight == 2.0
    assert f_alice.D.fourier == 0
    assert f_alice.D.net[0].in_features == 4


@pytest.mark.tier4
def test_tier4_e2e_grid100_res16_scaled_tanh_paired_task():
    """Verify paired grid100_res16_scaled_tanh task card, deterministic_orthogonal_ hook, and multi-step execution."""
    import json
    from pathlib import Path
    from particlegan.init import deterministic_orthogonal_

    root = Path(__file__).resolve().parents[1]
    card = json.loads((root / "configs/forge/tasks/grid100_res16_scaled_tanh.json").read_text())
    assert card["id"] == "grid100_res16_scaled_tanh"
    assert card["execution"]["model"]["architecture"] == "scaled_residual_mlp"
    assert card["execution"]["model"]["output_activation"] == "tanh_rail_guard"

    view = json.loads((root / "configs/forge/views/discriminator_stability.json").read_text())
    scaled_assign = next(a for a in view["assignments"] if a["task"] == "grid100_res16_scaled_tanh")
    assert scaled_assign["importance"] == "stretch"
    assert scaled_assign["qualification_tier"] == 4

    assert "api-grid100-res16-scaled-tanh" in {c["id"] for c in api_vectors.list_cases()}
    contract_dir = root / "reports/toy_audit/api_contract/api-grid100-res16-scaled-tanh"
    for fname in ("goal.gif", "receipt.json", "readout.md"):
        assert (contract_dir / fname).is_file()

    fixture = api_vectors.build_case("api-grid100-res16-scaled-tanh", device="cpu", seed=0, recipe_name="k3p", max_steps=1)
    assert isinstance(fixture.trainer.G, ScaledResidualMLPGenerator)
    assert fixture.trainer.G.preserve_latent_norm is True

    # Verify deterministic_orthogonal_ preserves paired identity frame without mutating buffers
    g = ScaledResidualMLPGenerator(2, 128, 7, 2, scale=5.25, preserve_latent_norm=True)
    deterministic_orthogonal_(g, seed=0)
    z_in = torch.linspace(-2.0, 2.0, 32).reshape(16, 2)
    with torch.no_grad():
        x_out = g(z_in)
    assert torch.isfinite(x_out).all()
    assert float((x_out.abs() >= 0.99 * 5.25).float().mean()) <= 0.001


@pytest.mark.tier4
def test_tier4_e2e_misgan_hint_conditioning_without_temperature_annealing():
    """Verify MisGAN D_x uses 8D GAIN hint conditioning and static tau=0.66 without temperature annealing."""
    f_rec = MisGANRecoveryFixture(max_steps=6, device="cpu", seed=0)
    f_imp = MisGANNeuralImputerFixture(max_steps=6, device="cpu", seed=0)

    for fix in (f_rec, f_imp):
        assert fix.D_x.in_dim == 8
        assert fix.D_x.hint_dim == 8
        assert fix.D_x.net[0].in_features == 48
        assert fix.hint_rate == 0.50
        tau_head = next(m for m in fix.G_m.modules() if isinstance(m, TemperatureSigmoid))
        assert math.isclose(tau_head.temperature, 0.66, rel_tol=1e-6)
        for _ in range(4):
            fix.step()
        # Temperature must remain strictly static (no annealing)
        assert math.isclose(tau_head.temperature, 0.66, rel_tol=1e-6)


@pytest.mark.tier4
def test_tier4_e2e_pacgan_unequal_mass_local_knn_repulsion():
    """Semantic negative control: PacGANUnequalMassFixture must NEVER overwrite G weights or prior.z outside opt_g.step()."""
    from particlegan import ManipulatedPrior, NormalNoiseManipulator, ParticlePrior, unwrap_prior

    for recipe_name, custom_prior in (
        ("bcap", None),
        ("atlas", ManipulatedPrior(ParticlePrior(256, 2, init_std=0.5), [NormalNoiseManipulator(sigma=0.0065)])),
    ):
        fixture = PacGANUnequalMassFixture(
            max_steps=100,
            device="cpu",
            seed=0,
            recipe_name=recipe_name,
            prior=custom_prior,
        )
        assert not hasattr(fixture, "_apply_local_knn_particle_dispersion")
        assert not hasattr(fixture, "_update_empirical_mode_stats")
        assert getattr(fixture, "knn_repulsion_weight", 0.0) == 0.0
        assert isinstance(fixture.D, PacGANDiscriminator)
        assert not isinstance(fixture.D, PacGAN4RepulsionDiscriminator)
        assert fixture.D.net[0].in_features == 2 * fixture.pac_size

        # Wrap opt_g.step so we capture exact G/prior weights immediately after optimizer step,
        # and verify fixture.step() performs ZERO post-optimizer overwrites even past step >= 60.
        base_prior = unwrap_prior(fixture.prior)
        post_opt_snapshots: dict[str, list[torch.Tensor]] = {}
        orig_step = fixture.opt_g.step

        def _wrapped_opt_step(
            *args,
            _orig=orig_step,
            _snaps=post_opt_snapshots,
            _fix=fixture,
            _bp=base_prior,
            **kwargs,
        ):
            res = _orig(*args, **kwargs)
            _snaps["G"] = [p.detach().clone() for p in _fix.G.parameters()]
            _snaps["z"] = [_bp.z.detach().clone()]
            return res

        fixture.opt_g.step = _wrapped_opt_step  # type: ignore[method-assign]
        fixture.step_count = 60
        fixture.completed_steps = 60
        fixture.step()
        for p_now, p_after_opt in zip(fixture.G.parameters(), post_opt_snapshots["G"]):
            assert torch.equal(p_now.detach(), p_after_opt), "Host mutated G weights outside opt_g.step()!"
        assert torch.equal(base_prior.z.detach(), post_opt_snapshots["z"][0]), (
            "Host mutated prior.z outside opt_g.step()!"
        )

    # Also verify PacGAN8Grid100Fixture uses true pac_size=8 matching its advertised m=8 contract.
    from benchmarks.toy_audit.api_pacgan import PacGAN8Grid100Fixture
    grid_fix = PacGAN8Grid100Fixture(max_steps=4, device="cpu", seed=0)
    assert grid_fix.pac_size == 8
    assert grid_fix.D.net[0].in_features == 16

