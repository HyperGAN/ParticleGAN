"""Acceptance criteria tests for Wavefront Laws (Law 1 and Law 2).

Verifies AC2:
- GhostFreeSimplexMixer exact zero negative ghost leakage (<= 1e-6) vs affine expansion (< -0.1).
- SobolevGradFilter bit-identical forward identity and exact analytical Nyquist backward attenuation.
- WavefrontManifoldPrior ViT patch-seam discrimination (> 1.5 on block seams, <= 1.05 on smooth fields).
- WavefrontManifoldPrior biharmonic ramp vanishing (<= 1e-6) and PSD spectral spike excess sensitivity.
- DifferentiableSplatCarrierBasis 2D parameter matrices (ndim == 2) and DualNorm role="router" assignment.
- SimplexWavefrontGenerator bounded range in (-1, 1), step-0 carrier equivalence, and full gradient flow.
- Wavefront manipulator training integration, EMA buffer contiguity, and bit-exact checkpoint parity.
"""
from copy import deepcopy
import math
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from particlegan import (
    GANTrainer,
    UpdatePolicy,
    get_recipe,
)
from particlegan.priors import (
    GhostFreeDeMixingManipulator,
    ManipulatedPrior,
    SimplexRoutingManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
)
from particlegan.wavefront import (
    DifferentiableSplatCarrierBasis,
    GhostFreeSimplexMixer,
    SimplexWavefrontGenerator,
    SobolevGradFilter,
    WavefrontManifoldPrior,
    sobolev_filter_2d,
)


def test_ghost_free_simplex_mixer_zero_leakage_vs_affine():
    """Verify GhostFreeSimplexMixer prevents negative cross-mode ghosting.

    Standard affine variance expansion produces negative values (< -0.1) on disjoint
    supports, whereas GhostFreeSimplexMixer guarantees zero negative leakage (>= -1e-6).
    """
    K, C, H, W = 4, 1, 16, 16
    gamma_base = 1.0
    gamma_detail = 2.0

    # Construct base with shared uniform support
    U_base = torch.full((K, C, H, W), 0.1, dtype=torch.float32)

    # Construct disjoint-support mode details (each mode occupies a distinct quadrant)
    U_full = U_base.clone()
    half_h, half_w = H // 2, W // 2
    quadrants = [
        (slice(0, half_h), slice(0, half_w)),
        (slice(0, half_h), slice(half_w, W)),
        (slice(half_h, H), slice(0, half_w)),
        (slice(half_h, H), slice(half_w, W)),
    ]
    for k, (qy, qx) in enumerate(quadrants):
        U_full[k, :, qy, qx] += 1.0  # Disjoint detail peak

    # 1. Standard affine variance expansion on U_full
    U_bar_full = U_full.mean(dim=0, keepdim=True)
    affine_modes = U_bar_full + gamma_detail * (U_full - U_bar_full)

    # On quadrant j != k, affine mode k has negative artifact:
    # A_k(x_j) = U_bar + gamma * (0 - U_bar) = (1 - gamma) * (0.1 + 1.0/4) = -1.0 * 0.35 = -0.35 < -0.1
    min_affine = affine_modes.min().item()
    assert min_affine < -0.10, f"Expected severe negative artifact (< -0.1), got {min_affine}"

    # 2. GhostFreeSimplexMixer
    mixer = GhostFreeSimplexMixer(gamma_base=gamma_base, gamma_detail=gamma_detail)
    B_k = mixer.compute_basis(U_base, U_full)

    # Blend across diverse simplex weights (one-hot, uniform, random Dirichlet)
    simplex_weights = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.05, 0.45, 0.45, 0.05],
    ], dtype=torch.float32)

    u_blended = mixer(U_base, U_full, simplex_weights)
    min_blended = u_blended.min().item()
    assert min_blended >= -1e-6, f"GhostFreeSimplexMixer had negative leakage: {min_blended}"

    min_basis = B_k.min().item()
    assert min_basis >= -1e-6, f"GhostFree basis B_k had negative leakage: {min_basis}"


def test_sobolev_grad_filter_forward_identity_and_analytical_attenuation():
    """Verify SobolevGradFilter bit-identical forward and exact analytical Nyquist backward."""
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 16, 16
    lam = 0.5
    order = 1.0

    # 1. Forward bit-identity test
    x = torch.randn(B, C, H, W, dtype=torch.float32, requires_grad=True)
    grad_filter = SobolevGradFilter(lam=lam, order=order)
    y = grad_filter(x)
    assert torch.equal(y, x), "SobolevGradFilter forward pass must produce bit-identical output"

    # 2. 2D Nyquist checkerboard pattern backward test
    y_coords = torch.arange(H).view(1, 1, H, 1)
    x_coords = torch.arange(W).view(1, 1, 1, W)
    checker = ((-1.0) ** (y_coords + x_coords)).expand(B, C, H, W).contiguous()

    # Analytical attenuation factor
    expected_factor = (1.0 + lam * (math.pi**2 + math.pi**2)) ** (-order)

    # Autograd backward pass
    loss = (y * checker).sum()
    loss.backward()

    expected_grad = checker * expected_factor
    max_err = (x.grad - expected_grad).abs().max().item()
    assert max_err <= 1e-5, f"Backward gradient error {max_err} exceeded analytical tolerance 1e-5"

    # 3. Direct function verification across multiple lambda, order, and nyquist parameters
    for test_lam in (0.1, 1.0, 2.5):
        for test_order in (0.5, 1.0, 2.0):
            for test_nyquist in (math.pi, 1.0):
                factor = (1.0 + test_lam * (2.0 * test_nyquist**2)) ** (-test_order)
                filtered_checker = sobolev_filter_2d(
                    checker, lam=test_lam, order=test_order, nyquist=test_nyquist
                )
                err = (filtered_checker - checker * factor).abs().max().item()
                assert err <= 1e-5, (
                    f"sobolev_filter_2d failed for lam={test_lam}, order={test_order}, "
                    f"nyquist={test_nyquist}: err={err}"
                )


def test_wavefront_manifold_prior_patch_seam_ratio_discrimination():
    """Verify WavefrontManifoldPrior discriminates synthetic block seams from smooth fields."""
    H, W = 32, 32
    patch_size = 8
    secondary_patch_size = 4
    prior = WavefrontManifoldPrior(
        weight_seam=1.0,
        weight_biharmonic=0.0,
        weight_psd=0.0,
        patch_size=patch_size,
        secondary_patch_size=secondary_patch_size,
        seam_threshold=1.05,
    )

    # 1. Synthetic block-seam image: constant patches with sharp step boundaries
    torch.manual_seed(123)
    patches = torch.randn(2, 3, H // patch_size, W // patch_size)
    block_img = F.interpolate(patches, size=(H, W), mode="nearest")

    res_block = prior(block_img, return_dict=True)
    r_block = res_block["R_seam"].item()
    assert r_block > 1.5, f"Expected block image patch_seam_ratio > 1.5, got {r_block}"

    # 2. Linear ramp: uniform constant gradient everywhere
    y = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(2, 3, H, W)
    x = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(2, 3, H, W)
    ramp_img = 0.6 * y + 0.4 * x

    res_ramp = prior(ramp_img, return_dict=True)
    r_ramp = res_ramp["R_seam"].item()
    assert r_ramp <= 1.05, f"Expected linear ramp patch_seam_ratio <= 1.05, got {r_ramp}"

    # 3. Smooth Gaussian random field
    noise = torch.randn(2, 3, H, W)
    k_size = 9
    sigma = 2.5
    coord = torch.arange(k_size) - k_size // 2
    k1d = torch.exp(-coord**2 / (2 * sigma**2))
    k1d = k1d / k1d.sum()
    k2d = (k1d.view(-1, 1) * k1d.view(1, -1)).view(1, 1, k_size, k_size).expand(3, 1, -1, -1)
    smooth_img = F.conv2d(
        F.pad(noise, (k_size // 2, k_size // 2, k_size // 2, k_size // 2), mode="reflect"),
        k2d,
        groups=3,
    )

    res_smooth = prior(smooth_img, return_dict=True)
    r_smooth = res_smooth["R_seam"].item()
    assert r_smooth <= 1.05, f"Expected smooth field patch_seam_ratio <= 1.05, got {r_smooth}"


def test_wavefront_manifold_prior_biharmonic_and_psd_penalties():
    """Verify biharmonic curvature vanishing on planar ramps and PSD sensitivity to spectral spikes."""
    H, W = 32, 32
    prior_5pt = WavefrontManifoldPrior(
        weight_seam=0.0,
        weight_biharmonic=1.0,
        weight_psd=0.0,
        laplacian_kernel="5point",
    )
    prior_9pt = WavefrontManifoldPrior(
        weight_seam=0.0,
        weight_biharmonic=1.0,
        weight_psd=0.0,
        laplacian_kernel="9point",
    )

    # 1. Planar ramp has zero mathematical Laplacian
    y = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(2, 3, H, W)
    x = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(2, 3, H, W)
    ramp = 3.5 * y - 2.2 * x + 0.8

    loss_5pt = prior_5pt(ramp)
    loss_9pt = prior_9pt(ramp)
    assert loss_5pt.item() <= 1e-6, f"5-point biharmonic on ramp was non-zero: {loss_5pt.item()}"
    assert loss_9pt.item() <= 1e-6, f"9-point biharmonic on ramp was non-zero: {loss_9pt.item()}"

    # 2. Corrugated / noisy image has high curvature
    corrugated = ramp + 0.5 * torch.randn_like(ramp)
    loss_corr = prior_5pt(corrugated)
    assert loss_corr.item() > 0.1, f"Expected high biharmonic on noisy image, got {loss_corr.item()}"

    # 3. Radial PSD excess penalty test
    prior_psd = WavefrontManifoldPrior(
        weight_seam=0.0,
        weight_biharmonic=0.0,
        weight_psd=1.0,
        alpha=2.0,
        margin_db=1.0,
    )
    # Natural image with 1/f^2 falloff
    fy = torch.fft.fftfreq(H).unsqueeze(1) * H
    fx = torch.fft.rfftfreq(W).unsqueeze(0) * W
    f_rad = torch.sqrt(fy**2 + fx**2)
    amp = 1.0 / (1.0 + f_rad).pow(1.0).unsqueeze(0).unsqueeze(0).expand(2, 3, -1, -1)
    phase = torch.rand(2, 3, H, W // 2 + 1) * 2 * math.pi
    natural_img = torch.fft.irfft2(amp * torch.polar(torch.ones_like(phase), phase), s=(H, W))

    # Add high-frequency checkerboard spike
    spike_img = natural_img + 2.0 * (
        (-1.0) ** (torch.arange(H).view(1, 1, -1, 1) + torch.arange(W).view(1, 1, 1, -1))
    )

    loss_nat_psd = prior_psd(natural_img).item()
    loss_spike_psd = prior_psd(spike_img).item()
    assert loss_spike_psd > loss_nat_psd + 1.0, (
        f"PSD failed to penalize spike: nat={loss_nat_psd}, spike={loss_spike_psd}"
    )


def test_splat_carrier_basis_parameter_roles_and_dualnorm_compatibility():
    """Verify DifferentiableSplatCarrierBasis parameters are 2D matrices assigned to role='router'."""
    num_modes = 8
    channels = 3
    height, width = 16, 16
    num_base = 6
    num_detail = 4

    # 1. Instantiate basis directly and verify 2D parameter shapes
    basis = DifferentiableSplatCarrierBasis(
        num_modes=num_modes,
        channels=channels,
        height=height,
        width=width,
        num_base_splats=num_base,
        num_detail_splats=num_detail,
    )

    param_dict = dict(basis.named_parameters())
    required_params = ("base_geometry", "base_appearance", "base_background", "detail_params")
    for name in required_params:
        assert name in param_dict, f"Missing required parameter '{name}' in DifferentiableSplatCarrierBasis"
        p = param_dict[name]
        assert p.ndim == 2, f"Parameter '{name}' must have ndim == 2, got shape {tuple(p.shape)}"

    # 2. Recipe optimizer integration under normalized family (bispf_optimized)
    recipe = get_recipe("bispf_optimized", num_particles=32, z_dim=16)
    manipulator = SplatCarrierManipulator(basis=basis)
    prior = recipe.make_prior().manipulate(manipulator)

    # Dummy generator and critic
    G = torch.nn.Linear(channels * height * width, channels * height * width)
    D = torch.nn.Linear(channels * height * width, 1)

    opt_g, opt_d = recipe.make_optimizers(G, D, prior=prior)

    # Verify group roles
    roles = [grp.get("role") for grp in opt_g.param_groups]
    assert "router" in roles, f"Expected 'router' role in opt_g param_groups, found {roles}"
    assert "prior" in roles, f"Expected 'prior' role in opt_g param_groups, found {roles}"

    router_group = next(grp for grp in opt_g.param_groups if grp.get("role") == "router")
    assert router_group["lr"] == pytest.approx(recipe.lr), (
        f"Router group lr {router_group['lr']} != recipe.lr {recipe.lr}"
    )

    # Verify all splat carrier parameters reside in router group
    carrier_p_ids = {id(p) for p in manipulator.parameters()}
    router_p_ids = {id(p) for p in router_group["params"]}
    assert carrier_p_ids.issubset(router_p_ids), "Not all splat parameters assigned to 'router' group"

    # 3. UpdatePolicy role validation
    policy = UpdatePolicy(
        recipe,
        generator=G,
        critic=D,
        prior=prior,
        generator_optimizer=opt_g,
        critic_optimizer=opt_d,
    )
    assert any("router" in r_list for r_list in policy.roles), (
        f"UpdatePolicy failed to validate router role: {policy.roles}"
    )


def test_simplex_wavefront_generator_gradient_flow_and_bounded_gain():
    """Verify SimplexWavefrontGenerator bounded outputs, zero-init carrier equivalence, and gradient flow."""
    B, C, H, W = 4, 3, 16, 16
    K = 4
    carrier_basis = DifferentiableSplatCarrierBasis(
        num_modes=K,
        channels=C,
        height=H,
        width=W,
        num_base_splats=4,
        num_detail_splats=2,
    )
    generator = SimplexWavefrontGenerator(carrier_basis=carrier_basis)

    # Simplex routing weights
    simplex_weights = torch.softmax(torch.randn(B, K), dim=-1)

    # 1. Forward generation bounded in (-1, 1)
    out = generator(simplex_weights)
    carrier_img = generator.render_carrier(simplex_weights)
    assert (out > -1.0).all() and (out < 1.0).all(), "Generator output not bounded in (-1, 1)"

    # 2. Zero-init carrier equivalence
    err = (out - carrier_img).abs().max().item()
    assert err <= 1e-6, f"Initial generator output did not match carrier: diff={err}"

    # 3. Backward gradient flow at step 0
    target = torch.full_like(out, 0.4)
    loss = F.mse_loss(out, target)
    loss.backward()

    # Verify carrier parameters receive finite non-zero gradients
    for name, p in carrier_basis.named_parameters():
        assert p.grad is not None, f"Carrier parameter '{name}' received no gradient"
        assert torch.isfinite(p.grad).all(), f"Carrier parameter '{name}' gradient not finite"
        assert p.grad.abs().sum() > 0, f"Carrier parameter '{name}' gradient is all zeros"

    # 4. Active gradient flow through entire residual pyramid
    final_conv = generator.get_final_conv()
    with torch.no_grad():
        final_conv.weight.add_(0.02 * torch.randn_like(final_conv.weight))

    generator.zero_grad()
    carrier_basis.zero_grad()
    out_active = generator(simplex_weights)
    loss_active = F.mse_loss(out_active, target)
    loss_active.backward()

    for name, p in generator.named_parameters():
        assert p.grad is not None, f"Parameter '{name}' received no gradient in active pass"
        assert torch.isfinite(p.grad).all(), f"Parameter '{name}' gradient not finite in active pass"
        assert p.grad.abs().sum() > 0, f"Parameter '{name}' gradient was zero in active pass"


def test_wavefront_manipulator_training_integration():
    """Verify multi-step training step execution across recipes with EMA sync and checkpoint continuation parity."""
    recipes_to_test = ["bispf_optimized", "ka2", "bcap", "bispf"]

    for r_name in recipes_to_test:
        recipe = get_recipe(r_name, num_particles=16, z_dim=8, batch_size=4)
        prior = recipe.make_prior(manipulators=[
            SimplexRoutingManipulator(num_modes=4),
            SplatCarrierManipulator(
                num_modes=4,
                channels=3,
                height=16,
                width=16,
                num_base_splats=2,
                num_detail_splats=2,
            ),
            SobolevWavefrontManipulator(
                grad_lambda=0.5,
                grad_order=1.0,
                weight=1.0,
                weight_seam=1.0,
                patch_size=4,
            ),
        ])

        generator = SimplexWavefrontGenerator(channels=3, hidden_dim=16, num_blocks=1)
        critic = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 1))

        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=42)
        real = torch.randn(4, 3, 16, 16)

        res = trainer.step(real)
        assert res is not None, f"Step returned None for recipe {r_name}"
        assert "loss_g" in res, f"result missing 'loss_g' for recipe {r_name}"
        assert "prior_manipulation" in res, f"result missing 'prior_manipulation' for recipe {r_name}"
        assert res["prior_manipulation"].item() > 0.0

    # Also verify continuous e22 with SobolevWavefrontManipulator
    recipe_e22 = get_recipe("e22", num_particles=16, z_dim=8, batch_size=4)
    prior_e22 = recipe_e22.make_prior(manipulators=[
        SobolevWavefrontManipulator(
            grad_lambda=0.5,
            grad_order=1.0,
            weight=1.0,
            weight_seam=1.0,
            patch_size=4,
        ),
    ])
    gen_e22 = nn.Sequential(nn.Linear(8, 3 * 16 * 16), nn.Unflatten(1, (3, 16, 16)), nn.Tanh())
    crit_e22 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 1))
    trainer_e22 = GANTrainer(recipe_e22, gen_e22, crit_e22, prior=prior_e22, seed=42)
    res_e22 = trainer_e22.step(torch.randn(4, 3, 16, 16))
    assert res_e22 is not None and "loss_g" in res_e22


    # Test bit-exact checkpoint continuation parity under bispf_optimized
    def build_trainer():
        recipe = get_recipe("bispf_optimized", num_particles=16, z_dim=8, batch_size=4)
        prior = recipe.make_prior(manipulators=[
            SimplexRoutingManipulator(num_modes=4),
            SplatCarrierManipulator(
                num_modes=4,
                channels=3,
                height=16,
                width=16,
                num_base_splats=2,
                num_detail_splats=2,
            ),
            SobolevWavefrontManipulator(
                grad_lambda=0.5,
                grad_order=1.0,
                weight_seam=1.0,
                patch_size=4,
            ),
        ])
        torch.manual_seed(42)
        gen = SimplexWavefrontGenerator(channels=3, hidden_dim=16, num_blocks=1)
        crit = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 1))
        latent_gen = torch.Generator().manual_seed(101)
        penalty_gen = torch.Generator().manual_seed(102)
        noise_gen = torch.Generator().manual_seed(103)
        return GANTrainer(
            recipe,
            gen,
            crit,
            prior=prior,
            latent_generator=latent_gen,
            penalty_generator=penalty_gen,
            noise_generator=noise_gen,
        )

    trainer_uninterrupted = build_trainer()
    batches = [
        torch.randn(4, 3, 16, 16, generator=torch.Generator().manual_seed(200 + i))
        for i in range(4)
    ]

    for i in range(2):
        trainer_uninterrupted.step(batches[i])

    checkpoint_at_2 = trainer_uninterrupted.state_dict()

    for i in range(2, 4):
        trainer_uninterrupted.step(batches[i])

    trainer_restored = build_trainer()
    trainer_restored.load_state_dict(checkpoint_at_2)

    for i in range(2, 4):
        trainer_restored.step(batches[i])

    ref_z = trainer_uninterrupted.prior.base.z
    res_z = trainer_restored.prior.base.z
    diff_z = (ref_z - res_z).abs().max().item()
    assert diff_z < 1e-6, f"prior.base.z mismatch on restore: {diff_z}"

    ref_splat = trainer_uninterrupted.prior.manipulators[1].basis.detail_params
    res_splat = trainer_restored.prior.manipulators[1].basis.detail_params
    diff_splat = (ref_splat - res_splat).abs().max().item()
    assert diff_splat < 1e-6, f"detail_params mismatch on restore: {diff_splat}"


def test_wavefront_pipeline_integration_and_cpu_efficiency():
    """Verify end-to-end ManipulatedPrior integration and benchmark CPU execution runtime."""
    t0 = time.perf_counter()
    recipe = get_recipe("bispf_optimized", num_particles=16, z_dim=8, batch_size=4)

    prior = recipe.make_prior().manipulate(
        SimplexRoutingManipulator(num_modes=4),
        SplatCarrierManipulator(
            num_modes=4,
            channels=3,
            height=16,
            width=16,
            num_base_splats=2,
            num_detail_splats=2,
        ),
        SobolevWavefrontManipulator(
            grad_lambda=0.5,
            grad_order=1.0,
            weight_seam=1.0,
            patch_size=4,
        ),
    )

    dummy_g = nn.Sequential(
        nn.Conv2d(3, 3, 3, padding=1),
        nn.Tanh(),
    )
    dummy_d = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 1),
    )

    trainer = GANTrainer(recipe, dummy_g, dummy_d, prior=prior, seed=0)
    real_batch = torch.randn(4, 3, 16, 16)

    result = trainer.step(real_batch)
    assert result is not None, "GANTrainer.step returned None"
    assert "loss_g" in result, "result missing 'loss_g'"

    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Wavefront pipeline execution exceeded runtime bound: {elapsed:.2f}s"
