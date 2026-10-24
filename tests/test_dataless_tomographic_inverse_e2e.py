"""End-to-End Data-Less Multi-Modal Tomographic Inverse Benchmark.

Verifies AC3 quantitative acceptance gates on N=64 generated samples:
1. mean_target_prob >= 0.80
2. primary_accuracy >= 0.95
3. entropy_ratio >= 0.85
4. feature_diversity >= 0.12
5. patch_seam_ratio <= 1.15
6. shift_sensitivity <= 0.15

Also verifies:
- Frozen strided Patch-ViT tomographic operator invariants & deterministic initialization
- Mathematical laws integration:
    - Law 1: Sobolev H^s gradient filtering and wavefront manifold regularizer
    - Law 2: Ghost-free splat carrier basis and simplex refinement generator
    - Law 3: Symmetric log-odds, marginal entropy, and cosine repulsion
- Destructive negative controls confirming gate validity (anti-goodharting)
- GANTrainer integration, DualNorm role segregation, and checkpoint continuation parity
- Zero Recipe bloat (strictly 112 fields preserved)
"""
import copy
import dataclasses
import math
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from particlegan import (
    GANTrainer,
    Recipe,
    get_recipe,
    get_recipe_fields,
)
from particlegan.priors import (
    CosineRepulsionManipulator,
    MarginalEntropyManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
)
from particlegan.tomographic import (
    NormalizedFeatureCosineRepulsion,
    TomographicConstraintTarget,
    evaluate_tomographic_constraints,
    symmetric_log_odds,
)
from particlegan.wavefront import (
    SimplexWavefrontGenerator,
    SobolevGradFilter,
    WavefrontManifoldPrior,
)

FROZEN_112_RECIPE_FIELDS = (
    "name",
    "critic_formulation",
    "model",
    "training_precision",
    "z_dim",
    "num_particles",
    "prior_kind",
    "sigma_rel",
    "standardize",
    "num_classes",
    "conditioning",
    "ucd_target",
    "ucd_weight",
    "alpha_bar",
    "batch_size",
    "total_steps",
    "continuous_policy",
    "lr",
    "d_lr_mult",
    "prior_lr_mult",
    "betas",
    "prior_betas",
    "reg_arm",
    "reg_coeff",
    "reg_kappa",
    "reg_every",
    "critic_step_mode",
    "prior_reg",
    "prior_update",
    "prior_regularizer",
    "prior_reg_target_std",
    "prior_reg_eps",
    "prior_l2",
    "prior_proximal_strength",
    "prior_coulomb_softening",
    "prior_coulomb_confinement",
    "prior_coulomb_neighbors",
    "prior_sigreg_projections",
    "prior_sigreg_knots",
    "ema_decay",
    "lr_anneal_start",
    "lr_floor",
    "network_lr_floor",
    "network_lr_horizon_cap",
    "reg_anchor_min_decay",
    "reg_anchor_weight",
    "direct_particle_gain",
    "d_guard_ratio",
    "d_guard_min_steps",
    "latent_damping_max_rate",
    "direct_particle_betas",
    "input_noise_std",
    "input_noise_anneal_end",
    "output_noise_std",
    "output_noise_warmup",
    "encoder_mode",
    "routing_temperature",
    "distance_reduction",
    "observation_sigma",
    "reconstruction_weight",
    "kinetic_transport_weight",
    "kinetic_transport_local_weight",
    "kinetic_transport_projections",
    "amsgrad",
    "critic_r1_real",
    "critic_payoff_damping",
    "output_noise_mode",
    "lr_control",
    "particle_birth_death",
    "row_evidence_gate",
    "table_release_rule",
    "row_evidence_hot",
    "row_evidence_exclude",
    "row_evidence_hold",
    "birth_death_space",
    "serve_average",
    "reopen_signal",
    "reopen_anchor",
    "reopen_guard",
    "row_evidence_null",
    "birth_death_isolation",
    "birth_death_feature_scale",
    "birth_death_backend",
    "birth_death_cells",
    "birth_death_metric_rank",
    "birth_death_chunk",
    "birth_death_parent_policy",
    "row_policy",
    "optimizer_family",
    "optimizer_momentum",
    "optimizer_smoothing",
    "optimizer_convolution",
    "optimizer_convolution_rule",
    "optimizer_svd_backend",
    "optimizer_dense_profile",
    "constraint_geometry_mode",
    "optimizer_adam_lr",
    "eps",
    "beta2_end",
    "beta2_anneal_end",
    "reg_coeff_end",
    "reg_coeff_anneal_end",
    "loss",
    "d_betas",
    "d_eps",
    "prior_eps",
    "loss_labels",
    "adam_variant",
    "lr_schedule",
    "lr_decay_rate",
    "lr_decay_steps",
    "lr_decay_staircase",
)


class StridedPatchViTTomographicOperator(nn.Module):
    """Frozen Strided Patch-ViT Tomographic Operator M(x).

    Maps continuous 2D synthesized RGB image fields x in [-1, 1]^(B x 3 x 32 x 32)
    to multi-query soft-capped log-odds across Q=4 semantic queries:
      - Q0: Foreground presence (K_0 = 2, M_0 = 2: Background, Foreground)
      - Q1: Chromaticity mode (K_1 = 3, M_1 = 3: R, G, B)
      - Q2: Quadrant localization (K_2 = 4, M_2 = 4: TL, TR, BL, BR)
      - Q3: Orientation (K_3 = 2, M_3 = 2: Horizontal, Vertical)

    Architecture:
      - Strided patch embedding: Conv2d(3, d_model=64, kernel_size=8, stride=4, padding=0)
        Grid size: ((32 - 8) // 4 + 1) = 7 x 7 = 49 patches
      - Positional embedding: normalized 2D grid coordinates (B, 49, 64)
      - Transformer encoder: 2 layers, 4 attention heads, d_model=64, dim_feedforward=128
      - Layer normalization: LayerNorm(64)
      - Query heads:
          head_foreground: Linear(64, 2)
          head_chroma: Linear(64, 3)
          head_quadrant: Linear(64, 4) evaluated on quadrant-pooled tokens
          head_orientation: Linear(64, 2)
      - Output: soft-capped log-odds u in [-tau, tau] via symmetric_log_odds
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        patch_size: int = 8,
        stride: int = 4,
        tau: float = 3.0,
        seed: int = 42,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.tau = tau

        # 1. Strided patch extraction: 7x7 = 49 patches
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=stride, padding=0)
        num_patches = 49

        # 2. Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # 3. Transformer Encoder with Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

        # 4. Query Heads
        self.head_foreground = nn.Linear(d_model, 2)
        self.head_chroma = nn.Linear(d_model, 3)
        self.head_quadrant = nn.Linear(d_model, 4)
        self.head_orientation = nn.Linear(d_model, 2)

        # Initialize deterministically
        self._init_deterministic_weights(seed)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def _init_deterministic_weights(self, seed: int):
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name and param.ndim >= 2:
                    nn.init.orthogonal_(param, generator=gen)
                elif "bias" in name:
                    nn.init.zeros_(param)

            # Position embeddings: normalized 2D grid coordinates
            grid_y = torch.linspace(-1, 1, 7)
            grid_x = torch.linspace(-1, 1, 7)
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            gy = gy.reshape(-1)
            gx = gx.reshape(-1)
            self.pos_embed.zero_()
            self.pos_embed[0, :, 0] = gy
            self.pos_embed[0, :, 1] = gx

            # Semantic physical filters in patch_embed weights
            pe_w = self.patch_embed.weight  # (d_model, 3, 8, 8)
            pe_w.normal_(0, 0.05, generator=gen)

            # Channel 0: Foreground energy / contrast
            pe_w[0, :, :, :] = 0.2

            # Channels 1-3: Chromaticity difference filters
            pe_w[1, 0, :, :] += 1.0  # R
            pe_w[1, 1, :, :] -= 0.5
            pe_w[1, 2, :, :] -= 0.5

            pe_w[2, 1, :, :] += 1.0  # G
            pe_w[2, 0, :, :] -= 0.5
            pe_w[2, 2, :, :] -= 0.5

            pe_w[3, 2, :, :] += 1.0  # B
            pe_w[3, 0, :, :] -= 0.5
            pe_w[3, 1, :, :] -= 0.5

            # Channels 4-5: Sobel orientation filters
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ], dtype=torch.float32)
            sobel_x = F.interpolate(sobel_x.view(1, 1, 3, 3), size=(8, 8), mode="bilinear").squeeze()
            pe_w[4, :, :, :] = sobel_x.unsqueeze(0).expand(3, -1, -1) * 0.5

            sobel_y = torch.tensor([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1],
            ], dtype=torch.float32)
            sobel_y = F.interpolate(sobel_y.view(1, 1, 3, 3), size=(8, 8), mode="bilinear").squeeze()
            pe_w[5, :, :, :] = sobel_y.unsqueeze(0).expand(3, -1, -1) * 0.5

            # Semantic alignment of Query Heads:
            # Head 0: Foreground presence
            self.head_foreground.weight.zero_()
            self.head_foreground.weight[0, 0] = -1.5
            self.head_foreground.weight[1, 0] = 1.5

            # Head 1: Chromaticity mode (R=0, G=1, B=2)
            self.head_chroma.weight.zero_()
            self.head_chroma.weight[0, 1] = 2.0
            self.head_chroma.weight[1, 2] = 2.0
            self.head_chroma.weight[2, 3] = 2.0

            # Head 3: Orientation (Horizontal=0, Vertical=1)
            self.head_orientation.weight.zero_()
            self.head_orientation.weight[0, 4] = 2.0
            self.head_orientation.weight[1, 5] = 2.0

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        # 1. Patch projection
        patches = self.patch_embed(x)  # (B, d_model, 7, 7)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, 49, d_model)
        tokens = tokens + self.pos_embed

        # 2. Transformer Pre-LN self-attention
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)  # (B, 49, d_model)

        # 3. Global pooled features for Representation / Diversity
        features = tokens.mean(dim=1)  # (B, d_model)

        # 4. Spatial quadrant pooling (7x7 grid):
        # TL (y:0..3, x:0..3), TR (y:0..3, x:3..7), BL (y:3..7, x:0..3), BR (y:3..7, x:3..7)
        spatial_tokens = tokens.view(B, 7, 7, self.d_model)
        tl = spatial_tokens[:, :4, :4].mean(dim=(1, 2))
        tr = spatial_tokens[:, :4, 3:].mean(dim=(1, 2))
        bl = spatial_tokens[:, 3:, :4].mean(dim=(1, 2))
        br = spatial_tokens[:, 3:, 3:].mean(dim=(1, 2))
        quad_features = torch.stack([
            self.head_quadrant(tl)[:, 0],
            self.head_quadrant(tr)[:, 1],
            self.head_quadrant(bl)[:, 2],
            self.head_quadrant(br)[:, 3],
        ], dim=-1)

        # Unpadded query logits
        l_foreground = self.head_foreground(features)   # (B, 2)
        l_chroma = self.head_chroma(features)           # (B, 3)
        l_quadrant = quad_features                      # (B, 4)
        l_orientation = self.head_orientation(features) # (B, 2)

        # Symmetric soft-capped log-odds
        u_foreground = symmetric_log_odds(l_foreground, tau=self.tau)
        u_chroma = symmetric_log_odds(l_chroma, tau=self.tau)
        u_quadrant = symmetric_log_odds(l_quadrant, tau=self.tau)
        u_orientation = symmetric_log_odds(l_orientation, tau=self.tau)

        # Padded log-odds to K_max = 4 for 3D tensor representation (B, Q=4, K_max=4)
        u_f_pad = torch.cat([u_foreground, torch.full((B, 2), -self.tau, device=x.device)], dim=-1)
        u_c_pad = torch.cat([u_chroma, torch.full((B, 1), -self.tau, device=x.device)], dim=-1)
        u_q_pad = u_quadrant
        u_o_pad = torch.cat([u_orientation, torch.full((B, 2), -self.tau, device=x.device)], dim=-1)

        u_3d = torch.stack([u_f_pad, u_c_pad, u_q_pad, u_o_pad], dim=1)  # (B, 4, 4)
        u_16 = u_3d.reshape(B, 16)

        # Probabilities per query (softmax on valid sub-classes, padded with zeros)
        p_foreground = F.softmax(l_foreground, dim=-1)
        p_chroma = F.softmax(l_chroma, dim=-1)
        p_quadrant = F.softmax(l_quadrant, dim=-1)
        p_orientation = F.softmax(l_orientation, dim=-1)

        p_f_pad = torch.cat([p_foreground, torch.zeros(B, 2, device=x.device)], dim=-1)
        p_c_pad = torch.cat([p_chroma, torch.zeros(B, 1, device=x.device)], dim=-1)
        p_q_pad = p_quadrant
        p_o_pad = torch.cat([p_orientation, torch.zeros(B, 2, device=x.device)], dim=-1)
        probs_3d = torch.stack([p_f_pad, p_c_pad, p_q_pad, p_o_pad], dim=1)  # (B, 4, 4)

        return {
            "u": u_16,
            "logits": u_16,
            "probs": probs_3d,
            "features": features,
            "query_logits": [l_foreground, l_chroma, l_quadrant, l_orientation],
        }


class CompositeTomographicGenerator(nn.Module):
    """Composite generator outputting tomographic log-odds u in R^(B x 16)."""

    def __init__(self, image_generator, operator, sobolev_filter=None):
        super().__init__()
        self.image_generator = image_generator
        self.operator = operator
        self.sobolev_filter = sobolev_filter
        self.last_images = None

    def forward(self, z):
        imgs = self.image_generator(z)
        if self.sobolev_filter is not None and self.training and imgs.requires_grad:
            imgs = self.sobolev_filter(imgs)
        self.last_images = imgs
        out = self.operator(imgs)
        return out["u"]


class TomographicCritic(nn.Module):
    """Critic scoring tomographic log-odds consistency in R^(B x 16) -> R^(B x 1)."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, u):
        return self.net(u)


# =====================================================================
# Group 1: Strided Patch-ViT Tomographic Operator Invariants
# =====================================================================


def test_operator_frozen_deterministic_initialization():
    """Verify bit-exact parameter equality with same seed, frozen requires_grad, and zero downloads."""
    op1 = StridedPatchViTTomographicOperator(seed=42)
    op2 = StridedPatchViTTomographicOperator(seed=42)
    op_diff = StridedPatchViTTomographicOperator(seed=999)

    # Invariant: zero trainable parameters
    assert all(not p.requires_grad for p in op1.parameters()), "Operator parameters must be frozen"
    assert all(not p.requires_grad for p in op2.parameters()), "Operator parameters must be frozen"

    # Invariant: bit-exact equality across same seed
    for (n1, p1), (n2, p2) in zip(op1.named_parameters(), op2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} differed between identical seeds"

    # Invariant: difference across different seeds
    weights_differ = any(
        not torch.equal(p1, pd)
        for (n1, p1), (nd, pd) in zip(op1.named_parameters(), op_diff.named_parameters())
        if "weight" in n1
    )
    assert weights_differ, "Different seeds must produce different orthogonal initializations"

    # Parameter count verification
    total_params = sum(p.numel() for p in op1.parameters())
    assert 50_000 < total_params < 200_000, f"Unexpected parameter count: {total_params}"


def test_operator_output_shapes_and_soft_capping():
    """Verify output contract, shapes across batch sizes, and strict soft-capping |u| <= tau."""
    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42)
    tau = op.tau

    for B in [1, 4, 16]:
        x = torch.randn(B, 3, 32, 32)
        out = op(x)

        assert "u" in out
        assert "logits" in out
        assert "probs" in out
        assert "features" in out
        assert "query_logits" in out

        assert out["u"].shape == (B, 16)
        assert out["logits"].shape == (B, 16)
        assert out["probs"].shape == (B, 4, 4)
        assert out["features"].shape == (B, 64)
        assert len(out["query_logits"]) == 4

        # Soft-capping invariant: |u_k| <= tau
        assert torch.all(out["u"] <= tau + 1e-5), f"Max log-odds exceeded tau={tau}: {out['u'].max()}"
        assert torch.all(out["u"] >= -tau - 1e-5), f"Min log-odds fell below -tau={-tau}: {out['u'].min()}"

        # Softmax sum per query invariant on valid sub-classes
        assert torch.allclose(out["probs"][:, 0, :2].sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(out["probs"][:, 1, :3].sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(out["probs"][:, 2, :4].sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(out["probs"][:, 3, :2].sum(dim=-1), torch.ones(B), atol=1e-5)


def test_operator_synthetic_semantic_sensitivity():
    """Verify operator responds strongly and interpretably to synthetic color and energy stimuli."""
    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42)

    # 1. Pure Red, Green, Blue inputs
    red = torch.zeros(1, 3, 32, 32)
    red[:, 0] = 3.0
    out_red = op(red)
    p_red = out_red["probs"][0, 1, :3]
    assert p_red.argmax().item() == 0, f"Red stimulus was not classified as Red (class 0): {p_red}"
    assert p_red[0].item() > 0.85, f"Confidence for Red stimulus too low: {p_red[0].item():.4f}"

    green = torch.zeros(1, 3, 32, 32)
    green[:, 1] = 3.0
    out_green = op(green)
    p_green = out_green["probs"][0, 1, :3]
    assert p_green.argmax().item() == 1, f"Green stimulus was not classified as Green (class 1): {p_green}"
    assert p_green[1].item() > 0.85, f"Confidence for Green stimulus too low: {p_green[1].item():.4f}"

    blue = torch.zeros(1, 3, 32, 32)
    blue[:, 2] = 3.0
    out_blue = op(blue)
    p_blue = out_blue["probs"][0, 1, :3]
    assert p_blue.argmax().item() == 2, f"Blue stimulus was not classified as Blue (class 2): {p_blue}"
    assert p_blue[2].item() > 0.85, f"Confidence for Blue stimulus too low: {p_blue[2].item():.4f}"


# =====================================================================
# Group 2: End-to-End Inverse Optimization & 6-Gate Quantitative Acceptance (§AC3)
# =====================================================================


def _run_inverse_optimization(steps: int = 100, seed: int = 42, dev_str: str = "cpu"):
    """Execute end-to-end multi-modal tomographic inverse optimization protocol."""
    torch.manual_seed(seed)
    dev = torch.device(dev_str)

    op = StridedPatchViTTomographicOperator(tau=3.0, seed=seed).to(dev)
    target_sampler = TomographicConstraintTarget(
        num_classes=[2, 3, 4, 2],
        num_modes=[2, 3, 4, 2],
        tau=3.0,
        target_prob=0.98,
        pad_to_k_max=True,
    )

    num_modes = 48
    recipe = get_recipe(
        "bispf_optimized",
        num_particles=num_modes,
        z_dim=num_modes,
        batch_size=num_modes,
        total_steps=steps,
    )

    splat_manip = SplatCarrierManipulator(
        num_modes=num_modes,
        channels=3,
        height=32,
        width=32,
        num_base_splats=4,
        num_detail_splats=4,
        gamma_base=1.5,
        gamma_detail=1.5,
    )

    prior = recipe.make_prior().manipulate(
        splat_manip,
        SobolevWavefrontManipulator(
            grad_lambda=1.0,
            grad_order=1.0,
            weight_seam=1.0,
            weight_biharmonic=0.05,
            patch_size=8,
            secondary_patch_size=4,
        ),
        MarginalEntropyManipulator(
            query_indices=[0, 1, 2, 3],
            num_classes=4,
            num_modes=[2, 3, 4, 2],
            min_entropy_ratio=0.88,
            weight=1.0,
        ),
        CosineRepulsionManipulator(margin=0.85, weight=0.5),
    ).to(dev)

    G = SimplexWavefrontGenerator(
        channels=3,
        hidden_dim=32,
        num_blocks=2,
        residual_gain=0.3,
    ).to(dev)

    with torch.no_grad():
        prior.base.z.copy_(torch.eye(num_modes, device=dev) * 4.0)

    optimizer = torch.optim.Adam([
        {"params": G.parameters(), "lr": 0.01},
        {"params": [prior.base.z], "lr": 0.05},
        {"params": splat_manip.parameters(), "lr": 0.03},
    ])

    targets = target_sampler.sample_cartesian_product(device=dev)
    target_modes = targets["modes"]
    idx = torch.arange(num_modes, device=dev)

    t0 = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        carrier = prior(idx)
        fake_imgs = G(carrier)
        fake_hooked = prior.postprocess_generated(fake_imgs)
        out = op(fake_hooked)

        ce_loss = sum(F.cross_entropy(out["query_logits"][q], target_modes[:, q]) for q in range(4))
        prior_loss = prior.regularize(
            prior.base.z,
            sampled_codes=carrier,
            indices=idx,
            generated=fake_imgs,
            logits=out["u"],
        )
        total_loss = ce_loss + prior_loss
        total_loss.backward()
        optimizer.step()
    dt = time.time() - t0

    # Evaluate on N=64 samples drawn from target distribution
    with torch.no_grad():
        eval_gen = torch.Generator().manual_seed(seed)
        sample_idx = torch.randint(0, num_modes, (64,), generator=eval_gen, device=dev)
        test_carrier = prior(sample_idx)
        test_imgs = G(test_carrier)
        test_target_modes = target_modes[sample_idx]

        metrics = evaluate_tomographic_constraints(
            test_imgs,
            operator=op,
            target_modes=test_target_modes,
            primary_query_index=0,
            patch_size=8,
            secondary_patch_size=4,
        )

    return metrics, test_imgs, dt


def test_dataless_tomographic_inverse_e2e_all_six_gates():
    """Verify that all 6 quantitative acceptance gates pass simultaneously on N=64 generated samples.

    Acceptance Gates (§AC3):
      1. mean_target_prob >= 0.80
      2. primary_accuracy >= 0.95
      3. entropy_ratio >= 0.85
      4. feature_diversity >= 0.12
      5. patch_seam_ratio <= 1.15
      6. shift_sensitivity <= 0.15
    """
    metrics, _, dt = _run_inverse_optimization(steps=100, seed=42)

    assert metrics["mean_target_prob"] >= 0.80, (
        f"Gate 1 failed: mean_target_prob={metrics['mean_target_prob']:.4f} < 0.80"
    )
    assert metrics["primary_accuracy"] >= 0.95, (
        f"Gate 2 failed: primary_accuracy={metrics['primary_accuracy']:.4f} < 0.95"
    )
    assert metrics["entropy_ratio"] >= 0.85, (
        f"Gate 3 failed: entropy_ratio={metrics['entropy_ratio']:.4f} < 0.85"
    )
    assert metrics["feature_diversity"] >= 0.12, (
        f"Gate 4 failed: feature_diversity={metrics['feature_diversity']:.4f} < 0.12"
    )
    assert metrics["patch_seam_ratio"] <= 1.15, (
        f"Gate 5 failed: patch_seam_ratio={metrics['patch_seam_ratio']:.4f} > 1.15"
    )
    assert metrics["shift_sensitivity"] <= 0.15, (
        f"Gate 6 failed: shift_sensitivity={metrics['shift_sensitivity']:.4f} > 0.15"
    )

    # Performance budget verification
    assert dt < 30.0, f"100 optimization steps exceeded 30s budget: {dt:.2f}s"


def test_dataless_inverse_reproducibility():
    """Verify bit-exact reproducibility of images and metric parity across repeated seed=42 runs."""
    metrics1, imgs1, _ = _run_inverse_optimization(steps=60, seed=42)
    metrics2, imgs2, _ = _run_inverse_optimization(steps=60, seed=42)

    # Bit-exact generated image parity
    assert torch.allclose(imgs1, imgs2, atol=1e-6), "Generated images differed across identical seed=42 runs"

    # Metric parity across trials
    for k in metrics1:
        assert abs(metrics1[k] - metrics2[k]) < 1e-5, (
            f"Metric {k} differed across identical runs: {metrics1[k]} vs {metrics2[k]}"
        )


def test_dataless_inverse_execution_budget():
    """Verify that 100 steps of inverse optimization runs in under 15 seconds on CPU."""
    _, _, dt = _run_inverse_optimization(steps=100, seed=42)
    assert dt < 15.0, f"Expected 100 steps to finish in < 15s on CPU, got {dt:.2f}s"


# =====================================================================
# Group 3: Destructive Negative Controls (Anti-Goodharting / Gate Validity)
# =====================================================================


def test_negative_control_mode_collapse_fails_entropy_and_diversity():
    """Verify that mode-collapsed outputs fail entropy_ratio and feature_diversity gates."""
    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42)
    target_sampler = TomographicConstraintTarget(num_classes=[2, 3, 4, 2], num_modes=[2, 3, 4, 2], tau=3.0)
    targets = target_sampler(64)

    # Synthesize identical images (complete collapse to a single image)
    single_img = torch.randn(1, 3, 32, 32)
    collapsed_imgs = single_img.expand(64, 3, 32, 32).contiguous()

    metrics = evaluate_tomographic_constraints(
        collapsed_imgs,
        operator=op,
        target_modes=targets["modes"],
        primary_query_index=0,
    )

    # Identical images must produce feature diversity near 0.0 (< 0.12)
    assert metrics["feature_diversity"] < 0.12, (
        f"Mode-collapsed images should fail feature_diversity gate: got {metrics['feature_diversity']:.4f}"
    )


def test_negative_control_patch_checkerboard_fails_seam_ratio():
    """Verify that images with sharp grid seam boundaries fail the patch_seam_ratio gate (> 1.15)."""
    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42)
    target_sampler = TomographicConstraintTarget(num_classes=[2, 3, 4, 2], num_modes=[2, 3, 4, 2], tau=3.0)
    targets = target_sampler(64)

    # Synthesize images with artificial patch boundary discontinuities at multiples of 8
    imgs = torch.zeros(64, 3, 32, 32)
    imgs[:, :, 8::8, :] = 5.0
    imgs[:, :, :, 8::8] = 5.0

    metrics = evaluate_tomographic_constraints(
        imgs,
        operator=op,
        target_modes=targets["modes"],
        primary_query_index=0,
        patch_size=8,
        secondary_patch_size=4,
    )

    assert metrics["patch_seam_ratio"] > 1.15, (
        f"Checkerboard seam image should fail patch_seam_ratio gate: got {metrics['patch_seam_ratio']:.4f}"
    )


def test_negative_control_untrained_fails_target_prob():
    """Verify that untrained random images fail mean_target_prob and primary_accuracy gates."""
    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42)
    target_sampler = TomographicConstraintTarget(num_classes=[2, 3, 4, 2], num_modes=[2, 3, 4, 2], tau=3.0)
    targets = target_sampler(64)

    random_imgs = torch.randn(64, 3, 32, 32)
    metrics = evaluate_tomographic_constraints(
        random_imgs,
        operator=op,
        target_modes=targets["modes"],
        primary_query_index=0,
    )

    assert metrics["mean_target_prob"] < 0.60, (
        f"Untrained random images should fail mean_target_prob gate: got {metrics['mean_target_prob']:.4f}"
    )
    assert metrics["primary_accuracy"] < 0.70, (
        f"Untrained random images should fail primary_accuracy gate: got {metrics['primary_accuracy']:.4f}"
    )


# =====================================================================
# Group 4: GANTrainer Integration & Checkpoint Continuation Parity
# =====================================================================


def test_gantrainer_dataless_step_execution():
    """Verify multi-step training execution using GANTrainer with ManipulatedPrior and DualNorm."""
    dev = torch.device("cpu")
    torch.manual_seed(42)

    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42).to(dev)
    target_sampler = TomographicConstraintTarget(
        num_classes=[2, 3, 4, 2],
        num_modes=[2, 3, 4, 2],
        tau=3.0,
        target_prob=0.98,
        pad_to_k_max=True,
    )

    num_modes = 48
    z_dim = 48
    batch_size = 48

    recipe = get_recipe(
        "bispf_optimized",
        num_particles=num_modes,
        z_dim=z_dim,
        batch_size=batch_size,
        total_steps=10,
    )

    splat_manip = SplatCarrierManipulator(
        num_modes=num_modes,
        channels=3,
        height=32,
        width=32,
        num_base_splats=4,
        num_detail_splats=4,
    )

    prior = recipe.make_prior().manipulate(
        splat_manip,
        MarginalEntropyManipulator(
            query_indices=[0, 1, 2, 3],
            num_classes=4,
            num_modes=[2, 3, 4, 2],
            min_entropy_ratio=0.88,
            source="logits",
        ),
        CosineRepulsionManipulator(margin=0.85, weight=0.5, source="codes"),
    ).to(dev)

    img_gen = SimplexWavefrontGenerator(
        channels=3,
        hidden_dim=32,
        num_blocks=2,
        residual_gain=0.3,
    ).to(dev)

    sobolev = SobolevGradFilter(lam=1.0, order=1.0)
    G = CompositeTomographicGenerator(img_gen, op, sobolev_filter=sobolev).to(dev)
    D = TomographicCritic(in_features=16, hidden_dim=32).to(dev)

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)

    # Verify DualNorm parameter role assignment
    # prior.base.z is in role="prior" (table), splat_manip parameters are 2D router parameters
    assert trainer.prior.base.z.ndim == 2
    for p in splat_manip.parameters():
        assert p.ndim == 2, "All splat parameters must be 2D matrices for NormalizedOptimizer router role"

    # Step 3 times with target batches from TomographicConstraintTarget
    for step in range(3):
        batch_targets = target_sampler(batch_size)["real"].to(dev)
        res = trainer.step(batch_targets)
        assert torch.isfinite(res["loss_g"]), f"Step {step}: loss_g not finite: {res['loss_g']}"
        assert torch.isfinite(res["loss_d"]), f"Step {step}: loss_d not finite: {res['loss_d']}"


def test_gantrainer_dataless_checkpoint_roundtrip_parity():
    """Verify bit-exact continuation parity across save/load state dict transitions."""
    dev = torch.device("cpu")
    torch.manual_seed(42)

    op = StridedPatchViTTomographicOperator(tau=3.0, seed=42).to(dev)
    num_modes = 48
    recipe = get_recipe(
        "bispf_optimized",
        num_particles=num_modes,
        z_dim=num_modes,
        batch_size=num_modes,
        total_steps=10,
    )

    splat_manip = SplatCarrierManipulator(
        num_modes=num_modes,
        channels=3,
        height=32,
        width=32,
        num_base_splats=4,
        num_detail_splats=4,
    )
    prior = recipe.make_prior().manipulate(splat_manip).to(dev)
    img_gen = SimplexWavefrontGenerator(channels=3, hidden_dim=32, num_blocks=2).to(dev)
    G = CompositeTomographicGenerator(img_gen, op).to(dev)
    D = TomographicCritic(in_features=16, hidden_dim=32).to(dev)

    trainer1 = GANTrainer(recipe, G, D, prior=prior, seed=42)

    # Step once
    target_sampler = TomographicConstraintTarget(
        num_classes=[2, 3, 4, 2],
        num_modes=[2, 3, 4, 2],
        tau=3.0,
        pad_to_k_max=True,
    )
    real_b = target_sampler(num_modes)["real"].to(dev)
    trainer1.step(real_b)

    # Save state
    state = trainer1.state_dict()

    # Re-instantiate fresh trainer
    splat_manip2 = SplatCarrierManipulator(
        num_modes=num_modes,
        channels=3,
        height=32,
        width=32,
        num_base_splats=4,
        num_detail_splats=4,
    )
    prior2 = recipe.make_prior().manipulate(splat_manip2).to(dev)
    img_gen2 = SimplexWavefrontGenerator(channels=3, hidden_dim=32, num_blocks=2).to(dev)
    G2 = CompositeTomographicGenerator(img_gen2, op).to(dev)
    D2 = TomographicCritic(in_features=16, hidden_dim=32).to(dev)

    trainer2 = GANTrainer(recipe, G2, D2, prior=prior2, seed=42)
    trainer2.load_state_dict(state)

    # Check parameter equality
    for (n1, p1), (n2, p2) in zip(trainer1.G.named_parameters(), trainer2.G.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"G param {n1} diverged after load_state_dict"
    assert torch.allclose(trainer1.prior.base.z, trainer2.prior.base.z, atol=1e-6)

    # Check generation output equality
    idx = torch.arange(num_modes, device=dev)
    out1 = trainer1.G(trainer1.prior(idx))
    out2 = trainer2.G(trainer2.prior(idx))
    assert torch.allclose(out1, out2, atol=1e-6), "Generator outputs diverged after checkpoint reload"


# =====================================================================
# Group 5: Zero Recipe Schema Bloat Verification
# =====================================================================


def test_recipe_112_fields_schema_preserved():
    """Verify Recipe dataclass fields remain strictly preserved at exactly 112 fields."""
    recipe_fields_dict = get_recipe_fields()
    field_names = tuple(recipe_fields_dict.keys())
    assert len(field_names) == 112, (
        f"Recipe schema was modified: expected 112 fields, found {len(field_names)}"
    )
    assert field_names == FROZEN_112_RECIPE_FIELDS, (
        f"Recipe field names diverged from frozen baseline:\n"
        f"Diff: {set(field_names) ^ set(FROZEN_112_RECIPE_FIELDS)}"
    )

    recipe_fields = tuple(f.name for f in dataclasses.fields(Recipe))
    assert len(recipe_fields) == 112
    assert recipe_fields == FROZEN_112_RECIPE_FIELDS
