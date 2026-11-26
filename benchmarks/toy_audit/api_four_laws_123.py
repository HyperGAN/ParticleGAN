"""Public-API Four Laws dynamical toy experiments for Laws 1, 2, and 3.

Five strictly public-API (``GANTrainer`` / ``Recipe``) dynamical falsification tasks
derived from the Four Laws audit (``reports/four-laws-audit/research-backlog-and-forge-proposals.md``):

1. ``fl_law1_aniso_unequal_modes`` (Law 1 / D-06, D-07):
   2D 8-mode mixture of 10:1 anisotropic rotated Gaussian ellipses with 4:1 unequal
   mode weights in a B_f < K streaming chart starvation regime.
2. ``fl_law1_noisy_atomic_codebook`` (Law 1 / E10, D-06):
   64-atom non-hypercube 2D hexagonal constellation with Gaussian jitter
   (sigma / Delta = 0.02), unequal Zipf weights (8:1 skew), and M != K (P=96 particles).
3. ``fl_law2_pure_adversarial_codim1`` (Law 2 intersect Law 4.5 / D-09, L2-6):
   2D codimension-1 thin circle (R = 1.5, sigma_perp = 0) trained via public GANTrainer
   with smooth SiLU generator and critic, tracking late-window normal-axis orbit oscillation
   and normal-direction critic gradients.
4. ``fl_law2_groupnorm_and_vit_sprites`` (Law 2 / L2-5, G4):
   32x32 4-class structured RGB geometric sprites with a compact ResNet+GroupNorm +
   Patch-4 multi-head attention (ViT-style SDPA) critic exposing fwAD vs RoR
   compatibility, gradient finiteness, and modern-backbone stability.
5. ``fl_law3_highd_orphan_d32`` (Law 3A / L3-7, D-08):
   Haar-rotated R^32 4-mode multi-scale unequal-mass mixture plus a distant 8% orphan
   cluster (dim = 32 > 16, exposing the hardcoded dim <= 16 Law-3A ceiling and screening
   dead zone).

All fixtures obey protocol seed 0, deterministic R2 prior initialization via
``particlegan.init.deterministic_orthogonal_``, isolated checkpointed RNG streams,
zero task-host training repairs, and fail-closed numerical + signed-margin scoring.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import (
    GANTrainer,
    Recipe,
    get_recipe,
    init,
    unwrap_prior,
)

from .reproducibility import DEFAULT_SEED, construction_rng

VERSION = "four-laws-123-v1"

TASKS: tuple[str, ...] = (
    "fl_law1_aniso_unequal_modes",
    "fl_law1_noisy_atomic_codebook",
    "fl_law2_pure_adversarial_codim1",
    "fl_law2_groupnorm_and_vit_sprites",
    "fl_law3_highd_orphan_d32",
)

V2_TASKS: tuple[str, ...] = (
    "fl_law1_noisy_atomic_codebook_v2",
    "fl_law1_highd_rank_d32_v2",
    "fl_law2_groupnorm_and_vit_sprites_v2",
    "fl_law3_highd_orphan_d32_v2",
)

ALL_TASKS: tuple[str, ...] = TASKS + V2_TASKS

# ---------------------------------------------------------------------------
# Frozen Task Specifications & Pre-Run Numerical Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, list[list[Any]]] = {
    "fl_law1_aniso_unequal_modes": [
        ["finite_fraction", "==", 1.0],
        ["mode_recall", ">=", 1.0],
        ["precision", ">=", 0.85],
        ["mass_tv", "<=", 0.14],
        ["min_cov_eig_ratio", ">=", 0.35],
        ["max_cov_eig_ratio", "<=", 2.20],
    ],
    "fl_law1_noisy_atomic_codebook": [
        ["finite_fraction", "==", 1.0],
        ["atom_recall", ">=", 0.96],
        ["mass_tv", "<=", 0.12],
        ["rms_nearest_dist", "<=", 0.08],
        ["outlier_rate", "<=", 0.05],
    ],
    "fl_law2_pure_adversarial_codim1": [
        ["finite_fraction", "==", 1.0],
        ["normal_orbit_r_tail", "<=", 0.06],
        ["normal_orbit_ptp", "<=", 0.12],
        ["manifold_mae", "<=", 0.06],
        ["manifold_coverage", ">=", 0.9375],
        ["angular_tv", "<=", 0.15],
    ],
    "fl_law2_groupnorm_and_vit_sprites": [
        ["finite_fraction", "==", 1.0],
        ["mode_coverage", ">=", 1.0],
        ["sprite_hq", ">=", 0.85],
        ["template_rmse", "<=", 0.12],
        ["mass_tv", "<=", 0.14],
    ],
    "fl_law3_highd_orphan_d32": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 5],
        ["core_eig_min", ">=", 0.25],
        ["orphan_recall", ">=", 0.50],
        ["precision", ">=", 0.85],
        ["mass_tv", "<=", 0.14],
    ],
}

SPECS: dict[str, dict[str, Any]] = {
    "fl_law1_aniso_unequal_modes": {
        "task_id": "fl_law1_aniso_unequal_modes",
        "case_id": "fl_law1_aniso_unequal_modes",
        "kind": "fl_law1_aniso_unequal_modes",
        "law": "law1",
        "defects": ["D-06", "D-07"],
        "title": "Law 1: 8-mode 10:1 anisotropic rotated ellipses with 4:1 mass skew (B_f < K)",
        "goal": (
            "Recover all 8 rotated 10:1 anisotropic Gaussian ellipses with 4:1 unequal mode "
            "weights under B_f < K streaming chart pressure, preserving per-mode covariance "
            "eigenvalue ratios and skewed mode masses."
        ),
        "target": (
            "2D 8-mode Gaussian mixture on R=3.0 circle with 10:1 rotated ellipse covariance "
            "(sigma_major=0.25, sigma_minor=0.025) and 4:1 unequal weights "
            "[0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625]."
        ),
        "resources": {"num_particles": 256, "z_dim": 4, "batch_size": 16},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "simple_mlp",
            "hidden": 128,
            "layers": 3,
            "out_dim": 2,
            "in_dim": 2,
            "fourier": 2,
        },
        "steps": 1500,
        "eval_samples": 4096,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law1_noisy_atomic_codebook": {
        "task_id": "fl_law1_noisy_atomic_codebook",
        "case_id": "fl_law1_noisy_atomic_codebook",
        "kind": "fl_law1_noisy_atomic_codebook",
        "law": "law1",
        "defects": ["E10", "D-06"],
        "title": "Law 1: 64-atom noisy Zipf hexagonal codebook with M != K (P=96 particles)",
        "goal": (
            "Capture all 64 non-hypercube hexagonal codebook atoms under sigma/Delta=0.02 "
            "Gaussian jitter, 8:1 Zipf mass skew, and M=96 != K=64 particles without "
            "barycentric ghosts or mass flattening."
        ),
        "target": (
            "2D 8x8 staggered hexagonal 64-atom constellation (spacing Delta=0.65) with "
            "Gaussian jitter sigma=0.02*Delta and permuted Zipf(s=0.50) weights (8:1 skew)."
        ),
        "resources": {"num_particles": 96, "z_dim": 4, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.015, "standardize": False},
        "model": {
            "family": "simple_mlp",
            "hidden": 128,
            "layers": 3,
            "out_dim": 2,
            "in_dim": 2,
            "fourier": 3,
        },
        "steps": 1600,
        "eval_samples": 4096,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law2_pure_adversarial_codim1": {
        "task_id": "fl_law2_pure_adversarial_codim1",
        "case_id": "fl_law2_pure_adversarial_codim1",
        "kind": "fl_law2_pure_adversarial_codim1",
        "law": "law2",
        "defects": ["D-09", "L2-6"],
        "title": "Law 2 intersect 4.5: Codimension-1 thin circle normal-orbit stability",
        "goal": (
            "Converge smoothly onto a zero-thickness 1D circle embedded in R^2 (R=1.5, "
            "sigma_perp=0) without late-window normal-direction limit cycles or arc collapse."
        ),
        "target": (
            "2D codimension-1 thin circle x = 1.5*(cos theta, sin theta), theta ~ Unif[0, 2pi), "
            "normal thickness sigma_perp = 0."
        ),
        "resources": {"num_particles": 128, "z_dim": 2, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "smooth_silu_mlp",
            "hidden": 64,
            "layers": 2,
            "out_dim": 2,
            "in_dim": 2,
        },
        "steps": 1200,
        "eval_samples": 2048,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law2_groupnorm_and_vit_sprites": {
        "task_id": "fl_law2_groupnorm_and_vit_sprites",
        "case_id": "fl_law2_groupnorm_and_vit_sprites",
        "kind": "fl_law2_groupnorm_and_vit_sprites",
        "law": "law2",
        "defects": ["L2-5", "G4"],
        "title": "Law 2: 32x32 4-class RGB sprites with ResNet+GroupNorm + Patch-4 ViT critic",
        "goal": (
            "Learn all 4 structured 32x32 RGB geometric sprite classes with crisp template "
            "fidelity and balanced mass using a compact ResNet+GroupNorm + Patch-4 ViT critic."
        ),
        "target": (
            "4-class uniform 32x32x3 RGB geometric sprites (red ring, green cross, blue frame, "
            "gold diamond) in [0, 1] with sigma=0.02 pixel jitter."
        ),
        "resources": {"num_particles": 64, "z_dim": 8, "batch_size": 32},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "resnet_gn_vit_sprites32",
            "image_size": 32,
            "channels": 3,
            "gen_channels": 32,
            "critic_channels": 16,
            "vit_dim": 32,
            "vit_heads": 4,
            "patch_size": 4,
        },
        "steps": 600,
        "eval_samples": 512,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "image",
    },
    "fl_law3_highd_orphan_d32": {
        "task_id": "fl_law3_highd_orphan_d32",
        "case_id": "fl_law3_highd_orphan_d32",
        "kind": "fl_law3_highd_orphan_d32",
        "law": "law3",
        "defects": ["L3-7", "D-08"],
        "title": "Law 3A: Haar-rotated R^32 multi-scale unequal-mass mixture + distant orphan",
        "goal": (
            "Recover all 4 multi-scale core clusters and a distant 8% orphan cluster in a "
            "Haar-rotated R^32 mixture (dim=32 > 16) with calibrated full-32D covariances "
            "and accurate unequal masses."
        ),
        "target": (
            "Haar-rotated R^32 5-mode Gaussian mixture: 4 core modes at (+-2.2, +-2.2) with "
            "widths [0.05, 0.09, 0.15, 0.22] and weights [0.36, 0.26, 0.18, 0.12] plus "
            "distant 8% orphan at (+7.5, 0.0) with width 0.08."
        ),
        "resources": {"num_particles": 256, "z_dim": 32, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "simple_mlp",
            "hidden": 192,
            "layers": 3,
            "out_dim": 32,
            "in_dim": 32,
            "fourier": 0,
        },
        "steps": 1200,
        "eval_samples": 6144,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
}

HOST_DEFINITIONS: dict[str, dict[str, Any]] = SPECS

# ---------------------------------------------------------------------------
# Calibrated v2 Specifications & Thresholds (Preserving Frozen v1 Untouched)
# ---------------------------------------------------------------------------

V2_THRESHOLDS: dict[str, list[list[Any]]] = {
    "fl_law1_noisy_atomic_codebook_v2": [
        ["finite_fraction", "==", 1.0],
        ["atom_recall", ">=", 0.96],
        ["mass_tv", "<=", 0.15],
        ["rms_nearest_dist", "<=", 0.08],
        ["outlier_rate", "<=", 0.10],
    ],
    "fl_law1_highd_rank_d32_v2": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 5],
        ["core_eig_min", ">=", 0.25],
        ["precision", ">=", 0.85],
        ["mass_tv", "<=", 0.14],
    ],
    "fl_law2_groupnorm_and_vit_sprites_v2": [
        ["finite_fraction", "==", 1.0],
        ["mode_coverage", ">=", 1.0],
        ["sprite_hq", ">=", 0.85],
        ["template_rmse", "<=", 0.12],
        ["mass_tv", "<=", 0.14],
    ],
    "fl_law3_highd_orphan_d32_v2": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 5],
        ["orphan_recall", ">=", 0.50],
        ["precision", ">=", 0.85],
        ["mass_tv", "<=", 0.14],
        ["canon_plane_eig_min", ">=", 0.25],
    ],
}

V2_SPECS: dict[str, dict[str, Any]] = {
    "fl_law1_noisy_atomic_codebook_v2": {
        "task_id": "fl_law1_noisy_atomic_codebook_v2",
        "case_id": "fl_law1_noisy_atomic_codebook_v2",
        "kind": "fl_law1_noisy_atomic_codebook_v2",
        "law": "law1",
        "defects": ["E10", "D-06", "Law1B-continuous-atom-ratio"],
        "title": "Law 1 (v2): 64-atom noisy Zipf hexagonal codebook with P=160 & matched prior sigma",
        "goal": (
            "Capture all 64 non-hypercube hexagonal codebook atoms under sigma/Delta=0.02 "
            "Gaussian jitter and 8:1 Zipf mass skew with P=160 particles and prior.sigma=0.010 <= target_sigma "
            "(atom_recall>=0.96, mass_tv<=0.15, rms_nearest_dist<=0.08, outlier_rate<=0.10)."
        ),
        "target": (
            "2D 8x8 staggered hexagonal 64-atom constellation (spacing Delta=0.65) with "
            "Gaussian jitter sigma=0.02*Delta (0.013) and permuted Zipf(s=0.50) weights (8:1 skew)."
        ),
        "resources": {"num_particles": 160, "z_dim": 4, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.010, "standardize": False},
        "model": {
            "family": "simple_mlp",
            "hidden": 128,
            "layers": 3,
            "out_dim": 2,
            "in_dim": 2,
            "fourier": 3,
        },
        "steps": 1600,
        "eval_samples": 4096,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law1_highd_rank_d32_v2": {
        "task_id": "fl_law1_highd_rank_d32_v2",
        "case_id": "fl_law1_highd_rank_d32_v2",
        "kind": "fl_law1_highd_rank_d32_v2",
        "law": "law1",
        "defects": ["L1-highd-rank", "P2-sketch-bias"],
        "title": "Law 1 (v2): Haar-rotated R^32 full-32D covariance rank with P=512 particles",
        "goal": (
            "Preserve full 32D intra-mode covariance rank (core_eig_min>=0.25) across all 5 Haar-rotated "
            "R^32 clusters with P=512 particles (>32 particles even on the 8% minority mode), without "
            "subspace collapse or sketch Jensen shrinkage."
        ),
        "target": (
            "Haar-rotated R^32 5-mode Gaussian mixture: 4 core modes at (+-2.2, +-2.2) with "
            "widths [0.05, 0.09, 0.15, 0.22] and weights [0.36, 0.26, 0.18, 0.12] plus "
            "distant 8% mode at (+7.5, 0.0) with width 0.08."
        ),
        "resources": {"num_particles": 512, "z_dim": 32, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "simple_mlp",
            "hidden": 192,
            "layers": 3,
            "out_dim": 32,
            "in_dim": 32,
            "fourier": 0,
        },
        "steps": 1400,
        "eval_samples": 6144,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law2_groupnorm_and_vit_sprites_v2": {
        "task_id": "fl_law2_groupnorm_and_vit_sprites_v2",
        "case_id": "fl_law2_groupnorm_and_vit_sprites_v2",
        "kind": "fl_law2_groupnorm_and_vit_sprites_v2",
        "law": "law2",
        "defects": ["L2-5", "G4"],
        "title": "Law 2 (v2): 32x32 4-class RGB sprites with ResNet+GroupNorm + Patch-4 ViT critic (1600 steps)",
        "goal": (
            "Learn all 4 structured 32x32 RGB geometric sprite classes with crisp template "
            "fidelity and balanced mass using a compact ResNet+GroupNorm + Patch-4 ViT critic "
            "over a calibrated 1600-step budget."
        ),
        "target": (
            "4-class uniform 32x32x3 RGB geometric sprites (red ring, green cross, blue frame, "
            "gold diamond) in [0, 1] with sigma=0.02 pixel jitter."
        ),
        "resources": {"num_particles": 64, "z_dim": 8, "batch_size": 32},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "resnet_gn_vit_sprites32",
            "image_size": 32,
            "channels": 3,
            "gen_channels": 32,
            "critic_channels": 16,
            "vit_dim": 32,
            "vit_heads": 4,
            "patch_size": 4,
        },
        "steps": 1600,
        "eval_samples": 512,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "image",
    },
    "fl_law3_highd_orphan_d32_v2": {
        "task_id": "fl_law3_highd_orphan_d32_v2",
        "case_id": "fl_law3_highd_orphan_d32_v2",
        "kind": "fl_law3_highd_orphan_d32_v2",
        "law": "law3",
        "defects": ["L3-7", "D-08"],
        "title": "Law 3A (v2): Haar-rotated R^32 multi-scale unequal-mass orphan transport",
        "goal": (
            "Isolate Law 3A high-D non-local orphan transport in Haar-rotated R^32 (dim=32 > 16): "
            "recover all 5 modes including the distant 8% orphan with accurate unequal masses, "
            "precision >= 0.85, and non-collapsed canonical-plane covariance."
        ),
        "target": (
            "Haar-rotated R^32 5-mode Gaussian mixture: 4 core modes at (+-2.2, +-2.2) with "
            "widths [0.05, 0.09, 0.15, 0.22] and weights [0.36, 0.26, 0.18, 0.12] plus "
            "distant 8% orphan at (+7.5, 0.0) with width 0.08."
        ),
        "resources": {"num_particles": 256, "z_dim": 32, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.05, "standardize": False},
        "model": {
            "family": "simple_mlp",
            "hidden": 192,
            "layers": 3,
            "out_dim": 32,
            "in_dim": 32,
            "fourier": 0,
        },
        "steps": 1200,
        "eval_samples": 6144,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
}

ALL_THRESHOLDS: dict[str, list[list[Any]]] = {**THRESHOLDS, **V2_THRESHOLDS}
ALL_SPECS: dict[str, dict[str, Any]] = {**SPECS, **V2_SPECS}

# ---------------------------------------------------------------------------
# Deterministic Geometry & Target Distributions
# ---------------------------------------------------------------------------

_ANISO_WEIGHTS = (0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625)
_ANISO_RADIUS = 3.0
_ANISO_SIGMA_MAJOR = 0.25
_ANISO_SIGMA_MINOR = 0.025

_CODEBOOK_GRID = 8
_CODEBOOK_NUM_ATOMS = _CODEBOOK_GRID * _CODEBOOK_GRID
_CODEBOOK_DELTA = 0.65
_CODEBOOK_JITTER_RATIO = 0.02
_CODEBOOK_ZIPF_S = 0.50

_CODIM1_RADIUS = 1.5
_CODIM1_ANGULAR_BINS = 16

_SPRITE_SIZE = 32
_SPRITE_CHANNELS = 3
_SPRITE_CLASSES = 4
_SPRITE_JITTER = 0.02
_SPRITE_HQ_RMSE_TOL = 0.16

_HIGHD_DIM = 32
_HIGHD_WEIGHTS = (0.36, 0.26, 0.18, 0.12, 0.08)
_HIGHD_SIGMAS = (0.05, 0.09, 0.15, 0.22, 0.08)
_HIGHD_CANON_CENTERS_2D = (
    (-2.2, -2.2),
    (-2.2, 2.2),
    (2.2, -2.2),
    (2.2, 2.2),
    (7.5, 0.0),
)
_HIGHD_CORE_Z_RADIUS = 8.0

_CACHED_HIGHD_Q: torch.Tensor | None = None
_CACHED_SPRITES: torch.Tensor | None = None
_CACHED_CODEBOOK: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def aniso_ellipse_parameters(
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(weights[8], means[8,2], covs[8,2,2], inv_sqrt_covs[8,2,2])``."""
    dev = torch.device(device)
    k_cnt = len(_ANISO_WEIGHTS)
    weights = torch.tensor(_ANISO_WEIGHTS, dtype=torch.float32, device=dev)
    means_list: list[list[float]] = []
    covs_list: list[torch.Tensor] = []
    inv_sqrt_list: list[torch.Tensor] = []
    for k in range(k_cnt):
        phi = 2.0 * math.pi * k / float(k_cnt)
        means_list.append([_ANISO_RADIUS * math.cos(phi), _ANISO_RADIUS * math.sin(phi)])
        theta = phi + 0.25 * math.pi + 0.35 * float(k)
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        diag_var = torch.diag(torch.tensor([_ANISO_SIGMA_MAJOR ** 2, _ANISO_SIGMA_MINOR ** 2], dtype=torch.float64))
        diag_inv_s = torch.diag(torch.tensor([1.0 / _ANISO_SIGMA_MAJOR, 1.0 / _ANISO_SIGMA_MINOR], dtype=torch.float64))
        covs_list.append((rot @ diag_var @ rot.T).to(dtype=torch.float32, device=dev))
        inv_sqrt_list.append((rot @ diag_inv_s @ rot.T).to(dtype=torch.float32, device=dev))
    means = torch.tensor(means_list, dtype=torch.float32, device=dev)
    covs = torch.stack(covs_list, dim=0)
    inv_sqrt = torch.stack(inv_sqrt_list, dim=0)
    return weights, means, covs, inv_sqrt


def noisy_codebook_parameters(
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Return ``(weights[64], centers[64,2], sigma, rank_order[64])`` for the 8x8 hex codebook."""
    global _CACHED_CODEBOOK
    if _CACHED_CODEBOOK is None:
        pts: list[list[float]] = []
        row_step = 0.5 * math.sqrt(3.0) * _CODEBOOK_DELTA
        for r in range(_CODEBOOK_GRID):
            offset = 0.5 * _CODEBOOK_DELTA if (r % 2 == 1) else 0.0
            for c in range(_CODEBOOK_GRID):
                x = (float(c) - 3.5) * _CODEBOOK_DELTA + offset
                y = (float(r) - 3.5) * row_step
                pts.append([x, y])
        centers = torch.tensor(pts, dtype=torch.float32)
        centers = centers - centers.mean(dim=0, keepdim=True)
        perm_gen = torch.Generator(device="cpu").manual_seed(12302)
        perm = torch.randperm(_CODEBOOK_NUM_ATOMS, generator=perm_gen)
        ranks = torch.arange(1, _CODEBOOK_NUM_ATOMS + 1, dtype=torch.float64)
        raw_w = ranks.pow(-_CODEBOOK_ZIPF_S)
        raw_w = raw_w / raw_w.sum()
        weights = torch.empty(_CODEBOOK_NUM_ATOMS, dtype=torch.float32)
        weights[perm] = raw_w.to(torch.float32)
        _CACHED_CODEBOOK = (weights, centers, perm)
    w_cpu, c_cpu, perm_cpu = _CACHED_CODEBOOK
    dev = torch.device(device)
    sigma = float(_CODEBOOK_JITTER_RATIO * _CODEBOOK_DELTA)
    return w_cpu.to(dev), c_cpu.to(dev), sigma, perm_cpu.to(dev)


def sprite_templates(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return deterministic 4-class ``(4, 3, 32, 32)`` RGB geometric sprites in ``[0, 1]``."""
    global _CACHED_SPRITES
    if _CACHED_SPRITES is None:
        h = _SPRITE_SIZE
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, h, dtype=torch.float32),
            indexing="ij",
        )
        rr = torch.sqrt(xx.square() + yy.square())
        imgs = torch.full((_SPRITE_CLASSES, _SPRITE_CHANNELS, h, h), 0.08, dtype=torch.float32)

        # Class 0: Red hollow ring
        ring = (rr >= 0.42) & (rr <= 0.68)
        imgs[0, 0, ring] = 0.94
        imgs[0, 1, ring] = 0.18
        imgs[0, 2, ring] = 0.18

        # Class 1: Green centered cross / plus
        cross = ((xx.abs() <= 0.16) & (yy.abs() <= 0.72)) | ((yy.abs() <= 0.16) & (xx.abs() <= 0.72))
        imgs[1, 0, cross] = 0.16
        imgs[1, 1, cross] = 0.92
        imgs[1, 2, cross] = 0.24

        # Class 2: Blue hollow square frame
        outer = (xx.abs() <= 0.68) & (yy.abs() <= 0.68)
        inner = (xx.abs() <= 0.36) & (yy.abs() <= 0.36)
        frame = outer & (~inner)
        imgs[2, 0, frame] = 0.18
        imgs[2, 1, frame] = 0.38
        imgs[2, 2, frame] = 0.95

        # Class 3: Gold/Yellow diagonal diamond + X accent
        l1 = xx.abs() + yy.abs()
        diamond = (l1 >= 0.42) & (l1 <= 0.78)
        imgs[3, 0, diamond] = 0.96
        imgs[3, 1, diamond] = 0.80
        imgs[3, 2, diamond] = 0.14

        _CACHED_SPRITES = imgs
    return _CACHED_SPRITES.to(device=torch.device(device))


def _haar_orthogonal_32() -> torch.Tensor:
    """Deterministic Haar-random orthogonal matrix Q in O(32) (QR with positive diag(R))."""
    global _CACHED_HIGHD_Q
    if _CACHED_HIGHD_Q is None:
        gen = torch.Generator(device="cpu").manual_seed(32001)
        mat = torch.randn((_HIGHD_DIM, _HIGHD_DIM), generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(mat)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        _CACHED_HIGHD_Q = (q * signs.unsqueeze(0)).to(torch.float32)
    return _CACHED_HIGHD_Q


def highd_orphan_parameters(
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(weights[5], means[5,32], sigmas[5], Q[32,32])`` for the Haar-rotated R^32 task."""
    dev = torch.device(device)
    q = _haar_orthogonal_32().to(dev)
    weights = torch.tensor(_HIGHD_WEIGHTS, dtype=torch.float32, device=dev)
    sigmas = torch.tensor(_HIGHD_SIGMAS, dtype=torch.float32, device=dev)
    canon = torch.zeros((len(_HIGHD_WEIGHTS), _HIGHD_DIM), dtype=torch.float32, device=dev)
    for idx, (cx, cy) in enumerate(_HIGHD_CANON_CENTERS_2D):
        canon[idx, 0] = float(cx)
        canon[idx, 1] = float(cy)
    means = canon @ q.T
    return weights, means, sigmas, q


def sample_target_for_kind(
    kind: str,
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Draw ``n`` i.i.d. target samples using only the supplied ``generator``."""
    count = int(n)
    dev = torch.device(device)
    g_dev = generator.device

    if kind == "fl_law1_aniso_unequal_modes":
        weights, means, covs, _ = aniso_ellipse_parameters("cpu")
        idx = torch.multinomial(weights, count, replacement=True, generator=generator).cpu()
        eps = torch.randn((count, 2), generator=generator, device=g_dev, dtype=torch.float32).cpu()
        chol = torch.linalg.cholesky(covs.double()).float()
        pts = means[idx] + torch.bmm(chol[idx], eps.unsqueeze(-1)).squeeze(-1)
        return pts.to(dev)

    if kind in ("fl_law1_noisy_atomic_codebook", "fl_law1_noisy_atomic_codebook_v2"):
        weights, centers, sigma, _ = noisy_codebook_parameters("cpu")
        idx = torch.multinomial(weights, count, replacement=True, generator=generator).cpu()
        eps = torch.randn((count, 2), generator=generator, device=g_dev, dtype=torch.float32).cpu()
        pts = centers[idx] + sigma * eps
        return pts.to(dev)

    if kind == "fl_law2_pure_adversarial_codim1":
        theta = (
            torch.rand((count,), generator=generator, device=g_dev, dtype=torch.float32).cpu()
            * (2.0 * math.pi)
        )
        pts = torch.stack((_CODIM1_RADIUS * torch.cos(theta), _CODIM1_RADIUS * torch.sin(theta)), dim=1)
        return pts.to(dev)

    if kind in ("fl_law2_groupnorm_and_vit_sprites", "fl_law2_groupnorm_and_vit_sprites_v2"):
        templates = sprite_templates("cpu")
        cls = torch.randint(0, _SPRITE_CLASSES, (count,), generator=generator, device=g_dev).cpu()
        noise = torch.randn(
            (count, _SPRITE_CHANNELS, _SPRITE_SIZE, _SPRITE_SIZE),
            generator=generator,
            device=g_dev,
            dtype=torch.float32,
        ).cpu()
        imgs = (templates[cls] + _SPRITE_JITTER * noise).clamp(0.0, 1.0)
        return imgs.to(dev)

    if kind in ("fl_law3_highd_orphan_d32", "fl_law3_highd_orphan_d32_v2", "fl_law1_highd_rank_d32_v2"):
        weights, means, sigmas, _ = highd_orphan_parameters("cpu")
        idx = torch.multinomial(weights, count, replacement=True, generator=generator).cpu()
        eps = torch.randn((count, _HIGHD_DIM), generator=generator, device=g_dev, dtype=torch.float32).cpu()
        pts = means[idx] + sigmas[idx].unsqueeze(1) * eps
        return pts.to(dev)

    raise ValueError(f"unsupported Four Laws 1-3 task kind: {kind!r}")


# ---------------------------------------------------------------------------
# Architectures (Strictly Public nn.Module; All Registered for deterministic_orthogonal_)
# ---------------------------------------------------------------------------


class SmoothSiLUGenerator(nn.Module):
    """Smooth C^infinity MLP generator for codimension-1 manifold stability probes."""

    def __init__(self, z_dim: int = 2, hidden_dim: int = 64, n_hidden: int = 2, out_dim: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        cur = int(z_dim)
        for _ in range(int(n_hidden)):
            blocks.append(nn.Linear(cur, int(hidden_dim)))
            blocks.append(nn.SiLU())
            cur = int(hidden_dim)
        blocks.append(nn.Linear(cur, int(out_dim)))
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SmoothSiLUCritic(nn.Module):
    """Smooth C^infinity MLP critic eliminating ReLU kink artifacts on codim-1 manifolds."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, n_hidden: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        cur = int(in_dim)
        for _ in range(int(n_hidden)):
            blocks.append(nn.Linear(cur, int(hidden_dim)))
            blocks.append(nn.SiLU())
            cur = int(hidden_dim)
        blocks.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class CompactSpriteGenerator(nn.Module):
    """Compact deterministic 32x32 RGB ConvTranspose + GroupNorm sprite generator."""

    def __init__(self, z_dim: int = 8, base_ch: int = 32, out_ch: int = 3) -> None:
        super().__init__()
        c = int(base_ch)
        c_mid = max(16, (c * 3) // 4)
        c_lo = max(12, c // 2)
        self.base_ch = c
        self.fc = nn.Linear(int(z_dim), c * 4 * 4)
        self.up1 = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(4, c)
        self.up2 = nn.ConvTranspose2d(c, c_mid, 4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(4, c_mid)
        self.up3 = nn.ConvTranspose2d(c_mid, c_lo, 4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(4, c_lo)
        self.head = nn.Conv2d(c_lo, int(out_ch), 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc(z)).reshape(-1, self.base_ch, 4, 4)
        h = F.silu(self.gn1(self.up1(h)))
        h = F.silu(self.gn2(self.up2(h)))
        h = F.silu(self.gn3(self.up3(h)))
        return torch.sigmoid(self.head(h))


class ResNetGroupNormViTCritic(nn.Module):
    """Compact 32x32 ResNet+GroupNorm stem followed by a Patch-4 ViT SDPA critic block."""

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        vit_dim: int = 32,
        vit_heads: int = 4,
    ) -> None:
        super().__init__()
        if vit_dim % vit_heads != 0:
            raise ValueError("vit_dim must be divisible by vit_heads")
        c = int(base_ch)
        d = int(vit_dim)
        self.vit_dim = d
        self.vit_heads = int(vit_heads)
        self.head_dim = d // self.vit_heads

        # Stage 1: ResNet + GroupNorm stem (32x32 -> 16x16 -> 8x8 Patch-4 grid)
        self.stem = nn.Conv2d(int(in_ch), c, 3, stride=2, padding=1)
        self.res_c1 = nn.Conv2d(c, c, 3, padding=1)
        self.res_gn1 = nn.GroupNorm(4, c)
        self.res_c2 = nn.Conv2d(c, c, 3, padding=1)
        self.res_gn2 = nn.GroupNorm(4, c)
        self.patch_down = nn.Conv2d(c, d, 3, stride=2, padding=1)
        self.patch_gn = nn.GroupNorm(4, d)

        # Deterministic sinusoidal 2D positional embedding for 8x8 = 64 Patch-4 tokens
        pos = self._sinusoidal_pos_2d(8, 8, d)
        self.register_buffer("pos_embed", pos, persistent=True)

        # Stage 2: Lightweight ViT Multi-Head Self-Attention block (smooth C^infinity
        # normalized sigmoid attention, natively compatible with both fwAD and RoR autodiff)
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    @staticmethod
    def _sinusoidal_pos_2d(gh: int, gw: int, dim: int) -> torch.Tensor:
        half = dim // 2
        quarter = max(1, half // 2)
        omega = torch.exp(
            -math.log(100.0) * torch.arange(quarter, dtype=torch.float32) / float(max(1, quarter))
        )
        ys = torch.linspace(-1.0, 1.0, gh, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, gw, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        py = gy.reshape(-1, 1) * omega.unsqueeze(0)
        px = gx.reshape(-1, 1) * omega.unsqueeze(0)
        emb = torch.cat((torch.sin(py), torch.cos(py), torch.sin(px), torch.cos(px)), dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb[:, :dim].unsqueeze(0) * 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.stem(x), 0.2)
        res = self.res_c2(F.leaky_relu(self.res_gn1(self.res_c1(h)), 0.2))
        h = F.leaky_relu(self.res_gn2(h + res), 0.2)
        h = F.leaky_relu(self.patch_gn(self.patch_down(h)), 0.2)  # (B, d, 8, 8)
        b, d, gh, gw = h.shape
        n_tok = gh * gw
        tok = h.flatten(2).transpose(1, 2) + self.pos_embed[:, :n_tok, :].to(dtype=h.dtype)
        q, k, v = self.qkv(self.ln1(tok)).chunk(3, dim=-1)
        q = q.reshape(b, n_tok, self.vit_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n_tok, self.vit_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n_tok, self.vit_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        weights = torch.sigmoid(scores)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).reshape(b, n_tok, d)
        tok = tok + self.proj(attn)
        tok = tok + self.mlp(self.ln2(tok))
        return self.head(self.ln_out(tok.mean(dim=1))).squeeze(-1)


def build_models_for_spec(spec: dict[str, Any], *, z_dim: int | None = None) -> tuple[nn.Module, nn.Module]:
    """Instantiate fresh ``(generator, discriminator)`` modules for a task specification."""
    model_cfg = spec["model"]
    family = model_cfg["family"]
    z_eff = int(spec["resources"]["z_dim"] if z_dim is None else z_dim)

    if family == "simple_mlp":
        hidden = int(model_cfg["hidden"])
        layers = int(model_cfg["layers"])
        out_dim = int(model_cfg["out_dim"])
        in_dim = int(model_cfg["in_dim"])
        fourier = int(model_cfg.get("fourier", 0))
        return (
            SimpleMLPGenerator(z_dim=z_eff, hidden_dim=hidden, n_hidden=layers, out_dim=out_dim),
            SimpleMLPDiscriminator(in_dim=in_dim, hidden_dim=hidden, n_hidden=layers, fourier=fourier),
        )
    if family == "smooth_silu_mlp":
        hidden = int(model_cfg["hidden"])
        layers = int(model_cfg["layers"])
        out_dim = int(model_cfg["out_dim"])
        in_dim = int(model_cfg["in_dim"])
        return (
            SmoothSiLUGenerator(z_dim=z_eff, hidden_dim=hidden, n_hidden=layers, out_dim=out_dim),
            SmoothSiLUCritic(in_dim=in_dim, hidden_dim=hidden, n_hidden=layers),
        )
    if family == "resnet_gn_vit_sprites32":
        return (
            CompactSpriteGenerator(
                z_dim=z_eff,
                base_ch=int(model_cfg["gen_channels"]),
                out_ch=int(model_cfg["channels"]),
            ),
            ResNetGroupNormViTCritic(
                in_ch=int(model_cfg["channels"]),
                base_ch=int(model_cfg["critic_channels"]),
                vit_dim=int(model_cfg["vit_dim"]),
                vit_heads=int(model_cfg["vit_heads"]),
            ),
        )
    raise ValueError(f"unsupported Four Laws 1-3 model family: {family!r}")


# ---------------------------------------------------------------------------
# Signed Normalized Margins & Threshold Evaluation
# ---------------------------------------------------------------------------


def signed_margins(metrics: dict[str, Any], task_id: str) -> dict[str, float]:
    """Compute signed normalized margin-to-threshold for every gate on ``task_id``.

    Positive/zero (``>= 0.0``) iff the gate passes; strictly negative (``< 0.0``) iff it
    fails. Missing or nonfinite values fail closed with ``-10.0`` (always finite for JSON).
    """
    if task_id not in ALL_THRESHOLDS:
        raise ValueError(f"unknown Four Laws task_id: {task_id!r}")
    out: dict[str, float] = {}
    for name, op, bound in ALL_THRESHOLDS[task_id]:
        b = float(bound)
        scale = max(abs(b), 1e-6) if b != 0.0 else 1.0
        raw = metrics.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            out[name] = -10.0
            continue
        v = float(raw)
        if op == ">=":
            margin = (v - b) / scale
        elif op == "<=":
            margin = (b - v) / scale
        elif op == "==":
            margin = 0.0 if v == b else -abs(v - b) / scale
        else:
            raise ValueError(f"unsupported threshold operator: {op!r}")
        out[name] = float(margin) if math.isfinite(margin) else -10.0
    return out


def _check_thresholds(metrics: dict[str, Any], task_id: str) -> list[str]:
    failed: list[str] = []
    for name, op, bound in ALL_THRESHOLDS[task_id]:
        raw = metrics.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            failed.append(f"{name} {op} {bound}")
            continue
        val = float(raw)
        b = float(bound)
        ok = (
            val <= b
            if op == "<="
            else val >= b
            if op == ">="
            else val == b
            if op == "=="
            else False
        )
        if not ok:
            failed.append(f"{name} {op} {bound}")
    return failed


def _finalize_score(
    task_id: str,
    metrics: dict[str, float | int],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_metrics: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            clean_metrics[key] = float(int(value))
        elif isinstance(value, int):
            clean_metrics[key] = int(value)
        else:
            fval = float(value)
            clean_metrics[key] = fval if math.isfinite(fval) else -99.0
    margins = signed_margins(clean_metrics, task_id)
    min_margin = float(min(margins.values())) if margins else -10.0
    clean_metrics["min_margin"] = min_margin
    failed = _check_thresholds(clean_metrics, task_id)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": clean_metrics,
        "margins": margins,
        "min_margin": min_margin,
        "views": views,
    }


def _default_reference(kind: str, n: int, seed: int) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    return sample_target_for_kind(kind, int(n), rng, device="cpu")


# ---------------------------------------------------------------------------
# Per-Task Scorers (Fail-Closed on NaN/Inf/Malformed Inputs + Goal Views)
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_fl_law1_aniso_unequal_modes(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score 8-mode 10:1 rotated anisotropic ellipses with 4:1 mass skew."""
    tid = "fl_law1_aniso_unequal_modes"
    weights, means, _, inv_sqrt = aniso_ellipse_parameters("cpu")
    ref = _default_reference(tid, 1024, 91001) if reference is None else reference.detach().cpu().to(torch.float32)

    valid = (
        isinstance(samples, torch.Tensor)
        and samples.ndim == 2
        and samples.shape[0] >= 8
        and samples.shape[1] == 2
    )
    if valid:
        pts = samples.detach().cpu().to(torch.float32)
        finite_fraction = float(torch.isfinite(pts).all(dim=-1).float().mean().item())
    else:
        pts = torch.zeros((2, 2), dtype=torch.float32)
        finite_fraction = 0.0

    if not valid or finite_fraction < 1.0:
        fallback_metrics: dict[str, float | int] = {
            "finite_fraction": finite_fraction,
            "mode_recall": 0.0,
            "modes": 0,
            "precision": 0.0,
            "mass_tv": 1.0,
            "min_cov_eig_ratio": 0.0,
            "max_cov_eig_ratio": 99.0,
            "mean_aspect_ratio": 0.0,
        }
        views = [
            {
                "kind": "scatter",
                "title": "8-mode 10:1 rotated ellipses (target vs API)",
                "target": ref[:512],
                "samples": torch.zeros((2, 2), dtype=torch.float32),
                "xlim": [-4.2, 4.2],
                "ylim": [-4.2, 4.2],
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Nonfinite or invalid-shape outputs fail closed.",
            },
            {
                "kind": "bar",
                "title": "4:1 unequal mode mass profile",
                "target": weights,
                "samples": torch.zeros_like(weights),
                "xlabel": "mode index",
                "ylabel": "mass",
                "caption": "Nonfinite outputs fail closed.",
            },
        ]
        return _finalize_score(tid, fallback_metrics, views)

    n = int(pts.shape[0])
    k_cnt = int(means.shape[0])
    assign = torch.cdist(pts, means).argmin(dim=1)
    counts = torch.bincount(assign, minlength=k_cnt).float() / float(n)
    mass_tv = float(0.5 * (counts - weights).abs().sum().item())

    diffs = pts - means[assign]
    whitened = torch.bmm(inv_sqrt[assign], diffs.unsqueeze(-1)).squeeze(-1)
    maha = whitened.norm(dim=1)
    in_core = maha <= 3.5
    precision = float(in_core.float().mean().item())

    core_shares = torch.bincount(assign[in_core], minlength=k_cnt).float() / float(n)
    recalled_mask = core_shares >= (0.40 * weights)
    modes = int(recalled_mask.sum().item())
    mode_recall = float(modes) / float(k_cnt)

    min_eigs: list[float] = []
    max_eigs: list[float] = []
    aspects: list[float] = []
    for k in range(k_cnt):
        w_k = whitened[(assign == k) & in_core]
        raw_k = pts[(assign == k) & in_core]
        if len(w_k) >= 6:
            cov_w = torch.cov(w_k.T, correction=0).double()
            ev_w = torch.linalg.eigvalsh(cov_w).clamp_min(0.0)
            min_eigs.append(float(ev_w[0].item()))
            max_eigs.append(float(ev_w[-1].item()))
            cov_raw = torch.cov(raw_k.T, correction=0).double()
            ev_raw = torch.linalg.eigvalsh(cov_raw).clamp_min(1e-12)
            aspects.append(float(math.sqrt(ev_raw[-1].item() / ev_raw[0].item())))
        else:
            min_eigs.append(0.0)
            max_eigs.append(99.0)
            aspects.append(0.0)

    metrics: dict[str, float | int] = {
        "finite_fraction": 1.0,
        "mode_recall": mode_recall,
        "modes": modes,
        "precision": precision,
        "mass_tv": mass_tv,
        "min_cov_eig_ratio": float(min(min_eigs)),
        "max_cov_eig_ratio": float(max(max_eigs)),
        "mean_aspect_ratio": float(sum(aspects) / len(aspects)),
    }
    m0 = means[0]
    views = [
        {
            "kind": "scatter",
            "title": "8-mode 10:1 rotated ellipses (target vs API)",
            "target": ref[:1024],
            "samples": pts[:1024].clamp(-4.15, 4.15),
            "xlim": [-4.2, 4.2],
            "ylim": [-4.2, 4.2],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": "Target has 10:1 rotated ellipses and 4:1 mass skew under B_f < K streaming charts.",
        },
        {
            "kind": "bar",
            "title": "4:1 unequal mode mass profile",
            "target": weights,
            "samples": counts,
            "xlabel": "mode index",
            "ylabel": "mass",
            "caption": "Target weights [0.25, 0.25, 0.125, 0.125, 0.0625x4].",
        },
        {
            "kind": "scatter",
            "title": "Mode 0 close-up (10:1 ellipse geometry)",
            "target": ref[:1024],
            "samples": pts[:1024],
            "xlim": [float(m0[0] - 0.9), float(m0[0] + 0.9)],
            "ylim": [float(m0[1] - 0.9), float(m0[1] + 0.9)],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": "Isotropic collapse fails whitened max/min covariance eigenvalue ratio bounds.",
        },
    ]
    return _finalize_score(tid, metrics, views)


@torch.no_grad()
def score_fl_law1_noisy_atomic_codebook(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score 64-atom noisy hexagonal Zipf codebook with M=96 particles."""
    tid = "fl_law1_noisy_atomic_codebook"
    weights, centers, _, perm = noisy_codebook_parameters("cpu")
    ref = _default_reference(tid, 1024, 91002) if reference is None else reference.detach().cpu().to(torch.float32)

    valid = (
        isinstance(samples, torch.Tensor)
        and samples.ndim == 2
        and samples.shape[0] >= 16
        and samples.shape[1] == 2
    )
    if valid:
        pts = samples.detach().cpu().to(torch.float32)
        finite_fraction = float(torch.isfinite(pts).all(dim=-1).float().mean().item())
    else:
        pts = torch.zeros((2, 2), dtype=torch.float32)
        finite_fraction = 0.0

    w_sorted = weights[perm]
    if not valid or finite_fraction < 1.0:
        fallback_metrics: dict[str, float | int] = {
            "finite_fraction": finite_fraction,
            "atom_recall": 0.0,
            "captured_atoms": 0,
            "mass_tv": 1.0,
            "rms_nearest_dist": 9.9,
            "outlier_rate": 1.0,
        }
        views = [
            {
                "kind": "scatter",
                "title": "64-atom noisy hexagonal codebook (target vs API)",
                "target": ref[:1024],
                "samples": torch.zeros((2, 2), dtype=torch.float32),
                "xlim": [-3.2, 3.2],
                "ylim": [-3.0, 3.0],
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Nonfinite or invalid-shape outputs fail closed.",
            },
            {
                "kind": "bar",
                "title": "64-atom Zipf mass profile (rank-ordered)",
                "target": w_sorted,
                "samples": torch.zeros_like(w_sorted),
                "xlabel": "Zipf rank",
                "ylabel": "mass",
                "caption": "Nonfinite outputs fail closed.",
            },
        ]
        return _finalize_score(tid, fallback_metrics, views)

    n = int(pts.shape[0])
    dists = torch.cdist(pts, centers)
    min_dist, assign = dists.min(dim=1)
    norm_dist = min_dist / _CODEBOOK_DELTA

    rms_nearest_dist = float(norm_dist.square().mean().sqrt().item())
    outlier_mask = norm_dist > 0.20
    outlier_rate = float(outlier_mask.float().mean().item())

    counts = torch.bincount(assign, minlength=_CODEBOOK_NUM_ATOMS).float() / float(n)
    mass_tv = float(0.5 * (counts - weights).abs().sum().item())

    in_atom = norm_dist <= 0.25
    in_counts = torch.bincount(assign[in_atom], minlength=_CODEBOOK_NUM_ATOMS).float()
    min_req = (0.25 * weights * float(n)).clamp_min(2.0)
    captured_atoms = int((in_counts >= min_req).sum().item())
    atom_recall = float(captured_atoms) / float(_CODEBOOK_NUM_ATOMS)

    metrics: dict[str, float | int] = {
        "finite_fraction": 1.0,
        "atom_recall": atom_recall,
        "captured_atoms": captured_atoms,
        "mass_tv": mass_tv,
        "rms_nearest_dist": rms_nearest_dist,
        "outlier_rate": outlier_rate,
    }
    views = [
        {
            "kind": "scatter",
            "title": "64-atom noisy hexagonal codebook (target vs API)",
            "target": ref[:1024],
            "samples": pts[:1024].clamp(-3.15, 3.15),
            "xlim": [-3.2, 3.2],
            "ylim": [-3.0, 3.0],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": "Non-hypercube 8x8 hex lattice with sigma/Delta=0.02 jitter, Zipf skew, and P=96 != 64.",
        },
        {
            "kind": "bar",
            "title": "64-atom Zipf mass profile (rank-ordered)",
            "target": w_sorted,
            "samples": counts[perm],
            "xlabel": "Zipf rank",
            "ylabel": "mass",
            "caption": "Tests mass preservation under unequal 8:1 Zipf weights and M=96 particles.",
        },
    ]
    return _finalize_score(tid, metrics, views)


@torch.no_grad()
def score_fl_law2_pure_adversarial_codim1(
    samples: torch.Tensor,
    *,
    orbit_trace: list[list[float]] | list[tuple[float, float]] | None = None,
    normal_grad_norm: float = 0.25,
    normal_grad_max: float = 0.50,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score 2D codimension-1 thin circle manifold fidelity and normal-direction orbit stability."""
    tid = "fl_law2_pure_adversarial_codim1"
    ref = _default_reference(tid, 1024, 91003) if reference is None else reference.detach().cpu().to(torch.float32)

    valid = (
        isinstance(samples, torch.Tensor)
        and samples.ndim == 2
        and samples.shape[0] >= 16
        and samples.shape[1] == 2
    )
    if valid:
        pts = samples.detach().cpu().to(torch.float32)
        finite_pts = float(torch.isfinite(pts).all(dim=-1).float().mean().item())
    else:
        pts = torch.zeros((2, 2), dtype=torch.float32)
        finite_pts = 0.0

    grad_ok = (
        isinstance(normal_grad_norm, (int, float))
        and not isinstance(normal_grad_norm, bool)
        and math.isfinite(float(normal_grad_norm))
        and isinstance(normal_grad_max, (int, float))
        and not isinstance(normal_grad_max, bool)
        and math.isfinite(float(normal_grad_max))
    )
    finite_fraction = finite_pts

    unif_bins = torch.full((_CODIM1_ANGULAR_BINS,), 1.0 / float(_CODIM1_ANGULAR_BINS), dtype=torch.float32)
    if not valid or finite_fraction < 1.0:
        fallback_metrics: dict[str, float | int] = {
            "finite_fraction": finite_fraction,
            "normal_orbit_r_tail": 9.9,
            "normal_orbit_ptp": 9.9,
            "manifold_mae": 9.9,
            "normal_grad_norm": float(normal_grad_norm) if grad_ok else 0.0,
            "normal_grad_max": float(normal_grad_max) if grad_ok else 0.0,
            "manifold_coverage": 0.0,
            "angular_tv": 1.0,
        }
        flat_line = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        views = [
            {
                "kind": "scatter",
                "title": "Codim-1 thin circle (R=1.5, sigma_perp=0)",
                "target": ref[:512],
                "samples": torch.zeros((2, 2), dtype=torch.float32),
                "xlim": [-2.5, 2.5],
                "ylim": [-2.5, 2.5],
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Nonfinite outputs or critic gradients fail closed.",
            },
            {
                "kind": "line",
                "title": "Normal-axis mean orbit trace E[||x|| - R]",
                "target": flat_line,
                "samples": flat_line,
                "xlabel": "update",
                "ylabel": "mean normal offset",
                "caption": "Nonfinite outputs fail closed.",
            },
            {
                "kind": "bar",
                "title": "16-sector angular manifold coverage",
                "target": unif_bins,
                "samples": torch.zeros_like(unif_bins),
                "xlabel": "angle bin",
                "ylabel": "mass",
                "caption": "Nonfinite outputs fail closed.",
            },
        ]
        return _finalize_score(tid, fallback_metrics, views)

    radii = pts.norm(dim=1)
    signed_err = radii - _CODIM1_RADIUS
    manifold_mae = float(signed_err.abs().mean().item())

    # Angular sector coverage along S^1
    angles = torch.atan2(pts[:, 1], pts[:, 0])
    angles = torch.where(angles < 0.0, angles + 2.0 * math.pi, angles)
    bin_idx = torch.floor(angles / (2.0 * math.pi) * float(_CODIM1_ANGULAR_BINS)).to(torch.long)
    bin_idx = bin_idx.clamp(0, _CODIM1_ANGULAR_BINS - 1)
    near_manifold = signed_err.abs() <= 0.25
    all_hist = torch.bincount(bin_idx, minlength=_CODIM1_ANGULAR_BINS).float() / float(len(pts))
    near_hist = torch.bincount(bin_idx[near_manifold], minlength=_CODIM1_ANGULAR_BINS).float() / float(len(pts))
    covered = int((near_hist >= (0.5 / float(_CODIM1_ANGULAR_BINS))).sum().item())
    manifold_coverage = float(covered) / float(_CODIM1_ANGULAR_BINS)
    angular_tv = float(0.5 * (all_hist - unif_bins).abs().sum().item())

    # Late-window normal-direction orbit amplitude (r_tail & ptp)
    if orbit_trace is not None and len(orbit_trace) > 0:
        trace_arr = torch.tensor([[float(s), float(v)] for s, v in orbit_trace], dtype=torch.float32)
        if not bool(torch.isfinite(trace_arr).all()):
            normal_orbit_r_tail = 9.9
            normal_orbit_ptp = 9.9
            trace_plot = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        else:
            tail_n = max(1, min(len(trace_arr), max(8, len(trace_arr) // 3)))
            tail_vals = trace_arr[-tail_n:, 1]
            normal_orbit_r_tail = float(tail_vals.square().mean().sqrt().item())
            normal_orbit_ptp = float((tail_vals.max() - tail_vals.min()).item())
            trace_plot = trace_arr
    else:
        cur_mean = float(signed_err.mean().item())
        normal_orbit_r_tail = abs(cur_mean)
        normal_orbit_ptp = 0.0
        trace_plot = torch.tensor([[0.0, cur_mean], [1.0, cur_mean]], dtype=torch.float32)

    if len(trace_plot) == 1:
        s0, v0 = float(trace_plot[0, 0].item()), float(trace_plot[0, 1].item())
        trace_plot = torch.tensor([[s0, v0], [s0 + 1.0, v0]], dtype=torch.float32)
    target_line = torch.tensor(
        [[float(trace_plot[0, 0].item()), 0.0], [float(trace_plot[-1, 0].item()), 0.0]],
        dtype=torch.float32,
    )

    metrics: dict[str, float | int] = {
        "finite_fraction": 1.0,
        "normal_orbit_r_tail": normal_orbit_r_tail,
        "normal_orbit_ptp": normal_orbit_ptp,
        "manifold_mae": manifold_mae,
        "normal_grad_norm": float(normal_grad_norm) if grad_ok else 0.0,
        "normal_grad_max": float(normal_grad_max) if grad_ok else 0.0,
        "manifold_coverage": manifold_coverage,
        "angular_tv": angular_tv,
    }
    views = [
        {
            "kind": "scatter",
            "title": "Codim-1 thin circle (R=1.5, sigma_perp=0)",
            "target": ref[:1024],
            "samples": pts[:1024].clamp(-2.45, 2.45),
            "xlim": [-2.5, 2.5],
            "ylim": [-2.5, 2.5],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": "Zero-thickness 1D circle exposing normal-direction rotational limit cycles.",
        },
        {
            "kind": "line",
            "title": "Normal-axis mean orbit trace E[||x|| - R]",
            "target": target_line,
            "samples": trace_plot,
            "xlabel": "update",
            "ylabel": "mean normal offset",
            "caption": "Oscillation around 0 fails late-window r_tail and peak-to-peak bounds.",
        },
        {
            "kind": "bar",
            "title": "16-sector angular manifold coverage",
            "target": unif_bins,
            "samples": all_hist,
            "xlabel": "angle bin",
            "ylabel": "mass",
            "caption": "Uniform coverage required around full circle without arc collapse.",
        },
    ]
    return _finalize_score(tid, metrics, views)


@torch.no_grad()
def score_fl_law2_groupnorm_and_vit_sprites(
    samples: torch.Tensor,
    *,
    critic_grad_finite: float = 1.0,
    critic_grad_med: float = 0.25,
    critic_grad_max: float = 0.50,
    mean_step_ms: float = 20.0,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score 32x32 4-class RGB geometric sprites trained with ResNet+GroupNorm+ViT critic."""
    tid = "fl_law2_groupnorm_and_vit_sprites"
    templates = sprite_templates("cpu")
    ref_grid = templates if reference is None else reference.detach().cpu().to(torch.float32)[:8]
    if ref_grid.ndim != 4 or ref_grid.shape[1:] != (_SPRITE_CHANNELS, _SPRITE_SIZE, _SPRITE_SIZE):
        ref_grid = templates

    valid = (
        isinstance(samples, torch.Tensor)
        and samples.ndim == 4
        and samples.shape[0] >= 4
        and samples.shape[1:] == (_SPRITE_CHANNELS, _SPRITE_SIZE, _SPRITE_SIZE)
    )
    if valid:
        imgs = samples.detach().cpu().to(torch.float32)
        finite_imgs = float(torch.isfinite(imgs).flatten(1).all(dim=1).float().mean().item())
    else:
        imgs = torch.zeros((4, _SPRITE_CHANNELS, _SPRITE_SIZE, _SPRITE_SIZE), dtype=torch.float32)
        finite_imgs = 0.0

    step_ok = (
        isinstance(mean_step_ms, (int, float))
        and not isinstance(mean_step_ms, bool)
        and math.isfinite(float(mean_step_ms))
        and float(mean_step_ms) >= 0.0
    )
    grad_fin_ok = (
        isinstance(critic_grad_finite, (int, float))
        and not isinstance(critic_grad_finite, bool)
        and math.isfinite(float(critic_grad_finite))
    )
    unif = torch.full((_SPRITE_CLASSES,), 1.0 / float(_SPRITE_CLASSES), dtype=torch.float32)

    if not valid or finite_imgs < 1.0:
        fallback_metrics: dict[str, float | int] = {
            "finite_fraction": finite_imgs if valid else 0.0,
            "critic_grad_finite": float(critic_grad_finite) if grad_fin_ok else 0.0,
            "critic_grad_med": 0.0,
            "critic_grad_max": 0.0,
            "mode_coverage": 0.0,
            "modes": 0,
            "sprite_hq": 0.0,
            "template_rmse": 1.0,
            "mass_tv": 1.0,
            "mean_step_ms": float(mean_step_ms) if step_ok else 0.0,
        }
        views = [
            {
                "kind": "image",
                "title": "32x32 RGB geometric sprites (target vs API)",
                "target": templates,
                "samples": torch.zeros_like(templates),
                "vmin": 0.0,
                "vmax": 1.0,
                "row_labels": ["Target sprites", "API samples"],
                "caption": "Nonfinite outputs or invalid shapes fail closed.",
            },
            {
                "kind": "bar",
                "title": "4-class sprite mass distribution",
                "target": unif,
                "samples": torch.zeros_like(unif),
                "xlabel": "sprite class",
                "ylabel": "mass",
                "caption": "Nonfinite outputs fail closed.",
            },
        ]
        return _finalize_score(tid, fallback_metrics, views)

    n = int(imgs.shape[0])
    flat_imgs = imgs.reshape(n, -1)
    flat_tpl = templates.reshape(_SPRITE_CLASSES, -1)
    dim_pix = float(flat_tpl.shape[1])
    rmses = torch.cdist(flat_imgs, flat_tpl) / math.sqrt(dim_pix)
    best_rmse, assign = rmses.min(dim=1)

    hq_mask = best_rmse <= _SPRITE_HQ_RMSE_TOL
    sprite_hq = float(hq_mask.float().mean().item())
    template_rmse = float(best_rmse.mean().item())

    all_counts = torch.bincount(assign, minlength=_SPRITE_CLASSES).float() / float(n)
    hq_counts = torch.bincount(assign[hq_mask], minlength=_SPRITE_CLASSES).float() / float(n)
    mass_tv = float(0.5 * (all_counts - unif).abs().sum().item())
    modes = int((hq_counts >= 0.10).sum().item())
    mode_coverage = float(modes) / float(_SPRITE_CLASSES)

    # Display up to 8 samples (ordered so representative samples are clear)
    disp_samples = imgs[:8].clamp(0.0, 1.0)
    disp_target = templates.repeat(2, 1, 1, 1)[: len(disp_samples)]

    metrics: dict[str, float | int] = {
        "finite_fraction": 1.0,
        "critic_grad_finite": float(critic_grad_finite) if grad_fin_ok else 0.0,
        "critic_grad_med": float(critic_grad_med) if math.isfinite(float(critic_grad_med)) else 0.0,
        "critic_grad_max": float(critic_grad_max) if math.isfinite(float(critic_grad_max)) else 0.0,
        "mode_coverage": mode_coverage,
        "modes": modes,
        "sprite_hq": sprite_hq,
        "template_rmse": template_rmse,
        "mass_tv": mass_tv,
        "mean_step_ms": float(mean_step_ms) if step_ok else 0.0,
    }
    views = [
        {
            "kind": "image",
            "title": "32x32 RGB geometric sprites (target vs API)",
            "target": disp_target,
            "samples": disp_samples,
            "vmin": 0.0,
            "vmax": 1.0,
            "row_labels": ["Target sprites", "API samples"],
            "caption": "ResNet+GroupNorm + Patch-4 ViT SDPA critic stability across 4 sprite classes.",
        },
        {
            "kind": "bar",
            "title": "4-class sprite mass distribution",
            "target": unif,
            "samples": all_counts,
            "xlabel": "sprite class",
            "ylabel": "mass",
            "caption": "Classes: 0=red ring, 1=green cross, 2=blue frame, 3=gold diamond.",
        },
    ]
    return _finalize_score(tid, metrics, views)


@torch.no_grad()
def score_fl_law3_highd_orphan_d32(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score Haar-rotated R^32 multi-scale unequal-mass mixture + distant 8% orphan."""
    tid = "fl_law3_highd_orphan_d32"
    weights, means, sigmas, q = highd_orphan_parameters("cpu")
    proj2 = q[:, :2]  # (32, 2) canonical plane projection
    ref = _default_reference(tid, 1024, 91005) if reference is None else reference.detach().cpu().to(torch.float32)
    ref_2d = ref[:1024] @ proj2

    valid = (
        isinstance(samples, torch.Tensor)
        and samples.ndim == 2
        and samples.shape[0] >= 32
        and samples.shape[1] == _HIGHD_DIM
    )
    if valid:
        pts = samples.detach().cpu().to(torch.float32)
        finite_fraction = float(torch.isfinite(pts).all(dim=-1).float().mean().item())
    else:
        pts = torch.zeros((2, _HIGHD_DIM), dtype=torch.float32)
        finite_fraction = 0.0

    if not valid or finite_fraction < 1.0:
        fallback_metrics: dict[str, float | int] = {
            "finite_fraction": finite_fraction,
            "modes": 0,
            "core_eig_min": 0.0,
            "core_eig_max": 99.0,
            "orphan_recall": 0.0,
            "precision": 0.0,
            "mass_tv": 1.0,
        }
        views = [
            {
                "kind": "scatter",
                "title": "R^32 canonical 2D projection (core + distant orphan)",
                "target": ref_2d,
                "samples": torch.zeros((2, 2), dtype=torch.float32),
                "xlim": [-4.5, 9.5],
                "ylim": [-4.5, 4.5],
                "aspect": "equal",
                "xlabel": "q0 (canonical axis 0)",
                "ylabel": "q1 (canonical axis 1)",
                "caption": "Nonfinite or invalid-shape outputs fail closed.",
            },
            {
                "kind": "bar",
                "title": "5-mode unequal mass + 8% orphan profile",
                "target": weights,
                "samples": torch.zeros_like(weights),
                "xlabel": "mode index (4 = orphan)",
                "ylabel": "mass",
                "caption": "Nonfinite outputs fail closed.",
            },
        ]
        return _finalize_score(tid, fallback_metrics, views)

    n = int(pts.shape[0])
    k_cnt = int(means.shape[0])
    dists = torch.cdist(pts, means)
    min_dist, assign = dists.min(dim=1)
    z_norm = min_dist / sigmas[assign]
    in_core = z_norm <= _HIGHD_CORE_Z_RADIUS
    precision = float(in_core.float().mean().item())

    counts = torch.bincount(assign, minlength=k_cnt).float() / float(n)
    mass_tv = float(0.5 * (counts - weights).abs().sum().item())

    core_shares = torch.bincount(assign[in_core], minlength=k_cnt).float() / float(n)
    modes = int((core_shares >= (0.40 * weights)).sum().item())
    orphan_recall = float(core_shares[4].item() / weights[4].item())

    min_eigs: list[float] = []
    max_eigs: list[float] = []
    for k in range(k_cnt):
        pts_k = pts[(assign == k) & in_core]
        target_var = float(sigmas[k].item()) ** 2
        if len(pts_k) >= _HIGHD_DIM + 2:
            cov_k = torch.cov(pts_k.T, correction=0).double()
            ev = torch.linalg.eigvalsh(cov_k).clamp_min(0.0)
            min_eigs.append(float(ev[0].item() / max(target_var, 1e-12)))
            max_eigs.append(float(ev[-1].item() / max(target_var, 1e-12)))
        else:
            min_eigs.append(0.0)
            max_eigs.append(99.0)

    pts_2d = (pts[:1024] @ proj2).clamp(-4.45, 9.45)
    pts_2d[:, 1] = pts_2d[:, 1].clamp(-4.45, 4.45)

    metrics: dict[str, float | int] = {
        "finite_fraction": 1.0,
        "modes": modes,
        "core_eig_min": float(min(min_eigs)),
        "core_eig_max": float(max(max_eigs)),
        "orphan_recall": orphan_recall,
        "precision": precision,
        "mass_tv": mass_tv,
    }
    views = [
        {
            "kind": "scatter",
            "title": "R^32 canonical 2D projection (core + distant orphan)",
            "target": ref_2d,
            "samples": pts_2d,
            "xlim": [-4.5, 9.5],
            "ylim": [-4.5, 4.5],
            "aspect": "equal",
            "xlabel": "q0 (canonical axis 0)",
            "ylabel": "q1 (canonical axis 1)",
            "caption": "Haar-rotated R^32 (dim > 16): 4 multi-scale cores + distant 8% orphan at (+7.5, 0).",
        },
        {
            "kind": "bar",
            "title": "5-mode unequal mass + 8% orphan profile",
            "target": weights,
            "samples": counts,
            "xlabel": "mode index (4 = orphan)",
            "ylabel": "mass",
            "caption": "Target weights [0.36, 0.26, 0.18, 0.12, 0.08 orphan].",
        },
    ]
    return _finalize_score(tid, metrics, views)


@torch.no_grad()
def score_fl_law1_noisy_atomic_codebook_v2(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score calibrated v2 64-atom noisy hexagonal Zipf codebook (P=160, sigma_prior=0.010)."""
    base = score_fl_law1_noisy_atomic_codebook(samples, reference=reference)
    metrics = {k: v for k, v in base["metrics"].items() if k != "min_margin"}
    return _finalize_score("fl_law1_noisy_atomic_codebook_v2", metrics, base["views"])


@torch.no_grad()
def score_fl_law2_groupnorm_and_vit_sprites_v2(
    samples: torch.Tensor,
    *,
    critic_grad_finite: float = 1.0,
    critic_grad_med: float = 0.25,
    critic_grad_max: float = 0.50,
    mean_step_ms: float = 20.0,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score calibrated 1600-step v2 32x32 RGB sprites with ResNet+GroupNorm+ViT critic."""
    base = score_fl_law2_groupnorm_and_vit_sprites(
        samples,
        critic_grad_finite=critic_grad_finite,
        critic_grad_med=critic_grad_med,
        critic_grad_max=critic_grad_max,
        mean_step_ms=mean_step_ms,
        reference=reference,
    )
    metrics = {k: v for k, v in base["metrics"].items() if k != "min_margin"}
    return _finalize_score("fl_law2_groupnorm_and_vit_sprites_v2", metrics, base["views"])


@torch.no_grad()
def _score_highd_d32_common(
    tid: str,
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Shared scorer for Law 3A high-D orphan transport (v2) and Law 1 full-32D rank (v2)."""
    weights, means, sigmas, q = highd_orphan_parameters("cpu")
    proj2 = q[:, :2]
    ref = _default_reference(tid, 1024, 91005) if reference is None else reference.detach().cpu().to(torch.float32)
    ref_2d = ref[:1024] @ proj2

    valid = (
        isinstance(samples, torch.Tensor)
        and samples.ndim == 2
        and samples.shape[0] >= 32
        and samples.shape[1] == _HIGHD_DIM
    )
    if valid:
        pts = samples.detach().cpu().to(torch.float32)
        all_finite = bool(torch.isfinite(pts).all().item())
        finite_fraction = 1.0 if all_finite else float(torch.isfinite(pts).all(dim=-1).float().mean().item())
    else:
        pts = torch.zeros((2, _HIGHD_DIM), dtype=torch.float32)
        all_finite = False
        finite_fraction = 0.0

    if not valid or not all_finite:
        fallback_metrics: dict[str, float | int] = {
            "finite_fraction": finite_fraction,
            "modes": 0,
            "core_eig_min": 0.0,
            "core_eig_max": 99.0,
            "canon_plane_eig_min": 0.0,
            "orphan_recall": 0.0,
            "precision": 0.0,
            "mass_tv": 1.0,
        }
        views = [
            {
                "kind": "scatter",
                "title": f"{tid}: R^32 canonical 2D projection",
                "target": ref_2d,
                "samples": torch.zeros((2, 2), dtype=torch.float32),
                "xlim": [-4.5, 9.5],
                "ylim": [-4.5, 4.5],
                "aspect": "equal",
                "xlabel": "q0 (canonical axis 0)",
                "ylabel": "q1 (canonical axis 1)",
                "caption": "Nonfinite or invalid-shape outputs fail closed.",
            },
            {
                "kind": "bar",
                "title": "5-mode unequal mass + 8% orphan profile",
                "target": weights,
                "samples": torch.zeros_like(weights),
                "xlabel": "mode index (4 = orphan)",
                "ylabel": "mass",
                "caption": "Nonfinite outputs fail closed.",
            },
        ]
        return _finalize_score(tid, fallback_metrics, views)

    n = int(pts.shape[0])
    k_cnt = int(means.shape[0])
    dists = torch.cdist(pts, means)
    min_dist, assign = dists.min(dim=1)
    z_norm = min_dist / sigmas[assign]
    in_core = z_norm <= _HIGHD_CORE_Z_RADIUS
    precision = float(in_core.float().mean().item())

    counts = torch.bincount(assign, minlength=k_cnt).float() / float(n)
    mass_tv = float(0.5 * (counts - weights).abs().sum().item())

    core_shares = torch.bincount(assign[in_core], minlength=k_cnt).float() / float(n)
    modes = int((core_shares >= (0.40 * weights)).sum().item())
    orphan_recall = float(core_shares[4].item() / weights[4].item())

    min_eigs: list[float] = []
    max_eigs: list[float] = []
    canon_min_eigs: list[float] = []
    for k in range(k_cnt):
        pts_k = pts[(assign == k) & in_core]
        target_var = float(sigmas[k].item()) ** 2
        if len(pts_k) >= _HIGHD_DIM + 2:
            cov_k = torch.cov(pts_k.T, correction=0).double()
            ev = torch.linalg.eigvalsh(cov_k).clamp_min(0.0)
            min_eigs.append(float(ev[0].item() / max(target_var, 1e-12)))
            max_eigs.append(float(ev[-1].item() / max(target_var, 1e-12)))
        else:
            min_eigs.append(0.0)
            max_eigs.append(99.0)
        if len(pts_k) >= 6:
            pts_k_2d = pts_k @ proj2
            cov_2d = torch.cov(pts_k_2d.T, correction=0).double()
            ev_2d = torch.linalg.eigvalsh(cov_2d).clamp_min(0.0)
            canon_min_eigs.append(float(ev_2d[0].item() / max(target_var, 1e-12)))
        else:
            canon_min_eigs.append(0.0)

    pts_2d = (pts[:1024] @ proj2).clamp(-4.45, 9.45)
    pts_2d[:, 1] = pts_2d[:, 1].clamp(-4.45, 4.45)
    metrics: dict[str, float | int] = {
        "finite_fraction": 1.0,
        "modes": modes,
        "core_eig_min": float(min(min_eigs)),
        "core_eig_max": float(max(max_eigs)),
        "canon_plane_eig_min": float(min(canon_min_eigs)),
        "orphan_recall": orphan_recall,
        "precision": precision,
        "mass_tv": mass_tv,
    }
    views = [
        {
            "kind": "scatter",
            "title": f"{tid}: R^32 canonical 2D projection",
            "target": ref_2d,
            "samples": pts_2d,
            "xlim": [-4.5, 9.5],
            "ylim": [-4.5, 4.5],
            "aspect": "equal",
            "xlabel": "q0 (canonical axis 0)",
            "ylabel": "q1 (canonical axis 1)",
            "caption": "Haar-rotated R^32 (dim > 16): 4 multi-scale cores + distant 8% orphan at (+7.5, 0).",
        },
        {
            "kind": "bar",
            "title": "5-mode unequal mass + 8% orphan profile",
            "target": weights,
            "samples": counts,
            "xlabel": "mode index (4 = orphan)",
            "ylabel": "mass",
            "caption": "Target weights [0.36, 0.26, 0.18, 0.12, 0.08 orphan].",
        },
    ]
    return _finalize_score(tid, metrics, views)


@torch.no_grad()
def score_fl_law3_highd_orphan_d32_v2(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score v2 Law 3A Haar-rotated R^32 orphan transport (canonical-plane eig, decoupled from 32D rank)."""
    return _score_highd_d32_common("fl_law3_highd_orphan_d32_v2", samples, reference=reference)


@torch.no_grad()
def score_fl_law1_highd_rank_d32_v2(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score v2 Law 1 full-32D covariance rank with P=512 particles."""
    return _score_highd_d32_common("fl_law1_highd_rank_d32_v2", samples, reference=reference)


_SCORERS = {
    "fl_law1_aniso_unequal_modes": score_fl_law1_aniso_unequal_modes,
    "fl_law1_noisy_atomic_codebook": score_fl_law1_noisy_atomic_codebook,
    "fl_law2_pure_adversarial_codim1": score_fl_law2_pure_adversarial_codim1,
    "fl_law2_groupnorm_and_vit_sprites": score_fl_law2_groupnorm_and_vit_sprites,
    "fl_law3_highd_orphan_d32": score_fl_law3_highd_orphan_d32,
    "fl_law1_noisy_atomic_codebook_v2": score_fl_law1_noisy_atomic_codebook_v2,
    "fl_law1_highd_rank_d32_v2": score_fl_law1_highd_rank_d32_v2,
    "fl_law2_groupnorm_and_vit_sprites_v2": score_fl_law2_groupnorm_and_vit_sprites_v2,
    "fl_law3_highd_orphan_d32_v2": score_fl_law3_highd_orphan_d32_v2,
}


def score_task(task_id: str, samples: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
    """Dispatch scoring to the named Four Laws 1-3 task scorer."""
    if task_id not in _SCORERS:
        raise ValueError(f"unknown Four Laws 1-3 task_id: {task_id!r}")
    return _SCORERS[task_id](samples, **kwargs)


# ---------------------------------------------------------------------------
# Fast Deterministic CPU Oracle & Destructive Controls (<0.5s total)
# ---------------------------------------------------------------------------


def _control_entry(result: dict[str, Any], *, expected_pass: bool) -> dict[str, Any]:
    return {
        "expected_pass": bool(expected_pass),
        "passed": bool(result["passed"]),
        "failed_bounds": list(result["failed_bounds"]),
        "min_margin": float(result["min_margin"]),
    }


def oracle_controls() -> dict[str, dict[str, Any]]:
    """Fast deterministic CPU positive oracle + destructive negative controls for all 5 tasks."""
    controls: dict[str, dict[str, Any]] = {}
    rng = torch.Generator(device="cpu").manual_seed(20261125)

    # --- 1. fl_law1_aniso_unequal_modes ---
    t1 = "fl_law1_aniso_unequal_modes"
    ref1 = sample_target_for_kind(t1, 1024, rng)
    ora1 = sample_target_for_kind(t1, 4096, rng)
    controls[f"{t1}_oracle_pass"] = _control_entry(
        score_fl_law1_aniso_unequal_modes(ora1, reference=ref1), expected_pass=True
    )
    w1, m1, _, _ = aniso_ellipse_parameters("cpu")
    idx_iso = torch.multinomial(w1, 4096, replacement=True, generator=rng)
    iso_pts = m1[idx_iso] + _ANISO_SIGMA_MAJOR * torch.randn((4096, 2), generator=rng)
    controls[f"{t1}_isotropic_collapse_fail"] = _control_entry(
        score_fl_law1_aniso_unequal_modes(iso_pts, reference=ref1), expected_pass=False
    )
    collapsed1 = m1[0:1].expand(4096, 2) + 0.02 * torch.randn((4096, 2), generator=rng)
    controls[f"{t1}_mode_collapsed_fail"] = _control_entry(
        score_fl_law1_aniso_unequal_modes(collapsed1, reference=ref1), expected_pass=False
    )
    nan1 = ora1.clone()
    nan1[0, 0] = float("nan")
    controls[f"{t1}_nonfinite_fail"] = _control_entry(
        score_fl_law1_aniso_unequal_modes(nan1, reference=ref1), expected_pass=False
    )

    # --- 2. fl_law1_noisy_atomic_codebook ---
    t2 = "fl_law1_noisy_atomic_codebook"
    ref2 = sample_target_for_kind(t2, 1024, rng)
    ora2 = sample_target_for_kind(t2, 4096, rng)
    controls[f"{t2}_oracle_pass"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook(ora2, reference=ref2), expected_pass=True
    )
    _, c2, s2, _ = noisy_codebook_parameters("cpu")
    unif_idx = torch.randint(0, _CODEBOOK_NUM_ATOMS, (4096,), generator=rng)
    unif_pts = c2[unif_idx] + s2 * torch.randn((4096, 2), generator=rng)
    controls[f"{t2}_uniform_mass_impostor_fail"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook(unif_pts, reference=ref2), expected_pass=False
    )
    ghost_pts = 0.5 * (c2[unif_idx] + c2[(unif_idx + 1) % _CODEBOOK_NUM_ATOMS])
    controls[f"{t2}_barycentric_ghosts_fail"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook(ghost_pts, reference=ref2), expected_pass=False
    )
    nan2 = ora2.clone()
    nan2[3, 1] = float("inf")
    controls[f"{t2}_nonfinite_fail"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook(nan2, reference=ref2), expected_pass=False
    )

    # --- 3. fl_law2_pure_adversarial_codim1 ---
    t3 = "fl_law2_pure_adversarial_codim1"
    ref3 = sample_target_for_kind(t3, 1024, rng)
    ora3 = sample_target_for_kind(t3, 2048, rng)
    steady_trace = [[float(k * 10), 0.002 * math.sin(0.3 * k)] for k in range(30)]
    controls[f"{t3}_oracle_pass"] = _control_entry(
        score_fl_law2_pure_adversarial_codim1(
            ora3, orbit_trace=steady_trace, normal_grad_norm=0.25, normal_grad_max=0.45, reference=ref3
        ),
        expected_pass=True,
    )
    osc_trace = [[float(k * 10), 0.25 * math.sin(0.9 * k)] for k in range(30)]
    controls[f"{t3}_oscillating_orbit_fail"] = _control_entry(
        score_fl_law2_pure_adversarial_codim1(
            ora3, orbit_trace=osc_trace, normal_grad_norm=0.25, normal_grad_max=0.45, reference=ref3
        ),
        expected_pass=False,
    )
    thick3 = ora3 + 0.35 * torch.randn((2048, 2), generator=rng)
    controls[f"{t3}_off_manifold_thick_fail"] = _control_entry(
        score_fl_law2_pure_adversarial_codim1(
            thick3, orbit_trace=steady_trace, normal_grad_norm=0.25, normal_grad_max=0.45, reference=ref3
        ),
        expected_pass=False,
    )
    theta_arc = torch.rand((2048,), generator=rng) * (0.4 * math.pi)
    arc3 = torch.stack((_CODIM1_RADIUS * torch.cos(theta_arc), _CODIM1_RADIUS * torch.sin(theta_arc)), dim=1)
    controls[f"{t3}_arc_collapse_fail"] = _control_entry(
        score_fl_law2_pure_adversarial_codim1(
            arc3, orbit_trace=steady_trace, normal_grad_norm=0.25, normal_grad_max=0.45, reference=ref3
        ),
        expected_pass=False,
    )
    nan3 = ora3.clone()
    nan3[1, 0] = float("nan")
    controls[f"{t3}_nonfinite_fail"] = _control_entry(
        score_fl_law2_pure_adversarial_codim1(
            nan3, orbit_trace=steady_trace, normal_grad_norm=0.25, normal_grad_max=0.45, reference=ref3
        ),
        expected_pass=False,
    )

    # --- 4. fl_law2_groupnorm_and_vit_sprites ---
    t4 = "fl_law2_groupnorm_and_vit_sprites"
    tpl4 = sprite_templates("cpu")
    ora4 = sample_target_for_kind(t4, 256, rng)
    controls[f"{t4}_oracle_pass"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites(
            ora4, critic_grad_finite=1.0, critic_grad_med=0.30, critic_grad_max=0.60, mean_step_ms=30.0, reference=tpl4
        ),
        expected_pass=True,
    )
    collapsed4 = tpl4[0:1].expand(256, -1, -1, -1).clone()
    controls[f"{t4}_single_class_collapse_fail"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites(
            collapsed4, critic_grad_finite=1.0, mean_step_ms=30.0, reference=tpl4
        ),
        expected_pass=False,
    )
    noise4 = torch.rand((256, _SPRITE_CHANNELS, _SPRITE_SIZE, _SPRITE_SIZE), generator=rng)
    controls[f"{t4}_uniform_noise_fail"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites(
            noise4, critic_grad_finite=1.0, mean_step_ms=30.0, reference=tpl4
        ),
        expected_pass=False,
    )
    nan4 = ora4.clone()
    nan4[0, 0, 0, 0] = float("nan")
    controls[f"{t4}_nonfinite_fail"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites(
            nan4, critic_grad_finite=1.0, mean_step_ms=30.0, reference=tpl4
        ),
        expected_pass=False,
    )

    # --- 5. fl_law3_highd_orphan_d32 ---
    t5 = "fl_law3_highd_orphan_d32"
    ref5 = sample_target_for_kind(t5, 1024, rng)
    ora5 = sample_target_for_kind(t5, 6144, rng)
    controls[f"{t5}_oracle_pass"] = _control_entry(
        score_fl_law3_highd_orphan_d32(ora5, reference=ref5), expected_pass=True
    )
    w5, m5, s5, _ = highd_orphan_parameters("cpu")
    w_no_orphan = w5[:4] / w5[:4].sum()
    idx_no = torch.multinomial(w_no_orphan, 6144, replacement=True, generator=rng)
    no_orphan = m5[idx_no] + s5[idx_no].unsqueeze(1) * torch.randn((6144, _HIGHD_DIM), generator=rng)
    controls[f"{t5}_missing_orphan_fail"] = _control_entry(
        score_fl_law3_highd_orphan_d32(no_orphan, reference=ref5), expected_pass=False
    )
    idx_col5 = torch.multinomial(w5, 6144, replacement=True, generator=rng)
    collapsed_cov5 = m5[idx_col5] + 0.003 * torch.randn((6144, _HIGHD_DIM), generator=rng)
    controls[f"{t5}_collapsed_covariance_fail"] = _control_entry(
        score_fl_law3_highd_orphan_d32(collapsed_cov5, reference=ref5), expected_pass=False
    )
    shifted5 = ora5 + 15.0
    controls[f"{t5}_shifted_exploded_fail"] = _control_entry(
        score_fl_law3_highd_orphan_d32(shifted5, reference=ref5), expected_pass=False
    )
    nan5 = ora5.clone()
    nan5[0, 5] = float("nan")
    controls[f"{t5}_nonfinite_fail"] = _control_entry(
        score_fl_law3_highd_orphan_d32(nan5, reference=ref5), expected_pass=False
    )

    return controls


def oracle_controls_v2() -> dict[str, dict[str, Any]]:
    """Fast deterministic CPU positive oracle + destructive controls for all 4 v2 Laws 1-3 probes."""
    controls: dict[str, dict[str, Any]] = {}
    rng = torch.Generator(device="cpu").manual_seed(20261126)

    # --- 1. fl_law1_noisy_atomic_codebook_v2 ---
    t1 = "fl_law1_noisy_atomic_codebook_v2"
    ref1 = sample_target_for_kind(t1, 1024, rng)
    ora1 = sample_target_for_kind(t1, 4096, rng)
    controls[f"{t1}_oracle_pass"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook_v2(ora1, reference=ref1), expected_pass=True
    )
    _, c1, s1, _ = noisy_codebook_parameters("cpu")
    unif_idx = torch.randint(0, _CODEBOOK_NUM_ATOMS, (4096,), generator=rng)
    ghost_pts = 0.5 * (c1[unif_idx] + c1[(unif_idx + 1) % _CODEBOOK_NUM_ATOMS])
    controls[f"{t1}_barycentric_ghosts_fail"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook_v2(ghost_pts, reference=ref1), expected_pass=False
    )
    half_pts = c1[torch.randint(0, 32, (4096,), generator=rng)] + s1 * torch.randn((4096, 2), generator=rng)
    controls[f"{t1}_half_codebook_fail"] = _control_entry(
        score_fl_law1_noisy_atomic_codebook_v2(half_pts, reference=ref1), expected_pass=False
    )

    # --- 2. fl_law2_groupnorm_and_vit_sprites_v2 ---
    t2 = "fl_law2_groupnorm_and_vit_sprites_v2"
    tpl2 = sprite_templates("cpu")
    ora2 = sample_target_for_kind(t2, 256, rng)
    controls[f"{t2}_oracle_pass"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites_v2(ora2, reference=tpl2), expected_pass=True
    )
    controls[f"{t2}_single_class_fail"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites_v2(tpl2[0:1].expand(256, -1, -1, -1).clone(), reference=tpl2),
        expected_pass=False,
    )
    controls[f"{t2}_noise_fail"] = _control_entry(
        score_fl_law2_groupnorm_and_vit_sprites_v2(
            torch.rand((256, _SPRITE_CHANNELS, _SPRITE_SIZE, _SPRITE_SIZE), generator=rng), reference=tpl2
        ),
        expected_pass=False,
    )

    # --- 3. fl_law3_highd_orphan_d32_v2 (Law 3A orphan transport, canonical-plane eig) ---
    t3 = "fl_law3_highd_orphan_d32_v2"
    ref3 = sample_target_for_kind(t3, 1024, rng)
    ora3 = sample_target_for_kind(t3, 6144, rng)
    controls[f"{t3}_oracle_pass"] = _control_entry(
        score_fl_law3_highd_orphan_d32_v2(ora3, reference=ref3), expected_pass=True
    )
    w5, m5, s5, q5 = highd_orphan_parameters("cpu")
    w_no = w5[:4] / w5[:4].sum()
    idx_no = torch.multinomial(w_no, 6144, replacement=True, generator=rng)
    no_orphan = m5[idx_no] + s5[idx_no].unsqueeze(1) * torch.randn((6144, _HIGHD_DIM), generator=rng)
    controls[f"{t3}_missing_orphan_fail"] = _control_entry(
        score_fl_law3_highd_orphan_d32_v2(no_orphan, reference=ref3), expected_pass=False
    )
    idx_all = torch.multinomial(w5, 6144, replacement=True, generator=rng)
    collapsed_canon = m5[idx_all] + 0.002 * torch.randn((6144, _HIGHD_DIM), generator=rng)
    controls[f"{t3}_collapsed_canon_fail"] = _control_entry(
        score_fl_law3_highd_orphan_d32_v2(collapsed_canon, reference=ref3), expected_pass=False
    )

    # --- 4. fl_law1_highd_rank_d32_v2 (Law 1 full-32D rank) ---
    t4 = "fl_law1_highd_rank_d32_v2"
    controls[f"{t4}_oracle_pass"] = _control_entry(
        score_fl_law1_highd_rank_d32_v2(ora3, reference=ref3), expected_pass=True
    )
    z_lowrank = torch.zeros((6144, _HIGHD_DIM), dtype=torch.float32)
    z_lowrank[:, :8] = torch.randn((6144, 8), generator=rng)
    lowrank_pts = m5[idx_all] + s5[idx_all].unsqueeze(1) * (z_lowrank @ q5.T)
    controls[f"{t4}_rank8_collapse_fail"] = _control_entry(
        score_fl_law1_highd_rank_d32_v2(lowrank_pts, reference=ref3), expected_pass=False
    )
    nan4 = ora3.clone()
    nan4[0, 3] = float("nan")
    controls[f"{t4}_nonfinite_fail"] = _control_entry(
        score_fl_law1_highd_rank_d32_v2(nan4, reference=ref3), expected_pass=False
    )
    return controls


# ---------------------------------------------------------------------------
# Case Listing, Public-API Fixture & GIF Helper
# ---------------------------------------------------------------------------


def _build_case_records(task_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for tid in task_ids:
        spec = ALL_SPECS[tid]
        cases.append(
            {
                "id": tid,
                "title": spec["title"],
                "goal": spec["goal"],
                "kind": spec["view_kind"],
                "scope": (
                    f"Public API Four Laws ({spec['law']}; defects {', '.join(spec['defects'])}) "
                    "dynamical falsification task."
                ),
                "legacy_ids": [f"four-laws-{tid}"],
                "default_steps": int(spec["steps"]),
                "batch_size": int(spec["resources"]["batch_size"]),
                "eval_samples": int(spec["eval_samples"]),
                "evaluation_observations": int(spec["observations"]),
                "terminal_observations": int(spec["minimum_stable_checks"]),
                "thresholds": {name: f"{op} {bound}" for name, op, bound in ALL_THRESHOLDS[tid]},
                "sampling": "Clean live public prior samples without additive output noise.",
                "default_recipe": "unified_v14",
                "protocol_seed": DEFAULT_SEED,
            }
        )
    return cases


def list_cases() -> list[dict[str, Any]]:
    """Return ``api_contract.validate_case``-compliant case metadata for the 5 frozen v1 tasks."""
    return _build_case_records(TASKS)


def list_v2_cases() -> list[dict[str, Any]]:
    """Return ``api_contract.validate_case``-compliant case metadata for the 4 calibrated v2 tasks."""
    return _build_case_records(V2_TASKS)


def list_all_cases() -> list[dict[str, Any]]:
    """Return case metadata for both v1 and v2 Laws 1-3 tasks."""
    return _build_case_records(ALL_TASKS)


def _seed_int(tag: str, seed: int) -> int:
    digest = hashlib.sha256(f"api_four_laws_123:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _measure_normal_critic_grad(critic: nn.Module, pts: torch.Tensor) -> tuple[float, float]:
    """Optional non-gating diagnostic: mean/max ``|grad_x D(x) . n_hat(x)|`` on circle probes."""
    n = min(len(pts), 128)
    if n < 1 or not bool(torch.isfinite(pts).all()):
        return float("nan"), float("nan")
    modes = [(m, m.training) for m in critic.modules()]
    first_param = next(critic.parameters(), None)
    if first_param is None:
        return float("nan"), float("nan")
    dev = first_param.device
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    try:
        critic.eval()
        with torch.random.fork_rng(devices=devices):
            probes = pts[:n].detach().to(device=dev, dtype=torch.float32).clone().requires_grad_(True)
            with torch.enable_grad():
                out = critic(probes)
                grad = torch.autograd.grad(out.sum(), probes, create_graph=False)[0]
        if not bool(torch.isfinite(grad).all()):
            return float("nan"), float("nan")
        n_hat = probes.detach() / probes.detach().norm(dim=1, keepdim=True).clamp_min(1e-6)
        proj = (grad.detach() * n_hat).sum(dim=1).abs()
        return float(proj.mean().item()), float(proj.max().item())
    except Exception:
        return float("nan"), float("nan")
    finally:
        for m, flag in modes:
            m.training = flag


def _measure_sprite_critic_grad(critic: nn.Module, imgs: torch.Tensor) -> tuple[float, float, float]:
    """Optional non-gating diagnostic: finite input gradient check on sprite probes."""
    n = min(len(imgs), 16)
    if n < 1 or not bool(torch.isfinite(imgs).all()):
        return 0.0, float("nan"), float("nan")
    modes = [(m, m.training) for m in critic.modules()]
    first_param = next(critic.parameters(), None)
    if first_param is None:
        return 0.0, float("nan"), float("nan")
    dev = first_param.device
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    try:
        critic.eval()
        with torch.random.fork_rng(devices=devices):
            probes = imgs[:n].detach().to(device=dev, dtype=torch.float32).clone().requires_grad_(True)
            with torch.enable_grad():
                out = critic(probes)
                if not bool(torch.isfinite(out).all()):
                    return 0.0, float("nan"), float("nan")
                grad = torch.autograd.grad(out.sum(), probes, create_graph=False)[0]
        if not bool(torch.isfinite(grad).all()):
            return 0.0, float("nan"), float("nan")
        norms = grad.detach().reshape(n, -1).norm(dim=1)
        return 1.0, float(norms.median().item()), float(norms.max().item())
    except Exception:
        return 0.0, float("nan"), float("nan")
    finally:
        for m, flag in modes:
            m.training = flag


class FourLawsFixture:
    """Checkpointable public-API ``GANTrainer`` fixture for Laws 1, 2, and 3 tasks."""

    api_components = ("GANTrainer", "Recipe", "init.deterministic_orthogonal_")

    def __init__(
        self,
        case_id: str,
        *,
        recipe_name: str | None = "unified_v14",
        recipe: Recipe | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
        components: Any = None,
    ) -> None:
        if case_id not in ALL_SPECS:
            raise ValueError(f"unknown Four Laws 1-3 task: {case_id!r}")
        self.task_id = case_id
        self.case_id = case_id
        self.kind = case_id
        self.spec = deepcopy(ALL_SPECS[case_id])
        self.case = next(c for c in list_all_cases() if c["id"] == case_id)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.limit = int(self.spec["steps"] if max_steps is None else max_steps)
        if type(self.limit) is not int or self.limit < 1 or self.seed < 0:
            raise ValueError("max_steps must be positive and seed must be nonnegative")
        self.components = components
        self.aux_trainers: dict[str, GANTrainer] = {}

        if components is not None:
            self.recipe = components.recipe
            g_mod = components.construct(
                lambda: build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)[0],
                component="generator",
            ).to(self.device)
            d_mod = components.construct(
                lambda: build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)[1],
                component="discriminator",
            ).to(self.device)
            self.trainer = components.build_trainer(g_mod, d_mod, max_steps=self.limit)
            self.data_rng = components.streams.generator(
                "data", component="target", purpose="training", device="cpu"
            )
            self.generator_data_rng = components.streams.generator(
                "data", component="target", purpose="generator_training", device="cpu"
            )
        else:
            res = self.spec["resources"]
            prior_cfg = self.spec["prior"]
            base_recipe = get_recipe(recipe_name or "unified_v14") if recipe is None else recipe
            prior_kind = "mog" if prior_cfg["kind"] == "mog" else "particles"
            updates: dict[str, Any] = {
                "num_particles": int(res["num_particles"]),
                "z_dim": int(res["z_dim"]),
                "batch_size": int(res["batch_size"]),
                "prior_kind": prior_kind,
                "sigma_rel": 0.0,
                "standardize": bool(prior_cfg["standardize"]),
                **(recipe_overrides or {}),
            }
            if base_recipe.total_steps is not None:
                updates["total_steps"] = self.limit
            self.recipe = base_recipe.replace(**updates)

            with construction_rng(self.seed, self.device):
                g_mod, d_mod = build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)
            g_mod, d_mod = g_mod.to(self.device), d_mod.to(self.device)
            for comp, mod in (("generator", g_mod), ("discriminator", d_mod)):
                seeds = {
                    name: _seed_int(f"{case_id}:init:{comp}:{name}", self.seed)
                    for name, p in mod.named_parameters()
                    if p.requires_grad and p.numel()
                }
                init.deterministic_orthogonal_(mod, parameter_seeds=seeds)

            prior_rng = torch.Generator(device=self.device).manual_seed(
                _seed_int(f"{case_id}:init:prior:locations", self.seed)
            )
            prior_opts: dict[str, Any] = {
                "device": self.device,
                "dtype": torch.float32,
                "generator": prior_rng,
                "init_std": 1.0,
            }
            if prior_kind == "mog":
                prior_opts["sigma"] = float(prior_cfg["sigma"])
            prior = self.recipe.replace(prior_update="learned").make_prior(**prior_opts)
            base_prior = unwrap_prior(prior)
            prior_seeds = {
                name: _seed_int(f"{case_id}:init:prior:{name}", self.seed)
                for name, p in base_prior.named_parameters()
                if p.requires_grad and p.numel()
            }
            if prior_seeds:
                init.deterministic_orthogonal_(base_prior, parameter_seeds=prior_seeds)
            self.recipe.apply_prior_policy(prior)

            trainer_kwargs: dict[str, Any] = {
                "prior": prior,
                "seed": self.seed,
                "max_steps": self.limit,
                "latent_generator": torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:prior:latent", self.seed)
                ),
                "penalty_generator": torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:noise:penalty", self.seed)
                ),
                "noise_generator": torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:noise:output", self.seed)
                ),
                "input_noise_generator": torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:noise:input", self.seed)
                ),
                "model_generator": torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:noise:model", self.seed)
                ),
                "eval_generator": torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:eval:sampler", self.seed)
                ),
                "require_latent_damping": self.recipe.latent_damping_max_rate > 0,
            }
            if prior_kind == "mog":
                trainer_kwargs["prior_noise_generator"] = torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{case_id}:noise:prior", self.seed)
                )
            self.trainer = GANTrainer(self.recipe, g_mod, d_mod, **trainer_kwargs)
            self.data_rng = torch.Generator(device="cpu").manual_seed(
                _seed_int(f"{case_id}:data:target:training", self.seed)
            )
            self.generator_data_rng = torch.Generator(device="cpu").manual_seed(
                _seed_int(f"{case_id}:data:target:generator_training", self.seed)
            )

        self._batch_sequence_sha256 = hashlib.sha256(b"").hexdigest()
        self._orbit_trace: list[list[float]] = []
        self._step_times_ms: list[float] = []  # Runtime-only telemetry (never enters state_dict)
        if self.task_id == "fl_law2_pure_adversarial_codim1":
            self._record_orbit_point(0)

        self.metadata = {
            "checkpoint_capabilities": {
                "round_trip": True,
                "preserves_rng": True,
                "exact_continuation": True,
                "owner": "GANTrainer.state_dict + caller data RNGs + deterministic orbit trace",
            }
        }

    @property
    def completed_steps(self) -> int:
        return int(self.trainer.completed_steps)

    @property
    def batch_sequence_sha256(self) -> str:
        return self._batch_sequence_sha256

    @torch.no_grad()
    def _record_orbit_point(self, step: int) -> None:
        """Record mean normal offset from public ``trainer.sample()`` without touching training RNGs."""
        g_modes = [(m, m.training) for m in self.trainer.G.modules()]
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        try:
            with torch.random.fork_rng(devices=devices):
                probe_gen = torch.Generator(device=self.device).manual_seed(
                    _seed_int(f"{self.task_id}:orbit_probe:{step}", self.seed)
                )
                out = self.trainer.sample(128, output_noise=False, generator=probe_gen)
            if bool(torch.isfinite(out).all()):
                val = float((out.norm(dim=1) - _CODIM1_RADIUS).mean().item())
            else:
                val = float("nan")
        finally:
            for m, flag in g_modes:
                m.training = flag
        self._orbit_trace.append([float(step), val])

    def step(self) -> dict[str, Any]:
        b_sz = int(self.recipe.batch_size)
        real_d_cpu = sample_target_for_kind(self.task_id, b_sz, self.data_rng, device="cpu")
        real_g_cpu = sample_target_for_kind(self.task_id, b_sz, self.generator_data_rng, device="cpu")
        step_hash = hashlib.sha256(
            real_d_cpu.numpy().tobytes() + real_g_cpu.numpy().tobytes()
        ).digest()
        self._batch_sequence_sha256 = hashlib.sha256(
            bytes.fromhex(self._batch_sequence_sha256) + step_hash
        ).hexdigest()

        t0 = time.perf_counter()
        out = self.trainer.step(
            real_d_cpu.to(self.device),
            generator_real=real_g_cpu.to(self.device),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._step_times_ms.append(dt_ms)

        if self.task_id == "fl_law2_pure_adversarial_codim1":
            cur_step = self.completed_steps
            if cur_step <= 32 or cur_step % 4 == 0 or cur_step == self.limit:
                self._record_orbit_point(cur_step)

        return {k: float(v.item()) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    @torch.no_grad()
    def observe(self, n: int | None = None, seed: int = 10000) -> dict[str, Any]:
        eval_n = int(self.spec["eval_samples"] if n is None else n)
        if self.components is not None:
            sample_stream = self.components.streams.generator(
                "eval", component="live", purpose="samples"
            )
            ref_stream = self.components.streams.generator(
                "eval", component="target", purpose="reference", device="cpu"
            )
        else:
            sample_stream = torch.Generator(device=self.device).manual_seed(int(seed))
            ref_stream = torch.Generator(device="cpu").manual_seed(int(seed) + 100003)

        g_modes = [(m, m.training) for m in self.trainer.G.modules()]
        d_modes = [(m, m.training) for m in self.trainer.D.modules()]
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        try:
            with torch.random.fork_rng(devices=devices):
                samples = self.trainer.sample(eval_n, output_noise=False, generator=sample_stream)
                ref_count = min(eval_n, 1024) if self.spec["view_kind"] == "vector" else 16
                reference = sample_target_for_kind(self.task_id, ref_count, ref_stream, device="cpu")

                if self.task_id == "fl_law1_aniso_unequal_modes":
                    return score_fl_law1_aniso_unequal_modes(samples, reference=reference)

                if self.task_id == "fl_law1_noisy_atomic_codebook":
                    return score_fl_law1_noisy_atomic_codebook(samples, reference=reference)

                if self.task_id == "fl_law1_noisy_atomic_codebook_v2":
                    return score_fl_law1_noisy_atomic_codebook_v2(samples, reference=reference)

                if self.task_id == "fl_law2_pure_adversarial_codim1":
                    probe_pts = torch.cat((reference[:64], samples.detach().cpu()[:64]), dim=0)
                    grad_mean, grad_max = _measure_normal_critic_grad(self.trainer.D, probe_pts)
                    return score_fl_law2_pure_adversarial_codim1(
                        samples,
                        orbit_trace=self._orbit_trace,
                        normal_grad_norm=grad_mean,
                        normal_grad_max=grad_max,
                        reference=reference,
                    )

                if self.task_id in ("fl_law2_groupnorm_and_vit_sprites", "fl_law2_groupnorm_and_vit_sprites_v2"):
                    probe_imgs = torch.cat((reference[:8], samples.detach().cpu()[:8]), dim=0)
                    g_fin, g_med, g_max = _measure_sprite_critic_grad(self.trainer.D, probe_imgs)
                    mean_ms = (
                        float(sum(self._step_times_ms) / len(self._step_times_ms))
                        if self._step_times_ms
                        else 0.0
                    )
                    scorer_fn = (
                        score_fl_law2_groupnorm_and_vit_sprites_v2
                        if self.task_id == "fl_law2_groupnorm_and_vit_sprites_v2"
                        else score_fl_law2_groupnorm_and_vit_sprites
                    )
                    return scorer_fn(
                        samples,
                        critic_grad_finite=g_fin,
                        critic_grad_med=g_med,
                        critic_grad_max=g_max,
                        mean_step_ms=mean_ms,
                        reference=reference,
                    )

                if self.task_id == "fl_law3_highd_orphan_d32":
                    return score_fl_law3_highd_orphan_d32(samples, reference=reference)

                if self.task_id == "fl_law3_highd_orphan_d32_v2":
                    return score_fl_law3_highd_orphan_d32_v2(samples, reference=reference)

                if self.task_id == "fl_law1_highd_rank_d32_v2":
                    return score_fl_law1_highd_rank_d32_v2(samples, reference=reference)

                raise ValueError(f"unsupported task_id: {self.task_id!r}")
        finally:
            for m, flag in g_modes:
                m.training = flag
            for m, flag in d_modes:
                m.training = flag

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "recipe": self.recipe.to_dict(),
            "batch_sequence_sha256": self._batch_sequence_sha256,
            "data_rng": self.data_rng.get_state().clone(),
            "generator_data_rng": self.generator_data_rng.get_state().clone(),
            "orbit_trace": [list(pair) for pair in self._orbit_trace],
            "trainer": self.trainer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != VERSION or state.get("case_id") != self.case_id:
            raise ValueError("FourLawsFixture checkpoint version or case_id mismatch")
        if state.get("recipe") != self.recipe.to_dict():
            raise ValueError("FourLawsFixture checkpoint recipe mismatch")
        self.seed = int(state["seed"])
        self._batch_sequence_sha256 = str(state["batch_sequence_sha256"])
        self.data_rng.set_state(state["data_rng"].cpu())
        self.generator_data_rng.set_state(state["generator_data_rng"].cpu())
        self._orbit_trace = [list(pair) for pair in state.get("orbit_trace", [])]
        self.trainer.load_state_dict(state["trainer"])


Fixture = FourLawsFixture


def build_case(
    case_id: str,
    *,
    device: str | torch.device = "cpu",
    seed: int = DEFAULT_SEED,
    recipe_name: str | None = None,
    max_steps: int | None = None,
    recipe: Recipe | None = None,
    recipe_overrides: dict[str, Any] | None = None,
    components: Any = None,
    **kwargs: Any,
) -> FourLawsFixture:
    """Build a public-API ``FourLawsFixture`` for any Law 1, 2, or 3 task."""
    del kwargs
    if case_id not in ALL_SPECS:
        raise ValueError(f"unknown Four Laws 1-3 toy case: {case_id!r}")
    return FourLawsFixture(
        case_id,
        recipe_name=recipe_name or "unified_v14",
        recipe=recipe,
        recipe_overrides=recipe_overrides,
        device=device,
        seed=seed,
        max_steps=max_steps,
        components=components,
    )


def render_task_gif(
    case_or_task_id: str | dict[str, Any],
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    full_budget: bool = False,
    requested_steps: int | None = None,
    final_verdict: str | None = None,
) -> dict[str, Any]:
    """Thin wrapper delegating animated goal-GIF rendering to ``api_run.render_gif``."""
    from .api_run import render_gif

    if isinstance(case_or_task_id, str):
        case = next(c for c in list_all_cases() if c["id"] == case_or_task_id)
    else:
        case = case_or_task_id
    steps = int(case["default_steps"] if requested_steps is None else requested_steps)
    return render_gif(
        case,
        records,
        Path(path),
        full_budget=bool(full_budget),
        requested_steps=steps,
        final_verdict=final_verdict,
    )
