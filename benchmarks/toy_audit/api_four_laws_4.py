"""Public-API Law 4 dynamical toy experiments for the Four Laws Adherence Suite.

Four falsifiable, problem-only, method-agnostic public-API (``GANTrainer`` / ``Recipe``)
dynamical tasks grading strictly observable generated-output quality (``trainer.sample()``)
with zero task-host repairs and zero internal optimizer/prior/critic thresholds:

1. ``fl_law41_trust_rare1pct_tanh`` (Laws 4.1 & 4.3 / ``D-04``):
   2D 5-mode mixture with 4 major modes (99% mass) + 1 separated 1% minority mode
   inside the saturating tail of a ``3.0 * tanh(...)`` generator. Graded solely on
   generated outputs: ``finite_fraction``, ``modes`` (5/5), ``rare_mode_recall``,
   ``precision``, and ``mass_tv``.

2. ``fl_law42_swap_deep_critic_16l`` (Law 4.2 / ``D-10``, P5):
   Shallow 3-layer MLP generator paired with a deep 16-layer skip-free LeakyReLU
   critic (``fourier=0``) on an 8-mode 2D Gaussian ring. Graded solely on generated
   ring outputs: ``finite_fraction``, ``modes`` (8/8), ``hq``, ``mass_tv``, and
   ``valley_fraction`` (inter-mode valley contamination).

3. ``fl_law42_deep_resbn_and_conv1x1`` (Laws 4.2 & 4.5 / Bug 2 + 4D 1x1 vs 2D Linear):
   Compact high-D (``D=128``, ``w=128``, ``depth=8``) branch-RMSNorm residual trunk
   (``ResBN d=8``) paired with its mathematically isomorphic ``4D Conv2d 1x1`` twin
   on identical initial weights and seen batches. Graded solely on generated outputs:
   ``finite_fraction``, ``sw1``, ``var_ratio_err``, ``conv1x1_sw1``, and
   ``conv1x1_sw1_ratio``.

4. ``fl_law44_highd_sparse_prior_and_shock`` (Law 4.4 / Bug 1 + ``D-03``/``D-05``):
   High-D sparse-observation manifold (``D=512``, ``P=1600``, ``Z=32``, 10% random
   observation mask, RMSNorm critic) paired with an extreme-scale (``1e-5x``) 2D
   4-mode GAN experiencing a mid-run transient ``1e4x`` scale spike and mode shift.
   Graded solely on generated outputs: ``finite_fraction``, ``sw1``, ``mode_recall``,
   ``var_ratio_err``, ``shock_modes``, ``shock_hq``, and ``shock_sw1``.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import (
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    get_recipe,
    init,
    unwrap_prior,
)

from .reproducibility import DEFAULT_SEED, construction_rng

TASKS: tuple[str, ...] = (
    "fl_law41_trust_rare1pct_tanh",
    "fl_law42_swap_deep_critic_16l",
    "fl_law42_deep_resbn_and_conv1x1",
    "fl_law44_highd_sparse_prior_and_shock",
)

V2_TASKS: tuple[str, ...] = (
    "fl_law42_deep_resbn_and_conv1x1_v2",
    "fl_law44_scale_shock",
)

ALL_TASKS: tuple[str, ...] = TASKS + V2_TASKS

# ---------------------------------------------------------------------------
# Frozen Task Geometry Constants
# ---------------------------------------------------------------------------

_RARE1PCT_CENTERS: tuple[tuple[float, float], ...] = (
    (-1.5, -1.5),
    (-1.5, 1.5),
    (1.5, -1.5),
    (1.5, 1.5),
    (0.0, 2.85),
)
_RARE1PCT_WEIGHTS: tuple[float, ...] = (0.2475, 0.2475, 0.2475, 0.2475, 0.01)
_RARE1PCT_SIGMA: float = 0.08
_RARE1PCT_OUT_BOUND: float = 3.0

_RING8_MODES: int = 8
_RING8_RADIUS: float = 2.5
_RING8_SIGMA: float = 0.08

_RESBN_DIM: int = 128
_RESBN_MODES: int = 64
_RESBN_WIDTH: int = 128
_RESBN_DEPTH: int = 8
_RESBN_SIGMA: float = 0.08

_HIGHD_DIM: int = 512
_HIGHD_MODES: int = 64
_HIGHD_HIDDEN: int = 512
_HIGHD_SIGMA: float = 0.05
_HIGHD_OBS_PROB: float = 0.10
_HIGHD_SHOCK_STEP: int = 200
_SHOCK_BASE_SCALE: float = 1e-5
_SHOCK_SPIKE_SCALE: float = 1e-1
_SHOCK_SIGMA_UNIT: float = 0.08
_SHOCK_PRE_CENTERS: tuple[tuple[float, float], ...] = (
    (-1.5, -1.5),
    (-1.5, 1.5),
    (1.5, -1.5),
    (1.5, 1.5),
)
_SHOCK_POST_CENTERS: tuple[tuple[float, float], ...] = (
    (2.0, 0.0),
    (-2.0, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
)

_MANIFOLD_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _seed_int(tag: str, seed: int) -> int:
    digest = hashlib.sha256(f"api_four_laws_4:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _ring8_centers_and_valleys(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    idx = torch.arange(_RING8_MODES, device=dev, dtype=torch.float32)
    ang = (2.0 * math.pi / float(_RING8_MODES)) * idx
    centers = torch.stack((_RING8_RADIUS * ang.cos(), _RING8_RADIUS * ang.sin()), dim=1)
    vang = (2.0 * math.pi / float(_RING8_MODES)) * (idx + 0.5)
    valleys = torch.stack((_RING8_RADIUS * vang.cos(), _RING8_RADIUS * vang.sin()), dim=1)
    return centers, valleys


def _highd_manifold(key: str) -> dict[str, torch.Tensor]:
    cached = _MANIFOLD_CACHE.get(key)
    if cached is not None:
        return cached
    if key == "resbn":
        dim, n_modes, seed = _RESBN_DIM, _RESBN_MODES, 420128
        g = torch.Generator(device="cpu").manual_seed(seed)
        centers = torch.randn((n_modes, dim), generator=g, dtype=torch.float32) * 0.4
    elif key == "highd_sparse":
        dim, n_modes, seed = _HIGHD_DIM, _HIGHD_MODES, 440512
        g = torch.Generator(device="cpu").manual_seed(seed)
        centers = torch.tanh(torch.randn((n_modes, dim), generator=g, dtype=torch.float32) * 0.8)
    else:
        raise ValueError(f"unknown high-D manifold key: {key!r}")

    centered = centers - centers.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered.double(), full_matrices=False)
    pca_basis = vh[:2].T.float().contiguous()
    g_proj = torch.Generator(device="cpu").manual_seed(seed + 77)
    proj = torch.randn((dim, 128), generator=g_proj, dtype=torch.float32)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-12)
    payload = {
        "centers": centers,
        "mean": centers.mean(dim=0, keepdim=True),
        "pca_basis": pca_basis,
        "sw_proj": proj,
    }
    _MANIFOLD_CACHE[key] = payload
    return payload


def _shock_2d_proj() -> torch.Tensor:
    cached = _MANIFOLD_CACHE.get("shock_2d_proj")
    if cached is not None:
        return cached["sw_proj"]
    angles = torch.linspace(0.0, math.pi, 32, dtype=torch.float32)
    proj = torch.stack((angles.cos(), angles.sin()), dim=0)
    _MANIFOLD_CACHE["shock_2d_proj"] = {"sw_proj": proj}
    return proj


# ---------------------------------------------------------------------------
# Public Specifications & Strictly Problem-Only Generated-Output Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, list[list[Any]]] = {
    "fl_law41_trust_rare1pct_tanh": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 5],
        ["rare_mode_recall", ">=", 0.45],
        ["precision", ">=", 0.85],
        ["mass_tv", "<=", 0.10],
    ],
    "fl_law42_swap_deep_critic_16l": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 8],
        ["hq", ">=", 0.85],
        ["mass_tv", "<=", 0.12],
        ["valley_fraction", "<=", 0.05],
    ],
    "fl_law42_deep_resbn_and_conv1x1": [
        ["finite_fraction", "==", 1.0],
        ["sw1", "<=", 0.10],
        ["var_ratio_err", "<=", 0.25],
        ["conv1x1_sw1", "<=", 0.10],
        ["conv1x1_sw1_ratio", "<=", 1.05],
    ],
    "fl_law44_highd_sparse_prior_and_shock": [
        ["finite_fraction", "==", 1.0],
        ["sw1", "<=", 0.14],
        ["mode_recall", ">=", 0.75],
        ["var_ratio_err", "<=", 0.25],
        ["shock_modes", "==", 4],
        ["shock_hq", ">=", 0.65],
        ["shock_sw1", "<=", 0.35],
    ],
}

SPECS: dict[str, dict[str, Any]] = {
    "fl_law41_trust_rare1pct_tanh": {
        "task_id": "fl_law41_trust_rare1pct_tanh",
        "case_id": "fl_law41_trust_rare1pct_tanh",
        "kind": "fl_law41_trust_rare1pct_tanh",
        "law": "law4.1",
        "defects": ["D-04"],
        "title": "Law 4.1/4.3: 1% minority mode under saturating tanh generator",
        "goal": (
            "Learn a 2D 5-mode mixture with 4 major modes (99% mass) and 1 separated 1% minority mode "
            "near the saturating 3*tanh rail (5/5 modes, rare_mode_recall >= 0.45, precision >= 0.85, mass_tv <= 0.10)."
        ),
        "target": (
            "2D 5-mode Gaussian mixture: 4 major modes at (+-1.5, +-1.5) (24.75% each) and "
            "1 separated 1% minority mode at (0.0, 2.85), sigma=0.08, under saturating 3.0*tanh generator"
        ),
        "resources": {"num_particles": 512, "z_dim": 4, "batch_size": 256},
        "prior": {"kind": "mog", "sigma": 0.08, "standardize": False},
        "model": {
            "generator_family": "saturating_tanh_mlp",
            "g_hidden": 128,
            "g_layers": 4,
            "out_dim": 2,
            "out_bound": _RARE1PCT_OUT_BOUND,
            "discriminator_family": "simple_mlp",
            "d_hidden": 128,
            "d_layers": 3,
            "fourier": 2,
        },
        "steps": 800,
        "eval_samples": 8192,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law42_swap_deep_critic_16l": {
        "task_id": "fl_law42_swap_deep_critic_16l",
        "case_id": "fl_law42_swap_deep_critic_16l",
        "kind": "fl_law42_swap_deep_critic_16l",
        "law": "law4.2",
        "defects": ["D-10"],
        "title": "Law 4.2: Shallow 3L G + deep 16L skip-free critic on 8-mode 2D ring",
        "goal": (
            "Learn an 8-mode 2D Gaussian ring (8/8 modes, hq >= 0.85, mass_tv <= 0.12, valley_fraction <= 0.05) "
            "when depth asymmetry is inverted onto a deep 16L skip-free LeakyReLU critic."
        ),
        "target": (
            "8-mode equal-mass 2D Gaussian ring (R=2.5, sigma=0.08) trained with shallow 3L MLP generator "
            "and deep 16L skip-free LeakyReLU critic (fourier=0)"
        ),
        "resources": {"num_particles": 256, "z_dim": 4, "batch_size": 256},
        "prior": {"kind": "mog", "sigma": 0.08, "standardize": False},
        "model": {
            "generator_family": "simple_mlp",
            "g_hidden": 64,
            "g_layers": 3,
            "out_dim": 2,
            "discriminator_family": "deep_skipfree_mlp",
            "d_hidden": 64,
            "d_layers": 16,
            "fourier": 0,
        },
        "steps": 800,
        "eval_samples": 4096,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law42_deep_resbn_and_conv1x1": {
        "task_id": "fl_law42_deep_resbn_and_conv1x1",
        "case_id": "fl_law42_deep_resbn_and_conv1x1",
        "kind": "fl_law42_deep_resbn_and_conv1x1",
        "law": "law4.2",
        "defects": ["Bug2", "D-10", "4D-1x1-vs-2D-Linear"],
        "title": "Law 4.2/4.5: Deep ResBN d=8 & isomorphic 4D Conv2d 1x1 output parity",
        "goal": (
            "Learn a 64-mode R^128 Gaussian mixture on an 8-block ResBN trunk (sw1 <= 0.10, "
            "var_ratio_err <= 0.25) with matching generated output quality on its isomorphic 4D Conv2d 1x1 twin "
            "(conv1x1_sw1 <= 0.10, conv1x1_sw1_ratio <= 1.05)."
        ),
        "target": (
            "64-mode R^128 Gaussian mixture (sigma=0.08) on paired 8-block ResBN trunks: "
            "2D Linear primary vs mathematically isomorphic 4D Conv2d 1x1 twin"
        ),
        "resources": {"num_particles": 512, "z_dim": 32, "batch_size": 256},
        "prior": {"kind": "mog", "sigma": 0.10, "standardize": False},
        "model": {
            "generator_family": "resbn_linear_and_conv1x1_twin",
            "discriminator_family": "resbn_linear_and_conv1x1_twin",
            "dim": _RESBN_DIM,
            "width": _RESBN_WIDTH,
            "depth": _RESBN_DEPTH,
        },
        "steps": 500,
        "eval_samples": 2048,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law44_highd_sparse_prior_and_shock": {
        "task_id": "fl_law44_highd_sparse_prior_and_shock",
        "case_id": "fl_law44_highd_sparse_prior_and_shock",
        "kind": "fl_law44_highd_sparse_prior_and_shock",
        "law": "law4.4",
        "defects": ["Bug1", "D-03", "D-05"],
        "title": "Law 4.4: High-D sparse manifold & extreme 1e-5x scale-shock recovery",
        "goal": (
            "Learn a 64-mode 10%-masked R^512 bounded manifold (sw1 <= 0.14, mode_recall >= 0.75, "
            "var_ratio_err <= 0.25) AND recover generated-sample quality on a 1e-5x-scaled 4-mode target "
            "after a transient 1e4x mid-run scale spike and mode shift (shock_modes == 4, shock_hq >= 0.65, shock_sw1 <= 0.35)."
        ),
        "target": (
            "64-mode bounded R^512 tanh manifold (P=1600, 10% observation mask, RMSNorm critic) "
            "plus paired 1e-5x-scaled 4-mode GAN with transient 1e4x mid-run scale shock and shift"
        ),
        "resources": {"num_particles": 1600, "z_dim": 32, "batch_size": 256},
        "prior": {"kind": "mog", "sigma": 0.10, "standardize": False},
        "model": {
            "generator_family": "highd_sparse_tanh_mlp",
            "discriminator_family": "highd_rmsnorm_mlp",
            "dim": _HIGHD_DIM,
            "hidden": _HIGHD_HIDDEN,
            "obs_prob": _HIGHD_OBS_PROB,
            "shock_step": _HIGHD_SHOCK_STEP,
            "shock_base_scale": _SHOCK_BASE_SCALE,
            "shock_spike_scale": _SHOCK_SPIKE_SCALE,
        },
        "steps": 400,
        "eval_samples": 2048,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
}

HOST_DEFINITIONS: dict[str, dict[str, Any]] = SPECS

# ---------------------------------------------------------------------------
# Calibrated v2 Law 4 Specifications & Thresholds (Preserving Frozen v1)
# ---------------------------------------------------------------------------

_V2_SHOCK_STEP: int = 300

V2_THRESHOLDS: dict[str, list[list[Any]]] = {
    "fl_law42_deep_resbn_and_conv1x1_v2": [
        ["finite_fraction", "==", 1.0],
        ["sw1", "<=", 0.10],
        ["var_ratio_err", "<=", 0.25],
        ["conv1x1_sw1", "<=", 0.10],
        ["conv1x1_sw1_abs_diff", "<=", 0.015],
    ],
    "fl_law44_scale_shock": [
        ["finite_fraction", "==", 1.0],
        ["shock_modes", "==", 4],
        ["shock_hq", ">=", 0.65],
        ["shock_sw1", "<=", 0.35],
        ["shock_mass_tv", "<=", 0.15],
    ],
}

V2_SPECS: dict[str, dict[str, Any]] = {
    "fl_law42_deep_resbn_and_conv1x1_v2": {
        "task_id": "fl_law42_deep_resbn_and_conv1x1_v2",
        "case_id": "fl_law42_deep_resbn_and_conv1x1_v2",
        "kind": "fl_law42_deep_resbn_and_conv1x1_v2",
        "law": "law4.2",
        "defects": ["Bug2", "D-10", "4D-1x1-vs-2D-Linear"],
        "title": "Law 4.2/4.5 (v2): Deep ResBN d=8 & isomorphic 4D Conv2d 1x1 absolute parity",
        "goal": (
            "Learn a 64-mode R^128 Gaussian mixture on an 8-block ResBN trunk (sw1 <= 0.10, "
            "var_ratio_err <= 0.25) with matching generated output quality on its isomorphic 4D Conv2d 1x1 twin "
            "(conv1x1_sw1 <= 0.10, abs(conv1x1_sw1 - sw1) <= 0.015)."
        ),
        "target": (
            "64-mode R^128 Gaussian mixture (sigma=0.08) on paired 8-block ResBN trunks: "
            "2D Linear primary vs mathematically isomorphic 4D Conv2d 1x1 twin"
        ),
        "resources": {"num_particles": 512, "z_dim": 32, "batch_size": 256},
        "prior": {"kind": "mog", "sigma": 0.10, "standardize": False},
        "model": {
            "generator_family": "resbn_linear_and_conv1x1_twin",
            "discriminator_family": "resbn_linear_and_conv1x1_twin",
            "dim": _RESBN_DIM,
            "width": _RESBN_WIDTH,
            "depth": _RESBN_DEPTH,
        },
        "steps": 500,
        "eval_samples": 2048,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
    "fl_law44_scale_shock": {
        "task_id": "fl_law44_scale_shock",
        "case_id": "fl_law44_scale_shock",
        "kind": "fl_law44_scale_shock",
        "law": "law4.4",
        "defects": ["D-03", "D-05", "Law2x4.4-scale-equivariance"],
        "title": "Law 4.4 (v2): Standalone extreme 1e-5x scale & 1e4x shock + 45-deg rotation recovery",
        "goal": (
            "Recover all 4 modes on a standalone 1e-5x-scaled 2D mixture after a mid-run step-300 "
            "transient 1e4x scale spike and 45-degree rotation with >=500 post-shock recovery steps "
            "(shock_modes == 4, shock_hq >= 0.65, shock_sw1 <= 0.35, shock_mass_tv <= 0.15)."
        ),
        "target": (
            "Standalone 1e-5x-scaled 2D 4-mode Gaussian mixture (sigma=0.08*1e-5) with transient "
            "1e4x scale spike at steps 300..301 and 45-degree mode rotation for steps 302..1000"
        ),
        "resources": {"num_particles": 128, "z_dim": 4, "batch_size": 128},
        "prior": {"kind": "mog", "sigma": 0.08, "standardize": False},
        "model": {
            "generator_family": "scaled_output_mlp",
            "discriminator_family": "simple_mlp",
            "hidden": 64,
            "layers": 2,
            "out_dim": 2,
            "fourier": 0,
            "shock_step": _V2_SHOCK_STEP,
            "shock_base_scale": _SHOCK_BASE_SCALE,
            "shock_spike_scale": _SHOCK_SPIKE_SCALE,
        },
        "steps": 1000,
        "eval_samples": 2048,
        "observations": 24,
        "minimum_stable_checks": 5,
        "sampling_law": "public_prior_without_output_noise",
        "view_kind": "vector",
    },
}

ALL_THRESHOLDS: dict[str, list[list[Any]]] = {**THRESHOLDS, **V2_THRESHOLDS}
ALL_SPECS: dict[str, dict[str, Any]] = {**SPECS, **V2_SPECS}


# ---------------------------------------------------------------------------
# Signed Normalized Margins & Threshold Checking
# ---------------------------------------------------------------------------

def signed_margins(metrics: dict[str, Any], task_id: str) -> dict[str, float]:
    """Compute signed normalized margin-to-threshold for every bound of ``task_id``.

    Positive (``>= 0.0``) iff the bound passes; strictly negative (``< 0.0``) iff it fails.
    Missing or non-finite values always return finite ``-10.0``.
    """
    if task_id not in ALL_THRESHOLDS:
        raise ValueError(f"unsupported four_laws_4 task_id: {task_id!r}")
    margins: dict[str, float] = {}
    for name, op, bound in ALL_THRESHOLDS[task_id]:
        b = float(bound)
        scale = max(abs(b), 1e-6) if b != 0.0 else 1.0
        val = metrics.get(name)
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            margins[name] = -10.0
            continue
        v = float(val)
        if op == ">=":
            margin = (v - b) / scale
        elif op == "<=":
            margin = (b - v) / scale
        elif op == "==":
            margin = 0.0 if v == b else -abs(v - b) / scale
        else:
            raise ValueError(f"unsupported threshold operator: {op!r}")
        margins[name] = float(margin) if math.isfinite(margin) else -10.0
    return margins


def _check_thresholds(metrics: dict[str, Any], task_id: str) -> list[str]:
    failed: list[str] = []
    for name, op, bound in ALL_THRESHOLDS[task_id]:
        val = metrics.get(name)
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            failed.append(f"{name} {op} {bound}")
            continue
        v = float(val)
        b = float(bound)
        ok = v <= b if op == "<=" else (v >= b if op == ">=" else (v == b if op == "==" else False))
        if not ok:
            failed.append(f"{name} {op} {bound}")
    return failed


def _finalize_observation(
    task_id: str,
    metrics: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_metrics: dict[str, float | int] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            v = v.detach().cpu().item()
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, bool):
            clean_metrics[k] = int(v)
        elif isinstance(v, int):
            clean_metrics[k] = v
        elif isinstance(v, float) and math.isfinite(v):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = -10.0 if k != "finite_fraction" else 0.0
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


# ---------------------------------------------------------------------------
# Task Architectures (Strictly Standard Registered PyTorch Layers)
# ---------------------------------------------------------------------------

class SaturatingTanhGenerator(nn.Module):
    """MLP generator bounded by ``out_bound * tanh(raw)`` placing rare modes near saturation."""

    def __init__(
        self,
        *,
        z_dim: int,
        hidden: int,
        layers: int,
        out_dim: int,
        out_bound: float = _RARE1PCT_OUT_BOUND,
    ) -> None:
        super().__init__()
        self.backbone = SimpleMLPGenerator(
            z_dim=int(z_dim),
            hidden_dim=int(hidden),
            n_hidden=int(layers),
            out_dim=int(out_dim),
        )
        self.out_bound = float(out_bound)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out_bound * torch.tanh(self.backbone(z))


class ScaledOutputGenerator(nn.Module):
    """SimpleMLPGenerator with a fixed scalar output scale (no extra trainable parameters)."""

    def __init__(
        self,
        *,
        z_dim: int,
        hidden: int,
        layers: int,
        out_dim: int,
        output_scale: float = _SHOCK_BASE_SCALE,
    ) -> None:
        super().__init__()
        self.backbone = SimpleMLPGenerator(
            z_dim=int(z_dim),
            hidden_dim=int(hidden),
            n_hidden=int(layers),
            out_dim=int(out_dim),
        )
        self.output_scale = float(output_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.backbone(z) * self.output_scale


class ResBNLinearBlock(nn.Module):
    """Branch-RMSNorm residual block using 2D Linear weights."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(width), int(width))
        self.fc2 = nn.Linear(int(width), int(width))
        self.scale = 0.5 / float(max(1, int(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.rms_norm(x, (x.shape[-1],))
        h = self.fc2(F.silu(self.fc1(h)))
        h = F.rms_norm(h, (h.shape[-1],))
        return x + self.scale * h


class ResBNLinearNet(nn.Module):
    """Deep branch-RMSNorm residual MLP (2D Linear primary trunk)."""

    def __init__(self, d_in: int, d_out: int, width: int, depth: int) -> None:
        super().__init__()
        self.stem = nn.Linear(int(d_in), int(width))
        self.blocks = nn.ModuleList([ResBNLinearBlock(width, depth) for _ in range(int(depth))])
        self.head = nn.Linear(int(width), int(d_out))
        self._squeeze_scalar = int(d_out) == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        return out.squeeze(-1) if self._squeeze_scalar else out


class ResBNConv1x1Block(nn.Module):
    """Branch-RMSNorm residual block using mathematically isomorphic 4D Conv2d 1x1 weights."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(int(width), int(width), kernel_size=1)
        self.conv2 = nn.Conv2d(int(width), int(width), kernel_size=1)
        self.scale = 0.5 / float(max(1, int(depth)))

    @staticmethod
    def _rms_norm_c(x: torch.Tensor) -> torch.Tensor:
        flat = x.squeeze(-1).squeeze(-1)
        return F.rms_norm(flat, (flat.shape[-1],)).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._rms_norm_c(x)
        h = self.conv2(F.silu(self.conv1(h)))
        h = self._rms_norm_c(h)
        return x + self.scale * h


class ResBNConv1x1Net(nn.Module):
    """Deep branch-RMSNorm residual trunk using 4D Conv2d 1x1 weights (isomorphic twin of ResBNLinearNet).

    Generator (``d_out > 1``) outputs native 4D ``(B, d_out, 1, 1)`` tensors; critic (``d_out == 1``)
    accepts native 4D ``(B, d_in, 1, 1)`` (or 2D ``(B, d_in)``) tensors so CUDA ``channels_last``
    autocast executes seamlessly without rank errors.
    """

    def __init__(self, d_in: int, d_out: int, width: int, depth: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(int(d_in), int(width), kernel_size=1)
        self.blocks = nn.ModuleList([ResBNConv1x1Block(width, depth) for _ in range(int(depth))])
        self.head = nn.Conv2d(int(width), int(d_out), kernel_size=1)
        self._squeeze_scalar = int(d_out) == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x4 = x if x.ndim == 4 else x.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.stem(x4))
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        return out.squeeze(-1).squeeze(-1).squeeze(-1) if self._squeeze_scalar else out


@torch.no_grad()
def copy_resbn_linear_to_conv1x1(src: ResBNLinearNet, dst: ResBNConv1x1Net) -> None:
    """Copy exact initial Linear weights/biases into the isomorphic 4D Conv2d 1x1 twin."""
    dst.stem.weight.copy_(src.stem.weight.reshape_as(dst.stem.weight))
    if src.stem.bias is not None and dst.stem.bias is not None:
        dst.stem.bias.copy_(src.stem.bias)
    for b_src, b_dst in zip(src.blocks, dst.blocks):
        b_dst.conv1.weight.copy_(b_src.fc1.weight.reshape_as(b_dst.conv1.weight))
        b_dst.conv2.weight.copy_(b_src.fc2.weight.reshape_as(b_dst.conv2.weight))
        if b_src.fc1.bias is not None and b_dst.conv1.bias is not None:
            b_dst.conv1.bias.copy_(b_src.fc1.bias)
        if b_src.fc2.bias is not None and b_dst.conv2.bias is not None:
            b_dst.conv2.bias.copy_(b_src.fc2.bias)
    dst.head.weight.copy_(src.head.weight.reshape_as(dst.head.weight))
    if src.head.bias is not None and dst.head.bias is not None:
        dst.head.bias.copy_(src.head.bias)


class HighDSparseTanhGenerator(nn.Module):
    """High-D bounded tanh generator applying an isolated 10% observation mask during training only."""

    def __init__(
        self,
        *,
        z_dim: int,
        hidden: int,
        out_dim: int,
        obs_prob: float = _HIGHD_OBS_PROB,
        mask_seed: int = 0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(out_dim)),
        )
        self.out_dim = int(out_dim)
        self.obs_prob = float(obs_prob)
        self._mask_seed = int(mask_seed)
        self._mask_rng: torch.Generator | None = None

    def _get_mask_rng(self, device: torch.device) -> torch.Generator:
        if self._mask_rng is None or self._mask_rng.device != device:
            self._mask_rng = torch.Generator(device=device).manual_seed(self._mask_seed)
        return self._mask_rng

    def mask_rng_state(self) -> torch.Tensor | None:
        return None if self._mask_rng is None else self._mask_rng.get_state().clone()

    def load_mask_rng_state(self, state: torch.Tensor | None, device: torch.device) -> None:
        if state is None:
            self._mask_rng = None
            return
        gen = self._get_mask_rng(device)
        gen.set_state(state.cpu() if device.type == "cpu" else state.to(device))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.net(z))
        if self.training:
            rng = self._get_mask_rng(z.device)
            mask = (torch.rand((z.shape[0], self.out_dim), generator=rng, device=z.device) < self.obs_prob).to(out.dtype)
            return out * mask
        return out


class HighDRMSNormCritic(nn.Module):
    """High-D RMSNorm-input MLP critic matching the Bug 1 manifold."""

    def __init__(self, *, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(F.rms_norm(x, (x.shape[-1],))).squeeze(-1)


def build_models_for_spec(spec: dict[str, Any], *, z_dim: int | None = None, seed: int = DEFAULT_SEED) -> tuple[nn.Module, nn.Module]:
    """Build primary ``(G, D)`` models for a Law 4 task specification."""
    kind = spec["kind"]
    cfg = spec["model"]
    z_eff = int(spec["resources"]["z_dim"] if z_dim is None else z_dim)
    if kind == "fl_law41_trust_rare1pct_tanh":
        g_mod = SaturatingTanhGenerator(
            z_dim=z_eff,
            hidden=int(cfg["g_hidden"]),
            layers=int(cfg["g_layers"]),
            out_dim=int(cfg["out_dim"]),
            out_bound=float(cfg["out_bound"]),
        )
        d_mod = SimpleMLPDiscriminator(
            in_dim=int(cfg["out_dim"]),
            hidden_dim=int(cfg["d_hidden"]),
            n_hidden=int(cfg["d_layers"]),
            fourier=int(cfg["fourier"]),
        )
        return g_mod, d_mod
    if kind == "fl_law42_swap_deep_critic_16l":
        g_mod = SimpleMLPGenerator(
            z_dim=z_eff,
            hidden_dim=int(cfg["g_hidden"]),
            n_hidden=int(cfg["g_layers"]),
            out_dim=int(cfg["out_dim"]),
        )
        d_mod = SimpleMLPDiscriminator(
            in_dim=int(cfg["out_dim"]),
            hidden_dim=int(cfg["d_hidden"]),
            n_hidden=int(cfg["d_layers"]),
            fourier=int(cfg["fourier"]),
        )
        return g_mod, d_mod
    if kind in ("fl_law42_deep_resbn_and_conv1x1", "fl_law42_deep_resbn_and_conv1x1_v2"):
        dim = int(cfg["dim"])
        w = int(cfg["width"])
        depth = int(cfg["depth"])
        return ResBNLinearNet(z_eff, dim, w, depth), ResBNLinearNet(dim, 1, w, depth)
    if kind == "fl_law44_scale_shock":
        g_mod = ScaledOutputGenerator(
            z_dim=z_eff,
            hidden=int(cfg["hidden"]),
            layers=int(cfg["layers"]),
            out_dim=int(cfg["out_dim"]),
            output_scale=float(cfg["shock_base_scale"]),
        )
        d_mod = SimpleMLPDiscriminator(
            in_dim=int(cfg["out_dim"]),
            hidden_dim=int(cfg["hidden"]),
            n_hidden=int(cfg["layers"]),
            fourier=int(cfg.get("fourier", 0)),
        )
        return g_mod, d_mod
    if kind == "fl_law44_highd_sparse_prior_and_shock":
        dim = int(cfg["dim"])
        h = int(cfg["hidden"])
        g_mod = HighDSparseTanhGenerator(
            z_dim=z_eff,
            hidden=h,
            out_dim=dim,
            obs_prob=float(cfg["obs_prob"]),
            mask_seed=_seed_int("highd_gen_mask", seed),
        )
        d_mod = HighDRMSNormCritic(in_dim=dim, hidden=h)
        return g_mod, d_mod
    raise ValueError(f"unsupported four_laws_4 task kind: {kind!r}")


# ---------------------------------------------------------------------------
# Target Samplers & Mathematical Observables
# ---------------------------------------------------------------------------

def sample_shock_target(
    n: int,
    generator: torch.Generator,
    *,
    step: int,
    shock_step: int = _HIGHD_SHOCK_STEP,
    device: str | torch.device = "cpu",
    eval_phase: bool = False,
) -> torch.Tensor:
    """Draw 2D 4-mode extreme-scale / transient-spike target samples for Law 4.4 shock twin."""
    dev = torch.device(device)
    gdev = generator.device
    count = int(n)
    if (not eval_phase) and shock_step <= int(step) <= shock_step + 1:
        scale = _SHOCK_SPIKE_SCALE
        raw_centers = _SHOCK_PRE_CENTERS
    elif int(step) > shock_step:
        scale = _SHOCK_BASE_SCALE
        raw_centers = _SHOCK_POST_CENTERS
    else:
        scale = _SHOCK_BASE_SCALE
        raw_centers = _SHOCK_PRE_CENTERS
    centers = torch.tensor(raw_centers, dtype=torch.float32, device=gdev) * float(scale)
    idx = torch.randint(0, len(raw_centers), (count,), generator=generator, device=gdev)
    pts = centers[idx] + (float(scale) * _SHOCK_SIGMA_UNIT) * torch.randn((count, 2), generator=generator, device=gdev)
    return pts.to(dev)


def sample_target_for_kind(
    kind: str,
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
    apply_mask: bool = False,
) -> torch.Tensor:
    """Draw exact i.i.d. target samples for any Law 4 task using only ``generator``."""
    dev = torch.device(device)
    gdev = generator.device
    count = int(n)
    if kind == "fl_law41_trust_rare1pct_tanh":
        weights = torch.tensor(_RARE1PCT_WEIGHTS, dtype=torch.float32, device=gdev)
        centers = torch.tensor(_RARE1PCT_CENTERS, dtype=torch.float32, device=gdev)
        idx = torch.multinomial(weights, count, replacement=True, generator=generator)
        pts = centers[idx] + _RARE1PCT_SIGMA * torch.randn((count, 2), generator=generator, device=gdev)
        return pts.to(dev)
    if kind == "fl_law42_swap_deep_critic_16l":
        centers, _ = _ring8_centers_and_valleys(gdev)
        idx = torch.randint(0, _RING8_MODES, (count,), generator=generator, device=gdev)
        pts = centers[idx] + _RING8_SIGMA * torch.randn((count, 2), generator=generator, device=gdev)
        return pts.to(dev)
    if kind in ("fl_law42_deep_resbn_and_conv1x1", "fl_law42_deep_resbn_and_conv1x1_v2"):
        mf = _highd_manifold("resbn")
        centers = mf["centers"].to(gdev)
        idx = torch.randint(0, _RESBN_MODES, (count,), generator=generator, device=gdev)
        pts = centers[idx] + _RESBN_SIGMA * torch.randn((count, _RESBN_DIM), generator=generator, device=gdev)
        return pts.to(dev)
    if kind == "fl_law44_scale_shock":
        return sample_shock_target(
            count,
            generator,
            step=_V2_SHOCK_STEP + 1,
            shock_step=_V2_SHOCK_STEP,
            device=dev,
            eval_phase=True,
        )
    if kind == "fl_law44_highd_sparse_prior_and_shock":
        mf = _highd_manifold("highd_sparse")
        centers = mf["centers"].to(gdev)
        idx = torch.randint(0, _HIGHD_MODES, (count,), generator=generator, device=gdev)
        pts = torch.clamp(
            centers[idx] + _HIGHD_SIGMA * torch.randn((count, _HIGHD_DIM), generator=generator, device=gdev),
            -1.0,
            1.0,
        )
        if apply_mask:
            mask = (torch.rand((count, _HIGHD_DIM), generator=generator, device=gdev) < _HIGHD_OBS_PROB).float()
            pts = pts * mask
        return pts.to(dev)
    raise ValueError(f"unsupported four_laws_4 task kind: {kind!r}")


@torch.no_grad()
def sliced_wasserstein_1(
    samples: torch.Tensor,
    reference: torch.Tensor,
    proj: torch.Tensor,
) -> float:
    """Deterministic sliced Wasserstein-1 distance along fixed unit directions ``proj``."""
    x = samples.detach().float().cpu()
    y = reference.detach().float().cpu()
    p = proj.detach().float().cpu()
    n = min(len(x), len(y))
    if n < 2 or x.shape[1] != p.shape[0] or y.shape[1] != p.shape[0]:
        return float("nan")
    px, _ = torch.sort(x[:n] @ p, dim=0)
    py, _ = torch.sort(y[:n] @ p, dim=0)
    val = float((px - py).abs().mean().item())
    return val if math.isfinite(val) else float("nan")


@torch.no_grad()
def _project_pca_2d(points: torch.Tensor, manifold: dict[str, torch.Tensor]) -> torch.Tensor:
    pts = points.detach().float().cpu()
    return (pts - manifold["mean"]) @ manifold["pca_basis"]


@torch.no_grad()
def _matrix_stable_rank(mat: torch.Tensor) -> float:
    x = mat.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not bool(torch.isfinite(x).all()):
        return 0.0
    sv = torch.linalg.svdvals(x)
    s1 = float(sv[0].item())
    if s1 <= 1e-30:
        return 0.0
    frob_sq = float(sv.square().sum().item())
    return float(frob_sq / (s1 * s1))


def measure_deep_critic_diagnostics(
    critic: nn.Module,
    *,
    device: torch.device,
) -> tuple[float, float]:
    """Non-gating diagnostic telemetry ``(valley_sep, critic_r_stable)`` on deterministic probes."""
    modes_state = [(m, m.training) for m in critic.modules()]
    try:
        critic.eval()
        g_ev = torch.Generator(device="cpu").manual_seed(421600)
        centers, valleys = _ring8_centers_and_valleys("cpu")
        n_probe = 512
        idx_r = torch.randint(0, _RING8_MODES, (n_probe,), generator=g_ev)
        x_real = (centers[idx_r] + _RING8_SIGMA * torch.randn((n_probe, 2), generator=g_ev)).to(device)
        idx_v = torch.randint(0, _RING8_MODES, (n_probe,), generator=g_ev)
        x_val = (valleys[idx_v] + _RING8_SIGMA * torch.randn((n_probe, 2), generator=g_ev)).to(device)
        with torch.no_grad():
            dr = critic(x_real).reshape(-1)
            dv = critic(x_val).reshape(-1)
            if not (bool(torch.isfinite(dr).all()) and bool(torch.isfinite(dv).all())):
                return 0.0, 0.0
            valley_sep = float((dr > dv).float().mean().item())

        hidden_params = [
            p for p in critic.parameters() if p.requires_grad and p.ndim == 2 and min(p.shape) >= 16
        ]
        if not hidden_params:
            return valley_sep, 0.0
        with torch.enable_grad():
            loss_probe = F.softplus(-(critic(x_real[:128]).reshape(-1) - critic(x_val[:128]).reshape(-1))).mean()
            grads = torch.autograd.grad(loss_probe, hidden_params, create_graph=False, allow_unused=True)
        sranks = sorted(_matrix_stable_rank(g) for g in grads if g is not None)
        med_srk = float(sranks[len(sranks) // 2]) if sranks else 0.0
        return valley_sep, med_srk
    finally:
        for m, flag in modes_state:
            m.training = flag


@torch.no_grad()
def measure_critic_max_s1(critic: nn.Module) -> float:
    """Non-gating diagnostic telemetry: max top singular value across hidden 2D/4D-1x1 critic weights."""
    vals: list[float] = []
    for p in critic.parameters():
        if p.ndim == 2 and min(p.shape) >= 16:
            mat = p.detach().float().cpu()
        elif p.ndim == 4 and p.shape[2] == 1 and p.shape[3] == 1 and min(p.shape[0], p.shape[1]) >= 16:
            mat = p.detach().float().cpu().reshape(p.shape[0], p.shape[1])
        else:
            continue
        if not bool(torch.isfinite(mat).all()):
            return 0.0
        vals.append(float(torch.linalg.matrix_norm(mat, ord=2).item()))
    return max(vals) if vals else 0.0


def _curve_view(
    title: str,
    steps: list[int],
    observed: list[float],
    target_level: float,
    *,
    ylabel: str,
    caption: str,
    ylim: list[float] | None = None,
) -> dict[str, Any]:
    xs = np.asarray(steps if steps else [0, 1], dtype=np.float32)
    ys = np.asarray(observed if observed else [0.0, 0.0], dtype=np.float32)
    if len(xs) == 1:
        xs = np.asarray([xs[0], xs[0] + 1.0], dtype=np.float32)
        ys = np.asarray([ys[0], ys[0]], dtype=np.float32)
    t_arr = np.column_stack((xs, np.full_like(xs, float(target_level))))
    s_arr = np.column_stack((xs, ys))
    view: dict[str, Any] = {
        "kind": "line",
        "title": title,
        "target": t_arr,
        "samples": s_arr,
        "xlabel": "training update",
        "ylabel": ylabel,
        "aspect": "auto",
        "caption": caption,
    }
    if ylim is not None:
        view["ylim"] = ylim
    return view


# ---------------------------------------------------------------------------
# Per-Task Scorers (Strictly Generated-Output Gates + Signed Margins)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_fl_law41_trust_rare1pct_tanh(
    samples: torch.Tensor,
    *,
    realized_step_max: float = 0.0,
    ood_disp_ratio: float = 1.0,
    reference: torch.Tensor | None = None,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score ``fl_law41_trust_rare1pct_tanh`` purely on generated sample distribution quality."""
    task_id = "fl_law41_trust_rare1pct_tanh"
    pts = samples.detach().cpu().to(dtype=torch.float32)
    ref = reference
    if ref is None:
        g_ref = torch.Generator(device="cpu").manual_seed(410001)
        ref = sample_target_for_kind(task_id, max(2048, min(4096, len(pts) if pts.ndim >= 1 else 2048)), g_ref)
    ref_cpu = ref.detach().cpu().to(dtype=torch.float32)
    centers = torch.tensor(_RARE1PCT_CENTERS, dtype=torch.float32)
    weights = torch.tensor(_RARE1PCT_WEIGHTS, dtype=torch.float32)
    minor_ref = centers[4:5] + _RARE1PCT_SIGMA * torch.randn(
        (256, 2), generator=torch.Generator(device="cpu").manual_seed(410002)
    )

    valid_shape = pts.ndim == 2 and pts.shape[1] == 2 and len(pts) > 0
    finite_fraction = float(torch.isfinite(pts).all(dim=-1).float().mean().item()) if valid_shape else 0.0
    if not valid_shape or finite_fraction < 1.0:
        fallback_metrics = {
            "finite_fraction": finite_fraction,
            "modes": 0,
            "rare_mode_recall": 0.0,
            "precision": 0.0,
            "mass_tv": 1.0,
            "diag_realized_step_max": 0.0,
            "diag_ood_disp_ratio": 0.0,
        }
        zeros = torch.zeros((4, 2), dtype=torch.float32)
        views = [
            {
                "kind": "scatter",
                "title": "Full 5-mode mixture (4 major + 1% minority)",
                "target": ref_cpu[:512],
                "samples": zeros,
                "xlim": [-3.4, 3.4],
                "ylim": [-3.4, 3.4],
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Invalid/nonfinite output fallback.",
            },
            {
                "kind": "scatter",
                "title": "1% minority mode close-up at (0.0, 2.85)",
                "target": minor_ref[:128],
                "samples": zeros,
                "xlim": [-0.8, 0.8],
                "ylim": [2.1, 3.4],
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Invalid/nonfinite output fallback.",
            },
        ]
        return _finalize_observation(task_id, fallback_metrics, views)

    dists = torch.cdist(pts, centers)
    min_d, nearest = dists.min(dim=1)
    hq_mask = min_d <= (3.5 * _RARE1PCT_SIGMA)
    precision = float(hq_mask.float().mean().item())

    n_total = float(len(pts))
    hq_counts = torch.bincount(nearest[hq_mask], minlength=5).float()
    hq_fracs = hq_counts / n_total
    rare_mode_recall = float(hq_fracs[4].item() / _RARE1PCT_WEIGHTS[4])
    modes = int(sum(1 for k in range(5) if float(hq_fracs[k].item()) >= 0.40 * _RARE1PCT_WEIGHTS[k]))

    emp_mass = torch.bincount(nearest, minlength=5).float() / n_total
    mass_tv = float(0.5 * (emp_mass - weights).abs().sum().item())

    metrics = {
        "finite_fraction": 1.0,
        "modes": modes,
        "rare_mode_recall": rare_mode_recall,
        "precision": precision,
        "mass_tv": mass_tv,
        # Non-gating diagnostic telemetry only:
        "diag_realized_step_max": float(realized_step_max) if math.isfinite(float(realized_step_max)) else 0.0,
        "diag_ood_disp_ratio": float(ood_disp_ratio) if math.isfinite(float(ood_disp_ratio)) else 0.0,
    }

    minor_samples = pts[(pts[:, 0].abs() <= 0.85) & (pts[:, 1] >= 2.05) & (pts[:, 1] <= 3.45)]
    if len(minor_samples) == 0:
        minor_samples = pts[:4]
    hist = history or {}
    curve_steps = [int(s) for s in hist.get("steps", [0])]
    curve_min_rec = [float(v) for v in hist.get("rare_mode_recall", [rare_mode_recall])]
    views = [
        {
            "kind": "scatter",
            "title": "5-mode target vs API samples (99% major + 1% rare)",
            "target": ref_cpu[:1024],
            "samples": pts[:1024],
            "xlim": [-3.4, 3.4],
            "ylim": [-3.4, 3.4],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": "4 major modes at (+-1.5, +-1.5) and 1% rare mode at (0, 2.85) near 3*tanh saturation.",
        },
        {
            "kind": "scatter",
            "title": "1% minority mode close-up at (0.0, 2.85)",
            "target": minor_ref[:128],
            "samples": minor_samples[:256],
            "xlim": [-0.8, 0.8],
            "ylim": [2.1, 3.4],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": f"rare_mode_recall={rare_mode_recall:.2f} (gate >=0.45), precision={precision:.3f}.",
        },
        _curve_view(
            "Rare 1% mode recall trajectory (gate >= 0.45)",
            curve_steps,
            curve_min_rec,
            0.45,
            ylabel="rare_mode_recall",
            caption=f"mass_tv={mass_tv:.3f} (gate <=0.10); modes={modes}/5.",
            ylim=[0.0, max(1.4, max(curve_min_rec) * 1.15)],
        ),
    ]
    return _finalize_observation(task_id, metrics, views)


@torch.no_grad()
def score_fl_law42_swap_deep_critic_16l(
    samples: torch.Tensor,
    *,
    valley_sep: float = 0.0,
    critic_r_stable: float = 0.0,
    reference: torch.Tensor | None = None,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score ``fl_law42_swap_deep_critic_16l`` purely on 8-mode 2D ring generated sample quality."""
    task_id = "fl_law42_swap_deep_critic_16l"
    pts = samples.detach().cpu().to(dtype=torch.float32)
    ref = reference
    if ref is None:
        g_ref = torch.Generator(device="cpu").manual_seed(420001)
        ref = sample_target_for_kind(task_id, max(1024, min(4096, len(pts) if pts.ndim >= 1 else 1024)), g_ref)
    ref_cpu = ref.detach().cpu().to(dtype=torch.float32)
    centers, valleys = _ring8_centers_and_valleys("cpu")

    valid_shape = pts.ndim == 2 and pts.shape[1] == 2 and len(pts) > 0
    finite_fraction = float(torch.isfinite(pts).all(dim=-1).float().mean().item()) if valid_shape else 0.0
    if not valid_shape or finite_fraction < 1.0:
        fallback = {
            "finite_fraction": finite_fraction,
            "modes": 0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "valley_fraction": 1.0,
            "diag_valley_sep": 0.0,
            "diag_critic_r_stable": 0.0,
        }
        zeros = torch.zeros((4, 2), dtype=torch.float32)
        views = [
            {
                "kind": "scatter",
                "title": "8-mode 2D ring (3L G + 16L skip-free D)",
                "target": ref_cpu[:512],
                "samples": zeros,
                "xlim": [-3.4, 3.4],
                "ylim": [-3.4, 3.4],
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Invalid/nonfinite output fallback.",
            }
        ]
        return _finalize_observation(task_id, fallback, views)

    dists = torch.cdist(pts, centers)
    min_d, nearest = dists.min(dim=1)
    hq_mask = min_d <= (3.5 * _RING8_SIGMA)
    hq = float(hq_mask.float().mean().item())

    valley_d = torch.cdist(pts, valleys).min(dim=1).values
    valley_fraction = float((valley_d <= (3.5 * _RING8_SIGMA)).float().mean().item())

    n_total = len(pts)
    hq_counts = torch.bincount(nearest[hq_mask], minlength=_RING8_MODES)
    min_mode_cnt = max(8, int(n_total * 0.04))
    modes = int((hq_counts >= min_mode_cnt).sum().item())
    emp_mass = torch.bincount(nearest, minlength=_RING8_MODES).float() / float(n_total)
    mass_tv = float(0.5 * (emp_mass - (1.0 / float(_RING8_MODES))).abs().sum().item())

    metrics = {
        "finite_fraction": 1.0,
        "modes": modes,
        "hq": hq,
        "mass_tv": mass_tv,
        "valley_fraction": valley_fraction,
        # Non-gating diagnostic telemetry only:
        "diag_valley_sep": float(valley_sep) if math.isfinite(float(valley_sep)) else 0.0,
        "diag_critic_r_stable": float(critic_r_stable) if math.isfinite(float(critic_r_stable)) else 0.0,
    }

    hist = history or {}
    curve_steps = [int(s) for s in hist.get("steps", [0])]
    curve_hq = [float(v) for v in hist.get("hq", [hq])]
    curve_srk = [float(v) for v in hist.get("diag_critic_r_stable", [metrics["diag_critic_r_stable"]])]
    views = [
        {
            "kind": "scatter",
            "title": "8-mode 2D ring: shallow 3L G + deep 16L skip-free D",
            "target": ref_cpu[:1024],
            "samples": pts[:1024],
            "xlim": [-3.4, 3.4],
            "ylim": [-3.4, 3.4],
            "aspect": "equal",
            "xlabel": "x1",
            "ylabel": "x2",
            "caption": f"modes={modes}/8, hq={hq:.3f}, valley_fraction={valley_fraction:.3f}.",
        },
        _curve_view(
            "Generated ring HQ trajectory (gate >= 0.85)",
            curve_steps,
            curve_hq,
            0.85,
            ylabel="hq",
            caption=f"mass_tv={mass_tv:.3f} (gate <=0.12); valley_fraction={valley_fraction:.3f} (<=0.05).",
            ylim=[0.0, 1.05],
        ),
        _curve_view(
            "Diagnostic (non-gating): 16L critic hidden r_stable",
            curve_steps,
            curve_srk,
            1.50,
            ylabel="diag_critic_r_stable",
            caption=f"Non-gating diagnostic only; diag_valley_sep={metrics['diag_valley_sep']:.3f}.",
            ylim=[0.0, max(5.0, max(curve_srk) * 1.15)],
        ),
    ]
    return _finalize_observation(task_id, metrics, views)


@torch.no_grad()
def score_fl_law42_deep_resbn_and_conv1x1(
    samples_linear: torch.Tensor,
    *,
    samples_conv1x1: torch.Tensor | None = None,
    stiff_gate_mean: float = 0.0,
    d_max_s1: float = 0.0,
    s_therm: float = 1.0,
    reference: torch.Tensor | None = None,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score ``fl_law42_deep_resbn_and_conv1x1`` purely on Linear2D & Conv1x1 generated outputs."""
    task_id = "fl_law42_deep_resbn_and_conv1x1"
    mf = _highd_manifold("resbn")
    pts_lin = samples_linear.detach().cpu().to(dtype=torch.float32)
    pts_conv = pts_lin if samples_conv1x1 is None else samples_conv1x1.detach().cpu().to(dtype=torch.float32)
    ref = reference
    if ref is None:
        g_ref = torch.Generator(device="cpu").manual_seed(421280)
        ref = sample_target_for_kind(task_id, max(1024, min(2048, len(pts_lin) if pts_lin.ndim >= 1 else 1024)), g_ref)
    ref_cpu = ref.detach().cpu().to(dtype=torch.float32)
    ref_pca = _project_pca_2d(ref_cpu[:512], mf)

    valid_lin = pts_lin.ndim == 2 and pts_lin.shape[1] == _RESBN_DIM and len(pts_lin) >= 4
    valid_conv = pts_conv.ndim == 2 and pts_conv.shape[1] == _RESBN_DIM and len(pts_conv) >= 4
    fin_lin = float(torch.isfinite(pts_lin).all(dim=-1).float().mean().item()) if valid_lin else 0.0
    fin_conv = float(torch.isfinite(pts_conv).all(dim=-1).float().mean().item()) if valid_conv else 0.0
    finite_fraction = min(fin_lin, fin_conv)
    if not valid_lin or not valid_conv or finite_fraction < 1.0:
        fallback = {
            "finite_fraction": finite_fraction,
            "sw1": 9.9,
            "var_ratio": 0.0,
            "var_ratio_err": 1.0,
            "conv1x1_sw1": 9.9,
            "conv1x1_sw1_ratio": 9.9,
            "diag_stiff_gate_mean": 0.0,
            "diag_d_max_s1": 0.0,
            "diag_s_therm": 0.0,
        }
        zeros = torch.zeros((4, 2), dtype=torch.float32)
        views = [
            {
                "kind": "scatter",
                "title": "ResBN d=8 2D PCA projection (Linear 2D)",
                "target": ref_pca,
                "samples": zeros,
                "aspect": "equal",
                "xlabel": "PC1",
                "ylabel": "PC2",
                "caption": "Invalid/nonfinite output fallback.",
            }
        ]
        return _finalize_observation(task_id, fallback, views)

    sw1_lin = sliced_wasserstein_1(pts_lin, ref_cpu, mf["sw_proj"])
    sw1_conv = sliced_wasserstein_1(pts_conv, ref_cpu, mf["sw_proj"])
    real_var = float(ref_cpu.var(dim=0, unbiased=False).mean().item())
    fake_var = float(pts_lin.var(dim=0, unbiased=False).mean().item())
    var_ratio = fake_var / max(real_var, 1e-8)
    var_ratio_err = abs(var_ratio - 1.0)
    conv1x1_sw1_ratio = sw1_conv / max(sw1_lin, 1e-4)

    metrics = {
        "finite_fraction": 1.0,
        "sw1": float(sw1_lin),
        "var_ratio": float(var_ratio),
        "var_ratio_err": float(var_ratio_err),
        "conv1x1_sw1": float(sw1_conv),
        "conv1x1_sw1_ratio": float(conv1x1_sw1_ratio),
        # Non-gating diagnostic telemetry only:
        "diag_stiff_gate_mean": float(stiff_gate_mean) if math.isfinite(float(stiff_gate_mean)) else 0.0,
        "diag_d_max_s1": float(d_max_s1) if math.isfinite(float(d_max_s1)) else 0.0,
        "diag_s_therm": float(s_therm) if math.isfinite(float(s_therm)) else 1.0,
    }

    lin_pca = _project_pca_2d(pts_lin[:512], mf)
    conv_pca = _project_pca_2d(pts_conv[:512], mf)
    hist = history or {}
    curve_steps = [int(s) for s in hist.get("steps", [0])]
    curve_sw1 = [float(v) for v in hist.get("sw1", [sw1_lin])]
    views = [
        {
            "kind": "scatter",
            "title": "ResBN d=8 2D PCA: target vs Linear 2D",
            "target": ref_pca,
            "samples": lin_pca,
            "xlim": [-2.2, 2.2],
            "ylim": [-2.2, 2.2],
            "aspect": "equal",
            "xlabel": "PC1",
            "ylabel": "PC2",
            "caption": f"sw1={sw1_lin:.4f} (<=0.10), var_ratio={var_ratio:.3f}.",
        },
        {
            "kind": "scatter",
            "title": "ResBN d=8 2D PCA: target vs 4D Conv2d 1x1 twin",
            "target": ref_pca,
            "samples": conv_pca,
            "xlim": [-2.2, 2.2],
            "ylim": [-2.2, 2.2],
            "aspect": "equal",
            "xlabel": "PC1",
            "ylabel": "PC2",
            "caption": f"conv1x1_sw1={sw1_conv:.4f}, ratio={conv1x1_sw1_ratio:.3f} (gate <=1.05).",
        },
        _curve_view(
            "ResBN d=8 sliced-W1 trajectory (gate <= 0.10)",
            curve_steps,
            curve_sw1,
            0.10,
            ylabel="sw1",
            caption=(
                f"Non-gating diagnostics: D_max_s1={metrics['diag_d_max_s1']:.2f}, "
                f"stiff_gate={metrics['diag_stiff_gate_mean']:.3f}."
            ),
            ylim=[0.0, max(0.35, max(curve_sw1) * 1.15)],
        ),
    ]
    return _finalize_observation(task_id, metrics, views)


@torch.no_grad()
def score_fl_law44_highd_sparse_prior_and_shock(
    samples: torch.Tensor,
    *,
    shock_samples: torch.Tensor | None = None,
    step: int = _HIGHD_SHOCK_STEP + 1,
    shock_step: int = _HIGHD_SHOCK_STEP,
    mean_dz: float = 0.0,
    prior_peak_rms: float = 0.0,
    reference: torch.Tensor | None = None,
    shock_reference: torch.Tensor | None = None,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score ``fl_law44_highd_sparse_prior_and_shock`` purely on generated sample quality."""
    task_id = "fl_law44_highd_sparse_prior_and_shock"
    mf = _highd_manifold("highd_sparse")
    pts = samples.detach().cpu().to(dtype=torch.float32)
    ref = reference
    if ref is None:
        g_ref = torch.Generator(device="cpu").manual_seed(440513)
        ref = sample_target_for_kind(task_id, max(1024, min(2048, len(pts) if pts.ndim >= 1 else 1024)), g_ref)
    ref_cpu = ref.detach().cpu().to(dtype=torch.float32)
    ref_pca = _project_pca_2d(ref_cpu[:512], mf)

    shock_ref = shock_reference
    if shock_ref is None:
        g_sref = torch.Generator(device="cpu").manual_seed(440514)
        shock_ref = sample_shock_target(1024, g_sref, step=step, shock_step=shock_step, device="cpu", eval_phase=True)
    shock_ref_unit = shock_ref.detach().cpu().to(dtype=torch.float32) / _SHOCK_BASE_SCALE

    s_pts = shock_ref if shock_samples is None else shock_samples.detach().cpu().to(dtype=torch.float32)

    valid_main = pts.ndim == 2 and pts.shape[1] == _HIGHD_DIM and len(pts) >= 4
    valid_shock = s_pts.ndim == 2 and s_pts.shape[1] == 2 and len(s_pts) >= 4
    fin_main = float(torch.isfinite(pts).all(dim=-1).float().mean().item()) if valid_main else 0.0
    fin_shock = float(torch.isfinite(s_pts).all(dim=-1).float().mean().item()) if valid_shock else 0.0
    finite_fraction = min(fin_main, fin_shock)
    if not valid_main or not valid_shock or finite_fraction < 1.0:
        fallback = {
            "finite_fraction": finite_fraction,
            "sw1": 9.9,
            "mode_recall": 0.0,
            "var_ratio": 0.0,
            "var_ratio_err": 1.0,
            "shock_modes": 0,
            "shock_hq": 0.0,
            "shock_sw1": 9.9,
            "diag_mean_dz": 0.0,
            "diag_prior_peak_rms": 0.0,
        }
        zeros = torch.zeros((4, 2), dtype=torch.float32)
        views = [
            {
                "kind": "scatter",
                "title": "R^512 10%-masked manifold 2D PCA projection",
                "target": ref_pca,
                "samples": zeros,
                "aspect": "equal",
                "xlabel": "PC1",
                "ylabel": "PC2",
                "caption": "Invalid/nonfinite output fallback.",
            }
        ]
        return _finalize_observation(task_id, fallback, views)

    # Primary R^512 manifold generated-output quality:
    sw1_val = sliced_wasserstein_1(pts, ref_cpu, mf["sw_proj"])
    real_var = float(ref_cpu.var(dim=0, unbiased=False).mean().item())
    fake_var = float(pts.var(dim=0, unbiased=False).mean().item())
    var_ratio = fake_var / max(real_var, 1e-8)
    var_ratio_err = abs(var_ratio - 1.0)

    centers_hd = mf["centers"]
    dists_hd = torch.cdist(pts, centers_hd)
    min_d_hd, nearest_hd = dists_hd.min(dim=1)
    # Typical Euclidean norm of N(0, sigma^2 I_512) is sigma * sqrt(512) ~= 1.13; allow 3.0x radius
    hd_radius = 3.0 * _HIGHD_SIGMA * math.sqrt(float(_HIGHD_DIM))
    valid_hd = min_d_hd <= hd_radius
    mode_counts_hd = torch.bincount(nearest_hd[valid_hd], minlength=_HIGHD_MODES)
    min_cnt_hd = max(2, len(pts) // (_HIGHD_MODES * 8))
    mode_recall = float((mode_counts_hd >= min_cnt_hd).float().mean().item())

    # Paired 1e-5x extreme-scale / post-shock generated-output quality (normalized by 1 / 1e-5):
    s_unit = s_pts / _SHOCK_BASE_SCALE
    active_centers_unit = torch.tensor(
        _SHOCK_POST_CENTERS if int(step) > shock_step else _SHOCK_PRE_CENTERS,
        dtype=torch.float32,
    )
    d_shock, idx_shock = torch.cdist(s_unit, active_centers_unit).min(dim=1)
    hq_shock_mask = d_shock <= (3.5 * _SHOCK_SIGMA_UNIT)
    shock_hq = float(hq_shock_mask.float().mean().item())
    shock_counts = torch.bincount(idx_shock[hq_shock_mask], minlength=4)
    min_cnt_shock = max(4, len(s_unit) // 32)
    shock_modes = int((shock_counts >= min_cnt_shock).sum().item())
    shock_sw1 = sliced_wasserstein_1(s_unit, shock_ref_unit, _shock_2d_proj())

    metrics = {
        "finite_fraction": 1.0,
        "sw1": float(sw1_val),
        "mode_recall": float(mode_recall),
        "var_ratio": float(var_ratio),
        "var_ratio_err": float(var_ratio_err),
        "shock_modes": shock_modes,
        "shock_hq": float(shock_hq),
        "shock_sw1": float(shock_sw1),
        # Non-gating diagnostic telemetry only:
        "diag_mean_dz": float(mean_dz) if math.isfinite(float(mean_dz)) else 0.0,
        "diag_prior_peak_rms": float(prior_peak_rms) if math.isfinite(float(prior_peak_rms)) else 0.0,
    }

    samp_pca = _project_pca_2d(pts[:512], mf)
    hist = history or {}
    curve_steps = [int(s) for s in hist.get("steps", [0])]
    curve_sw1 = [float(v) for v in hist.get("sw1", [sw1_val])]
    views = [
        {
            "kind": "scatter",
            "title": "R^512 10%-masked manifold: 2D PCA target vs clean G(z)",
            "target": ref_pca,
            "samples": samp_pca,
            "xlim": [-5.0, 5.0],
            "ylim": [-5.0, 5.0],
            "aspect": "equal",
            "xlabel": "PC1",
            "ylabel": "PC2",
            "caption": f"sw1={sw1_val:.4f} (<=0.14), mode_recall={mode_recall:.2f} (>=0.75).",
        },
        {
            "kind": "scatter",
            "title": "1e-5x scale & post-1e4x shock twin (normalized by 1e5)",
            "target": shock_ref_unit[:512],
            "samples": s_unit[:512].clamp(-4.0, 4.0),
            "xlim": [-3.0, 3.0],
            "ylim": [-3.0, 3.0],
            "aspect": "equal",
            "xlabel": "x1 / 1e-5",
            "ylabel": "x2 / 1e-5",
            "caption": f"shock_modes={shock_modes}/4, shock_hq={shock_hq:.3f} (>=0.65), shock_sw1={shock_sw1:.3f}.",
        },
        _curve_view(
            "R^512 manifold sliced-W1 trajectory (gate <= 0.14)",
            curve_steps,
            curve_sw1,
            0.14,
            ylabel="sw1",
            caption=f"var_ratio={var_ratio:.3f}; non-gating diag_mean_dz={metrics['diag_mean_dz']:.3f}.",
            ylim=[0.0, max(0.5, max(curve_sw1) * 1.15)],
        ),
    ]
    return _finalize_observation(task_id, metrics, views)


@torch.no_grad()
def score_fl_law42_deep_resbn_and_conv1x1_v2(
    samples_linear: torch.Tensor,
    *,
    samples_conv1x1: torch.Tensor | None = None,
    stiff_gate_mean: float = 0.0,
    d_max_s1: float = 0.0,
    s_therm: float = 1.0,
    reference: torch.Tensor | None = None,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score v2 ResBN d=8 & isomorphic Conv2d 1x1 twin with absolute parity ``|sw1_conv - sw1_lin| <= 0.015``."""
    task_id = "fl_law42_deep_resbn_and_conv1x1_v2"
    base = score_fl_law42_deep_resbn_and_conv1x1(
        samples_linear,
        samples_conv1x1=samples_conv1x1,
        stiff_gate_mean=stiff_gate_mean,
        d_max_s1=d_max_s1,
        s_therm=s_therm,
        reference=reference,
        history=history,
    )
    metrics = {k: v for k, v in base["metrics"].items() if k != "min_margin"}
    if metrics.get("finite_fraction", 0.0) == 1.0 and metrics.get("sw1", 9.9) < 9.0 and metrics.get("conv1x1_sw1", 9.9) < 9.0:
        metrics["conv1x1_sw1_abs_diff"] = float(abs(float(metrics["conv1x1_sw1"]) - float(metrics["sw1"])))
    else:
        metrics["conv1x1_sw1_abs_diff"] = 9.9
    return _finalize_observation(task_id, metrics, base["views"])


@torch.no_grad()
def score_fl_law44_scale_shock(
    samples: torch.Tensor,
    *,
    step: int = _V2_SHOCK_STEP + 1,
    shock_step: int = _V2_SHOCK_STEP,
    reference: torch.Tensor | None = None,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score standalone Law 4.4 extreme-scale (1e-5x) + transient 1e4x shock & 45-deg rotation recovery."""
    task_id = "fl_law44_scale_shock"
    pts = samples.detach().cpu().to(dtype=torch.float32)
    shock_ref = reference
    if shock_ref is None:
        g_sref = torch.Generator(device="cpu").manual_seed(440301)
        shock_ref = sample_shock_target(
            1024, g_sref, step=step, shock_step=shock_step, device="cpu", eval_phase=True
        )
    shock_ref_unit = shock_ref.detach().cpu().to(dtype=torch.float32) / _SHOCK_BASE_SCALE

    valid = pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 4
    all_finite = bool(torch.isfinite(pts).all().item()) if valid else False
    finite_fraction = 1.0 if all_finite else (float(torch.isfinite(pts).all(dim=-1).float().mean().item()) if valid else 0.0)
    if not valid or not all_finite:
        fallback = {
            "finite_fraction": finite_fraction,
            "shock_modes": 0,
            "shock_hq": 0.0,
            "shock_sw1": 9.9,
            "shock_mass_tv": 1.0,
        }
        zeros = torch.zeros((4, 2), dtype=torch.float32)
        views = [
            {
                "kind": "scatter",
                "title": "Standalone 1e-5x scale & post-1e4x shock (normalized by 1e5)",
                "target": shock_ref_unit[:512],
                "samples": zeros,
                "xlim": [-3.0, 3.0],
                "ylim": [-3.0, 3.0],
                "aspect": "equal",
                "xlabel": "x1 / 1e-5",
                "ylabel": "x2 / 1e-5",
                "caption": "Invalid/nonfinite output fallback.",
            }
        ]
        return _finalize_observation(task_id, fallback, views)

    s_unit = pts / _SHOCK_BASE_SCALE
    active_centers_unit = torch.tensor(
        _SHOCK_POST_CENTERS if int(step) > shock_step else _SHOCK_PRE_CENTERS,
        dtype=torch.float32,
    )
    d_shock, idx_shock = torch.cdist(s_unit, active_centers_unit).min(dim=1)
    hq_shock_mask = d_shock <= (3.5 * _SHOCK_SIGMA_UNIT)
    shock_hq = float(hq_shock_mask.float().mean().item())
    shock_counts = torch.bincount(idx_shock[hq_shock_mask], minlength=4)
    min_cnt_shock = max(4, len(s_unit) // 32)
    shock_modes = int((shock_counts >= min_cnt_shock).sum().item())
    shock_sw1 = sliced_wasserstein_1(s_unit, shock_ref_unit, _shock_2d_proj())
    emp_mass = torch.bincount(idx_shock, minlength=4).float() / float(len(s_unit))
    shock_mass_tv = float(0.5 * (emp_mass - 0.25).abs().sum().item())

    metrics = {
        "finite_fraction": 1.0,
        "shock_modes": shock_modes,
        "shock_hq": float(shock_hq),
        "shock_sw1": float(shock_sw1),
        "shock_mass_tv": float(shock_mass_tv),
    }
    hist = history or {}
    curve_steps = [int(s) for s in hist.get("steps", [0])]
    curve_sw1 = [float(v) for v in hist.get("shock_sw1", [shock_sw1])]
    views = [
        {
            "kind": "scatter",
            "title": "Standalone 1e-5x scale & post-1e4x shock (normalized by 1e5)",
            "target": shock_ref_unit[:512],
            "samples": s_unit[:512].clamp(-4.0, 4.0),
            "xlim": [-3.0, 3.0],
            "ylim": [-3.0, 3.0],
            "aspect": "equal",
            "xlabel": "x1 / 1e-5",
            "ylabel": "x2 / 1e-5",
            "caption": (
                f"shock_modes={shock_modes}/4, shock_hq={shock_hq:.3f} (>=0.65), "
                f"shock_sw1={shock_sw1:.3f} (<=0.35), shock_mass_tv={shock_mass_tv:.3f}."
            ),
        },
        _curve_view(
            "Scale-shock sliced-W1 recovery trajectory (gate <= 0.35)",
            curve_steps,
            curve_sw1,
            0.35,
            ylabel="shock_sw1",
            caption=f"Shock at step {shock_step}: 1e-5x -> 1e-1x spike -> 45-deg rotated 1e-5x modes.",
            ylim=[0.0, max(1.0, max(curve_sw1) * 1.15)],
        ),
    ]
    return _finalize_observation(task_id, metrics, views)


# ---------------------------------------------------------------------------
# Fast Deterministic CPU Oracle PASS & Destructive FAIL Controls
# ---------------------------------------------------------------------------

def oracle_controls() -> dict[str, dict[str, Any]]:
    """Fast deterministic CPU oracle PASS + destructive FAIL controls for all 4 Law 4 tasks."""
    controls: dict[str, dict[str, Any]] = {}

    def _rec(name: str, expected_pass: bool, res: dict[str, Any]) -> None:
        controls[name] = {
            "expected_pass": expected_pass,
            "passed": bool(res["passed"]),
            "failed_bounds": list(res["failed_bounds"]),
            "min_margin": float(res["min_margin"]),
        }

    # --- Task 1: fl_law41_trust_rare1pct_tanh ---
    g1 = torch.Generator(device="cpu").manual_seed(94101)
    ref1 = sample_target_for_kind("fl_law41_trust_rare1pct_tanh", 4096, g1)
    oracle1 = sample_target_for_kind("fl_law41_trust_rare1pct_tanh", 8192, g1)
    _rec("fl_law41_exact_oracle", True, score_fl_law41_trust_rare1pct_tanh(oracle1, reference=ref1))

    maj_centers = torch.tensor(_RARE1PCT_CENTERS[:4], dtype=torch.float32)
    idx_maj = torch.randint(0, 4, (8192,), generator=g1)
    no_minor = maj_centers[idx_maj] + _RARE1PCT_SIGMA * torch.randn((8192, 2), generator=g1)
    _rec("fl_law41_missing_1pct_minority", False, score_fl_law41_trust_rare1pct_tanh(no_minor, reference=ref1))

    spread_noise = torch.randn((8192, 2), generator=g1) * 2.0
    _rec("fl_law41_unfocused_noise", False, score_fl_law41_trust_rare1pct_tanh(spread_noise, reference=ref1))

    nan1 = oracle1.clone()
    nan1[0, 0] = float("nan")
    _rec("fl_law41_nonfinite", False, score_fl_law41_trust_rare1pct_tanh(nan1, reference=ref1))

    # --- Task 2: fl_law42_swap_deep_critic_16l ---
    g2 = torch.Generator(device="cpu").manual_seed(94201)
    ref2 = sample_target_for_kind("fl_law42_swap_deep_critic_16l", 2048, g2)
    oracle2 = sample_target_for_kind("fl_law42_swap_deep_critic_16l", 4096, g2)
    _rec("fl_law42_swap_exact_oracle", True, score_fl_law42_swap_deep_critic_16l(oracle2, reference=ref2))

    centers8, valleys8 = _ring8_centers_and_valleys("cpu")
    single_mode = centers8[0:1] + _RING8_SIGMA * torch.randn((4096, 2), generator=g2)
    _rec("fl_law42_swap_mode_collapse", False, score_fl_law42_swap_deep_critic_16l(single_mode, reference=ref2))

    # Valley contamination: 20% samples smeared into inter-mode valleys
    idx_v = torch.randint(0, _RING8_MODES, (1024,), generator=g2)
    valley_pts = valleys8[idx_v] + _RING8_SIGMA * torch.randn((1024, 2), generator=g2)
    contaminated = torch.cat((oracle2[:3072], valley_pts), dim=0)
    _rec("fl_law42_swap_valley_contamination", False, score_fl_law42_swap_deep_critic_16l(contaminated, reference=ref2))

    nan2 = oracle2.clone()
    nan2[0, 1] = float("nan")
    _rec("fl_law42_swap_nonfinite", False, score_fl_law42_swap_deep_critic_16l(nan2, reference=ref2))

    # --- Task 3: fl_law42_deep_resbn_and_conv1x1 ---
    g3 = torch.Generator(device="cpu").manual_seed(94301)
    ref3 = sample_target_for_kind("fl_law42_deep_resbn_and_conv1x1", 1024, g3)
    oracle3_lin = sample_target_for_kind("fl_law42_deep_resbn_and_conv1x1", 1024, g3)
    oracle3_conv = oracle3_lin.clone()
    _rec(
        "fl_law42_resbn_exact_oracle",
        True,
        score_fl_law42_deep_resbn_and_conv1x1(oracle3_lin, samples_conv1x1=oracle3_conv, reference=ref3),
    )
    degraded_conv = oracle3_lin * 0.45
    _rec(
        "fl_law42_resbn_conv1x1_parity_fail",
        False,
        score_fl_law42_deep_resbn_and_conv1x1(oracle3_lin, samples_conv1x1=degraded_conv, reference=ref3),
    )
    collapsed3 = torch.zeros_like(oracle3_lin)
    _rec(
        "fl_law42_resbn_collapsed_zero",
        False,
        score_fl_law42_deep_resbn_and_conv1x1(collapsed3, samples_conv1x1=collapsed3, reference=ref3),
    )
    nan3 = oracle3_lin.clone()
    nan3[0, 0] = float("nan")
    _rec(
        "fl_law42_resbn_nonfinite",
        False,
        score_fl_law42_deep_resbn_and_conv1x1(nan3, samples_conv1x1=oracle3_conv, reference=ref3),
    )

    # --- Task 4: fl_law44_highd_sparse_prior_and_shock ---
    g4 = torch.Generator(device="cpu").manual_seed(94401)
    ref4 = sample_target_for_kind("fl_law44_highd_sparse_prior_and_shock", 1024, g4)
    oracle4 = sample_target_for_kind("fl_law44_highd_sparse_prior_and_shock", 1024, g4)
    shock_ref = sample_shock_target(1024, g4, step=_HIGHD_SHOCK_STEP + 10, device="cpu", eval_phase=True)
    shock_oracle = sample_shock_target(1024, g4, step=_HIGHD_SHOCK_STEP + 10, device="cpu", eval_phase=True)
    _rec(
        "fl_law44_exact_oracle",
        True,
        score_fl_law44_highd_sparse_prior_and_shock(
            oracle4,
            shock_samples=shock_oracle,
            step=_HIGHD_SHOCK_STEP + 10,
            reference=ref4,
            shock_reference=shock_ref,
        ),
    )
    # Destructive 4a: high-D mode collapse (stuck on single center -> fails sw1, mode_recall, var_ratio_err)
    mf4 = _highd_manifold("highd_sparse")
    collapsed_hd = mf4["centers"][0:1].expand(1024, -1) + 0.01 * torch.randn((1024, _HIGHD_DIM), generator=g4)
    _rec(
        "fl_law44_highd_mode_collapse",
        False,
        score_fl_law44_highd_sparse_prior_and_shock(
            collapsed_hd,
            shock_samples=shock_oracle,
            step=_HIGHD_SHOCK_STEP + 10,
            reference=ref4,
            shock_reference=shock_ref,
        ),
    )
    # Destructive 4b: frozen at pre-shock modes after mid-run shock (fails shock_modes, shock_hq, shock_sw1)
    pre_shock_stuck = sample_shock_target(1024, g4, step=10, device="cpu", eval_phase=True)
    _rec(
        "fl_law44_shock_frozen_pre_shift",
        False,
        score_fl_law44_highd_sparse_prior_and_shock(
            oracle4,
            shock_samples=pre_shock_stuck,
            step=_HIGHD_SHOCK_STEP + 10,
            reference=ref4,
            shock_reference=shock_ref,
        ),
    )
    # Destructive 4c: NaN output
    nan4 = oracle4.clone()
    nan4[0, 0] = float("nan")
    _rec(
        "fl_law44_nonfinite",
        False,
        score_fl_law44_highd_sparse_prior_and_shock(
            nan4,
            shock_samples=shock_oracle,
            step=_HIGHD_SHOCK_STEP + 10,
            reference=ref4,
            shock_reference=shock_ref,
        ),
    )
    return controls


def oracle_controls_v2() -> dict[str, dict[str, Any]]:
    """Fast deterministic CPU oracle PASS + destructive FAIL controls for v2 Law 4 probes."""
    controls: dict[str, dict[str, Any]] = {}

    def _rec(name: str, expected_pass: bool, res: dict[str, Any]) -> None:
        controls[name] = {
            "expected_pass": expected_pass,
            "passed": bool(res["passed"]),
            "failed_bounds": list(res["failed_bounds"]),
            "min_margin": float(res["min_margin"]),
        }

    # --- Task v2-1: fl_law42_deep_resbn_and_conv1x1_v2 ---
    g3 = torch.Generator(device="cpu").manual_seed(94311)
    ref3 = sample_target_for_kind("fl_law42_deep_resbn_and_conv1x1_v2", 1024, g3)
    ora3_lin = sample_target_for_kind("fl_law42_deep_resbn_and_conv1x1_v2", 1024, g3)
    ora3_conv = ora3_lin.clone()
    _rec(
        "fl_law42_resbn_v2_exact_oracle",
        True,
        score_fl_law42_deep_resbn_and_conv1x1_v2(ora3_lin, samples_conv1x1=ora3_conv, reference=ref3),
    )
    # Small parity divergence (0.003 diff at sw1 ~ 0.023) must PASS absolute parity <= 0.015
    slightly_noisy_conv = ora3_lin + 0.01 * torch.randn_like(ora3_lin)
    _rec(
        "fl_law42_resbn_v2_small_diff_pass",
        True,
        score_fl_law42_deep_resbn_and_conv1x1_v2(ora3_lin, samples_conv1x1=slightly_noisy_conv, reference=ref3),
    )
    degraded_conv = ora3_lin * 0.45
    _rec(
        "fl_law42_resbn_v2_conv1x1_parity_fail",
        False,
        score_fl_law42_deep_resbn_and_conv1x1_v2(ora3_lin, samples_conv1x1=degraded_conv, reference=ref3),
    )
    nan3 = ora3_lin.clone()
    nan3[0, 0] = float("nan")
    _rec(
        "fl_law42_resbn_v2_nonfinite",
        False,
        score_fl_law42_deep_resbn_and_conv1x1_v2(nan3, samples_conv1x1=ora3_conv, reference=ref3),
    )

    # --- Task v2-2: fl_law44_scale_shock ---
    g4 = torch.Generator(device="cpu").manual_seed(94411)
    sref = sample_shock_target(2048, g4, step=_V2_SHOCK_STEP + 50, shock_step=_V2_SHOCK_STEP, device="cpu", eval_phase=True)
    sora = sample_shock_target(2048, g4, step=_V2_SHOCK_STEP + 50, shock_step=_V2_SHOCK_STEP, device="cpu", eval_phase=True)
    _rec(
        "fl_law44_scale_shock_exact_oracle",
        True,
        score_fl_law44_scale_shock(sora, step=_V2_SHOCK_STEP + 50, shock_step=_V2_SHOCK_STEP, reference=sref),
    )
    pre_stuck = sample_shock_target(2048, g4, step=10, shock_step=_V2_SHOCK_STEP, device="cpu", eval_phase=True)
    _rec(
        "fl_law44_scale_shock_frozen_pre_shift",
        False,
        score_fl_law44_scale_shock(pre_stuck, step=_V2_SHOCK_STEP + 50, shock_step=_V2_SHOCK_STEP, reference=sref),
    )
    nan4 = sora.clone()
    nan4[0, 0] = float("nan")
    _rec(
        "fl_law44_scale_shock_nonfinite",
        False,
        score_fl_law44_scale_shock(nan4, step=_V2_SHOCK_STEP + 50, shock_step=_V2_SHOCK_STEP, reference=sref),
    )
    return controls


# ---------------------------------------------------------------------------
# Case Listing & Public-API Fixture
# ---------------------------------------------------------------------------

def _build_case_records(task_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for tid in task_ids:
        spec = ALL_SPECS[tid]
        res = spec["resources"]
        cases.append(
            {
                "id": tid,
                "title": spec["title"],
                "goal": spec["goal"],
                "kind": spec["view_kind"],
                "scope": f"Public API Four Laws ({spec['law']}, defects={','.join(spec['defects'])}): {spec['title']}",
                "legacy_ids": [f"four-laws-{tid}"],
                "default_steps": int(spec["steps"]),
                "batch_size": int(res["batch_size"]),
                "eval_samples": int(spec["eval_samples"]),
                "evaluation_observations": int(spec["observations"]),
                "terminal_observations": int(spec["minimum_stable_checks"]),
                "thresholds": {name: f"{op} {bound}" for name, op, bound in ALL_THRESHOLDS[tid]},
                "sampling": "Clean live public MoGParticlePrior samples without training output noise",
                "default_recipe": "unified_v14",
                "protocol_seed": DEFAULT_SEED,
            }
        )
    return cases


def list_cases() -> list[dict[str, Any]]:
    """Return ``api_contract.validate_case``-compliant case declarations for the 4 frozen v1 Law 4 tasks."""
    return _build_case_records(TASKS)


def list_v2_cases() -> list[dict[str, Any]]:
    """Return ``api_contract.validate_case``-compliant case declarations for the 2 calibrated v2 Law 4 tasks."""
    return _build_case_records(V2_TASKS)


def list_all_cases() -> list[dict[str, Any]]:
    """Return case declarations for both v1 and v2 Law 4 tasks."""
    return _build_case_records(ALL_TASKS)


class FourLawsFixture:
    """Strictly public-API, method-agnostic ``GANTrainer`` fixture supporting all 4 Law 4 tasks."""

    api_components = ("GANTrainer", "MoGParticlePrior", "Recipe", "init.deterministic_orthogonal_")

    def __init__(
        self,
        case: dict[str, Any] | str,
        *,
        recipe_name: str | None = "unified_v14",
        recipe: Recipe | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
        components: Any = None,
    ) -> None:
        if isinstance(case, str):
            tid = case
            case_obj = next((c for c in list_all_cases() if c["id"] == tid), None)
            if case_obj is None:
                raise ValueError(f"unknown four_laws_4 task case: {tid!r}")
        else:
            case_obj = deepcopy(case)
            tid = str(case_obj["id"])
        if tid not in ALL_SPECS:
            raise ValueError(f"unknown four_laws_4 task case: {tid!r}")

        self.case = case_obj
        self.case_id = tid
        self.kind = tid
        self.spec = deepcopy(ALL_SPECS[tid])
        self.device = torch.device(device)
        self.seed = int(seed)
        self.limit = int(self.spec["steps"] if max_steps is None else max_steps)
        if type(self.limit) is not int or self.limit < 1 or self.seed < 0:
            raise ValueError("max_steps must be positive and seed must be nonnegative")

        res = self.spec["resources"]
        prior_spec = self.spec["prior"]
        if components is not None:
            self.recipe = components.recipe
            self._components = components
            self.device = components.device
            self.seed = int(components.streams.seed)
        else:
            self._components = None
            base_recipe = get_recipe(recipe_name or "unified_v14") if recipe is None else recipe
            prior_kind = "mog" if prior_spec["kind"] == "mog" else "particles"
            updates: dict[str, Any] = {
                "num_particles": int(res["num_particles"]),
                "z_dim": int(res["z_dim"]),
                "batch_size": int(res["batch_size"]),
                "prior_kind": prior_kind,
                "sigma_rel": 0.0,
                "standardize": bool(prior_spec["standardize"]),
                **(recipe_overrides or {}),
            }
            if base_recipe.total_steps is not None:
                updates["total_steps"] = self.limit
            self.recipe = base_recipe.replace(**updates)

        self.aux_trainers: dict[str, GANTrainer] = {}
        self._aux_data_rngs: dict[str, torch.Generator] = {}
        self._history: dict[str, list[float]] = {"steps": []}
        self._window_step_disps: list[float] = []
        self._window_ood_ratios: list[float] = []
        self._window_stiff_gates: list[float] = []
        self._batch_sequence_sha256: str = hashlib.sha256(b"").hexdigest()

        # Construct primary G, D, prior, and GANTrainer
        if components is not None:
            def _make_g() -> nn.Module:
                return build_models_for_spec(self.spec, z_dim=self.recipe.z_dim, seed=self.seed)[0]

            def _make_d() -> nn.Module:
                return build_models_for_spec(self.spec, z_dim=self.recipe.z_dim, seed=self.seed)[1]

            g_mod = components.construct(_make_g, component="generator").to(self.device)
            d_mod = components.construct(_make_d, component="discriminator").to(self.device)
            self.trainer = components.build_trainer(g_mod, d_mod, max_steps=self.limit)
            self.data_rng = components.streams.generator("data", component="target", purpose="training", device="cpu")
        else:
            with construction_rng(self.seed, self.device):
                g_mod, d_mod = build_models_for_spec(self.spec, z_dim=self.recipe.z_dim, seed=self.seed)
            g_mod, d_mod = g_mod.to(self.device), d_mod.to(self.device)
            for comp, mod in (("generator", g_mod), ("discriminator", d_mod)):
                seeds = {
                    name: _seed_int(f"{tid}:init:{comp}:{name}", self.seed)
                    for name, p in mod.named_parameters()
                    if p.requires_grad and p.numel()
                }
                init.deterministic_orthogonal_(mod, parameter_seeds=seeds)
            prior = self._make_standalone_prior(self.recipe, sigma=float(prior_spec["sigma"]), tag_prefix=f"{tid}:primary")
            self.trainer = self._make_standalone_trainer(self.recipe, g_mod, d_mod, prior, tag_prefix=f"{tid}:primary")
            self.data_rng = torch.Generator(device="cpu").manual_seed(_seed_int(f"{tid}:data:training", self.seed))

        base_prior = unwrap_prior(self.trainer.prior)
        self._initial_prior_z = base_prior.z.detach().clone() if hasattr(base_prior, "z") else None

        # Task-specific deterministic non-gating probes / paired public GANTrainer twins (ZERO host repairs!)
        if self.kind == "fl_law41_trust_rare1pct_tanh":
            g_pr = torch.Generator(device="cpu").manual_seed(_seed_int("law41:probes", self.seed))
            z_dim = int(self.recipe.z_dim)
            self._z_id_probe = torch.randn((64, z_dim), generator=g_pr, dtype=torch.float32).to(self.device)
            z_ood = torch.randn((64, z_dim), generator=g_pr, dtype=torch.float32)
            z_ood = z_ood / z_ood.norm(dim=1, keepdim=True).clamp_min(1e-8) * (3.5 * math.sqrt(float(z_dim)))
            self._z_ood_probe = z_ood.to(self.device)
            ref_probe = sample_target_for_kind(self.kind, 2048, g_pr, device="cpu")
            self._data_coord_rms = float(ref_probe.square().mean().sqrt().item())
        elif self.kind in ("fl_law42_deep_resbn_and_conv1x1", "fl_law42_deep_resbn_and_conv1x1_v2"):
            cfg = self.spec["model"]
            dim = int(cfg["dim"])
            w = int(cfg["width"])
            depth = int(cfg["depth"])
            with construction_rng(self.seed, self.device):
                g_conv = ResBNConv1x1Net(int(self.recipe.z_dim), dim, w, depth).to(self.device)
                d_conv = ResBNConv1x1Net(dim, 1, w, depth).to(self.device)
            copy_resbn_linear_to_conv1x1(self.trainer.G, g_conv)
            copy_resbn_linear_to_conv1x1(self.trainer.D, d_conv)
            prior_conv = self._make_standalone_prior(self.recipe, sigma=float(prior_spec["sigma"]), tag_prefix=f"{tid}:conv1x1")
            if self._initial_prior_z is not None and hasattr(unwrap_prior(prior_conv), "z"):
                with torch.no_grad():
                    unwrap_prior(prior_conv).z.copy_(self._initial_prior_z.to(device=self.device))
            self.aux_trainers["conv1x1"] = self._make_cloned_rng_trainer(
                self.recipe, g_conv, d_conv, prior_conv, source=self.trainer
            )
        elif self.kind == "fl_law44_highd_sparse_prior_and_shock":
            shock_recipe = self.recipe.replace(num_particles=128, z_dim=4, batch_size=128)
            with construction_rng(self.seed + 44, self.device):
                g_sh = ScaledOutputGenerator(
                    z_dim=4, hidden=64, layers=2, out_dim=2, output_scale=_SHOCK_BASE_SCALE
                ).to(self.device)
                d_sh = SimpleMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=2, fourier=0).to(self.device)
            for comp, mod in (("generator", g_sh), ("discriminator", d_sh)):
                seeds = {
                    name: _seed_int(f"{tid}:shock:init:{comp}:{name}", self.seed)
                    for name, p in mod.named_parameters()
                    if p.requires_grad and p.numel()
                }
                init.deterministic_orthogonal_(mod, parameter_seeds=seeds)
            prior_sh = self._make_standalone_prior(shock_recipe, sigma=0.08, tag_prefix=f"{tid}:shock")
            self.aux_trainers["scale_shock"] = self._make_standalone_trainer(
                shock_recipe, g_sh, d_sh, prior_sh, tag_prefix=f"{tid}:shock"
            )
            self._aux_data_rngs["scale_shock"] = torch.Generator(device="cpu").manual_seed(
                _seed_int(f"{tid}:shock:data:training", self.seed)
            )

        self.metadata = {
            "checkpoint_capabilities": {
                "round_trip": True,
                "preserves_rng": True,
                "exact_continuation": True,
                "owner": "GANTrainer.state_dict + aux_trainers + caller data_rng",
            }
        }

    def _make_standalone_prior(self, recipe: Recipe, *, sigma: float, tag_prefix: str) -> nn.Module:
        prior_spec = self.spec["prior"]
        prior_rng = torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:init:prior:loc", self.seed))
        if prior_spec["kind"] == "mog":
            prior: nn.Module = MoGParticlePrior(
                num_particles=recipe.num_particles,
                z_dim=recipe.z_dim,
                sigma=float(sigma),
                standardize=bool(prior_spec["standardize"]),
                init_std=1.0,
                device=self.device,
                dtype=torch.float32,
                learnable=True,
                generator=prior_rng,
            )
        else:
            prior = ParticlePrior(
                num_particles=recipe.num_particles,
                z_dim=recipe.z_dim,
                init_std=1.0,
                device=self.device,
                dtype=torch.float32,
                learnable=True,
                generator=prior_rng,
            )
        init.deterministic_orthogonal_(prior, parameter_seeds={"z": _seed_int(f"{tag_prefix}:init:prior:z", self.seed)})
        recipe.apply_prior_policy(prior)
        return prior

    def _make_standalone_trainer(
        self,
        recipe: Recipe,
        g_mod: nn.Module,
        d_mod: nn.Module,
        prior: nn.Module,
        *,
        tag_prefix: str,
    ) -> GANTrainer:
        opts: dict[str, Any] = {
            "prior": prior,
            "seed": self.seed,
            "max_steps": self.limit,
            "latent_generator": torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:latent", self.seed)),
            "penalty_generator": torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:penalty", self.seed)),
            "noise_generator": torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:output", self.seed)),
            "input_noise_generator": torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:input", self.seed)),
            "model_generator": torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:model", self.seed)),
            "eval_generator": torch.Generator(device=self.device).manual_seed(_seed_int(f"{tag_prefix}:eval", self.seed)),
            "require_latent_damping": recipe.latent_damping_max_rate > 0,
        }
        if self.spec["prior"]["kind"] == "mog":
            opts["prior_noise_generator"] = torch.Generator(device=self.device).manual_seed(
                _seed_int(f"{tag_prefix}:prior_noise", self.seed)
            )
        return GANTrainer(recipe, g_mod, d_mod, **opts)

    @staticmethod
    def _clone_generator(src: torch.Generator | None) -> torch.Generator | None:
        if src is None:
            return None
        cloned = torch.Generator(device=src.device)
        cloned.set_state(src.get_state().clone())
        return cloned

    def _make_cloned_rng_trainer(
        self,
        recipe: Recipe,
        g_mod: nn.Module,
        d_mod: nn.Module,
        prior: nn.Module,
        *,
        source: GANTrainer,
    ) -> GANTrainer:
        """Clone initial RNG stream states from ``source`` into independent twin generators."""
        opts: dict[str, Any] = {
            "prior": prior,
            "seed": self.seed,
            "max_steps": self.limit,
            "require_latent_damping": recipe.latent_damping_max_rate > 0,
        }
        for name in (
            "latent_generator",
            "prior_noise_generator",
            "penalty_generator",
            "noise_generator",
            "input_noise_generator",
            "model_generator",
            "eval_generator",
        ):
            cloned = self._clone_generator(getattr(source, name, None))
            if cloned is not None:
                opts[name] = cloned
        return GANTrainer(recipe, g_mod, d_mod, **opts)

    @property
    def completed_steps(self) -> int:
        return int(self.trainer.completed_steps)

    @property
    def batch_sequence_sha256(self) -> str:
        return self._batch_sequence_sha256

    @torch.no_grad()
    def _eval_gen_clean(self, gen_mod: nn.Module, z: torch.Tensor) -> torch.Tensor:
        modes = [(m, m.training) for m in gen_mod.modules()]
        try:
            gen_mod.eval()
            return gen_mod(z).detach().float()
        finally:
            for m, flag in modes:
                m.training = flag

    def step(self) -> dict[str, Any]:
        if self.kind == "fl_law44_scale_shock":
            shock_step = int(self.spec["model"]["shock_step"])
            real_cpu = sample_shock_target(
                self.recipe.batch_size,
                self.data_rng,
                step=self.completed_steps + 1,
                shock_step=shock_step,
                device="cpu",
                eval_phase=False,
            )
        else:
            apply_mask = self.kind == "fl_law44_highd_sparse_prior_and_shock"
            real_cpu = sample_target_for_kind(
                self.kind, self.recipe.batch_size, self.data_rng, device="cpu", apply_mask=apply_mask
            )
        step_hash = hashlib.sha256(real_cpu.numpy().tobytes()).digest()
        self._batch_sequence_sha256 = hashlib.sha256(
            bytes.fromhex(self._batch_sequence_sha256) + step_hash
        ).hexdigest()
        real_dev = real_cpu.to(self.device)

        y_id_pre = y_ood_pre = None
        if self.kind == "fl_law41_trust_rare1pct_tanh":
            y_id_pre = self._eval_gen_clean(self.trainer.G, self._z_id_probe)
            y_ood_pre = self._eval_gen_clean(self.trainer.G, self._z_ood_probe)

        out = self.trainer.step(real_dev)

        if self.kind == "fl_law41_trust_rare1pct_tanh" and y_id_pre is not None and y_ood_pre is not None:
            y_id_post = self._eval_gen_clean(self.trainer.G, self._z_id_probe)
            y_ood_post = self._eval_gen_clean(self.trainer.G, self._z_ood_probe)
            d_id = float((y_id_post - y_id_pre).square().mean().sqrt().item()) / max(self._data_coord_rms, 1e-6)
            d_ood = float((y_ood_post - y_ood_pre).square().mean().sqrt().item()) / max(self._data_coord_rms, 1e-6)
            ratio = d_ood / max(d_id, 1e-4) if math.isfinite(d_id) and math.isfinite(d_ood) else float("nan")
            self._window_step_disps.append(d_id)
            self._window_ood_ratios.append(ratio)
            if len(self._window_step_disps) > 64:
                self._window_step_disps = self._window_step_disps[-64:]
                self._window_ood_ratios = self._window_ood_ratios[-64:]

        if self.kind in ("fl_law42_deep_resbn_and_conv1x1", "fl_law42_deep_resbn_and_conv1x1_v2"):
            stiff_fn = getattr(self.trainer.opt_g, "current_stiff_gate", None)
            stiff_v = float(stiff_fn()) if callable(stiff_fn) else 0.0
            self._window_stiff_gates.append(stiff_v)
            if len(self._window_stiff_gates) > 64:
                self._window_stiff_gates = self._window_stiff_gates[-64:]
            self.aux_trainers["conv1x1"].step(real_dev.unsqueeze(-1).unsqueeze(-1))

        if self.kind == "fl_law44_highd_sparse_prior_and_shock":
            shock_step = int(self.spec["model"]["shock_step"])
            shock_batch = sample_shock_target(
                128,
                self._aux_data_rngs["scale_shock"],
                step=self.completed_steps,
                shock_step=shock_step,
                device="cpu",
                eval_phase=False,
            )
            self.aux_trainers["scale_shock"].step(shock_batch.to(self.device))

        return {k: float(v.item()) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    def _eval_stream(self, seed: int, *, purpose: str = "samples") -> torch.Generator:
        if self._components is not None and purpose == "samples":
            return self._components.streams.generator("eval", component="live", purpose="samples")
        return torch.Generator(device=self.device).manual_seed(_seed_int(f"{self.kind}:eval:{purpose}:{seed}", self.seed))

    def _record_history_point(self, step_now: int, values: dict[str, float]) -> None:
        steps = self._history["steps"]
        if steps and steps[-1] == step_now:
            for k, v in values.items():
                series = self._history.setdefault(k, [v] * len(steps))
                series[-1] = float(v)
        else:
            steps.append(step_now)
            for k, v in values.items():
                self._history.setdefault(k, []).append(float(v))

    @torch.no_grad()
    def observe(self, n: int | None = None, seed: int = 10000) -> dict[str, Any]:
        eval_n = int(self.spec["eval_samples"] if n is None else n)
        eval_stream = self._eval_stream(int(seed), purpose="samples")
        eval_state_before = (
            eval_stream.get_state().clone()
            if self.kind in ("fl_law42_deep_resbn_and_conv1x1", "fl_law42_deep_resbn_and_conv1x1_v2")
            else None
        )
        ref_rng = torch.Generator(device="cpu").manual_seed(_seed_int(f"{self.kind}:ref", self.seed))
        step_now = self.completed_steps
        if self.kind == "fl_law44_scale_shock":
            shock_step = int(self.spec["model"]["shock_step"])
            ref = sample_shock_target(
                min(4096, max(1024, eval_n)),
                ref_rng,
                step=step_now,
                shock_step=shock_step,
                device="cpu",
                eval_phase=True,
            )
        else:
            ref = sample_target_for_kind(self.kind, min(4096, max(1024, eval_n)), ref_rng, device="cpu", apply_mask=False)
        samples = self.trainer.sample(eval_n, output_noise=False, generator=eval_stream)

        if self.kind == "fl_law41_trust_rare1pct_tanh":
            step_max = max(self._window_step_disps) if self._window_step_disps else 0.0
            ood_ratios = sorted(self._window_ood_ratios) if self._window_ood_ratios else [1.0]
            ood_p95 = ood_ratios[min(len(ood_ratios) - 1, int(0.95 * (len(ood_ratios) - 1)))]
            pts_cpu = samples.detach().cpu().float()
            centers = torch.tensor(_RARE1PCT_CENTERS, dtype=torch.float32)
            if pts_cpu.ndim == 2 and pts_cpu.shape[1] == 2 and len(pts_cpu) > 0 and bool(torch.isfinite(pts_cpu).all()):
                d_m, nr_m = torch.cdist(pts_cpu, centers).min(dim=1)
                hq_m = d_m <= (3.5 * _RARE1PCT_SIGMA)
                rare_rec = float(((nr_m == 4) & hq_m).float().mean().item() / _RARE1PCT_WEIGHTS[4])
            else:
                rare_rec = 0.0
            self._record_history_point(step_now, {"rare_mode_recall": rare_rec})
            return score_fl_law41_trust_rare1pct_tanh(
                samples,
                realized_step_max=step_max,
                ood_disp_ratio=ood_p95,
                reference=ref,
                history=self._history,
            )

        if self.kind == "fl_law42_swap_deep_critic_16l":
            valley_sep, critic_r_stable = measure_deep_critic_diagnostics(self.trainer.D, device=self.device)
            pts_cpu = samples.detach().cpu().float()
            centers8, _ = _ring8_centers_and_valleys("cpu")
            if pts_cpu.ndim == 2 and pts_cpu.shape[1] == 2 and len(pts_cpu) > 0 and bool(torch.isfinite(pts_cpu).all()):
                hq_now = float((torch.cdist(pts_cpu, centers8).min(dim=1).values <= (3.5 * _RING8_SIGMA)).float().mean().item())
            else:
                hq_now = 0.0
            self._record_history_point(
                step_now,
                {"hq": hq_now, "diag_critic_r_stable": float(critic_r_stable)},
            )
            return score_fl_law42_swap_deep_critic_16l(
                samples,
                valley_sep=valley_sep,
                critic_r_stable=critic_r_stable,
                reference=ref,
                history=self._history,
            )

        if self.kind in ("fl_law42_deep_resbn_and_conv1x1", "fl_law42_deep_resbn_and_conv1x1_v2"):
            conv_stream = torch.Generator(device=eval_stream.device)
            assert eval_state_before is not None
            conv_stream.set_state(eval_state_before)
            samples_conv = self.aux_trainers["conv1x1"].sample(eval_n, output_noise=False, generator=conv_stream)
            if samples_conv.ndim == 4:
                samples_conv = samples_conv.reshape(samples_conv.shape[0], -1)
            stiff_mean = (
                float(sum(self._window_stiff_gates) / len(self._window_stiff_gates))
                if self._window_stiff_gates
                else 0.0
            )
            d_max_s1 = measure_critic_max_s1(self.trainer.D)
            rec_obj = getattr(self.trainer.opt_g, "_unified_v11_record", None)
            s_therm = float(getattr(rec_obj, "_thermal_scale", 1.0)) if rec_obj is not None else 1.0
            mf = _highd_manifold("resbn")
            sw1_now = (
                sliced_wasserstein_1(samples, ref, mf["sw_proj"])
                if samples.ndim == 2 and samples.shape[1] == _RESBN_DIM and bool(torch.isfinite(samples).all())
                else 9.9
            )
            self._record_history_point(step_now, {"sw1": float(sw1_now) if math.isfinite(sw1_now) else 9.9})
            scorer_fn = (
                score_fl_law42_deep_resbn_and_conv1x1_v2
                if self.kind == "fl_law42_deep_resbn_and_conv1x1_v2"
                else score_fl_law42_deep_resbn_and_conv1x1
            )
            return scorer_fn(
                samples,
                samples_conv1x1=samples_conv,
                stiff_gate_mean=stiff_mean,
                d_max_s1=d_max_s1,
                s_therm=s_therm,
                reference=ref,
                history=self._history,
            )

        if self.kind == "fl_law44_scale_shock":
            shock_step = int(self.spec["model"]["shock_step"])
            s_pts_cpu = samples.detach().cpu().float()
            ref_unit = ref.detach().cpu().float() / _SHOCK_BASE_SCALE
            sw1_now = (
                sliced_wasserstein_1(s_pts_cpu / _SHOCK_BASE_SCALE, ref_unit, _shock_2d_proj())
                if s_pts_cpu.ndim == 2 and s_pts_cpu.shape[1] == 2 and bool(torch.isfinite(s_pts_cpu).all())
                else 9.9
            )
            self._record_history_point(step_now, {"shock_sw1": float(sw1_now) if math.isfinite(sw1_now) else 9.9})
            return score_fl_law44_scale_shock(
                samples,
                step=step_now,
                shock_step=shock_step,
                reference=ref,
                history=self._history,
            )

        if self.kind == "fl_law44_highd_sparse_prior_and_shock":
            shock_step = int(self.spec["model"]["shock_step"])
            shock_stream = self._eval_stream(int(seed) + 29, purpose="shock_samples")
            shock_samples = self.aux_trainers["scale_shock"].sample(
                min(2048, max(256, eval_n)), output_noise=False, generator=shock_stream
            )
            shock_ref_rng = torch.Generator(device="cpu").manual_seed(_seed_int(f"{self.kind}:shock:ref", self.seed))
            shock_ref = sample_shock_target(
                min(2048, max(256, eval_n)),
                shock_ref_rng,
                step=step_now,
                shock_step=shock_step,
                device="cpu",
                eval_phase=True,
            )
            # Non-gating getattr-safe diagnostics:
            mean_dz = 0.0
            prior_pk = 0.0
            base_prior = unwrap_prior(self.trainer.prior)
            if self._initial_prior_z is not None and hasattr(base_prior, "z"):
                z_now = base_prior.z.detach()
                z0 = self._initial_prior_z.to(device=z_now.device, dtype=z_now.dtype)
                if bool(torch.isfinite(z_now).all()):
                    mean_dz = float((z_now - z0).norm(dim=1).mean().item())
                opt_state = getattr(self.trainer.opt_g, "state", {})
                pk_state = opt_state.get(base_prior.z, {}) if isinstance(opt_state, dict) else {}
                pk_val = pk_state.get("peak_rms", pk_state.get("peak_rms_val", 0.0))
                prior_pk = float(pk_val.item()) if isinstance(pk_val, torch.Tensor) else float(pk_val)

            mf = _highd_manifold("highd_sparse")
            sw1_now = (
                sliced_wasserstein_1(samples, ref, mf["sw_proj"])
                if samples.ndim == 2 and samples.shape[1] == _HIGHD_DIM and bool(torch.isfinite(samples).all())
                else 9.9
            )
            self._record_history_point(step_now, {"sw1": float(sw1_now) if math.isfinite(sw1_now) else 9.9})
            return score_fl_law44_highd_sparse_prior_and_shock(
                samples,
                shock_samples=shock_samples,
                step=step_now,
                shock_step=shock_step,
                mean_dz=mean_dz,
                prior_peak_rms=prior_pk,
                reference=ref,
                shock_reference=shock_ref,
                history=self._history,
            )

        raise ValueError(f"unsupported four_laws_4 kind: {self.kind!r}")

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "recipe": self.recipe.to_dict(),
            "batch_sequence_sha256": self._batch_sequence_sha256,
            "data_rng": self.data_rng.get_state().clone(),
            "aux_data_rngs": {k: g.get_state().clone() for k, g in self._aux_data_rngs.items()},
            "trainer": self.trainer.state_dict(),
            "aux_trainers": {k: tr.state_dict() for k, tr in self.aux_trainers.items()},
            "initial_prior_z": None if self._initial_prior_z is None else self._initial_prior_z.detach().cpu().clone(),
            "window_step_disps": list(self._window_step_disps),
            "window_ood_ratios": list(self._window_ood_ratios),
            "window_stiff_gates": list(self._window_stiff_gates),
        }
        if isinstance(self.trainer.G, HighDSparseTanhGenerator):
            state["gen_mask_rng"] = self.trainer.G.mask_rng_state()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("case_id") != self.case_id:
            raise ValueError("checkpoint case_id mismatch")
        if state.get("recipe") != self.recipe.to_dict():
            raise ValueError("checkpoint recipe mismatch")
        if "batch_sequence_sha256" not in state or "seed" not in state:
            raise ValueError("checkpoint missing required provenance fields")
        self.seed = int(state["seed"])
        self._batch_sequence_sha256 = str(state["batch_sequence_sha256"])
        self.data_rng.set_state(state["data_rng"].cpu())
        for k, g in self._aux_data_rngs.items():
            g.set_state(state["aux_data_rngs"][k].cpu())
        self.trainer.load_state_dict(state["trainer"])
        for k, tr in self.aux_trainers.items():
            tr.load_state_dict(state["aux_trainers"][k])
        init_z = state.get("initial_prior_z")
        self._initial_prior_z = None if init_z is None else init_z.to(device=self.device, dtype=torch.float32)
        self._window_step_disps = list(state["window_step_disps"])
        self._window_ood_ratios = list(state["window_ood_ratios"])
        self._window_stiff_gates = list(state["window_stiff_gates"])
        if isinstance(self.trainer.G, HighDSparseTanhGenerator):
            self.trainer.G.load_mask_rng_state(state.get("gen_mask_rng"), self.device)


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
    """Construct a public-API ``FourLawsFixture`` for any Law 4 task."""
    del kwargs
    if case_id not in ALL_SPECS:
        raise ValueError(f"unknown four_laws_4 case: {case_id!r}")
    case = next(c for c in list_all_cases() if c["id"] == case_id)
    return FourLawsFixture(
        case,
        recipe_name=recipe_name or "unified_v14",
        recipe=recipe,
        recipe_overrides=recipe_overrides,
        device=device,
        seed=seed,
        max_steps=max_steps,
        components=components,
    )


def render_task_gif(
    case: dict[str, Any],
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    full_budget: bool,
    requested_steps: int,
    final_verdict: str | None = None,
) -> dict[str, Any]:
    """Thin wrapper over ``benchmarks.toy_audit.api_run.render_gif`` for Law 4 observations."""
    from .api_run import render_gif

    return render_gif(
        case,
        records,
        Path(path),
        full_budget=full_budget,
        requested_steps=requested_steps,
        final_verdict=final_verdict,
    )
