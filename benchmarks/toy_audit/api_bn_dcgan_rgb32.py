"""Public-API Radford 32x32 BN-DCGAN convergence + B=256 throughput diagnostic guard.

Hermetic, 100% offline 32x32 RGB continuous multi-scale 10-class scene benchmark
matching the continuous natural-image spatial regime (pooled 8x8
``med_nn / med_all > 0.50``, matching CIFAR-10 rather than discrete templates).

Uses a standard Radford 32x32 BN-DCGAN (``BatchNorm2d`` in both generator and
critic, ``z_dim=128``, ``batch_size=256``, ``deterministic_orthogonal_`` init,
protocol seed ``0``, isolated checkpointable RNG streams, zero host-side
repairs) executed strictly through the public ``particlegan.GANTrainer`` /
``Recipe`` API.

Evaluates served clean live samples via a frozen, offline, deterministic
multi-scale convolutional + spatial + radial-FFT feature extractor (zero
downloads, zero external ImageNet weights) with oracle and destructive negative
controls, while recording synchronized per-step timing for the CUDA B=256
step-latency guard and exporting an actual-training ``goal.gif``.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.cifar_toy.metrics import (
    compute_class_coverage,
    compute_critic_diagnostics,
    compute_fid,
    compute_pixel_diversity_ratio,
    feature_moments,
)
from particlegan import (
    GANTrainer,
    Recipe,
    get_recipe,
    init,
    unwrap_prior,
)

from .reproducibility import DEFAULT_SEED, construction_rng

VERSION = "bn-dcgan-rgb32-v1"
TASK_ID = "bn_dcgan_rgb32_b256_diagnostic"
LEGACY_ID = "bn-dcgan-rgb32-b256-diagnostic"
NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

# ---------------------------------------------------------------------------
# Frozen Task Specification & Pre-Run Numerical Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: list[list[Any]] = [
    ["finite_fraction", "==", 1.0],
    ["feature_fid", "<=", 55.0],
    ["pixel_diversity_ratio", ">=", 0.70],
    ["pixel_diversity_ratio", "<=", 1.30],
    ["class_coverage", ">=", 0.90],
    ["class_mass_tv", "<=", 0.25],
    ["spectral_l1", "<=", 0.04],
]

CUDA_LATENCY_GATE: dict[str, Any] = {
    "metric": "update_ms_median",
    "max_ms": 38.0,
    "warmup_updates": 8,
    "enforced_on": "cuda",
}

SPEC: dict[str, Any] = {
    "task_id": TASK_ID,
    "case_id": TASK_ID,
    "kind": TASK_ID,
    "title": "Radford 32x32 BN-DCGAN (B=256) continuous RGB convergence & throughput diagnostic",
    "goal": (
        "Train a standard Radford 32x32 BN-DCGAN (BatchNorm2d in both G and D, z_dim=128, B=256) "
        "on a hermetic continuous multi-scale 10-class 32x32 RGB distribution (pooled 8x8 "
        "med_nn/med_all > 0.50) without BN critic collapse, mode collapse, spectral degradation, "
        "or O(B^2) CUDA step-latency regressions."
    ),
    "target": (
        "10-class continuous multi-scale 32x32x3 RGB scenes in [-1, 1] with continuous shift, "
        "scale, rotation, foreground/background color variation, smooth 2D lighting gradients, "
        "multi-frequency sinusoidal texture, and pixel noise (pooled 8x8 med_nn/med_all > 0.50)."
    ),
    "resources": {
        "num_particles": 1024,
        "z_dim": 128,
        "batch_size": 256,
    },
    "prior": {
        "kind": "mog",
        "sigma": 0.1,
        "standardize": False,
    },
    "model": {
        "family": "radford_bn_dcgan32",
        "image_size": IMAGE_SIZE,
        "channels": IMAGE_CHANNELS,
        "base_channels": 64,
        "generator_batchnorm": True,
        "discriminator_batchnorm": True,
    },
    "data": {
        "distribution": "continuous_multiscale_rgb32_10class_v1",
        "num_classes": NUM_CLASSES,
        "min_pooled8_med_nn_ratio": 0.50,
        "reference_samples": 2048,
        "reference_seed": 777,
        "evaluator_seed": 202611,
    },
    "steps": 1200,
    "warmup_updates": 8,
    "eval_samples": 1024,
    "observations": 12,
    "minimum_stable_checks": 3,
    "sampling_law": "public_prior_without_output_noise",
    "cuda_latency_gate": deepcopy(CUDA_LATENCY_GATE),
    "view_kind": "image",
}

# ---------------------------------------------------------------------------
# Continuous Natural-Image-Regime 32x32 RGB Data Distribution
# ---------------------------------------------------------------------------

_PALETTES = torch.tensor(
    [
        [0.86, 0.22, 0.20],  # 0: crimson disk
        [0.20, 0.80, 0.28],  # 1: emerald diagonal bands
        [0.20, 0.38, 0.90],  # 2: cobalt horizon wave
        [0.90, 0.80, 0.18],  # 3: gold vertical column
        [0.80, 0.24, 0.82],  # 4: magenta ring
        [0.18, 0.82, 0.84],  # 5: cyan corner quadrant
        [0.92, 0.50, 0.18],  # 6: amber frame
        [0.54, 0.30, 0.86],  # 7: violet diamond
        [0.26, 0.66, 0.48],  # 8: teal lattice weave
        [0.78, 0.78, 0.86],  # 9: silver cross
    ],
    dtype=torch.float32,
)

_CACHED_GRID: tuple[torch.Tensor, torch.Tensor] | None = None


def _coord_grid() -> tuple[torch.Tensor, torch.Tensor]:
    global _CACHED_GRID
    if _CACHED_GRID is None:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=torch.float32),
            indexing="ij",
        )
        _CACHED_GRID = (yy, xx)
    return _CACHED_GRID


def sample_continuous_rgb32(
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
    return_labels: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n`` continuous multi-scale 10-class 32x32 RGB images in ``[-1, 1]``.

    Consumes ONLY the caller-supplied CPU ``generator`` (zero ambient RNG use).
    Continuous shift, scale, orientation, color tint, 2D lighting gradients, and
    sinusoidal texture keep pooled 8x8 ``med_nn / med_all > 0.50`` (natural-image
    spatial geometry rather than discrete sprites).
    """
    count = int(n)
    if count < 1:
        raise ValueError("sample count must be positive")
    if generator.device.type != "cpu":
        raise ValueError("sample_continuous_rgb32 requires a deterministic CPU torch.Generator")

    yy, xx = _coord_grid()
    cls = torch.randint(0, NUM_CLASSES, (count,), generator=generator)
    sx = torch.empty((count, 1, 1), dtype=torch.float32).uniform_(-0.42, 0.42, generator=generator)
    sy = torch.empty((count, 1, 1), dtype=torch.float32).uniform_(-0.42, 0.42, generator=generator)
    scale = torch.empty((count, 1, 1), dtype=torch.float32).uniform_(0.68, 1.38, generator=generator)
    ang = torch.empty((count, 1, 1), dtype=torch.float32).uniform_(-0.55, 0.55, generator=generator)
    ca, sa = torch.cos(ang), torch.sin(ang)

    x0 = (xx.unsqueeze(0) - sx) / scale
    y0 = (yy.unsqueeze(0) - sy) / scale
    x = ca * x0 - sa * y0
    y = sa * x0 + ca * y0
    r = torch.sqrt(x * x + y * y + 1e-6)

    masks = torch.zeros((count, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    for c_idx in range(NUM_CLASSES):
        sel = cls == c_idx
        if not bool(sel.any()):
            continue
        xc, yc, rc = x[sel], y[sel], r[sel]
        if c_idx == 0:
            val = torch.sigmoid((0.52 - rc) * 9.0)
        elif c_idx == 1:
            val = 0.5 + 0.5 * torch.sin(5.0 * (xc + yc))
        elif c_idx == 2:
            val = torch.sigmoid((0.36 - (yc + 0.16 * torch.sin(4.5 * xc)).abs()) * 9.0)
        elif c_idx == 3:
            val = torch.sigmoid((0.34 - xc.abs()) * 9.0)
        elif c_idx == 4:
            val = torch.sigmoid((0.20 - (rc - 0.52).abs()) * 10.0)
        elif c_idx == 5:
            val = torch.sigmoid(xc * 7.0) * torch.sigmoid(yc * 7.0)
        elif c_idx == 6:
            val = torch.sigmoid((torch.maximum(xc.abs(), yc.abs()) - 0.54) * 9.0)
        elif c_idx == 7:
            val = torch.sigmoid((0.68 - (xc.abs() + yc.abs())) * 8.0)
        elif c_idx == 8:
            val = 0.5 + 0.5 * torch.sin(5.5 * xc) * torch.sin(5.5 * yc)
        else:
            val = torch.sigmoid((0.24 - torch.minimum(xc.abs(), yc.abs())) * 10.0)
        masks[sel] = val

    fg_base = _PALETTES[cls].unsqueeze(-1).unsqueeze(-1)
    tint = torch.empty((count, 3, 1, 1), dtype=torch.float32).uniform_(-0.20, 0.20, generator=generator)
    fg = (fg_base + tint).clamp(0.04, 0.98)

    bg_base = (1.0 - fg_base) * 0.34 + 0.14
    bg_tint = torch.empty((count, 3, 1, 1), dtype=torch.float32).uniform_(-0.18, 0.18, generator=generator)
    gx = torch.empty((count, 3, 1, 1), dtype=torch.float32).uniform_(-0.28, 0.28, generator=generator)
    gy = torch.empty((count, 3, 1, 1), dtype=torch.float32).uniform_(-0.28, 0.28, generator=generator)
    bg = (bg_base + bg_tint + gx * xx.view(1, 1, 32, 32) + gy * yy.view(1, 1, 32, 32)).clamp(0.02, 0.96)

    freq = torch.empty((count, 1, 1, 1), dtype=torch.float32).uniform_(2.5, 9.5, generator=generator)
    phase = torch.empty((count, 1, 1, 1), dtype=torch.float32).uniform_(0.0, 2.0 * math.pi, generator=generator)
    tex_ang = torch.empty((count, 1, 1, 1), dtype=torch.float32).uniform_(0.0, math.pi, generator=generator)
    tex = 0.085 * torch.sin(
        freq * (torch.cos(tex_ang) * xx.view(1, 1, 32, 32) + torch.sin(tex_ang) * yy.view(1, 1, 32, 32)) + phase
    )
    noise = torch.randn((count, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator, dtype=torch.float32) * 0.035

    imgs = (bg * (1.0 - masks.unsqueeze(1)) + fg * masks.unsqueeze(1) + tex + noise).clamp(0.0, 1.0)
    imgs = imgs * 2.0 - 1.0

    dev = torch.device(device)
    if return_labels:
        return imgs.to(dev), cls.to(dev)
    return imgs.to(dev)


@torch.no_grad()
def pooled8_med_nn_ratio(images: torch.Tensor, *, max_samples: int = 256) -> float:
    """Scale-invariant pooled 8x8 ``med_nn / med_all`` geometric cluster ratio.

    Matches the energy-preserving 8x8 average-pool geometry used to distinguish
    discrete pixel-template sprites (~0.03) from continuous natural images
    (CIFAR-10: ~0.54-0.65).
    """
    x = images[: int(max_samples)].detach().cpu().float()
    if x.ndim != 4 or x.shape[0] < 4:
        raise ValueError("pooled8_med_nn_ratio requires at least 4 NCHW images")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    ho, wo = min(8, h), min(8, w)
    ph, pw = (-h) % ho, (-w) % wo
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    pooled = x.reshape(x.shape[0], x.shape[1], ho, x.shape[2] // ho, wo, x.shape[3] // wo).mean((3, 5))
    flat = pooled.reshape(x.shape[0], -1) * math.sqrt(float(h * w) / float(max(1, ho * wo)))
    dists = torch.cdist(flat, flat, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    n = dists.shape[0]
    off_diag = dists[~torch.eye(n, dtype=torch.bool)]
    nn_d = dists.clone().fill_diagonal_(float("inf")).amin(dim=-1)
    med_all = float(off_diag.median().item())
    if med_all <= 1e-12:
        return 0.0
    return float(nn_d.median().item()) / med_all


# ---------------------------------------------------------------------------
# Standard Radford 32x32 BN-DCGAN Architecture
# ---------------------------------------------------------------------------


class RadfordBNGenerator(nn.Module):
    """Standard Radford 32x32 BN-DCGAN Generator (ConvTranspose2d + BatchNorm2d + ReLU -> Tanh)."""

    def __init__(self, z_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        c = int(base_channels)
        self.z_dim = int(z_dim)
        self.base_channels = c
        self.fc = nn.Linear(self.z_dim, (c * 4) * 4 * 4, bias=False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(c, IMAGE_CHANNELS, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(z.shape[0], self.base_channels * 4, 4, 4)
        return self.conv(h)


class RadfordBNCritic(nn.Module):
    """Standard Radford 32x32 BN-DCGAN Critic (Conv2d + BatchNorm2d + LeakyReLU -> scalar)."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = int(base_channels)
        self.base_channels = c
        self.main = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, c, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
            nn.Linear((c * 4) * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).reshape(-1, 1)


def build_models(
    spec: dict[str, Any] | None = None,
    *,
    z_dim: int | None = None,
) -> tuple[RadfordBNGenerator, RadfordBNCritic]:
    """Construct the declared Radford 32x32 BN-DCGAN pair."""
    cfg = SPEC if spec is None else spec
    model_cfg = cfg["model"]
    latent = int(cfg["resources"]["z_dim"] if z_dim is None else z_dim)
    base_ch = int(model_cfg.get("base_channels", 64))
    return RadfordBNGenerator(z_dim=latent, base_channels=base_ch), RadfordBNCritic(base_channels=base_ch)


# ---------------------------------------------------------------------------
# Offline Deterministic Multi-Scale Feature & Spectral Evaluator
# ---------------------------------------------------------------------------


class OfflineMultiScaleImageEvaluator(nn.Module):
    """Hermetic, frozen, deterministic multi-scale 32x32 RGB feature extractor.

    Combines 4x4 pooled RGB layout (48), color moments & cross-channel
    covariances (9), finite-difference spatial gradient RMS (6), 4-band radial
    2D-FFT log-power spectrum (12), and frozen deterministic-orthogonal 3-stage
    Conv2d/GELU activations (136) into a 211-dim feature vector. Zero downloads,
    zero external weights, and completely independent of the trained critic.
    """

    def __init__(self, seed: int = 202611) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=True)
        init.deterministic_orthogonal_(self, seed=int(seed), strict=True)
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

        fy = torch.fft.fftfreq(IMAGE_SIZE).view(IMAGE_SIZE, 1)
        fx = torch.fft.rfftfreq(IMAGE_SIZE).view(1, IMAGE_SIZE // 2 + 1)
        rad = torch.sqrt(fy * fy + fx * fx)
        edges = (0.0, 0.08, 0.18, 0.34, 0.75)
        masks = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = ((rad >= lo) & (rad < hi)).float()
            masks.append(mask / mask.sum().clamp_min(1.0))
        self.register_buffer("_fft_masks", torch.stack(masks, dim=0))

    @torch.no_grad()
    def radial_spectrum(self, imgs: torch.Tensor) -> torch.Tensor:
        x = imgs.float()
        spec = torch.fft.rfft2(x, norm="ortho")
        logp = torch.log1p(spec.real.square() + spec.imag.square())
        return torch.einsum("nchw,bhw->ncb", logp, self._fft_masks.to(x.device)).reshape(x.shape[0], -1)

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor, *, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        dev = self.conv1.weight.device
        feats, specs = [], []
        n = int(imgs.shape[0])
        for start in range(0, n, int(batch_size)):
            x = imgs[start : start + int(batch_size)].to(device=dev, dtype=torch.float32)
            h1 = F.gelu(self.conv1(x))
            h2 = F.gelu(self.conv2(h1))
            h3 = F.gelu(self.conv3(h2))
            p4 = F.avg_pool2d(x, 8).reshape(x.shape[0], -1)
            ch_mean = x.mean(dim=(-2, -1))
            ch_std = x.std(dim=(-2, -1), unbiased=False)
            xc = x - ch_mean.unsqueeze(-1).unsqueeze(-1)
            cov01 = (xc[:, 0] * xc[:, 1]).mean(dim=(-2, -1)).unsqueeze(1)
            cov02 = (xc[:, 0] * xc[:, 2]).mean(dim=(-2, -1)).unsqueeze(1)
            cov12 = (xc[:, 1] * xc[:, 2]).mean(dim=(-2, -1)).unsqueeze(1)
            gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).square().mean(dim=(-2, -1)).sqrt()
            gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).square().mean(dim=(-2, -1)).sqrt()
            spec = self.radial_spectrum(x)
            conv_f = torch.cat(
                [
                    h1.mean(dim=(-2, -1)),
                    h2.mean(dim=(-2, -1)),
                    h3.mean(dim=(-2, -1))[:, :32],
                    h3.std(dim=(-2, -1), unbiased=False)[:, :32],
                ],
                dim=1,
            )
            feat = torch.cat([p4, ch_mean, ch_std, cov01, cov02, cov12, gx, gy, spec, conv_f], dim=1)
            feats.append(feat.cpu())
            specs.append(spec.cpu())
        return torch.cat(feats, dim=0), torch.cat(specs, dim=0)


_CACHED_EVALUATOR: OfflineMultiScaleImageEvaluator | None = None
_CACHED_REFERENCE: dict[str, Any] | None = None
_CACHED_ORACLES: dict[str, dict[str, Any]] | None = None


def offline_evaluator() -> OfflineMultiScaleImageEvaluator:
    global _CACHED_EVALUATOR
    if _CACHED_EVALUATOR is None:
        _CACHED_EVALUATOR = OfflineMultiScaleImageEvaluator(seed=int(SPEC["data"]["evaluator_seed"]))
    return _CACHED_EVALUATOR


def reference_statistics() -> dict[str, Any]:
    """Cached deterministic real-distribution moments, class centroids, and spectrum."""
    global _CACHED_REFERENCE
    if _CACHED_REFERENCE is None:
        ref_n = int(SPEC["data"]["reference_samples"])
        ref_seed = int(SPEC["data"]["reference_seed"])
        gen = torch.Generator(device="cpu").manual_seed(ref_seed)
        ref_imgs, ref_lbls = sample_continuous_rgb32(ref_n, gen, device="cpu", return_labels=True)
        ev = offline_evaluator()
        ref_f, ref_s = ev.extract(ref_imgs)
        f_mu = ref_f.mean(dim=0, keepdim=True)
        f_std = ref_f.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-3)
        ref_fn = (ref_f - f_mu) / f_std
        mu_r, cov_r = feature_moments(ref_fn)
        centroids = torch.stack([ref_fn[ref_lbls == c].mean(dim=0) for c in range(NUM_CLASSES)], dim=0)
        med_nn = pooled8_med_nn_ratio(ref_imgs[:256], max_samples=256)
        _CACHED_REFERENCE = {
            "images": ref_imgs,
            "labels": ref_lbls,
            "f_mu": f_mu,
            "f_std": f_std,
            "mu_r": mu_r,
            "cov_r": cov_r,
            "centroids": centroids,
            "spec_mean": ref_s.mean(dim=0),
            "pooled8_med_nn_ratio": float(med_nn),
        }
    return _CACHED_REFERENCE


def _evaluate_bounds(
    metrics: dict[str, float],
    thresholds: list[list[Any]] = THRESHOLDS,
) -> tuple[bool, list[str], dict[str, float], float]:
    failed: list[str] = []
    margins: dict[str, float] = {}
    for name, op, bound in thresholds:
        val = metrics.get(name)
        b_val = float(bound)
        key = f"{name} {op} {b_val:g}"
        if val is None or not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(float(val)):
            failed.append(f"{name}: nonfinite")
            margins[key] = float("-inf")
            continue
        f_val = float(val)
        if op == "<=":
            margin = b_val - f_val
            ok = f_val <= b_val
        elif op == ">=":
            margin = f_val - b_val
            ok = f_val >= b_val
        elif op == "==":
            margin = -abs(f_val - b_val)
            ok = f_val == b_val
        else:
            raise ValueError(f"unsupported operator: {op!r}")
        margins[key] = float(margin)
        if not ok:
            failed.append(key)
    finite_margins = [m for m in margins.values() if math.isfinite(m)]
    min_margin = min(finite_margins) if (finite_margins and len(finite_margins) == len(margins)) else float("-inf")
    return (len(failed) == 0), failed, margins, float(min_margin)


def _display_rgb01(imgs: torch.Tensor) -> np.ndarray:
    """Convert ``[-1, 1]`` NCHW images to ``[0, 1]`` while preserving NaN/Inf faithfully."""
    arr = imgs.detach().cpu().float()
    finite = torch.isfinite(arr)
    mapped = torch.where(finite, arr.clamp(-1.0, 1.0).add(1.0).mul(0.5), arr)
    return mapped.numpy()


def score_samples(
    samples: torch.Tensor,
    *,
    critic: nn.Module | None = None,
    reference: torch.Tensor | None = None,
    update_ms_median: float | None = None,
    update_ms_p95: float | None = None,
    images_per_second: float | None = None,
    include_views: bool = True,
) -> dict[str, Any]:
    """Score served 32x32 RGB samples against the frozen offline multi-scale gate."""
    ref = reference_statistics()
    ref_imgs = ref["images"] if reference is None else reference.detach().cpu().float()
    fake = samples.detach().cpu().float()
    if fake.ndim != 4 or tuple(fake.shape[1:]) != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected NCHW (N, 3, 32, 32) samples, got {tuple(fake.shape)}")

    finite_mask = torch.isfinite(fake)
    finite_fraction = float(finite_mask.float().mean().item())
    class_probs = np.zeros((NUM_CLASSES,), dtype=np.float64)

    if finite_fraction < 1.0:
        metrics: dict[str, float] = {
            "finite_fraction": finite_fraction,
            "feature_fid": float("inf"),
            "pixel_diversity_ratio": 0.0,
            "class_coverage": 0.0,
            "class_mass_tv": 1.0,
            "spectral_l1": float("inf"),
            "class_entropy_norm": 0.0,
            "min_class_mass": 0.0,
            "data_pooled8_med_nn_ratio": float(ref["pooled8_med_nn_ratio"]),
            "sample_count": float(fake.shape[0]),
        }
    else:
        ev = offline_evaluator()
        ff, fs = ev.extract(fake)
        ffn = (ff - ref["f_mu"]) / ref["f_std"]
        mu_f, cov_f = feature_moments(ffn)
        fid = compute_fid(ref["mu_r"], ref["cov_r"], mu_f, cov_f)
        pdr = compute_pixel_diversity_ratio(ref["images"], fake)
        cc = compute_class_coverage(ffn, ref["centroids"], min_mass_threshold=0.02)
        spec_l1 = float((fs.mean(dim=0) - ref["spec_mean"]).abs().mean().item())
        dists = torch.cdist(ffn.float(), ref["centroids"].float())
        assigned = dists.argmin(dim=1)
        counts = torch.bincount(assigned, minlength=NUM_CLASSES).float()
        class_probs = (counts / max(1.0, float(counts.sum().item()))).numpy()
        metrics = {
            "finite_fraction": finite_fraction,
            "feature_fid": float(fid),
            "pixel_diversity_ratio": float(pdr),
            "class_coverage": float(cc["class_coverage"] / float(NUM_CLASSES)),
            "class_mass_tv": float(cc["class_mass_tv"]),
            "spectral_l1": float(spec_l1),
            "class_entropy_norm": float(cc["class_entropy_norm"]),
            "min_class_mass": float(cc["min_class_mass"]),
            "data_pooled8_med_nn_ratio": float(ref["pooled8_med_nn_ratio"]),
            "sample_count": float(fake.shape[0]),
        }

    if critic is not None and finite_fraction == 1.0:
        n_diag = min(256, int(fake.shape[0]), int(ref_imgs.shape[0]))
        if n_diag >= 2:
            diag = compute_critic_diagnostics(critic, ref_imgs[:n_diag], fake[:n_diag])
            for k, v in diag.items():
                if math.isfinite(float(v)):
                    metrics[k] = float(v)

    if update_ms_median is not None and math.isfinite(float(update_ms_median)):
        metrics["update_ms_median"] = float(update_ms_median)
    if update_ms_p95 is not None and math.isfinite(float(update_ms_p95)):
        metrics["update_ms_p95"] = float(update_ms_p95)
    if images_per_second is not None and math.isfinite(float(images_per_second)):
        metrics["images_per_second"] = float(images_per_second)

    passed, failed_bounds, margins, min_margin = _evaluate_bounds(metrics, THRESHOLDS)
    result: dict[str, Any] = {
        "passed": passed,
        "failed_bounds": failed_bounds,
        "metrics": metrics,
        "margins": margins,
        "min_margin": min_margin,
    }
    if include_views:
        timing_note = (
            f"; step={metrics['update_ms_median']:.2f}ms"
            if "update_ms_median" in metrics
            else ""
        )
        result["views"] = [
            {
                "kind": "image",
                "title": "32x32 RGB Continuous Scenes vs Radford BN-DCGAN",
                "target": _display_rgb01(ref_imgs[:8]),
                "samples": _display_rgb01(fake[:8]),
                "vmin": 0.0,
                "vmax": 1.0,
                "row_labels": ["Target RGB32", "BN-DCGAN output"],
                "caption": (
                    f"pooled8 med_nn/med_all={ref['pooled8_med_nn_ratio']:.3f}; "
                    f"FID={metrics['feature_fid']:.2f}; pdr={metrics['pixel_diversity_ratio']:.3f}"
                    f"{timing_note}"
                ),
            },
            {
                "kind": "bar",
                "title": "10-Class Feature-Centroid Mass Distribution",
                "target": np.full((NUM_CLASSES,), 1.0 / float(NUM_CLASSES), dtype=np.float64),
                "samples": class_probs,
                "xlabel": "Scene Class",
                "ylabel": "Mass fraction",
                "caption": (
                    f"cov={metrics['class_coverage']:.2f}; tv={metrics['class_mass_tv']:.3f}; "
                    f"spec_l1={metrics['spectral_l1']:.4f}"
                ),
            },
        ]
    return result


def oracle_controls() -> dict[str, dict[str, Any]]:
    """Run deterministic positive and destructive controls to verify scorer calibration."""
    global _CACHED_ORACLES
    if _CACHED_ORACLES is not None:
        return deepcopy(_CACHED_ORACLES)

    ref = reference_statistics()
    med_nn = float(ref["pooled8_med_nn_ratio"])
    min_ratio = float(SPEC["data"]["min_pooled8_med_nn_ratio"])

    gen = torch.Generator(device="cpu").manual_seed(999)
    holdout, h_lbls = sample_continuous_rgb32(1024, gen, device="cpu", return_labels=True)
    c0 = holdout[h_lbls == 0]
    single_class = c0.repeat((1024 // max(1, int(c0.shape[0]))) + 1, 1, 1, 1)[:1024]
    c01 = holdout[h_lbls < 2]
    two_classes = c01.repeat((1024 // max(1, int(c01.shape[0]))) + 1, 1, 1, 1)[:1024]
    mean_img = holdout.mean(dim=0, keepdim=True)
    low_div = 0.80 * mean_img + 0.20 * holdout
    nonfinite = holdout.clone()
    nonfinite[0, 0, 0, 0] = float("nan")

    cases: list[tuple[str, torch.Tensor, bool]] = [
        ("oracle_real_holdout", holdout, True),
        ("negative_constant_zero", torch.zeros_like(holdout), False),
        ("negative_single_class", single_class, False),
        ("negative_two_classes", two_classes, False),
        ("negative_low_diversity_collapsed", low_div, False),
        ("negative_uniform_noise", torch.empty_like(holdout).uniform_(-1.0, 1.0, generator=gen), False),
        ("negative_gaussian_noise", torch.randn(holdout.shape, generator=gen).clamp(-1.0, 1.0), False),
        (
            "negative_blurred_real",
            F.interpolate(F.avg_pool2d(holdout, 8), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False),
            False,
        ),
        ("negative_memorized_8", holdout[:8].repeat(128, 1, 1, 1), False),
        ("negative_nonfinite", nonfinite, False),
    ]

    controls: dict[str, dict[str, Any]] = {
        "data_natural_image_regime": {
            "expected_pass": True,
            "passed": bool(med_nn > min_ratio),
            "pooled8_med_nn_ratio": med_nn,
            "min_required": min_ratio,
        }
    }
    for name, tensor, expected_pass in cases:
        scored = score_samples(tensor, include_views=False)
        controls[name] = {
            "expected_pass": expected_pass,
            "passed": bool(scored["passed"]),
            "failed_bounds": list(scored["failed_bounds"]),
            "metrics": {
                k: v
                for k, v in scored["metrics"].items()
                if k in ("finite_fraction", "feature_fid", "pixel_diversity_ratio", "class_coverage", "class_mass_tv", "spectral_l1")
            },
        }

    _CACHED_ORACLES = controls
    return deepcopy(controls)


# ---------------------------------------------------------------------------
# Public-API Checkpointable Fixture
# ---------------------------------------------------------------------------


def _seed_int(tag: str, seed: int) -> int:
    digest = hashlib.sha256(f"api_bn_dcgan_rgb32:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def list_cases() -> list[dict[str, Any]]:
    """Return ``api_contract.validate_case``-compliant case metadata."""
    return [
        {
            "id": TASK_ID,
            "title": SPEC["title"],
            "goal": SPEC["goal"],
            "kind": SPEC["view_kind"],
            "scope": (
                "Hermetic offline Radford 32x32 BN-DCGAN (B=256) convergence & CUDA throughput "
                "diagnostic guard on continuous natural-image-regime RGB scenes."
            ),
            "legacy_ids": [LEGACY_ID],
            "default_steps": int(SPEC["steps"]),
            "batch_size": int(SPEC["resources"]["batch_size"]),
            "eval_samples": int(SPEC["eval_samples"]),
            "evaluation_observations": int(SPEC["observations"]),
            "terminal_observations": int(SPEC["minimum_stable_checks"]),
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS},
            "sampling": "Clean live public prior samples without additive output noise.",
            "default_recipe": "axiom",
            "protocol_seed": DEFAULT_SEED,
            "checkpoint_capabilities": {
                "round_trip": True,
                "preserves_rng": True,
                "exact_continuation": True,
                "owner": "GANTrainer.state_dict + isolated CPU data RNGs",
            },
        }
    ]


class BNDCGANFixture:
    """Checkpointable public-API ``GANTrainer`` fixture for Radford 32x32 BN-DCGAN."""

    api_components = ("GANTrainer", "Recipe", "init.deterministic_orthogonal_")

    def __init__(
        self,
        case_id: str = TASK_ID,
        *,
        recipe_name: str | None = "axiom",
        recipe: Recipe | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
        components: Any = None,
        spec_override: dict[str, Any] | None = None,
    ) -> None:
        if case_id != TASK_ID:
            raise ValueError(f"unknown BN-DCGAN guard case: {case_id!r}")
        self.task_id = case_id
        self.case_id = case_id
        self.kind = case_id
        self.spec = deepcopy(SPEC if spec_override is None else spec_override)
        self.case = list_cases()[0]
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.seed = int(seed)
        self.limit = int(self.spec["steps"] if max_steps is None else max_steps)
        if type(self.limit) is not int or self.limit < 1 or self.seed < 0:
            raise ValueError("max_steps must be positive and seed must be nonnegative")
        self.components = components

        if components is not None:
            self.recipe = components.recipe
            g_mod = components.construct(
                lambda: build_models(self.spec, z_dim=self.recipe.z_dim)[0],
                component="generator",
            ).to(self.device)
            d_mod = components.construct(
                lambda: build_models(self.spec, z_dim=self.recipe.z_dim)[1],
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
            base_recipe = get_recipe(recipe_name or "axiom") if recipe is None else recipe
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
                g_mod, d_mod = build_models(self.spec, z_dim=self.recipe.z_dim)
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
        # Runtime-only latency telemetry (never enters training state_dict).
        self._step_times_ms: list[float] = []
        self._warmup_updates = int(self.spec.get("warmup_updates", CUDA_LATENCY_GATE["warmup_updates"]))

        self.metadata = {
            "checkpoint_capabilities": {
                "round_trip": True,
                "preserves_rng": True,
                "exact_continuation": True,
                "owner": "GANTrainer.state_dict + isolated CPU data RNGs",
            }
        }

    @property
    def completed_steps(self) -> int:
        return int(self.trainer.completed_steps)

    @property
    def batch_sequence_sha256(self) -> str:
        return self._batch_sequence_sha256

    def timing_summary(self, *, warmup_updates: int | None = None) -> dict[str, float | int | None]:
        """Compute empirical median/p95 step latency and throughput over post-warmup updates."""
        warmup = self._warmup_updates if warmup_updates is None else int(warmup_updates)
        measured = self._step_times_ms[warmup:] if len(self._step_times_ms) > warmup else list(self._step_times_ms)
        if not measured:
            return {
                "warmup_updates": warmup,
                "measured_updates": 0,
                "update_ms_median": None,
                "update_ms_p95": None,
                "images_per_second": None,
            }
        arr = np.asarray(measured, dtype=np.float64)
        mean_ms = float(arr.mean())
        return {
            "warmup_updates": warmup,
            "measured_updates": len(measured),
            "update_ms_median": float(np.percentile(arr, 50)),
            "update_ms_p95": float(np.percentile(arr, 95)),
            "images_per_second": float(1000.0 * int(self.recipe.batch_size) / mean_ms) if mean_ms > 0 else None,
        }

    def step(self, *, collect_stats: bool = False) -> dict[str, Any]:
        b_sz = int(self.recipe.batch_size)
        real_d_cpu = sample_continuous_rgb32(b_sz, self.data_rng, device="cpu")
        real_g_cpu = sample_continuous_rgb32(b_sz, self.generator_data_rng, device="cpu")
        step_hash = hashlib.sha256(
            real_d_cpu.numpy().tobytes() + real_g_cpu.numpy().tobytes()
        ).digest()
        self._batch_sequence_sha256 = hashlib.sha256(
            bytes.fromhex(self._batch_sequence_sha256) + step_hash
        ).hexdigest()

        real_d_dev = real_d_cpu.to(self.device)
        real_g_dev = real_g_cpu.to(self.device)
        dev_ctx = torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        with dev_ctx:
            _sync_device(self.device)
            t0 = time.perf_counter()
            out = self.trainer.step(
                real_d_dev,
                generator_real=real_g_dev,
                collect_stats=collect_stats,
            )
            _sync_device(self.device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        self._step_times_ms.append(float(dt_ms))

        result = {
            k: float(v.item()) if isinstance(v, torch.Tensor) and v.numel() == 1 else v
            for k, v in out.items()
        }
        result["duration_ms"] = float(dt_ms)
        return result

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
                chunks: list[torch.Tensor] = []
                rem = eval_n
                chunk_b = min(256, max(1, eval_n))
                while rem > 0:
                    take = min(chunk_b, rem)
                    chunks.append(
                        self.trainer.sample(take, output_noise=False, generator=sample_stream).detach().cpu().float()
                    )
                    rem -= take
                samples = torch.cat(chunks, dim=0)
                ref_preview = sample_continuous_rgb32(min(256, max(16, eval_n)), ref_stream, device="cpu")
                timing = self.timing_summary()
                return score_samples(
                    samples,
                    critic=self.trainer.D,
                    reference=ref_preview,
                    update_ms_median=timing["update_ms_median"],
                    update_ms_p95=timing["update_ms_p95"],
                    images_per_second=timing["images_per_second"],
                    include_views=True,
                )
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
            "trainer": self.trainer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        required = {
            "version",
            "case_id",
            "seed",
            "limit",
            "recipe",
            "batch_sequence_sha256",
            "data_rng",
            "generator_data_rng",
            "trainer",
        }
        if not isinstance(state, dict) or required - set(state):
            raise ValueError("BNDCGANFixture checkpoint is missing required state fields")
        if state.get("version") != VERSION or state.get("case_id") != self.case_id:
            raise ValueError("BNDCGANFixture checkpoint version or case_id mismatch")
        if state.get("recipe") != self.recipe.to_dict():
            raise ValueError("BNDCGANFixture checkpoint recipe mismatch")
        self.seed = int(state["seed"])
        self.limit = int(state["limit"])
        self._batch_sequence_sha256 = str(state["batch_sequence_sha256"])
        self.data_rng.set_state(state["data_rng"].cpu())
        self.generator_data_rng.set_state(state["generator_data_rng"].cpu())
        self.trainer.load_state_dict(state["trainer"])


Fixture = BNDCGANFixture


def build_case(
    case_id: str = TASK_ID,
    *,
    device: str | torch.device = "cpu",
    seed: int = DEFAULT_SEED,
    recipe_name: str | None = None,
    max_steps: int | None = None,
    recipe: Recipe | None = None,
    recipe_overrides: dict[str, Any] | None = None,
    components: Any = None,
    spec_override: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BNDCGANFixture:
    """Build a public-API ``BNDCGANFixture``."""
    del kwargs
    if case_id != TASK_ID:
        raise ValueError(f"unknown BN-DCGAN guard toy case: {case_id!r}")
    return BNDCGANFixture(
        case_id,
        recipe_name=recipe_name or "axiom",
        recipe=recipe,
        recipe_overrides=recipe_overrides,
        device=device,
        seed=seed,
        max_steps=max_steps,
        components=components,
        spec_override=spec_override,
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
    """Delegate animated goal-GIF export to ``benchmarks.toy_audit.api_run.render_gif``."""
    from .api_run import render_gif

    case = list_cases()[0] if isinstance(case_or_task_id, str) else case_or_task_id
    steps = int(case["default_steps"] if requested_steps is None else requested_steps)
    return render_gif(
        case,
        records,
        Path(path),
        full_budget=bool(full_budget),
        requested_steps=steps,
        final_verdict=final_verdict,
    )
