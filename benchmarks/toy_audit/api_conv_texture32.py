"""Public-API Tier 1 experiment for 3x32x32 (d=3072) random-phase RGB texture acquisition.

Scientific question: in ambient image dimension ``d = 3 * 32 * 32 = 3072`` at
minibatch size ``B = 64``, real samples drawn from multi-frequency random-phase
texture classes concentrate on a high-dimensional shell whose 25th-percentile
real-to-real nearest-neighbor distance (``~16.1``) exceeds the distance from
flat interior DC class-mean blobs to the real shell (``~15.6``). Can a
convolutional generator and nonlinear feature-pyramid critic acquire both
oriented 12-mode RGB texture classes from a recipe-owned 64-particle
``MoGParticlePrior`` (``sigma=0.1``) within 288 updates—recovering high-frequency
carrier energy, balanced class mass, and within-class phase diversity—rather
than collapsing onto zero-carrier interior pixel averages?
"""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import (
    GANTrainer,
    MoGParticlePrior,
    Recipe,
    get_recipe,
    init,
)

from .reproducibility import DEFAULT_SEED, construction_rng

NUM_ORIENTATIONS = 2
BAND_RMS_AMP = 0.68
NOISE_STD = 0.02
REF_BAND_ENERGY = 0.210
PAIRWISE_AC_SCALE = 21.0

BAND_MODES: tuple[tuple[tuple[int, int], ...], ...] = (
    # Orientation 0: horizontal multi-frequency carrier band (12 half-plane 2D DFT modes).
    ((3, 0), (4, 0), (5, 0), (6, 0), (3, 1), (4, 1), (5, 1), (6, 1), (29, 1), (28, 1), (27, 1), (26, 1)),
    # Orientation 1: vertical multi-frequency carrier band (12 half-plane 2D DFT modes).
    ((0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6), (31, 3), (31, 4), (31, 5), (31, 6)),
)

CLASS_MEANS = torch.tensor(
    [
        [0.25, -0.20, -0.05],
        [-0.20, 0.25, -0.05],
    ],
    dtype=torch.float32,
)

CLASS_AXES = F.normalize(
    torch.tensor(
        [
            [1.0, -0.8, 0.2],
            [-0.8, 1.0, 0.2],
        ],
        dtype=torch.float32,
    ),
    dim=-1,
)

THRESHOLDS: dict[str, list[list]] = {
    "img_conv_texture32": [
        ["finite_fraction", "==", 1.0],
        ["hf_band_ratio", ">=", 0.55],
        ["clean_texture_fraction", ">=", 0.75],
        ["orientations", "==", 2],
        ["mass_tv", "<=", 0.18],
        ["pairwise_phase_diversity", ">=", 0.40],
    ],
}

HOST_DEFINITIONS: dict[str, dict] = {
    "img_conv_texture32": {
        "case_id": "api-img-conv-texture32",
        "kind": "img_conv_texture32",
        "target": (
            "Two equiprobable 3x32x32 (d=3072) random-phase 12-mode oriented RGB texture classes "
            "(horizontal coral-cyan vs vertical emerald-magenta, rms_amp=0.68, noise_std=0.02, clamped to [-1, 1])"
        ),
        "resources": {"num_particles": 64, "z_dim": 8, "batch_size": 64},
        "model": {
            "family": "conv32_residual_upsample_miniresnet_mlp",
            "image_shape": [3, 32, 32],
            "width": 16,
            "critic_mlp_hidden": 64,
        },
        "steps": 288,
    },
}

_BASIS_CACHE: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _band_basis(device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), dtype)
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    pts = torch.arange(32, device=device, dtype=dtype) / 32.0
    v, u = torch.meshgrid(pts, pts, indexing="ij")
    cos_b = torch.empty((NUM_ORIENTATIONS, 12, 32, 32), device=device, dtype=dtype)
    sin_b = torch.empty((NUM_ORIENTATIONS, 12, 32, 32), device=device, dtype=dtype)
    for c_idx, modes in enumerate(BAND_MODES):
        for m_idx, (fy, fx) in enumerate(modes):
            ky = fy if fy <= 16 else fy - 32
            kx = fx if fx <= 16 else fx - 32
            ang = 2.0 * math.pi * (kx * u + ky * v)
            cos_b[c_idx, m_idx] = torch.cos(ang)
            sin_b[c_idx, m_idx] = torch.sin(ang)
    _BASIS_CACHE[key] = (cos_b, sin_b)
    return cos_b, sin_b


def sample_conv_texture32(
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
    fixed_phase: bool = False,
    zero_carrier: bool = False,
    single_orientation: int | None = None,
) -> torch.Tensor:
    """Sample 3x32x32 oriented 12-mode random-phase RGB textures in [-1, 1]."""
    dev = torch.device(device)
    gen_dev = generator.device
    if single_orientation is None:
        cls = torch.randint(0, NUM_ORIENTATIONS, (n,), generator=generator, device=gen_dev)
    else:
        cls = torch.full((n,), int(single_orientation), dtype=torch.long, device=gen_dev)
    if fixed_phase:
        phi = torch.linspace(0.0, 2.0 * math.pi, 13, device=gen_dev, dtype=torch.float32)[:12].unsqueeze(0).expand(n, 12)
    else:
        phi = 2.0 * math.pi * torch.rand((n, 12), generator=generator, device=gen_dev, dtype=torch.float32)
    eps = NOISE_STD * torch.randn((n, 3, 32, 32), generator=generator, device=gen_dev, dtype=torch.float32)

    cls = cls.to(dev)
    phi = phi.to(dev)
    eps = eps.to(dev)
    cos_b, sin_b = _band_basis(dev, dtype=torch.float32)
    field = (
        cos_b[cls] * torch.cos(phi).view(n, 12, 1, 1)
        - sin_b[cls] * torch.sin(phi).view(n, 12, 1, 1)
    ).sum(dim=1) / math.sqrt(12.0)
    if zero_carrier:
        field = torch.zeros_like(field)

    means = CLASS_MEANS.to(dev)[cls].view(n, 3, 1, 1)
    axes = CLASS_AXES.to(dev)[cls].view(n, 3, 1, 1)
    return (means + BAND_RMS_AMP * axes * field.unsqueeze(1) + eps).clamp(-1.0, 1.0)


def sample_target_for_kind(
    kind: str,
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if kind == "img_conv_texture32":
        return sample_conv_texture32(n, generator, device=device)
    raise ValueError(f"unsupported conv_texture32 task kind: {kind!r}")


class ResidualUpConvGenerator32(nn.Module):
    """Nearest-upsample residual 3x32x32 RGB generator with linear DC shortcut."""

    def __init__(self, z_dim: int = 8, width: int = 16) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.width = int(width)
        self.fc = nn.Linear(self.z_dim, self.width * 4 * 4)
        self.c1 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.c2 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.c3 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.to_rgb = nn.Conv2d(self.width, 3, 3, padding=1)
        self.skip_rgb = nn.Linear(self.z_dim, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fc(z), 0.2).reshape(z.shape[0], self.width, 4, 4)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = h + 0.5 * F.leaky_relu(self.c1(h), 0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = h + 0.5 * F.leaky_relu(self.c2(h), 0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = h + 0.5 * F.leaky_relu(self.c3(h), 0.2)
        dc = self.skip_rgb(z).reshape(z.shape[0], 3, 1, 1)
        return torch.tanh(self.to_rgb(h) + dc)


class MiniResNetCritic32(nn.Module):
    """Strided residual 3x32x32 feature critic with non-inplace MLP head for 2nd-order penalties."""

    def __init__(self, width: int = 16, mlp_hidden: int = 64) -> None:
        super().__init__()
        self.width = int(width)
        self.mlp_hidden = int(mlp_hidden)
        self.stem = nn.Conv2d(3, self.width, 3, padding=1)
        self.b1 = nn.Conv2d(self.width, self.width, 3, stride=2, padding=1)
        self.s1 = nn.Conv2d(self.width, self.width, 1, stride=2)
        self.b2 = nn.Conv2d(self.width, self.width * 2, 3, stride=2, padding=1)
        self.s2 = nn.Conv2d(self.width, self.width * 2, 1, stride=2)
        self.b3 = nn.Conv2d(self.width * 2, self.width * 2, 3, stride=2, padding=1)
        self.s3 = nn.Conv2d(self.width * 2, self.width * 2, 1, stride=2)
        in_feat = self.width * 4 + 6
        self.head = nn.Sequential(
            nn.Linear(in_feat, self.mlp_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x.reshape(x.shape[0], 3, 32, 32)
        dc = img.mean(dim=(-2, -1))
        ac_rms = (img - dc.unsqueeze(-1).unsqueeze(-1)).square().mean(dim=(-2, -1)).add(1e-6).sqrt()
        h = F.leaky_relu(self.stem(img), 0.2)
        h = self.s1(h) + F.leaky_relu(self.b1(h), 0.2)
        h = self.s2(h) + F.leaky_relu(self.b2(h), 0.2)
        h = self.s3(h) + F.leaky_relu(self.b3(h), 0.2)
        h_mean = h.mean(dim=(-2, -1))
        h_rms = (h - h_mean.unsqueeze(-1).unsqueeze(-1)).square().mean(dim=(-2, -1)).add(1e-6).sqrt()
        feat = torch.cat((h_mean, h_rms, dc, ac_rms), dim=1)
        return self.head(feat)


def build_models_for_spec(spec: dict, *, z_dim: int) -> tuple[nn.Module, nn.Module]:
    model_cfg = spec["model"]
    if model_cfg["family"] != "conv32_residual_upsample_miniresnet_mlp":
        raise ValueError(f"unsupported conv_texture32 model family: {model_cfg['family']!r}")
    width = int(model_cfg["width"])
    mlp_hidden = int(model_cfg["critic_mlp_hidden"])
    return (
        ResidualUpConvGenerator32(z_dim=int(z_dim), width=width),
        MiniResNetCritic32(width=width, mlp_hidden=mlp_hidden),
    )


def _check_thresholds(metrics: dict, kind: str) -> list[str]:
    failed = []
    for name, op, bound in THRESHOLDS[kind]:
        val = metrics.get(name)
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            failed.append(f"{name} {op} {bound}")
            continue
        ok = (
            val <= bound
            if op == "<="
            else val >= bound
            if op == ">="
            else val == bound
            if op == "=="
            else False
        )
        if not ok:
            failed.append(f"{name} {op} {bound}")
    return failed


def _band_energy_matrix_f64(xf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dc = xf.mean(dim=(-2, -1))
    ac = xf - dc.unsqueeze(-1).unsqueeze(-1)
    fft2 = torch.fft.fft2(ac, dim=(-2, -1)) / (32.0 * 32.0)
    axes_f64 = CLASS_AXES.to(dtype=torch.float64, device=xf.device)
    energies = []
    for c_idx in range(NUM_ORIENTATIONS):
        modes = BAND_MODES[c_idx]
        fy_idx = [m[0] for m in modes]
        fx_idx = [m[1] for m in modes]
        coeffs = fft2[:, :, fy_idx, fx_idx]
        proj = (coeffs * axes_f64[c_idx].view(1, 3, 1)).sum(dim=1)
        e_band = 2.0 * (proj.real.square() + proj.imag.square()).sum(dim=1)
        energies.append(e_band)
    return torch.stack(energies, dim=1), dc, ac


def score_conv_texture32(
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict:
    """Evaluate 3x32x32 generated samples for HF carrier energy, orientation coverage, balance, and phase diversity."""
    kind = "img_conv_texture32"
    ref = reference
    if ref is None:
        ref_gen = torch.Generator(device="cpu").manual_seed(92001)
        ref = sample_conv_texture32(256, ref_gen, device="cpu")
    ref_cpu = ref.detach().cpu().to(torch.float32)
    ref_e, _, _ = _band_energy_matrix_f64(ref_cpu.to(torch.float64))
    ref_band_2d = ref_e.to(torch.float32)

    if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or samples.shape[1:] != (3, 32, 32) or len(samples) == 0:
        finite_fraction = 0.0
        if isinstance(samples, torch.Tensor) and samples.numel() > 0:
            finite_fraction = float(torch.isfinite(samples.reshape(len(samples), -1)).all(dim=1).float().mean().item())
            if samples.ndim != 4 or samples.shape[1:] != (3, 32, 32):
                finite_fraction = 0.0
        metrics = {
            "finite_fraction": finite_fraction,
            "hf_band_ratio": 0.0,
            "clean_texture_fraction": 0.0,
            "orientations": 0,
            "mass_tv": 1.0,
            "pairwise_phase_diversity": 0.0,
            "dc_mean_err": 2.0,
            "band_purity_mean": 0.0,
            "sample_count": int(len(samples)) if isinstance(samples, torch.Tensor) and samples.ndim >= 1 else 0,
        }
        failed = _check_thresholds(metrics, kind)
        blank = torch.zeros((8, 3, 32, 32), dtype=torch.float32)
        zeros_2d = torch.zeros((2, 2), dtype=torch.float32)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {
                    "kind": "image",
                    "title": "3x32x32 RGB textures (target vs live prior)",
                    "target": ref_cpu[:8],
                    "samples": blank,
                    "vmin": -1.0,
                    "vmax": 1.0,
                    "row_labels": ["Target", "Live G(z)"],
                },
                {
                    "kind": "scatter",
                    "title": "Oriented 2D DFT band energy (E_horiz, E_vert)",
                    "target": ref_band_2d,
                    "samples": zeros_2d,
                    "xlim": [-0.02, 0.42],
                    "ylim": [-0.02, 0.42],
                },
            ],
        }

    x = samples.detach().to(device="cpu", dtype=torch.float64)
    n = x.shape[0]
    finite_mask = torch.isfinite(x.reshape(n, -1)).all(dim=1)
    finite_fraction = float(finite_mask.to(torch.float64).mean().item())
    if finite_fraction < 1.0:
        metrics = {
            "finite_fraction": finite_fraction,
            "hf_band_ratio": 0.0,
            "clean_texture_fraction": 0.0,
            "orientations": 0,
            "mass_tv": 1.0,
            "pairwise_phase_diversity": 0.0,
            "dc_mean_err": 2.0,
            "band_purity_mean": 0.0,
            "sample_count": n,
        }
        failed = _check_thresholds(metrics, kind)
        blank = torch.zeros((8, 3, 32, 32), dtype=torch.float32)
        zeros_2d = torch.zeros((2, 2), dtype=torch.float32)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {
                    "kind": "image",
                    "title": "3x32x32 RGB textures (target vs live prior)",
                    "target": ref_cpu[:8],
                    "samples": blank,
                    "vmin": -1.0,
                    "vmax": 1.0,
                    "row_labels": ["Target", "Live G(z)"],
                },
                {
                    "kind": "scatter",
                    "title": "Oriented 2D DFT band energy (E_horiz, E_vert)",
                    "target": ref_band_2d,
                    "samples": zeros_2d,
                    "xlim": [-0.02, 0.42],
                    "ylim": [-0.02, 0.42],
                },
            ],
        }

    e_mat, dc, ac = _band_energy_matrix_f64(x)
    means_f64 = CLASS_MEANS.to(torch.float64)
    best_e, pred_cls = e_mat.max(dim=1)
    total_ac = ac.square().mean(dim=(-2, -1)).sum(dim=1).clamp_min(1e-12)

    hf_band_ratio = float((best_e / REF_BAND_ENERGY).clamp(max=1.5).mean().item())
    dc_err = (dc - means_f64[pred_cls]).norm(dim=1)
    band_purity = best_e / total_ac
    clean_mask = (best_e >= 0.38 * REF_BAND_ENERGY) & (band_purity >= 0.38) & (dc_err <= 0.22)
    clean_texture_fraction = float(clean_mask.to(torch.float64).mean().item())

    cls_scores = e_mat / REF_BAND_ENERGY - 2.0 * torch.cdist(dc, means_f64)
    assigned_cls = cls_scores.argmax(dim=1)
    counts_clean = torch.bincount(assigned_cls[clean_mask], minlength=NUM_ORIENTATIONS).to(torch.float64)
    shares_all = torch.bincount(assigned_cls, minlength=NUM_ORIENTATIONS).to(torch.float64) / max(n, 1)
    target_share = 1.0 / NUM_ORIENTATIONS
    mass_tv = float(0.5 * (shares_all - target_share).abs().sum().item())
    min_share = 0.45 * target_share
    orientations = int((counts_clean / max(n, 1) >= min_share).sum().item())

    div_scores = []
    flat_ac = ac.reshape(n, -1)
    for c_idx in range(NUM_ORIENTATIONS):
        c_mask = clean_mask & (assigned_cls == c_idx)
        if int(c_mask.sum().item()) < 6:
            div_scores.append(0.0)
            continue
        sub = flat_ac[c_mask][:96]
        pdist = torch.pdist(sub, p=2.0)
        div_scores.append(float(pdist.median().item() / PAIRWISE_AC_SCALE))
    pairwise_phase_diversity = float(min(div_scores))

    metrics = {
        "finite_fraction": 1.0,
        "hf_band_ratio": hf_band_ratio,
        "clean_texture_fraction": clean_texture_fraction,
        "orientations": orientations,
        "mass_tv": mass_tv,
        "pairwise_phase_diversity": pairwise_phase_diversity,
        "dc_mean_err": float(dc_err.mean().item()),
        "band_purity_mean": float(band_purity.clamp(max=1.0).mean().item()),
        "sample_count": n,
    }
    failed = _check_thresholds(metrics, kind)
    samp_cpu = samples.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    samp_band_2d = e_mat.to(torch.float32).clamp(0.0, 0.42)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                "kind": "image",
                "title": "3x32x32 RGB textures (target vs live prior)",
                "target": ref_cpu[:8],
                "samples": samp_cpu[:8],
                "vmin": -1.0,
                "vmax": 1.0,
                "row_labels": ["Target", "Live G(z)"],
                "caption": "Sharp random-phase 12-mode RGB textures vs interior zero-carrier DC mean collapse.",
            },
            {
                "kind": "scatter",
                "title": "Oriented 2D DFT band energy (E_horiz, E_vert)",
                "target": ref_band_2d.clamp(0.0, 0.42),
                "samples": samp_band_2d,
                "xlim": [-0.02, 0.42],
                "ylim": [-0.02, 0.42],
                "aspect": "equal",
                "xlabel": "E_horizontal",
                "ylabel": "E_vertical",
                "caption": "True textures populate the two carrier axes (~0.21); pixel-OT mean collapse piles at (0, 0).",
            },
        ],
    }


def score_conv_texture32_kind(
    kind: str,
    samples: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> dict:
    if kind != "img_conv_texture32":
        raise ValueError(f"unsupported conv_texture32 kind: {kind!r}")
    return score_conv_texture32(samples, reference=reference)


def conv_texture32_oracle_controls() -> dict[str, dict]:
    """Analytic positive and destructive negative controls proving metric selectivity."""
    controls: dict[str, dict] = {}
    ref = sample_conv_texture32(256, torch.Generator(device="cpu").manual_seed(92001), device="cpu")

    # 1. Positive control: 8 independent N=256 draws from the true target distribution.
    pos_draws = [
        score_conv_texture32(
            sample_conv_texture32(256, torch.Generator(device="cpu").manual_seed(92100 + idx), device="cpu"),
            reference=ref,
        )
        for idx in range(8)
    ]
    pos_failed = [b for draw in pos_draws for b in draw["failed_bounds"]]
    controls["true_target_oracle"] = {
        "expected_pass": True,
        "passed": all(draw["passed"] for draw in pos_draws),
        "failed_bounds": list(dict.fromkeys(pos_failed)),
    }

    # 2. Negative control: zero-carrier interior DC mean blobs (the exact high-d pixel-OT shell trap).
    zero_blobs = sample_conv_texture32(
        256, torch.Generator(device="cpu").manual_seed(92201), device="cpu", zero_carrier=True
    )
    neg_zero = score_conv_texture32(zero_blobs, reference=ref)
    controls["zero_carrier_dc_blob_negative"] = {
        "expected_pass": False,
        "passed": neg_zero["passed"],
        "failed_bounds": neg_zero["failed_bounds"],
    }

    # 3. Negative control: fixed-phase template memorization (passes HF energy + both classes, fails phase diversity).
    fixed_phase = sample_conv_texture32(
        256, torch.Generator(device="cpu").manual_seed(92202), device="cpu", fixed_phase=True
    )
    neg_fixed = score_conv_texture32(fixed_phase, reference=ref)
    controls["fixed_phase_memorization_negative"] = {
        "expected_pass": False,
        "passed": neg_fixed["passed"],
        "failed_bounds": neg_fixed["failed_bounds"],
    }

    # 4. Negative control: single-orientation collapse (crisp diverse textures on class 0 only).
    single_cls = sample_conv_texture32(
        256, torch.Generator(device="cpu").manual_seed(92203), device="cpu", single_orientation=0
    )
    neg_single = score_conv_texture32(single_cls, reference=ref)
    controls["single_orientation_collapse_negative"] = {
        "expected_pass": False,
        "passed": neg_single["passed"],
        "failed_bounds": neg_single["failed_bounds"],
    }

    # 5. Negative control: off-band white noise with matched class DC means.
    noise_gen = torch.Generator(device="cpu").manual_seed(92204)
    cls_idx = torch.randint(0, NUM_ORIENTATIONS, (256,), generator=noise_gen)
    white = (
        CLASS_MEANS[cls_idx].view(256, 3, 1, 1)
        + 0.45 * torch.randn((256, 3, 32, 32), generator=noise_gen)
    ).clamp(-1.0, 1.0)
    neg_white = score_conv_texture32(white, reference=ref)
    controls["off_band_white_noise_negative"] = {
        "expected_pass": False,
        "passed": neg_white["passed"],
        "failed_bounds": neg_white["failed_bounds"],
    }

    # 6. Negative control: 7x7 spatial box-blur damage applied to true target samples.
    clean_draw = sample_conv_texture32(256, torch.Generator(device="cpu").manual_seed(92205), device="cpu")
    blurred = F.avg_pool2d(clean_draw, kernel_size=7, stride=1, padding=3)
    neg_blur = score_conv_texture32(blurred, reference=ref)
    controls["box_blur_damage_negative"] = {
        "expected_pass": False,
        "passed": neg_blur["passed"],
        "failed_bounds": neg_blur["failed_bounds"],
    }

    return controls


def list_cases() -> list[dict]:
    return [
        {
            "id": "api-img-conv-texture32",
            "title": "3x32x32 random-phase 12-mode RGB texture acquisition (high-d concentration gate)",
            "goal": (
                "Acquire both equiprobable 3x32x32 (d=3072) random-phase 12-mode RGB texture orientations "
                "from a 64-particle MoGParticlePrior (sigma=0.1) within 288 updates, recovering carrier energy "
                "(hf_band_ratio>=0.55, clean_texture_fraction>=0.75), balanced classes, and phase diversity>=0.40."
            ),
            "kind": "image",
            "scope": "Public API Tier 1 gate: 3x32x32 feature-critic vs interior pixel-OT shell concentration",
            "legacy_ids": ["tier1-img-conv-texture32"],
            "default_steps": 288,
            "batch_size": 64,
            "eval_samples": 256,
            "evaluation_observations": 24,
            "terminal_observations": 5,
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS["img_conv_texture32"]},
            "sampling": "Clean live samples from recipe-owned MoGParticlePrior (M=64, sigma=0.1) without output noise",
        }
    ]


_CASE_TO_KIND = {spec["case_id"]: kind for kind, spec in HOST_DEFINITIONS.items()}


def _seed_int(tag: str, seed: int) -> int:
    digest = hashlib.sha256(f"api_conv_texture32:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


class ConvTexture32Fixture:
    """Public-API GANTrainer fixture for 3x32x32 random-phase RGB texture acquisition."""

    api_components = ("GANTrainer", "MoGParticlePrior", "Recipe", "init.deterministic_orthogonal_")

    def __init__(
        self,
        case: dict,
        *,
        recipe_name: str = "empf",
        recipe: Recipe | None = None,
        recipe_overrides: dict | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
    ) -> None:
        self.case = deepcopy(case)
        self.case_id = case["id"]
        self.kind = _CASE_TO_KIND[self.case_id]
        self.spec = deepcopy(HOST_DEFINITIONS[self.kind])
        self.device = torch.device(device)
        self.seed = int(seed)
        self.limit = int(case["default_steps"] if max_steps is None else max_steps)

        res = self.spec["resources"]
        base_recipe = get_recipe(recipe_name) if recipe is None else recipe
        updates = {
            "num_particles": int(res["num_particles"]),
            "z_dim": int(res["z_dim"]),
            "batch_size": int(res["batch_size"]),
            "prior_kind": "mog",
            "sigma_rel": 0.1,
            "standardize": False,
            **(recipe_overrides or {}),
        }
        if base_recipe.total_steps is not None:
            updates["total_steps"] = self.limit
        self.recipe = base_recipe.replace(**updates)

        with construction_rng(self.seed, self.device):
            G, D = build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)
        G, D = G.to(self.device), D.to(self.device)
        for comp, mod in (("generator", G), ("discriminator", D)):
            seeds = {
                name: _seed_int(f"init:{comp}:{name}", self.seed)
                for name, p in mod.named_parameters()
                if p.requires_grad and p.numel()
            }
            init.deterministic_orthogonal_(mod, parameter_seeds=seeds)

        prior_rng = torch.Generator(device=self.device).manual_seed(_seed_int("init:prior:locations", self.seed))
        prior = MoGParticlePrior(
            num_particles=self.recipe.num_particles,
            z_dim=self.recipe.z_dim,
            sigma=0.1,
            standardize=False,
            init_std=1.0,
            device=self.device,
            dtype=torch.float32,
            learnable=True,
            generator=prior_rng,
        )
        if hasattr(prior, "z") and isinstance(prior.z, nn.Parameter):
            init.deterministic_orthogonal_(prior, parameter_seeds={"z": _seed_int("init:prior:z", self.seed)})
        self.recipe.apply_prior_policy(prior)

        self.data_rng = torch.Generator(device="cpu").manual_seed(_seed_int("data:target:training", self.seed))
        self._batch_hasher = hashlib.sha256()
        self.trainer = GANTrainer(
            self.recipe,
            G,
            D,
            prior=prior,
            seed=self.seed,
            max_steps=self.limit,
            latent_generator=torch.Generator(device=self.device).manual_seed(_seed_int("prior:latent", self.seed)),
            prior_noise_generator=torch.Generator(device=self.device).manual_seed(_seed_int("noise:prior", self.seed)),
            penalty_generator=torch.Generator(device=self.device).manual_seed(_seed_int("noise:penalty", self.seed)),
            noise_generator=torch.Generator(device=self.device).manual_seed(_seed_int("noise:output", self.seed)),
            input_noise_generator=torch.Generator(device=self.device).manual_seed(_seed_int("noise:input", self.seed)),
            model_generator=torch.Generator(device=self.device).manual_seed(_seed_int("noise:model", self.seed)),
            eval_generator=torch.Generator(device=self.device).manual_seed(_seed_int("eval:sampler", self.seed)),
            require_latent_damping=self.recipe.latent_damping_max_rate > 0,
        )
        self.metadata = {
            "checkpoint_capabilities": {
                "round_trip": True,
                "preserves_rng": True,
                "owner": "GANTrainer.state_dict + caller data_rng",
            }
        }

    @property
    def completed_steps(self) -> int:
        return self.trainer.completed_steps

    @property
    def batch_sequence_sha256(self) -> str:
        return self._batch_hasher.hexdigest()

    def step(self) -> dict:
        real_cpu = sample_target_for_kind(self.kind, self.recipe.batch_size, self.data_rng, device="cpu")
        self._batch_hasher.update(real_cpu.numpy().tobytes())
        out = self.trainer.step(real_cpu.to(self.device))
        return {k: float(v.item()) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    @torch.no_grad()
    def observe(self, n: int = 256, seed: int = 10000) -> dict:
        eval_stream = torch.Generator(device=self.device).manual_seed(int(seed))
        ref_stream = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        samples = self.trainer.sample(int(n), output_noise=False, generator=eval_stream)
        reference = sample_target_for_kind(self.kind, int(n), ref_stream, device="cpu")
        return score_conv_texture32_kind(self.kind, samples, reference=reference)

    def state_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "data_rng": self.data_rng.get_state(),
            "trainer": self.trainer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("case_id") != self.case_id:
            raise ValueError("checkpoint case_id mismatch")
        self.data_rng.set_state(state["data_rng"].cpu())
        self.trainer.load_state_dict(state["trainer"])


def build_case(
    case_id: str,
    *,
    device: str | torch.device = "cpu",
    seed: int = DEFAULT_SEED,
    recipe_name: str | None = None,
    max_steps: int | None = None,
    recipe: Recipe | None = None,
    recipe_overrides: dict | None = None,
    **kwargs,
) -> ConvTexture32Fixture:
    del kwargs
    if case_id not in _CASE_TO_KIND:
        raise ValueError(f"unknown conv_texture32 toy case: {case_id!r}")
    return ConvTexture32Fixture(
        next(c for c in list_cases() if c["id"] == case_id),
        recipe_name=recipe_name or "empf",
        recipe=recipe,
        recipe_overrides=recipe_overrides,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
