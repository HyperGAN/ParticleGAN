"""Public-API ByteGAN benchmarks on the 8-bit binary hypercube {-1, +1}^8.

Scientific questions:
1. ``bytegan_hypercube8``: Can an exact-capacity zero-sigma ``ParticlePrior``
   (256 discrete particles, ``z_dim=8``, no MoG kernel smoothing) and an
   unbounded MLP generator learn all 256 ``uint8`` bytes on ``{-1, +1}^8`` in
   a crisp 1-to-1 bijection without particle collisions, barycentric ghosts, or
   exploded bit amplitudes?
2. ``bytegan_bigan_dictionary``: Can a joint BiGAN ``(G(z), E(x), D(x, z))``
   over the same 256 zero-sigma particles learn a bidirectional 1-to-1 byte
   dictionary with exact forward generation, exact canonical byte
   reconstruction ``G(E(x)) == x``, and 1-to-1 encoder particle indexing?
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Any

import torch
from torch import nn

from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import (
    GANTrainer,
    InputNoise,
    ParticlePrior,
    Recipe,
    UpdatePolicy,
    get_recipe,
    init,
    unwrap_prior,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward

from .reproducibility import DEFAULT_SEED, construction_rng

NUM_BYTES = 256
BYTE_DIM = 8
DEFAULT_BIT_TOL = 0.25

_BIT_WEIGHTS = torch.tensor([1 << k for k in range(BYTE_DIM)], dtype=torch.long)

# Canonical 8-bit binary hypercube corners {-1, +1}^8:
# row b has bit k = +1.0 if (b >> k) & 1 else -1.0.
BYTE_CORNERS = torch.tensor(
    [[1.0 if ((b >> k) & 1) else -1.0 for k in range(BYTE_DIM)] for b in range(NUM_BYTES)],
    dtype=torch.float32,
)

# Bijective linear projection from {-1, +1}^8 onto the regular 16x16 nibble grid in [-1, 1]^2:
# u = high nibble weighted sum / 15, v = low nibble weighted sum / 15.
_NIBBLE_PROJ = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, 1.0 / 15.0, 2.0 / 15.0, 4.0 / 15.0, 8.0 / 15.0],
        [1.0 / 15.0, 2.0 / 15.0, 4.0 / 15.0, 8.0 / 15.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
_CANONICAL_NIBBLE_XY = BYTE_CORNERS @ _NIBBLE_PROJ.T

_RGB_MISSING = (0.12, 0.13, 0.16)
_RGB_CRISP = (0.18, 0.78, 0.34)
_RGB_COLLISION = (0.88, 0.22, 0.24)
_RGB_UNCRISP = (0.95, 0.78, 0.18)

PRIOR_EXCEPTION_REASONS: dict[str, str] = {
    "bytegan_hypercube8": (
        "Exact-capacity zero-sigma discrete ParticlePrior (256 particles for 256 uint8 bytes) "
        "required to test 1-to-1 discrete byte codebook bijection without MoG kernel smoothing."
    ),
    "bytegan_bigan_dictionary": (
        "Exact-capacity zero-sigma discrete ParticlePrior (256 particles for 256 uint8 bytes) "
        "required to test bidirectional 1-to-1 byte dictionary generation and encoder particle "
        "indexing without MoG kernel smoothing."
    ),
}

THRESHOLDS: dict[str, list[list]] = {
    "bytegan_hypercube8": [
        ["finite_fraction", "==", 1.0],
        ["valid_particle_fraction", "==", 1.0],
        ["valid_unique_bytes", "==", 256],
        ["rms_bit_error", "<=", 0.12],
    ],
    "bytegan_bigan_dictionary": [
        ["finite_fraction", "==", 1.0],
        ["valid_particle_fraction", "==", 1.0],
        ["valid_unique_bytes", "==", 256],
        ["rms_bit_error", "<=", 0.12],
        ["recon_valid_exact_bytes", "==", 256],
        ["recon_rms_bit_error", "<=", 0.12],
        ["encoder_particle_bijection_bytes", "==", 256],
    ],
}

HOST_DEFINITIONS: dict[str, dict] = {
    "bytegan_hypercube8": {
        "case_id": "api-bytegan-hypercube8",
        "kind": "bytegan_hypercube8",
        "target": "Uniform random uint8 bytes (0..255) encoded on the 8-bit binary hypercube {-1, +1}^8",
        "resources": {"num_particles": 256, "z_dim": 8, "batch_size": 128},
        "model": {"family": "simple_mlp", "hidden": 128, "layers": 2, "out_dim": 8, "fourier": 2},
        "bit_tol": 0.25,
        "steps": 4000,
    },
    "bytegan_bigan_dictionary": {
        "case_id": "api-bytegan-bigan-dictionary",
        "kind": "bytegan_bigan_dictionary",
        "target": "Uniform random uint8 bytes (0..255) on {-1, +1}^8 with joint BiGAN (G(z), E(x), D(x, z)) bidirectional 1-to-1 dictionary lookup",
        "resources": {"num_particles": 256, "z_dim": 8, "batch_size": 128},
        "model": {
            "family": "simple_mlp_bigan",
            "hidden": 128,
            "layers": 2,
            "x_dim": 8,
            "z_dim": 8,
            "joint_in_dim": 16,
            "fourier": 2,
        },
        "bit_tol": 0.25,
        "steps": 6000,
    },
}

_NO_EMA_FORMULATIONS = frozenset({
    "bispf",
    "empf",
    "rhbi",
    "t1_coupling",
    "t2_secant",
    "rgdt",
    "t4_nash",
    "t4_nash_v3",
    "unified",
    "unified_v4",
    "unified_v11",
    "r4_autograd_critic",
    "cond_fiber",
    "minimal",
})

_CASE_TO_KIND = {spec["case_id"]: kind for kind, spec in HOST_DEFINITIONS.items()}


def _check_thresholds(metrics: dict, kind: str) -> list[str]:
    failed = []
    for name, op, bound in THRESHOLDS[kind]:
        val = metrics.get(name)
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            failed.append(f"{name} {op} {bound}")
            continue
        ok = (
            val <= bound if op == "<=" else
            val >= bound if op == ">=" else
            val == bound if op == "==" else False
        )
        if not ok:
            failed.append(f"{name} {op} {bound}")
    return failed


def decode_bytes(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap continuous R^8 outputs to nearest {-1, +1}^8 corner and uint8 byte index 0..255."""
    snapped = torch.where(values >= 0.0, 1.0, -1.0)
    bits = (values >= 0.0).to(torch.long)
    byte_ids = (bits * _BIT_WEIGHTS.to(bits.device)).sum(dim=-1)
    return snapped, byte_ids


def sample_byte_target(
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Draw uniform i.i.d. uint8 bytes (with replacement) encoded on {-1, +1}^8."""
    dev = torch.device(device)
    idx = torch.randint(0, NUM_BYTES, (int(n),), generator=generator, device=generator.device, dtype=torch.long)
    return BYTE_CORNERS[idx.cpu()].to(dev)


def sample_target_for_kind(
    kind: str,
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if kind not in HOST_DEFINITIONS:
        raise ValueError(f"unsupported ByteGAN task kind: {kind!r}")
    return sample_byte_target(n, generator, device=device)


def build_models_for_spec(spec: dict, *, z_dim: int | None = None) -> tuple[nn.Module, ...]:
    model_cfg = spec["model"]
    family = model_cfg["family"]
    hidden = int(model_cfg["hidden"])
    layers = int(model_cfg["layers"])
    fourier = int(model_cfg["fourier"])
    z_eff = int(spec["resources"]["z_dim"] if z_dim is None else z_dim)
    if family == "simple_mlp":
        out_dim = int(model_cfg["out_dim"])
        return (
            SimpleMLPGenerator(z_dim=z_eff, hidden_dim=hidden, n_hidden=layers, out_dim=out_dim),
            SimpleMLPDiscriminator(in_dim=out_dim, hidden_dim=hidden, n_hidden=layers, fourier=fourier),
        )
    if family == "simple_mlp_bigan":
        x_dim = int(model_cfg["x_dim"])
        joint_in_dim = int(model_cfg["joint_in_dim"])
        return (
            SimpleMLPGenerator(z_dim=z_eff, hidden_dim=hidden, n_hidden=layers, out_dim=x_dim),
            SimpleMLPGenerator(z_dim=x_dim, hidden_dim=hidden, n_hidden=layers, out_dim=z_eff),
            SimpleMLPDiscriminator(in_dim=joint_in_dim, hidden_dim=hidden, n_hidden=layers, fourier=fourier),
        )
    raise ValueError(f"unsupported ByteGAN model family: {family!r}")


@torch.no_grad()
def evaluate_clean_codes(
    module: nn.Module,
    codes: torch.Tensor,
    *,
    prior: nn.Module | None = None,
) -> torch.Tensor:
    """Evaluate a network cleanly in eval mode without perturbing ambient RNG streams."""
    roots = [module] + ([prior] if prior is not None else [])
    modes = [(m, m.training) for root in roots for m in root.modules()]
    devices = (
        [codes.device.index]
        if codes.device.type == "cuda" and codes.device.index is not None
        else ([torch.cuda.current_device()] if codes.device.type == "cuda" else [])
    )
    try:
        for root in roots:
            root.eval()
        with torch.random.fork_rng(devices=devices):
            out = module(codes)
            if prior is not None and hasattr(prior, "postprocess_generated"):
                out = prior.postprocess_generated(out)
            return out
    finally:
        for m, flag in modes:
            m.training = flag


def _join_byte_latent(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, z), dim=-1)


def _uniform_rgb_grid(rgb: tuple[float, float, float]) -> torch.Tensor:
    grid = torch.empty((1, 3, 16, 16), dtype=torch.float32)
    grid[0, 0].fill_(rgb[0])
    grid[0, 1].fill_(rgb[1])
    grid[0, 2].fill_(rgb[2])
    return grid


def _byte_occupancy_heatmap(raw_counts: torch.Tensor, valid_counts: torch.Tensor) -> torch.Tensor:
    """Build (1, 3, 16, 16) RGB heatmap indexed by (high_nibble, low_nibble)."""
    img = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    raw = raw_counts.detach().cpu()
    valid = valid_counts.detach().cpu()
    for b in range(NUM_BYTES):
        hi = b >> 4
        lo = b & 0xF
        rc = int(raw[b].item())
        vc = int(valid[b].item())
        if rc == 0:
            color = _RGB_MISSING
        elif rc >= 2:
            color = _RGB_COLLISION
        elif vc == 1:
            color = _RGB_CRISP
        else:
            color = _RGB_UNCRISP
        img[0, 0, hi, lo] = color[0]
        img[0, 1, hi, lo] = color[1]
        img[0, 2, hi, lo] = color[2]
    return img


def _recon_exactness_heatmap(exact_mask: torch.Tensor, sign_match_mask: torch.Tensor) -> torch.Tensor:
    """Build (1, 3, 16, 16) RGB heatmap of canonical byte reconstruction exactness."""
    img = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    exact = exact_mask.detach().cpu()
    sign_ok = sign_match_mask.detach().cpu()
    for b in range(NUM_BYTES):
        hi = b >> 4
        lo = b & 0xF
        if bool(exact[b].item()):
            color = _RGB_CRISP
        elif bool(sign_ok[b].item()):
            color = _RGB_UNCRISP
        else:
            color = _RGB_COLLISION
        img[0, 0, hi, lo] = color[0]
        img[0, 1, hi, lo] = color[1]
        img[0, 2, hi, lo] = color[2]
    return img


def _latent_nn_hamming_mean(z_particles: torch.Tensor, bits: torch.Tensor) -> float:
    """Mean Hamming distance between each particle's decoded byte and its latent nearest neighbor."""
    flat = z_particles.detach().reshape(len(z_particles), -1).to(dtype=torch.float64, device="cpu")
    n = len(flat)
    if n < 2 or not bool(torch.isfinite(flat).all()):
        return 4.0
    dists = torch.cdist(flat, flat)
    dists = dists + 1e-12 * torch.arange(n, dtype=torch.float64).unsqueeze(0)
    dists.fill_diagonal_(float("inf"))
    nn_idx = dists.argmin(dim=1)
    hamming = (bits != bits[nn_idx]).sum(dim=-1).to(torch.float64)
    return float(hamming.mean().item())


def _project_nibble_xy(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float32) @ _NIBBLE_PROJ.T


def score_bytegan_hypercube8(
    z_particles: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    bit_tol: float = DEFAULT_BIT_TOL,
) -> dict:
    """Score 256 enumerated clean outputs on the 8-bit binary hypercube {-1, +1}^8."""
    kind = "bytegan_hypercube8"
    target_grid = _uniform_rgb_grid(_RGB_CRISP)

    valid_shape = (
        isinstance(endpoints, torch.Tensor)
        and isinstance(z_particles, torch.Tensor)
        and endpoints.ndim == 2
        and endpoints.shape == (NUM_BYTES, BYTE_DIM)
        and z_particles.ndim == 2
        and z_particles.shape[0] == NUM_BYTES
    )
    if valid_shape:
        end_row_finite = torch.isfinite(endpoints).all(dim=-1)
        z_row_finite = torch.isfinite(z_particles).all(dim=-1)
        finite_fraction = float((end_row_finite & z_row_finite).float().mean().item())
    else:
        finite_fraction = 0.0

    if not valid_shape or finite_fraction < 1.0:
        metrics = {
            "finite_fraction": finite_fraction,
            "valid_particle_fraction": 0.0,
            "valid_unique_bytes": 0,
            "rms_bit_error": 99.0,
            "unique_bytes": 0,
            "collision_count": NUM_BYTES,
            "valid_collision_count": 0,
            "orphan_count": NUM_BYTES,
            "mass_tv": 1.0,
            "max_bit_error": 99.0,
            "min_bit_margin": 0.0,
            "mean_abs_bit": 0.0,
            "latent_nn_hamming_mean": 4.0,
        }
        failed = _check_thresholds(metrics, kind)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {
                    "kind": "image",
                    "title": "16x16 byte occupancy (hi_nibble, lo_nibble)",
                    "target": target_grid,
                    "samples": _uniform_rgb_grid(_RGB_MISSING),
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "row_labels": ["Target 1:1", "G(z_i)"],
                    "caption": "Nonfinite or invalid-shape outputs fail closed.",
                },
                {
                    "kind": "scatter",
                    "title": "2D nibble projection of {-1,+1}^8 bytes",
                    "target": _CANONICAL_NIBBLE_XY.clone(),
                    "samples": torch.zeros((2, 2), dtype=torch.float32),
                    "xlim": [-1.3, 1.3],
                    "ylim": [-1.3, 1.3],
                    "aspect": "equal",
                    "xlabel": "high nibble",
                    "ylabel": "low nibble",
                    "caption": "Nonfinite outputs fail closed.",
                },
            ],
        }

    end_cpu = endpoints.detach().cpu().to(torch.float32)
    z_cpu = z_particles.detach().cpu().to(torch.float32)
    snapped, byte_ids = decode_bytes(end_cpu)
    bits = (end_cpu >= 0.0).to(torch.long)

    bit_err = (end_cpu - snapped).abs()
    max_err_per_particle = bit_err.max(dim=-1).values
    valid_mask = max_err_per_particle <= float(bit_tol)

    raw_counts = torch.bincount(byte_ids, minlength=NUM_BYTES)
    unique_bytes = int((raw_counts > 0).sum().item())
    collision_count = int(NUM_BYTES - unique_bytes)

    valid_ids = byte_ids[valid_mask]
    valid_counts = torch.bincount(valid_ids, minlength=NUM_BYTES)
    valid_unique_bytes = int((valid_counts > 0).sum().item())
    valid_collision_count = int((valid_counts - 1).clamp_min(0).sum().item())
    orphan_count = int(NUM_BYTES - valid_unique_bytes)

    occ = valid_counts.to(torch.float64) / float(NUM_BYTES)
    mass_tv = float(0.5 * (occ - (1.0 / float(NUM_BYTES))).abs().sum().item())

    metrics = {
        "finite_fraction": 1.0,
        "valid_particle_fraction": float(valid_mask.float().mean().item()),
        "valid_unique_bytes": valid_unique_bytes,
        "rms_bit_error": float(bit_err.square().mean().sqrt().item()),
        "unique_bytes": unique_bytes,
        "collision_count": collision_count,
        "valid_collision_count": valid_collision_count,
        "orphan_count": orphan_count,
        "mass_tv": mass_tv,
        "max_bit_error": float(bit_err.max().item()),
        "min_bit_margin": float(end_cpu.abs().min().item()),
        "mean_abs_bit": float(end_cpu.abs().mean().item()),
        "latent_nn_hamming_mean": _latent_nn_hamming_mean(z_cpu, bits),
    }
    failed = _check_thresholds(metrics, kind)
    occ_heatmap = _byte_occupancy_heatmap(raw_counts, valid_counts)
    proj_xy = _project_nibble_xy(end_cpu).clamp(-1.28, 1.28)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                "kind": "image",
                "title": "16x16 byte occupancy (hi_nibble, lo_nibble)",
                "target": target_grid,
                "samples": occ_heatmap,
                "vmin": 0.0,
                "vmax": 1.0,
                "row_labels": ["Target 1:1", "G(z_i)"],
                "caption": "Green=crisp 1-to-1, red=collision (>=2), yellow=uncrisp, dark=missing byte.",
            },
            {
                "kind": "scatter",
                "title": "2D nibble projection of {-1,+1}^8 bytes",
                "target": _CANONICAL_NIBBLE_XY.clone(),
                "samples": proj_xy,
                "xlim": [-1.3, 1.3],
                "ylim": [-1.3, 1.3],
                "aspect": "equal",
                "xlabel": "high nibble",
                "ylabel": "low nibble",
                "caption": "Exact corners form a 16x16 grid on [-1,1]^2; barycentric ghosts collapse to (0,0).",
            },
        ],
    }


def score_bytegan_bigan_dictionary(
    z_particles: torch.Tensor,
    endpoints: torch.Tensor,
    encoded_canonical: torch.Tensor,
    recon_canonical: torch.Tensor,
    *,
    bit_tol: float = DEFAULT_BIT_TOL,
) -> dict:
    """Score bidirectional 256-byte generation, canonical reconstruction, and particle indexing."""
    kind = "bytegan_bigan_dictionary"
    target_grid = _uniform_rgb_grid(_RGB_CRISP)

    valid_shape = (
        isinstance(z_particles, torch.Tensor)
        and isinstance(endpoints, torch.Tensor)
        and isinstance(encoded_canonical, torch.Tensor)
        and isinstance(recon_canonical, torch.Tensor)
        and endpoints.ndim == 2
        and endpoints.shape == (NUM_BYTES, BYTE_DIM)
        and recon_canonical.ndim == 2
        and recon_canonical.shape == (NUM_BYTES, BYTE_DIM)
        and z_particles.ndim == 2
        and encoded_canonical.ndim == 2
        and z_particles.shape[0] == NUM_BYTES
        and encoded_canonical.shape == z_particles.shape
    )
    if valid_shape:
        row_finite = (
            torch.isfinite(z_particles).all(dim=-1)
            & torch.isfinite(endpoints).all(dim=-1)
            & torch.isfinite(encoded_canonical).all(dim=-1)
            & torch.isfinite(recon_canonical).all(dim=-1)
        )
        finite_fraction = float(row_finite.float().mean().item())
    else:
        finite_fraction = 0.0

    if not valid_shape or finite_fraction < 1.0:
        metrics = {
            "finite_fraction": finite_fraction,
            "valid_particle_fraction": 0.0,
            "valid_unique_bytes": 0,
            "rms_bit_error": 99.0,
            "unique_bytes": 0,
            "collision_count": NUM_BYTES,
            "valid_collision_count": 0,
            "orphan_count": NUM_BYTES,
            "mass_tv": 1.0,
            "max_bit_error": 99.0,
            "min_bit_margin": 0.0,
            "mean_abs_bit": 0.0,
            "latent_nn_hamming_mean": 4.0,
            "recon_valid_exact_bytes": 0,
            "recon_rms_bit_error": 99.0,
            "recon_max_bit_error": 99.0,
            "recon_sign_exact_bytes": 0,
            "encoder_particle_bijection_bytes": 0,
            "encoder_cycle_match_bytes": 0,
            "encoder_nearest_dist_mean": 99.0,
        }
        failed = _check_thresholds(metrics, kind)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {
                    "kind": "image",
                    "title": "Forward 16x16 particle byte occupancy",
                    "target": target_grid,
                    "samples": _uniform_rgb_grid(_RGB_MISSING),
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "row_labels": ["Target 1:1", "G(z_i)"],
                    "caption": "Nonfinite or invalid-shape outputs fail closed.",
                },
                {
                    "kind": "image",
                    "title": "Inverse 16x16 reconstruction exactness G(E(x))",
                    "target": target_grid,
                    "samples": _uniform_rgb_grid(_RGB_COLLISION),
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "row_labels": ["Target x", "G(E(x))"],
                    "caption": "Nonfinite or invalid-shape outputs fail closed.",
                },
            ],
        }

    fwd = score_bytegan_hypercube8(z_particles, endpoints, bit_tol=bit_tol)
    z_cpu = z_particles.detach().cpu().to(torch.float32)
    end_cpu = endpoints.detach().cpu().to(torch.float32)
    enc_cpu = encoded_canonical.detach().cpu().to(torch.float32)
    recon_cpu = recon_canonical.detach().cpu().to(torch.float32)

    recon_err = (recon_cpu - BYTE_CORNERS).abs()
    recon_max_per_byte = recon_err.max(dim=-1).values
    recon_exact_mask = recon_max_per_byte <= float(bit_tol)
    _, recon_byte_ids = decode_bytes(recon_cpu)
    canonical_ids = torch.arange(NUM_BYTES, dtype=torch.long)
    recon_sign_mask = recon_byte_ids == canonical_ids

    recon_valid_exact_bytes = int(recon_exact_mask.sum().item())
    recon_rms_bit_error = float(recon_err.square().mean().sqrt().item())
    recon_max_bit_error = float(recon_err.max().item())
    recon_sign_exact_bytes = int(recon_sign_mask.sum().item())

    dists = torch.cdist(enc_cpu.to(torch.float64), z_cpu.to(torch.float64))
    min_dists, nearest_p = dists.min(dim=1)
    encoder_particle_bijection_bytes = int(torch.unique(nearest_p).numel())
    _, end_byte_ids = decode_bytes(end_cpu)
    encoder_cycle_match_bytes = int((end_byte_ids[nearest_p] == canonical_ids).sum().item())
    encoder_nearest_dist_mean = float(min_dists.mean().item())

    metrics = {
        **fwd["metrics"],
        "recon_valid_exact_bytes": recon_valid_exact_bytes,
        "recon_rms_bit_error": recon_rms_bit_error,
        "recon_max_bit_error": recon_max_bit_error,
        "recon_sign_exact_bytes": recon_sign_exact_bytes,
        "encoder_particle_bijection_bytes": encoder_particle_bijection_bytes,
        "encoder_cycle_match_bytes": encoder_cycle_match_bytes,
        "encoder_nearest_dist_mean": encoder_nearest_dist_mean,
    }
    failed = _check_thresholds(metrics, kind)
    recon_heatmap = _recon_exactness_heatmap(recon_exact_mask, recon_sign_mask)
    recon_xy = _project_nibble_xy(recon_cpu).clamp(-1.28, 1.28)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                **fwd["views"][0],
                "title": "Forward 16x16 particle byte occupancy",
            },
            {
                "kind": "image",
                "title": "Inverse 16x16 reconstruction exactness G(E(x))",
                "target": target_grid,
                "samples": recon_heatmap,
                "vmin": 0.0,
                "vmax": 1.0,
                "row_labels": ["Target x", "G(E(x))"],
                "caption": "Green=crisp exact byte x, yellow=right sign but uncrisp, red=wrong byte.",
            },
            {
                "kind": "scatter",
                "title": "2D nibble projection of reconstructions G(E(x))",
                "target": _CANONICAL_NIBBLE_XY.clone(),
                "samples": recon_xy,
                "xlim": [-1.3, 1.3],
                "ylim": [-1.3, 1.3],
                "aspect": "equal",
                "xlabel": "high nibble",
                "ylabel": "low nibble",
                "caption": "Exact reconstructions align with the canonical 16x16 grid on [-1,1]^2.",
            },
        ],
    }


def bytegan_oracle_controls() -> dict[str, dict]:
    """Analytic positive and destructive negative controls proving metric selectivity."""
    controls: dict[str, dict] = {}
    gen = torch.Generator(device="cpu").manual_seed(92001)
    jitter = (2.0 * torch.rand((NUM_BYTES, BYTE_DIM), generator=gen) - 1.0) * 0.08
    z_clean = BYTE_CORNERS.clone() + 0.02 * (2.0 * torch.rand((NUM_BYTES, BYTE_DIM), generator=gen) - 1.0)

    # --- Hypercube8 controls ---
    y_pos = BYTE_CORNERS + jitter
    res = score_bytegan_hypercube8(z_clean, y_pos)
    controls["hypercube8_exact_bijection_oracle"] = {
        "expected_pass": True,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    y_col = y_pos.clone()
    y_col[255] = y_pos[0]
    res = score_bytegan_hypercube8(z_clean, y_col)
    controls["hypercube8_single_collision_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    y_ghost = 0.05 * BYTE_CORNERS
    res = score_bytegan_hypercube8(z_clean, y_ghost)
    controls["hypercube8_faint_barycentric_ghosts_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    y_exploded = 5.0 * BYTE_CORNERS
    res = score_bytegan_hypercube8(z_clean, y_exploded)
    controls["hypercube8_exploded_amplitude_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    y_half = BYTE_CORNERS[torch.arange(NUM_BYTES) % 128]
    res = score_bytegan_hypercube8(z_clean, y_half)
    controls["hypercube8_half_cube_collapse_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    y_nan = y_pos.clone()
    y_nan[0, 0] = float("nan")
    y_nan[1, 1] = float("inf")
    res = score_bytegan_hypercube8(z_clean, y_nan)
    controls["hypercube8_nonfinite_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    # --- BiGAN dictionary controls ---
    enc_pos = z_clean + 0.01 * (2.0 * torch.rand((NUM_BYTES, BYTE_DIM), generator=gen) - 1.0)
    recon_pos = BYTE_CORNERS + 0.06 * (2.0 * torch.rand((NUM_BYTES, BYTE_DIM), generator=gen) - 1.0)
    res = score_bytegan_bigan_dictionary(z_clean, y_pos, enc_pos, recon_pos)
    controls["bigan_exact_dictionary_oracle"] = {
        "expected_pass": True,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    recon_swapped = recon_pos.clone()
    recon_swapped[0], recon_swapped[1] = recon_pos[1].clone(), recon_pos[0].clone()
    res = score_bytegan_bigan_dictionary(z_clean, y_pos, enc_pos, recon_swapped)
    controls["bigan_swapped_reconstruction_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    # Encoder collapses all 256 bytes tightly near particle 0 while G(E(x)) == x still holds!
    enc_bypass = z_clean[0:1].expand(NUM_BYTES, BYTE_DIM) + 1e-4 * BYTE_CORNERS
    res = score_bytegan_bigan_dictionary(z_clean, y_pos, enc_bypass, recon_pos)
    controls["bigan_encoder_bypass_cloud_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    res = score_bytegan_bigan_dictionary(z_clean, y_col, enc_pos, recon_pos)
    controls["bigan_forward_collision_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    recon_blurry = 0.5 * BYTE_CORNERS
    res = score_bytegan_bigan_dictionary(z_clean, y_pos, enc_pos, recon_blurry)
    controls["bigan_blurry_reconstruction_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    recon_nan = recon_pos.clone()
    recon_nan[7, 3] = float("nan")
    recon_nan[8, 4] = float("inf")
    res = score_bytegan_bigan_dictionary(z_clean, y_pos, enc_pos, recon_nan)
    controls["bigan_nonfinite_negative"] = {
        "expected_pass": False,
        "passed": res["passed"],
        "failed_bounds": res["failed_bounds"],
    }

    return controls


def list_cases() -> list[dict]:
    return [
        {
            "id": "api-bytegan-hypercube8",
            "title": "ByteGAN 256-byte 8-bit binary hypercube 1-to-1 bijection (zero-sigma ParticlePrior)",
            "goal": (
                "Learn all 256 uniform uint8 bytes on {-1, +1}^8 with an exact-capacity zero-sigma "
                "ParticlePrior (256 particles, z_dim=8) in a crisp 1-to-1 bijection without collisions "
                "or barycentric ghosts."
            ),
            "kind": "vector",
            "scope": "Public API stretch variant: exact-capacity zero-sigma ParticlePrior 256-byte hypercube bijection",
            "legacy_ids": ["stretch-bytegan-hypercube8"],
            "default_steps": 4000,
            "batch_size": 128,
            "eval_samples": 256,
            "evaluation_observations": 24,
            "terminal_observations": 5,
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS["bytegan_hypercube8"]},
            "sampling": "Enumerated clean zero-sigma ParticlePrior outputs across all 256 particles",
        },
        {
            "id": "api-bytegan-bigan-dictionary",
            "title": "ByteGAN joint BiGAN bidirectional 256-byte dictionary lookup (zero-sigma ParticlePrior)",
            "goal": (
                "Learn a joint BiGAN (G(z), E(x), D(x, z)) over 256 zero-sigma ParticlePrior codes on "
                "{-1, +1}^8 with exact 256-byte generation, exact canonical reconstruction G(E(x))==x, "
                "and 1-to-1 encoder particle indexing."
            ),
            "kind": "vector",
            "scope": "Public API stretch variant: joint BiGAN bidirectional 256-byte dictionary over zero-sigma ParticlePrior",
            "legacy_ids": ["stretch-bytegan-bigan-dictionary"],
            "default_steps": 6000,
            "batch_size": 128,
            "eval_samples": 256,
            "evaluation_observations": 24,
            "terminal_observations": 5,
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS["bytegan_bigan_dictionary"]},
            "sampling": "Enumerated clean zero-sigma ParticlePrior generation plus paired canonical E(x) and G(E(x)) lookup",
        },
    ]


def _seed_int(tag: str, seed: int) -> int:
    digest = hashlib.sha256(f"api_bytegan:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _bind_formulation_record(
    recipe: Recipe,
    opt_d: Any,
    loss_fn: Any,
    prior_reg_fn: Any = None,
    generator: nn.Module | None = None,
) -> None:
    """Bind instance-local critic records to losses and prior regularizers matching GANTrainer."""
    record = getattr(opt_d, "record", None)
    if record is None:
        return
    eff = recipe.effective_critic_formulation
    if eff == "bispf":
        from particlegan.bispf import bind_thermal_source
        bind_thermal_source(record, loss_fn, prior_reg_fn)
    elif eff == "empf":
        from particlegan.empf import bind_empf_record
        bind_empf_record(record, loss_fn, prior_reg_fn)
    elif eff == "t2_secant":
        from particlegan.t2_critic import bind_t2_record
        bind_t2_record(record, loss_fn, prior_reg_fn)
    elif eff == "unified":
        from particlegan.unified import bind_unified_record
        bind_unified_record(record, loss_fn, prior_reg_fn)
    elif eff in ("unified_v4", "r4_autograd_critic"):
        from particlegan.unified_v4 import bind_unified_v4_record
        bind_unified_v4_record(record, loss_fn, prior_reg_fn)
        if prior_reg_fn is not None and getattr(prior_reg_fn, "_bound_device", None) is None and generator is not None:
            g_dev = next((p.device for p in generator.parameters()), None)
            if g_dev is not None:
                prior_reg_fn._bound_device = g_dev
    elif eff == "unified_v11":
        from particlegan.unified_v11 import bind_unified_v11_record
        bind_unified_v11_record(record, loss_fn, prior_reg_fn)
        if prior_reg_fn is not None and getattr(prior_reg_fn, "_bound_device", None) is None and generator is not None:
            g_dev = next((p.device for p in generator.parameters()), None)
            if g_dev is not None:
                prior_reg_fn._bound_device = g_dev
    elif eff == "cond_fiber":
        from particlegan.cond_fiber import bind_cond_fiber_record
        bind_cond_fiber_record(record, loss_fn, prior_reg_fn)
    elif eff == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0:
        from particlegan.rhbi import bind_rhbi_record
        bind_rhbi_record(record, loss_fn, prior_reg_fn)
    elif eff == "t1_coupling" or recipe.t1c_weight > 0:
        from particlegan.t1_coupling import bind_t1_coupling_record
        bind_t1_coupling_record(record, loss_fn, prior_reg_fn)
    elif eff == "rgdt":
        from particlegan.t3_dynamics import bind_rgdt_record
        bind_rgdt_record(record, loss_fn, prior_reg_fn)
    elif eff == "t4_nash":
        from particlegan.t4_nash import bind_t4_nash_record
        bind_t4_nash_record(record, loss_fn, prior_reg_fn)
    elif eff == "t4_nash_v3":
        from particlegan.nash_v3 import bind_nash_v3_record
        bind_nash_v3_record(record, loss_fn, prior_reg_fn)
    elif eff == "minimal":
        from particlegan.minimal import bind_minimal_record
        bind_minimal_record(record, loss_fn, prior_reg_fn)
        if prior_reg_fn is not None and getattr(prior_reg_fn, "_bound_device", None) is None and generator is not None:
            g_dev = next((p.device for p in generator.parameters()), None)
            if g_dev is not None:
                prior_reg_fn._bound_device = g_dev


class ByteGANToyFixture:
    """Public-API GANTrainer fixture for bytegan_hypercube8."""

    api_components = ("GANTrainer", "ParticlePrior", "Recipe", "init.deterministic_orthogonal_")

    def __init__(
        self,
        case: dict | None = None,
        *,
        recipe_name: str = "unified_v12",
        recipe: Recipe | None = None,
        recipe_overrides: dict | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
    ) -> None:
        if case is None:
            case = next(c for c in list_cases() if c["id"] == "api-bytegan-hypercube8")
        self.case = deepcopy(case)
        self.case_id = case["id"]
        self.kind = _CASE_TO_KIND[self.case_id]
        if self.kind != "bytegan_hypercube8":
            raise ValueError(f"ByteGANToyFixture only supports bytegan_hypercube8, got {self.kind!r}")
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
            "prior_kind": "particles",
            "sigma_rel": 0.0,
            "standardize": False,
            **(recipe_overrides or {}),
        }
        if base_recipe.total_steps is not None:
            updates["total_steps"] = self.limit
        self.recipe = base_recipe.replace(**updates)
        if self.recipe.prior_code_law != "table":
            raise ValueError("bytegan_hypercube8 requires Recipe.prior_code_law == 'table'")

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
        prior = ParticlePrior(
            num_particles=self.recipe.num_particles,
            z_dim=self.recipe.z_dim,
            init_std=1.0,
            device=self.device,
            dtype=torch.float32,
            learnable=True,
            generator=prior_rng,
        )
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
        real_cpu = sample_byte_target(self.recipe.batch_size, self.data_rng, device="cpu")
        self._batch_hasher.update(real_cpu.numpy().tobytes())
        out = self.trainer.step(real_cpu.to(self.device))
        return {k: float(v.item()) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    @torch.no_grad()
    def observe(self, n: int = NUM_BYTES, seed: int = 10000) -> dict:
        del n
        eval_stream = torch.Generator(device=self.device).manual_seed(int(seed))
        endpoints = self.trainer.sample(
            self.recipe.num_particles,
            fixed_first_n=True,
            output_noise=False,
            generator=eval_stream,
        )
        z_table = unwrap_prior(self.trainer.prior).z.detach()
        return score_bytegan_hypercube8(z_table, endpoints, bit_tol=float(self.spec["bit_tol"]))

    def state_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "data_rng": self.data_rng.get_state().clone(),
            "trainer": self.trainer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("case_id") != self.case_id:
            raise ValueError("checkpoint case_id mismatch")
        self.data_rng.set_state(state["data_rng"].cpu())
        self.trainer.load_state_dict(state["trainer"])


class ByteGANBiGANFixture:
    """Public-components joint BiGAN (G(z), E(x), D(x, z)) fixture for bytegan_bigan_dictionary."""

    api_components = (
        "get_recipe",
        "Recipe.make_prior",
        "init.deterministic_orthogonal_",
        "Recipe.make_generator_optimizer",
        "Recipe.make_critic_optimizer",
        "Recipe.make_loss",
        "Recipe.make_prior_regularizer",
        "Recipe.make_critic_penalty",
        "UpdatePolicy",
        "InputNoise",
        "constraint_geometry_backward",
    )

    def __init__(
        self,
        case: dict | None = None,
        *,
        recipe_name: str | None = "unified_v12",
        recipe: Recipe | None = None,
        recipe_overrides: dict | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
        components: Any = None,
    ) -> None:
        if case is None:
            case = next(c for c in list_cases() if c["id"] == "api-bytegan-bigan-dictionary")
        self.case = deepcopy(case)
        self.case_id = case["id"]
        self.kind = _CASE_TO_KIND[self.case_id]
        if self.kind != "bytegan_bigan_dictionary":
            raise ValueError(f"ByteGANBiGANFixture only supports bytegan_bigan_dictionary, got {self.kind!r}")
        self.spec = deepcopy(HOST_DEFINITIONS[self.kind])
        self.device = torch.device(device)
        self.seed = int(seed)
        self.max_steps = int(self.spec["steps"] if max_steps is None else max_steps)
        if type(self.max_steps) is not int or self.max_steps < 1 or self.seed < 0:
            raise ValueError("bytegan_bigan max_steps/seed must be positive/nonnegative integers")

        res = self.spec["resources"]
        if components is not None:
            self.recipe = components.recipe
        else:
            base_recipe = get_recipe(recipe_name or "unified_v12") if recipe is None else recipe
            updates = {
                "num_particles": int(res["num_particles"]),
                "z_dim": int(res["z_dim"]),
                "batch_size": int(res["batch_size"]),
                "prior_kind": "particles",
                "sigma_rel": 0.0,
                "standardize": False,
                **(recipe_overrides or {}),
            }
            if base_recipe.total_steps is not None:
                updates["total_steps"] = self.max_steps
            self.recipe = base_recipe.replace(**updates)

        if (
            self.recipe.model != "gan"
            or self.recipe.conditioning != "scalar"
            or self.recipe.encoder_mode != "none"
            or self.recipe.row_policy != "independent"
        ):
            raise ValueError("bytegan_bigan_dictionary requires scalar independent GAN recipe without internal encoder")
        if self.recipe.prior_code_law != "table":
            raise ValueError("bytegan_bigan_dictionary requires Recipe.prior_code_law == 'table'")
        if (self.recipe.num_particles, self.recipe.z_dim, self.recipe.batch_size) != (
            int(res["num_particles"]),
            int(res["z_dim"]),
            int(res["batch_size"]),
        ):
            raise ValueError("bytegan_bigan_dictionary resources contradict the frozen host specification")

        self.transport = None
        if self.recipe.kinetic_transport_weight or self.recipe.kinetic_transport_local_weight:
            if components is not None and components.component_transport != "output_marginal_v1":
                raise ValueError("bytegan_bigan transport requires an explicit output_marginal_v1 consumer")
            from particlegan.conditional_transport import OutputMarginalTransport
            self.transport = OutputMarginalTransport(self.recipe)

        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            if components is None:
                with construction_rng(self.seed, self.device):
                    G, E, D = build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)
                self.G, self.E, self.D = G.to(self.device), E.to(self.device), D.to(self.device)
                for comp, mod in (("generator", self.G), ("discriminator", self.D), ("encoder", self.E)):
                    seeds = {
                        name: _seed_int(f"bigan:init:{comp}:{name}", self.seed)
                        for name, p in mod.named_parameters()
                        if p.requires_grad and p.numel()
                    }
                    init.deterministic_orthogonal_(mod, parameter_seeds=seeds)
                prior_rng = torch.Generator(device=self.device).manual_seed(
                    _seed_int("bigan:init:prior:locations", self.seed)
                )
                self.prior = self.recipe.replace(prior_update="learned").make_prior(
                    device=self.device,
                    dtype=torch.float32,
                    generator=prior_rng,
                    init_std=1.0,
                )
                init.deterministic_orthogonal_(
                    unwrap_prior(self.prior),
                    parameter_seeds={"z": _seed_int("bigan:init:prior:z", self.seed)},
                )
                self.recipe.apply_prior_policy(self.prior)
            else:
                self.G = components.construct(
                    lambda: build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)[0],
                    component="generator",
                ).to(self.device)
                self.E = components.construct(
                    lambda: build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)[1],
                    component="encoder",
                ).to(self.device)
                self.D = components.construct(
                    lambda: build_models_for_spec(self.spec, z_dim=self.recipe.z_dim)[2],
                    component="discriminator",
                ).to(self.device)
                for mod, role in ((self.G, "generator"), (self.D, "discriminator"), (self.E, "encoder")):
                    components.initialize(mod, component=role)
                self.prior = components.build_prior()

            if type(unwrap_prior(self.prior)) is not ParticlePrior:
                raise ValueError("bytegan_bigan_dictionary requires a zero-sigma ParticlePrior")

            ema_critic = (
                None
                if self.recipe.effective_critic_formulation in _NO_EMA_FORMULATIONS
                else deepcopy(self.D)
            )
            self.opt_g, self.opt_d = self.recipe.make_optimizers(
                self.G,
                self.D,
                self.prior,
                encoder=self.E,
                ema_critic=ema_critic,
                separate_encoder_groups=True,
            )
            self.loss = self.recipe.make_loss()
            self.spread = self.recipe.make_prior_regularizer(prior=self.prior, reference=self.prior)
            self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=components is not None)
            _bind_formulation_record(self.recipe, self.opt_d, self.loss, self.spread, self.G)

            policy_streams = (
                {
                    "latent_generator": torch.Generator(device=self.device).manual_seed(
                        _seed_int("bigan:prior:latent", self.seed)
                    ),
                    "penalty_generator": torch.Generator(device=self.device).manual_seed(
                        _seed_int("bigan:noise:penalty", self.seed)
                    ),
                    "eval_generator": torch.Generator(device=self.device).manual_seed(
                        _seed_int("bigan:eval:sampler", self.seed)
                    ),
                    "noise_generator": torch.Generator(device=self.device).manual_seed(
                        _seed_int("bigan:noise:output", self.seed)
                    ),
                }
                if components is None
                else {
                    "latent_generator": components.streams.generator("prior", component="latent", purpose="indices"),
                    "penalty_generator": components.streams.generator("noise", component="penalty", purpose="training"),
                    "eval_generator": components.streams.generator("eval", component="sampler", purpose="samples"),
                    "noise_generator": components.streams.generator("noise", component="generator", purpose="output"),
                }
            )
            self.policy = UpdatePolicy(
                self.recipe,
                self.G,
                self.D,
                prior=self.prior,
                encoder=self.E,
                generator_optimizer=self.opt_g,
                critic_optimizer=self.opt_d,
                row_semantics="conditional",
                seed=self.seed,
                penalty=self.penalty,
                streams=policy_streams,
            )
            self.input_noise_generator = (
                torch.Generator(device=self.device).manual_seed(_seed_int("bigan:noise:input", self.seed))
                if components is None
                else components.streams.generator("noise", component="critic", purpose="input")
            )
            self.noisy_critic = InputNoise(self.D, generator=self.input_noise_generator)

        self.canonical_bytes = BYTE_CORNERS.to(self.device)
        self.data_rng = (
            torch.Generator(device="cpu").manual_seed(_seed_int("bigan:data:target:training", self.seed))
            if components is None
            else components.streams.generator("data", component="target", purpose="training", device="cpu")
        )
        self._batch_hasher = hashlib.sha256()
        self.metadata = {
            "checkpoint_capabilities": {
                "round_trip": True,
                "preserves_rng": True,
                "owner": "UpdatePolicy.state_dict + prior_regularizer + transport + caller RNGs",
            }
        }

    @property
    def completed_steps(self) -> int:
        return self.policy.completed_steps

    @property
    def batch_sequence_sha256(self) -> str:
        return self._batch_hasher.hexdigest()

    def step(self) -> dict:
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("bytegan_bigan execution budget exhausted")
        real_cpu = sample_byte_target(self.recipe.batch_size, self.data_rng, device="cpu")
        self._batch_hasher.update(real_cpu.numpy().tobytes())
        real_bytes = real_cpu.to(self.device)

        flags = [p.requires_grad for p in self.D.parameters()]
        modes = [(m, m.training) for root in (self.G, self.E, self.D) for m in root.modules()]
        base_prior = unwrap_prior(self.prior)
        postproc = bool(getattr(self.prior, "postprocesses_generated", False))
        try:
            with torch.autograd.set_multithreading_enabled(False):
                self.D.train()
                self.G.eval()
                self.E.eval()
                with torch.no_grad():
                    encoded_d = self.E(real_bytes)
                joint_real = _join_byte_latent(real_bytes, encoded_d)
                noise = self.policy.begin_step(joint_real, execution_limit=self.max_steps)
                self.noisy_critic.std = noise.input_sigma
                with torch.no_grad():
                    latent_d, rows_d = self.prior.sample(len(real_bytes), generator=self.policy.latent_generator)
                    fake_bytes_d = self.policy.generate(latent_d, sigma=noise.output_sigma, rows=rows_d)
                    if postproc:
                        fake_bytes_d = self.prior.postprocess_generated(fake_bytes_d)
                joint_fake = _join_byte_latent(fake_bytes_d, latent_d)
                self.policy.observe_critic_pair(joint_real, joint_fake)
                adversarial_d = self.loss.d_loss(self.noisy_critic(joint_real), self.noisy_critic(joint_fake))
                penalty = self.penalty(self.noisy_critic, joint_real, joint_fake)
                loss_d = adversarial_d + penalty
                self.opt_d.zero_grad(set_to_none=True)
                self.policy.before_critic_backward()
                loss_d.backward()
                self.opt_d.step()
                self.policy.after_critic_step()

                self.D.eval().requires_grad_(False)
                self.G.train()
                self.E.train()
                encoded = self.E(real_bytes)
                latent, rows = self.prior.sample(len(real_bytes), generator=self.policy.latent_generator)
                fake_bytes = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows)
                if postproc:
                    fake_bytes = self.prior.postprocess_generated(fake_bytes)
                if hasattr(self.opt_g, "bind_origin_sample"):
                    self.opt_g.bind_origin_sample(fake_bytes)
                loss_gan = self.loss.joint_g_loss(
                    self.noisy_critic(_join_byte_latent(fake_bytes, latent)),
                    self.noisy_critic(_join_byte_latent(real_bytes, encoded)),
                )
                prior_reg = self.recipe.prior_regularization(base_prior.z, regularizer=self.spread)
                loss_g = loss_gan + prior_reg
                reconstruction = None
                if self.recipe.joint_reconstruction_weight:
                    reconstruction = self.recipe.joint_reconstruction_loss(
                        self.G(encoded), real_bytes, categorical_dim=None
                    )
                    loss_g = loss_g + reconstruction
                if self.transport is not None:
                    loss_g = self.transport.add(loss_g, fake_bytes, real_bytes)

                self.opt_g.zero_grad(set_to_none=True)
                self.policy.before_generator_backward()
                protected_evaluator = None
                if self.recipe.constraint_geometry_mode == "strict_progress":
                    width_noise = (latent - base_prior.z[rows]).detach()

                    def protected_evaluator():
                        current_latent = base_prior.z[rows] + width_noise
                        gen_cur = self.G(current_latent)
                        if postproc:
                            gen_cur = self.prior.postprocess_generated(gen_cur)
                        return (
                            self.loss.joint_g_loss(
                                self.noisy_critic(_join_byte_latent(gen_cur, current_latent)),
                                self.noisy_critic(_join_byte_latent(real_bytes, self.E(real_bytes))),
                            ),
                        )

                constraint_geometry_backward(
                    loss_g,
                    self.opt_g,
                    (loss_gan,),
                    protected_evaluator=protected_evaluator,
                )
                self.policy.after_generator_backward(
                    loss_gan=loss_gan.detach(),
                    loss_critic=adversarial_d.detach(),
                )
                self.opt_g.step()
                self.policy.after_generator_step()
                self.policy.finish_step()
                result = {
                    "step": self.completed_steps,
                    "loss_d": loss_d.detach(),
                    "loss_g": loss_g.detach(),
                    "penalty": penalty.detach(),
                }
                if reconstruction is not None:
                    result["joint_reconstruction"] = reconstruction.detach()
                return result
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            for p, flag in zip(self.D.parameters(), flags):
                p.requires_grad_(flag)
            for m, flag in modes:
                m.training = flag

    @torch.no_grad()
    def observe(self, n: int = NUM_BYTES, seed: int = 10000) -> dict:
        del n, seed
        z_table = unwrap_prior(self.prior).z.detach()
        endpoints = evaluate_clean_codes(self.G, z_table, prior=self.prior)
        encoded_canonical = evaluate_clean_codes(self.E, self.canonical_bytes)
        recon_canonical = evaluate_clean_codes(self.G, encoded_canonical, prior=self.prior)
        return score_bytegan_bigan_dictionary(
            z_table,
            endpoints,
            encoded_canonical,
            recon_canonical,
            bit_tol=float(self.spec["bit_tol"]),
        )

    def state_dict(self) -> dict:
        state = {
            "case_id": self.case_id,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "recipe": self.recipe.to_dict(),
            "data_rng": self.data_rng.get_state().clone(),
            "input_noise_rng": self.input_noise_generator.get_state().clone(),
            "api_components": self.api_components,
            "api_state": self.policy.state_dict(),
            **self.recipe.prior_regularizer_checkpoint(self.spread),
        }
        if self.transport is not None:
            state["component_transport"] = self.transport.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("case_id") != self.case_id:
            raise ValueError("checkpoint case_id mismatch")
        if state.get("recipe") != self.recipe.to_dict():
            raise ValueError("checkpoint recipe mismatch")
        self.recipe.restore_prior_regularizer(self.spread, state.get("prior_regularizer"), validate_only=True)
        if self.transport is None:
            if state.get("component_transport") is not None:
                raise ValueError("inactive bytegan_bigan fixture cannot restore active transport")
        else:
            if state.get("component_transport") is None:
                raise ValueError("active bytegan_bigan fixture is missing transport checkpoint")
            self.transport.load_state_dict(state["component_transport"])
        self.policy.load_state_dict(state["api_state"])
        self.recipe.restore_prior_regularizer(self.spread, state.get("prior_regularizer"))
        self.data_rng.set_state(state["data_rng"].cpu())
        in_rng = state["input_noise_rng"]
        self.input_noise_generator.set_state(
            in_rng.cpu() if self.input_noise_generator.device.type == "cpu" else in_rng.to(self.input_noise_generator.device)
        )


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
):
    del kwargs
    if case_id not in _CASE_TO_KIND:
        raise ValueError(f"unknown ByteGAN toy case: {case_id!r}")
    case = next(c for c in list_cases() if c["id"] == case_id)
    if case_id == "api-bytegan-hypercube8":
        return ByteGANToyFixture(
            case,
            recipe_name=recipe_name or "unified_v12",
            recipe=recipe,
            recipe_overrides=recipe_overrides,
            device=device,
            seed=seed,
            max_steps=max_steps,
        )
    return ByteGANBiGANFixture(
        case,
        recipe_name=recipe_name or "unified_v12",
        recipe=recipe,
        recipe_overrides=recipe_overrides,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
