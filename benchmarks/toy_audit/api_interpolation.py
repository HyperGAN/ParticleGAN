"""Public-API toy experiments for unseen latent interpolation between discrete particles.

Scientific question: when a generator and zero-sigma ``ParticlePrior`` learn a
small finite set of latent particles ``z_1, ..., z_M`` without MoG kernel
smoothing, does unseen latent interior space (1D k-NN chords and 2D k-NN
simplex barycenters) map to valid, smoothly progressing manifold outputs rather
than chord-cutting dead space, double-exposed pixel ghosts, or Voronoi lookup
snapping?
"""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy

import torch
from torch import nn

from benchmarks.transfer_suite.image_tasks import Discriminator as ImageDiscriminator
from benchmarks.transfer_suite.image_tasks import Generator as ImageGenerator
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import (
    GANTrainer,
    ParticlePrior,
    Recipe,
    get_recipe,
    init,
    unwrap_prior,
)

from .reproducibility import DEFAULT_SEED, construction_rng

ARC_RADIUS = 2.0
ARC_SIGMA = 0.03
ARC_THETA_MIN = -0.25 * math.pi
ARC_THETA_MAX = 1.25 * math.pi
ARC_BINS = 32
CHORD_TS = (0.25, 0.50, 0.75)

PATCH_MIN = 1.5
PATCH_MAX = 6.5
PATCH_NOISE_STD = 0.01
PATCH_GRID_STEPS = 51

SADDLE_SIGMA = 0.03

PRIOR_EXCEPTION_REASON = (
    "Zero-sigma discrete ParticlePrior required to test whether unseen latent space "
    "between finite learned particles forms a valid, non-snapping manifold without MoG kernel smoothing."
)

THRESHOLDS: dict[str, list[list]] = {
    "interp_sparse_arc": [
        ["finite_fraction", "==", 1.0],
        ["endpoint_hq", ">=", 0.85],
        ["interp_hq", ">=", 0.80],
        ["interp_arc_bins", ">=", 22],
        ["midpoint_novelty", ">=", 0.20],
    ],
    "interp_img_patch_shift": [
        ["finite_fraction", "==", 1.0],
        ["endpoint_rmse", "<=", 0.09],
        ["interp_rmse", "<=", 0.10],
        ["interp_peak", ">=", 0.78],
        ["quadrant_coverage", "==", 4],
        ["midpoint_novelty", ">=", 0.18],
    ],
    "interp_saddle_barycentric": [
        ["finite_fraction", "==", 1.0],
        ["endpoint_surface_mae", "<=", 0.08],
        ["barycenter_surface_mae", "<=", 0.09],
        ["barycenter_hq", ">=", 0.78],
        ["xy_bins_covered", ">=", 12],
        ["barycenter_novelty", ">=", 0.18],
    ],
}

HOST_DEFINITIONS: dict[str, dict] = {
    "interp_sparse_arc": {
        "case_id": "api-interp-sparse-arc",
        "kind": "interp_sparse_arc",
        "target": "270-degree 2D circular arc (R=2.0, theta in [-pi/4, 5pi/4], sigma=0.03)",
        "resources": {"num_particles": 12, "z_dim": 2, "batch_size": 128},
        "model": {
            "family": "simple_mlp",
            "hidden": 64,
            "layers": 2,
            "out_dim": 2,
            "fourier": 2,
        },
        "interpolation": {"graph": "latent_knn_chords", "k_neighbors": 3, "t_grid": [0.25, 0.5, 0.75]},
        "steps": 600,
    },
    "interp_img_patch_shift": {
        "case_id": "api-interp-img-patch-shift",
        "kind": "interp_img_patch_shift",
        "target": "Continuous 1x8x8 single 2x2 unit-peak patch at (u, v) in [1.5, 6.5]^2 with clipped sigma=0.01 noise",
        "resources": {"num_particles": 16, "z_dim": 4, "batch_size": 64},
        "model": {
            "family": "image_conv_residual_upsample",
            "architecture": "residual_upsample",
            "width": 16,
        },
        "interpolation": {"graph": "latent_knn_chords", "k_neighbors": 2, "t_grid": [0.25, 0.5, 0.75]},
        "steps": 800,
    },
    "interp_saddle_barycentric": {
        "case_id": "api-interp-saddle-barycentric",
        "kind": "interp_saddle_barycentric",
        "target": "Non-harmonic 3D saddle sheet w = 0.8*(x^2 - y^2) + 0.9*cos(0.75*pi*x)*cos(0.75*pi*y) on [-1, 1]^2 (sigma=0.03)",
        "resources": {"num_particles": 24, "z_dim": 2, "batch_size": 128},
        "model": {
            "family": "simple_mlp",
            "hidden": 64,
            "layers": 2,
            "out_dim": 3,
            "fourier": 2,
        },
        "interpolation": {
            "graph": "latent_knn_barycenters",
            "k_neighbors": 4,
            "min_interior_ratio": 0.15,
        },
        "steps": 800,
    },
}

_PATCH_MANIFOLD_CACHE: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


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


def knn_edges(points: torch.Tensor, k: int) -> list[tuple[int, int]]:
    """Deterministic undirected k-NN edge list with row-index tie-breaking."""
    flat = points.detach().reshape(len(points), -1).to(dtype=torch.float64, device="cpu")
    n = len(flat)
    if n < 2:
        return []
    k_eff = min(max(1, int(k)), n - 1)
    dists = torch.cdist(flat, flat)
    dists = dists + 1e-12 * torch.arange(n, dtype=torch.float64).unsqueeze(0)
    dists.fill_diagonal_(float("inf"))
    nbrs = dists.topk(k_eff, largest=False).indices
    edges = {(min(int(i), int(j)), max(int(i), int(j))) for i in range(n) for j in nbrs[i].tolist() if int(i) != int(j)}
    return sorted(edges)


def knn_triangles(points: torch.Tensor, k: int, *, min_interior_ratio: float = 0.15) -> list[tuple[int, int, int]]:
    """Deterministic non-collinear latent k-NN triangles (falls back to all k-NN triples if degenerate)."""
    flat = points.detach().reshape(len(points), -1).to(dtype=torch.float64, device="cpu")
    n = len(flat)
    if n < 3:
        return []
    k_eff = min(max(2, int(k)), n - 1)
    dists = torch.cdist(flat, flat)
    dists = dists + 1e-12 * torch.arange(n, dtype=torch.float64).unsqueeze(0)
    dists.fill_diagonal_(float("inf"))
    nbrs = dists.topk(k_eff, largest=False).indices
    candidates: set[tuple[int, int, int]] = set()
    for i in range(n):
        row = [int(v) for v in nbrs[i].tolist()]
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                tri = tuple(sorted((i, row[a], row[b])))
                if len(set(tri)) == 3:
                    candidates.add(tri)  # type: ignore[arg-type]
    ordered = sorted(candidates)
    interior = []
    for i, j, m in ordered:
        verts = torch.stack((flat[i], flat[j], flat[m]), dim=0)
        bary = verts.mean(dim=0)
        diam = float(torch.pdist(verts).max().item())
        min_vertex_dist = float((verts - bary).norm(dim=-1).min().item())
        if min_vertex_dist / (diam + 1e-6) >= min_interior_ratio:
            interior.append((i, j, m))
    return interior if len(interior) >= 8 else ordered


def sample_arc_target(n: int, generator: torch.Generator, *, device: str | torch.device = "cpu") -> torch.Tensor:
    dev = torch.device(device)
    u = torch.rand((n,), generator=generator, device=generator.device, dtype=torch.float32).to(dev)
    eps = torch.randn((n,), generator=generator, device=generator.device, dtype=torch.float32).to(dev)
    theta = ARC_THETA_MIN + (ARC_THETA_MAX - ARC_THETA_MIN) * u
    r = ARC_RADIUS + ARC_SIGMA * eps
    return torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)


def render_patch_images(uv: torch.Tensor) -> torch.Tensor:
    """Exact 2x2 unit-peak window integral at continuous centers (u, v) in [1.5, 6.5]^2."""
    r = torch.arange(8, device=uv.device, dtype=uv.dtype).view(1, 8, 1)
    c = torch.arange(8, device=uv.device, dtype=uv.dtype).view(1, 1, 8)
    u = uv[:, 0].view(-1, 1, 1)
    v = uv[:, 1].view(-1, 1, 1)
    wr = (torch.minimum(r + 1.0, u + 1.0) - torch.maximum(r, u - 1.0)).clamp(0.0, 1.0)
    wc = (torch.minimum(c + 1.0, v + 1.0) - torch.maximum(c, v - 1.0)).clamp(0.0, 1.0)
    return (wr * wc).unsqueeze(1)


def patch_manifold_grid(device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    key = (str(dev), dtype)
    cached = _PATCH_MANIFOLD_CACHE.get(key)
    if cached is not None:
        return cached
    axis = torch.linspace(PATCH_MIN, PATCH_MAX, PATCH_GRID_STEPS, device=dev, dtype=dtype)
    ug, vg = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((ug.reshape(-1), vg.reshape(-1)), dim=-1)
    templates = render_patch_images(coords)
    _PATCH_MANIFOLD_CACHE[key] = (coords, templates)
    return coords, templates


def sample_patch_target(n: int, generator: torch.Generator, *, device: str | torch.device = "cpu") -> torch.Tensor:
    dev = torch.device(device)
    u = torch.rand((n, 2), generator=generator, device=generator.device, dtype=torch.float32).to(dev)
    noise = torch.randn((n, 1, 8, 8), generator=generator, device=generator.device, dtype=torch.float32).to(dev)
    centers = PATCH_MIN + (PATCH_MAX - PATCH_MIN) * u
    clean = render_patch_images(centers)
    return (clean + PATCH_NOISE_STD * noise).clamp(0.0, 1.0)


def saddle_height(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Non-harmonic curved saddle sheet so 3D Euclidean chord-cutting fails."""
    return 0.8 * (x.square() - y.square()) + 0.9 * torch.cos(0.75 * math.pi * x) * torch.cos(0.75 * math.pi * y)


def sample_saddle_target(n: int, generator: torch.Generator, *, device: str | torch.device = "cpu") -> torch.Tensor:
    dev = torch.device(device)
    xy = (2.0 * torch.rand((n, 2), generator=generator, device=generator.device, dtype=torch.float32) - 1.0).to(dev)
    eps = torch.randn((n,), generator=generator, device=generator.device, dtype=torch.float32).to(dev)
    w = saddle_height(xy[:, 0], xy[:, 1]) + SADDLE_SIGMA * eps
    return torch.stack((xy[:, 0], xy[:, 1], w), dim=-1)


def sample_target_for_kind(kind: str, n: int, generator: torch.Generator, *, device: str | torch.device = "cpu") -> torch.Tensor:
    if kind == "interp_sparse_arc":
        return sample_arc_target(n, generator, device=device)
    if kind == "interp_img_patch_shift":
        return sample_patch_target(n, generator, device=device)
    if kind == "interp_saddle_barycentric":
        return sample_saddle_target(n, generator, device=device)
    raise ValueError(f"unsupported interpolation task kind: {kind!r}")


def build_models_for_spec(spec: dict, *, z_dim: int) -> tuple[nn.Module, nn.Module]:
    model_cfg = spec["model"]
    if model_cfg["family"] == "simple_mlp":
        out_dim = int(model_cfg["out_dim"])
        hidden = int(model_cfg["hidden"])
        layers = int(model_cfg["layers"])
        fourier = int(model_cfg["fourier"])
        return (
            SimpleMLPGenerator(z_dim=z_dim, hidden_dim=hidden, n_hidden=layers, out_dim=out_dim),
            SimpleMLPDiscriminator(in_dim=out_dim, hidden_dim=hidden, n_hidden=layers, fourier=fourier),
        )
    if model_cfg["family"] == "image_conv_residual_upsample":
        img_spec = {"architecture": model_cfg["architecture"], "width": int(model_cfg["width"]), "z_dim": z_dim}
        return ImageGenerator(img_spec), ImageDiscriminator(img_spec)
    raise ValueError(f"unsupported interpolation model family: {model_cfg['family']!r}")


@torch.no_grad()
def evaluate_clean_codes(generator_module: nn.Module, codes: torch.Tensor, *, prior: nn.Module | None = None) -> torch.Tensor:
    """Evaluate generator cleanly on arbitrary latent query points without touching RNG streams."""
    roots = [generator_module] + ([prior] if prior is not None else [])
    modes = [(m, m.training) for root in roots for m in root.modules()]
    devices = [codes.device.index] if codes.device.type == "cuda" and codes.device.index is not None else (
        [torch.cuda.current_device()] if codes.device.type == "cuda" else []
    )
    try:
        for root in roots:
            root.eval()
        with torch.random.fork_rng(devices=devices):
            out = generator_module(codes)
            if prior is not None and hasattr(prior, "postprocess_generated"):
                out = prior.postprocess_generated(out)
            return out
    finally:
        for m, flag in modes:
            m.training = flag


def _arc_point_errors(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw_angle = torch.atan2(points[:, 1], points[:, 0])
    # Place branch cut at -pi/2, diametrically opposite the arc center at +pi/2.
    wrapped = torch.remainder(raw_angle + 0.5 * math.pi, 2.0 * math.pi) - 0.5 * math.pi
    clamped = wrapped.clamp(ARC_THETA_MIN, ARC_THETA_MAX)
    closest = torch.stack((ARC_RADIUS * torch.cos(clamped), ARC_RADIUS * torch.sin(clamped)), dim=-1)
    dist = (points - closest).norm(dim=-1)
    return dist, clamped


def _particle_set_novelty(endpoints: torch.Tensor, interior: torch.Tensor) -> float:
    """Distance of interior outputs from ALL discrete particle endpoints, normalized by local particle NN spacing."""
    if len(endpoints) < 2 or len(interior) == 0:
        return 0.0
    flat_end = endpoints.reshape(len(endpoints), -1).to(torch.float64)
    flat_int = interior.reshape(len(interior), -1).to(torch.float64)
    end_dists = torch.cdist(flat_end, flat_end)
    end_dists.fill_diagonal_(float("inf"))
    nn_spacing = end_dists.min(dim=1).values
    int_to_end = torch.cdist(flat_int, flat_end)
    min_dist, nearest_idx = int_to_end.min(dim=1)
    local_span = nn_spacing[nearest_idx]
    valid = local_span > 1e-4
    if not bool(valid.any()):
        return 0.0
    ratios = torch.where(valid, min_dist / (local_span + 1e-6), torch.zeros_like(min_dist))
    return float(ratios.mean().item())


def score_sparse_arc(
    z_particles: torch.Tensor,
    endpoints: torch.Tensor,
    eval_fn,
    *,
    reference: torch.Tensor | None = None,
    k_neighbors: int = 3,
) -> dict:
    kind = "interp_sparse_arc"
    z_cpu = z_particles.detach()
    edges = knn_edges(z_cpu, k_neighbors)
    chord_codes = torch.stack(
        [(1.0 - t) * z_particles[i] + t * z_particles[j] for i, j in edges for t in CHORD_TS],
        dim=0,
    )
    chords = eval_fn(chord_codes)

    ref = reference
    if ref is None:
        gen = torch.Generator(device="cpu").manual_seed(91001)
        ref = sample_arc_target(512, gen, device="cpu")
    ref_cpu = ref.detach().cpu().to(torch.float32)

    combined = torch.cat((endpoints.reshape(len(endpoints), -1), chords.reshape(len(chords), -1)), dim=0)
    finite_fraction = float(torch.isfinite(combined).all(dim=-1).float().mean().item())
    if finite_fraction < 1.0:
        metrics = {
            "finite_fraction": finite_fraction,
            "endpoint_hq": 0.0,
            "interp_hq": 0.0,
            "interp_arc_bins": 0,
            "midpoint_novelty": 0.0,
            "endpoint_arc_bins": 0,
            "interp_mean_dist": 99.0,
            "output_knn_interp_hq": 0.0,
            "all_pairs_interp_hq": 0.0,
            "edge_count": len(edges),
        }
        failed = _check_thresholds(metrics, kind)
        zeros = torch.zeros((2, 2), dtype=torch.float32)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {"kind": "scatter", "title": "12 discrete particle endpoints vs arc", "target": ref_cpu, "samples": zeros},
                {"kind": "scatter", "title": "Unseen latent k-NN chord points", "target": ref_cpu, "samples": zeros},
            ],
        }

    end_pts = endpoints.detach().cpu().to(torch.float32)
    chord_pts = chords.detach().cpu().to(torch.float32)

    tol = 3.0 * ARC_SIGMA
    end_dist, end_theta = _arc_point_errors(end_pts)
    chord_dist, chord_theta = _arc_point_errors(chord_pts)
    end_hq_mask = end_dist <= tol
    chord_hq_mask = chord_dist <= tol

    def _bins(thetas: torch.Tensor, mask: torch.Tensor) -> int:
        if not bool(mask.any()):
            return 0
        normed = (thetas[mask] - ARC_THETA_MIN) / (ARC_THETA_MAX - ARC_THETA_MIN)
        idx = (normed * ARC_BINS).long().clamp(0, ARC_BINS - 1)
        return int(torch.unique(idx).numel())

    all_thetas = torch.cat((end_theta, chord_theta), dim=0)
    all_hq = torch.cat((end_hq_mask, chord_hq_mask), dim=0)

    # Diagnostic output-space k-NN and all-pairs midpoint quality (non-gating).
    out_edges = knn_edges(end_pts, k_neighbors)
    out_codes = torch.stack([0.5 * z_particles[i] + 0.5 * z_particles[j] for i, j in out_edges], dim=0)
    out_mids = eval_fn(out_codes).detach().cpu().to(torch.float32)
    out_hq = float((_arc_point_errors(out_mids)[0] <= tol).float().mean().item()) if torch.isfinite(out_mids).all() else 0.0

    all_edges = [(i, j) for i in range(len(z_particles)) for j in range(i + 1, len(z_particles))]
    all_codes = torch.stack([0.5 * z_particles[i] + 0.5 * z_particles[j] for i, j in all_edges], dim=0)
    all_mids = eval_fn(all_codes).detach().cpu().to(torch.float32)
    all_pairs_hq = float((_arc_point_errors(all_mids)[0] <= tol).float().mean().item()) if torch.isfinite(all_mids).all() else 0.0

    metrics = {
        "finite_fraction": 1.0,
        "endpoint_hq": float(end_hq_mask.float().mean().item()),
        "interp_hq": float(chord_hq_mask.float().mean().item()),
        "interp_arc_bins": _bins(all_thetas, all_hq),
        "midpoint_novelty": _particle_set_novelty(end_pts, chord_pts),
        "endpoint_arc_bins": _bins(end_theta, end_hq_mask),
        "interp_mean_dist": float(chord_dist.mean().item()),
        "output_knn_interp_hq": out_hq,
        "all_pairs_interp_hq": all_pairs_hq,
        "edge_count": len(edges),
    }
    failed = _check_thresholds(metrics, kind)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                "kind": "scatter",
                "title": "12 discrete particle endpoints vs 270° arc",
                "target": ref_cpu,
                "samples": end_pts,
                "xlim": [-2.5, 2.5],
                "ylim": [-2.0, 2.5],
                "xlabel": "x",
                "ylabel": "y",
                "caption": "Zero-sigma ParticlePrior (M=12): discrete endpoints alone can cover at most 12/32 arc bins.",
            },
            {
                "kind": "scatter",
                "title": "Unseen latent k-NN chords (t in 0.25, 0.5, 0.75)",
                "target": ref_cpu,
                "samples": chord_pts,
                "xlim": [-2.5, 2.5],
                "ylim": [-2.0, 2.5],
                "xlabel": "x",
                "ylabel": "y",
                "caption": "Unseen interior latent points between neighbor particles; chord-cutting through (0,0) fails interp_hq.",
            },
        ],
    }


def _nearest_patch_projection(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords, manifold = patch_manifold_grid(device=images.device, dtype=images.dtype)
    rmses = []
    assigns = []
    for start in range(0, len(images), 16):
        chunk = images[start:start + 16]
        diff = (chunk[:, None] - manifold[None]).square().mean(dim=(2, 3, 4)).sqrt()
        best_rmse, best_idx = diff.min(dim=1)
        rmses.append(best_rmse)
        assigns.append(best_idx)
    nearest_rmse = torch.cat(rmses, dim=0)
    idx = torch.cat(assigns, dim=0)
    return nearest_rmse, coords[idx], manifold[idx]


def score_img_patch_shift(
    z_particles: torch.Tensor,
    endpoints: torch.Tensor,
    eval_fn,
    *,
    reference: torch.Tensor | None = None,
    k_neighbors: int = 2,
) -> dict:
    del reference
    kind = "interp_img_patch_shift"
    edges = knn_edges(z_particles.detach(), k_neighbors)
    chord_codes = torch.stack(
        [(1.0 - t) * z_particles[i] + t * z_particles[j] for i, j in edges for t in CHORD_TS],
        dim=0,
    )
    chords = eval_fn(chord_codes)
    midpoints = chords[1::len(CHORD_TS)]

    combined = torch.cat((endpoints.reshape(len(endpoints), -1), chords.reshape(len(chords), -1)), dim=0)
    finite_fraction = float(torch.isfinite(combined).all(dim=-1).float().mean().item())
    if finite_fraction < 1.0 or endpoints.ndim != 4 or endpoints.shape[1:] != (1, 8, 8):
        metrics = {
            "finite_fraction": finite_fraction if endpoints.ndim == 4 else 0.0,
            "endpoint_rmse": 1.0,
            "interp_rmse": 1.0,
            "interp_peak": 0.0,
            "quadrant_coverage": 0,
            "midpoint_novelty": 0.0,
            "endpoint_hq": 0.0,
            "interp_hq": 0.0,
            "output_knn_interp_rmse": 1.0,
            "output_knn_interp_peak": 0.0,
            "all_pairs_interp_rmse": 1.0,
            "edge_count": len(edges),
        }
        failed = _check_thresholds(metrics, kind)
        blank = torch.zeros((8, 1, 8, 8), dtype=torch.float32)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {"kind": "image", "title": "16 discrete particle endpoints (8x8)", "target": blank, "samples": blank},
                {"kind": "image", "title": "Unseen latent chord midpoints vs manifold", "target": blank, "samples": blank},
            ],
        }

    end_img = endpoints.detach().cpu().to(torch.float32)
    chord_img = chords.detach().cpu().to(torch.float32)
    mid_img = midpoints.detach().cpu().to(torch.float32)

    end_rmse, end_uv, end_nearest = _nearest_patch_projection(end_img)
    chord_rmse, chord_uv, _ = _nearest_patch_projection(chord_img)
    _, _, mid_nearest = _nearest_patch_projection(mid_img)

    hq_tol = 0.11
    all_rmse = torch.cat((end_rmse, chord_rmse), dim=0)
    all_uv = torch.cat((end_uv, chord_uv), dim=0)
    hq_uv = all_uv[all_rmse <= hq_tol]
    quads = 0
    if len(hq_uv) > 0:
        q_idx = 2 * (hq_uv[:, 0] >= 4.0).long() + (hq_uv[:, 1] >= 4.0).long()
        q_counts = torch.bincount(q_idx, minlength=4)
        quads = int((q_counts >= 2).sum().item())

    out_edges = knn_edges(end_img.reshape(len(end_img), -1), k_neighbors)
    out_codes = torch.stack([0.5 * z_particles[i] + 0.5 * z_particles[j] for i, j in out_edges], dim=0)
    out_mids = eval_fn(out_codes).detach().cpu().to(torch.float32)
    if torch.isfinite(out_mids).all():
        out_rmse_v, _, _ = _nearest_patch_projection(out_mids)
        out_rmse = float(out_rmse_v.mean().item())
        out_peak = float(out_mids.amax(dim=(1, 2, 3)).mean().item())
    else:
        out_rmse, out_peak = 1.0, 0.0

    all_edges = [(i, j) for i in range(len(z_particles)) for j in range(i + 1, len(z_particles))]
    all_codes = torch.stack([0.5 * z_particles[i] + 0.5 * z_particles[j] for i, j in all_edges], dim=0)
    all_mids = eval_fn(all_codes).detach().cpu().to(torch.float32)
    if torch.isfinite(all_mids).all():
        all_rmse_v, _, _ = _nearest_patch_projection(all_mids)
        all_pairs_rmse = float(all_rmse_v.mean().item())
    else:
        all_pairs_rmse = 1.0

    metrics = {
        "finite_fraction": 1.0,
        "endpoint_rmse": float(end_rmse.mean().item()),
        "interp_rmse": float(chord_rmse.mean().item()),
        "interp_peak": float(chord_img.amax(dim=(1, 2, 3)).mean().item()),
        "quadrant_coverage": quads,
        "midpoint_novelty": _particle_set_novelty(end_img, chord_img),
        "endpoint_hq": float((end_rmse <= hq_tol).float().mean().item()),
        "interp_hq": float((chord_rmse <= hq_tol).float().mean().item()),
        "output_knn_interp_rmse": out_rmse,
        "output_knn_interp_peak": out_peak,
        "all_pairs_interp_rmse": all_pairs_rmse,
        "edge_count": len(edges),
    }
    failed = _check_thresholds(metrics, kind)
    show_mids = min(12, len(mid_img))
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                "kind": "image",
                "title": "16 discrete particle endpoints vs nearest patch",
                "target": end_nearest[:12],
                "samples": end_img[:12],
                "vmin": 0.0,
                "vmax": 1.0,
                "row_labels": ["Nearest patch", "G(z_i)"],
                "caption": "Zero-sigma ParticlePrior (M=16) endpoints against nearest continuous 2x2 patch template.",
            },
            {
                "kind": "image",
                "title": "Unseen latent chord midpoints G(0.5 z_i + 0.5 z_j)",
                "target": mid_nearest[:show_mids],
                "samples": mid_img[:show_mids],
                "vmin": 0.0,
                "vmax": 1.0,
                "row_labels": ["Nearest patch", "Midpoint"],
                "caption": "Valid interpolation shifts a single sharp peak-1.0 patch; pixel blending creates dim 0.5 ghosts.",
            },
        ],
    }


def _saddle_surface_errors(points: torch.Tensor) -> torch.Tensor:
    cx = points[:, 0].clamp(-1.0, 1.0)
    cy = points[:, 1].clamp(-1.0, 1.0)
    box_pen = ((points[:, 0] - cx).square() + (points[:, 1] - cy).square()).sqrt()
    return (points[:, 2] - saddle_height(cx, cy)).abs() + box_pen


def score_saddle_barycentric(
    z_particles: torch.Tensor,
    endpoints: torch.Tensor,
    eval_fn,
    *,
    reference: torch.Tensor | None = None,
    k_neighbors: int = 4,
    min_interior_ratio: float = 0.15,
) -> dict:
    kind = "interp_saddle_barycentric"
    tris = knn_triangles(z_particles.detach(), k_neighbors, min_interior_ratio=min_interior_ratio)
    bary_codes = torch.stack(
        [(z_particles[i] + z_particles[j] + z_particles[m]) / 3.0 for i, j, m in tris],
        dim=0,
    )
    barys = eval_fn(bary_codes)

    edges = knn_edges(z_particles.detach(), 3)
    edge_codes = torch.stack([0.5 * z_particles[i] + 0.5 * z_particles[j] for i, j in edges], dim=0)
    edge_mids = eval_fn(edge_codes)

    ref = reference
    if ref is None:
        gen = torch.Generator(device="cpu").manual_seed(91003)
        ref = sample_saddle_target(512, gen, device="cpu")
    ref_cpu = ref.detach().cpu().to(torch.float32)
    ref_xy = ref_cpu[:, :2]
    ref_diag = torch.stack((saddle_height(ref_cpu[:, 0], ref_cpu[:, 1]), ref_cpu[:, 2]), dim=-1)

    combined = torch.cat(
        (endpoints.reshape(len(endpoints), -1), barys.reshape(len(barys), -1), edge_mids.reshape(len(edge_mids), -1)),
        dim=0,
    )
    finite_fraction = float(torch.isfinite(combined).all(dim=-1).float().mean().item())
    if finite_fraction < 1.0 or endpoints.shape[-1] != 3:
        metrics = {
            "finite_fraction": finite_fraction if endpoints.shape[-1] == 3 else 0.0,
            "endpoint_surface_mae": 9.9,
            "barycenter_surface_mae": 9.9,
            "barycenter_hq": 0.0,
            "xy_bins_covered": 0,
            "barycenter_novelty": 0.0,
            "endpoint_hq": 0.0,
            "edge_midpoint_surface_mae": 9.9,
            "output_knn_barycenter_mae": 9.9,
            "triangle_count": len(tris),
        }
        failed = _check_thresholds(metrics, kind)
        zeros = torch.zeros((2, 2), dtype=torch.float32)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {"kind": "scatter", "title": "Simplex barycenters (x, y) coverage", "target": ref_xy, "samples": zeros},
                {"kind": "scatter", "title": "Surface parity (w_true(x,y) vs w_gen)", "target": ref_diag, "samples": zeros},
            ],
        }

    end_pts = endpoints.detach().cpu().to(torch.float32)
    bary_pts = barys.detach().cpu().to(torch.float32)
    edge_pts = edge_mids.detach().cpu().to(torch.float32)

    end_err = _saddle_surface_errors(end_pts)
    bary_err = _saddle_surface_errors(bary_pts)
    edge_err = _saddle_surface_errors(edge_pts)

    hq_tol = 0.10
    all_pts = torch.cat((end_pts, bary_pts), dim=0)
    all_err = torch.cat((end_err, bary_err), dim=0)
    in_box = (all_pts[:, 0].abs() <= 1.02) & (all_pts[:, 1].abs() <= 1.02) & (all_err <= hq_tol)
    if bool(in_box.any()):
        hq_xy = all_pts[in_box, :2].clamp(-0.9999, 0.9999)
        bx = ((hq_xy[:, 0] + 1.0) * 2.0).long().clamp(0, 3)
        by = ((hq_xy[:, 1] + 1.0) * 2.0).long().clamp(0, 3)
        xy_bins = int(torch.unique(4 * bx + by).numel())
    else:
        xy_bins = 0

    bary_novelty = _particle_set_novelty(end_pts, bary_pts)

    out_tris = knn_triangles(end_pts[:, :2], k_neighbors, min_interior_ratio=0.10)
    out_codes = torch.stack([(z_particles[i] + z_particles[j] + z_particles[m]) / 3.0 for i, j, m in out_tris], dim=0)
    out_barys = eval_fn(out_codes).detach().cpu().to(torch.float32)
    out_mae = float(_saddle_surface_errors(out_barys).mean().item()) if torch.isfinite(out_barys).all() else 9.9

    metrics = {
        "finite_fraction": 1.0,
        "endpoint_surface_mae": float(end_err.mean().item()),
        "barycenter_surface_mae": float(bary_err.mean().item()),
        "barycenter_hq": float((bary_err <= hq_tol).float().mean().item()),
        "xy_bins_covered": xy_bins,
        "barycenter_novelty": bary_novelty,
        "endpoint_hq": float((end_err <= hq_tol).float().mean().item()),
        "edge_midpoint_surface_mae": float(edge_err.mean().item()),
        "output_knn_barycenter_mae": out_mae,
        "triangle_count": len(tris),
    }
    failed = _check_thresholds(metrics, kind)
    bary_true_w = saddle_height(bary_pts[:, 0].clamp(-1.0, 1.0), bary_pts[:, 1].clamp(-1.0, 1.0))
    bary_diag = torch.stack((bary_true_w, bary_pts[:, 2]), dim=-1)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                "kind": "scatter",
                "title": "2D simplex barycenters (x, y) over [-1, 1]^2",
                "target": ref_xy,
                "samples": bary_pts[:, :2],
                "xlim": [-1.2, 1.2],
                "ylim": [-1.2, 1.2],
                "xlabel": "x",
                "ylabel": "y",
                "caption": "Unseen latent triangle centroids (z_i + z_j + z_k)/3 from M=24 discrete particles.",
            },
            {
                "kind": "scatter",
                "title": "Curved saddle parity: w_target(x,y) vs w_generated",
                "target": ref_diag,
                "samples": bary_diag,
                "xlim": [-1.6, 1.9],
                "ylim": [-1.6, 1.9],
                "aspect": "equal",
                "xlabel": "w_target(x, y)",
                "ylabel": "w_generated",
                "caption": "Points on the curved 3D sheet lie on the diagonal; 3D Euclidean simplex chords scatter off-diagonal.",
            },
        ],
    }


def score_interpolation_kind(
    kind: str,
    z_particles: torch.Tensor,
    endpoints: torch.Tensor,
    eval_fn,
    *,
    reference: torch.Tensor | None = None,
) -> dict:
    spec = HOST_DEFINITIONS[kind]
    interp_cfg = spec["interpolation"]
    if kind == "interp_sparse_arc":
        return score_sparse_arc(
            z_particles, endpoints, eval_fn, reference=reference, k_neighbors=int(interp_cfg["k_neighbors"])
        )
    if kind == "interp_img_patch_shift":
        return score_img_patch_shift(
            z_particles, endpoints, eval_fn, reference=reference, k_neighbors=int(interp_cfg["k_neighbors"])
        )
    if kind == "interp_saddle_barycentric":
        return score_saddle_barycentric(
            z_particles,
            endpoints,
            eval_fn,
            reference=reference,
            k_neighbors=int(interp_cfg["k_neighbors"]),
            min_interior_ratio=float(interp_cfg["min_interior_ratio"]),
        )
    raise ValueError(f"unsupported interpolation kind: {kind!r}")


def interpolation_oracle_controls() -> dict[str, dict]:
    """Analytic positive and destructive negative controls proving metric selectivity."""
    controls: dict[str, dict] = {}

    # --- 1. Sparse Arc Controls ---
    z_arc = torch.stack(
        (torch.linspace(-1.4, 1.4, 12), 0.15 * torch.sin(torch.linspace(-2.0, 2.0, 12))),
        dim=-1,
    )

    def _arc_theta_from_z(z: torch.Tensor) -> torch.Tensor:
        frac = ((z[:, 0] + 1.4) / 2.8).clamp(0.0, 1.0)
        return ARC_THETA_MIN + (ARC_THETA_MAX - ARC_THETA_MIN) * frac

    def _arc_smooth(z: torch.Tensor) -> torch.Tensor:
        th = _arc_theta_from_z(z)
        return torch.stack((ARC_RADIUS * torch.cos(th), ARC_RADIUS * torch.sin(th)), dim=-1)

    arc_end = _arc_smooth(z_arc)
    pos_arc = score_sparse_arc(z_arc, arc_end, _arc_smooth)
    controls["arc_smooth_oracle"] = {
        "expected_pass": True,
        "passed": pos_arc["passed"],
        "failed_bounds": pos_arc["failed_bounds"],
    }

    # Linear chord cutter: exact endpoints on arc, straight Euclidean segments between latent vertices.
    def _arc_chord_cutter(z_query: torch.Tensor) -> torch.Tensor:
        s = z_query[:, 0].contiguous()
        s_grid = z_arc[:, 0].contiguous()
        # Make 2nd-neighbor chords cut straight across by snapping to coarse 6-vertex polyline.
        coarse_idx = torch.linspace(0, 11, 6).round().long()
        coarse_s = s_grid[coarse_idx].contiguous()
        coarse_y = arc_end[coarse_idx]
        chi = torch.searchsorted(coarse_s, s).clamp(1, len(coarse_s) - 1)
        clo = chi - 1
        alpha = ((s - coarse_s[clo]) / (coarse_s[chi] - coarse_s[clo] + 1e-8)).clamp(0.0, 1.0).unsqueeze(-1)
        return (1.0 - alpha) * coarse_y[clo] + alpha * coarse_y[chi]

    neg_arc_chord = score_sparse_arc(z_arc, arc_end, _arc_chord_cutter)
    controls["arc_chord_cutter_negative"] = {
        "expected_pass": False,
        "passed": neg_arc_chord["passed"],
        "failed_bounds": neg_arc_chord["failed_bounds"],
    }

    def _voronoi_lookup(z_ref: torch.Tensor, y_ref: torch.Tensor):
        def _fn(z_query: torch.Tensor) -> torch.Tensor:
            nearest = torch.cdist(z_query.to(torch.float64), z_ref.to(torch.float64)).argmin(dim=1)
            return y_ref[nearest]
        return _fn

    neg_arc_snap = score_sparse_arc(z_arc, arc_end, _voronoi_lookup(z_arc, arc_end))
    controls["arc_voronoi_snap_negative"] = {
        "expected_pass": False,
        "passed": neg_arc_snap["passed"],
        "failed_bounds": neg_arc_snap["failed_bounds"],
    }

    # --- 2. Image Patch Shift Controls ---
    g4 = torch.linspace(-1.0, 1.0, 4)
    gu, gv = torch.meshgrid(g4, g4, indexing="ij")
    z_patch = torch.stack((gu.reshape(-1), gv.reshape(-1), 0.2 * gu.reshape(-1), -0.2 * gv.reshape(-1)), dim=-1)

    def _patch_smooth(z: torch.Tensor) -> torch.Tensor:
        uv = PATCH_MIN + (PATCH_MAX - PATCH_MIN) * ((z[:, :2] + 1.1) / 2.2).clamp(0.0, 1.0)
        return render_patch_images(uv)

    patch_end = _patch_smooth(z_patch)
    pos_patch = score_img_patch_shift(z_patch, patch_end, _patch_smooth)
    controls["patch_smooth_oracle"] = {
        "expected_pass": True,
        "passed": pos_patch["passed"],
        "failed_bounds": pos_patch["failed_bounds"],
    }

    # Pixel-space ghost blend: exact endpoints, linear pixel average between nearest 2 latent particles.
    def _patch_ghost_blend(z_query: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(z_query.to(torch.float64), z_patch.to(torch.float64))
        top2 = d.topk(2, largest=False)
        d0, d1 = top2.values[:, 0], top2.values[:, 1]
        i0, i1 = top2.indices[:, 0], top2.indices[:, 1]
        w1 = (d0 / (d0 + d1 + 1e-8)).to(torch.float32).view(-1, 1, 1, 1)
        return (1.0 - w1) * patch_end[i0] + w1 * patch_end[i1]

    neg_patch_ghost = score_img_patch_shift(z_patch, patch_end, _patch_ghost_blend)
    controls["patch_ghost_blend_negative"] = {
        "expected_pass": False,
        "passed": neg_patch_ghost["passed"],
        "failed_bounds": neg_patch_ghost["failed_bounds"],
    }

    neg_patch_snap = score_img_patch_shift(z_patch, patch_end, _voronoi_lookup(z_patch, patch_end))
    controls["patch_voronoi_snap_negative"] = {
        "expected_pass": False,
        "passed": neg_patch_snap["passed"],
        "failed_bounds": neg_patch_snap["failed_bounds"],
    }

    # --- 3. Saddle Barycentric Controls ---
    sx = torch.linspace(-0.88, 0.88, 6)
    sy = torch.linspace(-0.88, 0.88, 4)
    sgx, sgy = torch.meshgrid(sx, sy, indexing="ij")
    z_sad = torch.stack((sgx.reshape(-1), sgy.reshape(-1)), dim=-1)

    def _saddle_smooth(z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0].clamp(-1.0, 1.0)
        y = z[:, 1].clamp(-1.0, 1.0)
        return torch.stack((x, y, saddle_height(x, y)), dim=-1)

    sad_end = _saddle_smooth(z_sad)
    pos_sad = score_saddle_barycentric(z_sad, sad_end, _saddle_smooth)
    controls["saddle_smooth_oracle"] = {
        "expected_pass": True,
        "passed": pos_sad["passed"],
        "failed_bounds": pos_sad["failed_bounds"],
    }

    # 3D Euclidean simplex chord cutter: exact endpoints on saddle, linear 3D blend at interior points.
    def _saddle_linear_simplex(z_query: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(z_query.to(torch.float64), z_sad.to(torch.float64))
        top3 = d.topk(3, largest=False)
        if float(top3.values[:, 0].max().item()) < 1e-7:
            return sad_end[top3.indices[:, 0]]
        inv = 1.0 / (top3.values + 1e-4)
        weights = (inv / inv.sum(dim=1, keepdim=True)).to(torch.float32)
        verts = sad_end[top3.indices]
        return (weights.unsqueeze(-1) * verts).sum(dim=1)

    neg_sad_lin = score_saddle_barycentric(z_sad, sad_end, _saddle_linear_simplex)
    controls["saddle_linear_simplex_negative"] = {
        "expected_pass": False,
        "passed": neg_sad_lin["passed"],
        "failed_bounds": neg_sad_lin["failed_bounds"],
    }

    def _saddle_flat(z_query: torch.Tensor) -> torch.Tensor:
        x = z_query[:, 0].clamp(-1.0, 1.0)
        y = z_query[:, 1].clamp(-1.0, 1.0)
        return torch.stack((x, y, torch.zeros_like(x)), dim=-1)

    neg_sad_flat = score_saddle_barycentric(z_sad, _saddle_flat(z_sad), _saddle_flat)
    controls["saddle_flat_plane_negative"] = {
        "expected_pass": False,
        "passed": neg_sad_flat["passed"],
        "failed_bounds": neg_sad_flat["failed_bounds"],
    }

    neg_sad_snap = score_saddle_barycentric(z_sad, sad_end, _voronoi_lookup(z_sad, sad_end))
    controls["saddle_voronoi_snap_negative"] = {
        "expected_pass": False,
        "passed": neg_sad_snap["passed"],
        "failed_bounds": neg_sad_snap["failed_bounds"],
    }

    return controls


def list_cases() -> list[dict]:
    return [
        {
            "id": "api-interp-sparse-arc",
            "title": "Interpolated sparse particles on a 270-degree 2D circular arc (no MoG)",
            "goal": (
                "Learn 12 zero-sigma ParticlePrior locations on a 270-degree circular arc (R=2.0, sigma=0.03) "
                "so unseen latent k-NN chords stay on the curved arc, bridge >=22/32 angular bins, and avoid Voronoi snapping."
            ),
            "kind": "vector",
            "scope": "Public API stretch variant: zero-sigma ParticlePrior latent chord interpolation",
            "legacy_ids": ["stretch-interp-sparse-arc"],
            "default_steps": 600,
            "batch_size": 128,
            "eval_samples": 512,
            "evaluation_observations": 24,
            "terminal_observations": 5,
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS["interp_sparse_arc"]},
            "sampling": "Enumerated clean zero-sigma ParticlePrior endpoints plus deterministic latent k-NN chords",
        },
        {
            "id": "api-interp-img-patch-shift",
            "title": "Interpolated discrete particles on a continuous 8x8 single-patch manifold (no MoG)",
            "goal": (
                "Learn 16 zero-sigma ParticlePrior codes on continuous 8x8 single-patch images so unseen latent "
                "k-NN chords translate one sharp bright patch (peak>=0.78, rmse<=0.10) instead of double-exposed 0.5 ghosts."
            ),
            "kind": "image",
            "scope": "Public API stretch variant: zero-sigma ParticlePrior image latent interpolation",
            "legacy_ids": ["stretch-interp-img-patch-shift"],
            "default_steps": 800,
            "batch_size": 64,
            "eval_samples": 512,
            "evaluation_observations": 24,
            "terminal_observations": 5,
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS["interp_img_patch_shift"]},
            "sampling": "Enumerated clean zero-sigma ParticlePrior endpoints plus deterministic latent k-NN chords",
        },
        {
            "id": "api-interp-saddle-barycentric",
            "title": "Interpolated 2D simplex barycenters on a curved 3D saddle sheet (no MoG)",
            "goal": (
                "Learn 24 zero-sigma ParticlePrior locations on a non-harmonic 3D saddle surface so unseen 2D "
                "latent k-NN triangle barycenters stay on the curved surface (mae<=0.09, hq>=0.78) without snapping."
            ),
            "kind": "vector",
            "scope": "Public API stretch variant: zero-sigma ParticlePrior 2D simplex barycenter interpolation",
            "legacy_ids": ["stretch-interp-saddle-barycentric"],
            "default_steps": 800,
            "batch_size": 128,
            "eval_samples": 512,
            "evaluation_observations": 24,
            "terminal_observations": 5,
            "thresholds": {name: f"{op} {bound}" for name, op, bound in THRESHOLDS["interp_saddle_barycentric"]},
            "sampling": "Enumerated clean zero-sigma ParticlePrior endpoints plus deterministic latent k-NN simplex barycenters",
        },
    ]


_CASE_TO_KIND = {spec["case_id"]: kind for kind, spec in HOST_DEFINITIONS.items()}


def _seed_int(tag: str, seed: int) -> int:
    digest = hashlib.sha256(f"api_interpolation:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


class InterpolationToyFixture:
    """Public-API GANTrainer fixture for zero-sigma ParticlePrior interpolation toys."""

    api_components = ("GANTrainer", "ParticlePrior", "Recipe", "init.deterministic_orthogonal_")

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
            "prior_kind": "particles",
            "sigma_rel": 0.0,
            "standardize": False,
            **(recipe_overrides or {}),
        }
        if base_recipe.total_steps is not None:
            updates["total_steps"] = self.limit
        self.recipe = base_recipe.replace(**updates)
        if self.recipe.prior_code_law != "table":
            raise ValueError("particle interpolation requires Recipe.prior_code_law == 'table'")

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
        real_cpu = sample_target_for_kind(self.kind, self.recipe.batch_size, self.data_rng, device="cpu")
        self._batch_hasher.update(real_cpu.numpy().tobytes())
        out = self.trainer.step(real_cpu.to(self.device))
        return {k: float(v.item()) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    @torch.no_grad()
    def observe(self, n: int = 512, seed: int = 10000) -> dict:
        eval_stream = torch.Generator(device=self.device).manual_seed(int(seed))
        ref_stream = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        endpoints = self.trainer.sample(
            self.recipe.num_particles, fixed_first_n=True, output_noise=False, generator=eval_stream
        )
        z_table = unwrap_prior(self.trainer.prior).z.detach()
        reference = sample_target_for_kind(self.kind, int(n), ref_stream, device="cpu")
        return score_interpolation_kind(
            self.kind,
            z_table,
            endpoints,
            lambda codes: evaluate_clean_codes(self.trainer.G, codes, prior=self.trainer.prior),
            reference=reference,
        )

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
) -> InterpolationToyFixture:
    del kwargs
    if case_id not in _CASE_TO_KIND:
        raise ValueError(f"unknown interpolation toy case: {case_id!r}")
    return InterpolationToyFixture(
        next(c for c in list_cases() if c["id"] == case_id),
        recipe_name=recipe_name or "empf",
        recipe=recipe,
        recipe_overrides=recipe_overrides,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
