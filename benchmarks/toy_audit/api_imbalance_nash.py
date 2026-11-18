"""Public-API stretch experiments for imbalanced & non-complementary (G, D) Nash convergence.

Scientific question (Theory 4A): when Generator and Discriminator capacities are
severely imbalanced (e.g. 16x width / 2x depth asymmetry -> >600x parameter-count
ratio, in BOTH D >> G and G >> D directions) AND initialized non-complementarily
(compressed or amplified generator shifted far off-manifold into a corner while
the critic is initialized with a sharp upside-down potential tilt pointing away
from the target manifold so step-0 margins are ~-30 logits), does an SVD-free
formulation still converge cleanly to the true multimodal Nash equilibrium?
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn

from benchmarks.toy_audit.ring16_quality import score_samples as ring16_score_samples
from benchmarks.transfer_suite.vector_tasks import (
    sample_target as vector_sample_target,
    score_samples as vector_score_samples,
)
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import init


class HostileImbalancedGenerator(nn.Module):
    """MLP generator with explicit constructor-preserved scale compression/amplification and off-manifold shift."""

    def __init__(
        self,
        *,
        z_dim: int,
        hidden: int,
        layers: int,
        out_dim: int,
        output_scale: float,
        output_offset: list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(output_offset) != out_dim:
            raise ValueError("output_offset length must equal out_dim")
        if not math.isfinite(output_scale) or output_scale <= 0.0:
            raise ValueError("output_scale must be finite and positive")
        self.backbone = SimpleMLPGenerator(
            z_dim=int(z_dim),
            hidden_dim=int(hidden),
            n_hidden=int(layers),
            out_dim=int(out_dim),
        )
        self.register_buffer("output_scale", torch.tensor(float(output_scale), dtype=torch.float32))
        self.shift_bias = nn.Parameter(torch.tensor([float(v) for v in output_offset], dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(z)
        return raw * self.output_scale.to(dtype=raw.dtype) + self.shift_bias.to(dtype=raw.dtype)


class HostileImbalancedCritic(nn.Module):
    """MLP critic with explicit constructor-preserved logit gain and non-complementary linear tilt."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden: int,
        layers: int,
        fourier: int,
        logit_gain: float,
        hostile_tilt: list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(hostile_tilt) != in_dim:
            raise ValueError("hostile_tilt length must equal in_dim")
        if not math.isfinite(logit_gain) or logit_gain <= 0.0:
            raise ValueError("logit_gain must be finite and positive")
        self.backbone = SimpleMLPDiscriminator(
            in_dim=int(in_dim),
            hidden_dim=int(hidden),
            n_hidden=int(layers),
            fourier=int(fourier),
        )
        self.register_buffer("logit_gain", torch.tensor(float(logit_gain), dtype=torch.float32))
        self.tilt_weight = nn.Parameter(torch.tensor([float(v) for v in hostile_tilt], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(x)
        x_flat = x.reshape(x.shape[0], -1).to(dtype=raw.dtype)
        tilt = (x_flat * self.tilt_weight.to(dtype=raw.dtype).unsqueeze(0)).sum(dim=1)
        if raw.ndim == 2 and raw.shape[1] == 1:
            return raw * self.logit_gain.to(dtype=raw.dtype) + tilt.unsqueeze(1)
        return raw.reshape(-1) * self.logit_gain.to(dtype=raw.dtype) + tilt


init.register(HostileImbalancedGenerator, {"shift_bias": init.KEEP})
init.register(HostileImbalancedCritic, {"tilt_weight": init.KEEP})


def _ring16_target_spec(*, radius: float = 3.0, sigma: float = 0.10, steps: int = 1600) -> dict:
    means = [
        [
            round(radius * math.cos(2.0 * math.pi * k / 16.0), 10),
            round(radius * math.sin(2.0 * math.pi * k / 16.0), 10),
        ]
        for k in range(16)
    ]
    var = round(sigma * sigma, 10)
    covs = [[[var, 0.0], [0.0, var]] for _ in range(16)]
    masses = [0.0625] * 16
    return {
        "kind": "gaussian_mixture",
        "identifiable": True,
        "particles": 256,
        "steps": int(steps),
        "means": means,
        "covariances": covs,
        "masses": masses,
        "thresholds": [],
    }


def _unequal_width_target_spec(*, steps: int = 1100) -> dict:
    corners = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
    widths = [0.07, 0.12, 0.20, 0.30]
    covs = [[[round(s * s, 10), 0.0], [0.0, round(s * s, 10)]] for s in widths]
    return {
        "kind": "gaussian_mixture",
        "identifiable": True,
        "particles": 256,
        "steps": int(steps),
        "means": corners,
        "covariances": covs,
        "masses": [0.25, 0.25, 0.25, 0.25],
        "thresholds": [],
    }


THRESHOLDS: dict[str, list[list]] = {
    "nash_overpowered_critic_ring16": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 16],
        ["hq", ">=", 0.80],
        ["mass_tv", "<=", 0.18],
        ["sw1_normalized", "<=", 0.12],
        ["component_core_covariance_error", "<=", 0.98],
        ["component_core_min_eigen_ratio", ">=", 0.15],
    ],
    "nash_overpowered_generator_ring16": [
        ["finite_fraction", "==", 1.0],
        ["modes", "==", 16],
        ["hq", ">=", 0.72],
        ["mass_tv", "<=", 0.18],
        ["sw1_normalized", "<=", 0.12],
        ["component_core_covariance_error", "<=", 1.28],
        ["component_core_min_eigen_ratio", ">=", 0.15],
    ],
    "nash_unequal_width_hostile_imbalance": [
        ["finite_fraction", "==", 1.0],
        ["min_mass_ratio", ">=", 0.60],
        ["hq", ">=", 0.82],
        ["mass_tv", "<=", 0.18],
        ["sw1_normalized", "<=", 0.16],
        ["resolved_core_covariance_error", "<=", 0.65],
        ["resolved_core_min_eigen_ratio", ">=", 0.15],
    ],
}


HOST_DEFINITIONS: dict[str, dict] = {
    "nash_overpowered_critic_ring16": {
        "case_id": "api-nash-overpowered-critic-ring16",
        "kind": "nash_overpowered_critic_ring16",
        "target": "16-mode equal-mass isotropic Gaussian ring (R=3.0, sigma=0.10) under 16x overpowered tilted critic vs compressed off-manifold generator",
        "target_spec": _ring16_target_spec(steps=1600),
        "resources": {"num_particles": 256, "z_dim": 4, "batch_size": 128},
        "generator": {
            "family": "hostile_imbalanced_mlp",
            "hidden": 32,
            "layers": 2,
            "out_dim": 2,
            "output_scale": 0.08,
            "output_offset": [-4.5, -4.5],
        },
        "discriminator": {
            "family": "hostile_imbalanced_mlp",
            "hidden": 512,
            "layers": 4,
            "in_dim": 2,
            "fourier": 3,
            "logit_gain": 3.5,
            "hostile_tilt": [-3.5, -3.5],
        },
        "steps": 1600,
    },
    "nash_overpowered_generator_ring16": {
        "case_id": "api-nash-overpowered-generator-ring16",
        "kind": "nash_overpowered_generator_ring16",
        "target": "16-mode equal-mass isotropic Gaussian ring (R=3.0, sigma=0.10) under 16x overpowered amplified off-manifold generator vs tiny underpowered tilted critic",
        "target_spec": _ring16_target_spec(steps=1600),
        "resources": {"num_particles": 256, "z_dim": 4, "batch_size": 128},
        "generator": {
            "family": "hostile_imbalanced_mlp",
            "hidden": 512,
            "layers": 4,
            "out_dim": 2,
            "output_scale": 2.2,
            "output_offset": [4.5, -4.5],
        },
        "discriminator": {
            "family": "hostile_imbalanced_mlp",
            "hidden": 32,
            "layers": 2,
            "in_dim": 2,
            "fourier": 2,
            "logit_gain": 0.25,
            "hostile_tilt": [-3.5, 3.5],
        },
        "steps": 1600,
    },
    "nash_unequal_width_hostile_imbalance": {
        "case_id": "api-nash-unequal-width-hostile-imbalance",
        "kind": "nash_unequal_width_hostile_imbalance",
        "target": "4-mode unequal-width Gaussian mixture (sigma in {0.07, 0.12, 0.20, 0.30}) under 16x overpowered tilted critic vs compressed off-manifold generator",
        "target_spec": _unequal_width_target_spec(steps=1100),
        "resources": {"num_particles": 256, "z_dim": 4, "batch_size": 128},
        "generator": {
            "family": "hostile_imbalanced_mlp",
            "hidden": 32,
            "layers": 2,
            "out_dim": 2,
            "output_scale": 0.08,
            "output_offset": [-4.0, 4.0],
        },
        "discriminator": {
            "family": "hostile_imbalanced_mlp",
            "hidden": 512,
            "layers": 4,
            "in_dim": 2,
            "fourier": 3,
            "logit_gain": 3.5,
            "hostile_tilt": [-3.5, 3.5],
        },
        "steps": 1100,
    },
}


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


def build_models_for_spec(spec: dict, *, z_dim: int) -> tuple[nn.Module, nn.Module]:
    g_cfg = spec["generator"]
    d_cfg = spec["discriminator"]
    if g_cfg.get("family") != "hostile_imbalanced_mlp" or d_cfg.get("family") != "hostile_imbalanced_mlp":
        raise ValueError("unsupported imbalance_nash model family")
    g_mod = HostileImbalancedGenerator(
        z_dim=int(z_dim),
        hidden=int(g_cfg["hidden"]),
        layers=int(g_cfg["layers"]),
        out_dim=int(g_cfg["out_dim"]),
        output_scale=float(g_cfg["output_scale"]),
        output_offset=list(g_cfg["output_offset"]),
    )
    d_mod = HostileImbalancedCritic(
        in_dim=int(d_cfg["in_dim"]),
        hidden=int(d_cfg["hidden"]),
        layers=int(d_cfg["layers"]),
        fourier=int(d_cfg["fourier"]),
        logit_gain=float(d_cfg["logit_gain"]),
        hostile_tilt=list(d_cfg["hostile_tilt"]),
    )
    return g_mod, d_mod


def sample_target_for_kind(
    kind: str,
    n: int,
    generator: torch.Generator,
    *,
    completed_steps: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if kind not in HOST_DEFINITIONS:
        raise ValueError(f"unsupported imbalance_nash task kind: {kind!r}")
    target_spec = HOST_DEFINITIONS[kind]["target_spec"]
    pts = vector_sample_target(target_spec, int(n), generator, int(completed_steps))
    return pts.to(device=torch.device(device), dtype=torch.float32)


@torch.no_grad()
def score_imbalance_nash_kind(
    kind: str,
    samples: torch.Tensor,
    *,
    completed_steps: int = 900,
    reference: torch.Tensor | None = None,
) -> dict:
    if kind not in HOST_DEFINITIONS:
        raise ValueError(f"unsupported imbalance_nash task kind: {kind!r}")
    spec = HOST_DEFINITIONS[kind]
    target_spec = deepcopy(spec["target_spec"])
    target_spec["thresholds"] = [row for row in THRESHOLDS[kind] if row[0] != "finite_fraction"]

    pts = samples.detach().cpu().to(dtype=torch.float32)
    finite_fraction = float(torch.isfinite(pts).all(dim=-1).float().mean().item()) if len(pts) > 0 else 0.0

    ref = reference
    if ref is None:
        gen = torch.Generator(device="cpu").manual_seed(94001)
        ref = sample_target_for_kind(kind, min(1024, max(512, len(pts))), gen, completed_steps=completed_steps)
    ref_cpu = ref.detach().cpu().to(dtype=torch.float32)

    is_ring = kind in ("nash_overpowered_critic_ring16", "nash_overpowered_generator_ring16")
    zoom_lim = [-3.6, 3.6] if is_ring else [-2.4, 2.4]
    arena_lim = [-6.2, 6.2]

    if finite_fraction < 1.0 or pts.ndim != 2 or pts.shape[1] != 2:
        metrics = {
            "finite_fraction": finite_fraction if (pts.ndim == 2 and pts.shape[1] == 2) else 0.0,
            "modes": 0,
            "min_mass_ratio": 0.0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "sw1_normalized": 9.9,
            "component_core_covariance_error": 9.9,
            "component_core_min_eigen_ratio": 0.0,
            "resolved_core_covariance_error": 9.9,
            "resolved_core_min_eigen_ratio": 0.0,
            "component_covariance_error": 99.0,
            "component_min_eigen_ratio": 0.0,
            "max_component_spill": 1.0,
        }
        failed = _check_thresholds(metrics, kind)
        zeros = torch.zeros((2, 2), dtype=torch.float32)
        return {
            "passed": False,
            "failed_bounds": failed,
            "metrics": metrics,
            "views": [
                {"kind": "scatter", "title": "Full arena (hostile init recovery)", "target": ref_cpu[:512], "samples": zeros},
                {"kind": "scatter", "title": "Target manifold close-up", "target": ref_cpu[:512], "samples": zeros},
            ],
        }

    scored = (
        ring16_score_samples(pts, target_spec, int(completed_steps))
        if is_ring
        else vector_score_samples(pts, target_spec, int(completed_steps))
    )
    metrics = {"finite_fraction": 1.0, **scored}
    failed = _check_thresholds(metrics, kind)
    vis_pts = pts[:512]
    vis_ref = ref_cpu[:512]
    offset_str = f"({spec['generator']['output_offset'][0]:+.1f}, {spec['generator']['output_offset'][1]:+.1f})"
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": [
            {
                "kind": "scatter",
                "title": f"Full arena: recovery from hostile init {offset_str}",
                "target": vis_ref,
                "samples": vis_pts,
                "xlim": arena_lim,
                "ylim": arena_lim,
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Non-complementary initial generator cloud vs upside-down tilted critic potential.",
            },
            {
                "kind": "scatter",
                "title": "Target manifold equilibrium coverage & shape",
                "target": vis_ref,
                "samples": vis_pts,
                "xlim": zoom_lim,
                "ylim": zoom_lim,
                "aspect": "equal",
                "xlabel": "x1",
                "ylabel": "x2",
                "caption": "Nash equilibrium requires full mode recall, balanced mass, and calibrated core covariances.",
            },
        ],
    }


def imbalance_nash_oracle_controls() -> dict[str, dict]:
    """Analytic positive and destructive negative controls verifying metric selectivity."""
    controls: dict[str, dict] = {}
    n = 4096

    # 1. Exact target oracle draws must PASS all 3 tasks.
    for idx, kind in enumerate(
        (
            "nash_overpowered_critic_ring16",
            "nash_overpowered_generator_ring16",
            "nash_unequal_width_hostile_imbalance",
        )
    ):
        gen = torch.Generator(device="cpu").manual_seed(95100 + idx)
        exact = sample_target_for_kind(kind, n, gen, completed_steps=900)
        res = score_imbalance_nash_kind(kind, exact, completed_steps=900)
        controls[f"{kind}_exact_oracle"] = {
            "expected_pass": True,
            "passed": res["passed"],
            "failed_bounds": res["failed_bounds"],
        }

    # 2. Stuck at hostile off-manifold initial corner (-4.5, -4.5) -> must FAIL.
    gen_stuck = torch.Generator(device="cpu").manual_seed(95201)
    stuck_pts = torch.tensor([-4.5, -4.5]) + 0.08 * torch.randn((n, 2), generator=gen_stuck)
    stuck_res = score_imbalance_nash_kind("nash_overpowered_critic_ring16", stuck_pts)
    controls["ring16_stuck_hostile_corner"] = {
        "expected_pass": False,
        "passed": stuck_res["passed"],
        "failed_bounds": stuck_res["failed_bounds"],
    }

    # 3. Single missing mode on 16-mode ring (15/16 modes covered) -> must FAIL modes == 16.
    ring_spec = deepcopy(HOST_DEFINITIONS["nash_overpowered_critic_ring16"]["target_spec"])
    ring_spec_15 = deepcopy(ring_spec)
    ring_spec_15["means"] = ring_spec["means"][:15]
    ring_spec_15["covariances"] = ring_spec["covariances"][:15]
    ring_spec_15["masses"] = [1.0 / 15.0] * 15
    gen_15 = torch.Generator(device="cpu").manual_seed(95202)
    pts_15 = vector_sample_target(ring_spec_15, n, gen_15, 900)
    res_15 = score_imbalance_nash_kind("nash_overpowered_generator_ring16", pts_15)
    controls["ring16_missing_one_mode"] = {
        "expected_pass": False,
        "passed": res_15["passed"],
        "failed_bounds": res_15["failed_bounds"],
    }

    # 4. Collapsed within-mode covariance (sigma=0.01 instead of {0.07, 0.12, 0.20, 0.30}) -> must FAIL shape gates.
    uw_collapsed_spec = deepcopy(HOST_DEFINITIONS["nash_unequal_width_hostile_imbalance"]["target_spec"])
    uw_collapsed_spec["covariances"] = [[[1e-4, 0.0], [0.0, 1e-4]] for _ in range(4)]
    gen_col = torch.Generator(device="cpu").manual_seed(95203)
    pts_col = vector_sample_target(uw_collapsed_spec, n, gen_col, 900)
    res_col = score_imbalance_nash_kind("nash_unequal_width_hostile_imbalance", pts_col)
    controls["unequal_width_collapsed_variance"] = {
        "expected_pass": False,
        "passed": res_col["passed"],
        "failed_bounds": res_col["failed_bounds"],
    }

    # 5. Severe mass imbalance on unequal-width mixture ([0.70, 0.10, 0.10, 0.10]) -> must FAIL mass_tv & min_mass_ratio.
    uw_biased_spec = deepcopy(HOST_DEFINITIONS["nash_unequal_width_hostile_imbalance"]["target_spec"])
    uw_biased_spec["masses"] = [0.70, 0.10, 0.10, 0.10]
    gen_bias = torch.Generator(device="cpu").manual_seed(95204)
    pts_bias = vector_sample_target(uw_biased_spec, n, gen_bias, 900)
    res_bias = score_imbalance_nash_kind("nash_unequal_width_hostile_imbalance", pts_bias)
    controls["unequal_width_biased_mass"] = {
        "expected_pass": False,
        "passed": res_bias["passed"],
        "failed_bounds": res_bias["failed_bounds"],
    }

    # 6. Non-finite samples -> must FAIL finite_fraction == 1.0.
    bad_pts = exact.clone()
    bad_pts[0, 0] = float("nan")
    res_nan = score_imbalance_nash_kind("nash_unequal_width_hostile_imbalance", bad_pts)
    controls["nonfinite_output"] = {
        "expected_pass": False,
        "passed": res_nan["passed"],
        "failed_bounds": res_nan["failed_bounds"],
    }
    return controls
