"""Public-API Theory-3 causal isolation toy experiments (Theory-1-free & Theory-2-free).

Scientific question: Is Problem #3 (Scale-Mass Gauge Non-Identifiability &
Unbalanced Horizon Starvation) a genuine 3rd fundamental failure mode of GANs,
strictly orthogonal to Theory 1 (continuous generator topology / ReLU bridges)
and Theory 2 (unbounded discriminator Lipschitz sharpness)?

Both benchmarks eliminate Theory 1 by construction (``G = nn.Identity()`` with
zero generator parameters over a direct data-space zero-sigma ``ParticlePrior``)
and verify Theory 2 explicitly at every observation via a smooth C^infinity
``SiLU`` MLP critic gated by ``critic_grad_med <= 1.0``:
1. ``t3_iso_scale_mass_8x``: 4-mode heteroscedastic Gaussian mixture in R^2
   with 8x mode-width disparity (sigma = [0.04, 0.08, 0.16, 0.32]) and 6x
   unequal mode weights (pi = [0.48, 0.26, 0.18, 0.08]).
2. ``t3_iso_horizon_saddle_shift``: 2 -> 4 equidistant orthogonal mode expansion
   in R^4 at step 300 (+-1.5 e_1 -> (+-1.5 e_1, +-1.5 e_2), sigma = 0.05,
   42.4-sigma inter-mode valley) with a paired step-300 frozen control.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

import torch
from torch import nn

from particlegan import (
    InputNoise,
    ParticlePrior,
    Recipe,
    UpdatePolicy,
    init,
    learning_rate_scales,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward

SCALE_MASS_WEIGHTS = (0.48, 0.26, 0.18, 0.08)
SCALE_MASS_RAW_CORNERS = (
    (-1.2, -1.2),
    (-1.2, 1.2),
    (1.2, -1.2),
    (1.2, 1.2),
)
SCALE_MASS_SIGMAS = (0.04, 0.08, 0.16, 0.32)

SADDLE_SHIFT_STEP = 300
SADDLE_SIGMA = 0.05
SADDLE_PRE_MEANS = (
    (1.5, 0.0, 0.0, 0.0),
    (-1.5, 0.0, 0.0, 0.0),
)
SADDLE_PRE_WEIGHTS = (0.5, 0.5)
SADDLE_POST_MEANS = (
    (1.5, 0.0, 0.0, 0.0),
    (-1.5, 0.0, 0.0, 0.0),
    (0.0, 1.5, 0.0, 0.0),
    (0.0, -1.5, 0.0, 0.0),
)
SADDLE_POST_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

PRIOR_EXCEPTION_REASON = (
    "Direct data-space zero-sigma ParticlePrior with parameterless G = nn.Identity() "
    "required to eliminate Theory-1 continuous generator topology completely while isolating Theory-3 dynamics."
)

THRESHOLDS: dict[str, list[list[Any]]] = {
    "t3_iso_scale_mass_8x": [
        ["finite_fraction", "==", 1.0],
        ["critic_grad_med", "<=", 1.0],
        ["modes", "==", 4],
        ["mass_tv", "<=", 0.10],
        ["hq", ">=", 0.94],
        ["core_component_covariance_error", "<=", 0.45],
        ["core_min_eigen_ratio", ">=", 0.45],
    ],
    "t3_iso_horizon_saddle_shift": [
        ["finite_fraction", "==", 1.0],
        ["critic_grad_med", "<=", 1.0],
        ["pre_shift_hq", ">=", 0.92],
        ["modes", "==", 4],
        ["mass_tv", "<=", 0.12],
        ["hq", ">=", 0.92],
        ["frozen_control_modes", "==", 2],
    ],
}

HOST_DEFINITIONS: dict[str, dict[str, Any]] = {
    "t3_iso_scale_mass_8x": {
        "case_id": "api-t3-iso-scale-mass-8x",
        "kind": "t3_iso_scale_mass_8x",
        "target": "Zero-mean 4-mode R^2 Gaussian mixture with 8x width ratio sigma=[0.04,0.08,0.16,0.32] and 6x mass ratio pi=[0.48,0.26,0.18,0.08]",
        "resources": {"num_particles": 512, "z_dim": 2, "batch_size": 128},
        "model": {
            "generator": "nn.Identity (0 trainable parameters; direct data-space ParticlePrior)",
            "discriminator_family": "smooth_silu_mlp",
            "hidden": 64,
            "layers": 2,
            "in_dim": 2,
        },
        "steps": 800,
    },
    "t3_iso_horizon_saddle_shift": {
        "case_id": "api-t3-iso-horizon-saddle-shift",
        "kind": "t3_iso_horizon_saddle_shift",
        "target": "R^4 orthogonal mode expansion +-1.5 e_1 -> (+-1.5 e_1, +-1.5 e_2) at step 300 (sigma=0.05, 42.4-sigma valley) with step-300 frozen control",
        "resources": {"num_particles": 128, "z_dim": 4, "batch_size": 128},
        "model": {
            "generator": "nn.Identity (0 trainable parameters; direct data-space ParticlePrior)",
            "discriminator_family": "smooth_silu_mlp",
            "hidden": 64,
            "layers": 2,
            "in_dim": 4,
        },
        "shift_step": SADDLE_SHIFT_STEP,
        "steps": 1000,
    },
}


def scale_mass_parameters(device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    weights = torch.tensor(SCALE_MASS_WEIGHTS, dtype=torch.float32, device=dev)
    raw = torch.tensor(SCALE_MASS_RAW_CORNERS, dtype=torch.float32, device=dev)
    means = raw - (weights.unsqueeze(1) * raw).sum(dim=0, keepdim=True)
    sigmas = torch.tensor(SCALE_MASS_SIGMAS, dtype=torch.float32, device=dev)
    return weights, means, sigmas


def saddle_shift_parameters(
    step: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    if int(step) <= SADDLE_SHIFT_STEP:
        weights = torch.tensor(SADDLE_PRE_WEIGHTS, dtype=torch.float32, device=dev)
        means = torch.tensor(SADDLE_PRE_MEANS, dtype=torch.float32, device=dev)
    else:
        weights = torch.tensor(SADDLE_POST_WEIGHTS, dtype=torch.float32, device=dev)
        means = torch.tensor(SADDLE_POST_MEANS, dtype=torch.float32, device=dev)
    sigmas = torch.full((len(weights),), SADDLE_SIGMA, dtype=torch.float32, device=dev)
    return weights, means, sigmas


def sample_mixture(
    weights: torch.Tensor,
    means: torch.Tensor,
    sigmas: torch.Tensor,
    n: int,
    generator: torch.Generator,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    dev = torch.device(device)
    w_cpu = weights.detach().to(device=generator.device, dtype=torch.float32)
    idx = torch.multinomial(w_cpu, int(n), replacement=True, generator=generator).to(dev)
    eps = torch.randn(
        (int(n), means.shape[1]),
        generator=generator,
        device=generator.device,
        dtype=torch.float32,
    ).to(dev)
    return means.to(dev)[idx] + sigmas.to(dev)[idx].unsqueeze(1) * eps


def sample_target_for_kind(
    kind: str,
    n: int,
    generator: torch.Generator,
    *,
    step: int = 1000,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if kind == "t3_iso_scale_mass_8x":
        weights, means, sigmas = scale_mass_parameters(device)
        return sample_mixture(weights, means, sigmas, n, generator, device=device)
    if kind == "t3_iso_horizon_saddle_shift":
        weights, means, sigmas = saddle_shift_parameters(step, device)
        return sample_mixture(weights, means, sigmas, n, generator, device=device)
    raise ValueError(f"unsupported Theory-3 isolation kind: {kind!r}")


class SmoothSiLUCritic(nn.Module):
    """C^infinity smooth MLP critic eliminating ReLU kink artifacts (Theory 2 clean)."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        cur = int(in_dim)
        for _ in range(int(layers)):
            blocks.append(nn.Linear(cur, int(hidden)))
            blocks.append(nn.SiLU())
            cur = int(hidden)
        blocks.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


def build_models_for_spec(spec: dict[str, Any]) -> tuple[nn.Module, nn.Module]:
    model_cfg = spec["model"]
    if model_cfg.get("discriminator_family") != "smooth_silu_mlp":
        raise ValueError(f"unsupported discriminator family: {model_cfg.get('discriminator_family')!r}")
    g = nn.Identity()
    d = SmoothSiLUCritic(
        in_dim=int(model_cfg["in_dim"]),
        hidden=int(model_cfg["hidden"]),
        layers=int(model_cfg["layers"]),
    )
    return g, d


def measure_critic_grad_median(
    critic_fn: Callable[[torch.Tensor], torch.Tensor],
    particles: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """Measure median ||grad_x D(x)||_2 across deterministic sample-reference midpoints."""
    n = min(len(particles), len(reference), 128)
    if n < 1 or not bool(torch.isfinite(particles).all()):
        return float("nan")
    dev = particles.device
    pts = particles.detach()[:n].float()
    ref = reference.detach()[:n].to(device=dev, dtype=torch.float32)
    alpha = torch.linspace(0.15, 0.85, n, device=dev, dtype=torch.float32).unsqueeze(1)
    probes = ((1.0 - alpha) * pts + alpha * ref).detach().requires_grad_(True)
    with torch.enable_grad():
        out = critic_fn(probes)
        grad = torch.autograd.grad(out.sum(), probes, create_graph=False)[0]
    norms = grad.reshape(n, -1).norm(dim=1)
    if not bool(torch.isfinite(norms).all()):
        return float("nan")
    return float(norms.median().item())


def _check_thresholds(metrics: dict[str, Any], kind: str) -> list[str]:
    failed: list[str] = []
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


@torch.no_grad()
def _mixture_mode_metrics(
    particles: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    min_mode_share: float,
) -> dict[str, float]:
    n = len(particles)
    m_cnt = len(weights)
    dim = means.shape[1]
    finite_mask = torch.isfinite(particles).all(dim=-1)
    all_finite = bool(finite_mask.all().item()) if n > 0 else False
    finite_fraction = 1.0 if all_finite else (float(finite_mask.float().mean().item()) if n > 0 else 0.0)
    if n == 0 or not all_finite:
        return {
            "finite_fraction": finite_fraction,
            "modes": 0,
            "mass_tv": 1.0,
            "hq": 0.0,
            "core_component_covariance_error": float("inf"),
            "core_min_eigen_ratio": 0.0,
            "full_component_covariance_error": float("inf"),
        }

    pts = particles.detach().to(device=means.device, dtype=torch.float32)
    dists = torch.cdist(pts, means)
    assign = dists.argmin(dim=1)
    counts = torch.bincount(assign, minlength=m_cnt).float() / float(n)
    mass_tv = float(0.5 * (counts - weights).abs().sum().item())

    z_norm = (pts - means[assign]).norm(dim=1) / sigmas[assign]
    in_core = z_norm <= 3.5
    hq = float(in_core.float().mean().item())

    q_counts = torch.bincount(assign[in_core], minlength=m_cnt).float() / float(n)
    modes = int((q_counts >= float(min_mode_share)).sum().item())

    core_cov_errs: list[float] = []
    full_cov_errs: list[float] = []
    eigen_ratios: list[float] = []
    eye_d = torch.eye(dim, device=means.device, dtype=torch.float32)
    norm_denom = math.sqrt(float(dim))

    for m in range(m_cnt):
        target_var = float(sigmas[m].item()) ** 2
        target_cov = target_var * eye_d

        pts_full = pts[assign == m]
        if len(pts_full) >= max(3, dim + 1):
            cov_f = torch.cov(pts_full.T, correction=0)
            full_cov_errs.append(float((cov_f - target_cov).norm().item() / (norm_denom * target_var)))
        else:
            full_cov_errs.append(10.0)

        pts_core = pts[(assign == m) & in_core]
        if len(pts_core) >= max(4, dim + 1):
            cov_c = torch.cov(pts_core.T, correction=0)
            core_cov_errs.append(float((cov_c - target_cov).norm().item() / (norm_denom * target_var)))
            evals = torch.linalg.eigvalsh(cov_c.double()).clamp_min(0.0).float()
            eigen_ratios.append(float(evals[0].item() / max(target_var, 1e-12)))
        else:
            core_cov_errs.append(10.0)
            eigen_ratios.append(0.0)

    return {
        "finite_fraction": finite_fraction,
        "modes": modes,
        "mass_tv": mass_tv,
        "hq": hq,
        "core_component_covariance_error": float(sum(core_cov_errs) / len(core_cov_errs)),
        "core_min_eigen_ratio": float(min(eigen_ratios)),
        "full_component_covariance_error": float(sum(full_cov_errs) / len(full_cov_errs)),
    }


def score_t3_isolation_kind(
    kind: str,
    particles: torch.Tensor,
    reference: torch.Tensor,
    *,
    critic_grad_med: float,
    pre_shift_hq: float | None = None,
    frozen_particles: torch.Tensor | None = None,
) -> dict[str, Any]:
    pts_cpu = particles.detach().to(device="cpu", dtype=torch.float32)
    ref_cpu = reference.detach().to(device="cpu", dtype=torch.float32)

    if kind == "t3_iso_scale_mass_8x":
        weights, means, sigmas = scale_mass_parameters("cpu")
        base = _mixture_mode_metrics(pts_cpu, weights, means, sigmas, min_mode_share=0.04)
        metrics = {
            **base,
            "critic_grad_med": float(critic_grad_med),
        }
        views = [
            {
                "kind": "scatter",
                "title": "8x Scale + 6x Mass Isolation (G=Identity, D Smooth)",
                "xlabel": "x",
                "ylabel": "y",
                "aspect": "equal",
                "target": ref_cpu[:, :2].numpy(),
                "samples": pts_cpu[:, :2].numpy(),
                "caption": "Direct ParticlePrior cloud vs 4-mode target (sigma=[0.04,0.08,0.16,0.32], pi=[0.48,0.26,0.18,0.08]).",
            }
        ]
    elif kind == "t3_iso_horizon_saddle_shift":
        weights_post, means_post, sigmas_post = saddle_shift_parameters(SADDLE_SHIFT_STEP + 1, "cpu")
        weights_pre, means_pre, sigmas_pre = saddle_shift_parameters(SADDLE_SHIFT_STEP, "cpu")
        if frozen_particles is None:
            pre_base = _mixture_mode_metrics(pts_cpu, weights_pre, means_pre, sigmas_pre, min_mode_share=0.20)
            eff_pre_hq = float(pre_shift_hq) if pre_shift_hq is not None else float(pre_base["hq"])
            froz_modes = 2
        else:
            froz_cpu = frozen_particles.detach().to(device="cpu", dtype=torch.float32)
            froz_post = _mixture_mode_metrics(froz_cpu, weights_post, means_post, sigmas_post, min_mode_share=0.125)
            eff_pre_hq = float(pre_shift_hq) if pre_shift_hq is not None else 0.0
            froz_modes = int(froz_post["modes"])
        post_base = _mixture_mode_metrics(pts_cpu, weights_post, means_post, sigmas_post, min_mode_share=0.125)
        metrics = {
            "finite_fraction": post_base["finite_fraction"],
            "critic_grad_med": float(critic_grad_med),
            "pre_shift_hq": eff_pre_hq,
            "modes": post_base["modes"],
            "mass_tv": post_base["mass_tv"],
            "hq": post_base["hq"],
            "frozen_control_modes": froz_modes,
            "core_component_covariance_error": post_base["core_component_covariance_error"],
        }
        views = [
            {
                "kind": "scatter",
                "title": "42.4-Sigma Orthogonal Saddle Shift (G=Identity)",
                "xlabel": "x",
                "ylabel": "y",
                "aspect": "equal",
                "target": ref_cpu[:, :2].numpy(),
                "samples": pts_cpu[:, :2].numpy(),
                "caption": "Leading (x_1, x_2) plane of R^4 cloud across step-300 2->4 orthogonal mode expansion.",
            }
        ]
    else:
        raise ValueError(f"unsupported Theory-3 isolation kind: {kind!r}")

    failed = _check_thresholds(metrics, kind)
    return {
        "passed": len(failed) == 0,
        "failed_bounds": failed,
        "metrics": metrics,
        "views": views,
    }


def t3_isolation_oracle_controls() -> dict[str, dict[str, Any]]:
    """Deterministic positive and destructive scorer controls verifying every gate."""
    rng = torch.Generator(device="cpu").manual_seed(20261118)
    ref_sm = sample_target_for_kind("t3_iso_scale_mass_8x", 2048, rng, device="cpu")
    oracle_sm = sample_target_for_kind("t3_iso_scale_mass_8x", 512, rng, device="cpu")

    # 1. Oracle samples + smooth critic (grad_med = 0.35 <= 1.0) -> PASS
    c_sm_pass = score_t3_isolation_kind("t3_iso_scale_mass_8x", oracle_sm, ref_sm, critic_grad_med=0.35)

    # 2. Uniform-mass impostor [0.25, 0.25, 0.25, 0.25] -> FAIL mass_tv <= 0.10
    _, means_sm, sigmas_sm = scale_mass_parameters("cpu")
    unif_w = torch.full((4,), 0.25)
    impostor_unif = sample_mixture(unif_w, means_sm, sigmas_sm, 512, rng, device="cpu")
    c_sm_unif = score_t3_isolation_kind("t3_iso_scale_mass_8x", impostor_unif, ref_sm, critic_grad_med=0.35)

    # 3. Sharp-critic violation (grad_med = 2.4 > 1.0) -> FAIL Theory-2 smoothness gate
    c_sm_sharp = score_t3_isolation_kind("t3_iso_scale_mass_8x", oracle_sm, ref_sm, critic_grad_med=2.40)

    # 4. Collapsed narrow mode (sigma_0 -> 0.002) -> FAIL core_min_eigen_ratio / covariance
    collapsed_sigmas = torch.tensor([0.002, 0.08, 0.16, 0.32])
    impostor_col = sample_mixture(torch.tensor(SCALE_MASS_WEIGHTS), means_sm, collapsed_sigmas, 512, rng, device="cpu")
    c_sm_col = score_t3_isolation_kind("t3_iso_scale_mass_8x", impostor_col, ref_sm, critic_grad_med=0.35)

    # 5. Saddle-shift oracle (4 post-shift modes + 2-mode frozen snapshot) -> PASS
    ref_sad = sample_target_for_kind("t3_iso_horizon_saddle_shift", 2048, rng, step=1000, device="cpu")
    oracle_pre = sample_target_for_kind("t3_iso_horizon_saddle_shift", 128, rng, step=300, device="cpu")
    oracle_post = sample_target_for_kind("t3_iso_horizon_saddle_shift", 128, rng, step=1000, device="cpu")
    c_sad_pass = score_t3_isolation_kind(
        "t3_iso_horizon_saddle_shift",
        oracle_post,
        ref_sad,
        critic_grad_med=0.30,
        pre_shift_hq=0.99,
        frozen_particles=oracle_pre,
    )

    # 6. Saddle-trapped 2-of-4 post-shift cloud -> FAIL modes == 4 & mass_tv <= 0.12
    c_sad_trapped = score_t3_isolation_kind(
        "t3_iso_horizon_saddle_shift",
        oracle_pre,
        ref_sad,
        critic_grad_med=0.30,
        pre_shift_hq=0.99,
        frozen_particles=oracle_pre,
    )

    # 7. Nonfinite particle -> fail-closed FAIL
    nan_pts = oracle_post.clone()
    nan_pts[0, 0] = float("nan")
    c_sad_nan = score_t3_isolation_kind(
        "t3_iso_horizon_saddle_shift",
        nan_pts,
        ref_sad,
        critic_grad_med=0.30,
        pre_shift_hq=0.99,
        frozen_particles=oracle_pre,
    )

    return {
        "scale_mass_oracle_pass": {**c_sm_pass, "expected_pass": True},
        "scale_mass_uniform_impostor_fail": {**c_sm_unif, "expected_pass": False},
        "scale_mass_sharp_critic_fail": {**c_sm_sharp, "expected_pass": False},
        "scale_mass_collapsed_narrow_fail": {**c_sm_col, "expected_pass": False},
        "saddle_shift_oracle_pass": {**c_sad_pass, "expected_pass": True},
        "saddle_shift_trapped_2mode_fail": {**c_sad_trapped, "expected_pass": False},
        "saddle_shift_nan_fail": {**c_sad_nan, "expected_pass": False},
    }


def _bind_formulation_record(recipe: Recipe, opt_d: Any, loss_fn: Any, prior_reg_fn: Any = None) -> None:
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
    elif eff == "unified_v11":
        from particlegan.unified_v11 import bind_unified_v11_record
        bind_unified_v11_record(record, loss_fn, prior_reg_fn)
    elif eff == "axiom":
        from particlegan.axiom import bind_axiom_record
        bind_axiom_record(record, loss_fn, prior_reg_fn)
    elif eff == "minimal":
        from particlegan.minimal import bind_minimal_record
        bind_minimal_record(record, loss_fn, prior_reg_fn)
    elif eff == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0:
        from particlegan.rhbi import bind_rhbi_record
        bind_rhbi_record(record, loss_fn, prior_reg_fn)
    elif eff == "t1_coupling" or recipe.t1c_weight > 0:
        from particlegan.t1_coupling import bind_t1_coupling_record
        bind_t1_coupling_record(record, loss_fn, prior_reg_fn)
    elif eff == "rgdt":
        from particlegan.t3_dynamics import bind_rgdt_record
        bind_rgdt_record(record, loss_fn, prior_reg_fn)


class T3IsolationFixture:
    """Checkpointable public-API direct-particle fixture with G = nn.Identity()."""

    def __init__(
        self,
        kind: str,
        recipe: Recipe,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
        discriminator: nn.Module | None = None,
        prior: ParticlePrior | None = None,
        streams: dict[str, torch.Generator] | None = None,
        max_steps: int | None = None,
    ) -> None:
        if kind not in HOST_DEFINITIONS:
            raise ValueError(f"unknown Theory-3 isolation kind: {kind!r}")
        if recipe.prior_update != "learned":
            raise ValueError("Theory-3 isolation requires Recipe.prior_update == 'learned'")
        self.kind = kind
        self.spec = deepcopy(HOST_DEFINITIONS[kind])
        self.recipe = recipe
        self.device = torch.device(device)
        self.max_steps = int(self.spec["steps"] if max_steps is None else max_steps)

        g_mod, d_mod = build_models_for_spec(self.spec)
        self.G = g_mod.to(self.device)
        if discriminator is None:
            init.deterministic_orthogonal_(d_mod, seed=int(seed) + 11)
        self.D = (d_mod if discriminator is None else discriminator).to(self.device)

        res = self.spec["resources"]
        if prior is None:
            p_mod = ParticlePrior(
                num_particles=int(res["num_particles"]),
                z_dim=int(res["z_dim"]),
                learnable=True,
            )
            init.deterministic_orthogonal_(p_mod, seed=int(seed) + 23)
        else:
            p_mod = prior
        if type(p_mod) is not ParticlePrior:
            raise TypeError("Theory-3 isolation requires a zero-sigma ParticlePrior")
        self.prior = p_mod.to(self.device)
        self.recipe.apply_prior_policy(self.prior)

        no_ema_forms = (
            "bispf",
            "empf",
            "rhbi",
            "t1_coupling",
            "t2_secant",
            "rgdt",
            "axiom",
            "unified_v11",
            "minimal",
        )
        ema_critic = None if recipe.effective_critic_formulation in no_ema_forms else deepcopy(self.D)
        self.opt_g, self.opt_d = recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            ema_critic=ema_critic,
        )
        self.loss = recipe.make_loss()
        self.penalty = recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        _bind_formulation_record(recipe, self.opt_d, self.loss, None)

        stream_map = dict(streams or {})
        latent_gen = stream_map.get("latent_generator") or torch.Generator(device=self.device).manual_seed(int(seed) + 101)
        penalty_gen = stream_map.get("penalty_generator") or torch.Generator(device=self.device).manual_seed(int(seed) + 102)
        noise_gen = stream_map.get("noise_generator") or torch.Generator(device=self.device).manual_seed(int(seed) + 103)
        input_noise_gen = stream_map.get("input_noise_generator") or torch.Generator(device=self.device).manual_seed(int(seed) + 104)
        eval_gen = stream_map.get("eval_generator") or torch.Generator(device=self.device).manual_seed(int(seed) + 105)
        self.data_d_rng = stream_map.get("data_d_generator") or torch.Generator(device="cpu").manual_seed(int(seed) + 201)
        self.data_g_rng = stream_map.get("data_g_generator") or torch.Generator(device="cpu").manual_seed(int(seed) + 202)
        self.input_noise_generator = input_noise_gen

        self.policy = UpdatePolicy(
            recipe,
            self.G,
            self.D,
            prior=self.prior,
            generator_optimizer=self.opt_g,
            critic_optimizer=self.opt_d,
            penalty=self.penalty,
            seed=int(seed),
            streams={
                "latent_generator": latent_gen,
                "penalty_generator": penalty_gen,
                "noise_generator": noise_gen,
                "eval_generator": eval_gen,
            },
            schedule=lambda s, cfg: learning_rate_scales(s, cfg),
        )
        self.critic = InputNoise(
            self.D,
            0.0,
            self.input_noise_generator,
            bf16=(recipe.training_precision == "bf16_channels_last"),
        )
        self.pre_shift_hq: float | None = None
        self.frozen_snapshot: torch.Tensor | None = None
        self.last_update: dict[str, float] = {}

    @property
    def completed_steps(self) -> int:
        return int(self.policy.completed_steps)

    def _bind_discrete_step(self, active: bool) -> None:
        for obj in (
            self.loss,
            getattr(self.loss, "rhbi", None),
            getattr(self.loss, "t1c", None),
        ):
            if obj is not None and hasattr(obj, "bind_discrete_prior_step"):
                obj.bind_discrete_prior_step(active)

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        next_step = self.completed_steps + 1
        b_sz = int(self.recipe.batch_size)
        real_d_cpu = sample_target_for_kind(self.kind, b_sz, self.data_d_rng, step=next_step, device="cpu")
        real_g_cpu = sample_target_for_kind(self.kind, b_sz, self.data_g_rng, step=next_step, device="cpu")
        real_d = real_d_cpu.to(self.device)
        real_g = real_g_cpu.to(self.device)

        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            game_rec = getattr(getattr(self.penalty, "regularizer", None), "record", None)
            noise = self.policy.begin_step(real_d, game_record=game_rec, execution_limit=self.max_steps)
            self.critic.std = noise.input_sigma

            self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                z_d, rows_d = self.prior.sample(b_sz, generator=self.policy.latent_generator)
                fake_d = self.policy.generate(
                    z_d,
                    rows=rows_d.flatten(),
                    sigma=noise.output_sigma,
                    stream=self.policy.noise_generator,
                )
            self.policy.observe_critic_pair(real_d, fake_d)
            adv_d = self.loss.d_loss(self.critic(real_d), self.critic(fake_d))
            pen_val = self.penalty(self.critic, real_d, fake_d)
            loss_d = adv_d + pen_val
            self.policy.before_critic_backward()
            loss_d.backward()
            self.policy.after_critic_backward()
            self.opt_d.step()
            self.policy.after_critic_step()

            for p in self.D.parameters():
                p.requires_grad_(False)
            self.opt_g.zero_grad(set_to_none=True)
            z_g, rows_g = self.prior.sample(b_sz, generator=self.policy.latent_generator)
            flat_rows = rows_g.flatten()
            self.policy.observe_sampled_rows(flat_rows)
            fake_g = self.policy.generate(
                z_g,
                rows=flat_rows,
                sigma=noise.output_sigma,
                stream=self.policy.noise_generator,
            )
            real_logits_g = self.critic(real_g)
            self._bind_discrete_step(True)
            loss_gan = self.loss.g_loss(self.critic(fake_g), real_logits_g)

            protected_evaluator = None
            if self.recipe.constraint_geometry_mode == "strict_progress":
                locations = self.prior.z.detach().clone()
                sampled_latent = z_g.detach().clone()

                def protected_evaluator() -> tuple[torch.Tensor]:
                    rec = getattr(self.opt_d, "record", None)
                    prev_probe = getattr(rec, "_in_protected_probe", False) if rec is not None else False
                    if rec is not None:
                        rec._in_protected_probe = True
                    try:
                        replay = sampled_latent + (self.prior.z[flat_rows] - locations[flat_rows])
                        gen_r = self.policy.generate(replay, rows=flat_rows, sigma=0.0, stream=None)
                        self._bind_discrete_step(True)
                        return (self.loss.g_loss(self.critic(gen_r), real_logits_g),)
                    finally:
                        self._bind_discrete_step(False)
                        if rec is not None:
                            rec._in_protected_probe = prev_probe

            self.policy.before_generator_backward()
            constraint_geometry_backward(
                loss_gan,
                self.opt_g,
                (loss_gan,),
                protected_evaluator=protected_evaluator,
            )
            self.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=adv_d.detach())
            self.opt_g.step()
            self.policy.after_generator_step()
            self.policy.finish_step()

            if self.kind == "t3_iso_horizon_saddle_shift" and self.completed_steps == SADDLE_SHIFT_STEP:
                with torch.no_grad():
                    snap = self.prior.z.detach().to(device="cpu", dtype=torch.float32).clone()
                    w_pre, m_pre, s_pre = saddle_shift_parameters(SADDLE_SHIFT_STEP, "cpu")
                    pre_m = _mixture_mode_metrics(snap, w_pre, m_pre, s_pre, min_mode_share=0.20)
                    self.pre_shift_hq = float(pre_m["hq"])
                    self.frozen_snapshot = snap

            self.last_update = {
                "loss_d": float(loss_d.detach().item()),
                "loss_gan": float(loss_gan.detach().item()),
                "penalty": float(pen_val.detach().item()),
            }
            return real_d_cpu, real_g_cpu
        except BaseException:
            self.policy.abort_step()
            raise
        finally:
            self._bind_discrete_step(False)
            for p, flag in zip(self.D.parameters(), flags):
                p.requires_grad_(flag)

    def observe(self, reference: torch.Tensor) -> dict[str, Any]:
        pts = self.prior.z.detach()
        grad_med = measure_critic_grad_median(self.D, pts, reference)
        return score_t3_isolation_kind(
            self.kind,
            pts,
            reference,
            critic_grad_med=grad_med,
            pre_shift_hq=self.pre_shift_hq,
            frozen_particles=self.frozen_snapshot,
        )

    def state_dict(self) -> dict[str, Any]:
        loss_fallback = getattr(self.loss, "_fallback_ell", None)
        return {
            "schema_version": 1,
            "kind": self.kind,
            "recipe": self.recipe.to_dict(),
            "max_steps": self.max_steps,
            "policy": self.policy.state_dict(),
            "data_d_rng": self.data_d_rng.get_state(),
            "data_g_rng": self.data_g_rng.get_state(),
            "input_noise_rng": self.input_noise_generator.get_state(),
            "pre_shift_hq": self.pre_shift_hq,
            "frozen_snapshot": None if self.frozen_snapshot is None else self.frozen_snapshot.clone(),
            "loss_fallback_ell": None if loss_fallback is None else loss_fallback.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected_keys = {
            "schema_version",
            "kind",
            "recipe",
            "max_steps",
            "policy",
            "data_d_rng",
            "data_g_rng",
            "input_noise_rng",
            "pre_shift_hq",
            "frozen_snapshot",
            "loss_fallback_ell",
        }
        if not isinstance(state, dict) or set(state.keys()) != expected_keys:
            raise ValueError("invalid T3IsolationFixture checkpoint keys")
        if state["schema_version"] != 1 or state["kind"] != self.kind:
            raise ValueError("incompatible T3IsolationFixture checkpoint schema or kind")
        if state["recipe"] != self.recipe.to_dict() or state["max_steps"] != self.max_steps:
            raise ValueError("T3IsolationFixture checkpoint recipe or max_steps mismatch")
        self.policy.load_state_dict(state["policy"])
        self.data_d_rng.set_state(state["data_d_rng"])
        self.data_g_rng.set_state(state["data_g_rng"])
        self.input_noise_generator.set_state(state["input_noise_rng"])
        self.pre_shift_hq = state["pre_shift_hq"]
        self.frozen_snapshot = None if state["frozen_snapshot"] is None else state["frozen_snapshot"].clone()
        if state["loss_fallback_ell"] is not None and hasattr(self.loss, "_fallback_ell"):
            self.loss._fallback_ell = state["loss_fallback_ell"].clone()
