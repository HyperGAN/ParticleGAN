"""Variational mutual information InfoGAN on structured particle priors (Chen et al., NeurIPS 2016).

Augments the ParticleGAN objective with a variational lower bound L_I(G, Q) on the
mutual information I(c; G(z, c)) between structured particle codes c = (c_disc, c_cont)
and generated samples on an 8-mode rotated anisotropic 2D Gaussian ring.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    MoGParticlePrior,
    Recipe,
    bind_formulation_records,
    critic_g_step_mode,
    get_recipe,
    init,
    scale_learning_rates,
    unwrap_prior,
)

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
    "unified_v15",
    "r4_autograd_critic",
    "cond_fiber",
    "axiom",
    "minimal",
})
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from .reproducibility import construction_rng


ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = ROOT / "configs/forge/tasks/infogan_anisotropic_modes.json"
CASE_ID = "api-infogan-anisotropic-modes"
VERSION = "api-infogan-v1"

NUM_MODES = 8
RING_RADIUS = 3.0
SIGMA_PARALLEL = 0.25
SIGMA_PERP = 0.05
MAHALANOBIS_THRESHOLD = 3.5
MIN_MODE_SAMPLES = 50

DEFAULT_STEPS = 3500
DEFAULT_BATCH_SIZE = 256
DEFAULT_PARTICLES = 256
DEFAULT_Z_DIM = 2
DEFAULT_RECIPE = "bispf"


def anisotropic_ring_geometry(device="cpu", dtype=torch.float32, radius: float = RING_RADIUS):
    """Return (centers, tangents, normals) of shape (8, 2) for the 8 rotated anisotropic modes."""
    angles = torch.arange(NUM_MODES, device=device, dtype=dtype) * (2.0 * math.pi / NUM_MODES)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    normals = torch.stack([cos_a, sin_a], dim=1)
    tangents = torch.stack([-sin_a, cos_a], dim=1)
    centers = float(radius) * normals
    return centers, tangents, normals


def sample_infogan_target(
    n: int,
    generator: torch.Generator | None = None,
    *,
    device="cpu",
    radius: float = RING_RADIUS,
    sigma_parallel: float = SIGMA_PARALLEL,
    sigma_perp: float = SIGMA_PERP,
    return_codes: bool = False,
):
    """Sample from the 8-mode rotated anisotropic 2D Gaussian ring target law."""
    dev = torch.device(device)
    centers, tangents, normals = anisotropic_ring_geometry(device=dev, radius=radius)
    mode_ids = torch.randint(0, NUM_MODES, (n,), generator=generator, device=dev)
    u_parallel = torch.randn(n, 1, generator=generator, device=dev)
    u_perp = torch.randn(n, 1, generator=generator, device=dev)
    samples = (
        centers[mode_ids]
        + (float(sigma_parallel) * u_parallel) * tangents[mode_ids]
        + (float(sigma_perp) * u_perp) * normals[mode_ids]
    )
    if return_codes:
        return samples, mode_ids, u_parallel.squeeze(-1)
    return samples


@torch.no_grad()
def score_infogan(
    samples: torch.Tensor,
    c_disc: torch.Tensor,
    c_cont: torch.Tensor,
    q_logits: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    *,
    radius: float = RING_RADIUS,
    sigma_parallel: float = SIGMA_PARALLEL,
    sigma_perp: float = SIGMA_PERP,
):
    """Evaluate InfoGAN samples against the 8-mode rotated anisotropic Gaussian ring gates."""
    samples = torch.as_tensor(samples).detach().float()
    dev = samples.device
    c_disc_t = torch.as_tensor(c_disc, device=dev).detach()
    if c_disc_t.ndim == 2:
        c_idx = c_disc_t.argmax(dim=1).long()
    else:
        c_idx = c_disc_t.reshape(-1).long()
    c_cont_flat = torch.as_tensor(c_cont, device=dev).detach().float().reshape(-1)

    if target is None:
        ref_rng = torch.Generator(device=dev).manual_seed(78013)
        target = sample_infogan_target(
            len(samples),
            generator=ref_rng,
            device=dev,
            radius=radius,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
        )
    else:
        target = torch.as_tensor(target, device=dev).detach().float()

    q_finite = True if q_logits is None else bool(torch.all(torch.isfinite(torch.as_tensor(q_logits))))
    if (
        samples.ndim != 2
        or samples.shape[1] != 2
        or len(samples) == 0
        or not bool(torch.all(torch.isfinite(samples)))
        or not bool(torch.all(torch.isfinite(c_cont_flat)))
        or not q_finite
    ):
        metrics = {
            "modes": 0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "discrete_nmi": 0.0,
            "q_acc": 0.0,
            "continuous_r2": 0.0,
        }
        return {
            "passed": False,
            "failed_bounds": [
                "modes",
                "hq",
                "mass_tv",
                "discrete_nmi",
                "q_acc",
                "continuous_r2",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": [],
        }

    centers, tangents, normals = anisotropic_ring_geometry(device=dev, radius=radius)
    diff = samples.unsqueeze(1) - centers.unsqueeze(0)  # (N, 8, 2)
    proj_parallel = (diff * tangents.unsqueeze(0)).sum(dim=-1)  # (N, 8)
    proj_perp = (diff * normals.unsqueeze(0)).sum(dim=-1)  # (N, 8)
    mahal_sq = (proj_parallel / float(sigma_parallel)) ** 2 + (proj_perp / float(sigma_perp)) ** 2
    mahal_dist = torch.sqrt(mahal_sq.clamp_min(0.0))

    min_mahal, nearest_k = mahal_dist.min(dim=1)
    is_hq = min_mahal <= MAHALANOBIS_THRESHOLD
    hq = float(is_hq.float().mean().item())

    hq_counts = torch.bincount(nearest_k[is_hq], minlength=NUM_MODES)
    modes = int((hq_counts >= MIN_MODE_SAMPLES).sum().item())

    all_counts = torch.bincount(nearest_k, minlength=NUM_MODES).float()
    empirical_freq = all_counts / max(len(samples), 1)
    mass_tv = float(0.5 * (empirical_freq - (1.0 / NUM_MODES)).abs().sum().item())

    # Normalized Mutual Information: 2 * I(c_disc; k_nearest) / (H(c_disc) + H(k_nearest))
    c_clamped = c_idx.clamp(0, NUM_MODES - 1)
    joint_idx = c_clamped * NUM_MODES + nearest_k
    joint_counts = torch.bincount(joint_idx, minlength=NUM_MODES * NUM_MODES).float().reshape(NUM_MODES, NUM_MODES)
    p_joint = joint_counts / max(len(samples), 1)
    p_c = p_joint.sum(dim=1)
    p_k = p_joint.sum(dim=0)

    h_c = -(p_c * torch.log(p_c.clamp_min(1e-12))).sum()
    h_k = -(p_k * torch.log(p_k.clamp_min(1e-12))).sum()
    h_ck = -(p_joint * torch.log(p_joint.clamp_min(1e-12))).sum()
    mi = (h_c + h_k - h_ck).clamp_min(0.0)
    denom_h = h_c + h_k
    discrete_nmi = (
        float((2.0 * mi / denom_h.clamp_min(1e-12)).clamp(0.0, 1.0).item())
        if float(denom_h.item()) > 1e-9
        else 0.0
    )

    # Recognition network classification accuracy q_acc
    if q_logits is not None:
        q_t = torch.as_tensor(q_logits, device=dev).detach()
        if q_t.ndim == 2:
            q_pred = q_t[:, :NUM_MODES].argmax(dim=1)
        else:
            q_pred = q_t.reshape(-1).long()
        q_acc = float((q_pred == c_clamped).float().mean().item())
    else:
        # Optimal cluster-to-code assignment accuracy when Q logits are not separately supplied
        q_acc = float((joint_counts.max(dim=0).values.sum() / max(len(samples), 1)).item())

    # Mean squared correlation R^2 = Corr(c_cont, (x - mu_k) . t_k)^2 across modes
    row_indices = torch.arange(len(samples), device=dev)
    tangential_coord = proj_parallel[row_indices, nearest_k]
    r2_per_mode = []
    for k in range(NUM_MODES):
        mask_k = nearest_k == k
        if int(mask_k.sum().item()) < 4:
            r2_per_mode.append(0.0)
            continue
        u = c_cont_flat[mask_k]
        v = tangential_coord[mask_k]
        u_c = u - u.mean()
        v_c = v - v.mean()
        denom_uv = float((u_c.square().sum() * v_c.square().sum()).item())
        if denom_uv <= 1e-12:
            r2_per_mode.append(0.0)
        else:
            corr_sq = float((u_c * v_c).sum().square().item()) / denom_uv
            r2_per_mode.append(min(1.0, max(0.0, corr_sq)))
    continuous_r2 = float(sum(r2_per_mode) / NUM_MODES)

    metrics = {
        "modes": modes,
        "hq": hq,
        "mass_tv": mass_tv,
        "discrete_nmi": discrete_nmi,
        "q_acc": q_acc,
        "continuous_r2": continuous_r2,
    }

    failed_bounds = []
    if metrics["modes"] < 8:
        failed_bounds.append("modes")
    if metrics["hq"] < 0.85:
        failed_bounds.append("hq")
    if metrics["mass_tv"] > 0.12:
        failed_bounds.append("mass_tv")
    if metrics["discrete_nmi"] < 0.85 or metrics["q_acc"] < 0.90:
        failed_bounds.append("discrete_nmi")
    if metrics["q_acc"] < 0.90:
        failed_bounds.append("q_acc")
    if metrics["continuous_r2"] < 0.65:
        failed_bounds.append("continuous_r2")

    passed = len(failed_bounds) == 0
    ref_line = torch.stack([c_cont_flat, float(sigma_parallel) * c_cont_flat], dim=1)
    obs_line = torch.stack([c_cont_flat, tangential_coord], dim=1)
    views = [
        dict(
            kind="scatter",
            title="Anisotropic ring generation vs target",
            target=target.detach().cpu(),
            samples=samples.detach().cpu(),
        ),
        dict(
            kind="scatter",
            title="Continuous code c_cont vs tangential projection",
            target=ref_line.detach().cpu(),
            samples=obs_line.detach().cpu(),
            xlabel="c_cont",
            ylabel="tangential projection",
        ),
        dict(
            kind="bar",
            title="Anisotropic mode mass distribution",
            target=torch.full((NUM_MODES,), 1.0 / NUM_MODES),
            samples=empirical_freq.detach().cpu(),
        ),
    ]
    return dict(
        metrics=metrics,
        passed=passed,
        failed_bounds=sorted(set(failed_bounds)),
        views=views,
    )


class InfoGANGenerator(nn.Module):
    """Generator G([z, c_disc, c_cont]): R^{2 + 8 + 1 = 11} -> R^2."""

    def __init__(self, z_dim: int = 2, n_disc: int = 8, n_cont: int = 1, out_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_disc = int(n_disc)
        self.n_cont = int(n_cont)
        self.in_dim = self.z_dim + self.n_disc + self.n_cont
        self.out_dim = int(out_dim)
        self.disc_net = nn.Sequential(
            nn.Linear(self.n_disc, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, self.out_dim, bias=False),
        )
        self.cont_net = nn.Sequential(
            nn.Linear(self.n_disc * (self.n_cont + 1), hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, self.out_dim, bias=False),
        )
        self.z_net = nn.Sequential(
            nn.Linear(self.n_disc * self.z_dim, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, self.out_dim, bias=False),
        )

    def forward(self, z_or_concat: torch.Tensor, c_disc: torch.Tensor | None = None, c_cont: torch.Tensor | None = None):
        if c_disc is not None and c_cont is not None:
            if c_disc.ndim == 1:
                c_disc = F.one_hot(c_disc.long(), num_classes=self.n_disc).to(z_or_concat.dtype)
            if c_cont.ndim == 1:
                c_cont = c_cont.unsqueeze(1)
            inp = torch.cat([z_or_concat, c_disc, c_cont], dim=1)
        else:
            inp = z_or_concat

        z = inp[:, : self.z_dim]
        c_d = inp[:, self.z_dim : self.z_dim + self.n_disc]
        c_c = inp[:, self.z_dim + self.n_disc : self.z_dim + self.n_disc + self.n_cont]

        c_cent = (c_d - (1.0 / self.n_disc)) * math.sqrt(self.n_disc)
        z_loc = torch.tanh(z)
        h_mode = self.disc_net(c_cent)
        cont_feat = torch.cat([c_d * c_c, c_cent], dim=1)
        h_tan = 0.5 * self.cont_net(cont_feat) * c_c
        z_feat = torch.cat([c_d * z_loc[:, :1], c_d * z_loc[:, 1:2]], dim=1)
        h_rad = 0.1 * self.z_net(z_feat)
        return h_mode + h_tan + h_rad


class InfoGANDiscriminator(nn.Module):
    """Discriminator D: R^2 -> R."""

    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class InfoGANRecognition(nn.Module):
    """Recognition network Q: R^2 -> R^{8 + 1} predicting 8-class logits and scalar mean c_cont."""

    def __init__(self, in_dim: int = 2, n_disc: int = 8, n_cont: int = 1, hidden: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_disc = int(n_disc)
        self.n_cont = int(n_cont)
        self.out_dim = self.n_disc + self.n_cont
        feat_dim = self.in_dim * 3
        self.disc_trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_disc),
        )
        self.cont_trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_disc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt((x * x).sum(dim=1, keepdim=True) + 1e-6)
        u = x / r
        feat = torch.cat([x, u, x / (1.0 + r)], dim=1)
        q_disc = self.disc_trunk(feat)
        probs = F.softmax(q_disc, dim=1)
        q_cont_all = self.cont_trunk(feat)
        q_cont = (probs * q_cont_all).sum(dim=1, keepdim=True)
        return torch.cat([q_disc, q_cont], dim=1)


def _infogan_case():
    task = json.loads(TASK_PATH.read_text())
    return {
        "id": CASE_ID,
        "legacy_ids": ["develop-infogan-anisotropic-modes"],
        "title": "Eight rotated anisotropic Gaussian modes: variational mutual information InfoGAN",
        "goal": task["description"],
        "kind": "vector",
        "default_steps": task["execution"]["steps"],
        "batch_size": task["execution"]["host_definition"]["batch"],
        "eval_samples": task["evaluation"]["eval_samples"],
        "default_recipe": DEFAULT_RECIPE,
        "terminal_observations": task["evaluation"].get("minimum_stable_checks", 5),
        "evaluation_observations": task["evaluation"].get("observations", 24),
        "protocol_seed": 0,
        "thresholds": {
            "modes_min": 8,
            "hq_min": 0.85,
            "mass_tv_max": 0.12,
            "discrete_nmi_min": 0.85,
            "q_acc_min": 0.90,
            "continuous_r2_min": 0.65,
        },
        "sampling": (
            "8-mode rotated anisotropic Gaussian ring (R=3.0, sigma_parallel=0.25, sigma_perp=0.05); "
            "structured particle prior c_disc = one_hot(idx mod 8, 8) and c_cont ~ N(0, 1); "
            "recognition network Q(c | x)"
        ),
        "scope": "Tier 4 variational mutual information InfoGAN (G([z, c_disc, c_cont]), D(x), Q(c|x)) on 8 rotated anisotropic modes.",
    }


def list_cases():
    return [_infogan_case()]


class InfoGANAnisotropicFixture:
    """Public-API InfoGAN fixture supporting all 8 ParticleGAN recipe families."""

    api_components = (
        "particlegan.Recipe",
        "particlegan.Recipe.make_optimizers",
        "particlegan.Recipe.make_critic_penalty",
        "particlegan.Recipe.prior_regularization",
        "particlegan.MoGParticlePrior",
        "particlegan.init",
    )

    def __init__(
        self,
        case=None,
        *,
        recipe_name: str | None = DEFAULT_RECIPE,
        recipe: Recipe | None = None,
        prior: nn.Module | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
        max_steps: int | None = None,
    ):
        if isinstance(case, str) and case != CASE_ID:
            recipe_name = case
            case = None
        if case is None:
            case = _infogan_case()
        self.metadata = deepcopy(case)
        self.case = self.metadata
        self.case_id = case["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.completed_steps = 0
        self.limit = case["default_steps"] if max_steps is None else int(max_steps)
        if type(self.limit) is not int or not 0 < self.limit <= case["default_steps"]:
            raise ValueError("max_steps must be in the declared budget")

        self.batch_size = int(case.get("batch_size", DEFAULT_BATCH_SIZE))
        self.num_particles = DEFAULT_PARTICLES
        self.z_dim = DEFAULT_Z_DIM
        self.info_weight = 1.0

        self.centers, self.tangents, self.normals = anisotropic_ring_geometry(device=self.device)

        with construction_rng(self.seed, self.device):
            self.G = InfoGANGenerator(
                z_dim=self.z_dim, n_disc=NUM_MODES, n_cont=1, out_dim=2, hidden=128
            ).to(self.device)
            self.D = InfoGANDiscriminator(in_dim=2, hidden=128).to(self.device)
            self.Q = InfoGANRecognition(
                in_dim=2, n_disc=NUM_MODES, n_cont=1, hidden=128
            ).to(self.device)
            self.E = self.Q
            if prior is None:
                self.prior = MoGParticlePrior(
                    num_particles=self.num_particles,
                    z_dim=self.z_dim,
                    sigma=0.05,
                    standardize=False,
                ).to(self.device)
            else:
                self.prior = prior.to(self.device)

            init.deterministic_orthogonal_(self.G, seed=self.seed + 1)
            init.deterministic_orthogonal_(self.D, seed=self.seed + 2)
            init.deterministic_orthogonal_(self.Q, seed=self.seed + 3)
            if prior is None:
                init.deterministic_orthogonal_(self.prior, seed=self.seed + 4)

        self.data_rng = torch.Generator(device=self.device).manual_seed(self.seed + 101)
        self.latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 102)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed + 103)
        self.code_rng = torch.Generator(device=self.device).manual_seed(self.seed + 104)
        self.eval_rng = torch.Generator(device=self.device).manual_seed(self.seed + 105)
        self.batch_sequence_sha256 = hashlib.sha256(b"api-infogan-anisotropic-batches-v1").hexdigest()
        self.last_losses = {}

        target_recipe = recipe if recipe is not None else (recipe_name or DEFAULT_RECIPE)
        self.rebind_trainer(target_recipe, prior=self.prior)

    def _resolve_recipe(self, recipe_or_name: Recipe | str | None) -> Recipe:
        if isinstance(recipe_or_name, Recipe):
            base = recipe_or_name
        else:
            name = recipe_or_name or DEFAULT_RECIPE
            base = get_recipe(name)
        horizon = None if base.total_steps is None else self.limit
        return base.replace(
            batch_size=self.batch_size,
            num_particles=self.num_particles,
            z_dim=self.z_dim,
            total_steps=horizon,
        )

    def rebind_trainer(self, recipe: Recipe | str, prior: nn.Module | None = None):
        """Bind or rebind recipe and prior across all 8 ParticleGAN trainer families."""
        if prior is not None:
            self.prior = prior.to(self.device)
        self.recipe = self._resolve_recipe(recipe)

        base_prior = unwrap_prior(self.prior)
        if self.recipe.prior_update == "learned" and hasattr(base_prior, "z") and not base_prior.z.requires_grad:
            rows = base_prior.z.detach().clone()
            del base_prior.z
            base_prior.z = nn.Parameter(rows, requires_grad=True)

        ema_critic = (
            None
            if self.recipe.effective_critic_formulation in _NO_EMA_FORMULATIONS
            else deepcopy(self.D)
        )
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            encoder=self.Q,
            ema_critic=ema_critic,
            foreach=False,
            fused=False,
        )
        for opt in (self.opt_g, self.opt_d):
            for group in opt.param_groups:
                if "betas" in group:
                    group.setdefault("_recipe_initial_betas", tuple(group["betas"]))

        self.base_rates = [[group["lr"] for group in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        bind_formulation_records(self.recipe, self.opt_d, self.loss, self.regularizer, generator=self.G)
        return self

    def sample_target(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_rng if generator is None else generator
        return sample_infogan_target(n, generator=rng, device=self.device)

    def _sample_latent_and_codes(
        self,
        n: int,
        *,
        latent_generator: torch.Generator | None = None,
        noise_generator: torch.Generator | None = None,
        code_generator: torch.Generator | None = None,
    ):
        lat_gen = self.latent_rng if latent_generator is None else latent_generator
        noi_gen = self.noise_rng if noise_generator is None else noise_generator
        cod_gen = self.code_rng if code_generator is None else code_generator

        if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
            z, idx = self.prior.sample(n, generator=lat_gen, noise_generator=noi_gen)
        else:
            z, idx = self.prior.sample(n, generator=lat_gen)

        c_idx = (idx % NUM_MODES).long()
        c_disc = F.one_hot(c_idx, num_classes=NUM_MODES).to(dtype=z.dtype)
        c_cont = torch.randn(n, 1, generator=cod_gen, device=self.device, dtype=z.dtype)
        return z, idx, c_idx, c_disc, c_cont

    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")

        if (
            getattr(self.recipe, "lr_control", "mobility") != "stationarity"
            and self.recipe.continuous_policy is None
            and (self.recipe.lr_schedule != "cosine" or self.recipe.total_steps is not None)
        ):
            scale_learning_rates(
                self.completed_steps,
                self.recipe,
                [self.opt_g, self.opt_d],
                self.base_rates,
                self.prior,
            )

        batch_size = self.recipe.batch_size
        real_x = self.sample_target(batch_size)
        self.batch_sequence_sha256 = hashlib.sha256(
            bytes.fromhex(self.batch_sequence_sha256) + real_x.detach().cpu().numpy().tobytes()
        ).hexdigest()

        self.G.train()
        self.D.train()
        self.Q.train()
        self.prior.train()

        # 1. Critic step
        with torch.no_grad():
            z_d, idx_d, _, c_disc_d, c_cont_d = self._sample_latent_and_codes(batch_size)
            x_fake_d = self.G(torch.cat([z_d, c_disc_d, c_cont_d], dim=1))
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                x_fake_d = self.prior.postprocess_generated(x_fake_d)

        self.opt_d.zero_grad(set_to_none=True)
        loss_d = self.loss.d_loss(self.D(real_x), self.D(x_fake_d))
        reg_d = self.penalty(self.D, real_x, x_fake_d)
        (loss_d + reg_d).backward()
        self.opt_d.step()

        # 2. Generator + Recognition Q + Prior step
        with critic_g_step_mode(self.D):
            z_g, idx_g, c_idx_g, c_disc_g, c_cont_g = self._sample_latent_and_codes(batch_size)
            x_fake = self.G(torch.cat([z_g, c_disc_g, c_cont_g], dim=1))
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                x_fake = self.prior.postprocess_generated(x_fake)

            fake_logits = self.D(x_fake)
            real_logits = self.D(real_x)
            loss_adv = self.loss.g_loss(fake_logits, real_logits)

            q_out = self.Q(x_fake)
            q_disc = q_out[:, :NUM_MODES]
            q_cont = q_out[:, NUM_MODES : NUM_MODES + 1]
            loss_ce = F.cross_entropy(q_disc, c_idx_g)
            loss_cont = F.mse_loss(q_cont, c_cont_g)
            loss_info = loss_ce + 0.5 * loss_cont

            base_prior = unwrap_prior(self.prior)
            loss_prior = self.recipe.prior_regularization(
                base_prior.z, regularizer=self.regularizer, prior=self.prior
            )
            if isinstance(self.prior, ManipulatedPrior) and self.prior.has_regularizer:
                loss_prior = loss_prior + self.prior.regularize(
                    base_prior.z,
                    sampled_codes=z_g,
                    indices=idx_g,
                    generated=x_fake,
                    logits=fake_logits,
                )

            loss_g = loss_adv + self.info_weight * loss_info + loss_prior

            self.opt_g.zero_grad(set_to_none=True)
            if hasattr(self.opt_g, "set_sampled_rows") and base_prior.z.requires_grad:
                self.opt_g.set_sampled_rows(base_prior.z, idx_g)
            if self.recipe.constraint_geometry_mode != "none":
                constraint_geometry_backward(loss_g, self.opt_g, (loss_adv,))
            else:
                loss_g.backward()
            self.opt_g.step()

        self.completed_steps += 1
        self.last_losses = {
            "loss_d": float(loss_d.detach().item()),
            "loss_g": float(loss_g.detach().item()),
            "loss_adv": float(loss_adv.detach().item()),
            "loss_info": float(loss_info.detach().item()),
            "loss_prior": float(loss_prior.detach().item()),
            "reg_d": float(reg_d.detach().item()),
        }
        if not all(math.isfinite(v) for v in self.last_losses.values()):
            raise FloatingPointError("nonfinite InfoGAN update")
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def observe(self, n: int = 4096, seed: int = 10000):
        modes = [
            (m, m.training)
            for model in (self.G, self.D, self.Q, self.prior)
            for m in model.modules()
        ]
        try:
            self.G.eval()
            self.D.eval()
            self.Q.eval()
            self.prior.eval()

            lat_gen = torch.Generator(device=self.device).manual_seed(int(seed))
            noi_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
            cod_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 2)
            tgt_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 3)

            z, _, c_idx, c_disc, c_cont = self._sample_latent_and_codes(
                n,
                latent_generator=lat_gen,
                noise_generator=noi_gen,
                code_generator=cod_gen,
            )
            samples = self.G(torch.cat([z, c_disc, c_cont], dim=1))
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                samples = self.prior.postprocess_generated(samples)
            q_out = self.Q(samples)
            target = self.sample_target(n, generator=tgt_gen)
            return score_infogan(samples, c_idx, c_cont, q_logits=q_out, target=target)
        finally:
            for module, mode in modes:
                module.training = mode

    def state_dict(self):
        return {
            "version": VERSION,
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "completed_steps": self.completed_steps,
            "recipe": self.recipe.to_dict(),
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "recognition": self.Q.state_dict(),
            "prior": self.prior.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "data_rng": self.data_rng.get_state(),
            "latent_rng": self.latent_rng.get_state(),
            "noise_rng": self.noise_rng.get_state(),
            "code_rng": self.code_rng.get_state(),
            "eval_rng": self.eval_rng.get_state(),
            "batch_sequence_sha256": self.batch_sequence_sha256,
            "last_losses": deepcopy(self.last_losses),
        }

    def load_state_dict(self, state):
        if state.get("case_id") != self.case_id:
            raise ValueError("checkpoint case_id mismatch")
        self.completed_steps = int(state["completed_steps"])
        self.G.load_state_dict(state["generator"])
        self.D.load_state_dict(state["discriminator"])
        self.Q.load_state_dict(state["recognition"])
        self.prior.load_state_dict(state["prior"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        if "data_rng" in state:
            self.data_rng.set_state(state["data_rng"])
        if "latent_rng" in state:
            self.latent_rng.set_state(state["latent_rng"])
        if "noise_rng" in state:
            self.noise_rng.set_state(state["noise_rng"])
        if "code_rng" in state:
            self.code_rng.set_state(state["code_rng"])
        if "eval_rng" in state:
            self.eval_rng.set_state(state["eval_rng"])
        if "batch_sequence_sha256" in state:
            self.batch_sequence_sha256 = state["batch_sequence_sha256"]
        self.last_losses = deepcopy(state.get("last_losses", {}))


def build_case(
    case_id: str = CASE_ID,
    *,
    device: str | torch.device = "cpu",
    seed: int = 0,
    recipe_name: str | None = None,
    max_steps: int | None = None,
    recipe: Recipe | None = None,
    prior: nn.Module | None = None,
):
    if case_id != CASE_ID:
        raise ValueError(f"unknown infogan case: {case_id}")
    return InfoGANAnisotropicFixture(
        _infogan_case(),
        device=device,
        seed=seed,
        recipe_name=recipe_name or DEFAULT_RECIPE,
        max_steps=max_steps,
        recipe=recipe,
        prior=prior,
    )
