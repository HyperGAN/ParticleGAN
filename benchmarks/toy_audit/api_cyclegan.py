"""Unpaired 2D domain translation & bijection (CycleGAN / DiscoGAN) toy audit.

Implements the multimodal constellation bijection benchmark inspired by:
- Kim et al. (ICML 2017), "Learning to Discover Cross-Domain Relations with
  Generative Adversarial Networks" (DiscoGAN, arXiv:1703.05192, Figure 2 toy).
- Zhu et al. (ICCV 2017), "Unpaired Image-to-Image Translation using
  Cycle-Consistent Adversarial Networks" (CycleGAN, arXiv:1703.10593).

Domain X is an 8-mode ring of radius R_X = 2.5 (sigma = 0.08).
Domain Y is an 8-mode square-perimeter constellation on {-2, 0, +2}^2 \\ {(0, 0)}
(sigma = 0.08). Unpaired adversarial translation with cycle-consistency
regularization establishes a mode-level bijection between Domain X and Domain Y.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    bind_formulation_records,
    critic_g_step_mode,
    get_recipe,
    get_recipe_fields,
    init,
    scale_learning_rates,
    unwrap_prior,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward


TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/cyclegan_unpaired_constellation.json"
CASE_ID = "api-cyclegan-unpaired-constellation"


def domain_x_centers(device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the 8 mode centers of Domain X (ring of radius R_X = 2.5)."""
    angles = torch.linspace(0.0, 2.0 * math.pi, 9, device=device, dtype=dtype)[:8]
    return torch.stack([2.5 * torch.cos(angles), 2.5 * torch.sin(angles)], dim=1)


def domain_y_centers(device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the 8 mode centers of Domain Y (square perimeter {-2, 0, +2}^2 \\ {(0, 0)}).

    Ordered counter-clockwise by ray angle theta_k = 2*pi*k/8 (k=0..7) so index k
    lies on the same ray angle as Domain X mode k.
    """
    coords = [
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
        (-2.0, 2.0),
        (-2.0, 0.0),
        (-2.0, -2.0),
        (0.0, -2.0),
        (2.0, -2.0),
    ]
    return torch.tensor(coords, device=device, dtype=dtype)


def _cyclegan_case() -> Dict[str, Any]:
    task = json.loads(TASK_PATH.read_text())
    spec = deepcopy(task["execution"]["host_definition"])
    return {
        "id": CASE_ID,
        "legacy_ids": ["develop-cyclegan-unpaired-constellation"],
        "title": "Unpaired 2D domain translation & bijection (CycleGAN / DiscoGAN)",
        "goal": task["description"],
        "kind": "vector",
        "default_steps": int(task["execution"]["steps"]),
        "batch_size": int(spec["batch"]),
        "eval_samples": int(task["evaluation"]["eval_samples"]),
        "default_recipe": "bispf",
        "particles": int(spec["particles"]),
        "z_dim": int(spec["z_dim"]),
        "terminal_observations": 5,
        "evaluation_observations": 20,
        "protocol_seed": 0,
        "task_sha256": hashlib.sha256(TASK_PATH.read_bytes()).hexdigest(),
        "thresholds": {
            "modes_y_min": 7,
            "modes_x_min": 7,
            "hq_y_min": 0.85,
            "hq_x_min": 0.85,
            "mass_tv_y_max": 0.15,
            "cycle_rmse_max": 0.12,
            "bijection_ber_max": 0.15,
        },
        "forge_thresholds": deepcopy(task["evaluation"]["thresholds"]),
        "sampling": (
            "Unpaired draws x ~ P_X (8-mode ring, R=2.5, sigma=0.08) and "
            "y ~ P_Y (8-mode square perimeter {-2, 0, +2}^2 \\ {(0,0)}, sigma=0.08); "
            "evaluated on N=4096 held-out unpaired samples without output noise."
        ),
        "scope": (
            "Unpaired 2D cross-domain translation (G_xy, G_yx, D_y, D_x) with "
            "cycle-consistency regularization and recipe-owned particle prior coupling."
        ),
    }


def list_cases():
    return [_cyclegan_case()]


@torch.no_grad()
def score_cyclegan(
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
    fake_x: torch.Tensor,
    cyc_x: torch.Tensor,
    cyc_y: torch.Tensor,
    mode_y_from_x_centers: torch.Tensor | None = None,
    *,
    centers_x: torch.Tensor | None = None,
    centers_y: torch.Tensor | None = None,
    sigma: float = 0.08,
) -> Dict[str, Any]:
    """Evaluate unpaired 2D CycleGAN translation, cycle consistency, and mode bijection."""
    device = real_x.device
    if centers_x is None:
        centers_x = domain_x_centers(device=device, dtype=real_x.dtype)
    else:
        centers_x = centers_x.to(device=device, dtype=real_x.dtype)
    if centers_y is None:
        centers_y = domain_y_centers(device=device, dtype=real_y.dtype)
    else:
        centers_y = centers_y.to(device=device, dtype=real_y.dtype)

    tensors = [real_x, real_y, fake_y, fake_x, cyc_x, cyc_y]
    if mode_y_from_x_centers is not None:
        tensors.append(mode_y_from_x_centers)

    if not all(torch.all(torch.isfinite(t)) for t in tensors):
        metrics = {
            "modes_y": 0,
            "modes_x": 0,
            "hq_y": 0.0,
            "hq_x": 0.0,
            "mass_tv_y": 1.0,
            "cycle_rmse": float("inf"),
            "bijection_ber": 1.0,
        }
        return {
            "passed": False,
            "failed_bounds": [
                "bijection_ber",
                "cycle_rmse",
                "hq_x",
                "hq_y",
                "mass_tv_y",
                "modes_x",
                "modes_y",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": [],
        }

    radius_gate = 3.0 * float(sigma)  # 0.24 for sigma = 0.08

    # Domain Y coverage & high-quality rate from translated fake_y = G_xy(real_x)
    dist_y, nearest_y = torch.cdist(fake_y, centers_y).min(dim=1)
    in_gate_y = dist_y <= radius_gate
    hq_y = in_gate_y.float().mean()
    counts_y = torch.bincount(nearest_y[in_gate_y], minlength=8)
    modes_y = int((counts_y >= 10).sum().item())
    mass_tv_y = 0.5 * (counts_y.float() / counts_y.sum().clamp_min(1) - (1.0 / 8.0)).abs().sum()

    # Domain X coverage & high-quality rate from translated fake_x = G_yx(real_y)
    dist_x, nearest_x = torch.cdist(fake_x, centers_x).min(dim=1)
    in_gate_x = dist_x <= radius_gate
    hq_x = in_gate_x.float().mean()
    counts_x = torch.bincount(nearest_x[in_gate_x], minlength=8)
    modes_x = int((counts_x >= 10).sum().item())

    # Combined cycle-consistency RMSE:
    # 0.5 * (sqrt(E||G_yx(G_xy(x)) - x||_2^2 / 2) + sqrt(E||G_xy(G_yx(y)) - y||_2^2 / 2))
    rmse_x = torch.sqrt(((cyc_x - real_x) ** 2).mean())
    rmse_y = torch.sqrt(((cyc_y - real_y) ** 2).mean())
    cycle_rmse = 0.5 * (rmse_x + rmse_y)

    # Bijection Error Rate (BER) on the 8 source mode centers mu_k^X -> G_xy(mu_k^X)
    if mode_y_from_x_centers is None:
        src_modes = torch.cdist(real_x, centers_x).argmin(dim=1)
        mapped_centers = []
        for k in range(8):
            mask = src_modes == k
            if mask.any():
                mapped_centers.append(fake_y[mask].mean(dim=0))
            else:
                mapped_centers.append(torch.zeros(2, device=device, dtype=real_x.dtype))
        mode_y_from_x_centers = torch.stack(mapped_centers, dim=0)
    else:
        mode_y_from_x_centers = mode_y_from_x_centers.to(device=device, dtype=real_x.dtype)

    pi = torch.cdist(mode_y_from_x_centers, centers_y).argmin(dim=1)
    unique_targets = len(set(pi.detach().cpu().tolist()))
    bijection_ber = 1.0 - (float(unique_targets) / 8.0)

    metrics = {
        "modes_y": int(modes_y),
        "modes_x": int(modes_x),
        "hq_y": float(hq_y.item()),
        "hq_x": float(hq_x.item()),
        "mass_tv_y": float(mass_tv_y.item()),
        "cycle_rmse": float(cycle_rmse.item()),
        "bijection_ber": float(bijection_ber),
    }

    failed_bounds = []
    if metrics["modes_y"] < 7:
        failed_bounds.append("modes_y")
    if metrics["modes_x"] < 7:
        failed_bounds.append("modes_x")
    if metrics["hq_y"] < 0.85:
        failed_bounds.append("hq_y")
    if metrics["hq_x"] < 0.85:
        failed_bounds.append("hq_x")
    if metrics["mass_tv_y"] > 0.15:
        failed_bounds.append("mass_tv_y")
    if metrics["cycle_rmse"] > 0.12:
        failed_bounds.append("cycle_rmse")
    if metrics["bijection_ber"] > 0.15:
        failed_bounds.append("bijection_ber")

    passed = len(failed_bounds) == 0
    views = [
        dict(
            kind="scatter",
            title="Forward translation G_xy(x) vs Domain Y square",
            target=real_y.detach().cpu(),
            samples=fake_y.detach().cpu(),
        ),
        dict(
            kind="scatter",
            title="Inverse translation G_yx(y) vs Domain X ring",
            target=real_x.detach().cpu(),
            samples=fake_x.detach().cpu(),
        ),
        dict(
            kind="scatter",
            title="Cycle reconstruction G_yx(G_xy(x)) vs Domain X",
            target=real_x.detach().cpu(),
            samples=cyc_x.detach().cpu(),
        ),
        dict(
            kind="bar",
            title="Domain Y mode mass distribution",
            target=torch.full((8,), 1.0 / 8.0),
            samples=counts_y.float().cpu() / counts_y.sum().clamp_min(1).cpu(),
        ),
    ]
    return {
        "metrics": metrics,
        "passed": passed,
        "failed_bounds": sorted(set(failed_bounds)),
        "views": views,
    }


class CycleGANTranslator(nn.Module):
    """2D cross-domain translator network with residual skip and optional prior modulation."""

    def __init__(self, in_dim: int = 2, out_dim: int = 2, hidden: int = 128, z_dim: int = 2, res_scale: float = 0.25):
        super().__init__()
        self.res_scale = float(res_scale)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )
        self.z_adapter = nn.Linear(z_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        out = x + self.res_scale * self.net(x)
        if z is not None:
            out = out + 0.005 * torch.tanh(self.z_adapter(z))
        return out


class PriorAnchorHead(nn.Module):
    """Maps particle prior latents z in R^2 to synthetic Domain X anchor samples."""

    def __init__(self, z_dim: int = 2, out_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + 0.25 * self.net(z)


class CycleGANCritic(nn.Module):
    """2D domain critic D: R^2 -> R."""

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


def _build_cyclegan_recipe(
    recipe_name: str,
    batch_size: int,
    total_steps: int,
    particles: int,
    z_dim: int,
    prior: nn.Module | None = None,
) -> Recipe:
    fields = get_recipe_fields(recipe_name)
    use_particle_table = bool(fields.get("particle_birth_death", False)) or (
        prior is not None and isinstance(unwrap_prior(prior), ParticlePrior)
    )
    kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        z_dim=z_dim,
        num_particles=particles,
        prior_kind="particles" if use_particle_table else "mog",
        sigma_rel=0.0 if use_particle_table else 0.08,
        standardize=False,
        output_noise_mode="fixed",
        output_noise_std=0.0,
        input_noise_std=0.0,
        output_noise_warmup=0.0,
    )
    if fields.get("continuous_policy") is None:
        kwargs["total_steps"] = total_steps
    return get_recipe(recipe_name, **kwargs)


class CycleGANConstellationFixture:
    """Public-API unpaired 2D CycleGAN / DiscoGAN constellation fixture."""

    api_components = (
        "particlegan.Recipe",
        "particlegan.MoGParticlePrior",
        "particlegan.ParticlePrior",
        "particlegan.ManipulatedPrior",
        "particlegan.init",
    )

    def __init__(
        self,
        case: Dict[str, Any] | None = None,
        *,
        recipe_name: str = "bispf",
        recipe: Recipe | None = None,
        prior: nn.Module | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
        max_steps: int | None = None,
        lambda_cyc: float = 10.0,
        lambda_prior: float = 0.25,
    ) -> None:
        if case is None:
            case = _cyclegan_case()
        self.metadata = deepcopy(case)
        self.case = self.metadata
        self.case_id = self.metadata["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.completed_steps = 0
        self.lambda_cyc = float(lambda_cyc)
        self.lambda_prior = float(lambda_prior)

        default_steps = int(self.metadata.get("default_steps", 3500))
        self.limit = default_steps if max_steps is None else int(max_steps)
        if type(self.limit) is not int or not (1 <= self.limit <= default_steps):
            raise ValueError("max_steps must be a positive integer within the declared budget")

        batch_size = int(self.metadata.get("batch_size", 256))
        particles = int(self.metadata.get("particles", 256))
        z_dim = int(self.metadata.get("z_dim", 2))

        if recipe is None:
            recipe = _build_cyclegan_recipe(
                recipe_name or self.metadata.get("default_recipe", "bispf"),
                batch_size=batch_size,
                total_steps=default_steps,
                particles=particles,
                z_dim=z_dim,
                prior=prior,
            )

        self.centers_x = domain_x_centers(device=self.device, dtype=torch.float32)
        self.centers_y = domain_y_centers(device=self.device, dtype=torch.float32)
        self.sigma = 0.08

        # Translators, particle prior anchor head, and domain critics
        self.G_xy = CycleGANTranslator(in_dim=2, out_dim=2, hidden=128, z_dim=z_dim).to(self.device)
        self.G_yx = CycleGANTranslator(in_dim=2, out_dim=2, hidden=128, z_dim=z_dim).to(self.device)
        self.G_prior = PriorAnchorHead(z_dim=z_dim, out_dim=2, hidden=64).to(self.device)
        self.D_y = CycleGANCritic(in_dim=2, hidden=128).to(self.device)
        self.D_x = CycleGANCritic(in_dim=2, hidden=128).to(self.device)

        self.G_bundle = nn.ModuleDict({
            "g_xy": self.G_xy,
            "g_yx": self.G_yx,
            "g_prior": self.G_prior,
        }).to(self.device)
        self.D_bundle = nn.ModuleDict({
            "d_y": self.D_y,
            "d_x": self.D_x,
        }).to(self.device)

        if prior is None:
            prior = MoGParticlePrior(
                num_particles=particles,
                z_dim=z_dim,
                sigma=self.sigma,
                standardize=False,
                init_std=1.0,
            ).to(self.device)
        else:
            prior = prior.to(self.device)

        init.deterministic_orthogonal_(self.G_xy, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.G_yx, seed=self.seed + 2)
        init.deterministic_orthogonal_(self.G_prior, seed=self.seed + 3)
        init.deterministic_orthogonal_(self.D_y, seed=self.seed + 4)
        init.deterministic_orthogonal_(self.D_x, seed=self.seed + 5)
        init.deterministic_orthogonal_(unwrap_prior(prior), seed=self.seed + 6)

        # Isolated RNG streams for unpaired domains X and Y, prior sampling, noise, and evaluation
        self.data_x_rng = torch.Generator(device=self.device).manual_seed(self.seed + 101)
        self.data_y_rng = torch.Generator(device=self.device).manual_seed(self.seed + 202)
        self.prior_rng = torch.Generator(device=self.device).manual_seed(self.seed + 303)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed + 404)
        self.eval_rng = torch.Generator(device=self.device).manual_seed(self.seed + 505)

        self.last_losses: Dict[str, float] = {}
        self.rebind_trainer(recipe, prior=prior)

    def rebind_trainer(self, recipe: Recipe, prior: nn.Module | None = None) -> None:
        """Bind or rebind a ParticleGAN Recipe and optional prior to the CycleGAN bundles."""
        self.recipe = recipe
        if prior is not None:
            self.prior = prior.to(self.device)
        self.recipe.apply_prior_policy(self.prior)

        needs_ema = (
            getattr(self.recipe, "effective_critic_formulation", "") != "bispf"
            and getattr(self.recipe, "reg_anchor_weight", 0.0) != 0.0
        )
        self.ema_critic = deepcopy(self.D_bundle).to(self.device) if needs_ema else None

        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G_bundle,
            self.D_bundle,
            self.prior,
            ema_critic=self.ema_critic,
            foreach=False,
            fused=False,
        )
        self.base_rates = [
            [float(group["lr"]) for group in opt.param_groups]
            for opt in (self.opt_g, self.opt_d)
        ]
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        bind_formulation_records(self.recipe, self.opt_d, self.loss, self.regularizer, generator=self.G_bundle)

    def _sample_prior(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
            return self.prior.sample(
                batch_size,
                generator=self.prior_rng,
                noise_generator=self.noise_rng,
            )
        return self.prior.sample(batch_size, generator=self.prior_rng)

    def sample_domain_x(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_x_rng if generator is None else generator
        mode_ids = torch.randint(0, 8, (n,), generator=rng, device=self.device)
        eps = torch.randn(n, 2, generator=rng, device=self.device)
        return self.centers_x[mode_ids] + self.sigma * eps

    def sample_domain_y(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_y_rng if generator is None else generator
        mode_ids = torch.randint(0, 8, (n,), generator=rng, device=self.device)
        eps = torch.randn(n, 2, generator=rng, device=self.device)
        return self.centers_y[mode_ids] + self.sigma * eps

    def step(self) -> Dict[str, Any]:
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")

        if (
            getattr(self.recipe, "continuous_policy", None) is None
            and getattr(self.recipe, "lr_control", "mobility") != "stationarity"
        ):
            scale_learning_rates(
                self.completed_steps,
                self.recipe,
                [self.opt_g, self.opt_d],
                self.base_rates,
                self.prior,
            )

        batch_size = int(self.recipe.batch_size)
        real_x = self.sample_domain_x(batch_size)
        real_y = self.sample_domain_y(batch_size)

        self.G_bundle.train()
        self.D_bundle.train()

        # ------------------------------------------------------------------
        # 1. Critic update (D_y on Domain Y and D_x on Domain X)
        # ------------------------------------------------------------------
        with torch.no_grad():
            z_d, idx_d = self._sample_prior(batch_size)
            fake_y_d = self.G_xy(real_x, z_d)
            fake_x_d = self.G_yx(real_y, z_d)
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                fake_y_d = self.prior.postprocess_generated(
                    fake_y_d, z=z_d, idx=idx_d, generator=self.noise_rng
                )
                fake_x_d = self.prior.postprocess_generated(
                    fake_x_d, z=z_d, idx=idx_d, generator=self.noise_rng
                )

        self.opt_d.zero_grad()
        d_real_y = self.D_y(real_y)
        d_fake_y = self.D_y(fake_y_d)
        loss_d_y = self.loss.d_loss(d_real_y, d_fake_y)
        reg_d_y = self.penalty(self.D_y, real_y, fake_y_d)

        if hasattr(self.opt_d, "record") and getattr(self.opt_d.record, "calls", 0) > 0:
            self.opt_d.record.calls -= 1

        d_real_x = self.D_x(real_x)
        d_fake_x = self.D_x(fake_x_d)
        loss_d_x = self.loss.d_loss(d_real_x, d_fake_x)
        reg_d_x = self.penalty(self.D_x, real_x, fake_x_d)

        loss_d = 0.5 * (loss_d_y + loss_d_x)
        reg_d = 0.5 * (reg_d_y + reg_d_x)
        (loss_d + reg_d).backward()
        self.opt_d.step()

        # ------------------------------------------------------------------
        # 2. Translator + Prior update (G_xy, G_yx, G_prior, prior.z)
        # ------------------------------------------------------------------
        with critic_g_step_mode(self.D_bundle):
            self.opt_g.zero_grad()

            z_g, idx_g = self._sample_prior(batch_size)
            base_prior = unwrap_prior(self.prior)
            if hasattr(self.opt_g, "set_sampled_rows") and base_prior.z.requires_grad:
                self.opt_g.set_sampled_rows(base_prior.z, idx_g)

            fake_y = self.G_xy(real_x, z_g)
            fake_x = self.G_yx(real_y, z_g)
            cyc_x = self.G_yx(fake_y, z_g)
            cyc_y = self.G_xy(fake_x, z_g)

            # Prior anchor synthetic samples in Domain X
            x_prior = self.G_prior(z_g)

            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                fake_y = self.prior.postprocess_generated(
                    fake_y, z=z_g, idx=idx_g, generator=self.noise_rng
                )
                fake_x = self.prior.postprocess_generated(
                    fake_x, z=z_g, idx=idx_g, generator=self.noise_rng
                )
                x_prior = self.prior.postprocess_generated(
                    x_prior, z=z_g, idx=idx_g, generator=self.noise_rng
                )

            with torch.no_grad():
                d_real_y_g = self.D_y(real_y)
                d_real_x_g = self.D_x(real_x)

            adv_y = self.loss.g_loss(self.D_y(fake_y), d_real_y_g)
            adv_x = self.loss.g_loss(self.D_x(fake_x), d_real_x_g)
            adv_prior_x = self.loss.g_loss(self.D_x(x_prior), d_real_x_g)

            loss_adv_main = 0.5 * (adv_y + adv_x)
            loss_adv = loss_adv_main + self.lambda_prior * adv_prior_x

            # Bidirectional cycle-consistency loss (Zhu et al. 2017; Kim et al. 2017)
            loss_cyc = F.mse_loss(cyc_x, real_x) + F.mse_loss(cyc_y, real_y)

            # Recipe-owned prior regularization
            reg_prior = self.recipe.prior_regularization(
                base_prior.z, regularizer=self.regularizer, prior=self.prior
            )
            if isinstance(self.prior, ManipulatedPrior) and self.prior.has_regularizer:
                reg_prior = reg_prior + self.prior.regularize(
                    z=z_g, idx=idx_g, x_fake=x_prior
                )

            loss_g = loss_adv + self.lambda_cyc * loss_cyc + reg_prior

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
            "loss_cyc": float(loss_cyc.detach().item()),
            "reg_d": float(reg_d.detach().item()),
        }
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def observe(self, n: int = 4096, seed: int = 10000) -> Dict[str, Any]:
        self.G_bundle.eval()
        self.D_bundle.eval()
        rng_x = torch.Generator(device=self.device).manual_seed(int(seed))
        rng_y = torch.Generator(device=self.device).manual_seed(int(seed) + 1)

        real_x = self.sample_domain_x(n, generator=rng_x)
        real_y = self.sample_domain_y(n, generator=rng_y)

        fake_y = self.G_xy(real_x)
        fake_x = self.G_yx(real_y)
        cyc_x = self.G_yx(fake_y)
        cyc_y = self.G_xy(fake_x)
        mode_y_from_x_centers = self.G_xy(self.centers_x)

        return score_cyclegan(
            real_x=real_x,
            real_y=real_y,
            fake_y=fake_y,
            fake_x=fake_x,
            cyc_x=cyc_x,
            cyc_y=cyc_y,
            mode_y_from_x_centers=mode_y_from_x_centers,
            centers_x=self.centers_x,
            centers_y=self.centers_y,
            sigma=self.sigma,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "completed_steps": self.completed_steps,
            "recipe": self.recipe.to_dict(),
            "g_bundle": self.G_bundle.state_dict(),
            "d_bundle": self.D_bundle.state_dict(),
            "prior": self.prior.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "data_x_rng": self.data_x_rng.get_state(),
            "data_y_rng": self.data_y_rng.get_state(),
            "prior_rng": self.prior_rng.get_state(),
            "noise_rng": self.noise_rng.get_state(),
            "eval_rng": self.eval_rng.get_state(),
            "last_losses": deepcopy(self.last_losses),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.completed_steps = int(state["completed_steps"])
        self.G_bundle.load_state_dict(state["g_bundle"])
        self.D_bundle.load_state_dict(state["d_bundle"])
        self.prior.load_state_dict(state["prior"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.data_x_rng.set_state(state["data_x_rng"].detach().to(device="cpu", dtype=torch.uint8).clone())
        self.data_y_rng.set_state(state["data_y_rng"].detach().to(device="cpu", dtype=torch.uint8).clone())
        self.prior_rng.set_state(state["prior_rng"].detach().to(device="cpu", dtype=torch.uint8).clone())
        self.noise_rng.set_state(state["noise_rng"].detach().to(device="cpu", dtype=torch.uint8).clone())
        self.eval_rng.set_state(state["eval_rng"].detach().to(device="cpu", dtype=torch.uint8).clone())
        self.last_losses = deepcopy(state["last_losses"])


def build_case(
    case_id: str,
    *,
    device: str | torch.device = "cpu",
    seed: int = 0,
    recipe_name: str | None = None,
    max_steps: int | None = None,
    recipe: Recipe | None = None,
    prior: nn.Module | None = None,
) -> CycleGANConstellationFixture:
    if case_id != CASE_ID:
        raise ValueError(f"unknown CycleGAN API case: {case_id}")
    return CycleGANConstellationFixture(
        _cyclegan_case(),
        recipe_name=recipe_name or "bispf",
        recipe=recipe,
        prior=prior,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
