"""Public-API CramerGAN multi-sample unbiased energy distance fixture and scorer.

Implements Bellemare et al. (NeurIPS 2017, *The Cramer Distance as a Solution
to Biased Wasserstein Gradients*, arXiv:1705.10743):
- Critic equipped with a 16D vector embedding head h_psi: R^2 -> R^16 and scalar
  logit projection so critic penalties (BISPFCriticPenalty, KA2GradientPenalty,
  K3P/BCap) operate directly on scalar critic outputs across all 8 trainer families.
- Two independent fake batches x_hat_1 = G(z_1), x_hat_2 = G(z_2) per step form
  the unbiased energy critic surrogate:
      f(x; x_hat_2) = ||h_psi(x) - h_psi(x_hat_2)||_2 - ||h_psi(x)||_2
  and the two-sample energy distance generator objective:
      L_G^Cramer(x, x_hat_1, x_hat_2) = E[||h_psi(x) - h_psi(x_hat_1)||_2
          + ||h_psi(x) - h_psi(x_hat_2)||_2 - ||h_psi(x_hat_1) - h_psi(x_hat_2)||_2].
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

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
from .reproducibility import DEFAULT_SEED


CASE_ID = "api-cramer-energy-overlap"
TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/cramer_energy_overlap.json"
CRAMER_SIGMA = 0.16
CRAMER_COORDS = (-1.5, 0.0, 1.5)
CRAMER_WEIGHT_VALUES = (0.18, 0.08, 0.18, 0.08, 0.16, 0.08, 0.10, 0.06, 0.08)


def cramer_centers(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Return the 9 mode centers of the 3x3 grid at {-1.5, 0.0, +1.5}^2."""
    pts = [[x, y] for x in CRAMER_COORDS for y in CRAMER_COORDS]
    return torch.tensor(pts, device=device, dtype=dtype)


def cramer_weights(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Return the heterogeneous mode weights summing to 1.0."""
    return torch.tensor(CRAMER_WEIGHT_VALUES, device=device, dtype=dtype)


def sample_cramer_target(
    n: int,
    generator: torch.Generator | None = None,
    *,
    device="cpu",
    centers: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    sigma: float = CRAMER_SIGMA,
) -> torch.Tensor:
    """Sample N points from the 9-mode heterogeneous-weight overlapping 2D Gaussian mixture."""
    dev = torch.device(device)
    c = cramer_centers(device=dev) if centers is None else centers.to(device=dev, dtype=torch.float32)
    w = cramer_weights(device=dev) if weights is None else weights.to(device=dev, dtype=torch.float32)
    rng_dev = generator.device if generator is not None else dev
    w_rng = w.to(device=rng_dev, dtype=torch.float32)
    mode_ids = torch.multinomial(w_rng, int(n), replacement=True, generator=generator).to(device=dev)
    eps = torch.randn((int(n), 2), generator=generator, device=rng_dev, dtype=torch.float32).to(device=dev)
    return c[mode_ids] + float(sigma) * eps


def _cramer_case() -> dict:
    task = json.loads(TASK_PATH.read_text())
    host = task["execution"]["host_definition"]
    return {
        "id": CASE_ID,
        "legacy_ids": ["develop-cramer-energy-overlap"],
        "title": "Nine overlapping heterogeneous Gaussians: CramerGAN two-sample energy distance",
        "goal": task["description"],
        "kind": "vector",
        "default_steps": int(task["execution"]["steps"]),
        "batch_size": int(host["batch"]),
        "eval_samples": int(task["evaluation"]["eval_samples"]),
        "default_recipe": "bispf",
        "terminal_observations": 5,
        "evaluation_observations": 24,
        "protocol_seed": DEFAULT_SEED,
        "thresholds": {
            "modes": 9,
            "hq_min": 0.88,
            "mass_tv_max": 0.10,
            "energy_distance_max": 0.06,
            "min_cov_eig_ratio_min": 0.35,
        },
        "sampling": (
            "9-mode 3x3 2D Gaussian mixture at {-1.5, 0.0, +1.5}^2 (sigma=0.16, "
            "weights=(0.18, 0.08, 0.18, 0.08, 0.16, 0.08, 0.10, 0.06, 0.08)); "
            "learned 256-particle prior; two independent fake draws per update"
        ),
        "scope": (
            "Tier 4 multi-sample unbiased energy distance (CramerGAN, Bellemare et al., "
            "NeurIPS 2017) on 9-mode overlapping heterogeneous-mass 2D Gaussian mixture."
        ),
    }


def list_cases() -> list[dict]:
    return [_cramer_case()]


@torch.no_grad()
def score_cramergan(
    samples: torch.Tensor,
    target: torch.Tensor | None = None,
    *,
    centers: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    sigma: float = CRAMER_SIGMA,
    eval_seed: int = 10001,
) -> dict:
    """Evaluate N=4096 generated samples against the 9-mode CramerGAN gates.

    Gates:
    - modes == 9 (>= 30 high-quality samples within 3*sigma = 0.48 per mode)
    - hq >= 0.88 (fraction of samples within 3*sigma = 0.48 of nearest mode center)
    - mass_tv <= 0.10 (TV distance between empirical mode frequencies and true weights w)
    - energy_distance <= 0.06 (empirical 2D energy distance 2E||X-Y|| - E||X-X'|| - E||Y-Y'||)
    - min_cov_eig_ratio >= 0.35 (minimum within-mode covariance eigenvalue / sigma^2)
    """
    samples = torch.as_tensor(samples)
    dev = samples.device
    c = cramer_centers(device=dev) if centers is None else centers.to(device=dev, dtype=torch.float32)
    w = cramer_weights(device=dev) if weights is None else weights.to(device=dev, dtype=torch.float32)
    if target is None:
        rng = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        target = sample_cramer_target(len(samples), generator=rng, device=dev, centers=c, weights=w, sigma=sigma)
    else:
        target = torch.as_tensor(target, device=dev, dtype=torch.float32)

    if (
        samples.ndim != 2
        or samples.shape[1] != 2
        or len(samples) == 0
        or not torch.all(torch.isfinite(samples))
        or not torch.all(torch.isfinite(target))
    ):
        metrics = {
            "modes": 0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "energy_distance": float("inf"),
            "min_cov_eig_ratio": 0.0,
        }
        return {
            "passed": False,
            "failed_bounds": [
                "modes",
                "hq",
                "mass_tv",
                "energy_distance",
                "min_cov_eig_ratio",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": [
                dict(
                    kind="scatter",
                    title="CramerGAN 2D output vs 9-mode overlapping mixture",
                    target=torch.zeros((1, 2), dtype=torch.float32),
                    samples=torch.zeros((1, 2), dtype=torch.float32),
                ),
                dict(
                    kind="bar",
                    title="Heterogeneous 9-mode mass distribution",
                    target=w.detach().cpu(),
                    samples=torch.zeros(9, dtype=torch.float32),
                ),
            ],
        }

    pts = samples.to(dtype=torch.float32)
    tgt = target.to(dtype=torch.float32)
    radius = 3.0 * float(sigma)
    dist, nearest = torch.cdist(pts, c).min(dim=1)
    hq_mask = dist <= radius
    hq = float(hq_mask.float().mean().item())

    hq_ids = nearest[hq_mask]
    hq_points = pts[hq_mask]
    counts = torch.bincount(hq_ids, minlength=9)
    modes = int((counts >= 30).sum().item())

    if int(counts.sum().item()) > 0:
        emp_weights = counts.float() / counts.sum().clamp_min(1).float()
        mass_tv = float(0.5 * (emp_weights - w).abs().sum().item())
    else:
        emp_weights = torch.zeros_like(w)
        mass_tv = 1.0

    # Empirical 2D energy distance: 2 E||X - Y||_2 - E||X - X'||_2 - E||Y - Y'||_2
    n_ed = min(len(pts), len(tgt))
    if n_ed > 1024 and n_ed % 1024 == 0:
        xs = pts[:n_ed].reshape(-1, 1024, 2)
        ys = tgt[:n_ed].reshape(-1, 1024, 2)
    else:
        sub = min(n_ed, 1024)
        xs = pts[:sub].unsqueeze(0)
        ys = tgt[:sub].unsqueeze(0)
    d_xy = torch.cdist(xs, ys).mean()
    d_xx = torch.cdist(xs, xs).mean()
    d_yy = torch.cdist(ys, ys).mean()
    energy_distance = float((2.0 * d_xy - d_xx - d_yy).clamp_min(0.0).item())

    # Minimum within-mode covariance eigenvalue ratio (divided by sigma^2 = 0.0256)
    if bool((counts >= 2).all()):
        residual = (hq_points - c[hq_ids]).to(torch.float64) / float(sigma)
        counts_f64 = counts.to(torch.float64).clamp_min(1.0)
        sums = torch.zeros((9, 2), dtype=torch.float64, device=dev)
        sums.index_add_(0, hq_ids, residual)
        centered_mean = sums / counts_f64[:, None]
        outer = (residual[:, :, None] * residual[:, None, :]).reshape(-1, 4)
        second = torch.zeros((9, 4), dtype=torch.float64, device=dev)
        second.index_add_(0, hq_ids, outer)
        covariance = (
            second.reshape(9, 2, 2) / counts_f64[:, None, None]
            - centered_mean[:, :, None] * centered_mean[:, None, :]
        )
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        min_cov_eig_ratio = float(eigenvalues[:, 0].min().item())
    else:
        min_cov_eig_ratio = 0.0

    metrics = {
        "modes": modes,
        "hq": hq,
        "mass_tv": mass_tv,
        "energy_distance": energy_distance,
        "min_cov_eig_ratio": min_cov_eig_ratio,
    }

    failed_bounds = []
    if metrics["modes"] != 9:
        failed_bounds.append("modes")
    if metrics["hq"] < 0.88:
        failed_bounds.append("hq")
    if metrics["mass_tv"] > 0.10:
        failed_bounds.append("mass_tv")
    if metrics["energy_distance"] > 0.06:
        failed_bounds.append("energy_distance")
    if metrics["min_cov_eig_ratio"] < 0.35:
        failed_bounds.append("min_cov_eig_ratio")

    passed = len(failed_bounds) == 0
    views = [
        dict(
            kind="scatter",
            title="CramerGAN 2D output vs 9-mode overlapping mixture",
            target=tgt.detach().cpu(),
            samples=pts.detach().cpu(),
            xlabel="x1",
            ylabel="x2",
            caption=(
                f"HQ={hq:.3f} (>=0.88), EnergyDist={energy_distance:.4f} (<=0.06), "
                f"MinCovEig={min_cov_eig_ratio:.3f} (>=0.35)"
            ),
        ),
        dict(
            kind="bar",
            title="Heterogeneous 9-mode mass distribution",
            target=w.detach().cpu(),
            samples=emp_weights.detach().cpu(),
            xlabel="Mode index (0..8)",
            ylabel="Probability mass",
            caption=f"Modes={modes}/9, Mass TV={mass_tv:.4f} (<=0.10)",
        ),
    ]
    return dict(
        metrics=metrics,
        passed=passed,
        failed_bounds=sorted(set(failed_bounds)),
        views=views,
    )


class CramerGenerator(nn.Module):
    """3-layer 128-wide LeakyReLU MLP generator G: R^2 -> R^2."""

    def __init__(self, z_dim: int = 2, out_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.z_dim = int(z_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CramerCritic(nn.Module):
    """Critic D: R^2 -> R with exposed 16D embedding head h_psi(x) in R^16 and scalar logit projection."""

    def __init__(self, in_dim: int = 2, embed_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU(0.2),
        )
        self.embedding_head = nn.Linear(self.hidden, self.embed_dim)
        self.logit_head = nn.Linear(self.embed_dim, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return pure learned 16D critic embedding h_psi(x) without data-space identity skips."""
        return self.embedding_head(self.trunk(x))

    def h_psi(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit_head(self.embed(x)).squeeze(-1)


def cramer_critic_surrogate(h_x: torch.Tensor, h_fake2: torch.Tensor) -> torch.Tensor:
    """Bellemare et al. (2017) energy critic surrogate:
    f(x; x_hat_2) = E_{x_hat_2}[||h_psi(x) - h_psi(x_hat_2)||_2] - ||h_psi(x)||_2.
    """
    pairwise = torch.sqrt((h_x[:, None, :] - h_fake2[None, :, :]).square().sum(dim=-1) + 1e-8)
    anchor = torch.sqrt(h_x.square().sum(dim=-1) + 1e-8)
    return pairwise.mean(dim=1) - anchor


def cramer_generator_loss(
    h_real: torch.Tensor,
    h_fake1: torch.Tensor,
    h_fake2: torch.Tensor,
    real_x: torch.Tensor | None = None,
    fake_x1: torch.Tensor | None = None,
    fake_x2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-sample unbiased Cramer energy distance generator objective in learned critic embedding space:
    L_G^Cramer(x, x_hat_1, x_hat_2) = E[||h(x) - h(x_hat_1)||_2 + ||h(x) - h(x_hat_2)||_2 - ||h(x_hat_1) - h(x_hat_2)||_2].
    """
    del real_x, fake_x1, fake_x2
    d_r1 = torch.sqrt((h_real[:, None, :] - h_fake1[None, :, :]).square().sum(dim=-1) + 1e-8)
    d_r2 = torch.sqrt((h_real[:, None, :] - h_fake2[None, :, :]).square().sum(dim=-1) + 1e-8)
    d_12 = torch.sqrt((h_fake1[:, None, :] - h_fake2[None, :, :]).square().sum(dim=-1) + 1e-8)
    return d_r1.mean() + d_r2.mean() - d_12.mean()


class CramerGANOverlapFixture:
    """Multi-family CramerGAN fixture on the 9-mode overlapping heterogeneous mixture."""

    api_components = (
        "particlegan.Recipe",
        "particlegan.ParticlePrior",
        "particlegan.MoGParticlePrior",
        "particlegan.ManipulatedPrior",
        "particlegan.Recipe.make_optimizers",
        "particlegan.Recipe.make_loss",
        "particlegan.Recipe.make_critic_penalty",
        "particlegan.Recipe.prior_regularization",
        "particlegan.init.deterministic_orthogonal_",
        "particlegan.scale_learning_rates",
    )

    def __init__(
        self,
        case: dict | None = None,
        *,
        recipe_name: str | None = "bispf",
        recipe: Recipe | None = None,
        prior: nn.Module | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
    ):
        self.metadata = deepcopy(case) if case is not None else _cramer_case()
        self.case = self.metadata
        self.case_id = self.case["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.completed_steps = 0
        self.default_steps = int(self.case.get("default_steps", 3000))
        self.batch_size = int(self.case.get("batch_size", 256))
        self.particles = 256
        self.z_dim = 2
        self.limit = self.default_steps if max_steps is None else int(max_steps)
        if type(self.limit) is not int or not 0 < self.limit <= self.default_steps:
            raise ValueError("max_steps must be in the declared budget")

        self.centers = cramer_centers(device=self.device)
        self.weights = cramer_weights(device=self.device)
        self.sigma = CRAMER_SIGMA

        self.G = CramerGenerator(z_dim=self.z_dim, out_dim=2, hidden=128).to(self.device)
        self.D = CramerCritic(in_dim=2, embed_dim=16, hidden=128).to(self.device)
        if prior is None:
            self.prior = ParticlePrior(num_particles=self.particles, z_dim=self.z_dim, init_std=1.0).to(self.device)
            init_prior = True
        else:
            self.prior = prior.to(self.device)
            init_prior = False

        init.deterministic_orthogonal_(self.G, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.D, seed=self.seed + 2)
        if init_prior:
            init.deterministic_orthogonal_(self.prior, seed=self.seed + 3)

        self.data_rng = torch.Generator(device=self.device).manual_seed(self.seed + 101)
        self.latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 102)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed + 103)
        self.eval_rng = torch.Generator(device=self.device).manual_seed(self.seed + 104)
        self.batch_digest = hashlib.sha256().hexdigest()
        self.last_losses = {}

        target_recipe = recipe if recipe is not None else (recipe_name or self.case.get("default_recipe", "bispf"))
        self.rebind_trainer(target_recipe, prior=self.prior)

    def _resolve_recipe(self, recipe_or_name: Recipe | str, prior: nn.Module) -> Recipe:
        base_prior = unwrap_prior(prior)
        p_kind = "mog" if isinstance(base_prior, MoGParticlePrior) else "particles"
        standardize = bool(getattr(base_prior, "standardize", False))
        if isinstance(recipe_or_name, Recipe):
            r = recipe_or_name
            updates = {}
            if r.prior_kind != p_kind:
                updates["prior_kind"] = p_kind
            if r.standardize != standardize:
                updates["standardize"] = standardize
            if r.z_dim != base_prior.z_dim:
                updates["z_dim"] = base_prior.z_dim
            if r.num_particles != base_prior.num_particles:
                updates["num_particles"] = base_prior.num_particles
            return r.replace(**updates) if updates else r

        name = str(recipe_or_name or "bispf")
        fields = get_recipe_fields(name)
        overrides = dict(
            batch_size=self.batch_size,
            z_dim=base_prior.z_dim,
            num_particles=base_prior.num_particles,
            prior_kind=p_kind,
            standardize=standardize,
        )
        if fields.get("continuous_policy") is None:
            overrides["total_steps"] = self.default_steps
        return get_recipe(name, **overrides)

    def rebind_trainer(self, recipe: Recipe | str, prior: nn.Module | None = None) -> None:
        """Bind or rebind any of the 8 ParticleGAN trainer families and prior types."""
        if prior is not None:
            self.prior = prior.to(self.device)
        self.recipe = self._resolve_recipe(recipe, self.prior)
        ema_critic = (
            None
            if self.recipe.effective_critic_formulation in _NO_EMA_FORMULATIONS
            else deepcopy(self.D)
        )
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            ema_critic=ema_critic,
            foreach=False,
            fused=False,
        )
        self.base_rates = [[group["lr"] for group in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.loss = self.recipe.make_loss()
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior, reference=self.prior)
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        bind_formulation_records(self.recipe, self.opt_d, self.loss, self.regularizer, generator=self.G)

    @property
    def batch_sequence_sha256(self) -> str:
        return self.batch_digest

    def sample_target(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_rng if generator is None else generator
        return sample_cramer_target(
            n,
            generator=rng,
            device=self.device,
            centers=self.centers,
            weights=self.weights,
            sigma=self.sigma,
        )

    def _sample_prior_batch(self, n: int, generator: torch.Generator | None = None):
        rng = self.latent_rng if generator is None else generator
        if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
            return self.prior.sample(n, generator=rng, noise_generator=self.noise_rng)
        return self.prior.sample(n, generator=rng)

    def _postprocess(self, generated: torch.Tensor) -> torch.Tensor:
        if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
            return self.prior.postprocess_generated(generated)
        return generated

    def _apply_lr_schedule(self) -> None:
        if (
            self.recipe.continuous_policy is None
            and getattr(self.recipe, "lr_control", "mobility") != "stationarity"
            and self.recipe.total_steps is not None
        ):
            scale_learning_rates(
                self.completed_steps,
                self.recipe,
                [self.opt_g, self.opt_d],
                self.base_rates,
                self.prior,
            )

    def step(self, real_x: torch.Tensor | None = None) -> dict:
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")

        self._apply_lr_schedule()
        if real_x is None:
            real_x = self.sample_target(self.batch_size)
        else:
            real_x = real_x.to(device=self.device, dtype=torch.float32)

        digest = hashlib.sha256(bytes.fromhex(self.batch_digest))
        digest.update(real_x.detach().cpu().contiguous().numpy().tobytes())
        self.batch_digest = digest.hexdigest()

        n_batch = len(real_x)
        self.G.train()
        self.D.train()

        # ------------------------------------------------------------------
        # Critic update: two independent fake batches x_hat_1, x_hat_2
        # ------------------------------------------------------------------
        with torch.no_grad():
            z1_d, _ = self._sample_prior_batch(n_batch)
            z2_d, _ = self._sample_prior_batch(n_batch)
            fake_x1_d = self._postprocess(self.G(z1_d))
            fake_x2_d = self._postprocess(self.G(z2_d))

        self.opt_d.zero_grad()
        d_real = self.D(real_x)
        d_fake1 = self.D(fake_x1_d)
        h_real_d = self.D.embed(real_x)
        h_fake1_d = self.D.embed(fake_x1_d)
        h_fake2_d = self.D.embed(fake_x2_d)

        f_real = cramer_critic_surrogate(h_real_d, h_fake2_d)
        f_fake = cramer_critic_surrogate(h_fake1_d, h_fake2_d)
        loss_cramer_d = F.softplus(-(f_real - f_fake)).mean()
        loss_d = self.loss.d_loss(d_real, d_fake1) + 0.25 * loss_cramer_d
        reg_d = self.penalty(self.D, real_x, fake_x1_d.detach())
        (loss_d + reg_d).backward()
        self.opt_d.step()

        # ------------------------------------------------------------------
        # Generator & Prior update: two-sample unbiased Cramer energy loss
        # ------------------------------------------------------------------
        with critic_g_step_mode(self.D):
            z1, idx1 = self._sample_prior_batch(n_batch)
            z2, idx2 = self._sample_prior_batch(n_batch)
            fake_x1 = self._postprocess(self.G(z1))
            fake_x2 = self._postprocess(self.G(z2))

            d_g1 = self.D(fake_x1)
            d_g2 = self.D(fake_x2)
            d_r = self.D(real_x)
            h_real_g = self.D.embed(real_x)
            h_fake1_g = self.D.embed(fake_x1)
            h_fake2_g = self.D.embed(fake_x2)

            loss_cramer_g = cramer_generator_loss(h_real_g, h_fake1_g, h_fake2_g)
            loss_gan = 0.5 * (self.loss.g_loss(d_g1, d_r) + self.loss.g_loss(d_g2, d_r))
            loss_adv = loss_gan + 0.5 * loss_cramer_g

            base_prior = unwrap_prior(self.prior)
            all_idx = torch.cat([idx1, idx2], dim=0)
            loss_prior = self.recipe.prior_regularization(
                base_prior.z,
                regularizer=self.regularizer,
                prior=self.prior,
            )
            if isinstance(self.prior, ManipulatedPrior) and self.prior.has_regularizer:
                raw_rows = base_prior.z if base_prior.num_particles <= 1024 else base_prior.z[torch.unique(all_idx)]
                loss_prior = loss_prior + self.prior.regularize(
                    raw_rows,
                    sampled_codes=z1,
                    indices=idx1,
                    generated=fake_x1,
                    logits=d_g1,
                )

            loss_g = loss_adv + loss_prior
            self.opt_g.zero_grad()
            if hasattr(self.opt_g, "set_sampled_rows") and base_prior.z.requires_grad:
                self.opt_g.set_sampled_rows(base_prior.z, all_idx)
            if hasattr(self.opt_g, "bind_origin_sample"):
                self.opt_g.bind_origin_sample(fake_x1)

            if self.recipe.constraint_geometry_mode != "none":
                protected_evaluator = None
                if self.recipe.constraint_geometry_mode == "strict_progress":
                    locations = base_prior.z.detach().clone()
                    sampled_z1 = z1.detach().clone()
                    sampled_z2 = z2.detach().clone()

                    def protected_evaluator():
                        rep1 = sampled_z1 + (base_prior.z[idx1] - locations[idx1])
                        rep2 = sampled_z2 + (base_prior.z[idx2] - locations[idx2])
                        fg1 = self._postprocess(self.G(rep1))
                        fg2 = self._postprocess(self.G(rep2))
                        dg1 = self.D(fg1)
                        dg2 = self.D(fg2)
                        hg1 = self.D.embed(fg1)
                        hg2 = self.D.embed(fg2)
                        cg = cramer_generator_loss(h_real_g, hg1, hg2)
                        lg = 0.5 * (self.loss.g_loss(dg1, d_r) + self.loss.g_loss(dg2, d_r))
                        return (lg + 0.5 * cg,)

                constraint_geometry_backward(loss_g, self.opt_g, (loss_adv,), protected_evaluator=protected_evaluator)
            else:
                loss_g.backward()
            self.opt_g.step()

        self.completed_steps += 1
        self.last_losses = {
            "loss_d": float(loss_d.detach().item()),
            "loss_g": float(loss_g.detach().item()),
            "loss_cramer_d": float(loss_cramer_d.detach().item()),
            "loss_cramer_g": float(loss_cramer_g.detach().item()),
            "reg_d": float(reg_d.detach().item()),
            "prior_regularization": float(loss_prior.detach().item()),
        }
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def sample(
        self,
        n: int = 4096,
        *,
        generator: torch.Generator | None = None,
        ema: bool = False,
        output_noise: bool = False,
    ) -> torch.Tensor:
        del ema, output_noise
        was_g = self.G.training
        was_p = self.prior.training
        try:
            self.G.eval()
            self.prior.eval()
            rng = self.eval_rng if generator is None else generator
            if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
                z, _ = self.prior.sample(int(n), generator=rng, noise_generator=rng)
            else:
                z, _ = self.prior.sample(int(n), generator=rng)
            return self._postprocess(self.G(z))
        finally:
            self.G.train(was_g)
            self.prior.train(was_p)

    @torch.no_grad()
    def observe(self, n: int = 4096, seed: int = 10000) -> dict:
        was_g, was_d, was_p = self.G.training, self.D.training, self.prior.training
        try:
            self.G.eval()
            self.D.eval()
            self.prior.eval()
            eval_gen = torch.Generator(device=self.device).manual_seed(int(seed))
            target_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
            if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
                z, _ = self.prior.sample(int(n), generator=eval_gen, noise_generator=eval_gen)
            else:
                z, _ = self.prior.sample(int(n), generator=eval_gen)
            samples = self._postprocess(self.G(z))
            target = self.sample_target(int(n), generator=target_gen)
            return score_cramergan(samples, target, centers=self.centers, weights=self.weights, sigma=self.sigma)
        finally:
            self.G.train(was_g)
            self.D.train(was_d)
            self.prior.train(was_p)

    def state_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "completed_steps": self.completed_steps,
            "batch_sequence_sha256": self.batch_digest,
            "recipe": self.recipe.to_dict(),
            "generator": self.G.state_dict(),
            "critic": self.D.state_dict(),
            "prior": self.prior.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "data_rng": self.data_rng.get_state(),
            "latent_rng": self.latent_rng.get_state(),
            "noise_rng": self.noise_rng.get_state(),
            "eval_rng": self.eval_rng.get_state(),
            "last_losses": deepcopy(self.last_losses),
        }

    def load_state_dict(self, state: dict) -> None:
        self.completed_steps = int(state["completed_steps"])
        self.batch_digest = str(state.get("batch_sequence_sha256", self.batch_digest))
        self.G.load_state_dict(state["generator"])
        self.D.load_state_dict(state["critic"])
        self.prior.load_state_dict(state["prior"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        if "data_rng" in state:
            self.data_rng.set_state(state["data_rng"])
        if "latent_rng" in state:
            self.latent_rng.set_state(state["latent_rng"])
        if "noise_rng" in state:
            self.noise_rng.set_state(state["noise_rng"])
        if "eval_rng" in state:
            self.eval_rng.set_state(state["eval_rng"])
        self.last_losses = deepcopy(state.get("last_losses", {}))


def build_case(
    case_id: str,
    *,
    device: str | torch.device = "cpu",
    seed: int = DEFAULT_SEED,
    recipe_name: str | None = None,
    max_steps: int | None = None,
    recipe: Recipe | None = None,
    prior: nn.Module | None = None,
    **kwargs,
) -> CramerGANOverlapFixture:
    del kwargs
    if case_id != CASE_ID:
        raise ValueError(f"unknown CramerGAN case: {case_id}")
    return CramerGANOverlapFixture(
        _cramer_case(),
        recipe_name=recipe_name or "bispf",
        recipe=recipe,
        prior=prior,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
