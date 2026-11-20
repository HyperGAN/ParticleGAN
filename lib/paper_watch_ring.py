"""Public-API 8-Gaussian ring arms for paper-watch metric studies.

Paper-watch metric studies (Gini distance correlation, PR #559; CADMA,
PR #562) need real GAN trajectories whose coverage is known by
construction: a *covered* arm trained on all eight ring modes and a
*collapsed* arm trained on two of them, both evaluated against the full
mixture. This module trains those arms through the public ParticleGAN API
only -- ``get_recipe()`` (the current default recipe, unmodified apart from
the task's ``total_steps`` budget), ``particlegan.init`` deterministic
initialization, ``GANTrainer.step`` and ``GANTrainer.sample`` -- at protocol
seed 0. The host never touches the update; it only feeds real batches and
reads samples/critic values at checkpoints.

Task identity (owned here, identical across arms):
  * target law: 8 isotropic Gaussians on a radius-2 ring, std 0.25;
    the collapsed arm's training law is the same mixture restricted to
    modes ``COLLAPSED_MODES``;
  * architecture: G = MLP(z_dim -> 128 -> 128 -> 2, LeakyReLU 0.2),
    D = MLP(2 -> 128 -> 128 -> 1, LeakyReLU 0.2), deterministic init
    (G seed 0, D seed 1, R2 prior table);
  * RNG streams: trainer seed 0; data stream ``torch.Generator`` seed 0;
    fixed evaluation real set (seed 999) and fixed sampling stream (seed
    998) re-seeded at every checkpoint so checkpoints see identical latents.

Rendering (``render_arms_gif``) draws the actual API samples at every
checkpoint beside the full target, one panel per arm.
"""
from __future__ import annotations

import io
import math
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init

RING_MODES = 8
RING_RADIUS = 2.0
RING_STD = 0.25
COLLAPSED_MODES = (0, 1)
ALL_MODES = tuple(range(RING_MODES))
PROTOCOL_SEED = 0
HIDDEN = 128
EVAL_REAL_SEED = 999
EVAL_LATENT_SEED = 998


def ring_centers() -> torch.Tensor:
    ang = torch.arange(RING_MODES, dtype=torch.float64) * (2 * math.pi / RING_MODES)
    return torch.stack([RING_RADIUS * torch.cos(ang), RING_RADIUS * torch.sin(ang)], dim=1)


def sample_ring(n: int, modes: Sequence[int], generator: torch.Generator,
                device="cpu") -> torch.Tensor:
    """Draw ``n`` points from the ring restricted to ``modes`` on ``generator``'s device."""
    device = torch.device(device)
    centers = ring_centers().to(device=device, dtype=torch.float32)[list(modes)]
    pick = torch.randint(0, len(modes), (n,), generator=generator, device=device)
    return centers[pick] + RING_STD * torch.randn(n, 2, generator=generator, device=device)


def eval_real(n: int) -> torch.Tensor:
    """The fixed full-mixture evaluation set (CPU)."""
    return sample_ring(n, ALL_MODES, torch.Generator().manual_seed(EVAL_REAL_SEED))


def build_trainer(total_steps: int, device="cpu"):
    """Default public recipe with the task's architecture and deterministic init."""
    device = torch.device(device)
    recipe = get_recipe(total_steps=total_steps)
    g = nn.Sequential(nn.Linear(recipe.z_dim, HIDDEN), nn.LeakyReLU(0.2),
                      nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2),
                      nn.Linear(HIDDEN, 2)).to(device)
    d = nn.Sequential(nn.Linear(2, HIDDEN), nn.LeakyReLU(0.2),
                      nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2),
                      nn.Linear(HIDDEN, 1)).to(device)
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(d, seed=1)
    prior = init.deterministic_orthogonal_(recipe.make_prior().to(device))
    return recipe, GANTrainer(recipe, g, d, prior=prior, seed=PROTOCOL_SEED)


def train_ring_arm(
    modes: Sequence[int],
    total_steps: int,
    check_every: int,
    *,
    device="cpu",
    n_eval: int = 2048,
    observe: Optional[Callable[[dict], dict]] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> Dict[str, object]:
    """Train one arm and return its checkpoint observations.

    Each observation holds ``step``, the run's own critic loss averaged over
    the window since the previous checkpoint (``critic_loss``), the same
    critic's loss against the full mixture (``critic_loss_full``) and the
    live API samples ``fake`` (CPU, ``n_eval`` rows, fixed latent stream).
    ``observe(obs)`` may add metric fields; it runs on the observation dict.
    """
    device = torch.device(device)
    recipe, trainer = build_trainer(total_steps, device)
    data = torch.Generator(device=device).manual_seed(PROTOCOL_SEED)
    real_full = eval_real(n_eval)
    observations: List[dict] = []
    window: List[float] = []
    for _ in range(total_steps):
        stats = trainer.step(sample_ring(recipe.batch_size, modes, data, device))
        window.append(float(stats["loss_d"]))
        step = int(stats["step"])
        if step % check_every and step != total_steps:
            continue
        fake = trainer.sample(n_eval, generator=torch.Generator(device=device).manual_seed(EVAL_LATENT_SEED))
        with torch.no_grad():
            critic_full = trainer.loss.d_loss(trainer.D(real_full.to(device)), trainer.D(fake))
        obs = {
            "step": step,
            "critic_loss": sum(window) / len(window),
            "critic_loss_full": float(critic_full),
            "fake": fake.detach().float().cpu(),
        }
        window = []
        if observe is not None:
            obs.update(observe(obs))
        if not bool(torch.isfinite(obs["fake"]).all()):
            obs["nonfinite"] = True
        observations.append(obs)
        if progress is not None:
            progress({k: v for k, v in obs.items() if k != "fake"})
    return {"recipe": recipe.to_dict(), "observations": observations, "trainer": trainer}


def render_arms_gif(path, arms: Dict[str, List[dict]], caption: Callable[[str, dict], str],
                    title: str = "", n_points: int = 1024, duration_ms: int = 450) -> int:
    """Write a GIF with one panel per arm: full target (grey) vs actual API samples.

    ``arms`` maps arm name to its observation list (equal lengths). Returns
    the frame count.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    target = eval_real(n_points).numpy()
    names = list(arms)
    count = len(arms[names[0]])
    if any(len(arms[name]) != count for name in names):
        raise ValueError("arms must have equal checkpoint counts")
    lim = RING_RADIUS + 4 * RING_STD + 0.3
    frames = []
    for i in range(count):
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6), dpi=72)
        axes = [axes] if len(names) == 1 else list(axes)
        for ax, name in zip(axes, names):
            obs = arms[name][i]
            fake = obs["fake"][:n_points].numpy()
            ax.scatter(target[:, 0], target[:, 1], s=3, c="#bbbbbb", label="target (full ring)")
            ax.scatter(fake[:, 0], fake[:, 1], s=3, c="#d6336c", alpha=0.6, label="GANTrainer.sample")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(caption(name, obs), fontsize=8)
        axes[0].legend(loc="lower left", fontsize=6, markerscale=3)
        if title:
            fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93) if title else None)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms,
                   loop=0, optimize=True)
    return len(frames)
