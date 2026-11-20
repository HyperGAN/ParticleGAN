"""Deterministic CIFAR-10 and synthetic 10-class 32x32 RGB data streams.

Zero ``torchvision`` dependency: loads ``cifar-10-batches-py`` directly from
local cache using Python's standard ``pickle`` module. Provides isolated
checkpointable CPU batch streams for critic and generator target draws plus an
exact SHA-256 batch-sequence digest so every trainer sees identical batches.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CIFAR10Dataset:
    """In-memory uint8 ``(N, 3, 32, 32)`` dataset + integer labels ``(N,)``."""

    images_uint8: torch.Tensor
    labels: torch.Tensor
    source_name: str
    source_path: str
    num_classes: int = 10

    @property
    def size(self) -> int:
        return int(self.images_uint8.shape[0])

    def to_float_tensor(self, indices: torch.Tensor, *, flip_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert indexed uint8 images to float32 in ``[-1, 1]`` on CPU."""
        imgs = self.images_uint8[indices].to(dtype=torch.float32).div_(127.5).sub_(1.0)
        if flip_mask is not None and bool(flip_mask.any()):
            imgs = torch.where(flip_mask.view(-1, 1, 1, 1), imgs.flip(-1), imgs)
        return imgs


_CANDIDATE_CIFAR_ROOTS = (
    Path("/nvme/dev/hypergan/HyperGAN/data"),
    Path.home() / ".hypergan" / "cache" / "cifar10",
    Path("data"),
)


def discover_cifar10_root(explicit_root: Optional[Path] = None) -> Optional[Path]:
    """Locate a local directory containing ``cifar-10-batches-py/data_batch_1``."""
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root).expanduser())
    env_root = os.environ.get("PARTICLEGAN_CIFAR10_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(_CANDIDATE_CIFAR_ROOTS)

    for root in candidates:
        if (root / "data_batch_1").is_file():
            return root
        sub = root / "cifar-10-batches-py"
        if (sub / "data_batch_1").is_file():
            return sub
    return None


def load_cifar10_dataset(
    data_root: Optional[Path] = None,
    *,
    split: str = "train",
) -> CIFAR10Dataset:
    """Load real CIFAR-10 ``train`` (50k) or ``test`` (10k) images as uint8 NCHW."""
    batch_dir = discover_cifar10_root(data_root)
    if batch_dir is None:
        raise FileNotFoundError(
            "CIFAR-10 cache not found. Pass --data-root pointing to cifar-10-batches-py "
            "or use --dataset synthetic for the hermetic synthetic cohort."
        )
    if split == "train":
        files = [batch_dir / f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        files = [batch_dir / "test_batch"]
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    chunks = []
    labels = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing CIFAR-10 batch file: {path}")
        with open(path, "rb") as fh:
            entry = pickle.load(fh, encoding="latin1")
        data = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        chunks.append(data)
        labels.extend(int(y) for y in entry["labels"])

    images_np = np.concatenate(chunks, axis=0)
    return CIFAR10Dataset(
        images_uint8=torch.from_numpy(images_np.copy()),
        labels=torch.tensor(labels, dtype=torch.long),
        source_name=f"cifar10_{split}",
        source_path=str(batch_dir.resolve()),
        num_classes=10,
    )


def make_synthetic_cifar_dataset(
    num_samples: int = 2000,
    *,
    seed: int = 0,
    num_classes: int = 10,
) -> CIFAR10Dataset:
    """Deterministic 10-class structured 32x32 RGB image cohort for hermetic tests.

    Each class has a distinct color palette + geometric spatial pattern
    (vertical/horizontal/diagonal bars, quadrants, center disk, frame, checker,
    radial gradient) with small deterministic per-sample jitter so FID/KID and
    class coverage are well-conditioned.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + 7919)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 32),
        torch.linspace(-1.0, 1.0, 32),
        indexing="ij",
    )
    palettes = torch.tensor(
        [
            [0.90, 0.15, 0.15],
            [0.15, 0.85, 0.20],
            [0.15, 0.30, 0.92],
            [0.92, 0.85, 0.15],
            [0.85, 0.20, 0.85],
            [0.15, 0.85, 0.88],
            [0.95, 0.52, 0.15],
            [0.55, 0.25, 0.85],
            [0.25, 0.65, 0.45],
            [0.82, 0.82, 0.85],
        ],
        dtype=torch.float32,
    )
    imgs = torch.empty((num_samples, 3, 32, 32), dtype=torch.float32)
    labels = torch.empty((num_samples,), dtype=torch.long)

    for i in range(num_samples):
        cls = i % num_classes
        labels[i] = cls
        shift_x = float(torch.empty(()).uniform_(-0.22, 0.22, generator=gen).item())
        shift_y = float(torch.empty(()).uniform_(-0.22, 0.22, generator=gen).item())
        x = xx - shift_x
        y = yy - shift_y
        if cls == 0:
            mask = (x.abs() < 0.35).float()
        elif cls == 1:
            mask = (y.abs() < 0.35).float()
        elif cls == 2:
            mask = ((x + y).abs() < 0.40).float()
        elif cls == 3:
            mask = ((x - y).abs() < 0.40).float()
        elif cls == 4:
            mask = ((x * x + y * y) < 0.28).float()
        elif cls == 5:
            r2 = x * x + y * y
            mask = ((r2 > 0.14) & (r2 < 0.48)).float()
        elif cls == 6:
            mask = ((x > 0.0) & (y > 0.0)).float()
        elif cls == 7:
            mask = ((x < 0.0) & (y < 0.0)).float()
        elif cls == 8:
            mask = (torch.sin(6.0 * x) * torch.sin(6.0 * y) > 0.0).float()
        else:
            mask = ((x.abs() > 0.55) | (y.abs() > 0.55)).float()

        fg = palettes[cls % len(palettes)].view(3, 1, 1)
        bg = (1.0 - fg) * 0.25 + 0.12
        noise = torch.randn((3, 32, 32), generator=gen) * 0.05
        img = (bg * (1.0 - mask.unsqueeze(0)) + fg * mask.unsqueeze(0) + noise).clamp(0.0, 1.0)
        imgs[i] = img

    images_uint8 = imgs.mul(255.0).round().clamp(0, 255).to(torch.uint8)
    return CIFAR10Dataset(
        images_uint8=images_uint8,
        labels=labels,
        source_name=f"synthetic_cifar10_n{num_samples}_seed{seed}",
        source_path="in_memory",
        num_classes=num_classes,
    )


class BatchSequenceStream:
    """Isolated deterministic critic/generator target batch streams with exact digest.

    Critic and generator target batches use separate CPU generators seeded from
    the protocol seed so every compared trainer receives the exact same ordered
    sequence of ``(real_d, real_g)`` batches.
    """

    def __init__(
        self,
        dataset: CIFAR10Dataset,
        batch_size: int,
        *,
        seed: int = 0,
        horizontal_flip: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.horizontal_flip = bool(horizontal_flip)
        self.critic_gen = torch.Generator(device="cpu").manual_seed(seed + 101)
        self.gen_gen = torch.Generator(device="cpu").manual_seed(seed + 202)
        self._hasher = hashlib.sha256()
        self.steps_drawn = 0

    def _draw_one(self, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        idx = torch.randint(0, self.dataset.size, (self.batch_size,), generator=rng)
        flips = None
        if self.horizontal_flip:
            flips = torch.randint(0, 2, (self.batch_size,), generator=rng, dtype=torch.bool)
        batch = self.dataset.to_float_tensor(idx, flip_mask=flips)
        return batch, idx, flips

    def next_pair(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        real_d, idx_d, flip_d = self._draw_one(self.critic_gen)
        real_g, idx_g, flip_g = self._draw_one(self.gen_gen)
        self._hasher.update(idx_d.numpy().tobytes())
        self._hasher.update(idx_g.numpy().tobytes())
        if flip_d is not None and flip_g is not None:
            self._hasher.update(flip_d.numpy().tobytes())
            self._hasher.update(flip_g.numpy().tobytes())
        self.steps_drawn += 1
        return real_d.to(device=device, non_blocking=False), real_g.to(device=device, non_blocking=False)

    @property
    def digest(self) -> str:
        return self._hasher.hexdigest()

    def state_dict(self) -> dict:
        return {
            "critic_gen": self.critic_gen.get_state(),
            "gen_gen": self.gen_gen.get_state(),
            "steps_drawn": self.steps_drawn,
        }


def select_eval_reference_subset(
    dataset: CIFAR10Dataset,
    count: int,
    *,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically select a balanced reference subset ``(images_float, labels)``."""
    n = min(int(count), dataset.size)
    rng = torch.Generator(device="cpu").manual_seed(seed + 303)
    perm = torch.randperm(dataset.size, generator=rng)[:n]
    imgs = dataset.to_float_tensor(perm, flip_mask=None)
    lbls = dataset.labels[perm].clone()
    return imgs, lbls


def compute_class_centroids(features: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Compute per-class mean feature vectors ``(num_classes, D)``."""
    feat_f = features.float()
    centroids = torch.zeros((num_classes, feat_f.shape[1]), dtype=torch.float32)
    for c in range(num_classes):
        mask = labels == c
        if bool(mask.any()):
            centroids[c] = feat_f[mask].mean(dim=0)
    return centroids


def sample_indices_for_eval(num_samples: int, batch_size: int) -> Sequence[int]:
    sizes = []
    rem = int(num_samples)
    while rem > 0:
        b = min(batch_size, rem)
        sizes.append(b)
        rem -= b
    return sizes
