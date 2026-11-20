"""Pure-PyTorch architectures for the CIFAR-style toy benchmark.

Zero ``torchvision`` dependency: standard PyTorch ResNet-18 / ResNet-34
``BasicBlock`` backbones load the official SHA-256-verified PyTorch hub
checkpoints directly.

Generator: ``TinyTransformerGenerator32`` (16 matrix/conv params, 2,489,731
trainable parameters) — exact structural match to HyperGAN's
``tiny-transformer-generator-32``.

Critic: ``ResNet18FeatureCritic32`` (9 trainable matrix/conv params, 171,779
trainable parameters on top of frozen ImageNet ResNet-18 ``layer1/2/3`` at
64px with fixed midgray context) — exact structural match to HyperGAN's
``resnet18-features-discriminator-32``.

Evaluator: ``ResNet34FeatureEvaluator`` — independent frozen ImageNet ResNet-34
``avgpool`` 512-dim feature extractor for Feature-FID, Feature-KID, and class
centroid mode coverage.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET34_SHA256 = "b627a593f4b504425f6d6c373712e796be8773f6cb3c5910b09070727651153e"

DEFAULT_RESNET18_WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
DEFAULT_RESNET34_WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet34-b627a593.pth"
DEFAULT_INCEPTION_WEIGHTS = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "weights-inception-2015-12-05-6726825d.pth"
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_matrix_conv_params(module: nn.Module) -> int:
    """Count trainable parameters with ``ndim >= 2`` (matrices and conv kernels)."""
    return sum(1 for p in module.parameters() if p.requires_grad and p.ndim >= 2)


def parameter_state_digest(module: nn.Module) -> str:
    """Deterministic SHA-256 digest of all trainable parameter tensors (CPU float32 bytes)."""
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        if not p.requires_grad:
            continue
        arr = p.detach().cpu().contiguous().float().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNetBackbone(nn.Module):
    """Standard torchvision-compatible ResNet-18 / ResNet-34 backbone without torchvision."""

    def __init__(self, layers: Sequence[int], *, include_layer4: bool = False) -> None:
        super().__init__()
        self.inplanes = 64
        self.include_layer4 = bool(include_layer4)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        if self.include_layer4:
            self.layer4 = self._make_layer(512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def load_torchvision_state_dict(self, state_dict: dict) -> None:
        own_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in own_keys}
        missing = {k for k in (own_keys - set(filtered.keys())) if not k.endswith(".num_batches_tracked")}
        if missing:
            raise ValueError(f"ResNet checkpoint missing expected keys: {sorted(missing)[:8]}")
        self.load_state_dict(filtered, strict=False)

    def forward_features_123(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3

    def forward_avgpool(self, x: torch.Tensor) -> torch.Tensor:
        if not self.include_layer4:
            raise RuntimeError("forward_avgpool requires include_layer4=True")
        _, _, f3 = self.forward_features_123(x)
        f4 = self.layer4(f3)
        return torch.flatten(self.avgpool(f4), 1)


def _load_verified_resnet_weights(
    weights_path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> Tuple[dict, str]:
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained ResNet weights not found: {path}")
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        # Accept prefix match if caller passes short hash, else strict mismatch error.
        if not actual_sha.startswith(expected_sha256):
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha}"
            )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd, actual_sha


class _TransformerBlock(nn.Module):
    """Pre-LayerNorm multi-head self-attention + GELU FFN with separate Q/K/V/Out Linears."""

    def __init__(self, d_model: int = 256, num_heads: int = 4, ffn_dim: int = 512) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        # Explicit scaled dot-product attention for deterministic CPU/CUDA compatibility.
        scale = 1.0 / math.sqrt(float(self.head_dim))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, c)
        x = x + self.out_proj(out)

        h2 = self.ffn_norm(x)
        x = x + self.fc2(F.gelu(self.fc1(h2)))
        return x


class TinyTransformerGenerator32(nn.Module):
    """2-block 64-token Transformer generator -> 32x32 RGB in [-1, 1].

    Exact structural match to HyperGAN's ``cifar10-tiny-transformer``:
    16 matrix/conv parameters (1 input projection + 2*(4 attn + 2 FFN) + 3 convs),
    2,489,731 trainable parameters total.
    """

    def __init__(self, z_dim: int = 64, d_model: int = 256, num_heads: int = 4, ffn_dim: int = 512) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(z_dim, 64 * d_model, bias=True)
        self.block0 = _TransformerBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)
        self.block1 = _TransformerBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)
        self.conv16 = nn.Conv2d(d_model, 128, kernel_size=3, padding=1, bias=True)
        self.conv32 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True)
        self.rgb = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        h = self.input_projection(z).view(b, 64, self.d_model)
        h = self.block0(h)
        h = self.block1(h)
        h = h.transpose(1, 2).contiguous().view(b, self.d_model, 8, 8)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = F.leaky_relu(self.conv16(h), negative_slope=0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = F.leaky_relu(self.conv32(h), negative_slope=0.2)
        return torch.tanh(self.rgb(h))


class _ResNetFeatureHead(nn.Module):
    """Single multi-scale ResNet feature head (3 matrix/conv params)."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(8, hidden_channels)
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.score = nn.Linear(hidden_channels * 4 * 4, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.norm(self.project(x)), negative_slope=0.2)
        h = F.leaky_relu(self.conv(h), negative_slope=0.2)
        h = F.adaptive_avg_pool2d(h, (4, 4)).flatten(1)
        return self.score(h)


class ResNet18FeatureCritic32(nn.Module):
    """Frozen ImageNet ResNet-18 64px feature critic with 3 trainable feature heads.

    Exact structural match to HyperGAN's ``resnet18-features-discriminator-32``:
    - Bilinear 32->64 resize + ImageNet normalization on ``[x, zeros_like(x)]``.
    - Frozen ResNet-18 ``layer1`` (64ch), ``layer2`` (128ch), ``layer3`` (256ch).
    - 3 trainable feature heads concatenating candidate + midgray-context features.
    - 9 trainable matrix/conv parameters, 171,779 trainable parameters total.
    - ``train()`` keeps the frozen backbone strictly in ``eval()`` mode so
      ``GANTrainer`` calling ``D.train()`` never mutates BatchNorm statistics.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = DEFAULT_RESNET18_WEIGHTS,
        *,
        backbone_mode: str = "pretrained",
        backbone_seed: int = 1001,
    ) -> None:
        super().__init__()
        if backbone_mode not in ("pretrained", "random"):
            raise ValueError(f"backbone_mode must be 'pretrained' or 'random', got {backbone_mode!r}")
        self.backbone_mode = backbone_mode
        self.backbone_sha256: Optional[str] = None
        self.backbone = _ResNetBackbone((2, 2, 2, 2), include_layer4=False)
        if backbone_mode == "pretrained":
            if weights_path is None:
                raise ValueError("weights_path is required when backbone_mode='pretrained'")
            sd, actual_sha = _load_verified_resnet_weights(Path(weights_path), expected_sha256=RESNET18_SHA256)
            self.backbone.load_torchvision_state_dict(sd)
            self.backbone_sha256 = actual_sha
        else:
            import particlegan as pg

            pg.init.deterministic_orthogonal_(self.backbone, seed=backbone_seed, strict=True)
            self.backbone_sha256 = f"deterministic_random_seed_{backbone_seed}"

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.register_buffer("_img_mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("_img_std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))

        self.head1 = _ResNetFeatureHead(64 * 2, 64)
        self.head2 = _ResNetFeatureHead(128 * 2, 64)
        self.head3 = _ResNetFeatureHead(256 * 2, 64)
        self._inv_sqrt3 = 1.0 / math.sqrt(3.0)

    def train(self, mode: bool = True) -> "ResNet18FeatureCritic32":
        super().train(mode)
        # Frozen ImageNet ResNet-18 must always remain in eval() mode (fixed BN stats).
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.zeros_like(x)
        feature_batch = torch.cat([x, context], dim=0)
        resized = F.interpolate(feature_batch, scale_factor=2, mode="bilinear", align_corners=False)
        normalized = (resized * 0.5 + 0.5 - self._img_mean.to(resized.dtype)) / self._img_std.to(resized.dtype)
        f1, f2, f3 = self.backbone.forward_features_123(normalized)
        cand1, ctx1 = f1.chunk(2, dim=0)
        cand2, ctx2 = f2.chunk(2, dim=0)
        cand3, ctx3 = f3.chunk(2, dim=0)
        s1 = self.head1(torch.cat([cand1, ctx1], dim=1))
        s2 = self.head2(torch.cat([cand2, ctx2], dim=1))
        s3 = self.head3(torch.cat([cand3, ctx3], dim=1))
        return (s1 + s2 + s3) * self._inv_sqrt3


class ResNet34FeatureEvaluator(nn.Module):
    """Independent frozen ImageNet ResNet-34 512-dim feature extractor for FID/KID.

    Uses ResNet-34 (different architecture depth and weights from the critic's
    frozen ResNet-18) so Feature-FID/KID and class-centroid coverage cannot be
    gamed by critic-backbone shortcuts.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = DEFAULT_RESNET34_WEIGHTS,
        *,
        backbone_mode: str = "pretrained",
        backbone_seed: int = 2002,
    ) -> None:
        super().__init__()
        if backbone_mode not in ("pretrained", "random"):
            raise ValueError(f"backbone_mode must be 'pretrained' or 'random', got {backbone_mode!r}")
        self.backbone_mode = backbone_mode
        self.backbone_sha256: Optional[str] = None
        self.backbone = _ResNetBackbone((3, 4, 6, 3), include_layer4=True)
        if backbone_mode == "pretrained":
            if weights_path is None:
                raise ValueError("weights_path is required when backbone_mode='pretrained'")
            sd, actual_sha = _load_verified_resnet_weights(Path(weights_path), expected_sha256=None)
            self.backbone.load_torchvision_state_dict(sd)
            self.backbone_sha256 = actual_sha
        else:
            import particlegan as pg

            pg.init.deterministic_orthogonal_(self.backbone, seed=backbone_seed, strict=True)
            self.backbone_sha256 = f"deterministic_random_seed_{backbone_seed}"

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("_img_mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("_img_std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))

    @torch.no_grad()
    def extract(self, images: torch.Tensor, *, batch_size: int = 256) -> torch.Tensor:
        """Extract L2-unnormalized 512-dim float32 features from images in [-1, 1]."""
        self.eval()
        device = self._img_mean.device
        feats = []
        n = int(images.shape[0])
        for start in range(0, n, batch_size):
            chunk = images[start : start + batch_size].to(device=device, dtype=torch.float32)
            resized = F.interpolate(chunk, size=(64, 64), mode="bilinear", align_corners=False)
            normed = (resized * 0.5 + 0.5 - self._img_mean) / self._img_std
            feats.append(self.backbone.forward_avgpool(normed).cpu())
        return torch.cat(feats, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(x.float(), size=(64, 64), mode="bilinear", align_corners=False)
        normed = (resized * 0.5 + 0.5 - self._img_mean) / self._img_std
        return self.backbone.forward_avgpool(normed)


def iter_trainable_modules(module: nn.Module) -> Iterable[nn.Module]:
    return (m for m in module.modules() if any(p.requires_grad for p in m.parameters(recurse=False)))
