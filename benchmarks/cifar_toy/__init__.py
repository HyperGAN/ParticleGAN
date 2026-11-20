"""Fast, reproducible CIFAR-style 32x32 Transformer-G / ResNet18-feature-D benchmark.

Matches the structural properties of HyperGAN's ``cifar10-tiny-transformer``
(4D ``(B, 3, 32, 32)`` RGB tensor with ``dim = 3072 > 1024``, 2-block 64-token
Transformer generator with 16 matrix/conv parameters vs. frozen ImageNet
ResNet-18 64px critic with 3 trainable feature heads and 9 matrix/conv
parameters, MoG particle prior in ``z_dim = 64``) while executing purely
through ParticleGAN's public ``Recipe`` / ``MoGParticlePrior`` / ``GANTrainer``
API under protocol seed 0.
"""
from .data import (
    BatchSequenceStream,
    CIFAR10Dataset,
    discover_cifar10_root,
    load_cifar10_dataset,
    make_synthetic_cifar_dataset,
)
from .metrics import (
    compute_class_coverage,
    compute_critic_diagnostics,
    compute_fid,
    compute_kid,
    compute_pixel_diversity_ratio,
)
from .models import (
    DEFAULT_RESNET18_WEIGHTS,
    DEFAULT_RESNET34_WEIGHTS,
    RESNET18_SHA256,
    RESNET34_SHA256,
    ResNet18FeatureCritic32,
    ResNet34FeatureEvaluator,
    TinyTransformerGenerator32,
    count_matrix_conv_params,
    count_trainable_params,
    parameter_state_digest,
)
from .runner import (
    DEFAULT_MOG_SIGMA,
    BenchmarkConfig,
    format_leaderboard_markdown,
    run_benchmark,
    summarize_run_directory,
)

__all__ = [
    "BatchSequenceStream",
    "BenchmarkConfig",
    "CIFAR10Dataset",
    "DEFAULT_MOG_SIGMA",
    "DEFAULT_RESNET18_WEIGHTS",
    "DEFAULT_RESNET34_WEIGHTS",
    "RESNET18_SHA256",
    "RESNET34_SHA256",
    "ResNet18FeatureCritic32",
    "ResNet34FeatureEvaluator",
    "TinyTransformerGenerator32",
    "compute_class_coverage",
    "compute_critic_diagnostics",
    "compute_fid",
    "compute_kid",
    "compute_pixel_diversity_ratio",
    "count_matrix_conv_params",
    "count_trainable_params",
    "discover_cifar10_root",
    "format_leaderboard_markdown",
    "load_cifar10_dataset",
    "make_synthetic_cifar_dataset",
    "parameter_state_digest",
    "run_benchmark",
    "summarize_run_directory",
]
