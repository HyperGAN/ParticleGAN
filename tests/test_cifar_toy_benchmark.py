"""Contract and semantic control tests for ``benchmarks.cifar_toy``."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.cifar_toy.data import compute_class_centroids, make_synthetic_cifar_dataset
from benchmarks.cifar_toy.metrics import compute_class_coverage, compute_fid, compute_kid, feature_moments
from benchmarks.cifar_toy.models import (
    ResNet18FeatureCritic32,
    ResNet34FeatureEvaluator,
    TinyTransformerGenerator32,
    count_matrix_conv_params,
    count_trainable_params,
)
from benchmarks.cifar_toy.runner import BenchmarkConfig, run_benchmark


def test_architecture_exact_param_and_matrix_counts():
    gen = TinyTransformerGenerator32(z_dim=64)
    disc = ResNet18FeatureCritic32(weights_path=None, backbone_mode="random")

    assert count_matrix_conv_params(gen) == 16
    assert count_trainable_params(gen) == 2_489_731

    assert count_matrix_conv_params(disc) == 9
    assert count_trainable_params(disc) == 171_779

    # Forward shapes match 4D RGB 32x32 -> scalar logit.
    z = torch.randn(4, 64)
    img = gen(z)
    assert img.shape == (4, 3, 32, 32)
    assert float(img.detach().min()) >= -1.0 and float(img.detach().max()) <= 1.0
    logits = disc(img)
    assert logits.shape == (4, 1)


def test_frozen_backbone_stays_eval_and_unmodified():
    disc = ResNet18FeatureCritic32(weights_path=None, backbone_mode="random")
    bn_mean_before = disc.backbone.bn1.running_mean.clone()
    bn_var_before = disc.backbone.bn1.running_var.clone()
    conv1_before = disc.backbone.conv1.weight.clone()

    # GANTrainer calls D.train(); backbone must stay frozen in eval mode.
    disc.train(True)
    assert disc.backbone.training is False
    assert all(not p.requires_grad for p in disc.backbone.parameters())

    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    loss = disc(x).sum()
    loss.backward()

    # Gradients flow through frozen backbone to input x, without modifying backbone stats or weights.
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0.0
    assert torch.equal(disc.backbone.bn1.running_mean, bn_mean_before)
    assert torch.equal(disc.backbone.bn1.running_var, bn_var_before)
    assert torch.equal(disc.backbone.conv1.weight, conv1_before)
    assert all(p.grad is None for p in disc.backbone.parameters())


def test_scorer_oracle_and_destructive_controls():
    ds = make_synthetic_cifar_dataset(num_samples=400, seed=0)
    ev = ResNet34FeatureEvaluator(weights_path=None, backbone_mode="random")

    imgs = ds.to_float_tensor(torch.arange(400))
    feats = ev.extract(imgs, batch_size=64)
    mu_r, cov_r = feature_moments(feats)
    centroids = compute_class_centroids(feats, ds.labels, num_classes=10)

    # Oracle control: real vs real => FID ~ 0, KID ~ 0, 10/10 coverage.
    oracle_fid = compute_fid(mu_r, cov_r, mu_r, cov_r)
    oracle_kid, _ = compute_kid(feats[:200], feats[200:], subset_size=100, num_subsets=5, seed=0)
    oracle_cov = compute_class_coverage(feats, centroids)
    assert oracle_fid < 1e-4
    assert abs(oracle_kid) < 1e-3
    assert oracle_cov["class_coverage"] == 10.0

    # Destructive control: constant collapsed image yields large FID/KID and collapsed coverage (1/10).
    collapsed = torch.zeros_like(imgs[:200])
    bad_feats = ev.extract(collapsed, batch_size=64)
    mu_b, cov_b = feature_moments(bad_feats)
    bad_fid = compute_fid(mu_r, cov_r, mu_b, cov_b)
    bad_kid, _ = compute_kid(feats[:200], bad_feats, subset_size=100, num_subsets=5, seed=0)
    bad_cov = compute_class_coverage(bad_feats, centroids)
    assert bad_fid > oracle_fid + 0.25
    assert bad_kid > oracle_kid + 1e-3
    assert bad_cov["class_coverage"] == 1.0


@pytest.mark.parametrize("recipes", [("empf", "unified_v11")])
def test_matched_initial_state_and_batch_sequence_across_trainers(tmp_path: Path, recipes):
    receipts = []
    for recipe in recipes:
        cfg = BenchmarkConfig(
            recipe=recipe,
            num_particles=16,
            batch_size=8,
            steps=2,
            eval_every=2,
            eval_samples=16,
            eval_ref_samples=64,
            eval_critic_samples=8,
            seed=0,
            dataset="synthetic",
            backbone_mode="random",
            device="cpu",
            run_tag=f"smoke_{recipe}",
        )
        receipts.append(run_benchmark(cfg, tmp_path))

    r0, r1 = receipts
    assert r0["status"] == "COMPLETE" and r1["status"] == "COMPLETE"
    # Identical initial weights, initial prior, and batch sequence across trainers.
    for key in ("initial_g_sha256", "initial_d_sha256", "initial_prior_sha256", "batch_sequence_digest"):
        assert r0["provenance"][key] == r1["provenance"][key], f"Mismatch on {key}"
    # Identical step-0 FID/logits before any recipe updates.
    assert r0["history"][0]["feature_fid"] == pytest.approx(r1["history"][0]["feature_fid"], abs=1e-4)
    assert r0["history"][0]["critic_margin"] == pytest.approx(r1["history"][0]["critic_margin"], abs=1e-4)
