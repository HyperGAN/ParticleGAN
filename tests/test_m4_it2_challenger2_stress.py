"""Milestone 4 Iteration 2 Challenger 2 Adversarial Stress and Fail-Closed Test Suite.

Adversarially challenges:
1. MomentTether regularizer edge cases & parameter boundaries:
   - Negative, NaN, Inf weight rejection
   - Zero, negative, NaN, Inf eps rejection
   - Unbound or partially bound prior reference rejection
   - Malformed row tensor shapes (1D, 3D, 0 latent dim, dimension mismatch)
   - Zero-weight bypass and empty-row handling
   - Incomplete reference buffer rejection in capture_initial_moments
   - Reference immutability across subsequent prior updates or re-bindings
2. Recipe fail-closed boundary enforcement:
   - Negative, NaN, Inf prior_reg rejection
   - Non-positive or non-finite prior_reg_eps rejection
   - Unsupported prior_regularizer identifier rejection
   - make_prior_regularizer unexpected keyword argument rejection
3. Authentic physical moment regularizer properties:
   - Pure centroid & covariance tethering vs rigid coordinate anchoring
   - Exact permutation invariance under arbitrary row shuffles
   - Autograd gradient correctness and finite numerical stability
   - Zero gradient when moments match reference despite particle rearrangement
4. Recipe and candidate integrity invariants:
   - atlas-particle-normal-v4 canonical frozen overrides
   - recipe-prior-refactor-bcap-pure-v1 configuration specifications
   - Candidate digest invariance across default additions
"""
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from particlegan import get_recipe
from particlegan.recipes import Recipe
from particlegan.moment_tether import (
    MomentTetherRegularizer,
    capture_initial_moments,
    REFERENCE_NAMES,
)
from particlegan.recipe_compat import without_default_additions


ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. MomentTether Fail-Closed Parameter & Shape Stress Tests
# ==============================================================================

def test_moment_tether_rejects_negative_and_nonfinite_weights():
    """Verify that MomentTetherRegularizer rejects negative, NaN, and Inf weights."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.randn(10, 2))
    capture_initial_moments(prior)

    for bad_w in [-1.0, -1e-6, float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="weight must be nonnegative and eps positive, both finite"):
            MomentTetherRegularizer(prior, weight=bad_w)


def test_moment_tether_rejects_nonpositive_and_nonfinite_eps():
    """Verify that MomentTetherRegularizer rejects <= 0, NaN, and Inf eps."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.randn(10, 2))
    capture_initial_moments(prior)

    for bad_eps in [0.0, -0.01, -1.0, float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="weight must be nonnegative and eps positive, both finite"):
            MomentTetherRegularizer(prior, eps=bad_eps)


def test_moment_tether_rejects_unbound_and_partially_bound_priors():
    """Verify that priors without captured moment references fail closed."""
    # Unbound module
    raw_prior = nn.Module()
    raw_prior.z = nn.Parameter(torch.randn(10, 2))
    with pytest.raises(ValueError, match="moment_tether requires initialized priors"):
        MomentTetherRegularizer(raw_prior)

    # Empty tuple of references
    with pytest.raises(ValueError, match="moment_tether requires initialized priors"):
        MomentTetherRegularizer(())

    # Partially bound module (e.g. missing count)
    part_prior = nn.Module()
    part_prior.z = nn.Parameter(torch.randn(10, 2))
    part_prior.register_buffer(REFERENCE_NAMES[0], torch.zeros(2))
    part_prior.register_buffer(REFERENCE_NAMES[1], torch.eye(2))
    with pytest.raises(ValueError, match="moment_tether requires initialized priors"):
        MomentTetherRegularizer(part_prior)


def test_capture_initial_moments_rejects_partially_present_buffers():
    """Verify capture_initial_moments raises explicit ValueError on incomplete buffers."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.randn(10, 2))
    prior.register_buffer(REFERENCE_NAMES[0], torch.zeros(2))
    with pytest.raises(ValueError, match="incomplete moment_tether initial reference"):
        capture_initial_moments(prior)


def test_capture_initial_moments_idempotence():
    """Verify calling capture_initial_moments repeatedly does not overwrite references."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    capture_initial_moments(prior)
    orig_mean = prior.moment_tether_mean.clone()

    # Mutate z and call capture again
    prior.z.data.add_(100.0)
    capture_initial_moments(prior)
    assert torch.equal(prior.moment_tether_mean, orig_mean)


def test_moment_tether_rejects_invalid_row_shapes():
    """Verify input shape enforcement for particle row batches."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.randn(10, 2))
    capture_initial_moments(prior)
    loss = MomentTetherRegularizer(prior, weight=1.0)

    # 1D rows
    with pytest.raises(ValueError, match=r"particles must have shape \[batch, positive latent dimension\]"):
        loss(torch.randn(10))

    # 3D rows
    with pytest.raises(ValueError, match=r"particles must have shape \[batch, positive latent dimension\]"):
        loss(torch.randn(5, 10, 2))

    # Zero latent dimension (N, 0)
    with pytest.raises(ValueError, match=r"particles must have shape \[batch, positive latent dimension\]"):
        loss(torch.zeros(10, 0))

    # Dimension mismatch with reference
    with pytest.raises(ValueError, match="moment_tether rows and initial reference dimensions differ"):
        loss(torch.randn(10, 3))


def test_moment_tether_zero_weight_and_empty_rows_safety():
    """Verify zero weight and empty rows short-circuit safely without autograd breakage."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.randn(10, 2))
    capture_initial_moments(prior)

    # Weight == 0 returns zero scalar linked to autograd graph
    rows = torch.randn(10, 2, requires_grad=True)
    loss_zero = MomentTetherRegularizer(prior, weight=0.0)(rows)
    assert loss_zero.item() == 0.0
    loss_zero.backward()
    assert rows.grad is not None and torch.equal(rows.grad, torch.zeros_like(rows))

    # Empty batch [0, 2] returns 0.0
    empty_rows = torch.zeros(0, 2, requires_grad=True)
    loss_empty = MomentTetherRegularizer(prior, weight=1.0)(empty_rows)
    assert loss_empty.item() == 0.0


# ==============================================================================
# 2. Recipe Fail-Closed Boundary Tests
# ==============================================================================

@pytest.mark.parametrize("bad_val", [-1.0, -1e-4, float("nan"), float("inf"), float("-inf")])
def test_recipe_rejects_invalid_prior_reg(bad_val):
    """Verify Recipe rejects negative and non-finite prior_reg."""
    with pytest.raises(ValueError, match="prior_reg must be finite and nonnegative"):
        Recipe.from_dict({"prior_reg": bad_val})


@pytest.mark.parametrize("bad_eps", [0.0, -0.01, -1.0, float("nan"), float("inf"), float("-inf")])
def test_recipe_rejects_invalid_prior_reg_eps(bad_eps):
    """Verify Recipe rejects nonpositive and non-finite prior_reg_eps."""
    with pytest.raises(ValueError):
        Recipe.from_dict({"prior_reg_eps": bad_eps})


def test_recipe_rejects_unsupported_prior_regularizer():
    """Verify Recipe rejects unknown prior_regularizer strings."""
    with pytest.raises(ValueError, match="prior_regularizer must be vicreg, none, coulomb, sigreg, moment_tether, initial_mmd, sbnp or empf"):
        Recipe.from_dict({"prior_regularizer": "unsupported_regularizer"})

    with pytest.raises(ValueError, match="prior_regularizer must be vicreg, none, coulomb, sigreg, moment_tether, initial_mmd, sbnp or empf"):
        Recipe.from_dict({"prior_regularizer": ""})


def test_recipe_rejects_unsupported_moment_tether_kwargs():
    """Verify Recipe.make_prior_regularizer rejects unexpected kwargs for moment_tether."""
    prior = nn.Module()
    prior.z = nn.Parameter(torch.randn(10, 2))
    capture_initial_moments(prior)

    recipe = get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=0.01)
    with pytest.raises(TypeError, match="unsupported moment_tether options"):
        recipe.make_prior_regularizer(reference=prior, bogus_option=123)


# ==============================================================================
# 3. Authentic Physical Moment Regularizer Properties
# ==============================================================================

def test_moment_tether_is_strictly_permutation_invariant():
    """Verify that rearranging particles in latent space incurs ZERO penalty difference."""
    torch.manual_seed(42)
    initial = torch.randn(20, 4, dtype=torch.float64)
    prior = nn.Module()
    prior.z = nn.Parameter(initial.clone())
    capture_initial_moments(prior)

    loss_fn = MomentTetherRegularizer(prior, weight=1.0, eps=1e-4)

    # Initial self-loss is 0.0
    assert loss_fn(initial).item() == 0.0

    # Permutation of rows has exactly 0.0 loss
    perm1 = initial[torch.randperm(20)]
    perm2 = initial[torch.randperm(20)]
    assert loss_fn(perm1).item() < 1e-15
    assert loss_fn(perm2).item() < 1e-15


def test_moment_tether_allows_free_clustering_at_constant_moments():
    """Verify that particles can form tight multimodal clusters without tether penalty
    as long as the ensemble's mean and covariance match the reference.
    This demonstrates authentic physical moment regularizing rather than mode collapse pinning.
    """
    # 4 points arranged symmetrically: (-1, -1), (-1, 1), (1, -1), (1, 1)
    # mean = (0, 0), cov = diag(4/3, 4/3)
    p = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    prior = nn.Module()
    prior.z = nn.Parameter(p.clone())
    capture_initial_moments(prior)

    loss_fn = MomentTetherRegularizer(prior, weight=1.0, eps=1e-4)

    # Consider 8 particles clustered tightly at those 4 locations (2 per cluster):
    # Cluster positions perturbed by tiny epsilon
    p_clustered = torch.cat([p, p], dim=0)  # 8 points
    # Both sets have exact mean (0, 0) and covariance:
    # cov of p_clustered: sum(x^2)/7 = 8 / 7.
    # If we rescale coordinates slightly so cov matches exactly:
    scale = math.sqrt( (4.0 / 3.0) / (8.0 / 7.0) )
    p_clustered_matched = p_clustered * scale
    
    # Check that tether loss is exactly 0
    tether_loss = loss_fn(p_clustered_matched)
    assert tether_loss.item() < 1e-15


def test_moment_tether_autograd_directional_pull():
    """Verify that gradients correctly point toward restoring initial mean and spread."""
    initial = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    prior = nn.Module()
    prior.z = nn.Parameter(initial.clone())
    capture_initial_moments(prior)

    loss_fn = MomentTetherRegularizer(prior, weight=1.0, eps=1e-2)

    # Shift particles right by +0.5 along x-axis
    shifted = (initial + torch.tensor([0.5, 0.0], dtype=torch.float64)).requires_grad_()
    loss = loss_fn(shifted)
    loss.backward()

    # Gradients must pull leftward (negative along x) to counter the shift
    assert shifted.grad is not None
    assert torch.all(shifted.grad[:, 0] > 0)  # positive gradient w.r.t positive displacement
    assert torch.allclose(shifted.grad[:, 1], torch.zeros(2, dtype=torch.float64), atol=1e-10)


# ==============================================================================
# 4. Recipe Invariant & Configuration Integrity Tests
# ==============================================================================

def test_atlas_particle_normal_v4_canonical_frozen_overrides():
    """Verify that atlas-particle-normal-v4 matches git HEAD without reg_coeff override."""
    idea_path = ROOT / "configs/forge/ideas/atlas-particle-normal-v4.json"
    data = json.loads(idea_path.read_text())
    expected = {
        "birth_death_cells": 256,
        "d_guard_min_steps": 0,
        "d_guard_ratio": 0.0,
        "output_noise_mode": "fixed",
        "output_noise_std": 0.0,
    }
    assert data["recipe_overrides"] == expected
    assert "reg_coeff" not in data["recipe_overrides"]


def test_recipe_prior_refactor_bcap_pure_v1_configuration():
    """Verify that recipe-prior-refactor-bcap-pure-v1 declares moment_tether with 0.01 weight."""
    idea_path = ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json"
    data = json.loads(idea_path.read_text())
    overrides = data["recipe_overrides"]
    assert overrides["prior_reg"] == 0.01
    assert overrides["prior_regularizer"] == "moment_tether"
    assert overrides["generator_gradient_response"] == "none"


def test_without_default_additions_filters_generator_gradient_response():
    """Verify without_default_additions properly handles generator_gradient_response."""
    test_dict = {
        "name": "test",
        "lr": 0.01,
        "ebi_beta": 30.0,
        "kappa_radial": 1.0,
        "generator_gradient_response": "none",
    }
    filtered = without_default_additions(test_dict, fields=("ebi_beta", "kappa_radial", "generator_gradient_response"))
    assert "generator_gradient_response" not in filtered
    assert "ebi_beta" not in filtered
    assert "kappa_radial" not in filtered
    assert filtered["lr"] == 0.01
