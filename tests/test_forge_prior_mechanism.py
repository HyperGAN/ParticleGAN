"""Tests for technique-owned prior_mechanism abstraction in Forge."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from particlegan import ManipulatedPrior, ParticlePrior, unwrap_prior
from particlegan.priors import NormalNoiseManipulator
from experiments.forge.api import (
    FormulationContext,
    policy_controls,
    task_formulation_context,
    task_policy_blockers,
)
from experiments.forge.contracts import read_json, validate_idea
from experiments.forge.policy_adapters import finite_policy_state
from experiments.forge.planning import candidate_revision_for, load_idea, resolve_idea
from experiments.forge.priors import (
    prior_code_path,
    prior_noise_sigma,
    prior_policy_receipt,
    prior_uses_noise_stream,
    resolve_prior,
    resolve_prior_mechanism,
    validate_prior_mechanism,
)
from experiments.forge.state import require_consistent_rng, state_digest
from experiments.forge.views import load_tasks

ROOT = Path(__file__).resolve().parents[1]

PARTICLE_NORMAL_MECHANISM = {
    "kind": "particles",
    "standardize": False,
    "manipulators": [{"kind": "normal_noise", "sigma": "task"}],
}


def test_validate_and_resolve_prior_mechanism():
    assert validate_prior_mechanism(None) is None
    validate_prior_mechanism(PARTICLE_NORMAL_MECHANISM)

    mog_task_prior = resolve_prior(
        {"kind": "mog", "sigma": 0.025, "standardize": False, "learnable": True},
        explicit=True,
    )
    resolved = resolve_prior_mechanism(PARTICLE_NORMAL_MECHANISM, mog_task_prior)
    assert resolved["kind"] == "particle_cloud"
    assert resolved["sigma"] == 0.0
    assert resolved["standardize"] is False
    assert resolved["owner"] == "technique"
    assert resolved["manipulators"] == [{"kind": "normal_noise", "sigma": 0.025}]
    assert prior_noise_sigma(resolved) == pytest.approx(0.025)
    assert prior_uses_noise_stream(resolved) is True
    assert prior_code_path(resolved) == "ManipulatedPrior(ParticlePrior, [NormalNoiseManipulator])"

    cloud_task_prior = resolve_prior(
        {
            "kind": "particle_cloud",
            "sigma": 0.0,
            "standardize": False,
            "learnable": True,
            "exception_reason": "enumerated 32-center image task",
        },
        explicit=True,
    )
    resolved_zero = resolve_prior_mechanism(PARTICLE_NORMAL_MECHANISM, cloud_task_prior)
    assert resolved_zero["kind"] == "particle_cloud"
    assert resolved_zero["manipulators"] == []
    assert prior_noise_sigma(resolved_zero) == 0.0
    assert prior_uses_noise_stream(resolved_zero) is False
    assert prior_code_path(resolved_zero) == "ParticlePrior"

    fixed_sigma_mech = {
        "kind": "particles",
        "standardize": False,
        "manipulators": [{"kind": "normal_noise", "sigma": 0.005}],
    }
    resolved_mog_fixed = resolve_prior_mechanism(fixed_sigma_mech, mog_task_prior)
    assert resolved_mog_fixed["manipulators"] == [{"kind": "normal_noise", "sigma": 0.005}]
    assert prior_noise_sigma(resolved_mog_fixed) == pytest.approx(0.005)
    assert prior_code_path(resolved_mog_fixed) == "ManipulatedPrior(ParticlePrior, [NormalNoiseManipulator])"

    resolved_cloud_fixed = resolve_prior_mechanism(fixed_sigma_mech, cloud_task_prior)
    assert resolved_cloud_fixed["manipulators"] == []
    assert prior_noise_sigma(resolved_cloud_fixed) == 0.0
    assert prior_uses_noise_stream(resolved_cloud_fixed) is False
    assert prior_code_path(resolved_cloud_fixed) == "ParticlePrior"


@pytest.mark.parametrize(
    "bad_mechanism,match",
    [
        ("particles", "prior_mechanism must be an object"),
        ({"kind": "particles"}, "prior_mechanism requires"),
        ({"kind": "particles", "standardize": True}, "requires standardize=False"),
        ({"kind": "gaussian", "standardize": False}, "kind must be particles or mog"),
        (
            {
                "kind": "mog",
                "standardize": False,
                "manipulators": [{"kind": "normal_noise", "sigma": "task"}],
            },
            "already includes a Gaussian kernel",
        ),
        (
            {
                "kind": "particles",
                "standardize": False,
                "manipulators": [{"kind": "normal_noise", "sigma": -0.1}],
            },
            "finite nonnegative",
        ),
    ],
)
def test_validate_prior_mechanism_rejects_invalid_declarations(bad_mechanism, match):
    with pytest.raises(ValueError, match=match):
        validate_prior_mechanism(bad_mechanism)


def test_v3_idea_allows_prior_mechanism_without_task_owned_recipe_overrides():
    idea = {
        "schema_version": 3,
        "id": "atlas-prior-mechanism-probe",
        "parent": "atlas",
        "changed_factors": ["Swap MoG prior for unstandardized particle table + normal noise"],
        "mechanism_class": "structural",
        "mechanism_rationale": "Eliminate standardization cross-row Jacobian coupling.",
        "recipe_preset": "atlas",
        "recipe_overrides": {"output_noise_mode": "fixed", "output_noise_std": 0.0},
        "prior_mechanism": deepcopy(PARTICLE_NORMAL_MECHANISM),
        "requires_capabilities": ["particle_cloud", "a2", "named_rng", "policy_controls", "policy_serving"],
        "claim_contract": {
            "schedule": "schedule_free",
            "scoring_weights": "live",
            "sampling_law": "task_declared",
        },
    }
    validate_idea(idea)


def test_existing_candidate_revision_unchanged_without_prior_mechanism():
    k3p = load_idea(ROOT, "k3p")
    resolved = resolve_idea(ROOT, "k3p")
    candidate = deepcopy(resolved["candidate"])
    rev_without = candidate_revision_for(resolved["source"]["digest"], candidate)
    assert rev_without == resolved["candidate_revision"]
    assert "prior_mechanism" not in k3p

    with_mech = deepcopy(candidate)
    with_mech["prior_mechanism"] = deepcopy(PARTICLE_NORMAL_MECHANISM)
    rev_with = candidate_revision_for(resolved["source"]["digest"], with_mech)
    assert rev_with != rev_without


def test_policy_blockers_respect_opt_in_prior_mechanism():
    tasks = load_tasks(ROOT)
    legacy_atlas = load_idea(ROOT, "atlas")
    assert task_policy_blockers(tasks["grid100"], legacy_atlas)
    assert task_policy_blockers(tasks["gaussian1d_smoke"], legacy_atlas)

    opted_in = {
        **deepcopy(legacy_atlas),
        "prior_mechanism": deepcopy(PARTICLE_NORMAL_MECHANISM),
        "recipe_overrides": {"output_noise_mode": "fixed", "output_noise_std": 0.0},
    }
    for runnable in (
        "gaussian1d_smoke",
        "ring16_acquisition",
        "grid100",
        "img_stripes2",
        "mode_hold",
        "ring_hold",
        "target_shift_recovery",
    ):
        assert task_policy_blockers(tasks[runnable], opted_in) == [], f"Unexpected blocker on {runnable}"

    for unblocked_component_host in (
        "two_pole",
        "ae_gan_hold",
        "unused_token_hold",
        "trajectory",
    ):
        assert task_policy_blockers(tasks[unblocked_component_host], opted_in) == [], (
            f"Unexpected blocker on {unblocked_component_host}"
        )


def test_native_profile_and_deterministic_init_unwrap_manipulated_prior():
    tasks = load_tasks(ROOT)
    protocol = read_json(ROOT / "configs/forge/protocols/ Forge-protocol-v1.json".replace(" ", "")) \
        if (ROOT / "configs/forge/protocols/forge-protocol-v1.json").exists() \
        else {"seed": 0}

    candidate = {
        "id": "atlas-mech-check",
        "recipe_preset": "atlas",
        "recipe_overrides": {"output_noise_mode": "fixed", "output_noise_std": 0.0},
        "prior_mechanism": deepcopy(PARTICLE_NORMAL_MECHANISM),
        "requires_capabilities": ["particle_cloud", "a2", "named_rng", "policy_controls", "policy_serving"],
        "claim_contract": {
            "schedule": "schedule_free",
            "scoring_weights": "live",
            "sampling_law": "task_declared",
        },
    }

    # 1. grid100 task context + explicit sample_distributions_v1 host_initialization
    grid_task = tasks["grid100"]
    ctx_grid = task_formulation_context(candidate, grid_task, protocol, device="cpu", root=ROOT)
    G = nn.Linear(2, 2, bias=True)
    D = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    trainer = ctx_grid.build_trainer(G, D, max_steps=10)

    assert isinstance(trainer.prior, ManipulatedPrior)
    assert isinstance(unwrap_prior(trainer.prior), ParticlePrior)
    assert isinstance(trainer.prior.manipulators[0], NormalNoiseManipulator)
    assert trainer.prior.manipulators[0].sigma == pytest.approx(0.025)
    assert trainer.prior_noise_generator is not None
    assert list(ctx_grid.initialization["prior"]["parameter_seeds"].keys()) == ["z"]

    # Also verify sample_distributions_v1 host_initialization unwraps ManipulatedPrior
    ctx_host_init = FormulationContext(
        recipe_preset="atlas",
        recipe_overrides={"num_particles": 64, "z_dim": 2, "batch_size": 32,
                          "output_noise_mode": "fixed", "output_noise_std": 0.0},
        prior={"kind": "mog", "sigma": 0.025, "standardize": False},
        prior_contract="recipe_owned_v1",
        prior_mechanism=deepcopy(PARTICLE_NORMAL_MECHANISM),
        host_initialization={
            "schema_version": 1,
            "owner": "task",
            "profile_sha256": "0" * 64,
            "components": {
                "prior": {
                    "method": "sample_distributions_v1",
                    "parameters": {"z": {"kind": "uniform", "low": -5.0, "high": 5.0}},
                }
            },
        },
        seed=0,
        device="cpu",
    )
    host_prior = ctx_host_init.build_prior()
    assert isinstance(host_prior, ManipulatedPrior)
    assert "z" in ctx_host_init.initialization["prior"]["parameters"]

    # Ownership receipt attributes prior_kind and standardize to technique
    receipt = ctx_grid.receipt()
    ownership = receipt["field_ownership"]
    assert ownership["recipe_fields"]["prior_kind"]["owner"] == "technique"
    assert ownership["recipe_fields"]["standardize"]["owner"] == "technique"
    assert ownership["task_contract"]["prior"]["owner"] == "task"
    assert ownership["task_contract"]["prior"]["technique_mechanism"]["owner"] == "technique"
    assert (
        ownership["task_contract"]["prior"]["code_path"]
        == "ManipulatedPrior(ParticlePrior, [NormalNoiseManipulator])"
    )

    # Step, checkpoint RNG consistency, finite policy state, and exact restore parity
    real = torch.randn(2048, 2)
    trainer.step(real)
    state = ctx_grid.state_dict()
    require_consistent_rng(state)
    assert policy_controls(ctx_grid.recipe)
    assert finite_policy_state(state)

    ctx_grid_2 = task_formulation_context(candidate, grid_task, protocol, device="cpu", root=ROOT)
    G2 = nn.Linear(2, 2, bias=True)
    D2 = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    ctx_grid_2.build_trainer(G2, D2, max_steps=10)
    ctx_grid_2.load_state_dict(state)
    assert state_digest(ctx_grid_2.state_dict()) == state_digest(state)

    # 2. Deterministic orthogonal parameter seed parity ("z" vs plain ParticlePrior)
    g1d_task = tasks["gaussian1d_smoke"]
    ctx_manip = task_formulation_context(candidate, g1d_task, protocol, device="cpu", root=ROOT)
    p_manip = ctx_manip.build_prior()

    ctx_plain = FormulationContext(
        recipe_preset="atlas",
        recipe_overrides={"num_particles": 256, "z_dim": 2, "batch_size": 128},
        prior={
            "kind": "particle_cloud",
            "sigma": 0.0,
            "standardize": False,
            "learnable": True,
            "exception_reason": "seed parity reference",
        },
        seed=0,
        device="cpu",
    )
    p_plain = ctx_plain.build_prior()
    assert ctx_manip.initialization["prior"]["parameter_seeds"] == ctx_plain.initialization["prior"]["parameter_seeds"]
    assert torch.equal(unwrap_prior(p_manip).z, p_plain.z)

    # 3. Prior policy receipt records manipulators cleanly
    policy_proof = prior_policy_receipt(ctx_manip.recipe, p_manip)
    assert policy_proof["trainable_locations"] is True
    assert policy_proof["manipulators"] == [
        [
            {
                "class": "particlegan.priors.NormalNoiseManipulator",
                "config": {"penalty_weight": 1.0, "sigma": 0.1, "kappa_radial": 1.0},
            }
        ]
    ]


def test_screening_submission_and_host_profile_validation_with_prior_mechanism():
    """Verify promotion.validate_screening_submission and hostprofiles accept prior_mechanism."""
    from experiments.forge.hostprofiles import _validate_candidate_identity
    from experiments.forge.promotion import validate_screening_submission

    request = resolve_idea(
        ROOT,
        "atlas-particle-normal-v1",
        study="atlas-particle-normal-replication-study-v1",
    )
    assert request["preflight_blockers"] == []
    assert request["study_review"]["status"] == "READY"
    validate_screening_submission(
        request,
        root=ROOT,
        campaign=request["study"]["campaign"],
    )
    _validate_candidate_identity(request)

