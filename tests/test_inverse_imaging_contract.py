"""Inverse-imaging extensions must preserve configuration and public initialization."""
from copy import deepcopy

import pytest
import torch

from particlegan import ParticlePrior
from particlegan.init import deterministic_orthogonal_
from particlegan.tomographic import (
    CosineRepulsionManipulator,
    MarginalEntropyManipulator,
)
from particlegan.wavefront import (
    GhostFreeDeMixingManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
    sobolev_filter_2d,
)


@pytest.mark.parametrize("factory,first,second", [
    (SobolevWavefrontManipulator, {"grad_lambda": 1.0}, {"grad_lambda": 2.0}),
    (SobolevWavefrontManipulator, {"patch_size": 4}, {"patch_size": 8}),
    (GhostFreeDeMixingManipulator,
     {"num_modes": 2, "representation_shape": (1, 4, 4), "gamma_detail": 1.0},
     {"num_modes": 2, "representation_shape": (1, 4, 4), "gamma_detail": 2.0}),
    (SplatCarrierManipulator, {"num_modes": 2, "gamma_base": 1.0},
     {"num_modes": 2, "gamma_base": 2.0}),
    (MarginalEntropyManipulator, {"min_entropy_ratio": .8}, {"min_entropy_ratio": .9}),
    (CosineRepulsionManipulator, {"margin": .8}, {"margin": .9}),
])
def test_checkpoint_rejects_a_different_inverse_imaging_law(factory, first, second):
    prior = ParticlePrior(8, 2).manipulate(factory(**first))
    state = deepcopy(prior.state_dict())
    identical = ParticlePrior(8, 2).manipulate(factory(**first))
    identical.load_state_dict(state)
    different = ParticlePrior(8, 2).manipulate(factory(**second))
    assert prior.manipulator_spec() != different.manipulator_spec()
    with pytest.raises(ValueError, match="configuration mismatch"):
        different.load_state_dict(state)


def test_public_initializer_covers_splat_geometry_and_ignores_constructor_rng():
    priors = []
    for constructor_seed in (13, 29):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(constructor_seed)
            prior = ParticlePrior(8, 2).manipulate(SplatCarrierManipulator(
                num_modes=2, channels=1, height=8, width=8,
                num_base_splats=2, num_detail_splats=2))
        before = torch.get_rng_state().clone()
        deterministic_orthogonal_(prior, seed=0)
        assert torch.equal(before, torch.get_rng_state())
        priors.append(prior)
    for name, value in priors[0].state_dict().items():
        other = priors[1].state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    basis = priors[0].manipulators[0].basis
    detail = basis.detail_params.view(2, 2, 6)
    torch.testing.assert_close(detail[:, :, 2:4], torch.full((2, 2, 2), torch.log(torch.tensor(.15))))


def test_public_splat_initializer_preserves_a_frozen_detail_layout():
    prior = ParticlePrior(8, 2).manipulate(SplatCarrierManipulator(num_modes=2))
    detail = prior.manipulators[0].basis.detail_params
    with torch.no_grad():
        detail.fill_(.123)
    detail.requires_grad_(False)
    before = detail.detach().clone()
    deterministic_orthogonal_(prior, seed=0)
    assert torch.equal(before, detail)


def test_public_api_example_records_real_frames_and_exact_replay(tmp_path):
    from PIL import Image
    from examples.inverse_imaging_smoke import run

    result = run(tmp_path)
    assert result["verdict"] == "PASS"
    assert result["updates"] == 3
    assert result["qualification_input"] is result["image_quality_graded"] is False
    with Image.open(tmp_path / "public-api-integration.gif") as media:
        assert media.n_frames == 4


def test_public_api_replay_gate_rejects_changed_prior_even_with_identical_losses(tmp_path, monkeypatch):
    import json
    from examples import inverse_imaging_smoke as example

    build = example._build
    calls = 0

    def corrupt_restored_prior():
        nonlocal calls
        trainer = build()
        calls += 1
        if calls == 2:
            step = trainer.step

            def corrupted_step(batch):
                result = step(batch)
                with torch.no_grad():
                    trainer.prior.z.add_(.1)
                return result

            trainer.step = corrupted_step
        return trainer

    monkeypatch.setattr(example, "_build", corrupt_restored_prior)
    with pytest.raises(RuntimeError, match="integration failed"):
        example.run(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["metric_gate"]["checkpoint_replay_exact"] is False


@pytest.mark.parametrize("factory", [MarginalEntropyManipulator, CosineRepulsionManipulator])
def test_invalid_penalty_source_cannot_silently_disable_the_objective(factory):
    with pytest.raises(ValueError, match="source"):
        factory(source="typo")
    with pytest.raises(ValueError, match="weight"):
        factory(weight=-1)


@pytest.mark.parametrize("options", [{"lam": -1}, {"order": float("nan")}, {"nyquist": 0}])
def test_functional_gradient_filter_validates_its_configuration(options):
    with pytest.raises(ValueError):
        sobolev_filter_2d(torch.ones(4, 4), **options)
