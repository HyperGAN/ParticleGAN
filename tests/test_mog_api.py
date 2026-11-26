"""Public MoG contracts, independent of experiment-only dependencies."""
import builtins
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, calibrate_mog_sigma, MoGParticlePrior, ParticlePrior, ParticleRegularizer, Recipe, get_recipe


def test_mog_components_use_common_training_defaults_and_explicit_resources():
    recipe = get_recipe(prior_kind='mog', sigma_rel=.025, num_particles=400, total_steps=28000)
    prior = recipe.make_prior()
    assert prior.z.shape == (400, 2) and prior.sigma_rel == 1/40 and prior.standardize
    assert prior.sigma > 0 and prior.sigma == prior.d0 * prior.sigma_rel
    assert recipe.num_particles == 400 and recipe.total_steps == 28000
    assert recipe.prior_lr_mult == get_recipe().prior_lr_mult and recipe.betas == (0., .999) and recipe.prior_betas == get_recipe().prior_betas
    assert isinstance(recipe.make_prior(), MoGParticlePrior)
    assert type(get_recipe().make_prior()) is ParticlePrior
    assert Recipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe

    from experiments.config import read_config
    from experiments.train_100gaussians import DEFAULTS
    root = Path(__file__).resolve().parents[1]
    cfg = {**DEFAULTS, **read_config(root / "configs/mog/default.toml")}
    reference = read_config(root / "configs/mog/scale_longer/n400_r1over40_28k_s1.yaml")
    # The archived run used the then-implicit cap target of 1.0. It is now
    # exposed in the trainer config, without changing that historical default.
    reference = {"reg_kappa": 1.0, "beta2": .999, "prior_lr_mult": 10., **reference}
    assert {k: v for k, v in cfg.items() if k != "out_dir"} == {
        k: v for k, v in reference.items() if k != "out_dir"}
    assert cfg['epochs'] * cfg['steps_per_epoch'] == recipe.total_steps
    assert cfg['particle_lr_multiplier'] * cfg['prior_lr_mult'] == 100.
    assert cfg['particle_beta1'] == .5


def test_explicit_ddgan_mog_study_config_and_overrides():
    from experiments.config import read_config
    from experiments.train_denoising import training_recipe

    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / "configs/denoising/mog_capacity_100k/ddgan_mog.yaml")
    recipe = training_recipe(cfg)
    assert recipe.lr == .0006 and recipe.prior_lr_mult == 100.
    assert recipe.model == "ddgan" and recipe.conditioning == "ucd"
    assert recipe.total_steps == 100_000 and recipe.lr_floor == 1.
    prior = recipe.make_prior()
    assert isinstance(prior, MoGParticlePrior)
    assert prior.z.shape == (400, 4) and prior.sigma > 0
    assert Recipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe
    changed = get_recipe(model='ddgan', conditioning='ucd', prior_kind='mog', sigma_rel=0.025, num_classes=8, z_dim=6, num_particles=32,
                         total_steps=2000, lr_floor=.05)
    assert changed.num_classes == 8 and changed.make_prior().z.shape == (32, 6)
    assert changed.total_steps == 2000 and changed.lr_floor == .05
    assert get_recipe().lr == get_recipe(DEFAULT_RECIPE).lr
    assert get_recipe(model='ddgan').total_steps == 7000
    assert get_recipe(model='ddgan', num_classes=4, conditioning='ucd').prior_kind == "particles"


def test_forward_matches_sampling_rng_and_flows_through_module_hooks():
    global_rng = torch.get_rng_state().clone()
    prior = MoGParticlePrior(12, 3, sigma=.1, generator=torch.Generator().manual_seed(7)).double()
    direct_rng = torch.Generator().manual_seed(17)
    sample_rng = torch.Generator().manual_seed(17)
    calls = []
    handle = prior.register_forward_hook(lambda *args: calls.append(True))
    indices = prior.sample_indices(20, direct_rng)
    direct = prior(indices, generator=direct_rng)
    sampled, sampled_indices = prior.sample(20, sample_rng)
    handle.remove()
    assert len(calls) == 2
    assert torch.equal(indices, sampled_indices) and torch.equal(direct, sampled)
    assert torch.equal(direct_rng.get_state(), sample_rng.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())
    assert not torch.equal(sampled, prior.means()[indices])
    sampled.square().mean().backward()
    assert prior.z.grad is not None and torch.isfinite(prior.z.grad).all()
    assert set(dict(prior.named_parameters())) == {'z'}


def test_fixed_epsilon_snapshot_and_zero_sigma_rng_contract():
    prior = MoGParticlePrior(8, 2, sigma=.1, standardize=False)
    eps = torch.arange(6, dtype=prior.z.dtype).reshape(3, 2)
    rng = torch.Generator().manual_seed(3)
    state = rng.get_state()
    z, ids = prior.sample(3, rng, fixed_first_n=True, offset=2, eps=eps)
    assert torch.equal(z, prior.z[2:5] + prior.sigma * eps)
    assert torch.equal(ids, torch.arange(2, 5)) and torch.equal(state, rng.get_state())
    for wrong in (eps[:2], eps.double()):
        with pytest.raises(ValueError, match='eps'):
            prior.sample(3, fixed_first_n=True, eps=wrong)
    atoms = ParticlePrior(8, 2, generator=torch.Generator().manual_seed(19))
    zero = MoGParticlePrior(8, 2, sigma=0, standardize=False,
                            generator=torch.Generator().manual_seed(19))
    for fixed in (False, True):
        a, b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        x, i = atoms.sample(5, a, fixed_first_n=fixed)
        y, j = zero.sample(5, b, fixed_first_n=fixed)
        assert torch.equal(x, y) and torch.equal(i, j) and torch.equal(a.get_state(), b.get_state())
    before = torch.get_rng_state().clone()
    assert torch.equal(zero(torch.tensor([0, 1])), atoms(torch.tensor([0, 1])))
    assert torch.equal(before, torch.get_rng_state())


def test_checkpoint_restores_read_settings_fixed_sigma_and_seeded_samples(tmp_path):
    prior = MoGParticlePrior(12, 2, sigma=.125, standardize=False).double()
    with torch.no_grad():
        prior.z.mul_(2).add_(3)  # Do not recalibrate after training/EMA updates.
    sigma = prior.sigma.clone()
    path = tmp_path / 'mog.pt'
    torch.save(prior.state_dict(), path)
    restored = MoGParticlePrior(12, 2, sigma=0, standardize=True).double()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert not restored.standardize and restored.sigma_rel == prior.sigma_rel
    assert torch.equal(sigma, restored.sigma)
    x, idx = prior.sample(32, torch.Generator().manual_seed(21))
    y, ids = restored.eval().sample(32, torch.Generator().manual_seed(21))
    assert torch.equal(x, y) and torch.equal(idx, ids)

    # A parent module also restores the MoG's extra state and noisy forward path.
    parent = nn.ModuleDict({'prior': MoGParticlePrior(12, 2, sigma=0)})
    parent.load_state_dict({'prior.' + k: v for k, v in prior.state_dict().items()})
    assert not parent['prior'].standardize and parent['prior']._noise_enabled

    # Historical checkpoints contain tensors only; caller supplies read standardization.
    legacy = {k: v for k, v in prior.state_dict().items() if k != '_extra_state'}
    old = MoGParticlePrior(12, 2, sigma=0, standardize=False).double()
    old.load_state_dict(legacy, strict=True)
    y, _ = old.sample(32, torch.Generator().manual_seed(21))
    assert torch.equal(x, y)


@pytest.mark.parametrize('use_scipy', [False, True])
def test_calibration_exact_even_median_and_torch_only_fallback(monkeypatch, use_scipy):
    if use_scipy:
        pytest.importorskip('scipy')
    else:
        original_import = builtins.__import__
        def torch_only(name, *args, **kwargs):
            if name.startswith(('scipy', 'numpy')):
                raise ImportError('optional dependency unavailable')
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', torch_only)
    centers = torch.tensor([[0.], [1.], [4.], [10.]], dtype=torch.float64,
                           requires_grad=True)
    original = centers.detach().clone()
    before = torch.get_rng_state().clone()
    sigma, d0 = calibrate_mog_sigma(centers, .5)
    # NN distances [1,1,3,6]: midpoint median is 2, not the lower middle value 1.
    assert d0.item() == 2 and sigma.item() == 1
    assert not sigma.requires_grad and not d0.requires_grad
    assert torch.equal(centers, original)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('model', ['gan', 'ddgan'])
def test_mog_recipe_optimizer_updates_raw_means_and_preserves_fixed_buffers(model):
    recipe = get_recipe(trainer='ka2', model=model, prior_kind='mog', sigma_rel=.025, num_particles=12, z_dim=2, prior_betas=[.5, .999])
    prior = recipe.make_prior()
    generator, critic = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, foreach=False)
    assert [group['lr'] for group in opt_g.param_groups] == [.00425, .0085]
    assert [group['betas'] for group in opt_g.param_groups] == [(0., .999), (.5, .999)]
    assert opt_d.param_groups[0]['betas'] == (0., .999)
    before, sigma = prior.z.detach().clone(), prior.sigma.clone()
    eps = torch.ones(6, 2)
    z, _ = prior.sample(6, fixed_first_n=True, eps=eps)
    loss = generator(z).square().mean() + ParticleRegularizer()(prior.z)
    loss.backward()
    opt_g.step()
    assert not torch.equal(before, prior.z) and torch.equal(sigma, prior.sigma)
    ema = copy.deepcopy(prior)
    with torch.no_grad():
        prior.z[0].add_(2)
        ema.z.lerp_(prior.z, .1)
    torch.testing.assert_close(ema.means().mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    assert not torch.equal(ema.means(), prior.means()) and torch.equal(sigma, ema.sigma)
    frozen_recipe = recipe.replace(prior_update="frozen")
    frozen = frozen_recipe.make_prior()
    assert not list(frozen.parameters())
    assert len(frozen_recipe.make_optimizers(generator, critic, frozen)[0].param_groups) == 1


@pytest.mark.parametrize('sigma', [-1, float('nan'), float('inf'), -float('inf'),
                                    [0.1], torch.tensor([0.1]), None, '0.1', 1j, True,
                                    1e100])
def test_invalid_sigma_fails(sigma):
    with pytest.raises(ValueError, match='sigma'):
        MoGParticlePrior(sigma=sigma)


def test_sigma_is_required_and_keyword_only():
    with pytest.raises(TypeError, match='sigma'):
        MoGParticlePrior()
    with pytest.raises(TypeError):
        MoGParticlePrior(4, 2, 1., None, None, True, None, .1)
    with pytest.raises(TypeError, match='sigma_rel'):
        MoGParticlePrior(sigma=.1, sigma_rel=.025)


@pytest.mark.parametrize('kwargs', [dict(standardize='false'), dict(num_particles=1)])
def test_invalid_mog_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        MoGParticlePrior(sigma=0, **kwargs)


@pytest.mark.parametrize('kwargs', [dict(prior_kind='typo'), dict(sigma_rel=-1),
                                     dict(standardize='false'), dict(prior_betas=[1., .9]),
                                     dict(prior_kind='particles', sigma_rel=.1)])
def test_invalid_recipe_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        get_recipe(**{'prior_kind': 'mog', 'sigma_rel': .025, **kwargs})


def test_factory_overrides_are_local_and_cannot_silently_discard_noise():
    original = get_recipe()
    mog = original.make_prior(prior_kind='mog', sigma_rel=.1, num_particles=8)
    assert isinstance(mog, MoGParticlePrior) and original.prior_kind == 'particles'
    with pytest.raises(ValueError, match='nonzero sigma_rel'):
        original.make_prior(sigma_rel=.1)
    with pytest.raises(ValueError, match='prior_kind'):
        original.make_prior(prior_kind='typo')


def forbid_calibration(monkeypatch):
    import particlegan.particle_prior as module
    def fail(*args, **kwargs):
        pytest.fail("unexpected calibration/distance calculation")
    monkeypatch.setattr(module, 'calibrate_mog_sigma', fail)
    monkeypatch.setattr(torch, 'cdist', fail)
    original_import = builtins.__import__
    def no_scipy(name, *args, **kwargs):
        if name.startswith('scipy'):
            fail()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_scipy)


@pytest.mark.parametrize('local_generator', [False, True])
def test_construction_never_calibrates_and_preserves_centers_and_rng(monkeypatch, local_generator):
    forbid_calibration(monkeypatch)
    # Also forbid computing means: initialization must only draw the raw centers.
    original_means = MoGParticlePrior.means
    monkeypatch.setattr(MoGParticlePrior, 'means', lambda self: pytest.fail('unexpected means read'))
    global_state = torch.get_rng_state().clone()
    a = torch.Generator().manual_seed(7) if local_generator else None
    expected = ParticlePrior(32, 4, generator=a)
    expected_global = torch.get_rng_state().clone()
    torch.set_rng_state(global_state)
    b = torch.Generator().manual_seed(7) if local_generator else None
    actual = MoGParticlePrior(32, 4, sigma=.212616428732872, generator=b)
    assert torch.equal(expected.z, actual.z)
    assert torch.equal(expected_global, torch.get_rng_state())
    if local_generator:
        assert torch.equal(a.get_state(), b.get_state())
    assert actual.sigma.ndim == 0 and not actual.sigma.requires_grad
    assert actual.d0 == 0 and actual.sigma_rel == 0
    # No positive spacing is needed for a fixed scale, including coincident means.
    monkeypatch.setattr(MoGParticlePrior, 'means', original_means)
    zero = MoGParticlePrior(1, 2, init_std=0, sigma=0, standardize=False)
    assert torch.equal(zero.sample(4)[0], torch.zeros(4, 2))


@pytest.mark.parametrize('extra_state', [False, True])
@pytest.mark.parametrize('standardize', [False, True])
@pytest.mark.parametrize('sigma', [0., .212616428732872])
def test_legacy_checkpoint_samples_and_rng_are_identical(monkeypatch, extra_state, standardize, sigma):
    forbid_calibration(monkeypatch)
    # Old-format state made independently of the new implementation; centers
    # represent already-trained values, whose current spacing must be ignored.
    centers = torch.randn(12, 3, generator=torch.Generator().manual_seed(9), dtype=torch.float64) * 3 + 2
    state = {'z': centers, 'sigma': torch.tensor(sigma, dtype=torch.float64),
             'd0': torch.tensor(1.7, dtype=torch.float64)}
    if extra_state:
        state['_extra_state'] = {'sigma_rel': .025, 'standardize': standardize}
    prior = MoGParticlePrior(12, 3, sigma=.9, dtype=torch.float64,
                             standardize=not standardize if extra_state else standardize)
    before = torch.get_rng_state().clone()
    prior.load_state_dict(state, strict=True)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(prior.z, centers) and torch.equal(prior.d0, state['d0'])
    assert torch.equal(prior.sigma, state['sigma'])
    assert prior.standardize == standardize
    if extra_state:
        assert prior.get_extra_state() == state['_extra_state']
    for fixed in (False, True):
        a, b = torch.Generator().manual_seed(21), torch.Generator().manual_seed(21)
        ids = torch.arange(8) if fixed else torch.randint(12, (8,), generator=a)
        means = (centers - centers.mean(0)) / (centers.std(0) + 1e-6) if standardize else centers
        expected = means[ids]
        if sigma > 0:
            expected = expected + state['sigma'] * torch.randn(expected.shape, dtype=expected.dtype, generator=a)
        actual, indices = prior.sample(8, b, fixed_first_n=fixed)
        assert torch.equal(actual, expected) and torch.equal(indices, ids)
        assert torch.equal(a.get_state(), b.get_state())
    # Re-saving must retain legacy metadata and fixed buffers.
    saved = prior.state_dict()
    assert torch.equal(saved['d0'], state['d0']) and torch.equal(saved['sigma'], state['sigma'])


def test_recipe_explicit_sigma_skips_calibration(monkeypatch):
    forbid_calibration(monkeypatch)
    prior = get_recipe('mog').make_prior(sigma=.2)
    assert prior.sigma == torch.tensor(.2) and prior.d0 == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_recipe_calibrates_original_centers_with_legacy_rounding(dtype):
    import numpy as np
    cKDTree = pytest.importorskip('scipy.spatial').cKDTree
    a, b = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
    centers = ParticlePrior(16, 3, dtype=dtype, generator=a).z.detach()
    means = (centers - centers.mean(0)) / (centers.std(0) + 1e-6)
    points = means.cpu().double().numpy()
    d0 = centers.new_tensor(float(np.median(cKDTree(points).query(points, k=2)[0][:, 1])))
    # This migration contract compares the historical random centers.
    prior = get_recipe('mog', num_particles=16, z_dim=3).make_prior(dtype=dtype, generator=b)
    assert torch.equal(prior.z, centers) and torch.equal(a.get_state(), b.get_state())
    assert torch.equal(prior.d0, d0) and torch.equal(prior.sigma, d0 * .025)


def test_set_sigma_updates_zero_noise_rng_behavior():
    prior = MoGParticlePrior(8, 2, sigma=0)
    rng = torch.Generator().manual_seed(5)
    before = rng.get_state()
    prior.set_sigma(torch.tensor(.2))
    prior.sample(4, rng, fixed_first_n=True)
    assert not torch.equal(before, rng.get_state())
    prior.set_sigma(0)
    before = rng.get_state()
    prior.sample(4, rng, fixed_first_n=True)
    assert torch.equal(before, rng.get_state())


@pytest.mark.parametrize('centers, sigma_rel', [
    (torch.zeros(1, 2), .025), (torch.zeros(2, 2), .025),
    (torch.tensor([[0.], [float('nan')]]), .025), (torch.ones(2), .025),
    (torch.ones(2, 1, dtype=torch.int64), .025), (torch.tensor([[0.], [1.]]), -1),
])
def test_calibration_rejects_invalid_input(centers, sigma_rel):
    with pytest.raises(ValueError):
        calibrate_mog_sigma(centers, sigma_rel)
