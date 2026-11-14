"""Named component configurations share current hyperparameters, without loops."""
import json
from pathlib import Path

import pytest
from torch import nn

from particlegan import DEFAULT_RECIPE, GANLoss, GANTrainer, Recipe, get_recipe


K3P_CONFIG = json.loads((Path(__file__).parents[1] / 'reports/toy100/gap-fill-20260925/sources/k3p/config.json').read_text())


def without_name(values):
    return {key: value for key, value in values.items() if key != 'name'}


def test_ka2_matches_every_recorded_winning_field():
    # The frozen K3P config ran arm a_r1r2 under the K3P patch. KA2 is now
    # retained explicitly; legacy arms are absent from its packet.
    actual = json.loads(json.dumps(get_recipe('ka2').to_dict()))
    assert not {'loss_type', 'gan_mode', 'reg_method'} & set(actual)
    assert get_recipe('ka2').reg_arm is None and 'reg_arm' not in actual
    assert (K3P_CONFIG['loss_type'], K3P_CONFIG['gan_mode']) == ('logistic', 'rp')
    shared = (set(actual) & set(K3P_CONFIG)) - {'name', 'reg_arm'}
    assert {key: actual[key] for key in shared} == {key: K3P_CONFIG[key] for key in shared}
    assert {'network_lr_floor', 'network_lr_horizon_cap', 'input_noise_std', 'output_noise_std',
            'output_noise_warmup', 'input_noise_anneal_end', 'batch_size', 'z_dim'} <= shared
    # Same recorded fields; the critic penalty is now KA2 (K3P's fixed anchor decay is
    # replaced by reg_anchor_min_decay, so it is not a shared field).
    assert actual['name'] == 'ka2'
    assert get_recipe('ka2') == Recipe.from_dict({})


def test_default_resolves_the_selected_trainer_preset():
    assert get_recipe() == get_recipe('gan') == get_recipe(DEFAULT_RECIPE)


def test_bcap_matches_the_complete_selected_leader_recipe():
    selected = json.loads((Path(__file__).parents[1] /
        'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())
    recipe = get_recipe('bcap')
    for key, value in selected['recipe_overrides'].items():
        if key in ('prior_reg', 'prior_regularizer'):
            continue
        actual = json.loads(json.dumps(getattr(recipe, key)))
        assert actual == value, key
    assert selected['recipe_overrides']['prior_reg'] == 0.01
    assert selected['recipe_overrides']['prior_regularizer'] == 'moment_tether'



@pytest.mark.parametrize('preset', ['ka2', 'k3p', 'bcap_adam', 'bcap', 'bispf', 'bispf_optimized', 'empf', 'halloween'])
def test_one_selector_promotes_all_model_families_without_changing_explicit_presets(monkeypatch, preset):
    import particlegan.recipes as recipes
    explicit = {name: get_recipe(name) for name in ('ka2', 'k3p', 'bcap_adam', 'bcap', 'bispf', 'bispf_optimized', 'empf', 'e22', 'atlas', 'halloween')}
    monkeypatch.setattr(recipes, 'DEFAULT_RECIPE', preset)
    assert get_recipe() == explicit[preset]
    for model in ('gan', 'mog', 'ddgan', 'ddgan_mog', 'ae_gan', 'vae_gan', 'ae_ddgan'):
        # Task dimensions and component choices are tested separately below.
        recipe = get_recipe(model, lr=.002)
        for field in ('critic_formulation', 'optimizer_family', 'loss', 'reg_arm',
                      'constraint_geometry_mode', 'optimizer_svd_backend', 'training_precision',
                      'continuous_policy', 'prior_update',
                      'prior_regularizer', 'prior_reg', 'prior_l2'):
            assert getattr(recipe, field) == getattr(explicit[preset], field)
        assert recipe.lr == .002
    assert {name: get_recipe(name) for name in explicit} == explicit
    assert Recipe.from_dict(explicit['ka2'].to_dict()) == explicit['ka2']


def test_default_selector_rejects_a_model_family(monkeypatch):
    import particlegan.recipes as recipes
    monkeypatch.setattr(recipes, 'DEFAULT_RECIPE', 'gan')
    with pytest.raises(ValueError, match='DEFAULT_RECIPE must name a trainer preset'):
        get_recipe()


def test_model_trainer_pin_and_invalid_combinations():
    assert get_recipe('gan', trainer='ka2') == get_recipe('ka2')
    assert get_recipe('ae_gan', trainer='ka2').optimizer_family == 'formulation'
    with pytest.raises(ValueError, match='trainer must name a trainer preset'):
        get_recipe('gan', trainer='unknown')
    with pytest.raises(ValueError, match='only supported with a model-family name'):
        get_recipe('bcap', trainer='ka2')


def test_recipe_stays_configuration_and_components_without_training_lifecycle():
    recipe = get_recipe()
    for method in ('make_trainer', 'step', 'sample', 'state_dict', 'load_state_dict'):
        assert not hasattr(recipe, method)
    # Persisting configuration does not create networks, optimizers or a trainer.
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    assert isinstance(recipe.make_loss(), GANLoss)


@pytest.mark.parametrize('name,model,prior,encoder,conditioning', [
    ('gan', 'gan', 'particles', 'none', 'scalar'),
    ('mog', 'gan', 'mog', 'none', 'scalar'),
    ('ddgan', 'ddgan', 'particles', 'none', 'ucd'),
    ('ddgan_mog', 'ddgan', 'mog', 'none', 'ucd'),
    ('ae_gan', 'gan', 'mog', 'ae', 'scalar'),
    ('vae_gan', 'gan', 'mog', 'hard', 'scalar'),
    ('ae_ddgan', 'ddgan', 'mog', 'ae', 'scalar'),
])
def test_named_components_share_current_training_hyperparameters(name, model, prior, encoder, conditioning):
    recipe = get_recipe(name)
    assert recipe == get_recipe(name=name)
    assert (recipe.model, recipe.prior_kind, recipe.encoder_mode, recipe.conditioning) == (
        model, prior, encoder, conditioning)
    fields = ('lr', 'd_lr_mult', 'prior_lr_mult', 'betas', 'prior_betas',
              'reg_coeff', 'reg_kappa', 'reg_every', 'reg_anchor_weight',
              'direct_particle_gain', 'prior_reg', 'ema_decay',
              'lr_anneal_start', 'lr_floor', 'total_steps', 'critic_formulation',
              'optimizer_family', 'loss', 'constraint_geometry_mode',
              'optimizer_smoothing', 'optimizer_convolution', 'prior_update',
              'prior_regularizer', 'prior_l2')
    for field in fields:
        assert getattr(recipe, field) == getattr(get_recipe(), field)
    if prior == 'mog':
        assert recipe.sigma_rel == .025
    assert Recipe.from_dict(recipe.to_dict()) == recipe


@pytest.mark.parametrize('name', ['gan_v1', 'gan_v2', 'gan_v3', 'gan_legacy', 'unknown'])
def test_historical_versions_are_not_selectable(name):
    with pytest.raises(ValueError, match='Unknown recipe'):
        get_recipe(name)


def test_named_recipe_overrides_and_small_objects():
    recipe = get_recipe('ae_gan', num_particles=8, z_dim=3, lr=.002)
    assert recipe.lr == .002 and recipe.z_dim == 3
    assert recipe.make_prior().z.shape == (8, 3)
    assert get_recipe('ddgan', num_classes=2).num_classes == 2
    assert get_recipe('gan') == get_recipe(DEFAULT_RECIPE)
    with pytest.raises(ValueError, match='particle encoders require'):
        get_recipe('ae_gan', prior_kind='particles', sigma_rel=0)
    with pytest.raises(TypeError):
        get_recipe('gan', not_a_field=True)


@pytest.mark.parametrize('name,overrides', [
    ('gan', dict(conditioning='ucd', num_classes=2)),
    ('ae_gan', {}), ('vae_gan', {}), ('ddgan', {}),
])
def test_named_components_do_not_implicitly_choose_training_control_flow(name, overrides):
    recipe = get_recipe(name, **overrides)
    with pytest.raises(ValueError, match='unconditional scalar GANs'):
        GANTrainer(recipe, nn.Linear(recipe.z_dim, 2), nn.Linear(2, 1))


def test_bcap_mog_uses_scalar_trainer_with_explicit_a2_eligibility():
    recipe = get_recipe('mog', trainer='bcap', num_particles=8)
    trainer = GANTrainer(recipe, nn.Linear(recipe.z_dim, 2), nn.Linear(2, 1))
    assert not trainer.prior_mechanisms['a2']['enabled']
    row_local = recipe.replace(standardize=False)
    trainer = GANTrainer(row_local, nn.Linear(row_local.z_dim, 2), nn.Linear(2, 1))
    assert not trainer.prior_mechanisms['a2']['enabled']
    legacy = get_recipe('ka2', prior_kind='mog', sigma_rel=.025, num_particles=8)
    with pytest.raises(ValueError, match='standardized MoG reads couple rows'):
        GANTrainer(legacy, nn.Linear(legacy.z_dim, 2), nn.Linear(2, 1))


def test_default_optimizer_roles_resolve_to_selected_absolute_rates():
    recipe = get_recipe(num_particles=8, z_dim=2)
    generator, discriminator = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, discriminator, recipe.make_prior())
    selected = get_recipe(DEFAULT_RECIPE)
    assert [group['lr'] for group in opt_g.param_groups] == [selected.lr, selected.lr * selected.prior_lr_mult]
    assert [group['lr'] for group in opt_d.param_groups] == [selected.lr * selected.d_lr_mult]


def test_json_integer_zero_moments_construct_real_adam_optimizers():
    recipe = get_recipe('ka2', num_particles=8, z_dim=2, betas=[0, .999], prior_betas=[0, 0])
    optimizers = recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), recipe.make_prior())
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            assert all(type(value) is float for value in group['betas'])
    assert optimizers[0].param_groups[1]['betas'] == (0., 0.)
