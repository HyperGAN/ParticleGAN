"""Public default binding and exact promotion of the measured discriminator."""
import pytest
import torch
from torch import nn

from particlegan import GANLoss, GANTrainer, LinearSkipDiscriminator, Recipe, get_recipe
from particlegan.grad_regularizers import GradientPenalty
from particlegan.ka2 import KA2GradientPenalty
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES, constructor


def test_public_discriminator_preserves_research_initialization_and_cap_gradients():
    card = next(c for c in ARCHITECTURES if c['name'] == 'linear_skip_d96_beta5')
    torch.manual_seed(0)
    reference = constructor(card)(hidden_dim=96)
    torch.manual_seed(0)
    promoted = LinearSkipDiscriminator()
    assert reference.state_dict().keys() == promoted.state_dict().keys()
    assert all(torch.equal(v, promoted.state_dict()[k]) for k, v in reference.state_dict().items())
    rng = torch.Generator().manual_seed(0)
    real, fake = torch.randn(16, 2, generator=rng), torch.randn(16, 2, generator=rng)
    for model in (reference, promoted):
        with torch.no_grad():
            model.main.net[-1].weight.mul_(100.)
            model.skip.weight.fill_(2.)
        loss = KA2GradientPenalty(**get_recipe("ka2")._penalty_options())(model, real, fake)
        assert loss > 0
        (loss + model(real).mean()).backward()
    assert torch.equal(reference(real), promoted(real))
    for a, b in zip(reference.parameters(), promoted.parameters()):
        assert a.grad is not None and torch.equal(a.grad, b.grad)


def test_bcap_optimizers_and_losses_bind_the_selected_leader():
    recipe = get_recipe("bcap", num_particles=16)
    g, d = nn.Linear(2, 2), LinearSkipDiscriminator()
    trainer = GANTrainer(recipe, g, d)
    assert [group['lr'] for group in trainer.opt_g.param_groups] == [.012, .03]
    assert [group['lr'] for group in trainer.opt_d.param_groups] == pytest.approx([.018])
    assert isinstance(trainer.penalty.regularizer, GradientPenalty)
    assert trainer.ema_D is None and trainer.latent_damping is None
    assert isinstance(trainer.loss, GANLoss)
    result = trainer.step(torch.randn(8, 2))
    assert torch.equal(result['loss_g'], result['loss_gan'] + 0. * result['prior_regularization'])
    assert Recipe.from_dict(trainer.state_dict()['recipe']) == recipe


@pytest.mark.parametrize('options', [dict(prior_kind='mog', sigma_rel=.025),
    dict(model='ddgan', conditioning='ucd', num_classes=4),
    dict(prior_kind='mog', sigma_rel=.025, encoder_mode='ae'),
    dict(prior_kind='mog', sigma_rel=.025, encoder_mode='hard')])
def test_component_choices_share_the_winning_training_defaults(options):
    recipe = get_recipe(**options)
    assert (recipe.reg_coeff, recipe.reg_kappa, recipe.prior_reg, recipe.betas) == (1., 1., .01, (0., .999))
    selected = get_recipe()
    for field in ("network_lr_floor", "network_lr_horizon_cap", "lr", "d_lr_mult", "prior_lr_mult",
                  "optimizer_family", "loss", "constraint_geometry_mode"):
        assert getattr(recipe, field) == getattr(selected, field)


@pytest.mark.parametrize('kwargs', [dict(in_dim=0), dict(hidden_dim=0), dict(n_hidden=0),
                                   dict(fourier=-1), dict(beta=0), dict(beta=float('nan'))])
def test_reference_discriminator_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        LinearSkipDiscriminator(**kwargs)


def test_historical_stock_comparison_preserves_its_original_settings(tmp_path):
    from benchmarks.locked_shared.grid_study import load_example, resolved_kwargs

    kwargs = resolved_kwargs(load_example(), tmp_path, 'stock', 'cpu', {})
    assert (kwargs['lr'], kwargs['beta1'], kwargs['beta2']) == (.0006, 0., .999)
    assert (kwargs['reg_coeff'], kwargs['reg_kappa'], kwargs['lambda_ep']) == (1., 1., 1.)
