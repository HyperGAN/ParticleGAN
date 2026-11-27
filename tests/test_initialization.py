"""particlegan.init: explicit deterministic initialization and its declaration registry."""
import copy
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import (BatchDistanceDiscriminator, GANTrainer, LinearSkipDiscriminator, MoGParticlePrior,
                         ParticlePrior, get_recipe, init)
from particlegan.particle_prior import calibrate_mog_sigma


def test_recipe_never_touches_weights():
    torch.manual_seed(0)
    recipe = get_recipe(num_particles=8)
    g, d = nn.Linear(2, 2), nn.Linear(2, 1)
    prior, ema = recipe.make_prior(), copy.deepcopy(d)
    modules = (g, d, prior, ema)
    before = [[p.clone() for p in m.parameters()] for m in modules]
    recipe.make_optimizers(g, d, prior, ema_critic=ema)
    GANTrainer(recipe, g, d, prior=prior)
    for saved, module in zip(before, modules):
        assert all(torch.equal(a, b) for a, b in zip(saved, module.parameters()))
    assert "initialization" not in recipe.to_dict()
    with pytest.raises(TypeError):
        get_recipe(initialization=None)


def test_explicit_init_is_rng_neutral_repeatable_and_seeded():
    torch.manual_seed(0)
    a, b = nn.Linear(4, 4), nn.Linear(4, 4)
    before = torch.get_rng_state().clone()
    assert init.deterministic_orthogonal_(a) is a
    init.deterministic_orthogonal_(b, seed=0)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(a.weight, b.weight)
    init.deterministic_orthogonal_(b, seed=1)
    assert not torch.equal(a.weight, b.weight)
    d = init.deterministic_orthogonal_(BatchDistanceDiscriminator(hidden_dim=8, n_hidden=1), seed=1)
    assert torch.count_nonzero(d.head.weight[:, -4:]) == 0


def test_priors_take_r2_tables_and_mog_spacing_follows():
    recipe = get_recipe("mog", num_particles=32, z_dim=3)
    prior = recipe.make_prior()
    sigma = prior.sigma.clone()
    init.deterministic_orthogonal_(prior)
    again = init.deterministic_orthogonal_(recipe.make_prior())
    assert torch.equal(prior.z, again.z) and torch.equal(prior.sigma, again.sigma)
    assert not torch.equal(prior.sigma, sigma)
    # sigma_rel=0 still calibrates d0 in make_prior, so d0 follows the new means too.
    unscaled = init.deterministic_orthogonal_(get_recipe(prior_kind="mog", num_particles=32, z_dim=3).make_prior())
    assert torch.equal(unscaled.d0, calibrate_mog_sigma(unscaled.means(), 0)[1])
    fixed = MoGParticlePrior(16, 2, sigma=0.3)
    init.deterministic_orthogonal_(fixed)
    assert float(fixed.sigma) == pytest.approx(0.3)
    frozen = get_recipe(num_particles=8, prior_update="frozen").make_prior()
    saved = frozen.z.clone()
    init.deterministic_orthogonal_(frozen)
    assert torch.equal(saved, frozen.z)



def test_public_modules_fully_declared():
    # The downstream check docs/api.md recommends, applied to our own modules.
    nets = (BatchDistanceDiscriminator(), LinearSkipDiscriminator(), ParticlePrior(16, 2),
            MoGParticlePrior(16, 2, sigma=0.3), nn.TransformerEncoderLayer(16, 2, dim_feedforward=32, batch_first=True))
    for net in nets:
        undeclared = [p for p, spec in init.declarations(net).items() if spec is None]
        assert not undeclared, (type(net).__name__, undeclared)


def test_values_depend_on_position_in_the_module_passed():
    net = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
    whole = init.deterministic_orthogonal_(copy.deepcopy(net))
    alone = init.deterministic_orthogonal_(copy.deepcopy(net[1]))
    assert not torch.equal(whole[1].weight, alone.weight)

@pytest.mark.parametrize("layer", [
    lambda: nn.Linear(1, 1), lambda: nn.Linear(7, 11),
    lambda: nn.Conv2d(4, 6, 3, groups=2), lambda: nn.ConvTranspose2d(4, 6, 3),
])
def test_matrix_gram_and_declared_scale(layer):
    module = init.deterministic_orthogonal_(layer().double())
    matrix = module.weight.flatten(1)
    rows, cols = matrix.shape
    gain2 = max(rows, cols) / (3 * cols)
    gram = matrix.T @ matrix if rows >= cols else matrix @ matrix.T
    torch.testing.assert_close(gram, torch.eye(min(rows, cols), dtype=torch.float64) * gain2,
                               atol=1e-14, rtol=1e-13)


def test_transformer_layers_and_lora_preserve_special_parameters():
    first = init.deterministic_orthogonal_(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16).double())
    second = init.deterministic_orthogonal_(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16).double())
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert torch.equal(first.norm1.weight, torch.ones_like(first.norm1.weight))
    assert torch.count_nonzero(first.self_attn.in_proj_bias) == 0
    embedded = init.deterministic_orthogonal_(nn.Embedding(9, 4, padding_idx=0))
    assert torch.count_nonzero(embedded.weight[0]) == 0
    adapter = nn.ModuleDict({"base": nn.Linear(8, 6).requires_grad_(False),
                             "A": nn.Linear(8, 2, bias=False),
                             "B": nn.Linear(2, 6, bias=False)})
    nn.init.zeros_(adapter["B"].weight)
    saved = adapter["base"].weight.clone()
    init.deterministic_orthogonal_(adapter)
    assert torch.equal(saved, adapter["base"].weight)
    assert torch.count_nonzero(adapter["B"].weight) == 0
    assert torch.count_nonzero(adapter["A"].weight) > 0


class Hopfield(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.patterns = nn.Parameter(torch.randn(6, 4))
        self.beta = nn.Parameter(torch.tensor(2.0))


class Gated(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.gate = nn.Parameter(torch.full((4,), .5))


def test_strict_raises_before_writing_and_register_extends(monkeypatch):
    monkeypatch.setattr(init, "_REGISTRY", dict(init._REGISTRY))
    net = nn.Sequential(nn.Linear(4, 4), Hopfield())
    before = {k: p.clone() for k, p in net.named_parameters()}
    with pytest.raises(ValueError, match=r"'1\.patterns' \(Hopfield\), '1\.beta' \(Hopfield\)"):
        init.deterministic_orthogonal_(net)
    assert all(torch.equal(before[k], p) for k, p in net.named_parameters())
    assert init.declarations(net)["1.patterns"] is None
    init.deterministic_orthogonal_(net, strict=False)
    assert torch.equal(net[1].patterns, before["1.patterns"]) and not torch.equal(net[0].weight, before["0.weight"])

    init.register(Hopfield, {"patterns": init.Normal(0., 1.), "beta": init.KEEP})
    assert init.declarations(net)["1.beta"] is init.KEEP
    init.deterministic_orthogonal_(net)
    assert not torch.equal(net[1].patterns, before["1.patterns"]) and net[1].beta.item() == 2.0

    # Subclasses inherit declarations and declare only what they add.
    with pytest.raises(ValueError, match="'gate' \\(Gated\\)"):
        init.deterministic_orthogonal_(Gated())
    init.register(Gated, lambda m: {"gate": init.KEEP})
    assert set(init.declarations(Gated())) == {"weight", "bias", "gate"}
    init.deterministic_orthogonal_(Gated())


def test_invalid_declarations_seed_and_lazy_modules(monkeypatch):
    monkeypatch.setattr(init, "_REGISTRY", dict(init._REGISTRY))
    init.register(Hopfield, {"pattern": init.KEEP})
    with pytest.raises(ValueError, match="does not have"):
        init.deterministic_orthogonal_(Hopfield())
    init.register(Hopfield, {"patterns": "normal"})
    with pytest.raises(TypeError, match="Uniform, Normal, R2Normal or KEEP"):
        init.deterministic_orthogonal_(Hopfield())
    with pytest.raises(TypeError):
        init.register(int, {})
    with pytest.raises(ValueError, match="seed"):
        init.deterministic_orthogonal_(nn.Linear(2, 2), seed=-1)
    with pytest.raises(ValueError, match="materialize"):
        init.deterministic_orthogonal_(nn.LazyLinear(2))


def test_explicit_api_matches_measured_hook_on_standard_networks():
    # Isolate the research hook; ordinary API calls never install it.
    script = r'''
import copy
import torch
from torch import nn
from particlegan import BatchDistanceDiscriminator, init
from benchmarks.init_research.batch_feature_init import install, uninstall
torch.set_num_threads(1)
torch.manual_seed(0)
install()
g = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
d = BatchDistanceDiscriminator(hidden_dim=8, n_hidden=1)
a, b = copy.deepcopy(g), copy.deepcopy(d)
torch.optim.Adam(g.parameters())
torch.optim.Adam(d.parameters())
uninstall()
init.deterministic_orthogonal_(a, seed=0)
init.deterministic_orthogonal_(b, seed=1)
for x, y in zip(list(g.parameters()) + list(d.parameters()), list(a.parameters()) + list(b.parameters())):
    assert torch.equal(x, y)
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_checkpoints_that_recorded_the_old_init_field_still_load():
    recipe = get_recipe(num_particles=8, batch_size=4, total_steps=4)
    def build():
        return GANTrainer(recipe, nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)))
    original = build()
    x = torch.arange(8).reshape(4, 2).float() / 8
    original.step(x)
    for value in (None, "batch_feature_zero"):
        checkpoint = copy.deepcopy(original.state_dict())
        checkpoint["recipe"]["initialization"] = value
        build().load_state_dict(checkpoint)
    checkpoint["recipe"]["initialization"] = "unknown"
    with pytest.raises(ValueError, match="recipe"):
        build().load_state_dict(checkpoint)


def test_deterministic_orthogonal_handles_rmsnorm_without_bias():
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.RMSNorm(8),
        nn.RMSNorm(8, elementwise_affine=False),
        nn.Linear(8, 2),
    )
    decls = init.declarations(net)
    assert decls["1.weight"] is init.KEEP
    assert "1.bias" not in decls
    before_scale = net[1].weight.detach().clone()
    init.deterministic_orthogonal_(net, seed=0)
    assert torch.equal(net[1].weight, before_scale)
