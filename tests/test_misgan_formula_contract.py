"""Software contracts for formula-faithful MisGAN execution, not quality qualification."""
from copy import deepcopy

import pytest
import torch

from benchmarks.toy_audit import api_conditionals as api
from experiments.forge.state import state_digest
from particlegan import get_recipe


CASES = ("api-misgan-block-recovery", "api-misgan-block-neural-imputer",
         "api-misgan-mcar-p50-recovery", "api-misgan-mcar-p80-anchored")


def fixture(case_id=CASES[0], *, family="bispf", steps=3, **overrides):
    case = deepcopy(api.CASES[case_id])
    recipe = get_recipe(family, batch_size=8, num_particles=256, z_dim=8,
                        total_steps=case["default_steps"], **overrides)
    owner = api.MisGANNeuralImputerFixture if "imputer" in case_id or "anchored" in case_id else api.MisGANRecoveryFixture
    return owner(case, recipe=recipe, batch_size=8, max_steps=steps, seed=0, device="cpu")


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("field,value", [("model", "ddgan"), ("training_precision", "bf16_channels_last")])
def test_unsupported_model_and_precision_refuse_before_optimizer_construction(monkeypatch, case_id, field, value):
    from particlegan import Recipe
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("unsupported recipe reached optimizer construction")

    monkeypatch.setattr(Recipe, "make_generator_optimizer", forbidden_optimizer)
    monkeypatch.setattr(Recipe, "make_critic_optimizer", forbidden_optimizer)
    with pytest.raises(UnsupportedRecipeOverrides, match=field):
        fixture(case_id, **{field: value})


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_neural_imputer_calls_live_network_and_never_ppost(monkeypatch):
    f = fixture(CASES[1])
    # A stale attribute from an old in-process caller must not change the sampler.
    f._subspace_ready = True
    calls = []
    def no_ppost(*args, **kwargs):
        raise AssertionError("the no-ppost question invoked posterior resampling")
    monkeypatch.setattr(api, "misgan_ppost_indices", no_ppost)
    monkeypatch.setattr(f.ema_G_i, "forward", no_ppost)
    def live_imputer(condition):
        calls.append(condition.detach().clone())
        return condition[:, :8] + 2
    monkeypatch.setattr(f.G_i, "forward", live_imputer)
    x = torch.ones(2, 8)
    mask = torch.zeros_like(x)
    mask[:, :2] = 1
    def score(predict, problem, eval_seed):
        samples = predict(x, mask, torch.Generator().manual_seed(0), draws=2)
        assert torch.equal(samples[:, :, :2], x[:, :2].expand(2, -1, -1))
        assert torch.equal(samples[:, :, 2:], torch.full((2, 2, 6), 2.))
        return {"passed": False, "failed_bounds": ["deliberately_wrong_imputer"]}
    monkeypatch.setattr(api, "score_misgan_neural_imputer", score)
    observed = f.observe()
    assert observed["failed_bounds"] == ["deliberately_wrong_imputer"]
    assert observed["observed_sampling"]["posterior_resampling"] is False
    assert observed["observed_sampling"]["predictor"] == "G_i"
    assert observed["observed_sampling"]["scoring_weights"] == "live"
    assert len(calls) == 2


def test_neural_imputer_metric_depends_on_live_gi_not_ema(monkeypatch):
    f = fixture(CASES[1])
    def score(predict, problem, eval_seed):
        x = torch.zeros(2, 8)
        mask = torch.zeros_like(x)
        result = predict(x, mask, torch.Generator().manual_seed(0), draws=1)
        return {"value": result.detach().clone()}
    monkeypatch.setattr(api, "score_misgan_neural_imputer", score)
    first = f.observe()["value"]
    with torch.no_grad():
        f.ema_G_i[-1].bias.add_(100)
    assert torch.equal(first, f.observe()["value"])
    with torch.no_grad():
        f.G_i[-1].bias.add_(1)
    assert torch.allclose(first + 1, f.observe()["value"])


def test_recovery_observer_uses_live_generator_and_prior(monkeypatch):
    f = fixture()
    def score(x, m, problem, **kwargs):
        return {"x": x.detach().clone(), "m": m.detach().clone()}
    f._scorer = score
    first = f.observe(n=8)
    with torch.no_grad():
        f.ema_G_x[-1].bias.add_(100)
        f.ema_G_m[-2].bias.add_(100)
        f.ema_prior_x.z.add_(100)
    assert state_digest(first) == state_digest(f.observe(n=8))
    with torch.no_grad():
        f.G_x[-1].bias.add_(1)
    assert torch.allclose(first["x"] + 1, f.observe(n=8)["x"])


@pytest.mark.parametrize("case_id", CASES[:2])
def test_late_update_has_no_unowned_model_or_particle_writes(case_id, monkeypatch):
    f = fixture(case_id, steps=7000)
    f.completed_steps = 3500  # Former hidden solver activation boundary.
    before = state_digest({"G": f.G.state_dict(), "priors": f.priors.state_dict()})
    for optimizer in (f.opt_g, *f.critic_optimizers.values()):
        monkeypatch.setattr(optimizer, "step", lambda: None)
    f.step()
    assert state_digest({"G": f.G.state_dict(), "priors": f.priors.state_dict()}) == before
    assert not hasattr(f, "_subspace_cache")
    assert not hasattr(f, "_subspace_ready")


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("family", ["bispf", "bcap"])
def test_fresh_fixture_restores_all_owned_state_and_next_update(case_id, family):
    uninterrupted = fixture(case_id, family=family)
    uninterrupted.step()
    saved = uninterrupted.state_dict()
    resumed = fixture(case_id, family=family)
    resumed.load_state_dict(saved)
    assert state_digest(saved) == state_digest(resumed.state_dict())
    assert uninterrupted.step() == resumed.step()
    assert state_digest(uninterrupted.state_dict()) == state_digest(resumed.state_dict())
    # The returned checkpoint must not alias subsequently mutated live weights.
    assert state_digest(saved) != state_digest(uninterrupted.state_dict())


@pytest.mark.parametrize("case_id", CASES[:2])
def test_each_critic_has_its_own_public_record_and_updated_clock(case_id):
    f = fixture(case_id)
    f.step(); f.step()
    records = [opt.record for opt in f.critic_optimizers.values()]
    assert len({id(r) for r in records}) == len(records)
    assert all(r.observed_steps == 2 and r.calls == 2 for r in records)
    for role, optimizer in f.critic_optimizers.items():
        assert getattr(f, "penalty_" + role).optimizer is optimizer
        assert getattr(f, "loss_" + role).thermal_source is optimizer.record


@pytest.mark.parametrize("defect", ["old_schema", "missing_rng", "bad_rng", "different_recipe", "bad_model"])
def test_bad_checkpoint_is_rejected_without_mutating_live_fixture(defect):
    f = fixture(CASES[1]); f.step()
    before = state_digest(f.state_dict())
    state = f.state_dict()
    if defect == "old_schema":
        state["version"] = api.VERSION
    elif defect == "missing_rng":
        del state["streams"]["noise_rng"]
    elif defect == "bad_rng":
        state["streams"]["noise_rng"] = torch.ones(1, dtype=torch.uint8)
    elif defect == "different_recipe":
        state["recipe"]["lr"] *= 2
    else:
        state["models"]["G_i"]["0.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        f.load_state_dict(state)
    assert state_digest(f.state_dict()) == before


def test_interrupted_update_requires_restoration_of_complete_checkpoint(monkeypatch):
    f = fixture(); saved = f.state_dict()
    with monkeypatch.context() as patch:
        def fail():
            raise RuntimeError("injected optimizer interruption")
        patch.setattr(f.opt_g, "step", fail)
        with pytest.raises(RuntimeError, match="injected"):
            f.step()
    with pytest.raises(RuntimeError, match="completed update boundary"):
        f.state_dict()
    with pytest.raises(RuntimeError, match="complete checkpoint restore"):
        f.step()
    f.load_state_dict(saved)
    assert state_digest(saved) == state_digest(f.state_dict())
    assert f.step()["step"] == 1


def test_model_random_stream_is_private_to_fixture():
    f = fixture()
    before = torch.get_rng_state().clone()
    f.step()
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("family", ["e22", "atlas", "k3p", "ka2"])
def test_unsupported_full_recipe_is_refused_without_silent_replacement(family):
    recipe = get_recipe(family, batch_size=8, num_particles=256, z_dim=8)
    declaration = recipe.to_dict()
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    with pytest.raises(UnsupportedRecipeOverrides, match="cannot execute the intact recipe mechanisms"):
        api.MisGANRecoveryFixture(recipe=recipe, max_steps=1)
    assert recipe.to_dict() == declaration


def test_frozen_prior_stays_frozen_in_actual_training():
    f = fixture(prior_update="frozen")
    before = state_digest(f.priors.state_dict())
    f.step()
    assert state_digest(f.priors.state_dict()) == before


def test_zero_ema_decay_keeps_ema_history_unchanged():
    f = fixture(ema_decay=0)
    before = state_digest(f.ema_G_x.state_dict())
    f.step()
    assert state_digest(f.ema_G_x.state_dict()) == before


@pytest.mark.parametrize("family", ["k3p", "ka2"])
def test_explicit_supported_scheduled_candidate_restores_exactly(family):
    delta = dict(input_noise_std=0., output_noise_std=0., latent_damping_max_rate=0.)
    f = fixture(family=family, **delta); f.step()
    resumed = fixture(family=family, **delta)
    resumed.load_state_dict(f.state_dict())
    assert f.step() == resumed.step()
    assert state_digest(f.state_dict()) == state_digest(resumed.state_dict())


def test_stochastic_regularizer_history_is_part_of_fresh_resume():
    f = fixture(prior_regularizer="sigreg", prior_reg=.01); f.step()
    saved = f.state_dict()
    resumed = fixture(prior_regularizer="sigreg", prior_reg=.01)
    resumed.load_state_dict(saved)
    assert f.step() == resumed.step()
    assert state_digest(f.state_dict()) == state_digest(resumed.state_dict())


def test_training_cannot_be_rebound_to_different_recipe():
    f = fixture(); f.step()
    before = state_digest(f.state_dict())
    with pytest.raises(ValueError, match="new execution identity"):
        f.rebind_trainer(f.recipe.replace(lr=f.recipe.lr / 2))
    assert state_digest(f.state_dict()) == before


def test_recipe_resources_cannot_silently_replace_host_capacity():
    recipe = get_recipe("bispf", batch_size=8, num_particles=32, z_dim=4)
    with pytest.raises(ValueError, match="explicitly declared host resources"):
        api.MisGANRecoveryFixture(recipe=recipe)


def test_supplied_prior_matches_task_owned_wrapped_cloud_and_constructor_streams():
    original = fixture()
    recipe = original.recipe
    restored = api.MisGANRecoveryFixture(
        deepcopy(original.case), recipe=recipe, prior=deepcopy(original.prior_x),
        batch_size=8, max_steps=3, seed=0)
    assert state_digest(original.state_dict()) == state_digest(restored.state_dict())
    assert original.case["initial_prior"]["kind"] == "particle_cloud"
    assert original.case["initial_prior"]["sigma"] == 0
    assert original.case["initial_prior"]["transform"]["sigma"] == .025


@pytest.mark.parametrize("defect", ["kind", "capacity", "dimension", "noise_sigma", "bare_cloud",
                                   "radial_noise", "init_scale", "initial_rows", "dtype"])
def test_supplied_prior_law_is_checked_before_any_optimizer(defect, monkeypatch):
    from particlegan import MoGParticlePrior, ParticlePrior, NormalNoiseManipulator
    original = fixture()
    supplied = deepcopy(original.prior_x)
    if defect == "kind":
        supplied = MoGParticlePrior(256, 8, sigma=.025, standardize=False)
    elif defect in ("capacity", "dimension"):
        supplied = ParticlePrior(128 if defect == "capacity" else 256, 4 if defect == "dimension" else 8)
        supplied = supplied.manipulate(NormalNoiseManipulator(.025))
    elif defect == "noise_sigma":
        supplied.manipulators[0].sigma = .05
    elif defect == "bare_cloud":
        supplied = supplied.base
    elif defect == "radial_noise":
        supplied.manipulators[0].kappa_radial = .5
    elif defect == "init_scale":
        supplied.base.init_std = 2.
    elif defect == "initial_rows":
        with torch.no_grad():
            supplied.z.add_(1.)
    else:
        supplied = supplied.double()
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer construction preceded prior-law validation")
    monkeypatch.setattr(type(original.recipe), "make_generator_optimizer", forbidden)
    monkeypatch.setattr(type(original.recipe), "make_critic_optimizer", forbidden)
    before = state_digest(supplied.state_dict())
    with pytest.raises(ValueError, match="MisGAN supplied prior"):
        api.MisGANRecoveryFixture(deepcopy(original.case), recipe=original.recipe,
                                 prior=supplied, batch_size=8, max_steps=3, seed=0)
    assert state_digest(supplied.state_dict()) == before


def test_positive_width_mog_is_its_own_explicit_task_cohort():
    case = deepcopy(api.CASES[CASES[0]])
    case["initial_prior"].update(kind="mog", sigma=.025, transform=None)
    recipe = get_recipe("bispf", batch_size=8, num_particles=256, z_dim=8,
                        prior_kind="mog", standardize=False)
    first = api.MisGANRecoveryFixture(case, recipe=recipe, batch_size=8, max_steps=3)
    supplied = deepcopy(first.prior_x)
    second = api.MisGANRecoveryFixture(deepcopy(case), recipe=recipe,
                                      prior=supplied, batch_size=8, max_steps=3)
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())
    assert first.step() == second.step()
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())


@pytest.mark.parametrize("defect", ["sigma", "standardize", "recipe_storage"])
def test_mog_storage_law_and_recipe_conflicts_are_rejected(defect):
    case = deepcopy(api.CASES[CASES[0]])
    case["initial_prior"].update(kind="mog", sigma=.025, transform=None)
    recipe = get_recipe("bispf", batch_size=8, num_particles=256, z_dim=8,
                        prior_kind="mog", standardize=False)
    original = api.MisGANRecoveryFixture(case, recipe=recipe, batch_size=8, max_steps=3)
    supplied = deepcopy(original.prior_x)
    if defect == "sigma":
        supplied.set_sigma(.05)
    elif defect == "standardize":
        supplied.standardize = True
    else:
        recipe = recipe.replace(prior_kind="particles", sigma_rel=0.)
    with pytest.raises(ValueError, match="MisGAN (supplied prior|recipe/task prior)"):
        api.MisGANRecoveryFixture(deepcopy(case), recipe=recipe, prior=supplied,
                                 batch_size=8, max_steps=3)


def test_manipulator_regularizer_is_explicitly_unsupported_before_optimizer(monkeypatch):
    from particlegan.priors import PriorManipulator
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    class Regularizer(PriorManipulator):
        def regularize(self, raw_rows, *, sampled_codes=None, indices=None, generated=None,
                       logits=None, **context):
            raise AssertionError("unsupported regularizer was silently invoked")
    original = fixture()
    supplied = deepcopy(original.prior_x).manipulate(Regularizer())
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported manipulator reached optimizer construction")
    monkeypatch.setattr(type(original.recipe), "make_generator_optimizer", forbidden)
    monkeypatch.setattr(type(original.recipe), "make_critic_optimizer", forbidden)
    with pytest.raises(UnsupportedRecipeOverrides, match="manipulator regularization context"):
        api.MisGANRecoveryFixture(deepcopy(original.case), recipe=original.recipe, prior=supplied,
                                 batch_size=8, max_steps=3)


@pytest.mark.parametrize("copy_name", ["optimizer", "penalty"])
def test_conflicting_copies_of_critic_history_are_rejected_atomically(copy_name):
    f = fixture(); f.step()
    state = f.state_dict()
    before = state_digest(state)
    record = (state["optimizers"]["m"]["regularizer"]["record"] if copy_name == "optimizer"
              else state["penalties"]["m"]["record"])
    record["calls"] += 1
    with pytest.raises(ValueError, match="invalid complete MisGAN checkpoint"):
        f.load_state_dict(state)
    assert state_digest(f.state_dict()) == before


def test_caller_cannot_change_bound_task_prior_through_a_metadata_alias():
    case = deepcopy(api.CASES[CASES[0]])
    f = api.MisGANRecoveryFixture(case, max_steps=1)
    before = state_digest(f.state_dict())
    case["initial_prior"]["transform"]["sigma"] = .5
    assert state_digest(f.state_dict()) == before
