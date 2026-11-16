"""Software-only independent-game and complete-state controls; no quality claim."""
import pytest
import torch

from benchmarks.toy_audit import api_model_misgan as api
from experiments.forge.state import state_digest
from particlegan.bispf import _LAST_THERMAL_SCALE
from particlegan.recipes import learning_rate_scales


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(**overrides):
    case = api.list_cases()[0]
    recipe = api.task_recipe(case).replace(**overrides)
    return api.MisGANFixture(case, recipe=recipe, max_steps=4)


def test_every_critic_owns_its_optimizer_clock_penalty_and_thermal_readers():
    f = fixture(loss="relativistic")
    f.step(); f.step()
    records = [optimizer.record for optimizer in f.critic_optimizers.values()]
    assert len({id(record) for record in records}) == 3
    for role, optimizer in f.critic_optimizers.items():
        assert optimizer.record.observed_steps == optimizer.record.calls == 2
        assert optimizer.critic is f.D[role]
        assert f.penalties[role].optimizer is optimizer
        assert f.losses[role].thermal_source is optimizer.record
        assert optimizer.record.legacy_thermal_channel is False
    assert f.regularizer.thermal_source is f.critic_optimizers["data"].record


def test_altering_mask_history_cannot_change_other_games_gradient_scale():
    f = fixture(loss="relativistic")
    # The public dense-loss gate activates at 1024; use its actual declared law.
    def gradient(role):
        fake = torch.linspace(-1., 1., 1024, requires_grad=True)
        real = torch.zeros_like(fake)
        return torch.autograd.grad(f.losses[role].g_loss(fake, real), fake)[0]
    for optimizer in f.critic_optimizers.values():
        optimizer.record._thermal_scale = .5
    before = {role: gradient(role) for role in f.D}
    prior_before = f.regularizer(f.prior.z).detach().clone()
    record = f.critic_optimizers["mask"].record
    record._thermal_scale = .1
    record.calls = 8000
    record.ema_residual = .0001
    assert torch.equal(before["data"], gradient("data"))
    assert torch.equal(before["imputer"], gradient("imputer"))
    assert not torch.equal(before["mask"], gradient("mask"))
    assert torch.equal(prior_before, f.regularizer(f.prior.z).detach())


def test_same_intact_schedule_reaches_all_critic_owners():
    f = fixture(lr_anneal_start=0., lr_floor=.05, network_lr_floor=.05,
                beta2_end=.9, optimizer_family="adam", optimizer_convolution="none", optimizer_convolution_rule="polar",
                optimizer_dense_profile="default", optimizer_smoothing=0., constraint_geometry_mode="none")
    f.step(); f.step()
    network, _ = learning_rate_scales(1, f.recipe)
    for role, optimizer in f.critic_optimizers.items():
        assert [group["lr"] for group in optimizer.param_groups] == [base * network for base in f.critic_initial_lrs[role]]
        assert optimizer.param_groups[0]["betas"] == f.opt_d.param_groups[0]["betas"]
        assert f.penalties[role].regularizer.coeff == f.penalty.regularizer.coeff


@pytest.mark.parametrize("regularizer", ["sbnp", "moment_tether", "sigreg"])
def test_fresh_object_restores_complete_state_and_exact_next_update(regularizer):
    f = fixture(prior_regularizer=regularizer)
    f.step()
    saved = f.state_dict()
    frozen = state_digest(saved)
    f.step()
    expected = state_digest(f.state_dict())
    assert state_digest(saved) == frozen
    resumed = fixture(prior_regularizer=regularizer)
    resumed.load_state_dict(saved)
    assert state_digest(resumed.state_dict()) == frozen
    resumed.step()
    assert state_digest(resumed.state_dict()) == expected
    assert state_digest(saved) == frozen


@pytest.mark.parametrize("defect", ["old_version", "missing_aux_optimizer", "bad_aux_moment",
                                    "contradictory_history", "bad_aux_cooler", "bad_aux_weight",
                                    "bad_rng", "bad_regularizer", "changed_recipe"])
def test_bad_auxiliary_state_is_rejected_before_any_live_mutation(defect):
    f = fixture(prior_regularizer="sigreg"); f.step()
    state = f.state_dict()
    if defect == "old_version":
        state["version"] = "misgan-four-mode-v1"
    elif defect == "missing_aux_optimizer":
        del state["critic_optimizers"]["mask"]
    elif defect == "bad_aux_moment":
        first = next(iter(state["critic_optimizers"]["imputer"]["state"].values()))
        first["exp_avg"] = torch.zeros(1)
    elif defect == "contradictory_history":
        state["penalties"]["mask"]["record"]["calls"] += 1
    elif defect == "bad_aux_cooler":
        state["penalties"]["mask"]["cooler"]["ema_residual"] = float("nan")
    elif defect == "bad_aux_weight":
        state["models"]["D"]["mask.0.weight"] = torch.zeros(1)
    elif defect == "bad_rng":
        state["streams"]["model_rng"] = torch.ones(1, dtype=torch.uint8)
    elif defect == "bad_regularizer":
        key = next(k for k, v in state["models"]["regularizer"].items() if isinstance(v, torch.Tensor))
        state["models"]["regularizer"][key] = torch.zeros(1)
    else:
        state["recipe"]["lr"] *= 2
    before = state_digest(f.state_dict())
    ambient = torch.get_rng_state().clone()
    thermal = dict(_LAST_THERMAL_SCALE)
    with pytest.raises(ValueError, match="checkpoint"):
        f.load_state_dict(state)
    assert state_digest(f.state_dict()) == before
    assert torch.equal(torch.get_rng_state(), ambient)
    assert dict(_LAST_THERMAL_SCALE) == thermal


def test_interrupted_update_requires_complete_restore(monkeypatch):
    f = fixture(); saved = f.state_dict()
    with monkeypatch.context() as patch:
        def fail():
            raise RuntimeError("interrupted auxiliary optimizer")
        patch.setattr(f.critic_optimizers["mask"], "step", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            f.step()
    with pytest.raises(RuntimeError, match="completed update boundary"):
        f.state_dict()
    with pytest.raises(RuntimeError, match="restoration"):
        f.step()
    f.load_state_dict(saved)
    assert state_digest(f.state_dict()) == state_digest(saved)
    f.step()


@pytest.mark.parametrize("overrides", [dict(input_noise_std=.1), dict(output_noise_std=.1),
                                       dict(kinetic_transport_weight=.1), dict(kinetic_transport_local_weight=.1),
                                       dict(training_precision="bf16_channels_last"), dict(kappa_radial=.5)])
def test_unsupported_mechanism_refuses_before_model_construction(monkeypatch, overrides):
    def no_models():
        raise AssertionError("unsupported recipe reached model construction")
    monkeypatch.setattr(api, "Generators", no_models)
    with pytest.raises(ValueError, match="intact recipe mechanisms"):
        fixture(**overrides)
