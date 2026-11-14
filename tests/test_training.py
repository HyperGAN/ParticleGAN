from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, GANTrainer, Recipe, get_recipe, learning_rate_scale, learning_rate_scales


def make_trainer(*, preset="gan", particles=12, steps=8, buffers=False, dropout=False, **overrides):
    recipe = get_recipe(preset, num_particles=particles, z_dim=3,
                        batch_size=6, total_steps=steps, **overrides)
    generator = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8) if buffers else nn.Identity(),
                              nn.LeakyReLU(.2), nn.Dropout(.25) if dropout else nn.Identity(),
                              nn.Linear(8, 2))
    discriminator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2),
                                  nn.Dropout(.2) if dropout else nn.Identity(), nn.Linear(8, 1))
    return GANTrainer(recipe, generator, discriminator)


def assert_models_equal(left, right):
    for name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key, tensor in left["models"][name].items():
            assert torch.equal(tensor, right["models"][name][key]), (name, key)


def assert_checkpoint_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_checkpoint_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_checkpoint_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["none", "coherent_G_mean_v1", "coherent_G_damped_v2"])
@pytest.mark.parametrize("warm_destination", [False, True])
def test_gradient_response_trainer_checkpoint_exact_continuation(mode, warm_destination, tmp_path):
    trainer = make_trainer(preset="bcap", generator_gradient_response=mode)
    real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    initial = trainer.state_dict()
    if mode == "none":
        assert "gradient_response" not in initial
    else:
        assert initial["gradient_response"]["packet"] is None
    trainer.step(real)
    trainer.step(real.flip(0))
    saved = trainer.state_dict()
    torch.save(saved, tmp_path / "trainer.pt")
    saved = torch.load(tmp_path / "trainer.pt", map_location="cpu", weights_only=True)
    preserved = deepcopy(saved)
    trainer.step(real)
    expected = trainer.state_dict()

    restored = make_trainer(preset="bcap", generator_gradient_response=mode)
    if warm_destination:
        restored.step(real + 1)
    restored.load_state_dict(saved)
    assert_checkpoint_equal(saved, restored.state_dict())
    restored.step(real)
    assert_checkpoint_equal(expected, restored.state_dict())
    assert_checkpoint_equal(saved, preserved)
    restored.load_state_dict(initial)
    assert_checkpoint_equal(initial, restored.state_dict())


@pytest.mark.parametrize("mode", ["coherent_G_mean_v1", "coherent_G_damped_v2"])
def test_enabled_gradient_response_rejects_legacy_missing_history_without_mutation(mode):
    trainer = make_trainer(preset="bcap", generator_gradient_response=mode)
    trainer.step(torch.zeros(6, 2))
    legacy = trainer.state_dict()
    del legacy["gradient_response"]
    trainer.step(torch.ones(6, 2))
    before = trainer.state_dict()
    with pytest.raises(ValueError, match="gradient response history"):
        trainer.load_state_dict(legacy)
    assert_checkpoint_equal(before, trainer.state_dict())


@pytest.mark.parametrize("broken", [
    "schema", "mode", "missing_field", "beta", "min_gain", "power", "response",
    "packet_none", "packet_type", "means_count", "means_shape", "means_dtype",
    "means_nonfinite", "second_shape", "second_negative", "second_nonfinite",
    "weight", "weight2", "count", "last", "last_nonfinite", "history_clock", "future_clock",
])
def test_malformed_gradient_response_checkpoint_is_atomic(broken):
    trainer = make_trainer(preset="bcap", generator_gradient_response="coherent_G_mean_v1")
    trainer.step(torch.zeros(6, 2))
    bad = trainer.state_dict()
    response = bad["gradient_response"]
    packet = response["packet"]
    if broken == "schema":
        response["schema"] = 99
    elif broken == "mode":
        response["mode"] = "unknown"
    elif broken == "missing_field":
        del response["packet"]
    elif broken == "beta":
        response["beta"] = 1.0
    elif broken == "min_gain":
        response["min_gain"] = float("nan")
    elif broken == "power":
        response["power"] = -1.0
    elif broken == "response":
        response["response"]["applied_steps"] = -1
    elif broken == "packet_none":
        response["packet"] = None
    elif broken == "packet_type":
        response["packet"] = []
    elif broken == "means_count":
        packet["means"].pop()
    elif broken == "means_shape":
        packet["means"][0] = packet["means"][0].reshape(-1)
    elif broken == "means_dtype":
        packet["means"][0] = packet["means"][0].float()
    elif broken == "means_nonfinite":
        packet["means"][0].fill_(float("nan"))
    elif broken == "second_shape":
        packet["second"] = packet["second"].reshape(1)
    elif broken == "second_negative":
        packet["second"].fill_(-1)
    elif broken == "second_nonfinite":
        packet["second"].fill_(float("inf"))
    elif broken in ("weight", "weight2"):
        packet[broken] = 0.0
    elif broken == "count":
        packet["count"] = True
    elif broken == "last":
        packet["last"] = {}
    elif broken == "last_nonfinite":
        packet["last"]["gain"] = float("nan")
    elif broken == "future_clock":
        response["response"]["applied_steps"] = packet["count"] + 1
    else:
        response["response"]["applied_steps"] = 0
    trainer.step(torch.ones(6, 2))
    before = trainer.state_dict()
    with pytest.raises(ValueError, match="gradient response"):
        trainer.load_state_dict(bad)
    assert_checkpoint_equal(before, trainer.state_dict())


@pytest.mark.parametrize("operation", ["save", "restore"])
def test_trainer_checkpoint_requires_boundary_inside_generator_callback(operation):
    trainer = make_trainer()
    real = torch.zeros(6, 2)
    saved = trainer.state_dict()
    calls = []

    def callback():
        before = trainer._state_dict()
        assert trainer.policy._phase == "generator"
        with pytest.raises(RuntimeError, match="completed update boundary"):
            trainer.state_dict() if operation == "save" else trainer.load_state_dict(saved)
        assert_checkpoint_equal(before, trainer._state_dict())
        assert trainer.policy._phase == "generator"
        calls.append(1)
        return real

    trainer.step(real, generator_real=callback)
    assert calls == [1]
    assert trainer.state_dict()["completed_steps"] == 1
    trainer.load_state_dict(saved)
    assert_checkpoint_equal(saved, trainer.state_dict())


def test_trainer_checkpoint_rejects_caller_owned_active_policy():
    trainer = make_trainer(preset="bcap")
    saved = trainer.state_dict()
    trainer.policy.begin_step(torch.zeros(6, 2))
    assert not trainer._in_training_step
    with pytest.raises(RuntimeError, match="completed update boundary"):
        trainer.state_dict()
    with pytest.raises(RuntimeError, match="completed update boundary"):
        trainer.load_state_dict(saved)
    trainer.policy.abort_step()
    trainer.load_state_dict(saved)
    assert_checkpoint_equal(saved, trainer.state_dict())


def test_winning_recipe_is_the_common_default():
    assert get_recipe() == get_recipe(DEFAULT_RECIPE)
    winner = get_recipe()
    assert winner.name == DEFAULT_RECIPE
    assert Recipe.from_dict(winner.to_dict()) == winner
    assert isinstance(make_trainer(), GANTrainer)


@pytest.mark.parametrize("prior_kind", ["particles", "mog"])
@pytest.mark.parametrize("global_stream", [False, True])
def test_shared_training_stream_preserves_sampling_isolation_and_checkpoint_resume(prior_kind, global_stream):
    def build():
        torch.manual_seed(51)
        recipe = get_recipe(num_particles=8, z_dim=2, total_steps=4,
                            prior_kind=prior_kind, sigma_rel=.025 if prior_kind == "mog" else 0,
                            standardize=False,
                            input_noise_std=.02, output_noise_std=.03, output_noise_warmup=0)
        shared = torch.default_generator if global_stream else torch.Generator().manual_seed(19)
        options = dict(latent_generator=shared, penalty_generator=shared,
                       noise_generator=shared, input_noise_generator=shared)
        if prior_kind == "mog":
            options["prior_noise_generator"] = shared
        return GANTrainer(recipe, nn.Linear(2, 2), nn.Linear(2, 1), **options)

    trainer = build()
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    trainer.step(real)
    before = trainer.latent_generator.get_state().clone()
    trainer.sample(7, output_noise=True)
    assert torch.equal(before, trainer.latent_generator.get_state())
    with pytest.raises(ValueError, match="separate from training"):
        trainer.sample(7, generator=trainer.penalty_generator)
    checkpoint = trainer.state_dict()
    trainer.step(real)
    expected = trainer.state_dict()
    restored = build()
    restored.load_state_dict(checkpoint)
    assert restored.latent_generator is restored.penalty_generator is restored.noise_generator
    restored.step(real)
    assert_checkpoint_equal(expected, restored.state_dict())
    before = restored.state_dict()
    bad = deepcopy(before)
    bad["streams"]["penalty_generator"] = torch.Generator().manual_seed(91).get_state()
    with pytest.raises(ValueError, match="RNG state"):
        restored.load_state_dict(bad)
    assert_checkpoint_equal(before, restored.state_dict())
    if global_stream:
        bad = deepcopy(before)
        for name in bad["streams"]:
            if getattr(restored, name) is torch.default_generator:
                bad["streams"][name] = torch.Generator().manual_seed(91).get_state()
        with pytest.raises(ValueError, match="RNG state"):
            restored.load_state_dict(bad)
        assert_checkpoint_equal(before, restored.state_dict())


@pytest.mark.parametrize("isolated", ["eval_generator", "model_generator"])
def test_eval_and_model_streams_cannot_alias_training(isolated):
    shared = torch.Generator().manual_seed(19)
    with pytest.raises(ValueError, match="must be separate"):
        GANTrainer(get_recipe(num_particles=8, z_dim=2), nn.Linear(2, 2), nn.Linear(2, 1),
                   latent_generator=shared, **{isolated: shared})


@pytest.mark.parametrize("global_binding", ["model_generator", "latent_generator", "noise_generator"])
def test_model_isolation_rejects_process_default_generator_bindings(global_binding):
    options = {"model_generator": torch.Generator().manual_seed(12), global_binding: torch.default_generator}
    with pytest.raises(ValueError, match="model stream must be separate"):
        GANTrainer(get_recipe(num_particles=8, z_dim=2), nn.Linear(2, 2), nn.Linear(2, 1),
                   **options)


def test_step_updates_prior_restores_frozen_parameters_and_detaches_results():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    original_flags = [p.requires_grad for p in trainer.D.parameters()]
    observations = []
    handle = trainer.D.register_forward_pre_hook(
        lambda model, inputs: observations.append((model.training, [p.requires_grad for p in model.parameters()])))
    before = trainer.prior.z.detach().clone()
    calls = []
    result = trainer.step(torch.randn(6, 2), generator_real=lambda: calls.append(1) or torch.zeros(6, 2), collect_stats=True)
    handle.remove()
    assert calls == [1]
    assert not torch.equal(before, trainer.prior.z)
    assert len(trainer.opt_g.param_groups) == 2
    assert [p.requires_grad for p in trainer.D.parameters()] == original_flags
    assert all(not any(flags) for training, flags in observations if not training)
    assert any(training for training, _ in observations)
    assert result["step"] == 1 and "penalty_stats" in result
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        assert result[key].ndim == 0 and not result[key].requires_grad
    assert torch.equal(result["loss_g"], result["loss_gan"] + trainer.recipe.prior_reg * result["prior_regularization"])


def test_freeze_restored_when_generator_callback_raises():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    flags = [p.requires_grad for p in trainer.D.parameters()]
    def fail():
        raise RuntimeError("caller failure")
    with pytest.raises(RuntimeError, match="caller failure"):
        trainer.step(torch.randn(6, 2), generator_real=fail)
    assert [p.requires_grad for p in trainer.D.parameters()] == flags


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("response", ["none", "coherent_G_mean_v1"])
def test_interrupted_update_requires_checkpoint_recovery(error, response):
    trainer = make_trainer(preset="bcap", generator_gradient_response=response)
    real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    saved = trainer.state_dict()
    trainer.step(real)
    expected = trainer.state_dict()
    trainer.load_state_dict(saved)

    def fail():
        raise error("caller failure")

    with pytest.raises(error, match="caller failure"):
        trainer.step(real, generator_real=fail)
    assert trainer.interrupted and trainer.policy.interrupted
    assert not trainer._in_training_step
    assert trainer.completed_steps == saved["completed_steps"]
    assert any(not torch.equal(value, trainer.D.state_dict()[name])
               for name, value in saved["models"]["D"].items())
    partial = trainer._state_dict()
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.state_dict()
    with pytest.raises(RuntimeError, match="boundary"):
        trainer.policy.state_dict()
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.step(real)
    trainer.policy.abort_step()  # Repeated cleanup cannot certify a boundary.
    bad = deepcopy(saved)
    bad["completed_steps"] = -1
    with pytest.raises(ValueError, match="step count"):
        trainer.load_state_dict(bad)
    assert trainer.interrupted
    assert_checkpoint_equal(partial, trainer._state_dict())

    trainer.load_state_dict(saved)
    assert not trainer.interrupted
    assert_checkpoint_equal(saved, trainer.state_dict())
    trainer.step(real)
    assert_checkpoint_equal(expected, trainer.state_dict())


def test_preflight_rejection_and_idle_abort_remain_at_checkpoint_boundary():
    trainer = make_trainer(steps=1)
    saved = trainer.state_dict()
    with pytest.raises(ValueError, match="nonempty"):
        trainer.step(torch.zeros(0, 2))
    with pytest.raises(ValueError, match="nonempty"):
        trainer.policy.begin_step(torch.zeros(0, 2))
    trainer.policy.abort_step()
    assert not trainer.interrupted
    assert_checkpoint_equal(saved, trainer.state_dict())
    trainer.step(torch.zeros(6, 2))
    complete = trainer.state_dict()
    assert not trainer.interrupted
    with pytest.raises(RuntimeError, match="budget"):
        trainer.step(torch.zeros(6, 2))
    assert not trainer.interrupted
    assert_checkpoint_equal(complete, trainer.state_dict())


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_historical_k3p_alias_cannot_resume_as_ka2(schema):
    trainer = make_trainer(preset="ka2")
    trainer.step(torch.randn(6, 2))
    checkpoint = trainer.state_dict()
    recipe = dict(checkpoint["recipe"])
    for key in ("reg_anchor_weight", "direct_particle_gain"):
        del recipe[key]
    old = {**checkpoint, "recipe": {**recipe, "loss_type": "logistic", "gan_mode": "rp",
                                    "reg_arm": "k3p", "reg_method": "autograd"}}
    old["schema"] = schema
    if schema == 2:
        old = deepcopy(old)
        generator = old["optimizers"][0].pop("regularizer")
        critic = old["optimizers"][1].pop("regularizer")
        assert generator["direct"] is None
        old["schema"] = 2
        old["k3p"] = {"latent": generator["latent"],
                       "critic": {"penalty": critic["record"], "ema": critic["ema"],
                                  "guard": critic["guard"]}}
    restored = make_trainer(preset="ka2")
    initial = restored.state_dict()
    with pytest.raises(ValueError, match="schema" if schema < 4 else "recipe"):
        restored.load_state_dict(old)
    assert_checkpoint_equal(initial, restored.state_dict())


@pytest.mark.parametrize("formulation", ["ka2", "k3p"])
def test_current_checkpoint_without_unused_arm_selector_still_resumes(formulation):
    trainer = make_trainer(preset=formulation)
    trainer.step(torch.arange(12, dtype=torch.float32).reshape(6, 2) / 8)
    checkpoint = trainer.state_dict()
    assert "reg_arm" not in checkpoint["recipe"]
    restored = make_trainer(preset=formulation)
    restored.load_state_dict(checkpoint)
    assert restored.completed_steps == 1 and restored.recipe.reg_arm is None
    assert restored.recipe.effective_critic_formulation == formulation
    for actual, expected in zip(restored.G.parameters(), trainer.G.parameters()):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("particles", [12, 1025])
def test_prior_regularizes_full_small_table_or_unique_large_sample(particles):
    trainer = make_trainer(particles=particles, prior_reg=.1)
    seen = []
    handle = trainer.prior_regularizer.register_forward_pre_hook(lambda module, args: seen.append(len(args[0])))
    trainer.step(torch.randn(6, 2))
    handle.remove()
    assert seen == [12] if particles == 12 else 1 <= seen[0] <= 6


def test_ema_updates_parameters_and_copies_float_and_integer_buffers():
    trainer = make_trainer(buffers=True)
    old = [p.clone() for p in trainer.ema_G.parameters()]
    trainer.step(torch.randn(6, 2))
    for previous, averaged, live in zip(old, trainer.ema_G.parameters(), trainer.G.parameters()):
        expected = previous.mul_(trainer.recipe.ema_decay).add_(live, alpha=1 - trainer.recipe.ema_decay)
        assert torch.equal(averaged, expected)
        assert not averaged.requires_grad
    for averaged, live in zip(trainer.ema_G.buffers(), trainer.G.buffers()):
        assert torch.equal(averaged, live)
    assert trainer.ema_G[1].num_batches_tracked.item() == 1
    assert not trainer.ema_G.training and not trainer.ema_prior.training


def test_sample_preserves_modes_and_training_randomness():
    trainer = make_trainer(buffers=True, dropout=True)
    trainer.G[1].eval()
    flags = [module.training for module in trainer.G.modules()]
    checkpoint = trainer.state_dict()
    real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    cpu_rng = torch.get_rng_state().clone()
    assert trainer.sample(17).shape == (17, 2)
    assert trainer.sample(9, ema=True).shape == (9, 2)
    assert flags == [module.training for module in trainer.G.modules()]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(checkpoint["streams"]["latent_generator"], trainer.latent_generator.get_state())
    first = trainer.step(real)
    trained = trainer.state_dict()
    trainer.load_state_dict(checkpoint)
    second = trainer.step(real)
    assert_models_equal(trained, trainer.state_dict())
    for key in ("loss_d", "loss_g", "penalty"):
        assert torch.equal(first[key], second[key])
    with pytest.raises(ValueError, match="separate"):
        trainer.sample(3, generator=trainer.latent_generator)


def test_checkpoint_exact_continuation_with_dropout_and_independent_storage():
    torch.manual_seed(0)
    trainer = make_trainer(dropout=True)
    real = torch.randn(6, 2)
    trainer.step(real)
    trainer.step(real)
    checkpoint = trainer.state_dict()
    preserved = deepcopy(checkpoint)
    expected_stats = trainer.step(real)
    expected = trainer.state_dict()
    assert_models_equal(checkpoint, preserved)
    restored = make_trainer(dropout=True)
    restored.load_state_dict(checkpoint)
    actual_stats = restored.step(real)
    assert_models_equal(expected, restored.state_dict())
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        assert torch.equal(expected_stats[key], actual_stats[key])
    assert restored.completed_steps == 3
    assert_models_equal(checkpoint, preserved)


def test_schedule_budget_and_rejected_checkpoint_do_not_change_models():
    trainer = make_trainer(steps=3)
    initial = deepcopy(trainer.initial_lrs)
    for step in range(3):
        trainer.step(torch.randn(6, 2))
        network, prior = learning_rate_scales(step, trainer.recipe)
        assert [g["lr"] for g in trainer.opt_g.param_groups] == [initial[0][0] * network, initial[0][1] * prior]
        assert trainer.opt_d.param_groups[0]["lr"] == initial[1][0] * network
    with pytest.raises(RuntimeError, match="budget"):
        trainer.step(torch.randn(6, 2))
    checkpoint = trainer.state_dict()
    bad = deepcopy(checkpoint)
    bad["recipe"]["reg_coeff"] = 7.
    with pytest.raises(ValueError, match="recipe"):
        trainer.load_state_dict(bad)
    assert_models_equal(checkpoint, trainer.state_dict())
    bad = deepcopy(checkpoint)
    bad["models"]["prior"]["z"] = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="incompatible"):
        trainer.load_state_dict(bad)
    assert_models_equal(checkpoint, trainer.state_dict())


def test_validation_rejects_unsupported_recipe_and_mismatched_models():
    trainer = make_trainer()
    with pytest.raises(ValueError, match="unconditional"):
        GANTrainer(get_recipe(model='ddgan', num_classes=4, conditioning='ucd'), trainer.G, trainer.D)
    with pytest.raises(ValueError, match="dimensions"):
        GANTrainer(trainer.recipe.replace(z_dim=4), trainer.G, trainer.D, prior=trainer.prior)
    with pytest.raises(ValueError, match="dtype"):
        GANTrainer(trainer.recipe, deepcopy(trainer.G).double(), trainer.D)
    with pytest.raises(ValueError, match="nonempty"):
        trainer.step(torch.zeros(0, 2))


@pytest.mark.parametrize("invalid", [
    torch.zeros(6, 2, dtype=torch.float64), torch.zeros(6, 2, device="meta"), torch.zeros(5, 2), torch.zeros(6, 3),
])
def test_generator_real_tensor_validation_precedes_any_update(invalid):
    trainer = make_trainer()
    real = torch.zeros(6, 2)
    trainer.step(real)
    before = trainer.state_dict()
    modes = [m.training for root in (trainer.G, trainer.D) for m in root.modules()]
    with pytest.raises(ValueError, match="generator_real"):
        trainer.step(real, generator_real=invalid)
    assert_checkpoint_equal(before, trainer.state_dict())
    assert modes == [m.training for root in (trainer.G, trainer.D) for m in root.modules()]


@pytest.mark.parametrize("broken", ["second_optimizer", "latent_rng", "cpu_rng"])
def test_malformed_checkpoint_does_not_mutate_any_live_state(broken):
    trainer = make_trainer(dropout=True)
    real = torch.zeros(6, 2)
    trainer.step(real)
    bad = trainer.state_dict()
    trainer.step(real)
    before = trainer.state_dict()
    if broken == "second_optimizer":
        bad["optimizers"][1]["param_groups"] = []
    elif broken == "latent_rng":
        bad["streams"]["latent_generator"] = torch.ones(1, dtype=torch.uint8)
    else:
        bad["cpu_rng"] = torch.ones(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="optimizer state|RNG state"):
        trainer.load_state_dict(bad)
    assert_checkpoint_equal(before, trainer.state_dict())


def test_checkpoint_requires_matching_parameter_freezing():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    checkpoint = trainer.state_dict()
    assert checkpoint["requires_grad"]["D"]["3.bias"] is False
    restored = make_trainer()
    before = restored.state_dict()
    with pytest.raises(ValueError, match="requires_grad"):
        restored.load_state_dict(checkpoint)
    assert_checkpoint_equal(before, restored.state_dict())
    restored.D[-1].bias.requires_grad_(False)
    restored.load_state_dict(checkpoint)
    assert_checkpoint_equal(checkpoint, restored.state_dict())


@pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_accepts_current_device_streams_and_restores_checkpoint():
    recipe = get_recipe(num_particles=12, z_dim=3, total_steps=3)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = nn.Sequential(nn.Linear(3, 8), nn.LeakyReLU(.2), nn.Linear(8, 2)).to(device)
    discriminator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1)).to(device)
    latent = torch.Generator(device="cuda").manual_seed(2)
    penalty = torch.Generator(device="cuda").manual_seed(3)
    trainer = GANTrainer(recipe, generator, discriminator,
                                  latent_generator=latent, penalty_generator=penalty)
    real = torch.randn(6, 2, device=device)
    trainer.step(real)
    checkpoint = trainer.state_dict()
    trainer.sample(7, generator=torch.Generator(device="cuda").manual_seed(4))
    trainer.step(real)
    expected = trainer.state_dict()
    trainer.load_state_dict(checkpoint)
    trainer.step(real)
    assert_models_equal(expected, trainer.state_dict())
    assert trainer.sample(5, ema=True).device == device
    with pytest.raises(ValueError, match="model device"):
        trainer.sample(1, generator=torch.Generator(device="cpu"))
    before = trainer.state_dict()
    invalid = deepcopy(checkpoint)
    invalid["cuda_rng"] = torch.ones(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="RNG state"):
        trainer.load_state_dict(invalid)
    assert_checkpoint_equal(before, trainer.state_dict())
