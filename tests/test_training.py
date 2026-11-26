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
        lambda model, inputs: observations.append(
            (model.training, torch.is_grad_enabled(), [p.requires_grad for p in model.parameters()])))
    before = trainer.prior.z.detach().clone()
    calls = []
    result = trainer.step(torch.randn(6, 2), generator_real=lambda: calls.append(1) or torch.zeros(6, 2), collect_stats=True)
    handle.remove()
    assert calls == [1]
    assert not torch.equal(before, trainer.prior.z)
    assert len(trainer.opt_g.param_groups) == 2
    assert [p.requires_grad for p in trainer.D.parameters()] == original_flags
    assert any(not training and grad_on for training, grad_on, _ in observations)
    assert all(not any(flags) for training, grad_on, flags in observations if not training and grad_on)
    assert any(training for training, _, _ in observations)
    assert result["step"] == 1 and "penalty_stats" in result
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        assert result[key].ndim == 0 and not result[key].requires_grad
    expected_loss_g = result["loss_gan"] + trainer.recipe.prior_reg * result["prior_regularization"]
    if "rhbi_simplex" in result:
        expected_loss_g = expected_loss_g + result["rhbi_simplex"]
    assert torch.equal(result["loss_g"], expected_loss_g)


def test_freeze_restored_when_generator_callback_raises():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    flags = [p.requires_grad for p in trainer.D.parameters()]
    def fail():
        raise RuntimeError("caller failure")
    with pytest.raises(RuntimeError, match="caller failure"):
        trainer.step(torch.randn(6, 2), generator_real=fail)
    assert [p.requires_grad for p in trainer.D.parameters()] == flags


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
    expected_batches = 3 if trainer.recipe.rhbi_simplex_weight > 0 else 1
    assert trainer.ema_G[1].num_batches_tracked.item() == expected_batches
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


def _bn_buffer_snapshot(module):
    return {
        name: (
            child.running_mean.detach().clone(),
            child.running_var.detach().clone(),
            child.num_batches_tracked.detach().clone(),
        )
        for name, child in module.named_modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.running_mean is not None
    }


def _make_bn_critic_trainer(*, preset="gan", dropout=False, track_running_stats=True, seed=0):
    torch.manual_seed(seed)
    recipe = get_recipe(preset, num_particles=12, z_dim=3, batch_size=6, total_steps=10)
    generator = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8), nn.LeakyReLU(.2), nn.Linear(8, 4))
    layers = [
        nn.Linear(4, 8, bias=False),
        nn.BatchNorm1d(8, track_running_stats=track_running_stats),
        nn.LeakyReLU(.2),
    ]
    if dropout:
        layers.append(nn.Dropout(.5))
    layers.append(nn.Linear(8, 1))
    discriminator = nn.Sequential(*layers)
    return GANTrainer(recipe, generator, discriminator, seed=seed)


@pytest.mark.parametrize("track_running_stats", [True, False])
def test_bn_critic_uses_batch_stats_in_g_step_without_mutating_running_stats(track_running_stats):
    trainer = _make_bn_critic_trainer(track_running_stats=track_running_stats)
    bn = trainer.D[1]
    real = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 6.0 - 0.5
    # Prime running stats with a distinct scale so batch-stat vs eval-stat outputs clearly diverge.
    if track_running_stats:
        with torch.no_grad():
            bn.running_mean.fill_(5.0)
            bn.running_var.fill_(9.0)

    phase = {"in_g": False, "post_d_buffers": None}
    g_root_obs, g_bn_obs, g_num_checks = [], [], []
    orig_after_d = trainer.policy.after_critic_step
    orig_after_g = trainer.policy.after_generator_step

    def _after_d():
        orig_after_d()
        phase["in_g"] = True
        phase["post_d_buffers"] = _bn_buffer_snapshot(trainer.D)

    def _after_g():
        orig_after_g()
        if track_running_stats:
            after_g_buffers = _bn_buffer_snapshot(trainer.D)
            for key, (mean0, var0, num0) in phase["post_d_buffers"].items():
                mean1, var1, num1 = after_g_buffers[key]
                assert torch.equal(mean0, mean1)
                assert torch.equal(var0, var1)
                assert torch.equal(num0, num1)
        phase["in_g"] = False

    trainer.policy.after_critic_step = _after_d
    trainer.policy.after_generator_step = _after_g

    def _root_pre(mod, _inp):
        if phase["in_g"]:
            g_root_obs.append((mod.training, [p.requires_grad for p in mod.parameters()]))

    def _bn_fwd(mod, inp, out):
        if phase["in_g"]:
            g_bn_obs.append((mod.training, mod.track_running_stats))
            x = inp[0].detach()
            expected_batch = nn.functional.batch_norm(
                x, None, None, mod.weight.detach(), mod.bias.detach(), True, 0.0, mod.eps
            )
            assert torch.allclose(out.detach(), expected_batch, atol=1e-6, rtol=1e-6)
            if track_running_stats:
                eval_out = nn.functional.batch_norm(
                    x, mod.running_mean, mod.running_var, mod.weight.detach(), mod.bias.detach(), False, 0.0, mod.eps
                )
                assert not torch.allclose(out.detach(), eval_out, atol=1e-2)
            g_num_checks.append(1)

    h_root = trainer.D.register_forward_pre_hook(_root_pre)
    h_bn = bn.register_forward_hook(_bn_fwd)
    try:
        trainer.step(real)
    finally:
        h_root.remove()
        h_bn.remove()

    assert g_root_obs and all(not training and not any(flags) for training, flags in g_root_obs)
    assert g_bn_obs and all(training and not tracking for training, tracking in g_bn_obs)
    assert len(g_num_checks) >= 2
    assert not trainer.D.training and not bn.training
    assert bn.track_running_stats is track_running_stats
    assert all(p.requires_grad for p in trainer.D.parameters())
    if track_running_stats:
        # Exactly two D-step forwards (real + fake) updated num_batches_tracked; G-step added zero.
        assert bn.num_batches_tracked.item() == 2


def test_frozen_eval_only_bn_submodule_stays_in_eval_during_g_step():
    class _FrozenBackboneMixedCritic(nn.Module):
        """Mimics ResNet18FeatureCritic32: frozen eval-only BN backbone + trainable head with BN."""

        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(6),
                nn.ReLU(),
            )
            with torch.no_grad():
                self.backbone[1].running_mean.copy_(torch.linspace(-1.5, 1.5, 6))
                self.backbone[1].running_var.copy_(torch.linspace(0.5, 2.5, 6))
                self.backbone[1].num_batches_tracked.fill_(42)
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(6, 8, bias=False),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(.2),
                nn.Linear(8, 1),
            )

        def train(self, mode=True):
            super().train(mode)
            self.backbone.eval()
            return self

        def forward(self, x):
            return self.head(self.backbone(x))

    torch.manual_seed(0)
    recipe = get_recipe("gan", num_particles=8, z_dim=4, batch_size=6, total_steps=6)
    generator = nn.Sequential(
        nn.Linear(4, 3 * 4 * 4),
        nn.Unflatten(1, (3, 4, 4)),
    )
    critic = _FrozenBackboneMixedCritic()
    trainer = GANTrainer(recipe, generator, critic, seed=0)

    frozen_bn = critic.backbone[1]
    head_bn = critic.head[3]
    init_frozen_buffers = _bn_buffer_snapshot(critic.backbone)
    init_flags = [p.requires_grad for p in critic.parameters()]

    phase = {"in_g": False, "post_d_head": None}
    frozen_modes, head_g_modes, frozen_eval_checks = [], [], []
    orig_after_d = trainer.policy.after_critic_step
    orig_after_g = trainer.policy.after_generator_step

    def _after_d():
        orig_after_d()
        phase["in_g"] = True
        phase["post_d_head"] = _bn_buffer_snapshot(critic.head)

    def _after_g():
        orig_after_g()
        for key, (m0, v0, n0) in phase["post_d_head"].items():
            m1, v1, n1 = _bn_buffer_snapshot(critic.head)[key]
            assert torch.equal(m0, m1) and torch.equal(v0, v1) and torch.equal(n0, n1)
        phase["in_g"] = False

    trainer.policy.after_critic_step = _after_d
    trainer.policy.after_generator_step = _after_g

    def _frozen_hook(mod, inp, out):
        frozen_modes.append((phase["in_g"], mod.training, mod.track_running_stats))
        expected_eval = nn.functional.batch_norm(
            inp[0].detach(), mod.running_mean, mod.running_var,
            mod.weight.detach(), mod.bias.detach(), False, 0.0, mod.eps,
        )
        assert torch.allclose(out.detach(), expected_eval, atol=1e-6, rtol=1e-6)
        frozen_eval_checks.append(phase["in_g"])

    def _head_hook(mod, _inp):
        if phase["in_g"]:
            head_g_modes.append((mod.training, mod.track_running_stats))

    h1 = frozen_bn.register_forward_hook(_frozen_hook)
    h2 = head_bn.register_forward_pre_hook(_head_hook)
    try:
        trainer.step(torch.randn(6, 3, 4, 4))
    finally:
        h1.remove()
        h2.remove()

    # Frozen backbone BN stayed strictly in eval() across BOTH D step and G step.
    assert any(in_g for in_g in frozen_eval_checks) and any(not in_g for in_g in frozen_eval_checks)
    assert all((not training) and tracking for _, training, tracking in frozen_modes)
    end_frozen_buffers = _bn_buffer_snapshot(critic.backbone)
    for key, (m0, v0, n0) in init_frozen_buffers.items():
        m1, v1, n1 = end_frozen_buffers[key]
        assert torch.equal(m0, m1) and torch.equal(v0, v1) and torch.equal(n0, n1)

    # Trainable head BN stayed in non-tracking batch-stat mode during G step, then restored cleanly.
    assert head_g_modes and all(training and not tracking for training, tracking in head_g_modes)
    assert not head_bn.training and head_bn.track_running_stats
    assert not frozen_bn.training and frozen_bn.track_running_stats
    assert [p.requires_grad for p in critic.parameters()] == init_flags


def test_dropout_stays_in_eval_while_bn_uses_batch_stats_in_g_step():
    trainer = _make_bn_critic_trainer(dropout=True)
    bn = trainer.D[1]
    drop = trainer.D[3]
    phase = {"in_g": False}
    orig_after_d = trainer.policy.after_critic_step
    orig_after_g = trainer.policy.after_generator_step
    trainer.policy.after_critic_step = lambda: (orig_after_d(), phase.__setitem__("in_g", True))
    trainer.policy.after_generator_step = lambda: (orig_after_g(), phase.__setitem__("in_g", False))

    drop_d_modes, drop_g_modes, bn_g_modes = [], [], []

    def _drop_hook(mod, inp, out):
        if phase["in_g"]:
            drop_g_modes.append(mod.training)
            assert torch.equal(out.detach(), inp[0].detach())
        else:
            drop_d_modes.append(mod.training)

    def _bn_hook(mod, _inp):
        if phase["in_g"]:
            bn_g_modes.append((mod.training, mod.track_running_stats))

    h_drop = drop.register_forward_hook(_drop_hook)
    h_bn = bn.register_forward_pre_hook(_bn_hook)
    try:
        trainer.step(torch.randn(6, 4))
    finally:
        h_drop.remove()
        h_bn.remove()

    assert drop_d_modes and any(drop_d_modes)
    assert drop_g_modes and all(not training for training in drop_g_modes)
    assert bn_g_modes and all(training and not tracking for training, tracking in bn_g_modes)
    assert not trainer.D.training and not drop.training and not bn.training and bn.track_running_stats


@pytest.mark.parametrize("failure_point", ["callback", "critic_forward"])
def test_bn_modes_buffers_and_freeze_restored_when_g_step_raises(failure_point):
    trainer = _make_bn_critic_trainer(dropout=True)
    trainer.D[-1].bias.requires_grad_(False)
    bn = trainer.D[1]
    original_flags = [p.requires_grad for p in trainer.D.parameters()]
    real = torch.randn(6, 4)

    phase = {"in_g": False, "post_d_buffers": None}
    orig_after_d = trainer.policy.after_critic_step

    def _after_d():
        orig_after_d()
        phase["in_g"] = True
        phase["post_d_buffers"] = _bn_buffer_snapshot(trainer.D)

    trainer.policy.after_critic_step = _after_d

    handle = None
    if failure_point == "critic_forward":
        def _boom(_mod, _inp):
            if phase["in_g"]:
                raise RuntimeError("g-step critic failure")
        handle = trainer.D.register_forward_pre_hook(_boom)

    def _fail_callback():
        raise RuntimeError("caller failure")

    try:
        with pytest.raises(RuntimeError, match="caller failure|g-step critic failure"):
            trainer.step(real, generator_real=_fail_callback if failure_point == "callback" else None)
    finally:
        if handle is not None:
            handle.remove()
        trainer.policy.after_critic_step = orig_after_d

    assert not trainer.D.training
    assert not bn.training
    assert bn.track_running_stats is True
    assert [p.requires_grad for p in trainer.D.parameters()] == original_flags
    for key, (m0, v0, n0) in phase["post_d_buffers"].items():
        m1, v1, n1 = _bn_buffer_snapshot(trainer.D)[key]
        assert torch.equal(m0, m1) and torch.equal(v0, v1) and torch.equal(n0, n1)

    # Next valid step must still track running stats normally during D step.
    tracked_before = bn.num_batches_tracked.item()
    trainer.step(real)
    assert bn.num_batches_tracked.item() == tracked_before + 2


@pytest.mark.parametrize("preset", ["gan", "axiom"])
def test_checkpoint_exact_continuation_with_bn_critic(preset, tmp_path):
    trainer = _make_bn_critic_trainer(preset=preset, dropout=True, seed=7)
    real = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 12.0 - 0.5
    trainer.step(real)
    trainer.step(real.flip(0))
    checkpoint = trainer.state_dict()
    torch.save(checkpoint, tmp_path / f"bn_{preset}.pt")
    loaded = torch.load(tmp_path / f"bn_{preset}.pt", map_location="cpu", weights_only=True)
    preserved = deepcopy(loaded)

    expected_stats = trainer.step(real)
    expected_state = trainer.state_dict()

    restored = _make_bn_critic_trainer(preset=preset, dropout=True, seed=99)
    restored.step(real + 1.0)
    restored.load_state_dict(loaded)
    assert_checkpoint_equal(loaded, restored.state_dict())

    actual_stats = restored.step(real)
    assert_checkpoint_equal(expected_state, restored.state_dict())
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        assert torch.equal(expected_stats[key], actual_stats[key])
    assert_checkpoint_equal(loaded, preserved)
