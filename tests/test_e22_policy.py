"""E22 conformance on one deterministic CPU trajectory, including event boundaries."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe


def components(recipe):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(709)
        G = nn.Sequential(nn.Linear(2, 8, dtype=torch.float32), nn.Tanh(),
                          nn.Linear(8, 2, dtype=torch.float32)).double()
        D = nn.Sequential(nn.Linear(2, 8, dtype=torch.float32), nn.Tanh(),
                          nn.Linear(8, 1, dtype=torch.float32)).double()
        prior = recipe.make_prior(dtype=torch.float32).double()
    return G, D, prior


def recipe(**overrides):
    return get_recipe("e22", num_particles=32, z_dim=2, batch_size=16,
                      output_noise_std=.04, **overrides)


def real_batch(index):
    x = torch.arange(16, dtype=torch.float64)
    return torch.stack((torch.cos(x * .37 + index * .11),
                        torch.sin(x * .19 - index * .07)), 1)


def assert_tree_equal(left, right):
    """Exact comparison, including missing keys, NaN diagnostics and RNG bytes."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def prime_table_verdict(owner, sign, *, block_scale):
    """A valid partial evidence window just before its final block.

    This supplies evidence, without replacing the tester or its decision code,
    so a short conformance test can cross a rare controller event deterministically.
    """
    tester = owner.lr_settle.testers[0][1]
    tester.b = float(block_scale)
    tester.tau = float(block_scale)
    tester.blocks = []
    tester.r_b = [torch.full((32,), float(sign), dtype=torch.float64, device=owner.prior.z.device)
                  for _ in range(tester.K)]
    tester.r_2b = [torch.full((32,), float(sign), dtype=torch.float64, device=owner.prior.z.device)
                   for _ in range(tester.K // 2)]
    tester.blocks_in_window = 2 * tester.K - 1
    tester.invalid_block_rows = None


def prime_particle_moves(owner):
    """Strong opposite row evidence; exercise the real BH, pairing and move code."""
    birth = owner.birth_death
    signs = torch.ones(32, dtype=torch.float64, device=owner.prior.z.device)
    signs[16:] = -1
    birth.S.copy_(signs * 1000)
    birth.W.copy_(signs * 1000)
    birth.n.fill_(2)
    birth.anchor.copy_(owner.prior.z.detach())
    birth.radius.fill_(1e6)


def native_receipt(trainer):
    state = trainer.state_dict()
    summaries = {}
    for name, model in state["models"].items():
        flat = torch.cat([v.flatten() for v in model.values()])
        summaries[name] = [float(flat.sum()), float(flat.square().sum()),
                           float(flat[0]), float(flat[-1])]
    return {"models": summaries, "output_sigma": trainer.output_sigma(),
            "log_sigma": float(trainer.log_output_sigma.detach()),
            "lrs": [[g["lr"] for g in opt.param_groups] for opt in (trainer.opt_g, trainer.opt_d)],
            "table_scale": trainer.lr_settle.testers[0][1].s,
            "table_anchor": trainer.lr_settle.testers[0][1].b_anchor,
            "table_counts": trainer.lr_settle.testers[0][1].counts,
            "birth_death_counts": trainer.birth_death.counters,
            "controller_updates": trainer.controller.updates,
            "row_evidence_counts": trainer.row_evidence.counters}


def native_trajectory(trainer):
    losses = []
    for index in range(7):
        if index == 3:
            prime_table_verdict(trainer, -1, block_scale=4)
            prime_particle_moves(trainer)
        if index == 4:
            prime_table_verdict(trainer, 1, block_scale=2)  # held below twice the anchor
        if index == 5:
            prime_table_verdict(trainer, 1, block_scale=8)  # accepted anchored release
        stats = trainer.step(real_batch(index), collect_stats=True)
        losses.append([float(stats[key]) for key in ("loss_d", "loss_g", "penalty")])
    return {"losses": losses, **native_receipt(trainer),
            "served_samples": trainer.sample(
                4, generator=torch.Generator().manual_seed(17), output_noise=True).tolist()}


def test_native_e22_retains_pre_extraction_behavior():
    # Recorded by the unmodified GANTrainer at 0cde71e4, before extraction.
    # Synthetic partial evidence crosses rare events without changing decision
    # code. Tight float tolerance permits harmless CPU kernel differences.
    expected = json.loads((Path(__file__).parent / "fixtures/e22_native.json").read_text())
    options = recipe()
    G, D, prior = components(options)
    trainer = GANTrainer(options, G, D, prior=prior, seed=101, serial_backward=True)
    actual = native_trajectory(trainer)
    for key in ("losses", "served_samples"):
        torch.testing.assert_close(torch.tensor(actual[key], dtype=torch.float64),
                                   torch.tensor(expected[key], dtype=torch.float64),
                                   rtol=2e-12, atol=2e-12)
    for actual_row, expected_row in zip(actual["lrs"], expected["lrs"]):
        torch.testing.assert_close(torch.tensor(actual_row, dtype=torch.float64),
                                   torch.tensor(expected_row, dtype=torch.float64),
                                   rtol=2e-12, atol=2e-12)
    for name in actual["models"]:
        torch.testing.assert_close(torch.tensor(actual["models"][name], dtype=torch.float64),
                                   torch.tensor(expected["models"][name], dtype=torch.float64),
                                   rtol=2e-12, atol=2e-12)
    for key in ("output_sigma", "log_sigma"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-12, abs=2e-12)
    for key in ("table_scale", "table_anchor", "table_counts", "birth_death_counts", "controller_updates"):
        assert actual[key] == expected[key]
    for key, value in expected["row_evidence_counts"].items():
        assert actual["row_evidence_counts"][key] == pytest.approx(value, rel=2e-12, abs=2e-12)


def paired_loops():
    from examples.e22_external_loop import make_loop

    options = recipe()
    G, D, prior = components(options)
    external = make_loop(options, deepcopy(G), deepcopy(D), deepcopy(prior), seed=101)
    native = GANTrainer(options, G, D, prior=prior, seed=101, serial_backward=True)
    return native, external


def test_caller_owned_loop_is_exactly_native_through_controller_and_particle_events():
    from examples.e22_external_loop import update

    native, external = paired_loops()
    assert_tree_equal(native.policy.state_dict(), external.policy.state_dict())
    initial_noise = external.policy.log_output_sigma.detach().clone()
    for index in range(7):
        if index == 3:
            for owner in (native, external.policy):
                prime_table_verdict(owner, -1, block_scale=4)
                prime_particle_moves(owner)
        elif index == 4:
            for owner in (native, external.policy):
                prime_table_verdict(owner, 1, block_scale=2)
        elif index == 5:
            for owner in (native, external.policy):
                prime_table_verdict(owner, 1, block_scale=8)
        cpu_rng = torch.get_rng_state()
        expected = native.step(real_batch(index), generator_real=real_batch(index + 1), collect_stats=True)
        torch.set_rng_state(cpu_rng)
        with torch.autograd.set_multithreading_enabled(False):
            actual = update(external, real_batch(index),
                            generator_real=lambda: real_batch(index + 1), collect_stats=True)
        assert_tree_equal(expected, actual)
        assert_tree_equal(native.policy.state_dict(), external.policy.state_dict())
        assert_tree_equal(native.policy.served_snapshot(), external.policy.served_snapshot())
        saved = external.policy.state_dict()
        served = external.policy.served_model()
        for with_noise in (False, True):
            expected_samples = native.sample(
                7, generator=torch.Generator().manual_seed(23), output_noise=with_noise)
            actual_samples = served.sample(
                7, generator=torch.Generator().manual_seed(23), output_noise=with_noise)
            assert_tree_equal(expected_samples, actual_samples)
        assert_tree_equal(saved, external.policy.state_dict())
        if index in (3, 4):
            assert external.policy.served_snapshot()["source"] == "averaged"
        if index == 5:
            assert external.policy.served_snapshot()["source"] == "fast"
    policy = external.policy
    tester = policy.lr_settle.testers[0][1]
    assert tester.counts["stationary"] == tester.counts["drift"] == 1
    assert tester.counts["drift_blocked"] == 1
    assert policy.birth_death.counters["moves"] == 16
    assert policy.row_evidence.counters["resets"] == 16
    assert not torch.equal(initial_noise, policy.log_output_sigma)


@pytest.mark.parametrize("resume_before", [3, 4, 5])
def test_policy_checkpoint_exact_resume_across_settle_move_and_anchored_release(resume_before):
    from examples.e22_external_loop import make_loop, update

    _, reference = paired_loops()
    policy = reference.policy
    for index in range(resume_before):
        if index == 3:
            prime_table_verdict(policy, -1, block_scale=4)
            prime_particle_moves(policy)
        elif index == 4:
            prime_table_verdict(policy, 1, block_scale=2)
        with torch.autograd.set_multithreading_enabled(False):
            update(reference, real_batch(index))
    if resume_before == 3:
        prime_table_verdict(policy, -1, block_scale=4)
        prime_particle_moves(policy)
    elif resume_before == 4:
        prime_table_verdict(policy, 1, block_scale=2)
    else:
        prime_table_verdict(policy, 1, block_scale=8)
    checkpoint = policy.state_dict()
    preserved = deepcopy(checkpoint)
    served_before = policy.served_snapshot()
    with torch.autograd.set_multithreading_enabled(False):
        expected = update(reference, real_batch(resume_before), collect_stats=True)
    expected_state = policy.state_dict()
    expected_served = policy.served_snapshot()

    options = recipe()
    restored = make_loop(options, *components(options), seed=999)
    restored.policy.load_state_dict(checkpoint)
    assert_tree_equal(checkpoint, restored.policy.state_dict())
    assert_tree_equal(served_before, restored.policy.served_snapshot())
    with torch.autograd.set_multithreading_enabled(False):
        actual = update(restored, real_batch(resume_before), collect_stats=True)
    assert_tree_equal(expected, actual)
    assert_tree_equal(expected_state, restored.policy.state_dict())
    assert_tree_equal(expected_served, restored.policy.served_snapshot())
    assert_tree_equal(checkpoint, preserved)


def served_outputs(loop, snapshot):
    G, prior = deepcopy(loop.generator).eval(), deepcopy(loop.prior).eval()
    G.load_state_dict(snapshot["models"]["generator"])
    prior.load_state_dict(snapshot["models"]["prior"])
    with torch.no_grad():
        # Compare a fixed set of clean particle centres. The serving snapshot
        # carries the selected weights; caller inference owns any extra noise.
        return G(prior.z)


def test_served_snapshot_preserves_fast_training_weights_and_restores_outputs():
    from examples.e22_external_loop import make_loop, update

    native, external = paired_loops()
    for index in range(4):
        if index == 3:
            for owner in (native, external.policy):
                prime_table_verdict(owner, -1, block_scale=4)
                prime_particle_moves(owner)
        native.step(real_batch(index))
        with torch.autograd.set_multithreading_enabled(False):
            update(external, real_batch(index))
    fast_before = deepcopy(external.generator.state_dict())
    table_before = external.prior.z.detach().clone()
    snapshot = external.policy.served_snapshot()
    assert snapshot["source"] == "averaged"
    assert_tree_equal(fast_before, external.generator.state_dict())
    assert torch.equal(table_before, external.prior.z)
    assert any(not torch.equal(value, snapshot["models"]["generator"][name])
               for name, value in fast_before.items())
    assert_tree_equal(snapshot, native.policy.served_snapshot())
    expected_outputs = served_outputs(external, snapshot)
    restored = make_loop(recipe(), *components(recipe()))
    restored.policy.load_state_dict(external.policy.state_dict())
    assert_tree_equal(snapshot, restored.policy.served_snapshot())
    assert_tree_equal(expected_outputs, served_outputs(restored, restored.policy.served_snapshot()))
    before = external.policy.state_dict()
    for with_noise in (False, True):
        original_served, restored_served = external.policy.served_model(), restored.policy.served_model()
        expected_samples = original_served.sample(
            11, generator=torch.Generator().manual_seed(59), output_noise=with_noise)
        restored_samples = restored_served.sample(
            11, generator=torch.Generator().manual_seed(59), output_noise=with_noise)
        assert_tree_equal(expected_samples, restored_samples)
        assert_tree_equal(original_served.generate(original_served.table[:3],
                                                   generator=torch.Generator().manual_seed(61), output_noise=with_noise),
                          restored_served.generate(restored_served.table[:3],
                                                   generator=torch.Generator().manual_seed(61), output_noise=with_noise))
    assert_tree_equal(before, external.policy.state_dict())
    # Snapshot storage is independent of parameters and future snapshots.
    snapshot["models"]["generator"]["0.weight"].zero_()
    snapshot["table"].zero_()
    assert_tree_equal(fast_before, external.generator.state_dict())
    assert torch.equal(table_before, external.prior.z)


def test_external_loop_preserves_ka2_anchor_controls_and_resumes_optimizer_state():
    from examples.e22_external_loop import update

    native, external = paired_loops()
    native.step(real_batch(0))
    with torch.autograd.set_multithreading_enabled(False):
        update(external, real_batch(0))
    for opt_d in (native.opt_d, external.opt_d):
        record = opt_d.record
        record.calls = 799
        record.sur_hist = [1.] * 64 + [10.] * 16
        record.sur_base = 1.
        record.last_sur = 10.
    checkpoint = external.policy.state_dict()
    for index in (1, 2):
        expected = native.step(real_batch(index), collect_stats=True)
        with torch.autograd.set_multithreading_enabled(False):
            actual = update(external, real_batch(index), collect_stats=True)
        assert_tree_equal(expected, actual)
        assert_tree_equal(native.policy.state_dict(), external.policy.state_dict())
        assert actual["penalty_stats"]["phase"] == "blend"
    assert external.opt_d.record.anchor_started
    # E22's no-data reopen policy keeps the anchor engaged despite self-surprise;
    # omitting the continuous-controller binding would release it here.
    assert external.opt_d.record.w == 1.
    assert external.opt_d.record.alpha > 0.
    expected = external.policy.state_dict()
    external.policy.load_state_dict(checkpoint)
    for index in (1, 2):
        with torch.autograd.set_multithreading_enabled(False):
            update(external, real_batch(index), collect_stats=True)
    assert_tree_equal(expected, external.policy.state_dict())


def explicit_role_policy():
    from particlegan import E22Policy

    options = recipe()
    G, D, prior = components(options)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        encoder, router = nn.Linear(2, 2).double(), nn.Linear(2, 2).double()
    opt_g = options.make_generator_optimizer([
        {"params": G.parameters()}, {"params": encoder.parameters()},
        {"params": router.parameters()}])
    opt_d = options.make_critic_optimizer(D, ema_critic=deepcopy(D))
    opt_table = options.make_generator_optimizer(
        [{"params": [prior.z], "lr": options.lr * options.prior_lr_mult}], latent_table=prior.z)
    penalty = options.make_critic_penalty(opt_d)
    return E22Policy(
        options, G, D, table=prior.z, generator_optimizer=opt_g,
        critic_optimizer=opt_d, table_optimizer=opt_table, encoder=encoder, router=router,
        roles=[["generator", "encoder", "router"], ["critic"], ["table"]],
        generation=lambda model, latent: model(router(encoder(latent))),
        critic_features=lambda samples: D[1](D[0](samples)), penalty=penalty, seed=101)


def explicit_role_update(policy, real):
    """An independent-row callback with separate auxiliary and table owners."""
    def draw():
        indices = torch.randint(len(policy.table), (len(real),), generator=policy.latent_generator)
        return policy.table[indices]

    loss = policy.recipe.make_loss()
    noise = policy.begin_step(real)
    policy.D.train()
    for module in (policy.G, policy.encoder, policy.router):
        module.eval()
    with torch.no_grad():
        fake = policy.generate(draw(), sigma=noise.output_sigma)
    policy.observe_critic_pair(real, fake)
    adversarial_d = loss.d_loss(policy.D(real), policy.D(fake))
    penalty = policy.penalty(policy.D, real, fake)
    loss_d = adversarial_d + penalty
    policy.opt_d.zero_grad()
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    policy.D.eval().requires_grad_(False)
    for module in (policy.G, policy.encoder, policy.router):
        module.train()
    try:
        loss_g = loss.g_loss(policy.D(policy.generate(draw(), sigma=noise.output_sigma)), policy.D(real))
        policy.opt_g.zero_grad()
        policy.table_optimizer.zero_grad()
        loss_g.backward()
        policy.after_generator_backward(loss_gan=loss_g, loss_critic=loss_d - penalty)
        policy.opt_g.step()
        policy.table_optimizer.step()
        policy.after_generator_step()
    finally:
        policy.D.requires_grad_(True)
    policy.finish_step()
    return loss_d.detach(), loss_g.detach()


def test_raw_table_separate_optimizer_and_auxiliary_roles_checkpoint_and_serve():
    policy = explicit_role_policy()
    initial = deepcopy(policy.state_dict())
    for index in (0, 1):
        with torch.autograd.set_multithreading_enabled(False):
            explicit_role_update(policy, real_batch(index))
    assert policy.prior is None
    assert policy.roles == [["generator", "encoder", "router", "noise"], ["critic"], ["table"]]
    for name in ("generator", "encoder", "router", "critic"):
        assert any(not torch.equal(value, initial["models"][name][key])
                   for key, value in policy.state_dict()["models"][name].items())
    assert not torch.equal(initial["table"], policy.table)
    assert policy.birth_death.counters["evals"] == 1
    checkpoint = policy.state_dict()
    with torch.autograd.set_multithreading_enabled(False):
        expected_losses = explicit_role_update(policy, real_batch(2))
    expected = policy.state_dict()
    restored = explicit_role_policy()
    restored.load_state_dict(checkpoint)
    with torch.autograd.set_multithreading_enabled(False):
        actual_losses = explicit_role_update(restored, real_batch(2))
    assert_tree_equal(expected_losses, actual_losses)
    assert_tree_equal(expected, restored.state_dict())

    # The generation factory binds separate roles to the frozen inference copies.
    restored.lr_settle.testers[2][0].last_decisive = -1
    factory = lambda models: lambda model, latent: model(models["router"](models["encoder"](latent)))
    served = restored.served_model(generation_factory=factory)
    assert served.source == "averaged"
    expected_samples = served.sample(5, generator=torch.Generator().manual_seed(11), output_noise=True)
    with torch.no_grad():
        for module in (restored.G, restored.encoder, restored.router):
            for parameter in module.parameters():
                parameter.add_(100)
    actual_samples = served.sample(5, generator=torch.Generator().manual_seed(11), output_noise=True)
    assert_tree_equal(expected_samples, actual_samples)
    assert all(not p.requires_grad for module in served.models.values() for p in module.parameters())


def test_policy_requires_ordered_hooks_and_completed_boundary_checkpoints():
    _, loop = paired_loops()
    policy = loop.policy
    checkpoint = policy.state_dict()
    with pytest.raises(RuntimeError, match="critic backward"):
        policy.before_critic_backward()
    with pytest.raises(RuntimeError, match="generator backward"):
        policy.after_generator_backward(loss_gan=torch.tensor(0.), loss_critic=torch.tensor(0.))
    policy.begin_step(real_batch(0))
    with pytest.raises(RuntimeError, match="boundary"):
        policy.state_dict()
    with pytest.raises(RuntimeError, match="boundary"):
        policy.load_state_dict(checkpoint)
    with pytest.raises(RuntimeError, match="after_generator_step"):
        policy.finish_step()


def test_caller_owned_abort_requires_validated_checkpoint_recovery():
    from examples.e22_external_loop import update

    native, external = paired_loops()
    policy = external.policy
    saved = policy.state_dict()

    def fail():
        raise RuntimeError("caller failure")

    with torch.autograd.set_multithreading_enabled(False), pytest.raises(RuntimeError, match="caller failure"):
        update(external, real_batch(0), generator_real=fail)
    policy.abort_step()
    assert policy.interrupted and policy.completed_steps == 0
    with pytest.raises(RuntimeError, match="interrupted"):
        policy.begin_step(real_batch(0))
    with pytest.raises(RuntimeError, match="boundary"):
        policy.state_dict()
    bad = deepcopy(saved)
    bad["completed_steps"] = -1
    with pytest.raises(ValueError, match="step count"):
        policy.load_state_dict(bad)
    assert policy.interrupted
    policy.load_state_dict(saved)
    assert not policy.interrupted
    assert_tree_equal(saved, policy.state_dict())
    expected = native.step(real_batch(0), generator_real=real_batch(1))
    policy.load_state_dict(saved)  # Restore global module RNG too.
    with torch.autograd.set_multithreading_enabled(False):
        actual = update(external, real_batch(0), generator_real=real_batch(1))
    assert_tree_equal(expected, actual)
    assert_tree_equal(native.policy.state_dict(), policy.state_dict())


def test_failure_during_begin_cannot_look_like_an_idle_boundary(monkeypatch):
    _, external = paired_loops()
    policy = external.policy
    saved = policy.state_dict()

    def fail(*args):
        raise RuntimeError("observation failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(policy.controller), "observe_game", fail)
        with pytest.raises(RuntimeError, match="observation failure"):
            policy.begin_step(real_batch(0))
    with pytest.raises(RuntimeError, match="boundary"):
        policy.state_dict()
    policy.abort_step()
    assert policy.interrupted
    policy.load_state_dict(saved)
    assert not policy.interrupted
    assert_tree_equal(saved, policy.state_dict())


@pytest.mark.parametrize("semantics", ["conditional", "dense_soft"])
def test_full_e22_rejects_conditioned_or_blended_row_laws_before_adding_noise(semantics):
    from particlegan import E22Policy

    options = recipe()
    G, D, prior = components(options)
    opt_g, opt_d = options.make_optimizers(G, D, prior, ema_critic=deepcopy(D))
    original_groups = len(opt_g.param_groups)
    with pytest.raises(ValueError, match="independent unconditional"):
        E22Policy(options, G, D, prior=prior, generator_optimizer=opt_g,
                  critic_optimizer=opt_d, row_semantics=semantics)
    assert len(opt_g.param_groups) == original_groups


@pytest.mark.parametrize("semantics", ["conditional", "dense_soft"])
def test_nonindependent_adaptations_serve_caller_latents_and_reject_uniform_sampling(semantics):
    from particlegan import UpdatePolicy

    options = recipe(row_evidence_gate=False, particle_birth_death=False,
                     birth_death_isolation=False, birth_death_feature_scale="none")
    G, D, prior = components(options)
    opt_g, opt_d = options.make_optimizers(G, D, prior, ema_critic=deepcopy(D))
    policy = UpdatePolicy(options, G, D, prior=prior, generator_optimizer=opt_g,
                          critic_optimizer=opt_d, row_semantics=semantics)
    assert policy.row_evidence is policy.birth_death is None
    served = policy.served_model()
    with pytest.raises(ValueError, match="routing; use generate"):
        served.sample(2)
    weights = torch.arange(64, dtype=torch.float64).reshape(2, 32).softmax(1)
    routed_latent = weights @ served.table if semantics == "dense_soft" else served.table[:2]
    before = policy.state_dict()
    generated = served.generate(routed_latent, generator=torch.Generator().manual_seed(41))
    assert generated.shape == (2, 2) and torch.isfinite(generated).all()
    assert_tree_equal(before, policy.state_dict())


def test_served_snapshot_handles_tied_generator_parameter_names():
    from particlegan import E22Policy

    options = recipe()
    _, D, prior = components(options)
    G = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)).double()
    G[1].weight = G[0].weight
    opt_g, opt_d = options.make_optimizers(G, D, prior, ema_critic=deepcopy(D))
    policy = E22Policy(options, G, D, prior=prior, generator_optimizer=opt_g,
                       critic_optimizer=opt_d)
    with torch.no_grad():
        G[0].weight.copy_(2 * torch.eye(2, dtype=torch.float64))
        policy.ema_G[0].weight.copy_(3 * torch.eye(2, dtype=torch.float64))
    policy.lr_settle.testers[0][1].last_decisive = -1
    snapshot = policy.served_snapshot()
    expected_weight = 3 * torch.eye(2, dtype=torch.float64)
    assert_tree_equal(snapshot["models"]["generator"]["0.weight"], expected_weight)
    assert_tree_equal(snapshot["models"]["generator"]["1.weight"], expected_weight)
    served = policy.served_model()
    latent = torch.tensor([[1., 2.]], dtype=torch.float64)
    assert_tree_equal(served.generator(latent), 9 * latent)
    assert_tree_equal(G[0].weight, 2 * torch.eye(2, dtype=torch.float64))


@pytest.mark.parametrize("surface", ["policy", "trainer"])
@pytest.mark.parametrize("broken", ["adam_moment", "latent_bandwidth"])
def test_malformed_recovery_state_is_rejected_without_mutating_served_or_training_state(surface, broken):
    from examples.e22_external_loop import update

    native, external = paired_loops()
    for index in range(4):
        if index == 3:
            for owner in (native, external.policy):
                prime_table_verdict(owner, -1, block_scale=4)
        native.step(real_batch(index))
        with torch.autograd.set_multithreading_enabled(False):
            update(external, real_batch(index))
    owner = native if surface == "trainer" else external.policy
    before = owner.state_dict()
    served_before = owner.served_snapshot()
    assert served_before["source"] == "averaged"
    bad = deepcopy(before)
    if broken == "adam_moment":
        optimizer = bad["optimizers"][0]
        param_id = optimizer["param_groups"][0]["params"][0]
        optimizer["state"][param_id]["exp_avg"] = torch.zeros(3, dtype=torch.float64)
    else:
        bad["controller"]["latent_bandwidth"] = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        owner.load_state_dict(bad)
    assert_tree_equal(before, owner.state_dict())
    assert_tree_equal(served_before, owner.served_snapshot())


@pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_external_loop_native_parity_and_exact_resume_at_particle_event(monkeypatch):
    from examples.e22_external_loop import make_loop, update

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda", torch.cuda.current_device())
    options = recipe()
    G, D, prior = (module.to(device) for module in components(options))
    native = GANTrainer(options, G, D, prior=prior, seed=101, serial_backward=True)
    external = make_loop(options, deepcopy(G), deepcopy(D), deepcopy(prior), seed=101)
    for index in range(3):
        native.step(real_batch(index).to(device))
        with torch.autograd.set_multithreading_enabled(False):
            update(external, real_batch(index).to(device))
    for owner in (native, external.policy):
        prime_table_verdict(owner, -1, block_scale=4)
        prime_particle_moves(owner)
    checkpoint = external.policy.state_dict()
    native_checkpoint = native.state_dict()
    expected_losses = native.step(real_batch(3).to(device), collect_stats=True)
    with torch.autograd.set_multithreading_enabled(False):
        actual_losses = update(external, real_batch(3).to(device), collect_stats=True)
    assert_tree_equal(expected_losses, actual_losses)
    assert_tree_equal(native.policy.state_dict(), external.policy.state_dict())
    expected = external.policy.state_dict()
    G, D, prior = (module.to(device) for module in components(options))
    restored = make_loop(options, G, D, prior, seed=1000)
    # Storage is commonly deserialized on CPU before loading into CUDA modules.
    # The compatible checkpoint still describes the same execution device.
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    cpu_checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    assert cpu_checkpoint["controller"]["latent_bandwidth"].device.type == "cpu"
    restored.policy.load_state_dict(cpu_checkpoint)
    with torch.autograd.set_multithreading_enabled(False):
        resumed = update(restored, real_batch(3).to(device), collect_stats=True)
    assert_tree_equal(expected_losses, resumed)
    assert_tree_equal(expected, restored.policy.state_dict())
    G, D, prior = (module.to(device) for module in components(options))
    restored_native = GANTrainer(options, G, D, prior=prior, seed=1000, serial_backward=True)
    buffer = io.BytesIO()
    torch.save(native_checkpoint, buffer)
    buffer.seek(0)
    restored_native.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    assert_tree_equal(expected_losses, restored_native.step(real_batch(3).to(device), collect_stats=True))
    assert_tree_equal(expected, restored_native.policy.state_dict())
    for with_noise in (False, True):
        expected_samples = native.sample(
            7, generator=torch.Generator(device=device).manual_seed(59), output_noise=with_noise)
        resumed_samples = restored.policy.served_model().sample(
            7, generator=torch.Generator(device=device).manual_seed(59), output_noise=with_noise)
        assert_tree_equal(expected_samples, resumed_samples)
