"""Software contracts for publishing a completed policy update as inference state."""
from copy import deepcopy

import pytest
import torch
from test_e22_policy import assert_tree_equal, paired_loops, real_batch

from examples.e22_external_loop import update


@pytest.mark.parametrize("owner_kind", ["trainer", "policy"])
@pytest.mark.parametrize("export", ["served_snapshot", "served_model"])
def test_export_rejects_every_active_update_phase_without_changing_the_update(monkeypatch, owner_kind, export):
    native, external = paired_loops()
    owner = native if owner_kind == "trainer" else external.policy
    policy = native.policy if owner_kind == "trainer" else external.policy
    phases = []

    def check_export():
        phases.append(policy._phase)
        parameters = [p for module in policy._training_modules().values() for p in module.parameters()]
        versions = [p._version for p in parameters]
        noise, fast = policy._noise, policy._fast
        streams = {name: getattr(policy, name).get_state().clone() for name in policy._STREAMS}
        cpu_rng = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="serving snapshots require a completed update boundary"):
            getattr(owner, export)()
        assert phases[-1] == policy._phase
        assert policy._noise is noise and policy._fast is fast
        assert versions == [p._version for p in parameters]
        assert torch.equal(cpu_rng, torch.get_rng_state())
        for name, before in streams.items():
            assert torch.equal(before, getattr(policy, name).get_state())

    def checked_hook(original):
        def hook(*args, **kwargs):
            result = original(*args, **kwargs)
            check_export()
            return result
        return hook

    for name in ("begin_step", "after_critic_step", "after_generator_backward", "after_generator_step"):
        monkeypatch.setattr(policy, name, checked_hook(getattr(policy, name)))

    cpu_rng = torch.get_rng_state()
    expected = native.step(real_batch(0))
    torch.set_rng_state(cpu_rng)
    with torch.autograd.set_multithreading_enabled(False):
        actual = update(external, real_batch(0))
    assert phases == ["critic", "generator", "generator_step", "finish"]
    assert_tree_equal(expected, actual)
    assert_tree_equal(native.policy.state_dict(), external.policy.state_dict())
    assert_tree_equal(native.served_snapshot(), external.policy.served_snapshot())
    assert getattr(owner, export)() is not None


@pytest.mark.parametrize("export", ["served_snapshot", "served_model"])
def test_generator_real_callback_can_use_a_previously_published_model_but_cannot_export(export):
    native, _ = paired_loops()
    published = native.served_model()
    def published_output():
        return published.generate(published.table[:2], generator=torch.Generator().manual_seed(29))

    expected = published_output()

    def next_real():
        assert_tree_equal(expected, published_output())
        with pytest.raises(RuntimeError, match="serving snapshots require a completed update boundary"):
            getattr(native, export)()
        return real_batch(1)

    native.step(real_batch(0), generator_real=next_real)
    assert_tree_equal(expected, published_output())
    assert published.completed_steps == 0 and native.completed_steps == 1


def test_feature_average_expiry_selects_fast_and_keeps_published_model_independent():
    from test_feature_auto import equal, make, real

    trainer = make()
    for step in range(8):
        trainer.step(real() + step * .001)
    birth = trainer.birth_death
    # A consistent measured-geometry fixture, as in the restore rejection test.
    # No quality or distribution claim is inferred from this forced eligibility.
    stamp = deepcopy(birth.paired_average)
    assert stamp["chart_valid"] and stamp["duplicate_ok"]
    stamp.update(finite_rows=1024, same_group_rows=1024, ema_eligible_rows=1024,
                 coherent_rows=1024, eligible=True)
    birth.paired_average = stamp
    birth.last["paired_average"] = deepcopy(stamp)
    birth.rows_since_eval = birth.N - 1
    trainer._serve_apply()
    assert trainer._fast is not None
    published = trainer.served_model()
    frozen = deepcopy(published.generator.state_dict())
    assert published.source == trainer.policy.state_dict()["served_source"] == "averaged"

    birth.rows_since_eval = birth.N
    snapshot = trainer.served_snapshot()
    assert snapshot["source"] == trainer.policy.state_dict()["served_source"] == "fast"
    assert trainer._fast is None
    equal(snapshot["models"]["generator"], trainer.G.state_dict())
    equal(frozen, published.generator.state_dict())
    assert published.source == "averaged"
