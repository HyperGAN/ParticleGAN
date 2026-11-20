"""Public EMPF observer controls on the continuous-vector width boundary."""
import pytest
import torch
from torch import nn

from particlegan import get_recipe, init


class ConditionalCritic(nn.Linear):
    def forward(self, rows, condition):
        return super().forward(torch.cat((rows, condition), dim=1)).squeeze(1)


def observe(real, fake, condition):
    recipe = get_recipe("empf", reg_every=1, batch_size=len(real))
    with torch.random.fork_rng(devices=[]):
        critic = ConditionalCritic(real.shape[1] + condition.shape[1], 1)
        init.deterministic_orthogonal_(critic, seed=0)
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer, input_domain="continuous", collect_stats=True)
    value = penalty(critic, real, fake, condition)
    value.backward()
    assert penalty.last_stats["applied"] and optimizer.record.calls == 1
    assert torch.isfinite(critic.weight.grad).all() and critic.weight.grad.norm() > 0
    assert optimizer.record.proto_sum is None
    return penalty.last_stats["ema_residual"]


@pytest.fixture(scope="module", params=(64, 65, 2150))
def measurements(request):
    # A block swap changes both conditional SDs while preserving every pooled row.
    width = request.param
    condition = torch.zeros(32, 7)
    condition[16:, 0] = 1
    real = torch.zeros(32, width)
    real[:, 0] = torch.tensor([-0.5, 0.5] * 8 + [-2., 2.] * 8)
    swapped = real.roll(16, dims=0)
    perm = torch.randperm(32, generator=torch.Generator().manual_seed(0))
    return (width, observe(real, real, condition), observe(real, swapped, condition),
            observe(real[perm], swapped[perm], condition[perm]), observe(real, real * .5, condition))


def test_pooled_contraction_and_joint_permutation(measurements):
    _, correct, swapped, reordered, halved = measurements
    assert correct == pytest.approx(0, abs=1e-6)
    assert halved > correct + 1e-5
    assert reordered == pytest.approx(swapped, abs=1e-6)


def test_observer_detects_swapped_conditional_spreads(measurements, request):
    width, correct, swapped, _, _ = measurements
    if width > 64:
        # Mark only this sensitivity assertion. A future fix yields strict XPASS.
        request.node.add_marker(pytest.mark.xfail(
            strict=True, reason="Released wide EMPF transport residual pools conditions (#555)"))
    assert swapped > correct + 1e-5
