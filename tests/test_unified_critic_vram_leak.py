"""Weakref, eviction, backward-alive semantics, and flat-VRAM regression tests for critic live-capture hooks."""
from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan._minimal.dynamics import MinimalStepRecord, _attach_minimal_critic_hook
from particlegan._minimal.geometry import MinimalCriticPenalty
from particlegan._nash_v3.penalty import _NashV3PenaltyKernel
from particlegan._nash_v3.record import NashV3StepRecord, _attach_nash_v3_critic_hook
from particlegan._t2_critic.critic import (
    _MAX_LIVE_CAPTURES,
    T2SecantStepRecord,
    _attach_t2_critic_hook,
)
from particlegan._t2_critic.penalty import _T2SecantPenaltyKernel
from particlegan._unified.critic import UnifiedStepRecord, _attach_unified_critic_hook
from particlegan._unified.penalty import _UnifiedPenaltyKernel
from particlegan._unified_v4.critic import UnifiedV4StepRecord, _attach_unified_v4_critic_hook
from particlegan._unified_v11.critic import UnifiedV11StepRecord, _attach_unified_v11_critic_hook
from particlegan._unified_v11.penalty import _UnifiedV11PenaltyKernel


class _SmallConvCritic(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).mean(dim=(2, 3))).squeeze(-1)


class _SmallConvGenerator(nn.Module):
    def __init__(self, z_dim: int = 8, channels: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim, 8 * 8 * 8)
        self.up1 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(8, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.fc(z)).reshape(z.shape[0], 8, 8, 8)
        h = nn.functional.silu(self.up1(h))
        return torch.tanh(self.up2(h))


_HOOK_FACTORIES = [
    pytest.param(UnifiedV11StepRecord, _attach_unified_v11_critic_hook, id="unified_v11"),
    pytest.param(UnifiedV4StepRecord, _attach_unified_v4_critic_hook, id="unified_v4"),
    pytest.param(UnifiedStepRecord, _attach_unified_critic_hook, id="unified"),
    pytest.param(T2SecantStepRecord, _attach_t2_critic_hook, id="t2_secant"),
    pytest.param(NashV3StepRecord, _attach_nash_v3_critic_hook, id="t4_nash_v3"),
    pytest.param(MinimalStepRecord, _attach_minimal_critic_hook, id="minimal"),
]


@pytest.mark.parametrize(("record_cls", "attach_fn"), _HOOK_FACTORIES)
@pytest.mark.parametrize("use_bf16", [False, True], ids=["fp32", "bf16"])
def test_live_capture_hook_releases_tensors_without_cycle(
    record_cls,
    attach_fn,
    use_bf16: bool,
) -> None:
    """`x_act` and `unscaled` must be GC'd after backward + clear, even if a caller holds `entry`."""
    torch.manual_seed(0)
    critic = _SmallConvCritic()
    if use_bf16:
        critic = critic.to(dtype=torch.bfloat16)
    record = record_cls()
    attach_fn(critic, record)

    x = torch.randn(4, 3, 32, 32, dtype=torch.bfloat16 if use_bf16 else torch.float32)
    out = critic(x)

    assert len(record._live_captures) == 1
    saved_entry = record._live_captures[0]
    assert isinstance(saved_entry[5], list)
    assert bool(saved_entry[5]) is True

    ref_x_act = weakref.ref(saved_entry[3])
    ref_unscaled = weakref.ref(saved_entry[4])

    out.mean().backward()
    assert bool(saved_entry[5]) is False

    record.clear_transient_captures()
    assert record._live_captures == []
    assert saved_entry[3] is None
    assert saved_entry[4] is None

    del x, out
    gc.collect()

    assert ref_x_act() is None, "x_act leaked after backward + clear_transient_captures"
    assert ref_unscaled() is None, "unscaled leaked after backward + clear_transient_captures"


@pytest.mark.parametrize(("record_cls", "attach_fn"), _HOOK_FACTORIES)
def test_live_capture_ring_eviction_releases_old_tensors(record_cls, attach_fn) -> None:
    """Evicting entries beyond `_MAX_LIVE_CAPTURES` must immediately drop tensor refs without leaking."""
    torch.manual_seed(0)
    critic = _SmallConvCritic()
    record = record_cls()
    attach_fn(critic, record)

    evicted_refs = []
    evicted_entries = []
    total_calls = _MAX_LIVE_CAPTURES + 4
    for i in range(total_calls):
        x = torch.randn(4, 3, 32, 32)
        out = critic(x)
        entry = record._live_captures[-1]
        if i < total_calls - _MAX_LIVE_CAPTURES:
            evicted_entries.append(entry)
            evicted_refs.append((weakref.ref(entry[3]), weakref.ref(entry[4])))
        out.mean().backward()
        del x, out, entry

    gc.collect()
    assert len(record._live_captures) == _MAX_LIVE_CAPTURES
    for entry, (ref_x, ref_u) in zip(evicted_entries, evicted_refs):
        assert entry[3] is None and entry[4] is None
        assert ref_x() is None, "evicted x_act leaked"
        assert ref_u() is None, "evicted unscaled leaked"


@pytest.mark.parametrize(("record_cls", "attach_fn"), _HOOK_FACTORIES)
def test_live_capture_load_state_dict_releases_tensors(record_cls, attach_fn) -> None:
    """Restoring a record checkpoint must clear tensor refs from any prior `_live_captures`."""
    torch.manual_seed(0)
    critic = _SmallConvCritic()
    record = record_cls()
    attach_fn(critic, record)
    snap = record.state_dict()

    x = torch.randn(4, 3, 32, 32)
    out = critic(x)
    saved_entry = record._live_captures[0]
    ref_x_act = weakref.ref(saved_entry[3])
    ref_unscaled = weakref.ref(saved_entry[4])

    record.load_state_dict(snap)
    assert record._live_captures == []
    assert saved_entry[3] is None
    assert saved_entry[4] is None

    del x, out
    gc.collect()
    assert ref_x_act() is None, "x_act leaked after load_state_dict"
    assert ref_unscaled() is None, "unscaled leaked after load_state_dict"


@pytest.mark.parametrize(
    ("record_cls", "attach_fn", "kernel_cls"),
    [
        pytest.param(UnifiedV11StepRecord, _attach_unified_v11_critic_hook, _UnifiedV11PenaltyKernel, id="unified_v11"),
        pytest.param(UnifiedStepRecord, _attach_unified_critic_hook, _UnifiedPenaltyKernel, id="unified"),
        pytest.param(T2SecantStepRecord, _attach_t2_critic_hook, _T2SecantPenaltyKernel, id="t2_secant"),
        pytest.param(NashV3StepRecord, _attach_nash_v3_critic_hook, _NashV3PenaltyKernel, id="t4_nash_v3"),
    ],
)
def test_live_capture_alive_flag_prevents_reuse_after_backward(
    record_cls,
    attach_fn,
    kernel_cls,
) -> None:
    """Before backward, live endpoint lookup hits; after backward, hook clears `alive` so lookup misses."""
    torch.manual_seed(0)
    critic = _SmallConvCritic()
    record = record_cls()
    attach_fn(critic, record)
    kernel = kernel_cls(record=record)

    x_real = torch.randn(4, 3, 32, 32)
    x_fake = torch.randn(4, 3, 32, 32)
    d_real = critic(x_real)
    d_fake = critic(x_fake)

    hit_before = kernel._lookup_live_endpoint(x_real, ())
    assert hit_before is not None and hit_before[1] is d_real

    (d_real.mean() - d_fake.mean()).backward()

    hit_after = kernel._lookup_live_endpoint(x_real, ())
    assert hit_after is None, "already-backwarded endpoint must not be reused"

    # Calling penalty after separate backward must succeed cleanly via fresh forward.
    pen_val, _ = kernel.penalty(critic, x_real, x_fake, step=1)
    assert torch.isfinite(pen_val)
    pen_val.backward()
    record.clear_transient_captures()
    assert record._live_captures == []


def test_minimal_penalty_skips_backwarded_capture_without_crash() -> None:
    """MinimalCriticPenalty must skip already-backwarded live captures cleanly."""
    torch.manual_seed(0)
    critic = _SmallConvCritic()
    recipe = get_recipe("minimal", training_precision="float32")
    opt_d = recipe.make_critic_optimizer(critic)
    pen = MinimalCriticPenalty(recipe, opt_d)
    record = opt_d.record

    x_real = torch.randn(4, 3, 32, 32)
    x_fake = torch.randn(4, 3, 32, 32)
    d_real = critic(x_real)
    d_fake = critic(x_fake)
    assert len(record._live_captures) == 2
    assert all(bool(entry[5]) for entry in record._live_captures)

    (d_real.mean() - d_fake.mean()).backward()
    assert all(not bool(entry[5]) for entry in record._live_captures)

    # Penalty after separate backward must not try to backward through freed graphs.
    pen_val = pen(critic, x_real, x_fake)
    assert torch.isfinite(pen_val)
    pen_val.backward()
    record.clear_transient_captures()
    assert record._live_captures == []


@pytest.mark.parametrize("preset", ["unified_v11", "unified_v4", "unified"])
def test_gan_trainer_cpu_releases_all_step_batches(preset: str) -> None:
    """Running `GANTrainer` across steps retains zero weakrefs to past real batches on CPU."""
    torch.manual_seed(0)
    recipe = get_recipe(
        preset,
        z_dim=8,
        batch_size=8,
        num_particles=16,
        total_steps=20,
        training_precision="float32",
    )
    gen = _SmallConvGenerator(z_dim=8, channels=3)
    disc = _SmallConvCritic(channels=3)
    trainer = GANTrainer(recipe, gen, disc, seed=0)

    batch_refs = []
    for _ in range(12):
        x_real = torch.randn(8, 3, 32, 32)
        batch_refs.append(weakref.ref(x_real))
        trainer.step(x_real)
        del x_real

    gc.collect()
    leaked = sum(1 for ref in batch_refs if ref() is not None)
    assert leaked == 0, f"{preset}: {leaked}/{len(batch_refs)} real batches were retained"
    assert getattr(trainer.opt_d.record, "_live_captures", []) == []


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VRAM leak regression test")
@pytest.mark.parametrize("preset", ["unified_v11", "unified_v4", "unified"])
def test_gan_trainer_cuda_flat_vram_over_many_steps(preset: str) -> None:
    """CUDA VRAM must stay flat over 40 CIFAR-10-sized steps with zero retained batch weakrefs."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    recipe = get_recipe(
        preset,
        z_dim=8,
        batch_size=32,
        num_particles=16,
        total_steps=60,
    )
    gen = _SmallConvGenerator(z_dim=8, channels=3).to(device)
    disc = _SmallConvCritic(channels=3).to(device)
    trainer = GANTrainer(recipe, gen, disc, seed=0)

    # Warm up allocator, optimizer states, and any lazy buffers.
    for _ in range(5):
        x_warm = torch.randn(32, 3, 32, 32, device=device)
        trainer.step(x_warm)
        del x_warm

    gc.collect()
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    batch_refs = []
    for _ in range(40):
        x_real = torch.randn(32, 3, 32, 32, device=device)
        batch_refs.append(weakref.ref(x_real))
        trainer.step(x_real)
        del x_real

    gc.collect()
    torch.cuda.synchronize(device)
    mem_after = torch.cuda.memory_allocated(device)

    leaked = sum(1 for ref in batch_refs if ref() is not None)
    assert leaked == 0, f"{preset}: {leaked}/{len(batch_refs)} CUDA input batches leaked"
    assert getattr(trainer.opt_d.record, "_live_captures", []) == []
    # Without the fix, 40 steps x 2 batches x (32*3*32*32*4 bytes) + autograd graphs leaks > 30 MiB.
    assert mem_after - mem_before <= 2 * 1024 * 1024, (
        f"{preset}: CUDA memory grew by {(mem_after - mem_before) / (1024 * 1024):.2f} MiB over 40 steps "
        f"(before={mem_before}, after={mem_after})"
    )
