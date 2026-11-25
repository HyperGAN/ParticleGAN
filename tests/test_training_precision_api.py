"""Tests for public training-precision helpers and Unified-v11/v12 BF16/fwAD contracts."""
from __future__ import annotations

import pytest
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

import particlegan
from particlegan import (
    GANTrainer,
    InputNoise,
    configure_bf16_autocast,
    critic_source_context,
    float_output,
    get_recipe,
    uses_bf16_autocast,
    uses_channels_last,
)
from particlegan.training import _convolution_autocast, _float_output
from particlegan.unified_v11 import bind_unified_v11_record, record_unified_v11_generator_probe


class _DictSubclass(dict):
    pass


class _ScoredWrapper(nn.Module):
    """Minimal analogue of HyperGAN's ScoredCritic (dynamic `.bf16` + `.critic` child)."""

    def __init__(self, critic: nn.Module, *, bf16: bool = False, channels_last: bool = False) -> None:
        super().__init__()
        self.critic = critic
        self.bf16 = bf16
        self.channels_last = channels_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with critic_source_context(self.critic, x):
            if not self.bf16 or x.device.type != "cuda":
                return self.critic(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inp = x.contiguous(memory_format=torch.channels_last) if (self.channels_last and x.ndim == 4) else x
                return float_output(self.critic(inp))


def test_public_exports_and_backward_compat_aliases() -> None:
    for name in (
        "float_output",
        "uses_channels_last",
        "uses_bf16_autocast",
        "configure_bf16_autocast",
        "critic_source_context",
    ):
        assert name in particlegan.__all__
        assert hasattr(particlegan, name)
        assert hasattr(particlegan.training, name)
    assert _float_output is float_output
    assert callable(_convolution_autocast)


def test_float_output_preserves_all_capture_metadata_containers_and_fwad() -> None:
    x_in = torch.randn(4, 3, 8, 8)
    cond = torch.tensor([0, 1, 0, 1])
    parts = (x_in,)
    rec = object()

    raw = torch.randn(4, 1, dtype=torch.bfloat16)
    raw._bispf_input = x_in
    raw._empf_input = x_in
    raw._empf_record = rec
    raw._empf_conditioned = True
    raw._empf_cond_tensors = (cond,)
    raw._t2_input = x_in
    raw._t2_record = rec
    raw._t2_conditioned = False
    raw._t2_cond_tensors = ()
    raw._rhbi_input = x_in
    raw._rhbi_cond = cond
    raw._rhbi_cont_cond = True
    raw._rhbi_parts = parts
    raw._rhbi_record = rec
    raw._rhbi_thermally_scaled = True
    raw._t1c_input = x_in
    raw._t1c_cond = cond
    raw._t1c_cont_cond = False
    raw._t1c_parts = parts
    raw._t1c_record = rec
    raw._t1c_thermally_scaled = True
    raw._t1c_bypassed = True
    raw._rgdt_input = x_in
    raw._rgdt_cond = cond
    raw._rgdt_cont_cond = True
    raw._rgdt_parts = parts
    raw._rgdt_record = rec
    raw._rgdt_thermally_scaled = False
    raw._t4_input = x_in
    raw._t4_record = rec
    raw._t4_conditioned = True
    raw._t4_cond_tensors = (cond,)
    raw._nash_v3_input = x_in
    raw._nash_v3_record = rec
    raw._nash_v3_conditioned = False
    raw._nash_v3_cond_tensors = ()
    raw._minimal_input = x_in
    raw._minimal_record = rec
    raw._minimal_conditioned = True
    raw._minimal_cond_tensors = (cond,)

    out = float_output(raw)
    assert out.dtype == torch.float32
    assert out is not raw
    for attr in (
        "_bispf_input",
        "_empf_input", "_empf_record", "_empf_conditioned", "_empf_cond_tensors",
        "_t2_input", "_t2_record", "_t2_conditioned", "_t2_cond_tensors",
        "_rhbi_input", "_rhbi_cond", "_rhbi_cont_cond", "_rhbi_parts", "_rhbi_record", "_rhbi_thermally_scaled",
        "_t1c_input", "_t1c_cond", "_t1c_cont_cond", "_t1c_parts", "_t1c_record", "_t1c_thermally_scaled", "_t1c_bypassed",
        "_rgdt_input", "_rgdt_cond", "_rgdt_cont_cond", "_rgdt_parts", "_rgdt_record", "_rgdt_thermally_scaled",
        "_t4_input", "_t4_record", "_t4_conditioned", "_t4_cond_tensors",
        "_nash_v3_input", "_nash_v3_record", "_nash_v3_conditioned", "_nash_v3_cond_tensors",
        "_minimal_input", "_minimal_record", "_minimal_conditioned", "_minimal_cond_tensors",
    ):
        got = getattr(out, attr)
        exp = getattr(raw, attr)
        if isinstance(exp, torch.Tensor):
            assert got is exp
        else:
            assert got == exp

    # Zero-copy fast path for non-BF16 tensors and containers without BF16 leaves
    fp32 = torch.randn(4, 1, dtype=torch.float32)
    assert float_output(fp32) is fp32
    fp32_tuple = (fp32, 7)
    assert float_output(fp32_tuple) is fp32_tuple
    fp32_dict = _DictSubclass(a=fp32)
    assert float_output(fp32_dict) is fp32_dict

    # Recursive container conversion preserves dict subclass type
    nested = _DictSubclass(a=[torch.randn(2, dtype=torch.bfloat16)], b=(torch.randn(2, dtype=torch.bfloat16), "tag"))
    casted = float_output(nested)
    assert isinstance(casted, _DictSubclass)
    assert casted["a"][0].dtype == torch.float32
    assert casted["b"][0].dtype == torch.float32
    assert casted["b"][1] == "tag"

    # Preserves fwAD dual tangents across BF16 -> FP32 cast
    with fwAD.dual_level():
        p = torch.randn(3, 2, dtype=torch.bfloat16)
        t = torch.randn(3, 2, dtype=torch.bfloat16)
        dual = fwAD.make_dual(p, t)
        dual._t1c_bypassed = True
        dual_out = float_output(dual)
        p_out, t_out = fwAD.unpack_dual(dual_out)
        assert p_out.dtype == torch.float32
        assert t_out is not None and t_out.dtype == torch.float32
        assert getattr(dual_out, "_t1c_bypassed", False) is True


def test_precision_predicates_and_configure_cpu() -> None:
    conv_wide = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1))
    conv_narrow = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1))
    mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))

    # CPU modules always stay FP32
    for mod in (conv_wide, conv_narrow, mlp):
        assert uses_channels_last(mod, "bf16_channels_last") is False
        assert uses_bf16_autocast(mod, "bf16_channels_last") is False
        assert uses_channels_last(mod, "float32") is False
        assert uses_bf16_autocast(mod, "float32") is False
        assert _convolution_autocast(mod, True) is False

    # Invalid precision strings rejected loudly
    with pytest.raises(ValueError, match="training_precision"):
        uses_channels_last(conv_wide, "fp16")
    with pytest.raises(ValueError, match="training_precision"):
        uses_bf16_autocast(conv_wide, "bf16")

    # Accepts Recipe instances directly
    rec_v12 = get_recipe("unified_v12")
    assert uses_channels_last(conv_wide, rec_v12) is False
    assert uses_bf16_autocast(conv_wide, rec_v12) is False

    # Formulation flags configured accurately and return bool(enabled)
    for preset, attr in (
        ("unified_v12", "_unified_v4_bf16_autocast"),
        ("unified_v11", "_unified_v4_bf16_autocast"),
        ("unified_v4", "_unified_v4_bf16_autocast"),
        ("t4_nash", "_t4_bf16_autocast"),
        ("t4_nash_v3", "_nash_v3_bf16_autocast"),
        ("minimal", "_minimal_bf16_autocast"),
    ):
        probe = nn.Sequential(nn.Linear(4, 4))
        rec = get_recipe(preset)
        assert configure_bf16_autocast(rec, probe, True) is True
        assert getattr(probe, attr) is True
        assert configure_bf16_autocast(rec, probe, False) is False
        assert getattr(probe, attr) is False
    assert configure_bf16_autocast(rec_v12, None, True) is True


def test_critic_source_context_reuses_live_endpoints_and_is_reentrant() -> None:
    torch.manual_seed(0)
    recipe = get_recipe("unified_v12", batch_size=4, z_dim=4, num_particles=16, total_steps=5, training_precision="float32")
    critic = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(16 * 8 * 8, 1))
    opt_d = recipe.make_critic_optimizer(critic)
    loss_fn = recipe.make_loss()
    bind_unified_v11_record(opt_d.record, loss_fn)
    penalty_fn = recipe.make_critic_penalty(opt_d, collect_stats=True)

    real = torch.randn(4, 3, 8, 8)
    fake = torch.randn(4, 3, 8, 8)

    class _CloneInputCritic(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.critic = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Simulates channels_last / noise allocating a fresh storage pointer.
            with critic_source_context(self, x):
                return self.critic(x.clone())

    wrapped = _CloneInputCritic(critic)
    d_real = wrapped(real)
    d_fake = wrapped(fake)
    assert getattr(critic, "_t2_pending_source_key", None) is None
    _ = loss_fn.d_loss(d_real, d_fake)
    pen = penalty_fn(wrapped, real, fake)
    assert torch.isfinite(pen)
    assert penalty_fn.last_stats["live_endpoints_reused"] is True

    # Re-entrant and exception-safe restoration of previous source key
    critic._t2_pending_source_key = ("outer",)
    with pytest.raises(RuntimeError, match="boom"):
        with critic_source_context(critic, real):
            assert critic._t2_pending_source_key != ("outer",)
            raise RuntimeError("boom")
    assert critic._t2_pending_source_key == ("outer",)
    critic._t2_pending_source_key = None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 precision verification")
def test_cuda_bf16_predicates_and_scored_critic_fwad_paths(monkeypatch) -> None:
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native BF16 not supported on this CUDA device")

    dev = torch.device("cuda")
    conv_wide = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1)).to(dev)
    conv_narrow = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1)).to(dev)
    mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)

    assert uses_channels_last(conv_wide, "bf16_channels_last") is True
    assert uses_bf16_autocast(conv_wide, "bf16_channels_last") is True
    assert uses_channels_last(conv_narrow, "bf16_channels_last") is False
    assert uses_bf16_autocast(conv_narrow, "bf16_channels_last") is False
    assert uses_channels_last(mlp, "bf16_channels_last") is False
    assert uses_bf16_autocast(mlp, "bf16_channels_last") is False
    assert uses_channels_last(conv_wide, "float32") is False
    assert uses_bf16_autocast(conv_wide, "float32") is False

    # ResNet, BatchNorm, and InstanceNorm critics all run native BF16 fwAD cleanly with live endpoint reuse
    recipe = get_recipe("unified_v12", batch_size=4, z_dim=8, num_particles=16, total_steps=5)
    critics = {
        "resnet": nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 16 * 16, 1),
        ),
        "bn_dcgan": nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 8 * 8, 1),
        ),
        "instance_norm": nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.InstanceNorm2d(64, affine=True), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 32 * 32, 1),
        ),
    }
    for name, mod in critics.items():
        mod = mod.to(dev)
        assert uses_channels_last(mod, recipe.training_precision) is True
        mod.to(memory_format=torch.channels_last)
        configure_bf16_autocast(recipe, mod, True)
        opt_d = recipe.make_critic_optimizer(mod)
        loss_fn = recipe.make_loss()
        bind_unified_v11_record(opt_d.record, loss_fn)
        penalty_fn = recipe.make_critic_penalty(opt_d, collect_stats=True)
        wrapped = _ScoredWrapper(mod, bf16=True, channels_last=True)

        real = torch.randn(4, 3, 32, 32, device=dev)
        fake = torch.randn(4, 3, 32, 32, device=dev)
        d_real = wrapped(real)
        d_fake = wrapped(fake)
        assert d_real.dtype == torch.float32 and d_fake.dtype == torch.float32
        assert getattr(d_real, "_t1c_bypassed", False) is True
        d_loss = loss_fn.d_loss(d_real, d_fake) + penalty_fn(wrapped, real, fake)
        assert torch.isfinite(d_loss), f"{name} nonfinite d_loss"
        assert penalty_fn.last_stats["live_endpoints_reused"] is True, f"{name} missed live reuse"
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in mod.parameters() if p.grad is not None), f"{name} nonfinite grad"
        opt_d.step()

    # Non-BN critic whose BF16 fwAD fails falls back cleanly to FP32 fwAD (not BF16 finite-diff)
    class _FailBf16Dual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.head = nn.Linear(64 * 8 * 8, 1)
            self.fp32_fwad_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x)
            _, tan = fwAD.unpack_dual(h)
            if tan is not None:
                if torch.is_autocast_enabled("cuda"):
                    raise RuntimeError("simulated exotic BF16 dual mismatch")
                self.fp32_fwad_calls += 1
            return self.head(h.flatten(1))

    exotic = _FailBf16Dual().to(dev).to(memory_format=torch.channels_last)
    opt_e = recipe.make_critic_optimizer(exotic)
    pen_e = recipe.make_critic_penalty(opt_e, collect_stats=True)
    wrapped_e = _ScoredWrapper(exotic, bf16=True, channels_last=True)
    real_e = torch.randn(4, 3, 8, 8, device=dev)
    fake_e = torch.randn(4, 3, 8, 8, device=dev)
    pen_val = pen_e(wrapped_e, real_e, fake_e)
    assert torch.isfinite(pen_val)
    assert exotic.fp32_fwad_calls >= 1
    assert wrapped_e.bf16 is True

    # Generator probe with configure_bf16_autocast runs both y_pre and y_post in BF16 cleanly
    gen = nn.Sequential(
        nn.Linear(8, 64 * 4 * 4),
        nn.Unflatten(1, (64, 4, 4)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 3, 4, 2, 1),
        nn.Tanh(),
    ).to(dev).to(memory_format=torch.channels_last)
    disc = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(64 * 4 * 4, 1),
    ).to(dev).to(memory_format=torch.channels_last)
    prior = recipe.make_prior().to(dev)
    opt_g, opt_d = recipe.make_optimizers(gen, disc, prior)
    bind_unified_v11_record(opt_d.record, opt_g)
    probe_dtypes = []
    hook = gen[4].register_forward_hook(lambda _m, _a, out: probe_dtypes.append(out.dtype))
    configure_bf16_autocast(recipe, gen, True)
    z, ids = prior.sample(4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fake_g = float_output(gen(z))
    assert record_unified_v11_generator_probe(gen, (z,), {}, fake_g, opt_d.record) is True
    opt_g.zero_grad(set_to_none=True)
    fake_g.square().mean().backward()
    opt_g.set_sampled_rows(prior.z, ids)
    opt_g.bind_protected_from_grads()
    opt_g.step()
    hook.remove()
    assert opt_g.probe_consumptions == 1
    assert probe_dtypes and all(dt == torch.bfloat16 for dt in probe_dtypes)

    # Unsupported CUDA fallback check
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)
    assert uses_channels_last(conv_wide, "bf16_channels_last") is False
    assert uses_bf16_autocast(conv_wide, "bf16_channels_last") is False
