"""Native-format software reference, gradient-penalty and checkpoint contracts."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.precision import prepare
from experiments.forge.state import state_digest
from particlegan import init
from particlegan.low_precision import (
    EmulatedLinear,
    _fp4,
    emulate_linear_training_,
    quantize_dequantize,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def task(format):
    return json.loads((ROOT / f"configs/forge/tasks/precision_gaussian_{format}.json").read_text())


def context_and_trainer(format):
    from experiments.forge.vectorprofiles import build_vector_models
    value = task(format)
    context = task_formulation_context({"recipe_preset": "bispf_optimized"}, value, {"seed": 0}, device="cpu")
    g, d = build_vector_models(context, value["execution"]["host_definition"])
    prepare(context, value, g, d)
    return context, context.build_trainer(g, d, max_steps=1000)


def test_e2m1_values_midpoint_ties_even_saturation_and_sign():
    values = torch.tensor([0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 20.])
    expected = torch.tensor([0., 0., 1., 1., 2., 2., 4., 4., 6.])
    torch.testing.assert_close(_fp4(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(_fp4(-values), -expected, rtol=0, atol=0)


def test_stochastic_fp4_rounding_is_replayable_and_unbiased():
    values = torch.full((32768,), .75)
    a, b = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    first = _fp4(values, generator=a)
    assert torch.equal(first, _fp4(values, generator=b))
    assert set(first.unique().tolist()) == {.5, 1.}
    assert abs(first.mean().item() - .75) < .01


def test_nvfp4_weight_tiles_are_transpose_equivalent_and_padding_is_zero():
    values = torch.randn(17, 35, generator=torch.Generator().manual_seed(0))
    left = quantize_dequantize(values, "nvfp4", weight=True)
    right = quantize_dequantize(values.T, "nvfp4", weight=True).T
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert left.shape == values.shape and torch.isfinite(left).all()
    assert torch.count_nonzero(quantize_dequantize(torch.zeros(1, 2), "nvfp4")) == 0


def test_nvfp4_uses_hierarchical_fp8_block_scales_and_not_integer_levels():
    values = torch.zeros(16, 32)
    values[0, 0] = 2688.  # global scale = 1; first block scale = 448
    values[0, 16:20] = torch.tensor([31., 7., 13., 21.])
    # Second block scale rounds (31/6) to E4M3 value 5. E2M1 then rounds
    # [6.2, 1.4, 2.6, 4.2] to [6, 1.5, 3, 4].
    result = quantize_dequantize(values, "nvfp4", weight=True)
    torch.testing.assert_close(result[0, 16:20], torch.tensor([30., 7.5, 15., 20.]), rtol=0, atol=0)
    assert result[0, 0] == 2688.


def test_conversion_preserves_shared_parameters_and_global_rng():
    linear = torch.nn.Linear(2, 3)
    network = torch.nn.ModuleDict({"left": linear, "right": linear})
    before = torch.get_rng_state().clone()
    parameters = dict(network.named_parameters())
    emulate_linear_training_(network, "nvfp4")
    assert network["left"] is network["right"]
    assert all(dict(network.named_parameters())[name] is value for name, value in parameters.items())
    assert torch.equal(before, torch.get_rng_state())
    assert isinstance(emulate_linear_training_(torch.nn.Linear(1, 2), "mxfp8"), EmulatedLinear)


def test_mxfp8_scale_is_power_of_two_rounded_up_and_blocks_are_independent():
    values = torch.cat((torch.linspace(-5., 5., 32), torch.linspace(-.001, .001, 32))).reshape(1, -1)
    actual = quantize_dequantize(values, "mxfp8")
    rounded = values.bfloat16().float().reshape(1, 2, 32)
    # These two independent scalar maxima require scales 2^-6 and 2^-18.
    scales = torch.tensor([2.**-6, 2.**-18]).reshape(1, 2, 1)
    expected = ((rounded / scales).to(torch.float8_e4m3fn).float() * scales).reshape_as(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mixed = torch.tensor([[1000., 1.], [.0001, .0001]])
    assert not torch.equal(quantize_dequantize(mixed.T, "mxfp8").T, quantize_dequantize(mixed, "mxfp8"))


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_forward_backward_and_critic_penalty_retain_a_finite_training_signal(format):
    layer = EmulatedLinear.from_linear(torch.nn.Linear(3, 5), format)
    init.deterministic_orthogonal_(layer)
    x = torch.randn(7, 3, generator=torch.Generator().manual_seed(0), requires_grad=True)
    output = layer(x)
    gradient = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
    penalty_gradient = torch.autograd.grad(gradient.square().mean(), layer.weight)[0]
    assert torch.isfinite(penalty_gradient).all() and penalty_gradient.abs().max() > 0
    assert layer.calls == {"forward": 1, "dgrad": 1, "wgrad": 1}
    assert not torch.equal(output, torch.nn.functional.linear(x, layer.weight, layer.bias))


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_public_trainer_resume_restores_rounding_streams_and_penalty_state(format):
    first, trainer = context_and_trainer(format)
    batches = torch.randn(3, 128, 1, generator=torch.Generator().manual_seed(0)) * .5 + 2.
    trainer.step(batches[0])
    checkpoint = deepcopy(first.state_dict())
    second, restored = context_and_trainer(format)
    second.load_state_dict(checkpoint)
    for batch in batches[1:]:
        trainer.step(batch)
        restored.step(batch)
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())
    before = {name: layer.rounding_rng.get_state().clone()
              for name, layer in trainer.G.named_modules() if isinstance(layer, EmulatedLinear)}
    trainer.sample(128, generator=torch.Generator().manual_seed(2), output_noise=False)
    assert all(torch.equal(before[name], layer.rounding_rng.get_state())
               for name, layer in trainer.G.named_modules() if isinstance(layer, EmulatedLinear))


def test_precision_conditions_keep_initial_master_parameters_and_other_streams_identical():
    contexts = [context_and_trainer(format) for format in ("fp32", "mxfp8", "nvfp4")]
    for role in ("G", "D", "prior"):
        states = [state_digest(dict(getattr(trainer, role).named_parameters())) for _, trainer in contexts]
        assert len(set(states)) == 1
    assert len({state_digest(context.streams.state_dict()) for context, _ in contexts}) == 1


def test_quantizer_rejects_unknown_formats_and_checkpoint_precision_changes():
    with pytest.raises(ValueError):
        quantize_dequantize(torch.ones(2, 2), "int4")
    a = EmulatedLinear.from_linear(torch.nn.Linear(2, 3), "nvfp4")
    b = EmulatedLinear.from_linear(torch.nn.Linear(2, 3), "mxfp8")
    with pytest.raises(ValueError, match="precision differs"):
        b.load_state_dict(a.state_dict())


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_emulated_linear_fwad_jvp_flows_gradients_with_zero_autograd_grad(format, dtype, monkeypatch):
    import torch.autograd.forward_ad as fwAD

    layer = EmulatedLinear.from_linear(torch.nn.Linear(4, 6), format)
    init.deterministic_orthogonal_(layer)
    layer = layer.to(dtype=dtype)
    x = torch.randn(5, 4, dtype=dtype)
    v = torch.randn(5, 4, dtype=dtype)

    def _forbid_autograd_grad(*_args, **_kwargs):
        raise AssertionError("torch.autograd.grad must not be called")

    monkeypatch.setattr(torch.autograd, "grad", _forbid_autograd_grad)
    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        dual_y = layer(dual_x)
        primal, tangent = fwAD.unpack_dual(dual_y)
    assert primal.dtype == dtype and tangent is not None and tangent.dtype == dtype
    loss = (primal.float().square().mean() + tangent.float().square().mean())
    loss.backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().max() > 0
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    assert layer.bias.grad.abs().max() > 0


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_emulated_linear_supports_multiple_dtypes_and_cpu_bf16_autocast(format, dtype):
    import torch.autograd.forward_ad as fwAD

    layer = EmulatedLinear.from_linear(torch.nn.Linear(4, 5), format).to(dtype=dtype)
    x = torch.randn(6, 4, dtype=dtype, requires_grad=True)
    y = layer(x)
    assert y.dtype == dtype and torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad.dtype == dtype and layer.weight.grad.dtype == dtype and layer.bias.grad.dtype == dtype

    layer.zero_grad(set_to_none=True)
    x_auto = torch.randn(6, 4, dtype=torch.float32, requires_grad=True)
    v_auto = torch.randn(6, 4, dtype=torch.float32)
    layer_fp32 = EmulatedLinear.from_linear(torch.nn.Linear(4, 5), format)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        y_auto = layer_fp32(x_auto)
        assert torch.isfinite(y_auto).all()
        with fwAD.dual_level():
            p_auto, j_auto = fwAD.unpack_dual(layer_fp32(fwAD.make_dual(x_auto.detach(), v_auto)))
            assert j_auto is not None and torch.isfinite(p_auto).all() and torch.isfinite(j_auto).all()
    y_auto.float().sum().backward()
    assert layer_fp32.weight.grad is not None and torch.isfinite(layer_fp32.weight.grad).all()


@pytest.mark.parametrize("bad_state", [
    torch.tensor([], dtype=torch.uint8),
    torch.tensor(4, dtype=torch.uint8),
    torch.tensor([4], dtype=torch.uint8),
    torch.tensor([4, 0, 0], dtype=torch.int64),
    torch.tensor([4, 1, 2, 3], dtype=torch.uint8),
    None,
    "malformed",
])
def test_emulated_linear_set_extra_state_rejects_malformed_inputs_with_value_error(bad_state):
    layer = EmulatedLinear.from_linear(torch.nn.Linear(2, 3), "nvfp4")
    before = layer.rounding_rng.get_state().clone()
    with pytest.raises(ValueError):
        layer.set_extra_state(bad_state)
    assert torch.equal(layer.rounding_rng.get_state(), before)


def test_trainer_load_state_dict_preflights_emulated_precision_atomically():
    from particlegan import GANTrainer, get_recipe, validate_emulated_extra_states

    recipe = get_recipe("bispf", num_particles=16, z_dim=2, batch_size=8, total_steps=10)
    g1 = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    d1 = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    emulate_linear_training_(g1, "nvfp4")
    emulate_linear_training_(d1, "nvfp4")
    t1 = GANTrainer(recipe, g1, d1, seed=0, max_steps=10)
    t1.step(torch.randn(8, 2))
    ckpt = deepcopy(t1.state_dict())

    g2 = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    d2 = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    emulate_linear_training_(g2, "mxfp8")
    emulate_linear_training_(d2, "mxfp8")
    t2 = GANTrainer(recipe, g2, d2, seed=1, max_steps=10)

    before_g = {k: v.detach().clone() for k, v in t2.G.named_parameters()}
    before_d = {k: v.detach().clone() for k, v in t2.D.named_parameters()}

    with pytest.raises(ValueError, match="precision differs"):
        validate_emulated_extra_states(t2.G, ckpt["models"]["G"])
    with pytest.raises(ValueError, match="precision differs"):
        t2.load_state_dict(ckpt)
    for k, v in t2.G.named_parameters():
        assert torch.equal(v, before_g[k]), f"G.{k} mutated on rejected load"
    for k, v in t2.D.named_parameters():
        assert torch.equal(v, before_d[k]), f"D.{k} mutated on rejected load"


def test_emulated_linear_direct_and_container_load_state_dict_atomic():
    base_nv = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
    base_mx = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
    emulate_linear_training_(base_nv, "nvfp4", seed_for=lambda _: 0)
    emulate_linear_training_(base_mx, "mxfp8", seed_for=lambda _: 0)

    # Direct single-layer load refuses without mutating weight/bias
    el_nv, el_mx = base_nv[0], base_mx[0]
    w0, b0 = el_nv.weight.detach().clone(), el_nv.bias.detach().clone()
    with pytest.raises(ValueError, match="precision differs"):
        el_nv.load_state_dict(el_mx.state_dict())
    assert torch.equal(el_nv.weight, w0) and torch.equal(el_nv.bias, b0)

    # Container load refuses without mutating any layer weight/bias
    before = {k: v.detach().clone() for k, v in base_nv.named_parameters()}
    with pytest.raises(ValueError, match="precision differs"):
        base_nv.load_state_dict(base_mx.state_dict())
    for k, v in base_nv.named_parameters():
        assert torch.equal(v, before[k]), f"{k} mutated on rejected container load"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device move test")
@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_emulated_linear_apply_updates_rounding_rng_on_cuda_roundtrip(format):
    layer = EmulatedLinear.from_linear(torch.nn.Linear(4, 6), format)
    assert layer.rounding_rng.device.type == "cpu"
    layer.to("cuda")
    assert layer.weight.device.type == "cuda"
    assert layer.rounding_rng.device.type == "cuda"
    x = torch.randn(4, 4, device="cuda", requires_grad=True)
    layer(x).sum().backward()
    layer.to("cpu")
    assert layer.weight.device.type == "cpu"
    assert layer.rounding_rng.device.type == "cpu"
    layer(torch.randn(4, 4)).sum().backward()
