"""Unit tests for Axiom v6 L0 state & capture layer (`particlegan._axiom.state`)."""

from __future__ import annotations

import ast
from copy import copy, deepcopy
import gc
import math
from pathlib import Path
import random
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from particlegan._axiom import state as axiom_state
from particlegan._axiom.state import (
    SCHEMA_VERSION,
    AxiomStepRecord,
    CaptureEntry,
    CaptureRegistry,
    WeakTensorMap,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    axiom_source_context,
    bind_axiom_optimizers,
    bind_axiom_record,
    clear_active_prior_rows,
    eval_probe_forward,
    get_active_prior_rows,
    get_axiom_record,
    get_module_record,
    get_tensor_record,
    lookup_axiom_capture,
    override_axiom_capture_input,
    recover_active_prior_rows,
    set_active_prior_rows,
    set_tensor_record,
)
from particlegan.training import float_output


# ---------------------------------------------------------------------------
# 1. Static Hygiene & Purity Contracts (L0, Zero Forbidden Ops, Zero Legacy)
# ---------------------------------------------------------------------------


def test_state_module_l0_purity_and_zero_forbidden_ops() -> None:
    src_path = Path(axiom_state.__file__)
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Zero internal _axiom imports and zero legacy sibling imports.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert node.level == 0, f"L0 state.py must have zero relative imports, got level={node.level} mod={mod!r}"
            assert "_axiom" not in mod, f"L0 state.py must not import internal _axiom modules: {mod!r}"
            for legacy in (
                "_unified",
                "bispf",
                "rhbi",
                "empf",
                "minimal",
                "k3p",
                "ka2",
                "grad_regularizers",
                "_t1",
                "_t2",
                "_t3",
                "_t4",
            ):
                assert legacy not in mod, f"L0 state.py imported forbidden legacy module: {mod!r}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "_axiom" not in alias.name

    # Zero forbidden PyTorch ops (autograd.grad, svd, eigh, pinv, compile).
    forbidden_attrs = {"svd", "svdvals", "eigh", "eigvalsh", "pinv", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs, f"Forbidden op {node.attr!r} found in state.py"
            if node.attr == "grad" and isinstance(node.value, ast.Attribute) and node.value.attr == "autograd":
                raise AssertionError("Forbidden torch.autograd.grad found in state.py")


# ---------------------------------------------------------------------------
# 2. Schema-2 Checkpoint Roundtrip, Packed GPU EMA Flush & Fail-Closed Reject
# ---------------------------------------------------------------------------


def test_step_record_schema2_roundtrip_copy_and_fail_closed_rejection() -> None:
    torch.manual_seed(0)
    rec = AxiomStepRecord()
    assert rec.schema == 2 and SCHEMA_VERSION == 2

    rec.calls = 11
    rec.observed_steps = 9
    rec.chord_phase = 0.375
    rec.last_ratio = 1.25
    rec.observe_grad_sq_tensor(torch.tensor(2.5))
    rec.observe_margin_and_scale(
        torch.ones(8, 1) * 1.5,
        -torch.ones(8, 1) * 0.5,
        torch.randn(8, 4),
        torch.randn(8, 4),
    )
    rec.observe_thermal(0.05, 0.01, -0.02, 0.03, 0.04, r_tail=0.01)
    rec.chart_state = {"centers": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    rec.atomic_ledger = {"row_chart": torch.tensor([0, 1], dtype=torch.long)}
    rec.cond_fiber_bank = {"schema": 2, "strata": []}
    rec.cond_initialized = True
    rec.cond_ema = 0.42
    rec.het_initialized = True
    rec.het_ema = 2.1
    rec.step_scale = 0.75
    rec.g_func_gain = 0.8

    # Pending device EMA tensors exist before readout/save, then flush cleanly in <= 1 sync on state_dict().
    keys_before, _ = rec.take_pending_ema_tensors()
    assert "ema_grad_sq" in keys_before and "ema_data_scale" in keys_before
    sync_count = {"n": 0}
    orig_item, orig_tolist = torch.Tensor.item, torch.Tensor.tolist

    def _cnt_item(self: torch.Tensor, *a, **kw):
        sync_count["n"] += 1
        return orig_item(self, *a, **kw)

    def _cnt_tolist(self: torch.Tensor, *a, **kw):
        sync_count["n"] += 1
        return orig_tolist(self, *a, **kw)

    torch.Tensor.item = _cnt_item  # type: ignore[assignment]
    torch.Tensor.tolist = _cnt_tolist  # type: ignore[assignment]
    try:
        sd = rec.state_dict()
    finally:
        torch.Tensor.item = orig_item  # type: ignore[assignment]
        torch.Tensor.tolist = orig_tolist  # type: ignore[assignment]
    assert sync_count["n"] <= 1, f"Expected <=1 packed flush sync on state_dict(), got {sync_count['n']}"
    assert sd["schema"] == 2
    assert rec.take_pending_ema_tensors() == ([], [])

    rec2 = AxiomStepRecord()
    rec2.load_state_dict(sd)
    assert rec2.state_dict()["schema"] == 2
    assert rec2.calls == 11 and rec2.observed_steps == 9
    assert math.isclose(rec2.chord_phase, 0.375)
    assert math.isclose(rec2.last_ratio or 0.0, 1.25)
    assert math.isclose(rec2.ema_grad_sq, rec.ema_grad_sq)
    assert math.isclose(rec2.ema_data_scale, rec.ema_data_scale)
    assert math.isclose(rec2.thermal_scale, rec.thermal_scale)
    assert torch.equal(rec2.chart_state["centers"], rec.chart_state["centers"])
    assert torch.equal(rec2.atomic_ledger["row_chart"], rec.atomic_ledger["row_chart"])

    # base_state compatibility adapter for plain_adam / NormalizedOptimizer.
    base = AxiomStepRecord.base_state(sd)
    assert set(base) == {
        "lr_max",
        "lr_last",
        "anchor_started",
        "calls",
        "observed_steps",
    }
    assert base["calls"] == 11 and base["observed_steps"] == 9 and base["anchor_started"] is False

    # copy() and deepcopy() produce independent state copies.
    rec._enable_secant_probe = False
    dup_shallow = copy(rec)
    dup_deep = deepcopy(rec)
    assert dup_shallow.calls == 11 and dup_deep._enable_secant_probe is False
    dup_deep.calls = 99
    assert rec.calls == 11

    # Fail-closed rejection: schema-1, missing/extra keys, wrong types, non-finite, out-of-range.
    bad_mutations = [
        lambda d: d.__setitem__("schema", 1),
        lambda d: d.__setitem__("schema", 3),
        lambda d: d.__setitem__("schema", True),
        lambda d: d.pop("calls"),
        lambda d: d.__setitem__("unexpected_extra", 0),
        lambda d: d.__setitem__("calls", -1),
        lambda d: d.__setitem__("calls", True),
        lambda d: d.__setitem__("observed_steps", -3),
        lambda d: d.__setitem__("anchor_started", True),
        lambda d: d.__setitem__("grad_sq_initialized", 1),
        lambda d: d.__setitem__("last_ratio", -0.1),
        lambda d: d.__setitem__("last_ratio", float("nan")),
        lambda d: d.__setitem__("chord_phase", 1.0),
        lambda d: d.__setitem__("chord_phase", -0.01),
        lambda d: d.__setitem__("ema_grad_sq", -0.5),
        lambda d: d.__setitem__("ema_margin", 61.0),
        lambda d: d.__setitem__("ema_data_scale", 0.0),
        lambda d: d.__setitem__("last_data_scale", 0.0),
        lambda d: d.__setitem__("thermal_scale", 1e-6),
        lambda d: d.__setitem__("thermal_scale", 1.1),
        lambda d: d.__setitem__("step_scale", 0.0),
        lambda d: d.__setitem__("g_func_gain", 0.0),
        lambda d: d.__setitem__("cond_ema", -0.6),
        lambda d: d.__setitem__("ema_ed", float("inf")),
        lambda d: d.__setitem__("chart_state", "not_a_dict"),
        lambda d: d.__setitem__("chart_state", {"bad": torch.tensor([float("nan")])}),
        lambda d: d.__setitem__("atomic_ledger", {"bad": float("inf")}),
        lambda d: d.__setitem__("cond_fiber_bank", {"bad_obj": object()}),
    ]
    # Verify atomic rejection leaves target untouched.
    before_sd = rec2.state_dict()
    for mutate in bad_mutations:
        bad = deepcopy(sd)
        mutate(bad)
        with pytest.raises(ValueError):
            rec2.load_state_dict(bad)
        with pytest.raises(ValueError):
            AxiomStepRecord.base_state(bad)
    assert rec2.calls == before_sd["calls"] and rec2.observed_steps == before_sd["observed_steps"]


def test_observe_margin_and_scale_homogeneity_spike_immunity_and_2step_shock_reset() -> None:
    torch.manual_seed(0)
    xr0, xf0 = torch.randn(16, 4), torch.randn(16, 4)
    z1, z0 = torch.ones(16, 1), -torch.ones(16, 1)

    # Exact degree-1 scale homogeneity across 1e-20 .. 1e20 and channels_last layout invariance.
    normed = []
    for scale in (1e-20, 1e-6, 1.0, 1e6, 1e20):
        rec = AxiomStepRecord()
        rec.observe_margin_and_scale(z1, z0, xr0 * scale, xf0 * scale)
        normed.append(rec.ema_data_scale / scale)
    for val in normed[1:]:
        assert math.isclose(normed[0], val, rel_tol=1e-5)

    img_r = torch.randn(8, 3, 16, 16)
    img_f = torch.randn(8, 3, 16, 16) * 0.5
    rec_c, rec_cl = AxiomStepRecord(), AxiomStepRecord()
    rec_c.observe_margin_and_scale(z1[:8], z0[:8], img_r, img_f)
    rec_cl.observe_margin_and_scale(
        z1[:8],
        z0[:8],
        img_r.to(memory_format=torch.channels_last),
        img_f.to(memory_format=torch.channels_last),
    )
    assert math.isclose(rec_c.ema_data_scale, rec_cl.ema_data_scale, rel_tol=1e-5)
    assert math.isclose(rec_c.ema_push_res, rec_cl.ema_push_res, rel_tol=1e-4)

    # 1-step transient 1e4x spike stays <= 1.04x; 2nd consecutive batch confirms shift and recovers in <= 5 steps.
    rec = AxiomStepRecord()
    rec.observe_margin_and_scale(z1, z0, xr0, xf0)
    s0 = rec.ema_data_scale
    rec.observe_margin_and_scale(z1, z0, xr0 * 1e4, xf0 * 1e4)
    assert rec.ema_data_scale <= s0 * 1.04
    rec.observe_margin_and_scale(z1, z0, xr0 * 1e4, xf0 * 1e4)
    assert rec.ema_data_scale > s0 * 500.0
    for _ in range(3):
        rec.observe_margin_and_scale(z1, z0, xr0 * 1e4, xf0 * 1e4)
    assert math.isclose(rec.ema_data_scale, s0 * 1e4, rel_tol=0.10)


def test_observe_thermal_cooling_locking_and_shock_reheat() -> None:
    rec = AxiomStepRecord()
    for step in range(80):
        sign = 0.01 if step % 2 == 0 else -0.01
        s_th = rec.observe_thermal(sign, 0.0, sign, sign, sign, r_tail=0.0, tau=0.084)
    assert rec.thermal_initialized is True
    assert rec.thermal_locked is True
    assert s_th <= 1e-4

    # Large distribution shock unlocks and reheats immediately.
    s_reheat = rec.observe_thermal(0.8, 0.5, 0.2, 0.2, 0.2, r_tail=0.15, tau=0.084)
    assert rec.thermal_locked is False
    assert s_reheat > 0.5


# ---------------------------------------------------------------------------
# 3. Weak Registries: Zero Monkey-Patching, Multi-Critic Isolation & GC Safety
# ---------------------------------------------------------------------------


def test_weak_module_and_tensor_registries_multi_critic_and_gc_cleanup() -> None:
    # Multi-critic (MisGAN-style) isolation with ZERO monkey-patched module attributes.
    critic_data = nn.Linear(4, 1)
    critic_mask = nn.Linear(4, 1)
    gen = nn.Linear(2, 4)
    rec_data = AxiomStepRecord()
    rec_mask = AxiomStepRecord()

    attrs_before_d = set(critic_data.__dict__.keys())
    attrs_before_g = set(gen.__dict__.keys())

    _attach_axiom_critic_hook(critic_data, rec_data)
    _attach_axiom_critic_hook(critic_mask, rec_mask)
    _attach_axiom_generator_hook(gen, rec_data)

    # Zero new monkey-patched attributes on user modules!
    assert set(critic_data.__dict__.keys()) == attrs_before_d
    assert set(gen.__dict__.keys()) == attrs_before_g
    assert not hasattr(critic_data, "_axiom_record")
    assert not hasattr(critic_data, "_axiom_hook_registered")
    assert not hasattr(critic_data, "_axiom_has_bn")
    assert not hasattr(gen, "_axiom_gen_hook_registered")

    assert get_module_record(critic_data) is rec_data
    assert get_module_record(critic_mask) is rec_mask
    assert get_module_record(gen) is rec_data

    # Deepcopying a hooked module (e.g. UpdatePolicy ema_G) does NOT inherit registry binding.
    ema_gen = deepcopy(gen)
    assert get_module_record(ema_gen) is None
    _ = ema_gen(torch.randn(8, 2))
    assert rec_data._g_probe_cache is None

    # WeakTensorMap handles multi-element and equal-valued scalar tensors safely (no Tensor.__eq__ crash).
    tmap = WeakTensorMap()
    p1 = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    p2 = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    s1 = torch.tensor(1.0)
    s2 = torch.tensor(1.0)
    tmap.set(p1, "p1")
    tmap.set(p2, "p2")
    tmap.set(s1, "s1")
    tmap.set(s2, "s2")
    assert tmap.get(p1) == "p1" and tmap.get(p2) == "p2"
    assert tmap.get(s1) == "s1" and tmap.get(s2) == "s2"
    assert len(tmap) == 4
    del p2, s2
    gc.collect()
    assert len(tmap) == 2 and tmap.get(p1) == "p1" and tmap.get(s1) == "s1"

    # Prior active rows & tensor record recovery via grad_fn without monkey-patching prior.z.
    table = nn.Parameter(torch.randn(8, 2))
    rows = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    set_active_prior_rows(table, rows)
    assert not hasattr(table, "_axiom_active_rows")
    assert torch.equal(get_active_prior_rows(table), rows)

    x_fake = gen(table[rows])
    rec_word = AxiomStepRecord()
    found_tab, found_rows = recover_active_prior_rows(x_fake, rec_word)
    assert found_tab is table and torch.equal(found_rows, rows)
    assert get_tensor_record(table) is rec_word
    assert not hasattr(table, "_axiom_record")
    clear_active_prior_rows(table)
    assert get_active_prior_rows(table) is None

    # GC cleanup of both modules and tensors.
    ref_c = weakref.ref(critic_mask)
    ref_t = weakref.ref(table)
    del critic_mask, table, x_fake, found_tab
    gc.collect()
    assert ref_c() is None and ref_t() is None


def test_bind_axiom_record_and_bind_axiom_optimizers_semantics() -> None:
    class _DummyOpt:
        def __init__(self, *, critic=None) -> None:
            self.param_groups = []
            self.critic = critic
            self.record = AxiomStepRecord() if critic is not None else None
            self._axiom_record = None
            self.layouts_bound = []
            self.probe_params = []

        def bind_module_layouts(self, mods) -> None:
            self.layouts_bound.extend(m for m in mods if m is not None)

        def set_generator_probe_parameters(self, params) -> None:
            self.probe_params = list(params)

    class _DummyPrior(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.z = nn.Parameter(torch.randn(6, 2))

    gen = nn.Linear(2, 4)
    crit = nn.Linear(4, 1)
    prior = _DummyPrior()
    opt_g = _DummyOpt(critic=None)
    opt_d = _DummyOpt(critic=crit)

    rec = bind_axiom_optimizers(
        opt_g,
        opt_d,
        generator=gen,
        critic=crit,
        prior=prior,
        base_prior=prior,
        enable_secant_probe=True,
    )
    assert rec is opt_d.record
    # opt_g.record stays None so opt_g.step() never double-counts observed_steps!
    assert opt_g.record is None and opt_g._axiom_record is rec
    assert get_axiom_record(opt_g) is rec
    assert get_axiom_record(crit) is rec
    assert get_axiom_record(gen) is rec
    assert get_axiom_record(prior) is rec
    assert get_axiom_record(prior.z) is rec
    assert gen in opt_g.layouts_bound and crit in opt_d.layouts_bound
    assert len(opt_g.probe_params) == 2
    assert not hasattr(prior, "_axiom_record") and not hasattr(prior.z, "_axiom_record")

    # Binding directly to a tensor registers in WeakTensorMap without mutating tensor.
    t_extra = torch.randn(3, 2)
    bind_axiom_record(rec, t_extra)
    assert get_tensor_record(t_extra) is rec and not hasattr(t_extra, "_axiom_record")
    t_direct = torch.randn(2, 2)
    set_tensor_record(t_direct, rec)
    assert get_tensor_record(t_direct) is rec and not hasattr(t_direct, "_axiom_record")


# ---------------------------------------------------------------------------
# 4. Single CaptureEntry, .detach() Resolution, float_output & Memory Bounds
# ---------------------------------------------------------------------------


def test_detached_logit_capture_resolution_float_output_and_memory_bounds() -> None:
    torch.manual_seed(0)

    class _CondCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(6, 1)

        def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat((x, xt), dim=-1)) + 0.01 * t.float().unsqueeze(-1)

    critic = _CondCritic()
    rec = AxiomStepRecord()
    _attach_axiom_critic_hook(critic, rec)

    xr = torch.randn(8, 4, requires_grad=True)
    xt = torch.randn(8, 2)
    t_idx = torch.randint(0, 4, (8,))
    raw = critic(xr, xt=xt, t=t_idx)

    # Single frozen _axiom_capture attribute only (no legacy 6 attrs).
    entry = getattr(raw, "_axiom_capture", None)
    assert isinstance(entry, CaptureEntry)
    assert not hasattr(raw, "_axiom_input")
    assert not hasattr(raw, "_axiom_parts")
    assert entry.record is rec and entry.has_cont_cond is True
    assert entry.parts is not None and len(entry.parts) == 2
    assert torch.equal(entry.parts[0], xr) and torch.equal(entry.parts[1], xt)
    assert entry.cond is not None and torch.equal(entry.cond.reshape(-1), t_idx.to(torch.int64))

    # Frozen immutability check.
    with pytest.raises(AttributeError):
        entry.has_cont_cond = False  # type: ignore[misc]

    # float_output preserves single _axiom_capture on BF16 -> FP32 conversion.
    bf16_logits = raw.to(torch.bfloat16)
    bf16_logits._axiom_capture = entry  # type: ignore[attr-defined]
    fp32_logits = float_output(bf16_logits)
    assert fp32_logits.dtype == torch.float32
    assert getattr(fp32_logits, "_axiom_capture", None) is entry
    assert not hasattr(fp32_logits, "_axiom_input")

    # DDGAN .detach() and .detach().reshape(B) resolve via bounded CaptureRegistry.
    det = raw.detach()
    det_view = raw.detach().reshape(8)
    assert getattr(det, "_axiom_capture", None) is None
    for probe in (det, det_view):
        resolved = lookup_axiom_capture(probe, rec)
        assert resolved is not None
        assert resolved.record is rec and resolved.has_cont_cond is True
        assert resolved.parts is not None and len(resolved.parts) == 2
        assert not resolved.x_in.requires_grad and all(not p.requires_grad for p in resolved.parts)
        assert torch.allclose(resolved.parts[0], xr.detach()) and torch.allclose(resolved.parts[1], xt)

    # override_axiom_capture_input replaces noisy x with clean x on both attr and ring.
    x_clean = torch.zeros_like(xr)
    updated = override_axiom_capture_input(raw, x_clean, critic=critic)
    assert updated is not None and updated.parts is not None
    assert torch.equal(updated.parts[0], x_clean)
    assert torch.equal(lookup_axiom_capture(raw.detach(), rec).parts[0], x_clean.detach())

    # Bounded LRU cap & step cleanup.
    for _ in range(24):
        _ = critic(torch.randn(8, 4), xt=xt, t=t_idx)
    assert rec.captures.ring_size <= 16
    rec.record_step()
    assert rec.captures.ring_size == 0 and rec.captures.ring_bytes == 0 and rec.captures.live_count == 0

    # High-res VRAM guard (> 16 KiB / sample never cached in detached ring).
    big_critic = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 1))
    big_rec = AxiomStepRecord()
    _attach_axiom_critic_hook(big_critic, big_rec)
    big_out = big_critic(torch.randn(2, 3, 64, 64)).detach()
    assert big_rec.captures.ring_size == 0 and lookup_axiom_capture(big_out, big_rec) is None

    # Explicit total-byte eviction check on CaptureRegistry.
    small_reg = CaptureRegistry(max_ring=16, max_sample_bytes=1024, max_total_bytes=64)
    for _ in range(5):
        lg = torch.randn(2, 1)
        small_reg.put_detached(lg, CaptureEntry(x_in=torch.randn(2, 4)))  # 32 bytes each
    assert small_reg.ring_bytes <= 64 and small_reg.ring_size <= 2


# ---------------------------------------------------------------------------
# 5. axiom_source_context Nesting & Live Endpoint Reuse
# ---------------------------------------------------------------------------


def test_axiom_source_context_nesting_and_live_endpoint_reuse() -> None:
    torch.manual_seed(0)
    critic = nn.Linear(4, 1)
    rec = AxiomStepRecord()
    _attach_axiom_critic_hook(critic, rec)

    class _Wrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.critic = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.critic(x)

    wrapped = _Wrapper(critic)
    src_outer = torch.randn(4, 4)
    src_inner = torch.randn(4, 4)
    noisy_inner = src_inner + 0.01 * torch.randn_like(src_inner)
    noisy_outer = src_outer + 0.01 * torch.randn_like(src_outer)

    with axiom_source_context(wrapped, src_outer):
        with axiom_source_context(critic, src_inner):
            out_inner = critic(noisy_inner)
        # Outer context must be restored cleanly after inner exits!
        out_outer = critic(noisy_outer)

    # Registry entry must be cleaned up after outermost exit.
    assert critic not in axiom_state._MODULE_PENDING_SOURCE
    assert not hasattr(critic, "_axiom_pending_source_key")

    live_in = rec.find_live_endpoint(src_inner)
    live_out = rec.find_live_endpoint(src_outer)
    assert live_in is out_inner and live_out is out_outer

    # Backward invalidates live endpoint automatically via tensor hook.
    out_inner.sum().backward()
    assert rec.find_live_endpoint(src_inner) is None
    assert rec.find_live_endpoint(src_outer) is out_outer


# ---------------------------------------------------------------------------
# 6. eval_probe_forward Bit-Exact Zero Global RNG Consumption & BN Preservation
# ---------------------------------------------------------------------------


def test_eval_probe_forward_zero_rng_mutation_and_bn_stat_preservation() -> None:
    class _StochasticBNNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(4)
            self.drop = nn.Dropout(p=0.5)
            self.fc = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Deliberately consume ALL global RNG streams inside forward!
            _ = random.random()
            _ = np.random.rand(3)
            noise = torch.randn_like(x) * 0.01
            return self.fc(self.drop(self.bn(x + noise)))

    torch.manual_seed(123)
    random.seed(456)
    np.random.seed(789)

    net = _StochasticBNNet()
    net.train()
    # Warm up BN stats once so running_mean/running_var are non-trivial.
    _ = net(torch.randn(16, 4) * 3.0 + 2.0)

    rm_before = net.bn.running_mean.clone()
    rv_before = net.bn.running_var.clone()
    nb_before = net.bn.num_batches_tracked.clone()

    cpu_state_before = torch.get_rng_state().clone()
    py_state_before = random.getstate()
    np_state_before = deepcopy(np.random.get_state())

    rec = AxiomStepRecord()
    x_probe = torch.randn(8, 4)
    # Re-snapshot RNG states right after constructing x_probe!
    cpu_state_before = torch.get_rng_state().clone()
    py_state_before = random.getstate()
    np_state_before = deepcopy(np.random.get_state())

    y1 = eval_probe_forward(net, (x_probe,), {}, rec)
    assert y1 is not None and y1.shape == (8, 2) and y1.dtype == torch.float32
    assert rec._in_probe is False
    assert net.training is True and net.bn.training is True and net.drop.training is True

    # BN running stats must be bit-identically untouched!
    assert torch.equal(net.bn.running_mean, rm_before)
    assert torch.equal(net.bn.running_var, rv_before)
    assert torch.equal(net.bn.num_batches_tracked, nb_before)

    # All global RNG streams must be bit-identically untouched!
    assert torch.equal(torch.get_rng_state(), cpu_state_before)
    assert random.getstate() == py_state_before
    np_state_after = np.random.get_state()
    assert np_state_after[0] == np_state_before[0]
    assert np.array_equal(np_state_after[1], np_state_before[1])
    assert np_state_after[2:] == np_state_before[2:]

    # Calling again produces bit-identical output because forked RNG started from identical state.
    y2 = eval_probe_forward(net, (x_probe,), {}, rec)
    assert y2 is not None and torch.equal(y1, y2)


def test_generator_hook_stratified_probe_and_element_budget() -> None:
    torch.manual_seed(0)
    gen = nn.Linear(4, 2)
    rec = AxiomStepRecord()
    _attach_axiom_generator_hook(gen, rec)

    z = torch.randn(64, 4)
    z[63] = 50.0  # rare tail outlier
    _ = gen(z)
    assert rec._g_probe_cache is not None
    mod, probe_args, _, y_pre = rec._g_probe_cache
    assert mod is gen and y_pre.shape == (32, 2)
    assert float(probe_args[0].abs().amax().item()) == 50.0

    # Large image generator (e.g., 3x64x64) is skipped by element budget to protect VRAM/speed.
    big_gen = nn.Linear(4, 3 * 64 * 64)
    rec_big = AxiomStepRecord()
    _attach_axiom_generator_hook(big_gen, rec_big)
    _ = big_gen(torch.randn(8, 4))
    assert rec_big._g_probe_cache is None


@pytest.mark.cuda
def test_eval_probe_forward_cuda_rng_and_packed_ema_sync() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device("cuda:0")
    net = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2)).to(dev)
    net.train()
    x = torch.randn(8, 4, device=dev)
    _ = net(x)
    cuda_before = torch.cuda.get_rng_state(dev).clone()
    cpu_before = torch.get_rng_state().clone()
    rec = AxiomStepRecord()
    y = eval_probe_forward(net, (x,), {}, rec)
    assert y is not None and y.device == dev
    assert torch.equal(torch.cuda.get_rng_state(dev), cuda_before)
    assert torch.equal(torch.get_rng_state(), cpu_before)

    # Packed EMA GPU shadow flush roundtrips cleanly.
    rec.observe_grad_sq_tensor(torch.tensor(1.75, device=dev))
    rec.observe_margin_and_scale(torch.ones(8, 1, device=dev), -torch.ones(8, 1, device=dev), x, x * 0.5)
    sd = rec.state_dict()
    assert sd["schema"] == 2 and math.isclose(sd["ema_grad_sq"], 1.75)
