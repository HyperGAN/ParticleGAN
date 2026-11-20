"""Unified Synthesis critic step records, live 1st-order endpoint + chart capture hook, and formulation Adam.

Combines:
- Track 2 (`t2_secant`) live unscaled 1st-order endpoint capture (`_live_captures`) so
  convolutional image critics reuse `d_loss` forward passes at zero extra endpoint cost.
- Track 1B (`t1_coupling`) persistent streaming chart atlas (`chart_sum`, `chart_count`,
  `chart_sq_dev`, `chart_fake_mass`), 6-channel split two-sample thermal state, and
  ring-buffered `(x, cond_disc, float_parts)` capture (supporting `.detach()` in component
  hosts such as `DDGANFixture`).
- Track 3 (`rgdt`) pooled-midpoint conditional fiber step-damping state (`_step_scale`).
"""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .._bispf.thermal import _LEGACY_THERMAL_CHANNEL
from .._t1_coupling.critic import (
    _T1C_PACKET_KEYS,
    _T1CouplingRecordMixin,
    _extract_t1c_inputs_parts,
    _lookup_t1c_capture_parts,
    _record_t1c_capture,
    _validate_t1c_packet,
)
from .._t2_critic.critic import (
    _MAX_LIVE_CAPTURES,
    _cond_signature,
    _tensor_identity_key,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

_UNIFIED_PACKET_KEYS = frozenset({*_T1C_PACKET_KEYS, "step_scale"})


def _validate_unified_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _UNIFIED_PACKET_KEYS:
        raise ValueError("invalid unified record packet fields")
    step_scale = packet["step_scale"]
    if (
        isinstance(step_scale, bool)
        or not isinstance(step_scale, (int, float))
        or not math.isfinite(step_scale)
        or float(step_scale) <= 0.0
        or float(step_scale) > 1.0
    ):
        raise ValueError("invalid unified step_scale outside (0, 1]")
    t1c_sub = {k: v for k, v in packet.items() if k != "step_scale"}
    _validate_t1c_packet(t1c_sub)
    return packet


class _UnifiedRecordMixin(_T1CouplingRecordMixin):
    """Unified critic record mixin combining Track 1B chart/thermal state, Track 2 live captures, and Track 3 step scale."""

    def _init_unified_fields(self) -> None:
        self._init_t1c_fields()
        self._step_scale: float = 1.0
        self._t1c_disc_gate: float = 0.0
        self._live_captures: List[List[Any]] = []
        self._last_probe_actual_input: Optional[torch.Tensor] = None

    @property
    def rgdt_cond_initialized(self) -> bool:
        return self.t1c_cond_initialized

    @rgdt_cond_initialized.setter
    def rgdt_cond_initialized(self, val: bool) -> None:
        self.t1c_cond_initialized = bool(val)

    @property
    def rgdt_cond_ema(self) -> float:
        return self.t1c_cond_ema

    @rgdt_cond_ema.setter
    def rgdt_cond_ema(self, val: float) -> None:
        self.t1c_cond_ema = float(val)

    def clear_transient_captures(self) -> None:
        if hasattr(self, "_live_captures") and self._live_captures:
            self._live_captures.clear()
        self._last_probe_actual_input = None

    def _unified_packet(self) -> Dict[str, Any]:
        return {
            **self._t1c_packet(),
            "step_scale": float(self._step_scale),
        }

    def _restore_unified_packet(self, packet: Dict[str, Any]) -> None:
        _validate_unified_packet(packet)
        t1c_sub = {k: v for k, v in packet.items() if k != "step_scale"}
        self._restore_t1c_packet(t1c_sub)
        self._step_scale = float(packet["step_scale"])
        self._t1c_disc_gate = 0.0
        self._live_captures = []
        self._last_probe_actual_input = None

    def _copy_unified_to(self, dup: Any) -> None:
        self._copy_t1c_to(dup)
        dup._step_scale = self._step_scale
        dup._t1c_disc_gate = 0.0
        dup._live_captures = []
        dup._last_probe_actual_input = None


class _UnifiedObserverStepRecord(_UnifiedRecordMixin, CriticStepRecord):
    """Observer record for DualNorm and plain Adam critics under ``critic_formulation='unified'``."""

    def __init__(self) -> None:
        super().__init__()
        self._init_unified_fields()

    def record_step(self, optimizer_or_lr: Any) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "unified": self._unified_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "unified" not in state:
            raise ValueError("unified observer checkpoint is missing 'unified' state")
        if set(state) != {*CriticStepRecord.KEYS, "unified"}:
            raise ValueError("invalid unified observer record fields")
        _validate_unified_packet(state["unified"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_unified_packet(state["unified"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_UnifiedObserverStepRecord":
        dup = _UnifiedObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_to(dup)
        return dup


class UnifiedStepRecord(_UnifiedRecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``UnifiedCriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "unified")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_unified_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "unified", "unified": self._unified_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid UnifiedStepRecord checkpoint keys")
        if state.get("formulation") != "unified":
            raise ValueError("checkpoint formulation is not 'unified'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid UnifiedStepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("UnifiedStepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            val = state[key]
            if key == "lr_last" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid UnifiedStepRecord {key}")
        _validate_unified_packet(state["unified"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_unified_packet(state["unified"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "UnifiedStepRecord":
        dup = UnifiedStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_to(dup)
        return dup


def _unified_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Single instance-bound forward hook capturing unscaled 1st-order endpoints, chart inputs, and thermal cooling."""
    rec = getattr(module, "_unified_record", getattr(module, "_t1c_record", None))
    x_first = args[0] if (args and isinstance(args[0], torch.Tensor)) else None

    if rec is not None and getattr(rec, "_in_penalty", False):
        if x_first is not None:
            rec._last_probe_actual_input = x_first.detach()
        return None

    raw_target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(raw_target, torch.Tensor) or raw_target.ndim < 1:
        return None

    # 1. Capture unscaled 1st-order live endpoint logits for Track 2 zero-extra-forward image secant reuse:
    if (
        rec is not None
        and x_first is not None
        and module.training
        and torch.is_grad_enabled()
        and raw_target.requires_grad
        and not x_first.requires_grad
        and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in args[1:])
        and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in kwargs.values())
    ):
        src_key = getattr(module, "_t2_pending_source_key", None) or _tensor_identity_key(x_first)
        if src_key is not None:
            unscaled = raw_target.float() if raw_target.dtype == torch.bfloat16 else raw_target
            x_act = x_first.detach()
            captures = getattr(rec, "_live_captures", None)
            if captures is None:
                captures = []
                rec._live_captures = captures
            c_sig = _cond_signature(args, kwargs)
            step_tag = int(getattr(rec, "observed_steps", 0))
            entry: List[Any] = [step_tag, src_key, c_sig, x_act, unscaled, True]
            unscaled.register_hook(lambda _g, _e=entry: _e.__setitem__(5, False))
            captures.append(entry)
            if len(captures) > _MAX_LIVE_CAPTURES:
                del captures[: len(captures) - _MAX_LIVE_CAPTURES]

    # 2. Apply thermal cooling to outgoing critic logits (outside penalty probes):
    modified = False
    target = raw_target
    if rec is not None and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and target.requires_grad:
            scaled = target * s + target.detach() * (1.0 - s)
            scaled._t1c_thermally_scaled = True
            if isinstance(output, torch.Tensor):
                output = scaled
                modified = True
            elif isinstance(output, tuple):
                output = (scaled, *output[1:])
                modified = True
            elif isinstance(output, list):
                output = [scaled, *output[1:]]
                modified = True
            target = scaled

    # 3. Record (x_in, cond_disc, has_cont_cond, float_parts) for UnifiedGANLoss / T1CouplingLoss
    #    (skip unconditional high-resolution 4D image tensors where UnifiedGANLoss uses raw logits directly):
    if (
        x_first is not None
        and x_first.ndim >= 4
        and x_first.shape[0] > 0
        and x_first[0].numel() > 1024
        and len(args) == 1
        and not kwargs
    ):
        return output if modified else None

    x_in, cond_disc, has_cont_cond, float_parts = _extract_t1c_inputs_parts(
        args, kwargs, int(target.shape[0])
    )
    if x_in is not None:
        _record_t1c_capture(
            output,
            x_in,
            cond_disc,
            rec,
            has_cont_cond=has_cont_cond,
            float_parts=float_parts,
            overwrite=True,
        )
    return output if modified else None


def _attach_unified_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local unified forward hook to ``critic`` (zero global state)."""
    if not hasattr(record, "_in_penalty"):
        record._in_penalty = False
    if not hasattr(record, "_in_protected_probe"):
        record._in_protected_probe = False
    if not hasattr(record, "_capture_ring"):
        record._capture_ring = OrderedDict()
    if not hasattr(record, "_live_captures"):
        record._live_captures = []
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL.clear()
    if not isinstance(critic, nn.Module):
        return
    critic._unified_record = record
    critic._t1c_record = record
    critic._t2_record = record
    if getattr(critic, "_unified_hook_registered", False):
        return
    critic.register_forward_hook(_unified_critic_forward_hook, with_kwargs=True)
    critic._unified_hook_registered = True


def bind_unified_record(record: Any, *consumers: Any) -> None:
    """Bind instance-local ``unified`` critic record to losses, optimizers, and priors."""
    if record is not None:
        record.legacy_thermal_channel = False
    for consumer in consumers:
        if consumer is None:
            continue
        if hasattr(consumer, "record"):
            consumer.record = record
        if hasattr(consumer, "thermal_source"):
            consumer.thermal_source = record
        if hasattr(consumer, "empf_record"):
            consumer.empf_record = record
        if hasattr(consumer, "unified_record"):
            consumer.unified_record = record


class UnifiedCriticAdam(K3PCriticAdam):
    """Formulation critic Adam for ``critic_formulation='unified'`` (zero EMA critic copy)."""

    def __init__(
        self,
        params: Any,
        *,
        critic: nn.Module | None = None,
        ema_critic: nn.Module | None = None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs: Any,
    ) -> None:
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = UnifiedStepRecord()
        self._t1c_record = self.record
        self._t2_record = self.record
        self._unified_record = self.record
        _attach_unified_critic_hook(critic, self.record)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "UnifiedCriticAdam",
    "UnifiedStepRecord",
    "_UnifiedObserverStepRecord",
    "_attach_unified_critic_hook",
    "_cond_signature",
    "_extract_t1c_inputs_parts",
    "_lookup_t1c_capture_parts",
    "_record_t1c_capture",
    "_tensor_identity_key",
    "_validate_unified_packet",
    "bind_unified_record",
]
