"""CondFiber critic step records, forward hook binding, and formulation Adam (composing on Unified)."""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict

import torch
from torch import nn

from .._unified.critic import (
    _UnifiedRecordMixin,
    _attach_unified_critic_hook,
    _validate_unified_packet,
    bind_unified_record,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam
from .bank import CondFiberBank, validate_bank_packet

_COND_FIBER_PACKET_KEYS = frozenset({"unified", "cond_fiber_bank"})


def _validate_cond_fiber_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _COND_FIBER_PACKET_KEYS:
        raise ValueError("invalid cond_fiber record packet fields")
    _validate_unified_packet(packet["unified"])
    validate_bank_packet(packet["cond_fiber_bank"])
    return packet


class _CondFiberRecordMixin(_UnifiedRecordMixin):
    """Unified critic record mixin extended with a checkpointed ``CondFiberBank``."""

    def _init_cond_fiber_fields(self) -> None:
        self._init_unified_fields()
        self.cond_fiber_bank = CondFiberBank()

    def _cond_fiber_packet(self) -> Dict[str, Any]:
        return {
            "unified": self._unified_packet(),
            "cond_fiber_bank": self.cond_fiber_bank.to_packet(),
        }

    def _restore_cond_fiber_packet(self, packet: Dict[str, Any]) -> None:
        _validate_cond_fiber_packet(packet)
        self._restore_unified_packet(packet["unified"])
        self.cond_fiber_bank.load_packet(packet["cond_fiber_bank"])

    def _copy_cond_fiber_to(self, dup: Any) -> None:
        self._copy_unified_to(dup)
        dup.cond_fiber_bank = self.cond_fiber_bank.copy()


class _CondFiberObserverStepRecord(_CondFiberRecordMixin, CriticStepRecord):
    """Observer record for DualNorm and plain Adam critics under ``critic_formulation='cond_fiber'``."""

    def __init__(self) -> None:
        super().__init__()
        self._init_cond_fiber_fields()

    def record_step(self, optimizer_or_lr: Any) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "cond_fiber": self._cond_fiber_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "cond_fiber" not in state:
            raise ValueError("cond_fiber observer checkpoint is missing 'cond_fiber' state")
        if set(state) != {*CriticStepRecord.KEYS, "cond_fiber"}:
            raise ValueError("invalid cond_fiber observer record fields")
        _validate_cond_fiber_packet(state["cond_fiber"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_cond_fiber_packet(state["cond_fiber"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_CondFiberObserverStepRecord":
        dup = _CondFiberObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_cond_fiber_to(dup)
        return dup


class CondFiberStepRecord(_CondFiberRecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``CondFiberCriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "cond_fiber")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_cond_fiber_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "cond_fiber", "cond_fiber": self._cond_fiber_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid CondFiberStepRecord checkpoint keys")
        if state.get("formulation") != "cond_fiber":
            raise ValueError("checkpoint formulation is not 'cond_fiber'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid CondFiberStepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("CondFiberStepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            val = state[key]
            if key == "lr_last" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid CondFiberStepRecord {key}")
        _validate_cond_fiber_packet(state["cond_fiber"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_cond_fiber_packet(state["cond_fiber"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "CondFiberStepRecord":
        dup = CondFiberStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_cond_fiber_to(dup)
        return dup


def _attach_cond_fiber_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local unified+cond_fiber forward hook to ``critic``."""
    _attach_unified_critic_hook(critic, record)
    if isinstance(critic, nn.Module):
        critic._cond_fiber_record = record


def bind_cond_fiber_record(record: Any, *consumers: Any) -> None:
    """Bind instance-local ``cond_fiber`` critic record to losses, optimizers, and priors."""
    bind_unified_record(record, *consumers)
    for consumer in consumers:
        if consumer is None:
            continue
        if hasattr(consumer, "cond_fiber_record"):
            consumer.cond_fiber_record = record


class CondFiberCriticAdam(K3PCriticAdam):
    """Formulation critic Adam for ``critic_formulation='cond_fiber'`` (zero EMA critic copy)."""

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
        self.record = CondFiberStepRecord()
        self._t1c_record = self.record
        self._t2_record = self.record
        self._unified_record = self.record
        self._cond_fiber_record = self.record
        _attach_cond_fiber_critic_hook(critic, self.record)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "CondFiberCriticAdam",
    "CondFiberStepRecord",
    "_CondFiberObserverStepRecord",
    "_CondFiberRecordMixin",
    "_attach_cond_fiber_critic_hook",
    "_validate_cond_fiber_packet",
    "bind_cond_fiber_record",
]
