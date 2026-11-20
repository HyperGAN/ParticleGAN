"""Track 2 (Theory 2) critic step records, live 1st-order capture hook, and formulation Adam."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .._empf.critic import (
    FROZEN_THERMAL_SCALE,
    compute_empf_step_scale,
    compute_thermal_scale,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

_MAX_LIVE_CAPTURES = 6


def _clear_live_capture_entries(entries: List[Any]) -> None:
    """Release tensor references and backward-alive flags from live-capture entries."""
    for entry in entries:
        if isinstance(entry, list) and len(entry) >= 5:
            entry[3] = None
            entry[4] = None
            if len(entry) >= 6 and isinstance(entry[5], list):
                entry[5].clear()
    entries.clear()


def _trim_live_captures(captures: List[Any], max_entries: int = _MAX_LIVE_CAPTURES) -> None:
    """Evict oldest live-capture entries while explicitly dropping their tensor references."""
    if len(captures) > max_entries:
        evict_n = len(captures) - max_entries
        _clear_live_capture_entries(captures[:evict_n])
        del captures[:evict_n]

_T2_STATE_KEYS = frozenset({
    "schema",
    "initialized",
    "ema_residual",
    "ema_data_scale",
    "ema_cond_energy",
    "thermal_scale",
    "step_scale",
    "chord_phase",
    "revival_phase",
    "revivals",
    "proto_initialized",
    "proto_sum",
    "proto_count",
})


def _tensor_identity_key(tensor: torch.Tensor) -> Optional[Tuple[int, int, Tuple[int, ...], str]]:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.numel() == 0:
        return None
    try:
        ptr = int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return None
    if ptr == 0:
        return None
    return (ptr, int(tensor.storage_offset()), tuple(int(s) for s in tensor.shape), str(tensor.device))


def _cond_signature(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    sig: List[Any] = []
    for item in args[1:]:
        if isinstance(item, torch.Tensor):
            sig.append(("T", _tensor_identity_key(item), str(item.dtype)))
        else:
            sig.append(("V", type(item).__qualname__, repr(item)))
    if kwargs:
        for key in sorted(kwargs):
            val = kwargs[key]
            if isinstance(val, torch.Tensor):
                sig.append(("KT", key, _tensor_identity_key(val), str(val.dtype)))
            else:
                sig.append(("KV", key, type(val).__qualname__, repr(val)))
    return tuple(sig)


def bind_t2_record(record: Any, *readers: Any) -> None:
    """Bind instance-local Track 2 critic state to thermal/loss readers (zero global state)."""
    for reader in readers:
        if reader is not None:
            if hasattr(reader, "t2_record"):
                reader.t2_record = record
            if hasattr(reader, "empf_record"):
                reader.empf_record = record


def read_t2_thermal(source: Any, default: float = 1.0) -> float:
    """Read instance-bound thermal scale, returning ``default`` when unbound."""
    if source is None:
        return float(default)
    return float(getattr(source, "_thermal_scale", default))


def _tag_t2_output(
    tensor: torch.Tensor,
    x_in: torch.Tensor,
    rec: Any,
    is_cond: bool,
    cond_tensors: Tuple[torch.Tensor, ...] = (),
) -> None:
    if not hasattr(tensor, "_t2_input"):
        tensor._t2_input = x_in
    if not hasattr(tensor, "_empf_input"):
        tensor._empf_input = x_in
    if rec is not None:
        if not hasattr(tensor, "_t2_record"):
            tensor._t2_record = rec
        if not hasattr(tensor, "_empf_record"):
            tensor._empf_record = rec
    tensor._t2_conditioned = bool(is_cond)
    tensor._t2_cond_tensors = cond_tensors
    tensor._empf_conditioned = bool(is_cond)
    tensor._empf_cond_tensors = cond_tensors


def _t2_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-bound 1st-order unscaled logit capture, input tagger, and thermal cooling."""
    rec = getattr(module, "_t2_record", None)
    is_cond = bool(kwargs or len(args) > 1)
    cond_list = [a.detach() for a in args[1:] if isinstance(a, torch.Tensor)]
    if kwargs:
        for _, val in sorted(kwargs.items()):
            if isinstance(val, torch.Tensor):
                cond_list.append(val.detach())
    cond_tensors = tuple(cond_list)
    x_in = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    raw_target = output[0] if isinstance(output, (tuple, list)) and output else output

    if x_in is not None and isinstance(raw_target, torch.Tensor):
        _tag_t2_output(raw_target, x_in, rec, is_cond, cond_tensors)
        if rec is not None:
            if getattr(rec, "_in_penalty", False):
                rec._last_probe_actual_input = x_in.detach()
            elif (
                module.training
                and torch.is_grad_enabled()
                and raw_target.requires_grad
                and not x_in.requires_grad
                and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in args[1:])
                and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in kwargs.values())
            ):
                src_key = getattr(module, "_t2_pending_source_key", None) or _tensor_identity_key(x_in)
                if src_key is not None:
                    unscaled = raw_target.float() if raw_target.dtype == torch.bfloat16 else raw_target
                    x_act = x_in.detach()
                    if x_act.dtype == torch.bfloat16:
                        x_act = x_act.float()
                    captures = getattr(rec, "_live_captures", None)
                    if captures is None:
                        captures = []
                        rec._live_captures = captures
                    c_sig = _cond_signature(args, kwargs)
                    step_tag = int(getattr(rec, "observed_steps", 0))
                    alive = [True]
                    entry: List[Any] = [step_tag, src_key, c_sig, x_act, unscaled, alive]
                    raw_target.register_hook(lambda _g, _a=alive: _a.clear())
                    captures.append(entry)
                    _trim_live_captures(captures)

    if module.training and rec is not None and not getattr(rec, "_in_penalty", False) and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            if x_in is not None:
                _tag_t2_output(out_scaled, x_in, rec, is_cond, cond_tensors)
            else:
                for attr in (
                    "_t2_input",
                    "_t2_record",
                    "_t2_conditioned",
                    "_t2_cond_tensors",
                    "_empf_input",
                    "_empf_record",
                    "_empf_conditioned",
                    "_empf_cond_tensors",
                ):
                    if hasattr(output, attr):
                        setattr(out_scaled, attr, getattr(output, attr))
            return out_scaled
    return None


def _attach_t2_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local Track 2 forward hook to ``critic`` (zero global state)."""
    record._in_penalty = False
    if not isinstance(critic, nn.Module):
        return
    critic._t2_record = record
    critic._empf_record = record
    if getattr(critic, "_t2_hook_registered", False):
        return
    critic.register_forward_hook(_t2_critic_forward_hook, with_kwargs=True)
    critic._t2_hook_registered = True


def _validate_t2_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _T2_STATE_KEYS:
        raise ValueError("invalid T2Secant record packet fields")
    if type(packet["schema"]) is not int or packet["schema"] != 1:
        raise ValueError("unsupported T2Secant record schema")
    for key in ("initialized", "proto_initialized"):
        if type(packet[key]) is not bool:
            raise ValueError(f"invalid T2Secant boolean field {key}")
    if type(packet["revivals"]) is not int or packet["revivals"] < 0:
        raise ValueError("invalid T2Secant revival count")
    for key in ("ema_residual", "ema_data_scale", "thermal_scale", "step_scale", "chord_phase", "revival_phase"):
        value = packet[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid T2Secant scalar field {key}")
    cond_e = packet["ema_cond_energy"]
    if isinstance(cond_e, bool) or not isinstance(cond_e, (int, float)) or not math.isfinite(cond_e):
        raise ValueError("invalid T2Secant scalar field ema_cond_energy")
    if float(packet["thermal_scale"]) > 1.0:
        raise ValueError("invalid T2Secant thermal scale > 1")
    if float(packet["step_scale"]) <= 0.0 or float(packet["step_scale"]) > 1.0:
        raise ValueError("invalid T2Secant step scale outside (0, 1]")
    for key in ("chord_phase", "revival_phase"):
        if float(packet[key]) >= 1.0:
            raise ValueError(f"invalid T2Secant phase {key} >= 1")

    proto_init = packet["proto_initialized"]
    p_sum = packet["proto_sum"]
    p_count = packet["proto_count"]
    if not proto_init:
        if p_sum is not None or p_count is not None:
            raise ValueError("uninitialized T2Secant prototype bank must have None tensors")
    else:
        if not isinstance(p_sum, torch.Tensor) or not isinstance(p_count, torch.Tensor):
            raise ValueError("initialized T2Secant prototype bank requires proto_sum and proto_count tensors")
        if p_sum.dtype != torch.float32 or p_count.dtype != torch.float32:
            raise ValueError("T2Secant prototype bank tensors must have float32 dtype")
        if p_sum.ndim != 2 or p_count.ndim != 1:
            raise ValueError("invalid T2Secant prototype tensor rank")
        if p_sum.shape[0] == 0 or p_sum.shape[1] == 0 or p_sum.shape[0] != p_count.shape[0]:
            raise ValueError("incompatible T2Secant prototype tensor shapes")
        if not bool(torch.isfinite(p_sum).all()) or not bool(torch.isfinite(p_count).all()):
            raise ValueError("nonfinite T2Secant prototype tensor values")
        if bool((p_count <= 0).any()):
            raise ValueError("T2Secant prototype counts must be strictly positive")
    return packet


class _T2RecordMixin:
    """Shared Track 2 sufficient-statistics, thermal settling, and transient capture state."""

    def _init_t2_fields(self) -> None:
        self.initialized: bool = False
        self.ema_residual: float = 1.0
        self.ema_data_scale: float = 1.0
        self.ema_cond_energy: float = 0.50
        self._thermal_scale: float = 1.0
        self._step_scale: float = 1.0
        self.chord_phase: float = 0.0
        self.revival_phase: float = 0.0
        self.revivals: int = 0
        self.proto_initialized: bool = False
        self.proto_sum: torch.Tensor | None = None
        self.proto_count: torch.Tensor | None = None
        self._in_penalty: bool = False
        self._live_captures: List[Tuple[Any, Any, torch.Tensor, torch.Tensor]] = []
        self._last_probe_actual_input: Optional[torch.Tensor] = None

    def clear_transient_captures(self) -> None:
        if hasattr(self, "_live_captures") and self._live_captures:
            _clear_live_capture_entries(self._live_captures)
        self._last_probe_actual_input = None

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for key, val in self.__dict__.items():
            if key == "_live_captures":
                setattr(clone, key, [])
            elif key == "_last_probe_actual_input":
                setattr(clone, key, None)
            else:
                setattr(clone, key, deepcopy(val, memo))
        return clone

    def observe_transport(self, res: float, data_scale: float, ema_decay: float = 0.92) -> None:
        res_f = float(res)
        scale_f = float(data_scale)
        if not self.initialized:
            self.ema_residual = res_f
            self.ema_data_scale = scale_f
            self.initialized = True
        elif res_f > max(5.5 * self.ema_residual, self.ema_residual + 0.55):
            self.ema_residual = res_f
            self.ema_data_scale = scale_f
        else:
            self.ema_residual = ema_decay * self.ema_residual + (1.0 - ema_decay) * res_f
            self.ema_data_scale = ema_decay * self.ema_data_scale + (1.0 - ema_decay) * scale_f

    def observe_conditional_energy(self, u_signed: float, data_scale: float, ema_decay: float = 0.96) -> None:
        u_f = float(u_signed)
        scale_f = float(data_scale)
        if not self.initialized:
            self.ema_cond_energy = max(0.0, u_f)
            self.ema_data_scale = scale_f
            self.ema_residual = math.sqrt(max(0.0, self.ema_cond_energy))
            self.initialized = True
        elif u_f > max(2.5 * max(self.ema_cond_energy, 0.01), self.ema_cond_energy + 0.15):
            self.ema_cond_energy = u_f
            self.ema_data_scale = scale_f
            self.ema_residual = math.sqrt(max(0.0, self.ema_cond_energy))
        else:
            self.ema_cond_energy = ema_decay * self.ema_cond_energy + (1.0 - ema_decay) * u_f
            self.ema_data_scale = ema_decay * self.ema_data_scale + (1.0 - ema_decay) * scale_f
            self.ema_residual = math.sqrt(max(0.0, self.ema_cond_energy))

    def update_prototypes(
        self,
        proto_sum: torch.Tensor,
        proto_count: torch.Tensor,
        revival_phase: float,
        revivals: int,
    ) -> None:
        self.proto_initialized = True
        self.proto_sum = proto_sum.detach().clone().to(dtype=torch.float32)
        self.proto_count = proto_count.detach().clone().to(dtype=torch.float32)
        self.revival_phase = float(revival_phase)
        self.revivals = int(revivals)

    def _t2_packet(self) -> Dict[str, Any]:
        return {
            "schema": 1,
            "initialized": bool(self.initialized),
            "ema_residual": float(self.ema_residual),
            "ema_data_scale": float(self.ema_data_scale),
            "ema_cond_energy": float(self.ema_cond_energy),
            "thermal_scale": float(self._thermal_scale),
            "step_scale": float(self._step_scale),
            "chord_phase": float(self.chord_phase),
            "revival_phase": float(self.revival_phase),
            "revivals": int(self.revivals),
            "proto_initialized": bool(self.proto_initialized),
            "proto_sum": None if self.proto_sum is None else self.proto_sum.detach().clone(),
            "proto_count": None if self.proto_count is None else self.proto_count.detach().clone(),
        }

    def _restore_t2_packet(self, packet: Dict[str, Any]) -> None:
        _validate_t2_packet(packet)
        self.initialized = bool(packet["initialized"])
        self.ema_residual = float(packet["ema_residual"])
        self.ema_data_scale = float(packet["ema_data_scale"])
        self.ema_cond_energy = float(packet["ema_cond_energy"])
        self._thermal_scale = float(packet["thermal_scale"])
        self._step_scale = float(packet["step_scale"])
        self.chord_phase = float(packet["chord_phase"])
        self.revival_phase = float(packet["revival_phase"])
        self.revivals = int(packet["revivals"])
        self.proto_initialized = bool(packet["proto_initialized"])
        self.proto_sum = None if packet["proto_sum"] is None else packet["proto_sum"].detach().clone()
        self.proto_count = None if packet["proto_count"] is None else packet["proto_count"].detach().clone()
        self.clear_transient_captures()

    def restore_device_state(self, device: torch.device) -> None:
        if self.proto_sum is not None:
            self.proto_sum = self.proto_sum.to(device=device)
        if self.proto_count is not None:
            self.proto_count = self.proto_count.to(device=device)


class _T2SecantObserverStepRecord(_T2RecordMixin, CriticStepRecord):
    """Observer record for DualNorm and plain Adam critics under ``t2_secant``."""

    def __init__(self) -> None:
        super().__init__()
        self._init_t2_fields()

    def record_step(self, optimizer_or_lr) -> None:
        super().record_step(optimizer_or_lr)
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "t2_secant": self._t2_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "t2_secant" not in state:
            raise ValueError("T2Secant observer checkpoint is missing 't2_secant' state")
        if set(state) != {*CriticStepRecord.KEYS, "t2_secant"}:
            raise ValueError("invalid T2Secant observer record fields")
        _validate_t2_packet(state["t2_secant"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_t2_packet(state["t2_secant"])


class T2SecantStepRecord(_T2RecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``T2SecantCriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "t2_secant")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_t2_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "t2_secant", "t2_secant": self._t2_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid T2SecantStepRecord checkpoint keys")
        if state.get("formulation") != "t2_secant":
            raise ValueError("checkpoint formulation is not 't2_secant'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid T2SecantStepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("T2SecantStepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            value = state[key]
            if key == "lr_last" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid T2SecantStepRecord {key}")
        _validate_t2_packet(state["t2_secant"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_t2_packet(state["t2_secant"])


class T2SecantCriticAdam(K3PCriticAdam):
    """Formulation critic Adam for ``t2_secant`` (no EMA critic copy; carries ``T2SecantStepRecord``)."""

    def __init__(
        self,
        params,
        *,
        critic=None,
        ema_critic=None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.999,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.critic = critic
        self.ema_critic = None
        self.anchor = None
        self.record = T2SecantStepRecord()
        _attach_t2_critic_hook(critic, self.record)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "FROZEN_THERMAL_SCALE",
    "T2SecantCriticAdam",
    "T2SecantStepRecord",
    "_T2SecantObserverStepRecord",
    "_attach_t2_critic_hook",
    "_cond_signature",
    "_tensor_identity_key",
    "bind_t2_record",
    "compute_empf_step_scale",
    "compute_thermal_scale",
    "read_t2_thermal",
]
