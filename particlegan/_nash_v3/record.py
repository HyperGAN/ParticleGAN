"""Pure-elementwise T4NashV3 game state record, two-player calibration coupling, and 1st-order live capture hooks."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from .._t2_critic.critic import _cond_signature, _tensor_identity_key
from ..grad_regularizers import CriticStepRecord

FROZEN_THERMAL_SCALE = 1e-4
_PROBE_MAX_BATCH = 16
_MAX_LIVE_CAPTURES = 6

_NASH_V3_STATE_KEYS = frozenset({
    "schema",
    "initialized",
    "margin_initialized",
    "ema_residual",
    "ema_data_scale",
    "ema_cond_energy",
    "ema_margin",
    "calibration_gate",
    "d_trust_scale",
    "g_trust_scale",
    "thermal_scale",
    "step_scale",
    "chord_phase",
    "revival_phase",
    "revivals",
    "proto_initialized",
    "proto_sum",
    "proto_count",
})


def _smoothstep01(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return float(x * x * (3.0 - 2.0 * x))


def bind_nash_v3_record(record: Any, *readers: Any) -> None:
    """Bind instance-local T4NashV3 game state to loss/regularizer/optimizer readers."""
    for reader in readers:
        if reader is None:
            continue
        if hasattr(reader, "nash_v3_record"):
            reader.nash_v3_record = record
        if hasattr(reader, "t4_record"):
            reader.t4_record = record
        if hasattr(reader, "empf_record"):
            reader.empf_record = record


def read_nash_v3_thermal(source: Any, default: float = 1.0) -> float:
    """Read instance-bound T4NashV3 thermal scale, returning ``default`` when unbound."""
    if source is None:
        return float(default)
    return float(getattr(source, "_thermal_scale", default))


def compute_two_player_coupling(
    ema_margin: float,
    ema_residual: float,
    thermal_scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Clock-free two-player relative-calibration coupling ``(c_D, gamma_D, gamma_G)``."""
    res = max(0.0, float(ema_residual))
    margin = float(ema_margin)
    therm = max(0.0, min(1.0, float(thermal_scale)))

    # Only gate down adversarial pull when critic is inverted (margin < 0) or genuinely far off-manifold & hot.
    hot_gate = _smoothstep01((therm - 0.05) / 0.30)
    far_weight = max(_smoothstep01((-margin) / 0.15), _smoothstep01((res - 0.08) / 0.14) * hot_gate)
    raw_cal = _smoothstep01((margin + 0.10) / 0.42)
    c_d = float(max(0.005, (1.0 - far_weight) + far_weight * raw_cal))

    m_cap = 1.60 + 1.20 * min(res, 2.0)
    excess = max(0.0, margin - m_cap)
    gamma_d = float(max(0.12, 1.0 / (1.0 + excess / 1.50)))

    gamma_g = float(0.55 + 0.45 * c_d)
    return c_d, gamma_d, gamma_g


def _slice_probe_item(item: Any, k: int) -> Any:
    if isinstance(item, torch.Tensor):
        if item.ndim >= 1 and item.shape[0] > k:
            return item[:k].detach()
        return item.detach()
    return item


def _tag_nash_v3_output(
    tensor: torch.Tensor,
    x_in: torch.Tensor,
    rec: Any,
    is_cond: bool,
    cond_tensors: Tuple[torch.Tensor, ...] = (),
) -> None:
    if not hasattr(tensor, "_nash_v3_input"):
        tensor._nash_v3_input = x_in
    if not hasattr(tensor, "_empf_input"):
        tensor._empf_input = x_in
    if rec is not None:
        if not hasattr(tensor, "_nash_v3_record"):
            tensor._nash_v3_record = rec
        if not hasattr(tensor, "_empf_record"):
            tensor._empf_record = rec
    tensor._nash_v3_conditioned = bool(is_cond)
    tensor._nash_v3_cond_tensors = cond_tensors
    tensor._empf_conditioned = bool(is_cond)
    tensor._empf_cond_tensors = cond_tensors


def _nash_v3_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-local T4NashV3 critic hook: tags inputs, captures 1st-order live endpoints & low-D probes, cools thermally."""
    rec = getattr(module, "_nash_v3_record", None)
    if rec is not None and getattr(rec, "_in_probe", False):
        return None

    is_cond = bool(kwargs or len(args) > 1)
    cond_list = [a.detach() for a in args[1:] if isinstance(a, torch.Tensor)]
    if kwargs:
        for _, val in sorted(kwargs.items()):
            if isinstance(val, torch.Tensor):
                cond_list.append(val.detach())
    cond_tensors = tuple(cond_list)
    x_in = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    target = output[0] if isinstance(output, (tuple, list)) and output else output

    if x_in is not None and isinstance(target, torch.Tensor):
        _tag_nash_v3_output(target, x_in, rec, is_cond, cond_tensors)
        if rec is not None:
            if getattr(rec, "_in_penalty", False):
                rec._last_probe_actual_input = x_in.detach()
            elif (
                module.training
                and torch.is_grad_enabled()
                and target.requires_grad
                and not x_in.requires_grad
                and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in args[1:])
                and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in kwargs.values())
            ):
                # Live 1st-order endpoint capture (reused at 0 extra forward cost by BF16 image secant critic).
                if x_in.ndim >= 3:
                    src_key = getattr(module, "_t2_pending_source_key", None) or _tensor_identity_key(x_in)
                    if src_key is not None:
                        unscaled = target.float() if target.dtype == torch.bfloat16 else target
                        x_act = x_in.detach()
                        if x_act.dtype == torch.bfloat16:
                            x_act = x_act.float()
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

                # Low-D secant probe capture for critic inversion-deficit trust band (skipped on images for speed).
                if (
                    getattr(rec, "_enable_secant_probe", False)
                    and target.ndim <= 2
                    and x_in.ndim == 2
                    and int(x_in.shape[1]) <= 32
                ):
                    k = min(_PROBE_MAX_BATCH, int(x_in.shape[0]))
                    if k >= 2 and int(target.reshape(-1).shape[0]) == int(x_in.shape[0]):
                        p_args = tuple(_slice_probe_item(a, k) for a in args)
                        p_kwargs = {key: _slice_probe_item(val, k) for key, val in kwargs.items()}
                        p_out = target.detach().reshape(-1)[:k].to(dtype=torch.float32)
                        bucket = getattr(rec, "_d_probe_list", None)
                        if bucket is None or len(bucket) >= 2:
                            bucket = []
                            rec._d_probe_list = bucket
                        bucket.append((p_args, p_kwargs, p_out))

    if rec is not None and not getattr(rec, "_in_penalty", False) and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            if x_in is not None:
                _tag_nash_v3_output(out_scaled, x_in, rec, is_cond, cond_tensors)
            else:
                for attr in (
                    "_nash_v3_input",
                    "_nash_v3_record",
                    "_nash_v3_conditioned",
                    "_nash_v3_cond_tensors",
                    "_empf_input",
                    "_empf_record",
                    "_empf_conditioned",
                    "_empf_cond_tensors",
                ):
                    if hasattr(output, attr):
                        setattr(out_scaled, attr, getattr(output, attr))
            return out_scaled
    return None


def _nash_v3_generator_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    """Instance-local T4NashV3 generator hook capturing low-D detached probe slice for pushforward secant band."""
    rec = getattr(module, "_nash_v3_record", None)
    if rec is None or getattr(rec, "_in_probe", False) or not getattr(rec, "_enable_secant_probe", False):
        return
    if not torch.is_grad_enabled():
        return
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if (
        not isinstance(target, torch.Tensor)
        or not target.requires_grad
        or not target.is_floating_point()
        or target.ndim != 2
        or int(target.shape[1]) > 32
        or not args
        or not isinstance(args[0], torch.Tensor)
        or args[0].ndim < 2
        or args[0].shape[0] != target.shape[0]
    ):
        return
    k = min(_PROBE_MAX_BATCH, int(target.shape[0]))
    if k < 2:
        return
    # Preserve the primary (first) generator forward probe per step so coupled AE/VAE
    # reconstruction passes do not overwrite the independent-prior generation probe.
    if getattr(rec, "_g_probe_cache", None) is not None:
        return
    p_args = tuple(_slice_probe_item(a, k) for a in args)
    p_kwargs = {key: _slice_probe_item(val, k) for key, val in kwargs.items()}
    p_out = target[:k].detach().to(dtype=torch.float32)
    rec._g_probe_cache = (module, p_args, p_kwargs, p_out)


def _attach_nash_v3_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local T4NashV3 forward hook to ``critic`` (zero global state)."""
    record._in_penalty = False
    record._in_probe = False
    if not isinstance(critic, nn.Module):
        return
    critic._nash_v3_record = record
    critic._t2_record = record  # Enables InputNoise source-key tracking for zero-overhead live endpoint reuse.
    critic._empf_record = record
    if getattr(critic, "_nash_v3_hook_registered", False):
        return
    critic.register_forward_hook(_nash_v3_critic_forward_hook, with_kwargs=True)
    critic._nash_v3_hook_registered = True


def _attach_nash_v3_generator_hook(generator: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local T4NashV3 forward hook to ``generator`` (zero global state)."""
    if not isinstance(generator, nn.Module):
        return
    generator._nash_v3_record = record
    if getattr(generator, "_nash_v3_gen_hook_registered", False):
        return
    generator.register_forward_hook(_nash_v3_generator_forward_hook, with_kwargs=True)
    generator._nash_v3_gen_hook_registered = True


def _validate_nash_v3_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _NASH_V3_STATE_KEYS:
        raise ValueError("invalid T4NashV3 record packet fields")
    if type(packet["schema"]) is not int or packet["schema"] != 1:
        raise ValueError("unsupported T4NashV3 record schema")
    for key in ("initialized", "margin_initialized", "proto_initialized"):
        if type(packet[key]) is not bool:
            raise ValueError(f"invalid T4NashV3 boolean field {key}")
    if type(packet["revivals"]) is not int or packet["revivals"] < 0:
        raise ValueError("invalid T4NashV3 revival count")
    for key in (
        "ema_residual",
        "ema_data_scale",
        "calibration_gate",
        "d_trust_scale",
        "g_trust_scale",
        "thermal_scale",
        "step_scale",
        "chord_phase",
        "revival_phase",
    ):
        value = packet[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid T4NashV3 scalar field {key}")
    for signed_key in ("ema_cond_energy", "ema_margin"):
        val = packet[signed_key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"invalid T4NashV3 scalar field {signed_key}")
    for unit_key in ("calibration_gate", "d_trust_scale", "g_trust_scale", "thermal_scale", "step_scale"):
        v = float(packet[unit_key])
        if v <= 0.0 or v > 1.0:
            raise ValueError(f"invalid T4NashV3 unit scale {unit_key} outside (0, 1]")
    for phase_key in ("chord_phase", "revival_phase"):
        if float(packet[phase_key]) >= 1.0:
            raise ValueError(f"invalid T4NashV3 phase {phase_key} >= 1")

    proto_init = packet["proto_initialized"]
    p_sum = packet["proto_sum"]
    p_count = packet["proto_count"]
    if not proto_init:
        if p_sum is not None or p_count is not None:
            raise ValueError("uninitialized T4NashV3 prototype bank must have None tensors")
    else:
        if not isinstance(p_sum, torch.Tensor) or not isinstance(p_count, torch.Tensor):
            raise ValueError("initialized T4NashV3 prototype bank requires proto_sum and proto_count tensors")
        if p_sum.dtype != torch.float32 or p_count.dtype != torch.float32:
            raise ValueError("T4NashV3 prototype bank tensors must have float32 dtype")
        if p_sum.ndim != 2 or p_count.ndim != 1:
            raise ValueError("invalid T4NashV3 prototype tensor rank")
        if p_sum.shape[0] == 0 or p_sum.shape[1] == 0 or p_sum.shape[0] != p_count.shape[0]:
            raise ValueError("incompatible T4NashV3 prototype tensor shapes")
        if not bool(torch.isfinite(p_sum).all()) or not bool(torch.isfinite(p_count).all()):
            raise ValueError("nonfinite T4NashV3 prototype tensor values")
        if bool((p_count <= 0).any()):
            raise ValueError("T4NashV3 prototype counts must be strictly positive")
    return packet


class _NashV3RecordMixin:
    """Shared T4NashV3 game calibration, transport, prototype, and 1st-order capture state."""

    def _init_nash_v3_fields(self) -> None:
        self.initialized: bool = False
        self.margin_initialized: bool = False
        self.ema_residual: float = 1.0
        self.ema_data_scale: float = 1.0
        self.ema_cond_energy: float = 0.50
        self.ema_margin: float = 0.0
        self.calibration_gate: float = 1.0
        self.d_trust_scale: float = 1.0
        self.g_trust_scale: float = 1.0
        self._thermal_scale: float = 1.0
        self._step_scale: float = 1.0
        self.chord_phase: float = 0.0
        self.revival_phase: float = 0.0
        self.revivals: int = 0
        self.proto_initialized: bool = False
        self.proto_sum: torch.Tensor | None = None
        self.proto_count: torch.Tensor | None = None
        self._last_d_logits: Tuple[torch.Tensor, torch.Tensor] | None = None
        self._d_probe_list: list | None = None
        self._g_probe_cache: Tuple[Any, ...] | None = None
        self._live_captures: list = []
        self._last_probe_actual_input: torch.Tensor | None = None
        self._enable_secant_probe: bool = False
        self._in_penalty: bool = False
        self._in_probe: bool = False

    def clear_transient_captures(self) -> None:
        self._live_captures = []
        self._last_probe_actual_input = None

    def observe_transport(self, res: float, data_scale: float, ema_decay: float = 0.92) -> None:
        res_f = float(res)
        scale_f = max(1e-6, float(data_scale))
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
        scale_f = max(1e-6, float(data_scale))
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

    def observe_margin(self, margin: float, *, ema_decay: float = 0.85) -> None:
        m_f = max(-50.0, min(50.0, float(margin)))
        if not self.margin_initialized:
            self.ema_margin = m_f
            self.margin_initialized = True
        else:
            self.ema_margin = ema_decay * self.ema_margin + (1.0 - ema_decay) * m_f
        c_d, gamma_d, gamma_g = compute_two_player_coupling(self.ema_margin, self.ema_residual, self._thermal_scale)
        self.calibration_gate = c_d
        self.d_trust_scale = gamma_d
        self.g_trust_scale = gamma_g

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

    def _nash_v3_packet(self) -> Dict[str, Any]:
        return {
            "schema": 1,
            "initialized": bool(self.initialized),
            "margin_initialized": bool(self.margin_initialized),
            "ema_residual": float(self.ema_residual),
            "ema_data_scale": float(self.ema_data_scale),
            "ema_cond_energy": float(self.ema_cond_energy),
            "ema_margin": float(self.ema_margin),
            "calibration_gate": float(self.calibration_gate),
            "d_trust_scale": float(self.d_trust_scale),
            "g_trust_scale": float(self.g_trust_scale),
            "thermal_scale": float(self._thermal_scale),
            "step_scale": float(self._step_scale),
            "chord_phase": float(self.chord_phase),
            "revival_phase": float(self.revival_phase),
            "revivals": int(self.revivals),
            "proto_initialized": bool(self.proto_initialized),
            "proto_sum": None if self.proto_sum is None else self.proto_sum.detach().clone(),
            "proto_count": None if self.proto_count is None else self.proto_count.detach().clone(),
        }

    def _restore_nash_v3_packet(self, packet: Dict[str, Any]) -> None:
        self.initialized = bool(packet["initialized"])
        self.margin_initialized = bool(packet["margin_initialized"])
        self.ema_residual = float(packet["ema_residual"])
        self.ema_data_scale = float(packet["ema_data_scale"])
        self.ema_cond_energy = float(packet["ema_cond_energy"])
        self.ema_margin = float(packet["ema_margin"])
        self.calibration_gate = float(packet["calibration_gate"])
        self.d_trust_scale = float(packet["d_trust_scale"])
        self.g_trust_scale = float(packet["g_trust_scale"])
        self._thermal_scale = float(packet["thermal_scale"])
        self._step_scale = float(packet["step_scale"])
        self.chord_phase = float(packet["chord_phase"])
        self.revival_phase = float(packet["revival_phase"])
        self.revivals = int(packet["revivals"])
        self.proto_initialized = bool(packet["proto_initialized"])
        self.proto_sum = None if packet["proto_sum"] is None else packet["proto_sum"].detach().clone()
        self.proto_count = None if packet["proto_count"] is None else packet["proto_count"].detach().clone()
        self._last_d_logits = None
        self._d_probe_list = None
        self._g_probe_cache = None
        self._live_captures = []
        self._last_probe_actual_input = None

    def restore_device_state(self, device: torch.device) -> None:
        if isinstance(self.proto_sum, torch.Tensor) and self.proto_sum.device != device:
            self.proto_sum = self.proto_sum.to(device=device)
        if isinstance(self.proto_count, torch.Tensor) and self.proto_count.device != device:
            self.proto_count = self.proto_count.to(device=device)


class _NashV3ObserverStepRecord(_NashV3RecordMixin, CriticStepRecord):
    """Observer record for plain Adam critics under ``t4_nash_v3``."""

    def __init__(self) -> None:
        super().__init__()
        self._init_nash_v3_fields()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "t4_nash_v3": self._nash_v3_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or "t4_nash_v3" not in state:
            raise ValueError("T4NashV3 observer checkpoint is missing 't4_nash_v3' state")
        if set(state) != {*CriticStepRecord.KEYS, "t4_nash_v3"}:
            raise ValueError("unexpected fields in T4NashV3 observer record")
        _validate_nash_v3_packet(state["t4_nash_v3"])
        base = {key: state[key] for key in CriticStepRecord.KEYS}
        super().load_state_dict(base)
        self._restore_nash_v3_packet(state["t4_nash_v3"])


class NashV3StepRecord(_NashV3RecordMixin, CriticStepRecord):
    """Strict checkpointed step record for T4NashV3 critic optimizers."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "t4_nash_v3")

    def __init__(self) -> None:
        super().__init__()
        self._init_nash_v3_fields()

    def record_step(self, optimizer) -> None:
        super().record_step(optimizer)
        self._last_d_logits = None
        self._d_probe_list = None
        self.clear_transient_captures()

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "formulation": "t4_nash_v3",
            "t4_nash_v3": self._nash_v3_packet(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "t4_nash_v3":
            raise ValueError("checkpoint formulation is not 't4_nash_v3'")
        if set(state) != set(self.RECORD_KEYS):
            raise ValueError("unexpected or missing fields in T4NashV3 step record")
        _validate_nash_v3_packet(state["t4_nash_v3"])
        base = {key: state[key] for key in CriticStepRecord.KEYS}
        super().load_state_dict(base)
        self._restore_nash_v3_packet(state["t4_nash_v3"])


__all__ = [
    "FROZEN_THERMAL_SCALE",
    "NashV3StepRecord",
    "_NashV3ObserverStepRecord",
    "_attach_nash_v3_critic_hook",
    "_attach_nash_v3_generator_hook",
    "_smoothstep01",
    "bind_nash_v3_record",
    "compute_two_player_coupling",
    "read_nash_v3_thermal",
]
