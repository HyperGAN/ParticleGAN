"""EMPF critic step records, instance-local thermal binding, and formulation Adam."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

FROZEN_THERMAL_SCALE = 1e-4

_EMPF_STATE_KEYS = frozenset({
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


def bind_empf_record(record: Any, *readers: Any) -> None:
    """Bind instance-local EMPF critic state to thermal readers (zero global state)."""
    for reader in readers:
        if reader is not None and hasattr(reader, "empf_record"):
            reader.empf_record = record


def read_empf_thermal(source: Any, default: float = 1.0) -> float:
    """Read instance-bound EMPF thermal scale, returning ``default`` when unbound."""
    if source is None:
        return float(default)
    return float(getattr(source, "_thermal_scale", default))


def _tag_empf_output(
    tensor: torch.Tensor,
    x_in: torch.Tensor,
    rec: Any,
    is_cond: bool,
    cond_tensors: Tuple[torch.Tensor, ...] = (),
) -> None:
    if not hasattr(tensor, "_empf_input"):
        tensor._empf_input = x_in
    if rec is not None and not hasattr(tensor, "_empf_record"):
        tensor._empf_record = rec
    tensor._empf_conditioned = bool(is_cond)
    tensor._empf_cond_tensors = cond_tensors


def _empf_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-bound straight-through thermal cooling and input/conditioning tagger."""
    rec = getattr(module, "_empf_record", None)
    is_cond = bool(kwargs or len(args) > 1)
    cond_list = [a.detach() for a in args[1:] if isinstance(a, torch.Tensor)]
    if kwargs:
        for _, val in sorted(kwargs.items()):
            if isinstance(val, torch.Tensor):
                cond_list.append(val.detach())
    cond_tensors = tuple(cond_list)
    x_in = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    if x_in is not None:
        target = output[0] if isinstance(output, (tuple, list)) and output else output
        if isinstance(target, torch.Tensor):
            _tag_empf_output(target, x_in, rec, is_cond, cond_tensors)
    if rec is not None and not getattr(rec, "_in_penalty", False) and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            if x_in is not None:
                _tag_empf_output(out_scaled, x_in, rec, is_cond, cond_tensors)
            else:
                if hasattr(output, "_empf_input"):
                    out_scaled._empf_input = output._empf_input
                if hasattr(output, "_empf_record"):
                    out_scaled._empf_record = output._empf_record
                if hasattr(output, "_empf_conditioned"):
                    out_scaled._empf_conditioned = output._empf_conditioned
                if hasattr(output, "_empf_cond_tensors"):
                    out_scaled._empf_cond_tensors = output._empf_cond_tensors
            return out_scaled
    return None


def _attach_empf_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local EMPF forward hook to ``critic`` (zero global state)."""
    record._in_penalty = False
    if not isinstance(critic, nn.Module):
        return
    critic._empf_record = record
    if getattr(critic, "_empf_hook_registered", False):
        return
    critic.register_forward_hook(_empf_critic_forward_hook, with_kwargs=True)
    critic._empf_hook_registered = True


def compute_thermal_scale(
    ema_residual: float,
    batch_size: int,
    dim: int,
    *,
    tau_cooling: float = 0.020,
    s_min: float = 1e-5,
    reference_batch: int = 128,
    min_batch: int = 16,
    batch_numerator: float = 1024.0,
    max_dim: int = 4,
) -> float:
    """State-driven smoothstep thermal scale without step clocks.

    - 1D continuous clouds: floor-subtracted 1/sqrt(B) quantile-W2 cooling so
      stationary 1D Gaussians settle past finite-sample quantile noise.
    - Compressed-bank dense multi-mode clouds (B >= 1024, 2D..4D): floor-subtracted
      cooling into DualNorm's linear-smoothing polishing band once all modes populate.
    - General 2D..4D clouds (B < 1024): preserves EMPF-v2's exact smoothstep law.
    """
    b = max(int(min_batch), int(batch_size))
    d = max(1, min(int(max_dim), int(dim)))
    if d == 1:
        inv_sqrt_b = 1.0 / math.sqrt(float(b))
        r_floor = 1.60 * inv_sqrt_b
        tau = 6.20 * inv_sqrt_b
        s_floor = max(float(s_min), 0.0012)
        u = max(0.0, min(1.0, (float(ema_residual) - r_floor) / max(tau - r_floor, 1e-12)))
        smooth = u * u * (3.0 - 2.0 * u)
        return float(s_floor + (1.0 - s_floor) * smooth)
    if b >= 1024 and 2 <= d <= 4:
        r_floor = 0.0120
        tau = 0.0300
        s_floor = max(float(s_min), 0.020)
        u = max(0.0, min(1.0, (float(ema_residual) - r_floor) / max(tau - r_floor, 1e-12)))
        smooth = u * u * (3.0 - 2.0 * u)
        return float(s_floor + (1.0 - s_floor) * smooth)
    tau = float(tau_cooling)
    if b < int(reference_batch):
        tau = tau * ((float(batch_numerator) / float(b)) ** (1.0 / float(d)))
    u = max(0.0, min(1.0, float(ema_residual) / max(tau, 1e-12)))
    smooth = u * u * (3.0 - 2.0 * u)
    return float(s_min + (1.0 - s_min) * smooth)


def compute_empf_step_scale(
    ema_residual: float,
    batch_size: int,
    dim: int,
    *,
    min_batch: int = 16,
) -> float:
    """State-driven DualNorm step-scale damping for 1D deep stationary/shifted holds."""
    if int(dim) != 1:
        return 1.0
    b = max(int(min_batch), int(batch_size))
    inv_sqrt_b = 1.0 / math.sqrt(float(b))
    r_step_floor = 1.52 * inv_sqrt_b
    r_step_tau = 2.25 * inv_sqrt_b
    u = max(0.0, min(1.0, (float(ema_residual) - r_step_floor) / max(r_step_tau - r_step_floor, 1e-12)))
    smooth = u * u * (3.0 - 2.0 * u)
    return float(0.48 + 0.52 * smooth)


def _validate_empf_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _EMPF_STATE_KEYS:
        raise ValueError("invalid EMPF record packet fields")
    if type(packet["schema"]) is not int or packet["schema"] != 2:
        raise ValueError("unsupported EMPF record schema")
    for key in ("initialized", "proto_initialized"):
        if type(packet[key]) is not bool:
            raise ValueError(f"invalid EMPF boolean field {key}")
    if type(packet["revivals"]) is not int or packet["revivals"] < 0:
        raise ValueError("invalid EMPF revival count")
    for key in ("ema_residual", "ema_data_scale", "thermal_scale", "step_scale", "chord_phase", "revival_phase"):
        value = packet[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid EMPF scalar field {key}")
    cond_e = packet["ema_cond_energy"]
    if isinstance(cond_e, bool) or not isinstance(cond_e, (int, float)) or not math.isfinite(cond_e):
        raise ValueError("invalid EMPF scalar field ema_cond_energy")
    if float(packet["thermal_scale"]) > 1.0:
        raise ValueError("invalid EMPF thermal scale > 1")
    if float(packet["step_scale"]) <= 0.0 or float(packet["step_scale"]) > 1.0:
        raise ValueError("invalid EMPF step scale outside (0, 1]")
    for key in ("chord_phase", "revival_phase"):
        if float(packet[key]) >= 1.0:
            raise ValueError(f"invalid EMPF phase {key} >= 1")

    proto_init = packet["proto_initialized"]
    p_sum = packet["proto_sum"]
    p_count = packet["proto_count"]
    if not proto_init:
        if p_sum is not None or p_count is not None:
            raise ValueError("uninitialized EMPF prototype bank must have None tensors")
    else:
        if not isinstance(p_sum, torch.Tensor) or not isinstance(p_count, torch.Tensor):
            raise ValueError("initialized EMPF prototype bank requires proto_sum and proto_count tensors")
        if p_sum.dtype != torch.float32 or p_count.dtype != torch.float32:
            raise ValueError("EMPF prototype bank tensors must have float32 dtype")
        if p_sum.ndim != 2 or p_count.ndim != 1:
            raise ValueError("invalid EMPF prototype tensor rank")
        if p_sum.shape[0] == 0 or p_sum.shape[1] == 0 or p_sum.shape[0] != p_count.shape[0]:
            raise ValueError("incompatible EMPF prototype tensor shapes")
        if not bool(torch.isfinite(p_sum).all()) or not bool(torch.isfinite(p_count).all()):
            raise ValueError("nonfinite EMPF prototype tensor values")
        if bool((p_count <= 0).any()):
            raise ValueError("EMPF prototype counts must be strictly positive")
    return packet


class _EMPFRecordMixin:
    """Shared EMPF sufficient-statistics and thermal state for critic records."""

    def _init_empf_fields(self) -> None:
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
        self._last_d_logits: Tuple[torch.Tensor, torch.Tensor] | None = None

    def observe_transport(self, res: float, data_scale: float, ema_decay: float = 0.92) -> None:
        res_f = float(res)
        scale_f = float(data_scale)
        if not self.initialized:
            self.ema_residual = res_f
            self.ema_data_scale = scale_f
            self.initialized = True
        elif res_f > max(5.5 * self.ema_residual, self.ema_residual + 0.55):
            # Genuine distribution shift / mode-loss shock: re-heat immediately.
            self.ema_residual = res_f
            self.ema_data_scale = scale_f
        else:
            # Smooth EMA filters normal mini-batch statistical variance.
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

    def _empf_packet(self) -> Dict[str, Any]:
        return {
            "schema": 2,
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

    def _restore_empf_packet(self, packet: Dict[str, Any]) -> None:
        _validate_empf_packet(packet)
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
        self._last_d_logits = None

    def restore_device_state(self, device: torch.device) -> None:
        if self.proto_sum is not None:
            self.proto_sum = self.proto_sum.to(device=device)
        if self.proto_count is not None:
            self.proto_count = self.proto_count.to(device=device)


class _EMPFObserverStepRecord(_EMPFRecordMixin, CriticStepRecord):
    """Observer record for DualNorm and plain Adam critics under EMPF."""

    def __init__(self) -> None:
        super().__init__()
        self._init_empf_fields()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "empf": self._empf_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "empf" not in state:
            raise ValueError("EMPF observer checkpoint is missing 'empf' state")
        if set(state) != {*CriticStepRecord.KEYS, "empf"}:
            raise ValueError("invalid EMPF observer record fields")
        _validate_empf_packet(state["empf"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_empf_packet(state["empf"])


class EMPFStepRecord(_EMPFRecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``EMPFCriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "empf")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_empf_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._last_d_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "empf", "empf": self._empf_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid EMPFStepRecord checkpoint keys")
        if state.get("formulation") != "empf":
            raise ValueError("checkpoint formulation is not 'empf'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid EMPFStepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("EMPFStepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            value = state[key]
            if key == "lr_last" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid EMPFStepRecord {key}")
        _validate_empf_packet(state["empf"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_empf_packet(state["empf"])


class EMPFCriticAdam(K3PCriticAdam):
    """Formulation critic Adam for EMPF (no EMA critic copy; carries ``EMPFStepRecord``)."""

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
        del ema_critic  # EMPF uses a zero-mean logit gauge instead of an EMA critic anchor.
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = EMPFStepRecord()
        _attach_empf_critic_hook(critic, self.record)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)
