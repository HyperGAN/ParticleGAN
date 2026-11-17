"""SMPF critic step records, instance-local capture ring & thermal binding, and formulation Adam."""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

FROZEN_THERMAL_SCALE = 1e-4
_MAX_CAPTURE_ENTRIES = 16

_SMPF_STATE_KEYS = frozenset({
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
    "proto_centroids",
    "proto_usage",
    "proto_sse",
    "feat_scale",
    "fast_sum",
    "fast_count",
})


def _storage_key(tensor: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.numel() == 0:
        return None
    try:
        ptr = int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return None
    if ptr == 0:
        return None
    return (ptr, int(tensor.storage_offset()), int(tensor.numel()), int(tensor.shape[0]))


def bind_smpf_record(record: Any, *readers: Any) -> None:
    """Bind instance-local SMPF critic state to loss/prior readers (zero global state)."""
    for reader in readers:
        if reader is not None and hasattr(reader, "smpf_record"):
            reader.smpf_record = record


def read_smpf_thermal(source: Any, default: float = 1.0) -> float:
    """Read instance-bound SMPF thermal scale, returning ``default`` when unbound."""
    if source is None:
        return float(default)
    return float(getattr(source, "_thermal_scale", default))


def _extract_smpf_call_parts(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[
    Optional[torch.Tensor],
    bool,
    Tuple[torch.Tensor, ...],
    Optional[torch.Tensor],
    Optional[Tuple[torch.Tensor, ...]],
]:
    if batch_size <= 0 or not args or not isinstance(args[0], torch.Tensor):
        return None, False, (), None, None
    x_primary = args[0]
    if x_primary.ndim < 1 or x_primary.shape[0] != batch_size:
        return None, False, (), None, None

    extra_items = list(args[1:])
    if kwargs:
        for key in sorted(kwargs):
            extra_items.append(kwargs[key])

    is_cond = bool(kwargs or len(args) > 1)
    cond_list = []
    float_parts = [x_primary.reshape(batch_size, -1)] if x_primary.is_floating_point() else []
    disc_parts = []
    for item in extra_items:
        if not isinstance(item, torch.Tensor) or item.ndim < 1 or item.shape[0] != batch_size or item.numel() == 0:
            continue
        cond_list.append(item.detach())
        if item.is_floating_point():
            float_parts.append(item.reshape(batch_size, -1))
        elif not item.is_complex():
            disc_parts.append(item.detach().to(dtype=torch.int64).reshape(batch_size, -1))

    cond_disc = None if not disc_parts else (
        disc_parts[0] if len(disc_parts) == 1 else torch.cat(disc_parts, dim=-1)
    )
    parts_tuple = tuple(float_parts) if len(float_parts) > 1 else None
    return x_primary, is_cond, tuple(cond_list), cond_disc, parts_tuple


def _record_smpf_capture(
    output: Any,
    x_in: Optional[torch.Tensor],
    rec: Any,
    is_cond: bool,
    cond_tensors: Tuple[torch.Tensor, ...] = (),
    cond_disc: Optional[torch.Tensor] = None,
    float_parts: Optional[Tuple[torch.Tensor, ...]] = None,
    *,
    thermally_scaled: bool = False,
    overwrite: bool = False,
) -> None:
    if x_in is None:
        return
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1 or target.shape[0] != x_in.shape[0]:
        return
    if overwrite or not hasattr(target, "_smpf_input"):
        target._smpf_input = x_in
        target._smpf_conditioned = bool(is_cond)
        target._smpf_cond_tensors = cond_tensors
        target._smpf_cond_disc = cond_disc
        target._smpf_parts = float_parts
        target._smpf_thermally_scaled = bool(thermally_scaled)
    if rec is not None and (overwrite or not hasattr(target, "_smpf_record")):
        target._smpf_record = rec

    if rec is not None and hasattr(rec, "_capture_ring"):
        key = _storage_key(target)
        if key is not None:
            ring: OrderedDict = rec._capture_ring
            if overwrite or key not in ring:
                ring[key] = (
                    x_in,
                    bool(is_cond),
                    cond_tensors,
                    cond_disc,
                    float_parts,
                    bool(thermally_scaled),
                )
                ring.move_to_end(key)
                while len(ring) > _MAX_CAPTURE_ENTRIES:
                    ring.popitem(last=False)


def _finalize_for_logits(logits: torch.Tensor, x_val: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x_val is None:
        return None
    if not logits.requires_grad and x_val.requires_grad:
        return x_val.detach()
    return x_val


def _lookup_smpf_capture(
    logits: Optional[torch.Tensor],
    fallback_record: Any = None,
) -> Tuple[
    Optional[torch.Tensor],
    Any,
    bool,
    Tuple[torch.Tensor, ...],
    Optional[torch.Tensor],
    Optional[Tuple[torch.Tensor, ...]],
    bool,
]:
    """Recover captured critic inputs even when callers pass ``critic(real).detach()``."""
    if logits is None or not isinstance(logits, torch.Tensor) or logits.ndim < 1:
        return None, fallback_record, False, (), None, None, False
    b = int(logits.shape[0])
    x_attr = getattr(logits, "_smpf_input", None)
    r_attr = getattr(logits, "_smpf_record", None)
    rec = r_attr if r_attr is not None else fallback_record
    if isinstance(x_attr, torch.Tensor) and x_attr.shape[0] == b:
        parts_raw = getattr(logits, "_smpf_parts", None)
        parts_fin = (
            tuple(_finalize_for_logits(logits, p) for p in parts_raw)  # type: ignore[misc]
            if parts_raw
            else None
        )
        return (
            _finalize_for_logits(logits, x_attr),
            rec,
            bool(getattr(logits, "_smpf_conditioned", False)),
            getattr(logits, "_smpf_cond_tensors", ()),
            getattr(logits, "_smpf_cond_disc", None),
            parts_fin,
            bool(getattr(logits, "_smpf_thermally_scaled", False)),
        )

    key = _storage_key(logits)
    if key is not None and rec is not None:
        ring = getattr(rec, "_capture_ring", None)
        if ring is not None and key in ring:
            x_c, is_c, ct_c, cd_c, p_c, th_c = ring[key]
            if isinstance(x_c, torch.Tensor) and x_c.shape[0] == b:
                parts_fin = (
                    tuple(_finalize_for_logits(logits, p) for p in p_c)  # type: ignore[misc]
                    if p_c
                    else None
                )
                return (
                    _finalize_for_logits(logits, x_c),
                    rec,
                    bool(is_c),
                    ct_c,
                    cd_c,
                    parts_fin,
                    bool(th_c),
                )
    return None, rec, False, (), None, None, False


def _smpf_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-bound straight-through thermal cooling and input/conditioning capture."""
    rec = getattr(module, "_smpf_record", None)
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    b_out = int(target.shape[0]) if isinstance(target, torch.Tensor) and target.ndim >= 1 else 0
    x_in, is_cond, cond_tensors, cond_disc, float_parts = _extract_smpf_call_parts(args, kwargs, b_out)
    if x_in is not None and (rec is None or not getattr(rec, "_in_penalty", False)):
        _record_smpf_capture(
            output,
            x_in,
            rec,
            is_cond,
            cond_tensors,
            cond_disc,
            float_parts,
            thermally_scaled=False,
        )
    if rec is not None and not getattr(rec, "_in_penalty", False) and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            if x_in is not None:
                _record_smpf_capture(
                    out_scaled,
                    x_in,
                    rec,
                    is_cond,
                    cond_tensors,
                    cond_disc,
                    float_parts,
                    thermally_scaled=True,
                    overwrite=True,
                )
            return out_scaled
    return None


def _attach_smpf_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local SMPF forward hook to ``critic`` (zero global state)."""
    record._in_penalty = False
    if not isinstance(critic, nn.Module):
        return
    critic._smpf_record = record
    if getattr(critic, "_smpf_hook_registered", False):
        return
    critic.register_forward_hook(_smpf_critic_forward_hook, with_kwargs=True)
    critic._smpf_hook_registered = True


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
    """State-driven smoothstep thermal scale without step clocks."""
    b = max(int(min_batch), int(batch_size))
    d = max(1, min(int(max_dim), int(dim)))
    if d == 1:
        inv_sqrt_b = 1.0 / math.sqrt(float(b))
        r_floor = 1.68 * inv_sqrt_b
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


def compute_smpf_step_scale(
    ema_residual: float,
    batch_size: int,
    dim: int,
    *,
    min_batch: int = 16,
) -> float:
    """State-driven DualNorm step-scale damping for 1D stationary/shifted holds."""
    if int(dim) != 1:
        return 1.0
    b = max(int(min_batch), int(batch_size))
    inv_sqrt_b = 1.0 / math.sqrt(float(b))
    r_step_floor = 1.50 * inv_sqrt_b
    r_step_tau = 2.80 * inv_sqrt_b
    u = max(0.0, min(1.0, (float(ema_residual) - r_step_floor) / max(r_step_tau - r_step_floor, 1e-12)))
    smooth = u * u * (3.0 - 2.0 * u)
    return float(0.20 + 0.80 * smooth)


def _validate_smpf_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _SMPF_STATE_KEYS:
        raise ValueError("invalid SMPF record packet fields")
    if type(packet["schema"]) is not int or packet["schema"] != 1:
        raise ValueError("unsupported SMPF record schema")
    for key in ("initialized", "proto_initialized"):
        if type(packet[key]) is not bool:
            raise ValueError(f"invalid SMPF boolean field {key}")
    if type(packet["revivals"]) is not int or packet["revivals"] < 0:
        raise ValueError("invalid SMPF revival count")
    for key in ("ema_residual", "ema_data_scale", "thermal_scale", "step_scale", "chord_phase", "revival_phase"):
        value = packet[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid SMPF scalar field {key}")
    cond_e = packet["ema_cond_energy"]
    if isinstance(cond_e, bool) or not isinstance(cond_e, (int, float)) or not math.isfinite(cond_e):
        raise ValueError("invalid SMPF scalar field ema_cond_energy")
    if float(packet["thermal_scale"]) > 1.0:
        raise ValueError("invalid SMPF thermal scale > 1")
    if float(packet["step_scale"]) <= 0.0 or float(packet["step_scale"]) > 1.0:
        raise ValueError("invalid SMPF step scale outside (0, 1]")
    for key in ("chord_phase", "revival_phase"):
        if float(packet[key]) >= 1.0:
            raise ValueError(f"invalid SMPF phase {key} >= 1")

    proto_init = packet["proto_initialized"]
    tensor_keys = ("proto_centroids", "proto_usage", "proto_sse", "feat_scale", "fast_sum", "fast_count")
    if not proto_init:
        if any(packet[tk] is not None for tk in tensor_keys):
            raise ValueError("uninitialized SMPF prototype bank must have None tensors")
    else:
        for tk in tensor_keys:
            t_val = packet[tk]
            if not isinstance(t_val, torch.Tensor):
                raise ValueError(f"initialized SMPF prototype bank requires tensor {tk}")
            if t_val.dtype != torch.float32:
                raise ValueError(f"SMPF prototype tensor {tk} must have float32 dtype")
            if not bool(torch.isfinite(t_val).all()):
                raise ValueError(f"nonfinite SMPF prototype tensor values in {tk}")
        p_c = packet["proto_centroids"]
        p_u = packet["proto_usage"]
        p_sse = packet["proto_sse"]
        p_fs = packet["feat_scale"]
        p_fsum = packet["fast_sum"]
        p_fcnt = packet["fast_count"]
        if p_c.ndim != 2 or p_fsum.ndim != 2 or p_u.ndim != 1 or p_sse.ndim != 1 or p_fs.ndim != 1 or p_fcnt.ndim != 1:
            raise ValueError("invalid SMPF prototype tensor rank")
        k_n, d_n = int(p_c.shape[0]), int(p_c.shape[1])
        if k_n == 0 or d_n == 0:
            raise ValueError("empty SMPF prototype tensor shape")
        if (
            p_fsum.shape != (k_n, d_n)
            or p_u.shape != (k_n,)
            or p_sse.shape != (k_n,)
            or p_fcnt.shape != (k_n,)
            or p_fs.shape != (d_n,)
        ):
            raise ValueError("incompatible SMPF prototype tensor shapes")
        if bool((p_u <= 0).any()) or bool((p_fcnt <= 0).any()) or bool((p_fs <= 0).any()) or bool((p_sse < 0).any()):
            raise ValueError("SMPF prototype usage, fast_count, and feat_scale must be positive and sse nonnegative")
    return packet


class _SMPFRecordMixin:
    """Shared SMPF s-means sufficient-statistics and thermal state for critic records."""

    def _init_smpf_fields(self) -> None:
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
        self.proto_centroids: torch.Tensor | None = None
        self.proto_usage: torch.Tensor | None = None
        self.proto_sse: torch.Tensor | None = None
        self.feat_scale: torch.Tensor | None = None
        self.fast_sum: torch.Tensor | None = None
        self.fast_count: torch.Tensor | None = None
        # Ephemeral runtime caches (never persisted; reconstructed or cleared cleanly).
        self._capture_ring: OrderedDict = OrderedDict()
        self._last_d_logits: Tuple[torch.Tensor, torch.Tensor] | None = None
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._discrete_prior_step: bool = False

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
        proto_centroids: torch.Tensor,
        proto_usage: torch.Tensor,
        proto_sse: torch.Tensor,
        feat_scale: torch.Tensor,
        fast_sum: torch.Tensor,
        fast_count: torch.Tensor,
        revival_phase: float,
        revivals: int,
    ) -> None:
        self.proto_initialized = True
        self.proto_centroids = proto_centroids.detach().clone().to(dtype=torch.float32)
        self.proto_usage = proto_usage.detach().clone().to(dtype=torch.float32)
        self.proto_sse = proto_sse.detach().clone().to(dtype=torch.float32)
        self.feat_scale = feat_scale.detach().clone().to(dtype=torch.float32)
        self.fast_sum = fast_sum.detach().clone().to(dtype=torch.float32)
        self.fast_count = fast_count.detach().clone().to(dtype=torch.float32)
        self.revival_phase = float(revival_phase)
        self.revivals = int(revivals)

    def readout_eb_prototypes(
        self,
        *,
        eb_tau: float = 0.20,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(c_eb, weights, ell_proto, feat_scale)`` from stored bank state without mutating state."""
        if not self.proto_initialized or self.proto_centroids is None:
            return None
        from .prototypes import _empirical_bayes_readout

        assert self.proto_usage is not None and self.proto_sse is not None
        assert self.feat_scale is not None and self.fast_sum is not None and self.fast_count is not None
        c_eb, weights, ell_proto = _empirical_bayes_readout(
            self.proto_centroids,
            self.proto_usage,
            self.proto_sse,
            self.fast_sum,
            self.fast_count,
            eb_tau=eb_tau,
            out_dtype=dtype,
        )
        f_sc = self.feat_scale.detach().to(dtype=dtype)
        if device is not None:
            c_eb = c_eb.to(device=device)
            weights = weights.to(device=device)
            ell_proto = ell_proto.to(device=device)
            f_sc = f_sc.to(device=device)
        return c_eb, weights, ell_proto, f_sc

    def _smpf_packet(self) -> Dict[str, Any]:
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
            "proto_centroids": None if self.proto_centroids is None else self.proto_centroids.detach().clone(),
            "proto_usage": None if self.proto_usage is None else self.proto_usage.detach().clone(),
            "proto_sse": None if self.proto_sse is None else self.proto_sse.detach().clone(),
            "feat_scale": None if self.feat_scale is None else self.feat_scale.detach().clone(),
            "fast_sum": None if self.fast_sum is None else self.fast_sum.detach().clone(),
            "fast_count": None if self.fast_count is None else self.fast_count.detach().clone(),
        }

    def _restore_smpf_packet(self, packet: Dict[str, Any]) -> None:
        _validate_smpf_packet(packet)
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
        self.proto_centroids = None if packet["proto_centroids"] is None else packet["proto_centroids"].detach().clone()
        self.proto_usage = None if packet["proto_usage"] is None else packet["proto_usage"].detach().clone()
        self.proto_sse = None if packet["proto_sse"] is None else packet["proto_sse"].detach().clone()
        self.feat_scale = None if packet["feat_scale"] is None else packet["feat_scale"].detach().clone()
        self.fast_sum = None if packet["fast_sum"] is None else packet["fast_sum"].detach().clone()
        self.fast_count = None if packet["fast_count"] is None else packet["fast_count"].detach().clone()
        self._capture_ring.clear()
        self._last_d_logits = None
        self._in_penalty = False
        self._in_protected_probe = False
        self._discrete_prior_step = False

    def restore_device_state(self, device: torch.device) -> None:
        for attr in ("proto_centroids", "proto_usage", "proto_sse", "feat_scale", "fast_sum", "fast_count"):
            val = getattr(self, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device=device))


class _SMPFObserverStepRecord(_SMPFRecordMixin, CriticStepRecord):
    """Observer record for DualNorm and plain Adam critics under SMPF."""

    def __init__(self) -> None:
        super().__init__()
        self._init_smpf_fields()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "smpf": self._smpf_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "smpf" not in state:
            raise ValueError("SMPF observer checkpoint is missing 'smpf' state")
        if set(state) != {*CriticStepRecord.KEYS, "smpf"}:
            raise ValueError("invalid SMPF observer record fields")
        _validate_smpf_packet(state["smpf"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_smpf_packet(state["smpf"])


class SMPFStepRecord(_SMPFRecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``SMPFCriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "smpf")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_smpf_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._last_d_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "smpf", "smpf": self._smpf_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid SMPFStepRecord checkpoint keys")
        if state.get("formulation") != "smpf":
            raise ValueError("checkpoint formulation is not 'smpf'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid SMPFStepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("SMPFStepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            value = state[key]
            if key == "lr_last" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid SMPFStepRecord {key}")
        _validate_smpf_packet(state["smpf"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_smpf_packet(state["smpf"])


class SMPFCriticAdam(K3PCriticAdam):
    """Formulation critic Adam for SMPF (no EMA critic copy; carries ``SMPFStepRecord``)."""

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
        del ema_critic  # SMPF uses a zero-mean logit gauge instead of an EMA critic anchor.
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = SMPFStepRecord()
        _attach_smpf_critic_hook(critic, self.record)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)
