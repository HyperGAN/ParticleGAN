"""RGDT critic state: instance-local step records, capture ring, thermal equilibrium controller, and Adam."""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

_MAX_RING_ENTRIES = 16

_RGDT_PACKET_KEYS = frozenset({
    "schema",
    "ell_ema",
    "ell_initialized",
    "chord_phase",
    "thermal_initialized",
    "ema_ed",
    "ema_mu_sq",
    "ema_spread",
    "ema_q90",
    "ema_hmax",
    "ema_tail",
    "cond_initialized",
    "cond_ema",
    "thermal_scale",
    "step_scale",
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


def _extract_rgdt_inputs_parts(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool, Optional[Tuple[torch.Tensor, ...]]]:
    """Extract concatenated floating inputs, discrete conditions, continuous-fiber flag, and raw float parts."""
    if batch_size <= 0:
        return None, None, False, None
    float_parts = []
    disc_parts = []
    items = list(args)
    if kwargs:
        for key in sorted(kwargs):
            items.append(kwargs[key])
    for item in items:
        if not isinstance(item, torch.Tensor) or item.ndim < 1 or item.shape[0] != batch_size or item.numel() == 0:
            continue
        if item.is_floating_point():
            float_parts.append(item.reshape(batch_size, -1))
        elif not item.is_complex():
            disc_parts.append(item.detach().to(dtype=torch.int64).reshape(batch_size, -1))
    if not float_parts:
        return None, None, False, None
    has_cont_cond = len(float_parts) > 1
    x_in = float_parts[0] if len(float_parts) == 1 else torch.cat(float_parts, dim=-1)
    cond_disc = None if not disc_parts else (
        disc_parts[0] if len(disc_parts) == 1 else torch.cat(disc_parts, dim=-1)
    )
    parts_tuple = tuple(float_parts) if has_cont_cond else None
    return x_in, cond_disc, has_cont_cond, parts_tuple


def _record_rgdt_capture(
    output: Any,
    x_input: Optional[torch.Tensor],
    cond_disc: Optional[torch.Tensor],
    record: Any,
    *,
    has_cont_cond: bool = False,
    float_parts: Optional[Tuple[torch.Tensor, ...]] = None,
    overwrite: bool = False,
) -> None:
    """Attach instance-local input capture to critic output and its bound record ring (zero global state)."""
    if x_input is None:
        return
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1 or target.shape[0] != x_input.shape[0]:
        return
    if overwrite or not hasattr(target, "_rgdt_input"):
        target._rgdt_input = x_input
        target._rgdt_cond = cond_disc
        target._rgdt_cont_cond = bool(has_cont_cond)
        target._rgdt_parts = float_parts
    if record is not None and (overwrite or not hasattr(target, "_rgdt_record")):
        target._rgdt_record = record
    if record is None or not hasattr(record, "_capture_ring"):
        return
    key = _storage_key(target)
    if key is None:
        return
    ring: OrderedDict = record._capture_ring
    if overwrite or key not in ring:
        ring[key] = (x_input, cond_disc, record, bool(has_cont_cond), float_parts)
        ring.move_to_end(key)
        while len(ring) > _MAX_RING_ENTRIES:
            ring.popitem(last=False)


def _finalize_x_for_logits(logits: torch.Tensor, x_val: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x_val is None:
        return None
    if not logits.requires_grad and x_val.requires_grad:
        return x_val.detach()
    return x_val


def _finalize_parts_for_logits(
    logits: torch.Tensor,
    parts: Optional[Tuple[torch.Tensor, ...]],
) -> Optional[Tuple[torch.Tensor, ...]]:
    if not parts:
        return None
    return tuple(_finalize_x_for_logits(logits, p) for p in parts)  # type: ignore[misc]


def _lookup_rgdt_capture_parts(
    logits: torch.Tensor,
    fallback_record: Any = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any, bool, Optional[Tuple[torch.Tensor, ...]]]:
    b = logits.shape[0] if logits.ndim >= 1 else 0
    x_attr = getattr(logits, "_rgdt_input", None)
    c_attr = getattr(logits, "_rgdt_cond", None)
    r_attr = getattr(logits, "_rgdt_record", None)
    cc_attr = bool(getattr(logits, "_rgdt_cont_cond", False))
    p_attr = getattr(logits, "_rgdt_parts", None)
    if isinstance(x_attr, torch.Tensor) and x_attr.shape[0] == b:
        return (
            _finalize_x_for_logits(logits, x_attr),
            c_attr,
            (r_attr if r_attr is not None else fallback_record),
            cc_attr,
            _finalize_parts_for_logits(logits, p_attr),
        )

    key = _storage_key(logits)
    if key is not None:
        for rec_candidate in (r_attr, fallback_record):
            ring = getattr(rec_candidate, "_capture_ring", None)
            if ring is not None and key in ring:
                item = ring[key]
                x_c, c_c, r_c, cc_c, p_c = item[0], item[1], item[2], bool(item[3]), item[4]
                if x_c.shape[0] == b:
                    return (
                        _finalize_x_for_logits(logits, x_c),
                        c_c,
                        (r_c if r_c is not None else fallback_record),
                        cc_c,
                        _finalize_parts_for_logits(logits, p_c),
                    )
    return None, None, (r_attr if r_attr is not None else fallback_record), False, None


def bind_rgdt_record(record: Any, *readers: Any) -> None:
    """Bind instance-local RGDT critic state to loss and prior regularizer readers."""
    for reader in readers:
        if reader is not None:
            if hasattr(reader, "record"):
                reader.record = record
            if hasattr(reader, "rgdt_record"):
                reader.rgdt_record = record


def _rgdt_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-local input/context capture and straight-through thermal cooling hook."""
    rec = getattr(module, "_rgdt_record", None)
    if rec is not None and getattr(rec, "_in_penalty", False):
        return None
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1:
        return None
    b_n = int(target.shape[0])
    x_in, cond_disc, has_cont_cond, float_parts = _extract_rgdt_inputs_parts(args, kwargs, b_n)
    _record_rgdt_capture(
        output,
        x_in,
        cond_disc,
        rec,
        has_cont_cond=has_cont_cond,
        float_parts=float_parts,
    )
    if rec is not None and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            out_scaled._rgdt_thermally_scaled = True
            _record_rgdt_capture(
                out_scaled,
                x_in,
                cond_disc,
                rec,
                has_cont_cond=has_cont_cond,
                float_parts=float_parts,
                overwrite=True,
            )
            return out_scaled
    return None


def _attach_rgdt_critic_hook(critic: Optional[nn.Module], record: CriticStepRecord) -> None:
    """Attach instance-local RGDT forward hook to ``critic`` (idempotent, zero global state)."""
    record._in_penalty = False
    if not isinstance(critic, nn.Module):
        return
    critic._rgdt_record = record
    if getattr(critic, "_rgdt_hook_registered", False):
        return
    critic.register_forward_hook(_rgdt_critic_forward_hook, with_kwargs=True)
    critic._rgdt_hook_registered = True


class _RGDTRecordMixin:
    """Instance-local RGDT conformal horizon, Weyl phase, and thermal equilibrium state."""

    def _init_rgdt_fields(self) -> None:
        self._rgdt_ell_scalar: float = 0.0
        self._rgdt_ell_t: Optional[torch.Tensor] = None
        self.rgdt_ell_initialized: bool = False
        self.rgdt_chord_phase: float = 0.0
        self.rgdt_thermal_initialized: bool = False
        self.rgdt_ema_ed: float = 1.0
        self.rgdt_ema_mu_sq: float = 1.0
        self.rgdt_ema_spread: float = 0.0
        self.rgdt_ema_q90: float = 1.0
        self.rgdt_ema_hmax: float = 1.0
        self.rgdt_ema_tail: float = 0.0
        self.rgdt_cond_initialized: bool = False
        self.rgdt_cond_ema: float = 1.0
        self._thermal_scale: float = 1.0
        self._step_scale: float = 1.0
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._capture_ring: OrderedDict = OrderedDict()

    @property
    def rgdt_ell_ema(self) -> float:
        if self._rgdt_ell_t is not None:
            return float(self._rgdt_ell_t.item())
        return self._rgdt_ell_scalar

    @rgdt_ell_ema.setter
    def rgdt_ell_ema(self, value: float) -> None:
        self._rgdt_ell_scalar = float(value)
        self._rgdt_ell_t = None

    @property
    def rgdt_ema_residual(self) -> float:
        return max(
            0.5 * max(0.0, float(self.rgdt_ema_ed)),
            math.sqrt(max(0.0, float(self.rgdt_ema_mu_sq))),
            0.60 * abs(float(self.rgdt_ema_spread)),
            max(0.0, float(self.rgdt_ema_q90)),
            max(0.0, float(self.rgdt_ema_hmax)),
            max(0.0, float(self.rgdt_ema_tail)),
        )

    def observe_rgdt_thermal(
        self,
        r_ed: float,
        r_mu_sq: float,
        r_spread: float,
        r_q90: float,
        r_hmax: float,
        tau: float,
        *,
        ema_decay: float = 0.96,
        s_min: float = 1e-6,
        shock_mu_ratio: float = 0.35,
        shock_ed_floor: float = 0.28,
        lock_hysteresis: float = 0.0,
        r_tail: float = 0.0,
    ) -> float:
        ed_f = float(r_ed)
        mu_sq_f = float(r_mu_sq)
        spread_f = float(r_spread)
        q90_f = float(r_q90)
        hmax_f = float(r_hmax)
        tail_f = float(r_tail)
        tau_f = float(tau)
        tau_pos = max(0.0, tau_f)

        cap_lin = max(0.20, 2.0 * tau_pos)
        cap_mu_sq = cap_lin * cap_lin
        cap_ed = max(0.70, 4.0 * tau_pos)
        ed_c = max(-cap_ed, min(ed_f, cap_ed))
        mu_sq_c = max(-cap_mu_sq, min(mu_sq_f, cap_mu_sq))
        spread_c = max(-cap_lin, min(spread_f, cap_lin))
        q90_c = max(-cap_lin, min(q90_f, cap_lin))
        hmax_c = max(-cap_lin, min(hmax_f, cap_lin))
        tail_c = max(-cap_lin, min(tail_f, cap_lin))

        shock_ed_thresh = max(float(shock_ed_floor), 4.0 * tau_pos * tau_pos)
        shock_mu_thresh = float(shock_mu_ratio) * shock_ed_thresh
        if not self.rgdt_thermal_initialized or (
            self._thermal_scale < 0.99
            and (ed_f > shock_ed_thresh or mu_sq_f > shock_mu_thresh or hmax_f > 0.65)
        ):
            self.rgdt_ema_ed = ed_c
            self.rgdt_ema_mu_sq = mu_sq_c
            self.rgdt_ema_spread = spread_c
            self.rgdt_ema_q90 = q90_c
            self.rgdt_ema_hmax = hmax_c
            self.rgdt_ema_tail = tail_c
            self.rgdt_thermal_initialized = True
        else:
            d = float(ema_decay)
            omd = 1.0 - d
            self.rgdt_ema_ed = d * self.rgdt_ema_ed + omd * ed_c
            self.rgdt_ema_mu_sq = d * self.rgdt_ema_mu_sq + omd * mu_sq_c
            self.rgdt_ema_spread = d * self.rgdt_ema_spread + omd * spread_c
            self.rgdt_ema_q90 = d * self.rgdt_ema_q90 + omd * q90_c
            self.rgdt_ema_hmax = d * self.rgdt_ema_hmax + omd * hmax_c
            self.rgdt_ema_tail = d * self.rgdt_ema_tail + omd * tail_c

        if tau_f <= 0.0:
            self._thermal_scale = 1.0
            return 1.0
        lo_frac = 0.50 + (float(lock_hysteresis) if self._thermal_scale <= 1e-4 else 0.0)
        tau_lo = min(0.92, lo_frac) * tau_f
        span = max(1e-8, tau_f - tau_lo)
        u = min(1.0, max(0.0, (self.rgdt_ema_residual - tau_lo) / span))
        theta = u * u * (3.0 - 2.0 * u)
        if theta >= 1.0:
            s = 1.0
        elif theta <= 0.0:
            s = float(s_min)
        else:
            s = max(float(s_min), min(1.0, 0.04 * theta / math.sqrt(max(1e-8, 1.0 - theta * theta))))
        self._thermal_scale = float(max(float(s_min), min(1.0, s)))
        return float(self._thermal_scale)

    def _rgdt_packet(self) -> Dict[str, Any]:
        return {
            "schema": 1,
            "ell_ema": float(self.rgdt_ell_ema),
            "ell_initialized": bool(self.rgdt_ell_initialized),
            "chord_phase": float(self.rgdt_chord_phase),
            "thermal_initialized": bool(self.rgdt_thermal_initialized),
            "ema_ed": float(self.rgdt_ema_ed),
            "ema_mu_sq": float(self.rgdt_ema_mu_sq),
            "ema_spread": float(self.rgdt_ema_spread),
            "ema_q90": float(self.rgdt_ema_q90),
            "ema_hmax": float(self.rgdt_ema_hmax),
            "ema_tail": float(self.rgdt_ema_tail),
            "cond_initialized": bool(self.rgdt_cond_initialized),
            "cond_ema": float(self.rgdt_cond_ema),
            "thermal_scale": float(self._thermal_scale),
            "step_scale": float(self._step_scale),
        }

    @staticmethod
    def _validate_rgdt_packet(packet: Any) -> Dict[str, Any]:
        if not isinstance(packet, dict) or set(packet) != _RGDT_PACKET_KEYS:
            raise ValueError("invalid RGDT record packet fields")
        if type(packet["schema"]) is not int or packet["schema"] != 1:
            raise ValueError("unsupported RGDT record schema")
        for key in ("ell_initialized", "thermal_initialized", "cond_initialized"):
            if type(packet[key]) is not bool:
                raise ValueError(f"invalid RGDT boolean field {key}")
        for key in ("ell_ema", "chord_phase", "thermal_scale", "step_scale"):
            val = packet[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"invalid RGDT nonnegative scalar {key}")
        for key in ("ema_ed", "ema_mu_sq", "ema_spread", "ema_q90", "ema_hmax", "ema_tail", "cond_ema"):
            sval = packet[key]
            if isinstance(sval, bool) or not isinstance(sval, (int, float)) or not math.isfinite(sval):
                raise ValueError(f"invalid RGDT scalar {key}")
        if float(packet["chord_phase"]) >= 1.0:
            raise ValueError("invalid RGDT chord_phase >= 1")
        if not (0.0 < float(packet["thermal_scale"]) <= 1.0):
            raise ValueError("invalid RGDT thermal_scale outside (0, 1]")
        if not (0.0 < float(packet["step_scale"]) <= 1.0):
            raise ValueError("invalid RGDT step_scale outside (0, 1]")
        return packet

    def _restore_rgdt_packet(self, packet: Dict[str, Any]) -> None:
        self._rgdt_ell_scalar = float(packet["ell_ema"])
        self._rgdt_ell_t = None
        self.rgdt_ell_initialized = bool(packet["ell_initialized"])
        self.rgdt_chord_phase = float(packet["chord_phase"])
        self.rgdt_thermal_initialized = bool(packet["thermal_initialized"])
        self.rgdt_ema_ed = float(packet["ema_ed"])
        self.rgdt_ema_mu_sq = float(packet["ema_mu_sq"])
        self.rgdt_ema_spread = float(packet["ema_spread"])
        self.rgdt_ema_q90 = float(packet["ema_q90"])
        self.rgdt_ema_hmax = float(packet["ema_hmax"])
        self.rgdt_ema_tail = float(packet["ema_tail"])
        self.rgdt_cond_initialized = bool(packet["cond_initialized"])
        self.rgdt_cond_ema = float(packet["cond_ema"])
        self._thermal_scale = float(packet["thermal_scale"])
        self._step_scale = float(packet["step_scale"])
        self._capture_ring = OrderedDict()

    def _copy_rgdt_into(self, dup: "_RGDTRecordMixin") -> None:
        dup._rgdt_ell_scalar = self._rgdt_ell_scalar
        dup._rgdt_ell_t = None if self._rgdt_ell_t is None else self._rgdt_ell_t.clone()
        dup.rgdt_ell_initialized = self.rgdt_ell_initialized
        dup.rgdt_chord_phase = self.rgdt_chord_phase
        dup.rgdt_thermal_initialized = self.rgdt_thermal_initialized
        dup.rgdt_ema_ed = self.rgdt_ema_ed
        dup.rgdt_ema_mu_sq = self.rgdt_ema_mu_sq
        dup.rgdt_ema_spread = self.rgdt_ema_spread
        dup.rgdt_ema_q90 = self.rgdt_ema_q90
        dup.rgdt_ema_hmax = self.rgdt_ema_hmax
        dup.rgdt_ema_tail = self.rgdt_ema_tail
        dup.rgdt_cond_initialized = self.rgdt_cond_initialized
        dup.rgdt_cond_ema = self.rgdt_cond_ema
        dup._thermal_scale = self._thermal_scale
        dup._step_scale = self._step_scale
        dup._in_penalty = self._in_penalty
        dup._in_protected_probe = self._in_protected_probe


class _RGDTObserverStepRecord(_RGDTRecordMixin, CriticStepRecord):
    """Normalized/plain-Adam observer record carrying checkpointed RGDT state."""

    def __init__(self) -> None:
        super().__init__()
        self._init_rgdt_fields()

    def record_step(self, optimizer_or_lr: Any) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "rgdt": self._rgdt_packet(),
        }

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "rgdt" not in state:
            raise ValueError("RGDT observer checkpoint is missing rgdt state")
        if set(state) != {*CriticStepRecord.KEYS, "rgdt"}:
            raise ValueError("invalid RGDT observer record fields")
        _RGDTRecordMixin._validate_rgdt_packet(state["rgdt"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_rgdt_packet(state["rgdt"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_RGDTObserverStepRecord":
        dup = _RGDTObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_rgdt_into(dup)
        return dup


class RGDTStepRecord(_RGDTRecordMixin, CriticStepRecord):
    """Formulation-owned RGDT critic record for ``RGDTCriticAdam``."""

    KEYS = (*CriticStepRecord.KEYS, "formulation", "rgdt")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_rgdt_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "formulation": "rgdt",
            "rgdt": self._rgdt_packet(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "rgdt":
            raise ValueError("RGDT critic optimizer requires an RGDT controller checkpoint")
        if set(state) != set(self.KEYS):
            raise ValueError("invalid RGDT controller state keys")
        base = {key: state[key] for key in CriticStepRecord.KEYS}
        packet = self._validate_rgdt_packet(state["rgdt"])
        super().load_state_dict(base)
        self._restore_rgdt_packet(packet)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "RGDTStepRecord":
        dup = RGDTStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_rgdt_into(dup)
        return dup


class RGDTCriticAdam(K3PCriticAdam):
    """Critic Adam optimizer paired with instance-local RGDT state (zero EMA critic copy)."""

    def __init__(
        self,
        params: Any,
        *,
        critic: Optional[nn.Module] = None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs: Any,
    ) -> None:
        record = RGDTStepRecord()
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = record
        self._rgdt_record = record
        _attach_rgdt_critic_hook(critic, record)
