"""RHBI critic state: step records, input/conditioning capture hook, and critic Adam optimizer."""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .._bispf.thermal import _LEGACY_THERMAL_CHANNEL, _publish_thermal
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

_MAX_RING_ENTRIES = 16
_RECENT_CAPTURES: "OrderedDict[Tuple[int, int, int, int], Tuple[torch.Tensor, Optional[torch.Tensor], Any, bool]]" = OrderedDict()


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


def _extract_critic_inputs_parts(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool, Optional[Tuple[torch.Tensor, ...]]]:
    """Extract concatenated floating inputs, discrete conditions, continuous-fibre flag, and raw float parts."""
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


def _extract_critic_inputs(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """Extract concatenated floating batch inputs, discrete conditions, and continuous-fibre flag."""
    x_in, cond_disc, has_cont_cond, _ = _extract_critic_inputs_parts(args, kwargs, batch_size)
    return x_in, cond_disc, has_cont_cond


def _record_rhbi_capture(
    output: Any,
    x_input: Optional[torch.Tensor],
    cond_disc: Optional[torch.Tensor],
    record: Any,
    *,
    has_cont_cond: bool = False,
    float_parts: Optional[Tuple[torch.Tensor, ...]] = None,
    overwrite: bool = False,
) -> None:
    if x_input is None:
        return
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1 or target.shape[0] != x_input.shape[0]:
        return
    if overwrite or not hasattr(target, "_rhbi_input"):
        target._rhbi_input = x_input
        target._rhbi_cond = cond_disc
        target._rhbi_cont_cond = bool(has_cont_cond)
        target._rhbi_parts = float_parts
    if record is not None and (overwrite or not hasattr(target, "_rhbi_record")):
        target._rhbi_record = record
    key = _storage_key(target)
    if key is None:
        return
    entry = (x_input, cond_disc, record, bool(has_cont_cond), float_parts)
    if record is not None and hasattr(record, "_capture_ring"):
        ring: OrderedDict = record._capture_ring
        if overwrite or key not in ring:
            ring[key] = entry
            ring.move_to_end(key)
            while len(ring) > _MAX_RING_ENTRIES:
                ring.popitem(last=False)
    if overwrite or key not in _RECENT_CAPTURES:
        _RECENT_CAPTURES[key] = entry
        _RECENT_CAPTURES.move_to_end(key)
        while len(_RECENT_CAPTURES) > _MAX_RING_ENTRIES:
            _RECENT_CAPTURES.popitem(last=False)


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


def _lookup_rhbi_capture_parts(
    logits: torch.Tensor,
    fallback_record: Any = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any, bool, Optional[Tuple[torch.Tensor, ...]]]:
    b = logits.shape[0] if logits.ndim >= 1 else 0
    x_attr = getattr(logits, "_rhbi_input", None)
    c_attr = getattr(logits, "_rhbi_cond", None)
    r_attr = getattr(logits, "_rhbi_record", None)
    cc_attr = bool(getattr(logits, "_rhbi_cont_cond", False))
    p_attr = getattr(logits, "_rhbi_parts", None)
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
                x_c, c_c, r_c = item[0], item[1], item[2]
                cc_c = bool(item[3]) if len(item) > 3 else False
                p_c = item[4] if len(item) > 4 else None
                if x_c.shape[0] == b:
                    return (
                        _finalize_x_for_logits(logits, x_c),
                        c_c,
                        (r_c if r_c is not None else fallback_record),
                        cc_c,
                        _finalize_parts_for_logits(logits, p_c),
                    )
        if key in _RECENT_CAPTURES:
            item = _RECENT_CAPTURES[key]
            x_c, c_c, r_c = item[0], item[1], item[2]
            cc_c = bool(item[3]) if len(item) > 3 else False
            p_c = item[4] if len(item) > 4 else None
            if x_c.shape[0] == b:
                return (
                    _finalize_x_for_logits(logits, x_c),
                    c_c,
                    (r_c if r_c is not None else fallback_record),
                    cc_c,
                    _finalize_parts_for_logits(logits, p_c),
                )
    return None, None, (r_attr if r_attr is not None else fallback_record), False, None


def _lookup_rhbi_capture_full(
    logits: torch.Tensor,
    fallback_record: Any = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any, bool]:
    x_c, c_c, r_c, cc_c, _ = _lookup_rhbi_capture_parts(logits, fallback_record)
    return x_c, c_c, r_c, cc_c


def _lookup_rhbi_capture(
    logits: torch.Tensor,
    fallback_record: Any = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any]:
    x_c, c_c, r_c, _ = _lookup_rhbi_capture_full(logits, fallback_record)
    return x_c, c_c, r_c


class _RHBIRecordThermalMixin:
    """Shared instance-local RHBI horizon, Weyl chord phase, and thermal settling state."""

    def _init_rhbi_v4_fields(self) -> None:
        self.rhbi_chord_phase: float = 0.0
        self.rhbi_thermal_initialized: bool = False
        self.rhbi_ema_ed: float = 1.0
        self.rhbi_ema_mu_sq: float = 1.0
        self.rhbi_ema_spread: float = 0.0
        self.rhbi_ema_q90: float = 1.0
        self.rhbi_ema_hmax: float = 1.0
        self.rhbi_ema_tail: float = 0.0
        self._thermal_scale: float = 1.0
        self.legacy_thermal_channel: bool = True

    @property
    def rhbi_ema_residual(self) -> float:
        return max(
            0.5 * max(0.0, float(self.rhbi_ema_ed)),
            math.sqrt(max(0.0, float(self.rhbi_ema_mu_sq))),
            0.60 * abs(float(self.rhbi_ema_spread)),
            max(0.0, float(self.rhbi_ema_q90)),
            max(0.0, float(self.rhbi_ema_hmax)),
            max(0.0, float(self.rhbi_ema_tail)),
        )

    @property
    def rhbi_thermal_scale(self) -> float:
        return float(self._thermal_scale)

    @rhbi_thermal_scale.setter
    def rhbi_thermal_scale(self, value: float) -> None:
        self._thermal_scale = float(value)

    def restore_device_state(self, device: torch.device) -> None:
        _publish_thermal(self, device, self._thermal_scale)

    def observe_rhbi_thermal(
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
        shock_mu_ratio: float = 0.15,
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
        shock_ed_thresh = max(0.30, 4.0 * tau_pos * tau_pos)
        shock_mu_thresh = float(shock_mu_ratio) * shock_ed_thresh
        if not self.rhbi_thermal_initialized or (
            self._thermal_scale < 0.99 and (ed_f > shock_ed_thresh or mu_sq_f > shock_mu_thresh)
        ):
            self.rhbi_ema_ed = ed_c
            self.rhbi_ema_mu_sq = mu_sq_c
            self.rhbi_ema_spread = spread_c
            self.rhbi_ema_q90 = q90_c
            self.rhbi_ema_hmax = hmax_c
            self.rhbi_ema_tail = tail_c
            self.rhbi_thermal_initialized = True
        else:
            self.rhbi_ema_ed = ema_decay * self.rhbi_ema_ed + (1.0 - ema_decay) * ed_c
            self.rhbi_ema_mu_sq = ema_decay * self.rhbi_ema_mu_sq + (1.0 - ema_decay) * mu_sq_c
            self.rhbi_ema_spread = ema_decay * self.rhbi_ema_spread + (1.0 - ema_decay) * spread_c
            self.rhbi_ema_q90 = ema_decay * self.rhbi_ema_q90 + (1.0 - ema_decay) * q90_c
            self.rhbi_ema_hmax = ema_decay * self.rhbi_ema_hmax + (1.0 - ema_decay) * hmax_c
            self.rhbi_ema_tail = ema_decay * self.rhbi_ema_tail + (1.0 - ema_decay) * tail_c
        if tau_f <= 0.0:
            self._thermal_scale = 1.0
            return 1.0
        tau_lo = 0.50 * tau_f
        span = max(1e-8, tau_f - tau_lo)
        u = min(1.0, max(0.0, (self.rhbi_ema_residual - tau_lo) / span))
        theta = u * u * (3.0 - 2.0 * u)
        if theta >= 1.0:
            s = 1.0
        elif theta <= 0.0:
            s = float(s_min)
        else:
            s = max(float(s_min), min(1.0, 0.04 * theta / math.sqrt(max(1e-8, 1.0 - theta * theta))))
        self._thermal_scale = max(float(s_min), min(1.0, s))
        return self._thermal_scale


class _RHBIObserverStepRecord(_RHBIRecordThermalMixin, CriticStepRecord):
    """Passive critic step record with complete RHBI horizon, chord, and thermal continuation."""

    _RHBI_PACKET_KEYS = frozenset({
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
        "thermal_scale",
    })

    def __init__(self) -> None:
        super().__init__()
        self._rhbi_ell_scalar: float = 0.0
        self._rhbi_ell_t: Optional[torch.Tensor] = None
        self.rhbi_ell_initialized: bool = False
        self._init_rhbi_v4_fields()
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._capture_ring: "OrderedDict[Tuple[int, int, int, int], Tuple[torch.Tensor, Optional[torch.Tensor], Any]]" = OrderedDict()

    @property
    def rhbi_ell_ema(self) -> float:
        if self._rhbi_ell_t is not None:
            return float(self._rhbi_ell_t.item())
        return self._rhbi_ell_scalar

    @rhbi_ell_ema.setter
    def rhbi_ell_ema(self, value: float) -> None:
        self._rhbi_ell_scalar = float(value)
        self._rhbi_ell_t = None

    def record_step(self, optimizer_or_lr) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "rhbi": {
                "schema": 3,
                "ell_ema": float(self.rhbi_ell_ema),
                "ell_initialized": bool(self.rhbi_ell_initialized),
                "chord_phase": float(self.rhbi_chord_phase),
                "thermal_initialized": bool(self.rhbi_thermal_initialized),
                "ema_ed": float(self.rhbi_ema_ed),
                "ema_mu_sq": float(self.rhbi_ema_mu_sq),
                "ema_spread": float(self.rhbi_ema_spread),
                "ema_q90": float(self.rhbi_ema_q90),
                "ema_hmax": float(self.rhbi_ema_hmax),
                "ema_tail": float(self.rhbi_ema_tail),
                "thermal_scale": float(self._thermal_scale),
            },
        }

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "rhbi" not in state:
            raise ValueError("RHBI observer checkpoint is missing transport horizon state")
        if set(state) != {*CriticStepRecord.KEYS, "rhbi"}:
            raise ValueError("invalid RHBI observer record fields")
        observer = state["rhbi"]
        if (
            not isinstance(observer, dict)
            or set(observer) != _RHBIObserverStepRecord._RHBI_PACKET_KEYS
            or type(observer["schema"]) is not int
            or observer["schema"] != 3
            or type(observer["ell_initialized"]) is not bool
            or type(observer["thermal_initialized"]) is not bool
        ):
            raise ValueError("invalid RHBI observer metadata")
        for key in ("ell_ema", "chord_phase", "thermal_scale"):
            fval = observer[key]
            if isinstance(fval, bool) or not isinstance(fval, (int, float)) or not math.isfinite(fval) or fval < 0.0:
                raise ValueError(f"invalid RHBI observer {key}")
        for key in ("ema_ed", "ema_mu_sq", "ema_spread", "ema_q90", "ema_hmax", "ema_tail"):
            sval = observer[key]
            if isinstance(sval, bool) or not isinstance(sval, (int, float)) or not math.isfinite(sval):
                raise ValueError(f"invalid RHBI observer {key}")
        if float(observer["chord_phase"]) >= 1.0:
            raise ValueError("invalid RHBI observer chord_phase >= 1")
        if not (0.0 < float(observer["thermal_scale"]) <= 1.0):
            raise ValueError("invalid RHBI observer thermal_scale")
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        observer = state["rhbi"]
        self._rhbi_ell_scalar = float(observer["ell_ema"])
        self._rhbi_ell_t = None
        self.rhbi_ell_initialized = bool(observer["ell_initialized"])
        self.rhbi_chord_phase = float(observer["chord_phase"])
        self.rhbi_thermal_initialized = bool(observer["thermal_initialized"])
        self.rhbi_ema_ed = float(observer["ema_ed"])
        self.rhbi_ema_mu_sq = float(observer["ema_mu_sq"])
        self.rhbi_ema_spread = float(observer["ema_spread"])
        self.rhbi_ema_q90 = float(observer["ema_q90"])
        self.rhbi_ema_hmax = float(observer["ema_hmax"])
        self.rhbi_ema_tail = float(observer["ema_tail"])
        self._thermal_scale = float(observer["thermal_scale"])
        self._capture_ring = OrderedDict()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_RHBIObserverStepRecord":
        dup = _RHBIObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        dup._rhbi_ell_scalar = self._rhbi_ell_scalar
        dup._rhbi_ell_t = None if self._rhbi_ell_t is None else self._rhbi_ell_t.clone()
        dup.rhbi_ell_initialized = self.rhbi_ell_initialized
        dup.rhbi_chord_phase = self.rhbi_chord_phase
        dup.rhbi_thermal_initialized = self.rhbi_thermal_initialized
        dup.rhbi_ema_ed = self.rhbi_ema_ed
        dup.rhbi_ema_mu_sq = self.rhbi_ema_mu_sq
        dup.rhbi_ema_spread = self.rhbi_ema_spread
        dup.rhbi_ema_q90 = self.rhbi_ema_q90
        dup.rhbi_ema_hmax = self.rhbi_ema_hmax
        dup.rhbi_ema_tail = self.rhbi_ema_tail
        dup._thermal_scale = self._thermal_scale
        dup.legacy_thermal_channel = self.legacy_thermal_channel
        dup._in_penalty = self._in_penalty
        dup._in_protected_probe = self._in_protected_probe
        return dup


class RHBIStepRecord(_RHBIRecordThermalMixin, CriticStepRecord):
    """Per-critic RHBI state traveling with ``RHBICriticAdam.state_dict()``."""

    _COUNTERS = ("calls", "observed_steps")
    _OPTIONAL = ("lr_last",)
    KEYS = (
        "formulation",
        "lr_max",
        "lr_last",
        "anchor_started",
        "calls",
        "observed_steps",
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
        "thermal_scale",
    )

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._rhbi_ell_scalar: float = 0.0
        self._rhbi_ell_t: Optional[torch.Tensor] = None
        self.rhbi_ell_initialized: bool = False
        self._init_rhbi_v4_fields()
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._capture_ring: "OrderedDict[Tuple[int, int, int, int], Tuple[torch.Tensor, Optional[torch.Tensor], Any]]" = OrderedDict()

    @property
    def rhbi_ell_ema(self) -> float:
        if self._rhbi_ell_t is not None:
            return float(self._rhbi_ell_t.item())
        return self._rhbi_ell_scalar

    @rhbi_ell_ema.setter
    def rhbi_ell_ema(self, value: float) -> None:
        self._rhbi_ell_scalar = float(value)
        self._rhbi_ell_t = None

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "formulation": "rhbi",
            "lr_max": float(self.lr_max),
            "lr_last": None if self.lr_last is None else float(self.lr_last),
            "anchor_started": bool(self.anchor_started),
            "calls": int(self.calls),
            "observed_steps": int(self.observed_steps),
            "ell_ema": float(self.rhbi_ell_ema),
            "ell_initialized": bool(self.rhbi_ell_initialized),
            "chord_phase": float(self.rhbi_chord_phase),
            "thermal_initialized": bool(self.rhbi_thermal_initialized),
            "ema_ed": float(self.rhbi_ema_ed),
            "ema_mu_sq": float(self.rhbi_ema_mu_sq),
            "ema_spread": float(self.rhbi_ema_spread),
            "ema_q90": float(self.rhbi_ema_q90),
            "ema_hmax": float(self.rhbi_ema_hmax),
            "ema_tail": float(self.rhbi_ema_tail),
            "thermal_scale": float(self._thermal_scale),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "rhbi":
            raise ValueError("RHBI critic optimizer requires an RHBI controller checkpoint")
        if set(state) != set(self.KEYS):
            raise ValueError("invalid RHBI controller state keys")
        for key in self._COUNTERS:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid RHBI controller {key}")
        for key in ("lr_max", "ell_ema", "chord_phase", "thermal_scale", *self._OPTIONAL):
            value = state[key]
            if value is None and key in self._OPTIONAL:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid RHBI controller {key}")
        for key in ("ema_ed", "ema_mu_sq", "ema_spread", "ema_q90", "ema_hmax", "ema_tail"):
            sval = state[key]
            if isinstance(sval, bool) or not isinstance(sval, (float, int)) or not math.isfinite(sval):
                raise ValueError(f"invalid RHBI controller {key}")
        if float(state["chord_phase"]) >= 1.0:
            raise ValueError("invalid RHBI controller chord_phase >= 1")
        if not (0.0 < float(state["thermal_scale"]) <= 1.0):
            raise ValueError("invalid RHBI controller thermal_scale")
        if (
            type(state["anchor_started"]) is not bool
            or type(state["ell_initialized"]) is not bool
            or type(state["thermal_initialized"]) is not bool
        ):
            raise ValueError("invalid RHBI controller boolean flags")
        self.lr_max = float(state["lr_max"])
        self.lr_last = None if state["lr_last"] is None else float(state["lr_last"])
        self.anchor_started = bool(state["anchor_started"])
        self.calls = int(state["calls"])
        self.observed_steps = int(state["observed_steps"])
        self._rhbi_ell_scalar = float(state["ell_ema"])
        self._rhbi_ell_t = None
        self.rhbi_ell_initialized = bool(state["ell_initialized"])
        self.rhbi_chord_phase = float(state["chord_phase"])
        self.rhbi_thermal_initialized = bool(state["thermal_initialized"])
        self.rhbi_ema_ed = float(state["ema_ed"])
        self.rhbi_ema_mu_sq = float(state["ema_mu_sq"])
        self.rhbi_ema_spread = float(state["ema_spread"])
        self.rhbi_ema_q90 = float(state["ema_q90"])
        self.rhbi_ema_hmax = float(state["ema_hmax"])
        self.rhbi_ema_tail = float(state["ema_tail"])
        self._thermal_scale = float(state["thermal_scale"])
        self._capture_ring = OrderedDict()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "RHBIStepRecord":
        dup = RHBIStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        dup._rhbi_ell_scalar = self._rhbi_ell_scalar
        dup._rhbi_ell_t = None if self._rhbi_ell_t is None else self._rhbi_ell_t.clone()
        dup.rhbi_ell_initialized = self.rhbi_ell_initialized
        dup.rhbi_chord_phase = self.rhbi_chord_phase
        dup.rhbi_thermal_initialized = self.rhbi_thermal_initialized
        dup.rhbi_ema_ed = self.rhbi_ema_ed
        dup.rhbi_ema_mu_sq = self.rhbi_ema_mu_sq
        dup.rhbi_ema_spread = self.rhbi_ema_spread
        dup.rhbi_ema_q90 = self.rhbi_ema_q90
        dup.rhbi_ema_hmax = self.rhbi_ema_hmax
        dup.rhbi_ema_tail = self.rhbi_ema_tail
        dup._thermal_scale = self._thermal_scale
        dup.legacy_thermal_channel = self.legacy_thermal_channel
        dup._in_penalty = self._in_penalty
        dup._in_protected_probe = self._in_protected_probe
        return dup


def _rhbi_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    rec = getattr(module, "_rhbi_record", None)
    if rec is not None and getattr(rec, "_in_penalty", False):
        return None
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1:
        return None
    modified = False
    if rec is not None and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and target.requires_grad:
            scaled = target * s + target.detach() * (1.0 - s)
            scaled._rhbi_thermally_scaled = True
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
    x_in, cond_disc, has_cont_cond, float_parts = _extract_critic_inputs_parts(
        args, kwargs, int(target.shape[0])
    )
    if x_in is not None:
        _record_rhbi_capture(
            output,
            x_in,
            cond_disc,
            rec,
            has_cont_cond=has_cont_cond,
            float_parts=float_parts,
            overwrite=True,
        )
    return output if modified else None


def _attach_rhbi_critic_hook(critic: nn.Module, record: CriticStepRecord) -> None:
    """Attach a forward hook so ``RHBIGANLoss`` can read critic input coordinates."""
    if not hasattr(record, "rhbi_ell_initialized"):
        record.rhbi_ell_ema = 0.0
        record.rhbi_ell_initialized = False
    if not hasattr(record, "rhbi_chord_phase"):
        record.rhbi_chord_phase = 0.0
    if not hasattr(record, "rhbi_thermal_initialized"):
        record.rhbi_thermal_initialized = False
        record.rhbi_ema_ed = 1.0
        record.rhbi_ema_mu_sq = 1.0
        record.rhbi_ema_spread = 0.0
        record.rhbi_ema_q90 = 1.0
        record.rhbi_ema_hmax = 1.0
        record.rhbi_ema_tail = 0.0
        record._thermal_scale = 1.0
        record.legacy_thermal_channel = True
    if not hasattr(record, "rhbi_ema_tail"):
        record.rhbi_ema_tail = 0.0
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL.clear()
    if not hasattr(record, "_in_penalty"):
        record._in_penalty = False
    if not hasattr(record, "_in_protected_probe"):
        record._in_protected_probe = False
    if not hasattr(record, "_capture_ring"):
        record._capture_ring = OrderedDict()
    critic._rhbi_record = record
    if getattr(critic, "_rhbi_hook_registered", False):
        return
    critic.register_forward_hook(_rhbi_critic_forward_hook, with_kwargs=True)
    critic._rhbi_hook_registered = True


def bind_rhbi_record(record: Any, *consumers: Any) -> None:
    """Bind a critic step record to ``RHBIGANLoss``, optimizers, and thermal-aware regularizers."""
    if record is not None:
        record.legacy_thermal_channel = False
    for consumer in consumers:
        if consumer is None:
            continue
        if hasattr(consumer, "record"):
            consumer.record = record
        if hasattr(consumer, "thermal_source"):
            consumer.thermal_source = record
        if hasattr(consumer, "param_groups"):
            consumer._rhbi_record = record


class RHBICriticAdam(K3PCriticAdam):
    """Critic Adam optimizer for ``critic_formulation='rhbi'`` with no EMA critic copy."""

    def __init__(
        self,
        params,
        *,
        critic: nn.Module,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        record = RHBIStepRecord()
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
        _attach_rhbi_critic_hook(critic, record)

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        device = self.param_groups[0]["params"][0].device
        _publish_thermal(self.record, device, self.record._thermal_scale)
