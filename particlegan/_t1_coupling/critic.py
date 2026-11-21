"""Track 1B critic state: step records, input/conditioning capture hook, and critic Adam."""
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
_T1C_RECENT_CAPTURES: "OrderedDict[Tuple[int, int, int, int], Tuple[Any, ...]]" = OrderedDict()

_T1C_PACKET_KEYS = frozenset({
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
    "het_initialized",
    "het_ema",
    "thermal_scale",
    "ema_data_scale",
    "revival_phase",
    "revivals",
    "chart_initialized",
    "chart_sum",
    "chart_count",
    "chart_sq_dev",
    "chart_fake_mass",
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


def _extract_t1c_inputs_parts(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool, Optional[Tuple[torch.Tensor, ...]]]:
    """Extract raw primary input tensor, discrete conditions, continuous-fibre flag, and float parts."""
    if batch_size <= 0:
        return None, None, False, None
    raw_float_items = []
    flat_float_parts = []
    disc_parts = []
    items = list(args)
    if kwargs:
        for key in sorted(kwargs):
            items.append(kwargs[key])
    for item in items:
        if not isinstance(item, torch.Tensor) or item.ndim < 1 or item.shape[0] != batch_size or item.numel() == 0:
            continue
        if item.is_floating_point():
            raw_float_items.append(item)
            flat_float_parts.append(item.reshape(batch_size, -1))
        elif not item.is_complex():
            disc_parts.append(item.detach().to(dtype=torch.int64).reshape(batch_size, -1))
    if not flat_float_parts:
        return None, None, False, None
    has_cont_cond = len(flat_float_parts) > 1
    # Preserve spatial rank on single-input image critics so project_chart_features can pool high-res tensors.
    x_in = raw_float_items[0] if not has_cont_cond else torch.cat(flat_float_parts, dim=-1)
    cond_disc = None if not disc_parts else (
        disc_parts[0] if len(disc_parts) == 1 else torch.cat(disc_parts, dim=-1)
    )
    parts_tuple = tuple(flat_float_parts) if has_cont_cond else None
    return x_in, cond_disc, has_cont_cond, parts_tuple


def _record_t1c_capture(
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
    if overwrite or not hasattr(target, "_t1c_input"):
        target._t1c_input = x_input
        target._t1c_cond = cond_disc
        target._t1c_cont_cond = bool(has_cont_cond)
        target._t1c_parts = float_parts
    if record is not None and (overwrite or not hasattr(target, "_t1c_record")):
        target._t1c_record = record
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
    if overwrite or key not in _T1C_RECENT_CAPTURES:
        _T1C_RECENT_CAPTURES[key] = entry
        _T1C_RECENT_CAPTURES.move_to_end(key)
        while len(_T1C_RECENT_CAPTURES) > _MAX_RING_ENTRIES:
            _T1C_RECENT_CAPTURES.popitem(last=False)


def _finalize_x(logits: torch.Tensor, x_val: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x_val is None:
        return None
    if not logits.requires_grad and x_val.requires_grad:
        return x_val.detach()
    return x_val


def _finalize_parts(
    logits: torch.Tensor,
    parts: Optional[Tuple[torch.Tensor, ...]],
) -> Optional[Tuple[torch.Tensor, ...]]:
    if not parts:
        return None
    return tuple(_finalize_x(logits, p) for p in parts)  # type: ignore[misc]


def _lookup_t1c_capture_parts(
    logits: torch.Tensor,
    fallback_record: Any = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any, bool, Optional[Tuple[torch.Tensor, ...]]]:
    if getattr(logits, "_t1c_bypassed", False):
        r_attr = getattr(logits, "_t1c_record", None)
        return None, None, (r_attr if r_attr is not None else fallback_record), False, None
    b = logits.shape[0] if logits.ndim >= 1 else 0
    x_attr = getattr(logits, "_t1c_input", None)
    c_attr = getattr(logits, "_t1c_cond", None)
    r_attr = getattr(logits, "_t1c_record", None)
    cc_attr = bool(getattr(logits, "_t1c_cont_cond", False))
    p_attr = getattr(logits, "_t1c_parts", None)
    if isinstance(x_attr, torch.Tensor) and x_attr.shape[0] == b:
        return (
            _finalize_x(logits, x_attr),
            c_attr,
            (r_attr if r_attr is not None else fallback_record),
            cc_attr,
            _finalize_parts(logits, p_attr),
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
                        _finalize_x(logits, x_c),
                        c_c,
                        (r_c if r_c is not None else fallback_record),
                        cc_c,
                        _finalize_parts(logits, p_c),
                    )
        if key in _T1C_RECENT_CAPTURES:
            item = _T1C_RECENT_CAPTURES[key]
            x_c, c_c, r_c, cc_c, p_c = item[0], item[1], item[2], bool(item[3]), item[4]
            if x_c.shape[0] == b:
                return (
                    _finalize_x(logits, x_c),
                    c_c,
                    (r_c if r_c is not None else fallback_record),
                    cc_c,
                    _finalize_parts(logits, p_c),
                )
    return None, None, (r_attr if r_attr is not None else fallback_record), False, None


def _validate_t1c_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _T1C_PACKET_KEYS:
        raise ValueError("invalid t1_coupling record packet fields")
    if type(packet["schema"]) is not int or packet["schema"] != 1:
        raise ValueError("unsupported t1_coupling record schema")
    for bkey in ("ell_initialized", "thermal_initialized", "cond_initialized", "het_initialized", "chart_initialized"):
        if type(packet[bkey]) is not bool:
            raise ValueError(f"invalid t1_coupling boolean field {bkey}")
    if type(packet["revivals"]) is not int or packet["revivals"] < 0:
        raise ValueError("invalid t1_coupling revival count")
    for nkey in ("ell_ema", "chord_phase", "thermal_scale", "ema_data_scale", "revival_phase"):
        val = packet[nkey]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
            raise ValueError(f"invalid t1_coupling nonnegative scalar {nkey}")
    for skey in ("ema_ed", "ema_mu_sq", "ema_spread", "ema_q90", "ema_hmax", "ema_tail", "cond_ema", "het_ema"):
        val = packet[skey]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"invalid t1_coupling scalar {skey}")
    if float(packet["chord_phase"]) >= 1.0 or float(packet["revival_phase"]) >= 1.0:
        raise ValueError("invalid t1_coupling phase >= 1")
    if not (0.0 < float(packet["thermal_scale"]) <= 1.0):
        raise ValueError("invalid t1_coupling thermal_scale")

    c_init = packet["chart_initialized"]
    c_sum = packet["chart_sum"]
    c_cnt = packet["chart_count"]
    c_sq = packet["chart_sq_dev"]
    c_fm = packet["chart_fake_mass"]
    if not c_init:
        if c_sum is not None or c_cnt is not None or c_sq is not None or c_fm is not None:
            raise ValueError("uninitialized t1_coupling chart bank must have None tensors")
    else:
        if (
            not isinstance(c_sum, torch.Tensor)
            or not isinstance(c_cnt, torch.Tensor)
            or not isinstance(c_sq, torch.Tensor)
            or not isinstance(c_fm, torch.Tensor)
        ):
            raise ValueError("initialized t1_coupling chart bank requires float32 tensors")
        if (
            c_sum.dtype != torch.float32
            or c_cnt.dtype != torch.float32
            or c_sq.dtype != torch.float32
            or c_fm.dtype != torch.float32
        ):
            raise ValueError("t1_coupling chart bank tensors must have float32 dtype")
        if c_sum.ndim != 2 or c_cnt.ndim != 1 or c_sq.ndim != 1 or c_fm.ndim != 1:
            raise ValueError("invalid t1_coupling chart bank tensor rank")
        if (
            c_sum.shape[0] == 0
            or c_sum.shape[1] == 0
            or c_sum.shape[0] != c_cnt.shape[0]
            or c_sum.shape[0] != c_sq.shape[0]
            or c_sum.shape[0] != c_fm.shape[0]
        ):
            raise ValueError("incompatible t1_coupling chart bank shapes")
        if not (
            bool(torch.isfinite(c_sum).all())
            and bool(torch.isfinite(c_cnt).all())
            and bool(torch.isfinite(c_sq).all())
            and bool(torch.isfinite(c_fm).all())
        ):
            raise ValueError("nonfinite t1_coupling chart bank values")
        if bool((c_cnt <= 0).any()) or bool((c_sq <= 0).any()) or bool((c_fm < 0).any()):
            raise ValueError("t1_coupling chart counts/variances must be positive and fake_mass nonnegative")
    return packet


class _T1CouplingRecordMixin:
    """Shared instance-local horizon, streaming chart atlas, and thermal settling state."""

    def _init_t1c_fields(self) -> None:
        self._t1c_ell_scalar: float = 0.0
        self._t1c_ell_t: Optional[torch.Tensor] = None
        self.t1c_ell_initialized: bool = False
        self.t1c_chord_phase: float = 0.0
        self.t1c_thermal_initialized: bool = False
        self.t1c_ema_ed: float = 1.0
        self.t1c_ema_mu_sq: float = 1.0
        self.t1c_ema_spread: float = 0.0
        self.t1c_ema_q90: float = 1.0
        self.t1c_ema_hmax: float = 1.0
        self.t1c_ema_tail: float = 0.0
        self.t1c_cond_initialized: bool = False
        self.t1c_cond_ema: float = 1.0
        self.t1c_het_initialized: bool = False
        self.t1c_het_ema: float = 1.0
        self._thermal_scale: float = 1.0
        self.ema_data_scale: float = 1.0
        self.revival_phase: float = 0.0
        self.revivals: int = 0
        self.chart_initialized: bool = False
        self.chart_sum: Optional[torch.Tensor] = None
        self.chart_count: Optional[torch.Tensor] = None
        self.chart_sq_dev: Optional[torch.Tensor] = None
        self.chart_fake_mass: Optional[torch.Tensor] = None
        self.legacy_thermal_channel: bool = True
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._capture_ring: OrderedDict = OrderedDict()

    @property
    def t1c_ell_ema(self) -> float:
        if self._t1c_ell_t is not None:
            return float(self._t1c_ell_t.item())
        return self._t1c_ell_scalar

    @t1c_ell_ema.setter
    def t1c_ell_ema(self, value: float) -> None:
        self._t1c_ell_scalar = float(value)
        self._t1c_ell_t = None

    @property
    def ema_residual(self) -> float:
        q90 = float(self.t1c_ema_q90)
        hmax = float(self.t1c_ema_hmax)
        spread = float(self.t1c_ema_spread)
        underdisp_gate = min(1.0, max(0.0, -spread) / 0.03)
        return max(
            0.5 * max(0.0, float(self.t1c_ema_ed)),
            math.sqrt(max(0.0, float(self.t1c_ema_mu_sq))),
            0.60 * abs(spread),
            max(q90, 0.60 * max(0.0, -q90) * underdisp_gate),
            max(hmax, 0.60 * max(0.0, -hmax) * underdisp_gate),
            max(0.0, float(self.t1c_ema_tail)),
        )

    def update_charts(
        self,
        chart_sum: torch.Tensor,
        chart_count: torch.Tensor,
        chart_sq_dev: torch.Tensor,
        chart_fake_mass: torch.Tensor,
        revival_phase: float,
        revivals: int,
        data_scale: float,
    ) -> None:
        self.chart_initialized = True
        self.chart_sum = chart_sum.detach().clone().to(dtype=torch.float32)
        self.chart_count = chart_count.detach().clone().to(dtype=torch.float32)
        self.chart_sq_dev = chart_sq_dev.detach().clone().to(dtype=torch.float32)
        self.chart_fake_mass = chart_fake_mass.detach().clone().to(dtype=torch.float32)
        self.revival_phase = float(revival_phase)
        self.revivals = int(revivals)
        if math.isfinite(data_scale) and data_scale > 0.0:
            self.ema_data_scale = 0.92 * self.ema_data_scale + 0.08 * float(data_scale)

    def observe_t1c_thermal(
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
        shock_ed_floor: float = 0.30,
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
        if not self.t1c_thermal_initialized or (
            self._thermal_scale < 0.99 and (ed_f > shock_ed_thresh or mu_sq_f > shock_mu_thresh)
        ):
            self.t1c_ema_ed = ed_c
            self.t1c_ema_mu_sq = mu_sq_c
            self.t1c_ema_spread = spread_c
            self.t1c_ema_q90 = q90_c
            self.t1c_ema_hmax = hmax_c
            self.t1c_ema_tail = tail_c
            self.t1c_thermal_initialized = True
        else:
            self.t1c_ema_ed = ema_decay * self.t1c_ema_ed + (1.0 - ema_decay) * ed_c
            self.t1c_ema_mu_sq = ema_decay * self.t1c_ema_mu_sq + (1.0 - ema_decay) * mu_sq_c
            self.t1c_ema_spread = ema_decay * self.t1c_ema_spread + (1.0 - ema_decay) * spread_c
            self.t1c_ema_q90 = ema_decay * self.t1c_ema_q90 + (1.0 - ema_decay) * q90_c
            self.t1c_ema_hmax = ema_decay * self.t1c_ema_hmax + (1.0 - ema_decay) * hmax_c
            self.t1c_ema_tail = ema_decay * self.t1c_ema_tail + (1.0 - ema_decay) * tail_c
        if tau_f <= 0.0:
            self._thermal_scale = 1.0
            return 1.0
        lo_frac = 0.50 + (float(lock_hysteresis) if self._thermal_scale <= 1e-4 else 0.0)
        tau_lo = min(0.92, lo_frac) * tau_f
        span = max(1e-8, tau_f - tau_lo)
        u = min(1.0, max(0.0, (self.ema_residual - tau_lo) / span))
        theta = u * u * (3.0 - 2.0 * u)
        if theta >= 1.0:
            s = 1.0
        elif theta <= 0.0:
            s = float(s_min)
        else:
            s = max(float(s_min), min(1.0, 0.04 * theta / math.sqrt(max(1e-8, 1.0 - theta * theta))))
        self._thermal_scale = max(float(s_min), min(1.0, s))
        return self._thermal_scale

    def _t1c_packet(self) -> Dict[str, Any]:
        return {
            "schema": 1,
            "ell_ema": float(self.t1c_ell_ema),
            "ell_initialized": bool(self.t1c_ell_initialized),
            "chord_phase": float(self.t1c_chord_phase),
            "thermal_initialized": bool(self.t1c_thermal_initialized),
            "ema_ed": float(self.t1c_ema_ed),
            "ema_mu_sq": float(self.t1c_ema_mu_sq),
            "ema_spread": float(self.t1c_ema_spread),
            "ema_q90": float(self.t1c_ema_q90),
            "ema_hmax": float(self.t1c_ema_hmax),
            "ema_tail": float(self.t1c_ema_tail),
            "cond_initialized": bool(self.t1c_cond_initialized),
            "cond_ema": float(self.t1c_cond_ema),
            "het_initialized": bool(self.t1c_het_initialized),
            "het_ema": float(self.t1c_het_ema),
            "thermal_scale": float(self._thermal_scale),
            "ema_data_scale": float(self.ema_data_scale),
            "revival_phase": float(self.revival_phase),
            "revivals": int(self.revivals),
            "chart_initialized": bool(self.chart_initialized),
            "chart_sum": None if self.chart_sum is None else self.chart_sum.detach().clone(),
            "chart_count": None if self.chart_count is None else self.chart_count.detach().clone(),
            "chart_sq_dev": None if self.chart_sq_dev is None else self.chart_sq_dev.detach().clone(),
            "chart_fake_mass": None if self.chart_fake_mass is None else self.chart_fake_mass.detach().clone(),
        }

    def _restore_t1c_packet(self, packet: Dict[str, Any]) -> None:
        _validate_t1c_packet(packet)
        self._t1c_ell_scalar = float(packet["ell_ema"])
        self._t1c_ell_t = None
        self.t1c_ell_initialized = bool(packet["ell_initialized"])
        self.t1c_chord_phase = float(packet["chord_phase"])
        self.t1c_thermal_initialized = bool(packet["thermal_initialized"])
        self.t1c_ema_ed = float(packet["ema_ed"])
        self.t1c_ema_mu_sq = float(packet["ema_mu_sq"])
        self.t1c_ema_spread = float(packet["ema_spread"])
        self.t1c_ema_q90 = float(packet["ema_q90"])
        self.t1c_ema_hmax = float(packet["ema_hmax"])
        self.t1c_ema_tail = float(packet["ema_tail"])
        self.t1c_cond_initialized = bool(packet["cond_initialized"])
        self.t1c_cond_ema = float(packet["cond_ema"])
        self.t1c_het_initialized = bool(packet["het_initialized"])
        self.t1c_het_ema = float(packet["het_ema"])
        self._thermal_scale = float(packet["thermal_scale"])
        self.ema_data_scale = float(packet["ema_data_scale"])
        self.revival_phase = float(packet["revival_phase"])
        self.revivals = int(packet["revivals"])
        self.chart_initialized = bool(packet["chart_initialized"])
        self.chart_sum = None if packet["chart_sum"] is None else packet["chart_sum"].detach().clone()
        self.chart_count = None if packet["chart_count"] is None else packet["chart_count"].detach().clone()
        self.chart_sq_dev = None if packet["chart_sq_dev"] is None else packet["chart_sq_dev"].detach().clone()
        self.chart_fake_mass = None if packet["chart_fake_mass"] is None else packet["chart_fake_mass"].detach().clone()
        self._capture_ring = OrderedDict()

    def restore_device_state(self, device: torch.device) -> None:
        if self.chart_sum is not None:
            self.chart_sum = self.chart_sum.to(device=device)
        if self.chart_count is not None:
            self.chart_count = self.chart_count.to(device=device)
        if self.chart_sq_dev is not None:
            self.chart_sq_dev = self.chart_sq_dev.to(device=device)
        if self.chart_fake_mass is not None:
            self.chart_fake_mass = self.chart_fake_mass.to(device=device)
        _publish_thermal(self, device, self._thermal_scale)

    def _copy_t1c_to(self, dup: Any) -> None:
        dup._t1c_ell_scalar = self._t1c_ell_scalar
        dup._t1c_ell_t = None if self._t1c_ell_t is None else self._t1c_ell_t.clone()
        dup.t1c_ell_initialized = self.t1c_ell_initialized
        dup.t1c_chord_phase = self.t1c_chord_phase
        dup.t1c_thermal_initialized = self.t1c_thermal_initialized
        dup.t1c_ema_ed = self.t1c_ema_ed
        dup.t1c_ema_mu_sq = self.t1c_ema_mu_sq
        dup.t1c_ema_spread = self.t1c_ema_spread
        dup.t1c_ema_q90 = self.t1c_ema_q90
        dup.t1c_ema_hmax = self.t1c_ema_hmax
        dup.t1c_ema_tail = self.t1c_ema_tail
        dup.t1c_cond_initialized = self.t1c_cond_initialized
        dup.t1c_cond_ema = self.t1c_cond_ema
        dup.t1c_het_initialized = self.t1c_het_initialized
        dup.t1c_het_ema = self.t1c_het_ema
        dup._thermal_scale = self._thermal_scale
        dup.ema_data_scale = self.ema_data_scale
        dup.revival_phase = self.revival_phase
        dup.revivals = self.revivals
        dup.chart_initialized = self.chart_initialized
        dup.chart_sum = None if self.chart_sum is None else self.chart_sum.detach().clone()
        dup.chart_count = None if self.chart_count is None else self.chart_count.detach().clone()
        dup.chart_sq_dev = None if self.chart_sq_dev is None else self.chart_sq_dev.detach().clone()
        dup.chart_fake_mass = None if self.chart_fake_mass is None else self.chart_fake_mass.detach().clone()
        dup.legacy_thermal_channel = self.legacy_thermal_channel
        dup._in_penalty = self._in_penalty
        dup._in_protected_probe = self._in_protected_probe


class _T1CouplingObserverStepRecord(_T1CouplingRecordMixin, CriticStepRecord):
    """Observer record for DualNorm and plain Adam critics under ``t1_coupling``."""

    def __init__(self) -> None:
        super().__init__()
        self._init_t1c_fields()

    def record_step(self, optimizer_or_lr) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "t1_coupling": self._t1c_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "t1_coupling" not in state:
            raise ValueError("t1_coupling observer checkpoint is missing 't1_coupling' state")
        if set(state) != {*CriticStepRecord.KEYS, "t1_coupling"}:
            raise ValueError("invalid t1_coupling observer record fields")
        _validate_t1c_packet(state["t1_coupling"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_t1c_packet(state["t1_coupling"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_T1CouplingObserverStepRecord":
        dup = _T1CouplingObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_t1c_to(dup)
        return dup


class T1CouplingStepRecord(_T1CouplingRecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``T1CouplingCriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "t1_coupling")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_t1c_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "t1_coupling", "t1_coupling": self._t1c_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid T1CouplingStepRecord checkpoint keys")
        if state.get("formulation") != "t1_coupling":
            raise ValueError("checkpoint formulation is not 't1_coupling'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid T1CouplingStepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("T1CouplingStepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            val = state[key]
            if key == "lr_last" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid T1CouplingStepRecord {key}")
        _validate_t1c_packet(state["t1_coupling"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_t1c_packet(state["t1_coupling"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "T1CouplingStepRecord":
        dup = T1CouplingStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_t1c_to(dup)
        return dup


def _t1c_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    rec = getattr(module, "_t1c_record", None)
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


def _attach_t1_coupling_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local forward capture & thermal hook to ``critic``."""
    if not hasattr(record, "_in_penalty"):
        record._in_penalty = False
    if not hasattr(record, "_in_protected_probe"):
        record._in_protected_probe = False
    if not hasattr(record, "_capture_ring"):
        record._capture_ring = OrderedDict()
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL.clear()
    if not isinstance(critic, nn.Module):
        return
    critic._t1c_record = record
    if getattr(critic, "_t1c_hook_registered", False):
        return
    critic.register_forward_hook(_t1c_critic_forward_hook, with_kwargs=True)
    critic._t1c_hook_registered = True


def bind_t1_coupling_record(record: Any, *consumers: Any) -> None:
    """Bind instance-local ``t1_coupling`` critic record to losses, optimizers, and priors."""
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
        if hasattr(consumer, "param_groups"):
            consumer._t1c_record = record


class T1CouplingCriticAdam(K3PCriticAdam):
    """Critic Adam optimizer for ``critic_formulation='t1_coupling'`` (zero EMA critic copy)."""

    def __init__(
        self,
        params,
        *,
        critic: nn.Module | None = None,
        ema_critic=None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        del ema_critic
        record = T1CouplingStepRecord()
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
        _attach_t1_coupling_critic_hook(critic, record)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)
