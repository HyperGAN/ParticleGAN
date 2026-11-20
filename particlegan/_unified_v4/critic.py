"""Unified-v4 4-Law critic/generator step records, live endpoint + secant-probe hooks, and formulation Adam.

Unifies:
- Law 1 (`t1_coupling` + `unified_v4` conditional strata): persistent streaming chart atlas,
  6-channel split two-sample thermal state, ring-buffered `(x, cond_disc, float_parts)` capture,
  and checkpointed cross-batch per-stratum conditional sufficient statistics (`cond_strata`).
- Law 2 (`t2_secant`): live unscaled 1st-order endpoint capture (`_live_captures`) for zero-extra-forward
  image secant reuse (`0` `autograd.grad` calls).
- Law 3 (`rgdt`): state-driven equilibrium step damping (`step_scale`).
- Law 4 (`t4_arch` + `t4_nash`): detached `B_probe <= 16` function-space secant probe slices
  (`_d_probe_list`, `_g_probe_cache`) and two-player margin/scale calibration (`ema_margin`,
  `ema_data_scale`, `ema_push_res`) for zero-SVD impedance-matched optimization.
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

_MAX_D_PROBES = 2
_PROBE_SLICE = 16

_UNIFIED_V4_PACKET_KEYS = frozenset({
    *_T1C_PACKET_KEYS,
    "step_scale",
    "ema_margin",
    "ema_data_scale",
    "ema_push_res",
    "margin_initialized",
    "cond_strata",
})

_STRATUM_KEYS = frozenset({
    "weight",
    "sum_c",
    "sum_y",
    "sum_cc",
    "sum_cy",
    "sum_yy",
    "g_weight",
    "sum_rx",
    "sum_cwrx",
    "sum_rxrx",
})


def _slice_probe_arg(val: Any, b_full: int, b_probe: int) -> Any:
    if isinstance(val, torch.Tensor):
        if val.ndim >= 1 and val.shape[0] == b_full:
            return val.detach()[:b_probe]
        return val.detach()
    return val


def _validate_cond_strata(strata: Any) -> None:
    if not isinstance(strata, dict) or len(strata) > 64:
        raise ValueError("invalid unified_v4 cond_strata mapping")
    for k, entry in strata.items():
        if type(k) is not int:
            raise ValueError("unified_v4 cond_strata keys must be int stratum IDs")
        if not isinstance(entry, dict) or set(entry) != _STRATUM_KEYS:
            raise ValueError("invalid unified_v4 cond_strata entry keys")
        w = entry["weight"]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or float(w) <= 0.0:
            raise ValueError("invalid unified_v4 cond_strata weight")
        gw = entry["g_weight"]
        if isinstance(gw, bool) or not isinstance(gw, (int, float)) or not math.isfinite(gw) or float(gw) < 0.0:
            raise ValueError("invalid unified_v4 cond_strata g_weight")
        sc, sy = entry["sum_c"], entry["sum_y"]
        scc, scy, syy = entry["sum_cc"], entry["sum_cy"], entry["sum_yy"]
        srx, scwrx, srxrx = entry["sum_rx"], entry["sum_cwrx"], entry["sum_rxrx"]
        for name, t in (
            ("sum_c", sc),
            ("sum_y", sy),
            ("sum_cc", scc),
            ("sum_cy", scy),
            ("sum_yy", syy),
            ("sum_rx", srx),
            ("sum_cwrx", scwrx),
            ("sum_rxrx", srxrx),
        ):
            if not isinstance(t, torch.Tensor) or not t.is_floating_point() or not bool(torch.isfinite(t).all()):
                raise ValueError(f"invalid unified_v4 cond_strata tensor {name}")
        if sc.ndim != 1 or sy.ndim != 1 or sc.numel() < 1 or sy.numel() < 1:
            raise ValueError("invalid unified_v4 cond_strata vector shape")
        dc, dy = int(sc.shape[0]), int(sy.shape[0])
        if (
            tuple(scc.shape) != (dc, dc)
            or tuple(scy.shape) != (dc, dy)
            or tuple(syy.shape) != (dy, dy)
            or tuple(srx.shape) != (dy,)
            or tuple(scwrx.shape) != (dc, dy)
            or tuple(srxrx.shape) != (dy, dy)
        ):
            raise ValueError("mismatched unified_v4 cond_strata second-moment shapes")


def _validate_unified_v4_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _UNIFIED_V4_PACKET_KEYS:
        raise ValueError("invalid unified_v4 record packet fields")
    step_scale = packet["step_scale"]
    if (
        isinstance(step_scale, bool)
        or not isinstance(step_scale, (int, float))
        or not math.isfinite(step_scale)
        or float(step_scale) <= 0.0
        or float(step_scale) > 1.0
    ):
        raise ValueError("invalid unified_v4 step_scale outside (0, 1]")
    ema_margin = packet["ema_margin"]
    if isinstance(ema_margin, bool) or not isinstance(ema_margin, (int, float)) or not math.isfinite(ema_margin):
        raise ValueError("invalid unified_v4 ema_margin")
    ema_data_scale = packet["ema_data_scale"]
    if (
        isinstance(ema_data_scale, bool)
        or not isinstance(ema_data_scale, (int, float))
        or not math.isfinite(ema_data_scale)
        or float(ema_data_scale) <= 0.0
    ):
        raise ValueError("invalid unified_v4 ema_data_scale")
    ema_push_res = packet["ema_push_res"]
    if (
        isinstance(ema_push_res, bool)
        or not isinstance(ema_push_res, (int, float))
        or not math.isfinite(ema_push_res)
        or float(ema_push_res) < 0.0
    ):
        raise ValueError("invalid unified_v4 ema_push_res")
    if type(packet["margin_initialized"]) is not bool:
        raise ValueError("invalid unified_v4 margin_initialized")
    _validate_cond_strata(packet["cond_strata"])
    t1c_sub = {k: packet[k] for k in _T1C_PACKET_KEYS}
    _validate_t1c_packet(t1c_sub)
    return packet


class _UnifiedV4RecordMixin(_T1CouplingRecordMixin):
    """Unified-v4 critic record mixin combining 4-Law persistent and transient states."""

    def _init_unified_v4_fields(self) -> None:
        self._init_t1c_fields()
        self._step_scale: float = 1.0
        self._t1c_disc_gate: float = 0.0
        self._simplicial_step: bool = False
        self.ema_margin: float = 0.0
        self.ema_data_scale: float = 1.0
        self.ema_push_res: float = 1.0
        self.margin_initialized: bool = False
        self.cond_strata: Dict[int, Dict[str, Any]] = {}
        self._live_captures: List[List[Any]] = []
        self._last_probe_actual_input: Optional[torch.Tensor] = None
        self._d_probe_list: List[Tuple[Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = []
        self._g_probe_cache: Optional[Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = None
        self._in_probe: bool = False
        self._enable_secant_probe: bool = True

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

    @torch.no_grad()
    def observe_margin_and_scale(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        x_real: Optional[torch.Tensor] = None,
        x_fake: Optional[torch.Tensor] = None,
    ) -> None:
        """Update checkpointed two-player critic margin, data scale, and pushforward residual EMAs."""
        if getattr(self, "_in_penalty", False) or getattr(self, "_in_probe", False) or getattr(self, "_in_protected_probe", False):
            return
        r_m = float(real_logits.detach().float().mean().item())
        f_m = float(fake_logits.detach().float().mean().item())
        margin = max(-50.0, min(50.0, r_m - f_m))
        if math.isfinite(margin):
            if not self.margin_initialized:
                self.ema_margin = margin
                self.margin_initialized = True
            else:
                self.ema_margin = 0.90 * self.ema_margin + 0.10 * margin
        if x_real is not None and x_real.numel() > 0:
            xr = x_real.detach().reshape(x_real.shape[0], -1).float()
            sc = float(xr.square().sum(dim=-1).mean().sqrt().clamp_min(0.05).item())
            if math.isfinite(sc):
                self.ema_data_scale = sc if self.observed_steps == 0 else (0.95 * self.ema_data_scale + 0.05 * sc)
            if x_fake is not None and x_fake.shape == x_real.shape:
                xf = x_fake.detach().reshape(x_fake.shape[0], -1).float()
                mean_gap = float((xf.mean(dim=0) - xr.mean(dim=0)).norm().item())
                std_gap = abs(float(xf.std().item()) - float(xr.std().item()))
                rel_gap = min(10.0, (mean_gap + 0.5 * std_gap) / max(0.05, self.ema_data_scale))
                if math.isfinite(rel_gap):
                    self.ema_push_res = rel_gap if self.observed_steps == 0 else (0.92 * self.ema_push_res + 0.08 * rel_gap)

    @torch.no_grad()
    def observe_cond_stratum(
        self,
        stratum_id: int,
        c_mat: torch.Tensor,
        y_mat: torch.Tensor,
        *,
        gamma: float = 0.992,
    ) -> Dict[str, Any]:
        """Update persistent cross-batch sufficient statistics for discrete stratum ``stratum_id``."""
        c64 = c_mat.detach().to(dtype=torch.float64)
        y64 = y_mat.detach().to(dtype=torch.float64)
        n = float(c64.shape[0])
        sc = c64.sum(dim=0)
        sy = y64.sum(dim=0)
        scc = c64.T @ c64
        scy = c64.T @ y64
        syy = y64.T @ y64
        sid = int(stratum_id)
        cur = self.cond_strata.get(sid)
        dc_s, dy_s = int(sc.shape[0]), int(sy.shape[0])
        if (
            cur is None
            or cur["sum_c"].shape != sc.shape
            or cur["sum_y"].shape != sy.shape
        ):
            entry = {
                "weight": n,
                "sum_c": sc.clone(),
                "sum_y": sy.clone(),
                "sum_cc": scc.clone(),
                "sum_cy": scy.clone(),
                "sum_yy": syy.clone(),
                "g_weight": 0.0,
                "sum_rx": torch.zeros((dy_s,), dtype=torch.float64, device=c64.device),
                "sum_cwrx": torch.zeros((dc_s, dy_s), dtype=torch.float64, device=c64.device),
                "sum_rxrx": torch.zeros((dy_s, dy_s), dtype=torch.float64, device=c64.device),
            }
            self.cond_strata[sid] = entry
            return entry
        dev = c64.device
        if cur["sum_c"].device != dev:
            for k in ("sum_c", "sum_y", "sum_cc", "sum_cy", "sum_yy", "sum_rx", "sum_cwrx", "sum_rxrx"):
                cur[k] = cur[k].to(device=dev, dtype=torch.float64)
        cur["weight"] = float(gamma) * float(cur["weight"]) + n
        cur["sum_c"].mul_(gamma).add_(sc)
        cur["sum_y"].mul_(gamma).add_(sy)
        cur["sum_cc"].mul_(gamma).add_(scc)
        cur["sum_cy"].mul_(gamma).add_(scy)
        cur["sum_yy"].mul_(gamma).add_(syy)
        return cur

    @torch.no_grad()
    def observe_cond_gen_stratum(
        self,
        stratum_id: int,
        cw_mat: torch.Tensor,
        rx_mat: torch.Tensor,
        *,
        gamma: float = 0.90,
    ) -> Optional[Dict[str, Any]]:
        """Update cross-batch generator innovation BLP moments `(g_weight, sum_rx, sum_cwrx, sum_rxrx)`."""
        cur = self.cond_strata.get(int(stratum_id))
        if cur is None:
            return None
        cw64 = cw_mat.detach().to(dtype=torch.float64)
        rx64 = rx_mat.detach().to(dtype=torch.float64)
        if not (bool(torch.isfinite(cw64).all()) and bool(torch.isfinite(rx64).all())):
            return cur
        dev = rx64.device
        if cur["sum_rx"].device != dev:
            for k in ("sum_rx", "sum_cwrx", "sum_rxrx"):
                cur[k] = cur[k].to(device=dev, dtype=torch.float64)
        n = float(rx64.shape[0])
        srx = rx64.sum(dim=0)
        scwrx = cw64.T @ rx64
        srxrx = rx64.T @ rx64
        cur["g_weight"] = float(gamma) * float(cur["g_weight"]) + n
        cur["sum_rx"].mul_(gamma).add_(srx)
        cur["sum_cwrx"].mul_(gamma).add_(scwrx)
        cur["sum_rxrx"].mul_(gamma).add_(srxrx)
        return cur

    def clear_transient_captures(self) -> None:
        if hasattr(self, "_live_captures") and self._live_captures:
            self._live_captures.clear()
        self._last_probe_actual_input = None

    def clear_step_probes(self) -> None:
        if hasattr(self, "_d_probe_list") and self._d_probe_list:
            self._d_probe_list.clear()

    def restore_device_state(self, device: torch.device) -> None:
        super().restore_device_state(device)
        for entry in self.cond_strata.values():
            for k in ("sum_c", "sum_y", "sum_cc", "sum_cy", "sum_yy", "sum_rx", "sum_cwrx", "sum_rxrx"):
                entry[k] = entry[k].to(device=device, dtype=torch.float64)

    def _unified_v4_packet(self) -> Dict[str, Any]:
        strata_cpu: Dict[int, Dict[str, Any]] = {}
        for sid, entry in sorted(self.cond_strata.items()):
            strata_cpu[int(sid)] = {
                "weight": float(entry["weight"]),
                "sum_c": entry["sum_c"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "sum_y": entry["sum_y"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "sum_cc": entry["sum_cc"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "sum_cy": entry["sum_cy"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "sum_yy": entry["sum_yy"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "g_weight": float(entry["g_weight"]),
                "sum_rx": entry["sum_rx"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "sum_cwrx": entry["sum_cwrx"].detach().to(device="cpu", dtype=torch.float64).clone(),
                "sum_rxrx": entry["sum_rxrx"].detach().to(device="cpu", dtype=torch.float64).clone(),
            }
        return {
            **self._t1c_packet(),
            "step_scale": float(self._step_scale),
            "ema_margin": float(self.ema_margin),
            "ema_data_scale": float(self.ema_data_scale),
            "ema_push_res": float(self.ema_push_res),
            "margin_initialized": bool(self.margin_initialized),
            "cond_strata": strata_cpu,
        }

    def _restore_unified_v4_packet(self, packet: Dict[str, Any]) -> None:
        _validate_unified_v4_packet(packet)
        t1c_sub = {k: packet[k] for k in _T1C_PACKET_KEYS}
        self._restore_t1c_packet(t1c_sub)
        self._step_scale = float(packet["step_scale"])
        self.ema_margin = float(packet["ema_margin"])
        self.ema_data_scale = float(packet["ema_data_scale"])
        self.ema_push_res = float(packet["ema_push_res"])
        self.margin_initialized = bool(packet["margin_initialized"])
        self.cond_strata = {
            int(sid): {
                "weight": float(entry["weight"]),
                "sum_c": entry["sum_c"].detach().to(dtype=torch.float64).clone(),
                "sum_y": entry["sum_y"].detach().to(dtype=torch.float64).clone(),
                "sum_cc": entry["sum_cc"].detach().to(dtype=torch.float64).clone(),
                "sum_cy": entry["sum_cy"].detach().to(dtype=torch.float64).clone(),
                "sum_yy": entry["sum_yy"].detach().to(dtype=torch.float64).clone(),
                "g_weight": float(entry["g_weight"]),
                "sum_rx": entry["sum_rx"].detach().to(dtype=torch.float64).clone(),
                "sum_cwrx": entry["sum_cwrx"].detach().to(dtype=torch.float64).clone(),
                "sum_rxrx": entry["sum_rxrx"].detach().to(dtype=torch.float64).clone(),
            }
            for sid, entry in sorted(packet["cond_strata"].items())
        }
        self._t1c_disc_gate = 0.0
        self._simplicial_step = False
        self._live_captures = []
        self._d_probe_list = []
        self._g_probe_cache = None
        self._last_probe_actual_input = None
        self._in_probe = False

    def _copy_unified_v4_to(self, dup: Any) -> None:
        self._copy_t1c_to(dup)
        dup._step_scale = self._step_scale
        dup.ema_margin = self.ema_margin
        dup.ema_data_scale = self.ema_data_scale
        dup.ema_push_res = self.ema_push_res
        dup.margin_initialized = self.margin_initialized
        dup.cond_strata = {
            int(sid): {
                "weight": float(entry["weight"]),
                "sum_c": entry["sum_c"].detach().clone(),
                "sum_y": entry["sum_y"].detach().clone(),
                "sum_cc": entry["sum_cc"].detach().clone(),
                "sum_cy": entry["sum_cy"].detach().clone(),
                "sum_yy": entry["sum_yy"].detach().clone(),
                "g_weight": float(entry["g_weight"]),
                "sum_rx": entry["sum_rx"].detach().clone(),
                "sum_cwrx": entry["sum_cwrx"].detach().clone(),
                "sum_rxrx": entry["sum_rxrx"].detach().clone(),
            }
            for sid, entry in self.cond_strata.items()
        }
        dup._t1c_disc_gate = 0.0
        dup._simplicial_step = False
        dup._live_captures = []
        dup._d_probe_list = []
        dup._g_probe_cache = None
        dup._last_probe_actual_input = None
        dup._in_probe = False
        dup._enable_secant_probe = self._enable_secant_probe


class _UnifiedV4ObserverStepRecord(_UnifiedV4RecordMixin, CriticStepRecord):
    """Observer record for UnifiedV4EquilibratedOptimizer, DualNorm, and plain Adam critics."""

    def __init__(self) -> None:
        super().__init__()
        self._init_unified_v4_fields()

    def record_step(self, optimizer_or_lr: Any) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()
        self.clear_step_probes()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "unified_v4": self._unified_v4_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "unified_v4" not in state:
            raise ValueError("unified_v4 observer checkpoint is missing 'unified_v4' state")
        if set(state) != {*CriticStepRecord.KEYS, "unified_v4"}:
            raise ValueError("invalid unified_v4 observer record fields")
        _validate_unified_v4_packet(state["unified_v4"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_unified_v4_packet(state["unified_v4"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_UnifiedV4ObserverStepRecord":
        dup = _UnifiedV4ObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_v4_to(dup)
        return dup


class UnifiedV4StepRecord(_UnifiedV4RecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``UnifiedV4CriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "unified_v4")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_unified_v4_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()
        self.clear_step_probes()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "unified_v4", "unified_v4": self._unified_v4_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid UnifiedV4StepRecord checkpoint keys")
        if state.get("formulation") != "unified_v4":
            raise ValueError("checkpoint formulation is not 'unified_v4'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid UnifiedV4StepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("UnifiedV4StepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            val = state[key]
            if key == "lr_last" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid UnifiedV4StepRecord {key}")
        _validate_unified_v4_packet(state["unified_v4"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_unified_v4_packet(state["unified_v4"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "UnifiedV4StepRecord":
        dup = UnifiedV4StepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_v4_to(dup)
        return dup


def _unified_v4_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-bound critic forward hook capturing live endpoints, secant probes, chart inputs, and thermal cooling."""
    rec = getattr(module, "_unified_v4_record", getattr(module, "_t1c_record", None))
    if rec is not None and getattr(rec, "_in_probe", False):
        return None
    x_first = args[0] if (args and isinstance(args[0], torch.Tensor)) else None

    if rec is not None and getattr(rec, "_in_penalty", False):
        if x_first is not None:
            rec._last_probe_actual_input = x_first.detach()
        return None

    raw_target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(raw_target, torch.Tensor) or raw_target.ndim < 1:
        return None

    # 1. Capture unscaled 1st-order live endpoint logits + detached B_probe<=16 critic secant probe:
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

        if (
            getattr(rec, "_enable_secant_probe", True)
            and x_first.ndim == 2
            and x_first.shape[0] >= 1
            and x_first.shape[1] <= 32
            and len(args) == 1
            and not kwargs
        ):
            b_full = int(x_first.shape[0])
            b_p = min(_PROBE_SLICE, b_full)
            p_args = tuple(_slice_probe_arg(a, b_full, b_p) for a in args)
            d_pre = raw_target.detach()[:b_p].to(dtype=torch.float32)
            probes = getattr(rec, "_d_probe_list", None)
            if probes is None:
                probes = []
                rec._d_probe_list = probes
            probes.append((p_args, {}, d_pre))
            if len(probes) > _MAX_D_PROBES:
                del probes[: len(probes) - _MAX_D_PROBES]

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

    # 3. Record (x_in, cond_disc, has_cont_cond, float_parts) for UnifiedV4GANLoss
    #    (skip unconditional high-resolution 4D image tensors where UnifiedV4GANLoss uses raw logits directly):
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


def _unified_v4_generator_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    """Capture a detached ``B_probe <= 16`` generator slice for function-space secant trust-band regulation."""
    rec = getattr(module, "_unified_v4_record", None)
    if (
        rec is None
        or getattr(rec, "_in_probe", False)
        or getattr(rec, "_in_penalty", False)
        or getattr(rec, "_in_protected_probe", False)
        or not getattr(rec, "_enable_secant_probe", True)
        or not module.training
        or not torch.is_grad_enabled()
        or len(args) != 1
        or bool(kwargs)
    ):
        return
    raw_out = output[0] if isinstance(output, (tuple, list)) and output else output
    if (
        not isinstance(raw_out, torch.Tensor)
        or not raw_out.requires_grad
        or raw_out.ndim != 2
        or raw_out.shape[0] < 1
        or raw_out.shape[1] > 32
    ):
        return
    # Skip dual-tangent forward passes inside fwAD.dual_level():
    import torch.autograd.forward_ad as fwAD
    if fwAD.unpack_dual(raw_out).tangent is not None:
        return
    b_full = int(raw_out.shape[0])
    b_p = min(_PROBE_SLICE, b_full)
    p_args = tuple(_slice_probe_arg(a, b_full, b_p) for a in args)
    p_kwargs = {k: _slice_probe_arg(v, b_full, b_p) for k, v in kwargs.items()}
    rec._g_probe_cache = (module, p_args, p_kwargs, raw_out.detach()[:b_p].to(dtype=torch.float32))


def _attach_unified_v4_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local unified_v4 forward hook to ``critic`` (zero global state)."""
    if not hasattr(record, "_in_penalty"):
        record._in_penalty = False
    if not hasattr(record, "_in_protected_probe"):
        record._in_protected_probe = False
    if not hasattr(record, "_in_probe"):
        record._in_probe = False
    if not hasattr(record, "_capture_ring"):
        record._capture_ring = OrderedDict()
    if not hasattr(record, "_live_captures"):
        record._live_captures = []
    if not hasattr(record, "_d_probe_list"):
        record._d_probe_list = []
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL.clear()
    if not isinstance(critic, nn.Module):
        return
    critic._unified_v4_record = record
    critic._unified_record = record
    critic._t1c_record = record
    critic._t2_record = record
    if getattr(critic, "_unified_v4_hook_registered", False):
        return
    critic.register_forward_hook(_unified_v4_critic_forward_hook, with_kwargs=True)
    critic._unified_v4_hook_registered = True


def _attach_unified_v4_generator_hook(generator: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local unified_v4 secant probe hook to ``generator``."""
    if not isinstance(generator, nn.Module):
        return
    generator._unified_v4_record = record
    if getattr(generator, "_unified_v4_gen_hook_registered", False):
        return
    generator.register_forward_hook(_unified_v4_generator_forward_hook, with_kwargs=True)
    generator._unified_v4_gen_hook_registered = True


def bind_unified_v4_record(record: Any, *consumers: Any) -> None:
    """Bind instance-local ``unified_v4`` critic record to losses, optimizers, and priors."""
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
        if hasattr(consumer, "unified_v4_record"):
            consumer.unified_v4_record = record


class UnifiedV4CriticAdam(K3PCriticAdam):
    """Formulation critic Adam for ``critic_formulation='unified_v4'`` (zero EMA critic copy)."""

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
        del ema_critic
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = UnifiedV4StepRecord()
        self._t1c_record = self.record
        self._t2_record = self.record
        self._unified_record = self.record
        self._unified_v4_record = self.record
        _attach_unified_v4_critic_hook(critic, self.record)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "UnifiedV4CriticAdam",
    "UnifiedV4StepRecord",
    "_UnifiedV4ObserverStepRecord",
    "_attach_unified_v4_critic_hook",
    "_attach_unified_v4_generator_hook",
    "_cond_signature",
    "_extract_t1c_inputs_parts",
    "_lookup_t1c_capture_parts",
    "_record_t1c_capture",
    "_tensor_identity_key",
    "_validate_unified_v4_packet",
    "bind_unified_v4_record",
]
