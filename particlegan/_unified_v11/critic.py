"""Unified-v11 4-Law + CondFiber critic/generator step records, live endpoint + secant-probe hooks, and Adam."""
from __future__ import annotations

from collections import OrderedDict
import math
import random
import sys
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
    _clear_live_capture_entries,
    _cond_signature,
    _tensor_identity_key,
    _trim_live_captures,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam
from .bank import UnifiedV11CondFiberBank, validate_v11_bank_packet

_MAX_D_PROBES = 2
_PROBE_SLICE = 16

_UNIFIED_V11_PACKET_KEYS = frozenset({
    *_T1C_PACKET_KEYS,
    "step_scale",
    "ema_margin",
    "ema_data_scale",
    "ema_push_res",
    "margin_initialized",
    "cond_fiber_bank",
})


def _slice_probe_arg(val: Any, b_full: int, b_probe: int) -> Any:
    if isinstance(val, torch.Tensor):
        if val.ndim >= 1 and val.shape[0] == b_full:
            return val.detach()[:b_probe]
        return val.detach()
    if isinstance(val, dict):
        return {k: _slice_probe_arg(v, b_full, b_probe) for k, v in val.items()}
    if isinstance(val, tuple):
        return tuple(_slice_probe_arg(v, b_full, b_probe) for v in val)
    if isinstance(val, list):
        return [_slice_probe_arg(v, b_full, b_probe) for v in val]
    return val


@torch.no_grad()
def _eval_v11_probe_forward(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    record: Any,
) -> Optional[torch.Tensor]:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` mode without mutating RNGs, hooks, or BN buffers."""
    modes = [(m, m.training) for m in module.modules()]
    prev_probe = bool(getattr(record, "_in_probe", False)) if record is not None else False
    cpu_rng = torch.get_rng_state()
    py_rng = random.getstate()
    np_mod = sys.modules.get("numpy")
    np_rng = np_mod.random.get_state() if np_mod is not None and hasattr(np_mod, "random") else None
    cuda_rng = None
    cuda_dev = None
    if torch.cuda.is_initialized():
        cuda_dev = torch.cuda.current_device()
        cuda_rng = torch.cuda.get_rng_state(cuda_dev)
    if record is not None:
        record._in_probe = True
    try:
        module.eval()
        first_p = next(module.parameters(), None)
        use_bf16 = (
            first_p is not None
            and first_p.device.type == "cuda"
            and bool(getattr(module, "_unified_v4_bf16_autocast", False))
        )
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = module(*args, **kwargs)
        else:
            out = module(*args, **kwargs)
        target = out[0] if isinstance(out, (tuple, list)) and out else out
        if not isinstance(target, torch.Tensor):
            return None
        return target.detach().to(dtype=torch.float32)
    finally:
        if record is not None:
            record._in_probe = prev_probe
        for m, flag in modes:
            m.training = flag
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None and cuda_dev is not None:
            torch.cuda.set_rng_state(cuda_rng, cuda_dev)
        random.setstate(py_rng)
        if np_rng is not None and np_mod is not None:
            np_mod.random.set_state(np_rng)


def _validate_unified_v11_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != _UNIFIED_V11_PACKET_KEYS:
        raise ValueError("invalid unified_v11 record packet fields")
    step_scale = packet["step_scale"]
    if (
        isinstance(step_scale, bool)
        or not isinstance(step_scale, (int, float))
        or not math.isfinite(step_scale)
        or float(step_scale) <= 0.0
        or float(step_scale) > 1.0
    ):
        raise ValueError("invalid unified_v11 step_scale outside (0, 1]")
    ema_margin = packet["ema_margin"]
    if isinstance(ema_margin, bool) or not isinstance(ema_margin, (int, float)) or not math.isfinite(ema_margin):
        raise ValueError("invalid unified_v11 ema_margin")
    ema_data_scale = packet["ema_data_scale"]
    if (
        isinstance(ema_data_scale, bool)
        or not isinstance(ema_data_scale, (int, float))
        or not math.isfinite(ema_data_scale)
        or float(ema_data_scale) <= 0.0
    ):
        raise ValueError("invalid unified_v11 ema_data_scale")
    ema_push_res = packet["ema_push_res"]
    if (
        isinstance(ema_push_res, bool)
        or not isinstance(ema_push_res, (int, float))
        or not math.isfinite(ema_push_res)
        or float(ema_push_res) < 0.0
    ):
        raise ValueError("invalid unified_v11 ema_push_res")
    if type(packet["margin_initialized"]) is not bool:
        raise ValueError("invalid unified_v11 margin_initialized")
    validate_v11_bank_packet(packet["cond_fiber_bank"])
    t1c_sub = {k: packet[k] for k in _T1C_PACKET_KEYS}
    _validate_t1c_packet(t1c_sub)
    return packet


class _UnifiedV11RecordMixin(_T1CouplingRecordMixin):
    """Unified-v11 critic record mixin combining 4-Law persistent state and zero-`eigh` CondFiberBank."""

    def _init_unified_v11_fields(self) -> None:
        self._init_t1c_fields()
        self._step_scale: float = 1.0
        self._t1c_disc_gate: float = 0.0
        self._simplicial_step: bool = False
        self.ema_margin: float = 0.0
        self.ema_data_scale: float = 1.0
        self.ema_push_res: float = 1.0
        self.margin_initialized: bool = False
        self.cond_fiber_bank = UnifiedV11CondFiberBank()
        self._live_captures: List[List[Any]] = []
        self._last_probe_actual_input: Optional[torch.Tensor] = None
        self._d_probe_list: List[Tuple[Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = []
        self._g_probe_cache: Optional[Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = None
        self._in_probe: bool = False
        self._enable_secant_probe: bool = True

    @property
    def ema_residual(self) -> float:
        base = super().ema_residual
        if self.margin_initialized:
            margin_res = max(0.0, abs(float(self.ema_margin)) - 0.80) * 0.25
            return max(base, margin_res)
        return base

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
            self.observe_scale(x_real, x_fake)

    @torch.no_grad()
    def observe_scale(
        self,
        x_real: torch.Tensor,
        x_fake: Optional[torch.Tensor] = None,
    ) -> None:
        """Update checkpointed data scale and pushforward residual EMAs."""
        if getattr(self, "_in_penalty", False) or getattr(self, "_in_probe", False) or getattr(self, "_in_protected_probe", False):
            return
        if x_real is None or x_real.numel() == 0:
            return
        step_tag = int(getattr(self, "observed_steps", 0))
        xr = x_real.detach().reshape(x_real.shape[0], -1).float()
        sc = float(xr.square().sum(dim=-1).mean().sqrt().clamp_min(0.05).item())
        if math.isfinite(sc):
            self.ema_data_scale = sc if step_tag == 0 else (0.95 * self.ema_data_scale + 0.05 * sc)
        if x_fake is not None and x_fake.shape == x_real.shape:
            xf = x_fake.detach().reshape(x_fake.shape[0], -1).float()
            mean_gap = float((xf.mean(dim=0) - xr.mean(dim=0)).norm().item())
            std_gap = abs(float(xf.std().item()) - float(xr.std().item()))
            rel_gap = min(10.0, (mean_gap + 0.5 * std_gap) / max(0.05, self.ema_data_scale))
            if math.isfinite(rel_gap):
                self.ema_push_res = rel_gap if step_tag == 0 else (0.92 * self.ema_push_res + 0.08 * rel_gap)

    def clear_transient_captures(self) -> None:
        if hasattr(self, "_live_captures") and self._live_captures:
            _clear_live_capture_entries(self._live_captures)
        self._last_probe_actual_input = None

    def clear_step_probes(self) -> None:
        if hasattr(self, "_d_probe_list") and self._d_probe_list:
            self._d_probe_list.clear()

    def _unified_v11_packet(self) -> Dict[str, Any]:
        return {
            **self._t1c_packet(),
            "step_scale": float(self._step_scale),
            "ema_margin": float(self.ema_margin),
            "ema_data_scale": float(self.ema_data_scale),
            "ema_push_res": float(self.ema_push_res),
            "margin_initialized": bool(self.margin_initialized),
            "cond_fiber_bank": self.cond_fiber_bank.to_packet(),
        }

    def _restore_unified_v11_packet(self, packet: Dict[str, Any]) -> None:
        _validate_unified_v11_packet(packet)
        t1c_sub = {k: packet[k] for k in _T1C_PACKET_KEYS}
        self._restore_t1c_packet(t1c_sub)
        self._step_scale = float(packet["step_scale"])
        self.ema_margin = float(packet["ema_margin"])
        self.ema_data_scale = float(packet["ema_data_scale"])
        self.ema_push_res = float(packet["ema_push_res"])
        self.margin_initialized = bool(packet["margin_initialized"])
        self.cond_fiber_bank.load_packet(packet["cond_fiber_bank"])
        self._t1c_disc_gate = 0.0
        self._simplicial_step = False
        self.clear_transient_captures()
        self._live_captures = []
        self._d_probe_list = []
        self._g_probe_cache = None
        self._last_probe_actual_input = None
        self._in_probe = False

    def _copy_unified_v11_to(self, dup: Any) -> None:
        self._copy_t1c_to(dup)
        dup._step_scale = self._step_scale
        dup.ema_margin = self.ema_margin
        dup.ema_data_scale = self.ema_data_scale
        dup.ema_push_res = self.ema_push_res
        dup.margin_initialized = self.margin_initialized
        dup.cond_fiber_bank = self.cond_fiber_bank.copy()
        dup._t1c_disc_gate = 0.0
        dup._simplicial_step = False
        dup._live_captures = []
        dup._d_probe_list = []
        dup._g_probe_cache = None
        dup._last_probe_actual_input = None
        dup._in_probe = False
        dup._enable_secant_probe = self._enable_secant_probe


class _UnifiedV11ObserverStepRecord(_UnifiedV11RecordMixin, CriticStepRecord):
    """Observer record for UnifiedV11EquilibratedOptimizer, DualNorm, and plain Adam critics."""

    def __init__(self) -> None:
        super().__init__()
        self._init_unified_v11_fields()

    def record_step(self, optimizer_or_lr: Any) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()
        self.clear_step_probes()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "unified_v11": self._unified_v11_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "unified_v11" not in state:
            raise ValueError("unified_v11 observer checkpoint is missing 'unified_v11' state")
        if set(state) != {*CriticStepRecord.KEYS, "unified_v11"}:
            raise ValueError("invalid unified_v11 observer record fields")
        _validate_unified_v11_packet(state["unified_v11"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_unified_v11_packet(state["unified_v11"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_UnifiedV11ObserverStepRecord":
        dup = _UnifiedV11ObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_v11_to(dup)
        return dup


class UnifiedV11StepRecord(_UnifiedV11RecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``UnifiedV11CriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "unified_v11")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_unified_v11_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()
        self.clear_step_probes()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "unified_v11", "unified_v11": self._unified_v11_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid UnifiedV11StepRecord checkpoint keys")
        if state.get("formulation") != "unified_v11":
            raise ValueError("checkpoint formulation is not 'unified_v11'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid UnifiedV11StepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("UnifiedV11StepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            val = state[key]
            if key == "lr_last" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid UnifiedV11StepRecord {key}")
        _validate_unified_v11_packet(state["unified_v11"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_unified_v11_packet(state["unified_v11"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "UnifiedV11StepRecord":
        dup = UnifiedV11StepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_v11_to(dup)
        return dup


def _unified_v11_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-bound critic forward hook capturing live endpoints, secant probes, chart inputs, and thermal cooling."""
    rec = getattr(module, "_unified_v11_record", getattr(module, "_t1c_record", None))
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
            alive = [True]
            entry: List[Any] = [step_tag, src_key, c_sig, x_act, unscaled, alive]
            raw_target.register_hook(lambda _g, _a=alive: _a.clear())
            captures.append(entry)
            _trim_live_captures(captures)

        if (
            getattr(rec, "_enable_secant_probe", True)
            and x_first.ndim == 2
            and x_first.shape[0] >= 1
            and len(args) == 1
            and not kwargs
        ):
            b_full = int(x_first.shape[0])
            in_dim = max(1, int(x_first[0].numel()))
            b_p = min(b_full, max(2, min(_PROBE_SLICE, 8192 // in_dim)))
            p_args = tuple(_slice_probe_arg(a, b_full, b_p) for a in args)
            if _module_needs_eval_probe_baseline(module):
                d_pre = _eval_v11_probe_forward(module, p_args, {}, rec)
            else:
                d_pre = raw_target.detach()[:b_p].to(dtype=torch.float32)
            if d_pre is not None:
                probes = getattr(rec, "_d_probe_list", None)
                if probes is None:
                    probes = []
                    rec._d_probe_list = probes
                probes.append((p_args, {}, d_pre))
                if len(probes) > _MAX_D_PROBES:
                    del probes[: len(probes) - _MAX_D_PROBES]

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

    if (
        x_first is not None
        and x_first.ndim >= 2
        and x_first.shape[0] > 0
        and x_first[0].numel() > 1024
        and len(args) == 1
        and not kwargs
    ):
        target._t1c_bypassed = True
        if rec is not None:
            target._t1c_record = rec
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


def _module_needs_eval_probe_baseline(module: nn.Module) -> bool:
    cached = getattr(module, "_unified_v11_has_bn", getattr(module, "_unified_v11_gen_has_bn", None))
    if isinstance(cached, bool):
        return cached
    return any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
        for m in module.modules()
    )


def record_unified_v11_generator_probe(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
    record: Any,
    *,
    require_module_training: bool = True,
) -> bool:
    """Capture an element-budgeted generator probe for Unified-v11 function-space secant regulation.

    Supports single-tensor and multi-input/composite generator modules whose ``forward(*args, **kwargs)``
    replays the full generator computation from sampled latent (and conditioning) to generated output.
    """
    if (
        record is None
        or getattr(record, "_in_probe", False)
        or getattr(record, "_in_penalty", False)
        or getattr(record, "_in_protected_probe", False)
        or not getattr(record, "_enable_secant_probe", True)
        or (require_module_training and not module.training)
        or not torch.is_grad_enabled()
    ):
        return False
    raw_out = output[0] if isinstance(output, (tuple, list)) and output else output
    if (
        not isinstance(raw_out, torch.Tensor)
        or not raw_out.is_floating_point()
        or not raw_out.requires_grad
        or raw_out.ndim < 2
        or raw_out.shape[0] < 1
    ):
        return False
    import torch.autograd.forward_ad as fwAD
    if fwAD.unpack_dual(raw_out).tangent is not None:
        return False
    b_full = int(raw_out.shape[0])
    out_dim = max(1, int(raw_out[0].numel()))
    b_p = min(b_full, max(2, min(_PROBE_SLICE, 8192 // out_dim)))
    p_args = tuple(_slice_probe_arg(a, b_full, b_p) for a in args)
    p_kwargs = {k: _slice_probe_arg(v, b_full, b_p) for k, v in kwargs.items()}
    if _module_needs_eval_probe_baseline(module):
        y_pre = _eval_v11_probe_forward(module, p_args, p_kwargs, record)
        if y_pre is None:
            return False
    else:
        y_pre = raw_out.detach()[:b_p].to(dtype=torch.float32)
    record._g_probe_cache = (module, p_args, p_kwargs, y_pre)
    return True


def _unified_v11_generator_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    """Capture a dimension-agnostic element-budgeted generator slice for function-space secant trust-band regulation."""
    rec = getattr(module, "_unified_v11_record", None)
    if not getattr(module, "_unified_v11_allow_multi_input", False) and (len(args) != 1 or bool(kwargs)):
        return
    record_unified_v11_generator_probe(module, args, kwargs, output, rec, require_module_training=True)


def _attach_unified_v11_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local unified_v11 forward hook to ``critic`` (zero global state)."""
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
    critic._unified_v11_record = record
    critic._unified_record = record
    critic._t1c_record = record
    critic._t2_record = record
    critic._unified_v11_has_bn = any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
        for m in critic.modules()
    )
    if getattr(critic, "_unified_v11_hook_registered", False):
        return
    critic.register_forward_hook(_unified_v11_critic_forward_hook, with_kwargs=True)
    critic._unified_v11_hook_registered = True


def _attach_unified_v11_generator_hook(
    generator: nn.Module | None,
    record: CriticStepRecord,
    *,
    allow_multi_input: bool = False,
) -> None:
    """Attach instance-local unified_v11 secant probe hook to ``generator``."""
    if not isinstance(generator, nn.Module):
        return
    generator._unified_v11_record = record
    generator._unified_v11_allow_multi_input = bool(allow_multi_input)
    generator._unified_v11_gen_has_bn = any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
        for m in generator.modules()
    )
    if getattr(generator, "_unified_v11_gen_hook_registered", False):
        return
    generator.register_forward_hook(_unified_v11_generator_forward_hook, with_kwargs=True)
    generator._unified_v11_gen_hook_registered = True


def bind_unified_v11_record(record: Any, *consumers: Any) -> None:
    """Bind instance-local ``unified_v11`` critic record to losses, optimizers, and priors."""
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
        if hasattr(consumer, "unified_v11_record"):
            consumer.unified_v11_record = record


class UnifiedV11CriticAdam(K3PCriticAdam):
    """Formulation critic Adam for ``critic_formulation='unified_v11'`` (zero EMA critic copy)."""

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
        self.record = UnifiedV11StepRecord()
        self._t1c_record = self.record
        self._t2_record = self.record
        self._unified_record = self.record
        self._unified_v11_record = self.record
        _attach_unified_v11_critic_hook(critic, self.record)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "UnifiedV11CriticAdam",
    "UnifiedV11StepRecord",
    "_UnifiedV11ObserverStepRecord",
    "_attach_unified_v11_critic_hook",
    "_attach_unified_v11_generator_hook",
    "_cond_signature",
    "_eval_v11_probe_forward",
    "_extract_t1c_inputs_parts",
    "_lookup_t1c_capture_parts",
    "_record_t1c_capture",
    "_tensor_identity_key",
    "_validate_unified_v11_packet",
    "bind_unified_v11_record",
    "record_unified_v11_generator_probe",
]
