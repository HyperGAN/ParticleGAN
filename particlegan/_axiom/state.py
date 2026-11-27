"""Axiom L0 State & Capture Layer: Schema-2 StepRecord, Weak Registries, and RNG-Isolated Probes."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import weakref

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

SCHEMA_VERSION: int = 2
_MAX_LIVE_CAPTURES: int = 6
_MAX_CAPTURE_RING: int = 16
_MAX_RING_SAMPLE_BYTES: int = 16_384
_MAX_RING_TOTAL_BYTES: int = 2 * 1024 * 1024
_PROBE_MAX_BATCH: int = 32
_PROBE_MAX_ELEMENTS: int = 1024
_MAX_GRAD_FN_DEPTH: int = 8
_MAX_GRAD_FN_NODES: int = 256

TensorId = Tuple[int, int, Tuple[int, ...], str]
CaptureKey = Tuple[int, int, int, int]
ScalarBound = Tuple[Optional[float], Optional[float], bool, bool]

_EMA_BOOLS = frozenset({"grad_sq_initialized", "margin_initialized"})
_BOOL_FIELDS: Tuple[str, ...] = (
    "anchor_started",
    "grad_sq_initialized",
    "margin_initialized",
    "ell_initialized",
    "thermal_initialized",
    "thermal_locked",
    "cond_initialized",
    "het_initialized",
)
_THERMAL_KEYS: Tuple[str, ...] = (
    "ema_ed",
    "ema_mu_sq",
    "ema_spread",
    "ema_q90",
    "ema_hmax",
    "ema_tail",
)
_SCALAR_BOUNDS: Dict[str, ScalarBound] = {
    "chord_phase": (0.0, 1.0, False, True),
    "ema_grad_sq": (0.0, None, False, False),
    "ema_margin": (-60.0, 60.0, False, False),
    "ema_data_scale": (0.0, None, True, False),
    "last_data_scale": (0.0, None, True, False),
    "ema_push_res": (0.0, None, False, False),
    "ell_ema": (0.0, None, False, False),
    "thermal_scale": (1e-5, 1.0, False, False),
    "step_scale": (0.0, 1.0, True, False),
    "cond_ema": (-0.5, 2.0, False, False),
    "het_ema": (0.0, None, False, False),
    "g_func_gain": (0.0, 1.0, True, False),
    **{key: (None, None, False, False) for key in _THERMAL_KEYS},
}
_DICT_FIELDS: Tuple[str, ...] = ("chart_state", "atomic_ledger", "cond_fiber_bank")
_RECORD_KEYS = frozenset(
    {
        "schema",
        "calls",
        "observed_steps",
        "last_ratio",
        *_BOOL_FIELDS,
        *_SCALAR_BOUNDS,
        *_DICT_FIELDS,
    }
)


# ---------------------------------------------------------------------------
# Basic Tensor, Identity & Validation Helpers
# ---------------------------------------------------------------------------


def first_tensor_output(output: Any) -> Any:
    """Return the primary element from a tensor or non-empty tuple/list output."""
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def format_logits_2d(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshape critic logits to ``(batch_size, -1)`` without modifying data."""
    if logits.ndim < 2:
        return logits.reshape(batch_size, 1)
    return logits.reshape(batch_size, -1)


def _storage_ptr(tensor: Any) -> int:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.numel() == 0:
        return 0
    try:
        return int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return 0


def tensor_id(tensor: Any) -> Optional[TensorId]:
    """Return a storage-backed tensor identity tuple or ``None`` for empty/non-tensors."""
    ptr = _storage_ptr(tensor)
    if not ptr:
        return None
    shape = tuple(int(dim) for dim in tensor.shape)
    return (ptr, int(tensor.storage_offset()), shape, str(tensor.device))


def capture_key(tensor: Any) -> Optional[CaptureKey]:
    """Return a view/detach-stable storage key ``(ptr, offset, numel, batch_size)``."""
    ptr = _storage_ptr(tensor)
    if not ptr:
        return None
    return (
        ptr,
        int(tensor.storage_offset()),
        int(tensor.numel()),
        int(tensor.shape[0]),
    )


def _sig_item(value: Any, name: Optional[str] = None) -> Tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        if name is None:
            return ("T", tensor_id(value), str(value.dtype))
        return ("KT", name, tensor_id(value), str(value.dtype))
    if name is None:
        return ("V", type(value).__qualname__, repr(value))
    return ("KV", name, type(value).__qualname__, repr(value))


def cond_signature(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Build a lightweight structural signature of auxiliary conditioning arguments."""
    pos_sig = [_sig_item(arg) for arg in args[1:]]
    kw_sig = [_sig_item(kwargs[key], key) for key in sorted(kwargs)]
    return tuple(pos_sig + kw_sig)


def check_scalar(
    val: Any,
    name: str,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    strict_lo: bool = False,
    strict_hi: bool = False,
) -> float:
    """Validate that ``val`` is a finite non-boolean number inside optional bounds."""
    if (
        isinstance(val, bool)
        or not isinstance(val, (int, float))
        or not math.isfinite(val)
    ):
        raise ValueError(f"invalid {name}: {val!r}")
    value = float(val)
    if lo is not None:
        too_low = value <= lo if strict_lo else value < lo
        if too_low:
            raise ValueError(f"{name}={value} below bound {lo}")
    if hi is not None:
        too_high = value >= hi if strict_hi else value > hi
        if too_high:
            raise ValueError(f"{name}={value} above bound {hi}")
    return value


def _walk_payload(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {key: _walk_payload(val, fn) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_walk_payload(item, fn) for item in obj)
    return deepcopy(obj)


def _clone_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _validate_dict_payload(mapping: Dict[Any, Any], name: str) -> None:
    for key, val in mapping.items():
        if not isinstance(key, (str, int, tuple)):
            raise ValueError(f"unsupported key in {name}")
        _validate_payload(val, name)


def _validate_payload(obj: Any, name: str) -> None:
    if obj is None or type(obj) in (bool, int, str):
        return
    if type(obj) is float:
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in {name}")
        return
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
            raise ValueError(f"non-finite tensor in {name}")
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _validate_payload(item, name)
        return
    if isinstance(obj, dict):
        _validate_dict_payload(obj, name)
        return
    raise ValueError(f"unsupported type {type(obj).__name__} in {name}")


def _module_has_bn_or_dropout(module: nn.Module) -> bool:
    return any(
        isinstance(
            sub, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
        )
        for sub in module.modules()
    )


def _replace_first_output(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    return replacement


def _index_arg(value: Any, batch_size: int, idx: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[0] == batch_size:
            return value.detach().index_select(0, idx.to(value.device))
        return value.detach()
    if isinstance(value, (tuple, list)):
        return type(value)(_index_arg(item, batch_size, idx) for item in value)
    if isinstance(value, dict):
        return {key: _index_arg(item, batch_size, idx) for key, item in value.items()}
    return value


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


# ---------------------------------------------------------------------------
# Weak Registries (Zero Monkey-Patched Module or Tensor Attributes)
# ---------------------------------------------------------------------------


class WeakTensorMap:
    """Identity-keyed weak map safe for ``torch.Tensor`` and ``nn.Parameter``.

    ``weakref.WeakKeyDictionary`` invokes ``Tensor.__eq__`` during lookup, which
    raises ``RuntimeError`` on multi-element tensors and conflates equal-valued
    scalar tensors. Keying by ``id(tensor)`` paired with a live ``weakref.ref``
    callback preserves exact identity semantics with automatic GC eviction.
    """

    def __init__(self) -> None:
        self._entries: Dict[int, Tuple[weakref.ReferenceType[Any], Any]] = {}

    def _evict(self, key: int, dead_ref: weakref.ReferenceType[Any]) -> None:
        entry = self._entries.get(key)
        if entry is not None and entry[0] is dead_ref:
            self._entries.pop(key, None)

    def set(self, tensor: torch.Tensor, value: Any) -> None:
        key = id(tensor)
        ref = weakref.ref(tensor, lambda dead, k=key: self._evict(k, dead))
        self._entries[key] = (ref, value)

    def get(self, tensor: torch.Tensor, default: Any = None) -> Any:
        entry = self._entries.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return entry[1]
        return default

    def pop(self, tensor: torch.Tensor, default: Any = None) -> Any:
        entry = self._entries.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            self._entries.pop(id(tensor), None)
            return entry[1]
        return default

    def __contains__(self, tensor: object) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        entry = self._entries.get(id(tensor))
        return entry is not None and entry[0]() is tensor

    def __len__(self) -> int:
        return sum(1 for ref, _ in list(self._entries.values()) if ref() is not None)

    def clear(self) -> None:
        self._entries.clear()


_MODULE_RECORDS: "weakref.WeakKeyDictionary[nn.Module, AxiomStepRecord]" = (
    weakref.WeakKeyDictionary()
)
_MODULE_HAS_BN: "weakref.WeakKeyDictionary[nn.Module, bool]" = (
    weakref.WeakKeyDictionary()
)
_MODULE_PENDING_SOURCE: "weakref.WeakKeyDictionary[nn.Module, Optional[TensorId]]" = (
    weakref.WeakKeyDictionary()
)
_CRITIC_HOOK_HANDLES: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
    weakref.WeakKeyDictionary()
)
_GEN_HOOK_HANDLES: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
    weakref.WeakKeyDictionary()
)
_PRIOR_ACTIVE_ROWS = WeakTensorMap()
_TENSOR_RECORDS = WeakTensorMap()


def set_module_record(module: nn.Module, record: "AxiomStepRecord") -> None:
    """Associate a user ``nn.Module`` with an ``AxiomStepRecord`` via weak reference."""
    _MODULE_RECORDS[module] = record


def get_module_record(module: Any) -> Optional["AxiomStepRecord"]:
    """Look up the ``AxiomStepRecord`` bound to ``module`` (unwrapping ``.critic``/``.base``)."""
    if not isinstance(module, nn.Module):
        return None
    record = _MODULE_RECORDS.get(module)
    if record is not None:
        return record
    for wrapper_attr in ("critic", "base"):
        inner = getattr(module, wrapper_attr, None)
        if isinstance(inner, nn.Module):
            record = _MODULE_RECORDS.get(inner)
            if record is not None:
                return record
    return None


def set_tensor_record(tensor: torch.Tensor, record: "AxiomStepRecord") -> None:
    """Associate a tensor (e.g., ``prior.z``) with an ``AxiomStepRecord`` without mutating it."""
    _TENSOR_RECORDS.set(tensor, record)


def get_tensor_record(tensor: torch.Tensor) -> Optional["AxiomStepRecord"]:
    """Look up the ``AxiomStepRecord`` weakly bound to ``tensor``."""
    return _TENSOR_RECORDS.get(tensor)


def get_axiom_record(obj: Any) -> Optional["AxiomStepRecord"]:
    """Resolve an ``AxiomStepRecord`` from a module, tensor, or Axiom component."""
    if obj is None:
        return None
    if isinstance(obj, AxiomStepRecord):
        return obj
    if isinstance(obj, torch.Tensor):
        return get_tensor_record(obj)
    if isinstance(obj, nn.Module):
        mod_rec = get_module_record(obj)
        if mod_rec is not None:
            return mod_rec
    for attr in ("record", "axiom_record", "_axiom_record"):
        candidate = getattr(obj, attr, None)
        if isinstance(candidate, AxiomStepRecord):
            return candidate
    return None


def set_active_prior_rows(table: torch.Tensor, rows: torch.Tensor) -> None:
    """Register active batch rows for ``table`` without attaching attributes to the tensor."""
    _PRIOR_ACTIVE_ROWS.set(table, rows.detach())


def get_active_prior_rows(table: torch.Tensor) -> Optional[torch.Tensor]:
    """Return active batch rows registered for ``table``, if any."""
    return _PRIOR_ACTIVE_ROWS.get(table)


def clear_active_prior_rows(table: torch.Tensor) -> None:
    """Remove any registered active batch rows for ``table``."""
    _PRIOR_ACTIVE_ROWS.pop(table, None)


def _walk_grad_fn_for_prior(
    grad_fn: Any,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    queue: List[Tuple[Any, int]] = [(grad_fn, 0)]
    seen = {id(grad_fn)}
    while queue and len(seen) <= _MAX_GRAD_FN_NODES:
        node, depth = queue.pop(0)
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor):
            rows = get_active_prior_rows(variable)
            if rows is not None:
                return variable, rows
        if depth >= _MAX_GRAD_FN_DEPTH:
            continue
        for nxt, _ in getattr(node, "next_functions", ()):
            if nxt is not None and id(nxt) not in seen:
                seen.add(id(nxt))
                queue.append((nxt, depth + 1))
    return None, None


def recover_active_prior_rows(
    x_fake: torch.Tensor,
    record: Optional["AxiomStepRecord"] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Recover ``(prior_table, active_rows)`` from ``record`` or the weak prior-row registry."""
    if record is not None:
        table = record._active_prior_table
        rows = record._active_batch_rows
        if table is not None and rows is not None:
            return table, rows
    grad_fn = getattr(x_fake, "grad_fn", None)
    if grad_fn is None or len(_PRIOR_ACTIVE_ROWS) == 0:
        return None, None
    found_table, found_rows = _walk_grad_fn_for_prior(grad_fn)
    if found_table is not None and record is not None:
        set_tensor_record(found_table, record)
    return found_table, found_rows


# ---------------------------------------------------------------------------
# Frozen CaptureEntry & Bounded Transient CaptureRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureEntry:
    """Immutable critic-forward capture metadata stored on ``logits._axiom_capture``."""

    x_in: torch.Tensor
    cond: Optional[torch.Tensor] = None
    record: Optional["AxiomStepRecord"] = None
    has_cont_cond: bool = False
    parts: Optional[Tuple[torch.Tensor, ...]] = None
    thermally_scaled: bool = False

    def detached(self) -> "CaptureEntry":
        """Return a detached copy suitable for transient ring lookup across ``.detach()``."""
        det_parts = (
            None if self.parts is None else tuple(part.detach() for part in self.parts)
        )
        return CaptureEntry(
            x_in=self.x_in.detach(),
            cond=None if self.cond is None else self.cond.detach(),
            record=self.record,
            has_cont_cond=self.has_cont_cond,
            parts=det_parts,
            thermally_scaled=self.thermally_scaled,
        )

    def nbytes(self) -> int:
        """Total tensor storage bytes held by this capture entry."""
        total = _tensor_nbytes(self.x_in) + _tensor_nbytes(self.cond)
        if self.parts is not None:
            total += sum(_tensor_nbytes(part) for part in self.parts)
        return total


@dataclass
class _LiveEndpoint:
    step: int
    source_key: TensorId
    cond_sig: Tuple[Any, ...]
    x_in: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    alive: List[bool]

    def release(self) -> None:
        self.x_in = None
        self.logits = None
        self.alive.clear()


class CaptureRegistry:
    """Bounded transient registry resolving detached logits and live critic endpoints."""

    def __init__(
        self,
        *,
        max_ring: int = _MAX_CAPTURE_RING,
        max_sample_bytes: int = _MAX_RING_SAMPLE_BYTES,
        max_total_bytes: int = _MAX_RING_TOTAL_BYTES,
        max_live: int = _MAX_LIVE_CAPTURES,
    ) -> None:
        self._max_ring = int(max_ring)
        self._max_sample_bytes = int(max_sample_bytes)
        self._max_total_bytes = int(max_total_bytes)
        self._max_live = int(max_live)
        self._ring: OrderedDict[CaptureKey, Tuple[CaptureEntry, int]] = OrderedDict()
        self._ring_bytes: int = 0
        self._live: List[_LiveEndpoint] = []

    @property
    def ring_size(self) -> int:
        return len(self._ring)

    @property
    def ring_bytes(self) -> int:
        return self._ring_bytes

    @property
    def live_count(self) -> int:
        return len(self._live)

    def _evict_oldest_ring(self) -> None:
        _, (_, evicted_bytes) = self._ring.popitem(last=False)
        self._ring_bytes = max(0, self._ring_bytes - evicted_bytes)

    def put_detached(self, logits: torch.Tensor, entry: CaptureEntry) -> bool:
        """Cache a detached capture entry if within per-sample and total memory bounds."""
        batch_size = max(1, int(entry.x_in.shape[0]))
        sample_bytes = _tensor_nbytes(entry.x_in) // batch_size
        if sample_bytes > self._max_sample_bytes:
            return False
        key = capture_key(logits)
        if key is None:
            return False
        det = entry.detached()
        entry_bytes = det.nbytes()
        if entry_bytes > self._max_total_bytes:
            return False
        prev = self._ring.pop(key, None)
        if prev is not None:
            self._ring_bytes = max(0, self._ring_bytes - prev[1])
        while self._ring and (
            len(self._ring) >= self._max_ring
            or self._ring_bytes + entry_bytes > self._max_total_bytes
        ):
            self._evict_oldest_ring()
        self._ring[key] = (det, entry_bytes)
        self._ring_bytes += entry_bytes
        return True

    def lookup_detached(self, logits: Any) -> Optional[CaptureEntry]:
        """Resolve a detached/reshaped logit tensor against the transient capture ring."""
        key = capture_key(logits)
        if key is None:
            return None
        hit = self._ring.get(key)
        return None if hit is None else hit[0]

    def record_live_endpoint(
        self,
        *,
        step: int,
        source_key: TensorId,
        cond_sig: Tuple[Any, ...],
        x_in: torch.Tensor,
        raw_logits: torch.Tensor,
    ) -> None:
        """Record a live critic output for reuse by Law 2 zero-mean logit gauge."""
        alive = [True]
        raw_logits.register_hook(lambda _grad, flag=alive: flag.clear())
        logits_ref = (
            raw_logits.float() if raw_logits.dtype == torch.bfloat16 else raw_logits
        )
        self._live.append(
            _LiveEndpoint(step, source_key, cond_sig, x_in.detach(), logits_ref, alive)
        )
        overflow = len(self._live) - self._max_live
        if overflow > 0:
            for item in self._live[:overflow]:
                item.release()
            del self._live[:overflow]

    def find_live_endpoint(
        self,
        step: int,
        x_tensor: torch.Tensor,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Return a matching live logit tensor for ``(step, x_tensor, cond_sig)`` if still alive."""
        src_key = tensor_id(x_tensor)
        if src_key is None:
            return None
        csig = cond_signature((x_tensor, *args), kwargs)
        for item in reversed(self._live):
            if (
                item.step == step
                and item.source_key == src_key
                and item.cond_sig == csig
                and item.alive
            ):
                if item.logits is not None:
                    return item.logits
        return None

    def clear(self) -> None:
        """Release all transient live endpoints and detached capture references."""
        for item in self._live:
            item.release()
        self._live.clear()
        self._ring.clear()
        self._ring_bytes = 0


def _is_batch_tensor(item: Any, batch_size: int) -> bool:
    return (
        isinstance(item, torch.Tensor)
        and item.ndim >= 1
        and item.shape[0] == batch_size
        and item.numel() > 0
    )


def _merge_discrete_cond(discretes: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not discretes:
        return None
    if len(discretes) == 1:
        return discretes[0]
    return torch.cat(discretes, dim=-1)


def extract_axiom_inputs_parts(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    bool,
    Optional[Tuple[torch.Tensor, ...]],
]:
    """Split critic arguments into primary/concatenated float input, discrete cond, and float parts."""
    items = list(args)
    if kwargs:
        items.extend(kwargs[key] for key in sorted(kwargs))
    floats: List[torch.Tensor] = []
    discretes: List[torch.Tensor] = []
    for item in items:
        if not _is_batch_tensor(item, batch_size):
            continue
        if item.is_floating_point():
            floats.append(item)
        elif not item.is_complex():
            discretes.append(item.detach().to(torch.int64).reshape(batch_size, -1))
    if not floats:
        return None, None, False, None
    cond = _merge_discrete_cond(discretes)
    if len(floats) > 1:
        parts = tuple(tensor.reshape(batch_size, -1) for tensor in floats)
        return torch.cat(parts, dim=-1), cond, True, parts
    return floats[0], cond, False, None


def record_axiom_capture(output: Any, entry: CaptureEntry) -> None:
    """Attach a single frozen ``_axiom_capture`` entry to ``output`` and populate the ring."""
    target = first_tensor_output(output)
    if (
        not isinstance(target, torch.Tensor)
        or target.ndim < 1
        or target.shape[0] != entry.x_in.shape[0]
    ):
        return
    target._axiom_capture = entry  # type: ignore[attr-defined]
    if entry.record is not None:
        entry.record.captures.put_detached(target, entry)


def lookup_axiom_capture(
    logits: Any,
    fallback_record: Optional["AxiomStepRecord"] = None,
) -> Optional[CaptureEntry]:
    """Resolve ``CaptureEntry`` directly from ``logits._axiom_capture`` or via ``fallback_record``."""
    entry = getattr(logits, "_axiom_capture", None)
    if isinstance(entry, CaptureEntry):
        if entry.record is None and fallback_record is not None:
            return CaptureEntry(
                x_in=entry.x_in,
                cond=entry.cond,
                record=fallback_record,
                has_cont_cond=entry.has_cont_cond,
                parts=entry.parts,
                thermally_scaled=entry.thermally_scaled,
            )
        return entry
    if fallback_record is not None:
        return fallback_record.lookup_capture(logits)
    return None


# ---------------------------------------------------------------------------
# AxiomStepRecord (Schema = 2)
# ---------------------------------------------------------------------------


def _ema_prop(name: str) -> property:
    is_bool = name in _EMA_BOOLS

    def _get(self: "AxiomStepRecord") -> Any:
        if name in self._ema_t:
            self._flush_ema()
        return self._ema[name]

    def _set(self: "AxiomStepRecord", value: Any) -> None:
        self._ema[name] = bool(value) if is_bool else float(value)
        self._ema_t.pop(name, None)

    return property(_get, _set)


def _validate_record_header(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != _RECORD_KEYS:
        raise ValueError("invalid AxiomStepRecord checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != SCHEMA_VERSION:
        raise ValueError(
            f"AxiomStepRecord requires schema={SCHEMA_VERSION}, got {state.get('schema')!r}"
        )
    for key in ("calls", "observed_steps"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f"invalid AxiomStepRecord.{key}: {state[key]!r}")
    for key in _BOOL_FIELDS:
        if type(state[key]) is not bool:
            raise ValueError(f"invalid AxiomStepRecord.{key}: {state[key]!r}")
    if state["anchor_started"] is not False:
        raise ValueError("AxiomStepRecord.anchor_started must be False")


def validate_axiom_record_state(state: Any) -> Tuple[Optional[float], Dict[str, float]]:
    """Fail-closed schema-2 validator for ``AxiomStepRecord`` checkpoints."""
    _validate_record_header(state)
    last_ratio = (
        None
        if state["last_ratio"] is None
        else check_scalar(state["last_ratio"], "last_ratio", lo=0.0)
    )
    scalars = {
        key: check_scalar(state[key], key, *bounds)
        for key, bounds in _SCALAR_BOUNDS.items()
    }
    for key in _DICT_FIELDS:
        payload = state[key]
        if payload is not None:
            if not isinstance(payload, dict):
                raise ValueError(f"AxiomStepRecord.{key} must be dict or None")
            _validate_payload(payload, key)
    return last_ratio, scalars


class AxiomStepRecord:
    """Explicit schema-2 checkpointed state linking Axiom critic, transport, prior, and optimizer."""

    SCHEMA = SCHEMA_VERSION

    ema_grad_sq = _ema_prop("ema_grad_sq")
    grad_sq_initialized = _ema_prop("grad_sq_initialized")
    ema_margin = _ema_prop("ema_margin")
    margin_initialized = _ema_prop("margin_initialized")
    ema_data_scale = _ema_prop("ema_data_scale")
    last_data_scale = _ema_prop("last_data_scale")
    ema_push_res = _ema_prop("ema_push_res")
    ell_ema = _ema_prop("ell_ema")

    def __init__(self) -> None:
        self.schema: int = SCHEMA_VERSION
        self.calls: int = 0
        self.observed_steps: int = 0
        self.anchor_started: bool = False
        self.last_ratio: Optional[float] = None
        self.chord_phase: float = 0.0
        self._ema: Dict[str, Any] = {
            "ema_grad_sq": 1.0,
            "grad_sq_initialized": False,
            "ema_margin": 0.0,
            "margin_initialized": False,
            "ema_data_scale": 1.0,
            "last_data_scale": 1.0,
            "ema_push_res": 1.0,
            "ell_ema": 0.0,
        }
        self.ell_initialized: bool = False
        self.thermal_initialized: bool = False
        self.thermal_locked: bool = False
        self.cond_initialized: bool = False
        self.het_initialized: bool = False
        self.thermal_scale: float = 1.0
        self.step_scale: float = 1.0
        self.het_ema: float = 1.0
        self.g_func_gain: float = 1.0
        self.ema_ed: float = 1.0
        self.ema_mu_sq: float = 1.0
        self.ema_spread: float = 1.0
        self.ema_q90: float = 1.0
        self.ema_hmax: float = 1.0
        self.ema_tail: float = 0.0
        self.cond_ema: float = 0.25
        self.chart_state: Optional[Dict[str, Any]] = None
        self.atomic_ledger: Optional[Dict[str, Any]] = None
        self.cond_fiber_bank: Optional[Dict[str, Any]] = None
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._ema_t: Dict[str, torch.Tensor] = {}
        self.captures: CaptureRegistry = CaptureRegistry()
        self._in_penalty: bool = False
        self._in_probe: bool = False
        self._in_protected_probe: bool = False
        self._simplicial_step: bool = False
        self._enable_secant_probe: bool = True
        self._disc_gate: float = 0.0
        self._last_probe_actual_input: Optional[torch.Tensor] = None
        self._g_probe_cache: Optional[
            Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any], torch.Tensor]
        ] = None
        self._active_prior_table: Optional[torch.Tensor] = None
        self._active_batch_rows: Optional[torch.Tensor] = None
        self._cond_bank_obj: Optional[Any] = None

    def lookup_capture(self, logits: Any) -> Optional[CaptureEntry]:
        """Look up a detached/reshaped logit tensor in the bounded capture ring."""
        return self.captures.lookup_detached(logits)

    def find_live_endpoint(
        self,
        x_tensor: torch.Tensor,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """Look up a live critic output for zero-mean logit gauge reuse."""
        return self.captures.find_live_endpoint(
            self.observed_steps, x_tensor, args, {} if kwargs is None else kwargs
        )

    def clear_transient_captures(self) -> None:
        """Clear all transient live endpoints, detached capture entries, and probe inputs."""
        self.captures.clear()
        self._last_probe_actual_input = None

    def record_step(self, optimizer: Any = None) -> None:
        """Advance observed critic step count and clear step-transient captures."""
        del optimizer
        self.observed_steps += 1
        self.clear_transient_captures()
        self._active_prior_table = None
        self._active_batch_rows = None

    def take_pending_ema_tensors(self) -> Tuple[List[str], List[torch.Tensor]]:
        """Return pending device-resident 0-D EMA shadow tensors for packed sync."""
        return list(self._ema_t.keys()), list(self._ema_t.values())

    def apply_flushed_ema_values(self, keys: List[str], values: List[float]) -> None:
        """Commit host floats from a packed flush back into the host EMA store."""
        for key, val in zip(keys, values):
            if key in self._ema:
                self._ema[key] = bool(val > 0.5) if key in _EMA_BOOLS else float(val)
                self._ema_t.pop(key, None)

    def _flush_ema(self) -> None:
        keys, tensors = self.take_pending_ema_tensors()
        if not tensors:
            return
        same_dev = all(tensor.device == tensors[0].device for tensor in tensors)
        if same_dev:
            values = [
                float(val)
                for val in torch.stack([tensor.double() for tensor in tensors]).tolist()
            ]
        else:
            values = [float(tensor.item()) for tensor in tensors]
        self.apply_flushed_ema_values(keys, values)

    def dev_ema(self, name: str, device: torch.device) -> torch.Tensor:
        """Return the 0-D float64 EMA tensor on ``device`` without host sync when resident."""
        current = self._ema_t.get(name)
        if current is not None and current.device == device:
            return current
        if current is not None:
            self._flush_ema()
        return torch.full(
            (), float(self._ema[name]), dtype=torch.float64, device=device
        )

    _dev_ema = dev_ema

    @torch.no_grad()
    def observe_grad_sq_tensor(self, s_hat_det: torch.Tensor) -> None:
        """Update ``ema_grad_sq`` purely on-device from detached critic squared gradient norm."""
        obs = s_hat_det.detach().double().clamp(0.0, 20.0)
        finite = torch.isfinite(obs)
        prev = self.dev_ema("ema_grad_sq", obs.device)
        prev_init = self.dev_ema("grad_sq_initialized", obs.device)
        updated = torch.where(prev_init > 0.5, 0.95 * prev + 0.05 * obs, obs)
        self._ema_t["ema_grad_sq"] = torch.where(finite, updated, prev)
        self._ema_t["grad_sq_initialized"] = torch.where(
            finite, prev_init.new_full((), 1.0), prev_init
        )

    _observe_grad_sq_tensor = observe_grad_sq_tensor

    @torch.no_grad()
    def _observe_scale_and_push_res(
        self,
        x_real: torch.Tensor,
        x_fake: Optional[torch.Tensor],
        was_init: torch.Tensor,
    ) -> None:
        dev = x_real.device
        xr = x_real.detach().reshape(x_real.shape[0], -1).double()
        xm = xr.abs().amax()
        sx = torch.where(xm > 0.0, xm, xm.new_ones(()))
        sc = ((xr / sx).square().sum(-1).mean().sqrt() * sx).clamp(1e-30, 1e30)
        psc = self.dev_ema("ema_data_scale", dev)
        lsc = self.dev_ema("last_data_scale", dev)
        lp = torch.log2(psc.clamp_min(1e-30))
        lc = torch.log2(sc)
        ll = torch.log2(lsc.clamp_min(1e-30))
        ok_s = torch.isfinite(sc) & (xm > 0.0)
        dl = lc - lp
        confirmed = (dl.abs() > 0.25) & ((lc - ll).abs() <= 1.5)
        step_l = torch.where(confirmed, 0.80 * dl, 0.05 * dl.clamp(-1.0, 1.0))
        next_scale = torch.where(
            ok_s, torch.where(was_init, torch.exp2(lp + step_l), sc), psc
        )
        self._ema_t["ema_data_scale"] = next_scale
        self._ema_t["last_data_scale"] = torch.where(ok_s, sc, lsc)
        if x_fake is None or x_fake.shape != x_real.shape:
            return
        xf = x_fake.detach().reshape(x_fake.shape[0], -1).double()
        mean_gap = (xf.mean(0) - xr.mean(0)).norm()
        std_gap = 0.5 * (xf.std() - xr.std()).abs()
        raw_res = ((mean_gap + std_gap) / next_scale.clamp_min(1e-30)).clamp_max(10.0)
        prev_res = self.dev_ema("ema_push_res", dev)
        upd_res = torch.where(was_init, 0.92 * prev_res + 0.08 * raw_res, raw_res)
        self._ema_t["ema_push_res"] = torch.where(
            torch.isfinite(raw_res), upd_res, prev_res
        )

    @torch.no_grad()
    def observe_margin_and_scale(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        x_real: Optional[torch.Tensor] = None,
        x_fake: Optional[torch.Tensor] = None,
    ) -> None:
        """Update critic logit margin, 2-step shock-aware data scale, and pushforward residual on-device."""
        if self._in_penalty or self._in_probe or self._in_protected_probe:
            return
        dev = real_logits.device
        margin = (
            real_logits.detach().float().mean().double()
            - fake_logits.detach().float().mean().double()
        ).clamp(-50.0, 50.0)
        fin_m = torch.isfinite(margin)
        prev_m = self.dev_ema("ema_margin", dev)
        prev_i = self.dev_ema("margin_initialized", dev)
        was_init = prev_i > 0.5
        self._ema_t["ema_margin"] = torch.where(
            fin_m, torch.where(was_init, 0.90 * prev_m + 0.10 * margin, margin), prev_m
        )
        self._ema_t["margin_initialized"] = torch.where(
            fin_m, prev_i.new_full((), 1.0), prev_i
        )
        if x_real is not None and x_real.numel() > 0:
            self._observe_scale_and_push_res(x_real, x_fake, was_init)

    @property
    def thermal_ema(self) -> float:
        """Aggregate 6-channel two-sample thermal residual."""
        under = min(1.0, max(0.0, -float(self.ema_spread)) / 0.03)
        q90 = float(self.ema_q90)
        hmax = float(self.ema_hmax)
        return max(
            0.5 * max(0.0, float(self.ema_ed)),
            math.sqrt(max(0.0, float(self.ema_mu_sq))),
            0.60 * abs(float(self.ema_spread)),
            max(q90, 0.60 * max(0.0, -q90) * under),
            max(hmax, 0.60 * max(0.0, -hmax) * under),
            max(0.0, float(self.ema_tail)),
        )

    ema_residual = thermal_ema

    def observe_thermal(
        self,
        r_ed: float,
        r_mu_sq: float,
        r_spread: float,
        r_q90: float,
        r_hmax: float,
        *,
        r_tail: float = 0.0,
        tau: float = 0.084,
        ema_decay: float = 0.96,
        s_min: float = 1e-5,
        shock_mu_ratio: float = 0.35,
        shock_ed_floor: float = 0.30,
        lock_hysteresis: float = 0.0,
    ) -> float:
        """Update 6-channel thermal EMA and return smoothstep thermal gradient scale."""
        ed_f, mu_f, tau_f = float(r_ed), float(r_mu_sq), float(tau)
        tau_pos = max(0.0, tau_f)
        lim = max(0.20, 2.0 * tau_pos)
        lim_ed = max(0.70, 4.0 * tau_pos)
        shock_ed = max(float(shock_ed_floor), 4.0 * tau_pos * tau_pos)
        clamped = (
            max(-lim_ed, min(ed_f, lim_ed)),
            max(-lim * lim, min(mu_f, lim * lim)),
            max(-lim, min(float(r_spread), lim)),
            max(-lim, min(float(r_q90), lim)),
            max(-lim, min(float(r_hmax), lim)),
            max(-lim, min(float(r_tail), lim)),
        )
        reheat = self.thermal_scale < 0.99 and (
            ed_f > shock_ed or mu_f > shock_mu_ratio * shock_ed
        )
        if not self.thermal_initialized or reheat:
            for key, val in zip(_THERMAL_KEYS, clamped):
                setattr(self, key, val)
            self.thermal_initialized = True
            self.thermal_locked = False
        else:
            decay = float(ema_decay)
            alpha = 1.0 - decay
            for key, val in zip(_THERMAL_KEYS, clamped):
                setattr(self, key, decay * getattr(self, key) + alpha * val)
        if tau_f <= 0.0:
            self.thermal_scale = 1.0
            self.thermal_locked = False
            return 1.0
        t_lo = (
            min(0.92, 0.50 + (float(lock_hysteresis) if self.thermal_locked else 0.0))
            * tau_f
        )
        u = min(1.0, max(0.0, (self.thermal_ema - t_lo) / max(1e-8, tau_f - t_lo)))
        th = u * u * (3.0 - 2.0 * u)
        s_floor = max(1e-5, float(s_min))
        if th >= 1.0:
            scale = 1.0
        elif th <= 0.0:
            scale = s_floor
        else:
            scale = max(
                s_floor, min(1.0, 0.04 * th / math.sqrt(max(1e-8, 1.0 - th * th)))
            )
        self.thermal_scale = max(s_floor, min(1.0, scale))
        self.thermal_locked = bool(self.thermal_scale <= 1e-4)
        return self.thermal_scale

    def restore_device_state(self, device: torch.device) -> None:
        """Move restored device-resident chart/ledger payloads onto ``device``."""
        for attr in ("chart_state", "atomic_ledger"):
            val = getattr(self, attr)
            if val is not None:
                setattr(
                    self,
                    attr,
                    _walk_payload(val, lambda tensor: tensor.to(device=device)),
                )

    def state_dict(self) -> Dict[str, Any]:
        """Serialize schema-2 checkpoint state with at most one packed GPU EMA flush."""
        self._flush_ema()
        bank_obj = self._cond_bank_obj
        has_packet = bank_obj is not None and hasattr(bank_obj, "to_packet")
        bank_pkt = bank_obj.to_packet() if has_packet else self.cond_fiber_bank
        result: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "calls": int(self.calls),
            "observed_steps": int(self.observed_steps),
            "last_ratio": None if self.last_ratio is None else float(self.last_ratio),
        }
        for key in _BOOL_FIELDS:
            result[key] = bool(getattr(self, key))
        for key in _SCALAR_BOUNDS:
            result[key] = float(getattr(self, key))
        for key in _DICT_FIELDS:
            val = bank_pkt if key == "cond_fiber_bank" else getattr(self, key)
            result[key] = None if val is None else _walk_payload(val, _clone_to_cpu)
        return result

    @staticmethod
    def validate_state_dict(state: Any) -> Tuple[Optional[float], Dict[str, float]]:
        """Validate a schema-2 checkpoint dictionary without mutating any instance."""
        return validate_axiom_record_state(state)

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        """Compatibility adapter for ``plain_adam`` / ``NormalizedOptimizer`` checkpoint validation."""
        validate_axiom_record_state(state)
        return {
            "lr_max": 0.0,
            "lr_last": None,
            "anchor_started": state["anchor_started"],
            "calls": state["calls"],
            "observed_steps": state["observed_steps"],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Atomically validate and restore a schema-2 checkpoint."""
        last_ratio, scalars = validate_axiom_record_state(state)
        restored_dicts = {
            key: None
            if state[key] is None
            else _walk_payload(state[key], _clone_to_cpu)
            for key in _DICT_FIELDS
        }
        self.schema = SCHEMA_VERSION
        self.calls = state["calls"]
        self.observed_steps = state["observed_steps"]
        self.last_ratio = last_ratio
        self._reset_transient()
        for key in _BOOL_FIELDS:
            setattr(self, key, state[key])
        for key, value in scalars.items():
            setattr(self, key, value)
        for key, value in restored_dicts.items():
            setattr(self, key, value)

    def __copy__(self) -> "AxiomStepRecord":
        dup = AxiomStepRecord()
        dup.load_state_dict(self.state_dict())
        return dup

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AxiomStepRecord":
        dup = self.__copy__()
        memo[id(self)] = dup
        dup._enable_secant_probe = self._enable_secant_probe
        return dup


# ---------------------------------------------------------------------------
# Canonical RNG-Isolated Probe Forward
# ---------------------------------------------------------------------------


def _collect_cuda_devices(
    module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> List[int]:
    if not torch.cuda.is_initialized():
        return []
    devices = {torch.cuda.current_device()}
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device.type == "cuda" and tensor.device.index is not None:
            devices.add(tensor.device.index)
    for item in (*args, *kwargs.values()):
        if (
            isinstance(item, torch.Tensor)
            and item.device.type == "cuda"
            and item.device.index is not None
        ):
            devices.add(item.device.index)
    return sorted(devices)


def _uses_bf16_probe(module: nn.Module) -> bool:
    first_param = next(module.parameters(), None)
    if first_param is None or first_param.device.type != "cuda":
        return False
    return bool(
        getattr(module, "_particlegan_bf16_autocast", False)
        or getattr(module, "_axiom_bf16_autocast", False)
    )


@torch.no_grad()
def eval_probe_forward(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    record: Optional[AxiomStepRecord] = None,
) -> Optional[torch.Tensor]:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` with zero global RNG or BN mutation."""
    modes = [(sub, sub.training) for sub in module.modules()]
    prev_probe = bool(record._in_probe) if record is not None else False
    py_rng = random.getstate()
    np_mod = sys.modules.get("numpy")
    np_rng = (
        np_mod.random.get_state()
        if (np_mod is not None and hasattr(np_mod, "random"))
        else None
    )
    cuda_devices = _collect_cuda_devices(module, args, kwargs)
    if record is not None:
        record._in_probe = True
    try:
        module.eval()
        use_bf16 = _uses_bf16_probe(module)
        with torch.random.fork_rng(devices=cuda_devices):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                target = first_tensor_output(module(*args, **kwargs))
        return (
            target.detach().to(torch.float32)
            if isinstance(target, torch.Tensor)
            else None
        )
    finally:
        if record is not None:
            record._in_probe = prev_probe
        for sub, was_training in modes:
            sub.training = was_training
        random.setstate(py_rng)
        if np_rng is not None and np_mod is not None:
            np_mod.random.set_state(np_rng)


# ---------------------------------------------------------------------------
# Forward Hooks, Source Context & Optimizer/Record Binding Helpers
# ---------------------------------------------------------------------------


def _resolve_critic_target(critic: Any) -> Optional[nn.Module]:
    if not isinstance(critic, nn.Module):
        return None
    if critic in _MODULE_RECORDS:
        return critic
    inner = getattr(critic, "critic", None)
    if isinstance(inner, nn.Module) and inner in _MODULE_RECORDS:
        return inner
    return None


@contextmanager
def axiom_source_context(critic: Any, source: Optional[torch.Tensor]) -> Iterator[None]:
    """Bind pre-transform ``source`` tensor identity for live-endpoint reuse (nestable, attr-free)."""
    target = _resolve_critic_target(critic)
    if target is None or source is None:
        yield
        return
    sentinel = object()
    prev = _MODULE_PENDING_SOURCE.get(target, sentinel)
    _MODULE_PENDING_SOURCE[target] = tensor_id(source)
    try:
        yield
    finally:
        if prev is sentinel:
            _MODULE_PENDING_SOURCE.pop(target, None)
        else:
            _MODULE_PENDING_SOURCE[target] = prev


def _override_existing_entry(
    prev: CaptureEntry,
    x: torch.Tensor,
    rec: Optional[AxiomStepRecord],
    batch_size: int,
) -> CaptureEntry:
    if prev.parts is not None and len(prev.parts) > 1:
        new_parts: Optional[Tuple[torch.Tensor, ...]] = (
            x.reshape(batch_size, -1),
            *prev.parts[1:],
        )
        new_x = torch.cat(new_parts, dim=-1)
    else:
        new_parts = None
        new_x = x
    return CaptureEntry(
        x_in=new_x,
        cond=prev.cond,
        record=rec,
        has_cont_cond=prev.has_cont_cond,
        parts=new_parts,
        thermally_scaled=prev.thermally_scaled,
    )


def override_axiom_capture_input(
    out: Any,
    x: torch.Tensor,
    *,
    critic: Any = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[CaptureEntry]:
    """Replace noisy critic capture input with clean ``x`` (for ``InputNoise.forward``)."""
    target = first_tensor_output(out)
    if not _is_batch_tensor(target, int(x.shape[0])):
        return None
    prev = getattr(target, "_axiom_capture", None)
    has_prev = isinstance(prev, CaptureEntry)
    rec = (
        prev.record
        if (has_prev and prev.record is not None)
        else get_module_record(critic)
    )
    if not has_prev and rec is None:
        return None
    if rec is not None and (rec._in_penalty or rec._in_probe):
        return None
    batch_size = int(target.shape[0])
    if has_prev:
        entry = _override_existing_entry(prev, x, rec, batch_size)
    else:
        x_in, cond, has_cc, parts = extract_axiom_inputs_parts(
            (x, *args), kwargs, batch_size
        )
        if x_in is None:
            return None
        entry = CaptureEntry(
            x_in=x_in, cond=cond, record=rec, has_cont_cond=has_cc, parts=parts
        )
    record_axiom_capture(out, entry)
    return entry


def _should_capture_live_endpoint(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    x0: Optional[torch.Tensor],
    raw: torch.Tensor,
) -> bool:
    if x0 is None or not module.training or not torch.is_grad_enabled():
        return False
    if not raw.requires_grad or x0.requires_grad:
        return False
    return not any(
        isinstance(val, torch.Tensor) and val.requires_grad
        for val in (*args[1:], *kwargs.values())
    )


def _maybe_record_live_endpoint(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    x0: Optional[torch.Tensor],
    raw: torch.Tensor,
    rec: AxiomStepRecord,
) -> None:
    if not _should_capture_live_endpoint(module, args, kwargs, x0, raw):
        return
    src_key = _MODULE_PENDING_SOURCE.get(module) or tensor_id(x0)
    if src_key is not None and x0 is not None:
        rec.captures.record_live_endpoint(
            step=rec.observed_steps,
            source_key=src_key,
            cond_sig=cond_signature(args, kwargs),
            x_in=x0,
            raw_logits=raw,
        )


def _maybe_scale_thermal_output(
    output: Any,
    raw: torch.Tensor,
    rec: AxiomStepRecord,
) -> Tuple[Any, torch.Tensor, bool]:
    if not (torch.is_grad_enabled() and raw.requires_grad and rec.thermal_scale < 1.0):
        return output, raw, False
    scale = float(rec.thermal_scale)
    tgt = raw * scale + raw.detach() * (1.0 - scale)
    return _replace_first_output(output, tgt), tgt, True


def _axiom_critic_hook(
    module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
) -> Any:
    rec = _MODULE_RECORDS.get(module)
    if rec is None:
        return None
    x0 = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    if rec._in_probe or rec._in_penalty:
        if rec._in_penalty and x0 is not None:
            rec._last_probe_actual_input = x0.detach()
        return None
    raw = first_tensor_output(output)
    if not isinstance(raw, torch.Tensor) or raw.ndim < 1:
        return None
    _maybe_record_live_endpoint(module, args, kwargs, x0, raw, rec)
    output, tgt, scaled = _maybe_scale_thermal_output(output, raw, rec)
    x_in, cond, has_cc, parts = extract_axiom_inputs_parts(
        args, kwargs, int(tgt.shape[0])
    )
    if x_in is not None:
        entry = CaptureEntry(
            x_in=x_in,
            cond=cond,
            record=rec,
            has_cont_cond=has_cc,
            parts=parts,
            thermally_scaled=scaled,
        )
        record_axiom_capture(output, entry)
    return output if scaled else None


def _select_stratified_probe_indices(
    raw: torch.Tensor, phase: float, n_probe: int
) -> torch.Tensor:
    batch_size = int(raw.shape[0])
    dev = raw.device
    if batch_size <= n_probe:
        return torch.arange(batch_size, device=dev, dtype=torch.long)
    n_unif = n_probe // 2
    offset = int(float(phase) * batch_size)
    unif_idx = (
        (torch.arange(n_unif, device=dev, dtype=torch.long) * batch_size) // n_unif
        + offset
    ) % batch_size
    norms = (
        raw.detach()
        .reshape(batch_size, -1)
        .double()
        .norm(dim=-1)
        .index_fill(0, unif_idx, float("-inf"))
    )
    tail_idx = torch.topk(norms, k=n_probe - n_unif).indices
    return torch.cat((unif_idx, tail_idx), dim=0)


def _record_allows_probe(rec: Optional[AxiomStepRecord]) -> bool:
    if rec is None or not rec._enable_secant_probe:
        return False
    return not (rec._in_probe or rec._in_penalty or rec._in_protected_probe)


def _probe_batch_cap(raw: Any) -> int:
    if (
        not isinstance(raw, torch.Tensor)
        or not raw.is_floating_point()
        or not raw.requires_grad
    ):
        return 0
    if raw.ndim < 2 or raw.shape[0] < 1 or fwAD.unpack_dual(raw).tangent is not None:
        return 0
    per_sample_dim = max(1, int(raw[0].numel()))
    cap = min(_PROBE_MAX_BATCH, _PROBE_MAX_ELEMENTS // per_sample_dim)
    return cap if cap >= 4 else 0


def _axiom_generator_hook(
    module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
) -> None:
    rec = _MODULE_RECORDS.get(module)
    if (
        not _record_allows_probe(rec)
        or not module.training
        or not torch.is_grad_enabled()
    ):
        return
    raw = first_tensor_output(output)
    probe_cap = _probe_batch_cap(raw)
    if probe_cap <= 0 or rec is None:
        return
    batch_size = int(raw.shape[0])
    idx = _select_stratified_probe_indices(raw, rec.chord_phase, probe_cap)
    probe_args = tuple(_index_arg(arg, batch_size, idx) for arg in args)
    probe_kwargs = {
        key: _index_arg(val, batch_size, idx) for key, val in kwargs.items()
    }
    if _MODULE_HAS_BN.get(module, False):
        y_pre = eval_probe_forward(module, probe_args, probe_kwargs, rec)
    else:
        y_pre = raw.detach().index_select(0, idx).to(torch.float32)
    if y_pre is not None:
        rec._g_probe_cache = (module, probe_args, probe_kwargs, y_pre)


def _attach_axiom_critic_hook(
    critic: Optional[nn.Module], record: AxiomStepRecord
) -> None:
    """Attach the Axiom critic capture/thermal forward hook without mutating ``critic`` attributes."""
    if not isinstance(critic, nn.Module):
        return
    _MODULE_RECORDS[critic] = record
    _MODULE_HAS_BN[critic] = _module_has_bn_or_dropout(critic)
    if critic not in _CRITIC_HOOK_HANDLES:
        _CRITIC_HOOK_HANDLES[critic] = critic.register_forward_hook(
            _axiom_critic_hook, with_kwargs=True
        )


def _attach_axiom_generator_hook(
    generator: Optional[nn.Module], record: AxiomStepRecord
) -> None:
    """Attach the Axiom generator stratified trust-probe hook without mutating ``generator`` attributes."""
    if not isinstance(generator, nn.Module):
        return
    _MODULE_RECORDS[generator] = record
    _MODULE_HAS_BN[generator] = _module_has_bn_or_dropout(generator)
    if generator not in _GEN_HOOK_HANDLES:
        _GEN_HOOK_HANDLES[generator] = generator.register_forward_hook(
            _axiom_generator_hook, with_kwargs=True
        )


attach_axiom_critic_hook = _attach_axiom_critic_hook
attach_axiom_generator_hook = _attach_axiom_generator_hook


def _bind_single_reader(record: AxiomStepRecord, reader: Any) -> None:
    if isinstance(reader, torch.Tensor):
        set_tensor_record(reader, record)
        return
    if isinstance(reader, nn.Module):
        set_module_record(reader, record)
        for attr in ("record", "axiom_record", "_axiom_record"):
            if hasattr(reader, attr):
                setattr(reader, attr, record)
        return
    is_gen_opt = (
        hasattr(reader, "param_groups") and getattr(reader, "critic", None) is None
    )
    reader._axiom_record = record
    if hasattr(reader, "record") and not is_gen_opt:
        reader.record = record
    if hasattr(reader, "axiom_record"):
        reader.axiom_record = record


def bind_axiom_record(record: AxiomStepRecord, *readers: Any) -> None:
    """Bind ``record`` cleanly to Axiom components, user modules, or tensors."""
    for reader in readers:
        if reader is not None:
            _bind_single_reader(record, reader)


def _bind_generator_side(
    opt_g: Any,
    generator: Optional[nn.Module],
    encoder: Optional[nn.Module],
    record: AxiomStepRecord,
    enable_secant_probe: bool,
) -> None:
    if hasattr(opt_g, "bind_module_layouts"):
        opt_g.bind_module_layouts((generator, encoder))
    if generator is None:
        return
    if hasattr(opt_g, "set_generator_probe_parameters"):
        opt_g.set_generator_probe_parameters(generator.parameters())
    elif hasattr(opt_g, "_gen_param_ids"):
        opt_g._gen_param_ids = {id(param) for param in generator.parameters()}
    if enable_secant_probe:
        record._enable_secant_probe = True
        _attach_axiom_generator_hook(generator, record)


def bind_axiom_optimizers(
    opt_g: Any,
    opt_d: Any,
    *,
    generator: Optional[nn.Module] = None,
    critic: Optional[nn.Module] = None,
    encoder: Optional[nn.Module] = None,
    prior: Optional[nn.Module] = None,
    base_prior: Optional[nn.Module] = None,
    enable_secant_probe: bool = False,
) -> Optional[AxiomStepRecord]:
    """Bind ``opt_d.record`` across optimizers, modules, hooks, and layouts in one seam call."""
    record = getattr(opt_d, "record", None)
    if not isinstance(record, AxiomStepRecord):
        return None
    bind_axiom_record(record, opt_g, opt_d, prior, base_prior)
    for mod in (prior, base_prior):
        table = getattr(mod, "z", None)
        if isinstance(table, torch.Tensor):
            set_tensor_record(table, record)
    _bind_generator_side(opt_g, generator, encoder, record, enable_secant_probe)
    if critic is not None:
        if hasattr(opt_d, "bind_module_layouts"):
            opt_d.bind_module_layouts((critic,))
        _attach_axiom_critic_hook(critic, record)
    return record


__all__ = [
    "AxiomStepRecord",
    "CaptureEntry",
    "CaptureRegistry",
    "SCHEMA_VERSION",
    "WeakTensorMap",
    "_attach_axiom_critic_hook",
    "_attach_axiom_generator_hook",
    "attach_axiom_critic_hook",
    "attach_axiom_generator_hook",
    "axiom_source_context",
    "bind_axiom_optimizers",
    "bind_axiom_record",
    "capture_key",
    "check_scalar",
    "clear_active_prior_rows",
    "cond_signature",
    "eval_probe_forward",
    "extract_axiom_inputs_parts",
    "first_tensor_output",
    "format_logits_2d",
    "get_active_prior_rows",
    "get_axiom_record",
    "get_module_record",
    "get_tensor_record",
    "lookup_axiom_capture",
    "override_axiom_capture_input",
    "record_axiom_capture",
    "recover_active_prior_rows",
    "set_active_prior_rows",
    "set_module_record",
    "set_tensor_record",
    "tensor_id",
    "validate_axiom_record_state",
]
