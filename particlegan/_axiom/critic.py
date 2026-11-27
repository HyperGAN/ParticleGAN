"""Axiom Module B: Law 2 Critic Potential Geometry, Law 4.5 Normal Dissipation, and StepRecord."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

from ..discriminators import BatchDistanceDiscriminator

_GOLDEN_CONJ = 0.6180339887498949
_MAX_LIVE_CAPTURES = 6
_MAX_CAPTURE_RING = 16
_PROBE_MAX_BATCH = 32
_FWAD_EXACT_MAX_DIM = 16
_MAX_VERIFIED_CRITICS = 256

_EMA_BOOLS = frozenset({"grad_sq_initialized", "margin_initialized"})
_BOOL_FIELDS = (
    "anchor_started",
    "grad_sq_initialized",
    "margin_initialized",
    "ell_initialized",
    "thermal_initialized",
    "thermal_locked",
    "cond_initialized",
    "het_initialized",
)
_THERMAL_KEYS = ("ema_ed", "ema_mu_sq", "ema_spread", "ema_q90", "ema_hmax", "ema_tail")
_SCALAR_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float], bool, bool]] = {
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
    **{k: (None, None, False, False) for k in _THERMAL_KEYS},
}
_DICT_FIELDS = ("chart_state", "atomic_ledger", "cond_fiber_bank")
_RECORD_KEYS = frozenset(
    {"schema", "calls", "observed_steps", "last_ratio", *_BOOL_FIELDS, *_SCALAR_BOUNDS, *_DICT_FIELDS}
)

_SECANT_MODULE_TYPES = (
    nn.MultiheadAttention,
    nn.Transformer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    nn.Softmax,
    nn.Softmax2d,
    nn.LogSoftmax,
    nn.Softmin,
)
_SOFTMAX_BACKWARD_PREFIXES = (
    "SoftmaxBackward",
    "SafeSoftmaxBackward",
    "LogSoftmaxBackward",
)


def _first_out(out: Any) -> Any:
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    return out


def _fmt_logits(y: torch.Tensor, batch_size: int) -> torch.Tensor:
    if y.ndim < 2:
        return y.reshape(batch_size, 1)
    return y.reshape(batch_size, -1)


def _storage_ptr(t: Any) -> int:
    if not isinstance(t, torch.Tensor) or t.ndim < 1 or t.numel() == 0:
        return 0
    try:
        return int(t.untyped_storage().data_ptr())
    except RuntimeError:
        return 0


def _tensor_id(t: Any) -> Optional[Tuple[int, int, Tuple[int, ...], str]]:
    ptr = _storage_ptr(t)
    if not ptr:
        return None
    return (ptr, int(t.storage_offset()), tuple(int(s) for s in t.shape), str(t.device))


def _capture_key(t: Any) -> Optional[Tuple[int, int, int, int]]:
    ptr = _storage_ptr(t)
    if not ptr:
        return None
    return (ptr, int(t.storage_offset()), int(t.numel()), int(t.shape[0]))


def _cond_sig(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    sig: List[Any] = []
    for x in args[1:]:
        if isinstance(x, torch.Tensor):
            sig.append(("T", _tensor_id(x), str(x.dtype)))
        else:
            sig.append(("V", type(x).__qualname__, repr(x)))
    for k in sorted(kwargs):
        val = kwargs[k]
        if isinstance(val, torch.Tensor):
            sig.append(("KT", k, _tensor_id(val), str(val.dtype)))
        else:
            sig.append(("KV", k, type(val).__qualname__, repr(val)))
    return tuple(sig)


def _walk_payload(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _walk_payload(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_walk_payload(v, fn) for v in obj)
    return deepcopy(obj)


def _validate_payload(obj: Any, name: str) -> None:
    if obj is None or type(obj) in (bool, int, str):
        return
    if type(obj) is float:
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in {name}")
    elif isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
            raise ValueError(f"non-finite tensor in {name}")
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _validate_payload(item, name)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int, tuple)):
                raise ValueError(f"unsupported key in {name}")
            _validate_payload(v, name)
    else:
        raise ValueError(f"unsupported type {type(obj).__name__} in {name}")


def _check_scalar(
    val: Any,
    name: str,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    slo: bool = False,
    shi: bool = False,
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ValueError(f"invalid {name}: {val!r}")
    fval = float(val)
    if lo is not None and (fval <= lo if slo else fval < lo):
        raise ValueError(f"{name}={fval} outside bounds")
    if hi is not None and (fval >= hi if shi else fval > hi):
        raise ValueError(f"{name}={fval} outside bounds")
    return fval


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


class AxiomStepRecord:
    """Standalone checkpointed state linking Axiom critic, transport, prior, and optimizer."""

    ema_grad_sq = _ema_prop("ema_grad_sq")
    grad_sq_initialized = _ema_prop("grad_sq_initialized")
    ema_margin = _ema_prop("ema_margin")
    ema_data_scale = _ema_prop("ema_data_scale")
    last_data_scale = _ema_prop("last_data_scale")
    ema_push_res = _ema_prop("ema_push_res")
    margin_initialized = _ema_prop("margin_initialized")
    ell_ema = _ema_prop("ell_ema")

    def __init__(self) -> None:
        self.schema = 1
        self.calls = 0
        self.observed_steps = 0
        self.anchor_started = False
        self.last_ratio: Optional[float] = None
        self.chord_phase = 0.0
        self._ema: Dict[str, Any] = {
            "ema_grad_sq": 1.0,
            "grad_sq_initialized": False,
            "ema_margin": 0.0,
            "ema_data_scale": 1.0,
            "last_data_scale": 1.0,
            "ema_push_res": 1.0,
            "margin_initialized": False,
            "ell_ema": 0.0,
        }
        self.ell_initialized = False
        self.thermal_initialized = False
        self.thermal_locked = False
        self.cond_initialized = False
        self.het_initialized = False
        self.thermal_scale = 1.0
        self.step_scale = 1.0
        self.het_ema = 1.0
        self.g_func_gain = 1.0
        self.ema_ed = 1.0
        self.ema_mu_sq = 1.0
        self.ema_spread = 1.0
        self.ema_q90 = 1.0
        self.ema_hmax = 1.0
        self.ema_tail = 0.0
        self.cond_ema = 0.25
        self.chart_state: Optional[Dict[str, Any]] = None
        self.atomic_ledger: Optional[Dict[str, Any]] = None
        self.cond_fiber_bank: Optional[Dict[str, Any]] = None
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._ema_t: Dict[str, torch.Tensor] = {}
        self._live_captures: List[List[Any]] = []
        self._capture_ring: OrderedDict[Tuple[int, int, int, int], Tuple[Any, ...]] = OrderedDict()
        self._in_penalty = False
        self._in_probe = False
        self._in_protected_probe = False
        self._simplicial_step = False
        self._enable_secant_probe = True
        self._disc_gate = 0.0
        self._last_probe_actual_input: Optional[torch.Tensor] = None
        self._g_probe_cache: Any = None
        self._active_prior_table: Optional[torch.Tensor] = None
        self._active_batch_rows: Optional[torch.Tensor] = None
        self._cond_bank_obj: Any = None

    def lookup_capture(self, logits: Any) -> Optional[Tuple[Any, ...]]:
        key = _capture_key(logits)
        if key is None:
            return None
        return self._capture_ring.get(key)

    def take_pending_ema_tensors(self) -> Tuple[List[str], List[torch.Tensor]]:
        return list(self._ema_t.keys()), list(self._ema_t.values())

    def record_step(self, optimizer: Any = None) -> None:
        del optimizer
        self.observed_steps += 1
        self.clear_transient_captures()

    def __copy__(self) -> "AxiomStepRecord":
        dup = AxiomStepRecord()
        dup.load_state_dict(self.state_dict())
        return dup

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AxiomStepRecord":
        dup = self.__copy__()
        memo[id(self)] = dup
        dup._enable_secant_probe = self._enable_secant_probe
        return dup

    def apply_flushed_ema_values(self, keys: List[str], vals: List[float]) -> None:
        for k, v in zip(keys, vals):
            if k in self._ema:
                self._ema[k] = bool(v > 0.5) if k in _EMA_BOOLS else float(v)
                self._ema_t.pop(k, None)

    def _flush_ema(self) -> None:
        ks, ts = self.take_pending_ema_tensors()
        if not ts:
            return
        if all(t.device == ts[0].device for t in ts):
            vals = torch.stack([t.double() for t in ts]).tolist()
        else:
            vals = [float(t.item()) for t in ts]
        self.apply_flushed_ema_values(ks, vals)

    def _dev_ema(self, name: str, dev: torch.device) -> torch.Tensor:
        cur = self._ema_t.get(name)
        if cur is not None and cur.device == dev:
            return cur
        if cur is not None:
            self._flush_ema()
        return torch.full((), float(self._ema[name]), dtype=torch.float64, device=dev)

    @torch.no_grad()
    def _observe_grad_sq_tensor(self, s_hat_det: torch.Tensor) -> None:
        obs = s_hat_det.detach().double().clamp(0.0, 20.0)
        fin = torch.isfinite(obs)
        prev = self._dev_ema("ema_grad_sq", obs.device)
        pi = self._dev_ema("grad_sq_initialized", obs.device)
        updated = torch.where(pi > 0.5, 0.95 * prev + 0.05 * obs, obs)
        self._ema_t["ema_grad_sq"] = torch.where(fin, updated, prev)
        self._ema_t["grad_sq_initialized"] = torch.where(fin, pi.new_full((), 1.0), pi)

    @torch.no_grad()
    def observe_margin_and_scale(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        x_real: Optional[torch.Tensor] = None,
        x_fake: Optional[torch.Tensor] = None,
    ) -> None:
        if self._in_penalty or self._in_probe or self._in_protected_probe:
            return
        dev = real_logits.device
        margin = (
            real_logits.detach().float().mean().double()
            - fake_logits.detach().float().mean().double()
        ).clamp(-50.0, 50.0)
        fm = torch.isfinite(margin)
        pm = self._dev_ema("ema_margin", dev)
        pi = self._dev_ema("margin_initialized", dev)
        was_init = pi > 0.5
        self._ema_t["ema_margin"] = torch.where(
            fm,
            torch.where(was_init, 0.90 * pm + 0.10 * margin, margin),
            pm,
        )
        self._ema_t["margin_initialized"] = torch.where(fm, pi.new_full((), 1.0), pi)
        if x_real is None or x_real.numel() == 0:
            return
        xr = x_real.detach().reshape(x_real.shape[0], -1).double()
        xm = xr.abs().amax()
        sx = torch.where(xm > 0.0, xm, xm.new_full((), 1.0))
        sc = ((xr / sx).square().sum(-1).mean().sqrt() * sx).clamp_min(1e-30)
        psc = self._dev_ema("ema_data_scale", dev)
        lsc = self._dev_ema("last_data_scale", dev)
        lp = torch.log2(psc.clamp_min(1e-30))
        lc = torch.log2(sc.clamp_min(1e-30))
        ll = torch.log2(lsc.clamp_min(1e-30))
        ok_s = torch.isfinite(sc) & (xm > 0.0)
        dl = lc - lp
        step_l = torch.where(
            (dl.abs() > 0.25) & ((lc - ll).abs() <= 1.5),
            0.80 * dl,
            0.05 * dl.clamp(-1.0, 1.0),
        )
        nsc = torch.where(ok_s, torch.where(was_init, torch.exp2(lp + step_l), sc), psc)
        self._ema_t["ema_data_scale"] = nsc
        self._ema_t["last_data_scale"] = torch.where(
            ok_s,
            torch.where(was_init, torch.exp2(ll + (lc - ll).clamp(-2.5, 2.5)), sc),
            lsc,
        )
        if x_fake is not None and x_fake.shape == x_real.shape:
            xf = x_fake.detach().reshape(x_fake.shape[0], -1).double()
            rg = (
                ((xf.mean(0) - xr.mean(0)).norm() + 0.5 * (xf.std() - xr.std()).abs())
                / nsc.clamp_min(1e-30)
            ).clamp_max(10.0)
            ppr = self._dev_ema("ema_push_res", dev)
            self._ema_t["ema_push_res"] = torch.where(
                torch.isfinite(rg),
                torch.where(was_init, 0.92 * ppr + 0.08 * rg, rg),
                ppr,
            )

    @property
    def thermal_ema(self) -> float:
        ud = min(1.0, max(0.0, -float(self.ema_spread)) / 0.03)
        q9 = float(self.ema_q90)
        hm = float(self.ema_hmax)
        return max(
            0.5 * max(0.0, float(self.ema_ed)),
            math.sqrt(max(0.0, float(self.ema_mu_sq))),
            0.60 * abs(float(self.ema_spread)),
            max(q9, 0.60 * max(0.0, -q9) * ud),
            max(hm, 0.60 * max(0.0, -hm) * ud),
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
        ed_f = float(r_ed)
        mu_f = float(r_mu_sq)
        tau_f = float(tau)
        tp = max(0.0, tau_f)
        cl = max(0.20, 2.0 * tp)
        ce = max(0.70, 4.0 * tp)
        sed = max(float(shock_ed_floor), 4.0 * tp * tp)
        vals = (
            max(-ce, min(ed_f, ce)),
            max(-cl * cl, min(mu_f, cl * cl)),
            max(-cl, min(float(r_spread), cl)),
            max(-cl, min(float(r_q90), cl)),
            max(-cl, min(float(r_hmax), cl)),
            max(-cl, min(float(r_tail), cl)),
        )
        if not self.thermal_initialized or (
            self.thermal_scale < 0.99 and (ed_f > sed or mu_f > shock_mu_ratio * sed)
        ):
            for k, v in zip(_THERMAL_KEYS, vals):
                setattr(self, k, v)
            self.thermal_initialized = True
            self.thermal_locked = False
        else:
            d = float(ema_decay)
            om = 1.0 - d
            for k, v in zip(_THERMAL_KEYS, vals):
                setattr(self, k, d * getattr(self, k) + om * v)
        if tau_f <= 0.0:
            self.thermal_scale = 1.0
            self.thermal_locked = False
            return 1.0
        tlo = min(0.92, 0.50 + (float(lock_hysteresis) if self.thermal_locked else 0.0)) * tau_f
        u = min(1.0, max(0.0, (self.thermal_ema - tlo) / max(1e-8, tau_f - tlo)))
        th = u * u * (3.0 - 2.0 * u)
        sm = max(1e-5, float(s_min))
        if th >= 1.0:
            s = 1.0
        elif th <= 0.0:
            s = sm
        else:
            s = max(sm, min(1.0, 0.04 * th / math.sqrt(max(1e-8, 1.0 - th * th))))
        self.thermal_scale = max(sm, min(1.0, s))
        self.thermal_locked = bool(s <= 1e-4)
        return self.thermal_scale

    def clear_transient_captures(self) -> None:
        for entry in self._live_captures:
            entry[3] = None
            entry[4] = None
            entry[5].clear()
        self._live_captures.clear()
        self._capture_ring.clear()
        self._last_probe_actual_input = None

    def restore_device_state(self, device: torch.device) -> None:
        for attr in ("chart_state", "atomic_ledger"):
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, _walk_payload(val, lambda t: t.to(device=device)))

    def state_dict(self) -> Dict[str, Any]:
        self._flush_ema()

        def _to_cpu(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu().clone()

        if self._cond_bank_obj is not None and hasattr(self._cond_bank_obj, "to_packet"):
            bank = self._cond_bank_obj.to_packet()
        else:
            bank = self.cond_fiber_bank
        result: Dict[str, Any] = {
            "schema": 1,
            "calls": int(self.calls),
            "observed_steps": int(self.observed_steps),
            "last_ratio": None if self.last_ratio is None else float(self.last_ratio),
        }
        for k in _BOOL_FIELDS:
            result[k] = bool(getattr(self, k))
        for k in _SCALAR_BOUNDS:
            result[k] = float(getattr(self, k))
        for dk in _DICT_FIELDS:
            val = bank if dk == "cond_fiber_bank" else getattr(self, dk)
            result[dk] = None if val is None else _walk_payload(val, _to_cpu)
        return result

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if (
            not isinstance(state, dict)
            or set(state) != _RECORD_KEYS
            or type(state["schema"]) is not int
            or state["schema"] != 1
            or any(type(state[ik]) is not int or state[ik] < 0 for ik in ("calls", "observed_steps"))
            or any(type(state[bk]) is not bool for bk in _BOOL_FIELDS)
            or state["anchor_started"] is not False
        ):
            raise ValueError("invalid AxiomStepRecord checkpoint")
        lr = (
            None
            if state["last_ratio"] is None
            else _check_scalar(state["last_ratio"], "last_ratio", 0.0)
        )
        scalars = {k: _check_scalar(state[k], k, *bounds) for k, bounds in _SCALAR_BOUNDS.items()}
        for dk in _DICT_FIELDS:
            if state[dk] is not None:
                if not isinstance(state[dk], dict):
                    raise ValueError(f"AxiomStepRecord.{dk} must be dict or None")
                _validate_payload(state[dk], dk)

        def _to_cpu(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu().clone()

        self.schema = 1
        self.calls = state["calls"]
        self.observed_steps = state["observed_steps"]
        self.last_ratio = lr
        for bk in _BOOL_FIELDS:
            setattr(self, bk, state[bk])
        for k, v in scalars.items():
            setattr(self, k, v)
        for dk in _DICT_FIELDS:
            setattr(self, dk, None if state[dk] is None else _walk_payload(state[dk], _to_cpu))
        self._reset_transient()


def _extract_axiom_inputs_parts(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool, Optional[Tuple[torch.Tensor, ...]]]:
    floats: List[torch.Tensor] = []
    discretes: List[torch.Tensor] = []
    items = [*args, *(kwargs[k] for k in sorted(kwargs))] if kwargs else list(args)
    for item in items:
        if isinstance(item, torch.Tensor) and item.ndim >= 1 and item.shape[0] == batch_size and item.numel() > 0:
            if item.is_floating_point():
                floats.append(item)
            elif not item.is_complex():
                discretes.append(item.detach().to(torch.int64).reshape(batch_size, -1))
    if not floats:
        return None, None, False, None
    if not discretes:
        cond = None
    elif len(discretes) == 1:
        cond = discretes[0]
    else:
        cond = torch.cat(discretes, dim=-1)
    parts = tuple(t.reshape(batch_size, -1) for t in floats) if len(floats) > 1 else None
    merged = floats[0] if parts is None else torch.cat(parts, dim=-1)
    return merged, cond, parts is not None, parts


def _record_axiom_capture(
    out: Any,
    x_in: Optional[torch.Tensor],
    cond: Optional[torch.Tensor],
    rec: Any,
    *,
    has_cont_cond: bool = False,
    parts: Optional[Tuple[torch.Tensor, ...]] = None,
    thermally_scaled: bool = False,
) -> None:
    tgt = _first_out(out)
    if x_in is None or not isinstance(tgt, torch.Tensor) or tgt.ndim < 1 or tgt.shape[0] != x_in.shape[0]:
        return
    tgt._axiom_input = x_in  # type: ignore[attr-defined]
    tgt._axiom_record = rec  # type: ignore[attr-defined]
    tgt._axiom_cond = cond  # type: ignore[attr-defined]
    tgt._axiom_cont_cond = bool(has_cont_cond)  # type: ignore[attr-defined]
    tgt._axiom_parts = parts  # type: ignore[attr-defined]
    tgt._axiom_thermally_scaled = bool(thermally_scaled)  # type: ignore[attr-defined]
    if rec is None or int(x_in[0].numel()) > 4096:
        return
    ring = getattr(rec, "_capture_ring", None)
    ck = _capture_key(tgt)
    if isinstance(ring, OrderedDict) and ck is not None:
        ring[ck] = (
            x_in.detach(),
            None if cond is None else cond.detach(),
            rec,
            bool(has_cont_cond),
            None if parts is None else tuple(p.detach() for p in parts),
        )
        ring.move_to_end(ck)
        while len(ring) > _MAX_CAPTURE_RING:
            ring.popitem(last=False)


@torch.no_grad()
def eval_probe_forward(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    record: Optional[AxiomStepRecord] = None,
) -> Optional[torch.Tensor]:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` without mutating RNGs, hooks, or BN buffers."""
    modes = [(m, m.training) for m in module.modules()]
    prev_p = bool(getattr(record, "_in_probe", False)) if record is not None else False
    cpu_r = torch.get_rng_state()
    py_r = random.getstate()
    np_m = sys.modules.get("numpy")
    np_r = np_m.random.get_state() if (np_m is not None and hasattr(np_m, "random")) else None
    c_dev = torch.cuda.current_device() if torch.cuda.is_initialized() else None
    c_r = torch.cuda.get_rng_state(c_dev) if c_dev is not None else None
    if record is not None:
        record._in_probe = True
    try:
        module.eval()
        p0 = next(module.parameters(), None)
        use_bf16 = bool(
            p0 is not None
            and p0.device.type == "cuda"
            and (
                getattr(module, "_axiom_bf16_autocast", False)
                or getattr(module, "_particlegan_bf16_autocast", False)
            )
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            tgt = _first_out(module(*args, **kwargs))
        return tgt.detach().to(torch.float32) if isinstance(tgt, torch.Tensor) else None
    finally:
        if record is not None:
            record._in_probe = prev_p
        for m, flag in modes:
            m.training = flag
        torch.set_rng_state(cpu_r)
        random.setstate(py_r)
        if c_r is not None and c_dev is not None:
            torch.cuda.set_rng_state(c_r, c_dev)
        if np_r is not None and np_m is not None:
            np_m.random.set_state(np_r)


def _axiom_critic_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    rec = getattr(module, "_axiom_record", None)
    x0 = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    if rec is not None and (rec._in_probe or rec._in_penalty):
        if rec._in_penalty and x0 is not None:
            rec._last_probe_actual_input = x0.detach()
        return None
    raw = _first_out(output)
    if not isinstance(raw, torch.Tensor) or raw.ndim < 1:
        return None
    if (
        rec is not None
        and x0 is not None
        and module.training
        and torch.is_grad_enabled()
        and raw.requires_grad
        and not x0.requires_grad
        and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in (*args[1:], *kwargs.values()))
    ):
        source_key = getattr(module, "_axiom_pending_source_key", None) or _tensor_id(x0)
        if source_key is not None:
            alive = [True]
            raw.register_hook(lambda _g, _a=alive: _a.clear())
            cap_val = raw.float() if raw.dtype == torch.bfloat16 else raw
            rec._live_captures.append(
                [rec.observed_steps, source_key, _cond_sig(args, kwargs), x0.detach(), cap_val, alive]
            )
            if len(rec._live_captures) > _MAX_LIVE_CAPTURES:
                for entry in rec._live_captures[:-_MAX_LIVE_CAPTURES]:
                    entry[3] = None
                    entry[4] = None
                    entry[5].clear()
                del rec._live_captures[:-_MAX_LIVE_CAPTURES]
    modified = False
    thermally_scaled = False
    tgt = raw
    if rec is not None and torch.is_grad_enabled() and raw.requires_grad and rec.thermal_scale < 1.0:
        s = float(rec.thermal_scale)
        tgt = raw * s + raw.detach() * (1.0 - s)
        thermally_scaled = True
        modified = True
        if isinstance(output, torch.Tensor):
            output = tgt
        elif isinstance(output, tuple):
            output = (tgt, *output[1:])
        else:
            output = [tgt, *output[1:]]
    x_in, cd, hc, parts = _extract_axiom_inputs_parts(args, kwargs, int(tgt.shape[0]))
    _record_axiom_capture(
        output,
        x_in,
        cd,
        rec,
        has_cont_cond=hc,
        parts=parts,
        thermally_scaled=thermally_scaled,
    )
    return output if modified else None


def _index_arg(val: Any, batch_size: int, idx: torch.Tensor) -> Any:
    if isinstance(val, torch.Tensor):
        if val.ndim >= 1 and val.shape[0] == batch_size:
            return val.detach().index_select(0, idx.to(val.device))
        return val.detach()
    if isinstance(val, (tuple, list)):
        return type(val)(_index_arg(v, batch_size, idx) for v in val)
    if isinstance(val, dict):
        return {k: _index_arg(v, batch_size, idx) for k, v in val.items()}
    return val


def _axiom_generator_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    rec = getattr(module, "_axiom_record", None)
    raw = _first_out(output)
    if (
        rec is None
        or rec._in_probe
        or rec._in_penalty
        or rec._in_protected_probe
        or not rec._enable_secant_probe
        or not module.training
        or not torch.is_grad_enabled()
        or not isinstance(raw, torch.Tensor)
        or not raw.is_floating_point()
        or not raw.requires_grad
        or raw.ndim < 2
        or raw.shape[0] < 1
        or int(raw[0].numel()) > 32
        or fwAD.unpack_dual(raw).tangent is not None
    ):
        return
    b = int(raw.shape[0])
    dev = raw.device
    nu = _PROBE_MAX_BATCH // 2
    if b <= _PROBE_MAX_BATCH:
        idx = torch.arange(b, device=dev, dtype=torch.long)
    else:
        u = ((torch.arange(nu, device=dev, dtype=torch.long) * b) // nu + int(float(rec.chord_phase) * b)) % b
        norms = raw.detach().reshape(b, -1).double().norm(dim=-1).index_fill(0, u, float("-inf"))
        top = torch.topk(norms, k=_PROBE_MAX_BATCH - nu).indices
        idx = torch.cat((u, top), dim=0)
    pa = tuple(_index_arg(a, b, idx) for a in args)
    pk = {k: _index_arg(v, b, idx) for k, v in kwargs.items()}
    if getattr(module, "_axiom_has_bn", False):
        y_pre = eval_probe_forward(module, pa, pk, rec)
    else:
        y_pre = raw.detach().index_select(0, idx).to(torch.float32)
    if y_pre is not None:
        rec._g_probe_cache = (module, pa, pk, y_pre)


def _attach_hook(mod: Optional[nn.Module], record: AxiomStepRecord, fn: Any, flag: str) -> None:
    if isinstance(mod, nn.Module):
        mod._axiom_record = record  # type: ignore[attr-defined]
        mod._axiom_has_bn = any(  # type: ignore[attr-defined]
            isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd))
            for m in mod.modules()
        )
        if not getattr(mod, flag, False):
            mod.register_forward_hook(fn, with_kwargs=True)
            setattr(mod, flag, True)


def _attach_axiom_critic_hook(critic: Optional[nn.Module], record: AxiomStepRecord) -> None:
    _attach_hook(critic, record, _axiom_critic_hook, "_axiom_hook_registered")


def _attach_axiom_generator_hook(generator: Optional[nn.Module], record: AxiomStepRecord) -> None:
    _attach_hook(generator, record, _axiom_generator_hook, "_axiom_gen_hook_registered")


def bind_axiom_record(record: AxiomStepRecord, *readers: Any) -> None:
    for reader in readers:
        if reader is not None:
            reader._axiom_record = record
            if hasattr(reader, "record") and (
                not hasattr(reader, "param_groups") or getattr(reader, "critic", None) is not None
            ):
                reader.record = record


def _needs_secant_module(mod: Any) -> bool:
    if not isinstance(mod, nn.Module):
        return False
    return any(isinstance(m, _SECANT_MODULE_TYPES) for m in mod.modules())


def _has_batch_distance(mod: Any) -> bool:
    if not isinstance(mod, nn.Module):
        return False
    return any(isinstance(m, BatchDistanceDiscriminator) for m in mod.modules())


def _tangent_has_inplace_softmax_bug(tangent: Optional[torch.Tensor]) -> bool:
    """Detect PyTorch's fwAD+backward inplace-Exp bug in functional softmax / math SDPA."""
    if tangent is None or tangent.grad_fn is None:
        return False
    queue: List[Any] = [tangent.grad_fn]
    seen: set[int] = {id(tangent.grad_fn)}
    while queue and len(seen) <= 512:
        fn = queue.pop(0)
        if type(fn).__name__.startswith(_SOFTMAX_BACKWARD_PREFIXES):
            return True
        for nxt, _ in getattr(fn, "next_functions", ()):
            if nxt is not None and id(nxt) not in seen:
                seen.add(id(nxt))
                queue.append(nxt)
    return False


def _bn_dual_bf16_hook(_m: nn.Module, _i: Tuple[Any, ...], out: Any) -> Any:
    if isinstance(out, torch.Tensor):
        primal, tangent = fwAD.unpack_dual(out)
        if tangent is not None and (primal.dtype != torch.bfloat16 or tangent.dtype != torch.bfloat16):
            return fwAD.make_dual(primal.to(torch.bfloat16), tangent.to(torch.bfloat16))
    return out


def _dense_rademacher(x: torch.Tensor, phi: float, salt: int) -> torch.Tensor:
    seed_mix = (int(phi * 1073741824.0) + salt * 7919) & 0xFFFFFFFF
    h = (
        torch.arange(x.numel(), device=x.device, dtype=torch.int64) * 2654435761
        + seed_mix * 3266489917
        + 374761393
    ) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    return torch.where(((h ^ (h >> 16)) & 1) == 0, 1.0, -1.0).to(dtype=x.dtype).reshape_as(x)


def _cslice(val: Any, batch_size: int, start: int, count: int) -> Any:
    if not isinstance(val, torch.Tensor) or val.ndim < 1 or val.shape[0] != batch_size or count >= batch_size:
        return val
    if start + count <= batch_size:
        return val[start : start + count]
    return torch.cat((val[start:], val[: start + count - batch_size]), dim=0)


def _repeat_batch_arg(val: Any, batch_size: int, reps: int) -> Any:
    if not isinstance(val, torch.Tensor) or val.ndim < 1 or val.shape[0] != batch_size:
        return val
    rest = val.shape[1:]
    return val.unsqueeze(1).expand(batch_size, reps, *rest).reshape(batch_size * reps, *rest)


def _dup_batch_arg(val: Any, batch_size: int) -> Any:
    if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == batch_size:
        return torch.cat((val, val), dim=0)
    return val


def _eff_dim_t(x: torch.Tensor, dim_x: int) -> torch.Tensor:
    b = int(x.shape[0])
    if b < 4 or dim_x <= 1:
        return x.new_full((), float(dim_x), dtype=torch.float64)
    xc = x.detach().reshape(b, dim_x).float()
    xc = xc - xc.mean(0, keepdim=True)
    sx = xc.abs().amax()
    xu = xc / torch.where(sx > 0.0, sx, sx.new_ones(()))
    g = (xu @ xu.T).double()
    dg = g.diagonal()
    s1 = dg.sum()
    s11 = dg.square().sum()
    s2 = g.square().sum()
    a = float((b - 1) * (b - 2))
    c = float(b * (b - 1))
    num = a * s1 * s1 - c * s11 + 2.0 * s2
    den = a * s2 - c * s11 + s1 * s1
    return torch.where(
        (sx > 0.0) & (num > 0.0),
        (num / den.clamp_min(1e-30)).clamp(1.0, float(dim_x)),
        s1.new_full((), float(dim_x)),
    )


def _enter_bn_eval_and_hooks(
    mod: Optional[nn.Module],
    has_bn: bool,
    bf16: bool,
    *,
    attach_dual_hooks: bool,
) -> Tuple[List[Tuple[nn.Module, bool]], List[Any]]:
    bn_modes: List[Tuple[nn.Module, bool]] = []
    bn_hooks: List[Any] = []
    if not isinstance(mod, nn.Module):
        return bn_modes, bn_hooks
    if not (has_bn or (bf16 and attach_dual_hooks)):
        return bn_modes, bn_hooks
    for m in mod.modules():
        if has_bn and isinstance(m, nn.modules.batchnorm._BatchNorm):
            bn_modes.append((m, m.training))
            m.eval()
            if bf16 and attach_dual_hooks:
                bn_hooks.append(m.register_forward_hook(_bn_dual_bf16_hook))
        elif bf16 and attach_dual_hooks and isinstance(m, nn.modules.instancenorm._InstanceNorm):
            bn_hooks.append(m.register_forward_hook(_bn_dual_bf16_hook))
    return bn_modes, bn_hooks


def _exit_bn_eval_and_hooks(
    bn_modes: List[Tuple[nn.Module, bool]],
    bn_hooks: List[Any],
) -> None:
    for hook in bn_hooks:
        hook.remove()
    for m, flag in bn_modes:
        m.training = flag


class AxiomPenaltyKernel:
    """Compute-regime-dispatched scale-equivariant R1+R2 + Law 4.5 floor + gauge (fwAD and secant probes only)."""

    def __init__(
        self,
        *,
        record: Optional[AxiomStepRecord] = None,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lambda_floor: float = 0.005,
        beta_ema: float = 1.5,
        gamma_gauge: float = 0.03,
        max_lambda: float = 2.0,
    ) -> None:
        self.record = AxiomStepRecord() if record is None else record
        self.coeff = _check_scalar(coeff, "coeff", 0.0)
        self.kappa = _check_scalar(kappa, "kappa", 0.0, slo=True)
        self.lambda_floor = _check_scalar(lambda_floor, "lambda_floor", 0.0, slo=True)
        self.beta_ema = _check_scalar(beta_ema, "beta_ema", 0.0)
        self.gamma_gauge = _check_scalar(gamma_gauge, "gamma_gauge", 0.0)
        self.max_lambda = _check_scalar(max_lambda, "max_lambda", self.lambda_floor)
        self._c_max = math.log(max(1.0, self.max_lambda / self.lambda_floor))
        self._fwad_unsupported: set[int] = set()
        self._fwad_verified: set[int] = set()

    def _verify_tangent(self, tangent: Optional[torch.Tensor], cid: int, mod: Optional[nn.Module]) -> torch.Tensor:
        if tangent is None:
            raise NotImplementedError("dual tangent dropped")
        if cid not in self._fwad_verified:
            if _tangent_has_inplace_softmax_bug(tangent):
                raise RuntimeError("fwAD softmax inplace-Exp backward unsupported")
            if mod is not None and len(self._fwad_verified) < _MAX_VERIFIED_CRITICS:
                self._fwad_verified.add(cid)
        return tangent

    def _fwad_exact_one(
        self,
        live: Callable[..., torch.Tensor],
        x: torch.Tensor,
        a: Tuple[Any, ...],
        kw: Dict[str, Any],
        sc_t: torch.Tensor,
        *,
        dim_x: int,
        bf16: bool,
        cid: int,
        mod: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(x.shape[0])
        rest = x.shape[1:]
        work_dt = torch.bfloat16 if bf16 else x.dtype
        x_w = x.detach().to(work_dt)
        eye_d = torch.eye(dim_x, device=x.device, dtype=work_dt)
        x_rep = x_w.unsqueeze(1).expand(b, dim_x, *rest).reshape(b * dim_x, *rest)
        v_rep = eye_d.view(1, dim_x, *rest).expand(b, dim_x, *rest).reshape(b * dim_x, *rest)
        a_rep = tuple(_repeat_batch_arg(v, b, dim_x) for v in a)
        kw_rep = {k: _repeat_batch_arg(v, b, dim_x) for k, v in kw.items()}
        with fwAD.dual_level():
            try:
                out = live(fwAD.make_dual(x_rep, v_rep), *a_rep, **kw_rep)
                y_raw, j_raw = fwAD.unpack_dual(out)
                j_raw = self._verify_tangent(j_raw, cid, mod)
                y_bd = _fmt_logits(y_raw, b * dim_x).reshape(b, dim_x, -1)
                j_bd = _fmt_logits(j_raw, b * dim_x).reshape(b, dim_x, -1).float() * sc_t
                return y_bd[:, 0, :], j_bd.square().sum(dim=1).mean(dim=-1)
            except RuntimeError:
                # Fallback for fixed-batch closures expecting exact batch size B.
                # If j == 0 also fails, propagate so penalty() falls back to secant_debiased.
                y_first: Optional[torch.Tensor] = None
                sq_acc: Optional[torch.Tensor] = None
                for j_idx in range(dim_x):
                    v_j = eye_d[j_idx].view(1, *rest).expand(b, *rest)
                    out_j = live(fwAD.make_dual(x_w, v_j), *a, **kw)
                    y_j, t_j = fwAD.unpack_dual(out_j)
                    t_j = self._verify_tangent(t_j, cid, mod)
                    if y_first is None:
                        y_first = _fmt_logits(y_j, b)
                    term = (_fmt_logits(t_j, b).float() * sc_t).square().mean(dim=-1)
                    sq_acc = term if sq_acc is None else (sq_acc + term)
                assert y_first is not None and sq_acc is not None
                return y_first, sq_acc

    def _fwad_exact(
        self,
        live: Callable[..., torch.Tensor],
        xr: torch.Tensor,
        xf: torch.Tensor,
        a: Tuple[Any, ...],
        kw: Dict[str, Any],
        phi: float,
        *,
        dim_x: int,
        mod: Optional[nn.Module],
        has_bn: bool,
        has_bd: bool,
        bf16: bool,
        cid: int,
        sc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        br = int(xr.shape[0])
        bf = int(xf.shape[0])
        bn_modes, bn_hooks = _enter_bn_eval_and_hooks(mod, has_bn, bf16, attach_dual_hooks=True)

        def _cap(sq: torch.Tensor) -> torch.Tensor:
            return torch.relu((sq + 1e-12).sqrt() - self.kappa).square().mean()

        self.record._in_penalty = True
        try:
            if not has_bn and not has_bd and br == bf and xr.shape == xf.shape:
                try:
                    if not a and not kw and xr.ndim == 2:
                        tc = (
                            0.15
                            + 0.70
                            * torch.remainder(
                                (torch.arange(br, device=xr.device, dtype=torch.float32) + 0.5) / float(br) + phi,
                                1.0,
                            )
                        ).to(xr.dtype).reshape((br,) + (1,) * (xr.ndim - 1))
                        x_chord = (1.0 - tc) * xr.detach() + tc * xf.detach()
                        x_cat = torch.cat((xr.detach(), xf.detach(), x_chord), dim=0)
                        y3, sq3 = self._fwad_exact_one(
                            live, x_cat, (), {}, sc_t, dim_x=dim_x, bf16=bf16, cid=cid, mod=mod
                        )
                        sqr, sqf, sqc = sq3[:br], sq3[br : 2 * br], sq3[2 * br :]
                        s_hat = 0.5 * (sqr.mean() + sqf.mean())
                        cap_tot = _cap(sqr) + 0.75 * _cap(sqf) + 0.25 * _cap(sqc)
                        return y3[:br], y3[br : 2 * br], s_hat, cap_tot
                    x_cat = torch.cat((xr.detach(), xf.detach()), dim=0)
                    a_dup = tuple(_dup_batch_arg(v, br) for v in a)
                    kw_dup = {k: _dup_batch_arg(v, br) for k, v in kw.items()}
                    y2, sq2 = self._fwad_exact_one(
                        live, x_cat, a_dup, kw_dup, sc_t, dim_x=dim_x, bf16=bf16, cid=cid, mod=mod
                    )
                    sqr, sqf = sq2[:br], sq2[br:]
                    return y2[:br], y2[br:], 0.5 * (sqr.mean() + sqf.mean()), _cap(sqr) + _cap(sqf)
                except RuntimeError:
                    pass
            dr_s, sqr = self._fwad_exact_one(
                live, xr, a, kw, sc_t, dim_x=dim_x, bf16=bf16, cid=cid, mod=mod
            )
            df_s, sqf = self._fwad_exact_one(
                live, xf, a, kw, sc_t, dim_x=dim_x, bf16=bf16, cid=cid, mod=mod
            )
            return dr_s, df_s, 0.5 * (sqr.mean() + sqf.mean()), _cap(sqr) + _cap(sqf)
        finally:
            self.record._in_penalty = False
            _exit_bn_eval_and_hooks(bn_modes, bn_hooks)

    def _fwad_debiased(
        self,
        live: Callable[..., torch.Tensor],
        xr: torch.Tensor,
        xf: torch.Tensor,
        a: Tuple[Any, ...],
        kw: Dict[str, Any],
        phi: float,
        *,
        mod: Optional[nn.Module],
        has_bn: bool,
        has_bd: bool,
        bf16: bool,
        cid: int,
        sc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        br = int(xr.shape[0])
        bf = int(xf.shape[0])
        bn_modes, bn_hooks = _enter_bn_eval_and_hooks(mod, has_bn, bf16, attach_dual_hooks=True)

        def _dual_run(
            xs: torch.Tensor,
            vs: torch.Tensor,
            ca: Tuple[Any, ...],
            ckw: Dict[str, Any],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x_in = xs.to(torch.bfloat16) if bf16 else xs
            v_in = vs.to(torch.bfloat16) if bf16 else vs
            out = live(fwAD.make_dual(x_in, v_in), *ca, **ckw)
            y_raw, j_raw = fwAD.unpack_dual(out)
            j_raw = self._verify_tangent(j_raw, cid, mod)
            n = int(xs.shape[0])
            return _fmt_logits(y_raw, n), _fmt_logits(j_raw, n).float() * sc_t

        def _two_probe_one(
            xs: torch.Tensor,
            ca: Tuple[Any, ...],
            ckw: Dict[str, Any],
            salt1: int,
            salt2: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            v1 = _dense_rademacher(xs, phi, salt1)
            v2 = _dense_rademacher(xs, phi, salt2)
            if not has_bd:
                try:
                    n = int(xs.shape[0])
                    x2 = torch.cat((xs, xs), dim=0)
                    v2_cat = torch.cat((v1, v2), dim=0)
                    ca2 = tuple(_dup_batch_arg(v, n) for v in ca)
                    ckw2 = {k: _dup_batch_arg(v, n) for k, v in ckw.items()}
                    y2, j2 = _dual_run(x2, v2_cat, ca2, ckw2)
                    return y2[:n], 0.5 * (j2[:n].square().mean() + j2[n:].square().mean())
                except RuntimeError:
                    pass
            y1, j1 = _dual_run(xs, v1, ca, ckw)
            _, j2 = _dual_run(xs, v2, ca, ckw)
            return y1, 0.5 * (j1.square().mean() + j2.square().mean())

        self.record._in_penalty = True
        try:
            sr = int(phi * br) % br
            sf = int(((phi + 0.5) % 1.0) * bf) % bf
            with fwAD.dual_level():
                if not a and not kw and br == bf and br >= 4 and not has_bd:
                    try:
                        b_reg = max(2, br // 4)
                        brs = b_reg // 2
                        bfs = b_reg - brs
                        xr_s = _cslice(xr.detach(), br, sr, brs)
                        xf_s = _cslice(xf.detach(), bf, sf, bfs)
                        x_pair = torch.cat((xr_s, xf_s), dim=0)
                        v1 = torch.cat(
                            (_dense_rademacher(xr_s, phi, 1), _dense_rademacher(xf_s, phi, 2)),
                            dim=0,
                        )
                        v2 = torch.cat(
                            (_dense_rademacher(xr_s, phi, 3), _dense_rademacher(xf_s, phi, 4)),
                            dim=0,
                        )
                        y_all, j_all = _dual_run(
                            torch.cat((x_pair, x_pair), dim=0),
                            torch.cat((v1, v2), dim=0),
                            (),
                            {},
                        )
                        j_p1 = j_all[:b_reg]
                        j_p2 = j_all[b_reg:]
                        s_r = 0.5 * (j_p1[:brs].square().mean() + j_p2[:brs].square().mean())
                        s_f = 0.5 * (j_p1[brs:].square().mean() + j_p2[brs:].square().mean())
                        return y_all[:brs], y_all[brs:b_reg], 0.5 * (s_r + s_f)
                    except RuntimeError:
                        pass
                brs = br if has_bd else max(1, min(br, br // 4))
                bfs = bf if has_bd else max(1, min(bf, bf // 4))
                try:
                    yr, sr_sq = _two_probe_one(
                        _cslice(xr.detach(), br, sr, brs),
                        tuple(_cslice(v, br, sr, brs) for v in a),
                        {k: _cslice(v, br, sr, brs) for k, v in kw.items()},
                        1,
                        3,
                    )
                    yf, sf_sq = _two_probe_one(
                        _cslice(xf.detach(), bf, sf, bfs),
                        tuple(_cslice(v, bf, sf, bfs) for v in a),
                        {k: _cslice(v, bf, sf, bfs) for k, v in kw.items()},
                        2,
                        4,
                    )
                except RuntimeError:
                    if brs == br and bfs == bf:
                        raise
                    yr, sr_sq = _two_probe_one(xr.detach(), a, kw, 1, 3)
                    yf, sf_sq = _two_probe_one(xf.detach(), a, kw, 2, 4)
                return yr, yf, 0.5 * (sr_sq + sf_sq)
        finally:
            self.record._in_penalty = False
            _exit_bn_eval_and_hooks(bn_modes, bn_hooks)

    def _secant_debiased(
        self,
        live: Callable[..., torch.Tensor],
        xr: torch.Tensor,
        xf: torch.Tensor,
        a: Tuple[Any, ...],
        kw: Dict[str, Any],
        phi: float,
        *,
        dim_x: int,
        mod: Optional[nn.Module],
        has_bn: bool,
        has_bd: bool,
        bf16: bool,
        sc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        br = int(xr.shape[0])
        bf = int(xf.shape[0])
        bn_modes, bn_hooks = _enter_bn_eval_and_hooks(mod, has_bn, bf16, attach_dual_hooks=False)

        def _secant_probe(
            xs: torch.Tensor,
            ca: Tuple[Any, ...],
            ckw: Dict[str, Any],
            salt: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            n = int(xs.shape[0])
            is_lowp = bf16 or xs.dtype in (torch.bfloat16, torch.float16)
            h_rel = 1e-2 if is_lowp else 1e-3
            x_f = xs.detach().float()
            x_rms = x_f.square().mean().sqrt()
            scale = torch.where(
                torch.isfinite(x_rms) & (x_rms > 1e-12),
                x_rms,
                x_f.new_ones(()),
            )
            h = h_rel * scale
            v_f = _dense_rademacher(xs, phi, salt).float()
            x_p = (x_f + h * v_f).to(xs.dtype)
            x_m = (x_f - h * v_f).to(xs.dtype)
            if not has_bd:
                try:
                    ca2 = tuple(_dup_batch_arg(v, n) for v in ca)
                    ckw2 = {k: _dup_batch_arg(v, n) for k, v in ckw.items()}
                    y_cat = _fmt_logits(live(torch.cat((x_p, x_m), dim=0), *ca2, **ckw2), 2 * n)
                    y_p, y_m = y_cat[:n], y_cat[n:]
                    t_k = ((y_p.float() - y_m.float()) / (2.0 * h)) * sc_t
                    return 0.5 * (y_p + y_m), t_k
                except RuntimeError:
                    pass
            y_p = _fmt_logits(live(x_p, *ca, **ckw), n)
            y_m = _fmt_logits(live(x_m, *ca, **ckw), n)
            t_k = ((y_p.float() - y_m.float()) / (2.0 * h)) * sc_t
            return 0.5 * (y_p + y_m), t_k

        def _two_secant_one(
            xs: torch.Tensor,
            ca: Tuple[Any, ...],
            ckw: Dict[str, Any],
            salt1: int,
            salt2: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            y1, t1 = _secant_probe(xs, ca, ckw, salt1)
            _, t2 = _secant_probe(xs, ca, ckw, salt2)
            return y1, 0.5 * (t1.square().mean() + t2.square().mean())

        self.record._in_penalty = True
        try:
            sr = int(phi * br) % br
            sf = int(((phi + 0.5) % 1.0) * bf) % bf
            subsample = (not has_bd) and br >= 8 and bf >= 8 and dim_x > 32
            brs = max(2, br // 4) if subsample else br
            bfs = max(2, bf // 4) if subsample else bf
            try:
                yr, sr_sq = _two_secant_one(
                    _cslice(xr.detach(), br, sr, brs),
                    tuple(_cslice(v, br, sr, brs) for v in a),
                    {k: _cslice(v, br, sr, brs) for k, v in kw.items()},
                    1,
                    3,
                )
                yf, sf_sq = _two_secant_one(
                    _cslice(xf.detach(), bf, sf, bfs),
                    tuple(_cslice(v, bf, sf, bfs) for v in a),
                    {k: _cslice(v, bf, sf, bfs) for k, v in kw.items()},
                    2,
                    4,
                )
            except RuntimeError:
                if brs == br and bfs == bf:
                    raise
                yr, sr_sq = _two_secant_one(xr.detach(), a, kw, 1, 3)
                yf, sf_sq = _two_secant_one(xf.detach(), a, kw, 2, 4)
            return yr, yf, 0.5 * (sr_sq + sf_sq)
        finally:
            self.record._in_penalty = False
            _exit_bn_eval_and_hooks(bn_modes, bn_hooks)

    def penalty(
        self,
        live: Callable[..., torch.Tensor],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *args: Any,
        collect_stats: bool = False,
        critic_module: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.coeff == 0.0:
            self.record.clear_transient_captures()
            stats = {"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {}
            return x_real.new_zeros(()), stats
        self.record.calls += 1
        phi = float(self.record.chord_phase)
        self.record.chord_phase = (phi + _GOLDEN_CONJ) % 1.0
        mod = critic_module if isinstance(critic_module, nn.Module) else (live if isinstance(live, nn.Module) else None)
        cid = id(mod) if mod is not None else id(live)
        dim_x = max(1, int(x_real[0].numel()))
        has_bn = isinstance(mod, nn.Module) and any(
            isinstance(m, nn.modules.batchnorm._BatchNorm) for m in mod.modules()
        )
        has_bd = _has_batch_distance(mod)
        bf16 = bool(
            x_real.is_cuda
            and isinstance(mod, nn.Module)
            and (
                getattr(mod, "bf16", False)
                or (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16)
            )
        )
        use_secant = _needs_secant_module(mod) or (cid in self._fwad_unsupported)
        use_exact = (not use_secant) and (dim_x <= _FWAD_EXACT_MAX_DIM) and (not has_bd)

        ds = self.record._dev_ema("ema_data_scale", x_real.device)
        mi = self.record._dev_ema("margin_initialized", x_real.device)
        if x_real.ndim >= 4 and dim_x > 256:
            denom_dim = ds.new_full((), float(dim_x))
        else:
            denom_dim = _eff_dim_t(x_real, dim_x)
        sc_t = torch.where(
            mi > 0.5,
            (ds / denom_dim.sqrt()).clamp_min(1e-30),
            ds.new_ones(()),
        ).float()

        csig = _cond_sig((x_real, *args), kwargs)
        kr = _tensor_id(x_real)
        kf = _tensor_id(x_fake)
        stp = self.record.observed_steps
        dr_l = next(
            (
                e[4]
                for e in reversed(self.record._live_captures)
                if e[0] == stp and e[1] == kr and e[2] == csig and e[5] and e[4] is not None
            ),
            None,
        )
        df_l = next(
            (
                e[4]
                for e in reversed(self.record._live_captures)
                if e[0] == stp and e[1] == kf and e[2] == csig and e[5] and e[4] is not None
            ),
            None,
        )
        reused = dr_l is not None and df_l is not None

        try:
            cap_tot: Optional[torch.Tensor] = None
            regime = "secant_debiased"
            if use_exact:
                try:
                    dr_s, df_s, s_hat, cap_tot = self._fwad_exact(
                        live,
                        x_real,
                        x_fake,
                        args,
                        kwargs,
                        phi,
                        dim_x=dim_x,
                        mod=mod,
                        has_bn=has_bn,
                        has_bd=has_bd,
                        bf16=bf16,
                        cid=cid,
                        sc_t=sc_t,
                    )
                    regime = "fwad_exact"
                except (RuntimeError, NotImplementedError):
                    self._fwad_unsupported.add(cid)
                    use_secant = True
            elif not use_secant:
                try:
                    dr_s, df_s, s_hat = self._fwad_debiased(
                        live,
                        x_real,
                        x_fake,
                        args,
                        kwargs,
                        phi,
                        mod=mod,
                        has_bn=has_bn,
                        has_bd=has_bd,
                        bf16=bf16,
                        cid=cid,
                        sc_t=sc_t,
                    )
                    regime = "fwad_debiased"
                except (RuntimeError, NotImplementedError):
                    self._fwad_unsupported.add(cid)
                    use_secant = True
            if use_secant:
                cap_tot = None
                dr_s, df_s, s_hat = self._secant_debiased(
                    live,
                    x_real,
                    x_fake,
                    args,
                    kwargs,
                    phi,
                    dim_x=dim_x,
                    mod=mod,
                    has_bn=has_bn,
                    has_bd=has_bd,
                    bf16=bf16,
                    sc_t=sc_t,
                )
                regime = "secant_debiased"

            dr2, df2 = (dr_l, df_l) if reused else (dr_s, df_s)
            ema_gsq = self.record._dev_ema("ema_grad_sq", s_hat.device)
            lam = (
                self.lambda_floor
                * torch.exp((self.beta_ema * torch.relu(ema_gsq - self.kappa * self.kappa)).clamp_max(self._c_max))
            ).to(s_hat.dtype)
            self.record._observe_grad_sq_tensor(s_hat.detach())
            pl = 0.5 * self.coeff * (lam * s_hat if cap_tot is None else (cap_tot + lam * s_hat))
            if self.gamma_gauge > 0.0:
                ml = 0.5 * (dr2.float().mean() + df2.float().mean())
                pg = 0.5 * self.coeff * self.gamma_gauge * ml.square()
            else:
                ml = s_hat.new_zeros(())
                pg = s_hat.new_zeros(())
            s_th = float(self.record.thermal_scale)
            if s_th < 1.0:
                if pl.requires_grad:
                    pl = pl * s_th + pl.detach() * (1.0 - s_th)
                if pg.requires_grad:
                    gu = ((ml.detach().abs() - 0.5) / 1.0).clamp(0.0, 1.0)
                    sg = s_th + (gu * gu * (3.0 - 2.0 * gu)) * (math.sqrt(max(0.0, s_th)) - s_th)
                    pg = pg * sg + pg.detach() * (1.0 - sg)
            pen = pl + pg
            if not collect_stats:
                return pen, {}
            pv, sv, lv = torch.stack((pen.detach().float(), s_hat.detach().float(), lam.detach().float())).tolist()
            return pen, {
                "applied": True,
                "pen": float(pv),
                "s_hat": float(sv),
                "mean_grad_norm": math.sqrt(max(0.0, float(sv))),
                "lambda_eff": float(lv),
                "center": self.kappa,
                "regime": regime,
                "s_thermal": s_th,
                "live_endpoints_reused": bool(reused),
            }
        finally:
            self.record.clear_transient_captures()


class AxiomCriticPenalty:
    """Public callable wrapper matching GANTrainer, WordFixture, DDGAN, and BehaviorComponents."""

    ema_critic = None

    def __init__(
        self,
        optimizer_or_recipe: Any,
        optimizer: Any = None,
        *,
        coeff: Optional[float] = None,
        kappa: Optional[float] = None,
        output: Any = None,
        collect_stats: bool = False,
        **options: Any,
    ) -> None:
        if optimizer is None:
            self.recipe = None
            self.optimizer = optimizer_or_recipe
        else:
            self.recipe = optimizer_or_recipe
            self.optimizer = optimizer
        if not hasattr(self.optimizer, "param_groups"):
            raise TypeError("AxiomCriticPenalty requires a critic optimizer")
        rec = getattr(self.optimizer, "record", None) or AxiomStepRecord()
        self.optimizer.record = rec
        self.optimizer._axiom_record = rec
        self.critic = getattr(self.optimizer, "critic", None)
        _attach_axiom_critic_hook(self.critic, rec)
        k_opts: Dict[str, Any] = {
            "coeff": float(getattr(self.recipe, "reg_coeff", 1.0)) if coeff is None else float(coeff),
            "kappa": float(getattr(self.recipe, "reg_kappa", 1.0)) if kappa is None else float(kappa),
        }
        for key in ("coeff", "kappa", "lambda_floor", "beta_ema", "gamma_gauge", "max_lambda"):
            if options.get(key) is not None:
                k_opts[key] = options[key]
        self.regularizer = AxiomPenaltyKernel(record=rec, **k_opts)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_out if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}

    def __call__(
        self,
        critic: Callable[..., torch.Tensor],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        pen, self.last_stats = self.regularizer.penalty(
            lambda x, *a, **kw: self.output(critic(x, *a, **kw)),
            x_real,
            x_fake,
            *args,
            collect_stats=self.collect_stats,
            critic_module=critic if isinstance(critic, nn.Module) else self.critic,
            **kwargs,
        )
        return pen

    def diagnostics(self) -> Dict[str, Any]:
        r = self.regularizer.record
        return {
            "axiom_calls": int(r.calls),
            "axiom_ema_grad_sq": float(r.ema_grad_sq),
            "axiom_chord_phase": float(r.chord_phase),
            "axiom_thermal_scale": float(r.thermal_scale),
            "axiom_step_scale": float(r.step_scale),
        }


__all__ = [
    "AxiomCriticPenalty",
    "AxiomPenaltyKernel",
    "AxiomStepRecord",
    "_attach_axiom_critic_hook",
    "_attach_axiom_generator_hook",
    "_extract_axiom_inputs_parts",
    "_record_axiom_capture",
    "bind_axiom_record",
    "eval_probe_forward",
]
