"""Axiom Module B: Law 2 Critic Potential Geometry, Law 4.5 Normal Dissipation, and StepRecord."""
from __future__ import annotations

from copy import deepcopy
import math, random, sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

_GOLDEN_CONJ, _MAX_LIVE_CAPTURES, _PROBE_MAX_BATCH = 0.6180339887498949, 6, 32
_EMA_BOOLS = frozenset({"grad_sq_initialized", "margin_initialized"})
_BOOL_FIELDS = ("anchor_started", "grad_sq_initialized", "margin_initialized", "ell_initialized", "thermal_initialized", "thermal_locked", "cond_initialized", "het_initialized")
_THERMAL_KEYS = ("ema_ed", "ema_mu_sq", "ema_spread", "ema_q90", "ema_hmax", "ema_tail")
_SCALAR_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float], bool, bool]] = {
    "chord_phase": (0.0, 1.0, False, True), "ema_grad_sq": (0.0, None, False, False), "ema_margin": (-60.0, 60.0, False, False),
    "ema_data_scale": (0.0, None, True, False), "ema_push_res": (0.0, None, False, False), "ell_ema": (0.0, None, False, False),
    "thermal_scale": (1e-5, 1.0, False, False), "step_scale": (0.0, 1.0, True, False), "cond_ema": (-0.5, 2.0, False, False),
    "het_ema": (0.0, None, False, False), "g_func_gain": (0.0, 1.0, True, False), **{k: (None, None, False, False) for k in _THERMAL_KEYS},
}
_DICT_FIELDS = ("chart_state", "atomic_ledger", "cond_fiber_bank")
_RECORD_KEYS = frozenset({"schema", "calls", "observed_steps", "last_ratio", *_BOOL_FIELDS, *_SCALAR_BOUNDS, *_DICT_FIELDS})
_first_out = lambda out: out[0] if isinstance(out, (tuple, list)) and out else out
_fmt_logits = lambda y, b: y.reshape(b, 1) if y.ndim < 2 else y.reshape(b, -1)

def _tensor_id(t: Any) -> Optional[Tuple[int, int, Tuple[int, ...], str]]:
    if not isinstance(t, torch.Tensor) or t.ndim < 1 or t.numel() == 0: return None
    try: ptr = int(t.untyped_storage().data_ptr())
    except RuntimeError: return None
    return None if ptr == 0 else (ptr, int(t.storage_offset()), tuple(int(s) for s in t.shape), str(t.device))

def _cond_sig(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    s = [("T", _tensor_id(x), str(x.dtype)) if isinstance(x, torch.Tensor) else ("V", type(x).__qualname__, repr(x)) for x in args[1:]]
    return tuple(s + [("KT", k, _tensor_id(kwargs[k]), str(kwargs[k].dtype)) if isinstance(kwargs[k], torch.Tensor) else ("KV", k, type(kwargs[k]).__qualname__, repr(kwargs[k])) for k in sorted(kwargs)])

def _walk_payload(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    return fn(obj) if isinstance(obj, torch.Tensor) else ({k: _walk_payload(v, fn) for k, v in obj.items()} if isinstance(obj, dict) else (type(obj)(_walk_payload(v, fn) for v in obj) if isinstance(obj, (list, tuple)) else deepcopy(obj)))

def _validate_payload(obj: Any, name: str) -> None:
    if obj is None or type(obj) in (bool, int, str): return
    if type(obj) is float:
        if not math.isfinite(obj): raise ValueError(f"non-finite float in {name}")
    elif isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()): raise ValueError(f"non-finite tensor in {name}")
    elif isinstance(obj, (list, tuple)):
        for it in obj: _validate_payload(it, name)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int, tuple)): raise ValueError(f"unsupported key in {name}")
            _validate_payload(v, name)
    else: raise ValueError(f"unsupported type {type(obj).__name__} in {name}")
def _check_scalar(val: Any, name: str, lo: Optional[float] = None, hi: Optional[float] = None, slo: bool = False, shi: bool = False) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val): raise ValueError(f"invalid {name}: {val!r}")
    f = float(val)
    if (lo is not None and (f <= lo if slo else f < lo)) or (hi is not None and (f >= hi if shi else f > hi)): raise ValueError(f"{name}={f} outside bounds")
    return f
def _ema_prop(name: str) -> property:
    is_b = name in _EMA_BOOLS
    return property(lambda s: (s._flush_ema() if name in s._ema_t else None, s._ema[name])[1],
                    lambda s, v: (s._ema.__setitem__(name, bool(v) if is_b else float(v)), s._ema_t.pop(name, None))[0])

class AxiomStepRecord:
    """Standalone checkpointed state linking Axiom critic, transport, prior, and optimizer."""
    ema_grad_sq, grad_sq_initialized, ema_margin = _ema_prop("ema_grad_sq"), _ema_prop("grad_sq_initialized"), _ema_prop("ema_margin")
    ema_data_scale, ema_push_res, margin_initialized, ell_ema = _ema_prop("ema_data_scale"), _ema_prop("ema_push_res"), _ema_prop("margin_initialized"), _ema_prop("ell_ema")

    def __init__(self) -> None:
        self.schema, self.calls, self.observed_steps, self.anchor_started, self.last_ratio, self.chord_phase = 1, 0, 0, False, None, 0.0
        self._ema: Dict[str, Any] = {"ema_grad_sq": 1.0, "grad_sq_initialized": False, "ema_margin": 0.0, "ema_data_scale": 1.0, "ema_push_res": 1.0, "margin_initialized": False, "ell_ema": 0.0}
        self.ell_initialized = self.thermal_initialized = self.thermal_locked = self.cond_initialized = self.het_initialized = False
        self.thermal_scale = self.step_scale = self.het_ema = self.g_func_gain = self.ema_ed = self.ema_mu_sq = self.ema_spread = self.ema_q90 = self.ema_hmax = 1.0
        self.ema_tail, self.cond_ema, self.chart_state, self.atomic_ledger, self.cond_fiber_bank = 0.0, 0.25, None, None, None
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._ema_t: Dict[str, torch.Tensor] = {}; self._live_captures: List[List[Any]] = []
        self._in_penalty = self._in_probe = self._in_protected_probe = self._simplicial_step = False; self._enable_secant_probe, self._disc_gate = True, 0.0
        self._last_probe_actual_input = self._g_probe_cache = self._active_prior_table = self._active_batch_rows = self._cond_bank_obj = None

    def take_pending_ema_tensors(self) -> Tuple[List[str], List[torch.Tensor]]: return list(self._ema_t.keys()), list(self._ema_t.values())
    def record_step(self, optimizer: Any = None) -> None: del optimizer; self.observed_steps += 1; self.clear_transient_captures()
    def __copy__(self) -> "AxiomStepRecord": dup = AxiomStepRecord(); dup.load_state_dict(self.state_dict()); return dup
    def __deepcopy__(self, memo: Dict[int, Any]) -> "AxiomStepRecord": dup = self.__copy__(); memo[id(self)], dup._enable_secant_probe = dup, self._enable_secant_probe; return dup

    def apply_flushed_ema_values(self, keys: List[str], vals: List[float]) -> None:
        for k, v in zip(keys, vals):
            if k in self._ema: self._ema[k] = bool(v > 0.5) if k in _EMA_BOOLS else float(v); self._ema_t.pop(k, None)
    def _flush_ema(self) -> None:
        ks, ts = self.take_pending_ema_tensors()
        if ts: self.apply_flushed_ema_values(ks, torch.stack([t.double() for t in ts]).tolist() if all(t.device == ts[0].device for t in ts) else [float(t.item()) for t in ts])
    def _dev_ema(self, name: str, dev: torch.device) -> torch.Tensor:
        if (cur := self._ema_t.get(name)) is not None and cur.device == dev: return cur
        if cur is not None: self._flush_ema()
        return torch.full((), float(self._ema[name]), dtype=torch.float64, device=dev)

    @torch.no_grad()
    def _observe_grad_sq_tensor(self, s_hat_det: torch.Tensor) -> None:
        obs = s_hat_det.detach().double().clamp(0.0, 20.0); fin, prev, pi = torch.isfinite(obs), self._dev_ema("ema_grad_sq", obs.device), self._dev_ema("grad_sq_initialized", obs.device)
        self._ema_t["ema_grad_sq"], self._ema_t["grad_sq_initialized"] = torch.where(fin, torch.where(pi > 0.5, 0.95 * prev + 0.05 * obs, obs), prev), torch.where(fin, pi.new_full((), 1.0), pi)

    @torch.no_grad()
    def observe_margin_and_scale(self, real_logits: torch.Tensor, fake_logits: torch.Tensor, x_real: Optional[torch.Tensor] = None, x_fake: Optional[torch.Tensor] = None) -> None:
        if self._in_penalty or self._in_probe or self._in_protected_probe: return
        dev, m = real_logits.device, (real_logits.detach().float().mean().double() - fake_logits.detach().float().mean().double()).clamp(-50.0, 50.0)
        fm, pm, pi = torch.isfinite(m), self._dev_ema("ema_margin", dev), self._dev_ema("margin_initialized", dev); wi = pi > 0.5
        self._ema_t["ema_margin"], self._ema_t["margin_initialized"] = torch.where(fm, torch.where(wi, 0.90 * pm + 0.10 * m, m), pm), torch.where(fm, pi.new_full((), 1.0), pi)
        if x_real is None or x_real.numel() == 0: return
        xr = x_real.detach().reshape(x_real.shape[0], -1).double(); xm = xr.abs().amax(); sx = torch.where(xm > 0.0, xm, xm.new_full((), 1.0))
        sc, psc = ((xr / sx).square().sum(-1).mean().sqrt() * sx).clamp_min(1e-30), self._dev_ema("ema_data_scale", dev)
        nsc = torch.where(torch.isfinite(sc) & (xm > 0.0), torch.where(wi, 0.95 * psc + 0.05 * sc, sc), psc); self._ema_t["ema_data_scale"] = nsc
        if x_fake is not None and x_fake.shape == x_real.shape:
            xf = x_fake.detach().reshape(x_fake.shape[0], -1).double()
            rg, ppr = (((xf.mean(0) - xr.mean(0)).norm() + 0.5 * (xf.std() - xr.std()).abs()) / nsc.clamp_min(1e-30)).clamp_max(10.0), self._dev_ema("ema_push_res", dev)
            self._ema_t["ema_push_res"] = torch.where(torch.isfinite(rg), torch.where(wi, 0.92 * ppr + 0.08 * rg, rg), ppr)

    @property
    def thermal_ema(self) -> float:
        ud, q9, hm = min(1.0, max(0.0, -float(self.ema_spread)) / 0.03), float(self.ema_q90), float(self.ema_hmax)
        return max(0.5 * max(0.0, float(self.ema_ed)), math.sqrt(max(0.0, float(self.ema_mu_sq))), 0.60 * abs(float(self.ema_spread)), max(q9, 0.60 * max(0.0, -q9) * ud), max(hm, 0.60 * max(0.0, -hm) * ud), max(0.0, float(self.ema_tail)))

    ema_residual = thermal_ema

    def observe_thermal(self, r_ed: float, r_mu_sq: float, r_spread: float, r_q90: float, r_hmax: float, *, r_tail: float = 0.0, tau: float = 0.084, ema_decay: float = 0.96, s_min: float = 1e-5, shock_mu_ratio: float = 0.35, shock_ed_floor: float = 0.30, lock_hysteresis: float = 0.0) -> float:
        ed_f, mu_f, tau_f = float(r_ed), float(r_mu_sq), float(tau); tp = max(0.0, tau_f); cl, ce, sed = max(0.20, 2.0 * tp), max(0.70, 4.0 * tp), max(float(shock_ed_floor), 4.0 * tp * tp)
        vals = (max(-ce, min(ed_f, ce)), max(-cl * cl, min(mu_f, cl * cl)), max(-cl, min(float(r_spread), cl)), max(-cl, min(float(r_q90), cl)), max(-cl, min(float(r_hmax), cl)), max(-cl, min(float(r_tail), cl)))
        if not self.thermal_initialized or (self.thermal_scale < 0.99 and (ed_f > sed or mu_f > shock_mu_ratio * sed)):
            for k, v in zip(_THERMAL_KEYS, vals): setattr(self, k, v)
            self.thermal_initialized, self.thermal_locked = True, False
        else:
            d, om = float(ema_decay), 1.0 - float(ema_decay)
            for k, v in zip(_THERMAL_KEYS, vals): setattr(self, k, d * getattr(self, k) + om * v)
        if tau_f <= 0.0: self.thermal_scale, self.thermal_locked = 1.0, False; return 1.0
        tlo = min(0.92, 0.50 + (float(lock_hysteresis) if self.thermal_locked else 0.0)) * tau_f
        u = min(1.0, max(0.0, (self.thermal_ema - tlo) / max(1e-8, tau_f - tlo))); th = u * u * (3.0 - 2.0 * u); sm = max(1e-5, float(s_min))
        s = 1.0 if th >= 1.0 else (sm if th <= 0.0 else max(sm, min(1.0, 0.04 * th / math.sqrt(max(1e-8, 1.0 - th * th)))))
        self.thermal_scale, self.thermal_locked = max(sm, min(1.0, s)), bool(s <= 1e-4); return self.thermal_scale

    def clear_transient_captures(self) -> None:
        for e in self._live_captures: e[3] = e[4] = None; e[5].clear()
        self._live_captures.clear(); self._last_probe_actual_input = None
    def restore_device_state(self, device: torch.device) -> None:
        for a in ("chart_state", "atomic_ledger"):
            if (v := getattr(self, a)) is not None: setattr(self, a, _walk_payload(v, lambda t: t.to(device=device)))

    def state_dict(self) -> Dict[str, Any]:
        self._flush_ema(); cpu = lambda t: t.detach().cpu().clone()
        bank = self._cond_bank_obj.to_packet() if (self._cond_bank_obj is not None and hasattr(self._cond_bank_obj, "to_packet")) else self.cond_fiber_bank
        return {"schema": 1, "calls": int(self.calls), "observed_steps": int(self.observed_steps), "last_ratio": None if self.last_ratio is None else float(self.last_ratio),
                **{k: bool(getattr(self, k)) for k in _BOOL_FIELDS}, **{k: float(getattr(self, k)) for k in _SCALAR_BOUNDS},
                **{dk: None if (v := (bank if dk == "cond_fiber_bank" else getattr(self, dk))) is None else _walk_payload(v, cpu) for dk in _DICT_FIELDS}}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if (not isinstance(state, dict) or set(state) != _RECORD_KEYS or type(state["schema"]) is not int or state["schema"] != 1
                or any(type(state[ik]) is not int or state[ik] < 0 for ik in ("calls", "observed_steps"))
                or any(type(state[bk]) is not bool for bk in _BOOL_FIELDS) or state["anchor_started"] is not False): raise ValueError("invalid AxiomStepRecord checkpoint")
        lr, sc = (None if state["last_ratio"] is None else _check_scalar(state["last_ratio"], "last_ratio", 0.0)), {k: _check_scalar(state[k], k, *b) for k, b in _SCALAR_BOUNDS.items()}
        for dk in _DICT_FIELDS:
            if state[dk] is not None:
                if not isinstance(state[dk], dict): raise ValueError(f"AxiomStepRecord.{dk} must be dict or None")
                _validate_payload(state[dk], dk)
        cpu = lambda t: t.detach().cpu().clone(); self.schema, self.calls, self.observed_steps, self.last_ratio = 1, state["calls"], state["observed_steps"], lr
        for k, v in ({bk: state[bk] for bk in _BOOL_FIELDS} | sc).items(): setattr(self, k, v)
        for dk in _DICT_FIELDS: setattr(self, dk, None if state[dk] is None else _walk_payload(state[dk], cpu))
        self._reset_transient()

def _extract_axiom_inputs_parts(args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]], b: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool, Optional[Tuple[torch.Tensor, ...]]]:
    fl, ds = [], []
    for it in ([*args, *(kwargs[k] for k in sorted(kwargs))] if kwargs else args):
        if isinstance(it, torch.Tensor) and it.ndim >= 1 and it.shape[0] == b and it.numel() > 0:
            (fl.append(it) if it.is_floating_point() else (ds.append(it.detach().to(torch.int64).reshape(b, -1)) if not it.is_complex() else None))
    if not fl: return None, None, False, None
    cd, parts = (None if not ds else (ds[0] if len(ds) == 1 else torch.cat(ds, -1))), (tuple(t.reshape(b, -1) for t in fl) if len(fl) > 1 else None)
    return (fl[0] if parts is None else torch.cat(parts, -1)), cd, parts is not None, parts

def _record_axiom_capture(out: Any, x_in: Optional[torch.Tensor], cond: Optional[torch.Tensor], rec: Any, *, has_cont_cond: bool = False, parts: Optional[Tuple[torch.Tensor, ...]] = None, thermally_scaled: bool = False) -> None:
    if x_in is not None and isinstance(tgt := _first_out(out), torch.Tensor) and tgt.ndim >= 1 and tgt.shape[0] == x_in.shape[0]:
        tgt._axiom_input, tgt._axiom_record, tgt._axiom_cond, tgt._axiom_cont_cond, tgt._axiom_parts, tgt._axiom_thermally_scaled = x_in, rec, cond, bool(has_cont_cond), parts, bool(thermally_scaled)

@torch.no_grad()
def eval_probe_forward(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], record: Optional[AxiomStepRecord] = None) -> Optional[torch.Tensor]:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` without mutating RNGs, hooks, or BN buffers."""
    modes, prev_p = [(m, m.training) for m in module.modules()], (bool(getattr(record, "_in_probe", False)) if record is not None else False)
    cpu_r, py_r, np_m = torch.get_rng_state(), random.getstate(), sys.modules.get("numpy")
    np_r, c_dev = (np_m.random.get_state() if (np_m is not None and hasattr(np_m, "random")) else None), (torch.cuda.current_device() if torch.cuda.is_initialized() else None)
    c_r = torch.cuda.get_rng_state(c_dev) if c_dev is not None else None
    if record is not None: record._in_probe = True
    try:
        module.eval(); p0 = next(module.parameters(), None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(p0 is not None and p0.device.type == "cuda" and getattr(module, "_axiom_bf16_autocast", False))):
            tgt = _first_out(module(*args, **kwargs))
        return tgt.detach().to(torch.float32) if isinstance(tgt, torch.Tensor) else None
    finally:
        if record is not None: record._in_probe = prev_p
        for m, f in modes: m.training = f
        torch.set_rng_state(cpu_r); random.setstate(py_r)
        if c_r is not None and c_dev is not None: torch.cuda.set_rng_state(c_r, c_dev)
        if np_r is not None and np_m is not None: np_m.random.set_state(np_r)

def _axiom_critic_hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> Any:
    rec, x0 = getattr(module, "_axiom_record", None), (args[0] if (args and isinstance(args[0], torch.Tensor)) else None)
    if rec is not None and (rec._in_probe or rec._in_penalty):
        if rec._in_penalty and x0 is not None: rec._last_probe_actual_input = x0.detach()
        return None
    if not isinstance(raw := _first_out(output), torch.Tensor) or raw.ndim < 1: return None
    if (rec is not None and x0 is not None and module.training and torch.is_grad_enabled() and raw.requires_grad and not x0.requires_grad
            and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in (*args[1:], *kwargs.values()))
            and (sk := getattr(module, "_axiom_pending_source_key", None) or _tensor_id(x0)) is not None):
        alive = [True]; raw.register_hook(lambda _g, _a=alive: _a.clear())
        rec._live_captures.append([rec.observed_steps, sk, _cond_sig(args, kwargs), x0.detach(), raw.float() if raw.dtype == torch.bfloat16 else raw, alive])
        if len(rec._live_captures) > _MAX_LIVE_CAPTURES:
            for e in rec._live_captures[:-_MAX_LIVE_CAPTURES]: e[3] = e[4] = None; e[5].clear()
            del rec._live_captures[:-_MAX_LIVE_CAPTURES]
    mod, th, tgt = False, False, raw
    if rec is not None and torch.is_grad_enabled() and raw.requires_grad and rec.thermal_scale < 1.0:
        s = float(rec.thermal_scale); tgt, th, mod = raw * s + raw.detach() * (1.0 - s), True, True
        output = tgt if isinstance(output, torch.Tensor) else ((tgt, *output[1:]) if isinstance(output, tuple) else [tgt, *output[1:]])
    x_in, cd, hc, parts = _extract_axiom_inputs_parts(args, kwargs, int(tgt.shape[0]))
    _record_axiom_capture(output, x_in, cd, rec, has_cont_cond=hc, parts=parts, thermally_scaled=th)
    return output if mod else None

def _index_arg(val: Any, b: int, idx: torch.Tensor) -> Any:
    if isinstance(val, torch.Tensor): return val.detach().index_select(0, idx.to(val.device)) if (val.ndim >= 1 and val.shape[0] == b) else val.detach()
    return type(val)(_index_arg(v, b, idx) for v in val) if isinstance(val, (tuple, list)) else ({k: _index_arg(v, b, idx) for k, v in val.items()} if isinstance(val, dict) else val)

def _axiom_generator_hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
    rec, raw = getattr(module, "_axiom_record", None), _first_out(output)
    if (rec is None or rec._in_probe or rec._in_penalty or rec._in_protected_probe or not rec._enable_secant_probe or not module.training or not torch.is_grad_enabled()
            or not isinstance(raw, torch.Tensor) or not raw.is_floating_point() or not raw.requires_grad or raw.ndim < 2 or raw.shape[0] < 1 or int(raw[0].numel()) > 32 or fwAD.unpack_dual(raw).tangent is not None):
        return
    b, dev, nu = int(raw.shape[0]), raw.device, _PROBE_MAX_BATCH // 2
    idx = torch.arange(b, device=dev, dtype=torch.long) if b <= _PROBE_MAX_BATCH else torch.cat((
        u := ((torch.arange(nu, device=dev, dtype=torch.long) * b) // nu + int(float(rec.chord_phase) * b)) % b,
        torch.topk(raw.detach().reshape(b, -1).double().norm(dim=-1).index_fill(0, u, float("-inf")), k=_PROBE_MAX_BATCH - nu).indices), dim=0)
    pa, pk = tuple(_index_arg(a, b, idx) for a in args), {k: _index_arg(v, b, idx) for k, v in kwargs.items()}
    if (y_pre := eval_probe_forward(module, pa, pk, rec) if getattr(module, "_axiom_has_bn", False) else raw.detach().index_select(0, idx).to(torch.float32)) is not None:
        rec._g_probe_cache = (module, pa, pk, y_pre)

def _attach_hook(mod: Optional[nn.Module], record: AxiomStepRecord, fn: Any, flag: str) -> None:
    if isinstance(mod, nn.Module):
        mod._axiom_record, mod._axiom_has_bn = record, any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)) for m in mod.modules())
        if not getattr(mod, flag, False): mod.register_forward_hook(fn, with_kwargs=True); setattr(mod, flag, True)

_attach_axiom_critic_hook = lambda critic, record: _attach_hook(critic, record, _axiom_critic_hook, "_axiom_hook_registered")
_attach_axiom_generator_hook = lambda generator, record: _attach_hook(generator, record, _axiom_generator_hook, "_axiom_gen_hook_registered")

def bind_axiom_record(record: AxiomStepRecord, *readers: Any) -> None:
    for r in readers:
        if r is not None:
            r._axiom_record = record
            if hasattr(r, "record") and (not hasattr(r, "param_groups") or getattr(r, "critic", None) is not None): r.record = record

def _has_quantized_linear(mod: Any) -> bool:
    if not isinstance(mod, nn.Module): return False
    try: from ..low_precision import EmulatedLinear
    except Exception: EmulatedLinear = ()  # type: ignore[assignment]
    return any(isinstance(m, EmulatedLinear) or getattr(m, "format", None) in ("nvfp4", "mxfp8") or "Emulated" in type(m).__name__ for m in mod.modules())

def _bn_dual_bf16_hook(_m: nn.Module, _i: Tuple[Any, ...], out: Any) -> Any:
    if isinstance(out, torch.Tensor):
        p, t = fwAD.unpack_dual(out)
        if t is not None and (p.dtype != torch.bfloat16 or t.dtype != torch.bfloat16): return fwAD.make_dual(p.to(torch.bfloat16), t.to(torch.bfloat16))
    return out

def _dense_rademacher(x: torch.Tensor, phi: float, salt: int) -> torch.Tensor:
    h = (torch.arange(x.numel(), device=x.device, dtype=torch.int64) * 2654435761 + ((int(phi * 1073741824.0) + salt * 7919) & 0xFFFFFFFF) * 3266489917 + 374761393) & 0xFFFFFFFF
    h = (((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF); h = (((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF)
    return torch.where(((h ^ (h >> 16)) & 1) == 0, 1.0, -1.0).to(dtype=x.dtype).reshape_as(x)

_cslice = lambda val, b, s, n: val if (not isinstance(val, torch.Tensor) or val.ndim < 1 or val.shape[0] != b or n >= b) else (val[s : s + n] if s + n <= b else torch.cat((val[s:], val[: s + n - b]), dim=0))

class AxiomPenaltyKernel:
    """Compute-regime-dispatched zero-Jensen linear-R1+R2 + Law 4.5 floor + swap-invariant gauge."""

    def __init__(self, *, record: Optional[AxiomStepRecord] = None, coeff: float = 1.0, kappa: float = 1.0,
                 lambda_floor: float = 0.005, beta_ema: float = 1.5, gamma_gauge: float = 0.03, max_lambda: float = 2.0) -> None:
        self.record = AxiomStepRecord() if record is None else record
        self.coeff, self.kappa = _check_scalar(coeff, "coeff", 0.0), _check_scalar(kappa, "kappa", 0.0, slo=True)
        self.lambda_floor, self.beta_ema = _check_scalar(lambda_floor, "lambda_floor", 0.0, slo=True), _check_scalar(beta_ema, "beta_ema", 0.0)
        self.gamma_gauge, self.max_lambda = _check_scalar(gamma_gauge, "gamma_gauge", 0.0), _check_scalar(max_lambda, "max_lambda", self.lambda_floor)
        self._c_max, self._fwad_unsupported = math.log(max(1.0, self.max_lambda / self.lambda_floor)), set()

    def _ror(self, live: Callable[..., torch.Tensor], x: torch.Tensor, a: Tuple[Any, ...], kw: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        b, xr = int(x.shape[0]), x.detach().requires_grad_(True); y2, sq = _fmt_logits(live(xr, *a, **kw), b), None
        for k in range(int(y2.shape[1])):
            if (gk := torch.autograd.grad(y2[:, k].sum(), xr, create_graph=True, allow_unused=True)[0]) is not None:
                term = gk.reshape(b, -1).float().square().sum(dim=-1); sq = term if sq is None else (sq + term)
        return y2, (y2.new_zeros(b, dtype=torch.float32) if sq is None else sq / float(y2.shape[1]))

    def _fwad(self, live: Callable[..., torch.Tensor], xr: torch.Tensor, xf: torch.Tensor, a: Tuple[Any, ...], kw: Dict[str, Any],
              phi: float, *, mod: Optional[nn.Module], has_bn: bool, has_bd: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        br, bf, bn_m, bn_h = int(xr.shape[0]), int(xf.shape[0]), [], []
        bf16 = bool(has_bn and xr.is_cuda and isinstance(mod, nn.Module) and (getattr(mod, "bf16", False) or getattr(mod, "_axiom_bf16_autocast", False) or (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16)))
        if has_bn and isinstance(mod, nn.Module):
            for m in mod.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    bn_m.append((m, m.training)); m.eval()
                    if bf16: bn_h.append(m.register_forward_hook(_bn_dual_bf16_hook))
        def _dual(xs: torch.Tensor, ca: Tuple[Any, ...], ckw: Dict[str, Any], salt: int) -> Tuple[torch.Tensor, torch.Tensor]:
            vs = _dense_rademacher(xs, phi, salt); y, j = fwAD.unpack_dual(live(fwAD.make_dual(xs.to(torch.bfloat16) if bf16 else xs, vs.to(torch.bfloat16) if bf16 else vs), *ca, **ckw))
            if j is None: raise NotImplementedError("dual tangent dropped")
            return _fmt_logits(y, int(xs.shape[0])), _fmt_logits(j, int(xs.shape[0])).float()
        self.record._in_penalty = True
        try:
            sr, sf = int(phi * br) % br, int(((phi + 0.5) % 1.0) * bf) % bf
            with fwAD.dual_level():
                if not a and not kw and br == bf and br >= 4 and not has_bd:
                    b_reg = max(2, br // 4); brs = b_reg // 2
                    y2, j2 = _dual(torch.cat((_cslice(xr.detach(), br, sr, brs), _cslice(xf.detach(), bf, sf, b_reg - brs)), dim=0), (), {}, 1)
                    return y2[:brs], y2[brs:], 0.5 * (j2[:brs].square().mean() + j2[brs:].square().mean())
                brs, bfs = (br if has_bd else max(1, min(br, br // 4))), (bf if has_bd else max(1, min(bf, bf // 4)))
                yr, jr = _dual(_cslice(xr.detach(), br, sr, brs), tuple(_cslice(v, br, sr, brs) for v in a), {k: _cslice(v, br, sr, brs) for k, v in kw.items()}, 1)
                yf, jf = _dual(_cslice(xf.detach(), bf, sf, bfs), tuple(_cslice(v, bf, sf, bfs) for v in a), {k: _cslice(v, bf, sf, bfs) for k, v in kw.items()}, 2)
                return yr, yf, 0.5 * (jr.square().mean() + jf.square().mean())
        finally:
            self.record._in_penalty = False
            for h in bn_h: h.remove()
            for m, f in bn_m: m.training = f

    def penalty(self, live: Callable[..., torch.Tensor], x_real: torch.Tensor, x_fake: torch.Tensor, *args: Any,
                collect_stats: bool = False, critic_module: Optional[nn.Module] = None, **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.coeff == 0.0: self.record.clear_transient_captures(); return x_real.new_zeros(()), ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})
        self.record.calls, phi = self.record.calls + 1, float(self.record.chord_phase); self.record.chord_phase = (phi + _GOLDEN_CONJ) % 1.0
        mod = critic_module if isinstance(critic_module, nn.Module) else (live if isinstance(live, nn.Module) else None)
        cid, has_bn = (id(mod) if mod is not None else id(live)), (isinstance(mod, nn.Module) and any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in mod.modules()))
        has_bd = isinstance(mod, nn.Module) and any("BatchDistance" in type(m).__name__ for m in mod.modules())
        use_ror = (x_real.ndim == 2 and int(x_real[0].numel()) <= 16) or _has_quantized_linear(mod) or (cid in self._fwad_unsupported)
        csig, kr, kf, stp = _cond_sig((x_real, *args), kwargs), _tensor_id(x_real), _tensor_id(x_fake), self.record.observed_steps
        dr_l = next((e[4] for e in reversed(self.record._live_captures) if e[0] == stp and e[1] == kr and e[2] == csig and e[5] and e[4] is not None), None)
        df_l = next((e[4] for e in reversed(self.record._live_captures) if e[0] == stp and e[1] == kf and e[2] == csig and e[5] and e[4] is not None), None)
        reused = dr_l is not None and df_l is not None
        try:
            if not use_ror:
                try: dr_s, df_s, s_hat = self._fwad(live, x_real, x_fake, args, kwargs, phi, mod=mod, has_bn=has_bn, has_bd=has_bd)
                except (RuntimeError, NotImplementedError, TypeError): self._fwad_unsupported.add(cid); use_ror = True
            if use_ror:
                self.record._in_penalty = True
                try:
                    br, bf = int(x_real.shape[0]), int(x_fake.shape[0])
                    if not has_bn and not has_bd and not args and not kwargs and br == bf:
                        y2, sq = self._ror(live, torch.cat((x_real.detach(), x_fake.detach()), dim=0), (), {}); dr_s, df_s, s_hat = y2[:br], y2[br:], 0.5 * (sq[:br].mean() + sq[br:].mean())
                    else: dr_s, sqr = self._ror(live, x_real, args, kwargs); df_s, sqf = self._ror(live, x_fake, args, kwargs); s_hat = 0.5 * (sqr.mean() + sqf.mean())
                finally: self.record._in_penalty = False
            dr2, df2 = (dr_l, df_l) if reused else (dr_s, df_s)
            lam = (self.lambda_floor * torch.exp((self.beta_ema * torch.relu(self.record._dev_ema("ema_grad_sq", s_hat.device) - self.kappa * self.kappa)).clamp_max(self._c_max))).to(s_hat.dtype)
            self.record._observe_grad_sq_tensor(s_hat.detach()); pl = 0.5 * self.coeff * lam * s_hat
            ml = 0.5 * (dr2.float().mean() + df2.float().mean()) if self.gamma_gauge > 0.0 else s_hat.new_zeros(())
            pg, s_th = (0.5 * self.coeff * self.gamma_gauge * ml.square() if self.gamma_gauge > 0.0 else s_hat.new_zeros(())), float(self.record.thermal_scale)
            if s_th < 1.0:
                pl = pl * s_th + pl.detach() * (1.0 - s_th) if pl.requires_grad else pl
                if pg.requires_grad:
                    gu = ((ml.detach().abs() - 0.5) / 1.0).clamp(0.0, 1.0); sg = s_th + (gu * gu * (3.0 - 2.0 * gu)) * (math.sqrt(max(0.0, s_th)) - s_th); pg = pg * sg + pg.detach() * (1.0 - sg)
            pen = pl + pg
            if not collect_stats: return pen, {}
            pv, sv, lv = torch.stack((pen.detach().float(), s_hat.detach().float(), lam.detach().float())).tolist()
            return pen, {"applied": True, "pen": float(pv), "s_hat": float(sv), "mean_grad_norm": math.sqrt(max(0.0, float(sv))), "lambda_eff": float(lv),
                         "center": self.kappa, "regime": "ror" if use_ror else "fwad", "s_thermal": s_th, "live_endpoints_reused": bool(reused)}
        finally: self.record.clear_transient_captures()

class AxiomCriticPenalty:
    """Public callable wrapper matching GANTrainer, WordFixture, DDGAN, and BehaviorComponents."""
    ema_critic = None
    def __init__(self, optimizer_or_recipe: Any, optimizer: Any = None, *, coeff: Optional[float] = None, kappa: Optional[float] = None, output: Any = None, collect_stats: bool = False, **options: Any) -> None:
        self.recipe, self.optimizer = (None, optimizer_or_recipe) if optimizer is None else (optimizer_or_recipe, optimizer)
        if not hasattr(self.optimizer, "param_groups"): raise TypeError("AxiomCriticPenalty requires a critic optimizer")
        rec = getattr(self.optimizer, "record", None) or AxiomStepRecord(); self.optimizer.record, self.optimizer._axiom_record, self.critic = rec, rec, getattr(self.optimizer, "critic", None)
        _attach_axiom_critic_hook(self.critic, rec)
        k_opts = {"coeff": float(getattr(self.recipe, "reg_coeff", 1.0)) if coeff is None else float(coeff), "kappa": float(getattr(self.recipe, "reg_kappa", 1.0)) if kappa is None else float(kappa),
                  **{k: options[k] for k in ("coeff", "kappa", "lambda_floor", "beta_ema", "gamma_gauge", "max_lambda") if options.get(k) is not None}}
        self.regularizer = AxiomPenaltyKernel(record=rec, **k_opts)
        self.initial_coeff, self.output, self.collect_stats, self.last_stats = self.regularizer.coeff, (_first_out if output is None else output), bool(collect_stats), {}
    def __call__(self, critic: Callable[..., torch.Tensor], x_real: torch.Tensor, x_fake: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        pen, self.last_stats = self.regularizer.penalty(lambda x, *a, **kw: self.output(critic(x, *a, **kw)), x_real, x_fake, *args, collect_stats=self.collect_stats, critic_module=critic if isinstance(critic, nn.Module) else self.critic, **kwargs)
        return pen
    def diagnostics(self) -> Dict[str, Any]:
        r = self.regularizer.record
        return {"axiom_calls": int(r.calls), "axiom_ema_grad_sq": float(r.ema_grad_sq), "axiom_chord_phase": float(r.chord_phase), "axiom_thermal_scale": float(r.thermal_scale), "axiom_step_scale": float(r.step_scale)}

__all__ = ["AxiomCriticPenalty", "AxiomPenaltyKernel", "AxiomStepRecord", "_attach_axiom_critic_hook", "_attach_axiom_generator_hook",
           "_extract_axiom_inputs_parts", "_record_axiom_capture", "bind_axiom_record", "eval_probe_forward"]
