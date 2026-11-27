"""Axiom L0 state & captures."""

from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from copy import deepcopy
import math
import random
import sys
import weakref

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

SCHEMA_VERSION = 2

_THERMAL_KEYS = "ema_ed ema_mu_sq ema_spread ema_q90 ema_hmax ema_tail".split()
_EMA_BOOLS = frozenset("grad_sq_initialized margin_initialized".split())
_BOOL_FIELDS = (
    *_EMA_BOOLS,
    *"anchor_started ell_initialized thermal_initialized thermal_locked cond_initialized het_initialized".split(),
)
_DICT_FIELDS = "chart_state atomic_ledger cond_fiber_bank".split()
_SCALAR_SPEC = {
    "ema_grad_sq": (1.0, 0.0),
    "ema_margin": (0.0, -60.0, 60.0),
    "ema_data_scale": (1.0, 0.0, None, 1),
    "last_data_scale": (1.0, 0.0, None, 1),
    "ema_push_res": (1.0, 0.0),
    "ell_ema": (0.0, 0.0),
    "chord_phase": (0.0, 0.0, 1.0, 0, 1),
    "thermal_scale": (1.0, 1e-5, 1.0),
    "step_scale": (1.0, 0.0, 1.0, 1),
    "cond_ema": (0.25, -0.5, 2.0),
    "het_ema": (1.0, 0.0),
    "g_func_gain": (1.0, 0.0, 1.0, 1),
    **{k: (0.0 if k == "ema_tail" else 1.0,) for k in _THERMAL_KEYS},
}
_EMA_DEFAULTS = {**dict.fromkeys(_EMA_BOOLS, False), **{k: v[0] for k, v in list(_SCALAR_SPEC.items())[:6]}}
_FIELD_DEFAULTS = {
    "schema": SCHEMA_VERSION,
    "calls": 0,
    "observed_steps": 0,
    "last_ratio": None,
    **dict.fromkeys(_BOOL_FIELDS, False),
    **{k: v[0] for k, v in _SCALAR_SPEC.items()},
    **dict.fromkeys(_DICT_FIELDS),
}
_RECORD_KEYS = frozenset(_FIELD_DEFAULTS)
_TRANSIENT_DEFAULTS = {
    **dict.fromkeys("_in_penalty _in_probe _in_protected_probe _simplicial_step".split(), False),
    "_enable_secant_probe": True,
    "_disc_gate": 0.0,
    **dict.fromkeys(
        "_last_probe_actual_input _g_probe_cache _active_prior_table _active_batch_rows _cond_bank_obj".split()
    ),
}


def first_tensor_output(output):
    return output[0] if isinstance(output, (tuple, list)) and output else output


def format_logits_2d(logits, batch_size):
    return logits.reshape(batch_size, 1 if logits.ndim < 2 else -1)


def _ptr(t):
    try:
        return t.untyped_storage().data_ptr() if (isinstance(t, torch.Tensor) and t.ndim and t.numel()) else 0
    except RuntimeError:
        return 0


def tensor_id(tensor):
    p = _ptr(tensor)
    return (p, tensor.storage_offset(), tuple(tensor.shape), str(tensor.device)) if p else None


def capture_key(tensor):
    p = _ptr(tensor)
    return (p, tensor.storage_offset(), tensor.numel(), tensor.shape[0]) if p else None


def cond_signature(args, kwargs):
    def s(v):
        return ("T", tensor_id(v), v.dtype) if isinstance(v, torch.Tensor) else ("V", type(v), repr(v))

    return (*map(s, args[1:]), *((k, s(kwargs[k])) for k in sorted(kwargs)))


def check_scalar(val, name, lo=None, hi=None, strict_lo=False, strict_hi=False):
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ValueError(name)
    v = float(val)
    if (lo is not None and (v <= lo if strict_lo else v < lo)) or (
        hi is not None and (v >= hi if strict_hi else v > hi)
    ):
        raise ValueError(name)
    return v


def _walk_payload(obj, fn, copy_leaf=True):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _walk_payload(v, fn, copy_leaf) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_walk_payload(x, fn, copy_leaf) for x in obj)
    return deepcopy(obj) if copy_leaf else obj


def _cpu_walk(v):
    return None if v is None else _walk_payload(v, lambda t: t.detach().cpu().clone())


def _validate_payload(obj, name):
    if obj is None or type(obj) in (bool, int, str):
        return
    if (type(obj) is float and math.isfinite(obj)) or (
        isinstance(obj, torch.Tensor) and (not obj.is_floating_point() or bool(torch.isfinite(obj).all()))
    ):
        return
    if isinstance(obj, (list, tuple, dict)):
        if isinstance(obj, dict) and any(not isinstance(k, (str, int, tuple)) for k in obj):
            raise ValueError(name)
        for x in obj.values() if isinstance(obj, dict) else obj:
            _validate_payload(x, name)
        return
    raise ValueError(name)


class WeakTensorMap:
    def __init__(self):
        self._m = {}

    def _entry(self, t):
        e = self._m.get(id(t)) if isinstance(t, torch.Tensor) else None
        return e if (e is not None and e[0]() is t) else None

    def set(self, tensor, value):
        k = id(tensor)
        self._m[k] = (weakref.ref(tensor, lambda d: self._m.get(k, (None,))[0] is d and self._m.pop(k)), value)

    def get(self, tensor, default=None):
        return e[1] if (e := self._entry(tensor)) is not None else default

    def pop(self, tensor, default=None):
        return self._m.pop(id(tensor))[1] if self._entry(tensor) is not None else default

    def __contains__(self, tensor):
        return self._entry(tensor) is not None

    def __len__(self):
        return sum(r() is not None for r, _ in list(self._m.values()))

    def clear(self):
        self._m.clear()


_MOD_REC, _MOD_BN, _MODULE_PENDING_SOURCE, _C_HOOKS, _G_HOOKS = (weakref.WeakKeyDictionary() for _ in range(5))
_PRIOR_ROWS, _TENSOR_REC = WeakTensorMap(), WeakTensorMap()


def set_module_record(module, record):
    _MOD_REC[module] = record


def get_module_record(module):
    if not isinstance(module, nn.Module):
        return None
    return next(
        (
            _MOD_REC[c]
            for c in (module, getattr(module, "critic", None), getattr(module, "base", None))
            if isinstance(c, nn.Module) and c in _MOD_REC
        ),
        None,
    )


set_tensor_record = _TENSOR_REC.set
get_tensor_record = _TENSOR_REC.get
get_active_prior_rows = _PRIOR_ROWS.get


def get_axiom_record(obj):
    if obj is None or isinstance(obj, AxiomStepRecord):
        return obj
    if isinstance(obj, torch.Tensor):
        return _TENSOR_REC.get(obj)
    if isinstance(obj, nn.Module) and (r := get_module_record(obj)) is not None:
        return r
    return next(
        (
            c
            for a in ("record", "axiom_record", "_axiom_record")
            if isinstance(c := getattr(obj, a, None), AxiomStepRecord)
        ),
        None,
    )


def set_active_prior_rows(table, rows):
    _PRIOR_ROWS.set(table, rows.detach())


def clear_active_prior_rows(table):
    _PRIOR_ROWS.pop(table, None)


def recover_active_prior_rows(x_fake, record=None):
    t, r = (record._active_prior_table, record._active_batch_rows) if record else (None, None)
    if t is not None and r is not None:
        return t, r
    if (fn := getattr(x_fake, "grad_fn", None)) is None or not _PRIOR_ROWS:
        return None, None
    q, seen = [(fn, 0)], {id(fn)}
    while q and len(seen) <= 256:
        node, d = q.pop(0)
        var = getattr(node, "variable", None)
        if isinstance(var, torch.Tensor) and (rows := _PRIOR_ROWS.get(var)) is not None:
            if record:
                _TENSOR_REC.set(var, record)
            return var, rows
        if d < 8:
            for nxt, _ in getattr(node, "next_functions", ()):
                if nxt is not None and id(nxt) not in seen:
                    seen.add(id(nxt))
                    q.append((nxt, d + 1))
    return None, None


class CaptureEntry(
    namedtuple(
        "_CE", "x_in cond record has_cont_cond parts thermally_scaled", defaults=(None, None, False, None, False)
    )
):
    def detached(self):
        return self._replace(
            **{k: _walk_payload(getattr(self, k), torch.Tensor.detach, False) for k in ("x_in", "cond", "parts")}
        )

    def nbytes(self):
        return sum(t.nbytes for t in (self.x_in, self.cond, *(self.parts or ())) if t is not None)


class CaptureRegistry:
    def __init__(self, *, max_ring=16, max_sample_bytes=16_384, max_total_bytes=2 << 20, max_live=6):
        self._mr, self._ms, self._mt, self._ml = max_ring, max_sample_bytes, max_total_bytes, max_live
        self._ring, self.ring_bytes, self._live = OrderedDict(), 0, []

    @property
    def ring_size(self):
        return len(self._ring)

    @property
    def live_count(self):
        return len(self._live)

    def put_detached(self, logits, entry):
        key = capture_key(logits)
        if key is None or entry.x_in.nbytes // max(1, entry.x_in.shape[0]) > self._ms:
            return False
        det = entry.detached()
        nb = det.nbytes()
        if nb > self._mt:
            return False
        self.ring_bytes -= self._ring.pop(key, (None, 0))[1]
        while self._ring and (len(self._ring) >= self._mr or self.ring_bytes + nb > self._mt):
            self.ring_bytes -= self._ring.popitem(last=False)[1][1]
        self._ring[key] = (det, nb)
        self.ring_bytes += nb
        return True

    def lookup_detached(self, logits):
        hit = self._ring.get(capture_key(logits))
        return None if hit is None else hit[0]

    def record_live_endpoint(self, *, step, source_key, cond_sig, x_in=None, raw_logits):
        alive = [True]
        raw_logits.register_hook(lambda _g, f=alive: f.clear())
        ref = raw_logits.float() if raw_logits.dtype == torch.bfloat16 else raw_logits
        self._live.append([step, source_key, cond_sig, ref, alive])
        while len(self._live) > self._ml:
            self._live.pop(0)[4].clear()

    def find_live_endpoint(self, step, x_tensor, args, kwargs):
        if (sk := tensor_id(x_tensor)) is None:
            return None
        cs = cond_signature((x_tensor, *args), kwargs)
        for st, k, c, ref, alive in reversed(self._live):
            if st == step and k == sk and c == cs and alive:
                return ref
        return None

    def clear(self):
        for it in self._live:
            it[4].clear()
        self._live.clear()
        self._ring.clear()
        self.ring_bytes = 0


def _is_batch(x, b):
    return isinstance(x, torch.Tensor) and x.ndim >= 1 and x.shape[0] == b and x.numel() > 0


def extract_axiom_inputs_parts(args, kwargs, batch_size):
    fl, ds = [], []
    for x in (*args, *(kwargs[k] for k in sorted(kwargs or ()))):
        if _is_batch(x, batch_size):
            if x.is_floating_point():
                fl.append(x)
            elif not x.is_complex():
                ds.append(x.detach().to(torch.int64).reshape(batch_size, -1))
    if not fl:
        return None, None, False, None
    cond = None if not ds else (ds[0] if len(ds) == 1 else torch.cat(ds, -1))
    if len(fl) > 1:
        parts = tuple(t.reshape(batch_size, -1) for t in fl)
        return torch.cat(parts, -1), cond, True, parts
    return fl[0], cond, False, None


def record_axiom_capture(output, entry):
    tgt = first_tensor_output(output)
    if _is_batch(tgt, entry.x_in.shape[0]):
        tgt._axiom_capture = entry
        if entry.record:
            entry.record.captures.put_detached(tgt, entry)


def lookup_axiom_capture(logits, fallback_record=None):
    e = getattr(logits, "_axiom_capture", None)
    if isinstance(e, CaptureEntry):
        return e._replace(record=fallback_record) if (e.record is None and fallback_record) else e
    return fallback_record.lookup_capture(logits) if fallback_record else None


def _ema_prop(name):
    is_b = name in _EMA_BOOLS

    def _get(self):
        if name in self._ema_t:
            self._flush_ema()
        return self._ema[name]

    def _set(self, val):
        self._ema[name] = bool(val) if is_b else float(val)
        self._ema_t.pop(name, None)

    return property(_get, _set)


def _validate_header(s):
    if (
        not isinstance(s, dict)
        or set(s) != _RECORD_KEYS
        or type(s["schema"]) is not int
        or s["schema"] != SCHEMA_VERSION
    ):
        raise ValueError("schema")
    if (
        any(type(s[k]) is not int or s[k] < 0 for k in ("calls", "observed_steps"))
        or any(type(s[k]) is not bool for k in _BOOL_FIELDS)
        or s["anchor_started"]
    ):
        raise ValueError("field")


def validate_axiom_record_state(state):
    _validate_header(state)
    lr = None if state["last_ratio"] is None else check_scalar(state["last_ratio"], "last_ratio", lo=0.0)
    sc = {k: check_scalar(state[k], k, *sp[1:]) for k, sp in _SCALAR_SPEC.items()}
    for k in _DICT_FIELDS:
        if (p := state[k]) is not None:
            if not isinstance(p, dict):
                raise ValueError(k)
            _validate_payload(p, k)
    return lr, sc


class AxiomStepRecord:
    SCHEMA = SCHEMA_VERSION
    validate_state_dict = staticmethod(validate_axiom_record_state)

    def __init__(self):
        self._ema = _EMA_DEFAULTS.copy()
        self.__dict__.update(_FIELD_DEFAULTS)
        self._reset_transient()

    def _reset_transient(self):
        self._ema_t, self.captures = {}, CaptureRegistry()
        self.__dict__.update(_TRANSIENT_DEFAULTS)

    def lookup_capture(self, logits):
        return self.captures.lookup_detached(logits)

    def find_live_endpoint(self, x_tensor, args=(), kwargs=None):
        return self.captures.find_live_endpoint(self.observed_steps, x_tensor, args, kwargs or {})

    def clear_transient_captures(self):
        self.captures.clear()
        self._last_probe_actual_input = None

    def record_step(self, optimizer=None):
        self.observed_steps += 1
        self.clear_transient_captures()
        self._active_prior_table = self._active_batch_rows = None

    def take_pending_ema_tensors(self):
        return [*self._ema_t], [*self._ema_t.values()]

    def apply_flushed_ema_values(self, keys, values):
        for k, v in zip(keys, values):
            if k in self._ema:
                self._ema[k] = v > 0.5 if k in _EMA_BOOLS else float(v)
                self._ema_t.pop(k, None)

    def _flush_ema(self):
        ks, ts = self.take_pending_ema_tensors()
        if ts:
            self.apply_flushed_ema_values(
                ks,
                torch.stack([t.double() for t in ts]).tolist()
                if all(t.device == ts[0].device for t in ts)
                else [t.item() for t in ts],
            )

    def dev_ema(self, name, device):
        cur = self._ema_t.get(name)
        if cur is not None:
            if cur.device == device:
                return cur
            self._flush_ema()
        return torch.full((), float(self._ema[name]), dtype=torch.float64, device=device)

    _dev_ema = dev_ema

    def _ema_upd(self, key, val, decay, was, init_key=None):
        fin, dev = torch.isfinite(val), val.device
        prev = self.dev_ema(key, dev)
        self._ema_t[key] = torch.where(fin, torch.where(was, decay * prev + (1 - decay) * val, val), prev)
        if init_key:
            pi = self.dev_ema(init_key, dev)
            self._ema_t[init_key] = torch.where(fin, pi.new_ones(()), pi)

    @torch.no_grad()
    def observe_grad_sq_tensor(self, s_hat_det):
        obs = s_hat_det.detach().double().clamp(0, 20)
        self._ema_upd(
            "ema_grad_sq", obs, 0.95, self.dev_ema("grad_sq_initialized", obs.device) > 0.5, "grad_sq_initialized"
        )

    _observe_grad_sq_tensor = observe_grad_sq_tensor

    @torch.no_grad()
    def observe_margin_and_scale(self, real_logits, fake_logits, x_real=None, x_fake=None):
        if self._in_penalty or self._in_probe or self._in_protected_probe:
            return
        m = (real_logits.detach().float().mean().double() - fake_logits.detach().float().mean().double()).clamp(-50, 50)
        was = self.dev_ema("margin_initialized", m.device) > 0.5
        self._ema_upd("ema_margin", m, 0.9, was, "margin_initialized")
        if x_real is None or not x_real.numel():
            return
        dev = x_real.device
        xr = x_real.detach().reshape(x_real.shape[0], -1).double()
        xm = xr.abs().amax()
        sx = torch.where(xm > 0, xm, xm.new_ones(()))
        sc = ((xr / sx).square().sum(-1).mean().sqrt() * sx).clamp(1e-30, 1e30)
        psc, lsc = self.dev_ema("ema_data_scale", dev), self.dev_ema("last_data_scale", dev)
        lp, lc, ll = (torch.log2(t.clamp_min(1e-30)) for t in (psc, sc, lsc))
        ok, dl = torch.isfinite(sc) & (xm > 0), lc - lp
        sl = torch.where((dl.abs() > 0.25) & ((lc - ll).abs() <= 1.5), 0.8 * dl, 0.05 * dl.clamp(-1, 1))
        nxt = torch.where(ok, torch.where(was, torch.exp2(lp + sl), sc), psc)
        self._ema_t["ema_data_scale"], self._ema_t["last_data_scale"] = nxt, torch.where(ok, sc, lsc)
        if x_fake is not None and x_fake.shape == x_real.shape:
            xf = x_fake.detach().reshape(x_fake.shape[0], -1).double()
            raw = (
                ((xf.mean(0) - xr.mean(0)).norm() + 0.5 * (xf.std() - xr.std()).abs()) / nxt.clamp_min(1e-30)
            ).clamp_max(10)
            self._ema_upd("ema_push_res", raw, 0.92, was)

    @property
    def thermal_ema(self):
        un = min(1.0, max(0.0, -self.ema_spread) / 0.03) * 0.6
        return max(
            0.0,
            0.5 * self.ema_ed,
            math.sqrt(max(0.0, self.ema_mu_sq)),
            0.6 * abs(self.ema_spread),
            self.ema_q90,
            -self.ema_q90 * un,
            self.ema_hmax,
            -self.ema_hmax * un,
            self.ema_tail,
        )

    ema_residual = thermal_ema

    def observe_thermal(
        self,
        r_ed,
        r_mu_sq,
        r_spread,
        r_q90,
        r_hmax,
        *,
        r_tail=0.0,
        tau=0.084,
        ema_decay=0.96,
        s_min=1e-5,
        shock_mu_ratio=0.35,
        shock_ed_floor=0.30,
        lock_hysteresis=0.0,
    ):
        ed, mu, tau, d = float(r_ed), float(r_mu_sq), float(tau), float(ema_decay)
        tp = max(0.0, tau)
        lim, led, sh = max(0.2, 2 * tp), max(0.7, 4 * tp), max(shock_ed_floor, 4 * tp * tp)
        rst = not self.thermal_initialized or (self.thermal_scale < 0.99 and (ed > sh or mu > shock_mu_ratio * sh))
        for k, v, lk in zip(_THERMAL_KEYS, (ed, mu, r_spread, r_q90, r_hmax, r_tail), (led, lim * lim, *(lim,) * 4)):
            c = max(-lk, min(float(v), lk))
            setattr(self, k, c if rst else d * getattr(self, k) + (1 - d) * c)
        if rst:
            self.thermal_initialized, self.thermal_locked = True, False
        if tau <= 0:
            self.thermal_scale, self.thermal_locked = 1.0, False
            return 1.0
        lo = min(0.92, 0.5 + (lock_hysteresis if self.thermal_locked else 0.0)) * tau
        u = min(1.0, max(0.0, (self.thermal_ema - lo) / max(1e-8, tau - lo)))
        th = u * u * (3 - 2 * u)
        sf = max(1e-5, s_min)
        sc = 1.0 if th >= 1 else (sf if th <= 0 else 0.04 * th / math.sqrt(max(1e-8, 1 - th * th)))
        self.thermal_scale = max(sf, min(1.0, sc))
        self.thermal_locked = self.thermal_scale <= 1e-4
        return self.thermal_scale

    def restore_device_state(self, device):
        for a in _DICT_FIELDS[:2]:
            if (v := getattr(self, a)) is not None:
                setattr(self, a, _walk_payload(v, lambda t: t.to(device)))

    def state_dict(self):
        self._flush_ema()
        b = self._cond_bank_obj
        pkt = b.to_packet() if hasattr(b, "to_packet") else self.cond_fiber_bank
        return {
            "schema": SCHEMA_VERSION,
            "last_ratio": None if self.last_ratio is None else float(self.last_ratio),
            **{k: int(getattr(self, k)) for k in ("calls", "observed_steps")},
            **{k: bool(getattr(self, k)) for k in _BOOL_FIELDS},
            **{k: float(getattr(self, k)) for k in _SCALAR_SPEC},
            **{k: _cpu_walk(pkt if k == "cond_fiber_bank" else getattr(self, k)) for k in _DICT_FIELDS},
        }

    @staticmethod
    def base_state(state):
        validate_axiom_record_state(state)
        return {"lr_max": 0.0, "lr_last": None, **{k: state[k] for k in ("anchor_started", "calls", "observed_steps")}}

    def load_state_dict(self, state):
        lr, sc = validate_axiom_record_state(state)
        ds = {k: _cpu_walk(state[k]) for k in _DICT_FIELDS}
        self._reset_transient()
        hdr = {
            "schema": SCHEMA_VERSION,
            "last_ratio": lr,
            **{k: state[k] for k in ("calls", "observed_steps", *_BOOL_FIELDS)},
        }
        for k, v in (hdr | sc | ds).items():
            setattr(self, k, v)

    def __copy__(self):
        d = AxiomStepRecord()
        d.load_state_dict(self.state_dict())
        return d

    def __deepcopy__(self, memo):
        d = self.__copy__()
        memo[id(self)] = d
        d._enable_secant_probe = self._enable_secant_probe
        return d


for _k in _EMA_DEFAULTS:
    _FIELD_DEFAULTS.pop(_k, None)
    setattr(AxiomStepRecord, _k, _ema_prop(_k))


def _probe_env(mod, args, kwargs):
    ps = tuple(mod.parameters())
    bf16 = bool(
        ps and ps[0].is_cuda and any(getattr(mod, f"_{p}_bf16_autocast", False) for p in ("particlegan", "axiom"))
    )
    if not torch.cuda.is_initialized():
        return bf16, []
    xs = (*ps, *mod.buffers(), *args, *kwargs.values())
    return bf16, sorted(
        {torch.cuda.current_device(), *(x.device.index for x in xs if isinstance(x, torch.Tensor) and x.is_cuda)}
    )


@torch.no_grad()
def eval_probe_forward(module, args, kwargs, record=None):
    ms = [(m, m.training) for m in module.modules()]
    pp = bool(record._in_probe) if record else False
    pr, nm = random.getstate(), sys.modules.get("numpy")
    nr = nm.random.get_state() if hasattr(nm, "random") else None
    bf16, devs = _probe_env(module, args, kwargs)
    if record:
        record._in_probe = True
    try:
        module.eval()
        with torch.random.fork_rng(devices=devs), torch.autocast("cuda", torch.bfloat16, bf16):
            tgt = first_tensor_output(module(*args, **kwargs))
        return tgt.detach().float() if isinstance(tgt, torch.Tensor) else None
    finally:
        if record:
            record._in_probe = pp
        for m, wt in ms:
            m.training = wt
        random.setstate(pr)
        if nr is not None:
            nm.random.set_state(nr)


@contextmanager
def axiom_source_context(critic, source):
    tgt = next(
        (m for m in (critic, getattr(critic, "critic", None)) if isinstance(m, nn.Module) and m in _MOD_REC), None
    )
    if tgt is None or source is None:
        yield
        return
    has, prev = tgt in _MODULE_PENDING_SOURCE, _MODULE_PENDING_SOURCE.get(tgt)
    _MODULE_PENDING_SOURCE[tgt] = tensor_id(source)
    try:
        yield
    finally:
        if has:
            _MODULE_PENDING_SOURCE[tgt] = prev
        else:
            _MODULE_PENDING_SOURCE.pop(tgt, None)


def override_axiom_capture_input(out, x, *, critic=None, args=(), kwargs=None):
    tgt = first_tensor_output(out)
    if not _is_batch(tgt, x.shape[0]):
        return None
    prev = getattr(tgt, "_axiom_capture", None)
    has_p = isinstance(prev, CaptureEntry)
    rec = (has_p and prev.record) or get_module_record(critic)
    if (not has_p and rec is None) or (rec and (rec._in_penalty or rec._in_probe)):
        return None
    b = tgt.shape[0]
    if has_p:
        parts = (x.reshape(b, -1), *prev.parts[1:]) if (prev.parts and len(prev.parts) > 1) else None
        entry = prev._replace(x_in=x if parts is None else torch.cat(parts, -1), record=rec, parts=parts)
    else:
        x_in, cond, hcc, parts = extract_axiom_inputs_parts((x, *args), kwargs, b)
        if x_in is None:
            return None
        entry = CaptureEntry(x_in, cond, rec, hcc, parts)
    record_axiom_capture(out, entry)
    return entry


def _maybe_record_live(mod, args, kwargs, x0, raw, rec):
    if x0 is None or not (mod.training and torch.is_grad_enabled() and raw.requires_grad) or x0.requires_grad:
        return
    if any(isinstance(v, torch.Tensor) and v.requires_grad for v in (*args[1:], *kwargs.values())):
        return
    if sk := _MODULE_PENDING_SOURCE.get(mod) or tensor_id(x0):
        rec.captures.record_live_endpoint(
            step=rec.observed_steps, source_key=sk, cond_sig=cond_signature(args, kwargs), raw_logits=raw
        )


def _scale_thermal(output, raw, rec):
    if not (torch.is_grad_enabled() and raw.requires_grad and rec.thermal_scale < 1.0):
        return output, raw, False
    s = rec.thermal_scale
    tgt = raw * s + raw.detach() * (1 - s)
    return (type(output)([tgt, *output[1:]]) if isinstance(output, (tuple, list)) else tgt), tgt, True


def _axiom_critic_hook(module, args, kwargs, output):
    if (rec := _MOD_REC.get(module)) is None:
        return None
    x0 = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    if rec._in_probe or rec._in_penalty:
        if rec._in_penalty and x0 is not None:
            rec._last_probe_actual_input = x0.detach()
        return None
    raw = first_tensor_output(output)
    if not isinstance(raw, torch.Tensor) or raw.ndim < 1:
        return None
    _maybe_record_live(module, args, kwargs, x0, raw, rec)
    output, tgt, scaled = _scale_thermal(output, raw, rec)
    x_in, cond, hcc, parts = extract_axiom_inputs_parts(args, kwargs, tgt.shape[0])
    if x_in is not None:
        record_axiom_capture(output, CaptureEntry(x_in, cond, rec, hcc, parts, scaled))
    return output if scaled else None


def _probe_indices(raw, phase):
    if not (
        isinstance(raw, torch.Tensor)
        and raw.is_floating_point()
        and raw.requires_grad
        and raw.ndim >= 2
        and raw.shape[0]
    ):
        return None
    cap = min(32, 1024 // max(1, raw[0].numel()))
    if cap < 4 or fwAD.unpack_dual(raw).tangent is not None:
        return None
    b, dev = raw.shape[0], raw.device
    if b <= cap:
        return torch.arange(b, device=dev)
    nu = cap // 2
    u = ((torch.arange(nu, device=dev) * b) // nu + int(phase * b)) % b
    nrms = raw.detach().flatten(1).double().norm(dim=-1).index_fill(0, u, -math.inf)
    return torch.cat((u, torch.topk(nrms, cap - nu).indices))


def _axiom_generator_hook(module, args, kwargs, output):
    rec = _MOD_REC.get(module)
    if not (rec and rec._enable_secant_probe and module.training and torch.is_grad_enabled()):
        return
    if rec._in_probe or rec._in_penalty or rec._in_protected_probe:
        return
    raw = first_tensor_output(output)
    if (idx := _probe_indices(raw, rec.chord_phase)) is None:
        return
    b = raw.shape[0]
    pa, pk = _walk_payload(
        (args, kwargs),
        lambda t: t.detach().index_select(0, idx.to(t.device)) if (t.ndim and t.shape[0] == b) else t.detach(),
        False,
    )
    y = eval_probe_forward(module, pa, pk, rec) if _MOD_BN.get(module) else raw.detach().index_select(0, idx).float()
    if y is not None:
        rec._g_probe_cache = (module, pa, pk, y)


def _attach_hook(mod, record, hook, handles):
    if isinstance(mod, nn.Module):
        _MOD_REC[mod] = record
        _MOD_BN[mod] = any(
            isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)) for m in mod.modules()
        )
        if mod not in handles:
            handles[mod] = mod.register_forward_hook(hook, with_kwargs=True)


def _attach_axiom_critic_hook(critic, record):
    _attach_hook(critic, record, _axiom_critic_hook, _C_HOOKS)


def _attach_axiom_generator_hook(generator, record):
    _attach_hook(generator, record, _axiom_generator_hook, _G_HOOKS)


attach_axiom_critic_hook = _attach_axiom_critic_hook
attach_axiom_generator_hook = _attach_axiom_generator_hook


def bind_axiom_record(record, *readers):
    for r in readers:
        if isinstance(r, torch.Tensor):
            _TENSOR_REC.set(r, record)
        elif r is not None:
            is_m = isinstance(r, nn.Module)
            if is_m:
                _MOD_REC[r] = record
            else:
                r._axiom_record = record
            is_g = not is_m and hasattr(r, "param_groups") and getattr(r, "critic", None) is None
            for a in ("record", "axiom_record", "_axiom_record"):
                if hasattr(r, a) and not (a == "record" and is_g):
                    setattr(r, a, record)


def bind_axiom_optimizers(
    opt_g, opt_d, *, generator=None, critic=None, encoder=None, prior=None, base_prior=None, enable_secant_probe=False
):
    rec = getattr(opt_d, "record", None)
    if not isinstance(rec, AxiomStepRecord):
        return None
    bind_axiom_record(
        rec,
        opt_g,
        opt_d,
        prior,
        base_prior,
        *(z for m in (prior, base_prior) if isinstance(z := getattr(m, "z", None), torch.Tensor)),
    )
    getattr(opt_g, "bind_module_layouts", lambda _: None)((generator, encoder))
    if generator is not None:
        ps = generator.parameters()
        if hasattr(opt_g, "set_generator_probe_parameters"):
            opt_g.set_generator_probe_parameters(ps)
        elif hasattr(opt_g, "_gen_param_ids"):
            opt_g._gen_param_ids = {id(p) for p in ps}
        if enable_secant_probe:
            rec._enable_secant_probe = True
            _attach_axiom_generator_hook(generator, rec)
    if critic is not None:
        getattr(opt_d, "bind_module_layouts", lambda _: None)((critic,))
        _attach_axiom_critic_hook(critic, rec)
    return rec


__all__ = sorted(
    k
    for k, v in list(globals().items())
    if (not k.startswith("_") and getattr(v, "__module__", None) == __name__)
    or k in ("SCHEMA_VERSION", "_attach_axiom_critic_hook", "_attach_axiom_generator_hook")
)
