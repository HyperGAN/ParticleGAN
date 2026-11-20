"""Half B: Function-Space Two-Player Dynamics — state, hooks, and pure-elementwise MP optimizers."""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from torch import nn
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam, K3PGeneratorAdam, _closure_loss
from ..optim.constraint_geometry import project_nonascent

FROZEN_THERMAL_SCALE = 1e-4
_BOUNDS = {"ema_residual": (0.0, None), "ema_data_scale": (1e-6, None), "ema_margin": (None, None), "calibration_gate": (0.0, 1.0),
           "d_trust_scale": (0.0, 2.0), "g_trust_scale": (0.0, 2.0), "thermal_scale": (0.0, 1.0), "step_scale": (0.0, 1.0), "chord_phase": (0.0, 1.0), "cond_ema": (-1.0, 2.0)}
_BOOL_KEYS = ("initialized", "margin_initialized", "cond_initialized")
_STATE_KEYS = frozenset({"schema", *_BOOL_KEYS, "cond_bank", *_BOUNDS})

def _smoothstep01(u: float) -> float: x = max(0.0, min(1.0, float(u))); return float(x * x * (3.0 - 2.0 * x))
def _is_finite_num(v: Any, lo: Optional[float] = None, hi: Optional[float] = None) -> bool: return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and (lo is None or v >= lo) and (hi is None or v <= hi)
def _tensor_key(t: Any) -> Optional[Tuple[int, int, Tuple[int, ...], str]]:
    if not isinstance(t, torch.Tensor) or t.ndim < 1 or t.numel() == 0: return None
    try: ptr = int(t.untyped_storage().data_ptr())
    except RuntimeError: return None
    return None if ptr == 0 else (ptr, int(t.storage_offset()), tuple(int(s) for s in t.shape), str(t.device))
def _cond_signature(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]: return tuple([_tensor_key(a) if isinstance(a, torch.Tensor) else repr(a) for a in args[1:]] + [(k, _tensor_key(kwargs[k]) if isinstance(kwargs[k], torch.Tensor) else repr(kwargs[k])) for k in sorted(kwargs)])
def compute_two_player_coupling(ema_margin: float, ema_residual: float, thermal_scale: float = 1.0) -> Tuple[float, float, float]:
    res, margin, therm = max(0.0, float(ema_residual)), float(ema_margin), max(0.0, min(1.0, float(thermal_scale)))
    far = max(_smoothstep01((-margin) / 0.15), _smoothstep01((res - 0.08) / 0.14) * _smoothstep01((therm - 0.05) / 0.30))
    c_d = float(max(0.005, (1.0 - far) + far * _smoothstep01((margin + 0.10) / 0.42)))
    return c_d, float(max(0.12, 1.0 / (1.0 + max(0.0, margin - (1.60 + 1.20 * min(res, 2.0))) / 1.50))), float(0.55 + 0.45 * c_d)

def _minimal_critic_hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> Any:
    rec = getattr(module, "_minimal_record", None)
    if rec is not None and rec._in_probe: return None
    is_cond = bool(kwargs or len(args) > 1)
    cond_t = tuple(a.detach() for a in args[1:] if isinstance(a, torch.Tensor)) + tuple(kwargs[k].detach() for k in sorted(kwargs) if isinstance(kwargs[k], torch.Tensor))
    x_in = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if x_in is not None and isinstance(target, torch.Tensor):
        target._minimal_input, target._minimal_record, target._minimal_conditioned, target._minimal_cond_tensors = x_in, rec, is_cond, cond_t
        if rec is not None and not rec._in_penalty and not x_in.requires_grad: rec._last_real_pair = (x_in.detach(), target.detach().reshape(-1))
        no_ig = not x_in.requires_grad and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in (*args[1:], *kwargs.values()))
        if rec is not None and not rec._in_penalty and module.training and torch.is_grad_enabled() and target.requires_grad and no_ig and (x_in.ndim >= 3 or x_in[0].numel() > 8):
            skey = getattr(module, "_t2_pending_source_key", None) or _tensor_key(x_in)
            if skey is not None:
                alive = [True]; rec._live_captures.append([int(rec.observed_steps), skey, _cond_signature(args, kwargs), x_in.detach(), target.float() if target.dtype == torch.bfloat16 else target, alive])
                target.register_hook(lambda _g, _a=alive: _a.clear()); [e.__setitem__(slice(3, 5), [None, None]) or e[5].clear() for e in rec._live_captures[:-6]]; del rec._live_captures[:-6]
    if rec is not None and not rec._in_penalty and torch.is_grad_enabled() and rec._thermal_scale < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
        scaled = output * rec._thermal_scale + output.detach() * (1.0 - rec._thermal_scale)
        if x_in is not None: scaled._minimal_input, scaled._minimal_record, scaled._minimal_conditioned, scaled._minimal_cond_tensors = x_in, rec, is_cond, cond_t
        return scaled
    return None

def _minimal_generator_hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
    rec = getattr(module, "_minimal_record", None)
    if rec is None or rec._in_probe or not rec._enable_secant_probe or not torch.is_grad_enabled() or rec._g_probe_cache is not None: return
    t = output[0] if isinstance(output, (tuple, list)) and output else output
    if isinstance(t, torch.Tensor) and t.requires_grad and t.is_floating_point() and t.ndim == 2 and t.shape[1] <= 32 and args and isinstance(args[0], torch.Tensor) and args[0].ndim >= 2 and args[0].shape[0] == t.shape[0] and t.shape[0] >= 2:
        k = min(16, int(t.shape[0])); sl = lambda v: (v[:k].detach() if v.ndim >= 1 and v.shape[0] > k else v.detach()) if isinstance(v, torch.Tensor) else v
        rec._g_probe_cache = (module, tuple(sl(a) for a in args), {kk: sl(vv) for kk, vv in kwargs.items()}, t[:k].detach().float())

def _attach_minimal_critic_hook(critic: nn.Module, record: Any) -> None:
    if isinstance(critic, nn.Module):
        critic._minimal_record = critic._t2_record = record
        if not getattr(critic, "_minimal_hook_attached", False): critic.register_forward_hook(_minimal_critic_hook, with_kwargs=True); critic._minimal_hook_attached = True
def _attach_minimal_generator_hook(generator: nn.Module, record: Any) -> None:
    if isinstance(generator, nn.Module):
        generator._minimal_record = record
        if not getattr(generator, "_minimal_gen_hook_attached", False): generator.register_forward_hook(_minimal_generator_hook, with_kwargs=True); generator._minimal_gen_hook_attached = True
def bind_minimal_record(record: Any, *readers: Any) -> None:
    for r in readers:
        if r is not None and hasattr(r, "minimal_record"): r.minimal_record = record

def _validate_cond_bank(bank: Any) -> Dict[str, Any]:
    if not isinstance(bank, dict) or set(bank) != {"d_c", "d_x", "strata"} or not isinstance(bank["strata"], list) or len(bank["strata"]) > 64:
        raise ValueError("invalid minimal cond_bank packet")
    dc, dx = bank["d_c"], bank["d_x"]
    if dc is None and dx is None:
        if bank["strata"]: raise ValueError("uninitialized minimal cond_bank cannot hold strata")
        return {"d_c": None, "d_x": None, "strata": []}
    if type(dc) is not int or type(dx) is not int or not (1 <= dc <= 32) or dx < 1: raise ValueError("invalid minimal cond_bank dimensions")
    seen = set()
    for e in bank["strata"]:
        if (not isinstance(e, dict) or set(e) != {"key", "ring_c", "ring_x", "ring_y"} or not isinstance(e["key"], tuple) or not e["key"] or not all(type(v) is int for v in e["key"]) or e["key"] in seen
                or any(not isinstance(e[n], torch.Tensor) or e[n].dtype != torch.float32 or e[n].ndim != 2 or e[n].shape[1] != c or e[n].shape[0] > 384 or not bool(torch.isfinite(e[n]).all()) for n, c in (("ring_c", dc), ("ring_x", dx), ("ring_y", dx)))
                or not (e["ring_c"].shape[0] == e["ring_x"].shape[0] == e["ring_y"].shape[0])):
            raise ValueError("invalid minimal cond_bank stratum")
        seen.add(e["key"])
    return deepcopy(bank)

class MinimalStepRecord(CriticStepRecord):
    def __init__(self) -> None:
        super().__init__()
        self.initialized = self.margin_initialized = self.cond_initialized = self._in_penalty = self._in_probe = self._in_protected_probe = False
        self.ema_residual = self.ema_data_scale = self.calibration_gate = self.d_trust_scale = self.g_trust_scale = self._thermal_scale = self._step_scale = 1.0
        self.ema_margin, self.chord_phase, self.cond_ema, self._enable_secant_probe = 0.0, 0.0, 0.25, True
        self.cond_bank: Dict[str, Any] = {"d_c": None, "d_x": None, "strata": []}
        self._live_captures: List[Any] = []; self._g_probe_cache: Any = None; self._last_real_pair: Any = None

    def observe_margin(self, margin: float, ema_decay: float = 0.90) -> None:
        if math.isfinite(margin):
            m = max(-60.0, min(60.0, float(margin)))
            self.ema_margin = m if not self.margin_initialized else (ema_decay * self.ema_margin + (1.0 - ema_decay) * m); self.margin_initialized = True
            self.calibration_gate, self.d_trust_scale, self.g_trust_scale = compute_two_player_coupling(self.ema_margin, self.ema_residual, self._thermal_scale)

    def observe_transport(self, residual: float, data_scale: float, *, is_conditioned: bool = False) -> None:
        r, ds = max(0.0, float(residual)), max(1e-4, float(data_scale))
        self.ema_residual = r if (not self.initialized or r > max(3.5 * self.ema_residual, self.ema_residual + 0.12)) else (0.92 * self.ema_residual + 0.08 * r)
        self.ema_data_scale, self.initialized = (ds if not self.initialized else (0.95 * self.ema_data_scale + 0.05 * ds)), True
        if is_conditioned or self.cond_bank["strata"]: self._thermal_scale = 1.0
        else:
            u_th = max(0.0, (self.ema_residual - 0.015) / 0.020)
            self._thermal_scale, self._step_scale = float(max(1e-5, min(1.0, 1.0 - math.exp(-u_th * u_th)))), float(0.15 + 0.85 * _smoothstep01((self.ema_residual - 0.026) / 0.024))
        self.calibration_gate, self.d_trust_scale, self.g_trust_scale = compute_two_player_coupling(self.ema_margin, self.ema_residual, self._thermal_scale)

    def observe_cond_fiber(self, r_cond: float) -> None:
        rc = max(-0.5, min(1.5, float(r_cond)))
        self.cond_ema = max(0.18, rc) if not self.cond_initialized else (0.99 * self.cond_ema + 0.01 * rc)
        self.cond_initialized, self._thermal_scale, self._step_scale = True, 1.0, float(1.0 - 0.85 * _smoothstep01((0.050 - max(0.0, self.cond_ema)) / 0.024))

    def clear_transient_captures(self) -> None: [e.__setitem__(slice(3, 5), [None, None]) or e[5].clear() for e in self._live_captures]; self._live_captures.clear()
    def record_step(self, optimizer: Any) -> None: super().record_step(optimizer); self.clear_transient_captures(); self._g_probe_cache = self._last_real_pair = None

    def state_dict(self) -> Dict[str, Any]: return {**super().state_dict(), "minimal_state": {"schema": 1, **{k: bool(getattr(self, k)) for k in _BOOL_KEYS}, **{k: float(getattr(self, "_" + k if k in ("thermal_scale", "step_scale") else k)) for k in _BOUNDS}, "cond_bank": deepcopy(self.cond_bank)}}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        sc = dict(state); pkt = sc.pop("minimal_state", None)
        if not isinstance(pkt, dict) or set(pkt) != _STATE_KEYS or pkt["schema"] != 1 or any(type(pkt[k]) is not bool for k in _BOOL_KEYS): raise ValueError("invalid or missing minimal_state packet")
        vals = {k: float(pkt[k]) for k, (lo, hi) in _BOUNDS.items() if _is_finite_num(pkt[k], lo, hi)}
        if len(vals) != len(_BOUNDS): raise ValueError("invalid minimal_state scalar")
        cb = _validate_cond_bank(pkt["cond_bank"]); super().load_state_dict(sc)
        for k in _BOOL_KEYS: setattr(self, k, pkt[k])
        for k, v in vals.items(): setattr(self, "_" + k if k in ("thermal_scale", "step_scale") else k, v)
        self.cond_bank, self._g_probe_cache = cb, None; self.clear_transient_captures()

def _collect_conv_layouts(modules: Iterable[Optional[nn.Module]]) -> Dict[int, Tuple[bool, int]]: return {id(c.weight): (isinstance(c, nn.ConvTranspose2d), int(c.groups)) for r in modules if isinstance(r, nn.Module) for c in r.modules() if isinstance(c, (nn.Conv2d, nn.ConvTranspose2d))}

def _mp_scale(param: torch.Tensor, *, conv_t: bool = False, conv_g: int = 1, deep_g: float = 0.0, c_zero: float = 1.0) -> float:
    if param.ndim == 2:
        do, di = int(param.shape[0]), int(param.shape[1])
        sc = (math.sqrt(max(1.0, do / max(1.0, float(di)))) * (0.65 if do == 1 and di >= 16 else 1.0)) / (0.55 * math.sqrt(max(1.0, float(do))) + math.sqrt(max(1.0, float(di))))
        return float(sc * ((1.0 - deep_g) + deep_g * max(c_zero, min(2.5, float((param.float().square().mean().sqrt() * math.sqrt(max(1.0, float(di)))).item())))) if deep_g > 0.0 else sc)
    r_d, c_d, kp, g = int(param.shape[0]), int(param.shape[1]), math.prod(int(s) for s in param.shape[2:]), max(1, int(conv_g))
    do, di, ke = (float(c_d * g), float(r_d), float(max(1, kp // 4))) if conv_t else (float(r_d), float(c_d * g), float(max(1, kp)))
    return float(math.sqrt(max(0.5, do / max(1.0, di))) / ((math.sqrt(max(1.0, float(r_d))) + math.sqrt(max(1.0, float(c_d)))) * math.sqrt(ke)))

def _apply_step(opt: Any, lr_m: float, deep_g: float, c_zero: float, track: bool, *,
                prior_ids: set[int] = set(), direct_ids: set[int] = set(), s_step: float = 1.0, s_th: float = 1.0, th_g: float = 1.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    deltas, srows = [], getattr(opt, "_sampled_rows", {})
    for grp in opt.param_groups:
        lr, (b1_g, b2), eps = float(grp["lr"]), tuple(grp["betas"]), float(grp["eps"])
        pr_grp = grp.get("role", grp.get("forge_role", "generator")) in ("prior", "table")
        fast_p, fast_g, fast_m, fast_v, fast_eff, slow_q = [], [], [], [], [], []
        for p in grp["params"]:
            if p.grad is None: continue
            st = opt.state[p]
            if not st: st.update(step=torch.tensor(0.0), exp_avg=torch.zeros_like(p, dtype=torch.float32), exp_avg_sq=torch.zeros_like(p, dtype=torch.float32), g_scale=torch.ones((), device=p.device), peak_rms=torch.zeros((), device=p.device), _fast=True)
            is_dir, is_pr = (id(p) in direct_ids), ((pr_grp and id(p) not in direct_ids) or (id(p) in prior_ids) or (p in srows))
            if st.get("_fast", False) and not is_pr and not is_dir and not track and deep_g == 0.0 and p.ndim >= 2 and p.dtype == torch.float32 and lr > 0.0:
                ct, cg = opt._conv_layouts.get(id(p), (False, 1))
                fast_p.append(p); fast_g.append(p.grad.detach()); fast_m.append(st["exp_avg"]); fast_v.append(st["exp_avg_sq"]); fast_eff.append(lr * lr_m * _mp_scale(p, conv_t=ct, conv_g=cg))
            else: slow_q.append((p, is_dir, is_pr))
        if fast_p:
            g_infs = torch.stack(torch._foreach_norm(fast_g, ord=float("inf")))
            if bool((torch.isfinite(g_infs) & (g_infs >= 1e-18) & (g_infs <= 1e18)).all()):
                for p in fast_p: opt.state[p]["step"].add_(1.0)
                si = int(opt.state[fast_p[0]]["step"].item()); bc1, bc2_rt = max(1.0 - b1_g ** si, 1e-8), max(math.sqrt(1.0 - b2 ** si), 1e-8)
                torch._foreach_mul_(fast_m, b1_g); torch._foreach_add_(fast_m, fast_g, alpha=1.0 - b1_g)
                torch._foreach_mul_(fast_v, b2); torch._foreach_addcmul_(fast_v, fast_g, fast_g, value=1.0 - b2)
                for p, m_t, v_t, eff in zip(fast_p, fast_m, fast_v, fast_eff):
                    pk = opt.state[p]["peak_rms"]; pk.copy_(torch.maximum(pk * 0.9995, (v_t.mean() + 1e-36).sqrt() / bc2_rt))
                    dm = v_t.sqrt().div_(bc2_rt).add_(pk * 0.015 + eps).clamp_min_(1e-30); p.addcdiv_(m_t, dm, value=-(eff / bc1))
            else: slow_q.extend((p, False, False) for p in fast_p)
        for p, is_dir, is_pr in slow_q:
            st = opt.state[p]
            if st.get("_fast", False): st["_fast"] = False; st["g_scale"].fill_(1.0); st["exp_avg_sq"].sqrt_()
            dg = 0.0 if (is_pr or is_dir) else deep_g; b1 = max(b1_g, 0.75 * dg); st["step"].add_(1.0); si, gf = int(st["step"].item()), p.grad.detach().float()
            g_max = gf.abs().amax(); fin = torch.isfinite(g_max); gfs = torch.where(fin, gf, torch.zeros_like(gf))
            sc = torch.maximum(st["g_scale"], torch.where(fin, g_max, g_max.new_zeros(()))).clamp_min(1e-38); r_sc = st["g_scale"] / sc
            m_n = torch.where(fin, (st["exp_avg"] * r_sc).mul(b1).add(gfs / sc, alpha=1.0 - b1), st["exp_avg"] * r_sc)
            v_n = torch.where(fin, (b2 * (st["exp_avg_sq"] * r_sc).square() + (1.0 - b2) * (gfs / sc).square()).sqrt(), st["exp_avg_sq"] * r_sc)
            ns = torch.maximum(m_n.abs().amax(), v_n.abs().amax()); ok = ns > 1e-38; inv_ns = torch.where(ok, ns, torch.ones_like(ns))
            st["exp_avg"].copy_(torch.where(ok, m_n / inv_ns, torch.zeros_like(m_n))); st["exp_avg_sq"].copy_(torch.where(ok, v_n / inv_ns, torch.zeros_like(v_n))); st["g_scale"].copy_(torch.where(ok, ns, torch.zeros_like(ns)) * sc)
            mh, vh = m_n / max(1.0 - b1 ** si, 1e-8), v_n / max(math.sqrt(1.0 - b2 ** si), 1e-8)
            peak = torch.maximum(st["peak_rms"] * 0.9995, (vh.square().mean() + 1e-36).sqrt() * sc); st["peak_rms"].copy_(peak)
            u = torch.where(fin & (g_max > 1e-38), mh / (vh + (0.015 * peak + eps) / sc).clamp_min(1e-30), torch.zeros_like(gf))
            coh = float(((1.0 - dg) + dg * (mh.norm() / vh.norm().clamp_min(1e-30)).clamp(0.0, 1.0)).item()) if dg > 0.0 else 1.0
            if lr <= 0.0: continue
            if is_pr and p.ndim == 2:
                r = srows.get(p) if p in srows else torch.arange(p.shape[0], device=p.device)
                if r.numel() > 0:
                    sel = u[r]; p.index_add_(0, r, (sel / torch.hypot((sel.square().sum(1, keepdim=True) + 1e-12).sqrt(), sel.new_full((sel.shape[0], 1), 0.04))).to(p.dtype), alpha=-min(lr, 0.065) * s_step * (1e-4 if s_th <= 1e-4 else max(0.25, math.sqrt(s_th))))
            elif is_dir and p.ndim >= 2:
                rr = (u.flatten(1).square().mean(1) + 1e-12).sqrt()
                p.add_((u / torch.hypot(rr, torch.full_like(rr, 0.025)).reshape((p.shape[0],) + (1,) * (p.ndim - 1))).to(p.dtype), alpha=-max(0.024, min(lr, 0.035)) * s_step * th_g)
            else:
                ct, cg = opt._conv_layouts.get(id(p), (False, 1))
                dn = (u * _mp_scale(p, conv_t=ct, conv_g=cg, deep_g=deep_g, c_zero=c_zero)).to(p.dtype) if p.ndim >= 2 else (u * (math.sqrt(min(2.0, float(max(1, p.numel())))) / torch.hypot((u.square().sum() + 1e-12).sqrt(), u.new_full((), 0.05)))).to(p.dtype)
                eff = lr * lr_m * coh
                if track: dp = dn * (-eff); p.add_(dp); deltas.append((p, dp))
                else: p.add_(dn, alpha=-eff)
    return deltas

def _validate_opt_states(opt: torch.optim.Optimizer) -> None:
    for grp in opt.param_groups:
        for p in grp["params"]:
            st = opt.state.get(p)
            if st and (any(k not in st or not isinstance(st[k], torch.Tensor) or not bool(torch.isfinite(st[k]).all()) for k in ("exp_avg", "exp_avg_sq", "g_scale", "peak_rms")) or st["peak_rms"].ndim != 0 or float(st["peak_rms"].item()) < 0.0):
                raise ValueError("invalid minimal optimizer state")

class MinimalCriticOptimizer(K3PCriticAdam):
    _OWN_ATTRS = (*K3PCriticAdam._OWN_ATTRS, "_conv_layouts", "last_step_gain")

    def __init__(self, params, *, critic: nn.Module, ema_critic: Any = None, guard_ratio: float = 0.0, guard_min_steps: int = 200, **kw) -> None:
        del ema_critic
        super().__init__(params, critic=critic, ema_critic=None, anchor_decay=0.0, guard_ratio=guard_ratio, guard_min_steps=guard_min_steps, **kw)
        self.record, self._conv_layouts, self.last_step_gain = MinimalStepRecord(), _collect_conv_layouts((critic,)), 1.0
        _attach_minimal_critic_hook(critic, self.record)

    @torch.no_grad()
    def step(self, closure=None):
        loss = _closure_loss(closure)
        if self.guard is not None: self.guard.apply_(self)
        rec = self.record; th_d = max(0.10, math.sqrt(rec._thermal_scale)) if rec._thermal_scale < 1.0 else 1.0
        n_m = max(1, sum(1 for g in self.param_groups for p in g["params"] if p.ndim == 2)); deep_g = _smoothstep01((n_m - 5.0) / 4.0)
        _apply_step(self, rec._step_scale * rec.d_trust_scale * th_d * ((1.0 - deep_g) + deep_g * (3.2 / max(3, n_m))), deep_g, 1.0 / math.sqrt(n_m), False)
        self.last_step_gain = 1.0; rec.record_step(self); return loss

    def state_dict(self) -> Dict[str, Any]: return {**super().state_dict(), "minimal_opt": {"schema": 1, "last_step_gain": float(self.last_step_gain)}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        sc = dict(state_dict); meta = sc.pop("minimal_opt", None)
        if not isinstance(meta, dict) or set(meta) != {"schema", "last_step_gain"} or meta["schema"] != 1 or not _is_finite_num(meta["last_step_gain"], 1e-12): raise ValueError("invalid MinimalCriticOptimizer state")
        super().load_state_dict(sc); _validate_opt_states(self); self.last_step_gain = float(meta["last_step_gain"])

class MinimalGeneratorOptimizer(K3PGeneratorAdam):
    _OWN_ATTRS = (*K3PGeneratorAdam._OWN_ATTRS, "_minimal_record", "_sampled_rows", "_direct_ids", "_prior_param_ids", "_conv_layouts", "_wakeup_boost", "_g_mod_scale", "last_step_gain", "_protected", "constraint_geometry_stats")

    def __init__(self, params, *, latent_table: Optional[torch.Tensor] = None, direct_particles: Optional[Iterable[torch.Tensor]] = None, module_owners: Iterable[Optional[nn.Module]] = (), generator_module: Optional[nn.Module] = None, **kw) -> None:
        super().__init__(params, latent_table=latent_table, direct_particles=direct_particles, **kw)
        self._minimal_record, self._sampled_rows, self._protected = None, {}, None
        self._direct_ids, self._prior_param_ids = (set() if direct_particles is None else {id(p) for p in direct_particles}), ({id(latent_table)} if latent_table is not None else set())
        self._conv_layouts, self._wakeup_boost, self._g_mod_scale, self.last_step_gain = _collect_conv_layouts((generator_module, *module_owners)), 1.0, 1.0, 1.0
        self.constraint_geometry_stats: Dict[str, Any] = {"steps": 0, "projected_steps": 0, "max_derivative_before": 0.0, "max_derivative_after": 0.0}

    def bind_modulation(self, *, g_scale: float = 1.0, wakeup_boost: float = 1.0) -> None: self._g_mod_scale, self._wakeup_boost = float(g_scale), float(wakeup_boost)
    def set_sampled_rows(self, param: torch.Tensor, rows: torch.Tensor) -> None: self._sampled_rows[param] = torch.unique(rows.reshape(-1).to(device=param.device, dtype=torch.long), sorted=True)
    def clear_sampled_rows(self) -> None: self._sampled_rows.clear()

    def _bind_protected_losses(self, losses, *, protected_evaluator=None) -> None:
        del protected_evaluator; losses = tuple(losses)
        if self._protected is not None or not (1 <= len(losses) <= 2) or any(lo.ndim != 0 or not lo.requires_grad for lo in losses): raise ValueError("invalid minimal protected losses")
        params = [p for g in self.param_groups for p in g["params"]]; saved, normals = [None if p.grad is None else p.grad.detach().clone() for p in params], []
        try:
            for lo in losses:
                for p in params: p.grad = None
                lo.backward(retain_graph=True); pieces = []
                for p in params:
                    pc = torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                    if p in self._sampled_rows: m = torch.zeros_like(pc); m[self._sampled_rows[p]] = pc[self._sampled_rows[p]]; pc = m
                    pieces.append(pc.flatten())
                normals.append(torch.cat(pieces))
        finally:
            for p, prev in zip(params, saved): p.grad = prev
        st = torch.stack(normals)
        if not bool(torch.isfinite(st).all()): raise ValueError("nonfinite minimal protected gradient")
        self._protected = st

    @torch.no_grad()
    def step(self, closure=None):
        loss, rec = _closure_loss(closure), self._minimal_record
        s_step, g_g, s_th = (rec._step_scale, rec.g_trust_scale, rec._thermal_scale) if rec is not None else (1.0, 1.0, 1.0)
        th_g = 1e-4 if s_th <= 1e-4 else (max(0.08, math.sqrt(s_th)) if s_th < 1.0 else 1.0)
        n_m = max([1] + [sum(1 for p in g["params"] if p.ndim == 2 and id(p) not in self._prior_param_ids and id(p) not in self._direct_ids) for g in self.param_groups])
        deep_g, is_cond = _smoothstep01((n_m - 5.0) / 4.0), bool(rec is not None and rec.cond_bank["strata"])
        mod_scale = self._g_mod_scale if self._g_mod_scale < 0.999 else th_g
        lr_m = s_step * g_g * mod_scale * ((1.0 - deep_g) + deep_g * min(1.0, (3.2 / max(3, n_m)) * self._wakeup_boost))
        all_p = [p for g in self.param_groups for p in g["params"]] if self._protected is not None else []
        orig, pcache = ([p.detach().clone() for p in all_p] if all_p else []), (rec._g_probe_cache if rec is not None else None)
        track = bool(pcache is not None and getattr(rec, "_enable_secant_probe", False) and not is_cond)
        try:
            deltas = _apply_step(self, lr_m, deep_g, 1.0 / math.sqrt(n_m), track, prior_ids=self._prior_param_ids, direct_ids=self._direct_ids, s_step=s_step, s_th=s_th, th_g=th_g)
            alpha_g = 1.0
            if track and deltas and pcache is not None and rec is not None:
                gm, pa, pkw, y_pre = pcache; rec._g_probe_cache, modes, prev = None, [(m, m.training) for m in gm.modules()], rec._in_probe; rec._in_probe = True
                try:
                    gm.eval(); out = gm(*pa, **pkw); t = out[0] if isinstance(out, (tuple, list)) and out else out
                    y_post = t.detach().float() if isinstance(t, torch.Tensor) else None
                finally:
                    rec._in_probe = prev
                    for m, fl in modes: m.training = fl
                if y_post is not None and y_post.shape == y_pre.shape:
                    d_rel = float((y_post - y_pre).square().mean().sqrt().item()) / max(0.05, rec.ema_data_scale / math.sqrt(max(1, y_pre[0].numel())))
                    er = rec.ema_residual; d_max = (0.020 + 0.23 * min(1.5, er)) * g_g * s_step * mod_scale * max(1.0, self._wakeup_boost)
                    d_min = ((0.0045 + 0.042 * min(1.0, er)) * _smoothstep01((er - 0.024) / 0.036) * _smoothstep01((s_th - 0.10) / 0.40) * s_step) if (er > 0.024 and s_th > 0.10 and self._g_mod_scale >= 0.5) else 0.0
                    if math.isfinite(d_rel) and d_rel > 1e-12:
                        alpha_g = max(1e-4, d_max / d_rel) if (d_max > 0.0 and d_rel > d_max) else (min(96.0, max(1.0, d_min / (d_rel + d_min / 96.0))) if (d_min > 0.0 and d_rel < d_min) else 1.0)
                    if alpha_g != 1.0:
                        for p, dp in deltas: p.add_(dp, alpha=alpha_g - 1.0)
            elif rec is not None: rec._g_probe_cache = None
            if self._protected is not None and all_p:
                disp = torch.cat([(p - o).flatten() for p, o in zip(all_p, orig)]); proj, st = project_nonascent(disp, self._protected), self.constraint_geometry_stats
                st["steps"] += 1; st["max_derivative_before"] = max(st["max_derivative_before"], float((self._protected.double() @ disp.double()).max().item()))
                if not torch.equal(proj, disp):
                    st["projected_steps"] += 1; off = 0
                    for p, o in zip(all_p, orig): p.copy_(o + proj[off : off + p.numel()].view_as(p)); off += p.numel()
                st["max_derivative_after"] = max(st["max_derivative_after"], float((self._protected.double() @ torch.cat([(p - o).flatten() for p, o in zip(all_p, orig)]).double()).max().item()))
            self.last_step_gain = float(alpha_g)
        finally:
            self._protected, self._wakeup_boost, self._g_mod_scale = None, 1.0, 1.0; self.clear_sampled_rows()
        return loss

    def state_dict(self) -> Dict[str, Any]: return {**super().state_dict(), "minimal_opt": {"schema": 1, "last_step_gain": float(self.last_step_gain), "stats": dict(self.constraint_geometry_stats), "pending": None if self._protected is None else self._protected.detach().clone()}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        sc = dict(state_dict); meta = sc.pop("minimal_opt", None)
        if not isinstance(meta, dict) or set(meta) != {"schema", "last_step_gain", "stats", "pending"} or meta["schema"] != 1 or not _is_finite_num(meta["last_step_gain"], 1e-12): raise ValueError("invalid MinimalGeneratorOptimizer state")
        super().load_state_dict(sc); _validate_opt_states(self)
        self.last_step_gain, self.constraint_geometry_stats, self._protected = float(meta["last_step_gain"]), dict(meta["stats"]), (None if meta["pending"] is None else meta["pending"].detach().clone())

class MinimalConstraintGeneratorOptimizer(MinimalGeneratorOptimizer):
    bind_protected_losses = MinimalGeneratorOptimizer._bind_protected_losses
