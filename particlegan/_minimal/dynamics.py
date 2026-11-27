"""Half B: Function-Space Two-Player Dynamics — state, hooks, and pure-elementwise MP optimizers."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam, K3PGeneratorAdam, _closure_loss

FROZEN_THERMAL_SCALE = 1e-4

_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "ema_residual": (0.0, None),
    "ema_data_scale": (1e-6, None),
    "ema_margin": (None, None),
    "calibration_gate": (0.0, 1.0),
    "d_trust_scale": (0.0, 2.0),
    "g_trust_scale": (0.0, 2.0),
    "thermal_scale": (0.0, 1.0),
    "step_scale": (0.0, 1.0),
    "chord_phase": (0.0, 1.0),
    "cond_ema": (-1.0, 2.0),
}
_BOOL_KEYS = ("initialized", "margin_initialized", "cond_initialized")
_STATE_KEYS = frozenset({"schema", *_BOOL_KEYS, "cond_bank", *_BOUNDS})
_FP32_OPT_KEYS = ("exp_avg", "exp_avg_sq", "g_scale", "peak_rms")


def _smoothstep01(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return float(x * x * (3.0 - 2.0 * x))


def _is_finite_num(v: Any, lo: Optional[float] = None, hi: Optional[float] = None) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return False
    if lo is not None and v < lo:
        return False
    if hi is not None and v > hi:
        return False
    return True


def _tensor_key(t: Any) -> Optional[Tuple[int, int, Tuple[int, ...], str]]:
    if not isinstance(t, torch.Tensor) or t.ndim < 1 or t.numel() == 0:
        return None
    try:
        ptr = int(t.untyped_storage().data_ptr())
    except RuntimeError:
        return None
    if ptr == 0:
        return None
    return (ptr, int(t.storage_offset()), tuple(int(s) for s in t.shape), str(t.device))


def _cond_signature(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    pos_sig = [_tensor_key(a) if isinstance(a, torch.Tensor) else repr(a) for a in args[1:]]
    kw_sig = [
        (k, _tensor_key(kwargs[k]) if isinstance(kwargs[k], torch.Tensor) else repr(kwargs[k]))
        for k in sorted(kwargs)
    ]
    return tuple(pos_sig + kw_sig)


@torch.no_grad()
def _zero_pinv_project_nonascent(displacement: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Closed-form analytic Euclidean projection onto ``{x : normals @ x <= 0}`` for 1-2 constraints (0 pinv)."""
    d = displacement.double()
    a = normals.double()
    lengths = a.norm(dim=1)
    mask = lengths > 0
    a = a[mask] / lengths[mask, None]
    n_c = int(a.shape[0])
    if n_c > 2:
        raise ValueError("minimal constraint geometry supports at most two protected losses")
    if n_c == 0 or bool((a @ d <= 0.0).all()):
        return displacement.clone()

    best = torch.zeros_like(d)
    best_dist = (d * d).sum()
    tol = 1e-10 * max(1.0, float(d.norm().item()))

    if n_c == 1:
        a0 = a[0]
        c0 = float((a0 @ d).item())
        if c0 >= -tol:
            cand = d - c0 * a0
            if float((a0 @ cand).item()) <= tol:
                dist = (cand - d).square().sum()
                if float(dist.item()) < float(best_dist.item()):
                    best = cand
        return best.to(displacement)

    a0, a1 = a[0], a[1]
    c0 = float((a0 @ d).item())
    c1 = float((a1 @ d).item())
    rho = float((a0 @ a1).item())

    # Active set {0}:
    if c0 >= -tol:
        cand = d - c0 * a0
        if float((a0 @ cand).item()) <= tol and float((a1 @ cand).item()) <= tol:
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best = cand
                best_dist = dist

    # Active set {1}:
    if c1 >= -tol:
        cand = d - c1 * a1
        if float((a0 @ cand).item()) <= tol and float((a1 @ cand).item()) <= tol:
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best = cand
                best_dist = dist

    # Active set {0, 1}: exact 2x2 symmetric Gram [[1, rho], [rho, 1]] inverse / collinear pseudoinverse:
    det = 1.0 - rho * rho
    if det > 1e-14:
        inv_det = 1.0 / det
        m0 = (c0 - rho * c1) * inv_det
        m1 = (c1 - rho * c0) * inv_det
    else:
        sgn = 1.0 if rho >= 0.0 else -1.0
        m0 = 0.25 * (c0 + sgn * c1)
        m1 = 0.25 * (sgn * c0 + c1)
    if m0 >= -tol and m1 >= -tol:
        cand = d - m0 * a0 - m1 * a1
        if float((a0 @ cand).item()) <= tol and float((a1 @ cand).item()) <= tol:
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best = cand

    return best.to(displacement)


def compute_two_player_coupling(
    ema_margin: float,
    ema_residual: float,
    thermal_scale: float = 1.0,
) -> Tuple[float, float, float]:
    res = max(0.0, float(ema_residual))
    margin = float(ema_margin)
    therm = max(0.0, min(1.0, float(thermal_scale)))
    far = max(
        _smoothstep01((-margin) / 0.15),
        _smoothstep01((res - 0.08) / 0.14) * _smoothstep01((therm - 0.05) / 0.30),
    )
    c_d = float(max(0.005, (1.0 - far) + far * _smoothstep01((margin + 0.10) / 0.42)))
    d_trust = float(max(0.12, 1.0 / (1.0 + max(0.0, margin - (1.60 + 1.20 * min(res, 2.0))) / 1.50)))
    g_trust = float(0.55 + 0.45 * c_d)
    return c_d, d_trust, g_trust


def _minimal_critic_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    rec = getattr(module, "_minimal_record", None)
    if rec is not None and rec._in_probe:
        return None
    is_cond = bool(kwargs or len(args) > 1)
    cond_pos = tuple(a.detach() for a in args[1:] if isinstance(a, torch.Tensor))
    cond_kw = tuple(kwargs[k].detach() for k in sorted(kwargs) if isinstance(kwargs[k], torch.Tensor))
    cond_t = cond_pos + cond_kw
    x_in = args[0] if (args and isinstance(args[0], torch.Tensor)) else None
    target = output[0] if isinstance(output, (tuple, list)) and output else output

    if x_in is not None and isinstance(target, torch.Tensor):
        target._minimal_input = x_in
        target._minimal_record = rec
        target._minimal_conditioned = is_cond
        target._minimal_cond_tensors = cond_t
        if rec is not None and not rec._in_penalty and not x_in.requires_grad:
            rec._last_real_pair = (x_in.detach(), target.detach().reshape(-1))
        no_ig = not x_in.requires_grad and not any(
            isinstance(t, torch.Tensor) and t.requires_grad for t in (*args[1:], *kwargs.values())
        )
        if (
            rec is not None
            and not rec._in_penalty
            and module.training
            and torch.is_grad_enabled()
            and target.requires_grad
            and no_ig
            and (x_in.ndim >= 3 or x_in[0].numel() > 8)
        ):
            skey = getattr(module, "_t2_pending_source_key", None) or _tensor_key(x_in)
            if skey is not None:
                alive = [True]
                captured_target = target.float() if target.dtype == torch.bfloat16 else target
                rec._live_captures.append(
                    [
                        int(rec.observed_steps),
                        skey,
                        _cond_signature(args, kwargs),
                        x_in.detach(),
                        captured_target,
                        alive,
                    ]
                )
                target.register_hook(lambda _g, _a=alive: _a.clear())
                if len(rec._live_captures) > 6:
                    for entry in rec._live_captures[:-6]:
                        entry[3] = None
                        entry[4] = None
                        entry[5].clear()
                    del rec._live_captures[:-6]

    if (
        rec is not None
        and not rec._in_penalty
        and torch.is_grad_enabled()
        and rec._thermal_scale < 1.0
        and isinstance(output, torch.Tensor)
        and output.requires_grad
    ):
        scaled = output * rec._thermal_scale + output.detach() * (1.0 - rec._thermal_scale)
        if x_in is not None:
            scaled._minimal_input = x_in
            scaled._minimal_record = rec
            scaled._minimal_conditioned = is_cond
            scaled._minimal_cond_tensors = cond_t
        return scaled
    return None


def _slice_probe_value(val: Any, k: int) -> Any:
    if isinstance(val, torch.Tensor):
        if val.ndim >= 1 and val.shape[0] > k:
            return val[:k].detach()
        return val.detach()
    return val


def _minimal_generator_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    rec = getattr(module, "_minimal_record", None)
    if (
        rec is None
        or rec._in_probe
        or not rec._enable_secant_probe
        or not torch.is_grad_enabled()
        or rec._g_probe_cache is not None
    ):
        return
    t = output[0] if isinstance(output, (tuple, list)) and output else output
    if (
        isinstance(t, torch.Tensor)
        and t.requires_grad
        and t.is_floating_point()
        and t.ndim == 2
        and t.shape[1] <= 32
        and args
        and isinstance(args[0], torch.Tensor)
        and args[0].ndim >= 2
        and args[0].shape[0] == t.shape[0]
        and t.shape[0] >= 2
    ):
        k = min(16, int(t.shape[0]))
        sliced_args = tuple(_slice_probe_value(a, k) for a in args)
        sliced_kwargs = {kk: _slice_probe_value(vv, k) for kk, vv in kwargs.items()}
        rec._g_probe_cache = (module, sliced_args, sliced_kwargs, t[:k].detach().float())


def _attach_minimal_critic_hook(critic: nn.Module, record: Any) -> None:
    if isinstance(critic, nn.Module):
        critic._minimal_record = record
        critic._t2_record = record
        if not getattr(critic, "_minimal_hook_attached", False):
            critic.register_forward_hook(_minimal_critic_hook, with_kwargs=True)
            critic._minimal_hook_attached = True


def _attach_minimal_generator_hook(generator: nn.Module, record: Any) -> None:
    if isinstance(generator, nn.Module):
        generator._minimal_record = record
        if not getattr(generator, "_minimal_gen_hook_attached", False):
            generator.register_forward_hook(_minimal_generator_hook, with_kwargs=True)
            generator._minimal_gen_hook_attached = True


def bind_minimal_record(record: Any, *readers: Any) -> None:
    for r in readers:
        if r is not None and hasattr(r, "minimal_record"):
            r.minimal_record = record


def _validate_cond_bank(bank: Any) -> Dict[str, Any]:
    if (
        not isinstance(bank, dict)
        or set(bank) != {"d_c", "d_x", "strata"}
        or not isinstance(bank["strata"], list)
        or len(bank["strata"]) > 64
    ):
        raise ValueError("invalid minimal cond_bank packet")
    dc = bank["d_c"]
    dx = bank["d_x"]
    if dc is None and dx is None:
        if bank["strata"]:
            raise ValueError("uninitialized minimal cond_bank cannot hold strata")
        return {"d_c": None, "d_x": None, "strata": []}
    if type(dc) is not int or type(dx) is not int or not (1 <= dc <= 32) or dx < 1:
        raise ValueError("invalid minimal cond_bank dimensions")
    seen = set()
    for entry in bank["strata"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"key", "ring_c", "ring_x", "ring_y"}
            or not isinstance(entry["key"], tuple)
            or not entry["key"]
            or not all(type(v) is int for v in entry["key"])
            or entry["key"] in seen
        ):
            raise ValueError("invalid minimal cond_bank stratum")
        for name, cols in (("ring_c", dc), ("ring_x", dx), ("ring_y", dx)):
            ten = entry[name]
            if (
                not isinstance(ten, torch.Tensor)
                or ten.dtype != torch.float32
                or ten.ndim != 2
                or ten.shape[1] != cols
                or ten.shape[0] > 384
                or not bool(torch.isfinite(ten).all())
            ):
                raise ValueError("invalid minimal cond_bank stratum")
        if not (entry["ring_c"].shape[0] == entry["ring_x"].shape[0] == entry["ring_y"].shape[0]):
            raise ValueError("invalid minimal cond_bank stratum")
        seen.add(entry["key"])
    return deepcopy(bank)


class MinimalStepRecord(CriticStepRecord):
    """Checkpointed two-player coupling state shared across minimal critic, loss, and optimizers."""

    def __init__(self) -> None:
        super().__init__()
        self.initialized = False
        self.margin_initialized = False
        self.cond_initialized = False
        self._in_penalty = False
        self._in_probe = False
        self._in_protected_probe = False

        self.ema_residual = 1.0
        self.ema_data_scale = 1.0
        self.calibration_gate = 1.0
        self.d_trust_scale = 1.0
        self.g_trust_scale = 1.0
        self._thermal_scale = 1.0
        self._step_scale = 1.0

        self.ema_margin = 0.0
        self.chord_phase = 0.0
        self.cond_ema = 0.25
        self._enable_secant_probe = True

        self.cond_bank: Dict[str, Any] = {"d_c": None, "d_x": None, "strata": []}
        self._live_captures: List[Any] = []
        self._g_probe_cache: Any = None
        self._last_real_pair: Any = None

    def observe_margin(self, margin: float, ema_decay: float = 0.90) -> None:
        if math.isfinite(margin):
            m = max(-60.0, min(60.0, float(margin)))
            if not self.margin_initialized:
                self.ema_margin = m
            else:
                self.ema_margin = ema_decay * self.ema_margin + (1.0 - ema_decay) * m
            self.margin_initialized = True
            self.calibration_gate, self.d_trust_scale, self.g_trust_scale = compute_two_player_coupling(
                self.ema_margin, self.ema_residual, self._thermal_scale
            )

    def observe_transport(self, residual: float, data_scale: float, *, is_conditioned: bool = False) -> None:
        r = max(0.0, float(residual))
        ds = max(1e-4, float(data_scale))
        if not self.initialized or r > max(3.5 * self.ema_residual, self.ema_residual + 0.12):
            self.ema_residual = r
        else:
            self.ema_residual = 0.92 * self.ema_residual + 0.08 * r
        if not self.initialized:
            self.ema_data_scale = ds
        else:
            self.ema_data_scale = 0.95 * self.ema_data_scale + 0.05 * ds
        self.initialized = True
        if is_conditioned or self.cond_bank["strata"]:
            self._thermal_scale = 1.0
        else:
            u_th = max(0.0, (self.ema_residual - 0.015) / 0.020)
            self._thermal_scale = float(max(1e-5, min(1.0, 1.0 - math.exp(-u_th * u_th))))
            self._step_scale = float(0.15 + 0.85 * _smoothstep01((self.ema_residual - 0.026) / 0.024))
        self.calibration_gate, self.d_trust_scale, self.g_trust_scale = compute_two_player_coupling(
            self.ema_margin, self.ema_residual, self._thermal_scale
        )

    def observe_cond_fiber(self, r_cond: float) -> None:
        rc = max(-0.5, min(1.5, float(r_cond)))
        if not self.cond_initialized:
            self.cond_ema = max(0.18, rc)
        else:
            self.cond_ema = 0.99 * self.cond_ema + 0.01 * rc
        self.cond_initialized = True
        self._thermal_scale = 1.0
        self._step_scale = float(1.0 - 0.85 * _smoothstep01((0.050 - max(0.0, self.cond_ema)) / 0.024))

    def clear_transient_captures(self) -> None:
        for entry in self._live_captures:
            entry[3] = None
            entry[4] = None
            entry[5].clear()
        self._live_captures.clear()

    def record_step(self, optimizer: Any) -> None:
        super().record_step(optimizer)
        self.clear_transient_captures()
        self._g_probe_cache = None
        self._last_real_pair = None

    def state_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"schema": 1}
        for key in _BOOL_KEYS:
            payload[key] = bool(getattr(self, key))
        for key in _BOUNDS:
            attr = "_" + key if key in ("thermal_scale", "step_scale") else key
            payload[key] = float(getattr(self, attr))
        payload["cond_bank"] = deepcopy(self.cond_bank)
        return {**super().state_dict(), "minimal_state": payload}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        sc = dict(state)
        pkt = sc.pop("minimal_state", None)
        if (
            not isinstance(pkt, dict)
            or set(pkt) != _STATE_KEYS
            or pkt["schema"] != 1
            or any(type(pkt[k]) is not bool for k in _BOOL_KEYS)
        ):
            raise ValueError("invalid or missing minimal_state packet")
        vals: Dict[str, float] = {}
        for key, (lo, hi) in _BOUNDS.items():
            if not _is_finite_num(pkt[key], lo, hi):
                raise ValueError("invalid minimal_state scalar")
            vals[key] = float(pkt[key])
        cb = _validate_cond_bank(pkt["cond_bank"])
        super().load_state_dict(sc)
        for key in _BOOL_KEYS:
            setattr(self, key, pkt[key])
        for key, value in vals.items():
            attr = "_" + key if key in ("thermal_scale", "step_scale") else key
            setattr(self, attr, value)
        self.cond_bank = cb
        self._g_probe_cache = None
        self.clear_transient_captures()


def _collect_conv_layouts(modules: Iterable[Optional[nn.Module]]) -> Dict[int, Tuple[bool, int]]:
    layouts: Dict[int, Tuple[bool, int]] = {}
    for root in modules:
        if not isinstance(root, nn.Module):
            continue
        for mod in root.modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                layouts[id(mod.weight)] = (isinstance(mod, nn.ConvTranspose2d), int(mod.groups))
    return layouts


def _mp_scale(
    param: torch.Tensor,
    *,
    conv_t: bool = False,
    conv_g: int = 1,
    deep_g: float = 0.0,
    c_zero: float = 1.0,
) -> float:
    if param.ndim == 2:
        do = int(param.shape[0])
        di = int(param.shape[1])
        head_damp = 0.65 if (do == 1 and di >= 16) else 1.0
        sc = (math.sqrt(max(1.0, do / max(1.0, float(di)))) * head_damp) / (
            0.55 * math.sqrt(max(1.0, float(do))) + math.sqrt(max(1.0, float(di)))
        )
        if deep_g > 0.0:
            rms_in = float((param.float().square().mean().sqrt() * math.sqrt(max(1.0, float(di)))).item())
            return float(sc * ((1.0 - deep_g) + deep_g * max(c_zero, min(2.5, rms_in))))
        return float(sc)

    r_d = int(param.shape[0])
    c_d = int(param.shape[1])
    kp = math.prod(int(s) for s in param.shape[2:])
    g = max(1, int(conv_g))
    if conv_t:
        do_f = float(c_d * g)
        di_f = float(r_d)
        ke_f = float(max(1, kp // 4))
    else:
        do_f = float(r_d)
        di_f = float(c_d * g)
        ke_f = float(max(1, kp))
    denom = (math.sqrt(max(1.0, float(r_d))) + math.sqrt(max(1.0, float(c_d)))) * math.sqrt(ke_f)
    return float(math.sqrt(max(0.5, do_f / max(1.0, di_f))) / denom)


def _apply_step(
    opt: Any,
    lr_m: float,
    deep_g: float,
    c_zero: float,
    track: bool,
    *,
    prior_ids: Optional[set[int]] = None,
    direct_ids: Optional[set[int]] = None,
    s_step: float = 1.0,
    s_th: float = 1.0,
    th_g: float = 1.0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    prior_id_set = set() if prior_ids is None else prior_ids
    direct_id_set = set() if direct_ids is None else direct_ids
    deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
    srows = getattr(opt, "_sampled_rows", {})

    for grp in opt.param_groups:
        lr = float(grp["lr"])
        b1_g, b2 = tuple(grp["betas"])
        eps = float(grp["eps"])
        role = grp.get("role", grp.get("forge_role", "generator"))
        pr_grp = role in ("prior", "table")

        fast_p: List[torch.Tensor] = []
        fast_g: List[torch.Tensor] = []
        fast_m: List[torch.Tensor] = []
        fast_v: List[torch.Tensor] = []
        fast_eff: List[float] = []
        slow_q: List[Tuple[torch.Tensor, bool, bool]] = []

        for p in grp["params"]:
            if p.grad is None:
                continue
            st = opt.state[p]
            if not st:
                st.update(
                    step=torch.tensor(0.0, dtype=torch.float32),
                    exp_avg=torch.zeros_like(p, dtype=torch.float32),
                    exp_avg_sq=torch.zeros_like(p, dtype=torch.float32),
                    g_scale=torch.ones((), device=p.device, dtype=torch.float32),
                    peak_rms=torch.zeros((), device=p.device, dtype=torch.float32),
                    _fast=True,
                )
            is_dir = id(p) in direct_id_set
            is_pr = (pr_grp and id(p) not in direct_id_set) or (id(p) in prior_id_set) or (p in srows)
            if (
                st.get("_fast", False)
                and not is_pr
                and not is_dir
                and not track
                and deep_g == 0.0
                and p.ndim >= 2
                and p.dtype == torch.float32
                and lr > 0.0
            ):
                ct, cg = opt._conv_layouts.get(id(p), (False, 1))
                fast_p.append(p)
                fast_g.append(p.grad.detach())
                fast_m.append(st["exp_avg"])
                fast_v.append(st["exp_avg_sq"])
                fast_eff.append(lr * lr_m * _mp_scale(p, conv_t=ct, conv_g=cg))
            else:
                slow_q.append((p, is_dir, is_pr))

        if fast_p:
            g_infs = torch.stack(torch._foreach_norm(fast_g, ord=float("inf")))
            if bool((torch.isfinite(g_infs) & (g_infs >= 1e-18) & (g_infs <= 1e18)).all()):
                for p in fast_p:
                    opt.state[p]["step"].add_(1.0)
                torch._foreach_mul_(fast_m, b1_g)
                torch._foreach_add_(fast_m, fast_g, alpha=1.0 - b1_g)
                torch._foreach_mul_(fast_v, b2)
                torch._foreach_addcmul_(fast_v, fast_g, fast_g, value=1.0 - b2)
                for p, m_t, v_t, eff in zip(fast_p, fast_m, fast_v, fast_eff):
                    # Per-parameter Adam bias correction (never share fast_p[0]'s step across params):
                    si = int(opt.state[p]["step"].item())
                    bc1 = max(1.0 - (b1_g ** si), 1e-8)
                    bc2_rt = max(math.sqrt(1.0 - (b2 ** si)), 1e-8)
                    pk = opt.state[p]["peak_rms"]
                    pk.copy_(torch.maximum(pk * 0.9995, (v_t.mean() + 1e-36).sqrt() / bc2_rt))
                    dm = v_t.sqrt().div_(bc2_rt).add_(pk * 0.015 + eps).clamp_min_(1e-30)
                    p.addcdiv_(m_t, dm, value=-(eff / bc1))
            else:
                slow_q.extend((p, False, False) for p in fast_p)

        for p, is_dir, is_pr in slow_q:
            st = opt.state[p]
            if st.get("_fast", False):
                st["_fast"] = False
                st["g_scale"].fill_(1.0)
                st["exp_avg_sq"].sqrt_()
            dg = 0.0 if (is_pr or is_dir) else deep_g
            b1 = max(b1_g, 0.75 * dg)
            st["step"].add_(1.0)
            si = int(st["step"].item())
            gf = p.grad.detach().float()
            g_max = gf.abs().amax()
            fin = torch.isfinite(g_max)
            gfs = torch.where(fin, gf, torch.zeros_like(gf))
            sc = torch.maximum(st["g_scale"], torch.where(fin, g_max, g_max.new_zeros(()))).clamp_min(1e-38)
            r_sc = st["g_scale"] / sc
            m_n = torch.where(fin, (st["exp_avg"] * r_sc).mul(b1).add(gfs / sc, alpha=1.0 - b1), st["exp_avg"] * r_sc)
            v_n = torch.where(
                fin,
                (b2 * (st["exp_avg_sq"] * r_sc).square() + (1.0 - b2) * (gfs / sc).square()).sqrt(),
                st["exp_avg_sq"] * r_sc,
            )
            ns = torch.maximum(m_n.abs().amax(), v_n.abs().amax())
            ok = ns > 1e-38
            inv_ns = torch.where(ok, ns, torch.ones_like(ns))
            st["exp_avg"].copy_(torch.where(ok, m_n / inv_ns, torch.zeros_like(m_n)))
            st["exp_avg_sq"].copy_(torch.where(ok, v_n / inv_ns, torch.zeros_like(v_n)))
            st["g_scale"].copy_(torch.where(ok, ns, torch.zeros_like(ns)) * sc)

            mh = m_n / max(1.0 - (b1 ** si), 1e-8)
            vh = v_n / max(math.sqrt(1.0 - (b2 ** si)), 1e-8)
            peak = torch.maximum(st["peak_rms"] * 0.9995, (vh.square().mean() + 1e-36).sqrt() * sc)
            st["peak_rms"].copy_(peak)
            u = torch.where(fin & (g_max > 1e-38), mh / (vh + (0.015 * peak + eps) / sc).clamp_min(1e-30), torch.zeros_like(gf))
            if dg > 0.0:
                coh = float(((1.0 - dg) + dg * (mh.norm() / vh.norm().clamp_min(1e-30)).clamp(0.0, 1.0)).item())
            else:
                coh = 1.0

            if lr <= 0.0:
                continue
            if is_pr and p.ndim == 2:
                rows = srows.get(p) if p in srows else torch.arange(p.shape[0], device=p.device)
                if rows.numel() > 0:
                    sel = u[rows]
                    denom_r = torch.hypot((sel.square().sum(1, keepdim=True) + 1e-12).sqrt(), sel.new_full((sel.shape[0], 1), 0.04))
                    pr_scale = 1e-4 if s_th <= 1e-4 else max(0.25, math.sqrt(s_th))
                    p.index_add_(0, rows, (sel / denom_r).to(p.dtype), alpha=-min(lr, 0.065) * s_step * pr_scale)
            elif is_dir and p.ndim >= 2:
                rr = (u.flatten(1).square().mean(1) + 1e-12).sqrt()
                denom_d = torch.hypot(rr, torch.full_like(rr, 0.025)).reshape((p.shape[0],) + (1,) * (p.ndim - 1))
                p.add_((u / denom_d).to(p.dtype), alpha=-max(0.024, min(lr, 0.035)) * s_step * th_g)
            else:
                ct, cg = opt._conv_layouts.get(id(p), (False, 1))
                if p.ndim >= 2:
                    dn = (u * _mp_scale(p, conv_t=ct, conv_g=cg, deep_g=deep_g, c_zero=c_zero)).to(p.dtype)
                else:
                    vec_denom = torch.hypot((u.square().sum() + 1e-12).sqrt(), u.new_full((), 0.05))
                    dn = (u * (math.sqrt(min(2.0, float(max(1, p.numel())))) / vec_denom)).to(p.dtype)
                eff = lr * lr_m * coh
                if track:
                    dp = dn * (-eff)
                    p.add_(dp)
                    deltas.append((p, dp))
                else:
                    p.add_(dn, alpha=-eff)
    return deltas


def _restore_fp32_opt_states(opt: torch.optim.Optimizer, saved_state_dict: Dict[str, Any]) -> None:
    """Restore internal FP32 moments/scales/step directly from saved state so BF16 params never truncate them."""
    saved_state_map = saved_state_dict.get("state", {})
    saved_groups = saved_state_dict.get("param_groups", [])
    for cur_grp, sav_grp in zip(opt.param_groups, saved_groups):
        for p, sav_pid in zip(cur_grp["params"], sav_grp.get("params", ())):
            sav_st = saved_state_map.get(sav_pid)
            if not sav_st:
                continue
            live_st = opt.state[p]
            if "step" in sav_st and isinstance(sav_st["step"], torch.Tensor):
                live_st["step"] = sav_st["step"].detach().to(dtype=torch.float32).clone()
            for key in _FP32_OPT_KEYS:
                if key in sav_st and isinstance(sav_st[key], torch.Tensor):
                    live_st[key] = sav_st[key].detach().to(device=p.device, dtype=torch.float32).clone()


def _validate_opt_states(opt: torch.optim.Optimizer) -> None:
    for grp in opt.param_groups:
        for p in grp["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            for key in _FP32_OPT_KEYS:
                if key not in st or not isinstance(st[key], torch.Tensor) or not bool(torch.isfinite(st[key]).all()):
                    raise ValueError("invalid minimal optimizer state")
            if st["peak_rms"].ndim != 0 or float(st["peak_rms"].item()) < 0.0:
                raise ValueError("invalid minimal optimizer state")


class MinimalCriticOptimizer(K3PCriticAdam):
    """Pure-elementwise MP critic optimizer for ``preset='minimal'``."""

    _OWN_ATTRS = (*K3PCriticAdam._OWN_ATTRS, "_conv_layouts", "last_step_gain")

    def __init__(
        self,
        params: Any,
        *,
        critic: nn.Module,
        ema_critic: Any = None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **kw: Any,
    ) -> None:
        del ema_critic
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **kw,
        )
        self.record = MinimalStepRecord()
        self._conv_layouts = _collect_conv_layouts((critic,))
        self.last_step_gain = 1.0
        _attach_minimal_critic_hook(critic, self.record)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = _closure_loss(closure)
        if self.guard is not None:
            self.guard.apply_(self)
        rec = self.record
        th_d = max(0.10, math.sqrt(rec._thermal_scale)) if rec._thermal_scale < 1.0 else 1.0
        n_m = max(1, sum(1 for g in self.param_groups for p in g["params"] if p.ndim == 2))
        deep_g = _smoothstep01((n_m - 5.0) / 4.0)
        lr_mult = rec._step_scale * rec.d_trust_scale * th_d * ((1.0 - deep_g) + deep_g * (3.2 / max(3, n_m)))
        _apply_step(self, lr_mult, deep_g, 1.0 / math.sqrt(n_m), False)
        self.last_step_gain = 1.0
        rec.record_step(self)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "minimal_opt": {"schema": 1, "last_step_gain": float(self.last_step_gain)},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        sc = dict(state_dict)
        meta = sc.pop("minimal_opt", None)
        if (
            not isinstance(meta, dict)
            or set(meta) != {"schema", "last_step_gain"}
            or meta["schema"] != 1
            or not _is_finite_num(meta["last_step_gain"], 1e-12)
        ):
            raise ValueError("invalid MinimalCriticOptimizer state")
        super().load_state_dict(sc)
        _restore_fp32_opt_states(self, state_dict)
        _validate_opt_states(self)
        self.last_step_gain = float(meta["last_step_gain"])


class MinimalGeneratorOptimizer(K3PGeneratorAdam):
    """Pure-elementwise MP generator/prior optimizer with function-space trust band and zero-pinv projection."""

    _OWN_ATTRS = (
        *K3PGeneratorAdam._OWN_ATTRS,
        "_minimal_record",
        "_sampled_rows",
        "_direct_ids",
        "_prior_param_ids",
        "_conv_layouts",
        "_wakeup_boost",
        "_g_mod_scale",
        "last_step_gain",
        "_protected",
        "constraint_geometry_stats",
    )

    def __init__(
        self,
        params: Any,
        *,
        latent_table: Optional[torch.Tensor] = None,
        direct_particles: Optional[Iterable[torch.Tensor]] = None,
        module_owners: Iterable[Optional[nn.Module]] = (),
        generator_module: Optional[nn.Module] = None,
        **kw: Any,
    ) -> None:
        super().__init__(params, latent_table=latent_table, direct_particles=direct_particles, **kw)
        self._minimal_record = None
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._protected: Optional[torch.Tensor] = None
        self._direct_ids = set() if direct_particles is None else {id(p) for p in direct_particles}
        self._prior_param_ids = {id(latent_table)} if latent_table is not None else set()
        self._conv_layouts = _collect_conv_layouts((generator_module, *module_owners))
        self._wakeup_boost = 1.0
        self._g_mod_scale = 1.0
        self.last_step_gain = 1.0
        self.constraint_geometry_stats: Dict[str, Any] = {
            "steps": 0,
            "projected_steps": 0,
            "max_derivative_before": 0.0,
            "max_derivative_after": 0.0,
        }

    def bind_modulation(self, *, g_scale: float = 1.0, wakeup_boost: float = 1.0) -> None:
        self._g_mod_scale = float(g_scale)
        self._wakeup_boost = float(wakeup_boost)

    def set_sampled_rows(self, param: torch.Tensor, rows: torch.Tensor) -> None:
        self._sampled_rows[param] = torch.unique(
            rows.reshape(-1).to(device=param.device, dtype=torch.long),
            sorted=True,
        )

    def clear_sampled_rows(self) -> None:
        self._sampled_rows.clear()

    def _bind_protected_losses(self, losses: Iterable[torch.Tensor], *, protected_evaluator: Any = None) -> None:
        del protected_evaluator
        loss_tuple = tuple(losses)
        if (
            self._protected is not None
            or not (1 <= len(loss_tuple) <= 2)
            or any(lo.ndim != 0 or not lo.requires_grad for lo in loss_tuple)
        ):
            raise ValueError("invalid minimal protected losses")
        params = [p for g in self.param_groups for p in g["params"]]
        trainable_p = [p for p in params if p.requires_grad]
        saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
        normals: List[torch.Tensor] = []
        try:
            for lo in loss_tuple:
                for p in params:
                    p.grad = None
                if trainable_p:
                    # Scope backward strictly to owned trainable parameters so external params stay untouched:
                    lo.backward(retain_graph=True, inputs=trainable_p)
                pieces: List[torch.Tensor] = []
                for p in params:
                    pc = torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                    if p in self._sampled_rows:
                        masked = torch.zeros_like(pc)
                        masked[self._sampled_rows[p]] = pc[self._sampled_rows[p]]
                        pc = masked
                    pieces.append(pc.flatten())
                normals.append(torch.cat(pieces))
        finally:
            for p, prev in zip(params, saved):
                p.grad = prev
        stacked = torch.stack(normals)
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError("nonfinite minimal protected gradient")
        self._protected = stacked

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = _closure_loss(closure)
        rec = self._minimal_record
        if rec is not None:
            s_step = rec._step_scale
            g_g = rec.g_trust_scale
            s_th = rec._thermal_scale
        else:
            s_step = 1.0
            g_g = 1.0
            s_th = 1.0

        th_g = 1e-4 if s_th <= 1e-4 else (max(0.08, math.sqrt(s_th)) if s_th < 1.0 else 1.0)
        n_m = max(
            [1]
            + [
                sum(
                    1
                    for p in g["params"]
                    if p.ndim == 2 and id(p) not in self._prior_param_ids and id(p) not in self._direct_ids
                )
                for g in self.param_groups
            ]
        )
        deep_g = _smoothstep01((n_m - 5.0) / 4.0)
        is_cond = bool(rec is not None and rec.cond_bank["strata"])
        mod_scale = self._g_mod_scale if self._g_mod_scale < 0.999 else th_g
        lr_m = s_step * g_g * mod_scale * ((1.0 - deep_g) + deep_g * min(1.0, (3.2 / max(3, n_m)) * self._wakeup_boost))

        all_p = [p for g in self.param_groups for p in g["params"]] if self._protected is not None else []
        orig = [p.detach().clone() for p in all_p] if all_p else []
        pcache = rec._g_probe_cache if rec is not None else None
        track = bool(pcache is not None and getattr(rec, "_enable_secant_probe", False) and not is_cond)

        try:
            deltas = _apply_step(
                self,
                lr_m,
                deep_g,
                1.0 / math.sqrt(n_m),
                track,
                prior_ids=self._prior_param_ids,
                direct_ids=self._direct_ids,
                s_step=s_step,
                s_th=s_th,
                th_g=th_g,
            )
            alpha_g = 1.0
            if track and deltas and pcache is not None and rec is not None:
                gm, pa, pkw, y_pre = pcache
                rec._g_probe_cache = None
                modes = [(m, m.training) for m in gm.modules()]
                prev = rec._in_probe
                rec._in_probe = True
                try:
                    gm.eval()
                    out = gm(*pa, **pkw)
                    t = out[0] if isinstance(out, (tuple, list)) and out else out
                    y_post = t.detach().float() if isinstance(t, torch.Tensor) else None
                finally:
                    rec._in_probe = prev
                    for m, fl in modes:
                        m.training = fl
                if y_post is not None and y_post.shape == y_pre.shape:
                    coord_scale = max(0.05, rec.ema_data_scale / math.sqrt(max(1, y_pre[0].numel())))
                    d_rel = float((y_post - y_pre).square().mean().sqrt().item()) / coord_scale
                    er = rec.ema_residual
                    d_max = (0.020 + 0.23 * min(1.5, er)) * g_g * s_step * mod_scale * max(1.0, self._wakeup_boost)
                    if er > 0.024 and s_th > 0.10 and self._g_mod_scale >= 0.5:
                        d_min = (
                            (0.0045 + 0.042 * min(1.0, er))
                            * _smoothstep01((er - 0.024) / 0.036)
                            * _smoothstep01((s_th - 0.10) / 0.40)
                            * s_step
                        )
                    else:
                        d_min = 0.0
                    if math.isfinite(d_rel) and d_rel > 1e-12:
                        if d_max > 0.0 and d_rel > d_max:
                            alpha_g = max(1e-4, d_max / d_rel)
                        elif d_min > 0.0 and d_rel < d_min:
                            alpha_g = min(96.0, max(1.0, d_min / (d_rel + d_min / 96.0)))
                    if alpha_g != 1.0:
                        for p, dp in deltas:
                            p.add_(dp, alpha=alpha_g - 1.0)
            elif rec is not None:
                rec._g_probe_cache = None

            if self._protected is not None and all_p:
                disp = torch.cat([(p - o).flatten() for p, o in zip(all_p, orig)])
                proj = _zero_pinv_project_nonascent(disp, self._protected)
                st = self.constraint_geometry_stats
                st["steps"] += 1
                st["max_derivative_before"] = max(
                    st["max_derivative_before"],
                    float((self._protected.double() @ disp.double()).max().item()),
                )
                if not torch.equal(proj, disp):
                    st["projected_steps"] += 1
                    off = 0
                    for p, o in zip(all_p, orig):
                        p.copy_(o + proj[off : off + p.numel()].view_as(p))
                        off += p.numel()
                applied = torch.cat([(p - o).flatten() for p, o in zip(all_p, orig)])
                st["max_derivative_after"] = max(
                    st["max_derivative_after"],
                    float((self._protected.double() @ applied.double()).max().item()),
                )
            self.last_step_gain = float(alpha_g)
        finally:
            self._protected = None
            self._wakeup_boost = 1.0
            self._g_mod_scale = 1.0
            self.clear_sampled_rows()
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "minimal_opt": {
                "schema": 1,
                "last_step_gain": float(self.last_step_gain),
                "stats": dict(self.constraint_geometry_stats),
                "pending": None if self._protected is None else self._protected.detach().clone(),
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        sc = dict(state_dict)
        meta = sc.pop("minimal_opt", None)
        if (
            not isinstance(meta, dict)
            or set(meta) != {"schema", "last_step_gain", "stats", "pending"}
            or meta["schema"] != 1
            or not _is_finite_num(meta["last_step_gain"], 1e-12)
        ):
            raise ValueError("invalid MinimalGeneratorOptimizer state")
        all_p = [p for g in self.param_groups for p in g["params"]]
        pending = meta["pending"]
        if pending is not None:
            total_numel = sum(p.numel() for p in all_p)
            if (
                not isinstance(pending, torch.Tensor)
                or pending.ndim != 2
                or not (1 <= int(pending.shape[0]) <= 2)
                or int(pending.shape[1]) != total_numel
                or not bool(torch.isfinite(pending).all())
            ):
                raise ValueError("invalid MinimalGeneratorOptimizer pending protected gradient")
        super().load_state_dict(sc)
        _restore_fp32_opt_states(self, state_dict)
        _validate_opt_states(self)
        ref_dev = all_p[0].device if all_p else torch.device("cpu")
        self.last_step_gain = float(meta["last_step_gain"])
        self.constraint_geometry_stats = dict(meta["stats"])
        self._protected = None if pending is None else pending.detach().to(device=ref_dev).clone()


class MinimalConstraintGeneratorOptimizer(MinimalGeneratorOptimizer):
    """Constraint-geometry-enabled minimal generator optimizer."""

    bind_protected_losses = MinimalGeneratorOptimizer._bind_protected_losses
