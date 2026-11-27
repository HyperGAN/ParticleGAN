"""Axiom Module D: Law 4.1-4.5 Role-Agnostic Equilibrated Optimizer."""
from __future__ import annotations

from copy import copy, deepcopy
import math
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

_ALLOWED_ROLES = frozenset({"generator", "encoder", "router", "noise", "critic", "prior", "table"})
_ALLOWED_OPTIONS = frozenset({"weight_decay", "amsgrad", "maximize", "differentiable", "capturable", "foreach", "fused"})
_FP32_STATE_KEYS = ("m_prior", "v_pow", "m_unit", "m_scale")
_ALLOWED_PARAM_STATE_KEYS = frozenset({"step", "log2_ref", "log2_init", *_FP32_STATE_KEYS})
_SKETCH_DIM = 128
_PHI1 = 0.6180339887498949
_QUINTIC_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_EYE_CACHE: Dict[Tuple[torch.device, int, torch.dtype], torch.Tensor] = {}


def _ss01_f(x: float) -> float:
    u = min(1.0, max(0.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


def _ss01_t(x: torch.Tensor) -> torch.Tensor:
    u = x.clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _cached_eye(m: int, dev: torch.device, dt: torch.dtype) -> torch.Tensor:
    key = (dev, m, dt)
    if key not in _EYE_CACHE:
        if len(_EYE_CACHE) >= 128:
            _EYE_CACHE.clear()
        _EYE_CACHE[key] = torch.eye(m, device=dev, dtype=dt)
    return _EYE_CACHE[key]


def collect_conv_layouts(modules: Iterable[Optional[nn.Module]]) -> Dict[int, Tuple[bool, int]]:
    layouts: Dict[int, Tuple[bool, int]] = {}
    for root in modules:
        if not isinstance(root, nn.Module):
            continue
        for mod in root.modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                layouts[id(mod.weight)] = (isinstance(mod, nn.ConvTranspose2d), int(mod.groups))
    return layouts


def _me_unit_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fin = torch.isfinite(x).all(dim=1, keepdim=True)
    xs = torch.where(fin, x, torch.zeros_like(x))
    gm = xs.abs().amax(dim=1, keepdim=True)
    act = fin & (gm > 0.0)
    one = torch.ones_like(gm)
    _, exp = torch.frexp(torch.where(act, gm, one))
    e1 = (-exp) // 2
    xm = (xs * torch.ldexp(one, e1)) * torch.ldexp(one, (-exp) - e1)
    nrm = xm.norm(dim=1, keepdim=True)
    ok = act & (nrm > 0.0) & torch.isfinite(nrm)
    unit = torch.where(ok, xm / nrm.clamp_min(1e-30), torch.zeros_like(xm))
    log2_norm = exp.float() + torch.log2(nrm.clamp_min(1e-30))
    return unit, log2_norm, ok


@torch.no_grad()
def mantissa_exponent_normalize(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact IEEE-754 power-of-2 mantissa-exponent normalization (degree-0 across [1e-44, 3e+38])."""
    if g.numel() == 0:
        return (
            g.float().clone(),
            g.new_zeros((), dtype=torch.float32),
            g.new_zeros((), dtype=torch.bool),
        )
    u, l2, ok = _me_unit_rows(g.float().reshape(1, -1))
    return u.reshape_as(g), l2.reshape(()), ok.reshape(())


@torch.no_grad()
def fp32_tikhonov_ns_batched(
    mats: torch.Tensor,
    steps: int = 5,
    *,
    rel_smooth: float | torch.Tensor = 0.0,
    pre_normalized: bool = False,
) -> torch.Tensor:
    """Batched zero-SVD FP32 Tikhonov Quintic Newton-Schulz (FP32 for m<=256, BF16 only m>256 CUDA)."""
    if mats.ndim != 3:
        raise ValueError(f"fp32_tikhonov_ns_batched expects 3D batch, got {tuple(mats.shape)}")
    bs, rd, cd = int(mats.shape[0]), int(mats.shape[1]), int(mats.shape[2])
    dt0 = mats.dtype
    if bs == 0 or rd == 0 or cd == 0:
        return torch.zeros_like(mats)
    if pre_normalized:
        xu = mats.float()
        ok: Optional[torch.Tensor] = None
    else:
        xu_f, _, ok_f = _me_unit_rows(mats.float().reshape(bs, rd * cd))
        xu = xu_f.view(bs, rd, cd)
        ok = ok_f.view(bs, 1, 1)
    if isinstance(rel_smooth, torch.Tensor):
        sr = rel_smooth.to(device=mats.device, dtype=torch.float32).reshape(bs, 1, 1).clamp_min(0.0)
        hs = True
    else:
        sr = xu.new_full((bs, 1, 1), max(0.0, float(rel_smooth)))
        hs = float(rel_smooth) > 0.0
    if rd == 1 or cd == 1:
        res = xu * (torch.rsqrt(1.0 + sr.square()) if hs else 1.0)
        if ok is not None:
            res = torch.where(ok, res, torch.zeros_like(res))
        return res.to(dt0)
    tr = rd > cd
    xi = xu.transpose(1, 2) if tr else xu
    md, nd = int(xi.shape[1]), int(xi.shape[2])
    wdt = torch.bfloat16 if (mats.is_cuda and md > 256) else torch.float32
    aug = hs and md <= 512
    if aug:
        ia = 1.0 / (torch.sqrt(1.0 + sr.square()).clamp_min(1e-30) * 1.02)
        eye_b = _cached_eye(md, mats.device, wdt).unsqueeze(0).expand(bs, md, md)
        z = torch.cat(((xi * ia).to(wdt), eye_b * ((sr / math.sqrt(float(md))) * ia).to(wdt)), dim=2)
    else:
        z = (xi / 1.02).to(wdt)
    for a, b, c in _QUINTIC_COEFFS[:steps]:
        g = torch.bmm(z, z.transpose(1, 2))
        z = a * z + torch.bmm(b * g + c * torch.bmm(g, g), z)
    zf = z.float()
    sc = (math.sqrt(float(md)) / zf.flatten(1).norm(dim=1).view(bs, 1, 1).clamp_min(1e-8)).clamp_max(1.15)
    scale_eff = sc if (aug or not hs) else (sc * torch.rsqrt(1.0 + sr.square()))
    out = zf[:, :, :nd] * scale_eff
    if tr:
        out = out.transpose(1, 2)
    fin_out = torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    valid = fin_out if ok is None else (ok & fin_out)
    return torch.where(valid, out, torch.zeros_like(out)).to(dt0)


@torch.no_grad()
def fp32_tikhonov_ns_2d(
    mat: torch.Tensor,
    steps: int = 5,
    *,
    rel_smooth: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    if mat.ndim != 2:
        raise ValueError(f"fp32_tikhonov_ns_2d expects a 2D matrix, got {tuple(mat.shape)}")
    rs = rel_smooth.reshape(1) if isinstance(rel_smooth, torch.Tensor) else rel_smooth
    return fp32_tikhonov_ns_batched(mat.unsqueeze(0), steps=steps, rel_smooth=rs).squeeze(0)


@torch.no_grad()
def _schatten6_decay_batched(
    w_c: torch.Tensor,
    rates: torch.Tensor,
    *,
    lam: float = 0.25,
    tau: float = 2.0,
) -> torch.Tensor:
    """Zero-SVD, zero-sync, exact-deadband Schatten-6 spectral ceiling (tau=2.0) on [B, m, n]."""
    bs, md = int(w_c.shape[0]), int(w_c.shape[1])
    wh = w_c.float() / tau
    wb = wh.to(torch.bfloat16 if (w_c.is_cuda and md >= 64) else torch.float32)
    g1 = torch.bmm(wb, wb.transpose(1, 2))
    g2 = torch.bmm(g1, g1)
    g2f = g2.float()
    sg2 = g2f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    g2s = g2f / sg2
    g4u = torch.bmm(g2s.to(wb.dtype), g2s.to(wb.dtype)).float()
    rho = (
        (g4u.flatten(1).norm(dim=1).view(bs, 1, 1) / g2s.flatten(1).norm(dim=1).view(bs, 1, 1).clamp_min(1e-12))
        * sg2
    ).clamp_min(0.0).pow(0.25)
    gate = _ss01_t((rho - 1.0) / 0.25)
    w5 = torch.bmm(g2, wb).float()
    rat = (w5.flatten(1).norm(dim=1).view(bs, 1, 1) / wh.flatten(1).norm(dim=1).view(bs, 1, 1).clamp_min(1e-12)).clamp_min(1e-6)
    eff = torch.minimum((lam / tau) * rates.view(bs, 1, 1) * gate, 0.25 / rat)
    out = -w5 * (eff * tau)
    valid = torch.isfinite(out).all(dim=(1, 2), keepdim=True) & torch.isfinite(eff) & (gate > 0.0)
    return torch.where(valid, out, torch.zeros_like(out))


def _solve_nonascent_2d(c0: float, c1: float, rho: float, d2: float) -> Optional[Tuple[float, float]]:
    bdist = d2
    tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d2)))
    best: Optional[Tuple[float, float]] = None
    for ck, oth, w in ((c0, c1 - c0 * rho, (c0, 0.0)), (c1, c0 - c1 * rho, (0.0, c1))):
        dist = ck * ck
        if ck >= -tol and oth <= tol and dist < bdist:
            best = w
            bdist = dist
    det = 1.0 - rho * rho
    s = 1.0 if rho >= 0.0 else -1.0
    if det > 1e-14:
        m0 = (c0 - rho * c1) / det
        m1 = (c1 - rho * c0) / det
    else:
        m0 = 0.25 * (c0 + s * c1)
        m1 = 0.25 * (s * c0 + c1)
    if (
        m0 >= -tol
        and m1 >= -tol
        and (c0 - m0 - m1 * rho) <= tol
        and (c1 - m0 * rho - m1) <= tol
        and (m0 * m0 + m1 * m1 + 2.0 * m0 * m1 * rho) < bdist
    ):
        best = (m0, m1)
    return best


@torch.no_grad()
def zero_pinv_project_nonascent(displacement: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Exact nearest displacement satisfying ``a_i . d <= 0`` for 1 or 2 constraints (`0` `pinv`)."""
    d = displacement.double()
    a = normals.double()
    nc = int(a.shape[0])
    if nc > 2:
        raise ValueError("Axiom direction_blend supports at most two protected losses")
    if nc == 0:
        return displacement.clone()
    lens = a.norm(dim=1)
    ad = a @ d
    d2 = (d * d).sum()
    if nc == 1:
        l0, ad0, d2f = torch.stack((lens[0], ad[0], d2)).tolist()
        if l0 <= 0.0 or ad0 <= 0.0:
            return displacement.clone()
        c0 = ad0 / l0
        tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d2f)))
        if c0 >= -tol and c0 * c0 < d2f:
            return (d - (c0 / l0) * a[0]).to(displacement)
        return torch.zeros_like(d).to(displacement)
    l0, l1, ad0, ad1, dot01, d2f = torch.stack(
        (lens[0], lens[1], ad[0], ad[1], (a[0] * a[1]).sum(), d2)
    ).tolist()
    act = [(i, l, v) for i, (l, v) in enumerate(((l0, ad0), (l1, ad1))) if l > 0.0]
    if not act:
        return displacement.clone()
    if len(act) == 1:
        i0, lk, vk = act[0]
        ck = vk / lk
        if ck <= 0.0:
            return displacement.clone()
        tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d2f)))
        if ck >= -tol and ck * ck < d2f:
            return (d - (ck / lk) * a[i0]).to(displacement)
        return torch.zeros_like(d).to(displacement)
    c0 = ad0 / l0
    c1 = ad1 / l1
    rho = dot01 / (l0 * l1)
    if c0 <= 0.0 and c1 <= 0.0:
        return displacement.clone()
    w = _solve_nonascent_2d(c0, c1, rho, d2f)
    if w is None:
        return torch.zeros_like(d).to(displacement)
    return (d - (w[0] / l0) * a[0] - (w[1] / l1) * a[1]).to(displacement)


@torch.no_grad()
def _eval_probe_forward(
    mod: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    rec: Any = None,
) -> Optional[torch.Tensor]:
    modes = [(m, m.training) for m in mod.modules()]
    prev_p = bool(getattr(rec, "_in_probe", False)) if rec is not None else False
    cpu_r = torch.get_rng_state()
    py_r = random.getstate()
    np_m = sys.modules.get("numpy")
    np_r = np_m.random.get_state() if (np_m is not None and hasattr(np_m, "random")) else None
    c_dev = torch.cuda.current_device() if torch.cuda.is_initialized() else None
    c_r = torch.cuda.get_rng_state(c_dev) if c_dev is not None else None
    if rec is not None:
        rec._in_probe = True
    try:
        mod.eval()
        p0 = next(mod.parameters(), None)
        use_bf16 = bool(
            p0 is not None
            and p0.device.type == "cuda"
            and (
                getattr(mod, "_axiom_bf16_autocast", False)
                or getattr(mod, "_particlegan_bf16_autocast", False)
            )
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = mod(*args, **kwargs)
        tgt = out[0] if isinstance(out, (tuple, list)) and out else out
        return tgt.detach().to(torch.float32) if isinstance(tgt, torch.Tensor) else None
    finally:
        if rec is not None:
            rec._in_probe = prev_p
        for m, flag in modes:
            m.training = flag
        torch.set_rng_state(cpu_r)
        random.setstate(py_r)
        if c_r is not None and c_dev is not None:
            torch.cuda.set_rng_state(c_r, c_dev)
        if np_r is not None and np_m is not None:
            np_m.random.set_state(np_r)


def _flat_nocopy(t: torch.Tensor) -> torch.Tensor:
    if t.is_contiguous():
        return t.view(-1)
    if t.ndim == 4 and t.is_contiguous(memory_format=torch.channels_last):
        return t.permute(0, 2, 3, 1).view(-1)
    return t.reshape(-1)


def _update_l2_ref(st: Dict[str, Any], lt: float, g_eq: float, dec: float) -> float:
    if not bool(st.get("log2_init", False)):
        st["log2_init"] = True
        st["log2_ref"] = float(lt)
        return float(lt)
    rp = float(st.get("log2_ref", 0.0))
    if lt > rp:
        st["log2_ref"] = rp + 0.05 * min(1.0, lt - rp)
    else:
        st["log2_ref"] = g_eq * (rp - dec) + (1.0 - g_eq) * (rp + 0.05 * max(-1.0, lt - rp))
    return float(st["log2_ref"])


def _apply_mom(st: Dict[str, Any], p: torch.Tensor, gu: torch.Tensor, b1: float) -> torch.Tensor:
    if b1 <= 0.0:
        return gu
    mu = st.setdefault("m_unit", torch.zeros_like(p, dtype=torch.float32))
    ms = st.setdefault("m_scale", torch.zeros((), device=p.device, dtype=torch.float32))
    gu, _, _ = mantissa_exponent_normalize(b1 * mu + (1.0 - b1) * gu)
    mu.copy_(gu)
    ms.fill_(1.0)
    return gu


def _is_fin_num(
    v: Any,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    slo: bool = False,
) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return False
    fv = float(v)
    if lo is not None and (fv <= lo if slo else fv < lo):
        return False
    if hi is not None and fv > hi:
        return False
    return True


class AxiomOptimizer(Optimizer):
    """Law 4.1-4.5 role-agnostic scale-invariant equilibrated optimizer."""

    family = "axiom"
    recipe_optimizer_family = "axiom"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 0.014,
        betas: Tuple[float, float] = (0.0, 0.999),
        eps: float = 1e-8,
        smoothing: float = 0.05,
        direction_blend: bool = False,
        critic: Optional[nn.Module] = None,
        record: Any = None,
        conv_layouts: Optional[Dict[int, Tuple[bool, int]]] = None,
        **options: Any,
    ) -> None:
        if any(not _is_fin_num(v, 0.0, None, slo) for v, slo in ((lr, False), (eps, True), (smoothing, False))):
            raise ValueError("invalid AxiomOptimizer scalar hyperparameter")
        betas = tuple(float(b) for b in betas)
        if (
            type(direction_blend) is not bool
            or len(betas) != 2
            or any(not (0.0 <= b < 1.0) or not math.isfinite(b) for b in betas)
        ):
            raise ValueError("invalid AxiomOptimizer betas or direction_blend")
        for k, v in options.items():
            if k not in _ALLOWED_OPTIONS:
                raise TypeError(f"unsupported AxiomOptimizer option: {k}")
            if v not in (None, False, 0, 0.0):
                raise ValueError(f"AxiomOptimizer requires disabled {k}")

        self.family = "axiom"
        self.recipe_optimizer_family = "axiom"
        self.smoothing = float(smoothing)
        self.direction_blend = bool(direction_blend)
        self.critic = critic
        self.record = record
        self._axiom_record = record
        self.ema_critic = None
        self.anchor = None
        self.guard = None
        self.latent_damping = None
        self.latent_history = None
        self.direct_response = None
        self.direct_history = None
        self._conv_layouts: Dict[int, Tuple[bool, int]] = dict(conv_layouts) if conv_layouts else {}
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._sampled_rows_unique: Dict[torch.Tensor, torch.Tensor] = {}
        self._row_scales: Dict[torch.Tensor, torch.Tensor] = {}
        self._gen_param_ids: set[int] = set()
        self._metrics_cache: Dict[int, Tuple[float, float, int, int, int]] = {}
        self._mod_g_scale: float | torch.Tensor = 1.0
        self._mod_wakeup_boost: float | torch.Tensor = 1.0
        self._mod_prior_scale = 1.0
        self.last_step_gain = 1.0
        self._ema_func_gain = 1.0
        self._ema_step_cos_val = 0.0
        self._log2_damp_val = 0.0
        self._crit_skew_ema = 0.0
        self._ema_step_cos_t: Optional[torch.Tensor] = None
        self._log2_damp_t: Optional[torch.Tensor] = None
        self._prev_step_sketch: Optional[torch.Tensor] = None
        self._fast_cache: Optional[Dict[str, Any]] = None
        self._protected: Optional[torch.Tensor] = None
        self._protected_fin_t: Optional[torch.Tensor] = None
        self._ss_coef: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.last_probe_consumed = False
        self._cg_stats: Dict[str, Any] = {
            "steps": 0,
            "projected_steps": 0,
            "max_derivative_before": 0.0,
            "max_derivative_after": 0.0,
        }
        self._db_stats: Dict[str, Any] = {
            "conflict_steps": 0,
            "blended_steps": 0,
            "pareto_stalls": 0,
            "retained_norm_ratio_sum": 0.0,
            "max_retained_norm_ratio": 0.0,
        }
        self._cg_pending_t: Optional[Dict[str, torch.Tensor]] = None
        self.ns_2d_activations = 0
        self.fast_path_steps = 0
        self.trust_backtracks_total = 0
        self.probe_consumptions = 0

        default_role = "critic" if critic is not None else "generator"
        super().__init__(
            params,
            {
                "lr": float(lr),
                "betas": betas,
                "eps": float(eps),
                "role": default_role,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "differentiable": False,
                "capturable": False,
                "foreach": None,
                "fused": None,
                **options,
            },
        )
        for grp in self.param_groups:
            role = grp.get("role", grp.get("forge_role", default_role))
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"unknown optimizer parameter role: {role!r}")
            grp["role"] = role
            if role in ("prior", "table"):
                if len(grp["params"]) != 1 or grp["params"][0].ndim != 2:
                    raise ValueError("prior/table group must hold one 2D latent table")
                grp["algorithm"] = "axiom_prior"
                grp.setdefault("sampled_rows_required", role == "prior")
            else:
                grp["algorithm"] = "axiom"
        if isinstance(critic, nn.Module):
            self.bind_module_layouts((critic,))

    def bind_module_layouts(self, modules: Iterable[Optional[nn.Module]]) -> None:
        self._conv_layouts.update(collect_conv_layouts(modules))
        self._metrics_cache.clear()

    def _active_record(self) -> Any:
        if self.record is not None:
            return self.record
        if self._axiom_record is not None:
            return self._axiom_record
        for grp in self.param_groups:
            for p in grp["params"]:
                rec = getattr(p, "_axiom_record", None)
                if rec is not None:
                    return rec
        return None

    def _parameters(self) -> List[torch.Tensor]:
        return [p for g in self.param_groups for p in g["params"]]

    def _ref_device(self) -> torch.device:
        for grp in self.param_groups:
            if grp["params"]:
                return grp["params"][0].device
        return torch.device("cpu")

    def set_generator_probe_parameters(self, parameters: Iterable[torch.Tensor]) -> None:
        self._gen_param_ids = {id(p) for p in parameters}

    def bind_origin_sample(self, sample: Optional[torch.Tensor]) -> None:
        del sample

    def _flush_cg_stats(self) -> None:
        pend = self._cg_pending_t
        if pend is None:
            return
        self._cg_pending_t = None
        vals = torch.stack(
            (
                pend["proj"].double(),
                pend["max_before"].double(),
                pend["max_after"].double(),
                pend["conflict"].double(),
                pend["blended"].double(),
                pend["ret_sum"].double(),
                pend["ret_max"].double(),
            )
        ).tolist()
        self._cg_stats["projected_steps"] += int(round(vals[0]))
        self._cg_stats["max_derivative_before"] = max(self._cg_stats["max_derivative_before"], float(vals[1]))
        self._cg_stats["max_derivative_after"] = max(self._cg_stats["max_derivative_after"], float(vals[2]))
        self._db_stats["conflict_steps"] += int(round(vals[3]))
        self._db_stats["blended_steps"] += int(round(vals[4]))
        self._db_stats["retained_norm_ratio_sum"] += float(vals[5])
        self._db_stats["max_retained_norm_ratio"] = max(self._db_stats["max_retained_norm_ratio"], float(vals[6]))

    @property
    def constraint_geometry_stats(self) -> Dict[str, Any]:
        self._flush_cg_stats()
        return self._cg_stats

    @constraint_geometry_stats.setter
    def constraint_geometry_stats(self, value: Dict[str, Any]) -> None:
        self._cg_pending_t = None
        self._cg_stats = value

    @property
    def direction_blend_stats(self) -> Dict[str, Any]:
        self._flush_cg_stats()
        return self._db_stats

    @direction_blend_stats.setter
    def direction_blend_stats(self, value: Dict[str, Any]) -> None:
        self._cg_pending_t = None
        self._db_stats = value

    def _flush_scalars(self) -> None:
        if self._ema_step_cos_t is None and self._log2_damp_t is None:
            return
        ts = [t.double() for t in (self._ema_step_cos_t, self._log2_damp_t) if t is not None]
        vs = torch.stack(ts).tolist() if len(ts) > 1 else [float(ts[0].item())]
        idx = 0
        if self._ema_step_cos_t is not None:
            self._ema_step_cos_val = float(vs[idx])
            self._ema_step_cos_t = None
            idx += 1
        if self._log2_damp_t is not None:
            self._log2_damp_val = float(vs[idx])
            self._log2_damp_t = None

    def _flush_fast_cache(self) -> None:
        cache = self._fast_cache
        if cache is None:
            return
        packed = torch.cat(
            (cache["log2_ref"].detach().double(), cache["log2_init"].detach().double()),
            dim=0,
        ).cpu().tolist()
        n = len(cache["params"])
        for i, p in enumerate(cache["params"]):
            self.state[p]["log2_ref"] = float(packed[i])
            self.state[p]["log2_init"] = bool(packed[n + i] > 0.5)

    @property
    def ema_step_cos(self) -> float:
        self._flush_scalars()
        return self._ema_step_cos_val

    @property
    def log2_damp(self) -> float:
        self._flush_scalars()
        return self._log2_damp_val

    def _param_metrics(self, p: torch.Tensor) -> Tuple[float, float, int, int, int]:
        cached = self._metrics_cache.get(id(p))
        if cached is not None:
            return cached
        if p.ndim == 4:
            c0, c1, kh, kw = (int(s) for s in p.shape)
            is_t, gr = self._conv_layouts.get(id(p), (False, 1))
            oc, ic = (c1 * gr, c0) if is_t else (c0, c1 * gr)
            if kh == 1 and kw == 1 and gr == 1 and not is_t:
                out = (
                    math.sqrt(max(1.0, float(oc) / float(max(1, ic)))),
                    0.5 * math.log2(float(max(1, min(oc, ic)))),
                    oc,
                    ic,
                    min(oc, ic),
                )
            else:
                k_eff = (kh * kw) // 4 if is_t else kh * kw
                fac = (
                    math.sqrt(float(oc) / float(max(1, ic)))
                    * math.sqrt(float(gr) / float(max(1, k_eff)))
                    * (float(max(1, min(oc // max(1, gr), ic // max(1, gr)))) ** 0.25)
                )
                out = (fac, 0.5 * math.log2(float(max(1, min(oc, ic) * kh * kw))), 1, 1, 1)
        elif p.ndim >= 2:
            rd = int(p.shape[0])
            cd = int(p.numel() // max(1, rd))
            out = (
                math.sqrt(max(1.0, float(rd) / float(max(1, cd)))),
                0.5 * math.log2(float(max(1, min(rd, cd)))),
                rd,
                cd,
                min(rd, cd),
            )
        else:
            out = (0.25 if p.numel() == 1 else 1.0, 0.0, 1, 1, 1)
        self._metrics_cache[id(p)] = out
        return out

    def set_sampled_rows(self, param: torch.Tensor, rows: torch.Tensor) -> None:
        rows_t = (rows if isinstance(rows, torch.Tensor) else torch.as_tensor(rows)).reshape(-1).to(
            dtype=torch.long, device=param.device
        )
        rec = self._active_record()
        if rec is not None:
            rec._active_prior_table = param
            rec._active_batch_rows = rows_t.detach()
        if rows_t.device.type == "cpu" and rows_t.numel() and bool(((rows_t < 0) | (rows_t >= len(param))).any()):
            raise ValueError("sampled rows out of bounds for prior table")
        if param not in self._sampled_rows:
            self._sampled_rows[param] = rows_t
        else:
            self._sampled_rows[param] = torch.cat((self._sampled_rows[param], rows_t))
        self._sampled_rows_unique.pop(param, None)
        param._axiom_active_rows = self._sampled_rows[param].detach()  # type: ignore[attr-defined]

    def sampled_rows_for(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        if param not in self._sampled_rows:
            return None
        uniq = self._sampled_rows_unique.setdefault(param, torch.unique(self._sampled_rows[param], sorted=True))
        return uniq.detach().clone()

    def clear_sampled_rows(self) -> None:
        for p in self._sampled_rows:
            if hasattr(p, "_axiom_active_rows"):
                delattr(p, "_axiom_active_rows")
        self._sampled_rows.clear()
        self._sampled_rows_unique.clear()
        self._row_scales.clear()
        rec = self._active_record()
        if rec is not None:
            rec._active_prior_table = None
            rec._active_batch_rows = None

    def set_row_scales(self, param: torch.Tensor, scales: Optional[torch.Tensor]) -> None:
        if scales is None:
            self._row_scales.pop(param, None)
        else:
            self._row_scales[param] = scales.detach().to(device=param.device, dtype=torch.float32)

    def bind_axiom_modulation(
        self,
        *,
        g_scale: float | torch.Tensor = 1.0,
        wakeup_boost: float | torch.Tensor = 1.0,
        prior_scale: float = 1.0,
    ) -> None:
        self._mod_g_scale = (
            g_scale.detach().float().reshape(()) if isinstance(g_scale, torch.Tensor) else float(g_scale)
        )
        self._mod_wakeup_boost = (
            wakeup_boost.detach().float().reshape(())
            if isinstance(wakeup_boost, torch.Tensor)
            else float(wakeup_boost)
        )
        self._mod_prior_scale = float(prior_scale)

    bind_v4_modulation = bind_axiom_modulation

    def _assemble_grad_normal(self, ref_dev: torch.device) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for grp in self.param_groups:
            is_ps = grp["algorithm"] == "axiom_prior" and grp.get("sampled_rows_required", True)
            for p in grp["params"]:
                if p.grad is None:
                    pieces.append(torch.zeros(p.numel(), device=ref_dev, dtype=torch.float32))
                elif is_ps:
                    rows = self.sampled_rows_for(p)
                    if rows is None:
                        raise ValueError("direction_blend requires explicit sampled rows")
                    masked = torch.zeros_like(p.grad, dtype=torch.float32)
                    masked[rows] = p.grad.detach()[rows].float()
                    pieces.append(masked.to(device=ref_dev).reshape(-1))
                else:
                    pieces.append(p.grad.detach().to(device=ref_dev, dtype=torch.float32).reshape(-1))
        if not pieces:
            return torch.zeros(0, device=ref_dev, dtype=torch.float32)
        return torch.cat(pieces)

    def bind_protected_from_grads(self) -> None:
        if not self.direction_blend:
            return
        if self._protected is not None:
            raise ValueError("Axiom direction_blend losses already pending")
        ref_dev = self._ref_device()
        stacked = self._assemble_grad_normal(ref_dev).unsqueeze(0)
        self._protected = stacked
        self._protected_fin_t = torch.isfinite(stacked).all()

    def bind_protected_losses(
        self,
        losses: Sequence[torch.Tensor],
        *,
        protected_evaluator: Any = None,
    ) -> None:
        del protected_evaluator
        if not self.direction_blend:
            return
        if self._protected is not None:
            raise ValueError("Axiom direction_blend losses already pending")
        lt = tuple(losses)
        if not 1 <= len(lt) <= 2 or any(
            not isinstance(x, torch.Tensor) or x.ndim != 0 or not x.requires_grad for x in lt
        ):
            raise ValueError("one or two active scalar protected losses required")
        params = self._parameters()
        ref_dev = self._ref_device()
        trainable = [p for p in params if p.requires_grad]
        saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
        normals: List[torch.Tensor] = []
        try:
            for lk in lt:
                for p in params:
                    p.grad = None
                if trainable:
                    lk.backward(retain_graph=True, inputs=trainable)
                normals.append(self._assemble_grad_normal(ref_dev))
        finally:
            for p, sg in zip(params, saved):
                p.grad = sg
        stacked = torch.stack(normals)
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError("nonfinite direction_blend protected gradient")
        self._protected = stacked
        self._protected_fin_t = None

    def abort_pending(self) -> None:
        self.clear_sampled_rows()
        self._mod_g_scale = 1.0
        self._mod_wakeup_boost = 1.0
        self._mod_prior_scale = 1.0
        self._protected = None
        self._protected_fin_t = None

    def _update_step_cosine(
        self,
        net_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        dev: torch.device,
        ema_fg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        if self._ema_step_cos_t is not None and self._ema_step_cos_t.device == dev:
            rho_p = self._ema_step_cos_t
        else:
            rho_p = torch.full((), self._ema_step_cos_val, dtype=torch.float64, device=dev)
        if self._log2_damp_t is not None and self._log2_damp_t.device == dev:
            damp_p = self._log2_damp_t
        else:
            damp_p = torch.full((), self._log2_damp_val, dtype=torch.float64, device=dev)
        rho_n = rho_p
        had_hist = self._prev_step_sketch is not None
        act_p = [(p, _flat_nocopy(g.detach())) for p, g in net_pairs if g.numel() > 0]
        if act_p:
            # Architecture-agnostic: sketch p.ndim >= 2 weights (falling back to all params if none exist)
            # using odd stride to prevent power-of-2 channel aliasing.
            w_fls = [fl for p, fl in act_p if p.ndim >= 2]
            if not w_fls:
                w_fls = [fl for _, fl in act_p]
            rows: List[torch.Tensor] = []
            for fl in w_fls:
                n_el = int(fl.numel())
                if n_el >= _SKETCH_DIM:
                    stride = max(1, n_el // _SKETCH_DIM)
                    if stride > 1 and stride % 2 == 0:
                        stride -= 1
                    rows.append(fl[::stride][:_SKETCH_DIM].float())
                else:
                    pad = fl.new_zeros(_SKETCH_DIM - n_el, dtype=torch.float32)
                    rows.append(torch.cat((fl.float(), pad), dim=0))
            bm = torch.stack(rows, dim=0)
            if bm.dtype != torch.float32:
                bm = bm.float()
            bmax = bm.abs().amax(dim=1, keepdim=True)
            fin = torch.isfinite(bmax)
            bm = torch.where(fin & (bmax < 1e-30), bm * 18446744073709551616.0, torch.where(fin, bm, torch.zeros_like(bm)))
            bmax = torch.where(
                fin & (bmax < 1e-30),
                bmax * 18446744073709551616.0,
                torch.where(fin, bmax, torch.zeros_like(bmax)),
            )
            bu = bm / bmax.clamp_min(1e-37)
            sk_acc = torch.where(
                bmax > 0.0,
                bu / bu.norm(dim=1, keepdim=True).clamp_min(1e-30),
                torch.zeros_like(bu),
            ).sum(dim=0)
            sk_n = sk_acc.norm()
            sk_ok = torch.isfinite(sk_n) & (sk_n > 0.0)
            sk_now = torch.where(sk_ok, sk_acc / sk_n.clamp_min(1e-30), torch.zeros_like(sk_acc))
            prev_sk = self._prev_step_sketch
            if prev_sk is not None:
                if prev_sk.device != dev:
                    prev_sk = prev_sk.to(device=dev, dtype=torch.float32)
                cos_val = (sk_now * prev_sk).sum().double().clamp(-1.0, 1.0)
                rho_n = torch.where(sk_ok, 0.90 * rho_p + 0.10 * cos_val, rho_p)
            self._prev_step_sketch = torch.where(sk_ok, sk_now, prev_sk if prev_sk is not None else sk_now)

        if self._ss_coef is None or self._ss_coef[0].device != dev:
            self._ss_coef = (
                torch.tensor([-1.0 / 0.25, 1.0 / 0.35, -1.0 / 0.35, -1.0 / 0.30], dtype=torch.float64, device=dev),
                torch.tensor([-0.45 / 0.25, 0.35 / 0.35, -0.35 / 0.35, 0.55 / 0.30], dtype=torch.float64, device=dev),
            )
        g4 = _ss01_t(rho_n * self._ss_coef[0] + self._ss_coef[1])
        d_dec = 0.12 * g4[0]
        log2_d = (damp_p - d_dec + 0.25 * g4[1]).clamp(-10.0, 0.0)
        self._ema_step_cos_t = rho_n
        self._log2_damp_t = log2_d
        stiff = torch.maximum(
            rho_n.new_full((), _ss01_f((0.18 - ema_fg) / 0.07)),
            _ss01_f((0.60 - ema_fg) / 0.30) * g4[0],
        )
        return torch.exp2(log2_d), float(self.smoothing) * g4[2], g4[3], stiff, had_hist

    def _detect_critic_pathology(
        self,
        crit_m4: List[Tuple[Any, ...]],
        pre_ok_skew: Optional[Tuple[bool, float]] = None,
    ) -> bool:
        """Compound Law 4.2 critic pathology witness: requires BOTH inter-layer skew > 4x AND min r_stable < 1.05."""
        if len(crit_m4) < 3:
            return False
        if pre_ok_skew is None:
            nrms = torch.stack(torch._foreach_norm([it[1].float() for it in crit_m4])).double()
            ok = torch.isfinite(nrms) & (nrms > 1e-17) & (nrms < 1e17)
            lt_v = torch.log2(torch.where(ok, nrms, torch.ones_like(nrms))) - torch.tensor(
                [it[4] for it in crit_m4], dtype=torch.float64, device=nrms.device
            )
            aok, skf = torch.stack((ok.all().double(), lt_v.amax() - lt_v.amin())).tolist()
            pre_ok_skew = (aok > 0.5, float(skf))
        if pre_ok_skew[0]:
            skew_now = pre_ok_skew[1]
        else:
            l2s: List[float] = []
            for it in crit_m4:
                _, l2n, ok_i = mantissa_exponent_normalize(it[1])
                if bool(ok_i.item()):
                    l2s.append(float(l2n.item()) - float(it[4]))
            if len(l2s) < 3:
                return False
            skew_now = max(l2s) - min(l2s)
        self._crit_skew_ema = max(skew_now, 0.97 * self._crit_skew_ema + 0.03 * skew_now)
        if self._crit_skew_ema <= 2.0:
            return False
        r_stabs: List[torch.Tensor] = []
        for p, g, _, _, _, rd, cd, _, _, _ in crit_m4:
            gu, _, ok_i = mantissa_exponent_normalize(g)
            mc = gu.reshape(rd, cd).T if rd > cd else gu.reshape(rd, cd)
            nd = int(mc.shape[1])
            st = self.state[p]
            vp = st.get("v_pow")
            if vp is None or vp.shape != (nd,) or vp.device != mc.device or vp.dtype != torch.float32:
                vp = torch.cos(torch.arange(1, nd + 1, device=mc.device, dtype=torch.float32) * (_PHI1 * 2.0 * math.pi))
                vp = vp / vp.norm().clamp_min(1e-12)
            for _ in range(2):
                u = mc @ vp
                un = u.norm()
                u = torch.where(un > 1e-20, u / un.clamp_min(1e-20), u)
                vn = mc.T @ u
                vnn = vn.norm()
                vp = torch.where(vnn > 1e-20, vn / vnn.clamp_min(1e-20), vp)
            st["v_pow"] = vp.detach()
            r_stabs.append(torch.where(ok_i, 1.0 / (mc @ vp).square().sum().clamp_min(1e-30), mc.new_full((), 1e6)))
        return min(torch.stack(r_stabs).tolist()) < 1.05

    def _two_obj_pyth(self) -> Dict[int, float]:
        if self._protected is None or int(self._protected.shape[0]) != 2:
            return {}
        n0, n1 = self._protected[0], self._protected[1]
        rp: Dict[str, List[int]] = {}
        spans: Dict[str, List[Tuple[int, int]]] = {}
        off = 0
        for grp in self.param_groups:
            pids = rp.setdefault(grp["role"], [])
            sp = spans.setdefault(grp["role"], [])
            for p in grp["params"]:
                sz = p.numel()
                pids.append(id(p))
                if p.grad is not None and sz > 0:
                    if sp and sp[-1][1] == off:
                        sp[-1] = (sp[-1][0], off + sz)
                    else:
                        sp.append((off, off + sz))
                off += sz
        sr: List[str] = []
        st_list: List[torch.Tensor] = []
        acc: Dict[str, List[float]] = {r: [0.0, 0.0, 0.0] for r in rp}
        for r, sp in spans.items():
            for s0, s1 in sp:
                g0 = n0[s0:s1].double()
                g1 = n1[s0:s1].double()
                sr.append(r)
                st_list.extend(((g0 * g0).sum(), (g1 * g1).sum(), (g0 * g1).sum()))
        if st_list:
            vs = torch.stack(st_list).tolist()
            for i, r in enumerate(sr):
                a2 = float(vs[3 * i])
                b2 = float(vs[3 * i + 1])
                dt = float(vs[3 * i + 2])
                if a2 > 1e-24 and b2 > 1e-24 and math.isfinite(a2) and math.isfinite(b2) and math.isfinite(dt):
                    acc[r][0] += a2
                    acc[r][1] += b2
                    acc[r][2] += dt
        out: Dict[int, float] = {}
        for r, pids in rp.items():
            a2, b2, dt = acc[r]
            if a2 > 1e-24 and b2 > 1e-24:
                boost = min(2.0, max(1.0, (math.sqrt(a2) + math.sqrt(b2)) / math.sqrt(max(a2 + b2 + 2.0 * dt, 1e-36))))
            else:
                boost = 1.0
            for pid in pids:
                out[pid] = boost
        return out

    def _prep_prior(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        grp: Dict[str, Any],
        b1: float,
    ) -> Optional[Tuple[Any, ...]]:
        st = self.state[p]
        r_sc = self._row_scales.get(p)
        req = bool(grp.get("sampled_rows_required", True))
        if req and r_sc is None:
            raw = self._sampled_rows[p]
            if raw.numel() == 0:
                return None
            ridx, _ = torch.sort(raw)
            is_f = torch.ones(int(ridx.shape[0]), device=p.device, dtype=torch.bool)
            if ridx.shape[0] > 1:
                is_f[1:] = ridx[1:] != ridx[:-1]
            rmask = is_f.unsqueeze(1)
        else:
            ridx = self.sampled_rows_for(p) if req else torch.arange(len(p), device=p.device, dtype=torch.long)
            if ridx is None or ridx.numel() == 0:
                return None
            rmask = torch.ones((int(ridx.shape[0]), 1), device=p.device, dtype=torch.bool)
        gr = grad[ridx].float()
        fin = torch.isfinite(gr).all(dim=-1, keepdim=True)
        dr = torch.where(fin, gr, torch.zeros_like(gr))
        if b1 > 0.0:
            mp = st.setdefault("m_prior", torch.zeros_like(p, dtype=torch.float32))
            mpv = mp[ridx]
            dr = torch.where(fin, mpv.mul(b1).add(dr, alpha=1.0 - b1), mpv)
            mp.index_add_(0, ridx, (dr - mpv) * rmask.float())
        u_row, l2, ok0 = _me_unit_rows(dr)
        ok = rmask & ok0
        nok = ok.float().sum()
        mean_l2 = ((l2 * ok.float()).sum() / nok.clamp_min(1.0)).double()
        return p, ridx, r_sc, u_row, l2, ok, nok.double(), mean_l2

    def _finish_prior(
        self,
        pr: Tuple[Any, ...],
        n_ok: float,
        mean_l2: float,
        *,
        base_lr: float,
        s_rel_pr: float,
        s_peak_pr: float,
        g_eq: float,
        dec: float,
        had_hist: bool,
    ) -> None:
        if base_lr <= 0.0:
            return
        p, ridx, r_sc, u_row, l2, ok, _, _ = pr
        st = self.state[p]
        if n_ok > 0.0:
            ref_n = _update_l2_ref(st, float(mean_l2), g_eq, dec)
        else:
            ref_n = float(st.get("log2_ref", 0.0))
        if had_hist:
            rat = torch.exp2((ref_n - l2).clamp(-30.0, 20.0))
            smooth = float(s_rel_pr) * rat + float(s_peak_pr) * (rat - 1.0).clamp_min(0.0)
            ur = torch.where(ok, u_row * torch.rsqrt(1.0 + smooth.square()), torch.zeros_like(u_row))
        else:
            ur = torch.where(ok, u_row, torch.zeros_like(u_row))
        scaled = ur * r_sc.unsqueeze(1) if (r_sc is not None and len(r_sc) == len(ridx)) else ur
        p.index_add_(0, ridx, scaled.to(p.dtype), alpha=-base_lr)

    def _run_foreach_fast(
        self,
        items: List[Tuple[torch.Tensor, torch.Tensor, float, float, float, bool]],
        *,
        c_osc: torch.Tensor,
        s_rel: torch.Tensor,
        s_eq_base: torch.Tensor,
        s_peak: torch.Tensor,
        g_eq: torch.Tensor,
        dec: float,
        sm_r: float,
        had_hist: bool,
        g_mult: float | torch.Tensor,
        d_mult: float | torch.Tensor,
        pre_norms: Optional[torch.Tensor] = None,
        pre_all_ok: Optional[bool] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, float, float, float, bool]]:
        params = [it[0] for it in items]
        grads = [it[1] for it in items]
        dev = params[0].device
        key = tuple(id(p) for p in params)
        lr_key = tuple(it[2] for it in items)
        fac_key = tuple(it[3] for it in items)
        c = self._fast_cache
        if c is None or c["key"] != key or c["device"] != dev or c["fac_key"] != fac_key:
            self._flush_fast_cache()
            c = {
                "key": key,
                "device": dev,
                "params": params,
                "lr_key": lr_key,
                "fac_key": fac_key,
                "lrs": torch.tensor(lr_key, dtype=torch.float64, device=dev),
                "factors": torch.tensor(fac_key, dtype=torch.float64, device=dev),
                "half_l2": torch.tensor([it[4] for it in items], dtype=torch.float64, device=dev),
                "is_gen": torch.tensor([it[5] for it in items], dtype=torch.bool, device=dev),
                "is_sc": torch.tensor([p.numel() == 1 for p in params], dtype=torch.bool, device=dev),
                "log2_ref": torch.tensor(
                    [float(self.state[p].get("log2_ref", 0.0)) for p in params],
                    dtype=torch.float64,
                    device=dev,
                ),
                "log2_init": torch.tensor(
                    [bool(self.state[p].get("log2_init", False)) for p in params],
                    dtype=torch.bool,
                    device=dev,
                ),
            }
            self._fast_cache = c
        elif c["lr_key"] != lr_key:
            c["lr_key"] = lr_key
            c["lrs"] = torch.tensor(lr_key, dtype=torch.float64, device=dev)

        if all(g.dtype == torch.float32 for g in grads):
            gs_f = grads
        else:
            gs_f = [g.detach() if g.dtype == torch.float32 else g.detach().float() for g in grads]
        norms = pre_norms if pre_norms is not None else torch.stack(torch._foreach_norm(gs_f)).double()
        ok = torch.isfinite(norms) & (norms > 1e-17) & (norms < 1e17)
        all_ok = pre_all_ok if pre_all_ok is not None else bool(ok.all().item())
        keep: List[int] = []
        ml: List[bool] = []
        if not all_ok:
            ml = ok.tolist()
            keep = [i for i, b in enumerate(ml) if b]
            if not keep:
                return list(items)
        lrs = c["lrs"]
        if isinstance(g_mult, torch.Tensor) or g_mult != 1.0:
            gm_v = g_mult.to(dev, dtype=torch.float64) if isinstance(g_mult, torch.Tensor) else float(g_mult)
            lrs = torch.where(c["is_gen"], lrs * gm_v, lrs)
        if isinstance(d_mult, torch.Tensor) or d_mult != 1.0:
            dm_v = d_mult.to(dev, dtype=torch.float64) if isinstance(d_mult, torch.Tensor) else float(d_mult)
            lrs = torch.where(~c["is_gen"], lrs * dm_v, lrs)
        sn = norms if all_ok else torch.where(ok, norms, torch.ones_like(norms))
        lt = torch.log2(sn) - c["half_l2"]
        rp = c["log2_ref"]
        ip = c["log2_init"]
        rn = torch.where(
            ip,
            torch.where(
                lt > rp,
                rp + 0.05 * (lt - rp).clamp_max(1.0),
                g_eq * (rp - dec) + (1.0 - g_eq) * (rp + 0.05 * (lt - rp).clamp_min(-1.0)),
            ),
            lt,
        )
        c["log2_ref"].copy_(rn if all_ok else torch.where(ok, rn, rp))
        c["log2_init"].copy_(ip | ok)
        if had_hist:
            rat = torch.exp2((c["log2_ref"] - lt).clamp(-30.0, 20.0))
            rs_v = (s_rel + s_eq_base * g_eq) * rat + (s_peak * g_eq) * (rat - 1.0).clamp_min(0.0)
            rs_eff = torch.where(c["is_sc"], torch.maximum(rs_v, (sm_r * 0.25) * rat), rs_v)
            damp = torch.rsqrt(1.0 + rs_eff.square())
        else:
            damp = 1.0
        self.fast_path_steps += 1
        sc = (-(lrs * c["factors"] * c_osc * damp) / sn).float()
        if all_ok:
            torch._foreach_addcmul_(params, grads, list(sc.unbind(0)))
            return []
        torch._foreach_addcmul_([params[i] for i in keep], [grads[i] for i in keep], [sc[i] for i in keep])
        return [it for it, b in zip(items, ml) if not b]

    def _apply_verified_trust(
        self,
        rec: Any,
        tracked: List[Tuple[torch.Tensor, torch.Tensor]],
        *,
        step_scale: float,
        s_therm: float,
        pre_gain: float | torch.Tensor,
        is_simp: bool,
        stiff_g: float,
    ) -> Tuple[float, bool]:
        cache = rec._g_probe_cache
        rec._g_probe_cache = None
        if cache is None:
            return 1.0, False
        g_mod, pa, pk, y_pre = cache
        deltas = [(p, o, (p - o).detach()) for p, o in tracked]
        yp = _eval_probe_forward(g_mod, pa, pk, rec)
        if yp is None or yp.shape != y_pre.shape:
            return 1.0, False
        bp = int(y_pre.shape[0])
        dev = yp.device
        ek, et = rec.take_pending_ema_tensors() if hasattr(rec, "take_pending_ema_tensors") else ([], [])
        fin0 = torch.isfinite(yp).all().double()
        diff0 = torch.where(torch.isfinite(yp), yp.double() - y_pre.double(), yp.new_zeros((), dtype=torch.float64))
        rms0 = diff0.reshape(bp, -1).square().mean().sqrt()
        has_pg_t = isinstance(pre_gain, torch.Tensor)
        can_pack_e = bool(et) and all(t.device == dev for t in et)
        pack_list = [fin0, rms0]
        if has_pg_t:
            pack_list.append(pre_gain.to(device=dev, dtype=torch.float64).reshape(()))
        if can_pack_e:
            pack_list.extend(t.double() for t in et)
        pv = torch.stack(pack_list).tolist()
        pg_f = float(pv[2]) if has_pg_t else float(pre_gain)
        if et:
            flushed_vals = [float(v) for v in pv[2 + int(has_pg_t) :]] if can_pack_e else [float(t.item()) for t in et]
            rec.apply_flushed_ema_values(ek, flushed_vals)
        coord_sc = max(1e-30, float(getattr(rec, "ema_data_scale", 1.0)) / math.sqrt(float(max(1, int(y_pre[0].numel())))))
        push_res = float(getattr(rec, "ema_push_res", 0.0))
        d_max = (0.045 + 0.16 * min(1.5, push_res)) * step_scale
        if (not is_simp) and push_res > 0.18 and s_therm > 0.10:
            d_min = 0.012 * _ss01_f((push_res - 0.18) / 0.25) * step_scale
        else:
            d_min = 0.0

        def _rel_rms(y: Optional[torch.Tensor]) -> float:
            if y is None:
                return float("inf")
            vs = torch.stack(
                (
                    torch.isfinite(y).all().double(),
                    (y.double() - y_pre.double()).reshape(bp, -1).square().mean().sqrt(),
                )
            ).tolist()
            if vs[0] > 0.5:
                return float(vs[1]) / coord_sc
            return float("inf")

        def _set_scaled(a_val: float) -> None:
            for p, o, d in deltas:
                p.copy_(o + d * a_val)

        d_obs = float(pv[1]) / coord_sc if pv[0] > 0.5 else float("inf")
        alpha = 1.0
        if d_min > 0.0 and math.isfinite(d_obs) and 1e-12 < d_obs < d_min:
            alpha = min(16.0, max(1.0, d_min / (d_obs + d_min / 16.0)))
            _set_scaled(alpha)
            d_obs = _rel_rms(_eval_probe_forward(g_mod, pa, pk, rec))
        for _ in range(4):
            if d_obs <= d_max:
                break
            self.trust_backtracks_total += 1
            if math.isfinite(d_obs) and d_obs > 1e-30:
                alpha = max(1e-6, min(0.5 * alpha, 0.9 * alpha * (d_max / d_obs)))
            else:
                alpha = max(1e-6, 0.1 * alpha)
            _set_scaled(alpha)
            d_obs = _rel_rms(_eval_probe_forward(g_mod, pa, pk, rec))
        if not math.isfinite(d_obs) or d_obs > d_max:
            alpha = max(1e-6, 0.5 * alpha)
            _set_scaled(alpha)
        if alpha < 1.0:
            self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * max(0.05, min(1.0, pg_f * alpha)))
        elif s_therm >= 0.5 and math.isfinite(d_obs) and (alpha > 1.0 or d_obs >= 0.012):
            fr = (d_max / max(d_obs, 1e-12)) * pg_f
            if fr > 1.5 and min(1.0, fr) > self._ema_func_gain:
                self._ema_func_gain = float(0.995 * self._ema_func_gain + 0.005 * max(0.05, min(1.0, fr)))
        rec.g_func_gain = float((1.0 - stiff_g) * self._ema_func_gain + stiff_g)
        return alpha, True

    def _apply_direction_blend(self, all_p: List[torch.Tensor], origs: List[torch.Tensor]) -> None:
        assert self._protected is not None
        normals = self._protected.double()
        ref_dev = normals.device
        nc = int(normals.shape[0])
        if nc == 0:
            self._cg_stats["steps"] += 1
            return
        d64 = torch.cat(
            [(p.detach() - old).to(device=ref_dev, dtype=torch.float64).reshape(-1) for p, old in zip(all_p, origs)]
        )
        if nc == 1:
            # Zero-host-sync exact 1-constraint non-ascent projection on-device.
            n0 = normals[0]
            b0 = (n0 * d64).sum()
            l0 = n0.norm()
            d2 = (d64 * d64).sum()
            has_conflict = (b0 > 0.0) & (l0 > 0.0)
            c0 = b0 / l0.clamp_min(1e-30)
            tol = 1e-10 * torch.maximum(d2.new_ones(()), d2.clamp_min(0.0).sqrt())
            proj_ok = (c0 >= -tol) & (c0 * c0 < d2)
            proj_d = torch.where(
                proj_ok,
                d64 - (c0 / l0.clamp_min(1e-30)) * n0,
                torch.zeros_like(d64),
            )
            off = 0
            applied_parts: List[torch.Tensor] = []
            for p, old in zip(all_p, origs):
                sz = p.numel()
                sl = proj_d[off : off + sz].view_as(p).to(device=p.device, dtype=p.dtype)
                cond_p = has_conflict.to(device=p.device)
                p.copy_(torch.where(cond_p, old + sl, p))
                applied_parts.append((p.detach() - old).to(device=ref_dev, dtype=torch.float64).reshape(-1))
                off += sz
            applied = torch.cat(applied_parts)
            after_val = (n0 * applied).sum()
            app_n2 = (applied * applied).sum()
            ratio = torch.where(
                has_conflict & (d2 > 0.0),
                app_n2.clamp_min(0.0).sqrt() / d2.clamp_min(1e-30).sqrt(),
                d2.new_zeros(()),
            )
            hc_l = has_conflict.long()
            self._cg_stats["steps"] += 1
            pend = self._cg_pending_t
            if pend is None:
                self._cg_pending_t = {
                    "proj": hc_l,
                    "max_before": b0,
                    "max_after": after_val,
                    "conflict": hc_l,
                    "blended": hc_l,
                    "ret_sum": ratio,
                    "ret_max": ratio,
                }
            else:
                if pend["proj"].device != ref_dev:
                    self._flush_cg_stats()
                    self._cg_pending_t = {
                        "proj": hc_l,
                        "max_before": b0,
                        "max_after": after_val,
                        "conflict": hc_l,
                        "blended": hc_l,
                        "ret_sum": ratio,
                        "ret_max": ratio,
                    }
                else:
                    pend["proj"] = pend["proj"] + hc_l
                    pend["max_before"] = torch.maximum(pend["max_before"], b0)
                    pend["max_after"] = torch.maximum(pend["max_after"], after_val)
                    pend["conflict"] = pend["conflict"] + hc_l
                    pend["blended"] = pend["blended"] + hc_l
                    pend["ret_sum"] = pend["ret_sum"] + ratio
                    pend["ret_max"] = torch.maximum(pend["ret_max"], ratio)
            return

        self._flush_cg_stats()
        stats = self._cg_stats
        blend = self._db_stats
        disp = d64.float()
        before = normals @ d64
        an = normals.norm(dim=1)
        d2 = (d64 * d64).sum()
        b0, b1, l0, l1, dot01, d2f = torch.stack(
            (before[0], before[1], an[0], an[1], (normals[0] * normals[1]).sum(), d2)
        ).tolist()
        bmax = max(b0, b1)
        proj = False
        if bmax > 0.0:
            blend["conflict_steps"] += 1
            if l0 > 0.0 and l1 > 0.0:
                rho = dot01 / (l0 * l1)
                poss = math.sqrt(max(0.0, 0.5 * (1.0 + rho))) > 1e-12
                w = _solve_nonascent_2d(b0 / l0, b1 / l1, rho, d2f) if poss else None
                if (not poss) or (w is None):
                    direc: Optional[torch.Tensor] = None
                    app_n2 = 0.0
                    after_max = 0.0
                    proj = True
                else:
                    direc = (d64 - (w[0] / l0) * normals[0] - (w[1] / l1) * normals[1]).to(disp)
                    app_n2 = max(
                        0.0,
                        d2f
                        - w[0] * (2.0 * (b0 / l0) - w[0])
                        - w[1] * (2.0 * (b1 / l1) - w[1])
                        + 2.0 * w[0] * w[1] * rho,
                    )
                    after_max = max(
                        b0 - w[0] * l0 - w[1] * (dot01 / l1),
                        b1 - w[0] * (dot01 / l0) - w[1] * l1,
                    )
                    proj = (w[0] != 0.0) or (w[1] != 0.0)
            else:
                poss = True
                direc = zero_pinv_project_nonascent(disp, normals)
                app = direc.double()
                app_n2 = float((app * app).sum().item())
                after_max = float((normals @ app).max().item())
                proj = not torch.equal(app, d64)
            blend["blended_steps" if poss else "pareto_stalls"] += 1
            for p, old in zip(all_p, origs):
                p.copy_(old)
            if direc is not None:
                off = 0
                for p in all_p:
                    sz = p.numel()
                    p.add_(direc[off : off + sz].view_as(p).to(device=p.device, dtype=p.dtype))
                    off += sz
            dn = math.sqrt(max(0.0, d2f))
            ratio_f = math.sqrt(max(0.0, app_n2)) / dn if dn > 0.0 else 0.0
            blend["retained_norm_ratio_sum"] += ratio_f
            blend["max_retained_norm_ratio"] = max(blend["max_retained_norm_ratio"], ratio_f)
        else:
            after_max = bmax
        stats["steps"] += 1
        stats["projected_steps"] += int(proj)
        stats["max_derivative_before"] = max(stats["max_derivative_before"], bmax)
        stats["max_derivative_after"] = max(stats["max_derivative_after"], after_max)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.direction_blend and (closure is not None or self._protected is None):
            self.abort_pending()
            raise ValueError("direction_blend step requires explicit protected-loss backward")

        net_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ref_dev: Optional[torch.device] = None
        for grp in self.param_groups:
            is_pr = grp["algorithm"] == "axiom_prior"
            for p in grp["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse or g.is_complex():
                    raise ValueError("AxiomOptimizer requires dense real gradients")
                if is_pr and grp.get("sampled_rows_required", True) and p not in self._sampled_rows:
                    raise ValueError("row-normalized prior updates require actual sampled rows")
                if ref_dev is None:
                    ref_dev = p.device
                if not is_pr:
                    net_pairs.append((p, g))

        all_p = self._parameters() if (self.direction_blend and self._protected is not None) else None
        origs = [p.detach().clone() for p in all_p] if all_p is not None else None

        try:
            rec = self._active_record()
            is_crit = self.critic is not None
            if rec is not None:
                step_scale = float(getattr(rec, "step_scale", 1.0))
                s_therm = float(getattr(rec, "thermal_scale", 1.0))
                is_simp = bool(getattr(rec, "_simplicial_step", False))
                is_cf = bool(getattr(rec, "cond_initialized", False))
            else:
                step_scale = 1.0
                s_therm = 1.0
                is_simp = False
                is_cf = False

            if ref_dev is None:
                if self._protected_fin_t is not None and not bool(self._protected_fin_t.item()):
                    raise ValueError("nonfinite direction_blend protected gradient")
                self.last_step_gain = 1.0
                self.last_probe_consumed = False
                if rec is not None and not is_crit:
                    rec._g_probe_cache = None
                return loss

            pyth = (
                self._two_obj_pyth()
                if (is_cf and self.direction_blend and self._protected is not None and int(self._protected.shape[0]) == 2)
                else {}
            )
            ema_fg = float(self._ema_func_gain)
            pg_b = 1.0 if ema_fg >= 0.75 else ema_fg
            if isinstance(self._mod_wakeup_boost, torch.Tensor):
                pre_g: float | torch.Tensor = (pg_b * self._mod_wakeup_boost.to(ref_dev)).clamp_max(1.0)
            else:
                pre_g = min(1.0, pg_b * float(self._mod_wakeup_boost))
            if isinstance(self._mod_g_scale, torch.Tensor):
                g_mult: float | torch.Tensor = self._mod_g_scale.to(ref_dev) * pre_g
            else:
                g_mult = self._mod_g_scale * pre_g
            d_cpl: float | torch.Tensor = 1.0
            if is_crit and rec is not None:
                g_fg = float(getattr(rec, "g_func_gain", 1.0))
                if g_fg < 0.75:
                    dg = max(0.12, g_fg)
                    if hasattr(rec, "_dev_ema"):
                        em = rec._dev_ema("ema_margin", ref_dev)
                        mi = rec._dev_ema("margin_initialized", ref_dev)
                        mg = torch.where(
                            mi > 0.5,
                            _ss01_t((em - 0.28) / 0.22) * _ss01_t((1.40 - em) / 0.30),
                            em.new_zeros(()),
                        )
                        d_cpl = (1.0 - mg) + mg * dg
                    elif getattr(rec, "margin_initialized", False):
                        em_f = float(getattr(rec, "ema_margin", 0.0))
                        mg_f = _ss01_f((em_f - 0.28) / 0.22) * _ss01_f((1.40 - em_f) / 0.30)
                        d_cpl = (1.0 - mg_f) + mg_f * dg

            gen_probe = (
                (not is_crit)
                and rec is not None
                and getattr(rec, "_g_probe_cache", None) is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cf
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )
            fast_it: List[Tuple[torch.Tensor, torch.Tensor, float, float, float, bool]] = []
            d2_it: List[Tuple[Any, ...]] = []
            slow_it: List[Tuple[Any, ...]] = []
            prior_it: List[Tuple[Any, ...]] = []
            crit_2d: List[Tuple[Any, ...]] = []
            tracked: List[Tuple[torch.Tensor, torch.Tensor]] = []
            s6_b: Dict[Tuple[int, int], List[Tuple[Any, ...]]] = {}

            s_cool = 1.0 if s_therm >= 0.9999 else (s_therm / math.hypot(s_therm, 0.0061))
            dec = 0.0007 if s_therm >= 0.5 else 0.0
            sm_r = float(self.smoothing) / 0.05

            for grp in self.param_groups:
                is_pr = grp["algorithm"] == "axiom_prior"
                is_gr = grp["role"] not in ("critic", "prior", "table")
                b1 = float(grp["betas"][0])
                b1_p = max(b1, 0.75) if (is_pr and self._mod_prior_scale < 1.0) else b1
                blr = float(grp["lr"]) * step_scale * s_cool * (self._mod_prior_scale if is_pr else 1.0)
                for p in grp["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    self.state[p]["step"] = int(self.state[p].get("step", 0)) + 1
                    lr_p = blr * pyth.get(id(p), 1.0)
                    if is_pr:
                        prior_it.append((p, g, lr_p, b1_p, grp))
                    elif lr_p > 0.0:
                        if gen_probe and (not self._gen_param_ids or id(p) in self._gen_param_ids):
                            tracked.append((p, p.detach().clone()))
                        fac, hl2, rd, cd, md = self._param_metrics(p)
                        if (p.ndim == 2 or p.ndim == 4) and md >= 2:
                            if is_gr:
                                d2_it.append((p, g, lr_p, fac, hl2, rd, cd, b1, True))
                            else:
                                crit_2d.append((p, g, lr_p, fac, hl2, rd, cd, md, b1, False))
                        elif (not gen_probe) and b1 == 0.0 and p.dtype == torch.float32 and g.dtype == torch.float32:
                            fast_it.append((p, g, lr_p, fac, hl2, is_gr))
                        else:
                            slow_it.append((p, g, lr_p, fac, hl2, b1, is_gr))

            pre_norms: Optional[torch.Tensor] = None
            pre_all_ok: Optional[bool] = None
            if crit_2d:
                crit_m4 = [it for it in crit_2d if it[7] >= 4]
                pre_os: Optional[Tuple[bool, float]] = None
                cn: Optional[torch.Tensor] = None
                f_ok = 0.0
                if (
                    len(crit_m4) >= 3
                    and all(
                        it[8] == 0.0
                        and it[0].dtype == torch.float32
                        and it[1].dtype == torch.float32
                        and it[0].device == ref_dev
                        for it in crit_2d
                    )
                    and all(it[0].device == ref_dev and it[1].dtype == torch.float32 for it in fast_it)
                ):
                    m4_i = [len(fast_it) + i for i, it in enumerate(crit_2d) if it[7] >= 4]
                    cn = torch.stack(
                        torch._foreach_norm([it[1] for it in fast_it] + [it[1] for it in crit_2d])
                    ).double()
                    ok_all = torch.isfinite(cn) & (cn > 1e-17) & (cn < 1e17)
                    cn4 = cn[m4_i]
                    ok4 = ok_all[m4_i]
                    lt4 = torch.log2(torch.where(ok4, cn4, torch.ones_like(cn4))) - torch.tensor(
                        [it[4] for it in crit_m4], dtype=torch.float64, device=ref_dev
                    )
                    f_ok, c_ok, sk_f = torch.stack((ok_all.all().double(), ok4.all().double(), lt4.amax() - lt4.amin())).tolist()
                    pre_os = (c_ok > 0.5, float(sk_f))
                if self._detect_critic_pathology(crit_m4, pre_os):
                    for p, g, lr_p, fac, hl2, rd, cd, _, b1, _ in crit_2d:
                        d2_it.append((p, g, lr_p, fac, hl2, rd, cd, b1, False))
                else:
                    for p, g, lr_p, fac, hl2, rd, cd, md, b1, _ in crit_2d:
                        fac_d = fac * math.sqrt(float(md))
                        if b1 == 0.0 and p.dtype == torch.float32 and g.dtype == torch.float32:
                            fast_it.append((p, g, lr_p, fac_d, hl2, False))
                        else:
                            slow_it.append((p, g, lr_p, fac_d, hl2, b1, False))
                        if s_therm > 0.0:
                            nt = rd > cd
                            s6_b.setdefault((cd, rd) if nt else (rd, cd), []).append(
                                (p, fac, nt, rd, cd, lr_p * s_therm)
                            )
                    if cn is not None:
                        pre_norms = cn
                        pre_all_ok = f_ok > 0.5

            c_osc, s_rel_raw, g_eq, stiff_t, had_hist = self._update_step_cosine(net_pairs, ref_dev, ema_fg)
            s_rel = s_rel_raw * (1.0 - stiff_t)
            s_peak_t = (sm_r * 0.20) * (1.0 - stiff_t)
            s_eq_t = (sm_r * 0.05) * (1.0 - stiff_t)

            if self._protected_fin_t is not None and not (prior_it or slow_it or d2_it):
                if not bool(self._protected_fin_t.item()):
                    raise ValueError("nonfinite direction_blend protected gradient")
                self._protected_fin_t = None

            if fast_it:
                spilled = self._run_foreach_fast(
                    fast_it,
                    c_osc=c_osc,
                    s_rel=s_rel,
                    s_eq_base=s_eq_t,
                    s_peak=s_peak_t,
                    g_eq=g_eq,
                    dec=dec,
                    sm_r=sm_r,
                    had_hist=had_hist,
                    g_mult=g_mult,
                    d_mult=d_cpl,
                    pre_norms=pre_norms,
                    pre_all_ok=pre_all_ok,
                )
                if spilled:
                    self._flush_fast_cache()
                    self._fast_cache = None
                    for p, g, lr_p, fac, hl2, is_gr in spilled:
                        slow_it.append((p, g, lr_p, fac, hl2, 0.0, is_gr))

            stiff_f = 0.0
            if prior_it or slow_it or d2_it:
                pre_pr = [(lr_p, self._prep_prior(p, g, grp, b1_p)) for p, g, lr_p, b1_p, grp in prior_it]
                has_gm_t = isinstance(g_mult, torch.Tensor)
                has_dm_t = isinstance(d_cpl, torch.Tensor)
                has_pfin = self._protected_fin_t is not None
                s_vec: List[torch.Tensor] = [c_osc.double(), s_rel.double(), g_eq.double(), stiff_t.double()]
                if has_gm_t:
                    s_vec.append(g_mult.to(ref_dev, dtype=torch.float64).reshape(()))
                if has_dm_t:
                    s_vec.append(d_cpl.to(ref_dev, dtype=torch.float64).reshape(()))
                pfin_idx = len(s_vec)
                if has_pfin:
                    assert self._protected_fin_t is not None
                    s_vec.append(self._protected_fin_t.to(device=ref_dev, dtype=torch.float64).reshape(()))
                pr_off = len(s_vec)
                for _, pr in pre_pr:
                    if pr is not None:
                        s_vec.extend((pr[6], pr[7]))
                net_all: List[Tuple[int, Tuple[Any, ...]]] = [(0, x) for x in slow_it] + [(1, x) for x in d2_it]
                off_n = len(s_vec)
                fp32_pairs = [
                    (i, it[1])
                    for i, (_, it) in enumerate(net_all)
                    if it[1].dtype == torch.float32 and it[1].device == ref_dev and it[1].numel() > 0
                ]
                fp32_idx = tuple(i for i, _ in fp32_pairs)
                if fp32_pairs:
                    s_vec.extend(n.double() for n in torch._foreach_norm([g for _, g in fp32_pairs]))
                sv = torch.stack(s_vec).tolist()
                if has_pfin:
                    self._protected_fin_t = None
                    if sv[pfin_idx] < 0.5:
                        raise ValueError("nonfinite direction_blend protected gradient")
                cosc_f = float(sv[0])
                srel_f = float(sv[1])
                geq_f = float(sv[2])
                stiff_f = float(sv[3])
                gm_f = float(sv[4]) if has_gm_t else float(g_mult)
                dm_f = float(sv[4 + int(has_gm_t)]) if has_dm_t else float(d_cpl)
                sp_eff = (sm_r * 0.20 * (1.0 - stiff_f)) * geq_f
                seq_eff = (sm_r * 0.05 * (1.0 - stiff_f)) * geq_f
                srel_pr = srel_f + (0.0 if self._mod_prior_scale < 1.0 else seq_eff)
                pi_k = pr_off
                for lr_p, pr in pre_pr:
                    if pr is not None:
                        self._finish_prior(
                            pr,
                            float(sv[pi_k]),
                            float(sv[pi_k + 1]),
                            base_lr=lr_p,
                            s_rel_pr=srel_pr,
                            s_peak_pr=1.25 * sp_eff,
                            g_eq=geq_f,
                            dec=dec,
                            had_hist=had_hist,
                        )
                        pi_k += 2
                fp32_map = {k: float(sv[off_n + j]) for j, k in enumerate(fp32_idx)}
                ns_b: Dict[Tuple[int, int], List[Tuple[Any, ...]]] = {}
                upd_p: List[torch.Tensor] = []
                upd_d: List[torch.Tensor] = []
                for i, (kind, it) in enumerate(net_all):
                    p, g = it[0], it[1]
                    nrm = fp32_map.get(i)
                    if nrm is not None and math.isfinite(nrm) and 1e-17 < nrm < 1e17:
                        gu = g / nrm
                        l2_v = math.log2(nrm)
                    else:
                        gu, l2n, ok = mantissa_exponent_normalize(g)
                        if not bool(ok.item()):
                            continue
                        l2_v = float(l2n.item())
                    if kind == 0:
                        _, _, lr_p, fac, hl2, b1, is_gr = it
                        rd = cd = 1
                    else:
                        _, _, lr_p, fac, hl2, rd, cd, b1, is_gr = it
                    st = self.state[p]
                    lt = l2_v - hl2
                    gu = _apply_mom(st, p, gu, b1)
                    rn = _update_l2_ref(st, lt, geq_f, dec)
                    rat = 2.0 ** max(-30.0, min(20.0, rn - lt))
                    rate = (lr_p * gm_f) if is_gr else (lr_p * dm_f)
                    rs = ((srel_f + seq_eff) * rat + sp_eff * max(0.0, rat - 1.0)) if had_hist else 0.0
                    if kind == 0:
                        if p.numel() == 1 and had_hist:
                            rs = max(rs, sm_r * 0.25 * rat)
                        upd_p.append(p)
                        upd_d.append((gu * (-(rate * fac) * cosc_f / math.sqrt(1.0 + rs * rs))).to(p.dtype))
                    else:
                        nt = rd > cd
                        mc = gu.reshape(rd, cd).T if nt else gu.reshape(rd, cd)
                        ns_b.setdefault((int(mc.shape[0]), int(mc.shape[1])), []).append(
                            (p, mc, nt, rate * fac * cosc_f, rs)
                        )
                for bkt in ns_b.values():
                    self.ns_2d_activations += len(bkt)
                    mats = torch.stack([b[1] for b in bkt], dim=0)
                    rs0 = bkt[0][4]
                    if all(b[4] <= 0.0 for b in bkt):
                        rs_arg: float | torch.Tensor = 0.0
                    elif all(b[4] == rs0 for b in bkt):
                        rs_arg = rs0
                    else:
                        rs_arg = torch.tensor([b[4] for b in bkt], dtype=torch.float32, device=mats.device)
                    orths = fp32_tikhonov_ns_batched(mats, steps=5, rel_smooth=rs_arg, pre_normalized=True).float()
                    for idx_b, (p_i, _, nt, mult, _) in enumerate(bkt):
                        upd_p.append(p_i)
                        orth_i = orths[idx_b].T if nt else orths[idx_b]
                        upd_d.append((orth_i * (-mult)).reshape_as(p_i).to(p_i.dtype))
                if upd_p:
                    if all(p.device == ref_dev and p.dtype == upd_p[0].dtype for p in upd_p):
                        torch._foreach_add_(upd_p, upd_d)
                    else:
                        for p_i, d_i in zip(upd_p, upd_d):
                            p_i.add_(d_i)

            for bkt_s in s6_b.values():
                if isinstance(d_cpl, torch.Tensor):
                    dm_t = d_cpl.to(ref_dev, dtype=torch.float32).reshape(())
                else:
                    dm_t = torch.full((), float(d_cpl), dtype=torch.float32, device=ref_dev)
                ws = torch.stack(
                    [
                        (p_i.float().reshape(rd, cd).T if nt else p_i.float().reshape(rd, cd)) / fac
                        for p_i, fac, nt, rd, cd, _ in bkt_s
                    ],
                    dim=0,
                )
                ds6 = _schatten6_decay_batched(
                    ws,
                    torch.tensor([r_s for *_, r_s in bkt_s], dtype=torch.float32, device=ws.device)
                    * dm_t.to(ws.device),
                )
                for idx_s, (p_i, fac, nt, _rd, _cd, _) in enumerate(bkt_s):
                    step_s = ds6[idx_s].T if nt else ds6[idx_s]
                    p_i.add_((step_s * fac).reshape_as(p_i).to(p_i.dtype))

            if gen_probe and tracked and rec is not None:
                alpha_g, p_ran = self._apply_verified_trust(
                    rec,
                    tracked,
                    step_scale=step_scale,
                    s_therm=s_therm,
                    pre_gain=pre_g,
                    is_simp=is_simp,
                    stiff_g=stiff_f,
                )
            else:
                alpha_g, p_ran = 1.0, False
                if rec is not None and not is_crit:
                    rec._g_probe_cache = None

            if self.direction_blend and origs is not None and self._protected is not None and all_p is not None:
                self._apply_direction_blend(all_p, origs)
            self.last_step_gain = float(alpha_g)
            self.last_probe_consumed = bool(p_ran)
            self.probe_consumptions += int(p_ran)
        finally:
            self.abort_pending()
        if self.record is not None:
            self.record.record_step(self)
        return loss

    def __getstate__(self) -> Dict[str, Any]:
        self._flush_scalars()
        self._flush_fast_cache()
        self._flush_cg_stats()
        return {**self.__dict__, "_fast_cache": None, "_metrics_cache": {}, "_ss_coef": None}

    def state_dict(self) -> Dict[str, Any]:
        self._flush_scalars()
        self._flush_fast_cache()
        self._flush_cg_stats()
        res = Optimizer.state_dict(self)
        axiom_meta: Dict[str, Any] = {
            "schema": 1,
            "family": self.family,
            "last_step_gain": float(self.last_step_gain),
            "ema_func_gain": float(self._ema_func_gain),
            "ema_step_cos": float(self._ema_step_cos_val),
            "log2_damp": float(self._log2_damp_val),
            "crit_skew_ema": float(self._crit_skew_ema),
            "direction_blend": bool(self.direction_blend),
            "prev_step_sketch": (
                None
                if self._prev_step_sketch is None
                else [float(x) for x in self._prev_step_sketch.detach().cpu().tolist()]
            ),
            "sampled_rows": [
                self.sampled_rows_for(g["params"][0]) if g["algorithm"] == "axiom_prior" else None
                for g in self.param_groups
            ],
        }
        if self.direction_blend:
            axiom_meta["constraint_geometry"] = {
                "stats": deepcopy(self._cg_stats),
                "blend_stats": deepcopy(self._db_stats),
                "pending": None if self._protected is None else self._protected.detach().cpu().clone(),
            }
        res["axiom"] = axiom_meta
        if self.record is not None:
            res["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return res

    def validate_state_dict(self, saved: Any) -> None:
        exp_top = {"state", "param_groups", "axiom"} | ({"regularizer"} if self.record is not None else set())
        if not isinstance(saved, dict) or set(saved) != exp_top:
            raise ValueError("invalid AxiomOptimizer checkpoint schema")
        meta = saved["axiom"]
        exp_meta = {
            "schema",
            "family",
            "last_step_gain",
            "ema_func_gain",
            "ema_step_cos",
            "log2_damp",
            "crit_skew_ema",
            "prev_step_sketch",
            "direction_blend",
            "sampled_rows",
        } | ({"constraint_geometry"} if self.direction_blend else set())
        if (
            not isinstance(meta, dict)
            or set(meta) != exp_meta
            or type(meta["schema"]) is not int
            or meta["schema"] != 1
            or meta["family"] != self.family
            or type(meta["direction_blend"]) is not bool
            or meta["direction_blend"] != self.direction_blend
            or not isinstance(meta["sampled_rows"], list)
            or len(meta["sampled_rows"]) != len(self.param_groups)
        ):
            raise ValueError("incompatible AxiomOptimizer metadata")
        scalar_checks = (
            ("last_step_gain", 0.0, None, True),
            ("ema_func_gain", 0.0, 1.0, True),
            ("ema_step_cos", -1.0, 1.0, False),
            ("log2_damp", -10.0, 0.0, False),
            ("crit_skew_ema", 0.0, None, False),
        )
        if any(not _is_fin_num(meta[k], lo, hi, slo) for k, lo, hi, slo in scalar_checks):
            raise ValueError("invalid AxiomOptimizer scalar metadata")
        sk = meta["prev_step_sketch"]
        if sk is not None and (not isinstance(sk, list) or len(sk) != _SKETCH_DIM or any(not _is_fin_num(x) for x in sk)):
            raise ValueError("invalid AxiomOptimizer prev_step_sketch")
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            total_numel = sum(p.numel() for p in self._parameters())
            if (
                not isinstance(cg, dict)
                or set(cg) != {"stats", "blend_stats", "pending"}
                or not isinstance(cg["stats"], dict)
                or set(cg["stats"]) != set(self._cg_stats)
                or not isinstance(cg["blend_stats"], dict)
                or set(cg["blend_stats"]) != set(self._db_stats)
            ):
                raise ValueError("invalid AxiomOptimizer direction_blend checkpoint")
            pending = cg["pending"]
            if pending is not None and (
                not isinstance(pending, torch.Tensor)
                or pending.ndim != 2
                or not (1 <= int(pending.shape[0]) <= 2)
                or int(pending.shape[1]) != total_numel
                or not bool(torch.isfinite(pending).all())
            ):
                raise ValueError("invalid AxiomOptimizer direction_blend checkpoint")
        if (
            not isinstance(saved["param_groups"], list)
            or len(saved["param_groups"]) != len(self.param_groups)
            or not isinstance(saved["state"], dict)
        ):
            raise ValueError("AxiomOptimizer group/state mismatch")
        for cur, sav in zip(self.param_groups, saved["param_groups"]):
            if (
                not isinstance(sav, dict)
                or len(sav.get("params", ())) != len(cur["params"])
                or sav.get("role") != cur["role"]
                or sav.get("algorithm") != cur["algorithm"]
            ):
                raise ValueError("AxiomOptimizer group mismatch")
        p_shapes = {
            sid: p.shape
            for cur, sav in zip(self.param_groups, saved["param_groups"])
            for p, sid in zip(cur["params"], sav["params"])
        }
        for sid, sv in saved["state"].items():
            if (
                not isinstance(sv, dict)
                or not set(sv.keys()).issubset(_ALLOWED_PARAM_STATE_KEYS)
                or type(sv.get("step", 0)) is not int
                or sv.get("step", 0) < 0
                or ("log2_ref" in sv and not _is_fin_num(sv["log2_ref"]))
                or ("log2_init" in sv and type(sv["log2_init"]) is not bool)
            ):
                raise ValueError("invalid AxiomOptimizer parameter state")
            for tk in _FP32_STATE_KEYS:
                if tk not in sv:
                    continue
                tv = sv[tk]
                if not isinstance(tv, torch.Tensor) or not tv.is_floating_point() or not bool(torch.isfinite(tv).all()):
                    raise ValueError("invalid AxiomOptimizer parameter state")
                if tk in ("m_unit", "m_prior") and sid in p_shapes and tv.shape != p_shapes[sid]:
                    raise ValueError("invalid AxiomOptimizer parameter state")
                if tk == "m_scale" and tv.ndim != 0:
                    raise ValueError("invalid AxiomOptimizer parameter state")
                if tk == "v_pow":
                    if tv.ndim != 1 or tv.numel() < 1:
                        raise ValueError("invalid AxiomOptimizer parameter state")
                    if sid in p_shapes and len(p_shapes[sid]) >= 2:
                        sh = p_shapes[sid]
                        rd = int(sh[0])
                        cd = int(math.prod(sh) // max(1, rd))
                        if int(tv.shape[0]) != max(rd, cd):
                            raise ValueError("invalid AxiomOptimizer parameter state")
        if self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid AxiomOptimizer regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        meta = state_dict["axiom"]
        Optimizer.load_state_dict(
            self,
            {k: v for k, v in state_dict.items() if k not in ("axiom", "regularizer")},
        )
        # PyTorch Optimizer.load_state_dict casts floating-point state tensors to p.dtype;
        # restore all FP32 optimizer state tensors bit-for-bit in torch.float32 on p.device.
        for cg, sg in zip(self.param_groups, state_dict["param_groups"]):
            for p, sid in zip(cg["params"], sg["params"]):
                sv = state_dict["state"].get(sid)
                if not isinstance(sv, dict):
                    continue
                for tk in _FP32_STATE_KEYS:
                    t = sv.get(tk)
                    if isinstance(t, torch.Tensor):
                        self.state[p][tk] = t.detach().to(device=p.device, dtype=torch.float32).clone()

        self.clear_sampled_rows()
        self._fast_cache = None
        self._ss_coef = None
        self._metrics_cache.clear()
        self._ema_step_cos_t = None
        self._log2_damp_t = None
        self._protected_fin_t = None
        self._cg_pending_t = None
        self.last_step_gain = float(meta["last_step_gain"])
        self._ema_func_gain = float(meta["ema_func_gain"])
        self._ema_step_cos_val = float(meta["ema_step_cos"])
        self._log2_damp_val = float(meta["log2_damp"])
        self._crit_skew_ema = float(meta["crit_skew_ema"])
        ref_dev = self._ref_device()
        self._prev_step_sketch = (
            None
            if meta["prev_step_sketch"] is None
            else torch.tensor(meta["prev_step_sketch"], dtype=torch.float32, device=ref_dev)
        )
        self.ns_2d_activations = 0
        self.fast_path_steps = 0
        self.trust_backtracks_total = 0
        self.probe_consumptions = 0
        self.last_probe_consumed = False
        for grp, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and grp["algorithm"] == "axiom_prior":
                self.set_sampled_rows(grp["params"][0], rows.to(grp["params"][0].device))
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            self._cg_stats = deepcopy(cg["stats"])
            self._db_stats = deepcopy(cg["blend_stats"])
            self._protected = (
                None
                if cg["pending"] is None
                else cg["pending"].detach().to(device=ref_dev, dtype=torch.float32).clone()
            )
        else:
            self._protected = None
        if self.record is not None:
            self.record.load_state_dict(state_dict["regularizer"]["record"])
            self.record.restore_device_state(ref_dev)


__all__ = [
    "AxiomOptimizer",
    "collect_conv_layouts",
    "fp32_tikhonov_ns_2d",
    "fp32_tikhonov_ns_batched",
    "mantissa_exponent_normalize",
    "zero_pinv_project_nonascent",
]
