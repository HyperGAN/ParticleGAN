"""Axiom Module D: Law 4.1-4.5 Role-Agnostic Equilibrated Optimizer."""
from __future__ import annotations

from copy import copy, deepcopy
import math, random, sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.optim import Optimizer

_ALLOWED_ROLES = frozenset({"generator", "encoder", "router", "noise", "critic", "prior", "table"})
_SKETCH_DIM = 128
_QUINTIC_COEFFS = ((4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012))
_EYE_CACHE: Dict[Tuple[torch.device, int, torch.dtype], torch.Tensor] = {}

def _cached_eye(m: int, dev: torch.device, dt: torch.dtype) -> torch.Tensor:
    if (key := (dev, m, dt)) not in _EYE_CACHE:
        if len(_EYE_CACHE) >= 128: _EYE_CACHE.clear()
        _EYE_CACHE[key] = torch.eye(m, device=dev, dtype=dt)
    return _EYE_CACHE[key]

def collect_conv_layouts(modules: Iterable[Optional[nn.Module]]) -> Dict[int, Tuple[bool, int]]:
    return {id(c.weight): (isinstance(c, nn.ConvTranspose2d), int(c.groups)) for r in modules if isinstance(r, nn.Module) for c in r.modules() if isinstance(c, (nn.Conv2d, nn.ConvTranspose2d))}

def _me_unit_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fin = torch.isfinite(x).all(dim=1, keepdim=True); xs = torch.where(fin, x, torch.zeros_like(x))
    gm = xs.abs().amax(dim=1, keepdim=True); act, one = fin & (gm > 0.0), torch.ones_like(gm)
    _, exp = torch.frexp(torch.where(act, gm, one)); e1 = (-exp) // 2
    xm = (xs * torch.ldexp(one, e1)) * torch.ldexp(one, (-exp) - e1); nrm = xm.norm(dim=1, keepdim=True)
    ok = act & (nrm > 0.0) & torch.isfinite(nrm)
    return torch.where(ok, xm / nrm.clamp_min(1e-30), torch.zeros_like(xm)), exp.float() + torch.log2(nrm.clamp_min(1e-30)), ok

@torch.no_grad()
def mantissa_exponent_normalize(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact IEEE-754 power-of-2 mantissa-exponent normalization (degree-0 across [1e-44, 3e+38])."""
    if g.numel() == 0: return g.float().clone(), g.new_zeros((), dtype=torch.float32), g.new_zeros((), dtype=torch.bool)
    u, l2, ok = _me_unit_rows(g.float().reshape(1, -1))
    return u.reshape_as(g), l2.reshape(()), ok.reshape(())

@torch.no_grad()
def fp32_tikhonov_ns_batched(mats: torch.Tensor, steps: int = 5, *, rel_smooth: float | torch.Tensor = 0.0) -> torch.Tensor:
    """Batched zero-SVD FP32 Tikhonov Quintic Newton-Schulz (FP32 for m<=256, BF16 only m>256 CUDA)."""
    if mats.ndim != 3: raise ValueError(f"fp32_tikhonov_ns_batched expects 3D batch, got {tuple(mats.shape)}")
    bs, rd, cd = int(mats.shape[0]), int(mats.shape[1]), int(mats.shape[2])
    if bs == 0 or rd == 0 or cd == 0: return torch.zeros_like(mats)
    dt0 = mats.dtype; xu_f, _, ok_f = _me_unit_rows(mats.float().reshape(bs, rd * cd))
    xu, ok = xu_f.view(bs, rd, cd), ok_f.view(bs, 1, 1)
    sr, hs = (rel_smooth.to(device=mats.device, dtype=torch.float32).reshape(bs, 1, 1).clamp_min(0.0), True) if isinstance(rel_smooth, torch.Tensor) else (xu.new_full((bs, 1, 1), max(0.0, float(rel_smooth))), float(rel_smooth) > 0.0)
    if rd == 1 or cd == 1: return torch.where(ok, xu * (torch.rsqrt(1.0 + sr.square()) if hs else 1.0), torch.zeros_like(xu)).to(dt0)
    tr = rd > cd; xi = xu.transpose(1, 2) if tr else xu; md, nd = int(xi.shape[1]), int(xi.shape[2])
    wdt, aug = (torch.bfloat16 if (mats.is_cuda and md > 256) else torch.float32), (hs and md <= 512)
    if aug:
        ia = 1.0 / (torch.sqrt(1.0 + sr.square()).clamp_min(1e-30) * 1.02)
        z = torch.cat(((xi * ia).to(wdt), _cached_eye(md, mats.device, wdt).unsqueeze(0).expand(bs, md, md) * ((sr / math.sqrt(float(md))) * ia).to(wdt)), dim=2)
    else: z = (xi / 1.02).to(wdt)
    for a, b, c in _QUINTIC_COEFFS[:steps]:
        g = torch.bmm(z, z.transpose(1, 2)); z = a * z + torch.bmm(b * g + c * torch.bmm(g, g), z)
    zf = z.float(); sc = (math.sqrt(float(md)) / zf.flatten(1).norm(dim=1).view(bs, 1, 1).clamp_min(1e-8)).clamp_max(1.15)
    out = zf[:, :, :nd] * (sc if aug or not hs else sc * torch.rsqrt(1.0 + sr.square())); out = out.transpose(1, 2) if tr else out
    return torch.where(ok & torch.isfinite(out).all(dim=(1, 2), keepdim=True), out, torch.zeros_like(out)).to(dt0)

@torch.no_grad()
def fp32_tikhonov_ns_2d(mat: torch.Tensor, steps: int = 5, *, rel_smooth: float | torch.Tensor = 0.0) -> torch.Tensor:
    if mat.ndim != 2: raise ValueError(f"fp32_tikhonov_ns_2d expects a 2D matrix, got {tuple(mat.shape)}")
    return fp32_tikhonov_ns_batched(mat.unsqueeze(0), steps=steps, rel_smooth=(rel_smooth.reshape(1) if isinstance(rel_smooth, torch.Tensor) else rel_smooth)).squeeze(0)

@torch.no_grad()
def zero_pinv_project_nonascent(displacement: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Exact nearest displacement satisfying ``a_i . d <= 0`` for 1 or 2 constraints (`0` `pinv`)."""
    d, a = displacement.double(), normals.double(); lens = a.norm(dim=1); a = a[lens > 0] / lens[lens > 0, None]
    if (nc := int(a.shape[0])) > 2: raise ValueError("Axiom direction_blend supports at most two protected losses")
    if nc == 0 or bool((a @ d <= 0.0).all()): return displacement.clone()
    best, bdist, tol = torch.zeros_like(d), float(d.square().sum().item()), 1e-10 * max(1.0, float(d.norm().item()))
    if nc == 1:
        c0 = float((a[0] @ d).item())
        return (d - c0 * a[0] if c0 >= -tol and float((c0 * a[0]).square().sum().item()) < bdist else best).to(displacement)
    a0, a1 = a[0], a[1]; c0, c1, rho = float((a0 @ d).item()), float((a1 @ d).item()), float((a0 @ a1).item())
    for ck, ak, ao in ((c0, a0, a1), (c1, a1, a0)):
        if ck >= -tol and float((ao @ (cand := d - ck * ak)).item()) <= tol and (dist := float((cand - d).square().sum().item())) < bdist: best, bdist = cand, dist
    det, s = 1.0 - rho * rho, (1.0 if rho >= 0.0 else -1.0)
    m0, m1 = ((c0 - rho * c1) / det, (c1 - rho * c0) / det) if det > 1e-14 else (0.25 * (c0 + s * c1), 0.25 * (s * c0 + c1))
    if m0 >= -tol and m1 >= -tol and float((a0 @ (cand := d - m0 * a0 - m1 * a1)).item()) <= tol and float((a1 @ cand).item()) <= tol and float((cand - d).square().sum().item()) < bdist: best = cand
    return best.to(displacement)

@torch.no_grad()
def _eval_probe_forward(mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], rec: Any = None) -> Optional[torch.Tensor]:
    modes, prev_p = [(m, m.training) for m in mod.modules()], (bool(getattr(rec, "_in_probe", False)) if rec is not None else False)
    cpu_r, py_r, np_m = torch.get_rng_state(), random.getstate(), sys.modules.get("numpy")
    np_r, c_dev = (np_m.random.get_state() if (np_m is not None and hasattr(np_m, "random")) else None), (torch.cuda.current_device() if torch.cuda.is_initialized() else None)
    c_r = torch.cuda.get_rng_state(c_dev) if c_dev is not None else None
    if rec is not None: rec._in_probe = True
    try:
        mod.eval(); p0 = next(mod.parameters(), None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(p0 is not None and p0.device.type == "cuda" and getattr(mod, "_axiom_bf16_autocast", False))): out = mod(*args, **kwargs)
        tgt = out[0] if isinstance(out, (tuple, list)) and out else out
        return tgt.detach().to(torch.float32) if isinstance(tgt, torch.Tensor) else None
    finally:
        if rec is not None: rec._in_probe = prev_p
        for m, f in modes: m.training = f
        torch.set_rng_state(cpu_r); random.setstate(py_r)
        if c_r is not None and c_dev is not None: torch.cuda.set_rng_state(c_r, c_dev)
        if np_r is not None and np_m is not None: np_m.random.set_state(np_r)

_flat_nocopy = lambda t: t.view(-1) if t.is_contiguous() else (t.permute(0, 2, 3, 1).view(-1) if (t.ndim == 4 and t.is_contiguous(memory_format=torch.channels_last)) else t.reshape(-1))

def _update_l2_ref(st: Dict[str, Any], lt: float, rp_f: float) -> float:
    if not bool(st.get("log2_init", False)): st["log2_init"], st["log2_ref"] = True, float(lt); return float(lt)
    rp = float(st.get("log2_ref", 0.0)); dl = max(-1.0, min(1.0, lt - rp))
    st["log2_ref"] = rp + (0.05 * dl if (rp_f >= 0.0 or dl > 0.0) else 0.0); return float(st["log2_ref"])

def _apply_mom(st: Dict[str, Any], p: torch.Tensor, gu: torch.Tensor, b1: float) -> torch.Tensor:
    if b1 <= 0.0: return gu
    mu, ms = st.setdefault("m_unit", torch.zeros_like(p, dtype=torch.float32)), st.setdefault("m_scale", torch.zeros((), device=p.device, dtype=torch.float32))
    gu, _, _ = mantissa_exponent_normalize(b1 * mu + (1.0 - b1) * gu); mu.copy_(gu); ms.fill_(1.0); return gu

class AxiomOptimizer(Optimizer):
    """Law 4.1-4.5 role-agnostic scale-invariant equilibrated optimizer."""
    family = recipe_optimizer_family = "axiom"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True

    def __init__(self, params: Iterable[Any], *, lr: float = 0.014, betas: Tuple[float, float] = (0.0, 0.999),
                 eps: float = 1e-8, smoothing: float = 0.05, direction_blend: bool = False,
                 critic: Optional[nn.Module] = None, record: Any = None,
                 conv_layouts: Optional[Dict[int, Tuple[bool, int]]] = None, **options: Any) -> None:
        for nm, v, slo in (("lr", lr, False), ("eps", eps, True), ("smoothing", smoothing, False)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v <= 0.0 if slo else v < 0.0): raise ValueError(f"invalid AxiomOptimizer {nm}")
        betas = tuple(float(b) for b in betas)
        if type(direction_blend) is not bool or len(betas) != 2 or any(not (0.0 <= b < 1.0) or not math.isfinite(b) for b in betas): raise ValueError("invalid AxiomOptimizer betas or direction_blend")
        for k, v in options.items():
            if k not in {"weight_decay", "amsgrad", "maximize", "differentiable", "capturable", "foreach", "fused"}: raise TypeError(f"unsupported AxiomOptimizer option: {k}")
            if v not in (None, False, 0, 0.0): raise ValueError(f"AxiomOptimizer requires disabled {k}")
        self.family = self.recipe_optimizer_family = "axiom"; self.smoothing, self.direction_blend, self.critic, self.record, self._axiom_record = float(smoothing), bool(direction_blend), critic, record, record
        self.ema_critic = self.anchor = self.guard = self.latent_damping = self.latent_history = self.direct_response = self.direct_history = None
        self._conv_layouts: Dict[int, Tuple[bool, int]] = {**(dict(conv_layouts) if conv_layouts else {}), **(collect_conv_layouts((critic,)) if isinstance(critic, nn.Module) else {})}
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}; self._sampled_rows_unique: Dict[torch.Tensor, torch.Tensor] = {}
        self._row_scales: Dict[torch.Tensor, torch.Tensor] = {}; self._gen_param_ids: set[int] = set(); self._metrics_cache: Dict[int, Tuple[float, float, int, int, int]] = {}
        self._mod_g_scale: float | torch.Tensor = 1.0; self._mod_wakeup_boost: float | torch.Tensor = 1.0
        self._mod_prior_scale = self.last_step_gain = self._ema_func_gain = 1.0; self._ema_step_cos_val = self._log2_damp_val = 0.0
        self._ema_step_cos_t = self._log2_damp_t = self._prev_step_sketch = self._fast_cache = self._protected = self._origin_norm_t = self._ss_coef = None
        self._protected_single = self.last_probe_consumed = False
        self.constraint_geometry_stats = dict(steps=0, projected_steps=0, max_derivative_before=0.0, max_derivative_after=0.0)
        self.direction_blend_stats = dict(conflict_steps=0, blended_steps=0, pareto_stalls=0, retained_norm_ratio_sum=0.0, max_retained_norm_ratio=0.0)
        self.ns_2d_activations = self.fast_path_steps = self.trust_backtracks_total = self.probe_consumptions = 0
        default_role = "critic" if critic is not None else "generator"
        super().__init__(params, {"lr": float(lr), "betas": betas, "eps": float(eps), "role": default_role, "weight_decay": 0.0, "amsgrad": False, "maximize": False, "differentiable": False, "capturable": False, "foreach": None, "fused": None, **options})
        for grp in self.param_groups:
            role = grp.get("role", grp.get("forge_role", default_role))
            if role not in _ALLOWED_ROLES: raise ValueError(f"unknown optimizer parameter role: {role!r}")
            grp["role"] = role
            if role in ("prior", "table"):
                if len(grp["params"]) != 1 or grp["params"][0].ndim != 2: raise ValueError("prior/table group must hold one 2D latent table")
                grp["algorithm"] = "axiom_prior"; grp.setdefault("sampled_rows_required", role == "prior")
            else: grp["algorithm"] = "axiom"

    def _active_record(self) -> Any: return self.record if self.record is not None else self._axiom_record
    def _parameters(self) -> List[torch.Tensor]: return [p for g in self.param_groups for p in g["params"]]
    def set_generator_probe_parameters(self, parameters: Iterable[torch.Tensor]) -> None: self._gen_param_ids = {id(p) for p in parameters}
    def bind_origin_sample(self, sample: Optional[torch.Tensor]) -> None: self._origin_norm_t = sample.detach().float().norm(dim=-1).amax() if (sample is not None and sample.ndim == 2 and sample.shape[1] == 2 and sample.numel() > 0) else None

    def _flush_scalars(self) -> None:
        if self._ema_step_cos_t is not None or self._log2_damp_t is not None:
            ts = [t.double() for t in (self._ema_step_cos_t, self._log2_damp_t) if t is not None]; vs, i = (torch.stack(ts).tolist() if len(ts) > 1 else [float(ts[0].item())]), 0
            if self._ema_step_cos_t is not None: self._ema_step_cos_val, self._ema_step_cos_t, i = float(vs[i]), None, i + 1
            if self._log2_damp_t is not None: self._log2_damp_val, self._log2_damp_t = float(vs[i]), None

    def _flush_fast_cache(self) -> None:
        if self._fast_cache is not None:
            for p, rv, iv in zip(self._fast_cache["params"], self._fast_cache["log2_ref"].detach().cpu().tolist(), self._fast_cache["log2_init"].detach().cpu().tolist()):
                self.state[p]["log2_ref"], self.state[p]["log2_init"] = float(rv), bool(iv)

    @property
    def ema_step_cos(self) -> float: self._flush_scalars(); return self._ema_step_cos_val
    @property
    def log2_damp(self) -> float: self._flush_scalars(); return self._log2_damp_val

    def _param_metrics(self, p: torch.Tensor) -> Tuple[float, float, int, int, int]:
        if (out := self._metrics_cache.get(id(p))) is not None: return out
        if p.ndim == 4:
            c0, c1, kh, kw = (int(s) for s in p.shape); is_t, gr = self._conv_layouts.get(id(p), (False, 1))
            oc, ic = (c1 * gr, c0) if is_t else (c0, c1 * gr); fac = math.sqrt(float(oc) / float(max(1, ic)))
            if kh != 1 or kw != 1: fac *= math.sqrt(float(gr) / float(max(1, (kh * kw) // 4 if is_t else kh * kw))) * (float(max(1, min(oc // max(1, gr), ic // max(1, gr)))) ** 0.25)
            out = (fac, 0.5 * math.log2(float(max(1, min(oc, ic) * kh * kw))), 1, 1, 1)
        elif p.ndim >= 2:
            rd, cd = int(p.shape[0]), int(p.numel() // max(1, int(p.shape[0])))
            out = (math.sqrt(max(1.0, float(rd) / float(max(1, cd)))), 0.5 * math.log2(float(max(1, min(rd, cd)))), rd, cd, min(rd, cd))
        else: out = (1.0, 0.0, 1, 1, 1)
        self._metrics_cache[id(p)] = out; return out

    def set_sampled_rows(self, param: torch.Tensor, rows: torch.Tensor) -> None:
        rows = (rows if isinstance(rows, torch.Tensor) else torch.as_tensor(rows)).reshape(-1).to(dtype=torch.long, device=param.device)
        if (rec := self._active_record()) is not None: rec._active_prior_table, rec._active_batch_rows = param, rows.detach()
        if rows.device.type == "cpu" and rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(param)).any())): raise ValueError("sampled rows out of bounds for prior table")
        self._sampled_rows[param] = rows if param not in self._sampled_rows else torch.cat((self._sampled_rows[param], rows)); self._sampled_rows_unique.pop(param, None)

    def sampled_rows_for(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        return None if param not in self._sampled_rows else self._sampled_rows_unique.setdefault(param, torch.unique(self._sampled_rows[param], sorted=True)).detach().clone()

    def clear_sampled_rows(self) -> None:
        self._sampled_rows.clear(); self._sampled_rows_unique.clear(); self._row_scales.clear()
        if (rec := self._active_record()) is not None: rec._active_prior_table = rec._active_batch_rows = None

    def set_row_scales(self, param: torch.Tensor, scales: Optional[torch.Tensor]) -> None:
        if scales is None: self._row_scales.pop(param, None)
        else: self._row_scales[param] = scales.detach().to(device=param.device, dtype=torch.float32)

    def bind_axiom_modulation(self, *, g_scale: float | torch.Tensor = 1.0, wakeup_boost: float | torch.Tensor = 1.0, prior_scale: float = 1.0) -> None:
        self._mod_g_scale = g_scale.detach().float().reshape(()) if isinstance(g_scale, torch.Tensor) else float(g_scale)
        self._mod_wakeup_boost = wakeup_boost.detach().float().reshape(()) if isinstance(wakeup_boost, torch.Tensor) else float(wakeup_boost)
        self._mod_prior_scale = float(prior_scale)
    bind_v4_modulation = bind_axiom_modulation

    def bind_protected_from_grads(self) -> None:
        if self.direction_blend:
            if self._protected is not None or self._protected_single: raise ValueError("Axiom direction_blend losses already pending")
            self._protected_single = True

    def bind_protected_losses(self, losses: Sequence[torch.Tensor], *, protected_evaluator: Any = None) -> None:
        del protected_evaluator
        if not self.direction_blend: return
        if self._protected is not None or self._protected_single: raise ValueError("Axiom direction_blend losses already pending")
        lt = tuple(losses)
        if not 1 <= len(lt) <= 2 or any(not isinstance(x, torch.Tensor) or x.ndim != 0 or not x.requires_grad for x in lt): raise ValueError("one or two active scalar protected losses required")
        if len(lt) == 1: self._protected_single = True; return
        params = self._parameters(); trainable, saved, normals = [p for p in params if p.requires_grad], [None if p.grad is None else p.grad.detach().clone() for p in params], []
        try:
            for lk in lt:
                for p in params: p.grad = None
                if trainable: lk.backward(retain_graph=True, inputs=trainable)
                pcs: List[torch.Tensor] = []
                for grp in self.param_groups:
                    is_ps = grp["algorithm"] == "axiom_prior" and grp.get("sampled_rows_required", True)
                    for p in grp["params"]:
                        if p.grad is None: pcs.append(p.new_zeros(p.numel()))
                        elif is_ps:
                            if (rows := self.sampled_rows_for(p)) is None: raise ValueError("direction_blend requires explicit sampled rows")
                            m = torch.zeros_like(p.grad); m[rows] = p.grad.detach()[rows]; pcs.append(m.reshape(-1))
                        else: pcs.append(p.grad.detach().reshape(-1))
                normals.append(torch.cat(pcs))
        finally:
            for p, sg in zip(params, saved): p.grad = sg
        if not bool(torch.isfinite(stacked := torch.stack(normals)).all()): raise ValueError("nonfinite direction_blend protected gradient")
        self._protected = stacked

    def abort_pending(self) -> None:
        self.clear_sampled_rows(); self._mod_g_scale = self._mod_wakeup_boost = self._mod_prior_scale = 1.0
        self._origin_norm_t, self._protected, self._protected_single = None, None, False

    def _update_step_cosine(self, net_grads: List[torch.Tensor], dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rho_p = self._ema_step_cos_t if (self._ema_step_cos_t is not None and self._ema_step_cos_t.device == dev) else torch.full((), self._ema_step_cos_val, dtype=torch.float64, device=dev)
        damp_p = self._log2_damp_t if (self._log2_damp_t is not None and self._log2_damp_t.device == dev) else torch.full((), self._log2_damp_val, dtype=torch.float64, device=dev)
        rho_n = rho_p
        if net_grads and (fls := [_flat_nocopy(g.detach()) for g in net_grads if g.numel() > 0]):
            rows = [fl[::max(1, int(fl.numel()) // _SKETCH_DIM)][:_SKETCH_DIM] for fl in fls if fl.numel() >= _SKETCH_DIM] or [torch.cat((fl.float(), fl.new_zeros(_SKETCH_DIM - int(fl.numel()), dtype=torch.float32)), dim=0) for fl in fls]
            bm = torch.stack(rows, dim=0).float(); bm = torch.where(torch.isfinite(bm).all(dim=1, keepdim=True), bm, torch.zeros_like(bm))
            bm = torch.where(bm.abs().amax(dim=1, keepdim=True) < 1e-30, bm * 18446744073709551616.0, bm)
            bmax = bm.abs().amax(dim=1, keepdim=True); bu = bm / bmax.clamp_min(1e-37)
            sk_acc = torch.where(bmax > 0.0, bu / bu.norm(dim=1, keepdim=True).clamp_min(1e-30), torch.zeros_like(bu)).sum(dim=0)
            sk_n = sk_acc.norm(); sk_ok = torch.isfinite(sk_n) & (sk_n > 0.0); sk_now = torch.where(sk_ok, sk_acc / sk_n.clamp_min(1e-30), torch.zeros_like(sk_acc))
            if (prev_sk := self._prev_step_sketch) is not None:
                prev_sk = prev_sk.to(device=dev, dtype=torch.float32) if prev_sk.device != dev else prev_sk
                rho_n = torch.where(sk_ok, 0.90 * rho_p + 0.10 * (sk_now * prev_sk).sum().double().clamp(-1.0, 1.0), rho_p)
            self._prev_step_sketch = torch.where(sk_ok, sk_now, prev_sk if prev_sk is not None else sk_now)
        if self._ss_coef is None or self._ss_coef[0].device != dev:
            self._ss_coef = (torch.tensor([-1.0 / 0.35, 1.0 / 0.35, -1.0 / 0.45], dtype=torch.float64, device=dev), torch.tensor([-0.45 / 0.35, 0.15 / 0.35, -0.15 / 0.45], dtype=torch.float64, device=dev))
        u3 = (rho_n * self._ss_coef[0] + self._ss_coef[1]).clamp_(0.0, 1.0); g3 = u3 * u3 * (3.0 - 2.0 * u3)
        log2_d = (damp_p - 0.08 * g3[0] + 0.12 * g3[1]).clamp(-10.0, 0.0); self._ema_step_cos_t, self._log2_damp_t = rho_n, log2_d
        return rho_p, rho_n, torch.exp2(log2_d), float(self.smoothing) * g3[2]

    def _two_obj_pyth(self) -> Dict[int, float]:
        if self._protected is None or int(self._protected.shape[0]) != 2: return {}
        n0, n1, sr, st, rp, off = self._protected[0], self._protected[1], [], [], {}, 0
        for grp in self.param_groups:
            pids = rp.setdefault(grp["role"], [])
            for p in grp["params"]:
                sz = p.numel(); g0, g1, off = n0[off : off + sz].double(), n1[off : off + sz].double(), off + sz; pids.append(id(p))
                if p.grad is not None and sz > 0: sr.append(grp["role"]); st.extend(((g0 * g0).sum(), (g1 * g1).sum(), (g0 * g1).sum()))
        acc = {r: [0.0, 0.0, 0.0] for r in rp}
        if st:
            vs = torch.stack(st).tolist()
            for i, r in enumerate(sr):
                a2, b2, dt = float(vs[3 * i]), float(vs[3 * i + 1]), float(vs[3 * i + 2])
                if a2 > 1e-24 and b2 > 1e-24 and math.isfinite(a2) and math.isfinite(b2) and math.isfinite(dt): acc[r][0], acc[r][1], acc[r][2] = acc[r][0] + a2, acc[r][1] + b2, acc[r][2] + dt
        return {pid: (min(2.0, max(1.0, (math.sqrt(a2) + math.sqrt(b2)) / math.sqrt(max(a2 + b2 + 2.0 * dt, 1e-36)))) if (a2 > 1e-24 and b2 > 1e-24) else 1.0)
                for r, pids in rp.items() for a2, b2, dt in (acc[r],) for pid in pids}

    def _step_prior(self, p: torch.Tensor, grad: torch.Tensor, grp: Dict[str, Any], *, base_lr: float, b1: float, s_rel: torch.Tensor, rho_p: float) -> None:
        if base_lr <= 0.0: return
        st, r_sc, req = self.state[p], self._row_scales.get(p), bool(grp.get("sampled_rows_required", True))
        if req and r_sc is None:
            if (raw := self._sampled_rows[p]).numel() == 0: return
            ridx, _ = torch.sort(raw); is_f = torch.ones(int(ridx.shape[0]), device=p.device, dtype=torch.bool)
            if ridx.shape[0] > 1: is_f[1:] = ridx[1:] != ridx[:-1]
            rmask = is_f.unsqueeze(1)
        else:
            ridx = self.sampled_rows_for(p) if req else torch.arange(len(p), device=p.device, dtype=torch.long)
            if ridx is None or ridx.numel() == 0: return
            rmask = torch.ones((int(ridx.shape[0]), 1), device=p.device, dtype=torch.bool)
        gr = grad[ridx].float(); fin = torch.isfinite(gr).all(dim=-1, keepdim=True); dr = torch.where(fin, gr, torch.zeros_like(gr))
        if b1 > 0.0:
            mp = st.setdefault("m_prior", torch.zeros_like(p, dtype=torch.float32)); mpv = mp[ridx]
            dr = torch.where(fin, mpv.mul(b1).add(dr, alpha=1.0 - b1), mpv); mp.index_add_(0, ridx, (dr - mpv) * rmask.float())
        u_row, l2, ok0 = _me_unit_rows(dr); ok = rmask & ok0; nok = ok.float().sum()
        n_ok, mean_l2 = torch.stack((nok.double(), ((l2 * ok.float()).sum() / nok.clamp_min(1.0)).double())).tolist()
        ref_n = _update_l2_ref(st, float(mean_l2), float(rho_p)) if n_ok > 0.0 else float(st.get("log2_ref", 0.0))
        ur = torch.where(ok, u_row * torch.rsqrt(1.0 + (s_rel.float() * torch.exp2((ref_n - l2).clamp(-30.0, 30.0))).square()), torch.zeros_like(u_row))
        p.index_add_(0, ridx, (ur * r_sc.unsqueeze(1) if (r_sc is not None and len(r_sc) == len(ridx)) else ur).to(p.dtype), alpha=-base_lr)

    def _run_foreach_fast(self, items: List[Tuple[torch.Tensor, torch.Tensor, float, float, float, bool]], *,
                          rho_p: torch.Tensor, c_osc: torch.Tensor, s_rel: torch.Tensor,
                          g_mult: float | torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, float, float, float, bool]]:
        params, grads = [it[0] for it in items], [it[1] for it in items]
        dev, key, lr_key, c = params[0].device, tuple(id(p) for p in params), tuple(it[2] for it in items), self._fast_cache
        if c is None or c["key"] != key or c["device"] != dev:
            self._flush_fast_cache()
            self._fast_cache = c = {"key": key, "device": dev, "params": params, "lr_key": lr_key,
                                    "lrs": torch.tensor(lr_key, dtype=torch.float64, device=dev), "factors": torch.tensor([it[3] for it in items], dtype=torch.float64, device=dev),
                                    "half_l2": torch.tensor([it[4] for it in items], dtype=torch.float64, device=dev), "is_gen": torch.tensor([it[5] for it in items], dtype=torch.bool, device=dev),
                                    "log2_ref": torch.tensor([float(self.state[p].get("log2_ref", 0.0)) for p in params], dtype=torch.float64, device=dev),
                                    "log2_init": torch.tensor([bool(self.state[p].get("log2_init", False)) for p in params], dtype=torch.bool, device=dev)}
        elif c["lr_key"] != lr_key: c["lr_key"], c["lrs"] = lr_key, torch.tensor(lr_key, dtype=torch.float64, device=dev)
        lrs = torch.where(c["is_gen"], c["lrs"] * (g_mult.to(dev, dtype=torch.float64) if isinstance(g_mult, torch.Tensor) else float(g_mult)), c["lrs"]) if (isinstance(g_mult, torch.Tensor) or g_mult != 1.0) else c["lrs"]
        norms = torch.stack(torch._foreach_norm(grads)).double(); ok = torch.isfinite(norms) & (norms > 1e-17) & (norms < 1e17)
        sn = torch.where(ok, norms, torch.ones_like(norms)); lt, rp, ip = torch.log2(sn) - c["half_l2"], c["log2_ref"], c["log2_init"]
        dl = (lt - rp).clamp(-1.0, 1.0); rn = torch.where(ok, torch.where(ip, torch.where((rho_p >= 0.0) | (dl > 0.0), rp + 0.05 * dl, rp), lt), rp)
        c["log2_ref"].copy_(rn); c["log2_init"].copy_(ip | ok)
        sc = torch.where(ok, -(lrs * c["factors"] * c_osc * torch.rsqrt(1.0 + (s_rel * torch.exp2((rn - lt).clamp(-30.0, 30.0))).square())) / sn, torch.zeros_like(norms)).float()
        if not bool(ok.all().item()):
            ml = ok.tolist()
            if (keep := [i for i, b in enumerate(ml) if b]):
                torch._foreach_addcmul_([params[i] for i in keep], [grads[i] for i in keep], [sc[i] for i in keep]); self.fast_path_steps += 1
            return [it for it, b in zip(items, ml) if not b]
        torch._foreach_addcmul_(params, grads, list(sc.unbind(0))); self.fast_path_steps += 1; return []

    def _apply_verified_trust(self, rec: Any, tracked: List[Tuple[torch.Tensor, torch.Tensor]], *,
                              step_scale: float, s_therm: float, pre_gain: float | torch.Tensor) -> Tuple[float, bool]:
        cache, rec._g_probe_cache = rec._g_probe_cache, None
        if cache is None: return 1.0, False
        g_mod, pa, pk, y_pre = cache; deltas, yp = [(p, o, (p - o).detach()) for p, o in tracked], _eval_probe_forward(g_mod, pa, pk, rec)
        if yp is None or yp.shape != y_pre.shape: return 1.0, False
        if hasattr(rec, "take_pending_ema_tensors"):
            ek, et = rec.take_pending_ema_tensors()
            if et: rec.apply_flushed_ema_values(ek, torch.stack([t.double() for t in et]).tolist() if all(t.device == et[0].device for t in et) else [float(t.item()) for t in et])
        coord_sc = max(1e-30, float(getattr(rec, "ema_data_scale", 1.0)) / math.sqrt(float(max(1, int(y_pre[0].numel())))))
        d_max, bp = (0.045 + 0.16 * min(1.5, float(getattr(rec, "ema_push_res", 0.0)))) * step_scale, int(y_pre.shape[0])
        _rel_p95 = lambda y: float("inf") if (y is None or not bool(torch.isfinite(y).all())) else float(torch.quantile((y.double() - y_pre.double()).reshape(bp, -1).square().mean(-1).sqrt(), 0.95).item()) / coord_sc
        d_obs, alpha = _rel_p95(yp), 1.0
        for _ in range(4):
            if d_obs <= d_max: break
            self.trust_backtracks_total += 1
            alpha = max(1e-6, min(0.5 * alpha, 0.9 * alpha * (d_max / d_obs)) if (math.isfinite(d_obs) and d_obs > 1e-30) else 0.1 * alpha)
            for p, o, d in deltas: p.copy_(o + d * alpha)
            d_obs = _rel_p95(_eval_probe_forward(g_mod, pa, pk, rec))
        if not math.isfinite(d_obs) or d_obs > d_max:
            alpha = max(1e-6, 0.5 * alpha)
            for p, o, d in deltas: p.copy_(o + d * alpha)
        pg_f = float(pre_gain.item()) if isinstance(pre_gain, torch.Tensor) else float(pre_gain)
        if alpha < 1.0: self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * max(0.05, min(1.0, pg_f * alpha)))
        elif s_therm >= 0.5 and math.isfinite(d_obs) and d_obs >= 0.012 and (fr := (d_max / max(d_obs, 1e-12)) * pg_f) > 1.5 and min(1.0, fr) > self._ema_func_gain:
            self._ema_func_gain = float(0.999 * self._ema_func_gain + 0.001 * max(0.05, min(1.0, fr)))
        rec.g_func_gain = float(self._ema_func_gain); return alpha, True

    def _apply_direction_blend(self, all_p: List[torch.Tensor], origs: List[torch.Tensor]) -> None:
        normals, disp = self._protected.double(), torch.cat([(p - o).reshape(-1) for p, o in zip(all_p, origs)])  # type: ignore[union-attr]
        d64, before, stats, blend, proj = disp.double(), normals @ disp.double(), self.constraint_geometry_stats, self.direction_blend_stats, False
        if bool((before > 0.0).any()):
            blend["conflict_steps"] += 1; an = normals.norm(dim=1); au = normals[an > 0] / an[an > 0, None]
            poss = int(au.shape[0]) == 0 or float(au.mean(0).norm().item()) > 1e-12
            direc, off = (zero_pinv_project_nonascent(disp, normals) if poss else torch.zeros_like(disp)), 0
            blend["blended_steps" if poss else "pareto_stalls"] += 1
            for p, o in zip(all_p, origs): p.copy_(o + direc[off : off + p.numel()].view_as(p) if poss else o); off += p.numel()
            app = torch.cat([(p - o).reshape(-1) for p, o in zip(all_p, origs)]).double() if poss else torch.zeros_like(d64)
            proj, dn = not torch.equal(app, d64), float(d64.norm().item()); ratio = float(app.norm().item()) / dn if dn > 0.0 else 0.0
            blend["retained_norm_ratio_sum"], blend["max_retained_norm_ratio"], after_max = blend["retained_norm_ratio_sum"] + ratio, max(blend["max_retained_norm_ratio"], ratio), float((normals @ app).max().item())
        else: after_max = float(before.max().item())
        stats["steps"], stats["projected_steps"] = stats["steps"] + 1, stats["projected_steps"] + int(proj)
        stats["max_derivative_before"], stats["max_derivative_after"] = max(stats["max_derivative_before"], float(before.max().item())), max(stats["max_derivative_after"], after_max)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        if self.direction_blend and (closure is not None or (self._protected is None and not self._protected_single)):
            self.abort_pending(); raise ValueError("direction_blend step requires explicit protected-loss backward")
        net_grads, ref_dev = [], None
        for grp in self.param_groups:
            is_pr = grp["algorithm"] == "axiom_prior"
            for p in grp["params"]:
                if (g := p.grad) is None: continue
                if g.is_sparse or g.is_complex(): raise ValueError("AxiomOptimizer requires dense real gradients")
                if is_pr and grp.get("sampled_rows_required", True) and p not in self._sampled_rows: raise ValueError("row-normalized prior updates require actual sampled rows")
                ref_dev = p.device if ref_dev is None else ref_dev
                if not is_pr: net_grads.append(g)
        all_p = self._parameters() if (self.direction_blend and self._protected is not None) else None
        origs = [p.detach().clone() for p in all_p] if all_p is not None else None
        try:
            rec, is_crit = self._active_record(), self.critic is not None
            step_scale, s_therm = (float(getattr(rec, "step_scale", 1.0)), float(getattr(rec, "thermal_scale", 1.0))) if rec is not None else (1.0, 1.0)
            is_simp, is_cf = (bool(getattr(rec, "_simplicial_step", False)), bool(getattr(rec, "cond_initialized", False))) if rec is not None else (False, False)
            if ref_dev is None:
                self.last_step_gain, self.last_probe_consumed = 1.0, False
                if rec is not None and not is_crit: rec._g_probe_cache = None
                return loss
            pyth = self._two_obj_pyth() if (is_cf and self.direction_blend and self._protected is not None and int(self._protected.shape[0]) == 2) else {}
            ema_fg = float(self._ema_func_gain); pg_b = 1.0 if ema_fg >= 0.75 else ema_fg
            pre_g = (pg_b * self._mod_wakeup_boost.to(ref_dev)).clamp_max(1.0) if isinstance(self._mod_wakeup_boost, torch.Tensor) else min(1.0, pg_b * float(self._mod_wakeup_boost))
            g_sc = self._mod_g_scale.to(ref_dev) if isinstance(self._mod_g_scale, torch.Tensor) else self._mod_g_scale
            g_mult: float | torch.Tensor = g_sc * pre_g
            gen_probe = ((not is_crit) and rec is not None and getattr(rec, "_g_probe_cache", None) is not None and getattr(rec, "_enable_secant_probe", True)
                         and not is_simp and not is_cf and (self._protected is None or int(self._protected.shape[0]) < 2))
            fast_it, d2_it, slow_it, tracked = [], [], [], []
            for grp in self.param_groups:
                is_pr, is_gr, b1 = grp["algorithm"] == "axiom_prior", grp["role"] not in ("critic", "prior", "table"), float(grp["betas"][0])
                blr = float(grp["lr"]) * step_scale * (self._mod_prior_scale if is_pr else 1.0)
                for p in grp["params"]:
                    if (g := p.grad) is None: continue
                    self.state[p]["step"] = int(self.state[p].get("step", 0)) + 1; lr_p = blr * pyth.get(id(p), 1.0)
                    if is_pr: slow_it.append((p, g, lr_p, 0.0, 0.0, b1, True, grp))
                    elif lr_p > 0.0:
                        if gen_probe and (not self._gen_param_ids or id(p) in self._gen_param_ids): tracked.append((p, p.detach().clone()))
                        fac, hl2, rd, cd, md = self._param_metrics(p)
                        if p.ndim == 2 and md >= 2: d2_it.append((p, g, lr_p, fac, hl2, rd, cd, b1, is_gr))
                        elif (not gen_probe) and b1 == 0.0 and p.dtype == torch.float32 and g.dtype == torch.float32: fast_it.append((p, g, lr_p, fac, hl2, is_gr))
                        else: slow_it.append((p, g, lr_p, fac, hl2, b1, is_gr, None))
            rho_p, rho_n, c_osc, s_rel = self._update_step_cosine(net_grads, ref_dev)
            if fast_it: slow_it.extend((p, g, lr_p, fac, hl2, 0.0, is_gr, None) for p, g, lr_p, fac, hl2, is_gr in self._run_foreach_fast(fast_it, rho_p=rho_p, c_osc=c_osc, s_rel=s_rel, g_mult=g_mult))
            if slow_it or d2_it:
                s_vec: List[torch.Tensor] = [rho_p.double(), c_osc.double(), s_rel.double()]
                if isinstance(g_mult, torch.Tensor): s_vec.append(g_mult.to(ref_dev, dtype=torch.float64).reshape(()))
                slow_norm, d2_norm = [], []
                for x in slow_it:
                    if x[7] is None:
                        gu, l2n, ok = mantissa_exponent_normalize(x[1]); slow_norm.append((x, gu))
                        s_vec.extend((ok.to(ref_dev, dtype=torch.float64), l2n.to(ref_dev, dtype=torch.float64)))
                for x in d2_it:
                    gu, l2n, ok = mantissa_exponent_normalize(x[1]); d2_norm.append((x, gu))
                    s_vec.extend((ok.to(ref_dev, dtype=torch.float64), l2n.to(ref_dev, dtype=torch.float64)))
                sv = torch.stack(s_vec).tolist()
                rp_f, cosc_f, srel_f = float(sv[0]), float(sv[1]), float(sv[2])
                idx_s, gm_f = (4, float(sv[3])) if isinstance(g_mult, torch.Tensor) else (3, float(g_mult))
                if slow_norm: self._flush_fast_cache(); self._fast_cache = None
                for x in slow_it:
                    if x[7] is not None: self._step_prior(x[0], x[1], x[7], base_lr=x[2], b1=x[5], s_rel=s_rel, rho_p=rp_f)
                for (p, _, lr_p, fac, hl2, b1, is_gr, _), gu in slow_norm:
                    ok_v, l2_v = sv[idx_s] > 0.5, float(sv[idx_s + 1]); idx_s += 2
                    if not ok_v: continue
                    gu, lt = _apply_mom(self.state[p], p, gu, b1), l2_v - hl2; rn = _update_l2_ref(self.state[p], lt, rp_f)
                    p.add_((gu * (-(((lr_p * gm_f) if is_gr else lr_p) * fac) * cosc_f / math.sqrt(1.0 + (srel_f * (2.0 ** max(-30.0, min(30.0, rn - lt)))) ** 2))).to(p.dtype))
                ns_b: Dict[Tuple[int, int], List[Tuple[torch.Tensor, torch.Tensor, bool, float, float]]] = {}
                for (p, _, lr_p, fac, hl2, rd, cd, b1, is_gr), gu in d2_norm:
                    ok_v, l2_v = sv[idx_s] > 0.5, float(sv[idx_s + 1]); idx_s += 2
                    if not ok_v: continue
                    st, lt = self.state[p], l2_v - hl2; gu, rn, nt = _apply_mom(st, p, gu, b1), _update_l2_ref(st, lt, rp_f), rd > cd
                    mc = gu.reshape(rd, cd).T if nt else gu.reshape(rd, cd)
                    rate, rs = ((lr_p * gm_f) if is_gr else lr_p), srel_f * (2.0 ** max(-30.0, min(30.0, rn - lt)))
                    ns_b.setdefault((int(mc.shape[0]), int(mc.shape[1])), []).append((p, mc, nt, rate * fac * cosc_f, rs))
                for bkt in ns_b.values():
                    self.ns_2d_activations += len(bkt); mats = torch.stack([b[1] for b in bkt], dim=0)
                    orths = fp32_tikhonov_ns_batched(mats, steps=5, rel_smooth=torch.tensor([b[4] for b in bkt], dtype=torch.float32, device=mats.device)).float()
                    for i, (p_i, _, nt, mult, _) in enumerate(bkt): p_i.add_(((orths[i].T if nt else orths[i]) * (-mult)).reshape_as(p_i).to(p_i.dtype))
            alpha_g, p_ran = self._apply_verified_trust(rec, tracked, step_scale=step_scale, s_therm=s_therm, pre_gain=pre_g) if (gen_probe and tracked and rec is not None) else (1.0, False)
            if not (gen_probe and tracked) and rec is not None and not is_crit: rec._g_probe_cache = None
            if self.direction_blend and origs is not None and self._protected is not None and all_p is not None: self._apply_direction_blend(all_p, origs)
            self.last_step_gain, self.last_probe_consumed, self.probe_consumptions = float(alpha_g), bool(p_ran), self.probe_consumptions + int(p_ran)
        finally: self.abort_pending()
        if self.record is not None: self.record.record_step(self)
        return loss

    def __getstate__(self) -> Dict[str, Any]:
        self._flush_scalars(); self._flush_fast_cache(); return {**self.__dict__, "_fast_cache": None, "_metrics_cache": {}, "_ss_coef": None}

    def state_dict(self) -> Dict[str, Any]:
        self._flush_scalars(); self._flush_fast_cache(); res = Optimizer.state_dict(self)
        meta: Dict[str, Any] = {"schema": 1, "family": self.family, "last_step_gain": float(self.last_step_gain), "ema_func_gain": float(self._ema_func_gain),
                                "ema_step_cos": float(self._ema_step_cos_val), "log2_damp": float(self._log2_damp_val), "direction_blend": bool(self.direction_blend),
                                "prev_step_sketch": None if self._prev_step_sketch is None else [float(x) for x in self._prev_step_sketch.detach().cpu().tolist()],
                                "sampled_rows": [self.sampled_rows_for(g["params"][0]) if g["algorithm"] == "axiom_prior" else None for g in self.param_groups]}
        if self.direction_blend: meta["constraint_geometry"] = dict(stats=deepcopy(self.constraint_geometry_stats), blend_stats=deepcopy(self.direction_blend_stats), pending=None if self._protected is None else self._protected.detach().cpu().clone())
        res["axiom"] = meta
        if self.record is not None: res["regularizer"] = {"optimizer_family": self.family, "record": self.record.state_dict(), "ema": None, "guard": None}
        return res

    def validate_state_dict(self, saved: Any) -> None:
        if not isinstance(saved, dict) or set(saved) != ({"state", "param_groups", "axiom"} | ({"regularizer"} if self.record is not None else set())): raise ValueError("invalid AxiomOptimizer checkpoint schema")
        meta, exp_m = saved["axiom"], {"schema", "family", "last_step_gain", "ema_func_gain", "ema_step_cos", "log2_damp", "prev_step_sketch", "direction_blend", "sampled_rows"} | ({"constraint_geometry"} if self.direction_blend else set())
        if (not isinstance(meta, dict) or set(meta) != exp_m or type(meta["schema"]) is not int or meta["schema"] != 1 or meta["family"] != self.family
                or type(meta["direction_blend"]) is not bool or meta["direction_blend"] != self.direction_blend or not isinstance(meta["sampled_rows"], list) or len(meta["sampled_rows"]) != len(self.param_groups)):
            raise ValueError("incompatible AxiomOptimizer metadata")
        for k, lo, hi, slo in (("last_step_gain", 0.0, None, True), ("ema_func_gain", 0.0, 1.0, True), ("ema_step_cos", -1.0, 1.0, False), ("log2_damp", -10.0, 0.0, False)):
            if isinstance(v := meta[k], bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (float(v) <= lo if slo else float(v) < lo) or (hi is not None and float(v) > hi): raise ValueError(f"invalid AxiomOptimizer {k}")
        if (sk := meta["prev_step_sketch"]) is not None and (not isinstance(sk, list) or len(sk) != _SKETCH_DIM or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in sk)): raise ValueError("invalid AxiomOptimizer prev_step_sketch")
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            if (not isinstance(cg, dict) or set(cg) != {"stats", "blend_stats", "pending"} or not isinstance(cg["stats"], dict) or set(cg["stats"]) != set(self.constraint_geometry_stats)
                    or not isinstance(cg["blend_stats"], dict) or set(cg["blend_stats"]) != set(self.direction_blend_stats) or (cg["pending"] is not None and (not isinstance(cg["pending"], torch.Tensor) or not bool(torch.isfinite(cg["pending"]).all())))):
                raise ValueError("invalid AxiomOptimizer direction_blend checkpoint")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups) or not isinstance(saved["state"], dict): raise ValueError("AxiomOptimizer group/state mismatch")
        for cur, sav in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav, dict) or len(sav.get("params", ())) != len(cur["params"]) or sav.get("role") != cur["role"] or sav.get("algorithm") != cur["algorithm"]: raise ValueError("AxiomOptimizer group mismatch")
        for sv in saved["state"].values():
            if (not isinstance(sv, dict) or type(sv.get("step", 0)) is not int or sv.get("step", 0) < 0
                    or ("log2_ref" in sv and (isinstance(sv["log2_ref"], bool) or not isinstance(sv["log2_ref"], (int, float)) or not math.isfinite(sv["log2_ref"])))
                    or ("log2_init" in sv and type(sv["log2_init"]) is not bool)
                    or any(tk in sv and (not isinstance(sv[tk], torch.Tensor) or not bool(torch.isfinite(sv[tk]).all())) for tk in ("v_pow", "m_unit", "m_scale", "m_prior"))):
                raise ValueError("invalid AxiomOptimizer parameter state")
        if self.record is not None:
            reg = saved["regularizer"]
            if not isinstance(reg, dict) or set(reg) != {"optimizer_family", "record", "ema", "guard"} or reg["optimizer_family"] != self.family or reg["ema"] is not None or reg["guard"] is not None: raise ValueError("invalid AxiomOptimizer regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict); meta = state_dict["axiom"]
        Optimizer.load_state_dict(self, {k: v for k, v in state_dict.items() if k not in ("axiom", "regularizer")})
        self.clear_sampled_rows(); self._fast_cache = self._ss_coef = None; self._metrics_cache.clear()
        self.last_step_gain, self._ema_func_gain, self._ema_step_cos_val, self._log2_damp_val = float(meta["last_step_gain"]), float(meta["ema_func_gain"]), float(meta["ema_step_cos"]), float(meta["log2_damp"])
        self._ema_step_cos_t = self._log2_damp_t = None
        ref_dev = self.param_groups[0]["params"][0].device if (self.param_groups and self.param_groups[0]["params"]) else torch.device("cpu")
        self._prev_step_sketch = None if meta["prev_step_sketch"] is None else torch.tensor(meta["prev_step_sketch"], dtype=torch.float32, device=ref_dev)
        self.ns_2d_activations = self.fast_path_steps = self.trust_backtracks_total = self.probe_consumptions = 0; self.last_probe_consumed = False
        for grp, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and grp["algorithm"] == "axiom_prior": self.set_sampled_rows(grp["params"][0], rows.to(grp["params"][0].device))
        if self.direction_blend:
            cg = meta["constraint_geometry"]; self.constraint_geometry_stats, self.direction_blend_stats = deepcopy(cg["stats"]), deepcopy(cg["blend_stats"])
            self._protected = None if cg["pending"] is None else cg["pending"].detach().to(ref_dev).clone()
        else: self._protected = None
        if self.record is not None: self.record.load_state_dict(state_dict["regularizer"]["record"]); self.record.restore_device_state(ref_dev)

__all__ = ["AxiomOptimizer", "collect_conv_layouts", "fp32_tikhonov_ns_2d", "fp32_tikhonov_ns_batched", "mantissa_exponent_normalize", "zero_pinv_project_nonascent"]
