"""Axiom prior regularizer & pullback."""

from math import log, tau

import torch
from torch import Tensor, bmm, cat, eye, maximum, nn, where
import torch.nn.functional as F

from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .atlas import (
    MAX_FULL_COV_DIM,
    PrototypeAtlas,
    SCHEMA_VERSION,
    _ATLAS_T,
    _B,
    _E,
    _F32,
    _INF,
    _VF,
    _chk_hdr,
    _chk_s,
    _chk_t,
    _cp,
    _dg,
    _flat,
    _norm,
    _req,
    _to,
    coalesce_pythagorean_ltv,
    cov_from_scalar,
    cov_trace_mean,
    validate_atlas_state,
)
from .math_ops import (
    allocate_deficit_quotas,
    born_infeld_cost as _bi,
    log_sinkhorn,
    safe_cdist as _cd,
    smoothstep01 as _ss,
    weyl_rademacher,
)
from .state import get_module_record, set_active_prior_rows

_EMA_KEYS = ("res_ema", "iso_ema", "iso_err_ema")
_ROW_SPECS = (("v_real_ema", _F32), ("v_tang_ema", _F32), ("row_m", _F32), ("row_v", _F32), ("row_init", _B))
_PRIOR_KEYS = frozenset(
    {"schema", "max_prototypes", "atlas", "rows_initialized", *_EMA_KEYS, *(k for k, _ in _ROW_SPECS)}
)
_PRIMES = tuple((c**0.5) % 1.0 for c in range(2, 320) if all(c % p for p in range(2, int(c**0.5) + 1)))


def _kernel_spec(prior):
    b, g = unwrap_prior(prior), getattr
    if g(b, "code_law", "table") != "table":
        return b, 0.0, False
    if not g(prior, "transforms_codes", False):
        return (
            (b, b._cached_sigma_float() if g(b, "_noise_enabled", False) else 0.0, True)
            if isinstance(b, MoGParticlePrior)
            else (b, 0.0, hasattr(b, "z"))
        )
    ms = g(prior, "manipulators", ())
    if len(ms) == 1 and g(ms[0], "normal_noise", False) and float(g(ms[0], "kappa_radial", 1.0)) == 1.0:
        return b, float(ms[0].sigma), True
    return b, 0.0, False


def has_positive_prior_sigma(prior):
    return _kernel_spec(prior)[1] > 0.0


@torch.no_grad()
def _chol(cov):
    x = cov.float()
    s = 0.5 * (x + x.mT)
    v, e = _dg(s).mean(-1, keepdim=True).clamp_min(_VF).unsqueeze(-1), eye(s.shape[-1], device=s.device)
    c, i = torch.linalg.cholesky_ex(s + 1e-6 * v * e)
    return where((i != 0).view(-1, 1, 1), v.sqrt() * e, c)


@torch.no_grad()
def _bm(rows, idx, c, ch, sp):
    d, k = c.shape[1], len(c)
    m = (d + 1) // 2
    t = ((rows.double() + 0.5).unsqueeze(1) * rows.new_tensor(_PRIMES[: 2 * m], dtype=torch.double)) % 1.0
    g, a = (-2.0 * t[:, 0::2].clamp(1e-4, 1.0 - 1e-4).log()).sqrt().clamp_max(2.95), tau * t[:, 1::2]
    x = torch.stack((g * a.cos(), g * a.sin()), -1).reshape(len(rows), 2 * m)[:, :d].to(c.device, _F32)
    n = idx.bincount(minlength=k).float()
    xm = x.new_zeros((k, d)).index_add_(0, idx, x) / n.clamp_min(1.0).unsqueeze(1)
    return c[idx] + sp[idx].unsqueeze(1) * bmm(
        ch[idx], where((n[idx] >= 2).unsqueeze(1), x - xm[idx], x).unsqueeze(-1)
    ).squeeze(-1)


def _off(fn, mu, sz, ds=None):
    m, d = mu.shape
    ds = eye(d, device=mu.device, dtype=mu.dtype).unsqueeze(0).expand(m, d, d) if ds is None else ds
    k, o, u = ds.shape[1], float(sz) * ds, mu.unsqueeze(1)
    x = fn(cat(((u + o).reshape(m * k, d), (u - o).reshape(m * k, d)))).reshape(2, m, k, -1)
    y = x.float() if x.element_size() < 4 else x
    return 0.5 * (y[0] - y[1])


def _rad(fn, mu, sz, ph, ss):
    m, d = mu.shape
    return _off(
        fn,
        mu,
        sz,
        cat([weyl_rademacher((m, 8, d), phase=ph, salt=s, device=mu.device, dtype=mu.dtype) for s in ss], 1)
        * (d**-0.5),
    )


def central_pullback_cov(gen_fn, mu, sigma_z, *, phase=0.0, salt=0, return_pair=False):
    d = mu.shape[1]
    if d <= 8:
        j = _off(gen_fn, mu, sigma_z)
        s = bmm(j.mT, j)
        return (s, s) if return_pair else s
    sc, j = d / 8.0, _rad(gen_fn, mu, sigma_z, phase, (1, 2) if return_pair else (salt,))
    if return_pair:
        return tuple(bmm(x.mT, x) * sc for x in (j[:, :8], j[:, 8:]))
    return bmm(j.mT, j) * sc


def _burg_ex(ad, b, w, eps, d, dz):
    r = min(d, dz)
    s = bmm(b, b.mT) if d <= dz else (dz / d) * bmm(b.mT, b)
    e, s = eye(r, device=b.device), 0.5 * (s + s.mT)
    v = (
        _dg(s).sum(-1)
        - 2.0 * _dg(torch.linalg.cholesky(s + 1e-3 * e)).clamp_min(_E).log().sum(-1)
        - r * (1.0 - log(1.001))
    ) / r
    with torch.no_grad():
        tv = (
            bmm(ad, eps.float().unsqueeze(-1)).squeeze(-1).square().sum(-1)
            if eps is not None
            else ad.square().sum((-2, -1))
        ) / d
        fr = (s.detach() - e).square().sum((-2, -1)).sqrt() / (r**0.5)
    return 4.5 * (w * v.clamp_min(0.0)).mean(), tv, fr


def _burg_2s(ad, b, w, eps, d, dz):
    e = eye(d, device=b.device)
    s1, s2 = (bmm(x, x.mT) * (dz / 8.0) for x in (b[:, :, :8], b[:, :, 8:]))
    with torch.no_grad():
        tv = (ad.square().sum((-2, -1)) * (dz / 16.0) / d) * (
            eps.float().square().sum(-1) / dz if eps is not None else 1.0
        )
        fr = (0.5 * (s1.detach() + s2.detach()) - e).square().sum((-2, -1)).sqrt() / (d**0.5)
    return 2.25 * (w * (((s1 - e) * (s2 - e)).sum((-2, -1)) / d)).mean(), tv, fr


def whitened_burg_pullback_loss(gen_fn, mu_det, sigma_z, target_cov, inv_w, *, eps_g=None, phase=0.0):
    b, dz = mu_det.shape
    d = target_cov.shape[-1]
    ex = dz <= max(d, 16)
    m = min(b, max(32, 2048 // max(1, dz))) if ex else min(b, 256)
    mu = mu_det[:m]
    a = (_off(gen_fn, mu, sigma_z) if ex else _rad(gen_fn, mu, sigma_z, phase, (1, 2))).float().mT
    bh = torch.linalg.solve_triangular(_chol(target_cov[:m]), a, upper=False)
    return (_burg_ex if ex else _burg_2s)(
        a.detach(), bh, inv_w[:m].float(), None if eps_g is None else eps_g[:m], d, dz
    )


def validate_axiom_prior_state(state):
    _chk_hdr(state, _PRIOR_KEYS)
    m, ri = state["max_prototypes"], state["rows_initialized"]
    _req(type(m) is int and m >= 2 and type(ri) is bool)
    for k in _EMA_KEYS:
        _chk_s(state[k], 0.0)
    a = validate_atlas_state(state["atlas"])
    _req(a["max_prototypes"] == m)
    k = len(a["proto_c"]) if a["initialized"] else 0
    for nm, _ in _ROW_SPECS[:2]:
        _chk_t(state[nm], (k,), True) if k else _req(state[nm] is None)
    if not ri:
        _req(all(state[nm] is None for nm, _ in _ROW_SPECS[2:]))
    else:
        r = _chk_t(state["row_m"])
        _req(r.ndim == 2 and min(r.shape) >= 1)
        _chk_t(state["row_v"], (len(r), 1), True)
        _chk_t(state["row_init"], (len(r), 1), dt=_B)
    return state


def _eprop(k):
    def _g(self):
        if k in self._ema_t:
            self._flush()
        return self._ema[k]

    def _s(self, v):
        self._ema[k] = float(v)
        self._ema_t.pop(k, None)

    return property(_g, _s)


def _aprop(k, b=False):
    return property(lambda self: getattr(self.atlas, k), lambda self, v: setattr(self.atlas, k, bool(v) if b else v))


def _smean(idx, v, k, w=None):
    den = (v.new_zeros(k).index_add_(0, idx, w) if w is not None else idx.bincount(minlength=k).float()).clamp_min(
        _E if w is not None else 1.0
    )
    return v.new_zeros(k).index_add_(0, idx, v if w is None else w * v) / den


class AxiomPriorRegularizer(nn.Module):
    SCHEMA = SCHEMA_VERSION
    res_ema, iso_ema, iso_err_ema = (_eprop(k) for k in _EMA_KEYS)
    quantizer_initialized = _aprop("initialized", True)
    proto_w, w_fake_ema, proto_n, psi, v_push = (_aprop(k) for k in _ATLAS_T[2:])

    def __init__(
        self,
        reference=None,
        *,
        weight=1.0,
        target_std=1.0,
        eps=1e-4,
        max_prototypes=256,
        bound_device=None,
        bound_shape=None,
    ):
        super().__init__()
        _req(type(max_prototypes) is int and max_prototypes >= 2)
        self.weight, self.target_std, self.eps = (
            _chk_s(weight, 0.0),
            *(_chk_s(v, 0.0, strict=True) for v in (target_std, eps)),
        )
        self.max_prototypes, self._bound_device, self._bound_shape = max_prototypes, bound_device, bound_shape
        self.record = self.axiom_record = self.thermal_source = self._vc = self._ov = None
        self.atlas = PrototypeAtlas(max_prototypes=max_prototypes, device=bound_device)
        for k, _ in _ROW_SPECS:
            setattr(self, k, None)
        self.rows_initialized = False
        self._ema, self._ema_t = {k: v for k, v in zip(_EMA_KEYS, (1.0, 1.0, 0.0))}, {}

    def _set_pc(self, v):
        self.atlas.proto_c = v
        if v is not None and v.ndim == 2:
            self.atlas.dim = v.shape[1]

    def _set_cov(self, v):
        self._ov, self.atlas.proto_cov = None, v

    def _set_var(self, v):
        self._ov = v
        if v is not None and self.atlas.dim >= 1:
            self.atlas.proto_cov = cov_from_scalar(v, self.atlas.dim)

    proto_c = property(lambda self: self.atlas.proto_c, _set_pc)
    proto_cov = property(lambda self: self.atlas.proto_cov, _set_cov)
    proto_var = property(lambda self: self._ov if self._ov is not None else self.atlas.proto_var, _set_var)

    def _flush(self):
        if not self._ema_t:
            return
        ts = tuple(self._ema_t.values())
        vs = (
            torch.stack([t.double() for t in ts]).tolist()
            if len({t.device for t in ts}) == 1
            else [float(t) for t in ts]
        )
        self._ema.update({k: float(v) for k, v in zip(self._ema_t, vs)})
        self._ema_t = {}

    def _dema(self, k, dev):
        c = self._ema_t.get(k)
        if c is not None:
            if c.device == dev:
                return c
            self._flush()
        return torch.full((), self._ema[k], device=dev)

    def _resize(self, r, k, dev):
        vr, vt = self.v_real_ema, self.v_tang_ema
        if vr is None or vt is None or len(vr) > k:
            self.v_real_ema, self.v_tang_ema = r.clone(), r.clone()
        else:
            self.v_real_ema, self.v_tang_ema = (
                cat((t.to(dev), r[len(t) :])) if len(t) < k else t.to(dev) for t in (vr, vt)
            )

    def _sync(self, dev=None):
        a = self.atlas
        c = a.proto_c
        if self._ov is not None and c is not None:
            a.dim, a.proto_cov, self._ov = (
                c.shape[1],
                cov_from_scalar(self._ov, c.shape[1]),
                None,
            )
        if not a.initialized or c is None:
            self.v_real_ema = self.v_tang_ema = None
            return
        k, dev = len(c), dev if dev is not None else c.device
        if a.proto_n is None or a.proto_n.numel() != k:
            a.proto_n = torch.full((k,), 25.0, device=dev)
        if a.v_push is None or a.v_push.numel() != k:
            a.v_push = cov_trace_mean(a.proto_cov).clone()
        self._resize(a.v_push.to(dev), k, dev)

    def forward(self, rows):
        _req(rows.ndim == 2 and rows.shape[1] >= 1)
        n = len(rows)
        if self.weight <= 0.0 or n < 2:
            return rows[:0].sum()
        m = min(n, 256)
        z = rows[(torch.arange(m, device=rows.device) * n) // m] if n > 256 else rows
        with torch.no_grad():
            s = _cd(z.detach().float()).square()
            i, j = torch.triu_indices(m, m, 1, device=z.device)
            r = s[i, j].median().clamp_min(_E).to(z.dtype)
            k = s.fill_diagonal_(_INF).topk(min(8, m - 1), dim=1, largest=False).indices
        v = self.weight * (
            (r / (r + (z.unsqueeze(1) - z[k]).square().sum(-1))).mean()
            + 0.15 * ((z.var(0, unbiased=False) + self.eps).sqrt() - self.target_std).square().mean()
        )
        sc = float(
            getattr(
                self.axiom_record or self.record or self.thermal_source or get_module_record(self), "thermal_scale", 1.0
            )
        )
        return (v * sc + v.detach() * (1.0 - sc)) if (sc < 1.0 and v.requires_grad) else v

    @torch.no_grad()
    def update_quantizer(self, yr):
        d = yr.device
        if self._bound_device is None:
            self._bound_device = d
        self._sync(d)
        self.atlas.update(yr)
        self._sync(d)

    @torch.no_grad()
    def update_curvature_ema(self, proto_ids, real_var_i, tang_var_i):
        self._sync(proto_ids.device)
        if self.v_real_ema is None:
            return
        k, a = len(self.v_real_ema), self.atlas
        h = proto_ids.bincount(minlength=k) > 0
        rm, tm = (
            _smean(proto_ids, real_var_i.float(), k).clamp_min(_VF),
            _smean(proto_ids, tang_var_i.float(), k).clamp_min(_VF),
        )
        self.v_real_ema, self.v_tang_ema, a.v_push = (
            where(h, 0.90 * c + 0.10 * x, c) for c, x in ((self.v_real_ema, rm), (self.v_tang_ema, tm), (a.v_push, tm))
        )

    @torch.no_grad()
    def curvature_gamma_per_component(self, coal):
        k = len(coal.comp_c)
        if self.v_real_ema is None or self.v_tang_ema is None:
            return coal.comp_c.new_ones(k)
        w = self.atlas.proto_w
        w = coal.comp_c.new_ones(len(self.v_real_ema)) if w is None else w.float()
        return (
            _smean(coal.proto_comp_id, self.v_tang_ema, k, w)
            / _smean(coal.proto_comp_id, self.v_real_ema, k, w).clamp_min(_VF)
        ).clamp(0.80, 1.35)

    @torch.no_grad()
    def update_laguerre_dual(self, xu_det, c_bar, w_mat, r_macro, laguerre_lr, *, row_mask=None):
        self._sync(xu_det.device)
        (k, d), a = c_bar.shape, self.atlas
        pw, wf, psi = a.proto_w, a.w_fake_ema, a.psi
        idx = (_cd(xu_det, c_bar).square() - (w_mat @ psi).unsqueeze(0)).argmin(1)
        m = w_mat[idx]
        if row_mask is None:
            n, wh = xu_det.new_full((), len(xu_det)), m.mean(0)
        else:
            rm = row_mask.float()
            n = rm.sum().clamp_min(1.0)
            wh = (m * rm.unsqueeze(1)).sum(0) / n
        a.w_fake_ema = wf = _norm(0.88 * wf + 0.12 * wh)
        pc, fc = w_mat @ pw, w_mat @ wf
        tv = 0.35 * (k / (n * 8.0).clamp_min(1.0)).sqrt()
        dm = ((pc - fc).clamp_min(0.0) / pc.clamp_min(1.0 / k)).amax()
        g = maximum(_ss((0.5 * (pw - wf).abs().sum() - tv) / tv.clamp_min(0.08)), _ss((dm - 0.15) / 0.20))
        r2 = d * r_macro.square()
        eta = g * min(3.0, max(0.35, float(laguerre_lr) * k)) * r2
        p = w_mat @ (psi * (1.0 - 0.004 * g * (pc <= fc).float()) + eta * (pw - (0.35 * wh + 0.65 * wf)))
        p -= p.mean()
        a.psi = p * ((2.5 * r2).clamp_min(_VF) / p.abs().amax().clamp_min(_VF)).clamp_max(1.0)
        return idx, g

    @torch.no_grad()
    def apply_prior_row_update(self, z_param, s_idx, is_first, g_rows, part_scale, row_lr, row_settle):
        p, d = z_param.shape
        if not self.rows_initialized or self.row_m is None or self.row_m.shape != (p, d):
            self.row_m, self.row_v, self.row_init, self.rows_initialized = (
                z_param.new_zeros((p, d), dtype=_F32),
                z_param.new_zeros((p, 1), dtype=_F32),
                z_param.new_zeros((p, 1), dtype=_B),
                True,
            )
        elif self.row_m.device != z_param.device:
            self.row_m, self.row_v, self.row_init = (getattr(self, k).to(z_param.device) for k, _ in _ROW_SPECS[2:])
        rm, rv, ri = self.row_m, self.row_v, self.row_init
        g, w = g_rows.detach().float(), is_first.float().unsqueeze(1)
        i, m, v = ri[s_idx], rm[s_idx], rv[s_idx]
        g2 = g.square().sum(-1, keepdim=True)
        mr, vr = where(i, 0.80 * m + 0.20 * g, g), where(i, 0.80 * v + 0.20 * g2, g2)
        rm.index_add_(0, s_idx, (mr - m) * w)
        rv.index_add_(0, s_idx, (vr - v) * w)
        ri[s_idx] = i | is_first.unsqueeze(1)
        s = vr.clamp_min(0.0).sqrt()
        r = ((s * w).sum() / w.sum().clamp_min(1.0)).clamp_min(1e-30)
        z_param.index_add_(
            0,
            s_idx,
            (
                (mr / (s + (1e-5 * r).clamp_min(1e-30)))
                * part_scale.float()
                * w
                * (-float(row_lr) * row_settle.float())
            ).to(z_param.dtype),
        )

    def state_dict(self, *a, **k):
        self._flush()
        self._sync()
        return {
            "schema": SCHEMA_VERSION,
            "max_prototypes": self.max_prototypes,
            "atlas": self.atlas.state_dict(),
            "rows_initialized": self.rows_initialized,
            **{k: float(self._ema[k]) for k in _EMA_KEYS},
            **{k: _cp(getattr(self, k)) for k, _ in _ROW_SPECS},
        }

    def load_state_dict(self, state_dict, strict=True, assign=False, *, validate_only=False):
        v = validate_axiom_prior_state(state_dict)
        _req(
            v["max_prototypes"] == self.max_prototypes
            and (not v["rows_initialized"] or self._bound_shape is None or v["row_m"].shape == tuple(self._bound_shape))
        )
        if validate_only:
            return
        dev = self._bound_device
        self._ov = self._vc = None
        self.atlas.load_state_dict(v["atlas"], device=dev)
        self._ema, self._ema_t, self.rows_initialized = {k: float(v[k]) for k in _EMA_KEYS}, {}, v["rows_initialized"]
        for k, dt in _ROW_SPECS:
            setattr(self, k, _to(v[k], dev, dt))

    state_dict_checkpoint, load_state_dict_checkpoint = state_dict, load_state_dict


def _is_pt(x):
    y = _flat(x)
    return bool(y.var(0, unbiased=False).sum() < (1e-11 * y.square().sum(1).mean()).clamp_min(1e-22))


def should_use_axiom_pullback(trainer, real, latent=None, indices=None):
    r = trainer.recipe
    if (
        r.prior_regularizer != "axiom"
        or r.t1_pullback_weight <= 0
        or r.encoder_mode != "none"
        or r.conditioning != "scalar"
        or real.ndim < 2
        or len(real) < 1
    ):
        return False
    d = real[0].numel()
    if not (1 <= d <= MAX_FULL_COV_DIM):
        return False
    if d >= 2 and has_positive_prior_sigma(trainer.prior):
        return True
    p, reg = _is_pt(real), getattr(trainer, "prior_regularizer", None)
    if hasattr(reg, "_vc"):
        reg._vc = (real, p)
    return p


class _Step:
    def __init__(self, tr, lat, idx, noise):
        self.tr, self.reg, self.r = tr, tr.prior_regularizer, tr.recipe
        self.bp, self.sz, self.cc = _kernel_spec(tr.prior)
        self.zp = getattr(self.bp, "z", None)
        self.zg = isinstance(self.zp, Tensor) and self.zp.requires_grad
        self.hs, self.lat, self.idx, self.ns = self.sz > 0.0, lat, idx, noise

    def _gen(self, z, so=0.0, rows=None):
        tr = self.tr
        return tr._postprocess(tr.prior, tr._generate(tr.G, z, so, self.ns, rows=rows))

    def _zrows(self):
        return self.zp if self.r.num_particles <= 1024 else self.zp[self.idx.unique()]

    def _sample(self, b, so, pt):
        bp, idx, lat = self.bp, self.idx, self.lat
        if pt:
            fg, md, eg = self._gen(lat, so, idx), None, None
        else:
            um = isinstance(bp, MoGParticlePrior) and not getattr(self.tr.prior, "transforms_codes", False)
            mu = bp.means()[idx] if um else (self.zp[idx] if self.cc else lat.detach())
            md = mu.detach()
            eg = ((lat - mu) / self.sz).detach() if self.hs else None
            fg = self._gen(md + self.sz * eg if eg is not None else md, so, idx)
        ff = fg.reshape(b, -1)
        self.fg, self.ff, self.xc, self.md, self.eg = (
            fg,
            ff,
            self._gen(md).reshape(b, -1) if eg is not None else ff,
            md,
            eg,
        )

    def _prep(self, x, gx, so, crit, pt):
        tr, b = self.tr, len(x)
        self._sample(b, so, pt)
        if hasattr(tr.opt_g, "bind_origin_sample"):
            tr.opt_g.bind_origin_sample(None if pt else self.fg)
        rg = gx() if callable(gx) else gx
        rg = x if rg is None else tr._batch(rg, "generator_real")
        _req(rg.shape == x.shape)
        fl, rl, rd = crit(self.fg), crit(rg), getattr(tr.opt_d, "record", None)
        if hasattr(rd, "observe_margin_and_scale"):
            rd.observe_margin_and_scale(rl, fl, rg, self.fg)
        self.rg, self.yr, self.fl, self.rd = rg, rg.detach().reshape(b, -1), fl, rd
        self.lg = F.softplus(-fl).mean() if self.r.loss == "non_saturating" else tr.loss.g_loss(fl, rl)

    def _opt(self, o, gs, wk, ps):
        if hasattr(o, "bind_axiom_modulation"):
            o.bind_axiom_modulation(g_scale=gs, wakeup_boost=wk, prior_scale=ps)
            return o.step()
        v, pgs = float(gs), o.param_groups
        sv = [g["lr"] for g in pgs]
        for g in pgs:
            g["lr"] *= v
        try:
            o.step()
        finally:
            for g, lr in zip(pgs, sv):
                g["lr"] = lr

    def _aux(self):
        tr, r, z, fg, rg = self.tr, self.r, self.lg.new_zeros(()), self.fg, self.rg
        kt = r.kinetic_transport_loss(fg, rg) if r.kinetic_transport_weight else z
        kl = r.kinetic_transport_local_loss(fg, rg) if r.kinetic_transport_local_weight else z
        pm = (
            tr.prior.regularize(self._zrows(), sampled_codes=self.lat, indices=self.idx, generated=fg, logits=self.fl)
            if getattr(tr, "_prior_penalty", False)
            else None
        )
        l2 = r.prior_l2 * self.zp.square().mean() if (self.zg and r.prior_l2 > 0) else z
        return pm, kt, kl, kt + kl + (z if pm is None else pm) + l2

    def _fin(self, loss, ld, pen, gs, wk, ps, rt=None):
        tr, rd, o, zp = self.tr, self.rd, self.tr.opt_g, self.zp
        pm, kt, kl, aux = self._aux()
        loss = loss + aux
        if rd is not None:
            rd._disc_gate = 0.0
            if rt is not None and hasattr(rd, "_ema_t"):
                rd._ema_t["ema_push_res"] = rt.detach().double()
            else:
                rd.ema_push_res = float(self.reg.res_ema)
        o.zero_grad()
        if getattr(o, "direction_blend", False) and hasattr(o, "bind_protected_from_grads"):
            loss.backward()
            o.bind_protected_from_grads()
        else:
            constraint_geometry_backward(loss, o, (loss,))
        if self.zg and zp.grad is None:
            zp.grad = zp.new_zeros(zp.shape)
        tr.policy.after_generator_backward(loss_gan=self.lg.detach(), loss_critic=(ld - pen).detach())
        self._opt(o, gs, wk, ps)
        tr.policy.after_generator_step()
        return loss, pm, kt, kl

    def _pt_step(self, ld, pen):
        reg, r, ff, yr, dt = self.reg, self.r, self.ff, self.yr, self.ff.dtype
        b, d = yr.shape
        with torch.no_grad():
            e = (ff.detach().float() - yr.float()).norm(dim=-1)
            c, p = max(float(e.mean()), 0.35 * float(e.amax())), float(reg.res_ema)
            cm = min(c, 10.0)
            re = cm if c > max(3.5 * p, p + 0.12) else 0.88 * p + 0.12 * cm
            reg.res_ema, gs = re, max(0.015, min(1.0, re / 0.025))
            z = self.lat.detach()
            o = 1e-3 / (max(1, z.shape[1]) ** 0.5)
            x = self._gen(cat((z + o, z - o))).reshape(2, b, d).float()
            s = ((x[0] - x[1]).abs() / 2e-3).clamp_min(1e-5)
            w = (s.quantile(0.5, dim=0, keepdim=True).clamp_min(max(0.15 * re, 1e-4)) / s).clamp(1.0, 12.0).to(dt)
        lo = ((1e-6 + w * (ff - yr.to(dt)).square()).sqrt() - 1e-3).mean()
        if self.zg:
            lo = lo + 0.004 * self.zp.float().square().mean().to(dt)
        pd = reg(self._zrows()).detach() if (self.zg and r.prior_reg > 0) else self.lg.new_zeros(())
        lg, pm, kt, kl = self._fin(
            0.02 * min(1.0, re / 0.25) * self.lg + r.t1_pullback_weight * lo, ld, pen, gs, 1.0, gs
        )
        return lg, self.lg, pd, pm, kt, kl

    @torch.no_grad()
    def _targets(self, coal, wi):
        reg, yr, c, w, v = self.reg, self.yr, coal.comp_c, coal.comp_w, coal.comp_var
        a, sd, dev, y, k = reg.atlas, yr.shape[1] ** 0.5, yr.device, yr.float(), len(c)
        xc, fd, yv = self.xc.detach().float(), self.ff.detach().float(), y.var(0, unbiased=False)
        rm = yv.clamp_min(0.0).sqrt().mean().clamp_min(_E)
        so = (0.5 * yv.mean().clamp_min(_VF).sqrt()).clamp_min(_E)
        sm = v.sqrt().quantile(0.35).clamp_min(_E)
        led = (
            a.commit_particles(xc, self.idx, len(self.zp), coalesced=coal)
            if (self.cc and self.zp is not None)
            else None
        )
        ci, x = led.batch_comp if led is not None else _cd(xc, c).argmin(1), where(coal.all_iso, xc, fd)
        sr = x.std(0, unbiased=False).mean().clamp_min(_E) / rm
        wk = where(sr < 0.50, 5.0, where(sr < 0.80, 2.0, 1.0))
        ip, rp = reg._dema("iso_ema", dev), reg._dema("res_ema", dev)
        iv = (self.sz * rm / sm).clamp(1.0, 4.5)
        ie = (iv if not wi else 0.95 * ip + 0.05 * iv) if self.hs else ip.new_ones(())
        dm = _cd(x, c)
        re = 0.92 * rp + 0.08 * (
            (maximum((w * dm.amin(0)).sum(), dm.amin(1).median()) - 1.45 * sd * sm).clamp_min(0.0) / rm
        ).clamp_max(10.0)
        reg._ema_t["iso_ema"], reg._ema_t["res_ema"] = ie, re
        ce = coal.comp_cov_tgt * reg.curvature_gamma_per_component(coal).view(k, 1, 1)
        ch = _chol(ce)
        if led is not None:
            yt = led.batch_target
        else:
            dr = dm
            if k >= 4:
                wu = w.unsqueeze(1)
                dc = c - (wu * c).sum(0, keepdim=True)
                dr = where(
                    sr < 0.35,
                    _cd(
                        (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(_E),
                        dc / (wu * dc.square()).sum(0, keepdim=True).sqrt().clamp_min(_E),
                    ),
                    dm,
                )
            yt = (
                _norm(
                    log_sinkhorn(
                        -(dr / dr.mean().clamp_min(_E)) / (0.25 * rp).clamp(0.010, 0.06),
                        dr.new_full((len(dr),), -log(max(1, len(dr)))),
                        w.clamp_min(_E).log(),
                        iters=10,
                    ).exp(),
                    1,
                )
                @ c
            ).float()
        sp = (1.0 - _smean(coal.proto_comp_id, a.v_push, k, a.proto_w) / v.clamp_min(_VF)).clamp(0.0, 0.75).sqrt()
        if self.cc and self.hs:
            hl = (
                ((x - c[ci]).norm(dim=-1) / (3.5 * sd * v[ci].sqrt()).clamp_min(_E))
                .clamp(0.0, 1.0)
                .square()
                .unsqueeze(1)
            )
            yt = where(
                (coal.all_iso & (sp[ci] > 1e-3) & (sr >= 0.65)).unsqueeze(1),
                hl * yt + (1.0 - hl) * _bm(self.idx, ci, c, ch, sp),
                yt,
            )
        self.rm, self.so, self.led, self.ie, self.re, self.ch, self.sp = rm, so, led is not None, ie, re, ch, sp
        return (
            yt,
            sm,
            y.abs().amax(0, keepdim=True) + 3.5 * sm,
            sr,
            wk,
            ce[ci],
            ((k * w[ci]).clamp_min(1e-4) ** -0.5).clamp(0.5, 4.0),
        )

    def _burg(self, ct, iw):
        reg = self.reg
        ie = reg._dema("iso_err_ema", self.yr.device)
        if not self.hs or self.md is None or self.r.t1_iso_weight <= 0:
            return self.ff.new_zeros(()), ie
        li, tv, fr = whitened_burg_pullback_loss(
            self._gen, self.md, self.sz, ct, iw, eps_g=self.eg, phase=float(getattr(self.rd, "chord_phase", 0.0))
        )
        with torch.no_grad():
            m = len(tv)
            xc = self.xc[:m].detach().float()
            reg.update_curvature_ema(
                _cd(xc, reg.atlas.proto_c).argmin(1), (self.ff[:m].detach().float() - xc).square().mean(-1), tv
            )
            ie = 0.92 * ie + 0.08 * (0.60 * fr.mean() + 0.40 * fr.quantile(0.92)).clamp_max(5.0)
            reg._ema_t["iso_err_ema"] = ie
        return li.to(self.ff.dtype), ie

    @torch.no_grad()
    def _rtgt(self, coal, si, isf, xu, dt, rfl):
        rc = self.reg.atlas.ledger.row_comp
        if self.led and rc is not None:
            rc = rc.to(si.device)
            cs = where(rc[si] >= 0, rc[si].clamp_min(0), _cd(xu, coal.comp_c).argmin(1))
            sc, gp = coal.comp_var[cs].sqrt().to(dt), xu.new_zeros(())
            ct = _bm(si, cs, coal.comp_c, self.ch, self.sp).to(dt) if self.hs else coal.comp_c[cs].to(dt)
        else:
            ks, gp = self.reg.update_laguerre_dual(
                xu, coal.proto_c_bar, coal.w_mat, self.rm, self.r.t1_laguerre_lr, row_mask=isf
            )
            sc, ct = coal.proto_var_bar[ks].sqrt().to(dt), coal.proto_c_bar[ks].to(dt)
        return ct, sc, maximum(rfl, maximum(self.re / 0.10, gp).clamp_max(1.0))

    def _rows(self, coal, rfl):
        bp, zp, si = self.bp, self.zp, self.idx.sort().values
        isf = cat((si.new_ones(1, dtype=_B), si[1:] != si[:-1]))
        wf = isf.float()
        nu, gp = wf.sum().clamp_min(1.0), tuple(self.tr.G.parameters())
        fls = [p.requires_grad for p in gp]
        try:
            for p in gp:
                p.requires_grad_(False)
            xu = self._gen((bp.means() if isinstance(bp, MoGParticlePrior) else zp)[si]).reshape(len(si), -1)
            xud = xu.detach().float()
            ct, sc, stl = self._rtgt(coal, si, isf, xud, xu.dtype, rfl)
            bi = _bi(xu, ct, (0.4 * sc).clamp_min(_E), aligned=True)[0]
            lm = (wf.to(bi.dtype) * bi).sum() / (nu * self.so).to(bi.dtype)
            if self.hs:
                mr = zp[si]
                dt = mr.dtype
                wm = (wf / nu).unsqueeze(1).to(dt)
                zs = ((wm * (mr - (wm * mr).sum(0, keepdim=True)).square()).sum(0) + 1e-6).sqrt()
                lm = lm + 0.15 * F.relu(self.ie.to(dt) - zs).square().mean() * (nu >= 2).to(dt)
            if zp.grad is not None:
                zp.grad.zero_()
            lm.backward()
            with torch.no_grad():
                gr = zp.grad[si] * nu.to(zp.dtype)
                zp.grad.zero_()
                ps = ((xud - ct.float()).norm(dim=-1) / (1.5 * sc.float()).clamp_min(_E)).clamp(0.02, 1.2).unsqueeze(1)
                self.reg.apply_prior_row_update(zp, si, isf, gr, ps, self.r.t1_prior_row_lr, stl)
            return lm.detach()
        finally:
            for p, f in zip(gp, fls):
                p.requires_grad_(f)
            if self.rd is not None:
                self.rd._g_probe_cache = None

    def run(self, x, gx, so, crit, ld, pen):
        if isinstance(self.zp, Tensor):
            set_active_prior_rows(self.zp, self.idx)
        reg, r = self.reg, self.r
        a = reg.atlas
        vc, reg._vc = reg._vc, None
        pt = (
            bool(vc[1])
            if (vc is not None and vc[0] is x)
            else (not (self.hs and a.initialized and not a._last_point_mass) and _is_pt(x))
        )
        self._prep(x, gx, so, crit, pt)
        if pt:
            return self._pt_step(ld, pen)
        wi = reg.quantizer_initialized
        reg.update_quantizer(self.yr)
        coal = a.coalesce(self.yr)
        yt, sm, yb, sr, wk, ct, iw = self._targets(coal, wi)
        xs = self.xc if self.hs else self.ff
        dt, so = xs.dtype, self.so
        lo = _bi(xs, yt.to(dt), sm.to(dt), aligned=True)[0].mean() / so.to(dt) + 2.0 * (
            F.relu(self.ff.float().abs() - yb) / so
        ).square().mean().to(dt)
        li, ie = self._burg(ct, iw)
        ai = coal.all_iso
        eq = where(ai, ((0.030 - self.re) / 0.020).clamp(0.0, 1.0), 0.0)
        aw, gfl, rfl = (
            where(ai, u * (1.0 - eq) + v * eq, c)
            for u, v, c in ((0.038, 0.008, 0.15), (0.022, 0.012, 0.04), (0.15, 0.38, 0.15))
        )
        aw = aw.to(self.lg.dtype)
        gs = where(
            sr < 0.60,
            1.0,
            maximum(gfl, maximum(self.re / 0.10, ((ie - 0.22) / 0.45).clamp(0.0, 1.0)).clamp_max(1.0)),
        )
        lg, pm, kt, kl = self._fin(
            aw * self.lg + r.t1_pullback_weight * lo + r.t1_iso_weight * li, ld, pen, gs, wk, 1.0, rt=self.re
        )
        pd = self._rows(coal, rfl) if (self.cc and self.zg and r.t1_prior_row_lr > 0) else self.lg.new_zeros(())
        return lg, self.lg, pd, pm, kt, kl


def axiom_pullback_step(trainer, real, generator_real, latent, indices, *, sigma_out, noise, critic, loss_d, penalty):
    return _Step(trainer, latent, indices, noise).run(real, generator_real, sigma_out, critic, loss_d, penalty)


__all__ = [
    "AxiomPriorRegularizer",
    "SCHEMA_VERSION",
    "allocate_deficit_quotas",
    "axiom_pullback_step",
    "central_pullback_cov",
    "coalesce_pythagorean_ltv",
    "has_positive_prior_sigma",
    "should_use_axiom_pullback",
    "validate_axiom_prior_state",
    "whitened_burg_pullback_loss",
]
