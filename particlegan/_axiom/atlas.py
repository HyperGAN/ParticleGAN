"""Axiom L1: PrototypeAtlas & GPU HamiltonCommitmentLedger."""

from collections import namedtuple
import math

import torch
from torch import Tensor, arange, bincount, cat, eye, full, maximum, ones, stack, where, zeros

from .math_ops import (
    PHI1,
    hamilton_quotas as _hq,
    power_principal_axis as _ppa,
    safe_cdist as _cd,
    smoothstep01 as _ss,
)

SCHEMA_VERSION = 2
MAX_FULL_COV_DIM = 64
MAX_LEDGER_PARTICLES = 2048
_VF, _E, _BETA_N, _ROUNDS, _INF = 1e-24, 1e-12, 0.04, 8, float("inf")
_I64, _F32, _B = torch.int64, torch.float32, torch.bool

_LEDGER_H = ("initialized", "num_particles", "dim")
_LEDGER_T = ("row_comp", "row_target", "row_dist")
_LEDGER_K = frozenset({"schema", *_LEDGER_H, *_LEDGER_T})
_ATLAS_T = ("proto_c", "proto_cov", "proto_w", "w_fake_ema", "proto_n", "psi", "v_push")
_ALL_T = (*_ATLAS_T, "_rho_atom_t", "_g_conn_t")
_ATLAS_K = frozenset(
    {"schema", "max_prototypes", "beta_n", "initialized", "dim", "rho_atom_ema", "g_conn_ema", *_ATLAS_T, "ledger"}
)

CoalescedAtlas = namedtuple(
    "CoalescedAtlas",
    "proto_comp_id proto_c_bar proto_cov_bar proto_var_bar w_mat multicell isolated "
    "comp_c comp_cov comp_cov_tgt comp_var comp_w comp_n_eff comp_isolated all_iso",
)
LedgerAssignment = namedtuple(
    "LedgerAssignment", "batch_comp batch_target nn_comp_dist quotas occupancy tail_residual row_comp"
)


def _req(c):
    if not c:
        raise ValueError("invalid state")


def _chk_hdr(s, keys):
    _req(isinstance(s, dict) and set(s) == keys and type(s["schema"]) is int and s["schema"] == SCHEMA_VERSION)


def _chk_s(v, lo=None, hi=None, strict=False):
    _req(type(v) in (int, float) and math.isfinite(float(v)))
    x = float(v)
    _req((lo is None or (x > lo if strict else x >= lo)) and (hi is None or (x < hi if strict else x <= hi)))
    return x


def _chk_t(x, shape=None, nonneg=False, dt=None):
    _req(isinstance(x, Tensor) and (x.dtype == dt if dt is not None else x.is_floating_point()))
    _req(shape is None or tuple(x.shape) == shape)
    if dt is None:
        _req(bool(torch.isfinite(x).all()) and (not nonneg or not bool((x < 0.0).any())))
    return x


def _cp(x):
    return None if x is None else x.detach().cpu().clone()


def _to(x, dev, dt=_F32):
    return None if x is None else x.detach().to(dev, dt).clone()


def _dg(x):
    return x.diagonal(dim1=-2, dim2=-1)


def _flat(x):
    return x.detach().reshape(len(x), -1).float()


def _norm(w, dim=None):
    return (w / w.sum().clamp_min(_E)) if dim is None else (w / w.sum(dim, keepdim=True).clamp_min(_E))


def _bc(x, f):
    return x.view(-1, 1, 1) if f else x.unsqueeze(1)


def _dedup(x, tol):
    return (_cd(x) <= tol).float().argmax(1) == arange(len(x), device=x.device)


def _occ(r, k):
    return bincount(where(r >= 0, r, k), minlength=k + 1)[:k]


def _san(cov, fac=None):
    if fac is not None:
        cov = cov * _bc(fac, cov.ndim == 3)
    if cov.ndim == 3:
        s = 0.5 * (cov + cov.mT)
        return s + torch.diag_embed((_VF - _dg(s)).clamp_min(0.0))
    return cov.clamp_min(_VF)


def cov_trace_mean(cov):
    return (_dg(cov) if cov.ndim == 3 else cov).mean(-1).clamp_min(_VF)


def cov_from_scalar(var, dim):
    v = var.float().clamp_min(_VF)
    return (
        v.view(-1, 1, 1) * eye(dim, device=v.device)
        if dim <= MAX_FULL_COV_DIM
        else v.unsqueeze(-1).expand(len(v), dim).clone()
    )


def cov_directional_var(cov, dirs):
    if cov.ndim == 3:
        return torch.einsum("kmd,kde,kme->km" if dirs.ndim == 3 else "kd,kde,ke->k", dirs, cov, dirs).clamp_min(_VF)
    return (dirs.square() * (cov.unsqueeze(1) if dirs.ndim == 3 else cov)).sum(-1).clamp_min(_VF)


def cov_add_outer(cov, delta, scale=1.0):
    if cov.ndim == 3:
        o = cov + (delta.unsqueeze(-1) * delta.unsqueeze(-2)) * (
            scale.view(*scale.shape, 1, 1) if isinstance(scale, Tensor) else float(scale)
        )
        return 0.5 * (o + o.mT)
    return (cov + delta.square() * (scale.unsqueeze(-1) if isinstance(scale, Tensor) else float(scale))).clamp_min(_VF)


def _scov(x, idx, k, f):
    d = x.shape[1]
    return (
        x.new_zeros((k, d, d)).index_add_(0, idx, x.unsqueeze(2) * x.unsqueeze(1))
        if f
        else x.new_zeros((k, d)).index_add_(0, idx, x.square())
    )


def cov_principal_split(cov_k, seed=0):
    x, dt = cov_k.detach().double(), cov_k.dtype
    if x.ndim == 2:
        s = 0.5 * (x + x.T)
        lam, u = _ppa(s, iters=6, seed=seed)
        su = s @ u
        r = (u * su).sum()
        v = where(r > _VF, 0.40 * su * r.clamp_min(_VF).rsqrt(), 0.0)
        c = _san((s - v.unsqueeze(1) * v.unsqueeze(0)).unsqueeze(0))[0]
        return lam.clamp_min(0.0).to(dt), v.to(dt), c.to(dt)
    g = x.clamp_min(0.0)
    lam, idx = g.max(-1)
    v, c = g.new_zeros(g.shape), g.clone()
    v[idx] = where(lam > _VF, 0.40 * lam.clamp_min(_VF).sqrt(), 0.0)
    c[idx] = where(lam > _VF, 0.84 * lam, g[idx])
    return lam.to(dt), v.to(dt), c.clamp_min(_VF).to(dt)


def cov_shrink_mp(cov, n_eff):
    d = cov.shape[-1]
    m, v = _ss((d / n_eff.float().clamp_min(1.0) - 0.25) / 0.75).clamp_min(1e-4), cov_trace_mean(cov)
    if cov.ndim == 3:
        return (1.0 - m.view(-1, 1, 1)) * cov + (m * v).view(-1, 1, 1) * eye(d, device=cov.device)
    return ((1.0 - m.unsqueeze(-1)) * cov + (m * v).unsqueeze(-1)).clamp_min(_VF)


@torch.no_grad()
def batch_annular_gap_ratio(y_real):
    x = _flat(y_real)
    b = len(x)
    if b < 4:
        return x.new_ones(())
    if b > 256:
        x, b = x[(arange(256, device=x.device) * b) // 256], 256
    i, j = torch.triu_indices(b, b, offset=1, device=x.device)
    s = _cd(x)[i, j].sort().values
    m = s[-1]
    q, n = s.clamp_min(1e-7 * m.clamp_min(_VF)), max(2, s.numel() // 2)
    g, r = (q[2 : n + 1].log() - q[1:n].log()).max(0)
    w = _ss((g - 0.50) / 0.70)
    return where(
        m <= _VF,
        0.0,
        (w * (s[(r + 1) // 2] / q[r + 2].clamp_min(_VF)) + (1.0 - w) * (-g.clamp_min(0.0)).exp()).clamp(0.0, 1.0),
    )


@torch.no_grad()
def manifold_connectivity_stats(y_real):
    x = _flat(y_real)
    b = len(x)
    a = x.abs().amax() if x.numel() > 0 else x.new_ones(())
    s = where(a < 1e-2, a.clamp_min(_E), a.clamp_min(1.0))
    if b < 4:
        z = x.new_zeros(())
        return z, z, x.new_full((), 0.1), s
    y = (x / s)[:: max(1, b // 256)] if b > 512 else (x / s)
    d = _cd(y)
    m = ~eye(len(y), dtype=_B, device=y.device)
    q = torch.quantile(d.masked_fill(~m, _INF).amin(-1), 0.90).clamp_min(1e-8)
    r = torch.quantile(d[m], 0.90) / q
    u = d.clone()
    for _ in range(8):
        u = torch.minimum(u, maximum(u.unsqueeze(2), u.unsqueeze(0)).amin(1))
    g = _ss((10.0 - u.amax() / q) / 3.0) * _ss((r - 1.15) / 0.25)
    return g, g * _ss((r - 3.80) / 1.00), q.clamp_min(1e-4), s


@torch.no_grad()
def compute_rho_atom(comp_c, comp_cov, comp_n_eff, y_real=None):
    k, d = comp_c.shape
    rb = batch_annular_gap_ratio(y_real) if y_real is not None else comp_c.new_ones(())
    if k < 2:
        return maximum(rb, _ss((comp_n_eff.median() - 2.0) / 6.0) if k == 1 else comp_c.new_zeros(())).clamp(0.0, 1.0)
    r, idx = _cd(comp_c, diag_inf=True).min(1)
    rc = (
        where(
            r > _E,
            (d * cov_directional_var(comp_cov, (comp_c - comp_c[idx]) / r.clamp_min(_E).unsqueeze(1))).sqrt()
            / r.clamp_min(_E),
            1.0,
        )
        .median()
        .clamp(0.0, 1.0)
    )
    if y_real is None:
        return rc
    w = _ss((comp_n_eff.median() - 2.0) / 6.0)
    return ((1.0 - w) * rb + w * rc).clamp(0.0, 1.0)


@torch.no_grad()
def compute_connectivity_gate(y_real, *, rho_atom=None):
    g = manifold_connectivity_stats(y_real)[0]
    return g.clamp(0.0, 1.0) if rho_atom is None else (g * _ss((rho_atom - 0.15) / 0.20)).clamp(0.0, 1.0)


def _ltc_pass(c, cov, w, a, e):
    k = len(c)
    o = a.clone().fill_diagonal_(1.0)
    for _ in range(min(8, max(4, (k - 1).bit_length()))):
        o = ((o @ o) > 0).float()
    s = o.sum(1)
    m = (s >= 1.0) & (s < k)
    cw = _norm(where(m[:, None] & m[None, :], o, e) * w.unsqueeze(0), 1)
    cc = cw @ c
    q = cov_add_outer(cov, c - cc)
    return o, s, m, cw, cc, _san((cw @ q.reshape(k, -1)).reshape(q.shape))


@torch.no_grad()
def coalesce_directional_ltc(proto_c, proto_cov, proto_w, proto_n=None):
    c, dv, tm = proto_c.detach().float(), cov_directional_var, cov_trace_mean
    k, d, dev = c.shape[0], c.shape[1], c.device
    w = _norm(proto_w.detach().float().clamp_min(_E))
    n = ones(k, device=dev) if proto_n is None else proto_n.detach().float().clamp_min(0.0)
    cov, e = _san(proto_cov.detach().float()), eye(k, device=dev)
    r, sd = _cd(c), math.sqrt(d)
    u = (c.unsqueeze(1) - c.unsqueeze(0)) / r.clamp_min(_E).unsqueeze(-1)
    sg, st = dv(cov_shrink_mp(cov, n), u).sqrt(), tm(cov).sqrt()
    wk = maximum(st[:, None], st[None, :]) <= (2.2 * torch.minimum(st[:, None], st[None, :]).clamp_min(_E))
    a, s, m, _, _, cov1 = _ltc_pass(c, cov, w, ((r <= (2.6 * sd * maximum(sg, sg.T))) & wk).float(), e)
    b = (r <= (2.6 * sd * dv(cov1, u).sqrt().T)) & ((s >= 1.5) & m)[None, :] & wk
    a, s, m, cw, cc, cov2 = _ltc_pass(c, cov, w, ((a > 0.5) | b | b.T).float(), e)
    q = r / (4.2 * sd * dv(cov2, u).sqrt()).clamp_min(_E)
    iso = (
        m
        & ((a @ (~((q * a).amax(1) <= 1.0)).float()) < 0.5)
        & ((a @ (~(q.masked_fill(a > 0.5, _INF).amin(1) >= 1.0)).float()) < 0.5)
    )
    ok2 = iso & ((w * iso.float()).sum() >= 0.70)
    cb, covb = where(ok2.unsqueeze(1), cc, c), where(_bc(ok2, cov.ndim == 3), cov2, cov)
    ea = where(ok2[:, None] & ok2[None, :], a, e)
    rep = (ea > 0.5).float().argmax(1)
    isr = rep == arange(k, device=dev)
    idx = isr.nonzero().squeeze(-1)
    ccov, wc, nc, ic = covb[idx], _norm(ea[idx] @ w), ea[idx] @ n, iso[idx]
    return CoalescedAtlas(
        (isr.long().cumsum(0) - 1)[rep],
        cb,
        covb,
        tm(covb),
        where(ok2.unsqueeze(1), cw, e),
        (s > 1.5) & ok2,
        iso,
        cb[idx],
        ccov,
        cov_shrink_mp(ccov, nc),
        tm(ccov),
        wc,
        nc,
        ic,
        ic.all() if idx.numel() >= 2 else zeros((), device=dev, dtype=_B),
    )


@torch.no_grad()
def coalesce_pythagorean_ltv(proto_c, proto_var, proto_w):
    c = coalesce_directional_ltc(proto_c, cov_from_scalar(proto_var, proto_c.shape[1]), proto_w)
    return c.proto_c_bar, c.proto_var_bar, c.w_mat, c.multicell


def validate_ledger_state(state):
    _chk_hdr(state, _LEDGER_K)
    i, p, d = state["initialized"], state["num_particles"], state["dim"]
    _req(type(i) is bool and type(p) is int and type(d) is int)
    if not i:
        _req(p == 0 and d == 0 and all(state[k] is None for k in _LEDGER_T))
        return state
    _req(1 <= p <= MAX_LEDGER_PARTICLES and d >= 1 and not bool((_chk_t(state["row_comp"], (p,), dt=_I64) < -1).any()))
    _chk_t(state["row_target"], (p, d))
    _chk_t(state["row_dist"], (p,), True)
    return state


def _val_at(s, k, d):
    f = d <= MAX_FULL_COV_DIM
    t = _chk_t(s["proto_cov"], (k, d, d) if f else (k, d), not f)
    if f:
        _req(
            float((t - t.mT).abs().amax()) <= 1e-4 * float(t.abs().amax().clamp_min(1.0))
            and not bool((_dg(t) < 0.0).any())
        )
    for nm in ("proto_w", "w_fake_ema"):
        _req(abs(float(_chk_t(s[nm], (k,), True).double().sum()) - 1.0) <= 1e-4)
    for nm, nn in (("proto_n", True), ("v_push", True), ("psi", False)):
        _chk_t(s[nm], (k,), nn)


def validate_atlas_state(state):
    _chk_hdr(state, _ATLAS_K)
    m, i, d = state["max_prototypes"], state["initialized"], state["dim"]
    _req(type(m) is int and m >= 2 and type(i) is bool and type(d) is int)
    for nm, lo, hi, st in (
        ("beta_n", 0.0, 1.0, True),
        ("rho_atom_ema", 0.0, None, False),
        ("g_conn_ema", 0.0, 1.0, False),
    ):
        _chk_s(state[nm], lo, hi, st)
    led = state["ledger"]
    if led is not None:
        validate_ledger_state(led)
    if not i:
        _req(d == 0 and all(state[k] is None for k in _ATLAS_T))
        return state
    c = _chk_t(state["proto_c"])
    _req(d >= 1 and c.ndim == 2 and c.shape[1] == d and 1 <= len(c) <= m)
    _val_at(state, len(c), d)
    _req(led is None or not led["initialized"] or led["dim"] == d)
    return state


def _rmask(g, d, w, lim, k, ge):
    o = (d.clamp_max(1e6) + 1e-6 * w).argsort(stable=True)
    o = o[g[o].argsort(stable=True)]
    s = g[o]
    n = bincount(s, minlength=k + 1)
    r = arange(g.numel(), device=g.device, dtype=_I64) - (n.cumsum(0) - n)[s]
    q = lim[s.clamp_max(k - 1)]
    return zeros(g.shape, dtype=_B, device=g.device).scatter_(0, o, (s < k) & (r >= q if ge else r < q))


class HamiltonCommitmentLedger:
    SCHEMA = SCHEMA_VERSION

    def __init__(self):
        self.reset()

    def reset(self):
        self.initialized, self.num_particles, self.dim = False, 0, 0
        self.row_comp = self.row_target = self.row_dist = None

    def state_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            **{k: getattr(self, k) for k in _LEDGER_H},
            **{k: _cp(getattr(self, k)) for k in _LEDGER_T},
        }

    def load_state_dict(self, state, *, device=None, validate_only=False):
        v = validate_ledger_state(state)
        if validate_only:
            return
        for k in _LEDGER_H:
            setattr(self, k, v[k])
        if not self.initialized:
            self.row_comp = self.row_target = self.row_dist = None
            return
        dev = device if device is not None else v["row_target"].device
        for k, dt in zip(_LEDGER_T, (_I64, _F32, _F32)):
            setattr(self, k, _to(v[k], dev, dt))

    def _assign(self, r, rt, rd, d, c, m, q, w):
        k, p, dev = len(c), r.numel(), r.device
        de, u = (q - _occ(r, k)).clamp_min(0), m & (r < 0)
        for _ in range(_ROUNDS):
            pd, pc = d.masked_fill((de <= 0).unsqueeze(0), _INF).min(1)
            a = _rmask(where(u & torch.isfinite(pd), pc, k), pd, w, de, k, False)
            r, rt, rd, u = where(a, pc, r), where(a.unsqueeze(1), c[pc], rt), where(a, pd, rd), u & (~a)
            de = (de - _occ(where(a, pc, -1), k)).clamp_min(0)
        um = d.masked_fill((de <= 0).unsqueeze(0), _INF).amin(1)
        co = d.masked_fill(~u.unsqueeze(1), _INF).amin(0).clamp_max(1e6).argsort(stable=True)
        e, rk = de[co].cumsum(0), arange(p, device=dev, dtype=_I64)
        sc = co[torch.searchsorted(e, rk, right=True).clamp_max(k - 1)]
        o = where(u, um.clamp_max(1e6) + 1e-6 * w, 1e12).argsort(stable=True)
        fid = full((p,), -1, dtype=_I64, device=dev).scatter_(0, o, where(u[o] & (rk < e[-1]), sc, -1))
        f, idx = fid >= 0, fid.clamp_min(0)
        return (
            where(f, idx, r),
            where(f.unsqueeze(1), c[idx], rt),
            where(f, d.gather(1, idx.unsqueeze(1)).squeeze(1), rd),
        )

    @torch.no_grad()
    def update(self, x_fake, rows, comp_c, comp_w, num_particles, *, allow_update=True):
        b, k, p, dev = len(x_fake), len(comp_c), int(num_particles), x_fake.device
        _req(rows.ndim == 1 and len(rows) == b and k >= 1 and 1 <= p <= MAX_LEDGER_PARTICLES)
        c = comp_c.detach().to(dev, _F32).reshape(k, -1)
        d = c.shape[1]
        x = x_fake.detach().to(dev, _F32).reshape(b, d)
        if self.initialized and self.num_particles == p and self.dim == d and self.row_comp is not None:
            r, rt, rd = self.row_comp.to(dev), self.row_target.to(dev), self.row_dist.to(dev)
        else:
            self.initialized, self.num_particles, self.dim = True, p, d
            r, rt, rd = full((p,), -1, dtype=_I64, device=dev), zeros((p, d), device=dev), full((p,), 1e30, device=dev)
        idx, dm = rows.detach().to(dev, _I64).clamp(0, p - 1), _cd(x, c)
        q = _hq(p, comp_w.detach().to(dev), active_scores=-dm.amin(0))
        if allow_update and b > 0:
            m = zeros(p, dtype=_B, device=dev)
            m[idx] = True
            fp = full((p,), b, dtype=_I64, device=dev).scatter_reduce_(
                0, idx, arange(b, device=dev, dtype=_I64), reduce="amin"
            )
            dp, w = dm[fp.clamp_max(b - 1)], ((arange(p, device=dev, dtype=_F32) + 0.5) * PHI1).remainder(1.0)
            a, nc = r >= 0, _cd(rt, c).argmin(1)
            mt = c[nc]
            r = where(a, nc, r)
            rd = where(
                m & (r >= 0),
                dp.gather(1, r.clamp_min(0).unsqueeze(1)).squeeze(1),
                where(a, maximum(rd, (rt - mt).norm(dim=1)), rd),
            )
            rt = where(a.unsqueeze(1), mt, rt)
            ev = _rmask(where(r >= 0, r, k), rd, w, q, k, True)
            r, rt, rd = self._assign(where(ev, -1, r), rt, where(ev, 1e30, rd), dp, c, m, q, w)
        self.row_comp, self.row_target, self.row_dist = r, rt, rd
        nn = _cd(c, diag_inf=True).amin(1).clamp_min(_E) if k >= 2 else ones(k, device=dev)
        tk, occ = where(r[idx] >= 0, r[idx].clamp_min(0), dm.argmin(1)), _occ(r, k)
        nu = p - ((r >= 0) & (rd <= 0.09 * nn[r.clamp_min(0)])).float().sum()
        rc = where((nu > 0.5) | (occ < q).any(), 0.18 + 0.12 * (nu.clamp_min(1.0) / max(1, p)), 0.0)
        return LedgerAssignment(
            tk, c[tk], nn, q, occ, maximum(1.6 * ((x - c[tk]).norm(dim=1) / nn[tk]).clamp_max(10.0).mean(), rc), r
        )


class PrototypeAtlas:
    SCHEMA = SCHEMA_VERSION
    proto_var = property(lambda self: None if self.proto_cov is None else cov_trace_mean(self.proto_cov))
    n_eff = property(lambda self: self.proto_n)
    num_prototypes = property(lambda self: 0 if self.proto_c is None else len(self.proto_c))

    def __init__(self, max_prototypes=256, *, beta_n=_BETA_N, device=None):
        _req(type(max_prototypes) is int and max_prototypes >= 2)
        self.beta_n, self.max_prototypes, self.device = _chk_s(beta_n, 0.0, 1.0, True), max_prototypes, device
        self.ledger = HamiltonCommitmentLedger()
        self.reset()

    def _flush(self):
        if self._rho_atom_t is not None and self._g_conn_t is not None:
            v = stack((self._rho_atom_t.double(), self._g_conn_t.double())).tolist()
            self._rho_atom_f, self._g_conn_f, self._rho_atom_t, self._g_conn_t = float(v[0]), float(v[1]), None, None

    @property
    def rho_atom_ema(self):
        self._flush()
        return self._rho_atom_f

    @property
    def g_conn_ema(self):
        self._flush()
        return self._g_conn_f

    def reset(self):
        self.initialized, self.dim, self._last_point_mass = False, 0, False
        for k in _ALL_T:
            setattr(self, k, None)
        self._rho_atom_f, self._g_conn_f = 1.0, 0.0
        self.ledger.reset()

    def state_dict(self):
        self._flush()
        return {
            "schema": SCHEMA_VERSION,
            "max_prototypes": self.max_prototypes,
            "beta_n": self.beta_n,
            "initialized": self.initialized,
            "dim": self.dim,
            "rho_atom_ema": self._rho_atom_f,
            "g_conn_ema": self._g_conn_f,
            **{k: _cp(getattr(self, k)) for k in _ATLAS_T},
            "ledger": self.ledger.state_dict() if self.ledger.initialized else None,
        }

    def load_state_dict(self, state, *, device=None, validate_only=False):
        v = validate_atlas_state(state)
        _req(v["max_prototypes"] == self.max_prototypes)
        if validate_only:
            return
        dev = device if device is not None else self.device
        self.beta_n, self.initialized, self.dim = float(v["beta_n"]), v["initialized"], v["dim"]
        self._rho_atom_f, self._g_conn_f, self._last_point_mass, self._rho_atom_t, self._g_conn_t = (
            float(v["rho_atom_ema"]),
            float(v["g_conn_ema"]),
            False,
            None,
            None,
        )
        for k in _ATLAS_T:
            setattr(self, k, _to(v[k], dev))
        self.ledger.reset() if v["ledger"] is None else self.ledger.load_state_dict(v["ledger"], device=dev)

    def _slots(self):
        return self.proto_c, self.proto_cov, self.proto_w, self.w_fake_ema, self.proto_n, self.psi, self.v_push

    def _append(self, c, v, w, n, keep=1.0):
        pc, cov, pw, wf, pn, psi, vp = self._slots()
        self.proto_c, self.proto_cov = cat((pc, c)), cat((cov, cov_from_scalar(v, self.dim)))
        self.v_push, self.proto_n = cat((vp, v)), cat((pn, n))
        self.proto_w, self.w_fake_ema = cat((pw * keep, w)), cat((wf * keep, w))
        p = cat((psi, c.new_zeros(len(c))))
        self.psi = p - p.mean()

    @torch.no_grad()
    def merge_pair(self, a, b):
        c, cov, w, wf, n, psi, v = self._slots()
        wa, wb = w[a].double(), w[b].double()
        wt = (wa + wb).clamp_min(_E)
        m, vm, pm = ((wa * t[a].double() + wb * t[b].double()) / wt for t in (c, v, psi))
        va, vb = (cov_add_outer(cov[i].double().unsqueeze(0), (c[i].double() - m).unsqueeze(0))[0] for i in (a, b))
        c[a], cov[a] = m.float(), _san(((wa * va + wb * vb) / wt).unsqueeze(0).float())[0]
        v[a], psi[a], psi[b], w[a], w[b] = vm.clamp_min(_VF).float(), pm.float(), 0.0, wt.float(), 0.0
        for t in (wf, n):
            t[a], t[b] = t[a] + t[b], 0.0

    @torch.no_grad()
    def split_slot(self, parent, dst):
        c, cov, w, wf, n, psi, v = self._slots()
        c0, cp = c[parent].double(), cov[parent]
        _, dl, ch = cov_principal_split(cp, seed=parent if isinstance(parent, int) else 0)
        c[parent], c[dst], cov[parent], cov[dst] = (
            (c0 - dl.double()).float(),
            (c0 + dl.double()).float(),
            ch,
            ch.clone(),
        )
        v[parent] = v[dst] = (
            v[parent] * (cov_trace_mean(ch.unsqueeze(0))[0] / cov_trace_mean(cp.unsqueeze(0))[0])
        ).clamp_min(_VF)
        n[parent] = n[dst] = 0.5 * n[parent]
        for t in (w, wf, psi):
            t[parent] = t[dst] = 0.5 * (t[parent] + t[dst])

    def _init_batch(self, x):
        b, d, dev = x.shape[0], x.shape[1], x.device
        vx = x.var(0, unbiased=False)
        r = 2.0 * vx.sum()
        pt = not bool(r > _VF)
        if pt:
            c = x[:1].clone()
        else:
            u = x[_dedup(x, 1e-5 * r.sqrt())]
            if len(u) < b and len(u) <= self.max_prototypes:
                c = u.clone()
            else:
                s = u if len(u) < b else x
                m = len(s)
                k = min(self.max_prototypes, m if len(u) < b or m < 8 else max(4, (3 * m) // 4))
                p = (arange(1, d + 1, device=dev, dtype=_F32) * PHI1).remainder(1.0) - 0.5
                o = (s @ (p / p.norm().clamp_min(1e-8))).argsort(stable=True)
                c = s[o[((arange(k, device=dev) * m + m // 2) // k).clamp(0, m - 1)]].clone()
        k = len(c)
        idx = _cd(x, c).argmin(1)
        cnt = bincount(idx, minlength=k).float()
        v = vx.mean().clamp_min(_VF) / max(1, k)
        if k >= 2:
            dn = _cd(c, diag_inf=True).amin(1)
            m = dn > _E
            mp = where(m, dn, _INF).sort().values[(m.sum() // 2).clamp_max(k - 1)]
            v = where(m.sum() > 0, (0.04 * mp.square() / d).clamp_min(_VF), v)
        bm = x.new_zeros((k, d)).index_add_(0, idx, x) / cnt.clamp_min(1.0).unsqueeze(1)
        ve = (x.new_zeros(k).index_add_(0, idx, (x - bm[idx]).square().mean(1)) / (cnt - 1.0).clamp_min(1.0)).clamp_min(
            _VF
        )
        vi = (torch.quantile(ve[cnt >= 2.0], 0.25) if bool((cnt >= 2.0).any()) else v).expand(k).clone()
        self.dim, self.initialized, self._last_point_mass = d, True, pt
        self.proto_c, self.proto_cov, self.proto_w = c, cov_from_scalar(vi, d), _norm(cnt.clamp_min(_E))
        self.w_fake_ema, self.proto_n, self.psi, self.v_push = (
            self.proto_w.clone(),
            cnt.clamp_min(1.0),
            zeros(k, device=dev),
            vi.clone(),
        )
        self.ledger.reset()
        r0 = batch_annular_gap_ratio(x)
        self._rho_atom_t, self._g_conn_t = r0, compute_connectivity_gate(x, rho_atom=r0)

    def _stream(self, x):
        c, cov, pw, _, pn, _, vp0 = self._slots()
        c, cov = c.clone(), cov.clone()
        b, d, k, f = x.shape[0], x.shape[1], len(c), cov.ndim == 3
        v = cov_trace_mean(cov)
        dm, idx = _cd(x, c).min(1)
        d2, rsp = dm.square(), 2.0 * x.var(0, unbiased=False).sum()
        hsp, psp = rsp > _VF, rsp.clamp_min(_VF)
        pt = (0.5 * rsp) < (1e-11 * x.square().sum(1).mean()).clamp_min(1e-22)
        s = (
            (where(hsp, (d2.mean() / maximum(psp, 6.0 * d * v[idx].mean().clamp_min(_VF))).sqrt(), 0.0) - 0.45) / 0.35
        ).clamp(0.0, 1.0)
        vm = v.median().clamp_min(_VF)
        up = (
            hsp
            & (s <= 0.5)
            & (
                d2
                > maximum(
                    36.0 * d * maximum(cov_directional_var(cov[idx], (x - c[idx]) / dm.clamp_min(_E).unsqueeze(1)), vm),
                    0.008 * psp,
                )
            )
            if k < self.max_prototypes
            else zeros(b, device=x.device, dtype=_B)
        )
        m = (~up).float()
        cnt = bincount(idx, weights=m, minlength=k).float()
        hit, m2 = cnt > 0.0, cnt >= 2.0
        self.proto_n = ((1.0 - self.beta_n) * where(s > 0.25, pn * (1.0 - 0.8 * s), pn) + cnt).clamp_min(1e-6)
        aw = 0.02 + 0.48 * s * s
        ak = where((s > 0.25) & hsp & (~hit) & ((c - x.mean(0, keepdim=True)).square().sum(1) > 100.0 * psp), 0.50, aw)
        self.proto_w = _norm((1.0 - ak) * pw + ak * (cnt / m.sum().clamp_min(1.0)))
        bm = x.new_zeros((k, d)).index_add_(0, idx, x * m.unsqueeze(1)) / cnt.clamp_min(1.0).unsqueeze(1)
        ac = ((0.035 + 0.25 * self.proto_n.clamp_min(1.0).rsqrt() + 0.38 * s * s) * (cnt / (cnt + 2.0))).clamp_max(0.50)
        self.proto_c = where(hit.unsqueeze(1), (1.0 - ac.unsqueeze(1)) * c + ac.unsqueeze(1) * bm, c)
        ce = _san(
            _scov((x - where(m2.unsqueeze(1), bm, c)[idx]) * m.unsqueeze(1), idx, k, f),
            1.0 / (cnt - m2.float()).clamp_min(1.0),
        )
        ve, vt = cov_trace_mean(ce), (rsp / (2.0 * d)).clamp_min(_VF)
        vc, lv = hsp & (b >= 8) & (v > 4.0 * vt), v.log()
        vg = (0.15 * lv + 0.85 * (vt / max(1, k)).clamp_min(_VF).log()).exp().clamp_min(_VF)
        cb = _san(cov, where(vc, vg / v, 1.0))
        vj = hit & (
            (s > 0.5)
            | (m2 & ((ve < 1e-8 * v) | (ve > 25.0 * v)))
            | (ve <= 1e-12 * v)
            | ((cnt >= 6.0) & (ve < 0.04 * v))
        )
        lr = ve.log() - lv
        sc = ((0.65 + 0.25 * maximum(s * s, _ss((lr.abs() - 3.2) / 3.2))).clamp_max(0.90) * lr).exp()
        a = _bc(0.12 * cnt / (cnt + 1.0), f)
        self.proto_cov = _san(where(_bc(hit, f), (1.0 - a) * where(_bc(vj, f), _san(cov, sc), cb) + a * ce, cb))
        self.v_push = where(vj, (v * sc).clamp_min(_VF), where(vc, vg, vp0))
        return idx, d2, v, hsp, psp, aw, up, pt

    def _rebalance(self, dead, df, pi, mv, g, mw):
        c, cov, w, _, _, _, _ = self._slots()
        k, ik = len(c), arange(len(c), device=c.device)
        if not df:
            m = ik == pi
            dm, fl = _cd(c, diag_inf=True).square().masked_fill(m[:, None] | m[None, :], _INF).reshape(-1).min(0)
            if not bool(dm <= 8.0 * self.dim * mv):
                return
            self.merge_pair(fl // k, fl % k)
            dsts = (fl % k,)
        else:
            dsts = tuple(dead.nonzero().squeeze(-1)[:4])
        for dst in dsts:
            p = ((w.clamp_min(_E) ** g) * cov_trace_mean(cov)).masked_fill(ik == dst, 0.0).argmax()
            if not bool((p != dst) & (w[p] > 0.20 * mw)):
                break
            self.split_slot(p, dst)

    def _spawn(self, x, wj, pm, qm, psp, aw, up, mw, pf, df):
        c, cov, w, wf, n, psi, v = self._slots()
        k, d = len(c), self.dim
        if pf:
            u = x[up]
            nc = u[_dedup(u, maximum(36.0 * d * pm, 0.008 * psp).sqrt())][: min(self.max_prototypes - k, 64)]
            if len(nc) > 0:
                ws = min(0.5 / len(nc), 1.0 / (k + len(nc)))
                self._append(
                    nc,
                    pm.expand(len(nc)),
                    nc.new_full((len(nc),), ws),
                    bincount(_cd(u, nc).argmin(1), minlength=len(nc)).float().clamp_min(1.0),
                    1.0 - len(nc) * ws,
                )
            return
        nw = (aw / len(x)).clamp_min(0.05 * mw)
        if not df and k < self.max_prototypes:
            self._append(x[wj].unsqueeze(0), qm.reshape(1), nw.reshape(1), ones(1, device=c.device))
            return
        sl = w.argmin() if df else _cd(c, diag_inf=True).square().argmin()
        if not df:
            self.merge_pair(sl // k, sl % k)
            sl = sl % k
        c[sl], cov[sl], v[sl], w[sl], n[sl], wf[sl], psi[sl] = (
            x[wj],
            cov_from_scalar(qm.reshape(1), d)[0],
            qm,
            nw,
            1.0,
            nw,
            0.0,
        )

    def _maintain(self, x, idx, d2, v, hsp, psp, aw, up, pt):
        c, cov, w, _, _, _, _ = self._slots()
        k, d = len(c), self.dim
        mw, vc = 1.0 / k, cov_trace_mean(cov)
        pm, qm = v.median().clamp_min(_VF), vc.median().clamp_min(_VF)
        dead, g = w < (0.04 / k), min(d, 4) / (min(d, 4) + 2)
        mur, wj = (d2 / maximum(36.0 * d * maximum(v[idx], pm), 0.008 * psp)).max(0)
        ge = (w.clamp_min(_E) ** g) * vc
        me, pi = ge.max(0)
        du, ad = (mur > 1.0) & hsp, dead.any()
        ng = (
            (me > 10.0 * ge.median().clamp_min(_VF)) & (w[pi] > 0.25 * mw)
            if k >= 16
            else zeros((), device=x.device, dtype=_B)
        )
        ds = (~du) & hsp & (ad | ng)
        ptf, pf, uf, sf, df = stack((pt, up.any(), du, ds, ad)).tolist()
        self._last_point_mass = bool(ptf)
        if pf or uf:
            self._spawn(x, wj, pm, qm, psp, aw, up, mw, pf, df)
        elif sf:
            self._rebalance(dead, bool(df), pi, pm, g, mw)
        if pf or uf or sf:
            _, _, w2, wf2, _, p2, _ = self._slots()
            self.proto_w, self.w_fake_ema, self.psi = _norm(w2), _norm(wf2), p2 - p2.mean()

    @torch.no_grad()
    def update(self, y_real):
        _req(y_real.ndim >= 2 and len(y_real) >= 1)
        x = _flat(y_real)
        _req(x.shape[1] >= 1)
        self.device = x.device
        for nm in _ALL_T:
            t = getattr(self, nm)
            if isinstance(t, Tensor) and t.device != x.device:
                setattr(self, nm, t.to(x.device))
        if not self.initialized or self.proto_c is None or self.dim != x.shape[1]:
            self._init_batch(x)
            return
        self._maintain(x, *self._stream(x))

    @torch.no_grad()
    def coalesce(self, y_real=None):
        if not self.initialized or self.proto_c is None:
            raise RuntimeError("atlas not initialized")
        c = coalesce_directional_ltc(self.proto_c, self.proto_cov, self.proto_w, self.proto_n)
        r = compute_rho_atom(c.comp_c, c.comp_cov, c.comp_n_eff, y_real=y_real)
        self._rho_atom_t = (
            0.85 * (self._rho_atom_t if self._rho_atom_t is not None else r.new_full((), self._rho_atom_f)) + 0.15 * r
        )
        if y_real is not None:
            g = compute_connectivity_gate(y_real, rho_atom=self._rho_atom_t)
            self._g_conn_t = (
                0.85 * (self._g_conn_t if self._g_conn_t is not None else g.new_full((), self._g_conn_f)) + 0.15 * g
            )
        return c

    @torch.no_grad()
    def commit_particles(self, x_fake, rows, num_particles, *, coalesced=None, allow_update=True):
        if not self.initialized or not (1 <= int(num_particles) <= MAX_LEDGER_PARTICLES):
            return None
        c = self.coalesce() if coalesced is None else coalesced
        if len(c.comp_c) < 2 or not bool(c.all_iso):
            return None
        return self.ledger.update(x_fake, rows, c.comp_c, c.comp_w, int(num_particles), allow_update=allow_update)


__all__ = [
    "CoalescedAtlas",
    "HamiltonCommitmentLedger",
    "LedgerAssignment",
    "MAX_FULL_COV_DIM",
    "MAX_LEDGER_PARTICLES",
    "PrototypeAtlas",
    "SCHEMA_VERSION",
    "batch_annular_gap_ratio",
    "coalesce_directional_ltc",
    "coalesce_pythagorean_ltv",
    "compute_connectivity_gate",
    "compute_rho_atom",
    "cov_add_outer",
    "cov_directional_var",
    "cov_from_scalar",
    "cov_principal_split",
    "cov_shrink_mp",
    "cov_trace_mean",
    "manifold_connectivity_stats",
    "validate_atlas_state",
    "validate_ledger_state",
]
