"""Axiom L2 Law 3A transport operator."""

from math import e as _E, log, pi, sqrt
from weakref import WeakKeyDictionary as _WKD

from torch import (
    Tensor,
    arange,
    bincount,
    bool as _B,
    cat,
    eye,
    float32 as _F32,
    float64 as _F64,
    is_grad_enabled,
    long as _L,
    nn,
    no_grad,
    quantile as _q,
    searchsorted,
    softmax,
    stack,
    where,
)
from torch.nn.functional import pad as _pd, relu as _rl, softplus as _sp

from .atlas import (
    MAX_LEDGER_PARTICLES as _MAX_P,
    PrototypeAtlas,
    batch_annular_gap_ratio as _ann_gap,
    cov_trace_mean as _cov_tr,
    manifold_connectivity_stats as _mconn,
)
from .fiber import AxiomCondFiberBank, conditional_fiber_loss
from .math_ops import (
    PHI1,
    PHI2,
    born_infeld_cost as _bi,
    log_sinkhorn as _sink,
    power_principal_axis as _ppa,
    safe_cdist as _cd,
    sinkhorn_with_slack as _sink_sl,
    smoothstep01 as _ss,
    sym_eigensplit_halley,
)
from .state import recover_active_prior_rows as _rec_rows, set_active_prior_rows

_INF, _ha, _ga = float("inf"), hasattr, getattr
_AK = "max_prototypes beta_n initialized dim proto_c proto_cov proto_w w_fake_ema proto_n psi v_push".split()
_LM = "num_particles dim row_comp row_target row_dist".split()
_IK = frozenset("iters chart_capacity max_spatial max_flat_dim".split())
_CK = (
    "weight rho tau eps iters chart_capacity chart_weight halo_mult max_spatial max_flat_dim "
    "mean_op_weight cov_op_weight imq_active_scale bank_mean_blend"
)
_CFG = dict(zip(_CK.split(), (1.0, 3.0, 2.0, 0.1, 8, 256, 1.4, 2.1, 8, 256, 0.36, 0.35, 0.4, 0.75)))
_FK = tuple(_CFG)[-4:]


def _f(t):
    return t.float()


def _dt(t):
    return t.detach()


def _d6(t):
    return t.double()


def _p(t, e=1e-30):
    return t.clamp_min(e)


def _s(x, a=0, b=1):
    return _ss((x - a) / b)


def _n(x):
    return x.norm(dim=-1)


def _sq(x):
    return (x**2).sum(-1)


def _di(d):
    return d.clone().fill_diagonal_(_INF)


def _r2(t):
    return _f(t.reshape(len(t), -1))


def _ar(n, d, dt=None):
    return arange(n, device=d, dtype=dt) if dt is not None else arange(n, device=d)


def _ey(n, d):
    return eye(n, dtype=_B, device=d)


def _tk(x, k, d=1):
    return x.topk(k, d, largest=False)


def _lu(t, n):
    return t.new_full((n,), -log(n))


def _pa(m):
    return _ppa(m, iters=6, seed=0)


def _b1(a, b, e, f64=False):
    return _bi(a, b, e, aligned=True, fp64=f64)[0]


def _sub(n, m, d, p=0.0):
    return ((_ar(m, d) * n) // m + int(p * n)) % n


def _gm(a, b):
    return (a[:, None] * b[None]).sqrt()


def _n1(t, d=None):
    return t / _p(t.sum() if d is None else t.sum(d, keepdim=True), 1e-12)


def _pq(d, m, k):
    v = where(m, d, _INF).reshape(-1).sort().values
    return v[(m.sum() // k).clamp_max(v.numel() - 1)]


def _sg(t, s):
    return (t * s + _dt(t) * (1 - s)) if (s < 1 and t.requires_grad) else t


def _ts(r):
    return float(_ga(r, "thermal_scale", 1.0))


def _mut(rec, allow=True):
    return allow and is_grad_enabled() and not (_ga(rec, "_in_probe", False) or _ga(rec, "_in_protected_probe", False))


def _flat(t):
    return t.reshape(len(t), -1).mean(1) if t.ndim > 1 else t.reshape(-1)


def _log_odds(a, b, c=20.0):
    a, b = _flat(a), _flat(_dt(b))
    m = 0.5 * (_d6(_dt(a)).mean() + _d6(b).mean())
    return c * ((_d6(a) - m).to(a.dtype) / c).tanh(), c * ((_d6(b) - m).to(b.dtype) / c).tanh()


def _dmask(a, b, m, n):
    if a is None or b is None or len(a) != m or len(b) != n:
        return None
    u, v = _dt(a).reshape(m, -1), _dt(b).reshape(n, -1)
    return (u[:, None] == v[None]).all(-1) if u.shape[1] == v.shape[1] else None


def _vptab(p, r, b):
    return p is not None and r is not None and p.ndim == 2 and r.ndim == 1 and len(r) == b and 1 <= len(p) <= _MAX_P


def _fv(v):
    return v.item() if isinstance(v, Tensor) else float(v)


def _ema(p, c):
    a = 0.1 + 0.7 * ((c / _p(p)).maximum(p / _p(c)) > 8).to(c.dtype)
    return p.lerp(c, a)


def project_features(x, max_spatial=8, max_flat_dim=256):
    b = len(x)
    if x.ndim >= 4 and (x.shape[-2] > max_spatial or x.shape[-1] > max_spatial):
        h, w = x.shape[-2:]
        r, c = min(max_spatial, h), min(max_spatial, w)
        z = _f(x.reshape(-1, x.shape[-3], h, w))
        p, q = (-h) % r, (-w) % c
        if p or q:
            z = _pd(z, (0, q, 0, p), "replicate")
        z = z.reshape(len(z), z.shape[1], r, (h + p) // r, c, (w + q) // c).mean((3, 5)).reshape(b, -1)
        z = z * sqrt((h * w) / max(1, r * c))
    else:
        z = _r2(x)
    d, c = z.shape[1], max(1, max_flat_dim)
    if d > c:
        g = (d + c - 1) // c
        if g * c > d:
            z = _pd(z, (0, g * c - d), "replicate")
        z = z.reshape(b, g, c).mean(1) * sqrt(d / c)
    return z.to(x.dtype)


project_transport_features = project_features


@no_grad()
def compute_contrast_confidence(y_feat, *, max_b=64):
    z = _r2(y_feat)
    b = len(z)
    if b < 4:
        return z.new_ones(())
    if b > max_b:
        z, b = z.index_select(0, _sub(b, max_b, z.device)), max_b
    d = _cd(z)
    a, m, v = d.reshape(-1).sort().values, b * (b - 1), _di(d).amin(1).sort().values
    ma, mn = _p(0.5 * (a[b + (m - 1) // 2] + a[b + m // 2])), 0.5 * (v[(b - 1) // 2] + v[b // 2])
    return where(a[-1] <= 1e-24, 1.0, _s(0.65, mn / ma, 0.35))


@no_grad()
def compute_discrete_atomic_gate(y_cf, *, charts=None, radii=None, record=None):
    y = _r2(y_cf)
    if len(y) < 8:
        return 0.0
    if charts is not None and len(charts) >= 4:
        c = charts.to(y.device, _F32)
        m = _q(_cd(c, diag_inf=True).amin(1), 0.5)
        q = _q(_f(radii), 0.75) if radii is not None else y.new_zeros(())
        if ((m > 1e-12) & (_q(_cd(y, c).amin(1), 0.75) < 1e-3 * m) & (q < 1e-3 * m)).item():
            return 1.0
    s = _ga(record, "chart_state", None)
    if isinstance(s, dict) and s.get("initialized") and "rho_atom_ema" in s and _fv(s["rho_atom_ema"]) < 1e-3:
        return 1.0
    d = _cd(y)
    m = d.amax()
    t, u = 1e-6 * m, _di(d).amin(1)
    a, s = _f(u <= t).sum(), d > t
    r = _f((u > t) & (u < (10 * t).maximum(0.25 * _pq(d, s, 20)))).sum()
    return where((m > 1e-24) & (a >= 2) & s.any(), _s(a / _p(a + r, 1), 0.7, 0.2), 0.0).item()


@no_grad()
def _batch_ell(y, q=0.25):
    b = len(y)
    if b <= 1:
        return y.new_tensor(1.0, dtype=_F32)
    a = max(0.0, 1 / max(1, y.shape[-1]) - 0.5)
    f = ((2 * _E * b) / max(1e-6, -log(max(1e-6, 1 - q)))) ** a if a > 0 else 1.0
    d = _cd(y)
    m = d.amax()
    t = 1e-5 * m
    p, r = where(d > t, d, _INF).amin(), _q(d.fill_diagonal_(_INF).amin(-1), q)
    return where(m <= 0, 1e-3, _p(where(r <= t, where(p.isfinite(), 0.15 * p, 0.05 * m), r * f), t))


@no_grad()
def _valley(y, d):
    j = d.topk(min(len(y) - 1, 2), -1).indices[:, -1]
    m, i = _cd(0.5 * (y + y[j]), y), _ar(len(y), y.device)
    m[i, i] = m[i, j] = _INF
    return _q(m.amin(-1), 0.85)


@no_grad()
def _conformal_horizon(x, y, e, g, cp):
    if g > 0.5 or cp or len(x) < 12 or len(y) < 12:
        return e, e, None, None, None
    b = _p(_f(e))
    f, d = 1e-6 * b, _cd(y)
    a, q, c = _p(_tk(_di(d), 3).values, f).unbind(1)
    r = where((q <= 2.8 * a) & (c <= 2.8 * q), q, a)
    p, i = _tk(d, 8)
    m = _f(p <= (3.2 * q)[:, None])
    m[:, 0] = 1
    r = (r[i] * m).sum(1) / _p(m.sum(1), 1)
    w = _s(_q(r, 0.85) / _p(_q(r, 0.15), f), 2.85, 0.85)
    if w.item() <= 1e-4:
        return e, e, None, None, d
    s = b.lerp((b * (r / _p(_q(r, 0.25), f))).clamp(0.7 * b, 4.5 * b), w)
    h = _p(2 * (2.4 * s) ** 2)[None]

    def u(m):
        return _p(softmax(-(m**2) / h, 1) @ s, f)

    v, z = u(d), u(_cd(x, y))
    return _gm(z, v).to(e.dtype), _gm(z, z).to(e.dtype), z, v, d


@no_grad()
def _hetero(x, y, e, a, b, d, w, g, rec):
    d = d.clone() if d is not None else _cd(y)
    o, c = d.clone().fill_diagonal_(0), _p(_f(e))
    w = w.maximum(_s(_valley(_f(y), o) / (2 * c), 2.55, 1.5) * (1 - g))
    p = o > 0
    m = where(p.any(), _p(_pq(o, p, 4)), c)
    h = _p(1.15 * m * (b / c))[:, None] if b is not None else (1.15 * m)
    r = (-(o**2) / _p(2 * h**2)).exp().sum(1)
    u = (w * (_q(r, 0.85) / _p(_q(r, 0.03), 1))).clamp(0, 20)
    v = u
    if _ha(rec, "het_initialized"):
        v = (0.92 * float(rec.het_ema) + 0.08 * u) if rec.het_initialized else u.clamp_max(3)
    q, o = _s(v, 3.4), _di(o)
    k, j = o.amin(1), _tk(o, min(2, len(y) - 1)).values[:, -1]
    s = q * _s(j / c, 2.8, 1.4)
    b = (b if b is not None else c.expand(len(y))).maximum(c + s * (0.64 * j - c).clamp(0, 3.4 * c))
    z, i = _cd(x, y).min(1)
    t = b[i]
    n = s[i] * _s(2.8 * t, z, _p(1.2 * t))
    a = (a if a is not None else c.expand(len(x))).maximum(c + n * (t - c)).to(e.dtype)
    b = b.to(e.dtype)
    return _gm(a, b), _gm(a, a), b, q, k, n, w, v, d


def _outlier(a, b, e, k, d, s, g, p, r):
    with no_grad():
        e, c = _p(_f(e)), _f(b)
        m = c.amin(1)
        t = _p(2.1 * _q(((e**2 + k**2).sqrt() - e) / e, 0.98), 4.2 * r)
        o = g * _s(_f(p).sum(), 0.72, 0.12) * _s(m, t, 0.4 * t) * ((-_f(d) - 0.35) / 0.2).sigmoid() * _f(s) ** 4
        w = softmax(-(c - m[:, None]) / 0.25, 1).to(a.dtype)
    return 2.2 * ((o.to(a.dtype) / len(a))[:, None] * w * a).sum()


def _orphan_macro_loss(x, y, c, r, s, *, rho, g_conn):
    m, n = len(x), len(y)
    if m < 2 or n < 2 or g_conn > 0.05:
        return x[:0].sum()
    with no_grad():
        g, d, v = _s(_f(c).amin(0), 2.8 * rho, max(1.2 * rho, 1e-6)), _cd(x, y), _f(y)
        if n >= 8 and m >= 8:
            u = _cd(v)
            o, k = _di(u), min(4, n - 1, m)
            g = g * _s(_tk(d, k, 0).values[-1] / _p(_tk(o, k).values[:, -1]), 3.52, 1.3)
            t = _valley(v, u)
            w, h = _s(t / _p(_q(o.amin(1), 0.5)), 3.2, 1.8), _p(0.38 * t)

            def f(z):
                return (-0.5 * (z / h) ** 2).exp()

            b, a = f(u).mean(1).clamp(1e-4, 1), f(d).mean(0)
            g = g * ((1 - w) + w * _s(m * b, 4.2, 2.3) * _s(0.45, a / b, 0.2))
        q = (n * _f(s)).clamp(0, 1) * g
        t = q.mean().item()
        k = m if t >= 0.85 else min(m - 1, int(round(m * t)))
        if t <= 1e-4 or k < 1:
            return x[:0].sum()
        h, a = _p(_q(d.reshape(-1), 0.5)), _f(g > 0.05)
        u = where((a > 0).any(), (d + (1 - a)[None] * (1e6 * h)).amin(1), d.amin(1))
        if k == m:
            i = _ar(m, x.device)
        else:
            z = 1.5 * (m * _f(r)).clamp(0, 1) - 0.65 * (u / h).clamp_max(4)
            i = (z + 1e-4 * (((_ar(m, x.device, _F32) + 0.5) * PHI1) % 1)).topk(k).indices.sort().values
        q = _n1(q + 1e-7)
        u = (v - (q[:, None] * v).sum(0)) * q.sqrt()[:, None]
        o = (v @ _f(_pa(_d6(u.T @ u))[1]) if v.shape[1] >= 2 else v[:, 0]).argsort()
        p = _sink(-(d[i][:, o] / h) / 0.03, _lu(d, k), _p(q[o], 1e-12).log(), iters=10).exp_()
        c = _n1(p, 1).cumsum(1)
        c[:, -1] = 1
        j = _ar(k, x.device)
        u = ((_f(j // 2 + (j % 2) * ((k + 1) // 2)) + 0.5) / k).clamp(1e-6, 1 - 1e-6)
        v = _dt(y)[o[searchsorted(c, u[:, None])[:, 0].clamp(0, n - 1)]]
    return _b1(x[i], v, _p(0.25 * h)).sum() / m


@no_grad()
def _res(x, y, e, g=0.58):
    x, y = _r2(x), _r2(y)
    m, n = len(x), len(y)
    a, b, c = _cd(x, y), _cd(x), _cd(y)

    def u(d, k):
        return d.sum() / max(1, k * (k - 1))

    p, q, r = a.mean(), u(b, m), u(c, n)
    h = _p(0.5 * (p + 0.5 * (q + r)))
    o, s, k = (2 * p - q - r) / h, (q - r) / h, min(m, n)

    def v(t):
        return t.var(0).sum() / len(t) if len(t) > 1 else 0.0

    t = (((x.mean(0) - y.mean(0)) ** 2).sum() - v(x) - v(y)) / h**2
    if k < 4:
        return o, 0.16 * t, s, o, o
    p, q, r = _di(a[:k, :k]), _di(c[:k, :k]).amin(1), _p(_f(e)) / h

    def f(z):
        return (0.15 + 5.5 * r).clamp(0.15, z) ** 2 * t

    u = z = f(0.4)
    if g != 0.4 and n >= 4:
        a, d, i = _di(c).amin(1).mean(), _cd(0.5 * (y + y.roll(1, 0)), y), _ar(n, y.device)
        d[i, i] = d[i, (i + 1) % n] = _INF
        w = where(a > 1e-30, _s(1.15, d.amin(1).mean() / _p(a), 0.2), 0.0)
        b = f(0.4 + w * (g - 0.4))
        z = where(a > 1e-30, where(t >= 0, u.maximum(b.clamp_max(0.045)), b), u)
    return o, z, s, 10.2 * (_q(p.amin(1), 0.75) - _q(q, 0.75)) / h, (_q(p.amin(0), 0.9) - _q(q, 0.9)) / h


def _split_j(x, y, t, r):
    m, n = len(x), len(y)
    if not _vptab(t, r, m) or m < 4 or n < 4:
        return None
    d, x, y = t.shape[1], x.reshape(m, -1), y.reshape(n, -1)
    if x.shape[1] <= d or y.shape[1] != x.shape[1]:
        return None
    r = r.to(t.device, _L)
    z = _dt(t)[r.clamp(0, len(t) - 1)].to(x.device, _F32)

    def e(u):
        return (_f(_dt(u)) - z).abs().amax()

    o, a, b = stack((_f((r < 0).any() | (r >= len(t)).any()).to(x.device), e(x[:, -d:]), e(x[:, :d]))).tolist()
    if o > 0.5:
        return None
    if a <= 1e-5:
        return x[:, :-d], x[:, -d:], y[:, :-d], y[:, -d:]
    return (x[:, d:], x[:, :d], y[:, d:], y[:, :d]) if b <= 1e-5 else None


def _hbi(x, y, r, e):
    c = _p(_f(_dt(e)))
    u = _rl((_sq(_f(x) - _f(_dt(y))) + _p((1e-12 * c) ** 2, 1e-36)).sqrt() - _f(_dt(r)))
    return (u**2 / _p((c**2 + u**2).sqrt() + c)).mean().to(x.dtype)


@no_grad()
def _curve_gate(y, g):
    if g <= 1e-4 or len(y) < 12:
        return 0.0
    if y.shape[1] < 2:
        return g
    p = _d6(_f(y)[_tk(_cd(y), min(9, len(y)), -1).indices])
    c = p - p.mean(1, keepdim=True)
    m = (c.mT @ c) if c.shape[2] < c.shape[1] else (c @ c.mT)
    l1, _ = _pa(m)
    t = _p(m.diagonal(dim1=-2, dim2=-1).sum(-1))
    return g * _s(0.11, (_p(t - l1, 0) / _p(l1)).median(), 0.06).item()


@no_grad()
def _simplicial_gates(y, w):
    if w <= 0 or len(y) < 4:
        return 0.0, 0.0, 0.1, _p(_f(y).abs().amax(), 1e-12), 0.0
    g, m, q, s = _mconn(y)
    a = _s(_ann_gap(y), 0.15, 0.2)
    g, m, q = stack((g * a, m * a, q)).tolist()
    return g, m, q, s, _curve_gate(y, g)


@no_grad()
def _bary(z, o, n, g):
    m, d, t = len(z), z.device, 2 * pi
    c = (_d6(_cd(_d6(z))) + 1e-12 * _ar(m, d, _F64)[None]).fill_diagonal_(_INF)
    k = min(4, m - 1)
    v = _tk(c, k, -1).indices
    b = (n // 2) if (m >= 3 and k >= 2 and n >= 4) else n
    j = int(round((1 - g) * (n - b)))
    h, r, q = n - j, _ar(n - j, d), min(3, k)

    def f(i, c, s):
        return ((_f(i) + 0.5) / max(1, c) + s) % 1

    w = max(1, m * q) if g > 0.5 else m
    p = (r + int(o * w)) % w
    a, b = (p // q, v[p // q, p % q]) if g > 0.5 else (p, v[p, r % q])
    u = f(r // w, (h + w - 1) // w, o) if g > 0.5 else f(r, h, o)
    e = (0.2 + 0.6 * u).to(z.dtype)
    x = z[a].lerp(z[b], e[:, None])
    if j <= 0:
        return x, h, a, b, e, None
    r, o = _ar(j, d), (o * PHI2) % 1
    i, c = (r + int(o * m)) % m, r % k
    u, w = v[i, c], v[i, (c + 1 + ((r // k) % max(1, k - 1))) % k]
    p, q = ((1 / 3 + 0.12 * _f(r % 2) * (t * f(r, j, o) + s).cos()).to(z.dtype)[:, None] for s in (0, t / 3))
    y = p * z[i] + q * z[u] + (1 - p - q) * z[w]
    return cat((x, y)), h, a, b, e, (i, u, w, y)


def _simp_iso(x, v, z, e, g, s, f):
    _, h, a, b, t, r = s
    p, q, t = _b1(x[:h], v[a], e, f), _b1(x[:h], v[b], e, f), t.to(x.dtype)
    o = ((1 - t) * p - t * q).abs().mean()
    if not r:
        return o
    *i, y = r
    u = stack([_b1(x[h:], v[j], e, f) for j in i])

    def d(p):
        return stack((_n(p[0] - p[1]), _n(p[1] - p[2]), _n(p[2] - p[0])))

    with no_grad():
        c, e = d([_d6(v[j]) for j in i]), _d6(_p(e))
        p = 0.5 * c.sum(0)
        q = _p(p * _p(p - c, 0).prod(0), 0).sqrt()
        m = 0.7 * ((e**2 + (q / _p(p, 1e-8)) ** 2).sqrt() - e)
        if g > 1e-4:
            k, w = [_d6(z[j]) for j in i], _d6(y)
            m = (1 - g + g * _s(stack([_n(w - j) for j in k]).amin(0) / _p(d(k).amax(0), 1e-8), 0.12, 0.1)) * m
    return 0.5 * (o + _rl(m.to(u.dtype) - u).mean())


def _simp_chart(z, x, v, y, e, g, r, q, s, o, w, f):
    m, n, u, lo = len(z), len(y), _f(y) / s, e.new_zeros((), dtype=v.dtype)
    if x is not None:
        c, _ = _bi(x, y, e, fp64=f)
        with no_grad():
            a = -(_f(c) / _p(_f(c).mean(), 1e-12)) / 0.08
            p = _sink(a, _lu(a, m), _lu(a, n), iters=10).exp_().to(c.dtype)
        lo = lo + (1.1 * g) * (p * c).sum()
        if r > 1e-4:
            lo = lo + (0.65 * g * r) * _b1(x, y[(_f(p) @ _cd(u)).argmin(1)], 0.5 * e, f).mean()
    if not z.requires_grad:
        return lo
    with no_grad():
        a = u if (m + n) <= 512 else u[_sub(n, min(n, 128), u.device, o)]
        b = cat((_f(v) / s, a))
        d = _d6(_cd(_d6(b) if f else b))
        a = d * (1 + (d / max(1e-4, 1.35 * q)) ** 2)
        for _ in range(6):
            a = a.minimum((a[:, :, None] + a[None]).amin(1))
        a, k = a[:m, :m], ~_ey(m, z.device)
        a = a / _p((a[k] ** 2).mean(), 1e-12).sqrt()
        b, c = _n1(where(k, (0.2 + a**2) ** -2, 0.0)), _n1(where(k, 1 + 2 * a**2, 0.0))
        b = b if r <= 1e-4 else b.lerp(c, r)
    p = _d6(z)
    c = _sq(p)
    d = c[:, None] + c[None] - 2 * (p @ p.T)
    d = d + (_p(_dt(d), 0) - _dt(d))
    c, h = (d + 1e-12).sqrt() / _p(d[k].mean(), 1e-12).sqrt(), _dt(e).to(lo.dtype)
    lo = lo + (g * w * (1 + 18 * r)) * h * (b * (c - a) ** 2).sum().to(lo.dtype)
    if r > 1e-4 and z.shape[1] >= 2:
        p = p - p.mean(0)
        q = _p((p**2).sum(), 1e-12)
        with no_grad():
            _, a = _pa(p.T @ p)
        lo = lo + (3.2 * g * r * w) * h * (_p(q - ((p @ a) ** 2).sum(), 0) / q).to(lo.dtype)
    return lo


def _obs_th(x, y, e, aux, disc, simp, cp, rec):
    w, v, d, r, h = aux
    s = simp or bool(_ga(rec, "_simplicial_step", False))
    m, b = min(len(x), len(y)), v is not None and _ha(rec, "het_initialized")
    if not s and not cp and m >= 4 and _ha(rec, "observe_thermal"):
        with no_grad():
            q = _res(x, y, e, 0.58)
            if not disc and not h and m >= 8:
                o = ~_ey(len(y), y.device)
                g = _q(_cd(x, y).amin(1), 0.96) - _q(d.masked_fill(~o, _INF).amin(1), 0.96)
                r = r.maximum(w * 1.2 * g / _p(d[o].mean()))
            u = stack([*q, r, v] if b else [*q, r]).tolist()
            if b:
                rec.het_ema, rec.het_initialized = u[6], True
            rec.observe_thermal(*u[:5], r_tail=u[5], shock_mu_ratio=0.35)
    elif b:
        rec.het_ema, rec.het_initialized = v.item(), True


def _obs_simp(v, x, y, e, m, g, rec):
    if m <= 1e-4 or len(y) < 4 or not _mut(rec) or not _ha(rec, "observe_thermal"):
        return
    with no_grad():
        r = stack(_res(cat((v, _dt(x))), y, e)).tolist()
        k = dict(tau=0.084 * (1 + 0.6 * g), shock_mu_ratio=0.35 + 0.55 * g, shock_ed_floor=0.3 + 0.5 * g)
        rec.observe_thermal(*r[:3], (1 - 0.5 * g) * r[3], r[4], lock_hysteresis=0.22 * g, **k)


class AxiomTransportLoss(nn.Module):
    """Unified Law 3A + Law 3B transport operator."""

    def __init__(self, **kw):
        super().__init__()
        b = set(kw) - set(_CFG)
        if b:
            raise TypeError(f"bad kwargs: {sorted(b)}")
        for k, v in _CFG.items():
            x = kw.get(k, v)
            setattr(self, k, max(2, int(x)) if k == "chart_capacity" else (int(x) if k in _IK else float(x)))
        self._discrete_step = self._simplicial_step = False
        self._ell_local = self.last_p_xy_slack = self._atlas_local = None
        self._rec_atlases = _WKD()

    def bind_discrete_prior_step(self, active, *, simplicial_active=False):
        self._discrete_step, self._simplicial_step = bool(active), bool(simplicial_active)

    def _proj(self, x):
        return project_features(x, self.max_spatial, self.max_flat_dim)

    def _sink_sl(self, c, a, b, m):
        return _sink_sl(c, a, b, rho=self.rho, tau=self.tau, eps=self.eps, iters=self.iters, mask=m)

    def _ell(self, y, rec, mut):
        with no_grad():
            e = _batch_ell(y, 0.25)
            if _ha(rec, "ell_initialized"):
                h = _ha(rec, "dev_ema") and _ha(rec, "_ema_t")
                p = _f(rec.dev_ema("ell_ema", e.device)) if h else e.new_tensor(float(rec.ell_ema))
                c = _ema(p, e) if rec.ell_initialized else e
                if not mut:
                    return (p if rec.ell_initialized else e).to(y.dtype)
                rec.ell_initialized = True
                if h:
                    rec._ema_t["ell_ema"] = _d6(c)
                else:
                    rec.ell_ema = c.item()
                return c.to(y.dtype)
            if mut:
                self._ell_local = e.clone() if self._ell_local is None else _ema(self._ell_local.to(y.device), e)
            return (self._ell_local.to(y.device) if self._ell_local is not None else e).to(y.dtype)

    def _step_at(self, y, rec, mut):
        d = y.device
        if rec is None:
            at = self._atlas_local = self._atlas_local or PrototypeAtlas(max_prototypes=self.chart_capacity, device=d)
        else:
            c, s = self._rec_atlases.get(rec), _ga(rec, "chart_state", None)
            if c and c[1] is s:
                at = c[0]
            else:
                ok = isinstance(s, dict) and s.get("schema") == 2 and "proto_c" in s
                m = int(s.get("max_prototypes", self.chart_capacity)) if ok else self.chart_capacity
                at = PrototypeAtlas(max_prototypes=m, device=d)
                if ok:
                    at.load_state_dict(
                        {**s, **{k: _fv(s[k]) for k in ("rho_atom_ema", "g_conn_ema") if k in s}}, device=d
                    )
                self._rec_atlases[rec] = (at, s)
        if mut or not at.initialized:
            at.update(y)
        return at, at.coalesce(y if mut else None)

    def _commit_at(self, x, y, t, r, mk, rec, mut):
        at, co = self._step_at(y, rec, mut)
        rh = nn = asg = None
        k, d = co.comp_c.shape
        if k >= mk:
            nn = _p(_cd(co.comp_c, diag_inf=True).amin(1)) if k >= 2 else co.comp_c.new_ones(k)
            rh = (self.halo_mult * _p(d * _cov_tr(co.comp_cov), 0).sqrt()).minimum(0.35 * nn)
            if _vptab(t, r, len(x)):
                asg = at.commit_particles(x, r, len(t), coalesced=co, allow_update=mut)
        if mut and _ha(rec, "chart_state"):
            lg = at.ledger
            ld = {"schema": 2, "initialized": True, **{u: _ga(lg, u) for u in _LM}} if lg.initialized else None
            a = at._rho_atom_t if at._rho_atom_t is not None else float(at._rho_atom_f)
            g = at._g_conn_t if at._g_conn_t is not None else float(at._g_conn_f)
            s = {"schema": 2, **{u: _ga(at, u) for u in _AK}, "rho_atom_ema": a, "g_conn_ema": g, "ledger": ld}
            rec.chart_state = s
            if _ha(rec, "atomic_ledger"):
                rec.atomic_ledger = ld
            self._rec_atlases[rec] = (at, s)
        return at, co, asg, rh, nn

    def _route(self, x, y, xf, e, g, rec, mut):
        z = e.new_zeros((), dtype=_F32)
        t, r = _rec_rows(xf, rec)
        at, co, asg, rh, nn = self._commit_at(x, y, t, r, 2, rec, mut)
        if rh is None:
            return None, None, z, z, None
        with no_grad():
            c = _p(_f(e))
            p = _q(_cd(0.5 * (y + y.roll(1, 0)), co.comp_c).amin(1), 0.75)
            q = _q(_cd(y, co.comp_c).amin(1), 0.75).maximum(c)
            w = _s(p / q, 0.85 * self.rho, 1.5) * (1 - g)
            b = bincount(_cd(y, co.comp_c).argmin(1), minlength=len(co.comp_c))
        if asg:
            k = asg.batch_comp
            a = _s(0.005, at._rho_atom_t if at._rho_atom_t is not None else z.new_full((), at._rho_atom_f), 0.004)
            lo = _hbi(x, asg.batch_target, rh[k], where(a > 0.5, _p(0.15 * nn.median()), c))
            with no_grad():
                u = _p(rh[k], 0.05 * nn[k])
                v = _f(b[k] <= 0) * _s(u, _n(_f(x) - asg.batch_target), 0.35 * u)
            return lo, v, a * asg.tail_residual, w, asg
        if not co.all_iso.item() or (1 - g) <= 1e-4:
            return None, None, z, w, None
        d, i = _cd(x, co.comp_c).min(1)
        lo = w.to(x.dtype) * _hbi(x, co.comp_c[i], rh[i], e)
        return lo, w * _f(b[i] <= 0) * _s(rh[i], d, _p(0.35 * rh[i])), z, w, None

    @no_grad()
    def _plans(self, a, b, df, dr, cf, cr, disc, s, u):
        m, n = a.shape
        p = self._sink_sl(a, df, dr, _dmask(cf, cr, m, n))
        k = _dmask(cf, cf, m, m)
        if disc and m > 1:
            q = (b > 1e-12) | _ey(m, b.device)
            k = q if k is None else (k & q)
        z = df.new_zeros(m)
        q = self._sink_sl(b, z, z, k)[:m, :m].fill_diagonal_(0)
        pa, ps, pc = p[:m, :n], p[:m, n], p[m, :n]
        self.last_p_xy_slack = _dt(ps)
        v = (m * pa.sum(1)).clamp(0, 1)
        if s is not None:
            v = (v + 0.5 * s.to(v.dtype)).clamp(0, 1)
        w = (m * ps).clamp(0, 1)
        return pa, ps, pc, q, v, w, ps * (1 + 0.35 * w) * ((1 - u.to(ps.dtype)) if u is not None else 1)

    def _assemble(self, x, y, xf, df, dr, e, g, cf, cr, disc, mut, rec):
        m, n, cp = len(x), len(y), (cf is not None or cr is not None)
        z = e.new_zeros((), dtype=_F32)
        p, q, a, b, d = _conformal_horizon(x, y, e, g, cp)
        if not cp and self.chart_weight > 0 and n >= 4 and m >= 2:
            lo, u, r, w, asg = self._route(x, y, xf, e, g, rec, mut)
        else:
            lo, u, r, w, asg = None, None, z, z, None
        if not cp and not asg and m >= 8 and n >= 8:
            p, q, t, o, k, s, w, v, d = _hetero(x, y, e, a, b, d, w, g, rec)
        else:
            t, o, k, s, v = b.to(e.dtype) if b is not None else e, z, None, None, None
        (a, c), (b, h) = _bi(x, y, p), _bi(x, x, q)
        pa, ps, pc, q, j, f, i = self._plans(c, h, df, dr, cf, cr, disc, s, u)
        u = (pa * a).sum() - 0.5 * (q * _gm(j, j) * b).sum()
        if lo is not None:
            u = u + self.chart_weight * lo.to(u.dtype)
        if k is not None:
            u = u + _outlier(a, c, t, k, df, f, o, pa, self.rho)
        if not cp:
            u = u + _orphan_macro_loss(x, y, c, ps, pc, rho=self.rho, g_conn=g)
        return u, pa, i, (w, v, d, r, bool(asg))

    def forward(
        self,
        x_fake,
        y_real,
        d_fake,
        d_real,
        *,
        record=None,
        cond_fake=None,
        cond_real=None,
        allow_state_update=True,
        thermally_scaled=False,
    ):
        if self.weight == 0:
            return x_fake[:0].sum()
        a, b = self._discrete_step, self._simplicial_step
        self._discrete_step = self._simplicial_step = False
        m, n, mut = len(x_fake), len(y_real), _mut(record, allow_state_update)
        cp = cond_fake is not None or cond_real is not None
        yp, (df, dr) = self._proj(_dt(y_real)), _log_odds(d_fake, d_real)
        k = compute_contrast_confidence(yp).item()
        r = 0.5 * self.tau * (_sp((dr - df) if m == n else (dr[None] - df[:, None])).mean() - log(2))
        if k <= 1e-4:
            if _ha(record, "_disc_gate"):
                record._disc_gate = 0.0
            return self.weight * r
        xp, e, g = self._proj(x_fake), self._ell(yp, record, mut), _mconn(yp)[0].item()
        if _ha(record, "_disc_gate"):
            record._disc_gate = 1 - g
        c, pa, s, aux = self._assemble(xp, yp, x_fake, df, dr, e, g, cond_fake, cond_real, a, mut, record)
        u, v = _f(_dt(xp)), _f(yp)

        def f(t):
            return _sq(t - t.mean(0)).mean().sqrt()

        h = f(v).maximum(0.25 * f(u)).maximum(_n(u.mean(0) - v.mean(0))).maximum(_f(e))
        c, w = c / where(h > 0, _p(h), 1.0).to(xp.dtype), 0.03 * k + 0.5 * (1 - k)
        o = -(s * (0.5 * self.tau * df)).sum() + (w * self.tau) * (pa * (_sp(dr[None] - df[:, None]) - log(2))).sum()
        t = k * (c + o) + (1 - k) * r
        if record is not None:
            if mut:
                _obs_th(xp, yp, e, aux, a, b, cp, record)
            q = _ts(record)
            t = (k * (_sg(c, q) + o) + (1 - k) * r) if thermally_scaled else _sg(t, q)
        return self.weight * t

    def simplicial_manifold_loss(self, z_table, gen_fn, y_real, *, record=None, iso_weight=0.45, chart_weight=4.0):
        m, n = len(z_table), len(y_real)
        if self.weight == 0 or m < 2 or n < 1:
            return y_real[:0].sum()
        o = float(_ga(record, "chord_phase", 0.0))
        if m > 512:
            z_table, m = z_table[_sub(m, 128, z_table.device, o)], 128
        y, f = _dt(y_real).reshape(n, -1), (m <= 256)
        with no_grad():
            g, a, q, s, r = _simplicial_gates(y, chart_weight)
            z, e = _dt(z_table), self._ell(y, record, False)
            k = max(4, n if n <= 512 else 128)
            u = _bary(z, o, k, r)
        h = g > 1e-4 and chart_weight > 0
        x = gen_fn(z_table).reshape(m, -1) if h else None
        v = _dt(x) if x is not None else _dt(gen_fn(z)).reshape(m, -1)
        p = gen_fn(u[0]).reshape(k, -1)
        _obs_simp(v, p, y, e, a, r, record)
        b, c = _bi(p, y, e, fp64=f)
        t = (softmax(-c / self.eps, 1).to(b.dtype) * b).sum(1).mean()
        w = ((1 - a) * iso_weight + a * 0.25) if a > 1e-4 else iso_weight
        if w > 0:
            t = t + w * _simp_iso(p, v, z, e, a, u, f)
        if h:
            t = t + _simp_chart(z_table, x, v, y, e, g, r, q, s, o, chart_weight, f)
        return self.weight * _sg(t, _ts(record))

    def _bigan_cb(self, x, y, z, t, r, rec, mut):
        _, co, asg, rh, nn = self._commit_at(x, _dt(y), t, r, 3, rec, mut)
        if not asg:
            return None
        lo = _hbi(x, asg.batch_target, rh[asg.batch_comp], _p(0.15 * nn.median()))
        with no_grad():
            p = t.to(y.device, _F32)
            m = asg.row_comp[None] == _cd(y, co.comp_c).argmin(1)[:, None]
            h, i = m.any(1), _cd(z, p).masked_fill(~m, _INF).argmin(1)
            q = _p(_cd(p, diag_inf=True).amin(1), 1e-4) if len(p) >= 2 else p.new_ones(len(p))
            u, e, w = p[i].to(z.dtype), _p(0.25 * q.median(), 1e-4).to(z.dtype), h.to(z.dtype)
            d, f = (_n(_f(z) - p[i]) / q[i]).clamp_max(10), _f(h)
            n, b = f.sum(), _f(((d > 0.12) & h).any() | ~h.all())
            a, b, c, g = stack((n, b, (d * f).sum() / _p(n, 1), asg.tail_residual)).tolist()
        if a > 0.5:
            o, s = (w * _b1(z, u, e)).sum() / _p(w.sum(), 1), max(1.6 * c, 0.2 if b > 0.5 else 0.0)
        else:
            o, s = z.new_zeros(()), 0.25
        return lo + o, max(g, s)

    def joint_relativistic_loss(
        self,
        x_fake,
        y_real,
        d_fake,
        d_real,
        *,
        record=None,
        cond_fake=None,
        cond_real=None,
        allow_state_update=True,
        thermally_scaled=False,
    ):
        if self.weight == 0:
            return d_fake[:0].sum()
        m, n, mut = len(d_fake), len(d_real), _mut(record, allow_state_update)
        xp, yp = self._proj(_dt(x_fake)), self._proj(_dt(y_real))
        e = self._ell(yp, record, mut)
        with no_grad():
            df, dr = _log_odds(d_fake, d_real)
            p = self._sink_sl(_bi(xp, yp, e)[1], df, dr, _dmask(cond_fake, cond_real, m, n))
            pa, ps, pc = p[:m, :n], p[:m, n], p[m, :n]
            self.last_p_xy_slack = ps
        a, b = _flat(d_fake), _flat(d_real)
        d = 0.5 * (_dt(a).mean() + _dt(b).mean())
        lo = self.weight * ((pa * _sp(b[None] - a[:, None])).sum() + (ps * _sp(d - a)).sum() + (pc * _sp(b - d)).sum())
        if record is not None and cond_fake is None and cond_real is None:
            t, r = _rec_rows(x_fake, record)
            j = _split_j(x_fake, y_real, t, r)
            if j:
                c = self._bigan_cb(j[0], j[2], j[3], t, r, record, mut)
                if c:
                    o, g = c
                    if mut and _ha(record, "observe_thermal"):
                        record.observe_thermal(0.0, 0.0, 0.0, 0.0, 0.0, r_tail=g)
                    s, w = _ts(record), self.weight * max(self.chart_weight, 1)
                    lo = (lo if thermally_scaled else _sg(lo, s)) + _sg(w * o, s)
        return lo

    def conditional_fiber_loss(self, x_fake, y_real, c_cont, *, cond_disc=None, record=None, allow_state_update=True):
        k = {u: _ga(self, u) for u in _FK}
        k.update(cond_disc=cond_disc, record=record, allow_state_update=allow_state_update)
        return conditional_fiber_loss(x_fake, y_real, c_cont, **k)


def prepare_axiom_transport_step(
    transport,
    record,
    real,
    *,
    prior_table=None,
    batch_rows=None,
    is_discrete_prior=False,
    use_simplex=False,
    simplex_chart_weight=0.0,
):
    if is_discrete_prior and prior_table is not None and batch_rows is not None:
        set_active_prior_rows(prior_table, batch_rows)
        if record is not None:
            record._active_prior_table, record._active_batch_rows = prior_table, _dt(batch_rows)
    a = False
    if use_simplex and len(real) >= 8:
        m, f = _ga(transport, "max_spatial", 8), _ga(transport, "max_flat_dim", 256)
        a = compute_discrete_atomic_gate(project_features(_dt(real), m, f), record=record) > 0.5
    on = bool(use_simplex and not a and float(simplex_chart_weight) > 0)
    if _ha(transport, "bind_discrete_prior_step"):
        transport.bind_discrete_prior_step(bool(is_discrete_prior), simplicial_active=on)
    if _ha(record, "_simplicial_step"):
        record._simplicial_step = on
    return on


sym_eigensplit_zero_eigh = sym_eigensplit_halley
__all__ = [
    "AxiomCondFiberBank",
    "AxiomTransportLoss",
    "compute_contrast_confidence",
    "compute_discrete_atomic_gate",
    "conditional_fiber_loss",
    "prepare_axiom_transport_step",
    "project_features",
    "project_transport_features",
    "sym_eigensplit_zero_eigh",
]
