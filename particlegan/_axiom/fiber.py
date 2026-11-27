"""Axiom L1 Law 3B / 5C conditional fiber bank and loss."""

from math import exp, isfinite, sqrt

from torch import (
    Tensor,
    arange,
    bool as _B,
    cat,
    cdist,
    device as _dev,
    einsum,
    eye,
    finfo,
    float32 as _F32,
    float64 as _F64,
    full,
    int64 as _I64,
    is_grad_enabled,
    long as _L,
    no_grad,
    ones,
    outer,
    softmax,
    tensor,
    where,
    zeros,
    zeros_like,
)
from torch.nn.functional import relu as _rl, softplus as _sp

from .math_ops import (
    safe_cdist as _cd,
    smoothstep01 as _ss,
    solve_spd_cholesky as _solve,
    sym_eigensplit_halley as _eig,
    sym_inv_sqrt_ns as _isqrt,
)

SCHEMA_VERSION, _MAX_S, _MAX_R = 2, 64, 384
_PKT = frozenset("schema d_c d_x strata".split())
_SC = tuple("n_eff n_fast w_eff w_fast".split())
_PSD = frozenset("s_cc g_pp g_pp_fast s_xx s_yy".split())
_RNG = tuple("ring_c ring_x ring_y".split())
_F64_KEYS = tuple("mu_c s_cc g_pp g_px g_py g_pp_fast g_px_fast g_py_fast s_xx s_yy".split()) + _RNG
_STK = frozenset(("key", *_SC, *_F64_KEYS))
_QK = tuple("n_eff ring_len omega m_x m_pool w_mat delta_m_bank delta_m_loo delta_cov_loc mean_err_sq z_innov".split())
_DK = (
    "gamma_base gamma_fast gamma_min innov_threshold lin_ridge nl_ridge "
    "ring_capacity knn_k tail_leverage_gain knn_res_weight"
)
_DEF = dict(zip(_DK.split(), (0.96, 0.88, 0.5, 2.5, 1e-3, 0.025, _MAX_R, 20, 0.9, 0.3)))


def _req(ok):
    if not ok:
        raise ValueError("invalid state")


def _c64(t):
    return t.detach().cpu().to(_F64)


def _sym(m):
    return 0.5 * (m + m.mT)


def _clone(s):
    return {**{k: float(s[k]) for k in _SC}, **{k: _c64(s[k]).clone() for k in _F64_KEYS}}


def _mix3(st, ks, xs, g):
    return tuple(g * st[k] + v for k, v in zip(ks, xs))


def _mut(rec, allow):
    pr = getattr(rec, "_in_probe", False) or getattr(rec, "_in_protected_probe", False)
    return allow and is_grad_enabled() and not pr


def _mv(a, b):
    return einsum("nij,nj->ni", a, b)


def _whiten(m, fl=1e-4):
    s = _sym(m)
    d = s.diagonal().mean().clamp_min(0).item()
    return _isqrt(s, eps=max(1e-4 * max(d, fl), 1e-6) if fl > 0 else (1e-4 * d if d > 1e-12 else 1e-6))


def _basis(c, mu, w, g):
    u = (c - mu[None]) @ w
    r2 = (u**2).mean(1)
    r = r2.clamp_min(1e-6).sqrt()
    h = (2 * (_sp(r - 1) - 0.3133))[:, None]
    q = (2 * (_sp(r - 0.8) - 0.3711) / (1 + 0.25 * r))[:, None]
    return cat((c.new_ones((len(c), 1)), u, h, q * u), 1), 1 + g * r2.clamp_max(4.5), u


def _sat(t, tau, dims):
    n = (t**2).sum(dims, keepdim=True).sqrt().clamp_min(1e-8)
    return t * ((tau * (n / tau).tanh()) / n)


def _imq3(a, b, e):
    s = (a**2).sum(-1)[:, None] + (b**2).sum(-1)[None] - 2 * (a @ b.T)
    s = s + (s.detach().clamp_min(0) - s.detach())
    o = s.new_zeros(s.shape)
    for m in (0.35, 1.0, 2.8):
        em = e * m
        r = (em**2 + s).sqrt()
        o = o + (em * s) / (r * (em + r)).clamp_min(1e-30)
    return o * (1 / 3)


def _chk_t(v, shp=None):
    _req(isinstance(v, Tensor) and v.dtype == _F64 and (shp is None or tuple(v.shape) == shp))
    _req(bool(v.isfinite().all()))
    return v


def _chk_st(st, dc, dx, p, seen):
    _req(isinstance(st, dict) and set(st) == _STK)
    k = st["key"]
    _req(isinstance(k, tuple) and bool(k) and all(type(v) is int for v in k) and k not in seen)
    seen.add(k)
    for s in _SC:
        v = st[s]
        _req(not isinstance(v, bool) and isinstance(v, (int, float)) and isfinite(v) and v > 0)
    shps = ((dc,), (dc, dc), (p, p), (p, dx), (p, dx), (p, p), (p, dx), (p, dx), (dx, dx), (dx, dx))
    for n, shp in zip(_F64_KEYS[:10], shps):
        t = _chk_t(st[n], shp)
        _req(n not in _PSD or not bool((t.diagonal() < -1e-9).any()))
    lens = set()
    for n, c in zip(_RNG, (dc, dx, dx)):
        r = _chk_t(st[n])
        _req(r.ndim == 2 and r.shape[1] == c and len(r) <= _MAX_R)
        lens.add(len(r))
    _req(len(lens) == 1)


def validate_fiber_bank_state(state):
    _req(isinstance(state, dict) and set(state) == _PKT and type(state["schema"]) is int)
    _req(state["schema"] == SCHEMA_VERSION)
    dc, dx, st = state["d_c"], state["d_x"], state["strata"]
    _req(isinstance(st, list) and len(st) <= _MAX_S)
    if dc is None and dx is None:
        _req(not st)
        return state
    _req(type(dc) is int and type(dx) is int and dc >= 1 and dx >= 1)
    seen, p = set(), AxiomCondFiberBank.basis_dim(dc)
    for e in st:
        _chk_st(e, dc, dx, p, seen)
    return state


class AxiomCondFiberBank:
    """Schema-2 FP64 conditional fiber bank."""

    def __init__(self, **kw):
        bad = set(kw) - set(_DEF)
        if bad:
            raise TypeError(f"bad kwargs: {sorted(bad)}")
        for k, v in _DEF.items():
            x = kw.get(k, v)
            val = max(1, int(x)) if k in ("ring_capacity", "knn_k") else float(x)
            setattr(self, k, min(val, _MAX_R) if k == "ring_capacity" else val)
        self.d_c = self.d_x = None
        self.strata = {}

    @staticmethod
    def basis_dim(d_c):
        return 2 + 2 * int(d_c)

    def reset(self):
        self.d_c, self.d_x, self.strata = None, None, {}

    def copy(self):
        d = AxiomCondFiberBank(**{k: getattr(self, k) for k in _DEF})
        d.load_state_dict(self.state_dict())
        return d

    def _ridge(self, dc):
        r = full((self.basis_dim(dc),), self.nl_ridge * sqrt(dc), dtype=_F64, device="cpu")
        r[: 1 + dc] = self.lin_ridge
        return r

    def _sxy(self, g, gx, gy, iw, dc, ri=False):
        return _solve(g * iw, cat((gx, gy), 1) * iw, ridge=self._ridge(dc), return_inv=ri)

    def _knn(self, uq, ur, dc):
        m, d = len(ur), cdist(uq, ur, p=2.0)
        h = d.topk(min(self.knn_k, max(6, m // 5), m), 1, largest=False).values[:, -1].clamp_min(0.25 * sqrt(dc))
        lg = -0.5 * (d / h[:, None]) ** 2 - (arange(m - 1, -1, -1, dtype=_F64, device=uq.device) / 128)[None]
        return softmax(lg - lg.amax(1, keepdim=True), 1)

    def state_dict(self):
        st = [{"key": tuple(int(v) for v in k), **_clone(s)} for k, s in sorted(self.strata.items())]
        return {"schema": SCHEMA_VERSION, "d_c": self.d_c, "d_x": self.d_x, "strata": st}

    def load_state_dict(self, state):
        validate_fiber_bank_state(state)
        self.d_c, self.d_x = state["d_c"], state["d_x"]
        self.strata = {tuple(int(v) for v in e["key"]): _clone(e) for e in state["strata"]}

    validate_state_dict = staticmethod(validate_fiber_bank_state)
    to_packet, load_packet, validate_packet = state_dict, load_state_dict, validate_state_dict

    def _decay(self, st, c, zi):
        ng, cb = float(len(c)), c.mean(0)
        d = c - cb
        cov = d.T @ d
        if st is None:
            return 1.0, 0.0, 0.0, ng, cb.clone(), _sym(cov)
        ex = max(0.0, zi - self.innov_threshold)
        dm = exp(-0.5 * ex * ex)
        gs = self.gamma_min + (self.gamma_base - self.gamma_min) * dm
        gf = self.gamma_min + (self.gamma_fast - self.gamma_min) * dm
        no = gs * st["n_eff"]
        nn, dc = no + ng, cb - st["mu_c"]
        return dm, gs, gf, nn, st["mu_c"] + (ng / nn) * dc, _sym(gs * st["s_cc"] + cov + (no * ng / nn) * outer(dc, dc))

    @no_grad()
    def query_stratum(self, key, c_g, x_g, y_g):
        dc, dx, dev = c_g.shape[1], x_g.shape[1], c_g.device
        st = self.strata.get(key) if (self.d_c == dc and self.d_x == dx) else None
        p = self.basis_dim(dc)
        if st is None or st["n_eff"] < max(p + 2, 12) or len(st["ring_c"]) < 8:
            return None
        with _dev("cpu"):
            ne, we, wf = st["n_eff"], st["w_eff"], st["w_fast"]
            mu, sc, g, gx, gy, gf, gxf, gyf, sx, sy, rc, rx, ry = map(st.__getitem__, _F64_KEYS)
            ng, rl = len(c_g), len(rc)
            c, x, y = map(_c64, (c_g, x_g, y_g))
            wc = _whiten(sc / max(ne, 1), 0)
            pq, om, uq = _basis(c, mu, wc, self.tail_leverage_gain)
            sol, mis = self._sxy(g, gx, gy, 1 / max(we, 1), dc, True)
            bx, by = sol[:, :dx], sol[:, dx:]
            bp = 0.5 * (bx + by)
            mx, my, mp = pq @ bx, pq @ by, pq @ bp
            iwf = 1 / max(wf, 1)
            dmb = 0.5 * (pq @ _solve(gf * iwf, (gxf - gyf) * iwf, ridge=self._ridge(dc))[0] + mx - my)
            if ng >= 2 and mis is not None:
                u, wp, prj = x - y, pq * om[:, None], pq @ mis
                den = (om.sum().item() - om).clamp_min(1)[:, None]
                dml = (prj @ (wp.T @ u) - (prj * wp).sum(1, keepdim=True) * u) / den
            else:
                dml = dmb
            wm = _whiten(0.5 * (sx + sy) / ne)
            pr, _, ur = _basis(rc, mu, wc, self.tail_leverage_gain)
            al, mpr = self._knn(uq, ur, dc), pr @ bp
            rx0, ry0 = rx - mpr, ry - mpr
            dcov = einsum("nm,mij->nij", al, rx0[:, :, None] * rx0[:, None] - ry0[:, :, None] * ry0[:, None])
            kd = al @ ((rx - pr @ bx) - (ry - pr @ by))

            def nrm(v):
                return v.norm().item()

            me = max(0.0, (((bx - by) @ wm) ** 2).sum().item() / max(1, dx) - 2 * p / max(ne, 1))
            e_xy = max(nrm((x - mx).mean(0) @ wm), nrm((y - my).mean(0) @ wm))
            zi = max(sqrt(ng / max(1, dx)) * e_xy, nrm((c.mean(0) - mu) @ wc) * sqrt(ng / max(1, dc)))
            vals = (ne, rl, om, mx + 0.5 * kd, mp, wm, dmb + self.knn_res_weight * kd, dml, dcov, me, zi)

        def mv(v):
            return v.to(dev) if dev.type != "cpu" and isinstance(v, Tensor) else v

        return dict(zip(_QK, map(mv, vals)))

    @no_grad()
    def update_stratum(self, key, c_g, x_g, y_g, *, z_innov=0.0):
        with _dev("cpu"):
            c, x, y = map(_c64, (c_g, x_g, y_g))
            ng, dc, dx = len(c), c.shape[1], x.shape[1]
            if ng == 0 or (key not in self.strata and len(self.strata) >= _MAX_S):
                return
            if self.d_c != dc or self.d_x != dx:
                self.reset()
                self.d_c, self.d_x = dc, dx
            st = self.strata.get(key)
            dm, gs, gf, nn, mu, sc = self._decay(st, c, float(z_innov))
            phi, om, _ = _basis(c, mu, _whiten(sc / max(nn, 1), 0), self.tail_leverage_gain)
            wb, wp, cap = om.sum().item(), phi * om[:, None], self.ring_capacity
            bg = (wp.T @ phi, wp.T @ x, wp.T @ y)
            if st is None:
                we = wf = wb
                nf = nn
                up, upf = bg, tuple(map(Tensor.clone, bg))
                rng = tuple(t[-cap:].clone() for t in (c, x, y))
            else:
                we, wf, nf = gs * st["w_eff"] + wb, gf * st["w_fast"] + wb, gf * st["n_fast"] + ng
                up, upf = _mix3(st, _F64_KEYS[2:5], bg, gs), _mix3(st, _F64_KEYS[5:8], bg, gf)
                cur = len(st["ring_c"])
                kp = int(round(cur * max(0.25, dm))) if dm < 0.9 else cur
                sl = slice(-kp, None) if kp > 0 else slice(0, 0)
                rng = tuple(cat((st[k][sl], t))[-cap:] for k, t in zip(_RNG, (c, x, y)))
            sol = self._sxy(*up, 1 / max(we, 1), dc)[0]
            rx, ry = x - phi @ sol[:, :dx], y - phi @ sol[:, dx:]
            sx, sy = rx.T @ rx, ry.T @ ry
            if st is not None:
                sx, sy = gs * st["s_xx"] + sx, gs * st["s_yy"] + sy
            vals = (nn, nf, we, wf, mu, sc, *up, *upf, _sym(sx), _sym(sy), *rng)
            self.strata[key] = dict(zip(_SC + _F64_KEYS, vals))


def _part(bn, cd):
    if cd is None:
        return [((0,), arange(bn, dtype=_L))], ones((bn, bn), dtype=_B)
    rows = cd.detach().reshape(bn, -1).to("cpu", _I64)
    bkt = {}
    for i, r in enumerate(rows.tolist()):
        bkt.setdefault(tuple(int(v) for v in r), []).append(i)
    return [(k, tensor(bkt[k], dtype=_L)) for k in sorted(bkt)], (rows[:, None] == rows[None]).all(-1)


def _fb(c, z, grps):
    dc, mh = c.shape[1], zeros_like(z)
    for _, i in grps:
        zg, zm = z[i], z[i].mean(0)
        if i.numel() < max(4, dc + 2):
            mh[i] = zm
            continue
        d = c[i] - c[i].mean(0)
        rg = 1e-4 * (d**2).sum().clamp_min(1e-8).item() / max(1, dc)
        mh[i] = zm + d @ _solve(d.T @ d, d.T @ (zg - zm), ridge=rg)[0]
    return mh


def _qu(bank, grps, c, x, y, mut, blend):
    bn, d = x.shape
    ho, ow = zeros(bn, dtype=_B), eye(d, dtype=_F64)[None].expand(bn, d, d).clone()
    dm, mx, mp = x.new_zeros((3, bn, d)).unbind(0)
    ec, rw, mg = x.new_zeros((bn, d, d)), x.new_zeros(bn), x.new_zeros((bn, 1))
    pend, rmax = [], 0.0
    if bank is not None:
        for k, i in grps:
            q, zi = bank.query_stratum(k, c[i], x[i], y[i]), 0.0
            if q is not None:
                _, rl, om, mq, mpq, wm, dmb, dml, dcov, me, zi = map(q.__getitem__, _QK)
                wg = rl / (rl + 32)
                mp[i], mg[i], ho[i], ow[i], mx[i] = mpq, wg, True, wm[None], mq
                dm[i] = _sat(((1 - blend) * dml + blend * dmb) @ wm, 3, (-1,))
                ec[i] = _sat(_sym(einsum("ab,nbc,cd->nad", wm, dcov, wm)), 1.5, (-2, -1))
                rw[i], rmax = wg * (om / om.mean().clamp_min(1e-6)), max(rmax, me)
            pend.append((k, i, zi))
    ao = bool(ho.any())
    if bool(ho.all()):
        mh = mp
    else:
        fb = _fb(c, 0.5 * (x + y), grps)
        mh = (mg * mp + (1 - mg) * fb) if ao else fb
    if bank is not None and mut:
        for k, i, zi in pend:
            bank.update_stratum(k, c[i], x[i], y[i], z_innov=zi)
    ep, en = _eig(ec) if ao else (ec, ec)
    return mh, ow, dm, mx, ep, en, rw, ao, rmax


def _imq_g(cw, yw, st, mv, vr):
    d = _cd(cw).to(cw.dtype)
    dm = d.amax()
    du = where(dm > 0, d / dm.clamp_min(1e-30), d)
    ds, si = du.masked_fill(~mv, float("inf")).topk(min(4, len(cw)), 1, largest=False)
    sr = ds.gather(1, (mv.sum(1).clamp(1, min(3, ds.shape[1])) - 1)[:, None])[:, 0]
    s = where(vr, where(sr.isfinite(), sr, 0.0).clamp_min(0) + 1e-3 * du.amax() + 1e-6, 1.0)
    kc = where(mv, (-(du**2) / (2 * s[:, None] * s[None]).clamp_min(1e-12)).exp(), 0.0)
    isd = (1 + kc.sum(1)).rsqrt()
    fm = (st & vr[None] & vr[:, None]).to(yw.dtype)
    es = (fm @ (0.5 * ((yw - yw[si[:, 0]]) ** 2).sum(-1, keepdim=True))) / fm.sum(1, keepdim=True).clamp_min(1)
    yl = yw.norm(dim=-1).amax()
    ef = where(yl > 0, 1e-2 * yl, 1e-5).clamp_min(1e-12)
    return _sym((kc * isd[:, None] * isd[None]) / len(cw)), where(vr[:, None], es.sqrt().clamp_min(ef), 1.0)


def _bank(rec, b):
    if b is not None or rec is None:
        return b
    ex = getattr(rec, "_cond_bank_obj", None)
    if ex is not None:
        return ex
    f, s = AxiomCondFiberBank(), getattr(rec, "cond_fiber_bank", None)
    if s is not None:
        f.load_state_dict(s)
    rec._cond_bank_obj = f
    return f


def _op_loss(xw, yd, ow, ops, mw, cw, odt):
    dev, dt, inv = xw.device, xw.dtype, 1 / max(1, xw.shape[-1])
    dm, mx, ep, en, rw = (t.to(dev, dt) for t in ops)
    lo = xw.new_zeros((), dtype=odt)
    if mw > 0:
        lo = lo + mw * (rw * (_mv(ow, xw - yd) * dm).sum(-1) * inv).mean().to(odt)
    if cw > 0:
        rx = _mv(ow, xw - mx)
        ex = _rl((rx**2).sum(-1, keepdim=True) * inv - (1 + 5 * sqrt(2 * inv)))
        rs = rx * (1 + 0.25 * (ex**2) / (1 + ex)).rsqrt()
        q = 0.5 * einsum("ni,nij,nj->n", rx, ep, rx) + 0.5 * einsum("ni,nij,nj->n", rs, en, rs)
        lo = lo + cw * (rw * q * inv).mean().to(odt)
    return lo


def _upd_ema(rec, wi, pxy, pxx, pyy, rw):
    num = (wi * (2 * pxy.detach() - pxx.detach() - pyy)).sum()
    ri = max(-0.5, min(0.5, (num / (wi * pyy).sum().clamp_min(1e-8)).item()))
    rc = ri + min(0.5, 0.11 * max(0.0, rw - 0.1))
    rec.cond_ema = 0.99 * float(rec.cond_ema) + 0.01 * rc if rec.cond_initialized else max(0.18, rc)
    rec.cond_initialized = True
    if hasattr(rec, "step_scale"):
        rec.step_scale = float(1 - 0.85 * _ss((0.05 - max(0.0, float(rec.cond_ema))) / 0.024))


def conditional_fiber_loss(
    x_fake,
    y_real,
    c_cont,
    *,
    cond_disc=None,
    record=None,
    bank=None,
    mean_op_weight=0.36,
    cov_op_weight=0.35,
    imq_active_scale=0.40,
    bank_mean_blend=0.75,
    allow_state_update=True,
):
    bn = len(x_fake)
    if bn <= 1 or len(y_real) != bn or len(c_cont) != bn:
        return x_fake[:0].sum()
    xf, yf = x_fake.reshape(bn, -1), y_real.detach().reshape(bn, -1)
    dev, d = xf.device, xf.shape[-1]
    dim, dt = max(1, d), (_F64 if xf.dtype == _F64 else _F32)
    fmax = finfo(xf.dtype).max * (0.25 / sqrt(dim + 1))
    mut, bk = _mut(record, allow_state_update), _bank(record, bank)
    with no_grad():
        grps, sc = _part(bn, cond_disc)
        if not any(i.numel() >= 2 for _, i in grps):
            return x_fake[:0].sum()
        cw, xd, yd = c_cont.detach().reshape(bn, -1).to(dt), xf.detach().to(dt), yf.to(dt)
        c0, x0, y0 = cat((cw, xd, yd), 1).to("cpu", _F64).split((cw.shape[1], d, d), 1)
        mh_c, ow_c, *ops, ao, rw = _qu(bk, grps, c0, x0, y0, mut, float(bank_mean_blend))
        mh, ow = mh_c.to(dev, dt), ow_c.to(dev, dt)
        owi = ow / ((ow**2).sum((-2, -1), keepdim=True) / dim).sqrt().clamp_min(1e-8)
        st = sc.to(dev)
        m0 = st.clone().fill_diagonal_(False)
        vr = m0.any(1)
        yw = _mv(owi, yd - mh)
        wi, ef = _imq_g(cw, yw, st, m0 & vr[:, None] & vr[None], vr)
        pyy = _imq3(yw, yw, ef)
    xw0 = xf.to(dt)
    xw = _mv(owi, xw0 - mh)
    pxy, pxx = _imq3(xw, yw, ef), _imq3(xw, xw, ef)

    def cm(t):
        return t.clamp_max(fmax)

    li = (wi.to(xf.dtype) * (cm(pxy) - 0.5 * cm(pxx) - 0.5 * cm(pyy)).to(xf.dtype)).sum()
    ho = ao and (mean_op_weight > 0 or cov_op_weight > 0)
    lo = _op_loss(xw0, yd, ow, ops, mean_op_weight, cov_op_weight, xf.dtype) if ho else xf.new_zeros(())
    if mut and hasattr(record, "cond_initialized"):
        with no_grad():
            _upd_ema(record, wi, pxy, pxx, pyy, rw)
    return ((imq_active_scale if ho else 1) * li + lo).clamp(-fmax, fmax).to(xf.dtype)


__all__ = ["AxiomCondFiberBank", "SCHEMA_VERSION", "conditional_fiber_loss", "validate_fiber_bank_state"]
