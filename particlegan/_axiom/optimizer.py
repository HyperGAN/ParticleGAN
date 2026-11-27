"""Axiom L1 Optimizer (Law 4)."""

from __future__ import annotations

from copy import copy
from math import hypot, isfinite, log2, sqrt

import torch
from torch import nn
from torch.optim import Optimizer

from .math_ops import (
    _apply_stream,
    _pdot,
    fp32_tikhonov_ns_2d,
    fp32_tikhonov_ns_batched,
    mantissa_exponent_normalize,
    me_unit_rows,
    schatten6_decay_batched,
    smoothstep01,
    solve_nonascent_multipliers,
    stable_rank_power,
    zero_pinv_project_nonascent,
)
from .state import (
    SCHEMA_VERSION,
    clear_active_prior_rows,
    eval_probe_forward,
    get_tensor_record,
    set_active_prior_rows,
)

_T = torch.Tensor
_F32, _F64, _BOOL, _LONG = torch.float32, torch.float64, torch.bool, torch.long
_ROLES = ("generator", "encoder", "router", "noise", "critic", "prior", "table")
_OPT_DEF = (
    ("weight_decay", 0.0),
    ("amsgrad", False),
    ("maximize", False),
    ("differentiable", False),
    ("capturable", False),
    ("foreach", None),
    ("fused", None),
)
_FP32 = ("v_pow", "m_unit", "m_scale", "m_prior")
_SCALARS = (
    ("last_step_gain", 0.0, None, True),
    ("ema_func_gain", 0.0, 1.0, True),
    ("ema_step_cos", -1.0, 1.0, False),
    ("log2_damp", -4.0, 0.0, False),
    ("crit_skew_ema", 0.0, None, False),
)
_META_KEYS = ("schema", "family", "direction_blend", "prev_step_sketch", "sampled_rows") + tuple(s[0] for s in _SCALARS)
_SKETCH = 67
_WL2_FL = log2(1e-12)


def _req(ok):
    if not ok:
        raise ValueError("invalid")


def _dkeys(d, ks):
    _req(isinstance(d, dict) and set(d) == ks)


def _fin(v, lo=None, hi=None, slo=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v):
        return False
    return (lo is None or (v > lo if slo else v >= lo)) and (hi is None or v <= hi)


def _tfin(t):
    return isinstance(t, _T) and torch.isfinite(t).all().item()


def _sc(x, dev, dt=_F64):
    return x.to(dev, dt).reshape(()) if isinstance(x, _T) else torch.full((), float(x), device=dev, dtype=dt)


def _is_loss(v):
    return isinstance(v, _T) and v.ndim == 0 and bool(v.requires_grad)


def _all_fin_sync(normals):
    bd = {}
    for g in normals:
        for t in g:
            if t is not None and t.numel() > 0:
                bd.setdefault(t.device, []).append(torch.isfinite(t).all())
    return all(torch.stack(fs).all().item() for fs in bd.values())


def _upd_l2(st, lt, ge, de, hw):
    if not st.get("log2_init", False):
        if hw:
            st["log2_init"], st["log2_ref"] = True, lt
        return lt
    rp = st.get("log2_ref", 0.0)
    st["log2_ref"] = rn = (
        rp + 0.25 * min(2.0, lt - rp) if lt > rp else rp - (0.25 * (1.0 - ge) + de * ge) * min(2.0, rp - lt)
    )
    return rn


def _mom(st, p, gu, b1):
    if b1 <= 0.0:
        return gu
    mu = st.setdefault("m_unit", p.new_zeros(p.shape, dtype=_F32))
    ms = st.setdefault("m_scale", p.new_zeros((), dtype=_F32))
    gn, _, _ = mantissa_exponent_normalize(b1 * mu + (1.0 - b1) * gu)
    mu.copy_(gn)
    ms.fill_(1.0)
    return gn


def _reach(loss, pids):
    hit, seen, stk = set(), set(), [getattr(loss, "grad_fn", None)]
    while stk:
        if (nd := stk.pop()) is None or id(nd) in seen:
            continue
        seen.add(id(nd))
        if isinstance(v := getattr(nd, "variable", None), _T) and id(v) in pids:
            hit.add(id(v))
        stk.extend(nx for nx, _ in getattr(nd, "next_functions", ()))
    return hit


def _dots(pairs, ref):
    z0 = ref.new_zeros((), dtype=_F64)
    return torch.stack([_pdot(x, y, z0) for x, y in pairs]).tolist()


def _blend_dots(disps, normals):
    ps = [(normals[0], normals[0]), (normals[0], disps), (disps, disps)]
    if len(normals) == 2:
        ps = [ps[0], (normals[1], normals[1]), ps[1], (normals[1], disps), (normals[0], normals[1]), ps[2]]
    v = _dots(ps, disps[0])
    return ([v[1]], [v[0]], 0.0, v[2]) if len(normals) == 1 else ([v[2], v[3]], [v[0], v[1]], v[4], v[5])


def _blend_post(ad, an2, a01, d2, al):
    a0, a1 = (al[0], al[1]) if len(al) > 1 else (al[0], 0.0)
    n0, n1, d0, d1 = (an2[0], an2[1], ad[0], ad[1]) if len(ad) > 1 else (an2[0], 0.0, ad[0], 0.0)
    aft = max(d0 - a0 * n0 - a1 * a01, d1 - a0 * a01 - a1 * n1) if len(ad) > 1 else (d0 - a0 * n0)
    return aft, max(0.0, d2 - 2.0 * (a0 * d0 + a1 * d1) + a0 * a0 * n0 + a1 * a1 * n1 + 2.0 * a0 * a1 * a01)


def _rms_fin(y, yp):
    return torch.isfinite(y).all().double(), torch.where(torch.isfinite(y), y.double() - yp.double(), 0.0).reshape(
        yp.shape[0], -1
    ).square().mean().sqrt()


def _role_pyth(g0s, g1s):
    if (ref := next((t for t in (*g0s, *g1s) if t is not None), None)) is None:
        return None
    a2, b2, dt = _dots(((g0s, g0s), (g1s, g1s), (g0s, g1s)), ref)
    if a2 > 1e-24 and b2 > 1e-24 and isfinite(a2 + b2 + dt):
        return min(1.5, max(1.0, (sqrt(a2) + sqrt(b2)) / sqrt(max(a2 + b2 + 2.0 * dt, 1e-36))))
    return None


def _copy_pend(pend, ps=None):
    if pend is None:
        return None
    return [
        [None if t is None else t.detach().to(p.device if ps else "cpu", t.dtype).clone() for t, p in zip(g, ps or g)]
        for g in pend
    ]


def _res_norm(p, g, pair, need_u=True):
    if pair is not None and 1e-17 < pair[0] < 1e17 and 0.0 <= pair[1] < 1e17:
        gn, pn = pair
        hw = pn > 1e-12
        return (g / gn if need_u else gn), log2(gn), (log2(pn) if hw else 0.0), hw
    gu, l2g, okg = mantissa_exponent_normalize(g)
    if not okg.item():
        return None
    _, l2p, okp = mantissa_exponent_normalize(p.detach())
    lp = l2p.item()
    hw = okp.item() and lp > _WL2_FL
    return gu, l2g.item(), (lp if hw else 0.0), hw


def _add_upd(up, ud, dev):
    if not up:
        return
    if all(p.device == dev and p.dtype == up[0].dtype for p in up):
        torch._foreach_add_(up, ud)
    else:
        for pi, di in zip(up, ud):
            pi.add_(di)


def collect_conv_layouts(modules):
    return {
        id(m.weight): (isinstance(m, nn.ConvTranspose2d), int(m.groups))
        for r in modules
        if isinstance(r, nn.Module)
        for m in r.modules()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    }


class AxiomOptimizer(Optimizer):
    family = recipe_optimizer_family = "axiom"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True
    ema_critic = anchor = guard = latent_damping = latent_history = direct_response = direct_history = None

    def __init__(
        self,
        params,
        *,
        lr=0.014,
        betas=(0.0, 0.999),
        eps=1e-8,
        smoothing=0.05,
        direction_blend=False,
        critic=None,
        record=None,
        conv_layouts=None,
        **options,
    ):
        self._check_init(lr, betas, eps, smoothing, direction_blend, options)
        self.smoothing, self.direction_blend, self.critic = float(smoothing), direction_blend, critic
        self.record = self._axiom_record = record
        self._psk = self._prot = None
        self._cl = dict(conv_layouts) if conv_layouts else {}
        self._sr, self._rsc, self._gpids, self._mc = {}, {}, set(), {}
        self._mgs = self._mwb = self._mps = self.last_step_gain = self.ema_func_gain = 1.0
        self.ema_step_cos = self.log2_damp = self.crit_skew_ema = 0.0
        self.constraint_geometry_stats = dict(
            steps=0, projected_steps=0, max_derivative_before=0.0, max_derivative_after=0.0
        )
        self.direction_blend_stats = dict(
            conflict_steps=0, blended_steps=0, pareto_stalls=0, retained_norm_ratio_sum=0.0, max_retained_norm_ratio=0.0
        )
        self._reset_counts()
        role0 = "critic" if critic is not None else "generator"
        defaults = {
            **dict(_OPT_DEF),
            "lr": float(lr),
            "betas": tuple(float(b) for b in betas),
            "eps": float(eps),
            "role": role0,
            **options,
        }
        super().__init__(params, defaults)
        for g in self.param_groups:
            g["role"] = role = g.get("role", g.get("forge_role", role0))
            _req(role in _ROLES)
            is_pr = role in ("prior", "table")
            if is_pr:
                _req(len(g["params"]) == 1 and g["params"][0].ndim == 2)
                g.setdefault("sampled_rows_required", role == "prior")
            g["algorithm"] = "axiom_prior" if is_pr else "axiom"
        if isinstance(critic, nn.Module):
            self.bind_module_layouts((critic,))

    def _reset_counts(self):
        self.ns_2d_activations = self.fast_path_steps = self.trust_backtracks_total = self.probe_consumptions = 0
        self.last_probe_consumed = False

    @staticmethod
    def _check_init(lr, betas, eps, sm, db, opts):
        _req(_fin(lr, 0.0) and _fin(eps, 0.0, slo=True) and _fin(sm, 0.0) and type(db) is bool and len(betas) == 2)
        for b in betas:
            _req(_fin(b, 0.0) and b < 1.0)
        ks = {k for k, _ in _OPT_DEF}
        for k, v in opts.items():
            if k not in ks:
                raise TypeError(k)
            _req(v in (None, False, 0, 0.0))

    def bind_module_layouts(self, modules):
        self._cl.update(collect_conv_layouts(modules))
        self._mc.clear()

    def _active_record(self):
        if (r := self.record if self.record is not None else self._axiom_record) is not None:
            return r
        return next((rec for p in self._parameters() if (rec := get_tensor_record(p)) is not None), None)

    def _parameters(self):
        return [p for g in self.param_groups for p in g["params"]]

    def set_generator_probe_parameters(self, parameters):
        self._gpids = {id(p) for p in parameters}

    def bind_origin_sample(self, sample):
        pass

    def _pmetrics(self, p):
        if (c := self._mc.get(id(p))) is not None:
            return c
        if p.ndim < 2:
            return self._mc.setdefault(id(p), (0.5 if p.numel() == 1 else 1.0, 0.0, 1, 1, 1))
        if p.ndim == 4:
            c0, c1, kh, kw = p.shape
            it, gr = self._cl.get(id(p), (False, 1))
            oc, ic = (c1 * gr, c0) if it else (c0, c1 * gr)
            if not (kh == kw == gr == 1 and not it):
                ar, fn = max(1, (kh * kw) // 4 if it else kh * kw), max(1, min(oc // max(1, gr), ic // max(1, gr)))
                out = (sqrt(oc / max(1, ic) * gr / ar) * (fn**0.25), 0.5 * log2(max(1, min(oc, ic) * kh * kw)), 1, 1, 1)
                return self._mc.setdefault(id(p), out)
            rd, cd = oc, ic
        else:
            rd, cd = p.shape[0], p.numel() // max(1, p.shape[0])
        md = min(rd, cd)
        return self._mc.setdefault(id(p), (sqrt(max(1.0, rd / max(1, cd))), 0.5 * log2(max(1, md)), rd, cd, md))

    def set_sampled_rows(self, param, rows):
        rt = torch.as_tensor(rows).reshape(-1).to(param.device, _LONG)
        if rt.device.type == "cpu" and rt.numel():
            _req(not ((rt < 0).any().item() or (rt >= len(param)).any().item()))
        prev = self._sr.get(param)
        self._sr[param] = rt if prev is None else torch.cat((prev, rt))
        set_active_prior_rows(param, self._sr[param])
        if (rec := self._active_record()) is not None:
            rec._active_prior_table, rec._active_batch_rows = param, rt.detach()

    def sampled_rows_for(self, param):
        return None if param not in self._sr else torch.unique(self._sr[param], sorted=True).detach().clone()

    def clear_sampled_rows(self):
        for p in self._sr:
            clear_active_prior_rows(p)
        self._sr.clear()
        self._rsc.clear()
        if (rec := self._active_record()) is not None:
            rec._active_prior_table = rec._active_batch_rows = None

    def set_row_scales(self, param, scales):
        if scales is None:
            self._rsc.pop(param, None)
        else:
            self._rsc[param] = scales.detach().to(param.device, _F32)

    def bind_axiom_modulation(self, *, g_scale=1.0, wakeup_boost=1.0, prior_scale=1.0):
        self._mgs = g_scale.detach() if isinstance(g_scale, _T) else float(g_scale)
        self._mwb = wakeup_boost.detach() if isinstance(wakeup_boost, _T) else float(wakeup_boost)
        self._mps = float(prior_scale)

    bind_v4_modulation = bind_axiom_modulation

    def _collect_normal(self, overrides=None, clone=False):
        out = []
        for g in self.param_groups:
            ps = g["algorithm"] == "axiom_prior" and g.get("sampled_rows_required", True)
            for p in g["params"]:
                ov = overrides is not None and id(p) in overrides
                src = overrides[id(p)] if ov else (None if p.grad is None else p.grad.detach())
                if src is not None and ps:
                    _req((rows := self.sampled_rows_for(p)) is not None)
                    pc = torch.zeros_like(src)
                    pc[rows] = src[rows]
                    src = pc
                elif src is not None and clone and not ov:
                    src = src.clone()
                out.append(src)
        return out

    def bind_protected_from_grads(self):
        if self.direction_blend:
            _req(self._prot is None)
            self._prot = [self._collect_normal()]

    def bind_protected_losses(self, losses, *, protected_evaluator=None):
        if not self.direction_blend:
            return
        _req(self._prot is None and 1 <= len(lt := tuple(losses)) <= 2 and all(_is_loss(x) for x in lt))
        ps, tr, sv, normals = self._parameters(), [], [], []
        for p in ps:
            if p.requires_grad:
                tr.append(p)
            sv.append(None if p.grad is None else p.grad.detach().clone())
        try:
            for lk in lt:
                for p in ps:
                    p.grad = None
                if tr:
                    lk.backward(retain_graph=True, inputs=tr)
                normals.append(self._collect_normal(clone=True))
        finally:
            for p, sg in zip(ps, sv):
                p.grad = sg
        _req(_all_fin_sync(normals))
        self._prot = normals

    def backward_with_protected(self, loss_gan, aux_loss=None):
        _req(_is_loss(loss_gan) and not (self.direction_blend and self._prot is not None))
        ha = _is_loss(aux_loss)
        loss_gan.backward()
        if not self.direction_blend:
            if ha:
                aux_loss.backward()
            return
        sv = None
        if ha:
            ps = self._parameters()
            tch = _reach(aux_loss, {id(p) for p in ps})
            sv = {id(p): (None if p.grad is None else p.grad.detach().clone()) for p in ps if id(p) in tch}
            aux_loss.backward()
        self._prot = [self._collect_normal(sv)]

    def abort_pending(self):
        self.clear_sampled_rows()
        self._mgs = self._mwb = self._mps = 1.0
        self._prot = None

    def _sketch(self, pairs, dev):
        rows = []
        for _, g in pairs:
            if g.numel() <= 0:
                continue
            fl = (
                (
                    g.permute(0, 2, 3, 1).reshape(-1)
                    if g.ndim == 4 and g.is_contiguous(memory_format=torch.channels_last)
                    else g.reshape(-1)
                )
                .detach()
                .float()
            )
            if (n := fl.numel()) >= _SKETCH:
                st = max(1, n // _SKETCH)
                rows.append(fl[:: st - (st > 1 and st % 2 == 0)][:_SKETCH])
            else:
                rows.append(torch.cat((fl, fl.new_zeros(_SKETCH - n))))
        if not rows:
            return torch.zeros(_SKETCH, dtype=_F32, device=dev), torch.zeros((), dtype=_BOOL, device=dev)
        acc = me_unit_rows(torch.stack(rows).to(dev))[0].sum(dim=0)
        sn = acc.norm()
        ok = torch.isfinite(sn) & (sn > 0.0)
        return torch.where(ok, acc / sn.clamp_min(1e-30), 0.0), ok

    def _critic_path(self, cm4, nmap, dev):
        l2s = [
            r[1] + r[2] - hl2
            for idx, (p, g, _, _, hl2, *_) in cm4
            if (r := _res_norm(p, g, nmap.get(idx), False)) is not None
        ]
        if len(l2s) < 3:
            return False
        sk = max(l2s) - min(l2s)
        self.crit_skew_ema = max(sk, 0.97 * self.crit_skew_ema + 0.03 * sk)
        if self.crit_skew_ema <= 2.0:
            return False
        rs = []
        for _, (p, g, *_, ext, _) in cm4:
            st = self.state[p]
            r, vn = stable_rank_power(g.reshape(ext[0], ext[1]), init_v=st.get("v_pow"), return_v=True)
            st["v_pow"] = vn.detach()
            rs.append(r.to(dev))
        return min(torch.stack(rs).tolist()) < 1.05

    @staticmethod
    def _critic_gammas(c2d):
        bkts, out = {}, [None] * len(c2d)
        for i, (_, (_, g, *_, (rd, cd, md), _)) in enumerate(c2d):
            bkts.setdefault((rd, cd, g.device), []).append((i, g.reshape(rd, cd), md))
        for its in bkts.values():
            rs = stable_rank_power(torch.stack([x[1] for x in its]))
            gs = torch.minimum(rs.new_full(rs.shape, sqrt(its[0][2])), 2.0 * rs.sqrt()).double()
            for j, (oi, _, _) in enumerate(its):
                out[oi] = gs[j]
        return out

    def _two_obj_pyth(self):
        if self._prot is None or len(self._prot) != 2:
            return {}
        n0, n1 = self._prot
        acc, idx = {}, 0
        for g in self.param_groups:
            pids, g0s, g1s = acc.setdefault(g["role"], ([], [], []))
            for p in g["params"]:
                pids.append(id(p))
                if p.grad is not None and p.numel() > 0:
                    g0s.append(n0[idx])
                    g1s.append(n1[idx])
                idx += 1
        res = {}
        for pids, g0s, g1s in acc.values():
            if (bo := _role_pyth(g0s, g1s)) is not None:
                for pid in pids:
                    res[pid] = bo
        return res

    def _prep_prior(self, p, grad, grp, b1):
        st, rsc, req = self.state[p], self._rsc.get(p), grp.get("sampled_rows_required", True)
        if req:
            ridx = torch.sort(self._sr[p])[0] if rsc is None else self.sampled_rows_for(p)
        else:
            ridx = torch.arange(len(p), device=p.device, dtype=_LONG)
        if ridx is None or ridx.numel() == 0:
            return None
        rm = ridx.new_ones(len(ridx), dtype=_BOOL)
        if req and rsc is None:
            rm[1:] = ridx[1:] != ridx[:-1]
        rm = rm.unsqueeze(1)
        gr = grad[ridx].float()
        fin = torch.isfinite(gr).all(dim=-1, keepdim=True)
        dr = torch.where(fin, gr, 0.0)
        if b1 > 0.0:
            mp = st.setdefault("m_prior", p.new_zeros(p.shape, dtype=_F32))
            mpv = mp[ridx]
            dr = torch.where(fin, mpv.mul(b1).add(dr, alpha=1.0 - b1), mpv)
            mp.index_add_(0, ridx, (dr - mpv) * rm.float())
        ur, l2, ok0 = me_unit_rows(dr)
        ok = rm & ok0
        okf = ok.float()
        nok = okf.sum()
        return p, ridx, rsc, ur, l2, ok, nok.double(), ((l2 * okf).sum() / nok.clamp_min(1.0)).double()

    def _finish_priors(self, pre_pr, sv, off, so, sb, sp, ge, de, hh):
        sr, spk, k = so + (0.0 if self._mps < 1.0 else ge * sb), 1.25 * ge * sp, off
        for lrp, pr in pre_pr:
            if pr is None:
                continue
            nok, ml2, k = sv[k], sv[k + 1], k + 2
            if lrp <= 0.0:
                continue
            p, ridx, rsc, ur, l2, ok, _, _ = pr
            st = self.state[p]
            rn = _upd_l2(st, ml2, ge, de, True) if nok > 0.0 else st.get("log2_ref", 0.0)
            u = torch.where(
                ok,
                ur
                * torch.rsqrt(
                    1.0
                    + (sr * (r := torch.exp2((rn - l2).clamp(-20.0, 20.0))) + spk * (r - 1.0).clamp_min(0.0)).square()
                )
                if hh
                else ur,
                0.0,
            )
            if rsc is not None and len(rsc) == len(ridx):
                u = u * rsc.unsqueeze(1)
            p.index_add_(0, ridx, u.to(p.dtype), alpha=-lrp)

    def _route_critic(self, net_all, nmap, st, dev):
        s6, c2d = {}, [(i, it) for i, it in enumerate(net_all) if it[8]]
        if not c2d or self._critic_path([(i, it) for i, it in c2d if it[7][2] >= 4], nmap, dev):
            return s6
        for (i, it), gt in zip(c2d, self._critic_gammas(c2d)):
            p, g, lr, fac, hl2, b1, isg, (rd, cd, _), _ = it
            net_all[i] = (p, g, lr, fac, hl2, b1, isg, gt, False)
            if st > 0.0:
                nt = rd > cd
                s6.setdefault((cd, rd) if nt else (rd, cd), []).append((p, fac, nt, rd, cd, lr * st))
        return s6

    def _step_net(self, net_all, nmap, scf, gp, dev):
        co, so, sb, sp, ge, de, hh, gm, dm = scf
        fps, fgs, fal, fgt, up, ud, nsb = [], [], [], [], [], [], {}
        for i, (p, g, lrp, fac, hl2, b1, isg, ext, _) in enumerate(net_all):
            can_f = not gp and b1 == 0.0 and not isinstance(ext, tuple)
            if (res := _res_norm(p, g, nmap.get(i), not can_f)) is None:
                continue
            gu, l2g, l2p, hw = res
            st = self.state[p]
            lt = l2g + l2p - hl2
            rn = _upd_l2(st, lt, ge, de, hw)
            rat = 2.0 ** max(-20.0, min(20.0, rn - lt))
            stv = (
                (ge * (sb * rat + sp * max(0.0, rat - 1.0)) + so * rat)
                if (hh and (hw or st.get("log2_init", False)))
                else 0.0
            )
            rate = lrp * (gm if isg else dm) * fac * co
            eff = -rate / hypot(1.0, stv)
            if isinstance(gu, float):
                fps.append(p)
                fgs.append(g)
                fal.append(eff / gu)
                fgt.append(ext)
            elif isinstance(ext, tuple):
                rd, cd, _ = ext
                nt = rd > cd
                mc = _mom(st, p, gu, b1).float().reshape(rd, cd).transpose(0, int(nt))
                nsb.setdefault((mc.shape[0], mc.shape[1]), []).append((p, mc, nt, rate, stv))
            else:
                d = (_mom(st, p, gu, b1) * eff).to(p.dtype)
                up.append(p)
                ud.append(d if ext is None else d * ext.to(d.dtype))
        self._apply_net_updates(fps, fgs, fal, fgt, up, ud, nsb, dev)

    def _apply_net_updates(self, fps, fgs, fal, fgt, up, ud, nsb, dev):
        if fps:
            sc = torch.tensor(fal, dtype=_F64, device=dev)
            if any(t is not None for t in fgt):
                one = sc.new_ones(())
                sc = sc * torch.stack([one if t is None else t for t in fgt])
            self.fast_path_steps += 1
            torch._foreach_addcmul_(fps, fgs, list(sc.float().unbind(0)))
        for bkt in nsb.values():
            self.ns_2d_activations += len(bkt)
            mats, rsv = torch.stack([b[1] for b in bkt]), [b[4] for b in bkt]
            lo, hi = min(rsv), max(rsv)
            rsa = 0.0 if hi <= 0.0 else (lo if lo == hi else torch.tensor(rsv, dtype=_F32, device=mats.device))
            orth = fp32_tikhonov_ns_batched(mats, rel_smooth=rsa, pre_normalized=True).float()
            for i, (pi, _, nt, mult, _) in enumerate(bkt):
                up.append(pi)
                ud.append((orth[i].transpose(0, int(nt)) * (-mult)).reshape_as(pi).to(pi.dtype))
        _add_upd(up, ud, dev)

    @staticmethod
    def _apply_s6(s6b, dm, dev):
        if not s6b:
            return
        dmt = _sc(dm, dev, _F32)
        for bkt in s6b.values():
            ws = torch.stack([p.float().reshape(rd, cd).transpose(0, int(nt)) / fac for p, fac, nt, rd, cd, _ in bkt])
            ds = schatten6_decay_batched(
                ws, torch.tensor([r for *_, r in bkt], dtype=_F32, device=ws.device) * dmt.to(ws.device)
            )
            for i, (p, fac, nt, _, _, _) in enumerate(bkt):
                p.add_((ds[i].transpose(0, int(nt)) * fac).reshape_as(p).to(p.dtype))

    @staticmethod
    def _flush_trust(rec, yp, y_pre, pg):
        dev = yp.device
        ek, et = rec.take_pending_ema_tensors() if hasattr(rec, "take_pending_ema_tensors") else ([], [])
        pv = torch.stack([*_rms_fin(yp, y_pre), _sc(pg, dev), *(t.to(dev, _F64) for t in et)]).tolist()
        if et:
            rec.apply_flushed_ema_values(ek, pv[3:])
        return pv[0], pv[1], pv[2]

    def _backtrack(self, cache, rec, deltas, csc, dobs, dmin, dmax):
        gm, pa, pk, y_pre = cache

        def _ev(a):
            for p, o, d in deltas:
                p.copy_(o + d * a)
            y = eval_probe_forward(gm, pa, pk, rec)
            if y is None or y.shape != y_pre.shape:
                return float("inf")
            f, r = torch.stack(_rms_fin(y, y_pre)).tolist()
            return r / csc if f > 0.5 else float("inf")

        al = 1.0
        if dmin > 0.0 and isfinite(dobs) and 1e-12 < dobs < dmin:
            al = min(16.0, max(1.0, dmin / (dobs + dmin / 16.0)))
            dobs = _ev(al)
        for _ in range(4):
            if dobs <= dmax:
                break
            self.trust_backtracks_total += 1
            al = max(1e-6, min(0.5 * al, 0.9 * al * dmax / dobs) if isfinite(dobs) and dobs > 1e-30 else 0.1 * al)
            dobs = _ev(al)
        if not isfinite(dobs) or dobs > dmax:
            al = max(1e-6, 0.5 * al)
            for p, o, d in deltas:
                p.copy_(o + d * al)
        return al, dobs

    def _upd_fgain(self, al, dobs, dmax, pgf, st):
        if al < 1.0:
            self.ema_func_gain = 0.70 * self.ema_func_gain + 0.30 * max(0.05, min(1.0, pgf * al))
        elif st >= 0.5 and isfinite(dobs) and (al > 1.0 or dobs >= 0.012):
            if dmax / max(dobs, 1e-12) * pgf > 1.5 and self.ema_func_gain < 1.0:
                self.ema_func_gain = 0.995 * self.ema_func_gain + 0.005

    def _finish_trust(self, rec, is_c, gp, trk, tinfo, stf):
        if not (gp and trk):
            if rec is not None and not is_c:
                rec._g_probe_cache = None
            return 1.0, False
        cache, rec._g_probe_cache = rec._g_probe_cache, None
        gm, pa, pk, y_pre = cache
        deltas = [(p, o, (p - o).detach()) for p, o in trk]
        yp = eval_probe_forward(gm, pa, pk, rec)
        if yp is None or yp.shape != y_pre.shape:
            return 1.0, False
        pg, ss, st, simp = tinfo
        f0, r0, pgf = self._flush_trust(rec, yp, y_pre, pg)
        csc = max(1e-30, float(getattr(rec, "ema_data_scale", 1.0)) / sqrt(max(1, y_pre[0].numel())))
        pr = float(getattr(rec, "ema_push_res", 0.0))
        dmax = (0.045 + 0.16 * min(1.5, pr)) * ss
        dmin = 0.012 * smoothstep01((pr - 0.18) / 0.25) * ss if (not simp and st > 0.10) else 0.0
        al, dobs = self._backtrack(cache, rec, deltas, csc, (r0 / csc) if f0 > 0.5 else float("inf"), dmin, dmax)
        self._upd_fgain(al, dobs, dmax, pgf, st)
        rec.g_func_gain = float((1.0 - stf) * self.ema_func_gain + stf)
        return al, True

    def _apply_dir_blend(self, all_p, origs):
        normals = self._prot
        if normals is None or not all_p:
            return
        disps = [p - o for p, o in zip(all_p, origs)]
        ad, an2, a01, d2 = _blend_dots(disps, normals)
        if not all(isfinite(x) for x in (*ad, *an2, a01, d2)):
            torch._foreach_copy_(all_p, origs)
            raise ValueError("invalid")
        bmax, st, bl = max(ad), self.constraint_geometry_stats, self.direction_blend_stats
        proj, aft = False, bmax
        if bmax > 0.0:
            bl["conflict_steps"] += 1
            if len(normals) == 2 and an2[0] > 0.0 and an2[1] > 0.0 and a01 / sqrt(an2[0] * an2[1]) <= -1.0 + 2e-24:
                al, iz, proj = (0.0, 0.0), True, True
                bl["pareto_stalls"] += 1
            else:
                al, iz, proj = solve_nonascent_multipliers(ad, an2, d2, a01)
                bl["blended_steps"] += 1
            if iz:
                torch._foreach_copy_(all_p, origs)
                aft = an = 0.0
            else:
                for p, o, nd in zip(all_p, origs, _apply_stream(disps, normals, al)):
                    p.copy_(o + nd)
                aft, an = _blend_post(ad, an2, a01, d2, al)
            rat = sqrt(an / d2) if d2 > 0.0 else 0.0
            bl["retained_norm_ratio_sum"] += rat
            bl["max_retained_norm_ratio"] = max(bl["max_retained_norm_ratio"], rat)
        st["steps"] += 1
        st["projected_steps"] += int(proj)
        st["max_derivative_before"] = max(st["max_derivative_before"], bmax)
        st["max_derivative_after"] = max(st["max_derivative_after"], aft)

    def _chk_grad(self, p, g, is_pr, req):
        _req(not g.is_sparse and not g.is_complex())
        if is_pr and req:
            _req(p in self._sr)

    def _partition(self, lr_sc, gp, pyth):
        pairs, net, pit, trk, dev = [], [], [], [], None
        for grp in self.param_groups:
            is_pr, req = grp["algorithm"] == "axiom_prior", grp.get("sampled_rows_required", True)
            isg, b1 = grp["role"] != "critic", grp["betas"][0]
            blr = grp["lr"] * lr_sc * (self._mps if is_pr else 1.0)
            for p in grp["params"]:
                if (g := p.grad) is None:
                    continue
                self._chk_grad(p, g, is_pr, req)
                dev = p.device if dev is None else dev
                self.state[p]["step"] = self.state[p].get("step", 0) + 1
                lrp = blr * pyth.get(id(p), 1.0)
                if is_pr:
                    pit.append((p, g, lrp, max(b1, 0.75) if self._mps < 1.0 else b1, grp))
                    continue
                pairs.append((p, g))
                if lrp <= 0.0:
                    continue
                if gp and (not self._gpids or id(p) in self._gpids):
                    trk.append((p, p.detach().clone()))
                fac, hl2, rd, cd, md = self._pmetrics(p)
                is2 = p.ndim in (2, 4) and md >= 2
                net.append((p, g, lrp, fac, hl2, b1, isg, (rd, cd, md) if is2 else None, is2 and not isg))
        return pairs, net, pit, trk, dev

    @staticmethod
    def _coupling(is_c, rec, dev):
        if not is_c or rec is None or (gfg := float(getattr(rec, "g_func_gain", 1.0))) >= 0.75:
            return 1.0
        has_d = hasattr(rec, "dev_ema")
        if not has_d and not getattr(rec, "margin_initialized", False):
            return 1.0
        em = rec.dev_ema("ema_margin", dev) if has_d else float(getattr(rec, "ema_margin", 0.0))
        mg = smoothstep01((em - 0.28) / 0.22) * smoothstep01((1.40 - em) / 0.30)
        if has_d:
            mg = torch.where(rec.dev_ema("margin_initialized", dev) > 0.5, mg, 0.0)
        return 1.0 + mg * (max(0.12, gfg) - 1.0)

    def _rec_state(self, rec, is_c):
        if rec is None:
            return 1.0, 1.0, False, False, False
        ss, st = float(getattr(rec, "step_scale", 1.0)), float(getattr(rec, "thermal_scale", 1.0))
        simp, cf = bool(getattr(rec, "_simplicial_step", False)), bool(getattr(rec, "cond_initialized", False))
        gp = not is_c and not simp and not cf and len(self._prot or ()) < 2
        gp = (
            gp and getattr(rec, "_g_probe_cache", None) is not None and bool(getattr(rec, "_enable_secant_probe", True))
        )
        return ss, st, simp, cf, gp

    def _sync_step_pack(self, pairs, pit, net, gm, dm, dev):
        hh = self._psk is not None
        sk, ok = self._sketch(pairs, dev)
        cos_t = (sk * self._psk.to(dev, _F32)).sum().double().clamp(-1.0, 1.0) if hh else ok.new_zeros((), dtype=_F64)
        pre_pr = [(lrp, self._prep_prior(p, g, grp, b1p)) for p, g, lrp, b1p, grp in pit]
        svec = [ok.double(), cos_t, gm, _sc(dm, dev)]
        poff = len(svec)
        for _, pr in pre_pr:
            if pr is not None:
                svec.extend((pr[6], pr[7]))
        fidx, ps, gs = [], [], []
        for i, (p, g, *_) in enumerate(net):
            if p.dtype == g.dtype == _F32 and p.device == dev and g.numel() > 0:
                fidx.append(i)
                ps.append(p)
                gs.append(g)
        noff = len(svec)
        if gs:
            svec.extend(n.double() for n in torch._foreach_norm(gs + ps))
        sv = torch.stack(svec).tolist()
        if pairs and (not hh or sv[0] > 0.5):
            self._psk = sk
        nf = len(fidx)
        return hh, sv, poff, pre_pr, dict(zip(fidx, zip(sv[noff : noff + nf], sv[noff + nf : noff + 2 * nf])))

    def _step_core(self, all_p, origs):
        rec, is_c = self._active_record(), self.critic is not None
        ss, st, simp, cf, gp = self._rec_state(rec, is_c)
        sc = 1.0 if st >= 0.9999 else st / hypot(st, 0.0061)
        pyth = self._two_obj_pyth() if (cf and self.direction_blend) else {}
        pairs, net, pit, trk, dev = self._partition(ss * sc, gp, pyth)
        if dev is None:
            self.last_step_gain, self.last_probe_consumed = 1.0, False
            if rec is not None and not is_c:
                rec._g_probe_cache = None
            return False
        efg = self.ema_func_gain
        pg = (_sc(self._mwb, dev) * (1.0 if efg >= 0.75 else efg)).clamp_max(1.0)
        gm, dm = _sc(self._mgs, dev) * pg, self._coupling(is_c, rec, dev)
        de = 0.0005 if st >= 0.5 else 0.0
        hh, sv, poff, pre_pr, nmap = self._sync_step_pack(pairs, pit, net, gm, dm, dev)
        rn = (0.90 * self.ema_step_cos + 0.10 * sv[1]) if (hh and sv[0] > 0.5) else self.ema_step_cos
        gd, gu = smoothstep01((-0.45 - rn) / 0.25), smoothstep01((rn + 0.35) / 0.35)
        self.ema_step_cos, self.log2_damp = rn, max(-4.0, min(0.0, self.log2_damp - 0.12 * gd + 0.25 * gu))
        co, sor = 2.0**self.log2_damp, self.smoothing * smoothstep01((-0.35 - rn) / 0.35)
        ge = smoothstep01((0.55 - rn) / 0.30)
        stf = max(smoothstep01((0.18 - efg) / 0.07), smoothstep01((0.60 - efg) / 0.30) * gd)
        s1 = 1.0 - stf
        sb = self.smoothing * s1
        so, sp = sor * s1, 4.0 * sb
        self._finish_priors(pre_pr, sv, poff, so, sb, sp, ge, de, hh)
        s6 = self._route_critic(net, nmap, st, dev)
        self._step_net(net, nmap, (co, so, sb, sp, ge, de, hh, sv[2], sv[3]), gp, dev)
        self._apply_s6(s6, dm, dev)
        alg, pran = self._finish_trust(rec, is_c, gp, trk, (pg, ss, st, simp), stf)
        if all_p is not None:
            self._apply_dir_blend(all_p, origs)
        self.last_step_gain, self.last_probe_consumed = float(alg), pran
        self.probe_consumptions += int(pran)
        return True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.direction_blend and (closure is not None or self._prot is None):
            self.abort_pending()
            raise ValueError("invalid")
        all_p = self._parameters() if self.direction_blend else None
        origs = [p.detach().clone() for p in all_p] if all_p is not None else None
        try:
            ran = self._step_core(all_p, origs)
        finally:
            self.abort_pending()
        if ran and self.record is not None:
            self.record.record_step(self)
        return loss

    def __getstate__(self):
        return {**self.__dict__, "_mc": {}}

    def state_dict(self):
        res = Optimizer.state_dict(self)
        srows = [
            r.cpu()
            if (g["algorithm"] == "axiom_prior" and (r := self.sampled_rows_for(g["params"][0])) is not None)
            else None
            for g in self.param_groups
        ]
        meta = {k: getattr(self, k) for k, *_ in _SCALARS} | {
            "schema": SCHEMA_VERSION,
            "family": self.family,
            "direction_blend": self.direction_blend,
            "prev_step_sketch": None if self._psk is None else self._psk.detach().cpu().tolist(),
            "sampled_rows": srows,
        }
        if self.direction_blend:
            meta["constraint_geometry"] = dict(
                stats=dict(self.constraint_geometry_stats),
                blend_stats=dict(self.direction_blend_stats),
                pending=_copy_pend(self._prot),
            )
        res["axiom"] = meta
        if self.record is not None:
            res["regularizer"] = dict(
                optimizer_family=self.family, record=self.record.state_dict(), ema=None, guard=None
            )
        return res

    def _val_cg(self, cg):
        _dkeys(cg, {"stats", "blend_stats", "pending"})
        _dkeys(cg["stats"], set(self.constraint_geometry_stats))
        _dkeys(cg["blend_stats"], set(self.direction_blend_stats))
        if (pend := cg["pending"]) is None:
            return
        ps = self._parameters()
        _req(isinstance(pend, list) and 1 <= len(pend) <= 2)
        for grp in pend:
            _req(isinstance(grp, list) and len(grp) == len(ps))
            for t, p in zip(grp, ps):
                _req(t is None or (_tfin(t) and t.shape == p.shape))

    def _val_meta(self, m):
        _dkeys(m, set(_META_KEYS) | ({"constraint_geometry"} if self.direction_blend else set()))
        _req(
            type(m["schema"]) is int
            and m["schema"] == SCHEMA_VERSION
            and m["family"] == self.family
            and type(m["direction_blend"]) is bool
            and m["direction_blend"] == self.direction_blend
        )
        _req(isinstance(m["sampled_rows"], list) and len(m["sampled_rows"]) == len(self.param_groups))
        _req(all(_fin(m[k], lo, hi, slo) for k, lo, hi, slo in _SCALARS))
        if (sk := m["prev_step_sketch"]) is not None:
            _req(isinstance(sk, list) and len(sk) == _SKETCH and all(_fin(x) for x in sk))
        if self.direction_blend:
            self._val_cg(m["constraint_geometry"])

    @staticmethod
    def _val_pstate(sv, sh):
        _req(isinstance(sv, dict) and "m_u" not in sv and type(st := sv.get("step", 0)) is int and st >= 0)
        _req("log2_ref" not in sv or _fin(sv["log2_ref"]))
        _req("log2_init" not in sv or type(sv["log2_init"]) is bool)
        for tk in _FP32:
            if tk in sv:
                _req(_tfin(t := sv[tk]) and (tk not in ("m_unit", "m_prior") or sh is None or t.shape == sh))

    def validate_state_dict(self, saved):
        _dkeys(saved, {"state", "param_groups", "axiom"} | ({"regularizer"} if self.record is not None else set()))
        self._val_meta(saved["axiom"])
        sgr, sst = saved["param_groups"], saved["state"]
        _req(isinstance(sgr, list) and len(sgr) == len(self.param_groups) and isinstance(sst, dict))
        shs = {}
        for c, s in zip(self.param_groups, sgr):
            _req(isinstance(s, dict) and s.get("role") == c["role"] and s.get("algorithm") == c["algorithm"])
            _req(len(sp := s.get("params", ())) == len(c["params"]))
            for p, sid in zip(c["params"], sp):
                shs[sid] = p.shape
        for sid, sv in sst.items():
            self._val_pstate(sv, shs.get(sid))
        if self.record is not None:
            reg = saved["regularizer"]
            _dkeys(reg, {"optimizer_family", "record", "ema", "guard"})
            _req(reg["optimizer_family"] == self.family and reg["ema"] is None and reg["guard"] is None)
            copy(self.record).load_state_dict(reg["record"])

    def _restore_fp32(self, sd):
        for cg, sg in zip(self.param_groups, sd["param_groups"]):
            for p, sid in zip(cg["params"], sg["params"]):
                if sv := sd["state"].get(sid):
                    for tk in _FP32:
                        if isinstance(t := sv.get(tk), _T):
                            self.state[p][tk] = t.detach().to(p.device, _F32).clone()

    def load_state_dict(self, state_dict):
        self.validate_state_dict(state_dict)
        meta = state_dict["axiom"]
        Optimizer.load_state_dict(self, {"state": state_dict["state"], "param_groups": state_dict["param_groups"]})
        self._restore_fp32(state_dict)
        self.clear_sampled_rows()
        self._mc.clear()
        for k, *_ in _SCALARS:
            setattr(self, k, float(meta[k]))
        ps = self._parameters()
        dev = ps[0].device if ps else torch.device("cpu")
        self._psk = None if (sk := meta["prev_step_sketch"]) is None else torch.tensor(sk, dtype=_F32, device=dev)
        self._reset_counts()
        for g, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and g["algorithm"] == "axiom_prior":
                self.set_sampled_rows(g["params"][0], rows)
        self._prot = None
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            self.constraint_geometry_stats, self.direction_blend_stats = dict(cg["stats"]), dict(cg["blend_stats"])
            self._prot = _copy_pend(cg["pending"], ps)
        if self.record is not None:
            self.record.load_state_dict(state_dict["regularizer"]["record"])
            self.record.restore_device_state(dev)


__all__ = [
    "AxiomOptimizer",
    "collect_conv_layouts",
    "fp32_tikhonov_ns_2d",
    "fp32_tikhonov_ns_batched",
    "mantissa_exponent_normalize",
    "zero_pinv_project_nonascent",
]
