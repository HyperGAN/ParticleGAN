"""Axiom L1 critic penalty."""

from contextlib import contextmanager, nullcontext
import math

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

from .math_ops import PHI1, smoothstep01, weyl_rademacher
from .state import (
    AxiomStepRecord,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    bind_axiom_record,
    check_scalar,
    eval_probe_forward,
    extract_axiom_inputs_parts as _extract_axiom_inputs_parts,
    first_tensor_output,
    format_logits_2d,
    record_axiom_capture as _record_axiom_capture,
)

_SECANT_DELTA_REL = 0.02


@torch.no_grad()
def effective_dimension(x):
    b, d = x.shape[0], max(1, x[0].numel())
    if b < 4 or d <= 1:
        return x.new_full((), float(d), dtype=torch.float64)
    xc = x.detach().reshape(b, d).float()
    xc = xc - xc.mean(0, True)
    sx = xc.abs().amax()
    xu = xc / torch.where(sx > 0, sx, sx.new_ones(()))
    gram = (xu @ xu.T).double()
    diag = gram.diagonal()
    s1, s11, s2 = diag.sum(), diag.square().sum(), gram.square().sum()
    a, c = (b - 1) * (b - 2), b * (b - 1)
    num, den = a * s1 * s1 - c * s11 + 2.0 * s2, a * s2 - c * s11 + s1 * s1
    d_est = (num / den.clamp_min(1e-30)).clamp(1.0, d)
    return torch.where((sx > 0) & (num > 0) & (den > 0), d_est, s1.new_full((), float(d)))


def probe_schedule(batch_size, dim):
    d, b = max(1, int(dim)), max(1, int(batch_size))
    k = min(d, max(1, 16 // d))
    return k, min(b, max(min(b, max(2, b // 4)), 32_768 // (d * k)))


@torch.no_grad()
def _coord_scale(x_real, x_fake, record):
    dev, sd = x_real.device, effective_dimension(x_real).sqrt()
    xr, xf = (t.detach().reshape(t.shape[0], -1).double() for t in (x_real, x_fake))
    xm = torch.maximum(xr.abs().amax(), xf.abs().amax())
    sx = torch.where(xm > 0, xm, xm.new_ones(()))
    xru, xfu = xr / sx, xf / sx
    mr, mf = xru.mean(0, True), xfu.mean(0, True)
    vr, vf = ((t - m).square().sum(-1).mean() for t, m in ((xru, mr), (xfu, mf)))
    sp = torch.maximum(torch.maximum(vr, 0.25 * vf), (mr - mf).square().sum())
    tot = torch.where(sp > 0, sp, xru.square().sum(-1).mean())
    s_cur = torch.where(xm > 0, tot.sqrt() * sx, sd)
    raw = torch.where(record.dev_ema("margin_initialized", dev) > 0.5, record.dev_ema("ema_data_scale", dev), s_cur)
    return (raw / sd).clamp(1e-30, 1e30).float()


def _cyclic_slice(val, b, start, count):
    if not isinstance(val, torch.Tensor) or val.ndim < 1 or val.shape[0] != b or count >= b:
        return val
    end = start + count
    return val[start:end] if end <= b else torch.cat((val[start:], val[: end - b]))


def _slice_args(args, kwargs, b, start, count):
    return (
        tuple(_cyclic_slice(v, b, start, count) for v in args),
        {k: _cyclic_slice(v, b, start, count) for k, v in kwargs.items()},
    )


def _secant_sparsity(dim, delta_rel, dtype):
    eps = torch.finfo(dtype if dtype.is_floating_point else torch.float32).eps
    r = delta_rel / max(4.0 * eps, 1e-30)
    return min(dim, max(1, int(math.floor(r * r))))


@torch.no_grad()
def _sparse_rademacher(b, d, s, phase, salt, dev):
    if s >= d:
        return weyl_rademacher((b, d), phase=phase, salt=salt, device=dev, dtype=torch.float32) / math.sqrt(d)
    st = max(1, int(round(d * PHI1))) % d or 1
    while math.gcd(st, d) != 1:
        st = (st + 1) % d or 1
    b0 = int(math.floor((phase % 1.0) * d)) + salt * 104_729
    r = (torch.arange(b, device=dev) * 7_919 + b0) % d
    c = (r[:, None] + torch.arange(s, device=dev)[None, :] * st) % d
    sg = weyl_rademacher((b, s), phase=phase, salt=salt + 31, device=dev, dtype=torch.float32)
    return torch.zeros((b, d), device=dev).scatter_(1, c, sg / math.sqrt(s))


@torch.no_grad()
def _probe_dirs(x, k, phase, salt, sec, dtype):
    b, d, dev = x.shape[0], max(1, x[0].numel()), x.device
    dt = torch.float32 if sec else dtype
    if k == d:
        eye = torch.eye(d, device=dev, dtype=dt)
        return [eye[i].expand(b, d).reshape_as(x) for i in range(d)]
    s = _secant_sparsity(d, _SECANT_DELTA_REL, x.dtype) if sec else d
    u0 = _sparse_rademacher(b, d, s, phase, salt, dev)
    sc = 1.0 if sec else math.sqrt(d)
    if k == 1:
        return [(u0 * sc).to(dt).reshape_as(x)]
    u1 = _sparse_rademacher(b, d, s, phase, salt + (19 if sec else 17), dev)
    if s >= d:
        v = u1 - (u1 * u0).sum(1, keepdim=True) * u0
        n = v.norm(dim=1, keepdim=True)
        u1 = torch.where(n > 1e-6, v / n.clamp_min(1e-6), u1)
    return [(u * sc).to(dt).reshape_as(x) for u in (u0, u1)]


def _bn_dual_bf16_hook(_m, _i, out):
    if isinstance(out, torch.Tensor):
        p, t = fwAD.unpack_dual(out)
        if t is not None and (p.dtype != torch.bfloat16 or t.dtype != torch.bfloat16):
            return fwAD.make_dual(p.to(torch.bfloat16), t.to(torch.bfloat16))
    return out


def _is_bf16(x, mod):
    ac = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
    return bool(x.is_cuda and isinstance(mod, nn.Module) and (getattr(mod, "bf16", False) or ac))


@contextmanager
def _penalty_eval_context(mod, record, *, bf16_dual=False):
    bns, hooks = [], []
    if isinstance(mod, nn.Module):
        for sub in mod.modules():
            is_bn = isinstance(sub, nn.modules.batchnorm._BatchNorm)
            if is_bn:
                bns.append((sub, sub.training))
                sub.eval()
            if bf16_dual and (is_bn or isinstance(sub, nn.modules.instancenorm._InstanceNorm)):
                hooks.append(sub.register_forward_hook(_bn_dual_bf16_hook))
    prev = record._in_penalty
    record._in_penalty = True
    try:
        yield
    finally:
        record._in_penalty = prev
        for h in hooks:
            h.remove()
        for sub, wt in bns:
            sub.training = wt


def _scale_grad(t, s):
    return (t * s + t.detach() * (1.0 - s)) if t.requires_grad else t


class AxiomPenaltyKernel:
    def __init__(
        self,
        *,
        record=None,
        coeff=1.0,
        kappa=1.0,
        lambda_floor=0.005,
        beta_ema=1.5,
        gamma_gauge=0.03,
        max_lambda=2.0,
    ):
        self.record = record or AxiomStepRecord()
        for name, val, sl in (
            ("coeff", coeff, 0),
            ("kappa", kappa, 1),
            ("lambda_floor", lambda_floor, 1),
            ("beta_ema", beta_ema, 0),
            ("gamma_gauge", gamma_gauge, 0),
        ):
            setattr(self, name, check_scalar(val, name, lo=0.0, strict_lo=sl))
        self.max_lambda = check_scalar(max_lambda, "max_lambda", lo=self.lambda_floor)
        self._c_max = math.log(max(1.0, self.max_lambda / self.lambda_floor))
        self._fwad_unsupported = set()

    def _probe_batch(self, live, x, args, kwargs, k, phi, salt, sc_t, sec, bf16):
        b, d = x.shape[0], max(1, x[0].numel())
        dt_in = torch.bfloat16 if (bf16 and not sec) else x.dtype
        xd = x.detach().to(dt_in)
        dt = (sc_t * _SECANT_DELTA_REL).clamp_min(1e-30) if sec else None
        dx, inv = (dt.to(xd.dtype), (0.5 * sc_t) / dt) if sec else (None, None)
        y_out = sq = None
        with nullcontext() if sec else fwAD.dual_level():
            for vec in _probe_dirs(xd, k, phi, salt, sec, dt_in):
                if sec:
                    st = vec.to(xd.dtype) * dx
                    yp, ym = (format_logits_2d(live(z, *args, **kwargs), b) for z in (xd + st, xd - st))
                    y2, j2 = 0.5 * (yp + ym), (yp.float() - ym.float()) * inv
                else:
                    yr, jr = fwAD.unpack_dual(live(fwAD.make_dual(xd, vec), *args, **kwargs))
                    if jr is None:
                        raise NotImplementedError("fwAD dropped tangent")
                    y2, j2 = format_logits_2d(yr, b), format_logits_2d(jr, b).float() * sc_t
                term = j2.square().mean(-1)
                sq = term if sq is None else sq + term
                if y_out is None:
                    y_out = y2
        return y_out, sq * (1.0 if k == d else (d if sec else 1.0) / k)

    def _eval_pair(self, fn, xr, xf, args, kwargs, b_reg, phi):
        br, bf = xr.shape[0], xf.shape[0]
        sr, sf = int(phi * br) % br, int(((phi + 0.5) % 1.0) * bf) % bf
        full = b_reg >= min(br, bf)
        br_s = br if full else max(1, min(br, b_reg // 2))
        bf_s = bf if full else max(1, min(bf, b_reg - br_s))
        xr_s, xf_s = _cyclic_slice(xr.detach(), br, sr, br_s), _cyclic_slice(xf.detach(), bf, sf, bf_s)
        if not args and not kwargs and br == bf and xr.shape == xf.shape:
            try:
                yc, sc = fn(torch.cat((xr_s, xf_s)), (), {}, 1)
                return yc[:br_s], yc[br_s:], sc[:br_s], sc[br_s:]
            except RuntimeError:
                pass

        def call(a, b):
            yr, sqr = fn(*a, 1)
            yf, sqf = fn(*b, 2)
            return yr, yf, sqr, sqf

        try:
            return call(
                (xr_s, *_slice_args(args, kwargs, br, sr, br_s)), (xf_s, *_slice_args(args, kwargs, bf, sf, bf_s))
            )
        except RuntimeError:
            if br_s == br and bf_s == bf:
                raise
            return call((xr.detach(), args, kwargs), (xf.detach(), args, kwargs))

    def _assemble_penalty(self, dr, df, sqr, sqf, *, exact_basis):
        s_hat = 0.5 * (sqr.mean() + sqf.mean())
        k2 = self.kappa * self.kappa
        excess = torch.relu(self.record.dev_ema("ema_grad_sq", s_hat.device) - k2)
        lam = (self.lambda_floor * torch.exp((self.beta_ema * excess).clamp_max(self._c_max))).to(s_hat.dtype)
        self.record.observe_grad_sq_tensor(s_hat.detach())
        ema_g = self.record.dev_ema("ema_grad_sq", s_hat.device).to(s_hat.dtype)
        cap_w = 1.0 if exact_basis else smoothstep01((ema_g / k2 - 0.25) / 0.15)
        slope = lam * s_hat + 0.5 * cap_w * sum(
            torch.relu((q + 1e-12).sqrt() - self.kappa).square().mean() for q in (sqr, sqf)
        )
        p_slope = 0.5 * self.coeff * slope
        if self.gamma_gauge > 0.0:
            ml = 0.5 * (dr.float().mean() + df.float().mean())
            p_gauge = 0.5 * self.coeff * self.gamma_gauge * ml.square()
        else:
            ml = p_gauge = s_hat.new_zeros(())
        s_th = self.record.thermal_scale
        if s_th < 1.0:
            p_slope = _scale_grad(p_slope, s_th)
            sg = s_th + smoothstep01(ml.detach().abs() - 0.5) * (math.sqrt(max(0.0, s_th)) - s_th)
            p_gauge = _scale_grad(p_gauge, sg)
        return p_slope + p_gauge, s_hat, lam, s_th

    def penalty(self, live, x_real, x_fake, *args, collect_stats=False, critic_module=None, **kwargs):
        if self.coeff == 0.0:
            self.record.clear_transient_captures()
            return x_real.new_zeros(()), ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})
        self.record.calls += 1
        phi = self.record.chord_phase
        self.record.chord_phase = (phi + PHI1) % 1.0
        mod = critic_module if isinstance(critic_module, nn.Module) else (live if isinstance(live, nn.Module) else None)
        cid = id(live if mod is None else mod)
        d = max(1, x_real[0].numel())
        k, b_reg = probe_schedule(min(x_real.shape[0], x_fake.shape[0]), d)
        sc_t = _coord_scale(x_real, x_fake, self.record)
        dr_l, df_l = (self.record.find_live_endpoint(x, args, kwargs) for x in (x_real, x_fake))
        reused = dr_l is not None and df_l is not None
        sec = cid in self._fwad_unsupported

        def run(use_s):
            bf16 = False if use_s else _is_bf16(x_real, mod)
            with _penalty_eval_context(mod, self.record, bf16_dual=bf16):
                return self._eval_pair(
                    lambda x, a, kw, s: self._probe_batch(live, x, a, kw, k, phi, s, sc_t, use_s, bf16),
                    x_real,
                    x_fake,
                    args,
                    kwargs,
                    b_reg,
                    phi,
                )

        try:
            if not sec:
                try:
                    dr_s, df_s, sqr, sqf = run(False)
                except (RuntimeError, NotImplementedError, TypeError):
                    self._fwad_unsupported.add(cid)
                    sec = True
            if sec:
                dr_s, df_s, sqr, sqf = run(True)
            dr_u, df_u = (dr_l, df_l) if reused else (dr_s, df_s)
            pen, s_hat, lam, s_th = self._assemble_penalty(dr_u, df_u, sqr, sqf, exact_basis=(k == d))
            if not collect_stats:
                return pen, {}
            pv, sv, lv = torch.stack((pen.detach().float(), s_hat.detach().float(), lam.detach().float())).tolist()
            return pen, {
                "applied": True,
                "pen": pv,
                "s_hat": sv,
                "mean_grad_norm": math.sqrt(max(0.0, sv)),
                "lambda_eff": lv,
                "center": self.kappa,
                "regime": "secant" if sec else "fwad",
                "k_probes": k,
                "b_reg": b_reg,
                "s_thermal": s_th,
                "live_endpoints_reused": reused,
            }
        finally:
            self.record.clear_transient_captures()


class AxiomCriticPenalty:
    ema_critic = None

    def __init__(
        self,
        optimizer_or_recipe,
        optimizer=None,
        *,
        coeff=None,
        kappa=None,
        output=None,
        collect_stats=False,
        **options,
    ):
        self.recipe, opt = (None, optimizer_or_recipe) if optimizer is None else (optimizer_or_recipe, optimizer)
        if not hasattr(opt, "param_groups"):
            raise TypeError("critic optimizer required")
        rec = getattr(opt, "record", None) or AxiomStepRecord()
        self.optimizer, opt.record, opt._axiom_record = opt, rec, rec
        self.critic = getattr(opt, "critic", None)
        _attach_axiom_critic_hook(self.critic, rec)
        self.regularizer = AxiomPenaltyKernel(
            record=rec,
            coeff=float(getattr(self.recipe, "reg_coeff", 1.0) if coeff is None else coeff),
            kappa=float(getattr(self.recipe, "reg_kappa", 1.0) if kappa is None else kappa),
            **{
                k: options[k]
                for k in ("lambda_floor", "beta_ema", "gamma_gauge", "max_lambda")
                if options.get(k) is not None
            },
        )
        self.initial_coeff = self.regularizer.coeff
        self.output = output or first_tensor_output
        self.collect_stats = bool(collect_stats)
        self.last_stats = {}

    def __call__(self, critic, x_real, x_fake, *args, **kwargs):
        pen, self.last_stats = self.regularizer.penalty(
            lambda x, *a, **kw: self.output(critic(x, *a, **kw)),
            x_real,
            x_fake,
            *args,
            collect_stats=self.collect_stats,
            critic_module=critic if isinstance(critic, nn.Module) else self.critic,
            **kwargs,
        )
        return pen

    def diagnostics(self):
        r = self.regularizer.record
        return {
            "axiom_calls": int(r.calls),
            **{
                f"axiom_{k}": float(getattr(r, k))
                for k in ("ema_grad_sq", "chord_phase", "thermal_scale", "step_scale")
            },
        }


__all__ = [
    "AxiomCriticPenalty",
    "AxiomPenaltyKernel",
    "AxiomStepRecord",
    "_attach_axiom_critic_hook",
    "_attach_axiom_generator_hook",
    "_extract_axiom_inputs_parts",
    "_record_axiom_capture",
    "bind_axiom_record",
    "effective_dimension",
    "eval_probe_forward",
    "probe_schedule",
]
