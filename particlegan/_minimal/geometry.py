"""Half A1+A2: 1st-Order fwAD/Secant Critic, Debiased Transport, Burg Covariance & Zero-eigh Conditional Fiber."""
from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from .._stats import pairwise_sq_dist
from ..gan_loss import GANLoss
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .dynamics import FROZEN_THERMAL_SCALE, MinimalCriticOptimizer, MinimalStepRecord, _attach_minimal_critic_hook, _cond_signature, _smoothstep01, _tensor_key

_PHI1, _PHI2 = 0.6180339887498949, 0.3247179572447460

def _protected_softplus_neg(delta: torch.Tensor, tau: float = 3.5) -> torch.Tensor:
    d32 = delta.float()
    with torch.no_grad(): s = torch.sqrt(1.0 + d32.detach().square().mean() / (tau * tau))
    return (s * F.softplus(-d32 / s)).to(delta.dtype)

@torch.no_grad()
def _sinkhorn_log(cost: torch.Tensor, scale: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
    nr, nc = int(cost.shape[0]), int(cost.shape[1])
    la, lb, lk = -math.log(max(1, nr)), -math.log(max(1, nc)), -(cost / scale.clamp_min(1e-12)) / max(1e-4, float(eps))
    u, v = torch.zeros(nr, device=cost.device), torch.zeros(nc, device=cost.device)
    for _ in range(max(int(iters), 7) if eps < 0.03 else int(iters)):
        u, v = la - torch.logsumexp(lk + v.unsqueeze(0), 1), lb - torch.logsumexp(lk + u.unsqueeze(1), 0)
    pi = torch.exp(lk + u.unsqueeze(1) + v.unsqueeze(0))
    return pi / (float(nr) * pi.sum(1, keepdim=True).clamp_min(1e-12))

def _eval_lowd_fwad_sq(live: Any, live_rep: Any, x: torch.Tensor, phase: float, can_stack: bool, quantized: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    b, dim, xd = int(x.shape[0]), int(x.shape[1]), x.detach()
    if dim <= 4: basis = torch.eye(dim, device=x.device, dtype=x.dtype)
    else:
        v = torch.cos(2.0 * math.pi * (phase + (torch.arange(dim, device=x.device, dtype=torch.float32) + 0.5) * _PHI1))
        vn = v / v.norm().clamp_min(1e-6); basis = (torch.eye(dim, device=x.device) - 2.0 * torch.outer(vn, vn)).to(x.dtype)
    if not quantized:
        try:
            with fwAD.dual_level():
                if can_stack:
                    po, jo = fwAD.unpack_dual(live_rep(fwAD.make_dual(xd.unsqueeze(0).expand(dim, b, dim).reshape(dim * b, dim), basis.unsqueeze(1).expand(dim, b, dim).reshape(dim * b, dim)), dim))
                    if jo is not None: return jo.reshape(dim, b, -1).mean(-1).square().sum(0), po.reshape(dim, b, -1)[0].mean(-1)
                sq, p0 = x.new_zeros((b,)), None
                for k in range(dim):
                    pk, jk = fwAD.unpack_dual(live(fwAD.make_dual(xd, basis[k].unsqueeze(0).expand(b, dim))))
                    if jk is None: raise RuntimeError("fwAD unavailable")
                    p0, sq = (pk.reshape(b, -1).mean(-1) if p0 is None else p0), sq + jk.reshape(b, -1).mean(-1).square()
                return sq, p0
        except (RuntimeError, NotImplementedError, TypeError): pass
    p0, sq = live(xd).reshape(b, -1).mean(-1), x.new_zeros((b,))
    for k in range(dim): tk = basis[k].unsqueeze(0).expand(b, dim); sq = sq + ((live(xd + 1e-3 * tk).reshape(b, -1).mean(-1) - live(xd - 1e-3 * tk).reshape(b, -1).mean(-1)) / 2e-3).square()
    return sq, p0

def _unit_probes(x: torch.Tensor, phase: float, k_probes: int) -> List[torch.Tensor]:
    b, dim = int(x.shape[0]), max(1, x[0].numel())
    if x.ndim >= 3 and dim > 1024:
        u = x.new_ones(x.shape, dtype=torch.float32)
        for ax, sz in enumerate(x.shape):
            u = u * torch.where(torch.sin(2.0 * math.pi * (phase + (ax + 1) * _PHI2 + (torch.arange(sz, device=x.device, dtype=torch.float32) + 0.5) * _PHI1)) >= 0.0, 1.0, -1.0).view((1,) * ax + (sz,) + (1,) * (x.ndim - 1 - ax))
        return [(u / math.sqrt(float(dim))).to(x.dtype)]
    probes, idx = [], torch.arange(b * dim, device=x.device, dtype=torch.float32)
    for k in range(k_probes):
        u = torch.where(torch.sin((idx + 1.0) * (12.9898 + 78.233 * (k + 1)) + 43758.5453 * (phase + (k + 1) * _PHI1)) >= 0.0, 1.0, -1.0).view(b, dim)
        for prev in probes: pf = prev.view(b, dim).float(); u = u - (u * pf).sum(1, keepdim=True) * pf
        probes.append((u / u.norm(dim=1, keepdim=True).clamp_min(1e-6)).view_as(x).to(x.dtype))
    return probes

class MinimalCriticPenalty(_K3PCriticPenalty):
    def __init__(self, recipe: Any, optimizer: Any, *, output: Any = None, collect_stats: bool = False, **overrides: Any) -> None:
        from ..discriminators import BatchDistanceDiscriminator
        from ..tensorflow_adam import TensorFlowV1Adam
        if not (isinstance(optimizer, (MinimalCriticOptimizer, K3PCriticAdam)) or (type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam) and hasattr(optimizer, "record"))):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer")
        self.recipe, self.optimizer, self.critic, self.record = recipe, optimizer, optimizer.critic, optimizer.record
        _attach_minimal_critic_hook(self.critic, self.record); opts = recipe._penalty_options(**overrides)
        self.arm, self.coeff, self.kappa, self.lazy_k = "minimal", float(opts.get("coeff", 1.0)), float(opts.get("kappa", 1.0)), int(opts.get("lazy_k", 1))
        self.ebi_beta, self.secant_eps = float(opts.get("ebi_beta", 8.0)), float(opts.get("secant_eps", 1e-5))
        self.regularizer, self.initial_coeff, self.output, self.collect_stats, self.last_stats, self._verified = self, self.coeff, (_first_output if output is None else output), bool(collect_stats), {}, set()
        self._names = {id(m): n for n, m in self.critic.named_modules()}
        self._has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules())
        self._has_bd = any(isinstance(m, BatchDistanceDiscriminator) for m in self.critic.modules())
        self._has_q = any(getattr(m, "format", None) in ("nvfp4", "mxfp8") or "Emulated" in type(m).__name__ for m in self.critic.modules())

    @torch.no_grad()
    def _batch_res(self, xr: torch.Tensor, xf: torch.Tensor) -> Tuple[float, float]:
        dim = max(1, xr[0].numel())
        if xr.ndim == 4 and dim > 1024:
            rf, ff = F.adaptive_avg_pool2d(xr.detach().float(), (4, 4)).flatten(1), F.adaptive_avg_pool2d(xf.detach().float(), (4, 4)).flatten(1)
            ref = rf.var(0, unbiased=False).sum().clamp_min(1e-4)
            return float(((rf - ff).square().sum(1).mean() / ref).sqrt().item()), float((ref * (dim / 48.0)).sqrt().item())
        rf, ff = xr.detach().flatten(1).float(), xf.detach().flatten(1).float()
        if len(rf) > 256: rf = rf[(torch.arange(256, device=rf.device) * len(rf)) // 256]
        if len(ff) > 256: ff = ff[(torch.arange(256, device=ff.device) * len(ff)) // 256]
        rr = torch.maximum(pairwise_sq_dist(rf).median(), rf.var(0, unbiased=False).sum()) if len(rf) >= 4 else rf.new_zeros(())
        ref = torch.where(rr > 1e-6, rr.clamp_min(1e-12), rf.new_full((), 0.04)); crf = pairwise_sq_dist(rf, ff) / ref
        return float(torch.sqrt((0.5 * (crf.min(1).values.mean() + crf.min(0).values.mean())).clamp_min(0.0)).item()), float(ref.sqrt().item())

    def __call__(self, critic: nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, *cond: Any, **ckw: Any) -> torch.Tensor:
        from ..recipe_schedules import apply_penalty_schedule
        if id(critic) not in self._verified: self._ema_view(critic); self._verified.add(id(critic))
        if self.recipe is not None: apply_penalty_schedule(self.record.observed_steps, self.recipe, self)
        if self.coeff == 0.0 or (self.lazy_k > 1 and (self.record.observed_steps + 1) % self.lazy_k != 0):
            self.last_stats = {"applied": False, "pen": 0.0, "center": self.kappa} if self.collect_stats else {}; return x_real.new_zeros(())
        rec, out_fn, b0, dim = self.record, self.output, int(x_real.shape[0]), max(1, x_real[0].numel())
        rec.calls += 1; rec.chord_phase = (rec.chord_phase + _PHI1) % 1.0
        has_bs = any(isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == b0 for v in vars(critic).values())
        is_cond, cond_sig = bool(has_bs or cond or ckw), _cond_signature((x_real, *cond), ckw)
        allow_reuse, can_cat = (out_fn is _first_output and not has_bs), (not self._has_bn and not self._has_bd and not has_bs and float(getattr(critic, "std", 0.0)) == 0.0 and out_fn is _first_output)
        res, d_scale = self._batch_res(x_real, x_fake); rec.observe_transport(res, d_scale, is_conditioned=is_cond)
        live = lambda x: out_fn(critic(x, *cond, **ckw))
        rep = lambda t, m: t.repeat((m,) + (1,) * (t.ndim - 1)) if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == b0 else t
        live_rep = lambda x, m: live(x) if (m <= 1 or (not cond and not ckw) or x.shape[0] != m * b0) else out_fn(critic(x, *(rep(t, m) for t in cond), **{k: rep(v, m) for k, v in ckw.items()}))
        rec._in_penalty = True
        try:
            if x_real.ndim == 2 and dim <= 8:
                gsq_r, d_r = _eval_lowd_fwad_sq(live, live_rep, x_real, rec.chord_phase, can_cat, self._has_q)
                gsq_f, d_f = _eval_lowd_fwad_sq(live, live_rep, x_fake, (rec.chord_phase + _PHI2) % 1.0, can_cat, self._has_q)
            else:
                k_p, cur = (2 if dim <= 1024 else 1), int(rec.observed_steps)
                _find = lambda x, exc=None: next(((xa, out) for st, sk, sg, xa, out, al in reversed(rec._live_captures) if al and st == cur and sk == _tensor_key(x) and sg == cond_sig and out is not exc and out.requires_grad and out.grad_fn is not None), None)
                gr = _find(x_real) if allow_reuse else None; gf = _find(x_fake, None if gr is None else gr[1]) if allow_reuse else None
                xr_a, dr_raw = (x_real.detach(), gr[1]) if gr is not None else (x_real.detach(), live(x_real.detach()))
                xf_a, df_raw = (x_fake.detach(), gf[1]) if gf is not None else (x_fake.detach(), live(x_fake.detach()))
                d_r, d_f, eps_s = dr_raw.reshape(b0, -1).mean(-1), df_raw.reshape(x_fake.shape[0], -1).mean(-1), self.secant_eps * max(1.0, d_scale / math.sqrt(float(dim)))
                if dim > 1024 and x_real.shape == x_fake.shape:
                    m_p = ((torch.arange(b0, device=x_real.device) + rec.calls) & 1) == 0; z_b = torch.where(m_p.view((b0,) + (1,) * (x_real.ndim - 1)), xr_a, xf_a)
                    gsq_r = gsq_f = ((live(z_b + eps_s * _unit_probes(z_b, rec.chord_phase, 1)[0]).reshape(b0, -1).mean(-1) - torch.where(m_p, d_r, d_f)) / eps_s).square() * float(dim)
                elif can_cat and k_p == 1 and x_real.shape == x_fake.shape and dim <= 4096:
                    dpr, dpf = live(torch.cat([xr_a + eps_s * _unit_probes(xr_a, rec.chord_phase, 1)[0], xf_a + eps_s * _unit_probes(xf_a, (rec.chord_phase + _PHI2) % 1.0, 1)[0]], 0)).reshape(2 * b0, -1).mean(-1).chunk(2, 0)
                    gsq_r, gsq_f = ((dpr - d_r) / eps_s).square() * float(dim), ((dpf - d_f) / eps_s).square() * float(dim)
                else:
                    gsq_r, gsq_f, sk = x_real.new_zeros((b0,)), x_fake.new_zeros((x_fake.shape[0],)), float(dim) / float(k_p)
                    for ur, uf in zip(_unit_probes(xr_a, rec.chord_phase, k_p), _unit_probes(xf_a, (rec.chord_phase + _PHI2) % 1.0, k_p)):
                        gsq_r = gsq_r + ((live(xr_a + eps_s * ur).reshape(b0, -1).mean(-1) - d_r) / eps_s).square() * sk
                        gsq_f = gsq_f + ((live(xf_a + eps_s * uf).reshape(x_fake.shape[0], -1).mean(-1) - d_f) / eps_s).square() * sk
        finally: rec._in_penalty = False
        lam = min(1.0, max(0.01, self.coeff * (0.25 + 0.75 * rec._step_scale)))
        bi = 0.5 * (((torch.sqrt(1.0 + self.ebi_beta * gsq_r.float()) - 1.0) / self.ebi_beta).mean() + ((torch.sqrt(1.0 + self.ebi_beta * gsq_f.float()) - 1.0) / self.ebi_beta).mean())
        nc = min(b0, int(d_f.shape[0]))
        chord = ((d_r[:nc].float() - d_f[:nc].float()) / (x_real.detach()[:nc].flatten(1).float() - x_fake.detach()[:nc].flatten(1).float()).norm(dim=1).clamp_min(1e-3 * max(1.0, d_scale))).square().mean()
        pen = (lam * (bi + 0.08 * chord) + 0.03 * (d_r.float().mean().square() + d_f.float().mean().square())).to(x_real.dtype)
        self.last_stats = {"applied": True, "pen": float(pen.detach().item()), "center": self.kappa} if self.collect_stats else {}
        return pen

@torch.no_grad()
def _born_infeld_disp(x_f: torch.Tensor, x_r: torch.Tensor, ema_res: float, c_d: float, phase: float) -> Tuple[torch.Tensor, torch.Tensor]:
    ff, rf = x_f.detach().flatten(1).float(), x_r.detach().flatten(1).float(); bf, br, dim = int(ff.shape[0]), int(rf.shape[0]), int(ff.shape[1])
    if dim == 1 and br >= 8:
        rs, fo, pos = torch.sort(rf[:, 0]).values, torch.argsort(ff[:, 0]), torch.linspace(0.0, float(br - 1), bf, device=rf.device)
        lo, disp = pos.floor().long().clamp(0, br - 1), torch.empty_like(ff[:, 0]); disp[fo] = ((1.0 - (pos - lo.float())) * rs[lo] + (pos - lo.float()) * rs[(lo + 1).clamp(0, br - 1)]) - ff[fo, 0]
        rstd = rf[:, 0].std(unbiased=(br > 1)) if br > 1 else rf.new_ones(())
        return disp.view_as(x_f), torch.where(rstd > 1e-4, rstd, rf.new_ones(())).clamp_min(1e-3)
    if br > 512: rf, br = rf[(torch.arange(512, device=rf.device) * br) // 512], 512
    c_rf = pairwise_sq_dist(rf, ff); rr = torch.maximum(pairwise_sq_dist(rf).median(), rf.var(0, unbiased=False).sum()) if br >= 4 else rf.new_zeros(())
    ref = torch.where(rr > 1e-6, rr.clamp_min(1e-12), rf.new_ones(()))
    c_sc, eps_e = torch.maximum(torch.where(rr > 1e-6, ref, c_rf.median().clamp_min(1e-12)), c_rf.median().clamp_min(1e-12)), max(0.025, min(0.12, 0.12 * (0.25 + 0.75 * min(1.0, ema_res / 0.25))))
    pi_rf, wq = _sinkhorn_log(c_rf, c_sc, eps_e, 6), max(_smoothstep01((ema_res - 0.14) / 0.12), _smoothstep01((0.85 - c_d) / 0.45))
    vq = rf[torch.searchsorted(torch.cumsum((cp := (bf * pi_rf.t()).clamp_min(1e-12)) / cp.sum(1, keepdim=True), 1), torch.remainder((torch.arange(bf, device=ff.device, dtype=torch.float32) + 0.5) / bf + phase + 0.5 * _PHI1, 1.0).unsqueeze(1)).squeeze(1).clamp(0, br - 1)] - ff if wq > 1e-3 else None
    vf = ((pi_rf / pi_rf.sum(0, keepdim=True).clamp_min(1e-12)).t() @ rf) - ((pff := _sinkhorn_log(pairwise_sq_dist(fs := (ff if bf <= 512 else ff[(torch.arange(512, device=ff.device) * bf) // 512]), ff), c_sc, eps_e, 6)) / pff.sum(0, keepdim=True).clamp_min(1e-12)).t() @ fs if wq < 0.999 else None
    return (vq if vf is None else (vf if vq is None else wq * vq + (1.0 - wq) * vf)).view_as(x_f), ref.sqrt().clamp_min(1e-3)

def _burg_cov_loss(x_f: torch.Tensor, x_r: torch.Tensor, ema_res: float, phase: float) -> torch.Tensor:
    ff, rf = x_f.flatten(1).float(), x_r.detach().flatten(1).float(); bf, dim = int(ff.shape[0]), int(ff.shape[1])
    if not x_f.requires_grad or bf < 16 or len(rf) < 16 or not (1 <= dim <= 8): return x_f.new_zeros(())
    if dim == 1:
        rm, rv = rf.mean(), rf.var(unbiased=True).clamp_min(1e-6); vr, vf = 1.03 * rv, ff.var(unbiased=True) + 0.03 * rv
        return (_smoothstep01((0.80 - ema_res) / 0.45) * rv.sqrt() * ((torch.sqrt(1.0 + (vr / vf - torch.log(vr / vf) - 1.0).clamp_min(0.0)) - 1.0) + 0.25 * (torch.sqrt(1.0 + (ff.mean() - rm).square() / rv) - 1.0))).to(x_f.dtype)
    rg = _smoothstep01((0.40 - ema_res) / 0.25)
    if rg <= 1e-6: return x_f.new_zeros(())
    with torch.no_grad():
        r_s = rf if len(rf) <= 512 else rf[(torch.arange(512, device=rf.device) * len(rf) // 512 + int(phase * len(rf))) % len(rf)]
        f_idx = None if bf <= 512 else (torch.arange(512, device=ff.device) * bf // 512 + int((phase + _PHI1) * bf)) % bf
        f_s, nr, crr = (ff.detach() if f_idx is None else ff.detach()[f_idx]), len(r_s), pairwise_sq_dist(r_s)
        h2 = torch.topk(crr, k=max(5, min(14, nr // 16)) + 1, dim=1, largest=False).values[:, -1].clamp_min(1e-6)
        crr_c, h2_c, unc, md2, sel, cur = crr.cpu(), h2.cpu(), torch.ones(nr, dtype=torch.bool), torch.full((nr,), float("inf")), [], int(phase * nr) % nr
        for _ in range(min(64, nr)):
            sel.append(cur); md2, unc = torch.minimum(md2, crr_c[cur]), unc & (crr_c[cur] > 4.5 * torch.maximum(h2_c[cur], h2_c))
            if not bool(unc.any()): break
            cur = int(torch.where(unc, md2, -1.0).argmax().item())
        st_i = torch.tensor(sel, device=rf.device, dtype=torch.long)
        d2_ar, d2_af, mg = crr[st_i], pairwise_sq_dist(r_s[st_i], f_s), torch.arange(len(sel), device=rf.device).unsqueeze(1)
        mr_f, mf_f, rad2 = d2_ar.argmin(0).unsqueeze(0) == mg, d2_af.argmin(0).unsqueeze(0) == mg, (3.0 * h2[st_i]).unsqueeze(1)
    fl, tot = (ff if f_idx is None else ff[f_idx]), ff.new_zeros(())
    for mr, mf in ((mr_f, mf_f), (mr_f & (d2_ar <= rad2), mf_f & (d2_af <= rad2))):
        with torch.no_grad():
            cr, cf = mr.sum(1).float(), mf.sum(1).float(); ok = (cr >= 4.0) & (cf >= 4.0)
            if not bool(ok.any()): continue
            wr, wf = mr.float() / cr.clamp_min(1.0).unsqueeze(1), mf.float() / cf.clamp_min(1.0).unsqueeze(1); mu_r = wr @ r_s; dr = r_s.unsqueeze(0) - mu_r.unsqueeze(1)
            cov_r = (cr / (cr - 1.0).clamp_min(1.0)).view(-1, 1, 1) * torch.einsum("mb,mbd,mbe->mde", wr, dr, dr)
            vs = (cov_r.diagonal(dim1=-2, dim2=-1).sum(-1) / dim).clamp_min(1e-6); ok = ok & (vs > 1e-5)
            if not bool(ok.any()): continue
            eye_r = (0.04 * vs).view(-1, 1, 1) * torch.eye(dim, device=rf.device).unsqueeze(0)
        mu_f = wf @ fl; df = fl.unsqueeze(0) - mu_f.unsqueeze(1)
        sr, sf = cov_r + eye_r, (cf / (cf - 1.0).clamp_min(1.0)).view(-1, 1, 1) * torch.einsum("mb,mbd,mbe->mde", wf, df, df) + eye_r
        if dim == 2:
            dr2, df2 = (sr[:, 0, 0] * sr[:, 1, 1] - sr[:, 0, 1].square()).clamp_min(1e-18), (sf[:, 0, 0] * sf[:, 1, 1] - sf[:, 0, 1].square()).clamp_min(1e-18); lr = torch.log(df2) - torch.log(dr2)
            db = ((sf[:, 1, 1] * sr[:, 0, 0] + sf[:, 0, 0] * sr[:, 1, 1] - 2.0 * sf[:, 0, 1] * sr[:, 0, 1]) / df2 + lr - 2.0).clamp_min(0.0) + 0.5 * ((sr[:, 1, 1] * sf[:, 0, 0] + sr[:, 0, 0] * sf[:, 1, 1] - 2.0 * sr[:, 0, 1] * sf[:, 0, 1]) / dr2 - lr - 2.0).clamp_min(0.0)
        else:
            l_r, l_f = torch.linalg.cholesky_ex(sr, check_errors=False)[0], torch.linalg.cholesky_ex(sf, check_errors=False)[0]
            lr = 2.0 * (torch.log(l_f.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)).sum(-1) - torch.log(l_r.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)).sum(-1))
            db = (torch.cholesky_solve(sr, l_f).diagonal(dim1=-2, dim2=-1).sum(-1) + lr - float(dim)).clamp_min(0.0) + 0.5 * (torch.cholesky_solve(sf, l_r).diagonal(dim1=-2, dim2=-1).sum(-1) - lr - float(dim)).clamp_min(0.0)
        ef = vs.sqrt() * ((torch.sqrt(1.0 + db) - 1.0) + 0.25 * (torch.sqrt(1.0 + (mu_f - mu_r).square().sum(-1) / (dim * vs)) - 1.0)) * ok.float()
        tot = tot + 0.5 * (0.65 * (ef.sum() / ok.float().sum().clamp_min(1.0)) + 0.35 * ef.amax())
    return (rg * tot).to(x_f.dtype)

def _sym2_ops(m: torch.Tensor, split: bool = False):
    if not split:
        a, b, d = m[0, 0], 0.5 * (m[0, 1] + m[1, 0]), m[1, 1]
        tr, disc = (a + d).clamp_min(1e-6), torch.sqrt(((a - d) * 0.5).square() + b * b); fl = max(1e-4 * float(tr.item()) * 0.5, 1e-6)
        l1, l2 = (0.5 * tr + disc).clamp_min(fl), (0.5 * tr - disc).clamp_min(fl); s1, s2 = l1.rsqrt(), l2.rsqrt()
        c1 = torch.where((l1 - l2) > 1e-10, (s1 - s2) / (l1 - l2).clamp_min(1e-12), -0.5 * l1.pow(-1.5))
        return (s1 - c1 * l1) * torch.eye(2, dtype=m.dtype, device=m.device) + c1 * torch.stack((torch.stack((a, b)), torch.stack((b, d))))
    a, b, d = m[:, 0, 0], 0.5 * (m[:, 0, 1] + m[:, 1, 0]), m[:, 1, 1]
    tr, diff, disc = 0.5 * (a + d), 0.5 * (a - d), torch.sqrt((0.5 * (a - d)).square() + b * b + 1e-24)
    l1, l2, u1, v1 = tr + disc, tr - disc, torch.where(diff >= 0, diff + disc, b), torch.where(diff >= 0, b, disc - diff)
    n1 = torch.sqrt(u1 * u1 + v1 * v1).clamp_min(1e-12); u1, v1 = u1 / n1, v1 / n1
    mk = lambda p1, p2: torch.stack((torch.stack((p1 * u1 * u1 + p2 * v1 * v1, (p1 - p2) * u1 * v1), -1), torch.stack(((p1 - p2) * u1 * v1, p1 * v1 * v1 + p2 * u1 * u1), -1)), -2)
    return mk(l1.clamp_min(0.0), l2.clamp_min(0.0)), mk(l1.clamp_max(0.0), l2.clamp_max(0.0))

def _cond_fiber_loss(x_f: torch.Tensor, x_r: torch.Tensor, cond_t: Tuple[torch.Tensor, ...], rec: Optional[MinimalStepRecord]) -> torch.Tensor:
    if not x_f.requires_grad or rec is None: return x_f.new_zeros(())
    b, dx = int(x_f.shape[0]), int(x_f.flatten(1).shape[1])
    fl_c = [t.detach().reshape(b, -1).double() for t in cond_t if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == b and t.is_floating_point()]
    disc_c = [t.detach().reshape(b, -1).long() for t in cond_t if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == b and not t.is_floating_point()]
    if not fl_c or dx < 1: return x_f.new_zeros(())
    c_cpu = torch.cat(fl_c, 1).cpu(); dc = int(c_cpu.shape[1])
    if dc > 32: return x_f.new_zeros(())
    xf_c, xr_c, groups = x_f.detach().flatten(1).double().cpu(), x_r.detach().flatten(1).double().cpu(), {}
    for i, r in enumerate(torch.cat(disc_c, 1).cpu().tolist() if disc_c else [(0,)] * b): groups.setdefault(tuple(int(v) for v in r), []).append(i)
    bank = rec.cond_bank
    if bank["d_c"] != dc or bank["d_x"] != dx: bank["d_c"], bank["d_x"], bank["strata"] = dc, dx, []
    smap, pd = {e["key"]: e for e in bank["strata"]}, ((1 + 3 * dc + (dc * (dc - 1)) // 2) if dc <= 6 else (2 + dc))
    ridge = torch.full((pd,), 0.025, dtype=torch.float64); ridge[: 1 + dc] = 1e-3
    def _phi(c64, mu, sd):
        u = (c64 - mu.unsqueeze(0)) / sd.unsqueeze(0); om, ones = 1.0 + 0.90 * u.square().mean(1).clamp_max(4.5), torch.ones((c64.shape[0], 1), dtype=torch.float64)
        if dc <= 6:
            pts = [ones, u, 2.0 * (F.softplus(u - 0.8) - 0.425), 2.0 * (F.softplus(-u - 0.8) - 0.425)]
            if dc >= 2: ia, ib = torch.triu_indices(dc, dc, offset=1); pts.append((u[:, ia] * u[:, ib]) * torch.rsqrt(1.0 + 0.15 * (u[:, ia].square() + u[:, ib].square())))
            return torch.cat(pts, 1), om, u
        return torch.cat((ones, u, 2.0 * (F.softplus(u.square().mean(1).clamp_min(1e-6).sqrt() - 1.0).unsqueeze(1) - 0.3133)), 1), om, u
    xf_l, xr_l, tot, r_op, r_imq_sum, tn = x_f.flatten(1).float(), x_r.detach().flatten(1).float(), x_f.new_zeros(()), 0.0, 0.0, 0
    for key, idxs in groups.items():
        if len(idxs) < 4: continue
        it, n, st = torch.tensor(idxs, dtype=torch.long), len(idxs), smap.get(key); cb, xb, yb = c_cpu[it], xf_c[it], xr_c[it]
        if st is not None and st["ring_c"].shape[0] >= max(24, int(2.5 * pd)):
            rc, rx, ry = st["ring_c"].double(), st["ring_x"].double(), st["ring_y"].double(); m_r = int(rc.shape[0])
            age = torch.arange(m_r - 1, -1, -1, dtype=torch.float64); rw = torch.exp(-age / 192.0); rwn = rw / rw.sum()
            mu_c = (rwn.unsqueeze(1) * rc).sum(0); sd_c = (rwn.unsqueeze(1) * (rc - mu_c).square()).sum(0).clamp_min(1e-4).sqrt()
            pr, omr, ur = _phi(rc, mu_c, sd_c); phi, om, u_s = _phi(cb, mu_c, sd_c); wr = rw * omr; w_eff = float(wr.sum().item()); wp = pr * wr.unsqueeze(1)
            ls = torch.linalg.cholesky_ex(0.5 * (wp.T @ pr + (wp.T @ pr).T) / w_eff + torch.diag(ridge), check_errors=False)[0]
            bxs, bys = torch.cholesky_solve((wp.T @ rx) / w_eff, ls), torch.cholesky_solve((wp.T @ ry) / w_eff, ls)
            bp = 0.5 * (bxs + bys); mp, mx, mpr = phi @ bp, phi @ bxs, pr @ bp
            cp = 0.5 * (((rx - mpr) * wr.unsqueeze(1)).T @ (rx - mpr) + ((ry - mpr) * wr.unsqueeze(1)).T @ (ry - mpr)) / w_eff
            wt = _sym2_ops(cp) if dx == 2 else torch.cholesky_solve(torch.eye(dx, dtype=torch.float64), torch.linalg.cholesky_ex(0.5 * (cp + cp.T) + max(1e-4 * float(cp.diagonal().clamp_min(0.0).mean().item()), 1e-6) * torch.eye(dx, dtype=torch.float64), check_errors=False)[0])
            r_op = max(r_op, max(0.0, float((((bxs - bys) @ wt).square().sum() / dx).item()) - 2.0 * pd / max(float(m_r), 1.0)))
            dmw = (phi @ (bxs - bys)) @ wt; dmn = dmw.norm(dim=1, keepdim=True).clamp_min(1e-12); dms = (3.0 * torch.tanh(dmn / 3.0) * (dmw / dmn)).to(device=x_f.device, dtype=torch.float32)
            d2 = torch.cdist(u_s, ur).square(); h2 = torch.topk(d2, k=min(20, m_r), dim=1, largest=False).values[:, -1].clamp_min(0.15)
            al = torch.softmax(-d2 / (2.0 * h2.unsqueeze(1)) - age.unsqueeze(0) / 128.0, dim=1)
            ew = torch.einsum("ac,icd,db->iab", wt, torch.einsum("ij,ja,jb->iab", al, rx - mpr, rx - mpr) - torch.einsum("ij,ja,jb->iab", al, ry - mpr, ry - mpr), wt)
            fn = ew.flatten(1).norm(dim=1).view(n, 1, 1).clamp_min(1e-12); ew = 1.5 * torch.tanh(fn / 1.5) * (ew / fn)
            ep_t, em_t = tuple(t.to(device=x_f.device, dtype=torch.float32) for t in _sym2_ops(ew, split=True)) if dx == 2 else (ew.to(device=x_f.device, dtype=torch.float32), None)
            id_d, wt_d, mp_d, mx_d, om_d = it.to(x_f.device), wt.to(device=x_f.device, dtype=torch.float32), mp.to(device=x_f.device, dtype=torch.float32), mx.to(device=x_f.device, dtype=torch.float32), om.to(device=x_f.device, dtype=torch.float32)
            om_d, gate, rwf, rwr = om_d / om_d.mean().clamp_min(1e-6), float(m_r) / (float(m_r) + 32.0), (xf_l[id_d] - mp_d) @ wt_d, (xr_l[id_d] - mp_d) @ wt_d
            dc_m = torch.cdist(u_s.to(device=x_f.device, dtype=torch.float32), u_s.to(device=x_f.device, dtype=torch.float32))
            sig_c = torch.topk(dc_m + torch.eye(n, device=x_f.device) * 1e6, k=min(3, n - 1), dim=1, largest=False).values[:, -1] + 1e-3 * dc_m.amax() + 1e-5
            kc = torch.exp(-dc_m.square() / (2.0 * sig_c.unsqueeze(1) * sig_c.unsqueeze(0)).clamp_min(1e-12)); kc.fill_diagonal_(0.0)
            idg = torch.rsqrt(1.0 + kc.sum(1)); w0 = (kc * idg.unsqueeze(1) * idg.unsqueeze(0)) / float(n); wimq, ell = 0.5 * (w0 + w0.T), math.sqrt(float(dx))
            psi_fn = lambda d: sum((ell * s * d) / (torch.sqrt((ell * s) ** 2 + d) * (ell * s + torch.sqrt((ell * s) ** 2 + d))) for s in (0.35, 1.0, 2.8)) / 3.0
            psi_fr, psi_ff = psi_fn(pairwise_sq_dist(rwf, rwr).clamp_min(0.0)), psi_fn(pairwise_sq_dist(rwf, rwf).clamp_min(0.0))
            imq = (wimq * psi_fr).sum() - 0.5 * (wimq * psi_ff).sum()
            with torch.no_grad():
                psi_rr = psi_fn(pairwise_sq_dist(rwr, rwr).clamp_min(0.0))
                r_imq_sum += float(n) * float(((wimq * (2.0 * psi_fr.detach() - psi_ff.detach() - psi_rr)).sum() / (wimq * psi_rr).sum().clamp_min(1e-8)).clamp(-0.5, 0.5).item())
            lm = gate * (om_d * ((xf_l[id_d] @ wt_d) * dms).sum(1)).mean() / float(dx); rxw = (xf_l[id_d] - mx_d) @ wt_d; rt = rxw * torch.rsqrt(1.0 + 0.25 * rxw.square().sum(1, keepdim=True) / float(dx))
            lc = gate * (om_d * 0.5 * (torch.einsum("ia,iab,ib->i", rxw, ep_t, rxw) + torch.einsum("ia,iab,ib->i", rt, em_t, rt))).mean() / float(dx) if em_t is not None else gate * (om_d * 0.5 * torch.einsum("ia,iab,ib->i", rt, ep_t, rt)).mean() / float(dx)
            tot, tn = tot + float(n) * (0.40 * imq + 0.36 * lm + 0.30 * lc), tn + n
        with torch.no_grad():
            if st is None:
                if len(bank["strata"]) >= 64: continue
                st = {"key": key, "ring_c": torch.empty((0, dc), dtype=torch.float32), "ring_x": torch.empty((0, dx), dtype=torch.float32), "ring_y": torch.empty((0, dx), dtype=torch.float32)}
                bank["strata"].append(st); smap[key] = st
            for rnm, src in (("ring_c", cb), ("ring_x", xb), ("ring_y", yb)): st[rnm] = torch.cat((st[rnm], src.float()), 0)[-384:]
    if tn > 0: rec.observe_cond_fiber((r_imq_sum / float(tn)) + min(0.5, 0.11 * max(0.0, r_op - 0.10)))
    return (tot / float(max(1, tn))).to(x_f.dtype) if tn > 0 else x_f.new_zeros(())

class MinimalLoss(GANLoss):
    def __init__(self, labels: Tuple[float, float, float] = (0.0, 1.0, 1.0), *, loss: str = "relativistic", tau_logit: float = 3.5, minimal_record: Any = None, **_: Any) -> None: super().__init__(loss, labels=labels); self.tau_logit, self.minimal_record = float(tau_logit), minimal_record

    def _rec(self, *ts: Optional[torch.Tensor]) -> Optional[MinimalStepRecord]: return self.minimal_record or next((getattr(t, "_minimal_record", None) for t in ts if t is not None and getattr(t, "_minimal_record", None) is not None), None)
    @staticmethod
    def _same_fiber(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]) -> bool:
        c1, c2 = getattr(t1, "_minimal_cond_tensors", ()), getattr(t2, "_minimal_cond_tensors", ())
        return bool(c1 and c2 and len(c1) == len(c2) and any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in c1) and all(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and bool(torch.equal(a, b)) for a, b in zip(c1, c2)))

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r, f, rec = real_logits.reshape(-1), fake_logits.reshape(-1), self._rec(fake_logits, real_logits)
        if rec is not None and torch.is_grad_enabled() and not rec._in_protected_probe: rec.observe_margin(float((r.detach().float().mean() - f.detach().float().mean()).item()))
        if self._same_fiber(real_logits, fake_logits) and r.shape == f.shape: return _protected_softplus_neg(r - f, self.tau_logit).mean()
        if self.loss == "relativistic":
            xr, xf = getattr(real_logits, "_minimal_input", None), getattr(fake_logits, "_minimal_input", None)
            if isinstance(xr, torch.Tensor) and isinstance(xf, torch.Tensor) and xr.shape[1:] == xf.shape[1:] and 1 <= xr[0].numel() <= 256 and 8 <= xr.shape[0] <= 2048 and 8 <= xf.shape[0] <= 2048 and xr.shape[0] == r.shape[0] and xf.shape[0] == f.shape[0]:
                crf = pairwise_sq_dist(xr.detach().flatten(1).float(), xf.detach().flatten(1).float())
                return (_sinkhorn_log(crf, crf.median().clamp_min(1e-12), 0.15, 5) * _protected_softplus_neg(r.unsqueeze(1) - f.unsqueeze(0), self.tau_logit)).sum()
            return 0.5 * (_protected_softplus_neg(r - f.mean(), self.tau_logit).mean() + _protected_softplus_neg(r.mean() - f, self.tau_logit).mean())
        return super().d_loss(real_logits, fake_logits)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        f, rec = fake_logits.reshape(-1), self._rec(fake_logits, real_logits)
        in_p, c_d, er, s_th, ph = (rec._in_protected_probe, rec.calibration_gate, rec.ema_residual, rec._thermal_scale, rec.chord_phase) if rec is not None else (False, 1.0, 1.0, 1.0, 0.0)
        if real_logits is not None:
            r, xr, xf = real_logits.reshape(-1), getattr(real_logits, "_minimal_input", None), getattr(fake_logits, "_minimal_input", None)
            if xr is None and rec is not None and rec._last_real_pair is not None and r.shape == rec._last_real_pair[1].shape and bool(torch.equal(r.detach(), rec._last_real_pair[1])): xr = rec._last_real_pair[0]
            if self._same_fiber(fake_logits, real_logits) and r.shape == f.shape:
                adv = _protected_softplus_neg(f - r, self.tau_logit).mean()
                return (adv + 1.5 * _cond_fiber_loss(xf, xr, getattr(fake_logits, "_minimal_cond_tensors", ()), rec)) if (not in_p and isinstance(xf, torch.Tensor) and isinstance(xr, torch.Tensor) and xf.requires_grad) else adv
            if self.loss in ("relativistic", "non_saturating"):
                if isinstance(xr, torch.Tensor) and isinstance(xf, torch.Tensor) and xr.shape[1:] == xf.shape[1:] and 1 <= xr[0].numel() <= 256 and 8 <= xr.shape[0] <= 2048 and 8 <= xf.shape[0] <= 2048 and xr.shape[0] == r.shape[0] and xf.shape[0] == f.shape[0]:
                    dim, bf = int(xf[0].numel()), float(xf.shape[0]); crf = pairwise_sq_dist(xr.detach().flatten(1).float(), xf.detach().flatten(1).float())
                    adv = (bf * _sinkhorn_log(crf, crf.median().clamp_min(1e-12), 0.15, 5).t() * _protected_softplus_neg(f.unsqueeze(1) - r.unsqueeze(0), self.tau_logit)).sum()
                    if not in_p and (xf.requires_grad or xr.requires_grad) and s_th > FROZEN_THERMAL_SCALE:
                        th_a = math.sqrt(s_th) if dim == 1 else max(0.15, math.sqrt(s_th))
                        w_ot, w_iso = ((0.45 if dim == 1 else 0.22) + 0.40 * _smoothstep01((er - 0.16) / 0.22) + 1.15 * (1.0 - c_d)) * th_a * bf, (0.35 * (0.70 if dim == 1 else 2.60)) * th_a * bf
                        aux = xf.new_zeros(())
                        for src, tgt in ((xf, xr), (xr, xf.detach())):
                            if src.requires_grad:
                                disp, sig = _born_infeld_disp(src, tgt, er, c_d, ph); tau = ((1.0 if dim == 1 else 0.12) * sig).clamp_min(1e-3)
                                aux = aux + w_ot * (tau * (torch.sqrt(1.0 + (src - (src.detach() + disp).detach()).flatten(1).float().square().sum(1) / (tau * tau)) - 1.0).mean()).to(src.dtype)
                                if 2 <= dim <= 8: aux = aux + w_iso * _burg_cov_loss(src, tgt, er, ph)
                        return c_d * adv + aux
                    return c_d * adv
                return 0.5 * (_protected_softplus_neg(f - r.mean(), self.tau_logit).mean() + _protected_softplus_neg(f.mean() - r, self.tau_logit).mean()) if self.loss == "relativistic" else _protected_softplus_neg(f, self.tau_logit).mean()
        return _protected_softplus_neg(f, self.tau_logit).mean() if self.loss in ("relativistic", "non_saturating") else super().g_loss(fake_logits, real_logits)
