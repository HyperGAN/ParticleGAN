"""Half A1+A2: 1st-Order fwAD/Secant Critic, Debiased Transport, Burg Covariance & Zero-eigh Conditional Fiber."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._stats import pairwise_sq_dist
from ..gan_loss import GANLoss
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .dynamics import (
    FROZEN_THERMAL_SCALE,
    MinimalCriticOptimizer,
    MinimalStepRecord,
    _attach_minimal_critic_hook,
    _cond_signature,
    _smoothstep01,
    _tensor_key,
)

_PHI1 = 0.6180339887498949
_PHI2 = 0.3247179572447460


def _protected_softplus_neg(delta: torch.Tensor, tau: float = 3.5) -> torch.Tensor:
    d32 = delta.float()
    with torch.no_grad():
        s = torch.sqrt(1.0 + d32.detach().square().mean() / (tau * tau))
    return (s * F.softplus(-d32 / s)).to(delta.dtype)


@torch.no_grad()
def _sinkhorn_log(cost: torch.Tensor, scale: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
    nr = int(cost.shape[0])
    nc = int(cost.shape[1])
    la = -math.log(max(1, nr))
    lb = -math.log(max(1, nc))
    lk = -(cost / scale.clamp_min(1e-12)) / max(1e-4, float(eps))
    u = torch.zeros(nr, device=cost.device)
    v = torch.zeros(nc, device=cost.device)
    n_iters = max(int(iters), 7) if eps < 0.03 else int(iters)
    for _ in range(n_iters):
        u = la - torch.logsumexp(lk + v.unsqueeze(0), 1)
        v = lb - torch.logsumexp(lk + u.unsqueeze(1), 0)
    pi = torch.exp(lk + u.unsqueeze(1) + v.unsqueeze(0))
    return pi / (float(nr) * pi.sum(1, keepdim=True).clamp_min(1e-12))


def _eval_lowd_fwad_sq(
    live: Any,
    live_rep: Any,
    x: torch.Tensor,
    phase: float,
    can_stack: bool,
    quantized: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b = int(x.shape[0])
    dim = int(x.shape[1])
    xd = x.detach()
    if dim <= 4:
        basis = torch.eye(dim, device=x.device, dtype=x.dtype)
    else:
        grid = torch.arange(dim, device=x.device, dtype=torch.float32) + 0.5
        v = torch.cos(2.0 * math.pi * (phase + grid * _PHI1))
        vn = v / v.norm().clamp_min(1e-6)
        basis = (torch.eye(dim, device=x.device) - 2.0 * torch.outer(vn, vn)).to(x.dtype)

    if not quantized:
        try:
            with fwAD.dual_level():
                if can_stack:
                    x_rep = xd.unsqueeze(0).expand(dim, b, dim).reshape(dim * b, dim)
                    u_rep = basis.unsqueeze(1).expand(dim, b, dim).reshape(dim * b, dim)
                    po, jo = fwAD.unpack_dual(live_rep(fwAD.make_dual(x_rep, u_rep), dim))
                    if jo is not None:
                        sq = jo.reshape(dim, b, -1).mean(-1).square().sum(0)
                        p0 = po.reshape(dim, b, -1)[0].mean(-1)
                        return sq, p0
                sq = x.new_zeros((b,))
                p0: Optional[torch.Tensor] = None
                for k in range(dim):
                    u_k = basis[k].unsqueeze(0).expand(b, dim)
                    pk, jk = fwAD.unpack_dual(live(fwAD.make_dual(xd, u_k)))
                    if jk is None:
                        raise RuntimeError("fwAD unavailable")
                    if p0 is None:
                        p0 = pk.reshape(b, -1).mean(-1)
                    sq = sq + jk.reshape(b, -1).mean(-1).square()
                assert p0 is not None
                return sq, p0
        except (RuntimeError, NotImplementedError, TypeError):
            pass

    h = 1e-2 if xd.dtype == torch.bfloat16 else 1e-3
    p0 = live(xd).reshape(b, -1).mean(-1)
    sq = x.new_zeros((b,), dtype=torch.float32)
    for k in range(dim):
        tk = basis[k].unsqueeze(0).expand(b, dim)
        xp = (xd.float() + h * tk.float()).to(xd.dtype).detach()
        xm = (xd.float() - h * tk.float()).to(xd.dtype).detach()
        step_len = (xp.float() - xm.float()).norm(dim=1).clamp_min(1e-12)
        yp = live(xp).reshape(b, -1).float().mean(-1)
        ym = live(xm).reshape(b, -1).float().mean(-1)
        sq = sq + ((yp - ym) / step_len).square()
    return sq.to(p0.dtype), p0


def _unit_probes(x: torch.Tensor, phase: float, k_probes: int) -> List[torch.Tensor]:
    b = int(x.shape[0])
    dim = max(1, x[0].numel())
    if x.ndim >= 3 and dim > 1024:
        u = x.new_ones(x.shape, dtype=torch.float32)
        for ax, sz in enumerate(x.shape):
            grid = torch.arange(sz, device=x.device, dtype=torch.float32) + 0.5
            signs = torch.where(
                torch.sin(2.0 * math.pi * (phase + (ax + 1) * _PHI2 + grid * _PHI1)) >= 0.0,
                1.0,
                -1.0,
            )
            view_shape = (1,) * ax + (sz,) + (1,) * (x.ndim - 1 - ax)
            u = u * signs.view(view_shape)
        return [(u / math.sqrt(float(dim))).to(x.dtype)]

    probes: List[torch.Tensor] = []
    idx = torch.arange(b * dim, device=x.device, dtype=torch.float32)
    for k in range(k_probes):
        raw = torch.sin((idx + 1.0) * (12.9898 + 78.233 * (k + 1)) + 43758.5453 * (phase + (k + 1) * _PHI1))
        u = torch.where(raw >= 0.0, 1.0, -1.0).view(b, dim)
        for prev in probes:
            pf = prev.view(b, dim).float()
            u = u - (u * pf).sum(1, keepdim=True) * pf
        unit = (u / u.norm(dim=1, keepdim=True).clamp_min(1e-6)).view_as(x).to(x.dtype)
        probes.append(unit)
    return probes


class MinimalCriticPenalty(_K3PCriticPenalty):
    """1st-order forward-AD / secant Born-Infeld critic penalty (0 autograd.grad, BN-safe)."""

    def __init__(
        self,
        recipe: Any,
        optimizer: Any,
        *,
        output: Any = None,
        collect_stats: bool = False,
        **overrides: Any,
    ) -> None:
        from ..discriminators import BatchDistanceDiscriminator
        from ..tensorflow_adam import TensorFlowV1Adam

        is_valid_opt = isinstance(optimizer, (MinimalCriticOptimizer, K3PCriticAdam)) or (
            type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam) and hasattr(optimizer, "record")
        )
        if not is_valid_opt:
            raise TypeError("optimizer must come from recipe.make_critic_optimizer")

        self.recipe = recipe
        self.optimizer = optimizer
        self.critic = optimizer.critic
        self.record = optimizer.record
        _attach_minimal_critic_hook(self.critic, self.record)

        opts = recipe._penalty_options(**overrides)
        self.arm = "minimal"
        self.coeff = float(opts.get("coeff", 1.0))
        self.kappa = float(opts.get("kappa", 1.0))
        self.lazy_k = int(opts.get("lazy_k", 1))
        self.ebi_beta = float(opts.get("ebi_beta", 8.0))
        self.secant_eps = float(opts.get("secant_eps", 1e-5))

        self.regularizer = self
        self.initial_coeff = self.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._verified: set[int] = set()
        self._names = {id(m): n for n, m in self.critic.named_modules()}
        self._has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules())
        self._has_bd = any(isinstance(m, BatchDistanceDiscriminator) for m in self.critic.modules())
        self._has_q = any(
            getattr(m, "format", None) in ("nvfp4", "mxfp8") or "Emulated" in type(m).__name__
            for m in self.critic.modules()
        )

    @torch.no_grad()
    def _batch_res(self, xr: torch.Tensor, xf: torch.Tensor) -> Tuple[float, float]:
        dim = max(1, xr[0].numel())
        if xr.ndim == 4 and dim > 1024:
            rf = F.adaptive_avg_pool2d(xr.detach().float(), (4, 4)).flatten(1)
            ff = F.adaptive_avg_pool2d(xf.detach().float(), (4, 4)).flatten(1)
            ref = rf.var(0, unbiased=False).sum().clamp_min(1e-4)
            pooled_dim = float(max(1, rf.shape[1]))
            rel_res = float(((rf - ff).square().sum(1).mean() / ref).sqrt().item())
            data_scale = float((ref * (float(dim) / pooled_dim)).sqrt().item())
            return rel_res, data_scale

        rf = xr.detach().flatten(1).float()
        ff = xf.detach().flatten(1).float()
        if len(rf) > 256:
            rf = rf[(torch.arange(256, device=rf.device) * len(rf)) // 256]
        if len(ff) > 256:
            ff = ff[(torch.arange(256, device=ff.device) * len(ff)) // 256]
        if len(rf) >= 4:
            rr = torch.maximum(pairwise_sq_dist(rf).median(), rf.var(0, unbiased=False).sum())
        else:
            rr = rf.new_zeros(())
        ref = torch.where(rr > 1e-6, rr.clamp_min(1e-12), rf.new_full((), 0.04))
        crf = pairwise_sq_dist(rf, ff) / ref
        chamfer = 0.5 * (crf.min(1).values.mean() + crf.min(0).values.mean())
        return float(torch.sqrt(chamfer.clamp_min(0.0)).item()), float(ref.sqrt().item())

    def __call__(
        self,
        critic: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *cond: Any,
        **ckw: Any,
    ) -> torch.Tensor:
        from ..discriminators import BatchDistanceDiscriminator
        from ..recipe_schedules import apply_penalty_schedule

        if id(critic) not in self._verified:
            self._ema_view(critic)
            self._verified.add(id(critic))
        if self.recipe is not None:
            apply_penalty_schedule(self.record.observed_steps, self.recipe, self)
        if self.coeff == 0.0 or (self.lazy_k > 1 and (self.record.observed_steps + 1) % self.lazy_k != 0):
            self.last_stats = {"applied": False, "pen": 0.0, "center": self.kappa} if self.collect_stats else {}
            return x_real.new_zeros(())

        rec = self.record
        out_fn = self.output
        b0 = int(x_real.shape[0])
        dim = max(1, x_real[0].numel())
        rec.calls += 1
        rec.chord_phase = (rec.chord_phase + _PHI1) % 1.0

        has_bn = self._has_bn or (
            critic is not self.critic
            and any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in critic.modules())
        )
        has_bd = self._has_bd or (
            critic is not self.critic
            and any(isinstance(m, BatchDistanceDiscriminator) for m in critic.modules())
        )
        has_bs = any(isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == b0 for v in vars(critic).values())
        is_cond = bool(has_bs or cond or ckw)
        cond_sig = _cond_signature((x_real, *cond), ckw)
        allow_reuse = (out_fn is _first_output) and (not has_bs) and (not has_bn)
        can_cat = (
            (not has_bn)
            and (not has_bd)
            and (not has_bs)
            and float(getattr(critic, "std", 0.0)) == 0.0
            and (out_fn is _first_output)
        )

        res, d_scale = self._batch_res(x_real, x_fake)
        rec.observe_transport(res, d_scale, is_conditioned=is_cond)

        def live(x: torch.Tensor) -> torch.Tensor:
            return out_fn(critic(x, *cond, **ckw))

        def _repeat_cond(t: Any, m: int) -> Any:
            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == b0:
                return t.repeat((m,) + (1,) * (t.ndim - 1))
            return t

        def live_rep(x: torch.Tensor, m: int) -> torch.Tensor:
            if m <= 1 or (not cond and not ckw) or x.shape[0] != m * b0:
                return live(x)
            rep_args = tuple(_repeat_cond(t, m) for t in cond)
            rep_kwargs = {k: _repeat_cond(v, m) for k, v in ckw.items()}
            return out_fn(critic(x, *rep_args, **rep_kwargs))

        # Preserve BN training/eval flags and evaluate penalty probes with BN in eval() so
        # real/fake secant probes never mix batch statistics or corrupt BN running stats:
        bn_modes: List[Tuple[nn.Module, bool]] = []
        if has_bn and isinstance(critic, nn.Module):
            for mod in critic.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    bn_modes.append((mod, mod.training))
                    mod.eval()

        rec._in_penalty = True
        try:
            if x_real.ndim == 2 and dim <= 8:
                gsq_r, d_r = _eval_lowd_fwad_sq(live, live_rep, x_real, rec.chord_phase, can_cat, self._has_q)
                gsq_f, d_f = _eval_lowd_fwad_sq(
                    live, live_rep, x_fake, (rec.chord_phase + _PHI2) % 1.0, can_cat, self._has_q
                )
            else:
                k_p = 2 if dim <= 1024 else 1
                cur = int(rec.observed_steps)

                def _find_capture(x_t: torch.Tensor, exc: Optional[torch.Tensor] = None) -> Any:
                    tkey = _tensor_key(x_t)
                    for st, sk, sg, xa, out, al in reversed(rec._live_captures):
                        if (
                            al
                            and st == cur
                            and sk == tkey
                            and sg == cond_sig
                            and out is not exc
                            and out.requires_grad
                            and out.grad_fn is not None
                        ):
                            return xa, out
                    return None

                gr = _find_capture(x_real) if allow_reuse else None
                gf = _find_capture(x_fake, None if gr is None else gr[1]) if allow_reuse else None
                xr_a = x_real.detach()
                xf_a = x_fake.detach()
                dr_raw = gr[1] if gr is not None else live(xr_a)
                df_raw = gf[1] if gf is not None else live(xf_a)
                d_r = dr_raw.reshape(b0, -1).mean(-1)
                d_f = df_raw.reshape(x_fake.shape[0], -1).mean(-1)
                if x_real.dtype == torch.bfloat16:
                    eps_s = max(1e-2 * max(1.0, d_scale), self.secant_eps * max(1.0, d_scale))
                else:
                    eps_s = self.secant_eps * max(1.0, d_scale / math.sqrt(float(dim)))

                if can_cat and dim > 1024 and x_real.shape == x_fake.shape:
                    m_p = ((torch.arange(b0, device=x_real.device) + rec.calls) & 1) == 0
                    z_b = torch.where(m_p.view((b0,) + (1,) * (x_real.ndim - 1)), xr_a, xf_a)
                    u_b = _unit_probes(z_b, rec.chord_phase, 1)[0]
                    z_p = (z_b.float() + eps_s * u_b.float()).to(z_b.dtype).detach()
                    step_b = (z_p.float() - z_b.float()).flatten(1).norm(dim=1).clamp_min(1e-12)
                    d_p = live(z_p).reshape(b0, -1).float().mean(-1)
                    d_base = torch.where(m_p, d_r.float(), d_f.float())
                    gsq_r = gsq_f = ((d_p - d_base) / step_b).square() * float(dim)
                else:
                    gsq_r = x_real.new_zeros((b0,), dtype=torch.float32)
                    gsq_f = x_fake.new_zeros((x_fake.shape[0],), dtype=torch.float32)
                    sk = float(dim) / float(k_p)
                    probes_r = _unit_probes(xr_a, rec.chord_phase, k_p)
                    probes_f = _unit_probes(xf_a, (rec.chord_phase + _PHI2) % 1.0, k_p)
                    for ur, uf in zip(probes_r, probes_f):
                        xp_r = (xr_a.float() + eps_s * ur.float()).to(xr_a.dtype).detach()
                        step_r = (xp_r.float() - xr_a.float()).flatten(1).norm(dim=1).clamp_min(1e-12)
                        dp_r = live(xp_r).reshape(b0, -1).float().mean(-1)
                        gsq_r = gsq_r + ((dp_r - d_r.float()) / step_r).square() * sk

                        xp_f = (xf_a.float() + eps_s * uf.float()).to(xf_a.dtype).detach()
                        step_f = (xp_f.float() - xf_a.float()).flatten(1).norm(dim=1).clamp_min(1e-12)
                        dp_f = live(xp_f).reshape(x_fake.shape[0], -1).float().mean(-1)
                        gsq_f = gsq_f + ((dp_f - d_f.float()) / step_f).square() * sk
        finally:
            rec._in_penalty = False
            for mod, flag in bn_modes:
                mod.training = flag

        lam = min(1.0, max(0.01, self.coeff * (0.25 + 0.75 * rec._step_scale)))
        bi_r = ((torch.sqrt(1.0 + self.ebi_beta * gsq_r.float()) - 1.0) / self.ebi_beta).mean()
        bi_f = ((torch.sqrt(1.0 + self.ebi_beta * gsq_f.float()) - 1.0) / self.ebi_beta).mean()
        bi = 0.5 * (bi_r + bi_f)
        nc = min(b0, int(d_f.shape[0]))
        chord_dist = (
            x_real.detach()[:nc].flatten(1).float() - x_fake.detach()[:nc].flatten(1).float()
        ).norm(dim=1).clamp_min(1e-3 * max(1.0, d_scale))
        chord = ((d_r[:nc].float() - d_f[:nc].float()) / chord_dist).square().mean()
        gauge = 0.03 * (d_r.float().mean().square() + d_f.float().mean().square())
        pen = (lam * (bi + 0.08 * chord) + gauge).to(x_real.dtype)
        self.last_stats = {"applied": True, "pen": float(pen.detach().item()), "center": self.kappa} if self.collect_stats else {}
        return pen


@torch.no_grad()
def _born_infeld_disp(
    x_f: torch.Tensor,
    x_r: torch.Tensor,
    ema_res: float,
    c_d: float,
    phase: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ff = x_f.detach().flatten(1).float()
    rf = x_r.detach().flatten(1).float()
    bf = int(ff.shape[0])
    br = int(rf.shape[0])
    dim = int(ff.shape[1])

    if dim == 1 and br >= 8:
        rs = torch.sort(rf[:, 0]).values
        fo = torch.argsort(ff[:, 0])
        pos = torch.linspace(0.0, float(br - 1), bf, device=rf.device)
        lo = pos.floor().long().clamp(0, br - 1)
        frac = pos - lo.float()
        disp = torch.empty_like(ff[:, 0])
        disp[fo] = ((1.0 - frac) * rs[lo] + frac * rs[(lo + 1).clamp(0, br - 1)]) - ff[fo, 0]
        rstd = rf[:, 0].std(unbiased=(br > 1)) if br > 1 else rf.new_ones(())
        return disp.view_as(x_f), torch.where(rstd > 1e-4, rstd, rf.new_ones(())).clamp_min(1e-3)

    if br > 512:
        rf = rf[(torch.arange(512, device=rf.device) * br) // 512]
        br = 512
    c_rf = pairwise_sq_dist(rf, ff)
    if br >= 4:
        rr = torch.maximum(pairwise_sq_dist(rf).median(), rf.var(0, unbiased=False).sum())
    else:
        rr = rf.new_zeros(())
    ref = torch.where(rr > 1e-6, rr.clamp_min(1e-12), rf.new_ones(()))
    c_med = c_rf.median().clamp_min(1e-12)
    c_sc = torch.maximum(torch.where(rr > 1e-6, ref, c_med), c_med)
    eps_e = max(0.025, min(0.12, 0.12 * (0.25 + 0.75 * min(1.0, ema_res / 0.25))))
    pi_rf = _sinkhorn_log(c_rf, c_sc, eps_e, 6)
    wq = max(_smoothstep01((ema_res - 0.14) / 0.12), _smoothstep01((0.85 - c_d) / 0.45))

    vq: Optional[torch.Tensor] = None
    if wq > 1e-3:
        cp = (bf * pi_rf.t()).clamp_min(1e-12)
        cdf = torch.cumsum(cp / cp.sum(1, keepdim=True), 1)
        u_qmc = torch.remainder(
            (torch.arange(bf, device=ff.device, dtype=torch.float32) + 0.5) / bf + phase + 0.5 * _PHI1,
            1.0,
        ).unsqueeze(1)
        j_idx = torch.searchsorted(cdf, u_qmc).squeeze(1).clamp(0, br - 1)
        vq = rf[j_idx] - ff

    vf: Optional[torch.Tensor] = None
    if wq < 0.999:
        bary_rf = (pi_rf / pi_rf.sum(0, keepdim=True).clamp_min(1e-12)).t() @ rf
        fs = ff if bf <= 512 else ff[(torch.arange(512, device=ff.device) * bf) // 512]
        pff = _sinkhorn_log(pairwise_sq_dist(fs, ff), c_sc, eps_e, 6)
        bary_ff = (pff / pff.sum(0, keepdim=True).clamp_min(1e-12)).t() @ fs
        vf = bary_rf - bary_ff

    if vf is None:
        assert vq is not None
        out = vq
    elif vq is None:
        out = vf
    else:
        out = wq * vq + (1.0 - wq) * vf
    return out.view_as(x_f), ref.sqrt().clamp_min(1e-3)


def _safe_cholesky(mat: torch.Tensor, base_jitter: float = 1e-4) -> torch.Tensor:
    """SPD Cholesky factor checking ``cholesky_ex`` ``info`` with adaptive jitter escalation."""
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    chol, info = torch.linalg.cholesky_ex(sym, check_errors=False)
    if bool((info == 0).all()):
        return chol
    d = int(sym.shape[-1])
    eye = torch.eye(d, dtype=sym.dtype, device=sym.device)
    if sym.ndim == 3:
        eye = eye.unsqueeze(0)
        scale = sym.diagonal(dim1=-2, dim2=-1).clamp_min(0.0).mean(dim=-1).clamp_min(1e-6).view(-1, 1, 1)
    else:
        scale = sym.diagonal().clamp_min(0.0).mean().clamp_min(1e-6)
    for mult in (base_jitter, 1e-3, 1e-2, 1e-1, 1.0):
        chol, info = torch.linalg.cholesky_ex(sym + (mult * scale) * eye, check_errors=False)
        if bool((info == 0).all()):
            return chol
    diag_safe = sym.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
    return torch.diag_embed(diag_safe.sqrt())


def _burg_cov_loss(x_f: torch.Tensor, x_r: torch.Tensor, ema_res: float, phase: float) -> torch.Tensor:
    ff = x_f.flatten(1).float()
    rf = x_r.detach().flatten(1).float()
    bf = int(ff.shape[0])
    dim = int(ff.shape[1])
    if not x_f.requires_grad or bf < 16 or len(rf) < 16 or not (1 <= dim <= 8):
        return x_f.new_zeros(())

    if dim == 1:
        rm = rf.mean()
        rv = rf.var(unbiased=True).clamp_min(1e-6)
        vr = 1.03 * rv
        vf = ff.var(unbiased=True) + 0.03 * rv
        burg = torch.sqrt(1.0 + (vr / vf - torch.log(vr / vf) - 1.0).clamp_min(0.0)) - 1.0
        mean_term = 0.25 * (torch.sqrt(1.0 + (ff.mean() - rm).square() / rv) - 1.0)
        return (_smoothstep01((0.80 - ema_res) / 0.45) * rv.sqrt() * (burg + mean_term)).to(x_f.dtype)

    rg = _smoothstep01((0.40 - ema_res) / 0.25)
    if rg <= 1e-6:
        return x_f.new_zeros(())

    with torch.no_grad():
        if len(rf) <= 512:
            r_s = rf
        else:
            r_idx = (torch.arange(512, device=rf.device) * len(rf) // 512 + int(phase * len(rf))) % len(rf)
            r_s = rf[r_idx]
        if bf <= 512:
            f_idx: Optional[torch.Tensor] = None
            f_s = ff.detach()
        else:
            f_idx = (torch.arange(512, device=ff.device) * bf // 512 + int((phase + _PHI1) * bf)) % bf
            f_s = ff.detach()[f_idx]
        nr = len(r_s)
        crr = pairwise_sq_dist(r_s)
        k_loc = max(5, min(14, nr // 16))
        h2 = torch.topk(crr, k=k_loc + 1, dim=1, largest=False).values[:, -1].clamp_min(1e-6)

        crr_c = crr.cpu()
        h2_c = h2.cpu()
        unc = torch.ones(nr, dtype=torch.bool)
        md2 = torch.full((nr,), float("inf"))
        sel: List[int] = []
        cur = int(phase * nr) % nr
        for _ in range(min(64, nr)):
            sel.append(cur)
            md2 = torch.minimum(md2, crr_c[cur])
            unc = unc & (crr_c[cur] > 4.5 * torch.maximum(h2_c[cur], h2_c))
            if not bool(unc.any()):
                break
            cur = int(torch.where(unc, md2, -1.0).argmax().item())

        st_i = torch.tensor(sel, device=rf.device, dtype=torch.long)
        d2_ar = crr[st_i]
        d2_af = pairwise_sq_dist(r_s[st_i], f_s)
        mg = torch.arange(len(sel), device=rf.device).unsqueeze(1)
        mr_f = d2_ar.argmin(0).unsqueeze(0) == mg
        mf_f = d2_af.argmin(0).unsqueeze(0) == mg
        rad2 = (3.0 * h2[st_i]).unsqueeze(1)

    fl = ff if f_idx is None else ff[f_idx]
    tot = ff.new_zeros(())
    for mr, mf in ((mr_f, mf_f), (mr_f & (d2_ar <= rad2), mf_f & (d2_af <= rad2))):
        with torch.no_grad():
            cr = mr.sum(1).float()
            cf = mf.sum(1).float()
            ok = (cr >= 4.0) & (cf >= 4.0)
            if not bool(ok.any()):
                continue
            wr = mr.float() / cr.clamp_min(1.0).unsqueeze(1)
            wf = mf.float() / cf.clamp_min(1.0).unsqueeze(1)
            mu_r = wr @ r_s
            dr = r_s.unsqueeze(0) - mu_r.unsqueeze(1)
            cov_r = (cr / (cr - 1.0).clamp_min(1.0)).view(-1, 1, 1) * torch.einsum("mb,mbd,mbe->mde", wr, dr, dr)
            vs = (cov_r.diagonal(dim1=-2, dim2=-1).sum(-1) / dim).clamp_min(1e-6)
            ok = ok & (vs > 1e-5)
            if not bool(ok.any()):
                continue
            eye_r = (0.04 * vs).view(-1, 1, 1) * torch.eye(dim, device=rf.device).unsqueeze(0)

        mu_f = wf @ fl
        df = fl.unsqueeze(0) - mu_f.unsqueeze(1)
        sr = cov_r + eye_r
        sf = (cf / (cf - 1.0).clamp_min(1.0)).view(-1, 1, 1) * torch.einsum("mb,mbd,mbe->mde", wf, df, df) + eye_r
        if dim == 2:
            dr2 = (sr[:, 0, 0] * sr[:, 1, 1] - sr[:, 0, 1].square()).clamp_min(1e-18)
            df2 = (sf[:, 0, 0] * sf[:, 1, 1] - sf[:, 0, 1].square()).clamp_min(1e-18)
            lr = torch.log(df2) - torch.log(dr2)
            tr_rf = (sf[:, 1, 1] * sr[:, 0, 0] + sf[:, 0, 0] * sr[:, 1, 1] - 2.0 * sf[:, 0, 1] * sr[:, 0, 1]) / df2
            tr_fr = (sr[:, 1, 1] * sf[:, 0, 0] + sr[:, 0, 0] * sf[:, 1, 1] - 2.0 * sr[:, 0, 1] * sf[:, 0, 1]) / dr2
            db = (tr_rf + lr - 2.0).clamp_min(0.0) + 0.5 * (tr_fr - lr - 2.0).clamp_min(0.0)
        else:
            l_r = _safe_cholesky(sr)
            l_f = _safe_cholesky(sf)
            lr = 2.0 * (
                torch.log(l_f.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)).sum(-1)
                - torch.log(l_r.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)).sum(-1)
            )
            tr_rf = torch.cholesky_solve(sr, l_f).diagonal(dim1=-2, dim2=-1).sum(-1)
            tr_fr = torch.cholesky_solve(sf, l_r).diagonal(dim1=-2, dim2=-1).sum(-1)
            db = (tr_rf + lr - float(dim)).clamp_min(0.0) + 0.5 * (tr_fr - lr - float(dim)).clamp_min(0.0)

        mean_gap = (mu_f - mu_r).square().sum(-1) / (dim * vs)
        ef = vs.sqrt() * ((torch.sqrt(1.0 + db) - 1.0) + 0.25 * (torch.sqrt(1.0 + mean_gap) - 1.0)) * ok.float()
        tot = tot + 0.5 * (0.65 * (ef.sum() / ok.float().sum().clamp_min(1.0)) + 0.35 * ef.amax())
    return (rg * tot).to(x_f.dtype)


def _sym2_ops(m: torch.Tensor, split: bool = False) -> Any:
    if not split:
        a = m[0, 0]
        b = 0.5 * (m[0, 1] + m[1, 0])
        d = m[1, 1]
        tr = (a + d).clamp_min(1e-6)
        disc = torch.sqrt(((a - d) * 0.5).square() + b * b)
        fl = max(1e-4 * float(tr.item()) * 0.5, 1e-6)
        l1 = (0.5 * tr + disc).clamp_min(fl)
        l2 = (0.5 * tr - disc).clamp_min(fl)
        s1 = l1.rsqrt()
        s2 = l2.rsqrt()
        c1 = torch.where((l1 - l2) > 1e-10, (s1 - s2) / (l1 - l2).clamp_min(1e-12), -0.5 * l1.pow(-1.5))
        sym_m = torch.stack((torch.stack((a, b)), torch.stack((b, d))))
        return (s1 - c1 * l1) * torch.eye(2, dtype=m.dtype, device=m.device) + c1 * sym_m

    a = m[:, 0, 0]
    b = 0.5 * (m[:, 0, 1] + m[:, 1, 0])
    d = m[:, 1, 1]
    tr = 0.5 * (a + d)
    diff = 0.5 * (a - d)
    disc = torch.sqrt(diff.square() + b * b + 1e-24)
    l1 = tr + disc
    l2 = tr - disc
    u1 = torch.where(diff >= 0, diff + disc, b)
    v1 = torch.where(diff >= 0, b, disc - diff)
    n1 = torch.sqrt(u1 * u1 + v1 * v1).clamp_min(1e-12)
    u1 = u1 / n1
    v1 = v1 / n1

    def _build_mat(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        row0 = torch.stack((p1 * u1 * u1 + p2 * v1 * v1, (p1 - p2) * u1 * v1), dim=-1)
        row1 = torch.stack(((p1 - p2) * u1 * v1, p1 * v1 * v1 + p2 * u1 * u1), dim=-1)
        return torch.stack((row0, row1), dim=-2)

    return _build_mat(l1.clamp_min(0.0), l2.clamp_min(0.0)), _build_mat(l1.clamp_max(0.0), l2.clamp_max(0.0))


def _whitening_matrix(cp: torch.Tensor) -> torch.Tensor:
    """Compute affine-invariant right-whitening factor ``W`` with ``W^T cp W = I`` (0 eigh/svd/pinv).

    For ``dx == 2``, uses closed-form symmetric ``cp^{-1/2}``.
    For ``dx >= 3``, uses ``L^{-T}`` via ``solve_triangular(L, I, upper=False).T`` where ``L L^T = cp``
    (checking ``cholesky_ex`` ``info`` with jitter escalation), ensuring exact Mahalanobis units
    ``||v @ W||_2^2 = v cp^{-1} v^T`` and affine invariance for all ``dx >= 2``.
    """
    dx = int(cp.shape[0])
    if dx == 2:
        return _sym2_ops(cp)
    if dx == 1:
        return cp[0, 0].clamp_min(1e-6).rsqrt().reshape(1, 1)

    sym = 0.5 * (cp + cp.T)
    base_scale = max(float(sym.diagonal().clamp_min(0.0).mean().item()), 1e-12)
    eye = torch.eye(dx, dtype=torch.float64, device=cp.device)
    chol: Optional[torch.Tensor] = None
    cand0, info0 = torch.linalg.cholesky_ex(sym, check_errors=False)
    if int(info0.item()) == 0:
        d0 = cand0.diagonal()
        if float(d0.min().item()) > 1e-6 * max(float(d0.max().item()), 1e-12):
            chol = cand0
    if chol is None:
        for mult in (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            cand, info = torch.linalg.cholesky_ex(sym + (mult * base_scale) * eye, check_errors=False)
            if int(info.item()) == 0:
                chol = cand
                break
    if chol is None:
        diag_safe = sym.diagonal().clamp_min(1e-6 * base_scale)
        return torch.diag(diag_safe.rsqrt())
    l_inv = torch.linalg.solve_triangular(chol, eye, upper=False)
    return l_inv.T


def _cond_fiber_loss(
    x_f: torch.Tensor,
    x_r: torch.Tensor,
    cond_t: Tuple[torch.Tensor, ...],
    rec: Optional[MinimalStepRecord],
) -> torch.Tensor:
    if not x_f.requires_grad or rec is None:
        return x_f.new_zeros(())
    b = int(x_f.shape[0])
    dx = int(x_f.flatten(1).shape[1])
    fl_c = [
        t.detach().reshape(b, -1).double()
        for t in cond_t
        if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == b and t.is_floating_point()
    ]
    disc_c = [
        t.detach().reshape(b, -1).long()
        for t in cond_t
        if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == b and not t.is_floating_point()
    ]
    if not fl_c or dx < 1:
        return x_f.new_zeros(())
    c_cpu = torch.cat(fl_c, 1).cpu()
    dc = int(c_cpu.shape[1])
    if dc > 32:
        return x_f.new_zeros(())

    xf_c = x_f.detach().flatten(1).double().cpu()
    xr_c = x_r.detach().flatten(1).double().cpu()
    groups: dict[Tuple[int, ...], List[int]] = {}
    disc_rows = torch.cat(disc_c, 1).cpu().tolist() if disc_c else [(0,)] * b
    for i, row in enumerate(disc_rows):
        groups.setdefault(tuple(int(v) for v in row), []).append(i)

    bank = rec.cond_bank
    if bank["d_c"] != dc or bank["d_x"] != dx:
        bank["d_c"] = dc
        bank["d_x"] = dx
        bank["strata"] = []
    smap = {entry["key"]: entry for entry in bank["strata"]}
    pd = (1 + 3 * dc + (dc * (dc - 1)) // 2) if dc <= 6 else (2 + dc)
    ridge = torch.full((pd,), 0.025, dtype=torch.float64)
    ridge[: 1 + dc] = 1e-3

    def _phi(c64: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = (c64 - mu.unsqueeze(0)) / sd.unsqueeze(0)
        om = 1.0 + 0.90 * u.square().mean(1).clamp_max(4.5)
        ones = torch.ones((c64.shape[0], 1), dtype=torch.float64)
        if dc <= 6:
            pts = [ones, u, 2.0 * (F.softplus(u - 0.8) - 0.425), 2.0 * (F.softplus(-u - 0.8) - 0.425)]
            if dc >= 2:
                ia, ib = torch.triu_indices(dc, dc, offset=1)
                cross = (u[:, ia] * u[:, ib]) * torch.rsqrt(1.0 + 0.15 * (u[:, ia].square() + u[:, ib].square()))
                pts.append(cross)
            return torch.cat(pts, 1), om, u
        radial = 2.0 * (F.softplus(u.square().mean(1).clamp_min(1e-6).sqrt() - 1.0).unsqueeze(1) - 0.3133)
        return torch.cat((ones, u, radial), 1), om, u

    xf_l = x_f.flatten(1).float()
    xr_l = x_r.detach().flatten(1).float()
    tot = x_f.new_zeros(())
    r_op = 0.0
    r_imq_sum = 0.0
    tn = 0

    for key, idxs in groups.items():
        if len(idxs) < 4:
            continue
        it = torch.tensor(idxs, dtype=torch.long)
        n = len(idxs)
        st = smap.get(key)
        cb = c_cpu[it]
        xb = xf_c[it]
        yb = xr_c[it]

        if st is not None and st["ring_c"].shape[0] >= max(24, int(2.5 * pd)):
            rc = st["ring_c"].double()
            rx = st["ring_x"].double()
            ry = st["ring_y"].double()
            m_r = int(rc.shape[0])

            age = torch.arange(m_r - 1, -1, -1, dtype=torch.float64)
            rw = torch.exp(-age / 192.0)
            rwn = rw / rw.sum()
            mu_c = (rwn.unsqueeze(1) * rc).sum(0)
            sd_c = (rwn.unsqueeze(1) * (rc - mu_c).square()).sum(0).clamp_min(1e-4).sqrt()

            pr, omr, ur = _phi(rc, mu_c, sd_c)
            phi, om, u_s = _phi(cb, mu_c, sd_c)
            wr = rw * omr
            w_eff = float(wr.sum().item())
            wp = pr * wr.unsqueeze(1)

            gram_p = 0.5 * (wp.T @ pr + (wp.T @ pr).T) / w_eff + torch.diag(ridge)
            ls = _safe_cholesky(gram_p)
            bxs = torch.cholesky_solve((wp.T @ rx) / w_eff, ls)
            bys = torch.cholesky_solve((wp.T @ ry) / w_eff, ls)
            bp = 0.5 * (bxs + bys)
            mp = phi @ bp
            mx = phi @ bxs
            mpr = pr @ bp

            cp = (
                0.5
                * (((rx - mpr) * wr.unsqueeze(1)).T @ (rx - mpr) + ((ry - mpr) * wr.unsqueeze(1)).T @ (ry - mpr))
                / w_eff
            )
            wt = _whitening_matrix(cp)

            r_op = max(r_op, max(0.0, float((((bxs - bys) @ wt).square().sum() / dx).item()) - 2.0 * pd / max(float(m_r), 1.0)))
            dmw = (phi @ (bxs - bys)) @ wt
            dmn = dmw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            dms = (3.0 * torch.tanh(dmn / 3.0) * (dmw / dmn)).to(device=x_f.device, dtype=torch.float32)

            d2 = torch.cdist(u_s, ur).square()
            h2 = torch.topk(d2, k=min(20, m_r), dim=1, largest=False).values[:, -1].clamp_min(0.15)
            al = torch.softmax(-d2 / (2.0 * h2.unsqueeze(1)) - age.unsqueeze(0) / 128.0, dim=1)
            cov_diff = (
                torch.einsum("ij,ja,jb->iab", al, rx - mpr, rx - mpr)
                - torch.einsum("ij,ja,jb->iab", al, ry - mpr, ry - mpr)
            )
            # W^T (cov_diff) W matches right-whitened vectors (x - mu) @ W for both symmetric and triangular W:
            ew = torch.einsum("ca,icd,db->iab", wt, cov_diff, wt)
            fn = ew.flatten(1).norm(dim=1).view(n, 1, 1).clamp_min(1e-12)
            ew = 1.5 * torch.tanh(fn / 1.5) * (ew / fn)

            if dx == 2:
                ep_raw, em_raw = _sym2_ops(ew, split=True)
                ep_t = ep_raw.to(device=x_f.device, dtype=torch.float32)
                em_t: Optional[torch.Tensor] = em_raw.to(device=x_f.device, dtype=torch.float32)
            else:
                ep_t = ew.to(device=x_f.device, dtype=torch.float32)
                em_t = None

            id_d = it.to(x_f.device)
            wt_d = wt.to(device=x_f.device, dtype=torch.float32)
            mp_d = mp.to(device=x_f.device, dtype=torch.float32)
            mx_d = mx.to(device=x_f.device, dtype=torch.float32)
            om_d = om.to(device=x_f.device, dtype=torch.float32)
            om_d = om_d / om_d.mean().clamp_min(1e-6)
            gate = float(m_r) / (float(m_r) + 32.0)

            rwf = (xf_l[id_d] - mp_d) @ wt_d
            rwr = (xr_l[id_d] - mp_d) @ wt_d

            u_dev = u_s.to(device=x_f.device, dtype=torch.float32)
            dc_m = torch.cdist(u_dev, u_dev)
            sig_c = (
                torch.topk(dc_m + torch.eye(n, device=x_f.device) * 1e6, k=min(3, n - 1), dim=1, largest=False).values[:, -1]
                + 1e-3 * dc_m.amax()
                + 1e-5
            )
            kc = torch.exp(-dc_m.square() / (2.0 * sig_c.unsqueeze(1) * sig_c.unsqueeze(0)).clamp_min(1e-12))
            kc.fill_diagonal_(0.0)
            idg = torch.rsqrt(1.0 + kc.sum(1))
            w0 = (kc * idg.unsqueeze(1) * idg.unsqueeze(0)) / float(n)
            wimq = 0.5 * (w0 + w0.T)
            ell = math.sqrt(float(dx))

            def _psi_fn(dist_sq: torch.Tensor) -> torch.Tensor:
                acc = dist_sq.new_zeros(dist_sq.shape)
                for s_mult in (0.35, 1.0, 2.8):
                    ls_val = ell * s_mult
                    rad = torch.sqrt(ls_val * ls_val + dist_sq)
                    acc = acc + (ls_val * dist_sq) / (rad * (ls_val + rad))
                return acc / 3.0

            psi_fr = _psi_fn(pairwise_sq_dist(rwf, rwr).clamp_min(0.0))
            psi_ff = _psi_fn(pairwise_sq_dist(rwf, rwf).clamp_min(0.0))
            imq = (wimq * psi_fr).sum() - 0.5 * (wimq * psi_ff).sum()
            with torch.no_grad():
                psi_rr = _psi_fn(pairwise_sq_dist(rwr, rwr).clamp_min(0.0))
                imq_ratio = (wimq * (2.0 * psi_fr.detach() - psi_ff.detach() - psi_rr)).sum() / (
                    (wimq * psi_rr).sum().clamp_min(1e-8)
                )
                r_imq_sum += float(n) * float(imq_ratio.clamp(-0.5, 0.5).item())

            lm = gate * (om_d * (rwf * dms).sum(1)).mean() / float(dx)
            rxw = (xf_l[id_d] - mx_d) @ wt_d
            rt = rxw * torch.rsqrt(1.0 + 0.25 * rxw.square().sum(1, keepdim=True) / float(dx))
            if em_t is not None:
                quad = torch.einsum("ia,iab,ib->i", rxw, ep_t, rxw) + torch.einsum("ia,iab,ib->i", rt, em_t, rt)
            else:
                quad = torch.einsum("ia,iab,ib->i", rt, ep_t, rt)
            lc = gate * (om_d * 0.5 * quad).mean() / float(dx)
            tot = tot + float(n) * (0.40 * imq + 0.36 * lm + 0.30 * lc)
            tn += n

        with torch.no_grad():
            if st is None:
                if len(bank["strata"]) >= 64:
                    continue
                st = {
                    "key": key,
                    "ring_c": torch.empty((0, dc), dtype=torch.float32),
                    "ring_x": torch.empty((0, dx), dtype=torch.float32),
                    "ring_y": torch.empty((0, dx), dtype=torch.float32),
                }
                bank["strata"].append(st)
                smap[key] = st
            for ring_name, src in (("ring_c", cb), ("ring_x", xb), ("ring_y", yb)):
                st[ring_name] = torch.cat((st[ring_name], src.float()), 0)[-384:]

    if tn > 0:
        rec.observe_cond_fiber((r_imq_sum / float(tn)) + min(0.5, 0.11 * max(0.0, r_op - 0.10)))
        return (tot / float(tn)).to(x_f.dtype)
    return x_f.new_zeros(())


class MinimalLoss(GANLoss):
    """Scale-consistent relativistic / Sinkhorn / Burg / CondFiber GAN loss for ``preset='minimal'``."""

    def __init__(
        self,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        *,
        loss: str = "relativistic",
        tau_logit: float = 3.5,
        minimal_record: Any = None,
        **_: Any,
    ) -> None:
        super().__init__(loss, labels=labels)
        self.tau_logit = float(tau_logit)
        self.minimal_record = minimal_record

    def _rec(self, *ts: Optional[torch.Tensor]) -> Optional[MinimalStepRecord]:
        if self.minimal_record is not None:
            return self.minimal_record
        for t in ts:
            if t is not None and getattr(t, "_minimal_record", None) is not None:
                return getattr(t, "_minimal_record")
        return None

    @staticmethod
    def _same_fiber(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]) -> bool:
        c1 = getattr(t1, "_minimal_cond_tensors", ())
        c2 = getattr(t2, "_minimal_cond_tensors", ())
        if not c1 or not c2 or len(c1) != len(c2):
            return False
        if not any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in c1):
            return False
        for a, b in zip(c1, c2):
            if (
                not isinstance(a, torch.Tensor)
                or not isinstance(b, torch.Tensor)
                or a.shape != b.shape
                or a.dtype != b.dtype
                or a.device != b.device
                or not bool(torch.equal(a, b))
            ):
                return False
        return True

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        rec = self._rec(fake_logits, real_logits)
        if rec is not None and torch.is_grad_enabled() and not rec._in_protected_probe:
            rec.observe_margin(float((r.detach().float().mean() - f.detach().float().mean()).item()))
        if self._same_fiber(real_logits, fake_logits) and r.shape == f.shape:
            return _protected_softplus_neg(r - f, self.tau_logit).mean()
        if self.loss == "relativistic":
            xr = getattr(real_logits, "_minimal_input", None)
            xf = getattr(fake_logits, "_minimal_input", None)
            if (
                isinstance(xr, torch.Tensor)
                and isinstance(xf, torch.Tensor)
                and xr.shape[1:] == xf.shape[1:]
                and 1 <= xr[0].numel() <= 256
                and 8 <= xr.shape[0] <= 2048
                and 8 <= xf.shape[0] <= 2048
                and xr.shape[0] == r.shape[0]
                and xf.shape[0] == f.shape[0]
            ):
                crf = pairwise_sq_dist(xr.detach().flatten(1).float(), xf.detach().flatten(1).float())
                pi = _sinkhorn_log(crf, crf.median().clamp_min(1e-12), 0.15, 5)
                return (pi * _protected_softplus_neg(r.unsqueeze(1) - f.unsqueeze(0), self.tau_logit)).sum()
            return 0.5 * (
                _protected_softplus_neg(r - f.mean(), self.tau_logit).mean()
                + _protected_softplus_neg(r.mean() - f, self.tau_logit).mean()
            )
        return super().d_loss(real_logits, fake_logits)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = fake_logits.reshape(-1)
        rec = self._rec(fake_logits, real_logits)
        if rec is not None:
            in_p = rec._in_protected_probe
            c_d = rec.calibration_gate
            er = rec.ema_residual
            s_th = rec._thermal_scale
            ph = rec.chord_phase
        else:
            in_p = False
            c_d = 1.0
            er = 1.0
            s_th = 1.0
            ph = 0.0

        if real_logits is not None:
            r = real_logits.reshape(-1)
            xr = getattr(real_logits, "_minimal_input", None)
            xf = getattr(fake_logits, "_minimal_input", None)
            if (
                xr is None
                and rec is not None
                and rec._last_real_pair is not None
                and r.shape == rec._last_real_pair[1].shape
                and bool(torch.equal(r.detach(), rec._last_real_pair[1]))
            ):
                xr = rec._last_real_pair[0]
            if self._same_fiber(fake_logits, real_logits) and r.shape == f.shape:
                adv = _protected_softplus_neg(f - r, self.tau_logit).mean()
                if not in_p and isinstance(xf, torch.Tensor) and isinstance(xr, torch.Tensor) and xf.requires_grad:
                    cond_t = getattr(fake_logits, "_minimal_cond_tensors", ())
                    return adv + 1.5 * _cond_fiber_loss(xf, xr, cond_t, rec)
                return adv
            if self.loss in ("relativistic", "non_saturating"):
                if (
                    isinstance(xr, torch.Tensor)
                    and isinstance(xf, torch.Tensor)
                    and xr.shape[1:] == xf.shape[1:]
                    and 1 <= xr[0].numel() <= 256
                    and 8 <= xr.shape[0] <= 2048
                    and 8 <= xf.shape[0] <= 2048
                    and xr.shape[0] == r.shape[0]
                    and xf.shape[0] == f.shape[0]
                ):
                    dim = int(xf[0].numel())
                    crf = pairwise_sq_dist(xr.detach().flatten(1).float(), xf.detach().flatten(1).float())
                    # Normalized O(1) Sinkhorn expectation (no O(B) discontinuity across dim=256):
                    pi_t = _sinkhorn_log(crf, crf.median().clamp_min(1e-12), 0.15, 5).t()
                    adv = (pi_t * _protected_softplus_neg(f.unsqueeze(1) - r.unsqueeze(0), self.tau_logit)).sum()
                    if not in_p and (xf.requires_grad or xr.requires_grad) and s_th > FROZEN_THERMAL_SCALE:
                        th_a = math.sqrt(s_th) if dim == 1 else max(0.15, math.sqrt(s_th))
                        w_ot = ((0.45 if dim == 1 else 0.22) + 0.40 * _smoothstep01((er - 0.16) / 0.22) + 1.15 * (1.0 - c_d)) * th_a
                        w_iso = (0.35 * (0.70 if dim == 1 else 2.60)) * th_a
                        aux = xf.new_zeros(())
                        for src, tgt in ((xf, xr), (xr, xf.detach())):
                            if src.requires_grad:
                                disp, sig = _born_infeld_disp(src, tgt, er, c_d, ph)
                                tau = ((1.0 if dim == 1 else 0.12) * sig).clamp_min(1e-3)
                                err_sq = (src - (src.detach() + disp).detach()).flatten(1).float().square().sum(1)
                                ot_term = (tau * (torch.sqrt(1.0 + err_sq / (tau * tau)) - 1.0).mean()).to(src.dtype)
                                aux = aux + w_ot * ot_term
                                if 2 <= dim <= 8:
                                    aux = aux + w_iso * _burg_cov_loss(src, tgt, er, ph)
                        return c_d * adv + aux
                    return c_d * adv
                if self.loss == "relativistic":
                    return 0.5 * (
                        _protected_softplus_neg(f - r.mean(), self.tau_logit).mean()
                        + _protected_softplus_neg(f.mean() - r, self.tau_logit).mean()
                    )
                return _protected_softplus_neg(f, self.tau_logit).mean()
        if self.loss in ("relativistic", "non_saturating"):
            return _protected_softplus_neg(f, self.tau_logit).mean()
        return super().g_loss(fake_logits, real_logits)
