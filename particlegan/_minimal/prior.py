"""Half A3: Unified Conformal Pullback, Laguerre Condensation, fwAD Rail Rescue & Zero-eigh Prior Regularizer."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._stats import pairwise_sq_dist
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .dynamics import FROZEN_THERMAL_SCALE, MinimalGeneratorOptimizer, _is_finite_num, _smoothstep01
from .geometry import _PHI1, _protected_softplus_neg

_Q_NAMES = ("proto_c", "proto_var", "proto_w", "proto_hits", "psi", "w_fake_ema")
_PRIOR_KEYS = frozenset({"schema", "quant_init", "rows_init", "iso_ema", "res_ema", *_Q_NAMES, "row_m", "row_v", "row_n"})


def _resolve_kernel(prior: Any, latent: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
    bp = unwrap_prior(prior)
    if isinstance(bp, MoGParticlePrior):
        sigma = float(bp.sigma.item() if isinstance(bp.sigma, torch.Tensor) else bp.sigma)
        return bp.means()[indices], sigma, bp.z.requires_grad
    return latent, 0.0, bool(hasattr(bp, "z") and bp.z.requires_grad)


class MinimalPriorRegularizer(nn.Module):
    """Zero-eigh prior regularizer and streaming prototype quantizer for ``preset='minimal'``."""

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 0.01,
        target_std: float = 1.0,
        max_prototypes: int = 256,
        bound_device: Optional[torch.device] = None,
        bound_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        del reference
        self.weight = float(weight)
        self.target_std = float(target_std)
        self.max_prototypes = int(max_prototypes)
        self._bound_device = bound_device
        self._bound_shape = bound_shape

        self.thermal_source: Any = None
        self.minimal_record: Any = None
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_var: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_hits: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.w_fake_ema: Optional[torch.Tensor] = None
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None
        self.quant_init = False
        self.rows_init = False
        self.iso_ema = 1.0
        self.res_ema = 1.0

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0.0 or rows.ndim != 2 or len(rows) < 2:
            return rows.new_zeros(())
        if len(rows) > 256:
            z = rows[(torch.arange(256, device=rows.device) * len(rows)) // 256]
        else:
            z = rows
        zc = z - z.mean(0, keepdim=True)
        cov = (zc.T @ zc) / float(max(1, len(z) - 1))
        d2 = pairwise_sq_dist(z) + torch.eye(len(z), device=z.device, dtype=z.dtype) * 1e6
        var_pen = F.relu(self.target_std - (zc.square().mean(0) + 1e-4).sqrt()).square().mean()
        cov_pen = (cov - torch.diag(cov.diagonal())).square().sum() / float(max(1, rows.shape[1]))
        rep_pen = 0.05 * torch.exp(-d2.min(1).values / max(0.5, 0.25 * float(rows.shape[1]))).mean()
        term = self.weight * (var_pen + cov_pen + rep_pen)
        s = float(getattr(self.thermal_source, "_thermal_scale", 1.0)) if self.thermal_source is not None else 1.0
        if s <= FROZEN_THERMAL_SCALE and term.requires_grad:
            return term * s + term.detach() * (1.0 - s)
        return term

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        x = yr.detach().float()
        b = int(x.shape[0])
        dim = int(x.shape[1])

        if not self.quant_init or self.proto_c is None or self.proto_c.shape[1] != dim:
            k = min(self.max_prototypes, max(4, (3 * b) // 4)) if b >= 8 else min(self.max_prototypes, b)
            proj = torch.remainder(torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) * _PHI1, 1.0) - 0.5
            order = torch.argsort(x @ (proj / proj.norm().clamp_min(1e-8)), stable=True)
            pick = ((torch.arange(k, device=x.device) * b + b // 2) // k).clamp(0, b - 1)
            self.proto_c = x[order[pick]].clone()
            d_cc = torch.cdist(self.proto_c, self.proto_c) + torch.eye(k, device=x.device) * 1e9
            nn_d = d_cc.min(1).values
            pos = nn_d[nn_d > 1e-7]
            if k >= 2 and pos.numel() > 0:
                v0 = 0.25 * (float(pos.median().item()) ** 2) / float(dim + 2)
            else:
                v0 = float(x.var(0, unbiased=False).mean().clamp_min(1e-6).item()) / float(max(1, k))
            self.proto_var = torch.full((k,), max(v0, 1e-8), device=x.device)
            self.proto_w = torch.full((k,), 1.0 / float(k), device=x.device)
            self.proto_hits = torch.ones(k, device=x.device)
            self.psi = torch.zeros(k, device=x.device)
            self.w_fake_ema = self.proto_w.clone()
            self.quant_init = True
            return

        if self.proto_c.device != x.device:
            for nm in (*_Q_NAMES, "row_m", "row_v", "row_n"):
                cur_t = getattr(self, nm)
                if isinstance(cur_t, torch.Tensor):
                    setattr(self, nm, cur_t.to(x.device))

        k = len(self.proto_c)
        c_prev = self.proto_c.clone()
        d2_min, assign = torch.cdist(x, c_prev).square().min(dim=1)
        med_v = self.proto_var.median().clamp_min(1e-10)
        b_cnt = torch.bincount(assign, minlength=k).float()
        raw_sp = 2.0 * x.var(0, unbiased=False).sum()
        has_sp = float(raw_sp.item()) > 1e-10
        pair_sp = raw_sp.clamp_min(1e-12)
        rel_dist = float(torch.sqrt(d2_min.mean() / pair_sp).item()) if has_sp else 0.0
        shock = max(0.0, min(1.0, (rel_dist - 0.45) / 0.35))
        if shock > 0.25:
            self.proto_hits.mul_(1.0 - 0.8 * shock).clamp_min_(1.0)

        alpha_w = 0.02 + 0.48 * (shock * shock)
        self.proto_w = (1.0 - alpha_w) * self.proto_w + alpha_w * (b_cnt / float(b))
        self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

        hit = b_cnt > 0
        if bool(hit.any()):
            sq_sum = torch.zeros(k, device=x.device).scatter_add_(0, assign, (x - c_prev[assign]).square().sum(1) / float(dim))
            self.proto_var[hit] = 0.88 * self.proto_var[hit] + 0.12 * (sq_sum[hit] / b_cnt[hit]).clamp_min(1e-10)
            self.proto_hits[hit] = (self.proto_hits[hit] + b_cnt[hit]).clamp_max(2000.0)
            ac = (
                ((0.035 + 0.25 / self.proto_hits[hit].sqrt()) + 0.38 * shock * shock)
                * (b_cnt[hit] / (b_cnt[hit] + 2.0))
            ).clamp_max(0.50).unsqueeze(1)
            sum_x = torch.zeros_like(c_prev).index_add_(0, assign, x)
            self.proto_c[hit] = (1.0 - ac) * c_prev[hit] + ac * (sum_x[hit] / b_cnt[hit].unsqueeze(1))

        mean_w = 1.0 / float(k)
        denom_ur = torch.maximum(36.0 * float(dim) * torch.maximum(self.proto_var[assign], med_v), 0.08 * pair_sp)
        max_ur, worst_j = (d2_min / denom_ur).max(0)
        if float(max_ur.item()) > 1.0 and k >= 8 and has_sp:
            if bool((self.proto_w < 0.04 * mean_w).any()):
                slot = int(self.proto_w.argmin().item())
            else:
                d_cc_sq = torch.cdist(self.proto_c, self.proto_c).square() + torch.eye(k, device=x.device) * 1e12
                slot = int(d_cc_sq.min(1).values.argmin().item())
            self.proto_c[slot] = x[worst_j]
            self.proto_var[slot] = torch.maximum(self.proto_var.amax(), 0.04 * pair_sp / float(dim))
            self.proto_hits[slot] = 2.0
            self.psi[slot] = 0.0
            new_w = max(alpha_w / float(b), 0.05 * mean_w)
            self.proto_w[slot] = new_w
            self.w_fake_ema[slot] = new_w
            self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        out: Dict[str, Any] = {
            "schema": 1,
            "quant_init": bool(self.quant_init),
            "rows_init": bool(self.rows_init),
            "iso_ema": float(self.iso_ema),
            "res_ema": float(self.res_ema),
        }
        for nm in (*_Q_NAMES, "row_m", "row_v", "row_n"):
            ten = getattr(self, nm)
            out[nm] = None if ten is None else ten.detach().cpu().clone()
        return out

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        del strict
        if (
            not isinstance(state, dict)
            or set(state) != _PRIOR_KEYS
            or state["schema"] != 1
            or any(type(state[k]) is not bool for k in ("quant_init", "rows_init"))
            or any(not _is_finite_num(state[k], 0.0) for k in ("iso_ema", "res_ema"))
        ):
            raise ValueError("invalid MinimalPriorRegularizer state")
        if state["quant_init"]:
            for nm in _Q_NAMES:
                ten = state[nm]
                if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                    raise ValueError("invalid MinimalPriorRegularizer prototype tensors")
            if state["proto_c"].ndim != 2 or state["proto_c"].shape[0] > self.max_prototypes:
                raise ValueError("invalid MinimalPriorRegularizer prototype tensors")
        elif any(state[nm] is not None for nm in _Q_NAMES):
            raise ValueError("uninitialized MinimalPriorRegularizer cannot carry prototypes")
        if state["rows_init"] and self._bound_shape is not None and tuple(state["row_m"].shape) != tuple(self._bound_shape):
            raise ValueError("MinimalPriorRegularizer row_m shape mismatch")

        self.quant_init = state["quant_init"]
        self.rows_init = state["rows_init"]
        self.iso_ema = float(state["iso_ema"])
        self.res_ema = float(state["res_ema"])
        for nm in (*_Q_NAMES, "row_m", "row_v", "row_n"):
            ten = state[nm]
            if ten is None:
                setattr(self, nm, None)
            else:
                target_dtype = torch.float32 if nm in ("row_m", "row_v", "row_n") else ten.dtype
                if self._bound_device is not None:
                    setattr(self, nm, ten.to(device=self._bound_device, dtype=target_dtype).clone())
                else:
                    setattr(self, nm, ten.to(dtype=target_dtype).clone())


def should_use_minimal_pullback(
    trainer: Any,
    real: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    r = trainer.recipe
    if (
        r.prior_regularizer != "minimal"
        or float(r.t1_pullback_weight) <= 0.0
        or r.encoder_mode != "none"
        or r.conditioning != "scalar"
        or real.ndim != 2
        or not (1 <= int(real.shape[1]) <= 16)
    ):
        return False
    is_point = float(real.detach().float().var(0, unbiased=False).sum().item()) < 1e-11
    return is_point or (int(real.shape[1]) >= 2 and _resolve_kernel(trainer.prior, latent, indices)[1] > 0.0)


@torch.no_grad()
def _coalesce_isolated(
    pc: torch.Tensor,
    pv: torch.Tensor,
    pw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = pc.device
    k = int(pc.shape[0])
    d = int(pc.shape[1])
    pc_c = pc.cpu()
    pv_c = pv.cpu()
    pw_c = pw.cpu()
    sig = pv_c.clamp_min(1e-6).sqrt()
    d_pp = torch.cdist(pc_c, pc_c)
    adj = d_pp <= 2.4 * math.sqrt(float(d)) * torch.maximum(sig.unsqueeze(1), sig.unsqueeze(0))
    for _ in range(3):
        adj = (adj.float() @ adj.float()) > 0
    cid = adj.float().argmax(1)
    wm = torch.eye(k)
    coal = torch.zeros(k, dtype=torch.bool)
    for c in torch.unique(cid):
        idx = torch.nonzero(cid == c, as_tuple=False).squeeze(-1)
        outside = d_pp[idx][:, cid != c]
        is_isolated = outside.numel() == 0 or float(outside.min().item()) >= 4.2 * math.sqrt(float(d)) * float(sig[idx].max().item())
        if 1 <= idx.numel() <= 4 and is_isolated:
            wm[idx] = 0.0
            wm[idx.unsqueeze(1), idx.unsqueeze(0)] = (pw_c[idx] / pw_c[idx].sum().clamp_min(1e-12)).unsqueeze(0)
            coal[idx] = True
    wm_d = wm.to(dev)
    coal_d = coal.to(dev)
    cb = wm_d @ pc
    sb = (wm_d @ (pv + (pc - cb).square().mean(1))).clamp_min(1e-6).sqrt()
    return cb, sb, wm_d, coal_d


def minimal_pullback_step(
    trainer: Any,
    real: torch.Tensor,
    generator_real: Any,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    sigma_out: Any,
    noise: torch.Generator,
    critic: nn.Module,
    loss_d: torch.Tensor,
    penalty: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    recipe = trainer.recipe
    reg = trainer.prior_regularizer
    bp = unwrap_prior(trainer.prior)
    b = len(real)

    mu_g, sig_z, can_cond = _resolve_kernel(trainer.prior, latent, indices)
    is_pt = float(real.detach().float().var(0, unbiased=False).sum().item()) < 1e-11
    if generator_real is None:
        real_g = real
    else:
        raw_g_real = generator_real() if callable(generator_real) else generator_real
        real_g = trainer._batch(raw_g_real, "generator_real")

    yr = real_g.detach().reshape(b, -1)
    dim = int(real_g.shape[1])
    mu_det = mu_g.detach()
    eps_g = ((latent - mu_g) / sig_z).detach() if (sig_z > 0.0 and not is_pt) else None
    if is_pt:
        z_in = latent
    elif eps_g is not None:
        z_in = mu_det + sig_z * eps_g
    else:
        z_in = mu_det

    fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_in, sigma_out, noise, rows=indices))
    fake_f = fake_g.reshape(b, -1)
    if eps_g is None:
        x_ctr = fake_f
    else:
        x_ctr = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, mu_det, 0.0, noise)).reshape(b, -1)

    fake_logits = critic(fake_g)
    real_logits = critic(real_g)
    if recipe.loss == "non_saturating":
        loss_gan = F.softplus(-fake_logits).mean()
    else:
        loss_gan = _protected_softplus_neg(fake_logits.reshape(-1) - real_logits.detach().reshape(-1).mean()).mean()

    with torch.no_grad():
        if is_pt:
            err = (fake_f.detach().float() - yr.float()).norm(dim=-1)
            cr = max(float(err.mean().item()), 0.35 * float(err.amax().item()))
            if cr > max(3.5 * reg.res_ema, reg.res_ema + 0.12):
                reg.res_ema = min(cr, 10.0)
            else:
                reg.res_ema = 0.88 * reg.res_ema + 0.12 * min(cr, 10.0)
            try:
                with fwAD.dual_level():
                    u_pt = torch.ones_like(z_in.detach()) / math.sqrt(max(1, z_in.shape[1]))
                    _, jvp = fwAD.unpack_dual(trainer._postprocess(trainer.prior, trainer.G(fwAD.make_dual(z_in.detach(), u_pt))))
            except (RuntimeError, NotImplementedError, TypeError):
                jvp = None
            sc = jvp.detach().float().reshape(b, -1).abs().clamp_min(1e-5) if jvp is not None else torch.ones_like(fake_f.float())
            med_sc = torch.maximum(torch.quantile(sc, 0.5, dim=0, keepdim=True), sc.new_full((1, dim), max(0.15 * reg.res_ema, 1e-4)))
            w_sat = (med_sc / sc).clamp(1.0, 12.0).to(fake_f.dtype)
            y_ot = yr.to(fake_f.dtype)
            wakeup = 1.0
            has_coal = False
            g_scale = max(0.001, min(2.0, reg.res_ema / 0.012))
            adv_w = 0.015 * _smoothstep01((reg.res_ema - 0.025) / 0.15)
        else:
            reg.update_quantizer(yr)
            yf = yr.float()
            r_mac = yf.std(0, unbiased=False).mean().clamp_min(1e-4)
            c_bar, s_bar, w_mat, coal = _coalesce_isolated(reg.proto_c, reg.proto_var, reg.proto_w)
            s_med = torch.quantile(s_bar, 0.35).clamp_min(1e-6)
            reg.iso_ema = 0.95 * reg.iso_ema + 0.05 * max(1.0, min(4.5, float((sig_z * r_mac / s_med).item())))
            near = torch.cdist(x_ctr.detach().float(), c_bar).argmin(1)
            is_c = coal[near]
            xff = torch.where(is_c.unsqueeze(1), x_ctr.detach().float(), fake_f.detach().float())
            sr = float((xff.std(0, unbiased=False).mean().clamp_min(1e-8) / r_mac).item())
            wakeup = 5.0 if sr < 0.50 else (2.0 if sr < 0.80 else 1.0)
            d_ph = torch.cdist(xff, c_bar)

            if sr < 0.35 and len(c_bar) >= 4:
                xff_norm = (xff - xff.mean(0, keepdim=True)) / xff.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
                c_mean = (reg.proto_w.unsqueeze(1) * c_bar).sum(0, keepdim=True)
                c_std = (reg.proto_w.unsqueeze(1) * (c_bar - c_mean).square()).sum(0, keepdim=True).sqrt().clamp_min(1e-6)
                d_rt = torch.cdist(xff_norm, (c_bar - c_mean) / c_std)
            else:
                d_rt = d_ph

            lx = -(d_rt / d_rt.mean().clamp_min(1e-5)) / max(0.010, min(0.06, 0.25 * reg.res_ema))
            la = -math.log(float(b))
            lb = torch.log(reg.proto_w.clamp_min(1e-12))
            u_s = torch.zeros(b, device=yr.device)
            v_s = torch.zeros(len(c_bar), device=yr.device)
            for _ in range(10):
                v_s = lb - torch.logsumexp(lx + u_s.unsqueeze(1), dim=0)
                u_s = la - torch.logsumexp(lx + v_s.unsqueeze(0), dim=1)
            rp = (float(b) * torch.exp(lx + u_s.unsqueeze(1) + v_s.unsqueeze(0))).clamp_min(0.0)
            y_ot = ((rp / rp.sum(1, keepdim=True).clamp_min(1e-12)) @ c_bar).to(fake_f.dtype)

            cover_dist = torch.maximum((reg.proto_w * d_ph.min(0).values).sum(), d_ph.min(1).values.median())
            res_obs = float(((cover_dist - 1.45 * math.sqrt(dim) * s_med).clamp_min(0.0) / r_mac).item())
            reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(res_obs, 10.0)

            has_coal = bool(coal.any().item())
            comp_rep = (w_mat > 0.0).float().argmax(1)
            raw_sig = s_bar[near]
            eff_sig = raw_sig
            if bool(is_c.any()):
                xcf_d = x_ctr.detach().float()
                cid_i = comp_rep[near]
                cnt_c = torch.bincount(cid_i, minlength=len(c_bar)).float()
                mean_c = torch.zeros_like(c_bar).index_add_(0, cid_i, xcf_d) / cnt_c.clamp_min(1.0).unsqueeze(1)
                vv = is_c & (cnt_c[cid_i] >= 2.0)
                if bool(vv.any()):
                    var_c = (cnt_c[cid_i[vv]] / (cnt_c[cid_i[vv]] - 1.0)) * (
                        (xcf_d[vv] - mean_c[cid_i[vv]]).square().sum(-1) / float(dim)
                    )
                    rho = (var_c / raw_sig[vv].square().clamp_min(1e-10)).clamp(0.0, 0.75).mean()
                    eff_sig = torch.where(is_c, raw_sig * torch.sqrt((1.05 - 0.85 * rho).clamp_min(0.35)), raw_sig)

            sig_eff = eff_sig.to(fake_f.dtype)
            inv_w = (1.0 / (len(reg.proto_c) * reg.proto_w[near]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_f.dtype)
            w_sat = torch.ones(b, device=yr.device, dtype=fake_f.dtype)
            if eps_g is not None:
                ve = eps_g.float().norm(dim=-1) >= 0.45
                if bool(ve.any()):
                    so = (fake_f.detach().float() - x_ctr.detach().float()).norm(dim=-1) / (
                        sig_z * eps_g.float().norm(dim=-1).clamp_min(0.45)
                    )
                    if has_coal:
                        sref = so[ve].median().clamp_min(1e-6)
                    else:
                        sref = eff_sig * math.sqrt(dim / max(1, mu_det.shape[1])) / max(sig_z, 1e-4)
                    thresh = (0.50 if has_coal else 0.40) * sref
                    w_sat = torch.where(ve & (so < thresh), (sref / so.clamp_min(1e-6)).clamp(1.0, 8.0), torch.ones_like(so)).to(fake_f.dtype)

            y_bnd = yf.abs().amax(0, keepdim=True) + 3.5 * s_med
            sc_med = s_med.to(fake_f.dtype)
            eq_g = max(0.0, min(1.0, (0.030 - reg.res_ema) / 0.020)) if has_coal else 0.0
            adv_w = (0.038 * (1.0 - eq_g) + 0.008 * eq_g) if has_coal else 0.15
            if sr < 0.60:
                g_scale = 1.0
            else:
                floor_g = (0.022 * (1.0 - eq_g) + 0.012 * eq_g) if has_coal else 0.04
                g_scale = max(floor_g, min(1.0, reg.res_ema / 0.10))

    if is_pt:
        l2_prior = 0.004 * bp.z.float().square().mean().to(fake_f.dtype) if bp.z.requires_grad else 0.0
        loss_ot = (torch.sqrt(1e-6 + w_sat * (fake_f - y_ot).square()) - 1e-3).mean() + l2_prior
        loss_iso = fake_f.new_zeros(())
    else:
        src_ot = torch.where(is_c.unsqueeze(1), x_ctr, fake_f)
        loss_ot = (
            (w_sat * (torch.sqrt(sc_med.square() + (src_ot - y_ot).square().sum(-1)) - sc_med)).mean()
            + 2.0 * F.relu(fake_f.float().abs() - y_bnd).square().mean().to(fake_f.dtype)
        )
        loss_iso = fake_f.new_zeros(())
        if eps_g is not None and recipe.t1_iso_weight > 0.0:
            dz = int(mu_det.shape[1])
            dx_ker = fake_f - x_ctr
            sc_d = math.sqrt(float(dim) / float(dz))
            t_norm = sig_eff * sc_d * eps_g.norm(dim=-1).clamp_min(0.25)
            iso_1 = (inv_w * ((dx_ker.norm(dim=-1) - t_norm) / (sc_med * sc_d).clamp_min(1e-6)).square()).mean()
            if dim == 2 and dz >= 2:
                mi = b if has_coal else min(b, 128)
                e0 = eps_g[:mi]
                if dz % 2 == 0:
                    ep = torch.stack((-e0[:, 1::2], e0[:, 0::2]), -1).reshape(mi, dz)
                else:
                    e_roll = e0.roll(1, 1)
                    ep = e_roll - ((e_roll * e0).sum(-1, keepdim=True) / e0.square().sum(-1, keepdim=True).clamp_min(1e-6)) * e0
                dx_p = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, mu_det[:mi] + sig_z * ep, 0.0, noise)).reshape(mi, -1) - x_ctr[:mi]
                iso_2 = (inv_w[:mi] * ((dx_p.norm(dim=-1) - t_norm[:mi]) / (sc_med * sc_d).clamp_min(1e-6)).square()).mean()
                m2 = torch.stack((dx_ker[:mi], dx_p), -1)
                sp2 = (m2 @ m2.transpose(1, 2)) / (e0.square().sum(-1) * (2.0 / float(dz))).clamp_min(0.15).view(mi, 1, 1)
                tg2 = sig_eff[:mi].square().view(mi, 1, 1) * torch.eye(2, device=yr.device, dtype=fake_f.dtype).unsqueeze(0)
                icov = 0.5 * (inv_w[:mi] * ((sp2 - tg2) / (sig_eff[:mi] * sc_med).clamp_min(1e-8).view(mi, 1, 1)).square().sum((1, 2))).mean()
                loss_iso = (0.35 * (iso_1 + iso_2) + 0.65 * icov) if has_coal else (0.45 * (iso_1 + iso_2) + 0.35 * icov)
            else:
                loss_iso = iso_1

    prior_l2_term = recipe.prior_l2 * bp.z.square().mean() if (bp.z.requires_grad and recipe.prior_l2 > 0) else 0.0
    loss_g = adv_w * loss_gan + float(recipe.t1_pullback_weight) * loss_ot + float(recipe.t1_iso_weight) * loss_iso + prior_l2_term
    if isinstance(trainer.opt_g, MinimalGeneratorOptimizer):
        trainer.opt_g.bind_modulation(g_scale=g_scale, wakeup_boost=wakeup)
    trainer.opt_g.zero_grad()
    constraint_geometry_backward(loss_g, trainer.opt_g, (loss_g,))
    if bp.z.requires_grad and bp.z.grad is None:
        bp.z.grad = torch.zeros_like(bp.z)
    trainer.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
    trainer.opt_g.step()
    trainer.policy.after_generator_step()

    prior_diag = loss_gan.new_zeros(())
    if (not is_pt) and can_cond and bp.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0:
        u_rows = torch.unique(indices)
        g_fl = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)
            cu = (bp.means() if isinstance(bp, MoGParticlePrior) else bp.z)[u_rows]
            xu = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, cu, 0.0, noise)).reshape(len(u_rows), -1)
            with torch.no_grad():
                ks = (torch.cdist(xu.detach().float(), c_bar).square() - (w_mat @ reg.psi).unsqueeze(0)).argmin(1)
                wh = w_mat[ks].mean(0)
                reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * wh
                tv_f = 0.35 * math.sqrt(len(reg.proto_c) / max(len(u_rows) * 8.0, 1.0))
                rel_def = float(((reg.proto_w - reg.w_fake_ema) / reg.proto_w.clamp_min(0.01)).clamp_min(0.0).amax().item())
                tv_val = 0.5 * float((reg.proto_w - reg.w_fake_ema).abs().sum().item())
                pg = max(0.0, min(1.0, max((tv_val - tv_f) / max(0.08, tv_f), (rel_def - 0.35) / 0.45)))
                pcap = 2.5 * dim * float((r_mac * r_mac).item()) + 0.5
                psi_step = pg * min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * len(reg.proto_c))) * dim * float((r_mac * r_mac).item())
                reg.psi.mul_(1.0 - 0.008 * pg).add_(reg.proto_w - (0.35 * wh + 0.65 * reg.w_fake_ema), alpha=psi_step)
                # Center strictly AFTER coalescing matrix multiplication (and re-center after clamping):
                psi_post = w_mat @ reg.psi
                psi_post = (psi_post - psi_post.mean()).clamp(-pcap, pcap)
                reg.psi = psi_post - psi_post.mean()
                ctgt = c_bar[ks].to(xu.dtype)
                sc_u = s_bar[ks].to(xu.dtype)

            sq_u = (xu - ctgt).square().sum(-1)
            if sig_z > 0.0 and len(u_rows) >= 2:
                z_u = bp.z[u_rows]
                z_std = ((z_u - z_u.mean(0, keepdim=True)).square().mean(0) + 1e-6).sqrt()
                iso_u = 0.15 * F.relu(reg.iso_ema - z_std).square().mean()
            else:
                iso_u = 0.0
            lm = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean() + iso_u
            prior_diag = lm.detach()

            if bp.z.grad is not None:
                bp.z.grad.zero_()
            lm.backward()

            if not reg.rows_init or reg.row_m is None or reg.row_m.shape != bp.z.shape:
                reg.row_m = torch.zeros(bp.z.shape, device=bp.z.device, dtype=torch.float32)
                reg.row_v = torch.zeros((len(bp.z), 1), device=bp.z.device, dtype=torch.float32)
                reg.row_n = torch.zeros((len(bp.z), 1), device=bp.z.device, dtype=torch.float32)
                reg.rows_init = True
            elif reg.row_m.dtype != torch.float32 or reg.row_v.dtype != torch.float32 or reg.row_n.dtype != torch.float32:
                reg.row_m = reg.row_m.float()
                reg.row_v = reg.row_v.float()
                reg.row_n = reg.row_n.float()

            with torch.no_grad():
                gr = (bp.z.grad[u_rows] * float(len(u_rows))).clone()
                bp.z.grad.zero_()
                ps = (sq_u.detach().sqrt().unsqueeze(1) / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)
                far = sq_u.detach().sqrt().float() > (2.8 * math.sqrt(dim) * sc_u.float())
                if cu.shape[1] <= 8 and bool(far.any()):
                    fi = torch.nonzero(far, as_tuple=False).squeeze(-1)
                    ii = torch.nonzero(~far, as_tuple=False).squeeze(-1)[:64]
                    if ii.numel() < 4:
                        ii = torch.arange(min(64, len(cu)), device=cu.device)
                    pi_i = torch.cat([fi, ii], 0)
                    dz_u = int(cu.shape[1])
                    cp = cu[pi_i].detach()
                    xp = xu.detach().float()[pi_i]
                    eps_fd = 1e-2 if cp.dtype == torch.bfloat16 else 1e-3
                    out_sgn = torch.where(cp >= 0.0, 1.0, -1.0)
                    eye_z = torch.eye(dz_u, device=cu.device, dtype=torch.float32)
                    cpert = (
                        cp.float().unsqueeze(1) + (eps_fd * out_sgn).unsqueeze(2) * eye_z.unsqueeze(0)
                    ).reshape(-1, dz_u).to(dtype=cp.dtype)
                    xpert = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, cpert, 0.0, noise)).float()
                    csen = (xpert.reshape(len(pi_i), dz_u, dim) - xp.unsqueeze(1)).norm(dim=-1) / eps_fd
                    ref_sen = torch.quantile(csen[len(fi) :], 0.5, dim=0, keepdim=True).clamp_min(1e-6)
                    dead = csen[: len(fi)] < (0.04 * ref_sen)
                    if bool(dead.any()):
                        g_mag = torch.maximum(gr[fi].float().norm(dim=-1, keepdim=True), torch.ones((len(fi), 1), device=cu.device))
                        inward = (torch.sign(bp.z[u_rows][fi].detach() - cu[ii].detach().mean(0, keepdim=True)).float() * g_mag).to(gr.dtype)
                        gr[fi] = torch.where(dead, inward, gr[fi])
                        ps[fi] = torch.where(dead.any(1, keepdim=True), torch.ones_like(ps[fi]), ps[fi])

                gr_f = gr.float()
                m = reg.row_m[u_rows] * 0.80 + 0.20 * gr_f
                v = reg.row_v[u_rows] * 0.80 + 0.20 * gr_f.square().sum(-1, keepdim=True)
                n = reg.row_n[u_rows] + 1.0
                reg.row_m[u_rows] = m
                reg.row_v[u_rows] = v
                reg.row_n[u_rows] = n
                bc = 1.0 - (0.80 ** n)
                step_u = (((m / bc) / ((v / bc).sqrt() + 1e-5)) * ps.float()).to(bp.z.dtype)
                lr_row = -float(recipe.t1_prior_row_lr) * max(0.15, min(1.0, max(reg.res_ema / 0.10, pg)))
                bp.z.index_add_(0, u_rows, step_u, alpha=lr_row)
        finally:
            for p, fl in zip(trainer.G.parameters(), g_fl):
                p.requires_grad_(fl)
            if getattr(trainer.opt_d, "record", None) is not None:
                trainer.opt_d.record._g_probe_cache = None

    zero = loss_gan.new_zeros(())
    return loss_g, loss_gan, prior_diag, None, zero, zero
