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
    if isinstance(bp, MoGParticlePrior): return bp.means()[indices], float(bp.sigma.item() if isinstance(bp.sigma, torch.Tensor) else bp.sigma), bp.z.requires_grad
    return latent, 0.0, bool(hasattr(bp, "z") and bp.z.requires_grad)

class MinimalPriorRegularizer(nn.Module):
    def __init__(self, reference: Any = None, *, weight: float = 0.01, target_std: float = 1.0, max_prototypes: int = 256, bound_device: Optional[torch.device] = None, bound_shape: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(); del reference
        self.weight, self.target_std, self.max_prototypes, self._bound_device, self._bound_shape = float(weight), float(target_std), int(max_prototypes), bound_device, bound_shape
        self.thermal_source = self.minimal_record = self.proto_c = self.proto_var = self.proto_w = self.proto_hits = self.psi = self.w_fake_ema = self.row_m = self.row_v = self.row_n = None
        self.quant_init = self.rows_init = False; self.iso_ema, self.res_ema = 1.0, 1.0

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0.0 or rows.ndim != 2 or len(rows) < 2: return rows.new_zeros(())
        z = rows[(torch.arange(256, device=rows.device) * len(rows)) // 256] if len(rows) > 256 else rows
        zc = z - z.mean(0, keepdim=True); cov = (zc.T @ zc) / float(max(1, len(z) - 1)); d2 = pairwise_sq_dist(z) + torch.eye(len(z), device=z.device, dtype=z.dtype) * 1e6
        term = self.weight * (F.relu(self.target_std - (zc.square().mean(0) + 1e-4).sqrt()).square().mean() + (cov - torch.diag(cov.diagonal())).square().sum() / float(max(1, rows.shape[1])) + 0.05 * torch.exp(-d2.min(1).values / max(0.5, 0.25 * float(rows.shape[1]))).mean())
        s = float(getattr(self.thermal_source, "_thermal_scale", 1.0)) if self.thermal_source is not None else 1.0
        return (term * s + term.detach() * (1.0 - s)) if (s <= FROZEN_THERMAL_SCALE and term.requires_grad) else term

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        x = yr.detach().float(); b, dim = int(x.shape[0]), int(x.shape[1])
        if not self.quant_init or self.proto_c is None or self.proto_c.shape[1] != dim:
            k = min(self.max_prototypes, max(4, (3 * b) // 4)) if b >= 8 else min(self.max_prototypes, b)
            proj = torch.remainder(torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) * _PHI1, 1.0) - 0.5
            order = torch.argsort(x @ (proj / proj.norm().clamp_min(1e-8)), stable=True)
            self.proto_c = x[order[((torch.arange(k, device=x.device) * b + b // 2) // k).clamp(0, b - 1)]].clone()
            d_cc = torch.cdist(self.proto_c, self.proto_c) + torch.eye(k, device=x.device) * 1e9; pos = d_cc.min(1).values[d_cc.min(1).values > 1e-7]
            v0 = (0.25 * float(pos.median().item()) ** 2 / float(dim + 2)) if (k >= 2 and pos.numel() > 0) else (float(x.var(0, unbiased=False).mean().clamp_min(1e-6).item()) / float(max(1, k)))
            self.proto_var, self.proto_w = torch.full((k,), max(v0, 1e-8), device=x.device), torch.full((k,), 1.0 / float(k), device=x.device)
            self.proto_hits, self.psi, self.w_fake_ema, self.quant_init = torch.ones(k, device=x.device), torch.zeros(k, device=x.device), self.proto_w.clone(), True; return
        if self.proto_c.device != x.device:
            for nm in (*_Q_NAMES, "row_m", "row_v", "row_n"):
                if isinstance(getattr(self, nm), torch.Tensor): setattr(self, nm, getattr(self, nm).to(x.device))
        k, c_prev = len(self.proto_c), self.proto_c.clone(); d2_min, assign = torch.cdist(x, c_prev).square().min(dim=1)
        med_v, b_cnt, raw_sp = self.proto_var.median().clamp_min(1e-10), torch.bincount(assign, minlength=k).float(), 2.0 * x.var(0, unbiased=False).sum()
        has_sp, pair_sp = float(raw_sp.item()) > 1e-10, raw_sp.clamp_min(1e-12)
        shock = max(0.0, min(1.0, ((float(torch.sqrt(d2_min.mean() / pair_sp).item()) if has_sp else 0.0) - 0.45) / 0.35))
        if shock > 0.25: self.proto_hits.mul_(1.0 - 0.8 * shock).clamp_min_(1.0)
        alpha_w = 0.02 + 0.48 * (shock * shock); self.proto_w = (1.0 - alpha_w) * self.proto_w + alpha_w * (b_cnt / float(b)); self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
        hit = b_cnt > 0
        if bool(hit.any()):
            self.proto_var[hit] = 0.88 * self.proto_var[hit] + 0.12 * (torch.zeros(k, device=x.device).scatter_add_(0, assign, (x - c_prev[assign]).square().sum(1) / float(dim))[hit] / b_cnt[hit]).clamp_min(1e-10)
            self.proto_hits[hit] = (self.proto_hits[hit] + b_cnt[hit]).clamp_max(2000.0)
            ac = (((0.035 + 0.25 / self.proto_hits[hit].sqrt()) + 0.38 * shock * shock) * (b_cnt[hit] / (b_cnt[hit] + 2.0))).clamp_max(0.50).unsqueeze(1)
            self.proto_c[hit] = (1.0 - ac) * c_prev[hit] + ac * (torch.zeros_like(c_prev).index_add_(0, assign, x)[hit] / b_cnt[hit].unsqueeze(1))
        mean_w = 1.0 / float(k); max_ur, worst_j = (d2_min / torch.maximum(36.0 * float(dim) * torch.maximum(self.proto_var[assign], med_v), 0.08 * pair_sp)).max(0)
        if float(max_ur.item()) > 1.0 and k >= 8 and has_sp:
            slot = int(self.proto_w.argmin().item()) if bool((self.proto_w < 0.04 * mean_w).any()) else int((torch.cdist(self.proto_c, self.proto_c).square() + torch.eye(k, device=x.device) * 1e12).min(1).values.argmin().item())
            self.proto_c[slot], self.proto_var[slot], self.proto_hits[slot], self.psi[slot] = x[worst_j], torch.maximum(self.proto_var.amax(), 0.04 * pair_sp / float(dim)), 2.0, 0.0
            self.proto_w[slot] = self.w_fake_ema[slot] = max(alpha_w / float(b), 0.05 * mean_w); self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: del args, kwargs; return {"schema": 1, "quant_init": bool(self.quant_init), "rows_init": bool(self.rows_init), "iso_ema": float(self.iso_ema), "res_ema": float(self.res_ema), **{nm: None if getattr(self, nm) is None else getattr(self, nm).detach().cpu().clone() for nm in (*_Q_NAMES, "row_m", "row_v", "row_n")}}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        del strict
        if not isinstance(state, dict) or set(state) != _PRIOR_KEYS or state["schema"] != 1 or any(type(state[k]) is not bool for k in ("quant_init", "rows_init")) or any(not _is_finite_num(state[k], 0.0) for k in ("iso_ema", "res_ema")):
            raise ValueError("invalid MinimalPriorRegularizer state")
        if state["quant_init"] and (any(not isinstance(state[n], torch.Tensor) or not bool(torch.isfinite(state[n]).all()) for n in _Q_NAMES) or state["proto_c"].ndim != 2 or state["proto_c"].shape[0] > self.max_prototypes): raise ValueError("invalid MinimalPriorRegularizer prototype tensors")
        if not state["quant_init"] and any(state[n] is not None for n in _Q_NAMES): raise ValueError("uninitialized MinimalPriorRegularizer cannot carry prototypes")
        if state["rows_init"] and self._bound_shape is not None and tuple(state["row_m"].shape) != tuple(self._bound_shape): raise ValueError("MinimalPriorRegularizer row_m shape mismatch")
        self.quant_init, self.rows_init, self.iso_ema, self.res_ema = state["quant_init"], state["rows_init"], float(state["iso_ema"]), float(state["res_ema"])
        for nm in (*_Q_NAMES, "row_m", "row_v", "row_n"):
            t = state[nm]; setattr(self, nm, None if t is None else (t.to(self._bound_device).clone() if self._bound_device is not None else t.clone()))

def should_use_minimal_pullback(trainer: Any, real: torch.Tensor, latent: torch.Tensor, indices: torch.Tensor) -> bool:
    r = trainer.recipe
    if r.prior_regularizer != "minimal" or float(r.t1_pullback_weight) <= 0.0 or r.encoder_mode != "none" or r.conditioning != "scalar" or real.ndim != 2 or not (1 <= int(real.shape[1]) <= 16): return False
    return float(real.detach().float().var(0, unbiased=False).sum().item()) < 1e-11 or (int(real.shape[1]) >= 2 and _resolve_kernel(trainer.prior, latent, indices)[1] > 0.0)

@torch.no_grad()
def _coalesce_isolated(pc: torch.Tensor, pv: torch.Tensor, pw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev, k, d = pc.device, int(pc.shape[0]), int(pc.shape[1]); pc_c, pv_c, pw_c = pc.cpu(), pv.cpu(), pw.cpu()
    sig, d_pp = pv_c.clamp_min(1e-6).sqrt(), torch.cdist(pc_c, pc_c); adj = d_pp <= 2.4 * math.sqrt(float(d)) * torch.maximum(sig.unsqueeze(1), sig.unsqueeze(0))
    for _ in range(3): adj = (adj.float() @ adj.float()) > 0
    cid, wm, coal = adj.float().argmax(1), torch.eye(k), torch.zeros(k, dtype=torch.bool)
    for c in torch.unique(cid):
        idx = torch.nonzero(cid == c, as_tuple=False).squeeze(-1)
        if 1 <= idx.numel() <= 4 and (d_pp[idx][:, cid != c].numel() == 0 or float(d_pp[idx][:, cid != c].min().item()) >= 4.2 * math.sqrt(float(d)) * float(sig[idx].max().item())):
            wm[idx] = 0.0; wm[idx.unsqueeze(1), idx.unsqueeze(0)] = (pw_c[idx] / pw_c[idx].sum().clamp_min(1e-12)).unsqueeze(0); coal[idx] = True
    wm_d, coal_d = wm.to(dev), coal.to(dev); cb = wm_d @ pc
    return cb, (wm_d @ (pv + (pc - cb).square().mean(1))).clamp_min(1e-6).sqrt(), wm_d, coal_d

def minimal_pullback_step(trainer: Any, real: torch.Tensor, generator_real: Any, latent: torch.Tensor, indices: torch.Tensor, *,
                          sigma_out: Any, noise: torch.Generator, critic: nn.Module, loss_d: torch.Tensor, penalty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    recipe, reg, bp, b = trainer.recipe, trainer.prior_regularizer, unwrap_prior(trainer.prior), len(real)
    mu_g, sig_z, can_cond = _resolve_kernel(trainer.prior, latent, indices); is_pt = float(real.detach().float().var(0, unbiased=False).sum().item()) < 1e-11
    real_g = real if generator_real is None else trainer._batch(generator_real() if callable(generator_real) else generator_real, "generator_real")
    yr, dim, mu_det = real_g.detach().reshape(b, -1), int(real_g.shape[1]), mu_g.detach()
    eps_g = ((latent - mu_g) / sig_z).detach() if (sig_z > 0.0 and not is_pt) else None
    z_in = latent if is_pt else (mu_det + sig_z * eps_g if eps_g is not None else mu_det)
    fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_in, sigma_out, noise, rows=indices)); fake_f = fake_g.reshape(b, -1)
    x_ctr = fake_f if eps_g is None else trainer._postprocess(trainer.prior, trainer._generate(trainer.G, mu_det, 0.0, noise)).reshape(b, -1)
    fake_logits, real_logits = critic(fake_g), critic(real_g)
    loss_gan = F.softplus(-fake_logits).mean() if recipe.loss == "non_saturating" else _protected_softplus_neg(fake_logits.reshape(-1) - real_logits.detach().reshape(-1).mean()).mean()

    with torch.no_grad():
        if is_pt:
            err = (fake_f.detach().float() - yr.float()).norm(dim=-1); cr = max(float(err.mean().item()), 0.35 * float(err.amax().item()))
            reg.res_ema = min(cr, 10.0) if cr > max(3.5 * reg.res_ema, reg.res_ema + 0.12) else (0.88 * reg.res_ema + 0.12 * min(cr, 10.0))
            try:
                with fwAD.dual_level(): _, jvp = fwAD.unpack_dual(trainer._postprocess(trainer.prior, trainer.G(fwAD.make_dual(z_in.detach(), torch.ones_like(z_in.detach()) / math.sqrt(max(1, z_in.shape[1]))))))
            except (RuntimeError, NotImplementedError, TypeError): jvp = None
            sc = jvp.detach().float().reshape(b, -1).abs().clamp_min(1e-5) if jvp is not None else torch.ones_like(fake_f.float())
            w_sat = (torch.maximum(torch.quantile(sc, 0.5, dim=0, keepdim=True), sc.new_full((1, dim), max(0.15 * reg.res_ema, 1e-4))) / sc).clamp(1.0, 12.0).to(fake_f.dtype)
            y_ot, wakeup, has_coal, g_scale, adv_w = yr.to(fake_f.dtype), 1.0, False, max(0.001, min(2.0, reg.res_ema / 0.012)), 0.015 * _smoothstep01((reg.res_ema - 0.025) / 0.15)
        else:
            reg.update_quantizer(yr); yf, r_mac = yr.float(), yr.float().std(0, unbiased=False).mean().clamp_min(1e-4)
            c_bar, s_bar, w_mat, coal = _coalesce_isolated(reg.proto_c, reg.proto_var, reg.proto_w); s_med = torch.quantile(s_bar, 0.35).clamp_min(1e-6)
            reg.iso_ema = 0.95 * reg.iso_ema + 0.05 * max(1.0, min(4.5, float((sig_z * r_mac / s_med).item())))
            near = torch.cdist(x_ctr.detach().float(), c_bar).argmin(1); is_c = coal[near]; xff = torch.where(is_c.unsqueeze(1), x_ctr.detach().float(), fake_f.detach().float())
            sr = float((xff.std(0, unbiased=False).mean().clamp_min(1e-8) / r_mac).item()); wakeup, d_ph = (5.0 if sr < 0.50 else (2.0 if sr < 0.80 else 1.0)), torch.cdist(xff, c_bar)
            d_rt = torch.cdist((xff - xff.mean(0, keepdim=True)) / xff.std(0, unbiased=False, keepdim=True).clamp_min(1e-6), (c_bar - (reg.proto_w.unsqueeze(1) * c_bar).sum(0, keepdim=True)) / (reg.proto_w.unsqueeze(1) * (c_bar - (reg.proto_w.unsqueeze(1) * c_bar).sum(0, keepdim=True)).square()).sum(0, keepdim=True).sqrt().clamp_min(1e-6)) if (sr < 0.35 and len(c_bar) >= 4) else d_ph
            lx = -(d_rt / d_rt.mean().clamp_min(1e-5)) / max(0.010, min(0.06, 0.25 * reg.res_ema))
            la, lb, u_s, v_s = -math.log(float(b)), torch.log(reg.proto_w.clamp_min(1e-12)), torch.zeros(b, device=yr.device), torch.zeros(len(c_bar), device=yr.device)
            for _ in range(10): v_s, u_s = lb - torch.logsumexp(lx + u_s.unsqueeze(1), dim=0), la - torch.logsumexp(lx + v_s.unsqueeze(0), dim=1)
            rp = (float(b) * torch.exp(lx + u_s.unsqueeze(1) + v_s.unsqueeze(0))).clamp_min(0.0); y_ot = ((rp / rp.sum(1, keepdim=True).clamp_min(1e-12)) @ c_bar).to(fake_f.dtype)
            reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(float(((torch.maximum((reg.proto_w * d_ph.min(0).values).sum(), d_ph.min(1).values.median()) - 1.45 * math.sqrt(dim) * s_med).clamp_min(0.0) / r_mac).item()), 10.0)
            has_coal, comp_rep, raw_sig = bool(coal.any().item()), (w_mat > 0.0).float().argmax(1), s_bar[near]; eff_sig = raw_sig
            if bool(is_c.any()):
                xcf_d, cid_i = x_ctr.detach().float(), comp_rep[near]; cnt_c = torch.bincount(cid_i, minlength=len(c_bar)).float()
                mean_c, vv = torch.zeros_like(c_bar).index_add_(0, cid_i, xcf_d) / cnt_c.clamp_min(1.0).unsqueeze(1), is_c & (cnt_c[cid_i] >= 2.0)
                if bool(vv.any()):
                    rho = (((cnt_c[cid_i[vv]] / (cnt_c[cid_i[vv]] - 1.0)) * ((xcf_d[vv] - mean_c[cid_i[vv]]).square().sum(-1) / float(dim))) / raw_sig[vv].square().clamp_min(1e-10)).clamp(0.0, 0.75).mean()
                    eff_sig = torch.where(is_c, raw_sig * torch.sqrt((1.05 - 0.85 * rho).clamp_min(0.35)), raw_sig)
            sig_eff, inv_w, w_sat = eff_sig.to(fake_f.dtype), (1.0 / (len(reg.proto_c) * reg.proto_w[near]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_f.dtype), torch.ones(b, device=yr.device, dtype=fake_f.dtype)
            if eps_g is not None and bool((ve := eps_g.float().norm(dim=-1) >= 0.45).any()):
                so = (fake_f.detach().float() - x_ctr.detach().float()).norm(dim=-1) / (sig_z * eps_g.float().norm(dim=-1).clamp_min(0.45))
                sref = so[ve].median().clamp_min(1e-6) if has_coal else (eff_sig * math.sqrt(dim / max(1, mu_det.shape[1])) / max(sig_z, 1e-4))
                w_sat = torch.where(ve & (so < (0.50 if has_coal else 0.40) * sref), (sref / so.clamp_min(1e-6)).clamp(1.0, 8.0), torch.ones_like(so)).to(fake_f.dtype)
            y_bnd, sc_med, eq_g = yf.abs().amax(0, keepdim=True) + 3.5 * s_med, s_med.to(fake_f.dtype), (max(0.0, min(1.0, (0.030 - reg.res_ema) / 0.020)) if has_coal else 0.0)
            adv_w, g_scale = ((0.038 * (1.0 - eq_g) + 0.008 * eq_g) if has_coal else 0.15), (1.0 if sr < 0.60 else max((0.022 * (1.0 - eq_g) + 0.012 * eq_g) if has_coal else 0.04, min(1.0, reg.res_ema / 0.10)))

    if is_pt:
        loss_ot, loss_iso = (torch.sqrt(1e-6 + w_sat * (fake_f - y_ot).square()) - 1e-3).mean() + (0.004 * bp.z.float().square().mean().to(fake_f.dtype) if bp.z.requires_grad else 0.0), fake_f.new_zeros(())
    else:
        loss_ot, loss_iso = (w_sat * (torch.sqrt(sc_med.square() + (torch.where(is_c.unsqueeze(1), x_ctr, fake_f) - y_ot).square().sum(-1)) - sc_med)).mean() + 2.0 * F.relu(fake_f.float().abs() - y_bnd).square().mean().to(fake_f.dtype), fake_f.new_zeros(())
        if eps_g is not None and recipe.t1_iso_weight > 0.0:
            dz, dx_ker = int(mu_det.shape[1]), fake_f - x_ctr; sc_d = math.sqrt(float(dim) / float(dz)); t_norm = sig_eff * sc_d * eps_g.norm(dim=-1).clamp_min(0.25)
            iso_1 = (inv_w * ((dx_ker.norm(dim=-1) - t_norm) / (sc_med * sc_d).clamp_min(1e-6)).square()).mean()
            if dim == 2 and dz >= 2:
                mi = b if has_coal else min(b, 128); e0 = eps_g[:mi]
                ep = torch.stack((-e0[:, 1::2], e0[:, 0::2]), -1).reshape(mi, dz) if dz % 2 == 0 else (e0.roll(1, 1) - ((e0.roll(1, 1) * e0).sum(-1, keepdim=True) / e0.square().sum(-1, keepdim=True).clamp_min(1e-6)) * e0)
                dx_p = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, mu_det[:mi] + sig_z * ep, 0.0, noise)).reshape(mi, -1) - x_ctr[:mi]
                iso_2, M2 = (inv_w[:mi] * ((dx_p.norm(dim=-1) - t_norm[:mi]) / (sc_med * sc_d).clamp_min(1e-6)).square()).mean(), torch.stack((dx_ker[:mi], dx_p), -1)
                sp2 = (M2 @ M2.transpose(1, 2)) / (e0.square().sum(-1) * (2.0 / float(dz))).clamp_min(0.15).view(mi, 1, 1); tg2 = sig_eff[:mi].square().view(mi, 1, 1) * torch.eye(2, device=yr.device, dtype=fake_f.dtype).unsqueeze(0)
                icov = 0.5 * (inv_w[:mi] * ((sp2 - tg2) / (sig_eff[:mi] * sc_med).clamp_min(1e-8).view(mi, 1, 1)).square().sum((1, 2))).mean()
                loss_iso = (0.35 * (iso_1 + iso_2) + 0.65 * icov) if has_coal else (0.45 * (iso_1 + iso_2) + 0.35 * icov)
            else: loss_iso = iso_1

    loss_g = adv_w * loss_gan + float(recipe.t1_pullback_weight) * loss_ot + float(recipe.t1_iso_weight) * loss_iso + (recipe.prior_l2 * bp.z.square().mean() if (bp.z.requires_grad and recipe.prior_l2 > 0) else 0.0)
    if isinstance(trainer.opt_g, MinimalGeneratorOptimizer): trainer.opt_g.bind_modulation(g_scale=g_scale, wakeup_boost=wakeup)
    trainer.opt_g.zero_grad(); constraint_geometry_backward(loss_g, trainer.opt_g, (loss_g,))
    if bp.z.requires_grad and bp.z.grad is None: bp.z.grad = torch.zeros_like(bp.z)
    trainer.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach()); trainer.opt_g.step(); trainer.policy.after_generator_step()

    prior_diag = loss_gan.new_zeros(())
    if (not is_pt) and can_cond and bp.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0:
        u_rows, g_fl = torch.unique(indices), [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters(): p.requires_grad_(False)
            cu = (bp.means() if isinstance(bp, MoGParticlePrior) else bp.z)[u_rows]
            xu = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, cu, 0.0, noise)).reshape(len(u_rows), -1)
            with torch.no_grad():
                ks = (torch.cdist(xu.detach().float(), c_bar).square() - (w_mat @ reg.psi).unsqueeze(0)).argmin(1)
                wh = w_mat[ks].mean(0); reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * wh
                tv_f, rel_def = 0.35 * math.sqrt(len(reg.proto_c) / max(len(u_rows) * 8.0, 1.0)), float(((reg.proto_w - reg.w_fake_ema) / reg.proto_w.clamp_min(0.01)).clamp_min(0.0).amax().item())
                pg = max(0.0, min(1.0, max((0.5 * float((reg.proto_w - reg.w_fake_ema).abs().sum().item()) - tv_f) / max(0.08, tv_f), (rel_def - 0.35) / 0.45)))
                pcap = 2.5 * dim * float((r_mac * r_mac).item()) + 0.5
                reg.psi.mul_(1.0 - 0.008 * pg).add_(reg.proto_w - (0.35 * wh + 0.65 * reg.w_fake_ema), alpha=pg * min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * len(reg.proto_c))) * dim * float((r_mac * r_mac).item()))
                reg.psi = (w_mat @ reg.psi).sub(reg.psi.mean()).clamp(-pcap, pcap); ctgt, sc_u = c_bar[ks].to(xu.dtype), s_bar[ks].to(xu.dtype)
            sq_u = (xu - ctgt).square().sum(-1)
            lm = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean() + (0.15 * F.relu(reg.iso_ema - ((bp.z[u_rows] - bp.z[u_rows].mean(0, keepdim=True)).square().mean(0) + 1e-6).sqrt()).square().mean() if (sig_z > 0.0 and len(u_rows) >= 2) else 0.0)
            prior_diag = lm.detach()
            if bp.z.grad is not None: bp.z.grad.zero_()
            lm.backward()
            if not reg.rows_init or reg.row_m is None or reg.row_m.shape != bp.z.shape:
                reg.row_m, reg.row_v, reg.row_n, reg.rows_init = torch.zeros_like(bp.z), torch.zeros((len(bp.z), 1), device=bp.z.device, dtype=bp.z.dtype), torch.zeros((len(bp.z), 1), device=bp.z.device, dtype=bp.z.dtype), True
            with torch.no_grad():
                gr = (bp.z.grad[u_rows] * float(len(u_rows))).clone(); bp.z.grad.zero_()
                ps, far = (sq_u.detach().sqrt().unsqueeze(1) / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2), sq_u.detach().sqrt().float() > (2.8 * math.sqrt(dim) * sc_u.float())
                if cu.shape[1] <= 8 and bool(far.any()):
                    fi, ii = torch.nonzero(far, as_tuple=False).squeeze(-1), torch.nonzero(~far, as_tuple=False).squeeze(-1)[:64]
                    ii = ii if ii.numel() >= 4 else torch.arange(min(64, len(cu)), device=cu.device)
                    pi_i, dz_u = torch.cat([fi, ii], 0), int(cu.shape[1]); cp, xp = cu[pi_i].detach(), xu.detach().float()[pi_i]
                    cpert = (cp.unsqueeze(1) + (1e-3 * torch.where(cp >= 0.0, 1.0, -1.0)).unsqueeze(2) * torch.eye(dz_u, device=cu.device, dtype=cu.dtype).unsqueeze(0)).reshape(-1, dz_u)
                    csen = (trainer._postprocess(trainer.prior, trainer._generate(trainer.G, cpert, 0.0, noise)).float().reshape(len(pi_i), dz_u, dim) - xp.unsqueeze(1)).norm(dim=-1) / 1e-3
                    dead = csen[: len(fi)] < (0.04 * torch.quantile(csen[len(fi) :], 0.5, dim=0, keepdim=True).clamp_min(1e-6))
                    if bool(dead.any()):
                        gr[fi] = torch.where(dead, (torch.sign(bp.z[u_rows][fi].detach() - cu[ii].detach().mean(0, keepdim=True)).float() * torch.maximum(gr[fi].float().norm(dim=-1, keepdim=True), torch.ones((len(fi), 1), device=cu.device))).to(gr.dtype), gr[fi])
                        ps[fi] = torch.where(dead.any(1, keepdim=True), torch.ones_like(ps[fi]), ps[fi])
                m, v, n = reg.row_m[u_rows] * 0.80 + 0.20 * gr, reg.row_v[u_rows] * 0.80 + 0.20 * gr.square().sum(-1, keepdim=True), reg.row_n[u_rows] + 1.0
                reg.row_m[u_rows], reg.row_v[u_rows], reg.row_n[u_rows] = m, v, n; bc = 1.0 - (0.80 ** n)
                bp.z.index_add_(0, u_rows, ((m / bc) / ((v / bc).sqrt() + 1e-5)) * ps, alpha=-float(recipe.t1_prior_row_lr) * max(0.15, min(1.0, max(reg.res_ema / 0.10, pg))))
        finally:
            for p, fl in zip(trainer.G.parameters(), g_fl): p.requires_grad_(fl)
            if getattr(trainer.opt_d, "record", None) is not None: trainer.opt_d.record._g_probe_cache = None
    zero = loss_gan.new_zeros(())
    return loss_g, loss_gan, prior_diag, None, zero, zero
