"""P5 Adversarial Mathematical & Empirical Audit of Law 4.4 (Generator-Critic Optimizer Asymmetry).

Reproduces every formal theorem, mechanistic causal disentanglement (Hypotheses a-f),
matched 5-arm factorial experiment (Neither / G-only / D-only / Both / Adaptive-Pathology)
across Regimes A, B, B+, C1, C2, and the forensic dissection of `unified_v12`'s
generator-free 2D critic Newton-Schulz fallback reported in:
  reports/four-laws-audit/p5-optimizer-asymmetry.md

Run:
  PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
      scripts/four_laws_audit/p5_asymmetry_experiments.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F

import particlegan
from benchmarks.toy_audit.api_images import (
    ImageCritic,
    ImageGenerator,
    _case as _image_case,
    score_case as score_image_case,
    template_bank,
)
from benchmarks.toy_audit.api_imbalance_nash import (
    HOST_DEFINITIONS as NASH_HOST_DEFS,
    build_models_for_spec as build_nash_models,
    sample_target_for_kind as sample_nash_target,
    score_imbalance_nash_kind,
)
from benchmarks.toy_audit.api_t3_isolation import (
    T3IsolationFixture,
    measure_critic_grad_median,
    sample_target_for_kind as sample_t3_target,
    score_t3_isolation_kind,
)
from lib.toy_models import (
    SimpleMLPDiscriminator,
    SimpleMLPGenerator,
    mode_coverage,
    sample_100gaussians,
)
from particlegan._unified_v4.optimizer import _band_scale, _smoothstep01
from particlegan._unified_v11.critic import _eval_v11_probe_forward as _eval_probe_forward
from particlegan._unified_v11.optimizer import (
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
    zero_pinv_project_nonascent,
)
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer
from particlegan.init import deterministic_orthogonal_


# ---------------------------------------------------------------------------
# 1. Exact Spectral & Stable-Rank Metrics (Matching P0 Definitions Exactly)
# ---------------------------------------------------------------------------

@torch.no_grad()
def spectral_metrics(mat: torch.Tensor) -> Dict[str, Any]:
    """Compute FP64 SVD rank, stable rank, participation ratio, and Roy-Vetterli erank."""
    x = mat.detach().to(device="cpu", dtype=torch.float64)
    r, c = int(x.shape[0]), int(x.shape[1])
    m = min(r, c)
    sv = torch.linalg.svdvals(x)
    s1 = float(sv[0].item()) if m > 0 else 0.0
    s2 = float(sv[1].item()) if m >= 2 else 0.0
    sr = float(sv[-1].item()) if m > 0 else 0.0
    frob_sq = float(sv.square().sum().item())
    frob = math.sqrt(frob_sq)
    if s1 <= 0.0 or frob_sq <= 0.0:
        return {
            "shape": [r, c],
            "frob": 0.0,
            "sigma_1": 0.0,
            "sigma_2": 0.0,
            "sigma2_over_sigma1": 0.0,
            "rank_tol_1e-12": 0,
            "rank_tol_1e-6": 0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "rv_erank": 0.0,
            "top1_energy": 0.0,
            "cond_full": float("inf"),
        }
    rank_12 = int((sv > 1e-12 * s1).sum().item())
    rank_6 = int((sv > 1e-6 * s1).sum().item())
    stable_rank = frob_sq / (s1 * s1)
    s4_sum = float(sv.pow(4).sum().item())
    pr = (frob_sq * frob_sq) / max(s4_sum, 1e-300)
    s_sum = float(sv.sum().item())
    p = sv[sv > 0.0] / s_sum
    rv_erank = math.exp(float(-(p * p.log()).sum().item()))
    top1_energy = (s1 * s1) / frob_sq
    cond_full = (s1 / sr) if sr > 1e-300 else float("inf")
    return {
        "shape": [r, c],
        "frob": frob,
        "sigma_1": s1,
        "sigma_2": s2,
        "sigma2_over_sigma1": s2 / s1,
        "rank_tol_1e-12": rank_12,
        "rank_tol_1e-6": rank_6,
        "stable_rank": stable_rank,
        "participation_ratio": pr,
        "rv_erank": rv_erank,
        "top1_energy": top1_energy,
        "cond_full": cond_full,
    }


@torch.no_grad()
def fast_stable_rank_power(mat_canon: torch.Tensor, iters: int = 6) -> Tuple[float, float]:
    """Scale-invariant zero-SVD estimate of (stable_rank, participation_ratio) for m <= n matrix.

    Uses deterministic power iteration on the Gram matrix ``G G^T`` after max-abs normalization
    so both outputs are strictly degree-0 scale-invariant: ``f(c * M) == f(M)`` for all ``c > 0``.
    Uses a single packed scalar sync per matrix.
    """
    x = mat_canon.detach().float()
    g_max = x.abs().amax()
    safe_max = torch.where(torch.isfinite(g_max) & (g_max > 1e-38), g_max, g_max.new_ones(()))
    xu = torch.where(torch.isfinite(x), x / safe_max, torch.zeros_like(x))
    frob_sq = xu.square().sum()
    gram = xu @ xu.T
    gram_f2 = gram.square().sum()

    # Deterministic Golden-ratio initial vector (zero RNG, works on any width m)
    m = int(xu.shape[0])
    idx = torch.arange(1, m + 1, device=xu.device, dtype=torch.float32)
    v = torch.remainder(idx * 0.61803398875 + 0.25, 1.0) - 0.5
    v = (v / v.norm().clamp_min(1e-12)).unsqueeze(1)
    for _ in range(iters):
        gv = gram @ v
        v = gv / gv.norm().clamp_min(1e-30)
    s1_sq = (v.T @ (gram @ v)).reshape(())
    g_max_f, frob_sq_f, gram_f2_f, s1_sq_f = torch.stack((g_max, frob_sq, gram_f2, s1_sq)).tolist()
    if (not math.isfinite(g_max_f)) or g_max_f <= 1e-38 or frob_sq_f <= 1e-38:
        return 1.0, 1.0
    pr = (frob_sq_f * frob_sq_f) / max(gram_f2_f, 1e-38)
    s_rk = frob_sq_f / max(s1_sq_f, 1e-38)
    s_rk = max(1.0, min(float(m), s_rk))
    pr = max(1.0, min(float(m), pr))
    return s_rk, pr


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[len(s) // 2])


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


@torch.no_grad()
def summarize_module_grad_spectra(mod: nn.Module, min_dim: int = 8) -> Dict[str, Any]:
    """Summarize 2D hidden weight gradient spectra for a module."""
    layers: List[Dict[str, Any]] = []
    for name, p in mod.named_parameters():
        if p.ndim == 2 and min(p.shape) >= min_dim and p.grad is not None:
            layers.append({"name": name, **spectral_metrics(p.grad)})
    if not layers:
        return {
            "num_layers": 0,
            "median_stable_rank": 0.0,
            "min_stable_rank": 0.0,
            "median_pr": 0.0,
            "median_rv_erank": 0.0,
            "median_s2_s1": 0.0,
            "norm_skew": 1.0,
            "layers": [],
        }
    frobs = [x["frob"] for x in layers if x["frob"] > 0.0]
    skew = (max(frobs) / max(min(frobs), 1e-38)) if frobs else 1.0
    return {
        "num_layers": len(layers),
        "median_stable_rank": _median([x["stable_rank"] for x in layers]),
        "min_stable_rank": min(x["stable_rank"] for x in layers),
        "median_pr": _median([x["participation_ratio"] for x in layers]),
        "median_rv_erank": _median([x["rv_erank"] for x in layers]),
        "median_s2_s1": _median([x["sigma2_over_sigma1"] for x in layers]),
        "norm_skew": skew,
        "layers": layers,
    }


def measure_input_grad_norm(D: nn.Module, x: torch.Tensor) -> float:
    """Measure median ||grad_x D(x)||_2 on batch x."""
    xr = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = D(xr)
        g = torch.autograd.grad(out.sum(), xr, create_graph=False)[0]
    norms = g.reshape(xr.shape[0], -1).norm(dim=1)
    if not bool(torch.isfinite(norms).all()):
        return float("nan")
    return float(norms.median().item())


# ---------------------------------------------------------------------------
# 2. Synthetic Multimodal Distributions & Evaluators
# ---------------------------------------------------------------------------

def sample_ring(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    *,
    n_modes: int = 8,
    radius: float = 2.5,
    sigma: float = 0.08,
) -> torch.Tensor:
    k = torch.randint(0, n_modes, (batch_size,), device=device, generator=generator)
    ang = (2.0 * math.pi / float(n_modes)) * k.float()
    centers = torch.stack((radius * ang.cos(), radius * ang.sin()), dim=1)
    return centers + torch.randn(batch_size, 2, device=device, generator=generator) * sigma


@torch.no_grad()
def evaluate_ring(
    G: nn.Module,
    prior: Optional[nn.Module],
    device: torch.device,
    *,
    n_modes: int = 8,
    radius: float = 2.5,
    sigma: float = 0.08,
    n_eval: int = 4096,
    z_dim: int = 2,
) -> Dict[str, Any]:
    was_train = G.training
    G.eval()
    g_ev = torch.Generator(device=device).manual_seed(999)
    if prior is None:
        z = torch.randn(n_eval, z_dim, device=device, generator=g_ev)
    else:
        z, _ = prior.sample(n_eval, generator=g_ev)
    fake = G(z)
    G.train(was_train)
    finite = bool(torch.isfinite(fake).all().item())
    if not finite:
        return {"finite": False, "modes": 0, "hq": 0.0, "mass_tv": 1.0}
    ang = (2.0 * math.pi / float(n_modes)) * torch.arange(n_modes, device=device, dtype=torch.float32)
    centers = torch.stack((radius * ang.cos(), radius * ang.sin()), dim=1)
    dists = torch.cdist(fake, centers)
    mind, idx = dists.min(dim=1)
    hq_mask = mind <= (3.5 * sigma)
    hq = float(hq_mask.float().mean().item())
    counts_hq = torch.bincount(idx[hq_mask], minlength=n_modes)
    min_cnt = max(10, n_eval // (n_modes * 8))
    modes = int((counts_hq >= min_cnt).sum().item())
    mass = torch.bincount(idx, minlength=n_modes).float() / float(n_eval)
    mass_tv = float(0.5 * (mass - (1.0 / float(n_modes))).abs().sum().item())
    return {"finite": True, "modes": modes, "hq": hq, "mass_tv": mass_tv}


def born_infeld_slope_penalty_2d(
    D: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    beta: float = 2.0,
    kappa: float = 1.0,
    gauge_weight: float = 0.05,
) -> Tuple[torch.Tensor, float]:
    """Exact 1st-order fwAD Born-Infeld spatial slope penalty + zero-mean logit gauge in R^2."""
    x = torch.cat((x_real.detach(), x_fake.detach()), dim=0)
    e1 = torch.zeros_like(x)
    e1[:, 0] = 1.0
    e2 = torch.zeros_like(x)
    e2[:, 1] = 1.0
    with fwAD.dual_level():
        y1 = D(fwAD.make_dual(x, e1))
        primal, j1 = fwAD.unpack_dual(y1)
        y2 = D(fwAD.make_dual(x, e2))
        _, j2 = fwAD.unpack_dual(y2)
    gn_sq = j1.square() + j2.square()
    bi = beta * (torch.sqrt(1.0 + gn_sq / (kappa * kappa)) - 1.0).mean()
    gauge = gauge_weight * primal.mean().square()
    return bi + gauge, float(gn_sq.sqrt().median().item())


# ---------------------------------------------------------------------------
# 3. Controlled Matched 5-Arm Optimizers (ParticleGAN & Pure-Adversarial)
# ---------------------------------------------------------------------------

class SymmetricLaw4Optimizer:
    """Strictly role-symmetric degree-0 equilibrated Law-4 optimizer for causal 5-arm trials.

    Applies identical update physics to G and D (zero hardcoded role branches), varying ONLY
    ``mode`` in ``{"frob", "ns", "adaptive", "sgd"}`` and ``frob_scale`` in ``{"sqrt_m", "spec", "aspect"}``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        lr: float = 0.01,
        mode: str = "frob",
        frob_scale: str = "sqrt_m",
        smoothing: float = 0.0061,
        tau_srk: float = 1.30,
        b1: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.mode = str(mode)
        self.frob_scale = str(frob_scale)
        self.smoothing = float(smoothing)
        self.tau_srk = float(tau_srk)
        self.b1 = float(b1)
        self.peak: Dict[int, float] = {}
        self.m_buf: Dict[int, torch.Tensor] = {}
        self.step_n = 0
        self.ns_calls = 0
        self.total_2d_calls = 0

    @torch.no_grad()
    def step(self) -> None:
        self.step_n += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.float()
            if self.mode == "sgd":
                if bool(torch.isfinite(g).all()):
                    p.add_(g.to(p.dtype), alpha=-self.lr)
                continue
            g_max = float(g.abs().amax().item())
            if not math.isfinite(g_max) or g_max <= 1e-38:
                continue
            pid = id(p)
            if self.b1 > 0.0:
                if pid not in self.m_buf:
                    self.m_buf[pid] = torch.zeros_like(g)
                self.m_buf[pid].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
                d = self.m_buf[pid] / (1.0 - self.b1 ** self.step_n)
            else:
                d = g
            d_max = float(d.abs().amax().item())
            if not math.isfinite(d_max) or d_max <= 1e-38:
                continue
            d_u = d / d_max
            d_u_norm = float(d_u.norm().item())
            d_norm = d_u_norm * d_max

            if p.ndim >= 2 and min(int(p.shape[0]), int(p.numel() // p.shape[0])) >= 2:
                self.total_2d_calls += 1
                mat_u = d_u.reshape(int(p.shape[0]), -1)
                r_d, c_d = int(mat_u.shape[0]), int(mat_u.shape[1])
                m_dim = min(r_d, c_d)
                aspect = math.sqrt(max(1.0, float(r_d) / float(c_d)))
                sv_rms = d_norm / math.sqrt(float(m_dim))
                pk = max(self.peak.get(pid, 0.0), sv_rms)
                self.peak[pid] = pk
                s_eff = min(self.smoothing, 0.30 * pk) if self.smoothing > 0.0 else 0.0
                needs_t = r_d > c_d
                mc_u = mat_u.T if needs_t else mat_u

                s_rk_est: Optional[float] = None
                if self.mode == "ns":
                    do_ns = True
                elif self.mode == "frob":
                    do_ns = False
                elif self.mode == "adaptive":
                    s_rk_est, _ = fast_stable_rank_power(mc_u, iters=6)
                    do_ns = s_rk_est < self.tau_srk
                else:
                    raise ValueError(f"Unknown mode: {self.mode!r}")

                if do_ns:
                    self.ns_calls += 1
                    mat_in = (mc_u * d_max) if s_eff > 0.0 else mc_u
                    orth = quintic_orthogonalize_2d(mat_in, steps=5, smoothing=s_eff).float()
                    if needs_t:
                        orth = orth.T
                    p.add_(orth.reshape_as(p).to(p.dtype), alpha=-self.lr * aspect)
                else:
                    if self.frob_scale == "sqrt_m":
                        fac = aspect * math.sqrt(float(m_dim))
                    elif self.frob_scale == "spec":
                        if s_rk_est is None:
                            s_rk_est, _ = fast_stable_rank_power(mc_u, iters=6)
                        fac = aspect * math.sqrt(max(1.0, s_rk_est))
                    elif self.frob_scale == "aspect":
                        fac = aspect
                    else:
                        raise ValueError(f"Unknown frob_scale: {self.frob_scale!r}")
                    s_u = (s_eff * math.sqrt(float(m_dim))) / d_max if s_eff > 0.0 else 0.0
                    denom_u = math.hypot(d_u_norm, s_u)
                    if denom_u > 1e-30:
                        p.add_(d_u.reshape_as(p).to(p.dtype), alpha=-self.lr * fac / denom_u)
            else:
                pk = max(self.peak.get(pid, 0.0), d_norm)
                self.peak[pid] = pk
                s_eff = min(self.smoothing, 0.30 * pk) if self.smoothing > 0.0 else 0.0
                s_u = s_eff / d_max if s_eff > 0.0 else 0.0
                denom_u = math.hypot(d_u_norm, s_u)
                if denom_u > 1e-30:
                    p.add_(d_u.to(p.dtype), alpha=-self.lr / denom_u)


class P5ControlledV12Optimizer(UnifiedV12Optimizer):
    """Exact `UnifiedV12Optimizer` subclass isolating 2D NS policy for matched 5-arm factorial runs.

    Holds Laws 1-3, prior row updates, 4D Conv2d Frobenius-MP, `peak_rms`, `stiff_gate`,
    `delta_g_max` secant band, and CountSketch flip control identical to `unified_v12`, while
    allowing clean orthogonal control over whether 2D dense layers (`min(r, c) >= 2`) use:
      - ``"frob"``: 0-NS Frobenius-MP (``sqrt_m`` polar-Frobenius match or ``spec`` spectral match)
      - ``"ns"``: 5-step fused ``Z=[X, Y]`` Quintic Tikhonov NS
      - ``"adaptive"``: Role-agnostic dynamic NS iff scale-invariant ``stable_rank < tau_srk``
      - ``"sgd"``: Unnormalized raw SGD on this player's non-prior layers (Law 4.1 control)
    """

    def __init__(
        self,
        *args: Any,
        ns_mode: str = "ns",
        tau_srk: float = 1.25,
        frob_scale_mode: str = "sqrt_m",
        smooth_cap: float = 0.50,
        p_dec_hot: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.p5_ns_mode = str(ns_mode)
        self.p5_tau_srk = float(tau_srk)
        self.p5_frob_scale_mode = str(frob_scale_mode)
        self.p5_smooth_cap = float(smooth_cap)
        self.p5_p_dec_hot = float(p_dec_hot)
        self.p5_total_2d_calls = 0
        self.p5_ns_2d_calls = 0
        self.p5_skipped_2d_calls = 0
        self.p5_srk_sum = 0.0
        self.p5_srk_count = 0

    def _p5_decide_ns(self, mat_canon: torch.Tensor) -> Tuple[bool, Optional[float]]:
        self.p5_total_2d_calls += 1
        if self.p5_ns_mode == "ns":
            self.p5_ns_2d_calls += 1
            return True, None
        if self.p5_ns_mode == "frob":
            self.p5_skipped_2d_calls += 1
            return False, None
        if self.p5_ns_mode == "adaptive":
            s_rk, _ = fast_stable_rank_power(mat_canon, iters=6)
            self.p5_srk_sum += s_rk
            self.p5_srk_count += 1
            if s_rk < self.p5_tau_srk:
                self.p5_ns_2d_calls += 1
                return True, s_rk
            self.p5_skipped_2d_calls += 1
            return False, s_rk
        raise ValueError(f"Unsupported p5_ns_mode: {self.p5_ns_mode!r}")

    def _p5_frob_factor(self, aspect: float, m_dim: int, mat_canon: torch.Tensor, s_rk: Optional[float]) -> float:
        if self.p5_frob_scale_mode == "sqrt_m":
            return aspect * math.sqrt(float(m_dim))
        if self.p5_frob_scale_mode == "spec":
            if s_rk is None:
                s_rk, _ = fast_stable_rank_power(mat_canon, iters=6)
            return aspect * math.sqrt(max(1.0, s_rk))
        return aspect

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        # Unnormalized raw SGD control for non-prior groups (Law 4.1 impedance contrast)
        if self.p5_ns_mode == "sgd":
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            try:
                rec = self._active_record()
                step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
                for grp in self.param_groups:
                    player = self._player(grp["role"])
                    lr_g = float(grp["lr"]) * step_scale
                    if player == "prior":
                        lr_g *= self._v4_prior_scale
                    for p in grp["params"]:
                        if p.grad is None:
                            continue
                        if player == "prior":
                            rows = self._resolve_unique_sampled_rows(p) if grp.get("sampled_rows_required", True) else torch.arange(len(p), device=p.device)
                            if rows is None or rows.numel() == 0:
                                continue
                            gr = p.grad[rows].float()
                            gn = gr.norm(dim=-1, keepdim=True).clamp_min(1e-36)
                            p.index_add_(0, rows, (gr / gn).to(p.dtype), alpha=-lr_g)
                        else:
                            p.add_(p.grad, alpha=-lr_g)
                self.last_step_gain = 1.0
                self.last_probe_consumed = False
            finally:
                self.abort_pending()
            if hasattr(self, "record") and self.record is not None:
                self.record.record_step(self)
            return loss

        # Otherwise run exact UnifiedV12Optimizer mechanics with controlled 2D NS routing on BOTH G and D!
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.is_complex():
                    raise ValueError(f"{self.family} requires dense real gradients")

        if self.direction_blend and self._protected is not None:
            all_params = self._parameters()
            originals = [p.detach().clone() for p in all_params]
        else:
            all_params = None
            originals = None

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            is_simp = bool(getattr(rec, "_simplicial_step", False)) if rec is not None else False
            is_cond_fiber = bool(getattr(rec, "t1c_cond_initialized", False)) if rec is not None else False
            is_critic_opt = getattr(self, "critic", None) is not None
            p_dec = self.p5_p_dec_hot if s_therm >= 0.5 else 1.0

            origin_damping = 1.0
            if self._origin_norm_t is not None:
                origin_damping = min(1.0, float(self._origin_norm_t.item()) / 0.05)
            elif self._origin_norm is not None and self._origin_norm < 0.05:
                origin_damping = min(1.0, self._origin_norm / 0.05)

            if not is_critic_opt:
                self._update_period2_flip_feedback(s_therm=s_therm)

            ema_fg = float(self._ema_func_gain)
            stiff_gate_raw = self.current_stiff_gate()
            pre_gain_base = 1.0 if ema_fg >= 0.75 else ema_fg
            v4_g_scale_f = float(self._v4_g_scale.item()) if isinstance(self._v4_g_scale, torch.Tensor) else float(self._v4_g_scale)
            v4_wake_f = float(self._v4_wakeup_boost.item()) if isinstance(self._v4_wakeup_boost, torch.Tensor) else float(self._v4_wakeup_boost)
            pre_g_float = min(1.0, pre_gain_base * v4_wake_f)

            gen_probe_active = (
                (not is_critic_opt)
                and rec is not None
                and getattr(rec, "_g_probe_cache", None) is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}
            coh_vals: List[torch.Tensor] = []

            for grp in self.param_groups:
                role = grp["role"]
                player = self._player(role)
                stiff_gate = stiff_gate_raw if player == "generator" else 0.0
                b1, b2 = grp["betas"]
                eps = float(grp["eps"])
                s_base = self.smoothing * (1.0 - stiff_gate)
                s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0

                if player == "prior":
                    b1_eff = max(float(b1), 0.75) if self._v4_prior_scale < 1.0 else float(b1)
                elif stiff_gate > 0.0:
                    b1_eff = max(float(b1), 0.75 * stiff_gate)
                else:
                    b1_eff = float(b1)
                b2_eff = min(max(float(b2), 0.95), 0.999)

                if player == "generator":
                    lr_eff = float(grp["lr"]) * step_scale * v4_g_scale_f * pre_g_float
                elif player == "critic":
                    lr_eff = float(grp["lr"]) * step_scale
                else:
                    lr_eff = float(grp["lr"]) * step_scale * self._v4_prior_scale

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t

                    if player == "prior":
                        if lr_eff <= 0.0:
                            continue
                        sampled_req = bool(grp.get("sampled_rows_required", True))
                        r_scales = self._row_scales.get(p)
                        if sampled_req and r_scales is None:
                            raw_rows = self._sampled_rows[p]
                            if raw_rows.numel() == 0:
                                continue
                            sorted_r, _ = torch.sort(raw_rows)
                            n_raw = int(sorted_r.shape[0])
                            is_first = torch.ones(n_raw, device=p.device, dtype=torch.bool)
                            if n_raw > 1:
                                is_first[1:] = sorted_r[1:] != sorted_r[:-1]
                            n_unique_t = is_first.sum()
                            w_u_col = is_first.float().unsqueeze(1)
                            gr = grad[sorted_r].float()
                            fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
                            gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
                            if b1_eff > 0.0:
                                if "m_prior" not in st:
                                    st["m_prior"] = torch.zeros_like(p, dtype=torch.float32)
                                m_prev = st["m_prior"][sorted_r]
                                m_new = torch.where(fin_row, m_prev.mul(b1_eff).add(gr_safe, alpha=1.0 - b1_eff), m_prev)
                                st["m_prior"].index_add_(0, sorted_r, (m_new - m_prev) * w_u_col)
                                dir_r = m_new
                            else:
                                dir_r = gr_safe
                            g_norm = gr_safe.norm(dim=-1, keepdim=True)
                            d_norm = dir_r.norm(dim=-1, keepdim=True)
                            active_r = is_first.unsqueeze(1) & fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)
                            if s_prior_base > 0.0:
                                s_p_row = torch.where(
                                    n_unique_t > 128,
                                    (s_prior_base * 128.0) / n_unique_t.clamp_min(1).float(),
                                    d_norm.new_full((), s_prior_base),
                                )
                                denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
                            else:
                                denom_r = d_norm.clamp_min(1e-36)
                            unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
                            p.index_add_(0, sorted_r, unit_r.to(p.dtype), alpha=-lr_eff)
                            continue
                        rows = (
                            self._resolve_unique_sampled_rows(p)
                            if sampled_req
                            else torch.arange(len(p), device=p.device, dtype=torch.long)
                        )
                        if rows is None or rows.numel() == 0:
                            continue
                        gr = grad[rows].float()
                        fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
                        gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
                        if b1_eff > 0.0:
                            if "m_prior" not in st:
                                st["m_prior"] = torch.zeros_like(p, dtype=torch.float32)
                            m_r = st["m_prior"][rows]
                            m_new = torch.where(fin_row, m_r.mul(b1_eff).add(gr_safe, alpha=1.0 - b1_eff), m_r)
                            st["m_prior"][rows] = m_new
                            dir_r = m_new
                        else:
                            dir_r = gr_safe
                        g_norm = gr_safe.norm(dim=-1, keepdim=True)
                        d_norm = dir_r.norm(dim=-1, keepdim=True)
                        active_r = fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)
                        if s_prior_base > 0.0:
                            s_cap = s_prior_base * (128.0 / float(len(rows))) if len(rows) > 128 else s_prior_base
                            denom_r = torch.hypot(d_norm, d_norm.new_full((), s_cap)).clamp_min(1e-36)
                        else:
                            denom_r = d_norm.clamp_min(1e-36)
                        unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
                        if r_scales is not None and len(r_scales) == len(rows):
                            unit_r = unit_r * r_scales.unsqueeze(1)
                        p.index_add_(0, rows, unit_r.to(p.dtype), alpha=-lr_eff)
                        continue

                    rate_p = lr_eff * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids)
                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax()
                    g_active = finite & (g_max > 1e-38)

                    is_conv = p.ndim == 4
                    r_d = c_d = m_dim = 1
                    aspect_factor = 1.0
                    if is_conv:
                        factor, sv_denom_sqrt = self._cached_conv_metrics(p)
                    elif p.ndim >= 2:
                        r_d, c_d, m_dim, aspect_factor, sv_denom_sqrt = self._cached_dense_metrics(p)
                        factor = aspect_factor
                    else:
                        factor, sv_denom_sqrt = 1.0, 1.0

                    safe_scale = torch.where(g_active, g_max, torch.ones_like(g_max))
                    g_unit_scaled = torch.where(g_active, gf_safe / safe_scale, torch.zeros_like(gf_safe))
                    scaled_norm = g_unit_scaled.norm()
                    cur_sv_rms_t = (scaled_norm * safe_scale) / sv_denom_sqrt
                    if "peak_rms" not in st:
                        st["peak_rms"] = gf_safe.new_zeros(())
                    peak_prev_t = st["peak_rms"]
                    cur_valid = finite & torch.isfinite(cur_sv_rms_t)
                    peak_new_t = torch.maximum(peak_prev_t * p_dec, torch.where(cur_valid, cur_sv_rms_t, peak_prev_t.new_zeros(())))
                    st["peak_rms"].copy_(torch.where(torch.isfinite(peak_new_t), peak_new_t, peak_prev_t))

                    has_smooth = s_base > 0.0
                    s_eff_t = (
                        torch.minimum(gf_safe.new_full((), s_base), peak_new_t * self.p5_smooth_cap)
                        if has_smooth
                        else gf_safe.new_zeros(())
                    )

                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    use_moments = (b1_eff > 0.0) or (stiff_gate > 0.0) or is_scalar_1d
                    if use_moments:
                        b1_use = max(b1_eff, 0.80) if is_scalar_1d else b1_eff
                        if "m_unit" not in st:
                            st["m_unit"] = torch.zeros_like(p, dtype=torch.float32)
                            st["m_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                            st["v_rms"] = torch.zeros_like(p, dtype=torch.float32)
                            st["v_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        m_prev_sc = st["m_scale"]
                        v_prev_sc = st["v_scale"]
                        sc = torch.maximum(torch.maximum(m_prev_sc, v_prev_sc), g_max).clamp_min(1e-38)
                        m_rescaled = st["m_unit"] * (m_prev_sc / sc)
                        v_rescaled = st["v_rms"] * (v_prev_sc / sc)
                        g_scaled = gf_safe / sc
                        m_next = torch.where(finite, m_rescaled.mul(b1_use).add(g_scaled, alpha=1.0 - b1_use), m_rescaled)
                        v_next = torch.where(
                            finite,
                            (b2_eff * v_rescaled.square() + (1.0 - b2_eff) * g_scaled.square()).sqrt(),
                            v_rescaled,
                        )
                        new_m_sc = m_next.abs().amax()
                        safe_m_sc = torch.where(new_m_sc > 1e-38, new_m_sc, torch.ones_like(new_m_sc))
                        st["m_unit"].copy_(torch.where(new_m_sc > 1e-38, m_next / safe_m_sc, torch.zeros_like(m_next)))
                        st["m_scale"].copy_(torch.where(new_m_sc > 1e-38, new_m_sc, torch.zeros_like(new_m_sc)) * sc)
                        new_v_sc = v_next.abs().amax()
                        safe_v_sc = torch.where(new_v_sc > 1e-38, new_v_sc, torch.ones_like(new_v_sc))
                        st["v_rms"].copy_(torch.where(new_v_sc > 1e-38, v_next / safe_v_sc, torch.zeros_like(v_next)))
                        st["v_scale"].copy_(torch.where(new_v_sc > 1e-38, new_v_sc, torch.zeros_like(new_v_sc)) * sc)
                        m_hat = m_next / max(1.0 - (b1_use ** step_t), 1e-8)
                        v_hat = v_next / max(math.sqrt(1.0 - (b2_eff ** step_t)), 1e-8)
                        m_norm = m_hat.norm()
                        v_norm = v_hat.norm()
                        raw_coh = torch.where(
                            g_active & (v_norm > 1e-30) & (m_norm > 1e-30),
                            (m_norm / v_norm.clamp_min(1e-30)).clamp(0.0, 1.0),
                            torch.zeros_like(m_norm),
                        )
                        if is_scalar_1d:
                            coh = raw_coh.clamp(0.35, 1.0)
                        elif stiff_gate > 0.0:
                            coh = (1.0 - stiff_gate) + stiff_gate * raw_coh
                        else:
                            coh = torch.ones_like(m_norm)
                        dir_unit = m_hat
                        dir_norm = m_norm
                        dir_scale = sc
                    else:
                        coh = gf_safe.new_ones(())
                        dir_unit = g_unit_scaled
                        dir_norm = scaled_norm
                        dir_scale = safe_scale

                    if rate_p <= 0.0:
                        continue
                    if track_secant:
                        coh_vals.append(coh)

                    if (not is_conv) and p.ndim >= 2 and m_dim >= 2:
                        mat_u = dir_unit.reshape(r_d, c_d)
                        needs_t = r_d > c_d
                        mc_u = mat_u.T if needs_t else mat_u
                        do_ns, s_rk_est = self._p5_decide_ns(mc_u)
                        if do_ns:
                            raw_dir = (dir_unit * dir_scale) if has_smooth else dir_unit
                            direc = torch.where(g_active, raw_dir, torch.zeros_like(dir_unit))
                            mat_d = direc.reshape(r_d, c_d)
                            mat_canon = mat_d.T if needs_t else mat_d
                            step_mult = (-rate_p * aspect_factor) * coh
                            m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                            pending_dense.setdefault(m_key, []).append(
                                (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                            )
                            continue
                        fac_f = self._p5_frob_factor(aspect_factor, m_dim, mc_u, s_rk_est)
                        if has_smooth:
                            s_scaled = (s_eff_t * sv_denom_sqrt) / dir_scale.clamp_min(1e-38)
                            denom = torch.hypot(dir_norm, s_scaled).clamp_min(1e-30)
                        else:
                            denom = dir_norm.clamp_min(1e-30)
                        unit_d = torch.where(g_active & (dir_norm > 1e-30), dir_unit / denom, torch.zeros_like(dir_unit))
                        delta_d = (unit_d * ((-rate_p * fac_f) * coh)).to(p.dtype)
                        p.add_(delta_d)
                        if track_secant:
                            applied_deltas.append((p, delta_d))
                        continue

                    if has_smooth:
                        s_scaled = (s_eff_t * sv_denom_sqrt) / dir_scale.clamp_min(1e-38)
                        denom = torch.hypot(dir_norm, s_scaled).clamp_min(1e-30)
                    else:
                        eps_sc = (eps / dir_scale.clamp_min(1e-38)) if p.ndim == 1 else 0.0
                        denom = (dir_norm + eps_sc).clamp_min(1e-30)
                    unit_g = torch.where(g_active & (dir_norm > 1e-30), dir_unit / denom, torch.zeros_like(dir_unit))
                    delta_g = (unit_g * ((-rate_p * factor) * coh)).to(p.dtype)
                    p.add_(delta_g)
                    if track_secant:
                        applied_deltas.append((p, delta_g))

            for (_, _, has_s_b), bucket in pending_dense.items():
                self.ns_2d_activations += len(bucket)
                self.ns_5step_calls += len(bucket)
                if is_critic_opt:
                    self.critic_ns_calls += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                    p_i.add_(delta_i)
                    if trk_i:
                        applied_deltas.append((p_i, delta_i))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        if trk_i:
                            applied_deltas.append((p_i, delta_i))

            # Generator-only secant trust band (exact match to UnifiedV12Optimizer)
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if gen_probe_active and applied_deltas:
                probe_cache = getattr(rec, "_g_probe_cache", None)
                rec._g_probe_cache = None
                if probe_cache is not None:
                    g_mod, p_args, p_kwargs, y_pre = probe_cache
                    y_post = _eval_probe_forward(g_mod, p_args, p_kwargs, rec)
                    if y_post is not None and y_post.shape == y_pre.shape:
                        probe_ran = True
                        out_dim = max(1, int(y_pre[0].numel()))
                        delta_rms = float((y_post - y_pre).square().mean().sqrt().item())
                        mean_coh = float(torch.stack(coh_vals).mean().item()) if coh_vals else 1.0
                        data_scale = float(getattr(rec, "ema_data_scale", 1.0))
                        coord_scale = max(0.05, data_scale / math.sqrt(float(out_dim)))
                        delta_rel = delta_rms / coord_scale
                        push_res = float(getattr(rec, "ema_push_res", 0.0))
                        delta_g_max = (0.045 + 0.16 * min(1.5, push_res)) * step_scale
                        if (not is_simp) and push_res > 0.18 and s_therm > 0.10:
                            off_gate = _smoothstep01((push_res - 0.18) / 0.25)
                            delta_g_min = 0.012 * off_gate * step_scale
                        else:
                            delta_g_min = 0.0
                        alpha_gain = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=48.0)
                        coh_eff = max(0.05, mean_coh)
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_g_float * coh_eff
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= 0.012
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, d_i in applied_deltas:
                        p_i.add_(d_i, alpha=adj)
                if probe_ran:
                    if alpha_gain < 1.0:
                        target_fg = max(0.05, min(1.0, pre_g_float * alpha_gain))
                        self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                    elif step_active_probe and free_ratio > 1.5:
                        target_fg = max(0.05, min(1.0, free_ratio))
                        if target_fg > self._ema_func_gain:
                            self._ema_func_gain = float(0.999 * self._ema_func_gain + 0.001 * target_fg)
                if rec is not None:
                    g_stiff_post = self.current_stiff_gate()
                    rec._g_func_gain = float((1.0 - g_stiff_post) * self._ema_func_gain + g_stiff_post)
            elif rec is not None and not is_critic_opt:
                rec._g_probe_cache = None

            if self.direction_blend and originals is not None and self._protected is not None and all_params is not None:
                normals = self._protected.double()
                displacement = torch.cat([(p - old).reshape(-1) for p, old in zip(all_params, originals)])
                disp64 = displacement.double()
                before = normals @ disp64
                if bool((before > 0.0).any()):
                    a_nrm = normals.norm(dim=1)
                    a_unit = normals[a_nrm > 0] / a_nrm[a_nrm > 0, None]
                    possible = int(a_unit.shape[0]) == 0 or float(a_unit.mean(dim=0).norm().item()) > 1e-12
                    direction = zero_pinv_project_nonascent(displacement, normals) if possible else torch.zeros_like(displacement)
                    offset = 0
                    for p, old in zip(all_params, originals):
                        size = p.numel()
                        p.copy_(old + direction[offset : offset + size].view_as(p) if possible else old)
                        offset += size

            self.last_step_gain = float(alpha_gain)
            self.last_probe_consumed = bool(probe_ran)
            if probe_ran:
                self.probe_consumptions += 1
        finally:
            self.abort_pending()

        if hasattr(self, "record") and self.record is not None:
            self.record.record_step(self)
        return loss


def _swap_trainer_optimizers(
    trainer: Any,
    *,
    g_ns_mode: str,
    d_ns_mode: str,
    tau_srk: float = 1.25,
    frob_scale_mode: str = "sqrt_m",
    d_smooth_cap: float = 0.50,
    d_p_dec_hot: float = 1.0,
) -> Tuple[P5ControlledV12Optimizer, P5ControlledV12Optimizer]:
    """Replace a freshly built `unified_v12` GANTrainer's optimizers with P5ControlledV12Optimizer."""
    old_g = trainer.opt_g
    old_d = trainer.opt_d

    def _clone_groups(opt: Any) -> List[Dict[str, Any]]:
        groups = []
        for grp in opt.param_groups:
            g_copy = {k: v for k, v in grp.items() if k != "params"}
            g_copy["params"] = list(grp["params"])
            groups.append(g_copy)
        return groups

    new_g = P5ControlledV12Optimizer(
        _clone_groups(old_g),
        lr=float(old_g.defaults["lr"]),
        betas=tuple(old_g.defaults["betas"]),
        eps=float(old_g.defaults["eps"]),
        smoothing=float(old_g.smoothing),
        direction_blend=bool(old_g.direction_blend),
        ns_mode=g_ns_mode,
        tau_srk=tau_srk,
        frob_scale_mode=frob_scale_mode,
        smooth_cap=0.50,
        p_dec_hot=1.0,
    )
    new_d = P5ControlledV12Optimizer(
        _clone_groups(old_d),
        lr=float(old_d.defaults["lr"]),
        betas=tuple(old_d.defaults["betas"]),
        eps=float(old_d.defaults["eps"]),
        smoothing=float(old_d.smoothing),
        direction_blend=bool(old_d.direction_blend),
        generator_free=bool(getattr(old_d, "_generator_free", False)),
        ns_mode=d_ns_mode,
        tau_srk=tau_srk,
        frob_scale_mode=frob_scale_mode,
        smooth_cap=d_smooth_cap,
        p_dec_hot=d_p_dec_hot,
    )
    # Preserve every bound record / critic / conv layout / gen_param_ids link
    for attr in ("_unified_v11_record", "_unified_record", "_t1c_record", "_t2_record", "_gen_param_ids", "_conv_layouts"):
        if hasattr(old_g, attr):
            setattr(new_g, attr, getattr(old_g, attr))
        if hasattr(old_d, attr):
            setattr(new_d, attr, getattr(old_d, attr))
    for attr in ("critic", "ema_critic", "record", "latent_damping", "latent_history", "continuous_controller"):
        if hasattr(old_d, attr):
            setattr(new_d, attr, getattr(old_d, attr))
        if hasattr(old_g, attr):
            setattr(new_g, attr, getattr(old_g, attr))

    trainer.opt_g = new_g
    trainer.opt_d = new_d
    if hasattr(trainer, "policy") and trainer.policy is not None:
        pol = trainer.policy
        pol.opt_g = new_g
        pol.opt_d = new_d
        pol.table_optimizer = new_g
        pol.optimizers = [new_g, new_d]
        if getattr(pol, "birth_death", None) is not None and hasattr(pol.birth_death, "rows"):
            pol.birth_death.rows.optimizer = new_g
        if getattr(pol, "lr_settle", None) is not None and hasattr(pol.lr_settle, "optimizers"):
            pol.lr_settle.optimizers = pol.optimizers
        if getattr(pol, "penalty", None) is not None and hasattr(pol.penalty, "optimizer"):
            pol.penalty.optimizer = new_d
    if hasattr(trainer, "penalty") and trainer.penalty is not None:
        if hasattr(trainer.penalty, "optimizer"):
            trainer.penalty.optimizer = new_d
    return new_g, new_d


# ---------------------------------------------------------------------------
# 4. Experiment 1: Analytical & Step-0 Mechanistic Disentanglement (Hypotheses a, b, d, e)
# ---------------------------------------------------------------------------

def run_experiment_1(device: torch.device) -> Dict[str, Any]:
    """Verify Theorems 1-3 and Hypotheses (a), (b), (d), (e) quantitatively."""
    out: Dict[str, Any] = {}

    # 1a. Theorem 1 (Hypothesis a): Scalar potential (d_out=1) rank-1 linear bottleneck vs vector pushforward
    # Run in FP64 so exact algebraic rank(G_W_l) <= min(d_out, B, W) is verified to machine precision (~1e-15).
    class _PlainMLP(nn.Module):
        def __init__(self, in_dim: int, width: int, depth: int, out_dim: int, linear_only: bool) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            d = in_dim
            for _ in range(depth):
                layers.append(nn.Linear(d, width))
                if not linear_only:
                    layers.append(nn.LeakyReLU(0.2))
                d = width
            layers.append(nn.Linear(d, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    rng_a = torch.Generator(device="cpu").manual_seed(0)
    x_in = torch.randn(256, 32, generator=rng_a, dtype=torch.float64)
    t1_rows: Dict[str, Any] = {}
    for linear_only in (True, False):
        mode_tag = "linear" if linear_only else "leaky_relu"
        t1_rows[mode_tag] = {}
        for d_out in (1, 2, 8, 32):
            mod = _PlainMLP(32, 64, 4, d_out, linear_only=linear_only)
            deterministic_orthogonal_(mod, seed=0)
            mod = mod.double()
            cot = torch.randn(256, d_out, generator=torch.Generator(device="cpu").manual_seed(7), dtype=torch.float64)
            mod.zero_grad(set_to_none=True)
            y = mod(x_in)
            (y * cot).mean().backward()
            mid_layers = [p.grad for _, p in mod.named_parameters() if p.ndim == 2 and p.shape == (64, 64)]
            sm_list = [spectral_metrics(g) for g in mid_layers]
            t1_rows[mode_tag][f"d_out_{d_out}"] = {
                "median_rank_tol_1e-12": _median([float(m["rank_tol_1e-12"]) for m in sm_list]),
                "median_rank_tol_1e-6": _median([float(m["rank_tol_1e-6"]) for m in sm_list]),
                "median_stable_rank": _median([m["stable_rank"] for m in sm_list]),
                "median_rv_erank": _median([m["rv_erank"] for m in sm_list]),
            }

    # Also measure Law-2 fwAD tangent penalty effect on d_out=1 scalar critic gradient rank (in FP64)!
    d_lin = _PlainMLP(2, 64, 4, 1, linear_only=True)
    d_nlin = _PlainMLP(2, 64, 4, 1, linear_only=False)
    deterministic_orthogonal_(d_lin, seed=0)
    deterministic_orthogonal_(d_nlin, seed=0)
    d_lin, d_nlin = d_lin.double(), d_nlin.double()
    xr2 = sample_ring(128, torch.device("cpu"), torch.Generator(device="cpu").manual_seed(0), n_modes=8).double()
    xf2 = (torch.randn(128, 2, generator=torch.Generator(device="cpu").manual_seed(1)) * 2.0).double()
    law2_rank_effect: Dict[str, Any] = {}
    for tag, d_mod in (("linear_d_out_1", d_lin), ("leaky_relu_d_out_1", d_nlin)):
        d_mod.zero_grad(set_to_none=True)
        F.softplus(-(d_mod(xr2).squeeze(-1) - d_mod(xf2).squeeze(-1))).mean().backward()
        adv_only = summarize_module_grad_spectra(d_mod, min_dim=16)
        d_mod.zero_grad(set_to_none=True)
        adv_l = F.softplus(-(d_mod(xr2).squeeze(-1) - d_mod(xf2).squeeze(-1))).mean()
        pen_l, _ = born_infeld_slope_penalty_2d(lambda t: d_mod(t).squeeze(-1), xr2, xf2, beta=12.0, kappa=1.0)
        (adv_l + pen_l).backward()
        adv_plus_bi = summarize_module_grad_spectra(d_mod, min_dim=16)
        law2_rank_effect[tag] = {
            "adv_only": {k: adv_only[k] for k in ("median_stable_rank", "median_rv_erank", "median_s2_s1")},
            "adv_only_rank_1e-12": _median([float(x["rank_tol_1e-12"]) for x in adv_only["layers"]]),
            "adv_plus_law2_bi": {k: adv_plus_bi[k] for k in ("median_stable_rank", "median_rv_erank", "median_s2_s1")},
            "adv_plus_law2_rank_1e-12": _median([float(x["rank_tol_1e-12"]) for x in adv_plus_bi["layers"]]),
        }
    out["exp1a_scalar_vs_vector_rank"] = {"mlp_sweep": t1_rows, "law2_tangent_rank_lift": law2_rank_effect}

    # 1b. Theorem 2 (Hypotheses a+b+d): Contrastive Mean-Cone Cancellation & Architectural Spectra
    g16 = SimpleMLPGenerator(2, 128, 16, 2)
    g3 = SimpleMLPGenerator(2, 128, 3, 2)
    d16_f0 = SimpleMLPDiscriminator(2, 128, 16, fourier=0)
    d16_f2 = SimpleMLPDiscriminator(2, 128, 16, fourier=2)
    d3_f0 = SimpleMLPDiscriminator(2, 128, 3, fourier=0)
    d3_f2 = SimpleMLPDiscriminator(2, 128, 3, fourier=2)
    for mod in (g16, g3, d16_f0, d16_f2, d3_f0, d3_f2):
        deterministic_orthogonal_(mod, seed=0)
        mod.to(device)
    rng_b = torch.Generator(device=device).manual_seed(0)
    real100 = sample_100gaussians(256, device, generator=rng_b)
    z2 = torch.randn(256, 2, device=device, generator=rng_b)

    # G16 adversarial against D3_f2
    g16.zero_grad(set_to_none=True)
    (-d3_f2(g16(z2)).mean()).backward()
    g16_spec = summarize_module_grad_spectra(g16, min_dim=16)

    # D16_f0 single-sign (-D(real).mean()) vs contrastive (-(D(real) - D(fake)).mean())
    d16_f0.zero_grad(set_to_none=True)
    (-d16_f0(real100).mean()).backward()
    d16_single_spec = summarize_module_grad_spectra(d16_f0, min_dim=16)

    fake3 = g3(z2).detach()
    d16_f0.zero_grad(set_to_none=True)
    (-(d16_f0(real100).mean() - d16_f0(fake3).mean())).backward()
    d16_diff_f0_spec = summarize_module_grad_spectra(d16_f0, min_dim=16)

    d16_f2.zero_grad(set_to_none=True)
    (-(d16_f2(real100).mean() - d16_f2(fake3).mean())).backward()
    d16_diff_f2_spec = summarize_module_grad_spectra(d16_f2, min_dim=16)

    # D3_f2 contrastive & G3 adversarial
    d3_f2.zero_grad(set_to_none=True)
    (-(d3_f2(real100).mean() - d3_f2(fake3).mean())).backward()
    d3_diff_spec = summarize_module_grad_spectra(d3_f2, min_dim=16)

    g3.zero_grad(set_to_none=True)
    (-d3_f2(g3(z2)).mean()).backward()
    g3_spec = summarize_module_grad_spectra(g3, min_dim=16)

    # Also measure Conv2d img_bars4 G & D spectra
    bars_case = _image_case("image-develop-img_bars4-source-transpose12")
    img_g = ImageGenerator(bars_case)
    img_d = ImageCritic(bars_case)
    deterministic_orthogonal_(img_g, seed=0)
    deterministic_orthogonal_(img_d, seed=1)
    img_g, img_d = img_g.to(device), img_d.to(device)
    bank4 = template_bank("bars4", device=device)
    ids4 = torch.randint(4, (64,), device=device, generator=torch.Generator(device=device).manual_seed(0))
    real_img = bank4[ids4]
    z_img = torch.randn(64, bars_case["z_dim"], device=device, generator=torch.Generator(device=device).manual_seed(1))
    img_d.zero_grad(set_to_none=True)
    (-(img_d(real_img).mean() - img_d(img_g(z_img).detach()).mean())).backward()
    img_g.zero_grad(set_to_none=True)
    (-img_d(img_g(z_img)).mean()).backward()
    conv_spectra: Dict[str, Any] = {}
    for role_k, mod in (("img_bars4_G", img_g), ("img_bars4_D", img_d)):
        p_rows = []
        for nm, p in mod.named_parameters():
            if p.grad is None:
                continue
            if p.ndim == 2 and min(p.shape) >= 2:
                p_rows.append({"name": nm, "kind": "2d", **spectral_metrics(p.grad)})
            elif p.ndim == 4 and min(p.shape[0], p.shape[1]) >= 2:
                c0, c1, kh, kw = p.shape
                sl = p.grad.permute(2, 3, 0, 1).reshape(kh * kw, c0, c1)
                sl_m = [spectral_metrics(sl[i]) for i in range(kh * kw)]
                p_rows.append({
                    "name": nm,
                    "kind": "4d_per_offset",
                    "shape": [c0, c1, kh, kw],
                    "stable_rank": _median([x["stable_rank"] for x in sl_m]),
                    "rv_erank": _median([x["rv_erank"] for x in sl_m]),
                })
        conv_spectra[role_k] = p_rows

    def _compact_spec(sp: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        layers = sp["layers"]
        sampled_idx = [i for i in (0, 3, 7, 11, 14) if i < len(layers)]
        res = {
            "num_layers": sp["num_layers"],
            "median_stable_rank": sp["median_stable_rank"],
            "min_stable_rank": sp["min_stable_rank"],
            "median_rv_erank": sp["median_rv_erank"],
            "median_s2_s1": sp["median_s2_s1"],
            "norm_skew": sp["norm_skew"],
            "sampled_layers": [
                {
                    "idx": i,
                    "name": layers[i]["name"],
                    "stable_rank": layers[i]["stable_rank"],
                    "rv_erank": layers[i]["rv_erank"],
                    "s2_s1": layers[i]["sigma2_over_sigma1"],
                    "frob": layers[i]["frob"],
                }
                for i in sampled_idx
            ],
        }
        if extra:
            res.update(extra)
        return res

    out["exp1b_contrastive_cone_and_arch_spectra"] = {
        "G16_adv": _compact_spec(g16_spec),
        "D16_f0_single_sign": _compact_spec(d16_single_spec, {"input_grad_med": measure_input_grad_norm(d16_f0, real100)}),
        "D16_f0_contrastive": _compact_spec(d16_diff_f0_spec, {"input_grad_med": measure_input_grad_norm(d16_f0, real100)}),
        "D16_f2_contrastive": _compact_spec(d16_diff_f2_spec, {"input_grad_med": measure_input_grad_norm(d16_f2, real100)}),
        "G3_adv": _compact_spec(g3_spec),
        "D3_f2_contrastive": _compact_spec(d3_diff_spec, {"input_grad_med": measure_input_grad_norm(d3_f2, real100)}),
        "img_bars4_conv": conv_spectra,
    }

    # 1c. Theorem 3: Spectral-norm distortion of unified_v12's sqrt(r) Frob-MP vs Polar NS vs Spec-Matched Frob-MP
    m_sz = 128
    rng_c = torch.Generator().manual_seed(0)
    q1, _ = torch.linalg.qr(torch.randn(m_sz, m_sz, generator=rng_c, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(m_sz, m_sz, generator=rng_c, dtype=torch.float64))
    t3_rows = []
    for target_srk in (1.0, 1.2, 4.0, 16.0, 64.0):
        # Construct spectrum with sigma_1 = 1 and sigma_{2..m} = c so stable_rank = 1 + (m-1)*c^2 = target_srk
        c_val = math.sqrt(max(0.0, (target_srk - 1.0) / float(m_sz - 1)))
        sv = torch.tensor([1.0] + [c_val] * (m_sz - 1), dtype=torch.float64)
        g_syn = (q1 * sv) @ q2.T
        frob = float(g_syn.norm().item())
        s1 = float(sv[0].item())
        srk = (frob * frob) / (s1 * s1)
        # Unit-lr updates (eta = 1.0)
        u_polar = q1 @ q2.T if c_val > 0 else torch.outer(q1[:, 0], q2[:, 0])
        u_v12_frob = math.sqrt(float(m_sz)) * g_syn / frob
        u_spec_frob = g_syn / s1
        s_v12 = torch.linalg.svdvals(u_v12_frob)
        s_spec = torch.linalg.svdvals(u_spec_frob)
        s_pol = torch.linalg.svdvals(u_polar)
        t3_rows.append({
            "target_srk": target_srk,
            "actual_srk": srk,
            "theory_sigma1_ratio": math.sqrt(float(m_sz) / srk),
            "v12_frob_sigma1": float(s_v12[0].item()),
            "v12_frob_sigma2": float(s_v12[1].item()),
            "v12_frob_norm": float(u_v12_frob.norm().item()),
            "spec_frob_sigma1": float(s_spec[0].item()),
            "spec_frob_sigma2": float(s_spec[1].item()),
            "spec_frob_norm": float(u_spec_frob.norm().item()),
            "polar_sigma1": float(s_pol[0].item()),
            "polar_sigma2": float(s_pol[1].item()),
        })
    out["exp1c_sqrt_r_spectral_distortion"] = t3_rows

    # 1d. Hypothesis (e): Realized function-space sensitivity Delta G(z) vs Delta D(x) & Delta(grad_x D)
    def _measure_func_sensitivity(
        label: str,
        G_mod: nn.Module,
        D_mod: nn.Module,
        z_probe: torch.Tensor,
        x_real: torch.Tensor,
        eps_frob: float = 0.05,
    ) -> Dict[str, Any]:
        G_mod.eval()
        D_mod.eval()
        with torch.no_grad():
            y_g0 = G_mod(z_probe).detach()
            d_r0 = D_mod(x_real).detach()
        xr_req = x_real.detach().clone().requires_grad_(True)
        grad_d0 = torch.autograd.grad(D_mod(xr_req).sum(), xr_req)[0].detach()

        # Compute actual adversarial gradients for G and D
        G_mod.zero_grad(set_to_none=True)
        (-D_mod(G_mod(z_probe)).mean()).backward()
        g_grads = [p.grad.detach().clone() for p in G_mod.parameters() if p.grad is not None]
        g_params = [p for p in G_mod.parameters() if p.grad is not None]

        D_mod.zero_grad(set_to_none=True)
        (-(D_mod(x_real).mean() - D_mod(y_g0).mean())).backward()
        d_grads = [p.grad.detach().clone() for p in D_mod.parameters() if p.grad is not None]
        d_params = [p for p in D_mod.parameters() if p.grad is not None]

        def _build_dirs(params: List[torch.Tensor], grads: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
            ns_dirs = []
            for p, g in zip(params, grads):
                if p.ndim == 2 and min(p.shape) >= 2:
                    r, c = g.shape
                    mc = g.T if r > c else g
                    od = quintic_orthogonalize_2d(mc.float(), steps=5, smoothing=0.0061).float()
                    ns_dirs.append((od.T if r > c else od).to(g.dtype))
                else:
                    gn = g.norm().clamp_min(1e-30)
                    ns_dirs.append(g / gn)
            # Normalize each vector field to exact global Frobenius norm == eps_frob
            def _norm_eps(vs: List[torch.Tensor]) -> List[torch.Tensor]:
                tot = math.sqrt(sum(float(v.float().square().sum().item()) for v in vs))
                if tot <= 1e-30:
                    return [torch.zeros_like(v) for v in vs]
                return [v * (eps_frob / tot) for v in vs]

            g_unit = _norm_eps(grads)
            ns_unit = _norm_eps(ns_dirs)
            # Orthogonal component of NS direction perpendicular to raw gradient
            dot = sum(float((a.float() * b.float()).sum().item()) for a, b in zip(ns_unit, g_unit))
            gn2 = sum(float(b.float().square().sum().item()) for b in g_unit)
            proj_c = dot / max(gn2, 1e-30)
            orth_raw = [a - proj_c * b for a, b in zip(ns_unit, g_unit)]
            orth_unit = _norm_eps(orth_raw)
            return {"raw_grad": g_unit, "ns_dir": ns_unit, "ns_orth_component": orth_unit}

        g_dir_map = _build_dirs(g_params, g_grads)
        d_dir_map = _build_dirs(d_params, d_grads)
        res_sens: Dict[str, Any] = {"label": label, "eps_frob": eps_frob, "G": {}, "D": {}}

        coord_scale = max(0.05, float(x_real.std().item()))
        for dname, deltas in g_dir_map.items():
            with torch.no_grad():
                for p, dv in zip(g_params, deltas):
                    p.add_(dv)
                y_g1 = G_mod(z_probe)
                for p, dv in zip(g_params, deltas):
                    p.sub_(dv)
                delta_g = float((y_g1 - y_g0).square().mean().sqrt().item()) / coord_scale
            res_sens["G"][dname] = {"delta_G_rel": delta_g}

        for dname, deltas in d_dir_map.items():
            with torch.no_grad():
                for p, dv in zip(d_params, deltas):
                    p.add_(dv)
                d_r1 = D_mod(x_real)
                delta_d = float((d_r1 - d_r0).square().mean().sqrt().item())
            xr_req2 = x_real.detach().clone().requires_grad_(True)
            grad_d1 = torch.autograd.grad(D_mod(xr_req2).sum(), xr_req2)[0].detach()
            delta_grad_d = float((grad_d1 - grad_d0).square().mean().sqrt().item())
            with torch.no_grad():
                for p, dv in zip(d_params, deltas):
                    p.sub_(dv)
            res_sens["D"][dname] = {"delta_D_rms": delta_d, "delta_grad_x_D_rms": delta_grad_d}

        return res_sens

    sens_list = []
    # Regimes A, B, C
    for lbl, g_l, d_l, d_f in (
        ("Regime_A_16LG_3LD", 16, 3, 2),
        ("Regime_B_3LG_16LD_f0", 3, 16, 0),
        ("Regime_C_3LG_3LD", 3, 3, 2),
    ):
        gm = SimpleMLPGenerator(2, 128, g_l, 2)
        dm = SimpleMLPDiscriminator(2, 128, d_l, fourier=d_f)
        deterministic_orthogonal_(gm, seed=0)
        deterministic_orthogonal_(dm, seed=0)
        gm, dm = gm.to(device), dm.to(device)
        sens_list.append(_measure_func_sensitivity(lbl, gm, dm, z2[:128], real100[:128], eps_frob=0.05))
    # Imbalanced Nash 620x D>>G and 548x G>>D
    for nkind in ("nash_overpowered_critic_ring16", "nash_overpowered_generator_ring16"):
        nspec = NASH_HOST_DEFS[nkind]
        gm, dm = build_nash_models(nspec, z_dim=nspec["resources"]["z_dim"])
        deterministic_orthogonal_(gm, seed=0)
        deterministic_orthogonal_(dm, seed=0)
        gm, dm = gm.to(device), dm.to(device)
        z_n = torch.randn(128, nspec["resources"]["z_dim"], device=device, generator=torch.Generator(device=device).manual_seed(0))
        r_n = sample_nash_target(nkind, 128, torch.Generator(device="cpu").manual_seed(0), device=device)
        sens_list.append(_measure_func_sensitivity(nkind, gm, dm, z_n, r_n, eps_frob=0.05))
    out["exp1d_function_space_sensitivity"] = sens_list

    return out


# ---------------------------------------------------------------------------
# 5. Experiment 2: Matched 5-Arm Factorial Across Structural Regimes A, B, B+, C1, C2
# ---------------------------------------------------------------------------

ARMS_5: Tuple[Tuple[str, str, str], ...] = (
    ("Neither", "frob", "frob"),
    ("G-only", "ns", "frob"),
    ("D-only", "frob", "ns"),
    ("Both", "ns", "ns"),
    ("Adaptive-Pathology", "adaptive", "adaptive"),
)


def _run_pure_adversarial_factorial(
    device: torch.device,
    *,
    regime_name: str,
    g_layers: int,
    d_layers: int,
    d_fourier: int,
    steps: int,
    frob_scale: str,
    lr_g: float,
    lr_d: float,
    law2_beta: float = 2.0,
    tau_srk: float = 1.25,
    arms: Tuple[Tuple[str, str, str], ...] = ARMS_5,
    g_b1: float = 0.0,
    d_b1: float = 0.0,
) -> Dict[str, Any]:
    """Matched seed-0, matched-batch-sequence pure adversarial RpGAN (+ optional Law 2 BI) factorial."""
    # Pre-generate the EXACT shared real-batch stream on CPU so every arm sees identical data
    data_rng = torch.Generator(device=device).manual_seed(0)
    real_batches = [sample_ring(256, device, data_rng, n_modes=8, radius=2.5, sigma=0.08) for _ in range(steps)]
    # Shared fixed probe inputs for realized Delta G / Delta D measurement
    probe_rng = torch.Generator(device=device).manual_seed(42)
    z_probe = torch.randn(128, 2, device=device, generator=probe_rng)
    x_probe = sample_ring(128, device, probe_rng, n_modes=8, radius=2.5, sigma=0.08)

    arm_results: Dict[str, Any] = {}
    for arm_name, g_mode, d_mode in arms:
        G = SimpleMLPGenerator(2, 128, g_layers, 2)
        D = SimpleMLPDiscriminator(2, 128, d_layers, fourier=d_fourier)
        prior = particlegan.MoGParticlePrior(256, 2, sigma=0.1, standardize=False, learnable=True)
        deterministic_orthogonal_(G, seed=0)
        deterministic_orthogonal_(D, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        G, D, prior = G.to(device), D.to(device), prior.to(device)

        opt_g = SymmetricLaw4Optimizer(
            G.parameters(), lr=lr_g, mode=g_mode, frob_scale=frob_scale, smoothing=0.003, tau_srk=tau_srk, b1=g_b1
        )
        opt_d = SymmetricLaw4Optimizer(
            D.parameters(), lr=lr_d, mode=d_mode, frob_scale=frob_scale, smoothing=0.0, tau_srk=tau_srk, b1=d_b1
        )
        lat_rng = torch.Generator(device=device).manual_seed(100)

        delta_g_hist: List[float] = []
        delta_d_hist: List[float] = []
        gn_hist: List[float] = []
        spec_snaps: Dict[str, Any] = {}
        snap_steps = {10, max(20, steps // 2), steps}

        t0 = time.perf_counter()
        finite_ok = True
        for st in range(1, steps + 1):
            real = real_batches[st - 1]
            # D step
            D.zero_grad(set_to_none=True)
            with torch.no_grad():
                z_d, _ = prior.sample(256, generator=lat_rng)
                fk = G(z_d)
                d_pre = D(x_probe) if (st % 25 == 0 or st == steps) else None
            dr = D(real)
            df = D(fk)
            adv_d = F.softplus(-(dr - df)).mean()
            if law2_beta > 0.0:
                pen, gn_med = born_infeld_slope_penalty_2d(D, real, fk, beta=law2_beta, kappa=1.0)
                loss_d = adv_d + pen
            else:
                loss_d = adv_d
                gn_med = measure_input_grad_norm(D, real[:64]) if (st % 25 == 0 or st == steps) else float("nan")
            loss_d.backward()
            if st in snap_steps:
                d_sp = summarize_module_grad_spectra(D, min_dim=16)
            else:
                d_sp = None
            opt_d.step()
            if d_pre is not None:
                with torch.no_grad():
                    d_post = D(x_probe)
                    delta_d_hist.append(float((d_post - d_pre).square().mean().sqrt().item()))
                if math.isfinite(gn_med):
                    gn_hist.append(gn_med)

            # G + prior step (100% purely through D — zero direct OT bypass!)
            G.zero_grad(set_to_none=True)
            prior.zero_grad(set_to_none=True)
            for p in D.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                g_pre = G(z_probe) if (st % 25 == 0 or st == steps) else None
            z_g, _ = prior.sample(256, generator=lat_rng)
            fk2 = G(z_g)
            adv_g = F.softplus(-(D(fk2) - D(real).detach())).mean()
            adv_g.backward()
            for p in D.parameters():
                p.requires_grad_(True)
            if st in snap_steps:
                g_sp = summarize_module_grad_spectra(G, min_dim=16)
                spec_snaps[f"step_{st}"] = {
                    "G_med_srk": g_sp["median_stable_rank"],
                    "G_med_rv": g_sp["median_rv_erank"],
                    "D_med_srk": d_sp["median_stable_rank"] if d_sp else 0.0,
                    "D_med_rv": d_sp["median_rv_erank"] if d_sp else 0.0,
                }
            opt_g.step()
            with torch.no_grad():
                if prior.z.grad is not None and bool(torch.isfinite(prior.z.grad).all()):
                    gn_p = prior.z.grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    prior.z.add_(prior.z.grad / gn_p, alpha=-0.03)
                if g_pre is not None:
                    g_post = G(z_probe)
                    delta_g_hist.append(float((g_post - g_pre).square().mean().sqrt().item()))
            if not (bool(torch.isfinite(adv_d).all()) and bool(torch.isfinite(adv_g).all())):
                finite_ok = False
                break

        wall_s = time.perf_counter() - t0
        ev = evaluate_ring(G, prior, device, n_modes=8, radius=2.5, sigma=0.08)
        # Critic discrimination quality: real ring8 vs between-mode valleys and vs Gaussian
        with torch.no_grad():
            ev_rng = torch.Generator(device=device).manual_seed(777)
            r_ev = sample_ring(1024, device, ev_rng, n_modes=8, radius=2.5, sigma=0.08)
            g_ev = torch.randn(1024, 2, device=device, generator=ev_rng) * 2.0
            k_v = torch.randint(0, 8, (1024,), device=device, generator=ev_rng)
            ang_v = (2.0 * math.pi / 8.0) * (k_v.float() + 0.5)
            val_ev = torch.stack((2.5 * ang_v.cos(), 2.5 * ang_v.sin()), dim=1) + torch.randn(1024, 2, device=device, generator=ev_rng) * 0.08
            acc_gauss = float((D(r_ev) > D(g_ev)).float().mean().item()) if finite_ok else 0.0
            acc_valley = float((D(r_ev) > D(val_ev)).float().mean().item()) if finite_ok else 0.0
        final_gn = measure_input_grad_norm(D, x_probe) if finite_ok else float("nan")
        tot_2d = opt_g.total_2d_calls + opt_d.total_2d_calls
        tot_ns = opt_g.ns_calls + opt_d.ns_calls
        arm_results[arm_name] = {
            "finite": bool(finite_ok and ev["finite"]),
            "modes": ev["modes"],
            "hq": ev["hq"],
            "mass_tv": ev["mass_tv"],
            "d_acc_gauss": acc_gauss,
            "d_acc_valley": acc_valley,
            "critic_grad_med": final_gn,
            "mean_delta_G": _mean(delta_g_hist),
            "mean_delta_D": _mean(delta_d_hist),
            "tail_delta_D": _mean(delta_d_hist[-4:]) if len(delta_d_hist) >= 4 else _mean(delta_d_hist),
            "ns_G": opt_g.ns_calls,
            "total_2d_G": opt_g.total_2d_calls,
            "ns_D": opt_d.ns_calls,
            "total_2d_D": opt_d.total_2d_calls,
            "ns_saved_frac_vs_both": 1.0 - (float(tot_ns) / float(max(1, tot_2d))),
            "wall_seconds": wall_s,
            "spectra": spec_snaps,
        }
        print(
            f"    [{regime_name}] {arm_name:18s} done ({wall_s:4.1f}s) | modes={ev['modes']}/8 hq={ev['hq']:.4f} acc_v={acc_valley:.3f} ns_G={opt_g.ns_calls} ns_D={opt_d.ns_calls}",
            flush=True,
        )
    return {
        "regime": regime_name,
        "g_layers": g_layers,
        "d_layers": d_layers,
        "d_fourier": d_fourier,
        "steps": steps,
        "frob_scale": frob_scale,
        "law2_beta": law2_beta,
        "tau_srk": tau_srk,
        "arms": arm_results,
    }


def _run_standalone_deep_critic_control(device: torch.device, steps: int = 300) -> Dict[str, Any]:
    """Standalone 16-layer skip-free critic discrimination with Law 2 BI (isolating D from G)."""
    data_rng = torch.Generator(device=device).manual_seed(0)
    real_b = [sample_ring(256, device, data_rng, n_modes=8, radius=2.5, sigma=0.08) for _ in range(steps)]
    fake_b = [torch.randn(256, 2, device=device, generator=data_rng) * 2.0 for _ in range(steps)]
    configs = [
        ("Frob-MP_v12_sqrt_r", "frob", "sqrt_m", 0.0),
        ("Frob-MP_spec_matched", "frob", "spec", 0.0),
        ("Frob-MP_aspect", "frob", "aspect", 0.0),
        ("5step_Quintic_NS", "ns", "spec", 0.0),
        ("Adaptive_Pathology", "adaptive", "spec", 0.0),
    ]
    res: Dict[str, Any] = {}
    for name, mode, fscale, b1 in configs:
        D = SimpleMLPDiscriminator(2, 128, 16, fourier=0)
        deterministic_orthogonal_(D, seed=0)
        D = D.to(device)
        opt_d = SymmetricLaw4Optimizer(D.parameters(), lr=0.012, mode=mode, frob_scale=fscale, smoothing=0.0, tau_srk=1.25, b1=b1)
        t0 = time.perf_counter()
        gn = 0.0
        for st in range(steps):
            D.zero_grad(set_to_none=True)
            dr = D(real_b[st])
            df = D(fake_b[st])
            adv = F.softplus(-(dr - df)).mean()
            pen, gn = born_infeld_slope_penalty_2d(D, real_b[st], fake_b[st], beta=2.0, kappa=1.0)
            (adv + pen).backward()
            opt_d.step()
        wall_s = time.perf_counter() - t0
        with torch.no_grad():
            ev_rng = torch.Generator(device=device).manual_seed(999)
            r_ev = sample_ring(1024, device, ev_rng, n_modes=8, radius=2.5, sigma=0.08)
            f_ev = torch.randn(1024, 2, device=device, generator=ev_rng) * 2.0
            k_v = torch.randint(0, 8, (1024,), device=device, generator=ev_rng)
            ang_v = (2.0 * math.pi / 8.0) * (k_v.float() + 0.5)
            val_ev = torch.stack((2.5 * ang_v.cos(), 2.5 * ang_v.sin()), dim=1) + torch.randn(1024, 2, device=device, generator=ev_rng) * 0.08
            acc_gauss = float((D(r_ev) > D(f_ev)).float().mean().item())
            acc_valley = float((D(r_ev) > D(val_ev)).float().mean().item())
            margin = float((D(r_ev).mean() - D(f_ev).mean()).item())
        res[name] = {
            "acc_gauss": acc_gauss,
            "acc_valley": acc_valley,
            "margin": margin,
            "critic_grad_med": gn,
            "ns_D": opt_d.ns_calls,
            "total_2d_D": opt_d.total_2d_calls,
            "ns_saved_frac": 1.0 - (float(opt_d.ns_calls) / float(max(1, opt_d.total_2d_calls))),
            "wall_seconds": wall_s,
        }
    return res


def _run_particlegan_mlp_factorial(
    device: torch.device,
    *,
    regime_name: str,
    task: str,
    g_layers: int,
    d_layers: int,
    d_fourier: int,
    steps: int,
    num_particles: int,
    batch_size: int,
    prior_sigma: float,
    tau_srk: float = 1.25,
    arms: Tuple[Tuple[str, str, str], ...] = ARMS_5,
) -> Dict[str, Any]:
    """Matched 5-arm factorial through real ParticleGAN `GANTrainer` (`unified_v12` Laws 1-3)."""
    data_rng = torch.Generator(device=device).manual_seed(0)
    if task == "grid100":
        real_batches = [sample_100gaussians(batch_size, device, generator=data_rng) for _ in range(steps)]
        n_modes_total = 100
    elif task == "ring16":
        real_batches = [sample_ring(batch_size, device, data_rng, n_modes=16, radius=3.0, sigma=0.10) for _ in range(steps)]
        n_modes_total = 16
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    probe_rng = torch.Generator(device=device).manual_seed(42)
    z_probe = torch.randn(128, 2, device=device, generator=probe_rng)
    x_probe = real_batches[0][:128].detach().clone()

    arm_results: Dict[str, Any] = {}
    for arm_name, g_mode, d_mode in arms:
        rec = particlegan.get_recipe(
            "unified_v12",
            prior_kind="mog",
            standardize=False,
            num_particles=num_particles,
            z_dim=2,
            batch_size=batch_size,
        )
        G = SimpleMLPGenerator(2, 128, g_layers, 2)
        D = SimpleMLPDiscriminator(2, 128, d_layers, fourier=d_fourier)
        prior = particlegan.MoGParticlePrior(num_particles, 2, sigma=prior_sigma, standardize=False, learnable=True)
        deterministic_orthogonal_(G, seed=0)
        deterministic_orthogonal_(D, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        G, D, prior = G.to(device), D.to(device), prior.to(device)
        trainer = particlegan.GANTrainer(rec, G, D, prior=prior, seed=0)
        opt_g, opt_d = _swap_trainer_optimizers(
            trainer,
            g_ns_mode=g_mode,
            d_ns_mode=d_mode,
            tau_srk=tau_srk,
            frob_scale_mode="sqrt_m",
        )

        delta_g_hist: List[float] = []
        delta_d_hist: List[float] = []
        spec_snaps: Dict[str, Any] = {}
        snap_steps = {10, max(20, steps // 2), steps}
        t0 = time.perf_counter()
        finite_ok = True
        for st in range(1, steps + 1):
            real = real_batches[st - 1]
            measure_step = (st % 50 == 0) or (st == steps)
            if measure_step:
                G.eval()
                D.eval()
                with torch.no_grad():
                    g_pre = G(z_probe)
                    d_pre = D(x_probe)
                G.train()
                D.train()
            step_out = trainer.step(real)
            if not (math.isfinite(float(step_out["loss_d"])) and math.isfinite(float(step_out["loss_g"]))):
                finite_ok = False
                break
            if measure_step:
                G.eval()
                D.eval()
                with torch.no_grad():
                    delta_g_hist.append(float((G(z_probe) - g_pre).square().mean().sqrt().item()))
                    delta_d_hist.append(float((D(x_probe) - d_pre).square().mean().sqrt().item()))
                G.train()
                D.train()
            if st in snap_steps:
                # Fresh gradient spectrum snapshot without perturbing trainer RNG
                srng = torch.Generator(device=device).manual_seed(700 + st)
                z_s, _ = prior.sample(min(256, batch_size), generator=srng)
                fk_s = G(z_s)
                G.zero_grad(set_to_none=True)
                (-D(fk_s).mean()).backward()
                g_sp = summarize_module_grad_spectra(G, min_dim=16)
                D.zero_grad(set_to_none=True)
                (-(D(real[: min(256, batch_size)]).mean() - D(fk_s.detach()).mean())).backward()
                d_sp = summarize_module_grad_spectra(D, min_dim=16)
                spec_snaps[f"step_{st}"] = {
                    "G_med_srk": g_sp["median_stable_rank"],
                    "G_med_rv": g_sp["median_rv_erank"],
                    "D_med_srk": d_sp["median_stable_rank"],
                    "D_med_rv": d_sp["median_rv_erank"],
                    "ema_fg": float(opt_g._ema_func_gain),
                }

        wall_s = time.perf_counter() - t0
        if not finite_ok:
            modes, hq, mass_tv = 0, 0.0, 1.0
            c_gn = float("nan")
        elif task == "grid100":
            modes, hq = mode_coverage(
                G,
                prior,
                device,
                n_eval=10000,
                sample_generator=torch.Generator(device=device).manual_seed(999),
            )
            mass_tv = float("nan")
            c_gn = measure_input_grad_norm(D, x_probe)
        else:
            ev = evaluate_ring(G, prior, device, n_modes=16, radius=3.0, sigma=0.10)
            modes, hq, mass_tv = ev["modes"], ev["hq"], ev["mass_tv"]
            c_gn = measure_input_grad_norm(D, x_probe)

        tot_2d = opt_g.p5_total_2d_calls + opt_d.p5_total_2d_calls
        tot_ns = opt_g.p5_ns_2d_calls + opt_d.p5_ns_2d_calls
        arm_results[arm_name] = {
            "finite": finite_ok,
            "modes": modes,
            "total_modes": n_modes_total,
            "hq": hq,
            "mass_tv": None if math.isnan(mass_tv) else mass_tv,
            "critic_grad_med": None if math.isnan(c_gn) else c_gn,
            "mean_delta_G": _mean(delta_g_hist),
            "mean_delta_D": _mean(delta_d_hist),
            "tail_delta_D": _mean(delta_d_hist[-4:]) if len(delta_d_hist) >= 4 else _mean(delta_d_hist),
            "ns_G": opt_g.p5_ns_2d_calls,
            "total_2d_G": opt_g.p5_total_2d_calls,
            "ns_D": opt_d.p5_ns_2d_calls,
            "total_2d_D": opt_d.p5_total_2d_calls,
            "ns_saved_frac_vs_both": 1.0 - (float(tot_ns) / float(max(1, tot_2d))),
            "wall_seconds": wall_s,
            "spectra": spec_snaps,
        }
        print(
            f"    [{regime_name}] {arm_name:18s} done ({wall_s:4.1f}s) | modes={modes}/{n_modes_total} hq={hq:.4f} ns_G={opt_g.p5_ns_2d_calls} ns_D={opt_d.p5_ns_2d_calls}",
            flush=True,
        )
    return {
        "regime": regime_name,
        "task": task,
        "g_layers": g_layers,
        "d_layers": d_layers,
        "d_fourier": d_fourier,
        "steps": steps,
        "tau_srk": tau_srk,
        "arms": arm_results,
    }


def _run_conv2d_bars4_factorial(device: torch.device, steps: int = 500, tau_srk: float = 1.25) -> Dict[str, Any]:
    """Regime C2: Balanced Conv2d multimodal `img_bars4` 5-arm factorial."""
    case = _image_case("image-develop-img_bars4-source-transpose12")
    centers = template_bank("bars4", device=device)
    data_rng = torch.Generator(device=device).manual_seed(101)
    bsz = int(case["batch_size"])
    real_batches = []
    for _ in range(steps):
        ids = torch.randint(len(centers), (bsz,), generator=data_rng, device=device)
        batch = (centers[ids] + 0.01 * torch.randn((bsz, 1, 8, 8), generator=data_rng, device=device)).clamp(0.0, 1.0)
        real_batches.append(batch)

    probe_rng = torch.Generator(device=device).manual_seed(42)
    z_probe = torch.randn(32, int(case["z_dim"]), device=device, generator=probe_rng)
    x_probe = real_batches[0][:32].detach().clone()

    arm_results: Dict[str, Any] = {}
    for arm_name, g_mode, d_mode in ARMS_5:
        rec = particlegan.get_recipe(
            "unified_v12",
            z_dim=int(case["z_dim"]),
            num_particles=32,
            prior_kind="particles",
            sigma_rel=0.0,
            batch_size=bsz,
            total_steps=steps,
        )
        G = ImageGenerator(case).to(device)
        D = ImageCritic(case).to(device)
        prior = rec.make_prior().to(device)
        deterministic_orthogonal_(G, seed=0)
        deterministic_orthogonal_(D, seed=1)
        deterministic_orthogonal_(prior, seed=2)
        trainer = particlegan.GANTrainer(
            rec,
            G,
            D,
            prior=prior,
            seed=0,
            max_steps=steps,
            model_generator=torch.Generator(device=device).manual_seed(8),
        )
        opt_g, opt_d = _swap_trainer_optimizers(
            trainer,
            g_ns_mode=g_mode,
            d_ns_mode=d_mode,
            tau_srk=tau_srk,
            frob_scale_mode="sqrt_m",
        )
        delta_g_hist: List[float] = []
        delta_d_hist: List[float] = []
        t0 = time.perf_counter()
        for st in range(1, steps + 1):
            measure_step = (st % 50 == 0) or (st == steps)
            if measure_step:
                G.eval()
                D.eval()
                with torch.no_grad():
                    g_pre = G(z_probe)
                    d_pre = D(x_probe)
                G.train()
                D.train()
            trainer.step(real_batches[st - 1])
            if measure_step:
                G.eval()
                D.eval()
                with torch.no_grad():
                    delta_g_hist.append(float((G(z_probe) - g_pre).square().mean().sqrt().item()))
                    delta_d_hist.append(float((D(x_probe) - d_pre).square().mean().sqrt().item()))
                G.train()
                D.train()
        wall_s = time.perf_counter() - t0
        ev_gen = torch.Generator(device=device).manual_seed(999)
        G.eval()
        with torch.no_grad():
            z_ev, _ = prior.sample(1024, generator=ev_gen)
            imgs = G(z_ev)
        scored = score_image_case(case["id"], imgs)
        m = scored["metrics"]
        tot_2d = opt_g.p5_total_2d_calls + opt_d.p5_total_2d_calls
        tot_ns = opt_g.p5_ns_2d_calls + opt_d.p5_ns_2d_calls
        arm_results[arm_name] = {
            "passed": bool(scored["passed"]),
            "modes": int(m["modes"]),
            "hq": float(m["hq"]),
            "mean_rmse": float(m["mean_rmse"]),
            "distribution_tv": float(m["distribution_tv"]),
            "mean_delta_G": _mean(delta_g_hist),
            "mean_delta_D": _mean(delta_d_hist),
            "ns_G": opt_g.p5_ns_2d_calls,
            "total_2d_G": opt_g.p5_total_2d_calls,
            "ns_D": opt_d.p5_ns_2d_calls,
            "total_2d_D": opt_d.p5_total_2d_calls,
            "ns_saved_frac_vs_both": 1.0 - (float(tot_ns) / float(max(1, tot_2d))),
            "wall_seconds": wall_s,
        }
        print(
            f"    [Regime_C2_Conv2d_img_bars4] {arm_name:18s} done ({wall_s:4.1f}s) | pass={bool(scored['passed'])} modes={int(m['modes'])}/4 hq={float(m['hq']):.4f} rmse={float(m['mean_rmse']):.4f} ns_G={opt_g.p5_ns_2d_calls} ns_D={opt_d.p5_ns_2d_calls}",
            flush=True,
        )
    return {"regime": "Regime_C2_Conv2d_img_bars4", "steps": steps, "tau_srk": tau_srk, "arms": arm_results}


def run_experiment_2(device: torch.device) -> Dict[str, Any]:
    """Run the matched 5-arm factorial across Regimes A, B (B1a/B1b/B2/B3), C1, and C2."""
    out: Dict[str, Any] = {}
    print("[Exp 2] Running Regime A: Deep 16L Skip-Free G + Shallow 3L D (grid100_mlp16)...", flush=True)
    out["regime_A_deep_G_shallow_D"] = _run_particlegan_mlp_factorial(
        device,
        regime_name="Regime_A_16LG_3LD_grid100",
        task="grid100",
        g_layers=16,
        d_layers=3,
        d_fourier=2,
        steps=1200,
        num_particles=5000,
        batch_size=512,
        prior_sigma=0.025,
        tau_srk=1.25,
    )

    print("[Exp 2] Running Regime B standalone 16L skip-free critic control...", flush=True)
    out["regime_B_standalone_16LD_control"] = _run_standalone_deep_critic_control(device, steps=300)

    print("[Exp 2] Running Regime B1a: Pure Adversarial RpGAN+Law2 (3L G + 16L D, v12 sqrt_r Frob-MP)...", flush=True)
    out["regime_B1a_pure_adv_deep_D_v12_sqrt_r"] = _run_pure_adversarial_factorial(
        device,
        regime_name="Regime_B1a_PureAdv_3LG_16LD_v12_sqrt_r",
        g_layers=3,
        d_layers=16,
        d_fourier=0,
        steps=450,
        frob_scale="sqrt_m",
        lr_g=0.008,
        lr_d=0.012,
        law2_beta=2.0,
        tau_srk=1.25,
    )

    print("[Exp 2] Running Regime B1b: Pure Adversarial RpGAN+Law2 (3L G + 16L D, spec-matched Frob-MP)...", flush=True)
    out["regime_B1b_pure_adv_deep_D_spec_matched"] = _run_pure_adversarial_factorial(
        device,
        regime_name="Regime_B1b_PureAdv_3LG_16LD_spec_matched",
        g_layers=3,
        d_layers=16,
        d_fourier=0,
        steps=450,
        frob_scale="spec",
        lr_g=0.008,
        lr_d=0.012,
        law2_beta=2.0,
        tau_srk=1.25,
    )

    print("[Exp 2] Running Regime B2: Full ParticleGAN (3L G + 16L D) Direct-OT Masking Confounder...", flush=True)
    out["regime_B2_particlegan_direct_ot_confounder"] = _run_particlegan_mlp_factorial(
        device,
        regime_name="Regime_B2_ParticleGAN_3LG_16LD_ring16",
        task="ring16",
        g_layers=3,
        d_layers=16,
        d_fourier=0,
        steps=350,
        num_particles=256,
        batch_size=256,
        prior_sigma=0.10,
        tau_srk=1.25,
    )

    print("[Exp 2] Running Regime B3: Both Deep (16L G + 16L D) Pure Adversarial RpGAN+Law2...", flush=True)
    out["regime_B3_both_deep_16LG_16LD"] = _run_pure_adversarial_factorial(
        device,
        regime_name="Regime_B3_PureAdv_16LG_16LD_sqrt_r",
        g_layers=16,
        d_layers=16,
        d_fourier=0,
        steps=400,
        frob_scale="sqrt_m",
        lr_g=0.008,
        lr_d=0.012,
        law2_beta=2.0,
        tau_srk=1.25,
    )

    print("[Exp 2] Running Regime C1: Balanced Shallow (3L G + 3L D, ring16)...", flush=True)
    out["regime_C1_balanced_shallow_ring16"] = _run_particlegan_mlp_factorial(
        device,
        regime_name="Regime_C1_Balanced_3LG_3LD_ring16",
        task="ring16",
        g_layers=3,
        d_layers=3,
        d_fourier=2,
        steps=350,
        num_particles=256,
        batch_size=256,
        prior_sigma=0.10,
        tau_srk=1.25,
    )

    print("[Exp 2] Running Regime C2: Balanced Conv2d Multimodal (img_bars4)...", flush=True)
    out["regime_C2_balanced_conv2d_img_bars4"] = _run_conv2d_bars4_factorial(device, steps=500, tau_srk=1.25)

    return out


# ---------------------------------------------------------------------------
# 6. Experiment 3: Causal Disentanglement of Hypotheses (c), (f), and `t3_iso` Fallback
# ---------------------------------------------------------------------------

def run_experiment_3(device: torch.device) -> Dict[str, Any]:
    """Disentangle Law 2 slope penalty (c), capacity ratio rho>>1 (f), and t3_iso fallback."""
    out: Dict[str, Any] = {}

    # 3a. Hypothesis (c): Law 2 Born-Infeld Slope Penalty ON vs OFF x 4 NS arms on shallow & deep D
    print("[Exp 3a] Running Hypothesis (c): Law 2 ON vs OFF ablation...", flush=True)
    arms_4 = (("Neither", "frob", "frob"), ("G-only", "ns", "frob"), ("D-only", "frob", "ns"), ("Both", "ns", "ns"))
    law2_ablation: Dict[str, Any] = {}
    for arch_lbl, d_l, d_f, fscale in (
        ("shallow_3LG_3LD", 3, 2, "spec"),
        ("deep_critic_3LG_16LD_v12sqrt", 16, 0, "sqrt_m"),
    ):
        law2_ablation[arch_lbl] = {}
        for beta_lbl, beta_val in (("law2_ON", 2.0), ("law2_OFF", 0.0)):
            law2_ablation[arch_lbl][beta_lbl] = _run_pure_adversarial_factorial(
                device,
                regime_name=f"{arch_lbl}_{beta_lbl}",
                g_layers=3,
                d_layers=d_l,
                d_fourier=d_f,
                steps=250,
                frob_scale=fscale,
                lr_g=0.008,
                lr_d=0.012,
                law2_beta=beta_val,
                arms=arms_4,
            )["arms"]
    out["exp3a_law2_on_vs_off"] = law2_ablation

    # 3b. Hypothesis (f): Capacity ratio rho >> 1 (exact Forge imbalanced Nash 620x D>>G & 548x G>>D)
    print("[Exp 3b] Running Hypothesis (f): 620x D>>G and 548x G>>D capacity ratio ablation...", flush=True)
    nash_res: Dict[str, Any] = {}
    nash_arms: Tuple[Tuple[str, str, str], ...] = (
        *ARMS_5,
        ("Equil-G_SGD-D", "ns", "sgd"),
        ("SGD-G_Equil-D", "sgd", "ns"),
    )
    nash_steps = 700
    for kind in ("nash_overpowered_critic_ring16", "nash_overpowered_generator_ring16"):
        spec = NASH_HOST_DEFS[kind]
        res_cfg = spec["resources"]
        bsz = int(res_cfg["batch_size"])
        z_dim = int(res_cfg["z_dim"])
        n_p = int(res_cfg["num_particles"])
        data_rng = torch.Generator(device="cpu").manual_seed(0)
        real_batches = [
            sample_nash_target(kind, bsz, data_rng, completed_steps=st, device=device)
            for st in range(1, nash_steps + 1)
        ]
        kind_out: Dict[str, Any] = {}
        for arm_name, g_mode, d_mode in nash_arms:
            rec = particlegan.get_recipe(
                "unified_v12",
                prior_kind="mog",
                standardize=False,
                num_particles=n_p,
                z_dim=z_dim,
                batch_size=bsz,
            )
            G, D = build_nash_models(spec, z_dim=z_dim)
            prior = particlegan.MoGParticlePrior(n_p, z_dim, sigma=0.1, standardize=False, learnable=True)
            deterministic_orthogonal_(G, seed=0)
            deterministic_orthogonal_(D, seed=0)
            deterministic_orthogonal_(prior, seed=0)
            G, D, prior = G.to(device), D.to(device), prior.to(device)
            trainer = particlegan.GANTrainer(rec, G, D, prior=prior, seed=0)
            opt_g, opt_d = _swap_trainer_optimizers(
                trainer,
                g_ns_mode=g_mode,
                d_ns_mode=d_mode,
                tau_srk=1.25,
                frob_scale_mode="sqrt_m",
            )
            t0 = time.perf_counter()
            finite_ok = True
            fail_step = None
            for st in range(1, nash_steps + 1):
                try:
                    so = trainer.step(real_batches[st - 1])
                    if not (math.isfinite(float(so["loss_d"])) and math.isfinite(float(so["loss_g"]))):
                        finite_ok = False
                        fail_step = st
                        break
                except (FloatingPointError, RuntimeError, ValueError):
                    finite_ok = False
                    fail_step = st
                    break
            wall_s = time.perf_counter() - t0
            if finite_ok:
                ev = trainer.sample(2048)
                sc = score_imbalance_nash_kind(kind, ev, completed_steps=nash_steps)
                m = sc["metrics"]
                passed = bool(sc["passed"])
                modes_v = int(m["modes"])
                hq_v = float(m["hq"])
                tv_v = float(m["mass_tv"])
            else:
                passed, modes_v, hq_v, tv_v = False, 0, 0.0, 1.0
            tot_2d = opt_g.p5_total_2d_calls + opt_d.p5_total_2d_calls
            tot_ns = opt_g.p5_ns_2d_calls + opt_d.p5_ns_2d_calls
            kind_out[arm_name] = {
                "finite": finite_ok,
                "fail_step": fail_step,
                "passed": passed,
                "modes": modes_v,
                "hq": hq_v,
                "mass_tv": tv_v,
                "ns_G": opt_g.p5_ns_2d_calls,
                "ns_D": opt_d.p5_ns_2d_calls,
                "ns_saved_frac_vs_both": 1.0 - (float(tot_ns) / float(max(1, tot_2d))),
                "wall_seconds": wall_s,
            }
            print(
                f"    [{kind}] {arm_name:18s} done ({wall_s:4.1f}s) | pass={passed} modes={modes_v}/16 hq={hq_v:.4f} tv={tv_v:.3f} ns_G={opt_g.p5_ns_2d_calls} ns_D={opt_d.p5_ns_2d_calls}",
                flush=True,
            )
        nash_res[kind] = kind_out
    out["exp3b_capacity_ratio_imbalanced_nash"] = nash_res

    # 3c. Forensic Disentanglement of unified_v12's Generator-Free Critic NS Fallback on t3_iso_scale_mass_8x
    print("[Exp 3c] Running forensic t3_iso_scale_mass_8x generator-free fallback dissection...", flush=True)
    t3_arms = [
        ("v12_fallback_ON_exact", "ns", "sqrt_m", 0.30, 0.9995, 1.25),
        ("v12_fallback_OFF_exact", "frob", "sqrt_m", 0.50, 1.0, 1.25),
        ("ns_only_ablation_sqrt_r", "frob", "sqrt_m", 0.30, 0.9995, 1.25),
        ("spec_matched_0ns_d", "frob", "spec", 0.30, 0.9995, 1.25),
        ("adaptive_pathology_d", "adaptive", "spec", 0.30, 0.9995, 1.35),
    ]
    t3_res: Dict[str, Any] = {}
    for arm_name, d_mode, fscale, scap, pdec, tau in t3_arms:
        rec = particlegan.get_recipe("unified_v12", num_particles=512, z_dim=2, batch_size=128)
        fix = T3IsolationFixture("t3_iso_scale_mass_8x", rec, device=device, seed=0)
        # Swap only opt_d with P5ControlledV12Optimizer (G = nn.Identity() has 0 params)
        old_d = fix.opt_d
        groups_d = [{**{k: v for k, v in grp.items() if k != "params"}, "params": list(grp["params"])} for grp in old_d.param_groups]
        new_d = P5ControlledV12Optimizer(
            groups_d,
            lr=float(old_d.defaults["lr"]),
            betas=tuple(old_d.defaults["betas"]),
            eps=float(old_d.defaults["eps"]),
            smoothing=float(old_d.smoothing),
            direction_blend=bool(old_d.direction_blend),
            generator_free=True,
            ns_mode=d_mode,
            tau_srk=tau,
            frob_scale_mode=fscale,
            smooth_cap=scap,
            p_dec_hot=pdec,
        )
        for attr in (
            "_unified_v11_record",
            "_unified_record",
            "_t1c_record",
            "_t2_record",
            "critic",
            "ema_critic",
            "record",
            "continuous_controller",
        ):
            if hasattr(old_d, attr):
                setattr(new_d, attr, getattr(old_d, attr))
        fix.opt_d = new_d
        fix.policy.opt_d = new_d
        fix.policy.optimizers = [fix.policy.opt_g, new_d]
        if getattr(fix.policy, "penalty", None) is not None and hasattr(fix.policy.penalty, "optimizer"):
            fix.policy.penalty.optimizer = new_d
        if hasattr(fix.penalty, "optimizer"):
            fix.penalty.optimizer = new_d

        t0 = time.perf_counter()
        # Evaluate on last 5 checkpoints (steps 600, 650, 700, 750, 800) matching sustained evaluation!
        tail_min_eigs: List[float] = []
        tail_passes = 0
        srk_snaps: Dict[str, float] = {}
        for st in range(1, 801):
            fix.step()
            if st in (10, 200, 400, 800):
                d_sp = summarize_module_grad_spectra(fix.D, min_dim=16)
                srk_snaps[f"step_{st}"] = d_sp["median_stable_rank"]
            if st in (600, 650, 700, 750, 800):
                ev_r = torch.Generator(device="cpu").manual_seed(900 + st)
                ref_s = sample_t3_target("t3_iso_scale_mass_8x", 2048, ev_r, step=st, device=device)
                pts_s = fix.prior.z.detach()
                c_gn_s = measure_critic_grad_median(fix.D, pts_s, ref_s)
                sc_s = score_t3_isolation_kind("t3_iso_scale_mass_8x", pts_s, ref_s, critic_grad_med=c_gn_s)
                tail_min_eigs.append(float(sc_s["metrics"]["core_min_eigen_ratio"]))
                if sc_s["passed"]:
                    tail_passes += 1
        wall_s = time.perf_counter() - t0
        # Final step-800 metrics
        ev_rng = torch.Generator(device="cpu").manual_seed(999)
        ref = sample_t3_target("t3_iso_scale_mass_8x", 2048, ev_rng, step=800, device=device)
        pts = fix.prior.z.detach()
        c_gn = measure_critic_grad_median(fix.D, pts, ref)
        sc = score_t3_isolation_kind("t3_iso_scale_mass_8x", pts, ref, critic_grad_med=c_gn)
        m = sc["metrics"]
        t3_res[arm_name] = {
            "final_passed": bool(sc["passed"]),
            "sustained_5_passes": tail_passes,
            "tail_min_eig_worst": min(tail_min_eigs),
            "tail_min_eig_mean": _mean(tail_min_eigs),
            "modes": int(m["modes"]),
            "hq": float(m["hq"]),
            "mass_tv": float(m["mass_tv"]),
            "core_min_eigen_ratio": float(m["core_min_eigen_ratio"]),
            "core_component_covariance_error": float(m["core_component_covariance_error"]),
            "critic_grad_med": float(m["critic_grad_med"]),
            "ns_D": new_d.p5_ns_2d_calls,
            "total_2d_D": new_d.p5_total_2d_calls,
            "ns_saved_frac": 1.0 - (float(new_d.p5_ns_2d_calls) / float(max(1, new_d.p5_total_2d_calls))),
            "wall_seconds": wall_s,
            "d_srk_snaps": srk_snaps,
        }
        print(
            f"    [t3_iso_scale_mass_8x] {arm_name:24s} done ({wall_s:4.1f}s) | pass={bool(sc['passed'])} sust={tail_passes}/5 min_eig={float(m['core_min_eigen_ratio']):.4f} hq={float(m['hq']):.4f} ||∇D||={float(m['critic_grad_med']):.3f} ns_D={new_d.p5_ns_2d_calls}/{new_d.p5_total_2d_calls}",
            flush=True,
        )
    out["exp3c_t3_iso_fallback_dissection"] = t3_res

    return out


# ---------------------------------------------------------------------------
# 7. Console Summary & Entrypoint
# ---------------------------------------------------------------------------

def print_summary(report: Dict[str, Any]) -> None:
    print("\n================ P5 LAW 4.4 OPTIMIZER ASYMMETRY AUDIT SUMMARY ================")
    print("particlegan:", report["meta"]["particlegan_file"])
    print("device:", report["meta"]["device"])

    if "experiment_1" in report:
        e1 = report["experiment_1"]
        print("\n--- Exp 1a: Scalar (d_out=1) vs Vector Output Rank Bottleneck ---")
        for m_k, sub in e1["exp1a_scalar_vs_vector_rank"]["mlp_sweep"].items():
            for d_k, v in sub.items():
                print(f"  {m_k:10s} {d_k:8s} | rk(1e-12)={v['median_rank_tol_1e-12']:4.1f} s_rk={v['median_stable_rank']:5.2f} rv={v['median_rv_erank']:5.2f}")
        for tag, v in e1["exp1a_scalar_vs_vector_rank"]["law2_tangent_rank_lift"].items():
            print(f"  Law2 lift {tag:20s} | adv_only rk={v['adv_only_rank_1e-12']:.1f} s_rk={v['adv_only']['median_stable_rank']:.2f} -> +Law2 rk={v['adv_plus_law2_rank_1e-12']:.1f} s_rk={v['adv_plus_law2_bi']['median_stable_rank']:.2f}")

        print("\n--- Exp 1b: Contrastive Mean-Cone Cancellation & Architectural Spectra ---")
        for k, v in e1["exp1b_contrastive_cone_and_arch_spectra"].items():
            if k == "img_bars4_conv":
                continue
            gn_str = f" ||∇D||={v['input_grad_med']:.2e}" if "input_grad_med" in v else ""
            print(f"  {k:22s} | med_srk={v['median_stable_rank']:.3f} min_srk={v['min_stable_rank']:.3f} rv={v['median_rv_erank']:5.2f} s2/s1={v['median_s2_s1']:.2e} skew={v['norm_skew']:8.1f}x{gn_str}")

        print("\n--- Exp 1c: sqrt(r) Polar-Frobenius Spectral-Norm Distortion (128x128) ---")
        for row in e1["exp1c_sqrt_r_spectral_distortion"]:
            print(f"  srk={row['actual_srk']:5.2f} | theory={row['theory_sigma1_ratio']:5.2f}x | v12_frob s1={row['v12_frob_sigma1']:5.2f} s2={row['v12_frob_sigma2']:.3f} | spec_frob s1={row['spec_frob_sigma1']:4.2f} | polar s1={row['polar_sigma1']:4.2f}")

        print("\n--- Exp 1d: Realized Function-Space Sensitivity (eps_frob=0.05) ---")
        for s in e1["exp1d_function_space_sensitivity"]:
            print(
                f"  {s['label']:32s} | dG(raw/ns/orth)={s['G']['raw_grad']['delta_G_rel']:.4f}/{s['G']['ns_dir']['delta_G_rel']:.4f}/{s['G']['ns_orth_component']['delta_G_rel']:.4f} "
                f"| dD(raw/ns/orth)={s['D']['raw_grad']['delta_D_rms']:.4f}/{s['D']['ns_dir']['delta_D_rms']:.4f}/{s['D']['ns_orth_component']['delta_D_rms']:.4f} "
                f"| d∇D(raw/ns/orth)={s['D']['raw_grad']['delta_grad_x_D_rms']:.4f}/{s['D']['ns_dir']['delta_grad_x_D_rms']:.4f}/{s['D']['ns_orth_component']['delta_grad_x_D_rms']:.4f}"
            )

    if "experiment_2" in report:
        e2 = report["experiment_2"]
        print("\n--- Exp 2: Standalone 16L Critic Control (f=0, Law 2 BI) ---")
        for nm, v in e2["regime_B_standalone_16LD_control"].items():
            print(f"  {nm:24s} | acc_g={v['acc_gauss']:.4f} acc_val={v['acc_valley']:.4f} margin={v['margin']:+.3f} ||∇D||={v['critic_grad_med']:.3f} ns={v['ns_D']}/{v['total_2d_D']}")

        for rk in (
            "regime_A_deep_G_shallow_D",
            "regime_B1a_pure_adv_deep_D_v12_sqrt_r",
            "regime_B1b_pure_adv_deep_D_spec_matched",
            "regime_B2_particlegan_direct_ot_confounder",
            "regime_B3_both_deep_16LG_16LD",
            "regime_C1_balanced_shallow_ring16",
            "regime_C2_balanced_conv2d_img_bars4",
        ):
            blk = e2[rk]
            print(f"\n--- Exp 2: {blk['regime']} ---")
            for arm, v in blk["arms"].items():
                d_val = f" acc_v={v['d_acc_valley']:.3f}" if "d_acc_valley" in v else ""
                gn_v = v.get("critic_grad_med")
                gn_s = f" ||∇D||={gn_v:.2e}" if isinstance(gn_v, (int, float)) and math.isfinite(gn_v) else ""
                print(
                    f"  {arm:18s} | modes={v['modes']:3d} hq={v['hq']:.4f}{d_val}{gn_s} "
                    f"| dG={v['mean_delta_G']:.4f} dD={v['mean_delta_D']:.4f} "
                    f"| ns_G={v['ns_G']:5d}/{v['total_2d_G']:5d} ns_D={v['ns_D']:5d}/{v['total_2d_D']:5d} "
                    f"saved={100*v['ns_saved_frac_vs_both']:5.1f}% | {v['wall_seconds']:4.1f}s"
                )

    if "experiment_3" in report:
        e3 = report["experiment_3"]
        print("\n--- Exp 3a: Law 2 ON vs OFF Ablation ---")
        for arch_k, sub in e3["exp3a_law2_on_vs_off"].items():
            for l2_k, arms_d in sub.items():
                print(f"  [{arch_k} / {l2_k}]")
                for arm, v in arms_d.items():
                    print(f"    {arm:8s} | modes={v['modes']}/8 hq={v['hq']:.4f} acc_v={v['d_acc_valley']:.3f} ||∇D||={v['critic_grad_med']:.3f} dD={v['mean_delta_D']:.4f}")

        print("\n--- Exp 3b: Capacity Ratio (620x D>>G & 548x G>>D) ---")
        for kind, arms_d in e3["exp3b_capacity_ratio_imbalanced_nash"].items():
            print(f"  [{kind}]")
            for arm, v in arms_d.items():
                print(f"    {arm:18s} | fin={str(v['finite']):5s} pass={str(v['passed']):5s} modes={v['modes']:2d}/16 hq={v['hq']:.4f} tv={v['mass_tv']:.4f} ns_G={v['ns_G']} ns_D={v['ns_D']} ({v['wall_seconds']:.1f}s)")

        print("\n--- Exp 3c: Forensic t3_iso_scale_mass_8x Generator-Free Fallback Dissection ---")
        for arm, v in e3["exp3c_t3_iso_fallback_dissection"].items():
            print(
                f"  {arm:24s} | fin_pass={str(v['final_passed']):5s} sust={v['sustained_5_passes']}/5 "
                f"| min_eig={v['core_min_eigen_ratio']:.4f} (worst5={v['tail_min_eig_worst']:.4f}) "
                f"cov={v['core_component_covariance_error']:.4f} hq={v['hq']:.4f} ||∇D||={v['critic_grad_med']:.4f} "
                f"| ns_D={v['ns_D']}/{v['total_2d_D']} ({v['wall_seconds']:.1f}s)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 Law 4.4 Generator-Critic Optimizer Asymmetry Audit")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sections", default="1,2,3", help="Comma-separated experiment sections to run (1,2,3)")
    parser.add_argument("--output", default="", help="Optional JSON output path (not tracked in git)")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    wanted = {s.strip() for s in args.sections.split(",") if s.strip()}
    report: Dict[str, Any] = {
        "meta": {
            "particlegan_file": str(particlegan.__file__),
            "torch_version": str(torch.__version__),
            "device": str(device),
            "seed": 0,
        }
    }
    t_all = time.perf_counter()
    if "1" in wanted:
        report["experiment_1"] = run_experiment_1(device)
    if "2" in wanted:
        report["experiment_2"] = run_experiment_2(device)
    if "3" in wanted:
        report["experiment_3"] = run_experiment_3(device)
    report["meta"]["total_wall_seconds"] = time.perf_counter() - t_all

    print_summary(report)
    if args.output:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote JSON summary to {args.output}")


if __name__ == "__main__":
    main()
