#!/usr/bin/env python3
"""Axiom Four-Laws Adversarial Head-to-Head Benchmark.

Deterministic (protocol seed=0, deterministic_orthogonal_ init) matched comparison of:
  - axiom        (Axiom family: clean-slate Laws 1-4)
  - unified_v12  (Unified v12 baseline)
  - unified_v13  (Unified v13 baseline)
  - minimal      (Minimal baseline)
  - rhbi         (RHBI baseline)

Covers all five Four-Laws audit stress suites:
  1. Law 4.2 (P5 + P0): Deep 16L skip-free critic swap (Regime B1), 16L generator
     (Regime A), healthy r256 DCGANCritic zero-NS guard, and P0 rank-1 NS leakage.
  2. Law 4.4 (P1): 5-regime scale invariance (1e-12..1e12 & underflow), 1e-12 tanh
     rail escape + settling, 1e4x spike recovery, 1/t decay + chatter settling, and
     adversarial bilinear rotation.
  3. Law 4.1 & 4.3 (P4): Verified iterative stratified p95 pushforward trust on
     saturating tanh / superlinear cubic / 16L tanh MLP, coord_scale homogeneity
     (1e-4..1e4), and 1% rare tail-mode capture.
  4. Law 2 (P6): Below-kappa linear-R1 unbiasedness (zero Jensen false-triggering),
     dimension-independent equilibrium normal dissipation (dim=2 vs dim=196,608),
     and EmulatedLinear (mxfp8/nvfp4) + fwAD-failure survival.
  5. Law 1 (P7): Exact proto_w & w_fake_ema mass conservation under shock
     teleportation, exact Gersho merge & split LTV + 10x scale guard + psi gauge,
     B_f < K deficit quota allocation, and real GANTrainer GPU host-sync count.

Run (keeping physical GPU 0 idle via CUDA_VISIBLE_DEVICES=1):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
      scripts/axiom_four_laws_benchmark.py --device cuda:0
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from experiments.forge.scaling_benchmark import DCGANCritic
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, mode_coverage, sample_100gaussians
import particlegan as pg
from particlegan._axiom.critic import AxiomStepRecord, _attach_axiom_generator_hook
from particlegan._axiom.optimizer import AxiomOptimizer, fp32_tikhonov_ns_2d
from particlegan._axiom.prior import AxiomPriorRegularizer, allocate_deficit_quotas, coalesce_pythagorean_ltv
from particlegan._minimal import dynamics as minimal_dynamics
from particlegan._minimal.dynamics import MinimalGeneratorOptimizer
from particlegan._minimal.prior import MinimalPriorRegularizer, _coalesce_isolated
from particlegan._unified_v11.critic import UnifiedV11StepRecord, _attach_unified_v11_generator_hook
from particlegan._unified_v11.optimizer import quintic_orthogonalize_2d
from particlegan._unified_v11.prior import (
    UnifiedV11PriorRegularizer,
    _compute_pythagorean_smoothed_cells_fast,
)
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer
from particlegan.init import deterministic_orthogonal_
from particlegan.low_precision import EmulatedLinear
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.particle_prior import MoGParticlePrior
from particlegan.training import GANTrainer

PRESETS: Tuple[str, ...] = ("axiom", "unified_v12", "unified_v13", "minimal", "rhbi")


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_ring(
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
def _evaluate_ring(
    gen_mod: nn.Module,
    prior: nn.Module,
    device: torch.device,
    *,
    n_modes: int = 8,
    radius: float = 2.5,
    sigma: float = 0.08,
    n_eval: int = 4096,
) -> Dict[str, Any]:
    was_train = gen_mod.training
    gen_mod.eval()
    g_ev = torch.Generator(device=device).manual_seed(999)
    z, _ = prior.sample(n_eval, generator=g_ev)
    fake = gen_mod(z)
    gen_mod.train(was_train)
    if not bool(torch.isfinite(fake).all().item()):
        return {"finite": False, "modes": 0, "hq": 0.0, "mass_tv": 1.0}
    ang = (2.0 * math.pi / float(n_modes)) * torch.arange(n_modes, device=device, dtype=torch.float32)
    centers = torch.stack((radius * ang.cos(), radius * ang.sin()), dim=1)
    mind, idx = torch.cdist(fake, centers).min(dim=1)
    hq_mask = mind <= (3.5 * sigma)
    hq = float(hq_mask.float().mean().item())
    counts_hq = torch.bincount(idx[hq_mask], minlength=n_modes)
    min_cnt = max(10, n_eval // (n_modes * 8))
    modes = int((counts_hq >= min_cnt).sum().item())
    mass = torch.bincount(idx, minlength=n_modes).float() / float(n_eval)
    mass_tv = float(0.5 * (mass - (1.0 / float(n_modes))).abs().sum().item())
    return {"finite": True, "modes": modes, "hq": hq, "mass_tv": mass_tv}


def _born_infeld_2d(
    critic: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    beta: float = 2.0,
    kappa: float = 1.0,
    gauge_weight: float = 0.05,
) -> torch.Tensor:
    """Exact 1st-order fwAD Born-Infeld spatial slope penalty + zero-mean gauge in R^2."""
    x = torch.cat((x_real.detach(), x_fake.detach()), dim=0)
    e1 = torch.zeros_like(x)
    e2 = torch.zeros_like(x)
    e1[:, 0] = 1.0
    e2[:, 1] = 1.0
    with fwAD.dual_level():
        primal, j1 = fwAD.unpack_dual(critic(fwAD.make_dual(x, e1)))
        _, j2 = fwAD.unpack_dual(critic(fwAD.make_dual(x, e2)))
    gn_sq = j1.square() + j2.square()
    bi = beta * (torch.sqrt(1.0 + gn_sq / (kappa * kappa)) - 1.0).mean()
    return bi + gauge_weight * primal.mean().square()


@contextmanager
def _track_minimal_orth(opt_g: Any = None, opt_d: Any = None):
    """Count exact calls to minimal's 2D orthogonalizer attributed accurately to G vs D."""
    state = {"active": "G", "G": 0, "D": 0}
    orig = getattr(minimal_dynamics, "_orth", None)
    orig_g_step = getattr(opt_g, "step", None) if opt_g is not None else None
    orig_d_step = getattr(opt_d, "step", None) if opt_d is not None else None
    if orig is None:
        yield state
        return

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        state[state["active"]] += 1
        return orig(*args, **kwargs)

    def _wrap_opt_step(orig_step: Any, role: str):
        def _step(*args: Any, **kwargs: Any) -> Any:
            prev = state["active"]
            state["active"] = role
            try:
                return orig_step(*args, **kwargs)
            finally:
                state["active"] = prev
        return _step

    minimal_dynamics._orth = _wrapped  # type: ignore[assignment]
    if opt_g is not None and orig_g_step is not None:
        opt_g.step = _wrap_opt_step(orig_g_step, "G")
    if opt_d is not None and orig_d_step is not None:
        opt_d.step = _wrap_opt_step(orig_d_step, "D")
    try:
        yield state
    finally:
        minimal_dynamics._orth = orig  # type: ignore[assignment]
        if opt_g is not None and orig_g_step is not None:
            opt_g.step = orig_g_step
        if opt_d is not None and orig_d_step is not None:
            opt_d.step = orig_d_step


def _extract_ns_count(opt: Any, min_count: int) -> int:
    if hasattr(opt, "ns_2d_activations"):
        return int(opt.ns_2d_activations)
    if hasattr(opt, "dense_profile_stats"):
        raw = opt.dense_profile_stats
        st = raw() if callable(raw) else raw
        if isinstance(st, dict):
            agg = st.get("aggregate", st)
            return int(agg.get("calls", 0))
    return int(min_count)


@contextmanager
def _count_host_syncs(device: torch.device):
    """Count explicit Tensor.item(), Tensor.tolist(), Tensor.__bool__(), torch.unique, and nonzero calls."""
    counts = {"item": 0, "tolist": 0, "bool": 0, "unique": 0, "nonzero": 0}
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_bool = torch.Tensor.__bool__
    orig_unique = torch.unique
    orig_nonzero = torch.nonzero
    orig_tensor_nonzero = torch.Tensor.nonzero

    def _wrap_item(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.device.type == device.type:
            counts["item"] += 1
        return orig_item(self, *args, **kwargs)

    def _wrap_tolist(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.device.type == device.type:
            counts["tolist"] += 1
        return orig_tolist(self, *args, **kwargs)

    def _wrap_bool(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.device.type == device.type:
            counts["bool"] += 1
        return orig_bool(self, *args, **kwargs)

    def _wrap_unique(inp: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(inp, torch.Tensor) and inp.device.type == device.type:
            counts["unique"] += 1
        return orig_unique(inp, *args, **kwargs)

    def _wrap_nonzero(inp: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(inp, torch.Tensor) and inp.device.type == device.type:
            counts["nonzero"] += 1
        return orig_nonzero(inp, *args, **kwargs)

    def _wrap_tensor_nonzero(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.device.type == device.type:
            counts["nonzero"] += 1
        return orig_tensor_nonzero(self, *args, **kwargs)

    torch.Tensor.item = _wrap_item  # type: ignore[assignment]
    torch.Tensor.tolist = _wrap_tolist  # type: ignore[assignment]
    torch.Tensor.__bool__ = _wrap_bool  # type: ignore[assignment]
    torch.unique = _wrap_unique  # type: ignore[assignment]
    torch.nonzero = _wrap_nonzero  # type: ignore[assignment]
    torch.Tensor.nonzero = _wrap_tensor_nonzero  # type: ignore[assignment]
    try:
        yield counts
    finally:
        torch.Tensor.item = orig_item  # type: ignore[assignment]
        torch.Tensor.tolist = orig_tolist  # type: ignore[assignment]
        torch.Tensor.__bool__ = orig_bool  # type: ignore[assignment]
        torch.unique = orig_unique  # type: ignore[assignment]
        torch.nonzero = orig_nonzero  # type: ignore[assignment]
        torch.Tensor.nonzero = orig_tensor_nonzero  # type: ignore[assignment]


def _make_standalone_opt(preset: str, params: List[torch.Tensor], lr: float) -> Any:
    """Instantiate each preset's actual generator optimizer class with matched lr for P1."""
    if preset == "axiom":
        return AxiomOptimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.05,
            direction_blend=False,
        )
    if preset in ("unified_v12", "unified_v13"):
        return UnifiedV12Optimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.0061,
            direction_blend=False,
        )
    if preset == "minimal":
        return MinimalGeneratorOptimizer(
            [{"params": params, "role": "generator"}],
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
        )
    if preset == "rhbi":
        return NormalizedOptimizer(
            [{"params": params, "role": "generator"}],
            family="dualnorm",
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8,
            smoothing=0.001,
            dense_profile="polar_cpu_v1",
        )
    raise ValueError(f"unknown preset: {preset!r}")


# ---------------------------------------------------------------------------
# Suite 1: Law 4.2 (P5 Role-Agnostic Pathology NS + P0 FP32 Tikhonov NS)
# ---------------------------------------------------------------------------

def run_suite1_law42_p5_p0(device: torch.device) -> Dict[str, Any]:
    print("\n=== SUITE 1: Law 4.2 (P5 Deep 16L Critic/Gen + Healthy Conv2d + P0 Rank-1 NS) ===", flush=True)

    # 1A: Regime B1 — Pure adversarial RpGAN + each recipe's own critic penalty (3L G + 16L skip-free D, ring8, 300 steps)
    steps_b1, bsz_b1, n_p_b1 = 300, 256, 512
    matched_lr_1, matched_d_lr_mult_1 = 0.014, 1.5
    data_rng = torch.Generator(device=device).manual_seed(0)
    real_b1 = [_sample_ring(bsz_b1, device, data_rng, n_modes=8, radius=2.5, sigma=0.08) for _ in range(steps_b1)]

    regime_b1: Dict[str, Any] = {}
    for preset in PRESETS:
        rec = pg.get_recipe(
            preset,
            num_particles=n_p_b1,
            z_dim=2,
            prior_kind="mog",
            batch_size=bsz_b1,
            lr=matched_lr_1,
            d_lr_mult=matched_d_lr_mult_1,
        )
        gen_m = SimpleMLPGenerator(2, 128, 3, 2)
        crit_m = SimpleMLPDiscriminator(2, 128, 16, fourier=0)
        prior = MoGParticlePrior(n_p_b1, 2, sigma=0.025, standardize=False, learnable=True)
        deterministic_orthogonal_(gen_m, seed=0)
        deterministic_orthogonal_(crit_m, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        gen_m, crit_m, prior = gen_m.to(device), crit_m.to(device), prior.to(device)
        rec.apply_prior_policy(prior)
        opt_g, opt_d = rec.make_optimizers(gen_m, crit_m, prior=prior)
        loss_fn = rec.make_loss()
        pen_fn = rec.make_critic_penalty(opt_d)
        pg.bind_formulation_records(rec, opt_d, loss_fn, None, generator=gen_m)
        lat_rng = torch.Generator(device=device).manual_seed(0)
        _sync(device)
        t0 = time.perf_counter()
        finite_ok = True
        with _track_minimal_orth(opt_g, opt_d) as min_orth:
            for st in range(steps_b1):
                real = real_b1[st]
                # D step (recipe's actual loss + critic penalty)
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_d, _ = prior.sample(bsz_b1, generator=lat_rng)
                    fk_d = gen_m(z_d)
                dr, df = crit_m(real), crit_m(fk_d)
                adv_d = loss_fn.d_loss(dr, df)
                pen = pen_fn(crit_m, real, fk_d)
                loss_d = adv_d + pen
                loss_d.backward()
                opt_d.step()

                # G + prior step (purely adversarial through D)
                opt_g.zero_grad(set_to_none=True)
                with pg.critic_g_step_mode(crit_m):
                    z_g, idx_g = prior.sample(bsz_b1, generator=lat_rng)
                    if hasattr(opt_g, "set_sampled_rows"):
                        opt_g.set_sampled_rows(prior.z, idx_g)
                    fk_g = gen_m(z_g)
                    adv_g = loss_fn.g_loss(crit_m(fk_g), crit_m(real).detach())
                    if getattr(opt_g, "direction_blend", False) and hasattr(opt_g, "bind_protected_from_grads"):
                        adv_g.backward()
                        opt_g.bind_protected_from_grads()
                    else:
                        constraint_geometry_backward(adv_g, opt_g, (adv_g,))
                opt_g.step()
                if hasattr(opt_g, "clear_sampled_rows"):
                    opt_g.clear_sampled_rows()
                if not (bool(torch.isfinite(loss_d).all()) and bool(torch.isfinite(adv_g).all())):
                    finite_ok = False
                    break
        _sync(device)
        wall_s = time.perf_counter() - t0
        ev = _evaluate_ring(gen_m, prior, device, n_modes=8, radius=2.5, sigma=0.08)
        with torch.no_grad():
            ev_rng = torch.Generator(device=device).manual_seed(777)
            r_ev = _sample_ring(1024, device, ev_rng, n_modes=8, radius=2.5, sigma=0.08)
            g_ev = torch.randn(1024, 2, device=device, generator=ev_rng) * 2.0
            k_v = torch.randint(0, 8, (1024,), device=device, generator=ev_rng)
            ang_v = (2.0 * math.pi / 8.0) * (k_v.float() + 0.5)
            val_ev = torch.stack((2.5 * ang_v.cos(), 2.5 * ang_v.sin()), dim=1) + (
                torch.randn(1024, 2, device=device, generator=ev_rng) * 0.08
            )
            acc_g = float((crit_m(r_ev) > crit_m(g_ev)).float().mean().item()) if finite_ok else 0.0
            acc_v = float((crit_m(r_ev) > crit_m(val_ev)).float().mean().item()) if finite_ok else 0.0
        ns_g = _extract_ns_count(opt_g, min_orth["G"])
        ns_d = _extract_ns_count(opt_d, min_orth["D"])
        regime_b1[preset] = {
            "finite": bool(finite_ok and ev["finite"]),
            "modes": int(ev["modes"]),
            "hq": float(ev["hq"]),
            "mass_tv": float(ev["mass_tv"]),
            "d_acc_gauss": acc_g,
            "d_acc_valley": acc_v,
            "ns_G": ns_g,
            "ns_D": ns_d,
            "wall_s": wall_s,
            "matched_lr": matched_lr_1,
            "matched_d_lr_mult": matched_d_lr_mult_1,
        }
        print(
            f"  [1A Regime B1 16L-D] {preset:12s} | modes={ev['modes']}/8 hq={ev['hq']:.4f} "
            f"acc_g={acc_g:.3f} acc_v={acc_v:.3f} ns_G={ns_g} ns_D={ns_d} ({wall_s:.1f}s)",
            flush=True,
        )

    # 1B: Regime A — Deep 16-layer skip-free Generator on grid100_mlp16 via real GANTrainer (500 steps)
    steps_a, bsz_a, n_p_a = 500, 256, 2048
    data_rng_a = torch.Generator(device=device).manual_seed(0)
    real_a = [sample_100gaussians(bsz_a, device, generator=data_rng_a) for _ in range(steps_a)]
    regime_a: Dict[str, Any] = {}
    for preset in PRESETS:
        rec = pg.get_recipe(
            preset,
            prior_kind="mog",
            standardize=False,
            num_particles=n_p_a,
            z_dim=2,
            batch_size=bsz_a,
            lr=matched_lr_1,
            d_lr_mult=matched_d_lr_mult_1,
        )
        gen_m = SimpleMLPGenerator(2, 128, 16, 2)
        crit_m = SimpleMLPDiscriminator(2, 128, 3, fourier=0)
        prior = MoGParticlePrior(n_p_a, 2, sigma=0.025, standardize=False, learnable=True)
        deterministic_orthogonal_(gen_m, seed=0)
        deterministic_orthogonal_(crit_m, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        gen_m, crit_m, prior = gen_m.to(device), crit_m.to(device), prior.to(device)
        trainer = GANTrainer(rec, gen_m, crit_m, prior=prior, seed=0)
        _sync(device)
        t0 = time.perf_counter()
        finite_ok = True
        with _track_minimal_orth(trainer.opt_g, trainer.opt_d) as min_orth:
            for st in range(steps_a):
                out = trainer.step(real_a[st])
                if not (math.isfinite(float(out["loss_d"])) and math.isfinite(float(out["loss_g"]))):
                    finite_ok = False
                    break
        _sync(device)
        wall_s = time.perf_counter() - t0
        if finite_ok:
            modes, hq = mode_coverage(
                gen_m,
                prior,
                device,
                n_eval=10000,
                sample_generator=torch.Generator(device=device).manual_seed(999),
            )
        else:
            modes, hq = 0, 0.0
        ns_g = _extract_ns_count(trainer.opt_g, min_orth["G"])
        ns_d = _extract_ns_count(trainer.opt_d, min_orth["D"])
        regime_a[preset] = {
            "finite": finite_ok,
            "modes": int(modes),
            "hq": float(hq),
            "ns_G": ns_g,
            "ns_D": ns_d,
            "wall_s": wall_s,
            "matched_lr": matched_lr_1,
            "matched_d_lr_mult": matched_d_lr_mult_1,
        }
        print(
            f"  [1B Regime A  16L-G] {preset:12s} | modes={modes}/100 hq={hq:.4f} "
            f"ns_G={ns_g} ns_D={ns_d} ({wall_s:.1f}s)",
            flush=True,
        )

    # 1C: Healthy r256 DCGANCritic zero-NS check + P0 128x128 rank-1 NS orthogonal leakage
    conv_ns: Dict[str, Any] = {}
    img_rng = torch.Generator(device=device).manual_seed(0)
    x_r_img = torch.randn(4, 3, 256, 256, device=device, generator=img_rng)
    x_f_img = torch.randn(4, 3, 256, 256, device=device, generator=img_rng)
    if device.type == "cuda":
        x_r_img = x_r_img.to(memory_format=torch.channels_last)
        x_f_img = x_f_img.to(memory_format=torch.channels_last)
    for preset in PRESETS:
        rec = pg.get_recipe(preset)
        d_conv = DCGANCritic(256, 64, 512).to(device)
        if device.type == "cuda":
            d_conv = d_conv.to(memory_format=torch.channels_last)
        deterministic_orthogonal_(d_conv, seed=0)
        g_dummy = nn.Linear(2, 2).to(device)
        _, opt_d = rec.make_optimizers(g_dummy, d_conv)
        with _track_minimal_orth() as min_orth:
            min_orth["active"] = "D"
            for _ in range(5):
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss = F.softplus(-(d_conv(x_r_img) - d_conv(x_f_img))).mean()
                loss.backward()
                opt_d.step()
        ns_c = _extract_ns_count(opt_d, min_orth["D"])
        conv_ns[preset] = {"ns_D_5steps": ns_c}

    # P0 rank-1 128x128 NS spectral leakage check on device
    g_p0 = torch.Generator(device="cpu").manual_seed(0)
    u_vec = torch.randn(128, 1, generator=g_p0)
    v_vec = torch.randn(1, 128, generator=g_p0)
    mat_r1 = (u_vec / u_vec.norm()) @ (v_vec / v_vec.norm())
    mat_r1_dev = mat_r1.to(device)

    def _spec_pair(mat_out: torch.Tensor) -> Dict[str, float]:
        sv = torch.linalg.svdvals(mat_out.detach().cpu().double())
        s1, s2 = float(sv[0].item()), float(sv[1].item())
        return {"sigma_1": s1, "sigma_2": s2, "sigma2_over_sigma1": s2 / max(s1, 1e-30)}

    p0_rank1 = {
        "axiom_fp32_ns_s0": _spec_pair(fp32_tikhonov_ns_2d(mat_r1_dev, steps=5, rel_smooth=0.0)),
        "axiom_fp32_ns_s0.05": _spec_pair(fp32_tikhonov_ns_2d(mat_r1_dev, steps=5, rel_smooth=0.05)),
        "v12_v13_bf16_ns_s0": _spec_pair(quintic_orthogonalize_2d(mat_r1_dev, steps=5, smoothing=0.0)),
        "v12_v13_bf16_ns_s0.0061": _spec_pair(quintic_orthogonalize_2d(mat_r1_dev, steps=5, smoothing=0.0061)),
        "exact_polar_ref": _spec_pair(mat_r1),
    }
    print(
        f"  [1C P0 Rank-1 128x128] axiom s2/s1={p0_rank1['axiom_fp32_ns_s0']['sigma2_over_sigma1']:.2e} "
        f"(s1={p0_rank1['axiom_fp32_ns_s0']['sigma_1']:.3f}) vs v12 s2/s1="
        f"{p0_rank1['v12_v13_bf16_ns_s0']['sigma2_over_sigma1']:.4f} "
        f"(s1={p0_rank1['v12_v13_bf16_ns_s0']['sigma_1']:.3f})",
        flush=True,
    )
    return {
        "regime_b1_16l_critic": regime_b1,
        "regime_a_16l_generator": regime_a,
        "healthy_r256_conv2d_ns": conv_ns,
        "p0_rank1_128x128": p0_rank1,
    }


# ---------------------------------------------------------------------------
# Suite 2: Law 4.4 (P1 5-Regime Scale / Rail / Spike / Decay / Rotation)
# ---------------------------------------------------------------------------

def run_suite2_law44_p1() -> Dict[str, Any]:
    print("\n=== SUITE 2: Law 4.4 (P1 5-Regime Scale / Rail / Spike / Decay / Rotation) ===", flush=True)
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn((8, 8), generator=gen, dtype=torch.float32) * 0.5
    q, _ = torch.linalg.qr(torch.randn((8, 8), generator=gen, dtype=torch.float32))
    h_mat = q @ torch.diag(torch.linspace(0.5, 2.0, 8, dtype=torch.float32)) @ q.T
    target = torch.randn((8, 8), generator=gen, dtype=torch.float32) * 0.5

    # 2A: 24-decade rescaling invariance + extreme underflow probe
    scales = (1e-12, 1e-6, 1.0, 1e6, 1e12)
    underflow_scales = (1e-18, 1e-25, 1e-37, 1e-40)
    rescaling: Dict[str, Any] = {}
    for preset in PRESETS:
        finals: Dict[float, torch.Tensor] = {}
        for c in scales:
            p = w0.clone().detach().requires_grad_(True)
            opt = _make_standalone_opt(preset, [p], lr=0.02)
            for _ in range(40):
                opt.zero_grad()
                p.grad = (h_mat @ (p.detach() - target)) * float(c)
                opt.step()
            finals[c] = p.detach().clone()
        ref = finals[1.0]
        diffs = {f"{c:.0e}": float((finals[c] - ref).abs().max().item()) for c in scales}
        u_moves: Dict[str, float] = {}
        for c in underflow_scales:
            p = w0.clone().detach().requires_grad_(True)
            opt = _make_standalone_opt(preset, [p], lr=0.02)
            p.grad = (h_mat @ (p.detach() - target)) * float(c)
            prev = p.detach().clone()
            opt.step()
            u_moves[f"{c:.0e}"] = float((p.detach() - prev).norm().item())
        rescaling[preset] = {
            "max_diff_24dec": max(diffs.values()),
            "diff_vs_1": diffs,
            "underflow_step1_move": u_moves,
        }

    # 2B: 1e-12 tanh rail escape + settling across c in {1e-6, 1.0, 1e6}
    rail_escape: Dict[str, Any] = {}
    for preset in PRESETS:
        per_c: Dict[str, Any] = {}
        for c in (1e-6, 1.0, 1e6):
            p = (torch.eye(4, dtype=torch.float32) * 14.0).detach().requires_grad_(True)
            opt = _make_standalone_opt(preset, [p], lr=0.08)
            esc_step = -1
            tail: List[float] = []
            for t in range(1, 1001):
                opt.zero_grad()
                x = p.detach().double()
                th = torch.tanh(x)
                p.grad = (th * (1.0 - th * th) * float(c)).float()
                prev = p.detach().clone()
                opt.step()
                if esc_step < 0 and float(p.detach().abs().max().item()) < 1.0:
                    esc_step = t
                if t > 950:
                    tail.append(float((p.detach() - prev).norm().item()))
            per_c[f"{c:.0e}"] = {
                "escaped_step": esc_step,
                "final_rms": float(p.detach().square().mean().sqrt().item()),
                "tail_mean_step": sum(tail) / len(tail),
            }
        rail_escape[preset] = per_c

    # 2C: Single 1e4x gradient spike recovery (step 20, 1520 steps, c=1e-6 and c=1.0)
    gen_s = torch.Generator().manual_seed(0)
    w0_s = torch.randn((4, 4), generator=gen_s, dtype=torch.float32)
    sp_dir = torch.randn((4, 4), generator=gen_s, dtype=torch.float32)
    sp_dir = sp_dir / sp_dir.norm()
    init_err_s = float(w0_s.norm().item())
    spike_rec: Dict[str, Any] = {}
    for preset in PRESETS:
        per_c = {}
        for c in (1e-6, 1.0):
            p = w0_s.clone().detach().requires_grad_(True)
            opt = _make_standalone_opt(preset, [p], lr=0.02)
            pre_s = post_s = s100 = 0.0
            for t in range(1, 1521):
                opt.zero_grad()
                p.grad = (sp_dir * (1e4 * float(c))) if t == 20 else (p.detach() * float(c))
                prev = p.detach().clone()
                opt.step()
                dn = float((p.detach() - prev).norm().item())
                if t == 19:
                    pre_s = dn
                elif t == 21:
                    post_s = dn
                elif t == 120:
                    s100 = dn
            fe = float(p.detach().norm().item())
            per_c[f"{c:.0e}"] = {
                "post_over_pre": post_s / max(pre_s, 1e-30),
                "step100_over_pre": s100 / max(pre_s, 1e-30),
                "final_err": fe,
                "converged_95pct": fe < 0.05 * init_err_s,
            }
        spike_rec[preset] = per_c

    # 2D: 1/t coherent decay (non-freezing) + constant-amplitude sign-flipping chatter settling
    w_unit = w0_s / w0_s.norm()
    decay_and_settle: Dict[str, Any] = {}
    for preset in PRESETS:
        # Coherent 1/t decay at c=1e-4
        p_d = torch.zeros((4, 4), dtype=torch.float32, requires_grad=True)
        opt_d = _make_standalone_opt(preset, [p_d], lr=0.02)
        d_s1 = d_s2000 = 0.0
        for t in range(1, 2001):
            opt_d.zero_grad()
            p_d.grad = w_unit * (1e-4 / float(t))
            prev = p_d.detach().clone()
            opt_d.step()
            dn = float((p_d.detach() - prev).norm().item())
            if t == 1:
                d_s1 = dn
            elif t == 2000:
                d_s2000 = dn
        # Period-2 sign-flipping equilibrium chatter at constant amplitude (c=1.0)
        p_c = torch.zeros((4, 4), dtype=torch.float32, requires_grad=True)
        opt_c = _make_standalone_opt(preset, [p_c], lr=0.02)
        c_s1 = 0.0
        c_tail: List[float] = []
        for t in range(1, 401):
            opt_c.zero_grad()
            p_c.grad = w_unit * (1.0 if (t % 2 == 1) else -1.0)
            prev = p_c.detach().clone()
            opt_c.step()
            dn = float((p_c.detach() - prev).norm().item())
            if t == 1:
                c_s1 = dn
            if t > 350:
                c_tail.append(dn)
        decay_and_settle[preset] = {
            "decay_ratio_2000_over_1": d_s2000 / max(d_s1, 1e-30),
            "chatter_tail_over_step1": (sum(c_tail) / len(c_tail)) / max(c_s1, 1e-30),
        }

    # 2E: Adversarial bilinear rotation min_x max_y <x, y> (alternating, 600 steps)
    gen_r = torch.Generator().manual_seed(0)
    x0 = torch.randn((4, 4), generator=gen_r, dtype=torch.float32)
    x0 = x0 / x0.norm()
    y0 = torch.randn((4, 4), generator=gen_r, dtype=torch.float32)
    y0 = y0 / y0.norm()
    r0 = math.sqrt(float(x0.square().sum().item() + y0.square().sum().item()))
    rotation: Dict[str, Any] = {}
    for preset in PRESETS:
        x = x0.clone().detach().requires_grad_(True)
        y = y0.clone().detach().requires_grad_(True)
        opt_x = _make_standalone_opt(preset, [x], lr=0.02)
        opt_y = _make_standalone_opt(preset, [y], lr=0.02)
        tail_x: List[float] = []
        for t in range(600):
            opt_x.zero_grad()
            x.grad = y.detach().clone()
            prev = x.detach().clone()
            opt_x.step()
            opt_y.zero_grad()
            y.grad = -x.detach().clone()
            opt_y.step()
            if t >= 550:
                tail_x.append(float((x.detach() - prev).norm().item()))
        r_end = math.sqrt(float(x.detach().square().sum().item() + y.detach().square().sum().item()))
        rotation[preset] = {
            "r_end_over_r0": r_end / r0,
            "tail_mean_step_x": sum(tail_x) / len(tail_x),
        }

    for preset in PRESETS:
        print(
            f"  [2 P1] {preset:12s} | 24dec_err={rescaling[preset]['max_diff_24dec']:.2e} "
            f"u(1e-37)={rescaling[preset]['underflow_step1_move']['1e-37']:.4f} "
            f"rail(1e6)_rms={rail_escape[preset]['1e+06']['final_rms']:.4f} "
            f"spike(1e-6)_err={spike_rec[preset]['1e-06']['final_err']:.4f} "
            f"chatter={decay_and_settle[preset]['chatter_tail_over_step1']:.4f} "
            f"rot={rotation[preset]['r_end_over_r0']:.3f}",
            flush=True,
        )
    return {
        "rescaling_24dec": rescaling,
        "tanh_rail_escape": rail_escape,
        "spike_1e4x_recovery": spike_rec,
        "decay_and_chatter_settling": decay_and_settle,
        "bilinear_rotation_alt": rotation,
    }


# ---------------------------------------------------------------------------
# Suite 3: Law 4.1 & 4.3 (P4 Verified Iterative Stratified p95 Trust)
# ---------------------------------------------------------------------------

class _ScalarMapGen(nn.Module):
    """Minimal differentiable generator y = scale * fn(theta) for exact trust testing."""

    def __init__(self, theta0: float, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([[theta0, theta0]], dtype=torch.float32))
        self._fn = fn

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._fn(self.theta).expand(z.shape[0], -1)


class _TailModeGen(nn.Module):
    """Generator where a single 1% rare outlier z_norm > 4 triggers a localized sharp bump."""

    def __init__(self) -> None:
        super().__init__()
        self.w_maj = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.w_rare = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        is_rare = (z.norm(dim=-1, keepdim=True) > 4.0).float()
        return (1.0 - is_rare) * (z @ self.w_maj) + is_rare * (z @ self.w_rare)


def _run_trust_probe_step(
    preset: str,
    gen_mod: nn.Module,
    z_batch: torch.Tensor,
    raw_grads: List[torch.Tensor],
    *,
    lr: float,
    data_scale: float = math.sqrt(2.0),
    push_res: float = 0.96875,  # gives d_max = 0.045 + 0.16 * 0.96875 = 0.20 exactly!
) -> Dict[str, Any]:
    """Run actual generator hook + optimizer trust step for each recipe."""
    rec_api = pg.get_recipe(preset)
    d_dummy = nn.Linear(2, 1)
    opt_g, opt_d = rec_api.make_optimizers(gen_mod, d_dummy)
    for grp in opt_g.param_groups:
        grp["lr"] = float(lr)
    if preset == "axiom":
        rec = AxiomStepRecord()
        rec.ema_data_scale = float(data_scale)
        rec.ema_push_res = float(push_res)
        rec.margin_initialized = True
        opt_g.record = opt_g._axiom_record = rec
        _attach_axiom_generator_hook(gen_mod, rec)
    elif preset in ("unified_v12", "unified_v13"):
        rec = UnifiedV11StepRecord()
        rec.ema_data_scale = float(data_scale)
        rec.ema_push_res = float(push_res)
        rec.margin_initialized = True
        opt_g.record = opt_g._unified_v11_record = rec
        _attach_unified_v11_generator_hook(gen_mod, rec)
    elif preset == "minimal":
        rec = getattr(opt_d, "record", None)
        if rec is not None:
            rec.ema_data_scale = float(data_scale)
            rec.ema_residual = float(push_res)
            rec.initialized = True
            opt_g._minimal_record = rec
    else:
        rec = None

    with torch.no_grad():
        y_pre_full = gen_mod(z_batch).detach().clone()
    gen_mod.train()
    # Run forward so generator hook populates _g_probe_cache on the actual training path
    y_live = gen_mod(z_batch)
    cache = getattr(rec, "_g_probe_cache", None) if rec is not None else None
    probe_captured_tail = False
    if cache is not None:
        probe_z = cache[1][0]
        probe_captured_tail = bool((probe_z.norm(dim=-1) > 4.0).any().item())
    # Assign exact raw gradients (avoiding any loss distortion)
    opt_g.zero_grad(set_to_none=True)
    for p, g in zip(gen_mod.parameters(), raw_grads):
        p.grad = g.clone()
    if getattr(opt_g, "direction_blend", False) and hasattr(opt_g, "bind_protected_from_grads"):
        opt_g.bind_protected_from_grads()
    elif hasattr(opt_g, "bind_protected_losses"):
        # For rhbi DirectionBlendOptimizer: use dummy scalar loss with matching grad
        loss_proxy = sum((p * g).sum() for p, g in zip(gen_mod.parameters(), raw_grads))
        opt_g.zero_grad(set_to_none=True)
        constraint_geometry_backward(loss_proxy, opt_g, (loss_proxy,))
    del y_live
    opt_g.step()
    with torch.no_grad():
        y_post_full = gen_mod(z_batch).detach().clone()
    out_dim = max(1, int(y_pre_full[0].numel()))
    true_coord_sc = data_scale / math.sqrt(float(out_dim))
    per_sample = (y_post_full.double() - y_pre_full.double()).reshape(z_batch.shape[0], -1).square().mean(-1).sqrt()
    rms_disp = float(per_sample.square().mean().sqrt().item()) / true_coord_sc
    p95_disp = float(torch.quantile(per_sample, 0.95).item()) / true_coord_sc
    max_disp = float(per_sample.max().item()) / true_coord_sc
    d_max = 0.045 + 0.16 * min(1.5, push_res)
    return {
        "alpha": float(getattr(opt_g, "last_step_gain", 1.0)),
        "probe_consumed": bool(getattr(opt_g, "last_probe_consumed", False)),
        "probe_captured_tail": probe_captured_tail,
        "backtracks": int(getattr(opt_g, "trust_backtracks_total", 0)),
        "rms_rel_disp": rms_disp,
        "p95_rel_disp": p95_disp,
        "max_rel_disp": max_disp,
        "d_max": d_max,
        "p95_over_dmax": p95_disp / d_max,
        "rms_over_dmax": rms_disp / d_max,
    }


def run_suite3_law41_43_p4() -> Dict[str, Any]:
    print("\n=== SUITE 3: Law 4.1 & 4.3 (P4 Verified Iterative Stratified p95 Trust) ===", flush=True)
    z_small = torch.ones(16, 2, dtype=torch.float32)

    # 3A-1: Saturating tanh map f(theta) = tanh(theta), theta0=0, raw step dtheta ~ 25
    sat_tanh: Dict[str, Any] = {}
    for preset in PRESETS:
        gm = _ScalarMapGen(0.0, torch.tanh)
        # Unit-normalized 2-entry parameter has ||g_unit|| = 1 -> lr=25*sqrt(2) gives |dtheta_i| = 25
        sat_tanh[preset] = _run_trust_probe_step(
            preset,
            gm,
            z_small,
            [torch.tensor([[-1.0, -1.0]])],
            lr=25.0 * math.sqrt(2.0),
            push_res=0.96875,
        )

    # 3A-2: Superlinear cubic map f(theta) = 0.0005 * theta^3, theta0=0, raw step dtheta = 1.0
    # Raw disp = 0.0005 << v12's delta_g_min = 0.012 -> v12 amplifies ~24x upward without re-checking!
    sup_cubic: Dict[str, Any] = {}
    for preset in PRESETS:
        gm = _ScalarMapGen(0.0, lambda th: 0.0005 * th.pow(3))
        sup_cubic[preset] = _run_trust_probe_step(
            preset,
            gm,
            z_small,
            [torch.tensor([[-1.0, -1.0]])],
            lr=1.0 * math.sqrt(2.0),
            push_res=0.96875,
        )

    # 3A-3: Real 16-layer saturating tanh MLP generator with large step (lr=0.35)
    deep_tanh_mlp: Dict[str, Any] = {}
    g_z = torch.Generator().manual_seed(0)
    z_mlp = torch.randn(32, 2, generator=g_z)
    for preset in PRESETS:
        layers: List[nn.Module] = [nn.Linear(2, 32), nn.Tanh()]
        for _ in range(14):
            layers.extend([nn.Linear(32, 32), nn.Tanh()])
        layers.append(nn.Linear(32, 2))
        gm = nn.Sequential(*layers)
        deterministic_orthogonal_(gm, seed=0)
        gm.zero_grad(set_to_none=True)
        gm(z_mlp).square().mean().backward()
        grads = [p.grad.detach().clone() for p in gm.parameters()]
        deep_tanh_mlp[preset] = _run_trust_probe_step(
            preset, gm, z_mlp, grads, lr=0.35, push_res=0.96875
        )

    # 3B: coord_scale scale homogeneity across c in {1e-4, 1e-3, 1.0, 1e4}
    # Measure each preset's actual instantiated optimizer & scale-tracking record (no hardcoded mocks).
    coord_homog: Dict[str, Any] = {}
    x_unit = torch.randn(32, 2, generator=torch.Generator().manual_seed(0))
    x_unit = x_unit / x_unit.square().mean().sqrt()  # exact per-coord RMS = 1.0
    z_hom = torch.ones(16, 2, dtype=torch.float32)
    ref_disp_c1: Dict[str, float] = {}
    raw_homog_by_c: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c in (1.0, 1e-4, 1e-3, 1e4):
        x_sc = x_unit * c
        c_key = f"{c:.0e}"
        raw_homog_by_c[c_key] = {}
        for preset in PRESETS:
            rec_api = pg.get_recipe(preset)
            gm = _ScalarMapGen(0.0, lambda th, sc=c: sc * th)
            d_dum = nn.Linear(2, 1)
            deterministic_orthogonal_(d_dum, seed=0)
            opt_g, opt_d = rec_api.make_optimizers(gm, d_dum)
            loss_fn = rec_api.make_loss()
            pen_fn = rec_api.make_critic_penalty(opt_d)
            pg.bind_formulation_records(rec_api, opt_d, loss_fn, None, generator=gm)
            # Feed scaled batch through the recipe's real critic loss + penalty so its record tracks x_sc naturally.
            dr, df = d_dum(x_sc), d_dum(x_sc * 0.5)
            _ = loss_fn.d_loss(dr, df)
            _ = pen_fn(d_dum, x_sc, x_sc * 0.5)
            for grp in opt_g.param_groups:
                grp["lr"] = 10.0
            with torch.no_grad():
                y_pre = gm(z_hom).detach().clone()
            gm.train()
            _ = gm(z_hom)
            opt_g.zero_grad(set_to_none=True)
            gm.theta.grad = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
            if getattr(opt_g, "direction_blend", False) and hasattr(opt_g, "bind_protected_from_grads"):
                opt_g.bind_protected_from_grads()
            elif hasattr(opt_g, "bind_protected_losses"):
                proxy = (gm.theta * gm.theta.grad.detach()).sum()
                opt_g.zero_grad(set_to_none=True)
                constraint_geometry_backward(proxy, opt_g, (proxy,))
            opt_g.step()
            with torch.no_grad():
                y_post = gm(z_hom).detach().clone()
            rms_rel = float((y_post.double() - y_pre.double()).square().mean().sqrt().item()) / float(c)
            if c == 1.0:
                ref_disp_c1[preset] = max(rms_rel, 1e-30)
            raw_homog_by_c[c_key][preset] = {
                "rms_rel_disp": rms_rel,
                "alpha": float(getattr(opt_g, "last_step_gain", 1.0)),
            }
    for c in (1e-4, 1e-3, 1.0, 1e4):
        c_key = f"{c:.0e}"
        entry: Dict[str, Any] = {"true_coord_rms": c}
        for preset in PRESETS:
            rel = raw_homog_by_c[c_key][preset]["rms_rel_disp"]
            entry[f"{preset}_rms_rel_disp"] = rel
            entry[f"{preset}_ratio_vs_c1"] = rel / ref_disp_c1[preset]
            entry[f"{preset}_alpha"] = raw_homog_by_c[c_key][preset]["alpha"]
        coord_homog[c_key] = entry

    # 3C: Rare 1.6% tail-mode capture (B=128, 2 rare tail samples with ||z|| = 6.0 at deterministic shuffled indices)
    tail_capture: Dict[str, Any] = {}
    z_tail = 0.25 * torch.randn(128, 2, generator=torch.Generator().manual_seed(0))
    perm_tail = torch.randperm(128, generator=torch.Generator().manual_seed(0))
    tail_indices = [int(perm_tail[0].item()), int(perm_tail[1].item())]
    z_tail[tail_indices[0]] = torch.tensor([6.0, 0.0])
    z_tail[tail_indices[1]] = torch.tensor([0.0, 6.0])
    for preset in PRESETS:
        gm = _TailModeGen()
        g_maj = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        g_rare = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        res_tail = _run_trust_probe_step(
            preset,
            gm,
            z_tail,
            [g_maj, g_rare],
            lr=1.0,
            push_res=0.96875,
        )
        res_tail["tail_indices"] = list(tail_indices)
        tail_capture[preset] = res_tail

    for preset in PRESETS:
        print(
            f"  [3 P4] {preset:12s} | tanh_sat={sat_tanh[preset]['p95_over_dmax']:.2f}x "
            f"cubic_sup={sup_cubic[preset]['p95_over_dmax']:.2f}x "
            f"mlp16_tanh={deep_tanh_mlp[preset]['p95_over_dmax']:.2f}x "
            f"tail_cap={tail_capture[preset]['probe_captured_tail']} "
            f"tail_max={tail_capture[preset]['max_rel_disp']:.3f}",
            flush=True,
        )
    return {
        "saturating_tanh_dtheta25": sat_tanh,
        "superlinear_cubic_amp": sup_cubic,
        "deep_16l_tanh_mlp": deep_tanh_mlp,
        "coord_scale_homogeneity": coord_homog,
        "rare_1pct_tail_capture": tail_capture,
    }


# ---------------------------------------------------------------------------
# Suite 4: Law 2 (P6 Zero-Jensen Linear-R1, Dim-Indep mu_D, Quant/fwAD Safety)
# ---------------------------------------------------------------------------

def _flatten_pgrads(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([(p.grad.detach().float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)).reshape(-1) for p in params])


class _NoJvpFunc(torch.autograd.Function):
    """Custom autograd op with backward but NO forward-AD jvp (simulating FlashAttention/custom kernels)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_out * (s * (1.0 + x * (1.0 - s)))


class _NoJvpCritic(nn.Module):
    def __init__(self, dim: int = 24) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_NoJvpFunc.apply(self.fc1(x)))


def _make_calibrated_mlp(dim: int, target_gn: float, seed: int = 0) -> nn.Module:
    mod = nn.Sequential(nn.Linear(dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))
    deterministic_orthogonal_(mod, seed=seed)
    x_cal = torch.randn(32, dim, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    gn = float(torch.autograd.grad(mod(x_cal).sum(), x_cal)[0].norm(dim=1).mean().item())
    with torch.no_grad():
        mod[-1].weight.mul_(target_gn / max(gn, 1e-8))
    return mod


def _make_bn_free_conv256(device: torch.device, target_gn: float = 0.25, seed: int = 0) -> nn.Module:
    """BN-free 256x256x3 Conv2d critic (dim=196,608) with identical train/eval Jacobian."""
    mod = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=4, padding=0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(16, 32, 4, stride=4, padding=0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, 4, stride=4, padding=0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 1, 4, stride=1, padding=0),
        nn.Flatten(1),
    ).to(device)
    deterministic_orthogonal_(mod, seed=seed)
    x_cal = torch.randn(4, 3, 256, 256, device=device, generator=torch.Generator(device=device).manual_seed(0)).requires_grad_(True)
    gn = float(torch.autograd.grad(mod(x_cal).sum(), x_cal)[0].flatten(1).norm(dim=1).mean().item())
    with torch.no_grad():
        mod[6].weight.mul_(target_gn / max(gn, 1e-8))
        if mod[6].bias is not None:
            mod[6].bias.zero_()
    return mod


def run_suite4_law2_p6(device: torch.device) -> Dict[str, Any]:
    print("\n=== SUITE 4: Law 2 (P6 Zero-Jensen Linear-R1, Dim-Indep mu_D, Quant/fwAD Safety) ===", flush=True)

    # 4A: Below-kappa (||grad_x D|| ~ 0.65 < 1.0) gradient unbiasedness & false-trigger check
    # Dimension d=24 forces all stochastic recipes onto their MC fwAD path. Zero mean logit isolates slope penalty!
    dim_j, bsz_j, n_mc = 24, 32, 128
    g_j = torch.Generator().manual_seed(0)
    xr_j = torch.randn(bsz_j, dim_j, generator=g_j)
    xf_j = torch.randn(bsz_j, dim_j, generator=g_j)
    base_d = _make_calibrated_mlp(dim_j, target_gn=0.65, seed=0)
    with torch.no_grad():
        base_d[-1].bias.zero_()
        base_d[-1].bias.sub_(0.5 * (base_d(xr_j).mean() + base_d(xf_j).mean()))

    def _exact_ror_penalty_grad(
        formula: str,
        *,
        x_chord: Optional[torch.Tensor] = None,
        minimal_lam: Optional[float] = None,
        minimal_d_scale: Optional[float] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Exact RoR reference matching each recipe's actual implemented critic penalty formula."""
        d_ref = _make_calibrated_mlp(dim_j, target_gn=0.65, seed=0)
        d_ref.load_state_dict(base_d.state_dict())
        d_ref.zero_grad(set_to_none=True)
        xr = xr_j.clone().requires_grad_(True)
        xf = xf_j.clone().requires_grad_(True)
        yr, yf = d_ref(xr), d_ref(xf)
        gr = torch.autograd.grad(yr.sum(), xr, create_graph=True)[0]
        gf = torch.autograd.grad(yf.sum(), xf, create_graph=True)[0]
        sr = gr.square().sum(dim=-1)
        sf = gf.square().sum(dim=-1)
        s_all = 0.5 * (sr.mean() + sf.mean())
        ml = 0.5 * (yr.mean() + yf.mean())
        if formula == "axiom":
            # Axiom: coeff=1.0, lambda_floor=0.005, gamma_gauge=0.03
            loss = 0.5 * 0.005 * s_all + 0.5 * 0.03 * ml.square()
        elif formula in ("unified_v12", "unified_v13"):
            # Unified v12/v13: ebi_beta=12.0, chord_weight=0.25, lambda_floor=0.005, gamma_gauge=0.03
            rr, rf = torch.sqrt(sr + 1e-12), torch.sqrt(sf + 1e-12)
            cap_r = F.softplus(rr - 1.0, beta=12.0).square().mean()
            cap_f = F.softplus(rf - 1.0, beta=12.0).square().mean()
            if x_chord is not None:
                xc = x_chord.detach().clone().requires_grad_(True)
                yc = d_ref(xc)
                gc = torch.autograd.grad(yc.sum(), xc, create_graph=True)[0]
                sc = gc.square().sum(dim=-1)
                rc = torch.sqrt(sc + 1e-12)
                cap_c = F.softplus(rc - 1.0, beta=12.0).square().mean()
                total = cap_r + 0.75 * cap_f + 0.25 * cap_c
            else:
                total = cap_r + cap_f
            total = total + 0.005 * (sr.mean() + sf.mean())
            loss = 0.5 * total + 0.5 * 0.03 * ml.square()
        elif formula == "minimal":
            # Minimal: ebi_beta=8.0, 0.08 * chord, gauge=0.03*(yr.mean()^2 + yf.mean()^2), lam=coeff*(0.25+0.75*step_scale)
            lam = 1.0 if minimal_lam is None else float(minimal_lam)
            d_scale = float(xr_j.detach().flatten(1).norm(dim=1).mean().item()) if minimal_d_scale is None else float(minimal_d_scale)
            bi = 0.5 * (
                ((torch.sqrt(1.0 + 8.0 * sr) - 1.0) / 8.0).mean()
                + ((torch.sqrt(1.0 + 8.0 * sf) - 1.0) / 8.0).mean()
            )
            chord_dist = (xr_j - xf_j).norm(dim=1).clamp_min(1e-3 * max(1.0, d_scale))
            chord = ((yr.squeeze(-1) - yf.squeeze(-1)) / chord_dist).square().mean()
            gauge = 0.03 * (yr.mean().square() + yf.mean().square())
            loss = lam * (bi + 0.08 * chord) + gauge
        elif formula == "rhbi":
            # RHBI: ebi_beta=20.0, lambda_floor=0.0, gamma_gauge=0.0, chord_weight=0.0
            rr, rf = torch.sqrt(sr + 1e-12), torch.sqrt(sf + 1e-12)
            loss = 0.5 * (F.softplus(rr - 1.0, beta=20.0).square().mean() + F.softplus(rf - 1.0, beta=20.0).square().mean())
        else:
            raise ValueError(formula)
        loss.backward()
        return float(loss.detach().item()), _flatten_pgrads(d_ref.parameters())

    jensen_results: Dict[str, Any] = {}
    for preset in PRESETS:
        rec = pg.get_recipe(preset)
        d_mc = _make_calibrated_mlp(dim_j, target_gn=0.65, seed=0)
        d_mc.load_state_dict(base_d.state_dict())
        g_dum = nn.Linear(2, 2)
        _, opt_d = rec.make_optimizers(g_dum, d_mc)
        pen_fn = rec.make_critic_penalty(opt_d, collect_stats=True)
        # Parallel exact-RoR reference tracker matching each step's exact chord phase / minimal lam:
        exact_losses: List[float] = []
        acc_exact_g: Optional[torch.Tensor] = None
        acc_g: Optional[torch.Tensor] = None
        losses: List[float] = []
        if preset in ("unified_v12", "unified_v13"):
            _, opt_d_shadow = rec.make_optimizers(nn.Linear(2, 2), _make_calibrated_mlp(dim_j, target_gn=0.65, seed=0))
            pen_shadow = rec.make_critic_penalty(opt_d_shadow, collect_stats=False)
        else:
            pen_shadow = None
        cached_static_exact: Optional[Tuple[float, torch.Tensor]] = (
            _exact_ror_penalty_grad(preset) if preset in ("axiom", "rhbi") else None
        )
        for _ in range(n_mc):
            if cached_static_exact is not None:
                ex_l, ex_g = cached_static_exact
            elif preset in ("unified_v12", "unified_v13"):
                assert pen_shadow is not None
                x_chord_i, _ = pen_shadow.regularizer._build_geodesic_chord(xr_j, xf_j, is_conditioned=False)
                ex_l, ex_g = _exact_ror_penalty_grad(preset, x_chord=x_chord_i)
            elif preset == "minimal":
                _, d_scale_m = pen_fn._batch_res(xr_j, xf_j)
                step_scale_m = float(getattr(pen_fn.record, "_step_scale", 1.0))
                lam_m = min(1.0, max(0.01, float(pen_fn.coeff) * (0.25 + 0.75 * step_scale_m)))
                ex_l, ex_g = _exact_ror_penalty_grad(preset, minimal_lam=lam_m, minimal_d_scale=d_scale_m)
            else:
                ex_l, ex_g = _exact_ror_penalty_grad(preset)
            exact_losses.append(ex_l)
            if acc_exact_g is None:
                acc_exact_g = torch.zeros_like(ex_g)
                acc_g = torch.zeros_like(ex_g)
            acc_exact_g.add_(ex_g)
            d_mc.zero_grad(set_to_none=True)
            # Match real GANTrainer sequence: live D(xr), D(xf) run before penalty so full-batch gauge endpoints are reused!
            _ = d_mc(xr_j)
            _ = d_mc(xf_j)
            val = pen_fn(d_mc, xr_j, xf_j)
            val.backward()
            acc_g.add_(_flatten_pgrads(d_mc.parameters()))
            losses.append(float(val.detach().item()))
        assert acc_exact_g is not None and acc_g is not None
        exact_grad = acc_exact_g / float(n_mc)
        exact_loss = sum(exact_losses) / len(exact_losses)
        mean_g = acc_g / float(n_mc)
        mean_l = sum(losses) / len(losses)
        n_ex = float(exact_grad.norm().item())
        n_mc_g = float(mean_g.norm().item())
        cos = float(F.cosine_similarity(mean_g.unsqueeze(0), exact_grad.unsqueeze(0)).item()) if (n_ex > 0 and n_mc_g > 0) else 0.0
        jensen_results[preset] = {
            "exact_loss": exact_loss,
            "mc_mean_loss": mean_l,
            "loss_mc_over_exact": mean_l / max(exact_loss, 1e-30),
            "grad_norm_mc_over_exact": n_mc_g / max(n_ex, 1e-30),
            "grad_cosine_vs_exact": cos,
        }

    # 4B: Dimension-independent equilibrium normal dissipation floor: dim=2 MLP vs dim=196,608 BN-free Conv2d r256
    # Both critics calibrated to exact ||grad_x D|| = 0.25 << kappa = 1.0 (s_true = 0.0625) and zero mean logit!
    dim_floor: Dict[str, Any] = {}
    x2_r = torch.randn(16, 2, generator=torch.Generator().manual_seed(1))
    x2_f = torch.randn(16, 2, generator=torch.Generator().manual_seed(2))
    d2_base = _make_calibrated_mlp(2, target_gn=0.25, seed=0)
    with torch.no_grad():
        d2_base[-1].bias.zero_()
        d2_base[-1].bias.sub_(0.5 * (d2_base(x2_r).mean() + d2_base(x2_f).mean()))
    x2r_req, x2f_req = x2_r.clone().requires_grad_(True), x2_f.clone().requires_grad_(True)
    s_true_2d = 0.5 * (
        float(torch.autograd.grad(d2_base(x2r_req).sum(), x2r_req)[0].square().sum(-1).mean().item())
        + float(torch.autograd.grad(d2_base(x2f_req).sum(), x2f_req)[0].square().sum(-1).mean().item())
    )

    xr_256 = torch.randn(8, 3, 256, 256, device=device, generator=torch.Generator(device=device).manual_seed(1))
    xf_256 = torch.randn(8, 3, 256, 256, device=device, generator=torch.Generator(device=device).manual_seed(2))
    d256_base = _make_bn_free_conv256(device, target_gn=0.25, seed=0)
    with torch.no_grad():
        d256_base[6].bias.sub_(0.5 * (d256_base(xr_256).mean() + d256_base(xf_256).mean()))
    x256r_req, x256f_req = xr_256.clone().requires_grad_(True), xf_256.clone().requires_grad_(True)
    s_true_256 = 0.5 * (
        float(torch.autograd.grad(d256_base(x256r_req).sum(), x256r_req)[0].flatten(1).square().sum(-1).mean().item())
        + float(torch.autograd.grad(d256_base(x256f_req).sum(), x256f_req)[0].flatten(1).square().sum(-1).mean().item())
    )

    for preset in PRESETS:
        rec = pg.get_recipe(preset)
        d2 = _make_calibrated_mlp(2, target_gn=0.25, seed=0)
        d2.load_state_dict(d2_base.state_dict())
        _, od2 = rec.make_optimizers(nn.Linear(2, 2), d2)
        p2_fn = rec.make_critic_penalty(od2, collect_stats=True)
        d2.zero_grad(set_to_none=True)
        _ = d2(x2_r)
        _ = d2(x2_f)
        v2 = p2_fn(d2, x2_r, x2_f)
        v2.backward()
        g2_norm = float(_flatten_pgrads(d2.parameters()).norm().item())
        eff_2d = float(v2.detach().item()) / max(s_true_2d, 1e-12)

        d256 = _make_bn_free_conv256(device, target_gn=0.25, seed=0)
        d256.load_state_dict(d256_base.state_dict())
        _, od256 = rec.make_optimizers(nn.Linear(2, 2).to(device), d256)
        p256_fn = rec.make_critic_penalty(od256, collect_stats=True)
        v256_list: List[float] = []
        g256_list: List[float] = []
        for _ in range(16):
            d256.zero_grad(set_to_none=True)
            _ = d256(xr_256)
            _ = d256(xf_256)
            v256 = p256_fn(d256, xr_256, xf_256)
            v256.backward()
            v256_list.append(float(v256.detach().item()))
            g256_list.append(float(_flatten_pgrads(d256.parameters()).norm().item()))
        eff_256 = (sum(v256_list) / len(v256_list)) / max(s_true_256, 1e-12)
        dim_floor[preset] = {
            "s_true_2d": s_true_2d,
            "s_true_r256": s_true_256,
            "pen_2d": float(v2.detach().item()),
            "grad_norm_2d": g2_norm,
            "eff_coeff_2d": eff_2d,
            "pen_r256": sum(v256_list) / len(v256_list),
            "grad_norm_r256": sum(g256_list) / len(g256_list),
            "eff_coeff_r256": eff_256,
            "r256_over_2d_ratio": eff_256 / max(eff_2d, 1e-30),
        }

    # 4C: EmulatedLinear (mxfp8 & nvfp4), NoJvp (fwAD-failing), and BF16 DCGANCritic survival
    compat: Dict[str, Dict[str, str]] = {}
    x_c_r = torch.randn(8, 24, generator=torch.Generator().manual_seed(3))
    x_c_f = torch.randn(8, 24, generator=torch.Generator().manual_seed(4))
    x_bf_r = torch.randn(4, 3, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(3))
    x_bf_f = torch.randn(4, 3, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(4))
    for preset in PRESETS:
        rec = pg.get_recipe(preset)
        row: Dict[str, str] = {}
        for tag, builder, dev_t, xr_t, xf_t, use_bf16 in (
            ("mxfp8", lambda: nn.Sequential(EmulatedLinear.from_linear(nn.Linear(24, 32), format="mxfp8", seed=0), nn.Tanh(), nn.Linear(32, 1)), torch.device("cpu"), x_c_r, x_c_f, False),
            ("nvfp4", lambda: nn.Sequential(EmulatedLinear.from_linear(nn.Linear(24, 32), format="nvfp4", seed=0), nn.Tanh(), nn.Linear(32, 1)), torch.device("cpu"), x_c_r, x_c_f, False),
            ("no_jvp_attn", lambda: _NoJvpCritic(24), torch.device("cpu"), x_c_r, x_c_f, False),
            ("dcgan_bf16", lambda: DCGANCritic(64, 32, 128).to(device), device, x_bf_r, x_bf_f, device.type == "cuda"),
        ):
            try:
                mod = builder()
                _, od = rec.make_optimizers(nn.Linear(2, 2).to(dev_t), mod)
                pf = rec.make_critic_penalty(od, collect_stats=True)
                mod.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    val = pf(mod, xr_t, xf_t)
                val.backward()
                non_none = [p.grad for p in mod.parameters() if p.grad is not None]
                ok = bool(torch.isfinite(val).all()) and len(non_none) > 0 and all(bool(torch.isfinite(g).all()) for g in non_none)
                row[tag] = "PASS" if ok else "NONFINITE"
            except Exception as exc:
                row[tag] = f"CRASH({type(exc).__name__})"
        compat[preset] = row

    for preset in PRESETS:
        j = jensen_results[preset]
        f = dim_floor[preset]
        c = compat[preset]
        print(
            f"  [4 P6] {preset:12s} | loss_mc/ex={j['loss_mc_over_exact']:7.2f}x "
            f"grad_cos={j['grad_cosine_vs_exact']:.4f} "
            f"eff_2d={f['eff_coeff_2d']:.2e} eff_r256={f['eff_coeff_r256']:.2e} (ratio={f['r256_over_2d_ratio']:.2e}) "
            f"compat=({c['mxfp8']},{c['nvfp4']},{c['no_jvp_attn']},{c['dcgan_bf16']})",
            flush=True,
        )
    return {
        "below_kappa_jensen_d24": jensen_results,
        "dim_indep_floor_2d_vs_r256": dim_floor,
        "quantized_and_fwad_safety": compat,
    }


# ---------------------------------------------------------------------------
# Suite 5: Law 1 (P7 Mass Conservation, Exact LTV, B_f < K Quota, Host Syncs)
# ---------------------------------------------------------------------------

def run_suite5_law1_p7(device: torch.device) -> Dict[str, Any]:
    print("\n=== SUITE 5: Law 1 (P7 Mass Conservation, Exact LTV, B_f < K Quota, Host Syncs) ===", flush=True)

    # 5A: proto_w & w_fake_ema mass conservation under single-outlier & 8-point shock teleportation
    c4 = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]], dtype=torch.float32)
    rng_m = torch.Generator(device="cpu").manual_seed(0)
    warmup_b = [
        c4[torch.randint(0, 4, (128,), generator=rng_m)] + 0.05 * torch.randn(128, 2, generator=rng_m)
        for _ in range(15)
    ]
    shock_1 = torch.cat(
        [c4[torch.randint(0, 4, (127,), generator=rng_m)] + 0.05 * torch.randn(127, 2, generator=rng_m),
         torch.tensor([[80.0, 80.0]])],
        dim=0,
    )
    mass_cons: Dict[str, Any] = {}
    for preset, cls in (
        ("axiom", AxiomPriorRegularizer),
        ("unified_v12", UnifiedV11PriorRegularizer),
        ("unified_v13", UnifiedV11PriorRegularizer),
        ("minimal", MinimalPriorRegularizer),
    ):
        reg = cls(max_prototypes=8)
        max_pw = max_wf = 0.0
        for yb in (*warmup_b, shock_1):
            reg.update_quantizer(yb)
            max_pw = max(max_pw, abs(float(reg.proto_w.sum().item()) - 1.0))
            max_wf = max(max_wf, abs(float(reg.w_fake_ema.sum().item()) - 1.0))
        mass_cons[preset] = {
            "max_proto_w_sum_err": max_pw,
            "max_w_fake_ema_sum_err": max_wf,
            "final_w_fake_ema_sum": float(reg.w_fake_ema.sum().item()),
        }
    mass_cons["rhbi"] = {"note": "MomentTetherRegularizer (no prototype quantizer atlas)"}

    # 5B: Exact Gersho merge & split LTV + 10x scale coalescence guard + post-coalesced psi gauge
    k_m = 16
    centers_m = torch.zeros((k_m, 2), dtype=torch.float32)
    for i in range(k_m):
        centers_m[i, 0] = float(i) * 10.0
    centers_m[1] = torch.tensor([10.0, 0.0])
    centers_m[2] = torch.tensor([11.0, 0.0])
    yr_m = centers_m[3:].repeat(4, 1)
    expected_v_merged = 0.5 * (0.10 + 0.40) + 0.25 * 1.0 / 2.0  # 0.375

    ltv_results: Dict[str, Any] = {}
    for preset, cls in (
        ("axiom", AxiomPriorRegularizer),
        ("unified_v12", UnifiedV11PriorRegularizer),
        ("unified_v13", UnifiedV11PriorRegularizer),
    ):
        reg = cls(max_prototypes=k_m)
        reg.quantizer_initialized = True
        reg.proto_c = centers_m.clone()
        reg.proto_var = torch.full((k_m,), 0.25, dtype=torch.float32)
        reg.proto_var[0], reg.proto_var[1], reg.proto_var[2] = 20.0, 0.10, 0.40
        reg.proto_w = torch.full((k_m,), 1.0 / float(k_m), dtype=torch.float32)
        reg.w_fake_ema = reg.proto_w.clone()
        reg.psi = torch.zeros(k_m, dtype=torch.float32)
        if preset == "axiom":
            reg.v_push = reg.proto_var.clone()
        else:
            reg.proto_hits = torch.full((k_m,), 50.0, dtype=torch.float32)
            reg.v_push = reg.proto_var.unsqueeze(1).expand_as(reg.proto_c).clone()
        reg.update_quantizer(yr_m)
        v1_after = float(reg.proto_var[1].item())
        v0_a, v2_a = float(reg.proto_var[0].item()), float(reg.proto_var[2].item())
        c0_a, c2_a = reg.proto_c[0], reg.proto_c[2]
        split_tot = 0.5 * (v0_a + v2_a) + 0.25 * float((c0_a - c2_a).square().sum().item()) / 2.0
        ltv_results[preset] = {
            "expected_v_merged": expected_v_merged,
            "actual_v_merged": v1_after,
            "merge_ltv_retention": v1_after / expected_v_merged,
            "split_ltv_retention": split_tot / 20.0,
        }
    ltv_results["minimal"] = {"note": "MinimalPriorRegularizer omits Gersho merge-before-split"}
    ltv_results["rhbi"] = {"note": "MomentTetherRegularizer (no prototype quantizer)"}

    # 10x width-ratio coalescence guard + post-coalesced psi zero-mean gauge
    pc_10x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]], dtype=torch.float32)
    pv_10x = torch.tensor([0.01, 1.0, 0.01], dtype=torch.float32)
    pw_10x = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    _, _, _, coal_ax = coalesce_pythagorean_ltv(pc_10x, pv_10x, pw_10x)
    _, _, _, coal_v11 = _compute_pythagorean_smoothed_cells_fast(pc_10x, pv_10x, pw_10x)
    _, _, _, coal_min = _coalesce_isolated(pc_10x, pv_10x, pw_10x)

    # Unequal-weight coalesced psi gauge check using each recipe's actual regularizer + real pullback step
    def _measure_post_coalesced_psi_err(preset_name: str) -> float:
        rec_p = pg.get_recipe(preset_name, num_particles=12, z_dim=2, prior_kind="mog", batch_size=12, standardize=False)
        g_id = SimpleMLPGenerator(2, 16, n_hidden=1, out_dim=2)
        d_id = SimpleMLPDiscriminator(2, 16, n_hidden=1)
        pr_id = MoGParticlePrior(num_particles=12, z_dim=2, sigma=0.025, standardize=False, learnable=True)
        deterministic_orthogonal_(g_id, seed=0)
        deterministic_orthogonal_(d_id, seed=0)
        deterministic_orthogonal_(pr_id, seed=0)
        tr_p = GANTrainer(rec_p, g_id, d_id, prior=pr_id, seed=0)
        reg_p = tr_p.prior_regularizer
        assert reg_p is not None
        reg_p.max_prototypes = 8
        reg_p.quantizer_initialized = True
        reg_p.proto_c = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=torch.float32)
        reg_p.proto_var = torch.full((3,), 0.05, dtype=torch.float32)
        reg_p.proto_w = torch.tensor([0.4, 0.1, 0.5], dtype=torch.float32)
        reg_p.w_fake_ema = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32)
        reg_p.psi = torch.tensor([2.0, -2.0, 0.0], dtype=torch.float32)
        if hasattr(reg_p, "proto_hits"):
            reg_p.proto_hits = torch.full((3,), 50.0, dtype=torch.float32)
        if preset_name == "axiom":
            reg_p.v_push = reg_p.proto_var.clone()
        elif hasattr(reg_p, "v_push"):
            reg_p.v_push = reg_p.proto_var.unsqueeze(1).expand_as(reg_p.proto_c).clone()
        # Keep the exact unequal-weight coalesced prototype geometry while executing the real Laguerre pullback step:
        x_coal = torch.tensor(
            [[0.0, 0.0]] * 5 + [[0.1, 0.0]] * 1 + [[5.0, 0.0]] * 6,
            dtype=torch.float32,
        )
        tr_p.step(x_coal)
        return abs(float(reg_p.psi.mean().item()))

    psi_err_axiom = _measure_post_coalesced_psi_err("axiom")
    psi_err_v12 = _measure_post_coalesced_psi_err("unified_v12")
    psi_err_v13 = _measure_post_coalesced_psi_err("unified_v13")
    psi_err_minimal = _measure_post_coalesced_psi_err("minimal")
    coalesce_gauge = {
        "false_merge_10x_axiom": bool(coal_ax[0].item()),
        "false_merge_10x_v12_v13": bool(coal_v11[0].item()),
        "false_merge_10x_minimal": bool(coal_min[0].item()),
        "post_coalesced_psi_mean_err_axiom": psi_err_axiom,
        "post_coalesced_psi_mean_err_unified_v12": psi_err_v12,
        "post_coalesced_psi_mean_err_unified_v13": psi_err_v13,
        "post_coalesced_psi_mean_err_minimal": psi_err_minimal,
    }

    # 5C: B_f < K (64 < 100) starved high-index mode allocation & permutation invariance
    # Compare actual implemented recipes (axiom, unified_v12, unified_v13, minimal) instead of legacy _t1_coupling.
    gx, gy = torch.meshgrid(torch.linspace(-4.5, 4.5, 10), torch.linspace(-4.5, 4.5, 10), indexing="ij")
    c_100 = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    w_100 = torch.full((100,), 0.01, dtype=torch.float32)
    # 64 particles on charts 0..63 -> charts 64..99 are completely starved (w_fake_ema[64:]=0)
    w_fake_starved = torch.cat([torch.full((64,), 1.0 / 64.0), torch.zeros(36)])
    q_ax_starved = allocate_deficit_quotas(w_100, w_fake_starved, n_slots=64)
    ax_high_allocated = int(q_ax_starved[64:].sum().item())

    # Permutation invariance of Axiom deficit allocation under strict distinct deficits
    perm = torch.randperm(100, generator=torch.Generator().manual_seed(0))
    w_fake_distinct = torch.linspace(0.002, 0.018, 100)
    w_fake_distinct = w_fake_distinct / w_fake_distinct.sum()
    q_orig = allocate_deficit_quotas(w_100, w_fake_distinct, n_slots=64)
    q_perm = allocate_deficit_quotas(w_100[perm], w_fake_distinct[perm], n_slots=64)
    perm_mismatch = int((q_perm != q_orig[perm]).sum().item())

    b_lt_k_actual: Dict[str, Dict[str, Any]] = {}
    for preset_5c in ("axiom", "unified_v12", "unified_v13", "minimal"):
        rec_5c = pg.get_recipe(
            preset_5c, num_particles=64, z_dim=2, prior_kind="mog", batch_size=64, standardize=False
        )
        g_5c = SimpleMLPGenerator(2, 32, n_hidden=2, out_dim=2)
        d_5c = SimpleMLPDiscriminator(2, 32, n_hidden=2)
        pr_5c = MoGParticlePrior(num_particles=64, z_dim=2, sigma=0.025, standardize=False, learnable=True)
        deterministic_orthogonal_(g_5c, seed=0)
        deterministic_orthogonal_(d_5c, seed=0)
        deterministic_orthogonal_(pr_5c, seed=0)
        tr_5c = GANTrainer(rec_5c, g_5c, d_5c, prior=pr_5c, seed=0)
        reg_5c = tr_5c.prior_regularizer
        assert reg_5c is not None
        reg_5c.max_prototypes = 100
        reg_5c.quantizer_initialized = True
        reg_5c.proto_c = c_100.clone()
        reg_5c.proto_var = torch.full((100,), 0.03, dtype=torch.float32)
        reg_5c.proto_w = w_100.clone()
        reg_5c.w_fake_ema = w_fake_starved.clone()
        reg_5c.psi = torch.zeros(100, dtype=torch.float32)
        if hasattr(reg_5c, "proto_hits"):
            reg_5c.proto_hits = torch.full((100,), 50.0, dtype=torch.float32)
        if preset_5c == "axiom":
            reg_5c.v_push = reg_5c.proto_var.clone()
        elif hasattr(reg_5c, "v_push"):
            reg_5c.v_push = reg_5c.proto_var.unsqueeze(1).expand_as(reg_5c.proto_c).clone()
        # Run 3 streaming B_f=64 < K=100 steps cycling across the 100 grid centers:
        g_cyc = torch.Generator().manual_seed(0)
        for _ in range(3):
            idx_r = torch.randperm(100, generator=g_cyc)[:64]
            tr_5c.step(c_100[idx_r])
        high_w_fake = float(reg_5c.w_fake_ema[64:].sum().item())
        high_psi_mean = float(reg_5c.psi[64:].mean().item())
        low_psi_mean = float(reg_5c.psi[:64].mean().item())
        b_lt_k_actual[preset_5c] = {
            "starved_high_idx_w_fake_ema_sum": high_w_fake,
            "starved_high_idx_psi_advantage": high_psi_mean - low_psi_mean,
            "starved_high_idx_active_charts_of_36": int((reg_5c.w_fake_ema[64:] > 1e-4).sum().item()),
        }

    b_lt_k = {
        "axiom_starved_high_idx_allocated_of_36": ax_high_allocated,
        "axiom_distinct_deficit_perm_mismatches": perm_mismatch,
        "actual_recipes_b64_k100": b_lt_k_actual,
    }

    # 5D: Real GANTrainer GPU-CPU host-sync count & ms/step on BOTH grid100 (MLP) and Conv2d 64x64
    g_data = torch.Generator(device=device).manual_seed(0)
    batches_5d = [sample_100gaussians(256, device, generator=g_data) for _ in range(15)]
    host_syncs: Dict[str, Any] = {}
    for preset in PRESETS:
        rec = pg.get_recipe(preset, num_particles=1024, z_dim=2, prior_kind="mog", batch_size=256)
        gen_m = SimpleMLPGenerator(2, 128, n_hidden=3, out_dim=2).to(device)
        crit_m = SimpleMLPDiscriminator(2, 128, n_hidden=3).to(device)
        prior = MoGParticlePrior(num_particles=1024, z_dim=2, sigma=0.025, device=device)
        deterministic_orthogonal_(gen_m, seed=0)
        deterministic_orthogonal_(crit_m, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        trainer = GANTrainer(rec, gen_m, crit_m, prior=prior, seed=0)
        for t in range(5):
            trainer.step(batches_5d[t])
        _sync(device)
        t0 = time.perf_counter()
        with _count_host_syncs(device) as sc:
            for t in range(5, 15):
                trainer.step(batches_5d[t])
        _sync(device)
        ms_step = (time.perf_counter() - t0) * 1000.0 / 10.0
        tot_s = sum(sc.values()) / 10.0
        host_syncs[preset] = {
            "ms_per_step": ms_step,
            "syncs_per_step_total": tot_s,
            "breakdown": {k: v / 10.0 for k, v in sc.items()},
        }
        print(
            f"  [5D MLP Syncs] {preset:12s} | syncs/step={tot_s:5.1f} "
            f"(item={sc['item']/10:.1f} tolist={sc['tolist']/10:.1f} bool={sc['bool']/10:.1f} uniq={sc['unique']/10:.1f} nz={sc['nonzero']/10:.1f}) "
            f"ms/step={ms_step:.2f}",
            flush=True,
        )

    # Also measure Conv2d r64 optimizer+trainer host syncs (where Axiom uses _run_foreach_fast!)
    from experiments.forge.scaling_benchmark import DCGANGenerator
    g_img = torch.Generator(device=device).manual_seed(0)
    batches_img = [torch.randn(16, 3, 64, 64, device=device, generator=g_img) for _ in range(12)]
    conv_syncs: Dict[str, Any] = {}
    for preset in PRESETS:
        rec = pg.get_recipe(preset, num_particles=64, z_dim=32, prior_kind="particles", batch_size=16)
        gen_c = DCGANGenerator(64, 32, 32, 128).to(device)
        crit_c = DCGANCritic(64, 32, 128).to(device)
        prior_c = rec.make_prior().to(device)
        deterministic_orthogonal_(gen_c, seed=0)
        deterministic_orthogonal_(crit_c, seed=0)
        deterministic_orthogonal_(prior_c, seed=0)
        trainer_c = GANTrainer(rec, gen_c, crit_c, prior=prior_c, seed=0)
        for t in range(2):
            trainer_c.step(batches_img[t])
        _sync(device)
        t0 = time.perf_counter()
        with _count_host_syncs(device) as sc_c:
            for t in range(2, 12):
                trainer_c.step(batches_img[t])
        _sync(device)
        ms_c = (time.perf_counter() - t0) * 1000.0 / 10.0
        tot_c = sum(sc_c.values()) / 10.0
        conv_syncs[preset] = {
            "ms_per_step": ms_c,
            "syncs_per_step_total": tot_c,
            "breakdown": {k: v / 10.0 for k, v in sc_c.items()},
        }
        print(
            f"  [5D Conv Syncs]{preset:12s} | syncs/step={tot_c:5.1f} "
            f"(item={sc_c['item']/10:.1f} tolist={sc_c['tolist']/10:.1f} bool={sc_c['bool']/10:.1f} uniq={sc_c['unique']/10:.1f} nz={sc_c['nonzero']/10:.1f}) "
            f"ms/step={ms_c:.2f}",
            flush=True,
        )

    print(
        f"  [5A-C P7] mass_wf_err(ax={mass_cons['axiom']['max_w_fake_ema_sum_err']:.2e}, "
        f"v12={mass_cons['unified_v12']['max_w_fake_ema_sum_err']:.4f}, "
        f"min={mass_cons['minimal']['max_w_fake_ema_sum_err']:.4f}) | "
        f"LTV_merge(ax={ltv_results['axiom']['merge_ltv_retention']:.3f}, v12={ltv_results['unified_v12']['merge_ltv_retention']:.3f}) | "
        f"LTV_split(ax={ltv_results['axiom']['split_ltv_retention']:.3f}, v12={ltv_results['unified_v12']['split_ltv_retention']:.3f}) | "
        f"B<K(ax_quota={ax_high_allocated}/36, ax_live={b_lt_k_actual['axiom']['starved_high_idx_active_charts_of_36']}/36, "
        f"v12_live={b_lt_k_actual['unified_v12']['starved_high_idx_active_charts_of_36']}/36, "
        f"min_live={b_lt_k_actual['minimal']['starved_high_idx_active_charts_of_36']}/36)",
        flush=True,
    )
    return {
        "mass_conservation_shock": mass_cons,
        "gersho_ltv_merge_split": ltv_results,
        "coalescence_guard_and_psi_gauge": coalesce_gauge,
        "b_lt_k_quota_allocation": b_lt_k,
        "gan_trainer_host_syncs": host_syncs,
        "gan_trainer_conv64_host_syncs": conv_syncs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sections", type=str, default="1,2,3,4,5", help="Comma-separated suites to run (1..5)")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write full JSON results")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    chosen = {s.strip() for s in args.sections.split(",") if s.strip()}
    t_all = time.perf_counter()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"Axiom Four-Laws Adversarial Benchmark | device={device} ({dev_name}) | torch={torch.__version__}", flush=True)

    results: Dict[str, Any] = {
        "device": str(device),
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "presets": list(PRESETS),
    }
    if "1" in chosen:
        results["suite1_law42_p5_p0"] = run_suite1_law42_p5_p0(device)
    if "2" in chosen:
        results["suite2_law44_p1"] = run_suite2_law44_p1()
    if "3" in chosen:
        results["suite3_law41_43_p4"] = run_suite3_law41_43_p4()
    if "4" in chosen:
        results["suite4_law2_p6"] = run_suite4_law2_p6(device)
    if "5" in chosen:
        results["suite5_law1_p7"] = run_suite5_law1_p7(device)

    results["total_wall_seconds"] = time.perf_counter() - t_all
    print(f"\nBenchmark complete in {results['total_wall_seconds']:.2f}s.", flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Wrote JSON summary to {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
