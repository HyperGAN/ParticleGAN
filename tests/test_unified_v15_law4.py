"""Layer-1 & Integration Contract Suite for Unified-v15 Law 4 Optimizer (`unified_v15_equil`).

Covers every Law-4 Four-Laws audit defect and semantic control:
  - D-02: Exact FP32 quintic polar for m <= 256 (Tikhonov & s=0), rank-1 sigma_1 <= 1.25 cap,
          and D-13 accurate docstrings.
  - D-03: 1e-25 / 1e-35 scale-safe norms (neural-G & generator-free critic) and 1D invariance
          across fast, slow, fg_damp, and generator-free paths.
  - 4D _conv_layouts dispatch: strict registration dispatch, ungrouped 1x1 Conv2d == 2D Linear
          exact single-step parity (G NS + D sqrt(m) + D Schatten-6), spatial/grouped/transpose
          Conv2d stays 0-NS/0-S6, and zero-sync Schatten-6 Frobenius deadband short-circuit.
  - D-04: C^2-smooth _band_scale with exact 1.0 deadband (0 extra forwards inside deadband),
          verified iterative backtracking on saturating & superlinear maps, mode-matched eval()
          y_pre capture for custom training-masked generators, and scale-homogeneous coord_scale
          + origin_damping at 1e-5x data scale.
  - D-05: Ceiling-free trajectory degree-0 across 1e-12..1e+12 (Defect A), Huber-capped
          1e4x transient spike immunity (Defect B), and live prior peak_rms semantic control (Defect F).
  - Clock-free invariance, fresh-process checkpoint continuation across mechanism boundaries,
    cross-family checkpoint rejection (v11/v12/v14), and tamper rejection.
"""
from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior
from particlegan._unified_v15.critic import record_unified_v15_generator_probe
from particlegan._unified_v15.optimizer import (
    UnifiedV15Optimizer,
    _band_scale,
    _schatten6_spectral_decay_2d,
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward


def _step_gen_loss(opt_g: Any, loss: torch.Tensor) -> None:
    opt_g.zero_grad(set_to_none=True)
    constraint_geometry_backward(loss, opt_g, (loss,))
    opt_g.step()


def _build_v15_pair(
    g_mod: nn.Module,
    d_mod: nn.Module,
    *,
    smoothing: float | None = None,
    num_particles: int = 64,
    z_dim: int = 4,
    batch_size: int = 32,
):
    kw: dict[str, Any] = {"num_particles": num_particles, "z_dim": z_dim, "batch_size": batch_size}
    if smoothing is not None:
        kw["optimizer_smoothing"] = smoothing
    rec = get_recipe("unified_v15", **kw)
    prior = rec.make_prior()
    opt_g, opt_d = rec.make_optimizers(g_mod, d_mod, prior)
    return rec, prior, opt_g, opt_d


# ---------------------------------------------------------------------------
# D-02 & D-13: FP32 m<=256 Quintic Polar, Rank-1 Cap & Docstring Accuracy
# ---------------------------------------------------------------------------

def test_d02_s0_quintic_rank1_frob_cap_and_fp32_m256() -> None:
    """Rank-1 s=0 polar updates must keep sigma_1 <= 1.25 and run m <= 256 in FP32."""
    torch.set_num_threads(1)
    m = 64
    qu, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(201), dtype=torch.float64))
    qv, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(202), dtype=torch.float64))
    g_r1 = torch.outer(qu[:, 0], qv[:, 0]).float()

    # Live optimizer with fg_damp -> s_base=0:
    g_mod = nn.Linear(m, m, bias=False)
    with torch.no_grad():
        g_mod.weight.zero_()
    _, _, opt_g, _ = _build_v15_pair(g_mod, nn.Linear(m, 1, bias=False))
    opt_g._ema_func_gain = 0.15
    lr = float(opt_g.param_groups[0]["lr"])
    _step_gen_loss(opt_g, (g_mod.weight * g_r1).sum())
    s1_opt = float(torch.linalg.svdvals(g_mod.weight.detach().float())[0].item()) / max(lr, 1e-12)
    assert s1_opt <= 1.25, f"optimizer s_base=0 rank-1 sigma_1/lr blew up to {s1_opt:.3f}"

    # Direct 2D and batched s=0 kernels:
    out_2d = quintic_orthogonalize_2d(g_r1, steps=5, smoothing=0.0)
    out_b = quintic_orthogonalize_batched(torch.stack([g_r1, g_r1 * 1e-6], dim=0), steps=5, smoothing=0.0)
    assert float(torch.linalg.svdvals(out_2d.double())[0].item()) <= 1.25
    assert float(torch.linalg.svdvals(out_b[0].double())[0].item()) <= 1.25
    assert torch.allclose(out_b[0], out_b[1], atol=1e-5, rtol=1e-4)

    # Guard against BF16 leakage for m <= 256 & stale D-13 docstrings:
    src_2d = inspect.getsource(quintic_orthogonalize_2d)
    src_b = inspect.getsource(quintic_orthogonalize_batched)
    assert "min(r_dim, c_dim) >= 64" not in src_2d
    assert "min(r_dim, c_dim) >= 64" not in src_b
    assert "m <= 256" in (quintic_orthogonalize_2d.__doc__ or "")
    assert "Schatten-6" not in (UnifiedV15Optimizer._step_generator_free_critic.__doc__ or "")

    if torch.cuda.is_available():
        g_128 = torch.randn((128, 128), generator=torch.Generator().manual_seed(203), dtype=torch.float32)
        cpu_out = quintic_orthogonalize_2d(g_128, steps=5, smoothing=0.0)
        cuda_out = quintic_orthogonalize_2d(g_128.cuda(), steps=5, smoothing=0.0).cpu()
        rel_err = float((cuda_out - cpu_out).norm().item() / max(float(cpu_out.norm().item()), 1e-12))
        assert rel_err < 1e-4, f"CUDA m=128 s=0 polar leaked into BF16 (rel_err={rel_err:.4e})"


# ---------------------------------------------------------------------------
# D-03: 1e-25 / 1e-35 Underflow Protection & 1D Exact Degree-0 Invariance
# ---------------------------------------------------------------------------

def test_d03_1e25_underflow_and_1d_scale_invariance() -> None:
    """peak_rms must stay positive at 1e-25/1e-35 and 1D params must stay degree-0 invariant."""
    torch.set_num_threads(1)
    base_g = torch.randn(16, 16, generator=torch.Generator().manual_seed(301))

    for tiny_scale in (1e-25, 1e-35):
        g_mod = nn.Linear(16, 16, bias=False)
        _, _, opt_g, _ = _build_v15_pair(g_mod, nn.Linear(16, 1, bias=False))
        _step_gen_loss(opt_g, (g_mod.weight * (base_g * tiny_scale)).sum())
        pk_g = opt_g.state[g_mod.weight].get("peak_rms", opt_g.state[g_mod.weight].get("peak_rms_val", 0.0))
        pk_g_val = float(pk_g.item()) if isinstance(pk_g, torch.Tensor) else float(pk_g)
        assert pk_g_val > 0.0, f"neural-G peak_rms underflowed at {tiny_scale}"

        d_gf = nn.Linear(16, 16, bias=False)
        _, _, _, opt_d_gf = _build_v15_pair(nn.Identity(), d_gf)
        d_gf.weight.grad = base_g * tiny_scale
        opt_d_gf.step()
        pk_d_val = float(opt_d_gf.state[d_gf.weight]["peak_rms"].item())
        assert pk_d_val > 0.0, f"generator-free critic peak_rms underflowed at {tiny_scale}"

    # 1D vector invariance across all 3 paths down to 1e-25:
    v_grad = torch.linspace(0.5, 1.5, 16)
    for mode in ("gen_free_default", "s0_fast_path", "fg_damp_default"):
        moves = {}
        for scale in (1e-6, 1e-12, 1e-25):
            p = nn.Parameter(torch.zeros(16))
            if mode == "gen_free_default":
                dm = nn.Module()
                dm.register_parameter("bias_1d", p)
                _, _, _, opt = _build_v15_pair(nn.Identity(), dm)
                p.grad = v_grad * scale
                opt.step()
            elif mode == "s0_fast_path":
                gm = nn.Module()
                gm.register_parameter("bias_1d", p)
                _, _, opt, _ = _build_v15_pair(gm, nn.Linear(4, 1), smoothing=0.0)
                _step_gen_loss(opt, (p * (v_grad * scale)).sum())
            else:
                gm = nn.Module()
                gm.register_parameter("bias_1d", p)
                _, _, opt, _ = _build_v15_pair(gm, nn.Linear(4, 1))
                opt._ema_func_gain = 0.15
                _step_gen_loss(opt, (p * (v_grad * scale)).sum())
            moves[scale] = float(p.detach().norm().item())
        assert abs(moves[1e-12] / moves[1e-6] - 1.0) < 1e-4, f"{mode}: 1e-12 ratio={moves[1e-12] / moves[1e-6]}"
        assert abs(moves[1e-25] / moves[1e-6] - 1.0) < 1e-4, f"{mode}: 1e-25 ratio={moves[1e-25] / moves[1e-6]}"


# ---------------------------------------------------------------------------
# 4D _conv_layouts Dispatch, 1x1 Conv2d == 2D Linear Parity & Schatten-6
# ---------------------------------------------------------------------------

def test_4d_conv1x1_equals_2d_linear_single_step_parity_and_schatten6() -> None:
    """Ungrouped 1x1 Conv2d must match 2D Linear bit-closely on G, neural-G D (incl. active S6), and gen-free D."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(601)
    w_small = torch.randn((128, 128), generator=g) * 0.05
    w_large = torch.randn((128, 128), generator=g) * 0.50  # sigma_1 > tau=2.0 -> active Schatten-6!
    grad0 = torch.randn((128, 128), generator=g) * 0.02

    for w0, expect_s6_active in ((w_small, False), (w_large, True)):
        g_lin, d_lin = nn.Linear(128, 128, bias=False), nn.Linear(128, 128, bias=False)
        g_c1, d_c1 = nn.Conv2d(128, 128, 1, bias=False), nn.Conv2d(128, 128, 1, bias=False)
        with torch.no_grad():
            g_lin.weight.copy_(w0)
            d_lin.weight.copy_(w0)
            g_c1.weight.copy_(w0.view(128, 128, 1, 1))
            d_c1.weight.copy_(w0.view(128, 128, 1, 1))

        _, _, ogl, odl = _build_v15_pair(g_lin, d_lin)
        _, _, ogc, odc = _build_v15_pair(g_c1, d_c1)

        _step_gen_loss(ogl, (g_lin.weight * grad0).sum())
        _step_gen_loss(ogc, (g_c1.weight * grad0.view(128, 128, 1, 1)).sum())
        d_lin.weight.grad = grad0.clone()
        odl.step()
        d_c1.weight.grad = grad0.view(128, 128, 1, 1).clone()
        odc.step()

        dg_l = g_lin.weight.detach() - w0
        dg_c = g_c1.weight.detach().view(128, 128) - w0
        dd_l = d_lin.weight.detach() - w0
        dd_c = d_c1.weight.detach().view(128, 128) - w0

        rel_g = float((dg_c - dg_l).norm().item() / max(float(dg_l.norm().item()), 1e-12))
        rel_d = float((dd_c - dd_l).norm().item() / max(float(dd_l.norm().item()), 1e-12))
        assert rel_g < 1e-6, f"G 1x1 Conv2d vs Linear gap={rel_g:.3e}"
        assert rel_d < 1e-6, f"D 1x1 Conv2d vs Linear gap={rel_d:.3e}"
        assert ogc.ns_2d_activations == 1
        if expect_s6_active:
            assert odl.s6_skipped_deadband == 0 and odc.s6_skipped_deadband == 0

    # Generator-free critic 1x1 Conv2d == 2D Linear parity:
    d_gf_l = nn.Linear(64, 64, bias=False)
    d_gf_c = nn.Conv2d(64, 64, 1, bias=False)
    w64 = torch.randn((64, 64), generator=torch.Generator().manual_seed(602)) * 0.05
    g64 = torch.randn((64, 64), generator=torch.Generator().manual_seed(603)) * 0.02
    with torch.no_grad():
        d_gf_l.weight.copy_(w64)
        d_gf_c.weight.copy_(w64.view(64, 64, 1, 1))
    _, _, _, od_gf_l = _build_v15_pair(nn.Identity(), d_gf_l)
    _, _, _, od_gf_c = _build_v15_pair(nn.Identity(), d_gf_c)
    d_gf_l.weight.grad = g64.clone()
    od_gf_l.step()
    d_gf_c.weight.grad = g64.view(64, 64, 1, 1).clone()
    od_gf_c.step()
    rel_gf = float(
        ((d_gf_c.weight.detach().view(64, 64) - w64) - (d_gf_l.weight.detach() - w64)).norm().item()
        / max(float((d_gf_l.weight.detach() - w64).norm().item()), 1e-12)
    )
    assert rel_gf < 1e-6 and od_gf_c.critic_ns_calls == 1


def test_spatial_conv_stays_zero_ns_nonconv_4d_uses_2d_ns_and_s6_deadband_skips() -> None:
    """Spatial k>1 / grouped / transpose Conv2d keep 0 NS; unregistered 4D uses 2D NS; S6 skips <= tau."""
    torch.set_num_threads(1)
    for mod in (
        nn.Conv2d(16, 16, kernel_size=3, bias=False),
        nn.Conv2d(16, 16, kernel_size=1, groups=2, bias=False),
        nn.ConvTranspose2d(16, 16, kernel_size=1, bias=False),
    ):
        _, _, og, _ = _build_v15_pair(mod, nn.Conv2d(16, 1, kernel_size=3, bias=False))
        _step_gen_loss(og, mod.weight.sum())
        assert og.ns_2d_activations == 0, f"{mod} unexpectedly triggered 2D NS"

    # Unregistered non-conv 4D tensor MUST NOT be misclassified as Conv2d:
    m_4d = nn.Module()
    m_4d.register_parameter("attn_4d", nn.Parameter(torch.randn(8, 8, 2, 2)))
    _, _, og_4d, _ = _build_v15_pair(m_4d, nn.Linear(4, 1))
    _step_gen_loss(og_4d, m_4d.attn_4d.sum())
    assert og_4d.ns_2d_activations == 1

    # Schatten-6 deadband short-circuit on small-Frobenius critic weight vs active decay above tau:
    d_small = nn.Linear(16, 16, bias=False)
    with torch.no_grad():
        d_small.weight.mul_(0.05)  # ||W||_F << 2.0
    _, _, _, od_small = _build_v15_pair(nn.Linear(16, 16), d_small)
    d_small.weight.grad = torch.randn(16, 16, generator=torch.Generator().manual_seed(604)) * 0.01
    od_small.step()
    assert od_small.s6_skipped_deadband == 1

    w_in = torch.eye(16) * 0.25  # ||W||_F = 1.0 <= tau=2.0
    assert torch.equal(_schatten6_spectral_decay_2d(w_in, torch.tensor(0.02)), torch.zeros_like(w_in))
    w_hi = torch.eye(16) * 3.5  # sigma_1 = 3.5 > tau=2.0
    disp_hi = _schatten6_spectral_decay_2d(w_hi, torch.tensor(0.02))
    assert float(torch.linalg.svdvals(w_hi + disp_hi)[0].item()) < float(torch.linalg.svdvals(w_hi)[0].item())


# ---------------------------------------------------------------------------
# D-04: C^2 _band_scale, Verified Backtracking, Mode-Matched Probes & 1e-5x Scale
# ---------------------------------------------------------------------------

def test_d04_c2_band_scale_verified_backtracking_and_zero_extra_deadband_forwards() -> None:
    """_band_scale must be C^2 at both boundaries, 0 extra forwards in deadband, and 0 nonlinear overshoots."""
    torch.set_num_threads(1)
    eps_fd = 1e-5
    d_hi_l = (_band_scale(0.20, 0.0, 0.20) - _band_scale(0.20 - eps_fd, 0.0, 0.20)) / eps_fd
    d_hi_r = (_band_scale(0.20 + eps_fd, 0.0, 0.20) - _band_scale(0.20, 0.0, 0.20)) / eps_fd
    d_lo_l = (_band_scale(0.05, 0.05, 0.20) - _band_scale(0.05 - eps_fd, 0.05, 0.20)) / eps_fd
    d_lo_r = (_band_scale(0.05 + eps_fd, 0.05, 0.20) - _band_scale(0.05, 0.05, 0.20)) / eps_fd
    assert abs(d_hi_l - d_hi_r) < 1e-3
    assert abs(d_lo_l - d_lo_r) < 1e-3
    assert _band_scale(0.10, 0.05, 0.20) == 1.0

    # Saturating tanh overshoot check (both knee-crossing and concave saturation):
    class _SatTanh(nn.Module):
        def __init__(self, w0: float) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([[w0, 0.0], [0.0, w0]], dtype=torch.float32))
            self.eval_calls = 0

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            if not self.training:
                self.eval_calls += 1
            return 2.0 * torch.tanh(z @ self.w.T)

    for w0, lr_t in ((3.5, 3.6), (0.0, 10.0), (1.0, 5.0)):
        gm = _SatTanh(w0)
        _, _, og, od = _build_v15_pair(gm, nn.Linear(2, 1), smoothing=0.0)
        rec_o = od.record
        og._unified_v11_record = rec_o
        rec_o.ema_data_scale = math.sqrt(2.0)
        rec_o.ema_push_res = 0.0
        rec_o._thermal_scale = 1.0
        zp = torch.ones((16, 2))
        yp = gm(zp)
        record_unified_v15_generator_probe(gm, (zp,), {}, yp, rec_o, require_module_training=True)
        og.param_groups[0]["lr"] = lr_t
        _step_gen_loss(og, gm.w.sum())
        with torch.no_grad():
            ypost = gm(zp)
        realized = float((ypost - yp.detach()).square().mean().sqrt().item())
        assert realized <= 1.10 * 0.045, f"saturating overshoot w0={w0} lr={lr_t}: realized={realized:.4f}"

    # Superlinear cubic amplification wakeup (alpha > 1 must be verified & contracted if it overshoots!):
    class _SuperCubic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([[0.05, 0.0], [0.0, 0.05]], dtype=torch.float32))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return (z @ self.w.T).pow(3)

    gc = _SuperCubic()
    _, _, ogc, odc = _build_v15_pair(gc, nn.Linear(2, 1), smoothing=0.0)
    rec_c = odc.record
    ogc._unified_v11_record = rec_c
    rec_c.ema_data_scale = math.sqrt(2.0)
    rec_c.ema_push_res = 0.60  # Activates delta_g_min wakeup!
    rec_c._thermal_scale = 1.0
    zp = torch.ones((16, 2))
    yp = gc(zp)
    record_unified_v15_generator_probe(gc, (zp,), {}, yp, rec_c, require_module_training=True)
    ogc.param_groups[0]["lr"] = 0.03
    _step_gen_loss(ogc, (-gc.w).sum())
    delta_g_max_c = 0.045 + 0.16 * 0.60
    with torch.no_grad():
        ypost_c = gc(zp)
    realized_c = float((ypost_c - yp.detach()).square().mean().sqrt().item())
    assert realized_c <= 1.10 * delta_g_max_c, f"superlinear amplification overshot: {realized_c:.4f} > {delta_g_max_c:.4f}"

    # Semantic control: inside deadband (alpha == 1.0), BN generator performs ZERO extra backtracking forwards:
    gm_bn = nn.Sequential(nn.Linear(2, 2, bias=False), nn.BatchNorm1d(2))
    with torch.no_grad():
        gm_bn[0].weight.copy_(torch.eye(2))
    gm_bn.train()
    _, _, ogb, odb = _build_v15_pair(gm_bn, nn.Linear(2, 1), smoothing=0.0)
    rec_b = odb.record
    ogb._unified_v11_record = rec_b
    rec_b.ema_data_scale = math.sqrt(2.0)
    rec_b.ema_push_res = 0.0
    rec_b._thermal_scale = 1.0
    zp_b = torch.randn((16, 2), generator=torch.Generator().manual_seed(405))
    yp_b = gm_bn(zp_b)
    record_unified_v15_generator_probe(gm_bn, (zp_b,), {}, yp_b, rec_b, require_module_training=True)
    calls = {"n": 0}
    orig_fwd = gm_bn.forward

    def _counted_fwd(*a: Any, **kw: Any) -> torch.Tensor:
        calls["n"] += 1
        return orig_fwd(*a, **kw)

    gm_bn.forward = _counted_fwd  # type: ignore[method-assign]
    ogb.param_groups[0]["lr"] = 0.005  # Small step strictly inside deadband
    _step_gen_loss(ogb, gm_bn[0].weight.sum())
    assert ogb.last_step_gain == 1.0
    assert calls["n"] == 1, f"expected 1 probe forward inside deadband, got {calls['n']}"


def test_d04_mode_matched_eval_probe_and_1e5_scale_homogeneity() -> None:
    """Custom training-masked generator must not crush ema_func_gain; 1e-5x origin_damping must not crush G."""
    torch.set_num_threads(1)

    class _MaskedGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 16, bias=False)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            out = torch.tanh(self.lin(z))
            if self.training:
                mask = torch.zeros_like(out)
                mask[:, 0] = 1.0  # 94% zeros during train(), 0% zeros during eval()
                return out * mask
            return out

    gm = _MaskedGen()
    _, _, og, od = _build_v15_pair(gm, nn.Linear(16, 1), smoothing=0.0)
    rec_o = od.record
    og._unified_v11_record = rec_o
    rec_o.ema_data_scale = 2.0
    rec_o.ema_push_res = 0.0
    rec_o._thermal_scale = 1.0
    z = torch.randn((16, 4), generator=torch.Generator().manual_seed(410))
    y_train = gm(z)
    record_unified_v15_generator_probe(gm, (z,), {}, y_train, rec_o, require_module_training=True)
    og.param_groups[0]["lr"] = 0.002
    _step_gen_loss(og, (gm.lin.weight * 0.01).sum())
    assert og._ema_func_gain == 1.0, f"train/eval mask mismatch crushed ema_func_gain to {og._ema_func_gain}"

    # Scale-homogeneous origin_damping at step > 80 on 1e-5x data scale:
    g1 = nn.Linear(4, 2, bias=False)
    g2 = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        g2.weight.copy_(g1.weight)
    _, _, og1, od1 = _build_v15_pair(g1, nn.Linear(2, 1), smoothing=0.0)
    _, _, og2, od2 = _build_v15_pair(g2, nn.Linear(2, 1), smoothing=0.0)
    og1._unified_v11_record = od1.record
    og2._unified_v11_record = od2.record
    od1.record.ema_data_scale = 1.0
    od2.record.ema_data_scale = 1e-5
    og1.state[g1.weight]["step"] = 100
    og2.state[g2.weight]["step"] = 100
    og1.bind_origin_sample(torch.ones((16, 2)) * 1.0)
    og2.bind_origin_sample(torch.ones((16, 2)) * 1e-5)
    grad_t = torch.randn((2, 4), generator=torch.Generator().manual_seed(411))
    _step_gen_loss(og1, (g1.weight * grad_t).sum())
    _step_gen_loss(og2, (g2.weight * (grad_t * 1e-5)).sum())
    assert torch.allclose(g1.weight, g2.weight, atol=1e-6, rtol=1e-5), "origin_damping broke 1e-5x scale invariance"


# ---------------------------------------------------------------------------
# D-05: Defects A (24-decade degree-0), B (1e4x spike recovery), F (live prior)
# ---------------------------------------------------------------------------

def test_d05_defects_a_b_f_ceiling_free_spike_immune_and_prior_live() -> None:
    """Verify exact 24-decade degree-0 (A), 1e4x spike immunity (B), and live prior peak_rms (F)."""
    torch.set_num_threads(1)

    # Defect A: network weights across 24 decades (1e-12 .. 1e+12):
    g_seq = [
        torch.randn(8, 8, generator=torch.Generator().manual_seed(510 + t)) * (0.2 ** (t // 3))
        for t in range(8)
    ]

    def _run_net_scale(scale: float) -> torch.Tensor:
        gm = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            gm.weight.copy_(torch.eye(8))
        _, _, og, _ = _build_v15_pair(gm, nn.Linear(8, 1))
        for gt in g_seq:
            _step_gen_loss(og, (gm.weight * (gt * scale)).sum())
        return gm.weight.detach().clone()

    w_ref = _run_net_scale(1.0)
    for sc in (1e-12, 1e-6, 1e4, 1e12):
        diff = float((_run_net_scale(sc) - w_ref).abs().max().item())
        assert diff < 1e-4, f"Defect A broken at scale={sc}: max_diff={diff:.4e}"

    # Defect B: single 1e4x transient spike at BOTH 1e-6 and 1.0 must not freeze subsequent steps:
    bg = torch.randn(8, 8, generator=torch.Generator().manual_seed(520))
    for base_sc in (1e-6, 1.0):
        gm = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            gm.weight.copy_(torch.eye(8))
        _, _, og, _ = _build_v15_pair(gm, nn.Linear(8, 1))
        for _ in range(3):
            _step_gen_loss(og, (gm.weight * (bg * base_sc)).sum())
        w_pre = gm.weight.detach().clone()
        _step_gen_loss(og, (gm.weight * (bg * base_sc)).sum())
        pre_step = float((gm.weight.detach() - w_pre).norm().item())
        # Inject single 1e4x spike:
        _step_gen_loss(og, (gm.weight * (bg * (base_sc * 1e4))).sum())
        post_step = 0.0
        for _ in range(100):
            w_b = gm.weight.detach().clone()
            _step_gen_loss(og, (gm.weight * (bg * base_sc)).sum())
            post_step = float((gm.weight.detach() - w_b).norm().item())
        ratio = post_step / max(pre_step, 1e-12)
        assert ratio >= 0.80, f"Defect B spike freeze at scale={base_sc}: post/pre={ratio:.4f}"

    # Defect F: prior peak_rms live usage + semantic mutation control + prior spike immunity:
    rec = get_recipe("unified_v15", num_particles=256, z_dim=16, batch_size=128)
    prior = MoGParticlePrior(256, 16, sigma=0.1, standardize=False, learnable=True)
    init.deterministic_orthogonal_(prior, seed=0)
    og_p, _ = rec.make_optimizers(nn.Linear(16, 16), nn.Linear(16, 1), prior)
    z0 = prior.z.detach().clone()
    rows = torch.arange(256, dtype=torch.long)
    gt = torch.randn(256, 16, generator=torch.Generator().manual_seed(501)) * 1e-5
    og_p.set_sampled_rows(prior.z, rows)
    _step_gen_loss(og_p, (prior.z * gt).sum())
    mean_dz = float((prior.z.detach() - z0).norm(dim=1).mean().item())
    assert mean_dz >= 0.01

    z_a = prior.z.detach().clone()
    og_p.set_sampled_rows(prior.z, rows)
    _step_gen_loss(og_p, (prior.z * (gt * 0.1)).sum())
    disp_norm = float((prior.z.detach() - z_a).norm(dim=1).mean().item())

    with torch.no_grad():
        prior.z.copy_(z_a)
        og_p.state[prior.z]["peak_rms"].fill_(1e-2)
    og_p.set_sampled_rows(prior.z, rows)
    _step_gen_loss(og_p, (prior.z * (gt * 0.1)).sum())
    disp_mut = float((prior.z.detach() - z_a).norm(dim=1).mean().item())
    assert abs(disp_norm - disp_mut) > 1e-3, "prior peak_rms state mutation had no effect"


# ---------------------------------------------------------------------------
# Clock-Free Invariance, Fresh-Process Continuation & Checkpoint Integrity
# ---------------------------------------------------------------------------

def test_clock_free_invariance_v15() -> None:
    """Changing total_steps / max_steps (40 vs 4000) must not alter early unified_v15 updates."""
    torch.set_num_threads(1)

    def _run(horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = get_recipe(
            "unified_v15",
            num_particles=32,
            z_dim=4,
            batch_size=32,
            prior_kind="mog",
            standardize=False,
            total_steps=horizon,
        )
        g_mod = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
        d_mod = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        prior = MoGParticlePrior(32, 4, sigma=0.1, standardize=False, learnable=True)
        init.deterministic_orthogonal_(g_mod, seed=10)
        init.deterministic_orthogonal_(d_mod, seed=20)
        init.deterministic_orthogonal_(prior, seed=30)
        tr = GANTrainer(rec, g_mod, d_mod, prior=prior, seed=0, max_steps=horizon)
        rng = torch.Generator().manual_seed(1234)
        for _ in range(6):
            tr.step(torch.randn((32, 2), generator=rng))
        return (
            torch.cat([p.detach().reshape(-1) for p in g_mod.parameters()]),
            torch.cat([p.detach().reshape(-1) for p in d_mod.parameters()]),
            unwrap_prior(tr.prior).z.detach().clone(),
        )

    gs, ds, zs = _run(40)
    gl, dl, zl = _run(4000)
    assert torch.allclose(gs, gl, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ds, dl, atol=1e-6, rtol=1e-6)
    assert torch.allclose(zs, zl, atol=1e-6, rtol=1e-6)


def test_fresh_process_checkpoint_continuation_across_boundary(tmp_path: Path) -> None:
    """Saving across step 80 (origin_damping boundary) and resuming in a fresh subprocess must be bit-exact."""
    ckpt_path = tmp_path / "v15_ckpt.pt"
    ref_path = tmp_path / "v15_ref.pt"
    resumed_path = tmp_path / "v15_resumed.pt"

    script = f"""
import torch
from torch import nn
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior

torch.set_num_threads(1)

def make_tr():
    rec = get_recipe("unified_v15", num_particles=32, z_dim=4, batch_size=32, prior_kind="mog", standardize=False)
    g = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
    d = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    pr = MoGParticlePrior(32, 4, sigma=0.1, standardize=False, learnable=True)
    init.deterministic_orthogonal_(g, seed=11)
    init.deterministic_orthogonal_(d, seed=21)
    init.deterministic_orthogonal_(pr, seed=31)
    return GANTrainer(rec, g, d, prior=pr, seed=0, max_steps=120)

rng = torch.Generator().manual_seed(5678)
batches = [torch.randn((32, 2), generator=rng) for _ in range(86)]

mode = "{ '{MODE}' }"
"""
    # Phase 1: run uninterrupted 86 steps + save split checkpoint at step 78 (crosses step-80 boundary after reload):
    phase1 = script.replace("{MODE}", "phase1") + f"""
tr_full = make_tr()
tr_split = make_tr()
for i in range(78):
    tr_full.step(batches[i])
    tr_split.step(batches[i])
torch.save(tr_split.state_dict(), {str(ckpt_path)!r})
for i in range(78, 86):
    tr_full.step(batches[i])
torch.save({{
    "g": [p.detach().clone() for p in tr_full.G.parameters()],
    "d": [p.detach().clone() for p in tr_full.D.parameters()],
    "z": unwrap_prior(tr_full.prior).z.detach().clone(),
}}, {str(ref_path)!r})
"""
    phase2 = script.replace("{MODE}", "phase2") + f"""
tr_res = make_tr()
tr_res.load_state_dict(torch.load({str(ckpt_path)!r}, weights_only=False))
for i in range(78, 86):
    tr_res.step(batches[i])
torch.save({{
    "g": [p.detach().clone() for p in tr_res.G.parameters()],
    "d": [p.detach().clone() for p in tr_res.D.parameters()],
    "z": unwrap_prior(tr_res.prior).z.detach().clone(),
}}, {str(resumed_path)!r})
"""
    subprocess.run([sys.executable, "-c", phase1], check=True)
    subprocess.run([sys.executable, "-c", phase2], check=True)

    ref = torch.load(ref_path, weights_only=False)
    res = torch.load(resumed_path, weights_only=False)
    for pa, pb in zip(ref["g"], res["g"]):
        assert torch.equal(pa, pb)
    for pa, pb in zip(ref["d"], res["d"]):
        assert torch.equal(pa, pb)
    assert torch.equal(ref["z"], res["z"])


def test_checkpoint_cross_family_and_tamper_rejection() -> None:
    """UnifiedV15Optimizer must reject v11/v12/v14 checkpoints and tampered state fields."""
    torch.set_num_threads(1)
    g15, d15 = nn.Linear(4, 2), nn.Linear(2, 1)
    _, _, og15, od15 = _build_v15_pair(g15, d15)
    _step_gen_loss(og15, g15.weight.sum())
    d15.weight.grad = torch.ones_like(d15.weight)
    od15.step()
    valid_g_sd = og15.state_dict()

    for old_name in ("unified_v11", "unified_v12", "unified_v14"):
        rec_old = get_recipe(old_name, num_particles=64, z_dim=4, batch_size=32)
        pr_old = rec_old.make_prior()
        og_old, od_old = rec_old.make_optimizers(nn.Linear(4, 2), nn.Linear(2, 1), pr_old)
        with pytest.raises(ValueError):
            og15.load_state_dict(og_old.state_dict())
        with pytest.raises(ValueError):
            od15.load_state_dict(od_old.state_dict())

    # Tampered metadata / state values must fail validation cleanly:
    bad_schema = copy.deepcopy(valid_g_sd)
    bad_schema["unified_v15_equil"]["schema"] = 99
    with pytest.raises(ValueError):
        og15.load_state_dict(bad_schema)

    bad_fg = copy.deepcopy(valid_g_sd)
    bad_fg["unified_v15_equil"]["ema_func_gain"] = 0.0
    with pytest.raises(ValueError):
        og15.load_state_dict(bad_fg)

    bad_flip = copy.deepcopy(valid_g_sd)
    bad_flip["unified_v15_equil"]["ema_flip_cos"] = 1.5
    with pytest.raises(ValueError):
        og15.load_state_dict(bad_flip)

    bad_peak = copy.deepcopy(valid_g_sd)
    first_k = next(iter(bad_peak["state"]))
    bad_peak["state"][first_k]["peak_rms_val"] = -1.0
    with pytest.raises(ValueError):
        og15.load_state_dict(bad_peak)


def test_zero_forbidden_primitives_and_frob_smoothing_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """UnifiedV15Optimizer must never invoke svd/eigh/pinv/autograd.grad and must pass 64x64 spike recovery."""
    torch.set_num_threads(1)

    def _forbid(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden primitive called inside UnifiedV15Optimizer")

    monkeypatch.setattr(torch.linalg, "svd", _forbid)
    monkeypatch.setattr(torch, "svd", _forbid)
    monkeypatch.setattr(torch.linalg, "eigh", _forbid)
    monkeypatch.setattr(torch.linalg, "pinv", _forbid)
    monkeypatch.setattr(torch, "pinverse", _forbid)
    monkeypatch.setattr(torch.autograd, "grad", _forbid)

    g_mod = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 8))
    d_mod = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 1))
    _, prior, opt_g, opt_d = _build_v15_pair(g_mod, d_mod)
    gen = torch.Generator().manual_seed(777)

    # Neural-G + Critic + Prior step (including Schatten-6 above deadband tau=2.0):
    with torch.no_grad():
        d_mod[0].weight.mul_(5.0)
    z_rows = torch.arange(16)
    opt_g.set_sampled_rows(prior.z, z_rows)
    w_g = [torch.randn(p.shape, generator=gen) * 1e-2 for p in g_mod.parameters()]
    w_p = torch.randn((16, prior.z.shape[1]), generator=gen) * 1e-3
    loss_g = sum((p * w).sum() for p, w in zip(g_mod.parameters(), w_g)) + (prior.z[z_rows] * w_p).sum()
    _step_gen_loss(opt_g, loss_g)
    for p in d_mod.parameters():
        p.grad = torch.randn(p.shape, generator=gen) * 1e-2
    opt_d.step()
    assert opt_g.ns_5step_calls >= 2

    # Generator-free critic step:
    d_gf = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))
    _, _, _, opt_d_gf = _build_v15_pair(nn.Identity(), d_gf)
    for p in d_gf.parameters():
        p.grad = torch.randn(p.shape, generator=gen) * 1e-2
    opt_d_gf.step()
    assert opt_d_gf.critic_ns_calls >= 1

    # 64x64 Frobenius-matched D-05 spike recovery (post/pre >= 0.90 after single 1e4x spike):
    lin64 = nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        lin64.weight.copy_(torch.eye(64))
    _, _, opt_g64, _ = _build_v15_pair(lin64, nn.Linear(64, 1))
    base_g = torch.randn(64, 64, generator=torch.Generator().manual_seed(778))
    for _ in range(3):
        _step_gen_loss(opt_g64, (lin64.weight * (base_g * 1e-6)).sum())
    w0 = lin64.weight.detach().clone()
    _step_gen_loss(opt_g64, (lin64.weight * (base_g * 1e-6)).sum())
    pre = float((lin64.weight.detach() - w0).norm().item())
    _step_gen_loss(opt_g64, (lin64.weight * (base_g * 1e-2)).sum())  # 1e4x spike
    w1 = lin64.weight.detach().clone()
    _step_gen_loss(opt_g64, (lin64.weight * (base_g * 1e-6)).sum())
    post = float((lin64.weight.detach() - w1).norm().item())
    assert post / max(pre, 1e-12) >= 0.90, f"64x64 spike recovery too low: {post / max(pre, 1e-12):.4f}"
