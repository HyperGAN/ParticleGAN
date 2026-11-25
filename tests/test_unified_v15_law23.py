"""Unit, invariance, negative-control, and checkpoint-continuation tests for unified_v15 Law 2 & Law 3.

Covers:
- L2-4: Convex-EBI Jensen bias below kappa (exact RoR for dim < 32768 + debiased fwAD for >=128^2).
- L2-5: Regime dispatch (GroupNorm/LayerNorm/Attention/small-res -> exact RoR with SDPBackend.MATH;
        >=128^2 pure Conv2d DCGAN -> debiased fwAD) and D-11 b=1 guard.
- D-09: Dimension-independent normal-axis dissipation floor (non-vanishing on r256) + negative control.
- D-04 (critic side): scale-homogeneous observe_scale at 1e-3 & 1e-5, stratified tail-inclusive probe
        eliminating b_p=2 prefix blindness, and eval-mode baseline parity for custom `self.training` generators.
- D-08: Screening dead-zone alignment in v15_zero_eigh_bipartite_deficit_macro_loss + covered-broad-mode guard.
- L3-7: Haar-O(32) rotation invariance, [length]^1 scale equivariance, equilibrium zero-bias, and
        high-D sparse hollow P_xx protection.
- Full bit-exact checkpoint save/restore continuation (fresh object + fresh subprocess) and
  strict rejection of incompatible v11..v14 / tampered checkpoints, plus clock-free invariance.
"""
from __future__ import annotations

import copy
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior
from particlegan._t3_dynamics.metric import born_infeld_c_hat_detached, sinkhorn_slack
from particlegan._unified_v15.critic import (
    UnifiedV15HighDBasinBank,
    UnifiedV15SparseObsBank,
    UnifiedV15StepRecord,
    record_unified_v15_generator_probe,
    validate_v15_highd_bank_packet,
    validate_v15_sparse_obs_bank_packet,
)
from particlegan._unified_v15.loss import (
    UnifiedV15CouplingLoss,
    unified_v15_highd_basin_stein_loss,
    unified_v15_sparse_obs_wiener_loss,
    v15_zero_eigh_bipartite_deficit_macro_loss,
)
from particlegan._unified_v15.penalty import (
    _debiased_fwad_sq,
    _prepare_batched_highd_tangent_plans,
)


class _LinearImgCritic(nn.Module):
    def __init__(self, shape: tuple[int, int, int] = (3, 16, 16), norm_val: float = 0.80) -> None:
        super().__init__()
        d = shape[0] * shape[1] * shape[2]
        u = torch.randn(d, generator=torch.Generator().manual_seed(777))
        self.w = nn.Parameter(u / u.norm() * float(norm_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1) @ self.w


class _PureConvDCGAN128Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 4, stride=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 8, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1).squeeze(-1)


class _GroupNormAttentionCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 8, stride=8)
        self.gn = nn.GroupNorm(2, 8)
        self.qkv = nn.Linear(8, 24)
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gn(F.silu(self.stem(x))).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.head(attn.mean(dim=1)).squeeze(-1)


def _make_opt_pair(g_mod: nn.Module, d_mod: nn.Module, *, batch_size: int = 16, z_dim: int = 4):
    rec = get_recipe("unified_v15", num_particles=32, z_dim=z_dim, batch_size=batch_size)
    prior = rec.make_prior()
    opt_g, opt_d = rec.make_optimizers(g_mod, d_mod, prior)
    return rec, prior, opt_g, opt_d


# ---------------------------------------------------------------------------
# L2-4 & L2-5: Convex-EBI Jensen Debiasing & Regime Dispatch
# ---------------------------------------------------------------------------


def test_l2_4_convex_ebi_jensen_bias_below_kappa_v15() -> None:
    """Layer-1 L2-4 contract: exact RoR on 3x16x16 gives ratio <= 5.0 (no 1015x false triggering)."""
    torch.set_num_threads(1)
    crit = _LinearImgCritic((3, 16, 16), norm_val=0.80)
    rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 4), crit, batch_size=8)
    pen = rec.make_critic_penalty(opt_d, collect_stats=True)

    g = torch.Generator().manual_seed(778)
    xr = torch.randn((8, 3, 16, 16), generator=g)
    xf = torch.randn((8, 3, 16, 16), generator=g)
    xr = xr - xr.mean(dim=0, keepdim=True)
    xf = xf - xf.mean(dim=0, keepdim=True)

    obs = [float(pen(crit, xr, xf).detach().item()) for _ in range(12)]
    mean_pen = sum(obs) / len(obs)
    assert pen.last_stats["grad_mode"] == "exact_ror"

    reg = pen.regularizer
    dim = float(3 * 16 * 16)
    exact_ebi = float(F.softplus(torch.tensor(0.80 - float(reg.kappa)), beta=float(reg.beta)).square().item())
    exact_floor = (0.0005 / dim) * (0.80 ** 2) if float(reg.lambda_floor) > 0.0 else 0.0
    exact_total = 0.5 * float(pen.coefficient) * (float(reg.real_weight) + float(reg.fake_weight)) * (exact_ebi + exact_floor)
    ratio = mean_pen / max(exact_total, 1e-12)
    assert ratio <= 5.0, f"unified_v15 convex-EBI Jensen ratio={ratio:.2f} > 5.0"


def test_l2_4_debiased_fwad_preserves_mean_and_cuts_variance() -> None:
    """Batch-shrinkage fwAD debiasing must preserve exact E[sq] while cutting 1-probe variance > 5x."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(2024)
    z = torch.randn((256, 16), generator=g)
    sq_raw = z.square() * 0.64  # Chi-square(1) 1-probe variance (Var = 2 * s^2)
    sq_deb = torch.stack([_debiased_fwad_sq(sq_raw[i]) for i in range(256)], dim=0)
    assert torch.allclose(sq_deb.mean(dim=1), sq_raw.mean(dim=1), atol=1e-6, rtol=1e-6)
    var_ratio = float(sq_deb.var().item() / max(float(sq_raw.var().item()), 1e-12))
    assert var_ratio < 0.20, f"Debiased fwAD did not reduce variance enough: ratio={var_ratio:.3f}"


def test_l2_5_regime_dispatch_and_sdpa_math_attention() -> None:
    """Pure Conv2d >=128^2 stays on fwAD; GroupNorm/Attention/small-res dispatch to exact RoR with finite grads."""
    torch.set_num_threads(1)

    # 1. Pure Conv2d 128x128 DCGAN -> fwad_debiased (preserves r128/r256 speed & energy win)
    dcgan = _PureConvDCGAN128Critic()
    rec_dc, _, _, opt_dc = _make_opt_pair(nn.Linear(4, 4), dcgan, batch_size=4)
    pen_dc = rec_dc.make_critic_penalty(opt_dc, collect_stats=True)
    x_128 = torch.randn((4, 3, 128, 128), generator=torch.Generator().manual_seed(101))
    val_dc = pen_dc(dcgan, x_128, x_128 * 0.9)
    val_dc.backward()
    assert pen_dc.last_stats["grad_mode"] == "fwad_debiased"
    assert all(torch.isfinite(p.grad).all() for p in dcgan.parameters() if p.grad is not None)

    # 2. GroupNorm + F.scaled_dot_product_attention even at 128x128 -> exact_ror + finite double-backward
    attn = _GroupNormAttentionCritic()
    rec_at, _, _, opt_at = _make_opt_pair(nn.Linear(4, 4), attn, batch_size=4)
    pen_at = rec_at.make_critic_penalty(opt_at, collect_stats=True)
    val_at = pen_at(attn, x_128, x_128 * 0.9)
    val_at.backward()
    assert pen_at.last_stats["grad_mode"] == "exact_ror"
    assert all(torch.isfinite(p.grad).all() for p in attn.parameters() if p.grad is not None)

    # 3. D-11 b=1 edge case in _prepare_batched_highd_tangent_plans
    x_b1 = torch.randn((2, 1, 16), generator=torch.Generator().manual_seed(102))
    u_all, knn_idx, weights = _prepare_batched_highd_tangent_plans(x_b1, phase=0.25)
    assert u_all.shape == (2, 1, 16) and knn_idx.shape == (2, 1, 1)
    assert torch.allclose(weights.sum(dim=2), torch.ones((2, 1)))


# ---------------------------------------------------------------------------
# D-09: Dimension-Independent Image Dissipation Floor
# ---------------------------------------------------------------------------


def test_d09_dimension_independent_image_floor_and_negative_control() -> None:
    """Image floor must stay O(1e-4) independent of dim (not 0.0005/dim -> 2.54e-9 on r256)."""
    torch.set_num_threads(1)

    def _measure_floor_pen(shape: tuple[int, int, int], *, lambda_floor: float | None = None) -> float:
        # Small ||grad|| = 0.05 << kappa=1.0 puts softplus(0.05-1.0, beta=12)^2 ~ 1e-11, isolating the floor term!
        crit = _LinearImgCritic(shape, norm_val=0.05)
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 4), crit, batch_size=8)
        ov = {"lambda_floor": lambda_floor} if lambda_floor is not None else {}
        pen = rec.make_critic_penalty(opt_d, **ov)
        g = torch.Generator().manual_seed(909)
        xr = torch.randn((8, *shape), generator=g)
        xf = torch.randn((8, *shape), generator=g)
        xr = xr - xr.mean(dim=0, keepdim=True)
        xf = xf - xf.mean(dim=0, keepdim=True)
        # Average over 12 calls so 128x128/256x256 fwAD expectation matches exact floor
        return sum(float(pen(crit, xr, xf).detach().item()) for _ in range(12)) / 12.0

    p_16 = _measure_floor_pen((3, 16, 16))
    p_256 = _measure_floor_pen((3, 256, 256))
    expected_floor_only = 0.5 * 1.0 * (1.0 + 1.0) * 1.0e-4 * (0.05 ** 2)  # coeff=1, real+fake=2, floor=1e-4, sq=0.0025
    assert p_16 >= 0.8 * expected_floor_only
    # On r256 (dim=196608), old 0.0005/dim gave 6.3e-12; v15 must stay within ~2.5x of p_16!
    assert p_256 >= 0.30 * p_16, f"D-09 floor vanished on r256: p_256={p_256:.3e} vs p_16={p_16:.3e}"

    # Semantic negative control: disabling lambda_floor drops curvature > 1000x
    p_zero = _measure_floor_pen((3, 16, 16), lambda_floor=0.0)
    assert p_zero < 1e-3 * p_16, f"Negative control failed: lambda_floor=0 still gave {p_zero:.3e}"


# ---------------------------------------------------------------------------
# D-04 (Critic Side): Scale Homogeneity, Stratified Probe, Eval Parity
# ---------------------------------------------------------------------------


def test_d04_critic_scale_homogeneity_and_stratified_probe() -> None:
    """observe_scale must be scale-homogeneous at 1e-3/1e-5, and stratified probe must capture 1% tail modes."""
    torch.set_num_threads(1)
    for sc in (1e-3, 1e-5):
        _, _, _, opt_d = _make_opt_pair(nn.Linear(4, 2), nn.Linear(2, 1), batch_size=32)
        x = torch.randn((64, 2), generator=torch.Generator().manual_seed(404)) * sc
        true_rms = float(x.square().mean().sqrt().item())
        opt_d.record.observe_scale(x, x)
        tracked = float(opt_d.record.ema_data_scale) / math.sqrt(2.0)
        assert tracked <= 1.10 * true_rms and tracked >= 0.90 * true_rms

    # Stratified probe on B=128, dim=4096 with 1 rare outlier at index 117 (missed 100% by [:2] prefix!)
    g_mod = nn.Linear(4, 4096, bias=False)
    _, _, _, opt_d = _make_opt_pair(g_mod, nn.Linear(4096, 1), batch_size=128)
    z = torch.randn((128, 4), generator=torch.Generator().manual_seed(405)) * 0.1
    z[117] = torch.tensor([50.0, 50.0, 50.0, 50.0])
    y = g_mod(z)
    assert record_unified_v15_generator_probe(g_mod, (z,), {}, y, opt_d.record)
    _, p_args, _, y_pre = opt_d.record._g_probe_cache
    assert y_pre.shape[0] >= 8, f"Expected >= 8 stratified probes, got {y_pre.shape[0]}"
    assert float(p_args[0].abs().max().item()) >= 49.0, "Rare tail outlier at index 117 was missed by probe!"

    # Low-D (dim=2) 1% minority cluster (fl_law41 geometry) captured even when outside [:16]
    g2 = nn.Identity()
    _, _, _, opt_d2 = _make_opt_pair(nn.Linear(2, 2), nn.Linear(2, 1), batch_size=256)
    g_rng = torch.Generator().manual_seed(407)
    maj_centers = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    y2 = maj_centers[torch.arange(256) % 4] + torch.randn((256, 2), generator=g_rng) * 0.05
    y2[240:243] = torch.tensor([[0.0, 2.85]]) + torch.randn((3, 2), generator=g_rng) * 0.02
    y2_req = y2.clone().requires_grad_(True)
    assert record_unified_v15_generator_probe(g2, (y2_req,), {}, y2_req, opt_d2.record)
    _, _, _, y_pre2 = opt_d2.record._g_probe_cache
    assert y_pre2.shape[0] == 16
    assert float(y_pre2[:, 1].max().item()) >= 2.7, "Low-D 1% minority mode was missed by zero-sync tail guard!"

    # Negative control: uniform 10x10 grid stays unbiased [:16] IID (no radial-tail distortion)
    coords = torch.linspace(-4.5, 4.5, 10)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    grid_pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    y_grid = grid_pts[torch.randint(0, 100, (512,), generator=g_rng)].clone().requires_grad_(True)
    assert record_unified_v15_generator_probe(g2, (y_grid,), {}, y_grid, opt_d2.record)
    _, _, _, y_pre_g = opt_d2.record._g_probe_cache
    assert torch.equal(y_pre_g, y_grid.detach()[:16])

    # Balanced high-D (dim=128) 64-mode batch stays unbiased [:32] (zero argsort rank-flipping vs 4D Conv2d 1x1)
    g128 = nn.Identity()
    _, _, _, opt_d128 = _make_opt_pair(nn.Linear(128, 128), nn.Linear(128, 1), batch_size=256)
    c64 = torch.randn((64, 128), generator=g_rng) * 0.4
    y128 = (c64[torch.arange(256) % 64] + 0.08 * torch.randn((256, 128), generator=g_rng)).requires_grad_(True)
    assert record_unified_v15_generator_probe(g128, (y128,), {}, y128, opt_d128.record)
    _, _, _, y_pre128 = opt_d128.record._g_probe_cache
    assert torch.equal(y_pre128, y128.detach()[:32])


def test_d04_custom_training_flag_generator_eval_parity() -> None:
    """Custom `if self.training:` generator without BN/Dropout must capture eval() y_pre matching post-step eval()."""
    torch.set_num_threads(1)

    class _ModeDependentGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            out = self.fc(z)
            return out * 3.0 if self.training else out

    g_mod = _ModeDependentGen()
    g_mod.train()
    _, _, _, opt_d = _make_opt_pair(g_mod, nn.Linear(8, 1), batch_size=16)
    z = torch.randn((16, 4), generator=torch.Generator().manual_seed(406))
    y_train = g_mod(z)
    assert record_unified_v15_generator_probe(g_mod, (z,), {}, y_train, opt_d.record)
    _, p_args, _, y_pre = opt_d.record._g_probe_cache
    assert getattr(g_mod, "_unified_v15_eval_verified", None) is True
    g_mod.eval()
    with torch.no_grad():
        y_eval = g_mod(p_args[0])
    assert torch.allclose(y_pre, y_eval, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# D-08: Screening Dead-Zone Elimination + Broad Covered Cluster Negative Control
# ---------------------------------------------------------------------------


def test_d08_macro_screening_alignment_and_broad_cluster_guard() -> None:
    """Orphan at c_hat = 1.9*rho (old dead zone [1.5*rho, 2.8*rho]) must rescue; broad covered mode stays 0."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(808)
    rho = 3.0
    ell = torch.tensor(0.25)

    # 1. Positive control: 56 fake points on cluster A (origin), real has 48 at origin + 16 orphan at (1.70, 0)
    # Distance 1.70 -> c_hat = (sqrt(0.25^2 + 1.70^2) - 0.25) / 0.25 = 5.87 = 1.96 * rho (deep inside old [4.5, 8.4] dead zone!)
    x_fake = (torch.randn((64, 2), generator=g) * 0.04).requires_grad_(True)
    y_main = torch.randn((48, 2), generator=g) * 0.04
    y_orph = torch.randn((16, 2), generator=g) * 0.04 + torch.tensor([[1.70, 0.0]])
    y_real = torch.cat([y_main, y_orph], dim=0)
    c_hat = born_infeld_c_hat_detached(x_fake, y_real, ell)
    col_min = float(c_hat[:, 48:].amin(dim=0).median().item())
    assert 1.5 * rho < col_min < 2.5 * rho, f"Test geometry col_min={col_min:.2f} outside old dead zone"
    zeros = torch.zeros(64)
    P_m = sinkhorn_slack(c_hat, zeros, zeros, rho=rho, tau=2.0, eps=0.1, iters=8)
    loss_orph = v15_zero_eigh_bipartite_deficit_macro_loss(
        x_fake, y_real, c_hat, P_m[:64, 64], P_m[64, :64], rho=rho, g_conn=0.0
    )
    assert float(loss_orph.item()) > 0.0, "D-08 dead zone still silent at 1.96*rho!"
    loss_orph.backward()
    assert x_fake.grad is not None and float(x_fake.grad.norm().item()) > 0.0

    # 2. Negative control: covered broad cluster (t3_iso_scale_mass_8x geometry, r_loc ~ 1.0) even with small ell
    x_broad = (torch.randn((64, 2), generator=g) * 2.0).requires_grad_(True)
    y_broad = torch.randn((64, 2), generator=g) * 2.0
    c_hat_b = born_infeld_c_hat_detached(x_broad, y_broad, torch.tensor(0.10))
    P_b = sinkhorn_slack(c_hat_b, zeros, zeros, rho=rho, tau=2.0, eps=0.1, iters=8)
    loss_broad = v15_zero_eigh_bipartite_deficit_macro_loss(
        x_broad, y_broad, c_hat_b, P_b[:64, 64], P_b[64, :64], rho=rho, g_conn=0.0
    )
    assert float(loss_broad.item()) == 0.0, f"Covered broad cluster triggered false macro kick: {loss_broad.item()}"


# ---------------------------------------------------------------------------
# L3-7: Haar-O(32) Rotation Invariance, Scale Equivariance, Equilibrium, Hollow R^512
# ---------------------------------------------------------------------------


def test_l3_7_highd_stein_haar_rotation_and_scale_equivariance() -> None:
    """High-D Basin Stein loss must be Haar-O(32)-invariant, [length]^1 scale-equivariant, and 0 at equilibrium."""
    torch.set_num_threads(1)
    d = 32
    g = torch.Generator().manual_seed(3201)
    q_mat, _ = torch.linalg.qr(torch.randn((d, d), generator=g, dtype=torch.float64))
    q32 = q_mat.float()

    # Build a 2-basin R^32 dataset and slightly collapsed fake batch
    y1 = torch.randn((48, d), generator=g) * 0.10
    y2 = torch.randn((48, d), generator=g) * 0.15 + torch.cat([torch.tensor([4.0]), torch.zeros(d - 1)])
    y_base = torch.cat([y1, y2], dim=0)
    x1 = torch.randn((48, d), generator=g) * 0.04
    x2 = torch.randn((48, d), generator=g) * 0.06 + torch.cat([torch.tensor([4.05]), torch.zeros(d - 1)])
    x_base = torch.cat([x1, x2], dim=0)

    def _eval_transform(rot: torch.Tensor, scale: float) -> tuple[float, torch.Tensor]:
        rec = UnifiedV15StepRecord()
        # Warm up streaming bank for 3 identical steps so w_y >= 0.4*d and cov_gate == 1.0
        for _ in range(3):
            x_in = (x_base @ rot.T * scale).detach().requires_grad_(True)
            y_in = (y_base @ rot.T * scale).detach()
            lv, _ = unified_v15_highd_basin_stein_loss(
                x_in, y_in, record=rec, generator=None, z_batch=None, allow_update=True
            )
        lv.backward()
        assert x_in.grad is not None
        return float(lv.detach().item()), x_in.grad.detach()

    l_id, g_id = _eval_transform(torch.eye(d), 1.0)
    l_rot, g_rot = _eval_transform(q32, 1.0)
    l_sc, g_sc = _eval_transform(torch.eye(d), 10.0)

    assert abs(l_id) > 1e-4
    assert abs(l_rot - l_id) / max(abs(l_id), 1e-8) < 1e-4, f"Haar invariance failed: {l_rot} vs {l_id}"
    g_rot_back = g_rot @ q32
    assert torch.allclose(g_rot_back, g_id, atol=1e-4, rtol=1e-3), "Gradient not O(32)-equivariant!"
    assert abs(l_sc - 10.0 * l_id) / max(abs(10.0 * l_id), 1e-8) < 1e-3, f"Scale equivariance failed: {l_sc} vs {10*l_id}"
    assert torch.allclose(g_sc, g_id, atol=1e-4, rtol=1e-3), "Gradient should be degree-0 scale-invariant!"

    # Equilibrium check: identical x == y yields ~0 loss and ~0 deficit
    rec_eq = UnifiedV15StepRecord()
    for _ in range(4):
        x_eq = y_base.clone().requires_grad_(True)
        lv_eq, def_eq = unified_v15_highd_basin_stein_loss(x_eq, y_base, record=rec_eq, allow_update=True)
    assert abs(float(lv_eq.item())) < 1e-3 and def_eq < 0.05


def test_highd_sparse_r512_uses_hollow_self_coupling() -> None:
    """On R^512 sparse vectors (D >= 256, non-atomic), t1c must use hollow P_xx eliminating 2x shrinkage bias."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(512)
    t1c = UnifiedV15CouplingLoss(self_coupling="feydy_atomic_v15", chart_weight=1.4)
    rec = UnifiedV15StepRecord()
    mask_x = (torch.rand((32, 512), generator=g) < 0.10).float()
    mask_y = (torch.rand((32, 512), generator=g) < 0.10).float()
    x_f = (torch.randn((32, 512), generator=g) * mask_x).requires_grad_(True)
    y_r = torch.randn((32, 512), generator=g) * mask_y
    d_f = torch.zeros((32, 1), requires_grad=True)
    d_r = torch.zeros((32, 1))
    loss_auto, _ = t1c(x_f, y_r, d_f, d_r, record=rec, allow_ell_update=False)

    t1c_hollow = UnifiedV15CouplingLoss(self_coupling="hollow", chart_weight=1.4)
    rec_h = UnifiedV15StepRecord()
    x_fh = x_f.detach().clone().requires_grad_(True)
    d_fh = torch.zeros((32, 1), requires_grad=True)
    loss_hol, _ = t1c_hollow(x_fh, y_r, d_fh, d_r, record=rec_h, allow_ell_update=False)
    assert torch.allclose(loss_auto, loss_hol, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoint Bit-Exact Continuation (Fresh Object + Fresh Subprocess) & Schema Rejection
# ---------------------------------------------------------------------------


def _make_d32_trainer() -> tuple[GANTrainer, torch.Generator]:
    rec = get_recipe("unified_v15", num_particles=64, z_dim=32, batch_size=32, prior_kind="mog", standardize=False)
    g_mod = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 32))
    d_mod = nn.Sequential(nn.Linear(32, 48), nn.LeakyReLU(0.2), nn.Linear(48, 1))
    prior = MoGParticlePrior(64, 32, sigma=0.05, standardize=False, learnable=True)
    init.deterministic_orthogonal_(g_mod, seed=11)
    init.deterministic_orthogonal_(d_mod, seed=21)
    init.deterministic_orthogonal_(prior, seed=31)
    tr = GANTrainer(rec, g_mod, d_mod, prior=prior, seed=0, max_steps=20)
    return tr, torch.Generator().manual_seed(9876)


def test_checkpoint_bit_exact_continuation_d32_highd_bank() -> None:
    """Fresh-object checkpoint save/restore must continue bit-exact across high-D basin bank activation."""
    torch.set_num_threads(1)
    tr_full, rng_full = _make_d32_trainer()
    tr_split, rng_split = _make_d32_trainer()

    def _batch(gen: torch.Generator) -> torch.Tensor:
        a = torch.randn((16, 32), generator=gen) * 0.1
        b = torch.randn((16, 32), generator=gen) * 0.1 + 3.0
        return torch.cat([a, b], dim=0)

    for _ in range(4):
        tr_full.step(_batch(rng_full))
        tr_split.step(_batch(rng_split))

    assert tr_split.opt_d.record.highd_basin_bank.k >= 1
    assert tr_split.opt_d.record.highd_basin_bank.sigma_z_init is True

    saved = copy.deepcopy(tr_split.state_dict())
    tr_resumed, _ = _make_d32_trainer()
    tr_resumed.load_state_dict(saved)

    for _ in range(4):
        tr_full.step(_batch(rng_full))
        tr_resumed.step(_batch(rng_split))

    for pa, pb in zip(tr_full.G.parameters(), tr_resumed.G.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(tr_full.D.parameters(), tr_resumed.D.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(unwrap_prior(tr_full.prior).z, unwrap_prior(tr_resumed.prior).z)
    b_f = tr_full.opt_d.record.highd_basin_bank
    b_r = tr_resumed.opt_d.record.highd_basin_bank
    assert b_f.k == b_r.k and b_f.sigma_z_ema == b_r.sigma_z_ema
    assert torch.equal(b_f.cov_y, b_r.cov_y) and torch.equal(b_f.cov_x, b_r.cov_x)


def test_checkpoint_rejects_incompatible_and_tampered_highd_packets() -> None:
    """Strict checkpoint validation must reject missing/corrupted highd_basin_bank and legacy v14 packets."""
    torch.set_num_threads(1)
    rec = UnifiedV15StepRecord()
    sd = rec.state_dict()

    # Missing highd_basin_bank (simulating legacy v14 packet)
    bad_missing = copy.deepcopy(sd)
    del bad_missing["unified_v15"]["highd_basin_bank"]
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_missing)

    # Wrong formulation tag
    bad_form = copy.deepcopy(sd)
    bad_form["formulation"] = "unified_v14"
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_form)

    # Tampered highd bank tensor (NaN / negative weight / shape mismatch)
    bank = UnifiedV15HighDBasinBank()
    bank.k, bank.dim = 1, 32
    bank.w_y = torch.tensor([10.0], dtype=torch.float64)
    bank.mu_y = torch.zeros((1, 32), dtype=torch.float64)
    bank.cov_y = torch.eye(32, dtype=torch.float64).unsqueeze(0)
    bank.rad_y = torch.tensor([1.0], dtype=torch.float64)
    bank.w_x = torch.tensor([10.0], dtype=torch.float64)
    bank.mu_x = torch.zeros((1, 32), dtype=torch.float64)
    bank.cov_x = torch.eye(32, dtype=torch.float64).unsqueeze(0)
    pkt = bank.to_packet()
    validate_v15_highd_bank_packet(pkt)

    bad_neg = copy.deepcopy(pkt)
    bad_neg["w_y"] = torch.tensor([-1.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        validate_v15_highd_bank_packet(bad_neg)

    bad_nan = copy.deepcopy(pkt)
    bad_nan["cov_x"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_v15_highd_bank_packet(bad_nan)


def test_fresh_process_checkpoint_continuation_and_clock_free() -> None:
    """Fresh Python subprocess resume must match unbroken trajectory bit-exact, and horizon changes must be clock-free."""
    torch.set_num_threads(1)
    tr_full, rng_full = _make_d32_trainer()
    batches = [
        torch.cat([torch.randn((16, 32), generator=rng_full) * 0.1, torch.randn((16, 32), generator=rng_full) * 0.1 + 3.0], dim=0)
        for _ in range(6)
    ]
    for b in batches[:3]:
        tr_full.step(b)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "ckpt.pt"
        batch_path = Path(td) / "batches.pt"
        out_path = Path(td) / "resumed.pt"
        torch.save(tr_full.state_dict(), ckpt_path)
        torch.save(batches[3:], batch_path)

        for b in batches[3:]:
            tr_full.step(b)

        script = (
            "import torch\n"
            "from torch import nn\n"
            "from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior\n"
            "torch.set_num_threads(1)\n"
            "rec = get_recipe('unified_v15', num_particles=64, z_dim=32, batch_size=32, prior_kind='mog', standardize=False)\n"
            "g_mod = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 32))\n"
            "d_mod = nn.Sequential(nn.Linear(32, 48), nn.LeakyReLU(0.2), nn.Linear(48, 1))\n"
            "prior = MoGParticlePrior(64, 32, sigma=0.05, standardize=False, learnable=True)\n"
            "init.deterministic_orthogonal_(g_mod, seed=11)\n"
            "init.deterministic_orthogonal_(d_mod, seed=21)\n"
            "init.deterministic_orthogonal_(prior, seed=31)\n"
            "tr = GANTrainer(rec, g_mod, d_mod, prior=prior, seed=0, max_steps=20)\n"
            f"tr.load_state_dict(torch.load({str(ckpt_path)!r}, weights_only=False))\n"
            f"for b in torch.load({str(batch_path)!r}, weights_only=False):\n"
            "    tr.step(b)\n"
            "torch.save({'g': [p.detach().clone() for p in tr.G.parameters()], 'z': unwrap_prior(tr.prior).z.detach().clone()}, "
            f"{str(out_path)!r})\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"Subprocess failed:\n{proc.stderr}"
        res = torch.load(out_path, weights_only=False)
        for pa, pb in zip(tr_full.G.parameters(), res["g"]):
            assert torch.equal(pa.detach(), pb)
        assert torch.equal(unwrap_prior(tr_full.prior).z.detach(), res["z"])

    # Clock-free invariance on D=32 (total_steps = 40 vs 4000)
    def _run_horizon(horizon: int) -> torch.Tensor:
        rec = get_recipe("unified_v15", num_particles=64, z_dim=32, batch_size=32, prior_kind="mog", standardize=False, total_steps=horizon)
        g_mod = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 32))
        d_mod = nn.Sequential(nn.Linear(32, 48), nn.LeakyReLU(0.2), nn.Linear(48, 1))
        prior = MoGParticlePrior(64, 32, sigma=0.05, standardize=False, learnable=True)
        init.deterministic_orthogonal_(g_mod, seed=11)
        init.deterministic_orthogonal_(d_mod, seed=21)
        init.deterministic_orthogonal_(prior, seed=31)
        tr = GANTrainer(rec, g_mod, d_mod, prior=prior, seed=0, max_steps=horizon)
        for b in batches[:5]:
            tr.step(b)
        return torch.cat([p.detach().reshape(-1) for p in g_mod.parameters()])

    assert torch.equal(_run_horizon(40), _run_horizon(4000))


def test_singleton_spatial_conv1x1_parity() -> None:
    """Singleton-spatial (B, C, 1, 1) Conv2d 1x1 tensors must match (B, C) Linear across penalty, loss, and hooks."""
    torch.set_num_threads(1)
    dim = 32
    lin_g = nn.Linear(dim, dim)
    lin_d = nn.Linear(dim, 1)
    init.deterministic_orthogonal_(lin_g, seed=501)
    init.deterministic_orthogonal_(lin_d, seed=502)

    conv_g = nn.Conv2d(dim, dim, kernel_size=1)
    conv_d = nn.Conv2d(dim, 1, kernel_size=1)
    with torch.no_grad():
        conv_g.weight.copy_(lin_g.weight.view(dim, dim, 1, 1))
        conv_g.bias.copy_(lin_g.bias)
        conv_d.weight.copy_(lin_d.weight.view(1, dim, 1, 1))
        conv_d.bias.copy_(lin_d.bias)

    rec_l, _, _, opt_dl = _make_opt_pair(lin_g, lin_d, batch_size=32)
    rec_c, _, _, opt_dc = _make_opt_pair(conv_g, conv_d, batch_size=32)
    loss_l = rec_l.make_loss()
    loss_c = rec_c.make_loss()
    pen_l = rec_l.make_critic_penalty(opt_dl, collect_stats=True)
    pen_c = rec_c.make_critic_penalty(opt_dc, collect_stats=True)

    g = torch.Generator().manual_seed(503)
    z2 = torch.randn((32, dim), generator=g)
    xr2 = torch.randn((32, dim), generator=g) * 0.25
    z4 = z2.view(32, dim, 1, 1)
    xr4 = xr2.view(32, dim, 1, 1)

    # 1. Penalty exact parity (both must route to _penalty_highd_batched, NOT image branch!)
    xf2 = lin_g(z2).detach()
    xf4 = conv_g(z4).detach()
    pl = pen_l(lin_d, xr2, xf2)
    pc = pen_c(conv_d, xr4, xf4)
    assert torch.allclose(pl, pc, atol=1e-6, rtol=1e-5), f"Penalty mismatch: {pl.item()} vs {pc.item()}"
    pl.backward()
    pc.backward()
    assert torch.allclose(lin_d.weight.grad, conv_d.weight.grad.view_as(lin_d.weight), atol=1e-5, rtol=1e-4)

    # 2. Generator loss (incl. highd_basin_stein_loss) exact parity
    lin_g.zero_grad()
    conv_g.zero_grad()
    for _ in range(3):
        dr2 = lin_d(xr2)
        df2 = lin_d(lin_g(z2))
        gl2 = loss_l.g_loss(df2, dr2)
        dr4 = conv_d(xr4)
        df4 = conv_d(conv_g(z4))
        gl4 = loss_c.g_loss(df4, dr4)
    assert torch.allclose(gl2, gl4, atol=1e-5, rtol=1e-4), f"g_loss mismatch: {gl2.item()} vs {gl4.item()}"
    gl2.backward()
    gl4.backward()
    assert torch.allclose(lin_g.weight.grad, conv_g.weight.grad.view_as(lin_g.weight), atol=1e-5, rtol=1e-4)


def test_observe_scale_spike_immunity_sparsity_and_mixed_shape() -> None:
    """observe_scale must resist 1e4x transient spikes, unbias sparse masks by 1/sqrt(p), and accept (B,C) vs (B,C,1,1)."""
    torch.set_num_threads(1)
    rec = UnifiedV15StepRecord()
    g = torch.Generator().manual_seed(601)
    base = torch.randn((64, 2), generator=g) * 1e-5

    # Step 0 initialization is exact (scale-homogeneous at 1e-5)
    rec.observed_steps = 0
    rec.observe_scale(base, base)
    sc0 = rec.ema_data_scale
    expected_sc0 = float(base.square().sum(dim=-1).mean().sqrt().item())
    assert abs(sc0 - expected_sc0) / expected_sc0 < 1e-5

    # Two transient 1e4x spike steps must NOT inflate ema_data_scale by 1000x (ratio cap <= 4x per step -> <= +15%/step)
    rec.observed_steps = 1
    rec.observe_scale(base * 1e4, base)
    rec.observed_steps = 2
    rec.observe_scale(base * 1e4, base)
    sc_post_spike = rec.ema_data_scale
    assert sc_post_spike < sc0 * 1.40, f"Transient 1e4x spike exploded ema_data_scale: {sc0:.3e} -> {sc_post_spike:.3e}"
    # Recovery back toward sc0 on normal steps
    for s in range(3, 15):
        rec.observed_steps = s
        rec.observe_scale(base, base)
    assert abs(rec.ema_data_scale - sc0) / sc0 < 0.25

    # Sparsity unbiasing on 10%-observed R^512 must recover full dense clean scale (within 15%)
    rec_sp = UnifiedV15StepRecord()
    clean = torch.randn((128, 512), generator=g) * 0.56
    mask = (torch.rand((128, 512), generator=g) < 0.10).float()
    rec_sp.observed_steps = 0
    rec_sp.observe_scale(clean * mask, clean * mask)
    clean_sc = float(clean.square().sum(dim=-1).mean().sqrt().item())
    assert abs(rec_sp.ema_data_scale - clean_sc) / clean_sc < 0.12

    # Dense R^512 stays exact (no sparsity correction triggered)
    rec_dn = UnifiedV15StepRecord()
    rec_dn.observed_steps = 0
    rec_dn.observe_scale(clean, clean)
    assert abs(rec_dn.ema_data_scale - clean_sc) / clean_sc < 1e-5

    # Mixed (B, C) real vs (B, C, 1, 1) fake must match (B, C) vs (B, C) bit-identically
    xr_m = torch.randn((32, 32), generator=g)
    xf_m = torch.randn((32, 32), generator=g)
    rec_a = UnifiedV15StepRecord()
    rec_b = UnifiedV15StepRecord()
    rec_a.observe_scale(xr_m, xf_m)
    rec_b.observe_scale(xr_m, xf_m.view(32, 32, 1, 1))
    assert rec_a.ema_data_scale == rec_b.ema_data_scale
    assert rec_a.ema_push_res == rec_b.ema_push_res

    # Degenerate exact-zero target (zero_tanh_smoke / zero_tanh_rail_escape): degree-1 homogeneous fallback to ||x_fake||_rms
    z_real = torch.zeros((64, 2), dtype=torch.float32)
    xf_base = torch.randn((64, 2), generator=g) * 0.5
    xf_rms = float(xf_base.square().sum(dim=-1).mean().double().sqrt().item())
    for c_scale in (1e-5, 1.0, 1e5):
        rec_z = UnifiedV15StepRecord()
        rec_z.observed_steps = 0
        rec_z.observe_scale(z_real, xf_base * c_scale)
        expected = xf_rms * c_scale
        assert abs(rec_z.ema_data_scale - expected) / expected < 1e-6, (
            f"Zero-target scale fallback broke degree-1 homogeneity at c={c_scale}: {rec_z.ema_data_scale} vs {expected}"
        )
    # Non-zero small target (1e-3) stays exact sc_real even when x_fake is 100x larger (D-04 invariant)
    xr_small = torch.randn((64, 2), generator=g) * 1e-3
    xr_small_rms = float(xr_small.square().sum(dim=-1).mean().double().sqrt().item())
    rec_nz = UnifiedV15StepRecord()
    rec_nz.observed_steps = 0
    rec_nz.observe_scale(xr_small, xr_small * 100.0)
    assert abs(rec_nz.ema_data_scale - xr_small_rms) / xr_small_rms < 1e-6
    # Degenerate zero target with x_fake=None safely clamps to 1e-12 (finite, positive)
    rec_z0 = UnifiedV15StepRecord()
    rec_z0.observed_steps = 0
    rec_z0.observe_scale(z_real, None)
    assert math.isfinite(rec_z0.ema_data_scale) and abs(rec_z0.ema_data_scale - 1e-12) < 1e-18


def test_highd_basin_bank_capacity_overflow_guard() -> None:
    """highd_basin_stein_loss must cleanly disable itself on >16-mode overflow while staying active on <=16 modes."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(701)

    # 32 well-separated modes in R^128 overflow max_basins=16 -> must return (0.0, 0.0)
    centers_32 = F.normalize(torch.randn((32, 128), generator=g), dim=1) * 6.0
    idx_32 = torch.arange(128) % 32
    y_32 = centers_32[idx_32] + 0.05 * torch.randn((128, 128), generator=g)
    x_32 = (y_32 * 0.5).clone().requires_grad_(True)
    rec_ov = UnifiedV15StepRecord()
    loss_ov, def_ov = unified_v15_highd_basin_stein_loss(x_32, y_32, record=rec_ov, allow_update=True)
    assert rec_ov.highd_basin_bank.k == rec_ov.highd_basin_bank.max_basins
    assert float(loss_ov.item()) == 0.0 and def_ov == 0.0
    # Read-only probe must also respect overflow guard
    loss_ov_ro, def_ov_ro = unified_v15_highd_basin_stein_loss(x_32, y_32, record=rec_ov, allow_update=False)
    assert float(loss_ov_ro.item()) == 0.0 and def_ov_ro == 0.0

    # 4 well-separated modes in R^32 fit cleanly within max_basins=16 -> must stay active!
    centers_4 = F.normalize(torch.randn((4, 32), generator=g), dim=1) * 5.0
    idx_4 = torch.arange(64) % 4
    y_4 = centers_4[idx_4] + 0.15 * torch.randn((64, 32), generator=g)
    x_4 = (centers_4[idx_4] + 0.03 * torch.randn((64, 32), generator=g)).requires_grad_(True)
    rec_ok = UnifiedV15StepRecord()
    for _ in range(4):
        loss_ok, def_ok = unified_v15_highd_basin_stein_loss(x_4, y_4, record=rec_ok, allow_update=True)
    assert rec_ok.highd_basin_bank.k == 4
    assert def_ok > 0.1 and loss_ok.requires_grad


def test_sparse_obs_wiener_loss_invariances_validation_and_subprocess_checkpoint() -> None:
    """Verify sparse-obs Wiener-NLM operator: scale invariance, equilibrium, read-only probe, schema checks & fresh-process resume."""
    torch.set_num_threads(1)
    dim, dz, n_modes, bs = 128, 16, 8, 64
    g = torch.Generator().manual_seed(801)
    q_orth, _ = torch.linalg.qr(torch.randn((dim, n_modes), generator=g))
    centers = q_orth.T * 6.0  # (8, 128) orthogonal modes, norm=6.0

    def _run_steps(scale: float) -> tuple[UnifiedV15StepRecord, torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = UnifiedV15StepRecord()
        gg = torch.Generator().manual_seed(802)
        last_loss = torch.tensor(0.0)
        last_xf = None
        last_z = None
        for _ in range(20):
            idx = torch.randint(0, n_modes, (bs,), generator=gg)
            y_clean = (centers[idx] + 0.03 * torch.randn((bs, dim), generator=gg)) * scale
            mr = (torch.rand((bs, dim), generator=gg) < 0.20).float()
            mf = (torch.rand((bs, dim), generator=gg) < 0.20).float()
            z_b = torch.randn((bs, dz), generator=gg)
            xf = (torch.randn((bs, dim), generator=gg) * 0.2 * scale * mf).requires_grad_(True)
            last_loss = unified_v15_sparse_obs_wiener_loss(
                xf, y_clean * mr, record=rec, z_batch=z_b, allow_update=True, start_proto_step=10
            )
            last_xf = xf
            last_z = z_b
        return rec, last_loss, last_xf, last_z

    rec_1, loss_1, xf_1, z_1 = _run_steps(1.0)
    rec_tiny, loss_tiny, _, _ = _run_steps(1e-4)

    # 1. Exact degree-0 scale invariance across 1e4x scale change
    assert rec_1.sparse_obs_bank.proto_k >= n_modes
    assert rec_1.sparse_obs_bank.proto_k == rec_tiny.sparse_obs_bank.proto_k
    assert torch.allclose(loss_1.detach(), loss_tiny.detach(), atol=1e-4, rtol=1e-3), (
        f"Sparse Wiener loss not scale-invariant: {loss_1.item():.6f} vs {loss_tiny.item():.6f}"
    )

    # 2. Prototype recovery accuracy from 20%-masked R^128 observations
    bank = rec_1.sparse_obs_bank
    assert isinstance(bank, UnifiedV15SparseObsBank)
    act = bank.proto_w[: bank.proto_k] >= 2.0
    protos = bank.proto_y[: bank.proto_k][act]
    med_err = float(torch.cdist(protos, centers).min(dim=1).values.median().item())
    assert med_err < 1.5, f"Wiener-NLM prototypes failed to recover centers: med_err={med_err:.3f}"

    # 3. Read-only protected-probe invariance (allow_update=False mutates ZERO bank state)
    pkt_before = copy.deepcopy(bank.to_packet())
    loss_ro = unified_v15_sparse_obs_wiener_loss(
        xf_1.detach().requires_grad_(True),
        torch.zeros_like(xf_1),
        record=rec_1,
        z_batch=z_1,
        allow_update=False,
    )
    assert loss_ro.requires_grad and torch.isfinite(loss_ro)
    pkt_after = bank.to_packet()
    assert pkt_before.keys() == pkt_after.keys()
    for k in pkt_before:
        va, vb = pkt_before[k], pkt_after[k]
        if isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"Read-only probe mutated sparse_obs_bank.{k}"
        else:
            assert va == vb

    # 4. Exact equilibrium zero-loss & zero-gradient when x_fake matches its assigned Laguerre prototype
    act_idx = act.nonzero(as_tuple=False).squeeze(-1)
    zu = F.normalize(z_1, dim=1)
    asg = (zu @ bank.proto_z[act_idx].T + bank.proto_b[act_idx].unsqueeze(0)).argmax(dim=1)
    tgt_eq = bank.proto_y[act_idx][asg]
    mf_eq = (torch.rand((bs, dim), generator=torch.Generator().manual_seed(809)) < 0.20).float()
    xf_eq = (tgt_eq * mf_eq).detach().requires_grad_(True)
    loss_eq = unified_v15_sparse_obs_wiener_loss(
        xf_eq, torch.zeros_like(xf_eq), record=rec_1, z_batch=z_1, allow_update=False
    )
    assert float(loss_eq.item()) == 0.0
    grad_eq = torch.autograd.grad(loss_eq, xf_eq)[0]
    assert float(grad_eq.abs().max().item()) == 0.0

    # 5. Strict checkpoint schema rejection of missing/tampered sparse_obs_bank packets
    sd = rec_1.state_dict()
    bad_miss = copy.deepcopy(sd)
    del bad_miss["unified_v15"]["sparse_obs_bank"]
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_miss)

    pkt_ok = bank.to_packet()
    validate_v15_sparse_obs_bank_packet(pkt_ok)
    bad_nan = copy.deepcopy(pkt_ok)
    bad_nan["S_y"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_v15_sparse_obs_bank_packet(bad_nan)
    bad_w = copy.deepcopy(pkt_ok)
    bad_w["proto_w"][0] = -1.0
    with pytest.raises(ValueError):
        validate_v15_sparse_obs_bank_packet(bad_w)
    bad_dtype = copy.deepcopy(pkt_ok)
    bad_dtype["buf_m"] = bad_dtype["buf_m"].float()
    with pytest.raises(ValueError):
        validate_v15_sparse_obs_bank_packet(bad_dtype)

    # 6. Fresh-subprocess bit-exact checkpoint continuation across step-8 prototype activation & recentering window!
    class _SparseMaskedG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dz, 32), nn.SiLU(), nn.Linear(32, dim))
            self._m_gen = torch.Generator().manual_seed(909)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            out = torch.tanh(self.net(z))
            if self.training:
                m = (torch.rand(out.shape, generator=self._m_gen) < 0.20).to(out.dtype)
                return out * m
            return out

    def _make_sparse_trainer() -> GANTrainer:
        rc = get_recipe("unified_v15", num_particles=32, z_dim=dz, batch_size=32, prior_kind="mog", standardize=False)
        gm = _SparseMaskedG()
        dm = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))
        pr = MoGParticlePrior(32, dz, sigma=0.05, standardize=False, learnable=True)
        init.deterministic_orthogonal_(gm, seed=41)
        init.deterministic_orthogonal_(dm, seed=42)
        init.deterministic_orthogonal_(pr, seed=43)
        return GANTrainer(rc, gm, dm, prior=pr, seed=0, max_steps=20)

    tr_full = _make_sparse_trainer()
    rng_b = torch.Generator().manual_seed(910)
    batches = []
    for _ in range(14):
        idx = torch.randint(0, n_modes, (32,), generator=rng_b)
        yc = centers[idx] + 0.03 * torch.randn((32, dim), generator=rng_b)
        mr = (torch.rand((32, dim), generator=rng_b) < 0.20).float()
        batches.append(yc * mr)

    for b in batches[:6]:
        tr_full.step(b)
    assert tr_full.opt_d.record.sparse_obs_bank.step == 6
    assert tr_full.opt_d.record.sparse_obs_bank.proto_k == 0  # pre-activation!

    with tempfile.TemporaryDirectory() as td:
        ckpt_p = Path(td) / "ckpt.pt"
        b_p = Path(td) / "batches.pt"
        out_p = Path(td) / "out.pt"
        torch.save({"trainer": tr_full.state_dict(), "m_gen": tr_full.G._m_gen.get_state()}, ckpt_p)
        torch.save(batches[6:], b_p)

        for b in batches[6:]:
            tr_full.step(b)
        assert tr_full.opt_d.record.sparse_obs_bank.proto_k >= 1  # crossed step-8 activation & recentering!

        script = (
            "import torch\n"
            "from torch import nn\n"
            "from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior\n"
            "torch.set_num_threads(1)\n"
            f"dim, dz = {dim}, {dz}\n"
            "class _SparseMaskedG(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(nn.Linear(dz, 32), nn.SiLU(), nn.Linear(32, dim))\n"
            "        self._m_gen = torch.Generator().manual_seed(909)\n"
            "    def forward(self, z):\n"
            "        out = torch.tanh(self.net(z))\n"
            "        if self.training:\n"
            "            m = (torch.rand(out.shape, generator=self._m_gen) < 0.20).to(out.dtype)\n"
            "            return out * m\n"
            "        return out\n"
            "rc = get_recipe('unified_v15', num_particles=32, z_dim=dz, batch_size=32, prior_kind='mog', standardize=False)\n"
            "gm = _SparseMaskedG()\n"
            "dm = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))\n"
            "pr = MoGParticlePrior(32, dz, sigma=0.05, standardize=False, learnable=True)\n"
            "init.deterministic_orthogonal_(gm, seed=41)\n"
            "init.deterministic_orthogonal_(dm, seed=42)\n"
            "init.deterministic_orthogonal_(pr, seed=43)\n"
            "tr = GANTrainer(rc, gm, dm, prior=pr, seed=0, max_steps=20)\n"
            f"blob = torch.load({str(ckpt_p)!r}, weights_only=False)\n"
            "tr.load_state_dict(blob['trainer'])\n"
            "tr.G._m_gen.set_state(blob['m_gen'])\n"
            f"for b in torch.load({str(b_p)!r}, weights_only=False):\n"
            "    tr.step(b)\n"
            f"torch.save({{'g': [p.detach().clone() for p in tr.G.parameters()], 'd': [p.detach().clone() for p in tr.D.parameters()], 'z': unwrap_prior(tr.prior).z.detach().clone(), 'sp': tr.opt_d.record.sparse_obs_bank.to_packet()}}, {str(out_p)!r})\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        assert proc.returncode == 0
        res = torch.load(out_p, weights_only=False)
        for pa, pb in zip(tr_full.G.parameters(), res["g"]):
            assert torch.equal(pa.detach(), pb)
        for pa, pb in zip(tr_full.D.parameters(), res["d"]):
            assert torch.equal(pa.detach(), pb)
        assert torch.equal(unwrap_prior(tr_full.prior).z.detach(), res["z"])
        sp_full = tr_full.opt_d.record.sparse_obs_bank.to_packet()
        sp_res = res["sp"]
        assert sp_full.keys() == sp_res.keys()
        for k in sp_full:
            va, vb = sp_full[k], sp_res[k]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), f"Subprocess sparse_obs_bank mismatch on {k}"
            else:
                assert va == vb


def test_sparse_obs_g_loss_keeps_live_adversarial_critic_gradient() -> None:
    """Semantic control (AGENTS.md): sparse-masked g_loss must keep live fake_logits critic gradient and clean transient z_live."""
    torch.set_num_threads(1)
    dim, dz, bs = 128, 16, 32

    class _SparseG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(dz, dim)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            m = (torch.rand(z.shape[0], dim, generator=torch.Generator().manual_seed(777)) < 0.15).float()
            return torch.tanh(self.fc(z)) * m

    g_mod = _SparseG()
    d_mod = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))
    init.deterministic_orthogonal_(g_mod, seed=71)
    init.deterministic_orthogonal_(d_mod, seed=72)
    rec, _, _, opt_d = _make_opt_pair(g_mod, d_mod, batch_size=bs, z_dim=dz)
    loss_fn = rec.make_loss()

    rng = torch.Generator().manual_seed(778)
    z_live = torch.randn((bs, dz), generator=rng, requires_grad=True)
    x_r = torch.randn((bs, dim), generator=rng) * (torch.rand((bs, dim), generator=rng) < 0.15).float()

    # Warm up 8 steps so sparse prototypes are active too
    for _ in range(9):
        z_cur = torch.randn((bs, dz), generator=rng, requires_grad=True)
        dr = d_mod(x_r)
        df = d_mod(g_mod(z_cur))
        gl = loss_fn.g_loss(df, dr)
        assert opt_d.record._g_last_latent_live is None, "Transient _g_last_latent_live leaked after g_loss!"

    # 1. Verify direct autograd sensitivity to fake_logits (must NEVER be bypassed!)
    dr = d_mod(x_r)
    xf = g_mod(z_live)
    df = d_mod(xf)
    df.retain_grad()
    gl = loss_fn.g_loss(df, dr)
    assert opt_d.record._g_last_latent_live is None
    g_fake_logits = torch.autograd.grad(gl, df, retain_graph=True)[0]
    assert float(g_fake_logits.abs().max().item()) > 1e-4, "fake_logits received zero gradient (critic bypassed!)"

    # 2. Semantic damage control: perturbing critic weights must measurably change G's parameter gradient!
    g_mod.zero_grad()
    gl.backward()
    grad_g_normal = g_mod.fc.weight.grad.detach().clone()
    grad_z_normal = z_live.grad.detach().clone()
    assert float(grad_z_normal.abs().max().item()) > 1e-5, "Learnable latent z received zero gradient"

    with torch.no_grad():
        for p in d_mod.parameters():
            p.mul_(5.0)
    z_live2 = z_live.detach().clone().requires_grad_(True)
    g_mod.zero_grad()
    dr2 = d_mod(x_r)
    df2 = d_mod(g_mod(z_live2))
    gl2 = loss_fn.g_loss(df2, dr2)
    gl2.backward()
    grad_g_scaled_d = g_mod.fc.weight.grad.detach().clone()
    diff_g = float((grad_g_scaled_d - grad_g_normal).abs().max().item())
    assert diff_g > 1e-5, f"Perturbing critic had zero effect on G gradient (diff={diff_g:.2e})"

    # 3. Spherical Voronoi alignment term must be purely tangential to S^{dz-1} (zero radial distortion on z_live)
    z_test = torch.randn((bs, dz), generator=rng, requires_grad=True)
    xf_det = xf.detach()
    sp_z = unified_v15_sparse_obs_wiener_loss(
        xf_det, x_r, record=opt_d.record, z_batch=z_test.detach(), z_live=z_test, allow_update=False, z_pull=1.0
    )
    gz_only = torch.autograd.grad(sp_z, z_test)[0]
    radial_dot = (gz_only * z_test.detach()).sum(dim=1).abs().max().item()
    assert float(gz_only.abs().max().item()) > 1e-5
    assert radial_dot < 1e-5, f"Voronoi latent alignment leaked radial force ({radial_dot:.2e})"



