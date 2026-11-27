"""Unit, invariance, negative-control, and checkpoint-continuation tests for unified_v15 Law 2 & Law 3.

Covers:
- L2-4: Convex-EBI Jensen bias below kappa (zero-autograd.grad fwad_exact / fwad_debiased / secant_debiased).
- L2-5: Regime dispatch (low-D -> fwad_exact; Conv2d/BN/GN/LN/EmulatedLinear -> fwad_debiased;
        Attention/SDPA/Softmax -> secant_debiased) and D-11 b=1 guard.
- Zero forbidden primitives runtime spy (autograd.grad, eigvalsh, eigh, svd, svdvals, pinv)
  across MLP, GN, LN, BN, EmulatedLinear, and SDPA/Attention critics in both FP32 & BF16.
- D-09: Dimension-independent normal-axis dissipation floor (non-vanishing on r256) + negative control.
- D-04 (critic side): scale-homogeneous observe_scale at 1e-3 & 1e-5, stratified tail-inclusive probe
        eliminating b_p=2 prefix blindness, and eval-mode baseline parity for custom `self.training` generators.
- D-08: Screening dead-zone alignment in v15_zero_eigh_bipartite_deficit_macro_loss + covered-broad-mode guard.
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
    UnifiedV15StepRecord,
    record_unified_v15_generator_probe,
)
from particlegan._unified_v15.loss import (
    UnifiedV15CouplingLoss,
    v15_zero_eigh_bipartite_deficit_macro_loss,
)
from particlegan._unified_v15.penalty import (
    _debiased_fwad_sq,
    _prepare_batched_highd_tangent_plans,
)
from particlegan.low_precision import emulate_linear_training_


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
    """Layer-1 L2-4 contract: zero-autograd.grad fwad_debiased on 3x16x16 gives ratio <= 5.0 (no 1015x false trigger)."""
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
    assert pen.last_stats["grad_mode"] == "fwad_debiased"

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


def test_l2_5_regime_dispatch_and_sdpa_secant_fallback() -> None:
    """3-regime dispatch: low-D -> fwad_exact; Conv2d/GN -> fwad_debiased; SDPA/Attention -> secant_debiased."""
    torch.set_num_threads(1)

    # 1. Low-D (d <= 16) MLP -> fwad_exact (batched identity-basis fwAD)
    mlp8 = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    rec_m, _, _, opt_m = _make_opt_pair(nn.Linear(4, 8), mlp8, batch_size=8)
    pen_m = rec_m.make_critic_penalty(opt_m, collect_stats=True)
    x_8 = torch.randn((8, 8), generator=torch.Generator().manual_seed(100))
    val_m = pen_m(mlp8, x_8, x_8 * 0.9)
    val_m.backward()
    assert pen_m.last_stats["grad_mode"] == "fwad_exact"
    assert all(torch.isfinite(p.grad).all() for p in mlp8.parameters() if p.grad is not None)

    # 2. Pure Conv2d 128x128 DCGAN -> fwad_debiased
    dcgan = _PureConvDCGAN128Critic()
    rec_dc, _, _, opt_dc = _make_opt_pair(nn.Linear(4, 4), dcgan, batch_size=4)
    pen_dc = rec_dc.make_critic_penalty(opt_dc, collect_stats=True)
    x_128 = torch.randn((4, 3, 128, 128), generator=torch.Generator().manual_seed(101))
    val_dc = pen_dc(dcgan, x_128, x_128 * 0.9)
    val_dc.backward()
    assert pen_dc.last_stats["grad_mode"] == "fwad_debiased"
    assert all(torch.isfinite(p.grad).all() for p in dcgan.parameters() if p.grad is not None)

    # 3. GroupNorm + F.scaled_dot_product_attention -> secant_debiased (0 autograd.grad, finite grads)
    attn = _GroupNormAttentionCritic()
    rec_at, _, _, opt_at = _make_opt_pair(nn.Linear(4, 4), attn, batch_size=4)
    pen_at = rec_at.make_critic_penalty(opt_at, collect_stats=True)
    val_at = pen_at(attn, x_128, x_128 * 0.9)
    val_at.backward()
    assert pen_at.last_stats["grad_mode"] == "secant_debiased"
    assert all(torch.isfinite(p.grad).all() for p in attn.parameters() if p.grad is not None)

    # 4. D-11 b=1 edge case in _prepare_batched_highd_tangent_plans
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
# Hollow R^512 Self-Coupling & Zero-Forbidden-Op Curve Gate / Critics (FP32 & BF16)
# ---------------------------------------------------------------------------


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


def test_checkpoint_bit_exact_continuation_d32() -> None:
    """Fresh-object checkpoint save/restore must continue bit-exact on D=32 without removed solver banks."""
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

    rec_sd = tr_split.opt_d.record.state_dict()
    assert "highd_basin_bank" not in rec_sd["unified_v15"]
    assert "sparse_obs_bank" not in rec_sd["unified_v15"]

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


def test_checkpoint_rejects_incompatible_and_tampered_v15_packets() -> None:
    """Strict checkpoint validation must reject missing/extra/corrupted v15 packets and legacy v14 packets."""
    torch.set_num_threads(1)
    rec = UnifiedV15StepRecord()
    sd = rec.state_dict()
    assert "highd_basin_bank" not in sd["unified_v15"]
    assert "sparse_obs_bank" not in sd["unified_v15"]

    # Missing required key
    bad_missing = copy.deepcopy(sd)
    del bad_missing["unified_v15"]["cond_fiber_bank"]
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_missing)

    # Extra/legacy solver bank keys must be rejected
    bad_extra = copy.deepcopy(sd)
    bad_extra["unified_v15"]["highd_basin_bank"] = {}
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_extra)

    # Wrong formulation tag
    bad_form = copy.deepcopy(sd)
    bad_form["formulation"] = "unified_v14"
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_form)

    # Non-finite scalar tamper
    bad_nan = copy.deepcopy(sd)
    bad_nan["unified_v15"]["ema_data_scale"] = float("nan")
    with pytest.raises(ValueError):
        UnifiedV15StepRecord().load_state_dict(bad_nan)


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


def test_v15_no_32x32_cluster_cliff_and_g_loss_weight_applied() -> None:
    """Removed 32x32 cluster-ratio cliff: continuous & sprites both run spatial transport; BN & GN use fwad_debiased."""
    from benchmarks.toy_audit.api_four_laws_123 import sample_target_for_kind

    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(321)

    # 1. Pure Conv2d+BN 32x32 continuous activates spatial transport uniformly & uses fwad_debiased
    bn_dcgan32 = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1),
    )
    rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 4), bn_dcgan32, batch_size=16)
    pen = rec.make_critic_penalty(opt_d, collect_stats=True)
    loss_fn = rec.make_loss()
    loss_fn.record = opt_d.record

    xr_cont = torch.randn((16, 3, 32, 32), generator=g)
    xf_cont = torch.randn((16, 3, 32, 32), generator=g, requires_grad=True)
    dr_c = bn_dcgan32(xr_cont).squeeze(-1)
    df_c = bn_dcgan32(xf_cont).squeeze(-1)
    ld_c = loss_fn.d_loss(dr_c, df_c)
    lg_c = loss_fn.g_loss(df_c, dr_c)
    p_val = pen(bn_dcgan32, xr_cont, xf_cont.detach())
    assert torch.isfinite(ld_c) and torch.isfinite(lg_c) and torch.isfinite(p_val)
    assert opt_d.record.t1c_ell_initialized is True, "32x32 continuous must run transport uniformly (no cluster cliff)"
    assert pen.last_stats["grad_mode"] == "fwad_debiased", "Pure Conv2d+BN 32x32 must use fwad_debiased"

    # 2. GroupNorm 32x32 sprites also route to fwad_debiased AND activate spatial transport
    gn32 = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.GroupNorm(2, 8),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    rec_sp, _, _, opt_d_sp = _make_opt_pair(nn.Linear(4, 4), gn32, batch_size=16)
    pen_sp = rec_sp.make_critic_penalty(opt_d_sp, collect_stats=True)
    loss_sp = rec_sp.make_loss()
    loss_sp.record = opt_d_sp.record
    xr_sp = sample_target_for_kind("fl_law2_groupnorm_and_vit_sprites_v2", 16, g)
    xf_sp = torch.randn((16, 3, 32, 32), generator=g, requires_grad=True)
    dr_s = gn32(xr_sp).squeeze(-1)
    df_s = gn32(xf_sp).squeeze(-1)
    lg_s = loss_sp.g_loss(df_s, dr_s)
    p_sp = pen_sp(gn32, xr_sp, xf_sp.detach())
    assert torch.isfinite(lg_s) and torch.isfinite(p_sp)
    assert opt_d_sp.record.t1c_ell_initialized is True
    assert pen_sp.last_stats["grad_mode"] == "fwad_debiased", "GroupNorm 32x32 critic must use zero-autograd fwad_debiased"

    # 3. Large-B (B=128) active spatial transport in UnifiedV15CouplingLoss bounds Sinkhorn to <=64 & keeps full d_fake grad
    coup = UnifiedV15CouplingLoss()
    sp128 = sample_target_for_kind("fl_law2_groupnorm_and_vit_sprites_v2", 128, g)
    xf128 = torch.randn((128, 3, 32, 32), generator=g, requires_grad=True)
    df128 = torch.zeros(128, requires_grad=True)
    dr128 = torch.ones(128)
    lc, _ = coup(xf128, sp128, df128, dr128, return_metrics=False)
    assert torch.isfinite(lc)
    assert coup.last_p_xy_slack is not None and coup.last_p_xy_slack.shape[0] == 64
    lc.backward()
    assert df128.grad is not None and bool((df128.grad.abs() > 0).all())

    # 4. Missing self.t1c.weight * fix on UnifiedV15GANLoss.g_loss when dim exceeds transport cap (e.g. 3x64x64)
    crit64 = nn.Sequential(nn.Conv2d(3, 4, 8, stride=8), nn.Flatten(), nn.Linear(4 * 8 * 8, 1))
    rec_w = get_recipe("unified_v15", num_particles=16, z_dim=4, batch_size=4, loss="relativistic")
    _, opt_dw = rec_w.make_optimizers(nn.Linear(4, 4), crit64, rec_w.make_prior())
    loss_w1 = rec_w.make_loss()
    loss_w1.record = opt_dw.record
    loss_w1.t1c.weight = 1.0
    loss_w2 = rec_w.make_loss()
    loss_w2.record = opt_dw.record
    loss_w2.t1c.weight = 3.5
    xr64 = torch.randn((4, 3, 64, 64), generator=g)
    xf64 = torch.randn((4, 3, 64, 64), generator=g, requires_grad=True)
    dr64 = crit64(xr64).squeeze(-1)
    df64 = crit64(xf64).squeeze(-1)
    g1 = loss_w1.g_loss(df64, dr64)
    # Re-evaluate to refresh critic capture for second loss call
    dr64_b = crit64(xr64).squeeze(-1)
    df64_b = crit64(xf64).squeeze(-1)
    g2 = loss_w2.g_loss(df64_b, dr64_b)
    assert torch.allclose(g2, 3.5 * g1, atol=1e-5, rtol=1e-5), f"t1c.weight not applied: {g2.item()} vs {3.5 * g1.item()}"


def test_zero_forbidden_ops_across_critics_and_curve_gate_fp32_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime spy forbidding autograd.grad/eigvalsh/eigh/svd/svdvals/pinv across MLP, GN, LN, BN, EmulatedLinear & SDPA (FP32/BF16)."""
    torch.set_num_threads(1)

    def _forbid(*_a: object, **_kw: object) -> None:
        raise AssertionError("forbidden primitive invoked inside unified_v15 Law 2/3")

    monkeypatch.setattr(torch.autograd, "grad", _forbid)
    monkeypatch.setattr(torch.linalg, "eigvalsh", _forbid)
    monkeypatch.setattr(torch.linalg, "eigh", _forbid)
    monkeypatch.setattr(torch.linalg, "svd", _forbid)
    monkeypatch.setattr(torch.linalg, "svdvals", _forbid)
    monkeypatch.setattr(torch.linalg, "pinv", _forbid)
    monkeypatch.setattr(torch, "svd", _forbid)

    class _MHACritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(8, 2, batch_first=True)
            self.head = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x.view(x.shape[0], 4, 8)
            out, _ = self.mha(h, h, h, need_weights=False)
            return self.head(out.mean(dim=1)).squeeze(-1)

    class _TransformerCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True)
            self.head = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.enc(x.view(x.shape[0], 4, 8))
            return self.head(h.mean(dim=1)).squeeze(-1)

    class _SoftmaxCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(32, 16)
            self.sm = nn.Softmax(dim=-1)
            self.head = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.sm(self.proj(x))).squeeze(-1)

    rng = torch.Generator().manual_seed(2026)

    for dtype in (torch.float32, torch.bfloat16):
        # 1. Low-D MLP (d=8 <= 16) -> fwad_exact
        mlp8 = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1)).to(dtype)
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 8), mlp8, batch_size=6)
        pen = rec.make_critic_penalty(opt_d, collect_stats=True)
        xr = torch.randn((6, 8), generator=rng).to(dtype)
        xf = torch.randn((6, 8), generator=rng).to(dtype)
        val = pen(mlp8, xr, xf)
        val.backward()
        assert pen.last_stats["grad_mode"] == "fwad_exact"
        assert torch.isfinite(val) and all(torch.isfinite(p.grad).all() for p in mlp8.parameters() if p.grad is not None)

        # 2. High-D MLP (d=64) -> fwad_debiased
        mlp64 = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)).to(dtype)
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 64), mlp64, batch_size=6)
        pen = rec.make_critic_penalty(opt_d, collect_stats=True)
        xr64 = torch.randn((6, 64), generator=rng).to(dtype)
        xf64 = torch.randn((6, 64), generator=rng).to(dtype)
        val = pen(mlp64, xr64, xf64)
        val.backward()
        assert pen.last_stats["grad_mode"] == "fwad_debiased"
        assert torch.isfinite(val) and all(torch.isfinite(p.grad).all() for p in mlp64.parameters() if p.grad is not None)

        # 3. GroupNorm + Conv2d (3x16x16) -> fwad_debiased
        gn_c = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=1),
            nn.GroupNorm(2, 8),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        ).to(dtype)
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 4), gn_c, batch_size=4)
        pen = rec.make_critic_penalty(opt_d, collect_stats=True)
        xr_i = torch.randn((4, 3, 16, 16), generator=rng).to(dtype)
        xf_i = torch.randn((4, 3, 16, 16), generator=rng).to(dtype)
        val = pen(gn_c, xr_i, xf_i)
        val.backward()
        assert pen.last_stats["grad_mode"] == "fwad_debiased"
        assert torch.isfinite(val) and all(torch.isfinite(p.grad).all() for p in gn_c.parameters() if p.grad is not None)

        # 4. LayerNorm MLP -> fwad_debiased
        ln_c = nn.Sequential(nn.Linear(32, 16), nn.LayerNorm(16), nn.SiLU(), nn.Linear(16, 1)).to(dtype)
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 32), ln_c, batch_size=6)
        pen = rec.make_critic_penalty(opt_d, collect_stats=True)
        xr32 = torch.randn((6, 32), generator=rng).to(dtype)
        xf32 = torch.randn((6, 32), generator=rng).to(dtype)
        val = pen(ln_c, xr32, xf32)
        val.backward()
        assert pen.last_stats["grad_mode"] == "fwad_debiased"
        assert torch.isfinite(val) and all(torch.isfinite(p.grad).all() for p in ln_c.parameters() if p.grad is not None)

        # 5. BatchNorm Conv2d -> fwad_debiased + BN eval/train mode preserved + running stats untouched + no mixed-batch explosion
        bn_c = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        ).to(dtype)
        bn_c.train()
        bn_layer = bn_c[1]
        assert isinstance(bn_layer, nn.BatchNorm2d)
        # Prime running stats with a real forward pass
        _ = bn_c(xr_i)
        tracked_before = int(bn_layer.num_batches_tracked.item())
        rm_before = bn_layer.running_mean.detach().clone()
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 4), bn_c, batch_size=4)
        pen = rec.make_critic_penalty(opt_d, collect_stats=True)
        # Shifted fake batch (mean +5) would trigger catastrophic mixed-batch BN explosion if real+fake were concatenated!
        val_bn = pen(bn_c, xr_i, xf_i + 5.0)
        val_bn.backward()
        assert pen.last_stats["grad_mode"] == "fwad_debiased"
        assert bn_layer.training is True, "BN training mode was not restored after penalty"
        assert int(bn_layer.num_batches_tracked.item()) == tracked_before, "Penalty mutated BN num_batches_tracked"
        assert torch.equal(bn_layer.running_mean, rm_before), "Penalty mutated BN running_mean"
        assert torch.isfinite(val_bn) and all(torch.isfinite(p.grad).all() for p in bn_c.parameters() if p.grad is not None)

        # 6. EmulatedLinear critic -> strictly zero autograd.grad (fwAD when jvp present, secant fallback otherwise)
        emu_c = emulate_linear_training_(
            nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1)).to(dtype),
            "mxfp8",
        )
        rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 32), emu_c, batch_size=6)
        pen = rec.make_critic_penalty(opt_d, collect_stats=True)
        val_emu = pen(emu_c, xr32, xf32)
        val_emu.backward()
        assert pen.last_stats["grad_mode"] in ("fwad_debiased", "secant_debiased")
        assert torch.isfinite(val_emu) and all(torch.isfinite(p.grad).all() for p in emu_c.parameters() if p.grad is not None)

        # 7. Attention / SDPA / Softmax critics -> secant_debiased (0 autograd.grad, finite grads)
        for att_c, x_a in (
            (_GroupNormAttentionCritic().to(dtype), torch.randn((4, 3, 16, 16), generator=rng).to(dtype)),
            (_MHACritic().to(dtype), xr32),
            (_TransformerCritic().to(dtype), xr32),
            (_SoftmaxCritic().to(dtype), xr32),
        ):
            rec, _, _, opt_d = _make_opt_pair(nn.Linear(4, 4), att_c, batch_size=x_a.shape[0])
            pen = rec.make_critic_penalty(opt_d, collect_stats=True)
            val_a = pen(att_c, x_a, x_a * 0.9)
            val_a.backward()
            assert pen.last_stats["grad_mode"] == "secant_debiased"
            assert torch.isfinite(val_a) and all(torch.isfinite(p.grad).all() for p in att_c.parameters() if p.grad is not None)

    # 8. _compute_curve_gate & simplicial_manifold_loss run with 0 eigvalsh / 0 svd and separate 1D curve vs 2D cloud
    coup = UnifiedV15CouplingLoss()
    t = torch.linspace(-2.0, 2.0, 64)
    y_curve = torch.stack((t, torch.sin(t)), dim=1)
    y_blob = torch.randn((64, 2), generator=rng)
    g_curve = coup._compute_curve_gate(y_curve, g_conn=1.0)
    g_blob = coup._compute_curve_gate(y_blob, g_conn=1.0)
    assert g_curve > 0.70, f"Zero-eigvalsh curve gate missed 1D curve: {g_curve:.3f}"
    assert g_blob < 0.35, f"Zero-eigvalsh curve gate false-triggered on 2D blob: {g_blob:.3f}"
    g_sim = nn.Linear(4, 2)
    z_tbl = torch.randn((32, 4), generator=rng, requires_grad=True)
    sim_loss = coup.simplicial_manifold_loss(z_tbl, g_sim, y_curve, chart_weight=1.0)
    assert torch.isfinite(sim_loss)





