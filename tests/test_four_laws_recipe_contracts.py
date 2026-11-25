"""Layer-1 Parametrized Four-Laws Mathematical Contract Suite (<15s CPU).

Parametrized across ``CONTRACT_RECIPES = ("unified_v11", "unified_v12", "unified_v13", "unified_v14", "unified_v15")``.

CI Contract:
  Every Four-Laws invariant currently violated by a recipe MUST be marked with
  ``@pytest.mark.xfail(strict=True, reason="Open Four-Laws defect D-XX — target for unified_v15")``
  — NEVER soften an assertion or fake a PASS. That keeps CI 100% green today and
  automatically flags ``XPASS`` the moment a future recipe closes the defect.
"""
from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init, unwrap_prior
from particlegan._t1_coupling.charts import laguerre_hard_commitment
from particlegan._unified_v4.optimizer import _band_scale
from particlegan._unified_v11.critic import record_unified_v11_generator_probe
from particlegan._unified_v11.optimizer import quintic_orthogonalize_2d
from particlegan._unified_v15.charts import laguerre_hard_commitment as laguerre_hard_commitment_v15
from particlegan._unified_v15.critic import record_unified_v15_generator_probe
from particlegan._unified_v15.optimizer import (
    _band_scale as _band_scale_v15,
    quintic_orthogonalize_2d as quintic_orthogonalize_2d_v15,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_RECIPES: tuple[str, ...] = (
    "unified_v11",
    "unified_v12",
    "unified_v13",
    "unified_v14",
    "unified_v15",
)
ALL_CURRENT = frozenset(("unified_v11", "unified_v12", "unified_v13", "unified_v14"))


def _parametrize_recipes(
    *,
    xfail_for: frozenset[str] = frozenset(),
    reason: str = "",
):
    params = [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=reason))
        if name in xfail_for
        else pytest.param(name)
        for name in CONTRACT_RECIPES
    ]
    return pytest.mark.parametrize("recipe_name", params)


def _step_gen_loss(opt_g: Any, loss: torch.Tensor) -> None:
    opt_g.zero_grad(set_to_none=True)
    constraint_geometry_backward(loss, opt_g, (loss,))
    opt_g.step()


def _build_simple_opt_pair(
    recipe_name: str,
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
    rec = get_recipe(recipe_name, **kw)
    prior = rec.make_prior()
    opt_g, opt_d = rec.make_optimizers(g_mod, d_mod, prior)
    return rec, prior, opt_g, opt_d


# ---------------------------------------------------------------------------
# D-02: s=0 Quintic Orthogonalization target_frob Blowup & BF16 Leakage
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-02 (s=0 target_frob sqrt(m) blowup & m>=64 BF16 leakage) — target for unified_v15",
)
def test_d02_s0_quintic_rank1_frob_blowup_and_bf16_leakage(recipe_name: str) -> None:
    """Polar direction of a rank-1 matrix must keep sigma_1 <= 1.25 (not sqrt(m)) and avoid m=64 BF16 leakage."""
    torch.set_num_threads(1)
    m = 64
    q_u, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(201), dtype=torch.float64))
    q_v, _ = torch.linalg.qr(torch.randn(m, m, generator=torch.Generator().manual_seed(202), dtype=torch.float64))
    g_r1 = torch.outer(q_u[:, 0], q_v[:, 0]).float()

    # Exercise the live recipe generator optimizer when s_base == 0 (including v14's fg_damp at ema_fg=0.15):
    g_mod = nn.Linear(m, m, bias=False)
    with torch.no_grad():
        g_mod.weight.zero_()
    _, _, opt_g, _ = _build_simple_opt_pair(recipe_name, g_mod, nn.Linear(m, 1, bias=False))
    if hasattr(opt_g, "_ema_func_gain"):
        opt_g._ema_func_gain = 0.15  # Drives fg_damp -> 1.0 (v14) or stiff_gate -> 1.0 (v12/v13)
    lr = float(opt_g.param_groups[0]["lr"])
    _step_gen_loss(opt_g, (g_mod.weight * g_r1).sum())
    dw = g_mod.weight.detach().float()
    s1_normed = float(torch.linalg.svdvals(dw)[0].item()) / max(lr, 1e-12)

    # Also check direct s=0 quintic kernel & BF16 threshold guard
    q_fn = quintic_orthogonalize_2d_v15 if recipe_name == "unified_v15" else quintic_orthogonalize_2d
    out_s0 = q_fn(g_r1, steps=5, smoothing=0.0)
    s1_kernel = float(torch.linalg.svdvals(out_s0.double())[0].item())
    src = inspect.getsource(q_fn)
    assert s1_kernel <= 1.25, f"{recipe_name}: s=0 quintic rank-1 sigma_1 blew up to {s1_kernel:.2f}"
    assert s1_normed <= 1.25, f"{recipe_name}: optimizer s_base=0 rank-1 sigma_1/lr blew up to {s1_normed:.2f}"
    assert "min(r_dim, c_dim) >= 64" not in src, "s=0 branch still casts m>=64 to BF16 (77.8% leakage)"


# ---------------------------------------------------------------------------
# D-03: 1e-25 Raw Norm Underflow & Unscaled 1D + eps Crush
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-03 (1e-25 raw norm underflow in v11 and _step_generator_free_critic) — target for unified_v15",
)
def test_d03_1e25_raw_norm_underflow(recipe_name: str) -> None:
    """peak_rms must remain strictly positive at 1e-25 gradient scale on BOTH neural-G and generator-free critic."""
    torch.set_num_threads(1)
    base_g = torch.randn(16, 16, generator=torch.Generator().manual_seed(301))

    # 1. Standard neural-G path
    g_mod = nn.Linear(16, 16, bias=False)
    d_mod = nn.Linear(16, 1, bias=False)
    _, _, opt_g, _ = _build_simple_opt_pair(recipe_name, g_mod, d_mod)
    _step_gen_loss(opt_g, (g_mod.weight * (base_g * 1e-25)).sum())
    st_g = opt_g.state[g_mod.weight]
    pk_g = st_g.get("peak_rms", st_g.get("peak_rms_val", 0.0))
    pk_g_val = float(pk_g.item()) if isinstance(pk_g, torch.Tensor) else float(pk_g)
    assert pk_g_val > 0.0, f"{recipe_name}: neural-G peak_rms underflowed to 0.0 at 1e-25"

    # 2. Generator-free critic path (G = nn.Identity())
    d_gf = nn.Linear(16, 16, bias=False)
    _, _, _, opt_d_gf = _build_simple_opt_pair(recipe_name, nn.Identity(), d_gf)
    d_gf.weight.grad = base_g * 1e-25
    opt_d_gf.step()
    st_d = opt_d_gf.state[d_gf.weight]
    pk_d = st_d.get("peak_rms", st_d.get("peak_rms_val", 0.0))
    pk_d_val = float(pk_d.item()) if isinstance(pk_d, torch.Tensor) else float(pk_d)
    assert pk_d_val > 0.0, f"{recipe_name}: generator-free critic peak_rms underflowed to 0.0 at 1e-25"


@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-03 (unscaled 1D + eps crush under s_base=0 / fg_damp / gen-free) — target for unified_v15",
)
def test_d03_unscaled_1d_eps_crush_at_1e12(recipe_name: str) -> None:
    """1D vector parameters must remain degree-0 scale-invariant at 1e-12 vs 1e-6 on default gen-free & s_base=0 paths."""
    torch.set_num_threads(1)
    v_grad = torch.linspace(0.5, 1.5, 16)

    def _run_1d(scale: float, *, mode: str) -> float:
        p = nn.Parameter(torch.zeros(16))
        if mode == "gen_free_default":
            # Default smoothing=0.0061 hits clamp_min(eps=1e-8) in _step_generator_free_critic (D-03):
            d_mod = nn.Module()
            d_mod.register_parameter("bias_1d", p)
            _, _, _, opt = _build_simple_opt_pair(recipe_name, nn.Identity(), d_mod)
            p.grad = v_grad * scale
            opt.step()
        elif mode == "s0_fast_path":
            # smoothing=0.0 hits denom = g_norm + eps in 1D fast path (D-03):
            g_mod = nn.Module()
            g_mod.register_parameter("bias_1d", p)
            _, _, opt, _ = _build_simple_opt_pair(recipe_name, g_mod, nn.Linear(4, 1), smoothing=0.0)
            _step_gen_loss(opt, (p * (v_grad * scale)).sum())
        else:
            # fg_damp at ema_fg=0.15 drives s_base -> 0.0 on v14 even under default smoothing=0.0061:
            g_mod = nn.Module()
            g_mod.register_parameter("bias_1d", p)
            _, _, opt, _ = _build_simple_opt_pair(recipe_name, g_mod, nn.Linear(4, 1))
            if hasattr(opt, "_ema_func_gain"):
                opt._ema_func_gain = 0.15
            _step_gen_loss(opt, (p * (v_grad * scale)).sum())
        return float(p.detach().norm().item())

    for mode in ("gen_free_default", "s0_fast_path", "fg_damp_default"):
        m_hi = _run_1d(1e-6, mode=mode)
        m_lo = _run_1d(1e-12, mode=mode)
        ratio = m_lo / max(m_hi, 1e-30)
        assert ratio >= 0.95, f"{recipe_name} ({mode}): 1D step crushed at 1e-12 (ratio={ratio:.4e})"


# ---------------------------------------------------------------------------
# D-04: _band_scale Non-C^1 Overshoot & coord_scale 0.05 Floor Distortion
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-04 (non-C^1 one-shot _band_scale saturating overshoot) — target for unified_v15",
)
def test_d04_band_scale_non_c1_and_saturating_overshoot(recipe_name: str) -> None:
    """Post-rescale generator output displacement on a saturating tanh map must stay <= 1.10 * delta_g_max."""
    torch.set_num_threads(1)

    class _SatTanh(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([[3.5, 0.0], [0.0, 3.5]], dtype=torch.float32))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return 2.0 * torch.tanh(z @ self.w.T)

    g_mod = _SatTanh()
    d_mod = nn.Linear(2, 1)
    _, _, opt_g, opt_d = _build_simple_opt_pair(recipe_name, g_mod, d_mod, smoothing=0.0)
    rec_obj = getattr(opt_d, "record", None)
    assert rec_obj is not None
    opt_g._unified_v11_record = rec_obj
    if recipe_name == "unified_v15":
        opt_g._unified_v15_record = rec_obj
    rec_obj.ema_data_scale = math.sqrt(2.0)  # coord_scale = 1.0
    rec_obj.ema_push_res = 0.0  # delta_g_max = 0.045
    rec_obj._thermal_scale = 1.0

    z_probe = torch.ones((16, 2), dtype=torch.float32)
    y_pre = g_mod(z_probe)
    probe_fn = record_unified_v15_generator_probe if recipe_name == "unified_v15" else record_unified_v11_generator_probe
    probe_fn(g_mod, (z_probe,), {}, y_pre, rec_obj, require_module_training=True)
    # Drive large opposing gradient pulling weights across tanh saturation knee:
    opt_g.param_groups[0]["lr"] = 3.6
    _step_gen_loss(opt_g, g_mod.w.sum())
    with torch.no_grad():
        y_post = g_mod(z_probe)
    realized = float((y_post - y_pre.detach()).square().mean().sqrt().item())
    delta_g_max = 0.045

    # Also check _band_scale C^1 continuity at delta_max:
    band_fn = _band_scale_v15 if recipe_name == "unified_v15" else _band_scale
    eps_fd = 1e-5
    d_left = (band_fn(0.20, 0.0, 0.20) - band_fn(0.20 - eps_fd, 0.0, 0.20)) / eps_fd
    d_right = (band_fn(0.20 + eps_fd, 0.0, 0.20) - band_fn(0.20, 0.0, 0.20)) / eps_fd
    assert abs(d_left - d_right) < 1e-2, f"_band_scale has C^1 kink at delta_max: {d_left:.3f} vs {d_right:.3f}"
    assert realized <= 1.10 * delta_g_max, (
        f"{recipe_name}: one-shot _band_scale overshot delta_g_max={delta_g_max:.3f} -> realized={realized:.4f}"
    )


@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-04 (coord_scale 0.05 floor 50x distortion on small-RMS data) — target for unified_v15",
)
def test_d04_coord_scale_floor_distortion(recipe_name: str) -> None:
    """Data-scale tracker must be scale-homogeneous on small per-coordinate RMS (1e-3), not clamped at 0.05."""
    torch.set_num_threads(1)
    _, _, _, opt_d = _build_simple_opt_pair(recipe_name, nn.Linear(4, 2), nn.Linear(2, 1))
    rec_obj = getattr(opt_d, "record", None)
    assert rec_obj is not None
    x_tiny = torch.randn(64, 2, generator=torch.Generator().manual_seed(404)) * 1e-3
    true_coord_rms = float(x_tiny.square().mean().sqrt().item())
    rec_obj.observe_scale(x_tiny, x_tiny)
    tracked_coord_scale = float(rec_obj.ema_data_scale) / math.sqrt(2.0)
    assert tracked_coord_scale <= 1.25 * true_coord_rms, (
        f"{recipe_name}: coord_scale distorted by {tracked_coord_scale / true_coord_rms:.1f}x at rms=1e-3"
    )


# ---------------------------------------------------------------------------
# D-05: Prior peak_rms Live Usage (Defect F), Ceiling-Free Degree-0 (Defect A),
#       and p_dec=1.0 Permanent Spike Freeze (Defect B)
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=frozenset({"unified_v11", "unified_v12", "unified_v13"}),
    reason="Open Four-Laws defect D-05 (Defect F: dead prior peak_rms freezing 1e-5 priors) — fixed in unified_v14",
)
def test_d05_defect_f_prior_peak_rms_live_usage(recipe_name: str) -> None:
    """Prior peak_rms must actively prevent high-D 1e-5 prior freeze (PASS on v14, xfail on v11/v12/v13)."""
    torch.set_num_threads(1)
    rec = get_recipe(recipe_name, num_particles=256, z_dim=16, batch_size=128)
    prior = MoGParticlePrior(256, 16, sigma=0.1, standardize=False, learnable=True)
    init.deterministic_orthogonal_(prior, seed=0)
    g_mod = nn.Linear(16, 16)
    d_mod = nn.Linear(16, 1)
    opt_g, _ = rec.make_optimizers(g_mod, d_mod, prior)

    z0 = prior.z.detach().clone()
    rows = torch.arange(256, dtype=torch.long)
    grad_tiny = torch.randn(256, 16, generator=torch.Generator().manual_seed(501)) * 1e-5
    opt_g.set_sampled_rows(prior.z, rows)
    _step_gen_loss(opt_g, (prior.z * grad_tiny).sum())
    mean_dz = float((prior.z.detach() - z0).norm(dim=1).mean().item())
    assert mean_dz >= 0.01, f"{recipe_name}: prior frozen at 1e-5 grad scale (mean_dz={mean_dz:.6f})"

    # Semantic control: mutating st["peak_rms"] must measurably change the next step!
    z_a = prior.z.detach().clone()
    opt_g.set_sampled_rows(prior.z, rows)
    _step_gen_loss(opt_g, (prior.z * (grad_tiny * 0.1)).sum())
    disp_normal = float((prior.z.detach() - z_a).norm(dim=1).mean().item())

    with torch.no_grad():
        prior.z.copy_(z_a)
        opt_g.state[prior.z]["peak_rms"].fill_(1e-2)
    opt_g.set_sampled_rows(prior.z, rows)
    _step_gen_loss(opt_g, (prior.z * (grad_tiny * 0.1)).sum())
    disp_mutated = float((prior.z.detach() - z_a).norm(dim=1).mean().item())
    assert abs(disp_normal - disp_mutated) > 1e-3, f"{recipe_name}: prior peak_rms state is dead code"


@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-05 (Defect A: s_base cap breaks trajectory degree-0 across 1e-6..1e+4) — target for unified_v15",
)
def test_d05_defect_a_ceiling_free_trajectory_degree0(recipe_name: str) -> None:
    """Multi-step trajectory must be degree-0 scale-invariant across 10 decades (1e-6 .. 1e+4)."""
    torch.set_num_threads(1)
    g_seq = [
        torch.randn(8, 8, generator=torch.Generator().manual_seed(510 + t)) * (0.2 ** (t // 3))
        for t in range(8)
    ]

    def _run_scale(scale: float) -> torch.Tensor:
        g_mod = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            g_mod.weight.copy_(torch.eye(8))
        _, _, opt_g, _ = _build_simple_opt_pair(recipe_name, g_mod, nn.Linear(8, 1))
        for grad_t in g_seq:
            _step_gen_loss(opt_g, (g_mod.weight * (grad_t * scale)).sum())
        return g_mod.weight.detach().clone()

    w_lo = _run_scale(1e-6)
    w_hi = _run_scale(1e4)
    max_diff = float((w_lo - w_hi).abs().max().item())
    assert max_diff < 1e-4, f"{recipe_name}: trajectory degree-0 broken across 1e-6..1e+4 (max_diff={max_diff:.4f})"


@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-05 (Defect B: p_dec=1.0 permanent freeze after 1e4x gradient spike) — target for unified_v15",
)
def test_d05_defect_b_spike_freeze_recovery(recipe_name: str) -> None:
    """A single transient 1e4x gradient spike at scale 1e-6 must not permanently freeze subsequent steps."""
    torch.set_num_threads(1)
    g_mod = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        g_mod.weight.copy_(torch.eye(8))
    _, _, opt_g, _ = _build_simple_opt_pair(recipe_name, g_mod, nn.Linear(8, 1))
    base_g = torch.randn(8, 8, generator=torch.Generator().manual_seed(520))

    for _ in range(3):
        _step_gen_loss(opt_g, (g_mod.weight * (base_g * 1e-6)).sum())
    w_pre = g_mod.weight.detach().clone()
    _step_gen_loss(opt_g, (g_mod.weight * (base_g * 1e-6)).sum())
    pre_step = float((g_mod.weight.detach() - w_pre).norm().item())

    # Inject single 1e4x transient spike:
    _step_gen_loss(opt_g, (g_mod.weight * (base_g * 1e-2)).sum())
    post_step = 0.0
    for _ in range(100):
        w_before = g_mod.weight.detach().clone()
        _step_gen_loss(opt_g, (g_mod.weight * (base_g * 1e-6)).sum())
        post_step = float((g_mod.weight.detach() - w_before).norm().item())

    ratio = post_step / max(pre_step, 1e-12)
    assert ratio >= 0.20, f"{recipe_name}: permanently frozen after transient spike (post/pre={ratio:.5f})"


# ---------------------------------------------------------------------------
# D-06 / D-07: Prior w_fake_ema Mass, Gersho Merge LTV, and B_f < K Eviction
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-07 (w_fake_ema unnormalized after uncovered-mode teleport) — target for unified_v15",
)
def test_d07_w_fake_ema_mass_conservation(recipe_name: str) -> None:
    """Prior regularizer w_fake_ema must conserve unit total mass after uncovered-mode teleportation."""
    torch.set_num_threads(1)
    rec = get_recipe(recipe_name, num_particles=16, z_dim=2, batch_size=32)
    prior = MoGParticlePrior(16, 2, sigma=0.05, standardize=False, learnable=True)
    reg = rec.make_prior_regularizer(prior=prior, reference=prior)

    # Seed quantizer cleanly on 1 cluster, then inject a far uncovered outlier to trigger teleportation:
    g = torch.Generator().manual_seed(701)
    x_init = torch.randn(32, 2, generator=g) * 0.1
    reg._update_quantizer(x_init)
    x_outlier = x_init.clone()
    x_outlier[0] = torch.tensor([50.0, 50.0])
    reg._update_quantizer(x_outlier)
    w_sum = float(reg.w_fake_ema.sum().item())
    assert abs(w_sum - 1.0) < 1e-4, f"{recipe_name}: w_fake_ema mass drifted to {w_sum:.6f}"


@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-07 (Gersho merge omits proto_var LTV update) — target for unified_v15",
)
def test_d07_gersho_merge_proto_var_ltv_conservation(recipe_name: str) -> None:
    """Gersho merge-before-split must update merged cell proto_var[a] via the exact Law of Total Variance."""
    torch.set_num_threads(1)
    rec = get_recipe(recipe_name, num_particles=4, z_dim=2, batch_size=16)
    prior = MoGParticlePrior(4, 2, sigma=0.05, standardize=False, learnable=True)
    reg = rec.make_prior_regularizer(prior=prior, reference=prior)

    # Construct 4 active prototypes (0 dead) where (0, 1) are close enough to merge while 3 is wide to split:
    reg.quantizer_initialized = True
    reg.proto_c = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    reg.proto_var = torch.tensor([0.05, 0.05, 0.05, 2.0], dtype=torch.float32)
    reg.proto_w = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    reg.w_fake_ema = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    reg.proto_hits = torch.tensor([50.0, 50.0, 50.0, 50.0], dtype=torch.float32)
    reg.psi = torch.zeros(4, dtype=torch.float32)
    reg.steps = 15  # Next step triggers split/merge cadence

    # Compute exact Pythagorean LTV for merging cells 0 & 1 (equal mass, ||c0 - c1||^2 = 1.0, dim = 2):
    expected_v0 = 0.05 + (0.25 * 0.25 / (0.50 * 0.50)) * (1.0 / 2.0)  # 0.05 + 0.125 = 0.175
    x_batch = reg.proto_c.repeat(4, 1)
    reg._update_quantizer(x_batch)
    actual_v0 = float(reg.proto_var[0].item())
    assert abs(actual_v0 - expected_v0) < 0.02, (
        f"{recipe_name}: Gersho merge lost LTV variance (actual={actual_v0:.4f} vs expected={expected_v0:.4f})"
    )


@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect D-06 (B_f < K deterministic -1e-7*arange(k) index-bias eviction) — target for unified_v15",
)
def test_d06_bf_lt_k_zero_eviction(recipe_name: str) -> None:
    """When B_f < K with uniform weights, exact-match particles on high-index charts must not be evicted."""
    torch.set_num_threads(1)
    charts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    x_exact_high = torch.tensor([[10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    laguerre_fn = laguerre_hard_commitment_v15 if recipe_name == "unified_v15" else laguerre_hard_commitment
    _, _, winners = laguerre_fn(
        x_exact_high,
        charts,
        torch.full((4,), 0.25),
        torch.ones(4),
        torch.tensor(0.5),
    )
    assert set(winners.tolist()) == {2, 3}, f"Exact-match high-index particles evicted to {winners.tolist()}"


# ---------------------------------------------------------------------------
# 4D 1x1 Conv2d == 2D Linear Exact Single-Step Update Parity
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=frozenset({"unified_v12", "unified_v13", "unified_v14"}),
    reason="Open Four-Laws defect (4D 1x1 Conv2d omits G-NS & D sqrt(m) -> 11.3x step gap) — target for unified_v15",
)
def test_4d_conv1x1_equals_2d_linear_single_step_parity(recipe_name: str) -> None:
    """A 1x1 ungrouped Conv2d(128,128,1) must take the exact same single step as Linear(128,128) on G and D."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(601)
    w0 = torch.randn((128, 128), generator=g) * 0.05
    grad0 = torch.randn((128, 128), generator=g) * 0.02

    g_lin = nn.Linear(128, 128, bias=False)
    d_lin = nn.Linear(128, 128, bias=False)
    g_c1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
    d_c1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
    with torch.no_grad():
        g_lin.weight.copy_(w0)
        d_lin.weight.copy_(w0)
        g_c1.weight.copy_(w0.view(128, 128, 1, 1))
        d_c1.weight.copy_(w0.view(128, 128, 1, 1))

    _, _, og_l, od_l = _build_simple_opt_pair(recipe_name, g_lin, d_lin)
    _, _, og_c, od_c = _build_simple_opt_pair(recipe_name, g_c1, d_c1)

    _step_gen_loss(og_l, (g_lin.weight * grad0).sum())
    _step_gen_loss(og_c, (g_c1.weight * grad0.view(128, 128, 1, 1)).sum())
    d_lin.weight.grad = grad0.clone()
    od_l.step()
    d_c1.weight.grad = grad0.view(128, 128, 1, 1).clone()
    od_c.step()

    dg_lin = g_lin.weight.detach() - w0
    dg_c1 = g_c1.weight.detach().view(128, 128) - w0
    dd_lin = d_lin.weight.detach() - w0
    dd_c1 = d_c1.weight.detach().view(128, 128) - w0

    rel_g = float((dg_c1 - dg_lin).norm().item() / max(float(dg_lin.norm().item()), 1e-12))
    rel_d = float((dd_c1 - dd_lin).norm().item() / max(float(dd_lin.norm().item()), 1e-12))
    assert rel_g < 1e-3, f"{recipe_name}: G 1x1 Conv2d vs 2D Linear single-step gap rel={rel_g:.4f}"
    assert rel_d < 1e-3, f"{recipe_name}: D 1x1 Conv2d vs 2D Linear single-step gap rel={rel_d:.4f}"


# ---------------------------------------------------------------------------
# L2-4: Convex-EBI 1-Probe Jensen Bias Check
# ---------------------------------------------------------------------------

@_parametrize_recipes(
    xfail_for=ALL_CURRENT,
    reason="Open Four-Laws defect L2-4 (1-probe convex-EBI Jensen false triggering below kappa) — target for unified_v15",
)
def test_l2_4_convex_ebi_jensen_bias_below_kappa(recipe_name: str) -> None:
    """On a linear image critic with true ||grad_x D||_2 = 0.80 < kappa=1.0, E[pen] must not exceed 5x exact EBI."""
    torch.set_num_threads(1)

    class _LinearImgCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            u = torch.randn(3 * 16 * 16, generator=torch.Generator().manual_seed(777))
            self.w = nn.Parameter(u / u.norm() * 0.80)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1) @ self.w

    crit = _LinearImgCritic()
    rec, _, _, opt_d = _build_simple_opt_pair(recipe_name, nn.Linear(4, 4), crit)
    pen = rec.make_critic_penalty(opt_d)

    g = torch.Generator().manual_seed(778)
    xr = torch.randn((8, 3, 16, 16), generator=g)
    xf = torch.randn((8, 3, 16, 16), generator=g)
    xr = xr - xr.mean(dim=0, keepdim=True)
    xf = xf - xf.mean(dim=0, keepdim=True)

    obs_vals = []
    for k in range(24):
        torch.manual_seed(900 + k)
        obs_vals.append(float(pen(crit, xr, xf).detach().item()))
    mean_pen = sum(obs_vals) / len(obs_vals)

    reg = pen.regularizer
    dim = float(3 * 16 * 16)
    exact_ebi = float(F.softplus(torch.tensor(0.80 - float(reg.kappa)), beta=float(reg.beta)).square().item())
    exact_floor = (0.0005 / dim) * (0.80 ** 2) if float(reg.lambda_floor) > 0.0 else 0.0
    exact_total = 0.5 * float(pen.coefficient) * (float(reg.real_weight) + float(reg.fake_weight)) * (exact_ebi + exact_floor)

    ratio = mean_pen / max(exact_total, 1e-12)
    assert ratio <= 5.0, (
        f"{recipe_name}: 1-probe convex-EBI Jensen over-penalization {ratio:.1f}x below kappa "
        f"(mean={mean_pen:.6f} vs exact={exact_total:.6f})"
    )


# ---------------------------------------------------------------------------
# Clock-Free Invariance & Checkpoint Bit-Exact Continuation (MUST PASS on all!)
# ---------------------------------------------------------------------------

@_parametrize_recipes()
def test_clock_free_invariance(recipe_name: str) -> None:
    """Horizon changes (max_steps/total_steps = 40 vs 4000) must not alter early-step dynamics."""
    torch.set_num_threads(1)

    def _run(horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = get_recipe(
            recipe_name,
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
        data_rng = torch.Generator().manual_seed(1234)
        for _ in range(6):
            tr.step(torch.randn((32, 2), generator=data_rng))
        return (
            torch.cat([p.detach().reshape(-1) for p in g_mod.parameters()]),
            torch.cat([p.detach().reshape(-1) for p in d_mod.parameters()]),
            unwrap_prior(tr.prior).z.detach().clone(),
        )

    g_short, d_short, z_short = _run(40)
    g_long, d_long, z_long = _run(4000)
    assert torch.allclose(g_short, g_long, atol=1e-6, rtol=1e-6)
    assert torch.allclose(d_short, d_long, atol=1e-6, rtol=1e-6)
    assert torch.allclose(z_short, z_long, atol=1e-6, rtol=1e-6)


@_parametrize_recipes()
def test_checkpoint_save_restore_bit_exact_continuation(recipe_name: str) -> None:
    """Fresh-object checkpoint save/restore must continue bit-exact for every recipe."""
    torch.set_num_threads(1)

    def _make_trainer() -> tuple[GANTrainer, torch.Generator]:
        rec = get_recipe(
            recipe_name,
            num_particles=32,
            z_dim=4,
            batch_size=32,
            prior_kind="mog",
            standardize=False,
        )
        g_mod = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
        d_mod = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        prior = MoGParticlePrior(32, 4, sigma=0.1, standardize=False, learnable=True)
        init.deterministic_orthogonal_(g_mod, seed=11)
        init.deterministic_orthogonal_(d_mod, seed=21)
        init.deterministic_orthogonal_(prior, seed=31)
        tr = GANTrainer(rec, g_mod, d_mod, prior=prior, seed=0, max_steps=20)
        return tr, torch.Generator().manual_seed(5678)

    tr_full, rng_full = _make_trainer()
    tr_split, rng_split = _make_trainer()

    for _ in range(4):
        b_f = torch.randn((32, 2), generator=rng_full)
        b_s = torch.randn((32, 2), generator=rng_split)
        tr_full.step(b_f)
        tr_split.step(b_s)

    saved = tr_split.state_dict()
    tr_resumed, _ = _make_trainer()
    tr_resumed.load_state_dict(saved)

    for _ in range(4):
        b_f = torch.randn((32, 2), generator=rng_full)
        b_s = torch.randn((32, 2), generator=rng_split)
        tr_full.step(b_f)
        tr_resumed.step(b_s)

    for pa, pb in zip(tr_full.G.parameters(), tr_resumed.G.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(tr_full.D.parameters(), tr_resumed.D.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(unwrap_prior(tr_full.prior).z, unwrap_prior(tr_resumed.prior).z)


# ---------------------------------------------------------------------------
# D-12: Occam LOC / AST Complexity Reporter & Budget Contract
# ---------------------------------------------------------------------------

def _recipe_source_files(recipe_name: str) -> list[Path]:
    rec = get_recipe(recipe_name)
    if rec.optimizer_family == "unified_v15_equil":
        files: list[Path] = sorted((REPO_ROOT / "particlegan" / "_unified_v15").glob("*.py"))
        for rel in (
            "particlegan/_unified_v11/bank.py",
            "particlegan/_unified_v11/optimizer.py",
            "particlegan/_t1_coupling/critic.py",
        ):
            dep = REPO_ROOT / rel
            if dep.is_file():
                files.append(dep)
        return files
    files = sorted((REPO_ROOT / "particlegan" / "_unified_v11").glob("*.py"))
    if rec.optimizer_family == "unified_v12_equil":
        files.extend(sorted((REPO_ROOT / "particlegan" / "_unified_v12").glob("*.py")))
    elif rec.optimizer_family == "unified_v14_equil":
        files.extend(sorted((REPO_ROOT / "particlegan" / "_unified_v14").glob("*.py")))
    if getattr(rec, "rhbi_self_coupling", None) == "feydy_atomic":
        atomic_py = REPO_ROOT / "particlegan" / "_t1_coupling" / "atomic.py"
        if atomic_py.is_file():
            files.append(atomic_py)
    return files


def measure_recipe_complexity(recipe_name: str) -> dict[str, Any]:
    """Static Occam reporter counting direct LOC, AST branch complexity, and surviving Round-4 epicycles."""
    files = _recipe_source_files(recipe_name)
    total_loc = 0
    total_branches = 0
    branch_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Try, ast.Match)
    has_rh_dz4 = False
    has_w_sat_pt = False

    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        total_loc += len(lines)
        tree = ast.parse(text, filename=str(path))
        total_branches += sum(isinstance(node, branch_types) for node in ast.walk(tree))
        if path.name == "prior.py":
            if "d_z == 4" in text or "z_dim == 4" in text or "radon" in text.lower():
                has_rh_dz4 = True
            if "w_sat_pt" in text:
                has_w_sat_pt = True

    return {
        "recipe": recipe_name,
        "files": len(files),
        "direct_loc": total_loc,
        "ast_branches": total_branches,
        "surviving_round4_epicycles": int(has_rh_dz4) + int(has_w_sat_pt),
    }


@_parametrize_recipes()
def test_d12_occam_complexity_reporter_accounting(recipe_name: str) -> None:
    """Occam LOC/AST reporter must succeed cleanly and produce positive deterministic metrics."""
    report = measure_recipe_complexity(recipe_name)
    assert report["files"] >= 6
    assert report["direct_loc"] > 1000
    assert report["ast_branches"] > 100
    if recipe_name == "unified_v15":
        assert report["surviving_round4_epicycles"] == 0, (
            f"{recipe_name}: {report['surviving_round4_epicycles']} Round-4 epicycles still present"
        )


@_parametrize_recipes(
    xfail_for=ALL_CURRENT | frozenset({"unified_v15"}),
    reason="Open Four-Laws defect D-12 (Occam LOC > 3500 / AST branches > 450; unified_v15 removes Round-4 epicycles=0 but retains full isolated package LOC)",
)
def test_d12_occam_complexity_budget(recipe_name: str) -> None:
    """Occam simplicity target for unified_v15: <= 3500 direct LOC, <= 450 AST branches, 0 Round-4 epicycles."""
    report = measure_recipe_complexity(recipe_name)
    assert report["direct_loc"] <= 3500, f"{recipe_name}: direct_loc={report['direct_loc']} > 3500"
    assert report["ast_branches"] <= 450, f"{recipe_name}: ast_branches={report['ast_branches']} > 450"
    assert report["surviving_round4_epicycles"] == 0, (
        f"{recipe_name}: {report['surviving_round4_epicycles']} Round-4 epicycles still present"
    )
