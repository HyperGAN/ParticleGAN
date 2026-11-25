"""Unit, contract, checkpoint-continuation, and negative-control tests for Unified-v15 Law 1
(``particlegan._unified_v15.{prior,atomic,charts}``).

Covers:
- D-06: index-unbiased ``laguerre_hard_commitment`` (``B_f < K`` zero eviction) and
  ``compute_atomic_quotas`` (``M < K``, ``M == K``, ``M > K`` Zipf).
- D-07: exact ``w_fake_ema`` unit-mass conservation after uncovered-mode teleports,
  exact Law of Total Variance / Covariance on Gersho merge-before-split, exact
  principal-axis LTC splits (``0.84 * v`` isotropic 2D), zero-error directional
  anisotropic LTC coalescence across ``kappa in {2, 3, 5, 10}`` (eliminating the
  ``kappa* = 8.665`` cliff), and zombie-safe dead-prototype revival.
- E10: scale/shift-invariant noise-tolerant ``compute_discrete_atomic_gate`` (0 false
  positives on ``grid100``, ``gaussian1d``, smooth arc, anisotropic ellipses, point mass;
  ~1.0 on exact and ``sigma/Delta <= 0.02`` noisy codebooks), Zipf-fair
  ``update_atomic_ledger``, and scale-invariant inverse matching.
- D-12: Occam pruning verification (no 4D Radon-Hurwitz or ``w_sat_pt`` epicycles).
- Scale equivariance down to ``1e-5x`` (Board #83 ``scale_shock`` contract).
- Zero forbidden primitives (``svd``, ``eigh``, ``pinv``, ``autograd.grad``), clock-free
  horizon invariance, schema-3 checkpointing & rejection of legacy checkpoints, fresh-object
  & fresh-process bit-exact continuation, and semantic negative controls.
"""
from __future__ import annotations

import copy
import math
import pathlib
import subprocess
import sys

import pytest
import torch

from benchmarks.toy_audit import api_four_laws_123 as fl123
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    MoGParticlePrior,
    get_recipe,
    init,
    unwrap_prior,
)
from particlegan._unified_v15.atomic import (
    compute_atomic_quotas,
    compute_discrete_atomic_gate,
    match_inverse_atomic_targets,
    split_joint_table_pair,
    update_atomic_ledger,
)
from particlegan._unified_v15.charts import (
    deconflicted_chart_routing,
    laguerre_hard_commitment,
    update_streaming_charts,
)
from particlegan._unified_v15.prior import (
    UnifiedV15PriorRegularizer,
    _compute_pythagorean_smoothed_cells_v15,
    _is_point_mass_batch,
    _sym_eig2x2_exact,
    _update_isolated_component_ledger,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_v15_trainer(
    *,
    z_dim: int = 4,
    num_particles: int = 64,
    sigma: float = 0.05,
    seed: int = 0,
    total_steps: int = 200,
) -> GANTrainer:
    gen = SimpleMLPGenerator(z_dim=z_dim, hidden_dim=32, n_hidden=2, out_dim=2)
    disc = SimpleMLPDiscriminator(in_dim=2, hidden_dim=32, n_hidden=2, fourier=2)
    prior = MoGParticlePrior(num_particles=num_particles, z_dim=z_dim, sigma=sigma)
    init.deterministic_orthogonal_(gen, seed=seed)
    init.deterministic_orthogonal_(disc, seed=seed + 1)
    init.deterministic_orthogonal_(prior, seed=seed + 2)
    recipe = get_recipe("unified_v15").replace(
        z_dim=z_dim, num_particles=num_particles, prior_kind="mog", total_steps=total_steps
    )
    return GANTrainer(recipe, gen, disc, prior=prior, seed=seed)


# ---------------------------------------------------------------------------
# D-06: Index-unbiased Laguerre commitment and Hamilton quotas
# ---------------------------------------------------------------------------


def test_d06_laguerre_hard_commitment_zero_eviction_all_pairs() -> None:
    """B_f < K particles sitting directly on any pair of charts must never be evicted by index bias."""
    charts = torch.tensor([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    weights = torch.full((4,), 0.25, dtype=torch.float32)
    radii = torch.full((4,), 0.10, dtype=torch.float32)
    ell = torch.tensor(0.20, dtype=torch.float32)

    evictions = 0
    for i in range(4):
        for j in range(i + 1, 4):
            x_fake = torch.stack((charts[i], charts[j]), dim=0)
            _, _, win = laguerre_hard_commitment(x_fake, charts, weights, radii, ell)
            if win.tolist() != [i, j]:
                evictions += 1
    assert evictions == 0


def test_d06_laguerre_hard_commitment_high_index_charts_not_evicted() -> None:
    """At B_f=64 < K=100, fake samples positioned on high-index charts [36..99] must commit to [36..99]."""
    grid = torch.linspace(-4.5, 4.5, 10)
    gx, gy = torch.meshgrid(grid, grid, indexing="ij")
    charts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    weights = torch.full((100,), 0.01, dtype=torch.float32)
    radii = torch.full((100,), 0.10, dtype=torch.float32)
    ell = torch.tensor(0.20, dtype=torch.float32)

    x_fake = charts[36:100].clone()
    _, _, win = laguerre_hard_commitment(x_fake, charts, weights, radii, ell)
    assert win.tolist() == list(range(36, 100))


def test_d06_compute_atomic_quotas_all_regimes_unbiased() -> None:
    """compute_atomic_quotas conserves exact sum M for M<K, M==K, M>K Zipf without index bias."""
    w_unif = torch.full((64,), 1.0 / 64.0)
    q_eq = compute_atomic_quotas(64, w_unif)
    assert q_eq.tolist() == [1] * 64

    # M < K with identical weights must not collapse strictly onto the prefix [0..M-1]
    q_lt = compute_atomic_quotas(16, w_unif)
    assert int(q_lt.sum().item()) == 16
    assert int((q_lt == 1).sum().item()) == 16
    assert q_lt[32:].sum().item() > 0  # Weyl tie-breaker spreads evenly across high indices

    # M > K Zipf (M=96, K=64): every chart gets >= 1 particle and TV vs true Zipf is small
    ranks = torch.arange(1, 65, dtype=torch.float32)
    w_zipf = ranks.pow(-0.5)
    w_zipf = w_zipf / w_zipf.sum()
    q_zipf = compute_atomic_quotas(96, w_zipf)
    assert int(q_zipf.sum().item()) == 96
    assert int(q_zipf.min().item()) >= 1
    tv = float(0.5 * (q_zipf.float() / 96.0 - w_zipf).abs().sum().item())
    # Exact L1-optimal TV with M=96, K=64, q>=1 is 0.0628; Hamilton quota sits within 0.002 of optimum
    assert tv < 0.07


# ---------------------------------------------------------------------------
# D-07: Prior quantizer mass conservation, exact LTV/LTC, and anisotropic coalescence
# ---------------------------------------------------------------------------


def test_d07_w_fake_ema_unit_mass_after_uncovered_teleport() -> None:
    """Uncovered-mode teleport must conserve exact unit mass sum(w_fake_ema) == 1.0."""
    reg = UnifiedV15PriorRegularizer(max_prototypes=16)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn((32, 2), generator=g) * 0.1
    reg._update_quantizer(x0)
    x1 = torch.cat([x0[:-1], torch.tensor([[20.0, 20.0]], dtype=torch.float32)], dim=0)
    reg._update_quantizer(x1)
    assert reg.w_fake_ema is not None and reg.proto_w is not None
    assert math.isclose(float(reg.w_fake_ema.sum().item()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(reg.proto_w.sum().item()), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_d07_gersho_merge_exact_ltv_and_split_ltc() -> None:
    """Gersho merge must conserve exact LTV (including proto_var[a]) and exact 0.84*v isotropic 2D split."""
    reg = UnifiedV15PriorRegularizer(max_prototypes=16)
    reg.quantizer_initialized = True
    reg.proto_c = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    reg.proto_var = torch.tensor([0.05, 0.05, 0.05, 2.0], dtype=torch.float32)
    reg.proto_w = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    reg.proto_hits = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
    reg.w_fake_ema = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    reg.psi = torch.zeros(4, dtype=torch.float32)
    reg.v_push = torch.full((4, 2), 0.05, dtype=torch.float32)

    xb = reg.proto_c.repeat(4, 1)
    reg._update_quantizer(xb)

    # Expected merge of slots 0 & 1 (d^2 = 1.0 -> between-var = 0.25 * 1.0 / 2 = 0.125 plus within ~0.045)
    assert abs(float(reg.proto_var[0].item()) - 0.175) < 0.02
    # Slots 1 and 3 hold the LTC-split children of slot 3 (exact 0.84 * v_pre_split)
    assert math.isclose(float(reg.proto_var[1].item()), float(reg.proto_var[3].item()), rel_tol=1e-5)
    # Exact total covariance conservation across the split pair
    c1, c3 = reg.proto_c[1], reg.proto_c[3]
    mid = 0.5 * (c1 + c3)
    between_cov = 0.5 * (torch.outer(c1 - mid, c1 - mid) + torch.outer(c3 - mid, c3 - mid))
    reconstructed = 0.5 * (reg.proto_cov[1] + reg.proto_cov[3]) + between_cov
    # Pre-split isotropic variance of slot 3 after 1 EMA step with zero batch deviation:
    alpha_10 = min(0.22, 0.045 + 0.16 / math.sqrt(10.0))
    expected_v3_pre = (1.0 - alpha_10) * 2.0
    assert math.isclose(float(reg.proto_var[1].item()) / expected_v3_pre, 0.84, rel_tol=1e-3)
    assert torch.allclose(reconstructed, expected_v3_pre * torch.eye(2), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("kappa", [2.0, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("theta", [0.0, math.pi / 6.0, math.pi / 3.0])
def test_d07_anisotropic_ltc_coalescence_zero_error_and_no_cliff(kappa: float, theta: float) -> None:
    """Directional LTC coalescence must coalesce rotated ellipses through kappa=10 with 0% Frobenius error."""
    s_maj = 0.25
    s_min = s_maj / kappa
    q1 = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float32)
    q2 = torch.tensor([-math.sin(theta), math.cos(theta)], dtype=torch.float32)
    d_off = s_maj * math.sqrt(2.0 / math.pi)
    v_in_maj = (s_maj ** 2) * (1.0 - 2.0 / math.pi)
    cov_true = (s_maj ** 2) * torch.outer(q1, q1) + (s_min ** 2) * torch.outer(q2, q2)
    cov_half = v_in_maj * torch.outer(q1, q1) + (s_min ** 2) * torch.outer(q2, q2)

    pc = torch.stack((-d_off * q1, d_off * q1, torch.tensor([10.0, 0.0])), dim=0)
    pcov = torch.stack((cov_half, cov_half, 0.01 * torch.eye(2)), dim=0)
    pw = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)

    c_bar, cov_bar, _, _, iso_coal, multicell_coal = _compute_pythagorean_smoothed_cells_v15(pc, pcov, pw)
    assert bool(multicell_coal[0]) and bool(multicell_coal[1])
    assert bool(iso_coal.all())
    assert float(c_bar[0].norm().item()) < 1e-6
    frob_err = float((cov_bar[0] - cov_true).norm().item() / cov_true.norm().item())
    assert frob_err < 1e-5


def test_d07_dead_revival_not_deadlocked_by_high_variance_zombies() -> None:
    """High-variance zero-weight zombies must never hijack ge.argmax() and block dead revival."""
    reg = UnifiedV15PriorRegularizer(max_prototypes=16)
    reg.quantizer_initialized = True
    reg.proto_c = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [10.0, 10.0], [12.0, 12.0]],
        dtype=torch.float32,
    )
    # Slots 4 & 5 are fat zombies (huge variance, tiny weight < 0.04 * mean_w)
    reg.proto_var = torch.tensor([0.0002, 0.0002, 0.0002, 0.0002, 0.25, 0.25], dtype=torch.float32)
    reg.proto_cov = reg.proto_var.view(6, 1, 1) * torch.eye(2).unsqueeze(0)
    reg.push_cov = reg.proto_cov.clone()
    reg.proto_w = torch.tensor([0.40, 0.20, 0.20, 0.199, 0.0005, 0.0005], dtype=torch.float32)
    reg.proto_hits = torch.full((6,), 100.0, dtype=torch.float32)
    reg.w_fake_ema = reg.proto_w.clone()
    reg.psi = torch.zeros(6, dtype=torch.float32)
    reg.v_push = reg.proto_var.unsqueeze(1).expand(6, 2).clone()

    # Batch drawn strictly from the 4 active tight modes
    xb = reg.proto_c[:4].repeat(8, 1)
    reg._update_quantizer(xb)
    # Both zombies must have been revived (moved off [10,10]/[12,12] and given positive mass)
    assert float(reg.proto_w.min().item()) > 0.05
    assert float(reg.proto_c[4].norm().item()) < 8.0
    assert float(reg.proto_c[5].norm().item()) < 8.0


# ---------------------------------------------------------------------------
# E10: Discrete atomic detection, ledger, and inverse matching
# ---------------------------------------------------------------------------


def test_e10_discrete_atomic_gate_selectivity_and_scale_invariance() -> None:
    """Gate must trigger (~1) on exact and noisy (sigma/Delta=0.02) codebooks and stay 0 on continuous/grid100."""
    g = torch.Generator().manual_seed(0)

    # Positive 1: fl_law1_noisy_atomic_codebook (20 batches)
    scores = [
        compute_discrete_atomic_gate(fl123.sample_target_for_kind("fl_law1_noisy_atomic_codebook", 128, g, device="cpu"))
        for _ in range(20)
    ]
    assert min(scores) >= 0.90

    # Positive 2: exact scale + proportional shift invariance across 12 decades (1e-6 .. 1e6)
    y_cb = fl123.sample_target_for_kind("fl_law1_noisy_atomic_codebook", 128, g, device="cpu")
    base_gate = compute_discrete_atomic_gate(y_cb)
    for sc in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        shifted = y_cb * sc + (17.25 * sc)
        assert math.isclose(compute_discrete_atomic_gate(shifted), base_gate, abs_tol=1e-4)

    # Positive 3: ByteGAN 8-bit binary hypercube
    y_byte = torch.randint(0, 2, (128, 8), generator=g).float() * 2.0 - 1.0
    assert compute_discrete_atomic_gate(y_byte) >= 0.99

    # Positive 4: general 4D random 64-atom codebook at sigma/Delta in {0, 1e-4, 1e-2}
    c4 = torch.randn((64, 4), generator=g)
    d4 = torch.cdist(c4, c4)
    d4.fill_diagonal_(float("inf"))
    delta4 = float(d4.min(dim=1).values.median().item())
    for sr in (0.0, 1e-4, 1e-2):
        idx = torch.randint(0, 64, (256,), generator=g)
        y4 = c4[idx] + (sr * delta4) * torch.randn((256, 4), generator=g)
        assert compute_discrete_atomic_gate(y4) >= 0.99

    # Negative controls: grid100 (sigma/Delta=0.03), 1D/2D Gaussians, arc, anisotropic ellipses, point mass
    coords = torch.linspace(-1.0, 1.0, 10)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    grid_c = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    delta_grid = float(coords[1] - coords[0])
    for b_sz in (64, 128, 256, 2048):
        idx = torch.randint(0, 100, (b_sz,), generator=g)
        y_grid = grid_c[idx] + (0.03 * delta_grid) * torch.randn((b_sz, 2), generator=g)
        assert compute_discrete_atomic_gate(y_grid) == 0.0

    for y_neg in (
        torch.randn((128, 1), generator=g),
        torch.randn((128, 2), generator=g),
        torch.stack((torch.cos(torch.rand(128, generator=g) * 3.0), torch.sin(torch.rand(128, generator=g) * 3.0)), dim=1),
        fl123.sample_target_for_kind("fl_law1_aniso_unequal_modes", 128, g, device="cpu"),
        # Tight 4-mode continuous Gaussian mixture (sigma/Delta=0.0125, K_eff=4) must NOT false-trigger
        torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])[torch.randint(0, 4, (128,), generator=g)]
        + 0.05 * torch.randn((128, 2), generator=g),
        torch.full((128, 2), 0.9),
    ):
        assert compute_discrete_atomic_gate(y_neg) == 0.0


def test_e10_atomic_ledger_zipf_coverage_and_allow_update_guard() -> None:
    """update_atomic_ledger covers all 64 Zipf charts at M=96, preserves M==K bijection, and obeys allow_update=False."""
    g = torch.Generator().manual_seed(0)
    w_true, atoms, _, _ = fl123.noisy_codebook_parameters("cpu")
    x_init = torch.randn((96, 2), generator=g)
    rows = torch.arange(96, dtype=torch.long)

    tgt, nn_cc, rc, rd, rt, r_atom = update_atomic_ledger(
        x_init, rows, 96, atoms, w_true, None, None, None, allow_update=True
    )
    assert len(set(rt.tolist())) == 64
    occ = torch.bincount(rt, minlength=64).float() / 96.0
    assert float(0.5 * (occ - w_true).abs().sum().item()) < 0.07
    assert r_atom > 0.0 and bool(torch.isfinite(nn_cc).all())

    # allow_update=False must not mutate committed ledger state
    tgt2, _, rc2, rd2, rt2, r_atom2 = update_atomic_ledger(
        x_init + 10.0, rows, 96, atoms, w_true, rc, rd, rt, allow_update=False
    )
    assert torch.equal(rt2, rt)
    assert torch.equal(rc2, rc)
    assert torch.equal(rd2, rd)
    assert torch.equal(tgt2, tgt)
    assert r_atom2 == 0.0


def test_e10_inverse_matching_and_joint_split_scale_invariant() -> None:
    """match_inverse_atomic_targets and split_joint_table_pair work across 1e-5x..1e5x scales."""
    g = torch.Generator().manual_seed(0)
    charts = torch.randn((16, 4), generator=g)
    table = torch.randn((16, 4), generator=g)
    row_target = torch.arange(16, dtype=torch.long)
    rows = torch.arange(16, dtype=torch.long)

    for sc in (1e-5, 1.0, 1e5):
        inv_z, has_m, ell_z, margin = match_inverse_atomic_targets(
            charts * sc, table * sc, charts * sc, row_target, table * sc
        )
        assert bool(has_m.all())
        assert torch.allclose(inv_z, table * sc, atol=1e-6 * sc, rtol=1e-5)
        assert float(ell_z.item()) > 0.0 and bool((margin > 0.0).all())

        x_fake = torch.cat((charts * sc, table[rows] * sc), dim=1)
        y_real = torch.cat((charts * sc, table[rows] * sc), dim=1)
        split = split_joint_table_pair(x_fake, y_real, table * sc, rows)
        assert split is not None
        assert torch.allclose(split[1], table[rows] * sc)


# ---------------------------------------------------------------------------
# D-12: Occam Prior Pruning
# ---------------------------------------------------------------------------


def test_d12_prior_occam_pruning_zero_round4_epicycles() -> None:
    """_unified_v15/prior.py must have zero surviving Round-4 prior epicycles (RH 4D & w_sat_pt)."""
    src = (REPO_ROOT / "particlegan" / "_unified_v15" / "prior.py").read_text(encoding="utf-8")
    low = src.lower()
    for forbidden in ("d_z == 4", "z_dim == 4", "radon", "w_sat_pt"):
        assert forbidden not in low, f"forbidden Round-4 epicycle token {forbidden!r} found in _unified_v15/prior.py"


# ---------------------------------------------------------------------------
# Board #83: 1e-5x Scale Equivariance
# ---------------------------------------------------------------------------


def test_scale_1e5_point_mass_predicate_and_pullback_stability() -> None:
    """1e-5x continuous clouds must never falsely trigger point-mass mode and must train finitely."""
    g = torch.Generator().manual_seed(0)
    assert _is_point_mass_batch(torch.zeros((32, 2)))
    assert _is_point_mass_batch(torch.full((32, 2), 0.9))

    centers = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]], dtype=torch.float32) * 1e-5
    idx = torch.randint(0, 4, (64,), generator=g)
    y_tiny = centers[idx] + (8e-7) * torch.randn((64, 2), generator=g)
    assert not _is_point_mass_batch(y_tiny)
    # Even a single mode at 1e-5x scale must not false-trigger point-mass mode
    y_single = centers[0:1].expand(64, 2) + (8e-7) * torch.randn((64, 2), generator=g)
    assert not _is_point_mass_batch(y_single)

    # Full unified_v15 trainer steps on 1e-5x 4-mode data stay strictly finite
    trainer = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.08, seed=0)
    for _ in range(12):
        idx_b = torch.randint(0, 4, (32,), generator=g)
        yb = centers[idx_b] + (8e-7) * torch.randn((32, 2), generator=g)
        trainer.step(yb)
    samps = trainer.sample(128)
    assert bool(torch.isfinite(samps).all())
    assert float(samps.abs().amax().item()) < 1.0

    # charts.py streaming update & routing scale equivariance at 1e-5x
    c1, w1, r1, bc1, s1, n1, v1, fm1, ph1, rev1 = update_streaming_charts(
        y_tiny, None, None, None, None, 0.0, 0, max_charts=16
    )
    assert bool(torch.isfinite(c1).all()) and bool(torch.isfinite(r1).all())
    assert float(r1.amax().item()) < 1e-4
    tc, th, rw, sh, nfm = deconflicted_chart_routing(
        y_tiny, c1, w1, r1, bc1, fm1, r1.median(), w_valley=1.0
    )
    assert bool(torch.isfinite(tc).all()) and bool(torch.isfinite(rw).all())


# ---------------------------------------------------------------------------
# Forbidden primitives, clock-free horizon, checkpointing & negative controls
# ---------------------------------------------------------------------------


def test_zero_forbidden_linalg_and_autograd_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Law 1 prior/atomic/charts paths must never invoke svd, svdvals, eigh, eigvalsh, pinv, or autograd.grad."""
    def _boom(*args, **kwargs):
        raise AssertionError("forbidden primitive invoked")

    for mod, attr in (
        (torch.linalg, "svd"),
        (torch.linalg, "svdvals"),
        (torch.linalg, "eigh"),
        (torch.linalg, "eigvalsh"),
        (torch.linalg, "pinv"),
        (torch.autograd, "grad"),
    ):
        monkeypatch.setattr(mod, attr, _boom)

    g = torch.Generator().manual_seed(0)
    # 1. Pullback with dim=2, d_z=4 (anisotropic isolated ledger + QMC push_cov)
    tr4 = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=0)
    centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    for _ in range(6):
        yb = centers[torch.randint(0, 4, (16,), generator=g)] + torch.randn((16, 2), generator=g) * torch.tensor([0.25, 0.025])
        tr4.step(yb)

    # 2. Pullback with dim=2, d_z=2 (2-tangent Jacobian pushforward)
    tr2 = _make_v15_trainer(z_dim=2, num_particles=32, sigma=0.05, seed=1)
    for _ in range(4):
        yb = centers[torch.randint(0, 4, (16,), generator=g)] + 0.08 * torch.randn((16, 2), generator=g)
        tr2.step(yb)

    # 3. Direct atomic & charts primitives
    y_cb = fl123.sample_target_for_kind("fl_law1_noisy_atomic_codebook", 128, g, device="cpu")
    assert compute_discrete_atomic_gate(y_cb) >= 0.90
    c, w, r, bc, *_ = update_streaming_charts(y_cb, None, None, None, None, 0.0, 0, max_charts=64)
    laguerre_hard_commitment(y_cb[:32], c, w, r, r.median())


def test_clock_free_horizon_invariance() -> None:
    """Unified-v15 Law 1 prior trajectory must be bit-identical under total_steps=40 vs 4000."""
    g_data = torch.Generator().manual_seed(7)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, -2.0]], dtype=torch.float32)
    batches = [
        centers[torch.randint(0, 4, (16,), generator=g_data)] + torch.randn((16, 2), generator=g_data) * torch.tensor([0.20, 0.03])
        for _ in range(12)
    ]

    states = []
    for total_steps in (40, 4000):
        tr = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=0, total_steps=total_steps)
        for yb in batches:
            tr.step(yb)
        states.append((
            tr.prior.z.detach().clone(),
            tr.prior_regularizer.state_dict(),
        ))

    assert torch.equal(states[0][0], states[1][0])
    for key in ("proto_c", "proto_cov", "push_cov", "proto_w", "w_fake_ema", "row_pos", "row_tgt_pos"):
        t0, t1 = states[0][1][key], states[1][1][key]
        if t0 is None:
            assert t1 is None
        else:
            assert torch.equal(t0, t1)


def test_checkpoint_continuation_and_legacy_schema_rejection(tmp_path: pathlib.Path) -> None:
    """Schema-3 checkpoint must continue bit-identically in-process and in a fresh subprocess, and reject old schemas."""
    g_data = torch.Generator().manual_seed(11)
    centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    batches = [
        centers[torch.randint(0, 4, (16,), generator=g_data)] + torch.randn((16, 2), generator=g_data) * torch.tensor([0.20, 0.025])
        for _ in range(18)
    ]

    # Reference uninterrupted run
    tr_ref = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=0)
    for yb in batches:
        tr_ref.step(yb)

    # Split run at step 10 (after quantizer + push_cov + ledger + QMC phase activation)
    tr_a = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=0)
    for yb in batches[:10]:
        tr_a.step(yb)
    ckpt = copy.deepcopy(tr_a.state_dict())
    reg_sd = tr_a.prior_regularizer.state_dict()
    assert reg_sd["schema"] == 3
    assert reg_sd["quantizer_initialized"] is True
    assert reg_sd["proto_cov"] is not None and reg_sd["push_cov"] is not None

    # 1. In-process fresh trainer continuation
    tr_b = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=99)
    tr_b.load_state_dict(ckpt)
    for yb in batches[10:]:
        tr_b.step(yb)
    assert torch.equal(tr_b.prior.z, tr_ref.prior.z)
    for p_b, p_r in zip(tr_b.G.parameters(), tr_ref.G.parameters()):
        assert torch.equal(p_b, p_r)
    assert math.isclose(tr_b.prior_regularizer.phase, tr_ref.prior_regularizer.phase, rel_tol=0.0, abs_tol=1e-12)

    # 2. Fresh-process subprocess continuation + import hygiene check
    ckpt_path = tmp_path / "v15_law1_ckpt.pt"
    torch.save({"ckpt": ckpt, "tail_batches": batches[10:]}, ckpt_path)
    out_path = tmp_path / "v15_law1_out.pt"
    code = (
        "import torch; "
        "from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator; "
        "from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, get_recipe, init; "
        "assert DEFAULT_RECIPE == 'unified_v12'; "
        f"payload = torch.load({str(ckpt_path)!r}, weights_only=False); "
        "gen = SimpleMLPGenerator(z_dim=4, hidden_dim=32, n_hidden=2, out_dim=2); "
        "disc = SimpleMLPDiscriminator(in_dim=2, hidden_dim=32, n_hidden=2, fourier=2); "
        "prior = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05); "
        "init.deterministic_orthogonal_(gen, seed=123); init.deterministic_orthogonal_(disc, seed=124); init.deterministic_orthogonal_(prior, seed=125); "
        "rec = get_recipe('unified_v15').replace(z_dim=4, num_particles=32, prior_kind='mog', total_steps=200); "
        "tr = GANTrainer(rec, gen, disc, prior=prior, seed=99); "
        "tr.load_state_dict(payload['ckpt']); "
        "[tr.step(yb) for yb in payload['tail_batches']]; "
        f"torch.save({{'z': tr.prior.z.detach(), 'g0': next(tr.G.parameters()).detach()}}, {str(out_path)!r})"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    sub_out = torch.load(out_path, weights_only=False)
    assert torch.equal(sub_out["z"], tr_ref.prior.z.detach())
    assert torch.equal(sub_out["g0"], next(tr_ref.G.parameters()).detach())

    # 3. Strict rejection of legacy schema-1/2 and corrupted schema-3 checkpoints
    reg_test = UnifiedV15PriorRegularizer(max_prototypes=16)
    for bad_schema in (1, 2, 4, "3"):
        bad = copy.deepcopy(reg_sd)
        bad["schema"] = bad_schema
        with pytest.raises(ValueError):
            reg_test.load_state_dict(bad)

    bad_missing = copy.deepcopy(reg_sd)
    bad_missing.pop("proto_cov")
    with pytest.raises(ValueError):
        reg_test.load_state_dict(bad_missing)

    bad_neg_cov = copy.deepcopy(reg_sd)
    bad_neg_cov["push_cov"] = bad_neg_cov["push_cov"].clone()
    bad_neg_cov["push_cov"][0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        reg_test.load_state_dict(bad_neg_cov)

    bad_mass = copy.deepcopy(reg_sd)
    bad_mass["w_fake_ema"] = bad_mass["w_fake_ema"] * 1.5
    with pytest.raises(ValueError):
        reg_test.load_state_dict(bad_mass)


def test_semantic_negative_control_damage_proto_cov_changes_step() -> None:
    """Damaging proto_cov / push_cov must measurably change the next generator step (load-bearing proof)."""
    g_data = torch.Generator().manual_seed(19)
    centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    batches = [
        centers[torch.randint(0, 4, (16,), generator=g_data)] + torch.randn((16, 2), generator=g_data) * torch.tensor([0.25, 0.025])
        for _ in range(12)
    ]

    tr_clean = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=0)
    for yb in batches[:11]:
        tr_clean.step(yb)
    ckpt = copy.deepcopy(tr_clean.state_dict())

    tr_dmg = _make_v15_trainer(z_dim=4, num_particles=32, sigma=0.05, seed=0)
    tr_dmg.load_state_dict(ckpt)
    # Deliberately corrupt push_cov by 25x along one axis
    assert tr_dmg.prior_regularizer.push_cov is not None
    tr_dmg.prior_regularizer.push_cov = tr_dmg.prior_regularizer.push_cov * 25.0

    tr_clean.step(batches[11])
    tr_dmg.step(batches[11])

    diff = sum(float((p1 - p2).abs().sum().item()) for p1, p2 in zip(tr_clean.G.parameters(), tr_dmg.G.parameters()))
    assert diff > 1e-4, "corrupting push_cov had no effect on the generator step"
    assert DEFAULT_RECIPE == "unified_v12"
    assert unwrap_prior(tr_clean.prior) is tr_clean.prior
    l1, l2, _, _ = _sym_eig2x2_exact(tr_clean.prior_regularizer.proto_cov)
    assert bool((l1 >= l2).all()) and bool((l2 > 0.0).all())


def test_d07_settled_ledger_hysteresis_resists_small_ema_jitter() -> None:
    """Settled isolated-component ledger must resist 2.8-SE EMA weight jitter yet immediately rescue 0-occ modes."""
    reg = UnifiedV15PriorRegularizer(max_prototypes=16)
    centers = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]], dtype=torch.float32)
    covs = 0.04 * torch.eye(2).unsqueeze(0).expand(4, 2, 2).clone()
    sigs = torch.full((4,), 0.2, dtype=torch.float32)
    w_mat = torch.eye(4, dtype=torch.float32)
    p_total = 256
    rows = torch.arange(p_total, dtype=torch.long)

    # Start with exact equal occupancy (64 per mode) on settled positions
    reg.proto_w = torch.full((4,), 0.25, dtype=torch.float32)
    x_settled = centers.repeat_interleave(64, dim=0)
    assigned0, *_ = _update_isolated_component_ledger(
        reg, x_settled, rows, p_total, centers, covs, sigs, w_mat, need_whiten=False, real_batch_size=16
    )
    assert torch.bincount(assigned0, minlength=4).tolist() == [64, 64, 64, 64]

    # Inject a realistic 2.8-SE EMA weight jitter at B=16 (SE = 256*sqrt((0.02/1.98)/16*0.25*0.75) ~= 2.78 -> ~7.8)
    reg.proto_w = torch.tensor([0.28, 0.22, 0.25, 0.25], dtype=torch.float32)
    assigned1, *_ = _update_isolated_component_ledger(
        reg, x_settled, rows, p_total, centers, covs, sigs, w_mat, need_whiten=False, real_batch_size=16
    )
    assert torch.equal(assigned1, assigned0), "settled particles were evicted by sub-3.6-SE EMA jitter"

    # Conversely, a newly added 5th mode with 0 occupancy must be rescued immediately on the very next call
    centers5 = torch.cat((centers, torch.tensor([[0.0, 0.0]])), dim=0)
    covs5 = 0.04 * torch.eye(2).unsqueeze(0).expand(5, 2, 2).clone()
    sigs5 = torch.full((5,), 0.2, dtype=torch.float32)
    w_mat5 = torch.eye(5, dtype=torch.float32)
    reg.proto_w = torch.tensor([0.24, 0.24, 0.24, 0.24, 0.04], dtype=torch.float32)
    assigned2, *_ = _update_isolated_component_ledger(
        reg, x_settled, rows, p_total, centers5, covs5, sigs5, w_mat5, need_whiten=False, real_batch_size=16
    )
    assert int((assigned2 == 4).sum().item()) >= 1, "zero-occupancy mode was not rescued immediately"
