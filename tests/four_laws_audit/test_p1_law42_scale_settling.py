"""Reproducible mathematical & empirical audit tests for Law 4.2 (P1).

Verifies every formal theorem, counterexample, and implementation-level finding in
`reports/four-laws-audit/p1-law42-scale-settling.md`:
  1. Theorem 1: Memoryless positive 0-homogeneous operators are discontinuous at 0 and
     enter a persistent O(eta) limit cycle on smooth quadratics.
  2. Theorem 2: Fixed-epsilon crossover prevents simultaneous 1e-12 rail escape and
     above-1e-12 settling; and ParticleGAN's capped `min(s_base, c * peak_rms)` breaks
     global trajectory scale invariance above `s_base / c`.
  3. Theorem 3: Stateful trajectory degree-0 invariance IS compatible with continuous
     settling — while exposing every sharp failure mode of monotonic `peak_rms`,
     decaying `peak_rms`, sliding-window `peak_rms`, and momentum coherence `||m||/||v||`.
  4. Theorem 4: Fundamental information-theoretic obstructions (step-1 near-equilibrium
     kick and time-varying scale-drop indistinguishability).
  5. Direct audits of actual ParticleGAN optimizers (`UnifiedV12Optimizer`,
     `UnifiedV11EquilibratedOptimizer`, `T4EquilibratedOptimizer`,
     `MinimalGeneratorOptimizer`, `NashV3GeneratorAdam`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts" / "four_laws_audit") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "four_laws_audit"))

from p1_scale_settling import (  # noqa: E402
    DynamicCoherenceSGD,
    DynamicPeakRMSSGD,
    FixedEpsTikhonovSGD,
    MemorylessNormalizedSGD,
    make_optimizer,
    run_regime1_rescaling,
    run_regime2_deterministic,
    run_regime3_noisy_stationary,
    run_regime4_adversarial_shift,
    run_regime5_adversarial_rotation,
)


# ---------------------------------------------------------------------------
# Theorem 1: Memoryless Degree-0 Impossibility & Persistent Limit-Cycle Chatter
# ---------------------------------------------------------------------------

def test_theorem1_memoryless_discontinuity_and_limit_cycle() -> None:
    """Any nontrivial memoryless degree-0 operator has liminf_{c->0+} ||U(cg)|| = ||U(g)|| > 0
    and chatters forever at amplitude >= eta/2 on L(x) = 0.5 x^2.
    """
    torch.set_num_threads(1)
    g0 = torch.tensor([0.6, -0.8], dtype=torch.float64)
    # Check positive 0-homogeneity down to c = 1e-300 in FP64 (scale-safe normalization avoids squaring underflow)
    norms = []
    for exp in (0, -6, -12, -50, -150, -300):
        c = 10.0 ** exp
        g = g0 * c
        g_u = g / g.abs().amax()
        u = g_u / g_u.norm()
        norms.append(float(u.norm().item()))
    assert all(abs(n - 1.0) < 1e-15 for n in norms)

    # 1D quadratic limit cycle: x_0 = 0.013 (not a multiple of eta = 0.02)
    eta = 0.02
    p = torch.tensor([0.013], dtype=torch.float32)
    opt = MemorylessNormalizedSGD([p], lr=eta)
    tail_abs = []
    for t in range(500):
        p.grad = p.detach().clone()
        opt.step()
        if t >= 450:
            tail_abs.append(float(p.detach().abs().item()))
    # Must oscillate forever between 0.013 and -0.007, max >= eta/2 = 0.01
    assert max(tail_abs) >= eta / 2.0 - 1e-6
    assert opt.last_step_norm == pytest.approx(eta, rel=1e-5)


# ---------------------------------------------------------------------------
# Theorem 2: Fixed-Epsilon Crossover & ParticleGAN s_base Cap Scale Breakdown
# ---------------------------------------------------------------------------

def test_theorem2_fixed_eps_crossover_impossibility() -> None:
    """No fixed eps > 0 can simultaneously keep >= 50% step at g=1e-12 and <= 50% step at g=1e-6."""
    g_rail = 1e-12
    g_eq = 1e-6
    for eps in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.0061):
        phi_rail = g_rail / math.hypot(g_rail, eps)
        phi_eq = g_eq / math.hypot(g_eq, eps)
        # Strict monotonicity of phi(g/eps) makes phi_rail >= 0.5 and phi_eq <= 0.5 impossible
        assert phi_rail < phi_eq
        assert not (phi_rail >= 0.5 and phi_eq <= 0.5)

    # Fixed-eps Adam (eps=1e-8) stalls completely on 1e-12 rail
    r2 = run_regime2_deterministic(arms=("fixed_eps_adam", "fixed_eps_tikh"))
    assert r2["tanh_rail_escape"]["fixed_eps_adam"]["1e+00"]["escaped_step"] == -1.0
    assert r2["tanh_rail_escape"]["fixed_eps_tikh"]["1e+00"]["escaped_step"] == -1.0


def test_regime1_exact_rescaling_and_actual_particlegan_breakpoints() -> None:
    """Verify exact trajectory degree-0 invariance on pure stateful mechanisms and expose
    exact scale-invariance breakpoints in actual ParticleGAN optimizers:
      - UnifiedV12Optimizer & UnifiedV11EquilibratedOptimizer: invariant for c in {1e-12, 1e-6}
        (below s_base/alpha) but BREAK between c=1e-6 and c=1/1e6/1e12 due to s_base=0.0061 cap!
      - T4EquilibratedOptimizer: invariant across all 24 orders of magnitude [1e-12, 1e12]!
      - MinimalGeneratorOptimizer & NashV3GeneratorAdam: BREAK at c=1e-12 due to additive eps=1e-8!
      - UnifiedV11EquilibratedOptimizer: cur_sv_rms underflows to 0 at c=1e-25 (gf.norm() in FP32)
        whereas UnifiedV12Optimizer preserves peak_rms down to c=1e-37 and cuts off at 1e-38!
      - UnifiedV11EquilibratedOptimizer 1D params hit clamp_min(eps=1e-8) at c=1e-12!
    """
    r1 = run_regime1_rescaling(
        arms=(
            "memless_norm_sgd",
            "fixed_eps_adam",
            "pure_peak_mono",
            "pure_peak_decay",
            "pure_peak_window",
            "pure_coherence",
            "pg_unified_v12",
            "pg_unified_v11",
            "pg_t4_equil",
            "pg_minimal",
            "pg_nash_v3",
        )
    )
    mat = r1["matrix_24decades"]

    # Pure trajectory-degree-0 operators and T4EquilibratedOptimizer hold invariance across 24 decades
    for arm in ("memless_norm_sgd", "pure_peak_mono", "pure_peak_decay", "pure_peak_window", "pure_coherence", "pg_t4_equil"):
        assert mat[arm]["max_diff_all"] < 1e-5, f"{arm} unexpectedly broke rescaling invariance: {mat[arm]}"

    # UnifiedV12 and UnifiedV11 ARE invariant below s_base/alpha (1e-12 vs 1e-6),
    # and BREAK invariance between 1e-6 and 1.0 / 1e6 due to s_base=0.0061 cap!
    for arm in ("pg_unified_v12", "pg_unified_v11"):
        assert mat[arm]["diff_1e-12_vs_1e-6"] < 1e-5
        assert mat[arm]["diff_vs_1"]["1e-06"] > 1e-2, f"{arm} should diverge between c=1e-6 and c=1 due to s_base cap"

    # Fixed-eps Adam, Minimal, and NashV3 all fail at c=1e-12 due to additive eps=1e-8
    for arm in ("fixed_eps_adam", "pg_minimal", "pg_nash_v3"):
        assert mat[arm]["moves"]["1e-12"] < 1e-3 * mat[arm]["moves"]["1e+00"], (
            f"{arm} expected crushed motion at c=1e-12 due to eps=1e-8"
        )

    # Underflow probe: unified_v11 cur_sv_rms underflows to 0.0 at c=1e-25 while unified_v12 tracks peak_rms to 1e-37
    uf = r1["underflow_probe"]
    assert uf["pg_unified_v11"]["peak_after_step1"]["1e-25"] == 0.0
    assert uf["pg_unified_v12"]["peak_after_step1"]["1e-25"] > 0.0
    assert uf["pg_unified_v12"]["peak_after_step1"]["1e-37"] > 0.0
    # At c=1e-40 (< 1e-38 clamp), both unified_v12 and pure_peak_mono freeze to exact 0
    assert uf["pg_unified_v12"]["step1_move"]["1e-40"] == 0.0
    assert uf["pure_peak_mono"]["step1_move"]["1e-40"] == 0.0

    # 1D vector clamp_min(eps=1e-8) bug in unified_v11 vs clean scaling in unified_v12
    v1d = r1["vec1d_clamp_probe"]
    assert v1d["pg_unified_v11"]["ratio_1e-12_over_1e-6"] < 1e-3
    assert abs(v1d["pg_unified_v12"]["ratio_1e-12_over_1e-6"] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Regime 2: Deterministic Convergence, 1e-12 Rail Escape, Near-Eq Kick & 1/t Decay
# ---------------------------------------------------------------------------

def test_regime2_rail_escape_settling_and_obstructions() -> None:
    """Verify:
      (a) Pure monotonic peak_rms escapes 1e-12 tanh rail AND settles identically across c in {1e-6, 1, 1e6}.
      (b) Actual UnifiedV12 escapes 1e-12 rail and settles at c=1e-6 and c=1, but LOSES settling at c=1e6
          (s_base=0.0061 cap -> persistent chatter at c=1e6!).
      (c) Actual MinimalGeneratorOptimizer & NashV3GeneratorAdam STALL on 1e-12 rail due to eps=1e-8!
      (d) Obstruction 4.1: EVERY nontrivial trajectory-degree-0 operator kicks W_0=1e-6 outward by >10^3x on step 1.
      (e) Theorem 3b/3d: Decaying peak (gamma=0.9995), sliding window (W=50), and coherence cooling
          ALL fail to settle on algebraic g_t = t^{-1}, while monotonic peak settles as O(1/t).
    """
    r2 = run_regime2_deterministic(
        arms=(
            "memless_norm_sgd",
            "fixed_eps_adam",
            "pure_peak_mono",
            "pure_peak_decay",
            "pure_peak_window",
            "pure_coherence",
            "pg_unified_v12",
            "pg_unified_v11",
            "pg_t4_equil",
            "pg_minimal",
            "pg_nash_v3",
        )
    )
    # Verify initial rail gradient really is ~2.76e-12
    assert 2.0e-12 < r2["init_rail_grad"] < 3.5e-12

    # (a) Memoryless escapes rail fast but chatters forever; pure_peak_mono escapes AND settles identically across scales
    rail = r2["tanh_rail_escape"]
    assert 0 < rail["memless_norm_sgd"]["1e+00"]["escaped_step"] < 500
    assert rail["memless_norm_sgd"]["1e+00"]["tail_mean_step"] > 0.07  # permanent chatter at lr=0.08!

    for scale_k in ("1e-06", "1e+00", "1e+06"):
        assert 0 < rail["pure_peak_mono"][scale_k]["escaped_step"] < 500
        assert rail["pure_peak_mono"][scale_k]["final_rms"] < 1e-5
        assert rail["pure_peak_mono"][scale_k]["tail_mean_step"] < 1e-5

    # (b) Actual UnifiedV12 escapes at all scales, settles at 1e-6 and 1, but CHATTERS at c=1e6 due to s_base cap!
    assert 0 < rail["pg_unified_v12"]["1e-06"]["escaped_step"] < 400
    assert rail["pg_unified_v12"]["1e-06"]["final_rms"] < 1e-4
    assert rail["pg_unified_v12"]["1e+06"]["tail_mean_step"] > 100.0 * rail["pg_unified_v12"]["1e-06"]["tail_mean_step"]

    # (c) Actual Minimal and NashV3 fail to escape 1e-12 rail because of additive eps=1e-8!
    assert rail["pg_minimal"]["1e+00"]["escaped_step"] == -1.0
    assert rail["pg_nash_v3"]["1e+00"]["escaped_step"] == -1.0

    # (d) Obstruction 4.1: near-equilibrium start (1e-6) suffers > 1000x step-1 kick on EVERY trajectory degree-0 arm
    nk = r2["near_equilibrium_kick"]
    for arm in ("memless_norm_sgd", "pure_peak_mono", "pure_peak_decay", "pure_coherence", "pg_unified_v12", "pg_t4_equil"):
        assert nk[arm]["step1_amplification"] > 1000.0, f"Obstruction 4.1 violated for {arm}: {nk[arm]}"

    # (e) Theorem 3b/3d: algebraic 1/t decay — monotonic peak settles (< 1e-3 ratio),
    # while decaying peak (gamma=0.9995) turns around past t=2000 (step_5000 > 1.5*step_2000) and
    # reverts to ratio=1.0 by t=25000; sliding window & coherence stay > 0.95!
    slow = r2["algebraic_decay_1_over_t"]
    assert slow["pure_peak_mono"]["ratio_5000_over_1"] < 1e-3
    assert slow["pg_unified_v12"]["ratio_5000_over_1"] < 1e-3  # p_dec=1.0 on 1-obj
    assert slow["pure_peak_decay"]["ratio_5000_over_2000"] > 1.5  # non-monotonic anti-settling!
    assert slow["pg_unified_v11"]["ratio_5000_over_2000"] > 1.5   # p_dec=0.9995 when hot!
    assert slow["pure_peak_decay_25k_1d"]["ratio_25000_over_1"] == pytest.approx(1.0, rel=1e-3)
    assert slow["pure_peak_window"]["ratio_5000_over_1"] > 0.95
    assert slow["pure_coherence"]["ratio_5000_over_1"] > 0.95


# ---------------------------------------------------------------------------
# Regime 3: Noisy Stationary Points — Coherence Positive Floor & Peak Stall
# ---------------------------------------------------------------------------

def test_regime3_noisy_stationary_floors() -> None:
    """Verify:
      (a) Coherence cooling on i.i.d. zero-mean noise converges to strictly positive floor
          sqrt((1 - b1) / (1 + b1)) (falsifying the claim `coherence -> 0`).
      (b) On constant-norm sphere noise, monotonic peak_rms gets ZERO settling (ratio == 1.0).
      (c) When noise variance drops 100x, peak_rms settles while coherence stays flat at its floor.
    """
    r3 = run_regime3_noisy_stationary(
        arms=("memless_norm_sgd", "pure_peak_mono", "pure_coherence", "pg_unified_v12", "pg_unified_v12_stiff", "pg_t4_equil")
    )

    # Exact formula check for coherence floor sqrt((1-b1)/(1+b1))
    for k, v in r3["coherence_iid_floor_verification"].items():
        assert v["abs_error"] < 0.02, f"Coherence floor mismatch for {k}: {v}"

    # On constant-norm sphere noise: monotonic peak_rms has ZERO contraction (tail/early == 1.0)
    sph = r3["sphere_const_norm_scale1e-3"]
    assert sph["pure_peak_mono"]["tail_over_early"] > 0.99
    assert sph["pg_unified_v12"]["tail_over_early"] > 0.99
    # Coherence has strictly positive floor (does NOT go to 0)
    assert sph["pure_coherence"]["final_coherence"] > 0.25

    # On 100x variance drop: peak_rms contracts strongly (< 0.05), coherence does NOT contract (> 0.8)
    vd = r3["variance_drop_100x"]
    assert vd["pure_peak_mono"]["tail_over_early"] < 0.05
    assert vd["pg_unified_v12"]["tail_over_early"] < 0.05
    assert vd["pure_coherence"]["tail_over_early"] > 0.80


# ---------------------------------------------------------------------------
# Regime 4: Adversarial Spike Ratcheting, Scale-Drop Shift & Non-Convex Saddle
# ---------------------------------------------------------------------------

def test_regime4_spike_freezing_shift_and_saddle() -> None:
    """CRITICAL ADVERSARIAL TESTS:
      (a) Single 1e4x transient spike PERMANENTLY freezes pure_peak_mono AND actual
          UnifiedV12Optimizer (at c=1e-6 where s_eff = 0.5*peak_rms < s_base and p_dec=1.0!),
          while UnifiedV12 at c=1.0 recovers ONLY because s_base=0.0061 broke scale invariance,
          and sliding-window W=50 recovers cleanly at ALL scales!
      (b) Post-settling shift + 1000x scale drop traps monotonic peak_rms (Obstruction 4.2).
      (c) Non-convex saddle L(x,y)=0.5x^2-0.5y^2+0.25y^4 from (1, 1e-8): tensor-wide monotonic
          peak_rms stalls at the saddle while memoryless/coordinatewise escape.
    """
    r4 = run_regime4_adversarial_shift(
        arms=(
            "memless_norm_sgd",
            "memless_sign_sgd",
            "pure_peak_mono",
            "pure_peak_decay",
            "pure_peak_window",
            "pure_coherence",
            "pg_unified_v12",
            "pg_unified_v11",
            "pg_t4_equil",
        )
    )

    # (a) Single spike at c=1e-6: permanent freeze on pure_peak_mono and actual UnifiedV12 (p_dec=1.0)!
    spk_small = r4["single_spike_1e4x"]["c=1e-6"]
    spk_one = r4["single_spike_1e4x"]["c=1e+0"]
    assert spk_small["pure_peak_mono"]["post_over_pre"] < 1e-3
    assert spk_small["pure_peak_mono"]["converged_95pct"] == 0.0
    assert spk_small["pg_unified_v12"]["post_over_pre"] < 1e-3
    assert spk_small["pg_unified_v12"]["converged_95pct"] == 0.0

    # Catch-22 proof: at c=1.0, UnifiedV12 ONLY escapes the spike because s_base=0.0061 capped s_eff!
    assert spk_one["pg_unified_v12"]["converged_95pct"] == 1.0

    # Sliding-window peak recovers completely at BOTH scales
    assert spk_small["pure_peak_window"]["converged_95pct"] == 1.0
    assert spk_one["pure_peak_window"]["converged_95pct"] == 1.0

    # (b) Post-settling distribution shift with 1000x scale drop traps monotonic peak_rms
    sh = r4["shift_with_1000x_scale_drop"]
    assert sh["pure_peak_mono"]["recovered_90pct"] == 0.0
    assert sh["pg_unified_v12"]["recovered_90pct"] == 0.0
    assert sh["pure_peak_window"]["recovered_90pct"] == 1.0

    # (c) Non-convex saddle escape: tensor-wide monotonic peak stalls, sign/memless_norm escape
    sd = r4["nonconvex_saddle_escape"]
    assert sd["memless_sign_sgd"]["escaped"] == 1.0
    assert sd["memless_norm_sgd"]["escaped"] == 1.0
    assert sd["pure_peak_mono"]["escaped"] == 0.0


# ---------------------------------------------------------------------------
# Regime 5: Persistent Adversarial Rotation (Bilinear Minimax Game)
# ---------------------------------------------------------------------------

def test_regime5_bilinear_rotation_and_coherence_formula() -> None:
    """Verify:
      (a) Exact harmonic rotation coherence matches (1-b1)/sqrt(1 - 2*b1*cos(w) + b1^2) to <1e-3.
      (b) On simultaneous bilinear minimax game, radius is strictly increasing so peak_rms
          settling is identically inactive (r_end > r_0) and positive momentum coherence
          amplifies divergence faster than zero-momentum!
      (c) On alternating bilinear game, neither peak_rms nor coherence drives radius to 0.
    """
    r5 = run_regime5_adversarial_rotation(
        arms=(
            "memless_norm_sgd",
            "pure_peak_mono",
            "pure_coherence",
            "pg_unified_v12",
            "pg_unified_v12_stiff",
            "pg_t4_equil",
        )
    )

    # (a) Exact harmonic coherence formula across frequencies (including period-2 flip omega=pi -> 1/9)
    for k, v in r5["harmonic_coherence_formula"].items():
        assert v["abs_error"] < 1e-3, f"Harmonic coherence formula mismatch for {k}: {v}"
    assert r5["harmonic_coherence_formula"]["omega=pi (period-2 flip)"]["theoretical_coh"] == pytest.approx(1.0 / 9.0, rel=1e-6)

    # (b) Simultaneous bilinear minimax game: outward spiral r_T/r_0 = sqrt(1 + T*eta_eff^2) > 1
    # so peak_rms is identically inactive (||g_t||/P_t == 1) and coherence momentum diverges even faster!
    sim = r5["bilinear_simultaneous"]
    assert sim["memless_norm_sgd"]["r_ratio"] == pytest.approx(math.sqrt(1.0 + 600 * 0.02 * 0.02), rel=0.02)
    assert sim["pure_peak_mono"]["r_ratio"] > 1.07
    assert sim["pg_unified_v12"]["r_ratio"] > 1.2
    # Positive momentum coherence lags in phase and spirals outward even faster than zero-momentum normalized SGD
    assert sim["pure_coherence"]["r_ratio"] > sim["memless_norm_sgd"]["r_ratio"]
    assert sim["pure_coherence"]["final_coherence"] > 0.90

    # (c) Alternating bilinear minimax game: none of peak_rms / coherence converges to (0, 0)
    alt = r5["bilinear_alternating"]
    for arm in ("memless_norm_sgd", "pure_peak_mono", "pure_coherence", "pg_unified_v12"):
        assert alt[arm]["r_ratio"] > 0.5
        assert alt[arm]["tail_mean_step_x"] > 1e-3
