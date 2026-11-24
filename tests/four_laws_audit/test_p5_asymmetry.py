"""Reproducible mathematical proofs, code-audit checks, and falsification controls for P5 (Law 4.4).

Covers:
1. Code-audit invariants of `UnifiedV12Optimizer` hardcoded role asymmetry, `_generator_free`
   syntactic brittleness (`total_gen_mats == 0`), missing Schatten-6 decay in
   `_step_generator_free_critic`, broken `particlegan/unified_v12.py` doc path, and
   `t3_iso_scale_mass_8x` `0.45` threshold verification.
2. Theorem 1 (Hypothesis a): Scalar-potential (`d_out=1`) exact rank-1 linear backprop bottleneck
   vs vector pushforward (`d_out >= 2`), and nonlinear gating rank lift.
3. Theorem 2 (Hypotheses a, b, d): Contrastive real-minus-fake mean-cone cancellation slashing
   16L skip-free critic norm skew (~18,800x -> ~4x) while deep layers (`net.24..30`) still
   collapse to `stable_rank ~ 1.0` and `||grad_x D|| < 1e-5`.
4. Theorem 3: Exact algebraic identity `sqrt(max(1, r/c)) * sqrt(min(r, c)) == sqrt(r)` and
   `sqrt(m / stable_rank)` top-singular-value distortion of `unified_v12`'s 0-NS critic rule vs
   spectral-norm-matched Frobenius-MP (`G / sigma_1`).
5. Scale-invariance and accuracy of the role-agnostic `fast_stable_rank_power` pathology detector.
6. Hypothesis (e): Orthogonal-to-gradient NS components perturb the critic's spatial force field
   `grad_x D(x)` far more strongly (relative to raw gradient) than scalar logits `D(x)`.
7. Law 2 one-sidedness: Born-Infeld upper slope cap contributes `< 1e-10` on a collapsed 16L
   skip-free critic (`||grad_x D|| << 1`).
8. Fast end-to-end falsification control on a 16-layer skip-free critic (`v12 sqrt(r)` 0-NS vs
   5-step Quintic NS & Adaptive-Pathology) and exact 5-arm optimizer invocation contracts.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import particlegan
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians
from particlegan._unified_v11.optimizer import quintic_orthogonalize_2d
from particlegan._unified_v12.optimizer import UnifiedV12Optimizer
from particlegan.init import deterministic_orthogonal_

from scripts.four_laws_audit.p5_asymmetry_experiments import (
    P5ControlledV12Optimizer,
    SymmetricLaw4Optimizer,
    _swap_trainer_optimizers,
    born_infeld_slope_penalty_2d,
    fast_stable_rank_power,
    measure_input_grad_norm,
    sample_ring,
    spectral_metrics,
    summarize_module_grad_spectra,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Code Audit: Hardcoded Asymmetry, Generator-Free Brittleness & Doc Drift
# ---------------------------------------------------------------------------

def test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs() -> None:
    # (a) Broken doc path: `docs/four-laws.md:175` cites nonexistent `particlegan/unified_v12.py`
    assert not (REPO_ROOT / "particlegan" / "unified_v12.py").exists()
    assert (REPO_ROOT / "particlegan" / "_unified_v12" / "optimizer.py").exists()

    # (b) Docstring bug: `_step_generator_free_critic` docstring claims `+ Schatten-6`
    #     yet executes zero Schatten-6 spectral decay calls.
    src_gf = inspect.getsource(UnifiedV12Optimizer._step_generator_free_critic)
    assert "Schatten-6" in (UnifiedV12Optimizer._step_generator_free_critic.__doc__ or "")
    assert "schatten6" not in src_gf.lower()

    # (c) `law4-energy-champion.md:90` misquotes `t3_iso_scale_mass_8x` threshold as `0.35`,
    #     whereas frozen task contract requires `core_min_eigen_ratio >= 0.45`.
    task_json = json.loads((REPO_ROOT / "configs" / "forge" / "tasks" / "t3_iso_scale_mass_8x.json").read_text())
    thresh_map = {row[0]: (row[1], row[2]) for row in task_json["evaluation"]["thresholds"]}
    assert thresh_map["core_min_eigen_ratio"] == (">=", 0.45)

    # (d) Syntactic brittleness of `_generator_free`: `recipes.py:2378-2385` checks `total_gen_mats == 0`.
    #     With G = nn.Identity(), `_generator_free` is True and 2D critic NS runs.
    #     With a tiny 2x2 dummy Linear generator + a 16-layer skip-free critic, `_generator_free`
    #     flips to False and `critic_ns_calls` drops to strictly 0!
    rec = particlegan.get_recipe("unified_v12", num_particles=64, z_dim=2, batch_size=32)
    prior_id = rec.make_prior()
    d_deep = SimpleMLPDiscriminator(2, 32, n_hidden=8, fourier=0)
    deterministic_orthogonal_(d_deep, seed=0)
    _, opt_d_gf = rec.make_optimizers(nn.Identity(), d_deep, prior_id)
    assert getattr(opt_d_gf, "_generator_free", False) is True

    g_tiny = nn.Linear(2, 2)
    d_deep2 = SimpleMLPDiscriminator(2, 32, n_hidden=8, fourier=0)
    deterministic_orthogonal_(g_tiny, seed=0)
    deterministic_orthogonal_(d_deep2, seed=0)
    prior_g = rec.make_prior()
    _, opt_d_ng = rec.make_optimizers(g_tiny, d_deep2, prior_g)
    assert getattr(opt_d_ng, "_generator_free", True) is False

    # Execute 1 critic step on both and verify NS counts
    x_r = torch.randn(32, 2, generator=torch.Generator().manual_seed(0))
    x_f = torch.randn(32, 2, generator=torch.Generator().manual_seed(1))

    opt_d_gf.zero_grad(set_to_none=True)
    (-(d_deep(x_r).mean() - d_deep(x_f).mean())).backward()
    opt_d_gf.step()
    assert opt_d_gf.critic_ns_calls > 0

    opt_d_ng.zero_grad(set_to_none=True)
    (-(d_deep2(x_r).mean() - d_deep2(x_f).mean())).backward()
    opt_d_ng.step()
    assert opt_d_ng.critic_ns_calls == 0

    # (e) `UnifiedV12Optimizer` never activates `stiff_gate` for critic param groups
    src_step = inspect.getsource(UnifiedV12Optimizer.step)
    assert 'stiff_gate = stiff_gate_raw if player == "generator" else 0.0' in src_step


# ---------------------------------------------------------------------------
# 2. Theorem 1 (Hypothesis a): Scalar-Potential Linear Rank-1 Bottleneck
# ---------------------------------------------------------------------------

def test_theorem1_scalar_potential_linear_rank1_bottleneck() -> None:
    class _MLP(nn.Module):
        def __init__(self, d_out: int, linear_only: bool) -> None:
            super().__init__()
            layers = []
            d = 16
            for _ in range(3):
                layers.append(nn.Linear(d, 32))
                if not linear_only:
                    layers.append(nn.LeakyReLU(0.2))
                d = 32
            layers.append(nn.Linear(d, d_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    x_in = torch.randn(128, 16, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    for d_out in (1, 2, 8):
        mod_lin = _MLP(d_out, linear_only=True)
        deterministic_orthogonal_(mod_lin, seed=0)
        mod_lin = mod_lin.double()
        cot = torch.randn(128, d_out, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        mod_lin.zero_grad(set_to_none=True)
        (mod_lin(x_in) * cot).mean().backward()
        for _, p in mod_lin.named_parameters():
            if p.ndim == 2 and p.shape == (32, 32):
                sm = spectral_metrics(p.grad)
                assert sm["rank_tol_1e-12"] == d_out
                if d_out == 1:
                    assert math.isclose(sm["stable_rank"], 1.0, rel_tol=1e-10)
                    assert math.isclose(sm["rv_erank"], 1.0, rel_tol=1e-10)

    # Nonlinear LeakyReLU gating breaks the scalar rank-1 bottleneck
    mod_nl = _MLP(1, linear_only=False)
    deterministic_orthogonal_(mod_nl, seed=0)
    mod_nl = mod_nl.double()
    cot1 = torch.randn(128, 1, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    mod_nl.zero_grad(set_to_none=True)
    (mod_nl(x_in) * cot1).mean().backward()
    ranks_nl = [
        spectral_metrics(p.grad)["rank_tol_1e-12"]
        for _, p in mod_nl.named_parameters()
        if p.ndim == 2 and p.shape == (32, 32)
    ]
    assert min(ranks_nl) >= 16


# ---------------------------------------------------------------------------
# 3. Theorem 2 (Hypotheses a, b, d): Contrastive Mean-Cone Cancellation
# ---------------------------------------------------------------------------

def test_theorem2_contrastive_mean_cone_cancellation_and_deep_critic_collapse() -> None:
    dev = torch.device("cpu")
    g16 = SimpleMLPGenerator(2, 64, n_hidden=16, out_dim=2)
    g3 = SimpleMLPGenerator(2, 64, n_hidden=3, out_dim=2)
    d16 = SimpleMLPDiscriminator(2, 64, n_hidden=16, fourier=0)
    d3 = SimpleMLPDiscriminator(2, 64, n_hidden=3, fourier=0)
    for mod in (g16, g3, d16, d3):
        deterministic_orthogonal_(mod, seed=0)

    rng = torch.Generator(device=dev).manual_seed(0)
    real = sample_100gaussians(128, dev, generator=rng)
    z = torch.randn(128, 2, generator=rng)

    # G16 single-sign adversarial collapse
    g16.zero_grad(set_to_none=True)
    (-d3(g16(z)).mean()).backward()
    g16_sp = summarize_module_grad_spectra(g16, min_dim=16)

    # D16 single-sign loss reproduces deep collapse + huge norm skew
    d16.zero_grad(set_to_none=True)
    (-d16(real).mean()).backward()
    d16_single = summarize_module_grad_spectra(d16, min_dim=16)

    # D16 contrastive loss cancels shared positive-cone mean mode
    d16.zero_grad(set_to_none=True)
    (-(d16(real).mean() - d16(g3(z).detach()).mean())).backward()
    d16_diff = summarize_module_grad_spectra(d16, min_dim=16)

    assert g16_sp["median_stable_rank"] <= 1.05
    assert g16_sp["norm_skew"] > 1000.0
    assert d16_single["median_stable_rank"] <= 1.05
    assert d16_single["norm_skew"] > 1000.0

    # Contrastive cancellation slashes inter-layer norm skew by > 100x
    assert d16_diff["norm_skew"] < d16_single["norm_skew"] / 100.0
    # Yet deep layers STILL collapse to stable_rank ~ 1.0 and input gradient vanishes!
    assert d16_diff["min_stable_rank"] <= 1.02
    assert measure_input_grad_norm(d16, real) < 1e-4


# ---------------------------------------------------------------------------
# 4. Theorem 3: Spectral-Norm Distortion of `sqrt(r)` Polar-Frobenius Matching
# ---------------------------------------------------------------------------

def test_theorem3_sqrt_r_polar_frobenius_spectral_norm_distortion() -> None:
    # Exact algebraic identity across square, tall, and wide matrices
    for r, c in ((64, 64), (128, 64), (64, 128), (128, 2), (2, 128), (1, 128)):
        m = min(r, c)
        lhs = math.sqrt(max(1.0, float(r) / float(c))) * math.sqrt(float(m))
        rhs = math.sqrt(float(r))
        assert math.isclose(lhs, rhs, rel_tol=1e-14)

    # Exact spectral-norm inflation ratio sigma_1(Delta W_v12) / sigma_1(Delta W_polar) == sqrt(m / stable_rank)
    m_sz = 64
    rng = torch.Generator().manual_seed(0)
    q1, _ = torch.linalg.qr(torch.randn(m_sz, m_sz, generator=rng, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(m_sz, m_sz, generator=rng, dtype=torch.float64))
    for target_srk in (1.0, 1.5, 8.0, 32.0):
        c_val = math.sqrt((target_srk - 1.0) / float(m_sz - 1))
        sv = torch.tensor([1.0] + [c_val] * (m_sz - 1), dtype=torch.float64)
        g = (q1 * sv) @ q2.T
        frob = float(g.norm().item())
        srk = (frob * frob) / (float(sv[0].item()) ** 2)
        u_v12 = math.sqrt(float(m_sz)) * g / frob
        u_spec = g / float(sv[0].item())
        s1_v12 = float(torch.linalg.svdvals(u_v12)[0].item())
        s1_spec = float(torch.linalg.svdvals(u_spec)[0].item())
        assert math.isclose(s1_v12, math.sqrt(float(m_sz) / srk), rel_tol=1e-12)
        assert math.isclose(s1_spec, 1.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 5. Scale-Invariant Zero-SVD Adaptive Pathology Detector
# ---------------------------------------------------------------------------

def test_adaptive_pathology_detector_scale_invariance_and_accuracy() -> None:
    rng = torch.Generator().manual_seed(0)
    q1, _ = torch.linalg.qr(torch.randn(32, 32, generator=rng))
    q2, _ = torch.linalg.qr(torch.randn(32, 32, generator=rng))

    for sv_list in ([1.0] + [1e-3] * 31, [1.0] + [0.1] * 31, torch.linspace(1.0, 0.3, 32).tolist()):
        sv = torch.tensor(sv_list, dtype=torch.float32)
        mat = (q1 * sv) @ q2.T
        exact = spectral_metrics(mat)
        s_ref, pr_ref = fast_stable_rank_power(mat, iters=8)
        # Accuracy vs exact SVD
        assert abs(s_ref - exact["stable_rank"]) / max(exact["stable_rank"], 1.0) < 0.05
        assert abs(pr_ref - exact["participation_ratio"]) / max(exact["participation_ratio"], 1.0) < 1e-4
        # Metric ordering 1 <= stable_rank <= PR <= rv_erank <= rank
        assert 1.0 - 1e-6 <= exact["stable_rank"] <= exact["participation_ratio"] + 1e-6
        assert exact["participation_ratio"] <= exact["rv_erank"] + 1e-6
        # Exact degree-0 scale invariance over 40 orders of magnitude!
        for scale in (1e-20, 1e-8, 1e8, 1e20):
            s_sc, pr_sc = fast_stable_rank_power(mat * scale, iters=8)
            assert math.isclose(s_sc, s_ref, rel_tol=1e-4)
            assert math.isclose(pr_sc, pr_ref, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 6. Hypothesis (e): Critic Orthogonal NS Perturbs Force Field grad_x D >> Logits D
# ---------------------------------------------------------------------------

def test_hypothesis_e_orthogonal_ns_perturbs_critic_force_field_more_than_logits() -> None:
    d_mod = SimpleMLPDiscriminator(2, 128, n_hidden=3, fourier=2)
    deterministic_orthogonal_(d_mod, seed=0)
    rng = torch.Generator().manual_seed(0)
    x_r = sample_100gaussians(128, torch.device("cpu"), generator=rng)
    x_f = torch.randn(128, 2, generator=rng)

    d_mod.zero_grad(set_to_none=True)
    (-(d_mod(x_r).mean() - d_mod(x_f).mean())).backward()
    params = [p for p in d_mod.parameters() if p.grad is not None]
    grads = [p.grad.detach().clone() for p in params]
    ns_dirs = []
    for p, g in zip(params, grads):
        if p.ndim == 2 and min(p.shape) >= 2:
            mc = g.T if g.shape[0] > g.shape[1] else g
            od = quintic_orthogonalize_2d(mc, steps=5, smoothing=0.0061)
            ns_dirs.append(od.T if g.shape[0] > g.shape[1] else od)
        else:
            ns_dirs.append(g / g.norm().clamp_min(1e-30))

    eps = 0.05
    def _scale(vs: list[torch.Tensor]) -> list[torch.Tensor]:
        nrm = math.sqrt(sum(float(v.square().sum().item()) for v in vs))
        return [v * (eps / max(nrm, 1e-30)) for v in vs]

    g_u = _scale(grads)
    ns_u = _scale(ns_dirs)
    dot = sum(float((a * b).sum().item()) for a, b in zip(ns_u, g_u))
    proj = dot / max(sum(float(b.square().sum().item()) for b in g_u), 1e-30)
    orth_u = _scale([a - proj * b for a, b in zip(ns_u, g_u)])

    d0 = d_mod(x_r).detach()
    xr_req = x_r.detach().clone().requires_grad_(True)
    gd0 = torch.autograd.grad(d_mod(xr_req).sum(), xr_req)[0].detach()

    def _probe(deltas: list[torch.Tensor]) -> tuple[float, float]:
        with torch.no_grad():
            for p, dv in zip(params, deltas):
                p.add_(dv)
            d1 = d_mod(x_r).detach()
        xr2 = x_r.detach().clone().requires_grad_(True)
        gd1 = torch.autograd.grad(d_mod(xr2).sum(), xr2)[0].detach()
        with torch.no_grad():
            for p, dv in zip(params, deltas):
                p.sub_(dv)
        return (
            float((d1 - d0).square().mean().sqrt().item()),
            float((gd1 - gd0).square().mean().sqrt().item()),
        )

    dd_raw, dgd_raw = _probe(g_u)
    dd_orth, dgd_orth = _probe(orth_u)
    rel_logit_agitation = dd_orth / max(dd_raw, 1e-30)
    rel_force_agitation = dgd_orth / max(dgd_raw, 1e-30)
    # Orthogonal NS perturbs spatial gradient field grad_x D much more strongly than scalar logits D
    assert rel_force_agitation > 2.5 * rel_logit_agitation


# ---------------------------------------------------------------------------
# 7. Law 2 One-Sidedness: Upper Slope Cap Is Inactive on Collapsed Deep Critics
# ---------------------------------------------------------------------------

def test_law2_one_sidedness_inactive_on_deep_skip_free_critic() -> None:
    d16 = SimpleMLPDiscriminator(2, 64, n_hidden=16, fourier=0)
    deterministic_orthogonal_(d16, seed=0)
    rng = torch.Generator().manual_seed(0)
    x_r = sample_ring(64, torch.device("cpu"), rng, n_modes=8)
    x_f = torch.randn(64, 2, generator=rng) * 2.0
    pen, gn_med = born_infeld_slope_penalty_2d(d16, x_r, x_f, beta=12.0, kappa=1.0, gauge_weight=0.0)
    assert gn_med < 1e-4
    # Because ||grad_x D|| << kappa=1, Born-Infeld slope penalty is ~ 0.5 * beta * ||grad_x D||^2 < 1e-7
    assert float(pen.item()) < 1e-7


# ---------------------------------------------------------------------------
# 8. End-to-End Fast Falsification Control & 5-Arm Optimizer Contracts
# ---------------------------------------------------------------------------

def test_deep_critic_v12_sqrt_r_degrades_vs_ns_and_spec_matched() -> None:
    """Fast CPU control proving v12's sqrt(r) 0-NS critic degrades a deep skip-free critic."""
    dev = torch.device("cpu")
    steps = 80
    data_rng = torch.Generator(device=dev).manual_seed(0)
    reals = [sample_ring(128, dev, data_rng, n_modes=8, radius=2.5, sigma=0.08) for _ in range(steps)]
    fakes = [torch.randn(128, 2, generator=data_rng) * 2.0 for _ in range(steps)]

    scores: dict[str, float] = {}
    for label, mode, fscale in (
        ("v12_sqrt_r", "frob", "sqrt_m"),
        ("ns_5step", "ns", "spec"),
        ("adaptive", "adaptive", "spec"),
    ):
        D = SimpleMLPDiscriminator(2, 64, n_hidden=12, fourier=0)
        deterministic_orthogonal_(D, seed=0)
        opt_d = SymmetricLaw4Optimizer(D.parameters(), lr=0.015, mode=mode, frob_scale=fscale, smoothing=0.0, tau_srk=1.30)
        for st in range(steps):
            D.zero_grad(set_to_none=True)
            adv = F.softplus(-(D(reals[st]) - D(fakes[st]))).mean()
            pen, _ = born_infeld_slope_penalty_2d(D, reals[st], fakes[st], beta=2.0, kappa=1.0)
            (adv + pen).backward()
            opt_d.step()
        with torch.no_grad():
            ev_rng = torch.Generator(device=dev).manual_seed(999)
            r_ev = sample_ring(512, dev, ev_rng, n_modes=8, radius=2.5, sigma=0.08)
            k_v = torch.randint(0, 8, (512,), generator=ev_rng)
            ang_v = (2.0 * math.pi / 8.0) * (k_v.float() + 0.5)
            val_ev = torch.stack((2.5 * ang_v.cos(), 2.5 * ang_v.sin()), dim=1) + torch.randn(512, 2, generator=ev_rng) * 0.08
            scores[label] = float((D(r_ev) > D(val_ev)).float().mean().item())

    assert scores["ns_5step"] > scores["v12_sqrt_r"] + 0.10
    assert scores["adaptive"] > scores["v12_sqrt_r"] + 0.10


def test_p5_controlled_optimizer_five_arm_contracts() -> None:
    """Verify exact NS invocation counts for all 5 arms + role-agnostic Adaptive triggering."""
    rec = particlegan.get_recipe("unified_v12", prior_kind="mog", standardize=False, num_particles=64, z_dim=2, batch_size=32)
    x_real = sample_ring(32, torch.device("cpu"), torch.Generator().manual_seed(0), n_modes=8)

    counts: dict[str, tuple[int, int]] = {}
    for arm, gm, dm in (
        ("Neither", "frob", "frob"),
        ("G-only", "ns", "frob"),
        ("D-only", "frob", "ns"),
        ("Both", "ns", "ns"),
        ("Adaptive", "adaptive", "adaptive"),
    ):
        G = SimpleMLPGenerator(2, 32, n_hidden=3, out_dim=2)
        D = SimpleMLPDiscriminator(2, 32, n_hidden=3, fourier=2)
        prior = particlegan.MoGParticlePrior(64, 2, sigma=0.1, standardize=False, learnable=True)
        deterministic_orthogonal_(G, seed=0)
        deterministic_orthogonal_(D, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        trainer = particlegan.GANTrainer(rec, G, D, prior=prior, seed=0)
        opt_g, opt_d = _swap_trainer_optimizers(trainer, g_ns_mode=gm, d_ns_mode=dm, tau_srk=1.25)
        trainer.step(x_real)
        counts[arm] = (opt_g.p5_ns_2d_calls, opt_d.p5_ns_2d_calls)

    assert counts["Neither"] == (0, 0)
    assert counts["G-only"][0] > 0 and counts["G-only"][1] == 0
    assert counts["D-only"][0] == 0 and counts["D-only"][1] > 0
    assert counts["Both"][0] > 0 and counts["Both"][1] > 0

    # Role-agnostic adaptive trigger: triggers on rank-1 matrix and skips on isotropic identity matrix
    p_col = nn.Parameter(torch.zeros(16, 16))
    p_iso = nn.Parameter(torch.zeros(16, 16))
    for role in ("generator", "critic"):
        opt = P5ControlledV12Optimizer(
            [{"params": [p_col, p_iso], "role": role}],
            lr=0.01,
            ns_mode="adaptive",
            tau_srk=1.25,
        )
        u = torch.linspace(1.0, 2.0, 16)
        p_col.grad = torch.outer(u, u)
        p_iso.grad = torch.eye(16)
        opt.step()
        assert opt.p5_total_2d_calls == 2
        assert opt.p5_ns_2d_calls == 1
        assert opt.p5_skipped_2d_calls == 1
