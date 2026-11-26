"""Comprehensive CPU+GPU semantic, invariant, tamper, hygiene, and clock-free tests for Axiom Module C."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_PATH = ROOT / "particlegan" / "_axiom" / "transport.py"


def _load_transport_module():
    """Import transport.py standalone so Stage-2 tests work before _axiom/__init__.py lands."""
    spec = importlib.util.spec_from_file_location("particlegan._axiom.transport", str(TRANSPORT_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_transport_module()


@dataclass
class _StubAxiomRecord:
    """Strict duck-typed AxiomRecordProtocol stub matching reports/axiom/design-spec.md §3.1.4."""

    calls: int = 0
    observed_steps: int = 0
    chord_phase: float = 0.123456
    ell_ema: float = 0.0
    ell_initialized: bool = False
    thermal_scale: float = 1.0
    thermal_initialized: bool = False
    thermal_ema: float = 1.0
    thermal_locked: bool = False
    step_scale: float = 1.0
    cond_ema: float = 0.25
    cond_initialized: bool = False
    het_ema: float = 1.0
    het_initialized: bool = False
    chart_state: Optional[Dict[str, Any]] = None
    atomic_ledger: Optional[Dict[str, Any]] = None
    cond_fiber_bank: Optional[Dict[str, Any]] = None
    _cond_bank_obj: Any = None
    _in_penalty: bool = False
    _in_probe: bool = False
    _in_protected_probe: bool = False
    _simplicial_step: bool = False
    _disc_gate: float = 0.0
    _active_prior_table: Optional[torch.Tensor] = None
    _active_batch_rows: Optional[torch.Tensor] = None
    thermal_calls: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    last_shock_mu_ratio: float = 0.35

    def observe_thermal(
        self,
        r_ed: float,
        r_mu_sq: float,
        r_spread: float,
        r_q90: float,
        r_hmax: float,
        *,
        r_tail: float = 0.0,
        tau: float = 0.084,
        shock_mu_ratio: float = 0.35,
        shock_ed_floor: float = 0.30,
        lock_hysteresis: float = 0.0,
    ) -> float:
        # Strict signature guard matching AxiomStepRecord.observe_thermal (6-channel thermal EMA).
        self.thermal_calls.append((float(r_ed), float(r_mu_sq), float(r_spread), max(float(r_q90), float(r_tail)), float(r_hmax), float(tau)))
        self.last_shock_mu_ratio = float(shock_mu_ratio)
        self.thermal_initialized = True
        self.thermal_ema = max(0.0, float(r_ed) + max(0.0, float(r_q90), float(r_tail)))
        return self.thermal_scale


def _make_arc_batch(n: int = 64, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(11)
    t = torch.rand(n, generator=g) * math.pi
    return torch.stack((torch.cos(t), torch.sin(t)), dim=1).to(device)


def _make_saddle_batch(n: int = 64, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(17)
    uv = torch.rand(n, 2, generator=g) * 2.0 - 1.0
    return torch.stack((uv[:, 0], uv[:, 1], uv[:, 0] * uv[:, 1]), dim=1).to(device)


def _make_disconnected_modes(n: int = 64, *, sigma: float = 0.12, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(23)
    centers = torch.tensor([[-6.0, -6.0], [-6.0, 6.0], [6.0, -6.0], [6.0, 6.0]])
    idx = torch.randint(0, 4, (n,), generator=g)
    return (centers[idx] + sigma * torch.randn(n, 2, generator=g)).to(device)


def _make_exact_atoms(k: int = 8, n: int = 64, d: int = 4, *, seed: int = 31, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    if k == 256 and d == 8:
        bits = ((torch.arange(256).unsqueeze(1) >> torch.arange(8)) & 1).float() * 2.0 - 1.0
        atoms = bits
    else:
        atoms = (torch.randint(0, 2, (k, d), generator=g).float() * 2.0 - 1.0) * 3.0
        atoms = atoms + torch.arange(k).float().unsqueeze(1) * 0.35
    idx = torch.randint(0, k, (n,), generator=g)
    return atoms[idx].to(device), atoms.to(device)


def test_transport_hygiene_loc_and_zero_forbidden_ops(monkeypatch: pytest.MonkeyPatch):
    src = TRANSPORT_PATH.read_text(encoding="utf-8")
    lines = src.splitlines()
    assert len(lines) <= 695, f"transport.py exceeds 695 LOC budget: {len(lines)}"
    tree = ast.parse(src, filename=str(TRANSPORT_PATH))

    forbidden_imports = {
        "_unified", "_unified_v4", "_unified_v11", "_t1_coupling", "_t2_critic", "_t3_dynamics",
        "_t4_arch", "_t4_nash", "_rhbi", "_empf", "_cond_fiber", "_minimal", "_bispf",
        "_nash_v3", "_r4_ablation", "bispf", "rhbi", "k3p", "grad_regularizers",
        "prior", "critic", "optimizer",
    }
    forbidden_attrs = {
        "eigh", "eigvalsh", "svd", "svdvals", "pinv", "compile", "matrix_norm",
        "observed_steps", "calls", "total_steps", "max_steps",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not (set(alias.name.split(".")) & forbidden_imports), f"Forbidden import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lstrip(".")
            assert not (set(mod.split(".")) & forbidden_imports), f"Forbidden import-from {node.module}"
            for alias in node.names:
                assert alias.name not in forbidden_imports, f"Forbidden imported name {alias.name}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs, f"Forbidden attribute .{node.attr} at L{node.lineno}"

    def _forbid(*_a, **_kw):
        raise AssertionError("Forbidden eigh/eigvalsh/svd/svdvals/pinv/matrix_norm/autograd.grad invoked!")

    monkeypatch.setattr(torch.linalg, "eigh", _forbid)
    monkeypatch.setattr(torch.linalg, "eigvalsh", _forbid)
    monkeypatch.setattr(torch.linalg, "svd", _forbid)
    monkeypatch.setattr(torch.linalg, "svdvals", _forbid)
    monkeypatch.setattr(torch.linalg, "pinv", _forbid)
    monkeypatch.setattr(torch.linalg, "matrix_norm", _forbid)
    monkeypatch.setattr(torch, "svd", _forbid)
    monkeypatch.setattr(torch.autograd, "grad", _forbid)

    loss_fn = T.AxiomTransportLoss()
    rec = _StubAxiomRecord()
    y_disc = _make_disconnected_modes(48)
    x_fake = (y_disc.clone() + 0.5).requires_grad_(True)
    d_fake = (x_fake.sum(dim=1) * 0.1).requires_grad_(True)
    d_real = y_disc.sum(dim=1) * 0.1
    l_disc = loss_fn(x_fake, y_disc, d_fake, d_real, record=rec)
    l_disc.backward()
    assert torch.isfinite(l_disc) and x_fake.grad is not None and torch.isfinite(x_fake.grad).all()

    z_tab = torch.randn(24, 2, requires_grad=True)
    gen = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))
    l_simp = loss_fn.simplicial_manifold_loss(z_tab, gen, _make_saddle_batch(48), record=rec)
    l_simp.backward()
    assert torch.isfinite(l_simp) and z_tab.grad is not None

    rec_c = _StubAxiomRecord()
    g = torch.Generator().manual_seed(7)
    for _ in range(3):
        c = torch.randn(32, 2, generator=g)
        yr = c * 1.5 + 0.3 * torch.randn(32, 2, generator=g)
        xf = (c * 1.2 + 0.4 * torch.randn(32, 2, generator=g)).requires_grad_(True)
        lc = loss_fn.conditional_fiber_loss(xf, yr, c, cond_disc=torch.zeros(32, 1, dtype=torch.long), record=rec_c)
        lc.backward()
        assert torch.isfinite(lc) and xf.grad is not None


def test_zero_eigh_symmetric_and_eigensplit_parity_and_sign_canon():
    g = torch.Generator().manual_seed(101)
    for d in (1, 2, 3, 4, 8, 16):
        a = torch.randn(d, d, dtype=torch.float64, generator=g)
        sym = 0.5 * (a + a.T)
        vals, vecs = T.zero_eigh_symmetric(sym)
        ref_vals, _ = torch.linalg.eigh(sym)
        assert torch.allclose(vals, ref_vals, atol=1e-12, rtol=1e-12), f"d={d} eigval mismatch"
        recon = (vecs * vals.unsqueeze(0)) @ vecs.T
        assert torch.allclose(recon, sym, atol=1e-12, rtol=1e-12), f"d={d} recon mismatch"
        assert torch.allclose(vecs.T @ vecs, torch.eye(d, dtype=torch.float64), atol=1e-12), f"d={d} non-orthonormal"
        piv = torch.gather(vecs, -2, vecs.abs().argmax(dim=-2, keepdim=True))
        assert bool((piv >= 0.0).all()), f"d={d} failed column-sign canonicalization"
        vals2, vecs2 = T.zero_eigh_symmetric(sym)
        assert torch.equal(vals, vals2) and torch.equal(vecs, vecs2)

    batch = torch.randn(2, 5, 4, 4, dtype=torch.float64, generator=g)
    bsym = 0.5 * (batch + batch.transpose(-1, -2))
    bvals, bvecs = T.zero_eigh_symmetric(bsym)
    ref_bvals, _ = torch.linalg.eigh(bsym)
    assert torch.allclose(bvals, ref_bvals, atol=1e-12, rtol=1e-12)
    ep, en = T.sym_eigensplit_zero_eigh(bsym)
    assert torch.allclose(ep + en, bsym, atol=1e-12, rtol=1e-12)
    assert float(torch.linalg.eigvalsh(ep).min().item()) >= -1e-11
    assert float(torch.linalg.eigvalsh(en).max().item()) <= 1e-11

    # Degenerate zero & diagonal matrices + invalid shape error handling
    zvals, zvecs = T.zero_eigh_symmetric(torch.zeros(4, 4, dtype=torch.float64))
    assert torch.allclose(zvals, torch.zeros(4, dtype=torch.float64))
    assert torch.allclose(zvecs.T @ zvecs, torch.eye(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        T.zero_eigh_symmetric(torch.randn(3, 4))
    with pytest.raises(ValueError):
        T.zero_eigh_symmetric(torch.randn(4))


def test_project_transport_features_and_scale_overflow_safety():
    vec = torch.randn(8, 12, requires_grad=True)
    assert torch.equal(T.project_transport_features(vec, max_spatial=8), vec)
    img8 = torch.randn(4, 3, 8, 8, requires_grad=True)
    assert T.project_transport_features(img8, max_spatial=8).shape == (4, 3 * 64)
    assert torch.equal(T.project_transport_features(img8, max_spatial=8), img8.reshape(4, -1))

    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        img65 = torch.ones(4, 3, 65, 65, requires_grad=True)
        proj = T.project_transport_features(img65, max_spatial=8)
        assert proj.shape == (4, 3 * 64)
        # Constant image preserves exact L2 norm: sqrt(3*65*65)
        expected_norm = math.sqrt(3.0 * 65.0 * 65.0)
        assert abs(float(proj[0].norm().item()) - expected_norm) < 1e-3
        proj.sum().backward()
        assert img65.grad is not None and torch.isfinite(img65.grad).all()
    finally:
        torch.use_deterministic_algorithms(prev_det)

    # Extreme 1e18 scale overflow safety
    scale = 1e18
    y_huge = _make_disconnected_modes(24) * scale
    x_huge = (y_huge.clone() + 0.1 * scale).requires_grad_(True)
    c, ch = T._bi_cost(x_huge, y_huge, torch.tensor(scale))
    assert torch.isfinite(c).all() and torch.isfinite(ch).all()
    c.mean().backward()
    assert x_huge.grad is not None and torch.isfinite(x_huge.grad).all()


def test_law3a_connectedness_conformal_horizon_and_dead_zone_free_orphan():
    g_arc, _, _, _ = T._manifold_gates(_make_arc_batch(64))
    g_sad, _, _, _ = T._manifold_gates(_make_saddle_batch(64))
    g_disc, _, _, _ = T._manifold_gates(_make_disconnected_modes(64))
    assert g_arc > 0.85 and g_sad > 0.85, f"Expected connected gates > 0.85, got arc={g_arc}, sad={g_sad}"
    assert g_disc < 0.05, f"Expected disconnected gate < 0.05, got {g_disc}"

    # Conformal horizon MUST turn off identically on connected manifolds (g_conn > 0.5)
    y_sad = _make_saddle_batch(64)
    ell = torch.tensor(0.25)
    exy_s, exx_s = T._conformal_horizon(y_sad + 0.1, y_sad, ell, g_sad, False)
    assert exy_s is ell and exx_s is ell

    # On unequal-width 8x disconnected mixture, tight cluster gets smaller horizon & X==Y gives exact symmetry
    g = torch.Generator().manual_seed(41)
    y_tight = torch.tensor([-12.0, 0.0]) + 0.15 * torch.randn(32, 2, generator=g)
    y_broad = torch.tensor([12.0, 0.0]) + 1.80 * torch.randn(32, 2, generator=g)
    y_mix = torch.cat([y_tight, y_broad], dim=0)
    ell_m = T._batch_ell(y_mix, 0.25)
    exy_m, exx_m = T._conformal_horizon(y_mix, y_mix, ell_m, 0.0, False)
    assert torch.allclose(exy_m, exx_m, atol=1e-6), "Feydy symmetry violated at X == Y"
    assert float(exy_m[:32, :32].mean().item()) < 0.7 * float(exy_m[32:, 32:].mean().item())

    # T1 orphan macro thresholds: silent at 2.0*rho (<=2.80*rho), active at 3.5*rho (>2.80*rho, r_loc>3.52)
    rho = 3.0
    y_mode0 = torch.zeros(24, 2) + 0.1 * torch.randn(24, 2, generator=g)
    y_orphan = torch.tensor([6.5, 0.0]) + 0.1 * torch.randn(24, 2, generator=g)
    y_two = torch.cat([y_mode0, y_orphan], dim=0)
    x_stuck = (torch.zeros(48, 2) + 0.1 * torch.randn(48, 2, generator=g)).requires_grad_(True)
    ch_xy = torch.full((48, 48), 0.2)
    ch_xy[:, 24:] = 2.0 * rho
    prs = torch.full((48,), 0.5 / 48.0)
    pcs = torch.cat([torch.zeros(24), torch.full((24,), 1.0 / 48.0)])
    l_near = T._orphan_macro_loss(x_stuck, y_two, ch_xy, prs, pcs, rho=rho, g_conn=0.0)
    assert float(l_near.item()) == 0.0, f"T1 orphan macro false-triggered at 2.0*rho: {l_near}"
    ch_xy_far = ch_xy.clone()
    ch_xy_far[:, 24:] = 3.5 * rho  # strictly above 2.80 * rho with r_loc > 3.52
    l_orph = T._orphan_macro_loss(x_stuck, y_two, ch_xy_far, prs, pcs, rho=rho, g_conn=0.0)
    assert float(l_orph.item()) > 0.05, f"T1 orphan macro failed to activate at 3.5*rho: {l_orph}"
    l_orph.backward()
    assert x_stuck.grad is not None and float(x_stuck.grad[:, 0].mean().item()) < -1e-4  # pulls toward +6.5!

    # Covered broad cluster (r_loc ~ 1): must NOT false-trigger
    y_cov = torch.randn(48, 2, generator=g)
    x_cov = torch.randn(48, 2, generator=g).requires_grad_(True)
    l_cov = T._orphan_macro_loss(x_cov, y_cov, torch.full((48, 48), 2.5 * rho), prs, torch.full((48,), 1.0 / 48.0), rho=rho, g_conn=0.0)
    assert float(l_cov.item()) == 0.0

    # Equidistant symmetric saddle: breaks symmetry cleanly toward BOTH -10 and +10
    y_sym = torch.cat([torch.full((16, 2), -10.0), torch.full((16, 2), 10.0)], dim=0)
    y_sym[:, 1] = 0.0
    x_mid = torch.zeros(32, 2, requires_grad=True)
    l_sym = T._orphan_macro_loss(x_mid, y_sym, torch.full((32, 32), 4.0 * rho), torch.full((32,), 1.0 / 32.0), torch.full((32,), 1.0 / 32.0), rho=rho, g_conn=0.0)
    l_sym.backward()
    assert x_mid.grad is not None
    assert int((x_mid.grad[:, 0] > 1e-4).sum().item()) >= 12 and int((x_mid.grad[:, 0] < -1e-4).sum().item()) >= 12


def test_law3a_atomic_gate_hamilton_d06_and_gpu_argsort_ledger():
    # Annular-gap duplicate purity gate
    y_byte, atoms_byte = _make_exact_atoms(k=256, n=128, d=8, seed=5)
    y_five, _ = _make_exact_atoms(k=5, n=64, d=4, seed=9)
    assert T.compute_discrete_atomic_gate(y_byte) == 1.0
    assert T.compute_discrete_atomic_gate(y_five) == 1.0
    # Persistent-chart fallback when batch has 0 duplicates (unique atoms subset)
    assert T.compute_discrete_atomic_gate(atoms_byte[:32], charts=atoms_byte, radii=torch.full((256,), 1e-6)) == 1.0
    # Single point-mass (K=1, zero diameter) MUST return 0.0 unless >=4 sharp charts already exist
    pt_mass = torch.full((32, 8), 0.7)
    assert T.compute_discrete_atomic_gate(pt_mass) == 0.0
    assert T.compute_discrete_atomic_gate(atoms_byte[:1].expand(32, 8), charts=atoms_byte, radii=torch.full((256,), 1e-6)) == 1.0
    # Must be identically 0.0 on all continuous curved manifolds & noisy MoGs (sigma >= 1e-3)
    assert T.compute_discrete_atomic_gate(_make_arc_batch(64)) == 0.0
    assert T.compute_discrete_atomic_gate(_make_saddle_batch(64)) == 0.0
    for sig in (1e-3, 1e-2, 1e-1):
        assert T.compute_discrete_atomic_gate(_make_disconnected_modes(64, sigma=sig)) == 0.0

    # Hamilton quotas + D-06 permutation invariance
    q_eq = T._hamilton_quotas(256, torch.ones(256) / 256.0)
    assert torch.equal(q_eq, torch.ones(256, dtype=torch.long))
    q_un = T._hamilton_quotas(64, torch.tensor([0.50, 0.25, 0.15, 0.07, 0.03]))
    assert q_un.tolist() == [32, 16, 10, 4, 2]
    g = torch.Generator().manual_seed(77)
    scores = torch.randn(64, generator=g)
    perm = torch.randperm(64, generator=g)
    q_m_lt_k = T._hamilton_quotas(32, torch.ones(64) / 64.0, active_scores=scores)
    q_m_lt_k_p = T._hamilton_quotas(32, torch.ones(64) / 64.0, active_scores=scores[perm])
    assert int(q_m_lt_k.sum().item()) == 32
    assert torch.equal(q_m_lt_k_p, q_m_lt_k[perm]), "D-06 violation: M < K quotas not permutation-invariant!"
    q_weyl = T._hamilton_quotas(32, torch.ones(64) / 64.0)
    mean_sel = float(torch.nonzero(q_weyl > 0, as_tuple=False).squeeze(-1).float().mean().item())
    assert 24.0 <= mean_sel <= 39.0, f"Low-index bias detected in Weyl tie-break: mean_sel={mean_sel}"

    # Atomic ledger for M=64, K=5 (q_k > 1: regression test against scatter_reduce_('amin')!)
    _, atoms5 = _make_exact_atoms(k=5, n=64, d=4, seed=13)
    w5 = torch.ones(5) / 5.0
    xf64 = torch.randn(64, 4, generator=g)
    rows64 = torch.arange(64)
    _, _, ledger5, _ = T._update_atomic_ledger(xf64, rows64, atoms5, w5, 64, None, allow_update=True)
    cnt5 = torch.bincount(ledger5["row_chart"], minlength=5)
    assert int((ledger5["row_chart"] >= 0).sum().item()) == 64
    assert sorted(cnt5.tolist()) == [12, 13, 13, 13, 13], f"q_k>1 eviction bug: {cnt5.tolist()}"

    # Chart-index permutation invariance of re-anchoring
    p5 = torch.tensor([3, 0, 4, 1, 2])
    _, _, ledger5_p, _ = T._update_atomic_ledger(xf64, rows64, atoms5[p5], w5[p5], 64, ledger5, allow_update=False)
    assert torch.allclose(ledger5_p["row_target"], ledger5["row_target"]), "Ledger re-anchoring broke under chart permutation"

    # Re-anchoring priority: surviving-chart resident (rd_anchor==0) beats vanished-chart orphan (rd_anchor>0) even when neither is in current batch
    surv_charts = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    prev_led = {
        "row_chart": torch.tensor([0, 1, 0], dtype=torch.long),
        "row_target": torch.tensor([[3.0, 0.0], [0.0, 0.0], [10.0, 0.0]]),  # row 0 orphaned from vanished [3,0]; row 1 sits on [0,0]
    }
    _, _, pruned_led, _ = T._update_atomic_ledger(
        torch.tensor([[10.0, 0.0]]), torch.tensor([2]), surv_charts, torch.tensor([0.3, 0.7]), 3, prev_led, allow_update=True,
    )
    # Quotas for M=3, w=[0.3, 0.7] are strictly [1, 2]; neither row 0 nor row 1 is in batch -> resident row 1 MUST keep chart 0 and orphan row 0 MUST be released!
    assert int(pruned_led["row_chart"][1].item()) == 0 and int(pruned_led["row_chart"][0].item()) == -1

    # _atomic_match bounds guard: out-of-range or mismatched batch rows return None cleanly (no IndexError)
    loss_guard = T.AxiomTransportLoss()
    assert loss_guard._atomic_match(y_five, y_five, _StubAxiomRecord(), torch.randn(16, 2), torch.full((64,), 99), True) is None
    assert loss_guard._atomic_match(y_five, y_five, _StubAxiomRecord(), torch.randn(16, 2), torch.arange(8), True) is None

    # ByteGAN M=256, K=256 exact 256/256 bijection
    w256 = torch.ones(256) / 256.0
    xf256 = atoms_byte + 0.15 * torch.randn(256, 8, generator=g)
    ledger256 = None
    for start in range(0, 256, 64):
        rows_b = torch.arange(start, start + 64)
        _, _, ledger256, _ = T._update_atomic_ledger(xf256[rows_b], rows_b, atoms_byte, w256, 256, ledger256, allow_update=True)
    assert ledger256 is not None
    assert int((ledger256["row_chart"] >= 0).sum().item()) == 256
    assert int(torch.unique(ledger256["row_chart"]).numel()) == 256


def test_law3a_bigan_joint_relativistic_and_simplicial_manifold_loss():
    loss_fn = T.AxiomTransportLoss()
    # 1. BiGAN joint_relativistic_loss with exact discrete atoms (both [x, z] and [z, x] orderings)
    y_atoms, atoms8 = _make_exact_atoms(k=8, n=32, d=4, seed=55)
    z_table = torch.randn(16, 3)
    rows = torch.arange(16).repeat(2)
    for z_first in (False, True):
        rec = _StubAxiomRecord(_active_prior_table=z_table, _active_batch_rows=rows)
        x_f = (y_atoms.clone() + 0.4).requires_grad_(True)
        z_r = torch.randn(32, 3, requires_grad=True)
        z_f = z_table[rows].detach()
        jf = torch.cat([z_f, x_f], dim=1) if z_first else torch.cat([x_f, z_f], dim=1)
        jr = torch.cat([z_r, y_atoms], dim=1) if z_first else torch.cat([y_atoms, z_r], dim=1)
        df = (jf.sum(dim=1) * 0.1).requires_grad_(True)
        dr = jr.sum(dim=1) * 0.1
        l_j = loss_fn.joint_relativistic_loss(jf, jr, df, dr, record=rec)
        l_j.backward()
        assert torch.isfinite(l_j)
        assert x_f.grad is not None and float(x_f.grad.abs().sum().item()) > 0.0
        assert z_r.grad is not None and float(z_r.grad.abs().sum().item()) > 0.0
        assert rec.chart_state is not None and rec.atomic_ledger is not None
        assert len(rec.thermal_calls) == 1

    # 2. Simplicial manifold loss on 1D curve (Rayleigh collinearity) & 2D curved saddle (Heron + FP64 MDS)
    for y_data, out_dim in ((_make_arc_batch(64), 2), (_make_saddle_batch(64), 3)):
        rec_s = _StubAxiomRecord(chord_phase=0.37)
        torch.manual_seed(0)
        z_tab = (torch.randn(24, 2) * 0.5).requires_grad_(True)
        gen = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, out_dim))
        l_s = loss_fn.simplicial_manifold_loss(z_tab, gen, y_data, record=rec_s)
        l_s.backward()
        assert torch.isfinite(l_s)
        assert z_tab.grad is not None and float(z_tab.grad.abs().sum().item()) > 0.0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen.parameters())


def test_law3b_cond_fiber_bank_fp64_tamper_and_continuation():
    bank = T.AxiomCondFiberBank(ring_capacity=64, knn_k=12)
    g = torch.Generator().manual_seed(88)
    for _ in range(4):
        c = torch.randn(24, 2, generator=g)
        y = torch.stack((1.2 * c[:, 0] - 0.4 * c[:, 1], 0.5 * c[:, 0] + c[:, 1]), dim=1) + 0.25 * torch.randn(24, 2, generator=g)
        x = torch.stack((1.0 * c[:, 0], 0.8 * c[:, 1]), dim=1) + 0.40 * torch.randn(24, 2, generator=g)
        bank.update_stratum((1,), c, x, y, z_innov=0.0)

    # Critical Target-2 check: ALL 13 stratum tensors (incl. ring_c/x/y) MUST be CPU float64
    pkt = bank.to_packet()
    assert len(pkt["strata"]) == 1
    st = pkt["strata"][0]
    for k in ("mu_c", "m2_c", "g_pp", "g_px", "g_py", "g_pp_fast", "g_px_fast", "g_py_fast", "s_xx", "s_yy", "ring_c", "ring_x", "ring_y"):
        assert st[k].dtype == torch.float64 and st[k].device.type == "cpu", f"{k} not CPU float64"
    assert st["ring_c"].shape[0] == 64  # ring_capacity enforced

    # Bit-exact round-trip continuation in fresh bank
    bank2 = T.AxiomCondFiberBank(ring_capacity=64, knn_k=12)
    bank2.load_packet(pkt)
    cq, xq, yq = torch.randn(16, 2, generator=g), torch.randn(16, 2, generator=g), torch.randn(16, 2, generator=g)
    q1 = bank.query_stratum((1,), cq, xq, yq)
    q2 = bank2.query_stratum((1,), cq, xq, yq)
    assert q1 is not None and q2 is not None
    for k, v in q1.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, q2[k]), f"query_stratum mismatch on {k}"
        else:
            assert v == q2[k]

    # Innovation shock trims FIFO ring
    bank.update_stratum((1,), cq[:8], xq[:8], yq[:8], z_innov=8.0)
    assert int(bank.strata[(1,)]["ring_c"].shape[0]) < 64

    # Comprehensive tamper rejection
    bad_cases = []
    p_bad = deepcopy(pkt); p_bad["schema"] = 2; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["schema"] = True; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["extra"] = 1; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["d_c"] = 0; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"].append(deepcopy(pkt["strata"][0])); bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["n_eff"] = -1.0; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["n_eff"] = True; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["ring_c"] = p_bad["strata"][0]["ring_c"].float(); bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["ring_x"] = p_bad["strata"][0]["ring_x"][:-1]; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["m2_c"][0] = -0.1; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["g_pp"][0, 0] = -1.0; bad_cases.append(p_bad)
    p_bad = deepcopy(pkt); p_bad["strata"][0]["s_xx"][0, 0] = float("nan"); bad_cases.append(p_bad)
    for bad in bad_cases:
        with pytest.raises(ValueError):
            T.AxiomCondFiberBank.validate_packet(bad)

    # conditional_fiber_loss: lazy checkpoint hydration, zero mutation in probe/no_grad, and exact continuation
    loss_fn = T.AxiomTransportLoss()
    rec_a = _StubAxiomRecord()
    t_disc = torch.cat([torch.zeros(16, 1, dtype=torch.long), torch.ones(16, 1, dtype=torch.long)], dim=0)
    for _ in range(3):
        c = torch.randn(32, 2, generator=g)
        y = c * 1.4 + 0.2 * torch.randn(32, 2, generator=g)
        x = (c * 1.1 + 0.3 * torch.randn(32, 2, generator=g)).requires_grad_(True)
        lv = loss_fn.conditional_fiber_loss(x, y, c, cond_disc=t_disc, record=rec_a)
        lv.backward()
        assert torch.isfinite(lv) and x.grad is not None and float(x.grad.abs().sum().item()) > 0.0
    assert rec_a.cond_initialized

    # Hydrate fresh record from packet & verify bit-exact continuation
    saved_pkt = rec_a._cond_bank_obj.to_packet()
    rec_b = _StubAxiomRecord(cond_fiber_bank=deepcopy(saved_pkt), cond_ema=rec_a.cond_ema, cond_initialized=True)
    c_next, y_next = torch.randn(32, 2, generator=g), torch.randn(32, 2, generator=g)
    x_next_a = torch.randn(32, 2, generator=g).requires_grad_(True)
    x_next_b = x_next_a.detach().clone().requires_grad_(True)
    la = loss_fn.conditional_fiber_loss(x_next_a, y_next, c_next, cond_disc=t_disc, record=rec_a)
    lb = loss_fn.conditional_fiber_loss(x_next_b, y_next, c_next, cond_disc=t_disc, record=rec_b)
    la.backward(); lb.backward()
    assert torch.equal(la, lb) and torch.equal(x_next_a.grad, x_next_b.grad)
    assert rec_a.cond_ema == rec_b.cond_ema

    # Probe & no_grad guards must leave bank and cond_ema untouched
    snap_pkt = rec_b._cond_bank_obj.to_packet()
    snap_ema = rec_b.cond_ema
    with torch.no_grad():
        loss_fn.conditional_fiber_loss(x_next_b.detach(), y_next, c_next, cond_disc=t_disc, record=rec_b)
    rec_b._in_probe = True
    loss_fn.conditional_fiber_loss(x_next_b.detach().clone().requires_grad_(True), y_next, c_next, cond_disc=t_disc, record=rec_b)
    rec_b._in_probe = False
    rec_b._in_protected_probe = True
    loss_fn.conditional_fiber_loss(x_next_b.detach().clone().requires_grad_(True), y_next, c_next, cond_disc=t_disc, record=rec_b)
    rec_b._in_protected_probe = False
    assert rec_b.cond_ema == snap_ema
    after_pkt = rec_b._cond_bank_obj.to_packet()
    assert after_pkt["strata"][0]["n_eff"] == snap_pkt["strata"][0]["n_eff"]
    assert torch.equal(after_pkt["strata"][0]["ring_c"], snap_pkt["strata"][0]["ring_c"])


def test_forward_no_mutation_guards_and_strict_clock_freedom():
    loss_fn = T.AxiomTransportLoss()
    y = _make_disconnected_modes(32)
    x0 = y.clone() + 0.4
    d_r = y.sum(dim=1) * 0.1

    # Strict clock-freedom: shifting observed_steps & calls by +8192 yields bit-identical outputs
    rec1 = _StubAxiomRecord(calls=0, observed_steps=0, chord_phase=0.25)
    rec2 = _StubAxiomRecord(calls=8192, observed_steps=8192, chord_phase=0.25)
    x1 = x0.clone().requires_grad_(True)
    x2 = x0.clone().requires_grad_(True)
    l1 = loss_fn(x1, y, (x1.sum(dim=1) * 0.1), d_r, record=rec1)
    l2 = loss_fn(x2, y, (x2.sum(dim=1) * 0.1), d_r, record=rec2)
    l1.backward(); l2.backward()
    assert torch.equal(l1, l2) and torch.equal(x1.grad, x2.grad)
    assert rec1.ell_ema == rec2.ell_ema and rec1.thermal_calls == rec2.thermal_calls

    # Probe / allow_state_update=False / no_grad must not mutate record
    rec_guard = _StubAxiomRecord()
    with torch.no_grad():
        loss_fn(x0.clone(), y, x0.sum(dim=1) * 0.1, d_r, record=rec_guard)
    loss_fn(x0.clone().requires_grad_(True), y, x0.sum(dim=1) * 0.1, d_r, record=rec_guard, allow_state_update=False)
    rec_guard._in_probe = True
    loss_fn(x0.clone().requires_grad_(True), y, x0.sum(dim=1) * 0.1, d_r, record=rec_guard)
    rec_guard._in_probe = False
    assert not rec_guard.ell_initialized and rec_guard.chart_state is None and rec_guard.thermal_calls == []


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_axiom_transport_cuda_smoke():
    dev = torch.device("cuda:0")
    loss_fn = T.AxiomTransportLoss().to(dev)
    rec = _StubAxiomRecord()
    y = _make_disconnected_modes(64, device=dev)
    x = (y.clone() + 0.5).requires_grad_(True)
    l_f = loss_fn(x, y, x.sum(dim=1) * 0.1, y.sum(dim=1) * 0.1, record=rec)
    l_f.backward()
    assert torch.isfinite(l_f) and x.grad is not None and torch.isfinite(x.grad).all()

    z_tab = torch.randn(32, 2, device=dev, requires_grad=True)
    gen = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 3)).to(dev)
    l_s = loss_fn.simplicial_manifold_loss(z_tab, gen, _make_saddle_batch(64, device=dev), record=rec)
    l_s.backward()
    assert torch.isfinite(l_s) and z_tab.grad is not None and torch.isfinite(z_tab.grad).all()

    # GPU atomic ledger (2 consecutive updates exercising argsort pruning) + BiGAN + >8x8 spatial proj
    img = torch.randn(4, 3, 65, 65, device=dev, requires_grad=True)
    T.project_transport_features(img, max_spatial=8).sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()

    y_a, atoms = _make_exact_atoms(k=8, n=32, d=4, seed=19, device=dev)
    w_a = torch.ones(8, device=dev) / 8.0
    tk1, nn1, nal, _ = T._update_atomic_ledger(y_a + 0.1, torch.arange(32, device=dev), atoms, w_a, 32, None, allow_update=True)
    tk2, nn2, nal2, _ = T._update_atomic_ledger(y_a + 0.05, torch.arange(32, device=dev), atoms, w_a, 32, nal, allow_update=True)
    assert tk1.device.type == "cuda" and tk2.device.type == "cuda" and nn1.device.type == "cuda" and nn2.device.type == "cuda"
    assert nal2["row_chart"].device.type == "cpu" and int((nal2["row_chart"] >= 0).sum().item()) == 32

    z_b = torch.randn(16, 3, device=dev)
    rows_b = torch.arange(16, device=dev).repeat(2)
    rec_b = _StubAxiomRecord(_active_prior_table=z_b, _active_batch_rows=rows_b)
    xf_b = (y_a.clone() + 0.3).requires_grad_(True)
    zr_b = torch.randn(32, 3, device=dev, requires_grad=True)
    jf_b, jr_b = torch.cat([xf_b, z_b[rows_b]], 1), torch.cat([y_a, zr_b], 1)
    lj_b = loss_fn.joint_relativistic_loss(jf_b, jr_b, jf_b.sum(1) * 0.1, jr_b.sum(1) * 0.1, record=rec_b)
    lj_b.backward()
    assert torch.isfinite(lj_b) and xf_b.grad is not None and zr_b.grad is not None

    rec_c = _StubAxiomRecord()
    for _ in range(3):
        c = torch.randn(32, 2, device=dev)
        yr = c * 1.2 + 0.2 * torch.randn(32, 2, device=dev)
        xf = (c * 0.9 + 0.3 * torch.randn(32, 2, device=dev)).requires_grad_(True)
        lc = loss_fn.conditional_fiber_loss(xf, yr, c, cond_disc=torch.zeros(32, 1, dtype=torch.long, device=dev), record=rec_c)
        lc.backward()
        assert torch.isfinite(lc) and xf.grad is not None and torch.isfinite(xf.grad).all()


def test_v2_weibull_1d_unscaled_log_odds_step_scale_and_simplicial_cooling():
    """Axiom-v2 transport regressions: 1D Weibull akt_exp, unscaled _log_odds, het_ema, step_scale & thermal cooling."""
    g = torch.Generator().manual_seed(91)
    y1d = torch.randn(128, 1, generator=g)
    ell_1d = float(T._batch_ell(y1d, 0.25).item())
    # 1D Weibull akt_exp = 0.5 lifts ell well above raw 1D quantile (~0.01 -> ~0.3+)
    assert ell_1d > 0.15, f"expected 1D Weibull akt_exp lift, got {ell_1d}"

    # Unscaled _log_odds must stay linear (not tanh-saturated) on O(1) logits regardless of small ell:
    df = torch.tensor([1.0, -1.0, 0.5, -0.5])
    dr = torch.tensor([-0.5, 0.5, -1.0, 1.0])
    dft, drt = T.AxiomTransportLoss._log_odds(df, dr)
    assert float((dft - df).abs().max().item()) < 0.01

    # Conditional fiber loss updates step_scale smoothly in [0.15, 1.0]:
    loss_fn = T.AxiomTransportLoss()
    rec = _StubAxiomRecord()
    rec.step_scale = 1.0
    for _ in range(3):
        c = torch.randn(32, 2, generator=g)
        yr = c * 1.1 + 0.15 * torch.randn(32, 2, generator=g)
        xf = yr.clone().requires_grad_(True)
        lc = loss_fn.conditional_fiber_loss(xf, yr, c, cond_disc=torch.zeros(32, 1, dtype=torch.long), record=rec)
        lc.backward()
    assert rec.cond_initialized is True and 0.15 <= rec.step_scale <= 1.0

    # Simplicial manifold loss must scale gradients by record.thermal_scale when < 1.0:
    from types import SimpleNamespace
    z_tab = torch.randn(16, 2, generator=g, requires_grad=True)
    gen = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
    y_arc = _make_arc_batch(32)
    loss_fn.simplicial_manifold_loss(z_tab, gen, y_arc, record=SimpleNamespace(thermal_scale=1.0, chord_phase=0.0)).backward()
    g_hot = z_tab.grad.detach().clone()
    z_tab.grad.zero_()
    loss_fn.simplicial_manifold_loss(z_tab, gen, y_arc, record=SimpleNamespace(thermal_scale=0.1, chord_phase=0.0)).backward()
    g_cool = z_tab.grad.detach().clone()
    assert torch.allclose(g_cool, 0.1 * g_hot, rtol=1e-4, atol=1e-5)

    # Atomic ledger finite row_dist sentinel (checkpoint-safe, zero Inf):
    y_a, atoms = _make_exact_atoms(k=4, n=16, d=4, seed=5)
    _, _, nal, _ = T._update_atomic_ledger(y_a + 1.0, torch.arange(8).repeat(2), atoms, torch.ones(4) / 4.0, 16, None, allow_update=True)
    assert "row_dist" in nal and bool(torch.isfinite(nal["row_dist"]).all())


def test_v3_conformal_horizon_deconflicted_charts_and_atomic_absent_shield():
    """Axiom-v3 transport regressions: conformal width gate, shock charts, deconflicted routing, atomic absent_shield."""
    g = torch.Generator().manual_seed(131)
    # 1. Conformal local horizons: active on 8x width-ratio mixture, silent on uniform-width mixture.
    y_unif = _make_disconnected_modes(64, sigma=0.25)
    ell_u = T._batch_ell(y_unif, 0.25)
    exy_u, exx_u, ex_u, ey_u, dyy_u = T._conformal_horizon(y_unif, y_unif, ell_u, 0.0, False, return_aux=True)
    assert exy_u is ell_u and exx_u is ell_u and ex_u is None and ey_u is None and dyy_u is not None

    y_mix = torch.cat([torch.tensor([-15.0, 0.0]) + 0.15 * torch.randn(32, 2, generator=g), torch.tensor([15.0, 0.0]) + 1.50 * torch.randn(32, 2, generator=g)], 0)
    ell_m = T._batch_ell(y_mix, 0.25)
    exy_m, exx_m, ex_m, ey_m, _ = T._conformal_horizon(y_mix, y_mix, ell_m, 0.0, False, return_aux=True)
    assert ex_m is not None and ey_m is not None and torch.allclose(exy_m, exx_m, atol=1e-6)
    assert float(ey_m[:32].mean().item()) < 0.6 * float(ey_m[32:].mean().item())

    # 2. Shock-accelerated _update_charts tracks fake_mass inside chart_state (zero new top-level keys).
    y_c0 = _make_disconnected_modes(48, sigma=0.10)
    c0, w0, r0, b0, cs0 = T._update_charts(y_c0, None, max_charts=16, allow_update=True)
    assert "fake_mass" in cs0 and cs0["fake_mass"].shape == w0.shape and bool(torch.isfinite(cs0["fake_mass"]).all())
    assert int(b0.sum().item()) == 48
    # Stationary update vs large jump shock:
    c_stat, _, _, _, cs_stat = T._update_charts(y_c0, cs0, max_charts=16, allow_update=True)
    c_jump, w_jump, _, _, cs_jump = T._update_charts(y_c0 + 25.0, cs0, max_charts=4, allow_update=True)
    assert float(((w_jump[:, None] * c_jump).sum(0) - (w0[:, None] * c0).sum(0)).norm().item()) > 15.0, "shock-accelerated charts failed to track jump"
    assert float((c_jump - c0).norm(dim=1).amax().item()) > 20.0, "nearest chart failed to jump under shock"
    assert bool(torch.isfinite(cs_stat["fake_mass"]).all()) and bool(torch.isfinite(cs_jump["fake_mass"]).all())

    # 3. _deconflicted_routing: absent-chart shield + surplus->deficit rebalance + allow_update=False guard.
    k_c = 6
    ang = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / k_c), k_c)
    cf = torch.stack((10.0 * torch.cos(ang), 10.0 * torch.sin(ang)), dim=1)
    wr = torch.full((k_c,), 1.0 / k_c)
    rads = torch.full((k_c,), 0.25)
    bcnt = torch.tensor([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # chart 0 absent from batch
    xf_bal = cf.repeat_interleave(8, dim=0)  # 48 particles, 8 seated at each chart
    _, _, rw_bal, ab_sh, fm_no = T._deconflicted_routing(xf_bal, cf, wr, rads, bcnt, wr.clone(), torch.tensor(0.25), wv_c=1.0, halo_mult=2.1, s_th=1.0, allow_update=False)
    assert torch.allclose(fm_no, wr), "allow_update=False mutated fake_mass"
    assert float(ab_sh[:8].mean().item()) > 0.9 and float(ab_sh[8:].amax().item()) == 0.0
    assert float(rw_bal.amax().item()) == 0.0, "balanced settled particles should have zero routing pull"
    # Surplus on chart 1, deficit on chart 0 -> triggers rebalance pull on chart-1 particles and moves fake_mass EMA.
    cur_fm_imb = torch.tensor([0.0, 0.45, 0.15, 0.15, 0.15, 0.10])
    xf_imb = cf[torch.tensor([1] * 24 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6)]
    _, _, rw_imb, _, fm_upd = T._deconflicted_routing(xf_imb, cf, wr, rads, torch.full((k_c,), 8.0), cur_fm_imb, torch.tensor(0.25), wv_c=1.0, halo_mult=2.1, s_th=1.0, allow_update=True)
    assert float(rw_imb[:24].amax().item()) > 0.1 and float(fm_upd[1].item()) > float(cur_fm_imb[1].item())

    # 4. forward() atomic absent_shield & thermal shock_mu_ratio=0.35.
    loss_fn = T.AxiomTransportLoss()
    y_atoms, atoms8 = _make_exact_atoms(k=8, n=32, d=4, seed=71)
    # Remove atom 0 from current minibatch y_real while keeping all 8 atoms in chart_state:
    y_no0 = atoms8[1:][torch.arange(32) % 7]
    z_tab = torch.randn(16, 3, generator=g)
    rows = torch.arange(16).repeat(2)
    rec = _StubAxiomRecord(_active_prior_table=z_tab, _active_batch_rows=rows)
    # Seed all 8 atoms into chart_state first:
    _ = loss_fn(atoms8.repeat(4, 1).requires_grad_(True), atoms8.repeat(4, 1), torch.zeros(32, requires_grad=True), torch.zeros(32), record=rec)
    assert rec.chart_state is not None and rec.atomic_ledger is not None
    assert rec.last_shock_mu_ratio == pytest.approx(0.35)
    xf_a = (atoms8.repeat(4, 1) + 0.05).requires_grad_(True)
    df_a = (xf_a.sum(1) * 0.1).requires_grad_(True)
    l_a = loss_fn(xf_a, y_no0, df_a, y_no0.sum(1) * 0.1, record=rec)
    l_a.backward()
    assert torch.isfinite(l_a) and xf_a.grad is not None and torch.isfinite(xf_a.grad).all()


def test_v4_fast_spectral_word_bfs_cpu_guard_and_tiny_scale():
    """Axiom-v4 transport regressions: analytic 2x2 / Pade-11 Halley / coupled NS, grad_fn BFS, CPU guard, 1e-5x scale."""
    g = torch.Generator().manual_seed(404)

    # 1. Degenerate & general 2x2 analytic zero_eigh + Pade-(1,1) Halley sym_eigensplit for d=8,16,32.
    m2_deg = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    v2, u2 = T.zero_eigh_symmetric(m2_deg)
    assert torch.allclose(v2, torch.tensor([3.0, 3.0], dtype=torch.float64))
    assert torch.allclose(u2.T @ u2, torch.eye(2, dtype=torch.float64), atol=1e-12)
    for d in (2, 4, 8, 16, 32):
        a = torch.randn(4, d, d, dtype=torch.float64, generator=g)
        sym = 0.5 * (a + a.transpose(-1, -2))
        ep, en = T.sym_eigensplit_zero_eigh(sym)
        assert torch.allclose(ep + en, sym, atol=1e-12, rtol=1e-12)
        ev_ref, ec_ref = torch.linalg.eigh(sym)
        ep_ref = (ec_ref * ev_ref.clamp_min(0.0).unsqueeze(-2)) @ ec_ref.transpose(-1, -2)
        assert float((ep - ep_ref).abs().max().item()) < 5e-4, f"d={d} Pade-11 eigensplit drift"

    # 2. 16-step coupled NS _sym_inv_sqrt across d=1..32 (incl. rank-deficient & zero cov).
    for d in (1, 2, 4, 8, 16, 32):
        x = torch.randn(48, d, dtype=torch.float64, generator=g)
        if d >= 2:
            x[:, -1] *= 1e-4
        c = (x.T @ x) / 48.0
        w_ns = T._sym_inv_sqrt(c)
        ev, ec = torch.linalg.eigh(0.5 * (c + c.T))
        eps = max(1e-4 * max(float(ev.clamp_min(0.0).mean().item()), 1e-4), 1e-6)
        w_ref = (ec * ((ev + eps) if d > 2 else ev.clamp_min(eps)).rsqrt().unsqueeze(0)) @ ec.T
        assert float((w_ns - w_ref).norm().item() / w_ref.norm().item()) < 1e-5, f"d={d} _sym_inv_sqrt drift"
    w_zero = T._sym_inv_sqrt(torch.zeros(8, 8, dtype=torch.float64))
    assert torch.allclose(w_zero, torch.eye(8, dtype=torch.float64) * 1e3, rtol=1e-5)

    # 3. _curve_gate min(D, 9) Gram handles D=1, 2, 3, 12 cleanly.
    assert T._curve_gate(torch.linspace(-1.0, 1.0, 24).unsqueeze(1), 1.0) == 1.0
    assert T._curve_gate(_make_arc_batch(32), 1.0) > 0.8
    assert T._curve_gate(_make_saddle_batch(32), 1.0) < 0.2
    assert 0.0 <= T._curve_gate(torch.randn(32, 12, generator=g), 1.0) <= 1.0

    # 4. WordFixture <=8-hop grad_fn BFS recovers (ptab, active_rows) when rec._active_prior_table is None.
    loss_fn = T.AxiomTransportLoss()
    y_atoms, _ = _make_exact_atoms(k=8, n=32, d=4, seed=405)
    ptab = nn.Parameter(torch.randn(16, 3, generator=g))
    rows = torch.arange(16).repeat(2)
    ptab._axiom_active_rows = rows.detach()
    gen_w = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 4))
    rec_bfs = _StubAxiomRecord()
    assert rec_bfs._active_prior_table is None
    z_s = ptab[rows]
    xf = gen_w(z_s)
    l_f = loss_fn(xf, y_atoms, (xf.sum(1) * 0.1), y_atoms.sum(1) * 0.1, record=rec_bfs)
    l_f.backward()
    assert torch.isfinite(l_f) and rec_bfs.atomic_ledger is not None and getattr(ptab, "_axiom_record", None) is rec_bfs

    # Also test joint_relativistic_loss BFS recovery:
    rec_jbfs = _StubAxiomRecord()
    ptab2 = nn.Parameter(torch.randn(16, 3, generator=g))
    ptab2._axiom_active_rows = rows.detach()
    z_s2 = ptab2[rows]
    jf = torch.cat((gen_w(z_s2), z_s2), dim=1)
    jr = torch.cat((y_atoms, torch.randn(32, 3, generator=g, requires_grad=True)), dim=1)
    l_j = loss_fn.joint_relativistic_loss(jf, jr, jf.sum(1) * 0.1, jr.sum(1) * 0.1, record=rec_jbfs)
    l_j.backward()
    assert torch.isfinite(l_j) and rec_jbfs.atomic_ledger is not None and getattr(ptab2, "_axiom_record", None) is rec_jbfs

    # 5. AxiomCondFiberBank CPU device guard even under non-CPU default device & 1e-5x scale-free transport.
    bank = T.AxiomCondFiberBank()
    assert bank._ridge(2).device.type == "cpu"
    y_tiny = _make_disconnected_modes(32) * 1e-5
    ell_tiny = T._batch_ell(y_tiny, 0.25)
    assert 1e-7 < float(ell_tiny.item()) < 1e-4, f"1e-5x _batch_ell distorted: {ell_tiny.item()}"
    c_t, ch_t = T._bi_cost((y_tiny + 1e-5).requires_grad_(True), y_tiny, ell_tiny)
    assert torch.isfinite(c_t).all() and torch.isfinite(ch_t).all() and float(c_t.amax().item()) > 1e-7
    _, _, rads_t, _, cs_t = T._update_charts(y_tiny, None, max_charts=8, allow_update=True)
    assert float(rads_t.amax().item()) < 1e-4 and cs_t["data_scale"] < 1e-3
    # _sc_safe restores exact 1.0 normal-scale floor (>=1e-2) while staying scale-free below 1e-2.
    assert float(T._sc_safe(torch.tensor(0.5)).item()) == 1.0
    assert float(T._sc_safe(torch.tensor(2.5)).item()) == 2.5
    assert float(T._sc_safe(torch.tensor(1e-5)).item()) == pytest.approx(1e-5)
    assert float(T._sc_safe(torch.tensor(0.0)).item()) == pytest.approx(1e-12)
