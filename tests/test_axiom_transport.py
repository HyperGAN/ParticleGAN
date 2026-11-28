"""Comprehensive CPU+CUDA tests for Axiom v6 L2 ``transport.py`` and ``loss.py``."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Tuple

import pytest
import torch
from torch import nn

from particlegan._axiom import atlas as A
from particlegan._axiom import loss as L
from particlegan._axiom import state as S
from particlegan._axiom import transport as T
from particlegan._axiom.math_ops import born_infeld_cost


def _cap(
    logits: torch.Tensor,
    x_in: torch.Tensor,
    record: S.AxiomStepRecord,
    *,
    has_cont_cond: bool = False,
    parts: tuple[torch.Tensor, ...] | None = None,
) -> None:
    S.record_axiom_capture(
        logits,
        S.CaptureEntry(
            x_in=x_in,
            cond=None,
            record=record,
            has_cont_cond=has_cont_cond,
            parts=parts,
        ),
    )


def _seed_exact_atlas(rec: S.AxiomStepRecord, atoms: torch.Tensor, *, max_prototypes: int = 256) -> None:
    """Seed schema-2 PrototypeAtlas on exact atoms so ``all_iso`` is True immediately."""
    k, d = int(atoms.shape[0]), int(atoms.shape[1])
    dev = atoms.device
    at = A.PrototypeAtlas(max_prototypes=max_prototypes, device=dev)
    at.initialized, at.dim = True, d
    at.proto_c = atoms.detach().float().clone()
    at.proto_cov = A.cov_from_scalar(torch.full((k,), 1e-10, device=dev), d)
    at.proto_w = torch.full((k,), 1.0 / float(k), device=dev)
    at.w_fake_ema = at.proto_w.clone()
    at.proto_n = torch.full((k,), 16.0, device=dev)
    at.psi = torch.zeros(k, device=dev)
    at.v_push = torch.full((k,), 1e-10, device=dev)
    at._rho_atom_f, at._g_conn_f = 1e-4, 0.0
    rec.chart_state = at.state_dict()


ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_PATH = ROOT / "particlegan" / "_axiom" / "transport.py"
LOSS_PATH = ROOT / "particlegan" / "_axiom" / "loss.py"


def _make_arc_batch(n: int = 64, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(11)
    t = torch.rand(n, generator=g) * math.pi
    return torch.stack((torch.cos(t), torch.sin(t)), dim=1).to(device)


def _make_saddle_batch(n: int = 64, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(17)
    uv = torch.rand(n, 2, generator=g) * 2.0 - 1.0
    return torch.stack((uv[:, 0], uv[:, 1], uv[:, 0] * uv[:, 1]), dim=1).to(device)


def _make_disconnected_modes(
    n: int = 64, *, sigma: float = 0.12, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(23)
    centers = torch.tensor([[-6.0, -6.0], [-6.0, 6.0], [6.0, -6.0], [6.0, 6.0]])
    idx = torch.randint(0, 4, (n,), generator=g)
    return (centers[idx] + sigma * torch.randn(n, 2, generator=g)).to(device)


def _make_exact_atoms(
    k: int = 8,
    n: int = 64,
    d: int = 4,
    *,
    seed: int = 31,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    if k == 256 and d == 8:
        atoms = ((torch.arange(256).unsqueeze(1) >> torch.arange(8)) & 1).float() * 2.0 - 1.0
    else:
        atoms = (torch.randint(0, 2, (k, d), generator=g).float() * 2.0 - 1.0) * 3.0
        atoms = atoms + torch.arange(k).float().unsqueeze(1) * 0.35
    idx = torch.randint(0, k, (n,), generator=g)
    return atoms[idx].to(device), atoms.to(device)


def _make_highd_mog(n: int = 64, d: int = 32, k: int = 8, sigma: float = 0.12, seed: int = 41) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.randn(k, d, generator=g) * 4.0
    idx = torch.randint(0, k, (n,), generator=g)
    return centers[idx] + sigma * torch.randn(n, d, generator=g)


def test_transport_and_loss_ast_hygiene_and_forbidden_ops(
    monkeypatch: pytest.MonkeyPatch,
):
    allowed_transport_rel = {"math_ops", "state", "atlas", "fiber"}
    allowed_loss_rel = {"state", "transport", "gan_loss"}
    forbidden_attrs = {
        "eigh",
        "eigvalsh",
        "svd",
        "svdvals",
        "pinv",
        "compile",
        "matrix_norm",
        "observed_steps",
        "calls",
        "total_steps",
        "max_steps",
        "_spatial_cluster_ratio",
        "_skip_spatial_transport",
        "grad_fn",
        "next_functions",
        "_axiom_active_rows",
        "_axiom_logit_capture",
    }
    rng_attrs = {"rand", "randn", "randint", "randperm", "normal", "uniform"}

    for path, allowed_rel in (
        (TRANSPORT_PATH, allowed_transport_rel),
        (LOSS_PATH, allowed_loss_rel),
    ):
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        assert all(";" not in ln for ln in lines), f"semicolon line in {path.name}"
        assert max(len(ln) for ln in lines) <= 120, f"line >120 chars in {path.name}"
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "random", f"random imported in {path.name}"
                    assert "_axiom" not in alias.name, f"non-relative _axiom import in {path.name}"
            elif isinstance(node, ast.ImportFrom) and node.level > 0:
                mod = (node.module or "").split(".")[-1]
                assert mod in allowed_rel, f"disallowed relative import {node.module} in {path.name}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs, f"forbidden .{node.attr} in {path.name}:L{node.lineno}"
                assert node.attr not in rng_attrs, f"RNG .{node.attr} in {path.name}:L{node.lineno}"

    def _forbid(*_a, **_kw):
        raise AssertionError("Forbidden spectral/autograd op invoked!")

    for mod, attr in (
        (torch.linalg, "eigh"),
        (torch.linalg, "eigvalsh"),
        (torch.linalg, "svd"),
        (torch.linalg, "svdvals"),
        (torch.linalg, "pinv"),
        (torch.linalg, "matrix_norm"),
        (torch, "svd"),
        (torch.autograd, "grad"),
    ):
        monkeypatch.setattr(mod, attr, _forbid)

    rec = S.AxiomStepRecord()
    loss_fn = T.AxiomTransportLoss()
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

    rec_c = S.AxiomStepRecord()
    g = torch.Generator().manual_seed(7)
    for _ in range(3):
        c = torch.randn(32, 2, generator=g)
        yr = c * 1.5 + 0.3 * torch.randn(32, 2, generator=g)
        xf = (c * 1.2 + 0.4 * torch.randn(32, 2, generator=g)).requires_grad_(True)
        lc = loss_fn.conditional_fiber_loss(xf, yr, c, cond_disc=torch.zeros(32, 1, dtype=torch.long), record=rec_c)
        lc.backward()
        assert torch.isfinite(lc) and xf.grad is not None


def test_project_features_energy_preservation_and_gradients():
    assert T.project_transport_features is T.project_features
    vec = torch.randn(8, 12, requires_grad=True)
    assert torch.equal(T.project_features(vec, max_spatial=8), vec)

    img8 = torch.randn(4, 3, 8, 8, requires_grad=True)
    p8 = T.project_features(img8, max_spatial=8)
    assert p8.shape == (4, 3 * 64)
    assert torch.equal(p8, img8.reshape(4, -1))

    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        img65 = torch.ones(4, 3, 65, 65, requires_grad=True)
        proj = T.project_features(img65, max_spatial=8)
        assert proj.shape == (4, 3 * 64)
        expected_norm = math.sqrt(3.0 * 65.0 * 65.0)
        assert abs(float(proj[0].norm().item()) - expected_norm) < 1e-3
        proj.sum().backward()
        assert img65.grad is not None and torch.isfinite(img65.grad).all()
        assert float(img65.grad.abs().min().item()) > 0.0
    finally:
        torch.use_deterministic_algorithms(prev_det)

    # High-D flat > 256: energy-preserving strided projection with live gradients on all coords
    high_d = torch.ones(6, 1024, requires_grad=True)
    ph = T.project_features(high_d, max_flat_dim=256)
    assert ph.shape == (6, 256)
    assert abs(float(ph[0].norm().item()) - math.sqrt(1024.0)) < 1e-3
    ph.sum().backward()
    assert high_d.grad is not None and float(high_d.grad.abs().min().item()) > 0.0


def test_kappa_struct_smooth_transition_and_zero_binary_cliff():
    g = torch.Generator().manual_seed(51)
    # Monotonic smooth sweep from tight clusters to isotropic noise in R^32 (no binary jump!)
    kappas = [
        float(T.compute_contrast_confidence(_make_highd_mog(64, 32, 8, sigma=s, seed=52)).item())
        for s in (0.05, 0.5, 1.5, 2.5, 4.0, 12.0)
    ]
    assert kappas[0] > 0.95 and kappas[-1] < 0.05
    assert any(0.05 < k < 0.95 for k in kappas), f"expected smooth transition, got {kappas}"
    for a, b in zip(kappas, kappas[1:]):
        assert a >= b - 1e-6, f"kappa_struct not monotone: {kappas}"

    # Structured targets of any dim stay near 1.0; isotropic high-D noise drops to ~0.0
    assert float(T.compute_contrast_confidence(_make_disconnected_modes(64)).item()) > 0.95
    assert float(T.compute_contrast_confidence(_make_highd_mog(64, 128, 8, sigma=0.1)).item()) > 0.95
    y_iso32 = torch.randn(64, 32, generator=g)
    assert float(T.compute_contrast_confidence(y_iso32).item()) < 0.05

    # High-D mock noise (256x256x3): fast path returns pure relativistic critic loss
    y_r256 = torch.randn(16, 3, 256, 256, generator=g)
    x_r256 = torch.randn(16, 3, 256, 256, generator=g, requires_grad=True)
    df = torch.randn(16, generator=g, requires_grad=True)
    dr = torch.randn(16, generator=g)
    assert float(T.compute_contrast_confidence(T.project_features(y_r256)).item()) < 1e-4
    fn = T.AxiomTransportLoss()
    rec = S.AxiomStepRecord()
    lv = fn(x_r256, y_r256, df, dr, record=rec)
    lv.backward()
    assert torch.isfinite(lv) and x_r256.grad is None and df.grad is not None
    assert float(df.grad.abs().sum().item()) > 0.0 and rec._disc_gate == 0.0

    # AxiomGANLoss.d_loss smoothly blends relativistic term by (1 - kappa_struct)
    gan_loss = L.AxiomGANLoss(fibre_rel_blend=0.50, record=S.AxiomStepRecord())
    d_r = torch.randn(32, 1, generator=g, requires_grad=True)
    d_f = torch.randn(32, 1, generator=g, requires_grad=True)
    base_d = super(L.AxiomGANLoss, gan_loss).d_loss(d_r, d_f)
    pair_d = L._paired_relativistic_loss(d_r, d_f)
    assert pair_d is not None

    rec_s = S.AxiomStepRecord()
    gan_loss.record = rec_s
    _cap(d_r, _make_disconnected_modes(32), rec_s)
    _cap(d_f, _make_disconnected_modes(32), rec_s)
    d_struct = gan_loss.d_loss(d_r, d_f)
    assert torch.allclose(d_struct, base_d, atol=1e-5)

    rec_n = S.AxiomStepRecord()
    gan_loss.record = rec_n
    _cap(d_r, torch.randn(32, 32, generator=g), rec_n)
    _cap(d_f, torch.randn(32, 32, generator=g), rec_n)
    d_noise = gan_loss.d_loss(d_r, d_f)
    assert torch.allclose(d_noise, 0.5 * base_d + 0.5 * pair_d, atol=1e-4)


def test_scale_equivariance_across_ten_orders_of_magnitude():
    g = torch.Generator().manual_seed(61)
    y_uni = torch.randn(48, 4, generator=g)
    x_uni = torch.randn(48, 4, generator=g) + 0.6
    y_disc = _make_disconnected_modes(64)
    x_disc = y_disc + 0.4 * torch.randn(64, 2, generator=g)

    for y0, x0 in ((y_uni, x_uni), (y_disc, x_disc)):
        ref_loss, ref_gnorm = None, None
        for alpha in (1e-5, 1e-2, 1.0, 1e2, 1e5):
            fn = T.AxiomTransportLoss()
            x = (x0 * alpha).detach().clone().requires_grad_(True)
            y = (y0 * alpha).detach().clone()
            df = torch.zeros(len(x), requires_grad=True)
            dr = torch.zeros(len(y))
            lv = fn(x, y, df, dr, allow_state_update=False)
            lv.backward()
            gn = float((x.grad * alpha).norm().item())
            assert torch.isfinite(lv) and x.grad is not None and torch.isfinite(x.grad).all()
            if ref_loss is None:
                ref_loss, ref_gnorm = float(lv.item()), gn
            else:
                assert math.isclose(float(lv.item()), ref_loss, rel_tol=2e-2, abs_tol=1e-5)
                assert math.isclose(gn, ref_gnorm, rel_tol=2e-2, abs_tol=1e-5)

    # _batch_ell linear scaling + 1D Weibull akt_exp lift + extreme 1e18 overflow safety
    ell_1 = float(T._batch_ell(y_disc, 0.25).item())
    ell_tiny = float(T._batch_ell(y_disc * 1e-5, 0.25).item())
    assert math.isclose(ell_tiny, ell_1 * 1e-5, rel_tol=1e-4)
    ell_1d = float(T._batch_ell(torch.randn(128, 1, generator=g), 0.25).item())
    assert ell_1d > 0.15

    y_huge = y_disc * 1e18
    x_huge = (y_huge + 1e17).requires_grad_(True)
    c_h, ch_h = born_infeld_cost(x_huge, y_huge, torch.tensor(1e18))
    assert torch.isfinite(c_h).all() and torch.isfinite(ch_h).all()
    c_h.mean().backward()
    assert x_huge.grad is not None and torch.isfinite(x_huge.grad).all()


def test_zero_feydy_self_coupling_bias_and_conformal_horizon():
    g = torch.Generator().manual_seed(71)
    y_gauss = torch.randn(48, 2, generator=g)
    y_mix = torch.cat(
        (
            torch.tensor([-14.0, 0.0]) + 0.15 * torch.randn(24, 2, generator=g),
            torch.tensor([14.0, 0.0]) + 1.60 * torch.randn(24, 2, generator=g),
        ),
        dim=0,
    )

    # Exact zero Feydy self-coupling bias at Nash equilibrium (X == Y, D == const)
    for y0 in (y_gauss, y_mix):
        fn = T.AxiomTransportLoss(chart_weight=0.0, iters=24)
        x = y0.clone().requires_grad_(True)
        df = torch.full((len(x),), 0.7, requires_grad=True)
        dr = torch.full((len(y0),), 0.7)
        lv = fn(x, y0, df, dr, allow_state_update=False)
        lv.backward()
        assert x.grad is not None and float(x.grad.abs().max().item()) < 5e-6

    # Conformal k=3 horizon: off on connected manifold & uniform mixture, on for 8x width ratio
    y_sad = _make_saddle_batch(64)
    ell_s = torch.tensor(0.25)
    exy_s, exx_s, ex_s, ey_s, _ = T._conformal_horizon(y_sad + 0.1, y_sad, ell_s, 0.9, False)
    assert exy_s is ell_s and exx_s is ell_s and ex_s is None and ey_s is None

    y_unif = _make_disconnected_modes(64, sigma=0.25)
    ell_u = T._batch_ell(y_unif, 0.25)
    exy_u, exx_u, ex_u, ey_u, dyy_u = T._conformal_horizon(y_unif, y_unif, ell_u, 0.0, False)
    assert exy_u is ell_u and exx_u is ell_u and ex_u is None and ey_u is None and dyy_u is not None

    ell_m = T._batch_ell(y_mix, 0.25)
    exy_m, exx_m, ex_m, ey_m, _ = T._conformal_horizon(y_mix, y_mix, ell_m, 0.0, False)
    assert ex_m is not None and ey_m is not None
    assert torch.allclose(exy_m, exx_m, atol=1e-6)
    assert float(ey_m[:24].mean().item()) < 0.65 * float(ey_m[24:].mean().item())


def test_high_dim_k4_orphan_macro_rescue_and_symmetry_breaking():
    g = torch.Generator().manual_seed(81)
    rho = 3.0
    prs = torch.full((48,), 0.5 / 48.0)
    pcs = torch.cat((torch.zeros(24), torch.full((24,), 1.0 / 48.0)))

    # 2D threshold: silent at 2.0*rho, active at 3.5*rho
    y_two = torch.cat(
        (
            0.1 * torch.randn(24, 2, generator=g),
            torch.tensor([6.5, 0.0]) + 0.1 * torch.randn(24, 2, generator=g),
        ),
        dim=0,
    )
    x_stuck = (0.1 * torch.randn(48, 2, generator=g)).requires_grad_(True)
    ch_near = torch.full((48, 48), 0.2)
    ch_near[:, 24:] = 2.0 * rho
    assert float(T._orphan_macro_loss(x_stuck, y_two, ch_near, prs, pcs, rho=rho, g_conn=0.0).item()) == 0.0

    ch_far = ch_near.clone()
    ch_far[:, 24:] = 3.5 * rho
    l_2d = T._orphan_macro_loss(x_stuck, y_two, ch_far, prs, pcs, rho=rho, g_conn=0.0)
    l_2d.backward()
    assert float(l_2d.item()) > 0.05 and x_stuck.grad is not None
    assert float(x_stuck.grad[:, 0].mean().item()) < -1e-4

    # High-D (d=32) orphan rescue pulls stuck fake particles toward unvisited mode at +10*e_0
    e0 = torch.zeros(32)
    e0[0] = 10.0
    y_32 = torch.cat(
        (
            0.1 * torch.randn(24, 32, generator=g),
            e0 + 0.1 * torch.randn(24, 32, generator=g),
        ),
        dim=0,
    )
    x_32 = (0.1 * torch.randn(48, 32, generator=g)).requires_grad_(True)
    l_32 = T._orphan_macro_loss(x_32, y_32, ch_far, prs, pcs, rho=rho, g_conn=0.0)
    l_32.backward()
    assert float(l_32.item()) > 0.05 and x_32.grad is not None
    assert float(x_32.grad[:, 0].mean().item()) < -1e-4

    # Covered broad cluster (r_loc ~ 1): must not false-trigger
    y_cov = torch.randn(48, 2, generator=g)
    x_cov = torch.randn(48, 2, generator=g).requires_grad_(True)
    l_cov = T._orphan_macro_loss(
        x_cov,
        y_cov,
        torch.full((48, 48), 2.5 * rho),
        prs,
        torch.full((48,), 1.0 / 48.0),
        rho=rho,
        g_conn=0.0,
    )
    assert float(l_cov.item()) == 0.0

    # Equidistant symmetric saddle breaks symmetry cleanly toward both -10 and +10
    y_sym = torch.cat((torch.full((16, 2), -10.0), torch.full((16, 2), 10.0)), dim=0)
    y_sym[:, 1] = 0.0
    x_mid = torch.zeros(32, 2, requires_grad=True)
    l_sym = T._orphan_macro_loss(
        x_mid,
        y_sym,
        torch.full((32, 32), 4.0 * rho),
        torch.full((32,), 1.0 / 32.0),
        torch.full((32,), 1.0 / 32.0),
        rho=rho,
        g_conn=0.0,
    )
    l_sym.backward()
    assert x_mid.grad is not None
    assert int((x_mid.grad[:, 0] > 1e-4).sum().item()) >= 12
    assert int((x_mid.grad[:, 0] < -1e-4).sum().item()) >= 12


def test_atomic_gate_unified_atlas_ledger_and_checkpoint_continuation():
    # Annular-gap duplicate purity gate
    y_byte, atoms_byte = _make_exact_atoms(k=256, n=128, d=8, seed=5)
    y_five, _ = _make_exact_atoms(k=5, n=64, d=4, seed=9)
    assert T.compute_discrete_atomic_gate(y_byte) == 1.0
    assert T.compute_discrete_atomic_gate(y_five) == 1.0
    assert T.compute_discrete_atomic_gate(atoms_byte[:32], charts=atoms_byte, radii=torch.full((256,), 1e-6)) == 1.0
    assert T.compute_discrete_atomic_gate(torch.full((32, 8), 0.7)) == 0.0
    assert T.compute_discrete_atomic_gate(_make_arc_batch(64)) == 0.0
    assert T.compute_discrete_atomic_gate(_make_saddle_batch(64)) == 0.0
    for sig in (1e-3, 1e-2, 1e-1):
        assert T.compute_discrete_atomic_gate(_make_disconnected_modes(64, sigma=sig)) == 0.0

    # Unified PrototypeAtlas + HamiltonCommitmentLedger routing + schema-2 checkpoint continuation
    g = torch.Generator().manual_seed(91)
    y_atoms, atoms8 = _make_exact_atoms(k=8, n=32, d=4, seed=71)
    z_tab = torch.randn(16, 3, generator=g)
    rows = torch.arange(16).repeat(2)
    rec_a = S.AxiomStepRecord()
    _seed_exact_atlas(rec_a, atoms8)
    fn_a = T.AxiomTransportLoss()
    for step in range(2):
        T.prepare_axiom_transport_step(
            fn_a,
            rec_a,
            atoms8.repeat(4, 1),
            prior_table=z_tab,
            batch_rows=rows,
            is_discrete_prior=True,
        )
        xf = (atoms8.repeat(4, 1) + (0.25 - 0.05 * step)).requires_grad_(True)
        df = (xf.sum(dim=1) * 0.1).requires_grad_(True)
        lv = fn_a(
            xf,
            atoms8.repeat(4, 1),
            df,
            atoms8.repeat(4, 1).sum(dim=1) * 0.1,
            record=rec_a,
        )
        lv.backward()
        assert torch.isfinite(lv) and xf.grad is not None and torch.isfinite(xf.grad).all()
    assert rec_a.chart_state is not None and rec_a.chart_state["schema"] == 2
    assert rec_a.atomic_ledger is not None and rec_a.atomic_ledger["schema"] == 2

    # Absent-atom shield: remove atom 0 from y_real while all 8 atoms persist in atlas
    y_no0 = atoms8[1:][torch.arange(32) % 7]
    sd = rec_a.state_dict()
    rec_b = S.AxiomStepRecord()
    rec_b.load_state_dict(sd)
    fn_b = T.AxiomTransportLoss()

    xf_a = (atoms8.repeat(4, 1) + 0.08).requires_grad_(True)
    xf_b = xf_a.detach().clone().requires_grad_(True)
    df_a = (xf_a.sum(dim=1) * 0.1).requires_grad_(True)
    df_b = (xf_b.sum(dim=1) * 0.1).requires_grad_(True)
    T.prepare_axiom_transport_step(fn_a, rec_a, y_no0, prior_table=z_tab, batch_rows=rows, is_discrete_prior=True)
    T.prepare_axiom_transport_step(fn_b, rec_b, y_no0, prior_table=z_tab, batch_rows=rows, is_discrete_prior=True)
    la = fn_a(xf_a, y_no0, df_a, y_no0.sum(dim=1) * 0.1, record=rec_a)
    lb = fn_b(xf_b, y_no0, df_b, y_no0.sum(dim=1) * 0.1, record=rec_b)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb) and torch.equal(xf_a.grad, xf_b.grad)


def test_bigan_joint_relativistic_and_codebook_bijection_contraction():
    g = torch.Generator().manual_seed(101)
    loss_fn = T.AxiomTransportLoss()
    y_atoms, atoms8 = _make_exact_atoms(k=8, n=32, d=4, seed=55)
    z_table = torch.randn(16, 3, generator=g)
    rows = torch.arange(16).repeat(2)

    for z_first in (False, True):
        rec = S.AxiomStepRecord()
        _seed_exact_atlas(rec, atoms8)
        T.prepare_axiom_transport_step(
            loss_fn,
            rec,
            y_atoms,
            prior_table=z_table,
            batch_rows=rows,
            is_discrete_prior=True,
        )
        x_f = (y_atoms.clone() + 0.4).requires_grad_(True)
        z_r = torch.randn(32, 3, generator=g, requires_grad=True)
        z_f = z_table[rows].detach()
        jf = torch.cat((z_f, x_f), dim=1) if z_first else torch.cat((x_f, z_f), dim=1)
        jr = torch.cat((z_r, y_atoms), dim=1) if z_first else torch.cat((y_atoms, z_r), dim=1)
        df = (jf.sum(dim=1) * 0.1).requires_grad_(True)
        dr = (jr.sum(dim=1) * 0.1).requires_grad_(True)
        l_j = loss_fn.joint_relativistic_loss(jf, jr, df, dr, record=rec)
        l_j.backward()
        assert torch.isfinite(l_j)
        assert x_f.grad is not None and float(x_f.grad.abs().sum().item()) > 0.0
        assert z_r.grad is not None and float(z_r.grad.abs().sum().item()) > 0.0
        assert rec.chart_state is not None and rec.atomic_ledger is not None
        assert rec.thermal_initialized

    # Out-of-bounds batch_rows guard: returns cleanly without IndexError
    rec_oob = S.AxiomStepRecord()
    rec_oob._active_prior_table = z_table
    rec_oob._active_batch_rows = torch.full((32,), 999, dtype=torch.long)
    jf_oob = torch.cat((y_atoms.clone(), z_table[rows]), dim=1).requires_grad_(True)
    jr_oob = torch.cat((y_atoms.clone(), torch.randn(32, 3, generator=g)), dim=1).requires_grad_(True)
    l_oob = loss_fn.joint_relativistic_loss(
        jf_oob, jr_oob, (jf_oob.sum(1) * 0.1), (jr_oob.sum(1) * 0.1), record=rec_oob
    )
    assert torch.isfinite(l_oob)


def test_simplicial_manifold_loss_gating_and_thermal_cooling():
    g = torch.Generator().manual_seed(111)
    loss_fn = T.AxiomTransportLoss()

    # Curve gate distinguishes 1D curves from 2D saddles & handles D=1 and D=12
    assert T._curve_gate(torch.linspace(-1.0, 1.0, 24).unsqueeze(1), 1.0) == 1.0
    assert T._curve_gate(_make_arc_batch(32), 1.0) > 0.8
    assert T._curve_gate(_make_saddle_batch(32), 1.0) < 0.2
    assert 0.0 <= T._curve_gate(torch.randn(32, 12, generator=g), 1.0) <= 1.0

    # Smooth g_conn disengagement on discrete atoms vs active on continuous manifolds
    y_byte, _ = _make_exact_atoms(k=256, n=128, d=8, seed=112)
    g_conn_atom, g_mac_atom, _, _, _ = T._simplicial_gates(y_byte, 4.0)
    assert g_conn_atom < 1e-4 and g_mac_atom < 1e-4
    rec_prep = S.AxiomStepRecord()
    assert not T.prepare_axiom_transport_step(
        loss_fn,
        rec_prep,
        y_byte,
        is_discrete_prior=True,
        use_simplex=True,
        simplex_chart_weight=4.0,
    )
    assert T.prepare_axiom_transport_step(
        loss_fn,
        rec_prep,
        _make_arc_batch(48),
        is_discrete_prior=True,
        use_simplex=True,
        simplex_chart_weight=4.0,
    )

    # 1D arc & 2D curved saddle produce live gradients on both z_table and generator
    for y_data, out_dim in ((_make_arc_batch(64), 2), (_make_saddle_batch(64), 3)):
        rec_s = S.AxiomStepRecord()
        rec_s.chord_phase = 0.37
        z_tab = (torch.randn(24, 2, generator=g) * 0.5).requires_grad_(True)
        gen = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, out_dim))
        l_s = loss_fn.simplicial_manifold_loss(z_tab, gen, y_data, record=rec_s)
        l_s.backward()
        assert torch.isfinite(l_s)
        assert z_tab.grad is not None and float(z_tab.grad.abs().sum().item()) > 0.0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen.parameters())

    # Thermal scale cooling scales simplicial gradients exactly
    z_tab = torch.randn(16, 2, generator=g, requires_grad=True)
    gen = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
    y_arc = _make_arc_batch(32)
    loss_fn.simplicial_manifold_loss(
        z_tab, gen, y_arc, record=SimpleNamespace(thermal_scale=1.0, chord_phase=0.0)
    ).backward()
    g_hot = z_tab.grad.detach().clone()
    z_tab.grad.zero_()
    loss_fn.simplicial_manifold_loss(
        z_tab, gen, y_arc, record=SimpleNamespace(thermal_scale=0.1, chord_phase=0.0)
    ).backward()
    assert torch.allclose(z_tab.grad, 0.1 * g_hot, rtol=1e-4, atol=1e-5)


def test_axiom_gan_loss_d_g_conditional_detach_capture_and_controls():
    g = torch.Generator().manual_seed(121)
    # Constructor validation
    with pytest.raises(ValueError):
        L.AxiomGANLoss(weight=-0.1)
    with pytest.raises(ValueError):
        L.AxiomGANLoss(cond_ot_weight=-1.0)
    with pytest.raises(ValueError):
        L.AxiomGANLoss(fibre_rel_blend=1.5)

    # Unconditional d_loss + g_loss including .detach() capture resolution (zero logit attrs)
    rec = S.AxiomStepRecord()
    gan_loss = L.AxiomGANLoss(record=rec)
    assert gan_loss.transport.bank_mean_blend == pytest.approx(0.85)
    y_real = _make_disconnected_modes(32)
    x_near = (y_real.clone() + 0.15).requires_grad_(True)
    x_far = (y_real.clone() + 4.0).requires_grad_(True)

    d_real = torch.zeros(32, 1)
    d_fake_near = torch.zeros(32, 1, requires_grad=True)
    _cap(d_real, y_real, rec)
    _cap(d_fake_near, x_near, rec)
    assert not hasattr(d_fake_near, "_axiom_logit_capture")
    l_near = gan_loss.g_loss(d_fake_near, d_real.detach())

    rec_far = S.AxiomStepRecord()
    gan_far = L.AxiomGANLoss(record=rec_far)
    d_fake_far = torch.zeros(32, 1, requires_grad=True)
    _cap(d_real, y_real, rec_far)
    _cap(d_fake_far, x_far, rec_far)
    l_far = gan_far.g_loss(d_fake_far, d_real.detach())
    l_far.backward()
    # Semantic control: shifting fake samples away strictly raises transport g_loss and pulls back
    assert float(l_far.item()) > float(l_near.item()) + 0.05
    assert x_far.grad is not None and float(x_far.grad.mean().item()) > 0.0

    # Continuous-conditioned DDGAN path via CaptureEntry.parts
    rec_c = S.AxiomStepRecord()
    gan_c = L.AxiomGANLoss(record=rec_c)
    c_ctx = torch.randn(32, 2, generator=g)
    y_c = 1.5 * c_ctx + 0.2 * torch.randn(32, 2, generator=g)
    x_c = (0.8 * c_ctx + 0.5 * torch.randn(32, 2, generator=g)).requires_grad_(True)
    df_c = (x_c.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    dr_c = y_c.sum(dim=1, keepdim=True) * 0.1
    _cap(
        df_c,
        torch.cat((x_c, c_ctx), dim=1),
        rec_c,
        has_cont_cond=True,
        parts=(x_c, c_ctx),
    )
    _cap(
        dr_c,
        torch.cat((y_c, c_ctx), dim=1),
        rec_c,
        has_cont_cond=True,
        parts=(y_c, c_ctx),
    )
    lg_c = gan_c.g_loss(df_c, dr_c.detach())
    lg_c.backward()
    assert torch.isfinite(lg_c) and x_c.grad is not None and float(x_c.grad.abs().sum().item()) > 0.0
    assert rec_c.cond_initialized

    # Missing real_logits raises when weight > 0, succeeds when weight == 0
    with pytest.raises(ValueError):
        gan_loss.g_loss(torch.randn(8, 1, requires_grad=True), None)
    gan_zero = L.AxiomGANLoss(weight=0.0)
    assert torch.isfinite(gan_zero.g_loss(torch.randn(8, 1, requires_grad=True), None))

    # joint_g_loss grad requirement + working captured path
    with pytest.raises(ValueError):
        gan_loss.joint_g_loss(torch.randn(8, 1, requires_grad=True), torch.randn(8, 1))
    rec_j = S.AxiomStepRecord()
    gan_j = L.AxiomGANLoss(record=rec_j)
    jf = torch.randn(16, 4, generator=g, requires_grad=True)
    jr = torch.randn(16, 4, generator=g, requires_grad=True)
    df_j = (jf.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    dr_j = (jr.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    _cap(df_j, jf, rec_j)
    _cap(dr_j, jr, rec_j)
    lj = gan_j.joint_g_loss(df_j, dr_j)
    lj.backward()
    assert torch.isfinite(lj) and jf.grad is not None and jr.grad is not None
    assert float(jf.grad.abs().sum().item()) > 0.0 and float(jr.grad.abs().sum().item()) > 0.0


def test_clock_freedom_no_mutation_guards_and_zero_global_rng():
    g = torch.Generator().manual_seed(131)
    y = _make_disconnected_modes(32)
    x0 = y.clone() + 0.4
    d_r = y.sum(dim=1) * 0.1

    # Strict clock-freedom: shifting calls & observed_steps by +8192 yields bit-identical results
    rec1 = S.AxiomStepRecord()
    rec2 = S.AxiomStepRecord()
    rec1.chord_phase = rec2.chord_phase = 0.25
    rec2.calls = 8192
    rec2.observed_steps = 8192
    x1, x2 = x0.clone().requires_grad_(True), x0.clone().requires_grad_(True)
    l1 = T.AxiomTransportLoss()(x1, y, x1.sum(dim=1) * 0.1, d_r, record=rec1)
    l2 = T.AxiomTransportLoss()(x2, y, x2.sum(dim=1) * 0.1, d_r, record=rec2)
    l1.backward()
    l2.backward()
    assert torch.equal(l1, l2) and torch.equal(x1.grad, x2.grad)
    assert rec1.ell_ema == rec2.ell_ema and rec1.thermal_ema == rec2.thermal_ema

    # Probe / allow_state_update=False / no_grad leave AxiomStepRecord completely untouched
    rec_guard = S.AxiomStepRecord()
    fn = T.AxiomTransportLoss()
    with torch.no_grad():
        fn(x0.clone(), y, x0.sum(dim=1) * 0.1, d_r, record=rec_guard)
    fn(
        x0.clone().requires_grad_(True),
        y,
        x0.sum(dim=1) * 0.1,
        d_r,
        record=rec_guard,
        allow_state_update=False,
    )
    rec_guard._in_probe = True
    fn(x0.clone().requires_grad_(True), y, x0.sum(dim=1) * 0.1, d_r, record=rec_guard)
    rec_guard._in_probe = False
    rec_guard._in_protected_probe = True
    fn(x0.clone().requires_grad_(True), y, x0.sum(dim=1) * 0.1, d_r, record=rec_guard)
    rec_guard._in_protected_probe = False
    assert not rec_guard.ell_initialized and rec_guard.chart_state is None and not rec_guard.thermal_initialized

    # Zero global RNG consumption across all transport + loss operations
    gen_net = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
    z_tab = torch.randn(16, 2, generator=g, requires_grad=True)
    y_arc = _make_arc_batch(32)
    c_ctx = torch.randn(32, 2, generator=g)
    y_c = 1.2 * c_ctx + 0.2 * torch.randn(32, 2, generator=g)
    x_c = (0.9 * c_ctx + 0.3 * torch.randn(32, 2, generator=g)).requires_grad_(True)
    y_atoms, _ = _make_exact_atoms(k=8, n=32, d=4, seed=132)
    z_bi = torch.randn(16, 3, generator=g)
    rows_bi = torch.arange(16).repeat(2)
    xf_bi = (y_atoms + 0.3).requires_grad_(True)
    zr_bi = torch.randn(32, 3, generator=g, requires_grad=True)
    jf_bi = torch.cat((xf_bi, z_bi[rows_bi]), dim=1)
    jr_bi = torch.cat((y_atoms, zr_bi), dim=1)

    py_rng = random.getstate()
    cpu_rng = torch.get_rng_state()

    rec_r = S.AxiomStepRecord()
    gl = L.AxiomGANLoss(record=rec_r)
    xf_u = (y + 0.3).requires_grad_(True)
    df_u = (xf_u.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    dr_u = (y.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    _cap(dr_u, y, rec_r)
    _cap(df_u, xf_u, rec_r)
    gl.d_loss(dr_u, df_u).backward(retain_graph=True)
    gl.g_loss(df_u, dr_u.detach()).backward()
    gl.transport.simplicial_manifold_loss(z_tab, gen_net, y_arc, record=S.AxiomStepRecord()).backward()
    rec_bi = S.AxiomStepRecord()
    T.prepare_axiom_transport_step(
        gl.transport,
        rec_bi,
        y_atoms,
        prior_table=z_bi,
        batch_rows=rows_bi,
        is_discrete_prior=True,
    )
    gl.transport.joint_relativistic_loss(jf_bi, jr_bi, jf_bi.sum(1) * 0.1, jr_bi.sum(1) * 0.1, record=rec_bi).backward()
    gl.transport.conditional_fiber_loss(
        x_c,
        y_c,
        c_ctx,
        cond_disc=torch.zeros(32, 1, dtype=torch.long),
        record=S.AxiomStepRecord(),
    ).backward()

    assert random.getstate() == py_rng
    assert torch.equal(torch.get_rng_state(), cpu_rng)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_axiom_transport_and_loss_cuda():
    dev = torch.device("cuda:0")
    g = torch.Generator(device="cpu").manual_seed(201)
    gen = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 3)).to(dev)
    z_tab = torch.randn(24, 2, generator=g).to(dev).requires_grad_(True)
    y_disc = _make_disconnected_modes(64, device=dev)
    y_sad = _make_saddle_batch(64, device=dev)
    y_a, atoms_cu = _make_exact_atoms(k=8, n=32, d=4, seed=19, device=dev)
    z_b = torch.randn(16, 3, generator=g).to(dev)
    rows_b = torch.arange(16, device=dev).repeat(2)
    img65 = torch.randn(4, 3, 65, 65, generator=g).to(dev).requires_grad_(True)
    y_r256 = torch.randn(16, 3, 256, 256, generator=g).to(dev)
    x_r256 = torch.randn(16, 3, 256, 256, generator=g).to(dev).requires_grad_(True)
    df256 = torch.randn(16, generator=g).to(dev).requires_grad_(True)
    dr256 = torch.randn(16, generator=g).to(dev)
    c_cu = torch.randn(32, 2, generator=g).to(dev)
    yr_cu = c_cu * 1.2 + 0.2 * torch.randn(32, 2, generator=g).to(dev)
    xf_cu = (c_cu * 0.9 + 0.3 * torch.randn(32, 2, generator=g).to(dev)).requires_grad_(True)

    py_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(dev)

    rec = S.AxiomStepRecord()
    gan_loss = L.AxiomGANLoss(record=rec)
    xf = (y_disc.clone() + 0.5).requires_grad_(True)
    df = (xf.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    dr = (y_disc.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    _cap(dr, y_disc, rec)
    _cap(df, xf, rec)
    gan_loss.d_loss(dr, df).backward(retain_graph=True)
    lg = gan_loss.g_loss(df, dr.detach())
    lg.backward()
    assert torch.isfinite(lg) and xf.grad is not None and torch.isfinite(xf.grad).all()
    assert rec.chart_state is not None and rec.chart_state["proto_c"].is_cuda

    # Simplicial manifold loss on CUDA
    ls = gan_loss.transport.simplicial_manifold_loss(z_tab, gen, y_sad, record=S.AxiomStepRecord())
    ls.backward()
    assert torch.isfinite(ls) and z_tab.grad is not None and torch.isfinite(z_tab.grad).all()

    # >8x8 spatial projection + r256 fast path on CUDA
    T.project_features(img65, max_spatial=8).sum().backward()
    assert img65.grad is not None and torch.isfinite(img65.grad).all()
    l256 = gan_loss.transport(x_r256, y_r256, df256, dr256)
    l256.backward()
    assert torch.isfinite(l256) and x_r256.grad is None and df256.grad is not None

    # BiGAN joint_g_loss with zero-CPU-bounce GPU atlas & Hamilton ledger
    rec_b = S.AxiomStepRecord()
    _seed_exact_atlas(rec_b, atoms_cu)
    gan_b = L.AxiomGANLoss(record=rec_b)
    T.prepare_axiom_transport_step(
        gan_b.transport,
        rec_b,
        y_a,
        prior_table=z_b,
        batch_rows=rows_b,
        is_discrete_prior=True,
    )
    xf_b = (y_a.clone() + 0.3).requires_grad_(True)
    zr_b = torch.randn(32, 3, generator=g).to(dev).requires_grad_(True)
    jf_b = torch.cat((xf_b, z_b[rows_b]), dim=1)
    jr_b = torch.cat((y_a, zr_b), dim=1)
    df_b = (jf_b.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    dr_b = (jr_b.sum(dim=1, keepdim=True) * 0.1).requires_grad_(True)
    _cap(df_b, jf_b, rec_b)
    _cap(dr_b, jr_b, rec_b)
    lj = gan_b.joint_g_loss(df_b, dr_b)
    lj.backward()
    assert torch.isfinite(lj) and xf_b.grad is not None and zr_b.grad is not None
    assert rec_b.chart_state is not None and rec_b.chart_state["proto_c"].is_cuda
    assert rec_b.atomic_ledger is not None and rec_b.atomic_ledger["row_comp"].is_cuda

    # Conditional fiber loss on CUDA
    lc = gan_loss.transport.conditional_fiber_loss(
        xf_cu,
        yr_cu,
        c_cu,
        cond_disc=torch.zeros(32, 1, dtype=torch.long, device=dev),
        record=S.AxiomStepRecord(),
    )
    lc.backward()
    assert torch.isfinite(lc) and xf_cu.grad is not None and torch.isfinite(xf_cu.grad).all()

    assert random.getstate() == py_rng
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(dev), cuda_rng)


def test_atomic_transit_shield_and_committed_ledger_kappa_override():
    # Absent-atom transit shield must stay active (== 1.0) for atomic targets even far outside r_eff,
    # and committed-ledger forward must override low contrast confidence kappa (e.g. {-1,+1}^8) to 1.0.
    _, atoms256 = _make_exact_atoms(k=256, n=64, d=8, seed=13)
    assert float(T.compute_contrast_confidence(atoms256[:64]).item()) < 0.6

    rec = S.AxiomStepRecord()
    _seed_exact_atlas(rec, atoms256)
    z_tab = torch.randn(256, 4, generator=torch.Generator().manual_seed(19))
    rows = torch.arange(64)
    y_subset = atoms256[64:128]  # atoms 0..63 are absent from this minibatch
    fn = T.AxiomTransportLoss()
    T.prepare_axiom_transport_step(fn, rec, y_subset, prior_table=z_tab, batch_rows=rows, is_discrete_prior=True)
    # Place fake particles far from their assigned absent atoms (outside r_eff)
    x_far = (atoms256[:64] + 1.5).requires_grad_(True)
    _, absent, _, _, assign = fn._route_charts(x_far, y_subset, x_far, torch.tensor(0.2), 0.0, rec, True)
    assert assign is not None and absent is not None
    # Absent committed rows get full atomic transit shielding (~1.0) despite dist_t >> r_eff
    hits = torch.bincount(T.safe_cdist(y_subset, atoms256).argmin(1), minlength=256)
    absent_mask = hits[assign.batch_comp] <= 0
    assert bool(absent_mask.any())
    assert float(absent[absent_mask].min().item()) > 0.99

    # Committed ledger overrides kappa to 1.0 so route_loss is not crushed by low contrast confidence
    x_far2 = (atoms256[:64] + 1.5).requires_grad_(True)
    df = torch.zeros(64, requires_grad=True)
    dr = torch.zeros(64)
    T.prepare_axiom_transport_step(fn, rec, y_subset, prior_table=z_tab, batch_rows=rows, is_discrete_prior=True)
    lv = fn(x_far2, y_subset, df, dr, record=rec, allow_state_update=False)
    lv.backward()
    assert torch.isfinite(lv) and x_far2.grad is not None and float(x_far2.grad.norm().item()) > 0.05

    # Even when a minibatch hits kappa == 0.0 (med_nn / med_all >= 0.55, e.g. even-parity hypercube
    # slice where min Hamming distance is 2 -> ratio ~0.707), an active atomic_ledger bypasses the
    # early relativistic fast-path return so committed transport still drives x_fake.
    y_hi_ratio = atoms256[((atoms256 > 0).sum(1) % 2) == 0][:64]
    assert float(T.compute_contrast_confidence(y_hi_ratio).item()) <= 1e-4
    x_far3 = (atoms256[:64] + 1.5).requires_grad_(True)
    df3 = torch.zeros(64, requires_grad=True)
    dr3 = torch.zeros(64)
    T.prepare_axiom_transport_step(fn, rec, y_hi_ratio, prior_table=z_tab, batch_rows=rows, is_discrete_prior=True)
    lv3 = fn(x_far3, y_hi_ratio, df3, dr3, record=rec, allow_state_update=False)
    lv3.backward()
    assert torch.isfinite(lv3) and x_far3.grad is not None and float(x_far3.grad.norm().item()) > 0.05


def test_continuous_quota_routing_conservation_zero_halo_pull_and_checkpoint():
    # Exact quota conservation of _gpu_quota_assign across multi-component scores
    g = torch.Generator().manual_seed(29)
    scores = torch.randn(64, 17, generator=g)
    weights = torch.rand(17, generator=g).clamp_min(1e-3)
    quotas = T.hamilton_quotas(64, weights, min_one_when_feasible=False)
    win = T._gpu_quota_assign(scores, quotas)
    assert win.shape == (64,) and torch.equal(torch.bincount(win, minlength=17), quotas)

    # Zero pull inside balanced halos & positive surplus->deficit transfer + exact w_fake_ema continuation
    centers = torch.tensor([[-5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, 5.0]], dtype=torch.float32)
    at = A.PrototypeAtlas(max_prototypes=16)
    at.initialized, at.dim = True, 2
    at.proto_c = centers.clone()
    at.proto_cov = A.cov_from_scalar(torch.full((4,), 0.04), 2)
    at.proto_w = torch.full((4,), 0.25)
    at.w_fake_ema = torch.full((4,), 0.25)
    at.proto_n = torch.full((4,), 32.0)
    at.psi = torch.zeros(4)
    at.v_push = torch.full((4,), 0.04)
    at._rho_atom_f, at._g_conn_f = 0.5, 0.0
    coal = at.coalesce()
    nn_d = T.safe_cdist(coal.comp_c, diag_inf=True).amin(1)
    r_halo = (2.1 * (2.0 * A.cov_trace_mean(coal.comp_cov)).sqrt()).minimum(0.35 * nn_d)

    # Balanced inside halos -> strictly zero route_w
    x_bal = centers.repeat_interleave(16, dim=0) + 0.05 * torch.randn(64, 2, generator=g)
    _, _, rw_bal, _ = T._continuous_quota_route(
        x_bal,
        at,
        coal,
        r_halo,
        nn_d,
        torch.full((4,), 16),
        torch.tensor(0.2),
        torch.tensor(1.0),
        2.1,
        1.0,
        False,
    )
    assert rw_bal is not None and float(rw_bal.abs().max().item()) == 0.0

    # Surplus on modes 0,1 and deficit on modes 2,3 -> routes surplus particles to modes 2,3 and conserves unit mass
    at.w_fake_ema = torch.tensor([0.5, 0.5, 0.0, 0.0])
    x_sur = centers[:2].repeat_interleave(32, dim=0) + 0.05 * torch.randn(64, 2, generator=g)
    tgt_c, _, rw_reb, _ = T._continuous_quota_route(
        x_sur,
        at,
        coal,
        r_halo,
        nn_d,
        torch.full((4,), 16),
        torch.tensor(0.2),
        torch.tensor(1.0),
        2.1,
        1.0,
        True,
    )
    assert tgt_c is not None and rw_reb is not None
    assert int((rw_reb > 0).sum().item()) >= 16
    assert int((tgt_c[:, 0] > 0.0).sum().item()) >= 16
    assert abs(float(at.w_fake_ema.sum().item()) - 1.0) <= 1e-6
    A.validate_atlas_state(at.state_dict())


def test_contrast_cache_reuse_invalidation_and_simplicial_barycenter_pull():
    g = torch.Generator().manual_seed(313)
    fn = T.AxiomTransportLoss()

    # Exact cache hit on reused real tensor + invalidation on in-place mutation or setting change
    y_buf = _make_disconnected_modes(32)
    yp1, k1 = fn._cached_contrast(y_buf)
    yp2, k2 = fn._cached_contrast(y_buf)
    assert yp1 is yp2 and k1 is k2
    y_buf.add_(0.5)
    yp3, k3 = fn._cached_contrast(y_buf)
    assert yp3 is not yp1 and torch.allclose(yp3, T.project_features(y_buf))
    fn.max_flat_dim = 64
    yp4, _ = fn._cached_contrast(y_buf)
    assert yp4 is not yp3

    # bf16 channels_last 256x256 projection matches fp32 block average closely
    img_bf16 = torch.randn(4, 3, 64, 64, generator=g, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    proj_bf16 = T.project_features(img_bf16, max_spatial=8, max_flat_dim=256)
    proj_ref = (img_bf16.float().reshape(4, 3, 8, 8, 8, 8).mean((3, 5)).reshape(4, -1) * 8.0).to(torch.bfloat16)
    assert proj_bf16.dtype == torch.bfloat16 and torch.allclose(proj_bf16.float(), proj_ref.float(), atol=5e-2)

    # Smooth local barycenter projection pulls an off-saddle interior sample back onto z = x * y
    y_sad = _make_saddle_batch(96)
    z_grid = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    z_off = torch.tensor(0.35, requires_grad=True)

    def _saddle_lift(z: torch.Tensor) -> torch.Tensor:
        return torch.stack((z[:, 0], z[:, 1], z[:, 0] * z[:, 1] + z_off), dim=1)

    loss_s = T.AxiomTransportLoss().simplicial_manifold_loss(
        z_grid, _saddle_lift, y_sad, record=SimpleNamespace(thermal_scale=1.0, chord_phase=0.0), chart_weight=0.0
    )
    loss_s.backward()
    assert z_off.grad is not None and float(z_off.grad.item()) > 0.05
