"""Unit and mathematical-invariant tests for L1 ``particlegan._axiom.atlas``."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import math
from pathlib import Path
import random
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pytest
import torch

from particlegan._axiom import atlas
from particlegan._axiom.atlas import (
    MAX_FULL_COV_DIM,
    MAX_LEDGER_PARTICLES,
    SCHEMA_VERSION,
    HamiltonCommitmentLedger,
    PrototypeAtlas,
    batch_annular_gap_ratio,
    coalesce_directional_ltc,
    coalesce_pythagorean_ltv,
    compute_connectivity_gate,
    compute_rho_atom,
    cov_directional_var,
    cov_from_scalar,
    cov_principal_split,
    cov_shrink_mp,
    cov_trace_mean,
    validate_atlas_state,
    validate_ledger_state,
)

ATLAS_PATH = Path(atlas.__file__).resolve()


def _radon_bool_cc(node: ast.AST) -> int:
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp),
        ):
            cc += 1
        elif isinstance(child, ast.With):
            cc += len(child.items)
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
    return cc


def _mccabe_c901(node: ast.AST) -> int:
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp),
        ):
            cc += 1
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
    return cc


def _rot2(angle_deg: float) -> torch.Tensor:
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def _sample_rotated_ellipse(
    n: int,
    center: torch.Tensor,
    sig_major: float,
    sig_minor: float,
    angle_deg: float,
    gen: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot = _rot2(angle_deg)
    scale = torch.diag(torch.tensor([sig_major, sig_minor], dtype=torch.float32))
    cov = rot @ (scale @ scale) @ rot.T
    z = torch.randn(n, 2, generator=gen)
    return center.unsqueeze(0) + (z @ scale) @ rot.T, cov


@contextmanager
def _count_cuda_syncs() -> Iterator[Dict[str, int]]:
    syncs = {"item": 0, "tolist": 0, "bool": 0, "unique": 0}
    orig = (torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.__bool__, torch.unique)

    def _wrap_t(key: str, fn: object):
        def _inner(self: torch.Tensor, *a: object, **kw: object) -> object:
            if self.device.type == "cuda":
                syncs[key] += 1
            return fn(self, *a, **kw)  # type: ignore[operator]

        return _inner

    def _wrap_u(inp: object, *a: object, **kw: object) -> object:
        if isinstance(inp, torch.Tensor) and inp.device.type == "cuda":
            syncs["unique"] += 1
        return orig[3](inp, *a, **kw)

    torch.Tensor.item = _wrap_t("item", orig[0])  # type: ignore[assignment]
    torch.Tensor.tolist = _wrap_t("tolist", orig[1])  # type: ignore[assignment]
    torch.Tensor.__bool__ = _wrap_t("bool", orig[2])  # type: ignore[assignment]
    torch.unique = _wrap_u  # type: ignore[assignment]
    try:
        yield syncs
    finally:
        torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.__bool__, torch.unique = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Static L1 Layer, Forbidden Ops, Style & Complexity Gates
# ---------------------------------------------------------------------------


def test_static_l1_imports_and_forbidden_ops() -> None:
    tree = ast.parse(ATLAS_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                assert node.level == 1 and node.module == "math_ops"
            elif node.module is not None:
                assert not node.module.startswith("particlegan")
        elif isinstance(node, ast.Import):
            assert all(not a.name.startswith("particlegan") for a in node.names)

    forbidden = {"svd", "svdvals", "eigh", "eigvalsh", "pinv", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, f"Forbidden op {node.attr!r}"
            is_ag = node.attr == "grad" and isinstance(node.value, ast.Attribute) and node.value.attr == "autograd"
            assert not is_ag, "Forbidden torch.autograd.grad in atlas.py"


def test_static_l1_style_caches_and_complexity() -> None:
    source = ATLAS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name, val in vars(atlas).items():
        if not name.startswith("__"):
            assert not isinstance(val, (torch.Tensor, dict, list, set))

    for lineno, line in enumerate(source.splitlines(), start=1):
        assert len(line) <= 120, f"Line {lineno} exceeds 120 chars ({len(line)})"
        assert chr(59) not in line, f"Semicolon found on line {lineno}"

    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    c901_vals = [_mccabe_c901(fn) for fn in fns]
    bool_vals = [_radon_bool_cc(fn) for fn in fns]
    assert max(c901_vals) <= 10, f"Max C901 {max(c901_vals)} > 10"
    assert max(bool_vals) <= 15, f"Max BoolCC {max(bool_vals)} > 15"
    mean_c901 = sum(c901_vals) / len(c901_vals)
    assert 2.5 <= mean_c901 <= 5.5, f"Mean C901 {mean_c901:.2f} out of target band"


# ---------------------------------------------------------------------------
# 2. Clock-Free Stationary Hit Confidence & Zero Global RNG Mutation
# ---------------------------------------------------------------------------


def test_clock_free_stationary_hit_confidence_and_zero_rng() -> None:
    gen = torch.Generator().manual_seed(0)
    centers = torch.tensor([[-4.0, 0.0], [4.0, 0.0]], dtype=torch.float32)

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()

    at = PrototypeAtlas(max_prototypes=4, beta_n=0.04)
    at.initialized, at.dim = True, 2
    at.proto_c = centers.clone()
    at.proto_cov = cov_from_scalar(torch.full((2,), 0.01), 2)
    at.proto_w = torch.tensor([0.5, 0.5])
    at.w_fake_ema = torch.tensor([0.5, 0.5])
    at.proto_n = torch.tensor([1.0, 1.0])
    at.psi = torch.zeros(2)
    at.v_push = torch.full((2,), 0.01)

    for _ in range(250):
        yb = torch.cat(
            [
                centers[0] + 0.1 * torch.randn(16, 2, generator=gen),
                centers[1] + 0.1 * torch.randn(16, 2, generator=gen),
            ],
            dim=0,
        )
        at.update(yb)

    assert at.proto_n is not None
    assert torch.allclose(at.proto_n, torch.full((2,), 400.0), rtol=0.05)

    n_after_250 = at.proto_n.clone()
    for _ in range(100):
        yb = torch.cat(
            [
                centers[0] + 0.1 * torch.randn(16, 2, generator=gen),
                centers[1] + 0.1 * torch.randn(16, 2, generator=gen),
            ],
            dim=0,
        )
        at.update(yb)
    assert torch.allclose(at.proto_n, n_after_250, rtol=0.02)

    n_pre_shock = float(at.proto_n.mean().item())
    y_shock = centers[0] + torch.tensor([80.0, 80.0]) + 0.1 * torch.randn(32, 2, generator=gen)
    at.update(y_shock)
    assert float(at.proto_n.mean().item()) < 0.5 * n_pre_shock

    coal = at.coalesce(y_shock)
    _ = at.commit_particles(y_shock, torch.arange(32), 32, coalesced=coal)
    sd = at.state_dict()
    PrototypeAtlas(max_prototypes=4).load_state_dict(sd)

    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


# ---------------------------------------------------------------------------
# 3. Unbiased Streaming Covariance (b_k >= 1 and B < K safe)
# ---------------------------------------------------------------------------


def test_unbiased_streaming_covariance_and_b_lt_k() -> None:
    gen = torch.Generator().manual_seed(11)
    centers = torch.tensor([[-6.0, 0.0], [6.0, 0.0], [0.0, -6.0], [0.0, 6.0]])
    angles = [25.0, 70.0, 115.0, 160.0]
    true_cov_t = torch.stack(
        [_sample_rotated_ellipse(2, centers[0], 0.50, 0.05, ang, gen)[1] for ang in angles],
        dim=0,
    )

    at = PrototypeAtlas(max_prototypes=4)
    at.initialized, at.dim = True, 2
    at.proto_c = centers.clone()
    at.proto_cov = cov_from_scalar(torch.full((4,), 0.10), 2)
    at.proto_w = torch.full((4,), 0.25)
    at.w_fake_ema = torch.full((4,), 0.25)
    at.proto_n = torch.ones(4)
    at.psi = torch.zeros(4)
    at.v_push = torch.full((4,), 0.10)

    # Strictly B = 3 < K = 4 so b_k == 1 happens on most hits!
    cov_acc = torch.zeros_like(true_cov_t)
    n_avg = 0
    for step in range(400):
        idx = torch.randint(0, 4, (3,), generator=gen)
        pts = [_sample_rotated_ellipse(1, centers[cid], 0.50, 0.05, angles[cid], gen)[0] for cid in idx.tolist()]
        at.update(torch.cat(pts, dim=0))
        if step >= 200:
            cov_acc += at.proto_cov
            n_avg += 1

    assert at.proto_cov is not None and at.proto_cov.shape == (4, 2, 2)
    assert torch.allclose(at.proto_cov, at.proto_cov.transpose(-1, -2), atol=1e-6)
    assert bool((torch.linalg.eigvalsh(at.proto_cov) > 0.0).all())
    # Time-averaged streaming covariance is unbiased under b_k == 1, and principal axis aligns tightly.
    cov_mean = cov_acc / float(n_avg)
    rel_err = (cov_mean - true_cov_t).norm(dim=(-2, -1)) / true_cov_t.norm(dim=(-2, -1))
    assert float(rel_err.mean().item()) < 0.15
    u_true = torch.linalg.eigh(true_cov_t).eigenvectors[:, :, -1]
    u_est = torch.linalg.eigh(cov_mean).eigenvectors[:, :, -1]
    assert float((u_true * u_est).sum(dim=-1).abs().min().item()) > 0.99


# ---------------------------------------------------------------------------
# 4. Exact LTC Merge & Principal-Axis Split (incl. 10:1 Rotated Ellipses)
# ---------------------------------------------------------------------------


def test_ltc_exact_merge_and_split_rotated_ellipses() -> None:
    gen = torch.Generator().manual_seed(7)
    _, cov_a = _sample_rotated_ellipse(2, torch.zeros(2), 1.0, 0.1, 37.0, gen)
    _, cov_b = _sample_rotated_ellipse(2, torch.zeros(2), 0.8, 0.08, 112.0, gen)
    ca, cb = torch.tensor([1.5, -2.0]), torch.tensor([2.5, -1.2])
    wa, wb = 0.3, 0.2

    at = PrototypeAtlas(max_prototypes=8)
    at.initialized, at.dim = True, 2
    at.proto_c = torch.stack([ca, cb, torch.tensor([10.0, 10.0])])
    at.proto_cov = torch.stack([cov_a, cov_b, 0.05 * torch.eye(2)])
    at.proto_w = torch.tensor([wa, wb, 0.5])
    at.w_fake_ema = torch.tensor([0.35, 0.15, 0.5])
    at.proto_n = torch.tensor([12.0, 8.0, 20.0])
    at.psi = torch.tensor([1.0, -0.5, -0.5])
    at.v_push = cov_trace_mean(at.proto_cov)

    wt = wa + wb
    c_ab = (wa * ca + wb * cb) / wt
    da, db = ca - c_ab, cb - c_ab
    expected_cov = (wa * (cov_a + torch.outer(da, da)) + wb * (cov_b + torch.outer(db, db))) / wt

    at.merge_pair(0, 1)
    assert torch.allclose(at.proto_c[0], c_ab, atol=1e-6)
    assert torch.allclose(at.proto_cov[0], expected_cov, atol=1e-5)
    assert math.isclose(float(at.proto_w.sum().item()), 1.0, abs_tol=1e-6)
    assert math.isclose(float(at.w_fake_ema.sum().item()), 1.0, abs_tol=1e-6)
    assert math.isclose(float(at.proto_n[0].item()), 20.0, abs_tol=1e-6)

    # Split rotated 10:1 ellipse (37 deg): exact 100% LTC, minor axis preserved, strictly PSD.
    cov_rot = cov_a.clone()
    at.proto_c[0] = torch.tensor([3.0, -1.0])
    at.proto_cov[0] = cov_rot
    at.proto_w[0], at.proto_w[1] = 0.5, 0.0
    at.w_fake_ema[0], at.w_fake_ema[1] = 0.5, 0.0
    at.psi = torch.tensor([0.6, 0.0, -0.6])

    at.split_slot(0, 1)
    c0, c1 = at.proto_c[0], at.proto_c[1]
    s0, s1 = at.proto_cov[0], at.proto_cov[1]
    c_mid = 0.5 * (c0 + c1)
    d0, d1 = c0 - c_mid, c1 - c_mid
    reconstructed = 0.5 * (s0 + torch.outer(d0, d0)) + 0.5 * (s1 + torch.outer(d1, d1))
    assert torch.allclose(c_mid, torch.tensor([3.0, -1.0]), atol=1e-6)
    assert torch.allclose(reconstructed, cov_rot, atol=1e-5)

    ev_child = torch.linalg.eigvalsh(s0)
    assert bool((ev_child > 0.0).all())
    assert math.isclose(float(ev_child[0].item()), 0.01, rel_tol=1e-4)
    assert math.isclose(float(ev_child[1].item()), 0.84, rel_tol=1e-4)

    # Extreme 1000:1 ellipse and rank-1 matrix stay strictly PSD and exact LTC.
    q8, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen))
    for diag_vals in (
        torch.tensor([1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-3]),
        torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        mat = q8 @ torch.diag(diag_vals) @ q8.T
        _, delta, child = cov_principal_split(mat, seed=3)
        assert torch.allclose(child + torch.outer(delta, delta), mat, atol=1e-4)
        assert float(torch.linalg.eigvalsh(child).min().item()) >= -1e-6


def test_gersho_merge_before_split_ltv_retention_in_update() -> None:
    k_m = 16
    centers = torch.zeros((k_m, 2), dtype=torch.float32)
    for i in range(k_m):
        centers[i, 0] = float(i) * 10.0
    centers[1], centers[2] = torch.tensor([10.0, 0.0]), torch.tensor([11.0, 0.0])
    vars_init = torch.full((k_m,), 0.25)
    vars_init[0], vars_init[1], vars_init[2] = 20.0, 0.10, 0.40

    at = PrototypeAtlas(max_prototypes=k_m)
    at.initialized, at.dim = True, 2
    at.proto_c = centers.clone()
    at.proto_cov = cov_from_scalar(vars_init, 2)
    at.proto_w = torch.full((k_m,), 1.0 / float(k_m))
    at.w_fake_ema = at.proto_w.clone()
    at.proto_n = torch.full((k_m,), 25.0)
    at.psi = torch.zeros(k_m)
    at.v_push = vars_init.clone()

    at.update(centers[3:].repeat(4, 1))
    assert at.proto_var is not None
    expected_v_merged = 0.5 * (0.10 + 0.40) + 0.25 * 1.0 / 2.0
    assert math.isclose(float(at.proto_var[1].item()), expected_v_merged, rel_tol=1e-4)
    split_tot = 0.5 * (float(at.proto_var[0].item()) + float(at.proto_var[2].item())) + (
        0.25 * float((at.proto_c[0] - at.proto_c[2]).square().sum().item()) / 2.0
    )
    assert math.isclose(split_tot, 20.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 5. Directional Coalescence (Anisotropic vs Isotropic) & MP Shrinkage
# ---------------------------------------------------------------------------


def test_directional_coalescence_anisotropic_vs_isotropic_and_mp_shrinkage() -> None:
    # Two parallel 10:1 rotated ellipses (sigma_maj=1.0, sigma_min=0.1) separated by 2.8 along minor axis.
    # Isotropic trace std is 0.711 -> external threshold 4.2*sqrt(2)*0.711 = 4.22 > 2.8 (isotropic FAILS).
    # Directional std along minor/diagonal is 0.10..0.22 -> threshold 0.59..1.34 << 2.8 (directional PASSES).
    rot = _rot2(33.0)
    u_maj, u_min = rot[:, 0], rot[:, 1]
    cov_ell = rot @ torch.diag(torch.tensor([1.0, 0.01])) @ rot.T

    c_m0, c_m1 = -1.4 * u_min, 1.4 * u_min
    proto_c = torch.stack([c_m0 - 0.3 * u_maj, c_m0 + 0.3 * u_maj, c_m1 - 0.3 * u_maj, c_m1 + 0.3 * u_maj])
    proto_cov = cov_ell.unsqueeze(0).expand(4, 2, 2).clone()
    proto_w = torch.full((4,), 0.25)
    proto_n = torch.full((4,), 20.0)

    coal = coalesce_directional_ltc(proto_c, proto_cov, proto_w, proto_n)
    assert int(coal.comp_c.shape[0]) == 2
    assert bool(coal.all_iso.item())
    assert bool(coal.multicell.all().item())
    expected_comp_cov = cov_ell + 0.09 * torch.outer(u_maj, u_maj)
    assert torch.allclose(coal.comp_cov[0], expected_comp_cov, atol=1e-5)
    assert torch.allclose(coal.comp_cov[1], expected_comp_cov, atol=1e-5)
    assert torch.allclose(coal.comp_n_eff, torch.tensor([40.0, 40.0]))

    # Isotropic scalar coalescence fails on the exact same modes due to the kappa* = 8.665 cliff.
    _, _, _, iso_multicell = coalesce_pythagorean_ltv(proto_c, cov_trace_mean(proto_cov), proto_w)
    assert not bool(iso_multicell.all().item())

    # 10x width-ratio guard (Suite 5B): must NOT false-merge var=0.01 with var=1.0.
    pc_10x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]])
    pv_10x = torch.tensor([0.01, 1.0, 0.01])
    pw_10x = torch.tensor([0.25, 0.25, 0.50])
    _, _, _, coal_10x = coalesce_pythagorean_ltv(pc_10x, pv_10x, pw_10x)
    assert not bool(coal_10x[0].item())

    # Marchenko-Pastur shrinkage: D=2, n_eff=40 -> lam=1e-4 (<1% bias on 100:1 ellipse).
    cov_100 = torch.diag(torch.tensor([1.0, 0.01])).unsqueeze(0)
    shrunk_low_r = cov_shrink_mp(cov_100, torch.tensor([40.0]))[0]
    assert math.isclose(float(shrunk_low_r[1, 1].item()), 0.01, rel_tol=0.01)

    # D=32, n_eff=16 (r_C = 2.0 >= 1.0) -> exact isotropic shrinkage.
    diag32 = torch.linspace(0.1, 3.2, 32)
    shrunk_high_r = cov_shrink_mp(torch.diag(diag32).unsqueeze(0), torch.tensor([16.0]))[0]
    assert torch.allclose(shrunk_high_r, diag32.mean() * torch.eye(32), atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Continuous Annular-Gap Ratio rho_atom & Connectivity Gate g_conn
# ---------------------------------------------------------------------------


def test_continuous_rho_atom_and_connectivity_gate_regimes() -> None:
    gen = torch.Generator().manual_seed(42)

    bits = torch.tensor([[((i >> b) & 1) * 2.0 - 1.0 for b in range(4)] for i in range(16)])
    y_exact = bits[torch.randint(0, 16, (128,), generator=gen)]
    y_noisy = bits[torch.randint(0, 16, (128,), generator=gen)] + 0.015 * torch.randn(128, 4, generator=gen)
    ang = (2.0 * math.pi / 8.0) * torch.arange(8, dtype=torch.float32)
    ring_c = torch.stack((3.0 * ang.cos(), 3.0 * ang.sin()), dim=1)
    y_gauss = ring_c[torch.randint(0, 8, (128,), generator=gen)] + 0.12 * torch.randn(128, 2, generator=gen)
    t = torch.linspace(0.5, 4.0 * math.pi, 128)
    y_spiral = torch.stack((t * t.cos(), t * t.sin()), dim=1) / (4.0 * math.pi)

    rho_b = [float(batch_annular_gap_ratio(y).item()) for y in (y_exact, y_noisy, y_gauss, y_spiral)]
    assert rho_b[0] < 1e-4
    assert rho_b[0] < rho_b[1] < rho_b[2] < rho_b[3]
    assert rho_b[3] > 0.40

    assert math.isclose(float(batch_annular_gap_ratio(y_noisy * 1e-3).item()), rho_b[1], abs_tol=1e-4)
    assert math.isclose(float(batch_annular_gap_ratio(y_noisy * 1e3).item()), rho_b[1], abs_tol=1e-4)

    r_exact = float(
        compute_rho_atom(bits, cov_from_scalar(torch.full((16,), 1e-24), 4), torch.full((16,), 20.0)).item()
    )
    r_noisy = float(
        compute_rho_atom(
            bits,
            cov_from_scalar(torch.full((16,), 0.015**2), 4),
            torch.full((16,), 20.0),
        ).item()
    )
    r_gauss = float(
        compute_rho_atom(
            ring_c,
            cov_from_scalar(torch.full((8,), 0.12**2), 2),
            torch.full((8,), 20.0),
        ).item()
    )
    assert r_exact < 1e-6 and 0.005 < r_noisy < 0.05 and 0.03 < r_gauss < 0.16

    for y_cur, r_cur in ((y_exact, r_exact), (y_noisy, r_noisy), (y_gauss, r_gauss)):
        assert float(compute_connectivity_gate(y_cur, rho_atom=torch.tensor(r_cur)).item()) == 0.0
    assert float(compute_connectivity_gate(y_spiral, rho_atom=torch.tensor(rho_b[3])).item()) > 0.75


# ---------------------------------------------------------------------------
# 7. Unified GPU-Vectorized Hamilton Commitment Ledger
# ---------------------------------------------------------------------------


def test_hamilton_ledger_quotas_eviction_and_bijection() -> None:
    # 7A: Exact P == K_C bijection even from fully collapsed initialization at origin.
    k_c = 32
    ang = (2.0 * math.pi / float(k_c)) * torch.arange(k_c, dtype=torch.float32)
    comp_c = torch.stack((5.0 * ang.cos(), 5.0 * ang.sin()), dim=1)
    comp_w = torch.full((k_c,), 1.0 / float(k_c))

    ledger = HamiltonCommitmentLedger()
    out = ledger.update(torch.zeros((k_c, 2)), torch.arange(k_c), comp_c, comp_w, num_particles=k_c)
    assert torch.equal(out.quotas, torch.ones(k_c, dtype=torch.long))
    assert torch.equal(out.occupancy, torch.ones(k_c, dtype=torch.long))
    assert int(torch.unique(out.batch_comp).numel()) == k_c

    # 7B: Over-quota eviction keeps closest particles and evicts furthest ones.
    c2 = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    pts4 = torch.tensor([[0.1, 0.0], [0.2, 0.0], [5.0, 0.0], [10.1, 0.0]])
    led2 = HamiltonCommitmentLedger()
    led2.update(pts4, torch.arange(4), c2, torch.tensor([0.75, 0.25]), num_particles=4)
    out2 = led2.update(pts4, torch.arange(4), c2, torch.tensor([0.5, 0.5]), num_particles=4)
    assert out2.batch_comp.tolist() == [0, 0, 1, 1]

    # 7C: Mini-batch (B < P) reaches exact full-table quotas and tracks shifted targets.
    led3 = HamiltonCommitmentLedger()
    gen = torch.Generator().manual_seed(9)
    p_tot = 64
    c4 = torch.tensor([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]])
    w4 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    x_all = torch.randn(p_tot, 2, generator=gen) * 3.0
    for start in range(0, p_tot, 16):
        rows_b = torch.arange(start, start + 16)
        out3 = led3.update(x_all[rows_b], rows_b, c4, w4, num_particles=p_tot)
    assert torch.equal(out3.occupancy, out3.quotas)
    assert int(out3.quotas.sum().item()) == p_tot

    c4_shifted = c4 + torch.tensor([[2.0, -1.0]])
    out_s = led3.update(x_all[:16], torch.arange(16), c4_shifted, w4, num_particles=p_tot)
    assert torch.allclose(out_s.batch_target, c4_shifted[out_s.batch_comp], atol=1e-6)

    # 7D: Disengaged on connected manifolds or when P > MAX_LEDGER_PARTICLES.
    at_m = PrototypeAtlas(max_prototypes=16)
    t = torch.linspace(0.5, 4.0 * math.pi, 128)
    y_sp = torch.stack((t * t.cos(), t * t.sin()), dim=1) / (4.0 * math.pi)
    for _ in range(5):
        at_m.update(y_sp)
    assert at_m.commit_particles(y_sp[:16], torch.arange(16), num_particles=64) is None

    at_iso = PrototypeAtlas(max_prototypes=8)
    at_iso.initialized, at_iso.dim = True, 2
    at_iso.proto_c, at_iso.proto_cov = (
        c4.clone(),
        cov_from_scalar(torch.full((4,), 0.01), 2),
    )
    at_iso.proto_w, at_iso.w_fake_ema = w4.clone(), w4.clone()
    at_iso.proto_n, at_iso.psi, at_iso.v_push = (
        torch.full((4,), 20.0),
        torch.zeros(4),
        torch.full((4,), 0.01),
    )
    assert at_iso.commit_particles(x_all[:8], torch.arange(8), MAX_LEDGER_PARTICLES + 1) is None
    assert at_iso.commit_particles(x_all[:8], torch.arange(8), 64) is not None


# ---------------------------------------------------------------------------
# 8. D <= 64 Full SPD vs D > 64 Diagonal Parity
# ---------------------------------------------------------------------------


def test_spd_d64_vs_diagonal_d65_parity() -> None:
    gen = torch.Generator().manual_seed(17)
    d64_vec = torch.linspace(0.05, 1.5, MAX_FULL_COV_DIM).unsqueeze(0).repeat(4, 1)
    d64_full = torch.diag_embed(d64_vec)
    dirs = torch.randn(4, 4, MAX_FULL_COV_DIM, generator=gen)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    assert torch.allclose(cov_trace_mean(d64_full), cov_trace_mean(d64_vec), atol=1e-6)
    assert torch.allclose(
        cov_directional_var(d64_full, dirs),
        cov_directional_var(d64_vec, dirs),
        atol=1e-5,
    )

    n_eff = torch.tensor([10.0, 32.0, 64.0, 256.0])
    assert torch.allclose(
        cov_shrink_mp(d64_full, n_eff).diagonal(dim1=-2, dim2=-1),
        cov_shrink_mp(d64_vec, n_eff),
        atol=1e-6,
    )

    for dim, expected_ndim in ((MAX_FULL_COV_DIM, 3), (MAX_FULL_COV_DIM + 1, 2)):
        at = PrototypeAtlas(max_prototypes=8)
        c_modes = torch.zeros((2, dim))
        c_modes[0, 0], c_modes[1, 0] = -8.0, 8.0
        for _ in range(10):
            idx = torch.randint(0, 2, (32,), generator=gen)
            at.update(c_modes[idx] + 0.1 * torch.randn(32, dim, generator=gen))
        assert at.proto_cov is not None and at.proto_cov.ndim == expected_ndim
        c0_pre, cov0_pre = at.proto_c[0].clone(), at.proto_cov[0].clone()
        at.split_slot(0, 1)
        delta = 0.5 * (at.proto_c[1] - at.proto_c[0])
        assert torch.allclose(0.5 * (at.proto_c[0] + at.proto_c[1]), c0_pre, atol=1e-5)
        rec = at.proto_cov[0] + torch.outer(delta, delta) if expected_ndim == 3 else at.proto_cov[0] + delta.square()
        assert torch.allclose(rec, cov0_pre, atol=1e-5)
        coal = at.coalesce()
        assert bool(coal.all_iso.item()) and int(coal.comp_c.shape[0]) == 2


# ---------------------------------------------------------------------------
# 9. < 5-Step Shock Recovery (Scale-Up, Scale-Down, Translation) & Mass Cons.
# ---------------------------------------------------------------------------


def test_shock_recovery_under_5_steps_and_mass_conservation() -> None:
    gen = torch.Generator().manual_seed(23)
    c4 = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]])
    at = PrototypeAtlas(max_prototypes=8)
    for _ in range(15):
        yb = c4[torch.randint(0, 4, (128,), generator=gen)] + 0.1 * torch.randn(128, 2, generator=gen)
        at.update(yb)
        assert abs(float(at.proto_w.sum().item()) - 1.0) < 1e-5
        assert abs(float(at.w_fake_ema.sum().item()) - 1.0) < 1e-5

    # 9A: Pure 100x scale-up shock (std 0.1 -> 10.0, var 0.01 -> 100) recovers in 3 steps.
    at_scale = PrototypeAtlas(max_prototypes=4)
    at_scale.initialized, at_scale.dim = True, 2
    at_scale.proto_c = c4.clone()
    at_scale.proto_cov = cov_from_scalar(torch.full((4,), 0.01), 2)
    at_scale.proto_w = torch.full((4,), 0.25)
    at_scale.w_fake_ema = torch.full((4,), 0.25)
    at_scale.proto_n = torch.full((4,), 25.0)
    at_scale.psi = torch.zeros(4)
    at_scale.v_push = torch.full((4,), 0.01)
    for _ in range(3):
        yb_su = c4[torch.randint(0, 4, (128,), generator=gen)] + 10.0 * torch.randn(128, 2, generator=gen)
        at_scale.update(yb_su)
    assert at_scale.proto_var is not None
    med_su = float(at_scale.proto_var.median().item())
    assert 25.0 <= med_su <= 250.0, f"Pure scale-up recovery failed: med_v={med_su}"

    # 9B: 100x teleport shock & 1e-2x return preserve unit mass & collapse spread < 5 steps.
    for _ in range(4):
        yb_up = (c4 * 100.0)[torch.randint(0, 4, (128,), generator=gen)] + 10.0 * torch.randn(128, 2, generator=gen)
        at.update(yb_up)
        assert abs(float(at.proto_w.sum().item()) - 1.0) < 1e-5
        assert abs(float(at.w_fake_ema.sum().item()) - 1.0) < 1e-5
    assert at.proto_var is not None
    w_var_up = float((at.proto_w * at.proto_var).sum().item())
    assert 20.0 <= w_var_up <= 1000.0, f"Teleport scale-up failed: w_var={w_var_up}"

    for _ in range(4):
        yb_dn = c4[torch.randint(0, 4, (128,), generator=gen)] + 0.1 * torch.randn(128, 2, generator=gen)
        at.update(yb_dn)
        assert abs(float(at.proto_w.sum().item()) - 1.0) < 1e-5
        assert abs(float(at.w_fake_ema.sum().item()) - 1.0) < 1e-5
    w_var_dn = float((at.proto_w * at.proto_var).sum().item())
    assert w_var_dn < 15.0, f"Teleport scale-down failed: w_var={w_var_dn}"

    # 9C: Pure unhit collapse protection (v_k > 4 * v_tot) shrinks variance in < 5 steps.
    at_col = PrototypeAtlas(max_prototypes=4)
    at_col.initialized, at_col.dim = True, 2
    at_col.proto_c = c4.clone()
    at_col.proto_cov = cov_from_scalar(torch.full((4,), 100.0), 2)
    at_col.proto_w = torch.full((4,), 0.25)
    at_col.w_fake_ema = torch.full((4,), 0.25)
    at_col.proto_n = torch.full((4,), 20.0)
    at_col.psi = torch.zeros(4)
    at_col.v_push = torch.full((4,), 100.0)
    for _ in range(3):
        yb_small = c4[torch.randint(0, 4, (128,), generator=gen)] + 0.1 * torch.randn(128, 2, generator=gen)
        at_col.update(yb_small)
    assert float(at_col.proto_var.median().item()) < 0.2

    # 9D: Single extreme outlier preserves exact unit mass on proto_w & w_fake_ema.
    shock_1 = torch.cat(
        [
            c4[torch.randint(0, 4, (127,), generator=gen)] + 0.05 * torch.randn(127, 2, generator=gen),
            torch.tensor([[80.0, 80.0]]),
        ],
        dim=0,
    )
    at.update(shock_1)
    assert abs(float(at.proto_w.sum().item()) - 1.0) < 1e-5
    assert abs(float(at.w_fake_ema.sum().item()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 10. Strict Schema-2 Checkpoint Roundtrip & Fail-Closed Tamper Rejection
# ---------------------------------------------------------------------------


def test_schema2_checkpoint_roundtrip_and_tamper_rejection() -> None:
    gen = torch.Generator().manual_seed(31)

    at_empty = PrototypeAtlas(max_prototypes=12)
    sd_empty = at_empty.state_dict()
    assert sd_empty["schema"] == SCHEMA_VERSION
    at_empty2 = PrototypeAtlas(max_prototypes=12)
    at_empty2.load_state_dict(sd_empty)
    assert not at_empty2.initialized and at_empty2.dim == 0

    c4 = torch.tensor([[-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, 5.0]])
    at = PrototypeAtlas(max_prototypes=12)
    for _ in range(8):
        yb = c4[torch.randint(0, 4, (64,), generator=gen)] + 0.08 * torch.randn(64, 2, generator=gen)
        at.update(yb)
    coal = at.coalesce(yb)
    _ = at.commit_particles(yb[:16], torch.arange(16), 32, coalesced=coal)

    sd = at.state_dict()
    at_restored = PrototypeAtlas(max_prototypes=12)
    at_restored.load_state_dict(sd)

    y_next = c4[torch.randint(0, 4, (64,), generator=gen)] + 0.08 * torch.randn(64, 2, generator=gen)
    at.update(y_next)
    at_restored.update(y_next)
    assert torch.equal(at.proto_c, at_restored.proto_c)
    assert torch.equal(at.proto_cov, at_restored.proto_cov)
    assert torch.equal(at.proto_n, at_restored.proto_n)
    l_a = at.commit_particles(y_next[:16], torch.arange(16, 32), 32, coalesced=at.coalesce(y_next))
    l_b = at_restored.commit_particles(y_next[:16], torch.arange(16, 32), 32, coalesced=at_restored.coalesce(y_next))
    assert l_a is not None and l_b is not None
    assert torch.equal(l_a.batch_comp, l_b.batch_comp) and torch.equal(l_a.row_comp, l_b.row_comp)

    def _mut(**overrides: object) -> Dict[str, object]:
        out = at.state_dict()
        out.update(overrides)
        return out

    bad_cases: List[Dict[str, object]] = [
        _mut(schema=1),
        _mut(schema=True),
        {k: v for k, v in sd.items() if k != "psi"},
        {**sd, "extra": 0},
        _mut(rho_atom_ema=float("nan")),
        _mut(g_conn_ema=1.5),
        _mut(proto_w=sd["proto_w"] * 0.5),
        _mut(proto_n=-sd["proto_n"]),
        _mut(proto_cov=sd["proto_cov"] + torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])),
        _mut(proto_cov=torch.full_like(sd["proto_cov"], float("inf"))),
        _mut(proto_cov=sd["proto_cov"][:, :, 0]),
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            validate_atlas_state(bad)

    with pytest.raises(ValueError):
        PrototypeAtlas(max_prototypes=16).load_state_dict(sd)

    lsd = at.ledger.state_dict()
    for bad_l in (
        {**lsd, "schema": 1},
        {**lsd, "row_comp": lsd["row_comp"] - 5},
        {**lsd, "num_particles": MAX_LEDGER_PARTICLES + 1},
        {**lsd, "row_dist": -lsd["row_dist"]},
    ):
        with pytest.raises(ValueError):
            validate_ledger_state(bad_l)


# ---------------------------------------------------------------------------
# 10b. Cold-Start Discrete Regimes (disc_4mode, five_word, bytegan_256, grid100)
# ---------------------------------------------------------------------------


def test_cold_start_discrete_regimes_and_manifold_separation() -> None:
    # 1. disc_4mode (D=2, B=64, sig=0.12): coalesces >10 protos/mode -> K_c=4, all_iso=True.
    gen = torch.Generator().manual_seed(0)
    c4 = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]])
    at4 = PrototypeAtlas(max_prototypes=256)
    for _ in range(25):
        y4 = c4[torch.randint(0, 4, (64,), generator=gen)] + 0.12 * torch.randn(64, 2, generator=gen)
        at4.update(y4)
        coal4 = at4.coalesce(y4)
    assert int(coal4.comp_c.shape[0]) == 4 and bool(coal4.all_iso.item())
    assert at4.commit_particles(y4[:16], torch.arange(16), 64, coalesced=coal4) is not None

    # 2. five_word (5 exact atoms, B=64): deduplicates on step 0 -> K_p=K_c=5, all_iso=True.
    gen = torch.Generator().manual_seed(0)
    w5 = torch.eye(5) * 2.0
    at5 = PrototypeAtlas(max_prototypes=256)
    for step in range(25):
        y5 = w5[torch.randint(0, 5, (64,), generator=gen)]
        at5.update(y5)
        coal5 = at5.coalesce(y5)
        if step == 0:
            assert int(at5.proto_c.shape[0]) == 5 and bool(coal5.all_iso.item())
    assert int(coal5.comp_c.shape[0]) == 5 and bool(coal5.all_iso.item())
    assert at5.rho_atom_ema < 1e-4 and at5.g_conn_ema < 1e-3
    assert at5.commit_particles(y5[:16], torch.arange(16), 64, coalesced=coal5) is not None

    # 3. bytegan_256 ({-1,+1}^8, B=64): discovers all 256 atoms with rho=0 & active ledger.
    gen = torch.Generator().manual_seed(0)
    b256 = torch.tensor([[((i >> b) & 1) * 2.0 - 1.0 for b in range(8)] for i in range(256)])
    at256 = PrototypeAtlas(max_prototypes=256)
    for _ in range(25):
        yb = b256[torch.randint(0, 256, (64,), generator=gen)]
        at256.update(yb)
        coal256 = at256.coalesce(yb)
        assert bool(coal256.all_iso.item())
    assert int(at256.proto_c.shape[0]) == 256 and int(coal256.comp_c.shape[0]) == 256
    assert at256.rho_atom_ema < 1e-4 and at256.g_conn_ema < 1e-3
    assert at256.commit_particles(yb[:16], torch.arange(16), 256, coalesced=coal256) is not None

    # 4. grid100 (10x10 modes, sig=0.05, B=64): discovers & isolates all 100 modes.
    gen = torch.Generator().manual_seed(0)
    gx = torch.linspace(-4.5, 4.5, 10)
    g100 = torch.stack(torch.meshgrid(gx, gx, indexing="ij"), dim=-1).reshape(100, 2)
    at100 = PrototypeAtlas(max_prototypes=256)
    for _ in range(40):
        yg = g100[torch.randint(0, 100, (64,), generator=gen)] + 0.05 * torch.randn(64, 2, generator=gen)
        at100.update(yg)
        coal100 = at100.coalesce(yg)
    assert int(coal100.comp_c.shape[0]) == 100 and bool(coal100.all_iso.item())
    assert at100.g_conn_ema < 0.05

    # 5. Connected spiral stays all_iso=False, g_conn_ema > 0.80, ledger disengaged.
    gen = torch.Generator().manual_seed(0)
    at_sp = PrototypeAtlas(max_prototypes=256)
    for _ in range(25):
        t = 0.5 + torch.rand(64, generator=gen) * (4.0 * math.pi - 0.5)
        ys = torch.stack((t * t.cos(), t * t.sin()), dim=1) / (4.0 * math.pi)
        at_sp.update(ys)
        coal_sp = at_sp.coalesce(ys)
    assert not bool(coal_sp.all_iso.item()) and at_sp.g_conn_ema > 0.80
    assert at_sp.commit_particles(ys[:16], torch.arange(16), 64, coalesced=coal_sp) is None


# ---------------------------------------------------------------------------
# 11. CUDA Execution & Zero-Sync GPU Hamilton Commitment Ledger Proof
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_cuda_atlas_and_zero_sync_hamilton_ledger() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(0)

    for dim, exp_ndim in ((16, 3), (96, 2)):
        at = PrototypeAtlas(max_prototypes=16, device=dev)
        c4 = torch.zeros((4, dim), device=dev)
        for idx in range(4):
            c4[idx, idx] = 8.0
        for _ in range(6):
            sel = torch.randint(0, 4, (64,), device=dev, generator=gen)
            yb = c4[sel] + 0.05 * torch.randn(64, dim, device=dev, generator=gen)
            at.update(yb)
        assert at.proto_cov is not None and at.proto_cov.device.type == "cuda"
        assert at.proto_cov.ndim == exp_ndim
        coal = at.coalesce(yb)
        assert coal.comp_c.device.type == "cuda" and bool(coal.all_iso.item())

    k_c = 64
    comp_c = torch.randn(k_c, 8, device=dev, generator=gen) * 5.0
    comp_w = torch.full((k_c,), 1.0 / float(k_c), device=dev)
    ledger = HamiltonCommitmentLedger()
    with _count_cuda_syncs() as syncs:
        out = ledger.update(
            torch.zeros((k_c, 8), device=dev),
            torch.arange(k_c, device=dev),
            comp_c,
            comp_w,
            num_particles=k_c,
        )
    assert sum(syncs.values()) == 0, f"Expected 0 CUDA host syncs, got {syncs}"
    assert out.batch_comp.device.type == "cuda" and out.batch_target.device.type == "cuda"
    assert torch.equal(out.occupancy, torch.ones(k_c, device=dev, dtype=torch.long))
    assert int(torch.unique(out.batch_comp).numel()) == k_c


def test_law1_wishart_shrinkage_fisher_isolation_and_hysteresis() -> None:
    # 1. Wishart-excess shrinkage shrinks noisy isotropic finite-sample cov stronger than true anisotropy.
    gen = torch.Generator().manual_seed(7)
    dim = 32
    samples = torch.randn(100, dim, generator=gen)
    emp_iso = (samples.T @ samples / 100.0).unsqueeze(0)
    aniso = torch.diag(torch.cat((torch.full((4,), 10.0), torch.full((dim - 4,), 0.1)))).unsqueeze(0)
    n_eff = torch.tensor([360.0])  # n_cov = 120, gam = 32/120 = 0.267
    s_iso = cov_shrink_mp(emp_iso, n_eff)[0]
    s_aniso = cov_shrink_mp(aniso, n_eff)[0]
    iso_cond = float(torch.linalg.eigvalsh(s_iso).max() / torch.linalg.eigvalsh(s_iso).min())
    aniso_cond = float(torch.linalg.eigvalsh(s_aniso).max() / torch.linalg.eigvalsh(s_aniso).min())
    assert iso_cond < 1.5 and aniso_cond > 20.0

    # 2. 1D Fisher isolation isolates D=32 modes at dist=18 (>2.6*sqrt(32)=14.7, <old 4.2*sqrt(32)=23.8).
    centers = torch.zeros((2, dim))
    centers[0, 0], centers[1, 0] = -9.0, 9.0  # dist = 18.0 > 2.6*sqrt(32) and > 6.0*1.0, < 4.2*sqrt(32)
    covs = torch.eye(dim).unsqueeze(0).expand(2, dim, dim).clone()
    coal = coalesce_directional_ltc(centers, covs, torch.tensor([0.5, 0.5]), torch.tensor([40.0, 40.0]))
    assert bool(coal.all_iso.item()) and int(coal.comp_c.shape[0]) == 2

    # 3. Non-isolated manifold prototypes receive isotropic comp_cov_tgt (no chord-sagitta explosion).
    theta = torch.linspace(0.0, 2.0 * math.pi, 17)[:-1]
    ring = torch.stack((1.5 * theta.cos(), 1.5 * theta.sin()), dim=1)
    tang = torch.stack((-theta.sin(), theta.cos()), dim=1)
    ring_cov = 0.2 * (tang.unsqueeze(2) * tang.unsqueeze(1)) + 1e-4 * torch.eye(2).unsqueeze(0)
    coal_ring = coalesce_directional_ltc(ring, ring_cov, torch.full((16,), 1.0 / 16.0), torch.full((16,), 20.0))
    assert not bool(coal_ring.all_iso.item())
    off_diag = coal_ring.comp_cov_tgt - torch.diag_embed(coal_ring.comp_cov_tgt.diagonal(dim1=-2, dim2=-1))
    assert float(off_diag.abs().amax()) < 1e-6

    # 4. Schmitt-trigger hysteresis avoids evicting settled particles on small weight wobble when all modes occupied.
    comp_c = torch.tensor([[-5.0, 0.0], [5.0, 0.0]])
    ledger = HamiltonCommitmentLedger()
    x_settled = torch.cat((comp_c[0].expand(6, 2), comp_c[1].expand(4, 2)), dim=0)
    rows = torch.arange(10)
    out0 = ledger.update(x_settled, rows, comp_c, torch.tensor([0.6, 0.4]), num_particles=10)
    assert out0.occupancy.tolist() == [6, 4]
    # Wobble weights slightly so exact quota shifts [5, 5], but settled count 6 <= floor(10*0.52 + 0.80)=6 -> kept!
    out1 = ledger.update(x_settled, rows, comp_c, torch.tensor([0.52, 0.48]), num_particles=10)
    assert out1.occupancy.tolist() == [6, 4]
    # If a component is empty, exact quotas are enforced immediately.
    ledger_empty = HamiltonCommitmentLedger()
    out_empty = ledger_empty.update(x_settled, rows, comp_c, torch.tensor([0.52, 0.48]), num_particles=10)
    assert out_empty.occupancy.tolist() == [5, 5]

    # 5. _rebalance_slots never clones dead or low-confidence (proto_n < 0.5) shock-spike slots.
    at = PrototypeAtlas(max_prototypes=8)
    at.dim, at.initialized = 2, True
    at.proto_c = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    at.proto_cov = cov_from_scalar(torch.tensor([100.0, 1.0, 1.0, 1e-4]), 2)
    at.proto_w = torch.tensor([0.30, 0.35, 0.34, 0.01])
    at.w_fake_ema = at.proto_w.clone()
    at.proto_n = torch.tensor([0.20, 20.0, 20.0, 0.01])  # slot 0 has huge spike cov but proto_n < 0.5
    at.psi = torch.zeros(4)
    at.v_push = torch.tensor([100.0, 1.0, 1.0, 1e-4])
    dead = (at.proto_w < (0.04 / 4)) | (at.proto_n < 0.05)
    at._rebalance_slots(dead, True, torch.tensor(0), torch.tensor(1.0), 0.5, 0.25)
    # Dead slot 3 must receive a split of a live slot (1 or 2 near x=10 or 20), NOT shock-spike slot 0 (x=0).
    assert abs(float(at.proto_c[0, 0])) < 1e-6 and float(at.proto_c[3, 0]) > 5.0


def test_fast_path_exact_parity_group_rank_ledger_and_coalesce() -> None:
    from particlegan._axiom.atlas import _group_rank

    # 1. Single-sort int64 bit-cast _group_rank matches reference two-pass stable sort (positive & negative keys).
    gen = torch.Generator().manual_seed(17)
    num_p, num_comp = 128, 16
    gid = torch.randint(0, num_comp + 1, (num_p,), generator=gen)
    dist = torch.randn(num_p, generator=gen) * 10.0
    dist[::7] = 0.0  # exact ties broken by weyl
    dist[::13] = -0.0  # negative zero canonicalized identically to +0.0
    dist[::11] = 1e30  # huge positive sentinel values
    dist[::17] = -1e30  # huge negative scores (e.g. transport.py -best_s)
    weyl = ((torch.arange(num_p, dtype=torch.float32) + 0.5) * 0.6180339887498949).remainder(1.0)
    order, sorted_g, rank = _group_rank(gid, dist, weyl, num_comp)
    ref_order = (dist.clamp(-1e6, 1e6) + 1e-6 * weyl).argsort(stable=True)
    ref_order = ref_order[gid[ref_order].argsort(stable=True)]
    assert torch.equal(order, ref_order) and torch.equal(sorted_g, gid[ref_order])

    # 2. HamiltonCommitmentLedger.update(allow_sync=True) matches allow_sync=False bit-for-bit.
    comp_c = torch.tensor([[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, -4.0]])
    comp_w = torch.tensor([0.25, 0.25, 0.25, 0.25])
    led_a, led_b = HamiltonCommitmentLedger(), HamiltonCommitmentLedger()
    x_pool = torch.randn(16, 2, generator=gen)
    for step in range(8):
        rows = torch.randperm(16, generator=gen)[:8]
        x_pool[rows] = x_pool[rows] + 0.4 * (comp_c[rows % 4] - x_pool[rows])
        oa = led_a.update(x_pool[rows], rows, comp_c, comp_w, 16, allow_sync=False)
        ob = led_b.update(x_pool[rows], rows, comp_c, comp_w, 16, allow_sync=True)
        assert torch.equal(oa.row_comp, ob.row_comp) and torch.equal(oa.occupancy, ob.occupancy)
        assert torch.allclose(oa.batch_target, ob.batch_target, atol=0.0)
        assert torch.allclose(oa.tail_residual, ob.tail_residual, atol=1e-6)

    # 3. Steady-state _warm_rho_batch / _rho_below_conn fast paths match fresh checkpoint reload exactly.
    modes = torch.tensor([[-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, 5.0]])
    at = PrototypeAtlas(max_prototypes=16)
    for _ in range(30):
        yb = modes[torch.randint(0, 4, (64,), generator=gen)]
        at.update(yb)
        at.coalesce(yb)
    sd = at.state_dict()
    at_reloaded = PrototypeAtlas(max_prototypes=16)
    at_reloaded.load_state_dict(sd)
    yb_next = modes[torch.randint(0, 4, (64,), generator=gen)]
    at.update(yb_next)
    at_reloaded.update(yb_next)
    c1, c2 = at.coalesce(yb_next), at_reloaded.coalesce(yb_next)
    assert abs(at.rho_atom_ema - at_reloaded.rho_atom_ema) < 1e-7
    assert abs(at.g_conn_ema - at_reloaded.g_conn_ema) < 1e-7
    assert torch.allclose(c1.comp_c, c2.comp_c, atol=1e-6)


@pytest.mark.cuda
def test_cuda_deterministic_algorithms_atlas_update_and_ledger() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        dev = torch.device("cuda:0")
        gen = torch.Generator(device=dev).manual_seed(0)
        modes = torch.tensor([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]], device=dev)
        at = PrototypeAtlas(max_prototypes=32, device=dev)
        for step in range(12):
            idx = torch.randint(0, 4 if step >= 2 else 2, (64,), device=dev, generator=gen)
            yb = modes[idx] + 0.1 * torch.randn(64, 2, device=dev, generator=gen)
            at.update(yb)
            coal = at.coalesce(yb)
        assert bool(coal.all_iso.item()) and float(torch.cdist(modes, coal.comp_c).amin(1).amax()) < 0.25
        led = at.commit_particles(yb[:16], torch.arange(16, device=dev), 32, coalesced=coal)
        assert led is not None and int(led.occupancy.sum().item()) > 0
        torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(prev)
