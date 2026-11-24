"""Unit, mathematical invariant, and checkpoint continuation tests for Axiom Module A (prior.py)."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import nn

from particlegan._axiom import prior as axiom_prior
from particlegan._axiom.prior import (
    AxiomPriorRegularizer,
    allocate_deficit_quotas,
    axiom_pullback_step,
    central_pullback_cov,
    coalesce_pythagorean_ltv,
    has_positive_prior_sigma,
    should_use_axiom_pullback,
    validate_axiom_prior_state,
)
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior
from particlegan.priors import ManipulatedPrior, NormalNoiseManipulator

PRIOR_FILE = Path(__file__).resolve().parents[1] / "particlegan" / "_axiom" / "prior.py"


def _total_scalar_trace_var(c: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Return weighted mean and exact scalar trace variance sum_k w_k (v_k + ||c_k - c_bar||^2 / D)."""
    cd, vd, wd = c.double(), v.double(), w.double()
    wd = wd / wd.sum()
    mean = (wd[:, None] * cd).sum(0)
    between = (cd - mean[None, :]).square().sum(-1) / float(cd.shape[1])
    return mean, float((wd * (vd + between)).sum().item())


def _make_mock_trainer(
    *,
    dim: int = 2,
    latent_dim: int = 2,
    num_particles: int = 48,
    sigma_z: float = 0.20,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    use_axiom_opt_protocol: bool = False,
) -> Tuple[Any, torch.Generator]:
    """Build a deterministic lightweight host matching GANTrainer's pullback contract."""
    g_rng = torch.Generator(device="cpu").manual_seed(seed)

    def _init_linear(in_f: int, out_f: int) -> nn.Linear:
        layer = nn.Linear(in_f, out_f)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(out_f, in_f, generator=g_rng) * (0.6 / math.sqrt(in_f)))
            layer.bias.copy_(torch.randn(out_f, generator=g_rng) * 0.05)
        return layer

    G = nn.Sequential(_init_linear(latent_dim, 24), nn.Tanh(), _init_linear(24, dim)).to(device)
    D = nn.Sequential(_init_linear(dim, 24), nn.Tanh(), _init_linear(24, 1)).to(device)
    prior = MoGParticlePrior(num_particles, latent_dim, sigma=sigma_z).to(device)
    with torch.no_grad():
        prior.z.copy_(torch.randn(num_particles, latent_dim, generator=g_rng).to(device))
    reg = AxiomPriorRegularizer(max_prototypes=32, bound_device=device, bound_shape=(num_particles, latent_dim))

    opt_g = torch.optim.Adam(list(G.parameters()) + [prior.z], lr=1e-2)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-2)

    class _RecordStub:
        def __init__(self) -> None:
            self.thermal_scale = 1.0
            self._disc_gate = 1.0
            self.ema_push_res = 1.0
            self._g_probe_cache = "stale"
            self.observed = 0

        def observe_margin_and_scale(self, real_logits, fake_logits, real_g, fake_g) -> None:
            del real_logits, fake_logits, real_g, fake_g
            self.observed += 1

    opt_d.record = _RecordStub()  # type: ignore[attr-defined]
    if use_axiom_opt_protocol:
        opt_g.modulations = []  # type: ignore[attr-defined]
        opt_g.origins = []  # type: ignore[attr-defined]
        opt_g.bind_origin_sample = lambda x: opt_g.origins.append(None if x is None else x.detach().clone())  # type: ignore[attr-defined]
        opt_g.bind_axiom_modulation = lambda **kw: opt_g.modulations.append(kw)  # type: ignore[attr-defined]

    recipe = SimpleNamespace(
        prior_regularizer="axiom",
        t1_pullback_weight=1.0,
        t1_iso_weight=0.5,
        t1_laguerre_lr=0.15,
        t1_prior_row_lr=0.05,
        encoder_mode="none",
        conditioning="scalar",
        loss="non_saturating",
        num_particles=num_particles,
        prior_l2=0.0,
        prior_reg=0.0,
        kinetic_transport_weight=0.0,
        kinetic_transport_local_weight=0.0,
        kinetic_transport_loss=lambda f, r: f.new_zeros(()),
        kinetic_transport_local_loss=lambda f, r: f.new_zeros(()),
    )

    class _PolicyStub:
        def __init__(self) -> None:
            self.backward_calls = 0
            self.step_calls = 0

        def after_generator_backward(self, **kwargs: Any) -> None:
            del kwargs
            self.backward_calls += 1

        def after_generator_step(self) -> None:
            self.step_calls += 1

    def _generate(gen: nn.Module, z: torch.Tensor, sigma_out: float, noise: torch.Generator, rows=None) -> torch.Tensor:
        del rows
        out = gen(z)
        if float(sigma_out) > 0.0:
            eps = torch.randn(out.shape, generator=noise, dtype=out.dtype).to(out.device)
            out = out + float(sigma_out) * eps
        return out

    trainer = SimpleNamespace(
        G=G,
        D=D,
        prior=prior,
        prior_regularizer=reg,
        opt_g=opt_g,
        opt_d=opt_d,
        recipe=recipe,
        policy=_PolicyStub(),
        _prior_penalty=False,
        _postprocess=lambda p, x: x,
        _generate=_generate,
        _batch=lambda x, nm: x,
        loss=SimpleNamespace(g_loss=lambda fl, rl: (-fl).mean()),
    )
    noise = torch.Generator(device="cpu").manual_seed(seed + 1000)
    return trainer, noise


def test_ast_hygiene_and_loc_budget() -> None:
    """Enforce zero legacy/sibling imports, zero forbidden spectral/RNG calls, and LOC <= 450."""
    source = PRIOR_FILE.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) <= 450, f"prior.py has {len(lines)} LOC, exceeding <= 450 clean target (and 480 cap)"

    tree = ast.parse(source, filename=str(PRIOR_FILE))
    forbidden_substrings = (
        "_unified", "_t1_", "_t2_", "_t3_", "_t4_", "_rhbi", "_empf",
        "_cond_fiber", "_minimal", "_bispf", "_nash_v3", "_r4_ablation",
        "bispf", "rhbi", "k3p", "grad_regularizers",
    )
    forbidden_siblings = {"critic", "transport", "optimizer", "loss"}
    allowed_internal = {"particle_prior", "priors", "optim.constraint_geometry"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                assert not any(tok in name for tok in forbidden_substrings), f"forbidden import: {name}"
                assert name != "random", "must not import Python random module"
                assert not name.startswith("particlegan._axiom."), f"forbidden sibling import: {name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(tok in mod for tok in forbidden_substrings), f"forbidden from-import: {mod}"
            assert mod != "random", "must not import from random"
            if node.level == 1:
                imported = {a.name for a in node.names} | ({mod} if mod else set())
                assert not (imported & forbidden_siblings), f"forbidden _axiom sibling import: {imported}"
            elif node.level >= 2 or mod.startswith("particlegan"):
                clean_mod = mod.removeprefix("particlegan.")
                assert clean_mod in allowed_internal, f"unexpected internal import: level={node.level} mod={mod}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {
                "eigh", "eigvalsh", "svd", "svdvals", "pinv", "compile",
                "forward_ad", "dual_level", "make_dual", "unpack_dual",
                "randn", "rand", "randint", "randperm", "normal_", "uniform_",
            }, f"forbidden attribute call/reference: .{node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "autograd":
                assert node.func.attr != "grad", "must not use torch.autograd.grad in prior.py"


def test_mass_conservation_all_transitions_including_p7_shock() -> None:
    """P7 regression control: proto_w AND w_fake_ema sum to 1.0 across every transition & shock."""
    gen = torch.Generator().manual_seed(7)
    base_data = torch.randn(64, 2, generator=gen) * 0.25

    # Case 1: Uncovered-mode teleportation with K < max_prototypes (spawn path).
    reg = AxiomPriorRegularizer(max_prototypes=64)
    reg.update_quantizer(base_data)
    k0 = len(reg.proto_c)
    assert k0 == 48 and k0 < 64
    assert abs(float(reg.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6

    shock_batch = base_data.clone()
    shock_batch[0] = torch.tensor([25.0, -25.0])
    reg.update_quantizer(shock_batch)
    assert len(reg.proto_c) == k0 + 1, "expected uncovered-mode spawn when K < max_prototypes"
    assert abs(float(reg.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg.psi.mean().item())) <= 1e-6
    assert bool((reg.proto_w >= 0.0).all() and (reg.w_fake_ema >= 0.0).all())

    # Case 2: Uncovered-mode teleportation reusing a dead slot (exact P7 violation site!).
    reg2 = AxiomPriorRegularizer(max_prototypes=64)
    reg2.update_quantizer(base_data)
    k2 = len(reg2.proto_c)
    reg2.proto_c[0] = torch.tensor([100.0, 100.0])  # deserted slot receives 0 batch hits -> stays dead
    reg2.proto_w.fill_(1.0 / (k2 - 1))
    reg2.proto_w[0] = 0.0
    reg2.proto_w = reg2.proto_w / reg2.proto_w.sum()
    reg2.w_fake_ema = torch.linspace(1.0, 2.0, k2)
    reg2.w_fake_ema = reg2.w_fake_ema / reg2.w_fake_ema.sum()
    reg2.psi = torch.linspace(-1.0, 1.0, k2)
    reg2.psi -= reg2.psi.mean()

    shock_batch2 = base_data.clone()
    shock_batch2[0] = torch.tensor([-30.0, 30.0])
    reg2.update_quantizer(shock_batch2)
    assert len(reg2.proto_c) == k2, "expected dead-slot reuse rather than growth"
    assert torch.allclose(reg2.proto_c[0], torch.tensor([-30.0, 30.0]), atol=1e-4)
    assert abs(float(reg2.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg2.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg2.psi.mean().item())) <= 1e-6

    # Case 3: Uncovered-mode teleportation at full capacity (K == max_prototypes -> LTV merge + place).
    reg3 = AxiomPriorRegularizer(max_prototypes=12)
    reg3.update_quantizer(base_data)
    assert len(reg3.proto_c) == 12
    reg3.w_fake_ema = torch.linspace(0.5, 1.5, 12)
    reg3.w_fake_ema = reg3.w_fake_ema / reg3.w_fake_ema.sum()
    shock_batch3 = base_data.clone()
    shock_batch3[0] = torch.tensor([40.0, 40.0])
    reg3.update_quantizer(shock_batch3)
    assert len(reg3.proto_c) == 12
    assert abs(float(reg3.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg3.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg3.psi.mean().item())) <= 1e-6

    # Case 4: Dead-slot split and Gersho merge-before-split both preserve exact mass on proto_w & w_fake_ema.
    reg4 = AxiomPriorRegularizer(max_prototypes=24)
    reg4.update_quantizer(base_data)
    k4 = len(reg4.proto_c)
    reg4.proto_w.fill_(1.0 / (k4 - 1))
    reg4.proto_w[1] = 0.0
    reg4.proto_w = reg4.proto_w / reg4.proto_w.sum()
    reg4.proto_var[3] = 2.5
    reg4.update_quantizer(base_data)
    assert abs(float(reg4.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg4.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg4.psi.mean().item())) <= 1e-6

    # Case 5: Gersho merge-before-split at K >= 16 with NO dead slots (high-variance outlier cell triggers need_g).
    reg5 = AxiomPriorRegularizer(max_prototypes=20)
    reg5.update_quantizer(base_data)
    assert len(reg5.proto_c) == 20
    reg5.proto_w.fill_(1.0 / 20.0)
    reg5.w_fake_ema = torch.linspace(0.5, 1.5, 20)
    reg5.w_fake_ema = reg5.w_fake_ema / reg5.w_fake_ema.sum()
    reg5.proto_var.fill_(0.05)
    reg5.proto_var[0] = 5.0  # >10x median Gersho energy
    reg5.proto_c[5] = reg5.proto_c[4] + torch.tensor([0.05, 0.0])  # guaranteed close pair for merge-before-split
    reg5.update_quantizer(base_data)
    assert float(reg5.proto_var[0].item()) < 4.5, "Gersho merge-before-split should have split cell 0"
    assert abs(float(reg5.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg5.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg5.psi.mean().item())) <= 1e-6


def test_pythagorean_ltv_coalescence_merge_and_split() -> None:
    """Verify exact scalar trace LTV on coalescence (with 2.2x & isolation guards), Gersho merge, and 0.84*v split."""
    # 1. Coalescence exact LTV on 2-cell and 3-cell sibling clusters in 2D and 4D.
    for dim in (2, 4):
        c = torch.zeros(5, dim)
        c[0, 0], c[1, 0] = -0.18, 0.22  # 2-cell sibling cluster around origin
        c[2, 0], c[3, 0], c[4, 0] = 15.0, 15.15, 15.30  # 3-cell sibling cluster far away
        v = torch.tensor([0.04, 0.06, 0.05, 0.07, 0.04])
        w = torch.tensor([0.15, 0.25, 0.20, 0.15, 0.25])
        mean_before, tv_before = _total_scalar_trace_var(c, v, w)

        c_bar, v_bar, w_mat, multicell = coalesce_pythagorean_ltv(c, v, w)
        assert bool(multicell.all()), "both isolated sibling clusters should coalesce"
        mean_after, tv_after = _total_scalar_trace_var(c_bar, v_bar, w)
        assert torch.allclose(mean_before, mean_after, atol=1e-6)
        assert abs(tv_after - tv_before) <= 1e-6 * max(1.0, tv_before), f"LTV mismatch: before={tv_before}, after={tv_after}"
        assert torch.allclose(w_mat.sum(1), torch.ones(5), atol=1e-6)

    # 2. Width-ratio guard (2.2x): overlapping cells with 10x width ratio MUST NOT coalesce!
    c_wr = torch.tensor([[0.0, 0.0], [0.05, 0.0], [20.0, 0.0]])
    v_wr = torch.tensor([0.01, 1.00, 0.01])  # sigma = [0.1, 1.0, 0.1] -> ratio = 10x > 2.2x
    w_wr = torch.tensor([0.4, 0.4, 0.2])
    c_b, v_b, _, mc_wr = coalesce_pythagorean_ltv(c_wr, v_wr, w_wr)
    assert not bool(mc_wr.any()), "10x width-ratio pair must be rejected by 2.2x guard"
    assert torch.allclose(c_b, c_wr) and torch.allclose(v_b, v_wr)

    # 3. External isolation guard: 2 siblings close to a non-coalescing 3rd cell MUST NOT coalesce.
    c_iso = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.9, 0.0]])
    v_iso = torch.tensor([0.04, 0.04, 0.001])  # 3rd cell has >2.2x width ratio so not a sibling, yet within 4.2*sqrt(D)*sig_C
    w_iso = torch.tensor([0.4, 0.4, 0.2])
    _, _, _, mc_iso = coalesce_pythagorean_ltv(c_iso, v_iso, w_iso)
    assert not bool(mc_iso.any()), "non-isolated sibling pair must be rejected by 4.2x isolation guard"

    # 4. Gersho merge (_merge_pair_ltv): fixes P7 Bug 2C(a) (updates proto_var[a] with exact LTV).
    reg = AxiomPriorRegularizer(max_prototypes=8)
    reg.quantizer_initialized = True
    reg.proto_c = torch.tensor([[0.0, 0.0], [2.0, 1.0], [10.0, -5.0]])
    reg.proto_var = torch.tensor([0.10, 0.25, 0.40])
    reg.v_push = reg.proto_var.clone()
    reg.proto_w = torch.tensor([0.30, 0.50, 0.20])
    reg.w_fake_ema = torch.tensor([0.25, 0.45, 0.30])
    reg.psi = torch.tensor([0.2, -0.3, 0.1])
    mean_b, tv_b = _total_scalar_trace_var(reg.proto_c, reg.proto_var, reg.proto_w)
    weighted_within = (0.30 * 0.10 + 0.50 * 0.25) / 0.80

    reg._merge_pair_ltv(0, 1)
    assert float(reg.proto_var[0].item()) > weighted_within + 0.5, "proto_var[a] must include between-centroid variance"
    assert float(reg.proto_w[1].item()) == 0.0 and float(reg.w_fake_ema[1].item()) == 0.0
    assert abs(float(reg.proto_w.double().sum().item()) - 1.0) <= 1e-6
    assert abs(float(reg.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
    mean_a, tv_a = _total_scalar_trace_var(reg.proto_c, reg.proto_var, reg.proto_w)
    assert torch.allclose(mean_b, mean_a, atol=1e-6)
    assert abs(tv_a - tv_b) <= 1e-6, f"Gersho merge LTV mismatch: before={tv_b}, after={tv_a}"

    # 5. Split (_split_slot_ltv): fixes P7 Bug 2C(b) (0.84*v + 0.16*v == 1.00*v) + principal axis alignment.
    theta = math.radians(35.0)
    axis = torch.tensor([math.cos(theta), math.sin(theta)])
    perp = torch.tensor([-math.sin(theta), math.cos(theta)])
    t = torch.linspace(-1.0, 1.0, 40)[:, None]
    pts = reg.proto_c[0][None, :] + 3.0 * t * axis[None, :] + 0.1 * (t.flip(0)) * perp[None, :]

    v_parent = float(reg.proto_var[0].item())
    mean_s0, tv_s0 = _total_scalar_trace_var(reg.proto_c, reg.proto_var, reg.proto_w)
    reg._split_slot_ltv(0, 1, pts)
    assert abs(float(reg.proto_var[0].item()) - 0.84 * v_parent) <= 1e-6
    assert abs(float(reg.proto_var[1].item()) - 0.84 * v_parent) <= 1e-6
    disp = reg.proto_c[1] - reg.proto_c[0]
    cos_align = abs(float((disp / disp.norm() @ axis).item()))
    assert cos_align > 0.999, f"split direction did not align with principal axis: cos={cos_align}"
    mean_s1, tv_s1 = _total_scalar_trace_var(reg.proto_c, reg.proto_var, reg.proto_w)
    assert torch.allclose(mean_s0, mean_s1, atol=1e-6)
    assert abs(tv_s1 - tv_s0) <= 1e-6, f"Split LTV mismatch: before={tv_s0}, after={tv_s1}"


def test_central_pullback_exact_linear_and_o_sigma2_nonlinear() -> None:
    """Verify exact linear anisotropic covariance, O(sigma_z^2) nonlinear slope, and bf16 grad flow."""
    gen = torch.Generator().manual_seed(11)

    # 1. Exact (< 1e-13) on linear anisotropic G across d_z in {2, 4, 8} (no d_z==4 RH epicycle).
    for d_z in (2, 4, 8):
        d_out = 3
        A = torch.randn(d_out, d_z, generator=gen, dtype=torch.float64)
        b = torch.randn(d_out, generator=gen, dtype=torch.float64)
        mu = torch.randn(5, d_z, generator=gen, dtype=torch.float64)
        sigma_z = 0.17
        cov = central_pullback_cov(lambda z, A=A, b=b: z @ A.T + b, mu, sigma_z)
        exact = (sigma_z ** 2) * (A @ A.T)[None].expand(5, d_out, d_out)
        assert float((cov - exact).abs().max().item()) < 1e-13, f"linear error for d_z={d_z}"

    # 2. High-D (d_z = 12 > 8) deterministic Hadamard-Rademacher path: exact isotropic trace & unbiased mean.
    mu12 = torch.randn(256, 12, generator=gen, dtype=torch.float64)
    cov_iso = central_pullback_cov(lambda z: 2.5 * z, mu12[:8], 0.2)
    tr_iso = cov_iso.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(tr_iso, torch.full_like(tr_iso, 12.0 * (2.5 * 0.2) ** 2), atol=1e-12)
    A12 = torch.randn(3, 12, generator=gen, dtype=torch.float64)
    cov_mean = central_pullback_cov(lambda z: z @ A12.T, mu12, 0.2).mean(0)
    exact12 = (0.2 ** 2) * (A12 @ A12.T)
    assert float((cov_mean - exact12).norm().item() / exact12.norm().item()) < 0.05

    # 3. Purely quadratic generator: central pullback is exact while one-sided forward has large O(sigma) error.
    def _quad(z: torch.Tensor) -> torch.Tensor:
        z0, z1 = z[:, 0], z[:, 1]
        return torch.stack((z0 + 0.4 * z0 * z0 + 0.3 * z1 * z1, z1 + 0.5 * z0 * z1), dim=-1)

    mu_q = torch.tensor([[0.6, -0.4], [-0.3, 0.8]], dtype=torch.float64)
    J_q = torch.stack((
        torch.stack((1.0 + 0.8 * mu_q[:, 0], 0.6 * mu_q[:, 1]), dim=-1),
        torch.stack((0.5 * mu_q[:, 1], 1.0 + 0.5 * mu_q[:, 0]), dim=-1),
    ), dim=1)
    exact_q = J_q @ J_q.transpose(-1, -2)
    cov_q = central_pullback_cov(_quad, mu_q, 0.1) / (0.1 ** 2)
    assert float((cov_q - exact_q).abs().max().item()) < 1e-13

    # 4. Genuinely nonlinear (tanh + cubic) generator: verify exact O(sigma_z^2) convergence slope (~2.0).
    def _nonlin(z: torch.Tensor) -> torch.Tensor:
        z0, z1 = z[:, 0], z[:, 1]
        return torch.stack((
            torch.tanh(0.9 * z0 + 0.3 * z1) + 0.25 * z0 ** 3 + 0.35 * z0 * z0,
            torch.tanh(-0.4 * z0 + 1.1 * z1) + 0.20 * z1 ** 3 - 0.30 * z1 * z1,
        ), dim=-1)

    mu_nl = torch.tensor([[0.45, -0.35], [-0.25, 0.55], [0.20, 0.30]], dtype=torch.float64)
    # Exact analytical Jacobian J(mu_nl).
    s0 = 1.0 - torch.tanh(0.9 * mu_nl[:, 0] + 0.3 * mu_nl[:, 1]).square()
    s1 = 1.0 - torch.tanh(-0.4 * mu_nl[:, 0] + 1.1 * mu_nl[:, 1]).square()
    J_nl = torch.stack((
        torch.stack((0.9 * s0 + 0.75 * mu_nl[:, 0] ** 2 + 0.70 * mu_nl[:, 0], 0.3 * s0), dim=-1),
        torch.stack((-0.4 * s1, 1.1 * s1 + 0.60 * mu_nl[:, 1] ** 2 - 0.60 * mu_nl[:, 1]), dim=-1),
    ), dim=1)
    exact_nl = J_nl @ J_nl.transpose(-1, -2)

    sigmas = [0.1, 0.05, 0.025, 0.0125]
    cent_errs, fwd_errs = [], []
    eye2 = torch.eye(2, dtype=torch.float64)
    for sz in sigmas:
        c_est = central_pullback_cov(_nonlin, mu_nl, sz) / (sz ** 2)
        cent_errs.append(float((c_est - exact_nl).norm().item()))
        # One-sided forward baseline for semantic control comparison.
        f0 = _nonlin(mu_nl)
        jf = torch.stack([(_nonlin(mu_nl + sz * eye2[a]) - f0) / sz for a in range(2)], dim=1)
        f_est = torch.einsum("mad,mae->mde", jf, jf)
        fwd_errs.append(float((f_est - exact_nl).norm().item()))

    cent_slopes = [math.log2(cent_errs[i] / cent_errs[i + 1]) for i in range(len(sigmas) - 1)]
    fwd_slopes = [math.log2(fwd_errs[i] / fwd_errs[i + 1]) for i in range(len(sigmas) - 1)]
    assert all(1.95 <= s <= 2.05 for s in cent_slopes), f"expected O(sigma^2) slope ~2.0, got {cent_slopes}"
    assert all(0.90 <= s <= 1.15 for s in fwd_slopes), f"semantic control: forward slope should be ~1.0, got {fwd_slopes}"
    assert cent_errs[-1] < 0.01 * fwd_errs[-1]

    # 5. Gradient flow in both float32 and bfloat16 (no underflow / NaN; output bias cancels identically).
    for dt in (torch.float32, torch.bfloat16):
        net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)).to(dtype=dt)
        mu_t = torch.randn(4, 2, generator=gen).to(dtype=dt)
        cov_t = central_pullback_cov(net, mu_t, 0.15)
        assert cov_t.dtype == torch.float32
        loss = (cov_t - 0.05 * torch.eye(2)[None]).square().mean()
        loss.backward()
        named = dict(net.named_parameters())
        for pname, param in named.items():
            assert param.grad is not None and bool(torch.isfinite(param.grad).all())
            if pname == "2.bias":
                assert float(param.grad.abs().max().item()) < 1e-8, "output bias must cancel in central difference"
            else:
                assert float(param.grad.abs().sum().item()) > 0.0, f"expected nonzero grad on {pname} ({dt})"


def test_laguerre_relative_deficit_and_unbiased_quotas() -> None:
    """Verify per-cell relative-deficit wakeup for a 1% starving mode, exact gauge under cap, and unbiased quotas."""
    k = 100
    reg = AxiomPriorRegularizer(max_prototypes=128)
    reg.quantizer_initialized = True
    grid = torch.stack(torch.meshgrid(torch.arange(10.0), torch.arange(10.0), indexing="ij"), dim=-1).reshape(k, 2)
    reg.proto_c = grid
    reg.proto_var = torch.full((k,), 0.05)
    reg.v_push = reg.proto_var.clone()
    reg.proto_w = torch.full((k,), 1.0 / k)
    reg.w_fake_ema = torch.full((k,), 1.0 / (k - 1))
    reg.w_fake_ema[-1] = 0.0  # Single starving 1% mode; global TV = 0.01 << tv_floor (~0.078)
    reg.psi = torch.zeros(k)

    # Particles cover first 99 modes only (mode 99 gets 0 particles).
    xu = grid[:99].repeat(2, 1)
    eye_k = torch.eye(k)
    r_macro = torch.tensor(3.0)
    _, g_psi = reg.update_laguerre_dual(xu, grid, eye_k, r_macro, laguerre_lr=0.15)
    assert float(g_psi.item()) > 0.90, f"1% starving mode must wake Laguerre gate (got g_psi={float(g_psi.item())})"
    assert float(reg.psi[-1].item()) > 0.05, "starving mode must receive positive Laguerre potential boost"
    assert abs(float(reg.psi.mean().item())) <= 1e-6, "psi must preserve exact zero-mean gauge"
    assert abs(float(reg.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6

    # Cap saturation + coalesced component invariance + zero-mean gauge simultaneously.
    reg.psi = torch.zeros(k)
    reg.psi[0] = 1e6
    reg.psi[1] = 1e6
    w_mat = eye_k.clone()
    w_mat[0, :2] = w_mat[1, :2] = 0.5
    _, _ = reg.update_laguerre_dual(xu, grid, w_mat, r_macro, laguerre_lr=0.15)
    cap = 2.5 * 2.0 * float(r_macro.square().item()) + 0.5
    assert float(reg.psi.abs().amax().item()) <= cap + 1e-5
    assert abs(float(reg.psi.mean().item())) <= 1e-6
    assert abs(float((reg.psi[0] - reg.psi[1]).item())) <= 1e-6, "coalesced component psi must stay equal under cap"

    # Equilibrium dead-zone: exact match gives g_psi == 0.0.
    reg_eq = AxiomPriorRegularizer(max_prototypes=16)
    reg_eq.quantizer_initialized = True
    reg_eq.proto_c = grid[:8].clone()
    reg_eq.proto_var = torch.full((8,), 0.05)
    reg_eq.v_push = reg_eq.proto_var.clone()
    reg_eq.proto_w = torch.full((8,), 0.125)
    reg_eq.w_fake_ema = torch.full((8,), 0.125)
    reg_eq.psi = torch.zeros(8)
    _, g_eq = reg_eq.update_laguerre_dual(grid[:8].repeat(16, 1), grid[:8], torch.eye(8), r_macro, 0.15)
    assert float(g_eq.item()) == 0.0

    # Unbiased B_f < K quota allocation (no -1e-7 * arange(k) low-index bias).
    pw = torch.full((100,), 0.01)
    wf = torch.full((100,), 0.01)
    q_unif = allocate_deficit_quotas(pw, wf, 64)
    assert int(q_unif.sum().item()) == 64 and int(q_unif.max().item()) == 1
    assert int(q_unif[64:].sum().item()) >= 15, f"high indices starved: {int(q_unif[64:].sum().item())} slots"

    # Starving high-index cells strictly prioritized.
    wf_starve = pw.clone()
    wf_starve[95:] = 0.0
    wf_starve[:95] = 1.0 / 95.0
    q_def = allocate_deficit_quotas(pw, wf_starve, 64)
    assert int(q_def.sum().item()) == 64 and bool((q_def[95:] == 1).all())

    # n_slots >= K proportional + remainder allocation.
    q_large = allocate_deficit_quotas(pw, wf_starve, 256)
    assert int(q_large.sum().item()) == 256 and bool((q_large >= 2).all()) and bool((q_large <= 3).all())


def test_regularizer_forward_and_dispatch_predicate() -> None:
    """Verify standalone IMQ forward (differentiable, thermal ST, subsampling) and pullback dispatch predicate."""
    reg = AxiomPriorRegularizer(weight=1.0, target_std=1.0)
    z = torch.randn(320, 4, requires_grad=True)
    loss = reg(z)
    assert loss.ndim == 0 and bool(torch.isfinite(loss))
    loss.backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all()) and float(z.grad.abs().sum().item()) > 0.0

    # Thermal straight-through scales gradient while keeping forward value unchanged.
    z1 = z.detach().clone().requires_grad_(True)
    l1 = reg(z1)
    l1.backward()
    g1 = z1.grad.clone()
    reg.thermal_source = SimpleNamespace(thermal_scale=0.25)
    z2 = z.detach().clone().requires_grad_(True)
    l2 = reg(z2)
    l2.backward()
    assert torch.allclose(l1.detach(), l2.detach(), atol=1e-6)
    assert torch.allclose(z2.grad, 0.25 * g1, atol=1e-5)

    assert float(AxiomPriorRegularizer(weight=0.0)(z.detach()).item()) == 0.0
    assert float(reg(torch.zeros(1, 4)).item()) == 0.0

    # Dispatch predicate checks.
    trainer, _ = _make_mock_trainer(dim=2, sigma_z=0.2)
    real_2d = torch.randn(32, 2)
    real_const = torch.full((32, 2), 3.0)
    assert has_positive_prior_sigma(trainer.prior)
    assert should_use_axiom_pullback(trainer, real_2d)

    # Zero-sigma prior: False on non-degenerate 2D, True on degenerate constant point-mass (Case A).
    trainer.prior = ParticlePrior(32, 2)
    assert not has_positive_prior_sigma(trainer.prior)
    assert not should_use_axiom_pullback(trainer, real_2d)
    assert should_use_axiom_pullback(trainer, real_const)

    # Wrapped NormalNoiseManipulator with sigma > 0 -> positive sigma True.
    wrapped = ManipulatedPrior(ParticlePrior(32, 2), [NormalNoiseManipulator(0.15)])
    assert has_positive_prior_sigma(wrapped)

    # Negative guards: images, encoder, non-scalar conditioning, other regularizer, zero weight.
    trainer.prior = MoGParticlePrior(32, 2, sigma=0.2)
    assert not should_use_axiom_pullback(trainer, torch.randn(8, 3, 8, 8))
    trainer.recipe.encoder_mode = "amortized"
    assert not should_use_axiom_pullback(trainer, real_2d)
    trainer.recipe.encoder_mode = "none"
    trainer.recipe.conditioning = "class"
    assert not should_use_axiom_pullback(trainer, real_2d)
    trainer.recipe.conditioning = "scalar"
    trainer.recipe.prior_regularizer = "unified_v11"
    assert not should_use_axiom_pullback(trainer, real_2d)
    trainer.recipe.prior_regularizer = "axiom"
    trainer.recipe.t1_pullback_weight = 0.0
    assert not should_use_axiom_pullback(trainer, real_2d)


def test_pullback_step_case_a_case_b_and_bitexact_checkpoint(tmp_path: Path) -> None:
    """Verify Case A convergence, Case B zero-RNG-drift + bit-exact checkpoint continuation, and damage control."""
    # --- Case A: Point-mass target ---
    tr_a, noise_a = _make_mock_trainer(dim=2, latent_dim=2, num_particles=32, sigma_z=0.0, seed=13, use_axiom_opt_protocol=True)
    target = torch.tensor([2.5, -1.5])
    real_a = target[None].expand(32, 2).clone()
    idx_a = torch.arange(32)
    rng_before = torch.get_rng_state()
    init_err = float((tr_a.G(tr_a.prior.z).detach() - real_a).norm(dim=-1).mean().item())
    for _ in range(12):
        lat_a = tr_a.prior.z[idx_a]
        loss_g, loss_gan, prior_diag, pm, tr, tr_l = axiom_pullback_step(
            tr_a, real_a, None, lat_a, idx_a, sigma_out=0.0, noise=noise_a,
            critic=tr_a.D, loss_d=torch.tensor(1.0), penalty=torch.tensor(0.1),
        )
        assert bool(torch.isfinite(loss_g) and torch.isfinite(loss_gan) and torch.isfinite(prior_diag))
        assert pm is None and float(tr.item()) == 0.0 and float(tr_l.item()) == 0.0
    final_err = float((tr_a.G(tr_a.prior.z).detach() - real_a).norm(dim=-1).mean().item())
    assert final_err < init_err, f"Case A point-mass failed to move toward target: {init_err} -> {final_err}"
    assert torch.equal(rng_before, torch.get_rng_state()), "Case A must not consume global RNG"
    assert len(tr_a.opt_g.modulations) == 12 and tr_a.opt_d.record.observed == 12

    # --- Case B: 4-mode MoG + exact fresh-object checkpoint continuation ---
    centers = torch.tensor([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, 3.0]])
    data_rng = torch.Generator().manual_seed(42)
    batches = []
    for _ in range(8):
        c_idx = torch.randint(0, 4, (48,), generator=data_rng)
        xr = centers[c_idx] + 0.25 * torch.randn(48, 2, generator=data_rng)
        ix = torch.arange(48)
        eps = torch.randn(48, 2, generator=data_rng)
        batches.append((xr, ix, eps))

    def _run_step(tr: Any, ns: torch.Generator, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        xr, ix, eps = batch
        lat = tr.prior.means()[ix] + tr.prior._cached_sigma_float() * eps
        return axiom_pullback_step(
            tr, xr, None, lat, ix, sigma_out=0.0, noise=ns,
            critic=tr.D, loss_d=torch.tensor(1.0), penalty=torch.tensor(0.1),
        )

    # Uninterrupted 8-step reference run.
    tr_ref, ns_ref = _make_mock_trainer(dim=2, latent_dim=2, num_particles=48, sigma_z=0.20, seed=21)
    rng_ref0 = torch.get_rng_state()
    ref_out = None
    for b_item in batches:
        ref_out = _run_step(tr_ref, ns_ref, b_item)
        assert abs(float(tr_ref.prior_regularizer.proto_w.double().sum().item()) - 1.0) <= 1e-6
        assert abs(float(tr_ref.prior_regularizer.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6
        assert abs(float(tr_ref.prior_regularizer.psi.mean().item())) <= 1e-6
    assert torch.equal(rng_ref0, torch.get_rng_state()), "Case B must not consume global RNG"

    # Interrupted run: 4 steps -> save to disk -> fresh objects -> load -> 4 steps.
    tr_part, ns_part = _make_mock_trainer(dim=2, latent_dim=2, num_particles=48, sigma_z=0.20, seed=21)
    for b_item in batches[:4]:
        _run_step(tr_part, ns_part, b_item)
    ckpt_path = tmp_path / "axiom_prior_ckpt.pt"
    torch.save({
        "reg": tr_part.prior_regularizer.state_dict_checkpoint(),
        "reg_alias": tr_part.prior_regularizer.state_dict(),
        "G": tr_part.G.state_dict(),
        "D": tr_part.D.state_dict(),
        "prior": tr_part.prior.state_dict(),
        "opt_g": tr_part.opt_g.state_dict(),
        "noise": ns_part.get_state(),
    }, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=True)
    tr_res, ns_res = _make_mock_trainer(dim=2, latent_dim=2, num_particles=48, sigma_z=0.20, seed=999)
    # validate_only=True must NOT mutate fresh state.
    tr_res.prior_regularizer.load_state_dict_checkpoint(loaded["reg"], validate_only=True)
    assert not tr_res.prior_regularizer.quantizer_initialized and tr_res.prior_regularizer.proto_c is None
    tr_res.prior_regularizer.load_state_dict(loaded["reg_alias"])
    tr_res.G.load_state_dict(loaded["G"])
    tr_res.D.load_state_dict(loaded["D"])
    tr_res.prior.load_state_dict(loaded["prior"])
    tr_res.opt_g.load_state_dict(loaded["opt_g"])
    ns_res.set_state(loaded["noise"])

    res_out = None
    for b_item in batches[4:]:
        res_out = _run_step(tr_res, ns_res, b_item)

    assert torch.equal(res_out[0], ref_out[0]), f"loss_g drift: {res_out[0]} vs {ref_out[0]}"
    assert torch.equal(res_out[2], ref_out[2]), f"prior_diag drift: {res_out[2]} vs {ref_out[2]}"
    assert torch.equal(tr_res.prior.z, tr_ref.prior.z)
    for p_res, p_ref in zip(tr_res.G.parameters(), tr_ref.G.parameters()):
        assert torch.equal(p_res, p_ref)
    s_res, s_ref = tr_res.prior_regularizer.state_dict_checkpoint(), tr_ref.prior_regularizer.state_dict_checkpoint()
    assert s_res.keys() == s_ref.keys()
    for k_name, v_ref in s_ref.items():
        if isinstance(v_ref, torch.Tensor):
            assert torch.equal(s_res[k_name], v_ref), f"checkpoint tensor mismatch on {k_name}"
        else:
            assert s_res[k_name] == v_ref, f"checkpoint scalar mismatch on {k_name}"

    # Semantic deliberate-damage control: corrupting restored proto_c must visibly alter step loss.
    tr_dmg, ns_dmg = _make_mock_trainer(dim=2, latent_dim=2, num_particles=48, sigma_z=0.20, seed=999)
    tr_dmg.prior_regularizer.load_state_dict_checkpoint(loaded["reg"])
    tr_dmg.G.load_state_dict(loaded["G"])
    tr_dmg.D.load_state_dict(loaded["D"])
    tr_dmg.prior.load_state_dict(loaded["prior"])
    tr_dmg.opt_g.load_state_dict(loaded["opt_g"])
    ns_dmg.set_state(loaded["noise"])
    tr_dmg.prior_regularizer.proto_c.add_(100.0)
    dmg_out = _run_step(tr_dmg, ns_dmg, batches[4])
    assert abs(float(dmg_out[0].item()) - float(res_out[0].item())) > 1.0


def test_checkpoint_validator_rejects_all_tampering() -> None:
    """Reject every schema, type, finiteness, sign, mass-conservation, and shape tampering."""
    reg = AxiomPriorRegularizer(max_prototypes=16, bound_shape=(12, 2))
    empty_state = reg.state_dict_checkpoint()
    assert validate_axiom_prior_state(empty_state) is empty_state

    reg.update_quantizer(torch.randn(24, 2, generator=torch.Generator().manual_seed(3)))
    reg.row_m = torch.zeros(12, 2)
    reg.row_v = torch.zeros(12, 1)
    reg.row_n = torch.zeros(12, 1)
    reg.rows_initialized = True
    good = reg.state_dict_checkpoint()
    assert validate_axiom_prior_state(good) is good

    def _bad(mutate) -> Dict[str, Any]:
        st = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in good.items()}
        mutate(st)
        return st

    bad_cases = [
        "not_a_dict",
        _bad(lambda s: s.__setitem__("schema", 2)),
        _bad(lambda s: s.__setitem__("schema", True)),
        _bad(lambda s: s.pop("psi")),
        _bad(lambda s: s.__setitem__("extra_key", 0)),
        _bad(lambda s: s.__setitem__("max_prototypes", 1)),
        _bad(lambda s: s.__setitem__("res_ema", -0.1)),
        _bad(lambda s: s.__setitem__("iso_ema", float("nan"))),
        _bad(lambda s: s.__setitem__("iso_err_ema", float("inf"))),
        _bad(lambda s: s.__setitem__("quantizer_initialized", False)),  # non-None quantizer tensors
        _bad(lambda s: s.__setitem__("rows_initialized", False)),  # non-None row tensors
        _bad(lambda s: s["proto_var"].__setitem__(0, -1e-3)),
        _bad(lambda s: s["v_push"].__setitem__(0, -1e-3)),
        _bad(lambda s: s["psi"].__setitem__(0, float("nan"))),
        _bad(lambda s: s["proto_w"].mul_(1.05)),  # breaks mass conservation
        _bad(lambda s: s["w_fake_ema"].mul_(0.90)),  # breaks mass conservation
        _bad(lambda s: s["proto_w"].__setitem__(0, -0.01)),
        _bad(lambda s: s["row_v"].__setitem__((0, 0), -1.0)),
        _bad(lambda s: s["row_n"].__setitem__((0, 0), float("inf"))),
        _bad(lambda s: s.__setitem__("proto_c", torch.zeros(20, 2))),  # exceeds max_prototypes=16
    ]
    for case in bad_cases:
        with pytest.raises(ValueError):
            validate_axiom_prior_state(case)

    # Configuration/shape mismatches rejected by load_state_dict_checkpoint.
    reg_other_cap = AxiomPriorRegularizer(max_prototypes=32)
    with pytest.raises(ValueError):
        reg_other_cap.load_state_dict_checkpoint(good)

    reg_other_shape = AxiomPriorRegularizer(max_prototypes=16, bound_shape=(8, 2))
    with pytest.raises(ValueError):
        reg_other_shape.load_state_dict_checkpoint(good)

    # Invalid constructor hyperparameters rejected.
    with pytest.raises(ValueError):
        AxiomPriorRegularizer(max_prototypes=1)
    with pytest.raises(ValueError):
        AxiomPriorRegularizer(weight=-1.0)
    with pytest.raises(ValueError):
        AxiomPriorRegularizer(target_std=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_quantizer_and_pullback_sanity() -> None:
    """GPU sanity check: quantizer shock, central pullback, full step, and CPU->CUDA checkpoint restore."""
    dev = torch.device("cuda:0")
    tr, ns = _make_mock_trainer(dim=2, latent_dim=2, num_particles=48, sigma_z=0.20, device=dev, seed=5)
    g_cpu = torch.Generator().manual_seed(55)
    xr = torch.randn(48, 2, generator=g_cpu).to(dev)
    ix = torch.arange(48, device=dev)
    eps = torch.randn(48, 2, generator=g_cpu).to(dev)
    for step in range(3):
        xr_s = xr.clone()
        if step == 2:
            xr_s[0] = torch.tensor([35.0, -35.0], device=dev)
        lat = tr.prior.means()[ix] + tr.prior._cached_sigma_float() * eps
        loss_g, loss_gan, prior_diag, _, _, _ = axiom_pullback_step(
            tr, xr_s, None, lat, ix, sigma_out=0.0, noise=ns,
            critic=tr.D, loss_d=torch.tensor(1.0, device=dev), penalty=torch.tensor(0.1, device=dev),
        )
        assert loss_g.device.type == "cuda" and bool(torch.isfinite(loss_g) and torch.isfinite(loss_gan) and torch.isfinite(prior_diag))
        assert abs(float(tr.prior_regularizer.proto_w.double().sum().item()) - 1.0) <= 1e-6
        assert abs(float(tr.prior_regularizer.w_fake_ema.double().sum().item()) - 1.0) <= 1e-6

    # CPU-serialized checkpoint loads cleanly onto CUDA bound_device.
    ckpt = tr.prior_regularizer.state_dict_checkpoint()
    assert ckpt["proto_c"].device.type == "cpu"
    reg_cuda = AxiomPriorRegularizer(max_prototypes=32, bound_device=dev, bound_shape=(48, 2))
    reg_cuda.load_state_dict_checkpoint(ckpt)
    assert reg_cuda.proto_c.device == dev and torch.equal(reg_cuda.proto_c, tr.prior_regularizer.proto_c)
    assert axiom_prior.__all__  # exercise module export list
