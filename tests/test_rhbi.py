"""Unit and semantic-control tests for Relativistic Hellinger-Born-Infeld Transport (RHBI)."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import (
    GANTrainer,
    InputNoise,
    ParticlePrior,
    RHBICriticAdam,
    RHBICriticPenalty,
    RHBIGANLoss,
    RHBILoss,
    bind_rhbi_record,
    get_recipe,
    init,
)
from particlegan.rhbi import RHBIStepRecord


def _make_mlp_pair(z_dim: int = 2, x_dim: int = 2, hidden: int = 32, seed: int = 0):
    p_rng = torch.Generator().manual_seed(seed + 2)
    G = nn.Sequential(nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, x_dim))
    D = nn.Sequential(nn.Linear(x_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
    init.deterministic_orthogonal_(G, seed=seed)
    init.deterministic_orthogonal_(D, seed=seed + 1)
    prior = ParticlePrior(num_particles=64, z_dim=z_dim)
    with torch.no_grad():
        prior.z.normal_(0.0, 1.0, generator=p_rng)
    return G, D, prior


def test_invariant_1_zero_strength_identity():
    """weight=0.0 leaves buffers untouched and matches pure baseline trajectory bit-for-bit."""
    op = RHBILoss(weight=0.0)
    x_fake = torch.randn(16, 4, requires_grad=True)
    y_real = torch.randn(16, 4)
    d_fake = torch.randn(16, requires_grad=True)
    d_real = torch.randn(16)

    loss, metrics = op(x_fake, y_real, d_fake, d_real)
    assert not bool(op.ell_initialized.item())
    assert float(op.ell_ema.item()) == 0.0
    assert loss.item() == 0.0
    assert metrics["rhbi_active_rate"] == 0.0

    # 10-step trajectory equivalence: bcap vs bcap with rhbi_weight=0.0
    data_rng_a = torch.Generator().manual_seed(42)
    data_rng_b = torch.Generator().manual_seed(42)
    G_a, D_a, P_a = _make_mlp_pair(seed=10)
    G_b, D_b, P_b = deepcopy(G_a), deepcopy(D_a), deepcopy(P_a)

    rec_base = get_recipe("bcap", total_steps=20, batch_size=16, num_particles=64, z_dim=2)
    rec_zero = get_recipe("bcap", total_steps=20, batch_size=16, num_particles=64, z_dim=2, rhbi_weight=0.0)

    tr_a = GANTrainer(rec_base, G_a, D_a, prior=P_a, seed=0)
    tr_b = GANTrainer(rec_zero, G_b, D_b, prior=P_b, seed=0)
    for _ in range(10):
        batch_a = torch.randn(16, 2, generator=data_rng_a)
        batch_b = torch.randn(16, 2, generator=data_rng_b)
        tr_a.step(batch_a)
        tr_b.step(batch_b)

    for pa, pb in zip(G_a.parameters(), G_b.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(D_a.parameters(), D_b.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(P_a.z, P_b.z)


def test_invariant_2_rng_isolation():
    """RHBILoss forward + backward consumes zero global CPU or CUDA RNG draws."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        op = RHBILoss(weight=1.0).to(dev)
        x_fake = torch.randn(32, 3, device=dev, requires_grad=True)
        y_real = torch.randn(32, 3, device=dev)
        d_fake = torch.randn(32, device=dev, requires_grad=True)
        d_real = torch.randn(32, device=dev)

        cpu_before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

        loss, _ = op(x_fake, y_real, d_fake, d_real)
        loss.backward()

        cpu_after = torch.get_rng_state()
        assert torch.equal(cpu_before, cpu_after)
        if torch.cuda.is_available():
            cuda_after = torch.cuda.get_rng_state_all()
            for sb, sa in zip(cuda_before, cuda_after):
                assert torch.equal(sb, sa)


def test_invariant_3_permutation_equivariance_and_logit_shift_invariance():
    """Row permutation equivariance and +1000.0 constant critic logit shift invariance."""
    gen = torch.Generator().manual_seed(7)
    x_base = torch.randn(24, 5, generator=gen)
    y_real = torch.randn(24, 5, generator=gen)
    df_base = torch.randn(24, generator=gen)
    dr_base = torch.randn(24, generator=gen)
    perm = torch.randperm(24, generator=gen)

    op1 = RHBILoss(weight=1.0)
    x1 = x_base.clone().requires_grad_(True)
    df1 = df_base.clone().requires_grad_(True)
    loss1, _ = op1(x1, y_real, df1, dr_base)
    loss1.backward()

    # Permuted rows + constant +1000.0 shift on all critic logits
    op2 = RHBILoss(weight=1.0)
    x2 = x_base[perm].clone().requires_grad_(True)
    df2 = (df_base[perm] + 1000.0).clone().requires_grad_(True)
    dr2 = dr_base + 1000.0
    loss2, _ = op2(x2, y_real, df2, dr2)
    loss2.backward()

    assert torch.allclose(loss1, loss2, rtol=1e-4, atol=5e-4)
    assert torch.allclose(x1.grad[perm], x2.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(df1.grad[perm], df2.grad, rtol=1e-4, atol=1e-4)


def test_invariant_4_outlier_coincident_and_singleton_finiteness():
    """float32 finiteness at 1e30 outliers, coincident points x == y, and B = 1."""
    # Case A: Extreme outlier 1e30 in float32
    op = RHBILoss(weight=1.0)
    y_real = torch.randn(16, 4, dtype=torch.float32)
    x_out = y_real.clone()
    x_out[0] = 1e30
    x_out.requires_grad_(True)
    d_fake = torch.zeros(16, dtype=torch.float32, requires_grad=True)
    d_real = torch.zeros(16, dtype=torch.float32)

    loss_out, _ = op(x_out, y_real, d_fake, d_real)
    loss_out.backward()
    assert torch.isfinite(loss_out)
    assert torch.isfinite(x_out.grad).all()
    assert torch.isfinite(d_fake.grad).all()
    assert abs(float(d_fake.grad[0].item())) > 0.01

    # Case A2: Distant tunneling outlier has active transport velocity strictly in (0.5, 1.0]
    op_tun = RHBILoss(weight=1.0)
    x_tun = y_real.clone()
    x_tun[0] = 50.0
    x_tun.requires_grad_(True)
    df_tun = torch.zeros(16, dtype=torch.float32)
    df_tun[0] = -20.0
    df_tun.requires_grad_(True)
    dr_tun = torch.full((16,), 20.0, dtype=torch.float32)
    loss_tun, _ = op_tun(x_tun, y_real, df_tun, dr_tun)
    loss_tun.backward()
    assert 0.5 < float(x_tun.grad[0].norm().item()) * 16.0 <= 1.0

    # Case B: Exact coincident points x_fake == y_real (no 0/0 NaNs)
    op_coin = RHBILoss(weight=1.0)
    x_same = y_real.clone().requires_grad_(True)
    d_f2 = torch.zeros(16, dtype=torch.float32, requires_grad=True)
    loss_same, _ = op_coin(x_same, y_real, d_f2, d_real)
    loss_same.backward()
    assert torch.isfinite(loss_same)
    assert torch.isfinite(x_same.grad).all()

    # Case C: Singleton batch B = 1 with hollow self-coupling
    op_one = RHBILoss(weight=1.0)
    x_one = torch.tensor([[1.0, -2.0]], requires_grad=True)
    y_one = torch.tensor([[0.5, 0.5]])
    df_one = torch.tensor([0.2], requires_grad=True)
    dr_one = torch.tensor([-0.1])
    loss_one, _ = op_one(x_one, y_one, df_one, dr_one)
    loss_one.backward()
    assert torch.isfinite(loss_one)
    assert torch.isfinite(x_one.grad).all()


def test_invariant_5_singleton_equilibrium_bypass_100_gaussians():
    """On 100-Gaussians at B=64 with flat critic, >85% of unmatched singleton mass routes to slack."""
    gen = torch.Generator().manual_seed(0)
    coords = torch.linspace(-9.0, 9.0, 10)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [100, 2], spacing 2.0

    B = 64
    idx_r = torch.randint(0, 100, (B,), generator=gen)
    idx_f = torch.randint(0, 100, (B,), generator=gen)
    y_real = centers[idx_r] + 0.05 * torch.randn(B, 2, generator=gen)
    x_fake = (centers[idx_f] + 0.05 * torch.randn(B, 2, generator=gen)).requires_grad_(True)
    d_fake = torch.zeros(B, requires_grad=True)
    d_real = torch.zeros(B)

    op = RHBILoss(weight=1.0)
    # Warm ell_t for a few real draws from the same 100-mode law so ell_t settles on intra-mode scale
    for _ in range(15):
        idx_w = torch.randint(0, 100, (B,), generator=gen)
        y_w = centers[idx_w] + 0.05 * torch.randn(B, 2, generator=gen)
        op._update_ell(y_w)

    loss, _ = op(x_fake, y_real, d_fake, d_real)
    loss.backward()

    with torch.no_grad():
        nn_to_real = torch.cdist(x_fake.detach(), y_real).amin(dim=1)
        unmatched = nn_to_real > 1.0  # No real draw in the same mode
        assert int(unmatched.sum().item()) >= 15
        # Each row's total marginal is 1/B; slack fraction for row i is B * P_xy[i, B].
        slack_frac = B * op.last_p_xy_slack[unmatched]
        assert float(slack_frac.mean().item()) > 0.85


@pytest.mark.parametrize("family", ["dualnorm", "adam", "formulation"])
def test_exact_checkpoint_continuation_and_tamper_rejection(family: str):
    """Exact bit-for-bit continuation across ell_initialized boundary + tamper rejection."""
    if family == "dualnorm":
        recipe = get_recipe("rhbi", total_steps=20, batch_size=16, num_particles=64, z_dim=2)
    elif family == "adam":
        recipe = get_recipe(
            "bcap_adam",
            critic_formulation="rhbi",
            reg_arm=None,
            rhbi_weight=1.0,
            ebi_beta=4.0,
            total_steps=20,
            batch_size=16,
            num_particles=64,
            z_dim=2,
        )
    else:
        recipe = get_recipe(
            "bispf",
            critic_formulation="rhbi",
            prior_regularizer="moment_tether",
            rhbi_weight=1.0,
            ebi_beta=4.0,
            total_steps=20,
            batch_size=16,
            num_particles=64,
            z_dim=2,
        )

    G1, D1, P1 = _make_mlp_pair(seed=21)
    G2, D2, P2 = _make_mlp_pair(seed=99)  # Distinct initial weights before load_state_dict

    batches = [torch.randn(16, 2, generator=torch.Generator().manual_seed(100 + k)) for k in range(6)]

    tr1 = GANTrainer(recipe, G1, D1, prior=P1, seed=0)
    for b in batches[:3]:
        tr1.step(b)
    ckpt = tr1.state_dict()
    assert tr1.opt_d.record.rhbi_ell_initialized is True
    ell_mid = tr1.opt_d.record.rhbi_ell_ema
    assert ell_mid > 0.0

    for b in batches[3:]:
        tr1.step(b)

    tr2 = GANTrainer(recipe, G2, D2, prior=P2, seed=55)
    # Tampered checkpoint missing RHBI observer/controller fields must be rejected
    bad_ckpt = deepcopy(ckpt)
    if family == "formulation":
        bad_ckpt["optimizers"][1]["regularizer"]["record"]["formulation"] = "bispf"
    else:
        bad_ckpt["optimizers"][1]["regularizer"]["record"].pop("rhbi", None)
    with pytest.raises(ValueError):
        tr2.load_state_dict(bad_ckpt)

    tr2.load_state_dict(ckpt)
    assert tr2.opt_d.record.rhbi_ell_ema == ell_mid
    for b in batches[3:]:
        tr2.step(b)

    for p1, p2 in zip(G1.parameters(), G2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(P1.z, P2.z)
    assert tr1.opt_d.record.rhbi_ell_ema == tr2.opt_d.record.rhbi_ell_ema


def test_hook_capture_views_input_noise_joint_bigan_and_discrete_cond():
    """Verify ring capture across .squeeze(-1).detach(), InputNoise, joint_g_loss, and class conditioning."""
    recipe = get_recipe("rhbi", total_steps=10, batch_size=8, num_particles=32, z_dim=2)

    class CondJointCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(6, 1)
            self.emb = nn.Embedding(4, 2)

        def forward(self, x, z, cls):
            return self.net(torch.cat([x, z, self.emb(cls)], dim=-1))

    D = CondJointCritic()
    opt_d = recipe.make_critic_optimizer(D)
    loss_fn = recipe.make_loss()
    bind_rhbi_record(opt_d.record, loss_fn)
    assert isinstance(loss_fn, RHBIGANLoss)

    # 1. InputNoise records clean x rather than noisy x
    noisy_D = InputNoise(D, std=0.5, generator=torch.Generator().manual_seed(3))
    x_clean = torch.ones(8, 2, requires_grad=True)
    z_clean = torch.zeros(8, 2, requires_grad=True)
    cls_a = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = noisy_D(x_clean, z_clean, cls_a).squeeze(-1)
    x_cap, _, cond_cap, _, _ = loss_fn._resolve_pair(out, out.detach())
    assert torch.equal(x_cap[:, :2], x_clean)
    assert torch.equal(cond_cap.squeeze(-1), cls_a)

    # 2. joint_g_loss delivers finite nonzero gradients to both G and E streams,
    #    even when logits are .squeeze(-1) views
    G = nn.Linear(2, 2)
    E = nn.Linear(2, 2)
    z = torch.randn(8, 2)
    real_x = torch.randn(8, 2)
    fake_x = G(z)
    enc_z = E(real_x)
    fake_logits = D(fake_x, z, cls_a).squeeze(-1)
    real_logits = D(real_x, enc_z, cls_a).squeeze(-1)
    j_loss = loss_fn.joint_g_loss(fake_logits, real_logits)
    j_loss.backward()
    assert G.weight.grad is not None and float(G.weight.grad.abs().sum()) > 0.0
    assert E.weight.grad is not None and float(E.weight.grad.abs().sum()) > 0.0

    # 3. Disjoint discrete class labels route 100% of active coupling into slack
    op = RHBILoss(weight=1.0)
    _, _ = op(
        torch.randn(8, 2, requires_grad=True),
        torch.randn(8, 2),
        torch.zeros(8, requires_grad=True),
        torch.zeros(8),
        cond_fake=torch.zeros(8, dtype=torch.long),
        cond_real=torch.ones(8, dtype=torch.long),
    )
    assert torch.allclose(op.last_p_xy_slack * 8.0, torch.ones(8), atol=1e-5)


def test_semantic_learning_and_zero_preset_hash_drift():
    """Verify RHBI learns a 2D target, deliberate damage degrades quality, and existing presets have zero drift."""
    for preset in ("ka2", "k3p", "bcap", "bispf", "bispf_optimized", "e22", "atlas", "halloween"):
        saved = get_recipe(preset).to_dict()
        assert not any(k.startswith("rhbi_") for k in saved), f"{preset} leaked rhbi_* into to_dict()"

    # Train RHBI briefly on a shifted 2D target and verify deliberate weight damage degrades MSE
    target_mean = torch.tensor([2.5, -2.0])
    G, D, prior = _make_mlp_pair(seed=5)
    recipe = get_recipe("rhbi", total_steps=250, batch_size=32, num_particles=64, z_dim=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
    data_rng = torch.Generator().manual_seed(11)
    for _ in range(250):
        batch = target_mean + 0.15 * torch.randn(32, 2, generator=data_rng)
        trainer.step(batch)

    eval_rng = torch.Generator().manual_seed(999)
    samples = trainer.sample(256, generator=eval_rng)
    trained_err = float((samples.mean(dim=0) - target_mean).norm().item())
    assert trained_err < 0.6

    # Deliberate model damage check (AGENTS.md semantic control)
    with torch.no_grad():
        for p in trainer.G.parameters():
            p.mul_(-3.0).add_(4.0)
    damaged_samples = trainer.sample(256, generator=torch.Generator().manual_seed(999))
    damaged_err = float((damaged_samples.mean(dim=0) - target_mean).norm().item())
    assert damaged_err > trained_err + 1.5
    assert isinstance(RHBICriticAdam, type)
    assert isinstance(RHBICriticPenalty, type)


def test_polar_cpu_dense_profile_checkpoint_and_parity():
    """Verify polar_cpu_v1 matches exact CPU polar updates and enforces checkpoint profile identity."""
    from particlegan.optim.dualnorm import NormalizedOptimizer, polar_factor

    torch.manual_seed(7)
    w1 = nn.Parameter(torch.randn(16, 8))
    b1 = nn.Parameter(torch.randn(16))
    w2 = nn.Parameter(w1.detach().clone())
    b2 = nn.Parameter(b1.detach().clone())

    opt_prof = NormalizedOptimizer(
        [w1, b1], lr=0.05, smoothing=0.001, dense_profile="polar_cpu_v1"
    )
    opt_ref = NormalizedOptimizer(
        [w2, b2], lr=0.05, smoothing=0.001, svd_backend="cpu"
    )
    g_w = torch.randn_like(w1)
    g_b = torch.randn_like(b1)
    w1.grad, b1.grad = g_w.clone(), g_b.clone()
    w2.grad, b2.grad = g_w.clone(), g_b.clone()
    opt_prof.step()
    opt_ref.step()
    assert torch.equal(w1, w2)
    assert torch.equal(b1, b2)
    assert opt_prof.state_dict()["dualnorm"]["dense_profile"] == "polar_cpu_v1"
    with pytest.raises(ValueError, match="dense profile"):
        opt_ref.load_state_dict(opt_prof.state_dict())
    assert polar_factor(g_w, smoothing=0.001, backend="cpu").shape == w1.shape


def test_rhbi_v2_linear_screening_and_feydy_self_coupling():
    """Verify linear Born-Infeld screening valley routing and Feydy diagonal self-coupling."""
    # 1. Midpoint bridge particle between two distant modes (c_hat ~ 13 >> rho=3)
    #    routes >99.99% to slack under linear screening vs partial active pull under rapidity.
    y_modes = torch.cat([
        torch.tensor([[-1.0, 0.0]]).expand(8, 2) + 0.02 * torch.randn(8, 2, generator=torch.Generator().manual_seed(1)),
        torch.tensor([[1.0, 0.0]]).expand(8, 2) + 0.02 * torch.randn(8, 2, generator=torch.Generator().manual_seed(2)),
    ], dim=0)
    x_bridge = y_modes.clone()
    x_bridge[0] = torch.tensor([0.0, 0.0])  # Symmetric valley midpoint
    d_zero_f = torch.zeros(16, requires_grad=True)
    d_zero_r = torch.zeros(16)

    op_lin = RHBILoss(weight=1.0, rho=3.0, tau=2.0, rapidity_screening=False, self_coupling="feydy")
    op_rap = RHBILoss(weight=1.0, rho=3.0, tau=2.0, rapidity_screening=True, self_coupling="hollow")
    for _ in range(10):
        op_lin._update_ell(y_modes)
        op_rap._update_ell(y_modes)

    loss_lin, _ = op_lin(x_bridge.clone().requires_grad_(True), y_modes, d_zero_f, d_zero_r)
    loss_lin.backward()
    slack_bridge_lin = float(16.0 * op_lin.last_p_xy_slack[0].item())
    assert slack_bridge_lin > 0.9999

    _, _ = op_rap(x_bridge.clone().requires_grad_(True), y_modes, torch.zeros(16, requires_grad=True), d_zero_r)
    slack_bridge_rap = float(16.0 * op_rap.last_p_xy_slack[0].item())
    assert slack_bridge_lin > slack_bridge_rap

    # 2. Feydy self-coupling absorbs isolated singleton self-mass on the zero-cost diagonal
    #    instead of forcing off-diagonal derangement or slack leakage.
    pts_isolated = torch.arange(8, dtype=torch.float32).unsqueeze(1) * 50.0
    ell = torch.tensor(1.0)
    _, c_hat_iso = RHBILoss._born_infeld_pair(pts_isolated, pts_isolated, ell)
    zeros = torch.zeros(8)
    P_feydy = op_lin._sinkhorn_slack(c_hat_iso, zeros, zeros, hollow=False)
    diag_mass = P_feydy[:8, :8].diagonal() * 8.0
    off_diag = P_feydy[:8, :8].clone().fill_diagonal_(0.0).sum(dim=1) * 8.0
    assert torch.all(diag_mass > 0.99)
    assert torch.all(off_diag < 1e-4)


def test_rhbi_v2_penalty_lambda_floor_and_gamma_gauge_concat_parity():
    """Verify lambda_floor and gamma_gauge match across concatenated and split critic passes."""
    from particlegan._rhbi.penalty import _RHBIPenaltyKernel

    torch.manual_seed(19)
    D1 = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
    with torch.no_grad():
        D1[-1].bias.fill_(3.5)  # Nonzero logit offset to activate gamma_gauge
    D2 = deepcopy(D1)

    x_real = torch.randn(12, 4)
    x_fake = torch.randn(12, 4)

    kern1 = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=20.0, lambda_floor=0.005, gamma_gauge=0.02)
    kern2 = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=20.0, lambda_floor=0.005, gamma_gauge=0.02)

    pen_cat, stats_cat = kern1.penalty(D1, x_real, x_fake, step=1, collect_stats=True, allow_concat=True)
    pen_sep, stats_sep = kern2.penalty(D2, x_real, x_fake, step=1, collect_stats=True, allow_concat=False)
    assert torch.allclose(pen_cat, pen_sep, rtol=1e-5, atol=1e-6)
    assert stats_cat["applied"] and stats_sep["applied"]

    pen_cat.backward()
    pen_sep.backward()
    non_none = 0
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        if p1.grad is None:
            assert p2.grad is None
        else:
            non_none += 1
            assert torch.allclose(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)
    assert non_none >= 3
    # Bias gradient must push positive logit offset back toward zero
    assert float(D1[-1].bias.grad.item()) > 0.01


def test_polar_cpu_v1_nondefault_smoothing_and_rhbi_v1_backward_compat():
    """Verify polar_cpu_v1 supports custom smoothing while rhbi preset defaults stay unchanged."""
    from particlegan.optim.dualnorm import NormalizedOptimizer

    rec_v1 = get_recipe("rhbi")
    assert rec_v1.rhbi_self_coupling == "hollow"
    assert rec_v1.rhbi_rapidity_screening is True
    assert rec_v1.rhbi_relativistic_weight == 0.0
    assert rec_v1.rhbi_lambda_floor == 0.0
    assert rec_v1.rhbi_gamma_gauge == 0.0

    rec_v2 = get_recipe(
        "rhbi",
        optimizer_dense_profile="polar_cpu_v1",
        optimizer_smoothing=0.0061,
        rhbi_rapidity_screening=False,
        rhbi_self_coupling="feydy",
        rhbi_lambda_floor=0.005,
    )
    assert rec_v2.optimizer_smoothing == pytest.approx(0.0061)
    with pytest.raises(ValueError, match=r"smoothing \.001"):
        get_recipe("rhbi", optimizer_dense_profile="gram_fp64_guarded_v1", optimizer_smoothing=0.0061)

    torch.manual_seed(31)
    w_a = nn.Parameter(torch.randn(12, 6))
    w_b = nn.Parameter(w_a.detach().clone())
    g = torch.randn_like(w_a)
    w_a.grad, w_b.grad = g.clone(), g.clone()
    NormalizedOptimizer([w_a], lr=0.014, smoothing=0.0061, dense_profile="polar_cpu_v1").step()
    NormalizedOptimizer([w_b], lr=0.014, smoothing=0.0061, svd_backend="cpu").step()
    assert torch.equal(w_a, w_b)


def test_rhbi_v3_stateful_view_wrapper_penalty_handles_batch_bound_closures():
    """Verify Recipe.make_critic_penalty avoids batch-doubling concat on trajectory-style stateful view modules."""
    class _CondCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))

        def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([slow, fast], dim=-1)).squeeze(-1)

    class _BatchStatefulView(nn.Module):
        def __init__(self, critic: nn.Module, slow: torch.Tensor) -> None:
            super().__init__()
            self.critic = critic
            self.slow = slow

        def forward(self, fast: torch.Tensor) -> torch.Tensor:
            return self.critic(self.slow, fast)

    torch.manual_seed(41)
    D1 = _CondCritic()
    D2 = deepcopy(D1)
    slow = torch.randn(12, 3)
    real = torch.randn(12, 3)
    fake = torch.randn(12, 3)

    recipe = get_recipe(
        "rhbi", reg_every=1, rhbi_lambda_floor=0.005, total_steps=10, batch_size=12, num_particles=32, z_dim=2
    )
    opt1 = recipe.make_critic_optimizer(D1)
    opt2 = recipe.make_critic_optimizer(D2)
    bound1 = recipe.make_critic_penalty(opt1)
    bound2 = recipe.make_critic_penalty(opt2)

    view = _BatchStatefulView(D1, slow)
    pen_view = bound1(view, real, fake)
    pen_direct, _ = bound2.regularizer.penalty(
        lambda x: D2(slow, x), real, fake, step=1, collect_stats=False, allow_concat=False
    )
    assert torch.allclose(pen_view, pen_direct, rtol=1e-5, atol=1e-6)
    pen_view.backward()
    pen_direct.backward()
    non_none = 0
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        if p1.grad is None:
            assert p2.grad is None
        else:
            non_none += 1
            assert torch.allclose(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)
    assert non_none >= 3


def test_rhbi_v3_conditional_fibre_slack_invariance_and_unconditional_parity():
    """Verify rhbi_conditional_fibre='slack' matches 'joint' unconditionally and cancels fibre baselines."""
    rec_default = get_recipe("rhbi")
    assert rec_default.rhbi_conditional_fibre == "joint"
    assert rec_default.rhbi_scale_log_odds is False

    # 1. Unconditional parity: 'slack' and 'joint' produce bit-identical losses and gradients.
    torch.manual_seed(53)
    D_j = nn.Sequential(nn.Linear(3, 12), nn.SiLU(), nn.Linear(12, 1))
    D_s = deepcopy(D_j)
    r_j = get_recipe("rhbi", rhbi_conditional_fibre="joint", total_steps=10, batch_size=16, num_particles=32, z_dim=2)
    r_s = get_recipe("rhbi", rhbi_conditional_fibre="slack", total_steps=10, batch_size=16, num_particles=32, z_dim=2)
    opt_j = r_j.make_critic_optimizer(D_j)
    opt_s = r_s.make_critic_optimizer(D_s)
    loss_j = r_j.make_loss()
    loss_s = r_s.make_loss()
    bind_rhbi_record(opt_j.record, loss_j)
    bind_rhbi_record(opt_s.record, loss_s)

    x_r = torch.randn(16, 3)
    x_f_a = torch.randn(16, 3, requires_grad=True)
    x_f_b = x_f_a.detach().clone().requires_grad_(True)
    la = loss_j.g_loss(D_j(x_f_a), D_j(x_r))
    la.backward()
    lb = loss_s.g_loss(D_s(x_f_b), D_s(x_r))
    lb.backward()
    assert torch.equal(la, lb)
    assert torch.equal(x_f_a.grad, x_f_b.grad)

    # 2. Continuous conditioning: arbitrary fibre-dependent additive baseline b(c) cancels identically.
    class _FibreCritic(nn.Module):
        def __init__(self, bias_scale: float) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)
            self.bias_scale = bias_scale

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            shift = self.bias_scale * cond.square().sum(dim=-1, keepdim=True)
            return self.lin(torch.cat([x, cond], dim=-1)) + shift

    Dc1 = _FibreCritic(bias_scale=0.0)
    Dc2 = _FibreCritic(bias_scale=25.0)
    Dc2.lin.load_state_dict(Dc1.lin.state_dict())
    opt_c1 = r_s.make_critic_optimizer(Dc1)
    opt_c2 = r_s.make_critic_optimizer(Dc2)
    l_c1 = r_s.make_loss()
    l_c2 = r_s.make_loss()
    bind_rhbi_record(opt_c1.record, l_c1)
    bind_rhbi_record(opt_c2.record, l_c2)

    cond = torch.randn(16, 2)
    yr = torch.randn(16, 2)
    xf1 = torch.randn(16, 2, requires_grad=True)
    xf2 = xf1.detach().clone().requires_grad_(True)
    gc1 = l_c1.g_loss(Dc1(xf1, cond), Dc1(yr, cond))
    gc1.backward()
    gc2 = l_c2.g_loss(Dc2(xf2, cond), Dc2(yr, cond))
    gc2.backward()
    assert torch.allclose(gc1, gc2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(xf1.grad, xf2.grad, rtol=1e-5, atol=1e-6)


def test_rhbi_v4_akt_horizon_1d_scaling_and_high_dim_identity():
    """Verify AKT empirical OT scaling multiplies 1D horizon by sqrt(B_y) and is exact identity for d >= 2."""
    gen = torch.Generator().manual_seed(71)
    y_1d = 2.0 + 0.5 * torch.randn(64, 1, generator=gen)
    q_base_1d = RHBILoss._batch_ell_quantile_tensor(y_1d, 0.25, akt_horizon=False)
    q_akt_1d = RHBILoss._batch_ell_quantile_tensor(y_1d, 0.25, akt_horizon=True)
    # (q - 1e-6) scales by exact sqrt(64) = 8.0 in d = 1
    assert torch.allclose(q_akt_1d - 1e-6, 8.0 * (q_base_1d - 1e-6), rtol=1e-5, atol=1e-7)

    for dim in (2, 4, 12):
        y_nd = torch.randn(32, dim, generator=gen)
        x_a = torch.randn(32, dim, generator=gen).requires_grad_(True)
        x_b = x_a.detach().clone().requires_grad_(True)
        df_a = torch.randn(32, generator=gen).requires_grad_(True)
        df_b = df_a.detach().clone().requires_grad_(True)
        dr = torch.randn(32, generator=gen)
        op_off = RHBILoss(weight=1.0, akt_horizon=False)
        op_on = RHBILoss(weight=1.0, akt_horizon=True)
        la, ma = op_off(x_a, y_nd, df_a, dr)
        lb, mb = op_on(x_b, y_nd, df_b, dr)
        la.backward()
        lb.backward()
        assert torch.equal(la, lb)
        assert torch.equal(x_a.grad, x_b.grad)
        assert ma["rhbi_ell"] == mb["rhbi_ell"]


def test_rhbi_v4_slack_gain_and_geodesic_chord_concat_parity():
    """Verify slack_gain boosts unmatched particles only and chord_weight matches across concat/split passes."""
    from particlegan._rhbi.penalty import _RHBIPenaltyKernel

    # 1. Slack gain boosts distant unmatched particle d_fake gradient while matched mode stays unchanged.
    y_cl = torch.zeros(8, 2) + 0.01 * torch.randn(8, 2, generator=torch.Generator().manual_seed(3))
    x_cl = y_cl.clone()
    x_cl[0] = torch.tensor([25.0, 25.0])  # Far into slack
    df1 = torch.zeros(8, requires_grad=True)
    df2 = torch.zeros(8, requires_grad=True)
    dr = torch.zeros(8)

    op0 = RHBILoss(weight=1.0, rho=2.0, tau=2.0, rapidity_screening=False, slack_gain=0.0)
    op1 = RHBILoss(weight=1.0, rho=2.0, tau=2.0, rapidity_screening=False, slack_gain=0.5)
    for _ in range(5):
        op0._update_ell(y_cl)
        op1._update_ell(y_cl)
    l0, _ = op0(x_cl.clone().requires_grad_(True), y_cl, df1, dr)
    l1, _ = op1(x_cl.clone().requires_grad_(True), y_cl, df2, dr)
    l0.backward()
    l1.backward()
    assert abs(float(df2.grad[0].item())) > 1.45 * abs(float(df1.grad[0].item()))

    # 2. Geodesic Born-Infeld chord penalty matches across allow_concat=True and allow_concat=False
    #    and consumes zero global RNG draws.
    torch.manual_seed(83)
    D1 = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
    D2 = deepcopy(D1)
    xr = torch.randn(10, 3)
    xf = torch.randn(10, 3)

    rng_before = torch.get_rng_state()
    k1 = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=15.0, lambda_floor=0.005, chord_weight=0.25)
    k2 = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=15.0, lambda_floor=0.005, chord_weight=0.25)
    k1.record.rhbi_ell_ema = 0.35
    k1.record.rhbi_ell_initialized = True
    k2.record.rhbi_ell_ema = 0.35
    k2.record.rhbi_ell_initialized = True

    p_cat, _ = k1.penalty(D1, xr, xf, step=1, collect_stats=True, allow_concat=True, is_conditioned=False)
    p_sep, _ = k2.penalty(D2, xr, xf, step=1, collect_stats=True, allow_concat=False, is_conditioned=False)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert k1.record.rhbi_chord_phase == pytest.approx(k2.record.rhbi_chord_phase)
    assert k1.record.rhbi_chord_phase > 0.0
    assert torch.allclose(p_cat, p_sep, rtol=1e-5, atol=1e-6)
    p_cat.backward()
    p_sep.backward()
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        if p1.grad is None:
            assert p2.grad is None
        else:
            assert torch.allclose(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["dualnorm", "formulation"])
def test_rhbi_v4_thermal_settling_shock_reopening_and_continuation(family: str):
    """Verify self-calibrated KH thermal cooling, 1-step shock reopening, and exact checkpoint continuation."""
    rec_test = RHBIStepRecord()
    op_th = RHBILoss(weight=1.0, rho=3.0, rapidity_screening=False, thermal_tau=0.25)
    gen = torch.Generator().manual_seed(91)
    # Matched draws -> residual settles near 0 and thermal_scale cools below 1.0
    for _ in range(12):
        y_m = torch.randn(64, 2, generator=gen)
        x_m = torch.randn(64, 2, generator=gen).requires_grad_(True)
        loss_m, _ = op_th(x_m, y_m, torch.zeros(64, requires_grad=True), torch.zeros(64), record=rec_test)
        loss_m.backward()
    cooled_s = rec_test.rhbi_thermal_scale
    assert cooled_s < 0.5

    # Sudden distribution shift shock -> 1-step peak-reset reopens thermal_scale to 1.0 immediately
    y_shift = 20.0 + torch.randn(64, 2, generator=gen)
    x_old = torch.randn(64, 2, generator=gen).requires_grad_(True)
    loss_sh, _ = op_th(x_old, y_shift, torch.zeros(64, requires_grad=True), torch.zeros(64), record=rec_test)
    loss_sh.backward()
    assert rec_test.rhbi_thermal_scale == pytest.approx(1.0)

    # Calibrated conditional fibre matches declared GANLoss proper scoring rule on continuous conditions
    r_cal = get_recipe(
        "rhbi",
        loss="non_saturating",
        rhbi_conditional_fibre="calibrated",
        total_steps=10,
        batch_size=8,
        num_particles=16,
        z_dim=2,
    )
    loss_cal = r_cal.make_loss()
    D_cond = nn.Linear(4, 1)
    opt_dc = r_cal.make_critic_optimizer(D_cond)
    bind_rhbi_record(opt_dc.record, loss_cal)

    class _TwoArgCritic(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return self.base(torch.cat([x, c], dim=-1))

    Dc_wrap = _TwoArgCritic(D_cond)
    opt_wrap = r_cal.make_critic_optimizer(Dc_wrap)
    bind_rhbi_record(opt_wrap.record, loss_cal)
    c_in = torch.randn(8, 2)
    xr_in = torch.randn(8, 2)
    xf_in = torch.randn(8, 2, requires_grad=True)
    fl = Dc_wrap(xf_in, c_in)
    rl = Dc_wrap(xr_in, c_in).detach()
    assert torch.allclose(loss_cal.g_loss(fl, rl), torch.nn.functional.softplus(-fl).mean())

    # Exact continuation across active chord_weight + thermal_tau + akt_horizon
    base_preset = "rhbi" if family == "dualnorm" else "bispf"
    extra = {} if family == "dualnorm" else {"critic_formulation": "rhbi", "prior_regularizer": "moment_tether", "rhbi_weight": 1.0, "ebi_beta": 12.0}
    recipe = get_recipe(
        base_preset,
        **extra,
        rhbi_akt_horizon=True,
        rhbi_slack_gain=0.25,
        rhbi_chord_weight=0.2,
        rhbi_thermal_tau=0.2,
        total_steps=16,
        batch_size=16,
        num_particles=64,
        z_dim=2,
    )
    G1, D1, P1 = _make_mlp_pair(seed=101)
    G2, D2, P2 = _make_mlp_pair(seed=202)
    batches = [0.2 * torch.randn(16, 2, generator=torch.Generator().manual_seed(300 + i)) for i in range(6)]

    tr1 = GANTrainer(recipe, G1, D1, prior=P1, seed=0)
    for b in batches[:3]:
        tr1.step(b)
    ckpt = tr1.state_dict()
    assert tr1.opt_d.record.rhbi_chord_phase > 0.0
    assert tr1.opt_d.record.rhbi_thermal_initialized is True

    # Tampered chord_phase >= 1.0 must be rejected
    bad_rec = deepcopy(tr1.opt_d.record.state_dict())
    if family == "formulation":
        bad_rec["chord_phase"] = 1.5
    else:
        bad_rec["rhbi"]["chord_phase"] = 1.5
    tr2 = GANTrainer(recipe, G2, D2, prior=P2, seed=7)
    with pytest.raises(ValueError, match="chord_phase"):
        tr2.opt_d.record.load_state_dict(bad_rec)
    bad = deepcopy(ckpt)
    bad["optimizers"][1]["regularizer"]["record"] = bad_rec
    with pytest.raises(ValueError, match="chord_phase|invalid checkpoint"):
        tr2.load_state_dict(bad)

    for b in batches[3:]:
        tr1.step(b)
    tr2.load_state_dict(ckpt)
    for b in batches[3:]:
        tr2.step(b)
    for p1, p2 in zip(G1.parameters(), G2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(P1.z, P2.z)
    assert tr1.opt_d.record.rhbi_chord_phase == tr2.opt_d.record.rhbi_chord_phase
    assert tr1.opt_d.record.rhbi_thermal_scale == tr2.opt_d.record.rhbi_thermal_scale


def test_rhbi_v4_hooked_thermal_scaling_preserves_first_order_impedance():
    """Verify hooked RHBIGANLoss scales both coordinate OT and critic-slack gradients uniformly by s^1 (never s^2)."""
    torch.manual_seed(131)
    D_hot = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    D_cool = deepcopy(D_hot)
    r_th = get_recipe(
        "rhbi",
        rhbi_rapidity_screening=False,
        rhbi_self_coupling="feydy",
        rhbi_slack_gain=0.35,
        rhbi_thermal_tau=0.22,
        total_steps=10,
        batch_size=16,
        num_particles=32,
        z_dim=2,
    )
    opt_hot = r_th.make_critic_optimizer(D_hot)
    opt_cool = r_th.make_critic_optimizer(D_cool)
    loss_hot = r_th.make_loss()
    loss_cool = r_th.make_loss()
    bind_rhbi_record(opt_hot.record, loss_hot)
    bind_rhbi_record(opt_cool.record, loss_cool)

    xr = torch.randn(16, 2)
    # Mix intra-mode active pairs and far slack-routed outliers so both OT and critic-slack terms are nonzero.
    xf_base = xr.clone()
    xf_base[:6] = xf_base[:6] + 15.0

    for opt, s_val in ((opt_hot, 1.0), (opt_cool, 0.25)):
        opt.record.rhbi_ell_ema = 0.4
        opt.record.rhbi_ell_initialized = True
        opt.record.rhbi_thermal_scale = s_val
        opt.record._in_protected_probe = True

    xf_hot = xf_base.clone().requires_grad_(True)
    xf_cool = xf_base.clone().requires_grad_(True)
    loss_hot.g_loss(D_hot(xf_hot), D_hot(xr)).backward()
    loss_cool.g_loss(D_cool(xf_cool), D_cool(xr)).backward()
    assert xf_hot.grad is not None and xf_cool.grad is not None
    assert float(xf_hot.grad.norm().item()) > 1e-3
    assert torch.allclose(xf_cool.grad, 0.25 * xf_hot.grad, rtol=1e-5, atol=1e-6)


def test_rhbi_v5_weibull_akt_shock_mu_settle_beta_and_row_local_fibre_blend():
    """Verify rhbi-v5 Weibull AKT horizon, false-shock filter, settling beta, row-local fibre blend, and continuation."""
    import math
    from particlegan._rhbi.penalty import _RHBIPenaltyKernel

    # 1. Weibull AKT scales d=1 by sqrt((2*e*B)/-ln(1-q)) while preserving exact 1.0 identity on d >= 2.
    y1d = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    q_off_1d = RHBILoss._batch_ell_quantile_tensor(y1d, 0.25, akt_horizon=False, akt_mode="weibull")
    q_sqrt_1d = RHBILoss._batch_ell_quantile_tensor(y1d, 0.25, akt_horizon=True, akt_mode="sqrt")
    q_wb_1d = RHBILoss._batch_ell_quantile_tensor(y1d, 0.25, akt_horizon=True, akt_mode="weibull")
    expected_wb_ratio = math.sqrt((2.0 * math.e * 64.0) / (-math.log(0.75)))
    assert float((q_sqrt_1d - 1e-6) / (q_off_1d - 1e-6)) == pytest.approx(8.0, rel=1e-5)
    assert float((q_wb_1d - 1e-6) / (q_off_1d - 1e-6)) == pytest.approx(expected_wb_ratio, rel=1e-5)

    y2d = torch.randn(64, 2, generator=torch.Generator().manual_seed(42))
    y3d = torch.randn(64, 3, generator=torch.Generator().manual_seed(43))
    assert torch.equal(
        RHBILoss._batch_ell_quantile_tensor(y2d, 0.25, akt_horizon=True, akt_mode="sqrt"),
        RHBILoss._batch_ell_quantile_tensor(y2d, 0.25, akt_horizon=True, akt_mode="weibull"),
    )
    assert torch.equal(
        RHBILoss._batch_ell_quantile_tensor(y3d, 0.25, akt_horizon=False),
        RHBILoss._batch_ell_quantile_tensor(y3d, 0.25, akt_horizon=True, akt_mode="weibull"),
    )

    # 2. False-shock filter (shock_mu_ratio=0.35) rejects moderate stationary batch jitter (mu_sq=0.06)
    #    that resets shock_mu_ratio=0.15, while still reopening in 1 step on real shifts (mu_sq=0.25).
    rec_15 = RHBIStepRecord()
    rec_35 = RHBIStepRecord()
    for rec in (rec_15, rec_35):
        rec.rhbi_thermal_initialized = True
        rec.rhbi_ema_ed = 0.01
        rec.rhbi_ema_mu_sq = 0.0004
        rec.rhbi_ema_spread = 0.01
        rec.rhbi_ema_q90 = 0.01
        rec.rhbi_ema_hmax = 0.01
        rec.rhbi_thermal_scale = 1e-6
    s_15 = rec_15.observe_rhbi_thermal(0.02, 0.06, 0.01, 0.01, 0.01, 0.084, shock_mu_ratio=0.15)
    s_35 = rec_35.observe_rhbi_thermal(0.02, 0.06, 0.01, 0.01, 0.01, 0.084, shock_mu_ratio=0.35)
    assert s_15 == pytest.approx(1.0)
    assert s_35 < 1e-3
    assert rec_35.rhbi_ema_mu_sq < 0.003 < rec_15.rhbi_ema_mu_sq
    # Genuine mean shift (mu_sq = 0.25 > 0.35 * 0.30 = 0.105) reopens immediately in 1 step:
    s_35_shift = rec_35.observe_rhbi_thermal(0.45, 0.25, 0.05, 0.20, 0.20, 0.084, shock_mu_ratio=0.35)
    assert s_35_shift == pytest.approx(1.0)

    # 3. State-driven settling Lipschitz sharpness tightens sub-threshold softplus wall without mutating ebi_beta.
    D_lin = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        D_lin.weight.copy_(torch.tensor([[0.95, 0.0]]))
    xr_p = torch.randn(8, 2, generator=torch.Generator().manual_seed(51))
    xf_p = torch.randn(8, 2, generator=torch.Generator().manual_seed(52))
    k_base = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=12.0, settle_beta=0.0)
    k_settle = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=12.0, settle_beta=22.0)
    for k in (k_base, k_settle):
        k.record.rhbi_thermal_initialized = True
        k.record.rhbi_ema_ed = 0.02
        k.record.rhbi_ema_mu_sq = 0.0001
        k.record.rhbi_ema_spread = 0.01
        k.record.rhbi_ema_q90 = 0.02
        k.record.rhbi_ema_hmax = 0.02
    pen_base, _ = k_base.penalty(D_lin, xr_p, xf_p, step=1, allow_concat=True)
    pen_settle, _ = k_settle.penalty(D_lin, xr_p, xf_p, step=1, allow_concat=True)
    assert k_settle.ebi_beta == 12.0
    # At ||grad|| = 0.95 < kappa = 1.0, sharper beta reduces softplus spillover below kappa!
    assert float(pen_settle.item()) < float(pen_base.item())

    # 4. Row-local calibrated-relativistic fibre blend is strictly row-independent (perturbing row j != i
    #    leaves row i's gradient unchanged) and aligns d_loss sensitivity with paired differences.
    class _CondCritic(nn.Module):
        def __init__(self, net: nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([x, c], dim=-1))

    r_fib = get_recipe(
        "rhbi",
        rhbi_conditional_fibre="slack",
        rhbi_fibre_rel_blend=0.7,
        total_steps=10,
        batch_size=8,
        num_particles=16,
        z_dim=2,
    )
    Dc = _CondCritic(nn.Linear(4, 1))
    opt_fib = r_fib.make_critic_optimizer(Dc)
    loss_fib = r_fib.make_loss()
    bind_rhbi_record(opt_fib.record, loss_fib)

    c_cond = torch.randn(8, 2, generator=torch.Generator().manual_seed(61))
    xr_a = torch.randn(8, 2, generator=torch.Generator().manual_seed(62)).requires_grad_(True)
    xf_a = torch.randn(8, 2, generator=torch.Generator().manual_seed(63)).requires_grad_(True)
    loss_a = loss_fib.d_loss(Dc(xr_a, c_cond), Dc(xf_a, c_cond))
    ga_r, ga_f = torch.autograd.grad(loss_a, (xr_a, xf_a))

    xr_b = xr_a.detach().clone()
    xf_b = xf_a.detach().clone()
    xr_b[1:] = xr_b[1:] + 7.5
    xf_b[1:] = xf_b[1:] - 4.2
    xr_b.requires_grad_(True)
    xf_b.requires_grad_(True)
    loss_b = loss_fib.d_loss(Dc(xr_b, c_cond), Dc(xf_b, c_cond))
    gb_r, gb_f = torch.autograd.grad(loss_b, (xr_b, xf_b))
    assert torch.allclose(ga_r[0], gb_r[0], rtol=1e-6, atol=1e-6)
    assert torch.allclose(ga_f[0], gb_f[0], rtol=1e-6, atol=1e-6)

    # 5. Exact checkpoint continuation across active rhbi-v5 settling boundaries in GANTrainer.
    r_v5 = get_recipe(
        "rhbi",
        rhbi_rapidity_screening=False,
        rhbi_self_coupling="feydy",
        rhbi_akt_horizon=True,
        rhbi_akt_mode="weibull",
        rhbi_slack_gain=0.35,
        rhbi_chord_weight=0.25,
        rhbi_gamma_gauge=0.03,
        rhbi_lambda_floor=0.005,
        ebi_beta=12.0,
        rhbi_settle_beta=22.0,
        rhbi_thermal_tau=0.20,
        rhbi_thermal_shock_mu=0.35,
        rhbi_conditional_fibre="slack",
        rhbi_fibre_rel_blend=0.7,
        total_steps=12,
        batch_size=16,
        num_particles=64,
        z_dim=2,
    )
    G1, D1, P1 = _make_mlp_pair(seed=401)
    G2, D2, P2 = _make_mlp_pair(seed=402)
    batches_1d = [0.25 * torch.randn(16, 2, generator=torch.Generator().manual_seed(500 + i)) for i in range(6)]
    t1 = GANTrainer(r_v5, G1, D1, prior=P1, seed=0)
    for b in batches_1d[:3]:
        t1.step(b)
    saved = deepcopy(t1.state_dict())
    for b in batches_1d[3:]:
        t1.step(b)

    t2 = GANTrainer(r_v5, G2, D2, prior=P2, seed=9)
    t2.load_state_dict(saved)
    for b in batches_1d[3:]:
        t2.step(b)
    for p1, p2 in zip(G1.parameters(), G2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(P1.z, P2.z)


def test_rhbi_v6_thermal_coupled_settling_and_continuation() -> None:
    """Verify rhbi-v6 thermal-coupled settling is exact no-op at s_thermal=1.0, sharpens on cooling, and checkpoints cleanly."""
    from particlegan._rhbi.penalty import _RHBIPenaltyKernel

    with pytest.raises(ValueError, match="rhbi_settle_mode"):
        get_recipe("rhbi", rhbi_settle_mode="invalid")

    D_lin = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        D_lin.weight.copy_(torch.tensor([[0.95, 0.0]]))
    xr_p = torch.randn(8, 2, generator=torch.Generator().manual_seed(71))
    xf_p = torch.randn(8, 2, generator=torch.Generator().manual_seed(72))

    k_base = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=12.0, settle_beta=0.0)
    k_th_hot = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=12.0, settle_beta=16.0, settle_mode="thermal")
    k_th_cool = _RHBIPenaltyKernel(coeff=1.0, kappa=1.0, ebi_beta=12.0, settle_beta=16.0, settle_mode="thermal")
    for k in (k_base, k_th_hot, k_th_cool):
        k.record.rhbi_thermal_initialized = True
        k.record.rhbi_ema_ed = 0.14  # < 0.22 (would trigger residual mode) but > 0.084 (s_thermal = 1.0)
        k.record.rhbi_ema_mu_sq = 0.01
        k.record.rhbi_ema_spread = 0.10
        k.record.rhbi_ema_q90 = 0.14
        k.record.rhbi_ema_hmax = 0.14
    k_th_hot.record._thermal_scale = 1.0
    k_th_cool.record._thermal_scale = 0.2

    pen_base, _ = k_base.penalty(D_lin, xr_p, xf_p, step=1, allow_concat=True)
    pen_hot, _ = k_th_hot.penalty(D_lin, xr_p, xf_p, step=1, allow_concat=True)
    pen_cool, _ = k_th_cool.penalty(D_lin, xr_p, xf_p, step=1, allow_concat=True)
    assert torch.equal(pen_hot, pen_base)
    assert float(pen_cool.item()) < float(pen_base.item()) * 0.2

    r_v6 = get_recipe(
        "rhbi",
        rhbi_rapidity_screening=False,
        rhbi_self_coupling="feydy",
        rhbi_akt_horizon=True,
        rhbi_akt_mode="weibull",
        rhbi_slack_gain=0.35,
        rhbi_chord_weight=0.25,
        rhbi_gamma_gauge=0.03,
        rhbi_lambda_floor=0.005,
        ebi_beta=12.0,
        rhbi_settle_beta=16.0,
        rhbi_settle_mode="thermal",
        rhbi_thermal_tau=0.20,
        rhbi_thermal_shock_mu=0.35,
        rhbi_conditional_fibre="slack",
        rhbi_fibre_rel_blend=0.0,
        total_steps=12,
        batch_size=16,
        num_particles=64,
        z_dim=2,
    )
    G1, D1, P1 = _make_mlp_pair(seed=601)
    G2, D2, P2 = _make_mlp_pair(seed=602)
    batches = [0.20 * torch.randn(16, 2, generator=torch.Generator().manual_seed(700 + i)) for i in range(6)]
    t1 = GANTrainer(r_v6, G1, D1, prior=P1, seed=0)
    for b in batches[:3]:
        t1.step(b)
    saved = deepcopy(t1.state_dict())
    for b in batches[3:]:
        t1.step(b)

    t2 = GANTrainer(r_v6, G2, D2, prior=P2, seed=9)
    t2.load_state_dict(saved)
    for b in batches[3:]:
        t2.step(b)
    for p1, p2 in zip(G1.parameters(), G2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(P1.z, P2.z)


def test_rhbi_v7_step_damp_and_conditional_fiber_transport():
    """Verify thermal step damping, multi-part conditional fiber transport, unbiasedness, and continuation."""
    r_nodamp = get_recipe(
        "rhbi",
        constraint_geometry_mode="none",
        rhbi_step_damp=0.0,
        total_steps=10,
        batch_size=8,
        num_particles=32,
        z_dim=2,
    )
    r_damp = get_recipe(
        "rhbi",
        constraint_geometry_mode="none",
        rhbi_step_damp=0.5,
        total_steps=10,
        batch_size=8,
        num_particles=32,
        z_dim=2,
    )
    Ga, Da, Pa = _make_mlp_pair(seed=801)
    Gb, Db, Pb = deepcopy(Ga), deepcopy(Da), deepcopy(Pa)
    Gc, Dc, Pc = deepcopy(Ga), deepcopy(Da), deepcopy(Pa)

    opt_ga, opt_da = r_nodamp.make_optimizers(Ga, Da, Pa)
    opt_gb, opt_db = r_damp.make_optimizers(Gb, Db, Pb)
    opt_gc, opt_dc = r_damp.make_optimizers(Gc, Dc, Pc)
    opt_db.record._thermal_scale = 1.0
    opt_dc.record._thermal_scale = 0.0

    idx = torch.arange(8)
    for G_m, P_m, opt_g_m in ((Ga, Pa, opt_ga), (Gb, Pb, opt_gb), (Gc, Pc, opt_gc)):
        opt_g_m.zero_grad()
        opt_g_m.set_sampled_rows(P_m.z, idx)
        loss = G_m(P_m.z[idx]).square().sum()
        loss.backward()
        opt_g_m.step()

    assert torch.equal(Pa.z, Pb.z)
    P0 = _make_mlp_pair(seed=801)[2]
    assert torch.allclose(Pc.z[idx] - P0.z[idx], 0.5 * (Pa.z[idx] - P0.z[idx]), atol=1e-6)

    class CondCriticFirst(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(5, 1)

        def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.lin(torch.cat([x, xt, t[:, None].to(x)], dim=1)).squeeze(-1)

    class CondCriticSecond(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, slow: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.lin(torch.cat([slow, x], dim=1)).squeeze(-1)

    r_cond = get_recipe(
        "rhbi",
        rhbi_conditional_fibre="slack",
        rhbi_cond_ot_weight=1.5,
        rhbi_cond_ot_mode="imq_cmmd",
        total_steps=10,
        batch_size=16,
        num_particles=32,
        z_dim=2,
    )
    dc1 = CondCriticFirst()
    _ = r_cond.make_critic_optimizer(dc1)
    loss_fn = r_cond.make_loss()

    gen = torch.Generator().manual_seed(911)
    xt = torch.randn(16, 2, generator=gen)
    t_step = torch.tensor([1, 2] * 8, dtype=torch.long)
    y_true = 0.5 * xt + 0.2 * torch.randn(16, 2, generator=gen)
    shift = nn.Parameter(torch.tensor([0.8, -0.6]))
    x_pred = (0.5 * xt + 0.2 * torch.randn(16, 2, generator=gen)) + shift

    gl = loss_fn.g_loss(dc1(x_pred, xt=xt, t=t_step), dc1(y_true, xt=xt, t=t_step).detach())
    gl.backward()
    assert shift.grad is not None and torch.isfinite(shift.grad).all()

    dc2 = CondCriticSecond()
    _ = r_cond.make_critic_optimizer(dc2)
    loss_fn2 = r_cond.make_loss()
    x_pred2 = (0.5 * xt).detach().requires_grad_(True)
    gl2 = loss_fn2.g_loss(dc2(xt, x_pred2), dc2(xt, y_true).detach())
    gl2.backward()
    assert x_pred2.grad is not None and torch.isfinite(x_pred2.grad).all()

    for mode_name in ("kernel_ed", "hollow_sinkhorn", "imq_cmmd"):
        x_eq = y_true.detach().clone().requires_grad_(True)
        val_eq = loss_fn.rhbi.conditional_fiber_transport_loss(x_eq, y_true, xt, cond_disc=t_step, mode=mode_name)
        val_eq.backward()
        assert x_eq.grad is not None and torch.allclose(x_eq.grad, torch.zeros_like(x_eq.grad), atol=1e-6)

    for mode_name in ("kernel_ed", "imq_cmmd"):
        grad_acc = torch.zeros(2)
        prim = loss_fn.rhbi
        for trial in range(64):
            g_t = torch.Generator().manual_seed(2000 + trial)
            c_b = torch.randn(24, 2, generator=g_t)
            t_b = torch.tensor([1, 2, 3] * 8, dtype=torch.long)
            y_b = 0.7 * c_b + 0.25 * torch.randn(24, 2, generator=g_t)
            mu = torch.zeros(2, requires_grad=True)
            x_b = 0.7 * c_b + 0.25 * torch.randn(24, 2, generator=g_t) + mu
            val = prim.conditional_fiber_transport_loss(x_b, y_b, c_b, cond_disc=t_b, mode=mode_name)
            val.backward()
            grad_acc += mu.grad.detach()
        mean_grad_at_truth = grad_acc / 64.0
        assert float(mean_grad_at_truth.norm().item()) < 0.03


def test_rhbi_v7_feydy_unique_and_simplicial_manifold_loss():
    """Verify feydy_unique clone deduplication, zero-RNG simplicial manifold loss, MoG bypass, and continuation."""
    with pytest.raises(ValueError, match="self_coupling"):
        RHBILoss(self_coupling="invalid_mode")

    op_fu = RHBILoss(weight=1.0, self_coupling="feydy_unique")
    op_f = RHBILoss(weight=1.0, self_coupling="feydy")

    # 1. Distinct continuous points: feydy_unique matches feydy bit-for-bit
    pts_cont = torch.randn(10, 3, generator=torch.Generator().manual_seed(42))
    y_cont = torch.randn(10, 3, generator=torch.Generator().manual_seed(43))
    d_f1 = torch.zeros(10, requires_grad=True)
    d_f2 = torch.zeros(10, requires_grad=True)
    d_r = torch.zeros(10)
    x1 = pts_cont.clone().requires_grad_(True)
    x2 = pts_cont.clone().requires_grad_(True)
    l_fu, _ = op_fu(x1, y_cont, d_f1, d_r)
    l_f, _ = op_f(x2, y_cont, d_f2, d_r)
    assert torch.equal(l_fu, l_f)
    l_fu.backward()
    l_f.backward()
    assert torch.equal(x1.grad, x2.grad)

    # 2. Exact discrete particle clone in batch: feydy_unique masks off-diagonal clone (0, 1)
    pts_clone = torch.arange(6, dtype=torch.float32).unsqueeze(1) * 20.0
    pts_clone[1] = pts_clone[0].clone()
    _, c_hat_clone = RHBILoss._born_infeld_pair(pts_clone, pts_clone, torch.tensor(1.0))
    zeros6 = torch.zeros(6)
    mask_clone = (c_hat_clone > 1e-12) | torch.eye(6, dtype=torch.bool)
    p_f = op_f._sinkhorn_slack(c_hat_clone, zeros6, zeros6, hollow=False)
    p_fu = op_fu._sinkhorn_slack(c_hat_clone, zeros6, zeros6, hollow=False, pair_mask=mask_clone)
    assert float(p_f[0, 1].item()) > 0.05
    assert float(p_fu[0, 1].item()) == 0.0
    assert float((p_fu[0, 0] * 6.0).item()) > 0.99

    # 3. Simplicial manifold loss: zero RNG consumption, detached z_table, finite G gradients
    G_s = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2))
    z_tab = nn.Parameter(torch.randn(12, 2, generator=torch.Generator().manual_seed(77)))
    y_arc = torch.randn(16, 2, generator=torch.Generator().manual_seed(78))
    op_fu._update_ell(y_arc)
    cpu_rng_before = torch.get_rng_state()
    s_loss = op_fu.simplicial_manifold_loss(z_tab, G_s, y_arc)
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    s_loss.backward()
    assert z_tab.grad is None
    assert any(p.grad is not None and float(p.grad.abs().sum().item()) > 0.0 for p in G_s.parameters())

    # 4. GANTrainer: active on ParticlePrior, exact bypass on MoGParticlePrior, exact continuation
    base_kw = dict(
        rhbi_rapidity_screening=False,
        rhbi_self_coupling="feydy_unique",
        rhbi_akt_horizon=True,
        rhbi_akt_mode="weibull",
        rhbi_slack_gain=0.35,
        rhbi_chord_weight=0.25,
        rhbi_gamma_gauge=0.03,
        rhbi_lambda_floor=0.005,
        ebi_beta=12.0,
        rhbi_settle_beta=16.0,
        rhbi_settle_mode="thermal",
        rhbi_thermal_tau=0.084,
        rhbi_thermal_shock_mu=0.35,
        rhbi_conditional_fibre="slack",
        rhbi_cond_ot_mode="imq_cmmd",
        rhbi_cond_ot_weight=1.5,
        total_steps=6,
        batch_size=8,
        num_particles=12,
        z_dim=2,
    )
    r_s0 = get_recipe("rhbi", rhbi_simplex_weight=0.0, **base_kw)
    r_s1 = get_recipe("rhbi", rhbi_simplex_weight=0.35, **base_kw)
    r_m0 = get_recipe("rhbi", prior_kind="mog", rhbi_simplex_weight=0.0, **base_kw)
    r_m1 = get_recipe("rhbi", prior_kind="mog", rhbi_simplex_weight=0.35, **base_kw)
    batches = [torch.randn(8, 2, generator=torch.Generator().manual_seed(300 + i)) for i in range(4)]

    Gm0, Dm0, _ = _make_mlp_pair(seed=901)
    Gm1, Dm1 = deepcopy(Gm0), deepcopy(Dm0)
    pm0 = r_m0.make_prior()
    with torch.no_grad():
        pm0.z.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(902))
    pm1 = deepcopy(pm0)
    tm0 = GANTrainer(r_m0, Gm0, Dm0, prior=pm0, prior_noise_generator=torch.Generator().manual_seed(903), seed=0)
    tm1 = GANTrainer(r_m1, Gm1, Dm1, prior=pm1, prior_noise_generator=torch.Generator().manual_seed(903), seed=0)
    for b in batches[:2]:
        tm0.step(b)
        tm1.step(b)
    for p0, p1 in zip(Gm0.parameters(), Gm1.parameters()):
        assert torch.equal(p0, p1)
    assert torch.equal(pm0.z, pm1.z)

    Gp0, Dp0, _ = _make_mlp_pair(seed=905)
    Pp0 = ParticlePrior(num_particles=12, z_dim=2)
    with torch.no_grad():
        Pp0.z.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(907))
    Gp1, Dp1, Pp1 = deepcopy(Gp0), deepcopy(Dp0), deepcopy(Pp0)
    Gp2, Dp2, _ = _make_mlp_pair(seed=906)
    Pp2 = ParticlePrior(num_particles=12, z_dim=2)
    tp0 = GANTrainer(r_s0, Gp0, Dp0, prior=Pp0, seed=0)
    tp1 = GANTrainer(r_s1, Gp1, Dp1, prior=Pp1, seed=0)
    for b in batches[:2]:
        tp0.step(b)
        tp1.step(b)
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(Gp0.parameters(), Gp1.parameters()))
    ckpt = deepcopy(tp1.state_dict())
    for b in batches[2:]:
        tp1.step(b)
    tp2 = GANTrainer(r_s1, Gp2, Dp2, prior=Pp2, seed=7)
    tp2.load_state_dict(ckpt)
    for b in batches[2:]:
        tp2.step(b)
    for p1, p2 in zip(Gp1.parameters(), Gp2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(Pp1.z, Pp2.z)

    # 5. feydy_discrete deduplicates on discrete ParticlePrior while matching feydy on quantized MoG ties
    op_fd = RHBILoss(weight=1.0, self_coupling="feydy_discrete")
    op_f_q = RHBILoss(weight=1.0, self_coupling="feydy")
    op_fu_q = RHBILoss(weight=1.0, self_coupling="feydy_unique")
    x_q_tie = torch.tensor([[0.0], [0.0], [0.3], [20.0]], requires_grad=True)
    y_q_ref = torch.tensor([[0.5], [1.0], [9.5], [19.5]])
    z4 = torch.zeros(4)
    # Without bind_discrete_prior_step (or False, i.e. MoG): matches feydy bit-for-bit
    op_fd.bind_discrete_prior_step(False)
    l_fd_mog, _ = op_fd(x_q_tie, y_q_ref, z4.clone().requires_grad_(True), z4)
    l_f_mog, _ = op_f_q(x_q_tie.detach().clone().requires_grad_(True), y_q_ref, z4.clone().requires_grad_(True), z4)
    _, _ = op_fu_q(x_q_tie.detach().clone().requires_grad_(True), y_q_ref, z4.clone().requires_grad_(True), z4)
    assert torch.equal(l_fd_mog, l_f_mog)
    # With bind_discrete_prior_step(True) (i.e. ParticlePrior): matches feydy_unique bit-for-bit and resets flag
    op_fd.bind_discrete_prior_step(True)
    l_fd_disc, _ = op_fd(x_q_tie.detach().clone().requires_grad_(True), y_q_ref, z4.clone().requires_grad_(True), z4)
    l_fu_disc, _ = op_fu_q(x_q_tie.detach().clone().requires_grad_(True), y_q_ref, z4.clone().requires_grad_(True), z4)
    assert torch.equal(l_fd_disc, l_fu_disc)
    assert not torch.equal(l_fd_disc, op_f_q(x_q_tie.detach().clone().requires_grad_(True), y_q_ref, z4.clone().requires_grad_(True), z4)[0])
    assert op_fd._discrete_prior_step is False


def test_rhbi_v9_cvx_mu_gain_and_connected_chart_simplex():
    """Verify shock-safe convex-unimodal mean gain and H_0 connected-manifold simplicial chart alignment."""
    with pytest.raises(ValueError, match="rhbi_cvx_mu_gain"):
        get_recipe("rhbi", rhbi_cvx_mu_gain=0.0)
    with pytest.raises(ValueError, match="rhbi_simplex_chart_weight"):
        get_recipe("rhbi", rhbi_simplex_chart_weight=-0.1)
    with pytest.raises(ValueError, match="cvx_mu_gain"):
        RHBILoss(cvx_mu_gain=0.0)

    # 1. Convex unimodal gain boosts moderate positive r_mu_sq (< 0.045) while capping stationary chi^2 spikes
    g_rng = torch.Generator().manual_seed(910)
    y_uni = 2.0 + 0.5 * torch.randn(64, 1, generator=g_rng)
    x_mod = 2.08 + 0.5 * torch.randn(64, 1, generator=g_rng)
    ell_1d = torch.tensor(0.35)
    _, r_mu_default, _, _, _ = RHBILoss._split_two_sample_residuals(x_mod, y_uni, ell=ell_1d, cvx_mu_gain=0.40)
    _, r_mu_boosted, _, _, _ = RHBILoss._split_two_sample_residuals(x_mod, y_uni, ell=ell_1d, cvx_mu_gain=0.58)
    assert r_mu_boosted > r_mu_default
    assert r_mu_boosted <= 0.045 + 1e-8

    # Moderate shift stays shock-safe (<= max(r_mu_base, 0.045)), while true 2-sigma shift triggers shock (> 0.105)
    x_spike = 2.24 + 0.5 * torch.randn(64, 1, generator=g_rng)
    _, r_mu_sp_base, _, _, _ = RHBILoss._split_two_sample_residuals(x_spike, y_uni, ell=ell_1d, cvx_mu_gain=0.40)
    _, r_mu_sp_cvx, _, _, _ = RHBILoss._split_two_sample_residuals(x_spike, y_uni, ell=ell_1d, cvx_mu_gain=0.58)
    assert r_mu_sp_base < 0.045
    assert r_mu_sp_cvx == pytest.approx(0.045, abs=1e-6)
    x_shift = 3.0 + 0.5 * torch.randn(64, 1, generator=g_rng)
    _, r_mu_shift, _, _, _ = RHBILoss._split_two_sample_residuals(x_shift, y_uni, ell=ell_1d, cvx_mu_gain=0.58)
    assert r_mu_shift > 0.105

    # Multimodal mixture leaves r_mu_sq bit-identical (w_cvx == 0)
    c_multi = (torch.randint(0, 2, (64, 1), generator=g_rng).float() * 2.0 - 1.0) * 4.0
    y_multi = c_multi + 0.05 * torch.randn(64, 2, generator=g_rng)
    x_multi = y_multi + 0.10
    res_m0 = RHBILoss._split_two_sample_residuals(x_multi, y_multi, ell=torch.tensor(0.2), cvx_mu_gain=0.40)
    res_m1 = RHBILoss._split_two_sample_residuals(x_multi, y_multi, ell=torch.tensor(0.2), cvx_mu_gain=0.58)
    assert res_m0 == res_m1

    # 2. H_0 connected-manifold simplicial chart loss: bit-identical on disconnected modes & high-D white noise,
    # active with z_table + G gradients and zero RNG consumption on connected low-D sheets
    op = RHBILoss(weight=1.0, eps=0.1)
    G_s, _, _ = _make_mlp_pair(seed=911)
    z_tab = torch.randn(16, 2, generator=torch.Generator().manual_seed(912), requires_grad=True)

    # Disconnected 2-cluster target -> kappa_H0 >> 10 -> g_conn == 0 -> bit-identical to chart_weight=0.0
    c_disc = (torch.arange(16) % 2).float().unsqueeze(1) * 10.0 - 5.0
    y_disc = c_disc.expand(16, 2) + 0.01 * torch.randn(16, 2, generator=torch.Generator().manual_seed(913))
    op._update_ell(y_disc, allow_update=True)
    l_disc_0 = op.simplicial_manifold_loss(z_tab, G_s, y_disc, chart_weight=0.0)
    l_disc_4 = op.simplicial_manifold_loss(z_tab, G_s, y_disc, chart_weight=4.0)
    assert torch.equal(l_disc_0, l_disc_4)

    # Connected 1D/2D sheet in R^2 -> g_conn == 1.0 -> non-zero z_table & G gradients, zero global RNG draws
    t_arc = torch.linspace(-1.2, 1.2, 16)
    y_sheet = torch.stack([t_arc, 0.4 * t_arc.square()], dim=-1)
    op._update_ell(y_sheet, allow_update=True)
    rng_before = torch.get_rng_state()
    G_s.zero_grad()
    z_live = z_tab.detach().clone().requires_grad_(True)
    l_conn_0 = op.simplicial_manifold_loss(z_live, G_s, y_sheet, chart_weight=0.0)
    l_conn_4 = op.simplicial_manifold_loss(z_live, G_s, y_sheet, chart_weight=4.0)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert l_conn_4.item() > l_conn_0.item()
    l_conn_4.backward()
    assert z_live.grad is not None and torch.isfinite(z_live.grad).all() and z_live.grad.abs().sum().item() > 0.0
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in G_s.parameters())

    # 3. Exact GANTrainer checkpoint continuation parity with both rhbi-v9 capabilities active
    r_v9 = get_recipe(
        "rhbi",
        constraint_geometry_mode="none",
        rhbi_thermal_tau=0.084,
        rhbi_cvx_mu_gain=0.58,
        rhbi_simplex_weight=0.35,
        rhbi_simplex_chart_weight=4.0,
        total_steps=6,
        batch_size=8,
        num_particles=12,
        z_dim=2,
    )
    batches_sheet = [
        torch.stack(
            [
                torch.linspace(-1.0 + 0.05 * k, 1.0 + 0.05 * k, 8),
                0.3 * torch.linspace(-1.0 + 0.05 * k, 1.0 + 0.05 * k, 8).square(),
            ],
            dim=-1,
        )
        for k in range(4)
    ]
    Ga, Da, _ = _make_mlp_pair(seed=914)
    Pa = ParticlePrior(num_particles=12, z_dim=2)
    with torch.no_grad():
        Pa.z.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(915))
    Gb, Db, _ = _make_mlp_pair(seed=914)
    Pb = ParticlePrior(num_particles=12, z_dim=2)
    ta = GANTrainer(r_v9, Ga, Da, prior=Pa, seed=0)
    for b in batches_sheet[:2]:
        ta.step(b)
    ckpt_v9 = deepcopy(ta.state_dict())
    for b in batches_sheet[2:]:
        ta.step(b)
    tb = GANTrainer(r_v9, Gb, Db, prior=Pb, seed=9)
    tb.load_state_dict(ckpt_v9)
    for b in batches_sheet[2:]:
        tb.step(b)
    for pa, pb in zip(Ga.parameters(), Gb.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(Da.parameters(), Db.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(Pa.z, Pb.z)


def test_rhbi_v10_tail_spill_and_macro_simplicial_settle():
    """Verify 6th-channel inter-mode valley tail-spill EMA, schema-3 checkpointing, frozen-prior control, and macro-simplicial settling."""
    from particlegan.particle_prior import MoGParticlePrior
    from particlegan.rhbi import _RHBIObserverStepRecord

    with pytest.raises(ValueError, match="rhbi_tail_spill_gain"):
        get_recipe("rhbi", rhbi_tail_spill_gain=-0.1)
    with pytest.raises(ValueError, match="rhbi_tail_spill_gain"):
        get_recipe("rhbi", rhbi_tail_spill_gain=True)
    with pytest.raises(ValueError, match="rhbi_simplex_macro_settle"):
        get_recipe("rhbi", rhbi_simplex_macro_settle=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tail_spill_gain"):
        RHBILoss(tail_spill_gain=-0.05)
    with pytest.raises(ValueError, match="simplex_macro_settle"):
        RHBILoss(simplex_macro_settle="yes")  # type: ignore[arg-type]

    op_val = RHBILoss(weight=1.0)
    with pytest.raises(ValueError, match="simplicial_active"):
        op_val.bind_discrete_prior_step(True, simplicial_active=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="macro_settle"):
        op_val.simplicial_manifold_loss(
            torch.randn(4, 2),
            lambda z: z,
            torch.randn(4, 2),
            macro_settle="yes",  # type: ignore[arg-type]
        )

    # 1. Schema-3 observer checkpointing + RHBIStepRecord ema_tail validation + rejection of schema-2 checkpoints
    obs_rec = _RHBIObserverStepRecord()
    obs_rec.observe_rhbi_thermal(0.01, 0.001, 0.0, 0.01, 0.01, 0.084, r_tail=0.12)
    obs_sd = deepcopy(obs_rec.state_dict())
    assert obs_sd["rhbi"]["schema"] == 3
    assert obs_sd["rhbi"]["ema_tail"] == pytest.approx(0.12)
    obs_restored = _RHBIObserverStepRecord()
    obs_restored.load_state_dict(obs_sd)
    assert obs_restored.rhbi_ema_tail == pytest.approx(0.12)
    assert deepcopy(obs_restored).rhbi_ema_tail == pytest.approx(0.12)

    bad_schema2 = deepcopy(obs_sd)
    bad_schema2["rhbi"]["schema"] = 2
    with pytest.raises(ValueError, match="invalid RHBI observer metadata"):
        _RHBIObserverStepRecord().load_state_dict(bad_schema2)
    bad_missing_tail = deepcopy(obs_sd)
    del bad_missing_tail["rhbi"]["ema_tail"]
    with pytest.raises(ValueError, match="invalid RHBI observer metadata"):
        _RHBIObserverStepRecord().load_state_dict(bad_missing_tail)

    step_rec = RHBIStepRecord()
    step_rec.observe_rhbi_thermal(0.01, 0.001, 0.0, -0.02, 0.01, 0.084, r_tail=-0.03)
    step_sd = deepcopy(step_rec.state_dict())
    assert step_sd["ema_tail"] == pytest.approx(-0.03)
    step_restored = RHBIStepRecord()
    step_restored.load_state_dict(step_sd)
    assert step_restored.rhbi_ema_tail == pytest.approx(-0.03)
    assert deepcopy(step_restored).rhbi_ema_tail == pytest.approx(-0.03)
    bad_step = deepcopy(step_sd)
    del bad_step["ema_tail"]
    with pytest.raises(ValueError, match="invalid RHBI controller state keys"):
        RHBIStepRecord().load_state_dict(bad_step)

    # 2. Zero-strength bypass resets both step flags cleanly
    op_zero = RHBILoss(weight=0.0, tail_spill_gain=1.8, simplex_macro_settle=True)
    op_zero.bind_discrete_prior_step(True, simplicial_active=True)
    _ = op_zero(torch.randn(8, 2, requires_grad=True), torch.randn(8, 2), torch.zeros(8, requires_grad=True), torch.zeros(8))
    assert op_zero._discrete_prior_step is False
    assert op_zero._simplicial_active_step is False

    # 3. Valley gate & 6th-channel tail-spill EMA: exact 0.0 on convex unimodal Gaussian, >0 on separated modes
    g_rng = torch.Generator().manual_seed(920)
    y_gauss = torch.randn(64, 2, generator=g_rng)
    assert RHBILoss._compute_valley_gate(y_gauss, torch.tensor(0.35), 3.0) == 0.0

    centers = (torch.arange(64) % 2).float().unsqueeze(1) * 12.0 - 6.0
    y_sep = torch.cat([centers, torch.zeros_like(centers)], dim=-1) + 0.15 * torch.randn(64, 2, generator=g_rng)
    w_val = RHBILoss._compute_valley_gate(y_sep, torch.tensor(0.20), 3.0)
    assert w_val > 0.9

    # 93% inside modes (prec_q75 <= 0) + 7% inter-mode valley bridges (prec_q96 >> 0):
    # tail_spill_gain=0.0 cools prematurely while tail_spill_gain=1.8 holds s_therm == 1.0, with zero RNG consumption
    x_bridged = y_sep.clone()
    x_bridged[:4] = torch.tensor([0.0, 0.0])  # 4/64 = 6.25% bridge points in deep inter-mode valley
    op_v9_th = RHBILoss(weight=1.0, rho=3.0, rapidity_screening=False, thermal_tau=0.084, tail_spill_gain=0.0)
    op_v10_th = RHBILoss(weight=1.0, rho=3.0, rapidity_screening=False, thermal_tau=0.084, tail_spill_gain=1.8)
    rec_v9 = RHBIStepRecord()
    rec_v10 = RHBIStepRecord()
    rng_before = torch.get_rng_state()
    for _ in range(15):
        xb9 = x_bridged.detach().clone().requires_grad_(True)
        xb10 = x_bridged.detach().clone().requires_grad_(True)
        op_v9_th(xb9, y_sep, torch.zeros(64, requires_grad=True), torch.zeros(64), record=rec_v9)[0].backward()
        op_v10_th(xb10, y_sep, torch.zeros(64, requires_grad=True), torch.zeros(64), record=rec_v10)[0].backward()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert rec_v9.rhbi_thermal_scale < 0.05
    assert rec_v10.rhbi_thermal_scale == pytest.approx(1.0)
    assert rec_v10.rhbi_ema_tail > 0.084

    # 4. Semantic control: frozen discrete ParticlePrior (z.requires_grad == False) never defers thermal observation
    u_grid = torch.linspace(-1.0, 1.0, 8)
    uu, vv = torch.meshgrid(u_grid, u_grid, indexing="ij")
    y_saddle = torch.stack([uu.reshape(-1), vv.reshape(-1), (uu * vv).reshape(-1)], dim=-1)
    _, g_macro_saddle, _, _ = RHBILoss._compute_manifold_gates(y_saddle)
    assert g_macro_saddle > 0.5
    _, g_macro_sep, _, _ = RHBILoss._compute_manifold_gates(y_sep)
    assert g_macro_sep == 0.0

    r_frozen = get_recipe(
        "rhbi",
        prior_update="frozen",
        constraint_geometry_mode="none",
        rhbi_thermal_tau=0.084,
        rhbi_simplex_weight=0.35,
        rhbi_simplex_chart_weight=4.0,
        rhbi_simplex_macro_settle=True,
        total_steps=4,
        batch_size=64,
        num_particles=16,
        z_dim=2,
    )
    Gf, Df, _ = _make_mlp_pair(x_dim=3, seed=921)
    Pf = ParticlePrior(num_particles=16, z_dim=2)
    tf = GANTrainer(r_frozen, Gf, Df, prior=Pf, seed=0)
    tf.step(y_saddle)
    assert tf.opt_d.record.rhbi_thermal_initialized is True

    # 5. Exact GANTrainer checkpoint continuation parity across active rhbi-v10 mechanisms (MoG + macro saddle sheet)
    r_mog_v10 = get_recipe(
        "rhbi",
        prior_kind="mog",
        sigma_rel=0.05,
        standardize=True,
        constraint_geometry_mode="none",
        rhbi_rho=3.0,
        rhbi_rapidity_screening=False,
        rhbi_thermal_tau=0.084,
        rhbi_tail_spill_gain=1.8,
        total_steps=6,
        batch_size=16,
        num_particles=16,
        z_dim=2,
    )
    mog_batches = [
        torch.cat(
            [
                ((torch.arange(16) % 2).float().unsqueeze(1) * 10.0 - 5.0),
                torch.zeros(16, 1),
            ],
            dim=-1,
        )
        + 0.1 * torch.randn(16, 2, generator=torch.Generator().manual_seed(930 + k))
        for k in range(4)
    ]
    Gm1, Dm1, _ = _make_mlp_pair(seed=935)
    Pm1 = MoGParticlePrior(num_particles=16, z_dim=2, sigma=0.05, standardize=True)
    with torch.no_grad():
        Pm1.z.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(936))
    Gm2, Dm2, _ = _make_mlp_pair(seed=935)
    Pm2 = MoGParticlePrior(num_particles=16, z_dim=2, sigma=0.05, standardize=True)
    tm1 = GANTrainer(r_mog_v10, Gm1, Dm1, prior=Pm1, seed=0)
    for b in mog_batches[:2]:
        tm1.step(b)
    assert tm1.opt_d.record.rhbi_ema_tail != 0.0
    ckpt_mog = deepcopy(tm1.state_dict())
    for b in mog_batches[2:]:
        tm1.step(b)
    tm2 = GANTrainer(r_mog_v10, Gm2, Dm2, prior=Pm2, seed=11)
    tm2.load_state_dict(ckpt_mog)
    for b in mog_batches[2:]:
        tm2.step(b)
    for p1, p2 in zip(Gm1.parameters(), Gm2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(Pm1.z, Pm2.z)

    r_sad_v10 = get_recipe(
        "rhbi",
        constraint_geometry_mode="none",
        rhbi_thermal_tau=0.084,
        rhbi_simplex_weight=0.35,
        rhbi_simplex_chart_weight=4.0,
        rhbi_simplex_macro_settle=True,
        total_steps=6,
        batch_size=64,
        num_particles=16,
        z_dim=2,
    )
    Gs1, Ds1, _ = _make_mlp_pair(x_dim=3, seed=940)
    Ps1 = ParticlePrior(num_particles=16, z_dim=2)
    with torch.no_grad():
        Ps1.z.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(941))
    Gs2, Ds2, _ = _make_mlp_pair(x_dim=3, seed=940)
    Ps2 = ParticlePrior(num_particles=16, z_dim=2)
    ts1 = GANTrainer(r_sad_v10, Gs1, Ds1, prior=Ps1, seed=0)
    for k in range(2):
        ts1.step(y_saddle + 0.01 * float(k))
    ckpt_sad = deepcopy(ts1.state_dict())
    for k in range(2, 4):
        ts1.step(y_saddle + 0.01 * float(k))
    ts2 = GANTrainer(r_sad_v10, Gs2, Ds2, prior=Ps2, seed=13)
    ts2.load_state_dict(ckpt_sad)
    for k in range(2, 4):
        ts2.step(y_saddle + 0.01 * float(k))
    for p1, p2 in zip(Gs1.parameters(), Gs2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(Ps1.z, Ps2.z)
