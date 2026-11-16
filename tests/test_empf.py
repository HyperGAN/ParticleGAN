"""Unit, continuation, and semantic-control tests for Exponential Moving Prototype Flow (EMPF)."""
from copy import deepcopy
import io
import math
import pytest
import torch
from torch import nn

from particlegan import (
    DEFAULT_RECIPE,
    EMPFCriticAdam,
    EMPFCriticPenalty,
    EMPFPriorRegularizer,
    EMPFSinkhornRelativisticLoss,
    EMPFStepRecord,
    GANTrainer,
    get_recipe,
    init,
)
from particlegan.empf import (
    emeans_effective_alpha,
    mass_weighted_sinkhorn,
    update_emeans_prototypes,
)


def test_default_recipe_and_incumbent_presets_unchanged():
    assert DEFAULT_RECIPE == "empf"
    assert get_recipe() == get_recipe("empf")
    empf = get_recipe("empf")
    assert empf.effective_critic_formulation == "empf"
    assert empf.prior_regularizer == "empf"
    assert empf.optimizer_family == "dualnorm"
    assert empf.reg_anchor_weight == 0.0
    assert empf.lr_floor == 1.0 and empf.network_lr_floor == 1.0

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe("empf", optimizer_family="formulation", optimizer_smoothing=0.0,
                   optimizer_convolution="none", optimizer_convolution_rule="polar",
                   optimizer_dense_profile="default", optimizer_svd_backend="native",
                   training_precision="float32", constraint_geometry_mode="none",
                   reg_anchor_weight=0.5)
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe("empf", reg_arm="b_cap", optimizer_family="adam",
                   optimizer_smoothing=0.0, optimizer_convolution="none",
                   optimizer_convolution_rule="polar", optimizer_dense_profile="default",
                   optimizer_svd_backend="native", training_precision="float32",
                   constraint_geometry_mode="none")


def test_emeans_hit_rate_scaling_and_mass_conserving_revival():
    alpha_small = emeans_effective_alpha(64, 64)
    alpha_large = emeans_effective_alpha(2048, 256)
    assert math.isclose(alpha_small, 0.08, rel_tol=1e-6)
    assert alpha_large > alpha_small
    assert 0.03 <= alpha_small <= 0.35 and 0.03 <= alpha_large <= 0.35

    rng = torch.Generator().manual_seed(0)
    x_init = -5.0 + 0.05 * torch.randn(32, 2, generator=rng)
    c0, w0, p_sum, p_count, phase, revivals = update_emeans_prototypes(
        x_init, None, None, 0.0, 0, max_prototypes=16
    )
    assert c0.shape == (16, 2)
    assert math.isclose(float(w0.sum()), 1.0, rel_tol=1e-6)
    initial_total_mass = float(p_count.sum())
    assert revivals == 0

    # Shift target abruptly to (+5, +5) so 15 of 16 prototypes starve and trigger proportional splits.
    for _ in range(45):
        x_shifted = 5.0 + 0.05 * torch.randn(32, 2, generator=rng)
        c_now, w_now, p_sum, p_count, phase, revivals = update_emeans_prototypes(
            x_shifted, p_sum, p_count, phase, revivals, max_prototypes=16
        )
    assert revivals > 0, "Expected dead prototypes to be revived via proportional donor splitting"
    assert math.isclose(float(p_count.sum()), initial_total_mass, rel_tol=1e-5)
    assert math.isclose(float(w_now.sum()), 1.0, rel_tol=1e-6)
    assert float((c_now - 5.0).norm(dim=1).max()) < 0.5


def test_mass_weighted_sinkhorn_preserves_rare_mode_marginals():
    prototypes = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([0.98, 0.02], dtype=torch.float32)
    rng = torch.Generator().manual_seed(7)
    x_fake = torch.randn(128, 2, generator=rng)

    pi, norm_cost, residual, data_scale = mass_weighted_sinkhorn(
        prototypes, weights, x_fake, sinkhorn_eps=0.15, sinkhorn_iters=25
    )
    assert pi.shape == (2, 128)
    assert torch.isfinite(norm_cost).all() and torch.isfinite(residual) and float(data_scale) > 0.0
    row_sums = pi.sum(dim=1)
    col_sums = pi.sum(dim=0)
    assert torch.allclose(row_sums, weights, atol=1e-3)
    assert torch.allclose(col_sums, torch.full((128,), 1.0 / 128.0), atol=1e-3)


def test_empf_prior_regularizer_dof_bypass_and_dead_particle_tether():
    reg = EMPFPriorRegularizer(None, weight=1.0, k_nn=5)

    # 1. Tiny unanchored discrete codebook returns exact zero; anchored codebook
    # returns zero when well-separated and repels only colliding prototype pairs.
    tiny = torch.randn(5, 2, requires_grad=True)
    tiny_loss = reg(tiny)
    assert float(tiny_loss.detach()) == 0.0

    recipe5 = get_recipe("empf", num_particles=5, z_dim=2)
    prior5 = recipe5.make_prior()
    init.deterministic_orthogonal_(prior5, seed=0)
    recipe5.apply_prior_policy(prior5)
    reg_anchored = recipe5.make_prior_regularizer(prior=prior5, reference=prior5)
    separated = torch.tensor(
        [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0], [0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    sep_loss = reg_anchored(separated)
    assert float(sep_loss.detach()) == 0.0

    colliding = separated.detach().clone()
    colliding[1] = colliding[0] + torch.tensor([1e-3, 0.0])
    colliding.requires_grad_(True)
    col_loss = reg_anchored(colliding)
    assert float(col_loss.detach()) > 0.0
    col_loss.backward()
    assert colliding.grad is not None and torch.isfinite(colliding.grad).all()
    assert float(colliding.grad[0, 0]) > 0.0 and float(colliding.grad[1, 0]) < 0.0
    assert torch.all(colliding.grad[2:] == 0.0)

    # 2. Spread bimodal cloud vs stranded rogue outlier particle.
    rng = torch.Generator().manual_seed(11)
    left = torch.tensor([-1.5, 0.0]) + 0.05 * torch.randn(16, 2, generator=rng)
    right = torch.tensor([1.5, 0.0]) + 0.05 * torch.randn(16, 2, generator=rng)
    clean_cloud = torch.cat([left, right], dim=0).requires_grad_(True)
    loss_clean = reg(clean_cloud)

    stranded_cloud_data = clean_cloud.detach().clone()
    stranded_cloud_data[0] = torch.tensor([12.0, 0.0])  # Stranded far from both modes
    stranded_cloud = stranded_cloud_data.requires_grad_(True)
    loss_stranded = reg(stranded_cloud)
    assert float(loss_stranded.detach()) > float(loss_clean.detach())
    loss_stranded.backward()
    assert stranded_cloud.grad is not None and torch.isfinite(stranded_cloud.grad).all()
    # Gradient on stranded x=12.0 particle points positively in loss -> negative descent pulls it left toward modes!
    assert float(stranded_cloud.grad[0, 0]) > 0.0


def test_conditional_critic_penalty_preserves_fiber_and_skips_marginal_bank():
    recipe = get_recipe("empf")

    class CondCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))

        def forward(self, x, *, cond):
            return self.net(torch.cat([x, cond], dim=1)).squeeze(-1)

    critic = CondCritic()
    init.deterministic_orthogonal_(critic, seed=0)
    opt_d = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)
    assert isinstance(penalty, EMPFCriticPenalty)

    rng = torch.Generator().manual_seed(3)
    cond = torch.randn(32, 2, generator=rng)
    real = cond + 0.1 * torch.randn(32, 2, generator=rng)
    fake = cond - 0.1 * torch.randn(32, 2, generator=rng)

    loss = penalty(critic, real, fake, cond=cond)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in critic.parameters())
    # Conditional call must not pollute an unconditional marginal prototype bank.
    assert opt_d.record.proto_initialized is False
    assert opt_d.record.proto_sum is None


@pytest.mark.parametrize("optimizer_family", ["dualnorm", "formulation", "adam"])
def test_empf_checkpoint_continuation_and_corruption_rejection(optimizer_family):
    overrides = dict(
        critic_formulation="empf",
        prior_regularizer="empf",
        prior_reg=0.02,
        optimizer_family=optimizer_family,
        lr=0.004,
        d_lr_mult=1.0,
        prior_lr_mult=0.05 if optimizer_family == "dualnorm" else 1.0,
        lr_floor=1.0,
        network_lr_floor=1.0,
        num_particles=32,
        z_dim=2,
        batch_size=16,
        total_steps=30,
    )
    if optimizer_family != "dualnorm":
        overrides.update(
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_dense_profile="default",
            optimizer_svd_backend="native",
            constraint_geometry_mode="none",
        )
    else:
        overrides.update(
            optimizer_smoothing=0.001,
            optimizer_convolution="per_offset",
            optimizer_convolution_rule="quintic",
            constraint_geometry_mode="direction_blend",
        )
    recipe = get_recipe("empf", **overrides)

    def make_models():
        g = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
        d = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        p = recipe.make_prior()
        for idx, mod in enumerate((g, d, p)):
            init.deterministic_orthogonal_(mod, seed=idx)
        return g, d, p

    g1, d1, p1 = make_models()
    g2, d2, p2 = deepcopy(g1), deepcopy(d1), deepcopy(p1)

    t1 = GANTrainer(recipe, g1, d1, prior=p1, seed=0)
    if optimizer_family == "formulation":
        assert isinstance(t1.opt_d, EMPFCriticAdam)
        assert isinstance(t1.opt_d.record, EMPFStepRecord)

    rng1 = torch.Generator().manual_seed(19)
    for step in range(12):
        shift = -2.5 if step < 5 else 2.5
        t1.step(shift + 0.1 * torch.randn(16, 2, generator=rng1))

    # Sampling must not mutate trainer/record state.
    state_before_sample = deepcopy(t1.state_dict())
    _ = t1.sample(24)
    state_after_sample = t1.state_dict()
    assert torch.equal(
        state_before_sample["optimizers"][1]["regularizer"]["record"]["empf"]["proto_sum"],
        state_after_sample["optimizers"][1]["regularizer"]["record"]["empf"]["proto_sum"],
    )
    assert (
        state_before_sample["optimizers"][1]["regularizer"]["record"]["calls"]
        == state_after_sample["optimizers"][1]["regularizer"]["record"]["calls"]
    )

    buf = io.BytesIO()
    torch.save({"trainer": state_after_sample, "rng": rng1.get_state()}, buf)
    buf.seek(0)
    saved = torch.load(buf, weights_only=True)

    for _ in range(8):
        t1.step(2.5 + 0.1 * torch.randn(16, 2, generator=rng1))

    t2 = GANTrainer(recipe, g2, d2, prior=p2, seed=77)
    # Corrupted EMPF packet must be rejected before mutating live state.
    bad_state = deepcopy(saved["trainer"])
    bad_state["optimizers"][1]["regularizer"]["record"]["empf"]["thermal_scale"] = 2.5
    with pytest.raises(ValueError, match="thermal"):
        t2.opt_d.load_state_dict(bad_state["optimizers"][1])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|thermal"):
        t2.load_state_dict(bad_state)

    t2.load_state_dict(saved["trainer"])
    rng2 = torch.Generator()
    rng2.set_state(saved["rng"])
    for _ in range(8):
        t2.step(2.5 + 0.1 * torch.randn(16, 2, generator=rng2))

    for pa, pb in zip(t1.G.parameters(), t2.G.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(t1.D.parameters(), t2.D.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(t1.sample(32), t2.sample(32))


def test_empf_relativistic_loss_option():
    recipe = get_recipe("empf", loss="relativistic")
    loss_fn = recipe.make_loss()
    assert isinstance(loss_fn, EMPFSinkhornRelativisticLoss)
    r = torch.randn(16, 1, requires_grad=True)
    f = torch.randn(16, 1, requires_grad=True)
    dl = loss_fn.d_loss(r, f)
    gl = loss_fn.g_loss(f, r.detach())
    assert torch.isfinite(dl) and torch.isfinite(gl)


def test_empf_deliberate_damage_and_instance_isolation():
    """Verify zero cross-instance thermal/prototype leakage and genuine generator serving."""
    recipe = get_recipe(
        "empf",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=80,
        lr=0.008,
        prior_lr_mult=0.02,
    )

    def build_pair(seed_offset: int):
        g = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
        d = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
        p = recipe.make_prior()
        for idx, mod in enumerate((g, d, p)):
            init.deterministic_orthogonal_(mod, seed=seed_offset + idx)
        return GANTrainer(recipe, g, d, prior=p, seed=seed_offset)

    trainer_main = build_pair(0)
    trainer_other = build_pair(100)

    target_center = torch.tensor([2.0, -1.5])
    init_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())

    rng = torch.Generator().manual_seed(42)
    for _ in range(60):
        trainer_main.step(target_center + 0.08 * torch.randn(32, 2, generator=rng))
        # Interleave steps on a wild second trainer to detect any global state leakage.
        trainer_other.step(torch.tensor([-50.0, 50.0]) + 5.0 * torch.randn(32, 2, generator=rng))

    rec_main = trainer_main.opt_d.record
    rec_other = trainer_other.opt_d.record
    assert rec_main is not rec_other
    assert rec_main.proto_sum.shape == (32, 2)
    assert rec_main.proto_sum.shape != rec_other.proto_sum.shape or not torch.allclose(
        rec_main.proto_sum, rec_other.proto_sum
    )
    assert rec_main.ema_data_scale != rec_other.ema_data_scale

    trained_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())
    assert trained_err < 0.5 * init_err, f"Expected EMPF training to reduce mean error ({trained_err} vs {init_err})"

    # Deliberate damage to served generator weights must degrade sample() immediately
    # (proving sample() evaluates G(prior) and never returns critic prototypes).
    with torch.no_grad():
        for param in trainer_main.G.parameters():
            param.fill_(25.0)
    damaged_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())
    assert damaged_err > 10.0 * trained_err


def test_empf_fresh_process_import_order_and_incumbent_hashes():
    """Verify import order leaves shared defaults and incumbent recipe digests untouched."""
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[1])
    probe_a = (
        "import json, particlegan.empf, particlegan.recipes as r; "
        "from experiments.forge.contracts import stable_hash; "
        "from experiments.forge.techniques import technique_signature; "
        "print(json.dumps({'default': r.DEFAULT_RECIPE, "
        "'hashes': {k: stable_hash(r.get_recipe(k).to_dict()) for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}, "
        "'sigs': {k: technique_signature(r.get_recipe(k))['digest'] for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}}))"
    )
    probe_b = (
        "import json, particlegan.recipes as r; "
        "from experiments.forge.contracts import stable_hash; "
        "from experiments.forge.techniques import technique_signature; "
        "import particlegan.empf; "
        "print(json.dumps({'default': r.DEFAULT_RECIPE, "
        "'hashes': {k: stable_hash(r.get_recipe(k).to_dict()) for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}, "
        "'sigs': {k: technique_signature(r.get_recipe(k))['digest'] for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}}))"
    )
    out_a = json.loads(subprocess.check_output([sys.executable, "-c", probe_a], cwd=root, text=True))
    out_b = json.loads(subprocess.check_output([sys.executable, "-c", probe_b], cwd=root, text=True))
    assert out_a == out_b
    assert out_a["default"] == "empf"
    assert out_a["hashes"]["bispf_optimized"].startswith("a59d78806e67")
    assert out_a["hashes"]["bispf"].startswith("6bb9c45dfbdb")
    assert out_a["hashes"]["bcap"].startswith("1c0aeefa46b0")
    assert out_a["sigs"]["empf"] != out_a["sigs"]["bispf_optimized"]


def test_empf_v2_settling_ebi_and_discrete_simplex_cooling():
    """Verify clock-free settling Born-Infeld sharpness and discrete-simplex post-acquisition cooling."""
    recipe = get_recipe("empf")
    assert recipe.ebi_beta == 6.0

    # 1. State-driven settling Born-Infeld sharpness sharpens once ema_residual < 0.16
    # and scales with prototype-bank batch resolution (b_r <= 64 vs b_r >= 128).
    critic = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
    opt_d = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)
    kernel = penalty.regularizer

    x_r64 = torch.zeros(64, 2)
    x_f64 = torch.ones(64, 2)
    kernel.record.initialized = True
    kernel.record.ema_residual = 0.50
    kernel.record.ema_data_scale = 1.0
    p_unsettled_64, _ = kernel.penalty(critic, x_r64, x_f64, step=1, is_conditioned=True)

    # Re-evaluate at identical x_real == x_fake so conditional residual is 0.0 (fully settled):
    p_settled_64, _ = kernel.penalty(critic, x_r64, x_r64, step=2, is_conditioned=True)
    # Tighter softplus beta lowers softplus(1 - 1, beta)^2 = (ln(2)/beta)^2 strictly!
    assert float(p_settled_64.detach()) < float(p_unsettled_64.detach())

    x_r128 = torch.zeros(128, 2)
    kernel.record.ema_residual = 0.0
    p_settled_128, _ = kernel.penalty(critic, x_r128, x_r128, step=3, is_conditioned=True)
    assert float(p_settled_128.detach()) < float(p_settled_64.detach())

    # 2. Discrete-simplex Sinkhorn thermal cooling stays at 1.0 during acquisition
    # (large residual) and cools down to 0.0004 (< optimizer_smoothing) when residual settles.
    disc_critic = nn.Linear(180, 1)
    init.deterministic_orthogonal_(disc_critic, seed=0)
    disc_opt = recipe.make_critic_optimizer(disc_critic)
    disc_pen = recipe.make_critic_penalty(disc_opt, collect_stats=True)

    eye = torch.eye(180)
    real_far = eye[torch.arange(64) % 5]
    fake_far = eye[(torch.arange(64) % 5) + 20]
    _ = disc_pen(disc_critic, real_far, fake_far)
    assert math.isclose(disc_pen.last_stats["s_thermal"], 1.0, rel_tol=1e-6)

    for _ in range(80):
        _ = disc_pen(disc_critic, real_far, real_far)
    assert math.isclose(disc_pen.last_stats["s_thermal"], 0.0004, rel_tol=1e-6)


def test_empf_v4_fixed_small_batch_bank_and_shift_revival():
    """Verify B=64 2-mode Gaussian keeps fixed K=64 bank with zero shock while 1D/2D shifts reacquire rapidly."""
    rng = torch.Generator().manual_seed(12)
    centers2 = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])

    p_sum, p_count, phase, revivals = None, None, 0.0, 0
    c_now = None
    for _ in range(40):
        idx = torch.randint(0, 2, (64,), generator=rng)
        batch = centers2[idx] + 0.25 * torch.randn(64, 2, generator=rng)
        c_now, _, p_sum, p_count, phase, revivals = update_emeans_prototypes(
            batch, p_sum, p_count, phase, revivals, max_prototypes=256
        )
    assert c_now is not None and c_now.shape[0] == 64

    # 1D Gaussian stationary -> shift (+1.0) tracks within < 30 steps.
    s_sum, s_count, s_phase, s_rev = None, None, 0.0, 0
    for _ in range(25):
        b1 = 2.0 + 0.25 * torch.randn(128, 1, generator=rng)
        _, _, s_sum, s_count, s_phase, s_rev = update_emeans_prototypes(
            b1, s_sum, s_count, s_phase, s_rev, max_prototypes=256
        )
    c_shift, w_shift = None, None
    for _ in range(30):
        b2 = 3.0 + 0.25 * torch.randn(128, 1, generator=rng)
        c_shift, w_shift, s_sum, s_count, s_phase, s_rev = update_emeans_prototypes(
            b2, s_sum, s_count, s_phase, s_rev, max_prototypes=256
        )
    assert c_shift is not None and w_shift is not None
    weighted_mean = float((w_shift * c_shift[:, 0]).sum())
    assert abs(weighted_mean - 3.0) < 0.08


def test_empf_v3_conditioned_fiber_pairing_and_schema2_checkpoint():
    """Verify conditioned fiber baseline cancellation, step_scale continuation, and schema-1 rejection."""
    recipe = get_recipe("empf", loss="relativistic")
    loss_fn = recipe.make_loss()
    assert isinstance(loss_fn, EMPFSinkhornRelativisticLoss)

    class FiberCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 1, bias=False)

        def forward(self, x, *, xt, t):
            # Massive context-dependent baseline offset cancels only under row-wise fiber pairing!
            baseline = 50.0 * xt[:, :1] - 25.0 * t.float().unsqueeze(-1)
            return (self.w(x) + baseline).squeeze(-1)

    g = nn.Linear(2, 2)
    d = FiberCritic()
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(d, seed=1)
    opt_g, opt_d = recipe.make_optimizers(g, d)
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)

    rng = torch.Generator().manual_seed(5)
    xt = torch.randn(32, 2, generator=rng)
    t_idx = torch.randint(0, 4, (32,), generator=rng)
    real = 0.5 * xt + 0.05 * torch.randn(32, 2, generator=rng)
    z = torch.randn(32, 2, generator=rng)
    fake = g(z)

    r_log = d(real, xt=xt, t=t_idx)
    f_log = d(fake, xt=xt, t=t_idx)
    assert getattr(r_log, "_empf_conditioned", False) is True
    assert getattr(f_log, "_empf_conditioned", False) is True

    # Verify baseline offset cancels identically in d_loss and g_loss:
    raw_diff = (d.w(real) - d.w(fake)).squeeze(-1)
    expected_d = torch.nn.functional.softplus(-raw_diff).mean()
    expected_g = torch.nn.functional.softplus(-(d.w(fake) - d.w(real).detach()).squeeze(-1)).mean()
    assert torch.allclose(loss_fn.d_loss(r_log, f_log.detach()), expected_d, atol=1e-6)
    gl = loss_fn.g_loss(f_log, r_log.detach())
    assert torch.allclose(gl, expected_g, atol=1e-6)

    # Settled conditional loop cools step_scale strictly below 1.0 and round-trips via schema-2 state_dict.
    for _ in range(80):
        pen = penalty(d, real, real + 1e-4 * torch.randn(32, 2, generator=rng), xt=xt, t=t_idx)
        opt_d.zero_grad()
        pen.backward()
        opt_d.step()
    assert 0.65 <= opt_d.record._step_scale < 0.99

    sd = deepcopy(opt_d.state_dict())
    assert sd["regularizer"]["record"]["empf"]["schema"] == 2

    # Schema-1 packets must be rejected rather than reconstructed.
    schema1_sd = deepcopy(sd)
    schema1_sd["regularizer"]["record"]["empf"]["schema"] = 1
    schema1_sd["regularizer"]["record"]["empf"].pop("ema_cond_energy")
    schema1_sd["regularizer"]["record"]["empf"].pop("step_scale")
    with pytest.raises(ValueError, match="fields|schema"):
        opt_d.load_state_dict(schema1_sd)

    # Fresh optimizer continuation preserves exact trajectory under step_scale < 1.
    g2, d2 = deepcopy(g), deepcopy(d)
    opt_g2, opt_d2 = recipe.make_optimizers(g2, d2)
    opt_g2.load_state_dict(deepcopy(opt_g.state_dict()))
    opt_d2.load_state_dict(sd)

    from particlegan.optim.constraint_geometry import constraint_geometry_backward

    z_step = torch.randn(32, 2, generator=rng)
    for gi, di, og, _ in ((g, d, opt_g, opt_d), (g2, d2, opt_g2, opt_d2)):
        og.zero_grad()
        fl = di(gi(z_step), xt=xt, t=t_idx)
        rl = di(real, xt=xt, t=t_idx).detach()
        lg = loss_fn.g_loss(fl, rl)
        constraint_geometry_backward(lg, og, (lg,))
        og.step()
    for p1, p2 in zip(g.parameters(), g2.parameters()):
        assert torch.equal(p1, p2)


def test_empf_v3_1d_thermal_and_step_scale_laws():
    """Verify 1D floor-subtracted thermal cooling, step-scale damping, and shock-filtered transport EMA."""
    from particlegan._empf.critic import compute_empf_step_scale, compute_thermal_scale

    # Deeply settled 1D cloud hits s_floor = 0.0012 and step_scale = 0.48.
    assert math.isclose(compute_thermal_scale(0.10, 128, 1), 0.0012, rel_tol=1e-6)
    assert math.isclose(compute_empf_step_scale(0.10, 128, 1), 0.48, rel_tol=1e-6)
    # Unsettled 1D cloud runs at full thermal and step scale (1.0).
    assert math.isclose(compute_thermal_scale(0.60, 128, 1), 1.0, rel_tol=1e-6)
    assert math.isclose(compute_empf_step_scale(0.25, 128, 1), 1.0, rel_tol=1e-6)
    # 2D+ clouds keep unit step_scale from compute_empf_step_scale.
    assert compute_empf_step_scale(0.05, 128, 2) == 1.0

    rec = EMPFStepRecord()
    rec.observe_transport(0.13, 0.50, ema_decay=0.92)
    assert math.isclose(rec.ema_residual, 0.13, rel_tol=1e-6)
    # Moderate single-batch quantile fluctuation (e.g. 0.55 < max(5.5*0.13, 0.13+0.55)=0.715) is EMA-filtered.
    rec.observe_transport(0.55, 0.50, ema_decay=0.92)
    assert rec.ema_residual < 0.20
    # Genuine shift shock (> 0.715) instant-reheats on step 1.
    rec.observe_transport(1.25, 0.50, ema_decay=0.92)
    assert math.isclose(rec.ema_residual, 1.25, rel_tol=1e-6)


def test_empf_v7_newton_schulz_hybrid_dense_profile_accuracy_and_checkpoint_roundtrip():
    """Verify hybrid dense polar bitwise parity on <=128, accuracy on >128, continuation, and cross-profile rejection."""
    from particlegan.optim.dualnorm import (
        NEWTON_SCHULZ_DENSE_PROFILE,
        NEWTON_SCHULZ_HYBRID_DENSE_PROFILE,
        _newton_schulz_polar,
    )

    recipe_hyb = get_recipe("empf")
    assert recipe_hyb.optimizer_dense_profile == NEWTON_SCHULZ_HYBRID_DENSE_PROFILE
    recipe_ns = get_recipe("empf", optimizer_dense_profile=NEWTON_SCHULZ_DENSE_PROFILE)
    recipe_ref = get_recipe("empf", optimizer_dense_profile="default")

    empty = torch.zeros(0, 4, dtype=torch.float32)
    assert _newton_schulz_polar(empty, smoothing=0.001).shape == (0, 4)

    rng = torch.Generator().manual_seed(23)
    # Small/medium matrices (min < 64 or max < 128) match default polar_factor bitwise.
    for shape in ((64, 64), (64, 127), (63, 128), (1, 128), (64, 2)):
        p_hyb = nn.Parameter(torch.zeros(shape))
        p_ref = nn.Parameter(torch.zeros(shape))
        opt_hyb = recipe_hyb.replace(constraint_geometry_mode="none").make_generator_optimizer([p_hyb])
        opt_ref = recipe_ref.replace(constraint_geometry_mode="none").make_generator_optimizer([p_ref])
        grad = torch.randn(shape, generator=rng) * torch.logspace(1, -6, shape[1])
        p_hyb.grad = grad.clone()
        p_ref.grad = grad.clone()
        opt_hyb.step()
        opt_ref.step()
        assert torch.equal(p_hyb, p_ref), f"expected exact parity on {shape}"

    # Large matrices (min >= 64 and max >= 128) use guarded Newton-Schulz and match polar_factor to <1e-4.
    for shape in ((128, 128), (64, 128), (168, 128), (256, 170)):
        p_hyb = nn.Parameter(torch.zeros(shape))
        p_ns = nn.Parameter(torch.zeros(shape))
        p_ref = nn.Parameter(torch.zeros(shape))
        opt_hyb = recipe_hyb.replace(constraint_geometry_mode="none").make_generator_optimizer([p_hyb])
        opt_ns = recipe_ns.replace(constraint_geometry_mode="none").make_generator_optimizer([p_ns])
        opt_ref = recipe_ref.replace(constraint_geometry_mode="none").make_generator_optimizer([p_ref])
        grad = torch.randn(shape, generator=rng)
        p_hyb.grad = grad.clone()
        p_ns.grad = grad.clone()
        p_ref.grad = grad.clone()
        opt_hyb.step()
        opt_ns.step()
        opt_ref.step()
        assert torch.equal(p_hyb, p_ns)
        rel_err = float((p_hyb.detach() - p_ref.detach()).norm() / p_ref.detach().norm().clamp_min(1e-12))
        assert rel_err < 1e-4, f"shape={shape} rel_err={rel_err}"

    # Checkpoint continuation and atomic cross-profile rejection.
    p1 = nn.Parameter(torch.zeros(16, 4))
    opt1 = recipe_hyb.replace(constraint_geometry_mode="none").make_generator_optimizer([p1])
    p1.grad = torch.randn(16, 4, generator=rng)
    opt1.step()
    saved = deepcopy(opt1.state_dict())
    assert saved["dualnorm"]["dense_profile"] == NEWTON_SCHULZ_HYBRID_DENSE_PROFILE

    for wrong_recipe in (recipe_ref, recipe_ns):
        p_wrong = nn.Parameter(torch.ones(16, 4))
        opt_wrong = wrong_recipe.replace(constraint_geometry_mode="none").make_generator_optimizer([p_wrong])
        before_p = p_wrong.detach().clone()
        with pytest.raises(ValueError, match="dense profile"):
            opt_wrong.load_state_dict(deepcopy(saved))
        assert torch.equal(p_wrong, before_p)

    p2 = nn.Parameter(p1.detach().clone())
    opt2 = recipe_hyb.replace(constraint_geometry_mode="none").make_generator_optimizer([p2])
    opt2.load_state_dict(saved)
    next_grad = torch.randn(16, 4, generator=rng)
    p1.grad = next_grad.clone()
    p2.grad = next_grad.clone()
    opt1.step()
    opt2.step()
    assert torch.equal(p1, p2)


def test_empf_v8_discrete_simplex_cooling_and_continuation():
    """Verify discrete-simplex post-acquisition cooling, shock reheat, and exact continuation across [0.064, 0.084]."""
    recipe = get_recipe("empf")
    rng = torch.Generator().manual_seed(17)
    # Construct 5 distinct 5-character word modes in 175-D concatenated simplex + encoder latent space (167 + 8 = 175).
    modes = torch.zeros(5, 175, dtype=torch.float32)
    for m in range(5):
        for pos in range(5):
            modes[m, pos * 26 + ((m * 5 + pos) % 26)] = 1.0
        modes[m, 167:] = (m - 2.0) * 0.5

    def sample_batch(active_modes: int, noise_std: float = 0.0) -> torch.Tensor:
        idx = torch.arange(64) % active_modes
        out = modes[idx].clone()
        if noise_std > 0.0:
            out = out + noise_std * torch.randn(out.shape, generator=rng)
        return out

    critic = nn.Linear(175, 1)
    init.deterministic_orthogonal_(critic, seed=0)
    opt = recipe.make_critic_optimizer(critic)
    pen = recipe.make_critic_penalty(opt, collect_stats=True)
    rec = pen.regularizer.record

    x_real = sample_batch(5)
    # 4-word partial coverage keeps Chamfer residual > 0.084 -> exact full DualNorm step (s_thermal == 1.0).
    x_fake_4 = sample_batch(4)
    _ = pen(critic, x_real, x_fake_4)
    assert rec.ema_residual > 0.084
    assert math.isclose(pen.last_stats["s_thermal"], 1.0, rel_tol=1e-6)

    # Transition to 5-word settled coverage: verify ema_decay=0.96 cools across (0.064, 0.084) into 0.0004
    # and preserves exact fresh-object checkpoint continuation mid-transition (both penalty and DualNorm step).
    x_fake_5 = sample_batch(5, noise_std=0.002)
    restored_pen: EMPFCriticPenalty | None = None
    restored_opt = None
    restored_critic: nn.Linear | None = None
    saw_mid_band = False
    for _ in range(80):
        opt.zero_grad()
        loss_a = critic(x_fake_5).mean() - critic(x_real).mean() + pen(critic, x_real, x_fake_5)
        loss_a.backward()
        opt.step()
        if restored_pen is not None and restored_opt is not None and restored_critic is not None:
            restored_opt.zero_grad()
            loss_b = (
                restored_critic(x_fake_5).mean()
                - restored_critic(x_real).mean()
                + restored_pen(restored_critic, x_real, x_fake_5)
            )
            loss_b.backward()
            restored_opt.step()
            assert torch.equal(loss_a.detach(), loss_b.detach())
            assert math.isclose(rec.ema_residual, restored_pen.regularizer.record.ema_residual, rel_tol=0.0, abs_tol=0.0)
            assert math.isclose(pen.last_stats["s_thermal"], restored_pen.last_stats["s_thermal"], rel_tol=0.0, abs_tol=0.0)
            for p_a, p_b in zip(critic.parameters(), restored_critic.parameters()):
                assert torch.equal(p_a, p_b)
        if not saw_mid_band and 0.064 < rec.ema_residual < 0.084:
            saw_mid_band = True
            assert 0.0004 < pen.last_stats["s_thermal"] < 1.0
            restored_critic = deepcopy(critic)
            restored_opt = recipe.make_critic_optimizer(restored_critic)
            restored_pen = recipe.make_critic_penalty(restored_opt, collect_stats=True)
            restored_opt.load_state_dict(deepcopy(opt.state_dict()))

    assert saw_mid_band, "expected cooling trajectory to traverse (0.064, 0.084)"
    assert rec.ema_residual < 0.064
    assert math.isclose(pen.last_stats["s_thermal"], 0.0004, rel_tol=1e-6)

    # Sudden distribution-shift shock (mode drop back to 2 words) instant-reheats s_thermal back to 1.0.
    x_fake_shock = sample_batch(2)
    _ = pen(critic, x_real, x_fake_shock)
    assert rec.ema_residual > 0.084
    assert math.isclose(pen.last_stats["s_thermal"], 1.0, rel_tol=1e-6)




