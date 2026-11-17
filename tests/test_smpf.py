"""Unit, continuation, and semantic-control tests for Scale- & Distortion-Aware Moving Prototype Flow (SMPF)."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    SMPFCriticAdam,
    SMPFCriticPenalty,
    SMPFGANLoss,
    SMPFPriorRegularizer,
    SMPFStepRecord,
    get_recipe,
    init,
)
from particlegan._stats import pairwise_sq_dist
from particlegan.smpf import (
    d15_initial_prototypes,
    smeans_effective_alpha,
    tiled_smeans_assign,
    update_smeans_prototypes,
)


def test_smpf_preset_and_incumbent_defaults_unchanged():
    assert DEFAULT_RECIPE == "empf"
    assert get_recipe() == get_recipe("empf")

    smpf = get_recipe("smpf")
    assert smpf.effective_critic_formulation == "smpf"
    assert smpf.prior_regularizer == "smpf"
    assert smpf.optimizer_family == "dualnorm"
    assert smpf.reg_anchor_weight == 0.0
    assert smpf.lr_floor == 1.0 and smpf.network_lr_floor == 1.0

    mog_smpf = get_recipe("mog", trainer="smpf")
    assert mog_smpf.effective_critic_formulation == "smpf"
    assert mog_smpf.prior_regularizer == "smpf"

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe(
            "smpf",
            optimizer_family="formulation",
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_dense_profile="default",
            optimizer_svd_backend="native",
            training_precision="float32",
            constraint_geometry_mode="none",
            reg_anchor_weight=0.5,
        )
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe(
            "smpf",
            reg_arm="b_cap",
            optimizer_family="adam",
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_dense_profile="default",
            optimizer_svd_backend="native",
            training_precision="float32",
            constraint_geometry_mode="none",
        )


def test_tiled_assign_exact_and_d15_seeding_zero_rng_rare_cluster_coverage():
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(197, 17, generator=rng)
    c = torch.randn(53, 17, generator=rng)

    labels_tiled, sq_tiled = tiled_smeans_assign(x, c, tile_rows=31, tile_cols=19)
    exact_d2 = pairwise_sq_dist(x, c)
    exact_sq, exact_labels = exact_d2.min(dim=1)
    assert torch.equal(labels_tiled, exact_labels)
    assert torch.allclose(sq_tiled, exact_sq, atol=1e-4, rtol=1e-4)

    # Verify zero global RNG consumption and rare 2% cluster discovery under D^1.5 seeding.
    heavy = torch.tensor([-2.0, -2.0]) + 0.08 * torch.randn(196, 2, generator=rng)
    rare = torch.tensor([3.0, 3.0]) + 0.05 * torch.randn(4, 2, generator=rng)
    skewed = torch.cat([heavy, rare], dim=0)

    rng_state_before = torch.get_rng_state()
    seeded = d15_initial_prototypes(skewed, k=16, rounds=4, power=1.5)
    rng_state_after = torch.get_rng_state()
    assert torch.equal(rng_state_before, rng_state_after)
    assert seeded.shape == (16, 2)
    min_dist_to_rare = float((seeded - torch.tensor([3.0, 3.0])).norm(dim=1).min())
    assert min_dist_to_rare < 0.25, f"Expected D^1.5 seeding to cover the 2% rare cluster, got {min_dist_to_rare}"


def test_smeans_distortion_anisotropic_revival_and_empirical_bayes_shrinkage():
    # Power-cosine settling reduces alpha smoothly at low residual while keeping acquisition responsive.
    alpha_acq = smeans_effective_alpha(128, 64, ema_residual=0.50)
    alpha_settled = smeans_effective_alpha(128, 64, ema_residual=0.005)
    assert alpha_acq > 3.0 * alpha_settled

    # Construct two equal-mass clusters: one tight (sigma=0.02 at (-3,0)) and one broad/anisotropic
    # (sigma_x=1.2, sigma_y=0.05 at (+3,0)), plus initially dead prototypes at (-20, -20).
    rng = torch.Generator().manual_seed(5)
    c_init = torch.tensor(
        [
            [-3.0, 0.0],  # tight low-SSE cluster
            [3.0, 0.0],   # high-SSE stretched cluster
            [-20.0, -20.0],
            [-20.0, -20.0],
        ],
        dtype=torch.float32,
    )
    u_init = torch.tensor([16.0, 16.0, 0.01, 0.01], dtype=torch.float32)
    sse_init = torch.tensor([0.02, 24.0, 1e-6, 1e-6], dtype=torch.float32)
    fscale_init = torch.tensor([1.44, 0.0025], dtype=torch.float32)
    fsum_init = c_init * u_init.unsqueeze(1)
    fcnt_init = u_init.clone()

    tight_batch = torch.tensor([-3.0, 0.0]) + torch.tensor([0.02, 0.02]) * torch.randn(32, 2, generator=rng)
    broad_batch = torch.tensor([3.0, 0.0]) + torch.tensor([1.20, 0.05]) * torch.randn(32, 2, generator=rng)
    x_batch = torch.cat([tight_batch, broad_batch], dim=0)

    (
        c_eb,
        w_out,
        ell_out,
        new_c,
        new_u,
        new_sse,
        new_fscale,
        new_fsum,
        new_fcnt,
        phase,
        revivals,
    ) = update_smeans_prototypes(
        x_batch,
        c_init,
        u_init,
        sse_init,
        fscale_init,
        fsum_init,
        fcnt_init,
        0.0,
        0,
        ema_residual=0.25,
        max_prototypes=4,
    )
    assert revivals == 1
    assert math.isclose(float(w_out.sum()), 1.0, rel_tol=1e-6)
    # Second batch splits the remaining dead prototype off the high-SSE (+3, 0) cluster while leaving
    # the converged tight (-3, 0) cluster untouched.
    (
        c_eb,
        w_out,
        ell_out,
        new_c,
        new_u,
        new_sse,
        new_fscale,
        new_fsum,
        new_fcnt,
        phase,
        revivals,
    ) = update_smeans_prototypes(
        x_batch,
        new_c,
        new_u,
        new_sse,
        new_fscale,
        new_fsum,
        new_fcnt,
        phase,
        revivals,
        ema_residual=0.25,
        max_prototypes=4,
    )
    assert revivals == 2
    assert math.isclose(float(w_out.sum()), 1.0, rel_tol=1e-6)
    # Revived prototypes must split the high-SSE (+3, 0) cluster rather than the tight (-3, 0) cluster!
    near_broad = int(((c_eb[:, 0] - 3.0).abs() < 2.5).sum().item())
    near_tight = int(((c_eb[:, 0] + 3.0).abs() < 0.5).sum().item())
    assert near_broad == 3 and near_tight == 1


def test_smpf_loss_capture_ring_survives_detach_on_conditioned_and_unconditioned():
    recipe = get_recipe("smpf")
    loss_fn = recipe.make_loss()
    assert isinstance(loss_fn, SMPFGANLoss)

    class CondCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))

        def forward(self, x, *, xt, t):
            return self.net(torch.cat([x, xt, t[:, None].to(x) / 4.0], dim=1)).squeeze(-1)

    critic = CondCritic()
    init.deterministic_orthogonal_(critic, seed=0)
    opt_d = recipe.make_critic_optimizer(critic)
    bind_rec = opt_d.record
    loss_fn.smpf_record = bind_rec
    pen = recipe.make_critic_penalty(opt_d)
    assert isinstance(pen, SMPFCriticPenalty)

    rng = torch.Generator().manual_seed(9)
    xt = torch.randn(32, 2, generator=rng)
    t = torch.randint(1, 5, (32,), generator=rng)
    real = 0.8 * xt + 0.1 * torch.randn(32, 2, generator=rng)
    fake = (0.2 * xt + 0.3 * torch.randn(32, 2, generator=rng)).requires_grad_(True)

    # Caller detaches real critic output just like api_model_ddgan.py!
    gl = loss_fn.g_loss(critic(fake, xt=xt, t=t), critic(real, xt=xt, t=t).detach())
    assert torch.isfinite(gl)
    gl.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert float(fake.grad.abs().sum().item()) > 0.0
    # Conditional call must not initialize marginal prototype bank.
    assert bind_rec.proto_initialized is False


def test_smpf_prior_regularizer_dof_bypass_and_outlier_tether():
    reg = SMPFPriorRegularizer(None, weight=1.0, k_nn=5)
    tiny = torch.randn(5, 2, requires_grad=True)
    assert float(reg(tiny).detach()) == 0.0

    rng = torch.Generator().manual_seed(11)
    left = torch.tensor([-1.5, 0.0]) + 0.05 * torch.randn(16, 2, generator=rng)
    right = torch.tensor([1.5, 0.0]) + 0.05 * torch.randn(16, 2, generator=rng)
    clean_cloud = torch.cat([left, right], dim=0).requires_grad_(True)
    loss_clean = reg(clean_cloud)

    stranded_data = clean_cloud.detach().clone()
    stranded_data[0] = torch.tensor([12.0, 0.0])
    stranded = stranded_data.requires_grad_(True)
    loss_stranded = reg(stranded)
    assert float(loss_stranded.detach()) > float(loss_clean.detach())
    loss_stranded.backward()
    assert stranded.grad is not None and float(stranded.grad[0, 0]) > 0.0


@pytest.mark.parametrize("optimizer_family", ["dualnorm", "formulation", "adam"])
def test_smpf_checkpoint_continuation_and_corruption_rejection(optimizer_family):
    overrides = dict(
        critic_formulation="smpf",
        prior_regularizer="smpf",
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
        training_precision="float32",
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
    recipe = get_recipe("smpf", **overrides)

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
        assert isinstance(t1.opt_d, SMPFCriticAdam)
        assert isinstance(t1.opt_d.record, SMPFStepRecord)

    rng1 = torch.Generator().manual_seed(19)
    for step in range(12):
        shift = -2.5 if step < 5 else 2.5
        t1.step(shift + 0.1 * torch.randn(16, 2, generator=rng1))

    state_before_sample = deepcopy(t1.state_dict())
    _ = t1.sample(24)
    state_after_sample = t1.state_dict()
    assert torch.equal(
        state_before_sample["optimizers"][1]["regularizer"]["record"]["smpf"]["proto_centroids"],
        state_after_sample["optimizers"][1]["regularizer"]["record"]["smpf"]["proto_centroids"],
    )

    buf = io.BytesIO()
    torch.save({"trainer": state_after_sample, "rng": rng1.get_state()}, buf)
    buf.seek(0)
    saved = torch.load(buf, weights_only=True)

    for _ in range(8):
        t1.step(2.5 + 0.1 * torch.randn(16, 2, generator=rng1))

    t2 = GANTrainer(recipe, g2, d2, prior=p2, seed=77)
    bad_state = deepcopy(saved["trainer"])
    bad_state["optimizers"][1]["regularizer"]["record"]["smpf"]["thermal_scale"] = 2.5
    with pytest.raises(ValueError, match="thermal"):
        t2.opt_d.load_state_dict(bad_state["optimizers"][1])

    bad_sse = deepcopy(saved["trainer"])
    bad_sse["optimizers"][1]["regularizer"]["record"]["smpf"]["proto_sse"][0] = -1.0
    with pytest.raises(ValueError, match="sse|positive"):
        t2.opt_d.load_state_dict(bad_sse["optimizers"][1])

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


def test_smpf_deliberate_damage_instance_isolation_and_import_order():
    recipe = get_recipe(
        "smpf",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=80,
        lr=0.008,
        prior_lr_mult=0.02,
        training_precision="float32",
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
    for _ in range(55):
        trainer_main.step(target_center + 0.08 * torch.randn(32, 2, generator=rng))
        trainer_other.step(torch.tensor([-50.0, 50.0]) + 5.0 * torch.randn(32, 2, generator=rng))

    trained_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())
    assert trained_err < 0.35 * init_err

    # Deliberate damage to served generator weights must degrade sample() immediately.
    with torch.no_grad():
        for param in trainer_main.G.parameters():
            param.fill_(25.0)
    damaged_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())
    assert damaged_err > 10.0 * trained_err

    # Fresh-process import order check: incumbent hashes must stay byte-identical.
    root = str(Path(__file__).resolve().parents[1])
    probe_a = (
        "import json, particlegan.smpf, particlegan.recipes as r; "
        "from experiments.forge.contracts import stable_hash; "
        "from experiments.forge.techniques import technique_signature; "
        "print(json.dumps({'default': r.DEFAULT_RECIPE, "
        "'hashes': {k: stable_hash(r.get_recipe(k).to_dict()) for k in ('bispf_optimized', 'bispf', 'bcap', 'empf', 'rhbi_optimized', 'smpf')}, "
        "'sigs': {k: technique_signature(r.get_recipe(k))['digest'] for k in ('empf', 'smpf', 'rhbi_optimized')}}))"
    )
    probe_b = (
        "import json, particlegan.recipes as r; "
        "from experiments.forge.contracts import stable_hash; "
        "from experiments.forge.techniques import technique_signature; "
        "import particlegan.smpf; "
        "print(json.dumps({'default': r.DEFAULT_RECIPE, "
        "'hashes': {k: stable_hash(r.get_recipe(k).to_dict()) for k in ('bispf_optimized', 'bispf', 'bcap', 'empf', 'rhbi_optimized', 'smpf')}, "
        "'sigs': {k: technique_signature(r.get_recipe(k))['digest'] for k in ('empf', 'smpf', 'rhbi_optimized')}}))"
    )
    out_a = json.loads(subprocess.check_output([sys.executable, "-c", probe_a], cwd=root, text=True))
    out_b = json.loads(subprocess.check_output([sys.executable, "-c", probe_b], cwd=root, text=True))
    assert out_a == out_b
    assert out_a["default"] == "empf"
    assert out_a["hashes"]["bispf_optimized"].startswith("a59d78806e67")
    assert out_a["hashes"]["bispf"].startswith("6bb9c45dfbdb")
    assert out_a["hashes"]["bcap"].startswith("1c0aeefa46b0")
    assert out_a["sigs"]["smpf"] != out_a["sigs"]["empf"]


def test_smpf_deepcopy_mid_graph_survives_nonleaf_capture():
    recipe = get_recipe(
        "smpf",
        num_particles=16,
        z_dim=4,
        batch_size=8,
        total_steps=10,
        training_precision="float32",
    )
    g = nn.Sequential(nn.Linear(4, 16), nn.LeakyReLU(0.2), nn.Linear(16, 6))
    d = nn.Sequential(nn.Linear(6, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    p = recipe.make_prior()
    trainer = GANTrainer(recipe, g, d, prior=p, seed=0)
    trainer.step(torch.randn(8, 6))

    # Populate _capture_ring with live non-leaf autograd tensors from G(z).
    z_live = p.z[:8]
    fake_live = g(z_live)
    logits_live = d(fake_live)
    assert logits_live.requires_grad and not fake_live.is_leaf

    d_copy = deepcopy(trainer.D)
    opt_d_copy = deepcopy(trainer.opt_d)
    rec_copy = deepcopy(trainer.opt_d.record)
    assert len(rec_copy._capture_ring) == 0
    assert d_copy._smpf_record.proto_initialized is True
    assert opt_d_copy.record.proto_centroids is not None
