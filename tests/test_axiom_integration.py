"""End-to-end public-API, Forge contract, hygiene, continuation & clock-free tests for Axiom."""
from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

import particlegan as pg
from particlegan.optim.constraint_geometry import constraint_geometry_backward


REPO_ROOT = Path(__file__).resolve().parents[1]
AXIOM_DIR = REPO_ROOT / "particlegan" / "_axiom"
AXIOM_FILES = (
    AXIOM_DIR / "prior.py",
    AXIOM_DIR / "critic.py",
    AXIOM_DIR / "transport.py",
    AXIOM_DIR / "optimizer.py",
    AXIOM_DIR / "loss.py",
    AXIOM_DIR / "__init__.py",
    REPO_ROOT / "particlegan" / "axiom.py",
)

FORBIDDEN_IMPORT_FRAGMENTS = (
    "_unified",
    "_t1_",
    "_t2_",
    "_t3_",
    "_t4_",
    "_rhbi",
    "_empf",
    "_cond_fiber",
    "_minimal",
    "_bispf",
    "_nash_v3",
    "_r4_ablation",
    "_law4_energy",
    "bispf",
    "rhbi",
    "empf",
    "k3p",
    "ka2",
    "grad_regularizers",
    "unified",
    "minimal",
    "cond_fiber",
    "t1_coupling",
    "t2_critic",
    "t3_dynamics",
    "t4_arch",
    "t4_nash",
    "nash_v3",
)

FORBIDDEN_LINALG_CALLS = frozenset({"eigh", "eigvalsh", "svd", "svdvals", "pinv", "compile"})


class _VecG(nn.Module):
    def __init__(self, z_dim: int = 2, out_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _VecD(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _CondD(nn.Module):
    def __init__(self, x_dim: int = 2, c_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, *, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_feat = t.float().reshape(-1, 1)
        return self.net(torch.cat((x, xt, t_feat), dim=-1)).squeeze(-1)


class _TinyConvG(nn.Module):
    def __init__(self, z_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(z_dim, 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z).view(z.shape[0], 8, 4, 4))


class _TinyConvD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _init_deterministic(*modules: nn.Module, seed: int = 0) -> None:
    for idx, mod in enumerate(modules):
        pg.init.deterministic_orthogonal_(mod, seed=seed + idx * 17)


def _run_mog_trainer(steps: int, *, total_steps: int = 200, seed: int = 0):
    recipe = pg.get_recipe(
        "axiom",
        prior_kind="mog",
        sigma_rel=0.05,
        num_particles=32,
        z_dim=2,
        batch_size=32,
        total_steps=total_steps,
    )
    g = _VecG(2, 2, 24)
    d = _VecD(2, 24)
    prior = recipe.make_prior()
    _init_deterministic(g, d, prior, seed=seed)
    trainer = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=total_steps, seed=seed)
    data_rng = torch.Generator(device="cpu").manual_seed(seed + 999)
    centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    last = None
    for _ in range(steps):
        cid = torch.randint(0, 4, (32,), generator=data_rng)
        batch = centers[cid] + 0.1 * torch.randn((32, 2), generator=data_rng)
        last = trainer.step(batch)
    return trainer, last


# ---------------------------------------------------------------------------
# 1. Static Hygiene, LOC Cap, Recipe Contract & Torch-Free Import Checks
# ---------------------------------------------------------------------------


def test_axiom_hygiene_loc_imports_and_no_forbidden_linalg():
    total_loc = 0
    core_modules = ("prior", "critic", "transport", "optimizer")
    for path in AXIOM_FILES:
        assert path.exists(), f"missing Axiom module: {path}"
        src = path.read_text()
        lines = src.splitlines()
        total_loc += len(lines)
        tree = ast.parse(src, filename=str(path))
        stem = path.stem

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in FORBIDDEN_IMPORT_FRAGMENTS:
                        assert bad not in alias.name, f"{path.name} imports forbidden module {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for bad in FORBIDDEN_IMPORT_FRAGMENTS:
                    assert bad not in mod.split("."), f"{path.name} imports from forbidden module {mod}"
                # Enforce zero mutual imports among the 4 parallel core modules:
                if path.parent == AXIOM_DIR and stem in core_modules and node.level >= 1:
                    first = mod.split(".")[0] if mod else ""
                    assert first not in core_modules, (
                        f"core module {path.name} must not import sibling core module {first}"
                    )
            elif isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_LINALG_CALLS, (
                    f"{path.name} references forbidden op .{node.attr}"
                )

    assert total_loc <= 2500, f"Axiom codebase exceeded 2,500 LOC cap: {total_loc}"


def test_recipe_invariants_179_fields_default_unchanged_and_torch_free():
    assert len(dataclasses.fields(pg.Recipe)) == 179
    assert pg.DEFAULT_RECIPE == "axiom"

    rec = pg.get_recipe("axiom")
    assert rec.critic_formulation == "axiom"
    assert rec.effective_critic_formulation == "axiom"
    assert rec.prior_regularizer == "axiom"
    assert rec.optimizer_family == "axiom"
    assert rec.constraint_geometry_mode == "direction_blend"
    assert rec.reg_every == 1
    assert rec.lr_floor == 1.0
    assert rec.resolved_network_lr_floor == 1.0

    # Verify standalone particlegan/recipes.py loads without importing torch:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, sys, dataclasses; "
                "spec = importlib.util.spec_from_file_location('r_only', 'particlegan/recipes.py'); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "assert 'torch' not in sys.modules; "
                "assert len(dataclasses.fields(m.Recipe)) == 179; "
                "assert m.DEFAULT_RECIPE == 'axiom'; "
                "assert m.get_recipe_fields('axiom')['critic_formulation'] == 'axiom'"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"recipes.py Torch-free check failed:\n{proc.stderr}\n{proc.stdout}"


def test_forge_idea_studies_and_clockfree_source_audit():
    from experiments.forge.clockfree import source_audit
    from experiments.forge.contracts import validate_idea
    from experiments.forge.preflight import task_preflight
    from experiments.forge.studies import validate_study
    from experiments.forge.techniques import recipe_field_active, technique_signature
    from experiments.forge.views import load_tasks

    tasks = load_tasks(REPO_ROOT)
    for idea_name, study_glob, min_studies in (
        ("axiom-v1", "axiom-v1-*.json", 4),
        ("axiom-v4", "axiom-v4-*.json", 9),
    ):
        idea_path = REPO_ROOT / "configs" / "forge" / "ideas" / f"{idea_name}.json"
        idea = json.loads(idea_path.read_text())
        validate_idea(idea)
        assert idea["id"] == idea_name
        assert idea.get("trainer_family") == "axiom"

        for tid in (
            "bytegan_hypercube8",
            "bytegan_bigan_dictionary",
            "t3_iso_scale_mass_8x",
            "t3_iso_horizon_saddle_shift",
            "dcgan_performance_r256",
            "grid100_res16_tanh",
            "fl_law1_aniso_unequal_modes",
            "fl_law3_highd_orphan_d32",
            "fl_law42_deep_resbn_and_conv1x1",
            "fl_law44_highd_sparse_prior_and_shock",
        ):
            assert task_preflight(tasks[tid], idea, {}, root=REPO_ROOT, tasks=tasks) == [], f"preflight failed for {idea_name}/{tid}"

        study_paths = sorted((REPO_ROOT / "configs" / "forge" / "studies").glob(study_glob))
        assert len(study_paths) >= min_studies
        for sp in study_paths:
            study = json.loads(sp.read_text())
            validate_study(study)
            assert study["candidate"] == idea_name

    rec = pg.get_recipe("axiom", total_steps=1000)
    audit = source_audit(rec.to_dict(), {})
    assert audit["unexplained_clock_dependencies"] == []
    for rel in (
        "particlegan/_axiom/prior.py",
        "particlegan/_axiom/critic.py",
        "particlegan/_axiom/transport.py",
        "particlegan/_axiom/optimizer.py",
        "particlegan/_axiom/loss.py",
    ):
        assert rel in audit["source_sha256"]

    sig = technique_signature(rec)
    assert sig["mechanisms"].get("axiom_optimizer") is True
    assert sig["mechanisms"].get("axiom_critic") is True
    assert sig["mechanisms"].get("axiom_transport") is True
    assert "t1c_transport" not in sig["mechanisms"]
    assert "t1c_chart_commitment" not in sig["mechanisms"]
    assert recipe_field_active("rhbi_fibre_rel_blend", rec) is True


def test_axiom_record_binding_not_shadowed_and_zero_legacy_import():
    check_script = (
        "import torch, particlegan as pg\n"
        "import particlegan.t1_coupling as legacy_t1c\n"
        "from benchmarks.toy_audit.api_t3_isolation import HOST_DEFINITIONS, T3IsolationFixture\n"
        "orig_bind = legacy_t1c.bind_t1_coupling_record\n"
        "def _boom(*args, **kwargs):\n"
        "    raise AssertionError('legacy bind_t1_coupling_record must not be called for axiom')\n"
        "legacy_t1c.bind_t1_coupling_record = _boom\n"
        "rec = pg.get_recipe('axiom', prior_kind='mog', sigma_rel=0.05, num_particles=16, z_dim=2, batch_size=16, total_steps=50)\n"
        "g = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.SiLU(), torch.nn.Linear(16, 2))\n"
        "d = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))\n"
        "prior = rec.make_prior()\n"
        "tr = pg.GANTrainer(rec, g, d, prior=prior, max_steps=50, seed=0)\n"
        "assert tr.loss._axiom_record is tr.opt_d.record\n"
        "assert tr.loss.record is tr.opt_d.record\n"
        "assert tr.prior_regularizer._axiom_record is tr.opt_d.record\n"
        "assert tr.prior_regularizer._bound_device is not None\n"
        "assert not hasattr(tr.loss, '_t1c_record')\n"
        "rec_adam = pg.get_recipe('axiom', optimizer_family='adam', optimizer_smoothing=0.0, constraint_geometry_mode='none', total_steps=50)\n"
        "opt_d_adam = rec_adam.make_critic_optimizer(d)\n"
        "assert type(opt_d_adam.record).__name__ == 'AxiomStepRecord'\n"
        "legacy_t1c.bind_t1_coupling_record = orig_bind\n"
        "res_t3 = HOST_DEFINITIONS['t3_iso_scale_mass_8x']['resources']\n"
        "rec_t3 = pg.get_recipe('axiom', num_particles=res_t3['num_particles'], z_dim=res_t3['z_dim'], batch_size=res_t3['batch_size'], total_steps=50)\n"
        "t3 = T3IsolationFixture('t3_iso_scale_mass_8x', rec_t3, device='cpu', seed=0, max_steps=50)\n"
        "assert t3.opt_d.ema_critic is None\n"
        "assert t3.loss.record is t3.opt_d.record\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", check_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"binding/import check failed:\n{proc.stderr}\n{proc.stdout}"


# ---------------------------------------------------------------------------
# 2. Public-API End-to-End Host Smoke Tests (CPU Deterministic)
# ---------------------------------------------------------------------------


def test_gan_trainer_mog_pullback_and_point_mass_smoke():
    trainer, last = _run_mog_trainer(steps=4, total_steps=100, seed=0)
    assert trainer.completed_steps == 4
    assert torch.isfinite(last["loss_d"]) and torch.isfinite(last["loss_g"])
    assert trainer.ema_D is None
    assert trainer.opt_g.record is None
    assert trainer.opt_d.record is not None
    assert trainer.opt_d.record.observed_steps == 4
    assert trainer.opt_d.record.calls == 4

    # Point-mass Case-A rail-escape smoke (raw_var == 0):
    recipe = pg.get_recipe("axiom", num_particles=16, z_dim=2, batch_size=16, total_steps=50)
    g, d = _VecG(2, 2, 16), _VecD(2, 16)
    prior = pg.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
    _init_deterministic(g, d, prior, seed=1)
    pt_trainer = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=50, seed=1)
    pt_batch = torch.full((16, 2), 0.75)
    for _ in range(3):
        out = pt_trainer.step(pt_batch)
        assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])


def test_zero_sigma_interp_manifold_and_exact_atomic_smoke():
    # Continuous curved arc (zero-sigma ParticlePrior -> simplicial manifold path):
    recipe = pg.get_recipe("axiom", num_particles=16, z_dim=2, batch_size=32, total_steps=100)
    g, d = _VecG(2, 2, 24), _VecD(2, 24)
    prior = pg.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
    _init_deterministic(g, d, prior, seed=2)
    trainer = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=100, seed=2)
    rng = torch.Generator().manual_seed(22)
    for _ in range(3):
        ang = torch.rand((32,), generator=rng) * 3.14159
        arc = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1) + 0.015 * torch.randn((32, 2), generator=rng)
        out = trainer.step(arc)
        assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])
    assert trainer.opt_d.record.atomic_ledger is None

    # Exact discrete 4-atom support (ByteGAN atomic bijection path):
    recipe_a = pg.get_recipe("axiom", num_particles=4, z_dim=2, batch_size=32, total_steps=100)
    g2, d2 = _VecG(2, 4, 24), _VecD(4, 24)
    prior2 = pg.ParticlePrior(num_particles=4, z_dim=2, learnable=True)
    _init_deterministic(g2, d2, prior2, seed=3)
    trainer2 = pg.GANTrainer(recipe_a, g2, d2, prior=prior2, max_steps=100, seed=3)
    atoms = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    rng2 = torch.Generator().manual_seed(33)
    for _ in range(4):
        ids = torch.randint(0, 4, (32,), generator=rng2)
        out2 = trainer2.step(atoms[ids])
        assert torch.isfinite(out2["loss_d"]) and torch.isfinite(out2["loss_g"])
    assert trainer2.opt_d.record.atomic_ledger is not None
    assert trainer2.opt_d.record.thermal_initialized is True


def test_conv2d_batchnorm_regime2_fwad_and_foreach_smoke():
    recipe = pg.get_recipe("axiom", num_particles=16, z_dim=8, batch_size=8, total_steps=50)
    g, d = _TinyConvG(8), _TinyConvD()
    prior = pg.ParticlePrior(num_particles=16, z_dim=8, learnable=True)
    _init_deterministic(g, d, prior, seed=4)
    trainer = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=50, seed=4)
    rng = torch.Generator().manual_seed(44)
    for _ in range(3):
        imgs = torch.randn((8, 1, 16, 16), generator=rng).clamp(-1.0, 1.0)
        out = trainer.step(imgs)
        assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])
    assert trainer.opt_d.record.calls == 3


def test_ddgan_continuous_conditioned_self_binding_smoke():
    recipe = pg.get_recipe("axiom", model="ddgan", num_particles=16, z_dim=2, batch_size=24, total_steps=100)
    g = _VecG(5, 2, 24)
    d = _CondD(2, 2, 24)
    prior = pg.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
    _init_deterministic(g, d, prior, seed=5)
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=deepcopy(d))
    loss = recipe.make_loss()
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)
    assert opt_d.ema_critic is None  # self-binding ignores caller ema_critic

    rng = torch.Generator().manual_seed(55)
    for _ in range(4):
        x0 = torch.randn((24, 2), generator=rng)
        xt = x0 + 0.5 * torch.randn((24, 2), generator=rng)
        t = torch.randint(1, 3, (24,), generator=rng)
        z, rows = prior.sample(24, generator=rng)
        g_in = torch.cat((z, xt, t.float().unsqueeze(-1)), dim=-1)
        fake = g(g_in)

        opt_d.zero_grad()
        dl = loss.d_loss(d(x0, xt=xt, t=t), d(fake.detach(), xt=xt, t=t))
        pen = penalty(d, x0, fake.detach(), xt=xt, t=t)
        (dl + pen).backward()
        opt_d.step()

        opt_g.zero_grad()
        opt_g.set_sampled_rows(prior.z, rows)
        gl = loss.g_loss(d(fake, xt=xt, t=t), d(x0, xt=xt, t=t))
        constraint_geometry_backward(gl, opt_g, (gl,))
        opt_g.step()
        assert torch.isfinite(dl) and torch.isfinite(pen) and torch.isfinite(gl)


def test_ae_gan_vae_gan_and_bigan_joint_g_loss_smoke():
    # AE-GAN & VAE-GAN recipe.encode + standalone AxiomPriorRegularizer:
    for enc_mode in ("ae", "categorical"):
        rec_enc = pg.get_recipe(
            "axiom", encoder_mode=enc_mode, prior_kind="mog", num_particles=16, z_dim=4, batch_size=16, total_steps=50
        )
        prior = pg.MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05, learnable=True)
        _init_deterministic(prior, seed=6)
        reg = rec_enc.make_prior_regularizer(prior=prior, reference=prior)
        reg_val = reg(prior.z)
        assert reg_val.ndim == 0 and torch.isfinite(reg_val)
        q = torch.randn((16, 4))
        if enc_mode == "ae":
            enc = rec_enc.encode(q, prior, offset=torch.zeros_like(q))
            assert enc.coord_whiten is True
        else:
            enc = rec_enc.encode(q, prior)
        assert torch.isfinite(enc.codes).all()

    # BiGAN joint_g_loss + InputNoise clean-input capture propagation:
    rec_bi = pg.get_recipe("axiom", num_particles=8, z_dim=2, batch_size=16, total_steps=50)
    g, e, d = _VecG(2, 4, 24), _VecG(4, 2, 24), _VecD(6, 24)
    prior = pg.ParticlePrior(num_particles=8, z_dim=2, learnable=True)
    _init_deterministic(g, e, d, prior, seed=7)
    opt_g, opt_d = rec_bi.make_optimizers(g, d, prior, encoder=e)
    loss = rec_bi.make_loss()
    penalty = rec_bi.make_critic_penalty(opt_d, collect_stats=True)
    noisy_d = pg.InputNoise(d, std=0.1, generator=torch.Generator().manual_seed(77))
    rng = torch.Generator().manual_seed(78)

    real_x = torch.randn((16, 4), generator=rng)
    enc_z = e(real_x)
    z, rows = prior.sample(16, generator=rng)
    fake_x = g(z)
    joint_real = torch.cat((real_x, enc_z.detach()), dim=-1)
    joint_fake = torch.cat((fake_x.detach(), z.detach()), dim=-1)

    opt_d.zero_grad()
    dl = loss.d_loss(noisy_d(joint_real), noisy_d(joint_fake)) + penalty(noisy_d, joint_real, joint_fake)
    dl.backward()
    opt_d.step()

    opt_g.zero_grad()
    opt_g.set_sampled_rows(prior.z, rows)
    joint_real_live = torch.cat((real_x, e(real_x)), dim=-1)
    joint_fake_live = torch.cat((g(z), z), dim=-1)
    gl = loss.joint_g_loss(noisy_d(joint_fake_live), noisy_d(joint_real_live))
    constraint_geometry_backward(gl, opt_g, (gl,))
    opt_g.step()
    assert torch.isfinite(dl) and torch.isfinite(gl)


def test_direction_blend_two_objective_nonascent_smoke():
    rec = pg.get_recipe("axiom", num_particles=8, z_dim=2, batch_size=16, total_steps=50)
    g = _VecG(2, 2, 16)
    _init_deterministic(g, seed=8)
    opt_g = rec.make_generator_optimizer(g)
    assert opt_g.direction_blend is True

    z = torch.randn((16, 2))
    out = g(z)
    l0 = (out - 1.5).square().mean()
    l1 = (out + 1.5).square().mean()
    total = l0 + l1
    opt_g.zero_grad()
    constraint_geometry_backward(total, opt_g, (l0, l1))
    opt_g.step()
    assert opt_g.constraint_geometry_stats["steps"] == 1
    assert opt_g.constraint_geometry_stats["max_derivative_after"] <= 1e-6


@pytest.mark.parametrize("fmt", ("mxfp8", "nvfp4"))
def test_emulated_low_precision_mxfp8_nvfp4_smoke(fmt: str):
    rec = pg.get_recipe("axiom", prior_kind="mog", sigma_rel=0.05, num_particles=16, z_dim=2, batch_size=16, total_steps=50)
    g, d = _VecG(2, 1, 16), _VecD(1, 16)
    prior = rec.make_prior()
    _init_deterministic(g, d, prior, seed=9)
    pg.emulate_linear_training_(g, format=fmt)
    pg.emulate_linear_training_(d, format=fmt)
    trainer = pg.GANTrainer(rec, g, d, prior=prior, max_steps=50, seed=9)
    rng = torch.Generator().manual_seed(99)
    for _ in range(3):
        batch = 2.0 + 0.5 * torch.randn((16, 1), generator=rng)
        out = trainer.step(batch)
        assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])


# ---------------------------------------------------------------------------
# 3. Fresh-Process Exact Checkpoint Continuation, Tamper Rejection & Clock-Free
# ---------------------------------------------------------------------------


def test_fresh_process_exact_checkpoint_continuation_and_tamper_rejection(tmp_path: Path):
    # Uninterrupted 6-step reference run:
    ref_trainer, ref_last = _run_mog_trainer(steps=6, total_steps=200, seed=0)

    # Interrupted 3-step prefix -> save checkpoint + data RNG -> resume 3 steps in fresh process:
    pre_trainer, _ = _run_mog_trainer(steps=3, total_steps=200, seed=0)
    ckpt_path = tmp_path / "axiom_resume.pt"
    out_path = tmp_path / "axiom_resumed_out.pt"
    data_rng = torch.Generator(device="cpu").manual_seed(999)
    centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    for _ in range(3):
        cid = torch.randint(0, 4, (32,), generator=data_rng)
        _ = centers[cid] + 0.1 * torch.randn((32, 2), generator=data_rng)

    saved = pre_trainer.state_dict()
    torch.save({"trainer": saved, "data_rng": data_rng.get_state()}, ckpt_path)

    # Tampered checkpoint must fail closed:
    tampered = deepcopy(saved)
    tampered["optimizers"][1]["axiom"]["schema"] = 999
    with pytest.raises(ValueError):
        pre_trainer.load_state_dict(tampered)
    tampered_prior = deepcopy(saved)
    tampered_prior["prior_regularizer"]["schema"] = 999
    with pytest.raises(ValueError):
        pre_trainer.load_state_dict(tampered_prior)

    resume_script = (
        "import torch, particlegan as pg\n"
        "from tests.test_axiom_integration import _VecG, _VecD, _init_deterministic\n"
        f"blob = torch.load({str(ckpt_path)!r}, weights_only=False)\n"
        "recipe = pg.get_recipe('axiom', prior_kind='mog', sigma_rel=0.05, num_particles=32, z_dim=2, batch_size=32, total_steps=200)\n"
        "g, d = _VecG(2, 2, 24), _VecD(2, 24)\n"
        "prior = recipe.make_prior()\n"
        "_init_deterministic(g, d, prior, seed=123)\n"
        "trainer = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=200, seed=123)\n"
        "trainer.load_state_dict(blob['trainer'])\n"
        "data_rng = torch.Generator(device='cpu')\n"
        "data_rng.set_state(blob['data_rng'])\n"
        "centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])\n"
        "for _ in range(3):\n"
        "    cid = torch.randint(0, 4, (32,), generator=data_rng)\n"
        "    last = trainer.step(centers[cid] + 0.1 * torch.randn((32, 2), generator=data_rng))\n"
        "torch.save({\n"
        "    'g': {k: v.detach().cpu() for k, v in trainer.G.state_dict().items()},\n"
        "    'd': {k: v.detach().cpu() for k, v in trainer.D.state_dict().items()},\n"
        "    'z': trainer.prior.z.detach().cpu(),\n"
        "    'loss_d': float(last['loss_d']),\n"
        "    'loss_g': float(last['loss_g']),\n"
        f"}}, {str(out_path)!r})\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", resume_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"fresh-process resume failed:\n{proc.stderr}\n{proc.stdout}"
    resumed = torch.load(out_path, weights_only=False)

    assert resumed["loss_d"] == pytest.approx(float(ref_last["loss_d"]), abs=0.0, rel=0.0)
    assert resumed["loss_g"] == pytest.approx(float(ref_last["loss_g"]), abs=0.0, rel=0.0)
    assert torch.equal(resumed["z"], ref_trainer.prior.z.detach().cpu())
    for k, v in ref_trainer.G.state_dict().items():
        assert torch.equal(resumed["g"][k], v.detach().cpu()), f"G param {k} diverged after fresh-process resume"
    for k, v in ref_trainer.D.state_dict().items():
        assert torch.equal(resumed["d"][k], v.detach().cpu()), f"D param {k} diverged after fresh-process resume"


def test_clockfree_horizon_invariance():
    t_short, l_short = _run_mog_trainer(steps=5, total_steps=100, seed=0)
    t_long, l_long = _run_mog_trainer(steps=5, total_steps=5000, seed=0)

    assert float(l_short["loss_d"]) == float(l_long["loss_d"])
    assert float(l_short["loss_g"]) == float(l_long["loss_g"])
    assert torch.equal(t_short.prior.z, t_long.prior.z)
    for (k, v1), (_, v2) in zip(t_short.G.state_dict().items(), t_long.G.state_dict().items()):
        assert torch.equal(v1, v2), f"clock-dependent drift in G.{k}"
    for (k, v1), (_, v2) in zip(t_short.D.state_dict().items(), t_long.D.state_dict().items()):
        assert torch.equal(v1, v2), f"clock-dependent drift in D.{k}"


def test_v4_ddgan_detached_capture_ring_and_separate_wordfixture_optimizers():
    # 1. DDGAN detached real_logits survives via AxiomStepRecord._capture_ring / lookup_capture!
    recipe = pg.get_recipe("axiom", model="ddgan", num_particles=16, z_dim=2, batch_size=24, total_steps=100)
    g, d = _VecG(5, 2, 24), _CondD(2, 2, 24)
    prior = pg.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
    _init_deterministic(g, d, prior, seed=401)
    opt_g, opt_d = recipe.make_optimizers(g, d, prior)
    loss = recipe.make_loss()
    rng = torch.Generator().manual_seed(402)
    x0 = torch.randn((24, 2), generator=rng)
    xt = x0 + 0.4 * torch.randn((24, 2), generator=rng)
    t = torch.randint(1, 3, (24,), generator=rng)
    z, rows = prior.sample(24, generator=rng)
    fake = g(torch.cat((z, xt, t.float().unsqueeze(-1)), dim=-1))
    d_real_det = d(x0, xt=xt, t=t).detach()  # strips _axiom_* Python attributes!
    assert not hasattr(d_real_det, "_axiom_input")
    opt_g.zero_grad()
    opt_g.set_sampled_rows(prior.z, rows)
    gl = loss.g_loss(d(fake, xt=xt, t=t), d_real_det)
    gl.backward()
    assert torch.isfinite(gl) and opt_d.record.cond_initialized is True

    # 2. WordFixture separate make_generator_optimizer / make_critic_optimizer path recovers prior rows via grad_fn BFS!
    rec_w = pg.get_recipe("axiom", num_particles=8, z_dim=2, batch_size=32, total_steps=100)
    gw, ew, dw = _VecG(2, 4, 24), _VecG(4, 2, 24), _VecD(6, 24)
    pw = pg.ParticlePrior(num_particles=8, z_dim=2, learnable=True)
    _init_deterministic(gw, ew, dw, pw, seed=403)
    groups_w = [
        {"params": list(gw.parameters()), "lr": rec_w.lr},
        {"params": list(ew.parameters()), "lr": rec_w.lr},
        {"params": list(pw.parameters()), "lr": rec_w.lr * rec_w.prior_lr_mult},
    ]
    opt_gw = rec_w.make_generator_optimizer(groups_w, latent_table=pw.z)
    opt_dw = rec_w.make_critic_optimizer(dw)
    loss_w = rec_w.make_loss()
    assert opt_gw._axiom_record is None
    atoms = torch.eye(4)
    ids = torch.arange(32) % 4
    rx = atoms[ids]
    z_w, rows_w = pw.sample(32, generator=rng)
    opt_gw.zero_grad()
    opt_gw.set_sampled_rows(pw.z, rows_w)
    jf = torch.cat((gw(z_w), z_w), dim=-1)
    jr = torch.cat((rx, ew(rx)), dim=-1)
    gl_w = loss_w.joint_g_loss(dw(jf), dw(jr))
    constraint_geometry_backward(gl_w, opt_gw, (gl_w,))
    opt_gw.step()
    assert torch.isfinite(gl_w)
    assert opt_dw.record.atomic_ledger is not None
    assert getattr(pw.z, "_axiom_record", None) is opt_dw.record
