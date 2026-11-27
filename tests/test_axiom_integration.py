"""End-to-end public-API, Forge contract, hygiene, complexity, host-sync & continuation tests for Axiom v6."""

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
from particlegan._axiom.atlas import PrototypeAtlas
from particlegan._axiom.fiber import AxiomCondFiberBank
from particlegan._axiom.optimizer import AxiomOptimizer
from particlegan._axiom.prior import AxiomPriorRegularizer
from particlegan._axiom.state import AxiomStepRecord, get_tensor_record
from particlegan.optim.constraint_geometry import constraint_geometry_backward

REPO_ROOT = Path(__file__).resolve().parents[1]
AXIOM_DIR = REPO_ROOT / "particlegan" / "_axiom"
AXIOM_FILES = (
    AXIOM_DIR / "math_ops.py",
    AXIOM_DIR / "state.py",
    AXIOM_DIR / "atlas.py",
    AXIOM_DIR / "fiber.py",
    AXIOM_DIR / "critic.py",
    AXIOM_DIR / "optimizer.py",
    AXIOM_DIR / "prior.py",
    AXIOM_DIR / "transport.py",
    AXIOM_DIR / "loss.py",
    AXIOM_DIR / "__init__.py",
    REPO_ROOT / "particlegan" / "axiom.py",
)

ALLOWED_DAG_IMPORTS: dict[str, frozenset[str]] = {
    "math_ops": frozenset(),
    "state": frozenset({"math_ops"}),
    "atlas": frozenset({"math_ops", "state"}),
    "fiber": frozenset({"math_ops", "state"}),
    "critic": frozenset({"math_ops", "state"}),
    "optimizer": frozenset({"math_ops", "state"}),
    "prior": frozenset({"math_ops", "state", "atlas"}),
    "transport": frozenset({"math_ops", "state", "atlas", "fiber"}),
    "loss": frozenset({"math_ops", "state", "transport"}),
}

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
FORBIDDEN_IDENTIFIERS = ("_bn_net_ids", "_EYE_CACHE", "_spatial_cluster_ratio")


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
    g, d = _VecG(2, 2, 24), _VecD(2, 24)
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


def _function_complexity(fn: ast.AST) -> tuple[int, int]:
    """Compute (Ruff-equivalent C901, Radon-equivalent BoolCC) for a single function."""
    c901 = 1
    bool_cc = 1
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)):
            c901 += 1
            bool_cc += 1
        elif isinstance(node, ast.IfExp):
            bool_cc += 1
        elif isinstance(node, ast.comprehension):
            c901 += 1 + len(node.ifs)
            bool_cc += 1 + len(node.ifs)
        elif isinstance(node, ast.BoolOp):
            bool_cc += len(node.values) - 1
        elif isinstance(node, ast.Match):
            cases = max(0, len(node.cases) - 1)
            c901 += cases
            bool_cc += cases
        stack.extend(ast.iter_child_nodes(node))
    return c901, bool_cc


def _check_single_file_hygiene(path: Path) -> None:
    assert path.exists(), f"missing Axiom module: {path}"
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    for lineno, line in enumerate(lines, start=1):
        assert len(line) <= 120, f"{path.name}:{lineno} exceeds 120 chars ({len(line)})"
    for bad_id in FORBIDDEN_IDENTIFIERS:
        assert bad_id not in src, f"{path.name} contains forbidden legacy identifier {bad_id}"
    tree = ast.parse(src, filename=str(path))
    stem = path.stem
    stmt_lines: dict[int, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and hasattr(node, "lineno") and hasattr(node, "col_offset"):
            prev_col = stmt_lines.get(node.lineno)
            if prev_col is not None and prev_col != node.col_offset:
                raise AssertionError(f"{path.name}:{node.lineno} has semicolon-packed statements")
            stmt_lines[node.lineno] = node.col_offset
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in FORBIDDEN_IMPORT_FRAGMENTS:
                    assert bad not in alias.name, f"{path.name} imports forbidden module {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for bad in FORBIDDEN_IMPORT_FRAGMENTS:
                assert bad not in mod.split("."), f"{path.name} imports from forbidden module {mod}"
            if path.parent == AXIOM_DIR and stem in ALLOWED_DAG_IMPORTS and node.level == 1:
                first = mod.split(".")[0] if mod else ""
                assert first in ALLOWED_DAG_IMPORTS[stem], (
                    f"DAG violation: {path.name} imports .{first} (allowed: {sorted(ALLOWED_DAG_IMPORTS[stem])})"
                )
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_LINALG_CALLS, f"{path.name} references forbidden op .{node.attr}"
            if node.attr == "grad" and isinstance(node.value, ast.Attribute) and node.value.attr == "autograd":
                raise AssertionError(f"{path.name} calls forbidden torch.autograd.grad")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            c901, bool_cc = _function_complexity(node)
            assert c901 <= 10, f"{path.name}:{node.name} C901={c901} > 10"
            assert bool_cc <= 15, f"{path.name}:{node.name} BoolCC={bool_cc} > 15"


# ---------------------------------------------------------------------------
# 1. Static Hygiene, Layered DAG, Dual Complexity, Recipe & Forge Contracts
# ---------------------------------------------------------------------------


def test_axiom_v6_ast_hygiene_dag_and_complexity() -> None:
    for path in AXIOM_FILES:
        _check_single_file_hygiene(path)


def test_recipe_invariants_179_fields_default_and_torch_free() -> None:
    assert len(dataclasses.fields(pg.Recipe)) == 179
    assert pg.DEFAULT_RECIPE == "axiom"
    assert pg.get_recipe() == pg.get_recipe("axiom")
    assert pg.get_recipe("gan") == pg.get_recipe("axiom")

    rec = pg.get_recipe("axiom")
    assert rec.critic_formulation == "axiom"
    assert rec.effective_critic_formulation == "axiom"
    assert rec.prior_regularizer == "axiom"
    assert rec.optimizer_family == "axiom"
    assert rec.constraint_geometry_mode == "direction_blend"
    assert rec.reg_every == 1
    assert rec.lr_floor == 1.0
    assert rec.resolved_network_lr_floor == 1.0

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


def test_forge_ideas_clockfree_source_audit_and_technique_signature() -> None:
    from experiments.forge.clockfree import source_audit
    from experiments.forge.contracts import validate_idea
    from experiments.forge.preflight import task_preflight
    from experiments.forge.techniques import recipe_field_active, technique_signature
    from experiments.forge.views import load_tasks

    tasks = load_tasks(REPO_ROOT)
    preflight_tasks = (
        "bytegan_hypercube8",
        "bytegan_bigan_dictionary",
        "t3_iso_scale_mass_8x",
        "t3_iso_horizon_saddle_shift",
        "dcgan_performance_r256",
        "grid100_res16_tanh",
        "fl_law1_aniso_unequal_modes",
        "fl_law2_groupnorm_and_vit_sprites",
        "fl_law3_highd_orphan_d32",
        "fl_law41_trust_rare1pct_tanh",
        "fl_law42_deep_resbn_and_conv1x1",
        "fl_law44_highd_sparse_prior_and_shock",
        "vector_spiral",
        "img_blobs4",
    )
    for idea_name in ("axiom-v1", "axiom-v4", "axiom-v5", "axiom-v6"):
        idea_path = REPO_ROOT / "configs" / "forge" / "ideas" / f"{idea_name}.json"
        idea = json.loads(idea_path.read_text(encoding="utf-8"))
        validate_idea(idea)
        assert idea["id"] == idea_name
        assert idea.get("trainer_family") == "axiom"
        for tid in preflight_tasks:
            errs = task_preflight(tasks[tid], idea, {}, root=REPO_ROOT, tasks=tasks)
            assert errs == [], f"preflight failed for {idea_name}/{tid}: {errs}"

    rec = pg.get_recipe("axiom", total_steps=1000)
    audit = source_audit(rec.to_dict(), {})
    assert audit["unexplained_clock_dependencies"] == []
    for rel in (
        "particlegan/_axiom/math_ops.py",
        "particlegan/_axiom/state.py",
        "particlegan/_axiom/atlas.py",
        "particlegan/_axiom/fiber.py",
        "particlegan/_axiom/prior.py",
        "particlegan/_axiom/critic.py",
        "particlegan/_axiom/transport.py",
        "particlegan/_axiom/optimizer.py",
        "particlegan/_axiom/loss.py",
    ):
        assert rel in audit["source_sha256"], f"missing {rel} in clockfree source_audit"

    sig = technique_signature(rec)
    assert sig["mechanisms"].get("axiom_optimizer") is True
    assert sig["mechanisms"].get("axiom_critic") is True
    assert sig["mechanisms"].get("axiom_transport") is True
    assert "t1c_transport" not in sig["mechanisms"]
    assert "t1c_chart_commitment" not in sig["mechanisms"]
    assert recipe_field_active("rhbi_fibre_rel_blend", rec) is True


def test_axiom_record_binding_zero_monkeypatch_and_zero_legacy_import() -> None:
    check_script = (
        "import torch, particlegan as pg\n"
        "import particlegan.t1_coupling as legacy_t1c\n"
        "from particlegan._axiom.state import get_tensor_record\n"
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
        "assert tr.loss.record is tr.opt_d.record\n"
        "assert tr.prior_regularizer.record is tr.opt_d.record\n"
        "assert get_tensor_record(prior.z) is tr.opt_d.record\n"
        "assert not hasattr(prior.z, '_axiom_record')\n"
        "assert not hasattr(prior.z, '_axiom_active_rows')\n"
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


def test_gan_trainer_mog_pullback_and_point_mass_smoke() -> None:
    trainer, last = _run_mog_trainer(steps=4, total_steps=100, seed=0)
    assert trainer.completed_steps == 4
    assert torch.isfinite(last["loss_d"]) and torch.isfinite(last["loss_g"])
    assert trainer.ema_D is None
    assert trainer.opt_g.record is None
    assert trainer.opt_d.record is not None
    assert trainer.opt_d.record.observed_steps == 4
    assert trainer.opt_d.record.calls == 4

    recipe = pg.get_recipe("axiom", num_particles=16, z_dim=2, batch_size=16, total_steps=50)
    g, d = _VecG(2, 2, 16), _VecD(2, 16)
    prior = pg.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
    _init_deterministic(g, d, prior, seed=1)
    pt_trainer = pg.GANTrainer(recipe, g, d, prior=prior, max_steps=50, seed=1)
    pt_batch = torch.full((16, 2), 0.75)
    for _ in range(3):
        out = pt_trainer.step(pt_batch)
        assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])


def test_zero_sigma_interp_manifold_and_exact_atomic_smoke() -> None:
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

    recipe_a = pg.get_recipe("axiom", num_particles=4, z_dim=2, batch_size=32, total_steps=100)
    g2, d2 = _VecG(2, 4, 24), _VecD(4, 24)
    prior2 = pg.ParticlePrior(num_particles=4, z_dim=2, learnable=True)
    _init_deterministic(g2, d2, prior2, seed=3)
    trainer2 = pg.GANTrainer(recipe_a, g2, d2, prior=prior2, max_steps=100, seed=3)
    atoms = torch.eye(4)
    rng2 = torch.Generator().manual_seed(33)
    for _ in range(4):
        ids = torch.randint(0, 4, (32,), generator=rng2)
        out2 = trainer2.step(atoms[ids])
        assert torch.isfinite(out2["loss_d"]) and torch.isfinite(out2["loss_g"])
    assert trainer2.opt_d.record.atomic_ledger is not None
    assert trainer2.opt_d.record.thermal_initialized is True


def test_conv2d_batchnorm_zero_state_and_direction_blend() -> None:
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

    # Verify 0.0 B/param tensor state on 4D Conv2d / ConvTranspose2d layers (no m_u momentum tensors):
    for opt in (trainer.opt_g, trainer.opt_d):
        for p, st in opt.state.items():
            if isinstance(p, torch.Tensor) and p.ndim == 4:
                tensor_bytes = sum(v.numel() * v.element_size() for v in st.values() if isinstance(v, torch.Tensor))
                assert tensor_bytes == 0, f"4D conv state must be 0 B/param, got {tensor_bytes}B: {list(st.keys())}"

    # Verify 1-loss and 2-loss direction_blend non-ascent via public API:
    rec_db = pg.get_recipe("axiom", num_particles=8, z_dim=2, batch_size=16, total_steps=50)
    gb = _VecG(2, 2, 16)
    _init_deterministic(gb, seed=8)
    opt_g = rec_db.make_generator_optimizer(gb)
    assert opt_g.direction_blend is True

    z = torch.randn((16, 2))
    out = gb(z)
    l0 = (out - 1.5).square().mean()
    l1 = (out + 1.5).square().mean()
    opt_g.zero_grad()
    constraint_geometry_backward(l0 + l1, opt_g, (l0, l1))
    opt_g.step()
    assert opt_g.constraint_geometry_stats["steps"] == 1
    assert opt_g.constraint_geometry_stats["max_derivative_after"] <= 1e-6


def test_ddgan_ae_vae_bigan_and_wordfixture_separate_optimizers_smoke() -> None:
    # 1. DDGAN self-binding + .detach() capture ring survival without _axiom_input monkey-patching:
    recipe = pg.get_recipe("axiom", model="ddgan", num_particles=16, z_dim=2, batch_size=24, total_steps=100)
    g, d = _VecG(5, 2, 24), _CondD(2, 2, 24)
    prior = pg.ParticlePrior(num_particles=16, z_dim=2, learnable=True)
    _init_deterministic(g, d, prior, seed=5)
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=deepcopy(d))
    loss = recipe.make_loss()
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)
    assert opt_d.ema_critic is None

    rng = torch.Generator().manual_seed(55)
    for _ in range(3):
        x0 = torch.randn((24, 2), generator=rng)
        xt = x0 + 0.5 * torch.randn((24, 2), generator=rng)
        t = torch.randint(1, 3, (24,), generator=rng)
        z, rows = prior.sample(24, generator=rng)
        fake = g(torch.cat((z, xt, t.float().unsqueeze(-1)), dim=-1))

        opt_d.zero_grad()
        dl = loss.d_loss(d(x0, xt=xt, t=t), d(fake.detach(), xt=xt, t=t))
        pen = penalty(d, x0, fake.detach(), xt=xt, t=t)
        (dl + pen).backward()
        opt_d.step()

        opt_g.zero_grad()
        opt_g.set_sampled_rows(prior.z, rows)
        d_real_det = d(x0, xt=xt, t=t).detach()
        assert not hasattr(d_real_det, "_axiom_input")
        gl = loss.g_loss(d(fake, xt=xt, t=t), d_real_det)
        constraint_geometry_backward(gl, opt_g, (gl,))
        opt_g.step()
        assert torch.isfinite(dl) and torch.isfinite(pen) and torch.isfinite(gl)
    assert opt_d.record.cond_initialized is True

    # 2. AE-GAN / VAE-GAN recipe.encode + standalone AxiomPriorRegularizer:
    for enc_mode in ("ae", "categorical"):
        rec_enc = pg.get_recipe(
            "axiom", encoder_mode=enc_mode, prior_kind="mog", num_particles=16, z_dim=4, batch_size=16, total_steps=50
        )
        p_enc = pg.MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05, learnable=True)
        _init_deterministic(p_enc, seed=6)
        reg = rec_enc.make_prior_regularizer(prior=p_enc, reference=p_enc)
        reg_val = reg(p_enc.z)
        assert reg_val.ndim == 0 and torch.isfinite(reg_val)
        q = torch.randn((16, 4))
        enc = rec_enc.encode(q, p_enc, offset=torch.zeros_like(q)) if enc_mode == "ae" else rec_enc.encode(q, p_enc)
        assert torch.isfinite(enc.codes).all()

    # 3. BiGAN joint_g_loss + InputNoise clean-input capture propagation:
    rec_bi = pg.get_recipe("axiom", num_particles=8, z_dim=2, batch_size=16, total_steps=50)
    gb, eb, db = _VecG(2, 4, 24), _VecG(4, 2, 24), _VecD(6, 24)
    pb = pg.ParticlePrior(num_particles=8, z_dim=2, learnable=True)
    _init_deterministic(gb, eb, db, pb, seed=7)
    opt_gb, opt_db = rec_bi.make_optimizers(gb, db, pb, encoder=eb)
    loss_bi = rec_bi.make_loss()
    pen_bi = rec_bi.make_critic_penalty(opt_db, collect_stats=True)
    noisy_d = pg.InputNoise(db, std=0.1, generator=torch.Generator().manual_seed(77))
    real_x = torch.randn((16, 4), generator=rng)
    z_b, rows_b = pb.sample(16, generator=rng)
    joint_r = torch.cat((real_x, eb(real_x).detach()), dim=-1)
    joint_f = torch.cat((gb(z_b).detach(), z_b.detach()), dim=-1)
    opt_db.zero_grad()
    (loss_bi.d_loss(noisy_d(joint_r), noisy_d(joint_f)) + pen_bi(noisy_d, joint_r, joint_f)).backward()
    opt_db.step()
    opt_gb.zero_grad()
    opt_gb.set_sampled_rows(pb.z, rows_b)
    gl_bi = loss_bi.joint_g_loss(
        noisy_d(torch.cat((gb(z_b), z_b), dim=-1)),
        noisy_d(torch.cat((real_x, eb(real_x)), dim=-1)),
    )
    constraint_geometry_backward(gl_bi, opt_gb, (gl_bi,))
    opt_gb.step()
    assert torch.isfinite(gl_bi)

    # 4. WordFixture separate optimizer path recovers prior rows via WeakTensorMap without monkey-patching:
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
    atoms = torch.eye(4)
    rx = atoms[torch.arange(32) % 4]
    z_w, rows_w = pw.sample(32, generator=rng)
    opt_gw.zero_grad()
    opt_gw.set_sampled_rows(pw.z, rows_w)
    gl_w = loss_w.joint_g_loss(dw(torch.cat((gw(z_w), z_w), dim=-1)), dw(torch.cat((rx, ew(rx)), dim=-1)))
    constraint_geometry_backward(gl_w, opt_gw, (gl_w,))
    opt_gw.step()
    assert torch.isfinite(gl_w)
    assert opt_dw.record.atomic_ledger is not None
    assert not hasattr(pw.z, "_axiom_record")
    assert get_tensor_record(pw.z) is opt_dw.record


@pytest.mark.parametrize("fmt", ("mxfp8", "nvfp4"))
def test_emulated_low_precision_mxfp8_nvfp4_smoke(fmt: str) -> None:
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
# 3. Fresh-Process Exact Continuation, BF16/Pending, Schema-1 Rejection & Clock-Free
# ---------------------------------------------------------------------------


def test_fresh_process_exact_checkpoint_continuation_and_schema1_rejection(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "v6_ckpt.pt"
    out_ref = tmp_path / "v6_ref.pt"
    out_res = tmp_path / "v6_res.pt"

    script = f"""
import sys, torch
from torch import nn
import particlegan as pg

mode = sys.argv[1]
def build():
    rec = pg.get_recipe("axiom", prior_kind="mog", sigma_rel=0.05, num_particles=32, z_dim=2, batch_size=32, total_steps=100)
    g = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 2))
    d = nn.Sequential(
        nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(),
        nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1),
    )
    prior = rec.make_prior()
    for i, m in enumerate((g, d, prior)):
        pg.init.deterministic_orthogonal_(m, seed=i * 13)
    return pg.GANTrainer(rec, g, d, prior=prior, max_steps=100, seed=0)

rng = torch.Generator().manual_seed(123)
batches = [torch.randn(32, 2, generator=rng) * (1.0 if t < 4 else 50.0) for t in range(10)]
tr = build()
if mode == "ref":
    for t in range(5):
        tr.step(batches[t])
    torch.save(tr.state_dict(), {str(ckpt_path)!r})
    for t in range(5, 10):
        tr.step(batches[t])
    torch.save({{
        "g": [p.detach().clone() for p in tr.G.parameters()],
        "d": [p.detach().clone() for p in tr.D.parameters()],
        "z": tr.prior.z.detach().clone(),
        "rec": tr.opt_d.record.state_dict(),
        "reg": tr.prior_regularizer.state_dict_checkpoint(),
    }}, {str(out_ref)!r})
else:
    tr.load_state_dict(torch.load({str(ckpt_path)!r}, weights_only=False))
    for t in range(5, 10):
        tr.step(batches[t])
    torch.save({{
        "g": [p.detach().clone() for p in tr.G.parameters()],
        "d": [p.detach().clone() for p in tr.D.parameters()],
        "z": tr.prior.z.detach().clone(),
        "rec": tr.opt_d.record.state_dict(),
        "reg": tr.prior_regularizer.state_dict_checkpoint(),
    }}, {str(out_res)!r})
"""
    for mode in ("ref", "resume"):
        proc = subprocess.run([sys.executable, "-c", script, mode], cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert proc.returncode == 0, f"{mode} subprocess failed:\n{proc.stderr}\n{proc.stdout}"
    a = torch.load(out_ref, weights_only=False)
    b = torch.load(out_res, weights_only=False)
    for pa, pb in zip(a["g"], b["g"]):
        assert torch.equal(pa, pb), "Generator weights diverged after fresh-process resume"
    for pa, pb in zip(a["d"], b["d"]):
        assert torch.equal(pa, pb), "Critic weights diverged after fresh-process resume"
    assert torch.equal(a["z"], b["z"]), "Prior table diverged after fresh-process resume"
    assert a["rec"]["schema"] == 2 and b["rec"]["schema"] == 2
    assert a["rec"]["ema_data_scale"] == b["rec"]["ema_data_scale"]

    # Fail-closed rejection of schema-1 legacy checkpoints across all schema-2 stateful objects:
    saved = torch.load(ckpt_path, weights_only=False)
    for bad_schema in (1, 999):
        bad_opt = deepcopy(saved)
        bad_opt["optimizers"][1]["axiom"]["schema"] = bad_schema
        tr_bad = _run_mog_trainer(steps=1, total_steps=100, seed=0)[0]
        with pytest.raises(ValueError):
            tr_bad.load_state_dict(bad_opt)

        bad_rec = deepcopy(a["rec"])
        bad_rec["schema"] = bad_schema
        with pytest.raises(ValueError):
            AxiomStepRecord().load_state_dict(bad_rec)

        bad_reg = deepcopy(a["reg"])
        bad_reg["schema"] = bad_schema
        with pytest.raises(ValueError):
            AxiomPriorRegularizer(max_prototypes=16).load_state_dict_checkpoint(bad_reg)

    with pytest.raises(ValueError):
        PrototypeAtlas(max_prototypes=16).load_state_dict({"schema": 1})
    with pytest.raises(ValueError):
        AxiomCondFiberBank().load_state_dict({"schema": 1})


def test_bf16_fp16_fp32_state_preservation_and_pending_direction_blend_continuation() -> None:
    # 1. Pure BF16 & FP16 reload must preserve FP32 optimizer state without downcast truncation:
    for dt in (torch.bfloat16, torch.float16):
        def _make_lowp(dtype=dt) -> nn.Sequential:
            torch.manual_seed(314)
            return nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 4)).to(dtype)

        m_ref, m_res = _make_lowp(), _make_lowp()
        o_ref = AxiomOptimizer(m_ref.parameters(), lr=0.02)
        o_res = AxiomOptimizer(m_res.parameters(), lr=0.02)
        g_rng = torch.Generator().manual_seed(77)
        xs = [torch.randn((8, 8), generator=g_rng).to(dt) for _ in range(4)]
        for t in range(2):
            o_ref.zero_grad()
            m_ref(xs[t]).float().square().mean().backward()
            o_ref.step()
            o_res.zero_grad()
            m_res(xs[t]).float().square().mean().backward()
            o_res.step()
        sd_m, sd_o = deepcopy(m_res.state_dict()), deepcopy(o_res.state_dict())
        m_res2 = _make_lowp()
        m_res2.load_state_dict(sd_m)
        o_res2 = AxiomOptimizer(m_res2.parameters(), lr=0.02)
        o_res2.load_state_dict(sd_o)
        for st in o_res2.state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    assert v.dtype == torch.float32, f"Expected FP32 optimizer state under {dt}, got {v.dtype}"
        for t in range(2, 4):
            o_ref.zero_grad()
            m_ref(xs[t]).float().square().mean().backward()
            o_ref.step()
            o_res2.zero_grad()
            m_res2(xs[t]).float().square().mean().backward()
            o_res2.step()
        for pa, pb in zip(m_ref.parameters(), m_res2.parameters()):
            assert torch.equal(pa.detach(), pb.detach()), f"Low-precision {dt} resume diverged"

    # 2. Pending direction_blend protected gradients survive checkpoint round-trip bit-for-bit:
    torch.manual_seed(21)
    g_ref = _VecG(2, 2, 16)
    g_res = deepcopy(g_ref)
    opt_ref = AxiomOptimizer(g_ref.parameters(), lr=0.02, direction_blend=True)
    opt_res = AxiomOptimizer(g_res.parameters(), lr=0.02, direction_blend=True)
    z = torch.randn(16, 2)
    for model, opt in ((g_ref, opt_ref), (g_res, opt_res)):
        out = model(z)
        l0, l1 = (out - 1.0).square().mean(), (out + 1.0).square().mean()
        opt.zero_grad()
        constraint_geometry_backward(l0 + l1, opt, (l0, l1))
    sd_pending = deepcopy(opt_res.state_dict())
    grads_saved = [p.grad.detach().clone() for p in g_res.parameters()]
    g_fresh = deepcopy(g_res)
    for p, gr in zip(g_fresh.parameters(), grads_saved):
        p.grad = gr.clone()
    opt_fresh = AxiomOptimizer(g_fresh.parameters(), lr=0.02, direction_blend=True)
    opt_fresh.load_state_dict(sd_pending)
    opt_ref.step()
    opt_fresh.step()
    for pa, pb in zip(g_ref.parameters(), g_fresh.parameters()):
        assert torch.equal(pa.detach(), pb.detach()), "Pending direction_blend resume diverged"


def test_clockfree_horizon_invariance() -> None:
    t_short, l_short = _run_mog_trainer(steps=5, total_steps=100, seed=0)
    t_long, l_long = _run_mog_trainer(steps=5, total_steps=5000, seed=0)

    assert float(l_short["loss_d"]) == float(l_long["loss_d"])
    assert float(l_short["loss_g"]) == float(l_long["loss_g"])
    assert torch.equal(t_short.prior.z, t_long.prior.z)
    for (k, v1), (_, v2) in zip(t_short.G.state_dict().items(), t_long.G.state_dict().items()):
        assert torch.equal(v1, v2), f"clock-dependent drift in G.{k}"
    for (k, v1), (_, v2) in zip(t_short.D.state_dict().items(), t_long.D.state_dict().items()):
        assert torch.equal(v1, v2), f"clock-dependent drift in D.{k}"


# ---------------------------------------------------------------------------
# 4. Host-Sync Budget Gate (grid100 <= 8 Syncs/Step)
# ---------------------------------------------------------------------------


def _run_grid100_host_sync_budget(dev: torch.device) -> None:
    from scripts.axiom_four_laws_benchmark import (
        SimpleMLPDiscriminator,
        SimpleMLPGenerator,
        _count_host_syncs,
        sample_100gaussians,
    )

    rec = pg.get_recipe("axiom", num_particles=1024, z_dim=2, prior_kind="mog", batch_size=256)
    gen = SimpleMLPGenerator(2, 128, n_hidden=3, out_dim=2).to(dev)
    crit = SimpleMLPDiscriminator(2, 128, n_hidden=3).to(dev)
    prior = pg.MoGParticlePrior(num_particles=1024, z_dim=2, sigma=0.025, device=dev)
    for mod in (gen, crit, prior):
        pg.init.deterministic_orthogonal_(mod, seed=0)
    trainer = pg.GANTrainer(rec, gen, crit, prior=prior, seed=0)
    rng = torch.Generator(device=dev).manual_seed(0)
    batches = [sample_100gaussians(256, dev, generator=rng) for _ in range(25)]
    for batch in batches[:10]:
        trainer.step(batch)
    with _count_host_syncs(dev) as sc:
        for batch in batches[10:25]:
            trainer.step(batch)
    cpu_only_guard = 2.0 if dev.type == "cpu" else 0.0
    syncs_per_step = (sum(sc.values()) / 15.0) - cpu_only_guard
    assert syncs_per_step <= 8.0, f"grid100 syncs/step={syncs_per_step:.2f} exceeded <= 8.0 budget ({sc})"


def test_grid100_host_syncs_at_most_8_per_step() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _run_grid100_host_sync_budget(dev)


@pytest.mark.cuda
def test_grid100_host_syncs_at_most_8_per_step_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _run_grid100_host_sync_budget(torch.device("cuda:0"))
