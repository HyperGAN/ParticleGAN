"""Unit, semantic, tamper, 0-forbidden-op, LOC/self-containment, clockfree, and fresh-process continuation tests for minimal."""
from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

import particlegan as pg
from particlegan._minimal.dynamics import MinimalGeneratorOptimizer

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_FILES = (
    ROOT / "particlegan/minimal.py",
    ROOT / "particlegan/_minimal/__init__.py",
    ROOT / "particlegan/_minimal/dynamics.py",
    ROOT / "particlegan/_minimal/geometry.py",
    ROOT / "particlegan/_minimal/prior.py",
)
FORBIDDEN_SIBLINGS = {
    "_unified",
    "_unified_v4",
    "_nash_v3",
    "_cond_fiber",
    "_t4_arch",
    "_t4_nash",
    "_t2_critic",
    "_t1_coupling",
    "_empf",
    "_rhbi",
    "_rgdt",
    "_bispf",
}


def _forbid(*_args, **_kwargs):
    raise AssertionError("Forbidden autograd.grad / SVD / eigh / pinv primitive invoked!")


def _patch_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.autograd, "grad", _forbid)
    monkeypatch.setattr(torch.linalg, "svd", _forbid)
    monkeypatch.setattr(torch.linalg, "svdvals", _forbid)
    monkeypatch.setattr(torch.linalg, "eigh", _forbid)
    monkeypatch.setattr(torch.linalg, "eigvalsh", _forbid)
    monkeypatch.setattr(torch.linalg, "pinv", _forbid)
    monkeypatch.setattr(torch, "svd", _forbid)


def _make_trainer(dim: int, zdim: int = 2, *, is_img: bool = False, seed: int = 0) -> pg.GANTrainer:
    torch.manual_seed(seed)
    if is_img:
        gen = nn.Sequential(
            nn.Unflatten(1, (zdim, 1, 1)),
            nn.ConvTranspose2d(zdim, 8, 4),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Tanh(),
        )
        disc = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8 * 16, 1),
        )
        rec = pg.get_recipe("minimal", z_dim=zdim, num_particles=16)
        return pg.GANTrainer(rec, gen, disc, seed=seed)
    gen = nn.Sequential(nn.Linear(zdim, 32), nn.SiLU(), nn.Linear(32, dim))
    disc = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))
    rec = pg.get_recipe("minimal", z_dim=zdim, num_particles=16)
    return pg.GANTrainer(rec, gen, disc, seed=seed)


def test_minimal_loc_budget_and_zero_sibling_imports():
    total_lines = 0
    for path in MINIMAL_FILES:
        text = path.read_text(encoding="utf-8")
        total_lines += len(text.splitlines())
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").lstrip(".")
                top = mod.split(".")[0] if mod else ""
                assert top not in FORBIDDEN_SIBLINGS, f"{path.name} imports sibling {mod!r}"
                for alias in node.names:
                    assert alias.name not in FORBIDDEN_SIBLINGS, f"{path.name} imports sibling {alias.name!r}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = set(alias.name.split("."))
                    assert not (parts & FORBIDDEN_SIBLINGS), f"{path.name} imports sibling {alias.name!r}"
    assert total_lines < 900, f"Minimal core must stay < 900 LOC, got {total_lines}"


def test_minimal_recipe_contract_and_import_order_in_fresh_process():
    assert pg.DEFAULT_RECIPE == "unified_v11"
    assert len(dataclasses.fields(pg.Recipe)) == 179
    rec = pg.get_recipe("minimal")
    assert rec.critic_formulation == "minimal"
    assert rec.prior_regularizer == "minimal"
    assert rec.loss == "relativistic"

    recipes_path = ROOT / "particlegan/recipes.py"
    code = (
        "import importlib.util, sys\n"
        "class BlockTorch:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "            raise AssertionError(f'Imported Torch: {fullname}')\n"
        "sys.meta_path.insert(0, BlockTorch())\n"
        f"spec = importlib.util.spec_from_file_location('standalone_recipes', {str(recipes_path)!r})\n"
        "r = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = r\n"
        "spec.loader.exec_module(r)\n"
        "assert 'torch' not in sys.modules\n"
        "assert r.DEFAULT_RECIPE == 'unified_v11'\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr

    # Reverse import order: particlegan.minimal before particlegan must preserve DEFAULT_RECIPE
    rev = subprocess.run(
        [
            sys.executable,
            "-c",
            "import particlegan.minimal as m, particlegan as pg; "
            "assert pg.DEFAULT_RECIPE == 'unified_v11'; "
            "assert hasattr(m, 'MinimalCriticPenalty')",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rev.returncode == 0, rev.stderr


def test_minimal_zero_forbidden_ops_across_all_five_regimes(monkeypatch: pytest.MonkeyPatch):
    _patch_forbidden(monkeypatch)
    rng = torch.Generator().manual_seed(0)

    # 1. 1D continuous
    t1 = _make_trainer(1, 2, seed=0)
    for _ in range(4):
        t1.step(torch.randn(32, 1, generator=rng))

    # 2. 1D point-mass (sigma == 0)
    tpm = _make_trainer(1, 2, seed=0)
    for _ in range(4):
        tpm.step(torch.full((32, 1), 0.7))
    assert all(torch.isfinite(p).all() for p in tpm.G.parameters())

    # 3. 2D multimodal (pullback isometry + Sinkhorn + Burg + coalescing + LTV) + MoG prior step tracking
    t2 = _make_trainer(2, 2, seed=0)
    for _ in range(4):
        t2.step(torch.randn(32, 2, generator=rng))

    torch.manual_seed(0)
    gm = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 2))
    dm = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
    pmog = pg.MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.10)
    tmog = pg.GANTrainer(
        pg.get_recipe("minimal", z_dim=4, num_particles=32, prior_kind="mog", sigma_rel=0.10),
        gm,
        dm,
        prior=pmog,
        seed=0,
    )
    for _ in range(3):
        tmog.step(torch.randn(32, 2, generator=rng))
    pz = pg.unwrap_prior(tmog.prior).z
    assert pz in tmog.opt_g.state and int(tmog.opt_g.state[pz]["step"].item()) == 3

    # 4. 4D image conv (_foreach fast path + live-endpoint secant reuse)
    t4 = _make_trainer(3, 4, is_img=True, seed=0)
    for _ in range(4):
        t4.step(torch.randn(8, 3, 4, 4, generator=rng))
    assert all(torch.isfinite(p).all() for p in t4.G.parameters())

    # 5. Conditional fiber crossing ring_c >= 24 activation boundary with zero eigh
    rec = pg.get_recipe("minimal")
    torch.manual_seed(0)
    gen = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 2))

    class _CondD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 1))

        def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([x, c], -1))

    class _Wrap(nn.Module):
        def __init__(self, d: nn.Module, c: torch.Tensor) -> None:
            super().__init__()
            self.d = d
            self.c = c

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.d(x, self.c)

    disc = _CondD()
    opt_g = rec.make_generator_optimizer(gen)
    opt_d = rec.make_critic_optimizer(disc)
    pen = rec.make_critic_penalty(opt_d)
    loss = rec.make_loss()
    for _ in range(5):
        x = torch.randn(16, 2, generator=rng)
        c = torch.randn(16, 3, generator=rng)
        z = torch.randn(16, 2, generator=rng)
        xf = gen(torch.cat([z, c], -1)).detach()
        dr, df = disc(x, c), disc(xf, c)
        ld = loss.d_loss(dr, df) + pen(_Wrap(disc, c), x, xf)
        opt_d.zero_grad()
        ld.backward()
        opt_d.step()
        xf2 = gen(torch.cat([z, c], -1))
        dr2, df2 = disc(x, c), disc(xf2, c)
        lg = loss.g_loss(df2, dr2)
        opt_g.zero_grad()
        lg.backward()
        opt_g.step()
    assert len(pen.record.cond_bank["strata"]) == 1
    assert pen.record.cond_bank["strata"][0]["ring_c"].shape == (80, 3)


def test_minimal_fresh_process_bit_exact_continuation(tmp_path: Path):
    ckpt_path = tmp_path / "minimal_ckpt.pt"
    ref_path = tmp_path / "minimal_ref.pt"
    script_part1 = f"""
import torch, torch.nn as nn, particlegan as pg
def make_tr(seed):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
    D = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
    return pg.GANTrainer(pg.get_recipe('minimal', z_dim=2, num_particles=16), G, D, seed=seed)
rng = torch.Generator().manual_seed(42)
batches = [torch.randn(32, 2, generator=rng) for _ in range(12)]
t1, t2 = make_tr(0), make_tr(0)
for b in batches[:6]:
    t1.step(b); t2.step(b)
torch.save({{'ckpt': t2.state_dict(), 'batches': batches[6:]}}, {str(ckpt_path)!r})
for b in batches[6:]:
    t1.step(b)
torch.save({{'G': t1.G.state_dict(), 'D': t1.D.state_dict()}}, {str(ref_path)!r})
"""
    script_part2 = f"""
import torch, torch.nn as nn, particlegan as pg
torch.manual_seed(999)
G = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
D = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
t3 = pg.GANTrainer(pg.get_recipe('minimal', z_dim=2, num_particles=16), G, D, seed=999)
blob = torch.load({str(ckpt_path)!r}, weights_only=False)
t3.load_state_dict(blob['ckpt'])
for b in blob['batches']:
    t3.step(b)
ref = torch.load({str(ref_path)!r}, weights_only=False)
for k, v in ref['G'].items():
    assert torch.equal(v, t3.G.state_dict()[k]), f'G mismatch {{k}}'
for k, v in ref['D'].items():
    assert torch.equal(v, t3.D.state_dict()[k]), f'D mismatch {{k}}'
"""
    r1 = subprocess.run([sys.executable, "-c", script_part1], cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run([sys.executable, "-c", script_part2], cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert r2.returncode == 0, r2.stderr


def test_minimal_cond_bank_continuation_across_activation_boundary(monkeypatch: pytest.MonkeyPatch):
    _patch_forbidden(monkeypatch)
    rec = pg.get_recipe("minimal")

    class _CondD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 1))

        def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([x, c], -1))

    class _Wrap(nn.Module):
        def __init__(self, d: nn.Module, c: torch.Tensor) -> None:
            super().__init__()
            self.d = d
            self.c = c

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.d(x, self.c)

    def make_sys(seed: int):
        torch.manual_seed(seed)
        gen = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 2))
        disc = _CondD()
        og = rec.make_generator_optimizer(gen)
        od = rec.make_critic_optimizer(disc)
        pen = rec.make_critic_penalty(od)
        return gen, disc, pen, rec.make_loss(), og, od

    def run_step(s, x, c, z):
        gen, disc, pen, loss, og, od = s
        xf = gen(torch.cat([z, c], -1)).detach()
        dr, df = disc(x, c), disc(xf, c)
        ld = loss.d_loss(dr, df) + pen(_Wrap(disc, c), x, xf)
        od.zero_grad()
        ld.backward()
        od.step()
        xf2 = gen(torch.cat([z, c], -1))
        dr2, df2 = disc(x, c), disc(xf2, c)
        lg = loss.g_loss(df2, dr2)
        og.zero_grad()
        lg.backward()
        og.step()

    s1, s2 = make_sys(0), make_sys(0)
    rng = torch.Generator().manual_seed(77)
    batches = [(torch.randn(16, 2, generator=rng), torch.randn(16, 3, generator=rng), torch.randn(16, 2, generator=rng)) for _ in range(8)]
    for x, c, z in batches[:4]:
        run_step(s1, x, c, z)
        run_step(s2, x, c, z)
    ckpt = {
        "G": deepcopy(s2[0].state_dict()),
        "D": deepcopy(s2[1].state_dict()),
        "og": deepcopy(s2[4].state_dict()),
        "od": deepcopy(s2[5].state_dict()),
    }
    s3 = make_sys(55)
    s3[0].load_state_dict(ckpt["G"])
    s3[1].load_state_dict(ckpt["D"])
    s3[4].load_state_dict(ckpt["og"])
    s3[5].load_state_dict(ckpt["od"])
    for x, c, z in batches[4:]:
        run_step(s1, x, c, z)
        run_step(s3, x, c, z)
    for p1, p3 in zip(s1[0].parameters(), s3[0].parameters()):
        assert torch.equal(p1, p3)
    for p1, p3 in zip(s1[1].parameters(), s3[1].parameters()):
        assert torch.equal(p1, p3)


def test_minimal_checkpoint_tamper_rejection():
    t = _make_trainer(2, 2, seed=0)
    rng = torch.Generator().manual_seed(0)
    for _ in range(3):
        t.step(torch.randn(32, 2, generator=rng))
    sd = t.state_dict()

    # 1. Corrupt MinimalStepRecord minimal_state schema version
    bad_schema = deepcopy(sd)
    bad_schema["optimizers"][1]["regularizer"]["record"]["minimal_state"]["schema"] = 999
    with pytest.raises(ValueError, match="invalid or missing minimal_state packet"):
        _make_trainer(2, 2, seed=1).opt_d.record.load_state_dict(bad_schema["optimizers"][1]["regularizer"]["record"])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        _make_trainer(2, 2, seed=1).load_state_dict(bad_schema)

    # 2. Inject NaN scalar into MinimalStepRecord minimal_state
    bad_nan = deepcopy(sd)
    bad_nan["optimizers"][1]["regularizer"]["record"]["minimal_state"]["ema_residual"] = float("nan")
    with pytest.raises(ValueError, match="invalid minimal_state scalar"):
        _make_trainer(2, 2, seed=1).opt_d.record.load_state_dict(bad_nan["optimizers"][1]["regularizer"]["record"])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        _make_trainer(2, 2, seed=1).load_state_dict(bad_nan)

    # 3. Corrupt cond_bank ring shape mismatch (valid tuple key so ring-length validation triggers)
    bad_bank = deepcopy(sd)
    bad_bank["optimizers"][1]["regularizer"]["record"]["minimal_state"]["cond_bank"] = {
        "d_c": 3,
        "d_x": 2,
        "strata": [
            {
                "key": (0,),
                "ring_c": torch.zeros(10, 3),
                "ring_x": torch.zeros(8, 2),
                "ring_y": torch.zeros(10, 2),
            }
        ],
    }
    with pytest.raises(ValueError, match="invalid minimal cond_bank stratum"):
        _make_trainer(2, 2, seed=1).opt_d.record.load_state_dict(bad_bank["optimizers"][1]["regularizer"]["record"])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        _make_trainer(2, 2, seed=1).load_state_dict(bad_bank)

    # 4. Corrupt MinimalOptimizer peak_rms to negative scalar
    bad_peak = deepcopy(sd)
    first_pid = next(iter(bad_peak["optimizers"][0]["state"]))
    bad_peak["optimizers"][0]["state"][first_pid]["peak_rms"] = torch.tensor(-1.0)
    with pytest.raises(ValueError, match="invalid minimal optimizer state"):
        _make_trainer(2, 2, seed=1).opt_g.load_state_dict(bad_peak["optimizers"][0])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        _make_trainer(2, 2, seed=1).load_state_dict(bad_peak)

    # 5. Corrupt MinimalPriorRegularizer state
    bad_prior = deepcopy(sd)
    bad_prior["prior_regularizer"]["schema"] = 99
    with pytest.raises(ValueError, match="invalid MinimalPriorRegularizer state"):
        _make_trainer(2, 2, seed=1).prior_regularizer.load_state_dict(bad_prior["prior_regularizer"])
    with pytest.raises(ValueError):
        _make_trainer(2, 2, seed=1).load_state_dict(bad_prior)


def test_minimal_1e37_underflow_safety_and_clockfree():
    p = nn.Parameter(torch.zeros(4, 4))
    opt = MinimalGeneratorOptimizer([p], lr=1e-3)
    p.grad = torch.full((4, 4), 1e-37)
    opt.step()
    assert torch.isfinite(p).all() and p.abs().max().item() > 0.0

    # Clockfree invariance: changing total_steps must not change the trajectory
    rng = torch.Generator().manual_seed(11)
    batches = [torch.randn(32, 2, generator=rng) for _ in range(5)]
    torch.manual_seed(0)
    g1, d1 = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2)), nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    t1 = pg.GANTrainer(pg.get_recipe("minimal", z_dim=2, num_particles=16, total_steps=100), g1, d1, seed=0)
    torch.manual_seed(0)
    g2, d2 = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2)), nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    t2 = pg.GANTrainer(pg.get_recipe("minimal", z_dim=2, num_particles=16, total_steps=99999), g2, d2, seed=0)
    for b in batches:
        t1.step(b)
        t2.step(b)
    for p1, p2 in zip(t1.G.parameters(), t2.G.parameters()):
        assert torch.equal(p1, p2)


def test_minimal_deliberate_damage_semantic_control():
    t = _make_trainer(2, 2, seed=0)
    rng = torch.Generator().manual_seed(3)
    centers = torch.tensor([[1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]])
    for _ in range(40):
        idx = torch.randint(0, 4, (64,), generator=rng)
        real = centers[idx] + 0.1 * torch.randn(64, 2, generator=rng)
        t.step(real)

    def _mean_dist_to_centers(trainer: pg.GANTrainer) -> float:
        with torch.no_grad():
            z = torch.randn(256, 2, generator=torch.Generator().manual_seed(9))
            fake = trainer.G(z)
            return float(torch.cdist(fake, centers).min(dim=1).values.mean().item())

    before = _mean_dist_to_centers(t)
    with torch.no_grad():
        for p in t.G.parameters():
            p.add_(10.0)
    after = _mean_dist_to_centers(t)
    assert after > before + 5.0, f"Deliberate G damage must degrade metric ({before:.3f} -> {after:.3f})"
