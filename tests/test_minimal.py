"""Unit, semantic, tamper, 0-forbidden-op, LOC/self-containment, clockfree, and fresh-process continuation tests for minimal."""
from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import io
import math
from pathlib import Path
import subprocess
import sys
import tokenize

import pytest
import torch
from torch import nn

import particlegan as pg
from particlegan._minimal.dynamics import (
    MinimalConstraintGeneratorOptimizer,
    MinimalGeneratorOptimizer,
)
from particlegan._minimal.geometry import (
    MinimalStepRecord,
    _cond_fiber_loss,
)
from particlegan._minimal.prior import MinimalPriorRegularizer

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


def test_minimal_hygiene_and_zero_sibling_imports():
    for path in MINIMAL_FILES:
        text = path.read_text(encoding="utf-8")
        # Forbid semicolon-packed lines (un-minified PEP-8 Python)
        semis = [
            tok.start
            for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type == tokenize.OP and tok.string == ";"
        ]
        assert not semis, f"{path.name} contains semicolon-packed statements at {semis}"

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
            elif isinstance(node, ast.ListComp):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr == "__setitem__":
                        raise AssertionError(f"{path.name} contains side-effect __setitem__ list comprehension")


def test_minimal_recipe_contract_and_import_order_in_fresh_process():
    assert pg.DEFAULT_RECIPE == "axiom"
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
        "assert r.DEFAULT_RECIPE == 'axiom'\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr

    # Reverse import order: particlegan.minimal before particlegan must preserve DEFAULT_RECIPE
    rev = subprocess.run(
        [
            sys.executable,
            "-c",
            "import particlegan.minimal as m, particlegan as pg; "
            "assert pg.DEFAULT_RECIPE == 'axiom'; "
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


def test_minimal_zero_pinv_constraint_projection_and_no_ext_grad_pollution(
    monkeypatch: pytest.MonkeyPatch,
):
    """MinimalConstraintGeneratorOptimizer must project 1- and 2-loss conflicts with 0 pinv/autograd.grad and 0 external-param pollution."""
    _patch_forbidden(monkeypatch)
    rng = torch.Generator().manual_seed(101)

    # 1. External parameter must never be polluted by bind_protected_losses
    p_owned = nn.Parameter(torch.randn((4, 4), generator=rng))
    p_ext = nn.Parameter(torch.randn((4, 4), generator=rng))
    opt = MinimalConstraintGeneratorOptimizer([p_owned], lr=1e-2)
    loss_shared = (p_owned * 2.0).sum() + (p_ext * 3.0).sum()
    opt.bind_protected_losses((loss_shared,))
    assert p_ext.grad is None, "bind_protected_losses polluted external parameter p_ext.grad!"
    opt._protected = None

    # 2. Single conflicting protected loss (non-ascent projection)
    g1 = torch.randn((4, 4), generator=rng)
    n1 = -g1 + 0.3 * torch.randn((4, 4), generator=rng)
    p0 = p_owned.detach().clone()
    opt.zero_grad(set_to_none=True)
    l_main = (p_owned * g1).sum()
    l_prot = (p_owned * n1).sum()
    opt.bind_protected_losses((l_prot,))
    l_main.backward()
    opt.step()
    disp1 = p_owned.detach() - p0
    assert opt.constraint_geometry_stats["projected_steps"] >= 1
    assert float((n1 * disp1).sum().item()) <= 1e-6, "1-constraint projection ascended protected loss"

    # 3. Two conflicting protected losses (zero-pinv 2-constraint analytic projection)
    p2 = nn.Parameter(torch.randn((8,), generator=rng))
    opt2 = MinimalConstraintGeneratorOptimizer([p2], lr=1e-2)
    v_main = torch.tensor([1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_c1 = torch.tensor([-1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_c2 = torch.tensor([0.2, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p2_0 = p2.detach().clone()
    opt2.zero_grad(set_to_none=True)
    lc1 = (p2 * v_c1).sum()
    lc2 = (p2 * v_c2).sum()
    lm = (p2 * v_main).sum()
    opt2.bind_protected_losses((lc1, lc2))
    lm.backward()
    opt2.step()
    disp2 = p2.detach() - p2_0
    assert opt2.constraint_geometry_stats["projected_steps"] >= 1
    assert float((v_c1 * disp2).sum().item()) <= 1e-6, "2-constraint projection violated constraint 1"
    assert float((v_c2 * disp2).sum().item()) <= 1e-6, "2-constraint projection violated constraint 2"


def test_minimal_bn_vs_ln_penalty_parity_at_dim_gt_1024(monkeypatch: pytest.MonkeyPatch):
    """At dim > 1024 (3x32x32), active BN must never interleave real+fake or suffer 2.30Mx penalty explosion vs LN."""
    _patch_forbidden(monkeypatch)
    torch.manual_seed(202)
    conv_w = torch.randn(8, 3, 4, 4) * 0.15
    lin_w = torch.randn(1, 8) * 0.3

    def _make_critic(use_bn: bool) -> nn.Sequential:
        conv = nn.Conv2d(3, 8, 4, stride=2, padding=1, bias=False)
        norm: nn.Module = nn.BatchNorm2d(8, affine=True) if use_bn else nn.GroupNorm(1, 8, affine=True)
        head = nn.Linear(8, 1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(conv_w)
            head.weight.copy_(lin_w)
        return nn.Sequential(conv, norm, nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), head)

    d_bn = _make_critic(use_bn=True)
    d_ln = _make_critic(use_bn=False)
    d_bn.train()
    d_ln.train()

    rng = torch.Generator().manual_seed(203)
    xr = torch.randn((8, 3, 32, 32), generator=rng)
    # Prime BN running stats with real batch so eval variance ~ 1.0 matching LN scale
    _ = d_bn(xr)
    bn_layer = d_bn[1]
    assert isinstance(bn_layer, nn.BatchNorm2d)
    tracked_before = int(bn_layer.num_batches_tracked.item())
    rm_before = bn_layer.running_mean.detach().clone()

    # Large distribution shift (+5.0) between real and fake triggers the old 2.30Mx mixed-batch BN bug!
    xf = torch.randn((8, 3, 32, 32), generator=rng) + 5.0
    rec = pg.get_recipe("minimal")
    od_bn = rec.make_critic_optimizer(d_bn)
    od_ln = rec.make_critic_optimizer(d_ln)
    pen_bn = rec.make_critic_penalty(od_bn)
    pen_ln = rec.make_critic_penalty(od_ln)

    val_bn = float(pen_bn(d_bn, xr, xf).detach().item())
    val_ln = float(pen_ln(d_ln, xr, xf).detach().item())
    assert math.isfinite(val_bn) and math.isfinite(val_ln)
    assert bn_layer.training is True, "MinimalCriticPenalty failed to restore BN training flag"
    assert int(bn_layer.num_batches_tracked.item()) == tracked_before, "Penalty mutated BN num_batches_tracked"
    assert torch.equal(bn_layer.running_mean, rm_before), "Penalty mutated BN running_mean"
    ratio = val_bn / max(val_ln, 1e-12)
    assert 0.05 <= ratio <= 20.0, f"BN vs LN penalty parity violated at dim=3072: ratio={ratio:.2e} ({val_bn} vs {val_ln})"


def test_minimal_cond_fiber_dx3_affine_invariant_whitening(monkeypatch: pytest.MonkeyPatch):
    """_cond_fiber_loss must use L^-T whitening so dx=3 is affine-invariant (ratio ~ 1.000)."""
    _patch_forbidden(monkeypatch)
    rng = torch.Generator().manual_seed(303)
    dx, dc, n_hist, n_batch = 3, 2, 48, 16

    c_hist = torch.randn((n_hist, dc), generator=rng)
    xr_hist = torch.randn((n_hist, dx), generator=rng) * 0.8 + c_hist @ torch.tensor([[0.6, -0.3, 0.2], [0.1, 0.7, -0.5]])
    xf_hist = torch.randn((n_hist, dx), generator=rng) * 0.4 + 0.3

    c_cur = torch.randn((n_batch, dc), generator=rng)
    xr_cur = torch.randn((n_batch, dx), generator=rng) * 0.8 + c_cur @ torch.tensor([[0.6, -0.3, 0.2], [0.1, 0.7, -0.5]])
    xf_cur = torch.randn((n_batch, dx), generator=rng) * 0.4 + 0.3

    # Non-orthogonal anisotropic invertible 3x3 affine map A + shift b
    a_mat = torch.tensor([[5.0, 1.2, -0.4], [0.3, 0.4, 0.7], [-0.2, 0.1, 2.5]], dtype=torch.float32)
    b_vec = torch.tensor([3.5, -2.0, 1.25], dtype=torch.float32)

    def _eval(a: torch.Tensor, b: torch.Tensor) -> tuple[float, torch.Tensor]:
        rec = MinimalStepRecord()
        rec.cond_bank = {
            "d_c": dc,
            "d_x": dx,
            "strata": [
                {
                    "key": (0,),
                    "ring_c": c_hist.double(),
                    "ring_x": (xf_hist @ a.T + b).double(),
                    "ring_y": (xr_hist @ a.T + b).double(),
                }
            ],
        }
        xf_in = (xf_cur @ a.T + b).detach().requires_grad_(True)
        xr_in = (xr_cur @ a.T + b).detach()
        loss = _cond_fiber_loss(xf_in, xr_in, (c_cur,), rec)
        loss.backward()
        assert xf_in.grad is not None
        return float(loss.detach().item()), xf_in.grad.detach()

    l_base, g_base = _eval(torch.eye(dx), torch.zeros(dx))
    l_aff, g_aff = _eval(a_mat, b_vec)
    assert abs(l_base) > 1e-4
    ratio = l_aff / l_base
    assert abs(ratio - 1.0) < 1e-2, f"dx=3 affine invariance broken: ratio={ratio:.6f} ({l_aff} vs {l_base})"
    # Chain rule: d L(x A^T + b) / dx = g_aff @ A must equal g_base!
    assert torch.allclose(g_aff @ a_mat, g_base, atol=1e-3, rtol=2e-2)


def test_minimal_bf16_checkpoint_roundtrip_preserves_fp32_state():
    """BF16 load_state_dict on Minimal optimizers & prior regularizer must keep FP32 moments/peak_rms/row_n and bit-exact resume."""
    rng = torch.Generator().manual_seed(404)

    def _make_bf16_trainer():
        torch.manual_seed(405)
        g_mod = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2)).to(torch.bfloat16)
        d_mod = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1)).to(torch.bfloat16)
        prior = pg.MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05).to(torch.bfloat16)
        rec = pg.get_recipe("minimal", z_dim=4, num_particles=16, prior_kind="mog", sigma_rel=0.05)
        return pg.GANTrainer(rec, g_mod, d_mod, prior=prior, seed=0)

    tr1 = _make_bf16_trainer()
    tr2 = _make_bf16_trainer()
    centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]], dtype=torch.bfloat16)
    batches = [
        centers[torch.randint(0, 4, (16,), generator=rng)] + (0.05 * torch.randn((16, 2), generator=rng)).to(torch.bfloat16)
        for _ in range(8)
    ]

    for yb in batches[:4]:
        tr1.step(yb)
        tr2.step(yb)

    preg1 = tr1.prior_regularizer
    preg2 = tr2.prior_regularizer
    assert isinstance(preg1, MinimalPriorRegularizer) and isinstance(preg2, MinimalPriorRegularizer)
    assert preg2.row_n is not None and preg2.row_n.dtype == torch.float32
    assert preg2.row_m is not None and preg2.row_m.dtype == torch.float32
    assert preg2.row_v is not None and preg2.row_v.dtype == torch.float32

    # Advance row_n to 256.0 on both trainers to verify FP32 survival past BF16 256 saturation limit
    preg1.row_n.fill_(256.0)
    preg2.row_n.fill_(256.0)

    ckpt = deepcopy(tr2.state_dict())
    tr3 = _make_bf16_trainer()
    tr3.load_state_dict(ckpt)

    fp32_keys = {"exp_avg", "exp_avg_sq", "g_scale", "peak_rms", "step"}
    for opt_c in (tr3.opt_g, tr3.opt_d):
        for p, st in opt_c.state.items():
            assert p.dtype == torch.bfloat16
            for k in fp32_keys:
                if k in st and isinstance(st[k], torch.Tensor):
                    assert st[k].dtype == torch.float32, f"BF16 load_state_dict downcast {k} to {st[k].dtype}"

    preg3 = tr3.prior_regularizer
    assert isinstance(preg3, MinimalPriorRegularizer)
    assert preg3.row_n is not None and preg3.row_n.dtype == torch.float32
    assert preg3.row_m is not None and preg3.row_m.dtype == torch.float32
    assert preg3.row_v is not None and preg3.row_v.dtype == torch.float32

    for yb in batches[4:]:
        tr1.step(yb)
        tr3.step(yb)

    assert float(preg3.row_n.max().item()) >= 257.0, "BF16 MinimalPriorRegularizer row_n saturated at 256"
    for p1, p3 in zip(tr1.G.parameters(), tr3.G.parameters()):
        assert torch.equal(p1, p3)
    for p1, p3 in zip(tr1.D.parameters(), tr3.D.parameters()):
        assert torch.equal(p1, p3)
    assert torch.equal(pg.unwrap_prior(tr1.prior).z, pg.unwrap_prior(tr3.prior).z)

    # Also verify MinimalConstraintGeneratorOptimizer preserves FP32 state and restores pending _protected on device
    p_bf = nn.Parameter(torch.randn((4, 4), generator=rng).to(torch.bfloat16))
    opt_c1 = MinimalConstraintGeneratorOptimizer([p_bf], lr=1e-3)
    ((p_bf * 0.5).sum()).backward()
    opt_c1.step()
    opt_c1.bind_protected_losses(((p_bf * 0.25).sum(),))
    assert opt_c1._protected is not None
    sd_c = deepcopy(opt_c1.state_dict())

    p_bf2 = nn.Parameter(p_bf.detach().clone())
    opt_c2 = MinimalConstraintGeneratorOptimizer([p_bf2], lr=1e-3)
    opt_c2.load_state_dict(sd_c)
    assert opt_c2._protected is not None and opt_c2._protected.device == p_bf2.device
    for k in fp32_keys:
        if k in opt_c2.state[p_bf2] and isinstance(opt_c2.state[p_bf2][k], torch.Tensor):
            assert opt_c2.state[p_bf2][k].dtype == torch.float32


def test_minimal_numerical_bug_regressions():
    """Verify fast_p per-param Adam bias correction, channel-agnostic _batch_res, O(1) dim=256/257 g_loss, and Laguerre centering."""
    rng = torch.Generator().manual_seed(505)

    # 1. fast_p per-param bias correction: staggered step counts (1 vs 50) must match independent optimizers
    p_a1, p_b1 = nn.Parameter(torch.zeros(4, 4)), nn.Parameter(torch.zeros(4, 4))
    p_a2, p_b2 = nn.Parameter(torch.zeros(4, 4)), nn.Parameter(torch.zeros(4, 4))
    opt_pair = MinimalGeneratorOptimizer([p_a1, p_b1], lr=1e-2)
    opt_a = MinimalGeneratorOptimizer([p_a2], lr=1e-2)
    opt_b = MinimalGeneratorOptimizer([p_b2], lr=1e-2)
    g0 = torch.randn((4, 4), generator=rng)
    # Step only p_a 50 times, leave p_b unstepped
    for _ in range(50):
        p_a1.grad = g0.clone()
        p_b1.grad = None
        opt_pair.step()
        p_a2.grad = g0.clone()
        opt_a.step()
    # Now step BOTH p_a (step 51) and p_b (step 1) together
    p_a1.grad, p_b1.grad = g0.clone(), g0.clone()
    p_a2.grad, p_b2.grad = g0.clone(), g0.clone()
    opt_pair.step()
    opt_a.step()
    opt_b.step()
    assert torch.allclose(p_a1, p_a2, atol=1e-6, rtol=1e-5)
    assert torch.allclose(p_b1, p_b2, atol=1e-6, rtol=1e-5), "fast_p[0] step leaked into p_b bias correction!"

    # 2. Channel-count-agnostic _batch_res on dim > 1024 (C=1 vs C=3 vs C=4 replicated channels)
    rec_m = pg.get_recipe("minimal")
    dummy_d = nn.Linear(4, 1)
    pen_m = rec_m.make_critic_penalty(rec_m.make_critic_optimizer(dummy_d))
    base_1ch = torch.randn((8, 1, 36, 36), generator=rng)
    fake_1ch = torch.randn((8, 1, 36, 36), generator=rng)
    res1, sc1 = pen_m._batch_res(base_1ch, fake_1ch)
    res3, sc3 = pen_m._batch_res(base_1ch.repeat(1, 3, 1, 1), fake_1ch.repeat(1, 3, 1, 1))
    res4, sc4 = pen_m._batch_res(base_1ch.repeat(1, 4, 1, 1), fake_1ch.repeat(1, 4, 1, 1))
    assert abs(res3 - res1) < 1e-4 and abs(res4 - res1) < 1e-4, f"rel_res depended on C: {res1}, {res3}, {res4}"
    assert abs(sc3 / math.sqrt(3.0) - sc1) < 1e-3, f"data_scale C=3 mismatch: {sc3} vs {sc1 * math.sqrt(3.0)}"
    assert abs(sc4 / math.sqrt(4.0) - sc1) < 1e-3, f"data_scale C=4 mismatch: {sc4} vs {sc1 * math.sqrt(4.0)}"

    # 3. O(1) MinimalLoss.g_loss in batch size B and across dim=256 vs dim=257 (no O(B) jump!)
    def _eval_g_loss_dim(dim: int, batch_size: int, *, in_probe: bool = False) -> float:
        d_m = nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            d_m.weight.zero_()
            d_m.weight[0, 0] = 0.25
        od_m = rec_m.make_critic_optimizer(d_m)
        rec_m.make_critic_penalty(od_m)
        od_m.record._in_protected_probe = in_probe
        loss_m = rec_m.make_loss()
        xr_d = torch.randn((batch_size, dim), generator=torch.Generator().manual_seed(600))
        xf_d = torch.randn((batch_size, dim), generator=torch.Generator().manual_seed(601), requires_grad=True)
        dr_d = d_m(xr_d)
        df_d = d_m(xf_d)
        return float(loss_m.g_loss(df_d, dr_d).detach().item())

    # Batch-size O(1) check at dim=256 (B=16 vs B=64 must stay ~1x, NOT 4x!)
    l256_b16 = _eval_g_loss_dim(256, 16)
    l256_b64 = _eval_g_loss_dim(256, 64)
    assert 0.5 < l256_b64 / max(l256_b16, 1e-8) < 2.0, f"dim=256 g_loss scaled with B: B=16 {l256_b16:.4f} vs B=64 {l256_b64:.4f}"

    # Adversarial continuity across dim=256 vs dim=257 at B=64 (previously jumped by 64x due to bf *!)
    l256_adv = _eval_g_loss_dim(256, 64, in_probe=True)
    l257_adv = _eval_g_loss_dim(257, 64, in_probe=True)
    assert 0.5 < l256_adv / max(l257_adv, 1e-8) < 2.0, f"O(B) discontinuity at dim=256/257: {l256_adv:.4f} vs {l257_adv:.4f}"

    # 4. Laguerre post-multiply centering keeps reg.psi strictly zero-mean
    torch.manual_seed(0)
    g_lag = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
    d_lag = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    pr_lag = pg.MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.05)
    tr_lag = pg.GANTrainer(
        pg.get_recipe("minimal", z_dim=4, num_particles=16, prior_kind="mog", sigma_rel=0.05),
        g_lag,
        d_lag,
        prior=pr_lag,
        seed=0,
    )
    centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])
    for _ in range(6):
        yb = centers[torch.randint(0, 4, (32,), generator=rng)] + 0.05 * torch.randn((32, 2), generator=rng)
        tr_lag.step(yb)
    preg = tr_lag.prior_regularizer
    assert isinstance(preg, MinimalPriorRegularizer) and preg.psi is not None
    assert abs(float(preg.psi.mean().item())) < 1e-6, f"Laguerre psi not zero-centered: mean={preg.psi.mean().item()}"

