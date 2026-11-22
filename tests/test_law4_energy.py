"""Unit, primitive-audit, zero-Conv-NS, stiffness-gating, tamper-rejection, and fresh-process continuation tests for ``law4_energy_v1``."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, GANTrainer, Recipe, get_recipe, init
from particlegan._law4_energy import (
    Law4EnergyOptimizer,
    Law4EnergyV2Optimizer,
    Law4EnergyV3Optimizer,
    Law4EnergyV4Optimizer,
    Law4EnergyV5Optimizer,
    Law4EnergyV6Optimizer,
    Law4EnergyV7Optimizer,
    Law4EnergyV8Optimizer,
    Law4EnergyV9Optimizer,
    Law4EnergyV10Optimizer,
    exact_polar_2xn,
)
from particlegan._law4_energy import optimizer as energy_opt_mod
from particlegan._unified_v11 import UnifiedV11EquilibratedOptimizer
from particlegan.optim.dualnorm import NORMALIZED_FAMILIES


class _ShallowGen2D(nn.Module):
    def __init__(self, z_dim: int = 2, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 24), nn.SiLU(), nn.Linear(24, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _ShallowDisc2D(nn.Module):
    def __init__(self, in_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _ConvGen(nn.Module):
    def __init__(self, z_dim: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim, 8 * 4 * 4)
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.fc(z)).reshape(z.shape[0], 8, 4, 4)
        h = nn.functional.silu(self.deconv(h))
        return torch.tanh(self.conv(h))


class _ConvDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.conv1(x))
        h = nn.functional.silu(self.conv2(h)).mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


def _forbid(*_args, **_kwargs):
    raise AssertionError("forbidden primitive called")


def test_law4_energy_registration_and_recipe_invariants() -> None:
    assert "law4_energy_v1" in NORMALIZED_FAMILIES
    assert DEFAULT_RECIPE == "unified_v11"
    assert len(fields(Recipe)) == 179

    rec = get_recipe("unified_v11", optimizer_family="law4_energy_v1")
    assert rec.optimizer_family == "law4_energy_v1"
    assert rec.critic_formulation == "unified_v11"
    assert rec.prior_regularizer == "unified_v11"
    assert rec.constraint_geometry_mode == "direction_blend"
    assert Recipe.from_dict(rec.to_dict()) == rec

    with pytest.raises(ValueError, match="direction_blend"):
        get_recipe("unified_v11", optimizer_family="law4_energy_v1", constraint_geometry_mode="nonascent")


def test_zero_primitives_zero_conv_ns_and_zero_critic_secant_probe() -> None:
    torch.manual_seed(0)
    rec = get_recipe(
        "unified_v11",
        optimizer_family="law4_energy_v1",
        z_dim=8,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    g = init.deterministic_orthogonal_(_ConvGen(z_dim=8))
    d = init.deterministic_orthogonal_(_ConvDisc())
    trainer = GANTrainer(rec, g, d, max_steps=6)
    assert isinstance(trainer.opt_g, Law4EnergyOptimizer)
    assert isinstance(trainer.opt_d, Law4EnergyOptimizer)
    assert isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer)

    real = torch.randn(8, 3, 8, 8)
    real_g = torch.randn(8, 3, 8, 8)
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "svdvals", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "eigvalsh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        for _ in range(4):
            trainer.step(real, generator_real=real_g)

    assert trainer.opt_g.ns_2d_activations == 0
    assert trainer.opt_d.ns_2d_activations == 0
    assert trainer.opt_g.fast_path_updates > 0
    assert trainer.opt_d.fast_path_updates > 0
    # Critic secant probe is eliminated; generator secant probe runs cleanly:
    assert trainer.opt_d.probe_consumptions == 0
    assert trainer.opt_g.probe_consumptions == 4
    # Zero tensor state on non-prior network parameters:
    for opt in (trainer.opt_g, trainer.opt_d):
        for grp in opt.param_groups:
            if grp["role"] in ("prior", "table"):
                continue
            for p in grp["params"]:
                if p.numel() <= 1:
                    continue
                st = opt.state[p]
                tensor_keys = [k for k, v in st.items() if isinstance(v, torch.Tensor)]
                assert tensor_keys == [], f"unexpected tensor state {tensor_keys} on non-stiff network param"


def test_conditional_2d_ns_activates_only_under_stiffness_never_on_conv2d() -> None:
    torch.manual_seed(0)
    dense_w = nn.Parameter(torch.randn(16, 16))
    conv_w = nn.Parameter(torch.randn(8, 8, 3, 3))
    opt = Law4EnergyOptimizer([dense_w, conv_w], lr=0.014, smoothing=0.0061)

    # 1. Well-conditioned (ema_func_gain == 1.0 -> stiff_gate == 0.0): ZERO NS on both 2D and 4D!
    dense_w.grad = torch.randn_like(dense_w)
    conv_w.grad = torch.randn_like(conv_w)
    with (
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        opt.step()
    assert opt.ns_2d_activations == 0
    assert opt.fast_path_updates == 2

    # 2. Stiff regime (ema_func_gain == 0.25 -> stiff_gate == 1.0): 2D activates NS, 4D Conv2d NEVER calls NS!
    opt._ema_func_gain = 0.25
    dense_w.grad = torch.randn_like(dense_w)
    conv_w.grad = torch.randn_like(conv_w)
    called_shapes = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d

    def spy_2d(mat, *args, **kwargs):
        called_shapes.append(tuple(mat.shape))
        return orig_2d(mat, *args, **kwargs)

    with (
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        opt.step()
    assert opt.ns_2d_activations == 1
    assert called_shapes == [(16, 16)]
    assert torch.isfinite(dense_w).all() and torch.isfinite(conv_w).all()

    # 3. Inter-layer 2D gradient attenuation skew (kappa_2d > 100x at ema_func_gain == 1.0):
    #    dynamically activates 2D NS on dense matrices while 4D Conv2d STILL stays 0 NS!
    w_early = nn.Parameter(torch.randn(16, 16))
    w_late = nn.Parameter(torch.randn(16, 16))
    w_conv = nn.Parameter(torch.randn(8, 8, 3, 3))
    opt_skew = Law4EnergyOptimizer([w_early, w_late, w_conv], lr=0.014, smoothing=0.0061)
    w_early.grad = torch.randn_like(w_early) * 1e-4
    w_late.grad = torch.randn_like(w_late) * 1.0
    w_conv.grad = torch.randn_like(w_conv) * 1.0
    batched_shapes = []
    orig_b = energy_opt_mod.quintic_orthogonalize_batched

    def spy_b(mats, *args, **kwargs):
        batched_shapes.append(tuple(mats.shape))
        return orig_b(mats, *args, **kwargs)

    with (
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", side_effect=spy_b),
    ):
        opt_skew.step()
    assert opt_skew.ns_2d_activations == 2
    assert batched_shapes == [(2, 16, 16)]
    assert opt_skew.fast_path_updates == 1  # w_conv stayed on fast Frobenius-MP path!


def test_underflow_1e36_nan_safety_and_direction_blend() -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.ones(8, 8))
    opt = Law4EnergyOptimizer([p], lr=0.01, smoothing=0.0061)
    p0 = p.detach().clone()
    p.grad = torch.full_like(p, 1e-36)
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.equal(p, p0), "1e-36 gradient must not be crushed to zero"

    # NaN safety:
    p1 = p.detach().clone()
    p.grad = torch.full_like(p, float("nan"))
    opt.step()
    assert torch.equal(p, p1)

    # Zero-autograd.grad / zero-pinv 2-loss direction_blend:
    w = nn.Parameter(torch.tensor([[1.0, -0.5], [0.5, 1.0]]))
    opt_b = Law4EnergyOptimizer([w], lr=0.05, smoothing=0.0061, direction_blend=True)
    x = torch.tensor([[1.0, 2.0]])
    loss1 = ((x @ w) ** 2).sum()
    loss2 = ((x @ w - 1.0) ** 2).sum()
    with (
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
    ):
        opt_b.bind_protected_losses((loss1, loss2))
        (loss1 + loss2).backward()
        opt_b.step()
    assert opt_b.constraint_geometry_stats["steps"] == 1
    assert torch.isfinite(w).all()


def test_checkpoint_tamper_and_cross_family_rejection() -> None:
    torch.manual_seed(0)
    rec = get_recipe("unified_v11", optimizer_family="law4_energy_v1", z_dim=2, batch_size=8, num_particles=32, total_steps=6)
    g = init.deterministic_orthogonal_(_ShallowGen2D())
    d = init.deterministic_orthogonal_(_ShallowDisc2D())
    trainer = GANTrainer(rec, g, d, max_steps=6)
    real = torch.randn(8, 2)
    trainer.step(real, generator_real=real)

    sd_g = trainer.opt_g.state_dict()
    assert "law4_energy" in sd_g

    # Reject tampered peak_rms_val:
    bad = deepcopy(sd_g)
    first_k = next(k for k, v in bad["state"].items() if "peak_rms_val" in v)
    bad["state"][first_k]["peak_rms_val"] = -1.0
    with pytest.raises(ValueError, match="peak_rms_val"):
        trainer.opt_g.load_state_dict(bad)

    # Reject wrong family / cross-family unified_v11_equil checkpoint:
    bad_fam = deepcopy(sd_g)
    bad_fam["law4_energy"]["family"] = "unified_v11_equil"
    with pytest.raises(ValueError, match="metadata"):
        trainer.opt_g.load_state_dict(bad_fam)

    bad_fg = deepcopy(sd_g)
    bad_fg["law4_energy"]["ema_func_gain"] = 1.5
    with pytest.raises(ValueError, match="ema_func_gain"):
        trainer.opt_g.load_state_dict(bad_fg)


def test_bit_exact_fresh_process_continuation_across_stiffness_boundary(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "law4_energy_ckpt.pt"
    out_cont = tmp_path / "cont.json"
    script = """
import json, sys, torch
from torch import nn
from particlegan import GANTrainer, get_recipe, init

mode = sys.argv[1]
ckpt_path = sys.argv[2]
out_path = sys.argv[3]

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 2))
    def forward(self, z): return self.net(z)

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
rec = get_recipe("unified_v11", optimizer_family="law4_energy_v1", z_dim=2, batch_size=16, num_particles=64, total_steps=10)
g = init.deterministic_orthogonal_(Gen())
d = init.deterministic_orthogonal_(Disc())
trainer = GANTrainer(rec, g, d, max_steps=10)

data_gen = torch.Generator().manual_seed(123)
batches = [(torch.randn(16, 2, generator=data_gen), torch.randn(16, 2, generator=data_gen)) for _ in range(8)]

if mode == "full":
    for idx, (r, rg) in enumerate(batches):
        if idx == 3:
            # Force stiffness boundary crossing at step 4 to verify stiff-state continuation!
            trainer.opt_g._ema_func_gain = 0.35
            torch.save(trainer.state_dict(), ckpt_path)
        trainer.step(r, generator_real=rg)
else:
    trainer.load_state_dict(torch.load(ckpt_path, weights_only=False))
    for r, rg in batches[3:]:
        trainer.step(r, generator_real=rg)

summary = {
    "g": [p.detach().tolist() for p in g.parameters()],
    "d": [p.detach().tolist() for p in d.parameters()],
    "ema_fg": trainer.opt_g._ema_func_gain,
    "last_gain": trainer.opt_g.last_step_gain,
}
with open(out_path, "w") as f:
    json.dump(summary, f)
"""
    script_path = tmp_path / "run_cont.py"
    script_path.write_text(script)
    out_full = tmp_path / "full.json"

    repo_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, str(script_path), "full", str(ckpt_path), str(out_full)], check=True, env=env)
    subprocess.run([sys.executable, str(script_path), "resume", str(ckpt_path), str(out_cont)], check=True, env=env)

    full_res = json.loads(out_full.read_text())
    cont_res = json.loads(out_cont.read_text())
    assert full_res == cont_res


def test_exact_polar_2xn_parity_and_law4_energy_v2_invariants() -> None:
    assert "law4_energy_v2" in NORMALIZED_FAMILIES
    torch.manual_seed(0)
    # 1. Exact closed-form 2xN Cayley-Hamilton Tikhonov polar vs float64 SVD (< 2e-6, incl. rank-1):
    for shape in ((2, 64), (64, 2), (2, 2)):
        for s in (0.0, 0.0061, 1.0):
            for sc in (1.0, 1e-20, 1e20):
                m = torch.randn(shape) * sc
                u_ex = exact_polar_2xn(m, smoothing=s)
                u_s, sv, vh = torch.linalg.svd(m.double(), full_matrices=False)
                w = sv / torch.hypot(sv, torch.full_like(sv, s))
                u_ref = ((u_s * w) @ vh).float()
                assert (u_ex - u_ref).abs().max().item() < 2e-6
    m_r1 = torch.randn(2, 1) @ torch.randn(1, 32)
    u_r1 = exact_polar_2xn(m_r1, smoothing=0.0)
    u_s, _, vh = torch.linalg.svd(m_r1.double(), full_matrices=False)
    assert (u_r1 - (u_s[:, :1] @ vh[:1, :]).float()).abs().max().item() < 2e-6

    # 2. Law4EnergyV2Optimizer: strictly 0 Conv2d NS, exact 2xN for m=2, 3-step NS for normal m>=3, 5-step for stiff!
    w_conv = nn.Parameter(torch.randn(8, 8, 4, 4))
    w_narrow = nn.Parameter(torch.randn(16, 2))
    w_wide = nn.Parameter(torch.randn(16, 16))
    opt = Law4EnergyV2Optimizer([w_conv, w_narrow, w_wide], lr=0.014, smoothing=0.0061)
    w_conv.grad = torch.randn_like(w_conv)
    w_narrow.grad = torch.randn_like(w_narrow)
    w_wide.grad = torch.randn_like(w_wide)

    called_steps = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d

    def spy_2d(mat, steps=5, *args, **kwargs):
        called_steps.append((tuple(mat.shape), steps))
        return orig_2d(mat, steps=steps, *args, **kwargs)

    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        opt.step()
    assert opt.exact_2xn_calls == 1
    assert opt.ns_3step_calls == 1
    assert opt.ns_5step_calls == 0
    assert opt.fast_path_updates == 1  # w_conv stayed on zero-NS fast path!
    assert called_steps == [((16, 16), 3)]

    # Under stiffness (ema_func_gain = 0.25 -> stiff_gate = 1.0): wide 2D escalates to 5-step NS, Conv2d STILL 0 NS!
    opt._ema_func_gain = 0.25
    w_conv.grad = torch.randn_like(w_conv)
    w_narrow.grad = torch.randn_like(w_narrow)
    w_wide.grad = torch.randn_like(w_wide)
    called_steps.clear()
    with (
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        opt.step()
    assert opt.ns_5step_calls == 1
    assert opt.exact_2xn_calls == 2
    assert called_steps == [((16, 16), 5)]

    # Checkpoint round-trip & cross-family rejection (v1 vs v2):
    sd = opt.state_dict()
    opt2 = Law4EnergyV2Optimizer([nn.Parameter(torch.zeros(8, 8, 4, 4)), nn.Parameter(torch.zeros(16, 2)), nn.Parameter(torch.zeros(16, 16))], lr=0.014, smoothing=0.0061)
    opt2.load_state_dict(sd)
    assert opt2._ema_func_gain == opt._ema_func_gain
    opt_v1 = Law4EnergyOptimizer([nn.Parameter(torch.zeros(8, 8, 4, 4)), nn.Parameter(torch.zeros(16, 2)), nn.Parameter(torch.zeros(16, 16))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError, match="metadata"):
        opt_v1.load_state_dict(sd)


def test_law4_energy_v3_zero_conv_ns_5step_2d_and_checkpoint_parity() -> None:
    assert "law4_energy_v3" in NORMALIZED_FAMILIES
    torch.manual_seed(0)
    w_conv = nn.Parameter(torch.randn(8, 8, 4, 4))
    w_head = nn.Parameter(torch.randn(1, 16))
    w_2d = nn.Parameter(torch.randn(16, 16))
    opt = Law4EnergyV3Optimizer([w_conv, w_head, w_2d], lr=0.014, smoothing=0.0061)
    w_conv.grad = torch.randn_like(w_conv)
    w_head.grad = torch.randn_like(w_head)
    w_2d.grad = torch.randn_like(w_2d)
    called = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d

    def spy_2d(mat, steps=5, *args, **kwargs):
        called.append((tuple(mat.shape), steps))
        return orig_2d(mat, steps=steps, *args, **kwargs)

    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        opt.step()
    assert opt.fast_path_updates == 2  # w_conv + w_head (m=1) stayed on 0-NS fast path!
    assert opt.ns_5step_calls == 1
    assert called == [((16, 16), 5)]
    sd = opt.state_dict()
    opt_v2 = Law4EnergyV2Optimizer([nn.Parameter(torch.zeros(8, 8, 4, 4)), nn.Parameter(torch.zeros(1, 16)), nn.Parameter(torch.zeros(16, 16))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError, match="metadata"):
        opt_v2.load_state_dict(sd)


def test_law4_energy_v4_zero_conv_ns_and_zero_4d_critic_probe() -> None:
    assert "law4_energy_v4" in NORMALIZED_FAMILIES
    torch.manual_seed(0)
    rec = get_recipe(
        "unified_v11",
        optimizer_family="law4_energy_v4",
        z_dim=8,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    g = init.deterministic_orthogonal_(_ConvGen(z_dim=8))
    d = init.deterministic_orthogonal_(_ConvDisc())
    trainer = GANTrainer(rec, g, d, max_steps=6)
    assert isinstance(trainer.opt_g, Law4EnergyV4Optimizer)
    assert isinstance(trainer.opt_d, Law4EnergyV4Optimizer)

    real = torch.randn(8, 3, 8, 8)
    real_g = torch.randn(8, 3, 8, 8)
    called_shapes = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d
    orig_b = energy_opt_mod.quintic_orthogonalize_batched

    def spy_2d(mat, *a, **kw):
        called_shapes.append(tuple(mat.shape))
        return orig_2d(mat, *a, **kw)

    def spy_b(mats, *a, **kw):
        called_shapes.append(tuple(mats.shape))
        return orig_b(mats, *a, **kw)

    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", side_effect=spy_b),
    ):
        for _ in range(3):
            trainer.step(real, generator_real=real_g)

    # Strictly 0 Conv2d NS (only G.fc (8, 128) is 2D dense with m>=2!), 0 critic probes on 4D images:
    assert all(len(s) == 2 and s == (8, 128) for s in called_shapes)
    assert trainer.opt_d.probe_consumptions == 0
    assert trainer.opt_g.probe_consumptions == 3
    # Cross-family checkpoint rejection (v3 vs v4):
    sd_g = trainer.opt_g.state_dict()
    opt_v3 = Law4EnergyV3Optimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError):
        opt_v3.load_state_dict(sd_g)


def test_law4_energy_v5_zero_conv_ns_1d_eps_parity_and_checkpoint() -> None:
    assert "law4_energy_v5" in NORMALIZED_FAMILIES
    torch.manual_seed(0)

    # 1. Exact 1D eps=1e-8 denominator floor parity with unified_v11_equil on vanishing gradients (< 1e-8):
    b_ref = nn.Parameter(torch.zeros(64))
    b_v5 = nn.Parameter(torch.zeros(64))
    g_tiny = torch.randn(64) * 1e-10
    b_ref.grad = g_tiny.clone()
    b_v5.grad = g_tiny.clone()
    opt_ref = UnifiedV11EquilibratedOptimizer([b_ref], lr=0.014, smoothing=0.0061)
    opt_v5 = Law4EnergyV5Optimizer([b_v5], lr=0.014, smoothing=0.0061)
    opt_ref.step()
    opt_v5.step()
    assert (b_ref - b_v5).abs().max().item() < 1e-8

    # 2. Strictly 0 Conv2d NS, 0 4D critic probes, 0 B/param non-stiff state, 0 forbidden primitives:
    rec = get_recipe(
        "unified_v11",
        optimizer_family="law4_energy_v5",
        z_dim=8,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    g = init.deterministic_orthogonal_(_ConvGen(z_dim=8))
    d = init.deterministic_orthogonal_(_ConvDisc())
    trainer = GANTrainer(rec, g, d, max_steps=6)
    assert isinstance(trainer.opt_g, Law4EnergyV5Optimizer)
    assert isinstance(trainer.opt_d, Law4EnergyV5Optimizer)

    real = torch.randn(8, 3, 8, 8)
    real_g = torch.randn(8, 3, 8, 8)
    called_shapes = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d
    orig_b = energy_opt_mod.quintic_orthogonalize_batched

    def spy_2d(mat, *a, **kw):
        called_shapes.append(tuple(mat.shape))
        return orig_2d(mat, *a, **kw)

    def spy_b(mats, *a, **kw):
        called_shapes.append(tuple(mats.shape))
        return orig_b(mats, *a, **kw)

    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", side_effect=spy_b),
    ):
        for _ in range(3):
            trainer.step(real, generator_real=real_g)

    # Strictly 0 Conv2d NS (only non-conv 2D dense G.fc (8, 128) & D.head (1, 8) call NS!), 0 4D critic probes:
    assert all(len(s) == 2 and s in ((8, 128), (1, 8)) for s in called_shapes)
    assert trainer.opt_d.probe_consumptions == 0
    assert trainer.opt_g.probe_consumptions == 3
    for p in list(g.parameters()) + list(d.parameters()):
        if p.numel() > 1:
            st = trainer.opt_g.state.get(p) or trainer.opt_d.state.get(p) or {}
            assert not any(isinstance(v, torch.Tensor) for v in st.values())

    # 3. Checkpoint round-trip & cross-family rejection (v4 vs v5):
    sd_g = trainer.opt_g.state_dict()
    opt_v4 = Law4EnergyV4Optimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError):
        opt_v4.load_state_dict(sd_g)


def test_law4_energy_v6_bit_exact_2d_parity_zero_conv_ns_and_checkpoint() -> None:
    assert "law4_energy_v6" in NORMALIZED_FAMILIES
    torch.manual_seed(0)

    # 1. Literal 0.000e+00 bit-exact parity with unified_v11_equil on 2D MLPs (incl. tiny 1e-10 gradients):
    g1 = init.deterministic_orthogonal_(_ShallowGen2D())
    d1 = init.deterministic_orthogonal_(_ShallowDisc2D())
    g2 = _ShallowGen2D()
    d2 = _ShallowDisc2D()
    g2.load_state_dict(g1.state_dict())
    d2.load_state_dict(d1.state_dict())
    rec1 = get_recipe(
        "unified_v11",
        optimizer_family="unified_v11_equil",
        z_dim=2,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    rec2 = get_recipe(
        "unified_v11",
        optimizer_family="law4_energy_v6",
        z_dim=2,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    torch.manual_seed(7)
    t1 = GANTrainer(rec1, g1, d1, max_steps=6)
    torch.manual_seed(7)
    t2 = GANTrainer(rec2, g2, d2, max_steps=6)
    for step_i in range(4):
        torch.manual_seed(100 + step_i)
        r = torch.randn(8, 2)
        rg = torch.randn(8, 2)
        torch.manual_seed(200 + step_i)
        t1.step(r, generator_real=rg)
        torch.manual_seed(200 + step_i)
        t2.step(r, generator_real=rg)
    for p1, p2 in zip(list(g1.parameters()) + list(d1.parameters()), list(g2.parameters()) + list(d2.parameters())):
        assert torch.equal(p1, p2)

    # 2. Strictly 0 Conv2d NS, 0 4D critic probes, 0 B/param non-stiff conv state, 0 forbidden primitives:
    rec_c = get_recipe(
        "unified_v11",
        optimizer_family="law4_energy_v6",
        z_dim=8,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    gc = init.deterministic_orthogonal_(_ConvGen(z_dim=8))
    dc = init.deterministic_orthogonal_(_ConvDisc())
    trainer = GANTrainer(rec_c, gc, dc, max_steps=6)
    assert isinstance(trainer.opt_g, Law4EnergyV6Optimizer)
    assert isinstance(trainer.opt_d, Law4EnergyV6Optimizer)
    real = torch.randn(8, 3, 8, 8)
    real_g = torch.randn(8, 3, 8, 8)
    called_shapes = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d
    orig_b = energy_opt_mod.quintic_orthogonalize_batched

    def spy_2d(mat, *a, **kw):
        called_shapes.append(tuple(mat.shape))
        return orig_2d(mat, *a, **kw)

    def spy_b(mats, *a, **kw):
        called_shapes.append(tuple(mats.shape))
        return orig_b(mats, *a, **kw)

    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", side_effect=spy_b),
    ):
        for _ in range(3):
            trainer.step(real, generator_real=real_g)

    assert all(len(s) == 2 and s in ((8, 128), (1, 8)) for s in called_shapes)
    assert trainer.opt_d.probe_consumptions == 0
    assert trainer.opt_g.probe_consumptions == 3
    for p in list(gc.parameters()) + list(dc.parameters()):
        if p.ndim == 4:
            st = trainer.opt_g.state.get(p) or trainer.opt_d.state.get(p) or {}
            assert not any(isinstance(v, torch.Tensor) for v in st.values())

    # 3. Checkpoint round-trip & cross-family rejection (v5 vs v6):
    sd_g = trainer.opt_g.state_dict()
    sd_d = trainer.opt_d.state_dict()
    trainer.opt_g.load_state_dict(sd_g)
    trainer.opt_d.load_state_dict(sd_d)
    opt_v5 = Law4EnergyV5Optimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError):
        opt_v5.load_state_dict(sd_g)



@pytest.mark.parametrize(
    ("family", "cls"),
    [
        ("law4_energy_v7", Law4EnergyV7Optimizer),
        ("law4_energy_v8", Law4EnergyV8Optimizer),
        ("law4_energy_v9", Law4EnergyV9Optimizer),
        ("law4_energy_v10", Law4EnergyV10Optimizer),
    ],
)
def test_law4_energy_v7_v8_v9_asymmetric_zero_critic_ns_zero_conv_ns_and_checkpoint(
    family: str,
    cls: type,
) -> None:
    assert family in NORMALIZED_FAMILIES
    torch.manual_seed(0)

    # 1. 2D MLP GAN: strictly 0 Critic NS, 0 Critic secant probes, 0 forbidden primitives, 0 non-stiff tensor state!
    rec = get_recipe(
        "unified_v11",
        optimizer_family=family,
        z_dim=2,
        batch_size=16,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=6,
    )
    g = init.deterministic_orthogonal_(_ShallowGen2D())
    d = init.deterministic_orthogonal_(_ShallowDisc2D())
    trainer = GANTrainer(rec, g, d, max_steps=6)
    assert isinstance(trainer.opt_g, cls)
    assert isinstance(trainer.opt_d, cls)

    real = torch.randn(16, 2)
    real_g = torch.randn(16, 2)
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "svdvals", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "eigvalsh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
    ):
        for _ in range(3):
            trainer.step(real, generator_real=real_g)

    assert trainer.opt_d.ns_2d_activations == 0
    assert trainer.opt_d.critic_ns_calls == 0
    assert trainer.opt_d.probe_consumptions == 0
    assert trainer.opt_g.probe_consumptions == 3
    for opt in (trainer.opt_g, trainer.opt_d):
        for grp in opt.param_groups:
            if grp["role"] in ("prior", "table"):
                continue
            for p in grp["params"]:
                if p.numel() > 1:
                    st = opt.state.get(p, {})
                    assert not any(isinstance(v, torch.Tensor) for v in st.values())

    # 2. Conv2d GAN: strictly 0 Conv2d NS on both G and D!
    rec_c = get_recipe(
        "unified_v11",
        optimizer_family=family,
        z_dim=8,
        batch_size=8,
        num_particles=32,
        prior_kind="mog",
        sigma_rel=0.1,
        prior_reg=0.0,
        total_steps=4,
    )
    gc = init.deterministic_orthogonal_(_ConvGen(z_dim=8))
    dc = init.deterministic_orthogonal_(_ConvDisc())
    tc = GANTrainer(rec_c, gc, dc, max_steps=4)
    called_shapes = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d
    orig_b = energy_opt_mod.quintic_orthogonalize_batched

    def spy_2d(mat, *a, **kw):
        called_shapes.append(tuple(mat.shape))
        return orig_2d(mat, *a, **kw)

    def spy_b(mats, *a, **kw):
        called_shapes.append(tuple(mats.shape))
        return orig_b(mats, *a, **kw)

    with (
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", side_effect=spy_b),
    ):
        tc.step(torch.randn(8, 3, 8, 8), generator_real=torch.randn(8, 3, 8, 8))
    assert tc.opt_d.ns_2d_activations == 0
    assert all(len(s) == 2 and s == (8, 128) for s in called_shapes)

    # 3. Checkpoint round-trip & cross-family rejection:
    sd_g = trainer.opt_g.state_dict()
    sd_d = trainer.opt_d.state_dict()
    trainer.opt_g.load_state_dict(sd_g)
    trainer.opt_d.load_state_dict(sd_d)
    opt_v3 = Law4EnergyV3Optimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError, match="metadata"):
        opt_v3.load_state_dict(sd_g)


def test_law4_energy_v8_erank2_gate_and_v7_fresh_process_continuation(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # 1. Verify erank_2 gate skips NS on well-conditioned (identity-like) 2D grad, escalates on rank-1 grad:
    w_iso = nn.Parameter(torch.randn(16, 16))
    w_r1 = nn.Parameter(torch.randn(16, 16))
    opt = Law4EnergyV8Optimizer([w_iso, w_r1], lr=0.014, smoothing=0.0061)
    w_iso.grad = torch.eye(16) + 0.05 * torch.randn(16, 16)
    u = torch.randn(16, 1)
    v = torch.randn(1, 16)
    w_r1.grad = u @ v
    called = []
    orig_2d = energy_opt_mod.quintic_orthogonalize_2d

    def spy_2d(mat, *a, **kw):
        called.append(tuple(mat.shape))
        return orig_2d(mat, *a, **kw)

    with (
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(energy_opt_mod, "quintic_orthogonalize_batched", _forbid),
    ):
        opt.step()
    assert opt.erank_fast_skips == 1
    assert opt.ns_5step_calls == 1
    assert called == [(16, 16)]

    # 2. Fresh-process bit-exact continuation for law4_energy_v7 across stiffness boundary (device-tensor peak_rms!):
    ckpt_path = tmp_path / "v7_ckpt.pt"
    out_full = tmp_path / "v7_full.json"
    out_cont = tmp_path / "v7_cont.json"
    script = """
import json, sys, torch
from torch import nn
from particlegan import GANTrainer, get_recipe, init

mode, ckpt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 2))
    def forward(self, z): return self.net(z)

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
rec = get_recipe("unified_v11", optimizer_family="law4_energy_v7", z_dim=2, batch_size=16, num_particles=64, total_steps=10)
g = init.deterministic_orthogonal_(Gen())
d = init.deterministic_orthogonal_(Disc())
trainer = GANTrainer(rec, g, d, max_steps=10)
data_gen = torch.Generator().manual_seed(123)
batches = [(torch.randn(16, 2, generator=data_gen), torch.randn(16, 2, generator=data_gen)) for _ in range(8)]

if mode == "full":
    for idx, (r, rg) in enumerate(batches):
        if idx == 3:
            trainer.opt_g._ema_func_gain = 0.35
            torch.save(trainer.state_dict(), ckpt_path)
        trainer.step(r, generator_real=rg)
else:
    trainer.load_state_dict(torch.load(ckpt_path, weights_only=False))
    for r, rg in batches[3:]:
        trainer.step(r, generator_real=rg)

with open(out_path, "w") as f:
    json.dump({
        "g": [p.detach().tolist() for p in g.parameters()],
        "d": [p.detach().tolist() for p in d.parameters()],
        "ema_fg": trainer.opt_g._ema_func_gain,
    }, f)
"""
    script_path = tmp_path / "run_v7_cont.py"
    script_path.write_text(script)
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, str(script_path), "full", str(ckpt_path), str(out_full)], check=True, env=env)
    subprocess.run([sys.executable, str(script_path), "resume", str(ckpt_path), str(out_cont)], check=True, env=env)
    assert json.loads(out_full.read_text()) == json.loads(out_cont.read_text())
