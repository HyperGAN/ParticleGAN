"""Unit, semantic-control, zero-primitive, zero-sync, Schatten-6, Pythagorean 2-obj, continuation, and clockfree tests for ``unified_v12``."""
from __future__ import annotations

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

from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, ParticlePrior, Recipe, get_recipe, init
from particlegan._unified_v11 import UnifiedV11EquilibratedOptimizer, UnifiedV11StepRecord
from particlegan._unified_v12 import UnifiedV12Optimizer
from particlegan._unified_v12 import optimizer as v12_opt_mod
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


class _ResBNTrunkGen(nn.Module):
    def __init__(self, z_dim: int = 8, width: int = 32, out_dim: int = 8, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.stem = nn.Linear(z_dim, width)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"ffn_in": nn.Linear(width, width), "ffn_out": nn.Linear(width, width)}) for _ in range(depth)]
        )
        self.head = nn.Linear(width, out_dim)

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = nn.functional.leaky_relu(self.stem(z), negative_slope=0.2)
        scale = 0.5 / float(max(1, self.depth))
        for blk in self.blocks:
            b = blk["ffn_out"](nn.functional.leaky_relu(blk["ffn_in"](self._rms_norm(h)), negative_slope=0.2))
            h = h + scale * self._rms_norm(b)
        return self.head(h)


class _DeepResBNDisc(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.depth = int(depth)
        self.inp = nn.Linear(in_dim, hidden)
        self.in_projs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out_projs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, 1)

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.leaky_relu(self.inp(x), negative_slope=0.2)
        scale = 0.5 / float(max(1, self.depth))
        for w_in, w_out in zip(self.in_projs, self.out_projs):
            br = w_out(nn.functional.leaky_relu(w_in(self._rms_norm(h)), negative_slope=0.2))
            h = h + scale * self._rms_norm(br)
        return self.head(self._rms_norm(h)).squeeze(-1)


def _forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("forbidden primitive called")


def test_unified_v12_registration_and_recipe_invariants() -> None:
    assert "unified_v12_equil" in NORMALIZED_FAMILIES
    assert DEFAULT_RECIPE == "unified_v12"
    assert get_recipe() == get_recipe("unified_v12") == get_recipe("gan")
    assert len(fields(Recipe)) == 179

    rec = get_recipe("unified_v12")
    assert rec.optimizer_family == "unified_v12_equil"
    assert rec.critic_formulation == "unified_v11"
    assert rec.prior_regularizer == "unified_v11"
    assert rec.constraint_geometry_mode == "direction_blend"
    assert Recipe.from_dict(rec.to_dict()) == rec

    with pytest.raises(ValueError, match="direction_blend"):
        get_recipe("unified_v12", constraint_geometry_mode="nonascent")

    assert UnifiedV12Optimizer._SUPPORTS_ZEROSYNC_V4_TENSORS is True


def test_unified_v12_zero_primitives_zero_conv_ns_zero_critic_ns_zero_new_tensor() -> None:
    torch.manual_seed(0)
    # 1. 2D MLP GAN: 0 Critic NS, 0 critic probes, 0 non-stiff network tensor state, 0 new_tensor, 0 forbidden primitives:
    rec = get_recipe(
        "unified_v12",
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
    assert isinstance(trainer.opt_g, UnifiedV12Optimizer)
    assert isinstance(trainer.opt_d, UnifiedV12Optimizer)
    assert isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer)
    assert trainer.opt_d._generator_free is False

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

    # Zero torch.Tensor.new_tensor calls inside step():
    with mock.patch.object(torch.Tensor, "new_tensor", _forbid):
        for p in g.parameters():
            p.grad = torch.randn_like(p)
        for p in d.parameters():
            p.grad = torch.randn_like(p)
        trainer.prior.z.grad = torch.randn_like(trainer.prior.z)
        trainer.opt_g.set_sampled_rows(trainer.prior.z, torch.tensor([3, 1, 3, 2, 1]))
        trainer.opt_g.bind_v4_modulation(g_scale=torch.tensor(0.8), wakeup_boost=torch.tensor(1.2), prior_scale=1.0)
        trainer.opt_g.bind_origin_sample(torch.tensor([[0.02, 0.0], [0.01, 0.0]]))
        trainer.opt_d.step()
        trainer.opt_g.bind_protected_from_grads()
        trainer.opt_g.step()

    # 2. Conv2d GAN: strictly 0 Conv2d NS on both G and D (only 2D G.fc uses 5-step NS!):
    rec_c = get_recipe(
        "unified_v12",
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
    orig_2d = v12_opt_mod.quintic_orthogonalize_2d
    orig_b = v12_opt_mod.quintic_orthogonalize_batched

    def spy_2d(mat, *a, **kw):
        called_shapes.append(tuple(mat.shape))
        return orig_2d(mat, *a, **kw)

    def spy_b(mats, *a, **kw):
        called_shapes.append(tuple(mats.shape))
        return orig_b(mats, *a, **kw)

    with (
        mock.patch.object(v12_opt_mod, "quintic_orthogonalize_2d", side_effect=spy_2d),
        mock.patch.object(v12_opt_mod, "quintic_orthogonalize_batched", side_effect=spy_b),
    ):
        tc.step(torch.randn(8, 3, 8, 8), generator_real=torch.randn(8, 3, 8, 8))
    assert tc.opt_d.ns_2d_activations == 0
    assert all(len(s) == 2 and s == (8, 128) for s in called_shapes)


def test_unified_v12_zerosync_v4_tensors_and_prior_dedup_parity() -> None:
    torch.manual_seed(0)
    # 1. Opt-in preserves raw sampled rows & GPU/tensor v4 modulation / origin norm:
    p_tab = nn.Parameter(torch.randn(16, 2))
    raw_dups = torch.tensor([3, 1, 3, 5, 1, 5], dtype=torch.long)
    opt_in = UnifiedV12Optimizer([{"params": [p_tab], "role": "prior"}], lr=0.01)
    assert opt_in._supports_zerosync_v4_tensors() is True
    opt_in.set_sampled_rows(p_tab, raw_dups)
    assert len(opt_in._sampled_rows[p_tab]) == len(raw_dups)
    assert torch.equal(opt_in.sampled_rows_for(p_tab), torch.tensor([1, 3, 5], dtype=torch.long))
    opt_in.bind_origin_sample(torch.tensor([[0.01, 0.0], [0.02, 0.0]]))
    assert opt_in._origin_norm is None and isinstance(opt_in._origin_norm_t, torch.Tensor)
    opt_in.bind_v4_modulation(g_scale=torch.tensor(0.25), wakeup_boost=torch.tensor(2.0), prior_scale=1.0)
    assert isinstance(opt_in._v4_g_scale, torch.Tensor) and isinstance(opt_in._v4_wakeup_boost, torch.Tensor)

    # 2. Exact parity between unsorted duplicate raw_rows (zero-sync sort+is_first) and unique rows:
    z_a = nn.Parameter(torch.randn(20, 4))
    z_b = nn.Parameter(z_a.detach().clone())
    g_z = torch.randn(20, 4)
    z_a.grad = g_z.clone()
    z_b.grad = g_z.clone()
    opt_a = UnifiedV12Optimizer([{"params": [z_a], "role": "prior"}], lr=0.02, smoothing=0.0061)
    opt_b = UnifiedV12Optimizer([{"params": [z_b], "role": "prior"}], lr=0.02, smoothing=0.0061)
    dup_idx = torch.tensor([7, 2, 7, 11, 2, 0, 11, 7], dtype=torch.long)
    uniq_idx = torch.unique(dup_idx, sorted=True)
    opt_a.set_sampled_rows(z_a, dup_idx)
    opt_b.set_sampled_rows(z_b, uniq_idx)
    opt_a.step()
    opt_b.step()
    assert torch.allclose(z_a, z_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(opt_a.state[z_a]["peak_rms"], opt_b.state[z_b]["peak_rms"], atol=1e-6, rtol=1e-6)


def test_unified_v12_pr612_stiff_gate_frob_mp_critic_and_group_split_controls() -> None:
    torch.manual_seed(0)
    # 1. Pure monotonic stiff_gate: 0.0 at ema_fg >= 0.60, 0.5 at 0.45, 1.0 at <= 0.30:
    w_g = nn.Linear(16, 16, bias=False)
    opt_g = UnifiedV12Optimizer([{"params": [w_g.weight], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_g._ema_func_gain = 0.70
    assert opt_g.current_stiff_gate() == 0.0
    opt_g._ema_func_gain = 0.60
    assert opt_g.current_stiff_gate() == 0.0
    w_g.weight.grad = torch.randn_like(w_g.weight)
    opt_g.step()
    assert "m_unit" not in opt_g.state[w_g.weight]

    opt_g._ema_func_gain = 0.45
    assert abs(opt_g.current_stiff_gate() - 0.5) < 1e-6
    opt_g._ema_func_gain = 0.30
    assert opt_g.current_stiff_gate() == 1.0
    opt_g._ema_func_gain = 0.10
    assert opt_g.current_stiff_gate() == 1.0
    w_g.weight.grad = torch.randn_like(w_g.weight)
    opt_g.step()
    assert "m_unit" in opt_g.state[w_g.weight]

    # 2. Exact degree-0 scale-invariance & 0-NS Frob-MP critic group-split invariance:
    w_a = torch.randn(16, 24) * 1.2
    w_b = torch.randn(16, 24) * 1.4
    p1a, p2a = nn.Parameter(w_a.clone()), nn.Parameter(w_b.clone())
    p1b, p2b = nn.Parameter(w_a.clone()), nn.Parameter(w_b.clone())
    ga, gb = torch.randn_like(w_a), torch.randn_like(w_b)
    opt_one = UnifiedV12Optimizer([{"params": [p1a, p2a], "role": "critic", "lr": 0.02}], smoothing=0.0061)
    opt_two = UnifiedV12Optimizer(
        [{"params": [p1b], "role": "critic", "lr": 0.02}, {"params": [p2b], "role": "critic", "lr": 0.02}],
        smoothing=0.0061,
    )
    p1a.grad, p2a.grad = ga.clone(), gb.clone()
    p1b.grad, p2b.grad = ga.clone(), gb.clone()
    opt_one.step()
    opt_two.step()
    assert torch.equal(p1a.detach(), p1b.detach()) and torch.equal(p2a.detach(), p2b.detach())
    assert opt_one.ns_2d_activations == 0 and opt_two.ns_2d_activations == 0

    # 3. Constant flip_thresh=0.45 deadband: leaves -0.40 untouched, smoothly damps -0.55:
    w_t = nn.Parameter(torch.randn(16, 16))
    rec_t = UnifiedV11StepRecord()
    rec_t._thermal_scale = 1.0
    opt_t = UnifiedV12Optimizer([{"params": [w_t], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_t._unified_v11_record = rec_t
    opt_t._ema_func_gain = 0.50
    opt_t._ema_flip_cos = -0.40
    opt_t._update_period2_flip_feedback(s_therm=1.0)
    assert opt_t._ema_func_gain == 0.50  # above -0.45 deadband -> untouched!
    opt_t._ema_flip_cos = -0.55
    opt_t._update_period2_flip_feedback(s_therm=1.0)
    assert opt_t._ema_func_gain < 0.50  # below -0.45 deadband -> smoothly damped!


def test_unified_v12_pythagorean_two_objective_geometry_invariants() -> None:
    torch.manual_seed(0)

    def _cond_rec() -> UnifiedV11StepRecord:
        rec_c = UnifiedV11StepRecord()
        rec_c.t1c_cond_initialized = True
        return rec_c

    # 1. Conditional-fiber K=2 Triangle-Inequality: exact sqrt(2) on orthogonal equal-norm 2-obj, 1.0 on single-loss E:
    p_shared = nn.Parameter(torch.zeros(4))
    p_enc = nn.Parameter(torch.zeros(4))
    opt_orth = UnifiedV12Optimizer(
        [{"params": [p_shared], "role": "generator", "lr": 0.1}, {"params": [p_enc], "role": "encoder", "lr": 0.1}],
        smoothing=0.0,
        direction_blend=True,
    )
    opt_orth._unified_v11_record = _cond_rec()
    l0 = p_shared[0]
    l1 = p_shared[1] + p_enc[0]
    opt_orth.bind_protected_losses((l0, l1))
    (l0 + l1).backward()
    opt_orth.step()
    # On p_shared: g = [1, 1, 0, 0], ||g|| = sqrt(2), mult = sqrt(2) -> step = -0.1 * [1, 1, 0, 0]:
    assert torch.allclose(p_shared.detach(), torch.tensor([-0.1, -0.1, 0.0, 0.0]), atol=1e-6)
    # On p_enc (touched only by L1): mult == 1.0 -> step = -0.1 * [1, 0, 0, 0]:
    assert torch.allclose(p_enc.detach(), torch.tensor([-0.1, 0.0, 0.0, 0.0]), atol=1e-6)

    # 2. Unconditional control (t1c_cond_initialized=False): strictly 1.0 (bit-exact to law4-energy-v11!):
    p_uncond = nn.Parameter(torch.zeros(4))
    opt_uncond = UnifiedV12Optimizer(
        [{"params": [p_uncond], "role": "generator", "lr": 0.1}],
        smoothing=0.0,
        direction_blend=True,
    )
    opt_uncond._unified_v11_record = UnifiedV11StepRecord()
    lu0, lu1 = p_uncond[0], p_uncond[1]
    opt_uncond.bind_protected_losses((lu0, lu1))
    (lu0 + lu1).backward()
    opt_uncond.step()
    inv_s2 = -0.1 / (2.0**0.5)
    assert torch.allclose(p_uncond.detach(), torch.tensor([inv_s2, inv_s2, 0.0, 0.0]), atol=1e-6)

    # 3. Aligned conditional-fiber objectives -> exact 1.0; strongly opposing conditional-fiber -> exact 2.0 cap:
    p_al = nn.Parameter(torch.zeros(4))
    opt_al = UnifiedV12Optimizer([{"params": [p_al], "role": "generator", "lr": 0.1}], smoothing=0.0, direction_blend=True)
    opt_al._unified_v11_record = _cond_rec()
    la, lb = p_al[0], 2.0 * p_al[0]
    opt_al.bind_protected_losses((la, lb))
    (la + lb).backward()
    opt_al.step()
    assert torch.allclose(p_al.detach(), torch.tensor([-0.1, 0.0, 0.0, 0.0]), atol=1e-6)

    p_opp = nn.Parameter(torch.zeros(4))
    opt_opp = UnifiedV12Optimizer([{"params": [p_opp], "role": "generator", "lr": 0.1}], smoothing=0.0, direction_blend=True)
    opt_opp._unified_v11_record = _cond_rec()
    # g0 = [1, 0.1, 0, 0], g1 = [-0.9, 0.1, 0, 0] -> (||g0|| + ||g1||)/||g0+g1|| > 2.0 -> capped strictly at 2.0:
    lo0 = p_opp[0] + 0.1 * p_opp[1]
    lo1 = -0.9 * p_opp[0] + 0.1 * p_opp[1]
    opt_opp.bind_protected_losses((lo0, lo1))
    (lo0 + lo1).backward()
    mults_opp = opt_opp._compute_two_obj_pyth_scales(is_cond_fiber=True)
    assert abs(mults_opp[id(p_opp)] - 2.0) < 1e-12
    opt_opp.abort_pending()

    # 4. Symmetry in (L0, L1), degree-0 scale invariance in (c*L0, c*L1), and role-level group-split invariance:
    w_init_a, w_init_b = torch.randn(8, 8), torch.randn(8, 8)
    p1a, p1b = nn.Parameter(w_init_a.clone()), nn.Parameter(w_init_b.clone())
    p2a, p2b = nn.Parameter(w_init_a.clone()), nn.Parameter(w_init_b.clone())
    x = torch.randn(4, 8)
    opt_1 = UnifiedV12Optimizer([{"params": [p1a, p1b], "role": "generator", "lr": 0.02}], smoothing=0.0, direction_blend=True)
    opt_2 = UnifiedV12Optimizer(
        [{"params": [p2a], "role": "generator", "lr": 0.02}, {"params": [p2b], "role": "generator", "lr": 0.02}],
        smoothing=0.0,
        direction_blend=True,
    )
    opt_1._unified_v11_record = _cond_rec()
    opt_2._unified_v11_record = _cond_rec()
    l1_a = (x @ p1a)[:, :4].square().mean() + (x @ p1b)[:, :4].square().mean()
    l1_b = ((x @ p1a)[:, 4:] - 1.0).square().mean() + ((x @ p1b)[:, 4:] + 0.5).square().mean()
    opt_1.bind_protected_losses((l1_a, l1_b))
    (l1_a + l1_b).backward()
    opt_1.step()

    l2_a = 17.0 * ((x @ p2a)[:, :4].square().mean() + (x @ p2b)[:, :4].square().mean())
    l2_b = 17.0 * (((x @ p2a)[:, 4:] - 1.0).square().mean() + ((x @ p2b)[:, 4:] + 0.5).square().mean())
    opt_2.bind_protected_losses((l2_b, l2_a))  # swapped + 17x scaled + split across 2 param_groups!
    (l2_b + l2_a).backward()
    opt_2.step()
    assert torch.allclose(p1a.detach(), p2a.detach(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(p1b.detach(), p2b.detach(), atol=1e-5, rtol=1e-5)


def test_unified_v12_period2_flip_controller_invariants_and_continuation(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # 1. Degree-0 scale invariance across 1e-30..1e30 & NaN safety of _sketch_grad_67:
    g0 = torch.randn(32, 32)
    sk_base = v12_opt_mod._sketch_grad_67(g0, roll=13)
    assert sk_base.shape == (67,)
    for scale in (1e-30, 1e-12, 1e12, 1e30):
        sk_sc = v12_opt_mod._sketch_grad_67(g0 * scale, roll=13)
        assert torch.allclose(sk_base, sk_sc, atol=1e-6, rtol=1e-6)
    g_nan = g0.clone()
    g_nan[0, 0] = float("nan")
    assert torch.all(v12_opt_mod._sketch_grad_67(g_nan, roll=13) == 0.0)

    # 2. Zero trigger when ema_fg >= 0.60 or s_therm < 0.5; zero false triggers on ResBN-like non-flipping grads:
    w_ok = nn.Parameter(torch.randn(16, 16))
    rec_hi = UnifiedV11StepRecord()
    rec_hi._thermal_scale = 1.0
    opt_hi = UnifiedV12Optimizer([{"params": [w_ok], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_hi._unified_v11_record = rec_hi
    opt_hi._ema_func_gain = 0.65
    for step_i in range(10):
        w_ok.grad = (1.0 if step_i % 2 == 0 else -1.0) * g0[:16, :16]
        opt_hi.step()
    assert opt_hi._prev_flip_sketch is None
    assert opt_hi._ema_flip_cos == 0.0
    assert opt_hi._ema_func_gain == 0.65

    w_resbn = nn.Parameter(torch.randn(16, 16))
    opt_resbn = UnifiedV12Optimizer([{"params": [w_resbn], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_resbn._unified_v11_record = rec_hi
    opt_resbn._ema_func_gain = 0.50
    for _ in range(30):
        w_resbn.grad = torch.randn(16, 16)
        opt_resbn.step()
    assert opt_resbn._ema_flip_cos > -0.45
    assert opt_resbn._ema_func_gain == 0.50

    # 3. Closed-loop period-2 sign-flipping drives ema_fg from 0.50 down to <= 0.25 -> stiff_gate == 1.0 & NaN-safe:
    w_flip = nn.Parameter(torch.randn(16, 16))
    opt_flip = UnifiedV12Optimizer([{"params": [w_flip], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_flip._unified_v11_record = rec_hi
    opt_flip._ema_func_gain = 0.50
    g_pat = torch.randn(16, 16)
    for step_i in range(40):
        w_flip.grad = (1.0 if step_i % 2 == 0 else -1.0) * g_pat
        opt_flip.step()
    assert opt_flip._ema_flip_cos < -0.70
    assert opt_flip._ema_func_gain <= 0.25
    assert opt_flip.current_stiff_gate() == 1.0
    prev_sk_before_nan = opt_flip._prev_flip_sketch.clone()
    w_flip.grad = torch.full_like(w_flip, float("nan"))
    opt_flip.step()
    assert torch.equal(opt_flip._prev_flip_sketch, prev_sk_before_nan)
    opt_flip.validate_state_dict(opt_flip.state_dict())

    # 4. Group-split invariance of period-2 flip controller:
    w_a0, w_b0 = torch.randn(16, 16), torch.randn(16, 16)
    pa1, pb1 = nn.Parameter(w_a0.clone()), nn.Parameter(w_b0.clone())
    pa2, pb2 = nn.Parameter(w_a0.clone()), nn.Parameter(w_b0.clone())
    o_one = UnifiedV12Optimizer([{"params": [pa1, pb1], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    o_two = UnifiedV12Optimizer(
        [{"params": [pa2], "role": "generator", "lr": 0.014}, {"params": [pb2], "role": "generator", "lr": 0.014}],
        smoothing=0.0061,
    )
    o_one._unified_v11_record = rec_hi
    o_two._unified_v11_record = rec_hi
    o_one._ema_func_gain = 0.50
    o_two._ema_func_gain = 0.50
    ga_pat, gb_pat = torch.randn(16, 16), torch.randn(16, 16)
    for step_i in range(18):
        sgn = 1.0 if step_i % 2 == 0 else -1.0
        pa1.grad, pb1.grad = sgn * ga_pat, sgn * gb_pat
        pa2.grad, pb2.grad = sgn * ga_pat, sgn * gb_pat
        o_one.step()
        o_two.step()
    assert o_one._ema_flip_cos == o_two._ema_flip_cos
    assert o_one._ema_func_gain == o_two._ema_func_gain
    assert torch.equal(pa1.detach(), pa2.detach()) and torch.equal(pb1.detach(), pb2.detach())

    # 5. Checkpoint tamper rejection for ema_flip_cos and prev_flip_sketch:
    good_sd = opt_flip.state_dict()
    for bad_cos in (1.5, -1.5, float("nan"), True):
        bad_sd = {**good_sd, "unified_v12_equil": {**good_sd["unified_v12_equil"], "ema_flip_cos": bad_cos}}
        with pytest.raises(ValueError, match="ema_flip_cos"):
            opt_flip.validate_state_dict(bad_sd)
    for bad_sk in ([0.0] * 10, [float("nan")] + [0.0] * 66, "bad"):
        bad_sd = {**good_sd, "unified_v12_equil": {**good_sd["unified_v12_equil"], "prev_flip_sketch": bad_sk}}
        with pytest.raises(ValueError, match="prev_flip_sketch"):
            opt_flip.validate_state_dict(bad_sd)

    # 6. Fresh-process bit-exact continuation across active period-2 flip -> stiff_gate saturation boundary:
    w_cont = nn.Parameter(w_a0.clone())
    opt_c = UnifiedV12Optimizer([{"params": [w_cont], "role": "generator", "lr": 0.014}], smoothing=0.0061)
    opt_c._unified_v11_record = rec_hi
    opt_c._ema_func_gain = 0.50
    grads_seq = [(1.0 if i % 2 == 0 else -1.0) * ga_pat for i in range(36)]
    for i in range(4):
        w_cont.grad = grads_seq[i].clone()
        opt_c.step()
    assert opt_c._prev_flip_sketch is not None
    assert -0.45 < opt_c._ema_flip_cos < -0.20 and opt_c._ema_func_gain == 0.50 and opt_c.current_stiff_gate() < 0.5
    ckpt_flip = tmp_path / "ckpt_flip.pt"
    out_flip = tmp_path / "out_flip.pt"
    torch.save({"w": w_cont.detach().clone(), "opt": opt_c.state_dict(), "grads": grads_seq[4:]}, ckpt_flip)
    for i in range(4, 36):
        w_cont.grad = grads_seq[i].clone()
        opt_c.step()
    assert opt_c.current_stiff_gate() == 1.0

    script_flip = f"""
import torch
from torch import nn
from particlegan._unified_v11 import UnifiedV11StepRecord
from particlegan._unified_v12 import UnifiedV12Optimizer

pl = torch.load({str(ckpt_flip)!r}, weights_only=False)
w = nn.Parameter(pl["w"].clone())
opt = UnifiedV12Optimizer([dict(params=[w], role="generator", lr=0.014)], smoothing=0.0061)
rec = UnifiedV11StepRecord()
rec._thermal_scale = 1.0
opt._unified_v11_record = rec
opt.load_state_dict(pl["opt"])
for g in pl["grads"]:
    w.grad = g.clone()
    opt.step()
torch.save({{"w": w.detach(), "ema_fg": opt._ema_func_gain, "ema_flip": opt._ema_flip_cos, "stiff": opt.current_stiff_gate()}}, {str(out_flip)!r})
"""
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, "-c", script_flip], check=True, env=env)
    res_flip = torch.load(out_flip, weights_only=False)
    assert torch.equal(w_cont.detach(), res_flip["w"])
    assert opt_c._ema_func_gain == res_flip["ema_fg"]
    assert opt_c._ema_flip_cos == res_flip["ema_flip"]
    assert opt_c.current_stiff_gate() == res_flip["stiff"] == 1.0


def test_unified_v12_generator_free_critic_ns_checkpoint_and_continuation(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # 1. Generator-free restores 5-step 2D Critic NS + Schatten-6 with 0 Conv2d NS, 0 critic probes, 0 new_tensor:
    rec_gf = get_recipe(
        "unified_v12",
        z_dim=4,
        batch_size=16,
        num_particles=32,
        prior_kind="particles",
        prior_reg=0.0,
        total_steps=6,
    )
    d_gf = init.deterministic_orthogonal_(_ShallowDisc2D(in_dim=4))
    prior_gf = ParticlePrior(num_particles=32, z_dim=4, learnable=True)
    _, opt_d_gf = rec_gf.make_optimizers(nn.Identity(), d_gf, prior_gf)
    assert isinstance(opt_d_gf, UnifiedV12Optimizer)
    assert opt_d_gf._generator_free is True
    for p in d_gf.parameters():
        p.grad = torch.randn_like(p)
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
        mock.patch.object(torch.Tensor, "new_tensor", _forbid),
    ):
        opt_d_gf.step()
    assert opt_d_gf.critic_ns_calls == 1
    assert opt_d_gf.probe_consumptions == 0

    # 2. Tamper, generator_free mismatch, and cross-family rejection:
    sd_gf = opt_d_gf.state_dict()
    assert "unified_v12_equil" in sd_gf
    assert sd_gf["unified_v12_equil"]["generator_free"] is True
    opt_d_gf.load_state_dict(sd_gf)

    d_neural = init.deterministic_orthogonal_(_ShallowDisc2D(in_dim=4))
    _, opt_d_neural = rec_gf.make_optimizers(_ShallowGen2D(z_dim=4, out_dim=4), d_neural, ParticlePrior(32, 4, learnable=True))
    with pytest.raises(ValueError, match="metadata"):
        opt_d_neural.load_state_dict(sd_gf)

    opt_v11 = UnifiedV11EquilibratedOptimizer([nn.Parameter(torch.zeros(4, 4))], lr=0.014, smoothing=0.0061)
    with pytest.raises(ValueError):
        opt_v11.load_state_dict(sd_gf)
    with pytest.raises(ValueError):
        opt_d_neural.load_state_dict(opt_v11.state_dict())

    # 3. Fresh-process bit-exact continuation on deep ResBN (no false stiff_gate, 0-NS Frob-MP critic):
    dim = 128
    rec = get_recipe(
        "unified_v12",
        prior_kind="mog",
        standardize=False,
        batch_size=8,
        num_particles=16,
        z_dim=8,
        total_steps=16,
        training_precision="float32",
    )
    g = _ResBNTrunkGen(z_dim=8, width=32, out_dim=dim, depth=4)
    d = _DeepResBNDisc(in_dim=dim, hidden=32, depth=4)
    p = MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False)
    init.deterministic_orthogonal_(g, seed=50)
    init.deterministic_orthogonal_(d, seed=51)
    init.deterministic_orthogonal_(p, seed=52)

    trainer = GANTrainer(rec, g, d, prior=p, seed=0)
    batches = [torch.randn(8, dim, generator=torch.Generator().manual_seed(1100 + i)) for i in range(10)]
    for i in range(5):
        trainer.step(batches[i])
    assert trainer.opt_g._ema_func_gain >= 0.60
    assert trainer.opt_g.current_stiff_gate() == 0.0
    assert all("m_unit" not in trainer.opt_g.state[param] for param in g.parameters())
    assert trainer.opt_d.ns_2d_activations == 0

    ckpt_path = tmp_path / "ckpt_v12_resbn.pt"
    out_path = tmp_path / "resumed_v12_resbn.pt"
    torch.save({"state": trainer.state_dict(), "batches": batches[5:]}, ckpt_path)
    for i in range(5, 10):
        trainer.step(batches[i])
    ref_sample = trainer.sample(4)
    ref_state = trainer.state_dict()
    ref_g_fg = trainer.opt_d.record._g_func_gain

    script = f"""
import torch
from particlegan import GANTrainer, MoGParticlePrior, Recipe
from tests.test_unified_v12 import _ResBNTrunkGen, _DeepResBNDisc

pl = torch.load({str(ckpt_path)!r}, weights_only=False)
state = pl["state"]
rec = Recipe.from_dict(state["recipe"])
tr = GANTrainer(
    rec,
    _ResBNTrunkGen(z_dim=8, width=32, out_dim=128, depth=4),
    _DeepResBNDisc(in_dim=128, hidden=32, depth=4),
    prior=MoGParticlePrior(num_particles=16, z_dim=8, sigma=0.05, standardize=False),
    seed=321,
)
tr.load_state_dict(state)
for b in pl["batches"]:
    tr.step(b)
torch.save({{"sample": tr.sample(4), "models": tr.state_dict()["models"], "g_fg": tr.opt_d.record._g_func_gain}}, {str(out_path)!r})
"""
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    resumed = torch.load(out_path, weights_only=False)
    assert torch.equal(ref_sample, resumed["sample"])
    assert resumed["g_fg"] == ref_g_fg
    for m_name in ("G", "D", "prior"):
        for k, v in ref_state["models"][m_name].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, resumed["models"][m_name][k])
            else:
                assert v == resumed["models"][m_name][k]


def test_unified_v12_clockfree_audit(tmp_path: Path) -> None:
    from experiments.forge.clockfree import run_clockfree, verify_probe

    repo_p = Path(__file__).resolve().parents[1]
    task = json.loads((repo_p / "configs/forge/tasks/clockfree_audit.json").read_text())
    cand = json.loads((repo_p / "configs/forge/ideas/unified-v12.json").read_text())
    audit_res = run_clockfree(
        {"candidate": cand, "protocol": {"seed": 0}},
        task,
        tmp_path / "clockfree_v12",
        "cpu",
    )
    comparisons, audit = verify_probe(task, audit_res["evidence"])
    assert audit["unexplained_clock_dependencies"] == []
    assert len(comparisons) == 4
    assert all(item["reference_sha256"] == item["changed_sha256"] for item in comparisons)
