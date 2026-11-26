"""Unit, parity, semantic-control, checkpointing, and clock-free tests for ``unified_v13`` (``feydy_atomic``)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import subprocess
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.api_bytegan import ByteGANBiGANFixture, ByteGANToyFixture
from experiments.forge.techniques import technique_signature
from particlegan import DEFAULT_RECIPE, GANTrainer, MoGParticlePrior, ParticlePrior, Recipe, get_recipe, init
from particlegan._t1_coupling.atomic import (
    compute_atomic_quotas,
    compute_discrete_atomic_gate,
    match_inverse_atomic_targets,
    split_joint_table_pair,
    update_atomic_ledger,
)
from particlegan._t1_coupling.critic import _T1C_PACKET_KEYS, _validate_t1c_packet
from particlegan._unified_v11.critic import _UNIFIED_V11_PACKET_KEYS, _validate_unified_v11_packet


class _TinyGen(nn.Module):
    def __init__(self, z_dim: int = 2, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 24), nn.SiLU(), nn.Linear(24, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _TinyDisc(nn.Module):
    def __init__(self, in_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("forbidden primitive called")


def test_unified_v13_preset_and_v12_default_preserved() -> None:
    assert DEFAULT_RECIPE == "axiom"
    assert len(fields(Recipe)) == 179

    r12 = get_recipe("unified_v12")
    r13 = get_recipe("unified_v13")
    assert r12.rhbi_self_coupling == "feydy_discrete"
    assert r13.rhbi_self_coupling == "feydy_atomic"
    assert r13.critic_formulation == "unified_v11"
    assert r13.prior_regularizer == "unified_v11"
    assert r13.optimizer_family == "unified_v12_equil"
    assert Recipe.from_dict(r13.to_dict()) == r13
    assert technique_signature(r13)["digest"] != technique_signature(r12)["digest"]


def test_discrete_atomic_gate_selectivity_and_scale_invariance() -> None:
    gen = torch.Generator().manual_seed(0)

    # Continuous distributions (1D Gaussian at B=256/2048/4096, 2D arc, 8D Gaussian, point mass) -> 0.0:
    y_1d_small = torch.randn(256, 1, generator=gen)
    y_1d_large = torch.randn(2048, 1, generator=gen)
    y_1d_huge = torch.randn(4096, 1, generator=gen)
    theta = torch.rand(128, 1, generator=gen) * (1.5 * 3.14159265)
    y_arc = torch.cat([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=1) + 0.03 * torch.randn(
        128, 2, generator=gen
    )
    y_8d_gauss = torch.randn(128, 8, generator=gen)
    y_point = torch.ones(64, 4)

    assert compute_discrete_atomic_gate(y_1d_small) == 0.0
    assert compute_discrete_atomic_gate(y_1d_large) == 0.0
    assert compute_discrete_atomic_gate(y_1d_huge) == 0.0
    assert compute_discrete_atomic_gate(y_arc) == 0.0
    assert compute_discrete_atomic_gate(y_8d_gauss) == 0.0
    assert compute_discrete_atomic_gate(y_point) == 0.0

    # General finite discrete atomic distributions (arbitrary K & D, zero hardcoded 256/8) -> 1.0:
    for k_atoms, dim in ((5, 168), (7, 3), (19, 5), (64, 6), (256, 8)):
        atoms = torch.randn(k_atoms, dim, generator=gen)
        idx = torch.randint(0, k_atoms, (128,), generator=gen)
        y_disc = atoms[idx]
        g_base = compute_discrete_atomic_gate(y_disc)
        assert g_base == 1.0, f"expected 1.0 for K={k_atoms}, D={dim}, got {g_base}"
        # Scale invariance across 6 orders of magnitude:
        assert compute_discrete_atomic_gate(y_disc * 1000.0) == g_base
        assert compute_discrete_atomic_gate(y_disc * 1e-3) == g_base


def test_atomic_quotas_and_ledger_symmetry_breaking() -> None:
    # Exact capacity M == K gives unit quotas even under noisy EMA weights:
    noisy_w = torch.linspace(0.5, 1.5, 17)
    q_eq = compute_atomic_quotas(17, noisy_w)
    assert q_eq.shape == (17,)
    assert int(q_eq.sum().item()) == 17
    assert bool((q_eq == 1).all())

    # Over-capacity M > K preserves floor >= 1 per atom and sums to M:
    w_uneq = torch.tensor([0.70, 0.20, 0.08, 0.02])
    q_over = compute_atomic_quotas(20, w_uneq)
    assert int(q_over.sum().item()) == 20
    assert bool((q_over >= 1).all())
    assert int(q_over[0].item()) > int(q_over[1].item()) > int(q_over[3].item())

    # Under-capacity M < K sums to M:
    q_under = compute_atomic_quotas(3, torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05]))
    assert int(q_under.sum().item()) == 3

    # 1-stays / surplus-moves deterministic symmetry breaking across colliding particles:
    charts = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    w_real = torch.full((4,), 0.25)
    # All 4 particles start colliding near chart 0 with slightly different distances:
    x_colliding = torch.tensor([[0.01, 0.0], [0.05, 0.0], [0.08, 0.0], [0.12, 0.0]])
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    tgt, nn_cc, rc, rd, rt, r_atom = update_atomic_ledger(
        x_colliding, rows, 4, charts, w_real, None, None, None, allow_update=True
    )
    assert torch.unique(tgt).numel() == 4
    assert int(tgt[0].item()) == 0  # Closest colliding particle stays on chart 0!
    assert r_atom >= 0.18  # Still unsettled until all 4 particles physically land!

    # Once all 4 particles land crisply on their assigned charts, r_atomic drops below tau_lo (0.042):
    x_landed = charts[rt] + 0.01 * torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    tgt2, _, rc2, rd2, rt2, r_atom2 = update_atomic_ledger(
        x_landed, rows, 4, charts, w_real, rc, rd, rt, allow_update=True
    )
    assert torch.equal(rt2, rt)  # Zero ping-pong!
    assert torch.equal(tgt2, rt)
    assert r_atom2 < 0.042


def test_joint_table_split_and_inverse_matching() -> None:
    table = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rows = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    fake_x = torch.randn(4, 5, requires_grad=True)
    real_x = torch.randn(4, 5)
    real_z = torch.randn(4, 2, requires_grad=True)

    # Suffix (x, z) split preserves autograd:
    jf = torch.cat([fake_x, table[rows]], dim=-1)
    jr = torch.cat([real_x, real_z], dim=-1)
    split = split_joint_table_pair(jf, jr, table, rows)
    assert split is not None
    fx, fz, rx, rz = split
    assert fx.shape == (4, 5) and fz.shape == (4, 2)
    assert fx.requires_grad and rz.requires_grad

    # Prefix (z, x) split also works:
    split_pre = split_joint_table_pair(torch.cat([table[rows], fake_x], dim=-1), torch.cat([real_z, real_x], dim=-1), table, rows)
    assert split_pre is not None
    assert split_pre[0].shape == (4, 5) and split_pre[1].shape == (4, 2)

    # Nonzero-sigma noisy latent codes cleanly refuse split:
    jf_noisy = torch.cat([fake_x, table[rows] + 0.05 * torch.randn(4, 2)], dim=-1)
    assert split_joint_table_pair(jf_noisy, jr, table, rows) is None

    # Inverse target matching maps each atom to its assigned particle code:
    charts = torch.eye(4)
    row_target = torch.tensor([2, 0, 3, 1], dtype=torch.long)  # particle 0->chart 2, 1->0, 2->3, 3->1
    inv_z, has_inv, ell_z, inv_m = match_inverse_atomic_targets(charts, torch.zeros(4, 2), charts, row_target, table)
    assert bool(has_inv.all())
    assert torch.equal(inv_z[0], table[1])
    assert torch.equal(inv_z[1], table[3])
    assert torch.equal(inv_z[2], table[0])
    assert torch.equal(inv_z[3], table[2])
    assert float(ell_z.item()) > 0.0 and bool((inv_m > 0.0).all())


@pytest.mark.parametrize("mode", ["gauss1d", "arc2d_zero_sigma", "mog2d"])
def test_unified_v13_bit_exact_parity_with_v12_on_continuous_tasks(mode: str) -> None:
    def _run(preset: str) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        if mode == "gauss1d":
            rec = get_recipe(preset, z_dim=2, batch_size=32, num_particles=32, prior_kind="mog", sigma_rel=0.1, total_steps=6)
            g = init.deterministic_orthogonal_(_TinyGen(2, 1))
            d = init.deterministic_orthogonal_(_TinyDisc(1))
            prior = None
        elif mode == "arc2d_zero_sigma":
            rec = get_recipe(preset, z_dim=2, batch_size=32, num_particles=12, prior_kind="particles", sigma_rel=0.0, standardize=False, total_steps=6)
            g = init.deterministic_orthogonal_(_TinyGen(2, 2))
            d = init.deterministic_orthogonal_(_TinyDisc(2))
            prior = ParticlePrior(12, 2, learnable=True, generator=torch.Generator().manual_seed(7))
        else:
            rec = get_recipe(preset, z_dim=2, batch_size=32, num_particles=32, prior_kind="mog", sigma_rel=0.1, standardize=False, total_steps=6)
            g = init.deterministic_orthogonal_(_TinyGen(2, 2))
            d = init.deterministic_orthogonal_(_TinyDisc(2))
            prior = MoGParticlePrior(32, 2, sigma=0.1, standardize=False, learnable=True, generator=torch.Generator().manual_seed(7))
        trainer = GANTrainer(rec, g, d, prior=prior, seed=0, max_steps=6)
        data_rng = torch.Generator().manual_seed(123)
        for _ in range(5):
            if mode == "gauss1d":
                real = torch.randn(32, 1, generator=data_rng)
            elif mode == "arc2d_zero_sigma":
                th = torch.rand(32, 1, generator=data_rng) * 4.5
                real = torch.cat([2.0 * torch.cos(th), 2.0 * torch.sin(th)], dim=1) + 0.03 * torch.randn(32, 2, generator=data_rng)
            else:
                real = torch.randn(32, 2, generator=data_rng)
            trainer.step(real)
        pkt = trainer.opt_d.record.state_dict()["unified_v11"]
        assert set(pkt) == _UNIFIED_V11_PACKET_KEYS  # atomic_ledger remains strictly inactive on continuous tasks!
        return {
            **{f"g.{k}": v.detach().clone() for k, v in g.state_dict().items()},
            **{f"d.{k}": v.detach().clone() for k, v in d.state_dict().items()},
            "z": trainer.prior.z.detach().clone(),
        }

    s12 = _run("unified_v12")
    s13 = _run("unified_v13")
    assert set(s12) == set(s13)
    for k in s12:
        assert torch.equal(s12[k], s13[k]), f"continuous parity drift on {mode} tensor {k}"


def test_unified_v13_zero_forbidden_primitives_on_bytegan_and_bigan() -> None:
    with (
        mock.patch.object(torch.linalg, "svd", _forbid),
        mock.patch.object(torch.linalg, "svdvals", _forbid),
        mock.patch.object(torch.linalg, "eigh", _forbid),
        mock.patch.object(torch.linalg, "eigvalsh", _forbid),
        mock.patch.object(torch.linalg, "pinv", _forbid),
        mock.patch.object(torch.autograd, "grad", _forbid),
    ):
        f_hc = ByteGANToyFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=6)
        for _ in range(4):
            f_hc.step()
        assert f_hc.trainer.opt_d.record.atomic_ledger_initialized is True

        f_bi = ByteGANBiGANFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=6)
        for _ in range(4):
            f_bi.step()
        assert f_bi.opt_d.record.atomic_ledger_initialized is True


@pytest.mark.parametrize("fixture_kind", ["hypercube8", "bigan"])
def test_unified_v13_fresh_object_bit_exact_continuation_across_atomic_activation(fixture_kind: str) -> None:
    def _build(max_steps: int = 12):
        if fixture_kind == "hypercube8":
            return ByteGANToyFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=max_steps)
        return ByteGANBiGANFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=max_steps)

    # Uninterrupted reference run (10 steps):
    ref = _build(12)
    for _ in range(10):
        ref.step()

    # Split run: 4 steps -> checkpoint -> fresh object -> load_state_dict -> 6 more steps:
    first = _build(12)
    for _ in range(4):
        first.step()
    rec_mid = first.trainer.opt_d.record if fixture_kind == "hypercube8" else first.opt_d.record
    assert rec_mid.atomic_ledger_initialized is True
    ckpt = deepcopy(first.state_dict())

    resumed = _build(12)
    resumed.load_state_dict(ckpt)
    for _ in range(6):
        resumed.step()

    if fixture_kind == "hypercube8":
        ref_rec, res_rec = ref.trainer.opt_d.record, resumed.trainer.opt_d.record
        for (k1, v1), (_, v2) in zip(ref.trainer.G.state_dict().items(), resumed.trainer.G.state_dict().items()):
            assert torch.equal(v1, v2), f"HC8 G.{k1} continuation drift"
        assert torch.equal(ref.trainer.prior.z, resumed.trainer.prior.z)
    else:
        ref_rec, res_rec = ref.opt_d.record, resumed.opt_d.record
        for (k1, v1), (_, v2) in zip(ref.G.state_dict().items(), resumed.G.state_dict().items()):
            assert torch.equal(v1, v2), f"BiGAN G.{k1} continuation drift"
        for (k1, v1), (_, v2) in zip(ref.E.state_dict().items(), resumed.E.state_dict().items()):
            assert torch.equal(v1, v2), f"BiGAN E.{k1} continuation drift"
        assert torch.equal(ref.prior.z, resumed.prior.z)

    assert torch.equal(ref_rec.atomic_row_chart, res_rec.atomic_row_chart)
    assert torch.equal(ref_rec.atomic_row_dist, res_rec.atomic_row_dist)
    assert torch.equal(ref_rec.atomic_row_target, res_rec.atomic_row_target)
    assert torch.equal(ref_rec.chart_sum, res_rec.chart_sum)
    assert ref_rec._thermal_scale == res_rec._thermal_scale
    assert ref.observe()["metrics"] == resumed.observe()["metrics"]


def test_atomic_ledger_checkpoint_validation_rejects_corruption() -> None:
    f = ByteGANToyFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=6)
    for _ in range(3):
        f.step()
    pkt = deepcopy(f.trainer.opt_d.record.state_dict()["unified_v11"])
    assert "atomic_ledger" in pkt
    _validate_unified_v11_packet(pkt)

    t1c_pkt = {k: pkt[k] for k in (*_T1C_PACKET_KEYS, "atomic_ledger")}
    _validate_t1c_packet(t1c_pkt)

    bad_keys = deepcopy(pkt)
    bad_keys["atomic_ledger"]["extra"] = torch.zeros(256, dtype=torch.long)
    with pytest.raises(ValueError, match="atomic_ledger"):
        _validate_unified_v11_packet(bad_keys)

    bad_dtype = deepcopy(pkt)
    bad_dtype["atomic_ledger"]["row_target"] = bad_dtype["atomic_ledger"]["row_target"].float()
    with pytest.raises(ValueError, match="dtypes"):
        _validate_unified_v11_packet(bad_dtype)

    bad_nan = deepcopy(pkt)
    bad_nan["atomic_ledger"]["row_dist"][0] = float("nan")
    with pytest.raises(ValueError, match="row_dist"):
        _validate_unified_v11_packet(bad_nan)

    bad_idx = deepcopy(pkt)
    bad_idx["atomic_ledger"]["row_target"][0] = -2
    with pytest.raises(ValueError, match="indices"):
        _validate_unified_v11_packet(bad_idx)

    # Unaltered unified_v12 never emits atomic_ledger:
    f12 = ByteGANToyFixture(device="cpu", seed=0, recipe_name="unified_v12", max_steps=4)
    for _ in range(2):
        f12.step()
    pkt12 = f12.trainer.opt_d.record.state_dict()["unified_v11"]
    assert set(pkt12) == _UNIFIED_V11_PACKET_KEYS


def test_unified_v13_clockfree_horizon_invariance_on_atomic_target() -> None:
    fa = ByteGANToyFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=50)
    fb = ByteGANToyFixture(device="cpu", seed=0, recipe_name="unified_v13", max_steps=4000)
    for _ in range(6):
        fa.step()
        fb.step()
    for (k, va), (_, vb) in zip(fa.trainer.G.state_dict().items(), fb.trainer.G.state_dict().items()):
        assert torch.equal(va, vb), f"horizon-dependent drift on G.{k}"
    assert torch.equal(fa.trainer.prior.z, fb.trainer.prior.z)
    assert torch.equal(fa.trainer.opt_d.record.atomic_row_target, fb.trainer.opt_d.record.atomic_row_target)


def test_fresh_process_import_order_isolation() -> None:
    code = (
        "from particlegan import DEFAULT_RECIPE, get_recipe; "
        "assert DEFAULT_RECIPE == 'axiom'; "
        "r12 = get_recipe('unified_v12'); r13 = get_recipe('unified_v13'); "
        "assert r12.rhbi_self_coupling == 'feydy_discrete'; "
        "assert r13.rhbi_self_coupling == 'feydy_atomic'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
