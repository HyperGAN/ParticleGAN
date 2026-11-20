"""Unit, fresh-process continuation, clock-free, and semantic-control tests for Theory 1A Pullback Prior."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import (
    DEFAULT_RECIPE,
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    T1PullbackPriorRegularizer,
    get_recipe,
    init,
)
from experiments.forge.boundaries import validate_registry
from experiments.forge.clockfree import run_clockfree
from experiments.forge.priors import prior_policy_receipt, validate_prior_policy
from experiments.forge.state import require_optimizer_steps, state_digest
from experiments.forge.views import grade_result


ROOT = Path(__file__).resolve().parents[1]


class _TinyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_default_recipe_and_incumbent_presets_byte_identical():
    validate_registry()
    assert DEFAULT_RECIPE == "empf"
    assert get_recipe() == get_recipe("empf")

    for preset_name in ("empf", "rhbi_optimized", "bcap", "ka2"):
        d = get_recipe(preset_name).to_dict()
        for t1_key in (
            "t1_pullback_weight",
            "t1_iso_weight",
            "t1_laguerre_lr",
            "t1_prior_row_lr",
            "t1_max_prototypes",
        ):
            assert t1_key not in d, f"{t1_key} leaked into {preset_name}.to_dict()"

    t1 = get_recipe("t1_prior")
    assert t1.prior_regularizer == "t1_pullback"
    assert t1.t1_pullback_weight > 0.0
    assert t1.lr_floor == 1.0 and t1.network_lr_floor == 1.0

    with pytest.raises(ValueError, match="t1_pullback_weight"):
        get_recipe("t1_prior", t1_pullback_weight=-0.1)
    with pytest.raises(ValueError, match="t1_iso_weight"):
        get_recipe("t1_prior", t1_iso_weight=-0.1)
    with pytest.raises(ValueError, match="t1_laguerre_lr"):
        get_recipe("t1_prior", t1_laguerre_lr=0.0)
    with pytest.raises(ValueError, match="t1_prior_row_lr"):
        get_recipe("t1_prior", t1_prior_row_lr=-0.01)
    with pytest.raises(ValueError, match="t1_max_prototypes"):
        get_recipe("t1_prior", t1_max_prototypes=1)


def test_disabled_t1_pullback_matches_empf_exactly():
    base_recipe = get_recipe(
        "empf",
        num_particles=32,
        z_dim=2,
        batch_size=16,
        total_steps=20,
        prior_kind="mog",
        sigma_rel=0.025,
        standardize=False,
        training_precision="float32",
    )
    inactive_t1 = base_recipe.replace(prior_regularizer="t1_pullback", t1_pullback_weight=0.0)

    results = []
    for rec in (base_recipe, inactive_t1):
        g = _TinyMLP(2, 2)
        d = _TinyMLP(2, 1)
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        prior = MoGParticlePrior(num_particles=32, z_dim=2, sigma=0.025, standardize=False, learnable=True)
        init.deterministic_orthogonal_(prior, seed=2)
        trainer = GANTrainer(rec, g, d, prior=prior, seed=0)
        data_rng = torch.Generator().manual_seed(42)
        for _ in range(5):
            real = torch.randn(16, 2, generator=data_rng)
            trainer.step(real)
        results.append((g.state_dict(), d.state_dict(), prior.z.detach().clone()))

    for k in results[0][0]:
        assert torch.equal(results[0][0][k], results[1][0][k])
    for k in results[0][1]:
        assert torch.equal(results[0][1][k], results[1][1][k])
    assert torch.equal(results[0][2], results[1][2])


def test_t1_prior_fresh_process_checkpoint_continuation_and_tamper_rejection(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rec = get_recipe(
            "t1_prior",
            num_particles=32,
            z_dim=2,
            batch_size=16,
            total_steps=20,
            prior_kind="mog",
            sigma_rel=0.025,
            standardize=False,
            training_precision="float32",
        )
        g = _TinyMLP(2, 2)
        d = _TinyMLP(2, 1)
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        prior = MoGParticlePrior(num_particles=32, z_dim=2, sigma=0.025, standardize=False, learnable=True)
        init.deterministic_orthogonal_(prior, seed=2)
        trainer = GANTrainer(rec, g, d, prior=prior, seed=0)
        assert isinstance(trainer.prior_regularizer, T1PullbackPriorRegularizer)

        data_rng = torch.Generator().manual_seed(99)
        for _ in range(6):
            real = torch.randn(16, 2, generator=data_rng)
            trainer.step(real)

        ckpt_mid = trainer.state_dict()
        assert "prior_regularizer" in ckpt_mid
        require_optimizer_steps({"trainer": ckpt_mid}, 6)

        mid_path = tmp_path / "mid.pt"
        out_path = tmp_path / "resumed.pt"
        torch.save(
            {
                "trainer": ckpt_mid,
                "data_rng": data_rng.get_state(),
                "recipe": rec.to_dict(),
            },
            mid_path,
        )

        # Continue uninterrupted reference for 6 more steps.
        for _ in range(6):
            real = torch.randn(16, 2, generator=data_rng)
            trainer.step(real)
        ref_final = trainer.state_dict()
        require_optimizer_steps({"trainer": ref_final}, 12)
        eval_rng_ref = torch.Generator().manual_seed(777)
        ref_samples = trainer.sample(64, generator=eval_rng_ref)

        # Resume from mid.pt inside a genuinely fresh Python subprocess with weights_only=True.
        script = f"""
import torch
from torch import nn
from particlegan import GANTrainer, MoGParticlePrior, Recipe, init

torch.set_num_threads(1)

class _TinyMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

payload = torch.load({str(mid_path)!r}, map_location="cpu", weights_only=True)
rec = Recipe.from_dict(payload["recipe"])
g = _TinyMLP(2, 2)
d = _TinyMLP(2, 1)
prior = MoGParticlePrior(num_particles=32, z_dim=2, sigma=0.025, standardize=False, learnable=True)
trainer = GANTrainer(rec, g, d, prior=prior, seed=0)
trainer.load_state_dict(payload["trainer"])

data_rng = torch.Generator()
data_rng.set_state(payload["data_rng"])
for _ in range(6):
    real = torch.randn(16, 2, generator=data_rng)
    trainer.step(real)

eval_rng = torch.Generator().manual_seed(777)
samples = trainer.sample(64, generator=eval_rng)
torch.save({{"trainer": trainer.state_dict(), "samples": samples}}, {str(out_path)!r})
"""
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"Fresh subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        resumed = torch.load(out_path, map_location="cpu", weights_only=True)
        require_optimizer_steps({"trainer": resumed["trainer"]}, 12)
        assert state_digest(ref_final) == state_digest(resumed["trainer"])
        assert torch.equal(ref_samples, resumed["samples"])

        # Verify corruption/tamper rejection fails closed without mutating live trainer.
        before_tamper = state_digest(trainer.state_dict())

        missing_reg = deepcopy(ckpt_mid)
        missing_reg.pop("prior_regularizer", None)
        with pytest.raises(ValueError):
            trainer.load_state_dict(missing_reg)

        bad_var = deepcopy(ckpt_mid)
        bad_var["prior_regularizer"]["proto_var"][0] = -1.0
        with pytest.raises(ValueError, match="proto_var"):
            trainer.load_state_dict(bad_var)

        bad_nan = deepcopy(ckpt_mid)
        bad_nan["prior_regularizer"]["proto_c"][0, 0] = float("nan")
        with pytest.raises(ValueError, match="nonfinite"):
            trainer.load_state_dict(bad_nan)

        assert state_digest(trainer.state_dict()) == before_tamper
    finally:
        torch.set_num_threads(previous_threads)


def test_t1_prior_clockfree_audit_and_sampling_purity(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit_measurement_v1.json").read_text())
        request = {
            "candidate": {"id": "t1-probe", "recipe_preset": "t1_prior", "recipe_overrides": {"training_precision": "float32"}},
            "protocol": {"seed": 0},
        }
        original_rng = torch.get_rng_state().clone()
        raw = run_clockfree(request, task, tmp_path, "cpu")
        assert torch.equal(original_rng, torch.get_rng_state())
        grade = grade_result(task, raw)
        assert grade["status"] == "PASS", grade
        assert grade["metrics"]["parity_comparisons"] == 4
        assert raw["evidence"]["source_audit"]["unexplained_clock_dependencies"] == []
    finally:
        torch.set_num_threads(previous_threads)


def test_t1_prior_component_host_and_deliberate_damage_semantic_control():
    # 1. Component host contract + Forge prior_policy_receipt validation
    rec = get_recipe("t1_prior", num_particles=32, z_dim=2, batch_size=16, total_steps=100, training_precision="float32")
    prior = ParticlePrior(num_particles=32, z_dim=2, learnable=True)
    init.deterministic_orthogonal_(prior, seed=0)
    rec.apply_prior_policy(prior)
    reg = rec.make_prior_regularizer(prior=prior, reference=prior)
    loss_comp = reg(prior.z)
    assert torch.isfinite(loss_comp)
    loss_comp.backward()
    assert prior.z.grad is not None and torch.isfinite(prior.z.grad).all()

    task_mock = {
        "adapter": "transfer_image",
        "execution": {
            "prior_contract": "recipe_owned_v1",
            "prior": {"kind": "particle_cloud", "sigma": 0.0, "standardize": False, "exception_reason": "unit test"},
        },
    }
    receipt = {"prior_policy": prior_policy_receipt(rec, prior)}
    assert validate_prior_policy(task_mock, receipt, rec.to_dict()) is None

    # 2. Deliberate-damage control: verify live G(prior) separation vs damaged generator/prior (zero hidden prototype fallback)
    rec_mog = get_recipe(
        "t1_prior",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=120,
        prior_kind="mog",
        sigma_rel=0.025,
        standardize=False,
        training_precision="float32",
    )
    g = _TinyMLP(2, 2, hidden=32)
    d = _TinyMLP(2, 1, hidden=32)
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(d, seed=1)
    mog = MoGParticlePrior(num_particles=64, z_dim=2, sigma=0.025, standardize=False, learnable=True)
    init.deterministic_orthogonal_(mog, seed=2)
    trainer = GANTrainer(rec_mog, g, d, prior=mog, seed=0)

    centers = torch.tensor([[-2.5, 0.0], [2.5, 0.0]])
    rng = torch.Generator().manual_seed(123)
    for _ in range(100):
        idx = torch.randint(2, (32,), generator=rng)
        real = centers[idx] + 0.05 * torch.randn(32, 2, generator=rng)
        trainer.step(real)

    eval_rng = torch.Generator().manual_seed(555)
    learned_samples = trainer.sample(512, generator=eval_rng)
    d_to_modes_learned = torch.cdist(learned_samples, centers).min(dim=1).values.mean().item()
    assert d_to_modes_learned < 0.35, f"Expected bimodal convergence, got mean mode dist {d_to_modes_learned:.3f}"

    # Deliberately damage G weights -> sample() must immediately reflect damaged G, never cached prototypes!
    with torch.no_grad():
        for p in trainer.G.parameters():
            p.zero_()
    eval_rng2 = torch.Generator().manual_seed(555)
    damaged_samples = trainer.sample(512, generator=eval_rng2)
    d_to_modes_damaged = torch.cdist(damaged_samples, centers).min(dim=1).values.mean().item()
    assert d_to_modes_damaged > 2.0, "Damaged generator still produced mode-aligned samples (forbidden fallback!)"


@pytest.mark.parametrize("mode", ["mog_z4_settle_and_shock", "zero_sigma_particle_table"])
def test_t1_prior_fresh_process_continuation_across_activation_boundaries(tmp_path, mode):
    """Verify exact fresh-process checkpoint + RNG continuation across T1 activation boundaries:
    (1) MoG z_dim=4 -> dim=2 Radon-Hurwitz 4-probe branch, settled chart cooling (res_ema < 0.10 -> g_scale < 1.0),
        dead-cell Zador-Gersho split, psi_gate == 0 freeze, AND post-checkpoint abrupt shift shock (shock > 0.25
        + uncovered-mode teleportation max_ur > 1.0 + psi_gate > 0 reopening);
    (2) Zero-sigma ParticlePrior (sigma_z == 0.0, n_tot <= 256) exercising the full-table w_hard Laguerre branch.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        is_mog = mode == "mog_z4_settle_and_shock"
        z_dim = 4 if is_mog else 2
        rec = get_recipe(
            "t1_prior",
            num_particles=48,
            z_dim=z_dim,
            batch_size=32,
            total_steps=60,
            prior_kind="mog" if is_mog else "particles",
            sigma_rel=0.025 if is_mog else 0.0,
            standardize=False,
            training_precision="float32",
        )
        g = _TinyMLP(z_dim, 2, hidden=24)
        d = _TinyMLP(2, 1, hidden=24)
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        if is_mog:
            prior = MoGParticlePrior(num_particles=48, z_dim=z_dim, sigma=0.025, standardize=False, learnable=True)
        else:
            prior = ParticlePrior(num_particles=48, z_dim=z_dim, learnable=True)
        init.deterministic_orthogonal_(prior, seed=2)
        trainer = GANTrainer(rec, g, d, prior=prior, seed=0)
        reg = trainer.prior_regularizer
        assert isinstance(reg, T1PullbackPriorRegularizer)

        data_rng = torch.Generator().manual_seed(2026)
        # Phase 1 (pre-checkpoint): converge tightly on centered target so res_ema < 0.10
        # (activating g_scale < 1.0) and dead cells / low-TV regime activate before checkpoint.
        c_pre = torch.tensor([[-0.35, 0.0], [0.35, 0.0]])
        for _ in range(48):
            idx = torch.randint(2, (32,), generator=data_rng)
            real = c_pre[idx] + 0.25 * torch.randn(32, 2, generator=data_rng)
            trainer.step(real)

        assert reg.quantizer_initialized and reg.rows_initialized
        assert reg.res_ema < 0.10, f"Expected active chart cooling res_ema < 0.10 before checkpoint, got {reg.res_ema}"

        ckpt_mid = trainer.state_dict()
        mid_path = tmp_path / f"mid_{mode}.pt"
        out_path = tmp_path / f"resumed_{mode}.pt"
        torch.save(
            {
                "trainer": ckpt_mid,
                "data_rng": data_rng.get_state(),
                "recipe": rec.to_dict(),
            },
            mid_path,
        )

        # Phase 2 (post-checkpoint): abrupt distribution jump to [+4.5, +4.5] / [-4.5, +4.5]
        # crossing shock > 0.25, uncovered-mode teleportation (max_ur > 1.0), and active psi_gate > 0!
        c_post = torch.tensor([[4.5, 4.5], [-4.5, 4.5]])
        for _ in range(10):
            idx = torch.randint(2, (32,), generator=data_rng)
            real = c_post[idx] + 0.08 * torch.randn(32, 2, generator=data_rng)
            trainer.step(real)

        assert reg.res_ema > 0.10, f"Expected post-shift shock to reopen res_ema > 0.10, got {reg.res_ema}"
        ref_final = trainer.state_dict()
        require_optimizer_steps({"trainer": ref_final}, 58)
        eval_rng_ref = torch.Generator().manual_seed(909)
        ref_samples = trainer.sample(64, generator=eval_rng_ref)

        prior_cls = "MoGParticlePrior" if is_mog else "ParticlePrior"
        prior_args = (
            f"num_particles=48, z_dim={z_dim}, sigma=0.025, standardize=False, learnable=True"
            if is_mog
            else f"num_particles=48, z_dim={z_dim}, learnable=True"
        )
        script = f"""
import torch
from torch import nn
from particlegan import GANTrainer, {prior_cls}, Recipe

torch.set_num_threads(1)

class _TinyMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

payload = torch.load({str(mid_path)!r}, map_location="cpu", weights_only=True)
rec = Recipe.from_dict(payload["recipe"])
g = _TinyMLP({z_dim}, 2, hidden=24)
d = _TinyMLP(2, 1, hidden=24)
prior = {prior_cls}({prior_args})
trainer = GANTrainer(rec, g, d, prior=prior, seed=0)
trainer.load_state_dict(payload["trainer"])

data_rng = torch.Generator()
data_rng.set_state(payload["data_rng"])
c_post = torch.tensor([[4.5, 4.5], [-4.5, 4.5]])
for _ in range(10):
    idx = torch.randint(2, (32,), generator=data_rng)
    real = c_post[idx] + 0.08 * torch.randn(32, 2, generator=data_rng)
    trainer.step(real)

eval_rng = torch.Generator().manual_seed(909)
samples = trainer.sample(64, generator=eval_rng)
torch.save({{"trainer": trainer.state_dict(), "samples": samples}}, {str(out_path)!r})
"""
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"Fresh subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        resumed = torch.load(out_path, map_location="cpu", weights_only=True)
        require_optimizer_steps({"trainer": resumed["trainer"]}, 58)
        assert state_digest(ref_final) == state_digest(resumed["trainer"])
        assert torch.equal(ref_samples, resumed["samples"])
    finally:
        torch.set_num_threads(previous_threads)
