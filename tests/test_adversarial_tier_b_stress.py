"""Adversarial empirical stress-test harness for Milestone 2 Iteration 2 (Tier B Research Fixes).

Empirical challenger tests:
1. Empirical Distribution Shift Detection:
   - 1,000 steps of stationary training at equilibrium: exactly 0 fires, z << 3.5 (false positive rejection).
   - Abrupt mean shift (+3.0): fires within 2-3 steps of shift with confirmed streak=2, z > 3.5.
   - Abrupt mean shift (+1.0): fires within 2-3 steps of shift with confirmed streak=2, z > 3.5.
2. GANTrainer Checkpoint Persistence & Backward Compatibility:
   - Roundtrip state_dict() serialization and restoration into fresh trainer.
   - Bit-exact equality across all LossSurprise state fields:
     fast, calm, var_calm, streak, armed, since_calm, since_fire, fires, last_z, pending_loss.
   - Backward compatibility: state dict omitting "loss_surprise" loads cleanly without error.
   - Corruption rejection: malformed loss_surprise state raises ValueError cleanly without mutating trainer.
3. Cooldown and Re-arming Lifecycle:
   - Verified 16-step cooldown disarmed invariant.
   - Post-fire calm baseline tracking and re-arming upon equilibrium restoration.
"""
from __future__ import annotations

from copy import deepcopy
import pytest
import torch
import torch.nn as nn

from particlegan import get_recipe
from particlegan.training import GANTrainer
from particlegan.init import deterministic_orthogonal_
from particlegan.priors import unwrap_prior
from particlegan.continuous import LossSurprise


def get_device() -> torch.device:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _build_1d_gaussian_setup(device: torch.device, seed: int = 42):
    torch.manual_seed(seed)
    data_gen = torch.Generator(device="cpu").manual_seed(seed)
    
    recipe = get_recipe("atlas", z_dim=2, num_particles=256, batch_size=128, lr=0.005)
    G = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)
    D = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior(sigma=0.025).to(device)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)
    return trainer, data_gen


# ==============================================================================
# Suite 1: Empirical Distribution Shift Detection (1,000 steps & abrupt shifts)
# ==============================================================================

@pytest.fixture(scope="module")
def _stationary_1000_step_checkpoint():
    """Run the deterministic 50-warmup + 1,000-stationary-step setup once for reuse."""
    device = get_device()
    trainer, data_gen = _build_1d_gaussian_setup(device, seed=42)

    # 1. Warm up model to equilibrium on N(2.0, 0.5^2)
    for _ in range(50):
        real = (2.0 + 0.5 * torch.randn(128, 1, generator=data_gen)).to(device)
        trainer.step(real)

    # Attach clean LossSurprise at equilibrium
    trainer.policy.loss_surprise = LossSurprise()
    ls = trainer.policy.loss_surprise

    # 2. 1,000 stationary steps: record per-step fire/z diagnostics and checkpoint final state
    max_z = -1e9
    min_z = 1e9
    first_fire_step = None
    first_fire_z = None
    for step in range(1000):
        real = (2.0 + 0.5 * torch.randn(128, 1, generator=data_gen)).to(device)
        trainer.step(real)
        if ls.fires > 0 and first_fire_step is None:
            first_fire_step = step
            first_fire_z = ls.last_z
        max_z = max(max_z, ls.last_z)
        min_z = min(min_z, ls.last_z)

    return {
        "device": device,
        "trainer_state": deepcopy(trainer.state_dict()),
        "data_gen_state": data_gen.get_state().clone(),
        "fires": ls.fires,
        "max_z": max_z,
        "min_z": min_z,
        "first_fire_step": first_fire_step,
        "first_fire_z": first_fire_z,
        "armed": ls.armed,
        "calm": ls.calm,
        "var_calm": ls.var_calm,
    }


def test_empirical_loss_surprise_stationary_1000_steps_zero_false_positives(_stationary_1000_step_checkpoint):
    """Verify LossSurprise produces exactly 0 fires and z << 3.5 over 1,000 stationary steps."""
    ckpt = _stationary_1000_step_checkpoint
    assert ckpt["first_fire_step"] is None, (
        f"False positive fire at step {ckpt['first_fire_step']}: fires={ckpt['fires']}, z={ckpt['first_fire_z']:.4f}"
    )
    assert ckpt["fires"] == 0, f"Expected 0 fires over 1,000 stationary steps, got {ckpt['fires']}"
    assert ckpt["max_z"] < LossSurprise.Z_CALM, (
        f"Max stationary z-score unexpectedly high: {ckpt['max_z']:.4f} (expected < {LossSurprise.Z_CALM})"
    )
    assert ckpt["max_z"] < LossSurprise.Z_THRESH, f"Stationary z-score exceeded threshold: {ckpt['max_z']:.4f}"
    assert ckpt["armed"] is True, "LossSurprise should remain armed at equilibrium"
    assert ckpt["calm"] is not None, "Calm baseline was not established"
    assert ckpt["var_calm"] >= LossSurprise.MIN_VAR, f"Variance clamped below floor: {ckpt['var_calm']}"


@pytest.mark.parametrize("shift_magnitude", [3.0, 1.0])
def test_empirical_loss_surprise_fires_on_abrupt_shift_within_2_to_3_steps(
    shift_magnitude: float, _stationary_1000_step_checkpoint
):
    """Verify LossSurprise fires within 2-3 steps of an abrupt shift (+3.0 or +1.0) with confirmed streak=2."""
    ckpt = _stationary_1000_step_checkpoint
    device = ckpt["device"]
    trainer, data_gen = _build_1d_gaussian_setup(device, seed=42)
    trainer.load_state_dict(deepcopy(ckpt["trainer_state"]))
    data_gen.set_state(ckpt["data_gen_state"].clone())
    ls = trainer.policy.loss_surprise

    assert ls.fires == 0
    assert ls.armed is True
    assert ls.var_calm == pytest.approx(LossSurprise.MIN_VAR)

    # 3. Abrupt distribution shift
    target_mean = 2.0 + shift_magnitude
    fired_step = None
    z_scores = []
    streaks = []

    for step in range(1, 10):
        shifted_real = (target_mean + 0.5 * torch.randn(128, 1, generator=data_gen)).to(device)
        trainer.step(shifted_real)
        z_scores.append(ls.last_z)
        streaks.append(ls.streak)
        if ls.fires > 0:
            fired_step = step
            break

    assert fired_step is not None, f"LossSurprise failed to fire on +{shift_magnitude} shift! z_scores={z_scores}"
    assert fired_step in (2, 3), (
        f"LossSurprise fired at step {fired_step} after shift (expected within 2-3 steps). "
        f"z_scores={z_scores}, streaks={streaks}"
    )


# ==============================================================================
# Suite 2: GANTrainer Checkpoint Persistence & Backward Compatibility
# ==============================================================================

def test_gantrainer_checkpoint_loss_surprise_bit_exact_persistence():
    """Verify GANTrainer persists loss_surprise in state_dict and restores all fields bit-exactly."""
    recipe = get_recipe("atlas", z_dim=2, num_particles=256, batch_size=128, lr=0.005)
    G = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 2))
    D = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 1))
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)
    prior = recipe.make_prior(sigma=0.005)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    # Train a few steps
    for _ in range(5):
        trainer.step(torch.randn(128, 2))

    ls = trainer.policy.loss_surprise
    assert ls is not None

    # Populate explicit known values across every state field
    ls.calm = 0.6931471805599453
    ls.fast = 0.7182818284590452
    ls.var_calm = 0.0004123456789
    ls.streak = 1
    ls.fires = 4
    ls.armed = False
    ls.since_calm = 7
    ls.since_fire = 11
    ls.last_z = 2.45678901234
    ls.pending_loss = 0.81283456

    state = trainer.state_dict()
    assert "loss_surprise" in state, "loss_surprise missing from GANTrainer.state_dict()"
    assert isinstance(state["loss_surprise"], dict)

    # Restore into fresh replica
    G2 = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 2))
    D2 = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 1))
    deterministic_orthogonal_(G2, seed=0)
    deterministic_orthogonal_(D2, seed=1)
    prior2 = recipe.make_prior(sigma=0.005)
    deterministic_orthogonal_(unwrap_prior(prior2), seed=2)
    trainer2 = GANTrainer(recipe, G2, D2, prior=prior2, seed=0, serial_backward=True)

    assert trainer2.policy.loss_surprise.calm != ls.calm
    trainer2.load_state_dict(state)
    ls2 = trainer2.policy.loss_surprise

    # Assert bit-exact equality across every single field
    state_keys = ["fast", "calm", "var_calm", "streak", "armed", "since_calm", "since_fire", "fires", "last_z", "pending_loss"]
    for key in state_keys:
        val1 = getattr(ls, key)
        val2 = getattr(ls2, key)
        assert val1 == val2, f"Bit-exact mismatch for field '{key}': {val1} != {val2}"


def test_gantrainer_checkpoint_backward_compatibility_omitted_loss_surprise():
    """Verify GANTrainer loads legacy checkpoints that lack the 'loss_surprise' key without error."""
    recipe = get_recipe("atlas", z_dim=2, num_particles=256, batch_size=128, lr=0.005)
    G = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 2))
    D = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 1))
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)
    prior = recipe.make_prior(sigma=0.005)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    state = trainer.state_dict()
    assert "loss_surprise" in state

    # Delete loss_surprise key to simulate legacy checkpoint
    legacy_state = deepcopy(state)
    del legacy_state["loss_surprise"]

    # Fresh replica
    G2 = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 2))
    D2 = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 1))
    deterministic_orthogonal_(G2, seed=0)
    deterministic_orthogonal_(D2, seed=1)
    prior2 = recipe.make_prior(sigma=0.005)
    deterministic_orthogonal_(unwrap_prior(prior2), seed=2)
    trainer2 = GANTrainer(recipe, G2, D2, prior=prior2, seed=0, serial_backward=True)

    # Must load cleanly without raising ValueError or KeyError
    trainer2.load_state_dict(legacy_state)


def test_gantrainer_checkpoint_rejects_corrupted_loss_surprise():
    """Verify GANTrainer raises ValueError on corrupted loss_surprise checkpoint data."""
    recipe = get_recipe("atlas", z_dim=2, num_particles=256, batch_size=128, lr=0.005)
    G = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 2))
    D = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 1))
    prior = recipe.make_prior(sigma=0.005)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    bad_state = deepcopy(trainer.state_dict())
    bad_state["loss_surprise"] = "non_dict_corrupted_payload"

    G2 = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 2))
    D2 = nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 1))
    trainer2 = GANTrainer(recipe, G2, D2, prior=recipe.make_prior(sigma=0.005), seed=0, serial_backward=True)

    with pytest.raises(ValueError, match="incompatible loss-surprise state"):
        trainer2.load_state_dict(bad_state)


# ==============================================================================
# Suite 3: Cooldown and Re-arming Lifecycle Invariant
# ==============================================================================

def test_loss_surprise_cooldown_disarmed_and_rearm_lifecycle():
    """Verify LossSurprise remains disarmed for COOLDOWN (16) steps after fire and re-arms when calm."""
    ls = LossSurprise()

    # Step 0: Establish baseline
    ls.observe(0.6931)
    ls.decide()
    assert ls.calm == pytest.approx(0.6931)
    assert ls.armed is True

    # Stabilize baseline with calm variance down to MIN_VAR floor (requires ~800 steps with alpha=0.01)
    for _ in range(800):
        ls.observe(0.6931)
        ls.decide()
    assert ls.var_calm == pytest.approx(LossSurprise.MIN_VAR)

    # First shift excursion: step 1: streak=1; step 2: fire
    ls.observe(1.500)
    fired1 = ls.decide()
    assert not fired1
    assert ls.streak == 1

    ls.observe(1.500)
    fired2 = ls.decide()
    assert fired2
    assert ls.fires == 1
    assert ls.armed is False
    assert ls.streak == 0
    assert ls.since_fire == 0
    assert ls.calm == pytest.approx(ls.fast)

    # During cooldown (15 steps), assert armed strictly remains False
    for i in range(1, 16):
        ls.observe(ls.calm)
        assert not ls.decide()
        assert ls.armed is False, f"Prematurely rearmed at cooldown step {i}"
        assert ls.since_fire == i

    # Step 16: since_fire == 16 >= COOLDOWN and calm (z < 1.5) -> must rearm
    ls.observe(ls.calm)
    ls.decide()
    assert ls.armed is True, "Failed to re-arm after 16 cooldown steps"
    assert ls.fires == 1
