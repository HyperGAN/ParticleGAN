"""Unit tests for LossSurprise abrupt distribution shift detector and SettledReopenGuard arming."""
from dataclasses import fields
import math

import pytest
import torch
import torch.nn as nn

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.continuous import LossSurprise, OptimizerSurprise, SettledReopenGuard
from particlegan.init import deterministic_orthogonal_
from particlegan.priors import unwrap_prior


def test_loss_surprise_quiet_at_equilibrium():
    """Verify that 500 steps of stationary critic loss produces 0 fires and streak stays 0."""
    detector = LossSurprise()
    torch.manual_seed(42)
    # Stationary Gaussian loss around 0.5 with small standard deviation 0.05
    losses = (0.5 + 0.05 * torch.randn(500)).tolist()

    for step, loss in enumerate(losses):
        detector.observe(loss)
        fire = detector.decide(step)
        assert not fire, f"Premature fire at step {step}"
        assert detector.streak == 0, f"Streak non-zero at step {step}: {detector.streak}"
        assert abs(detector.last_z) < detector.Z_THRESH, f"Exceeded threshold at equilibrium: z = {detector.last_z}"

    assert detector.fires == 0
    assert detector.armed is True


def test_loss_surprise_fires_on_abrupt_shift():
    """Verify that an abrupt loss jump (z > 3.0) fires on exactly the second consecutive step."""
    detector = LossSurprise()
    guard = SettledReopenGuard()

    # 100 calm steps to establish baseline L ~ 0.5, var ~ 0.0025 (sigma ~ 0.05)
    torch.manual_seed(42)
    for step in range(100):
        loss = 0.5 + 0.02 * math.sin(step)
        detector.observe(loss)
        fired = detector.decide(step, guard=guard)
        assert not fired

    assert detector.armed is True
    assert detector.streak == 0
    calm_baseline = detector.calm
    sigma_baseline = math.sqrt(detector.var_calm)

    # Step 100: Abrupt jump to 2.5 (z >> 3.0)
    # First step of jump: streak becomes 1, fire should be False (needs streak=2)
    jump_loss = calm_baseline + 8.0 * sigma_baseline
    detector.observe(jump_loss)
    fired_1 = detector.decide(100, guard=guard)
    assert not fired_1, "Should not fire on step 1 of abrupt shift (requires confirmation streak = 2)"
    assert detector.streak == 1
    assert detector.is_surprised is True
    assert detector.last_z > detector.Z_THRESH

    # Step 101: Second step of jump: streak becomes 2 -> fires!
    detector.observe(jump_loss)
    fired_2 = detector.decide(101, guard=guard)
    assert fired_2 is True, "Must fire on step 2 of abrupt shift"
    assert detector.fires == 1
    assert detector.armed is False
    assert detector.streak == 0
    assert detector.since_fire == 0


def test_loss_surprise_cooldown_rearm():
    """Verify that detector disarms after firing and re-arms only after calm and 16 cooldown steps."""
    detector = LossSurprise()

    # Establish baseline
    for step in range(50):
        detector.observe(0.5)
        detector.decide(step)

    # 2-step jump to trigger fire
    detector.observe(5.0)
    detector.decide(50)
    detector.observe(5.0)
    fired = detector.decide(51)
    assert fired is True
    assert detector.armed is False
    assert detector.since_fire == 0

    # Steps 52 to 66: cooldown in progress (since_fire < 16)
    for step in range(52, 51 + detector.COOLDOWN):
        detector.observe(0.5)
        detector.decide(step)
        assert detector.armed is False, f"Rearmed too early at step {step} (since_fire={detector.since_fire})"

    # Step 67: since_fire == 16 and calm is True -> re-arms!
    detector.observe(0.5)
    detector.decide(51 + detector.COOLDOWN + 1)
    assert detector.armed is True, "Should be rearmed after 16 cooldown steps"


def test_loss_surprise_arms_settled_guard_at_s1():
    """Verify SettledReopenGuard permits excursion when loss_surprised=True even if network is empty (s=1.0)."""
    guard = SettledReopenGuard()
    detector = OptimizerSurprise()
    guard.observe_epoch(False, detector)

    step = 50
    ratio = 3.0  # high ratio > CALM (1.25)
    calm_level = 1.25
    empty_network = {}  # network scale s = 1.0 everywhere

    # Without loss surprise: excursion network witness is empty -> returns False
    permitted_without = guard.observe(ratio, calm_level, step, empty_network, loss_surprised=False)
    assert permitted_without is False, "Empty network without loss surprise must NOT permit excursion"

    # With loss surprise: returns True even with empty network
    permitted_with = guard.observe(ratio, calm_level, step, empty_network, loss_surprised=True)
    assert permitted_with is True, "loss_surprised=True must permit excursion when network scale s = 1.0"


def test_loss_surprise_state_dict_roundtrip():
    """Verify bit-exact state serialization across state_dict and load_state_dict."""
    detector = LossSurprise()
    for step in range(30):
        detector.observe(0.4 + 0.05 * step)
        detector.decide(step)
    detector.observe(1.2)  # pending observation

    state = detector.state_dict()
    assert isinstance(state, dict)
    assert state["pending_loss"] == 1.2
    assert state["fast"] is not None
    assert state["calm"] is not None

    # Restore into fresh instance
    restored = LossSurprise()
    restored.load_state_dict(state)

    for k, v in state.items():
        assert getattr(restored, k) == v, f"Mismatch in field {k}: {getattr(restored, k)} vs {v}"

    # Invalid state raises ValueError
    with pytest.raises(ValueError, match="incompatible loss-surprise state"):
        restored.load_state_dict("not-a-dict")


def test_zero_recipe_bloat():
    """Verify zero fields added to Recipe dataclass in particlegan/recipes.py."""
    field_names = {f.name for f in fields(Recipe)}
    # Assert loss_surprise is NOT a recipe field
    assert "loss_surprise" not in field_names
    assert "loss_surprise_threshold" not in field_names
    assert "loss_surprise_cooldown" not in field_names
    # Recipe field count invariant
    # Independently declared objectives may extend the schema; loss-surprise
    # still contributes no public recipe fields.
    additions = ('reg_real_weight', 'reg_fake_weight', 'joint_reconstruction_weight', 'conditional_energy_weight', 'conditional_energy_samples', 'masked_mixture_weight', 'masked_transport_weight', 'imputer_component_weight', 'rhbi_weight', 'rhbi_rho', 'rhbi_tau', 'rhbi_eps', 'rhbi_iters', 'rhbi_c_max', 'rhbi_beta_ell', 'rhbi_nn_quantile', 'rhbi_critic_weight', 'rhbi_self_coupling', 'rhbi_relativistic_weight', 'rhbi_lambda_floor', 'rhbi_gamma_gauge', 'rhbi_rapidity_screening', 'rhbi_scale_log_odds', 'rhbi_conditional_fibre', 'rhbi_akt_horizon', 'rhbi_akt_mode', 'rhbi_slack_gain', 'rhbi_chord_weight', 'rhbi_thermal_tau', 'rhbi_thermal_shock_mu', 'rhbi_settle_beta', 'rhbi_settle_mode', 'rhbi_fibre_rel_blend', 'rhbi_step_damp', 'rhbi_cond_ot_weight', 'rhbi_cond_ot_mode', 'rhbi_simplex_weight', 'rhbi_cvx_mu_gain', 'rhbi_simplex_chart_weight', 'rhbi_tail_spill_gain', 'rhbi_simplex_macro_settle')
    assert set(additions) <= field_names
    assert len(field_names - set(additions)) == 118


def test_policy_loss_surprise_integration():
    """Verify UpdatePolicy instantiates LossSurprise, serializes it, and executes without error."""
    recipe = get_recipe("atlas", z_dim=2, num_particles=32, batch_size=16)
    G = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 2))
    D = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1))
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior(sigma=0.005)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    assert trainer.policy.loss_surprise is not None
    assert isinstance(trainer.policy.loss_surprise, LossSurprise)

    # Checkpoint roundtrip on policy preserves loss_surprise
    policy_state = trainer.policy.state_dict()
    assert "loss_surprise" in policy_state

    # Stepping updates loss_surprise
    real = torch.randn(16, 2)
    trainer.step(real)
    assert trainer.policy.loss_surprise.pending_loss is not None
    trainer.step(real)
    assert trainer.policy.loss_surprise.calm is not None

    # Restore policy state
    saved_state = trainer.policy.state_dict()
    trainer.policy.load_state_dict(saved_state)
    assert trainer.policy.loss_surprise.calm == saved_state["loss_surprise"]["calm"]
