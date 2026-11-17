from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.toy_audit.conditional_spread import BAGS, BAG_SIZE, WIDTH, score_samples, target_signals
from experiments.forge.adapters import adapter_preflight
from experiments.forge.conditional_spread import Game, context_for
from experiments.forge.state import state_digest
from experiments.forge.views import load_view


ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT / "configs/forge/tasks/wide_conditional_spread.json").read_text())
EMPF = json.loads((ROOT / "configs/forge/ideas/empf-v8.json").read_text())


def test_scorer_controls():
    samples = np.zeros((2, BAGS * BAG_SIZE, WIDTH), dtype=np.float32)
    samples[:, :, :2] = target_signals(101).reshape(2, -1, 2)
    good = score_samples(samples)
    assert all(good[key] <= bound for key, _, bound in TASK["evaluation"]["thresholds"])
    for damaged in (samples[::-1], samples * .5):
        bad = score_samples(damaged)
        assert bad["variability_coverage"] < .01 and bad["worst_interval_score_ratio"] > 5
        assert bad["max_scaled_mean_error"] < .15
    samples[:, :, -1] = 1
    assert score_samples(samples)["max_group_padding_energy"] == 1
    samples[0, 0, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_samples(samples)


def test_stretch_assignment_preserves_previous_assignments():
    view = load_view(ROOT, "discriminator_stability")
    old = json.loads((ROOT / view["previous_revision"]["archive"]).read_text())
    *previous, assignment = view["assignments"]
    assert previous == old["assignments"]
    assert (assignment["task"], assignment["qualification_tier"], assignment["importance"]) == (
        TASK["id"], 2, "stretch")


def test_public_game_replays_and_observation_preserves_training():
    assert not adapter_preflight(TASK, EMPF, root=ROOT)
    incompatible = deepcopy(EMPF)
    incompatible["recipe_overrides"]["reg_every"] = 2
    assert adapter_preflight(TASK, incompatible, root=ROOT)

    def new_game():
        return Game(context_for(EMPF, TASK, {"seed": 0}, "cpu"), TASK)

    with torch.autograd.set_multithreading_enabled(False):
        first = new_game()
        first.step()
        snapshot = deepcopy(first.state_dict())
        first.samples()
        assert state_digest(first.state_dict()) == state_digest(snapshot)
        first.step()
        second = new_game()
        second.load_state_dict(snapshot)
        second.step()
        assert first.opt_d.record.calls == second.opt_d.record.observed_steps == 2
        assert state_digest(first.state_dict()) == state_digest(second.state_dict())
