"""Checkpoint recording must preserve public VAE observations and exact replay."""
import torch
import pytest

from benchmarks.toy_audit.api_model_vae import build_case
from experiments.forge.observation_checkpoints import checkpoint_observer
from experiments.forge.state import state_digest


def test_recording_is_read_only_and_checkpoint_replays(tmp_path):
    fixture = build_case("api-model-vae-hard", max_steps=2)
    fixture.step()
    before = state_digest(fixture.state_dict())
    expected = fixture.observe(n=32, seed=10000)
    receipts = []
    fixture.observe = checkpoint_observer(fixture, tmp_path / "states", receipts)
    actual = fixture.observe(n=32, seed=10000)
    assert actual["metrics"] == expected["metrics"]
    assert state_digest(fixture.state_dict()) == before
    saved = torch.load(receipts[0]["path"], weights_only=False)
    assert state_digest(saved) == before
    replay = build_case("api-model-vae-hard", max_steps=2)
    replay.load_state_dict(saved)
    fixture.step()
    replay.step()
    assert state_digest(fixture.state_dict()) == state_digest(replay.state_dict())


def test_duplicate_observation_refuses_to_replace_state(tmp_path):
    fixture = build_case("api-model-vae-hard", max_steps=1)
    receipts = []
    fixture.observe = checkpoint_observer(fixture, tmp_path / "states", receipts)
    fixture.observe(n=16)
    with pytest.raises(FileExistsError):
        fixture.observe(n=16)
    assert len(receipts) == 1


def test_vae_transport_requires_its_real_public_consumer():
    import json
    from pathlib import Path
    from copy import deepcopy
    from experiments.forge.model_toy_adapter import preflight
    root = Path(__file__).resolve().parents[1]
    card = json.loads((root / 'configs/forge/ideas/bispf-vaegan-transport-v1.json').read_text())
    for mode in ('hard', 'categorical'):
        task = json.loads((root / f'configs/forge/tasks/model_vae_{mode}.json').read_text())
        assert preflight(task, card, root=root) == []
        for field, invalid in (('transport_consumer', None), ('host', 'api-model-ddgan-gaussian')):
            broken = deepcopy(task)
            broken['execution'][field] = invalid
            assert preflight(broken, card, root=root)
