"""The passive RHBI archive must preserve the complete public training state."""
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.planning import load_idea
from experiments.forge.rhbi_trajectory import TerminalArchive, TrajectoryArchive
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("candidate", ["rhbi-v12", "rhbi-v12-feature-transport", "rhbi-v12-feature-raw-self", "rhbi-v12-raw-assignment"])
@pytest.mark.parametrize("archive_type", [TrajectoryArchive, TerminalArchive])
def test_archive_preserves_updates(tmp_path, candidate, archive_type):
    card = json.loads((ROOT / "configs/forge/tasks/vector_unequal_mass.json").read_text())

    def build():
        ctx = task_formulation_context(load_idea(ROOT, candidate), card, {"seed": 0}, device="cpu", root=ROOT)
        return ctx, ctx.build_trainer(*build_vector_models(ctx, resolve_vector_spec(card)), max_steps=2)

    ordinary, trainer = build()
    observed, instrumented = build()
    archive = archive_type(instrumented, observed, {"execution": {"steps": 2}, "evaluation": {}}, tmp_path / "trace")
    with pytest.raises(ValueError, match="Incomplete"):
        archive.receipt()
    rng = torch.Generator().manual_seed(0)
    for step in (1, 2):
        real = torch.randn(128, 2, generator=rng)
        trainer.step(real, collect_stats=True)
        archive.step(real)
        state = torch.load(archive.directory / f"state-{step:04d}.pt", weights_only=True)
        assert state_digest(state) == state_digest(ordinary.state_dict()) == state_digest(observed.state_dict())
    verify_artifacts(archive.directory, archive.receipt()["artifacts"])
    if archive_type is TerminalArchive:
        assert archive.receipt()["state_steps"] == [0, 1, 2]
        return
    plans = torch.load(archive.directory / "plans-0002.pt", weights_only=True)
    assert plans["plan"].shape == (2, 128, 129)
    assert plans["generator_rows"].shape == (2, 128)
    assert plans["direct_tv"].gt(0).any().item() == (candidate in {"rhbi-v12-feature-transport", "rhbi-v12-feature-raw-self"})
