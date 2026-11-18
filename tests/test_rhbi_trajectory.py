"""The passive RHBI archive must preserve the complete public training state."""
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.planning import load_idea
from experiments.forge.rhbi_trajectory import TrajectoryArchive
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("candidate", ["rhbi-v12", "rhbi-v12-feature-transport"])
def test_archive_observes_actual_plans_without_changing_updates(tmp_path, candidate):
    card = json.loads((ROOT / "configs/forge/tasks/vector_unequal_mass.json").read_text())

    def build():
        ctx = task_formulation_context(load_idea(ROOT, candidate), card, {"seed": 0}, device="cpu", root=ROOT)
        return ctx, ctx.build_trainer(*build_vector_models(ctx, resolve_vector_spec(card)), max_steps=2)

    ordinary, trainer = build()
    observed, instrumented = build()
    archive = TrajectoryArchive(instrumented, observed, {"execution": {"steps": 2}}, tmp_path / "trace")
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
    plans = torch.load(archive.directory / "plans-0002.pt", weights_only=True)
    assert plans["plan"].shape == (2, 128, 129)
    assert plans["generator_rows"].shape == (2, 128)
    assert plans["direct_tv"].gt(0).any().item() == (candidate == "rhbi-v12-feature-transport")
