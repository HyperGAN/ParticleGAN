"""Public-trainer execution and fail-closed grading of precision stretch toys."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.views import grade_result, load_tasks, load_view

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {"recipe_preset": "bispf_optimized", "claim_contract": {"sampling_law": "task_declared"}}


def task(format):
    return json.loads((ROOT / f"configs/forge/tasks/precision_gaussian_{format}.json").read_text())


def test_precision_is_optional_and_preserves_every_original_required_assignment():
    view = load_view(ROOT, "discriminator_stability")
    v16 = json.loads((ROOT / view["previous_revision"]["archive"]).read_text())
    v15 = json.loads((ROOT / v16["previous_revision"]["archive"]).read_text())
    v14 = json.loads((ROOT / v15["previous_revision"]["archive"]).read_text())
    assert [a for a in view["assignments"] if not a["task"].startswith("interp_")] == v16["assignments"]
