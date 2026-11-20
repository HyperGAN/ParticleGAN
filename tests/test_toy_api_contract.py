"""Fail closed at the API/metric/media boundary, independently of model quality."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from benchmarks.toy_audit.api_contract import (
    coverage, evaluation_steps, metric_observations, validate_case, validate_observation,
)


def test_shared_seed_and_constructor_isolation_for_every_provider():
    from benchmarks.toy_audit.api_contract import discover
    from benchmarks.toy_audit.reproducibility import DEFAULT_SEED, construction_rng
    assert DEFAULT_SEED == 0
    assert {case["protocol_seed"] for case in discover().values()} == {0}
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    draws = []
    for _ in range(2):
        with construction_rng(DEFAULT_SEED, "cpu"):
            draws.append((random.random(), np.random.random(), torch.randn(4)))
    assert draws[0][:2] == draws[1][:2] and torch.equal(draws[0][2], draws[1][2])
    assert random.getstate() == python and torch.equal(torch.get_rng_state(), cpu)
    restored = np.random.get_state()
    assert numpy[0] == restored[0] and np.array_equal(numpy[1], restored[1]) and numpy[2:] == restored[2:]


def observation():
    return {"metrics": {"rmse": .01}, "passed": True, "failed_bounds": [],
            "views": [{"kind": "scatter", "title": "Target and generated modes",
                       "target": np.zeros((4, 2)), "samples": np.ones((4, 2))}]}


def test_nonfinite_metric_rejects_even_claimed_success():
    value = observation()
    value["metrics"]["rmse"] = float("nan")
    result = validate_observation(value)
    assert result["passed"] is False
    assert result["failed_bounds"] == ["rmse: nonfinite"]


@pytest.mark.parametrize("field,value", [("passed", "PASS"), ("metrics", {}),
                                         ("views", []), ("failed_bounds", "rmse")])
def test_incomplete_observation_is_not_qualification(field, value):
    result = observation()
    result[field] = value
    with pytest.raises(ValueError):
        validate_observation(result)


def test_failed_observation_must_explain_which_bound_failed():
    result = observation()
    result["passed"] = False
    with pytest.raises(ValueError, match="rejected metric"):
        validate_observation(result)


def test_goal_media_requires_both_reference_and_real_output():
    result = observation()
    result["views"][0]["samples"] = np.empty((0, 2))
    with pytest.raises(ValueError, match="finite actual"):
        validate_observation(result)


def test_feature_blocks_cannot_masquerade_as_image_color_channels():
    result = observation()
    result["views"][0].update(kind="image", target=np.zeros((8, 2, 1, 32)),
                              samples=np.zeros((8, 2, 1, 32)))
    with pytest.raises(ValueError, match="pack feature blocks"):
        validate_observation(result)
    result["views"][0].update(target=np.zeros((8, 1, 2, 32)), samples=np.zeros((8, 1, 2, 32)))
    assert validate_observation(result)["passed"]


def test_nonfinite_model_output_is_a_metric_failure_with_an_actual_view():
    result = observation()
    result["views"][0]["samples"][0, 1] = float("nan")
    result = validate_observation(result)
    assert result["passed"] is False
    assert result["metrics"]["nonfinite_output_values"] == 1
    assert "nonfinite output" in result["failed_bounds"][0]


def test_coverage_does_not_drop_old_questions_or_architecture_controls():
    cases = {"new-query": {"id": "new-query", "legacy_ids": ["old-a"]},
             "architecture-control": {"id": "architecture-control", "legacy_ids": ["old-a"]}}
    result = coverage(cases, ["old-a", "old-b"])
    assert result["missing"] == ["old-b"]
    assert result["mapping"]["old-a"] == ["architecture-control", "new-query"]


def test_evaluation_boundaries_include_initial_and_terminal_states_once():
    assert evaluation_steps(3, 9) == [0, 1, 2, 3]
    assert evaluation_steps(16, 5) == [0, 4, 8, 12, 16]


def test_metric_cadence_preserves_image_declaration_and_accepts_list_bounds():
    assert metric_observations({"default_steps": 600, "thresholds": {"observations": 24}}) == 24
    assert metric_observations({"default_steps": 600, "thresholds": ["rmse <= 0.1"]}) == 24
    assert metric_observations({"default_steps": 8, "thresholds": ["finite"]}) == 8


def test_source_only_description_cannot_register_as_runnable_case():
    case = {"id": "new-query", "title": "A query", "goal": "Recover both modes",
            "kind": "vector", "scope": "Public API variant", "legacy_ids": ["old-a"],
            "default_steps": 16, "batch_size": 32, "eval_samples": 256,
            "thresholds": {"coverage_min": 2}, "sampling": "Public served samples"}
    assert validate_case(case) == case
    incomplete = deepcopy(case)
    del incomplete["thresholds"]
    with pytest.raises(ValueError, match="pass/fail bounds"):
        validate_case(incomplete)


def test_executable_inventory_retains_every_historical_question_and_pr231():
    from benchmarks.toy_audit import api_contract, api_run
    cases = api_contract.discover()
    ledger = api_run.inventory(cases)
    assert ledger["coverage"]["required_questions"] == 110
    assert ledger["coverage"]["missing"] == []
    # Frozen176, original ring/scalar callers and revised ring smoke, plus the
    # N-tier cases api-zero-tanh-smoke, api-zero-tanh-rail-escape, api-grid100-mlp16,
    # api-grid100-res16-tanh, api-ring16-bigan-continuous, api-misgan-block-recovery
    # and api-misgan-block-neural-imputer.
    gan_stretch = {
        "api-alice-spiral-bigan", "api-cgan-overlapping-classes",
        "api-cramer-energy-overlap", "api-cyclegan-unpaired-constellation",
        "api-grid100-res16-scaled-tanh",
        "api-infogan-anisotropic-modes", "api-misgan-mcar-p50-recovery",
        "api-misgan-mcar-p80-anchored", "api-pacgan-unequal-mass", "api-pacgan8-grid100",
    }
    assert gan_stretch <= cases.keys()
    historical = {name: case for name, case in cases.items()
                  if case["provider"] not in api_contract.MODEL_PROVIDERS and name not in gan_stretch}
    assert len(historical) == 186
    assert len(cases) == 205  # Eight model cases, scoped MisGAN and ten GAN stretch cases.
    assert all(case["evaluation_observations"] >= case.get("terminal_observations", 5)
               for case in cases.values())
