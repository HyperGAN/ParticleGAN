"""Model-mode task binding, honest partial grading and checkpoint refusals."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.model_toy_adapter import ModelToyExecutionError, preflight, run, validate_task
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result
from recipe_entrypoint_compat import model_source_blockers

ROOT = Path(__file__).resolve().parents[1]


def candidate():
    return json.loads((ROOT / "configs/forge/ideas/bispf-optimized-quintic-v1.json").read_text())


def task(name="model_vae_hard"):
    return json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())


def test_objective_adaptation_preserves_the_global_trainer_and_prior_policy():
    for path in sorted((ROOT / "configs/forge/tasks").glob("model_*.json")):
        value = json.loads(path.read_text())
        # Changed restoration syntax is a new source cohort, even though the
        # selected recipe below is unchanged. Never rebind a frozen task hash.
        assert preflight(value, candidate(), root=ROOT) == model_source_blockers(ROOT, value)
        context = task_formulation_context(candidate(), value, {"seed": 0}, device="cpu")
        assert context.recipe.optimizer_convolution_rule == "quintic"
        assert context.recipe.lr == .012
        assert context.recipe.prior_regularizer == "sbnp"
        assert context.recipe.prior_reg == .02
        assert context.recipe.prior_update == "learned"
        assert context.recipe.encoder_mode == value["execution"]["recipe_fields"]["encoder_mode"]


def test_modified_gates_or_unbound_policies_are_refused_before_spend():
    value = task()
    value["evaluation"]["thresholds"][0][2] = 99
    with pytest.raises(ValueError, match="bounds differ"):
        validate_task(value)
    changed = candidate()
    changed["recipe_overrides"]["input_noise_std"] = .1
    assert preflight(task("model_ddgan_gaussian_chain"), changed)
    value = task()
    value["execution"]["recipe_fields"]["prior_regularizer"] = "vicreg"
    with pytest.raises(ValueError, match="prior policy"):
        validate_task(value)


@pytest.mark.parametrize("status,completed", [("ERROR", 0), ("INCOMPLETE", 1), ("BLOCKED", 0)])
def test_provider_failure_before_audit_preserves_original_cause_and_receipt(tmp_path, monkeypatch, status, completed):
    from benchmarks.toy_audit import api_run
    diagnostic = "API execution or metric error: FileNotFoundError: missing frozen provider declaration"
    receipt = dict(status=status, completed_updates=completed, failed_bounds=[diagnostic])
    def fail_before_audit(case, output, **kwargs):
        output.mkdir(parents=True)
        (output / "receipt.json").write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(api_run, "run_case", fail_before_audit)
    with pytest.raises(ModelToyExecutionError) as error:
        run({"candidate": candidate(), "protocol": {"seed": 0}}, task(), tmp_path, "cpu", execution_limit=2)
    message = str(error.value)
    assert diagnostic in message and f"{status} after {completed} updates" in message
    assert str(tmp_path / "api/receipt.json") in message
    assert json.loads((tmp_path / "api/receipt.json").read_text()) == receipt
    assert "'audit'" not in message


@pytest.mark.parametrize("metadata", [{"sigma_rel": .5, "standardize": False},
                                      {"sigma_rel": 0., "standardize": 0}])
def test_mog_policy_metadata_is_checked_before_mutation(metadata):
    from benchmarks.toy_audit.api_model_ae import build_case
    owner = build_case("api-model-ae-reconstruction", max_steps=2)
    owner.step()
    before = state_digest(owner.state_dict())
    saved = deepcopy(owner.state_dict())
    saved["policy"]["models"]["prior"]["_extra_state"] = metadata
    with pytest.raises(ValueError, match="extra state"):
        owner.load_state_dict(saved)
    assert state_digest(owner.state_dict()) == before


@pytest.mark.parametrize("name", ["model_ae_reconstruction", "model_ae_gan_multimodal", "model_vae_hard",
                                  "model_vae_categorical", "model_ddgan_gaussian_chain", "model_ddgan_bimodal_chain",
                                  "model_ae_ddgan_gaussian_chain", "model_ae_ddgan_bimodal_chain"])
def test_short_forge_execution_retains_real_media_and_cannot_qualify(tmp_path, name):
    value = task(name)
    result = run({"candidate": candidate(), "protocol": {"seed": 0}}, value, tmp_path, "cpu", execution_limit=2)
    assert result["scope"] == "integration_demo_only"
    assert result["cost"]["completed_steps"] == 2
    assert result["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert result["evidence"]["guards"]["all_finite"] is True
    assert result["evidence"]["guards"]["hooks_exercised"] is True
    expected = dict(generator=2, encoder=0 if name.startswith("model_ddgan") else 2, prior=2,
                    discriminator=0 if name == "model_ae_reconstruction" else 2)
    assert result["evidence"]["guards"]["optimizer_updates"] == expected
    assert (tmp_path / "api/goal.gif").read_bytes().startswith(b"GIF")
    assert grade_result(value, result)["gate_status"] != "PASS"
