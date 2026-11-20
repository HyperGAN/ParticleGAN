"""Reference recipes delegate only explicitly declared, task-owned fields."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.forge.adapters import _context, adapter_preflight
from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.contracts import validate_idea
from experiments.forge.hostprofiles import _validate_candidate_identity, _validate_task
from experiments.forge.planning import candidate_revision_for, resolve_idea
from experiments.forge.taskrecipes import (adaptation_receipt, bind_task_candidate,
                                          delegated_fields)


ROOT = Path(__file__).resolve().parents[1]
NAME = "release07-gan-v3-task-adapted-v1"


def card():
    return json.loads((ROOT / f"configs/forge/ideas/{NAME}.json").read_text())


def task(name):
    value = json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())
    # Explicit legacy cohort: archived task-owned policy and host penalties.
    value["execution"].pop("prior_contract", None)
    value["execution"]["prior"]["learnable"] = True
    return value


def test_original_24_task_adaptations_stay_ready_and_original_reference_stays_blocked():
    original = resolve_idea(ROOT, "release07-gan-v3-mog-v1", through_tier=3)
    adapted = resolve_idea(ROOT, NAME, through_tier=3)
    assert not adapted["preflight_blockers"]
    # The recorded integration covered these 24 hosts. New required smoke hosts
    # retain their own compatibility checks; they do not inherit that readiness.
    additions = {"gaussian1d_smoke", "gaussian1d_stability", "ring16_acquisition_6400", "five_word_joint_smoke",
                 "five_word_joint_hold", "clockfree_audit_measurement_v1", "zero_tanh_smoke",
                 "zero_tanh_rail_escape", "grid100_mlp16", "grid100_res16_tanh", "misgan_block_recovery",
                 "misgan_block_neural_imputer", "ring16_bigan_continuous"}
    original_view = json.loads((ROOT / "reports/forge/precision-stretch/initial-view-v14.json").read_text())
    original_hosts = set(adapted["tasks"]) & {a["task"] for a in original_view["assignments"]}
    original_hosts -= additions
    original_hosts -= {name for name, task in adapted["tasks"].items() if task["adapter"] == "public_api_model_toy"}
    original_hosts -= {name for name, task in adapted["tasks"].items() if "linear_precision" in task["execution"]}
    assert additions <= set(adapted["tasks"]) and len(original_hosts) == 24
    assert all(not adapted["tasks"][name]["preflight_blockers"] for name in original_hosts)
    assert sum(bool(original["tasks"][name]["preflight_blockers"]) for name in original_hosts) == 21
    assert original["tasks"]["two_pole"]["preflight_blockers"]
    assert card()["recipe_overrides"] == original["candidate"]["recipe_overrides"]
    # Reservation/dispatch validation must accept exactly the same bindings.
    for name in original_hosts:
        _validate_task(adapted["tasks"][name], adapted["candidate"], ROOT)


def test_behavior_binding_preserves_release_training_knobs_and_records_objective_scope():
    candidate = card()
    frozen = deepcopy(candidate)
    host = task("two_pole")
    components = BehaviorComponents({"candidate": candidate, "protocol": {"seed": 0}}, host)
    recipe = components.recipe
    assert (recipe.reg_arm, recipe.reg_coeff, recipe.reg_kappa, recipe.betas) == (
        "b_cap", 6., 1.25, (0., .99))
    assert (recipe.lr, recipe.prior_lr_mult, recipe.total_steps) == (.00425, 2., 80)
    assert recipe.prior_reg == 0.
    assert not recipe.direct_particle_gain and recipe.output_noise_std == 0.
    assert components.host_adaptation["delegated_reference_values"]["prior_reg"] == .05
    assert components.receipt()["host_adaptation"] == adaptation_receipt(candidate, host)
    assert candidate == frozen


def test_scalar_binding_retains_prior_spread_and_uses_actual_task_resources():
    host = task("vector_two_broad")
    spec = host["execution"]["host_definition"]
    resources = {"num_particles": spec["particles"], "z_dim": spec["z_dim"], "batch_size": spec["batch"]}
    context = _context({"candidate": card(), "protocol": {"seed": 0}}, host, "cpu", resources)
    assert context.recipe.prior_reg == .05
    assert context.recipe.total_steps == host["execution"]["steps"]
    assert all(getattr(context.recipe, field) == value for field, value in resources.items())
    assert set(context.receipt()["host_adaptation"]["delegated_reference_values"]) == set(resources) | {"total_steps"}


def test_ae_keeps_supported_routing_settings_and_delegates_its_objective():
    bound = bind_task_candidate(card(), task("ae_gan_hold"))
    assert "prior_reg" not in bound["recipe_overrides"]
    assert bound["recipe_overrides"]["routing_temperature"] == .25
    assert bound["recipe_overrides"]["distance_reduction"] == "sum"
    assert not adapter_preflight(task("ae_gan_hold"), card())


@pytest.mark.parametrize("field", ["reg_coeff", "lr", "output_noise_std", "continuous_policy", "made_up"])
def test_adaptation_cannot_remove_training_or_policy_mechanisms(field):
    candidate = card()
    candidate["recipe_overrides"][field] = 0.
    candidate["host_adaptation"]["recipe_fields"].append(field)
    with pytest.raises(ValueError, match="task-owned"):
        validate_idea(candidate)
    assert adapter_preflight(task("two_pole"), candidate)


@pytest.mark.parametrize("mutation", ["missing_reference", "duplicate", "bad_version", "extra_key"])
def test_malformed_delegations_are_rejected(mutation):
    candidate = card()
    if mutation == "missing_reference":
        candidate["recipe_overrides"].pop("total_steps")
    elif mutation == "duplicate":
        candidate["host_adaptation"]["recipe_fields"].append("total_steps")
    elif mutation == "bad_version":
        candidate["host_adaptation"]["schema_version"] = True
    else:
        candidate["host_adaptation"]["extra"] = "ignored"
    with pytest.raises(ValueError):
        delegated_fields(candidate, task("two_pole"))


def test_unlisted_host_conflicts_remain_blocked():
    candidate = card()
    candidate["host_adaptation"]["recipe_fields"].remove("batch_size")
    assert any("batch_size" in reason for reason in adapter_preflight(task("two_pole"), candidate))
    assert any("batch_size" in reason for reason in adapter_preflight(task("vector_two_broad"), candidate))


def test_adaptation_changes_identity_without_changing_reference_recipe():
    request = resolve_idea(ROOT, NAME)
    candidate = request["candidate"]
    without = deepcopy(candidate)
    without.pop("host_adaptation")
    assert candidate_revision_for(request["source"]["digest"], candidate) != candidate_revision_for(
        request["source"]["digest"], without)
    assert candidate["resolved_recipe"] == without["resolved_recipe"]
    changed = deepcopy(request)
    changed["candidate"]["host_adaptation"]["recipe_fields"].remove("total_steps")
    with pytest.raises(ValueError, match="scientific identity"):
        _validate_candidate_identity(changed)
