"""Infrastructure contracts for field ownership; these tests launch no training."""
from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from particlegan import Recipe
from experiments.forge import boundaries
from experiments.forge.boundaries import (
    RECIPE_FIELD_OWNERS, TUNABLE_FIELDS, ownership_receipt, recipe_field_owner,
    task_owned_recipe_fields, validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _task(name="vector_two_broad"):
    value = json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())
    # Explicit legacy cohort: archived task-owned policy and host penalties.
    value["execution"].pop("prior_contract", None)
    value["execution"]["prior"]["learnable"] = True
    return value


def _recipe(task, **overrides):
    execution = task["execution"]
    host = execution.get("host_definition", {})
    prior = execution["prior"]
    options = dict(prior_kind="mog" if prior["kind"] == "mog" else "particles",
                   standardize=prior["standardize"], sigma_rel=0., total_steps=execution["steps"])
    if task["adapter"] == "transfer_vector":
        options.update(z_dim=host["z_dim"], num_particles=host["particles"], batch_size=host["batch"])
    return asdict(Recipe.from_dict({**options, **overrides}))


def test_every_public_recipe_field_has_one_explicit_owner(monkeypatch):
    validate_registry()
    assert set(RECIPE_FIELD_OWNERS) == {field.name for field in fields(Recipe)}
    assert TUNABLE_FIELDS <= {name for name in RECIPE_FIELD_OWNERS if recipe_field_owner(name) == "hyperparameter"}
    original = fields(Recipe)
    monkeypatch.setattr(boundaries, "fields", lambda _: (*original, SimpleNamespace(name="new_public_field")))
    with pytest.raises(ValueError, match="new_public_field"):
        validate_registry()


@pytest.mark.parametrize("task_id,field,value", [
    ("two_pole", "prior_reg", .2),
    ("vector_two_broad", "batch_size", 4),
    ("vector_two_broad", "serve_average", 4),
])
def test_typed_extensions_cannot_replace_host_conditions_or_unbound_policy(monkeypatch, task_id, field, value):
    from experiments.forge import api
    registry = api.CapabilityRegistry()
    registry.register_extension(api.ExtensionSpec("probe", "int" if type(value) is int else "float",
        "recipe", field, "Ownership boundary probe"))
    monkeypatch.setattr(api, "default_registry", lambda: registry)
    with pytest.raises(ValueError, match="owned by the frozen host|recipe/extension conflict|policy-aware task|capability unavailable"):
        api.task_formulation_context({"extensions": {"probe": value}}, _task(task_id))


def test_effective_extension_value_has_registered_provenance(monkeypatch):
    from experiments.forge import api
    registry = api.CapabilityRegistry()
    registry.register_extension(api.ExtensionSpec("step_size", "float", "recipe", "lr", "Learning rate probe"))
    monkeypatch.setattr(api, "default_registry", lambda: registry)
    context = api.task_formulation_context({"extensions": {"step_size": .007}}, _task())
    field = context.receipt()["field_ownership"]["recipe_fields"]["lr"]
    assert field["value"] == .007 and field["owner"] == "hyperparameter"
    assert field["source"] == "registered candidate.extensions binding for Recipe.lr"


def test_unknown_fields_cannot_get_a_default_owner():
    with pytest.raises(ValueError, match="unknown public Recipe field"):
        recipe_field_owner("optimizer_magic")


def test_r1r2_repair_controls_remain_recipe_owned_with_explicit_host_applicability():
    from experiments.forge.taskrecipes import ADAPTABLE_FIELDS
    names = {"reg_real_weight", "reg_fake_weight", "joint_reconstruction_weight"}
    assert all(recipe_field_owner(name) == "technique" for name in names)
    assert not names & TUNABLE_FIELDS
    assert not names & ADAPTABLE_FIELDS
    word = _task("five_word_joint_smoke")
    options = dict(reg_arm="a_r1r2", reg_real_weight=1., reg_fake_weight=0.,
                   joint_reconstruction_weight=1.)
    word_receipt = ownership_receipt({}, word,
        _recipe(word, num_particles=5, z_dim=2, batch_size=256,
                total_steps=word["execution"]["original_schedule_horizon"], **options))["recipe_fields"]
    assert word_receipt["reg_fake_weight"]["value"] == 0.
    assert word_receipt["joint_reconstruction_weight"]["value"] == 1.
    assert all(word_receipt[name]["status"] == "effective" for name in names)
    vector = _task()
    vector_receipt = ownership_receipt({}, vector, _recipe(vector, **options))["recipe_fields"]
    assert vector_receipt["joint_reconstruction_weight"]["value"] is None
    assert vector_receipt["joint_reconstruction_weight"]["status"] == "not_applicable"
    assert vector_receipt["joint_reconstruction_weight"]["reference_recipe_value"] == 1.
    ae = _task("model_ae_reconstruction")
    ae_receipt = ownership_receipt({}, ae,
        _recipe(ae, num_particles=64, z_dim=2, batch_size=128, **options))["recipe_fields"]
    assert all(ae_receipt[name]["status"] == "not_applicable" for name in names)


def test_prior_regularization_ownership_depends_on_the_task_path():
    scalar, behavior = _task(), _task("two_pole")
    assert recipe_field_owner("prior_reg", scalar) == "hyperparameter"
    assert recipe_field_owner("prior_reg", behavior) == "task"
    assert "prior_reg" not in task_owned_recipe_fields(scalar)
    assert "prior_reg" in task_owned_recipe_fields(behavior)
    assert recipe_field_owner("routing_temperature", _task("ae_gan_hold")) == "hyperparameter"
    assert recipe_field_owner("routing_temperature", behavior) == "task"


def test_receipt_reports_active_settings_and_labels_legacy_constants():
    task = _task()
    candidate = {"recipe_overrides": {"lr": .00425, "prior_reg": .123},
                 "prior": {"kind": "particle_cloud", "sigma": 0.}, "initializer": "supplied"}
    recipe = _recipe(task, lr=.00425, prior_reg=.123)
    receipt = ownership_receipt(candidate, task, recipe, {"seed": 0, "rng": {"derivation": "forge-rng-v1"}},
                                initializer="deterministic_orthogonal")
    actual = receipt["recipe_fields"]
    assert actual["lr"] == dict(value=.00425, owner="hyperparameter", source="candidate.recipe_overrides.lr", status="effective")
    assert actual["prior_reg"]["value"] == .123
    assert actual["num_particles"]["source"] == "task.execution.host_definition.particles"
    assert receipt["inactive_legacy_host_fields"]["lr"]["value"] == .001
    assert receipt["inactive_legacy_host_fields"]["prior_reg"]["value"] == .05
    assert "lr" not in receipt["task_contract"]["host"]["value"]["definition"]
    assert receipt["task_contract"]["prior"]["code_path"] == "MoGParticlePrior"
    assert receipt["task_contract"]["prior"]["value"]["sigma"] == .025
    assert actual["sigma_rel"]["value"] == 0.
    assert receipt["reference_declarations"]["prior"]["kind"] == "particle_cloud"
    assert receipt["task_contract"]["initialization"]["value"] == "deterministic_orthogonal"
    assert receipt["protocol"]["seed"]["owner"] == "protocol"
    assert actual["name"]["status"] == "metadata"


@pytest.mark.parametrize("field,value,match", [
    ("prior_kind", "particles", "prior"), ("sigma_rel", .2, "prior"),
    ("standardize", True, "prior"), ("num_particles", 17, "resource"),
    ("batch_size", 17, "resource"), ("z_dim", 17, "resource"),
    ("total_steps", 17, "schedule horizon"),
])
def test_receipt_rejects_hidden_overrides_of_effective_task_fields(field, value, match):
    task = _task()
    recipe = _recipe(task)
    recipe[field] = value
    with pytest.raises(ValueError, match=match):
        ownership_receipt({}, task, recipe)


def test_receipt_rejects_incomplete_effective_recipe():
    with pytest.raises(ValueError, match="complete effective public Recipe"):
        ownership_receipt({}, _task(), {"lr": .001})


def test_receipt_rejects_actual_initialization_that_disagrees_with_the_task():
    task = _task()
    with pytest.raises(ValueError, match="task-owned initialization"):
        ownership_receipt({}, task, _recipe(task), initializer="supplied")


def test_receipt_records_explicit_native_architecture_and_recipe_resources():
    task = _task("grid100")
    receipt = ownership_receipt({}, task, _recipe(task))
    host = receipt["task_contract"]["host"]
    assert host["owner"] == "task"
    assert host["value"]["model"] == task["execution"]["model"]
    assert host["value"]["recipe_resources"] == task["execution"]["resources"]


def test_behavioral_objective_reference_is_not_claimed_as_effective():
    task = _task("two_pole")
    candidate = {"recipe_overrides": {"prior_reg": .123, "z_dim": 2},
                 "host_adaptation": {"schema_version": 1, "recipe_fields": ["prior_reg", "z_dim"]}}
    receipt = ownership_receipt(candidate, task, _recipe(task), initializer="deterministic_orthogonal")
    prior_reg = receipt["recipe_fields"]["prior_reg"]
    assert prior_reg == dict(value=None, owner="task", source="frozen behavioral host objective/component source",
                             status="host_owned", reference_recipe_value=0.)
    assert receipt["recipe_fields"]["num_particles"]["value"] is None
    assert receipt["delegated_reference_values"] == {"prior_reg": .123, "z_dim": 2}
    assert receipt["task_contract"]["initialization"]["fixed_initialization"] == task["execution"]["fixed_initialization"]
    assert receipt["task_contract"]["prior"]["code_path"] is None
    assert receipt["task_contract"]["prior"]["declared_code_path"] == "ParticlePrior"


def test_ae_encoder_recipe_resources_are_effective_and_objectives_stay_host_owned():
    task = _task("ae_gan_hold")
    receipt = ownership_receipt({}, task, _recipe(task, encoder_mode="ae", num_particles=12, z_dim=2, batch_size=64))
    for field, value in {"encoder_mode": "ae", "num_particles": 12, "z_dim": 2, "batch_size": 64}.items():
        assert receipt["recipe_fields"][field]["value"] == value
        assert receipt["recipe_fields"][field]["owner"] == "task"
        assert receipt["recipe_fields"][field]["status"] == "effective"
    assert receipt["recipe_fields"]["reconstruction_weight"]["value"] is None


def test_external_task_budget_is_separate_from_schedule_free_technique():
    task = _task()
    recipe = _recipe(task, total_steps=None, continuous_policy="dv12")
    receipt = ownership_receipt({}, task, recipe)
    assert receipt["recipe_fields"]["total_steps"]["owner"] == "technique"
    assert receipt["recipe_fields"]["total_steps"]["value"] is None
    assert receipt["task_contract"]["budget"]["value"]["steps"] == 1200


def test_receipt_is_deterministic_json_and_does_not_mutate_inputs():
    task, candidate = _task(), {"recipe_overrides": {"betas": [0., .9]}}
    recipe = _recipe(task, betas=(0., .9))
    inputs = deepcopy((candidate, task, recipe))
    first = ownership_receipt(candidate, task, recipe)
    second = ownership_receipt(candidate, task, dict(reversed(list(recipe.items()))))
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
    assert first["recipe_fields"]["betas"]["value"] == [0., .9]
    first["task_contract"]["evaluation"]["value"]["thresholds"].clear()
    assert (candidate, task, recipe) == inputs


def test_recipe_normalization_and_registered_extension_sources_are_recorded():
    task = _task()
    candidate = {"recipe_overrides": {"reg_arm": "a_r1r2"}, "extensions": {"numeric_extension": .001}}
    recipe = _recipe(task, reg_arm="a_r1r2", lr=.001)
    receipt = ownership_receipt(candidate, task, recipe, extension_recipe_bindings={"lr": .001})
    assert receipt["recipe_fields"]["lr"]["source"] == "registered candidate.extensions binding for Recipe.lr"
    assert receipt["recipe_fields"]["critic_formulation"]["value"] == "k3p"
    assert receipt["recipe_fields"]["critic_formulation"]["source"] == "public Recipe normalization of candidate.recipe_overrides.reg_arm"


def test_legacy_candidate_resource_and_initialization_bindings_are_labelled():
    task = _task("grid100")
    task["execution"].pop("initializer", None)
    task["execution"].pop("resources", None)
    receipt = ownership_receipt({"recipe_overrides": {"z_dim": 3}, "initializer": "supplied"},
                                task, _recipe(task, z_dim=3), initializer="supplied")
    assert receipt["recipe_fields"]["z_dim"] == dict(value=3, owner="technique", status="legacy_reference",
                                                       source="candidate.recipe_overrides.z_dim")
    assert receipt["task_contract"]["initialization"] == dict(value="supplied", owner="technique",
                                                               source="legacy candidate/API initializer fallback")


def test_declarative_recipe_field_specs_and_kinetic_transport_tunability():
    from dataclasses import fields
    from particlegan import Recipe
    from experiments.forge.boundaries import RECIPE_FIELD_SPECS, TUNABLE_FIELDS, recipe_field_owner

    assert set(RECIPE_FIELD_SPECS) == {f.name for f in fields(Recipe)}
    for name in ("kinetic_transport_weight", "kinetic_transport_local_weight"):
        assert name in TUNABLE_FIELDS
        assert recipe_field_owner(name) == "hyperparameter"
        spec = RECIPE_FIELD_SPECS[name]
        assert spec.owner == "hyperparameter" and spec.tunable is True and spec.default == 0.0
