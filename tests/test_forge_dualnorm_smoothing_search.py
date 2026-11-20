"""Read-only search admission and identity contracts; no learned gate claims."""
from dataclasses import asdict
from pathlib import Path

import pytest

from test_forge_configuration_search import checkout, spec
from experiments.forge import configuration_search as search
from experiments.forge import search_space as spaces
from experiments.forge.api import task_formulation_context
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.techniques import recipe_field_active, technique_signature
from particlegan import get_recipe


def enable_base(checkout, *, smoothing=1e-4):
    path = checkout / "configs/forge/ideas/base.json"
    base = read_json(path)
    base.update(recipe_preset="bcap", recipe_overrides={"optimizer_family": "dualnorm", "optimizer_smoothing": smoothing,
                                                       "lr": .012, "d_lr_mult": 1.5, "prior_lr_mult": 2.5})
    atomic_json(path, base)
    return base


def test_positive_scale_grid_is_active_owned_and_reusable(checkout, spec):
    base = enable_base(checkout)
    spec["grid"] = {"optimizer_smoothing": [1e-5, 1e-4, 1e-3]}
    plan = search.plan_search(checkout, checkout / "runs", spec)
    assert len(plan["trials"]) == 3
    assert plan["declared_worst_case_seconds"] == 30
    assert recipe_field_owner("optimizer_smoothing") == "hyperparameter"
    expected = technique_signature(get_recipe("bcap", loss="relativistic",
                                              optimizer_smoothing=1e-4, optimizer_convolution="none",
                                              constraint_geometry_mode="none"))
    task = read_json(checkout / "configs/forge/tasks/t1.json")
    for card, settings in search._declarations(checkout, spec):
        assert technique_signature(card["resolved_configuration_recipe"]) == expected
        context = task_formulation_context(card, task, root=checkout)
        assert context.recipe.optimizer_smoothing == settings["optimizer_smoothing"]
        assert recipe_field_active("optimizer_smoothing", context.recipe, task=task)
        search.validate_configuration_declaration(card)
    assert base["recipe_overrides"]["optimizer_smoothing"] == 1e-4


@pytest.mark.parametrize("base_scale, choices", [(0., [0., 1e-4]), (1e-4, [0., 1e-4])])
def test_zero_boundary_rejected_before_materialization(checkout, spec, base_scale, choices):
    enable_base(checkout, smoothing=base_scale)
    spec["grid"] = {"optimizer_smoothing": choices}
    with pytest.raises(ValueError, match="smoothed_dualnorm"):
        search.materialize_search(checkout, spec)
    assert not (checkout / "configs/forge/configurations").exists()


@pytest.mark.parametrize("scale", [True, None, -1., float("nan"), float("inf")])
def test_invalid_scale_rejected_before_writing(checkout, spec, scale):
    enable_base(checkout)
    spec["grid"] = {"optimizer_smoothing": [scale]}
    with pytest.raises(ValueError):
        search.materialize_search(checkout, spec)
    assert not (checkout / "configs/forge/configurations").exists()


def test_other_families_do_not_claim_an_active_smoothing_axis():
    for family in ("adam", "formulation", "sgda", "dualnorm_D_only", "particle_rownorm_only"):
        # Model families now inherit the DualNorm BCAP trainer; ka2 keeps the plain formulation optimizer.
        recipe = (get_recipe("gan", trainer="ka2", optimizer_family=family) if family == "formulation"
                  else get_recipe("bcap_adam", optimizer_family=family))
        assert not recipe_field_active("optimizer_smoothing", recipe)


def test_zero_default_preserves_recipe_signature_and_configuration_identity(checkout):
    base = enable_base(checkout, smoothing=0.)
    legacy = get_recipe("bcap", loss="relativistic", optimizer_smoothing=0., optimizer_convolution="none")
    recipe = asdict(legacy)
    historical = {name: value for name, value in recipe.items() if name != "optimizer_smoothing"}
    assert technique_signature(recipe) == technique_signature(historical)
    assert search.recipe_identity_fields(recipe) == search.recipe_identity_fields(historical)
    assert search.configuration_id(base, resolved_recipe=recipe) == search.configuration_id(base, resolved_recipe=historical)
    assert "optimizer_smoothing" not in legacy.to_dict()


def test_categorical_space_explicitly_compares_unsmoothed_and_smoothed_bases(checkout):
    base = enable_base(checkout, smoothing=0.)
    smooth = {**base, "id": "smooth", "recipe_overrides": {**base["recipe_overrides"], "optimizer_smoothing": 1e-4}}
    atomic_json(checkout / "configs/forge/ideas/smooth.json", smooth)
    definition = {
        "schema": spaces.SPACE_SCHEMA, "id": "smooth-space",
        "hypothesis": "Compare separately declared mechanisms with positive strength choices",
        "protocol": "screening", "view": "stability", "execution_backend": "cpu",
        "tuning_through_tier": 1, "samples": 3,
        "campaign": {"id": "smooth-space", "budget_seconds": 30, "candidate_budget_seconds": 10},
        "candidates": [
            {"base_candidate": "base", "trainer_family": "family",
             "parameters": {"lr": {"kind": "literal", "value": .012}}},
            {"base_candidate": "smooth", "trainer_family": "family",
             "parameters": {"optimizer_smoothing": {"kind": "choice", "values": [1e-5, 1e-4]}}},
        ],
    }
    manifest = spaces.compile_space(checkout, checkout / "runs", definition)
    assert manifest["population"] == 3
    assert len(manifest["searches"]) == 2
    assert manifest["default_adoption"] is False
    assert manifest["declared_worst_case_seconds"] == 30
    for spec in manifest["searches"]:
        for card, _ in search._declarations(checkout, spec):
            recipe = card["resolved_configuration_recipe"]
            assert (recipe["lr"], recipe["d_lr_mult"], recipe["prior_lr_mult"]) == (.012, 1.5, 2.5)


def test_committed_opt_in_example_plans_all_current_tier1_bindings_without_spend(tmp_path):
    root = Path(__file__).resolve().parents[1]
    example_spec = search._load_spec(root, "bcap-dualnorm-smoothing-tier1-v3")
    selected = (root / "configs/forge/selections/family-current-v1.json").read_bytes()
    plan = search.plan_search(root, tmp_path / "queue", example_spec)
    assert len(plan["trials"]) == 3
    assert plan["declared_worst_case_seconds"] <= example_spec["campaign"]["budget_seconds"]
    base = read_json(root / "configs/forge/ideas/bcap-dualnorm-smoothed-v1.json")
    tasks = [read_json(path) for path in (root / "configs/forge/tasks").glob("*.json")]
    view = read_json(root / "configs/forge/views/discriminator_stability.json")
    required = {row["task"] for row in view["assignments"]
                if row["qualification_tier"] == 1 and row["importance"] == "required"}
    for task in tasks:
        if task["id"] in required:
            context = task_formulation_context(base, task, root=root)
            assert context.recipe.optimizer_smoothing == 1e-4
            assert recipe_field_active("optimizer_smoothing", context.recipe, task=task)
    assert (root / "configs/forge/selections/family-current-v1.json").read_bytes() == selected
    assert not (tmp_path / "queue").exists()
