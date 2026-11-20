"""Applied-recipe packet compatibility; no model construction or training."""
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.views import load_tasks
from particlegan.recipe_compat import without_default_additions


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "gram_fp64_guarded_v1"


def test_selected_default_projection_preserves_older_full_recipe_fields():
    values = dict(optimizer_dense_profile="default", optimizer_svd_backend="native",
                  constraint_geometry_mode="none", lr=.01)
    assert without_default_additions(values, fields=("optimizer_dense_profile",)) == {
        "optimizer_svd_backend": "native", "constraint_geometry_mode": "none", "lr": .01}
    assert values["optimizer_dense_profile"] == "default"
    assert without_default_additions(values) == {"lr": .01}


@pytest.mark.parametrize("profile", ["default", PROFILE])
def test_component_applied_packet_omits_only_later_inactive_defaults(profile):
    request = dict(protocol=dict(seed=0), candidate=dict(recipe_preset="bcap", recipe_overrides=dict(
        optimizer_family="dualnorm", optimizer_smoothing=.001, optimizer_momentum=0.,
        optimizer_svd_backend="native", optimizer_dense_profile=profile)))
    components = BehaviorComponents(request, load_tasks(ROOT)["two_pole"], device="cpu")
    expected = asdict(components.recipe)
    for name in ("masked_mixture_weight", "masked_transport_weight", "imputer_component_weight"):
        assert expected.pop(name) == 0.
    if profile == "default":
        expected.pop("optimizer_dense_profile")
    for field in ("reg_real_weight", "reg_fake_weight", "joint_reconstruction_weight", "conditional_energy_weight", "conditional_energy_samples"):
        expected.pop(field)
    applied = components.receipt()
    assert applied["recipe"] == expected
    assert set(applied["field_ownership"]["recipe_fields"]) == set(expected)
    if profile != "default":
        assert applied["recipe"]["optimizer_dense_profile"] == PROFILE
        assert applied["field_ownership"]["recipe_fields"]["optimizer_dense_profile"]["value"] == PROFILE
    assert applied["recipe"]["optimizer_svd_backend"] == "native"
    assert applied["recipe"]["constraint_geometry_mode"] == "none"
