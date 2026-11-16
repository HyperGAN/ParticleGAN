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
    assert [a for a in v16["assignments"] if a["task"] != "grid100_res16_scaled_tanh"] == v15["assignments"]
    assert [a for a in v15["assignments"] if not a["task"].startswith("precision_gaussian_")] == v14["assignments"]
    assert [a for a in view["assignments"] if a["importance"] == "required"] == [
        a for a in v14["assignments"] if a["importance"] == "required"]
    assert {a["task"]: a["importance"] for a in view["assignments"] if a["task"].startswith("precision_")} == {
        "precision_gaussian_fp32": "diagnostic", "precision_gaussian_mxfp8": "stretch", "precision_gaussian_nvfp4": "stretch"}
    tasks = load_tasks(ROOT)
    first = tasks["precision_gaussian_fp32"]
    for format in ("mxfp8", "nvfp4"):
        other = tasks["precision_gaussian_" + format]
        assert first["evaluation"] == other["evaluation"]
        assert first["resources"] == other["resources"]
        assert {k: v for k, v in first["execution"].items() if k != "linear_precision"} == {
            k: v for k, v in other["execution"].items() if k != "linear_precision"}


def test_preflight_refuses_an_unimplemented_precision_policy():
    value = task("nvfp4")
    assert adapter_preflight(value, CANDIDATE, root=ROOT) == []
    value["execution"]["linear_precision"]["stochastic_gradient_rounding"] = False
    assert any("precision differs" in reason for reason in adapter_preflight(value, CANDIDATE, root=ROOT))


@pytest.mark.parametrize("policy", [None, "nvfp4", {}, {"format": "int4"}])
def test_malformed_precision_never_falls_back_to_unquantized_training(policy):
    from experiments.forge.precision import grade
    value = task("nvfp4")
    value["execution"]["linear_precision"] = policy
    assert adapter_preflight(value, CANDIDATE, root=ROOT)
    assert grade(value, {})[0] == "INVALID"


def test_new_stretch_tasks_stay_visible_without_invalidating_archived_coverage():
    from ops.refresh_forge_readouts import check_retained_diagnostic_integrity
    coverage = check_retained_diagnostic_integrity(ROOT)
    assert coverage["current"] is True
    assert coverage["missing_by_family"] == {}
    assert coverage["unmeasured_current_tasks"] == [
        "interp_img_patch_shift",
        "interp_saddle_barycentric",
        "interp_sparse_arc",
        "precision_gaussian_mxfp8",
        "precision_gaussian_nvfp4",
    ]


def test_published_preview_keeps_its_exact_original_source_and_media():
    import hashlib
    import subprocess
    saved = json.loads((ROOT / "reports/forge/precision-stretch/software/validation.json").read_text())
    for path, expected in saved["execution_scientific_sources_sha256"].items():
        original = subprocess.check_output(["git", "show", saved["source_commit"] + ":" + path], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == expected
    for prefix in saved["prefixes"]:
        assert prefix["quality_status"] == "INCOMPLETE"
        assert hashlib.sha256((ROOT / prefix["gif"]["path"]).read_bytes()).hexdigest() == prefix["gif"]["sha256"]


@pytest.mark.parametrize("family", ["atlas", "bcap-pure", "bispf", "e22", "k3p", "ka2", "r1r2", "release07-gan-v3"])
@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_every_selected_family_exercises_training_or_its_declared_blocker(family, format):
    from benchmarks.toy_audit.gaussian1d_quality import sample_target
    from experiments.forge.api import task_formulation_context
    from experiments.forge.planning import load_idea
    from experiments.forge.precision import prepare
    from experiments.forge.trainer_families import current_family_candidates
    from experiments.forge.vectorprofiles import build_vector_models
    value = task(format)
    choice = next(row for row in current_family_candidates(ROOT) if row["family"] == family)
    candidate = load_idea(ROOT, choice["candidate_id"])
    blockers = adapter_preflight(value, candidate, root=ROOT)
    if blockers:
        assert family == "e22" and any("policy" in reason for reason in blockers)
        return
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        context = task_formulation_context(candidate, value, {"seed": 0}, device="cpu", root=ROOT)
        spec = value["execution"]["host_definition"]
        g, d = build_vector_models(context, spec)
        prepare(context, value, g, d)
        trainer = context.build_trainer(g, d, max_steps=value["execution"]["steps"])
        rng = context.streams.generator("data", component="target", purpose="training", device="cpu")
        for step in range(3):
            trainer.step(sample_target(spec, spec["batch"], rng, step))
        assert trainer.completed_steps == 3
        assert all(torch.isfinite(p).all() for model in (g, d, trainer.prior) for p in model.parameters())
        assert torch.isfinite(trainer.sample(128, generator=torch.Generator().manual_seed(0), output_noise=False)).all()
    finally:
        torch.set_num_threads(old)


def test_adding_unmeasured_stretch_tasks_preserves_source_bound_display_records():
    from reports.forge.regenerate_technique_inventory import (
        _formulaic_display_payload,
        _selected_recipe_binary_task_statuses,
    )
    publication = json.loads((ROOT / "reports/forge/technique-inventory.json").read_text())
    before = deepcopy(publication["rows"])
    display = _formulaic_display_payload(publication, ROOT)
    statuses = _selected_recipe_binary_task_statuses(publication, ROOT)
    assert display and statuses
    assert display["qualification_input"] is False
    assert display["qualification_reuse"] is False
    from experiments.forge.family_reports import _selected_cohorts
    cohorts = {family["id"]: cohort for family, cohort in _selected_cohorts(publication["family_progress"])}
    for family, observed in display["families"].items():
        for name in observed["records"]:
            assert statuses[family][name] == cohorts[family]["tasks"][name]["status"]
        for name in ("precision_gaussian_nvfp4", "precision_gaussian_mxfp8"):
            assert name not in observed["records"]
            assert statuses[family][name] not in {"PASS", "FAIL"}
    assert publication["rows"] == before


def test_public_trainer_precision_prefixes_have_gifs_matching_starts_and_data(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    receipts = []
    try:
        for format in ("fp32", "mxfp8", "nvfp4"):
            original = task(format)
            value = deepcopy(original)
            # Explicit short software cohort, not a new quality measurement or
            # a replacement for the registered 1,000-update test.
            value["id"] += "-software-prefix"
            value["execution"]["steps"] = 3
            value["evaluation"]["observations"] = 3
            request = {"candidate": CANDIDATE, "protocol": {"seed": 0}, "tasks": {value["id"]: value}}
            raw = run_task(request, {"task_id": value["id"]}, tmp_path / format, "cpu")
            receipts.append(raw)
            assert raw["cost"]["completed_steps"] == 3
            assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
            assert raw["evidence"]["guards"]["all_finite"] is True
            assert (tmp_path / format / "goal.gif").stat().st_size > 100
            assert grade_result(original, raw)["gate_status"] == "INCOMPLETE"
            missing = deepcopy(raw)
            del missing["evidence"]["linear_precision"]
            assert grade_result(original, missing)["gate_status"] == "INCOMPLETE"
            wrong = deepcopy(raw)
            wrong["evidence"]["linear_precision"]["contract"]["format"] = "int4"
            assert grade_result(original, wrong)["gate_status"] == "INVALID"
        assert len({json.dumps(r["evidence"]["initial_parameters"], sort_keys=True) for r in receipts}) == 1
        assert len({r["evidence"]["batch_sequence_sha256"] for r in receipts}) == 1
    finally:
        torch.set_num_threads(old)
