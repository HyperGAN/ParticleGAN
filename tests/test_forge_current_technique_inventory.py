"""One current table can be rebuilt from committed, source-bound evidence."""
from copy import deepcopy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("forge_current_inventory", ROOT / "reports/forge/regenerate_technique_inventory.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)
# The board's diagnostics note may cite the historical Atlas replay in this one
# scoped sentence (experiments/forge/family_reports.py), never as a score.
ATLAS_HISTORY_NOTE = "Atlas's original 19/19 CUDA passes remain valid under their original task and sampling contracts."


def test_publication_prefers_recorded_then_canonical_declarations(monkeypatch):
    from experiments.forge import planning, trainer_families
    paths = [Path('alias.json'), Path('canonical.json'), Path('recorded.json')]
    original = lambda root: paths
    monkeypatch.setattr(planning, 'declaration_paths', original)
    monkeypatch.setattr(trainer_families, 'load_families',
                        lambda root: {'family': {'canonical_candidate': 'canonical'}})
    with publication._prefer_recorded_and_canonical_declarations(ROOT, {'recorded'}):
        assert [p.stem for p in planning.declaration_paths(ROOT)] == ['recorded', 'canonical', 'alias']
    assert planning.declaration_paths is original


def test_publication_restores_declarations_after_error(monkeypatch):
    from experiments.forge import planning, trainer_families
    original = lambda root: []
    monkeypatch.setattr(planning, 'declaration_paths', original)
    monkeypatch.setattr(trainer_families, 'load_families', lambda root: {})
    with pytest.raises(RuntimeError):
        with publication._prefer_recorded_and_canonical_declarations(ROOT):
            raise RuntimeError('publication failed before writing')
    assert planning.declaration_paths is original


def _current_family_roster():
    pins = read_json(ROOT / CURRENT_SELECTION)["selections"]
    pinned = {pin["trainer_family"] for pin in pins}
    assert len(pinned) == len(pins)
    registry = read_json(ROOT / "configs/forge/trainer-families.json")["families"]
    families = {family["id"] for family in registry}
    unmeasured = {family["id"] for family in registry if family.get("unmeasured_display_backend")}
    report = read_json(ROOT / "reports/forge/technique-inventory.json")
    represented = {row["trainer_family"] for row in report["rows"]}
    assert pinned <= represented <= families
    assert len(represented) == len(report["rows"])
    for row in report["rows"]:
        if row["trainer_family"] not in pinned:
            assert not row["attempt_ids"] and row["qualified_tier"] == 0
            assert row["selection"]["qualified"] is False
    # Editorial original-ID configuration alternatives need not manufacture
    # another row when their scientific declaration aliases an existing one.
    # A hidden declaration-only family (inventory_visible false) that no current
    # pin selects keeps only its historical row, as current_family_candidates skips it.
    hidden = {family["id"] for family in registry if family.get("inventory_visible", True) is False}
    assert unmeasured - hidden <= represented
    historical = {row["trainer_family"] for row in report.get("historical_family_rows", [])}
    assert (unmeasured & hidden) - pinned <= historical
    return represented


def _register(root, manifest, name, source, *, finished=None, passed=0):
    attempt = "attempt-" + source if finished else None
    row = {"candidate_id": name, "candidate_revision": "revision-" + source,
           "cohort": "cohort-" + source, "technique": name,
           "bindings": {"source_digest": source * 64, "prior": {"kind": "mog", "sigma": .025}},
           "runtime_cohort": {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "gpu-" + source}}},
           "tiers": {tier: {"passed": passed if tier == "1" else 0, "total": len(tasks)}
                     for tier, tasks in manifest["tier_requirements"].items()},
           "qualified_tier": 1 if passed == 3 else 0, "attempt_ids": [attempt] if attempt else [],
           "tasks": [{"task_id": "two_pole", "status": "FAIL" if attempt else "UNKNOWN"}],
           "cost": {"wall_seconds": 5.25 if attempt else None}}
    report = {key: deepcopy(manifest[key]) for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements")}
    report.update(schema_version=1, publication_scope="frozen_source", execution_backend="cuda",
                  frozen_source={"commit": "commit-" + source}, rows=[row], provenance={"qualified_receipts": {}})
    if attempt:
        proof = {"canonical_result_hash": "result-" + source, "source_digest": source * 64,
                 "original_file_sha256": {name: name + "-hash" for name in ("request", "evidence", "result")}}
        summary = {"candidate_id": name, "candidate_revision": row["candidate_revision"],
                   "certificate_validated": True, "qualification_input": False, "qualification_reuse": False,
                   "provenance": {"canonical_result_hash": proof["canonical_result_hash"], "source_digest": source * 64,
                                  "original_files": {name: {"sha256": digest} for name, digest in proof["original_file_sha256"].items()}}}
        atomic_json(root / f"reports/forge/technique-receipts/{attempt}.json", summary)
        report["provenance"]["qualified_receipts"][attempt] = proof
    report["provenance"]["input_digest"] = stable_hash(report)
    path = root / f"reports/forge/technique-evidence/{source}.json"
    atomic_json(path, report)
    entry = {"snapshot": path.relative_to(root).as_posix(), "json_sha256": file_hash(path),
             "source_commit": "commit-" + source, "candidates": {name: finished}}
    manifest["cohorts"].append(entry)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    atomic_json(root / f"configs/forge/ideas/{name}.json", {"id": name})
    return entry, report



def _as_common26_display_fixture(result):
    """Explicit pure comparison fixture; no new evidence or scientific row edits."""
    from experiments.forge.common26_comparison import FAMILIES, TASKS_BY_TIER
    value = deepcopy(result)
    indexed = {row.get("trainer_family", row["candidate_id"]): row for row in value["rows"]}
    value["rows"] = [indexed.get(family, {"trainer_family": family,
                      "candidate_id": "synthetic-" + family}) for family, _ in FAMILIES]
    value.update(view="discriminator_stability", view_revision=3,
                 tier_requirements={tier: list(tasks) for tier, tasks in TASKS_BY_TIER.items()})
    return value


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    policy = {"id": "discriminator_stability", "revision": 2,
              "assignments": [{"task": name, "importance": "required", "qualification_tier": 1}
                              for name in ("two_pole", "token", "ae")]}
    atomic_json(tmp_path / "configs/forge/views/discriminator_stability.json", policy)
    manifest = {"schema_version": 1, "view": policy["id"], "view_revision": 2,
                "policy_fingerprint": stable_hash(policy), "tier_requirements": {
                    "1": ["two_pole", "token", "ae"], "2": [f"q-{i}" for i in range(19)], "3": ["hold", "extension"]},
                "cohorts": [], "retired_publications": {}}
    _register(tmp_path, manifest, "bcap", "a", finished="2026-10-01T01:00:00+00:00", passed=3)
    _register(tmp_path, manifest, "atlas", "b")
    def unexpected(*args, **kwargs):
        pytest.fail("cached publication must not resolve live receipts or train")
    monkeypatch.setattr(publication, "write_report", unexpected)
    monkeypatch.setattr(publication, "regenerate", unexpected)
    # These cached-evidence fixtures intentionally have no executable toy
    # declarations; only the complete committed-checkout test uses real tasks.
    monkeypatch.setattr("experiments.forge.views.load_view", lambda root, view_id:
                        read_json(Path(root) / f"configs/forge/views/{view_id}.json"))
    # The cached metadata tests intentionally retain their two-row v2 science fixture.
    # Standalone common-26 controls enforce the real 12-row/v3 boundary without this adapter.
    projector, formatter = publication._common26_display_projection, publication._common26_current_markdown
    monkeypatch.setattr(publication, "_common26_display_projection",
                        lambda result, root: projector(_as_common26_display_fixture(result), root))
    monkeypatch.setattr(publication, "_common26_current_markdown",
                        lambda result, root, path: formatter(_as_common26_display_fixture(result), root, path))
    return tmp_path, manifest


def _outputs(root):
    paths = list((root / "reports/forge").glob("technique-inventory.*"))
    paths += list((root / "reports/forge/families").glob("*.md"))
    return {path: path.read_bytes() for path in paths}


def _family_text(root, result, family_id):
    from experiments.forge.family_reports import generated_pages
    return generated_pages(root, result)[root / f"reports/forge/families/{family_id}.md"]


def _family_pin(root, manifest, row):
    card = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
            "view": manifest["view"], "policy_fingerprint": manifest["policy_fingerprint"],
            "selections": [family_row_pin(row, selection_kind="historical_incumbent", reason="Recorded incumbent.")]}
    atomic_json(root / CURRENT_SELECTION, card)
    return card


def test_current_pin_retains_exact_incumbent_after_new_source_and_never_pools(evidence):
    root, manifest = evidence
    incumbent = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    _register(root, manifest, "bcap", "c", finished="2026-10-03T01:00:00+00:00", passed=1)
    result = read_json(publication.publish_current(root)["json"])
    selected = next(row for row in result["rows"] if row["candidate_id"] == "bcap")
    assert selected["candidate_revision"] == "revision-a"
    assert selected["tasks"] == incumbent["tasks"] and selected["tiers"] == incumbent["tiers"]
    assert len([row for row in result["rows"] if row["trainer_family"] == "bcap"]) == 1
    assert {row["candidate_revision"] for row in result["configuration_rows"] if row["trainer_family"] == "bcap"} == {"revision-a", "revision-c"}
    assert result["provenance"]["family_current_selection_sha256"] == file_hash(root / CURRENT_SELECTION)
    before = _outputs(root)
    assert publication.publish_current(root)["input_digest"] == result["provenance"]["input_digest"]
    assert _outputs(root) == before


@pytest.mark.parametrize("tamper", ["duplicate", "policy", "source", "recipe", "runtime", "whole_row", "standard"])
def test_invalid_current_family_pin_cannot_change_publication(evidence, tamper):
    root, manifest = evidence
    row = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    row["trainer_family"] = "bcap"
    card = _family_pin(root, manifest, row)
    publication.publish_current(root)
    before = _outputs(root)
    if tamper == "duplicate":
        card["selections"].append(deepcopy(card["selections"][0]))
    elif tamper == "policy":
        card["policy_fingerprint"] = "edited"
    elif tamper == "standard":
        card["selections"][0]["selection_kind"] = "configured_standard"
    else:
        key = {"source": "source_digest", "recipe": "recipe_sha256", "runtime": "runtime_cohort_sha256",
               "whole_row": "scientific_row_sha256"}[tamper]
        card["selections"][0][key] = "edited"
    atomic_json(root / CURRENT_SELECTION, card)
    with pytest.raises(ValueError):
        publication.publish_current(root)
    assert _outputs(root) == before


def _copy_word_diagnostics(root):
    selection = read_json(ROOT / Path("configs/forge/selections/word-joint-task-v1.json"))
    paths = [Path("configs/forge/selections/word-joint-task-v1.json")]
    paths += [Path(row[key]["path"]) for row in selection["recipes"] for key in ("receipt", "parent")]
    for relative in paths:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, root / relative)
    return selection


def _copy_completed_studies(root):
    from experiments.forge.completed_studies import REGISTRY
    registry = read_json(ROOT / REGISTRY)
    paths = [REGISTRY]
    for study in registry["studies"]:
        paths += [study[role]["path"] for role in ("report", "archive", "readout", "archive_readout")]
        paths += [pin["path"] for pin in study["media"]]
    for relative in paths:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, root / relative)
    return registry


def test_completed_studies_cannot_replace_selected_family_configs_or_history(evidence):
    root, manifest = evidence
    incumbent = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    baseline = read_json(publication.publish_current(root)["json"])
    assert "completed_api_studies" not in baseline
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    _copy_completed_studies(root)
    published = publication.publish_current(root)
    current = read_json(published["json"])
    for key, value in baseline.items():
        if key != "provenance":
            assert current[key] == value
    assert set(current) - set(baseline) == {"completed_api_studies", "original_pr223_atlas"}
    assert current["original_pr223_atlas"]["counts"] == {"PASS": 19}
    assert current["provenance"]["selected_rows_sha256"] == baseline["provenance"]["selected_rows_sha256"]
    assert current["provenance"]["input_digest"] != baseline["provenance"]["input_digest"]
    rows = current["completed_api_studies"]["rows"]
    assert rows[0]["counts"] == {"PASS": 19}
    assert rows[1]["counts"] == {"FAIL": 2}
    assert all(row["qualification_input"] is row["reuse"] is False for row in rows)
    assert all(row["counts"]["execution"] == {"FAIL": 2, "UNKNOWN": 14} for row in rows[2:])
    markdown = Path(published["report"]).read_text()
    assert "task_diagnostics" not in current
    assert "Five-word joint task diagnostics" not in markdown
    assert "<summary>Selected configurations and provenance</summary>" not in markdown
    assert "19/19" not in markdown and "2/2 hold FAIL" not in markdown
    assert "Original Atlas recipe and serving-law evidence" in _family_text(root, current, "atlas")
    assert markdown.count("| Family / selected recipe | Tier 1 requirements | Tier 2 requirements"
                          " | Tier 3 requirements | Total view requirements |") == 1
    outputs = _outputs(root)
    mtimes = {path: path.stat().st_mtime_ns for path in outputs}
    assert publication.publish_current(root) == published
    assert outputs == _outputs(root)
    assert mtimes == {path: path.stat().st_mtime_ns for path in outputs}
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest


def test_fresh_original_retest_context_preserves_whole_ordinary_selection(evidence):
    root, _ = evidence
    _copy_completed_studies(root)
    before = read_json(publication.publish_current(root)["json"])
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    helper_spec = importlib.util.spec_from_file_location(
        "private_passive_metadata_fixture", ROOT / "tests/test_forge_passive_publications.py")
    helper = importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    helper.make_cut(root)
    current = read_json(publication.publish_current(root)["json"])
    for key, value in before.items():
        if key not in {"original_pr223_atlas", "provenance"}:
            assert current[key] == value
    assert current["original_pr223_atlas"]["counts"] == {"PASS": 19}
    fresh = current["original_pr223_atlas"]["fresh_retest"]
    assert fresh["counts"] == {"PASS": 5, "NOT_RUN": 14} and fresh["status"] == "INCOMPLETE"
    assert current["provenance"]["selected_rows_sha256"] == before["provenance"]["selected_rows_sha256"]
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest
    outputs = _outputs(root)
    publication.publish_current(root)
    assert outputs == _outputs(root)


@pytest.mark.parametrize("role", ["report", "archive_readout", "gif"])
def test_completed_study_input_drift_preserves_public_outputs_and_evidence(evidence, role):
    root, _ = evidence
    registry = _copy_completed_studies(root)
    publication.publish_current(root)
    before = _outputs(root)
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    study = registry["studies"][0]
    pin = study["media"][0] if role == "gif" else study[role]
    path = root / pin["path"]
    path.write_bytes(path.read_bytes() + b"changed input")
    with pytest.raises(ValueError, match="pinned input drift"):
        publication.publish_current(root)
    assert _outputs(root) == before
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest


def test_baseline_diagnosis_is_additive_bound_navigation(evidence):
    root, _ = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    relative = Path("reports/forge/c6-baseline-debug-20261003")
    shutil.copytree(ROOT / relative, root / relative)
    current = read_json(publication.publish_current(root)["json"])
    for key, value in baseline.items():
        if key != "provenance":
            assert current[key] == value
    assert set(current) - set(baseline) == {"baseline_debugging"}
    debug = current["baseline_debugging"]
    assert debug["qualification_input"] is debug["qualification_reuse"] is False
    assert debug["files_sha256"] == {
        path.relative_to(root).as_posix(): file_hash(path) for path in (root / relative).iterdir()
    }
    before = _outputs(root)
    (root / relative / "BASELINE_SELECTION.json").unlink()
    with pytest.raises(FileNotFoundError):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_current_publication_needs_no_originals_and_is_idempotent(evidence):
    root, _ = evidence
    result = publication.publish_current(root)
    report = read_json(result["json"])
    assert result["report"] == str(root / "reports/forge/technique-inventory.md")
    assert len(report["rows"]) == 2 and report["publication_scope"] == "current_technique_inventory"
    assert report["rows"][0]["tiers"]["1"] == {"passed": 3, "total": 3}
    assert report["qualification_input"] is False and report["qualification_reuse"] is False
    assert not (root / "reports/forge/attempts").exists()
    assert list((root / "reports/forge").glob("technique-inventory.md")) == [Path(result["report"])]
    assert len(list((root / "reports/forge/families").glob("*.md"))) == 2
    outputs = _outputs(root)
    times = {path: path.stat().st_mtime_ns for path in outputs}
    assert result == publication.publish_current(root)
    assert outputs == _outputs(root)
    assert times == {path: path.stat().st_mtime_ns for path in outputs}


def test_recorded_policy_preserves_old_rows_after_view_and_declarations_change(evidence, capsys):
    root, manifest = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    archive = Path("configs/forge/view-history/discriminator_stability-v2.json")
    atomic_json(root / archive, read_json(root / "configs/forge/views/discriminator_stability.json"))
    atomic_json(root / "configs/forge/views/discriminator_stability.json", {
        "id": "discriminator_stability", "revision": 3,
        "assignments": [{"task": "new-acquisition", "qualification_tier": 1, "importance": "required"}]})
    atomic_json(root / "configs/forge/ideas/new-word-candidate.json", {"id": "new-word-candidate"})
    protected = {path: path.read_bytes() for path in (root / publication.EVIDENCE_MANIFEST.parent).glob("*.json")}
    before = _outputs(root)
    with pytest.raises(ValueError, match="current view policy differs"):
        publication.publish_current(root)
    assert _outputs(root) == before
    publication.main(["--root", str(root), "--recorded-policy", str(archive)])
    assert '"rows":2' in capsys.readouterr().out
    result = publication.publish_current(root, recorded_policy=archive)
    report = read_json(result["json"])
    assert report["recorded_policy"] == archive.as_posix()
    assert report["view_revision"] == 2
    assert report["tier_requirements"] == manifest["tier_requirements"]
    for key in ("rows", "configuration_rows", "evidence_rows"):
        assert report[key] == baseline[key]
    assert all(row["candidate_id"] != "new-word-candidate" for row in report["configuration_rows"])
    markdown = Path(result["report"]).read_text()
    assert "Recorded Forge trainer-family leaderboard" in markdown
    assert "discriminator_stability revision 2" in markdown
    assert "Tier 1: 3, Tier 2: 19, Tier 3: 2" in markdown
    assert "--recorded-policy " + archive.as_posix() in markdown
    assert not (root / "reports/forge/attempts").exists()
    assert all(path.read_bytes() == original for path, original in protected.items())
    outputs = _outputs(root)
    times = {path: path.stat().st_mtime_ns for path in outputs}
    assert result == publication.publish_current(root, recorded_policy=archive)
    assert _outputs(root) == outputs
    assert times == {path: path.stat().st_mtime_ns for path in outputs}


def test_publication_refresh_preserves_science_and_shows_new_unknown_requirement(evidence):
    root, manifest = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    policy = read_json(root / "configs/forge/views/discriminator_stability.json")
    atomic_json(root / "configs/forge/view-history/discriminator_stability-v2.json", policy)
    policy["revision"] = 3
    policy["assignments"].append(dict(task="new-scalar", qualification_tier=1, importance="required", order=3))
    atomic_json(root / "configs/forge/views/discriminator_stability.json", policy)
    protected = {path: path.read_bytes() for path in (root / publication.EVIDENCE_MANIFEST.parent).glob("*.json")}
    metadata = publication.refresh_publication(root)
    current = read_json(metadata["json"])
    for key in ("rows", "configuration_rows", "evidence_rows", "archived_evidence_rows", *publication.POLICY_FIELDS):
        assert current.get(key) == baseline.get(key)
    assert current["declared_view"]["revision"] == 3
    assert current["declared_view"]["added_required_tasks"]["1"] == ["new-scalar"]
    assert current["publication_refresh"] == dict(scientific_rows_preserved=True,
                                                 qualification_regraded=False, training_launched=False)
    markdown = Path(metadata["report"]).read_text()
    assert "family leaderboard" in markdown
    assert current["declared_view"]["added_required_tasks"]["1"] == ["new-scalar"]
    assert all(cohort["tasks"]["new-scalar"]["status"] == "UNKNOWN"
               for family in current["family_progress"]["families"] for cohort in family["cohorts"])
    assert "python reports/forge/regenerate_technique_inventory.py" in markdown
    assert all(path.read_bytes() == data for path, data in protected.items())
    before = _outputs(root)
    times = {path: path.stat().st_mtime_ns for path in before}
    assert publication.refresh_publication(root) == metadata
    assert _outputs(root) == before and times == {path: path.stat().st_mtime_ns for path in before}


def test_publication_refresh_names_held_and_removed_tasks_without_counting_them(evidence):
    root, manifest = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    policy = read_json(root / "configs/forge/views/discriminator_stability.json")
    atomic_json(root / "configs/forge/view-history/discriminator_stability-v2.json", policy)
    policy["revision"] = 3
    explanation = "reports/forge/test-dispositions-20261015/README.md"
    for assignment in policy["assignments"]:
        if assignment["task"] == "ae":
            assignment.update(importance="held", hold=dict(
                schema="forge_hold_v1", previous_importance="required", reason="no_global_recipe", blocked_by=[],
                summary="Fixture: measured and shown, not counted.", explanation=explanation + "#ae",
                reentry="Count again when one recipe passes it and every counted task."))
        elif assignment["task"] == "token":
            assignment.update(importance="diagnostic", removal=dict(
                schema="forge_count_removal_v1", previous_importance="required", reason="superseded",
                counted_by=None, summary="Fixture: replaced by a newer declaration.", explanation=explanation + "#token"))
    atomic_json(root / "configs/forge/views/discriminator_stability.json", policy)
    metadata = publication.refresh_publication(root)
    current = read_json(metadata["json"])
    for key in ("rows", "configuration_rows", "evidence_rows", "archived_evidence_rows", *publication.POLICY_FIELDS):
        assert current.get(key) == baseline.get(key)
    declared = current["declared_view"]
    assert declared["tier_requirements"] == {"1": ["two_pole"], "2": [], "3": []}
    assert declared["no_longer_required_tasks"]["1"] == ["token", "ae"]
    assert declared["held_tasks"] == {"1": ["ae"], "2": [], "3": []}
    assert declared["removed_from_count"] == {"1": [{"task": "token", "reason": "superseded", "counted_by": None}],
                                              "2": [], "3": []}
    markdown = Path(metadata["report"]).read_text()
    assert ("**1 means 1 counted view requirements, including repeated tests**, from 1 distinct test. 1 held and "
            "1 removed view assignments (1 held test and 1 removed test) are still measured") in markdown
    assert ("Counts follow the declared view revisions: discriminator_stability 3. The recorded qualification keeps "
            "discriminator_stability revision 2 until new ordinary evidence is registered") in markdown
    assert "| `ae` | held: no one recipe passes it together with the counted tests | discriminator_stability (Tier 1) |" in markdown
    assert "| `token` | removed: replaced by newer tests | discriminator_stability (Tier 1) |" in markdown
    assert "always add `--refresh-publication`" in markdown


@pytest.mark.parametrize("tamper,message", [("digest", "input digest"), ("row", "unregistered scientific row")])
def test_publication_refresh_rejects_changed_science_before_writing(evidence, tamper, message):
    root, _ = evidence
    publication.publish_current(root)
    path = root / publication.CURRENT_PREFIX.with_suffix(".json")
    report = read_json(path)
    report["rows"][0]["qualified_tier"] = 99
    if tamper == "row":
        report["provenance"].pop("input_digest")
        report["provenance"]["input_digest"] = stable_hash(report)
    atomic_json(path, report)
    before = _outputs(root)
    with pytest.raises(ValueError, match=message):
        publication.refresh_publication(root)
    assert _outputs(root) == before


def test_editorial_family_refresh_preserves_exact_pin_and_drifted_science(evidence, monkeypatch):
    from experiments.forge.trainer_families import REGISTRY, scientific_row_hash
    root, manifest = evidence
    registry = {"schema_version": 1, "families": [{"id": "bcap", "label": "BCap",
                "canonical_candidate": "bcap", "candidates": ["bcap"]}]}
    atomic_json(root / REGISTRY, registry)
    incumbent = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    baseline = read_json(publication.publish_current(root)["json"])
    pin_bytes = (root / CURRENT_SELECTION).read_bytes()
    evidence_bytes = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    # An evolved live contract must remain a coverage note, not fresh admission
    # of the old pin or a regrade of the original numerical row.
    policy = read_json(root / "configs/forge/views/discriminator_stability.json")
    atomic_json(root / "configs/forge/view-history/discriminator_stability-v2.json", policy)
    policy["revision"] = 3
    policy["assignments"].append({"task": "new-word-contract", "qualification_tier": 1, "importance": "required"})
    atomic_json(root / "configs/forge/views/discriminator_stability.json", policy)
    registry["families"][0].update(label="BCAP with K3P", tags=["critic-gradient-penalty"])
    atomic_json(root / REGISTRY, registry)
    monkeypatch.setattr("experiments.forge.trainer_families._current_pin",
                        lambda *args, **kwargs: pytest.fail("editorial refresh cannot re-admit a historical pin"))
    metadata = publication.refresh_publication(root)
    current = read_json(metadata["json"])
    assert current["rows"][0]["technique"] == "BCAP with K3P"
    assert [scientific_row_hash(row) for row in current["rows"]] == [scientific_row_hash(row) for row in baseline["rows"]]
    for key in ("configuration_rows", "evidence_rows", "historical_family_rows", "archived_evidence_rows"):
        assert current.get(key) == baseline.get(key)
    assert [row.get("selection") for row in current["rows"]] == [row.get("selection") for row in baseline["rows"]]
    assert current["trainer_families"]["bcap"] == registry["families"][0]
    assert current["trainer_family_registry"] == registry
    assert current["provenance"]["trainer_family_registry_sha256"] == file_hash(root / REGISTRY)
    assert (root / CURRENT_SELECTION).read_bytes() == pin_bytes
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == evidence_bytes
    before = _outputs(root)
    assert publication.refresh_publication(root) == metadata
    assert _outputs(root) == before


@pytest.mark.parametrize("bound_source", [True, False])
def test_legacy_editorial_refresh_requires_the_exact_bound_git_registry(evidence, monkeypatch, bound_source):
    import hashlib
    import json
    from experiments.forge.trainer_families import REGISTRY
    root, _ = evidence
    registry = {"schema_version": 1, "families": [{"id": "bcap", "label": "BCap",
                "canonical_candidate": "bcap", "candidates": ["bcap"]}]}
    atomic_json(root / REGISTRY, registry)
    original_registry = (root / REGISTRY).read_bytes()
    path = Path(publication.publish_current(root)["json"])
    previous = read_json(path)
    previous.pop("trainer_family_registry")  # Existing publications predate this metadata.
    _save_current_projection(path, previous)
    registry["families"][0]["label"] = "BCAP with K3P"
    atomic_json(root / REGISTRY, registry)
    returned = original_registry if bound_source else json.dumps(registry).encode()
    monkeypatch.setattr(publication.subprocess, "check_output", lambda *args, **kwargs: returned)
    assert hashlib.sha256(original_registry).hexdigest() == previous["provenance"]["trainer_family_registry_sha256"]
    before = _outputs(root)
    if bound_source:
        current = read_json(publication.refresh_publication(root)["json"])
        assert current["rows"][0]["technique"] == "BCAP with K3P"
        assert current["trainer_family_registry"] == registry
    else:
        with pytest.raises(ValueError, match="selection structure changed"):
            publication.refresh_publication(root)
        assert _outputs(root) == before


@pytest.mark.parametrize("change", ["members", "canonical", "search", "historical"])
def test_editorial_refresh_cannot_hide_family_selection_structure_changes(evidence, change):
    from experiments.forge.trainer_families import REGISTRY
    root, _ = evidence
    registry = {"schema_version": 1, "families": [{"id": "bcap", "label": "BCap",
                "canonical_candidate": "bcap", "candidates": ["bcap"]}]}
    atomic_json(root / REGISTRY, registry)
    publication.publish_current(root)
    if change == "members":
        registry["families"][0]["candidates"].append("new-member")
    elif change == "canonical":
        registry["families"][0]["canonical_candidate"] = "new-member"
    elif change == "search":
        registry["families"][0]["active_search_by_backend"] = {"cuda": "new-search"}
    else:
        registry["historical_families"] = [{"id": "old-bcap", "canonical_candidate": "bcap", "candidates": ["bcap"]}]
    registry["families"][0]["label"] = "Renamed family"
    atomic_json(root / REGISTRY, registry)
    before = _outputs(root)
    with pytest.raises(ValueError, match="selection structure changed"):
        publication.refresh_publication(root)
    assert _outputs(root) == before


def test_editorial_refresh_accepts_new_unmeasured_family_without_reselecting(evidence):
    from experiments.forge.trainer_families import REGISTRY, scientific_row_hash
    root, _ = evidence
    registry = {"schema_version": 1, "families": [{"id": "bcap", "label": "BCap",
                "canonical_candidate": "bcap", "candidates": ["bcap"]}]}
    atomic_json(root / REGISTRY, registry)
    baseline = read_json(publication.publish_current(root)["json"])
    evidence_bytes = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    registry["families"].append({"id": "transfer", "label": "Transfer",
                                "canonical_candidate": "new-transfer", "candidates": ["new-transfer"]})
    atomic_json(root / REGISTRY, registry)
    current = read_json(publication.refresh_publication(root)["json"])
    assert current["trainer_family_registry"] == registry
    assert [scientific_row_hash(row) for row in current["rows"]] == [scientific_row_hash(row) for row in baseline["rows"]]
    assert current["configuration_rows"] == baseline["configuration_rows"]
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == evidence_bytes
    registry["families"][-1]["active_search_by_backend"] = {"cpu": "new-search"}
    assert not publication._presentation_registry_compatible(
        baseline["trainer_family_registry"], registry, baseline)
    registry["families"][-1].pop("active_search_by_backend")
    # Binding an existing displayed candidate to that new family is forbidden.
    registry["families"][-1]["candidates"] = [baseline["rows"][0]["candidate_id"]]
    registry["families"][-1]["canonical_candidate"] = baseline["rows"][0]["candidate_id"]
    assert not publication._presentation_registry_compatible(
        baseline["trainer_family_registry"], registry, baseline)


def _unregistered_unmeasured_configuration(root):
    path = root / publication.CURRENT_PREFIX.with_suffix(".json")
    report = read_json(path)
    row = deepcopy(report["configuration_rows"][0])
    row.update(candidate_id="new-unmeasured-card", candidate_revision="unmeasured-revision", attempt_ids=[],
               qualified_tier=0, status="INCOMPLETE", cost={"wall_seconds": None, "measured_tasks": 0})
    row["tasks"] = [{"task_id": "two_pole", "status": "UNKNOWN"}]
    row["nonrequired_tasks"] = [{"task_id": "probe", "status": "BLOCKED"}]
    for tier in row["tiers"].values():
        tier["passed"] = 0
    report["configuration_rows"].append(row)
    return path, report, row


def _save_current_projection(path, report):
    report["provenance"].pop("input_digest", None)
    report["provenance"]["input_digest"] = stable_hash(report)
    atomic_json(path, report)


def test_refresh_keeps_zero_cost_unmeasured_configuration_projections(evidence):
    root, _ = evidence
    publication.publish_current(root)
    path, report, row = _unregistered_unmeasured_configuration(root)
    _save_current_projection(path, report)
    current = read_json(publication.refresh_publication(root)["json"])
    assert current["configuration_rows"][-1] == row
    assert current["rows"] == report["rows"] and current["evidence_rows"] == report["evidence_rows"]


@pytest.mark.parametrize("tamper", ["attempt", "pass", "failure", "diagnostic", "qualification", "tier_pass",
                                   "cost", "measured_tasks", "gate_status", "status"])
def test_refresh_unmeasured_exception_cannot_import_measured_or_qualified_science(evidence, tamper):
    root, _ = evidence
    publication.publish_current(root)
    path, report, row = _unregistered_unmeasured_configuration(root)
    if tamper == "attempt":
        row["attempt_ids"] = ["unregistered-attempt"]
    elif tamper in {"pass", "failure"}:
        row["tasks"][0]["status"] = "PASS" if tamper == "pass" else "FAIL"
    elif tamper == "diagnostic":
        row["nonrequired_tasks"][0]["status"] = "PASS"
    elif tamper == "qualification":
        row["qualified_tier"] = 1
    elif tamper == "tier_pass":
        row["tiers"]["1"]["passed"] = 1
    elif tamper == "cost":
        row["cost"]["wall_seconds"] = 1
    elif tamper == "measured_tasks":
        row["cost"]["measured_tasks"] = 1
    elif tamper == "gate_status":
        row["tasks"][0]["gate_status"] = "PASS"
    else:
        row["status"] = "PASS"
    _save_current_projection(path, report)
    before = _outputs(root)
    with pytest.raises(ValueError, match="unregistered measured configuration"):
        publication.refresh_publication(root)
    assert _outputs(root) == before


def test_default_cli_refreshes_later_view_without_advancing_recorded_qualification(evidence, capsys):
    root, _ = evidence
    before = read_json(publication.publish_current(root)["json"])
    path = root / "configs/forge/views/discriminator_stability.json"
    declared = read_json(path)
    atomic_json(root / "configs/forge/view-history/discriminator_stability-v2.json", declared)
    declared["revision"] = 3
    declared["assignments"].append({"task": "new-scalar", "importance": "required", "qualification_tier": 1})
    atomic_json(path, declared)
    publication.main(["--root", str(root)])
    metadata = __import__("json").loads(capsys.readouterr().out)
    refreshed = read_json(metadata["json"])
    assert refreshed["rows"] == before["rows"] and refreshed["view_revision"] == 2
    assert refreshed["declared_view"]["revision"] == 3
    assert metadata["qualification_regraded"] is metadata["training_launched"] is False
    # The declared fourth task is counted and named as not run.
    assert "[0/4 (3 not run)]" in Path(metadata["report"]).read_text()


def test_standalone_scalar_display_binds_actual_receipt_and_never_changes_qualification():
    scores = publication._standalone_api_scores(ROOT)
    score = next(score for score in scores if score["case"]["id"] == "api-gaussian1d-acquisition")
    assert score["run"]["verdict"] == "FAIL"
    assert score["run"]["final_metrics"]["cdf_ks"] == pytest.approx(.056740549361447234)
    assert score["prior"] == dict(kind="mog", sigma=.025, standardize=False, learnable=True)
    assert score["qualification_input"] is False
    current = read_json(ROOT / publication.CURRENT_PREFIX.with_suffix(".json"))
    current["standalone_api_scores"] = [score]
    before = deepcopy(current["rows"])
    markdown = publication._current_markdown(current, ROOT, ROOT / publication.CURRENT_PREFIX.with_suffix(".md"))
    family_page = _family_text(ROOT, current, score["trainer_family"])
    assert "Standalone API evidence: " + score["trainer_family"] + " · " + score["case"]["title"] in family_page
    assert "actual-training GIF" in family_page and "KS 0.05674" not in markdown
    assert __import__("os").path.relpath(ROOT / score["readout"], ROOT / "reports/forge/families") in family_page
    assert __import__("os").path.relpath(ROOT / score["gif"], ROOT / "reports/forge/families") in family_page
    assert current["rows"] == before


@pytest.mark.parametrize("tamper,message", [("publication", "publication identity"),
                                           ("gif", "GIF identity"), ("recipe", "binding mismatch")])
def test_standalone_scalar_display_rejects_broken_source_bindings(tmp_path, tamper, message):
    record_path = next((ROOT / "reports/forge/records").glob("gaussian1d-api-*.json"))
    record = read_json(record_path)
    report_dir = Path(record["source"]["path"]).parent
    for relative in (record_path.relative_to(ROOT), report_dir / "results.json",
                     report_dir / "README.md", report_dir / "goal.gif"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    if tamper == "recipe":
        record["provenance"]["recipe_sha256"] = "bad"
        atomic_json(tmp_path / record_path.relative_to(ROOT), record)
    else:
        target = tmp_path / report_dir / ("results.json" if tamper == "publication" else "goal.gif")
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match=message):
        publication._standalone_api_scores(tmp_path)


@pytest.mark.parametrize("field", ["id", "revision", "assignments"])
def test_recorded_policy_must_match_registered_identity(evidence, field):
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    policy = read_json(root / "configs/forge/views/discriminator_stability.json")
    policy[field] = "changed"
    archive = "configs/forge/view-history/changed.json"
    atomic_json(root / archive, policy)
    with pytest.raises(ValueError, match="recorded view policy differs"):
        publication.publish_current(root, recorded_policy=archive)
    assert _outputs(root) == before


def test_recorded_policy_cannot_register_source_evidence(evidence):
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    with pytest.raises(ValueError, match="cannot register new source evidence"):
        publication.publish_current(root, recorded_policy="unused.json", source_commit="commit-a")
    assert _outputs(root) == before


def test_newer_measured_revision_updates_one_row_without_pooling(evidence):
    root, manifest = evidence
    publication.publish_current(root)
    old = (root / manifest["cohorts"][0]["snapshot"]).read_bytes()
    _, newer = _register(root, manifest, "bcap", "c", finished="2026-10-02T01:00:00+00:00")
    result = read_json(publication.publish_current(root)["json"])
    rows = [row for row in result["rows"] if row["candidate_id"] == "bcap"]
    assert len(rows) == 1 and rows[0]["candidate_revision"] == "revision-c"
    assert rows[0]["tiers"]["1"]["passed"] == 0
    assert rows[0]["runtime_cohort"] == newer["rows"][0]["runtime_cohort"]
    assert rows[0]["bindings"] == newer["rows"][0]["bindings"]
    assert old == (root / manifest["cohorts"][0]["snapshot"]).read_bytes()
    assert len(list((root / "reports/forge").glob("technique-inventory.md"))) == 1


@pytest.mark.parametrize("field", ["cohort", "runtime_cohort"])
def test_equal_time_different_cohorts_cannot_silently_choose_a_winner(evidence, field):
    root, manifest = evidence
    publication.publish_current(root)
    before = _outputs(root)
    entry = deepcopy(manifest["cohorts"][0])
    report = read_json(root / entry["snapshot"])
    report["rows"][0][field] = "different-cohort" if field == "cohort" else {
        "execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "different-gpu"}}}
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    entry["snapshot"] = "reports/forge/technique-evidence/other-runtime.json"
    atomic_json(root / entry["snapshot"], report)
    entry["json_sha256"] = file_hash(root / entry["snapshot"])
    manifest["cohorts"].append(entry)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    with pytest.raises(ValueError, match="ambiguous latest recorded technique cohort"):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_declaration_fallback_requires_unanimous_backend_source():
    pins = {"first": {"execution_backend": "cuda", "source_digest": "a" * 64},
            "second": {"execution_backend": "cuda", "source_digest": "b" * 64},
            "third": {"execution_backend": "cpu", "source_digest": "c" * 64}}
    assert publication._declaration_fallback_sources(pins) == {"cpu": "c" * 64}
    pins["second"]["source_digest"] = "a" * 64
    assert publication._declaration_fallback_sources(pins) == {"cuda": "a" * 64, "cpu": "c" * 64}
    pins["second"]["source_digest"] = None
    assert publication._declaration_fallback_sources(pins) == {"cpu": "c" * 64}


@pytest.mark.parametrize("execution", ["attempt", "paid", "pass", "fail", "qualified", "tier_pass"])
def test_declaration_source_preference_cannot_hide_execution(execution):
    previous = {"status": "BLOCKED", "qualified_tier": 0, "attempt_ids": [],
                "bindings": {"source_digest": "old"}, "tasks": [{"status": "UNKNOWN"}],
                "cost": {"wall_seconds": None}, "tiers": {"1": {"passed": 0}}}
    current = deepcopy(previous)
    current["bindings"]["source_digest"] = "new"
    assert publication._declaration_source_preference(current, previous, "new") is True
    assert publication._declaration_source_preference(previous, current, "new") is False
    assert publication._declaration_source_preference(current, previous, None) is None
    assert publication._declaration_source_preference(current, previous, "other") is None
    assert publication._declaration_source_preference(current, current, "new") is None
    if execution == "attempt":
        current["attempt_ids"] = ["executed"]
    elif execution == "paid":
        current["cost"]["wall_seconds"] = 1
    elif execution in {"pass", "fail"}:
        current["tasks"][0]["status"] = execution.upper()
    elif execution == "qualified":
        current["qualified_tier"] = 1
    else:
        current["tiers"]["1"]["passed"] = 1
    assert publication._declaration_source_preference(current, previous, "new") is None
    assert publication._declaration_source_preference(previous, current, "new") is None


@pytest.mark.parametrize("historical_pin", [False, True])
@pytest.mark.parametrize("duplicate_snapshot", [False, True])
def test_current_source_declaration_fallback_retains_old_evidence_and_cached_identity(
        evidence, historical_pin, duplicate_snapshot):
    root, manifest = evidence
    atomic_json(root / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "bcap", "label": "BCAP", "candidates": ["bcap"], "canonical_candidate": "bcap"},
        {"id": "atlas", "label": "Atlas", "candidates": ["atlas"], "canonical_candidate": "atlas",
         "unmeasured_display_backend": "cuda", "unmeasured_display_reason": "Explicit declaration fixture."}]})
    incumbent = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)

    def mark_declaration(entry, report, source):
        row = report["rows"][0]
        row.update(status="BLOCKED", qualified_tier=0)
        row["bindings"]["source_digest"] = source * 64
        report["provenance"].pop("input_digest")
        report["provenance"]["input_digest"] = stable_hash(report)
        atomic_json(root / entry["snapshot"], report)
        entry["json_sha256"] = file_hash(root / entry["snapshot"])
        atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)

    old_entry = manifest["cohorts"][1]
    mark_declaration(old_entry, read_json(root / old_entry["snapshot"]), "b")
    old_bytes = (root / old_entry["snapshot"]).read_bytes()
    if duplicate_snapshot:
        duplicate = deepcopy(old_entry)
        duplicate["snapshot"] = "reports/forge/technique-evidence/b-again.json"
        (root / duplicate["snapshot"]).write_bytes(old_bytes)
        manifest["cohorts"].append(duplicate)
        atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    if historical_pin:
        card = read_json(root / CURRENT_SELECTION)
        old_row = read_json(root / old_entry["snapshot"])["rows"][0]
        old_row["trainer_family"] = "atlas"
        card["historical_selections"] = [family_row_pin(old_row, selection_kind="historical_incumbent",
                                                       reason="Retain the original declaration only.")]
        atomic_json(root / CURRENT_SELECTION, card)
    # Explicit metadata fixtures only: no model, sample, training or new gate.
    entry, report = _register(root, manifest, "atlas", "c")
    mark_declaration(entry, report, "a")
    published = publication.publish_current(root)
    result = read_json(published["json"])
    row = next(row for row in result["rows"] if row["candidate_id"] == "atlas")
    assert row["candidate_revision"] == "revision-c"
    assert row["qualified_tier"] == 0 and row["attempt_ids"] == []
    assert row["selection"]["selection_kind"] == "unmeasured_declaration"
    assert row["tiers"]["1"]["passed"] == 0 and row["cost"]["wall_seconds"] is None
    assert {row["candidate_revision"] for row in result["evidence_rows"]
            if row["candidate_id"] == "atlas"} == {"revision-b", "revision-c"}
    assert (root / old_entry["snapshot"]).read_bytes() == old_bytes
    if historical_pin:
        assert next(row for row in result["historical_family_rows"]
                    if row["candidate_id"] == "atlas")["candidate_revision"] == "revision-b"
    assert result["provenance"]["declaration_fallback_source_by_backend"] == {"cuda": "a" * 64}
    before = _outputs(root)
    assert publication.publish_current(root) == published and _outputs(root) == before


@pytest.mark.parametrize("unresolved", [False, True])
def test_new_card_is_discovered_with_truthful_unknown_or_blocked_status(evidence, monkeypatch, unresolved):
    root, manifest = evidence
    _, report = _register(root, deepcopy(manifest), "new-technique", "c")
    if unresolved:
        report["rows"][0].update(candidate_revision=None, cohort=None, bindings={})
        report["rows"][0]["tasks"][0]["status"] = "BLOCKED"
    # Leave the new card declared, but not registered as measured evidence.
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    def live(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json"))}
    monkeypatch.setattr(publication, "write_report", live)
    result = read_json(publication.publish_current(root)["json"])
    assert len(result["rows"]) == 3
    new = next(row for row in result["rows"] if row["candidate_id"] == "new-technique")
    assert new["tasks"][0]["status"] == ("BLOCKED" if unresolved else "UNKNOWN")
    assert new["tiers"]["1"] == {"passed": 0, "total": 3}
    assert not new["attempt_ids"] and "publication_key" not in new
    if unresolved:
        assert new["candidate_revision"] is None
        assert new["bindings"] == {"source_origin_commit": None}
        markdown = (root / "reports/forge/technique-inventory.md").read_text()
        assert "new-technique.md" in markdown and "[0/3 (2 not run, 1 incompatible)]" in markdown
        assert all(slot["fresh_common"]["passed"] is None
                   and slot["fresh_common"]["accepted_record"] is None
                   for slot in result["common26_display"]["rows"])


def test_explicit_frozen_regrade_registers_evidence_and_updates_canonical(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    _, report = _register(root, deepcopy(manifest), "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    atomic_json(root / "reports/forge/attempts/attempt-c/result.json", {"raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    def regrade(*args, source_commit, output_prefix, **kwargs):
        assert source_commit == "commit-c"
        assert root not in output_prefix.parents
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    result = publication.publish_current(root, source_commit="commit-c")
    current = read_json(result["json"])
    assert len(current["rows"]) == 2
    assert next(row for row in current["rows"] if row["candidate_id"] == "bcap")["candidate_revision"] == "revision-c"
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    assert len(updated["cohorts"]) == 3
    assert (root / updated["cohorts"][-1]["snapshot"]).is_file()
    assert len(list((root / "reports/forge").glob("technique-inventory.md"))) == 1
    assert publication.publish_current(root) == result


def test_source_registration_preserves_interrupted_receipt_with_missing_worker_time(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    finished = "2026-10-03T01:00:00+00:00"
    _, report = _register(root, deepcopy(manifest), "bcap", "c", finished=finished)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    originals = {
        "attempt-c-earlier": {"raw": {"finished_at": "2026-10-03T00:00:00+00:00"}},
        "attempt-c-interrupted": {"raw": {"attempt_status": "error", "reason": "worker stopped without terminal receipt"}},
        "attempt-c": {"raw": {"finished_at": finished}},
    }
    row = report["rows"][0]
    row["attempt_ids"] = list(originals)
    receipt_template = read_json(root / "reports/forge/technique-receipts/attempt-c.json")
    protected = {}
    for attempt, value in originals.items():
        result_path = root / f"reports/forge/attempts/{attempt}/result.json"
        atomic_json(result_path, value)
        summary = deepcopy(receipt_template)
        provenance = summary["provenance"]
        provenance["canonical_result_hash"] = stable_hash(value)
        provenance["original_files"]["result"]["sha256"] = file_hash(result_path)
        receipt_path = root / f"reports/forge/technique-receipts/{attempt}.json"
        atomic_json(receipt_path, summary)
        report["provenance"]["qualified_receipts"][attempt] = {
            "canonical_result_hash": provenance["canonical_result_hash"],
            "source_digest": provenance["source_digest"],
            "original_file_sha256": {name: item["sha256"] for name, item in provenance["original_files"].items()},
        }
        protected.update({path: path.read_bytes() for path in (result_path, receipt_path)})
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)

    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}

    monkeypatch.setattr(publication, "regenerate", regrade)
    result = publication.publish_current(root, source_commit="commit-c")
    registered = read_json(root / publication.EVIDENCE_MANIFEST)["cohorts"]
    assert len(registered) == 3
    entry = registered[-1]
    assert entry["candidates"] == {"bcap": finished}
    assert entry["missing_worker_finished_at"] == {"bcap": ["attempt-c-interrupted"]}
    assert read_json(root / entry["snapshot"]) == report
    current = read_json(result["json"])
    assert current["evidence_sources"][entry["json_sha256"]] == entry
    selected = next(item for item in current["rows"] if item["candidate_id"] == "bcap")
    for field in ("attempt_ids", "tasks", "tiers", "cost"):
        assert selected[field] == row[field]
    before = _outputs(root)
    assert publication.publish_current(root, source_commit="commit-c") == result
    assert publication.publish_current(root) == result
    assert len(read_json(root / publication.EVIDENCE_MANIFEST)["cohorts"]) == 3
    assert _outputs(root) == before
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize("raw", [{"attempt_status": "error"}, {"finished_at": None}])
def test_source_registration_rejects_cohort_without_any_recorded_worker_time(evidence, monkeypatch, raw):
    root, manifest = evidence
    publication.publish_current(root)
    _, report = _register(root, deepcopy(manifest), "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    result_path = root / "reports/forge/attempts/attempt-c/result.json"
    atomic_json(result_path, {"raw": raw})
    original = result_path.read_bytes()
    before = _outputs(root)
    manifest_before = (root / publication.EVIDENCE_MANIFEST).read_bytes()

    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}

    monkeypatch.setattr(publication, "regenerate", regrade)
    with pytest.raises(ValueError, match="no recorded worker completion time.*bcap"):
        publication.publish_current(root, source_commit="commit-c")
    assert result_path.read_bytes() == original
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest_before
    assert _outputs(root) == before


def _later_policy_evidence(root, manifest, monkeypatch, *, revision=3):
    policy = {"id": "discriminator_stability", "revision": revision,
              "assignments": [{"task": name, "qualification_tier": 1, "importance": "required"}
                              for name in ("two_pole", "token", "ae", "ring", "word")]}
    atomic_json(root / "configs/forge/views/discriminator_stability.json", policy)
    current = {**deepcopy(manifest), "view_revision": revision, "policy_fingerprint": stable_hash(policy),
               "tier_requirements": {**deepcopy(manifest["tier_requirements"]),
                                     "1": ["two_pole", "token", "ae", "ring", "word"]}, "cohorts": []}
    _, report = _register(root, current, "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    _, unknown = _register(root, deepcopy(current), "atlas", "d")
    unknown["rows"][0]["status"] = "BLOCKED"
    unknown["rows"][0]["tasks"][0]["status"] = "BLOCKED"
    report["rows"].extend(unknown["rows"])
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    atomic_json(root / "reports/forge/attempts/attempt-c/result.json", {
        "raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    monkeypatch.setattr(publication, "write_report", regrade)
    return report


def test_explicit_policy_advance_preserves_original_cohorts_without_new_credit(evidence, monkeypatch, capsys):
    root, manifest = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    protected = {entry["snapshot"]: (root / entry["snapshot"]).read_bytes() for entry in manifest["cohorts"]}
    _later_policy_evidence(root, manifest, monkeypatch)
    publication.main(["--root", str(root), "--source-commit", "commit-c", "--device", "cuda", "--advance-policy"])
    assert '"rows":2' in capsys.readouterr().out
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    assert updated["view_revision"] == 3
    assert len(updated["cohorts"]) == 1 and len(updated["archived_policies"]) == 1
    assert updated["cohorts"][0]["candidates"] == {
        "bcap": "2026-10-03T01:00:00+00:00", "atlas": None}
    archive = updated["archived_policies"][0]
    assert archive == {key: deepcopy(manifest[key]) for key in (*publication.POLICY_FIELDS, "cohorts")}
    def unexpected(*args, **kwargs):
        pytest.fail("frozen blockers must not resolve against live source")
    monkeypatch.setattr(publication, "write_report", unexpected)
    result = publication.publish_current(root)
    current = read_json(result["json"])
    bcap = next(row for row in current["rows"] if row["candidate_id"] == "bcap")
    assert bcap["candidate_revision"] == "revision-c"
    assert bcap["tiers"]["1"] == {"passed": 0, "total": 5}
    assert bcap["qualified_tier"] == 0
    atlas = next(row for row in current["rows"] if row["candidate_id"] == "atlas")
    assert atlas["attempt_ids"] == [] and atlas["cost"]["wall_seconds"] is None
    assert atlas["status"] == "BLOCKED" and atlas["candidate_revision"] == "revision-d"
    assert atlas["runtime_cohort"]["execution_backend"] == "cuda"
    assert len(current["configuration_rows"]) == 2
    assert all(row["candidate_revision"] != "revision-a" for row in current["configuration_rows"])
    original = next(row for row in current["archived_evidence_rows"] if row["candidate_id"] == "bcap")
    assert original["evidence_policy"]["view_revision"] == 2
    assert original["evidence_policy"]["tier_requirements"] == manifest["tier_requirements"]
    assert original["tiers"]["1"] == {"passed": 3, "total": 3}
    assert original["qualified_tier"] == 1
    original.pop("evidence_policy")
    assert original in baseline["evidence_rows"]
    assert all((root / relative).read_bytes() == content for relative, content in protected.items())
    markdown = Path(result["report"]).read_text()
    assert "0/5 (4 not run)" in markdown
    assert "Complete numerical publication and provenance" in markdown
    assert len(list((root / "reports/forge").glob("technique-inventory.md"))) == 1
    outputs = _outputs(root)
    assert publication.publish_current(root) == result
    assert _outputs(root) == outputs
    assert publication.publish_current(root, source_commit="commit-c", execution_backend="cuda", advance_policy=True)
    assert publication.publish_current(root) == result
    assert read_json(root / publication.EVIDENCE_MANIFEST) == updated


def test_policy_advance_refuses_old_source_and_retains_current_files(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    report = _later_policy_evidence(root, manifest, monkeypatch)
    report["policy_fingerprint"] = manifest["policy_fingerprint"]
    before = _outputs(root)
    manifest_bytes = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    with pytest.raises(ValueError, match="current view policy differs"):
        publication.publish_current(root, source_commit="commit-c")
    with pytest.raises(ValueError, match="new source evidence differs from the current view policy"):
        publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    assert _outputs(root) == before
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest_bytes


def _pin_existing_policy_rows(root, manifest):
    rows = [read_json(root / entry["snapshot"])["rows"][0] for entry in manifest["cohorts"]]
    atomic_json(root / "configs/forge/trainer-families.json", {
        "schema_version": 1, "families": [{"id": row["candidate_id"], "label": row["candidate_id"],
                                           "candidates": [row["candidate_id"]], "canonical_candidate": row["candidate_id"]}
                                          for row in rows]})
    card = _family_pin(root, manifest, {**rows[0], "trainer_family": rows[0]["candidate_id"]})
    card["selections"] += [family_row_pin({**row, "trainer_family": row["candidate_id"]},
                                         selection_kind="historical_incumbent", reason="Recorded incumbent.")
                           for row in rows[1:]]
    card["historical_selections"] = []
    atomic_json(root / CURRENT_SELECTION, card)
    return card


@pytest.mark.parametrize("blocked", [True, False])
def test_policy_advance_rebinds_same_selected_recipes_and_archives_exact_card(evidence, monkeypatch, blocked):
    root, manifest = evidence
    card = _pin_existing_policy_rows(root, manifest)
    if not blocked:
        path = root / CURRENT_SELECTION
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    publication.publish_current(root)
    original = (root / CURRENT_SELECTION).read_bytes()
    protected = {entry["snapshot"]: (root / entry["snapshot"]).read_bytes() for entry in manifest["cohorts"]}
    report = _later_policy_evidence(root, manifest, monkeypatch)
    if not blocked:
        report["rows"][1]["status"] = "UNKNOWN"
        report["rows"][1]["tasks"][0]["status"] = "UNKNOWN"
        report["provenance"].pop("input_digest")
        report["provenance"]["input_digest"] = stable_hash(report)
    result = publication.publish_current(root, source_commit="commit-c",
                                         execution_backend="cuda" if blocked else None, advance_policy=True)
    advanced = read_json(root / CURRENT_SELECTION)
    assert advanced["policy_fingerprint"] == report["policy_fingerprint"]
    assert advanced["historical_selections"] == card["historical_selections"]
    assert [(pin["trainer_family"], pin["candidate_id"], pin["execution_backend"]) for pin in advanced["selections"]] == [
        (pin["trainer_family"], pin["candidate_id"], pin["execution_backend"]) for pin in card["selections"]]
    assert [pin["candidate_revision"] for pin in advanced["selections"]] == ["revision-c", "revision-d"]
    assert all(pin["selection_kind"] == "historical_incumbent" for pin in advanced["selections"])
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    archive = updated["archived_policies"][0]["family_selection"]
    assert (root / archive["path"]).read_bytes() == original
    assert file_hash(root / archive["path"]) == archive["sha256"]
    current = read_json(result["json"])
    assert current["provenance"]["family_current_selection_sha256"] == file_hash(root / CURRENT_SELECTION)
    assert all(row["qualified_tier"] == 0 for row in current["rows"])
    assert all((root / path).read_bytes() == data for path, data in protected.items())
    outputs = _outputs(root)
    assert publication.publish_current(root, execution_backend="cuda") == result
    assert _outputs(root) == outputs
    assert publication.publish_current(root, source_commit="commit-c", execution_backend="cuda", advance_policy=True) == result
    assert read_json(root / publication.EVIDENCE_MANIFEST) == updated
    (root / archive["path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="archived family selection hash mismatch"):
        publication.publish_current(root)
    assert _outputs(root) == outputs


@pytest.mark.parametrize("failure", ["missing_new_row", "invalid_old_pin", "late_validation"])
def test_failed_policy_pin_advance_leaves_selection_and_publication_unchanged(evidence, monkeypatch, failure):
    root, manifest = evidence
    card = _pin_existing_policy_rows(root, manifest)
    publication.publish_current(root)
    report = _later_policy_evidence(root, manifest, monkeypatch)
    if failure == "missing_new_row":
        report["rows"].pop()
        report["provenance"].pop("input_digest")
        report["provenance"]["input_digest"] = stable_hash(report)
    elif failure == "invalid_old_pin":
        card["selections"][0]["scientific_row_sha256"] = "invalid"
        atomic_json(root / CURRENT_SELECTION, card)
    else:
        def fail_after_selection_validation(*args, **kwargs):
            raise ValueError("late publication validation failed")
        monkeypatch.setattr(publication, "_family_pages", fail_after_selection_validation)
    original = (root / CURRENT_SELECTION).read_bytes()
    manifest_bytes = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    outputs = _outputs(root)
    with pytest.raises(ValueError):
        publication.publish_current(root, source_commit="commit-c", execution_backend="cuda", advance_policy=True)
    assert (root / CURRENT_SELECTION).read_bytes() == original
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest_bytes
    assert _outputs(root) == outputs
    assert not (root / CURRENT_SELECTION.parent / "history").exists()


def test_policy_pin_advance_marks_fully_observed_new_failures_as_measurements(evidence, monkeypatch):
    root, manifest = evidence
    original = _pin_existing_policy_rows(root, manifest)
    original["historical_selections"] = deepcopy(original["selections"])
    atomic_json(root / CURRENT_SELECTION, original)
    prior = [(entry, *publication._snapshot(root, entry, manifest)) for entry in manifest["cohorts"]]
    report = _later_policy_evidence(root, manifest, monkeypatch)
    row = report["rows"][0]
    row["tasks"] = [{"task_id": name, "status": "FAIL" if name == "ring" else "PASS"}
                    for name in report["tier_requirements"]["1"]]
    staged, _ = publication._advanced_family_selection(
        root, manifest, report, {item["candidate_id"]: item for item in report["rows"]}, prior)
    assert staged["selections"][0]["selection_kind"] == "current_measurement"
    assert staged["selections"][0]["measurement_views"] == [manifest["view"]]
    assert staged["selections"][1]["selection_kind"] == "historical_incumbent"
    assert staged["historical_selections"] == original["historical_selections"]
    assert read_json(root / CURRENT_SELECTION)["policy_fingerprint"] == manifest["policy_fingerprint"]


def test_policy_pin_advance_requires_explicit_backend_for_mixed_selections(evidence, monkeypatch):
    root, manifest = evidence
    card = _pin_existing_policy_rows(root, manifest)
    publication.publish_current(root)
    _later_policy_evidence(root, manifest, monkeypatch)
    card["selections"][1]["execution_backend"] = "cpu"
    atomic_json(root / CURRENT_SELECTION, card)
    original = (root / CURRENT_SELECTION).read_bytes()
    outputs = _outputs(root)
    def unexpected(*args, **kwargs):
        pytest.fail("ambiguous backend must be refused before source regrade")
    monkeypatch.setattr(publication, "regenerate", unexpected)
    with pytest.raises(ValueError, match="mixed selected backends requires an explicit --device"):
        publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    assert (root / CURRENT_SELECTION).read_bytes() == original
    assert _outputs(root) == outputs


def test_policy_advance_requires_later_revision_and_source(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    before = _outputs(root)
    with pytest.raises(ValueError, match="requires --source-commit"):
        publication.publish_current(root, advance_policy=True)
    _later_policy_evidence(root, manifest, monkeypatch, revision=1)
    with pytest.raises(ValueError, match="later revision of the same view"):
        publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    assert _outputs(root) == before


def test_archived_policy_corruption_cannot_overwrite_current_publication(evidence, monkeypatch):
    root, manifest = evidence
    _later_policy_evidence(root, manifest, monkeypatch)
    publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    before = _outputs(root)
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    path = root / updated["archived_policies"][0]["cohorts"][0]["snapshot"]
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_same_source_cpu_registration_retains_cuda_evidence(evidence, monkeypatch):
    root, manifest = evidence
    original = read_json(root / manifest["cohorts"][0]["snapshot"])
    report = deepcopy(original)
    row = report["rows"][0]
    row["runtime_cohort"] = {"execution_backend": "cpu", "compute_profiles": {"cpu": {"model": "cpu-a"}}}
    row["candidate_revision"], row["cohort"] = "revision-cpu", "cohort-cpu"
    row["attempt_ids"] = ["attempt-cpu"]
    report["execution_backend"] = "cpu"
    proof = deepcopy(report["provenance"]["qualified_receipts"]["attempt-a"])
    proof["canonical_result_hash"] = "result-cpu"
    report["provenance"]["qualified_receipts"] = {"attempt-cpu": proof}
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    summary = read_json(root / "reports/forge/technique-receipts/attempt-a.json")
    summary["candidate_revision"] = "revision-cpu"
    summary["provenance"]["canonical_result_hash"] = "result-cpu"
    atomic_json(root / "reports/forge/technique-receipts/attempt-cpu.json", summary)
    atomic_json(root / "reports/forge/attempts/attempt-cpu/result.json", {"raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    # Restrict this fixture to its measured technique; Atlas has no CPU entry.
    (root / "configs/forge/ideas/atlas.json").unlink()
    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-a"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    cpu = publication.publish_current(root, execution_backend="cpu", source_commit="commit-a")
    assert read_json(cpu["json"])["rows"][0]["candidate_revision"] == "revision-cpu"
    cuda = read_json(publication.publish_current(root, execution_backend="cuda")["json"])
    assert cuda["rows"][0]["candidate_revision"] == "revision-a"
    with pytest.raises(ValueError, match="explicit whole-row family selection"):
        publication.publish_current(root)
    both = read_json(publication.publish_current(
        root, recorded_policy="configs/forge/views/discriminator_stability.json")["json"])
    assert {row["runtime_cohort"]["execution_backend"] for row in both["rows"]} == {"cpu", "cuda"}
    assert len(read_json(root / publication.EVIDENCE_MANIFEST)["cohorts"]) == 3


def test_unmeasured_other_backend_cannot_overwrite_measured_registration(evidence, monkeypatch):
    root, manifest = evidence
    _, report = _register(root, deepcopy(manifest), "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    unmeasured = deepcopy(report["rows"][0])
    unmeasured.update(attempt_ids=[], candidate_revision="unmeasured-cpu", cohort="unmeasured-cpu",
                      runtime_cohort={"execution_backend": "cpu"},
                      tasks=[{"task_id": "two_pole", "status": "UNKNOWN"}])
    report["rows"].append(unmeasured)
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    atomic_json(root / "reports/forge/attempts/attempt-c/result.json", {
        "raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    result = publication.publish_current(root, source_commit="commit-c")
    current = read_json(result["json"])
    selected = next(row for row in current["rows"] if row["candidate_id"] == "bcap")
    assert selected["candidate_revision"] == "revision-c"
    assert selected["attempt_ids"] == ["attempt-c"]
    assert selected["runtime_cohort"]["execution_backend"] == "cuda"
    assert publication.publish_current(root) == result


def test_family_new_hardware_keeps_its_own_canonical_fallback(evidence):
    root, manifest = evidence
    (root / "configs/forge/ideas/atlas.json").unlink()
    atomic_json(root / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "bcap", "label": "BCap", "canonical_candidate": "bcap", "candidates": ["bcap"]}]})
    entry, snapshot = _register(root, manifest, "bcap-trial", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / "configs/forge/ideas/bcap-trial.json", {"id": "bcap-trial", "trainer_family": "bcap"})
    canonical = deepcopy(snapshot["rows"][0])
    canonical.update(candidate_id="bcap", candidate_revision="revision-canonical-c", cohort="canonical-c", attempt_ids=[],
                     tasks=[{"task_id": "two_pole", "status": "UNKNOWN"}], cost={"wall_seconds": None})
    snapshot["rows"].append(canonical)
    snapshot["provenance"].pop("input_digest")
    snapshot["provenance"]["input_digest"] = stable_hash(snapshot)
    atomic_json(root / entry["snapshot"], snapshot)
    entry["json_sha256"] = file_hash(root / entry["snapshot"])
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    with pytest.raises(ValueError, match="explicit whole-row family selection"):
        publication.publish_current(root)
    recorded = read_json(publication.publish_current(
        root, recorded_policy="configs/forge/views/discriminator_stability.json")["json"])
    assert len(recorded["rows"]) == 2 and all(row["candidate_id"] == "bcap" for row in recorded["rows"])
    assert len(recorded["configuration_rows"]) == 3
    assert next(row for row in recorded["rows"] if row["cohort"] == "canonical-c")["attempt_ids"] == []
    incumbent = deepcopy(read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0])
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    result = read_json(publication.publish_current(root)["json"])
    assert len(result["rows"]) == 1 and result["rows"][0]["cohort"] == "cohort-a"
    assert len(result["configuration_rows"]) == 3
    assert {row["alternative_scope"] for row in result["configuration_rows"]} == {"selected", "archived_alternative"}


@pytest.mark.parametrize("tamper,message", [
    ("bytes", "snapshot hash mismatch"), ("digest", "input digest mismatch"),
    ("proof", "candidate/source cohort differs"), ("denominator", "tier denominator"),
    ("protocol", "incompatible tier_requirements"), ("policy", "current view policy differs"),
    ("contract", "invalid recipe_contracts identity"),
])
def test_invalid_evidence_cannot_overwrite_current_table(evidence, tamper, message):
    root, manifest = evidence
    publication.publish_current(root)
    before = _outputs(root)
    entry = manifest["cohorts"][0]
    path = root / entry["snapshot"]
    report = read_json(path)
    if tamper == "proof":
        summary_path = root / "reports/forge/technique-receipts/attempt-a.json"
        summary = read_json(summary_path)
        summary["candidate_revision"] = "wrong-revision"
        atomic_json(summary_path, summary)
    elif tamper == "policy":
        atomic_json(root / "configs/forge/views/discriminator_stability.json", {"id": "discriminator_stability", "revision": 3})
    else:
        if tamper in {"bytes", "digest", "denominator"}:
            report["rows"][0]["tiers"]["1"]["total"] = 2
        elif tamper == "protocol":
            report["tier_requirements"]["1"] = ["two_pole"]
        elif tamper == "contract":
            report["recipe_contracts"] = {"invalid-hash": {"reg_arm": "other"}}
        if tamper not in {"bytes", "digest"}:
            report["provenance"].pop("input_digest")
            report["provenance"]["input_digest"] = stable_hash(report)
        atomic_json(path, report)
        if tamper != "bytes":
            entry["json_sha256"] = file_hash(path)
            atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    with pytest.raises(ValueError, match=message):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_cli_has_one_default_destination(evidence, capsys):
    root, _ = evidence
    publication.main(["--root", str(root)])
    assert '"rows":2' in capsys.readouterr().out
    outputs = _outputs(root)
    # --output-prefix now writes an independent report; it can never refresh
    # or register the one current board.
    with pytest.raises(SystemExit):
        publication.main(["--root", str(root), "--output-prefix", "another-board", "--refresh-publication"])
    assert _outputs(root) == outputs


def test_raw_technique_export_cannot_replace_registered_current_evidence(evidence):
    from experiments.forge.technique_board import write_report
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    with pytest.raises(ValueError, match="regenerate_technique_inventory.py"):
        write_report(root)
    assert _outputs(root) == before


def _tier_display_fixture(tier1, later="UNKNOWN"):
    required = {"1": [f"smoke-{i}" for i in range(5)],
                "2": [f"quality-{i}" for i in range(19)],
                "3": [f"hold-{i}" for i in range(2)]}
    statuses = {"1": tier1, "2": [later] * 19, "3": [later] * 2}
    row = {"candidate_id": "atlas", "trainer_family": "atlas", "technique": "Atlas",
           "runtime_cohort": {"execution_backend": "cpu"}, "qualified_tier": 0,
           "tasks": [{"task_id": task, "status": status, "gate_status": status}
                     for tier, names in required.items() for task, status in zip(names, statuses[tier])],
           "tiers": {tier: {"passed": values.count("PASS"), "total": len(values),
                            "counts": dict(Counter(values))} for tier, values in statuses.items()}}
    return {"view": "discriminator_stability", "tier_requirements": required, "rows": [row], "evidence_sources": {},
            "provenance": {"input_digest": "synthetic-display-only"}}


@pytest.mark.parametrize("status,label", [
    ("BLOCKED", "BLOCKED"), ("UNKNOWN", "UNKNOWN"), ("NOT_RUN", "NOT RUN"),
    ("INCOMPLETE", "INCOMPLETE"), ("INVALID", "INVALID"),
])

def test_current_tier_cells_keep_blocked_and_unmeasured_denominators(tmp_path, status, label):
    result = _tier_display_fixture([status] * 5, later=status)
    before = deepcopy(result)
    row = result["rows"][0]
    for tier, required in result["tier_requirements"].items():
        rendered = publication._current_tier_cell(row, tier, required, "table.json")
        assert f"{label} ({len(required)} required)" == rendered
        assert "FAIL" not in rendered
    assert result == before


@pytest.mark.parametrize("statuses,expected", [
    (["PASS", "FAIL", "BLOCKED", "UNKNOWN", "NOT_RUN"],
     "1/5<br>BLOCKED 1 · FAIL 1 · NOT RUN 1 · UNKNOWN 1"),
    (["FAIL"] * 5, "0/5<br>FAIL 5"),
    (["FAIL"] + ["UNKNOWN"] * 4, "0/5<br>FAIL 1 · UNKNOWN 4"),
    (["NOT_RUN", "UNKNOWN", "NOT_RUN", "UNKNOWN", "UNKNOWN"],
     "NOT RUN/UNKNOWN (5 required)"),
    (["PASS"] * 5, "5/5 PASS"),
])

def test_current_tier_cells_show_actual_mixed_gate_counts(tmp_path, statuses, expected):
    result = _tier_display_fixture(statuses)
    before = deepcopy(result)
    row = result["rows"][0]
    assert publication._current_tier_cell(row, "1", result["tier_requirements"]["1"], "table.json") == expected
    assert publication._current_tier_cell(row, "2", result["tier_requirements"]["2"], "table.json") == "UNKNOWN (19 required)"
    assert publication._current_tier_cell(row, "3", result["tier_requirements"]["3"], "table.json") == "UNKNOWN (2 required)"
    assert result == before


def test_legacy_tier_display_does_not_invent_an_all_failed_breakdown(tmp_path):
    result = _tier_display_fixture(["BLOCKED"] * 5)
    row = result["rows"][0]
    row["tiers"]["1"].pop("counts")
    assert publication._current_tier_cell(row, "1", result["tier_requirements"]["1"], "table.json") == "BLOCKED (5 required)"
    # A legacy aggregate can lack enough task detail to reconstruct its counts.
    row["tiers"]["1"]["passed"] = 1
    before = deepcopy(result)
    rendered = publication._current_tier_cell(row, "1", result["tier_requirements"]["1"], "table.json")
    assert rendered == "1/5<br>[5 required; task statuses](table.json)"
    assert "FAIL" not in rendered and result == before



def test_main_policy_rows_link_separate_history_and_failed_hold_without_credit():
    result = read_json(ROOT / "reports/forge/technique-inventory.json")
    before = deepcopy(result)
    markdown = publication._current_markdown(result, ROOT, ROOT / "reports/forge/technique-inventory.md")
    selected = [row for row in result["rows"] if row["trainer_family"] in {"atlas", "e22"}]
    assert {row["trainer_family"] for row in selected} == {"atlas", "e22"}
    for row in selected:
        label = {"atlas": "Atlas", "e22": "E22"}[row["trainer_family"]]
        links = publication._separate_baseline_links(result, row, ROOT, ROOT / "reports/forge/technique-inventory.md")
        assert "[Atlas history: 19/19 PASS](continuous-baseline-20261003/README.md)" in links
        assert f"[C6 {label} hold FAIL](c6-baseline-debug-20261003/README.md)" in links
        assert row["qualified_tier"] == 0
    assert "19/19" not in markdown.replace(ATLAS_HISTORY_NOTE, "")
    assert "C6 baseline selection and retained diagnosis" in _family_text(ROOT, result, "atlas")
    assert result == before
    without_history = deepcopy(result)
    without_history.pop("completed_api_studies")
    assert publication._separate_baseline_links(without_history, selected[0], ROOT,
                                                ROOT / "reports/forge/technique-inventory.md") == []


def _copy_atlas_progress(root):
    for relative, _, _ in publication.ATLAS_PROGRESS_INPUTS.values():
        path = Path("reports/forge") / relative
        for name in (path, path.with_name("README.md")):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, root / name)


def test_atlas_progress_is_source_bound_additive_display_and_preserves_every_ordinary_row(evidence):
    root, _ = evidence
    before = read_json(publication.publish_current(root)["json"])
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    assert "atlas_unblocking_progress" not in before
    _copy_atlas_progress(root)
    published = publication.publish_current(root)
    after = read_json(published["json"])
    for key, value in before.items():
        if key != "provenance":
            assert after[key] == value
    assert set(after) - set(before) == {"atlas_unblocking_progress"}
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest
    assert after["provenance"]["selected_rows_sha256"] == before["provenance"]["selected_rows_sha256"]
    progress = after["atlas_unblocking_progress"]
    assert progress["baseline"]["counts"] == {"PASS": 7, "FAIL": 11, "BLOCKED": 8}
    assert "passed" not in progress and "required" not in progress
    assert [row["required"] for row in progress["adaptations"]] == [26] * 5
    assert [row["counts"].get("NOT_RUN") for row in progress["adaptations"]] == [22, 25, 25, 25, 25]
    assert [row["passed"] for row in progress["adaptations"]] == [4, 1, 1, 1, 0]
    assert len({row["source"]["digest"] for row in progress["adaptations"]}) == 2
    assert progress["word"]["status"] == "INVALID"
    assert progress["word"]["numerical_gate"] == "UNAVAILABLE"
    assert progress["word"]["raw_completed_steps"] == 20001
    assert progress["word"]["recorded_endpoint"]["mass_tv"] == .6
    assert all(progress[key] is False for key in (
        "qualification_input", "qualification_reuse", "ordinary_tier_credit", "cross_cohort_pooling",
        "default_adoption", "speed_ranking"))
    markdown = Path(published["report"]).read_text()
    assert markdown.count("| Family / selected recipe | Tier 1 requirements | Tier 2 requirements"
                          " | Tier 3 requirements | Total view requirements |") == 1
    assert "7/26 PASS" not in markdown and "4/26 PASS" not in markdown
    assert "INVALID 1" not in markdown and "7/8" not in markdown
    family_page = _family_text(root, after, "atlas")
    for item in [progress["baseline"], *progress["adaptations"], progress["word"]]:
        link = __import__("os").path.relpath(root / item["readout"], root / "reports/forge/families")
        assert f"]({link})" in family_page
    assert len(after["common26_display"]["rows"]) == 12
    assert all(row["fresh_common"]["passed"] is None for row in after["common26_display"]["rows"])
    # Scientific diagnostics and ordinary reference remain byte-identical data.
    assert next(row for row in after["rows"] if row["candidate_id"] == "atlas")["bindings"]["prior"]["kind"] == "mog"
    assert after["rows"] == before["rows"]
    outputs = _outputs(root)
    assert publication.publish_current(root) == published and _outputs(root) == outputs


@pytest.mark.parametrize("view,recorded_policy", [
    ("discriminator_stability", "configs/forge/view-history/discriminator_stability-v2.json"),
    ("quality_coverage", None),
])
def test_ordinary_atlas_progress_does_not_change_recorded_or_other_view_rendering(view, recorded_policy):
    result = _tier_display_fixture(["UNKNOWN"] * 5)
    result.update(view=view, view_revision=2)
    if recorded_policy:
        result["recorded_policy"] = recorded_policy
    path = ROOT / "reports/forge/another-view.md"
    original = publication._current_markdown(result, ROOT, path)
    result["atlas_unblocking_progress"] = publication._atlas_unblocking_progress(ROOT)
    before = deepcopy(result)
    rendered = publication._current_markdown(result, ROOT, path)
    assert rendered == original
    assert "## Ordinary MoG qualification" not in rendered
    assert "## Atlas unblocking progress" not in rendered
    assert "[Adaptation progress]" not in rendered
    assert result == before


@pytest.mark.parametrize("tamper", ["partial", "changed_count", "word_pass", "source", "credit"])
def test_changed_progress_evidence_cannot_rewrite_ordinary_outputs(evidence, tamper):
    root, _ = evidence
    _copy_atlas_progress(root)
    publication.publish_current(root)
    before = _outputs(root)
    if tamper == "partial":
        (root / "reports/forge" / publication.ATLAS_PROGRESS_INPUTS["word_context"][0]).unlink()
    else:
        key = "word_context" if tamper == "word_pass" else "baseline"
        path = root / "reports/forge" / publication.ATLAS_PROGRESS_INPUTS[key][0]
        value = read_json(path)
        if tamper == "changed_count":
            value["counts"]["PASS"] = 26
        elif tamper == "word_pass":
            value["execution_status"] = "PASS"
        elif tamper == "source":
            value["source"]["origin_commit"] = "0" * 40
        else:
            value["qualification_input"] = True
        atomic_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_committed_atlas_progress_display_matches_its_generator_without_raw_evidence():
    result = read_json(ROOT / "reports/forge/technique-inventory.json")
    expected = publication._atlas_unblocking_progress(ROOT)
    assert result["atlas_unblocking_progress"] == expected
    markdown = publication._current_markdown(result, ROOT, ROOT / "reports/forge/technique-inventory.md")
    assert (ROOT / "reports/forge/technique-inventory.md").read_text() == markdown
    families = _current_family_roster()
    assert {row["trainer_family"] for row in result["rows"]} == families
    cohort_count = len(result["family_progress"]["families"])
    assert "4/26 PASS" not in markdown and sum(line.startswith("| **[") for line in markdown.splitlines()) == cohort_count
    # No standalone 7/8 progress score. Totals such as 77/81 (81 counted view
    # requirements since the 2026-10-15 test dispositions) are not one.
    assert "Atlas unblocking progress" not in markdown and re.search(r"(?<![\d/])7/8(?![\d/])", markdown) is None


@pytest.mark.parametrize("name,prior,expected", [
    ("atlas", {"kind": "mog", "sigma": .025}, "MoG"),
    ("misleading-mog-name", {"kind": "particle_cloud", "sigma": 0.0}, "Particles"),
    ("release07-gan-v3-cloud", {"kind": "mog", "sigma": .025}, "MoG"),
    ("atlas", {}, "UNKNOWN"),
])
def test_representation_uses_declaration_not_model_name(name, prior, expected):
    row = dict(candidate_id=name, bindings=dict(prior=prior))
    assert publication._ordinary_representation({}, row) == expected


def test_suite_representation_discloses_separate_task_owned_particle_laws():
    result = {"task_contracts": {"particle-host": {"prior": {"kind": "particle_cloud"}},
                                 "mog-host": {"prior": {"kind": "mog"}}}}
    row = {"bindings": {"prior": {"kind": "mog"},
                        "task_contracts": {"two_pole": "particle-host", "ring": "mog-host"}}}
    before = deepcopy((result, row))
    assert publication._ordinary_representation(result, row) == "MoG + particles (per task)"
    # A requested particle family blocked on MoG tasks is not a measured hybrid.
    row["bindings"]["prior"] = {"kind": "particle_cloud"}
    assert publication._ordinary_representation(result, row) == "Particles"
    row["bindings"]["prior"] = before[1]["bindings"]["prior"]
    assert (result, row) == before


def test_source_scoped_representation_binds_applied_baseline_and_named_owners():
    progress = publication._atlas_unblocking_progress(ROOT)
    baseline = progress["baseline"]["representation"]
    assert baseline["label"] == "Particles" and len(baseline["cases"]) == 18
    assert all(case["prior"]["kind"] == "particle_cloud" and case["prior"]["sigma"] == 0.0
               for case in baseline["cases"].values())
    rows = {row["family"]: row for row in progress["adaptations"]}
    ae = rows["atlas_ae_routed"]["representation"]
    assert ae["label"] == "MoG (fixed σ .025; routed AE)"
    assert next(iter(ae["applied"].values()))["prior"]["sigma"] == .025
    conditional = rows["atlas_conditional"]["representation"]
    assert {item["table_ownership"]["owner"] for item in conditional["applied"].values()} == {"prior.z", "generator.bank"}
    unused = rows["atlas_routed"]["representation"]
    assert next(iter(unused["applied"].values()))["table_ownership"]["sampled_independent_prior"] is False
    assert next(iter(rows["atlas_multibank"]["representation"]["applied"].values()))["prior"]["sigma"] == 0.0
    word = rows["atlas_word_joint_min11"]
    assert word["counts"] == {"INVALID": 1, "NOT_RUN": 25}
    assert not word["representation"]["applied"]
    assert next(iter(word["media"].values()))["accepted_numeric_verdict"] == "UNAVAILABLE"


def test_shared_score_intro_preserves_unrelated_index_and_rebuilds_idempotently(evidence):
    root, _ = evidence
    _copy_atlas_progress(root)
    path = root / "reports/forge/shared-score-index-20261003/README.md"
    path.parent.mkdir(parents=True)
    unrelated = "## Completed current Atlas GPU diagnostic\n\nMartyn's existing report, costs, links and snapshots.\n"
    path.write_text("# Shared ParticleGAN score index\n\n## Atlas unblocking progress\n\nOld intro.\n\n" + unrelated)
    published = publication.publish_current(root)
    content = path.read_text()
    assert path.with_name("ARCHIVED_REPORTS.md").read_text() == unrelated
    assert "[Historical report archive](ARCHIVED_REPORTS.md)" in content
    assert "[current family leaderboard](../technique-inventory.md)" in content
    assert "| Family / view |" not in content
    assert "## Atlas unblocking progress" not in content and "7/8" not in content
    assert "atlas-named-gpu-diagnostics-native-v3-20261003/README.md" in _family_text(
        root, read_json(published["json"]), "atlas")
    assert publication.publish_current(root) == published and path.read_text() == content


def test_unknown_shared_index_boundary_rejects_before_public_output_changes(evidence):
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    path = root / "reports/forge/shared-score-index-20261003/README.md"
    path.parent.mkdir(parents=True)
    path.write_text("Unrelated index without the maintained boundary.\n")
    with pytest.raises(ValueError, match="preserved report boundary"):
        publication.publish_current(root)
    assert _outputs(root) == before and path.read_text() == "Unrelated index without the maintained boundary.\n"


def _synthetic_half_base_display():
    """Renderer-only fixture: no trained receipt, model or qualification."""
    return {"source": {"origin_commit": "f" * 40},
            "readout": "reports/forge/synthetic-half-base/README.md",
            "representation": {"label": "Particles (N11 joint cloud; free encoder)"},
            "media": {"path": "reports/forge/synthetic-half-base/goal.gif"},
            "cost": {"paid_seconds": 556.4301753160544},
            "accounting": {"inclusive_charged_seconds": 1466.669318475062,
                           "gpu0_charged_seconds": 234.82608077581972,
                           "gpu1_charged_seconds": 1231.8432376992423}}


def _copy_half_base_report(root):
    """Copy a sealed display input, without reading any raw tensors or streams."""
    relative = Path("reports/forge") / publication.WORD_HALF_BASE_INPUT[0]
    for path in (relative, relative.with_name("README.md"), relative.parent / "media/goal.gif"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, root / path)
    return root / relative


def test_sealed_half_base_is_additive_keeps_invalid_and_charges_predecessors_once(evidence):
    root, _ = evidence
    _copy_atlas_progress(root)
    before = read_json(publication.publish_current(root)["json"])
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    _copy_half_base_report(root)
    after = read_json(publication.publish_current(root)["json"])
    for key, value in before.items():
        if key != "provenance":
            assert after[key] == value
    assert set(after) - set(before) == {"word_half_base_diagnostic"}
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest
    assert after["provenance"]["selected_rows_sha256"] == before["provenance"]["selected_rows_sha256"]
    row = after["word_half_base_diagnostic"]
    assert row["status"] == "COMPLETE" and row["numerical_gate"] == "FAIL"
    assert row["passed"] == 0 and row["required"] == 26 and row["counts"] == {"FAIL": 1, "NOT_RUN": 25}
    assert row["representation"]["label"] == "Particles (N11 joint cloud; free encoder)"
    assert row["result"]["final_metrics"]["quality_fraction"] == .546875
    assert row["historical_word"]["status"] == "INVALID" and row["historical_word"]["recertified"] is False
    assert row["cost"]["prior_charged_seconds"] == 910.2391431590077
    assert row["cost"]["inclusive_charged_seconds"] == 1466.669318475062
    assert row["media"]["frames"] == 9 and row["result"]["original_grader_summary"]["passing_observations"] == 0
    assert all(row[flag] is False for flag in ("qualification_input", "qualification_reuse", "ordinary_tier_credit",
                                              "default_adoption", "speed_ranking", "cross_cohort_pooling"))
    markdown = (root / "reports/forge/technique-inventory.md").read_text()
    assert "0/26 PASS" not in markdown and "INVALID 1" not in markdown
    family_page = _family_text(root, after, "atlas")
    assert "Word half-base rate contrast" in family_page
    outputs = _outputs(root)
    publication.publish_current(root)
    assert _outputs(root) == outputs


@pytest.mark.parametrize("tamper", ["report_bytes", "report_missing", "gif_bytes", "readout_bytes", "status",
    "denominator", "source", "recipe", "particle_law", "cadence", "owners", "clock", "old_invalid",
    "cost", "credit"])
def test_half_base_input_or_scientific_summary_drift_cannot_rewrite_ordinary_outputs(evidence, monkeypatch, tamper):
    root, _ = evidence
    path = _copy_half_base_report(root)
    publication.publish_current(root)
    before = _outputs(root)
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    if tamper == "report_bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    elif tamper == "report_missing":
        path.unlink()
    elif tamper == "gif_bytes":
        gif = path.parent / "media/goal.gif"
        gif.write_bytes(gif.read_bytes() + b"changed")
    elif tamper == "readout_bytes":
        path.with_name("README.md").write_text("Changed readout.\n")
    else:
        report = read_json(path)
        if tamper == "status":
            report["accepted_numeric"] = "PASS"
        elif tamper == "denominator":
            report["slots"].pop("grid100")
        elif tamper == "source":
            report["source"]["origin_commit"] = "0" * 40
        elif tamper in {"recipe", "particle_law"}:
            report["resolved_recipe"]["lr" if tamper == "recipe" else "standardize"] = .0053125 if tamper == "recipe" else True
            report["resolved_recipe_sha256"] = stable_hash(report["resolved_recipe"])
        elif tamper == "cadence":
            report["protocol"]["metric_steps"] = report["protocol"]["metric_steps"][:-1]
        elif tamper == "owners":
            report["result"]["enabled_owners"].remove("birth_death")
        elif tamper == "clock":
            report["result"]["optimizer_updates"]["encoder"] = 0
        elif tamper == "old_invalid":
            report["historical_word"]["status"] = "FAIL"
        elif tamper == "cost":
            report["cost"]["inclusive_charged_seconds"] = report["cost"]["paid_seconds"]
        elif tamper == "credit":
            report["qualification_input"] = True
        atomic_json(path, report)
        # A coherent byte-pin substitution must still fail the named summary
        # guards, rather than hiding every negative behind one hash assertion.
        monkeypatch.setattr(publication, "WORD_HALF_BASE_INPUT",
                            (publication.WORD_HALF_BASE_INPUT[0], file_hash(path), publication.WORD_HALF_BASE_INPUT[2]))
    with pytest.raises(ValueError, match="half_base"):
        publication.publish_current(root)
    assert _outputs(root) == before
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest


def test_half_base_shared_index_cost_updates_without_touching_archived_index(evidence):
    root, _ = evidence
    relative = Path("reports/forge/shared-score-index-20261003/README.md")
    path = root / relative
    path.parent.mkdir(parents=True)
    original = (ROOT / relative).read_text()
    path.write_text(original)
    marker = "## Completed current Atlas GPU diagnostic\n"
    original_archive = (ROOT / relative).with_name("ARCHIVED_REPORTS.md")
    if original_archive.exists():
        archive_bytes = original_archive.read_bytes()
        shutil.copyfile(original_archive, path.with_name("ARCHIVED_REPORTS.md"))
    else:
        archive_bytes = (marker + original.split(marker, 1)[1]).encode()
    _copy_atlas_progress(root)
    _copy_half_base_report(root)
    publication.publish_current(root)
    rendered = path.read_text()
    assert path.with_name("ARCHIVED_REPORTS.md").read_bytes() == archive_bytes
    assert "[current family leaderboard](../technique-inventory.md)" in rendered
    assert "| Family / view |" not in rendered
    assert "| Model/configuration |" not in rendered
    assert "[Historical report archive](ARCHIVED_REPORTS.md)" in rendered
    assert "1466.669318475062 / 10500 seconds" in rendered
    assert "1231.8432376992423 / 3000" in rendered
    assert "556.4301753160544 / 900 seconds" in rendered
    assert "Prior charges are included once" in rendered
    assert "original C6 word INVALID is unchanged" in rendered



def test_half_base_render_keeps_old_invalid_and_separate_full_denominator():
    result = _as_common26_display_fixture(_tier_display_fixture(["BLOCKED"] * 5))
    result["atlas_unblocking_progress"] = publication._atlas_unblocking_progress(ROOT)
    result["word_half_base_diagnostic"] = _synthetic_half_base_display()
    before = deepcopy(result)
    markdown = publication._current_markdown(result, ROOT, ROOT / "reports/forge/synthetic-table.md")
    assert markdown.count("NOT_RUN (26 required)") == 12
    assert "0/26 PASS" not in markdown and "INVALID 1" not in markdown
    assert "Word half-base rate contrast" in markdown and "Retained word context" in markdown
    assert result == before



def test_one_visible_table_keeps_solution_families_with_view_breakdowns_and_separate_evidence():
    result = read_json(ROOT / "reports/forge/technique-inventory.json")
    before = deepcopy(result)
    families = _current_family_roster()
    assert len(result["rows"]) == len(families)
    assert {row["trainer_family"] for row in result["rows"]} == families
    text = publication._current_markdown(result, ROOT, ROOT / "reports/forge/technique-inventory.md")
    assert text.count("| Family / selected recipe | Tier 1 requirements | Tier 2 requirements"
                      " | Tier 3 requirements | Tier 4 requirements | Total view requirements |") == 1
    selected, details = text.split("## Full-suite results — failures included", 1)
    assert "<details>" not in selected
    table = [line for line in selected.splitlines() if line.startswith("|")][2:]
    cohort_count = len(result["family_progress"]["families"])
    assert len([line for line in table if line.startswith("| **[")]) == cohort_count
    labels = [family["label"] for family in result["family_progress"]["families"]]
    assert labels.count("BCAP") == 1 and labels.count("BCAP with K3P") == 0
    assert "tensorflow" not in "\n".join(table).lower() and "halloween" not in "\n".join(table).lower()
    bcap = next(family for family in result["family_progress"]["families"] if family["label"] == "BCAP")
    for cohort in bcap["cohorts"]:
        if cohort["anchor"] == bcap["current_cohort_anchor"]:
            selected = result["rows"][cohort["row_index"]]
            assert selected["candidate_id"] == bcap["benchmark_configuration"]["candidate_id"]
            assert selected["trainer_family"] == "bcap-dualnorm"
        else:
            assert cohort["total"]["passed"] == max(item["total"]["passed"] for item in cohort["configuration_alternatives"])
    page = _family_text(ROOT, result, bcap["id"])
    assert "Current benchmark configuration" in page and "optimizer_family=dualnorm" in page
    assert "Recorded observations" in page or "Metric" in page
    view_rows = sum(len(cohort["views"]) for family in result["family_progress"]["families"]
                    for cohort in family["cohorts"] if cohort["anchor"] == family["current_cohort_anchor"])
    assert sum(" | ↳ [" in line for line in details.splitlines() if line.startswith("|")) == view_rows
    assert all("families/" in line for line in table)
    assert "19/19" not in text.replace(ATLAS_HISTORY_NOTE, "") and "7/26 PASS" not in text and "0/26 PASS" not in text
    family_page = _family_text(ROOT, result, "atlas")
    assert "Original Atlas recipe and serving-law evidence" in family_page
    assert "Word half-base rate contrast" in family_page
    assert all(slot["fresh_common"]["passed"] is None for slot in result["common26_display"]["rows"])
    assert result == before


def test_original_pr223_score_uses_verified_full_original_law_not_changed_c6_cells():
    result = read_json(ROOT / "reports/forge/technique-inventory.json")
    before = deepcopy(result)
    row = publication._original_pr223_atlas(result)
    assert row == result["original_pr223_atlas"]
    assert row["counts"] == {"PASS": 19} and row["required"] == len(row["question_ids"]) == 19
    assert row["representation"] == "Particles"
    assert row["seeds"] == {"native": 1234, "moving": 1234, "portability": 0}
    assert row["publication"]["sha256"] == "4cdbec6b54e31ecdaac5ee2cd887258004c5da7c9a7102261e1c1c8c3939b66a"
    assert row["recipe"]["overrides"]["lr"] == .00425
    assert row["recipe"]["overrides"]["prior_lr_mult"] == 2.
    assert row["recipe"]["overrides"]["output_noise_mode"] == "learnable"
    assert not any(row[key] for key in ("qualification_input", "qualification_reuse",
                                       "default_adoption", "speed_ranking", "new_current_retest_credit"))
    text = publication._current_markdown(result, ROOT, ROOT / "reports/forge/technique-inventory.md")
    assert "Original Atlas recipe and serving-law evidence" in _family_text(ROOT, result, "atlas")
    families = _current_family_roster()
    assert {row["trainer_family"] for row in result["rows"]} == families
    cohort_count = len(result["family_progress"]["families"])
    assert "19/19" not in text.replace(ATLAS_HISTORY_NOTE, "")
    assert sum(line.startswith("| **[") for line in text.splitlines()) == cohort_count
    assert result == before


@pytest.mark.parametrize("tamper", ["scope", "duplicate", "partial", "clean_credit", "gate", "seed", "recipe", "source", "reuse"])
def test_original_pr223_display_rejects_changed_or_partial_original_claim(tamper):
    # Mutations are synthetic metadata controls, never new scientific evidence.
    result = deepcopy(read_json(ROOT / "reports/forge/technique-inventory.json"))
    study = next(row for row in result["completed_api_studies"]["rows"] if row["id"] == "atlas19_original")
    native = next(cell for cell in study["cells"] if cell["definition"]["group"] == "native")
    if tamper == "scope":
        study["required_cells"] = 26
    elif tamper == "duplicate":
        study["cells"][-1] = deepcopy(study["cells"][0])
    elif tamper == "partial":
        native["full_protocol_complete"] = False
    elif tamper == "clean_credit":
        native["native_gates"]["noisy"] = deepcopy(native["native_gates"]["clean"])
    elif tamper == "gate":
        native["definition"]["original_requirements"] = []
    elif tamper == "seed":
        study["cells"][0]["definition"]["original_host"]["seed"] = 1234
    elif tamper == "recipe":
        study["recipe"]["overrides"]["lr"] = .0053125
    elif tamper == "source":
        study["source"]["protected_files_sha256"]["configs/100gaussians/atlas.json"] = "0" * 64
    else:
        study["reuse"] = True
    with pytest.raises(ValueError, match="Original PR223"):
        publication._original_pr223_atlas(result)


@pytest.mark.parametrize("scope", ["recorded", "quality_coverage"])
def test_original_pr223_metadata_is_byte_inert_for_older_render_scopes(scope):
    result = _tier_display_fixture(["UNKNOWN"] * 5)
    result.update(view="quality_coverage" if scope == "quality_coverage" else "discriminator_stability", view_revision=2)
    if scope == "recorded":
        result["recorded_policy"] = "configs/forge/view-history/discriminator_stability-v2.json"
    path = ROOT / "reports/forge/older-view.md"
    original = publication._current_markdown(result, ROOT, path)
    result["original_pr223_atlas"] = publication._original_pr223_atlas(
        read_json(ROOT / "reports/forge/technique-inventory.json"))
    before = deepcopy(result)
    assert publication._current_markdown(result, ROOT, path) == original
    assert result == before


@pytest.mark.parametrize("scope", ["recorded", "quality_coverage"])
def test_half_base_metadata_is_byte_inert_for_older_render_scopes(scope):
    result = _tier_display_fixture(["UNKNOWN"] * 5)
    result.update(view="quality_coverage" if scope == "quality_coverage" else "discriminator_stability", view_revision=2)
    if scope == "recorded":
        result["recorded_policy"] = "configs/forge/view-history/discriminator_stability-v2.json"
    path = ROOT / "reports/forge/older-view.md"
    original = publication._current_markdown(result, ROOT, path)
    result["word_half_base_diagnostic"] = _synthetic_half_base_display()
    before = deepcopy(result)
    assert publication._current_markdown(result, ROOT, path) == original
    assert result == before


def test_original_completion_cohorts_rebuild_every_scientific_row_without_raw_logs(tmp_path):
    for relative in (publication.EVIDENCE_MANIFEST.parent, Path("reports/forge/technique-receipts"),
                     Path("configs/forge/ideas"), Path("configs/forge/configurations"),
                     Path("configs/forge/searches"), Path("configs/forge/views"),
                     Path("configs/forge/selections"),
                     Path("configs/forge/view-history"),
                     Path("configs/forge/tasks"), Path("configs/forge/protocols"),
                     Path("configs/forge/task-variants"),
                     Path("reports/forge/configuration-search"),
                     Path("reports/forge/tier1-completion")):
        if not (ROOT / relative).is_dir():
            continue
        shutil.copytree(ROOT / relative, tmp_path / relative)
    shutil.copyfile(ROOT / "configs/forge/trainer-families.json", tmp_path / "configs/forge/trainer-families.json")
    shutil.copyfile(ROOT / "configs/forge/defaults.json", tmp_path / "configs/forge/defaults.json")
    original = read_json(ROOT / "reports/forge/tier1-completion/publication.json")
    scoped_registry = Path("reports/forge/scoped-publications.json")
    if (ROOT / scoped_registry).is_file():
        # Replay the original navigation inputs, including its unchanged GIF
        # index, rather than the later pure BCAP media successor. Scoped
        # cohorts registered after this round are not in its recorded registry.
        registry = read_json(ROOT / scoped_registry)
        original_media = Path("reports/forge/tier1-completion/media.json")
        original_evidence = Path("reports/forge/tier1-completion/scoped-evidence.json")
        registry["cohorts"] = [entry for entry in registry["cohorts"] if entry["path"] == original_evidence.as_posix()]
        registry["media"] = {"path": original_media.as_posix(),
                             "sha256": file_hash(tmp_path / original_media)}
        atomic_json(tmp_path / scoped_registry, registry)
        assert file_hash(tmp_path / scoped_registry) == original["scoped_registry_sha256"]
        from experiments.forge.scoped_publications import HF_MEDIA_MANIFEST, load_publications
        if (ROOT / HF_MEDIA_MANIFEST).is_file():
            shutil.copyfile(ROOT / HF_MEDIA_MANIFEST, tmp_path / HF_MEDIA_MANIFEST)
        assert load_publications(tmp_path) == load_publications(
            ROOT, load=lambda path: registry if path == scoped_registry else read_json(ROOT / path))
    # This private checkout reconstructs the original completed publication.
    # Later source cohorts need their own task bindings, which the pure BCAP
    # composition controls exercise separately; no live admission guard changes.
    frozen = read_json(ROOT / "configs/forge/rounds/tier1-completion-v1.json")
    original_families = {row["family"] for row in frozen["candidate_roster"]}
    assert len(original_families) == len(frozen["candidate_roster"]) == 11
    # Restore the exact original publication card rather than deriving it
    # from later source refreshes or migrations. No recorded outcome changes.
    selection_archive = read_json(ROOT / "reports/forge/tier1-completion/selection-archive.json")
    assert selection_archive["scientific_source_commit"] == original["source_commit"]
    blob = subprocess.check_output(["git", "rev-parse",
                                   selection_archive["archive_commit"] + ":" + selection_archive["path"]], cwd=ROOT, text=True).strip()
    assert blob == selection_archive["git_blob"]
    (tmp_path / CURRENT_SELECTION).write_bytes(subprocess.check_output(
        ["git", "show", blob], cwd=ROOT))
    assert file_hash(tmp_path / CURRENT_SELECTION) == original["selection_sha256"]
    registry_path = tmp_path / "configs/forge/trainer-families.json"
    registry = json.loads(subprocess.check_output(
        ["git", "show", original["source_commit"] + ":configs/forge/trainer-families.json"], cwd=ROOT))
    registry["families"] = [family for family in registry["families"] if family["id"] in original_families]
    atomic_json(registry_path, registry)
    (tmp_path / publication.EVIDENCE_MANIFEST).write_bytes(subprocess.check_output(
        ["git", "show", selection_archive["archive_commit"] + ":" + publication.EVIDENCE_MANIFEST.as_posix()], cwd=ROOT))
    manifest = read_json(tmp_path / publication.EVIDENCE_MANIFEST)
    assert len(manifest["cohorts"]) == 1
    assert manifest["cohorts"][0]["source_commit"] == original["source_commit"]
    # Rebuild this recorded publication against its exact archived denominator.
    # The new scalar task does not relabel any of these scientific rows.
    policy_path = tmp_path / "configs/forge/views/discriminator_stability.json"
    policy_path.write_bytes(subprocess.check_output(
        ["git", "show", original["source_commit"] + ":configs/forge/views/discriminator_stability.json"], cwd=ROOT))
    assert stable_hash(read_json(policy_path)) == manifest["policy_fingerprint"]
    original_task_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", original["source_commit"], "--", "configs/forge/tasks"],
        cwd=ROOT, text=True).splitlines()
    for relative in original_task_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", original["source_commit"] + ":" + relative], cwd=ROOT))
    # Restore parent declarations from this publication's pinned source.
    # Current policy variants follow current task bindings and cannot supply
    # the original scorer contracts to a historical replay.
    for variant_path in (tmp_path / "configs/forge/task-variants").rglob("*.json"):
        variant = read_json(variant_path)
        parent = variant.get("execution", {}).get("policy_parent_definition")
        if parent is not None:
            relative = "configs/forge/tasks/" + parent["id"] + ".json"
            # Later scoped cohorts may name questions added after this
            # publication; those keep their current declarations here.
            if relative not in original_task_paths:
                continue
            archived_parent = __import__("json").loads(subprocess.check_output(
                ["git", "show", original["source_commit"] + ":" + relative], cwd=ROOT, text=True))
            atomic_json(tmp_path / relative, archived_parent)
    expected, all_snapshots, registered_rows, unregistered_shadows = {}, [], [], []
    snapshot_bytes = {}
    for entry in manifest["cohorts"]:
        snapshot_bytes[entry["snapshot"]] = (tmp_path / entry["snapshot"]).read_bytes()
        snapshot = read_json(tmp_path / entry["snapshot"])
        all_snapshots.extend(snapshot["rows"])
        for row in snapshot["rows"]:
            if row["candidate_id"] not in entry["candidates"]:
                continue
            recorded_at = entry["candidates"][row["candidate_id"]]
            # Completion-time registrations attest measured runtime rows. A
            # frozen all-backend roster also contains unmeasured CPU shadows
            # of the same CUDA config IDs; they are not registered evidence.
            if isinstance(recorded_at, str) and not row.get("attempt_ids"):
                unregistered_shadows.append(row)
                continue
            registered_rows.append(row)
            key = row["candidate_id"], row["runtime_cohort"]["execution_backend"]
            rank = recorded_at or ""
            if key not in expected or rank > expected[key][0]:
                expected[key] = rank, row
    # This fixture isolates cached reconstruction; separate tests exercise new
    # cards' live declaration resolution without needing a full Git checkout.
    for card in list((tmp_path / "configs/forge/ideas").glob("*.json")) + list((tmp_path / "configs/forge/configurations").glob("*.json")):
        if card.stem not in {key[0] for key in expected}:
            card.unlink()
    recorded = ("configs/forge/view-history/discriminator_stability-v2.json"
                if manifest["view_revision"] == 2 else None)
    result = read_json(publication.publish_current(tmp_path, recorded_policy=recorded)["json"])
    display_fields = {"technique", "publication_key", "qualification_input", "qualification_reuse",
                      "trainer_family", "configuration_id", "comparison_cohort", "selected_configuration", "alternative_scope"}
    science = lambda row: {key: value for key, value in row.items() if key not in display_fields}
    scientific_variants = [science(row) for row in result["configuration_rows"]]
    assert all(science(row) in scientific_variants for _, row in expected.values())
    assert all(science(row) not in scientific_variants for row in unregistered_shadows)
    assert all(row in [science(original) for original in all_snapshots] for row in scientific_variants)
    assert Counter(stable_hash(science(row)) for row in result["evidence_rows"]) == Counter(stable_hash(science(row)) for row in registered_rows)
    assert len(result["rows"]) == len({(row["trainer_family"], stable_hash(row["runtime_cohort"])) for row in result["configuration_rows"]})
    assert len(result["configuration_rows"]) >= len(expected)
    assert all((tmp_path / relative).read_bytes() == original for relative, original in snapshot_bytes.items())
    assert not (tmp_path / "reports/forge/attempts").exists()
    assert len(list((tmp_path / "reports/forge").glob("technique-inventory.md"))) == 1
    assert len(list((tmp_path / "reports/forge/families").glob("*.md"))) == (
        len(result["family_progress"]["families"]) + len(result["family_progress"].get("historical_families", [])) +
        len(result["family_progress"].get("configuration_families", [])))
    selection = _copy_word_diagnostics(tmp_path)
    updated = read_json(publication.publish_current(tmp_path)["json"])
    for key in ("rows", "configuration_rows", "evidence_rows", "archived_evidence_rows", "tier_requirements"):
        assert updated.get(key) == result.get(key)
    assert "task_diagnostics" not in updated
    assert "Five-word joint task diagnostics" not in (tmp_path / "reports/forge/technique-inventory.md").read_text()
    before = _outputs(tmp_path)
    times = {path: path.stat().st_mtime_ns for path in before}
    publication.publish_current(tmp_path)
    assert _outputs(tmp_path) == before
    assert times == {path: path.stat().st_mtime_ns for path in before}
    # A historical direct-task export is no longer an active solution input.
    (tmp_path / selection["recipes"][0]["receipt"]["path"]).write_text("{}\n")
    publication.publish_current(tmp_path)
    assert _outputs(tmp_path) == before
