"""Publication infrastructure fixtures; no training or scientific claims."""
from pathlib import Path
from copy import deepcopy
import base64
import json
import os
import shutil
from types import SimpleNamespace

import pytest

from experiments.forge.contracts import atomic_json, clone_json, file_hash, read_json, stable_hash
from experiments.forge import tier1_publication as publication
from experiments.forge.scoped_publications import REGISTRY, clock_audit_summary, load_publications
from experiments.forge.family_reports import build_progress, generated_pages
from experiments.forge.trainer_families import CURRENT_SELECTION, scientific_row_hash
from experiments.forge.views import load_tasks, load_view, task_execution_fingerprint, task_evaluation_fingerprint

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
DIGEST = "b" * 64


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@pytest.fixture(scope="module")
def _packet_template(tmp_path_factory):
    base = tmp_path_factory.mktemp("tier1-packet-template")
    shutil.copytree(ROOT / "configs/forge", base / "configs/forge", copy_function=_link_or_copy)
    # Publication renders validated implementation links. Keep the fixture's
    # local documentation sources available without changing their contracts.
    for card in (base / "configs/forge/family-documentation").glob("*.json"):
        documentation = read_json(card)
        sources = documentation.get("sources", [])
        if "illustration" in documentation:
            sources = [*sources, documentation["illustration"]["path"]]
        for source in sources:
            if "://" not in source:
                destination = base / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    _link_or_copy(ROOT / source, destination)
    from tests.archived_forge_contracts import restore_archived_contracts
    restore_archived_contracts(base, "tier1_completion")
    # The executed eleven-family registry predates maintained family prose and
    # tag declarations. Do not combine it with later editorial contracts.
    shutil.rmtree(base / "configs/forge/family-documentation")
    # This fixture represents the concluded eleven-family round. Later family
    # registrations must not silently expand its immutable scientific roster.
    frozen = read_json(base / publication.ROUND)
    frozen_families = {row["family"] for row in frozen["candidate_roster"]}
    assert len(frozen_families) == len(frozen["candidate_roster"]) == 11
    registry_path = base / "configs/forge/trainer-families.json"
    registry = read_json(registry_path)
    registry["families"] = [family for family in registry["families"] if family["id"] in frozen_families]
    assert {family["id"] for family in registry["families"]} == frozen_families
    atomic_json(registry_path, registry)
    choices = read_json(base / CURRENT_SELECTION)
    choices["selections"] = [pin for pin in choices["selections"] if pin["trainer_family"] in frozen_families]
    assert {pin["trainer_family"] for pin in choices["selections"]} == frozen_families
    atomic_json(base / CURRENT_SELECTION, choices)
    path = base / "configs/forge/views/discriminator_stability.json"
    main = read_json(path)
    if not any(row["task"] == "clockfree_audit_measurement_v1" for row in main["assignments"]):
        main["revision"] += 1
        main["assignments"].append({"task": "clockfree_audit_measurement_v1", "qualification_tier": 1,
                                    "importance": "diagnostic", "order": 100})
        atomic_json(path, main)
    from experiments.forge.planning import resolution_cache
    with resolution_cache(base):
        tasks = load_tasks(base)
        main_rows, policy_rows, roster, results, catalog = [], [], [], [], {}
        for pin in choices["selections"]:
            family, candidate = pin["trainer_family"], pin["candidate_id"]
            policy = family in publication.POLICY_FAMILIES
            view = load_view(base, publication.POLICY_VIEW if policy else publication.MAIN_VIEW)
            names = sorted(a["task"] for a in view["assignments"] if a["qualification_tier"] == 1)
            roster.append({"family": family, "candidate_id": candidate, "view": view["id"], "task_ids": names,
                           "measurement_tasks": [] if policy else ["clockfree_audit_measurement_v1"]})

            def row_for(selected_view, scoped=False):
                bindings = {}
                statuses = []
                for a in selected_view["assignments"]:
                    name = a["task"]
                    task = tasks[name]
                    contract = {"execution_sha256": task_execution_fingerprint(task),
                                "evaluation_sha256": task_evaluation_fingerprint(task),
                                "timeout_seconds": task["resources"]["timeout_seconds"],
                                "prior": task["execution"]["prior"], "sampling": task["evaluation"]}
                    digest = stable_hash(contract)
                    catalog[digest] = contract
                    bindings[name] = digest
                    if scoped:
                        status = "BLOCKED" if name.startswith(("unused_token_hold", "ae_gan_hold", "five_word")) else "FAIL"
                    else:
                        status = "BLOCKED" if policy else "FAIL" if a["qualification_tier"] == 1 else "UNKNOWN"
                    statuses.append({"task_id": name, "status": status, "role": a["importance"]})
                return {"candidate_id": candidate, "candidate_revision": stable_hash(candidate), "cohort": stable_hash(names),
                        "runtime_cohort": {"execution_backend": "cuda", "runtime": {"python": "fixture-runtime"},
                                           "compute_profiles": {"cuda": {"model": "fixture-gpu", "threads": 1}}}, "qualified_tier": 0,
                        "status": "INCOMPLETE" if scoped else "BLOCKED" if policy else "FAIL",
                        "attempt_ids": [family + "-policy-final"] if scoped else [] if policy else [family + "-final"],
                        "bindings": {"source_digest": DIGEST, "task_contracts": bindings},
                        "tasks": [{k: v for k, v in row.items() if k != "role"} for row in statuses if row["role"] == "required"],
                        "nonrequired_tasks": [{k: v for k, v in row.items() if k != "role"} for row in statuses if row["role"] != "required"],
                        "tiers": {"1": {"passed": 0, "total": sum(a["importance"] == "required" and a["qualification_tier"] == 1 for a in selected_view["assignments"])}}}
            main_row = row_for(main)
            main_rows.append(main_row)
            row = row_for(view, scoped=True) if policy else main_row
            if policy:
                policy_rows.append(row)
            grades = {t["task_id"]: t["status"] for t in row["tasks"] + row["nonrequired_tasks"]}
            results.append({"family": family, "candidate_id": candidate, "source": DIGEST, "source_commit": COMMIT,
                            "view": view["id"], "tasks": [{"task_id": name, "status": grades[name],
                                "attempt_id": row["attempt_ids"][0] if grades[name] in {"PASS", "FAIL"} else None,
                                "canonical_result_hash": stable_hash(family), "compatibility_key": stable_hash((family, name))}
                                for name in names]})

        def report(view, rows):
            return {"view": view["id"], "policy_fingerprint": stable_hash(view), "publication_scope": "frozen_source",
                    "frozen_source": {"commit": COMMIT}, "rows": rows, "task_contracts": catalog,
                    "provenance": {"input_digest": "fixture-only"}}
        definition = {"id": "tier1-completion-v1", "candidate_roster": roster}
        results_doc = {"round": "tier1-completion-v1", "candidates": results}
        main_rep = report(main, main_rows)
        scoped_rep = report(load_view(base, publication.POLICY_VIEW), policy_rows)
        # Synthetic rows need their own exact private pins; archived real evidence
        # must never be presented as the identity of these mocked measurements.
        card, _, _, _, _ = publication.prepare_selection(base, definition, results_doc, main_rep, scoped_rep)
        atomic_json(base / CURRENT_SELECTION, card)
    return base, definition, results_doc, main_rep, scoped_rep


@pytest.fixture
def packet(tmp_path, _packet_template):
    base, definition, results_doc, main_rep, scoped_rep = _packet_template
    shutil.copytree(base, tmp_path, dirs_exist_ok=True, copy_function=_link_or_copy)
    return (
        tmp_path,
        clone_json(definition),
        clone_json(results_doc),
        clone_json(main_rep),
        clone_json(scoped_rep),
    )


def select(packet):
    return publication.prepare_selection(*packet)


def test_complete_failures_are_measurements_and_policy_results_do_not_qualify_clean_parents(packet):
    root, _, _, main, _ = packet
    previous = read_json(root / CURRENT_SELECTION)
    card, clean, scoped, digest, commit = select(packet)
    assert len(clean) == 9 and set(scoped) == {"atlas", "e22"}
    assert digest == DIGEST and commit == COMMIT
    assert card["historical_selections"] == previous["historical_selections"]
    for pin in card["selections"]:
        assert pin["selection_kind"] == ("historical_incumbent" if pin["trainer_family"] in scoped else "current_measurement")
    assert all(row["qualified_tier"] == 0 for row in clean.values())
    assert all(scientific_row_hash(row) == scientific_row_hash(next(r for r in main["rows"] if r["candidate_id"] == row["candidate_id"])) for row in clean.values())


@pytest.mark.parametrize("change", ["clock_missing", "stale_evaluator", "other_source", "other_candidate", "policy_pass", "main_omitted", "policy_unmeasured"])
def test_incomplete_or_incompatible_evidence_cannot_make_current_measurement_pins(packet, change):
    root, definition, results, main, scoped = packet
    original_selection = (root / CURRENT_SELECTION).read_bytes()
    family = next(row for row in results["candidates"] if row["family"] not in publication.POLICY_FAMILIES)
    row = next(row for row in main["rows"] if row["candidate_id"] == family["candidate_id"])
    if change == "clock_missing":
        next(task for task in family["tasks"] if task["task_id"] == "clockfree_audit_measurement_v1")["status"] = "UNKNOWN"
    elif change == "stale_evaluator":
        digest = row["bindings"]["task_contracts"]["clockfree_audit_measurement_v1"]
        main["task_contracts"][digest]["evaluation_sha256"] = "old-law"
    elif change == "other_source":
        row["bindings"]["source_digest"] = "different-source"
    elif change == "other_candidate":
        family["candidate_id"] = "different-candidate"
    elif change == "main_omitted":
        frozen = next(item for item in definition["candidate_roster"] if item["family"] == family["family"])
        frozen["task_ids"].remove("gaussian1d_acquisition")
        family["tasks"] = [task for task in family["tasks"] if task["task_id"] != "gaussian1d_acquisition"]
    elif change == "policy_unmeasured":
        candidate = next(item for item in results["candidates"] if item["family"] == "atlas")
        next(task for task in candidate["tasks"] if task["task_id"].startswith("gaussian1d_"))["status"] = "UNKNOWN"
    else:
        next(row for row in main["rows"] if row["candidate_id"] == "atlas")["status"] = "PASS"
    with pytest.raises(ValueError):
        select(packet)
    assert (root / CURRENT_SELECTION).read_bytes() == original_selection


def add_media(packet):
    root, _, results, _, _ = packet
    _, clean, scoped, _, _ = select(packet)
    rows = {**clean, **scoped}
    items = []
    for candidate in results["candidates"]:
        row = rows[candidate["family"]]
        observed = [task for task in candidate["tasks"] if task["status"] in {"PASS", "FAIL"}]
        attempt = row["attempt_ids"][0]
        compact = {"candidate_id": row["candidate_id"], "candidate_revision": row["candidate_revision"],
                   "provenance": {"source_digest": DIGEST, "canonical_result_hash": stable_hash(candidate["family"])},
                   "task_results": [{"task_id": task["task_id"], "gate_status": task["status"],
                                     "compatibility_key": task["compatibility_key"]} for task in observed]}
        atomic_json(root / "reports/forge/technique-receipts" / (attempt + ".json"), compact)
        for task in observed:
            gif = root / "reports/forge/unit-fixture-media" / candidate["family"] / (task["task_id"] + ".gif")
            gif.parent.mkdir(parents=True, exist_ok=True)
            gif.write_bytes(base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="))
            receipt = {"task_id": task["task_id"], "recorded_grade": task["status"], "kind": "actual_training_saved_observations_gif",
                       "optimizer_updates_added": 0, "sampling_draws_added": 0, "observation_count": 1,
                       "gif_sha256": file_hash(gif)}
            atomic_json(gif.with_suffix(".json"), receipt)
            items.append({**receipt, "family": candidate["family"], "attempt_id": attempt, "gif": gif.relative_to(root).as_posix()})
    return rows, {"qualification_input": False, "items": items}


def test_all_final_measured_tasks_have_exact_attempt_bound_actual_training_media(packet):
    root, _, results, _, _ = packet
    rows, media = add_media(packet)
    verified = publication.validate_media(root, results, rows, media)
    assert len(verified) == 11
    assert len(media["items"]) == 71


@pytest.mark.parametrize("change", ["missing", "other_attempt", "changed_gif", "sidecar", "new_sampling", "result_hash"])
def test_missing_or_mismatched_media_prevents_publication(packet, change):
    root, _, results, _, _ = packet
    rows, media = add_media(packet)
    first = media["items"][0]
    if change == "missing":
        media["items"].pop(0)
    elif change == "other_attempt":
        first["attempt_id"] = "another-attempt"
    elif change == "changed_gif":
        (root / first["gif"]).write_bytes(b"changed")
    elif change == "sidecar":
        atomic_json((root / first["gif"]).with_suffix(".json"), {"forged": True})
    elif change == "new_sampling":
        first["sampling_draws_added"] = 1
    else:
        candidate = next(row for row in results["candidates"] if row["family"] == first["family"])
        next(task for task in candidate["tasks"] if task["task_id"] == first["task_id"])["canonical_result_hash"] = "another-result"
    with pytest.raises(ValueError):
        publication.validate_media(root, results, rows, media)


def test_publisher_rejection_restores_original_selection_and_removes_new_registry(packet):
    root = packet[0]
    card, _, _, _, _ = select(packet)
    original = (root / CURRENT_SELECTION).read_bytes()
    document_path = Path("reports/forge/unit-fixture-publication/scoped-evidence.json")
    manifest_path = Path("reports/forge/unit-fixture-manifest.json")
    atomic_json(root / manifest_path, {"policy_fingerprint": "old-policy"})
    def reject(*args, **kwargs):
        raise ValueError("independent source registration rejected")
    staged = {"root": str(root), "source_commit": COMMIT, "selection": card, "registry": {"test": "fixture"},
              "scoped_document": {"test": "fixture"}, "tracked_destinations": [CURRENT_SELECTION.as_posix(), REGISTRY.as_posix(), document_path.as_posix()]}
    publisher = SimpleNamespace(EVIDENCE_MANIFEST=manifest_path, publish_current=reject)
    with pytest.raises(ValueError, match="registration rejected"):
        publication.publish(staged, publisher=publisher)
    assert (root / CURRENT_SELECTION).read_bytes() == original
    assert not (root / REGISTRY).exists() and not (root / document_path).exists()


def staged_fixture(packet, monkeypatch):
    root, definition, results, main, scoped = packet
    _, media = add_media(packet)
    directory = root / "reports/forge/unit-fixture-publication"
    atomic_json(root / publication.ROUND, definition)
    atomic_json(directory / "results.json", results)
    atomic_json(directory / "media.json", media)
    monkeypatch.setattr(publication.subprocess, "check_output",
                        lambda command, **kwargs: json.dumps(definition) if command[1] == "show" else COMMIT + "\n")
    def reconstruct(root, *, view_id, output_prefix, **kwargs):
        path = Path(str(output_prefix) + ".json")
        atomic_json(path, main if view_id == publication.MAIN_VIEW else scoped)
        return {"json": str(path)}
    def audit(root, measured):
        return {**{key: measured[key] for key in ("family", "task_id", "attempt_id", "canonical_result_hash")},
                "recorded_grade": "FAIL", "qualification_input": False,
                "comparisons": {name: {"reference_sha256": "a" * 64, "changed_sha256": "b" * 64,
                                        "digest_equal": False}
                                for name in ("step_label", "horizon", "evaluation_cadence", "restart")},
                "clock_dependency_count": 1, "unexplained_clock_dependencies": ["fixture scheduled dependency"],
                "source_audit": {"source_sha256": {}, "allowed_state": [], "scope": "fixture only"}}
    monkeypatch.setattr(publication, "clock_audit_summary", audit)
    return publication.stage(root, COMMIT, directory, publisher=SimpleNamespace(regenerate=reconstruct))


def install_staged(packet, staged):
    root = packet[0]
    for relative, value in zip(staged["tracked_destinations"],
                               (staged["selection"], staged["registry"], staged["scoped_document"])):
        atomic_json(root / relative, value)


def display_fixture(packet):
    _, _, _, main, _ = packet
    families = {pin["candidate_id"]: pin["trainer_family"]
                for pin in read_json(packet[0] / CURRENT_SELECTION)["selections"]}
    for row in main["rows"]:
        family = families[row["candidate_id"]]
        row.update(trainer_family=family, technique=family)
    return {**main, "view_revision": load_view(packet[0], publication.MAIN_VIEW)["revision"]}


def test_stage_reconstructs_locally_before_any_selection_or_leaderboard_change(packet, monkeypatch):
    root = packet[0]
    selection_before = (root / CURRENT_SELECTION).read_bytes()
    leaderboard = root / "reports/forge/technique-inventory.md"
    leaderboard.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.write_text("existing current leaderboard\n")
    staged, directory = staged_fixture(packet, monkeypatch)
    assert directory.is_relative_to(root / "runs/forge")
    assert (directory / "publication-stage.json").is_file()
    assert (root / CURRENT_SELECTION).read_bytes() == selection_before
    assert not (root / REGISTRY).exists()
    assert not (root / staged["tracked_destinations"][2]).exists()
    assert leaderboard.read_text() == "existing current leaderboard\n"
    assert staged["measured_counts"] == {"FAIL": 71, "BLOCKED": 6}


def test_scoped_measurements_media_and_clock_evidence_do_not_fill_clean_parent_cells(packet, monkeypatch):
    root = packet[0]
    display = display_fixture(packet)
    staged, _ = staged_fixture(packet, monkeypatch)
    before = build_progress(root, display)
    install_staged(packet, staged)
    display["family_progress"] = build_progress(root, display)
    after = display["family_progress"]
    for old, new in zip(before["families"], after["families"]):
        assert new["cohorts"][0]["tiers"] == old["cohorts"][0]["tiers"]
        assert new["cohorts"][0]["total"] == old["cohorts"][0]["total"]
    atlas = next(family for family in after["families"] if family["id"] == "atlas")
    cohort = atlas["cohorts"][0]
    assert cohort["tasks"]["two_pole"]["status"] == "BLOCKED"
    assert cohort["tasks"]["two_pole_tier1_policy_selected_cloud_v1"]["status"] == "FAIL"
    tier1 = next(view for view in cohort["scoped_views"] if view["id"] == "tier1_policy_coverage")
    assert tier1["total"]["counts"] == {"FAIL": 4, "BLOCKED": 3}
    text = generated_pages(root, display)[root / atlas["page"]]
    assert "Actual-training GIF" in text and "no parent-cohort credit" in text
    assert "Recorded clock parity diagnostics" in text and "Recorded unexplained clock dependencies: **1**" in text
    assert "Certified parity digests and source audit" in text and "fixture scheduled dependency" in text
    assert len(load_publications(root)["final_measurements"]) == 71


def test_clock_display_summary_uses_all_certified_comparisons_without_regrading(tmp_path):
    row = {"task_id": "clockfree_audit_measurement_v1", "gate_status": "FAIL", "metrics": {},
           "evidence": {"comparisons": [{"condition": name, "reference_sha256": "a" * 64,
                "changed_sha256": ("b" if name == "step_label" else "a") * 64}
                for name in ("step_label", "horizon", "evaluation_cadence", "restart")],
                "source_audit": {"unexplained_clock_dependencies": ["scheduled horizon", "critic guard"],
                                 "source_sha256": {"particlegan/training.py": "c" * 64},
                                 "allowed_state": ["optimizer_moments"], "scope": "fixture only"}}}
    original = {"task_results": [row]}
    atomic_json(tmp_path / "reports/forge/attempts/clock-final/result.json", original)
    measured = {"family": "bcap", "task_id": row["task_id"], "attempt_id": "clock-final",
                "canonical_result_hash": stable_hash(original)}
    audit = clock_audit_summary(tmp_path, measured)
    assert audit["recorded_grade"] == "FAIL" and audit["clock_dependency_count"] == 2
    assert not audit["comparisons"]["step_label"]["digest_equal"]
    assert sum(check["digest_equal"] for check in audit["comparisons"].values()) == 3
    assert audit["source_audit"]["source_sha256"] == row["evidence"]["source_audit"]["source_sha256"]
    measured["canonical_result_hash"] = "another-result"
    with pytest.raises(ValueError, match="canonical result"):
        clock_audit_summary(tmp_path, measured)


def test_registered_media_receipt_changes_are_detected_before_rendering(packet, monkeypatch):
    staged, _ = staged_fixture(packet, monkeypatch)
    install_staged(packet, staged)
    root = packet[0]
    item = read_json(root / staged["registry"]["media"]["path"])["items"][0]
    atomic_json((root / item["gif"]).with_suffix(".json"), {"changed": True})
    with pytest.raises(ValueError, match="provenance receipt"):
        load_publications(root)


def test_final_attempt_owns_metrics_when_earlier_retry_receipts_are_retained(packet, monkeypatch):
    staged, _ = staged_fixture(packet, monkeypatch)
    install_staged(packet, staged)
    root = packet[0]
    display = display_fixture(packet)
    row = next(row for row in display["rows"] if row["trainer_family"] == "bcap")
    current = read_json(root / "reports/forge/technique-receipts/bcap-final.json")
    earlier = {**current, "task_results": [{**task, "gate_status": "INVALID", "reason": "earlier infrastructure failure"}
                                           for task in current["task_results"]]}
    atomic_json(root / "reports/forge/technique-receipts/bcap-earlier.json", earlier)
    row["attempt_ids"].append("bcap-earlier")
    # This control intentionally extends the mocked provenance roster. Bind
    # the private display pin to that exact row before testing final ownership.
    choices = read_json(root / CURRENT_SELECTION)
    pin = next(pin for pin in choices["selections"] if pin["trainer_family"] == "bcap")
    pin["scientific_row_sha256"] = scientific_row_hash(row)
    atomic_json(root / CURRENT_SELECTION, choices)
    display["family_progress"] = build_progress(root, display)
    cohort = next(family for family in display["family_progress"]["families"] if family["id"] == "bcap")["cohorts"][0]
    assert cohort["tasks"]["gaussian1d_acquisition"]["metrics"]["gate_status"] == "FAIL"
    assert cohort["tasks"]["gaussian1d_acquisition"]["receipt"].endswith("bcap-final.json")
    assert len(cohort["tasks"]["gaussian1d_acquisition"]["training_media"]) == 1


def test_scoped_results_cannot_replace_an_existing_parent_cell(packet, monkeypatch):
    staged, _ = staged_fixture(packet, monkeypatch)
    install_staged(packet, staged)
    root = packet[0]
    display = display_fixture(packet)
    atlas = next(row for row in display["rows"] if row["trainer_family"] == "atlas")
    atlas["nonrequired_tasks"].append({"task_id": "two_pole_tier1_policy_selected_cloud_v1", "status": "PASS"})
    with pytest.raises(ValueError, match="replace selected parent cells"):
        build_progress(root, display)


def frozen_rows():
    bound = {"candidate_id": "executed", "candidate_revision": "revision", "cohort": "cohort",
             "bindings": {"available": True, "source_digest": DIGEST}, "status": "FAIL", "qualified_tier": 0,
             "attempt_ids": ["certified-final"], "tasks": [{"task_id": "question", "status": "FAIL"}]}
    unresolved = {"candidate_id": "old-unresolved", "candidate_revision": None, "cohort": None,
                  "bindings": {"available": False, "task_contracts": {}, "task_keys_sha256": stable_hash({})},
                  "status": "BLOCKED", "qualified_tier": 0, "attempt_ids": [],
                  "blockers": [{"reason": "old control mapping does not bind the new diagnostic"}],
                  "cost": {"measured_tasks": 0, "wall_seconds": None},
                  "tasks": [{"task_id": "question", "status": "BLOCKED"}],
                  "nonrequired_tasks": [{"task_id": "clock_probe", "status": "BLOCKED"}]}
    return bound, unresolved


def test_unresolved_declarations_remain_explicit_without_source_or_attempt_credit():
    publisher = publication._publisher(ROOT)
    bound, unresolved = frozen_rows()
    report = {"rows": [bound], "configuration_rows": [bound, unresolved]}
    original = deepcopy(report)
    measured, diagnostics = publisher._bound_publication_rows(report, {DIGEST})
    assert measured == [bound] and diagnostics == [unresolved]
    assert report == original
    assert {attempt for row in measured for attempt in row["attempt_ids"]} == {"certified-final"}
    assert all(row["qualified_tier"] == 0 and not row["attempt_ids"] for row in diagnostics)


@pytest.mark.parametrize("change", ["other_source", "attempt", "pass_cell", "qualified_tier", "revision", "contract", "paid_measurement", "missing_blocker"])
def test_unresolved_diagnostic_exception_rejects_source_mismatch_or_scientific_credit(change):
    publisher = publication._publisher(ROOT)
    _, row = frozen_rows()
    if change == "other_source":
        row["bindings"]["source_digest"] = "different-actual-source"
    elif change == "attempt":
        row["attempt_ids"] = ["uncertified-attempt"]
    elif change == "pass_cell":
        row["tasks"][0]["status"] = "PASS"
    elif change == "qualified_tier":
        row["qualified_tier"] = 1
    elif change == "revision":
        row["candidate_revision"] = "a-scientific-revision"
    elif change == "contract":
        row["bindings"]["task_contracts"] = {"question": "scientific-task-contract"}
    elif change == "paid_measurement":
        row["cost"]["measured_tasks"] = 1
    else:
        row["blockers"] = []
    with pytest.raises(ValueError, match="absent from the validated original receipts"):
        publisher._bound_publication_rows({"rows": [row]}, {DIGEST})


@pytest.mark.parametrize("change", ["inactive_cpu", "active_cuda", "runtime", "backend", "missing_profile", "missing_runtime"])
def test_scoped_display_matches_only_exact_active_backend_and_runtime(packet, monkeypatch, change):
    root, _, _, main, scoped = packet
    runtime = {"execution_backend": "cuda", "runtime": {"python": "fixture-runtime"},
               "compute_profiles": {"cuda": {"model": "fixture-gpu", "threads": 1}}}
    parent = next(row for row in main["rows"] if row["candidate_id"] == "atlas")
    variant = next(row for row in scoped["rows"] if row["candidate_id"] == "atlas")
    parent["runtime_cohort"] = deepcopy(runtime)
    variant["runtime_cohort"] = deepcopy(runtime)
    if change == "inactive_cpu":
        parent["runtime_cohort"]["compute_profiles"]["cpu"] = {"model": "extra-inactive-cpu", "threads": 7}
    elif change == "active_cuda":
        variant["runtime_cohort"]["compute_profiles"]["cuda"]["model"] = "different-gpu"
    elif change == "runtime":
        variant["runtime_cohort"]["runtime"]["python"] = "different-runtime"
    elif change == "missing_profile":
        variant["runtime_cohort"]["compute_profiles"] = {}
        parent["runtime_cohort"]["compute_profiles"] = {}
    elif change == "missing_runtime":
        variant["runtime_cohort"].pop("runtime")
        parent["runtime_cohort"].pop("runtime")
    else:
        variant["runtime_cohort"]["execution_backend"] = "cpu"
        # Staging requires one exact CUDA source row, so this regression tests
        # the display comparison directly rather than admitting that row.
    if change == "backend":
        from experiments.forge.family_reports import _same_active_runtime
        assert not _same_active_runtime(variant["runtime_cohort"], parent["runtime_cohort"])
        return
    frozen_identity = scientific_row_hash(variant)
    staged, _ = staged_fixture(packet, monkeypatch)
    install_staged(packet, staged)
    display = display_fixture(packet)
    display["family_progress"] = build_progress(root, display)
    cohort = next(family for family in display["family_progress"]["families"] if family["id"] == "atlas")["cohorts"][0]
    clock = cohort["tasks"]["clockfree_audit_tier1_policy_selected_cloud_v1"]
    if change == "inactive_cpu":
        assert clock["status"] == "FAIL" and clock["current_contract"] == "matches"
        assert clock.get("clock_audit") and len(clock["training_media"]) == 1
        tier1 = next(view for view in cohort["scoped_views"] if view["id"] == "tier1_policy_coverage")
        assert tier1["total"]["counts"] == {"FAIL": 4, "BLOCKED": 3}
        assert cohort["scoped_publications"][0]["runtime_cohort_sha256"] == stable_hash(runtime)
        assert cohort["tasks"]["two_pole"]["status"] == "BLOCKED"
    else:
        assert clock["status"] == "UNKNOWN" and not clock.get("clock_audit") and not clock.get("training_media")
        assert not cohort["scoped_publications"]
    assert scientific_row_hash(variant) == frozen_identity


def test_first_write_snapshot_link_uses_validated_registration_before_file_exists(packet, monkeypatch):
    staged, _ = staged_fixture(packet, monkeypatch)
    install_staged(packet, staged)
    root = packet[0]
    display = display_fixture(packet)
    row = next(row for row in display["rows"] if row["trainer_family"] == "bcap")
    row["publication_key"] = "registered-snapshot"
    snapshot = "reports/forge/technique-evidence/pending-first-snapshot.json"
    display["evidence_sources"] = {row["publication_key"]: {"snapshot": snapshot, "json_sha256": "c" * 64}}
    display["family_progress"] = build_progress(root, display)
    family = next(family for family in display["family_progress"]["families"] if family["id"] == "bcap")
    page = root / family["page"]
    first = generated_pages(root, display)[page]
    assert not (root / snapshot).exists()
    assert "[Frozen numerical evidence](../technique-evidence/pending-first-snapshot.json)" in first
    atomic_json(root / snapshot, {"fixture": "validated pending snapshot"})
    assert generated_pages(root, display)[page] == first
