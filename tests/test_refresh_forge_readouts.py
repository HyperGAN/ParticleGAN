"""Unit and integration tests for single-command Forge readout refresh (ops/refresh_forge_readouts.py)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.forge import attempt_coverage
from experiments.forge.contracts import atomic_json, read_json
from ops import refresh_forge_readouts as rfr


ROOT = Path(__file__).resolve().parents[1]


def _assert_current_source_scope(report, root):
    measured = report["current_source_measurements"]
    assert measured["scope"] == "current_source_measurements"
    assert measured["source_digest"] == rfr.catalog_source_manifest(root)["digest"]
    if measured["stages"]["scaling_performance"]["stale_source_cells"]:
        assert measured["status"] == "NEEDS_REFRESH"
        assert "scaling_performance" in measured["stale_stages"]
    assert report["scope"] == "published_reports" and report["qualification_reuse"] is False
    return measured


def test_record_to_state_job_losslessly_round_trips_new_attempt_through_coverage_value():
    published_record = {
        "key": "abc123",
        "family": "atlas",
        "candidate_id": "atlas-particle-normal-v4",
        "candidate_revision": "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd",
        "source_digest": "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf",
        "task": "mode_hold",
        "execution_sha256": "exec-sha",
        "evaluation_sha256": "eval-sha",
        "status": "PASS",
        "attempted": True,
        "origin": "new_attempt",
        "diagnostic": True,
        "evidence": "reports/forge/attempt-coverage.json",
        "original_attempt": "/tmp/runs/forge/attempt-coverage-refresh/attempts/abc123/result.json",
        "updates": 4000,
    }
    retained_record = {
        "key": "def456",
        "family": "atlas",
        "candidate_id": "atlas-particle-normal-v4",
        "candidate_revision": "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd",
        "source_digest": "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf",
        "task": "gaussian1d_smoke",
        "execution_sha256": "exec-sha-2",
        "evaluation_sha256": "eval-sha-2",
        "status": "PASS",
        "attempted": True,
        "origin": "retained_selected_row",
        "diagnostic": False,
        "evidence": "reports/forge/technique-inventory.json",
    }

    state = {
        "jobs": {
            "abc123": rfr._record_to_state_job(published_record),
            "def456": rfr._record_to_state_job(retained_record),
        }
    }
    recomputed = attempt_coverage.coverage_value(state)
    by_key = {row["key"]: row for row in recomputed["records"]}

    assert by_key["abc123"] == published_record
    assert (
        by_key["abc123"]["original_attempt"]
        == "/tmp/runs/forge/attempt-coverage-refresh/attempts/abc123/result.json"
    )
    assert by_key["def456"]["evidence"] == "reports/forge/technique-inventory.json"
    assert "original_attempt" not in by_key["def456"]


def test_check_queue_status_handles_missing_active_valid_awaiting_and_stale_dangling(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        rfr,
        "catalog_source_manifest",
        lambda root, idea=None: {"digest": "live-digest-0001"},
    )

    # 1. Empty / non-existent queue root is current.
    empty_status = rfr.check_queue_status(tmp_path, tmp_path / "runs/forge")
    assert empty_status == {
        "current": True,
        "active_submissions": [],
        "awaiting_readout_submissions": [],
    }

    # 2. Active submission or valid live-source awaiting_readout marks queue non-current without error.
    state_path = tmp_path / "runs/forge/queue/state.json"
    atomic_json(
        state_path,
        {
            "jobs": {"job-1": {"result": {"task_results": []}}},
            "submissions": {
                "req-active": {
                    "status": "running",
                    "lifecycle": "running",
                    "request": {
                        "candidate": {"id": "cand-a"},
                        "source": {"digest": "live-digest-0001"},
                        "jobs": [],
                    },
                },
                "req-awaiting": {
                    "status": "completed",
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "cand-b"},
                        "source": {"digest": "live-digest-0001"},
                        "jobs": [{"compatibility_key": "job-1"}],
                    },
                },
            },
        },
    )
    busy_status = rfr.check_queue_status(tmp_path, tmp_path / "runs/forge")
    assert busy_status["current"] is False
    assert len(busy_status["active_submissions"]) == 1
    assert len(busy_status["awaiting_readout_submissions"]) == 1

    # 3. Superseded-source awaiting_readout with zero durable results raises CodeChangeRequiredError.
    atomic_json(
        state_path,
        {
            "jobs": {},
            "submissions": {
                "req-dangling": {
                    "status": "completed",
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "cand-stale"},
                        "source": {"digest": "superseded-digest-9999"},
                        "jobs": [{"compatibility_key": "missing-job"}],
                    },
                }
            },
        },
    )
    with pytest.raises(rfr.CodeChangeRequiredError, match="superseded source"):
        rfr.check_queue_status(tmp_path, tmp_path / "runs/forge")


def test_retry_shared_gpu_timeouts_retries_only_single_attempt_worktree_timeouts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        rfr,
        "_request_matches_worktree",
        lambda root, req: req.get("source", {}).get("digest") == "live-src",
    )
    queue_root = tmp_path / "runs/forge"
    atomic_json(
        queue_root / "queue/state.json",
        {
            "jobs": {
                "job-timeout-1": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}],
                    "result": {"raw": {"attempt_status": "timeout"}},
                },
                "job-timeout-2": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}, {"attempt_id": "a2"}],
                    "result": {"raw": {"attempt_status": "timeout"}},
                },
                "job-error-1": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}],
                    "result": {"raw": {"attempt_status": "error"}},
                },
                "job-stale-1": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}],
                    "result": {"raw": {"attempt_status": "timeout"}},
                },
            },
            "submissions": {
                "sub-live": {
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "recipe-prior-refactor-bcap-pure-v1"},
                        "source": {"digest": "live-src"},
                        "jobs": [
                            {"compatibility_key": "job-timeout-1", "task_id": "five_word_joint_smoke"},
                            {"compatibility_key": "job-timeout-2", "task_id": "grid100"},
                            {"compatibility_key": "job-error-1", "task_id": "two_pole"},
                        ],
                    },
                },
                "sub-stale": {
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "old-cand"},
                        "source": {"digest": "stale-src"},
                        "jobs": [{"compatibility_key": "job-stale-1", "task_id": "five_word_joint_smoke"}],
                    },
                },
            },
        },
    )

    retried_calls: list[tuple[str, str]] = []

    class _FakeQueue:
        def retry(self, key: str, *, reason: str):
            retried_calls.append((key, reason))

    retried = rfr._retry_shared_gpu_timeouts(_FakeQueue(), queue_root, tmp_path, json_mode=True)
    assert retried == ["job-timeout-1"]
    assert len(retried_calls) == 1
    assert retried_calls[0][0] == "job-timeout-1"
    assert "workers_per_gpu=1" in retried_calls[0][1]


def test_check_scaling_performance_detects_missing_corrupt_multicohort_and_stale_source(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [
            {"family": "atlas", "trainer_family": "atlas", "candidate_id": "atlas-particle-normal-v4"},
            {
                "family": "bcap-pure",
                "trainer_family": "bcap-dualnorm",
                "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            },
        ],
    )
    monkeypatch.setattr(
        rfr,
        "load_view",
        lambda root, name: {"assignments": [{"task": "dcgan_performance_r32"}]},
    )
    live_files_sha = {"particlegan/__init__.py": "sha-live"}
    monkeypatch.setattr(rfr, "scaling_source_files_sha256", lambda root: live_files_sha)

    # 1. Missing manifest -> needs refresh (not an exception).
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: None)
    missing = rfr.check_scaling_performance(tmp_path)
    assert missing["current"] is False
    assert missing["reason"] == "missing_manifest"
    assert missing["missing_cells"] == 2

    # 2. Invalid manifest -> raises CodeChangeRequiredError.
    def _bad_load(root: Path):
        raise ValueError("corrupt performance manifest")

    monkeypatch.setattr(rfr.scaling_publication, "load_performance", _bad_load)
    with pytest.raises(rfr.CodeChangeRequiredError, match="invalid scaling-performance.json"):
        rfr.check_scaling_performance(tmp_path)

    # 3. Complete single-cohort manifest with inventory trainer_family ('bcap-dualnorm') -> current=True.
    def _make_record(fam: str, cid: str, gpu_uuid: str, files_sha: dict[str, str], status: str = "COMPLETE"):
        return {
            "trainer_family": fam,
            "candidate_id": cid,
            "task_id": "dcgan_performance_r32",
            "result": {
                "status": status,
                "identity": {
                    "source": {"sha256": "src-sha-1", "files_sha256": files_sha},
                    "runtime_sha256": "rt-sha-1",
                    "hardware": {"backend": "cuda", "gpu": {"uuid": gpu_uuid, "name": "RTX A6000"}},
                },
            },
        }

    monkeypatch.setattr(rfr.scaling_publication, "performance_status", lambda res: res["status"])
    monkeypatch.setattr(
        rfr.scaling_publication,
        "hardware_key",
        lambda res: res["identity"]["hardware"],
    )

    good_manifest = {
        "unavailable": [],
        "records": [
            _make_record("atlas", "atlas-particle-normal-v4", "GPU-0", live_files_sha),
            _make_record("bcap-dualnorm", "recipe-prior-refactor-bcap-pure-v1", "GPU-0", live_files_sha),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: good_manifest)
    good = rfr.check_scaling_performance(tmp_path)
    assert good["current"] is True
    assert good["table_cohorts"] == 1

    # 4. Split across GPU-0 and GPU-1 -> table_cohorts == 2 -> current=False (prevents UNKNOWN table split).
    split_manifest = {
        "unavailable": [],
        "records": [
            _make_record("atlas", "atlas-particle-normal-v4", "GPU-0", live_files_sha),
            _make_record("bcap-dualnorm", "recipe-prior-refactor-bcap-pure-v1", "GPU-1", live_files_sha),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: split_manifest)
    split = rfr.check_scaling_performance(tmp_path)
    assert split["current"] is False
    assert split["published_source_current"] is False
    assert split["table_cohorts"] == 2

    # 5. Stale code file digest -> stale_source_cells == 1 -> current=False.
    stale_manifest = {
        "unavailable": [],
        "records": [
            _make_record("atlas", "atlas-particle-normal-v4", "GPU-0", {"particlegan/__init__.py": "old"}),
            _make_record("bcap-dualnorm", "recipe-prior-refactor-bcap-pure-v1", "GPU-0", live_files_sha),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: stale_manifest)
    stale = rfr.check_scaling_performance(tmp_path)
    assert stale["current"] is False
    assert stale["stale_source_cells"] == 1
    assert stale["published_source_current"] is True


@pytest.mark.parametrize("broken_stage", [None, "compiled_reports", "scaling_performance", "selection"])
def test_published_check_keeps_source_drift_separate_and_rejects_broken_originals(
    tmp_path, monkeypatch, capsys, broken_stage
):
    strict = {
        "status": "NEEDS_REFRESH", "scope": "current_source_measurements", "source_digest": "latest-source",
        "stale_stages": ["scaling_performance", "selected_family_readouts"],
        "stages": {
            "queue": {"current": True}, "attempt_coverage": {"current": False},
            "retained_diagnostic_integrity": {"current": True},
            "compiled_reports": {"current": broken_stage != "compiled_reports"},
            "scaling_performance": {"current": False, "stale_source_cells": 48,
                                    "published_source_current": broken_stage != "scaling_performance"},
            "selected_family_readouts": {"current": False, "missing_families": [{"family": "original-family"}]},
        },
    }
    original = deepcopy(strict)
    monkeypatch.setattr(rfr, "check_status", lambda *args, **kwargs: strict)
    monkeypatch.setattr(rfr, "_registered_evidence_satisfies_selection", lambda root: broken_stage != "selection")
    assert rfr.main(["--root", str(tmp_path), "--check", "--reports-only", "--json"]) == (
        rfr.CURRENT_EXIT if broken_stage is None else rfr.NEEDS_REFRESH_EXIT)
    report = json.loads(capsys.readouterr().out)
    assert report["scope"] == "published_reports" and report["qualification_reuse"] is False
    assert report["current_source_measurements"] == original
    assert report["stages"]["published_scaling_performance"]["stale_source_cells"] == 48
    assert strict == original
    assert not (tmp_path / "runs").exists()
    assert rfr.main(["--root", str(tmp_path), "--check", "--json"]) == rfr.NEEDS_REFRESH_EXIT
    assert json.loads(capsys.readouterr().out) == original


def test_selected_family_preserved_absent_paths_reads_snapshot_closure(tmp_path: Path):
    snap = tmp_path / "reports/forge/technique-evidence/snap.json"
    atomic_json(
        snap,
        {
            "frozen_catalog_support_closure": {
                "preserved_absent_paths": ["reports/transfer_suite/unadjusted/leading_profile.json"]
            }
        },
    )
    pub = {"evidence_sources": {"pk-1": {"snapshot": "reports/forge/technique-evidence/snap.json"}}}
    paths = rfr._selected_family_preserved_absent_paths(tmp_path, pub, {"publication_key": "pk-1"})
    assert paths == ["reports/transfer_suite/unadjusted/leading_profile.json"]
    assert rfr._selected_family_preserved_absent_paths(tmp_path, pub, {"publication_key": "missing"}) == []


def test_pin_checkpoint_pair_gpus_and_expected_preflight_refusals(tmp_path: Path, monkeypatch):
    # 0. _frozen_preflight_blockers forwards PYTHONPATH=frozen_checkout and parses JSON output.
    captured_env: dict[str, str] = {}

    class _FakeCompleted:
        stdout = '{"misgan_block_recovery": ["misgan_block_recovery: unsupported vector target declaration"]}\n'

    def _fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeCompleted()

    monkeypatch.setattr(rfr.subprocess, "run", _fake_run)
    parsed_blockers = rfr._frozen_preflight_blockers(tmp_path, "atlas-particle-normal-v4")
    assert captured_env.get("PYTHONPATH") == str(tmp_path)
    assert parsed_blockers == {
        "misgan_block_recovery": ["misgan_block_recovery: unsupported vector target declaration"]
    }

    # 1. Fresh PENDING producer -> consumer -> grandchild chain pins all three to pair_gpus[0].
    fresh_jobs: dict[str, dict[str, Any]] = {
        "k1": {"family": "atlas", "task": "gaussian1d_smoke", "status": "PENDING", "dependencies": []},
        "k2": {
            "family": "atlas",
            "task": "gaussian1d_stability",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "gaussian1d_smoke"}],
        },
        "k3": {
            "family": "atlas",
            "task": "gaussian1d_stage3",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "gaussian1d_stability"}],
        },
    }
    rfr._pin_checkpoint_pair_gpus(fresh_jobs, "atlas", ["0", "1"])
    assert fresh_jobs["k1"]["required_gpu"] == "0"
    assert fresh_jobs["k2"]["required_gpu"] == "0"
    assert fresh_jobs["k3"]["required_gpu"] == "0"

    # 2. Completed producer on gpu='1' pins PENDING consumer to '1' without mutating completed producer.
    resumed_jobs: dict[str, dict[str, Any]] = {
        "k1": {"family": "atlas", "task": "gaussian1d_smoke", "status": "PASS", "gpu": "1", "dependencies": []},
        "k2": {
            "family": "atlas",
            "task": "gaussian1d_stability",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "gaussian1d_smoke"}],
        },
    }
    rfr._pin_checkpoint_pair_gpus(resumed_jobs, "atlas", ["0", "1"])
    assert "required_gpu" not in resumed_jobs["k1"]
    assert resumed_jobs["k2"]["required_gpu"] == "1"

    # 3. Certified queue parent with device='cuda:1' pins PENDING consumer to '1'.
    cert_jobs: dict[str, dict[str, Any]] = {
        "k1": {
            "family": "atlas",
            "task": "grid100",
            "status": "PASS",
            "certified_parent": {"result": {"device": "cuda:1"}},
            "dependencies": [],
        },
        "k2": {
            "family": "atlas",
            "task": "grid100_14k",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "grid100"}],
        },
    }
    rfr._pin_checkpoint_pair_gpus(cert_jobs, "atlas", ["0", "1"])
    assert cert_jobs["k2"]["required_gpu"] == "1"

    # 4. Cross-GPU RNG mismatch vs static preflight blocker recognition.
    rng_err = {
        "family": "atlas",
        "task": "gaussian1d_stability",
        "status": "ERROR",
        "reason": "ValueError: trainer/named RNG checkpoint mismatch",
    }
    preflight_err = {
        "family": "atlas",
        "task": "misgan_block_recovery",
        "status": "ERROR",
        "reason": "ValueError: unsupported vector target declaration",
    }
    preflight_map = {
        ("atlas", "misgan_block_recovery"): ["misgan_block_recovery: unsupported vector target declaration"],
        ("atlas", "gaussian1d_stability"): [],
    }
    assert rfr._is_cross_gpu_rng_mismatch(rng_err) is True
    assert rfr._is_cross_gpu_rng_mismatch(preflight_err) is False
    assert rfr._is_expected_preflight_refusal(preflight_err, preflight_map) is True
    assert rfr._is_expected_preflight_refusal(rng_err, preflight_map) is False


def test_check_attempt_coverage_flags_superseded_selected_records_and_ignores_scoped_records(
    tmp_path: Path, monkeypatch
):
    pub_path = tmp_path / "reports/forge/technique-inventory.json"
    cov_path = tmp_path / "reports/forge/attempt-coverage.json"

    selection = {
        "candidate_id": "atlas-particle-normal-v4",
        "candidate_revision": "rev-v4",
        "publication_key": "pub-atlas",
        "bindings": {"source_digest": "src-v4"},
    }
    atomic_json(
        pub_path,
        {"evidence_sources": {"pub-atlas": {"source_commit": "deadbeef"}}},
    )
    monkeypatch.setattr(
        rfr.attempt_coverage,
        "selected_tasks",
        lambda pub: [("atlas", selection, ["two_pole", "mode_hold"])],
    )
    monkeypatch.setattr(
        rfr,
        "_git_task_names_at_commit",
        lambda root, commit: {"two_pole", "mode_hold"},
    )

    # Old superseded non-scoped record ('atlas' instead of 'atlas-particle-normal-v4') + missing 'mode_hold'.
    atomic_json(
        cov_path,
        {
            "summary": {"pending": 0, "running": 0},
            "records": [
                {
                    "family": "atlas",
                    "candidate_id": "atlas",
                    "candidate_revision": "rev-old",
                    "source_digest": "src-old",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "BLOCKED",
                },
                {
                    "family": "research:legacy",
                    "research_scope": "seed0_leaderboard_research",
                    "candidate_id": "old-research",
                    "candidate_revision": "rev-res",
                    "source_digest": "src-res",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "FAIL",
                },
            ],
        },
    )
    stale = rfr.check_retained_diagnostic_integrity(tmp_path)
    assert stale["current"] is False
    assert stale["superseded_record_count"] == 1
    assert stale["missing_by_family"] == {
        "atlas": {
            "candidate_id": "atlas-particle-normal-v4",
            "missing_tasks": ["mode_hold", "two_pole"],
        }
    }

    # Replace superseded record with current v4 records covering both tasks while keeping scoped research record.
    atomic_json(
        cov_path,
        {
            "summary": {"pending": 0, "running": 0},
            "records": [
                {
                    "family": "atlas",
                    "candidate_id": "atlas-particle-normal-v4",
                    "candidate_revision": "rev-v4",
                    "source_digest": "src-v4",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "FAIL",
                },
                {
                    "family": "atlas",
                    "candidate_id": "atlas-particle-normal-v4",
                    "candidate_revision": "rev-v4",
                    "source_digest": "src-v4",
                    "task": "mode_hold",
                    "attempted": True,
                    "status": "PASS",
                },
                {
                    "family": "research:legacy",
                    "research_scope": "seed0_leaderboard_research",
                    "candidate_id": "old-research",
                    "candidate_revision": "rev-res",
                    "source_digest": "src-res",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "FAIL",
                },
            ],
        },
    )
    fresh = rfr.check_retained_diagnostic_integrity(tmp_path)
    assert fresh["current"] is True
    assert fresh["superseded_record_count"] == 0
    assert fresh["missing_by_family"] == {}


def test_check_selected_family_readouts_handles_cohort_dedup_unknown_boards_and_task_errors(
    tmp_path: Path, monkeypatch
):
    rev = "77765918f0b8e49975abeef0068a12cef5c452594aa5d8f94003f0eaedfd3f52"
    fake_req = {
        "candidate_revision": rev,
        "view": {
            "assignments": [
                {"task": "gaussian1d_smoke", "qualification_tier": 1, "importance": "required"},
                {"task": "two_pole", "qualification_tier": 1, "importance": "required"},
                {"task": "mode_hold", "qualification_tier": 2, "importance": "required"},
            ]
        },
        "tasks": {
            "gaussian1d_smoke": {"preflight_blockers": [], "dependencies": []},
            "two_pole": {"preflight_blockers": [], "dependencies": []},
            "mode_hold": {"preflight_blockers": [], "dependencies": []},
        },
    }
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [{"family": "bcap-pure", "candidate_id": "recipe-prior-refactor-bcap-pure-v1"}],
    )
    monkeypatch.setattr(rfr.planning, "resolve_idea", lambda *args, **kwargs: deepcopy(fake_req))

    # 1. No readout record yet -> missing_families populated.
    initial = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert initial["current"] is False
    assert len(initial["missing_families"]) == 1

    # 2. Concluded readout exists (FAIL on two_pole stops Tier 2), and cohort-deduplicated leaderboard rows exist.
    readout_path = tmp_path / "reports/forge/records/readout-test.json"
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {"scope": {"view": "discriminator_stability", "execution_backend": "cuda"}},
            "task_results": [
                {"task_id": "gaussian1d_smoke", "gate_status": "PASS"},
                {"task_id": "two_pole", "gate_status": "FAIL"},
            ],
        },
    )
    lb_dir = tmp_path / "reports/forge/leaderboards"
    lb_dir.mkdir(parents=True, exist_ok=True)
    for view_id in rfr.ORDINARY_LEADERBOARD_VIEWS:
        (lb_dir / f"{view_id}.md").write_text(
            f"| bcap-prior-coulomb-control-v1 / {rev[:12]} / cuda (NVIDIA RTX A6000) | current | concluded | FAIL | Tier 0 |\n"
        )
    measured = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert measured["current"] is True

    # 3. An '| unknown |' status on one goal leaderboard flags stale_leaderboards.
    (lb_dir / "adaptation.md").write_text(
        f"| bcap-prior-coulomb-control-v1 / {rev[:12]} / cuda (NVIDIA RTX A6000) | current | concluded | unknown | Tier 0 |\n"
    )
    stale_lb = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert stale_lb["current"] is False
    assert stale_lb["stale_leaderboards"] == [
        {
            "family": "bcap-pure",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "view": "adaptation",
        }
    ]

    # 3b. A retried task containing both INCOMPLETE (attempt 1) and PASS (attempt 2) is accepted.
    for view_id in rfr.ORDINARY_LEADERBOARD_VIEWS:
        (lb_dir / f"{view_id}.md").write_text(
            f"| bcap-prior-coulomb-control-v1 / {rev[:12]} / cuda (NVIDIA RTX A6000) | current | concluded | FAIL | Tier 0 |\n"
        )
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {"scope": {"view": "discriminator_stability", "execution_backend": "cuda"}},
            "task_results": [
                {"task_id": "gaussian1d_smoke", "gate_status": "INCOMPLETE"},
                {"task_id": "gaussian1d_smoke", "gate_status": "PASS"},
                {"task_id": "two_pole", "gate_status": "FAIL"},
            ],
        },
    )
    retried_ok = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert retried_ok["current"] is True

    # 4. Unresolved ERROR / INCOMPLETE in current-source readout raises CodeChangeRequiredError.
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {"scope": {"view": "discriminator_stability", "execution_backend": "cuda"}},
            "task_results": [
                {"task_id": "gaussian1d_smoke", "gate_status": "ERROR"},
            ],
        },
    )
    with pytest.raises(rfr.CodeChangeRequiredError, match="has ERROR on gaussian1d_smoke"):
        rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")


def test_main_cli_exit_codes_check_noop_reports_only_and_code_change_required(
    tmp_path: Path, monkeypatch, capsys
):
    current_payload: dict[str, Any] = {
        "status": "CURRENT",
        "source_digest": "2e8a97a0aa84",
        "stale_stages": [],
        "stages": {
            "queue": {"current": True},
            "scaling_performance": {"current": True},
            "attempt_coverage": {"current": True},
            "selected_family_readouts": {"current": True},
            "compiled_reports": {"current": True},
        },
    }
    monkeypatch.setattr(rfr, "check_status", lambda *args, **kwargs: deepcopy(current_payload))

    assert rfr.main(["--root", str(tmp_path), "--check", "--json"]) == rfr.CURRENT_EXIT
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["status"] == "CURRENT"

    assert rfr.main(["--root", str(tmp_path), "--json"]) == rfr.CURRENT_EXIT
    noop = json.loads(capsys.readouterr().out)
    assert noop["action"] == "NOOP"

    stale_gpu_payload = deepcopy(current_payload)
    stale_gpu_payload["status"] = "NEEDS_REFRESH"
    stale_gpu_payload["stale_stages"] = ["scaling_performance"]
    stale_gpu_payload["stages"]["scaling_performance"] = {"current": False}
    monkeypatch.setattr(rfr, "check_status", lambda *args, **kwargs: deepcopy(stale_gpu_payload))
    monkeypatch.setattr(rfr, "check_compiled_reports", lambda root: {"current": True})

    assert rfr.main(["--root", str(tmp_path), "--check"]) == rfr.NEEDS_REFRESH_EXIT
    capsys.readouterr()
    assert rfr.main(["--root", str(tmp_path), "--reports-only", "--json"]) == rfr.NEEDS_REFRESH_EXIT
    partial = json.loads(capsys.readouterr().out)
    assert partial["action"] == "REPORTS_ONLY_PARTIAL"

    def _boom(*args, **kwargs):
        raise rfr.CodeChangeRequiredError("simulated adapter failure")

    monkeypatch.setattr(rfr, "check_status", _boom)
    assert rfr.main(["--root", str(tmp_path), "--check", "--json"]) == rfr.CODE_CHANGE_EXIT
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["status"] == "CODE_CHANGE_REQUIRED"
    assert "simulated adapter failure" in err_payload["error"]


def test_live_repository_forge_readouts_are_current_and_source_clean():
    report = rfr.check_published_status(ROOT)
    assert report["status"] == "CURRENT", json.dumps(report, indent=2, sort_keys=True)
    assert report["stale_stages"] == []
    measured = _assert_current_source_scope(report, ROOT)
    assert report["stages"]["published_scaling_performance"]["complete_cells"] == 60
    assert report["stages"]["published_scaling_performance"]["missing_cells"] == 0
    assert report["stages"]["published_scaling_performance"]["unavailable_cells"] == 0
    assert report["stages"]["published_scaling_performance"]["table_cohorts"] == 1
    assert report["stages"]["retained_diagnostic_integrity"]["pending"] == 0
    assert report["stages"]["retained_diagnostic_integrity"]["superseded_record_count"] == 0
    assert report["stages"]["retained_diagnostic_integrity"]["missing_by_family"] == {}
    assert report["stages"]["registered_selected_publications"]["current"] is True
    assert report["stages"]["compiled_reports"]["unknown_performance_rows"] == 0
    assert report["stages"]["compiled_reports"]["selected_diagnostic_rows_present"] is True
    assert report["stages"]["compiled_reports"]["clean_checkout_safe"] is True
    assert rfr.main(["--root", str(ROOT), "--check", "--reports-only"]) == rfr.CURRENT_EXIT
    assert rfr.main(["--root", str(ROOT), "--check"]) == (
        rfr.CURRENT_EXIT if measured["status"] == "CURRENT" else rfr.NEEDS_REFRESH_EXIT)


def test_clean_checkout_preserves_leaderboards_and_repairs_missing_attempts(
    tmp_path: Path, monkeypatch
):
    from reports.forge import regenerate_technique_inventory

    monkeypatch.setattr(
        regenerate_technique_inventory,
        "refresh_publication",
        lambda root: {"report": str(Path(root) / "reports/forge/technique-inventory.md")},
    )
    monkeypatch.setattr(rfr, "_ensure_tier_report_markdown", lambda root: None)
    monkeypatch.setattr(rfr, "_ensure_attempt_coverage_markdown", lambda root, json_mode=False: None)
    monkeypatch.setattr(rfr.history, "validate_inventory", lambda root: {"valid": True})

    compile_calls: list[bool] = []

    def _fake_compile_memory(root: Path, *, summaries_only: bool = False):
        compile_calls.append(bool(summaries_only))
        if not summaries_only:
            lb_path = Path(root) / "reports/forge/leaderboards/adaptation.md"
            lb_path.parent.mkdir(parents=True, exist_ok=True)
            lb_path.write_text("WIPED_WITHOUT_ATTEMPTS\n")
        return {"summaries_only": bool(summaries_only)}

    monkeypatch.setattr(rfr.knowledge, "compile_memory", _fake_compile_memory)

    lb_file = tmp_path / "reports/forge/leaderboards/adaptation.md"
    lb_file.parent.mkdir(parents=True, exist_ok=True)
    lb_file.write_text("COMMITTED_LEADERBOARD_CONTENT\n")

    # 1. Clean checkout (local attempt envelopes absent) skips summaries_only=False and preserves leaderboards.
    monkeypatch.setattr(
        rfr,
        "check_selected_family_readouts",
        lambda root, *, cuda_model=rfr.DEFAULT_CUDA_MODEL: {
            "current": True,
            "missing_families": [],
            "stale_leaderboards": [],
            "local_attempts_missing_families": [
                {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "candidate_revision": "rev"}
            ],
        },
    )
    res_clean = rfr.refresh_compiled_reports(tmp_path)
    assert res_clean["leaderboards_recompiled"] is False
    assert compile_calls == [True]
    assert lb_file.read_text() == "COMMITTED_LEADERBOARD_CONTENT\n"

    # 2. Local attempt envelopes present -> runs full compile_memory(summaries_only=False) then summaries_only=True.
    compile_calls.clear()
    monkeypatch.setattr(
        rfr,
        "check_selected_family_readouts",
        lambda root, *, cuda_model=rfr.DEFAULT_CUDA_MODEL: {
            "current": True,
            "missing_families": [],
            "stale_leaderboards": [],
            "local_attempts_missing_families": [],
        },
    )
    res_full = rfr.refresh_compiled_reports(tmp_path)
    assert res_full["leaderboards_recompiled"] is True
    assert compile_calls == [False, True]

    # 3. Stale leaderboard on clean checkout causes refresh_selected_family_readouts to target families
    #    missing local attempt envelopes so board() will not wipe them.
    resolved_cids: list[str] = []
    monkeypatch.setattr(
        rfr,
        "check_selected_family_readouts",
        lambda root, *, cuda_model=rfr.DEFAULT_CUDA_MODEL: {
            "current": False,
            "missing_families": [],
            "stale_leaderboards": [
                {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "view": "adaptation"}
            ],
            "local_attempts_missing_families": [
                {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "candidate_revision": "rev1"},
                {"family": "bispf", "candidate_id": "bispf-optimized-quintic-v1", "candidate_revision": "rev2"},
            ],
        },
    )
    monkeypatch.setattr(rfr, "catalog_source_manifest", lambda root, idea=None: {"digest": "2e8a97a0aa84" * 5 + "0000"})

    def _fake_resolve(root, cid, **kwargs):
        resolved_cids.append(cid)
        return {
            "request_id": f"req-{cid}",
            "candidate_revision": f"rev-{cid}",
            "policy_fingerprint": "pf-1",
            "preflight_blockers": [],
            "source": {"digest": "2e8a97a0aa84" * 5 + "0000"},
            "jobs": [{"compatibility_key": f"ck-{cid}", "budget_seconds": 60}],
        }

    monkeypatch.setattr(rfr.planning, "resolve_idea", _fake_resolve)
    monkeypatch.setattr(
        rfr,
        "_ensure_ordinary_requal_study",
        lambda root, fam, cid, base_req, campaign, **kw: {"id": f"study-{fam}", "campaign": deepcopy(campaign)},
    )
    monkeypatch.setattr(rfr, "snapshot_source", lambda root, qroot, src: Path(qroot) / "snap")
    monkeypatch.setattr(
        rfr.forge_queue,
        "Queue",
        lambda *args, **kwargs: type(
            "_Q",
            (),
            {"submit": lambda self, req, camp: {"request": req, "status": "queued"}},
        )(),
    )
    monkeypatch.setattr(rfr.forge_queue, "drain", lambda *args, **kwargs: None)
    monkeypatch.setattr(rfr, "_conclude_queue_submissions", lambda *args, **kwargs: ["readout-1", "readout-2"])

    concluded = rfr.refresh_selected_family_readouts(
        tmp_path, tmp_path / "runs/forge", gpus=["0", "1"], cuda_model="NVIDIA RTX A6000", json_mode=True
    )
    assert concluded == ["readout-1", "readout-2"]
    assert sorted(set(resolved_cids)) == ["atlas-particle-normal-v4", "bispf-optimized-quintic-v1"]


def test_compiled_reports_dynamic_diagnostics_unregistered_selection_and_markdown_sync(
    tmp_path: Path, monkeypatch
):
    # 1. Dynamic diagnostic heading computation works for any promoted successor (e.g. atlas-particle-normal-v5).
    live_inv = read_json(ROOT / "reports/forge/technique-inventory.json")
    mutated_inv = deepcopy(live_inv)
    for row in mutated_inv.get("rows", []):
        if row.get("candidate_id") == "atlas-particle-normal-v4":
            row["candidate_id"] = "atlas-particle-normal-v5"
    for fam in mutated_inv.get("family_progress", {}).get("families", []):
        if fam.get("id") == "atlas":
            for cohort in fam.get("cohorts", []):
                if cohort.get("candidate_id") == "atlas-particle-normal-v4":
                    cohort["candidate_id"] = "atlas-particle-normal-v5"

    phrases = rfr._expected_selected_diagnostic_phrases(ROOT, mutated_inv)
    assert "Selected recipe 1-for-1 baseline-35 diagnostics (atlas-particle-normal-v5); seed 0" in phrases
    assert "Selected recipe declared-suite Tier 1–3 diagnostics (atlas-particle-normal-v5); seed 0" in phrases
    assert not any("atlas-particle-normal-v4" in p for p in phrases)

    # 2. Unregistered family-current-v1.json modification raises CodeChangeRequiredError (exit code 2).
    sel_rel = rfr.CURRENT_SELECTION
    atomic_json(tmp_path / sel_rel, {"selections": []})
    inv_doc = {"provenance": {"family_current_selection_sha256": "different-registered-sha"}}
    with pytest.raises(rfr.CodeChangeRequiredError, match="differs from reports/forge/technique-inventory.json"):
        rfr._validate_publication_registration_sync(tmp_path, inv_doc)

    # 3. Missing/corrupted ATTEMPT_COVERAGE.md is detected and regenerated cleanly from attempt-coverage.json.
    cov_sample = {
        "schema_version": 1,
        "maximum_tests": 4,
        "maximum_per_gpu": 2,
        "records": [
            {
                "key": "k-atlas-1",
                "family": "atlas",
                "label": "Atlas (atlas-particle-normal-v4; current selected)",
                "candidate_id": "atlas-particle-normal-v4",
                "candidate_revision": "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd",
                "source_digest": "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf",
                "task": "gaussian1d_smoke",
                "execution_sha256": "e1",
                "evaluation_sha256": "v1",
                "status": "PASS",
                "attempted": True,
                "origin": "retained_selected_row",
                "diagnostic": False,
                "evidence": "reports/forge/technique-inventory.json",
            }
        ],
    }
    atomic_json(tmp_path / "reports/forge/attempt-coverage.json", cov_sample)
    assert rfr._attempt_coverage_markdown_synced(tmp_path, cov_sample) is False
    rfr._ensure_attempt_coverage_markdown(tmp_path, json_mode=True)
    assert (tmp_path / "reports/forge/ATTEMPT_COVERAGE.md").is_file()
    assert rfr._attempt_coverage_markdown_synced(tmp_path, cov_sample) is True


def test_requal_study_and_campaign_avoid_id_collisions_when_candidate_or_policy_changes(
    tmp_path: Path, monkeypatch
):
    src_digest = "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf"
    src12 = src_digest[:12]

    # 1. Campaign creation & collision avoidance when base campaign is already registered in queue state.
    camp1 = rfr._select_or_create_requal_campaign(
        tmp_path, {}, src12=src12, batch_sig="abcd1234", max_allowance=1200, family_count=2
    )
    assert camp1["id"] == f"stale-family-requal-{src12}"

    camp2 = rfr._select_or_create_requal_campaign(
        tmp_path,
        {"campaigns": {camp1["id"]: camp1}},
        src12=src12,
        batch_sig="abcd1234",
        max_allowance=1200,
        family_count=2,
    )
    assert camp2["id"] == f"stale-family-requal-{src12}-abcd1234"

    # 2. Study creation & collision avoidance when legacy study ID already exists for an older candidate or concluded run.
    def _make_req(cid: str, rev: str) -> dict[str, Any]:
        return {
            "candidate_revision": rev,
            "policy_fingerprint": "pf-v14",
            "source": {"digest": src_digest},
            "view": {
                "id": "discriminator_stability",
                "assignments": [{"task": "gaussian1d_smoke", "qualification_tier": 1}],
            },
            "tasks": {
                "gaussian1d_smoke": {
                    "preflight_blockers": [],
                    "evaluation": {"thresholds": [["abs_mean_error", "<=", 0.05]]},
                }
            },
            "jobs": [{"compatibility_key": f"ck-{cid}-{rev}"}],
        }

    req_v1 = _make_req("bispf-v1", "rev-1111")
    study1 = rfr._ensure_ordinary_requal_study(
        tmp_path,
        "bispf",
        "bispf-v1",
        req_v1,
        camp1,
        cuda_model="NVIDIA RTX A6000",
        through_tier=4,
        q_state={},
        concluded_records=[],
    )
    assert study1["id"] == f"requal-{src12}-bispf-ordinary"
    assert study1["scope"]["through_tier"] == 4

    # Promote bispf-optimized-quintic-v1 at the exact same source digest (no code changes):
    # must mint a fresh bound study ID rather than returning study1 for bispf-v1.
    req_v2 = _make_req("bispf-optimized-quintic-v1", "rev-2222")
    study2 = rfr._ensure_ordinary_requal_study(
        tmp_path,
        "bispf",
        "bispf-optimized-quintic-v1",
        req_v2,
        camp2,
        cuda_model="NVIDIA RTX A6000",
        through_tier=4,
        q_state={},
        concluded_records=[],
    )
    assert study2["id"] != study1["id"]
    assert study2["id"].startswith(f"requal-{src12}-bispf-")
    assert study2["candidate"] == "bispf-optimized-quintic-v1"
    assert study2["campaign"]["id"] == camp2["id"]


def test_check_selected_family_readouts_enforces_compat_key_cuda_model_and_tier3_unlocking(
    tmp_path: Path, monkeypatch
):
    rev = "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd"
    fake_req = {
        "candidate_revision": rev,
        "view": {
            "assignments": [
                {"task": "gaussian1d_smoke", "qualification_tier": 1, "importance": "required"},
                {"task": "mode_hold", "qualification_tier": 2, "importance": "required"},
                {"task": "ring_hold", "qualification_tier": 3, "importance": "required"},
                {"task": "grid100", "qualification_tier": 2, "importance": "held"},
                {"task": "grid100_14k", "qualification_tier": 3, "importance": "held"},
            ]
        },
        "tasks": {
            "gaussian1d_smoke": {"preflight_blockers": [], "dependencies": []},
            "mode_hold": {"preflight_blockers": [], "dependencies": []},
            "ring_hold": {"preflight_blockers": [], "dependencies": []},
            "grid100": {"preflight_blockers": [], "dependencies": []},
            "grid100_14k": {
                "preflight_blockers": [],
                "dependencies": [{"kind": "checkpoint", "task": "grid100"}],
            },
        },
        "jobs": [
            {"task_id": "gaussian1d_smoke", "task_ids": ["gaussian1d_smoke"], "compatibility_key": "ck-t1-new"},
            {"task_id": "mode_hold", "task_ids": ["mode_hold"], "compatibility_key": "ck-t2-mode"},
            {"task_id": "grid100", "task_ids": ["grid100"], "compatibility_key": "ck-t2-grid"},
            {"task_id": "ring_hold", "task_ids": ["ring_hold"], "compatibility_key": "ck-t3-ring"},
            {"task_id": "grid100_14k", "task_ids": ["grid100_14k"], "compatibility_key": "ck-t3-grid14k"},
        ],
    }
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [{"family": "atlas", "candidate_id": "atlas-particle-normal-v4"}],
    )
    monkeypatch.setattr(rfr.planning, "resolve_idea", lambda *args, **kwargs: deepcopy(fake_req))

    readout_path = tmp_path / "reports/forge/records/readout-atlas.json"

    # 1. Mismatched compatibility_key or wrong cuda_model is ignored -> Tier 1 remains unmeasured.
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "atlas-particle-normal-v4",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {
                "scope": {
                    "view": "discriminator_stability",
                    "execution_backend": "cuda",
                    "cuda_model": "NVIDIA RTX A6000",
                }
            },
            "task_results": [
                {"task_id": "gaussian1d_smoke", "compatibility_key": "ck-t1-STALE", "gate_status": "PASS"}
            ],
        },
    )
    stale_ck = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert stale_ck["current"] is False
    assert stale_ck["missing_families"][0]["unmeasured_tier1"] == ["gaussian1d_smoke"]

    # 2. Tier 1 required task PASS unlocks Tier 2 tasks (mode_hold and grid100).
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "atlas-particle-normal-v4",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {
                "scope": {
                    "view": "discriminator_stability",
                    "execution_backend": "cuda",
                    "cuda_model": "NVIDIA RTX A6000",
                }
            },
            "task_results": [
                {"task_id": "gaussian1d_smoke", "compatibility_key": "ck-t1-new", "gate_status": "PASS"}
            ],
        },
    )
    t2_unlocked = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert t2_unlocked["current"] is False
    assert t2_unlocked["missing_families"][0]["unmeasured_tier1"] == []
    assert sorted(t2_unlocked["missing_families"][0]["unmeasured_tier2"]) == ["grid100", "mode_hold"]

    # 3. Tier 1 + Tier 2 required tasks PASS (while held grid100 FAILs) unlocks Tier 3 ring_hold,
    #    while checkpoint-dependent grid100_14k is recognized as dependency-blocked.
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "atlas-particle-normal-v4",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {
                "scope": {
                    "view": "discriminator_stability",
                    "execution_backend": "cuda",
                    "cuda_model": "NVIDIA RTX A6000",
                }
            },
            "task_results": [
                {"task_id": "gaussian1d_smoke", "compatibility_key": "ck-t1-new", "gate_status": "PASS"},
                {"task_id": "mode_hold", "compatibility_key": "ck-t2-mode", "gate_status": "PASS"},
                {"task_id": "grid100", "compatibility_key": "ck-t2-grid", "gate_status": "FAIL"},
            ],
        },
    )
    t3_unlocked = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert t3_unlocked["current"] is False
    assert t3_unlocked["missing_families"][0]["unmeasured_by_tier"] == {"3": ["ring_hold"]}


def test_scaling_campaign_reuse_guard_and_revision_bound_readout_conclusion(
    tmp_path: Path, monkeypatch
):
    # 1. Existing scaling registration with already-published receipts and empty queue state is NOT overwritten.
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [{"family": "atlas", "candidate_id": "atlas-particle-normal-v4"}],
    )
    monkeypatch.setattr(
        rfr,
        "load_view",
        lambda root, name: {"assignments": [{"task": "dcgan_performance_r32"}]},
    )
    live_sha = {"particlegan/__init__.py": "sha-new"}
    monkeypatch.setattr(rfr, "scaling_source_files_sha256", lambda root: live_sha)
    monkeypatch.setattr(
        rfr,
        "catalog_source_manifest",
        lambda root, idea=None: {"digest": "feedface1234567890abcdef"},
    )

    perf_dir = tmp_path / "configs/forge/performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    (perf_dir / "dcgan-performance-cloud-v3.json").write_text("{}\n")
    old_manifest = {
        "id": "dcgan-performance-cloud-v3",
        "view": "dcgan_performance",
        "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000",
        "task_ids": ["dcgan_performance_r32"],
        "family_selections": [{"family": "atlas", "candidate_id": "atlas-particle-normal-v4"}],
        "campaign": {"id": "dcgan-performance-cloud-v3"},
        "studies": ["dcgan-performance-cloud-v3-atlas"],
    }
    atomic_json(
        tmp_path / "reports/forge/scaling-receipts/dcgan-performance-cloud-v3/cell-01.json",
        {"result": {"identity": {"source": {"files_sha256": live_sha}}}},
    )

    registered_ids: list[str] = []

    def _fake_load(root: Path, reg_id: str):
        if reg_id == "dcgan-performance-cloud-v3":
            return deepcopy(old_manifest)
        return {**deepcopy(old_manifest), "id": reg_id, "campaign": {"id": reg_id}}

    monkeypatch.setattr(rfr.scaling, "_load", _fake_load)
    monkeypatch.setattr(
        rfr.scaling,
        "register",
        lambda root, reg_id, **kwargs: registered_ids.append(reg_id),
    )

    chosen = rfr._select_or_register_scaling_campaign(
        tmp_path, tmp_path / "runs/forge", cuda_model="NVIDIA RTX A6000", json_mode=True
    )
    assert chosen["id"] == "dcgan-performance-cloud-feedface1234"
    assert registered_ids == ["dcgan-performance-cloud-feedface1234"]

    # 2. _conclude_queue_submissions binds candidate_ref as '<cid>@<candidate_revision>'.
    cand_rev = "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd"
    q_root = tmp_path / "runs/forge"
    atomic_json(
        q_root / "queue/state.json",
        {
            "jobs": {
                "job-1": {"result": {"task_results": [{"task_id": "dcgan_performance_r32", "gate_status": "PASS"}]}}
            },
            "submissions": {
                "sub-1": {
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "campaign_id": "camp-1",
                        "candidate": {"id": "atlas-particle-normal-v4"},
                        "candidate_revision": cand_rev,
                        "source": {"digest": "feedface1234567890abcdef"},
                        "view": {"id": "dcgan_performance"},
                        "jobs": [{"compatibility_key": "job-1"}],
                    },
                }
            },
        },
    )
    seen_refs: list[str] = []
    monkeypatch.setattr(
        rfr.knowledge,
        "readout",
        lambda root, ref, **kwargs: (seen_refs.append(ref) or {"record_id": "readout-new"}),
    )
    recs = rfr._conclude_queue_submissions(tmp_path, q_root, campaign_id="camp-1", json_mode=True)
    assert recs == ["readout-new"]
    assert seen_refs == [f"atlas-particle-normal-v4@{cand_rev}"]


def test_frozen_checkout_missing_commit_and_empty_gpu_flags_raise_code_change_required(
    tmp_path: Path, monkeypatch
):
    # 1. Missing frozen git commit raises CodeChangeRequiredError and leaves no partial checkout directory.
    missing_commit = "0000000000000000000000000000000000000000"
    checkout_root = tmp_path / "frozen-checkouts"
    with pytest.raises(rfr.CodeChangeRequiredError, match="git archive failed"):
        rfr._ensure_frozen_checkout(ROOT, missing_commit, checkout_root)
    assert not (checkout_root / missing_commit).exists()

    # 2. Empty --gpus list raises CodeChangeRequiredError when GPU stages are stale instead of defaulting to ['0','1'].
    monkeypatch.setattr(
        rfr,
        "check_status",
        lambda *args, **kwargs: {
            "status": "NEEDS_REFRESH",
            "source_digest": "2e8a97a0aa84",
            "stale_stages": ["scaling_performance"],
            "stages": {
                "queue": {"current": True},
                "scaling_performance": {"current": False},
                "attempt_coverage": {"current": True},
                "selected_family_readouts": {"current": True},
                "compiled_reports": {"current": True},
            },
        },
    )
    with pytest.raises(rfr.CodeChangeRequiredError, match="at least one GPU index is required"):
        rfr.refresh_all(tmp_path, gpus=[], json_mode=True)
    with pytest.raises(rfr.CodeChangeRequiredError, match="non-empty --scaling-gpu"):
        rfr.refresh_scaling_performance(
            tmp_path, tmp_path / "runs/forge", scaling_gpu="  ", cuda_model="NVIDIA RTX A6000", json_mode=True
        )


def test_resolve_scaling_gpu_auto_picks_uncontended_handles_cuda_visible_devices_and_fallbacks(monkeypatch):
    class _Proc:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout

    gpu_csv = (
        "0, GPU-ed080e41-3193-3755-6756-f3d46c433331, 1093, 2\n"
        "1, GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce, 15, 0\n"
    )
    app_csv = (
        "GPU-ed080e41-3193-3755-6756-f3d46c433331, 4123, 450\n"
        "GPU-ed080e41-3193-3755-6756-f3d46c433331, 4199, 600\n"
    )

    def _fake_run(cmd, **kwargs):
        query_arg = cmd[1]
        if query_arg.startswith("--query-gpu="):
            return _Proc(0, gpu_csv)
        return _Proc(0, app_csv)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(rfr.subprocess, "run", _fake_run)

    # 1. Explicit GPU index bypasses auto selection.
    assert rfr._resolve_scaling_gpu("0", ["0", "1"], json_mode=True) == "0"

    # 2. Auto picks clear GPU 1 over GUI-contended GPU 0.
    assert rfr._resolve_scaling_gpu("auto", ["0", "1"], json_mode=True) == "1"

    # 3. Respects CUDA_VISIBLE_DEVICES remapping (logical 0 -> physical 1 / clear UUID, including short UUID prefix).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-548116b7,GPU-ed080e41")
    assert rfr._resolve_scaling_gpu("auto", ["0", "1"], json_mode=True) == "0"

    # 3b. Single-device CUDA_VISIBLE_DEVICES="0" never leaks out-of-range ordinal "1"; out-of-range --gpus raises CodeChangeRequiredError.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert rfr._resolve_scaling_gpu("auto", ["0", "1"], json_mode=True) == "0"
    with pytest.raises(rfr.CodeChangeRequiredError, match="CUDA_VISIBLE_DEVICES"):
        rfr._resolve_scaling_gpu("auto", ["1"], json_mode=True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    # 3c. Non-numeric nvidia-smi telemetry tokens ([N/A], [Not Supported]) do not abort auto-selection.
    na_gpu_csv = (
        "0, GPU-ed080e41-3193-3755-6756-f3d46c433331, [N/A], [Not Supported]\n"
        "1, GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce, 15, [N/A]\n"
    )
    monkeypatch.setattr(
        rfr.subprocess,
        "run",
        lambda cmd, **kw: _Proc(0, na_gpu_csv if cmd[1].startswith("--query-gpu=") else app_csv),
    )
    assert rfr._resolve_scaling_gpu("auto", ["0", "1"], json_mode=True) == "1"

    # 4. Falls back to least-contended when all GPUs have external compute processes.
    busy_apps = (
        "GPU-ed080e41-3193-3755-6756-f3d46c433331, 4123, 450\n"
        "GPU-ed080e41-3193-3755-6756-f3d46c433331, 4199, 600\n"
        "GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce, 5500, 120\n"
    )
    monkeypatch.setattr(
        rfr.subprocess,
        "run",
        lambda cmd, **kw: _Proc(0, gpu_csv if cmd[1].startswith("--query-gpu=") else busy_apps),
    )
    assert rfr._resolve_scaling_gpu("auto", ["0", "1"], json_mode=True) == "1"

    # 5. Missing nvidia-smi falls back to first candidate; empty --gpus raises CodeChangeRequiredError.
    def _raise_oserror(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(rfr.subprocess, "run", _raise_oserror)
    assert rfr._resolve_scaling_gpu("auto", ["1", "0"], json_mode=True) == "1"
    with pytest.raises(rfr.CodeChangeRequiredError, match="empty --gpus"):
        rfr._resolve_scaling_gpu("auto", [], json_mode=True)


def test_scaling_contention_unpaired_rejection_and_superseded_job_eviction(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [
            {"family": "atlas", "trainer_family": "atlas", "candidate_id": "atlas-particle-normal-v4"},
            {"family": "bispf", "trainer_family": "bispf", "candidate_id": "bispf-optimized-quintic-v1"},
        ],
    )
    monkeypatch.setattr(
        rfr,
        "load_view",
        lambda root, name: {"assignments": [{"task": "dcgan_performance_r32"}]},
    )
    live_files_sha = {"particlegan/__init__.py": "sha-live"}
    monkeypatch.setattr(rfr, "scaling_source_files_sha256", lambda root: live_files_sha)
    monkeypatch.setattr(rfr.scaling_publication, "performance_status", lambda res: res["status"])
    monkeypatch.setattr(rfr.scaling_publication, "hardware_key", lambda res: res["identity"]["hardware"])

    def _rec(fam: str, cid: str, contention: str, comp_key: str):
        return {
            "trainer_family": fam,
            "candidate_id": cid,
            "task_id": "dcgan_performance_r32",
            "comparison_key": comp_key,
            "result": {
                "status": "COMPLETE",
                "hardware": {"contention_status": contention},
                "identity": {
                    "source": {"sha256": "src-sha-1", "files_sha256": live_files_sha},
                    "runtime_sha256": "rt-sha-1",
                    "hardware": {"backend": "cuda", "gpu": {"uuid": "GPU-1", "name": "RTX A6000"}},
                },
            },
        }

    # 1. Contended & unpaired rows fail check_scaling_performance even on a single hardware cohort.
    contended_manifest = {
        "unavailable": [],
        "records": [
            _rec("atlas", "atlas-particle-normal-v4", "contended", "priv-key-a"),
            _rec("bispf", "bispf-optimized-quintic-v1", "contended", "priv-key-b"),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: contended_manifest)
    bad = rfr.check_scaling_performance(tmp_path)
    assert bad["current"] is False
    assert bad["contended_cells"] == 2
    assert bad["unpaired_tasks"] == ["dcgan_performance_r32"]

    # 2. Clear & paired rows pass check_scaling_performance.
    clear_manifest = {
        "unavailable": [],
        "records": [
            _rec("atlas", "atlas-particle-normal-v4", "clear", "shared-key-1"),
            _rec("bispf", "bispf-optimized-quintic-v1", "clear", "shared-key-1"),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: clear_manifest)
    ok = rfr.check_scaling_performance(tmp_path)
    assert ok["current"] is True
    assert ok["contended_cells"] == 0
    assert ok["unpaired_tasks"] == []

    # 3. _evict_superseded_scaling_queue_jobs evicts contended dcgan_performance jobs, sparing non-scaling subscribers.
    q_root = tmp_path / "runs/forge"
    attempt_dir = tmp_path / "reports/forge/attempts/att-contended"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "result.json").write_text("{}\n")

    manifest = {
        "id": "dcgan-performance-cloud-test",
        "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000",
        "campaign": {"id": "dcgan-performance-cloud-test"},
        "family_selections": [
            {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "study_id": "st-atlas"},
            {"family": "bispf", "candidate_id": "bispf-optimized-quintic-v1", "study_id": "st-bispf"},
        ],
    }
    monkeypatch.setattr(
        rfr.planning,
        "resolve_idea",
        lambda root, cid, view_id, **kw: {
            "jobs": [{"compatibility_key": f"ck-{cid}"}]
        },
    )
    atomic_json(
        q_root / "queue/state.json",
        {
            "jobs": {
                "ck-atlas-particle-normal-v4": {
                    "state": "completed",
                    "subscribers": ["sub-atlas"],
                    "result": {
                        "attempt_id": "att-contended",
                        "raw": {"result": {"hardware": {"contention_status": "contended", "device": "cuda:0"}}},
                    },
                },
                "ck-bispf-optimized-quintic-v1": {
                    "state": "completed",
                    "subscribers": ["sub-bispf", "sub-ordinary"],
                    "result": {
                        "attempt_id": "att-shared",
                        "raw": {"result": {"hardware": {"contention_status": "contended", "device": "cuda:0"}}},
                    },
                },
            },
            "submissions": {
                "sub-atlas": {"request": {"view": {"id": "dcgan_performance"}}},
                "sub-bispf": {"request": {"view": {"id": "dcgan_performance"}}},
                "sub-ordinary": {"request": {"view": {"id": "discriminator_stability"}}},
            },
        },
    )
    evicted = rfr._evict_superseded_scaling_queue_jobs(tmp_path, q_root, manifest, "1", json_mode=True)
    assert evicted == 1
    after_state = read_json(q_root / "queue/state.json")
    assert "ck-atlas-particle-normal-v4" not in after_state["jobs"]
    assert "sub-atlas" not in after_state["submissions"]
    assert "ck-bispf-optimized-quintic-v1" in after_state["jobs"]
    assert not attempt_dir.exists()
    assert (q_root / "superseded-scaling/dcgan-performance-cloud-test/attempts/att-contended/result.json").is_file()


def test_publication_registration_sync_allows_committed_snapshot_republish_and_rejects_unregistered(
    tmp_path: Path, monkeypatch
):
    # Committed repository snapshots satisfy family-current-v1.json!
    assert rfr._registered_evidence_satisfies_selection(ROOT) is True

    live_inv = read_json(ROOT / "reports/forge/technique-inventory.json")
    stale_prov_inv = deepcopy(live_inv)
    stale_prov_inv["provenance"]["evidence_manifest_sha256"] = "0" * 64
    # Because committed snapshots satisfy family-current-v1.json, this reports False (NEEDS_REFRESH) rather than CodeChangeRequiredError.
    assert rfr._validate_publication_registration_sync(ROOT, stale_prov_inv) is False

    # And _republish_registered_inventory triggers publish_current + refresh_publication when out of sync.
    calls: list[tuple[str, Any]] = []
    from reports.forge import regenerate_technique_inventory as rti

    monkeypatch.setattr(
        rfr,
        "_validate_publication_registration_sync",
        lambda root, inv: False,
    )
    monkeypatch.setattr(
        rti,
        "publish_current",
        lambda root, **kw: (calls.append(("publish_current", kw.get("comparison_policy"))) or {"rows": 8}),
    )
    monkeypatch.setattr(
        rti,
        "refresh_publication",
        lambda root, **kw: (calls.append(("refresh_publication", kw.get("view_id"))) or {"rows": 8}),
    )
    summary = rfr._republish_registered_inventory(ROOT, json_mode=True)
    assert summary == {"rows": 8}
    assert [name for name, _ in calls] == ["publish_current", "refresh_publication"]
    manifest = read_json(ROOT / "reports/forge/technique-evidence/manifest.json")
    assert Path(calls[0][1]).name == f"{manifest['view']}-v{manifest['view_revision']}.json"

    # When registered evidence does NOT satisfy selection (unregistered snapshot), raises CodeChangeRequiredError.
    monkeypatch.undo()
    monkeypatch.setattr(rfr, "_registered_evidence_satisfies_selection", lambda root: False)
    with pytest.raises(rfr.CodeChangeRequiredError, match="differs from reports/forge/technique-inventory.json"):
        rfr._validate_publication_registration_sync(ROOT, stale_prov_inv)


def test_live_repository_forge_readouts_are_current_uncontended_and_surface_tier1_readouts():
    status = rfr.check_published_status(ROOT)
    assert status["status"] == "CURRENT", json.dumps(status, indent=2, sort_keys=True)
    assert status["stale_stages"] == []
    _assert_current_source_scope(status, ROOT)

    perf = status["stages"]["published_scaling_performance"]
    assert perf["complete_cells"] == 60
    assert perf["contended_cells"] == 0
    assert perf["unpaired_tasks"] == []
    assert perf["table_cohorts"] == 1

    comp = status["stages"]["compiled_reports"]
    assert comp["publication_registration_synced"] is True
    assert comp["current_source_tier1_synced"] is True
    assert comp["refresh_docs_synced"] is True

    inv_md = (ROOT / "reports/forge/technique-inventory.md").read_text()
    perf_block = inv_md.split("<!-- Forge scaling performance: start -->", 1)[1].split(
        "<!-- Forge scaling performance: end -->", 1
    )[0]
    assert perf_block.count("| clear |") == 60
    assert "| contended |" not in perf_block

    from experiments.forge.family_reports import _selected_cohorts, _selected_row
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    selected = {family["id"]: (family, _selected_row(publication, cohort))
                for family, cohort in _selected_cohorts(publication["family_progress"])}
    expected_families = (
        ("release07-gan-v3", "GAN v3 release 0.7", "recipe-prior-refactor-release07-gan-v3-v1"),
        ("r1r2", "R1/R2", "recipe-prior-refactor-r1r2-v1"),
        ("k3p", "K3P", "recipe-prior-refactor-k3p-v1"),
        ("ka2", "KA2", "recipe-prior-refactor-ka2-v1"),
        ("atlas", "Atlas", "atlas-particle-normal-v4"),
        ("bispf", "BISPF", "bispf-optimized-quintic-v1"),
        ("bcap-pure", "BCAP", "recipe-prior-refactor-bcap-pure-v1"),
        ("empf", "EMPF", "empf-v7"),
        ("rhbi", "RHBI", "rhbi-v11"),
        ("e22", "E22", "e22"),
    )
    assert set(selected) == {family for family, _, _ in expected_families}
    for family_id, label, candidate in expected_families:
        family, row = selected[family_id]
        assert family["label"] == label and row["candidate_id"] == candidate
        assert f"[{label}](families/{Path(family['page']).name})" in inv_md
        detail = (ROOT / family["page"]).read_text()
        assert f"Selected configuration: [{candidate}]" in detail
        assert f"Candidate revision: `{row['candidate_revision']}`" in detail
    for expected_snippet in (
        "python ops/refresh_forge_readouts.py --check",
        "python ops/refresh_forge_readouts.py --reports-only",
    ):
        assert expected_snippet in inv_md

    current_source_candidate_snippets = (
        "| [GAN v3 release 0.7](families/release07-gan-v3.md) | [`recipe-prior-refactor-release07-gan-v3-v1`]",
        "| [R1/R2](families/r1r2.md) | [`recipe-prior-refactor-r1r2-v1`]",
        "| [K3P](families/k3p.md) | [`recipe-prior-refactor-k3p-v1`]",
        "| [KA2](families/ka2.md) | [`recipe-prior-refactor-ka2-v1`]",
        "| [Atlas](families/atlas.md) | [`atlas-particle-normal-v4`]",
        "| [BISPF](families/bispf.md) | [`bispf-optimized-quintic-v1`]",
        "| [BCAP](families/bcap-pure.md) | [`recipe-prior-refactor-bcap-pure-v1`]",
        "| [EMPF](families/empf.md) | [`empf-v7`]",
        "| [RHBI](families/rhbi.md) | [`rhbi-v11`]",
        "| [E22](families/e22.md) | [`e22`]",
    )

    # A current page preserves exact-cohort outcomes. Diagnostic backfill cannot
    # turn a refusal into a numerical FAIL or transfer a PASS into another cohort.
    from reports.forge import regenerate_technique_inventory as rti
    from experiments.forge.publication_integrity import coverage_disposition
    assert "binary PASS or FAIL" not in inv_md
    assert "Scientific scope corrections" in inv_md
    original_cohorts = {family["id"]: cohort for family, cohort in
                        _selected_cohorts(publication["family_progress"])}
    projected = rti._project_publication_for_family_pages(ROOT, publication)
    projected_cohorts = {family["id"]: cohort for family, cohort in
                         _selected_cohorts(projected["family_progress"])}
    for family, original_cohort in original_cohorts.items():
        for task, original in original_cohort["tasks"].items():
            cell = projected_cohorts[family]["tasks"][task]
            assert cell["status"] == original["status"]
            assert cell["current_contract"] == original["current_contract"]
            assert cell.get("run_attempt") == original.get("run_attempt")
        assert projected_cohorts[family]["total"] == original_cohort["total"]

    tier1_readouts_block = rti._current_source_tier1_readouts_markdown(
        publication, ROOT, ROOT / "reports/forge/technique-inventory.md")
    if tier1_readouts_block:
        assert tier1_readouts_block in inv_md
        for expected_snippet in current_source_candidate_snippets:
            assert expected_snippet in tier1_readouts_block
        assert "ordinary Tier 1 readouts" in tier1_readouts_block
        assert "Tier 2 (" not in tier1_readouts_block
    else:
        assert "### Current-source selected-recipe Tier 1 readouts" not in inv_md

    # All archived observations remain available unchanged, including assisted
    # passes. A source-bound correction removes their claim eligibility, never
    # rewrites their original numerical outcome or cost.
    cov_doc = read_json(ROOT / "reports/forge/attempt-coverage.json")
    coverage_before = deepcopy(cov_doc)
    corrected = 0
    for rec in cov_doc["records"]:
        disposition = coverage_disposition(ROOT, rec)
        if disposition:
            corrected += 1
            assert disposition["eligible"] is False
            assert disposition["scope"] == "archived_contract_diagnostic"
    assert corrected > 0
    assert cov_doc == coverage_before

    from experiments.forge.family_reports import full_suite_tasks
    required_tasks = set(full_suite_tasks(publication["family_progress"]))
    for family_id, original_cohort in original_cohorts.items():
        family = selected[family_id][0]
        fam_md = (ROOT / family["page"]).read_text()
        view_table_header = (
            "| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |"
        )
        current_bench = fam_md.split(view_table_header, 1)[1].split("\n## adaptation\n", 1)[0]
        seen = set()
        status_index = None
        for line in current_bench.splitlines():
            if line.startswith("| ") and not line.startswith(("| [", "| ---")):
                headers = [cell.strip() for cell in line.split("|")[1:-1]]
                status_index = headers.index("Recorded result") if "Recorded result" in headers else None
            if not line.startswith("| ["):
                continue
            task_id = line.split("](", 1)[0].removeprefix("| [")
            if task_id not in original_cohort["tasks"]:
                continue
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            assert status_index is not None and len(cells) > status_index
            assert cells[status_index] == original_cohort["tasks"][task_id]["status"], (family_id, task_id, line)
            seen.add(task_id)
        # Gaps and operational outcomes must remain visible, never filled with
        # diagnostic grades to satisfy an all-binary/no-unknown display rule.
        assert required_tasks <= seen, (family_id, sorted(required_tasks - seen))
        for task_id in ("precision_gaussian_mxfp8", "precision_gaussian_nvfp4"):
            if family_id not in {"empf", "rhbi"}:
                assert original_cohort["tasks"][task_id]["status"] == "UNKNOWN"
            else:
                assert original_cohort["tasks"][task_id]["status"] in {"PASS", "FAIL"}
            assert task_id in seen
        if family_id == "empf":
            for task_id in ("model_ddgan_bimodal_chain", "model_ae_ddgan_bimodal_chain"):
                assert original_cohort["tasks"][task_id]["status"] == "UNKNOWN"
                assert task_id in seen



def test_recorded_overview_integrity_rejects_score_drift_without_current_source(monkeypatch):
    from experiments.forge import family_reports
    from reports.forge import regenerate_technique_inventory as rti

    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    # Renderer-only fixture uses the inspected reducer; no saved source pin or
    # scientific qualification is edited to make a maintenance branch current.
    from experiments.forge.contracts import file_hash
    publication["provenance"]["publication_reducer_sha256"] = file_hash(
        ROOT / "reports/forge/regenerate_technique_inventory.py")
    monkeypatch.setattr(rti, "_concluded_current_source_tier1_payload", lambda *args: None)
    monkeypatch.setattr(family_reports, "generated_pages", lambda root, pub: {})
    overview = ROOT / "reports/forge/technique-inventory.md"
    original = rti._current_markdown(publication, ROOT, overview)
    assert "### Current-source selected-recipe Tier 1 readouts" not in original
    assert rfr._generated_pages_synced(ROOT, publication, original) is True
    import re
    score = re.search(r"\*\*\[(\d+)([^/\]]*)/(\d+)\]", original)
    assert score is not None
    changed_score = f"**[{int(score.group(1)) + 1}{score.group(2)}/{score.group(3)}]"
    corrupted = original[:score.start()] + changed_score + original[score.end():]
    assert corrupted != original
    assert rfr._generated_pages_synced(ROOT, publication, corrupted) is False
    fake_current = original + "\n### Current-source selected-recipe Tier 1 readouts\n\nFalse qualification.\n"
    assert rfr._generated_pages_synced(ROOT, publication, fake_current) is False


def _copy_minimal_repo(src_root: Path, dst_root: Path) -> None:
    import shutil

    dst_root.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in ("configs", "particlegan", "experiments", "benchmarks", "lib"):
        s = src_root / rel
        if s.is_dir():
            shutil.copytree(s, dst_root / rel, ignore=ignore)
    for rel in ("pyproject.toml", "README.md"):
        s = src_root / rel
        if s.is_file():
            shutil.copy2(s, dst_root / rel)


def test_evict_superseded_scaling_queue_jobs_matches_real_queue_schema_and_prunes_subscribers(
    tmp_path: Path, monkeypatch
):
    from experiments.forge import sources

    monkeypatch.setattr(
        sources,
        "_gpu_models",
        lambda: frozenset({("NVIDIA RTX A6000", "615.78.08", "8.6")}),
    )
    repo = tmp_path / "repo"
    _copy_minimal_repo(ROOT, repo)
    (repo / "reports/forge/attempts/att-gpu0-contended").mkdir(parents=True)
    (repo / "reports/forge/attempts/att-gpu0-contended/result.json").write_text("{}\n")

    real_manifest = rfr.scaling._load(repo, "dcgan-performance-cloud-2e8a97a0aa84")
    atlas_sel = next(s for s in real_manifest["family_selections"] if s["family"] == "atlas")
    real_req = rfr.planning.resolve_idea(repo, atlas_sel["candidate_id"], study=atlas_sel["study"])
    ck_keep, ck_evict_wrong_gpu, ck_active_blocked = [j["compatibility_key"] for j in real_req["jobs"][:3]]

    q_root = tmp_path / "runs/forge"
    atomic_json(
        q_root / "queue/state.json",
        {
            "schema_version": 1,
            "jobs": {
                # Valid GPU 1 terminal job: worker.device='1', cuda_visible_devices='1', inside-worker device='cuda:0'.
                ck_keep: {
                    "status": "terminal",
                    "worker": {"device": "1"},
                    "subscribers": ["sub-mixed"],
                    "result": {
                        "attempt_id": "att-gpu1-clear",
                        "raw": {
                            "result": {
                                "hardware": {
                                    "contention_status": "clear",
                                    "cuda_visible_devices": "1",
                                    "device": "cuda:0",
                                    "physical_uuid": "GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce",
                                }
                            }
                        },
                    },
                },
                # Wrong-GPU (GPU 0) terminal job sharing 'sub-mixed' with ck_keep: must be evicted and sub-mixed pruned from ck_keep.
                ck_evict_wrong_gpu: {
                    "status": "terminal",
                    "worker": {"device": "0"},
                    "subscribers": ["sub-mixed"],
                    "result": {
                        "attempt_id": "att-gpu0-contended",
                        "raw": {
                            "result": {
                                "hardware": {
                                    "contention_status": "clear",
                                    "cuda_visible_devices": "0",
                                    "device": "cuda:0",
                                    "physical_uuid": "GPU-ed080e41-3193-3755-6756-f3d46c433331",
                                }
                            }
                        },
                    },
                },
                # Contended job attached to an active ('running') submission: must NOT be evicted mid-flight.
                ck_active_blocked: {
                    "status": "terminal",
                    "worker": {"device": "0"},
                    "subscribers": ["sub-running"],
                    "result": {
                        "attempt_id": "att-running",
                        "raw": {
                            "result": {
                                "hardware": {
                                    "contention_status": "contended",
                                    "cuda_visible_devices": "0",
                                    "device": "cuda:0",
                                }
                            }
                        },
                    },
                },
            },
            "submissions": {
                "sub-mixed": {"status": "concluded", "request": {"view": {"id": "dcgan_performance"}}},
                "sub-running": {"status": "running", "request": {"view": {"id": "dcgan_performance"}}},
            },
        },
    )

    evicted = rfr._evict_superseded_scaling_queue_jobs(repo, q_root, real_manifest, "1", json_mode=True)
    assert evicted == 1
    after = read_json(q_root / "queue/state.json")
    assert ck_keep in after["jobs"]
    assert after["jobs"][ck_keep]["subscribers"] == []
    assert ck_evict_wrong_gpu not in after["jobs"]
    assert "sub-mixed" not in after["submissions"]
    assert ck_active_blocked in after["jobs"]
    assert "sub-running" in after["submissions"]
    assert not (repo / "reports/forge/attempts/att-gpu0-contended").exists()
    assert (
        q_root / "superseded-scaling/dcgan-performance-cloud-2e8a97a0aa84/attempts/att-gpu0-contended/result.json"
    ).is_file()


def test_republish_registered_inventory_end_to_end_recovers_missing_and_unverified_inventory(tmp_path: Path):
    import re
    import shutil
    import subprocess
    from reports.forge import regenerate_technique_inventory as rti

    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        inv_path = wt / "reports/forge/technique-inventory.json"
        md_path = wt / "reports/forge/technique-inventory.md"

        # Scenario A: stale evidence_manifest_sha256 with self-consistent input_digest -> real publish_current + refresh_publication.
        doc_a = read_json(inv_path)
        doc_a["provenance"]["evidence_manifest_sha256"] = "0" * 64
        doc_a["provenance"].pop("input_digest", None)
        doc_a["provenance"]["input_digest"] = rfr.stable_hash(doc_a)
        atomic_json(inv_path, doc_a)
        assert rfr.check_compiled_reports(wt)["publication_registration_synced"] is False
        rfr._republish_registered_inventory(wt, json_mode=True)
        assert rfr._verified_current_publication(read_json(inv_path)) is True
        assert rfr.check_compiled_reports(wt)["publication_registration_synced"] is True

        # Scenario B: broken input_digest + stale evidence_manifest_sha256 -> HEAD baseline restoration + recovery.
        doc_b = read_json(inv_path)
        doc_b["provenance"]["evidence_manifest_sha256"] = "0" * 64
        atomic_json(inv_path, doc_b)
        assert rfr.check_compiled_reports(wt)["publication_registration_synced"] is False
        rfr._republish_registered_inventory(wt, json_mode=True)
        assert rfr._verified_current_publication(read_json(inv_path)) is True
        assert rfr.check_compiled_reports(wt)["publication_registration_synced"] is True

        # Scenario C: deleted or corrupt technique-inventory.json + deleted technique-inventory.md -> HEAD baseline restoration + recovery.
        inv_path.write_text("{corrupt-json\n")
        assert rfr.check_compiled_reports(wt)["current"] is False
        inv_path.unlink()
        md_path.unlink()
        assert rfr.check_compiled_reports(wt)["current"] is False
        rfr._republish_registered_inventory(wt, json_mode=True)
        assert inv_path.is_file() and md_path.is_file()
        assert rfr._verified_current_publication(read_json(inv_path)) is True
        recovered = read_json(inv_path)
        expected_tier1 = rti._current_source_tier1_readouts_markdown(recovered, wt, md_path)
        if expected_tier1:
            assert expected_tier1 in md_path.read_text()
        else:
            assert "### Current-source selected-recipe Tier 1 readouts" not in md_path.read_text()
            archived = rti._recorded_source_binary_projection(recovered, wt)
            assert archived is not None and archived["qualification_input"] is False
            assert rfr.check_selected_family_readouts(wt)["missing_families"]

        # Scenario D: stale selected scores and false current-source claims are
        # detected and repaired without inventing a new-source measurement.
        orig_md = md_path.read_text()
        stale_md, mutations = re.subn(
            r"(\*\*\[)(\d+)(/\d+\]\(families/atlas\.md#[^)]+-tier-1\)\*\*)",
            lambda match: f"{match[1]}{int(match[2]) + 1}{match[3]} (7 not run)",
            orig_md,
            count=1,
        )
        assert mutations == 1
        assert stale_md != orig_md
        md_path.write_text(stale_md)
        assert stale_md != rti._current_markdown(recovered, wt, md_path)
        assert rfr.check_compiled_reports(wt)["generated_pages_synced"] is False
        corrupt_status = rfr.check_published_status(wt)
        assert corrupt_status["status"] == "NEEDS_REFRESH"
        assert "compiled_reports" in corrupt_status["stale_stages"]
        md_path.write_text(stale_md + "\n### Current-source selected-recipe Tier 1 readouts\n\n"
                           "Stale claim: archived scores are current-source measurements.\n")
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False
        rfr._republish_registered_inventory(wt, json_mode=True)
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True
        assert "tier-1)** (7 not run)" not in md_path.read_text()
        assert md_path.read_text() == orig_md
        assert "Stale claim: archived scores" not in md_path.read_text()
        if expected_tier1:
            assert expected_tier1 in md_path.read_text()
        else:
            assert "### Current-source selected-recipe Tier 1 readouts" not in md_path.read_text()
    finally:
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(wt, ignore_errors=True)


def _sync_tracked_worktree_edits(src_root: Path, dst_root: Path) -> None:
    import shutil
    import subprocess

    diff_proc = subprocess.run(
        ["git", "-C", str(src_root), "diff", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    for rel in diff_proc.stdout.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        src_file = src_root / rel
        if src_file.is_file():
            dst_file = dst_root / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)


def _readout_workstation_envelope_fixture():
    """Declared metadata for software compatibility controls, with no training."""
    return {
        "schema_version": 1,
        "request": {
            "runtime": {
                "implementation": "CPython",
                "machine": "x86_64",
                "packages": {"numpy": "2.5.4", "scipy": "1.18.1", "torch": "2.14.1"},
                "python": "3.14.8",
                "system": "Linux",
            },
            "rng": {
                "runtime": {
                    "rng": "torch.Generator",
                    "torch": "2.14.1+cu130",
                }
            },
            "compute_profiles": {
                "cuda": {
                    "autograd_multithreading_enabled": False,
                    "backend": "cuda",
                    "compute_capability": "8.6",
                    "deterministic": True,
                    "driver": "615.78.08",
                    "model": "NVIDIA RTX A6000",
                    "tf32": False,
                    "threads": 1,
                }
            },
        },
    }


def _restore_recorded_source_cohort(root: Path) -> str:
    """Restore the pre-migration software cohort for archived tamper controls.

    Latest-source integrity is checked separately above. These controls need
    actual original source bytes, rather than relabeling the latest checkout.
    Use the frozen pre-precision view for these cache/tamper controls, including
    the existing DDGAN tier calibration. No published evidence is rewritten.
    """
    import io
    import subprocess
    import tarfile
    from experiments.forge.sources import SOURCE_DIRS, SOURCE_SUFFIXES

    assert root.resolve() != ROOT.resolve()
    revision = "6261a5458c1f3e14c7c9faf1fc6966e03b2acfb1"
    cohort_revision = "292582579ce8608067491d1c33c316d3dabe7695"
    current = set(rfr.catalog_source_manifest(root)["files"])
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True).splitlines()
    desired = {name for name in names if name in current or name.startswith("configs/forge/tasks/") or (
        Path(name).suffix in SOURCE_SUFFIXES and (
            Path(name).parts[0] in SOURCE_DIRS
            or name.startswith("configs/") and "forge" not in Path(name).parts))}
    for name in current - desired:
        (root / name).unlink()
    data = subprocess.check_output(["git", "archive", revision, "--", *sorted(desired)], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(root, filter="data")
    cohort_data = subprocess.check_output(
        [
            "git",
            "archive",
            cohort_revision,
            "--",
            "configs/forge/trainer-families.json",
            "configs/forge/selections/family-current-v1.json",
            "reports/forge/technique-inventory.json",
        ],
        cwd=ROOT,
    )
    with tarfile.open(fileobj=io.BytesIO(cohort_data)) as archive:
        archive.extractall(root, filter="data")
    (root / "configs/forge/views/discriminator_stability.json").write_bytes(
        (ROOT / "configs/forge/view-history/discriminator_stability-v14.json").read_bytes())
    manifest = rfr.catalog_source_manifest(root)
    assert manifest["digest"] == "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf"
    _bind_readout_software_fixture(root, manifest["digest"])
    return manifest["digest"]


def _bind_readout_software_fixture(root: Path, source_digest: str) -> None:
    """Bind copied metadata to this API for tamper tests, never qualification.

    Ordinary receipts used an earlier declaration identity. Preserve each
    original identity in the private fixture and derive compatibility keys
    independently from planner jobs and the declared workstation envelope.
    Published rows, gate outcomes and source files remain unchanged.
    """
    from experiments.forge import planning
    runtime = _readout_workstation_envelope_fixture()["request"]
    for choice in rfr.current_family_candidates(root):
        req = planning.resolve_idea(
            root, choice["candidate_id"], view_id="discriminator_stability",
            through_tier=rfr._discriminator_stability_max_tier(root), freeze_source=False,
            execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
        assert req["source"]["digest"] == source_digest
        jobs = deepcopy(req["jobs"])
        for job in jobs:
            science = job["science"]
            science["runtime"] = deepcopy(runtime["runtime"])
            science["rng"]["runtime"] = deepcopy(runtime["rng"]["runtime"])
            science["compute"] = {**runtime["compute_profiles"]["cuda"],
                                  "threads": science["compute"]["threads"]}
        planning.rekey_jobs(jobs)
        keys = {tid: job["compatibility_key"] for job in jobs
                for tid in job.get("task_ids", [job["task_id"]])}
        for path in (root / "reports/forge/records").glob("readout-*.json"):
            record = read_json(path)
            if (record.get("candidate_id") != choice["candidate_id"]
                    or record.get("evidence_scope") != "current"):
                continue
            old_revision = record["candidate_revision"]
            record["software_fixture_original_identity"] = {
                "candidate_revision": old_revision, "source": deepcopy(record["source"])}
            record["qualification_input"] = record["qualification_reuse"] = False
            record["candidate_revision"] = req["candidate_revision"]
            record["source"] = {"digest": source_digest}
            for row in record["task_results"]:
                if row["task_id"] in keys:
                    row["compatibility_key"] = keys[row["task_id"]]
            atomic_json(path, record)
            for board in (root / "reports/forge/leaderboards").glob("*.md"):
                text = board.read_text()
                board.write_text(text.replace(old_revision[:12], req["candidate_revision"][:12]))


def test_current_source_tier1_readouts_ignores_non_tier1_records_and_invalidates_cache_on_task_change(
    tmp_path: Path,
):
    import shutil
    import subprocess
    from reports.forge import regenerate_technique_inventory as rti

    wt = tmp_path / "wt-tier1"
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        _sync_tracked_worktree_edits(ROOT, wt)
        _restore_recorded_source_cohort(wt)
        live_inv = read_json(wt / "reports/forge/technique-inventory.json")
        atlas_req = rti._resolve_idea_for_readout_check(
            wt,
            "atlas-particle-normal-v4",
            view_id="discriminator_stability",
            through_tier=1,
            cuda_model="NVIDIA RTX A6000",
            planning_mod=rfr.planning,
        )
        # Add a concluded current-source record containing ONLY a non-Tier-1 task: must NOT be cited in Tier 1 Evidence column.
        atomic_json(
            wt / "reports/forge/records/readout-tier2-only.json",
            {
                "lifecycle": "concluded",
                "evidence_scope": "current",
                "candidate_id": "atlas-particle-normal-v4",
                "candidate_revision": atlas_req["candidate_revision"],
                "source": {"digest": atlas_req["source"]["digest"]},
                "study": {
                    "scope": {
                        "view": "discriminator_stability",
                        "execution_backend": "cuda",
                        "cuda_model": "NVIDIA RTX A6000",
                    }
                },
                "task_results": [{"task_id": "nonexistent_tier2_task", "gate_status": "PASS"}],
            },
        )
        md1 = rti._current_source_tier1_readouts_markdown(
            live_inv, wt, wt / "reports/forge/technique-inventory.md"
        )
        assert "### Current-source selected-recipe Tier 1 readouts" in md1
        assert "readout-tier2-only.json" not in md1

        v = read_json(wt / "configs/forge/views/discriminator_stability.json")
        req_t1_tasks = [
            item["task"]
            for item in sorted(v["assignments"], key=lambda x: x.get("order", 0))
            if int(item["qualification_tier"]) == 1 and item.get("importance") == "required"
        ]
        first_t1_task, second_t1_task = req_t1_tasks[0], req_t1_tasks[1]
        task_card_path = wt / "configs/forge/tasks" / f"{first_t1_task}.json"
        task_card = read_json(task_card_path)
        task_card["requires_capabilities"] = sorted({*task_card.get("requires_capabilities", []), "a2"})
        atomic_json(task_card_path, task_card)

        second_card_path = wt / "configs/forge/tasks" / f"{second_t1_task}.json"
        second_card = read_json(second_card_path)
        second_card["dependencies"] = [{"task": first_t1_task, "kind": "gate"}]
        atomic_json(second_card_path, second_card)

        # Cache must invalidate on task-card edit and refuse stale-revision readout records.
        md_stale = rti._current_source_tier1_readouts_markdown(
            live_inv, wt, wt / "reports/forge/technique-inventory.md"
        )
        assert md_stale == ""

        # Re-stamp Atlas readout to the new source/revision (without second_t1_task) to verify _dep_blocked ('1 blocked').
        new_atlas_req = rti._resolve_idea_for_readout_check(
            wt,
            "atlas-particle-normal-v4",
            view_id="discriminator_stability",
            through_tier=1,
            cuda_model="NVIDIA RTX A6000",
            planning_mod=rfr.planning,
        )
        new_compat = {
            tid: job["compatibility_key"]
            for job in new_atlas_req.get("jobs", [])
            for tid in job.get("task_ids", [job["task_id"]])
        }
        for rec_p in sorted((wt / "reports/forge/records").glob("readout-*.json")):
            if rec_p.name == "readout-tier2-only.json":
                continue
            rec_doc = read_json(rec_p)
            if rec_doc.get("candidate_id") == "atlas-particle-normal-v4":
                rec_doc["candidate_revision"] = new_atlas_req["candidate_revision"]
                rec_doc["source"] = {"digest": new_atlas_req["source"]["digest"]}
                rec_doc["task_results"] = [
                    {**r, "compatibility_key": new_compat.get(r.get("task_id"), r.get("compatibility_key"))}
                    for r in rec_doc.get("task_results", [])
                    if r.get("task_id") != second_t1_task
                ]
                atomic_json(rec_p, rec_doc)

        md2 = rti._current_source_tier1_readouts_markdown(
            live_inv, wt, wt / "reports/forge/technique-inventory.md"
        )
        assert "1 preflight-blocked, 1 blocked" in md2

        # Even after all synthetic runnable families have matching records,
        # preflight and dependency blockers retain their operational statuses.
        selected_cids = {c["candidate_id"] for c in rfr.current_family_candidates(wt)}
        for rec_p in sorted((wt / "reports/forge/records").glob("readout-*.json")):
            if rec_p.name == "readout-tier2-only.json":
                continue
            rec_doc = read_json(rec_p)
            cid = rec_doc.get("candidate_id")
            if cid not in selected_cids:
                continue
            fam_req = rti._resolve_idea_for_readout_check(
                wt,
                cid,
                view_id="discriminator_stability",
                through_tier=1,
                cuda_model="NVIDIA RTX A6000",
                planning_mod=rfr.planning,
            )
            fam_compat = {
                tid: job["compatibility_key"]
                for job in fam_req.get("jobs", [])
                for tid in job.get("task_ids", [job["task_id"]])
            }
            rec_doc["candidate_revision"] = fam_req["candidate_revision"]
            rec_doc["source"] = {"digest": fam_req["source"]["digest"]}
            rec_doc["task_results"] = [
                {
                    **r,
                    "compatibility_key": fam_compat.get(r.get("task_id"), r.get("compatibility_key")),
                    **(
                        {"gate_status": "FAIL", "status": "FAIL"}
                        if r.get("task_id") == first_t1_task
                        else {}
                    ),
                }
                for r in rec_doc.get("task_results", [])
                if r.get("task_id") != second_t1_task
            ]
            atomic_json(rec_p, rec_doc)

        md3 = rti._current_source_tier1_readouts_markdown(
            live_inv, wt, wt / "reports/forge/technique-inventory.md"
        )
        assert "| Required Tier 1 (11) | Diagnostic Tier 1 (1) | Total Tier 1 (12) |" in md3
        assert "1 preflight-blocked, 1 blocked" in md3
        assert "binary PASS or FAIL" not in md3
    finally:
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(wt, ignore_errors=True)


def test_clean_cpu_ci_and_non_a6000_hosts_match_committed_readouts_and_invalidate_task_edits(
    tmp_path: Path, monkeypatch
):
    import shutil
    import subprocess
    from experiments.forge import planning, sources
    from experiments.forge.family_reports import generated_pages
    from reports.forge import regenerate_technique_inventory as rti

    wt = tmp_path / "wt-clean-cpu-ci"
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        # Copy only tracked working-tree updates into the detached worktree while keeping
        # gitignored runs/forge/, reports/forge/attempts/, and test-performance.{json,html} absent.
        _sync_tracked_worktree_edits(ROOT, wt)
        for rel in (
            "configs/forge/views/discriminator_stability.json",
            "ops/formulaic_coverage.py",
            "ops/refresh_forge_readouts.py",
            "reports/forge/regenerate_technique_inventory.py",
            "reports/forge/attempt-coverage.json",
            "reports/forge/ATTEMPT_COVERAGE.md",
            "reports/forge/technique-inventory.json",
            "reports/forge/technique-inventory.md",
            "reports/forge/test-performance.md",
            "reports/forge/EXPERIMENTS_BY_TIER.md",
            "reports/forge/EXPERIMENT_MEMORY.md",
            "reports/forge/compilation.json",
            "reports/forge/problems.md",
        ):
            src_file = ROOT / rel
            if src_file.is_file():
                dst_file = wt / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
        fam_src = ROOT / "reports/forge/families"
        if fam_src.is_dir():
            for p in sorted(fam_src.glob("*.md")):
                shutil.copy2(p, wt / "reports/forge/families" / p.name)

        assert not (wt / "runs/forge").exists()
        assert not (wt / "reports/forge/test-performance.json").exists()
        assert not (wt / "reports/forge/test-performance.html").exists()

        # 1. Simulate GitHub Actions clean CPU-only CI: Python 3.11 +cpu packages, torch.__version__ +cpu, and no nvidia-smi.
        import torch

        cpu_ci_runtime = {
            "python": "3.11.9",
            "implementation": "CPython",
            "system": "Linux",
            "machine": "x86_64",
            "packages": {"torch": "2.4.0+cpu", "numpy": "1.26.4", "scipy": "1.13.1"},
        }
        if hasattr(sources._gpu_models, "cache_clear"):
            sources._gpu_models.cache_clear()
        rti._CURRENT_SOURCE_TIER1_CACHE.clear()
        rti._CONCLUDED_TIER1_RECORDS_CACHE.clear()
        monkeypatch.setattr(torch, "__version__", "2.4.0+cpu")
        monkeypatch.setattr(sources, "_gpu_models", lambda: None)
        monkeypatch.setattr(sources, "runtime_manifest", lambda: deepcopy(cpu_ci_runtime))
        monkeypatch.setattr(planning, "runtime_manifest", lambda: deepcopy(cpu_ci_runtime))

        # Resolve the latest checkout under the CPU CI runtime before checking
        # the original GPU-workstation receipts in the separate pinned cohort.
        probed_req = planning.resolve_idea(
            wt,
            "atlas-particle-normal-v4",
            view_id="discriminator_stability",
            through_tier=1,
            freeze_source=False,
            execution_backend="cuda",
            cuda_model="NVIDIA RTX A6000",
        )
        assert probed_req["runtime"]["python"] == "3.11.9"
        assert probed_req["rng"]["runtime"]["torch"] == "2.4.0+cpu"
        assert probed_req["compute_profiles"]["cuda"]["availability"] == "unavailable"
        # Published integrity is portable; latest-source measurement gaps remain explicit.
        status_cpu = rfr.check_published_status(wt)
        assert status_cpu["status"] == "CURRENT", json.dumps(status_cpu, indent=2, sort_keys=True)
        assert status_cpu["stale_stages"] == []
        _assert_current_source_scope(status_cpu, wt)

        wt_inv = read_json(wt / "reports/forge/technique-inventory.json")
        pages = generated_pages(wt, wt_inv)
        overview_path = wt / "reports/forge/technique-inventory.md"
        pages[overview_path] = rti._current_markdown(wt_inv, wt, overview_path)
        for page_path, expected_text in pages.items():
            assert page_path.read_text() == expected_text, page_path

        # 2. Simulate a non-A6000 GPU host (e.g. RTX 4090) where compute_profile("cuda", "NVIDIA RTX A6000") raises ValueError.
        monkeypatch.setattr(
            sources,
            "_gpu_models",
            lambda: frozenset({("NVIDIA GeForce RTX 4090", "560.35.03", "8.9")}),
        )
        status_4090 = rfr.check_published_status(wt)
        assert status_4090["status"] == "CURRENT", json.dumps(status_4090, indent=2, sort_keys=True)
        assert rti._current_markdown(wt_inv, wt, overview_path) == overview_path.read_text()

        # Restore CPU-only CI probe for invalidation checks below.
        monkeypatch.setattr(sources, "_gpu_models", lambda: None)

        # The following task/runtime tamper controls use their exact recorded
        # source cohort, independently of the latest checkout tested above.
        recorded_digest = _restore_recorded_source_cohort(wt)
        wt_inv = read_json(wt / "reports/forge/technique-inventory.json")
        # The complete old view remains incomplete: Atlas has no compatible
        # ordinary Tier 2 observations. Diagnostic coverage must not fill those
        # gaps. Scope the software controls below to the retained Tier 1 rows,
        # rather than inventing later-tier measurements or qualification.
        full_view_status = rfr.check_selected_family_readouts(wt)
        assert full_view_status["current"] is False
        atlas_gap, = full_view_status["missing_families"]
        assert atlas_gap["family"] == "atlas"
        assert atlas_gap["unmeasured_tier1"] == []
        assert atlas_gap["unmeasured_tier2"]
        view_path = wt / "configs/forge/views/discriminator_stability.json"
        software_view = read_json(view_path)
        software_view["assignments"] = [item for item in software_view["assignments"]
                                        if int(item["qualification_tier"]) == 1]
        atomic_json(view_path, software_view)
        rti._CURRENT_SOURCE_TIER1_CACHE.clear()
        probed_req = planning.resolve_idea(
            wt, "atlas-particle-normal-v4", view_id="discriminator_stability", through_tier=1,
            freeze_source=False, execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
        probed_compat = {
            tid: job["compatibility_key"]
            for job in probed_req["jobs"]
            for tid in job.get("task_ids", [job["task_id"]])
        }
        atlas_rec_path = next(
            p
            for p in sorted((wt / "reports/forge/records").glob("readout-*.json"))
            if read_json(p).get("lifecycle") == "concluded"
            and read_json(p).get("evidence_scope") == "current"
            and read_json(p).get("candidate_id") == "atlas-particle-normal-v4"
            and read_json(p).get("candidate_revision") == probed_req["candidate_revision"]
        )
        atlas_rec = read_json(atlas_rec_path)
        for row in atlas_rec["task_results"]:
            assert not (wt / "reports/forge/attempts" / row["_attempt_id"]).exists()
            # Out-of-scope historical rows remain present, without supplying
            # measurements to this Tier 1 software compatibility control.
            if row["task_id"] in probed_compat:
                assert row["compatibility_key"] != probed_compat[row["task_id"]]
        overview_path.write_text(rti._current_markdown(wt_inv, wt, overview_path))
        assert rfr.check_selected_family_readouts(wt)["current"] is True

        # 3. Editing a task card NOT listed in source.files (zero_tanh_smoke.json) keeps source_digest unchanged
        #    yet invalidates the stale readout row on the clean CPU worktree via attempt-coverage.json fingerprints
        #    without needing manual _CURRENT_SOURCE_TIER1_CACHE.clear().
        zt_path = wt / "configs/forge/tasks/zero_tanh_smoke.json"
        orig_zt_bytes = zt_path.read_bytes()
        zt_doc = read_json(zt_path)
        zt_doc["execution"]["steps"] = int(zt_doc["execution"]["steps"]) + 10
        atomic_json(zt_path, zt_doc)

        assert rfr.catalog_source_manifest(wt)["digest"] == recorded_digest
        sfr_after_edit = rfr.check_selected_family_readouts(wt)
        assert sfr_after_edit["current"] is False
        assert len(sfr_after_edit["missing_families"]) == 8
        for item in sfr_after_edit["missing_families"]:
            assert item["unmeasured_tier1"] == ["zero_tanh_smoke"]

        tier1_md_after_edit = rti._current_source_tier1_readouts_markdown(wt_inv, wt, overview_path)
        assert "1 not run" in tier1_md_after_edit
        assert rfr.check_compiled_reports(wt)["current"] is False

        # Restore zero_tanh_smoke.json and confirm CURRENT is restored automatically.
        zt_path.write_bytes(orig_zt_bytes)
        assert rfr.check_selected_family_readouts(wt)["current"] is True
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True

        # 4. Non-hex garbage compatibility_key on a clean checkout is rejected automatically.
        orig_atlas_bytes = atlas_rec_path.read_bytes()
        bad_ck_rec = deepcopy(atlas_rec)
        for row in bad_ck_rec["task_results"]:
            if row["task_id"] == "zero_tanh_smoke":
                row["compatibility_key"] = "ck-t1-STALE"
        atomic_json(atlas_rec_path, bad_ck_rec)
        sfr_bad_ck = rfr.check_selected_family_readouts(wt)
        assert sfr_bad_ck["current"] is False
        assert sfr_bad_ck["missing_families"] == [
            {
                "family": "atlas",
                "candidate_id": "atlas-particle-normal-v4",
                "candidate_revision": probed_req["candidate_revision"],
                "unmeasured_tier1": ["zero_tanh_smoke"],
                "unmeasured_tier2": [],
                "unmeasured_by_tier": {"1": ["zero_tanh_smoke"]},
            }
        ]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False
        atlas_rec_path.write_bytes(orig_atlas_bytes)
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True

        # 5. When a durable local attempt request.json exists on disk, full compatibility_key hash reconstruction
        #    is enforced (including sci["rng"]["runtime"]["torch"] when host torch.__version__ is +cpu)
        #    AND _CURRENT_SOURCE_TIER1_CACHE invalidates automatically on local envelope changes.
        zt_row = next(r for r in atlas_rec["task_results"] if r["task_id"] == "zero_tanh_smoke")
        zt_aid = zt_row["_attempt_id"]
        local_req_path = wt / "reports/forge/attempts" / zt_aid / "request.json"
        valid_workstation_envelope = _readout_workstation_envelope_fixture()
        atomic_json(local_req_path, valid_workstation_envelope)
        assert rfr.check_selected_family_readouts(wt)["current"] is True
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True

        # Non-A6000 GPU host (where _resolve_idea_for_readout_check falls back to execution_backend="cpu")
        # also reconstructs CUDA compatibility_key cleanly when durable local request.json is present.
        monkeypatch.setattr(
            sources,
            "_gpu_models",
            lambda: frozenset({("NVIDIA GeForce RTX 4090", "560.35.03", "8.9")}),
        )
        assert rfr.check_selected_family_readouts(wt)["current"] is True
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True
        monkeypatch.setattr(sources, "_gpu_models", lambda: None)

        # Fallback to job.science.{runtime,compute,rng.runtime} when request wrapper omits them.
        job_only_envelope = {
            "schema_version": 1,
            "job": {
                "science": {
                    "runtime": deepcopy(valid_workstation_envelope["request"]["runtime"]),
                    "rng": {"runtime": deepcopy(valid_workstation_envelope["request"]["rng"]["runtime"])},
                    "compute": deepcopy(valid_workstation_envelope["request"]["compute_profiles"]["cuda"]),
                }
            },
        }
        atomic_json(local_req_path, job_only_envelope)
        assert rfr.check_selected_family_readouts(wt)["current"] is True
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True

        # Full dual (request + job.science) envelope matches when consistent, and fails closed if either side
        # is tampered or malformed (never silently bypassing a tampered/corrupt side via fallback).
        dual_envelope = {
            "schema_version": 1,
            "request": deepcopy(valid_workstation_envelope["request"]),
            "job": deepcopy(job_only_envelope["job"]),
        }
        atomic_json(local_req_path, dual_envelope)
        assert rfr.check_selected_family_readouts(wt)["current"] is True
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is True

        tampered_job_rng_envelope = deepcopy(dual_envelope)
        tampered_job_rng_envelope["job"]["science"]["rng"]["runtime"]["torch"] = "2.99.0-tampered"
        atomic_json(local_req_path, tampered_job_rng_envelope)
        sfr_tampered_job_rng = rfr.check_selected_family_readouts(wt)
        assert sfr_tampered_job_rng["current"] is False
        assert sfr_tampered_job_rng["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        malformed_req_rt_dual = deepcopy(dual_envelope)
        malformed_req_rt_dual["request"]["runtime"] = "corrupt-non-dict"
        atomic_json(local_req_path, malformed_req_rt_dual)
        sfr_malformed_req_rt = rfr.check_selected_family_readouts(wt)
        assert sfr_malformed_req_rt["current"] is False
        assert sfr_malformed_req_rt["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        malformed_job_sci_envelope = {"schema_version": 1, "job": {"science": "corrupt-string"}}
        atomic_json(local_req_path, malformed_job_sci_envelope)
        sfr_malformed_job_sci = rfr.check_selected_family_readouts(wt)
        assert sfr_malformed_job_sci["current"] is False
        assert sfr_malformed_job_sci["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        tampered_rng_envelope = deepcopy(valid_workstation_envelope)
        tampered_rng_envelope["request"]["rng"]["runtime"]["torch"] = "2.99.0-tampered"
        atomic_json(local_req_path, tampered_rng_envelope)
        sfr_tampered_rng = rfr.check_selected_family_readouts(wt)
        assert sfr_tampered_rng["current"] is False
        assert sfr_tampered_rng["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        malformed_rng_envelope = deepcopy(valid_workstation_envelope)
        malformed_rng_envelope["request"]["rng"]["runtime"] = "corrupt-non-dict"
        atomic_json(local_req_path, malformed_rng_envelope)
        sfr_malformed_rng = rfr.check_selected_family_readouts(wt)
        assert sfr_malformed_rng["current"] is False
        assert sfr_malformed_rng["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        tampered_envelope = deepcopy(valid_workstation_envelope)
        tampered_envelope["request"]["runtime"]["python"] = "3.99.0-tampered"
        atomic_json(local_req_path, tampered_envelope)
        sfr_tampered_local = rfr.check_selected_family_readouts(wt)
        assert sfr_tampered_local["current"] is False
        assert sfr_tampered_local["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        local_req_path.write_text("{corrupt-attempt-request-json\n")
        sfr_corrupt_local = rfr.check_selected_family_readouts(wt)
        assert sfr_corrupt_local["current"] is False
        assert sfr_corrupt_local["missing_families"][0]["unmeasured_tier1"] == ["zero_tanh_smoke"]
        assert rfr.check_compiled_reports(wt)["current_source_tier1_synced"] is False

        # 6. Unit-level edge cases on _ReadoutCompatibilityMatcher:
        #    (a) rec["attempt_ids"] fallback with partial local attempts uses result.json task ownership so a corrupt
        #        attempt for task_a fails closed on task_a without poisoning task_b whose attempt is absent locally;
        #    (b) editing a sibling task inside a multi-task prerequisite execution group invalidates downstream tasks;
        #    (c) malformed job.task_ids (str or int instead of list) fails closed without TypeError or char-splitting bypass.
        unit_root = tmp_path / "matcher-unit"
        (unit_root / "reports/forge/attempts/aid_a").mkdir(parents=True)
        (unit_root / "reports/forge/attempts/aid_a/request.json").write_text("{corrupt\n")
        atomic_json(
            unit_root / "reports/forge/attempts/aid_a/result.json",
            {"task_results": [{"task_id": "task_a"}]},
        )
        (unit_root / "reports/forge/attempts/aid_bad_str_ids").mkdir(parents=True)
        atomic_json(
            unit_root / "reports/forge/attempts/aid_bad_str_ids/request.json",
            {"job": {"task_ids": "task_a"}},
        )
        (unit_root / "reports/forge/attempts/aid_bad_int_ids").mkdir(parents=True)
        atomic_json(
            unit_root / "reports/forge/attempts/aid_bad_int_ids/request.json",
            {"job": {"task_ids": 123}},
        )
        unit_req = {
            "source": {"digest": "a" * 64},
            "jobs": [
                {
                    "task_id": "task_a",
                    "task_ids": ["task_a"],
                    "execution_group": "task_a",
                    "compatibility_key": "1" * 64,
                    "resources": {"gpus": 1, "backend": "cuda"},
                    "science": {"execution": {"task_a": "ex_a"}, "evaluation": {"task_a": "ev_a"}, "compute": {"threads": 1}},
                },
                {
                    "task_id": "task_b",
                    "task_ids": ["task_b"],
                    "execution_group": "task_b",
                    "compatibility_key": "2" * 64,
                    "resources": {"gpus": 1, "backend": "cuda"},
                    "science": {"execution": {"task_b": "ex_b"}, "evaluation": {"task_b": "ev_b"}, "compute": {"threads": 1}},
                },
                {
                    "task_id": "ring_hold",
                    "task_ids": ["ring_hold", "ring_extension"],
                    "execution_group": "ring_group",
                    "compatibility_key": "3" * 64,
                    "resources": {"gpus": 1, "backend": "cuda"},
                    "science": {
                        "execution": {"ring_hold": "ex_rh", "ring_extension": "ex_re_EDITED"},
                        "evaluation": {"ring_hold": "ev_rh", "ring_extension": "ev_re"},
                        "compute": {"threads": 1},
                    },
                },
                {
                    "task_id": "ring_downstream",
                    "task_ids": ["ring_downstream"],
                    "execution_group": "ring_downstream",
                    "compatibility_key": "4" * 64,
                    "resources": {"gpus": 1, "backend": "cuda"},
                    "science": {
                        "execution": {"ring_downstream": "ex_rd"},
                        "evaluation": {"ring_downstream": "ev_rd"},
                        "prerequisites": {"ring_hold": "3" * 64},
                        "compute": {"threads": 1},
                    },
                },
            ],
        }
        unit_cov = {
            ("a" * 64, "task_a"): {("ex_a", "ev_a")},
            ("a" * 64, "task_b"): {("ex_b", "ev_b")},
            ("a" * 64, "ring_hold"): {("ex_rh", "ev_rh")},
            ("a" * 64, "ring_extension"): {("ex_re_ORIG", "ev_re")},
            ("a" * 64, "ring_downstream"): {("ex_rd", "ev_rd")},
        }
        unit_matcher = rti._ReadoutCompatibilityMatcher(
            unit_root, unit_req, coverage_fps=unit_cov, planning_mod=planning
        )
        unit_rec = {"attempt_ids": ["aid_a", "aid_b_absent"]}
        assert unit_matcher.matches(unit_rec, {"task_id": "task_a", "compatibility_key": "c" * 64}, "task_a") is False
        assert unit_matcher.matches(unit_rec, {"task_id": "task_b", "compatibility_key": "d" * 64}, "task_b") is True
        assert (
            unit_matcher.matches(
                {"attempt_ids": ["aid_bad_str_ids"]},
                {"task_id": "task_a", "compatibility_key": "c" * 64},
                "task_a",
            )
            is False
        )
        assert (
            unit_matcher.matches(
                {"attempt_ids": ["aid_bad_int_ids"]},
                {"task_id": "task_a", "compatibility_key": "c" * 64},
                "task_a",
            )
            is False
        )
        assert (
            unit_matcher.matches(
                {"attempt_ids": ["aid_absent"]},
                {"task_id": "ring_downstream", "compatibility_key": "e" * 64},
                "ring_downstream",
            )
            is False
        )
    finally:
        if hasattr(sources._gpu_models, "cache_clear"):
            sources._gpu_models.cache_clear()
        rti._CURRENT_SOURCE_TIER1_CACHE.clear()
        rti._CONCLUDED_TIER1_RECORDS_CACHE.clear()
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(wt, ignore_errors=True)


def test_refresh_attempt_coverage_delegates_to_formulaic_coverage_and_preserves_grid100_deep_recipes(
    tmp_path: Path, monkeypatch
):
    from ops import formulaic_coverage as fc

    called: list[tuple[Path, Path, list[str], bool]] = []

    def _fake_ensure(root: Path, attempt_state_root: Path, gpus: list[str], *, json_mode: bool = False):
        called.append((root, attempt_state_root, list(gpus), json_mode))
        return {"updated": 0, "status": "ok"}

    monkeypatch.setattr(fc, "ensure_formulaic_attempt_coverage", _fake_ensure)
    from experiments.forge.leaderboard_contract import FIXED_TEST_LIST
    full_suite_rel = FIXED_TEST_LIST
    atomic_json(tmp_path / full_suite_rel, {"tests": ["gaussian1d_smoke"]})
    out = rfr.refresh_attempt_coverage(
        tmp_path,
        tmp_path / "runs/forge",
        tmp_path / "runs/forge/attempt-coverage-refresh",
        gpus=["0", "1"],
        json_mode=True,
    )
    assert out == {"updated": 0, "status": "ok"}
    assert called == [
        (
            tmp_path.resolve(),
            (tmp_path / "runs/forge/attempt-coverage-refresh").resolve(),
            ["0", "1"],
            True,
        )
    ]

    # Coverage refresh cannot use the old unregistered path to change a recipe.
    for cid in ("atlas-particle-normal-v4", "e22"):
        card = fc.load_candidate_card(ROOT, cid)
        for task in ({"id": "grid100_mlp16"}, {"id": "model_ddgan_gaussian_chain"}):
            assert fc.adapt_candidate_for_task(card, task) == card
