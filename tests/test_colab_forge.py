"""Offline unit tests for ops/colab_forge.py (decentralized branch catch-up, right-sizing, safety)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from ops import colab_forge as cf


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_receipt(root: Path, attempt_id: str, candidate_id: str, task_id: str, *, status: str = "PASS", with_energy: bool = False, gpu_name: str | None = None, source_digest: str = "src1") -> None:
    cost: dict = {"wall_seconds": 12.0, "completed_steps": 100}
    if with_energy:
        cost["energy"] = {
            "joules_total": 420.0,
            "joules_per_step": 4.2,
            "avg_power_watts": 35.0,
            "gpu_name": gpu_name or "NVIDIA L4",
            "energy_backend": "nvml_energy_counter",
            "contended": False,
        }
    _write_json(
        root / "reports/forge/technique-receipts" / f"{attempt_id}.json",
        {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "candidate_revision": f"rev-{candidate_id}",
            "provenance": {"source_digest": source_digest},
            "task_results": [
                {
                    "task_id": task_id,
                    "gate_status": status,
                    "status": status,
                    "cost": cost,
                }
            ],
        },
    )


def test_gpu_right_sizing_auto_and_explicit():
    light_task = {"resources": {"timeout_seconds": 120, "gpus": 1}, "execution": {"steps": 400}}
    std_task = {"resources": {"timeout_seconds": 600, "gpus": 1}, "execution": {"steps": 2000}}
    heavy_task = {"resources": {"timeout_seconds": 900, "gpus": 1}, "execution": {"steps": 20001}}
    cpu_task = {"resources": {"timeout_seconds": 120, "gpus": 0}, "execution": {"steps": 100}}

    assert cf.choose_gpu_for_task("two_pole", light_task, tier=1, gpu_policy="auto", prefer_t4=True) == "T4"
    assert cf.choose_gpu_for_task("ring16_acquisition", std_task, tier=1, gpu_policy="auto") == "L4"
    assert cf.choose_gpu_for_task("five_word_joint_smoke", heavy_task, tier=1, gpu_policy="auto") == "A100"
    assert cf.choose_gpu_for_task("grid100_14k", std_task, tier=3, gpu_policy="auto") == "A100"
    assert cf.choose_gpu_for_task("cpu_only", cpu_task, tier=1, gpu_policy="auto") == "cpu"
    assert cf.choose_gpu_for_task("five_word_joint_smoke", heavy_task, tier=1, gpu_policy="L4") == "L4"


def test_hard_timeout_kills_hung_process():
    rc, out = cf.run_with_hard_timeout(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds=0.3,
        kill_grace=0.5,
        capture=True,
    )
    assert rc == 124
    assert "Hard-killing hung command" in out


def test_decentralized_branch_merge_catchup_only_fills_cross_cell(tmp_path: Path, monkeypatch):
    """Simulate Branch A (+new_task), Branch B (+new_family), and Merge C -> only (new_family, new_task) misses!"""
    # Minimal mock Forge declarations
    for cid in ("fam-a", "fam-b"):
        _write_json(tmp_path / "configs/forge/ideas" / f"{cid}.json", {"id": cid})
    for tid in ("task_1", "task_2"):
        _write_json(
            tmp_path / "configs/forge/tasks" / f"{tid}.json",
            {"id": tid, "resources": {"timeout_seconds": 300, "gpus": 1}, "execution": {"steps": 500}},
        )
    _write_json(
        tmp_path / "configs/forge/views/test_view.json",
        {
            "id": "test_view",
            "revision": 1,
            "assignments": [
                {"task": "task_1", "qualification_tier": 1, "importance": "required"},
                {"task": "task_2", "qualification_tier": 1, "importance": "required"},
            ],
        },
    )

    # Monkeypatch Forge loaders so this test is 100% isolated in tmp_path
    monkeypatch.setattr(cf, "_git_head", lambda _root: "deadbeef" * 5)
    monkeypatch.setattr("experiments.forge.sources.inspect_source", lambda _root: {"digest": "src1"})
    monkeypatch.setattr(
        "experiments.forge.planning.load_idea",
        lambda root, cid: json.loads((root / "configs/forge/ideas" / f"{cid}.json").read_text()),
    )
    monkeypatch.setattr(
        "experiments.forge.views.load_view",
        lambda root, vid: json.loads((root / "configs/forge/views" / f"{vid}.json").read_text()),
    )
    monkeypatch.setattr(
        "experiments.forge.views.load_tasks",
        lambda root: {p.stem: json.loads(p.read_text()) for p in (root / "configs/forge/tasks").glob("*.json")},
    )

    # Base (caught up): (fam-a, task_1) cached
    _write_receipt(tmp_path, "a001", "fam-a", "task_1", with_energy=True)
    # Branch A added task_2 and caught up existing family fam-a -> (fam-a, task_2) cached
    _write_receipt(tmp_path, "a002", "fam-a", "task_2", with_energy=True)
    # Branch B added fam-b and caught up existing task task_1 -> (fam-b, task_1) cached
    _write_receipt(tmp_path, "b001", "fam-b", "task_1", with_energy=True)

    # Merged Branch C: has both fam-a, fam-b AND both task_1, task_2
    plan = cf.compute_catchup_plan(
        tmp_path,
        view_id="test_view",
        through_tier=1,
        candidates=["fam-a", "fam-b"],
        include_selected_families=False,
        gpu_policy="L4",
    )
    assert plan["total_cells"] == 4
    assert plan["cached_count"] == 3
    assert plan["missing_count"] == 1
    assert [(m["candidate_id"], m["task_id"]) for m in plan["missing_cells"]] == [("fam-b", "task_2")]
    assert plan["work_unit_count"] == 1
    assert plan["work_units"][0]["tasks"] == ["task_2"]
    assert plan["work_units"][0]["is_full_view_slice"] is False

    # Energy mode catches up cells missing NVML energy even if status==PASS
    _write_receipt(tmp_path, "b002", "fam-b", "task_2", with_energy=False)
    plan_branch = cf.compute_catchup_plan(
        tmp_path,
        view_id="test_view",
        through_tier=1,
        candidates=["fam-a", "fam-b"],
        include_selected_families=False,
        gpu_policy="L4",
        require_energy=False,
    )
    assert plan_branch["missing_count"] == 0

    plan_energy = cf.compute_catchup_plan(
        tmp_path,
        view_id="test_view",
        through_tier=1,
        candidates=["fam-a", "fam-b"],
        include_selected_families=False,
        gpu_policy="L4",
        require_energy=True,
        changed_since=None,
    )
    assert [(m["candidate_id"], m["task_id"]) for m in plan_energy["missing_cells"]] == [("fam-b", "task_2")]

    # Smart change invalidation: editing task_2 invalidates ONLY task_2 across families (when source_digest differs)
    monkeypatch.setattr(cf, "detect_branch_changes", lambda _root, _ref: (set(), {"task_2"}))
    monkeypatch.setattr("experiments.forge.sources.inspect_source", lambda _root: {"digest": "src2"})
    plan_changed = cf.compute_catchup_plan(
        tmp_path,
        view_id="test_view",
        through_tier=1,
        candidates=["fam-a", "fam-b"],
        include_selected_families=False,
        gpu_policy="L4",
        changed_since="origin/develop",
    )
    assert sorted((m["candidate_id"], m["task_id"]) for m in plan_changed["missing_cells"]) == [
        ("fam-a", "task_2"),
        ("fam-b", "task_2"),
    ]
    assert plan_changed["cached_count"] == 2  # task_1 stays cached for both families!

    # Static preflight blocker skips unsupported cells instead of wasting GPUs
    monkeypatch.setattr(
        cf,
        "_static_task_blockers",
        lambda _r, idea, _v, _t, **_kw: ([], {"task_2": ["unsupported prior"]} if idea["id"] == "fam-b" else {}),
    )
    plan_blocked = cf.compute_catchup_plan(
        tmp_path,
        view_id="test_view",
        through_tier=1,
        candidates=["fam-a", "fam-b"],
        include_selected_families=False,
        gpu_policy="L4",
        changed_since="origin/develop",
    )
    assert plan_blocked["blocked_count"] == 1
    assert [(m["candidate_id"], m["task_id"]) for m in plan_blocked["missing_cells"]] == [("fam-a", "task_2")]


def test_ingest_tarball_rejects_traversal(tmp_path: Path):
    tar_path = tmp_path / "bad.tar.gz"
    payload = tmp_path / "evil.txt"
    payload.write_text("boom", encoding="utf-8")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(payload, arcname="reports/forge/attempts/../../evil.txt")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        cf.ingest_tarball(tmp_path, tar_path, project_receipts=False, register_submissions=False)


def test_ephemeral_catchup_study_passes_forge_validator(tmp_path: Path):
    from experiments.forge.studies import validate_study

    unit = cf.CatchupUnit(
        unit_id="bispf-l4-12345678",
        candidate_id="bispf-optimized-quintic-v1",
        view_id="discriminator_stability",
        through_tier=2,
        gpu="L4",
        tasks=["precision_gaussian_fp32", "nash_overpowered_critic_ring16"],
        is_full_view_slice=False,
        estimated_seconds=600.0,
    )
    view_file, study_file, study_id = cf.build_ephemeral_catchup_bundle(cf.REPO_ROOT, unit, tmp_path)
    assert view_file is not None and view_file.is_file()
    assert study_file.is_file()
    study_doc = json.loads(study_file.read_text(encoding="utf-8"))
    assert study_doc["id"] == study_id
    validate_study(study_doc)
    # Also verify prior_evidence file exists and matches declared identity
    pe = study_doc["prior_evidence"][0]
    target = json.loads((cf.REPO_ROOT / pe["path"]).read_text(encoding="utf-8"))
    for key, expected in pe["identity"].items():
        assert target.get(key) == expected


def test_ephemeral_view_includes_transitive_dependencies(tmp_path: Path):
    unit = cf.CatchupUnit(
        unit_id="empf-l4-dep12345",
        candidate_id="empf-v8",
        view_id="discriminator_stability",
        through_tier=2,
        gpu="L4",
        tasks=["model_ae_ddgan_bimodal_chain"],
        is_full_view_slice=False,
        estimated_seconds=1800.0,
    )
    view_file, _study_file, _sid = cf.build_ephemeral_catchup_bundle(cf.REPO_ROOT, unit, tmp_path)
    assert view_file is not None
    vdoc = json.loads(view_file.read_text(encoding="utf-8"))
    assigned = {a["task"]: a["qualification_tier"] for a in vdoc["assignments"]}
    assert "model_ae_ddgan_bimodal_chain" in assigned
    assert "model_ae_ddgan_gaussian_chain" in assigned
    assert set(assigned.values()) == {1}

    # Also verify Tier-4 held task (grid100_mlp16) produces a valid contiguous required Tier-1+ view
    from experiments.forge.views import load_tasks, validate_view

    all_tasks = load_tasks(cf.REPO_ROOT)
    for tid in (
        "model_ae_ddgan_bimodal_chain",
        "grid100_mlp16",
        "grid100_res16_scaled_tanh",
        "nash_overpowered_critic_ring16",
        "zero_tanh_rail_escape",
        "ring_extension",
    ):
        unit_chk = cf.CatchupUnit(
            unit_id=f"chk-{tid[:12]}",
            candidate_id="unified-v11",
            view_id="discriminator_stability",
            through_tier=4,
            gpu="L4",
            tasks=[tid],
            is_full_view_slice=False,
            estimated_seconds=3600.0,
        )
        vf, sf, _sid = cf.build_ephemeral_catchup_bundle(cf.REPO_ROOT, unit_chk, tmp_path)
        assert vf is not None
        vdoc_chk = json.loads(vf.read_text(encoding="utf-8"))
        sdoc_chk = json.loads(sf.read_text(encoding="utf-8"))
        validate_view(vdoc_chk, all_tasks)
        tiers = {a["qualification_tier"] for a in vdoc_chk["assignments"]}
        assert tiers == set(range(1, len(tiers) + 1))
        assert sdoc_chk["scope"]["through_tier"] == len(tiers)


def test_ephemeral_view_preserves_diagnostic_scope_for_four_laws_adherence(tmp_path: Path):
    from experiments.forge.studies import validate_study
    from experiments.forge.views import load_tasks, validate_view

    all_tasks = load_tasks(cf.REPO_ROOT)
    for tasks in (
        ["fl_law1_aniso_unequal_modes", "fl_law1_noisy_atomic_codebook", "fl_law2_pure_adversarial_codim1"],
        ["ring_extension"],
        ["zero_tanh_rail_escape"],
        ["gaussian1d_stability"],
        ["dcgan_performance_r256"],
    ):
        unit = cf.CatchupUnit(
            unit_id=f"fl-{tasks[0][:12]}",
            candidate_id="unified-v14",
            view_id="four-laws-adherence-v1",
            through_tier=1,
            gpu="L4",
            tasks=tasks,
            is_full_view_slice=False,
            estimated_seconds=1800.0,
        )
        vf, sf, _sid = cf.build_ephemeral_catchup_bundle(cf.REPO_ROOT, unit, tmp_path)
        assert vf is not None
        vdoc = json.loads(vf.read_text(encoding="utf-8"))
        sdoc = json.loads(sf.read_text(encoding="utf-8"))
        validate_view(vdoc, all_tasks)
        validate_study(sdoc)
        assert vdoc["evidence_scope"] == "research_diagnostic"
        assert all(a["importance"] == "diagnostic" and a["qualification_tier"] == 1 for a in vdoc["assignments"])
        assert sdoc["scope"]["through_tier"] == 1
        assert "law_groups" not in vdoc


# ---------------------------------------------------------------------------
# Hardened cmd_reap (default + isolated worker configs) & transport retries
# ---------------------------------------------------------------------------


def test_parse_colab_sessions_output_formats():
    sample = (
        "[colab] Pruned 1 stale local session(s).\n"
        "[default-l4] m-s-aaa111 | Hardware: L4 | Shape: Standard | Variant: GPU\n"
        "[?] m-s-bbb222 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
        "legacy-sess | Hardware: T4\n"
        "[2026-11-23T22:00:00Z] [timeout] Hard-killing hung command after 30s: colab sessions\n"
    )
    rows = cf._parse_colab_sessions_output(sample)
    assert rows == [
        {"name": "default-l4", "endpoint": "m-s-aaa111"},
        {"name": None, "endpoint": "m-s-bbb222"},
        {"name": "legacy-sess", "endpoint": None},
    ]


def test_cmd_reap_stops_default_and_isolated_configs_deduplicated(tmp_path: Path, monkeypatch, capsys):
    colab_root = tmp_path / "runs/forge/colab"

    # Older run with duplicate endpoint m-s-iso1 and a stale dead endpoint m-s-dead
    old_run = colab_root / "catchup-20261122-010000"
    old_dup_cfg = old_run / "forge-l4-w1-old.colab.json"
    _write_json(
        old_dup_cfg,
        {
            "forge-l4-w1-old": {"name": "forge-l4-w1-old", "endpoint": "m-s-iso1"},
            "forge-l4-dead": {"name": "forge-l4-dead", "endpoint": "m-s-dead"},
        },
    )
    # Empty (already cleanly stopped) and malformed configs should be skipped cleanly
    _write_json(old_run / "forge-l4-clean.colab.json", {})
    (old_run / "forge-l4-corrupt.colab.json").write_text("{not-json", encoding="utf-8")

    # Newer run with active isolated session m-s-iso1 + already-in-default m-s-def1
    new_run = colab_root / "catchup-20261123-020000"
    new_cfg = new_run / "forge-l4-w1-new.colab.json"
    _write_json(
        new_cfg,
        {
            "forge-l4-w1-new": {"name": "forge-l4-w1-new", "endpoint": "m-s-iso1"},
            "dup-default": {"name": "dup-default", "endpoint": "m-s-def1"},
        },
    )
    # Ensure deterministic mtime ordering (new_run strictly newer than old_run)
    os.utime(old_dup_cfg, (1_700_000_000, 1_700_000_000))
    os.utime(old_run, (1_700_000_000, 1_700_000_000))
    os.utime(new_cfg, (1_700_010_000, 1_700_010_000))
    os.utime(new_run, (1_700_010_000, 1_700_010_000))

    calls: list[tuple[list[str], float]] = []

    def fake_run(cmd: list[str], *, timeout_seconds: float, **_kwargs) -> tuple[int, str]:
        calls.append((list(cmd), timeout_seconds))
        if cmd == ["colab", "sessions"]:
            if sum(1 for c, _ in calls if c == ["colab", "sessions"]) == 1:
                body = (
                    "[default-l4] m-s-def1 | Hardware: L4 | Shape: Standard | Variant: GPU\n"
                    "[?] m-s-iso1 | Hardware: L4 | Shape: Standard | Variant: GPU\n"
                    "[?] m-s-orphan9 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
                )
                return 0, body
            return 0, "[colab] No active sessions found on server.\n"
        if cmd == ["colab", "usage"]:
            return 0, "Balance: 100.00 CU\n"
        if cmd == ["colab", "stop", "-s", "default-l4"]:
            return 0, "[colab] Session terminated.\n"
        if cmd == ["colab", "--config", str(new_cfg), "stop", "-s", "forge-l4-w1-new"]:
            return 0, "[colab] Session terminated.\n"
        raise AssertionError(f"Unexpected colab call: {cmd}")

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(cf.time, "sleep", lambda _s: None)

    rc = cf.cmd_reap(argparse.Namespace(root=tmp_path, max_runs=10, stop_timeout=25.0))
    assert rc == 0

    stop_calls = [(c, t) for c, t in calls if "stop" in c]
    assert stop_calls == [
        (["colab", "stop", "-s", "default-l4"], 25.0),
        (["colab", "--config", str(new_cfg), "stop", "-s", "forge-l4-w1-new"], 25.0),
    ]
    out = capsys.readouterr().out
    assert "Summary: stopped=2 (default=1, isolated=1)" in out
    assert "skipped_stale=1" in out
    assert "skipped_duplicate=2" in out
    assert "unmatched_orphans=1" in out
    assert "m-s-orphan9" in out


def test_reap_sessions_retries_rc124_and_falls_back_when_sessions_fails(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "runs/forge/colab/catchup-20261123-030000/forge-a100-w1.colab.json"
    _write_json(cfg, {"forge-a100-w1": {"name": "forge-a100-w1", "endpoint": "m-s-iso2"}})

    attempts: dict[tuple[str, ...], int] = {}
    sleeps: list[float] = []

    def fake_run(cmd: list[str], *, timeout_seconds: float, **_kwargs) -> tuple[int, str]:
        key = tuple(cmd)
        attempts[key] = attempts.get(key, 0) + 1
        if cmd == ["colab", "sessions"]:
            # Fail both attempts with rc=124 -> reap must still fall back to isolated configs
            return 124, "[timeout] Hard-killing hung command\n"
        if cmd == ["colab", "--config", str(cfg), "stop", "-s", "forge-a100-w1"]:
            # First stop hits transient rc=124, second succeeds
            if attempts[key] == 1:
                return 124, "[timeout] Hard-killing hung command\n"
            return 0, "[colab] Session terminated.\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(cf.time, "sleep", lambda s: sleeps.append(s))

    summary = cf.reap_sessions(tmp_path, retries=1, backoff_seconds=1.5)
    assert summary["sessions_rc"] == 124
    assert len(summary["stopped_isolated"]) == 1
    assert summary["stopped_isolated"][0]["session"] == "forge-a100-w1"
    assert summary["failed"] == []
    assert attempts[("colab", "sessions")] == 2
    assert attempts[("colab", "--config", str(cfg), "stop", "-s", "forge-a100-w1")] == 2
    assert sleeps == [1.5, 1.5]


def test_reap_sessions_skips_stale_configs_when_server_reports_no_sessions(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "runs/forge/colab/catchup-20261123-040000/forge-l4-w1.colab.json"
    _write_json(cfg, {"forge-l4-w1": {"name": "forge-l4-w1", "endpoint": "m-s-already-gone"}})

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs) -> tuple[int, str]:
        calls.append(list(cmd))
        if cmd == ["colab", "sessions"]:
            return 0, "[colab] No active sessions found on server.\n"
        raise AssertionError(f"Should not call stop on stale config: {cmd}")

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)
    summary = cf.reap_sessions(tmp_path, retries=1, backoff_seconds=0.0)
    assert calls == [["colab", "sessions"]]
    assert summary["stopped_default"] == []
    assert summary["stopped_isolated"] == []
    assert summary["skipped_stale"] == 1
    assert summary["configs_scanned"] == 1


def _make_test_worker(tmp_path: Path, *, branch_sync: dict | None = None) -> cf.ColabWorker:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return cf.ColabWorker(
        worker_name="forge-l4-w1-test",
        gpu="L4",
        run_dir=run_dir,
        repo_root=tmp_path,
        repo_url=cf.DEFAULT_REPO_URL,
        branch_sync=branch_sync or {"sha": "a" * 40, "pushed": True},
        transport_retries=1,
        transport_retry_backoff=0.5,
    )


def _make_test_unit() -> cf.CatchupUnit:
    return cf.CatchupUnit(
        unit_id="unit-retry-test",
        candidate_id="unified-v12",
        view_id="discriminator_stability",
        through_tier=1,
        gpu="L4",
        tasks=["ring16_acquisition"],
        is_full_view_slice=False,
        estimated_seconds=300.0,
    )


def test_run_unit_retries_transient_rc124_upload_and_download_but_not_forge_run(tmp_path: Path, monkeypatch):
    worker = _make_test_worker(tmp_path)
    unit = _make_test_unit()
    study_file = tmp_path / "study.json"
    view_file = tmp_path / "ephemeral.view.json"
    study_file.write_text("{}", encoding="utf-8")
    view_file.write_text("{}", encoding="utf-8")

    counts: dict[str, int] = {"study_upload": 0, "view_upload": 0, "ssh_run": 0, "download": 0}
    sleeps: list[float] = []

    def fake_run(cmd: list[str], **_kwargs) -> tuple[int, str]:
        if "upload" in cmd and str(study_file) in cmd:
            counts["study_upload"] += 1
            return (124, "") if counts["study_upload"] == 1 else (0, "")
        if "upload" in cmd and str(view_file) in cmd:
            counts["view_upload"] += 1
            return (124, "") if counts["view_upload"] == 1 else (0, "")
        if "ssh" in cmd:
            counts["ssh_run"] += 1
            # Non-zero scientific/Forge exit code must NEVER be retried
            return 1, ""
        if "download" in cmd:
            counts["download"] += 1
            if counts["download"] == 1:
                return 124, ""
            Path(cmd[-1]).write_bytes(b"tar-bytes")
            return 0, ""
        raise AssertionError(f"Unexpected cmd: {cmd}")

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(cf.time, "sleep", lambda s: sleeps.append(s))

    res = worker.run_unit(unit, study_file=study_file, study_id="study-1", ephemeral_view_file=view_file)
    assert counts == {"study_upload": 2, "view_upload": 2, "ssh_run": 1, "download": 2}
    assert sleeps == [0.5, 0.5, 0.5]
    assert res["run_rc"] == 1
    assert res["download_rc"] == 0
    assert res["tarball"] is not None


def test_run_unit_does_not_retry_non_transport_upload_error_or_ssh_timeout(tmp_path: Path, monkeypatch):
    worker = _make_test_worker(tmp_path)
    unit = _make_test_unit()
    study_file = tmp_path / "study.json"
    study_file.write_text("{}", encoding="utf-8")

    # 1. Non-transport rc=1 on study upload must fail immediately without retry
    upload_calls = 0

    def fake_upload_rc1(cmd: list[str], **_kwargs) -> tuple[int, str]:
        nonlocal upload_calls
        upload_calls += 1
        return 1, "error"

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_upload_rc1)
    with pytest.raises(RuntimeError, match=r"Study upload failed for unit-retry-test \(rc=1\)"):
        worker.run_unit(unit, study_file=study_file, study_id="study-1")
    assert upload_calls == 1

    # 2. Repeated rc=124 on study upload stops after 1 bounded retry (2 total attempts)
    upload_timeout_calls = 0
    monkeypatch.setattr(cf.time, "sleep", lambda _s: None)

    def fake_upload_rc124(cmd: list[str], **_kwargs) -> tuple[int, str]:
        nonlocal upload_timeout_calls
        upload_timeout_calls += 1
        return 124, "timeout"

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_upload_rc124)
    with pytest.raises(RuntimeError, match=r"Study upload failed for unit-retry-test \(rc=124\)"):
        worker.run_unit(unit, study_file=study_file, study_id="study-1")
    assert upload_timeout_calls == 2

    # 3. rc=124 on ssh Forge execution must NEVER be retried (wall-time exhaustion)
    ssh_calls = 0
    dl_calls = 0

    def fake_ssh_rc124(cmd: list[str], **_kwargs) -> tuple[int, str]:
        nonlocal ssh_calls, dl_calls
        if "upload" in cmd:
            return 0, ""
        if "ssh" in cmd:
            ssh_calls += 1
            return 124, ""
        if "download" in cmd:
            dl_calls += 1
            return 1, ""  # Non-transport download failure must also NOT be retried
        raise AssertionError(f"Unexpected cmd: {cmd}")

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_ssh_rc124)
    res = worker.run_unit(unit, study_file=study_file, study_id="study-1")
    assert ssh_calls == 1
    assert dl_calls == 1
    assert res["run_rc"] == 124
    assert res["download_rc"] == 1


def test_worker_stop_retries_transient_rc124_only(tmp_path: Path, monkeypatch):
    worker = _make_test_worker(tmp_path)
    worker.alive = True
    calls = 0

    def fake_stop(cmd: list[str], **_kwargs) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        return (124, "") if calls == 1 else (0, "[colab] Session terminated.\n")

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_stop)
    monkeypatch.setattr(cf.time, "sleep", lambda _s: None)
    worker.stop()
    assert calls == 2
    assert worker.alive is False


def test_reap_respects_max_runs_newest_first(tmp_path: Path, monkeypatch):
    colab_root = tmp_path / "runs/forge/colab"
    for idx in range(3):
        run_dir = colab_root / f"catchup-2026112{idx}-000000"
        cfg = run_dir / f"w{idx}.colab.json"
        _write_json(cfg, {f"w{idx}": {"name": f"w{idx}", "endpoint": f"m-s-{idx}"}})
        ts = 1_700_000_000 + idx * 1000
        os.utime(cfg, (ts, ts))
        os.utime(run_dir, (ts, ts))

    entries, scanned = cf._discover_isolated_config_sessions(tmp_path, max_runs=2)
    assert scanned == 2
    assert [e["name"] for e in entries] == ["w2", "w1"]


def test_start_retries_branch_bundle_upload_on_rc124(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "branch.bundle"
    bundle.write_bytes(b"bundle")
    worker = _make_test_worker(
        tmp_path,
        branch_sync={"sha": "b" * 40, "pushed": False, "bundle_path": str(bundle), "overlay_path": None, "dirty_file_count": 0},
    )
    upload_attempts = 0

    def fake_run(cmd: list[str], **_kwargs) -> tuple[int, str]:
        nonlocal upload_attempts
        if "upload" in cmd and str(bundle) in cmd:
            upload_attempts += 1
            return (124, "") if upload_attempts == 1 else (0, "")
        return 0, ""

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(cf.time, "sleep", lambda _s: None)
    try:
        worker.start()
        assert upload_attempts == 2
    finally:
        cf._GLOBAL_REGISTRY.unregister(worker.worker_name)
