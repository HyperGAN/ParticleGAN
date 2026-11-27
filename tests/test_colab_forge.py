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


def test_render_media_enabled_by_default_with_opt_out():
    parser = cf.build_parser()
    catchup_default = parser.parse_args(["catchup"])
    assert catchup_default.render_media is True

    catchup_explicit = parser.parse_args(["catchup", "--render-media"])
    assert catchup_explicit.render_media is True

    catchup_disabled = parser.parse_args(["catchup", "--no-render-media"])
    assert catchup_disabled.render_media is False

    run_default = parser.parse_args(["run", "--job", "unified-v11:study-a"])
    assert run_default.render_media is True

    run_disabled = parser.parse_args(["run", "--job", "unified-v11:study-a", "--no-render-media"])
    assert run_disabled.render_media is False


def test_run_unit_renders_media_on_vm_by_default_and_skips_when_disabled(tmp_path: Path, monkeypatch):
    worker = _make_test_worker(tmp_path)
    unit = _make_test_unit()
    study_file = tmp_path / "study.json"
    study_file.write_text("{}", encoding="utf-8")

    ssh_scripts: list[str] = []

    def fake_run(cmd: list[str], **_kwargs) -> tuple[int, str]:
        if "ssh" in cmd:
            ssh_scripts.append(cmd[-1])
        return 0, ""

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)

    worker.run_unit(unit, study_file=study_file, study_id="study-1")
    worker.run_unit(unit, study_file=study_file, study_id="study-1", render_media=False)

    assert len(ssh_scripts) == 2
    assert "experiments.forge.leaderboard_media export-attempt" in ssh_scripts[0]
    assert 'TAR_ITEMS+=("runs/forge/rendered_media")' in ssh_scripts[0]
    assert "experiments.forge.leaderboard_media export-attempt" not in ssh_scripts[1]


def test_execute_work_units_ingests_and_uploads_media_before_hf_inventory_publish(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    unit = _make_test_unit()
    study_file = tmp_path / "study.json"
    study_file.write_text("{}", encoding="utf-8")
    unit.study_id = "study-1"
    unit.study_path = str(study_file)

    tar_path = run_dir / "unit-out.tar.gz"
    gif_src = tmp_path / "ring16.gif"
    gif_src.write_bytes(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")
    run_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(gif_src, arcname="runs/forge/rendered_media/att01/ring16_acquisition.gif")

    events: list[str] = []

    monkeypatch.setattr(cf, "prepare_branch_bundle", lambda _r, _d: {"sha": "a" * 40, "pushed": True})
    monkeypatch.setattr(cf.ColabWorker, "start", lambda self: None)
    monkeypatch.setattr(cf.ColabWorker, "stop", lambda self: None)

    def fake_run_unit(self, u, *, study_file, study_id, ephemeral_view_file=None, render_media=True):
        assert render_media is True
        return {
            "unit_id": u.unit_id,
            "candidate_id": u.candidate_id,
            "study_id": study_id,
            "gpu": self.gpu,
            "tasks": u.tasks,
            "run_rc": 0,
            "download_rc": 0,
            "tarball": str(tar_path),
            "elapsed_seconds": 1.0,
        }

    monkeypatch.setattr(cf.ColabWorker, "run_unit", fake_run_unit)

    from experiments.forge import hf_publish, leaderboard_media

    def fake_ingest(root, source_dirs, staging_dir, *, cohort_match="colab_replay_media", write_index=True):
        events.append(f"ingest:{write_index}:{len(source_dirs)}")
        return {
            "copied": 1,
            "skipped": 0,
            "total_items": 1,
            "staged_relpaths": ["media/forge/leaderboard/unified/unified-v12/ring16_acquisition.gif"],
        }

    def fake_upload(root, staging_dir, *, dry_run=False, only_relpaths=None, commit_message=None):
        events.append(f"upload:{dry_run}:{sorted(only_relpaths or [])}")
        return {"file_count": 1, "dry_run": dry_run}

    def fake_publish(root, *, inventory=False, dry_run=False):
        events.append(f"publish:{inventory}:{dry_run}")
        return {"hf_revision": "deadbeef" * 5}

    monkeypatch.setattr(leaderboard_media, "ingest_rendered_media_dirs", fake_ingest)
    monkeypatch.setattr(leaderboard_media, "upload_staged_media_to_hf", fake_upload)
    monkeypatch.setattr(hf_publish, "publish_to_hf", fake_publish)

    summary = cf.execute_work_units(
        tmp_path,
        [unit],
        workers=1,
        stagger_seconds=0.0,
        boot_timeout=30,
        max_minutes=30,
        idle_grace_minutes=5,
        stall_minutes=10,
        repo_url=cf.DEFAULT_REPO_URL,
        run_dir=run_dir,
        project_receipts=False,
        publish_hf=True,
        hf_dry_run=False,
    )
    assert summary["render_media"] is True
    assert summary["rendered_media_gif_count"] == 1
    assert events == [
        "ingest:True:1",
        "upload:False:['media/forge/leaderboard/unified/unified-v12/ring16_acquisition.gif']",
        "publish:True:False",
    ]


def test_compute_catchup_plan_skips_tasks_with_cached_failed_gate_and_groups_shared_prereq(tmp_path: Path, monkeypatch):
    for cid in ("fam-pass", "fam-fail"):
        _write_json(tmp_path / "configs/forge/ideas" / f"{cid}.json", {"id": cid})
    _write_json(
        tmp_path / "configs/forge/tasks/parent_gate.json",
        {"id": "parent_gate", "resources": {"timeout_seconds": 300, "gpus": 1}, "execution": {"steps": 500}},
    )
    for child in ("child_a", "child_b"):
        _write_json(
            tmp_path / "configs/forge/tasks" / f"{child}.json",
            {
                "id": child,
                "dependencies": [{"task": "parent_gate", "kind": "gate"}],
                "resources": {"timeout_seconds": 600, "gpus": 1},
                "execution": {"steps": 1000},
            },
        )
    _write_json(
        tmp_path / "configs/forge/tasks/indep_fast.json",
        {"id": "indep_fast", "resources": {"timeout_seconds": 120, "gpus": 1}, "execution": {"steps": 200}},
    )
    _write_json(
        tmp_path / "configs/forge/views/gate_view.json",
        {
            "id": "gate_view",
            "revision": 1,
            "assignments": [
                {"task": "parent_gate", "qualification_tier": 1, "importance": "required"},
                {"task": "child_a", "qualification_tier": 1, "importance": "required"},
                {"task": "child_b", "qualification_tier": 1, "importance": "required"},
                {"task": "indep_fast", "qualification_tier": 1, "importance": "required"},
            ],
        },
    )
    monkeypatch.setattr(cf, "_git_head", lambda _root: "deadbeef" * 5)
    monkeypatch.setattr("experiments.forge.sources.inspect_source", lambda _root: {"digest": "src1"})
    monkeypatch.setattr("experiments.forge.planning.load_idea", lambda r, cid: json.loads((r / "configs/forge/ideas" / f"{cid}.json").read_text()))
    monkeypatch.setattr("experiments.forge.views.load_view", lambda r, vid: json.loads((r / "configs/forge/views" / f"{vid}.json").read_text()))
    monkeypatch.setattr("experiments.forge.views.load_tasks", lambda r: {p.stem: json.loads(p.read_text()) for p in (r / "configs/forge/tasks").glob("*.json")})

    _write_receipt(tmp_path, "p01", "fam-pass", "parent_gate", status="PASS")
    _write_receipt(tmp_path, "f01", "fam-fail", "parent_gate", status="FAIL")

    plan = cf.compute_catchup_plan(
        tmp_path,
        view_id="gate_view",
        through_tier=1,
        candidates=["fam-pass", "fam-fail"],
        include_selected_families=False,
        gpu_policy="L4",
        changed_since=None,
    )
    # fam-fail's child_a & child_b are gate-blocked (never waste GPU re-running failed parent_gate)
    assert plan["blocked_count"] == 2
    assert sorted((m["candidate_id"], m["task_id"]) for m in plan["missing_cells"]) == [
        ("fam-fail", "indep_fast"),
        ("fam-pass", "child_a"),
        ("fam-pass", "child_b"),
        ("fam-pass", "indep_fast"),
    ]
    # Fast independent units are scheduled first; child_a & child_b grouped into 1 unit sharing parent_gate
    unit_tasks = [(u["candidate_id"], sorted(u["tasks"])) for u in plan["work_units"]]
    assert unit_tasks[:2] == [("fam-fail", ["indep_fast"]), ("fam-pass", ["indep_fast"])]
    assert unit_tasks[2] == ("fam-pass", ["child_a", "child_b"])


def test_compute_catchup_plan_blocks_over_vram_tasks_on_pinned_gpu_and_auto_routes_to_a100(tmp_path: Path, monkeypatch):
    _write_json(tmp_path / "configs/forge/ideas/fam-vram.json", {"id": "fam-vram"})
    _write_json(
        tmp_path / "configs/forge/tasks/small_task.json",
        {"id": "small_task", "resources": {"timeout_seconds": 300, "gpus": 1, "gpu_memory_mb": 2048}, "execution": {"steps": 500}},
    )
    _write_json(
        tmp_path / "configs/forge/tasks/big_vram_task.json",
        {"id": "big_vram_task", "resources": {"timeout_seconds": 300, "gpus": 1, "gpu_memory_mb": 24576}, "execution": {"steps": 13}},
    )
    _write_json(
        tmp_path / "configs/forge/views/vram_view.json",
        {
            "id": "vram_view",
            "revision": 1,
            "assignments": [
                {"task": "small_task", "qualification_tier": 1, "importance": "diagnostic"},
                {"task": "big_vram_task", "qualification_tier": 1, "importance": "diagnostic"},
            ],
        },
    )
    monkeypatch.setattr(cf, "_git_head", lambda _root: "deadbeef" * 5)
    monkeypatch.setattr("experiments.forge.sources.inspect_source", lambda _root: {"digest": "src1"})
    monkeypatch.setattr("experiments.forge.planning.load_idea", lambda r, cid: json.loads((r / "configs/forge/ideas" / f"{cid}.json").read_text()))
    monkeypatch.setattr("experiments.forge.views.load_view", lambda r, vid: json.loads((r / "configs/forge/views" / f"{vid}.json").read_text()))
    monkeypatch.setattr("experiments.forge.views.load_tasks", lambda r: {p.stem: json.loads(p.read_text()) for p in (r / "configs/forge/tasks").glob("*.json")})

    # Pinned L4 (23034 MiB): big_vram_task (24576 MiB) is blocked instead of poisoning small_task's unit.
    plan_l4 = cf.compute_catchup_plan(
        tmp_path,
        view_id="vram_view",
        through_tier=1,
        candidates=["fam-vram"],
        include_selected_families=False,
        gpu_policy="L4",
        changed_since=None,
    )
    assert plan_l4["blocked_count"] == 1
    assert plan_l4["blocked_sample"][0]["task_id"] == "big_vram_task"
    assert "24576 MiB" in plan_l4["blocked_sample"][0]["blockers"][0]
    assert [u["tasks"] for u in plan_l4["work_units"]] == [["small_task"]]

    # Auto GPU policy upgrades big_vram_task to A100 while keeping small_task on L4.
    plan_auto = cf.compute_catchup_plan(
        tmp_path,
        view_id="vram_view",
        through_tier=1,
        candidates=["fam-vram"],
        include_selected_families=False,
        gpu_policy="auto",
        changed_since=None,
    )
    assert plan_auto["blocked_count"] == 0
    gpu_by_task = {m["task_id"]: m["target_gpu"] for m in plan_auto["missing_cells"]}
    assert gpu_by_task == {"small_task": "L4", "big_vram_task": "A100"}


def test_ingest_tarball_marks_submission_concluded_with_numeric_timestamp_and_queue_ignores_it(tmp_path: Path):
    import tarfile
    from experiments.forge.queue import Queue

    aid = "a" * 32
    attempt_dir = tmp_path / "staging" / "reports/forge/attempts" / aid
    attempt_dir.mkdir(parents=True)
    req_doc = {
        "request_id": "req1234567890abcdef12345",
        "request": {
            "request_id": "req1234567890abcdef12345",
            "campaign_id": "colab-test-camp",
            "candidate": {"id": "unified-v15"},
            "jobs": [{"compatibility_key": "ck1", "task_id": "ring16_acquisition", "budget_seconds": 60}],
            "view": {"id": "v1", "goal": "g1", "assignments": [{"task": "ring16_acquisition", "qualification_tier": 1, "importance": "required"}]},
            "through_tier": 1,
        },
        "job": {"compatibility_key": "ck1", "task_id": "ring16_acquisition", "budget_seconds": 60},
    }
    res_doc = {
        "attempt_id": aid,
        "candidate_id": "unified-v15",
        "candidate_revision": "rev1",
        "task_results": [{"task_id": "ring16_acquisition", "gate_status": "PASS"}],
    }
    _write_json(attempt_dir / "request.json", req_doc)
    _write_json(attempt_dir / "result.json", res_doc)
    _write_json(attempt_dir / "evidence.json", {"attempt_id": aid})

    tar_path = tmp_path / "out.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(attempt_dir, arcname=f"reports/forge/attempts/{aid}")

    out = cf.ingest_tarball(tmp_path, tar_path, project_receipts=False)
    assert out["attempts_extracted"] == [aid]
    state = json.loads((tmp_path / "runs/forge/queue/state.json").read_text())
    sub = state["submissions"]["req1234567890abcdef12345"]
    assert sub["status"] == "concluded"
    assert isinstance(sub["submitted_at"], (int, float))

    # Queue.claim() must not crash or touch the imported Colab submission even if jobs dict is empty.
    q = Queue(tmp_path / "runs/forge")
    assert q.claim([{"device": "cpu", "slot": 0, "memory_mb": 4096}]) is None


def test_colab_worker_shell_quotes_injected_parameters(tmp_path: Path, monkeypatch):
    """All shell-interpolated params in bootstrap_sh and run_sh must be shlex.quote-escaped."""
    import shlex

    evil_repo = "https://example.com/repo.git; touch /tmp/pwn"
    evil_sha = "deadbeef$(id)"
    evil_cand = "cand'; rm -rf / #"
    evil_study = "study $(whoami)"
    evil_unit = "unit`uname`"

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = cf.ColabWorker(
        worker_name="forge-l4-w1-sec",
        gpu="L4",
        run_dir=run_dir,
        repo_root=tmp_path,
        repo_url=evil_repo,
        branch_sync={"sha": evil_sha, "pushed": True},
    )
    unit = cf.CatchupUnit(
        unit_id=evil_unit,
        candidate_id=evil_cand,
        view_id="discriminator_stability",
        through_tier=1,
        gpu="L4",
        tasks=["ring16_acquisition"],
        is_full_view_slice=False,
        estimated_seconds=60.0,
    )
    study_file = tmp_path / "study.json"
    study_file.write_text("{}", encoding="utf-8")

    ssh_scripts: list[str] = []

    def fake_run(cmd: list[str], **_kwargs) -> tuple[int, str]:
        if "ssh" in cmd and "bash" in cmd:
            ssh_scripts.append(cmd[-1])
        return 0, ""

    monkeypatch.setattr(cf, "run_with_hard_timeout", fake_run)
    try:
        worker.start()
        worker.run_unit(unit, study_file=study_file, study_id=evil_study, render_media=False)
    finally:
        cf._GLOBAL_REGISTRY.unregister(worker.worker_name)

    assert len(ssh_scripts) == 2
    bootstrap_sh, run_sh = ssh_scripts
    assert shlex.quote(evil_repo) in bootstrap_sh
    assert shlex.quote(evil_sha) in bootstrap_sh
    assert f'"{evil_sha}"' not in bootstrap_sh

    assert shlex.quote(evil_cand) in run_sh
    assert shlex.quote(evil_study) in run_sh
    assert shlex.quote(evil_unit) in run_sh
    assert shlex.quote(f"/content/out-{evil_unit}.tar.gz") in run_sh
    assert f"'{evil_cand}'" not in run_sh


def test_ingest_tarball_verifies_result_hash_and_fails_closed_on_conflicts(tmp_path: Path):
    """ingest_tarball verifies certificate result_hash and refuses conflicting overwrites fail-closed."""
    from experiments.forge.contracts import stable_hash

    aid = "b" * 32
    res_v1 = {
        "attempt_id": aid,
        "candidate_id": "axiom-v5",
        "candidate_revision": "rev1",
        "task_results": [{"task_id": "ring16_acquisition", "gate_status": "PASS"}],
    }
    cert_v1 = {"result_hash": stable_hash(res_v1), "source": "s1"}

    def _make_tar(name: str, files: dict[str, dict]) -> Path:
        stage = tmp_path / f"stage_{name}" / "reports/forge/attempts" / aid
        stage.mkdir(parents=True, exist_ok=True)
        for fname, payload in files.items():
            _write_json(stage / fname, payload)
        tpath = tmp_path / f"{name}.tar.gz"
        with tarfile.open(tpath, "w:gz") as tf:
            tf.add(stage, arcname=f"reports/forge/attempts/{aid}")
        return tpath

    # 1. Valid archive with matching certificate result_hash ingests cleanly and is idempotent.
    tar_good = _make_tar("good", {"result.json": res_v1, "certificate.json": cert_v1})
    out1 = cf.ingest_tarball(tmp_path, tar_good, project_receipts=False, register_submissions=False)
    assert out1["attempts_extracted"] == [aid]
    orig_bytes = (tmp_path / "reports/forge/attempts" / aid / "result.json").read_bytes()
    out2 = cf.ingest_tarball(tmp_path, tar_good, project_receipts=False, register_submissions=False)
    assert out2["attempts_extracted"] == [aid]

    # 2. Conflicting result.json/certificate.json for existing attempt fails closed WITHOUT overwriting disk.
    res_v2 = {**res_v1, "task_results": [{"task_id": "ring16_acquisition", "gate_status": "FAIL"}]}
    cert_v2 = {"result_hash": stable_hash(res_v2), "source": "s1"}
    tar_conflict = _make_tar("conflict", {"result.json": res_v2, "certificate.json": cert_v2})
    with pytest.raises(ValueError, match="Conflicting (result_hash/)?digest"):
        cf.ingest_tarball(tmp_path, tar_conflict, project_receipts=False, register_submissions=False)
    assert (tmp_path / "reports/forge/attempts" / aid / "result.json").read_bytes() == orig_bytes

    # 2b. Conflicting result.json alone (matched against existing on-disk certificate/result) also fails closed.
    tar_res_only = _make_tar("conflict_res", {"result.json": res_v2})
    with pytest.raises(ValueError, match="mismatch|Conflicting"):
        cf.ingest_tarball(tmp_path, tar_res_only, project_receipts=False, register_submissions=False)
    assert (tmp_path / "reports/forge/attempts" / aid / "result.json").read_bytes() == orig_bytes

    # 3. Tampered certificate.json result_hash fails closed on a brand-new attempt.
    aid_bad = "c" * 32
    stage_bad = tmp_path / "stage_bad" / "reports/forge/attempts" / aid_bad
    stage_bad.mkdir(parents=True, exist_ok=True)
    _write_json(stage_bad / "result.json", {**res_v1, "attempt_id": aid_bad})
    _write_json(stage_bad / "certificate.json", {"result_hash": "0" * 64, "source": "s1"})
    tar_bad_cert = tmp_path / "bad_cert.tar.gz"
    with tarfile.open(tar_bad_cert, "w:gz") as tf:
        tf.add(stage_bad, arcname=f"reports/forge/attempts/{aid_bad}")
    with pytest.raises(ValueError, match="certificate result_hash mismatch"):
        cf.ingest_tarball(tmp_path, tar_bad_cert, project_receipts=False, register_submissions=False)
    assert not (tmp_path / "reports/forge/attempts" / aid_bad / "result.json").exists()
