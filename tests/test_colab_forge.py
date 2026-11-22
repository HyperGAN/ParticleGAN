"""Offline unit tests for ops/colab_forge.py (decentralized branch catch-up, right-sizing, safety)."""
from __future__ import annotations

import json
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
