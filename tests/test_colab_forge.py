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
