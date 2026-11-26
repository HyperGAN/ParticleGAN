"""Offline tests for ops/refresh_leaderboard.py."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from ops import refresh_leaderboard as rl

ROOT = Path(__file__).resolve().parents[1]


def test_status_collects_real_repo_health():
    status = rl.collect_status(ROOT, check_pr=False)
    assert status["git"]["head"]
    assert len(status["selected_families"]) >= 10
    assert "unified" in status["selected_families"]
    assert "axiom" in status["selected_families"]
    hf = status["hf_publication_receipt"]
    assert hf is not None
    assert hf["families"] >= 10


def test_sync_step_plan_local_vs_publish(tmp_path):
    local_steps = rl.build_sync_steps(ROOT, publish=False)
    names = [s["name"] for s in local_steps]
    assert names == [
        "forge_compile",
        "refresh_technique_inventory",
        "experiment_docs",
        "validate_github_pages_portal",
        "check_repo_bloat",
        "check_lint",
        "publish_hf_dataset",
        "deploy_hf_space",
    ]
    pub_step = next(s for s in local_steps if s["name"] == "publish_hf_dataset")
    space_step = next(s for s in local_steps if s["name"] == "deploy_hf_space")
    assert "--dry-run" in pub_step["cmd"]
    assert "--dry-run" in space_step["cmd"]

    media_dir = tmp_path / "staging"
    (media_dir / "media/forge/leaderboard/axiom/axiom-v5").mkdir(parents=True)
    (media_dir / "media/forge/leaderboard/axiom/axiom-v5/grid100.gif").write_bytes(b"GIF89a")

    publish_steps = rl.build_sync_steps(
        ROOT,
        publish=True,
        check_mkdocs=True,
        upload_media=True,
        staging_dir=media_dir,
        commit_message="Refresh test",
    )
    pnames = [s["name"] for s in publish_steps]
    assert "upload_hf_media" in pnames
    ppub = next(s for s in publish_steps if s["name"] == "publish_hf_dataset")
    pspace = next(s for s in publish_steps if s["name"] == "deploy_hf_space")
    pportal = next(s for s in publish_steps if s["name"] == "validate_github_pages_portal")
    assert "--dry-run" not in ppub["cmd"]
    assert "--receipt" in ppub["cmd"]
    assert "--commit-message" in ppub["cmd"]
    assert "--dry-run" not in pspace["cmd"]
    assert "--strict" in pportal["cmd"]


def test_run_sync_dry_run_executes_nothing():
    called = []

    def fake_runner(*args, **kwargs):
        called.append((args, kwargs))

    report = rl.run_sync(ROOT, publish=True, dry_run=True, runner_fn=fake_runner)
    assert report["mode"] == "plan_only"
    assert called == []
    assert len(report["steps"]) >= 8


def test_energy_refuses_busy_gpu_and_enriches_clean_run(tmp_path):
    busy_gpus = [{"index": 0, "name": "NVIDIA RTX A6000", "util_pct": 62, "mem_used_mib": 4000, "mem_total_mib": 49140}]
    with pytest.raises(RuntimeError, match="Refusing uncontended energy benchmark"):
        rl.run_energy_benchmark(
            ROOT,
            ideas=["axiom-v5"],
            output=tmp_path / "busy.json",
            gpu_query_fn=lambda: busy_gpus,
        )

    idle_gpus = [{"index": 0, "name": "NVIDIA RTX A6000", "util_pct": 0, "mem_used_mib": 100, "mem_total_mib": 49140}]
    out_file = tmp_path / "curated-energy.json"

    def fake_runner(cmd, cwd=None, check=True):
        raw_idx = cmd.index("--output") + 1
        raw_path = Path(cmd[raw_idx])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = {
            "device": "cuda:0",
            "mode": "energy-benchmark",
            "pytorch_version": "2.15.0+cu132",
            "energy_benchmark": {
                "axiom-v5": {
                    "dcgan_r256": {
                        "gpu_name": "NVIDIA RTX A6000",
                        "gpu_util_before": [0, 0],
                        "sustained_uninstrumented": {
                            "joules_per_step": 8.25,
                            "step_ms": {"median": 32.8},
                        },
                    },
                    "grid100": {
                        "joules_to_first_pass": 3511.6,
                        "first_pass_step": 1000,
                    },
                }
            },
        }
        raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

    result = rl.run_energy_benchmark(
        ROOT,
        ideas=["axiom-v5"],
        output=out_file,
        gpu_query_fn=lambda: idle_gpus,
        runner_fn=fake_runner,
    )
    assert result["dry_run"] is False
    assert result["contended"] is False
    curated = json.loads(out_file.read_text(encoding="utf-8"))
    entry = curated["energy_benchmark"]["axiom-v5"]
    assert entry["family_id"] == "axiom"
    assert entry["contended"] is False
    assert entry["occam_view"] == "r4-occam-ablation-v1"
    assert entry["occam_executed"] == 16
    assert entry["occam_pass"] == 15
    assert len(entry["occam_receipt_ids"]) == 16


def test_promote_rejects_dirty_source_receipt_manifest():
    # Commit 73250fe3 recorded receipts on a dirty tree; preflight must refuse cleanly.
    with pytest.raises(ValueError, match="do not match the clean Git tree"):
        rl.verify_source_commit_receipts(
            ROOT,
            "73250fe3bd6d3c3f6d03215535d141290661f053",
            ["unified-v15"],
        )


def test_verify_live_surfaces_with_mocked_endpoints():
    reg_doc = json.loads((ROOT / "configs/forge/trainer-families.json").read_text(encoding="utf-8"))
    rep_map = {f["id"]: f.get("reporting_family", f["id"]) for f in reg_doc["families"]}
    sel_doc = json.loads((ROOT / "configs/forge/selections/family-current-v1.json").read_text(encoding="utf-8"))
    families = [
        {"id": rep_map.get(entry["trainer_family"], entry["trainer_family"]), "candidate_id": entry["candidate_id"]}
        for entry in sel_doc["selections"]
    ]
    sample_lb = json.dumps(
        {
            "generated_at": "2026-11-26T08:00:00Z",
            "families": families,
            "energy_benchmarks": [
                {"candidate_id": "axiom-v5", "evidence_scope": "study_energy_benchmark"},
                {"candidate_id": "unified-v15", "evidence_scope": "study_energy_benchmark"},
            ],
        }
    ).encode("utf-8")
    local_app_bytes = (ROOT / "ops/hf_space/app.js").read_bytes()

    def fake_http_get(url: str) -> tuple[int, bytes]:
        if url.endswith("data/leaderboard.json"):
            return 200, sample_lb
        if url.endswith("/raw/main/app.js"):
            return 200, local_app_bytes
        return 200, b"<html>ok</html>"

    def fake_gh(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "number": 631,
                        "url": "https://github.com/HyperGAN/ParticleGAN/pull/631",
                        "title": "Release develop to master",
                        "body": f"See {rl.INVENTORY_URL}",
                    }
                ]
            ),
            stderr="",
        )

    res = rl.verify_live_surfaces(ROOT, http_get=fake_http_get, gh_runner=fake_gh)
    assert res["ok"] is True
    assert res["checks"]["hf_dataset"]["ok"] is True
    assert res["checks"]["hf_space"]["app_js_matches_local"] is True
    assert res["checks"]["github_pages"]["ok"] is True
    assert res["checks"]["develop_to_master_pr"]["ok"] is True

    # Simulate stale HF Space app.js -> overall ok must flip False.
    def stale_space_get(url: str) -> tuple[int, bytes]:
        if url.endswith("/raw/main/app.js"):
            return 200, b"// stale version"
        return fake_http_get(url)

    res_stale = rl.verify_live_surfaces(ROOT, http_get=stale_space_get, gh_runner=fake_gh)
    assert res_stale["ok"] is False
    assert res_stale["checks"]["hf_space"]["ok"] is False
