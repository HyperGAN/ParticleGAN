"""Offline tests for Forge Hugging Face Parquet/JSON publisher and receipt generation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from experiments.forge.__main__ import main
from experiments.forge.hf_publish import (
    DEFAULT_HF_REPO,
    build_inventory_exports,
    publish_to_hf,
    render_dataset_card,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_inventory_exports_matches_committed_inventory():
    exports = build_inventory_exports(REPO_ROOT, generated_at="2026-11-21T00:00:00Z")
    payload = exports["leaderboard_json"]
    assert payload["schema"] == "particlegan_forge_leaderboard_v1"
    assert payload["dataset_repo"] == DEFAULT_HF_REPO
    assert payload["summary"]["selected_families"] == len(exports["leaderboard_rows"])
    assert payload["summary"]["selected_families"] >= 10
    assert payload["summary"]["full_suite_tests"] == 69
    assert payload["summary"]["counted_view_requirements"] == 115
    assert len(exports["receipt_rows"]) > 100
    assert len(exports["attempt_rows"]) > 100

    # Verify Unified row numbers match technique-inventory.md
    unified = next(row for row in exports["leaderboard_rows"] if row["family_id"] == "unified")
    assert unified["candidate_id"] == "unified-v11"
    assert unified["tier1_pass"] == 27 and unified["tier1_total"] == 27
    assert unified["total_pass"] == 95 and unified["total_required"] == 115
    assert unified["full_suite_pass"] == 40 and unified["full_suite_total"] == 69
    assert unified["stretch_total_pass"] == 50 and unified["stretch_total"] == 68
    assert unified["stale"] is False

    bcap = next(row for row in exports["leaderboard_rows"] if row["family_id"] == "bcap-pure")
    assert bcap["stale"] is True
    assert bcap["total_pass"] == 71 and bcap["total_required"] == 115


def test_dataset_card_has_valid_yaml_configs():
    exports = build_inventory_exports(REPO_ROOT, generated_at="2026-11-21T00:00:00Z")
    readme = render_dataset_card(exports)
    assert readme.startswith("---\n")
    assert "config_name: leaderboard" in readme
    assert "path: data/leaderboard.parquet" in readme
    assert "config_name: receipts" in readme
    assert "path: data/receipts.parquet" in readme
    assert "config_name: attempts" in readme
    assert "path: data/attempts.parquet" in readme
    assert "Scientific superiority or public-default adoption" in readme


def test_publish_dry_run_writes_parquet_json_and_readme_without_network(tmp_path, monkeypatch, capsys):
    pa_pq = pytest.importorskip("pyarrow.parquet")
    import huggingface_hub

    def forbid_commit(*args, **kwargs):
        raise AssertionError("dry-run must never contact Hugging Face")

    monkeypatch.setattr(huggingface_hub.HfApi, "create_commit", forbid_commit)
    staging = tmp_path / "staged"
    receipt_out = tmp_path / "receipt.json"
    rc = main([
        "--root", str(REPO_ROOT),
        "publish", "--inventory", "--dry-run",
        "--output-dir", str(staging),
        "--receipt", str(receipt_out),
    ])
    assert rc == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["dry_run"] is True
    assert receipt["hf_revision"] is None
    assert receipt["qualification_input"] is False
    assert receipt["training_launched"] is False
    assert set(receipt["files"]) == {
        "README.md",
        "data/attempts.parquet",
        "data/leaderboard.json",
        "data/leaderboard.parquet",
        "data/receipts.parquet",
    }
    assert receipt_out.is_file()
    assert json.loads(receipt_out.read_text())["files"] == receipt["files"]

    # Read back Parquet tables to confirm valid schemas & row counts.
    lb_table = pa_pq.read_table(staging / "data/leaderboard.parquet")
    assert lb_table.num_rows == receipt["files"]["data/leaderboard.parquet"]["rows"]
    assert "family_id" in lb_table.column_names
    assert "full_suite_pass" in lb_table.column_names

    rec_table = pa_pq.read_table(staging / "data/receipts.parquet")
    assert rec_table.num_rows == receipt["files"]["data/receipts.parquet"]["rows"]
    assert "attempt_id" in rec_table.column_names

    att_table = pa_pq.read_table(staging / "data/attempts.parquet")
    assert att_table.num_rows == receipt["files"]["data/attempts.parquet"]["rows"]
    assert "candidate_id" in att_table.column_names

    lb_json = json.loads((staging / "data/leaderboard.json").read_text())
    assert len(lb_json["families"]) == lb_table.num_rows
    assert len(lb_json["tasks"]) > 0
    assert len(lb_json["task_definitions"]) > 0


def test_publish_mocked_commit_with_archive_card_records_hf_revision(tmp_path, monkeypatch):
    import huggingface_hub

    payload = b'{"gate_status":"PASS"}\n'
    tar_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as bundle:
        info = tarfile.TarInfo("reports/forge/attempts/a1/result.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    tar_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    card = {
        "archive": str(tar_path),
        "archive_sha256": tar_sha,
        "archive_bytes": tar_path.stat().st_size,
        "files": {"reports/forge/attempts/a1/result.json": hashlib.sha256(payload).hexdigest()},
    }
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(card))

    recorded_ops = []

    class FakeCommitInfo:
        oid = "f" * 40
        commit_url = f"https://huggingface.co/datasets/{DEFAULT_HF_REPO}/commit/{'f' * 40}"

    def fake_create_commit(self, *, repo_id, repo_type, operations, commit_message):
        recorded_ops.append((repo_id, repo_type, [op.path_in_repo for op in operations], commit_message))
        return FakeCommitInfo()

    monkeypatch.setattr(huggingface_hub.HfApi, "create_commit", fake_create_commit)
    receipt_out = tmp_path / "pub-receipt.json"
    receipt = publish_to_hf(
        tmp_path,
        archive_cards=[card_path],
        dry_run=False,
        receipt_path=receipt_out,
    )
    assert receipt["hf_revision"] == "f" * 40
    assert receipt["dry_run"] is False
    expected_hf_path = f"archives/sha256/{tar_sha}.tar.gz"
    assert recorded_ops == [
        (DEFAULT_HF_REPO, "dataset", [expected_hf_path], "Publish Forge archives (1 card(s))")
    ]
    assert receipt["files"][expected_hf_path]["sha256"] == tar_sha
    assert receipt["archive_cards"][0]["archive_sha256"] == tar_sha
    assert json.loads(receipt_out.read_text())["hf_revision"] == "f" * 40


def test_publish_requires_inventory_or_archive_card(tmp_path):
    with pytest.raises(ValueError, match="--inventory or --archive-card"):
        publish_to_hf(tmp_path, dry_run=True)
