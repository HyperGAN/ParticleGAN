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
    DEFAULT_PAGES_BASE_URL,
    build_inventory_exports,
    export_leaderboard_json,
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
    assert payload["summary"]["full_suite_tests"] == 71
    assert payload["summary"]["counted_view_requirements"] == 115
    assert len(exports["receipt_rows"]) > 100
    assert len(exports["attempt_rows"]) > 100
    assert len(exports["energy_rows"]) >= 10
    assert payload["summary"]["energy_benchmark_rows"] == len(exports["energy_rows"])
    assert len(payload["energy_benchmarks"]) == len(exports["energy_rows"])

    # Verify committed law4-phase2-energy-benchmark.json entries are exported faithfully.
    v3 = next(row for row in exports["energy_rows"] if row["candidate_id"] == "law4-energy-v3")
    assert v3["optimizer_family"] == "law4_energy_v3"
    assert v3["family_id"] == "unified"
    assert v3["hardware_gpu"] == "NVIDIA RTX A6000"
    assert v3["contended"] is False
    assert v3["source_file"] == "reports/forge/studies/law4-phase2-energy-benchmark.json"
    assert v3["r256_joules_per_step"] == pytest.approx(13.12414, rel=1e-4)
    assert v3["ring16_first_pass_step"] == 217
    assert v3["ring16_j_to_pass"] == pytest.approx(619.340378, rel=1e-4)
    assert v3["grid100_first_pass_step"] == 1500
    assert v3["grid100_sustained_pass_step"] == 2500
    assert v3["grid100_j_to_pass"] == pytest.approx(7643.074714, rel=1e-4)
    assert v3["grid100_j_to_sustained_pass"] == pytest.approx(12738.457857, rel=1e-4)
    assert v3["occam_pass"] == 14 and v3["occam_executed"] == 16

    # Verify historical receipts without cost.energy stay strictly null (never fabricated),
    # while energy-instrumented receipts export valid positive hardware NVML telemetry.
    with_energy = [row for row in exports["receipt_rows"] if row["joules_total"] is not None]
    without_energy = [row for row in exports["receipt_rows"] if row["joules_total"] is None]
    assert without_energy
    assert all(
        row["joules_per_step"] is None
        and row["joules_to_first_pass"] is None
        and row["energy_backend"] is None
        for row in without_energy
    )
    assert all(
        row["joules_total"] >= 0.0 and isinstance(row["energy_backend"], str) and row["energy_backend"]
        for row in with_energy
    )
    # Also verify our Phase-4 unified-v12 energy benchmark row is exported faithfully.
    v12 = next(row for row in exports["energy_rows"] if row["candidate_id"] == "unified-v12")
    assert v12["optimizer_family"] == "unified_v12_equil"
    assert v12["family_id"] == "unified"
    assert v12["contended"] is False
    assert v12["r256_joules_per_step"] == pytest.approx(11.36602, rel=1e-4)
    assert v12["ring16_j_to_pass"] == pytest.approx(462.5188, rel=1e-4)
    assert v12["grid100_j_to_pass"] == pytest.approx(4512.9094, rel=1e-4)
    assert v12["occam_pass"] == 16 and v12["occam_executed"] == 16

    # Verify Unified row numbers match technique-inventory.md AND incompatible study commit is never mixed.
    unified = next(row for row in exports["leaderboard_rows"] if row["family_id"] == "unified")
    assert unified["candidate_id"] == "unified-v11"
    assert unified["tier1_pass"] == 27 and unified["tier1_total"] == 27
    assert unified["total_pass"] == 95 and unified["total_required"] == 115
    assert unified["full_suite_pass"] == 40 and unified["full_suite_total"] == 71
    assert unified["stretch_total_pass"] == 50 and unified["stretch_total"] == 70
    assert unified["stale"] is False
    assert unified["r256_joules_per_step"] is None
    assert unified["ring16_j_to_pass"] is None
    assert unified["grid100_j_to_pass"] is None

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
    assert "config_name: energy" in readme
    assert "path: data/energy.parquet" in readme
    assert "J-to-PASS" in readme
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
        "data/energy.parquet",
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
    for col in ("r256_joules_per_step", "ring16_j_to_pass", "grid100_j_to_pass"):
        assert col in lb_table.column_names

    rec_table = pa_pq.read_table(staging / "data/receipts.parquet")
    assert rec_table.num_rows == receipt["files"]["data/receipts.parquet"]["rows"]
    assert "attempt_id" in rec_table.column_names
    for col in (
        "joules_total",
        "joules_per_step",
        "avg_power_watts",
        "first_pass_step",
        "sustained_pass_step",
        "joules_to_first_pass",
        "joules_to_sustained_pass",
        "energy_backend",
    ):
        assert col in rec_table.column_names

    att_table = pa_pq.read_table(staging / "data/attempts.parquet")
    assert att_table.num_rows == receipt["files"]["data/attempts.parquet"]["rows"]
    assert "candidate_id" in att_table.column_names

    energy_table = pa_pq.read_table(staging / "data/energy.parquet")
    assert energy_table.num_rows == receipt["files"]["data/energy.parquet"]["rows"]
    assert energy_table.num_rows >= 10
    expected_energy_cols = {
        "candidate_id",
        "family_id",
        "optimizer_family",
        "hardware_gpu",
        "contended",
        "source_commit",
        "source_file",
        "r256_ms_per_step",
        "r256_watts",
        "r256_joules_per_step",
        "state_bytes_per_param",
        "ring16_first_pass_step",
        "ring16_j_to_pass",
        "grid100_first_pass_step",
        "grid100_sustained_pass_step",
        "grid100_j_to_pass",
        "grid100_j_to_sustained_pass",
        "staggered100_first_pass_step",
        "staggered100_j_to_pass",
        "occam_pass",
        "occam_executed",
        "evidence_scope",
    }
    assert set(energy_table.column_names) == expected_energy_cols

    lb_json = json.loads((staging / "data/leaderboard.json").read_text())
    assert len(lb_json["families"]) == lb_table.num_rows
    assert len(lb_json["tasks"]) > 0
    assert len(lb_json["task_definitions"]) > 0
    assert len(lb_json["energy_benchmarks"]) == energy_table.num_rows


def test_receipt_energy_exports_and_strict_cohort_matching(tmp_path):
    """Technique receipts carrying cost.energy populate receipts/energy/leaderboard only for matched cohorts."""
    pa_pq = pytest.importorskip("pyarrow.parquet")
    # Build minimal synthetic root reusing the real technique-inventory.json for schema realism.
    reports_dir = tmp_path / "reports/forge"
    (reports_dir / "technique-receipts").mkdir(parents=True)
    (reports_dir / "studies").mkdir(parents=True)
    inv_doc = json.loads((REPO_ROOT / "reports/forge/technique-inventory.json").read_text())
    (reports_dir / "technique-inventory.json").write_text(json.dumps(inv_doc))

    # Extract unified-v11's actual recorded source_origin_commit and source_digest.
    unified_row = next(r for r in inv_doc["rows"] if r.get("candidate_id") == "unified-v11")
    bindings = unified_row["bindings"]
    matched_commit = bindings["source_origin_commit"]
    matched_digest = bindings["source_digest"]

    # 1. Matched receipt carrying cost.energy for unified-v11 on ring16_acquisition & grid100.
    matched_receipt = {
        "attempt_id": "att-energy-matched",
        "request_id": "req-1",
        "campaign_id": "camp-1",
        "candidate_id": "unified-v11",
        "candidate_revision": "rev-1",
        "evidence_scope": "published_summary",
        "attempt_status": "completed",
        "provenance": {
            "canonical_result_hash": "a" * 64,
            "source_origin_commit": matched_commit,
            "source_digest": matched_digest,
        },
        "task_results": [
            {
                "task_id": "ring16_acquisition",
                "status": "PASS",
                "gate_status": "PASS",
                "raw_status": "completed",
                "reason": "ok",
                "cost": {
                    "wall_seconds": 40.0,
                    "completed_steps": 1600,
                    "device": "0",
                    "energy": {
                        "joules_total": 4000.0,
                        "joules_per_step": 2.5,
                        "avg_power_watts": 100.0,
                        "first_pass_step": 200,
                        "sustained_pass_step": 280,
                        "joules_to_first_pass": 500.0,
                        "joules_to_sustained_pass": 700.0,
                        "gpu_name": "NVIDIA RTX A6000",
                        "energy_backend": "nvml_energy_counter",
                        "contended": False,
                    },
                },
            }
        ],
    }
    (reports_dir / "technique-receipts/att-energy-matched.json").write_text(json.dumps(matched_receipt))

    # 2. Contended receipt for unified-v11 on grid100 (MUST NOT populate leaderboard summary!).
    contended_receipt = {
        "attempt_id": "att-energy-contended",
        "candidate_id": "unified-v11",
        "evidence_scope": "published_summary",
        "attempt_status": "completed",
        "provenance": {
            "canonical_result_hash": "b" * 64,
            "source_origin_commit": matched_commit,
            "source_digest": matched_digest,
        },
        "task_results": [
            {
                "task_id": "grid100",
                "status": "PASS",
                "gate_status": "PASS",
                "cost": {
                    "wall_seconds": 200.0,
                    "completed_steps": 7000,
                    "energy": {
                        "joules_total": 35000.0,
                        "joules_per_step": 5.0,
                        "avg_power_watts": 175.0,
                        "first_pass_step": 1500,
                        "joules_to_first_pass": 7500.0,
                        "gpu_name": "NVIDIA RTX A6000",
                        "energy_backend": "nvml_energy_counter",
                        "contended": True,
                    },
                },
            }
        ],
    }
    (reports_dir / "technique-receipts/att-energy-contended.json").write_text(json.dumps(contended_receipt))

    staging = tmp_path / "staged"
    publish_to_hf(tmp_path, inventory=True, dry_run=True, output_dir=staging)

    rec_table = pa_pq.read_table(staging / "data/receipts.parquet")
    rec_by_id = {row["attempt_id"]: row for row in rec_table.to_pylist()}
    assert rec_by_id["att-energy-matched"]["joules_to_first_pass"] == pytest.approx(500.0)
    assert rec_by_id["att-energy-matched"]["energy_backend"] == "nvml_energy_counter"

    energy_table = pa_pq.read_table(staging / "data/energy.parquet")
    assert energy_table.num_rows == 2

    lb_table = pa_pq.read_table(staging / "data/leaderboard.parquet")
    unified_lb = next(r for r in lb_table.to_pylist() if r["family_id"] == "unified")
    # Uncontended matched cohort populates ring16_j_to_pass, while contended grid100 stays None!
    assert unified_lb["ring16_j_to_pass"] == pytest.approx(500.0)
    assert unified_lb["grid100_j_to_pass"] is None


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


def test_export_leaderboard_json_enriches_committed_inventory():
    payload = export_leaderboard_json(REPO_ROOT, generated_at="2026-11-24T00:00:00Z")
    assert payload["schema_version"] == 1
    assert payload["schema"] == "particlegan_forge_leaderboard_v1"
    assert payload["pages_base_url"] == DEFAULT_PAGES_BASE_URL
    assert isinstance(payload["laws_meta"], dict)
    assert payload["laws_meta"]["epistemic_note"]
    assert isinstance(payload["laws"], list)

    unified = next(f for f in payload["families"] if f["id"] == "unified")
    assert unified["pages_url"] == f"{DEFAULT_PAGES_BASE_URL}/families/unified/"
    assert isinstance(unified["overview"], str) and "Unified" in unified["overview"]
    assert isinstance(unified["equations"], list) and len(unified["equations"]) >= 1
    assert isinstance(unified["law_mechanisms"], dict)
    assert isinstance(unified["tags"], list) and len(unified["tags"]) >= 1

    ring16 = next(t for t in payload["task_definitions"] if t["task_id"] == "ring16_acquisition")
    assert ring16["pages_url"] == f"{DEFAULT_PAGES_BASE_URL}/experiments/ring16_acquisition/"
    assert ring16["steps"] == 1600
    assert ring16["prior_summary"] and "mog" in ring16["prior_summary"]
    assert ring16["model_summary"] and "transfer_vector" in ring16["model_summary"]
    assert ring16["thresholds"]["modes"] == ">= 16"
    assert ring16["thresholds"]["hq"] == ">= 0.85"
    assert ring16["mermaid"] and "flowchart " in ring16["mermaid"]
    assert ring16["task_config_path"] == "configs/forge/tasks/ring16_acquisition.json"
    assert "probe" in ring16["reproduction_cmd"] and "ring16_acquisition" in ring16["reproduction_cmd"]
    fl_law1 = next(t for t in payload["task_definitions"] if t["task_id"] == "fl_law1_aniso_unequal_modes")
    assert "law1" in fl_law1["laws"]
    assert fl_law1["svg_inline"] and fl_law1["svg_inline"].startswith("<svg")
    assert fl_law1["thresholds"]


def test_export_leaderboard_json_reads_dynamic_laws_family_mechanisms_and_task_docs(tmp_path):
    """Verify dynamic laws.json, family-documentation law_mechanisms, task-documentation sidecars, and SVGs."""
    reports_dir = tmp_path / "reports/forge"
    reports_dir.mkdir(parents=True)
    inv_doc = json.loads((REPO_ROOT / "reports/forge/technique-inventory.json").read_text())
    (reports_dir / "technique-inventory.json").write_text(json.dumps(inv_doc))

    # Dynamic 3-law synthetic taxonomy (proves zero hardcoded law count or IDs).
    laws_cfg = {
        "schema_version": 1,
        "epistemic_status": "provisional_evolving",
        "epigraph": "Test evolving epigraph.",
        "epistemic_note": "Synthetic 3-law evolving registry note.",
        "doc_path": "docs/four-laws.md",
        "laws": [
            {
                "id": "law_a",
                "short_label": "Law A",
                "title": "Measure Geometry",
                "domain_level": "Geometry",
                "question": "Can the prior span disconnected support?",
                "doc_anchor": "law-a",
                "status": "active",
                "sub_laws": [],
            },
            {
                "id": "law_b",
                "short_label": "Law B",
                "title": "Transport Identifiability",
                "domain_level": "Transport",
                "question": "Is transport uniquely identifiable?",
                "doc_anchor": "law-b",
                "status": "active",
                "sub_laws": [
                    {"id": "law_b1", "short_label": "Law B1", "title": "Ray Screening", "doc_anchor": "law-b"}
                ],
            },
            {
                "id": "law_c",
                "short_label": "Law C",
                "title": "Coupled Dynamics",
                "domain_level": "Dynamics",
                "question": "Do updates contract?",
                "doc_anchor": "law-c",
                "status": "active",
                "sub_laws": [],
            },
        ],
    }
    (tmp_path / "configs/forge").mkdir(parents=True)
    (tmp_path / "configs/forge/laws.json").write_text(json.dumps(laws_cfg))

    fam_doc_dir = tmp_path / "configs/forge/family-documentation"
    fam_doc_dir.mkdir(parents=True)
    (fam_doc_dir / "unified.json").write_text(
        json.dumps(
            {
                "overview": "Synthetic Unified overview.",
                "law_mechanisms": {
                    "law_a": {
                        "status": "addressed",
                        "mechanism": "Laguerre-Pythagorean prototype chart",
                        "how_it_works": "Maintains adaptive latent cells.",
                        "known_limits": "Fixed particle budget.",
                    },
                    "law_b": {
                        "status": "partial",
                        "mechanism": "Local density-ratio orphan gate",
                        "how_it_works": "Rescues isolated target modes.",
                        "known_limits": "Struggles on staggered100.",
                    },
                },
                "equations": [
                    {"label": "Orphan ratio", "latex": "r(y) = d_k(y, X) / d_k(y, Y)", "explanation": "Local ratio."}
                ],
                "mermaid_diagram": 'flowchart LR\n  P["Prior"] --> G["Generator"]',
                "quickstart": {
                    "preset": "unified_v11",
                    "snippet": ["from particlegan import get_recipe", 'recipe = get_recipe("unified_v11")'],
                },
                "tags": ["synthetic-tag"],
            }
        )
    )

    task_doc_dir = tmp_path / "configs/forge/task-documentation"
    task_doc_dir.mkdir(parents=True)
    (task_doc_dir / "ring16_acquisition.json").write_text(
        json.dumps(
            {
                "title": "Ring-16 Mode Acquisition",
                "laws": ["law_a", "law_b1"],
                "edge_cases": ["disconnected-support", "16-mode-ring"],
                "intuition": "Tests whether all 16 modes on a ring are acquired cleanly.",
                "failure_mode": "Mode dropping or filament bridges between modes.",
                "protocol_notes": "1600 steps, deterministic orthogonal init.",
            }
        )
    )

    tasks_dir = tmp_path / "configs/forge/tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "ring16_acquisition.json").write_text(
        (REPO_ROOT / "configs/forge/tasks/ring16_acquisition.json").read_text()
    )

    svg_dir = tmp_path / "reports/forge/experiments/assets"
    svg_dir.mkdir(parents=True)
    sample_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><circle cx="60" cy="60" r="40"/></svg>'
    (svg_dir / "ring16_acquisition.svg").write_text(sample_svg)

    payload = export_leaderboard_json(tmp_path, generated_at="2026-11-24T00:00:00Z")
    assert payload["laws_meta"]["source"] == "configs/forge/laws.json"
    assert payload["laws_meta"]["epistemic_note"] == "Synthetic 3-law evolving registry note."
    assert [law["id"] for law in payload["laws"]] == ["law_a", "law_b", "law_c"]
    assert payload["laws"][1]["sub_laws"][0]["id"] == "law_b1"

    unified = next(f for f in payload["families"] if f["id"] == "unified")
    assert unified["law_mechanisms"]["law_a"]["status"] == "addressed"
    assert unified["law_mechanisms"]["law_b"]["known_limits"] == "Struggles on staggered100."
    assert unified["mermaid_diagram"] == 'flowchart LR\n  P["Prior"] --> G["Generator"]'
    assert unified["quickstart"]["preset"] == "unified_v11"
    assert "synthetic-tag" in unified["tags"]

    ring = next(t for t in payload["task_definitions"] if t["task_id"] == "ring16_acquisition")
    assert ring["title"] == "Ring-16 Mode Acquisition"
    assert ring["laws"] == ["law_a", "law_b1"]
    assert ring["edge_cases"] == ["disconnected-support", "16-mode-ring"]
    assert ring["intuition"].startswith("Tests whether all 16 modes")
    assert ring["failure_mode"] == "Mode dropping or filament bridges between modes."
    assert ring["svg_relpath"] == "reports/forge/experiments/assets/ring16_acquisition.svg"
    assert ring["svg_inline"] == sample_svg


def test_export_leaderboard_json_graceful_fallback_on_minimal_root(tmp_path):
    """Minimal synthetic unit-test root without laws.json, family docs, or task JSONs never crashes."""
    reports_dir = tmp_path / "reports/forge"
    reports_dir.mkdir(parents=True)
    inv_doc = json.loads((REPO_ROOT / "reports/forge/technique-inventory.json").read_text())
    (reports_dir / "technique-inventory.json").write_text(json.dumps(inv_doc))

    payload = export_leaderboard_json(tmp_path, generated_at="2026-11-24T00:00:00Z")
    assert payload["schema_version"] == 1
    assert payload["laws"] == []
    assert payload["laws_meta"]["source"] == "unavailable"
    assert payload["laws_meta"]["epistemic_note"]
    assert len(payload["families"]) >= 10
    assert all(isinstance(f["law_mechanisms"], dict) for f in payload["families"])
    assert all(f["pages_url"].startswith(DEFAULT_PAGES_BASE_URL) for f in payload["families"])
    assert all(isinstance(t["laws"], list) and isinstance(t["thresholds"], dict) for t in payload["task_definitions"])
