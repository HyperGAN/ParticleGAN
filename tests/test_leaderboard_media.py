"""Offline unit tests for leaderboard actual-training GIF indexing & Colab media ingest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from experiments.forge import leaderboard_media as lm
from ops import colab_forge as cf

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_leaderboard_media_index_schema_and_provenance():
    doc = lm.load_leaderboard_media_index(REPO_ROOT)
    assert doc["schema"] == lm.MEDIA_INDEX_SCHEMA
    assert doc["qualification_input"] is False
    items = doc["items"]
    assert len(items) >= 295
    fams = {it["family_id"] for it in items}
    assert {"unified", "axiom", "rhbi", "empf", "bispf", "bcap-pure", "atlas"}.issubset(fams)
    by_fam_task = {(it["family_id"], it["task_id"]) for it in items}
    for top_fam in ("unified", "axiom", "rhbi", "empf", "bispf", "bcap-pure"):
        for flagship in ("grid100", "rotated100", "staggered100", "ring16_acquisition"):
            assert (top_fam, flagship) in by_fam_task, f"missing flagship GIF for {top_fam}:{flagship}"
    for it in items:
        assert it["kind"] in lm.VALID_GIF_KINDS
        assert it["optimizer_updates_added"] == 0
        assert it["sampling_draws_added"] == 0
        assert it["qualification_input"] is False
        assert it["gif_relpath"].startswith("media/")
        assert len(it["gif_sha256"]) == 64


def test_index_media_lookup_prefers_selected_cohort_over_replay(tmp_path: Path):
    idx_path = tmp_path / lm.DEFAULT_MEDIA_INDEX_RELPATH
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schema": lm.MEDIA_INDEX_SCHEMA,
                "qualification_input": False,
                "hf_repo": lm.DEFAULT_HF_REPO,
                "media_prefix": lm.DEFAULT_MEDIA_PREFIX,
                "items": [
                    {
                        "family_id": "unified",
                        "candidate_id": "unified-v11",
                        "task_id": "ring16_acquisition",
                        "attempt_id": "replay01",
                        "cohort_match": "colab_replay_media",
                        "grade_matches_selected": True,
                        "has_visual_samples": True,
                        "gif_relpath": "media/forge/leaderboard/unified/unified-v11/ring16_replay.gif",
                        "gif_sha256": "a" * 64,
                    },
                    {
                        "family_id": "unified",
                        "candidate_id": "unified-v11",
                        "task_id": "ring16_acquisition",
                        "attempt_id": "sel01",
                        "cohort_match": "selected_cohort",
                        "grade_matches_selected": True,
                        "has_visual_samples": True,
                        "gif_relpath": "media/forge/leaderboard/unified/unified-v11/ring16_sel.gif",
                        "gif_sha256": "b" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    by_fam, by_att = lm.index_media_lookup(tmp_path)
    assert by_fam[("unified", "ring16_acquisition")]["gif_relpath"].endswith("ring16_sel.gif")
    assert by_att[("replay01", "ring16_acquisition")]["gif_relpath"].endswith("ring16_replay.gif")


def test_colab_ingest_tarball_extracts_rendered_media_and_blocks_traversal(tmp_path: Path):
    tar_ok = tmp_path / "ok.tar.gz"
    gif_bytes = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"
    sidecar = {
        "kind": "actual_training_saved_observations_gif",
        "task_id": "ring16_acquisition",
        "gif_sha256": hashlib.sha256(gif_bytes).hexdigest(),
    }
    src_dir = tmp_path / "src" / "runs/forge/rendered_media/att123"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "ring16_acquisition.gif").write_bytes(gif_bytes)
    (src_dir / "ring16_acquisition.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with tarfile.open(tar_ok, "w:gz") as tf:
        tf.add(src_dir / "ring16_acquisition.gif", arcname="runs/forge/rendered_media/att123/ring16_acquisition.gif")
        tf.add(src_dir / "ring16_acquisition.json", arcname="runs/forge/rendered_media/att123/ring16_acquisition.json")

    out_media = tmp_path / "extracted_media"
    ing = cf.ingest_tarball(
        tmp_path,
        tar_ok,
        project_receipts=False,
        register_submissions=False,
        rendered_media_dir=out_media,
    )
    assert len(ing["rendered_media_gifs"]) == 1
    assert (out_media / "att123/ring16_acquisition.gif").read_bytes() == gif_bytes

    tar_bad = tmp_path / "bad.tar.gz"
    with tarfile.open(tar_bad, "w:gz") as tf:
        tf.add(src_dir / "ring16_acquisition.gif", arcname="runs/forge/rendered_media/../../escape.gif")
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        cf.ingest_tarball(
            tmp_path,
            tar_bad,
            project_receipts=False,
            register_submissions=False,
            rendered_media_dir=out_media,
        )
