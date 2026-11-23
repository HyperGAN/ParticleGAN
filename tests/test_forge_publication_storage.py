"""Storage reductions preserve scientific rows and certified media receipts."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from experiments.forge.contracts import stable_hash
from experiments.forge.publication_storage import (
    compact_current_publication,
    publication_json_text,
    publish_media_tree,
    RetainedMediaStore,
    rewrite_media_links,
)


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_compaction_preserves_referential_closure_without_mutation():
    publication = {
        "publication_scope": "current_technique_inventory",
        "rows": [{"status": "FAIL", "bindings": {"recipe_sha256": "recipe"}}],
        "evidence": {"key-reference": {"recorded_metric": 1.25}},
        "recipe_contracts": {
            "recipe": {"task": "task", "unchanged": [1, 2]},
            "key-reference": {"parameter": 7},
            "unused-recipe": {"task": "orphan-task"},
        },
        "task_contracts": {
            "task": {"protocol": "protocol"}, "orphan-task": {"orphan": True},
        },
        "protocol_contracts": {"protocol": {"cycle": "recipe"}, "orphan": {}},
    }
    before = deepcopy(publication)
    compact = compact_current_publication(publication)
    assert publication == before
    assert set(compact["recipe_contracts"]) == {"recipe", "key-reference"}
    assert set(compact["task_contracts"]) == {"task"}
    assert set(compact["protocol_contracts"]) == {"protocol"}
    for field in ("rows", "evidence"):
        assert compact[field] == publication[field]
    for field in ("recipe_contracts", "task_contracts", "protocol_contracts"):
        for identity, value in compact[field].items():
            assert value == publication[field][identity]
            assert stable_hash(value) == stable_hash(publication[field][identity])
    assert compact_current_publication(compact) == compact


def test_real_inventory_compaction_preserves_rows_and_current_bindings():
    original = json.loads((ROOT / "reports/forge/technique-inventory.json").read_text())
    compact = compact_current_publication(original)
    catalogs = {"recipe_contracts", "task_contracts", "protocol_contracts"}
    assert {k: v for k, v in compact.items() if k not in catalogs} == {
        k: v for k, v in original.items() if k not in catalogs}
    for name in catalogs:
        assert all(original[name][key] == value for key, value in compact[name].items())
    assert compact_current_publication(compact) == compact
    assert len(publication_json_text(compact).encode()) <= 25 * 1024 * 1024


def test_current_inventory_size_guard_preserves_archival_serialization():
    report = {"publication_scope": "current_technique_inventory", "rows": ["é" * 20]}
    encoded = publication_json_text(report).encode()
    assert publication_json_text(report, max_bytes=len(encoded))
    with pytest.raises(ValueError, match="publication budget"):
        publication_json_text(report, max_bytes=len(encoded) - 1)
    archived = {**report, "publication_scope": "frozen_source", "recipe_contracts": {"unreferenced": {}}}
    assert publication_json_text(archived, max_bytes=1) == (
        json.dumps(archived, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def test_regenerator_serializer_applies_only_current_storage_policy():
    spec = importlib.util.spec_from_file_location(
        "storage_regenerator", ROOT / "reports/forge/regenerate_technique_inventory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = {"publication_scope": "current_technique_inventory", "rows": [],
               "recipe_contracts": {"unused": {"setting": 1}}}
    assert json.loads(module._json_text(current))["recipe_contracts"] == {}
    archived = {**current, "publication_scope": "frozen_source"}
    assert json.loads(module._json_text(archived)) == archived


@pytest.fixture
def media_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _media_tree(path, gif):
    path.mkdir()
    (path / "actual.gif").write_bytes(gif)
    receipt = b'{ "original_recipe": "unchanged", "gate": "FAIL" }\n'
    (path / "actual.json").write_bytes(receipt)
    row = {"gif": "actual.gif", "gif_sha256": hashlib.sha256(gif).hexdigest(),
           "receipt": "actual.json", "receipt_sha256": hashlib.sha256(receipt).hexdigest(),
           "attempt_id": "original-attempt", "gate_status": "FAIL"}
    (path / "index.json").write_text(json.dumps({"media": [row]}))
    (path / "README.md").write_text("![Observed](actual.gif#frame) [Receipt](actual.json)\n")
    return receipt


def test_media_publishing_reuses_only_tracked_content_and_preserves_receipts(media_repository, tmp_path):
    root = media_repository
    tracked = root / "reports/original/observed.gif"
    tracked.parent.mkdir(parents=True)
    tracked.write_bytes(b"GIF89a-original")
    ignored = root / "reports/ignored/local.gif"
    ignored.parent.mkdir()
    ignored.write_bytes(b"GIF89a-new")
    (root / ".gitignore").write_text("reports/ignored/\n")
    subprocess.run(["git", "add", ".gitignore", "reports/original"], cwd=root, check=True)
    store = RetainedMediaStore(root)
    first = tmp_path / "first"
    receipt = _media_tree(first, tracked.read_bytes())
    first_output = root / "reports/new/first"
    mapping = publish_media_tree(first, first_output, store)
    assert mapping[(first / "actual.gif").resolve()] == tracked
    assert (first_output / "actual.json").read_bytes() == receipt
    assert not (first_output / "actual.gif").exists()
    index = json.loads((first_output / "index.json").read_text())["media"][0]
    assert (first_output / index["gif"]).resolve() == tracked
    assert "../../original/observed.gif#frame" in (first_output / "README.md").read_text()
    assert index["receipt_sha256"] == hashlib.sha256(receipt).hexdigest()

    # Identical ignored local bytes must not be selected as retained evidence.
    second = tmp_path / "second"
    _media_tree(second, ignored.read_bytes())
    second_output = root / "reports/new/second"
    publish_media_tree(second, second_output, store)
    second_index = json.loads((second_output / "index.json").read_text())["media"][0]
    canonical = (second_output / second_index["gif"]).resolve()
    assert canonical.parent == root / "reports/forge/media"
    assert canonical.name == hashlib.sha256(ignored.read_bytes()).hexdigest() + ".gif"
    assert canonical.read_bytes() == ignored.read_bytes()
    third = tmp_path / "third"
    _media_tree(third, ignored.read_bytes())
    publish_media_tree(third, root / "reports/new/third", store)
    assert list((root / "reports/forge/media").glob("*.gif")) == [canonical]
    assert tracked.read_bytes() == b"GIF89a-original"


def test_media_publication_refuses_invalid_receipts_and_existing_archives(media_repository, tmp_path):
    source = tmp_path / "source"
    _media_tree(source, b"GIF89a-fixture")
    destination = media_repository / "reports/new"
    store = RetainedMediaStore(media_repository)
    (source / "actual.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        publish_media_tree(source, destination, store)
    assert not destination.exists()
    assert not (media_repository / "reports/forge/media").exists()
    destination.mkdir(parents=True)
    marker = destination / "original.json"
    marker.write_bytes(b"archived")
    with pytest.raises(FileExistsError, match="preserve archived"):
        publish_media_tree(source, destination, store)
    assert marker.read_bytes() == b"archived"


def test_media_publication_rejects_unindexed_gifs_before_writes(media_repository, tmp_path):
    source = tmp_path / "source"
    _media_tree(source, b"GIF89a-fixture")
    (source / "preview.gif").write_bytes(b"GIF89a-unindexed")
    with pytest.raises(ValueError, match="missing from its certified index"):
        publish_media_tree(source, media_repository / "reports/new", RetainedMediaStore(media_repository))
    assert not (media_repository / "reports").exists()


def test_media_store_refuses_existing_canonical_symlinks(media_repository, tmp_path):
    source = tmp_path / "source.gif"
    source.write_bytes(b"GIF89a-fixture")
    canonical = media_repository / "reports/forge/media" / (hashlib.sha256(source.read_bytes()).hexdigest() + ".gif")
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(source)
    with pytest.raises(ValueError, match="must stay inside"):
        RetainedMediaStore(media_repository).retain(source)
    assert canonical.is_symlink()


def test_markdown_rebinding_preserves_receipt_links_and_remote_urls(tmp_path):
    original, copied, retained = tmp_path / "original", tmp_path / "copied", tmp_path / "retained.gif"
    text = "![Observed](actual.gif?download=1#frame) [Receipt](actual.json) [Remote](https://example.com/a.gif)"
    rebound = rewrite_media_links(text, original, copied, {(original / "actual.gif").resolve(): retained})
    assert "../retained.gif?download=1#frame" in rebound
    assert "[Receipt](actual.json)" in rebound
    assert "[Remote](https://example.com/a.gif)" in rebound


def test_hf_media_manifest_resolves_evicted_scoped_gif_offline_and_rejects_hash_drift(tmp_path):
    from experiments.forge.scoped_publications import (
        HF_MEDIA_MANIFEST,
        HF_MEDIA_MANIFEST_SCHEMA,
        REGISTRY,
        SCHEMA,
        hf_media_url,
        load_publications,
    )

    gif_rel = "reports/forge/pure-bcap/media/abc123/two_pole.gif"
    gif_sha = hashlib.sha256(b"GIF89a-evicted-bytes").hexdigest()
    prefix = "https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/" + ("a" * 40) + "/media/forge/"
    manifest = {
        "schema": HF_MEDIA_MANIFEST_SCHEMA,
        "url_prefix": prefix,
        "files": {gif_rel: [20, gif_sha, "b" * 40]},
    }
    manifest_path = tmp_path / HF_MEDIA_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    assert hf_media_url(tmp_path, gif_rel, gif_sha) == prefix + "pure-bcap/media/abc123/two_pole.gif"
    with pytest.raises(ValueError, match="changed after registration"):
        hf_media_url(tmp_path, gif_rel, "0" * 64)

    sidecar = tmp_path / Path(gif_rel).with_suffix(".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "kind": "actual_training_saved_observations_gif",
        "optimizer_updates_added": 0,
        "sampling_draws_added": 0,
        "gif_sha256": gif_sha,
        "task_id": "two_pole",
        "recorded_grade": "PASS",
    }
    sidecar.write_text(json.dumps(receipt))
    index_path = tmp_path / "reports/forge/media-index.json"
    index_bytes = json.dumps({
        "qualification_input": False,
        "items": [{**receipt, "family": "bcap-pure", "attempt_id": "att-1", "gif": gif_rel}],
    }).encode()
    index_path.write_bytes(index_bytes)
    registry_path = tmp_path / REGISTRY
    registry_path.write_text(json.dumps({
        "schema": SCHEMA,
        "qualification_input": False,
        "cohorts": [],
        "media": {
            "path": "reports/forge/media-index.json",
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
    }))

    loaded = load_publications(tmp_path)
    assert ("bcap-pure", "two_pole", "att-1") in loaded["media"]

    # Tampering manifest hash fails closed offline.
    manifest["files"][gif_rel][1] = "1" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed after registration"):
        load_publications(tmp_path)
