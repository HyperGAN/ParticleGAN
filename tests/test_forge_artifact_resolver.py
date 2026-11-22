"""Original archive retrieval controls: synthetic bytes, no scientific execution."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from experiments.forge.__main__ import main
from experiments.forge.artifact_resolver import (
    ArtifactError, archive_spec, hydrate_archive, hydrate_git, inspect_archive,
)


def sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def original_archive(tmp_path):
    """An actual tar archive with a complete original envelope, source and state."""
    payloads = {
        "reports/forge/attempts/attempt-one/request.json": b'{"source":{"commit":"pinned-original"}}\n',
        "reports/forge/attempts/attempt-one/evidence.json": b'{"original_evidence":true}\n',
        "reports/forge/attempts/attempt-one/result.json": b'{"gate_status":"FAIL","metric":0.25}\n',
        "frozen-scientific-source/particlegan/training.py": b"# exact frozen reproduction source\n",
        "queue/attempt-one/final-state.pt": b"synthetic checkpoint fixture bytes",
    }
    archive = tmp_path / "original.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in payloads.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            bundle.addfile(entry, io.BytesIO(content))
    card = {
        "schema_version": 1, "kind": "completed-frozen-scientific-round-archive",
        "archive": {"path": "/unavailable/original-machine/archive.tar.gz",
                    "sha256": sha(archive.read_bytes()), "bytes": archive.stat().st_size},
        "manifest": {"executed_commit": "a" * 40,
                     "original_receipts_sha256": {k: sha(v) for k, v in payloads.items() if k.startswith("reports/")},
                     "source_files_sha256": {"particlegan/training.py": sha(payloads["frozen-scientific-source/particlegan/training.py"])}}
    }
    mirror = tmp_path / "mirror"
    mirrored = mirror / "sha256" / (card["archive"]["sha256"] + ".tar.gz")
    mirrored.parent.mkdir(parents=True)
    mirrored.write_bytes(archive.read_bytes())
    return card, payloads, mirror, mirrored


def test_fresh_agent_hydrates_full_envelope_and_source_from_configured_mirror(tmp_path, original_archive):
    card, payloads, mirror, _ = original_archive
    fresh_checkout = tmp_path / "fresh-checkout"
    fresh_checkout.mkdir()
    output = tmp_path / "fresh-agent-originals"
    receipt = hydrate_archive(fresh_checkout, card, output, mirrors=[mirror])
    assert receipt["status"] == "HYDRATED"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["training_launched"] is False
    assert receipt["qualification_changed"] is False
    for row in receipt["files"]:
        assert (output / row["path"]).read_bytes() == payloads[row["path"]]
        assert row["integrity"] == "member_and_archive"
    assert len(receipt["files"]) == 4
    assert json.loads((output / "forge-hydration-receipt.json").read_text())["archive_sha256"] == card["archive"]["sha256"]
    assert not (fresh_checkout / "reports/forge/attempts").exists()


def test_selected_checkpoint_retains_archive_identity_and_computed_member_digest(tmp_path, original_archive):
    card, payloads, mirror, _ = original_archive
    name = "queue/attempt-one/final-state.pt"
    report = hydrate_archive(tmp_path, card, tmp_path / "state", members=[name], mirrors=[mirror])
    assert report["files"] == [{"path": name, "sha256": sha(payloads[name]), "integrity": "archive_only"}]
    assert report["archive_sha256"] == card["archive"]["sha256"]


def test_inspection_verifies_members_without_writing_queue_or_artifact_cache(tmp_path, original_archive):
    card, _, mirror, _ = original_archive
    root = tmp_path / "inspection"
    root.mkdir()
    report = inspect_archive(root, card, mirrors=[mirror])
    assert report["status"] == "AVAILABLE"
    assert report["archive_member_count"] == 5
    assert not list(root.iterdir())


def test_missing_is_actionable_and_does_not_create_destination(tmp_path, original_archive):
    card, _, _, _ = original_archive
    destination = tmp_path / "not-hydrated"
    with pytest.raises(ArtifactError, match="configure --mirror") as error:
        hydrate_archive(tmp_path, card, destination)
    assert error.value.status == "MISSING"
    assert not destination.exists()


def test_bad_archive_checksum_is_invalid_and_cannot_publish(tmp_path, original_archive):
    card, _, mirror, mirrored = original_archive
    mirrored.write_bytes(b"corrupted archive")
    with pytest.raises(ArtifactError, match="checksum/size mismatch") as error:
        hydrate_archive(tmp_path, card, tmp_path / "output", mirrors=[mirror])
    assert error.value.status == "INVALID"
    assert not (tmp_path / "output").exists()


def test_bad_declared_member_checksum_rolls_back_entire_hydration(tmp_path, original_archive):
    card, _, mirror, _ = original_archive
    card["manifest"]["original_receipts_sha256"]["reports/forge/attempts/attempt-one/result.json"] = "0" * 64
    with pytest.raises(ArtifactError, match="member checksum mismatch"):
        hydrate_archive(tmp_path, card, tmp_path / "output", mirrors=[mirror])
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".forge-hydrate-*"))


def test_never_overwrites_even_an_empty_destination_or_symlink(tmp_path, original_archive):
    card, _, mirror, _ = original_archive
    existing = tmp_path / "existing"
    existing.mkdir()
    for destination in (existing, tmp_path / "link"):
        if destination != existing:
            destination.symlink_to(existing, target_is_directory=True)
        with pytest.raises(ArtifactError, match="already exists"):
            hydrate_archive(tmp_path, card, destination, mirrors=[mirror])
    assert not list(existing.iterdir())


def test_member_missing_does_not_publish_partial_tree(tmp_path, original_archive):
    card, _, mirror, _ = original_archive
    with pytest.raises(ArtifactError, match="member absent.json is absent") as error:
        hydrate_archive(tmp_path, card, tmp_path / "output", mirrors=[mirror], members=["absent.json"])
    assert error.value.status == "MISSING"
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("attack", ["../escape", "/absolute", "a/../escape", "a\\escape", "symlink", "hardlink", "device", "duplicate"])
def test_unsafe_archive_even_unselected_member_is_rejected(tmp_path, attack):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        good = tarfile.TarInfo("good.json")
        good.size = 2
        bundle.addfile(good, io.BytesIO(b"{}"))
        member = tarfile.TarInfo(attack)
        if attack == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
        elif attack == "hardlink":
            member.type = tarfile.LNKTYPE
            member.linkname = "good.json"
        elif attack == "device":
            member.type = tarfile.CHRTYPE
        elif attack == "duplicate":
            member.name = "good.json"
        bundle.addfile(member)
    card = {"archive": str(archive), "archive_sha256": sha(archive.read_bytes()), "files": {"good.json": sha(b"{}")}}
    with pytest.raises(ArtifactError) as error:
        hydrate_archive(tmp_path, card, tmp_path / "output", members=["good.json"])
    assert error.value.status == "INVALID"
    assert not (tmp_path / "output").exists()


def test_local_content_addressed_cache_and_env_mirror(tmp_path, original_archive, monkeypatch):
    card, _, mirror, mirrored = original_archive
    monkeypatch.setenv("PARTICLEGAN_FORGE_ARTIFACT_MIRRORS", str(mirror))
    assert inspect_archive(tmp_path, card)["status"] == "AVAILABLE"
    monkeypatch.delenv("PARTICLEGAN_FORGE_ARTIFACT_MIRRORS")
    cached = tmp_path / "runs/forge/artifacts/sha256" / mirrored.name
    cached.parent.mkdir(parents=True)
    cached.write_bytes(mirrored.read_bytes())
    assert inspect_archive(tmp_path, card)["archive_path"] == str(cached)


def test_location_configuration_preserves_owner_and_retention(tmp_path, original_archive):
    card, _, _, mirrored = original_archive
    locations = tmp_path / "locations.json"
    locations.write_text(json.dumps({"archives": {card["archive"]["sha256"]: {
        "paths": [str(mirrored.relative_to(tmp_path))], "owner": "research-archive-maintainer", "retain_until": "2030-01-01"}}}))
    assert inspect_archive(tmp_path, card, locations=locations)["retention"] == {
        "owner": "research-archive-maintainer", "retain_until": "2030-01-01"}


@pytest.mark.parametrize("relative", ["../evil", "/evil", "x\\evil", "x/./evil"])
def test_manifest_paths_cannot_escape(relative):
    with pytest.raises(ArtifactError, match="unsafe artifact path"):
        archive_spec({"archive": "x", "archive_sha256": "a" * 64, "files": {relative: "b" * 64}})


def test_actual_committed_archive_cards_normalize_without_hydration():
    root = Path(__file__).resolve().parents[1]
    names = ["technique-inventory-archive.json", "release07-task-adaptation-archive.json",
             "family-winner-round1/phase1-archive.json", "family-winner-round1/policy-round1-archive.json",
             "family-winner-round1/policy-round2-archive.json", "family-winner-round1/policy-round3-archive.json",
             "family-winner-round1/policy-round4-archive.json", "family-winner-round1/policy-capacity-archive.json",
             "family-winner-round1/policy-atlas-initial-launch/archive.json"]
    for name in names:
        spec = archive_spec(json.loads((root / "reports/forge" / name).read_text()))
        assert spec["members"]
        assert len(spec["sha256"]) == 64


def test_git_original_checks_all_three_identities_before_hydrating(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, stderr=subprocess.DEVNULL, text=True).strip()
    git("init")
    git("config", "user.name", "Artifact fixture")
    git("config", "user.email", "fixture@example.invalid")
    payload = b'{"original_request":true}\n'
    (tmp_path / "request.json").write_bytes(payload)
    git("add", "request.json")
    git("commit", "-m", "Pinned fixture")
    commit, blob = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:request.json")
    output = tmp_path / "retrieved"
    report = hydrate_git(tmp_path, commit=commit, path="request.json", blob=blob, sha256=sha(payload), destination=output)
    assert report["commit"] == commit and report["git_blob"] == blob
    assert (output / "request.json").read_bytes() == payload
    for changes, message in [({"blob": "b" * 40}, "declared Git blob"), ({"sha256": "0" * 64}, "SHA-256 differs"),
                             ({"commit": "a" * 40}, "git fetch origin")]:
        options = {"commit": commit, "path": "request.json", "blob": blob, "sha256": sha(payload), "destination": tmp_path / "bad"}
        options.update(changes)
        with pytest.raises(ArtifactError, match=message):
            hydrate_git(tmp_path, **options)
        assert not (tmp_path / "bad").exists()


def test_artifacts_cli_is_read_only_and_missing_has_nonzero_exit(tmp_path, original_archive, capsys, monkeypatch):
    card, _, mirror, _ = original_archive
    manifest = tmp_path / "archive.json"
    manifest.write_text(json.dumps(card))
    # Any queue import/construction is unnecessary for artifact lookup.
    import experiments.forge.queue
    monkeypatch.setattr(experiments.forge.queue, "Queue", lambda *a, **k: pytest.fail("artifact lookup constructed a queue"))
    assert main(["--root", str(tmp_path), "artifacts", "inspect", str(manifest), "--mirror", str(mirror)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "AVAILABLE"
    assert main(["--root", str(tmp_path), "artifacts", "inspect", str(manifest)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "MISSING"
    card["archive"]["sha256"] = "invalid"
    manifest.write_text(json.dumps(card))
    assert main(["--root", str(tmp_path), "artifacts", "inspect", str(manifest)]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID"


def test_new_agent_process_hydrates_from_card_and_mirror_only(tmp_path, original_archive):
    card, payloads, mirror, _ = original_archive
    fresh = tmp_path / "new-checkout"
    fresh.mkdir()
    manifest = fresh / "archive.json"
    manifest.write_text(json.dumps(card))
    destination = fresh / "runs/forge/originals"
    command = [sys.executable, "-m", "experiments.forge", "--root", str(fresh), "artifacts", "hydrate",
               "archive.json", "--mirror", str(mirror), "--destination", str(destination)]
    run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    receipt = json.loads(run.stdout)
    assert receipt["status"] == "HYDRATED"
    for row in receipt["files"]:
        assert (destination / row["path"]).read_bytes() == payloads[row["path"]]


def test_concurrent_hydration_has_one_publisher_and_one_explicit_refusal(tmp_path, original_archive):
    from concurrent.futures import ThreadPoolExecutor
    card, _, mirror, _ = original_archive
    output = tmp_path / "shared-output"
    def restore():
        try:
            return hydrate_archive(tmp_path, card, output, mirrors=[mirror])["status"]
        except ArtifactError as error:
            return error.status
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: restore(), range(2)))
    assert sorted(results) == ["HYDRATED", "INVALID"]
    assert (output / "reports/forge/attempts/attempt-one/result.json").exists()


def test_corrupt_original_location_does_not_hide_verified_mirror(tmp_path, original_archive):
    card, _, mirror, _ = original_archive
    corrupt = tmp_path / "bad-original.tar.gz"
    corrupt.write_bytes(b"bad original")
    card["archive"]["path"] = str(corrupt)
    assert inspect_archive(tmp_path, card, mirrors=[mirror])["status"] == "AVAILABLE"


def test_hf_locator_is_offline_by_default_and_never_calls_hub(tmp_path, original_archive, monkeypatch):
    import huggingface_hub

    card, _, _, _ = original_archive
    card["hf"] = {
        "repo": "HyperGAN/particlegan-forge-evidence",
        "revision": "b" * 40,
        "path": f"archives/sha256/{card['archive']['sha256']}.tar.gz",
    }

    def forbid_network(**kwargs):
        raise AssertionError(f"unexpected HF network call while offline: {kwargs}")

    monkeypatch.delenv("PARTICLEGAN_FORGE_HF_REPO", raising=False)
    monkeypatch.delenv("PARTICLEGAN_FORGE_HF_REV", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbid_network)
    with pytest.raises(ArtifactError, match="allow-hf") as error:
        inspect_archive(tmp_path, card)
    assert error.value.status == "MISSING"


def test_hf_opt_in_hydrates_from_card_locator_and_caches_verified_tarball(tmp_path, original_archive, monkeypatch):
    import huggingface_hub

    card, payloads, _, mirrored = original_archive
    rev = "c" * 40
    hf_path = f"archives/sha256/{card['archive']['sha256']}.tar.gz"
    card["hf"] = {
        "repo": "HyperGAN/particlegan-forge-evidence",
        "revision": rev,
        "path": hf_path,
    }
    calls = []

    def fake_download(*, repo_id, filename, repo_type="dataset", revision=None):
        calls.append((repo_id, filename, repo_type, revision))
        return str(mirrored)

    monkeypatch.delenv("PARTICLEGAN_FORGE_HF_REPO", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    root = tmp_path / "checkout"
    root.mkdir()
    output = tmp_path / "hydrated"
    receipt = hydrate_archive(root, card, output, allow_hf=True)
    assert receipt["status"] == "HYDRATED"
    assert calls == [("HyperGAN/particlegan-forge-evidence", hf_path, "dataset", rev)]
    cached = root / "runs/forge/artifacts/sha256" / f"{card['archive']['sha256']}.tar.gz"
    assert cached.is_file()
    assert receipt["archive_path"] == str(cached)
    for row in receipt["files"]:
        assert (output / row["path"]).read_bytes() == payloads[row["path"]]
    # Second inspection hits local cache without any additional HF call.
    assert inspect_archive(root, card, allow_hf=True)["status"] == "AVAILABLE"
    assert len(calls) == 1


def test_hf_opt_in_constructs_path_from_cli_repo_and_rev_or_env(tmp_path, original_archive, monkeypatch, capsys):
    import huggingface_hub

    card, _, _, mirrored = original_archive
    rev = "d" * 40
    expected_path = f"archives/sha256/{card['archive']['sha256']}.tar.gz"
    calls = []

    def fake_download(*, repo_id, filename, repo_type="dataset", revision=None):
        calls.append((repo_id, filename, repo_type, revision))
        return str(mirrored)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    root = tmp_path / "cli-checkout"
    root.mkdir()
    manifest = root / "archive.json"
    manifest.write_text(json.dumps(card))
    rc = main([
        "--root", str(root), "artifacts", "inspect", "archive.json",
        "--hf-repo", "HyperGAN/particlegan-forge-evidence", "--hf-rev", rev,
    ])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "AVAILABLE"
    assert calls == [("HyperGAN/particlegan-forge-evidence", expected_path, "dataset", rev)]

    # Env var opt-in on a fresh root
    root_env = tmp_path / "env-checkout"
    root_env.mkdir()
    monkeypatch.setenv("PARTICLEGAN_FORGE_HF_REPO", "HyperGAN/particlegan-forge-evidence")
    monkeypatch.setenv("PARTICLEGAN_FORGE_HF_REV", rev)
    assert inspect_archive(root_env, card)["status"] == "AVAILABLE"
    assert len(calls) == 2


def test_hf_corrupt_download_is_rejected_without_polluting_cache(tmp_path, original_archive, monkeypatch):
    import huggingface_hub

    card, _, _, _ = original_archive
    corrupt = tmp_path / "corrupt-remote.tar.gz"
    corrupt.write_bytes(b"corrupted remote bytes")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(corrupt))
    root = tmp_path / "corrupt-checkout"
    root.mkdir()
    with pytest.raises(ArtifactError, match="checksum/size mismatch") as error:
        hydrate_archive(
            root, card, tmp_path / "out",
            allow_hf=True, hf_repo="HyperGAN/particlegan-forge-evidence", hf_rev="e" * 40,
        )
    assert error.value.status == "INVALID"
    cached = root / "runs/forge/artifacts/sha256" / f"{card['archive']['sha256']}.tar.gz"
    assert not cached.exists()
    assert not (tmp_path / "out").exists()


def test_hf_invalid_revision_is_rejected(tmp_path, original_archive):
    card, _, _, _ = original_archive
    with pytest.raises(ArtifactError, match="40-character") as error:
        inspect_archive(
            tmp_path, card, allow_hf=True,
            hf_repo="HyperGAN/particlegan-forge-evidence", hf_rev="main",
        )
    assert error.value.status == "INVALID"
