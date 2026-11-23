"""Resolve byte-exact archived evidence without training or qualification changes.

Only committed archive cards and explicit pinned Git identities are trusted.
Mirrors are mounted local directories, addressed by the archive SHA-256.
Hugging Face hydration is strictly opt-in (--allow-hf / --hf-repo /
PARTICLEGAN_FORGE_HF_REPO) and always verifies SHA-256, byte count, tar safety,
and declared member hashes before extraction.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

from .artifacts import validate_hf_locator
from .contracts import file_lock


class ArtifactError(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def _invalid(message):
    raise ArtifactError("INVALID", message)


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        _invalid("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        _invalid(f"unsafe artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {".", "..", ""} for p in value.rstrip("/").split("/")):
        _invalid(f"unsafe artifact path: {value!r}")
    return path.as_posix()


def archive_spec(card):
    """Normalize the three existing Forge archive card shapes, retaining identities."""
    declared = {}

    def add(name, digest):
        name, digest = _relative(name), _sha(digest)
        if name in declared and declared[name] != digest:
            _invalid(f"conflicting hashes for {name}")
        declared[name] = digest

    if isinstance(card.get("archive"), dict):
        archive = card["archive"]
        manifest = card.get("manifest", {})
        for name, digest in manifest.get("original_receipts_sha256", {}).items():
            add(name, digest)
        for name, digest in manifest.get("source_files_sha256", {}).items():
            add("frozen-scientific-source/" + name, digest)
        source_commit = manifest.get("executed_commit")
    elif isinstance(card.get("archive"), str):
        archive = {"path": card["archive"], "sha256": card.get("archive_sha256"),
                   "bytes": card.get("archive_bytes")}
        for field in ("files", "full_readouts"):
            for name, digest in card.get(field, {}).items():
                add(name, digest)
        source_commit = card.get("source_origin_commit")
    elif "path" in card and isinstance(card.get("files"), list):
        archive = card
        for row in card["files"]:
            add(row["archive_path"], row["sha256"])
        source_commit = card.get("source_commit")
    else:
        _invalid("unsupported archive card; use a Forge inventory, frozen-round or policy archive manifest")
    if not declared:
        _invalid("archive card declares no independently hashed members")
    size = archive.get("bytes")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        _invalid("invalid archive byte count")
    if not isinstance(archive.get("path"), str):
        _invalid("archive location must be a path")
    digest = _sha(archive.get("sha256"))
    raw_hf = card.get("hf")
    if raw_hf is None and isinstance(archive, dict):
        raw_hf = archive.get("hf")
    hf = None
    if raw_hf is not None:
        if not isinstance(raw_hf, dict):
            _invalid("archive card hf locator must be an object")
        hf_input = dict(raw_hf)
        hf_input.setdefault("path", f"archives/sha256/{digest}.tar.gz")
        try:
            hf = validate_hf_locator(hf_input)
        except ValueError as error:
            _invalid(str(error))
    return {"path": archive["path"], "sha256": digest, "bytes": size,
            "members": declared, "source_commit": source_commit, "hf": hf}


def _resolve_hf_locator(spec, *, allow_hf=False, hf_repo=None, hf_rev=None):
    env_repo = os.environ.get("PARTICLEGAN_FORGE_HF_REPO") or None
    env_rev = os.environ.get("PARTICLEGAN_FORGE_HF_REV") or None
    if not (allow_hf or hf_repo or env_repo):
        return None
    card_hf = spec.get("hf")
    default_path = f"archives/sha256/{spec['sha256']}.tar.gz"
    if hf_repo:
        same_repo = card_hf is not None and card_hf["repo"] == hf_repo
        locator = {
            "repo": hf_repo,
            "path": card_hf["path"] if same_repo else default_path,
            "repo_type": card_hf.get("repo_type", "dataset") if same_repo else "dataset",
        }
        revision = hf_rev or (card_hf.get("revision") if same_repo else None) or env_rev
    elif card_hf is not None:
        locator = {
            "repo": card_hf["repo"],
            "path": card_hf["path"],
            "repo_type": card_hf.get("repo_type", "dataset"),
        }
        revision = hf_rev or card_hf.get("revision") or env_rev
    elif env_repo:
        locator = {"repo": env_repo, "path": default_path, "repo_type": "dataset"}
        revision = hf_rev or env_rev
    else:
        return None
    if revision is not None:
        locator["revision"] = revision
    try:
        return validate_hf_locator(locator)
    except ValueError as error:
        _invalid(str(error))


def _download_from_hf(root: Path, spec: dict, locator: dict) -> Path:
    """Download a content-addressed archive from Hugging Face into local cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ArtifactError(
            "MISSING",
            "huggingface_hub is required for opt-in HF hydration; install with `pip install 'particlegan[hf]'`",
        ) from error
    cache_dir = Path(root) / "runs/forge/artifacts/sha256"
    target = cache_dir / f"{spec['sha256']}.tar.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(cache_dir / f".{spec['sha256']}.hf-download.lock"):
        if target.is_file():
            digest = hashlib.sha256()
            size = 0
            with target.open("rb") as existing:
                for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() == spec["sha256"] and (spec["bytes"] is None or size == spec["bytes"]):
                return target
        try:
            kwargs = {
                "repo_id": locator["repo"],
                "filename": locator["path"],
                "repo_type": locator.get("repo_type", "dataset"),
            }
            if locator.get("revision") is not None:
                kwargs["revision"] = locator["revision"]
            downloaded = Path(hf_hub_download(**kwargs))
        except Exception as error:
            raise ArtifactError(
                "MISSING",
                f"failed to download archive {spec['sha256']} from Hugging Face "
                f"{locator['repo']} ({locator['path']}): {error}",
            ) from error
        fd, tmp_name = tempfile.mkstemp(prefix=f".{spec['sha256']}-", suffix=".tar.gz", dir=cache_dir)
        tmp_path = Path(tmp_name)
        try:
            digest = hashlib.sha256()
            size = 0
            with downloaded.open("rb") as src, os.fdopen(fd, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(chunk)
                    dst.write(chunk)
                    size += len(chunk)
            if digest.hexdigest() != spec["sha256"] or (spec["bytes"] is not None and size != spec["bytes"]):
                tmp_path.unlink(missing_ok=True)
                _invalid(
                    f"archive checksum/size mismatch downloaded from HF {locator['repo']}:{locator['path']}; "
                    f"expected SHA-256 {spec['sha256']}"
                )
            tmp_path.replace(target)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return target


def _locations(root, spec, mirrors, locations):
    sha = spec["sha256"]
    candidates = [root / "runs/forge/artifacts/sha256" / (sha + ".tar.gz")]
    recorded = Path(spec["path"]).expanduser()
    candidates.append(recorded if recorded.is_absolute() else root / recorded)
    configured = {}
    if locations is not None:
        configured = json.loads(Path(locations).read_text()).get("archives", {}).get(sha, {})
        if not isinstance(configured, dict) or not isinstance(configured.get("paths", []), list):
            _invalid("location entry must contain a paths array")
        for name in configured.get("paths", []):
            path = Path(name).expanduser()
            # Configuration paths are relative to the configuration file, not the checkout.
            candidates.append(path if path.is_absolute() else Path(locations).resolve().parent / path)
    directories = [Path(p).expanduser() for p in mirrors]
    directories.extend(Path(p).expanduser() for p in os.environ.get("PARTICLEGAN_FORGE_ARTIFACT_MIRRORS", "").split(os.pathsep) if p)
    for directory in directories:
        candidates.extend([directory / "sha256" / (sha + ".tar.gz"), directory / (sha + ".tar.gz")])
    # A basename alone is not a stable identity: mirrors must be content addressed.
    return list(dict.fromkeys(p.absolute() for p in candidates)), configured


@contextmanager
def _open_verified_path(path: Path, spec: dict, retention: dict):
    # Hold one descriptor across verification and extraction (no path-swap race).
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != spec["sha256"] or (spec["bytes"] is not None and stream.tell() != spec["bytes"]):
            yield None
            return
        stream.seek(0)
        try:
            with tarfile.open(fileobj=stream, mode="r:*") as bundle:
                entries = {}
                for member in bundle.getmembers():
                    name = _relative(member.name)
                    if name in entries or not (member.isfile() or member.isdir()):
                        _invalid(f"duplicate or nonregular archive member: {name}")
                    entries[name] = member
                for name in entries:
                    for parent in PurePosixPath(name).parents:
                        if parent.as_posix() in entries and entries[parent.as_posix()].isfile():
                            _invalid(f"archive member has a file as its parent: {name}")
                missing = set(spec["members"]) - set(entries)
                if missing:
                    _invalid(f"archive omits declared members: {sorted(missing)[:5]}")
                yield spec, path, bundle, entries, retention
        except tarfile.TarError as error:
            _invalid(f"invalid tar archive: {error}")


@contextmanager
def _verified_archive(root, card, mirrors=(), locations=None, *, allow_hf=False, hf_repo=None, hf_rev=None):
    spec = archive_spec(card)
    candidates, retention = _locations(Path(root), spec, mirrors, locations)
    failures = []
    for path in candidates:
        if not path.is_file():
            continue
        with _open_verified_path(path, spec, retention) as verified:
            if verified is None:
                failures.append(str(path))
                continue
            yield verified
            return
    hf_locator_spec = _resolve_hf_locator(spec, allow_hf=allow_hf, hf_repo=hf_repo, hf_rev=hf_rev)
    if hf_locator_spec is not None:
        downloaded = _download_from_hf(Path(root), spec, hf_locator_spec)
        with _open_verified_path(downloaded, spec, retention) as verified:
            if verified is None:
                failures.append(str(downloaded))
            else:
                yield verified
                return
    if failures:
        _invalid(f"archive checksum/size mismatch at {failures}; expected SHA-256 {spec['sha256']}")
    raise ArtifactError("MISSING", f"archive {spec['sha256']} is unavailable; configure --mirror <directory> "
                        f"containing sha256/{spec['sha256']}.tar.gz, --locations <JSON>, or opt-in --allow-hf / "
                        f"--hf-repo <owner/repo>. Checked: " + ", ".join(str(p) for p in candidates))


def _selection(spec, entries, members):
    names = [_relative(name) for name in members] if members else sorted(spec["members"])
    if len(names) != len(set(names)):
        _invalid("duplicate requested member")
    for name in names:
        if name not in entries:
            raise ArtifactError("MISSING", f"member {name} is absent from verified archive {spec['sha256']}")
        if not entries[name].isfile():
            _invalid(f"selected member is not a regular file: {name}")
    return names


def _copy_member(bundle, member, output=None):
    digest = hashlib.sha256()
    with bundle.extractfile(member) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if output is not None:
                output.write(chunk)
    return digest.hexdigest()


def _receipt(spec, archive_path, retention, files):
    return {"artifact_kind": "archived_originals", "archive_sha256": spec["sha256"], "archive_path": str(archive_path),
            "source_commit": spec["source_commit"], "files": files,
            "retention": {"owner": retention.get("owner", "unassigned"),
                          "retain_until": retention.get("retain_until", "undeclared")},
            "training_launched": False, "qualification_changed": False}


def inspect_archive(root, card, *, members=(), mirrors=(), locations=None,
                    allow_hf=False, hf_repo=None, hf_rev=None):
    """Verify archive and selected member bytes; offline unless HF is opted in."""
    with _verified_archive(root, card, mirrors, locations,
                           allow_hf=allow_hf, hf_repo=hf_repo, hf_rev=hf_rev) as (spec, path, bundle, entries, retention):
        names = _selection(spec, entries, members)
        files = []
        for name in names:
            digest = _copy_member(bundle, entries[name])
            if name in spec["members"] and digest != spec["members"][name]:
                _invalid(f"member checksum mismatch: {name}")
            files.append({"path": name, "sha256": digest,
                          "integrity": "member_and_archive" if name in spec["members"] else "archive_only"})
        return {"status": "AVAILABLE", **_receipt(spec, path, retention, files),
                "archive_member_count": len(entries)}


@contextmanager
def _destination(destination):
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete, verified tree once; concurrent resolver calls are serialized.
    with file_lock(destination.parent / ".forge-artifact-hydration.lock"):
        if os.path.lexists(destination):
            _invalid(f"destination already exists; choose a new isolated directory: {destination}")
        with tempfile.TemporaryDirectory(prefix=".forge-hydrate-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "tree"
            staging.mkdir()
            yield staging
            if os.path.lexists(destination):
                _invalid(f"destination appeared during hydration: {destination}")
            staging.rename(destination)


def hydrate_archive(root, card, destination, *, members=(), mirrors=(), locations=None,
                    allow_hf=False, hf_repo=None, hf_rev=None):
    """Restore originals into a new isolated directory; never overwrite a checkout."""
    with _verified_archive(root, card, mirrors, locations,
                           allow_hf=allow_hf, hf_repo=hf_repo, hf_rev=hf_rev) as (spec, path, bundle, entries, retention):
        names = _selection(spec, entries, members)
        files = []
        with _destination(destination) as staging:
            for name in names:
                output_path = staging / name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("xb") as output:
                    digest = _copy_member(bundle, entries[name], output)
                if name in spec["members"] and digest != spec["members"][name]:
                    _invalid(f"member checksum mismatch: {name}")
                files.append({"path": name, "sha256": digest,
                              "integrity": "member_and_archive" if name in spec["members"] else "archive_only"})
            receipt = _receipt(spec, path, retention, files)
            # Avoid inventing evidence if an archive itself owns this filename.
            receipt_path = staging / "forge-hydration-receipt.json"
            if receipt_path.exists():
                _invalid("archive uses reserved forge-hydration-receipt.json path")
            receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        return {"status": "HYDRATED", "destination": str(Path(destination).absolute()), **receipt}


def hydrate_git(root, *, commit, path, blob, sha256, destination):
    """Restore an explicitly pinned original, preserving commit, blob and byte hash."""
    for value in (commit, blob):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
            _invalid("Git commit and blob must be complete hexadecimal object IDs")
    path, sha256 = _relative(path), _sha(sha256)
    try:
        actual = subprocess.check_output(["git", "rev-parse", "--verify", commit + ":" + path], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        if actual != blob:
            _invalid("pinned commit/path differs from the declared Git blob")
        content = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=root, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ArtifactError("MISSING", f"pinned Git original is unavailable; git fetch origin {commit}, "
                            "then retry the exact commit/path/blob identities") from error
    if hashlib.sha256(content).hexdigest() != sha256:
        _invalid("pinned Git original SHA-256 differs")
    receipt = {"artifact_kind": "git_pinned_bytes", "commit": commit, "git_blob": blob, "path": path, "sha256": sha256,
               "training_launched": False, "qualification_changed": False}
    with _destination(destination) as staging:
        target = staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        if (staging / "forge-hydration-receipt.json").exists():
            _invalid("original uses reserved forge-hydration-receipt.json path")
        (staging / "forge-hydration-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"status": "HYDRATED", "destination": str(Path(destination).absolute()), **receipt}
