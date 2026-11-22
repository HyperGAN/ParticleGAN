"""Storage discipline of what Git tracks, and the Forge declaration check.

These read the Git index (``git ls-files``), never ``git status``: stray local
files, runs or checkpoints in a developer checkout are allowed and must not
fail the suite. Only content that would be committed is checked.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

BULK_SUFFIXES = (".pt", ".pth", ".npz", ".jsonl", ".h5")
MAX_TRACKED_MB = 25


@pytest.fixture(scope="module")
def tracked(repo_root) -> list[str]:
    try:
        done = subprocess.run(["git", "ls-files", "-z"], cwd=str(repo_root), capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"not a Git checkout: {error}")
    return [name for name in done.stdout.decode().split("\0") if name]


def test_public_api_reports_track_no_bulk_tensors_or_streams(tracked):
    bulk = [name for name in tracked
            if name.startswith("reports/toy_audit/api_contract/") and name.endswith(BULK_SUFFIXES)]
    assert not bulk


def test_public_api_receipts_are_compact(repo_root, tracked):
    receipts = [name for name in tracked if name.startswith("reports/") and Path(name).name == "receipt.json"]
    oversized = [name for name in receipts if (repo_root / name).stat().st_size > 10 * 1024 * 1024]
    assert not oversized


def test_no_tracked_file_exceeds_the_size_cap(repo_root, tracked):
    cap = MAX_TRACKED_MB * 1024 * 1024
    oversized = [name for name in tracked if (repo_root / name).is_file() and (repo_root / name).stat().st_size > cap]
    assert not oversized


def test_runs_and_editor_artifacts_are_never_tracked(tracked):
    assert not [name for name in tracked if name.startswith("runs/")]
    assert not [name for name in tracked if name.endswith((".swp", ".swo", "~"))]


def _git(repo_root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo_root), capture_output=True)


def test_no_tracked_file_matches_an_ignore_rule(repo_root, tracked):
    # Ignored bulk must never be force-added; files that code reads from the
    # checkout are re-included explicitly in .gitignore instead.
    done = _git(repo_root, "ls-files", "-z", "--cached", "--ignored", "--exclude-standard")
    assert done.returncode == 0, done.stderr.decode()
    assert [name for name in done.stdout.decode().split("\0") if name] == []


def test_repo_root_gifs_are_cited_by_the_readme(repo_root, tracked):
    readme = (repo_root / "README.md").read_text()
    stray = [name for name in tracked if "/" not in name and name.endswith(".gif") and f"({name})" not in readme]
    assert not stray, f"unreferenced root GIFs belong in reports/log-archive.json: {stray}"


def test_archived_paths_stay_out_of_the_tree_with_their_recorded_blobs(repo_root, tracked):
    import json

    index = json.loads((repo_root / "reports/log-archive.json").read_text())
    current = set(tracked)
    for archive in index["archives"]:
        files = archive["files"]
        assert not current & {path for path, _, _ in files}, "an archived path is tracked again"
        if _git(repo_root, "cat-file", "-e", archive["commit"] + "^{commit}").returncode:
            continue  # shallow checkout: the archive commit is not available locally
        listing = _git(repo_root, "ls-tree", "-r", "-z", "--long", archive["commit"], "--", *(p for p, _, _ in files))
        assert listing.returncode == 0, listing.stderr.decode()
        found = {}
        for row in filter(None, listing.stdout.decode().split("\0")):
            meta, path = row.split("\t", 1)
            _, _, blob, size = meta.split()
            found[path] = [path, int(size), blob]
        assert [found.get(path) for path, _, _ in files] == [list(entry) for entry in files]


def test_forge_validate_accepts_the_committed_declarations(run_cli):
    res = run_cli([sys.executable, "-m", "experiments.forge", "validate"], timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]


def test_no_tracked_link_points_at_a_pruned_branch(repo_root):
    # Branch-name GitHub links 404 once the branch is pruned; pin them to the
    # recorded tip SHA instead (reports/branch-prune-*.json).
    import json
    import re

    deleted: dict[str, str] = {}
    for record in sorted((repo_root / "reports").glob("branch-prune-*.json")):
        deleted.update(json.loads(record.read_text())["deleted"])
    done = _git(repo_root, "grep", "-z", "-nE", r"github\.com/(HyperGAN|255BITS)/ParticleGAN/(blob|tree)/")
    link = re.compile(r"github\.com/(?:HyperGAN|255BITS)/ParticleGAN/(?:blob|tree)/([^)\s\"'>]+)")
    stale = [
        row for row in done.stdout.decode(errors="replace").split("\n")
        for ref in link.findall(row)
        if any(ref == branch or ref.startswith(branch + "/") for branch in deleted)
    ]
    assert not stale, f"pin these links to the pruned branch's recorded tip SHA: {stale}"
