"""Unit tests for the CI repository bloat guard."""
from __future__ import annotations

from pathlib import Path

from scripts.check_repo_bloat import (
    MAX_JSON_BYTES,
    check_repo_bloat,
    load_allowlist,
    main,
    scan_bloat,
    write_allowlist,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_live_repository_passes_bloat_guard():
    report = check_repo_bloat(REPO_ROOT)
    assert report["passed"] is True, f"unexpected unallowlisted bloat: {report['new_violations']}"
    assert report["new_violations"] == {}


def test_blocks_new_banned_extension_and_oversized_json(tmp_path):
    (tmp_path / "reports/new_study").mkdir(parents=True)
    (tmp_path / "reports/new_study/demo.gif").write_bytes(b"GIF89a")
    (tmp_path / "reports/new_study/small.json").write_text("{}\n")
    (tmp_path / "reports/new_study/huge.json").write_bytes(b"{" + b" " * MAX_JSON_BYTES + b"}")
    (tmp_path / "reports/new_study/notes.md").write_text("# Notes\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/hero.gif").write_bytes(b"GIF89a")

    tracked = [
        "reports/new_study/demo.gif",
        "reports/new_study/small.json",
        "reports/new_study/huge.json",
        "reports/new_study/notes.md",
        "docs/hero.gif",
    ]
    violations = scan_bloat(tmp_path, tracked=tracked)
    assert set(violations) == {"reports/new_study/demo.gif", "reports/new_study/huge.json"}

    allowlist = tmp_path / "configs/repo_bloat_allowlist.json"
    write_allowlist(allowlist, ["reports/new_study/demo.gif"])
    report = check_repo_bloat(tmp_path, allowlist_path=allowlist, tracked=tracked)
    assert report["passed"] is False
    assert set(report["new_violations"]) == {"reports/new_study/huge.json"}
    assert report["allowlisted"] == 1


def test_allowlist_prune_and_strict_mode(tmp_path):
    (tmp_path / "reports/old").mkdir(parents=True)
    (tmp_path / "reports/old/kept.gif").write_bytes(b"GIF89a")
    tracked = ["reports/old/kept.gif"]
    allowlist = tmp_path / "configs/repo_bloat_allowlist.json"
    write_allowlist(allowlist, ["reports/old/kept.gif", "reports/old/evicted.gif"])

    # Default mode passes when allowlist has already-evicted paths
    report = check_repo_bloat(tmp_path, allowlist_path=allowlist, tracked=tracked)
    assert report["passed"] is True
    assert report["stale_allowed"] == ["reports/old/evicted.gif"]

    # Strict mode flags evicted paths until pruned
    strict_report = check_repo_bloat(tmp_path, allowlist_path=allowlist, tracked=tracked, strict_allowlist=True)
    assert strict_report["passed"] is False
    assert load_allowlist(allowlist) == {"reports/old/kept.gif", "reports/old/evicted.gif"}


def test_cli_returns_zero_on_live_repo(capsys):
    assert main(["--root", str(REPO_ROOT)]) == 0
    assert "Repo bloat guard passed" in capsys.readouterr().out
