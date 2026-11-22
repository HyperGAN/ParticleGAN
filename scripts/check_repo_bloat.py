"""Reject newly tracked binary/bulk bloat under reports/ while allowing legacy exceptions.

Reads tracked paths from `git ls-files` and checks everything under `reports/`
against banned binary/bulk extensions and the 500 KB JSON cap. Pre-existing
legacy paths listed in `configs/repo_bloat_allowlist.json` are temporarily
allowed so eviction PRs can shrink the allowlist incrementally while blocking
any new stray bloat immediately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = Path("configs/repo_bloat_allowlist.json")
SCOPE_PREFIX = "reports/"
BANNED_EXTENSIONS = frozenset({
    ".ckpt",
    ".gif",
    ".gz",
    ".html",
    ".mp4",
    ".npz",
    ".pt",
    ".pth",
    ".zip",
})
MAX_JSON_BYTES = 500 * 1024


def tracked_files(root: Path) -> list[str]:
    done = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        capture_output=True,
        check=True,
    )
    return [name for name in done.stdout.decode().split("\0") if name]


def scan_bloat(root: Path, tracked: list[str] | None = None) -> dict[str, str]:
    """Return mapping of violating tracked paths under reports/ to reason strings."""
    root = Path(root).resolve()
    paths = tracked_files(root) if tracked is None else tracked
    violations: dict[str, str] = {}
    for rel in sorted(paths):
        if not rel.startswith(SCOPE_PREFIX):
            continue
        full = root / rel
        ext = full.suffix.lower()
        if ext in BANNED_EXTENSIONS:
            violations[rel] = f"banned extension {ext}"
        elif ext == ".json" and full.is_file():
            size = full.stat().st_size
            if size > MAX_JSON_BYTES:
                violations[rel] = f"oversized JSON ({size:,} bytes > {MAX_JSON_BYTES:,} bytes)"
    return violations


def load_allowlist(path: Path) -> set[str]:
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != 1 or not isinstance(doc.get("allowed_paths"), list):
        raise ValueError(f"unsupported bloat allowlist schema in {path}")
    allowed = doc["allowed_paths"]
    if any(not isinstance(item, str) or not item.startswith(SCOPE_PREFIX) for item in allowed):
        raise ValueError("allowlist entries must be paths under reports/")
    return set(allowed)


def write_allowlist(path: Path, allowed_paths: set[str] | list[str]) -> None:
    payload = {
        "schema_version": 1,
        "policy": {
            "scope_prefix": SCOPE_PREFIX,
            "banned_extensions": sorted(BANNED_EXTENSIONS),
            "max_json_bytes": MAX_JSON_BYTES,
        },
        "allowed_paths": sorted(set(allowed_paths)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def check_repo_bloat(
    root: Path,
    *,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    tracked: list[str] | None = None,
    strict_allowlist: bool = False,
) -> dict:
    root = Path(root).resolve()
    resolved_allowlist = allowlist_path if allowlist_path.is_absolute() else (root / allowlist_path)
    violations = scan_bloat(root, tracked=tracked)
    allowed = load_allowlist(resolved_allowlist) if resolved_allowlist.is_file() else set()
    new_violations = {p: reason for p, reason in sorted(violations.items()) if p not in allowed}
    stale_allowed = sorted(allowed - set(violations))
    passed = not new_violations and (not strict_allowlist or not stale_allowed)
    return {
        "passed": passed,
        "total_flagged": len(violations),
        "allowlisted": len(violations) - len(new_violations),
        "new_violations": new_violations,
        "stale_allowed_count": len(stale_allowed),
        "stale_allowed": stale_allowed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--update-allowlist", action="store_true", help="regenerate allowlist from currently tracked legacy bloat")
    parser.add_argument("--prune-allowlist", action="store_true", help="remove evicted paths from allowlist without adding new ones")
    parser.add_argument("--strict-allowlist", action="store_true", help="also fail if allowlist contains already-evicted paths")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    allowlist = args.allowlist if args.allowlist.is_absolute() else (root / args.allowlist)

    if args.update_allowlist:
        violations = scan_bloat(root)
        write_allowlist(allowlist, sorted(violations))
        print(f"Recorded {len(violations)} allowlisted paths in {allowlist.relative_to(root)}")
        return 0

    if args.prune_allowlist:
        violations = scan_bloat(root)
        existing = load_allowlist(allowlist) if allowlist.is_file() else set()
        pruned = sorted(existing & set(violations))
        write_allowlist(allowlist, pruned)
        print(f"Pruned allowlist to {len(pruned)} active paths (removed {len(existing) - len(pruned)})")
        return 0

    report = check_repo_bloat(root, allowlist_path=allowlist, strict_allowlist=args.strict_allowlist)
    if report["new_violations"]:
        for path, reason in report["new_violations"].items():
            print(f"BLOAT: {path}: {reason}")
        print(
            f"Found {len(report['new_violations'])} unallowlisted bloated file(s) under {SCOPE_PREFIX}. "
            "Upload bulk media/artifacts to Hugging Face instead of committing them to Git."
        )
        return 1
    if args.strict_allowlist and report["stale_allowed"]:
        print(
            f"Allowlist has {report['stale_allowed_count']} evicted path(s); "
            "run `python scripts/check_repo_bloat.py --prune-allowlist`."
        )
        return 1
    extra = (
        f"; {report['stale_allowed_count']} evicted path(s) can be pruned"
        if report["stale_allowed_count"]
        else ""
    )
    print(f"Repo bloat guard passed: 0 new violations ({report['allowlisted']} legacy allowlisted{extra})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
