"""Deploy the ParticleGAN Static Leaderboard Space to Hugging Face.

Validates required static Space assets and YAML frontmatter locally before
uploading ``ops/hf_space/`` to ``HyperGAN/particlegan-leaderboard``.
Use ``--dry-run`` to inspect the upload plan offline with zero network calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_SPACE_ID = "HyperGAN/particlegan-leaderboard"
DEFAULT_COMMIT_MESSAGE = "Deploy ParticleGAN Forge Leaderboard Space"
REQUIRED_FILES = ("README.md", "index.html", "app.js", "styles.css")
REQUIRED_FRONTMATTER_KEYS = ("title", "emoji", "sdk", "pinned", "license", "short_description")
ALLOWED_COLORS = frozenset({"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"})
MAX_SHORT_DESCRIPTION_LEN = 60
IGNORE_PATTERNS = (
    "deploy.py",
    "__pycache__",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
)
DELETE_PATTERNS = ("style.css",)


def default_source_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_frontmatter(readme_text: str) -> dict[str, Any]:
    """Parse YAML frontmatter block from a Hugging Face Space README.md."""
    lines = readme_text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Space README.md must begin with a YAML frontmatter block ('---')")
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError("Space README.md frontmatter is missing a closing '---' delimiter")

    block = "\n".join(lines[1:end_idx])
    try:
        import yaml

        parsed = yaml.safe_load(block)
    except ImportError:
        parsed = {}
        for line in lines[1:end_idx]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Malformed frontmatter line: {line!r}")
            key, raw_val = stripped.split(":", 1)
            val = raw_val.strip().strip("'\"")
            if val.lower() == "true":
                parsed[key.strip()] = True
            elif val.lower() == "false":
                parsed[key.strip()] = False
            else:
                parsed[key.strip()] = val

    if not isinstance(parsed, dict):
        raise ValueError("Space README.md frontmatter must decode to a mapping")
    return parsed


def validate_space_dir(source_dir: Path) -> dict[str, Any]:
    """Validate that required static Space files and README frontmatter exist."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Space source directory does not exist: {source_dir}")

    missing = [name for name in REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing required Space file(s) in {source_dir}: {', '.join(missing)}")

    frontmatter = parse_frontmatter((source_dir / "README.md").read_text(encoding="utf-8"))
    missing_keys = [key for key in REQUIRED_FRONTMATTER_KEYS if key not in frontmatter]
    if missing_keys:
        raise ValueError(f"Space README.md frontmatter missing key(s): {', '.join(missing_keys)}")
    if frontmatter.get("sdk") != "static":
        raise ValueError(f"Expected Space sdk='static', found {frontmatter.get('sdk')!r}")
    if not str(frontmatter.get("title", "")).strip():
        raise ValueError("Space README.md frontmatter 'title' must be non-empty")
    short_desc = str(frontmatter.get("short_description", "")).strip()
    if not short_desc:
        raise ValueError("Space README.md frontmatter 'short_description' must be non-empty")
    if len(short_desc) > MAX_SHORT_DESCRIPTION_LEN:
        raise ValueError(
            f"Space README.md frontmatter 'short_description' exceeds {MAX_SHORT_DESCRIPTION_LEN} chars ({len(short_desc)})"
        )
    for color_key in ("colorFrom", "colorTo"):
        color = frontmatter.get(color_key)
        if color is not None and color not in ALLOWED_COLORS:
            raise ValueError(f"Space README.md frontmatter '{color_key}' must be one of {sorted(ALLOWED_COLORS)}, got {color!r}")
    return frontmatter


def _is_ignored(rel_posix: str) -> bool:
    parts = Path(rel_posix).parts
    if "__pycache__" in parts:
        return True
    name = parts[-1]
    if name in {"deploy.py", ".DS_Store"}:
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    return False


def collect_upload_files(source_dir: Path) -> list[dict[str, Any]]:
    """Return sorted metadata for static Space files eligible for upload."""
    source_dir = Path(source_dir).resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir).as_posix()
        if _is_ignored(rel):
            continue
        data = path.read_bytes()
        files.append(
            {
                "path": rel,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return files


def deploy_space(
    source_dir: Path | None = None,
    space_id: str = DEFAULT_SPACE_ID,
    commit_message: str = DEFAULT_COMMIT_MESSAGE,
    *,
    dry_run: bool = False,
    api: Any = None,
) -> dict[str, Any]:
    """Validate and deploy ``source_dir`` to the Hugging Face Static Space."""
    resolved_dir = Path(source_dir).resolve() if source_dir is not None else default_source_dir()
    frontmatter = validate_space_dir(resolved_dir)
    files = collect_upload_files(resolved_dir)
    total_bytes = sum(item["bytes"] for item in files)

    result: dict[str, Any] = {
        "space_id": space_id,
        "space_url": f"https://huggingface.co/spaces/{space_id}",
        "source_dir": str(resolved_dir),
        "dry_run": dry_run,
        "commit_message": commit_message,
        "frontmatter": frontmatter,
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "ignore_patterns": list(IGNORE_PATTERNS),
        "delete_patterns": list(DELETE_PATTERNS),
    }

    if dry_run:
        return result

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    commit_info = api.upload_folder(
        folder_path=str(resolved_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message=commit_message,
        ignore_patterns=list(IGNORE_PATTERNS),
        delete_patterns=list(DELETE_PATTERNS),
    )
    result["commit_oid"] = getattr(commit_info, "oid", None)
    result["commit_url"] = getattr(commit_info, "commit_url", None)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID, help="Target Hugging Face Space repo ID")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Directory containing static Space files",
    )
    parser.add_argument("--commit-message", default=DEFAULT_COMMIT_MESSAGE, help="HF Space commit message")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and print plan without uploading")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print structured JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = deploy_space(
        source_dir=args.source_dir,
        space_id=args.space_id,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    mode = "DRY RUN" if report["dry_run"] else "DEPLOYED"
    print(f"[{mode}] Space: {report['space_id']} ({report['space_url']})")
    print(f"Source: {report['source_dir']} ({report['file_count']} files, {report['total_bytes']} bytes)")
    for item in report["files"]:
        print(f"  - {item['path']} ({item['bytes']} bytes, sha256={item['sha256'][:12]})")
    if not report["dry_run"]:
        print(f"Commit OID: {report.get('commit_oid') or 'unchanged'}")
        if report.get("commit_url"):
            print(f"Commit URL: {report['commit_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
