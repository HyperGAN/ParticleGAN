"""Run every collected test once across deterministic shards."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def check_receipts(paths):
    records = [json.loads(Path(path).read_text()) for path in paths]
    if not records:
        raise ValueError("No shard receipts")
    count = records[0]["shard_count"]
    if (len(records) != count or {r["shard_index"] for r in records} != set(range(count))
            or any(r["shard_count"] != count for r in records)):
        raise ValueError("Missing or duplicate shards")
    universes = {(r["collection_count"], r["collection_sha256"]) for r in records}
    if len(universes) != 1:
        raise ValueError("Shards collected different suites")
    selected = Counter(node for record in records for node in record["selected_nodeids"])
    expected = records[0]["collection_count"]
    if not expected or len(selected) != expected or any(n != 1 for n in selected.values()):
        raise ValueError("Collected tests were omitted or assigned more than once")
    digest = hashlib.sha256(json.dumps(sorted(selected)).encode()).hexdigest()
    if digest != records[0]["collection_sha256"]:
        raise ValueError("Shard union differs from collected suite")
    print(f"Verified {expected} collected tests assigned exactly once across {count} shards")


def run_shard(index, count, receipt, args, split_files=()):
    import pytest

    class Selection:
        collected = selected = 0

        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            all_ids = sorted(item.nodeid for item in items)
            keep, drop = [], []
            for item in items:
                filename = item.path.relative_to(config.rootpath).as_posix()
                key = item.name if filename in split_files else filename
                owner = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big") % count
                (keep if owner == index else drop).append(item)
            self.collected, self.selected = len(items), len(keep)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                "shard_index": index, "shard_count": count,
                "collection_count": len(all_ids),
                "collection_sha256": hashlib.sha256(json.dumps(all_ids).encode()).hexdigest(),
                "selected_nodeids": sorted(item.nodeid for item in keep),
            }, indent=2) + "\n")
            items[:] = keep
            config.hook.pytest_deselected(items=drop)

    plugin = Selection()
    result = pytest.main(args, plugins=[plugin])
    # An empty shard of a nonempty suite is valid; an empty suite is not.
    if result == pytest.ExitCode.NO_TESTS_COLLECTED and plugin.collected and not plugin.selected:
        return 0
    return int(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--count", type=int, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--split-file", action="append", default=[], metavar="PATH",
                     help="Shard this repository-relative file by test name instead of grouping it")
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)
    check = modes.add_parser("check")
    check.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "check":
        check_receipts(args.receipts)
        return 0
    if args.count < 1 or not 0 <= args.index < args.count:
        parser.error("require count > 0 and 0 <= index < count")
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return run_shard(args.index, args.count, args.receipt, pytest_args, args.split_file)


if __name__ == "__main__":
    raise SystemExit(main())
