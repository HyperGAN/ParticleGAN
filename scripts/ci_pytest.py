"""Run every collected test once across deterministic shards."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


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


def _load_pins(pin_lists, count):
    pinned_files = {}
    pinned_nodes = {}
    pinned_shards = set()
    for spec in pin_lists:
        shard_text, sep, list_path = spec.partition("=")
        if not sep or not shard_text.isdigit():
            raise ValueError(f"Invalid --pin-list specification (expected SHARD=PATH): {spec}")
        shard = int(shard_text)
        if not 0 <= shard < count:
            raise ValueError(f"Pinned shard {shard} out of range for count={count}")
        pinned_shards.add(shard)
        for raw in Path(list_path).read_text().splitlines():
            entry = raw.split("#", 1)[0].strip()
            if not entry:
                continue
            target = pinned_nodes if "::" in entry else pinned_files
            if entry in target and target[entry] != shard:
                raise ValueError(f"Conflicting shard pins for {entry}")
            target[entry] = shard
    return pinned_files, pinned_nodes, pinned_shards


def _resolve_workers(spec: str | int | None) -> int:
    if spec is None:
        return 1
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        return max(1, min(os.cpu_count() or 1, 4))
    value = int(spec)
    if value < 1:
        raise ValueError("--workers must be >= 1 or 'auto'")
    return value


def _owner_for_item(item, rootpath, count, split_files, pinned_files, pinned_nodes, unpinned_shards):
    filename = item.path.relative_to(rootpath).as_posix()
    base_node = f"{filename}::{getattr(item, 'originalname', item.name)}"
    if item.nodeid in pinned_nodes:
        return pinned_nodes[item.nodeid], filename
    if base_node in pinned_nodes:
        return pinned_nodes[base_node], filename
    if filename in pinned_files:
        return pinned_files[filename], filename
    key = item.name if filename in split_files else filename
    digest_int = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
    if unpinned_shards is not None:
        return unpinned_shards[digest_int % len(unpinned_shards)], filename
    return digest_int % count, filename


def _partition_worker_buckets(selected_items: list[tuple[str, str]], split_files: set[str], workers: int) -> list[list[str]]:
    """Group tests into atomic units (file-level unless in split_files) and balance across workers."""
    by_unit: dict[str, list[str]] = {}
    for nodeid, filename in selected_items:
        unit_key = nodeid if filename in split_files else filename
        by_unit.setdefault(unit_key, []).append(nodeid)

    units = list(by_unit.items())
    # Largest units first; tie-break by hash so split tests from the same heavy file
    # spread evenly across distinct workers rather than clustering sequentially.
    units.sort(key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()))

    eff_workers = min(workers, len(units))
    buckets: list[list[str]] = [[] for _ in range(eff_workers)]
    loads = [0] * eff_workers
    for _, nodeids in units:
        target = min(range(eff_workers), key=lambda idx: (loads[idx], idx))
        buckets[target].extend(nodeids)
        # Weight single-node split tests slightly higher than 1 so heavy split items spread evenly.
        loads[target] += max(len(nodeids), 3 if len(nodeids) == 1 else len(nodeids))
    return [bucket for bucket in buckets if bucket]


def _has_positional_path_args(pytest_args: list[str]) -> bool:
    """Return True if pytest_args already specifies an explicit test path or nodeid."""
    takes_value = {"-k", "-m", "-o", "-p", "-c", "-W", "--rootdir", "--ignore", "--ignore-glob",
                   "--deselect", "--confcutdir", "--basetemp", "--durations", "--durations-min",
                   "--tb", "--maxfail", "--junitxml", "--override-ini", "--log-level"}
    skip_next = False
    for arg in pytest_args:
        if skip_next:
            skip_next = False
            continue
        if arg in takes_value:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return True
    return False


def _run_worker(nodes_file: Path, args: list[str]) -> int:
    import pytest

    wanted_list = json.loads(Path(nodes_file).read_text())
    wanted = set(wanted_list)

    class WorkerSelection:
        matched = 0

        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            keep = [item for item in items if item.nodeid in wanted]
            drop = [item for item in items if item.nodeid not in wanted]
            self.matched = len(keep)
            items[:] = keep
            if drop:
                config.hook.pytest_deselected(items=drop)

    plugin = WorkerSelection()
    rc = int(pytest.main(args, plugins=[plugin]))
    if rc == 0 and plugin.matched != len(wanted):
        print(
            f"ERROR: worker expected {len(wanted)} nodeids but matched {plugin.matched}",
            file=sys.stderr,
        )
        return 2
    return rc


def run_shard(index, count, receipt, args, split_files=(), pin_lists=(), *, workers=1, exclusive_pin_shards=False):
    import pytest

    pinned_files, pinned_nodes, pinned_shards = _load_pins(pin_lists, count)
    split_set = set(split_files)
    unpinned_shards = None
    if exclusive_pin_shards and pinned_shards:
        unpinned_shards = [s for s in range(count) if s not in pinned_shards]
        if not unpinned_shards:
            raise ValueError("--exclusive-pin-shards leaves no shard available for unpinned tests")

    worker_count = _resolve_workers(workers)

    class Selection:
        collected = selected = 0
        selected_items: list[tuple[str, str]] = []

        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            all_ids = sorted(item.nodeid for item in items)
            keep, drop = [], []
            selected_pairs = []
            for item in items:
                owner, filename = _owner_for_item(
                    item, config.rootpath, count, split_set, pinned_files, pinned_nodes, unpinned_shards
                )
                if owner == index:
                    keep.append(item)
                    selected_pairs.append((item.nodeid, filename))
                else:
                    drop.append(item)
            self.collected, self.selected = len(items), len(keep)
            self.selected_items = selected_pairs
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
    if worker_count == 1:
        result = pytest.main(args, plugins=[plugin])
        # An empty shard of a nonempty suite is valid; an empty suite is not.
        if result == pytest.ExitCode.NO_TESTS_COLLECTED and plugin.collected and not plugin.selected:
            return 0
        return int(result)

    # Parallel worker path: collect once in-process to write the authoritative receipt,
    # then partition this shard's exact nodeids across worker subprocesses.
    collect_args = ["--collect-only", "-q", "-p", "no:cacheprovider", *args]
    collect_rc = int(pytest.main(collect_args, plugins=[plugin]))
    if collect_rc == pytest.ExitCode.NO_TESTS_COLLECTED and plugin.collected and not plugin.selected:
        return 0
    if collect_rc != 0:
        return collect_rc
    if not plugin.selected_items:
        return 0

    buckets = _partition_worker_buckets(plugin.selected_items, split_set, worker_count)
    # Sanity-check exact disjoint partition of this shard's selected nodeids.
    assigned = [nid for bucket in buckets for nid in bucket]
    if sorted(assigned) != sorted(nid for nid, _ in plugin.selected_items):
        raise RuntimeError("Internal error: worker bucket partition mismatch")

    if len(buckets) == 1:
        single_plugin = Selection()
        result = pytest.main(args, plugins=[single_plugin])
        if result == pytest.ExitCode.NO_TESTS_COLLECTED and single_plugin.collected and not single_plugin.selected:
            return 0
        return int(result)

    has_paths = _has_positional_path_args(list(args))
    script_path = str(Path(__file__).resolve())
    worker_env = {
        **os.environ,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "PYTHONUNBUFFERED": "1",
    }

    t_start = time.perf_counter()
    print(
        f"Shard {index}/{count}: running {len(assigned)} tests across {len(buckets)} parallel workers",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix=f"ci-pytest-shard-{index}-") as tmp_dir:
        tmp_root = Path(tmp_dir)

        def _exec_bucket(worker_idx: int, bucket_nodes: list[str]):
            nodes_path = tmp_root / f"worker-{worker_idx}.json"
            nodes_path.write_text(json.dumps(bucket_nodes))
            cmd = [
                sys.executable, script_path, "worker",
                "--nodes-file", str(nodes_path),
                "--", "-p", "no:cacheprovider", *args,
            ]
            if not has_paths:
                bucket_files = sorted({nid.split("::", 1)[0] for nid in bucket_nodes})
                cmd.extend(bucket_files)
            wt0 = time.perf_counter()
            proc = subprocess.run(cmd, env=worker_env, capture_output=True, text=True)
            return worker_idx, len(bucket_nodes), time.perf_counter() - wt0, proc.returncode, proc.stdout, proc.stderr

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(buckets)) as pool:
            futures = [pool.submit(_exec_bucket, idx, bucket) for idx, bucket in enumerate(buckets)]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                w_idx, n_tests, dt, rc, out, err = res
                print(
                    f"\n=== shard {index} worker {w_idx + 1}/{len(buckets)} "
                    f"({n_tests} tests, {dt:.1f}s, exit={rc}) ===",
                    flush=True,
                )
                if out:
                    print(out, end="" if out.endswith("\n") else "\n", flush=True)
                if err:
                    print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr, flush=True)

    total_dt = time.perf_counter() - t_start
    failed = [res for res in results if res[3] != 0]
    print(
        f"\nShard {index}/{count} finished in {total_dt:.1f}s "
        f"({len(buckets) - len(failed)}/{len(buckets)} workers passed)",
        flush=True,
    )
    if not failed:
        return 0
    codes = [res[3] for res in failed]
    return 1 if 1 in codes else codes[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--count", type=int, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--workers", default="1",
                     help="Parallel worker subprocesses inside this shard (integer >= 1 or 'auto')")
    run.add_argument("--exclusive-pin-shards", action="store_true",
                     help="Route unpinned tests only to shards not targeted by --pin-list")
    run.add_argument("--split-file", action="append", default=[], metavar="PATH",
                     help="Shard this repository-relative file by test name instead of grouping it")
    run.add_argument("--pin-list", action="append", default=[], metavar="SHARD=PATH",
                     help="Pin test files or file::function entries listed in PATH to SHARD")
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)

    worker = modes.add_parser("worker")
    worker.add_argument("--nodes-file", type=Path, required=True)
    worker.add_argument("pytest_args", nargs=argparse.REMAINDER)

    check = modes.add_parser("check")
    check.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "check":
        check_receipts(args.receipts)
        return 0
    if args.mode == "worker":
        pytest_args = args.pytest_args
        if pytest_args[:1] == ["--"]:
            pytest_args = pytest_args[1:]
        return _run_worker(args.nodes_file, pytest_args)
    if args.count < 1 or not 0 <= args.index < args.count:
        parser.error("require count > 0 and 0 <= index < count")
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return run_shard(
        args.index,
        args.count,
        args.receipt,
        pytest_args,
        args.split_file,
        args.pin_list,
        workers=args.workers,
        exclusive_pin_shards=args.exclusive_pin_shards,
    )


if __name__ == "__main__":
    raise SystemExit(main())
