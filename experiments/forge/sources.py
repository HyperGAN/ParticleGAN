"""Content-addressed source snapshots, including worktree edits.

Like run_grid.code_provenance and the LR-free harness package_digest, identities
use bytes rather than a branch name. Unlike a mutable package directory, the
source used by a queued child is frozen and verified before every launch.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from .contracts import atomic_json, file_hash, read_json, stable_hash

SOURCE_DIRS = ("particlegan", "experiments", "benchmarks", "lib")
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".sh"}

# Pure post-run reporting, documentation, Hugging Face publishing, media export,
# and Colab orchestration modules inside ``experiments/forge/``. None of these
# files are imported by any trainer, task host, evaluator, planner, or worker
# (enforced by tests/test_forge_sources_isolation.py), so editing them must not
# invalidate scientific source digests or force redundant GPU reruns.
NON_SCIENTIFIC_SOURCE_PATHS = frozenset({
    "experiments/forge/experiment_docs.py",
    "experiments/forge/family_documentation.py",
    "experiments/forge/family_reports.py",
    "experiments/forge/hf_publish.py",
    "experiments/forge/leaderboard_media.py",
    "experiments/forge/policy_cloud_media.py",
    "experiments/forge/publication_storage.py",
    "experiments/forge/scaling_publication.py",
    "experiments/forge/tier1_media.py",
})


def runtime_manifest() -> dict:
    packages = {}
    for name in ("torch", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(), "packages": packages}


from functools import lru_cache


@lru_cache(maxsize=4)
def _cpu_model() -> str:
    cpu = platform.processor()
    path = Path("/proc/cpuinfo")
    if path.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in path.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return cpu


@lru_cache(maxsize=4)
def _gpu_models() -> frozenset[tuple[str, ...]] | None:
    try:
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                                        "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        return frozenset(tuple(v.strip() for v in line.split(",")) for line in rows.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def compute_profile(backend: str, cuda_model: str | None = None, *, threads: int = 1) -> dict:
    """Scientific hardware cohort, independent of physical GPU index."""
    if backend not in {"cpu", "cuda"}:
        raise ValueError("execution backend must be cpu or cuda")
    if type(threads) is not int or threads < 1:
        raise ValueError("compute threads must be a positive integer")
    result = {"backend": backend, "threads": threads, "deterministic": True, "tf32": False,
              "autograd_multithreading_enabled": False}
    if backend == "cpu":
        result.update(model=_cpu_model(), machine=platform.machine())
        return result
    queried = _gpu_models()
    if queried is None:
        return {**result, "model": cuda_model, "availability": "unavailable"}
    models = set(queried)
    if cuda_model:
        models = {row for row in models if row[0] == cuda_model}
    if len(models) != 1:
        raise ValueError("select one compatible CUDA model cohort using --cuda-model")
    model, driver, capability = next(iter(models))
    result.update(model=model, driver=driver, compute_capability=capability)
    return result


def source_files(root: Path, extra_paths=()) -> list[Path]:
    root = Path(root).resolve()
    root_str = str(root)
    root_prefix = root_str if root_str.endswith(os.sep) else root_str + os.sep
    excluded_dirs = {"__pycache__", ".venv", "runs"}
    paths: set[Path] = set()
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.exists():
            continue
        prefix_len = len(root_prefix)
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
            for name in filenames:
                _, ext = os.path.splitext(name)
                if ext not in SOURCE_SUFFIXES:
                    continue
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    raise ValueError(f"source symlinks are not snapshot inputs: {full}")
                rel_posix = full[prefix_len:].replace(os.sep, "/")
                if rel_posix in NON_SCIENTIFIC_SOURCE_PATHS:
                    continue
                paths.add(Path(full))
    # Existing task hosts sometimes read repo configs by relative path.
    configs_base = root / "configs"
    if configs_base.exists():
        for dirpath, dirnames, filenames in os.walk(configs_base, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != "forge"]
            for name in filenames:
                _, ext = os.path.splitext(name)
                if ext not in SOURCE_SUFFIXES:
                    continue
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    raise ValueError(f"source symlinks are not snapshot inputs: {full}")
                paths.add(Path(full))
    for relative in extra_paths:
        raw_path = root / relative
        if raw_path.is_symlink():
            raise ValueError(f"source symlinks are not snapshot inputs: {raw_path}")
        resolved = raw_path.resolve()
        resolved_str = str(resolved)
        if not (resolved_str == root_str or resolved_str.startswith(root_prefix)) or not resolved.is_file():
            raise ValueError(f"declared source must be a file inside the worktree: {relative}")
        paths.add(resolved)
    return sorted(paths)


def inspect_source(root: Path, extra_paths=()) -> dict:
    root = Path(root).resolve()
    root_str = str(root)
    prefix_len = len(root_str) + (0 if root_str.endswith(os.sep) else 1)
    files = {os.fspath(p)[prefix_len:].replace(os.sep, "/"): file_hash(p) for p in source_files(root, extra_paths)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"schema_version": 1, "digest": stable_hash(files), "files": files, "origin_commit": commit}


def snapshot_source(root: Path, queue_root: Path, manifest: dict) -> Path:
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    destination = queue_root / "snapshots" / manifest["digest"]
    if destination.exists():
        verify_snapshot(destination, manifest)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
    try:
        for relative, digest in manifest["files"].items():
            source = root / relative
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"source changed during submission: {relative}; plan and submit again")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        atomic_json(temporary / "forge-source.json", manifest)
        try:
            os.rename(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
            verify_snapshot(destination, manifest)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_snapshot(path: Path, manifest: dict | None = None) -> None:
    path = Path(path)
    resolved_root = str(path.resolve())
    root_prefix = resolved_root if resolved_root.endswith(os.sep) else resolved_root + os.sep
    prefix_len = len(root_prefix)
    manifest = manifest or read_json(path / "forge-source.json")
    manifest_files = manifest["files"]
    if stable_hash(manifest_files) != manifest["digest"]:
        raise ValueError("invalid source manifest digest")
    actual: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name == "forge-source.json":
                continue
            _, ext = os.path.splitext(name)
            if ext not in SOURCE_SUFFIXES:
                continue
            full = os.path.join(dirpath, name)
            rel = full[prefix_len:].replace(os.sep, "/")
            actual.add(rel)
    unexpected = actual - set(manifest_files)
    if unexpected:
        raise ValueError(f"undeclared files appeared in source snapshot: {sorted(unexpected)}")
    for relative, digest in manifest_files.items():
        member_path = os.path.join(resolved_root, relative)
        if os.path.islink(member_path) or ".." in relative.split("/"):
            resolved_member = os.path.realpath(member_path)
            if not resolved_member.startswith(root_prefix):
                raise ValueError(f"source snapshot was changed: {relative}")
        if not os.path.isfile(member_path) or file_hash(Path(member_path)) != digest:
            raise ValueError(f"source snapshot was changed: {relative}")
