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
    paths = set()
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if (path.is_file() and path.suffix in SOURCE_SUFFIXES
                    and not any(p in {"__pycache__", ".venv", "runs"} for p in path.relative_to(root).parts)):
                rel_posix = path.relative_to(root).as_posix()
                if rel_posix in NON_SCIENTIFIC_SOURCE_PATHS:
                    continue
                paths.add(path)
    # Existing task hosts sometimes read repo configs by relative path.
    for path in (root / "configs").rglob("*"):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and "forge" not in path.relative_to(root).parts:
            paths.add(path)
    for relative in extra_paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"declared source must be a file inside the worktree: {relative}")
        paths.add(path)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source symlinks are not snapshot inputs: {path}")
    return sorted(paths)


def inspect_source(root: Path, extra_paths=()) -> dict:
    root = Path(root).resolve()
    files = {str(p.relative_to(root)): file_hash(p) for p in source_files(root, extra_paths)}
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
    manifest = manifest or read_json(path / "forge-source.json")
    if stable_hash(manifest["files"]) != manifest["digest"]:
        raise ValueError("invalid source manifest digest")
    actual = {str(member.relative_to(path)) for member in path.rglob("*")
              if member.is_file() and member.suffix in SOURCE_SUFFIXES
              and "__pycache__" not in member.parts and member.name != "forge-source.json"}
    unexpected = actual - set(manifest["files"])
    if unexpected:
        raise ValueError(f"undeclared files appeared in source snapshot: {sorted(unexpected)}")
    for relative, digest in manifest["files"].items():
        member = path / relative
        if not member.resolve().is_relative_to(path.resolve()) or file_hash(member) != digest:
            raise ValueError(f"source snapshot was changed: {relative}")
