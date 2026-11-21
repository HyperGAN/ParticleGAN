"""Policies of the test suite itself: GPU opt-in, CPU thread budget and CI scope."""
from pathlib import Path
import ast
import os
import re
import subprocess
import sys
import tomllib

import pytest
import torch
import yaml

from tests.conftest import thread_budget

ROOT = Path(__file__).resolve().parents[1]
CUDA_SKIP = re.compile(r"not\s+torch\.cuda\.is_available\(\)|RUN_CUDA_IMAGE_TESTS")
CUDA_MARK = "pytest.mark.cuda"


def _segment(lines, node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1:node.end_lineno])


def unmarked_cuda_skip_sites(source):
    """Names of tests (or helpers) that skip without CUDA but carry no cuda marker.

    Checked per skip site, not per file: a file with one marked CUDA test and a
    second unmarked one is reported. A module ``pytestmark`` with the marker
    covers the whole file. Module-level skip helpers (a ``skipif`` mark bound to
    a name, or a fixture/helper function that calls ``pytest.skip``) are
    followed to every test that references them.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "pytestmark" for t in node.targets):
            if CUDA_MARK in _segment(lines, node):
                return []
    helpers = set()
    tests = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_marked = any(CUDA_MARK in ast.get_source_segment(source, d) for d in node.decorator_list)
            tests += [(f"{node.name}.{item.name}", item, class_marked) for item in node.body
                      if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test")]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            tests.append((node.name, node, False))
        elif isinstance(node, (ast.Assign, ast.FunctionDef, ast.AsyncFunctionDef)):
            text = _segment(lines, node)
            if CUDA_SKIP.search(text) and CUDA_MARK not in text:
                names = [node.name] if hasattr(node, "name") else [t.id for t in node.targets if isinstance(t, ast.Name)]
                helpers.update(names)
    unmarked = []
    for name, node, class_marked in tests:
        text = _segment(lines, node)
        uses = CUDA_SKIP.search(text) or any(re.search(rf"\b{re.escape(h)}\b", text) for h in helpers)
        if uses and not class_marked and CUDA_MARK not in text:
            unmarked.append(name)
    return unmarked


def test_every_cuda_only_test_declares_the_cuda_marker():
    """A CUDA skip without the marker runs silently on a workstation GPU."""
    unmarked = {}
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        sites = unmarked_cuda_skip_sites(path.read_text(encoding="utf-8"))
        if sites:
            unmarked[path.relative_to(ROOT).as_posix()] = sites
    assert not unmarked


@pytest.mark.parametrize("source,expected", [
    # One marked CUDA test does not cover a second, unmarked one in the same file.
    ("import pytest, torch\n@pytest.mark.cuda\n@pytest.mark.skipif(not torch.cuda.is_available(), reason='')\n"
     "def test_a(): pass\n@pytest.mark.skipif(not torch.cuda.is_available(), reason='')\ndef test_b(): pass\n",
     ["test_b"]),
    ("import pytest, torch\ndef test_body():\n    if not torch.cuda.is_available():\n        pytest.skip('')\n",
     ["test_body"]),
    ("import pytest, torch\nCUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='')\n"
     "@pytest.mark.cuda\n@CUDA\ndef test_a(): pass\n@CUDA\ndef test_b(): pass\n", ["test_b"]),
    ("import pytest, torch\n@pytest.fixture\ndef gpu():\n    if not torch.cuda.is_available():\n        pytest.skip('')\n"
     "def test_a(gpu): pass\n@pytest.mark.cuda\ndef test_b(gpu): pass\n", ["test_a"]),
    ("import pytest, torch\n@pytest.mark.cuda\nclass TestGpu:\n"
     "    @pytest.mark.skipif(not torch.cuda.is_available(), reason='')\n    def test_a(self): pass\n", []),
    ("import pytest, torch\npytestmark = [pytest.mark.cuda]\n"
     "@pytest.mark.skipif(not torch.cuda.is_available(), reason='')\ndef test_a(): pass\n", []),
])
def test_cuda_marker_policy_is_checked_per_skip_site(source, expected):
    assert unmarked_cuda_skip_sites(source) == expected


def test_cuda_marker_is_declared_and_deselected_by_default():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert any(marker.startswith("cuda:") for marker in config["markers"])
    assert "-m 'not cuda'" in config["addopts"]


def test_default_collection_excludes_cuda_tests_and_opt_in_selects_them(tmp_path):
    marked = ROOT / "tests/test_e22_noise_floor.py"
    common = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(marked)]
    default = subprocess.run(common, cwd=ROOT, capture_output=True, text=True, timeout=300)
    opted = subprocess.run(common + ["-m", "cuda"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert default.returncode == 0 and opted.returncode == 0, default.stdout + opted.stdout
    default_ids = [line for line in default.stdout.splitlines() if "::" in line]
    opted_ids = [line for line in opted.stdout.splitlines() if "::" in line]
    assert default_ids and opted_ids
    assert not any("cuda" in line for line in default_ids)
    assert all("cuda" in line for line in opted_ids)


@pytest.mark.parametrize("environ,expected", [
    ({}, None), ({"OMP_NUM_THREADS": "3"}, 3), ({"MKL_NUM_THREADS": "2"}, 2),
    ({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "2"}, 2), ({"OMP_NUM_THREADS": "0"}, None),
    ({"OMP_NUM_THREADS": "many"}, None),
])
def test_thread_budget_respects_omp_and_mkl(environ, expected):
    assert thread_budget(environ) == expected


def test_session_pins_torch_threads_to_the_declared_budget(_pinned_torch_threads):
    budget = thread_budget()
    if budget is not None:
        assert _pinned_torch_threads == budget
    assert torch.get_num_threads() == _pinned_torch_threads


def test_a_test_that_changes_threads_does_not_leak_into_the_next_one(tmp_path):
    """Run the production fixtures on a fixed two-test order in a child session.

    Splitting this into two tests of this module would depend on run order
    (random ordering or xdist would let the second pass trivially).
    """
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import _pinned_torch_threads, _restore_torch_threads  # noqa: F401\n")
    (tmp_path / "test_leak.py").write_text(
        "import torch\n"
        "def test_1_changes_threads(_pinned_torch_threads):\n"
        "    torch.set_num_threads(_pinned_torch_threads + 3)\n"
        "    assert torch.get_num_threads() == _pinned_torch_threads + 3\n"
        "def test_2_sees_the_pinned_count(_pinned_torch_threads):\n"
        "    assert torch.get_num_threads() == _pinned_torch_threads\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:xdist",
         "--rootdir", str(tmp_path), str(tmp_path / "test_leak.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and "2 passed" in run.stdout, run.stdout + run.stderr


def test_thread_restore_survives_a_test_that_stubs_the_torch_module(monkeypatch):
    """Teardown must not look torch up in ``sys.modules`` (tests stub it there)."""
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))


def test_ci_runs_one_complete_regression_suite_on_minimum_python():
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    job = workflow["jobs"]["cpu-tests"]
    matrix = job["strategy"]["matrix"]
    assert matrix == {"shard": list(range(8))}
    python = [step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python")]
    assert len(python) == 1 and python[0]["with"]["python-version"] == "3.11"
    runs = [step for step in job["steps"] if "scripts/ci_pytest.py run" in step.get("run", "")]
    assert len(runs) == 1 and "if" not in runs[0]
    assert "--index ${{ matrix.shard }} --count 8" in runs[0]["run"]
    receipt = "/tmp/pytest-shard-${{ matrix.shard }}.json"
    assert f"--receipt {receipt}" in runs[0]["run"]
    uploads = [step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact")]
    assert len(uploads) == 1 and uploads[0]["if"] == "always()"
    assert uploads[0]["with"]["path"] == receipt
    distributions = workflow["jobs"]["distributions"]
    assert "cpu-tests" in distributions["needs"]
    checks = [step["run"] for step in distributions["steps"] if "scripts/ci_pytest.py check" in step.get("run", "")]
    assert checks == ["python scripts/ci_pytest.py check /tmp/pytest-shards/pytest-shard-*.json"]
