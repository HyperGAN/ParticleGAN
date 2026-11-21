"""Check actual collection, execution, coverage receipts and failure exit codes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts/ci_pytest.py"


def test_shards_execute_complete_suite_once_and_reject_incomplete_receipts(tmp_path):
    marker = tmp_path / "executed.jsonl"
    for name in ("one", "two", "three"):
        (tmp_path / f"test_{name}.py").write_text('''import json, os
from pathlib import Path
import pytest
@pytest.mark.parametrize("value", [0, 1])
def test_case(value, request):
    with Path(os.environ["CI_SHARD_EXECUTED"]).open("a") as stream:
        stream.write(json.dumps(request.node.nodeid) + "\\n")
    assert value >= 0
''')
    receipts = []
    for index in range(4):
        receipt = tmp_path / f"receipt-{index}.json"
        result = subprocess.run([
            sys.executable, str(RUNNER), "run", "--index", str(index),
            "--count", "4", "--receipt", str(receipt), "--split-file", "test_one.py",
            "--", "-q", str(tmp_path),
        ], cwd=tmp_path, env={**os.environ, "CI_SHARD_EXECUTED": str(marker)},
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        receipts.append(receipt)
    spec = importlib.util.spec_from_file_location("ci_shard_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_receipts(receipts)
    owners = {node: index for index, receipt in enumerate(receipts)
              for node in json.loads(receipt.read_text())["selected_nodeids"]}
    assert owners["test_one.py::test_case[0]"] != owners["test_one.py::test_case[1]"]
    for name in ("two", "three"):
        assert owners[f"test_{name}.py::test_case[0]"] == owners[f"test_{name}.py::test_case[1]"]
    executed = [json.loads(line) for line in marker.read_text().splitlines()]
    assert len(executed) == len(set(executed)) == 6
    with pytest.raises(ValueError, match="Missing or duplicate"):
        module.check_receipts(receipts[:-1])
    record = json.loads(receipts[0].read_text())
    record["selected_nodeids"].append("invented::test_case")
    receipts[0].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="omitted or assigned|differs"):
        module.check_receipts(receipts)


def test_shard_propagates_real_test_failure(tmp_path):
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert False\n")
    result = subprocess.run([
        sys.executable, str(RUNNER), "run", "--index", "0", "--count", "1",
        "--receipt", str(tmp_path / "receipt.json"), "--", "-q", str(tmp_path),
    ], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "1 failed" in result.stdout
