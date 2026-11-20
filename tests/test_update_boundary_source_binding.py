"""Update-boundary bindings retain every exact ancestor and original result."""
import hashlib
import json
from pathlib import Path

from copy import deepcopy
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _show(commit, relative):
    return subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT)


def _layer(relative, revision_id):
    """Return (bytes, commit holding its sources or None for the worktree) for one stacked layer."""
    data, source_commit = (ROOT / relative).read_bytes(), None
    while True:
        binding = json.loads(data)["evaluation"].get("integration_source_binding", {})
        if binding.get("id") == revision_id:
            return data, source_commit
        # Later source-only layers (e.g. pr-520 over pr-522) name their exact parent commit.
        assert binding.get("id", "").endswith("-audit-develop-integration-20261114"), (relative, revision_id)
        source_commit = binding["previous_task_commit"]
        data = _show(source_commit, relative)


def test_update_boundary_source_bindings_preserve_all_questions_and_original_receipts():
    for path in sorted((ROOT / "reports/forge/software-audits").glob("pr*-update-boundary-source-binding.json")):
        receipt = json.loads(path.read_text())
        assert receipt["schema"] == "pr_develop_integration_source_binding_v1"
        assert all(receipt[key] is False for key in ("qualification_results_changed",
            "scientific_training_launched", "automatic_cross_source_reuse", "qualification_input"))
        assert len(receipt["tasks"]) == 14
        for row in receipt["tasks"]:
            relative = row["previous_task_path"]
            data, source_commit = _layer(relative, receipt["revision_id"])
            member = ((ROOT / receipt["source_member"]).read_bytes() if source_commit is None
                      else _show(source_commit, receipt["source_member"]))
            assert hashlib.sha256(member).hexdigest() == receipt["current_source_sha256"]
            assert hashlib.sha256(data).hexdigest() == row["current_task_sha256"]
            current = json.loads(data)
            assert current["evaluation"]["integration_source_binding"]["qualification_input"] is False
            binding = current["evaluation"]["integration_source_binding"]
            previous_raw = _show(binding["previous_task_commit"], relative)
            assert hashlib.sha256(previous_raw).hexdigest() == binding["previous_task_sha256"]
            assert hashlib.sha1(f"blob {len(previous_raw)}\0".encode() + previous_raw).hexdigest() == binding["previous_task_blob"]
            previous = json.loads(previous_raw)
            restored = deepcopy(current)
            restored["evaluation"]["sources"] = previous["evaluation"]["sources"]
            restored["evaluation"]["integration_source_binding"] = previous["evaluation"]["integration_source_binding"]
            assert restored == previous
            assert current["evaluation"]["sources"] == {**previous["evaluation"]["sources"],
                receipt["source_member"]: receipt["current_source_sha256"]}
