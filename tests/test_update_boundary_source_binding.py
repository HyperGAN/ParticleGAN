"""Update-boundary bindings retain every exact ancestor and original result."""
import hashlib
import json
from pathlib import Path

from copy import deepcopy
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_update_boundary_source_bindings_preserve_all_questions_and_original_receipts():
    for path in sorted((ROOT / "reports/forge/software-audits").glob("pr*-update-boundary-source-binding.json")):
        receipt = json.loads(path.read_text())
        assert receipt["schema"] == "pr_develop_integration_source_binding_v1"
        assert all(receipt[key] is False for key in ("qualification_results_changed",
            "scientific_training_launched", "automatic_cross_source_reuse", "qualification_input"))
        assert len(receipt["tasks"]) == 10
        assert hashlib.sha256((ROOT / receipt["source_member"]).read_bytes()).hexdigest() == receipt["current_source_sha256"]
        for row in receipt["tasks"]:
            relative = row["previous_task_path"]
            data = (ROOT / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["current_task_sha256"]
            current = json.loads(data)
            assert current["evaluation"]["integration_source_binding"]["qualification_input"] is False
            binding = current["evaluation"]["integration_source_binding"]
            previous_raw = subprocess.check_output(["git", "show", binding["previous_task_commit"] + ":" + relative], cwd=ROOT)
            assert hashlib.sha256(previous_raw).hexdigest() == binding["previous_task_sha256"]
            assert hashlib.sha1(f"blob {len(previous_raw)}\0".encode() + previous_raw).hexdigest() == binding["previous_task_blob"]
            previous = json.loads(previous_raw)
            restored = deepcopy(current)
            restored["evaluation"]["sources"] = previous["evaluation"]["sources"]
            restored["evaluation"]["integration_source_binding"] = previous["evaluation"]["integration_source_binding"]
            assert restored == previous
            assert current["evaluation"]["sources"] == {**previous["evaluation"]["sources"],
                receipt["source_member"]: receipt["current_source_sha256"]}
