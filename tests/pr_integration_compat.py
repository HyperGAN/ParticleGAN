"""Verify every PR integration layer against exact archived task/source bytes."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess


INTEGRATION = re.compile(r"pr-\d+-(?:develop-integration-20261108|audit-develop-integration-20261109)$")
COMBINED_INTEGRATION = "reviewed-model-prs-integration-20261109"


def _binding(task):
    evaluation = task.get("evaluation", {})
    combined = evaluation.get("integration_source_binding")
    if isinstance(combined, dict) and combined.get("id") == COMBINED_INTEGRATION:
        return "integration_source_binding", combined
    for field in ("evaluator_revision", "integration_source_binding"):
        value = evaluation.get(field)
        if isinstance(value, dict) and INTEGRATION.fullmatch(value.get("id", "")):
            return field, value
    return None, None


def before_pr_integration(root, current, relative):
    root = Path(root)
    raw = (root / relative).read_bytes()
    source_commit = None
    while True:
        field, binding = _binding(current)
        if binding is None:
            return current
        assert json.loads(raw) == current
        previous_raw = subprocess.check_output(
            ["git", "show", binding["previous_task_commit"] + ":" + relative], cwd=root)
        assert binding["previous_task_path"] == relative
        assert sha256(previous_raw).hexdigest() == binding["previous_task_sha256"]
        import hashlib
        assert hashlib.sha1(f"blob {len(previous_raw)}\0".encode() + previous_raw).hexdigest() == binding["previous_task_blob"]
        previous = json.loads(previous_raw)
        default = "reports/forge/pr-integration-20261108/" + binding["id"].split("-develop-")[0] + ".json"
        combined = binding["id"] == COMBINED_INTEGRATION
        receipt_path = binding.get("integration_receipt", default)
        if combined and source_commit is not None:
            receipt = json.loads(subprocess.check_output(
                ["git", "show", source_commit + ":" + receipt_path], cwd=root))
        else:
            receipt = json.loads((root / receipt_path).read_text())
        if combined:
            assert receipt["schema"] == "pr_combined_integration_source_binding_v1"
            assert receipt["pr"] == "combined"
        else:
            assert receipt["schema"] == "pr_develop_integration_source_binding_v1"
        assert all(receipt[k] is False for k in ("training_rerun", "qualification_results_changed", "automatic_cross_source_reuse", "qualification_input"))
        row = next(row for row in receipt["tasks"] if row["previous_task_path"] == relative)
        if combined:
            for key in ("previous_task_commit", "previous_task_path", "previous_task_sha256", "previous_task_blob"):
                assert row[key] == binding[key]
        assert sha256(raw).hexdigest() == row["current_task_sha256"]
        assert row["sources_before"] == previous["evaluation"]["sources"]
        assert row["sources_after"] == current["evaluation"]["sources"]
        for path, expected in row["sources_after"].items():
            data = ((root / path).read_bytes() if source_commit is None else
                    subprocess.check_output(["git", "show", source_commit + ":" + path], cwd=root))
            assert sha256(data).hexdigest() == expected
        restored = deepcopy(current)
        restored["evaluation"].pop(field)
        if combined:
            # This layer changes source pins and adds one binding. Earlier
            # revision metadata remains exact and both ancestral fields survive.
            assert ("evaluator_revision" in current["evaluation"]) == ("evaluator_revision" in previous["evaluation"])
            assert current["evaluation"].get("evaluator_revision") == previous["evaluation"].get("evaluator_revision")
            if "integration_source_binding" in previous["evaluation"]:
                restored["evaluation"]["integration_source_binding"] = previous["evaluation"]["integration_source_binding"]
            if "execution_before" in row or "execution_after" in row:
                assert row["execution_before"] == previous["execution"]
                assert row["execution_after"] == current["execution"]
                restored["execution"] = deepcopy(row["execution_before"])
        if "evaluator_revision" in previous["evaluation"]:
            restored["evaluation"]["evaluator_revision"] = previous["evaluation"]["evaluator_revision"]
        restored["evaluation"]["sources"] = previous["evaluation"]["sources"]
        for path, value in row.get("execution_updates", {}).items():
            parts = path.split(".")
            owner = restored["execution"]
            for part in parts[:-1]:
                owner = owner[part]
            assert owner.pop(parts[-1]) == value
        if field == "integration_source_binding" and receipt["pr"] == 461 and current["id"] in ("model_vae_hard", "model_vae_categorical"):
            assert restored["execution"].pop("transport_consumer") == "output_marginal_v1"
            assert restored["execution"].pop("observation_checkpoints") is True
        assert restored == previous
        current, raw = previous, previous_raw
        source_commit = binding["previous_task_commit"]


def source_before_pr_integration(root, relative, member):
    root = Path(root)
    task = json.loads((root / relative).read_text())
    if _binding(task)[1] is not None:
        task = before_pr_integration(root, task, relative)
        return task["evaluation"]["sources"][member]
    return sha256((root / member).read_bytes()).hexdigest()
