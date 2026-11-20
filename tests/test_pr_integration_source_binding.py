"""Integrated future sources do not rewrite or qualify archived research."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from pr_integration_compat import COMBINED_INTEGRATION, before_pr_integration

ROOT = Path(__file__).resolve().parents[1]
MISGAN_V2_SOURCE = "ea75964cd20ed87e6f7a798e694205594b7e65a9"
COMBINED_RECEIPT = "reports/forge/reviewed-pr-integration-20261109/source-binding.json"


def test_every_integration_binding_preserves_the_exact_parent_contract():
    receipts = list((ROOT / "reports/forge").glob("pr-integration-202611*/pr-*.json"))
    assert receipts
    for path in receipts:
        receipt = json.loads(path.read_text())
        for row in receipt["tasks"]:
            relative = row["previous_task_path"]
            current = json.loads((ROOT / relative).read_text())
            assert before_pr_integration(ROOT, current, relative) != current


def test_misgan_v2_keeps_coupled_evidence_separate_and_unmeasured():
    receipt_path = "reports/forge/pr-integration-20261109/misgan-execution-v2.json"
    raw = (ROOT / receipt_path).read_bytes()
    assert raw == subprocess.check_output(
        ["git", "show", MISGAN_V2_SOURCE + ":" + receipt_path], cwd=ROOT)
    receipt = json.loads(raw)
    assert receipt["qualification_input"] is False
    assert receipt["training_rerun"] is False
    assert receipt["historical_quality_inherited"] is False
    assert receipt["corrected_execution_quality"] == "unmeasured"
    for record in [*receipt["coupled_v1"].values(), *receipt["unchanged_research_results"]]:
        data = subprocess.check_output([
            "git", "show", record["source_commit"] + ":" + record["source_path"]], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == record["git_blob"]
        if "archived_path" in record:
            assert (ROOT / record["archived_path"]).read_bytes() == data
        elif record in receipt["unchanged_research_results"]:
            assert (ROOT / record["source_path"]).read_bytes() == data
    archived = receipt["independent_v2"]
    originals = {}
    for member in ("provider", "task"):
        data = subprocess.check_output([
            "git", "show", MISGAN_V2_SOURCE + ":" + archived[member + "_path"]], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == archived[member + "_sha256"]
        originals[member] = data
    original_task = json.loads(originals["task"])
    task = json.loads((ROOT / archived["task_path"]).read_text())
    assert task["evaluation"]["integration_source_binding"]["id"] == COMBINED_INTEGRATION
    assert before_pr_integration(ROOT, task, archived["task_path"]) != task
    assert task["execution"] == original_task["execution"]
    assert {k: v for k, v in task["evaluation"].items()
            if k not in {"sources", "integration_source_binding"}} == {
                k: v for k, v in original_task["evaluation"].items() if k != "sources"}
    sources = task["evaluation"]["sources"]
    assert archived["provider_path"] in sources
    for path, digest in sources.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert task["execution"]["execution_contract"] == receipt["version"]
    assert task["execution"]["host_definition"]["execution_contract"] == receipt["version"]
    view = json.loads((ROOT / "configs/forge/views/misgan-bispf-research-v1.json").read_text())
    assert view["revision"] == 2
    assert "v2 quality is unmeasured" in view["policy_change_reason"]


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


def _write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def combined_binding(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "binding-test@example.invalid")
    _git(tmp_path, "config", "user.name", "Binding test")
    (tmp_path / "provider.py").write_text("original\n")
    original = {"id": "toy", "execution": {"steps": 2}, "evaluation": {
        "sources": {"provider.py": hashlib.sha256((tmp_path / "provider.py").read_bytes()).hexdigest()},
        "evaluator_revision": {"id": "original-revision"},
        "integration_source_binding": {"id": "original-binding"}}}
    _write(tmp_path, "task.json", original)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "original")

    def layer(text, execution=None):
        previous_raw = (tmp_path / "task.json").read_bytes()
        previous = json.loads(previous_raw)
        previous_commit = _git(tmp_path, "rev-parse", "HEAD")
        current = deepcopy(previous)
        (tmp_path / "provider.py").write_text(text)
        binding = dict(id=COMBINED_INTEGRATION, previous_task_commit=previous_commit,
            previous_task_path="task.json", previous_task_sha256=hashlib.sha256(previous_raw).hexdigest(),
            previous_task_blob=_git(tmp_path, "rev-parse", previous_commit + ":task.json"),
            integration_receipt=COMBINED_RECEIPT)
        current["evaluation"]["integration_source_binding"] = binding
        current["evaluation"]["sources"] = {
            "provider.py": hashlib.sha256((tmp_path / "provider.py").read_bytes()).hexdigest()}
        if execution is not None:
            current["execution"] = execution
        _write(tmp_path, "task.json", current)
        row = {key: binding[key] for key in (
            "previous_task_commit", "previous_task_path", "previous_task_sha256", "previous_task_blob")}
        row.update(current_task_sha256=hashlib.sha256((tmp_path / "task.json").read_bytes()).hexdigest(),
            sources_before=previous["evaluation"]["sources"], sources_after=current["evaluation"]["sources"])
        if execution is not None:
            row.update(execution_before=deepcopy(previous["execution"]), execution_after=deepcopy(current["execution"]))
        receipt = dict(schema="pr_combined_integration_source_binding_v1", pr="combined", tasks=[row],
            training_rerun=False, qualification_results_changed=False,
            automatic_cross_source_reuse=False, qualification_input=False)
        _write(tmp_path, COMBINED_RECEIPT, receipt)
        return current, receipt

    return tmp_path, original, layer


def test_combined_layers_restore_both_ancestral_bindings_and_use_exact_inner_source(combined_binding):
    root, original, layer = combined_binding
    first, _ = layer("first\n", {"steps": 2, "observation_checkpoints": True})
    assert before_pr_integration(root, first, "task.json") == original
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "first integration")
    second, _ = layer("second\n")
    # Both the source and receipt path now differ on disk from the inner layer.
    assert before_pr_integration(root, second, "task.json") == original


@pytest.mark.parametrize("damage", ["source", "parent_blob", "execution", "previous_revision", "qualification"])
def test_combined_binding_rejects_wrong_sources_contracts_and_quality_inheritance(combined_binding, damage):
    root, _, layer = combined_binding
    current, receipt = layer("current\n", {"steps": 2, "observation_checkpoints": True})
    row = receipt["tasks"][0]
    if damage == "source":
        (root / "provider.py").write_text("damaged\n")
    elif damage == "parent_blob":
        row["previous_task_blob"] = "0" * 40
    elif damage == "execution":
        current["execution"]["steps"] = 3
    elif damage == "previous_revision":
        current["evaluation"]["evaluator_revision"] = {"id": "changed"}
    else:
        receipt["qualification_input"] = True
    _write(root, "task.json", current)
    row["current_task_sha256"] = hashlib.sha256((root / "task.json").read_bytes()).hexdigest()
    _write(root, COMBINED_RECEIPT, receipt)
    with pytest.raises(AssertionError):
        before_pr_integration(root, current, "task.json")
