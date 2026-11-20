"""No worker or paid experiment: test real Forge admission and freeze guards."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.forge.contracts import read_json
from experiments.forge.views import load_tasks

ROOT = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location("phase2_orchestration_test", ROOT / "reports/forge/bcap-three-phase/phase2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def software_checkout(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-source")
    # Exact committed source/config bytes, isolated from the shared worktrees.
    archive = subprocess.Popen(["git", "archive", "HEAD", "particlegan", "experiments", "benchmarks", "lib", "configs"],
                               cwd=ROOT, stdout=subprocess.PIPE)
    extracted = subprocess.run(["tar", "-x", "-C", str(root)], stdin=archive.stdout)
    archive.stdout.close()
    assert archive.wait() == 0 and extracted.returncode == 0
    extras = {helper().ORIGINALS, "reports/forge/bcap-develop-integration/results.json"}
    for task in load_tasks(ROOT).values():
        extras.update(name for name in task["evaluation"].get("sources", {}) if name.startswith("reports/"))
    for name in extras:
        if not (ROOT / name).is_file():
            continue
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    from tests.archived_forge_contracts import restore_archived_contracts
    # This newer original baseline already declares optional transport hooks;
    # it is a separate immutable cohort from the older inactive comparator.
    restore_archived_contracts(root, "bcap_prior_baseline_questions")
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Software fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-q", "-m", "Exact source fixture for admission only"], cwd=root, check=True)
    return root


def spec():
    arms = []
    for role, parent, control in (
        ("incumbent", "bcap-develop-integration-winner-v1", "software-phase2-repair"),
        ("repair", "bcap-develop-integration-combined-v1", "software-phase2-incumbent")):
        arms.append(dict(role=role, candidate_id=f"software-phase2-{role}", parent=parent, control=control,
            changed_factors=["Original winner versus declared direction-blend/global/local combination"],
            mechanism_rationale="Exercise current public global trainer mechanisms and fresh admission only",
            hypothesis="One whole global formulation can preserve all original Tier 1 numerical gates",
            competing_explanation="A passing endpoint can still fail the sustained gradient cap or word deadline",
            prior_evidence=[dict(path="reports/forge/bcap-develop-integration/results.json", selector=[],
                identity=dict(schema_version=1, protocol_seed=0), use="motivation_only")],
            prediction=dict(task_id="two_pole", metric="grad_med", op="<=", threshold=1., phase="final"),
            falsifier=dict(task_id="two_pole", metric="grad_med", op=">", threshold=1., phase="final")))
    return dict(campaign_id="software-phase2-admission", arms=arms)


def test_prepare_really_resolves_ready_without_worker_or_queue(software_checkout, tmp_path):
    module = helper()
    source, registration = tmp_path / "spec.json", tmp_path / "registration.json"
    source.write_text(json.dumps(spec()))
    module.prepare(argparse.Namespace(repository=software_checkout, spec=source, registration=registration))
    result = read_json(registration)
    assert result["campaign_budget_seconds"] == 5040
    assert result["candidate_budget_seconds"] == 2520
    assert result["required_tasks"] == module.TIMEOUTS
    requests = module.resolved(software_checkout, result)
    assert len(requests) == 2
    assert all(request["wallclock_policy"] == "declared" for request in requests.values())
    assert all(plan["admission"] == "READY" and len(plan["active_tasks"]) == 7 for plan in result["arms"].values())
    assert not (software_checkout / "reports/forge/attempts").exists()
    assert not (software_checkout / "queue").exists()
    with pytest.raises(ValueError, match="fresh"):
        module.prepare(argparse.Namespace(repository=software_checkout, spec=source, registration=registration))
    # Change an actual numerical gate in the isolated fixture: source-pin
    # bookkeeping cannot authorize it or quietly absorb a relaxed question.
    card = software_checkout / "configs/forge/tasks/two_pole.json"
    original = card.read_bytes()
    task = json.loads(original)
    task["evaluation"]["thresholds"][1][2] = 2.
    card.write_text(json.dumps(task))
    try:
        with pytest.raises(ValueError, match="gates changed"):
            module.resolved(software_checkout, result)
    finally:
        card.write_bytes(original)


def test_execution_requires_explicit_source_and_prior_admission(software_checkout, tmp_path):
    module = helper()
    with pytest.raises(ValueError, match="exact committed source"):
        module.run(argparse.Namespace(repository=software_checkout, artifacts=tmp_path / "bulk",
            source_commit="not-the-current-commit", registration=tmp_path / "absent.json"))
    source = {"files": {"particlegan/__init__.py": "not-consumed-here"}}
    with pytest.raises(ValueError, match="declarations"):
        module.committed_source(software_checkout, source, ["configs/forge/ideas/uncommitted-declaration.json"])


def test_publisher_requires_complete_registered_campaign(tmp_path):
    module = helper()
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    (tmp_path / "phase2-progress.json").write_text('{"phase":"running"}')
    with pytest.raises(ValueError, match="complete registered campaign"):
        module.publish(argparse.Namespace(repository=ROOT, artifacts=tmp_path, output=tmp_path / "report",
                                           registration=registration))


def test_saved_comparison_accepts_three_arms_and_rejects_stream_tampering(monkeypatch):
    """Use explicit software states to test comparison, never qualification."""
    from experiments.forge.rng import NamedStreams
    from experiments.forge.state import state_digest
    module = helper()
    streams = NamedStreams(0, device="cpu")
    streams.generator("data", component="target", purpose="real")
    def entry(role):
        saved = dict(initializer="deterministic_orthogonal", initialization={"seed": 0}, prior={"kind": "mog"},
                     streams=deepcopy(streams.state_dict()))
        return dict(saved=saved, row=dict(task_id="two_pole", gate_status="PASS", evidence=dict(data_sha256="fixture")),
            item=dict(role=role, importance="required", mechanism_stats={},
                provenance_checkpoint=dict(completed_steps=2, state_sha256=state_digest(saved))),
            task=dict(adapter="transfer_behavior"))
    data = dict(final=[entry(role) for role in ("incumbent", "repair1", "repair2")], scopes=[])
    result = module.audit_saved_comparison(ROOT, data, ("incumbent", "repair1", "repair2"))
    assert result["status"] == "PASS"
    assert result["saved_state_comparisons"][0]["all_declared_arms_present"]
    assert result["saved_state_comparisons"][0]["consumed_non_eval_streams_and_batches_equal"]
    # Byte-valid RNG state from the same named stream after a real draw.
    import torch
    torch.rand(1, generator=streams.generator("data", component="target", purpose="real"))
    data["final"][-1]["saved"]["streams"] = streams.state_dict()
    with pytest.raises(ValueError, match="actual consumed training RNG states differ"):
        module.audit_saved_comparison(ROOT, data, ("incumbent", "repair1", "repair2"))
