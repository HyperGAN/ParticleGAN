"""Real READY admission and queue wiring; no model update or worker starts."""
import argparse
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from experiments.forge.contracts import read_json, file_hash
from experiments.forge.views import load_tasks

ROOT = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location("phase3_orchestration_test", ROOT / "reports/forge/bcap-three-phase/phase3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def software_checkout(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3-source")
    archive = subprocess.check_output(["git", "archive", "HEAD", "particlegan", "experiments", "benchmarks", "lib", "configs"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(root, filter="data")
    extras = {helper().phase2.ORIGINALS, "reports/forge/bcap-develop-integration/results.json"}
    for task in load_tasks(ROOT).values():
        extras.update(name for name in task["evaluation"].get("sources", {}) if name.startswith("reports/"))
    for name in extras:
        if (ROOT / name).is_file():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    from tests.archived_forge_contracts import restore_archived_contracts
    restore_archived_contracts(root, "bcap_prior_baseline_questions")
    subprocess.run(["git", "init", "-q", root], check=True)
    commit(root)
    return root


def commit(root):
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Software fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-q", "-m", "Software-only declaration freeze"], cwd=root, check=True)


def spec():
    module = helper()
    tasks = list(module.phase2.TIMEOUTS) + list(module.DEEP_TASKS)
    arms = []
    for role, parent, control in (("baseline", "bcap-develop-integration-winner-v1", "software-phase3-candidate"),
                                   ("candidate", "bcap-develop-integration-combined-v1", "software-phase3-baseline")):
        arms.append(dict(role=role, candidate_id=f"software-phase3-{role}", parent=parent, control=control,
            changed_factors=["Software-only admission: existing global winner versus existing combined recipe"],
            mechanism_rationale="Verify fixed original tasks and separately scoped paired admission without training",
            hypothesis="A global substantive mechanism preserves original sustained numerical gates",
            competing_explanation="Incomplete evidence or a cap violation can invalidate an apparent endpoint improvement",
            prior_evidence=[dict(path="reports/forge/bcap-develop-integration/results.json", selector=[],
                identity=dict(schema_version=1, protocol_seed=0), use="motivation_only")],
            prediction=dict(task_id="two_pole", metric="grad_med", op="<=", threshold=1., phase="final"),
            falsifier=dict(task_id="two_pole", metric="grad_med", op=">", threshold=1., phase="final")))
    return dict(schema_version=1, campaign_id="software-phase3-admission", view_id="software-phase3-diagnostic",
        task_ids=tasks, arms=arms, paid_ceiling_seconds=48000, candidate_budget_seconds=24000,
        software_allowance_seconds=300)


@pytest.fixture(scope="module")
def prepared(software_checkout):
    module = helper()
    source, registration = software_checkout / "software-spec.json", software_checkout / "software-registration.json"
    source.write_text(json.dumps(spec()))
    unrelated = software_checkout / "configs/forge/tasks/gaussian1d.json"
    if not unrelated.exists():
        unrelated = next(p for p in (software_checkout / "configs/forge/tasks").glob("gaussian*.json")
                         if p.stem not in spec()["task_ids"])
    original_unrelated = unrelated.read_bytes()
    # A real changed selected-source byte exercises explicit source ancestry.
    path = software_checkout / "particlegan/recipes.py"
    path.write_text(path.read_text() + "\n# Separate software orchestration source-binding fixture.\n")
    module.prepare(argparse.Namespace(repository=software_checkout, spec=source, registration=registration,
                                     source_commit=module.head(software_checkout)))
    assert unrelated.read_bytes() == original_unrelated
    return software_checkout, registration


def test_prepare_resolves_real_ready_all16_with_no_queue_or_workers(prepared):
    module = helper()
    root, registration = prepared
    result = read_json(registration)
    assert result["scope"] == "phase3_paired_research_diagnostic" and not result["qualification_input"]
    assert result["full_reservation_seconds"] == 45840 and result["per_arm_full_reservation_seconds"] == 22920
    assert result["campaign_ceiling_seconds"] == 48000 and result["candidate_ceiling_seconds"] == 24000
    assert len(result["task_ids"]) == 16 and set(result["source_rebindings"]) <= set(result["task_ids"])
    assert all(plan["admission"] == "READY" and len(plan["active_tasks"]) == 16 for plan in result["arms"].values())
    assert all(row["importance"] == "diagnostic" and row["qualification_tier"] == 1
               for plan in result["arms"].values() for row in plan["active_tasks"])
    requests = module.resolved(root, result)
    for request in requests.values():
        assert request["wallclock_policy"] == "declared"
        assert request["tasks"]["gaussian1d_stability"]["dependencies"] == [{"task": "gaussian1d_smoke", "kind": "checkpoint"}]
        assert request["tasks"]["five_word_joint_hold"]["dependencies"] == [{"task": "five_word_joint_smoke", "kind": "checkpoint"}]
        assert request["view"]["evidence_scope"] == "research_diagnostic"
        assert request["protocol"]["seed"] == 0
    assert not (root / "reports/forge/attempts").exists()
    assert not (root / "queue").exists()
    with pytest.raises(ValueError, match="fresh"):
        module.prepare(argparse.Namespace(repository=root, spec=root / "software-spec.json", registration=registration,
                                         source_commit=module.head(root)))


@pytest.mark.parametrize("mutation", ["roster", "budget", "baseline", "controls"])
def test_spec_rejects_unauthorized_scope_or_extra_configuration(mutation):
    module, value = helper(), spec()
    if mutation == "roster":
        value["arms"].append(deepcopy(value["arms"][-1]))
    elif mutation == "budget":
        value["paid_ceiling_seconds"] = 48001
    elif mutation == "baseline":
        value["arms"][0]["recipe_overrides"] = {"lr": 1.}
    else:
        value["arms"][1]["control"] = "foreign-control"
    with pytest.raises(ValueError):
        module.check_spec(value)


def test_changed_original_gate_and_missing_own_producer_fail_closed(prepared):
    module = helper()
    root, registration = prepared
    card = root / "configs/forge/tasks/two_pole.json"
    original = card.read_bytes()
    task = json.loads(original)
    task["evaluation"]["thresholds"][1][2] = 2.
    card.write_text(json.dumps(task))
    try:
        with pytest.raises(ValueError, match="gates changed"):
            module.resolved(root, read_json(registration))
    finally:
        card.write_bytes(original)
    tasks = spec()["task_ids"]
    with pytest.raises(ValueError, match="six original"):
        module.original_contract(root, [name for name in tasks if name != "five_word_joint_smoke"])


@pytest.mark.parametrize("mutation", ["source", "missing_task", "roles"])
def test_registration_cannot_misattribute_source_or_roles(prepared, mutation):
    module = helper()
    root, path = prepared
    registration = read_json(path)
    if mutation == "source":
        registration["source_digest"] = "wrong-scientific-source"
    elif mutation == "missing_task":
        registration["task_contracts"].pop("grid100")
    else:
        registration["arms"]["baseline"], registration["arms"]["candidate"] = (
            registration["arms"]["candidate"], registration["arms"]["baseline"])
    with pytest.raises(ValueError):
        module.resolved(root, registration)


def test_submit_freezes_both_and_drain_only_uses_exact_prior_admission(prepared, tmp_path, monkeypatch):
    module = helper()
    root, registration = prepared
    artifacts = tmp_path / "bulk"
    kwargs = dict(repository=root, registration=registration, artifacts=artifacts,
                  source_commit=module.head(root), submit=True, drain=False, gpus="0,1")
    with pytest.raises(ValueError, match="Commit"):
        module.run(argparse.Namespace(**kwargs))
    commit(root)
    kwargs["source_commit"] = module.head(root)
    from experiments.forge import queue as queue_module
    calls = []
    class SoftwareQueue:
        def __init__(self, *args, **kwargs):
            calls.append(("queue", args))
        def submit(self, request, campaign):
            assert request["source"]["origin_commit"] == module.head(root)
            assert request["source"]["snapshot_path"]
            assert campaign["budget_seconds"] == 48000
            calls.append(("submit", request["candidate"]["id"]))
            return {"request": {"request_id": request["candidate"]["id"]}}
    def software_drain(queue, gpus, **options):
        calls.append(("drain", gpus, options))
    monkeypatch.setattr(queue_module, "Queue", SoftwareQueue)
    monkeypatch.setattr(queue_module, "drain", software_drain)
    module.run(argparse.Namespace(**kwargs))
    assert [c[0] for c in calls].count("submit") == 2 and not any(c[0] == "drain" for c in calls)
    progress = read_json(artifacts / "phase3-progress.json")
    assert progress["phase"] == "admitted" and set(progress["requests"]) == {"baseline", "candidate"}
    with pytest.raises(ValueError, match="unchanged rerun"):
        module.run(argparse.Namespace(**kwargs))
    kwargs.update(submit=False, drain=True)
    bad = deepcopy(kwargs)
    bad["source_commit"] = "wrong-head"
    with pytest.raises(ValueError, match="exact committed HEAD"):
        module.run(argparse.Namespace(**bad))
    module.run(argparse.Namespace(**kwargs))
    assert calls[-1][0] == "drain" and calls[-1][2]["allow_sharing"]
    assert read_json(artifacts / "phase3-progress.json")["phase"] == "diagnostic_complete"


def test_publisher_rejects_active_or_ordinary_work(tmp_path):
    module = helper()
    progress = tmp_path / "phase3-progress.json"
    progress.write_text(json.dumps(dict(requests={"baseline": "b", "candidate": "c"})))
    path = tmp_path / "phase3-queue/queue/state.json"
    path.parent.mkdir(parents=True)
    def state(scope, status):
        return dict(submissions={key: dict(status=status, request=dict(view=dict(evidence_scope=scope)))
                                 for key in ("b", "c")}, jobs={})
    path.write_text(json.dumps(state("research_diagnostic", "running")))
    fake = argparse.Namespace(ACTIVE={"queued", "running", "paused"})
    with pytest.raises(ValueError, match="terminal"):
        module.diagnostic_scopes(fake, tmp_path, progress)
    path.write_text(json.dumps(state("ordinary", "terminal")))
    with pytest.raises(ValueError, match="Ordinary"):
        module.diagnostic_scopes(fake, tmp_path, progress)
    path.write_text(json.dumps(state("research_diagnostic", "terminal")))
    assert module.diagnostic_scopes(fake, tmp_path, progress)[0]["scope"] == "research_diagnostic"


def test_saved_paired_audit_rejects_actual_target_batch_tampering():
    module = helper()
    from experiments.forge.rng import NamedStreams
    from experiments.forge.state import state_digest
    streams = NamedStreams(0, device="cpu")
    streams.generator("data", component="target", purpose="real")
    def entry(role):
        saved = dict(initializer="deterministic_orthogonal", initialization={"seed": 0}, prior={"kind": "mog"},
                     streams=deepcopy(streams.state_dict()))
        return dict(saved=saved, row=dict(task_id="two_pole", gate_status="PASS", evidence=dict(data_sha256="same")),
            item=dict(role=role, importance="diagnostic", mechanism_stats={},
                provenance_checkpoint=dict(completed_steps=2, state_sha256=state_digest(saved))),
            task=dict(adapter="transfer_behavior"))
    data = dict(final=[entry(role) for role in module.ROLES], scopes=[])
    result = module.phase2.audit_saved_comparison(ROOT, data, module.ROLES)
    assert result["saved_state_comparisons"][0]["all_declared_arms_present"]
    data["final"][-1]["row"]["evidence"]["data_sha256"] = "changed-actual-batches"
    with pytest.raises(ValueError, match="actual target-batch sequence differs"):
        module.phase2.audit_saved_comparison(ROOT, data, module.ROLES)


@pytest.mark.parametrize("mutation", ["borrow_baseline", "different_revision", "failed_producer"])
def test_saved_continuation_requires_own_passing_producer(mutation):
    specification = importlib.util.spec_from_file_location("phase3_own_state_test", ROOT / "reports/forge/bcap-develop-integration/audit.py")
    audit = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(audit)
    parent = dict(item=dict(role="candidate", scope="research_diagnostic", attempt_id="software-parent"),
        row=dict(task_id="five_word_joint_smoke", gate_status="PASS", compatibility_key="parent-key",
                 evidence=dict(checkpoint=dict(sha256="checkpoint-bytes", state_sha256="checkpoint-state"))),
        request=dict(candidate_revision="candidate-revision"), task=dict(dependencies=[]))
    child = dict(item=dict(role="candidate", scope="research_diagnostic"),
        row=dict(task_id="five_word_joint_hold", gate_status="FAIL", evidence=dict(continuity=dict(
            restored_exactly=True, history_reset=False, parent_compatibility_key="parent-key",
            parent_candidate_revision="candidate-revision", parent_checkpoint_sha256="checkpoint-bytes",
            parent_state_sha256="checkpoint-state"))), request=dict(candidate_revision="candidate-revision"),
        task=dict(dependencies=[dict(task="five_word_joint_smoke", kind="checkpoint")]))
    if mutation == "borrow_baseline":
        parent["item"]["role"] = "baseline"
    elif mutation == "different_revision":
        parent["request"]["candidate_revision"] = "foreign-revision"
        child["row"]["evidence"]["continuity"]["parent_candidate_revision"] = "foreign-revision"
    else:
        parent["row"]["gate_status"] = "FAIL"
    with pytest.raises(ValueError, match="own certified producer|not this candidate passing state"):
        audit.own_checkpoints(dict(final=[parent, child]))


def test_publication_exports_diagnostic_importance_and_preserves_missing_holds(prepared, tmp_path, monkeypatch):
    """Controlled software collection tests report wiring, never certification."""
    from contextlib import contextmanager
    from PIL import Image
    module = helper()
    source, registration = prepared
    declared = read_json(registration)
    requests = module.resolved(source, declared)
    progress = dict(phase="diagnostic_complete", source_commit=module.head(ROOT),
                    registration_sha256=file_hash(registration), requests={r: r for r in module.ROLES})
    (tmp_path / "phase3-progress.json").write_text(json.dumps(progress))
    final, cells = [], []
    for role in module.ROLES:
        for task_id in declared["task_ids"]:
            hold = task_id in {"gaussian1d_stability", "five_word_joint_hold"}
            cells.append(dict(scope="research_diagnostic", role=role, task_id=task_id,
                              gate_status="BLOCKED" if hold else "PASS"))
            if hold:
                continue
            final.append(dict(item=dict(scope="research_diagnostic", role=role, task_id=task_id,
                              attempt_id=f"software-{role}-{task_id}", importance="diagnostic", gate_status="PASS"),
                              row=dict(task_id=task_id, gate_status="PASS", metrics=dict(grad_med=.5)),
                              task=dict(dependencies=[])))
    collection = dict(final=final, cells=cells, scopes=[], attempts={}, accounting=[])
    renders = []
    @contextmanager
    def forbid_live_execution():
        renders.append("guard_enter")
        yield
        renders.append("guard_exit")
    renderer = argparse.Namespace(forbid_live_execution=forbid_live_execution)
    def render_saved(entry, path, guarded_renderer):
        assert renders[-1] == "guard_enter"
        path.parent.mkdir(parents=True, exist_ok=True)
        first, second = Image.new("RGB", (3, 3), "black"), Image.new("RGB", (3, 3), "white")
        first.save(path, save_all=True, append_images=[second], duration=10)
        return dict(task_id=entry["item"]["task_id"], qualification_input=False,
                    optimizer_updates_added=0, sampling_draws_added=0)
    saved_publisher = argparse.Namespace(collect=lambda _: collection,
        _saved_renderer=lambda _: renderer, render_saved=render_saved)
    monkeypatch.setattr(module, "resolved", lambda *args: requests)
    monkeypatch.setattr(module.phase2, "publisher", lambda _: saved_publisher)
    monkeypatch.setattr(module.phase2, "audit_saved_comparison", lambda *args: dict(status="PASS", software_fixture=True))
    output = tmp_path / "software-publication"
    module.publish(argparse.Namespace(repository=ROOT, registration=registration, artifacts=tmp_path,
                                     output=output, source_commit=module.head(ROOT)))
    result = read_json(output / "phase3-results.json")
    assert result["scope"] == "phase3_paired_research_diagnostic" and not result["qualification_input"]
    assert result["original_placement_counts"] == dict(tier1=6, tier2=10)
    assert len(result["task_cells"]) == 32 and result["actual_training_gifs"] == 28
    assert result["outcomes"] == {role: {"PASS": 14, "BLOCKED": 2} for role in module.ROLES}
    assert all(outcome["outcome"] == "incomplete" and not outcome["qualification_input"]
               for outcome in result["study_outcomes"].values())
    assert len(renders) == 56
