"""Independent acquisition, strict retention, and CUDA checkpoint contracts."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.reproducibility import reproducible_execution, construction_rng
from experiments.forge.contracts import read_json
from experiments.forge.mechanisms import NAMES
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view
from experiments.forge.word_tasks import grade, schedule

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
GOOD = dict(sample_count=1024, quality_fraction=1., modes=5, mass_tv=0.,
            reconstruction_exact=1, minimum_reconstruction_token_probability=1.)


def evidence(kind="smoke"):
    prefix, updates = (0, 20001) if kind == "smoke" else (834, 4000)
    steps = ([prefix] if prefix else []) + [prefix + step for step in schedule(updates)]
    guards = dict(all_finite=True, hooks_exercised=True, unintended_rng_deviations=0,
        optimizer_updates={role: prefix + updates for role in ("generator", "encoder", "prior", "discriminator")},
        mechanism_audit=dict(schema_version=1, mechanisms={name: dict(requested=False, enabled=False,
            calls=updates, eligible=0, applied=0) for name in NAMES}))
    confirms = [dict(step=step, metrics=deepcopy(GOOD), primary_state_sha256="a" * 64,
        confirmed_state_sha256="a" * 64, training_state_unchanged=True,
        independent_stream="eval/live/word_confirmation",
        reconstruction_stream="eval/live/word_reconstruction_confirmation") for step in steps]
    return dict(executed_updates=updates, completed_steps=prefix + updates,
        observations=[dict(step=step, **GOOD) for step in steps], confirmations=confirms, guards=guards,
        checkpoint=dict(completed_steps=834, training_state_sha256="a" * 64, selection="earliest_confirmed_passing_state"),
        continuity=dict(prefix_steps=prefix, parent_confirmed_step=prefix, restored_exactly=True,
                        history_reset=False, parent_smoke_status="PASS", same_recipe_prior_architecture=True))


def task(kind="smoke"):
    value = load_tasks(ROOT)["five_word_joint_" + kind]
    # Synthetic numerical reducer fixture under the explicit legacy contract.
    value["execution"].pop("prior_contract", None)
    value["execution"]["prior"]["learnable"] = True
    return value


def test_one_confirmed_joint_hit_passes_smoke_despite_later_collapse():
    raw = evidence()
    for row in raw["observations"][1:]:
        row["reconstruction_exact"] = 0
    result = grade(task(), raw)
    assert result["gate_status"] == "PASS"
    assert result["evaluator_result"]["confirmed_steps"] == [834]
    assert result["evaluator_result"]["endpoint_passed"] is False


@pytest.mark.parametrize("mutation,status", [
    ("short_budget", "INCOMPLETE"), ("short_optimizer", "INCOMPLETE"),
    ("missing_check", "INCOMPLETE"), ("missing_confirmation", "INCOMPLETE"),
    ("wrong_state", "INVALID"), ("same_stream", "INVALID"), ("later_checkpoint", "INVALID"),
    ("generation_only", "FAIL"), ("confirmation_fail", "FAIL")])
def test_acquisition_rejects_incomplete_unconfirmed_or_wrong_checkpoint(mutation, status):
    raw = evidence()
    if mutation == "short_budget": raw["executed_updates"] -= 1
    elif mutation == "short_optimizer": raw["guards"]["optimizer_updates"]["encoder"] -= 1
    elif mutation == "missing_check": raw["observations"].pop()
    elif mutation == "missing_confirmation": raw["confirmations"].pop()
    elif mutation == "wrong_state": raw["confirmations"][0]["confirmed_state_sha256"] = "b" * 64
    elif mutation == "same_stream": raw["confirmations"][0]["independent_stream"] = "eval/live/generated_words"
    elif mutation == "later_checkpoint": raw["checkpoint"]["completed_steps"] = 1667
    elif mutation == "generation_only":
        for row in raw["observations"]: row["reconstruction_exact"] = 0
    else:
        for row in raw["confirmations"]: row["metrics"]["mass_tv"] = .2
    assert grade(task(), raw)["gate_status"] == status


@pytest.mark.parametrize("failed", [0, 1, 12, 24])
def test_hold_requires_restored_state_and_every_continuation_check(failed):
    raw = evidence("hold")
    assert grade(task("hold"), raw)["gate_status"] == "PASS"
    raw["observations"][failed]["reconstruction_exact"] = 0
    assert grade(task("hold"), raw)["gate_status"] == "FAIL"


def test_task_policy_preserves_old_sustained_definition_and_checked_budgets():
    tasks, view = load_tasks(ROOT), load_view(ROOT, "discriminator_stability")
    old = tasks["five_word_joint_acquisition"]
    assert old["evaluation"]["kind"] == "transfer_sustained"
    assert old["evaluation"]["minimum_stable_checks"] == 5
    assignments = {row["task"]: row for row in view["assignments"]}
    assert "five_word_joint_acquisition" not in assignments
    assert assignments["five_word_joint_smoke"]["qualification_tier"] == 1
    assert assignments["five_word_joint_hold"]["qualification_tier"] == 2
    smoke, hold = task(), task("hold")
    assert smoke["execution"]["host_definition"] == hold["execution"]["host_definition"] == old["execution"]["host_definition"]
    assert smoke["evaluation"]["thresholds"] == hold["evaluation"]["thresholds"] == old["evaluation"]["thresholds"]
    assert hold["dependencies"] == [dict(task="five_word_joint_smoke", kind="checkpoint")]
    campaign = read_json(ROOT / "configs/forge/campaigns/technique-inventory-word-split-v1.json")
    groups = {tasks[a["task"]]["execution"].get("execution_group", a["task"]):
              tasks[a["task"]]["resources"]["timeout_seconds"] for a in view["assignments"]}
    assert campaign["candidate_budget_seconds"] == 46620
    # The original campaign keeps its old allowance. New model-mode tasks add
    # an explicit reservation; they cannot silently fit the historical cap.
    assert sum(groups.values()) == 75120
    assert sum(value for key, value in groups.items() if key.startswith("precision_gaussian_")) == 900
    initial = read_json(ROOT / "reports/forge/precision-stretch/initial-view-v14.json")
    initial_keys = {tasks[a["task"]]["execution"].get("execution_group", a["task"]) for a in initial["assignments"]}
    original_groups = {key: value for key, value in groups.items() if key in initial_keys and not key.startswith("model_")}
    assert sum(original_groups.values()) == 58440


@pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="word ladder execution requires CUDA")
def test_cuda_bounded_public_word_draws_preserve_streams_and_save_complete_state(tmp_path):
    from experiments.forge.word_adapter import run_word
    candidate = read_json(next((ROOT / "configs/forge/configurations").glob("bcap-dualnorm--7beb*.json")))
    request = dict(candidate=candidate, candidate_revision="cuda-software-fixture", protocol=dict(seed=0), tasks={task()["id"]: task()})
    @reproducible_execution
    def execute(*, device):
        before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        raw, records = run_word(request, task(), tmp_path, device, execution_limit=2, capture_media=True)
        assert grade_result(task(), raw)["gate_status"] == "INCOMPLETE"
        assert [row["step"] for row in raw["evidence"]["confirmations"]] == [1, 2]
        assert all(row["training_state_unchanged"] for row in raw["evidence"]["confirmations"])
        assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
        assert [record["step"] for record in records] == [0, 1, 2]
        assert torch.equal(before_cpu, torch.get_rng_state())
        assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))
        state = torch.load(tmp_path / "state.pt", weights_only=True, map_location=device)
        assert state["fixture"]["api_state"]["completed_steps"] == 2
        assert all(p.is_cuda for p in state["fixture"]["api_state"]["models"]["generator"].values())
        purposes = {b["purpose"] for b in state["streams"]["manifest"]["bindings"].values()}
        assert {"generated_words", "paired_reconstruction", "word_confirmation", "word_reconstruction_confirmation"} <= purposes
    execute(device=DEVICE)


@pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="external-horizon restoration requires CUDA")
def test_cuda_public_policy_restore_after_schedule_horizon_keeps_actual_cap_and_rates():
    from particlegan import GANTrainer, get_recipe, init
    recipe = get_recipe("bcap", z_dim=2, num_particles=8, batch_size=8, total_steps=1,
                        optimizer_family="dualnorm", lr_floor=1., network_lr_floor=1.,
                        prior_kind="mog", sigma_rel=.1, standardize=False)
    @reproducible_execution
    def execute(*, device):
        def build():
            with construction_rng(0, device):
                G, D = nn.Linear(2, 2, device=device), nn.Linear(2, 1, device=device)
                prior = recipe.make_prior(sigma=.1).to(device)
                for index, model in enumerate((G, D, prior)): init.deterministic_orthogonal_(model, seed=index)
                return GANTrainer(recipe, G, D, prior=prior, seed=0, max_steps=3)
        owner = build()
        real = torch.linspace(-1., 1., 16, device=device).reshape(8, 2)
        owner.step(real); owner.step(real)
        saved = deepcopy(owner.state_dict())
        assert saved["completed_steps"] == 2 > recipe.total_steps
        restored = build(); restored.load_state_dict(saved)
        assert state_digest(restored.state_dict()) == state_digest(saved)
        rates = [[group["lr"] for group in opt.param_groups] for opt in (restored.opt_g, restored.opt_d)]
        restored.step(real)
        assert restored.completed_steps == 3 and restored.recipe.total_steps == 1
        assert rates == [[group["lr"] for group in opt.param_groups] for opt in (restored.opt_g, restored.opt_d)]
        with pytest.raises(RuntimeError): restored.step(real)
        tampered = deepcopy(saved); tampered["completed_steps"] = 4
        with pytest.raises(ValueError): build().load_state_dict(tampered)
    execute(device=DEVICE)
