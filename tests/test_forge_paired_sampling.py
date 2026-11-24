"""Paired sampling stays separate from required clean scoring and training RNG."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from benchmarks.toy100.accuracy_evidence import AccuracyEvidence
from benchmarks.toy100.problems import sample_real
from experiments.forge import adapters
from experiments.forge.api import FormulationContext
from experiments.forge.mechanisms import MechanismAudit, mechanism_blockers
from experiments.forge.nativeprofiles import (
    RELEASE_PROFILE_ID,
    build_native_models,
    native_host_initialization,
    native_profile_blockers,
    resolve_native_spec,
)
from experiments.forge.paired_sampling import (
    DECLARATION,
    FIELD,
    PairedOutputNoiseEvidence,
    add_output_noise,
    paired_sampling_blockers,
)
from experiments.forge.sampling import task_blockers
from experiments.forge.state import state_digest
from particlegan.training import output_noise_std

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_paired_addition_matches_public_sample_law_and_zero_amplitude_consumes_no_draws():
    context = FormulationContext(recipe_overrides={"num_particles": 12, "z_dim": 2,
        "total_steps": 2, "output_noise_std": .029, "output_noise_warmup": 0})
    trainer = context.build_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    before = context.state_dict()
    stream = torch.Generator().manual_seed(731)
    same = torch.Generator().set_state(stream.get_state())
    clean = trainer.sample(200, generator=stream)
    paired = add_output_noise(clean, output_noise_std(context.recipe, 0), stream)
    public_noisy = trainer.sample(200, generator=same, output_noise=True)
    assert torch.equal(paired, public_noisy)
    assert state_digest(context.state_dict()) == state_digest(before)
    old = stream.get_state().clone()
    assert add_output_noise(clean, 0., stream) is clean
    assert torch.equal(old, stream.get_state())


def test_paired_contract_is_native_diagnostic_only_and_required_policy_stays_clean():
    task = json.loads((ROOT / "configs/forge/tasks/grid100_affine_paired_laws_v1.json").read_text())
    assert task_blockers(task) == []
    assert task["evaluation"]["eval_output_noise"] == "clean"
    assert task["evaluation"][FIELD] == DECLARATION
    task["evaluation"][FIELD]["qualification_reuse"] = True
    assert paired_sampling_blockers(task)
    assert task_blockers(task)
    task["evaluation"][FIELD] = None
    assert paired_sampling_blockers(task)
    assert task_blockers(task)
    task["evaluation"][FIELD] = deepcopy(DECLARATION)
    task["adapter"] = "transfer_vector"
    assert paired_sampling_blockers(task)


def test_release_host_pins_full_resources_gaussian_cloud_and_actual_initial_tensors():
    task = json.loads((ROOT / "configs/forge/tasks/grid100_release07_cloud_named_v1.json").read_text())
    spec = resolve_native_spec(task)
    assert task["execution"]["native_profile"]["id"] == RELEASE_PROFILE_ID
    assert spec["resources"] == {"z_dim": 4, "batch_size": 256, "num_particles": 20000}
    prior = task["execution"]["prior"]
    assert native_profile_blockers(task, {"prior": prior}) == []
    assert native_profile_blockers(task, {"prior": prior, "recipe_overrides": {"z_dim": 2}})
    from test_release07_parity import NEUTRAL_K3P_FIELDS
    context = FormulationContext(prior=prior, host_initialization=native_host_initialization(task),
        recipe_overrides={**spec["resources"], **NEUTRAL_K3P_FIELDS, "reg_arm": "b_cap"})
    g, d = build_native_models(context, spec)
    before = torch.get_rng_state().clone()
    trainer = context.build_trainer(g, d)
    assert torch.equal(before, torch.get_rng_state())
    assert trainer.prior.z.shape == (20000, 4)
    expected = torch.empty_like(trainer.prior.z)
    stream = torch.Generator().manual_seed(context.streams.seed_for("init", component="prior", purpose="z"))
    nn.init.normal_(expected, 0., 1., generator=stream)
    assert torch.equal(trainer.prior.z, expected)
    assert not trainer.prior_mechanisms["a2"]["requested"]
    assert context.receipt()["initialization"]["prior"]["parameters"]["z"]["tensor_sha256"] == state_digest(expected)
    changed = deepcopy(task)
    changed["execution"]["prior"]["sigma"] = .025
    assert native_profile_blockers(changed, {"prior": prior})


@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
def test_fixed_penalty_mechanisms_never_claim_a_k3p_anchor(arm):
    context = FormulationContext(recipe_overrides={"reg_arm": arm, "num_particles": 12,
        "batch_size": 4, "total_steps": 1})
    trainer = context.build_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    audit = MechanismAudit(context.recipe, trainer.opt_d, [trainer.opt_g])
    result = trainer.step(torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]), collect_stats=True)
    audit.observe_penalty(result["penalty_stats"])
    receipt = audit.receipt()
    anchor = receipt["mechanisms"]["critic_anchor"]
    assert anchor["requested"] is False and anchor["enabled"] is False and anchor["applied"] == 0
    assert "probe" not in anchor
    assert receipt["mechanisms"]["critic_penalty"]["applied"] == 1
    assert mechanism_blockers(receipt) == []


def test_paired_output_noise_evidence_preserves_receipt_law_and_keeps_toy100_guard_strict(tmp_path):
    context = FormulationContext(recipe_overrides={
        "num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 5,
        "output_noise_std": .029, "output_noise_warmup": 0.,
    })
    trainer = context.build_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    steps = [0, 1, 2, 3, 4, 5]
    config = {
        "problem": "grid100", "steps": 5, "device": "cpu", "seed": 0,
        "eval_interval": 1, "early_eval_steps": steps, "eval_samples": 64,
        "snapshot_samples": 32, "eval_output_noise": "clean",
    }
    reference = sample_real("grid100", 64, generator=torch.Generator().manual_seed(11))

    # Semantic control: frozen toy100 AccuracyEvidence still rejects undeclared string laws.
    with pytest.raises(ValueError, match="eval_output_noise must be a boolean or a declared sampling law"):
        AccuracyEvidence({**config, "eval_output_noise": "public_recipe_schedule"},
                         tmp_path / "strict", steps, reference)

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_holdout = {
        "live": np.zeros((100_000, 2), dtype=np.float32),
        "ema": np.ones((100_000, 2), dtype=np.float32),
        "target": sample_real("grid100", 100_000, generator=torch.Generator().manual_seed(12)).numpy(),
    }
    np.savez_compressed(clean_dir / "holdout_samples.npz", **clean_holdout)
    clean_summary = {
        "status": "complete", "problem": "grid100", "config": config,
        "eval_output_noise": "clean", "budget_steps": 5, "completed_steps": 5,
        "eval_steps": steps, "snapshot_steps": steps, "final_samples_file": "final_samples.npz",
        "accuracy": {"protocol": "toy100-accuracy-v1", "check_steps": steps[-5:],
                     "sample_count": 64, "holdout_samples": 100_000},
    }

    before_trainer = state_digest(trainer.state_dict())
    before_streams = context.streams.audit()
    before_global_rng = torch.get_rng_state().clone()
    paired = PairedOutputNoiseEvidence(context, config, tmp_path / "paired-output-noise", steps, reference)
    for step in steps:
        for model in ("live", "ema"):
            paired.observe(step, model, reference, float(step))
    receipt = paired.finish(clean_summary, clean_dir)

    paired_dir = tmp_path / "paired-output-noise" / "grid100"
    written_config = json.loads((paired_dir / "config.json").read_text())
    written_summary = json.loads((paired_dir / "summary.json").read_text())
    assert written_config["eval_output_noise"] == "public_recipe_schedule"
    assert written_config[FIELD] == DECLARATION
    assert written_summary["eval_output_noise"] == "public_recipe_schedule"
    assert written_summary["config"] == written_config
    assert receipt["declaration"] == DECLARATION
    assert receipt["qualification_reuse"] is False
    assert receipt["holdout_output_noise_std"] == pytest.approx(.029)
    with np.load(paired_dir / "holdout_samples.npz") as holdout:
        assert not np.array_equal(holdout["live"], clean_holdout["live"])
        assert not np.array_equal(holdout["ema"], clean_holdout["ema"])
        np.testing.assert_array_equal(holdout["target"], clean_holdout["target"])
    assert state_digest(trainer.state_dict()) == before_trainer
    assert torch.equal(torch.get_rng_state(), before_global_rng)
    bindings = context.streams.manifest()["bindings"]
    allowed = [key for key, value in bindings.items() if value["family"] == "eval"]
    assert context.streams.compare(before_streams, context.streams.audit(), allowed=allowed)["unintended_rng_deviations"] == 0


@pytest.mark.parametrize("task_id", [
    "grid100_release07_cloud_named_v1",
    "grid100_affine_paired_laws_v1",
])
def test_paired_native100_tasks_validate_and_start_through_adapter(tmp_path, monkeypatch, task_id):
    task = json.loads((ROOT / "configs/forge/tasks" / f"{task_id}.json").read_text())
    candidate = {
        "id": "bcap-regression-probe",
        "recipe_preset": "bcap",
        "recipe_overrides": {},
        "claim_contract": {"sampling_law": "task_declared"},
    }
    assert task_blockers(task) == []
    assert paired_sampling_blockers(task) == []
    assert native_profile_blockers(task, candidate) == []
    assert resolve_native_spec(task) is not None

    class _StopAtFirstTrainStep(RuntimeError):
        pass

    monkeypatch.setattr(adapters._Run, "step", lambda self, real: (_ for _ in ()).throw(_StopAtFirstTrainStep()))
    frozen = {
        "candidate": candidate,
        "candidate_revision": "unit-regression",
        "protocol": {"seed": 0},
        "tasks": {task_id: task},
        "jobs": [{"task_id": task_id, "compatibility_key": "unit-regression"}],
    }
    with pytest.raises(_StopAtFirstTrainStep):
        adapters.run_task(frozen, {"task_id": task_id}, tmp_path, "cpu")

    clean_dir = tmp_path / "native100" / "grid100"
    paired_dir = tmp_path / "native100" / "paired-output-noise" / "grid100"
    assert json.loads((clean_dir / "config.json").read_text())["eval_output_noise"] == "clean"
    paired_cfg = json.loads((paired_dir / "config.json").read_text())
    assert paired_cfg["eval_output_noise"] == "public_recipe_schedule"
    assert paired_cfg[FIELD] == DECLARATION
    paired_events = [json.loads(line) for line in (paired_dir / "events.jsonl").read_text().splitlines()]
    assert [(row["step"], row["model"]) for row in paired_events] == [(0, "live"), (0, "ema")]


def test_native_adapter_completes_paired_output_noise_diagnostic_end_to_end(tmp_path):
    task = json.loads((ROOT / "configs/forge/tasks/grid100.json").read_text())
    task["id"] = "grid100_paired_smoke"
    task["execution"]["steps"] = 2
    task["execution"]["resources"].update(num_particles=12, batch_size=4)
    task["evaluation"].update(
        eval_interval=1, early_eval_steps=[0, 1, 2], eval_samples=64,
        **{FIELD: deepcopy(DECLARATION)},
    )
    frozen = {
        "candidate": {
            "id": "bcap-paired-smoke",
            "recipe_preset": "bcap",
            "recipe_overrides": {"output_noise_std": .02, "output_noise_warmup": 0.},
            "claim_contract": {"sampling_law": "task_declared"},
        },
        "candidate_revision": "unit-paired-smoke",
        "protocol": {"seed": 0},
        "tasks": {task["id"]: task},
        "jobs": [{"task_id": task["id"], "compatibility_key": "unit-paired-smoke"}],
    }
    raw = adapters.run_task(frozen, {"task_id": task["id"]}, tmp_path, "cpu")
    assert raw["evidence"]["eval_output_noise"] == "clean"
    assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
    paired = raw["evidence"]["paired_sampling_diagnostic"]
    assert paired["declaration"] == DECLARATION
    assert paired["qualification_reuse"] is False
    assert paired["holdout_output_noise_std"] == pytest.approx(.02)
    paired_dir = Path(raw["evidence"]["artifact_root"]) / "paired-output-noise" / "grid100"
    assert json.loads((paired_dir / "summary.json").read_text())["eval_output_noise"] == "public_recipe_schedule"
