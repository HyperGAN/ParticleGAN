"""End-to-end Forge view, study, CLI, problem-only gate, and api_run tests for Four Laws Adherence."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image

from benchmarks.toy_audit import api_four_laws_123 as fl123
from benchmarks.toy_audit import api_four_laws_4 as fl4
from benchmarks.toy_audit import api_run
from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.four_laws import (
    FOUR_LAWS_KINDS,
    _contract_payload,
    grade as four_laws_grade,
)
from experiments.forge.mechanisms import NAMES as MECHANISM_NAMES
from experiments.forge.planning import load_idea, resolve_idea
from experiments.forge.priors import RECIPE_PRIOR_CONTRACT
from experiments.forge.sampling import executed_receipt
from experiments.forge.studies import load_study
from experiments.forge.views import grade_result, load_tasks, load_view

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ANCHORS = {
    "grid100",
    "staggered100",
    "bytegan_hypercube8",
    "bytegan_bigan_dictionary",
    "interp_sparse_arc",
    "interp_saddle_barycentric",
    "img_bars4",
    "img_stripes2",
    "nash_overpowered_critic_ring16",
    "dcgan_performance_r256",
    "t3_iso_scale_mass_8x",
    "t3_iso_horizon_saddle_shift",
    "ring_extension",
    "target_shift_recovery",
    "model_ae_ddgan_gaussian_chain",
    "grid100_mlp16",
    "grid100_res16_tanh",
    "zero_tanh_rail_escape",
    "clockfree_audit_measurement_v1",
    "gaussian1d_stability",
}

EXPECTED_PREREQUISITES = {
    "mode_hold",
    "ring_hold",
    "zero_tanh_smoke",
    "gaussian1d_smoke",
}

FORBIDDEN_INTERNAL_THRESHOLD_FRAGMENTS = (
    "stiff_gate",
    "s_th",
    "d_max_s1",
    "mean_dz",
    "peak_rms",
    "normal_grad",
    "critic_grad",
    "r_stable",
    "band_scale",
    "step_ms",
    "diag_",
)


def test_four_laws_adherence_view_and_study_resolve_cleanly():
    tasks = load_tasks(ROOT)
    assert FOUR_LAWS_KINDS == set(fl123.TASKS) | set(fl4.TASKS)
    assert len(FOUR_LAWS_KINDS) == 9
    assert FOUR_LAWS_KINDS <= set(tasks)

    view = load_view(ROOT, "four-laws-adherence-v1")
    assert view["evidence_scope"] == "research_diagnostic"
    assigned = {a["task"]: a for a in view["assignments"]}
    assert set(assigned) == EXPECTED_PREREQUISITES | EXPECTED_ANCHORS | FOUR_LAWS_KINDS
    for name, assignment in assigned.items():
        assert assignment["qualification_tier"] == 1
        assert assignment["importance"] == "diagnostic"
        assert assignment.get("law") in {"prerequisite", "law1", "law2", "law3", "law4"}
        if name in FOUR_LAWS_KINDS:
            assert assignment.get("role") == "four_laws_probe"
        elif name in EXPECTED_ANCHORS:
            assert assignment.get("role") == "anchor"
        else:
            assert assignment.get("role") == "prerequisite"

    law_groups = view["law_groups"]
    assert set(law_groups) == {"law1", "law2", "law3", "law4"}
    grouped = {tid for members in law_groups.values() for tid in members}
    assert grouped == EXPECTED_ANCHORS | FOUR_LAWS_KINDS

    study = load_study(ROOT, "unified-v14-four-laws-adherence-study")
    assert study["scope"]["view"] == "four-laws-adherence-v1"
    # Colab-ready: do not hardcode a single local GPU model in study scope.
    assert "cuda_model" not in study["scope"]

    request = resolve_idea(
        ROOT,
        "unified-v14",
        view_id="four-laws-adherence-v1",
        through_tier=1,
        execution_backend="cpu",
    )
    for tid in sorted(FOUR_LAWS_KINDS):
        assert request["tasks"][tid]["preflight_blockers"] == [], f"{tid} blocked: {request['tasks'][tid]['preflight_blockers']}"


def test_forge_validate_and_probe_cli():
    val = subprocess.run(
        [sys.executable, "-m", "experiments.forge", "validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert val.returncode == 0, val.stderr[-2000:]

    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.forge",
            "probe",
            "unified-v14",
            "--task",
            "fl_law1_aniso_unequal_modes",
            "--view",
            "four-laws-adherence-v1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    payload = json.loads(probe.stdout)
    assert payload["candidate"] == "unified-v14"
    assert payload["task"] == "fl_law1_aniso_unequal_modes"
    assert payload["training_launched"] is False


def test_all_nine_four_laws_tasks_enforce_problem_only_output_thresholds():
    tasks = load_tasks(ROOT)
    for mod in (fl123, fl4):
        controls = mod.oracle_controls()
        assert controls and all(v["passed"] is v["expected_pass"] for v in controls.values())
        for tid in mod.TASKS:
            thresholds = mod.THRESHOLDS[tid]
            assert tasks[tid]["evaluation"]["thresholds"] == thresholds
            assert thresholds[0] == ["finite_fraction", "==", 1.0]
            for metric_name, op, bound in thresholds:
                lower = metric_name.lower()
                for forbidden in FORBIDDEN_INTERNAL_THRESHOLD_FRAGMENTS:
                    assert forbidden not in lower, (
                        f"Task {tid} threshold {metric_name!r} violates problem-only policy ({forbidden!r})"
                    )
                assert op in {"<=", ">=", "=="}
                assert isinstance(bound, (int, float)) and math.isfinite(float(bound))


def test_api_run_list_and_bounded_8step_gif_export_across_recipes(tmp_path):
    listed = subprocess.run(
        [sys.executable, "-m", "benchmarks.toy_audit.api_run", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert listed.returncode == 0, listed.stderr[-2000:]
    listed_ids = {line.split("\t", 1)[0] for line in listed.stdout.splitlines() if line.strip()}
    assert FOUR_LAWS_KINDS <= listed_ids

    for case_id, recipe_name in (
        ("fl_law1_aniso_unequal_modes", "unified_v14"),
        ("fl_law2_pure_adversarial_codim1", "bispf"),
        ("fl_law41_trust_rare1pct_tanh", "minimal"),
    ):
        out_dir = tmp_path / f"api_{case_id}_{recipe_name}"
        rc = api_run.main([
            "--case",
            case_id,
            "--steps",
            "8",
            "--eval-samples",
            "128",
            "--frames",
            "5",
            "--recipe",
            recipe_name,
            "--device",
            "cpu",
            "--output",
            str(out_dir),
        ])
        # Bounded 8-step prefix must honestly return FAIL (nonzero) for incomplete default budget.
        assert rc == 1
        case_dir = out_dir / case_id
        receipt = json.loads((case_dir / "receipt.json").read_text())
        assert receipt["status"] == "COMPLETE"
        assert receipt["completed_updates"] == 8
        assert receipt["default_protocol_complete"] is False
        assert receipt["verdict"] == "FAIL"
        gif_path = case_dir / "goal.gif"
        assert gif_path.is_file()
        assert receipt["artifacts"]["goal.gif"]["sha256"] == api_run.file_hash(gif_path)
        with Image.open(gif_path) as gif:
            assert gif.n_frames == 5


def test_forge_four_laws_adapter_preflight_prefix_and_fail_closed_grading(tmp_path):
    tasks = load_tasks(ROOT)
    v14_cand = load_idea(ROOT, "unified-v14")
    minimal_cand = {"recipe_preset": "minimal", "claim_contract": {"sampling_law": "task_declared"}}
    bispf_cand = {"recipe_preset": "bispf", "claim_contract": {"sampling_law": "task_declared"}}

    for tid in sorted(FOUR_LAWS_KINDS):
        task = tasks[tid]
        assert adapter_preflight(task, v14_cand, root=ROOT) == []
        assert adapter_preflight(task, minimal_cand, root=ROOT) == []
        assert adapter_preflight(task, bispf_cand, root=ROOT) == []

    # Software-prefix execution through Forge run_task on both unified-v14 and non-v14 minimal.
    for tid, cand in (
        ("fl_law1_aniso_unequal_modes", v14_cand),
        ("fl_law41_trust_rare1pct_tanh", minimal_cand),
    ):
        original = deepcopy(tasks[tid])
        prefix = deepcopy(original)
        prefix["id"] = f"{tid}-software-prefix"
        prefix["execution"]["steps"] = 5
        prefix["execution"]["host_definition"]["steps"] = 5
        prefix["evaluation"]["observations"] = 5
        prefix["evaluation"]["eval_samples"] = 128

        request = {"candidate": cand, "protocol": {"seed": 0}, "tasks": {prefix["id"]: prefix}}
        raw = run_task(request, {"task_id": prefix["id"]}, tmp_path / f"forge_{tid}", "cpu")
        assert raw["cost"]["completed_steps"] == 5
        guards = raw["evidence"]["guards"]
        assert guards["all_finite"] is True
        assert guards["hooks_exercised"] is True
        assert guards["unintended_rng_deviations"] == 0
        assert guards["optimizer_updates"] == {"generator": 5, "discriminator": 5, "prior": 5}
        gif_info = raw["evidence"]["actual_training_gif"]
        assert gif_info["frames"] == 5
        assert Path(gif_info["path"]).is_file()
        assert raw["evidence"]["final_margins"]
        # Grading a 5-step prefix against the full-budget task must fail closed as INCOMPLETE.
        assert grade_result(original, raw)["status"] == "INCOMPLETE"

    # Fail-closed synthetic grading checks across all 9 tasks.
    for tid in sorted(FOUR_LAWS_KINDS):
        task = deepcopy(tasks[tid])
        steps = task["execution"]["steps"]
        obs_count = task["evaluation"]["observations"]
        ckpts = sorted({math.ceil(i * steps / obs_count) for i in range(1, obs_count + 1)})
        passing_metrics, failing_metrics = {}, {}
        for name, op, bound in task["evaluation"]["thresholds"]:
            passing_metrics[name] = bound
            failing_metrics[name] = bound - 1.0 if op == ">=" else bound + 1.0

        base_ev = {
            **executed_receipt(task["evaluation"]["sampling_law"], eval_output_noise="clean"),
            "scoring_weights": "live",
            "prior_policy": {
                "contract": RECIPE_PRIOR_CONTRACT,
                "applicability": "latent_prior",
                "update": "learned",
                "trainable_locations": True,
                "regularizer": "vicreg",
                "weight": 0.0,
                "target_std": 1.0,
                "eps": 1e-4,
                "l2": 0.0,
            },
            "four_laws_contract": _contract_payload(tid, task["execution"]["host_definition"]),
            "actual_training_gif": {"sha256": "0" * 64, "frames": obs_count},
            "guards": {
                "all_finite": True,
                "optimizer_updates": {"generator": steps, "discriminator": steps, "prior": steps},
                "hooks_exercised": True,
                "mechanism_audit": {
                    "mechanisms": {
                        name: {"requested": False, "enabled": False, "calls": 0, "eligible": 0, "applied": 0}
                        for name in MECHANISM_NAMES
                    }
                },
                "unintended_rng_deviations": 0,
            },
            "observations": [{"step": s, **passing_metrics} for s in ckpts],
            "live": {"step": steps, **passing_metrics},
        }
        assert grade_result(task, {"evidence": base_ev})["status"] == "PASS"

        fail_ev = {
            **base_ev,
            "observations": [{"step": s, **failing_metrics} for s in ckpts],
            "live": {"step": steps, **failing_metrics},
        }
        assert grade_result(task, {"evidence": fail_ev})["status"] == "FAIL"

        missing_contract = deepcopy(base_ev)
        del missing_contract["four_laws_contract"]
        assert four_laws_grade(task, missing_contract)[0] == "INCOMPLETE"

        tampered = deepcopy(base_ev)
        tampered["four_laws_contract"]["num_particles"] += 1
        assert four_laws_grade(task, tampered)[0] == "INVALID"

        no_gif = deepcopy(base_ev)
        del no_gif["actual_training_gif"]
        assert four_laws_grade(task, no_gif)[0] == "INCOMPLETE"
