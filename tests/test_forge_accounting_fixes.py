"""Regression and semantic-control tests for Track A4 Forge accounting fixes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.forge import (
    adapters,
    decision_contracts,
    energy,
    experiment_docs,
    family_reports,
    hf_publish,
    leaderboard_contract,
    scaling_publication,
)
from experiments.forge.contracts import file_hash, read_json, stable_hash

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Cumulative NVML checkpoint Joules & legacy linear fallback
# ---------------------------------------------------------------------------

class _SteppedNVML:
    """Fake ctypes NVML backend whose energy counter advances nonlinearly."""

    def __init__(self, energy_mj_sequence: list[int], gpu_name: bytes = b"NVIDIA Fake A6000"):
        self._seq = list(energy_mj_sequence)
        self._idx = 0
        self._gpu_name = gpu_name

    def nvmlInit_v2(self):
        return 0

    def nvmlShutdown(self):
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index, handle_ref):
        return 0

    def nvmlDeviceGetName(self, handle, buf, size):
        buf.value = self._gpu_name
        return 0

    def nvmlDeviceGetUtilizationRates(self, handle, util_ref):
        util_ref._obj.gpu = 0
        util_ref._obj.memory = 0
        return 0

    def nvmlDeviceGetTotalEnergyConsumption(self, handle, mj_ref):
        val = self._seq[min(self._idx, len(self._seq) - 1)]
        self._idx += 1
        mj_ref._obj.value = val
        return 0


def test_energy_monitor_records_exact_cumulative_checkpoint_joules():
    # Start: 1000 J -> step 250: 1450 J (+450 J) -> step 500: 1600 J (+600 J) -> end (step 1000): 2000 J (+1000 J)
    # Note linear extrapolation at step 250 would be 250 J (wrong by 200 J!); measured cumulative is 450 J.
    fake = _SteppedNVML([1_000_000, 1_450_000, 1_600_000, 2_000_000])
    times = iter([10.0, 20.0])
    mon = energy.EnergyMonitor(0, lib_loader=lambda: fake, clock=lambda: next(times))
    assert energy.active_energy_monitor() is None
    mon.start()
    try:
        assert energy.active_energy_monitor() is mon
        assert mon.record_checkpoint(250) == pytest.approx(450.0)
        assert mon.record_checkpoint(500) == pytest.approx(600.0)
    finally:
        total_j = mon.stop()
    assert energy.active_energy_monitor() is None
    assert total_j == pytest.approx(1000.0)

    summary = mon.summary(1000, first_pass_step=250, sustained_pass_step=500)
    assert summary is not None
    assert set(summary) == set(energy.ENERGY_SUMMARY_KEYS)
    assert summary["joules_total"] == pytest.approx(1000.0)
    assert summary["joules_per_step"] == pytest.approx(1.0)
    assert summary["checkpoint_joules"] == {"250": 450.0, "500": 600.0}
    assert summary["cumulative_joules_to_first_pass"] == pytest.approx(450.0)
    assert summary["cumulative_joules_to_sustained_pass"] == pytest.approx(600.0)
    # Primary joules_to_* fields use measured cumulative NVML Joules, NOT linear 250 J / 500 J.
    assert summary["joules_to_first_pass"] == pytest.approx(450.0)
    assert summary["joules_to_sustained_pass"] == pytest.approx(600.0)
    mon.shutdown()


def test_compute_and_enrich_energy_summary_legacy_linear_fallback():
    legacy = energy.compute_energy_summary(
        energy_joules_total=2000.0,
        wall_seconds=20.0,
        completed_steps=1000,
        first_pass_step=250,
        sustained_pass_step=500,
        gpu_name="NVIDIA L4",
        contended=False,
    )
    assert legacy["checkpoint_joules"] is None
    assert legacy["cumulative_joules_to_first_pass"] is None
    assert legacy["cumulative_joules_to_sustained_pass"] is None
    assert legacy["joules_to_first_pass"] == pytest.approx(500.0)
    assert legacy["joules_to_sustained_pass"] == pytest.approx(1000.0)

    # Enriching a legacy receipt with checkpoint_joules upgrades to exact measured cumulative Joules.
    upgraded = energy.enrich_energy_summary(
        legacy,
        checkpoint_joules={250: 710.0, 500: 1120.0},
    )
    assert upgraded is not None
    assert upgraded["cumulative_joules_to_first_pass"] == pytest.approx(710.0)
    assert upgraded["cumulative_joules_to_sustained_pass"] == pytest.approx(1120.0)
    assert upgraded["joules_to_first_pass"] == pytest.approx(710.0)
    assert upgraded["joules_to_sustained_pass"] == pytest.approx(1120.0)

    # hf_publish extraction prefers cumulative_joules_to_* and recovers joules_per_step via metrics.completed_updates.
    extracted = hf_publish._extract_receipt_energy(
        {
            "energy": {
                "joules_total": 1000.0,
                "joules_per_step": None,
                "first_pass_step": 250,
                "sustained_pass_step": 500,
                "checkpoint_joules": {"250": 410.0, "500": 620.0},
                "gpu_name": "NVIDIA RTX A6000",
                "contended": False,
            }
        },
        metrics={"completed_updates": 1000},
    )
    assert extracted["joules_per_step"] == pytest.approx(1.0)
    assert extracted["cumulative_joules_to_first_pass"] == pytest.approx(410.0)
    assert extracted["joules_to_first_pass"] == pytest.approx(410.0)
    assert extracted["joules_to_sustained_pass"] == pytest.approx(620.0)


# ---------------------------------------------------------------------------
# 2. native100 / native_accuracy pass-step extraction
# ---------------------------------------------------------------------------

def test_extract_pass_steps_native100_events_jsonl(tmp_path):
    task = read_json(ROOT / "configs/forge/tasks/grid100.json")
    artifact_root = tmp_path / "native_out"
    prob_dir = artifact_root / "grid100"
    prob_dir.mkdir(parents=True)

    # Step 0 passing (must be ignored!), EMA rows passing early (must be ignored!),
    # live first passes at step 750, dips at 1000, then passes 5 consecutive checks 1250..2250.
    events = [
        {"event": "eval", "model": "live", "step": 0, "accuracy": {"passed": True}},
        {"event": "eval", "model": "ema", "step": 250, "accuracy": {"passed": True}},
        {"event": "eval", "model": "live", "step": 250, "accuracy": {"passed": False}},
        {"event": "eval", "model": "live", "step": 500, "accuracy": {"passed": False}},
        {"event": "eval", "model": "live", "step": 750, "accuracy": {"passed": True}},
        {"event": "eval", "model": "live", "step": 1000, "accuracy": {"passed": False}},
        {"event": "eval", "model": "live", "step": 1250, "accuracy": {"passed": True}},
        {"event": "eval", "model": "live", "step": 1500, "accuracy": {"passed": True}},
        {"event": "eval", "model": "live", "step": 1750, "accuracy": {"passed": True}},
        {"event": "eval", "model": "live", "step": 2000, "accuracy": {"passed": True}},
        {"event": "eval", "model": "live", "step": 2250, "accuracy": {"passed": True}},
    ]
    (prob_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")

    raw_result = {"task_id": "grid100", "evidence": {"artifact_root": str(artifact_root), "problem": "grid100"}}
    fp, sp = energy.extract_pass_steps(task, raw_result)
    assert fp == 750
    assert sp == 2250


def test_extract_pass_steps_native_accuracy_raw_metrics_and_saturation():
    task = read_json(ROOT / "configs/forge/tasks/grid100.json")
    task = {**task, "evaluation": {**task["evaluation"], "max_saturated_fraction": 0.05}}

    good = {
        "n": 20000,
        "valid_n": 20000,
        "modes": 100,
        "min_hq_mode_mass": 0.008,
        "precision": 0.95,
        "mass_tv": 0.03,
        "max_mode_mass": 0.014,
        "min_cov_eig_ratio": 0.85,
        "max_cov_eig_ratio": 1.15,
        "min_radial_median_ratio": 0.95,
        "max_radial_median_ratio": 1.05,
        "center_rms_sigma": 0.08,
        "abs_cov_trace_bias": 0.04,
        "radial_ks": 0.02,
        "saturated_fraction": 0.01,
    }
    sat_fail = {**good, "saturated_fraction": 0.25}
    raw_result = {
        "task_id": "grid100_res16_tanh",
        "evidence": {
            "eval_events": [
                {"step": 250, "model": "live", "metrics": sat_fail},
                {"step": 500, "model": "live", "metrics": good},
                {"step": 750, "model": "live", "metrics": good},
                {"step": 1000, "model": "live", "metrics": good},
                {"step": 1250, "model": "live", "metrics": good},
                {"step": 1500, "model": "live", "metrics": good},
            ]
        },
    }
    fp, sp = energy.extract_pass_steps(task, raw_result)
    assert fp == 500
    assert sp == 1500


# ---------------------------------------------------------------------------
# 3. completed_steps recovery without breaking behavior_adapters.py SHA-256 pin
# ---------------------------------------------------------------------------

def test_completed_steps_extraction_preserves_behavior_adapters_pin():
    # Confirm behavior_adapters.py remains byte-identical to its pinned SHA-256 across 10 task cards.
    expected_pin = "a7e07a688bfa5800f9de34dc5a116c53ca9b1e96ca339fc0a6e64794e74d8614"
    assert file_hash(ROOT / "experiments/forge/behavior_adapters.py") == expected_pin

    # Behavioral result lacking cost.completed_steps recovers completed_steps via provenance_checkpoint / guards.
    behavior_raw = {
        "task_id": "two_pole",
        "cost": {"wall_seconds": 4.0},
        "evidence": {
            "observations": [{"step": 40, "mean_abs": 0.5, "grad_med": 0.4}, {"step": 80, "mean_abs": 0.5, "grad_med": 0.4}],
            "guards": {"optimizer_updates": {"prior": 80, "discriminator": 80}},
            "provenance_checkpoint": {"completed_steps": 80},
        },
    }
    assert energy.extract_completed_steps(behavior_raw) == 80

    fake = _SteppedNVML([100_000, 180_000])
    times = iter([0.0, 4.0])
    mon = energy.EnergyMonitor(0, lib_loader=lambda: fake, clock=lambda: next(times))
    mon.start()
    mon.stop()
    task = read_json(ROOT / "configs/forge/tasks/two_pole.json")
    adapters._attach_energy_summary(behavior_raw, {"two_pole": task}, "two_pole", mon, "cpu")
    mon.shutdown()

    assert behavior_raw["cost"]["completed_steps"] == 80
    assert behavior_raw["cost"]["energy"]["joules_total"] == pytest.approx(80.0)
    assert behavior_raw["cost"]["energy"]["joules_per_step"] == pytest.approx(1.0)


def test_run_task_records_live_checkpoint_joules_via_event_and_stdout_hooks(tmp_path, monkeypatch):
    """End-to-end adapters.run_task() wires active_energy_monitor -> checkpoint_joules."""
    # 1. _event("observation", step=...) path (used by _vector/_image/_ring/_native/four_laws/interpolation/t3_isolation/bytegan/word/gaussian/adaptation)
    # Start: 1000 J -> step 10: 1350 J (+350 J) -> step 20: 1500 J (+500 J) -> end (step 40): 1600 J (+600 J)
    # Linear extrapolation at step 10 would be 150 J (600/40*10), whereas actual measured cumulative is 350 J!
    fake_nvml_event = _SteppedNVML([1_000_000, 1_350_000, 1_500_000, 1_600_000])
    clock_event = iter([0.0, 10.0])

    def _make_event_mon(device, **kwargs):
        return energy.EnergyMonitor(0, lib_loader=lambda: fake_nvml_event, clock=lambda: next(clock_event))

    def _fake_dispatch_event(request, job, output_dir, device):
        adapters._event("observation", task="demo_task", step=10, metrics={"err": 0.1})
        adapters._event("observation", task="demo_task", step=20, metrics={"err": 0.05})
        return {
            "task_id": "demo_task",
            "cost": {"wall_seconds": 10.0, "completed_steps": 40},
            "evidence": {
                "observations": [
                    {"step": 10, "err": 0.1},
                    {"step": 20, "err": 0.05},
                ],
                "live": {"err": 0.05},
            },
        }

    task_def = {
        "id": "demo_task",
        "evaluation": {"thresholds": [["err", "<=", 0.2]], "minimum_stable_checks": 2},
    }
    monkeypatch.setattr(adapters.EnergyMonitor, "for_device", classmethod(lambda cls, dev, **kw: _make_event_mon(dev, **kw)))
    monkeypatch.setattr(adapters, "_dispatch_task", _fake_dispatch_event)

    res = adapters.run_task({"tasks": {"demo_task": task_def}}, {"task_id": "demo_task"}, tmp_path / "ev", "cpu")
    en = res["cost"]["energy"]
    assert en["checkpoint_joules"] == {"10": 350.0, "20": 500.0}
    assert en["first_pass_step"] == 10
    assert en["sustained_pass_step"] == 20
    assert en["cumulative_joules_to_first_pass"] == pytest.approx(350.0)
    assert en["cumulative_joules_to_sustained_pass"] == pytest.approx(500.0)
    assert en["joules_to_first_pass"] == pytest.approx(350.0)
    assert en["joules_to_sustained_pass"] == pytest.approx(500.0)

    # 2. Stdout JSON path (used by pinned behavior_adapters.py, api_run.py, and model_toy_v2_adapter.py)
    fake_nvml_stdout = _SteppedNVML([2_000_000, 2_420_000, 2_700_000, 3_000_000])
    clock_stdout = iter([0.0, 20.0])

    def _make_stdout_mon(device, **kwargs):
        return energy.EnergyMonitor(0, lib_loader=lambda: fake_nvml_stdout, clock=lambda: next(clock_stdout))

    def _fake_dispatch_stdout(request, job, output_dir, device):
        # Simulate api_run / model_toy_v2_adapter stdout JSON checkpoints without calling _event
        print(json.dumps({"case": "ddgan_gaussian_chain_v2", "step": 250, "metric_passed": True}), flush=True)
        print(json.dumps({"case": "ddgan_gaussian_chain_v2", "step": 500, "metric_passed": True}), flush=True)
        return {
            "task_id": "demo_stdout",
            "cost": {"wall_seconds": 20.0},
            "evidence": {
                "observations": [
                    {"step": 250, "err": 0.1},
                    {"step": 500, "err": 0.05},
                ],
                "live": {"err": 0.05},
                "provenance_checkpoint": {"completed_steps": 1000},
            },
        }

    task_stdout = {
        "id": "demo_stdout",
        "evaluation": {"thresholds": [["err", "<=", 0.2]], "minimum_stable_checks": 2},
    }
    monkeypatch.setattr(adapters.EnergyMonitor, "for_device", classmethod(lambda cls, dev, **kw: _make_stdout_mon(dev, **kw)))
    monkeypatch.setattr(adapters, "_dispatch_task", _fake_dispatch_stdout)

    res2 = adapters.run_task({"tasks": {"demo_stdout": task_stdout}}, {"task_id": "demo_stdout"}, tmp_path / "so", "cpu")
    en2 = res2["cost"]["energy"]
    assert res2["cost"]["completed_steps"] == 1000
    assert en2["checkpoint_joules"] == {"250": 420.0, "500": 700.0}
    assert en2["cumulative_joules_to_first_pass"] == pytest.approx(420.0)
    assert en2["cumulative_joules_to_sustained_pass"] == pytest.approx(700.0)
    assert en2["joules_to_first_pass"] == pytest.approx(420.0)
    assert en2["joules_to_sustained_pass"] == pytest.approx(700.0)
    assert energy.active_energy_monitor() is None


def test_real_vector_adapter_run_task_records_non_null_checkpoint_joules(tmp_path, monkeypatch):
    """Run an actual real PyTorch Forge adapter (vector_two_broad, 4-step software run) through run_task()."""
    from experiments.forge.planning import resolve_idea

    task = read_json(ROOT / "configs/forge/tasks/vector_two_broad.json")
    task = json.loads(json.dumps(task))
    task["execution"]["steps"] = 4
    task["evaluation"]["observations"] = 2

    # Build minimal resolved candidate via public planner
    req = resolve_idea(ROOT, "axiom-v5", view_id="discriminator_stability", through_tier=1, execution_backend="cpu")
    req["tasks"]["vector_two_broad"] = task

    # Stepwise fake NVML counter: start=1000J, checkpoint@step2=1120J (+120J), checkpoint@step4=1260J (+260J), stop=1300J (+300J)
    fake_nvml = _SteppedNVML([1_000_000, 1_120_000, 1_260_000, 1_300_000])
    times = iter([0.0, 5.0])
    monkeypatch.setattr(
        adapters.EnergyMonitor,
        "for_device",
        classmethod(lambda cls, dev, **kw: energy.EnergyMonitor(0, lib_loader=lambda: fake_nvml, clock=lambda: next(times))),
    )

    out = adapters.run_task(req, {"task_id": "vector_two_broad"}, tmp_path / "real_vec", "cpu")
    en = out["cost"]["energy"]
    assert out["cost"]["completed_steps"] == 4
    assert en["checkpoint_joules"] == {"2": 120.0, "4": 260.0}
    assert en["joules_total"] == pytest.approx(300.0)
    # Enriching with synthetic first_pass_step=2 uses exact 120.0 J (not linear 150.0 J)!
    enriched = energy.enrich_energy_summary(en, first_pass_step=2, sustained_pass_step=4)
    assert enriched is not None
    assert enriched["cumulative_joules_to_first_pass"] == pytest.approx(120.0)
    assert enriched["cumulative_joules_to_sustained_pass"] == pytest.approx(260.0)
    assert enriched["joules_to_first_pass"] == pytest.approx(120.0)
    assert enriched["joules_to_sustained_pass"] == pytest.approx(260.0)


# ---------------------------------------------------------------------------
# 4. decision_contracts study-scoping & uncontended preference
# ---------------------------------------------------------------------------

def _dcgan_perf_request() -> dict:
    protocol = {"seed": 0}
    runtime = {"cuda": "13.0"}
    profiles = {"gpu": "NVIDIA RTX A6000"}
    source = {"digest": "a" * 64}
    study = {
        "schema_version": 1,
        "id": "axiom-v4-dcgan-perf-study",
        "status": "ready",
        "candidate": "axiom-v4",
        "control": {"candidate_id": "unified-v12", "task_map": {}},
        "hypothesis": "solo idle axiom-v4 <= 35 ms",
        "competing_explanation": "overhead > 35 ms",
        "prediction": {"task_id": "dcgan_performance_r256", "metric": "update_ms_median", "op": "<=", "threshold": 35.0, "phase": "final"},
        "falsifier": {"task_id": "dcgan_performance_r256", "metric": "update_ms_median", "op": ">", "threshold": 35.0, "phase": "final"},
        "scope": {"view": "dcgan_performance", "through_tier": 1, "execution_backend": "cuda", "cuda_model": "NVIDIA RTX A6000"},
        "campaign": {"id": "axiom-v4-dcgan-perf-study", "budget_seconds": 3600, "candidate_budget_seconds": 3600},
        "max_rounds": 1,
        "terminal_rules": {
            "prediction_observed": "review_saved_diagnostics",
            "falsified": "stop_revision",
            "inconclusive": "stop_and_readout",
            "incomplete": "request_missing_evidence",
        },
    }
    expected = {
        "prior_evidence": [
            {
                "path": "reports/forge/records/readout-59898e1a77c4b179c9aebbc1.json",
                "sha256": "e" * 64,
                "selector": [],
                "identity": {"candidate_id": "unified-v12"},
                "use": "motivation_only",
            }
        ],
        "task_ids": ["dcgan_performance_r256"],
        "task_map": {},
        "control_binding_sha256": "c" * 64,
        "candidate_binding_sha256": "b" * 64,
        "substantive_delta": ["optimizer_family"],
        "protocol_sha256": stable_hash(protocol),
        "source_digest": source["digest"],
        "execution_backend": "cuda",
        "runtime_cohort_sha256": stable_hash({"runtime": runtime, "compute_profiles": profiles}),
        "jobs_sha256": "d" * 64,
    }
    return {
        "protocol": protocol,
        "source": source,
        "execution_backend": "cuda",
        "runtime": runtime,
        "compute_profiles": profiles,
        "study": study,
        "study_review": {"expected": expected},
        "view": {"id": "dcgan_performance", "assignments": [{"task": "dcgan_performance_r256", "qualification_tier": 1}]},
        "tasks": {"dcgan_performance_r256": {}},
        "through_tier": 1,
        "candidate": {"id": "axiom-v4"},
        "jobs": [],
    }


def test_decision_contract_scopes_by_producer_study_and_prefers_uncontended():
    req = _dcgan_perf_request()
    solo_row = {
        "task_id": "dcgan_performance_r256",
        "gate_status": "PASS",
        "producer_study_id": "axiom-v4-dcgan-perf-study",
        "cost": {"energy": {"contended": False}},
        "metrics": {"update_ms_median": 34.861953},
    }
    foreign_contended = {
        "task_id": "dcgan_performance_r256",
        "gate_status": "PASS",
        "producer_study_id": "axiom-v4-four-laws-adherence-study",
        "cost": {"energy": {"contended": True}},
        "metrics": {"update_ms_median": 47.168107},
    }

    # Cross-study duplicate resolved by producer_study_id.
    out = decision_contracts.evaluate(req, [solo_row, foreign_contended])
    assert out["duplicate_task_ids"] == []
    assert out["outcome"] == "prediction_observed"
    assert out["observed"]["prediction"]["value"] == pytest.approx(34.861953)
    assert out["observed"]["prediction"]["satisfied"] is True

    # Same-study uncontended vs contended resolved in favor of uncontended.
    same_study_contended = {**foreign_contended, "producer_study_id": "axiom-v4-dcgan-perf-study"}
    out_same = decision_contracts.evaluate(req, [solo_row, same_study_contended])
    assert out_same["duplicate_task_ids"] == []
    assert out_same["outcome"] == "prediction_observed"
    assert out_same["observed"]["prediction"]["value"] == pytest.approx(34.861953)

    # Semantic control: genuine ambiguous duplicate (same study, both uncontended) stays incomplete!
    dup_solo = {**solo_row, "metrics": {"update_ms_median": 34.91}}
    out_ambig = decision_contracts.evaluate(req, [solo_row, dup_solo])
    assert out_ambig["duplicate_task_ids"] == ["dcgan_performance_r256"]
    assert out_ambig["outcome"] == "incomplete"

    # End-to-end concluded_outcomes check across same-cohort uncontended vs contended attempts.
    attempts = [
        {
            "attempt_id": "solo-attempt",
            "request": req,
            "result_hash": "1" * 64,
            "valid_receipt": True,
            "task_results": [{k: v for k, v in solo_row.items() if k != "producer_study_id"}],
        },
        {
            "attempt_id": "contended-attempt",
            "request": req,
            "result_hash": "2" * 64,
            "valid_receipt": True,
            "task_results": [{k: v for k, v in same_study_contended.items() if k != "producer_study_id"}],
        },
    ]
    concluded = decision_contracts.concluded_outcomes(attempts)
    assert len(concluded) == 1
    assert concluded[0]["duplicate_task_ids"] == []
    assert concluded[0]["outcome"] == "prediction_observed"
    assert concluded[0]["observed"]["prediction"]["value"] == pytest.approx(34.861953)


# ---------------------------------------------------------------------------
# 5. Pareto INCOMPLETE timeout exclusion
# ---------------------------------------------------------------------------

def test_pareto_occam_excludes_incomplete_timeouts_with_transparent_count(tmp_path):
    # Live repository axiom-v4 L4 benchmark excludes five_word_joint_smoke 900s timeout (13/15 + 1 timeout).
    exports = hf_publish.build_inventory_exports(ROOT, generated_at="2026-11-25T00:00:00Z")
    ax_l4 = next(
        row for row in exports["energy_rows"]
        if row["candidate_id"] == "axiom-v4" and row["hardware_gpu"] == "NVIDIA L4"
    )
    assert ax_l4["occam_pass"] == 13
    assert ax_l4["occam_executed"] == 15
    assert ax_l4["occam_timeouts"] == 1

    # Synthetic readout check via _readout_occam_counts
    rec_dir = tmp_path / "reports/forge/records"
    rec_dir.mkdir(parents=True)
    (rec_dir / "readout-synthetic.json").write_text(json.dumps({
        "provenance": {"source_origin_commit": "abc123"},
        "task_results": [
            {"task_id": "t1", "gate_status": "PASS", "raw_status": "completed"},
            {"task_id": "t2", "gate_status": "FAIL", "raw_status": "completed"},
            {"task_id": "t3", "gate_status": "INCOMPLETE", "raw_status": "timeout"},
            {"task_id": "t4", "gate_status": "TIMEOUT", "raw_status": "timeout"},
        ],
    }))
    passed, executed, timeouts, commit = hf_publish._readout_occam_counts(tmp_path, "readout-synthetic.json")
    assert (passed, executed, timeouts, commit) == (1, 2, 2, "abc123")


# ---------------------------------------------------------------------------
# 6. 59-runnable vs 12-backlog fixed full-suite split
# ---------------------------------------------------------------------------

def test_fixed_full_suite_59_runnable_and_12_backlog_split(tmp_path):
    fixed = leaderboard_contract.load_fixed_test_list(ROOT)
    prev = leaderboard_contract.load_fixed_test_list(
        ROOT, "configs/forge/leaderboards/technique-inventory-full-suite-2026-11-23.json"
    )
    assert fixed is not None and prev is not None
    assert fixed["dated"] == "2026-11-25"
    assert len(fixed["tests"]) == 59
    assert len(fixed["unimplemented_backlog"]) == 12
    assert set(fixed["unimplemented_backlog"]) == set(leaderboard_contract.UNIMPLEMENTED_TIER4_BACKLOG)
    assert not (set(fixed["tests"]) & set(fixed["unimplemented_backlog"]))
    assert set(fixed["tests"]) | set(fixed["unimplemented_backlog"]) == set(prev["tests"])

    # Semantic guard: overlapping tests & unimplemented_backlog must be rejected.
    bad_rel = Path("configs/forge/leaderboards/bad.json")
    (tmp_path / bad_rel).parent.mkdir(parents=True)
    (tmp_path / bad_rel).write_text(json.dumps({
        "schema": leaderboard_contract.FIXED_TEST_LIST_SCHEMA,
        "dated": "2026-11-25",
        "tests": ["a", "b"],
        "unimplemented_backlog": ["b", "c"],
    }))
    with pytest.raises(ValueError, match="malformed unimplemented_backlog"):
        leaderboard_contract.load_fixed_test_list(tmp_path, bad_rel)

    exports = hf_publish.build_inventory_exports(ROOT, generated_at="2026-11-25T00:00:00Z")
    lb = exports["leaderboard_json"]
    assert lb["summary"]["full_suite_tests"] == 59
    assert lb["summary"]["unimplemented_backlog_tests"] == 12
    assert lb["full_suite_list"]["count"] == 59
    assert lb["full_suite_list"]["unimplemented_backlog_count"] == 12
    assert set(lb["full_suite_list"]["unimplemented_backlog"]) == set(leaderboard_contract.UNIMPLEMENTED_TIER4_BACKLOG)


# ---------------------------------------------------------------------------
# 7. Diagnostic PASS (DIAG_PASS) vs UNKNOWN distinction & strict qualified_tier
# ---------------------------------------------------------------------------

def test_diagnostic_pass_distinguished_from_unknown_without_changing_qualification():
    # Semantic control: recorded FAIL must NEVER become DIAG_PASS!
    diag = {"status": "DIAG_PASS", "gate_status": "PASS", "qualification_input": False}
    assert family_reports.display_task_status({"status": "UNKNOWN"}, diag) == "DIAG_PASS"
    assert family_reports.display_task_status({"status": "NOT_RUN"}, diag) == "DIAG_PASS"
    assert family_reports.display_task_status({"status": "FAIL"}, diag) == "FAIL"
    assert family_reports.display_task_status({"status": "UNKNOWN"}, None) == "UNKNOWN"

    exports = hf_publish.build_inventory_exports(ROOT, generated_at="2026-11-26T00:00:00Z")
    lb = exports["leaderboard_json"]
    axiom_fam = next(f for f in lb["families"] if f["id"] == "axiom")
    # Strict qualification & full-suite pass counts are NEVER inflated by diagnostic passes!
    assert axiom_fam["candidate_id"] == "axiom-v5"
    assert axiom_fam["qualified_tier"] == 2
    assert axiom_fam["full_suite"]["pass"] == 45

    axiom_tasks = {t["task_id"]: t for t in lb["tasks"] if t["family_id"] == "axiom"}
    for tid in (
        "grid100_14k",
        "rotated100_14k",
        "staggered100_14k",
        "grid100_mlp16",
        "grid100_res16_scaled_tanh",
    ):
        cell = axiom_tasks[tid]
        assert cell["status"] == "DIAG_PASS", tid
        assert cell["recorded_status"] == "UNKNOWN", tid
        assert cell["diagnostic_pass"]["qualification_input"] is False
        assert cell["diagnostic_pass"]["qualification_reuse"] is False
        assert cell["diagnostic_pass"]["gate_status"] == "PASS"

    # Unified-v15 Tier-3/4 multi-view passes also surface as non-qualifying DIAG_PASS.
    unified_tasks = {t["task_id"]: t for t in lb["tasks"] if t["family_id"] == "unified"}
    for tid in ("ring_extension", "zero_tanh_rail_escape", "bytegan_bigan_dictionary", "grid100_res16_tanh"):
        cell = unified_tasks[tid]
        assert cell["status"] == "DIAG_PASS", tid
        assert cell["recorded_status"] == "UNKNOWN", tid
        assert cell["diagnostic_pass"]["qualification_input"] is False

    # Truly unmeasured tasks with no passing diagnostic receipt stay NOT_RUN/not_run.
    assert axiom_tasks["pacgan8_grid100"]["status"] != "DIAG_PASS"


# ---------------------------------------------------------------------------
# 8. scaling-performance.json backfill & provisional Law 5 rendering
# ---------------------------------------------------------------------------

def test_scaling_performance_backfill_matches_readout_50697705_and_surfaces_everywhere():
    # Existing receipt-backed manifest validation still passes with all 66 records intact.
    loaded = scaling_publication.load_performance(ROOT)
    assert loaded is not None
    assert len(loaded["records"]) == 66

    manifest = read_json(ROOT / "reports/forge/scaling-performance.json")
    check = {k: v for k, v in manifest.items() if k != "input_digest"}
    assert manifest["input_digest"] == stable_hash(check)

    measurements = manifest.get("measurements")
    assert isinstance(measurements, list) and len(measurements) == 18
    readout = read_json(ROOT / "reports/forge/records/readout-506977051c754e8f9694b117.json")
    readout_sha = file_hash(ROOT / "reports/forge/records/readout-506977051c754e8f9694b117.json")

    expected_tasks = {
        "dcgan_performance_r32",
        "dcgan_performance_r64",
        "dcgan_performance_b64",
        "dcgan_performance_w128",
        "dcgan_performance_r128",
        "dcgan_performance_r256",
    }
    ax4_rows = [m for m in measurements if m["candidate_id"] == "axiom-v4"]
    ax5_rows = [m for m in measurements if m["candidate_id"] == "axiom-v5"]
    u15_rows = [m for m in measurements if m["candidate_id"] == "unified-v15"]
    assert {m["task_id"] for m in ax4_rows} == expected_tasks
    assert {m["task_id"] for m in ax5_rows} == expected_tasks
    assert {m["task_id"] for m in u15_rows} == expected_tasks

    for m in ax4_rows:
        assert m["trainer_family"] == "axiom"
        assert m["contended"] is False
        assert m["producer_study_id"] == "axiom-v4-dcgan-perf-study"
        assert m["source_readout_sha256"] == readout_sha
        assert m["qualification_input"] is False

    # Verify exact historical solo idle GPU 1 r256 median_ms for axiom-v4 is preserved.
    solo_r256_v4 = next(
        tr["metrics"]["update_ms_median"]
        for tr in readout["task_results"]
        if tr["task_id"] == "dcgan_performance_r256" and tr["cost"]["energy"]["contended"] is False
    )
    ax4_r256 = next(m for m in ax4_rows if m["task_id"] == "dcgan_performance_r256")
    assert ax4_r256["median_ms"] == pytest.approx(solo_r256_v4, rel=1e-12)
    assert ax4_r256["median_ms"] == pytest.approx(34.861953, rel=1e-5)

    ax5_r256 = next(m for m in ax5_rows if m["task_id"] == "dcgan_performance_r256")
    u15_r256 = next(m for m in u15_rows if m["task_id"] == "dcgan_performance_r256")
    assert ax5_r256["median_ms"] == pytest.approx(34.391467, rel=1e-5)
    assert u15_r256["median_ms"] == pytest.approx(42.812907, rel=1e-5)

    # Surfaces across family_reports, hf_publish, and experiment_docs for promoted axiom-v5 & unified-v15!
    inv = read_json(ROOT / "reports/forge/technique-inventory.json")
    ax_row = next(r for r in inv["rows"] if r.get("trainer_family") == "axiom")
    r256_ms, _ = family_reports._r256_measurement(inv, ax_row, root=ROOT)
    assert r256_ms == pytest.approx(ax5_r256["median_ms"], rel=1e-12)

    exports = hf_publish.build_inventory_exports(ROOT, generated_at="2026-11-26T00:00:00Z")
    axiom_fam = next(f for f in exports["leaderboard_json"]["families"] if f["id"] == "axiom")
    unified_fam = next(f for f in exports["leaderboard_json"]["families"] if f["id"] == "unified")
    assert axiom_fam["scaling_r256_ms"] == pytest.approx(ax5_r256["median_ms"], rel=1e-12)
    assert unified_fam["scaling_r256_ms"] == pytest.approx(u15_r256["median_ms"], rel=1e-12)

    catalog = experiment_docs.build_experiment_catalog(ROOT)
    r256_scaling = catalog["experiments"]["dcgan_performance_r256"]["scaling_results"]
    ax_scaling = next(r for r in r256_scaling if r["family_id"] == "axiom")
    assert ax_scaling["candidate_id"] == "axiom-v5"
    assert ax_scaling["step_ms_median"] == pytest.approx(ax5_r256["median_ms"], rel=1e-12)


def test_provisional_law5_handled_cleanly_in_hf_publish_and_experiment_docs():
    exports = hf_publish.build_inventory_exports(ROOT, generated_at="2026-11-25T00:00:00Z")
    lb = exports["leaderboard_json"]
    assert lb["laws_meta"]["active_law_count"] >= 4
    assert lb["laws_meta"]["provisional_law_count"] >= 1
    law5 = next(law for law in lb["laws"] if law["id"] == "law5")
    assert law5["status"] == "provisional"

    catalog = experiment_docs.build_experiment_catalog(ROOT)
    hub_md = experiment_docs.render_laws_hub(ROOT, catalog)
    assert "| **[Law 5](#law5)** | `provisional` |" in hub_md
    assert "## Law 5 — Observation & Statistical-Metric Identifiability" in hub_md
