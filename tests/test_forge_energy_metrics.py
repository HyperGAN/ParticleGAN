"""Unit tests for observer-only Forge NVML hardware energy telemetry and J-to-PASS."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters
from experiments.forge.energy import (
    ENERGY_BACKEND_COUNTER,
    ENERGY_BACKEND_POWER_FALLBACK,
    ENERGY_SUMMARY_KEYS,
    EnergyMonitor,
    compute_energy_summary,
    enrich_energy_summary,
    extract_pass_steps,
    resolve_nvml_device_index,
)


class _FakeNVML:
    """Mock ctypes NVML library supporting energy counters, power polling, and failure modes."""

    def __init__(
        self,
        *,
        init_rc: int = 0,
        handle_rc: int = 0,
        energy_mj_seq: list[int] | None = None,
        energy_rc: int = 0,
        power_mw_seq: list[int] | None = None,
        power_rc: int = 0,
        gpu_name: bytes = b"NVIDIA RTX A6000",
        util: tuple[int, int] = (0, 0),
    ) -> None:
        self.init_rc = init_rc
        self.handle_rc = handle_rc
        self.energy_mj_seq = list(energy_mj_seq if energy_mj_seq is not None else [1_000_000, 1_500_000])
        self.energy_rc = energy_rc
        self.power_mw_seq = list(power_mw_seq if power_mw_seq is not None else [250_000, 250_000])
        self.power_rc = power_rc
        self.gpu_name = gpu_name
        self.util = util
        self.shutdown_calls = 0
        self.energy_calls = 0
        self.power_calls = 0
        self.seen_indices: list[int] = []

    def nvmlInit_v2(self) -> int:
        return self.init_rc

    def nvmlShutdown(self) -> int:
        self.shutdown_calls += 1
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, idx: int, handle_ptr) -> int:
        self.seen_indices.append(idx)
        if self.handle_rc == 0:
            handle_ptr._obj.value = 0x100 + idx
        return self.handle_rc

    def nvmlDeviceGetName(self, handle, buf, length: int) -> int:
        buf.value = self.gpu_name[: length - 1]
        return 0

    def nvmlDeviceGetTotalEnergyConsumption(self, handle, mj_ptr) -> int:
        if self.energy_rc != 0:
            return self.energy_rc
        val = self.energy_mj_seq[min(self.energy_calls, len(self.energy_mj_seq) - 1)]
        self.energy_calls += 1
        mj_ptr._obj.value = val
        return 0

    def nvmlDeviceGetPowerUsage(self, handle, mw_ptr) -> int:
        if self.power_rc != 0:
            return self.power_rc
        val = self.power_mw_seq[min(self.power_calls, len(self.power_mw_seq) - 1)]
        self.power_calls += 1
        mw_ptr._obj.value = val
        return 0

    def nvmlDeviceGetUtilizationRates(self, handle, util_ptr) -> int:
        util_ptr._obj.gpu = self.util[0]
        util_ptr._obj.memory = self.util[1]
        return 0


def test_resolve_nvml_device_index_handles_cpu_cuda_and_visible_devices():
    assert resolve_nvml_device_index(None) is None
    assert resolve_nvml_device_index("cpu") is None
    assert resolve_nvml_device_index("CPU") is None
    assert resolve_nvml_device_index("mps") is None
    assert resolve_nvml_device_index(False) is None
    assert resolve_nvml_device_index(-1) is None

    assert resolve_nvml_device_index(0, cuda_visible_devices="") == 0
    assert resolve_nvml_device_index("cuda", cuda_visible_devices="") == 0
    assert resolve_nvml_device_index("cuda:1", cuda_visible_devices="") == 1
    assert resolve_nvml_device_index("2", cuda_visible_devices="") == 2
    assert resolve_nvml_device_index("cuda:0", cuda_visible_devices="1,3") == 1
    assert resolve_nvml_device_index("cuda:1", cuda_visible_devices="1,3") == 3
    assert resolve_nvml_device_index("cuda:2", cuda_visible_devices="1,3") is None
    assert resolve_nvml_device_index("cuda:0", cuda_visible_devices="GPU-abc123") == "GPU-abc123"
    assert resolve_nvml_device_index(torch.device("cpu")) is None
    assert resolve_nvml_device_index(torch.device("cuda:0"), cuda_visible_devices="2") == 2


def test_energy_monitor_counter_mode_and_j_to_pass():
    fake = _FakeNVML(
        energy_mj_seq=[10_000_000, 14_566_565],  # 4566.565 J delta
        util=(0, 0),
    )
    ticks = iter([100.0, 149.737])
    mon = EnergyMonitor(0, lib_loader=lambda: fake, clock=lambda: next(ticks))
    assert mon.available is True
    mon.start()
    delta = mon.stop()
    assert delta == pytest.approx(4566.565, rel=1e-6)

    summary = mon.summary(completed_steps=1600, first_pass_step=217, sustained_pass_step=284)
    mon.shutdown()

    assert summary is not None
    assert set(summary) == set(ENERGY_SUMMARY_KEYS)
    assert summary["energy_backend"] == ENERGY_BACKEND_COUNTER
    assert summary["gpu_name"] == "NVIDIA RTX A6000"
    assert summary["contended"] is False
    assert summary["joules_total"] == pytest.approx(4566.565, rel=1e-6)
    assert summary["joules_per_step"] == pytest.approx(4566.565 / 1600.0, rel=1e-6)
    assert summary["avg_power_watts"] == pytest.approx(4566.565 / 49.737, rel=1e-6)
    assert summary["first_pass_step"] == 217
    assert summary["sustained_pass_step"] == 284
    assert summary["joules_to_first_pass"] == pytest.approx((4566.565 / 1600.0) * 217, rel=1e-6)
    assert summary["joules_to_sustained_pass"] == pytest.approx((4566.565 / 1600.0) * 284, rel=1e-6)
    assert fake.shutdown_calls == 1


def test_energy_monitor_power_polling_fallback_when_counter_denied():
    fake = _FakeNVML(
        energy_rc=3,  # NVML_ERROR_NO_PERMISSION / unsupported
        power_mw_seq=[200_000, 220_000, 240_000],  # 200W, 220W, 240W
        util=(42, 30),  # contended GPU
    )
    ticks = iter([10.0, 20.0])  # 10.0 s window
    mon = EnergyMonitor(0, lib_loader=lambda: fake, poll_interval_s=0.01, clock=lambda: next(ticks))
    assert mon.available is True
    mon.start()
    delta = mon.stop()
    summary = mon.summary(completed_steps=100, first_pass_step=25)
    mon.shutdown()

    assert delta is not None and delta > 0
    assert summary is not None
    assert summary["energy_backend"] == ENERGY_BACKEND_POWER_FALLBACK
    assert summary["contended"] is True
    assert 2000.0 <= summary["joules_total"] <= 2400.0
    assert summary["joules_to_first_pass"] == pytest.approx(summary["joules_per_step"] * 25, rel=1e-6)
    assert summary["joules_to_sustained_pass"] is None


def test_energy_monitor_graceful_fallback_on_cpu_missing_lib_or_error():
    # CPU never even touches lib_loader.
    called = []

    def forbid_loader():
        called.append(True)
        raise AssertionError("CPU monitor must never load NVML")

    cpu_mon = EnergyMonitor.for_device("cpu", lib_loader=forbid_loader)
    assert cpu_mon.available is False
    assert called == []
    with cpu_mon:
        pass
    assert cpu_mon.summary(100) is None

    # Missing shared library OSError degrades cleanly.
    def missing_lib():
        raise OSError("libnvidia-ml.so.1: cannot open shared object file")

    no_lib = EnergyMonitor(0, lib_loader=missing_lib)
    assert no_lib.available is False
    assert no_lib.total_energy_joules() is None
    assert no_lib.power_watts() is None
    assert no_lib.summary(100) is None

    # Permission error on nvmlInit_v2 degrades cleanly.
    denied = EnergyMonitor(0, lib_loader=lambda: _FakeNVML(init_rc=4))
    assert denied.available is False
    assert denied.summary(100) is None


def test_compute_energy_summary_math_and_invalid_inputs():
    summary = compute_energy_summary(
        35667.682,
        238.695,
        7000,
        first_pass_step=1500,
        sustained_pass_step=2500,
        gpu_name="NVIDIA RTX A6000",
        contended=False,
    )
    assert set(summary) == set(ENERGY_SUMMARY_KEYS)
    assert summary["joules_per_step"] == pytest.approx(35667.682 / 7000.0)
    assert summary["avg_power_watts"] == pytest.approx(35667.682 / 238.695)
    assert summary["joules_to_first_pass"] == pytest.approx((35667.682 / 7000.0) * 1500)
    assert summary["joules_to_sustained_pass"] == pytest.approx((35667.682 / 7000.0) * 2500)
    assert summary["energy_backend"] == ENERGY_BACKEND_COUNTER

    # Null / NaN / zero-step inputs must never fabricate numbers.
    empty = compute_energy_summary(None, 10.0, 100)
    assert all(empty[k] is None for k in ENERGY_SUMMARY_KEYS)

    nan_case = compute_energy_summary(math.nan, 0.0, 0, first_pass_step=-5)
    assert nan_case["joules_total"] is None
    assert nan_case["joules_per_step"] is None
    assert nan_case["avg_power_watts"] is None
    assert nan_case["first_pass_step"] is None
    assert nan_case["joules_to_first_pass"] is None


def test_extract_pass_steps_and_enrich_summary():
    task = {
        "evaluation": {
            "minimum_stable_checks": 3,
            "thresholds": [["modes", ">=", 16], ["hq", ">=", 0.9]],
        }
    }
    raw = {
        "evidence": {
            "observations": [
                {"step": 100, "modes": 12, "hq": 0.70},
                {"step": 200, "modes": 16, "hq": 0.92},  # first pass
                {"step": 300, "modes": 15, "hq": 0.88},  # streak reset
                {"step": 400, "modes": 16, "hq": 0.93},
                {"step": 500, "modes": 16, "hq": 0.95},
                {"step": 600, "modes": 16, "hq": 0.97},  # 3rd consecutive pass -> sustained
            ]
        }
    }
    fp, sp = extract_pass_steps(task, raw)
    assert fp == 200
    assert sp == 600

    base = compute_energy_summary(1200.0, 12.0, 600, gpu_name="RTX A6000", contended=False)
    enriched = enrich_energy_summary(base, first_pass_step=fp, sustained_pass_step=sp)
    assert enriched["joules_per_step"] == pytest.approx(2.0)
    assert enriched["joules_to_first_pass"] == pytest.approx(400.0)
    assert enriched["joules_to_sustained_pass"] == pytest.approx(1200.0)

    # Evaluator-provided convergence steps take precedence when present.
    grade = {
        "evaluator_summary": {
            "convergence": {"first_pass_step": 150, "confirmed_step": 450},
        }
    }
    assert extract_pass_steps(task, raw, grade) == (150, 450)


def test_run_task_energy_telemetry_is_strictly_observer_only(tmp_path: Path, monkeypatch):
    """Attaching NVML monitoring around run_task must have zero effect on RNG or outputs."""
    task = {
        "id": "synthetic_vector",
        "evaluation": {
            "minimum_stable_checks": 2,
            "thresholds": [["hq", ">=", 0.9]],
        },
    }
    request = {"tasks": {"synthetic_vector": task}}
    job = {"task_id": "synthetic_vector"}

    def fake_dispatch(req, jb, out_dir, dev):
        val = float(torch.randn(4).sum().item())
        return {
            "cost": {"completed_steps": 100, "wall_seconds": 2.0},
            "evidence": {
                "rng_probe": val,
                "observations": [
                    {"step": 50, "hq": 0.91},
                    {"step": 100, "hq": 0.95},
                ],
            },
        }

    monkeypatch.setattr(adapters, "_dispatch_task", fake_dispatch)

    # Run 1: CPU (no NVML, no cost["energy"] attached)
    torch.manual_seed(12345)
    out_cpu = tmp_path / "cpu"
    out_cpu.mkdir()
    res_cpu = adapters.run_task(request, job, out_cpu, "cpu")
    rng_after_cpu = torch.get_rng_state().clone()
    assert "energy" not in res_cpu["cost"]

    # Run 2: Mocked NVML active on cuda:0
    fake_nvml = _FakeNVML(energy_mj_seq=[5_000_000, 5_400_000], util=(0, 0))
    ticks = iter([1.0, 3.0])
    monkeypatch.setattr(
        adapters.EnergyMonitor,
        "for_device",
        classmethod(lambda cls, dev, **kw: cls(0, lib_loader=lambda: fake_nvml, clock=lambda: next(ticks))),
    )
    torch.manual_seed(12345)
    out_gpu = tmp_path / "gpu"
    out_gpu.mkdir()
    res_gpu = adapters.run_task(request, job, out_gpu, "cuda:0")
    rng_after_gpu = torch.get_rng_state().clone()

    # Exact RNG and scientific output invariance!
    assert torch.equal(rng_after_cpu, rng_after_gpu)
    assert res_gpu["evidence"] == res_cpu["evidence"]

    # Hardware energy & J-to-PASS automatically recorded in cost["energy"]!
    energy = res_gpu["cost"]["energy"]
    assert energy["joules_total"] == pytest.approx(400.0)
    assert energy["joules_per_step"] == pytest.approx(4.0)
    assert energy["avg_power_watts"] == pytest.approx(200.0)
    assert energy["first_pass_step"] == 50
    assert energy["sustained_pass_step"] == 100
    assert energy["joules_to_first_pass"] == pytest.approx(200.0)
    assert energy["joules_to_sustained_pass"] == pytest.approx(400.0)
    assert energy["energy_backend"] == ENERGY_BACKEND_COUNTER
    assert energy["contended"] is False
