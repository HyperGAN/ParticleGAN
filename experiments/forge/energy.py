"""Observer-only hardware GPU NVML energy telemetry and J-to-PASS accounting.

Zero external dependencies: talks directly to ``libnvidia-ml.so.1`` via ctypes.
Strictly observer-only: never touches PyTorch RNG streams, model weights,
optimizer state, or CPU execution. When CUDA or NVML is unavailable (or denied
by permissions), every measurement degrades cleanly to ``None``.
"""
from __future__ import annotations

import ctypes
import math
import os
import statistics
import threading
import time
from typing import Any, Callable


ENERGY_BACKEND_COUNTER = "nvml_energy_counter"
ENERGY_BACKEND_POWER_FALLBACK = "nvml_power_fallback"

ENERGY_SUMMARY_KEYS = (
    "joules_total",
    "joules_per_step",
    "avg_power_watts",
    "first_pass_step",
    "sustained_pass_step",
    "joules_to_first_pass",
    "joules_to_sustained_pass",
    "gpu_name",
    "energy_backend",
    "contended",
)


def _finite_nonneg(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and f >= 0.0 else None


def _valid_step(value: Any, *, allow_zero: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if allow_zero:
        return value if value >= 0 else None
    return value if value > 0 else None


def resolve_nvml_device_index(
    device: int | str | None,
    *,
    cuda_visible_devices: str | None = None,
) -> int | str | None:
    """Resolve a PyTorch/Forge device selector to a physical NVML index or UUID.

    Returns ``None`` for CPU or unparseable selectors so callers never open NVML
    on non-CUDA runs.
    """
    if device is None:
        return None
    logical_idx: int | None = None
    if isinstance(device, bool):
        return None
    if isinstance(device, int):
        if device < 0:
            return None
        logical_idx = device
    elif isinstance(device, str):
        text = device.strip()
        lower = text.lower()
        if not text or lower == "cpu":
            return None
        if lower == "cuda":
            logical_idx = 0
        elif lower.startswith("cuda:"):
            suffix = text[5:].strip()
            if not suffix.isdigit():
                return None
            logical_idx = int(suffix)
        elif text.isdigit():
            logical_idx = int(text)
        elif text.startswith(("GPU-", "MIG-")):
            return text
        else:
            return None
    else:
        # Handle torch.device-like objects without importing torch.
        dev_type = getattr(device, "type", None)
        if dev_type != "cuda":
            return None
        idx = getattr(device, "index", None)
        logical_idx = 0 if idx is None else int(idx)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if cuda_visible_devices is None else cuda_visible_devices
    if visible is not None and visible.strip() != "":
        tokens = [tok.strip() for tok in visible.split(",") if tok.strip() != ""]
        if not tokens or logical_idx >= len(tokens):
            return None
        selected = tokens[logical_idx]
        if selected.isdigit():
            return int(selected)
        if selected.startswith(("GPU-", "MIG-")):
            return selected
        return None
    return logical_idx


def _default_nvml_loader() -> Any:
    return ctypes.CDLL("libnvidia-ml.so.1")


class _NVMLUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class EnergyMonitor:
    """Zero-dependency ctypes NVML hardware energy monitor with power fallback.

    Prefers the hardware millijoule accumulator (``nvmlDeviceGetTotalEnergyConsumption``).
    When the energy counter is unsupported or denied but power polling succeeds,
    falls back to background ``nvmlDeviceGetPowerUsage`` integration over wall time.
    """

    def __init__(
        self,
        device_index: int | str | None = 0,
        *,
        lib_loader: Callable[[], Any] | None = None,
        poll_interval_s: float = 0.05,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.device_index = device_index
        self.poll_interval_s = max(0.002, float(poll_interval_s))
        self._clock = clock or time.monotonic
        self.available = False
        self._nvml: Any = None
        self._handle = ctypes.c_void_p()
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._start_joules: float | None = None
        self._end_joules: float | None = None
        self._gpu_util_before: tuple[int, int] | None = None
        self._gpu_name: str | None = None
        self._power_samples: list[float] = []
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None
        self._energy_backend: str | None = None
        self._measured_joules: float | None = None

        if device_index is None or isinstance(device_index, bool):
            return
        if isinstance(device_index, int) and device_index < 0:
            return
        if isinstance(device_index, str) and not device_index.startswith(("GPU-", "MIG-")):
            return

        loader = lib_loader or _default_nvml_loader
        try:
            nvml = loader()
            if nvml is None:
                return
            if nvml.nvmlInit_v2() != 0:
                return
            rc = -1
            if isinstance(device_index, str):
                get_by_uuid = getattr(nvml, "nvmlDeviceGetHandleByUUID", None)
                if get_by_uuid is not None:
                    rc = get_by_uuid(device_index.encode("utf-8"), ctypes.byref(self._handle))
            else:
                rc = nvml.nvmlDeviceGetHandleByIndex_v2(int(device_index), ctypes.byref(self._handle))
            if rc == 0:
                self._nvml = nvml
                self.available = True
                self._gpu_name = self.device_name()
            else:
                try:
                    nvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            self.available = False
            self._nvml = None

    @classmethod
    def for_device(
        cls,
        device: int | str | None,
        *,
        cuda_visible_devices: str | None = None,
        lib_loader: Callable[[], Any] | None = None,
        poll_interval_s: float = 0.05,
        clock: Callable[[], float] | None = None,
    ) -> "EnergyMonitor":
        resolved = resolve_nvml_device_index(device, cuda_visible_devices=cuda_visible_devices)
        return cls(resolved, lib_loader=lib_loader, poll_interval_s=poll_interval_s, clock=clock)

    def device_name(self) -> str | None:
        if not self.available or self._nvml is None:
            return None
        try:
            fn = getattr(self._nvml, "nvmlDeviceGetName", None)
            if fn is None:
                return None
            buf = ctypes.create_string_buffer(128)
            if fn(self._handle, buf, len(buf)) == 0:
                text = buf.value.decode("utf-8", errors="replace").strip()
                return text or None
        except Exception:
            return None
        return None

    def total_energy_joules(self) -> float | None:
        if not self.available or self._nvml is None:
            return None
        try:
            fn = getattr(self._nvml, "nvmlDeviceGetTotalEnergyConsumption", None)
            if fn is None:
                return None
            mj = ctypes.c_ulonglong()
            if fn(self._handle, ctypes.byref(mj)) == 0:
                return float(mj.value) / 1000.0
        except Exception:
            return None
        return None

    def power_watts(self) -> float | None:
        if not self.available or self._nvml is None:
            return None
        try:
            fn = getattr(self._nvml, "nvmlDeviceGetPowerUsage", None)
            if fn is None:
                return None
            mw = ctypes.c_uint()
            if fn(self._handle, ctypes.byref(mw)) == 0:
                return float(mw.value) / 1000.0
        except Exception:
            return None
        return None

    def gpu_utilization(self) -> tuple[int, int] | None:
        if not self.available or self._nvml is None:
            return None
        try:
            fn = getattr(self._nvml, "nvmlDeviceGetUtilizationRates", None)
            if fn is None:
                return None
            util = _NVMLUtilization()
            if fn(self._handle, ctypes.byref(util)) == 0:
                return int(util.gpu), int(util.memory)
        except Exception:
            return None
        return None

    def _poll_power_loop(self) -> None:
        while not self._stop_event.is_set():
            watts = self.power_watts()
            if watts is not None:
                self._power_samples.append(watts)
            self._stop_event.wait(self.poll_interval_s)

    def start(self) -> "EnergyMonitor":
        self._started_at = self._clock()
        self._stopped_at = None
        self._measured_joules = None
        self._energy_backend = None
        self._power_samples = []
        if not self.available:
            return self
        self._gpu_util_before = self.gpu_utilization()
        self._start_joules = self.total_energy_joules()
        if self._start_joules is None:
            first_w = self.power_watts()
            if first_w is not None:
                self._power_samples.append(first_w)
                self._stop_event.clear()
                self._sampler_thread = threading.Thread(target=self._poll_power_loop, daemon=True)
                self._sampler_thread.start()
        return self

    def stop(self) -> float | None:
        self._stopped_at = self._clock()
        if self._sampler_thread is not None:
            self._stop_event.set()
            self._sampler_thread.join(timeout=1.0)
            self._sampler_thread = None
        if not self.available:
            return None
        wall_s = self.wall_seconds
        if self._start_joules is not None:
            self._end_joules = self.total_energy_joules()
            if self._end_joules is not None and self._end_joules >= self._start_joules:
                self._measured_joules = self._end_joules - self._start_joules
                self._energy_backend = ENERGY_BACKEND_COUNTER
                return self._measured_joules
        # Fallback: integrate sampled power usage over elapsed wall time.
        tail_w = self.power_watts()
        if tail_w is not None:
            self._power_samples.append(tail_w)
        if self._power_samples and wall_s is not None and wall_s > 0.0:
            avg_w = float(statistics.mean(self._power_samples))
            self._measured_joules = max(0.0, avg_w * wall_s)
            self._energy_backend = ENERGY_BACKEND_POWER_FALLBACK
            return self._measured_joules
        return None

    @property
    def wall_seconds(self) -> float | None:
        if self._started_at is None or self._stopped_at is None:
            return None
        return max(0.0, self._stopped_at - self._started_at)

    @property
    def inferred_contended(self) -> bool | None:
        if self._gpu_util_before is None:
            return None
        return bool(self._gpu_util_before[0] > 5)

    def summary(
        self,
        completed_steps: int | None,
        *,
        first_pass_step: int | None = None,
        sustained_pass_step: int | None = None,
        gpu_name: str | None = None,
        contended: bool | None = None,
        wall_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if not self.available or self._measured_joules is None:
            return None
        effective_wall = self.wall_seconds if wall_seconds is None else wall_seconds
        effective_gpu = gpu_name or self._gpu_name
        effective_contended = self.inferred_contended if contended is None else contended
        return compute_energy_summary(
            self._measured_joules,
            effective_wall,
            completed_steps,
            first_pass_step=first_pass_step,
            sustained_pass_step=sustained_pass_step,
            gpu_name=effective_gpu,
            contended=effective_contended,
            energy_backend=self._energy_backend,
        )

    def shutdown(self) -> None:
        if self._sampler_thread is not None:
            self._stop_event.set()
            self._sampler_thread.join(timeout=1.0)
            self._sampler_thread = None
        if self.available and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self.available = False
        self._nvml = None

    def __enter__(self) -> "EnergyMonitor":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()
        self.shutdown()


def compute_energy_summary(
    energy_joules_total: float | None,
    wall_seconds: float | None,
    completed_steps: int | None,
    first_pass_step: int | None = None,
    sustained_pass_step: int | None = None,
    gpu_name: str | None = None,
    contended: bool | None = None,
    energy_backend: str | None = None,
) -> dict[str, Any]:
    """Compute standardized hardware energy and J-to-PASS telemetry summary."""
    joules_total = _finite_nonneg(energy_joules_total)
    wall_s = _finite_nonneg(wall_seconds)
    steps = _valid_step(completed_steps, allow_zero=False)
    fp_step = _valid_step(first_pass_step, allow_zero=True)
    sp_step = _valid_step(sustained_pass_step, allow_zero=True)

    joules_per_step = (joules_total / float(steps)) if (joules_total is not None and steps is not None) else None
    avg_power_watts = (
        (joules_total / wall_s) if (joules_total is not None and wall_s is not None and wall_s > 0.0) else None
    )
    joules_to_first_pass = (
        (joules_per_step * float(fp_step)) if (joules_per_step is not None and fp_step is not None) else None
    )
    joules_to_sustained_pass = (
        (joules_per_step * float(sp_step)) if (joules_per_step is not None and sp_step is not None) else None
    )

    clean_gpu = gpu_name.strip() if isinstance(gpu_name, str) and gpu_name.strip() else None
    if isinstance(energy_backend, str) and energy_backend.strip():
        clean_backend: str | None = energy_backend.strip()
    elif joules_total is not None:
        clean_backend = ENERGY_BACKEND_COUNTER
    else:
        clean_backend = None
    clean_contended = contended if isinstance(contended, bool) else None

    return {
        "joules_total": joules_total,
        "joules_per_step": joules_per_step,
        "avg_power_watts": avg_power_watts,
        "first_pass_step": fp_step,
        "sustained_pass_step": sp_step,
        "joules_to_first_pass": joules_to_first_pass,
        "joules_to_sustained_pass": joules_to_sustained_pass,
        "gpu_name": clean_gpu,
        "energy_backend": clean_backend,
        "contended": clean_contended,
    }


def enrich_energy_summary(
    energy_summary: dict[str, Any] | None,
    *,
    first_pass_step: int | None = None,
    sustained_pass_step: int | None = None,
) -> dict[str, Any] | None:
    """Attach pass-step J-to-PASS values to an existing hardware energy summary."""
    if not isinstance(energy_summary, dict):
        return None
    j_per_step = _finite_nonneg(energy_summary.get("joules_per_step"))
    out = dict(energy_summary)
    fp = _valid_step(out.get("first_pass_step"), allow_zero=True)
    sp = _valid_step(out.get("sustained_pass_step"), allow_zero=True)
    if fp is None and first_pass_step is not None:
        fp = _valid_step(first_pass_step, allow_zero=True)
    if sp is None and sustained_pass_step is not None:
        sp = _valid_step(sustained_pass_step, allow_zero=True)
    out["first_pass_step"] = fp
    out["sustained_pass_step"] = sp
    if j_per_step is not None:
        out["joules_to_first_pass"] = (j_per_step * float(fp)) if fp is not None else None
        out["joules_to_sustained_pass"] = (j_per_step * float(sp)) if sp is not None else None
    return out


def _point_passes_thresholds(point: dict[str, Any], thresholds: list[Any]) -> bool:
    for item in thresholds:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return False
        metric, op, bound = item
        val = point.get(metric)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            return False
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(float(bound)):
            return False
        fval, fbound = float(val), float(bound)
        if op == "<=":
            ok = fval <= fbound
        elif op == ">=":
            ok = fval >= fbound
        elif op == "<":
            ok = fval < fbound
        elif op == ">":
            ok = fval > fbound
        elif op == "==":
            ok = fval == fbound
        else:
            return False
        if not ok:
            return False
    return True


def _extract_convergence_dict(container: Any) -> tuple[int | None, int | None]:
    if not isinstance(container, dict):
        return None, None
    conv = container.get("convergence")
    if isinstance(conv, dict):
        fp = _valid_step(conv.get("first_pass_step"), allow_zero=True)
        sp = _valid_step(
            conv.get("confirmed_step")
            if conv.get("confirmed_step") is not None
            else conv.get("stable_from_step"),
            allow_zero=True,
        )
        if fp is not None or sp is not None:
            return fp, sp
    fp = _valid_step(container.get("first_pass_step"), allow_zero=True)
    sp = _valid_step(
        container.get("sustained_pass_step")
        if container.get("sustained_pass_step") is not None
        else container.get("confirmed_step"),
        allow_zero=True,
    )
    return fp, sp


def extract_pass_steps(
    task: dict[str, Any] | None,
    raw_result: dict[str, Any] | None,
    grade: dict[str, Any] | None = None,
) -> tuple[int | None, int | None]:
    """Derive ``(first_pass_step, sustained_pass_step)`` from scheduled observations.

    Purely read-only inspection of already-recorded observations or evaluator
    summaries; never invokes models, evaluators, or RNG streams.
    """
    # 1. Prefer explicit convergence steps already computed by the evaluator/grade.
    for candidate in (
        (grade or {}).get("evaluator_summary") if isinstance(grade, dict) else None,
        (grade or {}).get("evaluator_result") if isinstance(grade, dict) else None,
        (raw_result or {}).get("evaluator_summary") if isinstance(raw_result, dict) else None,
    ):
        fp, sp = _extract_convergence_dict(candidate)
        if fp is not None or sp is not None:
            return fp, sp

    if not isinstance(raw_result, dict):
        return None, None
    evidence = raw_result.get("evidence")
    if not isinstance(evidence, dict):
        return None, None

    raw_points = evidence.get("observations")
    if not isinstance(raw_points, list) or not raw_points:
        raw_points = evidence.get("curve")
    if not isinstance(raw_points, list) or not raw_points:
        raw_points = evidence.get("dense")
    if not isinstance(raw_points, list) or not raw_points:
        return None, None

    evaluation = task.get("evaluation") if isinstance(task, dict) and isinstance(task.get("evaluation"), dict) else {}
    thresholds = evaluation.get("thresholds")
    min_stable_raw = evaluation.get("minimum_stable_checks", 5)
    min_stable = int(min_stable_raw) if isinstance(min_stable_raw, int) and min_stable_raw >= 1 else 5

    first_pass: int | None = None
    sustained_pass: int | None = None
    streak = 0

    for entry in raw_points:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        merged = {**nested, **{k: v for k, v in entry.items() if k != "metrics"}}
        step = _valid_step(merged.get("step"), allow_zero=True)
        if step is None:
            continue
        if isinstance(thresholds, list) and thresholds:
            passed = _point_passes_thresholds(merged, thresholds)
        elif isinstance(merged.get("passed"), bool):
            passed = merged["passed"]
        else:
            return None, None

        if passed:
            if first_pass is None:
                first_pass = step
            streak += 1
            if sustained_pass is None and streak >= min_stable:
                sustained_pass = step
        else:
            streak = 0

    return first_pass, sustained_pass
