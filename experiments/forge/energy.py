"""Observer-only hardware GPU NVML energy telemetry and J-to-PASS accounting.

Zero external dependencies: talks directly to ``libnvidia-ml.so.1`` via ctypes.
Strictly observer-only: never touches PyTorch RNG streams, model weights,
optimizer state, or CPU execution. When CUDA or NVML is unavailable (or denied
by permissions), every measurement degrades cleanly to ``None``.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
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
    "cumulative_joules_to_first_pass",
    "cumulative_joules_to_sustained_pass",
    "checkpoint_joules",
    "gpu_name",
    "energy_backend",
    "contended",
)

_ACTIVE_MONITOR_STATE = threading.local()


def active_energy_monitor() -> "EnergyMonitor | None":
    """Return the currently running thread-local ``EnergyMonitor``, if any."""
    stack = getattr(_ACTIVE_MONITOR_STATE, "stack", None)
    if stack:
        return stack[-1]
    return None


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


def _normalize_checkpoint_joules(raw: Any) -> dict[str, float] | None:
    """Normalize per-checkpoint cumulative Joules to ``{str(step): float(joules)}``."""
    if raw is None:
        return None
    parsed: list[tuple[int, float]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            step: int | None = None
            if isinstance(k, int) and not isinstance(k, bool):
                step = _valid_step(k, allow_zero=True)
            elif isinstance(k, str) and k.strip().isdigit():
                step = _valid_step(int(k.strip()), allow_zero=True)
            joules = _finite_nonneg(v)
            if step is not None and joules is not None:
                parsed.append((step, joules))
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                step = _valid_step(item.get("step"), allow_zero=True)
                joules = _finite_nonneg(
                    item.get("cumulative_joules")
                    if item.get("cumulative_joules") is not None
                    else item.get("joules")
                )
                if step is not None and joules is not None:
                    parsed.append((step, joules))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                step = _valid_step(item[0], allow_zero=True)
                joules = _finite_nonneg(item[1])
                if step is not None and joules is not None:
                    parsed.append((step, joules))
    if not parsed:
        return None
    parsed.sort(key=lambda pair: pair[0])
    out: dict[str, float] = {}
    for step, joules in parsed:
        out[str(step)] = joules
    return out or None


def _lookup_checkpoint_joules(checkpoints: dict[str, float] | None, step: int | None) -> float | None:
    if not isinstance(checkpoints, dict) or step is None:
        return None
    val = checkpoints.get(str(step))
    if val is None:
        val = checkpoints.get(step)  # type: ignore[arg-type]
    return _finite_nonneg(val)


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
        self._checkpoint_joules: dict[str, float] = {}
        self._registered_active = False

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

    def _push_active(self) -> None:
        if self._registered_active:
            return
        stack = getattr(_ACTIVE_MONITOR_STATE, "stack", None)
        if stack is None:
            stack = []
            _ACTIVE_MONITOR_STATE.stack = stack
        stack.append(self)
        self._registered_active = True

    def _pop_active(self) -> None:
        if not self._registered_active:
            return
        stack = getattr(_ACTIVE_MONITOR_STATE, "stack", None)
        if stack:
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx] is self:
                    stack.pop(idx)
                    break
        self._registered_active = False

    def elapsed_joules(self) -> float | None:
        """Read cumulative NVML Joules elapsed since ``start()`` without stopping."""
        if not self.available or self._started_at is None:
            return None
        if self._start_joules is not None:
            now_j = self.total_energy_joules()
            if now_j is not None and now_j >= self._start_joules:
                return now_j - self._start_joules
        elapsed_s = max(0.0, self._clock() - self._started_at)
        samples = list(self._power_samples)
        if not samples:
            now_w = self.power_watts()
            if now_w is not None:
                samples.append(now_w)
        if samples and elapsed_s >= 0.0:
            avg_w = float(statistics.mean(samples))
            return max(0.0, avg_w * elapsed_s)
        return None

    def record_checkpoint(self, step: int, *, overwrite: bool = False) -> float | None:
        """Record actual cumulative NVML Joules at an evaluation checkpoint step."""
        clean_step = _valid_step(step, allow_zero=True)
        if clean_step is None or not self.available:
            return None
        key = str(clean_step)
        if not overwrite and key in self._checkpoint_joules:
            return self._checkpoint_joules[key]
        joules = self.elapsed_joules()
        if joules is not None:
            self._checkpoint_joules[key] = joules
        return joules

    @property
    def checkpoint_joules(self) -> dict[str, float] | None:
        return dict(self._checkpoint_joules) if self._checkpoint_joules else None

    def start(self) -> "EnergyMonitor":
        self._started_at = self._clock()
        self._stopped_at = None
        self._measured_joules = None
        self._energy_backend = None
        self._power_samples = []
        self._checkpoint_joules = {}
        self._push_active()
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
        self._pop_active()
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
        checkpoint_joules: dict[int | str, float] | list[Any] | None = None,
        gpu_name: str | None = None,
        contended: bool | None = None,
        wall_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if not self.available or self._measured_joules is None:
            return None
        effective_wall = self.wall_seconds if wall_seconds is None else wall_seconds
        effective_gpu = gpu_name or self._gpu_name
        effective_contended = self.inferred_contended if contended is None else contended
        effective_checkpoints = checkpoint_joules if checkpoint_joules is not None else self.checkpoint_joules
        return compute_energy_summary(
            self._measured_joules,
            effective_wall,
            completed_steps,
            first_pass_step=first_pass_step,
            sustained_pass_step=sustained_pass_step,
            checkpoint_joules=effective_checkpoints,
            gpu_name=effective_gpu,
            contended=effective_contended,
            energy_backend=self._energy_backend,
        )

    def shutdown(self) -> None:
        self._pop_active()
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
    checkpoint_joules: dict[int | str, float] | list[Any] | None = None,
) -> dict[str, Any]:
    """Compute standardized hardware energy and J-to-PASS telemetry summary.

    When per-checkpoint cumulative NVML Joules (``checkpoint_joules``) cover
    ``first_pass_step`` / ``sustained_pass_step``, reports actual measured
    cumulative Joules in ``cumulative_joules_to_*`` and ``joules_to_*``.
    Falls back cleanly to linear extrapolation (``joules_per_step * pass_step``)
    only for legacy receipts lacking per-checkpoint Joules.
    """
    joules_total = _finite_nonneg(energy_joules_total)
    wall_s = _finite_nonneg(wall_seconds)
    steps = _valid_step(completed_steps, allow_zero=False)
    fp_step = _valid_step(first_pass_step, allow_zero=True)
    sp_step = _valid_step(sustained_pass_step, allow_zero=True)
    checkpoints = _normalize_checkpoint_joules(checkpoint_joules)

    joules_per_step = (joules_total / float(steps)) if (joules_total is not None and steps is not None) else None
    avg_power_watts = (
        (joules_total / wall_s) if (joules_total is not None and wall_s is not None and wall_s > 0.0) else None
    )
    cum_j_fp = _lookup_checkpoint_joules(checkpoints, fp_step)
    cum_j_sp = _lookup_checkpoint_joules(checkpoints, sp_step)

    if cum_j_fp is not None:
        joules_to_first_pass: float | None = cum_j_fp
    elif joules_per_step is not None and fp_step is not None:
        joules_to_first_pass = joules_per_step * float(fp_step)
    else:
        joules_to_first_pass = None

    if cum_j_sp is not None:
        joules_to_sustained_pass: float | None = cum_j_sp
    elif joules_per_step is not None and sp_step is not None:
        joules_to_sustained_pass = joules_per_step * float(sp_step)
    else:
        joules_to_sustained_pass = None

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
        "cumulative_joules_to_first_pass": cum_j_fp,
        "cumulative_joules_to_sustained_pass": cum_j_sp,
        "checkpoint_joules": checkpoints,
        "gpu_name": clean_gpu,
        "energy_backend": clean_backend,
        "contended": clean_contended,
    }


def enrich_energy_summary(
    energy_summary: dict[str, Any] | None,
    *,
    first_pass_step: int | None = None,
    sustained_pass_step: int | None = None,
    checkpoint_joules: dict[int | str, float] | list[Any] | None = None,
    completed_steps: int | None = None,
) -> dict[str, Any] | None:
    """Attach pass-step J-to-PASS values to an existing hardware energy summary."""
    if not isinstance(energy_summary, dict):
        return None
    out = dict(energy_summary)
    joules_total = _finite_nonneg(out.get("joules_total"))
    j_per_step = _finite_nonneg(out.get("joules_per_step"))
    steps = _valid_step(completed_steps, allow_zero=False)
    if j_per_step is None and joules_total is not None and steps is not None:
        j_per_step = joules_total / float(steps)
        out["joules_per_step"] = j_per_step

    existing_ckpts = _normalize_checkpoint_joules(out.get("checkpoint_joules"))
    extra_ckpts = _normalize_checkpoint_joules(checkpoint_joules)
    if existing_ckpts and extra_ckpts:
        merged_ckpts = _normalize_checkpoint_joules({**existing_ckpts, **extra_ckpts})
    else:
        merged_ckpts = extra_ckpts if extra_ckpts is not None else existing_ckpts
    if merged_ckpts is not None or "checkpoint_joules" in out:
        out["checkpoint_joules"] = merged_ckpts

    fp = _valid_step(out.get("first_pass_step"), allow_zero=True)
    sp = _valid_step(out.get("sustained_pass_step"), allow_zero=True)
    if fp is None and first_pass_step is not None:
        fp = _valid_step(first_pass_step, allow_zero=True)
    if sp is None and sustained_pass_step is not None:
        sp = _valid_step(sustained_pass_step, allow_zero=True)
    out["first_pass_step"] = fp
    out["sustained_pass_step"] = sp

    cum_fp = _lookup_checkpoint_joules(merged_ckpts, fp)
    if cum_fp is None:
        cum_fp = _finite_nonneg(out.get("cumulative_joules_to_first_pass"))
    cum_sp = _lookup_checkpoint_joules(merged_ckpts, sp)
    if cum_sp is None:
        cum_sp = _finite_nonneg(out.get("cumulative_joules_to_sustained_pass"))
    if cum_fp is not None or "cumulative_joules_to_first_pass" in out:
        out["cumulative_joules_to_first_pass"] = cum_fp
    if cum_sp is not None or "cumulative_joules_to_sustained_pass" in out:
        out["cumulative_joules_to_sustained_pass"] = cum_sp

    if cum_fp is not None:
        out["joules_to_first_pass"] = cum_fp
    elif j_per_step is not None:
        out["joules_to_first_pass"] = (j_per_step * float(fp)) if fp is not None else None

    if cum_sp is not None:
        out["joules_to_sustained_pass"] = cum_sp
    elif j_per_step is not None:
        out["joules_to_sustained_pass"] = (j_per_step * float(sp)) if sp is not None else None

    return out


def extract_completed_steps(raw_result: dict[str, Any] | None) -> int | None:
    """Read-only extraction of completed training steps across Forge adapters."""
    if not isinstance(raw_result, dict):
        return None
    cost = raw_result.get("cost")
    if isinstance(cost, dict):
        for key in ("completed_steps", "completed_updates"):
            step = _valid_step(cost.get(key), allow_zero=False)
            if step is not None:
                return step
    evidence = raw_result.get("evidence")
    if isinstance(evidence, dict):
        prov_ckpt = evidence.get("provenance_checkpoint")
        if isinstance(prov_ckpt, dict):
            step = _valid_step(prov_ckpt.get("completed_steps"), allow_zero=False)
            if step is not None:
                return step
        guards = evidence.get("guards")
        if isinstance(guards, dict) and isinstance(guards.get("optimizer_updates"), dict):
            positive = [
                v for v in guards["optimizer_updates"].values()
                if isinstance(v, int) and not isinstance(v, bool) and v > 0
            ]
            if positive:
                return min(positive)
    metrics = raw_result.get("metrics")
    if isinstance(metrics, dict):
        for key in ("completed_steps", "completed_updates"):
            step = _valid_step(metrics.get(key), allow_zero=False)
            if step is not None:
                return step
    raw = raw_result.get("raw")
    if isinstance(raw, dict):
        for key in ("completed_steps", "steps"):
            step = _valid_step(raw.get(key), allow_zero=False)
            if step is not None:
                return step
    return None


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _point_passes_native_accuracy(entry: dict[str, Any], merged: dict[str, Any], evaluation: dict[str, Any]) -> bool | None:
    """Evaluate a ``native100`` / ``native_accuracy`` eval event against declared gates.

    Supports both pre-computed ``entry["accuracy"]`` dicts (with ``passed`` or
    ``frozen_pass`` + ``accuracy_pass``) and raw metric dicts evaluated against
    ``evaluation["coverage_thresholds"]`` and ``evaluation["accuracy_limits"]``
    (plus optional ``max_saturated_fraction``).
    """
    max_sat = _finite_number(evaluation.get("max_saturated_fraction"))
    if max_sat is not None:
        sat = _finite_number(merged.get("saturated_fraction"))
        if sat is None or sat > max_sat:
            return False

    acc = entry.get("accuracy")
    if isinstance(acc, dict) and acc:
        if isinstance(acc.get("passed"), bool):
            return acc["passed"]
        if isinstance(acc.get("frozen_pass"), bool) and isinstance(acc.get("accuracy_pass"), bool):
            return bool(acc["frozen_pass"] and acc["accuracy_pass"])
        if isinstance(acc.get("frozen_pass"), bool) and isinstance(evaluation.get("accuracy_limits"), dict):
            if not acc["frozen_pass"]:
                return False
            for key, bound in evaluation["accuracy_limits"].items():
                val = _finite_number(acc.get(key))
                fbound = _finite_number(bound)
                if val is None or fbound is None or val > fbound:
                    return False
            return True

    cov_thresholds = evaluation.get("coverage_thresholds")
    acc_limits = evaluation.get("accuracy_limits")
    if not isinstance(cov_thresholds, dict) and not isinstance(acc_limits, dict):
        return None

    if isinstance(cov_thresholds, dict) and cov_thresholds:
        n = _finite_number(merged.get("n"))
        valid_n = _finite_number(merged.get("valid_n"))
        if cov_thresholds.get("all_finite") is True:
            if merged.get("all_finite") is False:
                return False
            if n is not None and valid_n is not None and valid_n != n:
                return False
        min_samples = _finite_number(cov_thresholds.get("min_samples"))
        if min_samples is not None and n is not None and n < min_samples:
            return False
        cov_checks = (
            ("modes", ">=", cov_thresholds.get("min_modes")),
            ("min_hq_mode_mass", ">=", cov_thresholds.get("min_hq_mode_mass")),
            ("precision", ">=", cov_thresholds.get("min_precision")),
            ("mass_tv", "<=", cov_thresholds.get("max_mass_tv")),
            ("max_mode_mass", "<=", cov_thresholds.get("max_mode_mass")),
            ("min_cov_eig_ratio", ">=", cov_thresholds.get("min_cov_eig_ratio")),
            ("max_cov_eig_ratio", "<=", cov_thresholds.get("max_cov_eig_ratio")),
            ("min_radial_median_ratio", ">=", cov_thresholds.get("min_radial_median_ratio")),
            ("max_radial_median_ratio", "<=", cov_thresholds.get("max_radial_median_ratio")),
        )
        checked_any = False
        for metric_key, op, bound in cov_checks:
            fbound = _finite_number(bound)
            if fbound is None:
                continue
            val = _finite_number(merged.get(metric_key))
            if val is None:
                return False
            checked_any = True
            if op == ">=" and val < fbound - 1e-12:
                return False
            if op == "<=" and val > fbound + 1e-12:
                return False
        if not checked_any and not (isinstance(acc_limits, dict) and acc_limits):
            return None

    if isinstance(acc_limits, dict) and acc_limits:
        # Check accuracy limits whenever those fields are present in the event.
        # On early coverage-only eval rows (before accuracy fields are recorded),
        # fall through to coverage only if no non-mass_tv accuracy keys exist.
        acc_source = acc if isinstance(acc, dict) else merged
        extra_keys = [k for k in acc_limits if k != "mass_tv"]
        has_accuracy_fields = all(_finite_number(acc_source.get(k)) is not None for k in extra_keys) if extra_keys else True
        if has_accuracy_fields:
            for key, bound in acc_limits.items():
                fbound = _finite_number(bound)
                val = _finite_number(acc_source.get(key))
                if fbound is None:
                    continue
                if val is None or val > fbound + 1e-12:
                    return False
        elif not (isinstance(cov_thresholds, dict) and cov_thresholds):
            return None

    return True


def _load_native_events_jsonl(evidence: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Read live eval events from a ``native100`` artifact's ``events.jsonl`` (read-only)."""
    candidates: list[Path] = []
    for key in ("events_path", "events_jsonl"):
        raw = evidence.get(key)
        if isinstance(raw, str) and raw.strip():
            candidates.append(Path(raw.strip()))
    artifact_root = evidence.get("artifact_root")
    problem = evidence.get("problem")
    if isinstance(artifact_root, str) and artifact_root.strip() and isinstance(problem, str) and problem.strip():
        candidates.append(Path(artifact_root.strip()) / problem.strip() / "events.jsonl")
    for path in candidates:
        if not path.is_file():
            continue
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                if item.get("event") == "train":
                    continue
                model = item.get("model")
                if model is not None and model != "live":
                    continue
                rows.append(item)
        except Exception:
            continue
        if rows:
            return rows
    return None


def extract_pass_steps(
    task: dict[str, Any] | None,
    raw_result: dict[str, Any] | None,
    grade: dict[str, Any] | None = None,
) -> tuple[int | None, int | None]:
    """Derive ``(first_pass_step, sustained_pass_step)`` from scheduled observations.

    Purely read-only inspection of already-recorded observations or evaluator
    summaries (including ``native100`` / ``native_accuracy`` event histories);
    never invokes models, evaluators, or RNG streams.
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

    evaluation = task.get("evaluation") if isinstance(task, dict) and isinstance(task.get("evaluation"), dict) else {}
    adapter_name = task.get("adapter") if isinstance(task, dict) else None
    is_native_accuracy = (
        evaluation.get("kind") == "native_accuracy"
        or adapter_name in {"native100", "native100_continuation"}
    )

    evidence = raw_result.get("evidence") if isinstance(raw_result, dict) else None
    raw_points: Any = None
    if isinstance(evidence, dict):
        for key in ("observations", "curve", "dense", "events", "eval_events", "history"):
            val = evidence.get(key)
            if isinstance(val, list) and val:
                raw_points = val
                break
        if raw_points is None and is_native_accuracy:
            raw_points = _load_native_events_jsonl(evidence)

    # Fallback for native_accuracy grades that expose terminal_checks when events.jsonl is absent.
    if raw_points is None and isinstance(grade, dict):
        ev_res = grade.get("evaluator_result")
        if isinstance(ev_res, dict):
            for key in ("observations", "eval_events", "terminal_checks"):
                val = ev_res.get(key)
                if isinstance(val, list) and val:
                    raw_points = val
                    break

    if not isinstance(raw_points, list) or not raw_points:
        return None, None

    thresholds = evaluation.get("thresholds")
    min_stable_raw = evaluation.get("minimum_stable_checks", 5)
    min_stable = int(min_stable_raw) if isinstance(min_stable_raw, int) and min_stable_raw >= 1 else 5

    first_pass: int | None = None
    sustained_pass: int | None = None
    streak = 0

    for entry in raw_points:
        if not isinstance(entry, dict):
            continue
        if entry.get("event") == "train":
            continue
        model = entry.get("model")
        if model is not None and model != "live":
            continue
        nested = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        acc_nested = entry.get("accuracy") if isinstance(entry.get("accuracy"), dict) else {}
        merged = {
            **acc_nested,
            **nested,
            **{k: v for k, v in entry.items() if k not in ("metrics", "accuracy")},
        }
        step = _valid_step(merged.get("step"), allow_zero=not is_native_accuracy)
        if step is None:
            continue
        if isinstance(thresholds, list) and thresholds:
            passed = _point_passes_thresholds(merged, thresholds)
        elif is_native_accuracy or isinstance(evaluation.get("coverage_thresholds"), dict) or isinstance(evaluation.get("accuracy_limits"), dict):
            verdict = _point_passes_native_accuracy(entry, merged, evaluation)
            if verdict is None:
                if isinstance(merged.get("passed"), bool):
                    passed = merged["passed"]
                else:
                    return None, None
            else:
                passed = verdict
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
