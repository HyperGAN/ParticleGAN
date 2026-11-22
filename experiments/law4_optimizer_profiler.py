#!/usr/bin/env python3
"""Standalone public-API profiling, energy, and mechanistic telemetry for Law 4 optimizer studies.

Zero edits to ``particlegan/_unified_v11/`` or any core formulation package.
Uses NVML hardware energy counters (``nvmlDeviceGetTotalEnergyConsumption`` in mJ)
and high-frequency power polling (``nvmlDeviceGetPowerUsage`` in mW), synchronized
CUDA events for phase breakdowns, and non-destructive bit-exact parameter probes
for mechanistic gradient/impedance diagnostics.

Usage examples:
  python experiments/law4_optimizer_profiler.py --mode dcgan-profile --device cuda:0 \\
      --output reports/forge/studies/law4-phase1-dcgan-profile.json
  python experiments/law4_optimizer_profiler.py --mode mechanistic --device cuda:0 \\
      --output reports/forge/studies/law4-phase1-mechanistic.json
  python experiments/law4_optimizer_profiler.py --mode energy-benchmark --device cuda:0 \\
      --ideas unified-v11 --output reports/forge/studies/law4-phase2-energy-benchmark.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import json
import math
import statistics
import threading
import time
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

ROOT = Path(os.environ.get("PARTICLEGAN_PROFILER_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
if str(ROOT) not in sys.path or sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Hardware NVML Energy & Power Monitor (zero external pip dependencies)
# ---------------------------------------------------------------------------


class NVMLMonitor:
    """Direct ctypes interface to libnvidia-ml.so.1 for exact mJ energy & W power."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)
        self.available = False
        self._nvml = None
        self._handle = ctypes.c_void_p()
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            if nvml.nvmlInit_v2() == 0:
                if nvml.nvmlDeviceGetHandleByIndex_v2(self.device_index, ctypes.byref(self._handle)) == 0:
                    self._nvml = nvml
                    self.available = True
        except OSError:
            self.available = False

    def power_watts(self) -> Optional[float]:
        if not self.available or self._nvml is None:
            return None
        mw = ctypes.c_uint()
        if self._nvml.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(mw)) == 0:
            return float(mw.value) / 1000.0
        return None

    def total_energy_joules(self) -> Optional[float]:
        if not self.available or self._nvml is None:
            return None
        mj = ctypes.c_ulonglong()
        if self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(mj)) == 0:
            return float(mj.value) / 1000.0
        return None

    def gpu_utilization(self) -> Optional[Tuple[int, int]]:
        if not self.available or self._nvml is None:
            return None

        class _Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        u = _Util()
        if self._nvml.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(u)) == 0:
            return int(u.gpu), int(u.memory)
        return None

    def measure_idle_power(self, duration_s: float = 0.4, interval_s: float = 0.01) -> Optional[float]:
        if not self.available:
            return None
        samples: List[float] = []
        end_t = time.perf_counter() + duration_s
        while time.perf_counter() < end_t:
            w = self.power_watts()
            if w is not None:
                samples.append(w)
            time.sleep(interval_s)
        return float(statistics.median(samples)) if samples else None

    def shutdown(self) -> None:
        if self.available and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self.available = False


class PowerSampler:
    """High-frequency background NVML power sampler during a sustained loop."""

    def __init__(self, monitor: NVMLMonitor, interval_s: float = 0.005) -> None:
        self.monitor = monitor
        self.interval_s = float(interval_s)
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            w = self.monitor.power_watts()
            if w is not None:
                self.samples.append(w)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PowerSampler":
        if self.monitor.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Helpers: Idea Loading, Optimizer State Accounting, CUDA Timing
# ---------------------------------------------------------------------------


def load_idea(idea_id: str) -> dict:
    path = ROOT / "configs" / "forge" / "ideas" / f"{idea_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Forge idea not found: {path}")
    return json.loads(path.read_text())


def load_task(task_id: str) -> dict:
    path = ROOT / "configs" / "forge" / "tasks" / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Forge task not found: {path}")
    return json.loads(path.read_text())


def _state_bytes_for_optimizer(opt: torch.optim.Optimizer) -> Dict[str, Any]:
    """Compute exact persistent tensor state bytes in opt.state (excluding scalar Python ints)."""
    param_elems = 0
    param_bytes = 0
    state_tensor_bytes = 0
    state_scalar_tensor_bytes = 0
    per_key_bytes: Dict[str, int] = {}
    seen_params: set[int] = set()
    for grp in opt.param_groups:
        for p in grp["params"]:
            if id(p) in seen_params or not p.requires_grad:
                continue
            seen_params.add(id(p))
            param_elems += int(p.numel())
            param_bytes += int(p.numel() * p.element_size())
            st = opt.state.get(p, {})
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    b = int(v.numel() * v.element_size())
                    if v.numel() <= 1:
                        state_scalar_tensor_bytes += b
                    else:
                        state_tensor_bytes += b
                    per_key_bytes[str(k)] = per_key_bytes.get(str(k), 0) + b
    total_state_bytes = state_tensor_bytes + state_scalar_tensor_bytes
    return {
        "param_count": param_elems,
        "param_bytes": param_bytes,
        "state_tensor_bytes": state_tensor_bytes,
        "state_scalar_tensor_bytes": state_scalar_tensor_bytes,
        "total_state_bytes": total_state_bytes,
        "state_bytes_per_param": (float(total_state_bytes) / float(param_elems)) if param_elems else 0.0,
        "per_key_bytes": per_key_bytes,
    }


def optimizer_state_summary(trainer: Any) -> Dict[str, Any]:
    g_info = _state_bytes_for_optimizer(trainer.opt_g)
    d_info = _state_bytes_for_optimizer(trainer.opt_d)
    total_params = g_info["param_count"] + d_info["param_count"]
    total_state = g_info["total_state_bytes"] + d_info["total_state_bytes"]
    return {
        "opt_g_and_prior": g_info,
        "opt_d": d_info,
        "total_trainable_params": total_params,
        "total_state_bytes": total_state,
        "total_state_mib": float(total_state) / float(2**20),
        "state_bytes_per_param": (float(total_state) / float(total_params)) if total_params else 0.0,
    }


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(vals)
    idx95 = min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)
    return {
        "mean": float(statistics.mean(s)),
        "median": float(statistics.median(s)),
        "p95": float(s[idx95]),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


# ---------------------------------------------------------------------------
# Phase 1A: Exact DCGAN Step-Time, Phase Breakdown & Hardware Energy Profiler
# ---------------------------------------------------------------------------


def _build_dcgan_trainer(
    idea: dict,
    task: dict,
    device: torch.device,
    total_steps: int,
    *,
    optimizer_family_override: Optional[str] = None,
    recipe_extra: Optional[Dict[str, Any]] = None,
):
    from experiments.forge.api import task_formulation_context
    from experiments.forge.scaling_benchmark import DCGANCritic, DCGANGenerator

    cand = deepcopy(idea)
    if optimizer_family_override is not None:
        cand.setdefault("recipe_overrides", {})["optimizer_family"] = optimizer_family_override
    if recipe_extra:
        cand.setdefault("recipe_overrides", {}).update(recipe_extra)

    spec = deepcopy(task["execution"]["performance"])
    task_copy = deepcopy(task)
    task_copy["execution"]["steps"] = int(total_steps)
    task_copy["execution"]["performance"]["measured_updates"] = max(1, int(total_steps) - int(spec["warmup_updates"]))

    context = task_formulation_context(cand, task_copy, {"seed": 0}, device=str(device))
    g = context.construct(
        lambda: DCGANGenerator(spec["image_size"], spec["latent_dim"], spec["base_channels"], spec["max_channels"]),
        component="generator",
    ).to(device)
    d = context.construct(
        lambda: DCGANCritic(spec["image_size"], spec["base_channels"], spec["max_channels"]),
        component="discriminator",
    ).to(device)
    trainer = context.build_trainer(g, d, max_steps=total_steps)
    return context, trainer, g, d, spec


@contextmanager
def _instrument_step_phases(trainer: Any, device: torch.device, phase_records: Dict[str, List[float]]):
    """Non-invasively wrap public trainer & optimizer methods with synchronized CUDA events.

    Restores every attribute cleanly on exit; zero edits to particlegan source files.
    """
    import particlegan._unified_v11.optimizer as v11_opt_mod

    state = {
        "active_opt": None,  # "d" or "g"
        "cur": {},
    }

    def _cuda_time(fn: Callable[..., Any], bucket_key: str) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if device.type == "cuda":
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                out = fn(*args, **kwargs)
                e1.record()
                torch.cuda.synchronize(device)
                ms = float(e0.elapsed_time(e1))
            else:
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                ms = (time.perf_counter() - t0) * 1000.0
            state["cur"][bucket_key] = state["cur"].get(bucket_key, 0.0) + ms
            return out

        return _wrapped

    def _cuda_time_opt_sub(fn: Callable[..., Any], sub_name: str) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            active = state["active_opt"]
            if active is None:
                return fn(*args, **kwargs)
            key = f"opt_{active}_{sub_name}"
            if device.type == "cuda":
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                out = fn(*args, **kwargs)
                e1.record()
                torch.cuda.synchronize(device)
                ms = float(e0.elapsed_time(e1))
            else:
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                ms = (time.perf_counter() - t0) * 1000.0
            state["cur"][key] = state["cur"].get(key, 0.0) + ms
            return out

        return _wrapped

    class _TimedCallableProxy:
        def __init__(self, target: Any, bucket_key: str) -> None:
            object.__setattr__(self, "_target", target)
            object.__setattr__(self, "_bucket_key", bucket_key)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            target = object.__getattribute__(self, "_target")
            bucket_key = object.__getattribute__(self, "_bucket_key")
            if device.type == "cuda":
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                out = target(*args, **kwargs)
                e1.record()
                torch.cuda.synchronize(device)
                ms = float(e0.elapsed_time(e1))
            else:
                t0 = time.perf_counter()
                out = target(*args, **kwargs)
                ms = (time.perf_counter() - t0) * 1000.0
            state["cur"][bucket_key] = state["cur"].get(bucket_key, 0.0) + ms
            return out

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_target"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(object.__getattribute__(self, "_target"), name, value)

    # Save originals
    orig_penalty_obj = trainer.penalty
    orig_opt_d_step = trainer.opt_d.step
    orig_opt_g_step = trainer.opt_g.step
    orig_after_d = trainer.policy.after_critic_step
    orig_g_loss = trainer.loss.g_loss
    orig_prior_reg_obj = trainer.prior_regularizer
    orig_q2d = v11_opt_mod.quintic_orthogonalize_2d
    orig_qb = v11_opt_mod.quintic_orthogonalize_batched
    orig_probe = v11_opt_mod._eval_probe_forward
    orig_opt_d_zg = trainer.opt_d.zero_grad
    orig_opt_g_zg = trainer.opt_g.zero_grad

    step_start_ev: List[Any] = [None]
    d_phase_end_ev: List[Any] = [None]
    d_bwd_start_ev: List[Any] = [None]
    g_bwd_start_ev: List[Any] = [None]

    def _record_now() -> Any:
        if device.type == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def _elapsed_ms(a: Any, b: Any) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            return float(a.elapsed_time(b))
        return (b - a) * 1000.0

    def _wrap_opt_d_zg(*args: Any, **kwargs: Any) -> Any:
        out = orig_opt_d_zg(*args, **kwargs)
        d_bwd_start_ev[0] = _record_now()
        return out

    def _wrap_opt_d_step(*args: Any, **kwargs: Any) -> Any:
        if d_bwd_start_ev[0] is not None:
            now = _record_now()
            state["cur"]["d_backward"] = state["cur"].get("d_backward", 0.0) + _elapsed_ms(d_bwd_start_ev[0], now)
            d_bwd_start_ev[0] = None
        state["active_opt"] = "d"
        t_start = _record_now()
        try:
            out = orig_opt_d_step(*args, **kwargs)
        finally:
            t_end = _record_now()
            state["active_opt"] = None
        state["cur"]["opt_d_step"] = state["cur"].get("opt_d_step", 0.0) + _elapsed_ms(t_start, t_end)
        return out

    def _wrap_after_d(*args: Any, **kwargs: Any) -> Any:
        out = orig_after_d(*args, **kwargs)
        d_phase_end_ev[0] = _record_now()
        return out

    def _wrap_opt_g_zg(*args: Any, **kwargs: Any) -> Any:
        out = orig_opt_g_zg(*args, **kwargs)
        g_bwd_start_ev[0] = _record_now()
        return out

    def _wrap_opt_g_step(*args: Any, **kwargs: Any) -> Any:
        if g_bwd_start_ev[0] is not None:
            now = _record_now()
            state["cur"]["g_backward"] = state["cur"].get("g_backward", 0.0) + _elapsed_ms(g_bwd_start_ev[0], now)
            g_bwd_start_ev[0] = None
        state["active_opt"] = "g"
        t_start = _record_now()
        try:
            out = orig_opt_g_step(*args, **kwargs)
        finally:
            t_end = _record_now()
            state["active_opt"] = None
        state["cur"]["opt_g_step"] = state["cur"].get("opt_g_step", 0.0) + _elapsed_ms(t_start, t_end)
        return out

    # Also support other optimizer families that might call quintic_orthogonalize from _unified_v4 or _r4_ablation
    extra_patches: List[Tuple[Any, str, Any]] = []
    for mod_name in (
        "particlegan._unified_v4.optimizer",
        "particlegan._law4_alt.optimizer",
        "particlegan._law4_energy.optimizer",
    ):
        try:
            m = __import__(mod_name, fromlist=["*"])
        except ImportError:
            continue
        for fn_name, sub in (
            ("quintic_orthogonalize_2d", "quintic_ns"),
            ("quintic_orthogonalize_batched", "quintic_ns"),
            ("_eval_probe_forward", "secant_probe"),
        ):
            if hasattr(m, fn_name):
                orig_fn = getattr(m, fn_name)
                extra_patches.append((m, fn_name, orig_fn))
                setattr(m, fn_name, _cuda_time_opt_sub(orig_fn, sub))

    trainer.penalty = _TimedCallableProxy(orig_penalty_obj, "d_penalty")
    trainer.opt_d.zero_grad = _wrap_opt_d_zg
    trainer.opt_d.step = _wrap_opt_d_step
    trainer.policy.after_critic_step = _wrap_after_d
    trainer.loss.g_loss = _cuda_time(orig_g_loss, "g_loss_prior")
    if orig_prior_reg_obj is not None:
        trainer.prior_regularizer = _TimedCallableProxy(orig_prior_reg_obj, "g_loss_prior")
    trainer.opt_g.zero_grad = _wrap_opt_g_zg
    trainer.opt_g.step = _wrap_opt_g_step
    v11_opt_mod.quintic_orthogonalize_2d = _cuda_time_opt_sub(orig_q2d, "quintic_ns")
    v11_opt_mod.quintic_orthogonalize_batched = _cuda_time_opt_sub(orig_qb, "quintic_ns")
    v11_opt_mod._eval_probe_forward = _cuda_time_opt_sub(orig_probe, "secant_probe")

    def _run_instrumented_step(real: torch.Tensor, real_g: torch.Tensor, record: bool) -> None:
        state["cur"] = {}
        d_phase_end_ev[0] = None
        d_bwd_start_ev[0] = None
        g_bwd_start_ev[0] = None
        step_start_ev[0] = _record_now()
        trainer.step(real, generator_real=real_g)
        step_end = _record_now()
        if not record:
            return
        total_ms = _elapsed_ms(step_start_ev[0], step_end)
        d_total_ms = _elapsed_ms(step_start_ev[0], d_phase_end_ev[0]) if d_phase_end_ev[0] is not None else 0.0
        g_total_ms = max(0.0, total_ms - d_total_ms)

        c = state["cur"]
        d_pen = c.get("d_penalty", 0.0)
        d_opt = c.get("opt_d_step", 0.0)
        d_opt_ns = c.get("opt_d_quintic_ns", 0.0)
        d_opt_sec = c.get("opt_d_secant_probe", 0.0)
        d_opt_ovh = max(0.0, d_opt - d_opt_ns - d_opt_sec)
        d_fwd_bwd = max(0.0, d_total_ms - d_pen - d_opt)
        d_bwd = c.get("d_backward", 0.0)
        d_fwd = max(0.0, d_fwd_bwd - d_bwd)

        g_lp = c.get("g_loss_prior", 0.0)
        g_opt = c.get("opt_g_step", 0.0)
        g_opt_ns = c.get("opt_g_quintic_ns", 0.0)
        g_opt_sec = c.get("opt_g_secant_probe", 0.0)
        g_opt_ovh = max(0.0, g_opt - g_opt_ns - g_opt_sec)
        g_fwd_bwd = max(0.0, g_total_ms - g_lp - g_opt)
        g_bwd = c.get("g_backward", 0.0)
        g_fwd = max(0.0, g_fwd_bwd - g_bwd)

        row = {
            "step_total_ms": total_ms,
            "d_phase_total_ms": d_total_ms,
            "d_forward_backward_ms": d_fwd_bwd,
            "d_forward_ms": d_fwd,
            "d_backward_ms": d_bwd,
            "d_penalty_ms": d_pen,
            "opt_d_step_ms": d_opt,
            "opt_d_quintic_ns_ms": d_opt_ns,
            "opt_d_secant_probe_ms": d_opt_sec,
            "opt_d_overhead_ms": d_opt_ovh,
            "g_phase_total_ms": g_total_ms,
            "g_forward_backward_ms": g_fwd_bwd,
            "g_forward_ms": g_fwd,
            "g_backward_ms": g_bwd,
            "g_loss_prior_ms": g_lp,
            "opt_g_step_ms": g_opt,
            "opt_g_quintic_ns_ms": g_opt_ns,
            "opt_g_secant_probe_ms": g_opt_sec,
            "opt_g_overhead_ms": g_opt_ovh,
        }
        for k, v in row.items():
            phase_records.setdefault(k, []).append(v)

    try:
        yield _run_instrumented_step
    finally:
        trainer.penalty = orig_penalty_obj
        trainer.opt_d.zero_grad = orig_opt_d_zg
        trainer.opt_d.step = orig_opt_d_step
        trainer.policy.after_critic_step = orig_after_d
        trainer.loss.g_loss = orig_g_loss
        trainer.prior_regularizer = orig_prior_reg_obj
        trainer.opt_g.zero_grad = orig_opt_g_zg
        trainer.opt_g.step = orig_opt_g_step
        v11_opt_mod.quintic_orthogonalize_2d = orig_q2d
        v11_opt_mod.quintic_orthogonalize_batched = orig_qb
        v11_opt_mod._eval_probe_forward = orig_probe
        for m, fn_name, orig_fn in extra_patches:
            setattr(m, fn_name, orig_fn)


def profile_dcgan_task(
    idea_id: str,
    task_id: str,
    device: torch.device,
    *,
    warmup_steps: int = 5,
    energy_steps: int = 50,
    breakdown_steps: int = 20,
    optimizer_family_override: Optional[str] = None,
    recipe_extra: Optional[Dict[str, Any]] = None,
    run_breakdown: bool = True,
) -> Dict[str, Any]:
    from benchmarks.toy_audit.reproducibility import reproducible_execution

    idea = load_idea(idea_id)
    task = load_task(task_id)
    dev_idx = device.index if (device.type == "cuda" and device.index is not None) else 0
    nvml = NVMLMonitor(dev_idx)

    @reproducible_execution
    def _exec(*, seed: int = 0) -> Dict[str, Any]:
        del seed
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            time.sleep(0.8)
        idle_w = nvml.measure_idle_power(duration_s=0.4)
        util_before = nvml.gpu_utilization()

        # --- Pass A: Uninstrumented sustained energy & latency loop ---
        total_a = warmup_steps + energy_steps
        context, trainer, g, d, spec = _build_dcgan_trainer(
            idea,
            task,
            device,
            total_a,
            optimizer_family_override=optimizer_family_override,
            recipe_extra=recipe_extra,
        )
        shape = (spec["batch_size"], 3, spec["image_size"], spec["image_size"])
        c_stream = context.streams.generator("data", component="critic", purpose="training")
        g_stream = context.streams.generator("data", component="generator", purpose="training")

        # Keep only ONE active pair on GPU at a time (matching Forge scaling_benchmark's
        # resident-batch Peak VRAM contract ~1372.6 MiB on r256) by regenerating in-place!
        real_buf = torch.empty(shape, device=device, dtype=torch.float32)
        real_g_buf = torch.empty(shape, device=device, dtype=torch.float32)

        for _ in range(warmup_steps):
            real_buf.uniform_(-1.0, 1.0, generator=c_stream)
            real_g_buf.uniform_(-1.0, 1.0, generator=g_stream)
            trainer.step(real_buf, generator_real=real_g_buf)
        _sync_cuda(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        step_ms_list: List[float] = []
        # Reuse one fixed resident batch pair during the sustained energy loop so GPU power
        # reflects pure trainer.step() without RNG-fill jitter or extra VRAM footprint.
        real_buf.uniform_(-1.0, 1.0, generator=c_stream)
        real_g_buf.uniform_(-1.0, 1.0, generator=g_stream)
        _sync_cuda(device)
        e_start_j = nvml.total_energy_joules()
        t0 = time.perf_counter()
        with PowerSampler(nvml, interval_s=0.005) as ps:
            for _ in range(warmup_steps, total_a):
                if device.type == "cuda":
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    trainer.step(real_buf, generator_real=real_g_buf)
                    ev1.record()
                    torch.cuda.synchronize(device)
                    step_ms_list.append(float(ev0.elapsed_time(ev1)))
                else:
                    s0 = time.perf_counter()
                    trainer.step(real_buf, generator_real=real_g_buf)
                    step_ms_list.append((time.perf_counter() - s0) * 1000.0)
        _sync_cuda(device)
        wall_s = time.perf_counter() - t0
        e_end_j = nvml.total_energy_joules()

        peak_alloc_mib = float(torch.cuda.max_memory_allocated(device)) / float(2**20) if device.type == "cuda" else 0.0
        peak_res_mib = float(torch.cuda.max_memory_reserved(device)) / float(2**20) if device.type == "cuda" else 0.0
        state_info = optimizer_state_summary(trainer)
        g_params = sum(int(p.numel()) for p in g.parameters())
        d_params = sum(int(p.numel()) for p in d.parameters())
        p_params = sum(int(p.numel()) for p in trainer.prior.parameters())

        step_stats = _stats(step_ms_list)
        mean_ms = step_stats["mean"]
        img_per_sec = 1000.0 * float(spec["batch_size"]) / mean_ms if mean_ms > 0 else 0.0

        energy_total_j = (e_end_j - e_start_j) if (e_start_j is not None and e_end_j is not None) else None
        energy_j_per_step = (energy_total_j / float(energy_steps)) if energy_total_j is not None else None
        avg_watts_from_energy = (energy_total_j / wall_s) if (energy_total_j is not None and wall_s > 0) else None
        power_poll_stats = _stats(ps.samples) if ps.samples else None
        dyn_j_per_step = (
            max(0.0, (avg_watts_from_energy - idle_w) * (wall_s / float(energy_steps)))
            if (avg_watts_from_energy is not None and idle_w is not None)
            else None
        )

        # Free Pass A before Pass B
        del real_buf, real_g_buf, trainer, g, d, context
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- Pass B: CUDA-event instrumented phase breakdown ---
        breakdown_summary: Optional[Dict[str, Any]] = None
        if run_breakdown:
            total_b = 3 + breakdown_steps
            context_b, trainer_b, g_b, d_b, _ = _build_dcgan_trainer(
                idea,
                task,
                device,
                total_b,
                optimizer_family_override=optimizer_family_override,
                recipe_extra=recipe_extra,
            )
            c_stream_b = context_b.streams.generator("data", component="critic", purpose="training")
            g_stream_b = context_b.streams.generator("data", component="generator", purpose="training")
            real_b = torch.rand(shape, generator=c_stream_b, device=device).mul_(2).sub_(1)
            real_g_b = torch.rand(shape, generator=g_stream_b, device=device).mul_(2).sub_(1)
            phase_records: Dict[str, List[float]] = {}
            with _instrument_step_phases(trainer_b, device, phase_records) as run_step:
                for i in range(total_b):
                    run_step(real_b, real_g_b, record=(i >= 3))

            med_total = statistics.median(phase_records["step_total_ms"])
            breakdown_summary = {}
            for k, vals in phase_records.items():
                st = _stats(vals)
                st["pct_of_step"] = (100.0 * st["median"] / med_total) if med_total > 0 else 0.0
                if energy_j_per_step is not None and med_total > 0:
                    st["est_joules_per_step"] = energy_j_per_step * (st["median"] / med_total)
                breakdown_summary[k] = st

            del real_b, real_g_b, trainer_b, g_b, d_b, context_b
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "idea_id": idea_id,
            "optimizer_family": optimizer_family_override or idea["recipe_overrides"]["optimizer_family"],
            "task_id": task_id,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "gpu_util_before": util_before,
            "spec": {
                "image_size": spec["image_size"],
                "batch_size": spec["batch_size"],
                "base_channels": spec["base_channels"],
                "max_channels": spec["max_channels"],
                "latent_dim": spec["latent_dim"],
                "n_particles": spec["n_particles"],
            },
            "param_counts": {"generator": g_params, "discriminator": d_params, "prior": p_params, "total": g_params + d_params + p_params},
            "sustained_uninstrumented": {
                "warmup_steps": warmup_steps,
                "measured_steps": energy_steps,
                "wall_seconds": wall_s,
                "step_ms": step_stats,
                "images_per_second": img_per_sec,
                "idle_power_watts": idle_w,
                "nvml_energy_total_joules": energy_total_j,
                "joules_per_step": energy_j_per_step,
                "dynamic_joules_per_step": dyn_j_per_step,
                "avg_power_watts_from_energy": avg_watts_from_energy,
                "power_poll_watts": power_poll_stats,
                "peak_allocated_mib": peak_alloc_mib,
                "peak_reserved_mib": peak_res_mib,
            },
            "optimizer_state": state_info,
            "phase_breakdown_ms": breakdown_summary,
        }

    try:
        return _exec(seed=0)
    finally:
        nvml.shutdown()


# ---------------------------------------------------------------------------
# Phase 1B: Mechanistic Gradient, Condition Number & Impedance Diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def _grad_spectrum_metrics(module: torch.nn.Module) -> Dict[str, Any]:
    """Compute per-layer gradient norms, max/min norm ratio, and 2D matrix singular-value spectrum."""
    layer_norms: List[Tuple[str, float]] = []
    cond_numbers: List[float] = []
    eff_ranks: List[float] = []
    top1_energy: List[float] = []
    mat_details: List[Dict[str, Any]] = []

    for name, p in module.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            continue
        nrm = float(g.norm().item())
        if p.ndim >= 2:
            layer_norms.append((name, nrm))
            mat = g.reshape(p.shape[0], -1)
            r, c = int(mat.shape[0]), int(mat.shape[1])
            if min(r, c) >= 2 and nrm > 1e-30:
                sv = torch.linalg.svdvals(mat)
                s1 = float(sv[0].item())
                sr = float(sv[-1].item())
                nz = sv[sv > 1e-12 * s1]
                s_min_nz = float(nz[-1].item()) if nz.numel() > 0 else sr
                cond_full = (s1 / sr) if sr > 1e-30 else float("inf")
                cond_nz = (s1 / s_min_nz) if s_min_nz > 1e-30 else float("inf")
                frob_sq = float(sv.square().sum().item())
                erank = frob_sq / max(s1 * s1, 1e-38)
                t1 = (s1 * s1) / max(frob_sq, 1e-38)
                if math.isfinite(cond_full):
                    cond_numbers.append(cond_full)
                eff_ranks.append(erank)
                top1_energy.append(t1)
                mat_details.append(
                    {
                        "name": name,
                        "shape": [r, c],
                        "grad_norm": nrm,
                        "sigma_1": s1,
                        "sigma_r": sr,
                        "cond_full": cond_full if math.isfinite(cond_full) else None,
                        "cond_nz": cond_nz if math.isfinite(cond_nz) else None,
                        "eff_rank": erank,
                        "top1_energy_frac": t1,
                    }
                )

    pos_norms = [n for _, n in layer_norms if n > 0.0]
    ratio = (max(pos_norms) / min(pos_norms)) if len(pos_norms) >= 2 else 1.0
    total_norm = math.sqrt(sum(n * n for _, n in layer_norms)) if layer_norms else 0.0
    return {
        "num_weight_layers": len(layer_norms),
        "total_weight_grad_norm": total_norm,
        "min_layer_grad_norm": min(pos_norms) if pos_norms else 0.0,
        "max_layer_grad_norm": max(pos_norms) if pos_norms else 0.0,
        "first_layer_grad_norm": layer_norms[0][1] if layer_norms else 0.0,
        "last_layer_grad_norm": layer_norms[-1][1] if layer_norms else 0.0,
        "max_over_min_layer_grad_ratio": ratio,
        "median_sigma1_over_sigmar": float(statistics.median(cond_numbers)) if cond_numbers else None,
        "max_sigma1_over_sigmar": float(max(cond_numbers)) if cond_numbers else None,
        "median_eff_rank": float(statistics.median(eff_ranks)) if eff_ranks else None,
        "median_top1_energy_frac": float(statistics.median(top1_energy)) if top1_energy else None,
        "layers": mat_details,
    }


@torch.no_grad()
def _probe_raw_sgd_displacement(
    module: torch.nn.Module,
    probe_input: torch.Tensor,
    lr: float,
    *,
    is_critic: bool,
) -> float:
    """Bit-exact non-destructive probe of function-space RMS displacement under raw SGD (-lr * grad)."""
    was_train = module.training
    module.eval()
    saved: List[Tuple[torch.Tensor, torch.Tensor]] = []
    try:
        y_before = module(probe_input).detach().float()
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                saved.append((p, p.data.detach().clone()))
                p.data.add_(p.grad.detach(), alpha=-float(lr))
        y_after = module(probe_input).detach().float()
        diff = y_after - y_before
        rms = float(diff.square().mean().sqrt().item())
        return rms
    finally:
        for p, orig in saved:
            p.data.copy_(orig)
        module.train(was_train)


def _build_regime_runner(idea: dict, task_id: str, device: torch.device):
    """Instantiate exact official Forge task models, trainer, and data sampler."""
    from experiments.forge.api import task_formulation_context

    task = load_task(task_id)
    adapter = task["adapter"]

    if adapter == "transfer_vector":
        from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
        from experiments.forge.tanh_point_mass import accepts_rail_escape, sample_rail_escape_target
        from benchmarks.transfer_suite.vector_tasks import sample_target

        spec = resolve_vector_spec(task)
        context = task_formulation_context(idea, task, {"seed": 0}, device=str(device))
        g, d = build_vector_models(context, spec)
        trainer = context.build_trainer(g, d, max_steps=task["execution"]["steps"])
        data_rng = context.streams.generator("data", component="target", purpose="training", device="cpu")
        is_rail = accepts_rail_escape(task, spec)

        def next_batch(step: int) -> torch.Tensor:
            if is_rail:
                return sample_rail_escape_target(task, spec, context.recipe.batch_size, data_rng, step - 1).to(device)
            return sample_target(spec, context.recipe.batch_size, data_rng, step - 1).to(device)

        return context, trainer, g, d, task, next_batch

    if adapter == "native100":
        from experiments.forge.nativeprofiles import build_declared_native_models
        from benchmarks.toy100.problems import sample_real

        context = task_formulation_context(idea, task, {"seed": 0}, device=str(device))
        g, d = build_declared_native_models(context, task["execution"]["model"])
        trainer = context.build_trainer(g, d, max_steps=task["execution"]["steps"])
        data_rng = context.streams.generator("data", component="target", purpose="training")
        problem = task["execution"]["problem"]

        def next_batch(step: int) -> torch.Tensor:
            del step
            return sample_real(problem, context.recipe.batch_size, device=device, generator=data_rng)

        return context, trainer, g, d, task, next_batch

    if adapter == "imbalance_nash":
        from benchmarks.toy_audit.api_imbalance_nash import build_models_for_spec, sample_target_for_kind

        kind = task["evaluation"]["kind"]
        spec = task["execution"]["host_definition"]
        context = task_formulation_context(idea, task, {"seed": 0}, device=str(device))
        g = context.construct(lambda: build_models_for_spec(spec, z_dim=context.recipe.z_dim)[0], component="generator").to(device)
        d = context.construct(lambda: build_models_for_spec(spec, z_dim=context.recipe.z_dim)[1], component="discriminator").to(device)
        trainer = context.build_trainer(g, d, max_steps=task["execution"]["steps"])
        data_rng = context.streams.generator("data", component="target", purpose="training", device="cpu")

        def next_batch(step: int) -> torch.Tensor:
            return sample_target_for_kind(kind, context.recipe.batch_size, data_rng, completed_steps=step - 1, device="cpu").to(device)

        return context, trainer, g, d, task, next_batch

    raise ValueError(f"Unsupported task adapter: {adapter}")


def run_mechanistic_regime(
    idea_id: str,
    task_id: str,
    device: torch.device,
    *,
    num_steps: int,
    sample_steps: List[int],
) -> Dict[str, Any]:
    from benchmarks.toy_audit.reproducibility import reproducible_execution

    idea = load_idea(idea_id)
    sample_set = set(sample_steps)

    @reproducible_execution
    def _exec(*, seed: int = 0) -> Dict[str, Any]:
        del seed
        context, trainer, g, d, task, next_batch = _build_regime_runner(idea, task_id, device)
        g_params = sum(int(p.numel()) for p in g.parameters() if p.requires_grad)
        d_params = sum(int(p.numel()) for p in d.parameters() if p.requires_grad)
        lr_g = float(context.recipe.lr)
        lr_d = float(context.recipe.lr * context.recipe.d_lr_mult)

        # Fixed evaluation probe tensors (separate RNG, never touches training streams)
        probe_rng = torch.Generator(device=device).manual_seed(4242)
        with torch.no_grad():
            z_probe = trainer.prior.sample(min(256, len(trainer.prior.z)), generator=probe_rng)[0].detach().clone()
        x_probe_holder: List[Optional[torch.Tensor]] = [None]

        # Wrap opt_d.step and opt_g.step to capture pre-step grad spectrum, raw SGD displacement,
        # and post-step equilibrated displacement at sample_steps (bit-exact restoration!).
        orig_opt_d_step = trainer.opt_d.step
        orig_opt_g_step = trainer.opt_g.step
        cur_step_box = [0]
        cur_row: Dict[str, Any] = {}

        def _hooked_d_step(*args: Any, **kwargs: Any) -> Any:
            s = cur_step_box[0]
            if s in sample_set and x_probe_holder[0] is not None:
                xp = x_probe_holder[0]
                cur_row["d_grad"] = _grad_spectrum_metrics(d)
                cur_row["delta_d_sgd"] = _probe_raw_sgd_displacement(d, xp, lr_d, is_critic=True)
                with torch.no_grad():
                    was_t = d.training
                    d.eval()
                    d_pre = d(xp).detach().float()
                    d.train(was_t)
                out = orig_opt_d_step(*args, **kwargs)
                with torch.no_grad():
                    was_t = d.training
                    d.eval()
                    d_post = d(xp).detach().float()
                    d.train(was_t)
                    cur_row["delta_d_equil"] = float((d_post - d_pre).square().mean().sqrt().item())
                cur_row["opt_d_last_step_gain"] = float(getattr(trainer.opt_d, "last_step_gain", 1.0))
                cur_row["opt_d_ema_func_gain"] = float(getattr(trainer.opt_d, "_ema_func_gain", 1.0))
                return out
            return orig_opt_d_step(*args, **kwargs)

        def _hooked_g_step(*args: Any, **kwargs: Any) -> Any:
            s = cur_step_box[0]
            if s in sample_set:
                cur_row["g_grad"] = _grad_spectrum_metrics(g)
                cur_row["delta_g_sgd"] = _probe_raw_sgd_displacement(g, z_probe, lr_g, is_critic=False)
                with torch.no_grad():
                    was_t = g.training
                    g.eval()
                    g_pre = g(z_probe).detach().float()
                    g.train(was_t)
                out = orig_opt_g_step(*args, **kwargs)
                with torch.no_grad():
                    was_t = g.training
                    g.eval()
                    g_post = g(z_probe).detach().float()
                    g.train(was_t)
                    cur_row["delta_g_equil"] = float((g_post - g_pre).square().mean().sqrt().item())
                    if task_id == "zero_tanh_rail_escape":
                        sech2 = (1.0 - g_post.square()).clamp_min(0.0)
                        tgt_pair = (
                            task["execution"]["shifted_target"]
                            if s >= int(task["execution"]["shift_step"])
                            else task["execution"]["initial_target"]
                        )
                        tgt = torch.tensor(tgt_pair, dtype=torch.float32, device=g_post.device).unsqueeze(0)
                        cur_row["tanh_rail"] = {
                            "phase": "post_shift" if s >= int(task["execution"]["shift_step"]) else "pre_shift",
                            "target": list(tgt_pair),
                            "mean_abs_y": float(g_post.abs().mean().item()),
                            "max_abs_y": float(g_post.abs().amax().item()),
                            "sech2_min": float(sech2.amin().item()),
                            "sech2_p10": float(torch.quantile(sech2.reshape(-1), 0.10).item()),
                            "sech2_median": float(sech2.median().item()),
                            "sech2_mean": float(sech2.mean().item()),
                            "mae_to_target": float((g_post - tgt).abs().mean().item()),
                        }
                cur_row["opt_g_last_step_gain"] = float(getattr(trainer.opt_g, "last_step_gain", 1.0))
                cur_row["opt_g_ema_func_gain"] = float(getattr(trainer.opt_g, "_ema_func_gain", 1.0))
                return out
            return orig_opt_g_step(*args, **kwargs)

        trainer.opt_d.step = _hooked_d_step
        trainer.opt_g.step = _hooked_g_step

        timeline: List[Dict[str, Any]] = []
        g_clamps = 0
        d_clamps = 0
        try:
            for s in range(1, num_steps + 1):
                cur_step_box[0] = s
                cur_row = {"step": s}
                batch = next_batch(s)
                if x_probe_holder[0] is None:
                    # Mix real batch + initial fake batch for critic probe
                    with torch.no_grad():
                        g.eval()
                        f0 = g(z_probe[: min(len(z_probe), len(batch))]).detach()
                        g.train()
                        x_probe_holder[0] = torch.cat([batch[: min(128, len(batch))], f0[: min(128, len(f0))]], dim=0).detach().clone()
                trainer.step(batch)
                if float(getattr(trainer.opt_g, "last_step_gain", 1.0)) < 0.999:
                    g_clamps += 1
                if float(getattr(trainer.opt_d, "last_step_gain", 1.0)) < 0.999:
                    d_clamps += 1
                if s in sample_set:
                    dg_s = cur_row.get("delta_g_sgd", 0.0)
                    dd_s = cur_row.get("delta_d_sgd", 0.0)
                    dg_e = cur_row.get("delta_g_equil", 0.0)
                    dd_e = cur_row.get("delta_d_equil", 0.0)
                    cur_row["ratio_d_over_g_sgd"] = (dd_s / dg_s) if dg_s > 1e-30 else None
                    cur_row["ratio_d_over_g_equil"] = (dd_e / dg_e) if dg_e > 1e-30 else None
                    # Compactify per-layer list for steps other than first/mid/last
                    if s not in (sample_steps[0], sample_steps[len(sample_steps) // 2], sample_steps[-1]):
                        if "g_grad" in cur_row:
                            cur_row["g_grad"].pop("layers", None)
                        if "d_grad" in cur_row:
                            cur_row["d_grad"].pop("layers", None)
                    timeline.append(cur_row)
        finally:
            trainer.opt_d.step = orig_opt_d_step
            trainer.opt_g.step = orig_opt_g_step

        # Aggregate summary statistics across sampled steps
        def _med(key_fn: Callable[[dict], Optional[float]]) -> Optional[float]:
            vals = [v for row in timeline for v in (key_fn(row),) if v is not None and math.isfinite(v)]
            return float(statistics.median(vals)) if vals else None

        def _max(key_fn: Callable[[dict], Optional[float]]) -> Optional[float]:
            vals = [v for row in timeline for v in (key_fn(row),) if v is not None and math.isfinite(v)]
            return float(max(vals)) if vals else None

        summary = {
            "g_params": g_params,
            "d_params": d_params,
            "d_over_g_param_ratio": float(d_params) / float(max(1, g_params)),
            "steps_run": num_steps,
            "opt_g_secant_clamp_frac": float(g_clamps) / float(num_steps),
            "opt_d_secant_clamp_frac": float(d_clamps) / float(num_steps),
            "g_layer_grad_ratio_median": _med(lambda r: r.get("g_grad", {}).get("max_over_min_layer_grad_ratio")),
            "g_layer_grad_ratio_max": _max(lambda r: r.get("g_grad", {}).get("max_over_min_layer_grad_ratio")),
            "d_layer_grad_ratio_median": _med(lambda r: r.get("d_grad", {}).get("max_over_min_layer_grad_ratio")),
            "d_layer_grad_ratio_max": _max(lambda r: r.get("d_grad", {}).get("max_over_min_layer_grad_ratio")),
            "g_grad_cond_median": _med(lambda r: r.get("g_grad", {}).get("median_sigma1_over_sigmar")),
            "g_grad_cond_max": _max(lambda r: r.get("g_grad", {}).get("max_sigma1_over_sigmar")),
            "g_grad_eff_rank_median": _med(lambda r: r.get("g_grad", {}).get("median_eff_rank")),
            "g_grad_top1_energy_median": _med(lambda r: r.get("g_grad", {}).get("median_top1_energy_frac")),
            "delta_d_sgd_median": _med(lambda r: r.get("delta_d_sgd")),
            "delta_g_sgd_median": _med(lambda r: r.get("delta_g_sgd")),
            "ratio_d_over_g_sgd_median": _med(lambda r: r.get("ratio_d_over_g_sgd")),
            "delta_d_equil_median": _med(lambda r: r.get("delta_d_equil")),
            "delta_g_equil_median": _med(lambda r: r.get("delta_g_equil")),
            "ratio_d_over_g_equil_median": _med(lambda r: r.get("ratio_d_over_g_equil")),
            "final_opt_g_ema_func_gain": timeline[-1].get("opt_g_ema_func_gain") if timeline else None,
            "final_opt_d_ema_func_gain": timeline[-1].get("opt_d_ema_func_gain") if timeline else None,
        }

        return {
            "idea_id": idea_id,
            "task_id": task_id,
            "summary": summary,
            "timeline": timeline,
        }

    return _exec(seed=0)


# ---------------------------------------------------------------------------
# Phase 2: Matched Uncontended Energy & Joules-to-PASS Benchmark
# ---------------------------------------------------------------------------


def benchmark_joules_to_pass_ring16(
    idea_id: str,
    device: torch.device,
    *,
    optimizer_family_override: Optional[str] = None,
    recipe_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full ring16_acquisition (1600 steps) measuring hardware Joules/step & Joules-to-first-PASS."""
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    from benchmarks.toy_audit.ring16_quality import score_samples
    from experiments.forge.vectorprofiles import resolve_vector_spec

    idea = deepcopy(load_idea(idea_id))
    if optimizer_family_override is not None:
        idea.setdefault("recipe_overrides", {})["optimizer_family"] = optimizer_family_override
    if recipe_extra:
        idea.setdefault("recipe_overrides", {}).update(recipe_extra)

    task = load_task("ring16_acquisition")
    spec = resolve_vector_spec(task)
    thresholds = task["evaluation"]["thresholds"]
    total_steps = int(task["execution"]["steps"])
    count = int(task["evaluation"].get("observations", 96))
    checkpoints = set(sorted({math.ceil(i * total_steps / count) for i in range(1, count + 1)}))
    min_stable = int(task["evaluation"].get("minimum_stable_checks", 5))

    dev_idx = device.index if (device.type == "cuda" and device.index is not None) else 0
    nvml = NVMLMonitor(dev_idx)

    def _passes(metrics: dict) -> bool:
        for field, op, bound in thresholds:
            v = metrics.get(field)
            if v is None or not math.isfinite(float(v)):
                return False
            if op == ">=" and not (float(v) >= float(bound)):
                return False
            if op == "<=" and not (float(v) <= float(bound)):
                return False
            if op == "==" and not (float(v) == float(bound)):
                return False
        return True

    @reproducible_execution
    def _exec(*, seed: int = 0) -> Dict[str, Any]:
        del seed
        context, trainer, g, d, _, next_batch = _build_regime_runner(idea, "ring16_acquisition", device)
        eval_stream = context.streams.generator("eval", component="live", purpose="ring16_eval")

        # Measure pure training step energy over steps 1..total_steps (excluding eval calls!)
        train_j_accum = 0.0
        train_s_accum = 0.0
        pass_flags: List[Tuple[int, bool]] = []
        first_pass_step: Optional[int] = None
        first_sustained_step: Optional[int] = None
        consec = 0
        nan_step: Optional[int] = None
        power_samples: List[float] = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # Measure energy in chunks between checkpoints for clean hardware NVML counter integration
        chunk_e0 = nvml.total_energy_joules()
        chunk_t0 = time.perf_counter()
        for step in range(1, total_steps + 1):
            batch = next_batch(step)
            try:
                res = trainer.step(batch)
            except FloatingPointError:
                _sync_cuda(device)
                chunk_t1 = time.perf_counter()
                chunk_e1 = nvml.total_energy_joules()
                train_s_accum += (chunk_t1 - chunk_t0)
                if chunk_e0 is not None and chunk_e1 is not None and chunk_e1 >= chunk_e0:
                    train_j_accum += (chunk_e1 - chunk_e0)
                nan_step = step
                break
            if step % 20 == 0:
                w = nvml.power_watts()
                if w is not None:
                    power_samples.append(w)

            # Check finiteness
            if any(isinstance(v, float) and not math.isfinite(v) for v in res.values()):
                _sync_cuda(device)
                chunk_t1 = time.perf_counter()
                chunk_e1 = nvml.total_energy_joules()
                train_s_accum += (chunk_t1 - chunk_t0)
                if chunk_e0 is not None and chunk_e1 is not None and chunk_e1 >= chunk_e0:
                    train_j_accum += (chunk_e1 - chunk_e0)
                nan_step = step
                break

            if step in checkpoints:
                _sync_cuda(device)
                chunk_t1 = time.perf_counter()
                chunk_e1 = nvml.total_energy_joules()
                train_s_accum += (chunk_t1 - chunk_t0)
                if chunk_e0 is not None and chunk_e1 is not None and chunk_e1 >= chunk_e0:
                    train_j_accum += (chunk_e1 - chunk_e0)

                with torch.no_grad():
                    samples = trainer.sample(4096, generator=eval_stream, ema=False, output_noise=False).cpu()
                metrics = score_samples(samples, spec, step)
                ok = _passes(metrics)
                pass_flags.append((step, ok))
                if ok:
                    if first_pass_step is None:
                        first_pass_step = step
                    consec += 1
                    if consec >= min_stable and first_sustained_step is None:
                        first_sustained_step = step
                else:
                    consec = 0

                _sync_cuda(device)
                chunk_e0 = nvml.total_energy_joules()
                chunk_t0 = time.perf_counter()

        steps_done = nan_step if nan_step is not None else total_steps
        ms_per_step = 1000.0 * train_s_accum / float(max(1, steps_done))
        j_per_step = train_j_accum / float(max(1, steps_done)) if nvml.available else None
        avg_w = (train_j_accum / train_s_accum) if (nvml.available and train_s_accum > 0) else None
        terminal_pass = len(pass_flags) >= min_stable and all(ok for _, ok in pass_flags[-min_stable:])
        state_info = optimizer_state_summary(trainer)
        peak_mib = float(torch.cuda.max_memory_allocated(device)) / float(2**20) if device.type == "cuda" else 0.0

        return {
            "idea_id": idea_id,
            "optimizer_family": idea["recipe_overrides"]["optimizer_family"],
            "task_id": "ring16_acquisition",
            "steps_completed": steps_done,
            "nan_step": nan_step,
            "terminal_sustained_pass": terminal_pass,
            "first_pass_step": first_pass_step,
            "first_sustained_5check_step": first_sustained_step,
            "ms_per_step": ms_per_step,
            "avg_power_watts": avg_w,
            "power_poll_median_watts": float(statistics.median(power_samples)) if power_samples else None,
            "joules_per_step": j_per_step,
            "joules_to_first_pass": (j_per_step * first_pass_step) if (j_per_step is not None and first_pass_step is not None) else None,
            "joules_to_sustained_pass": (j_per_step * first_sustained_step) if (j_per_step is not None and first_sustained_step is not None) else None,
            "total_train_joules": train_j_accum if nvml.available else None,
            "peak_allocated_mib": peak_mib,
            "state_bytes_per_param": state_info["state_bytes_per_param"],
        }

    try:
        return _exec(seed=0)
    finally:
        nvml.shutdown()


def benchmark_joules_to_pass_toy100(
    idea_id: str,
    task_id: str,
    device: torch.device,
    *,
    optimizer_family_override: Optional[str] = None,
    recipe_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a native100 task (grid100 / staggered100, 7000 steps) measuring hardware Joules/step & Joules-to-PASS."""
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    from benchmarks.toy100.accuracy_gate import evaluate_accuracy
    from benchmarks.toy100.metrics import evaluate_samples

    idea = deepcopy(load_idea(idea_id))
    if optimizer_family_override is not None:
        idea.setdefault("recipe_overrides", {})["optimizer_family"] = optimizer_family_override
    if recipe_extra:
        idea.setdefault("recipe_overrides", {}).update(recipe_extra)

    task = load_task(task_id)
    problem = str(task["execution"]["problem"])
    total_steps = int(task["execution"]["steps"])
    eval_interval = int(task["evaluation"]["eval_interval"])
    eval_samples = int(task["evaluation"]["eval_samples"])
    min_stable = int(task["evaluation"]["minimum_stable_checks"])
    checkpoints = set(range(eval_interval, total_steps + 1, eval_interval))

    dev_idx = device.index if (device.type == "cuda" and device.index is not None) else 0
    nvml = NVMLMonitor(dev_idx)

    def _eval_toy100(draw: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        gate_m = evaluate_samples(draw, problem)
        acc = evaluate_accuracy(draw, problem, gate_metrics=gate_m)
        return bool(acc.get("passed", False)), acc

    @reproducible_execution
    def _exec(*, seed: int = 0) -> Dict[str, Any]:
        del seed
        context, trainer, g, d, _, next_batch = _build_regime_runner(idea, task_id, device)
        eval_stream = context.streams.generator("eval", component="live", purpose=f"{task_id}_eval")

        train_j_accum = 0.0
        train_s_accum = 0.0
        pass_flags: List[Tuple[int, bool]] = []
        first_pass_step: Optional[int] = None
        first_sustained_step: Optional[int] = None
        last_acc: Dict[str, Any] = {}
        consec = 0
        nan_step: Optional[int] = None
        power_samples: List[float] = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Measure energy in chunks between checkpoints for clean NVML counter integration
        chunk_e0 = nvml.total_energy_joules()
        chunk_t0 = time.perf_counter()
        for step in range(1, total_steps + 1):
            batch = next_batch(step)
            try:
                res = trainer.step(batch)
            except FloatingPointError:
                _sync_cuda(device)
                chunk_t1 = time.perf_counter()
                chunk_e1 = nvml.total_energy_joules()
                train_s_accum += (chunk_t1 - chunk_t0)
                if chunk_e0 is not None and chunk_e1 is not None and chunk_e1 >= chunk_e0:
                    train_j_accum += (chunk_e1 - chunk_e0)
                nan_step = step
                break
            if step % 50 == 0:
                w = nvml.power_watts()
                if w is not None:
                    power_samples.append(w)
            if any(isinstance(v, float) and not math.isfinite(v) for v in res.values()):
                _sync_cuda(device)
                chunk_t1 = time.perf_counter()
                chunk_e1 = nvml.total_energy_joules()
                train_s_accum += (chunk_t1 - chunk_t0)
                if chunk_e0 is not None and chunk_e1 is not None and chunk_e1 >= chunk_e0:
                    train_j_accum += (chunk_e1 - chunk_e0)
                nan_step = step
                break

            if step in checkpoints:
                _sync_cuda(device)
                chunk_t1 = time.perf_counter()
                chunk_e1 = nvml.total_energy_joules()
                train_s_accum += (chunk_t1 - chunk_t0)
                if chunk_e0 is not None and chunk_e1 is not None and chunk_e1 >= chunk_e0:
                    train_j_accum += (chunk_e1 - chunk_e0)

                with torch.no_grad():
                    draw = trainer.sample(eval_samples, generator=eval_stream, ema=False, output_noise=False)
                    ok, acc = _eval_toy100(draw)
                last_acc = acc
                pass_flags.append((step, ok))
                if ok:
                    if first_pass_step is None:
                        first_pass_step = step
                    consec += 1
                    if consec >= min_stable and first_sustained_step is None:
                        first_sustained_step = step
                else:
                    consec = 0

                _sync_cuda(device)
                chunk_e0 = nvml.total_energy_joules()
                chunk_t0 = time.perf_counter()

        steps_done = nan_step if nan_step is not None else total_steps
        ms_per_step = 1000.0 * train_s_accum / float(max(1, steps_done))
        j_per_step = train_j_accum / float(max(1, steps_done)) if nvml.available else None
        avg_w = (train_j_accum / train_s_accum) if (nvml.available and train_s_accum > 0) else None
        terminal_pass = len(pass_flags) >= min_stable and all(ok for _, ok in pass_flags[-min_stable:])
        state_info = optimizer_state_summary(trainer)
        peak_mib = float(torch.cuda.max_memory_allocated(device)) / float(2**20) if device.type == "cuda" else 0.0

        return {
            "idea_id": idea_id,
            "optimizer_family": idea["recipe_overrides"]["optimizer_family"],
            "task_id": task_id,
            "steps_completed": steps_done,
            "nan_step": nan_step,
            "terminal_sustained_pass": terminal_pass,
            "first_pass_step": first_pass_step,
            "first_sustained_5check_step": first_sustained_step,
            "final_precision": last_acc.get("precision"),
            "final_center_rms_sigma": last_acc.get("center_rms_sigma"),
            "final_cov_frob_rms": last_acc.get("cov_frob_rms"),
            "ms_per_step": ms_per_step,
            "avg_power_watts": avg_w,
            "power_poll_median_watts": float(statistics.median(power_samples)) if power_samples else None,
            "joules_per_step": j_per_step,
            "joules_to_first_pass": (j_per_step * first_pass_step) if (j_per_step is not None and first_pass_step is not None) else None,
            "joules_to_sustained_pass": (j_per_step * first_sustained_step) if (j_per_step is not None and first_sustained_step is not None) else None,
            "total_train_joules": train_j_accum if nvml.available else None,
            "peak_allocated_mib": peak_mib,
            "state_bytes_per_param": state_info["state_bytes_per_param"],
        }

    try:
        return _exec(seed=0)
    finally:
        nvml.shutdown()


def benchmark_joules_to_pass_grid100(
    idea_id: str,
    device: torch.device,
    *,
    optimizer_family_override: Optional[str] = None,
    recipe_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run grid100 (7000 steps) measuring hardware Joules/step & Joules-to-first-PASS."""
    return benchmark_joules_to_pass_toy100(
        idea_id,
        "grid100",
        device,
        optimizer_family_override=optimizer_family_override,
        recipe_extra=recipe_extra,
    )


def benchmark_joules_to_pass_staggered100(
    idea_id: str,
    device: torch.device,
    *,
    optimizer_family_override: Optional[str] = None,
    recipe_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run staggered100 (7000 steps) measuring hardware Joules/step & Joules-to-first-PASS."""
    return benchmark_joules_to_pass_toy100(
        idea_id,
        "staggered100",
        device,
        optimizer_family_override=optimizer_family_override,
        recipe_extra=recipe_extra,
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Law 4 Optimizer Energy, Latency & Mechanistic Profiler")
    parser.add_argument(
        "--mode",
        choices=("dcgan-profile", "mechanistic", "energy-benchmark", "phase1"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--idea", default="unified-v11")
    parser.add_argument("--ideas", nargs="*", default=None, help="Idea IDs for energy-benchmark")
    parser.add_argument("--dcgan-tasks", nargs="*", default=["dcgan_performance_r256", "dcgan_performance_r64", "dcgan_performance_w128"])
    parser.add_argument("--energy-steps", type=int, default=50)
    parser.add_argument("--breakdown-steps", type=int, default=20)
    parser.add_argument("--skip-r256", action="store_true")
    parser.add_argument("--skip-ring16", action="store_true")
    parser.add_argument("--skip-grid100", action="store_true")
    parser.add_argument("--include-staggered100", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_data: Dict[str, Any] = {
        "mode": args.mode,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if args.mode in ("dcgan-profile", "phase1"):
        dcgan_results: Dict[str, Any] = {}
        for tid in args.dcgan_tasks:
            print(f"[profiler] Profiling {args.idea} on {tid} ({args.energy_steps} sustained + {args.breakdown_steps} breakdown steps)...", flush=True)
            res = profile_dcgan_task(
                args.idea,
                tid,
                device,
                warmup_steps=5,
                energy_steps=args.energy_steps,
                breakdown_steps=args.breakdown_steps,
                run_breakdown=True,
            )
            dcgan_results[tid] = res
            s = res["sustained_uninstrumented"]
            print(
                f"  -> {tid}: median={s['step_ms']['median']:.2f} ms/step, "
                f"W={s['avg_power_watts_from_energy']:.1f}W, J/step={s['joules_per_step']:.4f} J, "
                f"state={res['optimizer_state']['state_bytes_per_param']:.2f} B/param, "
                f"peak={s['peak_allocated_mib']:.1f} MiB",
                flush=True,
            )
        out_data["dcgan_profile"] = dcgan_results

    if args.mode in ("mechanistic", "phase1"):
        regimes = [
            ("ring16_acquisition", 200, [1, 2, 5, 10, 20, 50, 100, 150, 200]),
            ("grid100_mlp16", 200, [1, 2, 5, 10, 20, 50, 100, 150, 200]),
            (
                "zero_tanh_rail_escape",
                1200,
                [1, 10, 50, 200, 500, 990, 1000, 1001, 1002, 1005, 1010, 1020, 1050, 1100, 1200],
            ),
            ("nash_overpowered_critic_ring16", 200, [1, 2, 5, 10, 20, 50, 100, 150, 200]),
            ("nash_overpowered_generator_ring16", 200, [1, 2, 5, 10, 20, 50, 100, 150, 200]),
        ]
        mech_results: Dict[str, Any] = {}
        for tid, n_steps, s_steps in regimes:
            print(f"[profiler] Mechanistic diagnostics: {args.idea} on {tid} ({n_steps} steps)...", flush=True)
            res = run_mechanistic_regime(args.idea, tid, device, num_steps=n_steps, sample_steps=s_steps)
            mech_results[tid] = res
            sm = res["summary"]
            print(
                f"  -> {tid}: G grad ratio med={sm['g_layer_grad_ratio_median']}, "
                f"G cond med={sm['g_grad_cond_median']}, "
                f"SGD dD/dG med={sm['ratio_d_over_g_sgd_median']}, "
                f"Equil dD/dG med={sm['ratio_d_over_g_equil_median']}",
                flush=True,
            )
        out_data["mechanistic"] = mech_results

    if args.mode == "energy-benchmark":
        idea_list = args.ideas or [args.idea]
        bench_results: Dict[str, Any] = {}
        for iid in idea_list:
            print(f"[energy-benchmark] Candidate: {iid}", flush=True)
            entry: Dict[str, Any] = {}
            if not args.skip_r256:
                entry["dcgan_r256"] = profile_dcgan_task(
                    iid,
                    "dcgan_performance_r256",
                    device,
                    warmup_steps=5,
                    energy_steps=args.energy_steps,
                    breakdown_steps=0,
                    run_breakdown=False,
                )
            if not args.skip_ring16:
                entry["ring16_acquisition"] = benchmark_joules_to_pass_ring16(iid, device)
            if not args.skip_grid100:
                entry["grid100"] = benchmark_joules_to_pass_grid100(iid, device)
            if args.include_staggered100:
                entry["staggered100"] = benchmark_joules_to_pass_staggered100(iid, device)
            bench_results[iid] = entry
        out_data["energy_benchmark"] = bench_results

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, sort_keys=True) + "\n")
    print(f"[profiler] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
