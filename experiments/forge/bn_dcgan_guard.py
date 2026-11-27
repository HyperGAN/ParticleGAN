"""Forge adapter and fail-closed grader for the Radford 32x32 BN-DCGAN B=256 diagnostic guard.

This diagnostic task exercises the complete public ``GANTrainer`` / ``Recipe``
update path on a standard Radford 32x32 BN-DCGAN (``BatchNorm2d`` in both G and
D, ``z_dim=128``, ``batch_size=256``) over a hermetic continuous natural-image-
regime 32x32 RGB distribution (pooled 8x8 ``med_nn / med_all > 0.50``).

It guards both:
1. Served clean live distribution quality (offline deterministic multi-scale
   convolutional/spatial/spectral FID, pixel diversity ratio, 10-class coverage
   and mass TV, radial spectral L1, and ``finite_fraction == 1.0``), AND
2. Synchronized B=256 step latency (independently recomputed from a SHA-256-
   verified ``updates.jsonl`` trace, enforced on CUDA to surface 40+ ms/step
   O(B^2) regressions while reporting CPU latency honestly without false fails).

Scoped strictly as a nonqualifying ``research_diagnostic`` guard.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from particlegan import unwrap_prior

from .api import CapabilityError, task_formulation_context
from .artifacts import save_provenance_checkpoint
from .contracts import atomic_json, canonical, file_hash
from .priors import prior_policy_receipt, recipe_owned_prior
from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
from .state import state_digest
from .telemetry import PhaseTimer

ADAPTER = "bn_dcgan_guard"
KIND = "bn_dcgan_rgb32_b256_diagnostic"
EVALUATOR = "experiments.forge.bn_dcgan_guard:grade"
SAMPLING_LAW = PUBLIC_PRIOR_CLEAN

_VERIFIED_ORACLES: dict[str, bool] = {}


def _is_bn_dcgan_guard_task(task: dict) -> bool:
    return (
        isinstance(task, dict)
        and task.get("adapter") == ADAPTER
        and isinstance(task.get("evaluation"), dict)
        and task["evaluation"].get("kind") == KIND
    )


def _verify_oracle_controls() -> None:
    from benchmarks.toy_audit import api_bn_dcgan_rgb32 as provider

    mod_file = getattr(provider, "__file__", None)
    digest = file_hash(Path(mod_file)) if mod_file and Path(mod_file).is_file() else "memory"
    if _VERIFIED_ORACLES.get(digest):
        return
    controls = provider.oracle_controls()
    if not isinstance(controls, dict) or not controls:
        raise ValueError("bn_dcgan_guard provider returned empty oracle_controls")
    for name, control in controls.items():
        if not isinstance(control, dict) or control.get("passed") is not control.get("expected_pass"):
            raise ValueError(f"bn_dcgan_guard scorer oracle control failed: {name}")
    _VERIFIED_ORACLES[digest] = True


def _contract_payload(spec: dict[str, Any], *, g_params: int, d_params: int) -> dict[str, Any]:
    res = spec["resources"]
    prior = spec["prior"]
    model = spec["model"]
    data = spec["data"]
    return {
        "schema_version": 1,
        "kind": KIND,
        "prior_kind": str(prior["kind"]),
        "prior_sigma": float(prior["sigma"]),
        "num_particles": int(res["num_particles"]),
        "z_dim": int(res["z_dim"]),
        "batch_size": int(res["batch_size"]),
        "sampling_law": str(spec["sampling_law"]),
        "model_family": str(model["family"]),
        "generator_batchnorm": bool(model["generator_batchnorm"]),
        "discriminator_batchnorm": bool(model["discriminator_batchnorm"]),
        "parameter_count_generator": int(g_params),
        "parameter_count_discriminator": int(d_params),
        "data_distribution": str(data["distribution"]),
        "cuda_latency_gate": deepcopy(spec["cuda_latency_gate"]),
    }


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite_model(*models: nn.Module) -> bool:
    for model in models:
        tensors = [
            value
            for value in model.state_dict().values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ]
        if tensors and not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
            return False
    return True


def _count_updates(optimizer, parameters) -> int:
    values = [
        int(optimizer.state[p]["step"])
        for p in parameters
        if p in optimizer.state and "step" in optimizer.state[p]
    ]
    return min(values) if values else 0


def validate_task(
    task: dict,
    *,
    root: Path | None = None,
    allow_software_prefix: bool = False,
) -> dict[str, Any]:
    """Validate the frozen BN-DCGAN diagnostic task contract and oracle controls."""
    from benchmarks.toy_audit.api_bn_dcgan_rgb32 import (
        CUDA_LATENCY_GATE,
        SPEC,
        THRESHOLDS,
    )

    if not _is_bn_dcgan_guard_task(task):
        raise ValueError(f"bn_dcgan_guard requires adapter={ADAPTER!r} and evaluation.kind={KIND!r}")
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        raise TypeError("bn_dcgan_guard task requires execution and evaluation objects")
    if execution.get("host") != KIND:
        raise ValueError(f"bn_dcgan_guard execution.host must equal {KIND!r}")
    if execution.get("initializer") != "deterministic_orthogonal":
        raise ValueError("bn_dcgan_guard requires named deterministic_orthogonal initialization")

    expected_spec = deepcopy(SPEC)
    actual_spec = execution.get("host_definition")
    if not isinstance(actual_spec, dict):
        raise TypeError("bn_dcgan_guard execution.host_definition must be an object")

    is_prefix = allow_software_prefix and str(task.get("id", "")).endswith("-software-prefix")
    steps = execution.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("bn_dcgan_guard execution.steps must be a positive integer")

    if is_prefix:
        expected_cmp = deepcopy(expected_spec)
        for field in ("steps", "eval_samples", "observations", "minimum_stable_checks"):
            if field in actual_spec:
                expected_cmp[field] = actual_spec[field]
        if actual_spec != expected_cmp:
            raise ValueError("bn_dcgan_guard host_definition differs from the frozen task specification")
    else:
        if actual_spec != expected_spec or steps != int(expected_spec["steps"]):
            raise ValueError("bn_dcgan_guard host_definition or step budget differs from the frozen specification")

    if execution.get("resources") != actual_spec["resources"]:
        raise ValueError("bn_dcgan_guard execution.resources must match host_definition.resources")

    if not recipe_owned_prior(task):
        raise ValueError("bn_dcgan_guard requires prior_contract='recipe_owned_v1'")
    prior = execution.get("prior")
    expected_prior = expected_spec["prior"]
    if (
        not isinstance(prior, dict)
        or prior.get("kind") != expected_prior["kind"]
        or float(prior.get("sigma", -1.0)) != float(expected_prior["sigma"])
        or prior.get("standardize") is not expected_prior["standardize"]
        or "learnable" in prior
    ):
        raise ValueError("bn_dcgan_guard execution.prior differs from its frozen MoG specification")

    if evaluation.get("evaluator") != EVALUATOR:
        raise ValueError(f"bn_dcgan_guard requires evaluation.evaluator == {EVALUATOR!r}")
    if evaluation.get("qualification_input") is not False:
        raise ValueError("bn_dcgan_guard must declare evaluation.qualification_input = false")
    if evaluation.get("scoring_weights", "live") != "live":
        raise ValueError("bn_dcgan_guard scores live unaveraged weights")
    if evaluation.get("sampling_law") != SAMPLING_LAW or evaluation.get("eval_output_noise") != "clean":
        raise ValueError("bn_dcgan_guard requires clean public-prior evaluation sampling")
    if evaluation.get("thresholds") != THRESHOLDS:
        raise ValueError("bn_dcgan_guard evaluation.thresholds differ from the frozen scorer")

    if evaluation.get("cuda_latency_gate") != CUDA_LATENCY_GATE:
        raise ValueError("bn_dcgan_guard evaluation.cuda_latency_gate differs from the frozen contract")

    expected_min_stable = int(actual_spec["minimum_stable_checks"] if is_prefix else expected_spec["minimum_stable_checks"])
    min_stable = evaluation.get("minimum_stable_checks")
    if type(min_stable) is not int or min_stable < 1 or (not is_prefix and min_stable != expected_min_stable):
        raise ValueError(f"bn_dcgan_guard requires minimum_stable_checks == {expected_spec['minimum_stable_checks']}")

    expected_obs = int(actual_spec["observations"] if is_prefix else expected_spec["observations"])
    obs_count = evaluation.get("observations")
    if type(obs_count) is not int or obs_count < 1 or (not is_prefix and obs_count != expected_obs):
        raise ValueError(f"bn_dcgan_guard requires {expected_spec['observations']} scheduled observations")

    expected_eval_n = int(actual_spec["eval_samples"] if is_prefix else expected_spec["eval_samples"])
    eval_samples = evaluation.get("eval_samples")
    if type(eval_samples) is not int or eval_samples < 16 or (not is_prefix and eval_samples != expected_eval_n):
        raise ValueError(f"bn_dcgan_guard requires eval_samples == {expected_spec['eval_samples']}")

    if root is not None:
        for rel_path, expected_sha in evaluation.get("sources", {}).items():
            if file_hash(Path(root) / rel_path) != expected_sha:
                raise ValueError(f"bn_dcgan_guard source binding differs: {rel_path}")

    _verify_oracle_controls()
    return deepcopy(actual_spec if is_prefix else expected_spec)


def bn_dcgan_context(
    request: dict,
    task: dict,
    device: str,
    *,
    root: Path | None = None,
    allow_software_prefix: bool = False,
):
    validate_task(task, root=root, allow_software_prefix=allow_software_prefix)
    context = task_formulation_context(
        request["candidate"], task, request.get("protocol"), device=device, root=root
    )
    if context.initializer != "deterministic_orthogonal":
        raise CapabilityError(["bn_dcgan_guard requires named deterministic_orthogonal initialization"])
    if context.recipe.prior_code_law != "table":
        raise CapabilityError(["bn_dcgan_guard requires Recipe.prior_code_law == 'table'"])
    if context.prior_config.get("kind") != "mog":
        raise CapabilityError(["bn_dcgan_guard requires an effective MoGParticlePrior"])
    if context.recipe.constraint_geometry_mode == "strict_progress":
        raise CapabilityError(["strict_progress cannot execute standard DCGAN BatchNorm running buffers"])
    return context


def preflight(task: dict, candidate: dict, *, root: Path | None = None) -> list[str]:
    try:
        bn_dcgan_context(
            {"candidate": candidate, "protocol": {"seed": 0}},
            task,
            "cpu",
            root=root,
            allow_software_prefix=str(task.get("id", "")).endswith("-software-prefix"),
        )
    except CapabilityError as error:
        return list(error.blockers)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return [str(error)]
    return []


def _render_training_gif(task: dict, frames: list[dict], output_dir: Path, *, passed: bool, full_budget: bool) -> dict:
    from benchmarks.toy_audit.api_run import render_gif

    steps = int(task["execution"]["steps"])
    path = Path(output_dir) / "goal.gif"
    render_gif(
        {
            "id": task["id"],
            "goal": task.get("description") or task["execution"]["host_definition"]["goal"],
            "default_steps": steps,
        },
        frames,
        path,
        full_budget=full_budget,
        requested_steps=steps,
        final_verdict="PASS" if passed else "FAIL",
    )
    return {"path": str(path.resolve()), "sha256": file_hash(path), "frames": len(frames)}


def run(request: dict, task: dict, output_dir: Path, device: str) -> dict:
    """Execute the Radford 32x32 BN-DCGAN B=256 diagnostic guard through the public API."""
    is_prefix = str(task.get("id", "")).endswith("-software-prefix")
    req_dev = torch.device(device)
    dev_ctx = torch.cuda.device(req_dev) if req_dev.type == "cuda" else nullcontext()
    with dev_ctx:
        return _run_on_device(request, task, output_dir, device, is_prefix=is_prefix)


def _run_on_device(request: dict, task: dict, output_dir: Path, device: str, *, is_prefix: bool) -> dict:
    from benchmarks.toy_audit import api_bn_dcgan_rgb32 as provider
    from benchmarks.toy_audit import api_contract
    from .adapters import _checkpoints, _event, _finite_tree, _host_receipt
    from .mechanisms import MechanismAudit, mechanism_blockers
    from .optimizer_diagnostics import attach

    context = bn_dcgan_context(request, task, device, allow_software_prefix=is_prefix)
    spec = deepcopy(task["execution"]["host_definition"])
    steps = int(task["execution"]["steps"])
    warmup_updates = int(task["evaluation"]["cuda_latency_gate"]["warmup_updates"])
    eval_samples = int(task["evaluation"]["eval_samples"])
    seed = int((request.get("protocol") or {}).get("seed", 0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "updates.jsonl"
    if trace_path.exists():
        raise FileExistsError("bn_dcgan_guard updates.jsonl already exists; use an exclusive directory")

    fixture = provider.build_case(
        KIND,
        device=context.device,
        seed=seed,
        max_steps=steps,
        components=context,
        spec_override=spec,
    )
    trainer = fixture.trainer

    # Confirm both learned networks genuinely contain BatchNorm2d layers.
    has_g_bn = any(isinstance(m, nn.BatchNorm2d) for m in trainer.G.modules())
    has_d_bn = any(isinstance(m, nn.BatchNorm2d) for m in trainer.D.modules())
    if not (has_g_bn and has_d_bn):
        raise RuntimeError("bn_dcgan_guard requires BatchNorm2d in both generator and discriminator")

    initial_params = {
        role: state_digest(dict(model.named_parameters()))
        for role, model in (
            ("generator", trainer.G),
            ("discriminator", trainer.D),
            ("prior", unwrap_prior(trainer.prior)),
        )
    }
    host = _host_receipt(spec, None, trainer.G, trainer.D)
    host["initial_prior_sha256"] = state_digest(trainer.prior.state_dict())
    g_param_count = sum(p.numel() for p in trainer.G.parameters())
    d_param_count = sum(p.numel() for p in trainer.D.parameters())

    checkpoints = set(_checkpoints(task))
    audit = MechanismAudit(context.recipe, trainer.opt_d, [trainer.opt_g])
    timing = PhaseTimer(
        synchronize=(lambda: torch.cuda.synchronize(context.device))
        if context.device.type == "cuda"
        else None
    )
    optimizer_diagnostics = attach(
        {"G": trainer.opt_g, "D": trainer.opt_d},
        trainer.D,
        output,
        checkpoints,
        prior=trainer.prior,
    )

    observations: list[dict] = []
    observation_margins: list[dict] = []
    frames: list[dict] = []
    rng_audits: list[dict] = []
    started = time.monotonic()
    finite = True

    with trace_path.open("x", buffering=1, encoding="utf-8") as trace_stream:
        for step in range(1, steps + 1):
            phase = "warmup" if (step - 1) < warmup_updates else "measured"
            with timing.measure("training_updates"):
                update = fixture.step(collect_stats=True)
            duration_ms = float(update["duration_ms"])
            losses = {
                k: float(v)
                for k, v in update.items()
                if k != "duration_ms" and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            step_finite = bool(losses) and all(math.isfinite(v) for v in losses.values()) and math.isfinite(duration_ms)
            finite = finite and step_finite and _finite_tree(update)
            trace_row = {
                "step": int(trainer.completed_steps),
                "phase": phase,
                "duration_ms": duration_ms,
                "finite_losses": step_finite,
                "losses": {k: (v if math.isfinite(v) else str(v)) for k, v in losses.items()},
            }
            trace_stream.write(canonical(trace_row) + "\n")

            p_stats = (
                update.get("penalty_stats")
                if isinstance(update, dict) and isinstance(update.get("penalty_stats"), dict)
                else getattr(getattr(trainer, "penalty", None), "last_stats", None)
            )
            audit.observe_penalty(p_stats or {})
            if not finite:
                raise FloatingPointError(f"nonfinite bn_dcgan_guard public-API update at step {step}")
            if step not in checkpoints:
                continue

            before = context.streams.audit()
            global_before = torch.get_rng_state().clone()
            cuda_before = (
                torch.cuda.get_rng_state(context.device).clone()
                if context.device.type == "cuda"
                else None
            )
            with timing.measure("evaluation"):
                raw_observed = fixture.observe(n=eval_samples, seed=seed + 10000)
                observed = api_contract.validate_observation(raw_observed)
            metrics = deepcopy(observed["metrics"])
            margins = deepcopy(raw_observed.get("margins", {}))
            finite = finite and _finite_tree(metrics) and _finite_tree(margins)

            allowed = [
                key
                for key, binding in context.streams.manifest()["bindings"].items()
                if binding["family"] == "eval"
            ]
            rng_audit = context.streams.compare(before, context.streams.audit(), allowed=allowed)
            if not torch.equal(global_before, torch.get_rng_state()):
                rng_audit["unintended_rng_deviations"] += 1
                rng_audit["unintended_streams"].append("global_cpu")
            if cuda_before is not None and not torch.equal(cuda_before, torch.cuda.get_rng_state(context.device)):
                rng_audit["unintended_rng_deviations"] += 1
                rng_audit["unintended_streams"].append("global_cuda")
            rng_audits.append(rng_audit)

            row = {"step": step, **metrics}
            observations.append(row)
            observation_margins.append({
                "step": step,
                "margins": margins,
                "min_margin": raw_observed.get("min_margin"),
            })
            frames.append({
                "step": step,
                "passed": observed["passed"],
                "failed_bounds": list(observed["failed_bounds"]),
                "metrics": dict(metrics),
                "views": observed["views"],
            })
            compact = {
                k: metrics[k]
                for k in (
                    "finite_fraction",
                    "feature_fid",
                    "pixel_diversity_ratio",
                    "class_coverage",
                    "class_mass_tv",
                    "spectral_l1",
                    "critic_margin",
                    "update_ms_median",
                )
                if k in metrics
            }
            _event(
                "observation",
                task=task["id"],
                step=step,
                metrics=compact,
                metric_passed=observed["passed"],
                failed_bounds=observed["failed_bounds"],
            )

    finite_model = _finite_model(trainer.G, trainer.D, trainer.prior)
    final_params = {
        role: state_digest(dict(model.named_parameters()))
        for role, model in (
            ("generator", trainer.G),
            ("discriminator", trainer.D),
            ("prior", unwrap_prior(trainer.prior)),
        )
    }
    changed_g = initial_params["generator"] != final_params["generator"]
    changed_d = initial_params["discriminator"] != final_params["discriminator"]

    timing_info = fixture.timing_summary(warmup_updates=warmup_updates)
    min_stable = int(task["evaluation"]["minimum_stable_checks"])
    obs_target = int(task["evaluation"]["observations"])
    quality_ok = (
        len(frames) == obs_target
        and len(frames) >= min_stable
        and all(frame["passed"] for frame in frames[-min_stable:])
    )
    max_ms = float(task["evaluation"]["cuda_latency_gate"]["max_ms"])
    latency_ok = (
        True
        if context.device.type != "cuda"
        else (
            timing_info["update_ms_median"] is not None
            and float(timing_info["update_ms_median"]) <= max_ms
        )
    )
    gif_receipt = _render_training_gif(
        task,
        frames,
        output,
        passed=bool(quality_ok and latency_ok and finite and finite_model and changed_g and changed_d),
        full_budget=(not is_prefix and len(frames) == obs_target),
    )

    state = fixture.state_dict()
    finite = finite and finite_model and _finite_tree(state)

    counts = {
        "generator": _count_updates(trainer.opt_g, trainer.G.parameters()),
        "discriminator": _count_updates(trainer.opt_d, trainer.D.parameters()),
        "prior": _count_updates(trainer.opt_g, trainer.prior.parameters()),
    }
    mechanisms = audit.receipt()
    a2 = trainer.prior_mechanisms["a2"]
    hooks = (
        trainer.opt_d.record.observed_steps == trainer.completed_steps
        and (not a2["requested"] or (a2["enabled"] and trainer.latent_history is not None))
        and not mechanism_blockers(mechanisms)
    )
    guards = {
        "all_finite": finite,
        "optimizer_updates": counts,
        "hooks_exercised": hooks,
        "mechanism_audit": mechanisms,
        "unintended_rng_deviations": sum(row["unintended_rng_deviations"] for row in rng_audits),
    }

    runtime_info: dict[str, Any] = {
        "device": str(context.device),
        "device_type": context.device.type,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(context.device) if context.device.type == "cuda" else None
        ),
    }

    evidence = {
        "observations": observations,
        "live": observations[-1],
        "final_margins": observation_margins[-1]["margins"],
        "observation_margins": observation_margins,
        "host": host,
        "initial_parameters": initial_params,
        "final_parameters": final_params,
        "batch_sequence_sha256": fixture.batch_sequence_sha256,
        "bn_dcgan_guard_contract": _contract_payload(spec, g_params=g_param_count, d_params=d_param_count),
        "actual_training_gif": gif_receipt,
        "artifacts": {
            "updates": {
                "path": str(trace_path.resolve()),
                "sha256": file_hash(trace_path),
            },
            "gif": {
                "path": gif_receipt["path"],
                "sha256": gif_receipt["sha256"],
            },
        },
        "runtime": runtime_info,
        "timing": timing_info,
        "validity": {
            "finite_model_state": finite_model,
            "changed_generator": changed_g,
            "changed_discriminator": changed_d,
            "generator_has_batchnorm2d": has_g_bn,
            "discriminator_has_batchnorm2d": has_d_bn,
        },
        "scoring_weights": "live",
        "guards": guards,
        "rng_audits": rng_audits,
        **executed_receipt(SAMPLING_LAW, eval_output_noise="clean"),
    }
    if recipe_owned_prior(task):
        evidence["prior_policy"] = prior_policy_receipt(context.recipe, trainer.prior)
    if optimizer_diagnostics is not None:
        evidence["optimizer_diagnostics"] = optimizer_diagnostics.receipt()

    checkpoint = {"fixture": state, "streams": context.streams.state_dict()}
    evidence["provenance_checkpoint"] = save_provenance_checkpoint(
        output, checkpoint, completed_steps=fixture.completed_steps
    )
    if task["execution"].get("produces_state", False):
        torch.save(checkpoint, output / "state.pt")

    receipt = {
        **context.receipt(),
        "api_components": list(fixture.api_components),
        "evidence": evidence,
        "qualification_input": False,
        "qualification_reuse": False,
        "cost": {
            "optimizer_updates": counts,
            "completed_steps": fixture.completed_steps,
            "adapter_loop_seconds": time.monotonic() - started,
            "phase_timing": timing.snapshot(),
        },
        "scope": "integration_demo_only" if is_prefix else "research_diagnostic",
        "declared_task_updates": steps,
        "execution_limit": steps,
    }
    atomic_json(output / "adapter-receipt.json", receipt)
    return receipt


def _verdict(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "gate_status": status,
        "reasons": [reason],
        "qualification_input": False,
        "qualification_reuse": False,
        **extra,
    }


def grade(task: dict, evidence: dict) -> dict[str, Any]:
    """Independently grade quality observations and SHA-256-verified B=256 step latency."""
    from benchmarks.toy_audit.api_bn_dcgan_rgb32 import build_models

    try:
        spec = validate_task(task)
    except (ValueError, TypeError, KeyError) as error:
        return _verdict("INVALID", str(error))

    g_ref, d_ref = build_models(spec, z_dim=int(spec["resources"]["z_dim"]))
    expected_contract = _contract_payload(
        spec,
        g_params=sum(p.numel() for p in g_ref.parameters()),
        d_params=sum(p.numel() for p in d_ref.parameters()),
    )
    observed_contract = evidence.get("bn_dcgan_guard_contract")
    if not isinstance(observed_contract, dict):
        return _verdict("INCOMPLETE", "missing observed bn_dcgan_guard execution contract")
    if observed_contract != expected_contract:
        return _verdict("INVALID", "observed bn_dcgan_guard contract differs from the frozen task")

    gif = evidence.get("actual_training_gif")
    if not isinstance(gif, dict) or not _hex_digest(gif.get("sha256")):
        return _verdict("INCOMPLETE", "missing hashed actual-training GIF")

    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict) or "device_type" not in runtime:
        return _verdict("INCOMPLETE", "missing runtime device_type provenance")
    device_type = runtime.get("device_type")
    if device_type not in {"cpu", "cuda"}:
        return _verdict("INVALID", f"unsupported runtime device_type: {device_type!r}")

    # Independently verify and recompute latency from the hashed updates.jsonl trace.
    artifacts = evidence.get("artifacts")
    trace = artifacts.get("updates") if isinstance(artifacts, dict) else None
    if not isinstance(trace, dict) or not trace.get("path") or not _hex_digest(trace.get("sha256")):
        return _verdict("INCOMPLETE", "missing hashed per-update timing trace")
    trace_path = Path(trace["path"])
    if not trace_path.is_file():
        return _verdict("INCOMPLETE", "hashed per-update timing trace file is unavailable")
    if file_hash(trace_path) != trace["sha256"]:
        return _verdict("INVALID", "per-update timing trace differs from its frozen artifact hash")

    steps = int(task["execution"]["steps"])
    latency_gate = task["evaluation"]["cuda_latency_gate"]
    warmup_updates = int(latency_gate["warmup_updates"])
    max_ms = float(latency_gate["max_ms"])
    batch_size = int(spec["resources"]["batch_size"])

    try:
        with trace_path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (OSError, ValueError) as error:
        return _verdict("INVALID", f"unreadable per-update timing trace: {error}")

    if any(not isinstance(row, dict) for row in rows):
        return _verdict("INVALID", "timing trace rows must be JSON objects")
    if [row.get("step") for row in rows] != list(range(1, len(rows) + 1)):
        return _verdict("INVALID", "update trace must contain each completed update exactly once in order")
    for idx, row in enumerate(rows):
        expected_phase = "warmup" if idx < warmup_updates else "measured"
        dur = row.get("duration_ms")
        if row.get("phase") != expected_phase or not _finite_number(dur) or float(dur) <= 0.0:
            return _verdict("INVALID", "timing trace has an invalid phase or duration")

    finite_trace = all(
        row.get("finite_losses") is True
        and isinstance(row.get("losses"), dict)
        and bool(row["losses"])
        and all(_finite_number(v) for v in row["losses"].values())
        for row in rows
    )
    validity = evidence.get("validity")
    if not isinstance(validity, dict):
        return _verdict("INCOMPLETE", "missing model validity audit")
    if not finite_trace or validity.get("finite_model_state") is False:
        return _verdict("FAIL", "public trainer produced nonfinite losses or model state")
    if len(rows) < steps:
        return _verdict("INCOMPLETE", f"update budget incomplete ({len(rows)}/{steps} updates)")
    if len(rows) > steps:
        return _verdict("INVALID", f"update trace exceeds declared budget ({len(rows)}/{steps} updates)")
    if validity.get("finite_model_state") is not True:
        return _verdict("INCOMPLETE", "missing actual finite model-state audit")
    if any(validity.get(k) is not True for k in ("changed_generator", "changed_discriminator")):
        return _verdict("FAIL", "both generator and discriminator parameters must change during training")

    measured_durations = [float(row["duration_ms"]) for row in rows[warmup_updates:]]
    if not measured_durations:
        return _verdict("INCOMPLETE", "no post-warmup measured updates recorded")
    arr = np.asarray(measured_durations, dtype=np.float64)
    update_ms_median = float(np.percentile(arr, 50))
    update_ms_p95 = float(np.percentile(arr, 95))
    images_per_second = float(1000.0 * batch_size / float(arr.mean()))

    # Independently recompute quality thresholds over the complete observation schedule.
    evaluation = task["evaluation"]
    thresholds = evaluation["thresholds"]
    obs_count = int(evaluation["observations"])
    expected_steps = sorted({math.ceil(i * steps / obs_count) for i in range(1, obs_count + 1)})
    raw_points = evidence.get("observations")
    if not isinstance(raw_points, list) or not raw_points:
        return _verdict("INCOMPLETE", "missing bn_dcgan_guard observation curve")
    points: list[dict[str, Any]] = []
    for entry in raw_points:
        if not isinstance(entry, dict) or type(entry.get("step")) is not int or entry["step"] < 0:
            return _verdict("INVALID", "observation entries must be objects with nonnegative integer steps")
        for metric_name, _, _ in thresholds:
            if metric_name not in entry or not _finite_number(entry[metric_name]):
                return _verdict("INVALID", f"missing or nonfinite observed metric {metric_name}")
        points.append(entry)
    observed_steps = [pt["step"] for pt in points]
    if any(b <= a for a, b in zip(observed_steps, observed_steps[1:])):
        return _verdict("INVALID", "observation steps must be unique and strictly increasing")
    if observed_steps != expected_steps:
        return _verdict("INCOMPLETE", f"all {len(expected_steps)} declared observation checkpoints are required")

    live = evidence.get("live")
    if (
        not isinstance(live, dict)
        or live.get("step") != steps
        or any(not _finite_number(live.get(metric_name)) for metric_name, _, _ in thresholds)
    ):
        return _verdict("INCOMPLETE", "finite final live metrics at the complete update budget are required")

    def _point_failures(pt: dict[str, Any]) -> list[str]:
        bad: list[str] = []
        for name, op, bound in thresholds:
            val = float(pt[name])
            b_val = float(bound)
            ok = (
                val <= b_val
                if op == "<="
                else val >= b_val
                if op == ">="
                else val == b_val
                if op == "=="
                else False
            )
            if not ok:
                bad.append(f"{name} {op} {b_val:g}")
        return bad

    min_stable = int(evaluation["minimum_stable_checks"])
    if len(points) < min_stable:
        return _verdict("INCOMPLETE", f"requires at least {min_stable} observations, got {len(points)}")
    tail = points[-min_stable:]
    tail_failures = [_point_failures(pt) for pt in tail]
    terminal_passed = sum(1 for errs in tail_failures if not errs)
    failed_bounds = list(dict.fromkeys(err for errs in tail_failures for err in errs))
    live_failures = _point_failures(live)
    for err in live_failures:
        if err not in failed_bounds:
            failed_bounds.append(err)
    quality_passed = (terminal_passed == len(tail)) and (len(live_failures) == 0)

    if device_type == "cuda":
        cuda_latency_passed: bool | None = update_ms_median <= max_ms
        cuda_latency_status = "PASS" if cuda_latency_passed else "FAIL"
        if not cuda_latency_passed:
            failed_bounds.append(f"update_ms_median <= {max_ms:g} (cuda: {update_ms_median:.2f}ms)")
    else:
        cuda_latency_passed = None
        cuda_latency_status = "not_applicable_cpu"

    passed = quality_passed and (cuda_latency_passed is not False)
    metrics = deepcopy(live)
    metrics.update(
        completed_updates=len(rows),
        measured_updates=len(measured_durations),
        update_ms_median=update_ms_median,
        update_ms_p95=update_ms_p95,
        images_per_second=images_per_second,
        cuda_latency_gate_status=cuda_latency_status,
        runtime_device_type=device_type,
    )
    if passed:
        reason = (
            f"recomputed {KIND} quality ({terminal_passed}/{len(tail)} terminal checks) "
            f"and B=256 latency ({update_ms_median:.2f}ms, cuda_gate={cuda_latency_status}); "
            "research diagnostic only"
        )
    else:
        reason = f"failed {KIND} guard: {', '.join(failed_bounds)}"
    return _verdict(
        "PASS" if passed else "FAIL",
        reason,
        metrics=metrics,
        evaluator_result={
            "passed": passed,
            "quality_passed": quality_passed,
            "terminal_passed": terminal_passed,
            "minimum_stable_checks": min_stable,
            "cuda_latency_passed": cuda_latency_passed,
            "cuda_latency_gate_status": cuda_latency_status,
            "cuda_latency_max_ms": max_ms,
            "update_ms_median": update_ms_median,
            "failed_bounds": failed_bounds,
        },
    )
