"""Forge adapter and fail-closed contract validation for imbalanced & non-complementary Nash toys."""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path

import torch
from particlegan import unwrap_prior

from .api import CapabilityError, task_formulation_context
from .contracts import file_hash
from .priors import recipe_owned_prior
from .sampling import PUBLIC_PRIOR_CLEAN
from .state import state_digest

IMBALANCE_NASH_KINDS = frozenset({
    "nash_overpowered_critic_ring16",
    "nash_overpowered_generator_ring16",
    "nash_unequal_width_hostile_imbalance",
})

EXPECTED_PRIOR_SIGMA = {
    "nash_overpowered_critic_ring16": 0.10,
    "nash_overpowered_generator_ring16": 0.10,
    "nash_unequal_width_hostile_imbalance": 0.025,
}


def _contract_payload(kind: str, spec: dict, *, g_params: int, d_params: int) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "prior_code_path": "MoGParticlePrior",
        "prior_sigma": float(EXPECTED_PRIOR_SIGMA[kind]),
        "generator_hidden": int(spec["generator"]["hidden"]),
        "generator_layers": int(spec["generator"]["layers"]),
        "generator_output_scale": float(spec["generator"]["output_scale"]),
        "generator_output_offset": [float(v) for v in spec["generator"]["output_offset"]],
        "discriminator_hidden": int(spec["discriminator"]["hidden"]),
        "discriminator_layers": int(spec["discriminator"]["layers"]),
        "discriminator_logit_gain": float(spec["discriminator"]["logit_gain"]),
        "discriminator_hostile_tilt": [float(v) for v in spec["discriminator"]["hostile_tilt"]],
        "parameter_count_generator": int(g_params),
        "parameter_count_discriminator": int(d_params),
    }


def validate_task(task: dict, *, root: Path | None = None, allow_software_prefix: bool = False) -> dict:
    from benchmarks.toy_audit.api_imbalance_nash import (
        HOST_DEFINITIONS,
        THRESHOLDS,
        imbalance_nash_oracle_controls,
    )

    del root
    if not isinstance(task, dict) or task.get("adapter") != "imbalance_nash":
        raise ValueError("imbalance_nash validator requires adapter='imbalance_nash'")
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        raise TypeError("imbalance_nash task requires execution and evaluation objects")
    kind = evaluation.get("kind")
    if not isinstance(kind, str) or kind not in IMBALANCE_NASH_KINDS:
        raise ValueError(f"unsupported imbalance_nash kind: {kind!r}")
    if execution.get("host") != kind:
        raise ValueError("imbalance_nash execution.host must match evaluation.kind")

    expected_spec = deepcopy(HOST_DEFINITIONS[kind])
    actual_spec = execution.get("host_definition")
    if not isinstance(actual_spec, dict):
        raise TypeError("imbalance_nash execution.host_definition must be an object")

    is_prefix = allow_software_prefix and str(task.get("id", "")).endswith("-software-prefix")
    steps = execution.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("imbalance_nash execution.steps must be a positive integer")
    if is_prefix:
        expected_cmp = {
            **expected_spec,
            "steps": actual_spec.get("steps", expected_spec["steps"]),
            "target_spec": {
                **expected_spec["target_spec"],
                "steps": actual_spec.get("target_spec", {}).get("steps", expected_spec["steps"]),
            },
        }
        if actual_spec != expected_cmp:
            raise ValueError("imbalance_nash host_definition differs from the frozen task specification")
    else:
        if actual_spec != expected_spec or steps != expected_spec["steps"]:
            raise ValueError("imbalance_nash host_definition or step budget differs from the frozen specification")

    if not recipe_owned_prior(task):
        raise ValueError("imbalance_nash tasks require prior_contract='recipe_owned_v1'")
    prior = execution.get("prior")
    expected_sigma = EXPECTED_PRIOR_SIGMA[kind]
    if (
        not isinstance(prior, dict)
        or prior.get("kind") != "mog"
        or float(prior.get("sigma", -1.0)) != expected_sigma
        or prior.get("standardize") is not False
        or "learnable" in prior
    ):
        raise ValueError(
            f"imbalance_nash task {kind} requires recipe-owned MoG prior with sigma={expected_sigma} and standardize=False"
        )

    if evaluation.get("thresholds") != THRESHOLDS[kind]:
        raise ValueError("imbalance_nash evaluation.thresholds differ from the frozen scorer")
    min_stable = evaluation.get("minimum_stable_checks")
    if type(min_stable) is not int or min_stable != 5:
        raise ValueError("imbalance_nash tasks require minimum_stable_checks == 5")
    obs_count = evaluation.get("observations", 24)
    if type(obs_count) is not int or obs_count < 1 or (not is_prefix and obs_count != 24):
        raise ValueError("imbalance_nash tasks require 24 scheduled observations")
    eval_samples = evaluation.get("eval_samples", 4096)
    if type(eval_samples) is not int or eval_samples < 64 or (not is_prefix and eval_samples != 4096):
        raise ValueError("imbalance_nash tasks require eval_samples == 4096")

    for name, control in imbalance_nash_oracle_controls().items():
        if control["passed"] is not control["expected_pass"]:
            raise ValueError(f"imbalance_nash scorer oracle control failed: {name}")
    return expected_spec


def imbalance_nash_context(
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
        raise CapabilityError(["imbalance_nash requires named deterministic_orthogonal initialization"])
    if context.recipe.prior_code_law != "table":
        raise CapabilityError(["imbalance_nash requires Recipe.prior_code_law == 'table'"])
    if context.prior_config.get("kind") != "mog":
        raise CapabilityError(["imbalance_nash requires an effective MoGParticlePrior"])
    return context


def preflight(task: dict, candidate: dict, *, root: Path | None = None) -> list[str]:
    try:
        imbalance_nash_context(
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


def grade(task: dict, evidence: dict) -> tuple[str, str] | None:
    from benchmarks.toy_audit.api_imbalance_nash import build_models_for_spec

    try:
        spec = validate_task(task)
    except (ValueError, TypeError, KeyError) as error:
        return "INVALID", str(error)

    kind = task["evaluation"]["kind"]
    g_ref, d_ref = build_models_for_spec(spec, z_dim=int(spec["resources"]["z_dim"]))
    expected_contract = _contract_payload(
        kind,
        spec,
        g_params=sum(p.numel() for p in g_ref.parameters()),
        d_params=sum(p.numel() for p in d_ref.parameters()),
    )
    observed = evidence.get("imbalance_nash_contract")
    if not isinstance(observed, dict):
        return "INCOMPLETE", "missing observed imbalance_nash execution contract"
    if observed != expected_contract:
        return "INVALID", "observed imbalance_nash contract differs from the frozen task"

    gif = evidence.get("actual_training_gif")
    if not isinstance(gif, dict) or not isinstance(gif.get("sha256"), str) or not gif["sha256"]:
        return "INCOMPLETE", "missing actual-training GIF"

    steps = int(task["execution"]["steps"])
    count = int(task["evaluation"].get("observations", 24))
    expected_steps = sorted({math.ceil(i * steps / count) for i in range(1, count + 1)})
    points = evidence.get("observations")
    if not isinstance(points, list) or [pt.get("step") for pt in points if isinstance(pt, dict)] != expected_steps:
        return "INCOMPLETE", f"all {len(expected_steps)} declared imbalance_nash checkpoints are required"
    live = evidence.get("live")
    if not isinstance(live, dict) or live.get("step") != steps:
        return "INCOMPLETE", "final live metrics at the complete update budget are required"
    return None


def _render_training_gif(task: dict, frames: list[dict], output_dir: Path) -> dict:
    from benchmarks.toy_audit.api_run import render_gif

    steps = int(task["execution"]["steps"])
    min_stable = int(task["evaluation"].get("minimum_stable_checks", 5))
    obs_target = int(task["evaluation"].get("observations", 24))
    passed = (
        len(frames) == obs_target
        and len(frames) >= min_stable
        and all(frame["passed"] for frame in frames[-min_stable:])
    )
    path = Path(output_dir) / "goal.gif"
    render_gif(
        {
            "id": task["id"],
            "goal": task.get("description") or task["execution"]["host_definition"]["target"],
            "default_steps": steps,
        },
        frames,
        path,
        full_budget=(len(frames) == obs_target),
        requested_steps=steps,
        final_verdict="PASS" if passed else "FAIL",
    )
    return {"path": str(path.resolve()), "sha256": file_hash(path), "frames": len(frames)}


def run(request: dict, task: dict, output_dir: Path, device: str) -> dict:
    from benchmarks.toy_audit.api_imbalance_nash import (
        build_models_for_spec,
        sample_target_for_kind,
        score_imbalance_nash_kind,
    )

    from .adapters import _checkpoints, _event, _host_receipt, _Run

    is_prefix = str(task.get("id", "")).endswith("-software-prefix")
    context = imbalance_nash_context(request, task, device, allow_software_prefix=is_prefix)
    kind = task["evaluation"]["kind"]
    spec = task["execution"]["host_definition"]
    steps = int(task["execution"]["steps"])

    def _make_g():
        g_mod, _ = build_models_for_spec(spec, z_dim=context.recipe.z_dim)
        return g_mod

    def _make_d():
        _, d_mod = build_models_for_spec(spec, z_dim=context.recipe.z_dim)
        return d_mod

    g = context.construct(_make_g, component="generator").to(context.device)
    d = context.construct(_make_d, component="discriminator").to(context.device)
    trainer = context.build_trainer(g, d, max_steps=steps)

    # Verify non-complementary KEEP parameters survived deterministic_orthogonal_ intact.
    expected_offset = torch.tensor(spec["generator"]["output_offset"], dtype=torch.float32, device=g.shift_bias.device)
    expected_tilt = torch.tensor(spec["discriminator"]["hostile_tilt"], dtype=torch.float32, device=d.tilt_weight.device)
    if not torch.allclose(g.shift_bias.detach().to(torch.float32), expected_offset, atol=1e-6):
        raise RuntimeError("non-complementary generator shift_bias was unexpectedly overwritten during init")
    if not torch.allclose(d.tilt_weight.detach().to(torch.float32), expected_tilt, atol=1e-6):
        raise RuntimeError("non-complementary discriminator tilt_weight was unexpectedly overwritten during init")

    initial_params = {
        role: state_digest(dict(model.named_parameters()))
        for role, model in (("generator", g), ("discriminator", d), ("prior", unwrap_prior(trainer.prior)))
    }
    host_receipt = _host_receipt(spec, None, g, d)
    g_param_count = sum(p.numel() for p in g.parameters())
    d_param_count = sum(p.numel() for p in d.parameters())

    runner = _Run(context, trainer, output_dir, task, sampling_law=PUBLIC_PRIOR_CLEAN)
    data_rng = context.streams.generator("data", component="target", purpose="training", device="cpu")
    eval_samples = int(task["evaluation"].get("eval_samples", 4096))
    ref_target = runner.evaluate(
        lambda: sample_target_for_kind(
            kind,
            min(1024, eval_samples),
            context.streams.generator("eval", component="target", purpose="reference", device="cpu"),
            completed_steps=steps,
            device="cpu",
        )
    )

    checkpoints = set(_checkpoints(task))
    observations: list[dict] = []
    frames: list[dict] = []
    batch_digest = hashlib.sha256()

    def _observe(current_step: int) -> dict:
        samples = runner.sample(eval_samples).cpu()
        scored = score_imbalance_nash_kind(
            kind,
            samples,
            completed_steps=current_step,
            reference=ref_target,
        )
        frames.append(
            {
                "step": current_step,
                "passed": scored["passed"],
                "failed_bounds": list(scored["failed_bounds"]),
                "metrics": dict(scored["metrics"]),
                "views": scored["views"],
            }
        )
        return dict(scored["metrics"])

    for step in range(1, steps + 1):
        batch_cpu = sample_target_for_kind(
            kind, context.recipe.batch_size, data_rng, completed_steps=step - 1, device="cpu"
        )
        batch_digest.update(batch_cpu.numpy().tobytes())
        runner.step(batch_cpu.to(context.device))
        if step in checkpoints:
            row = runner.evaluate(lambda s=step: _observe(s))
            observations.append({"step": step, **row})
            compact = {
                k: v
                for k, v in row.items()
                if k in (
                    "modes",
                    "hq",
                    "mass_tv",
                    "sw1_normalized",
                    "component_core_covariance_error",
                    "resolved_core_covariance_error",
                    "min_mass_ratio",
                )
            }
            _event("observation", task=task["id"], step=step, metrics=compact)

    gif_receipt = _render_training_gif(task, frames, Path(output_dir))
    evidence = {
        "observations": observations,
        "live": observations[-1],
        "host": host_receipt,
        "initial_parameters": initial_params,
        "batch_sequence_sha256": batch_digest.hexdigest(),
        "imbalance_nash_contract": _contract_payload(kind, spec, g_params=g_param_count, d_params=d_param_count),
        "actual_training_gif": gif_receipt,
    }
    return runner.receipt(evidence, save_state=task["execution"].get("produces_state", False))
