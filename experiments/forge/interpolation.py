"""Forge adapter and fail-closed contract validation for discrete particle interpolation toys."""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path

from particlegan import unwrap_prior

from .api import CapabilityError, task_formulation_context
from .contracts import file_hash
from .priors import recipe_owned_prior
from .sampling import ENUMERATED_PRIOR_CLEAN
from .state import state_digest

INTERPOLATION_KINDS = frozenset({
    "interp_sparse_arc",
    "interp_img_patch_shift",
    "interp_saddle_barycentric",
})


def _contract_payload(kind: str, spec: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "prior_code_path": "ParticlePrior",
        "sigma": 0.0,
        "graph": spec["interpolation"]["graph"],
        "k_neighbors": int(spec["interpolation"]["k_neighbors"]),
    }


def validate_task(task: dict, *, root: Path | None = None, allow_software_prefix: bool = False) -> dict:
    from benchmarks.toy_audit.api_interpolation import (
        HOST_DEFINITIONS,
        THRESHOLDS,
        interpolation_oracle_controls,
    )

    del root
    if not isinstance(task, dict) or task.get("adapter") != "particle_interpolation":
        raise ValueError("particle_interpolation validator requires adapter='particle_interpolation'")
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        raise TypeError("interpolation task requires execution and evaluation objects")
    kind = evaluation.get("kind")
    if not isinstance(kind, str) or kind not in INTERPOLATION_KINDS:
        raise ValueError(f"unsupported particle interpolation kind: {kind!r}")
    if execution.get("host") != kind:
        raise ValueError("interpolation execution.host must match evaluation.kind")

    expected_spec = deepcopy(HOST_DEFINITIONS[kind])
    actual_spec = execution.get("host_definition")
    if not isinstance(actual_spec, dict):
        raise TypeError("interpolation execution.host_definition must be an object")

    is_prefix = allow_software_prefix and str(task.get("id", "")).endswith("-software-prefix")
    steps = execution.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("interpolation execution.steps must be a positive integer")
    if is_prefix:
        expected_cmp = {**expected_spec, "steps": actual_spec.get("steps", expected_spec["steps"])}
        if actual_spec != expected_cmp:
            raise ValueError("interpolation host_definition differs from the frozen task specification")
    else:
        if actual_spec != expected_spec or steps != expected_spec["steps"]:
            raise ValueError("interpolation host_definition or step budget differs from the frozen specification")

    if not recipe_owned_prior(task):
        raise ValueError("interpolation tasks require prior_contract='recipe_owned_v1'")
    prior = execution.get("prior")
    if (
        not isinstance(prior, dict)
        or prior.get("kind") != "particle_cloud"
        or prior.get("sigma") != 0.0
        or prior.get("standardize") is not False
        or "learnable" in prior
        or not isinstance(prior.get("exception_reason"), str)
        or not prior["exception_reason"].strip()
    ):
        raise ValueError(
            "interpolation tasks require recipe-owned zero-sigma particle_cloud prior with standardize=False and exception_reason"
        )

    if evaluation.get("thresholds") != THRESHOLDS[kind]:
        raise ValueError("interpolation evaluation.thresholds differ from the frozen scorer")
    min_stable = evaluation.get("minimum_stable_checks")
    if type(min_stable) is not int or min_stable != 5:
        raise ValueError("interpolation tasks require minimum_stable_checks == 5")
    obs_count = evaluation.get("observations", 24)
    if type(obs_count) is not int or obs_count < 1 or (not is_prefix and obs_count != 24):
        raise ValueError("interpolation tasks require 24 scheduled observations")

    for name, control in interpolation_oracle_controls().items():
        if control["passed"] is not control["expected_pass"]:
            raise ValueError(f"interpolation scorer oracle control failed: {name}")
    return expected_spec


def interpolation_context(
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
        raise CapabilityError(["particle interpolation requires named deterministic_orthogonal initialization"])
    if context.recipe.prior_code_law != "table":
        raise CapabilityError(["particle interpolation requires Recipe.prior_code_law == 'table'"])
    if context.prior_config.get("kind") != "particle_cloud" or float(context.prior_config.get("sigma", -1.0)) != 0.0:
        raise CapabilityError(["particle interpolation requires an effective zero-sigma ParticlePrior"])
    return context


def preflight(task: dict, candidate: dict, *, root: Path | None = None) -> list[str]:
    try:
        interpolation_context(
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
    try:
        spec = validate_task(task)
    except (ValueError, TypeError, KeyError) as error:
        return "INVALID", str(error)

    kind = task["evaluation"]["kind"]
    expected_contract = _contract_payload(kind, spec)
    observed = evidence.get("interpolation_contract")
    if not isinstance(observed, dict):
        return "INCOMPLETE", "missing observed particle-interpolation execution contract"
    if observed != expected_contract:
        return "INVALID", "observed particle-interpolation contract differs from the frozen task"

    gif = evidence.get("actual_training_gif")
    if not isinstance(gif, dict) or not isinstance(gif.get("sha256"), str) or not gif["sha256"]:
        return "INCOMPLETE", "missing actual-training GIF"

    steps = int(task["execution"]["steps"])
    count = int(task["evaluation"].get("observations", 24))
    expected_steps = sorted({math.ceil(i * steps / count) for i in range(1, count + 1)})
    points = evidence.get("observations")
    if not isinstance(points, list) or [pt.get("step") for pt in points if isinstance(pt, dict)] != expected_steps:
        return "INCOMPLETE", f"all {len(expected_steps)} declared interpolation checkpoints are required"
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
    from benchmarks.toy_audit.api_interpolation import (
        build_models_for_spec,
        evaluate_clean_codes,
        sample_target_for_kind,
        score_interpolation_kind,
    )

    from .adapters import _checkpoints, _event, _host_receipt, _Run

    is_prefix = str(task.get("id", "")).endswith("-software-prefix")
    context = interpolation_context(request, task, device, allow_software_prefix=is_prefix)
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
    initial_params = {
        role: state_digest(dict(model.named_parameters()))
        for role, model in (("generator", g), ("discriminator", d), ("prior", unwrap_prior(trainer.prior)))
    }
    host_receipt = _host_receipt(spec, None, g, d)

    runner = _Run(context, trainer, output_dir, task, sampling_law=ENUMERATED_PRIOR_CLEAN)
    data_rng = context.streams.generator("data", component="target", purpose="training", device="cpu")
    eval_samples = int(task["evaluation"].get("eval_samples", 512))
    ref_target = runner.evaluate(
        lambda: sample_target_for_kind(
            kind,
            eval_samples,
            context.streams.generator("eval", component="target", purpose="reference"),
            device="cpu",
        )
    )

    checkpoints = set(_checkpoints(task))
    observations: list[dict] = []
    frames: list[dict] = []
    batch_digest = hashlib.sha256()

    def _observe(current_step: int) -> dict:
        endpoints = trainer.sample(
            context.recipe.num_particles,
            fixed_first_n=True,
            generator=context.streams.generator("eval", component="live", purpose="enumerated_samples"),
        )
        z_table = unwrap_prior(trainer.prior).z.detach()
        scored = score_interpolation_kind(
            kind,
            z_table,
            endpoints,
            lambda codes: evaluate_clean_codes(trainer.G, codes, prior=trainer.prior),
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
        batch_cpu = sample_target_for_kind(kind, context.recipe.batch_size, data_rng, device="cpu")
        batch_digest.update(batch_cpu.numpy().tobytes())
        runner.step(batch_cpu.to(context.device))
        if step in checkpoints:
            row = runner.evaluate(lambda s=step: _observe(s))
            observations.append({"step": step, **row})
            _event("observation", task=task["id"], step=step, metrics=row)

    gif_receipt = _render_training_gif(task, frames, Path(output_dir))
    evidence = {
        "observations": observations,
        "live": observations[-1],
        "host": host_receipt,
        "initial_parameters": initial_params,
        "batch_sequence_sha256": batch_digest.hexdigest(),
        "interpolation_contract": _contract_payload(kind, spec),
        "actual_training_gif": gif_receipt,
    }
    return runner.receipt(evidence, save_state=task["execution"].get("produces_state", False))
