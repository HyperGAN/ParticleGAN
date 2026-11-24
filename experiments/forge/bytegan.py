"""Forge adapter and fail-closed contract validation for ByteGAN 256-byte stretch tasks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import torch

from particlegan import unwrap_prior

from .api import CapabilityError, task_formulation_context
from .artifacts import save_provenance_checkpoint
from .contracts import atomic_json, file_hash
from .priors import prior_noise_sigma, prior_policy_receipt, recipe_owned_prior
from .sampling import ENUMERATED_PRIOR_CLEAN, JOINT_WORDS_CLEAN, executed_receipt
from .state import state_digest
from .telemetry import PhaseTimer

BYTEGAN_KINDS = frozenset({
    "bytegan_hypercube8",
    "bytegan_bigan_dictionary",
})

ADAPTER_TO_KIND = {
    "bytegan": "bytegan_hypercube8",
    "bytegan_bigan": "bytegan_bigan_dictionary",
}


def _contract_payload(kind: str, spec: dict) -> dict:
    res = spec["resources"]
    model = spec["model"]
    return {
        "schema_version": 1,
        "kind": kind,
        "adapter": "bytegan" if kind == "bytegan_hypercube8" else "bytegan_bigan",
        "prior_code_path": "ParticlePrior",
        "sigma": 0.0,
        "num_particles": int(res["num_particles"]),
        "z_dim": int(res["z_dim"]),
        "bit_tol": float(spec["bit_tol"]),
        "model_family": str(model["family"]),
    }


def validate_task(task: dict, *, root: Path | None = None, allow_software_prefix: bool = False) -> dict:
    from benchmarks.toy_audit.api_bytegan import (
        HOST_DEFINITIONS,
        THRESHOLDS,
        bytegan_oracle_controls,
    )

    if not isinstance(task, dict) or task.get("adapter") not in ADAPTER_TO_KIND:
        raise ValueError("bytegan validator requires adapter in {'bytegan', 'bytegan_bigan'}")
    adapter = task["adapter"]
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        raise TypeError("bytegan task requires execution and evaluation objects")

    kind = evaluation.get("kind")
    expected_kind = ADAPTER_TO_KIND[adapter]
    if not isinstance(kind, str) or kind not in BYTEGAN_KINDS or kind != expected_kind:
        raise ValueError(f"bytegan adapter {adapter!r} requires evaluation.kind == {expected_kind!r}, got {kind!r}")
    if execution.get("host") != kind:
        raise ValueError("bytegan execution.host must match evaluation.kind")
    if adapter == "bytegan_bigan" and execution.get("execution_path") != "public_components":
        raise ValueError("bytegan_bigan requires execution.execution_path == 'public_components'")

    expected_spec = deepcopy(HOST_DEFINITIONS[kind])
    actual_spec = execution.get("host_definition")
    if not isinstance(actual_spec, dict):
        raise TypeError("bytegan execution.host_definition must be an object")

    is_prefix = allow_software_prefix and str(task.get("id", "")).endswith("-software-prefix")
    steps = execution.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("bytegan execution.steps must be a positive integer")
    if is_prefix:
        expected_cmp = {**expected_spec, "steps": actual_spec.get("steps", expected_spec["steps"])}
        if actual_spec != expected_cmp:
            raise ValueError("bytegan host_definition differs from the frozen task specification")
    else:
        if actual_spec != expected_spec or steps != expected_spec["steps"]:
            raise ValueError("bytegan host_definition or step budget differs from the frozen specification")

    if not recipe_owned_prior(task):
        raise ValueError("bytegan tasks require prior_contract='recipe_owned_v1'")
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
            "bytegan tasks require recipe-owned zero-sigma particle_cloud prior with standardize=False and exception_reason"
        )

    if evaluation.get("thresholds") != THRESHOLDS[kind]:
        raise ValueError("bytegan evaluation.thresholds differ from the frozen scorer")
    min_stable = evaluation.get("minimum_stable_checks")
    if type(min_stable) is not int or min_stable != 5:
        raise ValueError("bytegan tasks require minimum_stable_checks == 5")
    obs_count = evaluation.get("observations", 24)
    if type(obs_count) is not int or obs_count < 1 or (not is_prefix and obs_count != 24):
        raise ValueError("bytegan tasks require 24 scheduled observations")
    if evaluation.get("scoring_weights", "live") != "live":
        raise ValueError("bytegan tasks score live unaveraged weights")

    if root is not None:
        for rel_path, expected_sha in evaluation.get("sources", {}).items():
            if file_hash(Path(root) / rel_path) != expected_sha:
                raise ValueError(f"bytegan source binding differs: {rel_path}")

    for name, control in bytegan_oracle_controls().items():
        if control["passed"] is not control["expected_pass"]:
            raise ValueError(f"bytegan scorer oracle control failed: {name}")
    return expected_spec


def bytegan_context(
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
        raise CapabilityError(["bytegan requires named deterministic_orthogonal initialization"])
    if context.recipe.prior_code_law != "table":
        raise CapabilityError(["bytegan requires Recipe.prior_code_law == 'table'"])
    if (
        context.prior_config.get("kind") != "particle_cloud"
        or float(context.prior_config.get("sigma", -1.0)) != 0.0
        or prior_noise_sigma(context.prior_config) != 0.0
    ):
        raise CapabilityError(["bytegan requires an effective zero-sigma ParticlePrior"])
    if task.get("adapter") == "bytegan_bigan":
        if (
            context.recipe.model != "gan"
            or context.recipe.conditioning != "scalar"
            or context.recipe.encoder_mode != "none"
            or context.recipe.row_policy != "independent"
        ):
            raise CapabilityError([
                "bytegan_bigan binds its own encoder and joint adversarial objective; "
                "requires model='gan', conditioning='scalar', encoder_mode='none', row_policy='independent'"
            ])
    return context


def preflight(task: dict, candidate: dict, *, root: Path | None = None) -> list[str]:
    try:
        bytegan_context(
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
    observed = evidence.get("bytegan_contract")
    if not isinstance(observed, dict):
        return "INCOMPLETE", "missing observed bytegan execution contract"
    if observed != expected_contract:
        return "INVALID", "observed bytegan contract differs from the frozen task"

    gif = evidence.get("actual_training_gif")
    if not isinstance(gif, dict) or not isinstance(gif.get("sha256"), str) or not gif["sha256"]:
        return "INCOMPLETE", "missing actual-training GIF"

    steps = int(task["execution"]["steps"])
    count = int(task["evaluation"].get("observations", 24))
    expected_steps = sorted({math.ceil(i * steps / count) for i in range(1, count + 1)})
    points = evidence.get("observations")
    if not isinstance(points, list) or [pt.get("step") for pt in points if isinstance(pt, dict)] != expected_steps:
        return "INCOMPLETE", f"all {len(expected_steps)} declared bytegan checkpoints are required"
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


def _run_hypercube8(context, task: dict, output_dir: Path) -> dict:
    from benchmarks.toy_audit.api_bytegan import (
        build_models_for_spec,
        sample_byte_target,
        score_bytegan_hypercube8,
    )
    from .adapters import _checkpoints, _event, _host_receipt, _Run

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
        scored = score_bytegan_hypercube8(
            z_table,
            endpoints,
            bit_tol=float(spec["bit_tol"]),
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
        batch_cpu = sample_byte_target(context.recipe.batch_size, data_rng, device="cpu")
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
        "bytegan_contract": _contract_payload(kind, spec),
        "actual_training_gif": gif_receipt,
    }
    return runner.receipt(evidence, save_state=task["execution"].get("produces_state", False))


def _run_bigan_dictionary(context, request: dict, task: dict, output_dir: Path) -> dict:
    from benchmarks.toy_audit.api_bytegan import ByteGANBiGANFixture
    from .adapters import _checkpoints, _event, _finite_tree, _host_receipt
    from .mechanisms import MechanismAudit, mechanism_blockers
    from .optimizer_diagnostics import attach

    kind = task["evaluation"]["kind"]
    spec = task["execution"]["host_definition"]
    steps = int(task["execution"]["steps"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fixture = ByteGANBiGANFixture(
        device=context.device,
        seed=int((request.get("protocol") or {}).get("seed", 0)),
        recipe_name=None,
        max_steps=steps,
        components=context,
    )
    initial_params = {
        role: state_digest(dict(model.named_parameters()))
        for role, model in (
            ("generator", fixture.G),
            ("encoder", fixture.E),
            ("discriminator", fixture.D),
            ("prior", unwrap_prior(fixture.prior)),
        )
    }
    host = _host_receipt(spec, None, fixture.G, fixture.D)
    host["models"]["encoder"] = {
        "class": type(fixture.E).__module__ + "." + type(fixture.E).__qualname__,
        "parameters": sum(p.numel() for p in fixture.E.parameters()),
        "parameter_shapes": {name: list(p.shape) for name, p in fixture.E.named_parameters()},
        "initial_state_sha256": state_digest(fixture.E.state_dict()),
    }
    host["initial_prior_sha256"] = state_digest(fixture.prior.state_dict())

    checkpoints = set(_checkpoints(task))
    audit = MechanismAudit(context.recipe, fixture.opt_d, [fixture.opt_g])
    timing = PhaseTimer(
        synchronize=(lambda: torch.cuda.synchronize(context.device))
        if context.device.type == "cuda"
        else None
    )
    optimizer_diagnostics = attach(
        {"G": fixture.opt_g, "D": fixture.opt_d},
        fixture.D,
        output,
        checkpoints,
        prior=fixture.prior,
    )

    observations: list[dict] = []
    frames: list[dict] = []
    rng_audits: list[dict] = []
    started = time.monotonic()
    finite = True

    for step in range(1, steps + 1):
        with timing.measure("training_updates"):
            update = fixture.step()
        finite = finite and _finite_tree(update)
        audit.observe_penalty(fixture.penalty.last_stats)
        if not finite:
            raise FloatingPointError("nonfinite ByteGAN BiGAN public-API update")
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
            observed = fixture.observe()
        metrics = deepcopy(observed["metrics"])
        finite = finite and _finite_tree(metrics)
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
        frames.append(
            {
                "step": step,
                "passed": observed["passed"],
                "failed_bounds": list(observed["failed_bounds"]),
                "metrics": dict(metrics),
                "views": observed["views"],
            }
        )
        _event(
            "observation",
            task=task["id"],
            step=step,
            metrics=metrics,
            metric_passed=observed["passed"],
            failed_bounds=observed["failed_bounds"],
        )

    gif_receipt = _render_training_gif(task, frames, output)
    state = fixture.state_dict()
    finite = finite and _finite_tree(state)

    def _count(optimizer, parameters):
        values = [
            int(optimizer.state[p]["step"])
            for p in parameters
            if p in optimizer.state and "step" in optimizer.state[p]
        ]
        return min(values) if values else 0

    counts = {
        role: _count(opt, mod.parameters())
        for role, opt, mod in (
            ("generator", fixture.opt_g, fixture.G),
            ("encoder", fixture.opt_g, fixture.E),
            ("prior", fixture.opt_g, fixture.prior),
            ("discriminator", fixture.opt_d, fixture.D),
        )
    }
    mechanisms = audit.receipt()
    guards = {
        "all_finite": finite,
        "optimizer_updates": counts,
        "hooks_exercised": (
            fixture.opt_d.record.observed_steps == fixture.completed_steps
            and not mechanism_blockers(mechanisms)
        ),
        "mechanism_audit": mechanisms,
        "unintended_rng_deviations": sum(row["unintended_rng_deviations"] for row in rng_audits),
    }
    evidence = {
        "observations": observations,
        "live": observations[-1],
        "host": host,
        "initial_parameters": initial_params,
        "batch_sequence_sha256": fixture.batch_sequence_sha256,
        "bytegan_contract": _contract_payload(kind, spec),
        "actual_training_gif": gif_receipt,
        "scoring_weights": "live",
        "guards": guards,
        "rng_audits": rng_audits,
        **executed_receipt(JOINT_WORDS_CLEAN, eval_output_noise="clean"),
    }
    if recipe_owned_prior(task):
        evidence["prior_policy"] = prior_policy_receipt(context.recipe, fixture.prior)
    if fixture.transport is not None:
        evidence["component_transport"] = fixture.transport.state_dict()
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
        "cost": {
            "optimizer_updates": counts,
            "completed_steps": fixture.completed_steps,
            "adapter_loop_seconds": time.monotonic() - started,
            "phase_timing": timing.snapshot(),
        },
        "scope": (
            "integration_demo_only"
            if str(task.get("id", "")).endswith("-software-prefix")
            else "ordinary_full_task"
        ),
        "declared_task_updates": steps,
        "execution_limit": steps,
    }
    atomic_json(output / "adapter-receipt.json", receipt)
    return receipt


def run(request: dict, task: dict, output_dir: Path, device: str) -> dict:
    is_prefix = str(task.get("id", "")).endswith("-software-prefix")
    context = bytegan_context(request, task, device, allow_software_prefix=is_prefix)
    if task["adapter"] == "bytegan":
        return _run_hypercube8(context, task, output_dir)
    return _run_bigan_dictionary(context, request, task, output_dir)
