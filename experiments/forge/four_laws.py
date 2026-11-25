"""Forge adapter and fail-closed contract validation for Four Laws Adherence dynamical toys."""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import math
from pathlib import Path
import time

import torch

from particlegan import unwrap_prior

from .api import CapabilityError, task_formulation_context
from .artifacts import save_provenance_checkpoint
from .contracts import atomic_json, file_hash
from .priors import prior_noise_sigma, prior_policy_receipt, recipe_owned_prior
from .sampling import ENUMERATED_PRIOR_CLEAN, PUBLIC_PRIOR_CLEAN, executed_receipt
from .state import state_digest
from .telemetry import PhaseTimer

FOUR_LAWS_123_KINDS = frozenset({
    "fl_law1_aniso_unequal_modes",
    "fl_law1_noisy_atomic_codebook",
    "fl_law2_pure_adversarial_codim1",
    "fl_law2_groupnorm_and_vit_sprites",
    "fl_law3_highd_orphan_d32",
})

FOUR_LAWS_4_KINDS = frozenset({
    "fl_law41_trust_rare1pct_tanh",
    "fl_law42_swap_deep_critic_16l",
    "fl_law42_deep_resbn_and_conv1x1",
    "fl_law44_highd_sparse_prior_and_shock",
})

FOUR_LAWS_KINDS = FOUR_LAWS_123_KINDS | FOUR_LAWS_4_KINDS

_ALLOWED_SAMPLING_LAWS = frozenset({
    PUBLIC_PRIOR_CLEAN,
    ENUMERATED_PRIOR_CLEAN,
})

_VERIFIED_ORACLES: dict[tuple[str, str], bool] = {}


def provider_module_name(kind: str) -> str:
    if kind in FOUR_LAWS_123_KINDS:
        return "benchmarks.toy_audit.api_four_laws_123"
    if kind in FOUR_LAWS_4_KINDS:
        return "benchmarks.toy_audit.api_four_laws_4"
    raise ValueError(f"unsupported four_laws task kind: {kind!r}")


def _provider_module(kind: str):
    return import_module(provider_module_name(kind))


def _spec_and_thresholds(kind: str) -> tuple[dict, list[list]]:
    mod = _provider_module(kind)
    if kind not in mod.SPECS or kind not in mod.THRESHOLDS:
        raise ValueError(f"four_laws provider missing spec or thresholds for {kind!r}")
    return deepcopy(mod.SPECS[kind]), deepcopy(mod.THRESHOLDS[kind])


def expected_sampling_law(task: dict) -> str:
    kind = task.get("evaluation", {}).get("kind")
    if not isinstance(kind, str) or kind not in FOUR_LAWS_KINDS:
        raise ValueError(f"unsupported four_laws evaluation kind: {kind!r}")
    spec, _ = _spec_and_thresholds(kind)
    law = spec.get("sampling_law")
    if law not in _ALLOWED_SAMPLING_LAWS:
        raise ValueError(f"four_laws task {kind!r} has unsupported sampling_law: {law!r}")
    return law


def _verify_oracle_controls(kind: str) -> None:
    mod = _provider_module(kind)
    mod_file = getattr(mod, "__file__", None)
    digest = file_hash(Path(mod_file)) if mod_file and Path(mod_file).is_file() else "memory"
    cache_key = (mod.__name__, digest)
    if _VERIFIED_ORACLES.get(cache_key):
        return
    controls = mod.oracle_controls()
    if not isinstance(controls, dict) or not controls:
        raise ValueError(f"four_laws provider {mod.__name__} returned empty oracle_controls")
    for name, control in controls.items():
        if not isinstance(control, dict) or control.get("passed") is not control.get("expected_pass"):
            raise ValueError(f"four_laws scorer oracle control failed: {name}")
    _VERIFIED_ORACLES[cache_key] = True


def _contract_payload(kind: str, spec: dict) -> dict:
    res = spec["resources"]
    prior = spec["prior"]
    return {
        "schema_version": 1,
        "kind": kind,
        "law": str(spec["law"]),
        "prior_kind": str(prior["kind"]),
        "prior_sigma": float(prior["sigma"]),
        "num_particles": int(res["num_particles"]),
        "z_dim": int(res["z_dim"]),
        "batch_size": int(res["batch_size"]),
        "sampling_law": str(spec["sampling_law"]),
    }


def validate_task(task: dict, *, root: Path | None = None, allow_software_prefix: bool = False) -> dict:
    if not isinstance(task, dict) or task.get("adapter") != "four_laws":
        raise ValueError("four_laws validator requires adapter == 'four_laws'")
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        raise TypeError("four_laws task requires execution and evaluation objects")

    kind = evaluation.get("kind")
    if not isinstance(kind, str) or kind not in FOUR_LAWS_KINDS:
        raise ValueError(f"unsupported four_laws evaluation.kind: {kind!r}")
    if execution.get("host") != kind:
        raise ValueError("four_laws execution.host must match evaluation.kind")

    expected_spec, expected_thresholds = _spec_and_thresholds(kind)
    actual_spec = execution.get("host_definition")
    if not isinstance(actual_spec, dict):
        raise TypeError("four_laws execution.host_definition must be an object")

    is_prefix = allow_software_prefix and str(task.get("id", "")).endswith("-software-prefix")
    steps = execution.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("four_laws execution.steps must be a positive integer")
    if is_prefix:
        expected_cmp = {**expected_spec, "steps": actual_spec.get("steps", expected_spec["steps"])}
        if actual_spec != expected_cmp:
            raise ValueError("four_laws host_definition differs from the frozen task specification")
    else:
        if actual_spec != expected_spec or steps != int(expected_spec["steps"]):
            raise ValueError("four_laws host_definition or step budget differs from the frozen specification")

    if not recipe_owned_prior(task):
        raise ValueError("four_laws tasks require prior_contract='recipe_owned_v1'")
    expected_prior = expected_spec["prior"]
    prior = execution.get("prior")
    if (
        not isinstance(prior, dict)
        or prior.get("kind") != expected_prior["kind"]
        or float(prior.get("sigma", -1.0)) != float(expected_prior["sigma"])
        or prior.get("standardize") is not expected_prior["standardize"]
        or "learnable" in prior
    ):
        raise ValueError(f"four_laws task {kind} prior differs from its frozen specification")
    if prior["kind"] == "particle_cloud":
        if (
            float(prior["sigma"]) != 0.0
            or prior["standardize"] is not False
            or not isinstance(prior.get("exception_reason"), str)
            or not prior["exception_reason"].strip()
        ):
            raise ValueError(
                "four_laws particle_cloud tasks require sigma=0.0, standardize=False, and exception_reason"
            )

    if evaluation.get("thresholds") != expected_thresholds:
        raise ValueError("four_laws evaluation.thresholds differ from the frozen scorer")
    expected_min_stable = int(expected_spec.get("minimum_stable_checks", 5))
    min_stable = evaluation.get("minimum_stable_checks")
    if type(min_stable) is not int or min_stable != expected_min_stable:
        raise ValueError(f"four_laws task {kind} requires minimum_stable_checks == {expected_min_stable}")
    expected_obs = int(expected_spec.get("observations", 24))
    obs_count = evaluation.get("observations", expected_obs)
    if type(obs_count) is not int or obs_count < 1 or (not is_prefix and obs_count != expected_obs):
        raise ValueError(f"four_laws task {kind} requires {expected_obs} scheduled observations")
    expected_eval_samples = int(expected_spec["eval_samples"])
    eval_samples = evaluation.get("eval_samples", expected_eval_samples)
    if type(eval_samples) is not int or eval_samples < 1 or (not is_prefix and eval_samples != expected_eval_samples):
        raise ValueError(f"four_laws task {kind} requires eval_samples == {expected_eval_samples}")
    if evaluation.get("scoring_weights", "live") != "live":
        raise ValueError("four_laws tasks score live unaveraged weights")
    if evaluation.get("sampling_law") != expected_spec["sampling_law"]:
        raise ValueError("four_laws evaluation.sampling_law differs from frozen specification")

    if root is not None:
        for rel_path, expected_sha in evaluation.get("sources", {}).items():
            if file_hash(Path(root) / rel_path) != expected_sha:
                raise ValueError(f"four_laws source binding differs: {rel_path}")

    _verify_oracle_controls(kind)
    return expected_spec


def four_laws_context(
    request: dict,
    task: dict,
    device: str,
    *,
    root: Path | None = None,
    allow_software_prefix: bool = False,
):
    spec = validate_task(task, root=root, allow_software_prefix=allow_software_prefix)
    context = task_formulation_context(
        request["candidate"], task, request.get("protocol"), device=device, root=root
    )
    if context.initializer != "deterministic_orthogonal":
        raise CapabilityError(["four_laws requires named deterministic_orthogonal initialization"])
    if context.recipe.prior_code_law != "table":
        raise CapabilityError(["four_laws requires Recipe.prior_code_law == 'table'"])
    expected_prior = spec["prior"]
    if context.prior_config.get("kind") != expected_prior["kind"]:
        raise CapabilityError([f"four_laws task {spec['kind']} requires prior kind {expected_prior['kind']!r}"])
    if expected_prior["kind"] == "particle_cloud":
        if float(context.prior_config.get("sigma", -1.0)) != 0.0 or prior_noise_sigma(context.prior_config) != 0.0:
            raise CapabilityError([f"four_laws task {spec['kind']} requires an effective zero-sigma ParticlePrior"])
    return context


def preflight(task: dict, candidate: dict, *, root: Path | None = None) -> list[str]:
    try:
        four_laws_context(
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
    observed = evidence.get("four_laws_contract")
    if not isinstance(observed, dict):
        return "INCOMPLETE", "missing observed four_laws execution contract"
    if observed != expected_contract:
        return "INVALID", "observed four_laws contract differs from the frozen task"

    gif = evidence.get("actual_training_gif")
    if not isinstance(gif, dict) or not isinstance(gif.get("sha256"), str) or not gif["sha256"]:
        return "INCOMPLETE", "missing actual-training GIF"

    steps = int(task["execution"]["steps"])
    count = int(task["evaluation"].get("observations", spec.get("observations", 24)))
    expected_steps = sorted({math.ceil(i * steps / count) for i in range(1, count + 1)})
    points = evidence.get("observations")
    if not isinstance(points, list) or [pt.get("step") for pt in points if isinstance(pt, dict)] != expected_steps:
        return "INCOMPLETE", f"all {len(expected_steps)} declared four_laws checkpoints are required"
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
            "goal": task.get("description") or task["execution"]["host_definition"]["goal"],
            "default_steps": steps,
        },
        frames,
        path,
        full_budget=(len(frames) == obs_target),
        requested_steps=steps,
        final_verdict="PASS" if passed else "FAIL",
    )
    return {"path": str(path.resolve()), "sha256": file_hash(path), "frames": len(frames)}


def _count_updates(optimizer, parameters) -> int:
    values = [
        int(optimizer.state[p]["step"])
        for p in parameters
        if p in optimizer.state and "step" in optimizer.state[p]
    ]
    return min(values) if values else 0


def run(request: dict, task: dict, output_dir: Path, device: str) -> dict:
    from benchmarks.toy_audit import api_contract
    from .adapters import _checkpoints, _event, _finite_tree, _host_receipt
    from .mechanisms import MechanismAudit, mechanism_blockers
    from .optimizer_diagnostics import attach

    is_prefix = str(task.get("id", "")).endswith("-software-prefix")
    context = four_laws_context(request, task, device, allow_software_prefix=is_prefix)
    kind = task["evaluation"]["kind"]
    spec = task["execution"]["host_definition"]
    steps = int(task["execution"]["steps"])
    eval_samples = int(task["evaluation"].get("eval_samples", spec["eval_samples"]))
    seed = int((request.get("protocol") or {}).get("seed", 0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mod = _provider_module(kind)
    fixture = mod.build_case(
        kind,
        device=context.device,
        seed=seed,
        max_steps=steps,
        components=context,
    )
    trainer = fixture.trainer
    _orig_step = trainer.step

    def _audited_step(real, *, generator_real=None, collect_stats=False):
        del collect_stats
        return _orig_step(real, generator_real=generator_real, collect_stats=True)

    trainer.step = _audited_step

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

    for step in range(1, steps + 1):
        with timing.measure("training_updates"):
            update = fixture.step()
        finite = finite and _finite_tree(update)
        p_stats = (
            update.get("penalty_stats")
            if isinstance(update, dict) and isinstance(update.get("penalty_stats"), dict)
            else getattr(getattr(trainer, "penalty", None), "last_stats", None)
        )
        audit.observe_penalty(p_stats or {})
        if not finite:
            raise FloatingPointError(f"nonfinite four_laws public-API update on {kind}")
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
            "min_margin": raw_observed.get("min_margin", metrics.get("min_margin")),
        })
        frames.append({
            "step": step,
            "passed": observed["passed"],
            "failed_bounds": list(observed["failed_bounds"]),
            "metrics": dict(metrics),
            "views": observed["views"],
        })
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

    sampling_law = expected_sampling_law(task)
    evidence = {
        "observations": observations,
        "live": observations[-1],
        "final_margins": observation_margins[-1]["margins"],
        "observation_margins": observation_margins,
        "host": host,
        "initial_parameters": initial_params,
        "batch_sequence_sha256": fixture.batch_sequence_sha256,
        "four_laws_contract": _contract_payload(kind, spec),
        "actual_training_gif": gif_receipt,
        "scoring_weights": "live",
        "guards": guards,
        "rng_audits": rng_audits,
        **executed_receipt(sampling_law, eval_output_noise="clean"),
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
        "cost": {
            "optimizer_updates": counts,
            "completed_steps": fixture.completed_steps,
            "adapter_loop_seconds": time.monotonic() - started,
            "phase_timing": timing.snapshot(),
        },
        "scope": "integration_demo_only" if is_prefix else "ordinary_full_task",
        "declared_task_updates": steps,
        "execution_limit": steps,
    }
    atomic_json(output / "adapter-receipt.json", receipt)
    return receipt
