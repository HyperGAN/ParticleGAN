"""Forge adapter and fail-closed contract validation for Theory-3 causal isolation toys."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import torch

from particlegan import ParticlePrior

from .api import CapabilityError, task_formulation_context
from .artifacts import manifest_artifacts
from .contracts import atomic_json, file_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .priors import prior_policy_receipt, recipe_owned_prior
from .sampling import PARTICLES_AND_GRADIENT, executed_receipt
from .state import state_digest

T3_ISOLATION_KINDS = frozenset({
    "t3_iso_scale_mass_8x",
    "t3_iso_horizon_saddle_shift",
})


def _contract_payload(kind: str, spec: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "generator_class": "Identity",
        "generator_trainable_params": 0,
        "prior_code_path": "ParticlePrior",
        "sigma": 0.0,
        "num_particles": int(spec["resources"]["num_particles"]),
        "z_dim": int(spec["resources"]["z_dim"]),
    }


def validate_task(task: dict, *, root: Path | None = None, allow_software_prefix: bool = False) -> dict:
    from benchmarks.toy_audit.api_t3_isolation import (
        HOST_DEFINITIONS,
        THRESHOLDS,
        t3_isolation_oracle_controls,
    )

    if not isinstance(task, dict) or task.get("adapter") != "t3_isolation":
        raise ValueError("t3_isolation validator requires adapter='t3_isolation'")
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        raise TypeError("t3_isolation task requires execution and evaluation objects")
    kind = evaluation.get("kind")
    if not isinstance(kind, str) or kind not in T3_ISOLATION_KINDS:
        raise ValueError(f"unsupported Theory-3 isolation kind: {kind!r}")
    if execution.get("host") != kind:
        raise ValueError("t3_isolation execution.host must match evaluation.kind")
    if execution.get("execution_path") != "public_components":
        raise ValueError("t3_isolation requires execution.execution_path == 'public_components'")
    if execution.get("prior_applicability") != "not_sampled":
        raise ValueError("t3_isolation requires execution.prior_applicability == 'not_sampled'")

    expected_spec = deepcopy(HOST_DEFINITIONS[kind])
    actual_spec = execution.get("host_definition")
    if not isinstance(actual_spec, dict):
        raise TypeError("t3_isolation execution.host_definition must be an object")

    is_prefix = allow_software_prefix and str(task.get("id", "")).endswith("-software-prefix")
    steps = execution.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("t3_isolation execution.steps must be a positive integer")
    if is_prefix:
        expected_cmp = {**expected_spec, "steps": actual_spec.get("steps", expected_spec["steps"])}
        if actual_spec != expected_cmp:
            raise ValueError("t3_isolation host_definition differs from the frozen task specification")
    else:
        if actual_spec != expected_spec or steps != expected_spec["steps"]:
            raise ValueError("t3_isolation host_definition or step budget differs from the frozen specification")

    if not recipe_owned_prior(task):
        raise ValueError("t3_isolation tasks require prior_contract='recipe_owned_v1'")
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
            "t3_isolation tasks require recipe-owned zero-sigma particle_cloud prior with standardize=False and exception_reason"
        )

    if evaluation.get("thresholds") != THRESHOLDS[kind]:
        raise ValueError("t3_isolation evaluation.thresholds differ from the frozen scorer")
    min_stable = evaluation.get("minimum_stable_checks")
    if type(min_stable) is not int or min_stable != 5:
        raise ValueError("t3_isolation tasks require minimum_stable_checks == 5")
    obs_count = evaluation.get("observations", 24)
    if type(obs_count) is not int or obs_count < 1 or (not is_prefix and obs_count != 24):
        raise ValueError("t3_isolation tasks require 24 scheduled observations")
    if evaluation.get("scoring_weights") != "live":
        raise ValueError("t3_isolation tasks score live unaveraged weights")

    if root is not None:
        for rel_path, expected_sha in evaluation.get("sources", {}).items():
            if file_hash(Path(root) / rel_path) != expected_sha:
                raise ValueError(f"t3_isolation source binding differs: {rel_path}")

    for name, control in t3_isolation_oracle_controls().items():
        if control["passed"] is not control["expected_pass"]:
            raise ValueError(f"t3_isolation scorer oracle control failed: {name}")
    return expected_spec


def t3_isolation_context(
    request: dict,
    task: dict,
    device: str,
    *,
    root: Path | None = None,
    allow_software_prefix: bool = False,
):
    validate_task(task, root=root, allow_software_prefix=allow_software_prefix)
    cand = deepcopy(request["candidate"])
    cand.pop("prior_mechanism", None)
    context = task_formulation_context(cand, task, request.get("protocol"), device=device, root=root)
    if context.initializer != "deterministic_orthogonal":
        raise CapabilityError(["t3_isolation requires named deterministic_orthogonal initialization"])
    if context.recipe.prior_update != "learned":
        raise CapabilityError(["t3_isolation requires Recipe.prior_update == 'learned'"])
    if context.recipe.prior_code_law != "table":
        raise CapabilityError(["t3_isolation requires Recipe.prior_code_law == 'table'"])
    if context.prior_config.get("kind") != "particle_cloud" or float(context.prior_config.get("sigma", -1.0)) != 0.0:
        raise CapabilityError(["t3_isolation requires an effective zero-sigma ParticlePrior"])
    return context


def preflight(task: dict, candidate: dict, *, root: Path | None = None) -> list[str]:
    try:
        t3_isolation_context(
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
    observed = evidence.get("t3_isolation_contract")
    if not isinstance(observed, dict):
        return "INCOMPLETE", "missing observed Theory-3 isolation execution contract"
    if observed != expected_contract:
        return "INVALID", "observed Theory-3 isolation contract differs from the frozen task"

    gif = evidence.get("actual_training_gif")
    if not isinstance(gif, dict) or not isinstance(gif.get("sha256"), str) or not gif["sha256"]:
        return "INCOMPLETE", "missing actual-training GIF"

    steps = int(task["execution"]["steps"])
    count = int(task["evaluation"].get("observations", 24))
    expected_steps = sorted({math.ceil(i * steps / count) for i in range(1, count + 1)})
    points = evidence.get("observations")
    if not isinstance(points, list) or [pt.get("step") for pt in points if isinstance(pt, dict)] != expected_steps:
        return "INCOMPLETE", f"all {len(expected_steps)} declared Theory-3 isolation checkpoints are required"
    live = evidence.get("live")
    if not isinstance(live, dict) or live.get("step") != steps:
        return "INCOMPLETE", "final live metrics at the complete update budget are required"
    guards = evidence.get("guards", {})
    upd = guards.get("optimizer_updates", {}) if isinstance(guards, dict) else {}
    if upd.get("generator") != 0 or upd.get("prior") != steps or upd.get("discriminator") != steps:
        return "INVALID", "Theory-3 isolation requires 0 generator updates and exact prior/discriminator update counts"
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
    from benchmarks.toy_audit.api_t3_isolation import (
        T3IsolationFixture,
        build_models_for_spec,
        sample_target_for_kind,
    )

    from .adapters import _checkpoints, _event

    is_prefix = str(task.get("id", "")).endswith("-software-prefix")
    if request.get("protocol", {}).get("seed", 0) != 0:
        raise CapabilityError(["t3_isolation comparisons use protocol seed 0"])
    context = t3_isolation_context(request, task, device, allow_software_prefix=is_prefix)
    kind = task["evaluation"]["kind"]
    spec = task["execution"]["host_definition"]
    steps = int(task["execution"]["steps"])
    res = spec["resources"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def _make_d():
        _, d_mod = build_models_for_spec(spec)
        return d_mod

    def _make_prior():
        return ParticlePrior(
            num_particles=int(res["num_particles"]),
            z_dim=int(res["z_dim"]),
            learnable=True,
        )

    d_mod = context.construct(_make_d, component="discriminator").to(context.device)
    p_mod = context.construct(_make_prior, component="prior").to(context.device)

    streams = {
        "latent_generator": context.streams.generator("prior", component="latent", purpose="indices"),
        "penalty_generator": context.streams.generator("noise", component="penalty", purpose="training"),
        "noise_generator": context.streams.generator("noise", component="generator", purpose="output"),
        "input_noise_generator": context.streams.generator("noise", component="critic", purpose="input"),
        "eval_generator": context.streams.generator("eval", component="sampler", purpose="samples"),
        "data_d_generator": context.streams.generator("data", component="target", purpose="critic", device="cpu"),
        "data_g_generator": context.streams.generator("data", component="target", purpose="generator", device="cpu"),
    }
    fixture = T3IsolationFixture(
        kind,
        context.recipe,
        device=context.device,
        seed=context.streams.seed,
        discriminator=d_mod,
        prior=p_mod,
        streams=streams,
        max_steps=steps,
    )
    initial_models = {
        "generator": state_digest(dict(fixture.G.named_parameters())),
        "discriminator": state_digest(dict(fixture.D.named_parameters())),
        "prior": state_digest(dict(fixture.prior.named_parameters())),
    }

    eval_samples = int(task["evaluation"].get("eval_samples", 2048))
    ref_target = sample_target_for_kind(
        kind,
        eval_samples,
        context.streams.generator("eval", component="target", purpose="reference", device="cpu"),
        step=steps,
        device="cpu",
    )

    audit = MechanismAudit(context.recipe, fixture.opt_d, [fixture.opt_g])
    counts = {"generator": 0, "prior": 0, "discriminator": 0}
    optimizer_states: dict[int, tuple] = {}
    all_finite = True

    def _make_post_hook(roles: tuple[str, ...]):
        owners = {"generator": fixture.G, "prior": fixture.prior, "discriminator": fixture.D}

        def _after(optimizer, _args, _kwargs):
            nonlocal all_finite
            clock = tuple((id(p), int(vals["step"])) for p, vals in optimizer.state.items() if "step" in vals)
            if optimizer_states.get(id(optimizer)) == clock:
                return
            optimizer_states[id(optimizer)] = clock
            for role in roles:
                mod = owners[role]
                if any(p.grad is not None for p in mod.parameters()):
                    counts[role] += 1
                all_finite &= all(bool(torch.isfinite(p).all()) for p in mod.parameters())
                all_finite &= all(bool(torch.isfinite(p.grad).all()) for p in mod.parameters() if p.grad is not None)

        return _after

    handles = [
        fixture.opt_g.register_step_post_hook(_make_post_hook(("generator", "prior"))),
        fixture.opt_d.register_step_post_hook(_make_post_hook(("discriminator",))),
    ]

    checkpoints = set(_checkpoints(task))
    observations: list[dict] = []
    frames: list[dict] = []
    batch_digest = hashlib.sha256()
    eval_rng_deviations = 0

    t_start = time.perf_counter()
    train_seconds = 0.0
    eval_seconds = 0.0
    try:
        for step in range(1, steps + 1):
            t0 = time.perf_counter()
            real_d_cpu, real_g_cpu = fixture.step()
            audit.observe_penalty(fixture.penalty.last_stats)
            batch_digest.update(real_d_cpu.numpy().tobytes())
            batch_digest.update(real_g_cpu.numpy().tobytes())
            train_seconds += time.perf_counter() - t0

            if step in checkpoints:
                e0 = time.perf_counter()
                rng_before = (
                    fixture.data_d_rng.get_state().clone(),
                    fixture.data_g_rng.get_state().clone(),
                    fixture.policy.latent_generator.get_state().clone(),
                    fixture.policy.penalty_generator.get_state().clone(),
                )
                scored = fixture.observe(ref_target)
                rng_after = (
                    fixture.data_d_rng.get_state(),
                    fixture.data_g_rng.get_state(),
                    fixture.policy.latent_generator.get_state(),
                    fixture.policy.penalty_generator.get_state(),
                )
                if any(not torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
                    eval_rng_deviations += 1
                row = {
                    "step": step,
                    **{k: int(v) if type(v) is bool else v for k, v in scored["metrics"].items()},
                }
                observations.append(row)
                frames.append(
                    {
                        "step": step,
                        "passed": scored["passed"],
                        "failed_bounds": list(scored["failed_bounds"]),
                        "metrics": dict(scored["metrics"]),
                        "views": scored["views"],
                    }
                )
                eval_seconds += time.perf_counter() - e0
                _event("observation", task=task["id"], step=step, metrics=row)
    finally:
        for h in handles:
            h.remove()

    gif_receipt = _render_training_gif(task, frames, output)
    state_root = output / "provenance"
    state_root.mkdir(exist_ok=True)
    state_file = state_root / "provenance-state.pt"
    final_state = fixture.state_dict()
    torch.save(final_state, state_file)

    def _finite_tree(val) -> bool:
        if isinstance(val, torch.Tensor):
            return bool(torch.isfinite(val).all())
        if isinstance(val, dict):
            return all(_finite_tree(v) for v in val.values())
        if isinstance(val, (list, tuple)):
            return all(_finite_tree(v) for v in val)
        return True

    all_finite &= _finite_tree(final_state)
    audit_receipt = audit.receipt()
    guards = {
        "all_finite": bool(all_finite),
        "optimizer_updates": counts,
        "unintended_rng_deviations": int(eval_rng_deviations),
        "hooks_exercised": not mechanism_blockers(audit_receipt),
        "mechanism_audit": audit_receipt,
    }

    observed_contract = {
        "schema_version": 1,
        "kind": kind,
        "generator_class": type(fixture.G).__name__,
        "generator_trainable_params": sum(p.numel() for p in fixture.G.parameters() if p.requires_grad),
        "prior_code_path": type(fixture.prior).__name__,
        "sigma": 0.0,
        "num_particles": int(fixture.prior.z.shape[0]),
        "z_dim": int(fixture.prior.z.shape[1]),
    }

    evidence = {
        "observations": observations,
        "live": observations[-1] if observations else {},
        "guards": guards,
        "initial_models": initial_models,
        "prior_policy": prior_policy_receipt(context.recipe),
        "mechanism_audit": audit_receipt,
        "batch_sequence_sha256": batch_digest.hexdigest(),
        "t3_isolation_contract": observed_contract,
        "actual_training_gif": gif_receipt,
        "provenance_checkpoint": {
            "purpose": "provenance_only",
            "prerequisite_eligible": False,
            "artifact_root": str(state_root.resolve()),
            "path": state_file.name,
            "artifact_manifest": manifest_artifacts(state_root),
            "sha256": file_hash(state_file),
            "state_sha256": state_digest(final_state),
            "completed_steps": fixture.completed_steps,
            "state_owner": "complete T3IsolationFixture model/optimizer/policy/RNG checkpoint",
            "optimizer_updates_added": 0,
            "sampling_draws_added": 0,
        },
        **executed_receipt(PARTICLES_AND_GRADIENT, eval_output_noise="not_applied_to_measurement"),
    }
    elapsed = time.perf_counter() - t_start
    receipt = {
        **context.receipt(),
        "evidence": evidence,
        "cost": {
            "completed_steps": fixture.completed_steps,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "adapter_loop_seconds": elapsed,
        },
        "scope": "ordinary_full_task" if not is_prefix else "integration_demo_only",
        "declared_task_updates": steps,
        "execution_limit": steps,
    }
    atomic_json(output / "adapter-receipt.json", receipt)
    return receipt
