"""Bind clean oracle-free v2 model-mode research toys to Forge planning and grading.

Separates the clean oracle-free v2 DDGAN / AE-DDGAN cohort from frozen v1
source-binding pins (`model_toy_adapter.py`) and honors each case's calibrated
`terminal_observations` (e.g. `1` per `ddgan_tail_query_calibration_rationale.md`)
rather than hardcoding `minimum_stable_checks == 5`.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import math
from pathlib import Path
import shutil

import torch

from .api import CapabilityError, task_formulation_context
from .contracts import atomic_json, file_hash, stable_hash
from .sampling import MODEL_TOY_CLEAN, executed_receipt
from .state import state_digest

V2_MODEL_PROVIDERS = ("api_model_ddgan_v2", "api_model_ae_ddgan_v2")


class ModelToyV2ExecutionError(RuntimeError):
    """A v2 model provider did not complete; preserve its original diagnostic receipt."""


def definition(task):
    from benchmarks.toy_audit.api_contract import validate_case

    host = task["execution"]["host_definition"]
    provider = host["provider"].removeprefix("benchmarks.toy_audit.")
    if provider not in V2_MODEL_PROVIDERS:
        raise ValueError("v2 model task requires a registered v2 model-toy provider")
    module = import_module(f"benchmarks.toy_audit.{provider}")
    matches = [case for case in module.list_cases() if case["id"] == host["case_id"]]
    if len(matches) != 1:
        raise ValueError("v2 model task must name exactly one executable case")
    return {**validate_case(deepcopy(matches[0])), "provider": provider}


def test_verdict(spec, result):
    """Grade calibrated model-toy observation curves honoring `minimum_stable_checks` (including 1)."""
    from benchmarks.locked_shared import baseline
    from benchmarks.locked_shared.observation import OBSERVATIONS, sustained, threshold_margin

    if result is None:
        return dict(status="MISSING", attempted=False, passed=False, confirmation_fraction=2.0, shortfall=2.0)
    if result.get("error"):
        return dict(status="ERROR", attempted=True, passed=False, confirmation_fraction=2.0, shortfall=2.0)
    count = int(spec.get("observations", OBSERVATIONS))
    min_stable = int(spec.get("minimum_stable_checks", 1))
    if type(count) is not int or count < 1 or min_stable < 1 or min_stable > count:
        raise ValueError("invalid calibrated model-toy observation/terminal check count")
    observations = result.get("observations", result.get("curve", []))
    steps = sorted({math.ceil(i * spec["steps"] / count) for i in range(1, count + 1)})
    requirements = spec["thresholds"]
    if min_stable >= 2:
        convergence = sustained(observations, requirements, expected_steps=steps, minimum=min_stable)
    else:
        for _, op, bound in requirements:
            threshold_margin(bound, op, bound)
        if any(b["step"] <= a["step"] for a, b in zip(observations, observations[1:])):
            raise ValueError("observations must have unique increasing steps")

        def _passes(point):
            for key, op, bound in requirements:
                value = point.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return False
                if not threshold_margin(value, op, bound) >= 0:
                    return False
            return True

        passing = [_passes(p) for p in observations]
        complete = [p["step"] for p in observations] == steps
        first = next((p for p, ok in zip(observations, passing) if ok), None)
        start = len(observations)
        while start and passing[start - 1]:
            start -= 1
        suffix = observations[start:]
        stable = complete and len(suffix) >= 1
        convergence = {
            "complete": complete,
            "observations": len(observations),
            "passing_observations": sum(passing),
            "minimum_stable_checks": 1,
            "passing_suffix": len(suffix),
            "first_pass_step": first["step"] if first else None,
            "stable_from_step": suffix[0]["step"] if stable else None,
            "stable_from_seconds": suffix[0].get("seconds") if stable else None,
            "confirmed_step": suffix[0]["step"] if stable else None,
            "confirmed_seconds": suffix[0].get("seconds") if stable else None,
        }
    cells = baseline.score_metrics(result.get("live", {}), requirements)
    passed = convergence["confirmed_step"] is not None and all(c["status"] == "PASS" for c in cells)
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    deficits = [
        2.0 if c["margin"] is None else min(2.0, max(0.0, -c["margin"]) / (abs(c["threshold"]) or 1.0))
        for c in cells
    ]
    return dict(
        status=status,
        attempted=True,
        passed=passed,
        metrics=cells,
        convergence=convergence,
        shortfall=sum(deficits) / len(deficits) if convergence["complete"] else 2.0,
        confirmation_fraction=convergence["confirmed_step"] / spec["steps"] if passed else 2.0,
    )


def validate_task(task, *, root=None):
    from benchmarks.toy_audit.api_model_ddgan import RESOURCES, MixtureTarget
    from .taskrecipes import ADAPTABLE_FIELDS

    if task.get("adapter") != "public_api_model_toy_v2":
        raise ValueError("v2 model adapter requires adapter == 'public_api_model_toy_v2'")
    case = definition(task)
    execution, evaluation = task["execution"], task["evaluation"]
    if execution.get("execution_path") != "public_components":
        raise ValueError("model toys require their explicit public-components path")
    if execution.get("initializer") != "deterministic_orthogonal":
        raise ValueError("model toys require the public deterministic initializer")
    if execution.get("steps") != case["default_steps"]:
        raise ValueError("model task update budget differs from its executable case")
    host = execution["host_definition"]
    resources = host["resources"]
    if resources.get("batch_size") != case["batch_size"] or set(resources) != {"num_particles", "z_dim", "batch_size"}:
        raise ValueError("model task resources differ from the executable host")
    target = MixtureTarget(case["target"])
    declared = dict(
        kind=case["target"],
        means=target.means.tolist(),
        variances=target.variances.tolist(),
        weights=target.weights.tolist(),
    )
    if stable_hash(host["target"]) != stable_hash(declared):
        raise ValueError("DDGAN v2 task target differs from the executable target law")
    if resources != RESOURCES:
        raise ValueError("model task prior capacity differs from the executable host")
    fields = execution.get("recipe_fields")
    if not isinstance(fields, dict) or not fields or set(fields) - (ADAPTABLE_FIELDS - {"prior_reg"}):
        raise ValueError("model task may adapt only declared objectives; prior policy belongs to the recipe")
    if evaluation.get("thresholds") != case["forge_thresholds"]:
        raise ValueError("model task bounds differ from its executable scorer")
    expected_terminal = int(case.get("terminal_observations", 1))
    min_stable = evaluation.get("minimum_stable_checks")
    if (
        evaluation.get("eval_samples") != case["eval_samples"]
        or evaluation.get("observations") != 24
        or type(min_stable) is not int
        or min_stable != expected_terminal
        or evaluation.get("kind") != "transfer_sustained"
        or evaluation.get("evaluator") != "experiments.forge.model_toy_v2_adapter:test_verdict"
    ):
        raise ValueError(
            f"model v2 task must retain full sample count, 24 observations, "
            f"minimum_stable_checks={expected_terminal}, and model_toy_v2_adapter:test_verdict"
        )
    if evaluation.get("scoring_weights") != "live":
        raise ValueError("model toys score their actual live weights")
    if root is not None:
        for path, expected in evaluation.get("sources", {}).items():
            if file_hash(Path(root) / path) != expected:
                raise ValueError(f"model source binding differs: {path}")
    return case


def preflight(task, candidate, *, root=None):
    try:
        case = validate_task(task, root=root)
        context = task_formulation_context(candidate, task, {"seed": 0}, device="cpu", root=root)
        module = import_module(f"benchmarks.toy_audit.{case['provider']}")
        if not callable(getattr(module, "task_recipe", None)):
            raise ValueError("model provider must validate its supported recipe before reservation")
        module.task_recipe(case, recipe=context.recipe)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return error.blockers if isinstance(error, CapabilityError) else [str(error)]
    return []


def _run_v2_case(case, output, *, device="cpu", seed=0, steps=None, eval_samples=None,
                 frames=9, wall_cap_seconds=None, recipe=None, audit_fixture=None):
    """Execute a clean v2 model fixture with full api_run receipt/GIF/purity contracts."""
    import json
    import platform
    import time

    import numpy as np

    from benchmarks.toy_audit import api_contract as contract
    from benchmarks.toy_audit import api_run
    from benchmarks.toy_audit.reproducibility import (
        VERSION as COMPARISON_VERSION,
        construction_rng,
        reproducible_execution,
    )

    @reproducible_execution
    def _execute(*, device):
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=False)
        run_steps = case["default_steps"] if steps is None else steps
        samples = case["eval_samples"] if eval_samples is None else eval_samples
        if type(samples) is not int or samples < 1:
            raise ValueError("evaluation sample count must be positive")
        if wall_cap_seconds is not None and (
            isinstance(wall_cap_seconds, bool)
            or not isinstance(wall_cap_seconds, (int, float))
            or not math.isfinite(wall_cap_seconds)
            or wall_cap_seconds <= 0
        ):
            raise ValueError("wall cap must be finite positive seconds")
        boundaries = contract.evaluation_steps(run_steps, frames)
        metric_count = contract.metric_observations(case)
        metric_steps = contract.evaluation_steps(run_steps, min(run_steps, metric_count) + 1)
        evaluation_steps = sorted(set(boundaries) | set(metric_steps))
        required_terminal = int(case.get("terminal_observations", 1))
        receipt = {
            "schema": "particlegan_api_toy_run_v1",
            "case": case,
            "checkpoint_capabilities": api_run.json_value(case.get("checkpoint_capabilities", {})),
            "recipe": None,
            "requested_recipe_overrides": {},
            "api_components": [],
            "source": api_run.source_identity(),
            "seed": seed,
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": str(device),
                "cuda": torch.version.cuda,
                "autograd_multithreading_enabled": torch.autograd.is_multithreading_enabled(),
                "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "protocol": {
                "updates": run_steps,
                "default_updates": case["default_steps"],
                "evaluation_samples": samples,
                "default_evaluation_samples": case["eval_samples"],
                "evaluation_steps": evaluation_steps,
                "metric_evaluation_steps": metric_steps,
                "media_steps": boundaries,
                "metric_observations": metric_count,
                "media_frames": frames,
                "terminal_observations": required_terminal,
                "wall_cap_seconds": wall_cap_seconds,
            },
            "timing": {
                "method": "monotonic elapsed execution; CUDA synchronized at observations",
                "start": "immediately before fixture construction",
                "includes": "fixture initialization, updates, source binding, sampling and evaluation",
                "excludes": "initial inventory/queue wait and post-run artifact/media export",
            },
            "historical_results_changed": False,
            "observations": [],
        }
        full = (
            run_steps >= case["default_steps"]
            and samples >= case["eval_samples"]
            and len(metric_steps) - 1 >= metric_count
        )
        records, arrays = [], {}
        started = time.monotonic()
        completed = 0
        fixture = None

        def capture(step):
            before = state_digest(fixture.state_dict())
            with api_run.isolated_evaluation():
                record = contract.validate_observation(fixture.observe(n=samples, seed=seed + 10000))
            if state_digest(fixture.state_dict()) != before:
                raise ValueError("v2 model toy evaluation changed training state or consumed a training RNG")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(torch.device(device))
            record.update(step=step, elapsed_seconds=time.monotonic() - started)
            for index, view in enumerate(record["views"]):
                for role in ("target", "samples"):
                    view[role] = contract.array(view[role]).copy()
                    arrays[f"step{step}_view{index}_{role}"] = view[role]
            records.append(record)
            compact = {key: value for key, value in record.items() if key != "views"}
            compact["views"] = [
                {key: value for key, value in view.items() if key not in {"target", "samples"}}
                for view in record["views"]
            ]
            receipt["observations"].append(compact)
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "step": step,
                        "metric_passed": record["passed"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "failed_bounds": record["failed_bounds"][:5],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        try:
            provider = import_module(f"benchmarks.toy_audit.{case['provider']}")
            with construction_rng(seed, device):
                fixture = provider.build_case(
                    case["id"],
                    device=device,
                    seed=seed,
                    recipe_name=case["default_recipe"],
                    max_steps=run_steps,
                    recipe=recipe,
                )
            if audit_fixture is not None:
                audit_fixture(fixture)
            receipt["reproducibility"] = {
                "comparison_version": COMPARISON_VERSION,
                "initial_state_sha256": state_digest(fixture.state_dict()),
                "conditions": "case-owned architecture, prior, sampling, batch size, update budget and scoring cadence",
                "historical_evidence": "original source, seed and initialization retained; no regrading",
                "initial_models": {
                    name: state_digest(module.state_dict())
                    for name in ("G", "D", "E", "prior")
                    if isinstance(module := getattr(fixture, name, None), torch.nn.Module)
                },
            }
            receipt["recipe"] = api_run.json_value(fixture.recipe.to_dict())
            receipt["api_components"] = list(fixture.api_components)
            if torch.device(device).type == "cuda":
                receipt["runtime"]["cuda_device_model"] = torch.cuda.get_device_name(torch.device(device))
            receipt["source"] = api_run.source_identity()
            for step in range(run_steps + 1):
                if wall_cap_seconds is not None and time.monotonic() - started >= wall_cap_seconds:
                    raise TimeoutError("declared wall cap reached; unchanged full protocol is incomplete")
                if step:
                    fixture.step()
                    completed = step
                if step not in evaluation_steps:
                    continue
                capture(step)
            if wall_cap_seconds is not None and time.monotonic() - started >= wall_cap_seconds:
                raise TimeoutError("declared wall cap reached; complete acquisition exceeded the frozen resource allowance")
            terminal = [
                record for record in records if record["step"] > 0 and record["step"] in metric_steps
            ][-required_terminal:]
            sustained_ok = len(terminal) == required_terminal and all(record["passed"] for record in terminal)
            failed = list(records[-1]["failed_bounds"])
            if not full:
                failed.append("default budget, evaluation draw count or metric cadence not completed")
            if not sustained_ok:
                failed.append(f"last {required_terminal} post-update metric observations do not all pass")
            receipt.update(
                status="COMPLETE",
                metric_passed=records[-1]["passed"],
                sustained_metric_passed=sustained_ok,
                default_protocol_complete=full,
                passed=full and sustained_ok,
                verdict="PASS" if full and sustained_ok else "FAIL",
                failed_bounds=failed,
                completed_updates=completed,
            )
        except Exception as error:
            status = (
                "INCOMPLETE"
                if isinstance(error, TimeoutError)
                else "BLOCKED"
                if isinstance(error, contract.UnsupportedRecipeOverrides)
                else "ERROR"
            )
            receipt.update(
                status=status,
                passed=False,
                verdict="FAIL",
                failed_bounds=[f"API execution or metric error: {type(error).__name__}: {error}"],
                completed_updates=completed,
                default_protocol_complete=False,
            )
            if (
                isinstance(error, TimeoutError)
                and fixture is not None
                and completed > 0
                and (not records or records[-1]["step"] != completed)
            ):
                try:
                    capture(completed)
                    receipt["partial_terminal_observation_added"] = True
                except Exception as capture_error:
                    receipt["failed_bounds"].append(f"partial goal observation unavailable: {capture_error}")
        receipt["elapsed_seconds"] = time.monotonic() - started
        if "reproducibility" in receipt and hasattr(fixture, "batch_sequence_sha256"):
            receipt["reproducibility"]["batch_sequence_sha256"] = fixture.batch_sequence_sha256
        receipt["artifacts"] = {}
        receipt["gif_frames"] = 0
        if records:
            try:
                np.savez_compressed(out_dir / "observations.npz", **arrays)
                receipt["artifacts"]["observations.npz"] = {
                    "sha256": file_hash(out_dir / "observations.npz"),
                    "bytes": (out_dir / "observations.npz").stat().st_size,
                }
                torch.save(fixture.state_dict(), out_dir / "final-state.pt")
                receipt["artifacts"]["final-state.pt"] = {
                    "sha256": file_hash(out_dir / "final-state.pt"),
                    "bytes": (out_dir / "final-state.pt").stat().st_size,
                    "checkpoint_capabilities": receipt["checkpoint_capabilities"],
                }
                gif = out_dir / "goal.gif"
                media_records = [record for record in records if record["step"] in boundaries]
                if receipt["status"] != "COMPLETE" and not any(
                    record["step"] == records[-1]["step"] for record in media_records
                ):
                    media_records.append(records[-1])
                api_run.render_gif(
                    case,
                    media_records,
                    gif,
                    full_budget=receipt.get("default_protocol_complete", False),
                    requested_steps=run_steps,
                    final_verdict=(
                        receipt["verdict"]
                        if receipt["status"] == "COMPLETE"
                        else f"{receipt['status']} / FAIL: {receipt['failed_bounds'][0]}"
                    ),
                )
                receipt["artifacts"]["goal.gif"] = {
                    "sha256": file_hash(gif),
                    "bytes": gif.stat().st_size,
                }
                receipt["gif_frames"] = len(media_records)
            except Exception as error:
                receipt.update(passed=False, verdict="FAIL", status="ERROR")
                receipt["failed_bounds"].append(f"goal media/state error: {type(error).__name__}: {error}")
        receipt["source_unchanged"] = all(
            (contract.ROOT / path).is_file() and file_hash(contract.ROOT / path) == expected
            for path, expected in receipt["source"]["files_sha256"].items()
        )
        if not receipt["source_unchanged"]:
            receipt.update(passed=False, verdict="FAIL", status="ERROR")
            receipt["failed_bounds"].append("source changed during execution")
        api_run.write_json(out_dir / "receipt.json", receipt)
        return receipt

    return _execute(device=device)


def run(request, task, output, device, *, execution_limit=None):
    from .artifacts import manifest_artifacts
    from .mechanisms import MechanismAudit, mechanism_blockers
    from .priors import prior_policy_receipt

    case = validate_task(task)
    if request.get("protocol", {}).get("seed", 0) != 0:
        raise CapabilityError(["model-toy comparisons use protocol seed 0"])
    context = task_formulation_context(request["candidate"], task, request.get("protocol"), device=device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in ("generator", "encoder", "prior", "discriminator")}
    observed = {"finite": True}
    handles = []
    optimizer_states = {}
    observation_checkpoints = []

    def bind(fixture):
        if fixture.recipe != context.recipe:
            raise CapabilityError(["model fixture changed the injected candidate recipe"])
        if not getattr(fixture, "oracle_free", False):
            raise CapabilityError(["v2 model fixture must run strictly oracle-free"])
        prior = fixture.prior
        if float(prior.sigma) != float(torch.tensor(task["execution"]["prior"]["sigma"], dtype=prior.sigma.dtype)):
            raise CapabilityError(["model fixture changed the task's absolute initial MoG width"])
        resources = task["execution"]["host_definition"]["resources"]
        if tuple(prior.z.shape) != (resources["num_particles"], resources["z_dim"]):
            raise CapabilityError(["model fixture changed the task's initial prior capacity"])
        owners = {
            role: getattr(fixture, name, None)
            for role, name in (("generator", "G"), ("encoder", "E"), ("prior", "prior"), ("discriminator", "D"))
        }
        observed["initial_models"] = {
            role: state_digest(module.state_dict())
            for role, module in owners.items()
            if module is not None
        }
        observed["fixture"] = fixture
        observed["audit"] = MechanismAudit(context.recipe, fixture.opt_d, [fixture.opt_g])
        # Enforce evaluation purity identically to v1 without modifying pinned api_contract.py.
        _raw_observe = fixture.observe

        def _checked_observe(*args, **kwargs):
            before = state_digest(fixture.state_dict())
            result = _raw_observe(*args, **kwargs)
            if state_digest(fixture.state_dict()) != before:
                raise ValueError("v2 model toy evaluation changed training state or consumed a training RNG")
            return result

        fixture.observe = _checked_observe
        if task["execution"].get("observation_checkpoints", False):
            from .observation_checkpoints import checkpoint_observer

            fixture.observe = checkpoint_observer(
                fixture, output / "observation-checkpoints", observation_checkpoints
            )

        def hook(roles):
            def after(optimizer, args, kwargs):
                clock = tuple(
                    (id(p), int(values["step"]))
                    for p, values in optimizer.state.items()
                    if "step" in values
                )
                if optimizer_states.get(id(optimizer)) == clock:
                    return
                optimizer_states[id(optimizer)] = clock
                for role in roles:
                    module = owners[role]
                    if module is not None and any(p.grad is not None for p in module.parameters()):
                        counts[role] += 1
                    if module is not None:
                        observed["finite"] &= all(bool(torch.isfinite(p).all()) for p in module.parameters())
                        observed["finite"] &= all(
                            bool(torch.isfinite(p.grad).all())
                            for p in module.parameters()
                            if p.grad is not None
                        )
                if "discriminator" in roles:
                    observed["audit"].observe_penalty(fixture.penalty.last_stats)

            return after

        handles.extend(
            (
                fixture.opt_g.register_step_post_hook(hook(("generator", "encoder", "prior"))),
                fixture.opt_d.register_step_post_hook(hook(("discriminator",))),
            )
        )

    try:
        raw = _run_v2_case(
            case,
            output / "api",
            device=device,
            seed=0,
            steps=execution_limit,
            recipe=context.recipe,
            audit_fixture=bind,
            wall_cap_seconds=task["resources"]["timeout_seconds"],
        )
    finally:
        for handle in handles:
            handle.remove()

    if raw.get("status") != "COMPLETE":
        diagnostics = "; ".join(str(value) for value in raw.get("failed_bounds", []))
        raise ModelToyV2ExecutionError(
            f"v2 model provider {raw.get('status', 'UNKNOWN')} after "
            f"{raw.get('completed_updates', 0)} updates: {diagnostics or 'no provider diagnostic'}; "
            f"original receipt: {output / 'api/receipt.json'}"
        )
    if "audit" not in observed or "fixture" not in observed:
        raise ModelToyV2ExecutionError(
            f"v2 model provider completed without the required public fixture audit; "
            f"original receipt: {output / 'api/receipt.json'}"
        )
    rows = [
        {
            "step": row["step"],
            **{name: int(value) if type(value) is bool else value for name, value in row["metrics"].items()},
        }
        for row in raw["observations"]
        if row["step"] in raw["protocol"]["metric_evaluation_steps"] and row["step"] > 0
    ]
    audit = observed["audit"].receipt()
    expected_mechanisms = task["evaluation"].get("guards", {}).get("mechanism_exercised", False)
    guards = {
        "all_finite": observed["finite"] and raw["status"] == "COMPLETE",
        "optimizer_updates": counts,
        "unintended_rng_deviations": int("evaluation changed" in " ".join(raw.get("failed_bounds", []))),
        "hooks_exercised": not mechanism_blockers(audit) if expected_mechanisms else True,
        "mechanism_audit": audit,
    }
    fixture = observed["fixture"]

    def finite_tensors(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tensors(item) for item in value)
        return True

    guards["all_finite"] &= finite_tensors(fixture.state_dict())
    evidence = {
        "observations": rows,
        "live": rows[-1] if rows else {},
        "guards": guards,
        "initial_models": observed["initial_models"],
        "prior_policy": prior_policy_receipt(context.recipe, fixture.prior),
        "oracle_free": True,
        "source": raw["source"],
        "api_run_status": raw["status"],
        "api_run_verdict": raw["verdict"],
        "api_receipt_sha256": file_hash(output / "api/receipt.json"),
        "batch_sequence_sha256": getattr(fixture, "batch_sequence_sha256", None),
        "observation_checkpoints": observation_checkpoints,
        "actual_training_gif": {
            "path": str((output / "api/goal.gif").resolve()),
            **raw.get("artifacts", {}).get("goal.gif", {}),
        },
        **executed_receipt(MODEL_TOY_CLEAN, eval_output_noise="clean"),
    }
    checkpoint = torch.load(output / "api/final-state.pt", weights_only=False, map_location="cpu")
    state_root = output / "provenance"
    state_root.mkdir(exist_ok=True)
    state_file = state_root / "provenance-state.pt"
    shutil.copyfile(output / "api/final-state.pt", state_file)
    evidence["provenance_checkpoint"] = {
        "purpose": "provenance_only",
        "prerequisite_eligible": False,
        "artifact_root": str(state_root.resolve()),
        "path": state_file.name,
        "artifact_manifest": manifest_artifacts(state_root),
        "sha256": file_hash(state_file),
        "state_sha256": state_digest(checkpoint),
        "completed_steps": raw["completed_updates"],
        "state_owner": "complete provider model/optimizer/policy/RNG checkpoint",
        "optimizer_updates_added": 0,
        "sampling_draws_added": 0,
    }
    receipt = {
        **context.receipt(),
        "api_components": raw["api_components"],
        "evidence": evidence,
        "cost": {
            "completed_steps": raw["completed_updates"],
            "optimizer_updates": counts,
            "adapter_loop_seconds": raw["elapsed_seconds"],
        },
        "scope": "ordinary_full_task" if raw["default_protocol_complete"] else "integration_demo_only",
        "declared_task_updates": task["execution"]["steps"],
        "execution_limit": raw["protocol"]["updates"],
    }
    atomic_json(output / "adapter-receipt.json", receipt)
    return receipt
