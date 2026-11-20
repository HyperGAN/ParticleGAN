"""Bind model-mode research toys to ordinary Forge planning and grading.

Providers own their public-component updates and mathematical scorers. The
existing API runner owns execution, observations, checkpoints and actual GIFs;
this adapter adds recipe ownership and measured optimizer/evaluation guards.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from pathlib import Path

import torch

from .contracts import atomic_json, file_hash, stable_hash
from .api import CapabilityError, task_formulation_context
from .sampling import MODEL_TOY_CLEAN, executed_receipt
from .state import state_digest


class ModelToyExecutionError(RuntimeError):
    """A provider did not complete; preserve its original diagnostic receipt."""


def definition(task):
    from benchmarks.toy_audit.api_contract import MODEL_PROVIDERS, validate_case
    host = task["execution"]["host_definition"]
    provider = host["provider"].removeprefix("benchmarks.toy_audit.")
    if provider not in MODEL_PROVIDERS:
        raise ValueError("model task requires a registered model-toy provider")
    module = import_module(f"benchmarks.toy_audit.{provider}")
    matches = [case for case in module.list_cases() if case["id"] == host["case_id"]]
    if len(matches) != 1:
        raise ValueError("model task must name exactly one executable case")
    return {**validate_case(deepcopy(matches[0])), "provider": provider}


def validate_task(task, *, root=None):
    from .taskrecipes import ADAPTABLE_FIELDS
    case = definition(task)
    execution, evaluation = task["execution"], task["evaluation"]
    if execution.get("execution_path") != "public_components":
        raise ValueError("model toys require their explicit public-components path")
    if execution.get("initializer") != "deterministic_orthogonal":
        raise ValueError("model toys require the public deterministic initializer")
    if execution.get("steps") != case["default_steps"]:
        raise ValueError("model task update budget differs from its executable case")
    resources = execution["host_definition"]["resources"]
    if resources.get("batch_size") != case["batch_size"] or set(resources) != {"num_particles", "z_dim", "batch_size"}:
        raise ValueError("model task resources differ from the executable host")
    host = execution["host_definition"]
    module = import_module(f"benchmarks.toy_audit.{case['provider']}")
    if case["provider"] == "api_model_ae":
        expected_resources = dict(num_particles=64, z_dim=2, batch_size=128)
        target = dict(kind="gaussian_mixture", means=module.CENTERS, sigma=module.TARGET_SIGMA, masses=[.25] * 4)
        if (stable_hash(host["target"]) != stable_hash(target)
                or any(host.get(key) != value for key, value in dict(hidden=64, layers=2, fourier=2, init_std=.5).items())):
            raise ValueError("AE task target or network declaration differs from the executable host")
    elif case["provider"] == "api_model_vae":
        expected_resources = dict(num_particles=module.PARTICLES, z_dim=module.Z_DIM, batch_size=module.BATCH)
        if any(stable_hash(host[key]) != stable_hash(case[key]) for key in ("target", "architecture")):
            raise ValueError("VAE task target or network declaration differs from the executable host")
    elif case["provider"] == "api_model_misgan":
        expected_resources = module.RESOURCES
        if host != case["host_definition"]:
            raise ValueError("MisGAN task host differs from the executable definition")
    else:
        from benchmarks.toy_audit.api_model_ddgan import RESOURCES, MixtureTarget
        expected_resources = RESOURCES
        target = MixtureTarget(case["target"])
        declared = dict(kind=case["target"], means=target.means.tolist(),
                        variances=target.variances.tolist(), weights=target.weights.tolist())
        if stable_hash(host["target"]) != stable_hash(declared):
            raise ValueError("DDGAN task target differs from the executable target law")
    if resources != expected_resources:
        raise ValueError("model task prior capacity differs from the executable host")
    fields = execution.get("recipe_fields")
    if not isinstance(fields, dict) or not fields or set(fields) - (ADAPTABLE_FIELDS - {"prior_reg"}):
        raise ValueError("model task may adapt only declared objectives; prior policy belongs to the recipe")
    if evaluation.get("thresholds") != case["forge_thresholds"]:
        raise ValueError("model task bounds differ from its executable scorer")
    if (evaluation.get("eval_samples") != case["eval_samples"]
            or evaluation.get("observations") != 24
            or evaluation.get("minimum_stable_checks") != 5
            or evaluation.get("kind") != "transfer_sustained"):
        raise ValueError("model task must retain its full sample count and 24/five sustained protocol")
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


def run(request, task, output, device, *, execution_limit=None):
    from benchmarks.toy_audit.api_run import run_case
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
        prior = fixture.prior
        if float(prior.sigma) != float(torch.tensor(task["execution"]["prior"]["sigma"], dtype=prior.sigma.dtype)):
            raise CapabilityError(["model fixture changed the task's absolute initial MoG width"])
        resources = task["execution"]["host_definition"]["resources"]
        if tuple(prior.z.shape) != (resources["num_particles"], resources["z_dim"]):
            raise CapabilityError(["model fixture changed the task's initial prior capacity"])
        owners = {role: getattr(fixture, name, None) for role, name in
                  (("generator", "G"), ("encoder", "E"), ("prior", "prior"), ("discriminator", "D"))}
        observed["initial_models"] = {role: state_digest(module.state_dict())
                                      for role, module in owners.items() if module is not None}
        observed["fixture"] = fixture
        observed["audit"] = MechanismAudit(context.recipe, fixture.opt_d, [fixture.opt_g])
        if task["execution"].get("observation_checkpoints", False):
            from .observation_checkpoints import checkpoint_observer
            fixture.observe = checkpoint_observer(fixture, output / "observation-checkpoints",
                                                  observation_checkpoints)

        def hook(roles):
            def after(optimizer, args, kwargs):
                # PyTorch can invoke inherited post-hooks at both the inner
                # optimizer and its wrapper. Count changed optimizer clocks,
                # so those notifications do not invent a second update.
                clock = tuple((id(p), int(values["step"])) for p, values in optimizer.state.items()
                              if "step" in values)
                if optimizer_states.get(id(optimizer)) == clock:
                    return
                optimizer_states[id(optimizer)] = clock
                for role in roles:
                    module = owners[role]
                    if module is not None and any(p.grad is not None for p in module.parameters()):
                        counts[role] += 1
                    if module is not None:
                        observed["finite"] &= all(bool(torch.isfinite(p).all()) for p in module.parameters())
                        observed["finite"] &= all(bool(torch.isfinite(p.grad).all())
                                                  for p in module.parameters() if p.grad is not None)
                if "discriminator" in roles:
                    observed["audit"].observe_penalty(fixture.penalty.last_stats)
            return after
        handles.extend((fixture.opt_g.register_step_post_hook(hook(("generator", "encoder", "prior"))),
                        fixture.opt_d.register_step_post_hook(hook(("discriminator",)))))

    try:
        raw = run_case(case, output / "api", device=device, seed=0,
                       steps=execution_limit, recipe=context.recipe, audit_fixture=bind,
                       wall_cap_seconds=task["resources"]["timeout_seconds"])
    finally:
        for handle in handles:
            handle.remove()
    # run_case retains provider failures in api/receipt.json, including errors
    # before audit_fixture is called. Do not replace that cause with a missing
    # audit/fixture KeyError or turn a partial execution into a scientific FAIL.
    if raw.get("status") != "COMPLETE":
        diagnostics = "; ".join(str(value) for value in raw.get("failed_bounds", []))
        raise ModelToyExecutionError(
            f"model provider {raw.get('status', 'UNKNOWN')} after "
            f"{raw.get('completed_updates', 0)} updates: {diagnostics or 'no provider diagnostic'}; "
            f"original receipt: {output / 'api/receipt.json'}")
    if "audit" not in observed or "fixture" not in observed:
        raise ModelToyExecutionError(
            f"model provider completed without the required public fixture audit; "
            f"original receipt: {output / 'api/receipt.json'}")
    rows = [{"step": row["step"], **{name: int(value) if type(value) is bool else value
                                     for name, value in row["metrics"].items()}}
            for row in raw["observations"] if row["step"] in raw["protocol"]["metric_evaluation_steps"]
            and row["step"] > 0]
    audit = observed["audit"].receipt()
    expected_mechanisms = task["evaluation"].get("guards", {}).get("mechanism_exercised", False)
    guards = {"all_finite": observed["finite"] and raw["status"] == "COMPLETE",
              "optimizer_updates": counts,
              "unintended_rng_deviations": int("evaluation changed" in " ".join(raw.get("failed_bounds", []))),
              "hooks_exercised": not mechanism_blockers(audit) if expected_mechanisms else True,
              "mechanism_audit": audit}
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
    evidence = {"observations": rows, "live": rows[-1] if rows else {}, "guards": guards,
                "initial_models": observed["initial_models"],
                "prior_policy": prior_policy_receipt(context.recipe, fixture.prior),
                "source": raw["source"], "api_run_status": raw["status"],
                "api_run_verdict": raw["verdict"], "api_receipt_sha256": file_hash(output / "api/receipt.json"),
                "batch_sequence_sha256": getattr(fixture, "batch_sequence_sha256", None),
                "observation_checkpoints": observation_checkpoints,
                "actual_training_gif": {"path": str((output / "api/goal.gif").resolve()),
                                        **raw.get("artifacts", {}).get("goal.gif", {})},
                **executed_receipt(MODEL_TOY_CLEAN, eval_output_noise="clean")}
    checkpoint = torch.load(output / "api/final-state.pt", weights_only=False, map_location="cpu")
    # Providers checkpoint every consumed stream under their own schema. Keep
    # those actual states intact instead of fabricating a GANTrainer layout.
    import shutil
    state_root = output / "provenance"
    state_root.mkdir(exist_ok=True)
    state_file = state_root / "provenance-state.pt"
    shutil.copyfile(output / "api/final-state.pt", state_file)
    evidence["provenance_checkpoint"] = {
        "purpose": "provenance_only", "prerequisite_eligible": False,
        "artifact_root": str(state_root.resolve()), "path": state_file.name,
        "artifact_manifest": manifest_artifacts(state_root), "sha256": file_hash(state_file),
        "state_sha256": state_digest(checkpoint), "completed_steps": raw["completed_updates"],
        "state_owner": "complete provider model/optimizer/policy/RNG checkpoint",
        "optimizer_updates_added": 0, "sampling_draws_added": 0}
    receipt = {**context.receipt(), "api_components": raw["api_components"], "evidence": evidence,
               "cost": {"completed_steps": raw["completed_updates"], "optimizer_updates": counts,
                        "adapter_loop_seconds": raw["elapsed_seconds"]},
               "scope": "ordinary_full_task" if raw["default_protocol_complete"] else "integration_demo_only",
               "declared_task_updates": task["execution"]["steps"], "execution_limit": raw["protocol"]["updates"]}
    atomic_json(output / "adapter-receipt.json", receipt)
    return receipt
