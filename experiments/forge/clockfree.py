"""Measured clock perturbations on the public trainer, with saved state proof.

Passing a short probe is only one eligibility check. The declared long
continuations remain required to expose state-dependent delayed failures.
"""
from copy import deepcopy
from pathlib import Path
import time

import torch

from .artifacts import manifest_artifacts, verify_artifacts
from .contracts import atomic_json, file_hash, stable_hash
from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
from .state import require_consistent_rng, require_optimizer_steps, state_digest


ALLOWED_STATE = ["parameters_and_ema", "optimizer_moments_and_bias_correction",
                 "observed_gradients_and_row_visit_history",
                 "continuous_policy_statistics_and_fifo_reservoirs", "named_training_rng"]


def source_audit(recipe, extensions):
    """Conservative declaration for the shipped scalar API; unknown hooks block."""
    dependencies = []
    continuous = recipe.get("continuous_policy")
    if continuous is not None:
        if continuous != "dv12" or recipe.get("total_steps") is not None or recipe.get("lr_control") != "stationarity":
            dependencies.append("continuous policy lifecycle needs a separate reviewed clock/state audit")
    elif (recipe["lr_floor"] != 1 or recipe["network_lr_floor"] not in (None, 1)
          or (recipe.get("lr_schedule") == "exponential" and recipe.get("lr_decay_rate", 0.96) != 1)):
        dependencies.append("learning-rate annealing depends on completed steps and horizon")
    if recipe.get("beta2_end") is not None or recipe.get("reg_coeff_end") is not None:
        dependencies.append("optimizer or penalty schedule depends on completed steps and horizon")
    if recipe["input_noise_std"]:
        dependencies.append("input-noise annealing depends on completed steps and horizon")
    if recipe["output_noise_std"] and recipe["output_noise_warmup"]:
        dependencies.append("output-noise warmup depends on completed steps and horizon")
    if recipe["reg_coeff"] and recipe["reg_every"] != 1:
        dependencies.append("lazy critic penalty uses a periodic update counter")
    if recipe["d_guard_ratio"] and recipe["d_guard_min_steps"]:
        dependencies.append("critic guard releases at a fixed minimum update count")
    if extensions:
        dependencies.append("additional formulation bindings need an explicit clock/state audit")
    root = Path(__file__).resolve().parents[2]
    files = ["particlegan/training.py", "particlegan/recipes.py", "particlegan/recipe_schedules.py",
             "particlegan/k3p.py", "particlegan/ka2.py", "particlegan/policy.py", "particlegan/continuous.py",
             "particlegan/birth_death.py", "particlegan/feature_cells.py", "particlegan/feature_policy.py",
             "particlegan/feature_reference.py", "particlegan/latent_geometry.py", "particlegan/mean_transport.py",
             "particlegan/row_evidence.py", "particlegan/priors.py",
             "particlegan/_t1_prior/pullback.py",
             "particlegan/_t4_arch/optimizer.py", "particlegan/_t4_arch/pullback.py",
             "particlegan/_unified_v4/critic.py", "particlegan/_unified_v4/loss.py",
             "particlegan/_unified_v4/optimizer.py", "particlegan/_unified_v4/penalty.py",
             "particlegan/_unified_v4/prior.py",
             "particlegan/_unified_v11/bank.py", "particlegan/_unified_v11/critic.py",
             "particlegan/_unified_v11/loss.py", "particlegan/_unified_v11/optimizer.py",
             "particlegan/_unified_v11/penalty.py", "particlegan/_unified_v11/prior.py",
             "particlegan/_unified_v15/atomic.py", "particlegan/_unified_v15/charts.py",
             "particlegan/_unified_v15/critic.py", "particlegan/_unified_v15/loss.py",
             "particlegan/_unified_v15/optimizer.py", "particlegan/_unified_v15/penalty.py",
             "particlegan/_unified_v15/prior.py",
             "particlegan/_r4_ablation/optimizer.py", "particlegan/_r4_ablation/penalty.py",
             "particlegan/_r4_ablation/prior.py",
             "particlegan/_minimal/dynamics.py", "particlegan/_minimal/geometry.py",
             "particlegan/_minimal/prior.py",
             "particlegan/_axiom/math_ops.py", "particlegan/_axiom/state.py",
             "particlegan/_axiom/atlas.py", "particlegan/_axiom/fiber.py",
             "particlegan/_axiom/prior.py", "particlegan/_axiom/critic.py",
             "particlegan/_axiom/transport.py", "particlegan/_axiom/optimizer.py",
             "particlegan/_axiom/loss.py",
             "particlegan/grad_regularizers.py", "experiments/forge/clockfree.py",
             "experiments/forge/api.py", "experiments/forge/rng.py"]
    return {"source_sha256": {name: file_hash(root / name) for name in files},
            "allowed_state": ALLOWED_STATE, "unexplained_clock_dependencies": dependencies,
            "scope": "shipped public scalar trainer and explicitly listed state; no arbitrary-extension certification"}


def learning_state(state):
    """Exclude only external clocks/configuration and independent eval streams."""
    trainer = state["trainer"]
    stream_names = state["streams"]["manifest"]["bindings"]
    named = {key: value for key, value in state["streams"]["states"].items()
             if stream_names[key]["family"] != "eval"}
    cleaned_trainer = {key: value for key, value in trainer.items()
                       if key not in {"recipe", "completed_steps", "max_steps", "streams"}}
    if "lr_settle" in cleaned_trainer:
        def _sanitize_settle(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if k in ("last_look", "last") and isinstance(v, dict):
                        out[k] = {lk: lv for lk, lv in v.items() if lk != "step"}
                    elif k == "log" and isinstance(v, list):
                        out[k] = [row[1:] if isinstance(row, list) and len(row) == 4 else row for row in v]
                    else:
                        out[k] = _sanitize_settle(v)
                return out
            if isinstance(obj, list):
                return [_sanitize_settle(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_sanitize_settle(x) for x in obj)
            return obj
        cleaned_trainer["lr_settle"] = _sanitize_settle(cleaned_trainer["lr_settle"])
    if isinstance(cleaned_trainer.get("birth_death"), dict):
        bd = dict(cleaned_trainer["birth_death"])
        if isinstance(bd.get("iso_log"), list):
            bd["iso_log"] = [row[1:] if isinstance(row, list) and len(row) == 3 else row for row in bd["iso_log"]]
        if isinstance(bd.get("last"), dict):
            last = {k: v for k, v in bd["last"].items() if k not in ("step", "eval_seconds")}
            if isinstance(last.get("mean_transport"), dict):
                last["mean_transport"] = {k: v for k, v in last["mean_transport"].items() if k != "step"}
            if isinstance(last.get("paired_average"), dict):
                last["paired_average"] = {k: v for k, v in last["paired_average"].items() if k != "step"}
            bd["last"] = last
        if isinstance(bd.get("paired_average"), dict):
            bd["paired_average"] = {k: v for k, v in bd["paired_average"].items() if k != "step"}
        cleaned_trainer["birth_death"] = bd
    if isinstance(cleaned_trainer.get("surprise"), dict):
        surprise = dict(cleaned_trainer["surprise"])
        if isinstance(surprise.get("log"), list):
            surprise["log"] = [row[1:] if isinstance(row, list) and len(row) == 2 else row for row in surprise["log"]]
        cleaned_trainer["surprise"] = surprise
    if isinstance(cleaned_trainer.get("reopen_guard"), dict):
        guard = dict(cleaned_trainer["reopen_guard"])
        if isinstance(guard.get("excursion"), dict):
            guard["excursion"] = {k: v for k, v in guard["excursion"].items() if k != "step"}
        cleaned_trainer["reopen_guard"] = guard
    return {"trainer": cleaned_trainer,
            "training_rng": named}


def _proof_digest(value, proof):
    if proof["recipe"].get("continuous_policy") is not None:
        from .policy_adapters import typed_state_digest
        return typed_state_digest(value)
    return state_digest(value)


def _comparisons(proof):
    digest = lambda value: _proof_digest(value, proof)
    initial = learning_state(proof["initial"])
    common = {"permitted_state_sha256": digest(initial),
              "rng_state_sha256": digest(initial["training_rng"])}
    reference = digest([learning_state(s) for s in proof["trajectories"]["reference"]])
    return [{"condition": name, **common, "reference_sha256": reference,
             "changed_sha256": digest([learning_state(s) for s in proof["trajectories"][name]])}
            for name in ("step_label", "horizon", "evaluation_cadence", "restart")]


def run_clockfree(request, task, output, device):
    from .adapters import _context, _models, _event

    spec = task["execution"]
    probe_steps, warmup = spec["probe_steps"], spec["warmup_steps"]
    offset = spec["step_label_offset"]
    if any(type(n) is not int or n < 1 for n in (probe_steps, warmup, offset, spec["perturbed_horizon"])):
        raise ValueError("clock probes need positive declared steps, offset and horizon")
    maximum = offset + warmup + probe_steps + 1
    started = time.monotonic()

    def construct():
        context = _context(request, task, device, spec["resources"])
        trainer = context.build_trainer(*_models(context, spec["model"]), max_steps=maximum)
        # Bind every probe stream before the checkpoint, including eval-only names.
        data = context.streams.generator("data", component="clock_probe", purpose="training")
        context.streams.generator("eval", component="clock_probe", purpose="samples")
        return context, trainer, data

    def update(context, trainer, data):
        real = torch.randn(context.recipe.batch_size, 2, device=device, generator=data)
        trainer.step(real)

    context, trainer, data = construct()
    for _ in range(warmup):
        update(context, trainer, data)
    initial = context.state_dict()
    proof = {"schema_version": 1, "execution": deepcopy(spec), "initial": initial,
             "recipe": context.recipe.to_dict(), "extensions": context.extension_values,
             "trajectories": {}, "branch_initial": {}}
    directory = Path(output) / "clockfree-proof"
    directory.mkdir(parents=True)
    torch.save(initial, directory / "initial.pt")
    for name in ("reference", "step_label", "horizon", "evaluation_cadence", "restart"):
        if name != "reference":
            context, trainer, data = construct()
            restored = (torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)
                        if name == "restart" else deepcopy(initial))
            context.load_state_dict(restored)
        proof["branch_initial"][name] = context.state_dict()
        if name == "step_label":
            trainer.completed_steps += offset
        elif name == "horizon":
            if context.recipe.total_steps is None:
                # Public continuous Recipes reject a training horizon. Their
                # only legal horizon is the independent external execution cap.
                trainer.max_steps = spec["perturbed_horizon"]
            else:
                context.recipe = context.recipe.replace(total_steps=spec["perturbed_horizon"])
                trainer.recipe = context.recipe
                trainer.policy.recipe = context.recipe
        trajectory = []
        for _ in range(probe_steps):
            if name == "evaluation_cadence":
                generator = context.streams.generator("eval", component="clock_probe", purpose="samples")
                trainer.sample(spec["evaluation_samples"], generator=generator, output_noise=False)
            update(context, trainer, data)
            trajectory.append(context.state_dict())
        proof["trajectories"][name] = trajectory
        _event("clock_probe", task=task["id"], condition=name, updates=probe_steps)
    torch.save(proof, directory / "comparisons.pt")
    law = (task["evaluation"]["sampling_law"] if context.policy_task is not None else PUBLIC_PRIOR_CLEAN)
    evidence = {**executed_receipt(law, eval_output_noise="clean"),
                "scoring_weights": task["evaluation"].get("scoring_weights", "live"),
                "comparisons": _comparisons(proof),
                "source_audit": source_audit(proof["recipe"], proof["extensions"]),
                "artifact_root": str(directory.resolve()), "artifact_manifest": manifest_artifacts(directory),
                "artifact_portability": {"storage": "local", "requires_bulk_artifacts": True}}
    result = {"evidence": evidence, "cost": {"training_seconds": time.monotonic() - started,
              "completed_updates": warmup + 5 * probe_steps}, "execution_path": "public_trainer",
              "recipe": proof["recipe"]}
    atomic_json(Path(output) / "adapter-receipt.json", result)
    return result


def verify_probe(task, evidence):
    """Recompute state comparisons from certified tensors in the frozen evaluator."""
    root = Path(evidence["artifact_root"])
    verify_artifacts(root, evidence["artifact_manifest"])
    proof = torch.load(root / "comparisons.pt", map_location="cpu", weights_only=True)
    if proof.get("schema_version") != 1 or proof.get("execution") != task["execution"]:
        raise ValueError("clock probe execution differs from its task")
    names = {"reference", "step_label", "horizon", "evaluation_cadence", "restart"}
    if set(proof["trajectories"]) != names or set(proof["branch_initial"]) != names:
        raise ValueError("clock probe must preserve every comparison branch")
    digest = lambda value: _proof_digest(value, proof)
    if proof["recipe"].get("continuous_policy") is not None:
        from .policy_adapters import finite_policy_state
        states = [proof["initial"], *proof["branch_initial"].values(),
                  *(state for trajectory in proof["trajectories"].values() for state in trajectory)]
        if not all(finite_policy_state(state) for state in states):
            raise ValueError("clock proof contains nonfinite learned public policy state")
    initial_hash = digest(proof["initial"])
    if proof["initial"]["trainer"]["completed_steps"] != task["execution"]["warmup_steps"]:
        raise ValueError("clock initial state is not at the declared warmup")
    require_optimizer_steps(proof["initial"], task["execution"]["warmup_steps"])
    require_consistent_rng(proof["initial"])
    if (proof["recipe"] != proof["initial"]["recipe"]
            or proof["recipe"] != proof["initial"]["trainer"]["recipe"]
            or proof["extensions"] != proof["initial"]["extensions"]):
        raise ValueError("clock source audit formulation differs from the measured initial state")
    if initial_hash != digest(torch.load(root / "initial.pt", map_location="cpu", weights_only=True)):
        raise ValueError("serialized restart differs from the measured initial state")
    for name in names:
        if digest(proof["branch_initial"][name]) != initial_hash:
            raise ValueError("clock comparison did not start from identical own state")
        trajectory = proof["trajectories"][name]
        if len(trajectory) != task["execution"]["probe_steps"]:
            raise ValueError("clock comparison lacks its complete common prefix")
        offset = task["execution"]["step_label_offset"] if name == "step_label" else 0
        for index, state in enumerate(trajectory, 1):
            require_optimizer_steps(state, task["execution"]["warmup_steps"] + index)
            require_consistent_rng(state)
            if state["trainer"]["completed_steps"] != task["execution"]["warmup_steps"] + offset + index:
                raise ValueError("clock perturbation labels do not match the declared probe")
            continuous = proof["recipe"].get("continuous_policy") is not None
            horizon = (task["execution"]["perturbed_horizon"] if name == "horizon" and not continuous
                       else proof["recipe"]["total_steps"])
            if continuous:
                maximum = (task["execution"]["perturbed_horizon"] if name == "horizon"
                           else proof["initial"]["trainer"]["max_steps"])
                if state["trainer"].get("max_steps") != maximum:
                    raise ValueError("continuous clock probe external horizon differs")
            expected_recipe = {**proof["recipe"], "total_steps": horizon}
            if (state["recipe"] != expected_recipe or state["trainer"]["recipe"] != expected_recipe
                    or state["extensions"] != proof["extensions"]
                    or state["prior"] != proof["initial"]["prior"]
                    or state["initializer"] != proof["initial"]["initializer"]
                    or state["initialization"] != proof["initial"]["initialization"]):
                raise ValueError("clock perturbation changed formulation beyond its declared horizon")
    comparisons = _comparisons(proof)
    audit = source_audit(proof["recipe"], proof["extensions"])
    if stable_hash(comparisons) != stable_hash(evidence.get("comparisons")) or audit != evidence.get("source_audit"):
        raise ValueError("clock verdict declarations disagree with the saved state or source audit")
    verify_artifacts(root, evidence["artifact_manifest"])
    return comparisons, audit
