"""Explicit adaptation of a reference recipe to task-owned resources/objectives.

Only the fields already owned by existing adapters may be delegated. Training
mechanisms, priors, serving laws and gates cannot be removed by this contract.
The complete reference recipe remains in the frozen candidate declaration.
"""
from copy import deepcopy
from .priors import recipe_owned_prior


BEHAVIOR_HOST_FIELDS = frozenset({
    "total_steps", "batch_size", "num_particles", "z_dim", "encoder_mode",
    "model", "num_classes", "conditioning", "ucd_target", "ucd_weight", "alpha_bar",
    "prior_reg", "reconstruction_weight", "observation_sigma",
})
RESOURCE_FIELDS = frozenset({"total_steps", "batch_size", "num_particles", "z_dim"})
ADAPTABLE_FIELDS = BEHAVIOR_HOST_FIELDS | {"routing_temperature", "distance_reduction"}


def validate_host_adaptation(candidate):
    contract = candidate.get("host_adaptation")
    if contract is None:
        return
    if not isinstance(contract, dict) or set(contract) != {"schema_version", "recipe_fields"}:
        raise ValueError("host_adaptation requires schema_version and recipe_fields")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        raise ValueError("unsupported host_adaptation schema_version")
    names = contract["recipe_fields"]
    if (not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or set(names) - ADAPTABLE_FIELDS):
        raise ValueError("host_adaptation may delegate only distinct task-owned recipe fields")
    if set(names) - set(candidate.get("recipe_overrides", {})):
        raise ValueError("host_adaptation fields must have explicit reference recipe values")


def delegated_fields(candidate, task):
    validate_host_adaptation(candidate)
    contract = candidate.get("host_adaptation")
    if contract is None:
        return {}
    host = task.get("execution", {}).get("host", task.get("id"))
    if task.get("adapter") == "transfer_behavior" and host != "mode_hold":
        owned = BEHAVIOR_HOST_FIELDS
        if host != "ae_gan_hold":
            owned = owned | {"routing_temperature", "distance_reduction"}
    elif task.get("adapter") in {"public_api_model_toy", "public_api_model_toy_v2"}:
        owned = RESOURCE_FIELDS | set(task["execution"].get("recipe_fields", {}))
    else:
        owned = RESOURCE_FIELDS
    if recipe_owned_prior(task):
        owned = owned - {"prior_reg"}
    return {name: deepcopy(candidate["recipe_overrides"][name])
            for name in sorted(set(contract["recipe_fields"]) & owned)}


def bind_task_candidate(candidate, task):
    """Use the same opt-in binding at preflight and execution boundaries."""
    delegated = delegated_fields(candidate, task)
    bound = deepcopy(candidate)
    bound["recipe_overrides"] = {name: value for name, value in
                                 bound.get("recipe_overrides", {}).items() if name not in delegated}
    # Binding is materialized exactly once; the reference stays in the request.
    bound.pop("host_adaptation", None)
    return bound


def adaptation_receipt(candidate, task):
    if candidate.get("host_adaptation") is None:
        return None
    receipt = {"schema_version": 1, "task_id": task["id"],
            "contract": deepcopy(candidate["host_adaptation"]),
            "delegated_reference_values": delegated_fields(candidate, task),
            "owner": "frozen task resources and original host objectives",
            "host_definition": deepcopy(task["execution"].get("host_definition", {})),
            "note": "Effective recipe and active components are recorded separately. Behavioral host objectives replace delegated reference objectives; scalar trainer prior_reg is retained."}
    if recipe_owned_prior(task):
        receipt["note"] = "Task adaptation never delegates prior policy or regularization under recipe_owned_v1."
    return receipt
