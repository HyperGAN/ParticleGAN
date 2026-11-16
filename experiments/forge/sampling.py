"""Versioned evaluation sampling contracts, separate from training noise/prior width.

Version 1 is required for new planning. An archived task without the version
marker keeps its frozen grading semantics; a historical label is never upgraded
or interpreted as an alias. Policies describe the executed measurement path,
including parameter-only hosts and the clock probe's evaluation-cadence branch.
"""
from __future__ import annotations

VERSION = 1
FIELDS = ("sampling_contract_version", "sampling_law", "eval_output_noise")
PUBLIC_PRIOR_CLEAN = "public_prior_without_output_noise"
ENUMERATED_PRIOR_CLEAN = "enumerated_prior_without_output_noise"
CONDITIONAL_CENTERS = "conditional_prior_centers_with_scheduled_output_noise"
GENERATED_AND_RECONSTRUCTED = "generated_and_reconstructed_prior_with_scheduled_output_noise"
JOINT_WORDS_CLEAN = "generated_and_paired_reconstructed_prior_without_output_noise"
PARTICLES_AND_GRADIENT = "learned_particles_and_critic_gradient"
PARAMETER_MEASUREMENT = "learned_parameter_measurement"
TRAINING_UPDATE_MEASUREMENT = "training_update_measurement"
MODEL_TOY_CLEAN = "task_model_kernel_and_posterior_without_serving_noise"

# A scheduled policy remains the same policy when its configured amplitude is
# zero. Prior kernels remain governed by execution.prior, never by this field.
POLICIES = {
    PUBLIC_PRIOR_CLEAN: "clean",
    ENUMERATED_PRIOR_CLEAN: "clean",
    CONDITIONAL_CENTERS: "public_recipe_schedule",
    GENERATED_AND_RECONSTRUCTED: "public_recipe_schedule",
    JOINT_WORDS_CLEAN: "clean",
    PARTICLES_AND_GRADIENT: "not_applied_to_measurement",
    PARAMETER_MEASUREMENT: "not_applied_to_measurement",
    TRAINING_UPDATE_MEASUREMENT: "not_applied_to_measurement",
    MODEL_TOY_CLEAN: "clean",
}
# Scoped selected-policy laws cannot impersonate the original clean/live laws.
# Each registered policy cohort has its own prefix (see tier1_policy cohorts).
POLICY_LAW_PREFIXES = ("tier1_selected_", "policy_cloud_")
POLICIES.update({prefix + law: noise for prefix in POLICY_LAW_PREFIXES for law, noise in list(POLICIES.items())})
# The as-served measurement keeps the formulation's learned output noise
# exactly as GANTrainer serves it: GANTrainer.sample(output_noise=True) at the
# measured state (learned sigma with its policy floor), drawn from the isolated
# evaluation stream. Only clean sample laws have an as-served counterpart, and
# each name exists only under a registered as-served cohort prefix, so no
# ordinary task or clean cohort can declare it.
SERVED_LEARNED_NOISE = "learned_as_served"
AS_SERVED_LAWS = {PUBLIC_PRIOR_CLEAN: "public_prior_with_learned_output_noise",
                  ENUMERATED_PRIOR_CLEAN: "enumerated_prior_with_learned_output_noise"}
AS_SERVED_LAW_PREFIXES = ("policy_cloud_as_served_",)
POLICIES.update({prefix + law: SERVED_LEARNED_NOISE
                 for prefix in AS_SERVED_LAW_PREFIXES for law in AS_SERVED_LAWS.values()})

ADAPTER_POLICIES = {
    "public_api_model_toy": MODEL_TOY_CLEAN,
    "particle_interpolation": ENUMERATED_PRIOR_CLEAN,
    "dcgan_performance_v1": TRAINING_UPDATE_MEASUREMENT,
    "word_joint": JOINT_WORDS_CLEAN,
    "transfer_vector": PUBLIC_PRIOR_CLEAN,
    "transfer_image": ENUMERATED_PRIOR_CLEAN,
    "native100": PUBLIC_PRIOR_CLEAN,
    "native100_continuation": PUBLIC_PRIOR_CLEAN,
    "ring_endurance": PUBLIC_PRIOR_CLEAN,
    "paired_adaptation": PUBLIC_PRIOR_CLEAN,
    # The metric compares state. This law describes the actual samples drawn in
    # its evaluation-cadence perturbation, not a distribution-quality score.
    "clockfree_audit": PUBLIC_PRIOR_CLEAN,
}
BEHAVIOR_POLICIES = {
    "mode_hold": PUBLIC_PRIOR_CLEAN,
    "two_pole": PARTICLES_AND_GRADIENT,
    "trajectory": CONDITIONAL_CENTERS,
    "residual_student": CONDITIONAL_CENTERS,
    "ae_gan_hold": GENERATED_AND_RECONSTRUCTED,
    "unipolar": PARAMETER_MEASUREMENT,
    "cover_leftover": PARAMETER_MEASUREMENT,
    "unused_token_hold": PARAMETER_MEASUREMENT,
    "mid_scale_identity": PARAMETER_MEASUREMENT,
}


def executed_receipt(sampling_law: str, *, eval_output_noise: str) -> dict:
    """Attest an explicitly chosen executed path; never copy task declarations.

    Call at the adapter branch that actually selected the sampler/measurement.
    This records its policy; RNG/state isolation remains independently audited.
    """
    if not isinstance(sampling_law, str) or sampling_law not in POLICIES:
        raise ValueError(f"unsupported evaluation sampling law: {sampling_law!r}")
    if eval_output_noise != POLICIES[sampling_law]:
        raise ValueError(f"sampling law {sampling_law!r} requires eval_output_noise={POLICIES[sampling_law]!r}")
    return {"sampling_contract_version": VERSION, "sampling_law": sampling_law,
            "eval_output_noise": eval_output_noise}


def expected_policy(task: dict) -> dict:
    """One central binding of implemented adapter/host paths to their policies."""
    from .tier1_policy import task_cohort
    cohort = task_cohort(task)
    if cohort is not None:
        from .tier1_policy import validate
        parent = validate(task)
        original = expected_policy(parent)
        return executed_receipt(cohort.sampling_law(original["sampling_law"]),
                                eval_output_noise=cohort.eval_output_noise(original["eval_output_noise"]))
    adapter = task.get("adapter")
    if adapter == "transfer_behavior":
        host = task.get("execution", {}).get("host", task.get("id"))
        law = BEHAVIOR_POLICIES.get(host)
    else:
        law = ADAPTER_POLICIES.get(adapter)
    if law is None:
        raise ValueError(f"{task.get('id', '<task>')}: no supported evaluation sampling policy for adapter/host")
    return executed_receipt(law, eval_output_noise=POLICIES[law])


def validate_declaration(task: dict, *, required: bool = False) -> dict | None:
    """Validate an opt-in declaration without changing archived task semantics."""
    evaluation = task.get("evaluation", {})
    if "sampling_contract_version" not in evaluation:
        if required:
            raise ValueError("new planning requires evaluation.sampling_contract_version=1")
        return None
    version = evaluation["sampling_contract_version"]
    if type(version) is not int or version != VERSION:
        raise ValueError(f"unsupported sampling_contract_version: {version!r}")
    missing = set(FIELDS) - evaluation.keys()
    if missing:
        raise ValueError(f"sampling contract lacks {sorted(missing)}")
    return executed_receipt(evaluation["sampling_law"], eval_output_noise=evaluation["eval_output_noise"])


def candidate_blockers(candidate: dict) -> list[str]:
    claims = candidate.get("claim_contract", {})
    if not isinstance(claims, dict) or claims.get("sampling_law") != "task_declared":
        return ["new planning requires claim_contract.sampling_law='task_declared'; evaluation policies belong to each task"]
    return []


def task_blockers(task: dict) -> list[str]:
    from .paired_sampling import paired_sampling_blockers
    paired_blockers = paired_sampling_blockers(task)
    if paired_blockers:
        return paired_blockers
    try:
        declared = validate_declaration(task, required=True)
        if declared != expected_policy(task):
            raise ValueError("evaluation sampling declaration differs from the implemented adapter/host policy")
    except (KeyError, TypeError, ValueError) as exc:
        return [f"{task.get('id', '<task>')}: {exc}"]
    return []


def grade_sampling(task: dict, evidence: dict) -> dict | None:
    """Return only contract errors; the scientific grader still judges metrics."""
    try:
        declared = validate_declaration(task)
        if declared is None:
            return None
        if declared != expected_policy(task):
            raise ValueError("evaluation sampling declaration differs from the implemented adapter/host policy")
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "INVALID", "reason": str(exc)}
    missing = set(FIELDS) - evidence.keys()
    if missing:
        return {"status": "INCOMPLETE", "reason": f"missing observed sampling contract fields: {sorted(missing)}"}
    if type(evidence["sampling_contract_version"]) is not int or any(
            evidence[key] != declared[key] for key in FIELDS):
        return {"status": "INVALID", "reason": "observed sampling policy differs from the frozen task contract"}
    return None


def validate_request_sampling(request: dict, *, task_ids=None) -> None:
    """Recompute prospective policies at queue/dispatch boundaries.

    The verified execution source, not a removable request flag, sets the
    version floor. Historical snapshots without this module keep their frozen
    execution contract. A new snapshot cannot downgrade by deleting its source
    manifest entry: verification also rejects undeclared files on disk.

    Queue callers check all authorized grouped members. Runtime callers pass
    every member of the job immediately before executing its adapter.
    """
    from pathlib import Path
    from .sources import verify_snapshot

    module = "experiments/forge/sampling.py"
    source = request.get("source", {})
    declared = module in source.get("files", {})
    snapshot = source.get("snapshot_path")
    present = isinstance(snapshot, str) and bool(snapshot) and (Path(snapshot) / module).is_file()
    if not declared and not present:
        return
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("sampling contract validation requires a frozen source snapshot")
    verify_snapshot(Path(snapshot), source)
    blockers = candidate_blockers(request.get("candidate", {}))
    if task_ids is None:
        authorized = {row["task"] for row in request["view"]["assignments"]
                      if row["qualification_tier"] <= request["through_tier"]}
        members = set()
        for job in request["jobs"]:
            group = set(job.get("task_ids", [job["task_id"]]))
            if group & authorized:
                members.update(group)
        # Missing jobs must not erase an authorized task from this validation.
        members.update(authorized)
    else:
        members = set(task_ids)
    tasks = request.get("tasks", {})
    for member in sorted(members):
        if member not in tasks:
            blockers.append(f"{member}: authorized task lacks its frozen definition")
        else:
            blockers.extend(task_blockers(tasks[member]))
    if blockers:
        raise ValueError("sampling contract blocked: " + "; ".join(blockers))
