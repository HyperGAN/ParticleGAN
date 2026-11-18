"""Verify the three matched arms, their independently graded gates and earlier controls."""
import argparse
import json
from pathlib import Path

from experiments.forge.contracts import file_hash, read_json, stable_hash
from experiments.forge.views import grade_result

HERE = Path(__file__).resolve().parent
SOURCE = "d2abfed51fd36f494c896317e8055c9c814bd046"
RAW = "rhbi-v12"
FULL = "rhbi-v12-feature-transport"
CANDIDATE = "rhbi-v12-feature-raw-self"
MODES = {RAW: False, FULL: True, CANDIDATE: "raw_self"}


def prior_observations(runs, stage):
    if stage == "texture":
        previous = read_json(HERE.parent / "assignment/results.json")["arms"]
        return {a: previous[a]["observations_sha256"] for a in (RAW, FULL)}
    prior = read_json(HERE.parent / "trajectory/results.json")["tasks"]["interp_sparse_arc"]
    reference = {}
    for arm, role in ((RAW, "parent"), (FULL, "feature")):
        source = prior[role]
        path = runs / "rhbi-retention-trajectory-arc-v1" / source["attempt"] / "result.json"
        if file_hash(path) != source["result_file_sha256"]:
            raise ValueError(f"Archived arc control differs for {arm}")
        reference[arm] = stable_hash(read_json(path)["task_results"][0]["evidence"]["observations"])
    return reference


def compare(runs, stage):
    task_id, steps = ("img_conv_texture32", 288) if stage == "texture" else ("interp_sparse_arc", 600)
    expected = prior_observations(runs, stage)
    common, arms = None, {}
    for envelope in sorted((runs / f"rhbi-raw-self-{stage}-v1").glob("*/request.json")):
        directory = envelope.parent
        env = read_json(envelope)
        req, science = env["request"], env["job"]["science"]
        arm = req["candidate"]["id"]
        source = req["source"]
        if (arm not in MODES or arm in arms or source["origin_commit"] != SOURCE
                or env["job"]["task_id"] != task_id or science["seed"] != 0):
            raise ValueError(f"Unexpected attempt {directory}")
        task = req["tasks"][task_id]
        result = read_json(directory / "result.json")["task_results"][0]
        raw = read_json(directory / "raw-result.json")
        evidence = raw["evidence"]
        grade = grade_result(task, raw)
        observations = evidence["observations"]
        guards, ckpt = evidence["guards"], evidence["provenance_checkpoint"]
        recipe = dict(raw["recipe"])
        if (recipe.pop("rhbi_feature_transport", False) != MODES[arm]
                or result["task_id"] != task_id or grade["gate_status"] not in {"PASS", "FAIL"}
                or grade["gate_status"] != result["gate_status"] or grade["evaluator_result"] != result["evaluator_result"]
                or observations != result["evidence"]["observations"] or len(observations) != 24
                or observations[-1]["step"] != steps or guards["unintended_rng_deviations"] != 0
                or guards["optimizer_updates"] != dict(generator=steps, discriminator=steps, prior=steps)
                or file_hash(directory / "goal.gif") != evidence["actual_training_gif"]["sha256"]
                or file_hash(directory / "provenance" / ckpt["path"]) != ckpt["sha256"]):
            raise ValueError(f"Inconsistent measurement, recipe or artifact for {arm}")
        contract = {"source_commit": SOURCE, "source_sha256": source["digest"], "seed": science["seed"],
                    "runtime": raw["rng"]["runtime"], "compute": science["compute"],
                    "initializer": raw["initializer"], "initial_parameters": evidence["initial_parameters"],
                    "initial_prior": raw["initial_prior"], "other_recipe_fields_sha256": stable_hash(recipe),
                    "named_initial_streams_sha256": stable_hash(raw["rng"]["bindings"]),
                    "named_terminal_streams_sha256": stable_hash(ckpt["named_stream_state_sha256"]),
                    "seen_batches_sha256": evidence["batch_sequence_sha256"], "execution": stable_hash(task["execution"]),
                    "evaluation": stable_hash(task["evaluation"]), "scoring_weights": evidence["executed_scoring_weights"],
                    "sampling_law": evidence["sampling_law"], "eval_output_noise": evidence["eval_output_noise"]}
        if common is not None and common != contract:
            raise ValueError(f"Mismatched controls for {arm}: {[k for k in contract if contract[k] != common[k]]}")
        common = contract
        digest = stable_hash(observations)
        data = {"attempt": directory.name, "candidate_revision": req["candidate_revision"], "mode": MODES[arm],
                "gate": grade["gate_status"], "terminal": grade["evaluator_result"], "terminal_observations": observations[-5:],
                "observations_sha256": digest, "gif_sha256": evidence["actual_training_gif"]["sha256"],
                "checkpoint_sha256": ckpt["sha256"], "result_sha256": file_hash(directory / "result.json"),
                "worker_seconds": result["cost"]["wall_seconds"]}
        if arm in expected:
            data["reproduced_previous_24_observations"] = digest == expected[arm]
        arms[arm] = data
    if set(arms) != set(MODES):
        raise ValueError("Expected three complete distinct arms")
    return {"task": task_id, "qualification_input": False, "matched": common, "arms": arms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forge_runs", type=Path)
    parser.add_argument("stage", choices=("texture", "arc"))
    args = parser.parse_args()
    print(json.dumps(compare(args.forge_runs, args.stage), sort_keys=True, indent=2, allow_nan=False))
