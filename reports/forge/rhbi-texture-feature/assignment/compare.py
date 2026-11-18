"""Reduce the frozen texture comparison and verify its matched controls."""
import argparse
import json
from pathlib import Path

from experiments.forge.contracts import file_hash, read_json, stable_hash

TASK = "img_conv_texture32"
CAMPAIGN = "rhbi-assignment-texture-v1"
SOURCE = "c64bdb77b2bf4aeedac5954d3d10a188b27886d4"
MODES = {"rhbi-v12": False, "rhbi-v12-feature-transport": True,
         "rhbi-v12-feature-assignment": "assignment_only"}


def compare(runs):
    old = read_json(Path(__file__).resolve().parents[1] / "comparison.json")["arms"]
    arms, shared = {}, None
    for path in sorted((runs / CAMPAIGN).glob("*/request.json")):
        directory = path.parent
        env = read_json(path)
        request, science = env["request"], env["job"]["science"]
        arm = request["candidate"]["id"]
        source = request["source"]
        if (arm in arms or arm not in MODES or source["origin_commit"] != SOURCE
                or science["seed"] != 0 or env["job"]["task_id"] != TASK):
            raise ValueError(f"Unexpected attempt {directory}")
        result = read_json(directory / "result.json")["task_results"][0]
        raw = read_json(directory / "raw-result.json")
        evidence = raw["evidence"]
        task = request["tasks"][TASK]
        recipe = dict(raw["recipe"])
        if recipe.pop("rhbi_feature_transport", False) != MODES[arm]:
            raise ValueError(f"Wrong recipe for {arm}")
        guards = evidence["guards"]
        observations = evidence["observations"]
        gif = evidence["actual_training_gif"]
        checkpoint = evidence["provenance_checkpoint"]
        if (result["gate_status"] not in {"PASS", "FAIL"} or result["task_id"] != TASK
                or len(observations) != 24 or observations[-1]["step"] != 288
                or guards["optimizer_updates"] != dict(generator=288, discriminator=288, prior=288)
                or guards["unintended_rng_deviations"] != 0 or file_hash(directory / "goal.gif") != gif["sha256"]
                or file_hash(directory / "provenance/provenance-state.pt") != checkpoint["sha256"]):
            raise ValueError(f"Incomplete run or media for {arm}")
        contract = {"source_commit": source["origin_commit"], "source_sha256": source["digest"],
                    "protocol_seed": science["seed"], "runtime": raw["rng"]["runtime"], "compute": science["compute"],
                    "initializer": raw["initializer"], "initial_parameters": evidence["initial_parameters"],
                    "initial_prior": raw["initial_prior"], "other_recipe_fields_sha256": stable_hash(recipe),
                    "named_stream_bindings_sha256": stable_hash(raw["rng"]["bindings"]),
                    "terminal_named_streams_sha256": stable_hash(checkpoint["named_stream_state_sha256"]),
                    "seen_batches_sha256": evidence["batch_sequence_sha256"], "task_execution": stable_hash(task["execution"]),
                    "task_evaluation": stable_hash(task["evaluation"]), "scoring_weights": evidence["executed_scoring_weights"],
                    "sampling_law": evidence["sampling_law"], "eval_output_noise": evidence["eval_output_noise"]}
        if shared is not None and contract != shared:
            raise ValueError(f"Unmatched {arm}: {[k for k in contract if contract[k] != shared[k]]}")
        shared = contract
        data = {"mode": MODES[arm], "attempt": directory.name, "revision": request["candidate_revision"],
                "gate": result["gate_status"], "terminal": result["evaluator_result"],
                "final_observation": observations[-1], "terminal_observations": observations[-5:],
                "max_clean_texture_fraction": max(obs["clean_texture_fraction"] for obs in observations),
                "observations_sha256": stable_hash(observations), "actual_training_gif_sha256": gif["sha256"],
                "checkpoint_sha256": checkpoint["sha256"], "result_sha256": file_hash(directory / "result.json"), "seconds": result["cost"]["wall_seconds"]}
        if arm in old:
            reference = old[arm]
            past_dir = runs / "rhbi-texture-feature-v1" / reference["attempt"]
            past = read_json(past_dir / "raw-result.json")["evidence"]
            if (file_hash(past_dir / "provenance/provenance-state.pt") != reference["checkpoint_sha256"]
                    or past["actual_training_gif"]["sha256"] != reference["actual_training_gif_sha256"]):
                raise ValueError(f"Historical source mismatch for {arm}")
            data["earlier_attempt"] = reference["attempt"]
            data["reproduced_24_observations"] = observations == past["observations"]
        arms[arm] = data
    if arms.keys() != MODES.keys():
        raise ValueError("Expected the three frozen arms")
    return {"task": TASK, "research_diagnostic": True, "qualification_input": False, "matched": shared, "arms": arms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forge_runs", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.forge_runs), indent=2, sort_keys=True, allow_nan=False))
