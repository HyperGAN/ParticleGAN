"""Verify matched attempts, regrade frozen retention gates and audit interpolation scores."""
import argparse
import json
import operator
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch

from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import file_hash, read_json, stable_hash
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import render
from experiments.forge.views import grade_result
from interpolation_replay import cross_graph_scores, replay

HERE = Path(__file__).resolve().parent
SOURCE = "7d4b9d0b57b6b01ca2a55dc9a504ea337392d02a"
CAMPAIGN = "rhbi-raw-assignment-retention-v1"
RAW, PARENT, CANDIDATE = "rhbi-v12", "rhbi-v12-feature-raw-self", "rhbi-v12-raw-assignment"
MODES = {RAW: False, PARENT: "raw_self", CANDIDATE: "raw_assignment"}
VECTOR = "vector_unequal_mass"
BUDGET = {VECTOR: 1200, "interp_img_patch_shift": 800, "interp_saddle_barycentric": 800}
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def summarize(envelope, runs, media):
    directory = envelope.parent
    env = read_json(envelope)
    request, science, task_id = env["request"], env["job"]["science"], env["job"]["task_id"]
    arm = request["candidate"]["id"]
    task = request["tasks"][task_id]
    steps = BUDGET[task_id]
    require(request["source"]["origin_commit"] == SOURCE and request["view"]["id"] == CAMPAIGN
            and science["seed"] == request["protocol"]["seed"] == 0 and science["evidence_use"] == "research_diagnostic"
            and task["execution"]["steps"] == steps, "Unexpected source, task, budget or cohort")
    result = read_json(directory / "result.json")["task_results"][0]
    raw = read_json(directory / "raw-result.json")
    evidence = raw["evidence"]
    grade = grade_result(task, raw)
    observations = evidence["observations"]
    require(result["task_id"] == task_id and result["gate_status"] == grade["gate_status"] in {"PASS", "FAIL"}
            and result["evaluator_result"] == grade["evaluator_result"] and observations == result["evidence"]["observations"]
            and len(observations) == 24 and observations[-1]["step"] == steps, f"Incomplete or inconsistent grade {directory}")
    guards = evidence["guards"]
    require(guards["optimizer_updates"] == dict(generator=steps, discriminator=steps, prior=steps)
            and guards["all_finite"] and guards["unintended_rng_deviations"] == 0, "Update or RNG guard failed")
    recipe = raw["recipe"].copy()
    require(recipe.pop("rhbi_feature_transport", False) == MODES[arm], f"Unexpected transport mode {arm}")
    ckpt, archive = evidence["provenance_checkpoint"], evidence["rhbi_trajectory"]
    require(file_hash(directory / "provenance" / ckpt["path"]) == ckpt["sha256"], "Checkpoint bytes differ")
    saved = directory / archive["directory"]
    verify_artifacts(saved, archive["artifacts"])
    terminal = observations[-5:]
    require(archive["state_steps"] == [0, *(row["step"] for row in terminal)], "Incomplete terminal archive")
    init = torch.load(saved / "state-0000.pt", map_location="cpu", weights_only=True)
    batches = archive["real_batch_sequence_sha256"]
    require(evidence.get("batch_sequence_sha256", batches) == batches, "Observer batch hash differs")
    require(set(archive["artifacts"]["files"]) == {f"state-{step:04d}.pt" for step in archive["state_steps"]},
            "Terminal archive file set differs")
    require(raw["initializer"] == "deterministic_orthogonal"
            and evidence["executed_scoring_weights"] == task["evaluation"]["scoring_weights"] == "live"
            and evidence["sampling_law"] == task["evaluation"]["sampling_law"]
            and evidence["eval_output_noise"] == task["evaluation"]["eval_output_noise"] == "clean",
            "Executed initialization or scoring law differs")
    common = dict(source_sha256=request["source"]["digest"], seed=0, compute=science["compute"],
                  execution_sha256=stable_hash(task["execution"]), evaluation_sha256=stable_hash(task["evaluation"]),
                  initializer=raw["initializer"], initial_prior=raw["initial_prior"], runtime=raw["rng"]["runtime"],
                  initial_models_sha256=state_digest(init["trainer"]["models"]), initial_streams_sha256=state_digest(init["streams"]),
                  host_sha256=stable_hash(evidence["host"]), prior_capacity=recipe["num_particles"],
                  named_terminal_streams_sha256=stable_hash(ckpt["named_stream_state_sha256"]), seen_batches_sha256=batches,
                  other_recipe_fields_sha256=stable_hash(recipe), scoring_weights=evidence["executed_scoring_weights"],
                  sampling_law=evidence["sampling_law"], eval_output_noise=evidence["eval_output_noise"])
    bounds = task["evaluation"]["thresholds"]
    rows = []
    for point in terminal:
        failed = [key for key, op, threshold in bounds if not OPS[op](point[key], threshold)]
        row = dict(step=point["step"], failed=failed, metrics={key: point[key] for key, _, _ in bounds})
        if task_id == VECTOR:
            rare = point["target_mass"].index(min(point["target_mass"]))
            row.update(rare_count=point["component_counts"][rare], rare_core_min_eigen_ratio=point["component_core_eigen_ratios"][rare])
        rows.append(row)
    count = sum(not row["failed"] for row in rows)
    verdict = grade["evaluator_result"]
    require((count == 5) == (grade["gate_status"] == "PASS"), "Terminal checks disagree with grade")
    data = dict(attempt=directory.name, candidate_revision=request["candidate_revision"], transport=MODES[arm],
                gate=grade["gate_status"], terminal_passing=count, terminal_observations=rows,
                observations_sha256=stable_hash(observations), result_sha256=file_hash(directory / "result.json"),
                terminal_checkpoint_sha256=ckpt["sha256"], terminal_archive_sha256=archive["artifacts"]["sha256"],
                worker_seconds=result["cost"]["wall_seconds"])
    trained = None
    if task_id == VECTOR:
        data["passing_suffix"] = verdict["convergence"]["passing_suffix"]
        descriptor = evidence["saved_observer_outputs"]
        require(file_hash(directory / descriptor["path"]) == descriptor["sha256"], "Scored vector samples differ")
        data["observed_samples_sha256"] = descriptor["sha256"]
        if arm == RAW:
            prior = read_json(HERE.parents[1] / "trajectory/results.json")["tasks"][VECTOR]["parent"]
            previous = runs / "rhbi-retention-trajectory-v1" / prior["attempt"] / "result.json"
            require(file_hash(previous) == prior["result_file_sha256"], "Archived raw result differs")
            old = read_json(previous)["task_results"][0]["evidence"]["observations"]
            data["reproduced_previous_24_observations"] = observations == old
        if arm == CANDIDATE:
            with TemporaryDirectory() as tmp:
                gif = Path(tmp) / "width-training.gif"
                receipt = render(task, result, directory, gif)
                data["gif_sha256"] = receipt["gif_sha256"]
                if media:
                    shutil.copyfile(gif, media / gif.name)
    else:
        require(count == verdict["terminal_passed"], "Interpolation terminal count differs")
        gif = directory / "goal.gif"
        require(file_hash(gif) == evidence["actual_training_gif"]["sha256"], "Interpolation media differs")
        data["gif_sha256"] = file_hash(gif)
        data["integer_finiteness_diagnostic"], trained = replay(request, task, saved, terminal)
        if arm == CANDIDATE and media:
            label = "patch" if task_id == "interp_img_patch_shift" else "saddle"
            shutil.copyfile(gif, media / f"{label}-training.gif")
    return task_id, arm, bounds, common, data, trained


@reproducible_execution
def compare(runs, media):
    tasks, models = {}, {}
    for envelope in sorted((runs / CAMPAIGN).glob("*/request.json")):
        task, arm, bounds, common, data, trained = summarize(envelope, runs, media)
        summary = tasks.setdefault(task, dict(bounds=bounds, matched=common, arms={}))
        require(summary["matched"] == common, f"Mismatch for {task}: {[k for k in common if common[k] != summary['matched'][k]]}")
        require(arm not in summary["arms"], "Duplicate attempt")
        summary["arms"][arm] = data
        if trained is not None:
            models.setdefault(task, {})[arm] = trained
    require(set(tasks) == set(BUDGET) and all(set(s["arms"]) == set(MODES) for s in tasks.values()), "Missing cells")
    for task, trained in models.items():
        tasks[task]["cross_graph_scores"] = cross_graph_scores(task, trained, tasks[task]["arms"])
    return dict(campaign=CAMPAIGN, source_commit=SOURCE, qualification_input=False, tasks=tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--media", type=Path, help="export candidate actual-training GIFs to this directory")
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, args.media), indent=2, sort_keys=True, allow_nan=False))
