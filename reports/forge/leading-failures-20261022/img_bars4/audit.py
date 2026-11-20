"""Verify archived public-API observations; no construction, sampling or training.

Run from any checkout: python PATH/audit.py --readout ORIGINAL --output result.json
The required original hash is taken from the committed quality archive receipt.
This script checks aggregate consistency, not qualification or pixel reconstruction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

TASK = "img_bars4"
SOURCE = "d9d72d53761c898bce20102e1995fc3d899c1447"
CANDIDATE = "bispf-optimized-quintic-v1"
REVISION = "8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def"
ROOT = Path(__file__).resolve().parents[4]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(path):
    publication = load(ROOT / "reports/forge/bispf-optimized/quality.json")
    archive = publication["raw_readout_archive"]
    raw = path.read_bytes()
    require(len(raw) == archive["bytes"] and sha(raw) == archive["sha256"], "Original readout bytes differ")
    original = json.loads(raw)
    require(original["candidate_id"] == CANDIDATE and original["candidate_revision"] == REVISION, "Wrong candidate")
    matches = [r for r in original["task_results"] if r["task_id"] == TASK]
    require(len(matches) == 1, "Missing/duplicate task")
    row = matches[0]
    attempt = row["_attempt_id"]
    compact_path = ROOT / "reports/forge/technique-receipts" / (attempt + ".json")
    compact = load(compact_path)
    require(compact["candidate_id"] == CANDIDATE and compact["candidate_revision"] == REVISION, "Wrong compact candidate")
    require(compact["provenance"]["source_origin_commit"] == SOURCE, "Wrong executed source")
    projected = next(r for r in compact["task_results"] if r["task_id"] == TASK)
    require(all(row["metrics"][k] == v for k, v in projected["metrics"].items()), "Endpoint projection changed")
    card_path = ROOT / "configs/forge/tasks" / (TASK + ".json")
    card = load(card_path)
    frozen_card_bytes = subprocess.check_output(["git", "show", SOURCE + ":" + str(card_path.relative_to(ROOT))], cwd=ROOT)
    require(card_path.read_bytes() == frozen_card_bytes, "Current task differs from executed declaration")
    evidence = row["evidence"]
    n = row["recipe"]["num_particles"]
    measurement = card["evaluation"]["measurement"]
    mode_min = math.ceil(n * measurement["min_mode_fraction"])
    hq_min = math.ceil(n * measurement["hq_min"])
    observations = evidence["observations"]
    require([r["step"] for r in observations] == list(range(25, 601, 25)), "Wrong observation cadence")
    require(row["execution_path"] == "public_trainer" and row["rng"]["seed"] == 0, "Wrong public API/seed")
    require(row["gate_status"] == projected["gate_status"] == "FAIL", "Original FAIL changed")
    guards = evidence["guards"]
    require(guards["all_finite"] and guards["unintended_rng_deviations"] == 0, "Execution guard failed")
    require(guards["optimizer_updates"] == dict(discriminator=600, generator=600, prior=600), "Wrong update budget")
    require(evidence["sampling_law"] == "enumerated_prior_without_output_noise" and evidence["eval_output_noise"] == "clean", "Wrong sampling")
    inventory = load(ROOT / "reports/forge/technique-inventory.json")
    leader = next(r for r in inventory["rows"] if r.get("candidate_id") == CANDIDATE)
    require(leader["candidate_revision"] == REVISION and leader["bindings"]["source_origin_commit"] == SOURCE, "Selected leader changed")
    contract_id = leader["bindings"]["task_contracts"][TASK]
    contract = inventory["task_contracts"][contract_id]
    require(contract["host_adaptation"]["host_definition"] == evidence["host"]["definition"], "Host definition changed")
    derived = []
    for point in observations:
        counts = []
        for field in ("mode_fractions", "quality_mode_fractions"):
            fractions = point[field]
            require(len(fractions) == measurement["modes"], "Wrong fraction dimensions")
            require(all(0 <= f <= 1 and abs(n*f-round(n*f)) < 1e-10 for f in fractions), "Nonintegral particle counts")
            counts.append([round(n*f) for f in fractions])
        assigned, clean = counts
        require(sum(assigned) == n and all(a >= q for a, q in zip(assigned, clean)), "Invalid clean/assigned partition")
        require(point["hq"] == sum(clean)/n, "HQ does not equal clean fraction")
        require(point["modes"] == sum(q >= mode_min for q in clean), "Mode count differs")
        tv = sum(abs(a/n-1/len(assigned)) for a in assigned)/2
        require(point["distribution_tv"] == tv, "Distribution TV differs")
        passed = point["modes"] >= measurement["modes"] and point["hq"] >= measurement["hq_min"]
        derived.append(dict(step=point["step"], assigned=assigned, clean=clean, modes=point["modes"], clean_total=sum(clean), passed=passed))
    passing = [p["step"] for p in derived if p["passed"]]
    suffix = 0
    for point in reversed(derived):
        if not point["passed"]:
            break
        suffix += 1
    convergence = projected["evaluator_summary"]["convergence"]
    require(len(passing) == convergence["passing_observations"] and suffix == convergence["passing_suffix"], "Archived sustained result differs")
    require(row["metrics"] == observations[-1], "Endpoint is not final observation")
    media_path = ROOT / "reports/forge/bispf-optimized/quality-media" / (TASK + ".json")
    media = load(media_path)
    gif_path = media_path.with_suffix(".gif")
    require(sha(gif_path.read_bytes()) == media["gif_sha256"], "Published actual-training GIF changed")
    require(media["recorded_grade"] == "FAIL" and media["optimizer_updates_added"] == media["sampling_draws_added"] == 0, "Media execution changed")
    require(list(media["source_inputs"].values()) == [evidence["saved_observer_outputs"]["sha256"]], "Media belongs to another output tensor")
    source_files = ["benchmarks/transfer_suite/image_tasks.py", "experiments/forge/adapters.py", "particlegan/recipes.py", "particlegan/training.py", "particlegan/bispf.py", "particlegan/optim/dualnorm.py"]
    source_hashes = {name: sha(subprocess.check_output(["git", "show", SOURCE + ":" + name], cwd=ROOT)) for name in source_files}
    milestones = {25, 300, 350, 375, 400, 500, 550, 575, 600}
    return dict(schema_version=1, task_id=TASK, scope="zero_training_archived_observation_audit", audit_status="PASS", original_gate="FAIL", qualification_input=False,
        training_updates_added=0, sampling_draws_added=0, original_readout=dict(original_path=archive["path"], inspected_path=str(path), sha256=sha(raw), bytes=len(raw)),
        candidate=dict(id=CANDIDATE, revision=REVISION, executed_commit=SOURCE, source_digest=compact["provenance"]["source_digest"], attempt_id=attempt, canonical_result_hash=compact["provenance"]["canonical_result_hash"]),
        task_contract=dict(sha256=contract_id, contract=contract), effective_recipe=row["recipe"], prior=row["prior"], prior_policy=evidence["prior_policy"], initialization=row["initialization"],
        streams=dict(seed=row["rng"]["seed"], version=row["rng"]["version"], checkpoint_stream_state_sha256=evidence["provenance_checkpoint"]["named_stream_state_sha256"], checkpoint_sha256=evidence["provenance_checkpoint"]["sha256"], unintended_deviations=guards["unintended_rng_deviations"]),
        metrics=row["metrics"], checks=dict(observations_checked=len(observations), minimum_clean_per_mode=mode_min, minimum_total_clean=hq_min, passing_observations=len(passing), passing_suffix=suffix, required_suffix=measurement["minimum_stable_checks"], maximum_modes=max(p["modes"] for p in derived), maximum_clean_total=max(p["clean_total"] for p in derived), first_all_modes_step=next((p["step"] for p in derived if p["modes"] == measurement["modes"]), None), final_assigned_counts=derived[-1]["assigned"], final_clean_counts=derived[-1]["clean"]),
        selected_milestones=[p for p in derived if p["step"] in milestones], execution_guards=guards,
        inputs=dict(compact_receipt=dict(path=str(compact_path.relative_to(ROOT)),sha256=sha(compact_path.read_bytes())), task_declaration=dict(path=str(card_path.relative_to(ROOT)),sha256=sha(card_path.read_bytes())), actual_training_gif=dict(path=str(gif_path.relative_to(ROOT)),sha256=media["gif_sha256"], saved_samples_sha256=evidence["saved_observer_outputs"]["sha256"]), frozen_source_files=source_hashes),
        limits=["Counts derived from original API metric fractions, not newly evaluated pixels.", "No full original envelope regrade, training or qualification update.", "Saved image tensor and trained G/D checkpoints were not located; source hashes and receipts do not supply gradient evidence.", "Milestones describe observations every 25 updates; intermediate acquisition/loss is unknown."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.readout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(task=TASK, audit=result["audit_status"], original_gate=result["original_gate"], checks=result["checks"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
