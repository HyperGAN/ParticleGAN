"""Export exact saved Forge evidence and the single current research board.

No model is constructed, no training/sampling is performed, and archived grades
are copied instead of recomputed. Missing full receipts remain INCOMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def export(queue, campaigns):
    runs = []
    for campaign in campaigns:
        for path in sorted((queue / campaign).glob("*/request.json")):
            resolved = read(path)
            request, job = resolved["request"], resolved["job"]
            if not str(job["task_id"]).startswith("ddgan_repair_"):
                continue
            directory = path.parent
            raw_path, adapter_path = directory / "api/receipt.json", directory / "adapter-receipt.json"
            terminal_path = directory / "terminal.json"
            terminal = read(terminal_path) if terminal_path.exists() else {}
            row = dict(candidate=request["candidate"]["id"], task=job["task_id"],
                attempt=directory.name, campaign=campaign,
                request_id=resolved["request_id"], candidate_revision=request["candidate_revision"],
                source=request["source"], task_contract=request["tasks"][job["task_id"]],
                protocol_seed=0, qualification_input=False,
                attempt_status=terminal.get("attempt_status", "running"),
                attempt_reason=terminal.get("reason"),
                supervised_elapsed_seconds=terminal.get("elapsed_seconds"),
                reservation_seconds=job["budget_seconds"],
                terminal_sha256=sha(terminal_path) if terminal_path.exists() else None,
                artifact_directory=str(directory), completed_updates=None,
                verdict="INCOMPLETE", terminal_passes=None, final_metrics={}, failed_bounds=[])
            # The full source file map is already content-addressed in Forge;
            # retain compact manifest identities here, not another bulk copy.
            row["source"] = {key: value for key, value in row["source"].items() if key != "files"}
            # Forge adds candidate-specific resolved ownership to each task;
            # it is not part of the shared authored task contract.
            ownership = row["task_contract"].pop("field_ownership", None)
            row["field_ownership_sha256"] = value_sha(ownership)
            row["task_contract"].pop("preflight_blockers", None)
            row["task_contract_sha256"] = value_sha(row["task_contract"])
            log = directory / "run.log"
            if log.exists():
                row["execution_log_sha256"] = sha(log)
                rows = [json.loads(line) for line in log.read_text().splitlines()
                        if line.startswith('{"case"')]
                row["last_observed_update"] = rows[-1]["step"] if rows else None
            if raw_path.exists():
                raw = read(raw_path)
                complete = raw.get("default_protocol_complete", False) and raw.get("status") == "COMPLETE"
                scheduled = raw["protocol"]["metric_evaluation_steps"]
                checks = [obs for obs in raw["observations"] if obs["step"] > 0 and obs["step"] in scheduled]
                row.update(completed_updates=raw["completed_updates"],
                    api_execution_status=raw.get("status"),
                    verdict=raw["verdict"] if complete else "INCOMPLETE",
                    full_protocol_complete=complete, terminal_passes=sum(obs["passed"] for obs in checks[-5:]),
                    scheduled_checks=len(checks), final_metrics=raw["observations"][-1]["metrics"],
                    failed_bounds=raw["failed_bounds"], recipe=raw["recipe"],
                    reproducibility=raw["reproducibility"], runtime=raw["runtime"],
                    raw_receipt_sha256=sha(raw_path), elapsed_seconds=raw["elapsed_seconds"],
                    artifact_hashes=raw["artifacts"])
                artifact_root = (directory / "api").resolve()
                for relative, recorded in raw["artifacts"].items():
                    artifact = (artifact_root / relative).resolve()
                    if (not artifact.is_relative_to(artifact_root) or not artifact.is_file()
                            or sha(artifact) != recorded["sha256"]
                            or artifact.stat().st_size != recorded["bytes"]):
                        raise ValueError(f"saved API artifact differs: {artifact}")
                gif = directory / "api/goal.gif"
                if gif.exists():
                    if sha(gif) != raw["artifacts"]["goal.gif"]["sha256"]:
                        raise ValueError("actual-training GIF differs from saved receipt")
                    destination = REPORT / "media" / f'{row["candidate"]}-{job["task_id"]}.gif'
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(gif, destination)
                    row["gif"] = str(destination.relative_to(ROOT))
                if adapter_path.exists():
                    adapter = read(adapter_path)
                    row.update(adapter_receipt_sha256=sha(adapter_path),
                        guards=adapter["evidence"]["guards"],
                        prior_policy=adapter["evidence"]["prior_policy"],
                        initial_models=adapter["evidence"]["initial_models"],
                        batch_sequence_sha256=adapter["evidence"]["batch_sequence_sha256"])
                graded = directory / "graded-result.json"
                if graded.exists():
                    grade = read(graded)["grades"][job["task_id"]]
                    row["forge_grade"] = {key: value for key, value in grade.items() if key != "metrics"}
                    evaluator = row["forge_grade"].get("evaluator_result", {})
                    evaluator.pop("metrics", None)  # Same bounds/values are stored once above.
                    row["graded_receipt_sha256"] = sha(graded)
                    if complete:
                        row["verdict"] = row["forge_grade"]["gate_status"]
                elif row["verdict"] == "PASS":
                    row["verdict"] = "INCOMPLETE"
            runs.append(row)
    # Check the actual task contract and consumed clean/timestep sequence agree
    # across all recipe arms; timeout-only variants compare as their own tasks.
    for task in {row["task"] for row in runs}:
        if len({row["task_contract_sha256"] for row in runs if row["task"] == task}) > 1:
            raise ValueError(f"shared task contract changed across arms: {task}")
        measured = [row for row in runs if row["task"] == task and row.get("full_protocol_complete")]
        for name in ("initial_models", "batch_sequence_sha256"):
            if len({json.dumps(row.get(name), sort_keys=True) for row in measured}) > 1:
                raise ValueError(f"unmatched {name} for {task}")
    for candidate in {row["candidate"] for row in runs}:
        recipes = [{key: value for key, value in row["recipe"].items() if key != "total_steps"}
                   for row in runs if row["candidate"] == candidate and row.get("recipe")]
        if len({value_sha(recipe) for recipe in recipes}) > 1:
            raise ValueError(f"candidate is not one global recipe: {candidate}")
    contracts = {row["task"]: row.pop("task_contract") for row in runs}
    result = dict(schema="ddgan_conditional_repair_publication_v1", qualification_input=False,
        goal="ddgan_conditional_repair", seed=0,
        baseline="bispf-optimized-quintic-v1", campaigns=campaigns,
        task_contracts=contracts, runs=runs,
        paid_supervised_seconds=sum(row.get("supervised_elapsed_seconds") or 0. for row in runs),
        total_reserved_attempt_seconds=sum(row["reservation_seconds"] for row in runs),
        cost_scope="Summed supervised attempt cost includes startup, actual updates, evaluation, media and independent grading; timed-out attempts are included. GPU0 was shared; no speed comparison.")
    (REPORT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    board = ROOT / "reports/forge/leaderboards/ddgan_conditional_repair.md"
    board.parent.mkdir(exist_ok=True)
    active = {item["task"] for item in read(ROOT / "configs/forge/views/ddgan-conditional-repair-v1.json")["assignments"]}
    summary = []
    for candidate in sorted({row["candidate"] for row in runs}):
        rows = [row for row in runs if row["candidate"] == candidate]
        current = {row["task"]: row for row in rows if row["task"] in active}
        passed = sum(row["verdict"] == "PASS" for row in current.values())
        failed = sum(row["verdict"] == "FAIL" for row in current.values())
        unknown = len(active) - passed - failed
        summary.append((candidate, passed, failed, unknown,
                        sum(row["verdict"] == "INCOMPLETE" for row in rows)))
    summary.sort(key=lambda row: (-row[1], row[3], row[0]))
    lines = ["# DDGAN conditional repair: current research leaderboard", "",
        "GPU 0 shared-load diagnostic only; no ordinary qualification or speed ranking. "
        "All runs use protocol seed 0, clean live four-step sampling and unchanged scientific gates.", "",
        "One global recipe per row; current tasks are Gaussian and the full-budget bimodal wall-allowance variant. "
        "A missing full result remains unknown. Passing marginal subgates do not count as task passes.", "",
        "| Global recipe | Current PASS / 2 | Current FAIL | Unknown | Retained incomplete attempts |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for candidate, passed, failed, unknown, incomplete in summary:
        lines.append(f"| `{candidate}` | {passed}/2 | {failed} | {unknown} | {incomplete} |")
    lines += ["", "All retained attempts and exact endpoint metrics:", "",
        "| Recipe | Task | Verdict | Updates | Last 5 passes | Final CDF | Worst conditional CDF | GIF |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in runs:
        metrics = row["final_metrics"]
        final = f'{metrics["final_projection_cdf"]:.4f}' if "final_projection_cdf" in metrics else "—"
        worst = f'{metrics["worst_transition_cdf"]:.4f}' if "worst_transition_cdf" in metrics else "—"
        gif = f'[actual training](../../{Path(row["gif"]).relative_to("reports")})' if row.get("gif") else "—"
        updates = row["completed_updates"]
        displayed_updates = f"{updates:,}" if updates is not None else f'>={row["last_observed_update"]:,}'
        terminal_passes = f'{row["terminal_passes"]}/5' if row.get("full_protocol_complete") else "—"
        lines.append(f'| `{row["candidate"]}` | `{row["task"]}` | **{row["verdict"]}** | '
            f'{displayed_updates}/{contracts[row["task"]]["execution"]["steps"]:,} | '
            f'{terminal_passes} | {final} | {worst} | {gif} |')
    lines += ["", "CDF bounds: each final and conditional panel <=0.12. A PASS also requires all mean, "
        "covariance and bimodal mode/local-spread bounds at every final five scheduled checks. "
        "[Full failed bounds, provenance and exact final metrics](../ddgan-conditional-repair/results.json).", ""]
    board.write_text("\n".join(lines))
    print(json.dumps(dict(runs=len(runs), outcomes=[(r["candidate"], r["task"], r["verdict"]) for r in runs])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--campaign", action="append", required=True)
    args = parser.parse_args()
    export(args.queue.resolve(), args.campaign)
