"""Collect the verified adopt measurements of 2026-10-16 into results.json.

  python reports/forge/adopt-20261016/collect.py [--state /ml2/hypergan/adopt-20261016]

Reads the drained Forge queues of the six adopt studies (plan, enqueue, drain
--no-publish on source 1e02f185, seed 0) and writes one compact, source-bound
diagnostic record per measured cell. Only cells an independent verifier marked
publishable are imported (PUBLISHABLE below); the collector refuses a queue
whose measured set differs. Nothing trains, nothing is regraded: every status is
the queue's own independent grade (graded-result.json == terminal.json grading).
Raw receipts, logs and checkpoints stay in the state directory; their sha256 are
recorded here. These records never enter qualification or family pins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("results.json")
SOURCE_COMMIT = "1e02f18537b89be13c511515f2716de2de0dc800"
SOURCE_DIGEST = "862b24b12df86f526f1b5a2efcf4deea64edca5b86f1052a9791da948f6da0ed"
MEASURED_VIEW = "reports/forge/adopt-20261016/measured-view-draft10.json"
RECEIPTS = ("request.json", "result.json", "graded-result.json", "terminal.json", "run.log")

# group -> (board family id, candidate id, queue directory relative to --state, label)
GROUPS = {
    "k3p": ("k3p", "recipe-prior-refactor-k3p-v2", "k3p/queue",
            "Adopt: k3p-v2, no LR horizon cap; ring16 6,400; source 1e02f185; seed 0"),
    "ka2": ("ka2", "recipe-prior-refactor-ka2-v2", "ka2/queue",
            "Adopt: ka2-v2, no LR horizon cap; ring16 6,400; source 1e02f185; seed 0"),
    "r1r2": ("r1r2", "recipe-prior-refactor-r1r2-v1", "r1r2-ganv3/r1r2/queue",
             "Adopt: selected recipe unchanged; ring16 6,400; source 1e02f185; seed 0"),
    "gan-v3": ("release07-gan-v3", "recipe-prior-refactor-release07-gan-v3-v1", "r1r2-ganv3/gan-v3/queue",
               "Adopt: selected recipe unchanged; ring16 6,400; source 1e02f185; seed 0"),
    "bcap": ("bcap-pure", "recipe-prior-refactor-bcap-pure-v1", "bcap-bispf/bcap/queue",
             "Adopt: selected recipe unchanged, partial (campaign paused); ring16 6,400; source 1e02f185; seed 0"),
    "bispf": ("bispf", "bispf-v1", "bcap-bispf/bispf/queue",
              "Adopt: selected recipe unchanged, partial (campaign paused); ring16 6,400; source 1e02f185; seed 0"),
}
TIER1 = ["gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition_6400",
         "five_word_joint_smoke"]
PARTIAL = TIER1 + ["gaussian1d_stability", "five_word_joint_hold"]
FULL = None  # every counted test through Tier 2 of the measured view
DIAGNOSTIC = "clockfree_audit_measurement_v1"
# Verifier verdicts (workflow check step): every measured counted cell is publishable;
# Tier 3 only as not run (tier veto); the clock-free audit is a diagnostic record.
PUBLISHABLE = {"k3p": FULL, "ka2": FULL, "r1r2": FULL, "gan-v3": FULL, "bcap": PARTIAL, "bispf": PARTIAL}
TIER_VETO = ("ring_hold", "ring_extension")
# Verifier corrections carried with the cells.
NOTES = {
    ("r1r2", "ring16_acquisition_6400"): "Thin margin: final spread ratio 0.157 against 0.15; stable only from update "
                                          "6,334 of 6,400 (exactly the 5 required final checks), 69/384 checks passing.",
    ("ka2", "five_word_joint_hold"): "The v1 PASS it is sometimes compared with was a diagnostic run on older source "
                                     "a5eafc73; context only, not a measured regression.",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def summary(evaluator):
    """Compact evaluator summary: scalar fields, convergence and final metric checks."""
    result = {k: v for k, v in evaluator.items() if _scalar(v)}
    for key in ("confirmed_steps",):
        if key in evaluator:
            result[key] = evaluator[key]
    if isinstance(evaluator.get("convergence"), dict):
        result["convergence"] = {k: v for k, v in evaluator["convergence"].items() if _scalar(v)}
    if isinstance(evaluator.get("metrics"), list):
        result["metric_checks"] = [{k: m.get(k) for k in ("metric", "op", "threshold", "value", "status")}
                                   for m in evaluator["metrics"] if isinstance(m, dict)]
    return result


def collect_group(state, group):
    family, candidate, queue, label = GROUPS[group]
    queue = Path(state) / queue
    campaign = f"adopt-20261016-{group}-v1"
    records = []
    for directory in sorted((queue / campaign).iterdir()):
        graded_path = directory / "graded-result.json"
        if not graded_path.is_file():
            continue
        request, graded, terminal = read(directory / "request.json"), read(graded_path), read(directory / "terminal.json")
        if terminal["attempt_status"] != "completed" or terminal["grading"] != graded:
            raise ValueError(f"{directory}: attempt incomplete or its grade differs from terminal.json")
        body = request["request"]
        if (body["candidate"]["id"] != candidate or body["source"]["origin_commit"] != SOURCE_COMMIT
                or body["source"]["digest"] != SOURCE_DIGEST or graded["source_digest"] != SOURCE_DIGEST
                or request["job"]["science"]["seed"] != 0 or body["view"]["revision"] != 10):
            raise ValueError(f"{directory}: not the frozen adopt source, candidate, seed 0 and measured view")
        for task in request["job"]["task_ids"]:
            grade = graded["grades"][task]
            status = grade["gate_status"]
            if status not in {"PASS", "FAIL"}:
                raise ValueError(f"{directory}: {task} has no PASS/FAIL grade ({status})")
            execution = body["tasks"][task]["execution"]
            steps = execution.get("steps")
            wallclock = terminal.get("wallclock") or {}
            if wallclock.get("decision") is not None:
                raise ValueError(f"{directory}: wall-clock decision recorded; such a cell is NOT MEASURED")
            record = {
                "group": group, "family": family, "candidate_id": candidate,
                "candidate_revision": body["candidate_revision"], "task": task, "status": status,
                "diagnostic_only": task == DIAGNOSTIC, "attempt_id": directory.name, "campaign": campaign,
                "seed": 0, "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST,
                "view": {"id": body["view"]["id"], "revision": body["view"]["revision"],
                         "policy_fingerprint": body["policy_fingerprint"]},
                "declared_steps": steps, "elapsed_seconds": round(terminal["elapsed_seconds"], 1),
                "wallclock": {k: wallclock.get(k) for k in ("declared_limit", "effective_limit", "decision")},
                "evaluator": summary(grade.get("evaluator_result") or {}),
                "reasons": grade.get("reasons"),
                "final_metrics": {k: grade.get("metrics", {}).get(k) for k in ("precision", "mass_tv")
                                  if isinstance(grade.get("metrics", {}).get(k), (int, float))
                                  and task in {"grid100", "rotated100", "staggered100"}},
                "receipt_sha256": {name: sha256(directory / name) for name in RECEIPTS if (directory / name).is_file()},
            }
            if (group, task) in NOTES:
                record["note"] = NOTES[(group, task)]
            records.append(record)
    measured = {r["task"] for r in records if not r["diagnostic_only"]}
    view = read(ROOT / MEASURED_VIEW)
    counted = [a for a in view["assignments"] if a["importance"] == "required"]
    expected = PUBLISHABLE[group]
    if expected is None:
        expected = [a["task"] for a in counted if a["qualification_tier"] in (1, 2)]
    if measured != set(expected):
        raise ValueError(f"{group}: measured {sorted(measured)} differ from the verified set {sorted(expected)}")
    status = read(queue / "status.json")
    submission = next(iter(status["submissions"].values()))
    tasks = {r["task"]: r["status"] for r in records if not r["diagnostic_only"]}
    unmeasured = {}
    for assignment in counted:
        name = assignment["task"]
        if name in tasks:
            continue
        if PUBLISHABLE[group] is None and name in TIER_VETO:
            unmeasured[name] = {"status": "BLOCKED", "block_kind": "tier_veto",
                                "reason": "complete_current_tier released no Tier 3 work after Tier 2 required failures"}
        else:
            unmeasured[name] = {"status": "NOT_RUN",
                                "reason": "campaign paused after five_word_joint_hold; not run on this source"}
    requirements = {str(t): sorted(a["task"] for a in counted if a["qualification_tier"] == t) for t in (1, 2, 3)}
    row = {"family": family, "group": group, "candidate_id": candidate, "label": label,
           "tasks": {**tasks, **{name: value["status"] for name, value in unmeasured.items()}},
           "requirements": requirements, "unmeasured": unmeasured,
           "queue": {"submission_status": submission["status"], "lifecycle": submission.get("lifecycle"),
                     "paused": any(c.get("paused") for c in status["campaigns"].values()),
                     "spent_seconds": round(sum(c["spent_seconds"] for c in status["campaigns"].values()), 1)}}
    return records, row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("/ml2/hypergan/adopt-20261016"))
    args = parser.parse_args(argv)
    records, rows = [], []
    for group in GROUPS:
        found, row = collect_group(args.state, group)
        records += found
        rows.append(row)
    value = {
        "schema": "adopt_20261016_measurements_v1",
        "qualification_input": False, "qualification_reuse": False, "formulations_modified": False,
        "recipe_revisions": "configs/forge/selections/family-revisions-20261016.json",
        "source": {"origin_commit": SOURCE_COMMIT, "digest": SOURCE_DIGEST},
        "measured_view": {"path": MEASURED_VIEW, "id": "discriminator_stability", "revision": 10,
                          "sha256": sha256(ROOT / MEASURED_VIEW),
                          "note": "Unmerged adopt-branch draft: revision 9 with ring16_acquisition_6400 in place of "
                                  "ring16_acquisition. Current view revision 12 counts the same tests minus develop's "
                                  "holds and the ring_hold phase."},
        "seed": 0,
        "readme": "reports/forge/adopt-20261016/README.md",
        "diagnostic_rows": rows,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(value, indent=1, sort_keys=False) + "\n")
    print(json.dumps({"records": len(records), "rows": {r["group"]: {
        "pass": sum(v == "PASS" for v in r["tasks"].values()), "fail": sum(v == "FAIL" for v in r["tasks"].values()),
        "blocked": sum(v == "BLOCKED" for v in r["tasks"].values()),
        "not_run": sum(v == "NOT_RUN" for v in r["tasks"].values())} for r in rows}}, indent=1))


if __name__ == "__main__":
    main()
