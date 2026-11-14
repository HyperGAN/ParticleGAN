"""ParticleGAN Forge. Read EXPERIMENTATION.md before running new ideas."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contracts import atomic_json, read_json
from .execution_policy import policy as execution_policy


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def queue_location(root: Path, override=None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("PARTICLEGAN_FORGE_QUEUE"):
        return Path(os.environ["PARTICLEGAN_FORGE_QUEUE"]).expanduser().resolve()
    try:
        common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        repository = (root / common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        repository = root
    return repository / "runs/forge"


def _qualification_tier(val: str) -> int:
    try:
        i = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid qualification tier integer: {val!r}")
    if i < 1:
        raise argparse.ArgumentTypeError(f"qualification tier must be >= 1, got {i}")
    return i


def _minimum_tier(val: str) -> int:
    try:
        i = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid tier integer: {val!r}")
    if i < 0:
        raise argparse.ArgumentTypeError(f"tier must be >= 0, got {i}")
    return i


CLI_EPILOG = """Lifecycle command groups:
  1. Orient & Inspect:
     recall, board, techniques, experiments-by-tier, stats, validate, history
  2. Propose & Plan:
     probe, new, study, plan, search, cadence, scaling
  3. Execute & Monitor:
     enqueue, run, drain, queue, logs, inventory, pause, resume, cancel, retry
  4. Conclude & Maintain:
     readout, compile, abandon, supersede, promotion, retier, test-dispositions,
     calibrate, calibration-preflight, calibration-lane, artifacts

Note: --view and --goal are accepted interchangeably across subcommands.
"""


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="Forge checkout containing declarations and durable reports")
    p.add_argument("--queue-root", type=Path, help="shared repo-local operational queue; default common Git repository runs/forge")
    commands = p.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser("artifacts", help="verify/hydrate byte-exact originals; no training or regrading")
    artifact_stages = artifacts.add_subparsers(dest="stage", required=True)
    for stage in ("inspect", "hydrate"):
        artifact = artifact_stages.add_parser(stage)
        artifact.add_argument("manifest", type=Path, help="committed Forge archive card")
        artifact.add_argument("--member", action="append", default=[], help="exact archive member; repeat to select originals")
        artifact.add_argument("--mirror", action="append", type=Path, default=[], help="mounted content-addressed mirror directory")
        artifact.add_argument("--locations", type=Path, help="local SHA-256 location/retention configuration")
        if stage == "hydrate":
            artifact.add_argument("--destination", type=Path, required=True, help="new isolated directory (must not exist)")
    original = artifact_stages.add_parser("git", help="hydrate an exact commit/path/blob original already available in Git")
    original.add_argument("--commit", required=True)
    original.add_argument("--path", required=True)
    original.add_argument("--blob", required=True)
    original.add_argument("--sha256", required=True)
    original.add_argument("--destination", type=Path, required=True)
    probe = commands.add_parser("probe", help="zero-ceremony preflight/formulation sandbox probe for a candidate and task; no study required")
    probe.add_argument("candidate", help="candidate or configuration id")
    probe.add_argument("--task", required=True, help="task id to probe against")
    probe.add_argument("--view", "--goal", dest="view", default="discriminator_stability")
    probe.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    probe.add_argument("--override", action="append", default=[], metavar="KEY=JSON_VALUE",
                       help="ephemeral Recipe override for exploratory probing; repeat as needed")
    new = commands.add_parser("new", help="scaffold one small idea declaration")
    new.add_argument("--id", required=True)
    new.add_argument("--parent", default="k3p")
    new.add_argument("--goal", "--view", dest="goal", default="discriminator_stability")
    new.add_argument("--hypothesis")
    new.add_argument("--study-id", help="companion study id (default CANDIDATE-study)")
    study = commands.add_parser("study", help="declare a new question using an existing candidate; no training")
    study_stages = study.add_subparsers(dest="stage", required=True)
    study_new = study_stages.add_parser("new")
    study_new.add_argument("--id", required=True)
    study_new.add_argument("--candidate", required=True)
    study_new.add_argument("--control", required=True)
    study_new.add_argument("--view", "--goal", dest="view", default="discriminator_stability")
    study_new.add_argument("--hypothesis")
    study_new.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name in ("plan", "enqueue", "run"):
        c = commands.add_parser(name, help={"plan": "show exact tasks/reuse/cost without writing or training", "enqueue": "freeze and submit; do not launch", "run": "enqueue and drain through the declared tier cap"}[name])
        c.add_argument("candidate")
        c.add_argument("--view", "--goal", dest="view")
        c.add_argument("--study", help="explicit research study id or JSON path")
        c.add_argument("--through-tier", type=_qualification_tier, default=None,
                       help="maximum qualification tier (>= 1)")
        c.add_argument("--device", choices=("cpu", "cuda"), default=None, help="compute cohort (default CUDA; run infers it from --gpus)")
        c.add_argument("--cuda-model", help="required CUDA model on heterogeneous machines")
        if name == "plan":
            c.add_argument("--all-tiers", action="store_true", help="include full details for tasks outside the requested tier cap")
            c.add_argument("--show-boundaries", action="store_true", help="include each effective field's task, technique, hyperparameter or protocol owner")
        if name != "plan":
            c.add_argument("--campaign", type=Path, help="legacy campaign file; a study owns its campaign")
        if name == "run":
            c.add_argument("--gpus", default="cpu", help="physical GPU indices, or cpu")
    d = commands.add_parser("drain", help="execute eligible work with a single queue coordinator")
    d.add_argument("--gpus", required=True, help="comma separated physical GPU indices, or cpu")
    d.add_argument("--workers-per-gpu", type=int, default=1)
    d.add_argument("--allow-sharing", action="store_true")
    d.add_argument("--watch", action="store_true")
    d.add_argument("--no-publish", action="store_true",
                   help="skip the post-completion compile_memory hook (it rewrites reports/forge boards)")
    d.add_argument("--campaign")
    d.add_argument("--goal", "--view", dest="goal")
    d.add_argument("--request", help="drain only this exact submitted request ID")
    d.add_argument("--shared-fleet-fd", type=int, action="append", default=[],
                   help="open external fleet lease descriptor to retain in every worker and child; repeat for each token")
    commands.add_parser("queue", help="show active/blocked requests, costs and log paths")
    commands.add_parser("stats", help="report measured automation costs, reuse, avoided work and execution limits")
    logs = commands.add_parser("logs", help="tail the collector's timestamped event stream")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--lines", type=int, default=40)
    for name in ("candidate", "task", "worker", "gpu", "campaign"):
        logs.add_argument(f"--{name}")
    b = commands.add_parser("board", help="regrade a goal view over compatible evidence; no launches")
    b.add_argument("--goal", "--view", dest="goal", default="discriminator_stability")
    b.add_argument("--json", action="store_true", help="emit full machine-readable board instead of a compact table")
    b.add_argument("--scope", choices=("all", "current", "pinned", "historical", "calibration_diagnostic"), default="all")
    b.add_argument("--candidate", help="filter candidate names by substring")
    b.add_argument("--tier", type=_minimum_tier, help="minimum qualified tier (>= 0)")
    b.add_argument("--outcome", help="filter status or presence of a task gate status")
    b.add_argument("--family", action="append", help="display task/host/source family; repeat for alternatives")
    b.add_argument("--evidence-quality", action="append", help="display receipt provenance, e.g. certified_pinned or imported_recorded; repeat for alternatives")
    techniques = commands.add_parser("techniques", help="regenerate technique rows with full per-tier denominators; no training")
    techniques.add_argument("--goal", "--view", dest="goal", default="discriminator_stability")
    techniques.add_argument("--device", choices=("cpu", "cuda"), help="show one execution cohort (default both)")
    techniques.add_argument("--json", action="store_true")
    techniques.add_argument("--output", type=Path, help="write Markdown and compact JSON using this path prefix")
    inventory = commands.add_parser("inventory", help="run one selected current configuration per family through ordinary Forge gates")
    inventory_stages = inventory.add_subparsers(dest="stage", required=True)
    for stage in ("plan", "enqueue", "run"):
        inv = inventory_stages.add_parser(stage)
        inv.add_argument("--view", "--goal", dest="view", default="discriminator_stability")
        inv.add_argument("--through-tier", type=_qualification_tier, default=3,
                         help="maximum qualification tier (>= 1)")
        inv.add_argument("--device", choices=("cpu", "cuda"), default=None)
        inv.add_argument("--cuda-model")
        inv.add_argument("--campaign", type=Path, default=Path("configs/forge/campaigns/technique-inventory-word-split-v1.json"))
        if stage == "run":
            inv.add_argument("--gpus", default="0,1", help="physical GPU indices, or cpu")
    scaling = commands.add_parser("scaling", help="bounded DCGAN mock-data speed and memory diagnostics; no qualification")
    scaling_stages = scaling.add_subparsers(dest="stage", required=True)
    scaling_register = scaling_stages.add_parser("register", help="freeze exact selected recipes, studies and finite reservation; no training")
    scaling_register.add_argument("registration")
    scaling_register.add_argument("--view", "--goal", dest="view", default="dcgan_performance")
    scaling_register.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    scaling_register.add_argument("--cuda-model")
    scaling_register.add_argument("--family", action="append", help="select a current family; repeat, default all")
    scaling_register.add_argument("--candidate", action="append", default=[], metavar="FAMILY=CANDIDATE",
                                  help="explicit diagnostic candidate override within a selected family; repeat")
    scaling_register.add_argument("--campaign", type=Path,
                                  help="optional immutable shared performance campaign JSON")
    for stage in ("plan", "enqueue"):
        scaling_stages.add_parser(stage).add_argument("registration")
    scaling_publish = scaling_stages.add_parser("publish", help="export frozen queue performance summaries and refresh the current overview; no training")
    scaling_publish.add_argument("registration")
    scaling_publish.add_argument("--allow-partial", action="store_true", help="explicitly publish a snapshot with queued/running cells marked unknown")
    tiers = commands.add_parser("experiments-by-tier", help="review experiment tiers, questions, published results and GIFs; no training")
    tiers.add_argument("--view", "--goal", dest="view", help="show one view; default all current views")
    tiers.add_argument("--json", action="store_true", help="render machine-readable inventory instead of Markdown")
    tiers.add_argument("--output", type=Path, help="write the report here, relative to --root; default print to stdout")
    search = commands.add_parser("search", help="bounded deterministic public Recipe grids; no trainer copies")
    search_stages = search.add_subparsers(dest="stage", required=True)
    compile_search = search_stages.add_parser("compile", help="freeze tagged finite choices and categorical candidate draws; no training")
    compile_search.add_argument("spec", type=Path, help="forge_search_space_v1 JSON declaration")
    compile_search.add_argument("--output", type=Path, required=True, help="immutable compiled manifest")
    for stage in ("plan", "enqueue", "run", "report"):
        trial = search_stages.add_parser(stage)
        trial.add_argument("spec", help="search id or configs/forge/searches JSON path")
        if stage == "run":
            trial.add_argument("--gpus", default=None, help="physical GPU indices, or cpu; must match the fixed spec")
    cadence = commands.add_parser("cadence", help="matched lazy-regularization studies and generated comparisons")
    cadence_stages = cadence.add_subparsers(dest="stage", required=True)
    for stage in ("plan", "enqueue", "run", "report"):
        c = cadence_stages.add_parser(stage)
        c.add_argument("spec", help="cadence study id or JSON path; any supported base technique")
        c.add_argument("--output", type=Path, help="write the generated .md report and compact .json companion")
        c.add_argument("--json", action="store_true", help="include complete per-task comparison details on stdout")
        if stage == "run":
            c.add_argument("--gpus", help="physical GPU indices or cpu; must match the declaration")
    r = commands.add_parser("recall", help="find successes, failures and unknowns before a new idea")
    r.add_argument("--query", default="")
    r.add_argument("--goal", "--view", dest="goal")
    r.add_argument("--limit", type=int, default=8, help="maximum concise prior-art matches (default 8)")
    r.add_argument("--full-provenance", action="store_true", help="include complete source file manifests in recall matches")
    compile_parser = commands.add_parser("compile", help="rebuild deterministic memory and leaderboards without training")
    compile_parser.add_argument("--check", action="store_true", help="read-only freshness check; exit nonzero for stale memory")
    compile_parser.add_argument("--summaries-only", action="store_true", help="refresh recall while preserving published qualification/telemetry snapshots")
    compile_parser.add_argument("--regrade-attempts", action="store_true", help="force full receipt regrading even when published evidence is preserved without local attempts")
    h = commands.add_parser("history", help="classify and import pinned history without training")
    h.add_argument("--check", action="store_true", help="check inventory coverage instead of importing")
    commands.add_parser("validate", help="validate task/view definitions and maintained family reports without training")
    calibration = commands.add_parser("calibrate", help="replay saved calibration evidence; no training")
    calibration.add_argument("--profile", default="initial")
    feasibility = commands.add_parser("calibration-preflight", help="check whether a frozen current profile can meet its criteria; no writes or training")
    feasibility.add_argument("--profile", required=True)
    feasibility.add_argument("--require-feasible", action="store_true", help="exit nonzero for infeasible criteria or unresolved receipt issues")
    lane = commands.add_parser("calibration-lane", help="register selected bounded diagnostics; never qualify a candidate")
    lane_stages = lane.add_subparsers(dest="stage", required=True)
    lane_register = lane_stages.add_parser("register")
    lane_register.add_argument("--contract", type=Path, required=True)
    for stage in ("plan", "enqueue"):
        lane_stages.add_parser(stage).add_argument("registration")
    lane_imports = lane_stages.add_parser("imports", help="print exact saved diagnostic bindings for a new profile; no writes or training")
    lane_imports.add_argument("registration")
    lane_imports.add_argument("--lineage", required=True)
    lane_imports.add_argument("--tasks", nargs="+", required=True)
    promotion = commands.add_parser("promotion", help="register and run one frozen finished-candidate robustness stage")
    stages = promotion.add_subparsers(dest="stage", required=True)
    registration = stages.add_parser("register")
    registration.add_argument("candidate")
    registration.add_argument("--contract", type=Path, required=True)
    for stage in ("plan", "enqueue", "report"):
        stage_parser = stages.add_parser(stage)
        stage_parser.add_argument("registration")
    retier = commands.add_parser("retier", help="change only view policy, archive the previous revision, validate and recompute; never enqueue")
    retier.add_argument("view")
    retier.add_argument("--task", required=True)
    retier.add_argument("--tier", type=_qualification_tier, required=True,
                        help="new qualification tier (>= 1)")
    retier.add_argument("--reason", required=True)
    dispositions = commands.add_parser("test-dispositions", help="check or apply held/removed-from-count test dispositions as archived view revisions; no training")
    dispositions.add_argument("stage", choices=("check", "apply"), help="check writes nothing and exits 1 on drift; apply archives first and is idempotent")
    dispositions.add_argument("--registry", type=Path, required=True, help="forge_test_dispositions_v1 JSON, relative to --root")
    out = commands.add_parser("readout", help="conclude an idea with comparison and recommendation, then compile memory")
    out.add_argument("candidate")
    out.add_argument("--study", help="conclude only this frozen study, preserving other uses of the recipe")
    out.add_argument("--conclusion", required=True)
    out.add_argument("--comparison", required=True)
    out.add_argument("--next-action", required=True)
    cancel = commands.add_parser("cancel", help="detach a request and terminate unneeded child process groups")
    cancel.add_argument("request_id")
    retry = commands.add_parser("retry", help="retry a repaired execution failure, never a scientific failure")
    retry.add_argument("compatibility_key")
    retry.add_argument("--reason", required=True)
    for name in ("abandon", "supersede"):
        disposition = commands.add_parser(name, help=f"{name} a stopped revision while preserving its readout and evidence")
        disposition.add_argument("candidate")
        disposition.add_argument("--reason", required=True)
        if name == "supersede":
            disposition.add_argument("--successor", required=True)
    for name in ("pause", "resume"):
        c = commands.add_parser(name, help=f"{name} campaign claims")
        c.add_argument("campaign")
    return p


def follow_logs(path: Path, args):
    offset = 0
    first = True
    while True:
        if path.exists():
            with path.open() as stream:
                stream.seek(offset)
                lines = []
                while True:
                    previous = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        stream.seek(previous)
                        break
                    lines.append(line)
                offset = stream.tell()
            if first:
                lines = lines[-args.lines:]
                first = False
            for line in lines:
                row = json.loads(line)
                if all(getattr(args, field) is None or str(row.get(field)) == getattr(args, field)
                       for field in ("candidate", "task", "worker", "gpu", "campaign")):
                    print(line, end="", flush=True)
        if not args.follow:
            break
        time.sleep(.2)


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "artifacts":
        from .artifact_resolver import ArtifactError, hydrate_archive, hydrate_git, inspect_archive
        try:
            if args.stage == "git":
                result = hydrate_git(root, commit=args.commit, path=args.path, blob=args.blob,
                                     sha256=args.sha256, destination=args.destination)
            else:
                card = read_json(root / args.manifest)
                options = {"members": args.member, "mirrors": args.mirror, "locations": args.locations}
                result = (inspect_archive(root, card, **options) if args.stage == "inspect"
                          else hydrate_archive(root, card, args.destination, **options))
            emit(result)
            return 0
        except ArtifactError as error:
            emit({"status": error.status, "message": str(error), "training_launched": False,
                  "qualification_changed": False})
            return 2 if error.status == "MISSING" else 3
        except (OSError, ValueError, KeyError, TypeError) as error:
            emit({"status": "INVALID", "message": str(error), "training_launched": False,
                  "qualification_changed": False})
            return 3
    if args.command == "probe":
        from dataclasses import asdict
        from .api import task_formulation_context
        from .boundaries import ownership_receipt, recipe_field_owner
        from .planning import load_idea
        from .techniques import recipe_field_active, technique_signature
        from .views import load_tasks
        idea = dict(load_idea(root, args.candidate))
        tasks = load_tasks(root)
        if args.task not in tasks:
            raise ValueError(f"unknown task: {args.task}")
        task = tasks[args.task]
        overrides = dict(idea.get("recipe_overrides", {}))
        for raw_override in args.override:
            key, sep, val = raw_override.partition("=")
            if not sep or not key:
                raise ValueError(f"probe --override requires KEY=JSON_VALUE, got {raw_override!r}")
            try:
                overrides[key] = json.loads(val)
            except ValueError:
                overrides[key] = val
        idea["recipe_overrides"] = overrides
        context = task_formulation_context(idea, task, root=root)
        resolved_recipe = asdict(context.recipe)
        receipt = ownership_receipt(idea, task, resolved_recipe)
        signature = technique_signature(context.recipe)
        override_Diagnostics = {
            name: {
                "value": resolved_recipe.get(name),
                "owner": recipe_field_owner(name, task),
                "active_on_task": recipe_field_active(name, context.recipe, task=task),
                "status": receipt["recipe_fields"].get(name, {}).get("status"),
            }
            for name in sorted(overrides)
            if name in resolved_recipe
        }
        emit({
            "cohort": "exploratory_probe",
            "qualified": False,
            "training_launched": False,
            "candidate": args.candidate,
            "task": args.task,
            "view": args.view,
            "device": args.device,
            "technique_digest": signature["digest"],
            "active_mechanisms": {k: v for k, v in signature["mechanisms"].items() if v},
            "override_diagnostics": override_Diagnostics,
            "prior_binding": receipt["task_contract"]["prior"]["control_binding"],
        })
        return 0
    if args.command == "calibration-preflight":
        from .calibration_feasibility import preflight
        result = preflight(root, args.profile)
        emit(result)
        return int(args.require_feasible and (result["feasibility"]["status"] != "POSSIBLE" or bool(result["receipt_issues"])))
    if args.command == "experiments-by-tier":
        from .tier_report import build_report, render_markdown, write_report
        report = build_report(root, args.view)
        if args.output is not None:
            path = write_report(report, root, args.output, as_json=args.json)
            emit({"output": str(path), "tasks": report["task_count"], "views": len(report["views"]),
                  "unassigned_tasks": len(report["unassigned_tasks"]), "training_launched": False})
        elif args.json:
            emit(report)
        else:
            print(render_markdown(report, root), end="")
        return 0
    if args.command == "test-dispositions":
        from .dispositions import apply, check
        registry = args.registry if args.registry.is_absolute() else root / args.registry
        result = (check if args.stage == "check" else apply)(root, registry)
        emit(result)
        return int(args.stage == "check" and result["drift"])
    queue_root = queue_location(root, args.queue_root)
    from .queue import Queue, drain
    def publish():
        from .knowledge import compile_memory
        compile_memory(root)
    # The inventory already records every attempt in the live event stream.
    # Compile the full set of boards once after draining this campaign.
    batch_inventory = (args.command in {"run", "drain"}
                       or (args.command in {"inventory", "search", "cadence"} and args.stage == "run"))
    queue = Queue(queue_root, report_root=root / "reports/forge",
                  on_completion=None if batch_inventory or getattr(args, "no_publish", False) else publish)
    command = args.command
    if command == "new":
        from .planning import new_idea
        path = new_idea(root, args.id, args.parent, goal=args.goal, hypothesis=args.hypothesis, study_id=args.study_id)
        from .studies import study_path
        emit({"candidate": str(path), "study": str(study_path(root, args.study_id or f"{args.id}-study")),
              "next": "edit the recipe and study, then plan CANDIDATE --study STUDY", "guide": str(root / "EXPERIMENTATION.md")})
    elif command == "study":
        from .planning import load_idea
        from .studies import scaffold, study_path
        load_idea(root, args.candidate)
        load_idea(root, args.control)
        path = study_path(root, args.id)
        if path.exists():
            raise ValueError("study already exists")
        atomic_json(path, scaffold(args.id, args.candidate, args.control, args.view,
                                  hypothesis=args.hypothesis, execution_backend=args.device))
        emit({"study": str(path), "training_launched": False})
    elif command in {"plan", "enqueue", "run"}:
        from .planning import plan_summary, resolve_idea
        request = resolve_idea(root, args.candidate, view_id=args.view, through_tier=args.through_tier,
                               queue_root=queue_root, freeze_source=command != "plan",
                               execution_backend=args.device or (("cpu" if args.gpus == "cpu" else "cuda") if command == "run" else None),
                               cuda_model=args.cuda_model, study=args.study)
        if command == "plan":
            summary = plan_summary(request, queue.inspect(), include_ownership=args.show_boundaries)
            from .knowledge import freshness
            summary["research_memory"] = freshness(root)
            if not args.all_tiers:
                summary["deferred_tasks"] = [t["task"] for t in summary["tasks"] if not t["permitted_by_tier_cap"]]
                summary["tasks"] = [t for t in summary["tasks"] if t["permitted_by_tier_cap"]]
            emit(summary)
        else:
            if args.study and args.campaign:
                raise ValueError("study owns its campaign; omit --campaign")
            campaign = request["study"]["campaign"] if args.study else read_json(root / (args.campaign or Path("configs/forge/campaigns/smoke.json")))
            entry = queue.submit(request, campaign)
            emit({"request_id": entry["request"]["request_id"], "status": entry["status"],
                  "queue_root": str(queue_root), "logs": str(queue_root / "events.jsonl"),
                  "request": str(queue_root / "queue/requests" / f"{entry['request']['request_id']}.json")})
            if command == "run":
                drain(queue, args.gpus.split(","), campaign=campaign["id"])
                from .knowledge import compile_memory
                compile_memory(root, summaries_only=True)
                emit(queue.inspect()["submissions"][entry["request"]["request_id"]]["status"])
    elif command == "drain":
        drain(queue, args.gpus.split(","), workers_per_gpu=args.workers_per_gpu,
              allow_sharing=args.allow_sharing, watch=args.watch, campaign=args.campaign, goal=args.goal,
              request_filter=args.request, shared_fleet_fds=tuple(args.shared_fleet_fd))
        if not args.no_publish:
            from .knowledge import compile_memory
            compile_memory(root, summaries_only=True)
        emit({"status": "drained", "logs": str(queue_root / "events.jsonl")})
    elif command == "queue":
        state = queue.inspect()
        emit({"queue_root": str(queue_root), "campaigns": state["campaigns"],
              "requests": [{"id": k, "candidate": v["request"]["candidate"]["id"],
                            "execution_policy": execution_policy(v["request"]),
                            "status": v["status"], "reason": v["reason"], "lifecycle": v["lifecycle"]}
                           for k, v in state["submissions"].items()], "logs": str(queue_root / "events.jsonl")})
    elif command == "stats":
        from .telemetry import summarize_automation
        selected_queue = queue_root if args.queue_root or os.environ.get("PARTICLEGAN_FORGE_QUEUE") else None
        emit(summarize_automation(root, selected_queue))
    elif command == "logs":
        follow_logs(queue_root / "events.jsonl", args)
    elif command == "techniques":
        from .technique_board import technique_board, render_markdown, write_report
        if args.output:
            emit(write_report(root, args.goal, output_prefix=args.output, execution_backend=args.device))
        else:
            result = technique_board(root, args.goal, execution_backend=args.device)
            if args.json:
                emit(result)
            else:
                print(render_markdown(result))
    elif command == "inventory":
        from .technique_inventory import plan_inventory, enqueue_inventory, run_inventory
        backend = args.device or ("cpu" if args.stage == "run" and args.gpus == "cpu" else "cuda")
        if args.stage == "run" and backend != ("cpu" if args.gpus == "cpu" else "cuda"):
            raise ValueError("inventory --device must agree with --gpus")
        options = dict(view_id=args.view, through_tier=args.through_tier,
                       execution_backend=backend, cuda_model=args.cuda_model, campaign=args.campaign)
        if args.stage == "plan":
            emit(plan_inventory(root, queue_root, **options))
        elif args.stage == "enqueue":
            emit(enqueue_inventory(root, queue_root, queue=queue, **options))
        else:
            emit(run_inventory(root, queue_root, devices=args.gpus.split(","), queue=queue, **options))
            publish()
    elif command == "scaling":
        from .scaling import register, plan, enqueue
        if args.stage == "register":
            overrides = {}
            for value in args.candidate:
                family, separator, candidate = value.partition("=")
                if not separator or not family or not candidate or family in overrides:
                    raise ValueError("performance --candidate requires distinct FAMILY=CANDIDATE entries")
                overrides[family] = candidate
            emit(register(root, args.registration, view_id=args.view, execution_backend=args.device,
                          cuda_model=args.cuda_model, family_ids=args.family, candidate_overrides=overrides,
                          campaign=read_json(args.campaign) if args.campaign else None))
        elif args.stage == "plan":
            emit(plan(root, queue_root, args.registration, queue=queue))
        elif args.stage == "publish":
            from .scaling_export import publish as publish_scaling
            emit(publish_scaling(root, queue_root, args.registration, allow_partial=args.allow_partial))
        else:
            emit(enqueue(root, queue_root, args.registration, queue=queue))
    elif command == "search":
        from .configuration_search import plan_search, enqueue_search, run_search, report_search
        from .search_space import (MANIFEST_SCHEMA, write_compilation, plan_compilation,
                                   enqueue_compilation, report_compilation, run_compilation)
        if args.stage == "compile":
            emit(write_compilation(root, queue_root, args.spec, args.output))
            return 0
        path = Path(args.spec)
        if path.suffix == ".json" and (root / path).exists() and read_json(root / path).get("schema") == MANIFEST_SCHEMA:
            plan_search, enqueue_search, report_search, run_search = (
                plan_compilation, enqueue_compilation, report_compilation, run_compilation)
        if args.stage == "plan":
            emit(plan_search(root, queue_root, args.spec))
        elif args.stage == "enqueue":
            emit(enqueue_search(root, queue_root, args.spec, queue=queue))
        elif args.stage == "report":
            emit(report_search(root, queue_root, args.spec, queue=queue))
        else:
            emit(run_search(root, queue_root, args.spec, devices=args.gpus.split(",") if args.gpus else None, queue=queue))
            publish()
    elif command == "cadence":
        from .cadence_study import (plan_study, comparison_report, enqueue_study,
                                    run_study, report_study, write_report)
        if args.stage == "plan":
            result = comparison_report(plan_study(root, queue_root, args.spec, queue=queue))
        elif args.stage == "enqueue":
            result = enqueue_study(root, queue_root, args.spec, queue=queue)
        elif args.stage == "report":
            result = report_study(root, queue_root, args.spec, queue=queue)
        else:
            result = run_study(root, queue_root, args.spec, queue=queue,
                               devices=args.gpus.split(",") if args.gpus else None)
        destination = args.output
        if destination is None and args.stage in {"run", "report"}:
            destination = Path("reports/forge/cadence-studies") / f"{result['study_id']}.md"
        if destination:
            output = destination if destination.is_absolute() else root / destination
            result["outputs"] = write_report(result, output)
        if args.json:
            emit(result)
        else:
            emit({**{key: value for key, value in result.items() if key != "lanes"},
                  "lanes": {lane: {"recommendation": details["recommendation"],
                                     "evidence_scope": details["evidence_scope"],
                                     "arms": len(details["trials"])}
                            for lane, details in result["lanes"].items()}})
    elif command in {"compile", "recall", "board", "readout"}:
        from . import knowledge
        if command == "compile":
            if args.check:
                result = knowledge.freshness(root)
                emit(result)
                return 0 if result["fresh"] else 1
            summaries_only = args.summaries_only
            if not summaries_only and not getattr(args, "regrade_attempts", False):
                manifest_path = root / "reports/forge/compilation.json"
                if (manifest_path.is_file()
                        and read_json(manifest_path).get("qualification_refresh", {}).get("mode") == "preserved_published_evidence"
                        and not any((root / "reports/forge/attempts").glob("*/result.json"))):
                    summaries_only = True
            emit(knowledge.compile_memory(root, summaries_only=summaries_only))
        elif command == "recall":
            if args.limit < 1:
                raise ValueError("recall --limit must be positive")
            matches = knowledge.recall(root, query=args.query, goal=args.goal,
                                       full_provenance=args.full_provenance)
            in_flight = knowledge.in_flight_work(root, query=args.query, goal=args.goal)
            emit({"matches_total": len(matches), "shown": min(len(matches), args.limit), "matches": matches[:args.limit],
                  "in_flight": in_flight,
                  "memory": str(root / "reports/forge/EXPERIMENT_MEMORY.md"), "research_memory": knowledge.freshness(root)})
        elif command == "board":
            result = knowledge.board(root, args.goal)
            from .board_filters import filter_rows
            result = filter_rows(result, family=args.family, evidence_quality=args.evidence_quality)
            rows = [r for r in result["rows"] if (args.scope == "all" or r.get("evidence_scope") == args.scope)
                    and (not args.candidate or args.candidate in r.get("candidate_id", ""))
                    and (args.tier is None or (r.get("qualified_tier") or 0) >= args.tier)
                    and (not args.outcome or r.get("status") == args.outcome or r.get("counts", {}).get(args.outcome, 0))]
            if args.json:
                emit({**result, "rows": rows})
            else:
                print("| Candidate | Revision | Scope | Compute | Tier | PASS / total | FAIL | BLOCKED | Wall seconds | Status |")
                print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                for row in rows:
                    counts = row.get("counts", {})
                    seconds = row.get("cost", {}).get("wall_seconds")
                    seconds = f"{seconds:.3f}" if isinstance(seconds, (int, float)) else "unknown"
                    revision = str(row.get("candidate_revision") or "unknown")[:10]
                    name = str(row.get("candidate_id", "unknown")).replace("|", "\\|")
                    runtime = row.get("runtime_cohort", {})
                    backend = runtime.get("execution_backend", "unrecorded")
                    profile = runtime.get("compute_profiles", {}).get(backend, {})
                    compute = f"{backend}: {profile['model']}" if profile.get("model") else backend
                    compute = compute.replace("|", "\\|")
                    print(f"| {name} | {revision} | {row.get('evidence_scope')} | {compute} | {row.get('qualified_tier', '—')} | "
                          f"{counts.get('PASS', 0)}/{sum(counts.values())} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} | {seconds} | "
                          f"{row['status']}{knowledge.held_note(row)} |")
                print(f"\nFull evidence: reports/forge/leaderboards/{args.goal}.json; use --json for raw metrics/cohorts.")
        else:
            emit(knowledge.readout(root, args.candidate, args.conclusion, args.comparison, args.next_action, study_id=args.study))
    elif command == "history":
        from .history import import_history, inventory
        result = inventory(root) if args.check else import_history(root)
        emit({"counts": result["counts"], "coverage": result["coverage"], "catalog": "configs/forge/catalog.json"} if args.check else result)
        if not result["coverage"]["valid"]:
            return 1
    elif command == "validate":
        from .views import load_tasks, load_view
        from .dispositions import unresolved_explanations
        tasks = load_tasks(root)
        declared = [load_view(root, p.stem) for p in sorted((root / "configs/forge/views").glob("*.json"))]
        problems = unresolved_explanations(root, declared)
        if problems:
            raise ValueError("; ".join(problems))
        views = [view["id"] for view in declared]
        from .planning import declaration_paths, load_idea
        from .studies import load_study
        from .family_documentation import validate_family_documentation
        candidates = [load_idea(root, path.stem)["id"] for path in declaration_paths(root)]
        studies = [load_study(root, path.stem)["id"] for path in sorted((root / "configs/forge/studies").glob("*.json"))]
        emit({"tasks": len(tasks), "views": views, "candidates": len(candidates), "studies": studies,
              "family_documentation": validate_family_documentation(root), "training_launched": False})
    elif command == "calibrate":
        from .calibration import calibrate
        emit(calibrate(root, profile=args.profile))
    elif command == "calibration-lane":
        from .calibration_lane import register, plan_calibration
        if args.stage == "register":
            emit(register(root, root / args.contract))
        elif args.stage == "imports":
            from .calibration import diagnostic_imports
            emit(diagnostic_imports(root, args.registration, {args.lineage: args.tasks}))
        else:
            requests = plan_calibration(root, args.registration, queue_root, freeze_source=args.stage == "enqueue")
            if args.stage == "plan":
                from .planning import plan_summary
                emit([plan_summary(request, queue.inspect()) for request in requests])
            else:
                emit([{"request_id": queue.submit(request, request["calibration_campaign"])["request"]["request_id"],
                       "calibration_lane": request["calibration_lane"]} for request in requests])
    elif command == "promotion":
        from .promotion import register, plan_promotion, summarize_promotion
        if args.stage == "register":
            emit(register(root, args.candidate, root / args.contract))
        elif args.stage == "report":
            emit(summarize_promotion(root, args.registration))
        else:
            requests = plan_promotion(root, args.registration, queue_root, freeze_source=args.stage == "enqueue")
            if args.stage == "plan":
                from .planning import plan_summary
                emit([plan_summary(request, queue.inspect()) for request in requests])
            else:
                emit([{"request_id": queue.submit(request, request["promotion_campaign"])["request"]["request_id"],
                       "promotion": request["promotion"]} for request in requests])
    elif command == "retier":
        from .dispositions import write_view_revision
        from .views import load_view
        view = load_view(root, args.view)
        found = False
        for assignment in view["assignments"]:
            if assignment["task"] == args.task:
                assignment["qualification_tier"] = args.tier
                found = True
        if not found:
            raise ValueError("task is not assigned to this view")
        # Archives the live revision byte for byte first; refuses a different archive.
        write_view_revision(root, view, args.reason)
        from .knowledge import board
        emit(board(root, args.view))
    elif command == "cancel":
        queue.cancel(args.request_id)
        emit({"cancelled": args.request_id})
    elif command == "retry":
        queue.retry(args.compatibility_key, reason=args.reason)
        emit({"retry": args.compatibility_key})
    elif command in {"abandon", "supersede"}:
        from .lifecycle import abandon, supersede
        result = (abandon(root, args.candidate, args.reason, queue_root=queue_root)
                  if command == "abandon" else
                  supersede(root, args.candidate, args.successor, args.reason, queue_root=queue_root))
        publish()
        emit(result)
    elif command in {"pause", "resume"}:
        queue.pause(args.campaign, command == "pause")
        emit({"campaign": args.campaign, "paused": command == "pause"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Forge: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Forge: coordinator stopped claiming; active workers retain their deadlines and leases.", file=sys.stderr)
        raise SystemExit(130)
