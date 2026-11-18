"""Score the three final sparse-arc models on each other's learned particle graphs."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.toy_audit import api_interpolation as arc
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import file_hash, read_json
from experiments.forge.interpolation import interpolation_context
from experiments.forge.state import state_digest
from particlegan import unwrap_prior


def compare_saved_paths(campaign, arms, pair, *, verify_raw_assignment=False):
    """Compare the recorded model tensors and the active cross-plan on every update."""
    traces = []
    for arm in pair:
        folder = campaign / arms[arm]["attempt"]
        archive = read_json(folder / "raw-result.json")["evidence"]["rhbi_trajectory"]
        directory = folder / archive["directory"]
        verify_artifacts(directory, archive["artifacts"])
        traces.append((directory, archive))
    (full, a), (candidate, b) = traces
    if a["state_steps"] != b["state_steps"] or a["plan_updates"] != b["plan_updates"] or a["plan_updates"] != 600:
        raise ValueError("Incomplete paired trace")
    same_models = 0
    for step in a["state_steps"]:
        states = [torch.load(d / f"state-{step:04d}.pt", map_location="cpu", weights_only=True)
                  for d in (full, candidate)]
        same_models += state_digest(states[0]["trainer"]["models"]) == state_digest(states[1]["trainer"]["models"])
    same_plans = 0
    raw_replay_matches = 0
    parent_max_raw_tv = 0.0
    for end in range(50, 601, 50):
        left, right = [torch.load(d / f"plans-{end:04d}.pt", map_location="cpu", weights_only=True)
                       for d in (full, candidate)]
        if (not torch.equal(left["generator_rows"], right["generator_rows"])
                or not torch.equal(left["real"], right["real"]) or left["plan"].shape[0] != 50):
            raise ValueError("Unmatched consumed particles or real examples")
        same_plans += int((left["plan"] == right["plan"]).flatten(1).all(1).sum())
        if verify_raw_assignment:
            raw_replay_matches += int((right["direct_tv"] == 0).sum())
            parent_max_raw_tv = max(parent_max_raw_tv, left["direct_tv"].max().item())
    result = {"saved_states": len(a["state_steps"]), "identical_model_states": same_models,
            "updates": a["plan_updates"], "identical_executed_cross_plans": same_plans}
    if verify_raw_assignment:
        if raw_replay_matches != a["plan_updates"]:
            raise ValueError("Executed candidate plans differ from their same-state raw replay")
        result["candidate_same_state_raw_replay_matches"] = raw_replay_matches
        result["parent_max_same_state_raw_replay_tv"] = parent_max_raw_tv
    return result


@reproducible_execution
@torch.no_grad()
def cross_score(runs, measurement, study="raw-self"):
    campaign = runs / f"rhbi-{study}-arc-v1"
    graphs, models = {}, {}
    for arm, row in measurement["arms"].items():
        folder = campaign / row["attempt"]
        request = read_json(folder / "request.json")["request"]
        task = request["tasks"]["interp_sparse_arc"]
        spec = task["execution"]["host_definition"]
        ctx = interpolation_context(request, task, "cuda:0")
        def model(role, index):
            return ctx.construct(lambda: arc.build_models_for_spec(spec, z_dim=ctx.recipe.z_dim)[index],
                                 component=role).to(ctx.device)
        trainer = ctx.build_trainer(model("generator", 0), model("discriminator", 1), max_steps=600)
        path = folder / "provenance/provenance-state.pt"
        if file_hash(path) != row["checkpoint_sha256"]:
            raise ValueError("Terminal checkpoint differs from the measured run")
        state = torch.load(path, map_location="cpu", weights_only=True)
        ctx.load_state_dict(state)
        if trainer.completed_steps != 600 or state_digest(ctx.state_dict()) != state_digest(state):
            raise ValueError("Terminal state failed exact restoration")
        models[arm] = trainer
        graphs[arm] = arc.knn_edges(unwrap_prior(trainer.prior).z, spec["interpolation"]["k_neighbors"])

    counts = {}
    for arm, trainer in models.items():
        counts[arm] = {}
        codes = unwrap_prior(trainer.prior).z.detach()
        for owner, edges in graphs.items():
            interior = torch.stack([(1 - t) * codes[i] + t * codes[j] for i, j in edges for t in arc.CHORD_TS])
            points = arc.evaluate_clean_codes(trainer.G, interior, prior=trainer.prior).cpu().float()
            errors, _ = arc._arc_point_errors(points)
            good = errors <= 3 * arc.ARC_SIGMA
            counts[arm][owner] = {"passing": int(good.sum()), "total": len(good)}
            if arm == owner and good.float().mean().item() != measurement["arms"][arm]["terminal_observations"][-1]["interp_hq"]:
                raise ValueError("Rescored native graph differs from the official final observation")
    result = {"scope": "exploratory_terminal_graph_comparison", "edges": graphs,
              "passing_chords_by_model_and_edge_owner": counts}
    if study == "raw-self":
        pair = ("rhbi-v12-feature-transport", "rhbi-v12-feature-raw-self")
        result["full_feature_vs_raw_self"] = compare_saved_paths(campaign, measurement["arms"], pair)
    else:
        pair = ("rhbi-v12-feature-raw-self", "rhbi-v12-raw-assignment")
        result["parent_vs_raw_assignment"] = compare_saved_paths(campaign, measurement["arms"], pair,
                                                                  verify_raw_assignment=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forge_runs", type=Path)
    parser.add_argument("--study", choices=("raw-self", "raw-assignment"), default="raw-self")
    args = parser.parse_args()
    measure = read_json(Path(__file__).resolve().parents[1] / args.study / "arc.json")
    print(json.dumps(cross_score(args.forge_runs, measure, args.study), sort_keys=True, indent=2))
