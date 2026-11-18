"""Replay the pinned CUDA score, then count finite rows on the same decoder outputs."""
import torch

from benchmarks.toy_audit import api_interpolation as api
from experiments.forge.interpolation import interpolation_context
from experiments.forge.state import state_digest
from particlegan import unwrap_prior


@torch.no_grad()
def replay(request, task, directory, observations):
    kind = task["id"]
    spec = task["execution"]["host_definition"]
    ctx = interpolation_context(request, task, "cuda:0")

    def model(role, index):
        return ctx.construct(lambda: api.build_models_for_spec(spec, z_dim=ctx.recipe.z_dim)[index],
                             component=role).to(ctx.device)

    trainer = ctx.build_trainer(model("generator", 0), model("discriminator", 1),
                                max_steps=task["execution"]["steps"])
    reference = api.sample_target_for_kind(kind, task["evaluation"]["eval_samples"],
        ctx.streams.generator("eval", component="target", purpose="reference"), device="cpu")
    rows = []
    for observation in observations:
        step = observation["step"]
        state = torch.load(directory / f"state-{step:04d}.pt", map_location="cpu", weights_only=True)
        ctx.load_state_dict(state)
        if trainer.completed_steps != step or state_digest(ctx.state_dict()) != state_digest(state):
            raise ValueError(f"State restoration differed at {kind} {step}")
        codes = unwrap_prior(trainer.prior).z.detach()
        endpoints = trainer.sample(len(codes), fixed_first_n=True, output_noise=False,
            generator=ctx.streams.generator("eval", component="live", purpose="enumerated_samples"))
        outputs = {}

        def decode(query):
            key = state_digest(query)
            if key not in outputs:
                outputs[key] = api.evaluate_clean_codes(trainer.G, query, prior=trainer.prior)
            return outputs[key]

        native = api.score_interpolation_kind(kind, codes, endpoints, decode, reference=reference)
        if {"step": step, **native["metrics"]} != observation:
            raise ValueError(f"Pinned CUDA replay differed at {kind} {step}")
        # The first query is the chords or barycenters. Saddle also checks edge midpoints.
        primary = [endpoints, *list(outputs.values())[:1 if kind == "interp_img_patch_shift" else 2]]
        flags = torch.cat([torch.isfinite(value).flatten(1).all(1).cpu() for value in primary])
        checked = api.score_interpolation_kind(kind, codes, endpoints.cpu(), lambda q: decode(q).cpu(),
                                               reference=reference)
        count = int(flags.sum())
        if count != len(flags) or (native["metrics"]["finite_fraction"] == 1 and checked["metrics"] != native["metrics"]):
            raise ValueError(f"Nonfinite outputs or measured score mismatch at {kind} {step}")
        rows.append({"step": step, "finite_rows": count, "total_rows": len(flags),
                     "cuda_finite_fraction": native["metrics"]["finite_fraction"],
                     "passed": checked["passed"], "failed_bounds": checked["failed_bounds"],
                     "metrics": {name: checked["metrics"][name] for name, _, _ in task["evaluation"]["thresholds"]}})
    cfg = spec["interpolation"]
    if kind == "interp_img_patch_shift":
        graph = api.knn_edges(codes, cfg["k_neighbors"])
    else:
        graph = api.knn_triangles(codes, cfg["k_neighbors"], min_interior_ratio=cfg["min_interior_ratio"])
    return rows, (trainer, graph)


@torch.no_grad()
def cross_graph_scores(kind, trained, results):
    """Decode each graph owner's row IDs with each model's own final prior and generator."""
    metric = "interp_peak" if kind == "interp_img_patch_shift" else "barycenter_hq"
    matrix = {}
    for arm, (trainer, _) in trained.items():
        codes = unwrap_prior(trainer.prior).z.detach()
        matrix[arm] = {}
        for owner, (_, graph) in trained.items():
            if kind == "interp_img_patch_shift":
                query = torch.stack([(1 - t) * codes[i] + t * codes[j]
                                     for i, j in graph for t in api.CHORD_TS])
            else:
                query = torch.stack([(codes[i] + codes[j] + codes[k]) / 3 for i, j, k in graph])
            outputs = api.evaluate_clean_codes(trainer.G, query, prior=trainer.prior).cpu().float()
            if kind == "interp_img_patch_shift":
                cell = dict(value=outputs.amax(dim=(1, 2, 3)).mean().item(), queries=len(outputs))
            else:
                passing = api._saddle_surface_errors(outputs) <= 0.10
                cell = dict(value=passing.float().mean().item(), passing=int(passing.sum()), queries=len(outputs))
            matrix[arm][owner] = cell
            if arm == owner and results[arm]["integer_finiteness_diagnostic"][-1]["metrics"][metric] != cell["value"]:
                raise ValueError(f"Native graph mismatch for {kind} {arm}")
    return dict(metric=metric, step=800, scope="exploratory_final_graph_row_substitution",
                by_model_then_graph_owner=matrix)
