"""Compare transport on identical live outputs from frozen retention and blurry texture checkpoints."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.toy_audit import api_conv_texture32 as texture
from benchmarks.toy_audit import api_interpolation as arc
from benchmarks.toy_audit.reproducibility import reproducible_execution
from benchmarks.transfer_suite.vector_tasks import sample_target, score_samples
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import file_hash
from experiments.forge.planning import load_idea
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from particlegan import unwrap_prior

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "img_conv_texture32": ("e72d41bc328748b09108666b8bce01bf",
                           "dc57017901ac94f8aff53c2071b7a7dcd4370fdd60880bd59cc7b0cb49688ba5"),
    "vector_unequal_mass": ("152fe091abcc4fefaaa36e4da6c20a22",
                           "9b60dc389d12428c54741e98e377acd301ce084b797fa0d667b01d599b25d37d"),
    "interp_sparse_arc": ("a0ba74974db94bd693d3ae7f3d85ce2e",
                          "8e92c4f12440a5210067e7b4cab6e1af246c63e8292f62b47f75f0292ddb5293"),
}
PANELS = 32


def restore(task_id, checkpoint):
    if file_hash(checkpoint) != PINS[task_id][1]:
        raise ValueError(f"Checkpoint differs from the retained {task_id} endpoint")
    task = json.loads((ROOT / "configs/forge/tasks" / f"{task_id}.json").read_text())
    is_texture = task_id == "img_conv_texture32"
    candidate = load_idea(ROOT, "rhbi-v12" if is_texture else "rhbi-v12-feature-transport")
    ctx = task_formulation_context(candidate, task, {"seed": 0}, device="cuda:0", root=ROOT)
    spec = task["execution"]["host_definition"]
    if task_id == "vector_unequal_mass":
        g, d = build_vector_models(ctx, resolve_vector_spec(task))
    else:
        builder = texture.build_models_for_spec if is_texture else arc.build_models_for_spec

        def model(index, role):
            return ctx.construct(lambda: builder(spec, z_dim=ctx.recipe.z_dim)[index],
                                 component=role).to(ctx.device)
        g, d = model(0, "generator"), model(1, "discriminator")
    trainer = ctx.build_trainer(g, d, max_steps=spec["steps"])
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    ctx.load_state_dict(saved)
    if state_digest(ctx.state_dict()) != state_digest(saved):
        raise ValueError("Restored state differs from the hash-bound checkpoint")
    if trainer.completed_steps != spec["steps"] or trainer.loss.feature_transport != (not is_texture):
        raise ValueError("Checkpoint has the wrong budget or recipe")
    trainer.D.eval()
    return ctx, trainer, spec


def evaluate_metric(trainer, fake, real, mode):
    """Capture the production loss's cross-plan, including its slack column."""
    x = fake.detach().requires_grad_()
    loss = trainer.loss
    core = loss.rhbi
    head = loss.scalar_feature_head(trainer.D)
    fake_logits = loss.score_with_features(trainer.D, head, x)
    with torch.no_grad():
        real_logits = loss.score_with_features(trainer.D, head, real)
    if mode == "fixed_features":
        fake_logits._rhbi_features = fake_logits._rhbi_features.detach()
    solve = core._sinkhorn_slack
    plans = []

    def capture(*args, **kwargs):
        result = solve(*args, **kwargs)
        plans.append(result.detach())
        return result

    with (patch.object(core, "_sinkhorn_slack", side_effect=capture),
          patch.object(loss, "feature_transport", mode != "raw"),
          patch.object(trainer.opt_d.record, "_in_protected_probe", True)):
        core.bind_discrete_prior_step(trainer.prior_noise_generator is None)
        value = loss.g_loss(fake_logits, real_logits)
        grad, = torch.autograd.grad(value, x)
    if len(plans) != 2 or plans[0].shape != (len(fake) + 1, len(real) + 1):
        raise ValueError("Expected the marginal cross solve followed by the fake self solve")
    return plans[0][:-1].double(), grad.double(), value.detach()


def add_panel(totals, rows, distance, raw, feat, fixed):
    """Pool actual transported mass. Include the slack atom when comparing full rows."""
    p, grad_raw, _ = raw
    q, grad_feature, feature_loss = feat
    q_fixed, grad_fixed, fixed_loss = fixed
    torch.testing.assert_close(q_fixed, q, atol=0, rtol=0)
    torch.testing.assert_close(fixed_loss, feature_loss, atol=0, rtol=0)
    p, q, cost = p[rows], q[rows], distance[rows]
    if len(p) == 0:
        return
    totals["rows"] += len(p)
    totals["row_mass"] += p.sum().item()
    totals["plan_l1"] += (p - q).abs().sum().item()
    totals["grad_raw_sq"] += grad_raw[rows].square().sum().item()
    totals["grad_feature_sq"] += grad_feature[rows].square().sum().item()
    totals["grad_dot"] += (grad_raw[rows] * grad_feature[rows]).sum().item()
    totals["grad_delta_sq"] += (grad_feature[rows] - grad_raw[rows]).square().sum().item()
    totals["grad_feature_path_sq"] += (grad_feature[rows] - grad_fixed[rows]).square().sum().item()
    for label, plan in (("raw", p), ("feature", q)):
        active = plan[:, :-1]
        totals[label + "_active"] += active.sum().item()
        totals[label + "_cost"] += (active * cost).sum().item()
    # Weight by common active mass so that rows sent entirely to slack
    # cannot appear to change targets through a meaningless argmax.
    wp, wq = p[:, :-1], q[:, :-1]
    common = torch.minimum(wp.sum(1), wq.sum(1))
    i, j = wp.argmax(1), wq.argmax(1)
    ix = torch.arange(len(i), device=i.device)
    totals["common"] += common.sum().item()
    flipped = i != j
    totals["winner_flip"] += common[flipped].sum().item()
    raw_gap = (wp[ix, i] - wp[ix, j]) / wp.sum(1).clamp_min(1e-300)
    totals["raw_winner_gap"] += (common[flipped] * raw_gap[flipped]).sum().item()
    for label, condition in (("worse", cost[ix, j] > cost[ix, i]),
                             ("better", cost[ix, j] < cost[ix, i])):
        totals[label] += common[condition].sum().item()


def summarize(t):
    def ratio(n, d):
        return n / d if d else None
    return {
        "fake_rows": int(t["rows"]),
        "plan_tv_including_slack": t["plan_l1"] / (2 * t["row_mass"]),
        "raw_active_fraction": t["raw_active"] / t["row_mass"],
        "feature_active_fraction": t["feature_active"] / t["row_mass"],
        "raw_active_mean_task_distance": ratio(t["raw_cost"], t["raw_active"]),
        "feature_active_mean_task_distance": ratio(t["feature_cost"], t["feature_active"]),
        "top_target_changed": ratio(t["winner_flip"], t["common"]),
        "raw_probability_gap_on_flips": ratio(t["raw_winner_gap"], t["winner_flip"]),
        "top_task_distance_worsened": ratio(t["worse"], t["common"]),
        "top_task_distance_improved": ratio(t["better"], t["common"]),
        "gradient_cosine": ratio(t["grad_dot"], math.sqrt(t["grad_feature_sq"] * t["grad_raw_sq"])),
        "gradient_delta_over_raw_l2": math.sqrt(t["grad_delta_sq"] / t["grad_raw_sq"]),
        "direct_feature_gradient_over_raw_l2": math.sqrt(t["grad_feature_path_sq"] / t["grad_raw_sq"]),
    }


def arc_interiors(trainer, latents):
    codes = unwrap_prior(trainer.prior).z.detach()
    edges = torch.tensor(arc.knn_edges(codes, 3), device=codes.device)
    a, b = edges.T
    endpoints = trainer.sample(len(codes), fixed_first_n=True, generator=latents)
    t = codes.new_tensor(arc.CHORD_TS)[None, :, None]
    chord_codes = ((1 - t) * codes[a, None] + t * codes[b, None]).flatten(0, 1)
    chords = arc.evaluate_clean_codes(trainer.G, chord_codes, prior=trainer.prior)
    errors, _ = arc._arc_point_errors(chords)
    endpoint_error, angles = arc._arc_point_errors(endpoints)
    spans = (angles[a] - angles[b]).abs().rad2deg().repeat_interleave(t.shape[1])
    failed = errors > 3 * arc.ARC_SIGMA
    return chords, failed, {
        "edges": len(edges), "chords": len(chords), "failed_chords": int(failed.sum()),
        "bad_endpoints": int((endpoint_error > 3 * arc.ARC_SIGMA).sum()),
        "median_edge_degrees_failed_chords": spans[failed].median().item(),
        "median_edge_degrees_passing_chords": spans[~failed].median().item(),
        "failed_inside_circle": int((failed & (chords.norm(dim=1) < arc.ARC_RADIUS)).sum()),
    }


def saved_rare_width(directory, spec, rare):
    samples_file = directory / "observed-samples.pt"
    digest = file_hash(samples_file)
    if digest != "3990471c7dd4adb47513931e694fe3183b50116e6a75c8a659b31503488e9e87":
        raise ValueError("Expected the exact unequal-mass observations from the retention run")
    frames = torch.load(samples_file, weights_only=True, map_location="cpu")
    rows = []
    for frame in frames[-5:]:
        metrics = score_samples(frame["samples"], spec, frame["step"])
        rows.append({"step": frame["step"], "samples": metrics["sample_count"],
                     "rare_samples": metrics["component_counts"][rare],
                     "rare_core_eigen_ratio": metrics["component_core_eigen_ratios"][rare],
                     "gate_min_eigen_ratio": metrics["component_min_eigen_ratio"]})
    return {"sha256": digest, "terminal_observations": rows}


def probe(task_id, archive):
    attempt, sha = PINS[task_id]
    is_texture = task_id == "img_conv_texture32"
    campaign = "rhbi-texture-feature-v1" if is_texture else "rhbi-texture-retention-v1"
    ctx, trainer, spec = restore(task_id, archive / campaign / attempt / "provenance/provenance-state.pt")
    before = state_digest(ctx.state_dict())
    data = torch.Generator().manual_seed(0)
    latents = torch.Generator(device=ctx.device).manual_seed(0)
    out = {"attempt": attempt, "checkpoint_sha256": sha, "updates": trainer.completed_steps,
           "raw_sample_horizon": trainer.opt_d.record.rhbi_ell_ema}
    groups = defaultdict(lambda: defaultdict(float))
    batch = ctx.recipe.batch_size
    is_arc = task_id == "interp_sparse_arc"
    if is_arc:
        chords, failed, out["chord_geometry"] = arc_interiors(trainer, latents)
    elif not is_texture:
        centers = torch.tensor(spec["means"], device=ctx.device)
        rare = spec["masses"].index(min(spec["masses"]))
        out["saved_observations"] = saved_rare_width(archive / campaign / attempt, spec, rare)
        out["panels_without_rare_real"] = 0

    for _ in range(PANELS):
        fake = trainer.sample(batch, generator=latents)
        panels = [(fake, {"sample": torch.ones(batch, dtype=torch.bool, device=ctx.device)})]
        if is_texture:
            real = texture.sample_conv_texture32(batch, data).to(ctx.device)
        elif is_arc:
            real = arc.sample_arc_target(batch, data, device=ctx.device)
            panels.append((chords, {"failed_chords": failed, "passing_chords": ~failed}))
        else:
            real = sample_target(spec, batch, data, trainer.completed_steps).to(ctx.device)
            real_mode = torch.cdist(real, centers).argmin(1)
            rare_absent = not (real_mode == rare).any().item()
            out["panels_without_rare_real"] += int(rare_absent)
            rare_rows = torch.cdist(fake, centers).argmin(1) == rare
            panels[0][1]["rare_fake"] = rare_rows
            panels[0][1]["rare_fake_target_absent" if rare_absent else "rare_fake_target_present"] = rare_rows
        for samples, masks in panels:
            if is_texture:
                distance = torch.cdist(samples.flatten(1), real.flatten(1)) / math.sqrt(real[0].numel())
            elif is_arc:
                _, a = arc._arc_point_errors(samples)
                _, b = arc._arc_point_errors(real)
                distance = (a[:, None] - b).abs().rad2deg()
            else:
                fake_mode = torch.cdist(samples, centers).argmin(1)
                distance = (fake_mode[:, None] != real_mode).float()
            metrics = [evaluate_metric(trainer, samples, real, mode)
                       for mode in ("raw", "feature", "fixed_features")]
            for name, mask in masks.items():
                add_panel(groups[name], mask, distance, *metrics)
    if state_digest(ctx.state_dict()) != before:
        raise RuntimeError("Probe changed the serialized trainer or a named RNG stream")
    out["state_sha256_before_and_after"] = before
    out["panels"] = {k: summarize(v) for k, v in groups.items() if v["rows"]}
    return out


@reproducible_execution
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forge_runs", type=Path, help="Parent directory of the two original Forge campaigns")
    args = parser.parse_args()
    result = {task: probe(task, args.forge_runs) for task in PINS}
    print(json.dumps({"source_commit": "27803af6ae036e91b7c7a627ddd03771a51a13eb",
                      "protocol_seed": 0, "target_batches_per_task": PANELS,
                      "optimizer_updates_added": 0, "tasks": result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
