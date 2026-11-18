"""Reduce the four original Forge attempts and the previous feature trajectory."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from benchmarks.toy_audit import api_interpolation as arc
from benchmarks.toy_audit.reproducibility import reproducible_execution
from benchmarks.transfer_suite.vector_tasks import CORE_MAHALANOBIS2, score_samples, target_scale
from experiments.forge.api import task_formulation_context
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import file_hash, read_json, stable_hash
from experiments.forge.rhbi_trajectory import ARC_VIEW, VIEW, capture_steps
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from particlegan import unwrap_prior

ROOT = Path(__file__).resolve().parents[4]
SOURCES = {"vector_unequal_mass": "32255fa7e51272a4b07cfe89bd1b441821a20edf",
           "interp_sparse_arc": "28b534dfc431cdefcea0b334d8d63fec01c4a837"}
ARMS = ("rhbi-v12", "rhbi-v12-feature-transport")
OLD = {"vector_unequal_mass": "152fe091abcc4fefaaa36e4da6c20a22",
       "interp_sparse_arc": "a0ba74974db94bd693d3ae7f3d85ce2e"}


def tensor_file(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Run:
    def __init__(self, directory):
        envelope = read_json(directory / "request.json")
        self.request = envelope["request"]
        self.task_id = envelope["job"]["task_id"]
        self.task = self.request["tasks"][self.task_id]
        self.arm = self.request["candidate"]["id"]
        self.directory = directory
        self.raw = read_json(directory / "raw-result.json")
        self.evidence = self.raw["evidence"]
        self.archive = self.evidence["rhbi_trajectory"]
        self.files = directory / self.archive["directory"]
        self.budget = self.task["execution"]["steps"]
        self.steps = sorted(capture_steps(self.budget))
        self.result = read_json(directory / "result.json")["task_results"][0]
        if (self.request["source"]["origin_commit"] != SOURCES[self.task_id] or self.request["protocol"]["seed"] != 0
                or self.request["view"]["id"] != (ARC_VIEW if self.task_id == "interp_sparse_arc" else VIEW) or self.archive["state_steps"] != self.steps
                or self.archive["plan_updates"] != self.budget or self.result["gate_status"] not in {"FAIL", "PASS"}
                or self.evidence["guards"]["unintended_rng_deviations"] != 0):
            raise ValueError(f"Unexpected or incomplete attempt {directory}")
        verify_artifacts(self.files, self.archive["artifacts"])
        self.ctx = task_formulation_context(self.request["candidate"], self.task, self.request["protocol"],
                                            device="cuda:0", root=ROOT)
        self.spec = self.task["execution"]["host_definition"]
        if self.task_id == "vector_unequal_mass":
            models = build_vector_models(self.ctx, resolve_vector_spec(self.task))
        else:
            def model(index, role):
                return self.ctx.construct(lambda: arc.build_models_for_spec(self.spec, z_dim=self.ctx.recipe.z_dim)[index],
                                          component=role).to(self.ctx.device)
            models = (model(0, "generator"), model(1, "discriminator"))
        self.trainer = self.ctx.build_trainer(*models, max_steps=self.budget)

    def restore(self, step):
        state = tensor_file(self.files / f"state-{step:04d}.pt")
        self.ctx.load_state_dict(state)
        digest = state_digest(state)
        if self.trainer.completed_steps != step or state_digest(self.ctx.state_dict()) != digest:
            raise ValueError(f"Restoration differs at {self.arm} {self.task_id} {step}")
        return digest

    def quality(self, step, edges):
        rng = torch.Generator(device=self.ctx.device).manual_seed(0)
        if edges is None:
            samples = self.trainer.sample(4096, generator=rng).cpu()
            metrics = score_samples(samples, self.spec, step)
            units = target_scale(self.spec, step)
            means = torch.tensor(self.spec["means"]) * units
            rare = self.spec["masses"].index(min(self.spec["masses"]))
            cov = torch.tensor(self.spec["covariances"][rare]) * units ** 2
            delta = samples[torch.cdist(samples, means).argmin(1) == rare] - means[rare]
            mahal = torch.einsum("ni,ij,nj->n", delta, cov.inverse(), delta)
            count = int((mahal <= CORE_MAHALANOBIS2).sum())
            return {"count": len(delta), "core_count": count,
                    "width": metrics["component_core_eigen_ratios"][rare] if count >= 10 else None}
        codes = unwrap_prior(self.trainer.prior).z.detach()
        endpoints = self.trainer.sample(len(codes), fixed_first_n=True, generator=rng)
        a, b = edges.T
        t = codes.new_tensor(arc.CHORD_TS)[None, :, None]
        interior = ((1 - t) * codes[a, None] + t * codes[b, None]).flatten(0, 1)
        points = arc.evaluate_clean_codes(self.trainer.G, interior, prior=self.trainer.prior)
        errors = arc._arc_point_errors(points)[0]
        end_errors = arc._arc_point_errors(endpoints)[0]
        return {"mean_arc_error": errors.double().mean().item(),
                "fixed_edge_hq": (errors <= 3 * arc.ARC_SIGMA).double().mean().item(),
                "endpoint_hq": (end_errors <= 3 * arc.ARC_SIGMA).double().mean().item()}

    def summary(self):
        grade = self.result["evaluator_result"]
        terminal = ({"terminal_passed": grade["terminal_passed"]} if "terminal_passed" in grade
                    else {"passing_suffix": grade["convergence"]["passing_suffix"]})
        return {**terminal, "attempt": self.directory.name, "result_file_sha256": file_hash(self.directory / "result.json"),
                "candidate_revision": self.request["candidate_revision"], "status": self.result["gate_status"],
                "worker_seconds": self.result["cost"]["wall_seconds"], "scored_law": self.evidence["sampling_law"],
                "terminal_state_sha256": self.evidence["provenance_checkpoint"]["state_sha256"],
                "trace_manifest_sha256": self.archive["artifacts"]["sha256"]}


def plan_timing(parent, feature):
    rows = []
    for end in range(50, parent.budget + 1, 50):
        a, b = [tensor_file(run.files / f"plans-{end:04d}.pt") for run in (parent, feature)]
        if (a["plan"].shape != b["plan"].shape or a["plan"].shape[0] != 50
                or not torch.equal(a["generator_rows"], b["generator_rows"]) or a["direct_tv"].count_nonzero()):
            raise ValueError(f"Nonmatching generator rows, shapes or raw-plan replay at {end}")
        p, q = a["plan"].double(), b["plan"].double()
        tv = (p - q).abs().sum((1, 2)) / (p + q).sum((1, 2))
        for i in range(len(p)):
            rows.append({"update": end - len(p) + i + 1, "between_tv": tv[i].item(),
                         "direct_tv": b["direct_tv"][i].item(),
                         "same_critic_target": torch.equal(a["real"][i], b["real"][i])})
    if len(rows) != parent.budget:
        raise ValueError("Missing update")
    if not all(row["same_critic_target"] for row in rows):
        raise ValueError("The arms consumed different critic targets")
    return rows


def crossings(rows, key):
    over = [r[key] >= 0.01 for r in rows]
    first = next((i for i in range(len(over) - 2) if all(over[i:i+3])), None)
    return {"first_single": next((i + 1 for i, v in enumerate(over) if v), None),
            "first_three": None if first is None else first + 1,
            "tv_at_first_three": [] if first is None else [r[key] for r in rows[first:first+3]],
            "max": max(r[key] for r in rows), "updates_above": sum(over)}


def quality_crossing(rows, key, threshold, *, acquired=False):
    previous = None
    for row in rows:
        a, b = row["parent"], row["feature"]
        if (a[key] is not None and b[key] is not None and
                (not acquired or min(a["endpoint_hq"], b["endpoint_hq"]) >= .85)):
            if abs(a[key] - b[key]) >= threshold:
                return {"previous_observation": previous, "first_observation": row}
            previous = row
        else:
            previous = None
    return None


def first_parameter_changes(parent, feature, initial):
    """Exploratory timing of the first bitwise difference in live parameters."""
    roles = {role: list(dict(getattr(parent.trainer, role).named_parameters())) for role in ("G", "D", "prior")}
    changes = {}
    previous, previous_step = initial, 0
    for step in parent.steps[1:]:
        a, b = [tensor_file(run.files / f"state-{step:04d}.pt")["trainer"]["models"] for run in (parent, feature)]
        for role, names in roles.items():
            if role in changes or all(torch.equal(a[role][k], b[role][k]) for k in names):
                continue
            def distance(left, right):
                return sum((left[role][k].double() - right[role][k].double()).square().sum().item() for k in names) ** 0.5
            movement = distance(a, previous)
            changes[role] = {"previous_equal": previous_step, "first_different": step,
                             "difference_over_parent_movement": distance(a, b) / movement if movement else None}
        if len(changes) == len(roles):
            break
        previous, previous_step = a, step
    return changes


def compare(parent, feature, archive, traces):
    contracts = [deepcopy(run.task) for run in (parent, feature)]
    for contract in contracts:
        contract["field_ownership"]["recipe_fields"].pop("rhbi_feature_transport", None)
    if (parent.request["source"]["digest"] != feature.request["source"]["digest"] or contracts[0] != contracts[1]
            or parent.archive["real_batch_sequence_sha256"] != feature.archive["real_batch_sequence_sha256"]
            or parent.raw["initializer"] != feature.raw["initializer"]
            or parent.evidence["sampling_law"] != feature.evidence["sampling_law"]):
        raise ValueError("Incompatible matched comparison")
    recipes = [r.raw["recipe"] for r in (parent, feature)]
    delta = {k for k in set(recipes[0]) | set(recipes[1]) if recipes[0].get(k) != recipes[1].get(k)}
    if delta != {"rhbi_feature_transport"} or recipes[0].get("rhbi_feature_transport", False) or not recipes[1]["rhbi_feature_transport"]:
        raise ValueError("Unexpected recipe delta")
    init = [tensor_file(run.files / "state-0000.pt") for run in (parent, feature)]
    initial = state_digest(init[0]["trainer"]["models"])
    if initial != state_digest(init[1]["trainer"]["models"]) or state_digest(init[0]["streams"]) != state_digest(init[1]["streams"]):
        raise ValueError("Different starting models, prior or named streams")
    original = archive / "rhbi-texture-retention-v1" / OLD[parent.task_id]
    old = read_json(original / "raw-result.json")["evidence"]
    replay = {"all_official_observations": old["observations"] == feature.evidence["observations"],
              "terminal_public_state": old["provenance_checkpoint"]["state_sha256"] == feature.summary()["terminal_state_sha256"]}
    # Pin the earlier endpoint independently of the mutable campaign directory.
    original_pins = read_json(Path(__file__).resolve().parent.parent / "retention-probe.json")["tasks"][parent.task_id]
    if original_pins["attempt"] != original.name or original_pins["state_sha256_before_and_after"] != old["provenance_checkpoint"]["state_sha256"]:
        raise ValueError("Original terminal state differs from the published endpoint")
    final_states = []
    for directory, evidence in ((original, old), (feature.directory, feature.evidence)):
        sample = tensor_file(directory / "provenance/provenance-state.pt")
        if state_digest(sample) != evidence["provenance_checkpoint"]["state_sha256"]:
            raise ValueError("Terminal public state differs from its own receipt")
        final_states.append(sample)
    replay["ambient_rng_equal"] = {name: torch.equal(final_states[0]["trainer"][name], final_states[1]["trainer"][name])
                                   for name in ("cpu_rng", "cuda_rng")}
    for sample in final_states:
        for name in ("cpu_rng", "cuda_rng"):
            sample["trainer"].pop(name)
    replay["all_other_state_including_named_rng"] = state_digest(final_states[0]) == state_digest(final_states[1])
    plans = plan_timing(parent, feature)
    qualities, edges = [], None
    is_arc = parent.task_id == "interp_sparse_arc"
    for step in parent.steps:
        measurements = []
        for run in (parent, feature):
            expected = run.restore(step)
            if is_arc and step == 0 and edges is None:
                edges = torch.tensor(arc.knn_edges(unwrap_prior(run.trainer.prior).z, 3), device=run.ctx.device)
            measurements.append(run.quality(step, edges))
            if state_digest(run.ctx.state_dict()) != expected:
                raise ValueError(f"Diagnostic consumed trainer state at {step}")
        qualities.append({"step": step, "parent": measurements[0], "feature": measurements[1]})
    traces.mkdir(parents=True, exist_ok=True)
    for suffix, entries in (("plans", plans), ("quality", qualities)):
        with (traces / f"{parent.task_id}-{suffix}.jsonl").open("w") as f:
            for row in entries:
                print(json.dumps(row, sort_keys=True), file=f)
    key, threshold = ("mean_arc_error", .03) if is_arc else ("width", .05)
    result = {"source_sha256": parent.request["source"]["digest"], "initial_models_sha256": initial,
              "batch_sequence_sha256": parent.archive["real_batch_sequence_sha256"],
              "initial_streams_sha256": state_digest(init[0]["streams"]),
              "all_generator_row_ids_equal": True, "equal_critic_target_updates": sum(r["same_critic_target"] for r in plans),
              "direct": crossings(plans, "direct_tv"), "between_arms": crossings(plans, "between_tv"),
              "quality_crossing": quality_crossing(qualities, key, threshold), "endpoint_quality": qualities[-1],
              "candidate_exact_replay": replay, "parent": parent.summary(), "feature": feature.summary(),
              "archive_observations_sha256": stable_hash(old["observations"]),
              "plan_probe_updates": len(plans), "quality_probe_states_per_arm": len(qualities)}
    result["exploratory_first_parameter_changes"] = first_parameter_changes(parent, feature, init[0]["trainer"]["models"])
    if is_arc:
        graphs = [arc.knn_edges(unwrap_prior(r.trainer.prior).z, 3) for r in (parent, feature)]
        result["exploratory_endpoint_graphs"] = {
            "edges_parent_feature_shared": [len(graphs[0]), len(graphs[1]), len(set(graphs[0]) & set(graphs[1]))],
            "hq_by_model_and_graph": [[r.quality(r.budget, torch.tensor(g, device=r.ctx.device))["fixed_edge_hq"]
                                       for g in graphs] for r in (parent, feature)]}
        result["common_initial_edges"] = edges.tolist()
        result["after_both_endpoints_acquired"] = quality_crossing(qualities, key, threshold, acquired=True)
        result["official_final_chord_hq"] = [r.evidence["live"]["interp_hq"] for r in (parent, feature)]
    else:
        result["first_width_comparable"] = next((r for r in qualities if r["parent"]["width"] is not None and r["feature"]["width"] is not None), None)
        result["first_width_floor_loss"] = next((r for r in qualities if r["parent"]["width"] is not None and r["feature"]["width"] is not None
                                                 and r["parent"]["width"] >= .15 > r["feature"]["width"]), None)
        result["official_terminal_width"] = [[{"step": o["step"], "width": o["component_core_eigen_ratios"][3]}
                                              for o in r.evidence["observations"][-5:]] for r in (parent, feature)]
    return result


@reproducible_execution
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forge_runs", type=Path)
    parser.add_argument("local_traces", type=Path)
    args = parser.parse_args()
    runs = {}
    for path in sorted(p for campaign in (VIEW, ARC_VIEW) for p in (args.forge_runs / campaign).glob("*/request.json")):
        run = Run(path.parent)
        key = (run.task_id, run.arm)
        if key in runs:
            raise ValueError(f"Duplicate attempt {key}")
        runs[key] = run
    if set(runs) != {(task, arm) for task in OLD for arm in ARMS}:
        raise ValueError("Expected exactly the four frozen runs")
    results = {task: compare(runs[task, ARMS[0]], runs[task, ARMS[1]], args.forge_runs, args.local_traces) for task in OLD}
    print(json.dumps({"source_commits": SOURCES, "qualification_input": False, "tasks": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
