"""Prospective output-noise diagnostics on the same native clean draws.

The required task retains its clean/live verdict. These additional artifacts
use the unchanged frozen evaluators, never supply another reference cell, and
draw only from separately named evaluation streams.
"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from particlegan.training import output_noise_std

from .contracts import atomic_json

FIELD = "paired_output_noise_diagnostic"
DECLARATION = {"version": 1, "sampling_law": "public_recipe_scheduled_output_noise",
               "base_draws": "same_required_clean_samples", "scoring_weights": ["live", "ema"],
               "checks": "all_declared_and_independent_holdout", "qualification_reuse": False,
               "noise_rng": "named_eval_output_noise_per_model_and_holdout"}


def paired_sampling_blockers(task):
    evaluation = task.get("evaluation", {})
    if FIELD not in evaluation:
        return []
    value = evaluation[FIELD]
    if value != DECLARATION or task.get("adapter") != "native100":
        return [f"{task.get('id', '<task>')}: unsupported paired output-noise diagnostic declaration"]
    return []


def add_output_noise(samples, sigma, generator):
    """The additive public sampling law; sigma=0 consumes no extra RNG."""
    if sigma == 0:
        return samples
    return samples + sigma * torch.randn(samples.shape, dtype=samples.dtype,
                                        device=samples.device, generator=generator)


class PairedOutputNoiseEvidence:
    def __init__(self, context, config, output, steps, reference):
        from benchmarks.toy100.accuracy_evidence import AccuracyEvidence
        self.context = context
        self.config = {**deepcopy(config), "eval_output_noise": "public_recipe_schedule",
                       FIELD: deepcopy(DECLARATION)}
        self.output = Path(output) / config["problem"]
        (self.output / "snapshots").mkdir(parents=True)
        atomic_json(self.output / "config.json", self.config)
        # PairedOutputNoiseEvidence owns paired holdout noise via
        # clean_output/holdout_samples.npz and only delegates terminal
        # quality_checks recording to AccuracyEvidence.observe(). Pass
        # toy100's boolean switch while keeping the named policy in receipts.
        self.accuracy = AccuracyEvidence(
            {**self.config, "eval_output_noise": True}, self.output, steps, reference)
        self.target = reference.detach().cpu().numpy()
        self.sigmas = {}
        self.final = {}
        self.arrays = {}

    def _noise(self, points, model, *, holdout=False):
        sigma = output_noise_std(self.context.recipe, self.context._trainer.completed_steps)
        stream = self.context.streams.generator("eval", component=f"paired_noisy_{model}",
            purpose="holdout_output_noise" if holdout else "output_noise")
        return add_output_noise(points, sigma, stream), sigma

    def observe(self, step, model, clean_draw, elapsed):
        from benchmarks.toy100.metrics import evaluate_samples
        noisy, sigma = self._noise(clean_draw, model)
        metrics = evaluate_samples(noisy, self.config["problem"])
        fidelity = self.accuracy.observe(step, model, noisy, metrics)
        self.sigmas[str(step)] = sigma
        self.final[model] = metrics
        self.arrays[model] = noisy.detach().cpu().numpy()
        event = {"event": "eval", "model": model, "step": step, "elapsed": elapsed,
                 "metrics": metrics, "accuracy": fidelity, "output_noise_std": sigma}
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        if model == "ema":
            arrays = {**self.arrays, "target": self.target}
            np.savez_compressed(self.output / "snapshots" / f"step_{step:06d}.npz",
                **{key: values[:self.config["snapshot_samples"]] for key, values in arrays.items()})

    def finish(self, clean_summary, clean_output):
        from benchmarks.toy100.accuracy import evaluate_accuracy
        from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
        from benchmarks.toy100.gate import evaluate_suite as coverage_suite
        if self.accuracy.pending:
            raise RuntimeError("incomplete paired terminal evidence")
        np.savez_compressed(self.output / "final_samples.npz", **self.arrays, target=self.target)
        holdout = {}
        with np.load(Path(clean_output) / "holdout_samples.npz", allow_pickle=False) as archive:
            arrays = {"target": archive["target"].copy()}
            for model in ("live", "ema"):
                clean = torch.from_numpy(archive[model].copy()).to(self.context.device)
                noisy, _ = self._noise(clean, model, holdout=True)
                arrays[model] = noisy.cpu().numpy()
        for model, points in arrays.items():
            holdout[model] = evaluate_accuracy(points, self.config["problem"])
        np.savez_compressed(self.output / "holdout_samples.npz", **arrays)
        summary = {**deepcopy(clean_summary), "config": self.config,
                   "eval_output_noise": "public_recipe_schedule", "final": self.final,
                   "holdout": holdout, "output_noise_std_by_step": self.sigmas,
                   "evidence_use": "paired_sampling_diagnostic", "qualification_reuse": False}
        atomic_json(self.output / "summary.json", summary)
        return {"declaration": deepcopy(DECLARATION), "artifact_subdirectory": "paired-output-noise",
                "output_noise_std_by_step": self.sigmas,
                "holdout_output_noise_std": output_noise_std(self.context.recipe, self.context._trainer.completed_steps),
                "coverage": coverage_suite(self.output.parent, problem=self.config["problem"], write=False),
                "accuracy": accuracy_suite(self.output.parent, problem=self.config["problem"], write=False),
                "evidence_use": "paired_sampling_diagnostic", "qualification_reuse": False,
                "independent_reference_credits": 0}
