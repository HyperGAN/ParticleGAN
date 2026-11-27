"""Clean oracle-free v2 AE-DDGAN composition (paired reconstruction + independent chain).

Compared with the frozen v1 cohort (`api_model_ae_ddgan.py`), v2 removes:
- host-owned ground-truth covariance whitening of reconstruction errors,
- host-owned target-mean cluster assignment (`self.target.means`) and prior tethering,
- host-owned batch `codes_det` prior tethering, and
- host-owned target-variance `div_loss`.

Only the public `ParticleEncoding.reconstruction_loss` weighted by
`recipe.reconstruction_weight` is added to the public adversarial + prior losses.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch

from experiments.forge.rng import NamedStreams

from .api_model_ae_ddgan import (
    PAIRED_BOUNDS,
    ParticleEncoder,
    paired_chain,
    paired_metrics,
)
from .api_model_ddgan_v2 import (
    DDGANV2Fixture,
    list_cases as ddgan_v2_cases,
    task_recipe as task_recipe,
)
from .reproducibility import DEFAULT_SEED

VERSION = "api-model-ae-ddgan-v2"

_CASES = []
for _parent in ddgan_v2_cases():
    _case = deepcopy(_parent)
    _v1_id = _parent["id"].replace("api-model-ddgan-", "api-model-ae-ddgan-").removesuffix("-v2")
    _case.update(
        id=_v1_id + "-v2",
        legacy_ids=[_v1_id, "ae_ddgan-public-composition", "denoising-toy-full-chain"],
        title="AE-DDGAN " + _parent["target"] + " paired reconstruction and independent chain (oracle-free v2)",
        goal=(
            "Reconstruct held-out inputs through a learned particle AE and four public DDGAN "
            "reverse steps while preserving independent generation and conditional-transition "
            "quality without host-owned target whitening, target-mean cluster tethering, or div_loss"
        ),
        kind="ae_ddgan_chain",
        encoder_mode="ae",
        terminal_observations=1,
        oracle_free=True,
        scope=(
            "Clean oracle-free v2 AE-DDGAN cohort: identical analytic target, schedule, G/D/E "
            "architecture, prior and calibrated terminal_observations=1 protocol as v1, with "
            "host-owned target whitening, target-mean cluster tethering, batch codes_det prior "
            "tethering, and target-variance div_loss removed."
        ),
    )
    _case["thresholds"]["paired"] = deepcopy(PAIRED_BOUNDS)
    _case["forge_thresholds"] += deepcopy(PAIRED_BOUNDS)
    _case["sampling"] += (
        "; separate paired cohort: encode held-out clean input once, hold its deterministic AE code "
        "through all four reverse steps from independent N(0,I), fresh Gaussian reverse noise each step; "
        "shuffled-code control shares identical start and reverse noise"
    )
    if _case["id"] == "api-model-ae-ddgan-gaussian-chain-v2":
        _gaussian_overrides = {
            "final_normalized_mean": 0.45,
            "final_projection_cdf": 0.20,
            "t1_q0_covariance_eigen_max": 1.80,
            "t1_q1_covariance_eigen_max": 1.80,
            "t1_q0_normalized_mean": 0.65,
            "t1_q1_normalized_mean": 0.65,
        }
        for _entry in _case["forge_thresholds"]:
            if _entry[0] in _gaussian_overrides:
                _entry[2] = _gaussian_overrides[_entry[0]]
    elif _case["id"] == "api-model-ae-ddgan-bimodal-chain-v2":
        _bimodal_overrides = {
            "final_hq": 0.90,
            "final_local_mean_sigma": 1.50,
            "final_local_covariance_max": 2.05,
            "final_local_covariance_min": 0.30,
            "final_projection_cdf": 0.32,
        }
        for _entry in _case["forge_thresholds"]:
            if _entry[0] in _bimodal_overrides:
                _entry[2] = _bimodal_overrides[_entry[0]]
    _case["thresholds"] = {
        name: f"{op} {bound}" for name, op, bound in _case["forge_thresholds"]
    }
    _CASES.append(_case)
CASES = {case["id"]: case for case in _CASES}


def _evaluate_thresholds(metrics: dict, thresholds: list) -> list[str]:
    """Evaluate every declared threshold transparently without silent bound dropping."""
    failed: list[str] = []
    for name, op, bound in thresholds:
        val = metrics.get(name)
        if (
            val is None
            or isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(float(val))
        ):
            failed.append(name)
            continue
        v = float(val)
        b = float(bound)
        ok = v <= b if op == "<=" else (v >= b if op == ">=" else (v == b if op == "==" else False))
        if not ok:
            failed.append(name)
    return sorted(set(failed))


def list_cases():
    return deepcopy(_CASES)


class AEDDGANV2Fixture(DDGANV2Fixture):
    """Oracle-free AE-DDGAN fixture: pure public reconstruction + adversarial + recipe prior losses."""

    schema_version: str = VERSION
    api_components = DDGANV2Fixture.api_components + ("Recipe.encode", "ParticleEncoding.reconstruction_loss")
    lambda_div: float = 0.0
    reconstruction_scale: float = 1.0

    def __init__(self, case, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
        super().__init__(case, device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps, recipe=recipe)
        self.whitening = torch.eye(2, device=self.device, dtype=torch.float32)
        self.reconstruction_scale = 1.0

    def make_encoder(self):
        return ParticleEncoder(self.recipe.z_dim)

    def encode(self, clean):
        query, offset = self.E(clean)
        return self.recipe.encode(query, self.prior, offset=offset)

    def extra_generator_objective(self, clean, xt, t):
        self.E.train()
        encoding = self.encode(clean)
        self.bind_prior_rows(encoding.indices)
        codes = encoding.codes[:, 0]
        predicted = self.G(xt, codes, t)
        reconstruction = encoding.reconstruction_loss(predicted[:, None], clean)
        weighted = self.recipe.reconstruction_weight * reconstruction
        return weighted, (reconstruction,), dict(
            reconstruction_mse=float(reconstruction.detach()),
            tether_loss=0.0,
        )

    @torch.no_grad()
    def observe(self, n=2048, seed=10000):
        modes = {module: module.training for model in self._models().values() for module in model.modules()}
        try:
            self.E.eval()
            record = super().observe(n=n, seed=seed)
            self.G.eval()
            evaluation = NamedStreams(int(seed), device=self.device)
            clean = self.target.sample(
                n, evaluation.generator("eval", component="paired_target", device="cpu")
            ).to(self.device)
            encoding = self.encode(clean)
            start = evaluation.generator("eval", component="paired_start", device=self.device)
            reverse = evaluation.generator("eval", component="paired_reverse", device=self.device)
            start_state, reverse_state = start.get_state(), reverse.get_state()
            reconstructed, states = paired_chain(
                self.G, self.schedule, encoding.codes[:, 0],
                start_rng=start, reverse_rng=reverse, return_states=True,
            )
            start.set_state(start_state)
            reverse.set_state(reverse_state)
            shuffled = paired_chain(
                self.G, self.schedule, encoding.codes[:, 0].roll(1, 0),
                start_rng=start, reverse_rng=reverse,
            )
            metrics, _ = paired_metrics(clean, reconstructed, shuffled, self.target)
            metrics["paired_reverse_steps"] = len(states) - 1
            record["metrics"].update(metrics)
            nonfinite_tags = [
                tag for tag in record["failed_bounds"] if "nonfinite" in str(tag)
            ]
            if not (
                bool(torch.isfinite(reconstructed).all())
                and bool(torch.isfinite(shuffled).all())
            ):
                nonfinite_tags.append("paired_nonfinite")
            evaluated_failures = _evaluate_thresholds(
                record["metrics"], self.case["forge_thresholds"]
            )
            record["failed_bounds"] = sorted(set(evaluated_failures + nonfinite_tags))
            record["passed"] = not record["failed_bounds"]
            record["views"].append(
                dict(
                    kind="scatter",
                    title="Paired four-step reconstruction (oracle-free v2)",
                    target=clean.cpu(),
                    samples=reconstructed.cpu(),
                    caption=(
                        "Held-out input -> learned AE code -> four actual reverse steps. "
                        "Independent-prior chain is scored in the first panel."
                    ),
                )
            )
            desired = torch.stack((clean[:, 0], clean[:, 0]), 1).cpu()
            actual = torch.stack((clean[:, 0], reconstructed[:, 0]), 1).cpu()
            record["views"].append(
                dict(
                    kind="scatter",
                    title="Input x versus reconstructed x (oracle-free v2)",
                    target=desired,
                    samples=actual,
                    caption=(
                        f"Desired diagonal; code shuffle uses identical chain noise. "
                        f"NMSE={metrics['paired_nmse']:.3g}; shuffle gain={metrics['paired_shuffled_gain']:.3g}."
                    ),
                )
            )
            return record
        finally:
            for module, mode in modes.items():
                module.training = mode


def build_case(case_id, *, device="cpu", seed=DEFAULT_SEED, recipe_name=None, max_steps=None, recipe=None):
    if case_id not in CASES:
        raise ValueError("unknown AE-DDGAN v2 API toy: " + case_id)
    if type(seed) is not int or seed != 0:
        raise ValueError("this prospective protocol is bound to seed 0")
    return AEDDGANV2Fixture(
        CASES[case_id],
        device=device,
        seed=seed,
        recipe_name=recipe_name,
        max_steps=max_steps,
        recipe=recipe,
    )
