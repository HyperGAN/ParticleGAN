"""Shared discovery and validation for public-API toy variants.

Providers own mathematical targets, API hosts and frozen numerical gates. This
module owns the executable inventory and fails closed on malformed observations.
Historical audit definitions and receipts are inputs, never migration outputs.
"""
from __future__ import annotations

from .reproducibility import DEFAULT_SEED, construction_rng, VERSION as COMPARISON_VERSION

from importlib import import_module
from copy import deepcopy
import math
from pathlib import Path
import re

import numpy as np
import torch

from particlegan import Recipe, get_recipe


MODEL_PROVIDERS = ("api_model_ae", "api_model_vae", "api_model_ddgan", "api_model_ae_ddgan", "api_model_misgan")
PROVIDERS = ("api_images", "api_vectors", "api_conditionals", "api_diagnostics", "api_ring16", "api_gaussian1d") + MODEL_PROVIDERS
ROOT = Path(__file__).resolve().parents[2]
CASE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_case(case):
    """Require a runnable, scoped definition, not a source-only placeholder."""
    if not CASE_ID.fullmatch(case.get("id", "")):
        raise ValueError("invalid toy case ID")
    for name in ("title", "goal", "kind", "scope"):
        if not isinstance(case.get(name), str) or not case[name].strip():
            raise ValueError(f"{case['id']}: missing {name}")
    legacy = case.get("legacy_ids")
    if not isinstance(legacy, (list, tuple)) or not legacy or any(
            not isinstance(value, str) or not value for value in legacy):
        raise ValueError(f"{case['id']}: explicit historical/question mapping required")
    if len(legacy) != len(set(legacy)):
        raise ValueError(f"{case['id']}: repeated historical mapping")
    for name in ("default_steps", "batch_size", "eval_samples"):
        _positive_integer(case.get(name), f"{case['id']}.{name}")
    terminal = case.get("terminal_observations", 5)
    _positive_integer(terminal, f"{case['id']}.terminal_observations")
    if terminal > case["default_steps"]:
        raise ValueError(f"{case['id']}: terminal observations exceed the execution budget")
    if terminal > metric_observations(case):
        raise ValueError(f"{case['id']}: terminal observations exceed the declared metric cadence")
    if not case.get("thresholds"):
        raise ValueError(f"{case['id']}: frozen pass/fail bounds required")
    if not case.get("sampling"):
        raise ValueError(f"{case['id']}: sampling law required")
    return case


def discover():
    cases = {}
    for name in PROVIDERS:
        provider = import_module(f"benchmarks.toy_audit.{name}")
        for definition in provider.list_cases():
            case = validate_case(dict(definition))
            if case["id"] in cases:
                raise ValueError(f"duplicate API case: {case['id']}")
            case["provider"] = name
            case.setdefault("default_recipe", "atlas")
            case.setdefault("protocol_seed", DEFAULT_SEED)
            case["comparison_version"] = COMPARISON_VERSION
            case.setdefault("evaluation_observations", metric_observations(case))
            cases[case["id"]] = case
    return cases


def metric_observations(case):
    """Frozen post-update scoring count, independent of GIF frame selection."""
    thresholds = case.get("thresholds", {})
    inherited = thresholds.get("observations") if isinstance(thresholds, dict) else None
    count = case.get("evaluation_observations", inherited if inherited is not None else min(24, case["default_steps"]))
    _positive_integer(count, "evaluation_observations")
    if count > case["default_steps"]:
        raise ValueError("metric observations exceed distinct execution updates")
    return count


def coverage(cases, historical_ids):
    """Retain every useful old question, even when a new variant is narrower."""
    mappings = {name: [] for name in historical_ids}
    for case in cases.values():
        for name in case["legacy_ids"]:
            mappings.setdefault(name, []).append(case["id"])
    return {"required_questions": len(historical_ids),
            "api_variants": len(cases),
            "missing": sorted(name for name in historical_ids if not mappings[name]),
            "mapping": {name: sorted(values) for name, values in sorted(mappings.items())}}


class UnsupportedRecipeOverrides(ValueError):
    """This fixture has no declared contract for arbitrary tuning overrides."""


def validate_caller_objectives(recipe, *, provider, kind=None):
    """Reject an active objective unless this caller actually consumes it."""
    if recipe.conditional_energy_weight and provider != "api_model_ddgan":
        raise UnsupportedRecipeOverrides("conditional energy score requires the explicit DDGAN component consumer")
    if recipe.joint_reconstruction_weight and (provider != "api_images" or kind != "word"):
        raise UnsupportedRecipeOverrides("joint reconstruction objective requires the explicit word joint consumer")
    if ((recipe.masked_mixture_weight or recipe.masked_transport_weight or recipe.imputer_component_weight)
            and provider != "api_model_misgan"):
        raise UnsupportedRecipeOverrides("masked mixture/imputer objectives require the explicit MisGAN component consumer")


def validate_recipe_overrides(case, recipe_name, recipe_overrides):
    """Allow public tuning knobs without changing the host or sampling law."""
    from experiments.forge.configuration_search import TUNABLE_FIELDS
    if recipe_overrides is None:
        return {}
    if not isinstance(recipe_overrides, dict):
        raise ValueError("recipe_overrides must be a JSON object")
    if not recipe_overrides:
        return {}
    forbidden = set(recipe_overrides) - TUNABLE_FIELDS
    if forbidden:
        raise ValueError(f"non-tuning Recipe overrides are forbidden: {sorted(forbidden)}")
    if (case.get("provider") not in {"api_vectors", "api_images"} or case.get("query")
            or case.get("caller_owned") or case.get("penalty_arm") or case.get("kind") == "word"
            or case.get("id", "").startswith("api-stress-")):
        raise UnsupportedRecipeOverrides(f"{case['id']}: tuning overrides are unsupported for this caller/conditional/stress host")
    result = deepcopy(recipe_overrides)
    get_recipe(recipe_name, **result)  # Public validation before any optimizer construction.
    return result


def build(case, *, device="cpu", seed=DEFAULT_SEED, recipe_name=None, max_steps=None, recipe_overrides=None, recipe=None):
    if case.get("checkpoint_required") and not case.get("checkpoint_capabilities", {}).get("exact_continuation"):
        raise UnsupportedRecipeOverrides(f"{case['id']}: required exact checkpoint continuation is unsupported by this provider")
    provider = import_module(f"benchmarks.toy_audit.{case['provider']}")
    recipe_name = recipe_name or case["default_recipe"]
    overrides = validate_recipe_overrides(case, recipe_name, recipe_overrides)
    options = {"recipe_overrides": overrides} if overrides else {}
    if recipe is not None:
        if case["provider"] not in (*MODEL_PROVIDERS, "api_vectors") or not isinstance(recipe, Recipe) or overrides:
            raise ValueError("resolved Recipe injection requires a supported model/vector toy without tuning overrides")
        validate_caller_objectives(recipe, provider=case["provider"], kind=case.get("kind"))
        options["recipe"] = recipe
    with construction_rng(seed, device):
        fixture = provider.build_case(case["id"], device=device, seed=seed,
                                      recipe_name=recipe_name, max_steps=max_steps, **options)
    if not isinstance(fixture.recipe, Recipe):
        raise TypeError(f"{case['id']}: recipe must be the public particlegan.Recipe")
    validate_caller_objectives(fixture.recipe, provider=case["provider"], kind=case.get("kind"))
    if not fixture.api_components or any(not isinstance(name, str) or not name
                                          for name in fixture.api_components):
        raise ValueError(f"{case['id']}: public API components must be identified")
    for name in ("step", "observe", "state_dict"):
        if not callable(getattr(fixture, name, None)):
            raise TypeError(f"{case['id']}: missing executable {name}")
    return fixture


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_observation(observation):
    """A metric exception/NaN can never silently earn a PASS."""
    result = dict(observation)
    if type(result.get("passed")) is not bool:
        raise ValueError("observation must contain a binary passed value")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("numeric pass/fail metrics required")
    failures = result.get("failed_bounds")
    if not isinstance(failures, (list, tuple)) or any(not isinstance(x, str) for x in failures):
        raise ValueError("failed_bounds must identify rejected metric bounds")
    failures = list(failures)
    normalized = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (int, float, bool)):
            raise ValueError(f"metric {name} must be scalar")
        normalized[name] = value
        if not math.isfinite(value):
            failures.append(f"{name}: nonfinite")
    views = result.get("views")
    if not isinstance(views, (list, tuple)) or not views:
        raise ValueError("goal-illustrating target/output views required")
    for view in views:
        if view.get("kind") not in {"scatter", "image", "line", "bar", "text"}:
            raise ValueError("unsupported goal view")
        if not view.get("title"):
            raise ValueError("goal view title required")
        if "row_labels" in view and (view["kind"] != "image" or
                not isinstance(view["row_labels"], (list, tuple)) or
                len(view["row_labels"]) != 2 or any(
                    not isinstance(label, str) or not label for label in view["row_labels"])):
            raise ValueError("image row labels must identify both displayed rows")
        if view["kind"] == "text":
            for role in ("target", "sample"):
                labels = view.get(f"{role}_labels")
                if not isinstance(labels, (list, tuple)) or not labels or any(
                        not isinstance(label, str) for label in labels):
                    raise ValueError("text goal views require decoded actual/reference labels")
        for role in ("target", "samples"):
            values = array(view[role])
            if not values.size or not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"goal view {role} must contain finite actual values")
            invalid = int((~np.isfinite(values)).sum())
            if invalid and role == "target":
                raise ValueError("goal reference must be finite")
            if invalid:
                failures.append(f"{view['title']}: {invalid} nonfinite output values")
                normalized["nonfinite_output_values"] = normalized.get("nonfinite_output_values", 0) + invalid
            if view["kind"] == "image" and (values.ndim not in (2, 3, 4) or
                    (values.ndim == 4 and values.shape[1] not in (1, 3, 4))):
                raise ValueError("image goal views require a matrix or NCHW grayscale/RGB values; pack feature blocks explicitly")
    result["metrics"] = normalized
    result["failed_bounds"] = sorted(set(failures))
    result["passed"] = result["passed"] and not result["failed_bounds"]
    if not result["passed"] and not result["failed_bounds"]:
        raise ValueError("FAIL must identify at least one rejected metric bound")
    return result


def evaluation_steps(steps, frames=9):
    _positive_integer(steps, "steps")
    _positive_integer(frames, "frames")
    return sorted({0, steps, *(math.ceil(steps * i / max(1, frames - 1))
                               for i in range(1, frames))})
