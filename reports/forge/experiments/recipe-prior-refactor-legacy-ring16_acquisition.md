<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 16-Mode 2D Gaussian Ring Acquisition (Legacy Prior Cohort) (`recipe-prior-refactor-legacy-ring16_acquisition`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 2](../LAWS_HUB.md#law2) |
| **Edge Cases** | `disconnected-support` `ring-mixture` `covariance-calibration` `legacy-prior-cohort` |
| **Primary View (`discriminator_stability`)** | Not in `discriminator_stability` |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | No (diagnostic / auxiliary cohort) |
| **Adapter / Domain** | `transfer_vector` / `transfer_sustained` |
| **Prior** | `mog` (sigma=`0.1`, standardize=`False`) |
| **Update Budget** | `1600` steps |

## What This Experiment Tests

Historical pre-refactor comparison cohort of `ring16_acquisition` retained under its original prior ownership contract so legacy baselines stay strictly separated from `recipe_owned_v1` tasks. Core Tier-1 multimodal acquisition test: recover all 16 equally weighted 2D Gaussians arranged on a circle (R=3.0, sigma=0.10) with balanced occupancy (mass_tv <= 0.15), high quality (hq >= 0.85), and non-collapsed per-mode covariance.

> [!NOTE]
> **Protocol notes:** Retained legacy diagnostic cohort for `ring16_acquisition` before the recipe-owned prior refactor; cannot substitute for current ordinary qualification cells.

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **Edge-case tags:** `disconnected-support`, `ring-mixture`, `covariance-calibration`, `legacy-prior-cohort`

## Common Failure Mode

Mode dropping, inter-mode bridges along chords of the circle, or collapsed per-mode eigenvalues (component_min_eigen_ratio < 0.15) caused by poor Law 1 support geometry or unstable critic forces.

## Visual Schematic

![Target and protocol schematic for recipe-prior-refactor-legacy-ring16_acquisition](assets/recipe-prior-refactor-legacy-ring16_acquisition.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.1, P=256, z=4)"] --> G["Generator G(z)"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: transfer_sustained (1600 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.1`, standardize=`False`) |
| Update Budget | `1600` steps |
| Batch Size | `128` |
| Particles / Latent Dim | `P=256` / `z_dim=4` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `sample_count` | `>=` | `4096` |
| `modes` | `>=` | `16` |
| `mass_tv` | `<=` | `0.15` |
| `hq` | `>=` | `0.85` |
| `component_covariance_error` | `<=` | `0.85` |
| `component_min_eigen_ratio` | `>=` | `0.15` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

No selected-family cohort cells are recorded for this task in the current `reports/forge/technique-inventory.json`.

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks recipe-prior-refactor-legacy-ring16_acquisition
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json](../../../configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json)
- Source code: [benchmarks/transfer_suite/vector_tasks.py](../../../benchmarks/transfer_suite/vector_tasks.py)
- Source code: [benchmarks/locked_shared/observation.py](../../../benchmarks/locked_shared/observation.py)
- Source code: [benchmarks/toy_audit/ring16_quality.py](../../../benchmarks/toy_audit/ring16_quality.py)
- Source code: [benchmarks/transfer_suite/protocol.py](../../../benchmarks/transfer_suite/protocol.py)
- Source code: [experiments/forge/transfer_cadence.py](../../../experiments/forge/transfer_cadence.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json](../../../configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/recipe-prior-refactor-legacy-ring16_acquisition.json](../../../configs/forge/task-documentation/recipe-prior-refactor-legacy-ring16_acquisition.json)
- **Read-only Python AST source:** [benchmarks/transfer_suite/vector_tasks.py](../../../benchmarks/transfer_suite/vector_tasks.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/locked_shared/observation.py](../../../benchmarks/locked_shared/observation.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/toy_audit/ring16_quality.py](../../../benchmarks/toy_audit/ring16_quality.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/transfer_suite/protocol.py](../../../benchmarks/transfer_suite/protocol.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/transfer_cadence.py](../../../experiments/forge/transfer_cadence.py) (AST symbols: `__doc__`)
