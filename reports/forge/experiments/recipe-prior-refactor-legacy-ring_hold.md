<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 8-Mode Ring First-Convergence & Hold Phase (Legacy Prior Cohort) (`recipe-prior-refactor-legacy-ring_hold`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3A](../LAWS_HUB.md#law3a) · [Law 4.5](../LAWS_HUB.md#law4_5) |
| **Edge Cases** | `long-horizon-stability` `uninterrupted-endurance` `checkpoint-producer` `legacy-prior-cohort` |
| **Primary View (`discriminator_stability`)** | Not in `discriminator_stability` |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | No (diagnostic / auxiliary cohort) |
| **Adapter / Domain** | `ring_endurance` / `ring_hold` |
| **Prior** | `mog` (sigma=`0.025`, standardize=`False`) |
| **Update Budget** | `7500` steps |

## What This Experiment Tests

Historical pre-refactor comparison cohort of `ring_hold` retained under its original prior ownership contract so legacy baselines stay strictly separated from `recipe_owned_v1` tasks. Monitors dense first-convergence and sustained hold on an 8-mode 2D Gaussian ring (modes == 8, hq >= 0.90) using the archived K3P ConvergenceGate, producing the exact state extended by `ring_extension`.

> [!NOTE]
> **Protocol notes:** Retained legacy diagnostic cohort for `ring_hold` before the recipe-owned prior refactor; cannot substitute for current ordinary qualification cells.

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3A: Unconditional W1 Ray-Length & Feature-Space Screening Identifiability](../LAWS_HUB.md#law3a)** ([Law 3A theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 4.5: Coupled Minimax Contraction](../LAWS_HUB.md#law4_5)** ([Law 4.5 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `long-horizon-stability`, `uninterrupted-endurance`, `checkpoint-producer`, `legacy-prior-cohort`

## Common Failure Mode

Recipes that briefly hit 8/8 modes and then drift below hq=0.90 or drop a mode fail to lock in a valid hold state.

## Visual Schematic

![Target and protocol schematic for recipe-prior-refactor-legacy-ring_hold](assets/recipe-prior-refactor-legacy-ring_hold.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.025)"] --> G["Generator G(z)"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: ring_hold (7500 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.025`, standardize=`False`) |
| Update Budget | `7500` steps |
| Batch Size | `host-declared` |
| Particles / Latent Dim | `P=host` / `z_dim=host` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Dependencies | [`mode_hold`](mode_hold.md) (`gate`) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `modes` | `==` | `8` |
| `hq` | `>=` | `0.9` |
| `hq` | `<=` | `1.0` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

No selected-family cohort cells are recorded for this task in the current `reports/forge/technique-inventory.json`.

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks recipe-prior-refactor-legacy-ring_hold
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json](../../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json)
- Source code: [reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py](../../toy100/gap-fill-20260925/sources/k3p/convergence_gate.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json](../../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/recipe-prior-refactor-legacy-ring_hold.json](../../../configs/forge/task-documentation/recipe-prior-refactor-legacy-ring_hold.json)
- **Read-only Python AST source:** [reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py](../../toy100/gap-fill-20260925/sources/k3p/convergence_gate.py) (AST symbols: `__doc__`)
