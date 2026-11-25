<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Five-Word Joint BiGAN 4k Retention Hold (Legacy Prior Cohort) (`recipe-prior-refactor-legacy-five_word_joint_hold`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3B](../LAWS_HUB.md#law3b) · [Law 4.5](../LAWS_HUB.md#law4_5) |
| **Edge Cases** | `discrete-vocabulary` `bidirectional-encoder` `checkpoint-continuation` `legacy-prior-cohort` |
| **Primary View (`discriminator_stability`)** | Not in `discriminator_stability` |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | No (diagnostic / auxiliary cohort) |
| **Adapter / Domain** | `word_joint` / `word_hold` |
| **Prior** | `particle_cloud` (sigma=`0.0`, standardize=`False`) |
| **Update Budget** | `4000` steps |

## What This Experiment Tests

Historical pre-refactor comparison cohort of `five_word_joint_hold` retained under its original prior ownership contract so legacy baselines stay strictly separated from `recipe_owned_v1` tasks. Continues the candidate's earliest passing `five_word_joint_smoke` checkpoint for 4,000 updates, requiring every single scheduled generation and inverse-reconstruction check to stay passing without drift.

> [!NOTE]
> **Protocol notes:** Retained legacy diagnostic cohort for `five_word_joint_hold` before the recipe-owned prior refactor; cannot substitute for current ordinary qualification cells.

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3B: Conditional Fiber & Stratum Identifiability](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 4.5: Coupled Minimax Contraction](../LAWS_HUB.md#law4_5)** ([Law 4.5 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `discrete-vocabulary`, `bidirectional-encoder`, `checkpoint-continuation`, `legacy-prior-cohort`

## Common Failure Mode

Even after clean acquisition, coupled BiGAN minimax dynamics gradually rotate or collapse latent word codes, causing late-window reconstruction drop or mode loss.

## Visual Schematic

![Target and protocol schematic for recipe-prior-refactor-legacy-five_word_joint_hold](assets/recipe-prior-refactor-legacy-five_word_joint_hold.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: particle_cloud (sigma=0.0, P=5, z=2)"] --> G["Generator G(z)"]
    X["Real Data x"] --> E["Encoder E(x)"]
    G --> D["Joint Critic D(x, z)"]
    E --> D
    X --> D
    D --> Gate["Gate: word_hold (4000 steps, clean) <- continues five_word_joint_smoke"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `particle_cloud` (sigma=`0.0`, standardize=`False`) |
| Update Budget | `4000` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=5` / `z_dim=2` |
| Sampling Law | `generated_and_paired_reconstructed_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Continuation Parent | [`five_word_joint_smoke`](five_word_joint_smoke.md) |
| Dependencies | [`five_word_joint_smoke`](five_word_joint_smoke.md) (`checkpoint`) |
| Integrity Guards | `finite_state`, `mechanism_exercised`, `optimizer_roles`, `rng_isolation` |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `sample_count` | `>=` | `1024` |
| `quality_fraction` | `>=` | `0.95` |
| `modes` | `==` | `5` |
| `mass_tv` | `<=` | `0.1` |
| `reconstruction_exact` | `==` | `1` |
| `minimum_reconstruction_token_probability` | `>=` | `0.9` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

No selected-family cohort cells are recorded for this task in the current `reports/forge/technique-inventory.json`.

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks recipe-prior-refactor-legacy-five_word_joint_hold
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json](../../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json)
- Source code: [benchmarks/toy_audit/api_images.py](../../../benchmarks/toy_audit/api_images.py)
- Source code: [benchmarks/locked_shared/baseline.py](../../../benchmarks/locked_shared/baseline.py)
- Source code: [benchmarks/locked_shared/observation.py](../../../benchmarks/locked_shared/observation.py)
- Source code: [benchmarks/toy_audit/definition_quality.py](../../../benchmarks/toy_audit/definition_quality.py)
- Source code: [benchmarks/toy_audit/reproducibility.py](../../../benchmarks/toy_audit/reproducibility.py)
- Source code: [benchmarks/transfer_suite/protocol.py](../../../benchmarks/transfer_suite/protocol.py)
- Source code: [experiments/forge/behavior_adapters.py](../../../experiments/forge/behavior_adapters.py)
- Source code: [experiments/forge/word_adapter.py](../../../experiments/forge/word_adapter.py)
- Source code: [experiments/forge/word_tasks.py](../../../experiments/forge/word_tasks.py)
- Source code: [particlegan/conditional_transport.py](../../../particlegan/conditional_transport.py)
- Source code: [particlegan/kinetic_transport.py](../../../particlegan/kinetic_transport.py)
- Source code: [particlegan/policy.py](../../../particlegan/policy.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json](../../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/recipe-prior-refactor-legacy-five_word_joint_hold.json](../../../configs/forge/task-documentation/recipe-prior-refactor-legacy-five_word_joint_hold.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/api_images.py](../../../benchmarks/toy_audit/api_images.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/locked_shared/baseline.py](../../../benchmarks/locked_shared/baseline.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/locked_shared/observation.py](../../../benchmarks/locked_shared/observation.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/toy_audit/definition_quality.py](../../../benchmarks/toy_audit/definition_quality.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/toy_audit/reproducibility.py](../../../benchmarks/toy_audit/reproducibility.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [benchmarks/transfer_suite/protocol.py](../../../benchmarks/transfer_suite/protocol.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/behavior_adapters.py](../../../experiments/forge/behavior_adapters.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/word_adapter.py](../../../experiments/forge/word_adapter.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/word_tasks.py](../../../experiments/forge/word_tasks.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [particlegan/conditional_transport.py](../../../particlegan/conditional_transport.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [particlegan/kinetic_transport.py](../../../particlegan/kinetic_transport.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [particlegan/policy.py](../../../particlegan/policy.py) (AST symbols: `__doc__`)
