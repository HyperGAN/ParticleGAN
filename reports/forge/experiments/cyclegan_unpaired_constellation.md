<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# CycleGAN Unpaired Ring-to-Square Constellation Translation (`cyclegan_unpaired_constellation`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3B](../LAWS_HUB.md#law3b) · [Law 4.5](../LAWS_HUB.md#law4_5) |
| **Edge Cases** | `unpaired-translation` `dual-loop` `cycle-consistency` |
| **Primary View (`discriminator_stability`)** | Tier 4 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `cyclegan_unpaired_constellation` |
| **Prior** | `mog` (sigma=`0.08`, standardize=`False`) |
| **Update Budget** | `3500` steps |

## What This Experiment Tests

Simultaneously trains dual translators (G_XY, G_YX) and dual critics (D_Y, D_X) for unpaired domain translation between an 8-mode circular ring and an 8-mode square perimeter, verifying bidirectional coverage and cycle RMSE <= 0.12.

> [!NOTE]
> **Protocol notes:** Tier-4 stretch benchmark (3,500 updates).

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3B: Conditional Fiber & Stratum Identifiability](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 4.5: Coupled Minimax Contraction](../LAWS_HUB.md#law4_5)** ([Law 4.5 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `unpaired-translation`, `dual-loop`, `cycle-consistency`

## Common Failure Mode

Dual minimax loops decouple or collapse multiple source modes onto a single target mode, breaking cycle-consistency or losing target coverage in one direction.

## Visual Schematic

![Target and protocol schematic for cyclegan_unpaired_constellation](assets/cyclegan_unpaired_constellation.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    X["Domain X: 8-Mode Ring"] --> Gxy["Translator G_XY"]
    Gxy --> Dy["Critic D_Y"]
    Gxy --> Gyx["Translator G_YX"]
    Y["Domain Y: 8-Mode Square"] --> Gyx
    Gyx --> Dx["Critic D_X"]
    Dx --> Gate["Gate: cyclegan_unpaired_constellation (3500 steps, clean)"]
    Dy --> Gate
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.08`, standardize=`False`) |
| Update Budget | `3500` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=256` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Assigned Views | `discriminator_stability` (T4 stretch), `r3-tier4-frontier-holdout-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `modes_y` | `>=` | `7` |
| `modes_x` | `>=` | `7` |
| `hq_y` | `>=` | `0.85` |
| `hq_x` | `>=` | `0.85` |
| `mass_tv_y` | `<=` | `0.15` |
| `cycle_rmse` | `<=` | `0.12` |
| `bijection_ber` | `<=` | `0.15` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **BISPF** | `bispf-optimized-quintic-v1` | **PASS** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | **PASS** | No recorded result for this selected configuration and source. | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | **PASS** | No recorded result for this selected configuration and source. | [R1/R2](../families/r1r2.md#cohort-cuda-1bf9d7d34422) | — |
| **Atlas** | `atlas-particle-normal-v4` | **FAIL** | No recorded result for this selected configuration and source. | [Atlas](../families/atlas.md#cohort-cuda-bc3debd3fa5e) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **FAIL** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | cyclegan_unpaired_constellation: no registered public Forge adapter for this stretch host; | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | **FAIL** | No recorded result for this selected configuration and source. | [K3P](../families/k3p.md#cohort-cuda-1bf9d7d34422) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | **FAIL** | No recorded result for this selected configuration and source. | [KA2](../families/ka2.md#cohort-cuda-1bf9d7d34422) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | cyclegan_unpaired_constellation: no registered public Forge adapter for this stretch host; | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `BLOCKED` | cyclegan_unpaired_constellation: no registered public Forge adapter for this stretch host; | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks cyclegan_unpaired_constellation
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/cyclegan_unpaired_constellation.json](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/cyclegan_unpaired_constellation.json](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/cyclegan_unpaired_constellation.json](../../../configs/forge/task-documentation/cyclegan_unpaired_constellation.json)
