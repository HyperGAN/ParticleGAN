<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# ALICE Cycle-Consistent Spiral BiGAN (`alice_spiral_bigan`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3B](../LAWS_HUB.md#law3b) |
| **Edge Cases** | `bidirectional-encoder` `continuous-manifold` `cycle-consistency` |
| **Primary View (`discriminator_stability`)** | Tier 4 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `alice_spiral_bigan` |
| **Prior** | `mog` (sigma=`0.06`, standardize=`False`) |
| **Update Budget** | `4000` steps |

## What This Experiment Tests

Joint BiGAN (G(z), E(x), D(x, z)) with ALICE cycle-consistency regularization on a 2D Archimedes spiral, requiring >= 15/16 arc bins, high sample quality, accurate x and z reconstructions, and arc consistency >= 0.90.

> [!NOTE]
> **Protocol notes:** Tier-4 stretch benchmark (4,000 updates, MoGParticlePrior sigma=0.06).

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3B: Conditional Fiber & Stratum Identifiability](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **Edge-case tags:** `bidirectional-encoder`, `continuous-manifold`, `cycle-consistency`

## Common Failure Mode

Without stable joint-distribution matching, encoder-generator pairs fold spiral turns onto one another, degrading latent_recon_rmse or arc_consistency.

## Visual Schematic

![Target and protocol schematic for alice_spiral_bigan](assets/alice_spiral_bigan.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.06, P=256, z=2)"] --> G["Generator G(z)"]
    X["Real Data x"] --> E["Encoder E(x)"]
    G --> D["Joint Critic D(x, z)"]
    E --> D
    X --> D
    D --> Gate["Gate: alice_spiral_bigan (4000 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.06`, standardize=`False`) |
| Update Budget | `4000` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=256` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Assigned Views | `discriminator_stability` (T4 stretch), `r3-tier4-frontier-holdout-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `arc_bins` | `>=` | `15` |
| `hq` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.16` |
| `recon_rmse` | `<=` | `0.14` |
| `latent_recon_rmse` | `<=` | `0.25` |
| `arc_consistency` | `>=` | `0.9` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **PASS** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **PASS** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v4` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [Axiom](../families/axiom.md#cohort-cuda-daa283c8a640) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `BLOCKED` | alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks alice_spiral_bigan
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/alice_spiral_bigan.json](../../../configs/forge/tasks/alice_spiral_bigan.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/alice_spiral_bigan.json](../../../configs/forge/tasks/alice_spiral_bigan.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/alice_spiral_bigan.json](../../../configs/forge/task-documentation/alice_spiral_bigan.json)
