<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 2-Pair MisGAN: 8D MCAR p=0.50 Coordinate Missingness (`misgan_mcar_p50_recovery`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3B](../LAWS_HUB.md#law3b) |
| **Edge Cases** | `masked-observation` `mcar-50pct` `8d-subspace` |
| **Primary View (`discriminator_stability`)** | Tier 4 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `misgan_mcar_p50_recovery` |
| **Prior** | `mog` (sigma=`0.025`, standardize=`False`) |
| **Update Budget** | `4000` steps |

## What This Experiment Tests

Recovers a 100-mode 2D manifold embedded in 8D when 50% of coordinates are randomly missing completely at random (MCAR), testing simultaneous data manifold and mask distribution recovery.

> [!NOTE]
> **Protocol notes:** Tier-4 stretch benchmark (4,000 updates, empty masks resampled).

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3B: Conditional Fiber & Stratum Identifiability](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **Edge-case tags:** `masked-observation`, `mcar-50pct`, `8d-subspace`

## Common Failure Mode

High coordinate dropout blurs the 8D subspace constraint (off_plane > 0.05) and reduces recovered modes below 90/100.

## Visual Schematic

![Target and protocol schematic for misgan_mcar_p50_recovery](assets/misgan_mcar_p50_recovery.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    Obs["Incomplete 8D Data (x * m, m)"] --> Dx["Data Critic D_x (detached m)"]
    Obs --> Dm["Mask Critic D_m"]
    P["Prior: mog (sigma=0.025, P=256, z=8)"] --> Gx["Data Generator G_x"]
    Gm["Mask Generator G_m"] --> Dm
    Gx --> Dx
    Dx --> Gate["Gate: misgan_mcar_p50_recovery (4000 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.025`, standardize=`False`) |
| Update Budget | `4000` steps |
| Batch Size | `512` |
| Particles / Latent Dim | `P=256` / `z_dim=8` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Assigned Views | `discriminator_stability` (T4 stretch), `r3-tier4-frontier-holdout-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `modes` | `>=` | `90` |
| `hq` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.12` |
| `off_plane` | `<=` | `0.05` |
| `mask_tv` | `<=` | `0.05` |
| `acc` | `>=` | `0.9` |
| `impute_rmse` | `<=` | `0.12` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **PASS** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **PASS** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **PASS** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v4` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [Axiom](../families/axiom.md#cohort-cuda-daa283c8a640) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `BLOCKED` | misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declar | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks misgan_mcar_p50_recovery
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/misgan_mcar_p50_recovery.json](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/misgan_mcar_p50_recovery.json](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/misgan_mcar_p50_recovery.json](../../../configs/forge/task-documentation/misgan_mcar_p50_recovery.json)
