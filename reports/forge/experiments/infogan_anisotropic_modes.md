<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# InfoGAN Disentanglement on 8 Anisotropic Ring Modes (`infogan_anisotropic_modes`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3B](../LAWS_HUB.md#law3b) |
| **Edge Cases** | `disentanglement` `anisotropic-covariance` `variational-mi` |
| **Primary View (`discriminator_stability`)** | Tier 4 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `infogan_anisotropic_modes` |
| **Prior** | `mog` (sigma=`0.05`, standardize=`False`) |
| **Update Budget** | `3500` steps |

## What This Experiment Tests

Trains InfoGAN (G([z, c_disc, c_cont]), D(x), Q(c|x)) on 8 rotated anisotropic Gaussians on a ring, requiring both 8/8 mode coverage AND disentangled recovery of discrete mode identity (NMI >= 0.85, q_acc >= 0.90) and continuous long-axis position (R^2 >= 0.65).

> [!NOTE]
> **Protocol notes:** Tier-4 stretch benchmark (3,500 updates).

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3B: Conditional Fiber & Stratum Identifiability](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **Edge-case tags:** `disentanglement`, `anisotropic-covariance`, `variational-mi`

## Common Failure Mode

The generator ignores the continuous latent code c_cont or scrambles discrete codes across modes when adversarial and mutual-information gradients conflict.

## Visual Schematic

![Target and protocol schematic for infogan_anisotropic_modes](assets/infogan_anisotropic_modes.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.05, P=256, z=2)"] --> G["Generator G(z)"]
    G --> D["Critic D(x) + Variational Q(c|x)"]
    D --> Gate["Gate: infogan_anisotropic_modes (3500 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.05`, standardize=`False`) |
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
| `modes` | `>=` | `8` |
| `hq` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.12` |
| `discrete_nmi` | `>=` | `0.85` |
| `q_acc` | `>=` | `0.9` |
| `continuous_r2` | `>=` | `0.65` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v4` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [Axiom](../families/axiom.md#cohort-cuda-daa283c8a640) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **FAIL** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `BLOCKED` | infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; decla | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks infogan_anisotropic_modes
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/infogan_anisotropic_modes.json](../../../configs/forge/tasks/infogan_anisotropic_modes.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/infogan_anisotropic_modes.json](../../../configs/forge/tasks/infogan_anisotropic_modes.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/infogan_anisotropic_modes.json](../../../configs/forge/task-documentation/infogan_anisotropic_modes.json)
