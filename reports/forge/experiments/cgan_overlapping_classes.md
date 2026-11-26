<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Conditional Projection GAN on Overlapping Classes (`cgan_overlapping_classes`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 3B](../LAWS_HUB.md#law3b) · [Law 2](../LAWS_HUB.md#law2) |
| **Edge Cases** | `conditional-class-overlap` `projection-critic` `within-class-variance` |
| **Primary View (`discriminator_stability`)** | Tier 4 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Unimplemented Tier-4 adapter backlog (planned; excluded from active runnable denominator) |
| **Adapter / Domain** | `transfer_vector` / `cgan_overlapping_classes` |
| **Prior** | `mog` (sigma=`0.025`, standardize=`False`) |
| **Update Budget** | `3000` steps |

## What This Experiment Tests

Evaluates class-conditional generation with a Projection Discriminator on 4 overlapping 2D Gaussian classes (sigma=0.35), specifically checking that within-class variance and central overlap density (>= 0.55) are preserved.

> [!NOTE]
> **Protocol notes:** Tier-4 stretch benchmark (3,000 updates).

## Laws & Edge Cases Targeted

- **[Law 3B: Conditional Fiber & Stratum Identifiability (Unifies with Provisional Law 5)](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **Edge-case tags:** `conditional-class-overlap`, `projection-critic`, `within-class-variance`

## Common Failure Mode

Conditioned critics push classes artificially apart to make classification easier, hollowing out the true central overlap region (overlap_density_ratio < 0.55) and shrinking per-class variance.

## Visual Schematic

![Target and protocol schematic for cgan_overlapping_classes](assets/cgan_overlapping_classes.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.025, P=256, z=2)"] --> G["Generator G(z)"]
    G --> D["Critic: Projection Discriminator D(x, y)"]
    D --> Gate["Gate: cgan_overlapping_classes (3000 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.025`, standardize=`False`) |
| Update Budget | `3000` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=256` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Assigned Views | `discriminator_stability` (T4 stretch), `r3-tier4-frontier-holdout-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `classes_covered` | `==` | `4` |
| `class_mass_tv` | `<=` | `0.08` |
| `mean_center_err` | `<=` | `0.12` |
| `min_cov_eig_ratio` | `>=` | `0.6` |
| `max_cov_eig_ratio` | `<=` | `1.5` |
| `overlap_density_ratio` | `>=` | `0.55` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **PASS** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **PASS** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **PASS** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v5` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [Axiom](../families/axiom.md#cohort-cuda-42a130a78fc3) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v15` | `BLOCKED` | cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declar | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks cgan_overlapping_classes
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/cgan_overlapping_classes.json](../../../configs/forge/tasks/cgan_overlapping_classes.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/cgan_overlapping_classes.json](../../../configs/forge/tasks/cgan_overlapping_classes.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/cgan_overlapping_classes.json](../../../configs/forge/task-documentation/cgan_overlapping_classes.json)
