<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# PacGAN m=8 Packed Critic on 100-Mode Grid (`pacgan8_grid100`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 2](../LAWS_HUB.md#law2) |
| **Edge Cases** | `packed-critic` `100-modes` `disconnected-support` |
| **Primary View (`discriminator_stability`)** | Tier 4 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Unimplemented Tier-4 adapter backlog (planned; excluded from active runnable denominator) |
| **Adapter / Domain** | `transfer_vector` / `pacgan8_grid100` |
| **Prior** | `mog` (sigma=`0.05`, standardize=`False`) |
| **Update Budget** | `4000` steps |

## What This Experiment Tests

Packs m=8 samples jointly into the critic D(x_1..x_8) on a 100-mode 10x10 Gaussian grid (sigma=0.05), testing high-order product-measure mode coverage (>= 95/100 modes) and covariance calibration.

> [!NOTE]
> **Protocol notes:** Tier-4 stretch benchmark (4,000 updates).

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **Edge-case tags:** `packed-critic`, `100-modes`, `disconnected-support`

## Common Failure Mode

Packing increases critic input dimensionality 8x; improperly scaled gradient penalties or weak Law 1 priors still leave collapsed or under-represented modes.

## Visual Schematic

![Target and protocol schematic for pacgan8_grid100](assets/pacgan8_grid100.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.05, P=512, z=2)"] --> G["Generator G(z)"]
    G --> D["Critic: PacGAN m=8 Packed D(x1..x8)"]
    D --> Gate["Gate: pacgan8_grid100 (4000 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.05`, standardize=`False`) |
| Update Budget | `4000` steps |
| Batch Size | `512` |
| Particles / Latent Dim | `P=512` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Assigned Views | `discriminator_stability` (T4 stretch), `r3-tier4-frontier-holdout-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `modes` | `>=` | `95` |
| `hq` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.15` |
| `min_cov_eig_ratio` | `>=` | `0.15` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v5` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [Axiom](../families/axiom.md#cohort-cuda-42a130a78fc3) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **FAIL** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v15` | `BLOCKED` | pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and imp | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks pacgan8_grid100
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/pacgan8_grid100.json](../../../configs/forge/tasks/pacgan8_grid100.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/pacgan8_grid100.json](../../../configs/forge/tasks/pacgan8_grid100.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/pacgan8_grid100.json](../../../configs/forge/task-documentation/pacgan8_grid100.json)
