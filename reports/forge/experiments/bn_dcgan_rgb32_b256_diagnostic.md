<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Radford 32x32 BN-DCGAN Convergence & B=256 Throughput Guard (`bn_dcgan_rgb32_b256_diagnostic`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 2](../LAWS_HUB.md#law2) · [Law 4.2](../LAWS_HUB.md#law4_2) |
| **Edge Cases** | `batchnorm-critic-mode` `batchnorm-optimizer-stability` `continuous-rgb32-b256` `throughput-scaling` |
| **Primary View (`discriminator_stability`)** | Not in `discriminator_stability` |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | No (diagnostic / auxiliary cohort) |
| **Adapter / Domain** | `bn_dcgan_guard` / `bn_dcgan_rgb32_b256_diagnostic` |
| **Prior** | `mog` (sigma=`0.1`, standardize=`False`) |
| **Update Budget** | `1200` steps |

## What This Experiment Tests

Bounded hermetic 1,200-step diagnostic guard running a standard Radford 32x32 BatchNorm DCGAN (B=256, z_dim=128) on a continuous 10-class RGB image distribution (pooled 8x8 nn/med > 0.50) to catch BatchNorm critic train/eval collapse, optimizer period-2 limit cycles, natural-image quality regressions, and O(B^2) spatial-transport latency cliffs before promotion.

> [!NOTE]
> **Protocol notes:** Hermetic research_diagnostic task (qualification_input=false; 1,200 updates, B=256, seed 0, deterministic_orthogonal_ init, isolated RNGs, SHA-256 verified updates.jsonl latency trace, actual-training goal.gif).

## Laws & Edge Cases Targeted

- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **[Law 4.2: Pathology-Driven Spectral Conditioning](../LAWS_HUB.md#law4_2)** ([Law 4.2 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `batchnorm-critic-mode`, `batchnorm-optimizer-stability`, `continuous-rgb32-b256`, `throughput-scaling`

## Common Failure Mode

Switching active critic BatchNorm layers to eval() during the G step causes immediate critic collapse; firing O(B^2) pixel-space transport on continuous 32x32 images pushes B=256 latency above 38 ms/step and degrades quality; BatchNorm optimizer limit cycles or low-rank critic penalty inflation stall convergence.

## Visual Schematic

![Target and protocol schematic for bn_dcgan_rgb32_b256_diagnostic](assets/bn_dcgan_rgb32_b256_diagnostic.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.1, P=1024, z=128)"] --> G["Generator G(z)"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: bn_dcgan_rgb32_b256_diagnostic (1200 steps, 3 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.1`, standardize=`False`) |
| Update Budget | `1200` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=1024` / `z_dim=128` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Architecture | `family=radford_bn_dcgan32, image_size=32, channels=3, base_channels=64, generator_batchnorm=True, discriminator_batchnorm=True` |
| Integrity Guards | `finite_state`, `optimizer_roles`, `exact_optimizer_updates`, `mechanism_exercised`, `rng_isolation` |
| Assigned Views | `bn_dcgan_b256_diagnostic` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `finite_fraction` | `==` | `1.0` |
| `feature_fid` | `<=` | `55.0` |
| `pixel_diversity_ratio` | `>=` | `0.7` |
| `pixel_diversity_ratio` | `<=` | `1.3` |
| `class_coverage` | `>=` | `0.9` |
| `class_mass_tv` | `<=` | `0.25` |
| `spectral_l1` | `<=` | `0.04` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Atlas** | `atlas-particle-normal-v4` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v5` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Axiom](../families/axiom.md#cohort-cuda-42a130a78fc3) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v15` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view bn_dcgan_b256_diagnostic --candidate unified-v12 --tasks bn_dcgan_rgb32_b256_diagnostic
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/bn_dcgan_rgb32_b256_diagnostic.json](../../../configs/forge/tasks/bn_dcgan_rgb32_b256_diagnostic.json)
- Source code: [benchmarks/toy_audit/api_bn_dcgan_rgb32.py](../../../benchmarks/toy_audit/api_bn_dcgan_rgb32.py)
- Source code: [experiments/forge/bn_dcgan_guard.py](../../../experiments/forge/bn_dcgan_guard.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/bn_dcgan_rgb32_b256_diagnostic.json](../../../configs/forge/tasks/bn_dcgan_rgb32_b256_diagnostic.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/bn_dcgan_rgb32_b256_diagnostic.json](../../../configs/forge/task-documentation/bn_dcgan_rgb32_b256_diagnostic.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/api_bn_dcgan_rgb32.py](../../../benchmarks/toy_audit/api_bn_dcgan_rgb32.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/bn_dcgan_guard.py](../../../experiments/forge/bn_dcgan_guard.py) (AST symbols: `__doc__`)
