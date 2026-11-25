<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 16-Mode Ring Continuous Joint BiGAN (`ring16_bigan_continuous`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 1](../LAWS_HUB.md#law1) · [Law 3B](../LAWS_HUB.md#law3b) |
| **Edge Cases** | `bidirectional-encoder` `ring-mixture` `mode-consistency` |
| **Primary View (`discriminator_stability`)** | Tier 4 (held) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `ring16_bigan_continuous` |
| **Prior** | `mog` (sigma=`0.1`, standardize=`False`) |
| **Update Budget** | `4000` steps |

## What This Experiment Tests

Trains a joint BiGAN (G(z), E(x), D(x, z)) on a 16-mode Gaussian ring with MoGParticlePrior(sigma=0.1), requiring both 16/16 generative mode recovery and high encoder-decoder mode consistency (>= 0.98) and reconstruction accuracy.

> [!NOTE]
> **Protocol notes:** Tier-4 held stretch task (4,000 updates).

## Laws & Edge Cases Targeted

- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **[Law 3B: Conditional Fiber & Stratum Identifiability](../LAWS_HUB.md#law3b)** ([Law 3B theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **Edge-case tags:** `bidirectional-encoder`, `ring-mixture`, `mode-consistency`

## Common Failure Mode

Joint critic D(x, z) fails to couple encoder and generator distributions, leading to accurate marginal samples but scrambled reconstructions (recon_rmse > 0.12 or mode_consistency < 0.98).

**Historical audit diagnosis** (`reports/forge/problem-definitions.json`): In the standalone seed-0 `bcap` public-API run without cycle-consistency or kernel-augmented critics, the joint game discovers all 16/16 modes (`modes = 16`) and achieves high round-trip mode consistency (`mode_consistency = 0.9951 >= 0.98`), but within-mode variance remains too broad (`hq = 0.238` vs. `>= 0.85`, `recon_rmse = 0.220` vs. `<= 0.12`).

- *Recommended research focus:* Extend kernel-augmented joint `(x, z)` curvature regularization and cycle-consistency conditioning to the remaining non-kernel trainer families so all selected families contract within-mode variance without losing 16-mode coverage.

## Visual Schematic

![Target and protocol schematic for ring16_bigan_continuous](assets/ring16_bigan_continuous.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.1, P=256, z=2)"] --> G["Generator G(z)"]
    X["Real Data x"] --> E["Encoder E(x)"]
    G --> D["Joint Critic D(x, z)"]
    E --> D
    X --> D
    D --> Gate["Gate: ring16_bigan_continuous (4000 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.1`, standardize=`False`) |
| Update Budget | `4000` steps |
| Batch Size | `128` |
| Particles / Latent Dim | `P=256` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Assigned Views | `discriminator_stability` (T4 held), `prior-ablation-full-suite-v1` (T1 diagnostic), `r3-tier4-frontier-holdout-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `modes` | `>=` | `16` |
| `hq` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.15` |
| `recon_rmse` | `<=` | `0.12` |
| `mode_consistency` | `>=` | `0.98` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **K3P** | `recipe-prior-refactor-k3p-v1` | **PASS** | No recorded result for this selected configuration and source. | [K3P](../families/k3p.md#cohort-cuda-1bf9d7d34422) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | **PASS** | No recorded result for this selected configuration and source. | [KA2](../families/ka2.md#cohort-cuda-1bf9d7d34422) | — |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | ring16_bigan_continuous: unsupported vector target declaration | [Atlas](../families/atlas.md#cohort-cuda-bc3debd3fa5e) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | **FAIL** | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | ring16_bigan_continuous: no registered public Forge adapter for this stretch host; declare | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | **FAIL** | No recorded result for this selected configuration and source. | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | **FAIL** | No recorded result for this selected configuration and source. | [R1/R2](../families/r1r2.md#cohort-cuda-1bf9d7d34422) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | ring16_bigan_continuous: no registered public Forge adapter for this stretch host; declare | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `BLOCKED` | ring16_bigan_continuous: no registered public Forge adapter for this stretch host; declare | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks ring16_bigan_continuous
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/ring16_bigan_continuous.json](../../../configs/forge/tasks/ring16_bigan_continuous.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/ring16_bigan_continuous.json](../../../configs/forge/tasks/ring16_bigan_continuous.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/ring16_bigan_continuous.json](../../../configs/forge/task-documentation/ring16_bigan_continuous.json)
