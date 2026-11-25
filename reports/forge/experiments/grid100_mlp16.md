<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 100-Mode Grid with Deep 16-Layer Skip-Free MLP (`grid100_mlp16`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 4.2](../LAWS_HUB.md#law4_2) · [Law 1](../LAWS_HUB.md#law1) |
| **Edge Cases** | `deep-spectral-contraction` `16-layer-mlp` `100-modes` |
| **Primary View (`discriminator_stability`)** | Tier 4 (held) |
| **Four Laws Adherence Role** | law4 (anchor) |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `native100` / `native_accuracy` |
| **Prior** | `mog` (sigma=`0.025`, standardize=`False`) |
| **Update Budget** | `7000` steps |

## What This Experiment Tests

Replaces the shallow generator on `grid100` with a deep 16-layer skip-free MLP to isolate Law 4.2 (Pathology-Driven Spectral Conditioning): ordinary optimizers suffer exponential singular-value collapse across 16 layers.

> [!NOTE]
> **Protocol notes:** Tier-4 held benchmark and Four Laws Adherence Law 4 anchor.

## Laws & Edge Cases Targeted

- **[Law 4.2: Pathology-Driven Spectral Conditioning](../LAWS_HUB.md#law4_2)** ([Law 4.2 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **[Law 1: Representational Geometry](../LAWS_HUB.md#law1)** ([Law 1 theory](../../../docs/four-laws.md#law-1--representational-geometry)) — Can the system represent the required probability measure efficiently and faithfully?
- **Edge-case tags:** `deep-spectral-contraction`, `16-layer-mlp`, `100-modes`

## Common Failure Mode

Without spectral equilibration (e.g., polar/Newton-Schulz conditioning), gradients vanish or explode through 16 skip-free linear layers and the generator collapses onto a handful of modes.

**Historical audit diagnosis** (`reports/forge/problem-definitions.json`): Increasing generator depth from 3 to 16 hidden layers contracts initial output spread by ~156x and attenuates first-layer and particle-prior gradients by ~73x at step 1. In the standalone seed-0 `k3p` run, the 16-layer generator collapses to 0/100 modes (`output_sigma = 0.029`, `mass_tv = 0.99`).

- *Recommended research focus:* Pair layer-wise gradient normalization (such as `bcap-nsgda-layer` or `bcap-dualnorm`) with balanced prior-to-generator step scaling so early updates do not collapse the 16-layer Jacobian before particles separate across the 10x10 grid.

## Visual Schematic

![Target and protocol schematic for grid100_mlp16](assets/grid100_mlp16.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.025, P=20000, z=2)"] --> G["Generator: 16-Layer Skip-Free MLP"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: native_accuracy (7000 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.025`, standardize=`False`) |
| Update Budget | `7000` steps |
| Batch Size | `2048` |
| Particles / Latent Dim | `P=20000` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Architecture | `hidden=128, layers=16, fourier=2` |
| Dependencies | [`grid100`](grid100.md) (`gate`) |
| Assigned Views | `discriminator_stability` (T4 held), `four-laws-adherence-v1` (T1 diagnostic), `r3-tier4-frontier-holdout-v1` (T1 diagnostic), `r4-occam-ablation-v1` (T1 diagnostic), `t4-strange-architectures-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `min_samples` | `>=` | `20000` |
| `min_modes` | `>=` | `100` |
| `min_hq_mode_mass` | `>=` | `0.005` |
| `min_precision` | `>=` | `0.75` |
| `max_mass_tv` | `<=` | `0.1` |
| `max_mode_mass` | `<=` | `0.02` |
| `min_cov_eig_ratio` | `>=` | `0.4` |
| `max_cov_eig_ratio` | `<=` | `1.7` |
| `min_radial_median_ratio` | `>=` | `0.65` |
| `max_radial_median_ratio` | `<=` | `1.4` |
| `all_finite` | `==` | `True` |
| `accuracy.mass_tv` | `<=` | `0.06` |
| `accuracy.center_rms_sigma` | `<=` | `0.2` |
| `accuracy.abs_cov_trace_bias` | `<=` | `0.1` |
| `accuracy.radial_ks` | `<=` | `0.04` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | native model variants require an explicit profile; grid100_mlp16: native model variants re | [Atlas](../families/atlas.md#cohort-cuda-bc3debd3fa5e) | — |
| **Axiom** | `axiom-v4` | `NOT MEASURED` | no compatible result | [Axiom](../families/axiom.md#cohort-cuda-daa283c8a640) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `BLOCKED` | native model variants require an explicit profile; grid100_mlp16: native model variants re | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [K3P](../families/k3p.md#cohort-cuda-1bf9d7d34422) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [KA2](../families/ka2.md#cohort-cuda-1bf9d7d34422) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [R1/R2](../families/r1r2.md#cohort-cuda-1bf9d7d34422) | — |
| **RHBI** | `rhbi-v12` | `BLOCKED` | native model variants require an explicit profile; grid100_mlp16: native model variants re | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `NOT MEASURED` | no compatible result | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

### Four Laws Adherence Diagnostic Receipts (Separate Cohort)

*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*

| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |
| --- | --- | --- | :---: | ---: | ---: | --- | --- |
| `law4-energy-v11` | `unified` | NVIDIA L4 | **PASS** | — | 38.85 kJ | abs_cov_trace_bias=0.08094, accuracy_score=0.6505, center_max_sigma=0.139, center_rms_sigma=0.06468 | [d8337039f6ba4a45af98cea2404bbab7.json](../technique-receipts/d8337039f6ba4a45af98cea2404bbab7.json) |
| `unified-v11` | `unified` | NVIDIA L4 | **PASS** | — | 44.59 kJ | abs_cov_trace_bias=0.07649, accuracy_score=0.6221, center_max_sigma=0.134, center_rms_sigma=0.06464 | [9c29dc49af2b447caedcf16babea49ac.json](../technique-receipts/9c29dc49af2b447caedcf16babea49ac.json) |
| `unified-v12` | `unified` | NVIDIA L4 | **PASS** | — | 42.31 kJ | abs_cov_trace_bias=0.07521, accuracy_score=0.6223, center_max_sigma=0.1602, center_rms_sigma=0.06431 | [462dd0d4625b4578a7b3eb6567382416.json](../technique-receipts/462dd0d4625b4578a7b3eb6567382416.json) |
| `axiom-v4` | `axiom` | NVIDIA RTX A6000 | **FAIL** | — | 68.50 kJ | abs_cov_trace_bias=0.08757, accuracy_score=0.6385, center_max_sigma=0.1565, center_rms_sigma=0.06759 | [readout-9066cb4353eb435be7819b0c.json](../records/readout-9066cb4353eb435be7819b0c.json) |
| `minimal-v2` | `minimal` | NVIDIA L4 | **FAIL** | — | 35.78 kJ | mass_tv=0.4852, n=100000, precision=0.03929, precision_gap=0.9496 | [35df9474e2784e8e830160b726085286.json](../technique-receipts/35df9474e2784e8e830160b726085286.json) |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks grid100_mlp16
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/grid100_mlp16.json](../../../configs/forge/tasks/grid100_mlp16.json)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/grid100_mlp16.json](../../../configs/forge/tasks/grid100_mlp16.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/grid100_mlp16.json](../../../configs/forge/task-documentation/grid100_mlp16.json)
