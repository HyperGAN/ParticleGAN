<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 1D Gaussian Stationary Hold & Target Shift Recovery (`gaussian1d_stability`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 3A](../LAWS_HUB.md#law3a) · [Law 4.4](../LAWS_HUB.md#law4_4) · [Law 4.5](../LAWS_HUB.md#law4_5) |
| **Edge Cases** | `non-stationary-shift` `settling-vs-tracking` `checkpoint-continuation` |
| **Primary View (`discriminator_stability`)** | Tier 2 (required) |
| **Four Laws Adherence Role** | law4 (anchor) |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `gaussian_stability` |
| **Prior** | `mog` (sigma=`0.1`, standardize=`False`) |
| **Update Budget** | `6000` steps |

## What This Experiment Tests

Continues the candidate's own passing `gaussian1d_smoke` checkpoint to test the Law 4.4 settling-versus-tracking dilemma: first hold N(2, 0.5^2) stably across stationary updates, then rapidly re-acquire a shifted target within deadline and hold it.

> [!NOTE]
> **Protocol notes:** Strict checkpoint continuation from `gaussian1d_smoke`; cannot pass without a valid passing smoke parent.

## Laws & Edge Cases Targeted

- **[Law 3A: Unconditional W1 Ray-Length & Screening Identifiability](../LAWS_HUB.md#law3a)** ([Law 3A theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 4.4: Trajectory Scale vs. Settling](../LAWS_HUB.md#law4_4)** ([Law 4.4 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **[Law 4.5: Coupled Minimax Contraction](../LAWS_HUB.md#law4_5)** ([Law 4.5 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `non-stationary-shift`, `settling-vs-tracking`, `checkpoint-continuation`

## Common Failure Mode

Recipes with aggressive LR annealing freeze and miss the shifted target, whereas recipes without settling damping oscillate around the stationary bell curve and fail hold checks.

**Historical audit diagnosis** (`reports/forge/problem-definitions.json`): BCAP and BISPF keep oscillating instead of settling. BISPF passes only 5 of 72 stationary checks, misses deadline re-acquisition and passes 0 of 24 shifted-hold checks. A one-time smoke pass does not establish stability.

- *Recommended research focus:* Find an update rule that settles when correct but responds after a shift, then verify the same recipe on ring16 and mode hold. Existing trade-offs do not prove the problem impossible.

## Visual Schematic

![Target and protocol schematic for gaussian1d_stability](assets/gaussian1d_stability.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.1, P=256, z=2)"] --> G["Generator G(z)"]
    G --> M["Stationary Hold -> Target Shift -> Deadline"]
    M --> D["Critic D(x)"]
    D --> Gate["Gate: gaussian_stability (6000 steps, 5 sustained) <- continues gaussian1d_smoke"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.1`, standardize=`False`) |
| Update Budget | `6000` steps |
| Batch Size | `128` |
| Particles / Latent Dim | `P=256` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Continuation Parent | [`gaussian1d_smoke`](gaussian1d_smoke.md) |
| Dependencies | [`gaussian1d_smoke`](gaussian1d_smoke.md) (`checkpoint`) |
| Integrity Guards | `finite_state`, `rng_isolation`, `mechanism_exercised`, `optimizer_roles` |
| Assigned Views | `bcap-develop-integration-deeper-diagnostic-v1` (T1 diagnostic), `bcap-projection-baseline-repair-diagnostic-v1` (T1 diagnostic), `discriminator_stability` (T2 required), `four-laws-adherence-v1` (T1 diagnostic), `lazy_cadence_coverage` (T1 diagnostic), `prior-ablation-full-suite-v1` (T1 diagnostic), `r4-occam-ablation-v1` (T1 diagnostic), `recipe-prior-refactor-k3p-diagnostic-v1` (T1 diagnostic) (+3 more) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `sample_count` | `>=` | `4096` |
| `finite_fraction` | `==` | `1.0` |
| `mean_error_sigma` | `<=` | `0.2` |
| `std_ratio` | `>=` | `0.8` |
| `std_ratio` | `<=` | `1.2` |
| `cdf_ks` | `<=` | `0.05` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **EMPF** | `empf-v8` | **PASS** | stationary hold, deadline reacquisition and shifted hold | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **RHBI** | `rhbi-v12` | **PASS** | stationary hold, deadline reacquisition and shifted hold | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | **PASS** | stationary hold, deadline reacquisition and shifted hold | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |
| **Atlas** | `atlas-particle-normal-v4` | **FAIL** | no compatible result | [Atlas](../families/atlas.md#cohort-cuda-bc3debd3fa5e) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** | stationary hold, deadline reacquisition and shifted hold | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** | stationary hold, deadline reacquisition and shifted hold | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `BLOCKED` | Skipped incompatible test: gaussian1d_stability: clean/live scoring has no declared policy | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | **FAIL** | no compatible result | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | **FAIL** | no compatible result | [K3P](../families/k3p.md#cohort-cuda-1bf9d7d34422) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | **FAIL** | no compatible result | [KA2](../families/ka2.md#cohort-cuda-1bf9d7d34422) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | **FAIL** | no compatible result | [R1/R2](../families/r1r2.md#cohort-cuda-1bf9d7d34422) | — |

### Four Laws Adherence Diagnostic Receipts (Separate Cohort)

*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*

| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |
| --- | --- | --- | :---: | ---: | ---: | --- | --- |
| `law4-energy-v11` | `unified` | NVIDIA L4 | **PASS** | — | 1.85 kJ (J-to-PASS) | cdf_ks=0.01991, finite_fraction=1, mean=3.014, mean_error_sigma=0.02763 | [f5b98ecfd28949c18e5876106cdcb3b4.json](../technique-receipts/f5b98ecfd28949c18e5876106cdcb3b4.json) |
| `unified-v11` | `unified` | NVIDIA L4 | **PASS** | — | 2.86 kJ (J-to-PASS) | cdf_ks=0.01486, finite_fraction=1, mean=3, mean_error_sigma=0.0008137 | [c04063912f6149e5910486262cbfa945.json](../technique-receipts/c04063912f6149e5910486262cbfa945.json) |
| `unified-v12` | `unified` | NVIDIA L4 | **PASS** | — | 2.13 kJ (J-to-PASS) | cdf_ks=0.01991, finite_fraction=1, mean=3.014, mean_error_sigma=0.02763 | [e54b50c8e1324c04a0ddc0cd12d47a3c.json](../technique-receipts/e54b50c8e1324c04a0ddc0cd12d47a3c.json) |
| `atlas-particle-normal-v4` | `atlas` | NVIDIA L4 | **FAIL** | — | 6.07 kJ | cdf_ks=0.06111, finite_fraction=1, mean=3.01, mean_error_sigma=0.01927 | [ea9f5f48b804429193c4109e442df60a.json](../technique-receipts/ea9f5f48b804429193c4109e442df60a.json) |
| `bispf-optimized-quintic-v1` | `bispf` | NVIDIA L4 | **FAIL** | — | 5.06 kJ | cdf_ks=0.1212, finite_fraction=1, mean=3.117, mean_error_sigma=0.2346 | [e8cbdd97b4a84993857ff7fb2c96415b.json](../technique-receipts/e8cbdd97b4a84993857ff7fb2c96415b.json) |
| `empf-v8` | `empf` | NVIDIA L4 | **FAIL** | — | 5.46 kJ | cdf_ks=0.0333, finite_fraction=1, mean=3.039, mean_error_sigma=0.07887 | [96032fc843ec4575b5ded52104567ec2.json](../technique-receipts/96032fc843ec4575b5ded52104567ec2.json) |
| `minimal-v2` | `minimal` | NVIDIA L4 | **FAIL** | — | 7.49 kJ | cdf_ks=0.02766, finite_fraction=1, mean=3.003, mean_error_sigma=0.005378 | [4c4f915cef924dc2939f8c9069d332da.json](../technique-receipts/4c4f915cef924dc2939f8c9069d332da.json) |
| `recipe-prior-refactor-bcap-pure-v1` | `bcap-dualnorm` | NVIDIA L4 | **FAIL** | — | 3.37 kJ | cdf_ks=0.1585, finite_fraction=1, mean=2.835, mean_error_sigma=0.33 | [c888451a157a43e48b9a61978123e6cd.json](../technique-receipts/c888451a157a43e48b9a61978123e6cd.json) |
| `recipe-prior-refactor-k3p-v1` | `k3p` | NVIDIA L4 | **FAIL** | — | 3.86 kJ | cdf_ks=0.02068, finite_fraction=1, mean=2.988, mean_error_sigma=0.02462 | [ca4e041e0aba4f9085d1be677e5c3055.json](../technique-receipts/ca4e041e0aba4f9085d1be677e5c3055.json) |
| `recipe-prior-refactor-ka2-v1` | `ka2` | NVIDIA L4 | **FAIL** | — | 3.66 kJ | cdf_ks=0.0157, finite_fraction=1, mean=2.995, mean_error_sigma=0.01058 | [81e918b846094db78eb81d9536be25e3.json](../technique-receipts/81e918b846094db78eb81d9536be25e3.json) |
| `recipe-prior-refactor-r1r2-v1` | `r1r2` | NVIDIA L4 | **FAIL** | — | 2.46 kJ | cdf_ks=0.07533, finite_fraction=1, mean=2.917, mean_error_sigma=0.1662 | [ce4ebb8f02f5417fb4171ffa181ea93f.json](../technique-receipts/ce4ebb8f02f5417fb4171ffa181ea93f.json) |
| `recipe-prior-refactor-release07-gan-v3-v1` | `release07-gan-v3` | NVIDIA L4 | **FAIL** | — | 2.30 kJ | cdf_ks=0.1221, finite_fraction=1, mean=2.948, mean_error_sigma=0.1036 | [e8c1ae94430c41a580e5b06a83fe28a8.json](../technique-receipts/e8c1ae94430c41a580e5b06a83fe28a8.json) |
| `rhbi-v12` | `rhbi` | NVIDIA L4 | **FAIL** | — | 7.81 kJ | cdf_ks=0.05258, finite_fraction=1, mean=3.007, mean_error_sigma=0.01314 | [dda369ab998841d8ba72c46bf28f5fc0.json](../technique-receipts/dda369ab998841d8ba72c46bf28f5fc0.json) |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks gaussian1d_stability
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/gaussian1d_stability.json](../../../configs/forge/tasks/gaussian1d_stability.json)
- Source code: [benchmarks/toy_audit/gaussian1d_quality.py](../../../benchmarks/toy_audit/gaussian1d_quality.py)
- Source code: [experiments/forge/gaussian_tasks.py](../../../experiments/forge/gaussian_tasks.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/gaussian1d_stability.json](../../../configs/forge/tasks/gaussian1d_stability.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/gaussian1d_stability.json](../../../configs/forge/task-documentation/gaussian1d_stability.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/gaussian1d_quality.py](../../../benchmarks/toy_audit/gaussian1d_quality.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/gaussian_tasks.py](../../../experiments/forge/gaussian_tasks.py) (AST symbols: `__doc__`)
