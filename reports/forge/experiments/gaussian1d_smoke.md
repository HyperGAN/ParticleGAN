<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 1D Gaussian Acquisition Smoke (`gaussian1d_smoke`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 2](../LAWS_HUB.md#law2) · [Law 4.4](../LAWS_HUB.md#law4_4) |
| **Edge Cases** | `scalar-1d` `acquisition-smoke` `checkpoint-producer` |
| **Primary View (`discriminator_stability`)** | Tier 1 (required) |
| **Four Laws Adherence Role** | prerequisite (prerequisite) |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `transfer_vector` / `gaussian_smoke` |
| **Prior** | `mog` (sigma=`0.1`, standardize=`False`) |
| **Update Budget** | `1000` steps |

## What This Experiment Tests

Minimal sanity gate asking whether public training acquires a single scalar Gaussian N(2, 0.5^2) from random initialization within 1,000 updates, matching its mean, standard deviation, and full CDF (KS <= 0.05).

> [!NOTE]
> **Protocol notes:** Produces the saved checkpoint consumed by `gaussian1d_stability`; requires independent same-state confirmation.

## Laws & Edge Cases Targeted

- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **[Law 4.4: Trajectory Scale vs. Settling](../LAWS_HUB.md#law4_4)** ([Law 4.4 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `scalar-1d`, `acquisition-smoke`, `checkpoint-producer`

## Common Failure Mode

Broken critic scaling, NaN divergence, or collapsed variance prevents even a unimodal 1D Gaussian from converging within bounds.

## Visual Schematic

![Target and protocol schematic for gaussian1d_smoke](assets/gaussian1d_smoke.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.1, P=256, z=2)"] --> G["Generator G(z)"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: gaussian_smoke (1000 steps, clean)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.1`, standardize=`False`) |
| Update Budget | `1000` steps |
| Batch Size | `128` |
| Particles / Latent Dim | `P=256` / `z_dim=2` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Integrity Guards | `finite_state`, `rng_isolation`, `mechanism_exercised`, `optimizer_roles` |
| Assigned Views | `bcap-develop-integration-deeper-diagnostic-v1` (T1 diagnostic), `bcap-projection-baseline-repair-diagnostic-v1` (T1 diagnostic), `discriminator_stability` (T1 required), `four-laws-adherence-v1` (T1 diagnostic), `lazy_cadence_coverage` (T1 diagnostic), `prior-ablation-full-suite-v1` (T1 diagnostic), `r4-occam-ablation-v1` (T1 diagnostic), `recipe-prior-refactor-k3p-diagnostic-v1` (T1 diagnostic) (+3 more) |

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
| **Atlas** | `atlas-particle-normal-v4` | **PASS** | any scheduled full pass with independent same-state confirmation | [Atlas](../families/atlas.md#cohort-cuda-bc3debd3fa5e) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **PASS** | any scheduled full pass with independent same-state confirmation | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | **PASS** | any scheduled full pass with independent same-state confirmation | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | **PASS** | any scheduled full pass with independent same-state confirmation | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | **PASS** | any scheduled full pass with independent same-state confirmation | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | **PASS** | any scheduled full pass with independent same-state confirmation | [K3P](../families/k3p.md#cohort-cuda-1bf9d7d34422) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | **PASS** | any scheduled full pass with independent same-state confirmation | [KA2](../families/ka2.md#cohort-cuda-1bf9d7d34422) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | **PASS** | any scheduled full pass with independent same-state confirmation | [R1/R2](../families/r1r2.md#cohort-cuda-1bf9d7d34422) | — |
| **RHBI** | `rhbi-v12` | **PASS** | any scheduled full pass with independent same-state confirmation | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | **PASS** | any scheduled full pass with independent same-state confirmation | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |
| **E22** | `e22` | `BLOCKED` | Skipped incompatible test: gaussian1d_smoke: clean/live scoring has no declared policy-con | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |

### Four Laws Adherence Diagnostic Receipts (Separate Cohort)

*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*

| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |
| --- | --- | --- | :---: | ---: | ---: | --- | --- |
| `atlas-particle-normal-v4` | `atlas` | NVIDIA L4 | **PASS** | — | 200 J (J-to-PASS) | cdf_ks=0.06777, finite_fraction=1, mean=1.975, mean_error_sigma=0.05042 | [c0682a1625074e3b85a20e2057cf37f9.json](../technique-receipts/c0682a1625074e3b85a20e2057cf37f9.json) |
| `bispf-optimized-quintic-v1` | `bispf` | NVIDIA L4 | **PASS** | — | 127 J (J-to-PASS) | cdf_ks=0.1371, finite_fraction=1, mean=1.917, mean_error_sigma=0.1657 | [b711be6782954f34995912bc5b554851.json](../technique-receipts/b711be6782954f34995912bc5b554851.json) |
| `empf-v8` | `empf` | NVIDIA L4 | **PASS** | — | 973 J (J-to-PASS) | cdf_ks=0.06861, finite_fraction=1, mean=2.065, mean_error_sigma=0.1292 | [e402e6d7695d4d1e9c1335bb9e5a19cc.json](../technique-receipts/e402e6d7695d4d1e9c1335bb9e5a19cc.json) |
| `law4-energy-v11` | `unified` | NVIDIA L4 | **PASS** | — | 570 J (J-to-PASS) | cdf_ks=0.0282, finite_fraction=1, mean=2.01, mean_error_sigma=0.02015 | [49c68c744d8a4057a38446b73d7099d5.json](../technique-receipts/49c68c744d8a4057a38446b73d7099d5.json) |
| `minimal-v2` | `minimal` | NVIDIA L4 | **PASS** | — | 398 J (J-to-PASS) | cdf_ks=0.01744, finite_fraction=1, mean=1.999, mean_error_sigma=0.00282 | [d58073325f27494dab1a449a987e650f.json](../technique-receipts/d58073325f27494dab1a449a987e650f.json) |
| `recipe-prior-refactor-bcap-pure-v1` | `bcap-dualnorm` | NVIDIA L4 | **PASS** | — | 488 J (J-to-PASS) | cdf_ks=0.2416, finite_fraction=1, mean=1.743, mean_error_sigma=0.5132 | [ec9dc1e31a76463dab66024aa3d1a89e.json](../technique-receipts/ec9dc1e31a76463dab66024aa3d1a89e.json) |
| `recipe-prior-refactor-k3p-v1` | `k3p` | NVIDIA L4 | **PASS** | — | 131 J (J-to-PASS) | cdf_ks=0.02049, finite_fraction=1, mean=2.007, mean_error_sigma=0.01363 | [0c42b43343b14db5b72aa51b6c1753c0.json](../technique-receipts/0c42b43343b14db5b72aa51b6c1753c0.json) |
| `recipe-prior-refactor-ka2-v1` | `ka2` | NVIDIA L4 | **PASS** | — | 105 J (J-to-PASS) | cdf_ks=0.03276, finite_fraction=1, mean=2.027, mean_error_sigma=0.05334 | [5e88d9a8052e4c0ab68f45980e2ead23.json](../technique-receipts/5e88d9a8052e4c0ab68f45980e2ead23.json) |
| `recipe-prior-refactor-r1r2-v1` | `r1r2` | NVIDIA L4 | **PASS** | — | 342 J (J-to-PASS) | cdf_ks=0.03312, finite_fraction=1, mean=2.035, mean_error_sigma=0.06948 | [6270e3b5f6714fa2aec0af80c3ac3006.json](../technique-receipts/6270e3b5f6714fa2aec0af80c3ac3006.json) |
| `recipe-prior-refactor-release07-gan-v3-v1` | `release07-gan-v3` | NVIDIA L4 | **PASS** | — | 124 J (J-to-PASS) | cdf_ks=0.08383, finite_fraction=1, mean=2.088, mean_error_sigma=0.1764 | [da02252f834246b9ad60d33337d190e9.json](../technique-receipts/da02252f834246b9ad60d33337d190e9.json) |
| `rhbi-v12` | `rhbi` | NVIDIA L4 | **PASS** | — | 395 J (J-to-PASS) | cdf_ks=0.04817, finite_fraction=1, mean=1.999, mean_error_sigma=0.00162 | [095ee59ff9b54504bfb9c2ce161a7d78.json](../technique-receipts/095ee59ff9b54504bfb9c2ce161a7d78.json) |
| `unified-v11` | `unified` | NVIDIA L4 | **PASS** | — | 670 J (J-to-PASS) | cdf_ks=0.02743, finite_fraction=1, mean=2.022, mean_error_sigma=0.04356 | [95a4722c588c485b9d830fb70f6a1da1.json](../technique-receipts/95a4722c588c485b9d830fb70f6a1da1.json) |
| `unified-v12` | `unified` | NVIDIA L4 | **PASS** | — | 563 J (J-to-PASS) | cdf_ks=0.0282, finite_fraction=1, mean=2.01, mean_error_sigma=0.02015 | [66b628d15c3c47e6b90578f8ca4f722b.json](../technique-receipts/66b628d15c3c47e6b90578f8ca4f722b.json) |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks gaussian1d_smoke
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/gaussian1d_smoke.json](../../../configs/forge/tasks/gaussian1d_smoke.json)
- Source code: [benchmarks/toy_audit/gaussian1d_quality.py](../../../benchmarks/toy_audit/gaussian1d_quality.py)
- Source code: [experiments/forge/gaussian_tasks.py](../../../experiments/forge/gaussian_tasks.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/gaussian1d_smoke.json](../../../configs/forge/tasks/gaussian1d_smoke.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/gaussian1d_smoke.json](../../../configs/forge/task-documentation/gaussian1d_smoke.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/gaussian1d_quality.py](../../../benchmarks/toy_audit/gaussian1d_quality.py) (AST symbols: `__doc__`)
- **Read-only Python AST source:** [experiments/forge/gaussian_tasks.py](../../../experiments/forge/gaussian_tasks.py) (AST symbols: `__doc__`)
