<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Law 4.2 Probe: Swapped Depth Asymmetry (Shallow G vs Deep 16L Critic) (`fl_law42_swap_deep_critic_16l`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 4.2](../LAWS_HUB.md#law4_2) · [Law 2](../LAWS_HUB.md#law2) |
| **Edge Cases** | `deep-critic-16l` `deep-spectral-contraction` `ring-mixture` |
| **Primary View (`discriminator_stability`)** | Not in `discriminator_stability` |
| **Four Laws Adherence Role** | law4 (four_laws_probe) |
| **Fixed Full-Suite Denominator** | No (diagnostic / auxiliary cohort) |
| **Adapter / Domain** | `four_laws` / `fl_law42_swap_deep_critic_16l` |
| **Prior** | `mog` (sigma=`0.08`, standardize=`False`) |
| **Update Budget** | `800` steps |

## What This Experiment Tests

Inverts depth asymmetry onto the critic: pairs a shallow 3-layer MLP generator with a deep 16-layer skip-free LeakyReLU critic (fourier=0) on an 8-mode 2D ring. Tests whether spectral conditioning rescues deep critics as well as deep generators.

> [!NOTE]
> **Protocol notes:** Four Laws Adherence diagnostic probe (800 updates, 8-mode ring R=2.5, valley_fraction <= 0.05).

## Laws & Edge Cases Targeted

- **[Law 4.2: Pathology-Driven Spectral Conditioning](../LAWS_HUB.md#law4_2)** ([Law 4.2 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **Edge-case tags:** `deep-critic-16l`, `deep-spectral-contraction`, `ring-mixture`

## Common Failure Mode

Without Law 4.2 spectral conditioning on D, the 16-layer critic's input Jacobian contracts to ~5e-7, leaving one-sided upper slope caps inactive and starving the generator of any useful force field.

## Visual Schematic

![Target and protocol schematic for fl_law42_swap_deep_critic_16l](assets/fl_law42_swap_deep_critic_16l.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.08, P=256, z=4)"] --> G["Generator G(z)"]
    G --> D["Critic: 16-Layer Skip-Free LeakyReLU (F0)"]
    D --> Gate["Gate: fl_law42_swap_deep_critic_16l (800 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.08`, standardize=`False`) |
| Update Budget | `800` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=256` / `z_dim=4` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Architecture | `generator_family=simple_mlp, g_hidden=64, g_layers=3, out_dim=2, discriminator_family=deep_skipfree_mlp, d_hidden=64, d_layers=16, fourier=0` |
| Integrity Guards | `finite_state`, `optimizer_roles`, `exact_optimizer_updates`, `mechanism_exercised`, `rng_isolation` |
| Assigned Views | `four-laws-adherence-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `finite_fraction` | `==` | `1.0` |
| `modes` | `==` | `8` |
| `hq` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.12` |
| `valley_fraction` | `<=` | `0.05` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Atlas** | `atlas-particle-normal-v4` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Atlas](../families/atlas.md#cohort-cuda-bc3debd3fa5e) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [K3P](../families/k3p.md#cohort-cuda-1bf9d7d34422) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [KA2](../families/ka2.md#cohort-cuda-1bf9d7d34422) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [R1/R2](../families/r1r2.md#cohort-cuda-1bf9d7d34422) | — |
| **RHBI** | `rhbi-v12` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

### Four Laws Adherence Diagnostic Receipts (Separate Cohort)

*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*

| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |
| --- | --- | --- | :---: | ---: | ---: | --- | --- |
| `minimal-v2` | `minimal` | NVIDIA L4 | **PASS** | `+0.000` | 2.03 kJ (J-to-PASS) | diag_critic_r_stable=1.078, diag_valley_sep=0.5391, finite_fraction=1, hq=1 | [be358e85f54f4fe7865713de3a8fbb0c.json](../technique-receipts/be358e85f54f4fe7865713de3a8fbb0c.json) |
| `recipe-prior-refactor-k3p-v1` | `k3p` | NVIDIA L4 | **PASS** | `+0.000` | 1.51 kJ (J-to-PASS) | diag_critic_r_stable=1.153, diag_valley_sep=0.7793, finite_fraction=1, hq=0.8997 | [55224c993b964e7db2d5c3c44aaf617a.json](../technique-receipts/55224c993b964e7db2d5c3c44aaf617a.json) |
| `recipe-prior-refactor-ka2-v1` | `ka2` | NVIDIA L4 | **PASS** | `+0.000` | 1.59 kJ (J-to-PASS) | diag_critic_r_stable=1.101, diag_valley_sep=0.793, finite_fraction=1, hq=0.896 | [ea52a6624e5843c1bfee55939257a244.json](../technique-receipts/ea52a6624e5843c1bfee55939257a244.json) |
| `unified-v11` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.47 kJ (J-to-PASS) | diag_critic_r_stable=1.037, diag_valley_sep=0.4512, finite_fraction=1, hq=1 | [25e26d6d20ca4527a6262888c42af5f6.json](../technique-receipts/25e26d6d20ca4527a6262888c42af5f6.json) |
| `unified-v12` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.09 kJ (J-to-PASS) | diag_critic_r_stable=1.008, diag_valley_sep=0.4961, finite_fraction=1, hq=1 | [8a5971f1c22c4d689749572fb2512c5d.json](../technique-receipts/8a5971f1c22c4d689749572fb2512c5d.json) |
| `unified-v13` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.01 kJ (J-to-PASS) | diag_critic_r_stable=1.008, diag_valley_sep=0.4961, finite_fraction=1, hq=1 | [dfc82b3daec543fc91296d6939ee5b5d.json](../technique-receipts/dfc82b3daec543fc91296d6939ee5b5d.json) |
| `unified-v14` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.13 kJ (J-to-PASS) | diag_critic_r_stable=1.008, diag_valley_sep=0.4961, finite_fraction=1, hq=1 | [bbe5537d119f4cfc913753d0865f2e11.json](../technique-receipts/bbe5537d119f4cfc913753d0865f2e11.json) |
| `recipe-prior-refactor-release07-gan-v3-v1` | `release07-gan-v3` | NVIDIA L4 | **FAIL** | `-1.388` | 1.14 kJ | diag_critic_r_stable=1.056, diag_valley_sep=0.5293, finite_fraction=1, hq=0.1277 | [eb0e06fbfc364c9b8033c58f2f157c68.json](../technique-receipts/eb0e06fbfc364c9b8033c58f2f157c68.json) |
| `bispf-optimized-quintic-v1` | `bispf` | NVIDIA L4 | **FAIL** | `-5.250` | 2.99 kJ | diag_critic_r_stable=1, diag_valley_sep=0.5117, finite_fraction=1, hq=0 | [6623278427d54893a344ec47d2216fe0.json](../technique-receipts/6623278427d54893a344ec47d2216fe0.json) |
| `empf-v8` | `empf` | NVIDIA L4 | **FAIL** | `-5.250` | 2.95 kJ | diag_critic_r_stable=1, diag_valley_sep=0.5195, finite_fraction=1, hq=0 | [106c47182e7145dc9b86d77bca688753.json](../technique-receipts/106c47182e7145dc9b86d77bca688753.json) |
| `recipe-prior-refactor-bcap-pure-v1` | `bcap-dualnorm` | NVIDIA L4 | **FAIL** | `-5.250` | 2.46 kJ | diag_critic_r_stable=1, diag_valley_sep=0.5195, finite_fraction=1, hq=0 | [f8ac5538a24f42429c582914cabff018.json](../technique-receipts/f8ac5538a24f42429c582914cabff018.json) |
| `rhbi-v12` | `rhbi` | NVIDIA L4 | **FAIL** | `-5.250` | 2.37 kJ | diag_critic_r_stable=1, diag_valley_sep=0.541, finite_fraction=1, hq=0 | [723a020b07c7445799fafd2afe9e7964.json](../technique-receipts/723a020b07c7445799fafd2afe9e7964.json) |
| `recipe-prior-refactor-r1r2-v1` | `r1r2` | NVIDIA L4 | **FAIL** | `-6.292` | 1.18 kJ | diag_critic_r_stable=1.01, diag_valley_sep=0.5156, finite_fraction=1, hq=0 | [60615b5ef5324777a3f0e4ea389235ab.json](../technique-receipts/60615b5ef5324777a3f0e4ea389235ab.json) |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view four-laws-adherence-v1 --candidate unified-v12 --tasks fl_law42_swap_deep_critic_16l
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/fl_law42_swap_deep_critic_16l.json](../../../configs/forge/tasks/fl_law42_swap_deep_critic_16l.json)
- Source code: [benchmarks/toy_audit/api_four_laws_4.py](../../../benchmarks/toy_audit/api_four_laws_4.py)
- Source code: [experiments/forge/four_laws.py](../../../experiments/forge/four_laws.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/fl_law42_swap_deep_critic_16l.json](../../../configs/forge/tasks/fl_law42_swap_deep_critic_16l.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/fl_law42_swap_deep_critic_16l.json](../../../configs/forge/task-documentation/fl_law42_swap_deep_critic_16l.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/api_four_laws_4.py](../../../benchmarks/toy_audit/api_four_laws_4.py) (AST symbols: `__doc__`, `THRESHOLDS`, `SPECS`)
- **Read-only Python AST source:** [experiments/forge/four_laws.py](../../../experiments/forge/four_laws.py) (AST symbols: `__doc__`)
