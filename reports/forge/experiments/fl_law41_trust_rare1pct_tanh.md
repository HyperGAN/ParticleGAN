<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Law 4.1/4.3 Probe: 1% Minority Mode Inside Saturating 3*tanh Rail (`fl_law41_trust_rare1pct_tanh`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 4.1](../LAWS_HUB.md#law4_1) · [Law 4.3](../LAWS_HUB.md#law4_3) |
| **Edge Cases** | `rare-mode` `tanh-saturation` `pushforward-trust` |
| **Primary View (`discriminator_stability`)** | Not in `discriminator_stability` |
| **Four Laws Adherence Role** | law4 (four_laws_probe) |
| **Fixed Full-Suite Denominator** | No (diagnostic / auxiliary cohort) |
| **Adapter / Domain** | `four_laws` / `fl_law41_trust_rare1pct_tanh` |
| **Prior** | `mog` (sigma=`0.08`, standardize=`False`) |
| **Update Budget** | `800` steps |

## What This Experiment Tests

Places 4 major modes (99% mass) at (+-1.5, +-1.5) and a single 1% minority mode at (0.0, 2.85) inside the saturating tail of a `3.0 * tanh(...)` generator, grading strictly on generated outputs (5/5 modes, rare_mode_recall >= 0.45).

> [!NOTE]
> **Protocol notes:** Four Laws Adherence diagnostic probe (800 updates, P=512, B=256, 4L saturating tanh generator).

## Laws & Edge Cases Targeted

- **[Law 4.1: Functional Realization](../LAWS_HUB.md#law4_1)** ([Law 4.1 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **[Law 4.3: Displacement & Kreiss Control](../LAWS_HUB.md#law4_3)** ([Law 4.3 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `rare-mode`, `tanh-saturation`, `pushforward-trust`

## Common Failure Mode

Global pushforward trust regions calibrated by the 99% bulk modes suppress the steep pre-activation step needed to reach the 1% rail-adjacent mode, or vanishing tanh gradients starve it completely.

## Visual Schematic

![Target and protocol schematic for fl_law41_trust_rare1pct_tanh](assets/fl_law41_trust_rare1pct_tanh.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.08, P=512, z=4)"] --> G["Generator: 4L MLP + 3.0*tanh"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: fl_law41_trust_rare1pct_tanh (800 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.08`, standardize=`False`) |
| Update Budget | `800` steps |
| Batch Size | `256` |
| Particles / Latent Dim | `P=512` / `z_dim=4` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Architecture | `generator_family=saturating_tanh_mlp, g_hidden=128, g_layers=4, out_dim=2, out_bound=3.0, discriminator_family=simple_mlp, d_hidden=128, d_layers=3` |
| Integrity Guards | `finite_state`, `optimizer_roles`, `exact_optimizer_updates`, `mechanism_exercised`, `rng_isolation` |
| Assigned Views | `four-laws-adherence-v1` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `finite_fraction` | `==` | `1.0` |
| `modes` | `==` | `5` |
| `rare_mode_recall` | `>=` | `0.45` |
| `precision` | `>=` | `0.85` |
| `mass_tv` | `<=` | `0.1` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Atlas** | `atlas-particle-normal-v4` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v4` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Axiom](../families/axiom.md#cohort-cuda-daa283c8a640) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

### Four Laws Adherence Diagnostic Receipts (Separate Cohort)

*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*

| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |
| --- | --- | --- | :---: | ---: | ---: | --- | --- |
| `unified-v11` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.94 kJ (J-to-PASS) | diag_ood_disp_ratio=1.809, diag_realized_step_max=0.03228, finite_fraction=1, mass_tv=0.05517 | [58cc4d1aaf694e6f9aee1119c9131b15.json](../technique-receipts/58cc4d1aaf694e6f9aee1119c9131b15.json) |
| `unified-v12` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.26 kJ (J-to-PASS) | diag_ood_disp_ratio=1.915, diag_realized_step_max=0.03226, finite_fraction=1, mass_tv=0.07251 | [6d4a5086c25f4c67a4b481647e68b24b.json](../technique-receipts/6d4a5086c25f4c67a4b481647e68b24b.json) |
| `unified-v13` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.32 kJ (J-to-PASS) | diag_ood_disp_ratio=1.915, diag_realized_step_max=0.03226, finite_fraction=1, mass_tv=0.07251 | [336539c956484dae9cac31dccb6f1cdf.json](../technique-receipts/336539c956484dae9cac31dccb6f1cdf.json) |
| `unified-v14` | `unified` | NVIDIA L4 | **PASS** | `+0.000` | 1.33 kJ (J-to-PASS) | diag_ood_disp_ratio=1.915, diag_realized_step_max=0.03226, finite_fraction=1, mass_tv=0.07251 | [dc73c539ddc4435681d404c651882456.json](../technique-receipts/dc73c539ddc4435681d404c651882456.json) |
| `axiom-v4` | `axiom` | NVIDIA L4 | **FAIL** | `-0.051` | 2.03 kJ | diag_ood_disp_ratio=1.946, diag_realized_step_max=0.009067, finite_fraction=1, mass_tv=0.03503 | [c5992cf0a6594fefa4f68c595bfa9445.json](../technique-receipts/c5992cf0a6594fefa4f68c595bfa9445.json) |
| `minimal-v2` | `minimal` | NVIDIA L4 | **FAIL** | `-0.291` | 2.55 kJ | diag_ood_disp_ratio=2.005, diag_realized_step_max=0.00174, finite_fraction=1, mass_tv=0.1291 | [a5e02197a0d74e429a3f199432b91a89.json](../technique-receipts/a5e02197a0d74e429a3f199432b91a89.json) |
| `axiom-v4` | `axiom` | NVIDIA RTX A6000 | **FAIL** | `-0.620` | 5.39 kJ | diag_ood_disp_ratio=1.924, diag_realized_step_max=0.008407, finite_fraction=1, mass_tv=0.03698 | [readout-9066cb4353eb435be7819b0c.json](../records/readout-9066cb4353eb435be7819b0c.json) |
| `bispf-optimized-quintic-v1` | `bispf` | NVIDIA L4 | **FAIL** | `-0.647` | 2.24 kJ | diag_ood_disp_ratio=2.246, diag_realized_step_max=0.1742, finite_fraction=1, mass_tv=0.06738 | [ac319162d98444eca5f035cea62738e1.json](../technique-receipts/ac319162d98444eca5f035cea62738e1.json) |
| `recipe-prior-refactor-r1r2-v1` | `r1r2` | NVIDIA L4 | **FAIL** | `-0.750` | 800 J | diag_ood_disp_ratio=6.701, diag_realized_step_max=0.03526, finite_fraction=1, mass_tv=0.175 | [744b95de2b7344a79704d1db53362c39.json](../technique-receipts/744b95de2b7344a79704d1db53362c39.json) |
| `recipe-prior-refactor-k3p-v1` | `k3p` | NVIDIA L4 | **FAIL** | `-1.000` | 938 J | diag_ood_disp_ratio=8.11, diag_realized_step_max=0.005101, finite_fraction=1, mass_tv=0.1825 | [239908ac244f48c8a597765d6aa0f067.json](../technique-receipts/239908ac244f48c8a597765d6aa0f067.json) |
| `recipe-prior-refactor-ka2-v1` | `ka2` | NVIDIA L4 | **FAIL** | `-1.000` | 982 J | diag_ood_disp_ratio=7.702, diag_realized_step_max=0.00521, finite_fraction=1, mass_tv=0.181 | [daaf79789e4444fb9b92dcbe8f0e865c.json](../technique-receipts/daaf79789e4444fb9b92dcbe8f0e865c.json) |
| `recipe-prior-refactor-release07-gan-v3-v1` | `release07-gan-v3` | NVIDIA L4 | **FAIL** | `-1.000` | 815 J | diag_ood_disp_ratio=1.816, diag_realized_step_max=0.06702, finite_fraction=1, mass_tv=0.02661 | [f055fd2415074923a2b56c8f023767d0.json](../technique-receipts/f055fd2415074923a2b56c8f023767d0.json) |
| `rhbi-v12` | `rhbi` | NVIDIA L4 | **FAIL** | `-1.000` | 2.07 kJ | diag_ood_disp_ratio=0.04849, diag_realized_step_max=5.719e-06, finite_fraction=1, mass_tv=0.1068 | [94d4cbcec9cc4ae2a8b42f4fb2069641.json](../technique-receipts/94d4cbcec9cc4ae2a8b42f4fb2069641.json) |
| `recipe-prior-refactor-bcap-pure-v1` | `bcap-dualnorm` | NVIDIA L4 | **FAIL** | `-1.006` | 1.81 kJ | diag_ood_disp_ratio=1.239, diag_realized_step_max=0.1226, finite_fraction=1, mass_tv=0.2006 | [b0f371c49bdf42779995a52dfadcd190.json](../technique-receipts/b0f371c49bdf42779995a52dfadcd190.json) |
| `empf-v8` | `empf` | NVIDIA L4 | **FAIL** | `-1.336` | 1.65 kJ | diag_ood_disp_ratio=1.31, diag_realized_step_max=0.0001521, finite_fraction=1, mass_tv=0.2336 | [db03c89262ad4daaa49fe6e210755f02.json](../technique-receipts/db03c89262ad4daaa49fe6e210755f02.json) |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view four-laws-adherence-v1 --candidate unified-v12 --tasks fl_law41_trust_rare1pct_tanh
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/fl_law41_trust_rare1pct_tanh.json](../../../configs/forge/tasks/fl_law41_trust_rare1pct_tanh.json)
- Source code: [benchmarks/toy_audit/api_four_laws_4.py](../../../benchmarks/toy_audit/api_four_laws_4.py)
- Source code: [experiments/forge/four_laws.py](../../../experiments/forge/four_laws.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/fl_law41_trust_rare1pct_tanh.json](../../../configs/forge/tasks/fl_law41_trust_rare1pct_tanh.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/fl_law41_trust_rare1pct_tanh.json](../../../configs/forge/task-documentation/fl_law41_trust_rare1pct_tanh.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/api_four_laws_4.py](../../../benchmarks/toy_audit/api_four_laws_4.py) (AST symbols: `__doc__`, `THRESHOLDS`, `SPECS`)
- **Read-only Python AST source:** [experiments/forge/four_laws.py](../../../experiments/forge/four_laws.py) (AST symbols: `__doc__`)
