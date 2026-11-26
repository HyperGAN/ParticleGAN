<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# 8-Mode Ring Long-Horizon Endurance Extension (`ring_extension`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 3A](../LAWS_HUB.md#law3a) · [Law 4.4](../LAWS_HUB.md#law4_4) · [Law 4.5](../LAWS_HUB.md#law4_5) |
| **Edge Cases** | `long-horizon-stability` `checkpoint-continuation` `mode-retention` |
| **Primary View (`discriminator_stability`)** | Tier 3 (required) |
| **Four Laws Adherence Role** | law3 (anchor) |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `ring_endurance` / `ring_extension` |
| **Prior** | `mog` (sigma=`0.025`, standardize=`False`) |
| **Update Budget** | `7500` steps |

## What This Experiment Tests

Continues the candidate's converged `ring_hold` state without resetting or re-annealing to test whether all 8 modes and high sample quality (hq >= 0.90) persist over an extended training horizon.

> [!NOTE]
> **Protocol notes:** Tier-3 required endurance task and Four Laws Adherence Law 3 anchor; uninterrupted continuation of `ring_hold`.

## Laws & Edge Cases Targeted

- **[Law 3A: Unconditional W1 Ray-Length & Screening Identifiability](../LAWS_HUB.md#law3a)** ([Law 3A theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 4.4: Trajectory Scale vs. Settling](../LAWS_HUB.md#law4_4)** ([Law 4.4 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **[Law 4.5: Coupled Minimax Contraction](../LAWS_HUB.md#law4_5)** ([Law 4.5 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `long-horizon-stability`, `checkpoint-continuation`, `mode-retention`

## Common Failure Mode

Slow prior drift, step-size decay artifacts, or undamped minimax cycles cause delayed mode collapse hundreds of updates after initial convergence.

**Historical audit diagnosis** (`reports/forge/problem-definitions.json`): BISPF lacks an ordinary qualifying run in the selected cohort. BCAP's diagnostic run loses quality during the hold phase; the extension cannot rescue a failed hold.

- *Recommended research focus:* Measure the unchanged BISPF recipe in its ordinary cohort on current source and preserve the whole acquisition/hold/extension protocol.

## Visual Schematic

![Target and protocol schematic for ring_extension](assets/ring_extension.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.025)"] --> G["Generator G(z)"]
    G --> D["Critic D(x)"]
    D --> Gate["Gate: ring_extension (7500 steps, clean) <- continues ring_hold"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.025`, standardize=`False`) |
| Update Budget | `7500` steps |
| Batch Size | `host-declared` |
| Particles / Latent Dim | `P=host` / `z_dim=host` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Continuation Parent | [`ring_hold`](ring_hold.md) |
| Dependencies | [`ring_hold`](ring_hold.md) (`checkpoint`) |
| Assigned Views | `bispf_throughput_endurance` (T1 diagnostic), `clockfree_continuous` (T3 required), `discriminator_stability` (T3 required), `formulation_comparison` (T3 required), `four-laws-adherence-v1` (T1 diagnostic), `host_profile_transfer` (T3 required), `lazy_cadence_coverage` (T1 diagnostic), `prior-ablation-full-suite-v1` (T1 diagnostic) (+1 more) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `modes` | `==` | `8` |
| `hq` | `>=` | `0.9` |
| `hq` | `<=` | `1.0` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **BISPF** | `bispf-optimized-quintic-v1` | **PASS** | recomputed every extension observation without selecting another window | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | **PASS** | no compatible result | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **Atlas** | `atlas-particle-normal-v4` | `NOT MEASURED` | no compatible result | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **Axiom** | `axiom-v4` | `NOT MEASURED` | no compatible result | [Axiom](../families/axiom.md#cohort-cuda-daa283c8a640) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** | no compatible result | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `BLOCKED` | Skipped incompatible test: ring_extension: clean/live scoring has no declared policy-contr | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `NOT MEASURED` | no compatible result | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `NOT MEASURED` | no compatible result | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `NOT MEASURED` | no compatible result | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `NOT MEASURED` | no compatible result | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `NOT MEASURED` | no compatible result | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |
| **Unified** | `unified-v11` | `NOT MEASURED` | no compatible result | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |

### Four Laws Adherence Diagnostic Receipts (Separate Cohort)

*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*

| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |
| --- | --- | --- | :---: | ---: | ---: | --- | --- |
| `axiom-v4` | `axiom` | NVIDIA RTX A6000 | **PASS** | — | 9.19 kJ (J-to-PASS) | confirmation_checks=200, converged_step=1400, extension_checks=300, extension_passing=300 | [readout-9066cb4353eb435be7819b0c.json](../records/readout-9066cb4353eb435be7819b0c.json) |
| `law4-energy-v11` | `unified` | NVIDIA L4 | **PASS** | — | 2.29 kJ (J-to-PASS) | confirmation_checks=200, converged_step=1400, extension_checks=300, extension_passing=300 | [2ce9b0dcefb0448996148bc72a949ee0.json](../technique-receipts/2ce9b0dcefb0448996148bc72a949ee0.json) |
| `unified-v11` | `unified` | NVIDIA L4 | **PASS** | — | 3.42 kJ (J-to-PASS) | confirmation_checks=200, converged_step=1400, extension_checks=300, extension_passing=300 | [55cf35e8da80417f9d921f4e92ad1aa3.json](../technique-receipts/55cf35e8da80417f9d921f4e92ad1aa3.json) |
| `unified-v12` | `unified` | NVIDIA L4 | **PASS** | — | 2.27 kJ (J-to-PASS) | confirmation_checks=200, converged_step=1400, extension_checks=300, extension_passing=300 | [d5de4369af7f46dca241b99a499339c0.json](../technique-receipts/d5de4369af7f46dca241b99a499339c0.json) |
| `unified-v14` | `unified` | NVIDIA L4 | **PASS** | — | 2.49 kJ (J-to-PASS) | confirmation_checks=200, converged_step=1400, extension_checks=300, extension_passing=300 | [2136c47e66e747808acefb3d265916a6.json](../technique-receipts/2136c47e66e747808acefb3d265916a6.json) |
| `minimal-v2` | `minimal` | NVIDIA L4 | **FAIL** | — | 7.36 kJ | confirmation_checks=200, converged_step=2816, first_hold_failure=2841, hold_budget=1200 | [bff5b9ea00b94159bc757045832938c8.json](../technique-receipts/bff5b9ea00b94159bc757045832938c8.json) |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks ring_extension
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/ring_extension.json](../../../configs/forge/tasks/ring_extension.json)
- Source code: [reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py](../../toy100/gap-fill-20260925/sources/k3p/convergence_gate.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/ring_extension.json](../../../configs/forge/tasks/ring_extension.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/ring_extension.json](../../../configs/forge/task-documentation/ring_extension.json)
- **Read-only Python AST source:** [reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py](../../toy100/gap-fill-20260925/sources/k3p/convergence_gate.py) (AST symbols: `__doc__`)
