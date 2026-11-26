<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Extreme Nash Imbalance on Unequal-Width 4-Mode Mixture (`nash_unequal_width_hostile_imbalance`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 2](../LAWS_HUB.md#law2) · [Law 3A](../LAWS_HUB.md#law3a) · [Law 4.3](../LAWS_HUB.md#law4_3) · [Law 4.5](../LAWS_HUB.md#law4_5) |
| **Edge Cases** | `hostile-nash-imbalance` `multi-scale-width` `off-manifold-init` |
| **Primary View (`discriminator_stability`)** | Tier 2 (stretch) |
| **Four Laws Adherence Role** | — |
| **Fixed Full-Suite Denominator** | Yes (59-test active runnable denominator) |
| **Adapter / Domain** | `imbalance_nash` / `nash_unequal_width_hostile_imbalance` |
| **Prior** | `mog` (sigma=`0.025`, standardize=`False`) |
| **Update Budget** | `1100` steps |

## What This Experiment Tests

Combines >600x critic overcapacity and hostile off-manifold initialization with 4 unequal-width Gaussian modes (sigma in {0.07, 0.12, 0.20, 0.30}), testing whether Nash equilibration preserves per-mode scale fidelity.

> [!NOTE]
> **Protocol notes:** Tier-2 stretch task (1,100 updates).

## Laws & Edge Cases Targeted

- **[Law 2: Potential Geometry](../LAWS_HUB.md#law2)** ([Law 2 theory](../../../docs/four-laws.md#law-2--potential-geometry)) — Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?
- **[Law 3A: Unconditional W1 Ray-Length & Feature-Space Screening Identifiability](../LAWS_HUB.md#law3a)** ([Law 3A theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **[Law 4.3: Displacement & Kreiss Control](../LAWS_HUB.md#law4_3)** ([Law 4.3 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **[Law 4.5: Coupled Minimax Contraction](../LAWS_HUB.md#law4_5)** ([Law 4.5 theory](../../../docs/four-laws.md#law-4--functional-and-game-dynamics)) — Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?
- **Edge-case tags:** `hostile-nash-imbalance`, `multi-scale-width`, `off-manifold-init`

## Common Failure Mode

Even if the generator escapes the initial corner, critic dominance collapses the narrow sigma=0.07 mode or distorts component covariances.

## Visual Schematic

![Target and protocol schematic for nash_unequal_width_hostile_imbalance](assets/nash_unequal_width_hostile_imbalance.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: mog (sigma=0.025, P=256, z=4)"] --> G["Generator: 32x2 Compressed (0.08x)"]
    G --> D["Critic: 512x4 Overpowered Tilted D"]
    D --> Gate["Gate: nash_unequal_width_hostile_imbalance (1100 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `mog` (sigma=`0.025`, standardize=`False`) |
| Update Budget | `1100` steps |
| Batch Size | `128` |
| Particles / Latent Dim | `P=256` / `z_dim=4` |
| Sampling Law | `public_prior_without_output_noise` |
| Evaluation Output Noise | `clean` |
| Scoring Weights | `live` |
| Integrity Guards | `finite_state`, `optimizer_roles`, `exact_optimizer_updates`, `mechanism_exercised`, `rng_isolation` |
| Assigned Views | `discriminator_stability` (T2 stretch), `t4_imbalance_nash_stretch` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `finite_fraction` | `==` | `1.0` |
| `min_mass_ratio` | `>=` | `0.6` |
| `hq` | `>=` | `0.82` |
| `mass_tv` | `<=` | `0.18` |
| `sw1_normalized` | `<=` | `0.16` |
| `resolved_core_covariance_error` | `<=` | `0.65` |
| `resolved_core_min_eigen_ratio` | `>=` | `0.15` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Axiom** | `axiom-v5` | **PASS** | recomputed nash_unequal_width_hostile_imbalance gates (5/5 terminal checks passed) | [Axiom](../families/axiom.md#cohort-cuda-42a130a78fc3) | [HF GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/main/media/forge/leaderboard/axiom/axiom-v4/nash_unequal_width_hostile_imbalance.gif) |
| **Unified** | `unified-v15` | **PASS** | recomputed nash_unequal_width_hostile_imbalance gates (5/5 terminal checks passed) | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |
| **Atlas** | `atlas-particle-normal-v4` | `BLOCKED` | imbalance_nash requires an effective MoGParticlePrior | [Atlas](../families/atlas.md#cohort-cuda-daa283c8a640) | — |
| **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BCAP](../families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | — |
| **BISPF** | `bispf-optimized-quintic-v1` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [BISPF](../families/bispf.md#cohort-cuda-1bf9d7d34422) | — |
| **E22** | `e22` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [E22](../families/e22.md#cohort-cuda-1bf9d7d34422) | — |
| **EMPF** | `empf-v8` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [EMPF](../families/empf.md#cohort-cuda-23a94f16ca88) | — |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | `NOT MEASURED` | no compatible result | [GAN v3 release 0.7](../families/release07-gan-v3.md#cohort-cuda-daa283c8a640) | — |
| **K3P** | `recipe-prior-refactor-k3p-v1` | `NOT MEASURED` | no compatible result | [K3P](../families/k3p.md#cohort-cuda-daa283c8a640) | — |
| **KA2** | `recipe-prior-refactor-ka2-v1` | `NOT MEASURED` | no compatible result | [KA2](../families/ka2.md#cohort-cuda-daa283c8a640) | — |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `NOT MEASURED` | no compatible result | [R1/R2](../families/r1r2.md#cohort-cuda-daa283c8a640) | — |
| **RHBI** | `rhbi-v12` | `NOT MEASURED` | No recorded result for this selected configuration and source. | [RHBI](../families/rhbi.md#cohort-cuda-23a94f16ca88) | — |

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks nash_unequal_width_hostile_imbalance
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/nash_unequal_width_hostile_imbalance.json](../../../configs/forge/tasks/nash_unequal_width_hostile_imbalance.json)
- Source code: [benchmarks/toy_audit/api_imbalance_nash.py](../../../benchmarks/toy_audit/api_imbalance_nash.py)
- Source code: [experiments/forge/imbalance_nash.py](../../../experiments/forge/imbalance_nash.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/nash_unequal_width_hostile_imbalance.json](../../../configs/forge/tasks/nash_unequal_width_hostile_imbalance.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/nash_unequal_width_hostile_imbalance.json](../../../configs/forge/task-documentation/nash_unequal_width_hostile_imbalance.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/api_imbalance_nash.py](../../../benchmarks/toy_audit/api_imbalance_nash.py) (AST symbols: `__doc__`, `THRESHOLDS`)
- **Read-only Python AST source:** [experiments/forge/imbalance_nash.py](../../../experiments/forge/imbalance_nash.py) (AST symbols: `__doc__`)
