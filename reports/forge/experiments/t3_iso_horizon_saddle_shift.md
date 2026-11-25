<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->

[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · [Explore by Laws](../LAWS_HUB.md) · [Research Portal](https://hypergan.github.io/ParticleGAN/)

# Law 3A Isolation: Orthogonal 2->4 Shift Across 42.4-Sigma Valley (`t3_iso_horizon_saddle_shift`)

| Field | Value |
| --- | --- |
| **Laws Tested** | [Law 3A](../LAWS_HUB.md#law3a) |
| **Edge Cases** | `theory-isolation` `non-stationary-shift` `screening-horizon` |
| **Primary View (`discriminator_stability`)** | Tier 2 (stretch) |
| **Four Laws Adherence Role** | law3 (anchor) |
| **Fixed Full-Suite Denominator** | Yes (71-test fixed denominator) |
| **Adapter / Domain** | `t3_isolation` / `t3_iso_horizon_saddle_shift` |
| **Prior** | `particle_cloud` (sigma=`0.0`, standardize=`False`) |
| **Update Budget** | `1000` steps |

## What This Experiment Tests

With G = nn.Identity() and a smooth SiLU critic, expands an R^4 target at step 300 from 2 modes (+-1.5 e1) to 4 equidistant orthogonal modes (+-1.5 e1, +-1.5 e2) separated by a 42.4-sigma valley, paired with a frozen control.

> [!NOTE]
> **Protocol notes:** Tier-2 stretch benchmark and Four Laws Adherence Law 3 anchor (step-300 shift + paired frozen control).

## Laws & Edge Cases Targeted

- **[Law 3A: Unconditional W1 Ray-Length & Screening Identifiability](../LAWS_HUB.md#law3a)** ([Law 3A theory](../../../docs/four-laws.md#law-3--transport-identifiability)) — Does the learning system possess enough information to determine appropriate transport?
- **Edge-case tags:** `theory-isolation`, `non-stationary-shift`, `screening-horizon`

## Common Failure Mode

Beyond-horizon screening traps particles at the pre-shift symmetric saddle (remaining at 2/4 modes) because local critic forces vanish or cancel across the 42.4-sigma empty valley.

## Visual Schematic

![Target and protocol schematic for t3_iso_horizon_saddle_shift](assets/t3_iso_horizon_saddle_shift.svg)

*Target & Protocol Schematic · not training output — generated deterministically from the declared task contract.*

## Protocol & Architecture Diagram

```mermaid
flowchart LR
    P["Prior: particle_cloud (sigma=0.0, P=128, z=4)"] --> G["Generator: nn.Identity (0 params)"]
    G --> M["Target Shift @ Step 300"]
    M --> D["Critic: Smooth SiLU MLP"]
    D --> Gate["Gate: t3_iso_horizon_saddle_shift (1000 steps, 5 sustained)"]
```

## Task Contract & Numerical Gates

### Execution & Sampling Contract

| Contract Field | Declared Value |
| --- | --- |
| Initializer | `deterministic_orthogonal` |
| Prior | `particle_cloud` (sigma=`0.0`, standardize=`False`) |
| Update Budget | `1000` steps |
| Batch Size | `128` |
| Particles / Latent Dim | `P=128` / `z_dim=4` |
| Sampling Law | `learned_particles_and_critic_gradient` |
| Evaluation Output Noise | `not_applied_to_measurement` |
| Scoring Weights | `live` |
| Architecture | `generator=nn.Identity (0 trainable parameters; direct data-space ParticlePrior), discriminator_family=smooth_silu_mlp, hidden=64, layers=2, in_dim=4` |
| Shift Step | `300` |
| Integrity Guards | `finite_state`, `optimizer_roles`, `exact_optimizer_updates`, `mechanism_exercised`, `rng_isolation` |
| Assigned Views | `discriminator_stability` (T2 stretch), `four-laws-adherence-v1` (T1 diagnostic), `r4-occam-ablation-v1` (T1 diagnostic), `t3_causal_isolation_stretch` (T1 diagnostic), `t3_factorial_ablation` (T1 diagnostic) |

### Numerical Pass/Fail Thresholds

| Metric | Operator | Threshold |
| --- | :---: | ---: |
| `finite_fraction` | `==` | `1.0` |
| `critic_grad_med` | `<=` | `1.0` |
| `pre_shift_hq` | `>=` | `0.92` |
| `modes` | `==` | `4` |
| `mass_tv` | `<=` | `0.12` |
| `hq` | `>=` | `0.92` |
| `frozen_control_modes` | `==` | `2` |

## Recorded Family Results

> [!IMPORTANT]
> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. `NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — and confer no pass or qualification credit.

| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |
| --- | --- | :---: | --- | --- | --- |
| **Unified** | `unified-v11` | **PASS** | recomputed t3_iso_horizon_saddle_shift gates (5/5 terminal checks passed) | [Unified](../families/unified.md#cohort-cuda-42a130a78fc3) | — |
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

## Reproduce This Experiment

```bash
# Validate this task declaration and inspect planned dependencies
python -m experiments.forge validate
python -m experiments.forge plan --view discriminator_stability --candidate unified-v12 --tasks t3_iso_horizon_saddle_shift
```

**Declaration & Source Files:**

- Task contract: [configs/forge/tasks/t3_iso_horizon_saddle_shift.json](../../../configs/forge/tasks/t3_iso_horizon_saddle_shift.json)
- Source code: [benchmarks/toy_audit/api_t3_isolation.py](../../../benchmarks/toy_audit/api_t3_isolation.py)
- Source code: [experiments/forge/t3_isolation.py](../../../experiments/forge/t3_isolation.py)

## Annotation Provenance

This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:

- **Task contract JSON:** [configs/forge/tasks/t3_iso_horizon_saddle_shift.json](../../../configs/forge/tasks/t3_iso_horizon_saddle_shift.json)
- **Editorial task documentation sidecar:** [configs/forge/task-documentation/t3_iso_horizon_saddle_shift.json](../../../configs/forge/task-documentation/t3_iso_horizon_saddle_shift.json)
- **Read-only Python AST source:** [benchmarks/toy_audit/api_t3_isolation.py](../../../benchmarks/toy_audit/api_t3_isolation.py) (AST symbols: `__doc__`, `THRESHOLDS`, `HOST_DEFINITIONS`)
- **Read-only Python AST source:** [experiments/forge/t3_isolation.py](../../../experiments/forge/t3_isolation.py) (AST symbols: `__doc__`)
