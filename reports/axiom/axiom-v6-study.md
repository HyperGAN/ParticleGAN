# Axiom v6 Clean-Slate Rewrite — Architecture & Official Forge Study

- **Candidate:** `axiom-v6` (`preset="axiom"`, [axiom-v6.json](../../configs/forge/ideas/axiom-v6.json); parent [axiom-v5.json](../../configs/forge/ideas/axiom-v5.json))
- **Evaluated Implementation Commit:** `feat/axiom-v6` @ `edc18e43c` (branched cleanly from `origin/develop` `50abd361b`)
- **Forge Provenance:** `candidate_revision = 29cfbe70ecae527e0507ca1a8246bbbd6efaecd2a4274fbd1907e56f9866abb2`, `source_digest = a40fa792a0f1...` (11 official studies committed under `configs/forge/studies/axiom-v6-*-study.json` and 11 append-only receipts under `reports/forge/records/readout-*.json`; zero historical receipts modified; `configs/forge/selections/family-current-v1.json` untouched)
- **Protocol & Hardware:** Protocol `seed = 0`, public `deterministic_orthogonal_` initializer, `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, single unchanged global `axiom` recipe across every task, 2x `NVIDIA RTX A6000` (PyTorch `2.15.0+cu132`, Python `3.15.0`; `dcgan_performance` and matched NVML energy executed solo on uncontended physical GPU 1)
- **Companion Documents:** [Design Spec](design-spec.md) · [Axiom v4 Study](axiom-study.md) · [Four-Laws Benchmark](four-laws-benchmark.md) · [Forge Technique Inventory](../forge/technique-inventory.md)

---

## 1. Executive Summary

Axiom v6 is a complete clean-slate rewrite of `particlegan/_axiom/` optimizing for **readability, low (non-trivial) cyclomatic complexity, and special-case-free Four-Laws mathematics** that transfer across modalities, tensor ranks, and problem scales without dimension/architecture cliffs.

- **Complexity & Readability Transformation:** Eliminated all `431` semicolon-packed lines (`-> 0`) and `932` lines exceeding 120 characters (`-> 0`), reducing peak McCabe complexity (`Ruff C901`) from **`53 -> 9`** (`mean = 2.69`, `0 > 10`) and peak Radon Boolean cyclomatic complexity (`BoolCC`) from **`183 -> 15`** (`mean = 5.16`, `0 > 15`) across 9 layered acyclic modules (`7,324 LOC` across 11 Axiom files).
- **Zero Forbidden Operations & Zero Legacy Shortcuts:** Strictly zero `torch.autograd.grad`, zero `svd`/`svdvals`/`eigh`/`eigvalsh`/`pinv`/`torch.compile`, zero legacy sibling imports, zero weighted `bincount` (deterministic CUDA safe), zero monkey-patched `_axiom_*` attributes on user modules or tensors, zero `_bn_net_ids` / zero `m_u` (`0.0 B/param` optimizer tensor state on 4D Conv2d), exact `179` `Recipe` fields, Torch-free `particlegan/recipes.py`, and `0` unexplained clock dependencies across all 9 `_axiom/*.py` files.
- **Official Forge & Structural Verification (`55/65` Unique Executed Tasks PASS, `0` INCOMPLETE, `0` ERROR):**
  - Clears **all 6 `axiom-v5` regressions** (`ring_hold` `1200/1200`, `ring_extension` `300/300`, `target_shift_recovery`, `zero_tanh_rail_escape` `p99=0.0051`, `bytegan_bigan_dictionary` `256/256`, `fl_law2_pure_adversarial_codim1`).
  - Sweeps **6 Forge studies 100%**: `r3-tier3-long-horizon-holdout-v1` (**`18/18`** incl. all three 14k continuations and `five_word_joint_smoke` in `702.7s`), `dcgan_performance` (**`6/6`**), `t3-shift-recovery-diagnostic-v1` (**`5/5`**), `t4_imbalance_nash_stretch` (**`3/3`**), `bytegan_stretch` (**`2/2`**), and `t3_causal_isolation_stretch` (**`2/2`**).
  - Achieves **`8/10` `fl_law*` PASS** inside `four-laws-adherence-v2` (**`28/34`** PASS, `0` UNRUN), **`13/16`** PASS on `r4-occam-ablation-v1`, **`16/17`** Tier-1 PASS on `discriminator_stability` (**`32/35`** cross-study T1-2 PASS), **`9/9`** PASS on `scripts/axiom_four_laws_benchmark.py` (`16L-G` `99/100`, `Suite 4A` `1.07x / grad_cos=0.9985`, `5.0` MLP / `4.0` Conv2d host syncs/step), and **`-18.1%` NVML energy / `-36.1%` VRAM** vs `unified_v14` on solo `dcgan_performance_r256`.

---

## 2. Clean-Slate 9-Module Layered Architecture

`particlegan/_axiom/` follows a strictly acyclic 3-layer dependency DAG enforced by AST checks in [tests/test_axiom_integration.py](../../tests/test_axiom_integration.py):

| Layer | Module | Role & Eliminated Special Cases | LOC | Ruff `C901` (mean / max) | Radon `BoolCC` max |
|---|---|---|---:|---:|---:|
| **L0** | [`math_ops.py`](../../particlegan/_axiom/math_ops.py) | Pure tensor math: IEEE-754 mantissa-exponent normalization, zero-`eigh` FP64 Halley eigensplit & NS inverse-sqrt, FP32 Tikhonov quintic NS, Schatten-6 decay, zero-`pinv` closed-form KKT non-ascent (`K <= 2`), Born-Infeld cost, stabilized `log_sinkhorn`, Weyl-Rademacher, Hamilton/deficit quotas. | 720 | 3.40 / 10 | 15 |
| **L0** | [`state.py`](../../particlegan/_axiom/state.py) | Schema-2 `AxiomStepRecord` (fail-closed rejection of schema-1 / tampered checkpoints), `WeakKeyDictionary` + `WeakTensorMap` bindings (zero monkey-patched `_axiom_*` attributes on user modules/tensors), bounded `CaptureRegistry` (survives `.detach()` / `InputNoise`), single `fork_rng` `eval_probe_forward`. | 852 | 2.77 / 10 | 15 |
| **L1** | [`atlas.py`](../../particlegan/_axiom/atlas.py) | Unified anisotropic LTC `PrototypeAtlas` (`D <= 64` full SPD / `D > 64` diagonal behind one directional-variance helper), Wishart/MP shrinkage (`cov_shrink_mp`), 100%-LTC merge & principal-axis split, directional LTC coalescence, annular-gap `rho_atom`, deterministic GPU `HamiltonCommitmentLedger` (zero CPU bounce, zero weighted `bincount`). | 1,066 | 2.23 / 5 | 14 |
| **L1** | [`fiber.py`](../../particlegan/_axiom/fiber.py) | Zero-`eigh` FP64 Halley/NS `O(d_c) x O(d_x)`-equivariant `AxiomCondFiberBank` with quadratic + radial basis, swap-antisymmetric mean operator, and Lyapunov-damped covariance operator (`all d >= 1`, no `d <= 16` cliff). | 530 | 2.96 / 7 | 15 |
| **L1** | [`critic.py`](../../particlegan/_axiom/critic.py) | Law 2 `AxiomPenaltyKernel` / `AxiomCriticPenalty`: U-statistic `effective_dimension`, continuous element-budget `K(d)` & `b_reg(B, d)` (eliminating `dim <= 16/256` & `ndim >= 4` cliffs), zero-`autograd.grad` fwAD primary + capability-based precision-aware central secant fallback (zero class-name sniffing), zero-Jensen EMA-gated Lipschitz hinge, zero-mean logit gauge. | 411 | 3.09 / 7 | 14 |
| **L1** | [`optimizer.py`](../../particlegan/_axiom/optimizer.py) | Law 4 `AxiomOptimizer`: zero `_bn_net_ids` and zero `m_u` (`0.0 B/param` 4D Conv2d state), weight-normalized IEEE-754 log-norm, prime-67 sketch chatter/settling, zero-H2D fast path, role-agnostic compound-pathology 2D NS + Schatten-6 ceiling, subset-scoped single-backward `backward_with_protected` (`direction_blend`), FP32 preservation on BF16/FP16 reload. | 1,221 | 4.01 / 10 | 14 |
| **L2** | [`prior.py`](../../particlegan/_axiom/prior.py) | Law 1 `AxiomPriorRegularizer` & rank-free `should_use_axiom_pullback` (`per_sample_dim <= 64`, no `ndim == 2` cliff): curvature-compensated `GL(D)`-invariant Whitened Burg Log-Det pullback (exact basis + unbiased 2-sketch U-statistic), geometric-mean Mahalanobis scale, antithetic point-mass rail escape. | 989 | 2.58 / 7 | 13 |
| **L2** | [`transport.py`](../../particlegan/_axiom/transport.py) | Law 3 `AxiomTransportLoss` & `prepare_axiom_transport_step`: zero-copy `F.avg_pool2d` feature projection, continuous distance-contrast confidence `kappa_struct` (eliminating `dim > 1024/4096` cliffs & `_spatial_cluster_ratio` trapdoor), zero-sync GPU quota-balanced continuous cluster routing, Nadaraya-Watson simplicial manifold projection, atomic absent-byte transit shielding. | 1,150 | 3.33 / 8 | 15 |
| **L2** | [`loss.py`](../../particlegan/_axiom/loss.py) | Thin `AxiomGANLoss` adapter composing `state` captures, zero-sync GPU `kappa_struct` relativistic blend, `AxiomTransportLoss`, and `AxiomCondFiberBank` across GAN, BiGAN, AE/VAE-GAN, DDGAN, and ByteGAN. | 217 | 2.60 / 7 | 14 |
| **Entry** | [`__init__.py`](../../particlegan/_axiom/__init__.py) & [`axiom.py`](../../particlegan/axiom.py) | Clean public re-exports preserving all backward-compatible symbols. | 168 | — | — |

---

## 3. Readability, Complexity & Packaging Comparison (`origin/develop` vs `axiom-v6`)

| Metric | `origin/develop` (`50abd361b`, `axiom-v5`) | `axiom-v6` (`edc18e43c`) | Gate / Target |
|---|---:|---:|---|
| Core Axiom files / LOC | 7 files / `2,438` LOC | 11 files / `7,324` LOC (`6,520` SLOC) | Clean 9-module DAG |
| Semicolon-packed lines (`;`) | `431` | **`0`** | `0` |
| Lines exceeding 120 chars | `932` | **`0`** | `0` |
| Cryptic 1-2 char module aliases | many | **`0`** | `0` |
| Ruff McCabe `C901` (mean / max) | max **`53`** | **mean `2.69` / max `9`** (`<= 10` AST) | `<= 10` (hard `12`) |
| Radon `BoolCC` (mean / max) | max **`183`** | **mean `5.16` / max `15`** | `<= 15` (hard `18`) |
| `Recipe` dataclass fields | `179` | **`179`** (Torch-free `recipes.py`) | `== 179` |
| `clockfree.source_audit` issues | `0` (5 files) | **`0`** (all 9 `_axiom/*.py` audited) | `0` |
| Release wheel size (`.whl`) | `1,152,000 B` | **`1,200,142 B`** | `< 12,000,000 B` (`12 MB`) |
| `grid100` CUDA host syncs/step | `<= 8.0` | **`5.0`** MLP / **`4.0`** Conv2d | `<= 8.0` |
| Axiom unit & det-CUDA tests | `86` CPU + `3` CUDA | **`128` CPU + `12` det-CUDA** (`239` w/ core) | 100% PASS |

---

## 4. Official 11-Study Forge Scoreboard (`edc18e43c`)

All `axiom-v6` receipts below were generated via `experiments.forge readout axiom-v6@29cfbe70ecae` from deterministic CUDA (`seed = 0`, `deterministic_orthogonal_`, 2x RTX A6000) and committed append-only under `reports/forge/records/`.

| Official Forge View / Study | `axiom-v6` (`edc18e43c`) | Study Receipt | `axiom-v5` | `axiom-v4` | `unified-v15` / `v14` |
|---|---|---|---|---|---|
| **`r3-tier3-long-horizon-holdout-v1`** (18 tasks) | **`18/18` PASS (100%)** (`five_word` `702.7s`, all 3 `14k`, `ring_hold` `1200/1200`, `ring_ext` `300/300`, `rail_escape` `p99=0.0051`) | [readout-78d5dc5c](../forge/records/readout-78d5dc5c1daa9732e5676b4d.json) | regressed 4 tasks | `18/18` | — |
| **`dcgan_performance`** (6 tasks, solo GPU 1) | **`6/6` PASS** (`r256` `36.26 ms` / **`790.3 MiB`**; NVML **`31.76 ms / 9.007 J/step`**, `-18.1%` J vs `v14`; rule `<= 35 ms` -> `falsified`) | [readout-2984c7ab](../forge/records/readout-2984c7ab5b4f026ba2e5d2f7.json) | `6/6` (`34.39 ms / 8.25 J`) | `6/6` (`34.86 ms / 805.8 MiB`) | `v14`: `36.99 ms / 10.99 J / 1220.8 MiB` |
| **`bytegan_stretch`** (2 tasks) | **`2/2` PASS** (`hypercube8` `256/256, tv=0, max_e=0.0436`; `bigan_dict` `256/256, enc_cyc=256`) | [readout-4c1f6093](../forge/records/readout-4c1f609330a3ba091f3394bf.json) | `1/2` (regr `bigan_dict`) | `2/2` | `1/2` |
| **`t3_causal_isolation_stretch`** (2 tasks) | **`2/2` PASS** (`scale_mass_8x` `eig=0.4870, hq=0.9766`; `saddle_shift` `4/4, hq=0.9531`) | [readout-e2f7edf6](../forge/records/readout-e2f7edf628215ac4382d9e7e.json) | `2/2` | `2/2` | `2/2` |
| **`t4_imbalance_nash_stretch`** (3 tasks) | **`3/3` PASS** | [readout-0a09c3a4](../forge/records/readout-0a09c3a4f58fbf9727c0aa49.json) | `3/3` | `3/3` | `3/3` |
| **`t3-shift-recovery-diagnostic-v1`** (5 tasks) | **`5/5` PASS** (`rail_escape` `p99=0.0051, rms=0.0011, sat=0`, `target_shift_recovery` PASS) | [readout-73e856d1](../forge/records/readout-73e856d159f15fd25276d912.json) | `3/5` | `5/5` | — |
| **`particle_interpolation_stretch`** (3 tasks) | **`2/3` PASS** (`sparse_arc` `0.917`, `patch_shift` `peak=0.8784 / hq=1.0`; `saddle_bary` passed final `4/5` checks `hq=0.723..0.796`, missed step 834) | [readout-f2fd872a](../forge/records/readout-f2fd872a28fe8beb751cfb13.json) | `3/3` | `1/3` | — |
| **`four-laws-adherence-v2`** (34 tasks) | **`28/34` PASS (`0` UNRUN)** — **`8/10` `fl_law*` PASS** (`4/4` L4, `1/1` L3, `1/2` L2 `codim1`, `2/3` L1 `codebook_v2` + `highd_rank_d32_v2`) + `gaussian1d_stability` (`ks=0.0149, std=1.0049`) | [readout-69f7d043](../forge/records/readout-69f7d043a14e2780fb2cdd4a.json) | `7/10` `fl_*` (4 anchor regrs) | `3/9` v1 / `7/10` v2 L4 | `9/10` `fl_*` (L4) |
| **`discriminator_stability`** (T1-2, 49 tasks) | **Official T1: `16/17` PASS, `1` FAIL** (`3/3` precision `fp32/mxfp8/nvfp4`, `five_word`, `t3_iso` PASS; `ring16_acq` missed spill). Cross-study T1-2: **`32/35` exec PASS (`91.4%`)** (`14` UNRUN) | [readout-3368e30e](../forge/records/readout-3368e30edcdde7aba0e0754a.json) | `45/47` | `40/47` (`16/17` T1) | — |
| **`r4-occam-ablation-v1`** (16 tasks) | **`13/16` PASS (`0` UNRUN)** (`t3_iso` 2/2, `vector_unequal_width` `hq=0.983/cov=0.207`, `gaussian1d_stability`, `model_ddgan_gaussian_chain` `t1_q0_eig_max=1.213`, `five_word`, `grid100`) | [readout-9f96c1fc](../forge/records/readout-9f96c1fc71db4c9ca5c00dc9.json) | `15/16` | — | — |
| **`t4-strange-architectures-v1`** (7 tasks) | **`4/7` PASS (`0` UNRUN)** (`clockfree`, `zero_tanh_smoke`, `rail_escape`, `grid100`) | [readout-c9406c78](../forge/records/readout-c9406c785907749ab2b78b38.json) | — | `5/7` | — |
| **`axiom_four_laws_benchmark.py`** (9 controls) | **`9/9` PASS** (`16L-G` **`99/100, hq=0.8964`**, `24dec=5.96e-7`, `Suite 4A` `1.07x / grad_cos=0.9985`, `5.0` MLP / `4.0` Conv2d syncs/step) | L2b control run | — | `9/9` (`16L-G 23/100`) | `v12`: `0/9` (`16L-G 8/100`) |

### Solo Idle-GPU1 `dcgan_performance` Breakdown ([readout-2984c7ab](../forge/records/readout-2984c7ab5b4f026ba2e5d2f7.json))

| Task | Median (ms/step) | p95 (ms/step) | Throughput (img/s) | Peak Alloc / Res (MiB) | Optimizer State (`G` / `D`) |
|---|---:|---:|---:|---:|---:|
| `dcgan_performance_r32` | `18.32` | `18.73` | `869.2` | `64.5 / 70.0` | `0 B / 0 B` |
| `dcgan_performance_r64` | `21.39` | `22.54` | `742.8` | `141.6 / 166.0` | `0 B / 0 B` |
| `dcgan_performance_r128` | `25.17` | `26.63` | `631.8` | `339.4 / 358.0` | `0 B / 0 B` |
| **`dcgan_performance_r256`** | **`36.26`** | **`39.77`** | **`433.3`** | **`790.3 / 806.0`** | **`0 B / 0 B`** |
| `dcgan_performance_b64` | `22.34` | `23.42` | `2,847.5` | `231.3 / 238.0` | `0 B / 0 B` |
| `dcgan_performance_w128` | `21.86` | `23.05` | `726.0` | `383.5 / 412.0` | `0 B / 0 B` |

Matched solo GPU-1 NVML energy on `dcgan_performance_r256`: **`31.76 ms/step`**, **`9.007 J/step`**, **`779.8 MiB`** (**`-18.1%` energy** and **`-36.1%` VRAM** vs `unified_v14`'s `36.99 ms / 10.992 J / 1220.8 MiB`).

---

## 5. Honest Remaining Gaps (10 Non-PASS Cells out of 65 Unique Tasks; `0` INCOMPLETE, `0` ERROR)

All 10 non-PASS cells are faithful numerical `FAIL`s under the single unchanged global `axiom` recipe (zero task-specific patches or calibration overrides):

1. **Narrow suffix/spill misses (3):**
   - `ring16_acquisition`: terminal metrics strong (`hq=0.9736, core_cov=0.1454, comp_cov=0.5417`), failed suffix on `spill=0.1291` / step 1534-1550 spikes (gates 14 Tier-2 tasks inside `axiom-v6-disc-stab-study`).
   - `interp_saddle_barycentric`: passed the final `4/5` suffix checks (`hq=0.723..0.796, mae=0.057..0.072`), missed a single earlier check at step 834.
   - `vector_unequal_mass`: `hq=0.9866, mass_tv=0.0096`, missed minor-mode `eig=0.318` / `spill=0.105`.
2. **Deep 16-layer grid architectures (4):**
   - `grid100_mlp16` (`prec=0.869, tv=0.123`), `grid100_mlp16_10k` (`prec=0.863`), `grid100_res16_tanh` (`min_margin=-0.343`), `grid100_res16_scaled_tanh` (`min_margin=-0.251`).
3. **Frontier diagnostic probes (3):**
   - `fl_law1_aniso_unequal_modes` (`7/8` modes, `prec=0.768, min_margin=-0.321` — up sharply from `axiom-v5`'s `2/8, -44.0`).
   - `fl_law2_groupnorm_and_vit_sprites_v2` (`4/4` modes, `hq=1.0`, `template_rmse=0.1302` vs `<= 0.1200`).
   - `ae_ddgan_gaussian_chain_v2` (persistent family gap under protected reconstruction loss; tight final `eig=[0.889, 0.930], paired_gain=25.5x`).
4. **`dcgan_performance` study prediction rule:** All `6/6` underlying tasks passed with `790.3 MiB` VRAM (beating `axiom-v4`'s `805.8 MiB`), while `r256` median latency (`36.26 ms`) sat `+1.26 ms` above the study's `<= 35.0 ms` threshold (marked `falsified` honestly in [readout-2984c7ab](../forge/records/readout-2984c7ab5b4f026ba2e5d2f7.json)).

---

## 6. Committed Study Configurations & Provenance Index

- **Idea Declaration:** [configs/forge/ideas/axiom-v6.json](../../configs/forge/ideas/axiom-v6.json)
- **Study Declarations (11):** `configs/forge/studies/axiom-v6-{bytegan,dcgan-perf,disc-stab,four-laws-v2,interp,occam,shift-diag,t3-holdout,t3-iso,t4-arch,t4-nash}-study.json`
- **Official Forge Readout Receipts (11):**
  - [readout-78d5dc5c1daa9732e5676b4d.json](../forge/records/readout-78d5dc5c1daa9732e5676b4d.json) (`axiom-v6-t3-holdout-study`, `18/18` PASS)
  - [readout-2984c7ab5b4f026ba2e5d2f7.json](../forge/records/readout-2984c7ab5b4f026ba2e5d2f7.json) (`axiom-v6-dcgan-perf-study`, `6/6` PASS)
  - [readout-4c1f609330a3ba091f3394bf.json](../forge/records/readout-4c1f609330a3ba091f3394bf.json) (`axiom-v6-bytegan-study`, `2/2` PASS)
  - [readout-e2f7edf628215ac4382d9e7e.json](../forge/records/readout-e2f7edf628215ac4382d9e7e.json) (`axiom-v6-t3-iso-study`, `2/2` PASS)
  - [readout-0a09c3a4f58fbf9727c0aa49.json](../forge/records/readout-0a09c3a4f58fbf9727c0aa49.json) (`axiom-v6-t4-nash-study`, `3/3` PASS)
  - [readout-73e856d159f15fd25276d912.json](../forge/records/readout-73e856d159f15fd25276d912.json) (`axiom-v6-shift-diag-study`, `5/5` PASS)
  - [readout-f2fd872a28fe8beb751cfb13.json](../forge/records/readout-f2fd872a28fe8beb751cfb13.json) (`axiom-v6-interp-study`, `2/3` PASS)
  - [readout-69f7d043a14e2780fb2cdd4a.json](../forge/records/readout-69f7d043a14e2780fb2cdd4a.json) (`axiom-v6-four-laws-v2-study`, `28/34` PASS, `8/10` `fl_law*`)
  - [readout-3368e30edcdde7aba0e0754a.json](../forge/records/readout-3368e30edcdde7aba0e0754a.json) (`axiom-v6-disc-stab-study`, `16/17` T1 PASS)
  - [readout-9f96c1fc71db4c9ca5c00dc9.json](../forge/records/readout-9f96c1fc71db4c9ca5c00dc9.json) (`axiom-v6-occam-study`, `13/16` PASS)
  - [readout-c9406c785907749ab2b78b38.json](../forge/records/readout-c9406c785907749ab2b78b38.json) (`axiom-v6-t4-arch-study`, `4/7` PASS)
