# Axiom Family Comparative Forge Study (`axiom-v4` Release)

- **Candidate:** `axiom-v4` (`preset="axiom"`, [axiom-v4.json](../../configs/forge/ideas/axiom-v4.json); parent [`axiom-v1.json`](../../configs/forge/ideas/axiom-v1.json))
- **Source & Revision:** `origin/feat/axiom-family` commit `758b397a5` (PR #672, `axiom-v4` complete), Forge `candidate_revision = 0e31275ce41d6a1e161a3f8e8ed8c10e5781b8c9829789a73c57010aae08f2f6`
- **Core Footprint:** **`2,409` LOC** across 7 clean-slate modules ([`__init__.py`](../../particlegan/_axiom/__init__.py) 70, [`critic.py`](../../particlegan/_axiom/critic.py) 378, [`loss.py`](../../particlegan/_axiom/loss.py) 213, [`optimizer.py`](../../particlegan/_axiom/optimizer.py) 550, [`prior.py`](../../particlegan/_axiom/prior.py) 468, [`transport.py`](../../particlegan/_axiom/transport.py) 694, [`axiom.py`](../../particlegan/axiom.py) 36; strictly `<= 2,500` LOC cap; zero legacy sibling imports, zero `svd`/`eigh`/`pinv`/`torch.compile`, zero new `Recipe` fields, 100% clock-free).
- **Protocol & Hardware:** Protocol `seed = 0`, public `deterministic_orthogonal_` initializer, clean live weights, fixed global recipe across every task, `NVIDIA RTX A6000` (PyTorch `2.15.0+cu132`, `dcgan_performance` + matched 6-recipe NVML energy executed solo on pristine uncontended physical GPU 1 `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`, pre-run `0%` util, `15 MiB` VRAM, `25.98 W` idle).
- **Companion Documents:** [Design Spec](design-spec.md) · [Stage-1 Gap Analysis](gap-analysis.md) · [Four-Laws Benchmark](four-laws-benchmark.md) · [Forge Family Leaderboard](../forge/technique-inventory.md) · [Unified ByteGAN Study](../forge/studies/unified-bytegan-bijection.md)

---

## 1. Executive Summary & Recommendation

> [!IMPORTANT]
> **Recommendation: MERGE `feat/axiom-family` into `develop` as an opt-in research/production family (`preset="axiom"`).**
> Keep `DEFAULT_RECIPE = "unified_v12"` and `configs/forge/selections/family-current-v1.json` **untouched**. Scientific superiority and default adoption require accepted compatible calibration and full Tier-1/Tier-2 parity; `axiom-v4` delivers major repository breakthroughs across untouched 14,000-update endurance (`18/18` sweep), Conv2d throughput/VRAM/NVML energy, ByteGAN accuracy, `t3_iso` covariance margin, Nash imbalance (`3/3`), shift recovery (`5/5`), and Four-Laws structural controls (`9/9`), while cleanly documenting honest remaining Tier-2 and `fl_*` diagnostic gaps for `axiom-v5`.

### Headline Breakthroughs (`axiom-v4`, `758b397a5`)

1. **18/18 (100%) Untouched 14,000-Update Tier-3 Long-Horizon Holdout Sweep (`r3-tier3-long-horizon-holdout-v1`):**
   - Sweeps every single task (**`18/18` PASS**, up from `axiom-v3`'s `10/18`), beating every recorded historical family on this holdout: `unified-v11` (`15/18`), `empf-v8` (`10/18`), and `rhbi-v12` (`8/18`).
   - Passes all three 14,000-update fresh-process continuations (`grid100_14k` `score=0.3269`, `rotated100_14k` `score=0.3425`, `staggered100_14k` `score=0.3144`), `five_word_joint_smoke` (`5/5` words, `mass_tv=0.0190`), `five_word_joint_hold` (`5/5` words, `mass_tv=0.0209`, `quality_fraction=1.0`), `ring_hold` (`200/200` settling + `1200/1200` hold, `0` failures), `ring_extension` (`300/300` passing), `target_shift_recovery`, and `zero_tanh_rail_escape` (`p99=0.0108`).
2. **#1 Lowest-VRAM & Lowest-Energy High-Resolution Conv2d Family (`6/6` PASS Solo Idle GPU 1 + Matched 6-Recipe NVML Energy):**
   - Official solo idle-GPU1 `dcgan_performance_r256`: **`34.86 ms/step`** median (`p95 = 36.78 ms`, `456.3 img/s`, **`805.8 MiB`** peak alloc / `836.0 MiB` reserved) — **`-26.2%` latency / `+35.6%` throughput / `-29.8%` VRAM** vs `unified-v14` (`47.22 ms`, `336.6 img/s`, `1147.6 MiB`), **`-15.8%` latency / `-35.2%` VRAM** vs `unified-v12` (`41.39 ms`, `1243.4 MiB`), **`-12.7%` latency / `-29.8%` VRAM** vs `unified-v13` (`39.94 ms`, `1147.6 MiB`), **`-37.2%`** vs `minimal-v2` (`55.50 ms`), **`2.56x`** faster than `unified-v11`, **`4.96x`** faster than `empf-v8`, and **`17.21x`** faster than `rhbi-v12`.
   - Matched same-session idle-GPU1 NVML energy (`dcgan_performance_r256`): **`7.307 J/step`** (`33.25 ms` med, `216.83 W`) — **`-29.8%`** below `unified_v14` (`10.403 J/step`), **`-28.3%`** below `unified_v13` (`10.188 J/step`), **`-26.7%`** below `unified_v12` (`9.975 J/step`), **`-33.4%`** below `minimal` (`10.977 J/step`), and **`13.1x` lower energy** than `rhbi` (`95.989 J/step`).
3. **Full Tier-1 + Tier-2 `discriminator_stability` Unlock (`40/47` Executed PASS, `16/17` Tier 1):**
   - Massive leap from `axiom-v3` (`11/17`, blocked at Tier 1) to **`16/17` Tier 1 PASS** (all `11/11` required Tier-1 tasks PASS, including `ring16_acquisition` at `hq=0.9912, min_eig=0.3534` and `five_word_joint_smoke`) and full Tier-2 execution (**`40/47` PASS** across executed tasks, beating `empf-v8` `24/41` and `rhbi-v12` `34/41`, and approaching `unified-v11` `41/47`).
   - Passes **`3/3` `precision_gaussian_{fp32,nvfp4,mxfp8}`**, `4/4` AE/VAE model modes, `5/6` vector suites, `3/3` Nash imbalance, `2/2` `t3_iso`, `five_word_joint_{smoke,hold}`, `grid100`/`rotated100`/`staggered100`, and `bytegan_hypercube8`.
4. **Repository-Best `bytegan_stretch` (`2/2` PASS) & `t3_causal_isolation_stretch` (`2/2` PASS), Plus Perfect `3/3` Nash & `5/5` Shift Recovery:**
   - `bytegan_hypercube8`: **`256/256`** unique valid bytes, `0` collisions, `0` orphans, `mass_tv=0.0`, **`rms_bit_error = 0.00471`** (**`9.3x` lower** than `unified-v14` `0.04393`, **`9.4x` lower** than `unified-v13` `0.0445`), `max_bit_error = 0.0796` (`2.0x` lower than `unified-v14` `0.1605`), `min_bit_margin = 0.9626` (vs `unified-v14` `0.8395`). `bytegan_bigan_dictionary`: `256/256` forward/recon/encoder bijection, `latent_nn_hamming_mean = 1.0000`.
   - `t3_iso_scale_mass_8x`: **`core_min_eigen_ratio = 0.5447`** (**`+0.0724`** above `unified-v12`/`v13`/`v14` `0.4723`) and `core_component_covariance_error = 0.2376` (beats `unified-v14` `0.2959`).
   - `t4_imbalance_nash_stretch`: **`3/3` PASS** (up from `axiom-v3` `1/3`), `t3-shift-recovery-diagnostic-v1`: **`5/5` PASS** (up from `3/5`), `t4-strange-architectures-v1`: **`5/7` PASS** (up from `4/7`).
5. **Four-Laws Structural Controls (`9/9` PASS) & First Recorded `four-laws-adherence-v1` Diagnostic Baseline (`22/33` PASS):**
   - Retains `9/9` PASS on the Four-Laws adversarial stress controls vs `0/9` for `unified_v12`/`v13`/`minimal`/`rhbi` ([four-laws-benchmark.md](four-laws-benchmark.md)).
   - On the new `four-laws-adherence-v1` view (`research_diagnostic` scope only; grants zero ordinary qualification credit), achieves **`22/33` unique PASS** = **`19/24` curated Forge anchors** + **`3/9` adversarial `fl_*` probes** (`fl_law2_pure_adversarial_codim1`, `fl_law42_swap_deep_critic_16l` at `8/8, hq=1.0, valley=0.0`, and `fl_law42_deep_resbn_and_conv1x1` at `sw1=0.0236, conv1x1_sw1=0.0210`).
6. **Honest Remaining Gaps (`axiom-v5` Targets):**
   - `disc-stab` `7` executed `FAIL`s: `model_ddgan_gaussian_chain` & `model_ae_ddgan_gaussian_chain` (which gate `model_{ddgan,ae_ddgan}_bimodal_chain`, leaving 2 downstream tasks unexecuted), `interp_img_patch_shift` (regressed from `v3` diagnostic pass to zero collapse), `interp_saddle_barycentric` (`0.7639` vs `0.78`, 2 triangles short), `interp_sparse_arc` (terminal `1.0` / `0.8056` passed, missed earlier suffix), `vector_spiral` (`sw1=0.1263`), and `img_blobs4` (`2/4` modes).
   - `t4-strange-architectures-v1` `2` `FAIL`s: `grid100_mlp16` (terminal `accuracy_pass=True` & `frozen_pass=True`, missed live suffix) and `grid100_res16_tanh` (`cov_trace_bias=-0.1184` vs `0.10`, `0.0%` saturation).
   - `four-laws-adherence-v1` `6/9` failing `fl_*` diagnostic probes: `fl_law1_aniso_unequal_modes`, `fl_law1_noisy_atomic_codebook`, `fl_law2_groupnorm_and_vit_sprites`, `fl_law3_highd_orphan_d32`, `fl_law41_trust_rare1pct_tanh`, `fl_law44_highd_sparse_prior_and_shock`.

---

## 2. Official Forge Study Scoreboard (Head-to-Head Across Families)

All `axiom-v4` rows below come from **official frozen Forge studies** (`seed = 0`, `NVIDIA RTX A6000`, `candidate_revision = 0e31275ce41d`) with append-only committed receipts under `reports/forge/records/` (preserving all 8 prior `axiom-v1`/`v3` receipts and every historical receipt byte-for-byte). `unified-v14` numbers cite its committed Forge receipts on `origin/develop` (`c4168544a`, PR #668: `readout-030b7549`, `readout-d393e495`, `readout-f8b7f645`, `readout-9b4a34dd`).

| Official Forge View / Study | **`axiom-v4`** (`758b397a5`) | `axiom-v3` (`37f537db8`) | `unified-v14` (`origin/develop`) | `unified-v13` | `unified-v12` | `unified-v11` | `rhbi-v12` | `empf-v8` | `minimal-v2` |
|---|---|---|---|---|---|---|---|---|---|
| **`dcgan_performance` (`r256` med / VRAM)** | **`6/6` · `34.86 ms` / `805.8 MiB`** ([ro](../forge/records/readout-506977051c754e8f9694b117.json)) | `6/6` · `32.94 ms` / `805.8 MiB` ([ro](../forge/records/readout-be28b86f5953f29b4aff1d7a.json)) | `6/6` · `47.22 ms` / `1147.6 MiB` (`d393e495`) | `r256`: `39.94 ms` / `1147.6 MiB` | `6/6` · `41.39 ms` / `1243.4 MiB` ([ro](../forge/records/readout-59898e1a77c4b179c9aebbc1.json)) | `6/6` · `89.19 ms` / `1372.6 MiB` ([ro](../forge/records/readout-1e5fb2354920c2770b8b582c.json)) | `6/6` · `599.88 ms` / `3706.6 MiB` ([ro](../forge/records/readout-49c12ce12c1befe480bc694b.json)) | `6/6` · `173.05 ms` / `1435.4 MiB` ([ro](../forge/records/readout-dcac6347fd2669483212c6db.json)) | `6/6` · `55.50 ms` / `1344.8 MiB` ([ro](../forge/records/readout-d50a87c606ed4cd8e01a42fe.json)) |
| **`r3-tier3-long-horizon-holdout-v1` (18 tasks)** | **`18/18` PASS (100%)** ([ro](../forge/records/readout-41f9c26a18492a4834dbe767.json)) | `10/18` PASS ([ro](../forge/records/readout-c5afa913aaf559f91a6d97a2.json)) | — | — | — | `15/18` PASS ([ro](../forge/records/readout-43bffb4ee1739e9c86931f95.json)) | `8/18` PASS ([ro](../forge/records/readout-a97cc4411799f84fe22dec86.json)) | `10/18` PASS ([ro](../forge/records/readout-d0f74bbbd3c76f240c0e2729.json)) | — |
| **`bytegan_stretch` (2 tasks)** | **`2/2` PASS (`rms=0.0047`)** ([ro](../forge/records/readout-cf0f08081668d4e14130cd6b.json)) | `2/2` PASS (`rms=0.0061`) ([ro](../forge/records/readout-3a4a7316efe91140aae6130a.json)) | `2/2` PASS (`rms=0.0439`) (`f8b7f645`) | `2/2` PASS (`rms=0.0445`) | `0/2` FAIL | `0/2` | `0/2` | `0/2` | `0/2` |
| **`t3_causal_isolation_stretch` (2 tasks)** | **`2/2` PASS (`eig=0.5447`)** ([ro](../forge/records/readout-b6570e904053603948226d93.json)) | `2/2` PASS (`eig=0.5234`) ([ro](../forge/records/readout-a67cef1eb1208f825e13c67d.json)) | `2/2` PASS (`eig=0.4723`) (`030b7549`) | `2/2` PASS (`eig=0.4723`) | `2/2` PASS (`eig=0.4723`) | `2/2` PASS ([ro](../forge/records/readout-d1e36bf514f755e46b333e6d.json)) | `0/2` FAIL ([ro](../forge/records/readout-8a63e947313fcb7d2e62a8d2.json)) | `0/2` FAIL ([ro](../forge/records/readout-7e0c315962234ea1c2c2ba3f.json)) | `1/2` FAIL ([ro](../forge/records/readout-1deb8069997d547690b0036c.json)) |
| **`t4_imbalance_nash_stretch` (3 tasks)** | **`3/3` PASS** ([ro](../forge/records/readout-9a10b57d1dda5dd13bd560e6.json)) | `1/3` PASS ([ro](../forge/records/readout-f711bad4b080dc992add928d.json)) | `3/3` PASS (`030b7549`) | `3/3` PASS | `3/3` PASS | `3/3` PASS ([ro](../forge/records/readout-2eb07e97714a1da4177d5d25.json)) | — | — | `2/3` PASS ([ro](../forge/records/readout-2f4291537e93817e412994e3.json)) |
| **`t3-shift-recovery-diagnostic-v1` (5 tasks)** | **`5/5` PASS** ([ro](../forge/records/readout-d3967c0e415122139e9783d8.json)) | `3/5` PASS ([ro](../forge/records/readout-cec19ecba5301eebc1e9820e.json)) | — | — | — | `5/5` PASS ([ro](../forge/records/readout-2029c37a1359498d3d8f2eb3.json)) | — | — | `2/4` PASS ([ro](../forge/records/readout-71264208cbaf820429ca4975.json)) |
| **`t4-strange-architectures-v1` (7 tasks)** | **`5/7` PASS** ([ro](../forge/records/readout-18bc7392296d5574083ccab4.json)) | `4/7` PASS ([ro](../forge/records/readout-fa20a38c3690656c64c2a7fc.json)) | `7/7` PASS (`9b4a34dd`) | — | `7/7` PASS ([ro](../forge/records/readout-c1ee5c3bb7ac4f432532da63.json)) | `6/7` PASS ([ro](../forge/records/readout-b96d9de47bb52cc91fbb4a4f.json)) | — | — | `3/7` PASS ([ro](../forge/records/readout-6bac8895ffa14ef20dcbee75.json)) |
| **`discriminator_stability` (T1 / T1–2)** | **`40/47` T1–2** (`16/17` T1) ([ro](../forge/records/readout-d83ce97f520f7668f39996df.json)) | `11/17` T1 (blocked) ([ro](../forge/records/readout-109bbb98231a0a7a4e6903f9.json)) | `45/49` T1–2 (`030b7549`, 1 inc.) | `17/17` T1 (`47/51` live) | `17/17` T1 (`46/48` [ro](../forge/records/readout-b51ab67c4d97526186e23651.json)) | `41/47` T1–2 ([ro](../forge/records/readout-fc5b67a767e57d748f460610.json)) | `34/41` ([ro](../forge/records/readout-47591bd9555febfe3409fd0d.json)) | `24/41` ([ro](../forge/records/readout-33f7628135472eef06f0d92a.json)) | `10/17` T1 ([ro](../forge/records/readout-889d4d1d2b24d0a08ab4017c.json)) |
| **`four-laws-adherence-v1`** *(diagnostic only)* | **`22/33` PASS** (`19/24` + `3/9` `fl_*`) ([ro](../forge/records/readout-9066cb4353eb435be7819b0c.json)) | — | — | — | — | — | — | — | — |
| **Four-Laws Adversarial Stress Controls** | **`9/9` PASS** ([bench](four-laws-benchmark.md)) | `9/9` PASS | — | `0/9` | `0/9` | `0/9` | `0/9` | `0/9` | `0/9` |

---

## 3. High-Resolution Conv2d Performance & Matched NVML Energy (`dcgan_performance`)

### 3.1 Official Solo Idle-GPU1 `axiom-v4-dcgan-perf-study` (`6/6` Unique Tasks PASS)

Executed solo (`--gpus 1 --workers-per-gpu 1`, no GPU sharing) on pristine uncontended physical RTX A6000 GPU 1 (`GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`, `0%` prior util, `15 MiB` prior VRAM, `25.98 W` idle) via [readout-506977051c754e8f9694b117.json](../forge/records/readout-506977051c754e8f9694b117.json).

> [!NOTE]
> Because `dcgan_performance_r256` also ran earlier under GPU sharing inside `axiom-v4-four-laws-adherence-study` (attempt `bf4e0cbf`, `contended=true`, `47.17 ms`), Forge's cross-study receipt selector preserves both compatible `r256` attempts verbatim inside `readout-50697705` (and `readout-9066cb43`), which marks `decision_outcomes[0].outcome = "incomplete"` under Forge's duplicate-task guard. All 6 rows below report the uncontended solo idle-GPU1 (`contended=false`, `producer_study_id = "axiom-v4-dcgan-perf-study"`) receipts from `readout-50697705`.

| Task | Config | `axiom-v4` Median (ms) | `axiom-v4` p95 (ms) | Throughput (img/s) | Peak Alloc / Res (MiB) | `unified-v14` Median (ms) | `unified-v12` Median (ms) | Latency Delta vs `v14` / `v12` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dcgan_performance_r32` | `32x32, B16, W64` | **`16.21`** | `16.83` | `982.6` | `62.1 / 68.0` | `19.36` | `19.17` | **`-16.3%` / `-15.4%`** |
| `dcgan_performance_r64` | `64x64, B16, W64` | **`19.11`** | `20.56` | `824.7` | `142.4 / 166.0` | `23.92` | `22.04` | **`-20.1%` / `-13.3%`** |
| `dcgan_performance_r128` | `128x128, B16, W64` | **`22.87`** | `23.83` | `698.0` | `343.7 / 360.0` | `28.74` | `26.47` | **`-20.4%` / `-13.6%`** |
| **`dcgan_performance_r256`** | **`256x256, B16, W64`** | **`34.86`** | **`36.78`** | **`456.3`** | **`805.8 / 836.0`** | `47.22` (`1147.6 MiB`) | `41.39` (`1243.4 MiB`) | **`-26.2%` / `-15.8%`** (`-29.8%` / `-35.2%` VRAM) |
| `dcgan_performance_b64` | `64x64, B64, W64` | **`20.14`** | `21.38` | `3,154.6` | `235.6 / 240.0` | `25.99` | `23.46` | **`-22.5%` / `-14.2%`** |
| `dcgan_performance_w128` | `64x64, B16, W128` | **`19.42`** | `20.42` | `816.0` | `384.3 / 414.0` | `24.71` | `22.79` | **`-21.4%` / `-14.8%`** |

### 3.2 Matched Same-Session Idle-GPU1 NVML Energy Benchmark (`dcgan_performance_r256`)

All six public recipes executed sequentially on the exact same uncontended physical RTX A6000 GPU 1 (`GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`, `FormulationContext`, `DCGANGenerator(256,128,64,512)`, `DCGANCritic(256,64,512)`, `ParticlePrior(256,128,sigma=0)`, `5` warmup + `40` measured updates [`10` for `rhbi`], `20 ms` NVML `power.draw` polling):

| Recipe (`preset`) | Training Precision | Median (ms/step) | p95 (ms/step) | Mean Power (W) | Energy (`J/step`) | Peak Alloc (MiB) | Energy Delta vs `v14` / `v12` |
|---|---|---:|---:|---:|---:|---:|---:|
| **`axiom` (`axiom-v4`)** | `bf16_channels_last` | **`33.25`** | **`35.82`** | `216.83` | **`7.307`** | **`1,741.8`** | **`-29.8%` / `-26.7%`** |
| `unified_v12` | `bf16_channels_last` | `40.99` | `43.76` | `240.82` | `9.975` | `2,083.6` | `-4.1%` / baseline |
| `unified_v13` | `bf16_channels_last` | `41.39` | `45.09` | `243.25` | `10.188` | `2,083.6` | `-2.1%` / `+2.1%` |
| `unified_v14` | `bf16_channels_last` | `42.15` | `44.41` | `244.58` | `10.403` | `2,083.6` | baseline / `+4.3%` |
| `minimal` | `bf16_channels_last` | `48.97` | `51.33` | `222.41` | `10.977` | `1,993.3` | `+5.5%` / `+10.1%` |
| `rhbi` | `float32` | `350.31` | `350.77` | `273.84` | `95.989` | `2,653.1` | `13.1x` vs `axiom` |

**Why Axiom wins by 29.8% energy and 29.8–35.2% VRAM:** Clock-free Weyl-phase `b_reg = max(2, B // 4)` interleaved `fwAD` R1+R2 with `BN.eval()` dual-BF16 eliminates `autograd.grad` backward-of-backward graphs and cuts dual-batch activation memory by 4x; `_capture_ring` stays strictly zero-allocation (`0` entries) on high-res inputs (`> 4096` elements/sample); and the 3-kernel contiguous `foreach` `AxiomOptimizer` runs `0` NS on healthy spatial Conv2d with only `4.0` GPU-CPU syncs/step.

---

## 4. Detailed Per-Study Breakdown (All 9 Official `axiom-v4` Forge Receipts)

### 4.1 `axiom-v4-t3-holdout-study` (`r3-tier3-long-horizon-holdout-v1`) — **18/18 PASS (100% Sweep)** ([readout-41f9c26a](../forge/records/readout-41f9c26a18492a4834dbe767.json))

| Group | Task | Verdict | Key `axiom-v4` Metrics |
|---|---|---|---|
| Calibration smoke (5) | `two_pole`, `unused_token_hold`, `ae_gan_hold`, `clockfree_audit`, `zero_tanh_smoke` | **5/5 PASS** | `two_pole` `grad_med=0.4821`; `unused_hold=1.0000`; `ae_gan_hold` `recon_mse=0.0103`; `clockfree_audit` `4/4`; `zero_tanh_smoke` `p99=0.0157` |
| Mode & shift endurance (4) | `mode_hold`, `ring_hold`, `ring_extension`, `target_shift_recovery` | **4/4 PASS** | `mode_hold` `8/8, hq=0.9951`; `ring_hold` converged at `1400` (`200/200` streak, `1200/1200` hold, `0` fails); `ring_extension` `300/300`; `target_shift_recovery` live+frozen control PASS |
| Rail escape & discrete tokens (3) | `zero_tanh_rail_escape`, `five_word_joint_smoke`, `five_word_joint_hold` | **3/3 PASS** | `rail_escape` `p99=0.0108, sat=0.0`; `five_word_smoke` `5/5, tv=0.0190, min_tok_p=0.9999`; `five_word_hold` `5/5, tv=0.0209, qual=1.0` |
| 100-mode base (3) | `grid100`, `rotated100`, `staggered100` | **3/3 PASS** | `grid100` `prec=0.9802, tv=0.0232, bias=-0.0649, score=0.5342`; `rotated100` `prec=0.9868, tv=0.0215, bias=-0.0512, score=0.4283`; `staggered100` `prec=0.9862, tv=0.0227, bias=-0.0549, score=0.4588` |
| 14,000-update continuations (3) | **`grid100_14k`**, **`rotated100_14k`**, **`staggered100_14k`** | **3/3 PASS** | `grid100_14k` `prec=0.9851, tv=0.0211, bias=-0.0289, score=0.3269`; `rotated100_14k` `prec=0.9898, tv=0.0241, bias=-0.0317, score=0.3425`; `staggered100_14k` `prec=0.9854, tv=0.0221, bias=-0.0288, score=0.3144` |

### 4.2 `axiom-v4-bytegan-study` — **2/2 PASS** ([readout-cf0f0808](../forge/records/readout-cf0f08081668d4e14130cd6b.json))

| Task | Verdict | `unique_bytes` | `collisions` | `mass_tv` | `rms_bit_error` | `max_bit_error` | `min_bit_margin` | Extra Metrics |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `bytegan_hypercube8` | **PASS** | **`256 / 256`** | **`0`** | **`0.0000`** | **`0.00471`** (`v14`: `0.04393`) | **`0.0796`** (`v14`: `0.1605`) | **`0.9626`** (`v14`: `0.8395`) | `orphan_count=0`, `latent_nn_hamming=1.035` |
| `bytegan_bigan_dictionary` | **PASS** | **`256 / 256`** | **`0`** | **`0.0000`** | **`0.00352`** | `0.0364` | `0.9636` | `enc_bij=256`, `enc_cycle=256`, `enc_dist=0.0273`, `latent_nn_hamming=1.0000`, `recon_exact=256` |

### 4.3 `axiom-v4-t3-iso-study` — **2/2 PASS** ([readout-b6570e90](../forge/records/readout-b6570e904053603948226d93.json))

| Task | Verdict | Modes | `hq` | `mass_tv` | `core_min_eigen_ratio` | `core_component_cov_err` | Comparison |
|---|---|---:|---:|---:|---:|---:|---|
| `t3_iso_scale_mass_8x` | **PASS** | `4/4` | `0.9609` | `0.0276` | **`0.5447`** (gate `>= 0.45`) | **`0.2376`** | Beats `v12`/`v13`/`v14` (`0.4723`, cov `0.2959`), `axiom-v3` (`0.5234`), `rhbi` (FAIL), `empf` (FAIL) |
| `t3_iso_horizon_saddle_shift` | **PASS** | `4/4` | `0.9531` | `0.0625` | — | `0.6081` | `pre_shift_hq=0.9766`, `frozen_control_modes=2`; beats `minimal`/`rhbi`/`empf` (all FAIL) |

### 4.4 `axiom-v4-t4-nash-study` (**3/3 PASS**), `axiom-v4-shift-diagnostic-study` (**5/5 PASS**), and `axiom-v4-t4-arch-study` (**5/7 PASS**)

- **`t4_imbalance_nash_stretch`** ([readout-9a10b57d](../forge/records/readout-9a10b57d1dda5dd13bd560e6.json)) — **`3/3` PASS** (up from `axiom-v3` `1/3`):
  - `nash_unequal_width_hostile_imbalance`: `hq=0.9976`, `mass_tv=0.0286`, `min_eig=0.3726`, `sw1=0.0490`.
  - `nash_overpowered_critic_ring16`: `16/16` modes, `hq=0.9556`, `mass_tv=0.0400`, `min_eig=0.3521` (vs `v3` `0.1330` FAIL).
  - `nash_overpowered_generator_ring16`: `16/16` modes, `hq=0.9639`, `mass_tv=0.0454`, `min_eig=0.1791` (vs `v3` `0.1336` FAIL).
- **`t3-shift-recovery-diagnostic-v1`** ([readout-d3967c0e](../forge/records/readout-d3967c0e415122139e9783d8.json)) — **`5/5` PASS** (up from `axiom-v3` `3/5`):
  - PASS on `ae_gan_hold` (`recon_mse=0.0103`), `mode_hold` (`8/8, hq=0.9951`), `zero_tanh_smoke` (`p99=0.0157`), `zero_tanh_rail_escape` (`p99=0.0108, sat=0.0`), and `target_shift_recovery` (live recovery + matched frozen negative control).
- **`t4-strange-architectures-v1`** ([readout-18bc7392](../forge/records/readout-18bc7392296d5574083ccab4.json)) — **`5/7` PASS** (up from `axiom-v3` `4/7`):
  - **PASS (5):** `clockfree_audit_measurement_v1`, `zero_tanh_smoke`, `zero_tanh_rail_escape`, `grid100` (`score=0.5342`), and `grid100_res16_scaled_tanh` (`prec=0.9843, tv=0.0232, bias=-0.0697, score=0.5325`).
  - **FAIL (2):** `grid100_mlp16` (terminal `accuracy_pass=True`, `frozen_pass=True`, `prec=0.9845, tv=0.0259, bias=-0.0876, score=0.6385` — missed earlier live suffix) and `grid100_res16_tanh` (`prec=0.9820, tv=0.0249, sat=0.0`, `abs_cov_trace_bias=0.1184 > 0.10`).

### 4.5 `axiom-v4-disc-stab-study` (`discriminator_stability` Tier 1 + Tier 2) — **40/47 Executed PASS** ([readout-d83ce97f](../forge/records/readout-d83ce97f520f7668f39996df.json))

- **Tier 1 (`16/17` PASS, up from `11/17` in `axiom-v3`):** All `11/11` required Tier-1 tasks PASS (`gaussian1d_smoke`, `two_pole`, `unused_token_hold`, `ae_gan_hold`, `zero_tanh_smoke`, `five_word_joint_smoke`, `ring16_acquisition`, `model_ae_gan_multimodal`, `model_ae_reconstruction`, `model_vae_categorical`, `model_vae_hard`), plus `clockfree_audit_measurement_v1`, `t3_iso_scale_mass_8x`, and all **`3/3` precision quantization tasks** (`precision_gaussian_fp32` `ks=0.0204`, `precision_gaussian_mxfp8` `ks=0.0285`, `precision_gaussian_nvfp4` `ks=0.0401`). Only Tier-1 stretch miss: `interp_sparse_arc`.
- **Tier 2 (`24/30` Executed PASS; `2` downstream bimodal chain tasks unexecuted due to Gaussian chain parents):**
  - **PASS highlights:** `five_word_joint_hold` (`5/5, tv=0.0209`), `gaussian1d_stability` (`ks=0.0147, mean_err=0.0137s`), `grid100`, `rotated100`, `staggered100`, `mode_hold`, `bytegan_hypercube8`, `t3_iso_horizon_saddle_shift`, `3/3` Nash imbalance, `5/6` vectors (`vector_two_broad` `min_eig=0.9569`, `vector_unequal_width` `hq=0.9990`, `vector_unequal_mass` `hq=0.9985`, `vector_anisotropic` `hq=0.9912`, `vector_overlap` `sw1=0.0608`), `3/4` image tasks (`img_bars4` `hq=1.0, tv=0.0313`, `img_stripes2` `hq=1.0`, `img_intensity2` `hq=1.0, tv=0.0`), and all `5/5` legacy concept/trajectory holds (`cover_leftover`, `mid_scale_identity`, `residual_student`, `trajectory`, `unipolar`).
  - **Executed `FAIL`s (7):** `interp_sparse_arc` (terminal `endpoint_hq=1.0, interp_hq=0.8056` passed, missed earlier suffix), `interp_saddle_barycentric` (`barycenter_hq=0.7639 < 0.78`, `bary_mae=0.0694 <= 0.09`, still `+6.1pp` above `unified-v14`'s `0.7027`), `interp_img_patch_shift` (zero collapse: `endpoint_hq=0.0, interp_peak=0.0, endpoint_rmse=1.0`), `model_ddgan_gaussian_chain` (`final_eig_max=1.1185, final_cdf=0.0357, t1_q0_mean=0.2685, worst_cdf=0.0862`), `model_ae_ddgan_gaussian_chain` (`final_eig_max=1.0313, paired_nmse=0.0440, paired_gain=46.0x, t1_q0_cdf=0.2129`), `vector_spiral` (`sw1=0.1263, cov_err=0.4283`), and `img_blobs4` (`2/4` modes, `hq=0.5313, tv=0.2813`).

### 4.6 `axiom-v4-four-laws-adherence-study` (`four-laws-adherence-v1`, **`research_diagnostic` Scope Only**) — **22/33 Unique PASS** ([readout-9066cb43](../forge/records/readout-9066cb4353eb435be7819b0c.json))

> [!NOTE]
> `four-laws-adherence-v1` is strictly a **`research_diagnostic`** view (no ordinary qualification credit). Its 33 unique tasks combine 24 curated Forge anchors (`19/24` PASS) and 9 newly authored adversarial Four-Laws Forge tasks (`fl_*`, `3/9` PASS).

| New Adversarial Four-Laws Forge Task (`fl_*`) | Verdict | Key Observed Metrics | Failure Threshold(s) / Margin |
|---|---|---|---|
| **`fl_law2_pure_adversarial_codim1`** | **PASS** (5/5) | `manifold_cov=1.0, mae=0.0180, angular_tv=0.0571, normal_grad=0.0623, orbit_ptp=0.0244` | Clean pass on pure adversarial codim-1 manifold |
| **`fl_law42_swap_deep_critic_16l`** | **PASS** (5/5) | `modes=8/8, hq=1.0000, valley_frac=0.0, mass_tv=0.0173, r_stable=1.0011` | Clean pass on 16-layer skip-free critic swap |
| **`fl_law42_deep_resbn_and_conv1x1`** | **PASS** (5/5) | `sw1=0.0236, conv1x1_sw1=0.0210 (ratio 0.887), var_ratio=0.9713, d_max_s1=1.1327` | Clean pass on deep ResBN + 1x1 Conv2d parity |
| `fl_law3_highd_orphan_d32` | **FAIL** | `modes=5/5, precision=1.0, mass_tv=0.0310, orphan_recall=0.9318, core_eig_max=2.277, core_eig_min=0.00091` | Missed `core_eig_min >= 0.25` (collapsed minor axis in $\mathbb{R}^{32}$) |
| `fl_law41_trust_rare1pct_tanh` | **FAIL** | `modes=4/5, precision=0.9773, mass_tv=0.0370, rare_mode_recall=0.1709` | Missed `modes == 5, rare_mode_recall >= 0.45` |
| `fl_law1_aniso_unequal_modes` | **FAIL** | `modes=2, mode_recall=0.25, precision=0.4961, max_eig=3.851, min_eig=0.421` | Missed `mode_recall >= 1.0, precision >= 0.85, max_eig <= 2.2` |
| `fl_law1_noisy_atomic_codebook` | **FAIL** | `captured_atoms=43/64 (0.672), mass_tv=0.2076, rms_dist=0.2260, outlier=0.3474` | Missed `atom_recall >= 0.96, mass_tv <= 0.12, rms <= 0.08` |
| `fl_law44_highd_sparse_prior_and_shock` | **FAIL** | `mode_recall=0.0, sw1=0.1388, var_ratio=0.4907, shock_modes=0, shock_hq=0.0142` | Missed `mode_recall >= 0.75, var_ratio_err <= 0.25, shock_modes == 4` |
| `fl_law2_groupnorm_and_vit_sprites` | **FAIL** | `modes=0, sprite_hq=0.0, template_rmse=0.2789, mass_tv=0.75, grad_finite=1.0` | Missed `mode_coverage >= 1.0, sprite_hq >= 0.85` (finite gradients, stalled learning) |

---

## 5. Refreshed Four-Laws Adversarial Stress Controls (`axiom-v4` vs Legacy Families)

Full measurements on `NVIDIA RTX A6000` (`seed = 0`, `scripts/axiom_four_laws_benchmark.py`, see [four-laws-benchmark.md](four-laws-benchmark.md)):

| Structural Law Control | **`axiom-v4`** | `unified_v12` | `unified_v13` | `minimal` | `rhbi` |
|---|---:|---:|---:|---:|---:|
| **1A. 16L Skip-Free Critic Swap (`ring8` B1)** | **`8/8` · `hq=0.9417`** (**PASS**) | `8/8` · `0.5000` | `8/8` · `0.5000` | `0/8` · `0.0073` | `0/8` · `0.0000` |
| **1C. P0 Rank-1 `128x128` NS Leakage (`s=0`)** | **`s2/s1=1.29e-5` (`s1=1.124`)** (**PASS**) | `0.4484` (`s1=6.823`) | `0.4484` (`s1=6.823`) | N/A | exact polar |
| **2A. P1 24-Decade Scale Invariance & `1e-40`** | **`1.61e-6` / `u(1e-40)=0.0566`** (**PASS**) | `0.326` / `0.0` | `0.326` / `0.0` | `0.176` / `0.0` | `0.606` / `0.0` |
| **2B–D. P1 Rail / `1e4x` Spike / `1/t` Decay / Chatter** | **`1.95e-5 / 1.95e-5 / 1.000 / 0.00097`** (**PASS**) | `0.015 / 3.02 / 0.001 / 1.00` | `0.015 / 3.02 / 0.001 / 1.00` | `0.0 / 2.93 / 0.020 / 1.00` | `6.85 / 3.54 / 0.0005 / 1.00` |
| **3A–C. P4 Verified `p95` Trust + `coord_scale` + Tail** | **`0.54x / 0.0025x / 1.000x / tail=True`** (**PASS**) | `5.00x / 10.24x / 500x / False` | `5.00x / 10.24x / 500x / False` | `5.00x / 301x / 500x / False` | `5.00x / 0.00x / N/A / False` |
| **4A–B. P6 Below-$\kappa$ Unbiased & Dim-Indep $\mu_D$** | **`1.07x` (`cos=0.990`) · `r256/2d=1.06x`** (**PASS**) | `1423.3x` · `3.60e-3` | `1423.3x` · `3.60e-3` | `0.65x` · `1953x` | `0.004x` · `7.09e-3` |
| **5A–C. P7 Mass Conservation / LTV / $B_f<K$** | **`5.96e-8` / `(1.00, 1.00)` / `36/36`** (**PASS**) | `0.119` / `(0.27, 0.81)` / `0/36` | `0.119` / `(0.27, 0.81)` / `0/36` | `0.119` / `FAIL` | N/A |
| **5D. Conv2d `64x64` Latency & Host Syncs/step** | **`21.48 ms` · `4.0` syncs (`#1`)** | `22.88 ms` · `5.0` | `21.79 ms` · `5.0` | `27.96 ms` · `6.0` | `91.99 ms` · `75.0` |

---

## 6. Honest Failure Diagnosis & `axiom-v5` Engineering Roadmap

The remaining `axiom-v4` failures decompose cleanly into four concrete, non-hackable research targets for `axiom-v5`:

1. **Conditional Reverse-Diffusion Chains (`model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`):**
   - Both chains reach near-unit terminal covariances (`final_eig_max = 1.1185` / `1.0313`, `paired_nmse = 0.0440`, `paired_gain = 46.0x`), missing primarily on the noisy early transition `t1_q0` (`t1_q0_normalized_mean = 0.2685` / `0.4315`, `t1_q0_projection_cdf = 0.0862` / `0.2129`). Fixing `t1_q0` conditional fiber conditioning unlocks both Gaussian chains AND the two gated bimodal chain tasks in `disc-stab`.
2. **Manifold Interpolation & Small-Image Modes (`interp_img_patch_shift`, `interp_saddle_barycentric`, `interp_sparse_arc`, `img_blobs4`, `vector_spiral`):**
   - `interp_sparse_arc` already meets every threshold at terminal step `600` (`endpoint_hq = 1.0, interp_hq = 0.8056`), and `interp_saddle_barycentric` is just 2 triangles short (`0.7639` vs `0.78`, `bary_mae = 0.0694 <= 0.09`). However, `interp_img_patch_shift` regressed from `axiom-v3`'s diagnostic pass (`interp_peak = 0.8660`) to constant-zero collapse under `V4E`'s faster `dec=0.02` + non-2D `s_eq_base=0.05` damping on `B=16` image generators — the same interaction implicated in `img_blobs4` (`2/4` modes) and `fl_law2_groupnorm_and_vit_sprites` (`0` modes).
3. **16-Layer 100-Mode Sustained Settling & Trace Bias (`grid100_mlp16`, `grid100_res16_tanh`):**
   - `grid100_mlp16` already satisfies terminal `accuracy_pass = True` and `frozen_pass = True` (`bias = -0.0876`), missing only earlier live suffix checks due to `V4E`'s slower early 16L acquisition ramp; `grid100_res16_tanh` sits `0.0184` past the `0.10` covariance trace-bias ceiling with `0.0%` tanh saturation.
4. **High-D Anisotropic/Codebook & Rare-Tail `fl_*` Probes (`fl_law1_*`, `fl_law3_highd_orphan_d32`, `fl_law41_trust_rare1pct_tanh`, `fl_law44_highd_sparse_prior_and_shock`):**
   - `fl_law3_highd_orphan_d32` already captures `5/5` modes at `precision = 1.0, mass_tv = 0.0310, orphan_recall = 0.9318` and needs isotropic minor-axis conditioning (`core_eig_min = 0.00091 -> >= 0.25`); `fl_law41_trust_rare1pct_tanh` needs rare-1% mode discovery (`4/5` -> `5/5`); `fl_law1_noisy_atomic_codebook` (`43/64` atoms) and `fl_law1_aniso_unequal_modes` (`2` modes, `max_eig=3.85`) point to noisy-atom discrimination and anisotropic pullback covariance matching; `fl_law44_highd_sparse_prior_and_shock` highlights sparse-prior shock recovery in high dimensions.

---

## 7. Provenance, Official Receipts & Reproduction

- **Verification on `758b397a5`:** `pytest -q tests/test_axiom_*.py` -> **69 CPU + 3 CUDA PASS** (bit-exact checkpoint roundtrip across `T3IsolationFixture`, `ByteGANToyFixture`, `ByteGANBiGANFixture`, `WordFixture`, and `ConditionalChain`), zero `svd`/`eigh`/`pinv`/`torch.compile` AST calls, `2,409 / 2,500` core LOC, `scripts/check_lint.py` clean, `scripts/check_repo_bloat.py` clean, `tests/test_docs_links.py` clean, and `experiments.forge validate` clean.
- **Official `axiom-v4` Forge Readouts (`axiom-v4@0e31275ce41d`, Commit `758b397a5`):**
  - [`readout-506977051c754e8f9694b117.json`](../forge/records/readout-506977051c754e8f9694b117.json) — `axiom-v4-dcgan-perf-study` (`6/6` unique PASS; solo GPU 1 `r256`: `34.86 ms / 805.8 MiB`)
  - [`readout-41f9c26a18492a4834dbe767.json`](../forge/records/readout-41f9c26a18492a4834dbe767.json) — `axiom-v4-t3-holdout-study` (**`18/18` PASS**)
  - [`readout-cf0f08081668d4e14130cd6b.json`](../forge/records/readout-cf0f08081668d4e14130cd6b.json) — `axiom-v4-bytegan-study` (`2/2` PASS, `rms_bit_error=0.00471`)
  - [`readout-b6570e904053603948226d93.json`](../forge/records/readout-b6570e904053603948226d93.json) — `axiom-v4-t3-iso-study` (`2/2` PASS, `core_min_eigen_ratio=0.5447`)
  - [`readout-9a10b57d1dda5dd13bd560e6.json`](../forge/records/readout-9a10b57d1dda5dd13bd560e6.json) — `axiom-v4-t4-nash-study` (**`3/3` PASS**)
  - [`readout-d3967c0e415122139e9783d8.json`](../forge/records/readout-d3967c0e415122139e9783d8.json) — `axiom-v4-shift-diagnostic-study` (**`5/5` PASS**)
  - [`readout-18bc7392296d5574083ccab4.json`](../forge/records/readout-18bc7392296d5574083ccab4.json) — `axiom-v4-t4-arch-study` (`5/7` PASS)
  - [`readout-d83ce97f520f7668f39996df.json`](../forge/records/readout-d83ce97f520f7668f39996df.json) — `axiom-v4-disc-stab-study` (`40/47` executed PASS; `16/17` Tier 1 + Tier 2)
  - [`readout-9066cb4353eb435be7819b0c.json`](../forge/records/readout-9066cb4353eb435be7819b0c.json) — `axiom-v4-four-laws-adherence-study` (`22/33` unique PASS; `research_diagnostic` only)
- **Preserved Historical `axiom-v1`/`v3` Forge Readouts (`axiom-v1@69b0f27211ec`, Commit `37f537db8`):**
  - [`readout-be28b86f5953f29b4aff1d7a.json`](../forge/records/readout-be28b86f5953f29b4aff1d7a.json) (`dcgan-perf` `6/6`) · [`readout-a67cef1eb1208f825e13c67d.json`](../forge/records/readout-a67cef1eb1208f825e13c67d.json) (`t3-iso` `2/2`) · [`readout-3a4a7316efe91140aae6130a.json`](../forge/records/readout-3a4a7316efe91140aae6130a.json) (`bytegan` `2/2`) · [`readout-c5afa913aaf559f91a6d97a2.json`](../forge/records/readout-c5afa913aaf559f91a6d97a2.json) (`t3-holdout` `10/18`) · [`readout-fa20a38c3690656c64c2a7fc.json`](../forge/records/readout-fa20a38c3690656c64c2a7fc.json) (`t4-arch` `4/7`) · [`readout-cec19ecba5301eebc1e9820e.json`](../forge/records/readout-cec19ecba5301eebc1e9820e.json) (`shift-diag` `3/5`) · [`readout-f711bad4b080dc992add928d.json`](../forge/records/readout-f711bad4b080dc992add928d.json) (`t4-nash` `1/3`) · [`readout-109bbb98231a0a7a4e6903f9.json`](../forge/records/readout-109bbb98231a0a7a4e6903f9.json) (`disc-stab` `11/17` T1)
- **Reproduction Commands:**
  ```bash
  # Enqueue and drain any frozen axiom-v4 Forge study in an isolated queue:
  PYTHONPATH="$PWD" python -m experiments.forge --queue-root /tmp/forge_axiom_v4 enqueue --study axiom-v4-t3-holdout-study axiom-v4
  PYTHONPATH="$PWD" python -m experiments.forge --queue-root /tmp/forge_axiom_v4 drain --gpus 1 --workers-per-gpu 1

  # Solo idle-GPU dcgan_performance study:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" python -m experiments.forge --queue-root /tmp/forge_axiom_v4_perf enqueue --study axiom-v4-dcgan-perf-study axiom-v4
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" python -m experiments.forge --queue-root /tmp/forge_axiom_v4_perf drain --gpus 1 --workers-per-gpu 1

  # Four-Laws adversarial benchmark:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python scripts/axiom_four_laws_benchmark.py --device cuda:0
  ```
