# Unified ByteGAN Bijection Study (`unified-v13` vs `unified-v12` Control)

**Candidate:** `unified-v13` (`preset="unified_v13"`, `configs/forge/ideas/unified-v13.json`, `rev=4e17ba843948`)
**Control:** `unified-v12` (`DEFAULT_RECIPE="unified_v12"`, `configs/forge/ideas/unified-v12.json`, `rev=fb2d42843858`)
**Declared Delta:** Single recipe field change: `rhbi_self_coupling: "feydy_discrete" -> "feydy_atomic"` (zero new `Recipe` fields; `len(fields(Recipe)) == 179`; `DEFAULT_RECIPE` and `family-current-v1.json` untouched; zero frozen task/adapter/scorer edits).
**Cohort & Protocol:** Matched live CUDA evaluation (`seed = 0`, `deterministic_orthogonal`, clean live weights, `NVIDIA RTX A6000`, Python `3.15.0`, PyTorch `2.15.0`, source digest `bed1d3cdeaa7`, recipe commit `1ca6144bd`).
**Compact Evidence JSON:** [unified-bytegan-bijection.json](unified-bytegan-bijection.json) · **Studies:** [unified-v13-bytegan-study.json](../../../configs/forge/studies/unified-v13-bytegan-study.json), [unified-v13-disc-stab-study.json](../../../configs/forge/studies/unified-v13-disc-stab-study.json), [unified-v13-dcgan-perf-study.json](../../../configs/forge/studies/unified-v13-dcgan-perf-study.json).

---

## 1. Executive Summary & Merge Recommendation

> **Recommendation: CLEAR MERGE TO `develop` (as opt-in preset `unified_v13` / candidate `unified-v13`).**
> `DEFAULT_RECIPE` remains `"unified_v12"` and `configs/forge/selections/family-current-v1.json` remains untouched (Forge calibration acceptance remains `BLOCKED` under `criteria-v1.json`; any future default promotion belongs in a separate calibrated promotion stage).

| Suite / Scope (Matched Same-Cohort RTX A6000, `seed=0`) | `unified-v12` Control | `unified-v13` Candidate | Net Delta |
| :--- | :---: | :---: | :---: |
| **Target: ByteGAN Stretch (2 tasks)** | `0 / 2` PASS | **`2 / 2` PASS** | **`+2` PASS, `0` reg** |
| **Brief Evaluation Matrix (43 tasks: ByteGAN + T1 + Key T2 + r256)** | `38 / 43` PASS | **`40 / 43` PASS** | **`+2` PASS, `0` reg** |
| **All Matched Live Control Pairs (44 tasks)** | `38 / 44` PASS | **`40 / 44` PASS** | **`+2` PASS, `0` reg** |
| **Matched Non-ByteGAN Regression Parity (41 tasks)** | Baseline | **`41 / 41` bit-exact (`max_abs_diff == 0.0`)** | **`0.000e+00` drift** |
| **Full Measured `unified-v13` Suite (all 49 T1–2 + BiGAN + r256 = 51 tasks)** | — | **`47 / 51` PASS** (`0` ERROR / TIMEOUT / BLOCKED) | `7/7` extra T2 PASS |
| **Uncontended Solo GPU 0 `dcgan_performance_r256`** | `41.11 ms` (`1147.6 MiB`) | **`39.94 ms`** (`1147.6 MiB`) | **`-1.18 ms`, `0.0 MiB`** |

### Key Scientific Takeaways
1. **First unchanged global ParticleGAN recipe to solve BOTH ByteGAN 256-byte discrete bijection tasks (`0/2 -> 2/2` PASS, `5/5` terminal checks each):** `unified-v13` eliminates all `102` collisions on `bytegan_hypercube8` (`139 -> 256` valid unique bytes, `rms_bit_error 0.0875 -> 0.0445`) and achieves exact bidirectional 256-byte dictionary lookup on `bytegan_bigan_dictionary` (`177 -> 256` forward, `0 -> 256` exact reconstructions with `recon_rms_bit_error 0.8769 -> 0.0124`, `152 -> 256` encoder-to-particle bijection, `256/256` cycle matches, `encoder_nearest_dist_mean = 0.0330`).
2. **Strict `0.000e+00` bit-exact invariance on all 41 matched non-ByteGAN regression tasks:** Across all 16 Tier-1 tasks (including `five_word_joint_smoke`, `clockfree_audit_measurement_v1`, `interp_sparse_arc`, `t3_iso_scale_mass_8x`, `precision_gaussian_{mxfp8,nvfp4}`), all 24 key Tier-2 tasks (including `five_word_joint_hold`, `gaussian1d_stability`, `grid100`, `rotated100`, `staggered100`, all 4 discrete images, all 5 vectors, all 3 extreme Nash tasks, and DDGAN chains), plus `model_ae_ddgan_bimodal_chain`, every single numerical metric of `unified-v13` matches live `unified-v12` to `0.000e+00`.
3. **Zero Conv2d latency or VRAM regression:** On uncontended solo GPU 0 `dcgan_performance_r256`, `unified-v13` records `39.94 ms/step` median (`p95 = 40.63 ms`, `399.49 img/s`) and identical `1147.65 MiB` peak allocated / `1188.00 MiB` peak reserved VRAM vs matched `unified-v12` (`41.11 ms/step` median, `p95 = 42.77 ms`, `1147.65 MiB` / `1188.00 MiB`).
4. **Honest Cohort Disclosure (Bimodal DDGAN Chains):** On this current Python `3.15.0` / PyTorch `2.15.0` RTX A6000 runtime cohort, **both** matched live `unified-v12` control and `unified-v13` record identical `FAIL` curves (`max_abs_diff = 0.000e+00`) on `model_ddgan_bimodal_chain` (step `4375` `t1_q1_covariance_eigen_max = 4.6961 > 4.5`; `5/6` final checks PASS) and `model_ae_ddgan_bimodal_chain` (step `4375` `final_local_covariance_max = 2.0592 > 2.05`; `5/6` final checks PASS), whereas earlier committed readout [`readout-b51ab67c`](../records/readout-b51ab67c4d97526186e23651.json) recorded `PASS` on its prior cohort. Likewise, optional stretch tasks `interp_img_patch_shift` and `interp_saddle_barycentric` remain honest `FAIL` for both `unified-v12` and `unified-v13` (`0.000e+00` diff).

---

## 2. Target Benchmarks: ByteGAN 256-Byte Bijection (`seed = 0`, RTX A6000)

### 2.1 `bytegan_hypercube8` (Tier 2 Stretch, 4,000 Steps, Unconditional 256-Byte Codebook)

| Candidate | Verdict (Term · Total · First Pass) | `valid_unique_bytes` | `unique_bytes` | `collision_count` | `orphan_count` | `valid_particle_fraction` | `rms_bit_error` | `max_bit_error` | `mass_tv` | Wall (s) | Energy (J) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unified-v12` (control) | **FAIL** (`0/5` · `0/24` · —) | `139 / 256` | `154 / 256` | `102` | `117` | `0.8555` | `0.0875` | `0.9700` | `0.3848` | `265.7` | `52354.9` |
| **`unified-v13` (candidate)** | **PASS** (`5/5` · `18/24` · @834) | `256 / 256` | `256 / 256` | `0` | `0` | `1.0000` | `0.0445` | `0.1640` | `0.0000` | `173.9` | `15431.8` |

### 2.2 `bytegan_bigan_dictionary` (Tier 4 Stretch, 6,000 Steps, Bidirectional Joint BiGAN Dictionary)

| Candidate | Verdict (Term · Total · First Pass) | `valid_unique_bytes` | `rms_bit_error` | `recon_valid_exact_bytes` | `recon_rms_bit_error` | `encoder_particle_bijection_bytes` | `encoder_cycle_match_bytes` | `encoder_nearest_dist_mean` | `mass_tv` | Wall (s) | Energy (J) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unified-v12` (control) | **FAIL** (`0/5` · `0/24` · —) | `177 / 256` | `0.1416` | `0 / 256` | `0.8769` | `152 / 256` | `88 / 256` | `2.0930` | `0.2539` | `163.6` | `29141.9` |
| **`unified-v13` (candidate)** | **PASS** (`5/5` · `20/24` · @1250) | `256 / 256` | `0.0040` | `256 / 256` | `0.0124` | `256 / 256` | `256 / 256` | `0.0330` | `0.0000` | `251.6` | `23026.5` |

*All four ByteGAN runs completed full declared update budgets (`4,000` / `6,000` steps) with `all_finite = true`, `hooks_exercised = true`, `unintended_rng_deviations = 0`, and exact optimizer update counts across every role. Actual-training GIF SHA-256 digests (`unified-v13`): `bytegan_hypercube8` `e5ea67c8cf6ed8f0…`, `bytegan_bigan_dictionary` `ea9b77cf1d39eb20…` (kept out of Git per `scripts/check_repo_bloat.py`; historical `unified-v12` media under `reports/toy_audit/api_contract/bytegan/media/` preserved byte-for-byte).*

---

## 3. High-Resolution Conv2d Overhead Check (`dcgan_performance_r256`, Solo Idle GPU 0)

| Candidate | Verdict | Median (ms/step) | p95 (ms/step) | Throughput (img/s) | Peak Alloc (MiB) | Peak Reserved (MiB) | Energy (J) | Contended |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `unified-v12` (control) | **PASS** | `41.11` | `42.77` | `387.74` | `1147.65` | `1188.00` | `536.4` | `False` |
| **`unified-v13` (candidate)** | **PASS** | `39.94` | `40.63` | `399.49` | `1147.65` | `1188.00` | `514.2` | `False` |

---

## 4. Full Matched Regression Table (Tiers 1–2 + ByteGAN + Performance)

Every row below compares **one unchanged global recipe** (`unified-v13` vs matched live `unified-v12` control on the exact same post-merge checkout and PyTorch/CUDA runtime; `—` in `unified-v12` live column denotes the 7 extra Tier-2 tasks executed on `unified-v13` beyond the 44 matched live pairs, all `7/7` PASS matching committed `readout-b51ab67c`).

| Group | Task | `unified-v12` (Live) | `unified-v13` (Live) | Delta | `max_abs_diff` | Key `unified-v13` Metrics | Wall v12 / v13 (s) |
| :--- | :--- | :---: | :---: | :---: | ---: | :--- | ---: |
| ByteGAN | `bytegan_hypercube8` | FAIL | **PASS** | **NEW PASS** | `1.17e+02` | `valid=256/256`, `raw=256/256`, `vfrac=1.0000`, `rms=0.0445`, `coll=0` | 265.7 / 173.9 |
| ByteGAN | `bytegan_bigan_dictionary` | FAIL | **PASS** | **NEW PASS** | `2.56e+02` | `fwd=256/256`, `recon=256/256`, `recon_rms=0.0124`, `enc_bij=256/256`, `rms=0.0040` | 163.6 / 251.6 |
| T1 req | `gaussian1d_smoke` | PASS | **PASS** | 0 | `0.00e+00` | `ks=0.0283`, `std_r=1.0007`, `mean_err=0.0195` | 54.0 / 49.3 |
| T1 req | `two_pole` | PASS | **PASS** | 0 | `0.00e+00` | `grad_med=0.2633`, `mean_abs=0.9865` | 5.2 / 5.8 |
| T1 req | `unused_token_hold` | PASS | **PASS** | 0 | `0.00e+00` | `concept_move=0.9778`, `unused_hold=0.9872` | 8.7 / 9.6 |
| T1 req | `ae_gan_hold` | PASS | **PASS** | 0 | `0.00e+00` | `hold=0.0346`, `recon_mse=0.0071` | 15.6 / 12.7 |
| T1 req | `ring16_acquisition` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9902`, `modes=16`, `tv=0.0781`, `min_eig=0.3884` | 50.6 / 63.8 |
| T1 req | `five_word_joint_smoke` | PASS | **PASS** | 0 | `0.00e+00` | `modes=5`, `q=1.0000`, `tv=0.0189`, `recon=1`, `min_tok=0.9991` | 398.3 / 374.4 |
| T1 req | `zero_tanh_smoke` | PASS | **PASS** | 0 | `0.00e+00` | `mae=0.0011`, `rms=0.0025`, `sat=0.0000` | 23.7 / 29.0 |
| T1 req | `model_ae_gan_multimodal` | PASS | **PASS** | 0 | `0.00e+00` | `component_center_rms_sigma=0.7285`, `component_max_eigen_ratio=1.8789`, `component_min_eigen_ratio=0.6799` | 49.7 / 48.8 |
| T1 req | `model_ae_reconstruction` | PASS | **PASS** | 0 | `0.00e+00` | `component_center_rms_sigma=4.8541`, `component_max_eigen_ratio=25.6059`, `component_min_eigen_ratio=16.8483` | 20.1 / 31.1 |
| T1 req | `model_vae_categorical` | PASS | **PASS** | 0 | `0.00e+00` | `aggregate_effective_components=9.0077`, `decoder_mean_hq=0.9998`, `decoder_mean_mass_tv=0.0552` | 43.1 / 43.0 |
| T1 req | `model_vae_hard` | PASS | **PASS** | 0 | `0.00e+00` | `aggregate_effective_components=2.2629`, `decoder_mean_hq=0.9395`, `decoder_mean_mass_tv=0.0642` | 71.0 / 61.6 |
| T1 diag/str | `clockfree_audit_measurement_v1` | PASS | **PASS** | 0 | `0.00e+00` | `parity=4/4` | 4.2 / 4.5 |
| T1 diag/str | `interp_sparse_arc` | PASS | **PASS** | 0 | `0.00e+00` | `endpoint_hq=1.0000`, `interp_hq=0.8406` | 34.1 / 35.8 |
| T1 diag/str | `t3_iso_scale_mass_8x` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9629`, `modes=4`, `tv=0.0139`, `core_eig=0.4723` | 61.5 / 53.9 |
| T1 diag/str | `precision_gaussian_mxfp8` | PASS | **PASS** | 0 | `0.00e+00` | `ks=0.0200`, `std_r=0.9864`, `mean_err=0.0076` | 84.8 / 65.4 |
| T1 diag/str | `precision_gaussian_nvfp4` | PASS | **PASS** | 0 | `0.00e+00` | `ks=0.0360`, `std_r=1.0321`, `mean_err=0.0351` | 155.2 / 98.3 |
| T2 key | `gaussian1d_stability` | PASS | **PASS** | 0 | `0.00e+00` | `ks=0.0249`, `std_r=0.9915`, `mean_err=0.0325` | 155.5 / 233.2 |
| T2 key | `five_word_joint_hold` | PASS | **PASS** | 0 | `0.00e+00` | `modes=5`, `q=1.0000`, `tv=0.0209`, `recon=1`, `min_tok=0.9987` | 84.2 / 82.8 |
| T2 key | `mode_hold` | PASS | **PASS** | 0 | `0.00e+00` | `hq=1.0000`, `modes=8` | 36.8 / 45.9 |
| T2 key | `grid100` | PASS | **PASS** | 0 | `0.00e+00` | `prec=0.9878`, `ctr=0.0925`, `cov=0.1062`, `tv=0.0365` | 329.3 / 237.9 |
| T2 key | `rotated100` | PASS | **PASS** | 0 | `0.00e+00` | `prec=0.9867`, `ctr=0.0620`, `cov=0.1122`, `tv=0.0370` | 305.3 / 306.2 |
| T2 key | `staggered100` | PASS | **PASS** | 0 | `0.00e+00` | `prec=0.9919`, `ctr=0.0976`, `cov=0.1080`, `tv=0.0342` | 351.6 / 322.1 |
| T2 key | `img_stripes2` | PASS | **PASS** | 0 | `0.00e+00` | `modes=2`, `hq=1.0000`, `mean_rmse=0.0026` | 23.3 / 33.6 |
| T2 key | `img_bars4` | PASS | **PASS** | 0 | `0.00e+00` | `modes=4`, `hq=0.9688`, `mean_rmse=0.0112` | 37.8 / 27.4 |
| T2 key | `img_blobs4` | PASS | **PASS** | 0 | `0.00e+00` | `modes=4`, `hq=0.9688`, `mean_rmse=0.0135` | 37.9 / 23.5 |
| T2 key | `img_intensity2` | PASS | **PASS** | 0 | `0.00e+00` | `modes=2`, `hq=1.0000`, `mean_rmse=0.0098` | 23.0 / 24.0 |
| T2 key | `vector_unequal_mass` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9976`, `tv=0.1124`, `min_eig=0.6008` | 35.3 / 37.2 |
| T2 key | `vector_unequal_width` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9985`, `tv=0.1304`, `min_eig=0.4610` | 48.9 / 44.9 |
| T2 key | `vector_anisotropic` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9963`, `tv=0.1173`, `min_eig=0.7668` | 35.7 / 36.0 |
| T2 key | `vector_overlap` | PASS | **PASS** | 0 | `0.00e+00` | `sw1_normalized=0.0421`, `mean_error=0.0304`, `covariance_error=0.0620` | 47.4 / 35.3 |
| T2 key | `vector_spiral` | PASS | **PASS** | 0 | `0.00e+00` | `sw1_normalized=0.0344`, `mean_error=0.0314`, `covariance_error=0.0536` | 61.5 / 45.4 |
| T2 key | `interp_img_patch_shift` | FAIL | **FAIL** | 0 | `0.00e+00` | `endpoint_hq=0.6250`, `interp_hq=0.5789`, `interp_rmse=0.1002`, `interp_peak=0.4915` | 41.7 / 42.2 |
| T2 key | `interp_saddle_barycentric` | FAIL | **FAIL** | 0 | `0.00e+00` | `endpoint_hq=0.8750`, `barycenter_hq=0.7027`, `barycenter_surface_mae=0.0679` | 43.9 / 57.7 |
| T2 key | `t3_iso_horizon_saddle_shift` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9531`, `modes=4`, `tv=0.0781` | 71.4 / 46.5 |
| T2 key | `nash_overpowered_critic_ring16` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9592`, `modes=16`, `tv=0.0720`, `min_eig=0.3849` | 63.9 / 77.1 |
| T2 key | `nash_overpowered_generator_ring16` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9741`, `modes=16`, `tv=0.0977`, `min_eig=0.5864` | 104.5 / 89.4 |
| T2 key | `nash_unequal_width_hostile_imbalance` | PASS | **PASS** | 0 | `0.00e+00` | `hq=0.9731`, `tv=0.0857`, `min_eig=0.7456` | 44.8 / 53.1 |
| T2 key | `model_ddgan_gaussian_chain` | PASS | **PASS** | 0 | `0.00e+00` | `t1_q0_cdf=0.0692`, `final_cdf=0.0513` | 106.4 / 133.1 |
| T2 key | `model_ddgan_bimodal_chain` | FAIL | **FAIL** | 0 | `0.00e+00` | `t1_q0_cdf=0.1651`, `final_cdf=0.0517` | 295.6 / 251.7 |
| T2 key | `model_ae_ddgan_gaussian_chain` | PASS | **PASS** | 0 | `0.00e+00` | `t1_q0_cdf=0.2227`, `final_cdf=0.0722`, `nmse=0.0480` | 114.5 / 175.9 |
| T2 extra | `model_ae_ddgan_bimodal_chain` | FAIL | **FAIL** | 0 | `0.00e+00` | `t1_q0_cdf=0.1278`, `final_cdf=0.0519`, `nmse=0.0313` | 231.9 / 212.9 |
| T2 extra | `precision_gaussian_fp32` | — | **PASS** | v13 PASS | — | `ks=0.0284`, `std_r=0.9939`, `mean_err=0.0429` | — / 39.8 |
| T2 extra | `trajectory` | — | **PASS** | v13 PASS | — | `identity_mse=0.0009` | — / 454.5 |
| T2 extra | `residual_student` | — | **PASS** | v13 PASS | — | `identity_mse=0.0008`, `success_rate=1.0000`, `wrong_pad_rate=0.0000` | — / 363.9 |
| T2 extra | `unipolar` | — | **PASS** | v13 PASS | — | `cover=0.9859`, `neu_hold=0.9936`, `off_caption=0.0000` | — / 23.4 |
| T2 extra | `cover_leftover` | — | **PASS** | v13 PASS | — | `content_kept=0.9599`, `leak_ratio=0.0199`, `pole_rel_err_minus=0.0251` | — / 47.8 |
| T2 extra | `mid_scale_identity` | — | **PASS** | v13 PASS | — | `concept_cos_minus=1.0000`, `concept_cos_plus=0.9998`, `concept_mag_minus=0.9962` | — / 117.2 |
| T2 extra | `vector_two_broad` | — | **PASS** | v13 PASS | — | `hq=0.9954`, `tv=0.0068`, `min_eig=0.9413` | — / 43.7 |
| Perf | `dcgan_performance_r256` | PASS | **PASS** | 0 | `4.91e+01` | `med=39.94 ms`, `p95=40.63 ms`, `alloc=1147.6 MiB` | 8.2 / 7.8 |

---

## 5. Mechanism Summary & Governance Boundaries

All changes are strictly recipe-owned inside `particlegan/` (commit `1ca6144bd`):
- **Scale-invariant discrete-atomic support gate** (`particlegan/_t1_coupling/atomic.py:compute_discrete_atomic_gate`): returns `0.0` on continuous 1D/nD densities, curved manifolds, noisy mixtures, and single point masses, and `1.0` on finite discrete atomic distributions (`K >= 2` in any `R^D`, invariant across 6+ orders of scale).
- **Persistent 1-stays / surplus-moves bijection ledger** (`update_atomic_ledger`, `compute_atomic_quotas`): pins closest particle(s) per discovered chart via deterministic Hamilton quotas (exact unit quotas whenever `M == K`) and routes only surplus duplicates onto nearest deficit charts, with zero-radius Born-Infeld commitment, absent-chart shielding, and non-splitting streaming charts (`allow_dead_revival=False` in atomic mode).
- **Discrete atomic thermal residual**: keeps thermal cooling open while particles remain unassigned or uncrisp relative to inter-atom spacing, cooling smoothly to `1e-6` once settled.
- **Joint BiGAN inverse codebook contraction** (`split_joint_table_pair`, `match_inverse_atomic_targets`): aligns encoder codes `E(x)` to each real atom's assigned latent particle code inside `joint_relativistic_loss` with zero extra forward passes and complete checkpointing through `opt_d.record`.
- **Zero forbidden primitives & full checkpointing:** Strictly `0` `torch.autograd.grad`, `0` SVD/`eigh`/`pinv`, clock-free (`clockfree_audit_measurement_v1` `4/4` PASS), and verified bit-exact across fresh-object checkpoint save/load in [`tests/test_unified_bytegan.py`](../../../tests/test_unified_bytegan.py).

### Reproduction
```bash
python -m experiments.forge plan unified-v13 --study unified-v13-bytegan-study
python -m experiments.forge plan unified-v13 --study unified-v13-disc-stab-study
python -m experiments.forge plan unified-v13 --study unified-v13-dcgan-perf-study
pytest -q tests/test_unified_bytegan.py tests/test_forge_bytegan.py tests/test_unified_v11.py tests/test_unified_v12.py
```
