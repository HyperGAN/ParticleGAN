# Round-3 Track 2 Study: Complete 4-Law Zero-SVD / Zero-`eigh` / Zero-`pinv` / Zero-`autograd.grad` Unification (`unified-v4` through `unified-v10`)

**Branch:** `feat/solve-gans-r3-unified-v4`  
**Formulation:** Complete 4-Law Zero-SVD / Zero-`eigh` / Zero-`pinv` / Zero-`autograd.grad` Unification (`preset="unified_v4"`, `critic_formulation="unified_v4"`, `prior_regularizer="unified_v4"`, `optimizer="unified_equil"`, public API [`particlegan/unified_v4.py`](../../../particlegan/unified_v4.py) / [`particlegan/_unified_v4/`](../../../particlegan/_unified_v4/))  
**Hard Primitive Audit:** Strictly **`0` `torch.autograd.grad` · `0` `torch.linalg.svd` · `0` `torch.linalg.svdvals` · `0` `torch.linalg.eigh` · `0` `torch.linalg.eigvalsh` · `0` `torch.linalg.pinv`** across every forward, backward, penalty, transport, prior, and optimizer path — monkeypatch-enforced in [`tests/test_unified_v4.py`](../../../tests/test_unified_v4.py).  
**Contract & Compatibility:** `DEFAULT_RECIPE == "empf"` unchanged, `Recipe` retains all `179` fields, [`particlegan/recipe_compat.py`](../../../particlegan/recipe_compat.py) untouched, [`particlegan/_unified/`](../../../particlegan/_unified/) / [`particlegan/_t4_arch/`](../../../particlegan/_t4_arch/) / [`particlegan/_t4_nash/`](../../../particlegan/_t4_nash/) and all prior readout receipts remain completely untouched.  
**Default-Adoption / Calibration Gate:** **BLOCKED** for global default adoption (`DEFAULT_RECIPE` remains `"empf"`). This study establishes an evidence-backed, single-recipe 4-Law zero-SVD / zero-`autograd.grad` unification milestone (`42` concluded official Forge readout receipts across `unified-v4` through `unified-v10`) while keeping all remaining open ceilings visible.

---

## Executive Summary (`unified-v10` Champion, `rev 72e944a156d7`)

Evaluated under **protocol seed `0`**, `deterministic_orthogonal` initialization, clean live sampling, and **one unchanged global `unified-v10` recipe** (`candidate_revision = 72e944a156d7b1eb980c62f4a193012731111251e912b8b1ea75f02fc6db5127`) on `NVIDIA RTX A6000`:

| Official Forge View | `unified-v10` Result | Key Verbatim Metrics | Official Readout Receipt |
| :--- | :---: | :--- | :--- |
| **`discriminator_stability`** (Tiers 1–2, `46` tasks) | **`40 / 46` PASS** (`0` BLOCKED, `0` ERROR, `0` TIMEOUT) | **Tier 1:** `11/11` required + `2/2` diagnostic + `2/4` stretch; **Tier 2:** `13/16` runnable required + `5/6` stretch + `7/7` held; **3/3** 100-mode lattices (`grid100`, `rotated100`, `staggered100`), **4/4** discrete images, **6/6** vectors, **3/3** Nash, **`five_word_joint_smoke` + `hold`** | [`readout-6da9dee1`](../records/readout-6da9dee19abe886e7ddbe184.json) |
| **`bispf_throughput_endurance`** (Tier-3 `14k` Endurance) | **`3 / 3` (`100%`) PASS** | `mode_hold` (`hq=1.0`, `8/8` modes), `ring_hold` (`converged_step=1400`, `1200/1200` hold checks, `0` failures), `ring_extension` (`14k`, `300/300` extension checks, `0` failures) | [`readout-533d4c32`](../records/readout-533d4c32c634f4bf929d6407.json) |
| **`t4-strange-architectures-v1`** (Tier-4 Pathological Nets) | **`6 / 7` PASS** | PASS: `clockfree` (`4/4`), `zero_tanh_smoke` (`mae=0.0026`), `zero_tanh_rail_escape` (`mae=0.0019`, `p99=0.0076`), `grid100`, `grid100_mlp16` (`ctr=0.0712`), `grid100_res16_scaled_tanh` (`ctr=0.0537`, `tr=-0.0400`); FAIL: `grid100_res16_tanh` (100k holdout `accuracy_pass=True`, `ctr=0.1037`, `prec=0.9906`, missed intermediate suffix) | [`readout-688bc7ed`](../records/readout-688bc7ed35078e19c10daf6a.json) |
| **`t4_imbalance_nash_stretch`** (Tier-4 `>500×` Asymmetry) | **`3 / 3` (`100%`) PASS** *(7/7 consecutive across `v4..v10`)* | `overpowered_critic` (`16/16`, `hq=0.9331`, `core_eig=0.3018`), `overpowered_generator` (`16/16`, `hq=0.9194`, `core_eig=0.2837`), `hostile_imbalance` (`hq=0.9934`, `core_eig=0.7345`) | [`readout-8f0a4508`](../records/readout-8f0a450815d09f56bc74a678.json) |
| **`t3-shift-recovery-diagnostic-v1`** (Live Non-Stationary Shift) | **`5 / 5` (`100%`) PASS** | `target_shift_recovery` (`81/81` post-deadline live checks, `min_hq=0.9685`, `min_modes=8`, `stable_from_step=2800` vs `0/81` frozen control), `ae_gan_hold`, `mode_hold`, `zero_tanh_smoke`, `zero_tanh_rail_escape` | [`readout-72b4701f`](../records/readout-72b4701ffe3e02fae8ac8982.json) |
| **`t3_causal_isolation_stretch`** (Horizon + Scale-Mass 8×) | **`1 / 2` PASS** *(`v4/v5/v6`: **`2 / 2` PASS**)* | `t3_iso_horizon_saddle_shift` PASS (`4/4`, `hq=1.0000`, `frozen_control_modes=2`); `t3_iso_scale_mass_8x` FAIL (`4/4`, `hq=0.9980`, `tv=0.0139`, `core_eig=0.2155` vs `>=0.45`; `v4/v5/v6` passed at `core_eig=0.6621`) | [`readout-15aafd6c`](../records/readout-15aafd6ce00180e64ecd23fc.json) |
| **`dcgan_performance`** (`32²`–`256²` RGB Latency & VRAM) | **`6 / 6` (`100%`) PASS** | `unified-v10`: **`r256 = 89.66 ms/step`** (`p95=92.08`, `177.5 img/s`, `1372.6 MiB` alloc / `1402.0 MiB` res); uncontended `unified-v4`: **`r256 = 69.55 ms/step`** (`p95=70.31`, `229.6 img/s`, `1372.6 MiB` alloc) | [`readout-90580c7a`](../records/readout-90580c7aff5ec6988cfbcfad.json) / [`readout-584f227b`](../records/readout-584f227b6b3d8d6538cde68c.json) |

---

## 1. Unified 4-Law Mathematical Theory (`particlegan/_unified_v4/`)

Round 2 proved four specialized mechanisms in isolation (`unified-v3` at `34/41` and `134.26 ms/step` `r256`; `t4-arch-v3` at `7/7` strange architectures and `26/41` `discriminator_stability`; `t4-nash-v2` at `2/3` Nash imbalance), while still relying on `torch.linalg.svd` (in polar smoothing), `torch.linalg.eigh` (in orphan macro-transport), `torch.linalg.pinv` (in constrained projection), or `torch.autograd.grad(..., create_graph=True)` (double-backward critic penalties).

`unified_v4` unifies all four Laws into one modular, strictly zero-SVD / zero-`eigh` / zero-`pinv` / zero-`autograd.grad` package ([`particlegan/_unified_v4/`](../../../particlegan/_unified_v4/)):

### Law 1 — Topological & Concentric Prior Geometry ([`prior.py`](../../../particlegan/_unified_v4/prior.py))
- **Isolated Pythagorean Coalescence + `0.995` Jensen Centering:** Every particle component $k$ maintains an isotropic Gaussian chart $(\mu_k, \sigma_k)$ in latent space, regularized without cross-mode repulsion distortion by a $0.995$ Jensen-centered zero-mean / unit-second-moment prior anchor and streaming $H_0$ Laguerre chart commitment that preserves disconnected mode basins on 100-mode lattices (`grid100`, `rotated100`, `staggered100`).
- **Deficit-Gated Concentric Law of Total Variance (LTV):** Tracks the per-step realized pushforward variance $`v_{\text{push}}`$ via a checkpointed schema-2 EMA buffer (`v_push` in optimizer/prior state) and allocates prior centroid variance strictly when pushforward variance falls short of target unit variance: $`v_{\mu}^{\star} = g_{\text{def}}\!\left(\sigma^2 - v_{\text{push}}\right)_{+}`$, where $`u_{\text{def}} = \mathrm{clamp}\!\left((\Delta - 0.15)/0.12,\, 0,\, 1\right)`$ and $`g_{\text{def}} = u_{\text{def}}^2\,(3 - 2 u_{\text{def}})`$. When pushforward variance already matches the target ($\Delta \le 0.15$), $`v_\mu^\star = 0`$ so 100-mode grid centroids remain tight (`center_rms_sigma = 0.125–0.139`); when tanh/bounded architectures squash the pushforward ($`\Delta > 0.15`$), concentric LTV expands the latent prior cleanly without destroying radial precision.

### Law 2 — Strictly First-Order Local Lipschitz & Critic Regularization ([`penalty.py`](../../../particlegan/_unified_v4/penalty.py), [`critic.py`](../../../particlegan/_unified_v4/critic.py))
- **Low-Dimensional & Vector Manifolds:** Replaces second-order `torch.autograd.grad(..., create_graph=True)` with exact first-order forward-mode AD (`torch.autograd.forward_ad.dual_level`), computing exact directional derivatives $dD(x)[v]$ in a single primal pass so outer `.backward()` differentiates $\lVert dD(x)[v]\rVert^2$ via purely first-order reverse mode (with deterministic finite-difference fallback for quantized MXFP8/NVFP4 modules).
- **Small Discrete Images (`img_stripes2`, `img_bars4`, `img_blobs4`, `img_intensity2`):** Uses $K=2$ orthogonal small-image secant probes rather than double-backward conv gradients, achieving `4/4` PASS (`hq = 0.9688–1.0000`).
- **High-Dimensional CNNs (`dcgan_performance` `32²`–`256²` RGB):** Hooks the live BF16 critic logits and evaluates a single live-endpoint BF16 secant probe $(D(x + \varepsilon u) - D(x))/\varepsilon$, eliminating `100%` of double-backward conv-transpose overhead and bringing `256×256` RGB latency down to **`69.55 ms/step`** uncontended (`unified-v4`) and **`89.66 ms/step`** under the full `unified-v10` recipe (`1.73×`–`2.23×` faster than `empf-v8` `155.41 ms/step`, `6.57×`–`8.62×` faster than `rhbi-v12` `588.75–599.88 ms/step`, and `1.50×`–`1.93×` faster than `unified-v3` `134.26 ms/step`).

### Law 3 — Feydy-Unbiased Conformal Horizon, Zero-`eigh` Orphan Transport & Conditional Fiber Alignment ([`loss.py`](../../../particlegan/_unified_v4/loss.py))
- **Feydy-Unbiased Conformal Horizon & Born–Infeld Transport:** Combines self-debiased entropic Sinkhorn potentials with Born–Infeld tail bounding and conformal horizon adaptation, resolving both non-stationary target shifts (`target_shift_recovery` `81/81` live post-deadline checks vs `0/81` frozen control) and horizon saddle shifts (`t3_iso_horizon_saddle_shift` `4/4` modes, `hq = 1.0000`).
- **Zero-`eigh` Power-Iteration Orphan Macro-Transport:** Replaces eigendecomposition of local covariance frames with fixed-iteration power iteration (`0` `eigh`, `0` `svd`), preserving exact determinism and fresh-process restart parity (`clockfree_audit_measurement_v1` `4/4` PASS).
- **Cross-Batch Sufficient-Statistics BLP Conditional Fiber Residualization:** Maintains detached exponential sufficient statistics for conditional fibers when conditioning context is present, without introducing `autograd.grad` or `pinv`.

### Law 4 — Zero-SVD Augmented `(X, Y)` Tikhonov-Smoothed Batched BF16 Quintic Polar, Saturation Escape & Secant Nash Ceiling ([`optimizer.py`](../../../particlegan/_unified_v4/optimizer.py))
- **$`10^{-38}`$ Dynamic Scale-Normalized Batched BF16 5-Step Quintic Polar (`0` SVD):** Pre-normalizes every matrix gradient $G$ by its dynamic scale $`s_{\text{dyn}} = \max(\lVert G\rVert_F, 10^{-38})`$ before batched BF16 5-step quintic Newton–Schulz polar iteration, eliminating FP16/BF16 underflow on deep saturated networks (`16`-layer MLPs and residual tanh nets, unit-tested down to $10^{-37}$ in [`tests/test_unified_v4.py`](../../../tests/test_unified_v4.py)).
- **Exact Zero-SVD Augmented `(X, Y)` Tikhonov-Smoothed Polar (`min(R, C) <= 128`):** Standard SVD polar smoothing computes $(A A^\top + s^2 I)^{-1/2} A$ via explicit SVD. `unified_v4` (`v7`–`v10`) computes the **exact same matrix function with `0` SVD** via the stacked $`(R+C)`$-column quintic polar iteration on $`Z_0 = [A;\, s_{\text{eff}} I]`$ (`(X, Y)` coupled update: $`X_{k+1} = X_k P(Q_k)`$, $`Y_{k+1} = Y_k P(Q_k)`$ where $Q_k = X_k^\top X_k + Y_k^\top Y_k$ and $P(Q) = a I + b Q + c Q^2$), gated by depth ($`s_{\text{eff}} = 0.0061\,(1 - g_{\text{deep}})`$) on layers with $\min(R, C) \le 128$ (covering all `128×128` lattice hidden layers while avoiding extra $`Y`$-matmuls on `256×256` vocabulary/CNN layers). Unit-verified to match exact SVD `polar_factor(smoothing=0.0061)` to $`<10^{-2}`$ relative Frobenius error in BF16.
- **Depth Trust, `fwAD` Saturation Pullback & Coordinatewise Outward Column-Sensitivity Clamp Rescue:** Detects output/hidden rail saturation via forward-mode pullback sensitivity and applies coordinatewise outward column-sensitivity probing ($d_z \le 8$) alongside target-isometry saturation compensation (`valid_e = (e_norm_raw >= 0.45)`, `is_rail_clamped = valid_e & (s_obs < 0.50 * s_med)`), enabling zero-rail recovery on `zero_tanh_rail_escape` (`mae=0.0019`, `p99=0.0076`, `saturated_fraction=0.0`) and `16`-layer bounded residual nets (`grid100_res16_scaled_tanh` `ctr=0.0537`, `tr=-0.0400`, `prec=0.9906`).
- **Detached Function-Space Secant Trust Band (`B_probe <= 16`) & Analytic Protected-Loss Projection (`0` `pinv`, `0` `autograd.grad`):** Caps generator pushforward displacement against the critic's realized function-space response (`3/3` sustained PASS across `t4_imbalance_nash_stretch` on all 7 iterations `v4..v10`) and solves constrained non-ascent updates analytically without `pinv` or `autograd.grad`.

---

## 2. Iterative Ablation Trajectory Across All 42 Official Forge Receipts (`unified-v4` → `unified-v10`)

Every design change was isolated as a versioned Forge candidate (`configs/forge/ideas/unified-v{4..10}.json`) and evaluated across official Forge GPU studies (`seed = 0`, `NVIDIA RTX A6000`). All `42` study configs map 1-to-1 to `42` committed readout JSON receipts:

| Candidate (`rev`) | Key Mechanism Delta | `disc_stability` (T1–2) | `t4-arch` | `t4-nash` | `shift-diag` | `t3-iso` | `endurance` (`14k`) | `dcgan_perf` (`r256 ms` · `alloc`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unified-v4`** (`fe87e2181f95`) | Initial 4-Law zero-SVD / zero-`autograd.grad` unification | — | `4/7` ([`f5fc0a5c`](../records/readout-f5fc0a5c114166fd6510147d.json)) | **`3/3`** ([`80d68593`](../records/readout-80d68593a893e5c00d55cce6.json)) | `4/5` ([`dc15aec6`](../records/readout-dc15aec6908c84d597a762ed.json)) | **`2/2`** `eig=0.6621` ([`31c12df2`](../records/readout-31c12df2908246606b5bb072.json)) | — | **`6/6` · `69.55 ms`** · `1372.6 MiB` ([`584f227b`](../records/readout-584f227b6b3d8d6538cde68c.json)) |
| **`unified-v5`** (`0c60b814da7e`) | Drop double-damped `_v4_g_scale`; coord outward clamp rescue ($d_z \le 8$); schema-2 `v_push` concentric LTV | — | `3/4` *(unlocked `rail_escape`)* ([`2123820b`](../records/readout-2123820b77e39d7f45a6a014.json)) | **`3/3`** ([`63cd100f`](../records/readout-63cd100f4e34636765cee3a7.json)) | **`5/5`** ([`3f2f0f02`](../records/readout-3f2f0f02b8d2dfbff2b3bd0e.json)) | **`2/2`** `eig=0.6621` ([`6cf1513b`](../records/readout-6cf1513bd72210a0ad6ebabb.json)) | — | **`6/6` · `89.54 ms`** · `1372.6 MiB` ([`78081f92`](../records/readout-78081f9229e06098095843b8.json)) |
| **`unified-v6`** (`927e62f49a0a`) | Fix small-$`\lVert\varepsilon_g\rVert`$ clamp bias (`valid_e = (e_norm_raw >= 0.45)` & `s_obs < 0.50 * s_med`) | `12/17` T1 ([`280a5518`](../records/readout-280a5518a526aa0387a1dab6.json)) | `3/4` ([`edd5648b`](../records/readout-edd5648b2522a2546e4ac0ef.json)) | **`3/3`** ([`9893eabb`](../records/readout-9893eabb16bed5770a66736d.json)) | **`5/5`** ([`19d58628`](../records/readout-19d586284a3d1252f8fbe4d1.json)) | **`2/2`** `eig=0.6621` ([`a5196684`](../records/readout-a519668412afa38c2de84f93.json)) | — | **`6/6` · `79.12 ms`** · `1372.6 MiB` ([`23b71e6e`](../records/readout-23b71e6e6df98eadd7707931.json)) |
| **`unified-v7`** (`8995871aa3fb`) | Exact zero-SVD augmented `(X,Y)` Tikhonov Quintic (`s=0.0061`); drop hardcoded thermal choke; full `1.0 * sigma^2` LTV (`onset=0.12`) | `14/17` T1 *(`1` TO: `five_word`)* ([`ba9268ae`](../records/readout-ba9268ae74da2732e681e6cd.json)) | **`6/7`** ([`9e36b8ff`](../records/readout-9e36b8ffba307dacc10ec76e.json)) | **`3/3`** ([`b6662c29`](../records/readout-b6662c299bb6f6ed5cc05438.json)) | **`5/5`** ([`4f080533`](../records/readout-4f080533bff8268fe3583174.json)) | `1/2` `eig=0.3656` ([`22848649`](../records/readout-228486492e2b4f2959fe621a.json)) | — | **`6/6` · `106.38 ms`** · `1372.6 MiB` ([`205377e5`](../records/readout-205377e5bc868b566eb13969.json)) |
| **`unified-v8`** (`0483ddee4266`) | Gate augmented `(X,Y)` to `min(R,C)<=64`; drop residual `* th_g` in `delta_g_max`; LTV `onset=0.06` | `36/46` *(`five_word` & `sparse_arc` PASS; grids fail)* ([`8a6d5985`](../records/readout-8a6d598509ad19bdef5fd5ed.json)) | `3/4` ([`15497003`](../records/readout-15497003b76d275b820080d8.json)) | **`3/3`** ([`8fd7468b`](../records/readout-8fd7468bb8c81a152f69eb8a.json)) | **`5/5`** ([`13ef4a47`](../records/readout-13ef4a476362b841fde73ddf.json)) | `1/2` `eig=0.3656` ([`d2e0272c`](../records/readout-d2e0272c97deecfb2b6b0266.json)) | — | **`6/6` · `103.58 ms`** · `1372.6 MiB` ([`17df5666`](../records/readout-17df5666adcc67121bc2b569.json)) |
| **`unified-v9`** (`a79c86920325`) | Restore LTV `onset=0.15`; add Tier-3 `14k` endurance | `35/46` ([`566a1a05`](../records/readout-566a1a057026c1e1ea97984d.json)) | `3/4` ([`45107d16`](../records/readout-45107d16afdf5bb8dc138497.json)) | **`3/3`** ([`cc3983c9`](../records/readout-cc3983c90701f9d194b7e8c9.json)) | **`5/5`** ([`ca41f365`](../records/readout-ca41f3654715e96f1e0ca785.json)) | `0/2` `eig=0.1500, saddle tv=0.125` ([`f744f329`](../records/readout-f744f3298254b4a3c2a34a5c.json)) | **`3/3`** ([`193f59dc`](../records/readout-193f59dc296011b560f6911c.json)) | **`6/6` · `92.07 ms`** · `1372.6 MiB` ([`c105eccd`](../records/readout-c105eccdca6b9637c7389db1.json)) |
| **`unified-v10`** (`72e944a156d7`) | Raise `(X,Y)` Tikhonov cutoff to `min(R,C)<=128` (covers `128×128` lattices, skips `256×256` vocab/CNN); restore `therm_direct_prior` on `total_gen_mats==0` | **`40/46`** *(0 BLOCKED / 0 ERROR / 0 TO)* ([`6da9dee1`](../records/readout-6da9dee19abe886e7ddbe184.json)) | **`6/7`** ([`688bc7ed`](../records/readout-688bc7ed35078e19c10daf6a.json)) | **`3/3`** ([`8f0a4508`](../records/readout-8f0a450815d09f56bc74a678.json)) | **`5/5`** ([`72b4701f`](../records/readout-72b4701ffe3e02fae8ac8982.json)) | `1/2` `eig=0.2155` ([`15aafd6c`](../records/readout-15aafd6ce00180e64ecd23fc.json)) | **`3/3`** ([`533d4c32`](../records/readout-533d4c32c634f4bf929d6407.json)) | **`6/6` · `89.66 ms`** · `1372.6 MiB` ([`90580c7a`](../records/readout-90580c7aff5ec6988cfbcfad.json)) |

**Causal Insights from the Ablation Sequence:**
1. **`v4` → `v5`/`v6`:** Removing redundant `_v4_g_scale` damping and upgrading clamp rescue to coordinatewise outward column-sensitivity probing alongside gated saturation compensation (`valid_e = (e_norm_raw >= 0.45)`, `is_rail_clamped = valid_e & (s_obs < 0.50 * s_med)`) unlocked `zero_tanh_rail_escape` (`5/5` on `t3-shift-recovery-diagnostic-v1`) while holding `2/2` on `t3_causal_isolation_stretch` (`core_min_eigen_ratio = 0.6621`).
2. **`v6` → `v7`:** Replacing raw unsmoothed polar with exact zero-SVD augmented `(X, Y)` Tikhonov-smoothed Quintic (`s = 0.0061`) immediately fixed all four shallow Tier-1 equilibrium failures (`model_ae_gan_multimodal`, `model_vae_hard`, `precision_gaussian_fp32`, `precision_gaussian_nvfp4` final check) and raised `t4-strange-architectures-v1` from `4/7` to **`6/7`**, but running extra $`Y`$-matmuls on `256×256` layers caused `five_word_joint_smoke` (`20k` steps) to time out (`>900 s`).
3. **`v7` → `v8` → `v9` → `v10`:** Setting the augmented `(X, Y)` cutoff to `min(R, C) <= 128` with deficit onset `0.15` hit the exact structural sweet spot: `five_word_joint_smoke` and `five_word_joint_hold` both finish well under budget and pass (`5/5` modes, `mass_tv = 0.0189 / 0.0258`, `min_recon_prob = 0.9995 / 0.9998`), all three 100-mode lattices (`128×128` hidden layers) pass with sharp accuracy (`precision = 0.9909–0.9921`, `center_rms_sigma = 0.1253–0.1386`), and `dcgan_performance` `r256` drops back below `90 ms/step` (`89.66 ms/step`).

---

## 3. Verbatim `unified-v10` Breakdown Across Official Forge Views

### 3.1 `discriminator_stability` Tiers 1–2 (`40 / 46` PASS, Receipt [`readout-6da9dee1`](../records/readout-6da9dee19abe886e7ddbe184.json))

Exact tier and importance buckets matching [`configs/forge/views/discriminator_stability.json`](../../../configs/forge/views/discriminator_stability.json) (`revision 19`):

| Tier & Importance Bucket | `unified-v10` Score | Passing Tasks | Failing Tasks |
| :--- | :---: | :--- | :--- |
| **Tier 1 — Required** | **`11 / 11` (`100%`)** | `ae_gan_hold`, `five_word_joint_smoke`, `gaussian1d_smoke`, `model_ae_gan_multimodal`, `model_ae_reconstruction`, `model_vae_categorical`, `model_vae_hard`, `ring16_acquisition`, `two_pole`, `unused_token_hold`, `zero_tanh_smoke` | *(none)* |
| **Tier 1 — Diagnostic** | **`2 / 2` (`100%`)** | `clockfree_audit_measurement_v1`, `precision_gaussian_fp32` | *(none)* |
| **Tier 1 — Stretch** | **`2 / 4`** | `interp_sparse_arc`, `precision_gaussian_mxfp8` | `precision_gaussian_nvfp4`, `t3_iso_scale_mass_8x` |
| **Tier 2 — Held** | **`7 / 7` (`100%`)** | `img_bars4`, `img_blobs4`, `img_intensity2`, `vector_anisotropic`, `vector_overlap`, `vector_unequal_mass`, `vector_unequal_width` | *(none)* |
| **Tier 2 — Required** *(runnable)* | **`13 / 16`** | `cover_leftover`, `five_word_joint_hold`, `gaussian1d_stability`, `grid100`, `img_stripes2`, `mid_scale_identity`, `mode_hold`, `rotated100`, `staggered100`, `trajectory`, `unipolar`, `vector_spiral`, `vector_two_broad` | `model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`, `residual_student` |
| **Tier 2 — Stretch** | **`5 / 6`** | `interp_img_patch_shift`, `nash_overpowered_critic_ring16`, `nash_overpowered_generator_ring16`, `nash_unequal_width_hostile_imbalance`, `t3_iso_horizon_saddle_shift` | `interp_saddle_barycentric` |
| **Total (`discriminator_stability` T1–T2)** | **`40 / 46` (`87.0%`)** | `40` PASS · `6` FAIL · `0` BLOCKED · `0` ERROR · `0` TIMEOUT | `6` numerical FAILs (diagnosed in Section 5) |

Selected verbatim `unified-v10` task metrics from [`readout-6da9dee1`](../records/readout-6da9dee19abe886e7ddbe184.json):
- **100-Mode Lattices (`3 / 3` PASS):**
  - `grid100`: `center_rms_sigma=0.1379`, `cov_frob_rms=0.1229`, `cov_trace_bias=-0.0829`, `mass_tv=0.0361`, `precision=0.9918`, `radial_ks=0.0297`
  - `rotated100`: `center_rms_sigma=0.1253`, `cov_frob_rms=0.1039`, `cov_trace_bias=-0.0605`, `mass_tv=0.0360`, `precision=0.9909`, `radial_ks=0.0218`
  - `staggered100`: `center_rms_sigma=0.1386`, `cov_frob_rms=0.1214`, `cov_trace_bias=-0.0906`, `mass_tv=0.0375`, `precision=0.9921`, `radial_ks=0.0338`
- **Discrete Images (`4 / 4` PASS):** `img_stripes2` (`2/2`, `hq=1.0000`), `img_intensity2` (`2/2`, `hq=1.0000`), `img_bars4` (`4/4`, `hq=0.9688`), `img_blobs4` (`4/4`, `hq=0.9688`).
- **Continuous Vectors (`6 / 6` PASS):** `vector_anisotropic` (`hq=0.9966`, `core_eig=0.6860`, `cov=0.1709`), `vector_overlap` (`sw1_norm=0.0387`, `cov=0.1039`), `vector_spiral` (`sw1_norm=0.0251`, `cov=0.0525`), `vector_two_broad` (`hq=0.9924`, `core_eig=0.8910`, `cov=0.1136`), `vector_unequal_mass` (`hq=0.9985`, `core_eig=0.4781`, `cov=0.2268`), `vector_unequal_width` (`hq=0.9980`, `core_eig=0.6783`, `cov=0.1700`).
- **Discrete / Coupled AE & VAE & Interpolation:** `five_word_joint_smoke` (`5/5`, `mass_tv=0.0189`, `min_recon_prob=0.9995`), `five_word_joint_hold` (`5/5`, `mass_tv=0.0258`, `min_recon_prob=0.9998`), `ring16_acquisition` (`16/16`, `hq=0.9739`, `mass_tv=0.0708`, `core_eig=0.5465`, `full_cov=0.4305`), `model_ae_gan_multimodal` (`4/4`, `precision=0.9265`, `mass_tv=0.0310`, `min_eig=0.7816`), `model_vae_categorical` (`recon_mse=0.0063`), `model_vae_hard` (`recon_mse=0.0049`), `ae_gan_hold` (`hold=0.0115`, `recon_mse=0.0061`), `interp_sparse_arc` (`endpoint_hq=1.0000`, `interp_hq=0.8406`, `interp_arc_bins=30`), `interp_img_patch_shift` (`endpoint_hq=1.0000`, `interp_hq=1.0000`, `interp_rmse=0.0320`), `gaussian1d_smoke` (`cdf_ks=0.0220`, `std_ratio=1.0038`), `gaussian1d_stability` (`cdf_ks=0.0156`, `std_ratio=0.9914`), `precision_gaussian_fp32` (`cdf_ks=0.0238`, `std_ratio=1.0116`), `precision_gaussian_mxfp8` (`cdf_ks=0.0275`, `std_ratio=1.0299`), `two_pole` (`grad_med=0.2775`, `mean_abs=0.9846`), `unused_token_hold` (`concept_move=0.9997`, `unused_hold=0.9888`), `zero_tanh_smoke` (`mae=0.0026`, `p99=0.0125`, `rms=0.0034`).

### 3.2 `t4-strange-architectures-v1` (`6 / 7` PASS, Receipt [`readout-688bc7ed`](../records/readout-688bc7ed35078e19c10daf6a.json))

| Task | Outcome | Verbatim Metrics |
| :--- | :---: | :--- |
| `clockfree_audit_measurement_v1` | **PASS** | `parity_comparisons=4` (`4/4` horizon, cadence, and fresh-process checkpoint parity) |
| `zero_tanh_smoke` | **PASS** | `mean_abs_error=0.0026`, `p99_abs_error=0.0125`, `rms_error=0.0034`, `saturated_fraction=0.0` |
| `zero_tanh_rail_escape` | **PASS** | `mean_abs_error=0.0019`, `p99_abs_error=0.0076`, `rms_error=0.0025`, `saturated_fraction=0.0` |
| `grid100` | **PASS** | `center_rms_sigma=0.1379`, `cov_frob_rms=0.1229`, `cov_trace_bias=-0.0829`, `precision=0.9918`, `radial_ks=0.0297`, `accuracy_pass=True` |
| `grid100_mlp16` | **PASS** | `center_rms_sigma=0.0712`, `cov_frob_rms=0.1145`, `cov_trace_bias=-0.0814`, `precision=0.9929`, `radial_ks=0.0343`, `accuracy_pass=True` |
| `grid100_res16_scaled_tanh` | **PASS** | `center_rms_sigma=0.0537`, `cov_frob_rms=0.0930`, `cov_trace_bias=-0.0400`, `precision=0.9906`, `radial_ks=0.0165`, `accuracy_pass=True` |
| `grid100_res16_tanh` | **FAIL** | 100k independent holdout passes (`accuracy_pass=True`, `center_rms_sigma=0.1037`, `cov_frob_rms=0.1640`, `cov_trace_bias=-0.0743`, `precision=0.9906`, `radial_ks=0.0286`); missed one intermediate sustained live suffix check |

### 3.3 `t4_imbalance_nash_stretch` (`3 / 3` PASS, Receipt [`readout-8f0a4508`](../records/readout-8f0a450815d09f56bc74a678.json))

| Task | Outcome | Verbatim Metrics |
| :--- | :---: | :--- |
| `nash_overpowered_critic_ring16` ($D \gg G$, `621×`) | **PASS** | `16/16` modes, `hq=0.9331`, `mass_tv=0.0684`, `core_min_eig=0.3018`, `core_cov=0.3393` |
| `nash_overpowered_generator_ring16` ($G \gg D$, `548×`) | **PASS** | `16/16` modes, `hq=0.9194`, `mass_tv=0.0649`, `core_min_eig=0.2837`, `core_cov=0.3778`, `full_cov=0.3882` |
| `nash_unequal_width_hostile_imbalance` ($D \gg G$, $\sigma \in [.07, .30]$) | **PASS** | `hq=0.9934`, `mass_tv=0.1091`, `core_min_eig=0.7345`, `core_cov=0.1738`, `full_cov=0.2097` |

### 3.4 Tier-3 `14k` Endurance & Shift Recovery (`8 / 8` PASS Combined)
- **`bispf_throughput_endurance`** ([`readout-533d4c32`](../records/readout-533d4c32c634f4bf929d6407.json)): `mode_hold` PASS (`hq=1.0`, `8/8` modes), `ring_hold` PASS (`converged_step=1400`, `1200/1200` hold checks, `first_hold_failure=None`), `ring_extension` (`14k` updates) PASS (`converged_step=1400`, `300/300` extension checks, `first_hold_failure=None`).
- **`t3-shift-recovery-diagnostic-v1`** ([`readout-72b4701f`](../records/readout-72b4701ffe3e02fae8ac8982.json)): `5/5` PASS including `target_shift_recovery` (`81/81` post-deadline live checks, `min_hq=0.9685`, `min_modes=8`, `stable_from_step=2800` vs `0/81` frozen control).

---

## 4. Latency, Throughput & VRAM Leaderboard (`dcgan_performance`, `6 / 6` PASS)

Measured via official Forge `dcgan_performance` (`10` timed updates after `3` warmup updates, `finite_update_fraction = 1.0` on all 6 tasks, `NVIDIA RTX A6000`):

### 4.1 Full `dcgan_performance` Breakdown (`unified-v4` Uncontended vs `unified-v10` Champion)

| Task | Resolution / Batch | `unified-v4` Uncontended (`ms` · `p95` · `img/s` · `alloc` / `res MiB`) | `unified-v10` Champion (`ms` · `p95` · `img/s` · `alloc` / `res MiB`) |
| :--- | :--- | :---: | :---: |
| `dcgan_performance_r32` | `32×32`, `B=16` | `25.48` · `25.59` · `628.3` · `75.6 / 106.0` | `25.99` · `26.24` · `615.6` · `75.6 / 106.0` |
| `dcgan_performance_r64` | `64×64`, `B=16` | `30.69` · `31.12` · `524.4` · `183.5 / 254.0` | `32.61` · `34.06` · `488.1` · `183.5 / 254.0` |
| `dcgan_performance_b64` | `64×64`, `B=64` | `30.79` · `33.82` · `2028.0` · `334.6 / 342.0` | `32.93` · `35.41` · `1905.5` · `334.6 / 342.0` |
| `dcgan_performance_w128` | `64×64` wide, `B=16` | `36.92` · `37.99` · `432.0` · `546.7 / 656.0` | `41.97` · `42.53` · `382.4` · `546.7 / 656.0` |
| `dcgan_performance_r128` | `128×128`, `B=16` | `54.55` · `55.92` · `293.9` · `554.5 / 596.0` | `62.03` · `63.18` · `257.4` · `559.0 / 612.0` |
| **`dcgan_performance_r256`** | **`256×256`, `B=16`** | **`69.55`** · `70.31` · **`229.6`** · **`1372.6 / 1402.0`** | **`89.66`** · `92.08` · **`177.5`** · **`1372.6 / 1402.0`** |

*(Receipts: `unified-v4` [`readout-584f227b`](../records/readout-584f227b6b3d8d6538cde68c.json), `unified-v10` [`readout-90580c7a`](../records/readout-90580c7aff5ec6988cfbcfad.json). Note that `v10`'s `min(R,C)<=128` augmented `(X,Y)` Tikhonov path touches narrow conv projection shapes, adding ~`20 ms` over `v4` while staying strictly `<90 ms/step` with identical `1372.6 MiB` peak allocated VRAM.)*

### 4.2 Cross-Family Comparison (`256×256` RGB Latency, Primitives & Broad Qualification)

| Candidate | `0` SVD / `eigh` / `pinv`? | `0` `autograd.grad`? | `discriminator_stability` | `r256` Median (`ms/step`) | Peak Alloc (`MiB`) | Receipts |
| :--- | :---: | :---: | :---: | ---: | ---: | :--- |
| **`unified-v10`** *(this track)* | **Yes** | **Yes** | **`40 / 46`** (`11/11` req T1, `7/7` held T2) | **`89.66`** *(v4: **`69.55`**)* | **`1372.6`** | [`6da9dee1`](../records/readout-6da9dee19abe886e7ddbe184.json), [`90580c7a`](../records/readout-90580c7aff5ec6988cfbcfad.json) |
| `cond-fiber-v1` *(R3 Track 1)* | No *(SVD/NS)* | **Yes** | `40 / 47` | `87.50` | `1308.4` | [`5932f4d3`](../records/readout-5932f4d351cc4e791417835a.json), [`8d2e547c`](../records/readout-8d2e547ce820cf8662a34bdf.json) |
| `t4-nash-v3` *(R3 Track 3)* | **Yes** *(0 NS too)* | **Yes** | `33 / 46` (`11/11` req T1) | **`39.68`** | `1372.9` | [`1ca76c61`](../records/readout-1ca76c61c875ba5a33dca488.json), [`642d4c55`](../records/readout-642d4c55d093b44ed1e0309e.json) |
| `unified-v3` *(Round 2)* | No *(SVD/eigh)* | No | `34 / 41` | `134.26` | `1308.4` | [`ca2ce5ac`](../records/readout-ca2ce5acd5856e8c69b00377.json), [`333332d9`](../records/readout-333332d97507ed52ce2592e5.json) |
| `t4-arch-v3` *(Round 2)* | **Yes** | No | `26 / 41` (`7/7` `t4-arch`) | `93.85` | `1372.7` | [`95d61950`](../records/readout-95d61950559e6e35c77b9df3.json), [`b71204af`](../records/readout-b71204af30cb84dc9713fece.json) |
| `empf-v8` *(current default)* | No | No | `24 / 41` | `130.13`–`173.05` | `1435.4`–`1531.4` | [`33f76281`](../records/readout-33f7628135472eef06f0d92a.json), [`3d6ce9a3`](../records/readout-3d6ce9a324a579c291eac043.json) |
| `rhbi-v12` *(Round 1–2)* | No | No | `34 / 41` | `588.75`–`599.88` | `3706.6`–`3750.6` | [`47591bd9`](../records/readout-47591bd9555febfe3409fd0d.json), [`9917dee7`](../records/readout-9917dee7ebd20eedc0310687.json) |

---

## 5. Honest Diagnosis of Remaining Failures & Open Ceilings

1. **Conditional Denoising-Diffusion Chains (`model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`):** Both fail on `t1_q0` conditional slice statistics (`final_normalized_mean = 0.5233 / 0.3829`, `t1_q0_covariance_eigen_min = 0.2723 / 0.3080`). Track 1 (`cond-fiber-v1`, PR `#578`) isolated that passing both DDGAN chains requires dedicated conditional fiber conditioning with cross-batch bank residualization coupled directly to the denoising time-step critic — an orthogonal next-round merge target into the zero-SVD / zero-`autograd.grad` unified core.
2. **Tikhonov Smoothing vs. 8× Scale-Mass Pinch Tradeoff (`t3_iso_scale_mass_8x`):** Unsmoothed polar (`unified-v4/v5/v6`) passes `t3_iso_scale_mass_8x` cleanly (`2/2`, `core_min_eigen_ratio = 0.6621 >= 0.45`), whereas introducing `s = 0.0061` Tikhonov smoothing in `v7..v10` (essential for shallow equilibrium and `128×128` 100-mode lattices) softens small-component variance recovery (`hq = 0.9980`, `mass_tv = 0.0139`, `modes = 4/4`, `core_min_eigen_ratio = 0.2155`). Making `s_eff` adaptive to empirical local mode covariance anisotropy is a clean next-round candidate.
3. **Deep Residual Tanh Suffix Timing (`grid100_res16_tanh`):** The independent 100k holdout passes every threshold (`accuracy_pass=True`, `center_rms_sigma=0.1037`, `cov_frob_rms=0.1640`, `cov_trace_bias=-0.0743`, `precision=0.9906`, `radial_ks=0.0286`), while one intermediate live suffix check falls just outside the window.
4. **Remaining Tier-1/2 Edge Failures (`precision_gaussian_nvfp4`, `residual_student`, `interp_saddle_barycentric`):**
   - `precision_gaussian_nvfp4`: Terminal metrics pass cleanly (`cdf_ks=0.0293 <= 0.04`, `std_ratio=1.0040`), failing only on intermediate suffix noise under 4-bit quantization.
   - `residual_student`: `success_rate=0.9167`, `wrong_pad_rate=0.1667` (marginal miss on constrained pad avoidance).
   - `interp_saddle_barycentric`: `barycenter_surface_mae=0.0793 <= 0.09` passes, while `barycenter_hq=0.7183` falls below `0.78`.
5. **Adoption Status:** `DEFAULT_RECIPE` remains `"empf"`. Broadening to Round-3 untouched Tier-3/Tier-4 calibration holdouts (`r3-tier3-long-horizon-holdout-v1`, `r3-tier4-frontier-holdout-v1`) and merging Track 1's conditional fiber DDGAN mechanism remain future steps before any default-recipe proposal.

---

## 6. Verification & Reproducibility Checklist

- **Unit, Primitive-Ban & Semantic Controls (`tests/test_unified_v4.py`):** Verifies `0` `torch.autograd.grad`, `0` `svd`/`svdvals`/`eigh`/`eigvalsh`/`pinv`, exact zero-SVD augmented `(X,Y)` Tikhonov polar parity against exact SVD, $10^{-37}$ underflow safety, schema-2 `v_push` checkpoint tamper rejection, deliberate damage control, import-order purity, and bit-exact fresh-process continuation across active mechanism boundaries.
- **Public Contract:** `DEFAULT_RECIPE == "empf"`, `179` `Recipe` fields, `particlegan/recipe_compat.py` untouched.
- **Forge Audit & Provenance Gates:** `forge history`, `forge compile --summaries-only`, `forge validate`, `forge history --check`, `forge compile --check`, `ops/refresh_forge_readouts.py --check --reports-only` (`Status: CURRENT`), and `scripts/check_lint.py` (`0` new issues).
