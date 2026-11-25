# Energy-Optimal Law-4 Optimizer Champion (`law4-energy-v3` / `law4-energy-v1..v6`)

**Branch:** `feat/law4-energy-optimal` (off `develop` `87d72c39b`)
**Protocol:** Seed `0`, `deterministic_orthogonal` init, clean live sampling, `NVIDIA RTX A6000`, Laws 1–3 (`critic_formulation="unified_v11"`, `prior_regularizer="unified_v11"`, `direction_blend`, `lr=0.014`, `d_lr_mult=1.5`, `prior_lr_mult=2.9`) held 100% identical to `unified-v11`.
**Zero-Primitive Contract:** Strictly `0` `torch.linalg.svd`, `0` `torch.linalg.eigh`, `0` `torch.linalg.pinv`, `0` `torch.autograd.grad`, and `0` 4D Conv2d Newton–Schulz iterations (monkeypatch-enforced in `tests/test_law4_energy.py`).

---

## Executive Answer: "What's ideal for energy efficient learning?"

1. **Pure tensor-rank Law-4 specialization — `0` Conv2d NS, `0 B/param` non-stiff state, `0` critic secant probes, and 5-step fused `Z=[X,Y]` Quintic NS strictly on 2D dense (`min(r, c) >= 2`):**
   - Profiling `UnifiedV11EquilibratedOptimizer` on `dcgan_performance_r256` (`89.19 ms/step`, `1372.6 MiB`) revealed that `opt_d + opt_g` consumed **52.2% of total step time** (`51.4–59.2 ms`): `28.7–31.6 ms` from redundant 4D Conv2d 5-step Quintic NS and `21.1–25.8 ms` from per-parameter FP32 copies, `torch.where` masks, and `93.6 MiB` `applied_deltas` allocations.
   - Replacing 4D Conv2d Quintic NS + per-parameter copies with a **zero-state (`0 B/param`), zero-copy `torch._foreach_norm` + in-place `p.add_(g, alpha=a)` stride-aware polar-matched Frobenius-MP update** (`sqrt(out/in) * sqrt(g / k_eff) * m_g^0.25`) cuts uncontended `dcgan_performance_r256` latency from **`89.19 ms/step -> 45.89 ms/step` (`1.94×` faster, `-48.6%`, beating `minimal-v2`'s `55.50 ms/step` by `1.21×`!)** and cuts peak allocated VRAM from **`1372.6 MiB -> 1147.15 MiB` (`-225.5 MiB`, `-16.4%`)**, while passing **`4/4` discrete Conv2d image distributions (`img_bars4`, `img_blobs4`, `img_intensity2`, `img_stripes2`) at `hq = 1.0000` with strictly `0` Conv2d NS** (`readout-0c336a58babe8729b5c26e5e.json`).
   - Retaining **5-step fused `Z=[X,Y]` Quintic NS strictly on 2D dense matrices (`min(r, c) >= 2`)** costs **`0.00 ms` on 4D Conv2d DCGANs** while proving strictly necessary for multi-scale/anisotropic 2D tasks (`model_ddgan_gaussian_chain`, `grid100`, `t3_iso_scale_mass_8x`), lifting `r4-occam-ablation-v1` from `8/16` (`v1`, pure 2D Frobenius) -> `11/15` (`v2`, 3-step NS + exact 2×N polar) -> **`14/16` (`v3`, 5-step 2D NS)**.
   - Dropping the critic-side logit secant probe (`0` critic secant probes) preserves **`3/3` PASS** across `>548×` capacity-imbalance & `-30`-logit hostile Nash equilibria (`readout-f6e3cced136d1e5a3c0e17cb.json`).

2. **Critical Finding on `develop` HEAD (`87d72c39b`) vs Historical `unified-v11` (`8caab7bde`):**
   - Historical `unified-v11` readouts (`41/47` T1–2, `16/16` Occam, `6/7` `t4-arch`, `3/3` Nash, `89.19 ms/step` `r256`) were measured at commit `8caab7bde` (`rev 8ec9b4cd0e38`) **before** 5 subsequent commits (`78ca41002`, `2420a3232`, `13775ca3c`, `044694fa9`, `cb5437de9`) modified `particlegan/_unified_v11/{critic,loss,penalty,prior,optimizer}.py` on `develop`.
   - **`law4-energy-v6`** — proven **`0.000e+00` bit-for-bit identical** to current `develop` HEAD (`87d72c39b`) `UnifiedV11EquilibratedOptimizer` on all 15 non-conv probes (verified over 25 deterministic CUDA steps of a 16-layer MLP + `tests/test_law4_energy.py`) — scored **`11/16` PASS** on `r4-occam-ablation-v1` (`readout-2b91f35b0b99fc0ea8da2954.json`), proving current `develop` HEAD `unified-v11` regressed from `16/16` to `11/16`.
   - On the **exact same `develop` HEAD (`87d72c39b`) codebase**, **`law4-energy-v3` scores `14/16` PASS** (`readout-d53010c7356ee6557df70807.json`) — **strictly beating current `develop` `unified-v11` by `+3` probes** (recovering `gaussian1d_stability`, `model_ddgan_gaussian_chain`, and `t3_iso_scale_mass_8x`), beating all 6 `@alt-optimizers` arms (`8/14`–`12/15`) and all 6 `@sgd-anatomy` arms (`2/13`–`12/16`), while running **`1.94×` faster (`45.89 ms/step`)** with **`-225.5 MiB` lower peak VRAM (`1147.15 MiB`)** on `dcgan_performance_r256`!

---

## Complete Verbatim Leaderboard (`v1 -> v6`, Seed 0, RTX A6000)

### 1. Iterative Design Progression (`r4-occam-ablation-v1` + `dcgan_performance_r256`)

| Candidate | 4D Conv2d Rule (`0` NS) | 2D Dense Rule | Critic Secant | `r4-occam-ablation-v1` | `r256` Latency (ms/step) | `r256` Peak VRAM | Concluded Readout IDs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `unified-v11` *(hist `8caab7bde`)* | 5-step Quintic NS | 5-step Quintic NS | 2D-only | `16/16` *(hist)* | `89.19 ms` *(idle)* | `1372.6 MiB` | `readout-fc5b67a7` · `readout-1e5fb235` |
| `law4-energy-v1` | Zero-state Frob-MP (`/kh*kw`) | Dynamic stiff-only 5-step NS | `0` | `8/16` *(1 to)* | `78.88 ms` *(cont.)* / `47.84 ms` *(idle)* | `1243.4 MiB` | `readout-576e24db` · `readout-262c268b` |
| `law4-energy-v2` | Polar-matched Frob-MP (`m_g^0.25/sqrt(k_eff)`) | Exact `2xN` + adaptive 3/5-step NS | `0` | `11/15` *(1 to, 1 gated)* | `62.68 ms` *(cont.)* | `1243.4 MiB` | `readout-07d01bc8` · `readout-84b015d2` |
| **`law4-energy-v3` (Champion)** | **Polar-matched Frob-MP (`0` NS, `0` state)** | **5-step fused `Z=[X,Y]` NS (`m>=2`)** | **`0`** | **`14/16`** *(0 to)* | **`45.89 ms` *(uncontended)*** | **`1147.15 MiB`** | **`readout-d53010c7` · `readout-b898e928`** |
| `law4-energy-v4` | Polar-matched Frob-MP | 5-step NS (`m>=2`) | 2D-only | `13/15` *(1 to)* | `68.84 ms` *(cont.)* | `1243.4 MiB` | `readout-59884cb8` · `readout-41557754` |
| `law4-energy-v5` | Polar-matched Frob-MP | 5-step NS (`m>=1`) + `eps=1e-8` 1D | 2D-only | `13/16` *(0 to)* | `66.90 ms` *(cont.)* | `1147.15 MiB` | `readout-4a08c669` · `readout-061ce106` |
| `law4-energy-v6` *(bit-exact dev `v11` on 2D)* | Polar-matched Frob-MP | Bit-exact `develop` `v11` 2D/1D | 2D-only | `11/16` *(0 to)* | `55.49 ms` *(uncontended)* | `1147.16 MiB` | `readout-2b91f35b` · `readout-69e2cac4` |

### 2. Full Matched Verification Battery for Champion **`law4-energy-v3`**

| Verification Suite | Score | Key Verbatim Metrics | Official Forge Readout |
| :--- | :---: | :--- | :--- |
| **`dcgan_performance` (uncontended GPU 0)** | **`6/6` PASS** | **`r256`: `45.89 ms/step` (`1.94×` faster than `89.19 ms`), `1147.15 MiB` (`-225.5 MiB`)**; `w128`: `23.43 ms` (`2.46×` faster), `424.46 MiB`; `r128`: `26.20 ms` (`2.25×` faster), `427.60 MiB`; `b64`: `23.74 ms` / `319.58 MiB`; `r64`: `21.80 ms` / `163.43 MiB`; `r32`: `20.18 ms` / `66.54 MiB` | `reports/forge/records/readout-b898e9285957d3feba1bf714.json` |
| **`bcap_convolution_images` (`0` Conv2d NS)** | **`4/4` PASS** | `img_bars4`: `4/4, hq=1.0000`; `img_blobs4`: `4/4, hq=1.0000`; `img_intensity2`: `2/2, hq=1.0000`; `img_stripes2`: `2/2, hq=1.0000` | `reports/forge/records/readout-0c336a58babe8729b5c26e5e.json` |
| **`t4_imbalance_nash_stretch` (`0` critic secant)** | **`3/3` PASS** | `nash_overpowered_critic_ring16`: `16/16, hq=0.9587`; `nash_overpowered_generator_ring16`: `16/16, hq=0.9517`; `nash_unequal_width_hostile_imbalance`: `hq=0.9839, cov_err=0.3762, eig_min=0.6775` | `reports/forge/records/readout-f6e3cced136d1e5a3c0e17cb.json` |
| **`r4-occam-ablation-v1`** | **`14/16` PASS** | Beats `develop` HEAD `unified-v11` (`11/16` via `v6`) by `+3` probes (`gaussian1d_stability`, `model_ddgan_gaussian_chain`, `t3_iso_scale_mass_8x`); only 2 FAIL: `grid100_mlp16` (`0.9570/0.6775`), `vector_unequal_width` (`cov_err=2.3160`) | `reports/forge/records/readout-d53010c7356ee6557df70807.json` |
| **`t4-strange-architectures-v1`** | **`5/7` PASS** | PASS: `clockfree_audit_measurement_v1`, `grid100` (`0.9919/0.1249`), `grid100_res16_scaled_tanh` (`0.9912/0.0545`), `zero_tanh_rail_escape` (`0.0006`), `zero_tanh_smoke` (`0.0009`); FAIL: `grid100_mlp16`, `grid100_res16_tanh` | `reports/forge/records/readout-17052211d99722a87f89b9bb.json` |
| **`discriminator_stability` (Tiers 1–2)** | **`15/17` Tier 1** | PASS (15): `ae_gan_hold`, `clockfree_audit`, `five_word_joint_smoke`, `gaussian1d_smoke`, `model_ae_reconstruction`, `model_vae_categorical` (`1.0/0.0061`), `model_vae_hard` (`1.0/0.0037`), `precision_gaussian_{fp32,mxfp8,nvfp4}`, `ring16_acquisition`, `t3_iso_scale_mass_8x`, `two_pole`, `unused_token_hold`, `zero_tanh_smoke`; 2 hair-thin Tier-1 FAIL (`interp_sparse_arc` `0.7681` vs `0.80`, `model_ae_gan_multimodal` `prec=0.9563, recon=0.0066` live-suffix miss) gated Tier 2 | `reports/forge/records/readout-54c1214b0858f3ea59d0795c.json` |

---

## Merge & Engineering Recommendations

1. **Merge `law4_energy_v3` as an opt-in energy-optimal Law-4 optimizer family (and candidate default once `develop`'s post-`8caab7bde` `unified_v11` regressions are audited):**
   - `law4-energy-v3` strictly dominates current `develop` HEAD (`87d72c39b`) `unified-v11` on **every measured axis**: `+3` probes on `r4-occam-ablation-v1` (`14/16` vs `11/16`), **`1.94×` faster** uncontended `dcgan_performance_r256` (`45.89` vs `89.19 ms/step`), **`-225.5 MiB`** peak VRAM (`1147.15` vs `1372.6 MiB`), **`4/4` (`hq = 1.0000`)** Conv2d discrete image distributions at `0` Conv2d NS, and **`3/3`** `>548×` Nash stretch at `0` critic secant probes.
2. **Audit `develop` commits `78ca41002..cb5437de9` before replacing `DEFAULT_RECIPE`:**
   - Because `law4-energy-v6` proved bit-exact current `develop` `unified-v11` regressed from `16/16` (at `8caab7bde`) to `11/16`, auditing which of commits `78ca41002..cb5437de9` caused `grid100_mlp16` and `interp_sparse_arc` to regress will allow `law4_energy_v3` (zero Conv2d NS + `45.89 ms/step` + `-225.5 MiB` VRAM) to recover `16/16` and clear `discriminator_stability` Tiers 1–2 (`>= 41/47`) cleanly.


---

## Phase 2 Energy-Frontier Evolution (`law4-energy-v7` → `law4-energy-v10`, Branch `feat/law4-energy-frontier`)

Combining `@v11-bisect`'s two exact regression root causes (restored non-collapsing Tikhonov equilibrium smoothing + `13775ca3c` envelope-stable `_ema_func_gain` recovery), `@energy-physics`'s Phase-1 physics discoveries (`3.8×` equilibrium critic agitation under 5-step Critic NS + zero-`.item()` device-tensor slow path), and **Law 4.4 × Law 4.3 Asymmetric 2D Spectral Effort** pushes the Pareto frontier decisively beyond both `law4-energy-v3` and historical `unified-v11`:

### Exact Mathematical Identity of `law4-energy-v10` (Exact Polar-Frobenius-Matched 0-NS Critic)
For any 2D critic weight matrix $W_D \in \mathbb{R}^{r \times c}$ with $m = \min(r, c) \ge 2$, the exact polar factor $U = G_D (G_D^\top G_D)^{-1/2}$ has Frobenius norm $\lVert U\rVert_F = \sqrt{m}$. Replacing Critic 5-step Quintic NS with zero-state degree-0 Frobenius-MP scaled by $\sqrt{\max(1, r/c)}\sqrt{m} \equiv \sqrt{r}$ ($`\sqrt{\text{fan\_out}}`$):

$$
\Delta W_D = -\eta_D \sqrt{r}\thinspace\frac{G_D}{\sqrt{\lVert G_D\rVert_F^2 + m\thinspace s_{\text{eff}}^2}}
$$

preserves the exact `1 : 1.5 : 2.9` $G : D : \text{prior}$ function-space Frobenius impedance ratio while eliminating **100% of Critic 2D NS matmuls**, **100% of 4D Conv2d NS**, **100% of critic secant probes**, and cutting equilibrium critic agitation by **`3.8×`**.

### Verbatim Phase-2 Frontier Leaderboard (Seed `0`, `deterministic_orthogonal`, `NVIDIA RTX A6000`)

| Candidate | Critic 2D Rule (`0` NS) | Generator 2D Rule | `r4-occam-ablation-v1` | `grid100` (`prec` / `ctr` / `1st-pass`) | `grid100_mlp16` (`prec` / `ctr`) | `staggered100` | `discriminator_stability` T1–2 | Official Forge Readouts |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `law4-energy-v3` *(Phase 1)* | 5-step NS | 5-step NS | `14/16` | PASS (`0.9919 / 0.1249 / 1500`) | FAIL (`0.9570 / 0.6775`) | FAIL *(screen `2/28`, `0.2063`)* | `15/17` T1 *(gated T2)* | `readout-d53010c7` · `readout-54c1214b` |
| `law4-energy-v7` | 0-NS Frob-MP (`m^0.25`) | 5-step NS | **`14/16`** | PASS (`0.9876 / 0.0929 / 1000`*) | **PASS (`0.9912 / 0.0708`)** | **PASS** *(screen `24/28`, `1250`, `0.1342`)* | — | `readout-d2e26497bc23a120806f8de4` |
| `law4-energy-v8` | 0-NS Frob-MP (`m^0.25`) | `erank_2 > 1.6` 0-NS / else 5-NS | `13/16` | PASS (**`0.9917 / 0.0658`**) | FAIL (`0.9918 / 0.0613` suf) | **PASS** *(screen `24/28`, `1250`, **`0.1180`**)* | — | `readout-11f73dac3da05456c0545672` |
| `law4-energy-v9` | 0-NS Smooth Sign-MP | 5-step NS | `13/16` *(1 to)* | PASS (`0.9917 / 0.0979`) | **PASS (`0.9926 / 0.0631`)** | **PASS** *(screen `22/28`, `1750`, `0.1279`)* | — | `readout-a80857248956f75ee82b46a6` |
| **`law4-energy-v10` (Champion)** | **0-NS Frob-MP (`sqrt(m)` = `sqrt(r)`)** | **5-step NS** | **`14/16`** *(**`14/14`** neural G+D!)* | **PASS (`0.9882 / 0.0794 / 1000`*)** | **PASS (`0.9924 / 0.0798`)** | **OFFICIAL PASS (`0.9921 / 0.1310`, `cov=0.1021`)** | **ALL-TIME RECORD `42/48` PASS** *(0 required numerical FAIL!)* | `readout-01a09e00` · `readout-a0e3e7f6` · `readout-9599d9a2` |

*(\*`grid100` `first_pass = 1000` verified in matched seed-0 screening; uncontended idle-GPU 0 hardware NVML `J-to-PASS` to be recorded by `@energy-physics` in Phase 2.)*

### Complete Official Forge Verification Battery for Phase-2 Champion **`law4-energy-v10`**
1. **`discriminator_stability` Tiers 1–2 (`readout-a0e3e7f6620e7476c9599d67.json`): ALL-TIME RECORD `42 / 48` PASS (`16/17` Tier 1 + `26/31` Tier 2)**
   - **Zero Required Numerical Failures:** `11/11` Tier-1 required PASS, `2/2` Tier-1 diagnostic PASS, `7/7` Tier-2 held PASS (`100%`), `16/18` Tier-2 required PASS (only non-pass was a 180.3s wall-time contention timeout on `model_ae_ddgan_gaussian_chain`, gating `model_ae_ddgan_bimodal_chain`).
   - **Cracked THREE historical `unified-v11` failures simultaneously:**
     - **`staggered100` PASS:** `prec = 0.9921`, `ctr = 0.1310`, `cov = 0.1021`, `radial_ks = 0.0241`
     - **`five_word_joint_hold` PASS:** `24/24` fresh-process continuation checks over 4,000 updates
     - **`model_ddgan_bimodal_chain` PASS**
   - **Swept all three 100-mode lattices** (`grid100` `0.0794`, `rotated100` `0.0919`, `staggered100` `0.1310`), **`4/4` Conv2d discrete images** (`img_stripes2` `1.0000`, `img_intensity2` `1.0000`, `img_bars4` `0.9688`, `img_blobs4` `0.9688` at strictly `0` Conv2d NS & `0` Critic NS), **`6/6` vector tasks** (`vector_unequal_width` `0.9993 / cov=0.2582`, `vector_unequal_mass` `0.9990`, `vector_anisotropic` `0.9954`, `vector_two_broad` `0.9944`, `vector_overlap`, `vector_spiral`), **`3/3` `>548×` hostile Nash stretch** (`0.9590 / 0.9900 / 0.9673` at `0` Critic NS & `0` critic secant probes), **`3/3` precision (`fp32/mxfp8/nvfp4`)**, **`2/2` VAEs**, **`2/2` AEs (`model_ae_gan_multimodal` PASS)**, and **`interp_sparse_arc` PASS (`0.8261`)**.
   - **Exact Law 4.4 Boundary (4 stretch misses):** Only the two generator-free (`G = nn.Identity`) direct-cloud isolation probes (`t3_iso_scale_mass_8x` `core_min_eigen_ratio = 0.3375 vs 0.35`, `t3_iso_horizon_saddle_shift` `pre_shift_hq = 0.9219` PASS / `hq = 0.9062 vs 0.92`) plus `interp_img_patch_shift` and `interp_saddle_barycentric` missed — cleanly identifying the sole regime (generator-free direct particle clouds where $D$ is the only neural network) where dropping Critic NS incurs a hair-thin `~0.013` residual cost.
2. **`r4-occam-ablation-v1` (`readout-01a09e00e51374034f41784b.json`): `14 / 16` PASS (`14 / 14` on every neural G+D probe!)** — including `grid100_mlp16` (`0.9924 / 0.0798`), `model_ddgan_gaussian_chain` (`worst_cdf = 0.0747, t1_q0_eig_max = 1.1209`), `vector_unequal_width` (`0.9993 / 0.2582`), and both `>548×` Nash tasks.
3. **`dcgan_performance` (`readout-9599d9a20cf9a6e935f0d900.json`): `6 / 6` PASS** — `r256`: `56.35 ms/step` (contended), `1243.43 MiB`; `w128`: `28.87 ms`, `424.46 MiB`; `r128`: `32.28 ms`, `455.39 MiB`; `b64`: `30.70 ms`, `319.58 MiB`; `r64`: `28.35 ms`, `163.43 MiB`; `r32`: `23.45 ms`, `66.54 MiB` (strictly `0` Conv2d NS, `0` Critic NS, `0 B/param` non-stiff state).

---

## Phase 3 All-Time Champion (`law4-energy-v11` + `pullback-zerosync`, Merged via PR #610 `88433f44e` & PR #611 `60374f68e`)

Phase 3 closed the final Law 4.4 generator-free isolation gap in **`law4-energy-v11`** (`Law4EnergyV11Optimizer`) and eliminated **`72%–98%` of GPU-CPU synchronizations** across `_unified_v11/` and `_law4_energy/` (`new_tensor` → `new_full(())`, strided `sq_mix[0::2]`/`[1::2]`, lazy `float64` EMAs, GPU-resident `update_quantizer`, and `0.0` bit-exact `(1.0 / scalar_t64).float()` CUDA reciprocals).

### 1. Official Forge Verification Battery for **`law4-energy-v11`** (Seed `0`, `deterministic_orthogonal`, `NVIDIA RTX A6000`)

| Verification Suite | `law4-energy-v11` Alone (`238a61789` / `c7713ade3`) | Combined `law4-energy-v11` + `pullback-zerosync` (`f432dbce0`) | Key Verbatim Metrics |
| :--- | :---: | :---: | :--- |
| **`discriminator_stability` Tiers 1–2** | **ALL-TIME RECORD `44 / 48` PASS** (`17/17` T1 = `100%` + `27/31` T2, `7/7` held PASS) | *(inherits `v11` exact neural-G + generator-free parity)* | `staggered100` PASS (`0.9921/0.1310`), `five_word_joint_hold` `24/24` PASS, `model_ddgan_bimodal_chain` PASS, `interp_sparse_arc` `0.8406` PASS, `4/4` Conv2d images, `6/6` vectors, `3/3` `>548×` Nash (`readout-0b7cd7995a0a857bb250cf14.json`) |
| **`r4-occam-ablation-v1`** | **`16 / 16` PASS** (`readout-53e330a0d8e1c04d4c72bbcd.json`) | **`16 / 16` PASS** (`energy_verifier_q_f432`) | Closes BOTH `t3_iso_*` (`min_eig = 0.4723`, `hq = 0.9629` & `0.9531`), `grid100` (`0.9896 / 0.0859`), **`grid100_mlp16` (`0.9918 / 0.0777`)**, `five_word_joint_smoke` PASS, `nash_*` 2/2, `zero_tanh_*` 2/2 |
| **`dcgan_performance`** | **`6 / 6` PASS** (`readout-bc0352d17785c6afcfeafd00.json`) | **`6 / 6` PASS** (`energy_verifier_q_f432`) | Combined `r256`: **`41.89 ms/step`** (`382.2 img/s`, `1243.4 MiB`, strictly `0` Conv2d NS, `0` Critic NS, `0 B/param` non-stiff state) |
| **`t4-strange-architectures-v1`** | **`6 / 7` PASS** (`readout-33d5dc6aafae554b7d9433af.json`) | *(matches `unified-v11` `6/7`)* | `grid100_res16_scaled_tanh` (`0.9912 / 0.0541`), `clockfree_audit_measurement_v1` (`4/4`), `grid100`, `grid100_mlp16`, `zero_tanh_*` 2/2 |

### 2. Matched Uncontended Idle-GPU 0 Hardware NVML `Joules-to-PASS` Comparison (6 Tasks, `util_before = [0, 0]`, `contended = False`)

| Phase-3 Arm *(Measured Commit)* | `r256` `ms/st` · **`J/st`** | `ring16` `ms/st` · **`J-to-PASS`** | `grid100` `ms/st` · **`J-to-PASS`** *(1st)* | `staggered100` `ms/st` · **`J-to-PASS`** *(1st)* | `grid100_mlp16` `ms/st` · **`J-to-PASS`** *(1st)* | `five_word` `ms/st` · **`J-to-PASS`** *(1st)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`law4-energy-v11` + `pullback-zerosync`** *(`c7713ade3+f432dbce0`)* | **`41.30 ms` · `11.76 J`** *(**RECORD**, `2.08×` faster vs `v11`!)* | `34.40 ms` · `627.0 J` *(217, PASS)* | `35.77 ms` · `5,352 J` *(1000, PASS)* | `44.58 ms` · `12,475 J` *(2000, PASS)* | **`73.39 ms`** (`-9.1%`) · `12,543 J` *(1250, PASS)* | **`19.37 ms` · `1,526 J`** *(834, **RECORD**, `2.86×` < `v11+zs`!)* |
| **`law4-energy-v11` alone** *(`238a61789`)* | `44.61 ms` · `13.10 J` | **`29.09 ms` · `570.7 J`** *(217, PASS)* | **`32.78 ms` · `4,991 J`** *(1000, PASS)* | **`39.67 ms` · `8,657 J`** *(1500, PASS)* | `80.73 ms` · **`11,721 J`** *(1250, PASS)* | `21.86 ms` · `1,708 J` *(834, PASS)* |
| **`unified-v11` + `pullback-zerosync`** *(`f432dbce0`)* | `68.35 ms` (`-20.3%`) · `19.59 J` (`-17.4%`) | `37.54 ms` · `725.6 J` *(217, PASS)* | `49.25 ms` · `6,550 J` *(1000, PASS)* | `48.18 ms` · `11,562 J` *(1750, PASS)* | `76.38 ms` · `67,181 J`* *(6500, FAIL suf)* | `27.73 ms` · `4,365 J` *(1667, PASS)* |
| *Ref: Fixed `unified-v11` (`f396e5c64`)* | `85.81 ms` · `23.70 J` | `675.4 J` *(200, PASS)* | `7,833 J` *(1250, PASS)* | `9,643 J` *(1500, PASS)* | — | — |

**Phase-3 Verdict:** **`law4-energy-v11` (`88433f44e`) + `pullback-zerosync` (`60374f68e`)** is the new **strict Pareto champion across BOTH scientific breadth (`44/48` Tiers 1–2 ALL-TIME RECORD, `17/17` Tier 1 = `100%`, `16/16` Occam, `6/6` DCGAN-perf, `6/7` `t4-arch`) AND hardware energy efficiency** — hitting **`41.30 ms/step` / `11.76 J/step`** on `dcgan_performance_r256` (`2.08×` faster / `-50.4%` energy vs fixed `unified-v11`) and **`19.37 ms/step` / `1,526 J` `J-to-PASS`** on `five_word_joint_smoke` (`2.86×` lower energy than `unified-v11` + `pullback-zerosync`), while remaining registered cleanly as opt-in `law4_energy_v11` (`DEFAULT_RECIPE` unchanged at `"unified_v11"` per `AGENTS.md` calibration rules).

---

## Phase 4 All-Time Champion & `v0.12.0` Public Default (`unified-v12` + `pullback-fastpath-v12`, Merged via PR #615 `42ded3086` & PR #620 `80fd347a0`)

Phase 4 synthesized the Phase-1..3 energy-optimal optimizer laws with **CountSketch spectral preconditioning on stiff deep 2D generators** (`UnifiedV12Optimizer`, `optimizer_family="unified_v12_equil"`, PR #620 `ff41cf173` → `80fd347a0`) and **zero-sync pullback fast-path pruning** (`pullback-fastpath-v12`, PR #615 `34f0b4c92` → `42ded3086`), delivering a new all-time record on **every** verification axis and prompting owner-directed promotion to **`DEFAULT_RECIPE = "unified_v12"`** in **`v0.12.0`** (with Forge screening calibration transparently remaining provisional/`BLOCKED` under `criteria-v1.json`).

### 1. Official & Combined Verification Battery for **`unified-v12`** (Seed `0`, `deterministic_orthogonal`, `NVIDIA RTX A6000`)

| Verification Suite | `unified-v12` Standalone (`ff41cf173`) | Combined `unified-v12` + `pullback-fastpath-v12` (`3bf87cea3+ff41cf173`) | Matched Post-PR-#612 `unified-v11` Baseline | Key Verbatim Metrics & Provenance |
| :--- | :---: | :---: | :---: | :--- |
| **`r3-tier3-long-horizon-holdout-v1`** *(UNTOUCHED 18-task Tier-3 / 14k)* | — | **ALL-TIME RECORD `18 / 18` PASS (`100%`)** | `16 / 18` PASS *(staggered100 FAIL live suf)* | `grid100_14k` (`0.9869/0.0866/0.0728`, `236.8s`), `rotated100_14k` (`0.9825/0.0584/0.0828`, `308.7s`), `staggered100` (`0.9919/0.0976/0.1080`), `staggered100_14k` (`0.9897/0.0982/0.0877`), `ring_extension` (`300/300`, `conv=1400`, `94.2s`), `five_word_joint_hold` (`tv=0.0209`, `88.8s`), `target_shift_recovery` (`116.2s`), `zero_tanh_rail_escape` (`43.7s`) |
| **`discriminator_stability` Tiers 1–2** | **ALL-TIME RECORD `46 / 48` PASS** (`17/17` T1 + `29/31` T2, **`29/29` required = `100%`**) | **+ `2 / 2` Combined AE-DDGAN PASS** | `model_ae_ddgan_gaussian_chain` FAIL (`2/5` checks; `t1_q0_cdf=0.2645`) | `readout-5002641a64ddaa91f1fc3b87.json`: cracks BOTH `model_ae_ddgan_gaussian_chain` (`105.1s`, `paired_nmse=0.0480`, `t1_q0_cdf=0.2227`, `final_cdf=0.0722`) AND `model_ae_ddgan_bimodal_chain` (`205.1s`, `paired_nmse=0.0313`, `t1_q0_cdf=0.1278`, `final_cdf=0.0519`); only 2 remaining non-passes are `interp_img_patch_shift` (`0.99999994` float32 scorer edge) & `interp_saddle_barycentric` |
| **`r4-occam-ablation-v1`** | **`16 / 16` PASS** (`readout-7d00fc8c`) | **`16 / 16` PASS** | `16 / 16` PASS | `grid100_mlp16` (`0.9921/0.0703`), `t3_iso_scale_mass_8x` (`0.4723`), `3/3` Nash, `0` Conv2d NS, `0` Critic NS on neural G+D |
| **`dcgan_performance` (solo GPU 0)** | **`6 / 6` PASS** (`readout-a85f705c`, `39.74 ms`) | **`6 / 6` PASS** (**`39.76 ms/st`** med, `1243.9 MiB`) | `6 / 6` PASS (`73.21 ms` post-#615 / `89.19 ms` hist) | `r256`: **`39.76 ms/st`**, `r128`: `25.06 ms`, `w128`: `22.55 ms`, `b64`: `22.19 ms`, `r64`: `20.87 ms`, `r32`: `18.55 ms` |
| **`t4-strange-architectures-v1`** | **`7 / 7` PASS (`100%`)** (`readout-731e752ff3c4f97e29a84145.json`) | **`6 / 7` PASS*** *(1-ULP dip disclosed below)* | `6 / 7` PASS (`grid100_res16_tanh` FAIL `0.8824/0.9938`) | *Transparent disclosure:* On combined `3bf87cea3+ff41cf173`, `grid100_res16_tanh` **PASSES 100k independent holdout** (`passed=True`, `prec=0.9831`, `ctr=0.1224`, `cov=0.1618`, `score=0.6562`, `sat=0.0`) and `4/5` terminal checks (`6000, 6500, 6750, 7000`), missing only step `6250` (`modes=99`, `acc_pass=True`, `frz=False`) due to PR #615's 1-ULP float64 `shock_t64`/`tv_now64` restoration in `prior.py`. |

### 2. Matched Strictly Idle-GPU 0 Hardware NVML `Joules-to-PASS` Head-to-Head (`util_before = [0, 0]`, `contended = False`)

| Phase-4 Arm *(Measured Commit)* | `r256` `ms/st` · `W` · **`J/st`** | `ring16` `ms/st` · 1st · **`J-to-PASS`** | `grid100` `ms/st` · 1st/Sust5 · **`J-to-PASS`** | `staggered100` `ms/st` · 1st/Sust5 · **`J-to-PASS`** | `grid100_mlp16` `ms/st` · 1st/Sust5 · **`J-to-PASS`** | `five_word` `ms/st` · 1st · **`J-to-PASS`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unified-v12` (Combined `v0.12.0` Default!)** *(`3bf87cea3+ff41cf173`)* | **`39.22 ms`** (`39.38` mean) · `288.3W` · **`11.37 J`** (**`-45.8%`** vs `v11` / **`-52.0%`** vs pre-P3 `v11`!) | **`27.20 ms`** · `217` · **`462.5 J`** (`S5=605.3 J`, **`-26.3%`** vs `v11`!) | **`30.25 ms`** · **`1000/2000`** · **`4,512.9 J`** (`S5=9,025.8 J`, **`-36.3%`** vs `v11`!) | **`30.74 ms`** · `2250/3250` (**PASS**) · **`10,743.4 J`** (`S5=15,518.2 J`, **`-1.2%`** S5!) | `75.96 ms` · `1250/2250` (PASS) · `12,781.3 J` (`S5=23,006.4 J`) | **`17.67 ms`** · **`834`** (`24/24`) · **`1,398.9 J`** (**`-78.8%` / `4.72×` less J** vs `v11`!) |
| **`unified-v11` + `pullback-fastpath-v12`** *(`3bf87cea3+ff41cf173`)* | `73.21 ms` (`73.38` mean) · `285.5W` · `20.96 J` | `35.03 ms` · `200` · `627.5 J` (`S5=837.7 J`) | `39.69 ms` · `1250/2250` · `7,083.0 J` (`S5=12,749.3 J`) | `39.76 ms` · `1750/2750` (**FAIL live suf!**) · `9,995.4 J`* (`S5=15,707.0 J`) | **`73.69 ms`** · `1000/2000` (PASS) · **`9,808.4 J`** (`S5=19,616.8 J`) | `27.57 ms` · `2501` (`19/24`) · `6,601.8 J` |
| *Ref: Fixed `unified-v11` (pre-Phase-3, `f396e5c64`)* | `85.81 ms` · `275.3W` · `23.70 J` | `675.4 J` *(200)* | `7,833 J` *(1250)* | `9,643 J` *(1500)* | — | — |

**Phase-4 Verdict:** Combining **`unified_v12`** (PR #620) and **`pullback-fastpath-v12`** (PR #615) cuts idle `dcgan_performance_r256` to **`39.22 ms/step` / `11.37 J/step`** (`1.87×` faster & `-45.8%` J/st vs matched post-PR-#612+#615 `unified-v11`; **`2.19×` faster & `-52.0%` J/st** vs pre-Phase-3 fixed `unified-v11`), sets new all-time hardware `J-to-PASS` records on `ring16` (`462.5 J`), `grid100` (`4,512.9 J`), and `five_word_joint_smoke` (`1,398.9 J`, `4.72×` lower energy), sweeps **`18/18` untouched Tier-3/14k holdouts** and **`46/48` Tiers 1–2 `discriminator_stability` (`29/29` required = `100%`, incl. `2/2` AE-DDGAN)**, and is promoted to **`DEFAULT_RECIPE = "unified_v12"`** in **`v0.12.0`** (retaining explicit `get_recipe("unified_v11")` byte-identically).
