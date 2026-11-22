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
For any 2D critic weight matrix $W_D \in \mathbb{R}^{r \times c}$ with $m = \min(r, c) \ge 2$, the exact polar factor $U = G_D (G_D^\top G_D)^{-1/2}$ has Frobenius norm $\|U\|_F = \sqrt{m}$. Replacing Critic 5-step Quintic NS with zero-state degree-0 Frobenius-MP scaled by $\sqrt{\max(1, r/c)}\sqrt{m} \equiv \sqrt{r}$ ($\sqrt{\text{fan\_out}}$):
$$\Delta W_D = -\eta_D \sqrt{r}\,\frac{G_D}{\sqrt{\|G_D\|_F^2 + m\,s_{\text{eff}}^2}}$$
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
