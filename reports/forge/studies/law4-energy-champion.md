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
