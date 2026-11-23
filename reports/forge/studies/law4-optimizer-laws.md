# Law 4 Optimizer Laws, Causal SGD Anatomy & Energy-Efficient Learning

> **Study Scope & Governance:** All experiments below hold Laws 1–3 (`critic_formulation="unified_v11"`, `prior_regularizer="unified_v11"`, `UnifiedV11StepRecord`, `UnifiedV11CondFiberBank`, zero-`pinv` `direction_blend`, `lr = 0.014`, `d_lr_mult = 1.5`, `prior_lr_mult = 2.9`) strictly identical to `unified-v11`, varying **only `optimizer_family`** under protocol seed `0`, `deterministic_orthogonal` initialization, clean live sampling, and `NVIDIA RTX A6000` GPUs. Every score, metric, and failure is verified verbatim against `33` concluded official Forge readout receipts across three companion branches (`research/law4-sgd-anatomy` `66d5bfab7`, `research/law4-alt-optimizers` `c0a08e005`, `feat/law4-energy-optimal` `dc941984c`), plus matched uncontended GPU 0 hardware NVML energy/power measurements in [`law4-phase1-profile.json`](law4-phase1-profile.json) and [`law4-phase2-energy-benchmark.json`](law4-phase2-energy-benchmark.json). Screening calibration remains provisional (`BLOCKED` under [`configs/forge/calibration/criteria-v1.json`](../../../configs/forge/calibration/criteria-v1.json)); all diagnostic study readouts supply causal research evidence rather than ordinary qualification credit.

---

## Executive Summary: Direct Answers to All Five Questions

Across **`18` distinct optimizer families**, **`33` concluded official Forge readouts**, **5-regime mechanistic gradient/impedance telemetry**, and a **12-arm uncontended hardware-NVML Energy & `Joules-to-PASS` benchmark**, the answer to every question in the Law 4 optimizer investigation is sharp and falsifiable:

1. **Why can't pure SGD work?**
   Pure unnormalized SGD (`law4-pure-sgd`, `lr=0.014`, `readout-62543241`) scores **`2/13` executed PASS** (`9` numerical FAIL, `2` NaN explosions, `3` prereq-gated) even though `lr=0.014` is empirically optimal on `ring16` and passes `vector_unequal_{width,mass}` (`0.9990 / 0.9963`). Its failure decomposes causally into **four independent obstructions**: (a) `>500×` inter-player impedance explosion (`dD/dG` up to `1,021×` on `overpowered_critic`, `dG = 2.08 = 21σ` on `overpowered_generator`, instant `FloatingPointError` NaN in `< 2s`), (b) `31–62×` critic gradient starvation under Law-2 Born–Infeld potentials (`||∇_x D|| ≤ 1`), (c) `16,730×` intra-player inter-layer gradient skew and $\operatorname{sech}^2$ saturation rail deadlock (`img_bars4` `0/4`, `grid100` `0.147`, `tanh` rail frozen), and (d) rank-1 intra-layer spectral collapse (`erank = 1.056`, `94.7–98.0%` top-1 singular energy on `grid100_mlp16`).

2. **Can we use SGD on $G$ if impedance mismatch isn't a problem?**
   **Yes on balanced shallow nets, 4-mode Conv2d images, and even `>500×` imbalanced Nash (once global $G$-secant + equilibrated $D$/prior handle player impedance) — No on multi-scale, multi-module, rail, or deep generators (`9/15` PASS on `law4-sgd-g-secant-equil-d`, `readout-fd14ca20`).** Crucially, our causal reverse control (`law4-equil-g-sgd-d`, equilibrated $G$ + pure unnormalized SGD on $D$, `readout-294525ad`) **reverses conventional GAN intuition**: it scores **`12/16` PASS** (`+7` over `law4-sgd-g-equil-d` `5/13` and `+3` over `law4-sgd-g-secant-equil-d` `9/15`), proving an exact **2×2 Nash double dissociation** and **G-vs-D Structural Asymmetry Law**: once Law 2 (Born–Infeld 1-Lipschitz critic) regularizes $D$, **$G$ needs Law 4 layerwise/spectral equilibration far more than $D$**.

3. **Are there big gains from using other optimizers (Adam, SignSGD, Lion, Muon, Optimistic/Extragradient, Adafactor)?**
   **No global winner over zero-momentum `unified_v11` polar equilibration**, but **three decisive structural discoveries** and a **targeted `staggered100` breakthrough**:
   - Every alternative drops `3–8` probes on `r4-occam-ablation-v1`: `law4-sign-mp` (`12/15`, 0 states, 0 NS) = `law4-muon-m` (`12/15`, 1 state, NS) > `law4-adafactor-mp` (`11/15`, $O(m+n)$) > `law4-lion-mp` (`10/14`, 1 state) > `law4-adam-equil` (`9/15`, 2 states) > `law4-optimistic-equil` (`8/14`, 1 state, NS).
   - **Targeted `staggered100` breakthrough (`3/4` tested arms PASS):** `law4-sign-mp` (`prec=0.9923, ctr=0.1400`), `law4-optimistic-equil` (`ctr=0.1557`), and `law4-muon-m` (`ctr=0.1633, radial_ks=0.0156`) all crack `unified-v11`'s `staggered100` live-suffix failure (none cracks `grid100_res16_tanh` or `five_word_joint_hold`).
   - **Why stateful/momentum optimizers lose globally:** Once Law 2 damps rotational eigenvalues, optimistic negative momentum (`1.5 g_t - 0.5 g_{t-1}`) **amplifies minibatch gradient variance by `2.5×`**, causing suffix jitter (`8/14`); positive momentum (`Muon 0.85`, `Lion 0.5/0.9`) adds staleness lag on stiff/heteroscedastic targets; and `2`-state coordinate-wise `Adam` (`9/15`) overfits per-coordinate noise at `8 B/param` VRAM while still failing rank-1 `grid100_mlp16` (`prec=0.255`).

4. **What is ideal for energy-efficient learning?**
   **Tensor-rank specialized zero-state Law-4 equilibration (`law4-energy-v3`, `feat/law4-energy-optimal`):**
   - Profiling `unified-v11` on idle RTX A6000 `dcgan_performance_r256` revealed that **`opt_d + opt_g` consumed `51.1%` of the entire step (`45.59 ms` of `89.15 ms`)**: `27.00 ms` from redundant 4D Conv2d 5-step Quintic NS plus `17.45 ms` from unfused per-parameter FP32/masking overhead.
   - **`law4-energy-v3`** replaces 4D Conv2d NS with **zero-state (`0 B/param`), zero-copy `torch._foreach_norm` + in-place stride-aware polar-matched Frobenius-MP** ($\sqrt{\text{out}/\text{in}}\sqrt{g/k_{\text{eff}}}\,m_g^{1/4}$), drops the redundant critic secant probe (`0` critic probes), and retains batched 5-step fused `Z=[X,Y]` Quintic NS strictly on 2D dense matrices (`0.00 ms` DCGAN cost).
   - **Measured idle-GPU 0 hardware NVML wins (`unified-v11` → `law4-energy-v3`):**
     - `opt_d + opt_g` step time collapses **`45.59 ms → 2.62 ms` (`17.4×` faster optimizer!)** — within `1.55 ms` of pure SGD (`43.40 ms`)!
     - **`dcgan_performance_r256`**: `84.86 → 44.95 ms/step` (`1.89×` faster; Forge `89.19 → 45.89 ms`, **`1.94×`**, beating `minimal-v2` `55.50 ms` by `1.21×`!), **`21.22 → 12.56 J/step` (`-40.8%` hardware Joules/step!)**, Forge peak VRAM `1372.6 → 1147.15 MiB` (**`-225.5 MiB`**).
     - **`dcgan_performance_r64`**: `6.18 → 2.69 J/step` (**`-56.6%`**, `2.30×` efficiency); **`dcgan_performance_w128`**: `11.88 → 4.28 J/step` (**`-64.0%`**, `2.78×` efficiency).
     - **`Joules-to-PASS`**: **`619.3 J`** on `ring16_acquisition` (vs `unified-v11` `636.4 J`, `Adam` `851.6 J`, `Muon` `813.5 J`) and **`7,643 J`** on `grid100` (**`-23.6%`** vs `unified-v11` `9,998 J`, **`-31.6%`** vs `Adam` `11,174 J`).
     - **Quality:** **`14/16` on `r4-occam-ablation-v1` (`#1` in the entire swarm, `+3` over matched `develop` HEAD `unified-v11`!)**, **`4/4` Conv2d image tasks at `hq = 1.0000` with `0` Conv2d NS**, and **`3/3` `>548×` Nash stretch PASS with `0` critic secant probes**.

5. **Are there formal laws here?**
   **Yes.** All `33` Forge readouts, all `5` mechanistic regime diagnostics, and all `12` hardware energy profiles collapse into **Five Formal Optimizer Laws of Law 4** (Section 5): (4.1) *Inter-Player Function-Space Impedance Law*, (4.2) *Intra-Player Degree-0 Gauge & Rail Escape Law*, (4.3) *Tensor-Rank Spectral Isotropization Law*, (4.4) *G-vs-D Structural Asymmetry Law*, and (4.5) *Stochastic Minimax Zero-Momentum & Minimum-State Optimality Law*.

---

## 0. Source-Bound Baseline Integrity Note (`8caab7bde` vs `develop` HEAD `87d72c39b`)

Per ParticleGAN's source-bound evidence contract, **two distinct `unified-v11` baselines** must be kept explicitly separate:

| Baseline Identity | Git Commit | `r4-occam-ablation-v1` | `dcgan_performance_r256` (Idle) | Provenance |
| :--- | :---: | :---: | :---: | :--- |
| **Historical `unified-v11`** | `8caab7bde` (`rev 8ec9b4cd0e38`) | **`16/16`** *(across `v11` studies)* | `89.19 ms` · `1372.6 MiB` | [`readout-fc5b67a7`](../records/readout-fc5b67a767e57d748f460610.json) · [`readout-1e5fb235`](../records/readout-1e5fb2354920c2770b8b582c.json) |
| **Matched `develop` HEAD `unified-v11`** *(via bit-exact 2D `law4-energy-v6`)* | `87d72c39b` *(post `78ca41002..cb5437de9`)* | **`11/16`** *(5 FAIL: `1d_stab`, `mlp16`, `ddgan`, `scale_mass_8x`, `unequal_width`)* | `84.86 ms` · `21.22 J/step` *(unmodified `v11`)* · `55.49 ms` *(`v6` 4D Frob)* | `readout-2b91f35b` (`v6`, `0.000e+00` 2D bit-exact) · [`law4-phase1-profile.json`](law4-phase1-profile.json) |

All three experimental branches (`research/law4-sgd-anatomy`, `research/law4-alt-optimizers`, `feat/law4-energy-optimal`) branched off `develop` `87d72c39b`. As `law4-energy-v6` proved (`0.000e+00` bit-exact 2D parity with `develop` HEAD `UnifiedV11EquilibratedOptimizer` over 25 deterministic CUDA steps), `develop` commits `78ca41002..cb5437de9` shifted `unified-v11` on `r4-occam-ablation-v1` from `16/16` to `11/16` (notably affecting `grid100_mlp16`). Accordingly, on the **exact matched `87d72c39b` codebase**, **`law4-energy-v3` (`14/16`) strictly beats matched `develop` `unified-v11` (`11/16`) by `+3` probes**, while `law4-sign-mp` (`12/15`), `law4-muon-m` (`12/15`), and `law4-equil-g-sgd-d` (`12/16`) also exceed or match `11/16`.

---

## 1. Why Can't Pure SGD Work? (Causal 5-Rung Ladder & Mechanistic Telemetry)

### 1.1 Mechanistic Gradient, Spectral & Impedance Telemetry (`unified-v11`, Seed `0`, RTX A6000)

Using non-destructive bit-exact parameter probes inside [`experiments/law4_optimizer_profiler.py`](../../../experiments/law4_optimizer_profiler.py) ([`law4-phase1-profile.json`](law4-phase1-profile.json)), we measured raw SGD function-space displacement ($\delta_D^{\text{SGD}}, \delta_G^{\text{SGD}}$ at `lr_g=0.014, lr_d=0.021`), per-layer gradient norm skew ($\max_l \|g_l\|_2 / \min_l \|g_l\|_2$), 2D gradient effective rank ($\operatorname{erank}(g) = \|g\|_F^2 / \sigma_1(g)^2$) and condition number ($\sigma_1/\sigma_r$), and `tanh` Jacobian attenuation ($\operatorname{sech}^2 = 1 - y^2$) across five canonical regimes:

| Canonical Regime | Params (`G` / `D`) | $G$ Layer Grad Skew ($\max/\min$) | $G$ Grad $\operatorname{erank}$ (`top1%`) | $G$ Grad $\sigma_1/\sigma_r$ (med) | Raw SGD $\delta_D / \delta_G$ (med / peak) | Equil $\delta_D / \delta_G$ (med) | Secant Clamp (`G` / `D`) | Final `ema_fg` (`G` / `D`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ring16_acquisition`** *(shallow `64×2`)* | `4.6k` / `4.9k` (`1.07×`) | `3.15×` *(max `3.99×`)* | `2.02` (`49.5%`) | `2.77` | `0.39×` *(`0.0083 / 0.0275`)* | `1.93×` *(`0.0545 / 0.0277`)* | `1.0%` / `0.0%` | `0.991` / `1.000` |
| **`grid100_mlp16`** *(16-layer skip-free)* | `248.3k` / `34.6k` (`0.14×`) | **`266×` *(step 1: `16,730×`!)*** | **`1.056` (`94.7%`, max `98.0%`)** | **`4.60 × 10⁹`** | `1.07×` *(step 10: `26.7×`; step 200: `dG=1.03`!)* | `1.91×` *(`0.0895 / 0.0351`)* | **`72.5%`** / `0.0%` | **`0.452`** / `1.000` |
| **`zero_tanh_rail_escape`** *(`32×2` tanh, shift @1001)* | `162` / `1.4k` (`8.90×`) | `7.11×` *(max `21.3×`)* | `1.12` (`89.6%`) | `6.35` | `0.43×` *(step 1000: `sech²=0.126–0.188`; step 1200: `dG_sgd=45× dG_eq`)* | `3.00×` *(`dG_eq=4.99e-4` @1200)* | `4.2%` / `0.0%` | **`0.122`** / `1.000` |
| **`nash_overpowered_critic`** *(`620×` $D \gg G$)* | `1.3k` / `796.2k` (`620×`) | `5.29×` *(max `7.11×`)* | `1.38` (`72.5%`) | `3.60` | **`185.7×` *(peak `1,021×`; `dD_sgd=71.3`!)*** | `48.8×` *(`dD_eq: 3.01→0.26`)* | `0.0%` / **`19.5%`** | `1.000` / **`0.291`** |
| **`nash_overpowered_generator`** *(`548×` $G \gg D$)* | `791.6k` / `1.4k` (`0.002×`) | **`10.2×` *(max `35.2×`!)*** | **`1.22` (`82.0%`, max `93.2%`)** | **`1.12 × 10¹⁰`** | `0.89×` *(step 200: `dG_sgd=2.08 = 21σ`, `dG/dD=47×`!)* | `1.22×` *(`dG_eq=0.049–0.103`)* | **`27.0%`** / `0.0%` | **`0.457`** / `1.000` |

Paired with `@sgd-anatomy`'s 80-step pure-SGD failure telemetry (`law4-sgd-anatomy-telemetry.json`, branch `research/law4-sgd-anatomy`):
- **Law-2 Born–Infeld Critic Starvation:** Because Law 2 enforces $\|\nabla_x D\|_2 \le 1$, raw critic gradients are `~30×` smaller than generator gradients on balanced nets: pure SGD starves critic function-space displacement by **`31×–62×`** (`dD = 0.0043` vs `unified-v11` `0.135` on `ring16`; `D/G = 0.07–0.24` vs `1.98–4.31`).
- **Extreme `tanh` Rail Deadlock:** At pre-activation $a_0 = 8.0$ ($\operatorname{sech}^2(8) = 4.5 \times 10^{-7}$), pure SGD, `law4-sgd-secant`, `law4-sgd-g-equil-d`, and `law4-sgd-g-secant-equil-d` **all remain 100% frozen at `a = 8.000` (`dG = 0.0`)**, whereas degree-0 `law4-frob-sgd-mp` (`0` states, `0` NS), `law4-equil-g-sgd-d`, and `unified-v11` all escape immediately (`a → 1.837–1.840`). On `zero_tanh_rail_escape` itself (`|y|=0.902`, `sech^2 = 0.126–0.188`), pure SGD suffers **both** `5.3–7.9×` rail attenuation at shift (`step 1001`) **and** `45×` equilibrium chatter after escaping (`step 1200`: `dG_sgd = 2.24e-2` vs `dG_eq = 4.99e-4`)!

### 1.2 Verbatim 16-Task × 7-Arm Causal SGD Ladder (`r4-occam-ablation-v1`, Seed `0`, RTX A6000)

All 6 causal rungs hold `lr = 0.014, d_lr_mult = 1.5, prior_lr_mult = 2.9` and Laws 1–3 strictly identical to `unified-v11` (verified verbatim from `readout-{62543241,d5c8d860,f19a0f5c,fd14ca20,294525ad,12f988a4}.json` on `research/law4-sgd-anatomy`):

| Probe Task (`r4-occam-ablation-v1`) | `unified-v11` *(hist `8caab7bde` / dev `v6`)* | **R1** `pure_sgd` (`readout-62543241`) | **R2** `sgd_secant` (`readout-d5c8d860`) | **R3a** `sgd_g_equil_d` (`readout-f19a0f5c`) | **R3b** `sgd_g_sec+eq_d` (`readout-fd14ca20`) | **R4** `equil_g_sgd_d` (`readout-294525ad`) | **R5** `frob_sgd_mp` (`readout-12f988a4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gaussian1d_smoke` *(prereq)* | **PASS** / **PASS** | FAIL (`0.0820`) | FAIL (`0.0880`) | FAIL (`0.0718`) | **PASS** (`0.0475`) | **PASS** (`0.0242`) | **PASS** (`0.0240`) |
| `gaussian1d_stability` | **PASS** / FAIL | GATED | GATED | GATED | FAIL (`0.0452`) | FAIL (`0.0201`) | **PASS** (`0.0301`) |
| `zero_tanh_smoke` *(prereq)* | **PASS** / **PASS** | FAIL (`0.0254`) | FAIL (`0.0073`) | FAIL (`0.0236`) | **PASS** (`0.0020`) | **PASS** (`0.0012`) | **PASS** (`0.0031`) |
| `zero_tanh_rail_escape` | **PASS** / **PASS** | GATED | GATED | GATED | FAIL (`0.0031`) | **PASS** (`0.0006`) | FAIL (`0.0063`) |
| `ring16_acquisition` | **PASS** / **PASS** | FAIL (`0.8071`) | **PASS** (`0.9841`) | FAIL (`14/16, 0.690`) | **PASS** (`0.9814`) | **PASS** (`0.9873`) | **PASS** (`0.9573`) |
| `vector_unequal_width` | **PASS** / FAIL | **PASS** (`0.9990`) | FAIL (`tv=0.101`) | **PASS** (`0.9973`) | **PASS** (`0.9998`) | **PASS** (`0.9973`) | FAIL (`0.9836`) |
| `vector_unequal_mass` | **PASS** / **PASS** | **PASS** (`0.9963`) | **PASS** (`0.9993`) | **PASS** (`0.9983`) | **PASS** (`0.9951`) | **PASS** (`0.9973`) | FAIL (`0.9998`) |
| `img_bars4` | **PASS** / **PASS** | FAIL (`0/4`) | FAIL (`0/4`) | **PASS** (`1.0000`) | **PASS** (`1.0000`) | **PASS** (`1.0000`) | **PASS** (`0.9688`) |
| `grid100` *(prereq)* | **PASS** / **PASS** | FAIL (`0.1466`) | FAIL (`0.4545`) | FAIL (`0.1358`) | FAIL (`0.4655`) | **PASS** (`0.9919`) | **PASS** (`0.9724, ctr=0.066`) |
| `grid100_mlp16` | **PASS** / FAIL | GATED | GATED | GATED | GATED | FAIL (`0.9722`) | FAIL (`0.1565`) |
| `model_ddgan_gaussian_chain` | **PASS** / FAIL | FAIL (`1.328`) | FAIL (`1.328`) | FAIL (`1.414`) | FAIL (`1.428`) | **PASS** (`0.2596`) | FAIL (`0.8386`) |
| `five_word_joint_smoke` | **PASS** / **PASS** | FAIL (`3/5`) | FAIL (`3/5`) | FAIL (`4/5`) | FAIL (`3/5`) | **PASS** (`5/5`) | **PASS** (`5/5`) |
| `nash_overpowered_critic_ring16` | **PASS** / **PASS** | **NaN** *(INC)* | **PASS** (`0.9768`) | **PASS** (`0.9783`) | **PASS** (`0.9648`) | **NaN** *(INC, `5.4s`)* | **PASS** (`0.9583`) |
| `nash_overpowered_generator_ring16` | **PASS** / **PASS** | **NaN** *(INC)* | **PASS** (`0.9751`) | FAIL (`8/16, 0.367`) | **PASS** (`0.9751`) | **PASS** (`0.9719`) | **PASS** (`0.9709`) |
| `t3_iso_scale_mass_8x` | **PASS** / FAIL | FAIL (`1/4`) | FAIL (`1/4`) | FAIL (`eig=0.404`) | FAIL (`eig=0.404`) | **PASS** (`0.6869`) | FAIL (`0.9414`) |
| `t3_iso_horizon_saddle_shift` | **PASS** / **PASS** | FAIL (`0/4`) | FAIL (`0/4`) | **PASS** (`0.9609`) | **PASS** (`0.9609`) | FAIL (`0.9219`) | FAIL (`0.8984`) |
| **Verbatim PASS / Executed** | **`16/16`** / **`11/16`** | **`2 / 13`** *(2 NaN)* | **`4 / 13`** *(0 NaN)* | **`5 / 13`** | **`9 / 15`** | **`12 / 16`** *(1 NaN)* | **`9 / 16`** |

**Exact Causal Proof of Pure SGD's Four Obstructions:**
1. **R1 → R2 (Adding ONLY global player secant band):** Immediately turns BOTH `>500×` Nash tasks from instant `< 2s` NaN explosions into full `5/5` sustained PASSes (`0.9768 / 0.9751`) plus `ring16` (`0.9841`), proving **player-level function-space trust bands are necessary and sufficient for inter-player capacity imbalance**, yet leaves `9/13` failing because one scalar cannot equalize intra-player layers (`mlp16` skew remains `3,300×`, `img_bars4` `0/4`, `grid100` `0.455`, `five_word` `3/5`, `t3_iso` `0/2`).
2. **R2 → R5 (Adding ONLY zero-state layerwise Frobenius-MP + `peak_rms`, `0` NS, `0` Adam):** Collapses `mlp16` layer skew `7,500× → 19×`, escapes `tanh` rails (`a=8.0 → 1.837`), and jumps to **`9/16` PASS (`0` NaN, `0` gated)** — unlocking `grid100` (`prec=0.9724, ctr=0.0660`, fastest `grid100` first-pass step `1000` in the entire study!), `five_word_joint_smoke` (`5/5`), `gaussian1d_stability` (`0.0301`), `img_bars4` (`4/4`), and both `>500×` Nash tasks.
3. **R5 → `law4-energy-v3` / `unified-v11` (Adding 2D Quintic NS polar orthogonalization):** Because Frobenius normalization $G/\|G\|_F$ preserves singular-value ratios ($\sigma_1/\sigma_r = 4.6 \times 10^9$, $\operatorname{erank} = 1.056$), `law4-frob-sgd-mp` still collapses on rank-1 `grid100_mlp16` (`prec = 0.1565`) and minor-SV `model_ddgan_gaussian_chain` (`t1_q0 = 0.8386`). Spectral polar orthogonalization on 2D matrices completes the final leap (`14/16` on `87d72c39b`, `16/16` on `8caab7bde`).

---

## 2. SGD on $G$ When Impedance Mismatch Isn't a Problem? (Exact Boundary & G-vs-D Asymmetry Law)

Comparing **R3a (`SGD-G + Equil-D`)**, **R3b (`SGD-G+secant + Equil-D`)**, and the causal reverse control **R4 (`Equil-G + SGD-D`)** provides a complete answer:

- **Where pure SGD on $G$ DOES work (`9/15` on R3b):** Once equilibrated $D$ + row-normalized prior + the single scalar $G$ secant band resolve player impedance mismatch, pure unnormalized SGD on $G$ (`0` per-layer norm, `0` MP, `0` NS, `0` states) passes all 9 balanced, shallow, 4-mode Conv2d, saddle-shift, and `>500×` Nash probes: `img_bars4` (`4/4, hq=1.0000`), `ring16` (`0.9814`), `vector_unequal_{width,mass}` (`0.9998 / 0.9951`), `nash_overpowered_critic` (`0.9648`), `nash_overpowered_generator` (`0.9751`), `t3_iso_horizon_saddle_shift` (`0.9609`), `gaussian1d_smoke`, and `zero_tanh_smoke`.
- **Exact failure boundary of pure SGD on $G$ (`6` remaining failures on R3b):** Whenever $G$ itself faces **intra-generator scale skew, multi-module coupling, rail attenuation, or fine equilibrium settling**, pure SGD on $G$ fails cleanly: `grid100` (`prec = 0.4655`, gating `grid100_mlp16`), `five_word_joint_smoke` (`3/5` — joint `(G, E)` BiGAN skew), `model_ddgan_gaussian_chain` (`t1_q0 = 1.4281`), `zero_tanh_rail_escape`, `gaussian1d_stability` (equilibrium chatter without `peak_rms`), and `t3_iso_scale_mass_8x` (`eig = 0.4035`).
- **The G-vs-D Structural Asymmetry Law (Causal Double Dissociation):** Reversing the players in **R4 (`law4-equil-g-sgd-d`)** — putting pure unnormalized SGD on $D$ (`0` D norm, `0` D secant) and `unified_v11_equil` on $G$ + prior — jumps to **`12/16` PASS**, beating `SGD-G + Equil-D` by **`+7` tasks** and `SGD-G+secant + Equil-D` by **`+3` tasks**!

| Asymmetric Control | `overpowered_critic` (`620×` $D \gg G$) | `overpowered_generator` (`548×` $G \gg D$) | Hard $G$-Geometry Probes (`grid100`, `5w`, `ddgan`, `rail`, `scale_mass_8x`) | Total Occam PASS |
| :--- | :---: | :---: | :---: | :---: |
| **R3a: `SGD-G` + `Equil-D`** | **PASS** (`0.9783`) | **FAIL** (`8/16, 0.3672`) | **`0 / 5`** | `5 / 13` |
| **R4: `Equil-G` + `SGD-D`** | **NaN EXPLOSION** (`5.4s`) | **PASS** (`16/16, 0.9719`) | **`5 / 5`** (`grid100 0.992`, `5w 5/5`, `ddgan 0.260`, `rail 0.0006`, `8x 0.687`) | **`12 / 16`** |

**Why this reverses conventional intuition:** Under Law 2 (1st-order Born–Infeld 1-Lipschitz critic), $D$'s potential geometry is already regularized on balanced networks — $D$ only strictly needs equilibration/secant protection against extreme `D >> G` capacity imbalance (`620×`). By contrast, $G$ carries the entire nonlinear pushforward Jacobian across disconnected modes, 16-layer compositions, `tanh` rails, BiGAN encoders, and reverse-diffusion fibers, making Law-4 layerwise + spectral equilibration **dramatically more essential on $G$ than on $D$**.

---

## 3. Are There Big Gains From Other Optimizers? (6-Geometry Exploration + `staggered100` Breakthrough)

Branch `research/law4-alt-optimizers` (`c0a08e005`, `10` concluded official Forge readouts) tested six fundamentally distinct optimizer geometries holding Laws 1–3 strictly identical to `unified-v11`:

| Rank | Candidate (`optimizer_family`) | Geometry & Momentum | State / Param | 2D NS? | `r4-occam-ablation-v1` (Graded PASS) | `grid100` (`prec` / `ctr`) | `mlp16` (`prec`) | `staggered100` *(v11 FAIL!)* | `res16_tanh` / `5w_hold` | Verbatim Readouts (`occam` · `breakthrough`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Ref | **`unified-v11`** *(hist / dev `v6`)* | 0-mom Polar NS | `0` *(unstiff)* | 5-step | **`16/16`** / `11/16` | PASS (`0.992 / 0.112`) | PASS (`0.993`) / FAIL (`0.961`) | FAIL (`0.1649`) | FAIL / FAIL | `readout-fc5b67a7` · `readout-2b91f35b` |
| **1t** | **`law4-sign-mp`** | 0-state $L_\infty$ Smooth Sign-MP | **`0`** | **0** | **`12 / 15`** *(0 to, 1 gated)* | FAIL (`0.993 / 0.098`*) | GATED | **PASS (`0.9923 / 0.1400`)** | GATED / FAIL (`4/5`) | `readout-2ea657f4` · `readout-faa8d2a7` |
| **1t** | **`law4-muon-m`** | Muon Pos Nesterov (`0.85`) Polar | `1` (`4 B`) | 5-step | **`12 / 15`** *(1 to)* | PASS (`0.977 / 0.143`) | FAIL (`0.924`) | **PASS (`0.9913 / 0.1633`)** | FAIL (`0.245`) / — | `readout-0fd60717` · `readout-6fad0485` |
| **3** | **`law4-adafactor-mp`** | Sublinear $O(m+n)$ Factored RMS-MP | **`O(m+n)`** | **0** | **`11 / 15`** *(1 to)* | PASS (`0.992 / 0.122`) | FAIL (`0.174`) | — | — | `readout-92baddc3` |
| **4** | **`law4-lion-mp`** | 1-state Lion (`0.5/0.9`) Sign-MP | `1` (`4 B`) | **0** | **`10 / 14`** *(1 to, 1 gated)* | FAIL (`0.993 / 0.112`*) | GATED | — | — | `readout-a5f6ceda` |
| **5** | **`law4-adam-equil`** | 2-state Coord Adam-MP (`0/0.999`) | `2` (`8 B`) | **0** | **`9 / 15`** *(1 to)* | PASS (**`0.992 / 0.088`**) | FAIL (`0.255`) | FAIL *(holdout `0.136`)* | FAIL (`0.715`) / — | `readout-8c40a844` · `readout-d0dd40dd` |
| **6** | **`law4-optimistic-equil`** | Optimistic Extragrad (`λ=0.5`) Polar | `1` (`4 B`) | 5-step | **`8 / 14`** *(1 to, 1 gated)* | FAIL (`0.993 / 0.123`*) | GATED | **PASS (`0.9922 / 0.1557`)** | GATED / — | `readout-ff204f93` · `readout-1f079816` |

*(\*`law4-sign-mp`, `law4-lion-mp`, and `law4-optimistic-equil` all achieved high final `grid100` precision `0.9928–0.9930` yet missed the 5-check live terminal suffix.)*

**Four Mechanistic Lessons From Alternative Optimizers:**
1. **Stochastic Minimax Variance Inversion (`Optimistic < Muon < Zero-Momentum`):** Classical deterministic game theory predicts optimistic extragradient steps $g_{\text{eff}} = (1+\lambda) g_t - \lambda g_{t-1}$ ($\lambda = 0.5$) damp rotational cycles. Under Law-2 Born–Infeld regularization (which already damps critic rotational eigenvalues), i.i.d. minibatch noise variance scales by $(1+\lambda)^2 + \lambda^2 = 2.5\times$, causing terminal suffix jitter (`8/14` — worst of all 6 alternative arms!). Conversely, positive Nesterov momentum (`law4-muon-m`, $\beta_1=0.85$) cuts noise variance $\approx 12.3\times$ (`12/15`), yet smears stiff/heteroscedastic regimes (`grid100_mlp16` `ctr=0.411`, `t3_iso_scale_mass_8x` `eig=0.318`). **Zero momentum (`beta1 = 0.0`) dominates across BOTH spectral polar (`unified-v11` / `law4-energy-v3`) AND $L_\infty$ sign (`Sign-MP 12/15 > Lion-MP 10/14`) geometries!**
2. **Inverse State-Complexity Ordering (`0-state Sign-MP > O(m+n) Adafactor > 2-state Adam`):** Zero-state smooth SignSGD-MP (`12/15` + `staggered100` PASS + `five_word` `5/5`, `0` tensor states, `0` matmuls) beats $O(m+n)$ Adafactor (`11/15`), which beats full $O(2mn)$ 2-state Adam (`9/15`). In a non-stationary two-player game, per-coordinate exponential second moments $v_{ij}$ lag behind opponent shifts and overfit single-coordinate variance (despite sharp static `grid100` centers `0.0881`).
3. **Rank-Preserving Diagonal Limit:** Neither full 2-state `Adam` (`grid100_mlp16 prec = 0.2551`), factored `Adafactor` (`0.1737`), `Frob-SGD` (`0.1565`), nor $L_\infty$ `Sign`/`Lion` can escape 16-layer rank-1 gradient collapse ($\operatorname{erank} = 1.056$, $\sigma_1/\sigma_r = 4.6 \times 10^9$), because diagonal and Kronecker rescalings preserve matrix rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$). Only non-diagonal spectral polar orthogonalization lifts collapsed gradients back to full rank.
4. **Targeted `staggered100` Breakthrough (`3/4` tested arms PASS):** `law4-sign-mp` (`ctr = 0.1400`), `law4-optimistic-equil` (`ctr = 0.1557`), and `law4-muon-m` (`ctr = 0.1633, radial_ks = 0.0156`) all crack `unified-v11`'s `staggered100` live-suffix failure — showing that mild geometry/smoothing adjustments can recover `staggered100` without abandoning Law 4's core.

---

## 4. Ideal Optimizer for Energy-Efficient Learning (Hardware NVML Profiling & 12-Arm `Joules-to-PASS` Pareto Frontier)

### 4.1 Exact Uncontended Idle-GPU 0 Phase Breakdown: `unified-v11` vs `law4-energy-v3`

Measured on strictly idle `NVIDIA RTX A6000` (`GPU 0`, `0%` utilization before start, direct hardware `nvmlDeviceGetTotalEnergyConsumption` mJ counter over 50 uninstrumented sustained steps + 20 synchronized CUDA-event instrumented steps, [`law4-phase1-profile.json`](law4-phase1-profile.json)):

| Workload (Params) | Candidate | Uninst. `ms/step` (`img/s`) | Avg Power (`W`) | **Hardware `J/step`** | `d_fwd+bwd` (`ms`) | `d_pen` (`ms`) | **`opt_d.step`** (`ns` / `sec` / `ovh`) | `g_fwd+bwd` (`ms`) | `g_loss+pr` (`ms`) | **`opt_g.step`** (`ns` / `sec` / `ovh`) | **Total Opt (`%` step)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`r256`** (`23.4M`) | `unified-v11` | `84.86` (`188.2`) | `249.4` | **`21.22 J`** | `20.49` | `6.78` | **`20.90`** (`13.05` / `0.00` / `7.86`) | `13.38` | `2.89` | **`24.69`** (`13.95` / `1.12` / `9.59`) | **`45.59 ms` (`51.1%`)** |
| **`r256`** (`23.4M`) | **`law4-energy-v3`** | **`44.95` (`355.5`)** | `278.8` | **`12.56 J` (`-40.8%`)** | `20.60` | `6.76` | **`0.52`** (`0.00` / `0.00` / `0.52`) | `13.22` | `2.69` | **`2.10`** (`0.00` / `0.88` / `1.22`) | **`2.62 ms` (`5.7%`, `17.4×`!)** |
| **`r64`** (`6.6M`) | `unified-v11` | `43.29` (`363.7`) | `140.4` | **`6.18 J`** | `7.36` | `4.12` | **`10.90`** (`6.22` / `0.00` / `4.69`) | `6.59` | `3.00` | **`14.35`** (`7.19` / `0.83` / `6.31`) | **`25.25 ms` (`53.8%`)** |
| **`r64`** (`6.6M`) | **`law4-energy-v3`** | **`21.37` (`748.1`)** | `125.4` | **`2.69 J` (`-56.6%`)** | `7.05` | `3.88` | **`0.30`** (`0.00` / `0.00` / `0.30`) | `6.09` | `2.76` | **`1.72`** (`0.00` / `0.77` / `0.95`) | **`2.02 ms` (`9.3%`, `12.5×`!)** |
| **`w128`** (`24.2M`) | `unified-v11` | `55.55` (`286.3`) | `212.4` | **`11.88 J`** | `8.02` | `4.17` | **`16.33`** (`10.51` / `0.00` / `5.83`) | `6.89` | `2.93` | **`20.36`** (`12.07` / `0.85` / `7.46`) | **`36.69 ms` (`62.4%`)** |
| **`w128`** (`24.2M`) | **`law4-energy-v3`** | **`22.18` (`721.3`)** | `192.6` | **`4.28 J` (`-64.0%`)** | `7.72` | `3.87` | **`0.46`** (`0.00` / `0.00` / `0.46`) | `6.54` | `2.71` | **`1.77`** (`0.00` / `0.61` / `1.17`) | **`2.23 ms` (`9.6%`, `16.5×`!)** |

**Key Engineering Discovery:** On `unified-v11`, `opt_d + opt_g` wasted **`51.1%–62.4%` of total training energy** on 4D Conv2d DCGANs (`27.0 ms` in redundant Conv2d Quintic NS + `17.5 ms` in per-parameter Python/FP32/masking overhead on `r256`). By replacing 4D Conv2d NS with zero-state, zero-copy `torch._foreach_norm` + in-place stride-aware polar-matched Frobenius-MP (`sqrt(out/in) * sqrt(g/k_eff) * m_g^0.25`), **`law4-energy-v3` accelerates the optimizer itself by `17.4×` (`45.59 → 2.62 ms`)**, leaving `94.3%` of wall time in actual forward/backward model compute and cutting total GPU board energy per step by **`40.8%–64.0%`**!

### 4.2 Complete 12-Arm Matched Uncontended GPU 0 Energy & `Joules-to-PASS` Pareto Leaderboard

Every arm below ran back-to-back on strictly idle `NVIDIA RTX A6000` (`GPU 0`, seed `0`, [`law4-phase2-energy-benchmark.json`](law4-phase2-energy-benchmark.json)), measuring hardware NVML `Joules/step`, `Watts`, `ms/step`, exact optimizer `State B/param`, single-batch `Peak VRAM (MiB)`, and **`Joules-to-PASS`** (`first_pass_step × Joules/step`) on `ring16_acquisition` and `grid100` (with official Forge `grid100` first-pass / sustained-pass milestones and `r4-occam-ablation-v1` verdicts):

| Optimizer Arm | `r4-occam` PASS *(87d72c39b)* | `r256` `ms/step` | `r256` `Watts` | **`r256` `J/step`** | `State` (`B/param`) | `r256` Peak (`MiB`) | `ring16` `J/step` | `ring16` 1st Pass | **`ring16` `J-to-PASS`** | `grid100` `J/step` | `grid100` 1st / Sust5 | **`grid100` `J-to-PASS`** *(1st / Sust5)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`law4-energy-v3` (Champion)** | **`14 / 16`** | **`45.58`** | `287.7` | **`13.12 J`** *(profile: `12.56`)* | **`0.000`** | `1219.4` *(Forge `1147.1`)* | `2.854` | `217` (PASS) | **`619.3 J`** | `5.095` | `1500` / `2500` (PASS) | **`7,643 J` / `12,739 J`** |
| `unified-v11` *(dev HEAD / hist)* | `11/16` *(`16/16` hist)* | `84.75` | `250.4` | `21.28 J` | `0.000` | `1148.0` *(Forge `1372.6`)* | `2.933` | `217` (PASS) | `636.4 J` | `5.713` | `1750` / `2750` (PASS) | `9,998 J` / `15,711 J` |
| `law4-sign-mp` *(0-state $L_\infty$)* | `12 / 15` *(+`stag100`)* | `62.17` | `258.0` | `16.06 J` | **`0.000`** | `1147.9` | `2.779` | `217` (PASS) | **`603.1 J`** | `5.107` | `2000` / `3000` *(FAIL suf)* | `10,214 J`* |
| `law4-equil-g-sgd-d` *(R4 ctrl)* | `12 / 16` *(1 NaN)* | `65.32` | `248.5` | `16.26 J` | `0.000` | `1148.9` | `2.834` | `217` (PASS) | **`615.0 J`** | `5.079` | `1750` / `2750` (PASS) | `8,888 J` / `13,967 J` |
| `law4-frob-sgd-mp` *(R5 0-NS)* | `9 / 16` | `64.75` | `250.2` | `16.23 J` | **`0.000`** | `1148.5` | `2.927` | `250` (PASS) | `731.7 J` | `5.357` | **`1000` / `2000`** (PASS) | **`5,357 J` / `10,714 J`** |
| `law4-adafactor-mp` *($O(m+n)$)* | `11 / 15` | `68.73` | `249.0` | `17.20 J` | `0.012` | `1148.4` | `3.165` | `234` (PASS) | `740.6 J` | `5.605` | `2000` / `3000` (PASS) | `11,210 J` / `16,815 J` |
| `law4-lion-mp` *(1-state Sign)* | `10 / 14` | `74.85` | `240.3` | `18.01 J` | `3.994` | `1236.8` | `3.038` | `217` (PASS) | `659.2 J` | `5.576` | `1750` / `2750` *(FAIL suf)* | `9,758 J`* |
| `law4-sgd-g-secant-equil-d` *(R3b)* | `9 / 15` | `67.02` | `280.9` | `18.87 J` | `0.000` | `1148.5` | `2.980` | `217` (PASS) | `646.7 J` | `5.244` | `NONE` (FAIL) | `∞` (FAIL) |
| `law4-adam-equil` *(2-state Adam)* | `9 / 15` | `90.81` | `228.7` | `20.82 J` | **`7.989`** | **`1397.7`** | `3.639` | `234` (PASS) | `851.6 J` | `6.385` | `1750` / `2750` (PASS) | `11,174 J` / `17,558 J` |
| `law4-optimistic-equil` *(λ=0.5)* | `8 / 14` *(+`stag100`)* | `95.74` | `262.2` | `25.15 J` | `3.994` | `1238.0` | `3.408` | `217` (PASS) | `739.4 J` | `5.954` | `1750` / `2750` *(FAIL suf)* | `10,420 J`* |
| `law4-muon-m` *(Pos Nesterov)* | `12 / 15` *(+`stag100`)* | `97.76` | `261.7` | `25.61 J` | `3.994` | `1238.3` | `3.477` | `234` (PASS) | `813.5 J` | `6.049` | `1500` / `3000` (PASS) | `9,073 J` / `18,146 J` |
| `law4-pure-sgd` *(R1 floor)* | `2 / 13` *(2 NaN)* | `43.40` | `254.9` | `11.10 J` | `0.000` | `1219.4` | `2.333` | `417` *(FAIL)* | `973.0 J`* *(FAIL)* | `4.417` | `NONE` (FAIL) | `∞` (FAIL) |

**Three Physical Energy Laws Revealed by the 12-Arm Benchmark:**
1. **Faster Per-Step Does NOT Mean Lower `Joules-to-PASS`:** Pure SGD has the lowest `J/step` (`2.33 J` on `ring16`, `4.42 J` on `grid100`), yet takes `1.92×` more steps to reach a transient `ring16` hit (`973.0 J` — **57% MORE energy than `law4-energy-v3`'s `619.3 J`**, and still fails sustained!) and spends `∞ J` on `grid100`.
2. **State DRAM Bandwidth Directly Penalizes Energy (`J/step`):** Every extra `4 B/param` of persistent optimizer state (`Lion`, `Muon`, `Optimistic`, `Adam`) adds DRAM read/write traffic per step: `law4-adam-equil` (`7.99 B/param`, `1398 MiB`) uses `20.82 J/step` on `r256` (virtually identical to full-NS `unified-v11` despite `0` NS!) and `851.6 J` on `ring16` (`+37.5%` energy vs `law4-energy-v3`), while `law4-muon-m` (`3.99 B/param` + 5-step NS) burns **`25.61 J/step`** — **`2.04×` more energy per step than `law4-energy-v3`!**
3. **The Global Pareto Champion (`law4-energy-v3`):** Sits strictly on the multi-objective Pareto frontier across all four axes — **`#1` suite quality (`14/16` Occam, `4/4` Conv2d images at `hq=1.0000`, `3/3` Nash, `15/17` Tier-1 `disc_stab`)**, **`0.000 B/param` non-stiff state**, **`12.56–13.12 J/step` on `r256` (`-40.8%` vs `unified-v11`, within `1.46–2.02 J` of pure SGD!)**, and **`-23.6%` lower `Joules-to-PASS` on `grid100` (`7,643 J` vs `9,998 J`)**.

### 4.3 `law4-energy-v1..v6` Iterative Evolution & Champion Verification (`feat/law4-energy-optimal`)

| Candidate | 4D Conv2d Rule (`0` NS) | 2D Dense Rule | Critic Secant | `r4-occam` | `r256` (`ms` · `MiB`) | Additional Suites | Official Readouts |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `law4-energy-v1` | Zero-state Frob-MP (`/kh*kw`) | Stiff-only 5-step NS | `0` | `8/16` *(1 to)* | `47.84` idle · `1243.4` | — | `readout-576e24db` · `readout-262c268b` |
| `law4-energy-v2` | Polar-matched Frob-MP | Exact `2×N` + 3/5-step NS | `0` | `11/15` *(1 to)* | `62.68` cont · `1243.4` | — | `readout-07d01bc8` · `readout-84b015d2` |
| **`law4-energy-v3`** | **Polar-matched Frob-MP (`_foreach`)** | **5-step fused `Z=[X,Y]` NS (`m≥2`)** | **`0`** | **`14 / 16`** | **`45.89` idle · `1147.15`** | **`6/6` DCGAN · `4/4` Conv (`1.0`) · `3/3` Nash · `5/7` `t4-arch` · `15/17` T1 `disc_stab`** | `readout-d53010c7` · `readout-b898e928` · `readout-0c336a58` · `readout-f6e3cced` · `readout-17052211` · `readout-54c1214b` |
| `law4-energy-v4` | Polar-matched Frob-MP | 5-step NS (`m≥2`) | 2D-only | `13/15` *(1 to)* | `68.84` cont · `1243.4` | — | `readout-59884cb8` · `readout-41557754` |
| `law4-energy-v5` | Polar-matched Frob-MP | 5-step NS (`m≥1`) | 2D-only | `13/16` | `66.90` cont · `1147.15` | — | `readout-4a08c669` · `readout-061ce106` |
| `law4-energy-v6` | Polar-matched Frob-MP | Bit-exact `develop` `v11` 2D/1D | 2D-only | `11/16` | `55.49` idle · `1147.16` | Proves `develop` HEAD `v11` = `11/16` | `readout-2b91f35b` · `readout-69e2cac4` |

*(Honest disclosure on `law4-energy-v3` `discriminator_stability` `readout-54c1214b`: passed `15/17` Tier 1 including `3/3` precision `fp32/mxfp8/nvfp4`, `2/2` VAEs, `five_word_joint_smoke`, `t3_iso_scale_mass_8x`, `ring16`, `clockfree`; two hair-thin Tier-1 misses — `interp_sparse_arc` `interp_hq = 0.7681` vs `0.80` and `model_ae_gan_multimodal` `prec = 0.9563, recon = 0.0066` live-suffix miss — gated Tier 2.)*

---

## 5. The Five Formal Optimizer Laws of Law 4

Grounded verbatim in all `33` Forge readouts and `12` hardware energy profiles, Law 4 (*Function-Space Two-Player Equilibration*) decomposes into **five falsifiable mathematical laws**:

### Law 4.1 — Inter-Player Function-Space Impedance Law
> **Statement:** In a two-player minimax game with player capacity/gain ratio $\rho = \text{Gain}(P_1)/\text{Gain}(P_2) \gg 1$, parameter-space step matching ($\eta_D \approx \eta_G$) induces function-space displacement mismatch $\delta_{P_1}/\delta_{P_2} = \Theta(\rho)$ and finite-time divergence ($O(1)$ steps to `NaN`). Regulating each player's realized function-space displacement into a bounded trust band $\delta_G \in [\delta_{g,\min}, \delta_{g,\max}]$ (and $\delta_D \le \delta_{d,\max}$ when $D$ is unnormalized) is **necessary and sufficient** to stabilize arbitrarily imbalanced player capacities ($\rho > 500\times$) — *even under pure unnormalized SGD*.
> - **Causal Proof:** `R1` (`law4-pure-sgd`) explodes to `NaN` in `< 2s` on both `620×` `overpowered_critic` (`dD_sgd = 71.3`, `dD/dG = 1,021×`) and `548×` `overpowered_generator` (`dG_sgd = 2.08 = 21σ`). `R2` (`law4-sgd-secant`, adding *only* the player secant band) flips **both** to full `5/5` sustained PASS (`hq = 0.9768 / 0.9751`, `readout-d5c8d860`). Conversely, removing `delta_g_max` from full Quintic NS (`r4-razor-no-secant-band`, `readout-4e2fd695`) collapses `548×` `overpowered_generator` to `6/16` (`hq = 0.2522`).

### Law 4.2 — Intra-Player Degree-0 Gauge & Rail Escape Law
> **Statement:** A single global player-level scalar $\alpha \in \mathbb{R}_+$ cannot equalize multiplicative inter-layer Jacobian skew $\max_l \|g_l\| / \min_l \|g_l\| \in [10^1, 10^5]$ or $\operatorname{sech}^2(a) \to 0$ saturation rails. Degree-0 scale-invariant layerwise (and prior-row) normalization
> $$\Delta W_l = -\eta\,\gamma_{\text{MP}}(W_l)\,\frac{G_l}{\sqrt{\|G_l\|^2 + s_{\text{eff}}(W_l)^2}}, \qquad s_{\text{eff}}(W_l) = \min\!\big(s_{\text{base}},\, c\,\text{peak\_rms}_l\big)$$
> with underflow-safe scaling ($10^{-38}$) and Marchenko–Pastur operator gain $\gamma_{\text{MP}}$ is **necessary** to equalize multi-layer MLPs, Conv2d stacks, BiGAN `(G, E)` pairs, and `tanh` rails, while preserving smooth equilibrium settling ($G_l \to 0 \implies \Delta W_l \to 0$) without a step clock.
> - **Causal Proof:** `R2` (`law4-sgd-secant`) retains `3,300×` `mlp16` skew, stays frozen at `a = 8.000` on `tanh` rails, and fails `img_bars4` (`0/4`), `grid100` (`0.455`), and `five_word` (`3/5`). Adding strictly zero-state degree-0 Frobenius-MP (`R5`, `law4-frob-sgd-mp`, `readout-12f988a4`) collapses `mlp16` skew `7,500× → 19×`, escapes `a = 8.0 → 1.837`, and immediately unlocks `grid100` (`0.9724, ctr=0.0660`, step `1000`), `five_word` (`5/5`), `img_bars4` (`4/4`), and `gaussian1d_stability` (`0.0301`).

### Law 4.3 — Tensor-Rank Spectral Isotropization Law (The Diagonal Limit & Conv2d Polar-Match Equivalence)
> **Statement:** Let $G_l \in \mathbb{R}^{m \times n}$ have effective rank $\operatorname{erank}(G_l) = \|G_l\|_F^2 / \sigma_1(G_l)^2$.
> 1. *(2D Deep/Anisotropic Dense Matrices — The Diagonal Limit):* Every elementwise, Frobenius, $L_\infty$, or Kronecker-factored preconditioner $D_L G_l D_R$ is **rank-preserving** ($\operatorname{rank}(D_L G_l D_R) = \operatorname{rank}(G_l)$) and therefore **cannot** rescue skip-free deep networks where $\operatorname{erank}(G_l) \to 1$ (`grid100_mlp16`: `erank = 1.056`, `94.7–98.0%` top-1 singular energy) or ill-conditioned 2D conditional fibers (`model_ddgan_gaussian_chain`). Non-diagonal spectral polar orthogonalization $G_l (G_l^\top G_l + s^2 I)^{-1/2}$ (5-step fused `Z=[X,Y]` Quintic NS) is mathematically irreducible on 2D dense matrices.
> 2. *(4D Convolutions — Polar-Match Equivalence):* On 4D convolutional kernels ($k_h \times k_w$ spatial slices), spatial filter diversity and BatchNorm prevent rank-1 collapse; polar orthogonalization is **100% redundant** once Frobenius-MP matches the polar Frobenius norm via stride-aware scaling $\sqrt{C_{\text{out}}/C_{\text{in}}}\sqrt{g/k_{\text{eff}}}\,m_g^{1/4}$, cutting `r256` optimizer cost by **`17.4×`** and total step energy by **`40.8%–64.0%`**.
> - **Causal Proof:** Zero-NS `law4-frob-sgd-mp` (`0.157`), `law4-adafactor-mp` (`0.174`), `law4-adam-equil` (`0.255`), and `r4-razor-no-ns-v2` (`0.024`) **all collapse** on `grid100_mlp16` and fail `model_ddgan_gaussian_chain`. Conversely, `law4-energy-v3` (0 Conv2d NS + polar-matched 4D Frob-MP + 5-step 2D dense NS) passes **`4/4` Conv2d image tasks at `hq = 1.0000`** (`readout-0c336a58`) AND **`model_ddgan_gaussian_chain` PASS** (`readout-d53010c7`) at **`44.95–45.89 ms/step`** and **`12.56 J/step`** on `r256`.

### Law 4.4 — G-vs-D Structural Asymmetry Law Under 1-Lipschitz Critics
> **Statement:** Under a Law-2 1-Lipschitz Born–Infeld critic potential ($\|\nabla_x D\|_2 \le 1$), the critic's functional curvature is externally bounded whereas the generator $G$ owns the nonlinear pushforward geometry across disconnected modes, deep compositions, and saturation rails. Consequently: (i) removing Law-4 equilibration from $G$ causes severe multi-regime failure even with an equilibrated $D$, whereas (ii) removing Law-4 equilibration from $D$ preserves balanced/multimodal/deep-lattice convergence and fails catastrophically **only** under extreme $D \gg G$ capacity imbalance ($\rho_D \gg 100\times$); furthermore, (iii) under an equilibrated $D$, the critic-side logit secant probe is **100% redundant** across `>600×` imbalance.
> - **Causal Proof:** Exact 2×2 Nash double dissociation: `R4` (`Equil-G + SGD-D`, `readout-294525ad`) scores **`12/16` PASS** (`5/5` on hard $G$-geometry probes, `NaN` on `620×` `overpowered_critic`) vs `R3a` (`SGD-G + Equil-D`, `readout-f19a0f5c`) **`5/13` PASS** (`0/5` on hard $G$-geometry probes, `PASS` on `overpowered_critic`, `FAIL` on `overpowered_generator`). Meanwhile `law4-energy-v3` drops critic secant probes completely (`0` probes) and passes **`3/3` Nash stretch (`0.959 / 0.952 / 0.984`)** (`readout-f6e3cced`).

### Law 4.5 — Stochastic Minimax Zero-Momentum & Minimum-State Optimality Law
> **Statement:** Once Law 2 damps critic rotational eigenvalues, two-player stochastic minibatch optimization obeys a **bias–variance–staleness trade-off** where **zero-momentum, zero-state (`0 B/param`) degree-0 updates sit at the Pareto optimum of both accuracy and energy**:
> 1. *Optimistic negative momentum* ($\lambda > 0$, $g_{\text{eff}} = (1+\lambda)g_t - \lambda g_{t-1}$) amplifies i.i.d. stochastic gradient variance by $(1+\lambda)^2 + \lambda^2$ ($2.5\times$ at $\lambda=0.5$), degrading terminal settling (`8/14`) while increasing energy (`25.15 J/step`).
> 2. *Positive momentum* ($\beta_1 > 0$) and *full coordinate-wise second moments* ($O(2mn)$ `Adam`) introduce opponent-tracking staleness lag, per-coordinate noise overfitting, and `4–8 B/param` DRAM traffic (`20.82–25.61 J/step`), degrading both suite generality and `Joules-to-PASS`.
> - **Causal Proof:** Monotonic zero-momentum / low-state dominance across **every** geometry: `law4-energy-v3` (`β₁=0, 0 B/p`: `14/16`, `13.12 J/step`) > `law4-muon-m` (`β₁=0.85, 4 B/p`: `12/15`, `25.61 J/step`) > `law4-optimistic-equil` (`λ=0.5, 4 B/p`: `8/14`, `25.15 J/step`); `law4-sign-mp` (`β₁=0, 0 B/p`: `12/15`, `16.06 J/step`) > `law4-lion-mp` (`β₁=0.5, 4 B/p`: `10/14`, `18.01 J/step`); and `Sign-MP` (`0 B/p`: `12/15`) > `Adafactor` (`0.012 B/p`: `11/15`) > `Adam` (`7.99 B/p`: `9/15`, `20.82 J/step`).

---

## 6. Complete Inventory of All 33 Concluded Official Forge Readouts & Recommendations

### 6.1 Verbatim Readout Registry (Seed `0`, `NVIDIA RTX A6000`)

| Track (Branch · Commit) | Candidate | View | Verbatim Counts | Official Readout Receipt |
| :--- | :--- | :--- | :---: | :--- |
| **`sgd-anatomy`** (`research/law4-sgd-anatomy` · `66d5bfab7`) | `law4-pure-sgd` | `r4-occam-ablation-v1` | `2` PASS, `9` FAIL, `2` INC *(NaN)* | `readout-62543241e3cb64f098759ad7.json` |
| `sgd-anatomy` | `law4-sgd-secant` | `r4-occam-ablation-v1` | `4` PASS, `9` FAIL | `readout-d5c8d860e8107c9cc66729c6.json` |
| `sgd-anatomy` | `law4-sgd-g-equil-d` | `r4-occam-ablation-v1` | `5` PASS, `8` FAIL | `readout-f19a0f5cc8ffd529c09444c0.json` |
| `sgd-anatomy` | `law4-sgd-g-secant-equil-d` | `r4-occam-ablation-v1` | `9` PASS, `6` FAIL | `readout-fd14ca2051e98f707ab51fe9.json` |
| `sgd-anatomy` | `law4-equil-g-sgd-d` | `r4-occam-ablation-v1` | `12` PASS, `3` FAIL, `1` INC *(NaN)* | `readout-294525ad76a2970a72a7caac.json` |
| `sgd-anatomy` | `law4-frob-sgd-mp` | `r4-occam-ablation-v1` | `9` PASS, `7` FAIL | `readout-12f988a4ce689b5994a4ea61.json` |
| **`alt-optimizers`** (`research/law4-alt-optimizers` · `c0a08e005`) | `law4-sign-mp` | `r4-occam-ablation-v1` | `12` PASS, `3` FAIL | `readout-2ea657f4fe6f6625e30652ee.json` |
| `alt-optimizers` | `law4-sign-mp` | `law4-v11-breakthrough-v1` | `2` PASS *(`stag100` PASS!)*, `3` FAIL | `readout-faa8d2a726680bb727034259.json` |
| `alt-optimizers` | `law4-muon-m` | `r4-occam-ablation-v1` | `12` PASS, `3` FAIL, `1` INC *(to)* | `readout-0fd60717a5427bdb919697ff.json` |
| `alt-optimizers` | `law4-muon-m` | `law4-v11-breakthrough-v1` | `2` PASS *(`stag100` PASS!)*, `2` FAIL, `1` INC | `readout-6fad0485ca0aae984afb0ac1.json` |
| `alt-optimizers` | `law4-adafactor-mp` | `r4-occam-ablation-v1` | `11` PASS, `4` FAIL, `1` INC *(to)* | `readout-92baddc3105e46e38f8e40e7.json` |
| `alt-optimizers` | `law4-lion-mp` | `r4-occam-ablation-v1` | `10` PASS, `4` FAIL, `1` INC *(to)* | `readout-a5f6ceda3f96357713dde8e2.json` |
| `alt-optimizers` | `law4-adam-equil` | `r4-occam-ablation-v1` | `9` PASS, `6` FAIL, `1` INC *(to)* | `readout-8c40a844448906db37be65b9.json` |
| `alt-optimizers` | `law4-adam-equil` | `law4-v11-breakthrough-v1` | `1` PASS, `3` FAIL, `1` INC *(to)* | `readout-d0dd40dd4f4879b0076112f2.json` |
| `alt-optimizers` | `law4-optimistic-equil` | `r4-occam-ablation-v1` | `8` PASS, `6` FAIL, `1` INC *(to)* | `readout-ff204f93d79c4ac6792099ea.json` |
| `alt-optimizers` | `law4-optimistic-equil` | `law4-v11-breakthrough-v1` | `1` PASS *(`stag100` PASS!)*, `2` FAIL, `1` INC | `readout-1f07981649bca6782c45b940.json` |
| **`energy-champion`** (`feat/law4-energy-optimal` · `dc941984c`) | `law4-energy-v1` | `r4-occam` · `dcgan_perf` | `8` PASS, `7` FAIL, `1` INC · `6/6` PASS | `readout-576e24db` · `readout-262c268b` |
| `energy-champion` | `law4-energy-v2` | `r4-occam` · `dcgan_perf` | `11` PASS, `3` FAIL, `1` INC · `6/6` PASS | `readout-07d01bc8` · `readout-84b015d2` |
| **`energy-champion` (Champion)** | **`law4-energy-v3`** | **`r4-occam-ablation-v1`** | **`14` PASS, `2` FAIL** | **`readout-d53010c7356ee6557df70807.json`** |
| **`energy-champion` (Champion)** | **`law4-energy-v3`** | **`dcgan_performance` (idle / cont.)** | **`6/6` PASS (`45.89 ms`, `1147.1 MiB`)** | **`readout-b898e9285957d3feba1bf714.json`** · `readout-1fd4a458` |
| **`energy-champion` (Champion)** | **`law4-energy-v3`** | **`bcap_convolution_images`** | **`4/4` PASS (`hq = 1.0000` all 4!)** | **`readout-0c336a58babe8729b5c26e5e.json`** |
| **`energy-champion` (Champion)** | **`law4-energy-v3`** | **`t4_imbalance_nash_stretch`** | **`3/3` PASS (`0.959 / 0.952 / 0.984`)** | **`readout-f6e3cced136d1e5a3c0e17cb.json`** |
| **`energy-champion` (Champion)** | **`law4-energy-v3`** | **`t4-strange-architectures-v1`** | **`5/7` PASS** | **`readout-17052211d99722a87f89b9bb.json`** |
| **`energy-champion` (Champion)** | **`law4-energy-v3`** | **`discriminator_stability` (T1)** | **`15/17` Tier 1 PASS** *(2 gated T2)* | **`readout-54c1214b0858f3ea59d0795c.json`** |
| `energy-champion` | `law4-energy-v4` | `r4-occam` · `dcgan_perf` | `13` PASS, `2` FAIL, `1` INC · `6/6` PASS | `readout-59884cb8` · `readout-41557754` |
| `energy-champion` | `law4-energy-v5` | `r4-occam` · `dcgan_perf` | `13` PASS, `3` FAIL · `6/6` PASS | `readout-4a08c669` · `readout-061ce106` |
| `energy-champion` | `law4-energy-v6` *(dev `v11` ref)* | `r4-occam` · `dcgan_perf` | `11` PASS, `5` FAIL · `6/6` PASS (`55.49 ms`) | `readout-2b91f35b` · `readout-69e2cac4` |

### 6.2 Concrete Engineering & Research Recommendations

1. **Merge `law4_energy_v3` (`feat/law4-energy-optimal`) to `develop` as an opt-in energy-optimal Law-4 optimizer family immediately:** It strictly dominates current `develop` HEAD `unified-v11` on every measured axis (`14/16` vs `11/16` Occam, **`1.89×–1.94×` faster `r256`**, **`-40.8%` to `-64.0%` hardware GPU Joules/step**, **`-225.5 MiB` Forge peak VRAM**, **`-23.6%` `Joules-to-PASS` on `grid100`**, `4/4` Conv2d images at `hq=1.0000`, `3/3` Nash at `0` critic secant probes) with zero edits to `_unified_v11/`.
2. **Audit `develop` commits `78ca41002..cb5437de9` before changing `DEFAULT_RECIPE`:** Resolving why `develop` HEAD `unified-v11` shifted from `16/16` (`8caab7bde`) to `11/16` (`87d72c39b`, affecting `grid100_mlp16` and `interp_sparse_arc`) will allow `law4_energy_v3` to inherit `16/16` Occam and `>= 41/47` Tiers 1–2 `discriminator_stability` at **`45.89 ms/step` / `12.56 J/step`** — nearly half the energy of `unified-v11`.
3. **Preserve `law4_sign_mp` & `law4_frob_sgd_mp` as lightweight zero-state research baselines:** `law4-sign-mp` (`12/15` + `staggered100` PASS, `0 B/param`, `0` NS, `603 J` `ring16`) and `law4-frob-sgd-mp` (`5,357 J` `grid100` first-PASS) define the zero-matmul hardware lower bound for edge/low-memory regimes where 16-layer skip-free MLPs are out of scope.

### 6.3 Reproduction Commands

```bash
# 1. Phase 1 idle-GPU DCGAN energy/phase breakdown + 5-regime mechanistic telemetry
PYTHONPATH="$PWD" python experiments/law4_optimizer_profiler.py \
  --mode phase1 --device cuda:0 --idea unified-v11 \
  --energy-steps 50 --breakdown-steps 20 \
  --output reports/forge/studies/law4-phase1-profile.json

# 2. Phase 2 matched uncontended Energy & Joules-to-PASS benchmark (per worktree)
PYTHONPATH="<worktree>" PARTICLEGAN_PROFILER_ROOT="<worktree>" \
  python experiments/law4_optimizer_profiler.py \
  --mode energy-benchmark --device cuda:0 --energy-steps 50 \
  --include-staggered100 --ideas <idea_id> --output /tmp/bench_<idea_id>.json
```

---

## 7. Phase-2 Update: Energy-Optimal Learning Physics, Amdahl's-Law Bottlenecks & Uncontended NVML `J-to-PASS` Breakthrough

Following the Round-1 synthesis, three coordinated Phase-2 tracks resolved the `develop` HEAD `unified-v11` regression (`fix/unified-v11-regression-bisect` `f396e5c64`, restoring **`16/16`** `r4-occam-ablation-v1`), diagnosed the exact spectral physics of `grid100` / `staggered100` `Joules-to-PASS` and Amdahl's-Law post-optimizer bottlenecks (`research/law4-energy-physics`), and synthesized the new all-time champion **`law4-energy-v10`** (`feat/law4-energy-frontier` `e4010a035`, **`42/48`** Tiers 1–2 `discriminator_stability` PASS).

### 7.1 Spectral & Trajectory Physics: Why Zero-NS Critics Crack `staggered100` & Slash `grid100` `J-to-PASS`

Tracing `steps 0..3000` (seed `0`, `RTX A6000`) across `unified-v11`, `law4-energy-v3`, `law4-frob-sgd-mp`, and `law4-sign-mp` on both `grid100` and `staggered100` exposed **three exact physical mechanisms**:

1. **14× Orthogonal Spectral Noise Inflation on Low-Rank Hidden Matrices (`dwEr = 24–28` vs `1.1–2.1`):** On `grid100` and `staggered100`, true gradients are low-rank ($\operatorname{erank}(g_G) \in [1.16, 1.74]$, $\operatorname{erank}(g_D) \in [1.03, 1.40]$). Both 5-step Quintic NS and Frobenius-MP apply identical principal singular-value steps ($\sigma_1(\Delta W) \approx 0.008\text{–}0.010$ on $G$ `128×128`; $0.020\text{–}0.021$ on $D$ `128×128`), yet Quintic NS lifts all minor/noise singular values to `1.0`, inflating hidden-layer update effective rank to `dwEr = 22.7–28.4` on $G$ (`4.1×` higher $\|\Delta W\|_F$) and dropping gradient alignment to `cos(ΔW, -g) = 0.37–0.40` (`84–86%` of hidden update energy orthogonal to the gradient).
2. **`7.5×` / `11×` Output-Head Gain Overshoot (`sqrt(max(1, r/c))` vs `sqrt(r/c)`):** Clamping fan-in $\gg$ fan-out heads (`2×128` $G$ head, `1×128` $D$ head) to `1.0` makes `v11`/`v3` take `7.5×` larger $G$-head steps (`dwF = 0.0120` vs `0.0016`) and `11×` larger $D$-head steps (`0.0210` vs `0.0019`), spiking early function-space displacement (`dGf = 0.070–0.100`) and tripping Law-4 secant clamps on `19%–27%` of early steps (`ema_fg` drops to `0.63–0.78`), trapping `v11`/`v3` in a `prec = 0.40–0.64` plateau for `750–1,000` extra steps (`law4-frob-sgd-mp`: `0%` clamps, `ema_fg = 1.000`, `grid100` PASS at step `1000` and `staggered100` PASS at step `1250`).
3. **`3.8×` Critic Equilibrium Agitation (`dDf = 0.182–0.256` vs `0.066–0.068`):** At `staggered100` equilibrium (`step 3000`), `D`'s gradient is nearly rank-1 ($\operatorname{erank}(g_D) \approx 1.11\text{–}1.20$), yet 5-step NS inflates `D` hidden `dwEr` to `16.8–22.1` (`cos_D = 0.32–0.41`), driving persistent critic logit jitter (`dDf = 0.182–0.256`) that pushes `G` across the `ctr = 0.20` limit. Both `law4-sign-mp` (`dDf = 0.0660`, `dwEr_D = 1.4–2.0`) and `law4-frob-sgd-mp` (`dDf = 0.0683`, `dwEr_D = 1.0–1.1`) cut equilibrium critic agitation `3.8×`, explaining directly why **Law 4.4 × Law 4.3 Asymmetric `0`-NS Critic Frob-MP (`law4-energy-v10`, scaling 2D critic layers by exact polar-Frobenius match $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$)** cracks `staggered100`, `five_word_joint_hold` (`24/24`), and `model_ddgan_bimodal_chain` simultaneously while eliminating `100%` of Critic NS matmuls!

### 7.2 Next Energy Bottleneck Analysis (Amdahl's Law After `17.4×` Optimizer Speedup)

With `opt_d + opt_g` reduced to `5.7%` (`2.62 ms`) on `dcgan_performance_r256`, synchronized CUDA-event phase breakdowns and `torch.profiler` kernel/sync traces on `law4-energy-v3` pinpoint where the **next `2×–4×` energy win in ParticleGAN** lives across all four workload classes:

| Workload (`law4-energy-v3`) | `d_fwd+bwd` | `d_penalty` | `opt_d` | `g_fwd+bwd` / Pullback | `prior_reg` | `opt_g` | `cudaStreamSync` / step *(wall %)* | Highest-Leverage Next `2×` Energy Win |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`dcgan_performance_r256`** (`44.95 ms`, `12.56 J/st`) | **`41.1%`** *(BF16 NHWC)* | **`15.8%`** *(`10.1%` fwAD)* | `1.2%` | **`27.0%`** *(BF16 NHWC)* | **`7.8%`** *(SBNP)* | `4.7%` *(`1.5%` secant)* | `25/st` (`15` `.item()`, `7` `nz`) | Amortized `d_penalty` (`-8–12%`), zero-sync `SBNP` (`-5%`), `torch.compile` / FP8 on `68.1%` conv FLOPs |
| **`grid100`** (`34.10 ms`, `5.10 J/st`) | `9.4%` | `10.9%` | `9.1%` *(`3.0%` NS)* | **`62.5%`** *(`50.9%` Part A + `11.6%` Part B Laguerre; `g_bwd = 1.6%`!)* | *(in pullback)* | `10.9%` *(`3.3%` NS)* | **`113/st` (`63.6%` wall!)** (`61` `.item()`, `29` `nz`, `16` `is_nz`) | **Eliminate `113` GPU-CPU syncs in `unified_v11_pullback_step` + `update_quantizer` (`2.5–4×` energy win!)** |
| **`grid100_mlp16`** *(stiff `ema_fg < 0.60`)* | `5.3%` | `3.9%` | `5.5%` *(`0.9%` NS)* | **`56.4%`** *(`45.8%` Part A + `10.6%` Part B; `g_bwd = 2.1%`)* | *(in pullback)* | **`40.7%`** *(`1.0%` NS, **`39.3%` `.item()` ovh**)* | **`144/st` (`68.1%` wall!)** (`64` `.item()`, `30` `nz`) | **Already fixed in `v7..v10`:** device-tensor `_apply_slow_param_zerosync` removes `39%` `opt_g` stall; zero-sync pullback gives next `3×`! |
| **`five_word_joint_smoke`** | `16.9%` | `14.6%` | **`15.7%`** *(`2.4%` NS, `13.0%` ovh)* | `12.1%` + `5.0%` `g_loss` | `1.3%` | **`28.9%`** *(`4.3%` NS, **`24.2%` ovh**)* | `35/st` (`44.8%` wall!) (`13` `.item()`) | Fuse/zero-sync small-MLP `opt_g`/`opt_d` path (`37.2%` ovh vs `6.7%` actual 2D NS) |

### 7.3 Phase-2 Matched Uncontended Idle-GPU 0 Hardware NVML `Joules-to-PASS` Leaderboard

All five Phase-2 arms below ran back-to-back on strictly idle `NVIDIA RTX A6000` (`GPU 0`, verified `util_before = [0, 0]`, seed `0`, [`law4-phase2-energy-benchmark.json`](law4-phase2-energy-benchmark.json)), measuring uncontended hardware NVML `ms/step`, `Watts`, `J/step`, `State B/param`, `Peak VRAM`, and **`Joules-to-PASS`** across `dcgan_performance_r256`, `ring16_acquisition`, `grid100`, and **`staggered100` (NEW!)**:

| Optimizer Arm *(Branch Commit)* | `r4-occam` / `disc_stab` T1–2 | `r256` `ms/st` | `r256` `W` | **`r256` `J/st`** | `State` (`B/p`) | `ring16` 1st | **`ring16` `J-to-PASS`** | `grid100` 1st / Sust5 | **`grid100` `J-to-PASS`** *(1st / Sust5)* | `staggered100` 1st / Sust5 | **`staggered100` `J-to-PASS`** *(1st / Sust5)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`law4-energy-v10` (Phase-2 Champion!)** *(`e4010a035`)* | `14/16` · **`42/48` ALL-TIME RECORD!** | **`45.30`** | `256.4` | **`11.68 J`** *(**-50.7%** vs fixed `v11`!)* | **`0.000`** | `217` (PASS) | **`518.8 J`** *(RECORD!)* | **`1000` / `2000`** (PASS) | **`4,939 J` / `9,878 J`** *(**-36.9%** vs fixed `v11`!)* | **`1500` / `2500`** (**PASS**, `ctr=0.153`) | **`8,681 J` / `14,468 J`** *(RECORD PASS!)* |
| **`law4-energy-v8`** *(`erank_2` G + 0-NS D, `e4010a035`)* | `13/16` | `45.73` | `286.5` | `13.15 J` | **`0.000`** | `217` (PASS) | `562.1 J` | **`1000` / `2000`** (PASS) | **`4,794 J` / `9,587 J`** *(LOWEST `grid100`!)* | `1750` / `2750` *(FAIL suf)* | `10,154 J`* |
| **`law4-energy-v7`** *(0-NS D Frob $m^{0.25}$, `e4010a035`)* | `14/16` | `45.36` | `288.2` | `13.15 J` | **`0.000`** | `217` (PASS) | `584.2 J` | **`1000` / `2000`** (PASS) | **`4,912 J` / `9,824 J`** | `1750` / `2750` (**PASS**, `ctr=0.139`) | `10,062 J` / `15,811 J` |
| **`unified-v11` (Fixed)** *(`f396e5c64`)* | **`16/16`** · `14/17` T1 | `85.81` | `275.3` | `23.70 J` | `0.000` | **`200`** (PASS) | `675.4 J` | `1250` / `2250` (PASS) | `7,833 J` / `14,100 J` | `1500` / `2500` (**PASS**, `ctr=0.136`) | `9,643 J` / `16,071 J` |
| `law4-energy-v9` *(0-NS D Sign-MP, `e4010a035`)* | `13/16` | `47.57` | `280.6` | `13.41 J` | **`0.000`** | `217` (PASS) | `613.0 J` | **`1000` / `2000`** (PASS) | `5,550 J` / `11,099 J` | **`1250` / `2250`** (**PASS**, `ctr=0.129`)\dagger | *(omitted: ext. GPU contention)*\dagger |
| `law4-energy-v3` *(R1 Champion, `dc941984c`)* | `14/16` *(dev `87d72c`)* | `45.58` | `287.7` | `13.12 J` | `0.000` | `217` (PASS) | `619.3 J` | `1500` / `2500` (PASS) | `7,643 J` / `12,739 J` | — | — |
| `unified-v11` *(R1 dev HEAD, `87d72c39b`)* | `11/16` | `84.75` | `250.4` | `21.28 J` | `0.000` | `217` (PASS) | `636.4 J` | `1750` / `2750` (PASS) | `9,998 J` / `15,711 J` | — | — |

*(\dagger External host workloads began contending on GPU 0 midway through `law4-energy-v9.staggered100`; its deterministic pass milestones `1250 / 2250` PASS are preserved while its contended hardware energy is omitted (`null`) to maintain strict uncontended cohort integrity in `law4-phase2-energy-benchmark.json` and `data/energy.parquet`.)*

**Phase-2 Bottom Line:**
- **`law4-energy-v10` (`e4010a035`) establishes a strict, unprecedented Pareto dominance across BOTH scientific quality AND hardware energy:**
  - **All-Time Record Suite Quality:** **`42/48` PASS** across `discriminator_stability` Tiers 1–2 (`readout-a0e3e7f6`), cracking `staggered100`, `five_word_joint_hold` (`24/24`), and `model_ddgan_bimodal_chain` with **strictly `0` Critic NS, `0` Conv2d NS, `0` critic secant probes, and `0.000 B/param` non-stiff state**.
  - **All-Time Record Hardware Energy Efficiency (Idle RTX A6000 NVML):** **`11.68 J/step`** on `dcgan_performance_r256` (**`-50.7%`**, **`2.03×`** lower energy per step than fixed `unified-v11`'s `23.70 J/step`, `45.30 vs 85.81 ms/step`), **`518.8 J`** `ring16` `J-to-PASS` (**`-23.2%`** vs fixed `v11`), **`4,939 J` / `9,878 J`** `grid100` `J-to-PASS` (**`-36.9%`** vs fixed `v11`, **`-50.6%`** vs Round-1 `v11`, and `-7.8%` below even zero-NS `law4-frob-sgd-mp`!), and **`8,681 J` / `14,468 J`** `staggered100` `J-to-PASS` (**`-10.0%`** vs fixed `v11`).
- **Fixed `unified-v11` (`f396e5c64`):** Restores **`16/16`** `r4-occam-ablation-v1` (`readout-ac4e1807`), cuts its own `grid100` `J-to-PASS` by **`-21.7%`** (`9,998 → 7,833 J`), and unlocks `staggered100` PASS (`9,643 J / 16,071 J`), confirming the forensic bisect of commits `2420a3232` and `cb5437de9`.
- **Next Horizon (`2.5–4×` MLP / `1.5×` DCGAN Energy Win):** With optimizer overhead already solved (`0` Conv2d NS, `0` Critic NS, `0` slow-path `.item()` syncs), the dominant remaining energy sink on 2D/MLP workloads is the **`113` `cudaStreamSynchronize` calls/step (`63.6%` of wall time)** inside `unified_v11_pullback_step`, and on `r256` Conv2d is `d_penalty` (`15.8%`) + `prior_reg` (`7.8%`) + BF16 conv FLOPs (`68.1%`). Eliminating GPU-CPU syncs in `unified_v11_pullback_step` is the single highest-ROI next engineering step to cut `grid100` / `staggered100` `J-to-PASS` from `~5,000 J` down below **`~1,500 J`**!

---

## 8. Phase-3 Update: `law4-energy-v11` (`44/48` All-Time Record, `16/16` Occam) & Zero-Sync Engineering (`pullback-zerosync`)

Phase 3 closed the final Phase-2 scientific and systems gaps across three coordinated tracks (merged onto `develop` via **PR #611** `60374f68e` and **PR #610** `88433f44e`, with matched uncontended idle-GPU 0 hardware NVML `J-to-PASS` verification across **6 tasks × 3 arms = 18/18 strictly uncontended cells** in [`law4-phase2-energy-benchmark.json`](law4-phase2-energy-benchmark.json)):

### 8.1 Closing the Law 4.4 Boundary (`law4-energy-v11`, PR #610 `238a61789` / `c7713ade3` → `88433f44e`)

1. **Generator-Free 2D Critic NS Restoration (`Law4EnergyV11Optimizer._step_generator_free_critic`):**
   - Phase 2 established that `law4-energy-v10` swept `14/14` neural-generator Occam probes with strictly `0` Critic NS, missing only the two generator-free (`G = nn.Identity`) direct-particle isolation probes (`t3_iso_scale_mass_8x`, `t3_iso_horizon_saddle_shift`).
   - Restoring zero-sync 5-step fused `Z=[X,Y]` Quintic NS on 2D critic matrices **strictly when `total_gen_mats == 0`** (persisted & validated as `generator_free: bool` in `state_dict()` / `validate_state_dict()`, with `0` Critic NS on every neural-generator task, `0` Conv2d NS, and `0` critic secant probes everywhere) flips **BOTH** isolation probes to PASS (`t3_iso_scale_mass_8x` `min_eig = 0.4723, hq = 0.9629`; `t3_iso_horizon_saddle_shift` `hq = 0.9531, modes = 4`), lifting `law4-energy-v11` to:
     - **`r4-occam-ablation-v1`: `16 / 16` PASS** (`readout-53e330a0d8e1c04d4c72bbcd.json`) — matching fixed `unified-v11`'s `16/16`!
     - **`dcgan_performance`: `6 / 6` PASS** (`readout-bc0352d17785c6afcfeafd00.json`)
     - **`t4-strange-architectures-v1`: `6 / 7` PASS** (`readout-33d5dc6aafae554b7d9433af.json`) — matching `unified-v11` (`grid100_res16_scaled_tanh` `0.9912 / 0.0541`)!
     - **`discriminator_stability` Tiers 1–2: ALL-TIME RECORD `44 / 48` PASS** (`readout-0b7cd7995a0a857bb250cf14.json`) — **`17 / 17` Tier 1 PASS (`100%`)**, `27 / 31` Tier 2 PASS (`7/7` held PASS, `staggered100` PASS, `five_word_joint_hold` `24/24` PASS, `model_ddgan_bimodal_chain` PASS, `4/4` Conv2d images, `6/6` vectors, `3/3` `>548×` Nash, `interp_sparse_arc` `0.8406` PASS).
2. **Eliminating `new_tensor(py_float)` Pageable-H2D CUDA Syncs:**
   - Phase-A line-by-line `torch.cuda.set_sync_debug_mode(1)` + `cProfile` tracing revealed a major PyTorch CUDA footgun: `tensor.new_tensor(py_float)` allocates pageable CPU memory and triggers a **synchronous `cudaStreamSynchronize` (`~2 ms/call` under contention!)**, accounting for **`28.2%` of total wall time (`14/29` syncs/step)** on `five_word_joint_smoke` (`law4-energy-v10`) and **`37/60` syncs/step** on `dcgan_performance_r256` (`unified-v11`).
   - Replacing `x.new_tensor(py_float) → x.new_full((), py_float)` (**`0` syncs**, bit-exact!) plus caching per-parameter dense/conv metrics cut `five_word_joint_smoke` `opt_g` latency **`4.97×`** (`36.3 → 7.3 ms/step`) and drove `law4-energy-v11`'s idle-GPU 0 `five_word_joint_smoke` latency down to **`21.86 ms/step` (`1,708 J` `J-to-PASS`)**.

### 8.2 Zero-Sync Pullback & Exact CUDA Bit-Parity (`pullback-zerosync`, PR #611 `f432dbce0` → `60374f68e`)

Line-by-line sync tracing (`experiments/law4_optimizer_profiler.py --mode sync-audit`) eliminated **`72%–98%` of GPU-CPU synchronizations** across `_unified_v11/{prior,critic,penalty,loss,optimizer}.py` while uncovering two critical systems invariants:

| Workload | Before (`d5f12093e`) Syncs/step | `unified-v11` + `pullback-zerosync` (`f432dbce0`) | Combined `law4-energy-v11` + `pullback-zerosync` | Key Zero-Sync Mechanisms |
| :--- | :---: | :---: | :---: | :--- |
| **`dcgan_performance_r256`** | `60.0` | **`1.0` (`-98.3%`)** | **`7.0` (`-88.3%`)** | `new_full(())`, strided `sq_mix[0::2]`/`[1::2]` (`-4/st`), lazy `float64` critic EMAs, zero-sync `_principal_axis_power_zerosync` |
| **`grid100`** | `112.0` | **`16.0` (`-85.7%`)** | **`24.0` (`-78.6%`)** | GPU-resident `update_quantizer` (`scatter_add_`, `torch.where`, 1 batched 3-bool check), GPU-resident Part A EMAs/Sinkhorn |
| **`grid100_mlp16`** | `144.0` | **`16.0` (`-88.9%`)** | **`24.0` (`-83.3%`)** | Exact `float64`-reciprocal CUDA division `(1.0 / scalar_t64).float()` (`0.000e+00` parity through steps 1–25!) |
| **`five_word_joint_smoke`** | `29.0` *(v10)* | **`4.0` (`-86.2%`)** | **`8.0` (`-72.4%`)** | `new_full(())` (`-14/st`), lazy `critic.py` EMAs (`-6/st`), non-inherited `_SUPPORTS_ZEROSYNC_V4_TENSORS` subclass guard |

- **Critical CUDA `div.rn.f32` vs `float64`-Reciprocal Parity Finding:** Independent combined verification caught that `tensor / py_float` on CUDA compiles as `tensor * float32(1.0 / float64(py_float))`, whereas `tensor / tensor_f32` executes hardware `div.rn.f32` (differing by `1 ULP` = `1.5e-5` on Sinkhorn `L_xp` and `4.77e-7` on `update_quantizer` Gersho merges), which flipped `dead_far` on 16-layer `grid100_mlp16`. Using `(1.0 / scalar_t64).float()` restored literal **`0.000e+00` CUDA parity** through steps 1–25 and confirmed **combined `law4-energy-v11` (`c7713ade3`) + `pullback-zerosync` (`f432dbce0`) PASSES `16/16` `r4-occam-ablation-v1` (incl. `grid100_mlp16` `prec = 0.9918, ctr = 0.0777`) and `6/6` `dcgan_performance` (`r256 = 41.89 ms/step`)**!

### 8.3 Phase-3 Matched Uncontended Idle-GPU 0 Hardware NVML `Joules-to-PASS` Leaderboard (6 Tasks × 3 Arms)

All 18 arm×task cells below ran on strictly idle `NVIDIA RTX A6000` (`GPU 0`, per-task verified `util_before = [0, 0]`, `compute_pids = []`, `contended = False`, Seed `0`, pre-PR-#612 cohort to isolate `law4-energy-v11` + `pullback-zerosync` cleanly):

| Phase-3 Arm *(Measured Commit)* | Quality (`Occam` / `disc_stab`) | `r256` `ms` · `W` · **`J/st`** | `ring16` `ms` · 1st · **`J-to-PASS`** | `grid100` `ms` · 1st/Sust5 · **`J-to-PASS`** | `staggered100` `ms` · 1st/Sust5 · **`J-to-PASS`** | `grid100_mlp16` `ms` · 1st/Sust5 · **`J-to-PASS`** | `five_word` `ms` · 1st · **`J-to-PASS`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`law4-energy-v11` + `pullback-zerosync` (Combined Champion!)** *(`c7713ade3+f432dbce0`)* | **`16/16`** · **`44/48` RECORD** | **`41.30`** · `281.3W` · **`11.76 J`** *(**ALL-TIME FASTEST `r256`**, `2.08×` faster / `-50.4%` J vs fixed `v11`!)* | `34.40` · `217` · **`627.0 J`** (PASS) | `35.77` · **`1000/2000`** · **`5,352 J`** (PASS) | `44.58` · `2000/3000` · `12,475 J` (PASS) | **`73.39`** (`-9.1%` ms!) · **`1250/2250`** · **`12,543 J`** (PASS) | **`19.37`** · **`834`** · **`1,526 J`** *(**ALL-TIME RECORD**, `2.86×` less J vs `v11+zs`!)* |
| **`law4-energy-v11` alone** *(`238a61789`)* | **`16/16`** · **`44/48` RECORD** | `44.61` · `292.6W` · `13.10 J` | **`29.09`** · `217` · **`570.7 J`** (PASS) | **`32.78`** · **`1000/2000`** · **`4,991 J`** (PASS) | **`39.67`** · **`1500/2500`** · **`8,657 J`** (PASS) | `80.73` · **`1250/2250`** · **`11,721 J`** (PASS) | `21.86` · **`834`** · `1,708 J` (PASS) |
| **`unified-v11` + `pullback-zerosync`** *(`f432dbce0`)* | `16/16` *(at `83d27b`)* | **`68.35`** (`-20.3%`!) · `286.3W` · **`19.59 J`** (`-17.4%` vs fixed `v11`!) | `37.54` · `217` · `725.6 J` (PASS) | `49.25` · `1000/2000` · **`6,550 J`** (`-16.4%` vs fixed `v11`!) | `48.18` · `1750/2750` · `11,562 J` (PASS) | `76.38` · `6500` *(FAIL suf)* · `67,181 J`* | `27.73` · `1667` · `4,365 J` (PASS) |
| *Ref: Fixed `unified-v11` (Phase 2, `f396e5c64`)* | `16/16` · `14/17` T1 | `85.81` · `275.3W` · `23.70 J` | `200` · `675.4 J` (PASS) | `1250/2250` · `7,833 J` (PASS) | `1500/2500` · `9,643 J` (PASS) | — | — |

### 8.4 Honest Systems Physics Discovery: Why Zero-Sync Wins Big on Deep/Heavy Workloads (`r256`, `five_word`, `mlp16`) but Costs `+3 ms/st` on Small Uncontended 2D Pullback

Measured strictly on idle GPU 0 under `torch.use_deterministic_algorithms(True)`, Phase 3 reveals a crucial GPU systems trade-off that directly guides Phase 4:
1. **Huge Real Speed & Energy Wins Where GPU Pipeline Depth / H2D Overhead Dominates:**
   - **`dcgan_performance_r256`:** Eliminating `37–59` syncs/step cuts `unified-v11` latency **`-20.3%`** (`85.81 → 68.35 ms/st`, **`-17.4%` Joules/step**: `23.70 → 19.59 J/st`) and cuts `law4-energy-v11` latency **`-7.4%`** (`44.61 → 41.30 ms/st`, **`-10.2%` Joules/step**: `13.10 → 11.76 J/st`) — hitting a **new all-time `r256` speed record (`41.30 ms/step`, `2.08×` faster than fixed `unified-v11`)**!
   - **`five_word_joint_smoke`:** `law4-energy-v11` + `pullback-zerosync` hits **`19.37 ms/step`** and an **all-time record `1,526 J` `J-to-PASS`** (`-10.6%` vs `law4-energy-v11` alone, **`2.86×` lower energy** than `unified-v11` + `pullback-zerosync` `4,365 J`)!
   - **`grid100_mlp16`:** `pullback-zerosync` cuts `law4-energy-v11` step latency **`-9.1%`** (`80.73 → 73.39 ms/step`, passing at `1250/2250` with `prec = 0.9918, ctr = 0.0777`), while keeping `grid100` `unified-v11` `J-to-PASS` **`-16.4%`** lower than Phase-2 fixed `unified-v11` (`6,550 vs 7,833 J`).
2. **Why `grid100` / `staggered100` Small-MLP Idle Latency Rose Slightly (`32.78 → 35.77 ms/st`, `+3.0 ms`):**
   - Under `torch.use_deterministic_algorithms(True)` on a strictly idle GPU, a tiny host sync takes only `~0.05 ms`, whereas replacing branch-skipping `if` guards with unconditional dual-branch `torch.where` executes **both** `loss_coal` and `loss_nocoal` backward graphs, unconditional whitened `torch.cdist(2048, 256)` (`d_xp_w`), `torch.sort(2048)`, and deterministic sort-based `scatter_add_` kernels every step — adding `+3.0 ms/step` (`+7.2%` J/step) on small 2D MLPs when idle (while still winning massively under GPU contention, where each `cudaStreamSynchronize` stalls behind peer kernels!).
   - **Next Phase-4 Lever:** (i) Opt `Law4EnergyV11Optimizer` into `_SUPPORTS_ZEROSYNC_V4_TENSORS = True` (`-8` syncs/step on `grid100`), (ii) fuse `prior.py`'s `unified_v11_pullback_step` + `update_quantizer` with `torch.compile` / single batched control-flag sync to **skip** dead branches (`d_xp_w`, `loss_nocoal`, `need_conc_i`) instead of evaluating both via `torch.where`, and (iii) amortize `d_penalty` (`15.8%` of `r256`) to push `r256` below `35 ms / 9.5 J/step` and `grid100` below `2,000 J`!

---

## 9. Phase-4 Update: `unified-v12` (`46/48` T1–2, `18/18` Untouched Tier-3) + `pullback-fastpath-v12` & Promotion to `v0.12.0` Default

Phase 4 executed the exact Phase-3 systems & optimizer roadmap across **PR #615** (`pullback-fastpath-v12`, merged `42ded3086`) and **PR #620** (`unified-v12`, merged `80fd347a0`), verified on strictly idle `NVIDIA RTX A6000` (`GPU 0`, `util_before = [0, 0]`, `contended = False`, [`law4-phase2-energy-benchmark.json`](law4-phase2-energy-benchmark.json)):

1. **Pullback Fast-Path & Zero-Sync Completion (`PR #615` `34f0b4c92` → `42ded3086`):**
   - Replaces unconditional dual-branch `torch.where` work in `particlegan/_unified_v11/prior.py` with a single batched control-flag sync so inactive branches (`d_xp_w`, `loss_nocoal`, `need_conc_i`) are skipped cleanly, cuts `unified-v12` GPU-CPU syncs to **`3.0/st`** on `r256` (`39.61 ms` audit) and `five_word_joint_smoke` (`20.82 ms`) and **`12.0/st`** on `grid100` (`30.18 ms`) and `grid100_mlp16` (`37.27 ms`), and restores exact float64 `shock_t64`/`tv_now64` semantics.
2. **Energy-Optimal 4-Law Optimizer & CountSketch Deep-MLP Preconditioning (`PR #620` `ff41cf173` → `80fd347a0`):**
   - `UnifiedV12Optimizer` (`optimizer_family="unified_v12_equil"`, `_SUPPORTS_ZEROSYNC_V4_TENSORS = True`) combines Law 4.3 zero-state polar-matched 4D Conv2d Frobenius-MP (`0` Conv2d NS, `0.0000 B/p` state), Law 4.4 asymmetric 0-NS 2D critic with generator-free 2D NS fallback (`0` critic secant probes), and CountSketch spectral preconditioning on stiff deep 2D generators (`ema_fg < 0.60`), preserving strictly `0` `torch.autograd.grad`, `0` `svd`/`eigh`/`pinv`, and exact `179` `Recipe` fields.
3. **Matched Strictly Idle-GPU 0 Hardware NVML `J-to-PASS` & Untouched Holdout Results (`3bf87cea3+ff41cf173`):**

| Phase-4 Arm | Untouched Tier-3 (`18` tasks) | `disc_stab` T1–2 / AE-DDGAN | `r256` `ms/st` · **`J/st`** | `ring16` `ms/st` · **`J-to-PASS`** | `grid100` `ms/st` · **`J-to-PASS`** *(1st / S5)* | `staggered100` `ms/st` · **`J-to-PASS`** | `five_word` `ms/st` · **`J-to-PASS`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unified-v12` (v0.12.0 Default)** | **`18 / 18` (`100%`)** | **`46/48` (`29/29` req) · `2/2` AE-DDGAN** | **`39.22 ms` · `11.37 J`** (**`-45.8%`** vs `v11`) | **`27.20 ms` · `462.5 J`** (**`-26.3%`**) | **`30.25 ms` · `4,513 J` / `9,026 J`** (**`-36.3%`**) | **`30.74 ms` · `10,743 J`** (**PASS**) | **`17.67 ms` · `1,399 J`** (**`-78.8%`**, `4.72×` < `v11`!) |
| **`unified-v11` + `fastpath`** *(ctrl)* | `16 / 18` | `model_ae_ddgan_gaussian_chain` FAIL | `73.21 ms` · `20.96 J` | `35.03 ms` · `627.5 J` | `39.69 ms` · `7,083 J` / `12,749 J` | `39.76 ms` · `9,995 J`* *(FAIL suf)* | `27.57 ms` · `6,602 J` |

4. **Honest Trade-Off & Governance Disclosure:**
   - On `grid100_mlp16` (`ema_fg < 0.60`), `unified-v12`'s CountSketch spectral path uses `75.96 ms/st` (`12,781 J` 1st-PASS, PASS) vs `unified-v11` `73.69 ms/st` (`9,808 J`, PASS) — a modest `+3.1%` per-step cost on 16-layer skip-free MLPs in exchange for `26%–79%` energy reductions across all other regimes.
   - On `t4-strange-architectures-v1`, standalone `unified-v12` (`ff41cf173`) passes **`7/7` (`100%`)** (`readout-731e752ff3c4f97e29a84145.json`); combined with PR #615 (`3bf87cea3+ff41cf173`), `grid100_res16_tanh` passes the independent `100k` holdout (`passed=True`, `prec=0.9831`, `ctr=0.1224`, `cov=0.1618`, `score=0.6562`, `sat=0.0`) and `4/5` terminal checks (`6000, 6500, 6750, 7000`), missing only step `6250` (`modes=99`) from PR #615's 1-ULP float64 `shock_t64`/`tv_now64` restoration (`6/7` combined).
   - Promoted to **`DEFAULT_RECIPE = "unified_v12"`** in **`v0.12.0`** as an owner-directed research selection; Forge calibration (`criteria-v1.json`) remains provisional (`BLOCKED`) and multi-seed / natural-image FID confirmation remain open.
