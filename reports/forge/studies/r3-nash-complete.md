# Round-3 Track 3 Study: Pure-Elementwise (`0` SVD · `0` `eigh` · `0` `pinv` · `0` Newton–Schulz · `0` `autograd.grad`) `3/3` Imbalance-Invariant Nash & Tier-1/Tier-4 Completion (`t4_nash_v3`)

**Branch:** `feat/solve-gans-r3-nash-complete`  
**Formulation:** Pure-Elementwise Impedance-Matched Nash Equilibrium with Pullback Burg Covariance Isometry (`critic_formulation="t4_nash_v3"`, preset `"t4_nash_v3"`, public API [`particlegan/nash_v3.py`](../../../particlegan/nash_v3.py) / [`particlegan/_nash_v3/`](../../../particlegan/_nash_v3/))  
**Compute Complexity & Hard Primitive Audit:** Strictly **`0` `svd` · `0` `eigh` · `0` `pinv` · `0` Newton–Schulz · `0` `torch.autograd.grad`** per step ($O(\text{params}) + O(BKD)$ low-D / $O(\text{params}) + O(B)$ images), monkeypatch-enforced across 1D/2D/170D/Conv/constraint-geometry paths in [`tests/test_nash_v3.py`](../../../tests/test_nash_v3.py).  
**Frozen Prior Receipts:** [`particlegan/_t4_nash/`](../../../particlegan/_t4_nash/) and all PR `#571` (`t4-nash-v1`/`v2`) receipts remain completely untouched.

**Concluded Candidates & Official Forge Readout Receipts (`seed = 0`, `NVIDIA RTX A6000`):**
- **`t4-nash-v3`** (`4ce780c06a3414187c2a94cf69657afb3cc46d41eefd6fe24ad51fba89c7c324` — one unchanged global recipe across all views):
  - **`t4_imbalance_nash_stretch`:** **`3 / 3` (`100%`) sustained `5/5` `PASS`** ([`readout-6f996ec2`](../records/readout-6f996ec256a149b2f1620f64.json))
  - **`discriminator_stability` (Tiers 1–2):** **`11 / 11` (`100%`) Required Tier 1 `PASS`** (`16/17` total Tier 1 incl. `clockfree` + `fp32/mxfp8/nvfp4` + `t3_iso_scale_mass_8x`), **`17 / 29` Tier 2 `PASS`** (`10/16` required, `4/6` stretch incl. `t3_iso_horizon_saddle_shift` + `3/3` Nash, `3/7` held), **`33 / 46` overall `PASS` (`0` BLOCKED, `0` ERROR)** ([`readout-1ca76c61`](../records/readout-1ca76c61c875ba5a33dca488.json))
  - **`dcgan_performance` (`32²`–`256²` RGB):** **`6 / 6` (`100%`) `PASS`**, **`39.68 ms/step` at `256×256` RGB** (`3.92×` faster than `empf-v8`, `14.84×` faster than `rhbi-v12`, `2.50×` faster than `t4-nash-v1`) ([`readout-642d4c55`](../records/readout-642d4c55d093b44ed1e0309e.json))
- **Bidirectional `100×` LR-Asymmetry Stress Sweeps (`t4_imbalance_nash_stretch`):**
  - **`t4-nash-v3-d10x`** (`3a97a4db39ea70d6...`, `10:1` critic LR): [`readout-84d832df`](../records/readout-84d832df34d5eb3f04f8113f.json)
  - **`t4-nash-v3-g10x`** (`4eb5094b538e898b...`, `1:10` critic LR): [`readout-696ab5f8`](../records/readout-696ab5f8318bc8b356af8217.json)

---

## 1. First-Principles Diagnosis of the Four Remaining `t4-nash-v2` Ceilings

In Round 2 (`PR #571`, [`t4-imbalance-nash.md`](t4-imbalance-nash.md)), `t4-nash-v2` proved that SVD-free Marchenko–Pastur + `peak_rms` + realized pushforward/inversion-deficit secant bands lift `t4_imbalance_nash_stretch` from `0/3` across all naive-Adam controls to `2/3` sustained `5/5` PASS and cut `256²` DCGAN latency to `99.35 ms/step`, while still leaving four concrete first-principles gaps:

1. **Second-Order `torch.autograd.grad` Overhead & Graph Materialization:** `T4NashCriticPenalty` still called `torch.autograd.grad(D(x), x, create_graph=True)` for R1/R2 regularization (`~60 ms` of double-backward overhead on `256²` RGB DCGAN), and `T4NashConstraintGeneratorAdam` called `torch.autograd.grad` for protected-loss normals.
2. **Underpowered Critic Angular-Bandwidth Pinch (`nash_overpowered_generator_ring16`, $G \gg D$):** A tiny `32×2` Fourier-2 critic (`1,443` params vs `790,532`-param generator — `548×` parameter ratio) lacks the angular harmonic capacity to simultaneously separate 16 modes and maintain circular local covariance around every mode, causing `t4-nash-v2` to pinch 1–2 modes (`core_min_eigen_ratio = 0.075 < 0.15`).
3. **1D Equilibrium Under-Dispersion & 2-Mode Median Cliff (`gaussian1d_*`, `precision_gaussian_*`, `model_vae_*`):**
   - In $d = 1$ ($B = 64$), entropic Sinkhorn pairing contracted tails by `~5%` (`std_ratio = 0.94–0.96`, `cdf_ks = 0.065–0.079 > 0.04`), while Born–Infeld tail scale $\tau_{\text{BI}} = 0.12\sigma$ clipped 1D Gaussian tails and the irreducible $O(B^{-1/2}) \approx 0.18$ empirical residual triggered false disequilibrium amplification.
   - On equal-weight 2-mode mixtures (`model_vae_categorical`, `model_vae_hard`), exactly 50% of real pairs lie within-mode ($d^2 \approx 0.01$), collapsing `pairwise_sq_dist(x_real).median()` by $30\times$ and inflating `ema_residual`.
4. **Coupled Encoder–Generator Primary-Probe Clobbering & Equilibrium Rattle (`model_ae_gan_multimodal`, `ring16_acquisition`):** In AE/VAE tasks, the reconstruction forward $G(E(x))$ ran after $G(z_{\text{prior}})$ and overwrote the secant probe cache, while full-sample bridge stragglers and an overly loose equilibrium step floor inflated `component_covariance_error` (`0.9016 > 0.85` on `ring16_acquisition`) and `component_max_eigen_ratio` (`> 4.0` on `model_ae_gan_multimodal`).

---

## 2. Mathematical Formulation of `t4_nash_v3`

Implemented cleanly in [`particlegan/_nash_v3/`](../../../particlegan/_nash_v3/), `t4_nash_v3` resolves all four ceilings simultaneously with **strictly `0` `svd`, `0` `eigh`, `0` `pinv`, `0` Newton–Schulz, and `0` `torch.autograd.grad` calls**:

### 2.1 Strictly First-Order Critic Regularization & Protected Gradients ([`penalty.py`](../../../particlegan/_nash_v3/penalty.py), [`optim.py`](../../../particlegan/_nash_v3/optim.py))
- **Low-Dimensional Manifolds ($d \le 8$) & Discrete Simplex ($d > 8$):** Replaces `torch.autograd.grad(..., create_graph=True)` with exact first-order forward-mode AD (`torch.autograd.forward_ad`, canonical basis for $d \le 4$ with central finite-difference fallback for quantized MXFP8/NVFP4 modules; Weyl/Householder orthonormal frames for $5 \le d \le 8$; Murmur32 Rademacher + local $k$-NN chord probes for $d > 8$). Because `fwAD` propagates directional derivatives $dD(x)[v]$ in a single primal pass, the outer `loss_d.backward()` computes $\nabla_\psi \|dD(x)[v]\|^2$ using standard first-order reverse mode only.
- **High-Dimensional Images (`dcgan_performance`, `img_*`):** Captures the live BF16 critic logits via a forward hook (`_nash_v3_critic_forward_hook`) and evaluates a single BF16 micro-secant probe $(D(x + \varepsilon u) - D(x))/\varepsilon$ — eliminating **100%** of double-backward conv-transpose passes and dropping `256×256` RGB step latency from `99.35 ms` (`t4-nash-v1`) to **`39.68 ms/step`**.
- **Protected-Loss Non-Ascent Projection:** `NashV3ConstraintGeneratorAdam._bind_protected_losses` extracts protected normals via standard `.backward(retain_graph=True)` with parameter-gradient save/restore (strictly `0` `torch.autograd.grad`) and solves the $1\times 1 / 2\times 2$ active-set projection analytically in FP64 via Cramér's rule (`0` `pinv`, `0` `svd`).

### 2.2 $O(BKD)$ Voronoi + Core-Ball Pinch-Weighted Symmetric Burg Pullback Covariance Isometry ([`transport.py`](../../../particlegan/_nash_v3/transport.py))
To supply circular local covariance restoration even when a `1.4k`-param critic cannot resolve 16 angular modes—without leaking mass across modes or dropping outliers—`pullback_covariance_isometry_loss` performs:
**1. Deterministic Adaptive $k$-NN Mode Covering:** Selects well-separated anchors $\{a_1,\dots,a_M\} \subset X_{\text{real}}$ via deterministic farthest-first covering at local $k$-NN radius $4.5\,\max(h_c^2, h_i^2)$ (zero RNG, zero knowledge of mode count $K$).

**2. Exact Voronoi + Core-Ball Partition:** Partitions *all* real and fake samples into Voronoi cells $C_m = \{i : m = \arg\min_k \|x_i - a_k\|^2\}$ (pulling every outlier and inter-mode bridge straggler back into its mode) and matched core balls $C_m^{\text{core}} = \{i \in C_m : \|x_i - a_m\|^2 \le 3.5\,h_m^2\}$ (preventing outliers from masking a pinched inner core).

**3. Closed-Form Pinch-Weighted Symmetric Burg Log-Det Divergence (0 `svd`, 0 `eigh`, 0 `pinv`):** For each valid cell $m$ with Bessel-corrected empirical covariances $(\Sigma_{r,m}, \Sigma_{f,m})$ and ridge $\epsilon_m I = 0.04\,\bar{\sigma}_m^2 I$:

$$
D_{\text{PB}}(\Sigma_r, \Sigma_f) = \underbrace{\operatorname{tr}(\Sigma_f^{-1}\Sigma_r) + \log\frac{|\Sigma_f|}{|\Sigma_r|} - d}_{D_{\text{pinch}} \sim \lambda_{\min}^{-1}\text{ wall against collapse}} \;+\; \frac{1}{2}\underbrace{\left(\operatorname{tr}(\Sigma_r^{-1}\Sigma_f) - \log\frac{|\Sigma_f|}{|\Sigma_r|} - d\right)}_{D_{\text{spread}}\text{ contraction of over-dispersion}}
$$

computed via exact analytic $2\times 2$ determinant/cofactor formulas for $d = 2$ and Cholesky (`cholesky_ex` + `cholesky_solve`) for $3 \le d \le 8$, wrapped in a convex Born–Infeld cone $\sigma_m(\sqrt{1 + D_{\text{PB}}} - 1)$ with worst-mode `mean + max` aggregation ($0.55\,\mathcal{L}_{\text{full}} + 0.45\,\mathcal{L}_{\text{core}}$). At Nash equilibrium $p_{\text{fake}} = p_{\text{real}}$, both $\Sigma_f = \Sigma_r$ and $\mu_f = \mu_r$, so both value and gradient vanish to exact zero.

**4. Symmetric BiGAN Joint Support ([`loss.py`](../../../particlegan/_nash_v3/loss.py)):** When `x_real.requires_grad` (joint BiGAN `[x, E(x)]` vs `[G(z), z]`), `NashV3Loss` applies Born–Infeld debiased transport and pullback covariance isometry symmetrically to the encoder branch as well.

### 2.3 Exact 1D Quantile Monge, Median-Cliff Fix & Primary-Probe Equilibrium Band ([`transport.py`](../../../particlegan/_nash_v3/transport.py), [`record.py`](../../../particlegan/_nash_v3/record.py), [`optim.py`](../../../particlegan/_nash_v3/optim.py))
- **Exact $O(B\log B)$ 1D Quantile Monge (`dim == 1`):** Replaces entropic Sinkhorn barycentric shrinkage in 1D with exact order-statistic quantile matching (`quantile_1d_monge_displacement`) and full tail scale $\tau_{\text{BI}} = 1.0\,\sigma_{\text{ref}}$, driving `std_ratio` from `0.94` to **`0.980–0.993`** and clearing all four 1D/precision gates (`fp32`, `mxfp8`, `nvfp4`, `gaussian1d_smoke` + `gaussian1d_stability`).
- **2-Mode Median-Cliff Elimination:** Replaces raw `cost_rr.median()` with $\max(\operatorname{median}(d^2_{rr}),\, \sum_j \operatorname{Var}(x_{\text{real},j}))$, eliminating the $30\times$ scale collapse on equal-weight 2-mode mixtures (`model_vae_categorical`, `model_vae_hard`).
- **Primary Pushforward Probe & Settled Equilibrium Band:** `_nash_v3_generator_forward_hook` preserves the first/primary $G(z_{\text{prior}})$ probe per step across coupled AE/VAE reconstruction forwards, while `NashV3GeneratorAdam` enforces the realized pushforward ceiling $\delta_{G,\max} = (0.020 + 0.23\min(1.5, r_{\text{OT}}))\gamma_G s_{\text{step}}\,\text{therm}_G$ across all coupled network parameters.

---

## 3. Leaderboard 1: `t4_imbalance_nash_stretch` (`3 / 3` = `100%` Sustained `5/5` `PASS`)

Evaluated under **protocol seed 0**, `deterministic_orthogonal` initialization, clean live `MoGParticlePrior` sampling (`4096` eval draws, `24` scheduled checks, **sustained `5/5` terminal-check requirement**), and one unchanged global recipe on `NVIDIA RTX A6000`:

| Candidate (All **0 SVD**; `v3` also **0 `autograd.grad`**) | Role / LR Ratio | `nash_overpowered_critic_ring16` ($D \gg G$, `621×`) | `nash_overpowered_generator_ring16` ($G \gg D$, `548×`) | `nash_unequal_width_hostile_imbalance` ($D \gg G$, $\sigma \in [.07,.30]$) | Sustained 5/5 PASS | Readout Receipt |
| --- | --- | --- | --- | --- | :---: | --- |
| **`t4-nash-v3`** | **Global `t4_nash_v3` (`1.25:1`)** | **PASS 5/5** *(`16/16`, `hq=0.8696`, `tv=0.0696`, `sw1=0.0585`, `core_cov=0.8021`, `core_eig=0.5717`)* | **PASS 5/5** *(`16/16`, `hq=0.8298`, `tv=0.0627`, `sw1=0.0654`, `core_cov=1.0293`, **`core_eig=0.3892`**)* | **PASS 5/5** *(`min_mr=0.8281`, `hq=0.9802`, `tv=0.0771`, `sw1=0.0968`, `core_cov=0.3150`, `core_eig=0.8295`)* | **3 / 3 (`100%`)** | [`readout-6f996ec2`](../records/readout-6f996ec256a149b2f1620f64.json) |
| **`t4-nash-v3-d10x`** | **10:1 Critic LR** (`d_lr_mult=10.0`) | FAIL *(4/5 checks; `16/16`, `hq=0.8735`, `core_cov=0.9270`, `core_eig=0.5713`)* | **PASS 5/5** *(`16/16`, `hq=0.8630`, `tv=0.0679`, `sw1=0.0621`, `core_cov=0.8601`, **`core_eig=0.4345`**)* | FAIL *(`hq=0.9866`, `core_cov=0.1803`, `core_eig=0.7411`, `tv=0.1960` vs `.18`, `min_mr=0.4541`)* | **1 / 3** *(G>>D 5/5 PASS!)* | [`readout-84d832df`](../records/readout-84d832df34d5eb3f04f8113f.json) |
| **`t4-nash-v3-g10x`** | **1:10 Critic LR** (`d_lr_mult=0.10`) | FAIL *(`16/16`, `hq=0.8315`, `tv=0.0774`, `sw1=0.0350`, `core_cov=1.0235`, `core_eig=0.3194`)* | FAIL *(2/5 checks; `16/16`, `hq=0.7615`, `tv=0.0610`, `core_cov=0.9252`, **`core_eig=0.3331`**)* | FAIL *(`min_mr=0.7256`, `hq=0.9817`, `tv=0.1677`, `core_cov=0.1664`, `core_eig=0.7552`, `sw1=0.1916`)* | **0 / 3** *(16/16 & eig>=0.32 on all 3!)* | [`readout-696ab5f8`](../records/readout-696ab5f8318bc8b356af8217.json) |
| `t4-nash-v2` *(PR #571)* | Prior `t4_nash` (`1.25:1`) | **PASS 5/5** *(`16/16`, `hq=0.848`, `core_cov=0.744`, `core_eig=0.429`)* | FAIL *(`15/16` @1600, `hq=0.745`, `core_cov=1.112`, `core_eig=0.075`)* | **PASS 5/5** *(`min_mr=0.944`, `hq=0.957`, `core_cov=0.275`)* | 2 / 3 | [`readout-0a06aa66`](../records/readout-0a06aa66bdb37bbb9f4ec2c1.json) |
| `t4-nash-loss-only-adam` | Causal ablation (plain Adam) | FAIL *(`1/16`, `hq=0.123`)* | FAIL *(`3/16`, `hq=0.099`)* | FAIL *(`min_mr=0.000`, `hq=0.385`)* | 0 / 3 | [`readout-53cd231b`](../records/readout-53cd231b9980a87ce588ee55.json) |
| `t4-ctrl-{bcap,empf,r1r2,rhbi}-adam` | 4 Naive-Adam baselines | 0/4 PASS *(`2–14/16`)* | 0/4 PASS *(`0–6/16`)* | 0/4 PASS *(`hq=0.000–0.149`)* | 0 / 3 | [see PR #571](t4-imbalance-nash.md#t4-imbalance-nash-stretch) |

**Key Takeaways:**
- `t4-nash-v3` raises `nash_overpowered_generator_ring16` `core_min_eigen_ratio` by **$5.19\times$** over `t4-nash-v2` (`0.075 -> 0.3892 >= 0.15`), unlocking **`3 / 3` (`100%`) sustained `5/5` PASS** under one unchanged global recipe with strictly `0` SVD, `0` NS, and `0` `autograd.grad`.
- Across the entire **$100\times$ bidirectional LR-ratio sweep** (`d_lr_mult` from `10.0` down to `0.10`, where `g10x` on $G \gg D$ compounds $>24{,}000\times$ combined asymmetry), `t4-nash-v3` recovers **`16/16` modes** on every ring and sustains **`core_min_eigen_ratio >= 0.319`** (`>2.1×` above the `0.15` gate!) on all 9 candidate–task cells, upgrading `d10x` on $G \gg D$ from `3/5` FAIL (`0.132`) to **sustained `5/5` PASS (`0.4345`)**.

---

## 4. Leaderboard 2: `discriminator_stability` Tiers 1–2 (`11 / 11` Required Tier 1 = `100%`, `33 / 46` Overall)

Evaluated with the **exact same unchanged global `t4-nash-v3` recipe** (`seed = 0`, `NVIDIA RTX A6000`, receipt [`readout-1ca76c61`](../records/readout-1ca76c61c875ba5a33dca488.json)):

### 4.1 Tier 1 Complete Table (`11 / 11` Required `PASS`, `16 / 17` Total Tier 1 `PASS`)

| Tier 1 Task | Importance | `empf-v8` (SVD/NS) | `t4-nash-v2` (0 SVD) | **`t4-nash-v3` (0 SVD, 0 `autograd.grad`)** | Verbatim `t4-nash-v3` Metrics |
| --- | :---: | :---: | :---: | :---: | --- |
| `clockfree_audit_measurement_v1` | diagnostic | PASS | PASS | **PASS** | `4/4` horizon, cadence & fresh-process restart parity |
| `precision_gaussian_fp32` | diagnostic | PASS | FAIL | **PASS** | `cdf_ks=0.0258`, `std_ratio=0.9934` |
| `precision_gaussian_mxfp8` | stretch | FAIL | FAIL | **PASS** | `cdf_ks=0.0357`, `std_ratio=0.9888` |
| `precision_gaussian_nvfp4` | stretch | FAIL | FAIL | **PASS** | `cdf_ks=0.0407`, `std_ratio=0.9803` |
| `t3_iso_scale_mass_8x` | stretch | — | — | **PASS** | `4/4`, `hq=1.0000`, `tv=0.0154`, `core_cov=0.1393`, `core_min_eig=0.7804` |
| `interp_sparse_arc` | stretch | FAIL | FAIL | FAIL | Shared open stretch (discontinuous manifold interpolation) |
| `ae_gan_hold` | **required** | PASS | PASS | **PASS** | `hold=0.0063` |
| `five_word_joint_smoke` | **required** | PASS | PASS | **PASS** | `5/5`, `mass_tv=0.0189` |
| `gaussian1d_smoke` | **required** | PASS | FAIL | **PASS** | `cdf_ks=0.0342`, `std_ratio=0.9915` |
| `model_ae_gan_multimodal` | **required** | PASS | FAIL | **PASS** | `4/4`, `prec=0.9214`, `tv=0.0461`, `min_eig=1.2686`, `max_eig=3.7451`, `ctr=0.4494`, `recon=0.0087` |
| `model_ae_reconstruction` | **required** | PASS | PASS | **PASS** | `recon_mse=0.0060` |
| `model_vae_categorical` | **required** | PASS | FAIL | **PASS** | `recon_mse=0.0068` |
| `model_vae_hard` | **required** | PASS | FAIL | **PASS** | `recon_mse=0.0039` |
| `ring16_acquisition` | **required** | PASS | FAIL | **PASS** | `16/16`, `hq=0.9321`, `tv=0.0664`, `sw1=0.0523`, **`full_cov=0.6168 <= 0.85`**, `core_cov=0.4908`, `min_eig=0.7095` |
| `two_pole` | **required** | PASS | PASS | **PASS** | `grad_med=0.1788 <= 1.0` |
| `unused_token_hold` | **required** | PASS | PASS | **PASS** | `concept_move=0.9999`, `unused_hold=0.9886` |
| `zero_tanh_smoke` | **required** | PASS | PASS | **PASS** | `mean_abs_error=0.0040 <= 0.005` |
| **Tier 1 Required / Total** | — | **11/11** (`13/16`) | **6/11** (`7/16`) | **11/11 (`100%`) · 16/17 Total** | **`+5` Required Tier-1 PASS & `0` BLOCKED / `0` ERROR** |

### 4.2 Tier 2 Summary (`17 / 29` `PASS` — `10/16` Required, `4/6` Stretch, `3/7` Held)

- **Required Tier-2 PASS (`10/16`):** `cover_leftover`, `five_word_joint_hold` (`5/5`, `tv=0.0209`), `gaussian1d_stability` (`cdf_ks=0.0265`, `std_ratio=0.9719`), **`img_stripes2` (`2/2`, `hq=1.0000`)**, `mid_scale_identity`, `residual_student`, `trajectory`, `unipolar`, `vector_spiral` (`sw1=0.0436`, `mean_err=0.0414`), `vector_two_broad` (`hq=0.9856`, `cov=0.1884`, `min_eig=0.8427`).
- **Stretch Tier-2 PASS (`4/6` — `100%` of T3 + T4 stretches!):** `t3_iso_horizon_saddle_shift` (`4/4`, `hq=0.9922`, `core_cov=0.3638`), `nash_overpowered_critic_ring16` (`PASS 5/5`), `nash_overpowered_generator_ring16` (`PASS 5/5`), `nash_unequal_width_hostile_imbalance` (`PASS 5/5`).
- **Held Tier-2 PASS (`3/7`):** `vector_anisotropic` (`hq=0.9951`, `cov=0.0917`), `vector_overlap` (`sw1=0.0536`), `vector_unequal_width` (`hq=0.9817`, `cov=0.1901`).
- **Honest Failure Analysis of Remaining Tier-2 Fails (`6` required, `2` interp stretches, `4` held):**
  1. **100-mode dense grids (`grid100`, `rotated100`, `staggered100`, `precision=0.28–0.39`):** Pure-elementwise Marchenko–Pastur scaling lacks layerwise polar singular-spectrum whitening needed to resolve 100 tightly packed grid cells in a single continuous MLP without chart partitioning; `mass_tv` is already sharp (`0.062–0.093`).
  2. **Denoising-diffusion chains (`model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`) & `mode_hold` (`7/8` modes, `hq=0.8655`):** Multi-step conditional DDGAN denoising chains require conditional time-step spectral conditioning (addressed in Track 2) rather than unconditional spatial Voronoi covering.
  3. **Held multi-mode images (`img_bars4` `4/4 hq=0.9375`, `img_intensity2`, `img_blobs4`) & `vector_unequal_mass` (`hq=0.9829`, `tv=0.0471`):** High-D image critic micro-secant skips low-D pullback covariance by design ($d > 8$), leaving multi-mode image spectral separation to polar/whitened conv updates.

---

## 5. Leaderboard 3: Minimal-Energy & VRAM Benchmark (`dcgan_performance`, `6 / 6` `PASS`, `39.68 ms/step` `r256`)

Measured via official Forge `dcgan_performance` (`10` timed updates after warmup; `finite_update_fraction = 1.0` on all 6 tasks; receipt [`readout-642d4c55`](../records/readout-642d4c55d093b44ed1e0309e.json)):

| Benchmark Task | Config | `empf-v8` (`ms` · `MiB`) | `rhbi-v12` (`ms` · `MiB`) | `rgdt-v2` (`ms` · `MiB`) | `t4-nash-v1` (`ms` · `MiB`) | **`t4-nash-v3` (`ms` · `p95` · `img/s` · `MiB`)** | Speedup vs `empf-v8` / `t4-nash-v1` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dcgan_performance_r32` | `32²`, B16, W64 | `23.94` · `168.7` | `47.60` · `98.6` | `24.43` · `227.3` | `25.99` · `114.6` | **`21.19 ms` · `21.69` · `764.6/s` · `82.0`** | **1.13× / 1.23× · 2.06× leaner VRAM** |
| `dcgan_performance_r64` | `64²`, B16, W64 | `32.63` · `336.6` | `83.02` · `692.7` | `33.83` · `314.8` | `32.13` · `303.7` | **`26.69 ms` · `30.06` · `590.6/s` · `216.3`** | **1.22× / 1.20× · 1.56× leaner VRAM** |
| `dcgan_performance_b64` | `64²`, B64, W64 | `38.48` · `488.0` | `141.12` · `873.7` | `39.37` · `404.9` | `35.55` · `396.2` | **`22.67 ms` · `26.03` · `2753.8/s` · `389.1`** | **1.70× / 1.57×** |
| `dcgan_performance_w128` | `64²`, B16, W128 | `63.18` · `923.0` | `177.94` · `685.9` | `62.64` · `859.8` | `42.18` · `1029.7` | **`27.95 ms` · `32.34` · `565.1/s` · `595.2`** | **2.26× / 1.51× · 1.55× leaner VRAM** |
| `dcgan_performance_r128` | `128²`, B16, W64 | `65.92` · `660.9` | `200.86` · `1248.5` | `61.32` · `584.7` | `44.26` · `675.0` | **`28.82 ms` · `38.00` · `519.3/s` · `551.3`** | **2.29× / 1.54×** |
| `dcgan_performance_r256` | `256²`, B16, W64 | `155.41` · `1531.4` | `588.75` · `3750.6` | `129.81` · `1471.4` | `99.35` · `1512.5` | **`39.68 ms` · `40.91` · `402.7/s` · `1372.9`** | **3.92× faster than `empf-v8` · 2.50× faster than `t4-nash-v1`!** |

---

## 6. Tier-4 Multimodal BiGAN Diagnosis (`ring16_bigan_continuous` / `Ring16BiGANFixture`)

Because `ring16_bigan_continuous` is a held Tier-4 task whose Forge adapter currently rejects its vector target declaration (`experiments/forge/vectorprofiles.py:128`), we diagnosed it directly via its canonical public-API fixture `Ring16BiGANFixture` ([`benchmarks/toy_audit/api_ring16.py`](../../../benchmarks/toy_audit/api_ring16.py), `BIGAN_CASE_ID`, protocol `seed = 0`, `4000` steps, `NVIDIA RTX A6000`), comparing frozen `t4_nash_v3` vs default `empf`:

| Recipe (Seed 0, 4000 steps) | Step 1000 | Step 2000 | Step 3000 | **Final Step 4000** | Wall Time |
| --- | --- | --- | --- | --- | ---: |
| **`t4_nash_v3` (0 SVD, 0 `autograd.grad`)** | **PASS** (`16/16`, `hq=0.8762`, `tv=0.0656`, `recon_rmse=0.0997`, `cons=0.9995`) | FAIL (`recon_rmse=0.1211`, `hq=0.9480`, `cons=1.0000`) | FAIL (`recon_rmse=0.1292`, `hq=0.9214`, `cons=1.0000`) | **PASS (`16/16`, `hq=0.9829`, `tv=0.0614`, `recon_rmse=0.0583`, `mode_consistency=1.0000`)** | `242.9 s` |
| `empf` (DualNorm SVD/NS baseline) | FAIL (`16/16`, `hq=0.8940`, `recon_rmse=0.1936`, `cons=0.9998`) | FAIL (`recon_rmse=0.1970`, `hq=0.9268`, `cons=0.9990`) | FAIL (`recon_rmse=0.1960`, `hq=0.9250`, `cons=0.9963`) | FAIL (`16/16`, `hq=0.9211`, `tv=0.0534`, `recon_rmse=0.1905`, `mode_consistency=0.9932`) | `243.7 s` |

**Root-Cause Finding:** `empf` fails `Ring16BiGANFixture` across all 4 checkpoints (`recon_rmse = 0.1905–0.1970`) because its joint critic signal alone does not preserve fine within-mode covariance on the encoder pullback branch $[x, E(x)]$. `t4_nash_v3`'s symmetric joint BiGAN branch support (`x_r.requires_grad` in `NashV3Loss.g_loss`) cuts `recon_rmse` by **$3.27\times$** (`0.1905 -> 0.0583`), lifts generation quality to **`hq = 0.9829`** and **`mode_consistency = 1.0000`**, and passes all fixture gates at step 4000 (`passed = true`, `failed_bounds = []`).

---

## 7. Recommendations & Reproduction

1. **Keep `DEFAULT_RECIPE = "empf"` unchanged** while exposing `get_recipe("t4_nash_v3")` (`particlegan.nash_v3`) as the repository's pure-elementwise (`0` SVD, `0` `eigh`, `0` `pinv`, `0` NS, `0` `autograd.grad`), `3/3` imbalance-invariant, `11/11` Tier-1, sub-40 ms `256²` DCGAN formulation.
2. **Next Synthesis Step (Tier-2 100-Mode Grids & Multi-Mode Images):** Combine `t4_nash_v3`'s 0-`autograd.grad` `fwAD`/micro-secant critic, Voronoi+core Burg pullback covariance isometry, and realized pushforward secant bands with low-rank/chart-partitioned generator updates (`grid100`/`rotated100`/`staggered100`) and conditional DDGAN time-embedding regularization (Track 2) to push pure-elementwise qualification through Tier 2.

```bash
# 1. Verify strict 0-SVD/0-eigh/0-pinv/0-NS/0-autograd.grad, fresh-process restart, tamper, and clockfree tests:
pytest tests/test_nash_v3.py tests/test_t4_nash.py -q

# 2. Validate all Forge views, ideas, studies, history, and compiled reports:
python3 -m experiments.forge validate
python3 -m experiments.forge history --check
python3 -m experiments.forge compile --check
python3 ops/refresh_forge_readouts.py --check --reports-only

# 3. Reproduce any official t4-nash-v3 Forge study on an isolated queue:
export PARTICLEGAN_FORGE_QUEUE=/nvme/dev/hypergan/ParticleGAN/runs/forge/tmp/nash_v3_repro
python3 -m experiments.forge run t4-nash-v3 --study t4-nash-v3-stretch-study --gpus 0,1 --cuda-model "NVIDIA RTX A6000" 2>&1 | tee "$PARTICLEGAN_FORGE_QUEUE/stretch.log"
python3 -m experiments.forge logs --follow
```
