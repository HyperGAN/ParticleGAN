# Round-3 Track 1: Solving Conditional Reverse Diffusion Chains (`cond_fiber` / `cond-fiber-v1`)

> **Executive Summary**
> Across all 10 historical leaderboard families and all Round-1/Round-2 PRs, **`0` recipes passed `model_ddgan_gaussian_chain`** (blocking `model_ddgan_bimodal_chain`) or **`model_ae_ddgan_gaussian_chain`**. Even `unified-v3` (`readout-ca2ce5acd5856e8c69b00377`) passed 36/38 bounds on `model_ddgan_gaussian_chain` and 40/42 on `model_ae_ddgan_gaussian_chain`, failing solely on the outer conditioning tail `t=1, q_0`.
>
> Introducing **`CondFiber` (`preset="cond_fiber"`, `critic_formulation="cond_fiber"`)** — a strictly clock-free, swap-invariant, fully checkpointed **cross-batch per-stratum sufficient-statistics conditional operator bank + midpoint-canceling paired-context covariance FIFO ring** composing cleanly on `UnifiedGANLoss` with **zero edits to `particlegan/_unified/`**, **zero `torch.autograd.grad` calls**, and **zero overhead on unconditional tasks** — achieves under one unchanged global recipe (`cond-fiber-v1`, revision `87872c20bdcd`, protocol seed `0`):
>
> - **FIRST-EVER official Forge `PASS` on `model_ddgan_gaussian_chain`**: **38/38 bounds PASS**, **9 consecutive terminal passes**, `confirmed_step = 2084`, reducing `t1_q0_normalized_mean` **2.09x** (`0.6427 -> 0.3076`), `t1_q0_covariance_eigen_max` (`1.6122 -> 1.2447 <= 1.40`), and `t1_q0_projection_cdf` (`0.1975 -> 0.1050 <= 0.18`).
> - **FIRST-EVER unlocked & official Forge `PASS` on `model_ddgan_bimodal_chain`**: **8 consecutive terminal passes**, `confirmed_step = 4375`, `worst_transition_cdf = 0.2034`, `final_normalized_mean = 0.1929`, `final_covariance_eigen_max = 1.0293`.
> - **FIRST-EVER `4/4` consecutive `42/42` bounds PASS on `model_ae_ddgan_gaussian_chain`** (`steps 2188..2500`; final `t1_q0_normalized_mean = 0.5740` vs `unified-v3` `0.9964`, `t1_q0_covariance_eigen_max = 1.5549`, `t1_q0_projection_cdf = 0.2502`, `final_normalized_mean = 0.0170`, `paired_nmse = 0.0501`, `paired_worst_axis_nrmse = 0.2290`, `paired_shuffled_gain = 39.00x`), faithfully graded `FAIL` by 1 check on Forge's 5-check convergence suffix (`step 2084 t1_q0_normalized_mean = 0.6863` vs `0.65`).
> - **40/47 PASS (`0 ERROR`, `0 TIMEOUT`, `0 BLOCKED`) on `discriminator_stability` Tiers 1–2** (`readout-5932f4d351cc4e791417835a.json`), preserving **11/11 required + 2/2 diagnostic Tier-1 PASS** (`15/17` Tier-1 total incl. `2/4` stretch) and **7/7 Tier-2 held PASS** (`25/30` Tier-2 total; +6 over `unified-v3` `34/41` and `rhbi-v12` `34/41`, +7 over `t4-nash-v3` `33/46`, +16 over `empf-v8` `24/41`).
> - **6/6 PASS on `dcgan_performance`** (`readout-8d2e547ce820cf8662a34bdf.json`), `r256 update_ms_median = 87.50 ms/step` (`p95 = 88.49 ms/step`, `182.7 img/s`, `1308.4 MiB` allocated / `1746.0 MiB` reserved), well under the `95 ms/step` target.

---

## 1. Root-Cause Diagnosis of `t=1, q_0` Failure in Prior Families

In `model_ddgan_gaussian_chain` and `model_ae_ddgan_gaussian_chain`, a single conditional generator $G_\theta(x_t, z, t)$ learns all $T=4$ reverse diffusion transitions $p(x_{t-1} \mid x_t, t)$ from minibatches of size $B=64$ stratified uniformly over $t \in \{1,2,3,4\}$ ($\approx 16$ samples per stratum per step).

Diagnostic profiling of `unified-v3` revealed four compounding small-sample statistical pathologies isolated to the outer evaluation quantile `t=1, q_0` ($x_1 = (-0.50, +0.25)$, Mahalanobis radius $\|u\| = 2.434$, outer $5\%$ radial tail, $\approx 0.8$ samples per minibatch):

1. **Finite-sample tail leverage starvation**: Single-batch ridge/OLS regression on $n_t \approx 16$ samples either overfits or heavily shrinks the slope in the outer $5\%$ tail, while hollow context kernels $k_c(c_i, c_j)$ pair rare tail points with central bulk neighbors where error $\approx 0$.
2. **Piecewise-linear LeakyReLU kink & 2D diagonal corner error**: Profiling $G(x_t, z, t=1)$ across a 2D context grid showed a localized LeakyReLU slope kink for $x_{t,0} < -0.20$ and a diagonal bilinear interaction $u_0 u_1$ at the $(-, +)$ corner `q_0` that linear/additive bases miss.
3. **U-shaped conditional variance bowl**: While bulk conditional standard deviation matched the true target ($\approx 0.268$), outer tail variance inflated into a U-shaped bowl (`eig_max = 1.61–1.65`), which global residual covariance scaling cannot resolve because bulk and tail errors have opposite signs.
4. **Auxiliary AE reconstruction gradient competition**: On `model_ae_ddgan_gaussian_chain`, naively summing new conditional operator terms on top of full-weight IMQ-CMMD doubled the conditional GAN gradient budget relative to the paired autoencoder reconstruction objective (`ae_loss`), inflating `paired_worst_axis_nrmse` above `0.25`.

---

## 2. Formulation & Mathematical Derivation (`CondFiber`)

`CondFiber` is implemented in `particlegan/_cond_fiber/` (`bank.py`, `critic.py`, `loss.py`, `penalty.py`) and exported via `particlegan/cond_fiber.py` (`preset="cond_fiber"`, `critic_formulation="cond_fiber"`). It subclasses `UnifiedGANLoss`, `UnifiedCriticAdam`, and `UnifiedStepRecord` with **zero modifications to `particlegan/_unified/`**, zero new `Recipe` dataclass fields, and zero `torch.autograd.grad` calls.

### 2.1 Strictly Pre-Update Cross-Batch Operator Bank (`CondFiberBank`)

For each discrete stratum $t$ (partitioned deterministically on CPU), `CondFiberBank` maintains strictly pre-update float64 sufficient statistics and a bounded float32 paired-context FIFO ring ($M_{\max}=384$), queried **before** updating on the current batch (eliminating same-batch leakage and variance collapse):

1. **Exact Parallel-Welford Context Standardization**:
   Maintains $(n_{\text{eff}}, \mu_c, m_{2,c})$ under clock-free forgetting factor $\gamma_s \in [\gamma_{\min}, \gamma_{\text{base}}] = [0.50, 0.96]$, yielding standardized contexts $u = (c - \mu_c) / \sigma_c$ and shared context leverage weight:
   $$\omega(c) = 1 + 0.90 \cdot \min\!\left(\frac{\|u\|_2^2}{d_c},\; 4.5\right).$$
   Because $\omega(c)$ depends exclusively on the shared context $c$ (never on $X$ or $Y$), it preserves exact swap invariance and exact zero bias at equilibrium $P(X \mid c) = P(Y \mid c)$.

2. **Piecewise-Hinge + Bounded Bilinear Feature Basis**:
   For $d_c \le 6$ ($p = 1 + 3 d_c + \binom{d_c}{2}$, so $p=8$ for $d_c=2$):
   $$\phi(c) = \left[1,\; u,\; 2(\operatorname{softplus}(u - 0.8) - 0.425),\; 2(\operatorname{softplus}(-u - 0.8) - 0.425),\; \frac{u_a u_b}{\sqrt{1 + 0.15(u_a^2 + u_b^2)}}\right]_{a < b}.$$
   The unit-scaled two-sided softplus hinges capture outer-tail LeakyReLU slope kinks with $O(\|u\|)$ asymptotic growth, while the bounded pairwise bilinear terms resolve 2D diagonal corner biases without quadratic blowup.

3. **Dual-Timescale Symmetric WLS Mean Operator & Residual Whitening**:
   Accumulates slow ($\gamma_{\text{base}} = 0.96$, $\sim 400$ samples/stratum) and fast ($\gamma_{\text{fast}} = 0.85$, $\sim 105$ samples/stratum) weighted Gram and cross-moments $(G_{\phi\phi}, G_{\phi X}, G_{\phi Y})$ with anisotropic Tikhonov ridge $\Lambda = \operatorname{diag}(\lambda_{\text{lin}} = 10^{-3}, \lambda_{\text{nl}} = 0.025)$:
   $$B_X^{\text{slow}}, B_Y^{\text{slow}} = \left(\frac{G_{\phi\phi}}{W_{\text{eff}}} + \Lambda\right)^{-1} \frac{[G_{\phi X}, G_{\phi Y}]}{W_{\text{eff}}}, \qquad B_{\text{pool}} = \frac{1}{2}(B_X^{\text{slow}} + B_Y^{\text{slow}}).$$
   - **Swap-invariant pooled drift**: $m_{\text{pool}}(c) = \phi(c) B_{\text{pool}}$ removes cross-context mean drift before multi-scale hollow IMQ-CMMD with exact pair-swap invariance.
   - **Swap-invariant residual whitening**: $W_t = \left(\frac{1}{2}(\Sigma_{XX} + \Sigma_{YY}) + \epsilon I\right)^{-1/2}$ from exact symmetric float64 eigendecomposition.
   - **Critically-damped mean discrepancy**: Combines $85\%$ fast-stream tracking $\Delta m_{\text{bank}}(c) = \phi(c)(B_X^{\text{fast}} - B_Y^{\text{slow}})$ (preventing 25-step Hopf delay oscillation on moving $G_\theta$) with $15\%$ zero-lag exact slow-Gram leave-one-out residual $\Delta m_{\text{LOO}}(c_i)$ (instant damping, zero self-attraction), smoothly odd-saturated:
     $$\widetilde{\Delta m}_W(c_i) = \tau_m \tanh\!\left(\frac{\|\Delta m_i W_t\|_2}{\tau_m}\right) \frac{\Delta m_i W_t}{\|\Delta m_i W_t\|_2}, \qquad \tau_m = 3.0.$$

4. **Midpoint-Canceling Nonparametric Paired-Context FIFO Ring (Local Conditional Covariance)**:
   From strictly past paired contexts $\{(c_j, X_j, Y_j)\}_{j=1}^{M}$ ($M \le 384$), compute recency-weighted adaptive $k$-NN ($k=20$) kernel weights $\alpha_{ij} \propto \exp\!\left(-\frac{\|u_i - u_j\|^2}{2 h_i^2} - \frac{\text{age}_j}{128}\right)$ and pooled-midpoint residuals $r_{X,j} = X_j - m_{\text{pool}}(c_j)$, $r_{Y,j} = Y_j - m_{\text{pool}}(c_j)$:
   $$E_{\text{cov}}^{\text{loc}}(c_i) = \sum_{j=1}^M \alpha_{ij}\left(r_{X,j} r_{X,j}^\top - r_{Y,j} r_{Y,j}^\top\right).$$
   **Exact Midpoint Cancellation Theorem**: Writing $m_{\text{pool}} = \frac{1}{2}(m_X + m_Y)$ and $\Delta = m_X - m_Y$,
   $$\mathbb{E}[r_X r_X^\top \mid c] = \Sigma_X(c) + \tfrac{1}{4}\Delta\Delta^\top, \qquad \mathbb{E}[r_Y r_Y^\top \mid c] = \Sigma_Y(c) + \tfrac{1}{4}\Delta\Delta^\top,$$
   so the mean-bias outer product $\tfrac{1}{4}\Delta\Delta^\top$ **cancels identically** in $r_X r_X^\top - r_Y r_Y^\top$, isolating pure local conditional covariance error $\Sigma_X(c_i) - \Sigma_Y(c_i)$ with exact swap antisymmetry and zero variance collapse.

5. **Split-Sign Bounded Covariance Matching**:
   Whitening $E_W(c_i) = \operatorname{sat}_F(W_t E_{\text{cov}}^{\text{loc}}(c_i) W_t; \tau_c = 1.5)$ and eigensplitting on CPU (15x faster than CUDA batched cuSOLVER `eigh` for small $d_x$) into $E_W = E_+ + E_-$ ($E_+ \succeq 0$, $E_- \preceq 0$):
   $$\mathcal{L}_{\text{cov}} = \frac{1}{B} \sum_{i=1}^B w_i \cdot \frac{1}{2 d_x}\left(r_{W,i}^\top E_{+,i} r_{W,i} + \tilde{r}_{W,i}^\top E_{-,i} \tilde{r}_{W,i}\right), \qquad \tilde{r}_{W,i} = \frac{r_{W,i}}{\sqrt{1 + 0.25 \|r_{W,i}\|^2 / d_x}}.$$
   Positive eigenvalues contract excess variance linearly, while negative eigenvalues expand under-dispersed modes with radially bounded gradients — preventing concave outlier blowup while maintaining exact zero first-order bias at $E_W = 0$.

6. **Variance-Preserving Conditional Budget Partition & Operator-Aware Settling**:
   Once the bank is warm, the conditional discrepancy partitions the total conditional energy budget cleanly relative to auxiliary objectives (such as `ae_loss`):
   $$\mathcal{L}_{\text{cond}} = 0.40\,\mathcal{L}_{\text{IMQ-CMMD}}(m_{\text{pool}}) + 0.36\,\mathcal{L}_{\text{mean}} + 0.30\,\mathcal{L}_{\text{cov}}.$$
   To prevent premature step-size cooling while outer-tail errors remain unresolved (without positive-rectification bias), the strictly clock-free settling residual adds parameter-space excess operator discrepancy:
   $$r_{\text{cond}} = r_{\text{IMQ}} + \min\!\left(0.5,\; 0.11 \max\!\left(0,\; \max_t \left(\frac{\|(B_X^{\text{slow}} - B_Y^{\text{slow}}) W_t\|_F^2}{d_x} - \frac{2p}{n_{\text{eff}}}\right) - 0.10\right)\right),$$
   driving the symmetric DualNorm step scale $\text{step\_scale} = 1 - 0.85 \cdot \operatorname{smoothstep}_{[0.026, 0.050]}(\text{EMA}_{0.99}(r_{\text{cond}}))$.

7. **Swap-Invariant Innovation Reset**:
   Clock-free forgetting modulation $\gamma = \gamma_{\min} + (\gamma_0 - \gamma_{\min}) \exp(-\tfrac{1}{2}\max(0, z_{\text{innov}} - 2.5)^2)$ driven by $z_{\text{innov}} = \max(e_X, e_Y, c_{\text{jump}})$ rapidly adapts to non-stationary shifts without any step counter.

---

## 3. Official Forge Results & Leaderboard (`seed = 0`, One Global Recipe)

All evidence below is derived verbatim from concluded official Forge readouts on `NVIDIA RTX A6000` (`deterministic=True`, `tf32=False`, `initializer=deterministic_orthogonal`, `forge-rng-v1`, protocol seed `0`, candidate revision `87872c20bdcd`):

- **`discriminator_stability` Tiers 1–2 receipt**: [`readout-5932f4d351cc4e791417835a.json`](../records/readout-5932f4d351cc4e791417835a.json)
- **`dcgan_performance` receipt**: [`readout-8d2e547ce820cf8662a34bdf.json`](../records/readout-8d2e547ce820cf8662a34bdf.json)

### 3.1 Conditional Reverse Diffusion Chains — Cross-Family Comparison

| Candidate | Receipt | `disc_stab` T1–2 PASS | `ddgan_gaussian` Status | `ddgan` `t1_q0` Mean ($\le 0.40$) | `ddgan` `t1_q0` EigMax ($\le 1.40$) | `ddgan` `t1_q0` CDF ($\le 0.18$) | `ddgan_bimodal` Status | `ae_ddgan_gaussian` Status | `ae_ddgan` `t1_q0` Mean ($\le 0.65$) | `ae_ddgan` `t1_q0` EigMax ($\le 1.80$) | `ae_ddgan` Paired NMSE ($\le 0.08$) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cond-fiber-v1`** | `readout-5932f4d3` | **40/47** (`0` BLOCKED) | **PASS** (9/9 suffix, `s=2084`) | **0.3076** | **1.2447** | **0.1050** | **PASS** (8/8 suffix, `s=4375`) | FAIL (4/4 suffix `s=2188..2500`)* | **0.5740** | **1.5549** | **0.0501** |
| `unified-v3` | `readout-ca2ce5ac` | 34/41 | FAIL | 0.6427 | 1.6122 | 0.1975 | NOT_RUN | FAIL | 0.9964 | 1.6463 | 0.0450 |
| `rhbi-v12` | `readout-47591bd9` | 34/41 | FAIL | 0.5539 | 1.8245 | 0.2103 | NOT_RUN | FAIL | 0.6643 | 2.3426 | 0.0403 |
| `t4-nash-v3` | `readout-1ca76c61` | 33/46 | FAIL | 0.2926 | 0.9008 | 0.1926 | NOT_RUN | FAIL | 0.8661 | 2.1233 | 0.0515 |
| `empf-v8` | `readout-33f76281` | 24/41 | FAIL | 1.2986 | 1.0247 | 0.6396 | NOT_RUN | FAIL | 0.5693 | 1.1191 | 0.1041 |
| `rgdt-v3` | `readout-aec7fddd` | 23/41 | FAIL | 0.6817 | 1.5418 | 0.2665 | NOT_RUN | FAIL | 0.8708 | 1.8675 | 0.0392 |

\* On `model_ae_ddgan_gaussian_chain`, `cond-fiber-v1` passes **all 42/42 bounds** on steps `2188, 2292, 2396, 2500` (`4` consecutive checks; final `final_normalized_mean = 0.0170`, `t1_q0_projection_cdf = 0.2502`, `paired_worst_axis_nrmse = 0.2290`, `paired_shuffled_gain = 39.00x`, `wall = 125.1s`). It is faithfully graded `FAIL` because step `2084` (`t1_q0_normalized_mean = 0.6863` vs `0.65`) was 1 observation short of Forge's `minimum_stable_checks = 5` convergence suffix, which also keeps `model_ae_ddgan_bimodal_chain` prerequisite-gated (`NOT_RUN`).

### 3.2 Full `discriminator_stability` Tiers 1–2 Breakdown (`40/47 PASS`, `0 BLOCKED`, `0 ERROR`)

- **Tier 1 Required (`11/11` PASS) + Diagnostic (`2/2` PASS) + Stretch (`2/4` PASS)** (`15/17` Tier-1 total):
  - Required PASS (`11/11`): `ae_gan_hold`, `five_word_joint_smoke`, `gaussian1d_smoke`, `model_ae_gan_multimodal`, `model_ae_reconstruction`, `model_vae_categorical`, `model_vae_hard`, `ring16_acquisition`, `two_pole`, `unused_token_hold`, `zero_tanh_smoke`.
  - Diagnostic PASS (`2/2`): `clockfree_audit_measurement_v1`, `precision_gaussian_fp32`.
  - Stretch PASS (`2/4`): `precision_gaussian_nvfp4`, `t3_iso_scale_mass_8x` (FAIL: `interp_sparse_arc`, `precision_gaussian_mxfp8`).
- **Tier 2 Required (`15/17` PASS) + Stretch (`3/6` PASS) + Held (`7/7` PASS)** (`25/30` Tier-2 total):
  - Required PASS (`15/17`): `cover_leftover`, `five_word_joint_hold`, `gaussian1d_stability`, `img_stripes2`, `mid_scale_identity`, `mode_hold`, `model_ddgan_bimodal_chain`, `model_ddgan_gaussian_chain`, `residual_student` (`success_rate = 1.0`, `identity_mse = 0.0006`), `rotated100`, `staggered100`, `trajectory` (`identity_mse = 0.0007`), `unipolar`, `vector_spiral`, `vector_two_broad` (FAIL: `grid100`, `model_ae_ddgan_gaussian_chain`; `model_ae_ddgan_bimodal_chain` prerequisite-gated `NOT_RUN`).
  - Stretch PASS (`3/6`): `nash_overpowered_generator_ring16`, `nash_unequal_width_hostile_imbalance`, `t3_iso_horizon_saddle_shift` (FAIL: `interp_img_patch_shift`, `interp_saddle_barycentric`, `nash_overpowered_critic_ring16`).
  - Held PASS (`7/7` — 100%): `img_bars4`, `img_blobs4`, `img_intensity2`, `vector_anisotropic`, `vector_overlap`, `vector_unequal_mass`, `vector_unequal_width`.
- **Faithful Numerical FAILs (`7/47`)**: `model_ae_ddgan_gaussian_chain` (4/4 terminal 42/42 passes, 1 check shy of 5-check suffix), `grid100`, `precision_gaussian_mxfp8`, `nash_overpowered_critic_ring16`, `interp_img_patch_shift`, `interp_saddle_barycentric`, `interp_sparse_arc` (the last 6 inherited verbatim from `UnifiedGANLoss`, since `CondFiberGANLoss` delegates all unconditional domains directly to `UnifiedGANLoss`).

### 3.3 `dcgan_performance` (`6/6 PASS`, `readout-8d2e547ce820cf8662a34bdf.json`)

| Task | Gate | Median (`ms/step`) | p95 (`ms/step`) | Throughput (`img/s`) | Peak Alloc (`MiB`) | Peak Reserved (`MiB`) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dcgan_performance_r32` | PASS | 26.85 | 29.37 | 589.5 | 230.5 | 564.0 |
| `dcgan_performance_r64` | PASS | 34.40 | 35.59 | 463.4 | 328.4 | 742.0 |
| `dcgan_performance_b64` | PASS | 30.05 | 30.90 | 2128.9 | 396.6 | 696.0 |
| `dcgan_performance_r128` | PASS | 49.90 | 51.19 | 319.5 | 634.2 | 1106.0 |
| `dcgan_performance_w128` | PASS | 56.75 | 58.01 | 282.6 | 874.0 | 1654.0 |
| **`dcgan_performance_r256`** | **PASS** | **87.50** | **88.49** | **182.7** | **1308.4** | **1746.0** |

---

## 4. Key Ablation Lessons (Exact Forge Contract)

1. **Variance-preserving conditional budget partition (`0.40 IMQ + 0.36 mean + 0.30 cov`)**: Unpartitioned `1.0 IMQ + 0.48 mean + 0.43 cov` doubled conditional gradient weight relative to `ae_loss`, inflating `paired_worst_axis_nrmse` from `0.224` to `0.265`. Partitioning the budget back to `~1.06` restored `paired_worst_axis_nrmse = 0.229` (`0` AE failures across steps `1563..2500`) while simultaneously taking `model_ddgan_gaussian_chain` to `9` consecutive 38/38 passes.
2. **Zero slow-$X$ lag in mean tracking**: Including slow-bank $B_X^{\text{slow}}$ (`gamma_base = 0.96`, 25-step lag) or flat boxcar ring residuals in $\Delta m_{\text{bank}}$ triggered ~200-step Hopf delay oscillations; keeping `gamma_fast = 0.85` + $15\%$ zero-lag slow-Gram LOO eliminated ringing completely.
3. **CPU eigensplitting & single-sync transfers**: Moving batched $d_x \times d_x$ `eigh` off CUDA cuSOLVER to CPU (`0.046 ms` vs `0.719 ms`, 15x faster), explicit `device="cpu"` allocation (preventing crashes under `torch.set_default_device("cuda")` on `trajectory`/`residual_student`), packing `(c, x, y)` into one D2H copy, and bypassing fallback `_zero_shrinkage_pooled_mean` once warmed cut DDGAN wall time from `218s` to `112s` and AE-DDGAN from `>180s` timeout to `125s`.

---

## 5. Verification, Contracts & Reproduction

- **Contract compliance**: `DEFAULT_RECIPE == "empf"`, zero edits to `particlegan/_unified/` or `particlegan/recipe_compat.py`, zero new `Recipe` fields, zero `torch.autograd.grad` calls, strictly clock-free, exact swap invariance, exact equilibrium unbiasedness, 6 checkpoint tamper rejections, and bit-exact fresh-process DDGAN continuation across active `CondFiberBank` boundaries (all verified in `tests/test_cond_fiber.py`).
- **Reproduce official Forge studies (with easy-to-tail logs)**:
  ```bash
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  QR=runs/forge/queues/cond-ddgan-repro
  LOG=runs/forge/tmp/cond-ddgan-repro.log
  python -m experiments.forge --queue-root "$QR" run cond-fiber-v1 \
    --study cond-fiber-v1-dcgan-perf-study --gpus 0 --cuda-model "NVIDIA RTX A6000" >> "$LOG" 2>&1
  python -m experiments.forge --queue-root "$QR" run cond-fiber-v1 \
    --study cond-fiber-v1-study --gpus 0,1 --cuda-model "NVIDIA RTX A6000" >> "$LOG" 2>&1
  tail -f "$LOG"
  ```
- **Recommendation**: Adopt `CondFiberBank` + midpoint-canceling paired-context covariance ring as the canonical conditional-fiber operator for `unified-v4+` (already shared via swarm handoff `#213`), and investigate accelerating AE encoder–generator outer-tail convergence by ~100 steps (e.g., joint encoder tail leverage) to move `model_ae_ddgan_gaussian_chain` from 4/4 to 5/5+ and unlock `model_ae_ddgan_bimodal_chain`.
