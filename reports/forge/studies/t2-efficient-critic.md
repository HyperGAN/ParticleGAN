# Track 2 Study: Energy-Efficient First-Principles Solution to Discriminator Sharpness (`t2_secant`)

## 1. Executive Summary

Modern Lipschitz- and Born-Infeld-regularized GAN critics (`BCap`, `BISPF`, `RHBI`, `EMPF`) bound discriminator sharpness $\|\nabla_x D(x)\|_2 \le \kappa$ by calling `torch.autograd.grad(..., create_graph=True)` across real, fake, and transport-chord batches and then backpropagating parameter gradients $\nabla_\theta$ through that higher-order graph. On $256 \times 256$ RGB convolutional critics (`dcgan_performance_r256`, NVIDIA RTX A6000), second-order double-backpropagation inflates step latency to **3,113.09–4,845.34 ms/step** (`BCap`), **588.62–599.88 ms/step** (`RHBI`, **6,320 MiB** reserved VRAM), **340.02–349.26 ms/step** (`BISPF`), and **130.13–173.05 ms/step** (`EMPF`).

In this study (`feat/solve-gans-t2-efficient-critic`, owned in `particlegan/_t2_critic/` and `particlegan/t2_critic.py`), we derive and implement **`t2_secant`** — a **strictly clock-free, purely 1st-order Born-Infeld critic formulation that calls `torch.autograd.grad` zero times**:

1. **Three Mathematical Failure Modes of Naive Finite-Difference Secants Identified & Proved**:
   - **Symmetric-Flank Chord Cancellation**: Macro secants $|D(b)-D(a)|/\|b-a\|_2$ vanish across equal-height mode bumps ($D(\mu+ru) \approx D(\mu-ru)$) even when local flank slopes $\|\nabla_x D\|_2 \gg \kappa$.
   - **High-Dimensional Concentration-of-Measure Projection Loss**: Along any single chord or isotropic unit probe $\hat{u} \in S^{d-1}$, $\mathbb{E}[(\nabla_x D \cdot \hat{u})^2] = \|\nabla_x D\|_2^2 / d$, underestimating slope by $\sqrt{d}$ ($13.04\times$ in 170D `five_word_joint_smoke`, allowing actual $\|\nabla_x D\|_2$ to spike to $2.45$).
   - **Piecewise-Linear Parameter-Jacobian Kink Impulses**: While ReLU/LeakyReLU $D_\theta(x)$ is continuous across activation boundaries $w^\top h + b = 0$, $\nabla_\theta D_\theta(x)$ jumps discontinuously; finite-difference subtraction $(\nabla_\theta D(x+\delta u)-\nabla_\theta D(x))/\delta$ injects $O(1/\delta)$ spurious parameter-gradient impulses whenever a neuron flips sign (`param_grad_cosine` drops to `0.22–0.32`).
2. **First-Principles 1st-Order Resolution (`t2-secant-v5`)**:
   - **Hybrid Forward-Mode Tangent Reconstruction (Manifold / MLP / Vector / Discrete-Simplex / BiGAN / VAE Critics)**: Propagates $(D(x), \nabla_x D(x)\cdot u)$ at the **exact primal activation gates** via `torch.autograd.forward_ad.dual_level()` (zero kink impulses, zero `create_graph=True` second-order graph, single standard 1st-order `.backward()`):
     - $1 \le d \le 4$: exact canonical orthonormal basis $I_d$ on `x_real` and `x_fake` (zero cross-mode neighbor bias inside modes) paired with 1-tangent Weyl-rotated Spatial-Frame forward AD along Sinkhorn transport chords (`chord_blend = 0.06`, smoothing empty inter-mode saddle bridges), plus symmetric $O(\delta^2)$ central-difference fallback (`fd_delta = 1e-3`) for custom emulated-precision ops without `jvp` rules.
     - $5 \le d \le 8$: zero-sync analytical chained Householder $O(d)$ orthonormal frames ($Q = H_2 H_1$) reconstructed across nearest basis-slot neighbors.
     - $d > 8$ non-image manifolds (e.g., 170D discrete-simplex BiGAN): RNG-free 32-bit Murmur3 Rademacher unit tangents + local $k$-NN Gaussian-kernel Hutchinson pooling ($k=32$, zero RNG stream consumption).
   - **Live-Endpoint Reuse + Separable Rademacher BF16 Micro-Secants + $8 \times 8$ Pooled Chamfer Scale (Convolutional Image Critics)**: Reuses the live unscaled 1st-order `D(x_real)` and `D(x_fake)` logits from `d_loss` at **zero extra endpoint forward cost**, evaluates **one** detached BF16 micro-secant probe batch alongside free transport-chord and cyclic roll secants, and tracks image scale via $8 \times 8$ spatially pooled Chamfer moments in $O(B^2)$ scalar time.
3. **Key Empirical Outcomes (Protocol Seed 0, 12 Concluded Forge Readout Receipts)**:
   - **DCGAN Energy / Latency / VRAM Speedup**: Passes **6/6** `dcgan_performance` tasks (`r32`–`r256`), cutting uncontended `dcgan_performance_r256` median update time to **87.96 ms/step** (**181.92 img/s**, **1,308.46 MiB** peak allocated / **1,756.00 MiB** reserved) — **1.77x faster** than published `empf-v8` (`155.41 ms`) and **1.48x–1.97x faster** than matched-host `empf-v8` (`130.13–173.05 ms`), **3.87x–3.97x faster** than `bispf` (`340.02–349.26 ms`), **6.69x–6.82x faster** than `rhbi` (`588.62–599.88 ms`, **-64.7% allocated / -72.2% reserved VRAM**), and **35.39x–55.09x faster** than `bcap` (`3,113.09–4,845.34 ms`).
   - **Discriminator Stability (`t2-secant-v5`, `readout-a2d41a678887a49f9ecc6aa9`)**: Clears **11/11 required Tier 1 gates (`100%`)** + `clockfree_audit_measurement_v1` (4/4 parity) + stretch `precision_gaussian_mxfp8` (`cdf_ks = 0.0353 <= 0.05`, where `empf-v8` fails) — including actual critic input-gradient bound `two_pole` `grad_med = 0.7568 <= 1.0`, `ring16_acquisition` `component_covariance_error = 0.4860 <= 0.85` (`hq = 0.9429`, `suffix = 33`, where `empf-v8` degrades to `3.1455 FAIL` under `torch 2.15.0+cu132`), `five_word_joint_smoke` (`modes = 5/5`, `p_min = 1.0000`), `model_ae_gan_multimodal` (`0.7782 <= 1.0`), and `model_vae_hard` (`suffix = 14`). Unlocks and executes all **25 Tier 2 tasks** with zero `ERROR`/`TIMEOUT`/`BLOCKED`, passing **8/25 Tier 2 tasks** (**21/41 total**), including **`five_word_joint_hold`** (`modes = 5/5`, `p_min = 1.0000`), `trajectory`, `residual_student`, `unipolar`, `cover_leftover`, `mid_scale_identity`, `vector_two_broad`, and `vector_spiral`, while improving `mass_tv` over `empf-v8` on `grid100` (`0.1782` vs `0.1879`), `rotated100` (`0.1392` vs `0.1405`), and `staggered100` (`0.1236` vs `0.1294`, 100/100 modes).

---

## 2. First-Principles Mathematical Derivation

### 2.1 Why Second-Order `create_graph=True` Blows Up Critic Energy and Memory

Let $F_D$ denote the FLOPs of one forward evaluation $D_\theta(x)$ on batch size $B$. Standard 1st-order critic training on `d_loss(D(x_real), D(x_fake))` requires 2 forwards + 1 parameter backward $\approx 6 F_D$.

By contrast, a 3-point second-order input-gradient regularizer (`EMPF`, `BISPF`, `RHBI`) evaluates $D_\theta$ on $(x_{\text{real}}, x_{\text{fake}}, x_{\text{chord}})$ (`3 F_D`), computes $\nabla_x D_\theta$ via `torch.autograd.grad(..., create_graph=True)` while retaining all intermediate forward and backward activation tensors (`+6 F_D`), and then backpropagates $\nabla_\theta \|\nabla_x D_\theta\|_2^2$ through the retained backward graph (`+6 F_D`), totaling **$\approx 15 F_D$–$21 F_D$** and forcing non-fused higher-order convolution backward kernels (`BCap` additionally triggers exact SVD / non-fused higher-order conv paths exceeding $3\text{ s/step}$).

### 2.2 Three Obstructions to Naive 1st-Order Secant Penalties

Replacing $\|\nabla_x D(x)\|_2$ naively by transport-chord or pairwise finite-difference secants

$$
S(a, b) = \frac{|D_\theta(a) - D_\theta(b)|}{\sqrt{\|a - b\|_2^2 + \varepsilon^2}}
$$

(`t2-secant-v1`, `readout-2d0a1240650dbb7b14eb74d1`) passed 6/11 required Tier 1 gates (`two_pole` `grad_med = 0.1962 <= 1.0`, `ring16_acquisition` `hq = 0.9321`) and sped up `r256` to `108.43 ms/step`, but exposed three fundamental mathematical obstructions:

1. **Theorem 1 (Symmetric-Flank Cancellation)**: Suppose $D_\theta(x) = \psi(\|x - \mu\|_2)$ is radially symmetric around a mode center $\mu$. For opposite flank points $a = \mu - r u$ and $b = \mu + r u$ ($\|u\|_2 = 1$), $S(a, b) = |\psi(r) - \psi(r)| / (2r) = 0$ identically, even as $|\psi'(r)| \to \infty$. Macro chords cannot detect symmetric mode spikes.
2. **Theorem 2 (High-Dimensional Projection Attenuation & Variance)**: For $\hat{u} \sim \mathcal{U}(S^{d-1})$ or normalized Rademacher $u \in \{\pm d^{-1/2}\}^d$, $\mathbb{E}[(\nabla_x D(x) \cdot u)^2] = \frac{1}{d}\|\nabla_x D(x)\|_2^2$. Unscaled secants underestimate $\|\nabla_x D\|_2$ by $\sqrt{d}$ ($13.04\times$ at $d = 170$ on `five_word_joint_smoke`). Conversely, multiplying a single probe ($K=1$) by $d$ yields kurtosis $\text{Var}[d(\nabla_x D \cdot u)^2] \approx 2 \|\nabla_x D\|_2^4$ (relative standard error $\sqrt{2} \approx 141\%$).
3. **Theorem 3 (Piecewise-Linear Parameter-Jacobian Kink Impulses)**: For a ReLU/LeakyReLU critic $D_\theta(x) = \sum_\ell \alpha_\ell(x)^\top x + \beta_\ell(x)$, $D_\theta$ is continuous across hyperplane kinks $H_j = \{x : w_j^\top h(x) + b_j = 0\}$, but $\nabla_\theta D_\theta(x)$ jumps across $H_j$. Whenever the segment $[x, x + \delta u]$ crosses $H_j$, the finite-difference parameter gradient

$$
\nabla_\theta \left[\frac{D_\theta(x + \delta u) - D_\theta(x)}{\delta}\right] = \frac{\nabla_\theta D_\theta(x + \delta u) - \nabla_\theta D_\theta(x)}{\delta}
$$

contains an **$O(1/\delta)$ impulse** from the mismatched gate indicator $\mathbb{I}[w_j^\top h(x+\delta u)+b_j > 0] \ne \mathbb{I}[w_j^\top h(x)+b_j > 0]$ (empirically degrading parameter-gradient cosine similarity against exact R1/Born-Infeld gradients to `0.22–0.32` at $\delta = 2 \times 10^{-3}$).

### 2.3 Spatial-Frame Forward-Mode Tangent Reconstruction & Live-Endpoint BF16 Envelope

**Eliminating Kink Impulses via Forward-Mode AD (`fwAD.dual_level`)**: Evaluating $D_\theta$ on the dual number $x + \epsilon u$ ($\epsilon^2 = 0$) propagates $(h, \dot{h})$ through the **exact primal activation mask** $\sigma'(z) \odot \dot{z}$ in a single forward pass. The resulting directional derivative $\tau(x, u) = \nabla_x D_\theta(x) \cdot u$ is an ordinary 1st-order tensor in $\theta$: a single standard `.backward()` differentiates $\tau(x, u)^2$ w.r.t. $\theta$ **without** `autograd.grad` and **without** $O(1/\delta)$ kink impulses (`param_grad_cosine = 0.999–1.000` for $d \le 8$).

**Eliminating Projection Variance Across Dimensions (`_T2SecantPenaltyKernel`)**:
- **Low-Dimensional Manifolds ($1 \le d \le 4$)**: Inside populated clusters (`x_real`, `x_fake`), exact canonical orthonormal tangents $I_d = [e_1, \dots, e_d]$ reconstruct $\|\nabla_x D(x)\|_2^2 = \sum_{m=1}^d (\nabla_x D(x) \cdot e_m)^2$ with **zero directional variance and zero cross-mode neighbor bias** (resolving 4-quadrant `model_ae_gan_multimodal`). Along Sinkhorn transport chords $x_{\text{chord}} = (1-\alpha)x_{\text{fake}} + \alpha p_{\pi}$, a 1-tangent Weyl-rotated Spatial-Frame $Q(\phi) \in O(d)$ pairs opposite-slot spatial neighbors (`chord_blend = 0.06`), smoothing saddle ridges across empty inter-mode bridges (resolving 16-mode `ring16_acquisition`). Custom emulated-precision layers without `jvp` rules automatically use the symmetric central-difference fallback $(D(x+\delta u) - D(x-\delta u)) / (2\delta)$ at $\delta = 10^{-3}$ ($O(\delta^2)$ curvature cancellation).
- **Mid-Dimensional Manifolds ($5 \le d \le 8$)**: Two chained analytical Householder reflections $Q(\phi) = (I - 2 v_1 v_1^\top)(I - 2 v_0 v_0^\top) \in O(d)$ construct an exact orthonormal frame in $O(d)$ on GPU with zero cuSOLVER QR synchronization and zero RNG state consumption.
- **High-Dimensional Manifolds ($d > 8$, e.g., $d = 170$ Word Simplexes)**: Deterministic 32-bit Murmur3 Rademacher unit tangents $u_i \in \{\pm d^{-1/2}\}^d$ are pooled across $k = 32$ local Gaussian-weighted nearest neighbors:

$$
\widehat{\|\nabla_x D(x_i)\|_2^2} = d \sum_{m \in \mathcal{N}_k(i)} w_{im} \big(\nabla_x D(x_m) \cdot u_m\big)^2, \qquad \sum_{m} w_{im} = 1,
$$

cutting Hutchinson relative error by $1/\sqrt{32}$ (`param_grad_cosine = 0.79` at $d=170$) in **one $1.0\times$-batch forward-AD pass**.
- **BF16 Convolutional Image Critics (`is_image_tensor`)**: An instance-bound forward hook (`_attach_t2_critic_hook`) captures the live unscaled 1st-order logits $D(x_{\text{real}})$ and $D(x_{\text{fake}})$ computed during `d_loss` (guarded by step tags, `module.training`, and tensor backward-invalidation hooks so freed graphs are never reused). `_T2SecantPenaltyKernel` reuses these live endpoints at **0 extra forward cost**, evaluates **1** interleaved BF16 probe batch $z_{\text{base}} + \delta u_{\text{sep}}$ ($\delta = 0.02$, $O(B(CH+W))$ separable Murmur3 Rademacher perturbation), forms the multi-scale secant envelope across micro-probes, real-fake chords, and cyclic intra-batch rolls, and tracks Chamfer scale on $8 \times 8$ pooled spatial moments (`F.adaptive_avg_pool2d(..., 8)`) in $O(B^2)$ time — cutting total critic cost from $\sim 15 F_D$ to **$\sim 6 F_D$**.

---

## 3. DCGAN Energy, Latency & VRAM Speedup Benchmark

All benchmarks execute through the public `GANTrainer` API on NVIDIA RTX A6000 (`bf16_channels_last`, DualNorm quintic per-offset convolution, protocol seed 0, 10 measured updates after 3 warmup updates, `100%` finite updates across all cells).

### 3.1 `dcgan_performance_r256` ($256 \times 256$ RGB, 11.16M Critic / 12.20M Generator Params)

| Candidate / Formulation | Autograd Order | Receipt | Median `ms/step` | P95 `ms/step` | Throughput (`img/s`) | Peak Alloc (`MiB`) | Peak Reserved (`MiB`) | Speedup vs `t2-secant-v2` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`t2-secant-v2`** (`t2_secant`, uncontended GPU 1) | **1st-order (`0` `autograd.grad`)** | `readout-14cc2a44f2d8da942d2873dc` | **87.96** | **89.14** | **181.92** | **1,308.46** | **1,756.00** | **1.00x** |
| **`t2-secant-v1`** (`t2_secant`, uncontended GPU 1) | **1st-order (`0` `autograd.grad`)** | `readout-fbe38229308f61967c632eba` | **108.43** | 111.54 | 148.43 | **1,296.44** | **1,650.00** | 1.23x slower |
| **`t2-secant-v5`** (`t2_secant`, shared swarm GPU 0) | **1st-order (`0` `autograd.grad`)** | `readout-4edd360d71f5421c6149bd8a` | **115.08** | 121.02 | 139.07 | **1,308.46** | **1,756.00** | *(shared GPU)* |
| `empf-v8` (matched uncontended Cohort B) | 2nd-order `create_graph=True` | `readout-9672bf970e5f31704c112ba1` | 130.13 | 132.19 | 122.58 | 1,435.44 | 2,090.00 | **1.48x faster** |
| `empf-v8` (published inventory baseline) | 2nd-order `create_graph=True` | `technique-inventory.md` | 155.41 | — | — | 1,435.44 | — | **1.77x faster** |
| `empf-v8` (matched Cohort A) | 2nd-order `create_graph=True` | `readout-dcac6347fd2669483212c6db` | 173.05 | 175.83 | 92.49 | 1,435.44 | 2,090.00 | **1.97x faster** |
| `bispf-optimized-quintic-v1` (published / matched) | 2nd-order `create_graph=True` | `readout-b38b20b1f5b01560bb3fd2d8` | 349.26 *(340.02 pub)* | 358.08 | 45.65 | 2,087.28 | 2,272.00 | **3.87x–3.97x faster** |
| `rhbi-v12` (published / matched) | 2nd-order `create_graph=True` | `readout-49c12ce12c1befe480bc694b` | 599.88 *(588.62 pub)* | 601.66 | 26.70 | 3,706.55 | 6,320.00 | **6.69x–6.82x faster** |
| `recipe-prior-refactor-bcap-pure-v1` (published / matched) | 2nd-order `create_graph=True` | `readout-9004f032128bc809a1fcc61d` | 4,845.34 *(3,113.09 pub)* | 5,142.70 | 3.26 | 2,086.28 | 2,282.00 | **35.39x–55.09x faster** |

### 3.2 Full Resolution, Width & Batch Scaling (`r32`, `r64`, `b64`, `w128`, `r128`, `r256` Median `ms/step`)

| Task | `t2-secant-v2` (Uncontended) | `t2-secant-v5` (Shared GPU) | `empf-v8` (Matched B / Pub) | `bispf-quintic-v1` | `rhbi-v12` | `bcap-pure-v1` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dcgan_performance_r32` ($32^2$, $B=16$) | **21.02 ms** (`739 img/s`) | 23.29 ms | 24.07 / 24.48 ms | 31.81 ms | 64.08 ms | 717.76 ms |
| `dcgan_performance_r64` ($64^2$, $B=16$) | **27.98 ms** (`564 img/s`) | 28.28 ms | 33.85 / 32.17 ms | 51.70 ms | 86.97 ms | 1,281.55 ms |
| `dcgan_performance_b64` ($64^2$, $B=64$) | **31.42 ms** (**`2,038 img/s`**) | 37.84 ms | 39.89 / 41.05 ms | 122.28 ms | 145.83 ms | 1,390.99 ms |
| `dcgan_performance_r128` ($128^2$, $B=16$) | **50.32 ms** (`316 img/s`) | 51.35 ms | 61.02 / 64.61 ms | 122.64 ms | 205.45 ms | 3,287.03 ms |
| `dcgan_performance_w128` ($64^2$, $2\times$ width) | **55.25 ms** (`288 img/s`) | 85.42 ms | 63.45 / 67.35 ms | 145.53 ms | 182.53 ms | 2,432.54 ms |
| `dcgan_performance_r256` ($256^2$, $B=16$) | **87.96 ms** (`182 img/s`) | 115.08 ms | 130.13 / 155.41 ms | 349.26 ms | 599.88 ms | 4,845.34 ms |

> [!NOTE]
> `t2-secant-v2` and `t2-secant-v5` execute the **exact identical** convolutional image code path (`1,308.46 MiB` peak allocated / `1,756.00 MiB` peak reserved on `r256`, matching `r64` `28.28` vs `27.98 ms` and `r128` `51.35` vs `50.32 ms`); `v5`'s `r256` (`115.08 ms`) and `w128` (`85.42 ms`) ran while concurrent swarm workers shared GPU 0, and still beat even uncontended `empf-v8` (`155.41 ms`), `bispf` (`340.02 ms`), `rhbi` (`588.62 ms`), and `bcap` (`3,113.09 ms`).

---

## 4. Current `discriminator_stability` Leaderboard (Protocol Seed 0, Tiers 1–2)

Receipts: `empf-v8` (`readout-33f7628135472eef06f0d92a`), `t2-secant-v1` (`readout-2d0a1240650dbb7b14eb74d1`), `t2-secant-v2` (`readout-d1539c52c610cf0b249ed2e5`), `t2-secant-v3` (`readout-2dc35b433395cd50fadc41b9`), `t2-secant-v4` (`readout-e9cdca8471c210869cd70d91`), and **`t2-secant-v5`** (`readout-a2d41a678887a49f9ecc6aa9`). `-` indicates Tier 2 was unreached because Forge halts progression when any required Tier 1 gate is `FAIL` (or prerequisite DDGAN chain failed).

| Tier | Importance | Task | `empf-v8` (2nd-Order) | `v1` | `v2` | `v3` | `v4` | **`t2-secant-v5`** (1st-Order) | `t2-secant-v5` Key Metrics (Seed 0) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T1** | required | `gaussian1d_smoke` | PASS | FAIL | PASS | PASS | PASS | **PASS** | `cdf_ks=0.0918`, `std_ratio=1.0699`, `mean_err=0.1886` |
| **T1** | required | `two_pole` | PASS | PASS | PASS | PASS | PASS | **PASS** | `grad_med=0.7568 <= 1.0`, `mean_abs=0.9583`, `suffix=17` |
| **T1** | required | `unused_token_hold` | PASS | FAIL | PASS | PASS | PASS | **PASS** | `concept_move=0.9936`, `unused_hold=0.9901`, `suffix=13` |
| **T1** | required | `ae_gan_hold` | PASS | PASS | PASS | PASS | PASS | **PASS** | `hold=0.0160`, `recon_mse=0.0046`, `suffix=22` |
| **T1** | required | `ring16_acquisition` | PASS* | PASS | PASS | FAIL | FAIL | **PASS** | `modes=16/16`, `hq=0.9429`, `cov_err=0.4860 <= 0.85`, `suffix=33` |
| **T1** | required | `five_word_joint_smoke` | PASS | FAIL | PASS | PASS | PASS | **PASS** | `modes=5/5`, `p_min=1.0000`, `mass_tv=0.0189` |
| **T1** | required | `zero_tanh_smoke` | PASS | PASS | PASS | PASS | PASS | **PASS** | `mean_abs_error=0.0024`, `p99_abs_error=0.0084` |
| **T1** | required | `model_ae_gan_multimodal` | PASS | FAIL | FAIL | FAIL | PASS | **PASS** | `modes=4/4`, `center_rms=0.7782 <= 1.0`, `mse=0.0078`, `suffix=11` |
| **T1** | required | `model_ae_reconstruction` | PASS | PASS | PASS | PASS | PASS | **PASS** | `recon_mse=0.0065 <= 0.025`, `agree=1.0000`, `suffix=24` |
| **T1** | required | `model_vae_categorical` | PASS | PASS | PASS | PASS | PASS | **PASS** | `dec_hq=0.9995`, `lik_hq=0.9758`, `mse=0.0071`, `suffix=11` |
| **T1** | required | `model_vae_hard` | PASS | FAIL | PASS | PASS | FAIL | **PASS** | `dec_hq=1.0000`, `lik_hq=0.9941`, `lik_tv=0.0059`, `suffix=14` |
| **T1** | diagnostic | `clockfree_audit_measurement_v1` | PASS | PASS | PASS | PASS | PASS | **PASS** | `4/4` perturbation parity (`step_label`, `horizon`, `eval`, `restart`) |
| **T1** | diagnostic | `precision_gaussian_fp32` | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | `cdf_ks=0.3299`, `std_ratio=0.9279`, `mean_err=0.5224` |
| **T1** | stretch | `precision_gaussian_mxfp8` | FAIL | FAIL | FAIL | FAIL | FAIL | **PASS** | `cdf_ks=0.0353 <= 0.05`, `std_ratio=1.0292`, `suffix=10` |
| **T1** | stretch | `precision_gaussian_nvfp4` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | `cdf_ks=0.2477`, `std_ratio=2.3950`, `mean_err=0.0668` |
| **T1** | stretch | `interp_sparse_arc` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | `all_pairs_interp_hq=0.1515`, `endpoint_arc_bins=11` |
| **T2** | required | `gaussian1d_stability` | PASS | - | - | - | - | FAIL | `cdf_ks=0.0998`, `std_ratio=1.0792`, `mean_err=0.2399` |
| **T2** | required | `five_word_joint_hold` | PASS | - | - | - | - | **PASS** | `modes=5/5`, `p_min=1.0000`, `mass_tv=0.0346` |
| **T2** | required | `trajectory` | PASS | - | - | - | - | **PASS** | `identity_mse=0.0002`, `suffix=18` |
| **T2** | required | `residual_student` | PASS | - | - | - | - | **PASS** | `success_rate=1.0000`, `identity_mse=0.0002`, `suffix=20` |
| **T2** | required | `unipolar` | PASS | - | - | - | - | **PASS** | `cover=0.9983`, `neu_hold=0.9892`, `off_caption=0.0`, `suffix=21` |
| **T2** | required | `cover_leftover` | PASS | - | - | - | - | **PASS** | `content_kept=0.9909`, `leak_ratio=0.0024`, `suffix=22` |
| **T2** | required | `mid_scale_identity` | PASS | - | - | - | - | **PASS** | `concept_cos_minus=0.9999`, `concept_cos_plus=1.0000`, `suffix=21` |
| **T2** | required | `mode_hold` | PASS | - | - | - | - | FAIL | `modes=7/8`, `hq=0.9609` |
| **T2** | required | `model_ae_ddgan_gaussian_chain` | FAIL | - | - | - | - | FAIL | `final_projection_cdf=0.3091`, `paired_worst_axis_nrmse=0.3456` |
| **T2** | required | `vector_two_broad` | PASS | - | - | - | - | **PASS** | `hq=0.9917`, `mass_tv=0.0144`, `cov_err=0.4854`, `suffix=22` |
| **T2** | held | `vector_unequal_mass` | FAIL | - | - | - | - | FAIL | `hq=0.9792`, `mass_tv=0.0668`, `min_eig=0.1842 >= 0.15`, `cov_err=3.083` |
| **T2** | held | `vector_unequal_width` | FAIL | - | - | - | - | FAIL | `hq=0.9695`, `mass_tv=0.2456` *(empf: 0.2732)*, `min_eig=0.2789` |
| **T2** | required | `model_ddgan_gaussian_chain` | FAIL | - | - | - | - | FAIL | `final_projection_cdf=0.3463` |
| **T2** | held | `vector_anisotropic` | FAIL | - | - | - | - | FAIL | `cov_err=0.2579 <= 0.85` *(empf: 1.1672)*, `mass_tv=0.1959` |
| **T2** | held | `vector_overlap` | FAIL | - | - | - | - | FAIL | `sw1_normalized=0.1168` *(empf: 0.1277)*, `suffix=1` |
| **T2** | required | `vector_spiral` | PASS | - | - | - | - | **PASS** | `sw1_normalized=0.0365`, `suffix=23` |
| **T2** | required | `img_stripes2` | PASS | - | - | - | - | FAIL | `modes=1/2`, `hq=1.0000` |
| **T2** | held | `img_bars4` | FAIL | - | - | - | - | FAIL | `modes=2/4`, `hq=0.7188` |
| **T2** | held | `img_blobs4` | FAIL | - | - | - | - | FAIL | `modes=0/4`, `hq=0.0000` |
| **T2** | held | `img_intensity2` | FAIL | - | - | - | - | FAIL | `modes=2/2`, `hq=0.6875` |
| **T2** | required | `grid100` | FAIL | - | - | - | - | FAIL | `mass_tv=0.1782` *(empf: 0.1879)*, `precision=0.2251` |
| **T2** | required | `rotated100` | FAIL | - | - | - | - | FAIL | `100/100` modes, `mass_tv=0.1392` *(empf: 0.1405)*, `prec=0.3210` |
| **T2** | required | `staggered100` | FAIL | - | - | - | - | FAIL | `100/100` modes, `score=5.5901` *(empf: 5.9189)*, `mass_tv=0.1236` |
| **T2** | stretch | `interp_img_patch_shift` | FAIL | - | - | - | - | FAIL | `all_pairs_interp_rmse=0.1875` |
| **T2** | stretch | `interp_saddle_barycentric` | FAIL | - | - | - | - | FAIL | `barycenter_hq=0.6944`, `surface_mae=0.0741` |
| **Summary** | **Required T1 / Total T1 / Total T1+T2** | — | **11/11 · 13/16 · 24/41** | **6/11 · 7/16** | **10/11 · 11/16** | **9/11 · 10/16** | **9/11 · 10/16** | **11/11 · 13/16 · 21/41** | **Zero `torch.autograd.grad` calls across all 41 tasks** |

*\*Note on `ring16_acquisition`*: `empf-v8`'s archived `PASS` (`readout-33f7628135472eef06f0d92a`) was recorded on `torch 2.14.1+cu130`; when re-evaluated on the current `torch 2.15.0+cu132` runtime, `empf-v8` leaves 2 stray inter-mode bridge particles (`component_covariance_error = 3.1455 > 0.85`, `FAIL`), whereas `t2-secant-v5`'s spatial-frame chord smoothing clears `ring16_acquisition` cleanly on `torch 2.15.0+cu132` (`component_covariance_error = 0.4860 <= 0.85`, `suffix = 33`, `PASS`).

---

## 5. Failure Diagnoses & Concrete Next-Step Recommendations

1. **4D Discrete Image Tasks (`img_stripes2`, `img_bars4`, `img_blobs4`, `img_intensity2`) — Single-Probe Rademacher Variance**:
   - *Diagnosis*: While the 1-probe BF16 micro-secant envelope achieves `87.96 ms/step` on $256 \times 256$ images, in $d = 3{,}072$ ($32 \times 32 \times 3$) a single unpooled Rademacher direction has relative variance $\text{Var}[d(\nabla_x D \cdot u)^2] \approx 2 \|\nabla_x D\|_2^4$. Taking `amax` across candidates occasionally over-penalizes subtle discrete image decision boundaries (`img_stripes2` collapses from 2 stripes to `modes = 1, hq = 1.0`).
   - *Recommendation*: Pool `micro_gn_sq` across $k = 8$ nearest neighbors in the already-computed $8 \times 8$ average-pooled feature space (`r_obs`, `f_obs`) before applying the Born-Infeld softplus cap — zero extra CNN forwards ($0.0\text{ ms}$ overhead on `r256`) while reducing directional variance $8\times$.
2. **Conditioned Diffusion Chains (`model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`) — Joint Fiber Tangents**:
   - *Diagnosis*: On conditioned transitions $D(x_t, x_{t-1}, t)$, `_eval_exact_lowd_forward_grad_sq` propagated canonical tangents only on $x_t$ while holding $x_{t-1}$ at zero tangent.
   - *Recommendation*: Propagate orthonormal tangents jointly across $(x_t, x_{t-1})$ within matched timestep fibers $t$.
3. **1D Emulated FP32 & Long Stationary Holds (`precision_gaussian_fp32`, `gaussian1d_stability`, `mode_hold`)**:
   - *Diagnosis*: `EmulatedPrecisionLinear` executes custom `autograd.Function` GEMMs where `fwAD` drops or quantizes dual tangents differently across `mxfp8` (`PASS`, `cdf_ks = 0.0353`) vs `fp32` software reference (`cdf_ks = 0.3299`), and `chord_blend = 0.06` spatial-frame chord pairing on 1D (`b=128`) slightly shifts mean location over 5,000+ stationary hold updates (`gaussian1d_stability` `mean_error_sigma = 0.2399` vs `<= 0.20`; `mode_hold` `modes = 7/8`).
   - *Recommendation*: For $d = 1$ (where spatial neighbor rotation is unnecessary because $O(1) = \{\pm 1\}$ has only 1 axis), use the exact 1D tangent directly on `x_chord` without neighbor pairing, and detect custom `autograd.Function` linear wrappers explicitly to invoke the symmetric central-difference path.
4. **Cross-Track Synthesis (Combining Track 2 with Track 1 & Track 3)**:
   - `T2SecantCriticPenalty` (`particlegan/_t2_critic/`) is a drop-in, purely 1st-order critic module (`critic_formulation="t2_secant"`) orthogonal to generator/prior continuity (Track 1) and unbalanced scale-mass gauge transport (Track 3). Pairing `t2_secant` with Track 1/3's multi-mode coupling and horizon controllers immediately removes second-order `autograd.grad` cost across the combined recipe.
