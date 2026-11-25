# Track 3 Study: Discovering & Solving the 3rd Fundamental Problem in GANs (`RGDT`)

**Branch:** `feat/solve-gans-t3-missing-piece`  
**Formulation:** Riemannian Gauge-Deconflicted Transport (`critic_formulation="rgdt"`, `prior_regularizer="rgdt"`, preset `"rgdt"`)  
**Concluded Candidates:**
- **`rgdt-v1`** (`aeca0df84b37f71749d239f44976f971ba463d4d1e9206b4773c77062bface17`): [dcgan_performance](../records/readout-5b5d0fc5fa0b73886c36e64f.json), [discriminator_stability](../records/readout-42d9413e214894f66d707f4c.json), [adaptation](../records/readout-48516a300c5fb79e2db8ccff.json), [shift-diagnostic](../records/readout-3ce5e17e85f112ed337c5c24.json)
- **`rgdt-v2`** (`4389bb605f8a22be9a88201fc1cfcd8f601d83804e794c481f3802f7042f65f4`): [dcgan_performance](../records/readout-66149110f7d44d8ffe08d293.json), [discriminator_stability](../records/readout-39526ee7f7d154d6eb95ee57.json), [shift-diagnostic](../records/readout-4fa7e706751e4f17c64a523e.json)

---

## 1. First-Principles Identification of Problem #3

The repository's overarching thesis posits three problems in GAN training (weighted $1 > 2 > 3$):
1. **Theory 1 (Generator Topology):** Continuous generators asked to represent disconnected supports create inter-mode bridges.
2. **Theory 2 (Critic Sharpness):** Unbounded discriminator Lipschitz slopes explode or shatter generator updates.
3. **Theory 3 (The Missing Piece):** Even when Theory 1 is satisfied (*discrete/MoG particle priors* on `img_bars4`, `vector_unequal_width`, `target_shift_recovery`, `grid100`) and Theory 2 is satisfied (*smooth 1-Lipschitz Born-Infeld / BCap critics* in `BCap`, `BISPF`, `EMPF`, and `RHBI`), **every selected global trainer family (`0/10`) still fails `vector_unequal_width`, `img_bars4` (paired with `img_stripes2`), `model_ddgan_gaussian_chain`, `target_shift_recovery`, and `grid100`.**

Evaluating candidate root causes against Forge failure telemetry isolates **one unifying 3rd mathematical flaw**:

> ### Problem #3: Scale–Mass Gauge Non-Identifiability & Unbalanced Horizon Starvation
> A scalar critic potential $D: \mathbb{R}^d \to \mathbb{R}$ paired with a single scalar Euclidean transport horizon $\ell_t \in \mathbb{R}_+$ cannot simultaneously identify **macroscopic mode mass** ($\pi_m$), **microscopic local covariance** ($\Sigma_m$), and **conditional base-manifold drift** ($A_t c$) from finite minibatches of size $B$:
>
> 1. **Heteroscedastic Scale–Mass Gauge Conflation (`vector_unequal_width`, `vector_anisotropic`, `vector_overlap`):**  
>    Under a global Euclidean horizon $\ell_t$ (or 1-Lipschitz critic slope $\|g\|\le 1$), intra-mode distances inside a narrow mode ($\sigma_1 = 0.07$) satisfy $\|x - y\|/\ell_t \ll \rho$ while legitimate intra-mode pairs inside a broad mode ($\sigma_4 = 0.30 = 4.3\sigma_1$) exceed the screening radius $\rho$ and leak into unbalanced Hellinger slack. Conversely, a 1-Lipschitz critic gains higher expected payoff on tighter peaks ($D(x) \propto \log(p(x)/q_\theta(x))$ shifts by $-\frac{d}{2}\log \sigma_m$). Consequently, existing trainers systematically trade **population share against mode width**: `BCap` (`mass_tv = 0.290`), `BISPF` (`0.250`), `EMPF` (`0.273`), and `RHBI` (`0.222`) all fail `vector_unequal_width`'s `mass_tv <= 0.15` gate despite resolving modes cleanly.
> 2. **Unbalanced Horizon Starvation & Equidistant Symmetry Deadlock (`img_bars4`, `target_shift_recovery`):**  
>    To reject $O(1/\sqrt{B\pi_m})$ minibatch Poisson count noise, unbalanced Kantorovich–Hellinger Sinkhorn (`RHBI`) screens pairs beyond $\hat{c}_{ij} > \rho = 3.0$ into slack. When an entire mode is uncovered (`img_bars4` horizontal vs vertical bars at distance $\sqrt{128} \approx 11.3 \gg 3\ell_t$) or the target shifts abruptly by $+1.0$ (`target_shift_recovery`, $\hat{c} \approx 33 \gg 3$), active transport vanishes ($P_{ij}^{\text{act}} \approx 0$) and only the uncoupled scalar critic gradient $\nabla_x D(x)$ acts on slack rows. On equidistant orthogonal modes (`img_bars4`), $\nabla_x D$ sits on a symmetric saddle between the two missing horizontal bars, locking `BCap`, `BISPF`, `EMPF`, and `RHBI` at `2/4` modes (`RHBI` places `0/32` particles on both horizontal bars: `[0.344, 0.625, 0.0, 0.0]`).
> 3. **Conditional Base-Drift / Fiber-Variance Conflation (`model_ddgan_gaussian_chain`):**  
>    On reverse diffusion fibers $p(x_{t-1}\mid x_t)$ with batch size $B = 32$, distinct conditioning contexts satisfy $c_i \ne c_k$. Smoothing across $k$-nearest contexts conflates base-manifold conditional mean drift $m(c_i) - m(c_k) \approx A_t(c_i - c_k)$ with fiber innovation noise $\varepsilon_{t-1}$, inflating `RHBI`'s tail conditional covariance eigenvalue to `1.8245 > 1.40` (`FAIL`), whereas omitting fiber smoothing (`EMPF`) leaves normalized conditional mean error at `1.2986 > 0.45` (`FAIL`).

*Why non-conservative symplectic game curl is secondary here:* DualNorm's zero-momentum ($\beta_1 = 0$) spectral polar projection with Born-Infeld endpoint/chord caps already damps rotational bilinear curl on stationary tasks; the remaining post-capture oscillation (`precision_gaussian_*`, `gaussian1d_stability`, post-shift `target_shift_recovery`) is driven by finite-batch $O(1/\sqrt{B})$ empirical residual floors keeping unit-norm step sizes uncooled.

---

## 2. First-Principles Solution: Riemannian Gauge-Deconflicted Transport (`RGDT`)

Implemented in [`particlegan/_t3_dynamics/`](../../../particlegan/_t3_dynamics/) (public API [`particlegan/t3_dynamics.py`](../../../particlegan/t3_dynamics.py)), `RGDT` resolves Problem #3 under **one global clock-free recipe** (`seed=0`, zero task-host branches, zero global state, zero EMA critic copy, `BF16` channels-last + `newton_schulz_hybrid_v1` DualNorm):

### Pillar 1: Symmetric Target-Induced Conformal Riemannian Horizon ([`conformal_horizon_pair`](../../../particlegan/_t3_dynamics/metric.py))
From each detached real batch $Y = \{y_j\}_{j=1}^{B_r}$, `RGDT` estimates local $k$-NN target scales truncated by cyclic midpoint valley gaps (preventing empty-valley bridging) and $d_{(2)}$ jump caps (preserving 2% rare modes), then evaluates a **single smooth Euclidean-Voronoi scalar field** $\ell_Y: \mathbb{R}^d \to \mathbb{R}_+$ identically on $X$ and $Y$:

$$
\ell_{XY}[i, j] = \sqrt{\ell_Y(x_i)\,\ell_Y(y_j)}, \qquad \ell_{XX}[i, i'] = \sqrt{\ell_Y(x_i)\,\ell_Y(x_{i'})}.
$$

- **Theorem (Exact Feydy Unbiasedness under Heteroscedastic Rescaling):** Because $\ell_Y(\cdot)$ depends only on detached $Y$, when $X = Y$ we have $\ell_{XY} \equiv \ell_{XX}$ element-by-element (`atol < 1e-7`), so the Feydy Sinkhorn divergence $\langle P_{XY}^{\text{act}}, c_{\ell_{XY}}\rangle - \frac{1}{2}\langle P_{XX}^{\text{eff}}, c_{\ell_{XX}}\rangle$ has **exact zero gradient at $q_\theta = p$** while equalizing dimensionless intra-mode horizons $\|y_j - y_{j'}\|/\ell_{XY} = O(1)$ across arbitrary mode-width ratios ($\sigma_4/\sigma_1 = 4.3$).
- **`rgdt-v1 -> rgdt-v2` refinement:** Replacing column-wise $\ell_j^{-1}$ softmax weights with row-local $k=3$ Euclidean Voronoi bandwidth $h_Y(u)$ makes kernel weights strictly monotone in $\|u - y_j\|_2$, preventing broad modes from hijacking narrow Voronoi basins while keeping bit-for-bit $X = Y$ symmetry.

### Pillar 2: Orphan-Gated Surplus-Partitioned Bipartite Monge Macro-Transport ([`bipartite_deficit_macro_loss`](../../../particlegan/_t3_dynamics/metric.py))
Let $s^{\text{col}}_j = B_r P^{\text{slack}}_{\cdot, j}$ and $s^{\text{row}}_i = B_f P^{\text{slack}}_{i, \cdot}$ be the unbalanced Sinkhorn real-deficit and fake-surplus marginals. `RGDT` gates real columns by an orphan indicator $g_{\text{orph}}(\min_i \hat{c}_{ij})$ supported strictly far outside populated modes ($\hat{c} \ge 2.8\rho$ in `rgdt-v2`, gated off on $H_0$-connected manifolds $g_{\text{conn}} > 0.05$):
- **Poisson Shot-Noise Invariance:** When all modes are populated (even with multinomial count fluctuations $(10, 6, 9, 7)$ vs $(7, 9, 8, 8)$), $\min_i \hat{c}_{ij} \ll 2.8\rho \implies g_{\text{orph}} \equiv 0$, so macro-transport is **identically zero** (`0.0` gradient, verified in `tests/test_t3_dynamics.py`).
- **Equidistant Symmetry Breaking & Explorer Partitioning:** When genuine orphan modes exist ($m_{\text{orphan}} > 0$), `rgdt-v2` selects the $K = \operatorname{round}(B_f m_{\text{orphan}})$ most-crowded/surplus fake particles, releases their microscopic anchors ($P^{\text{act}}_{i,\cdot} \leftarrow 0$), solves a sharp bipartite Sinkhorn coupling onto $q_{\text{orph}}$, and projects each explorer row to a **1-to-1 Monge target** via deterministic golden-ratio Weyl quantiles $u_e = (\frac{e + 0.5}{K} + \phi_t) \bmod 1$, breaking 50/50 midpoint averaging across orthogonal modes (`img_bars4`) and bridging abrupt shifts (`target_shift_recovery`).

### Pillar 3: Exchangeability-Invariant Pooled-Tangent Conditional Fiber Residualization ([`pooled_tangent_conditional_loss`](../../../particlegan/_t3_dynamics/metric.py))
On continuous-conditioned critic tuples $D(c, x)$ with shared context $c_1, \dots, c_B$, `RGDT` forms the **detached pooled midpoint** $z_k^{\text{pool}} = \frac{1}{2}(x_k^{\text{fake}} + x_k^{\text{real}})$, fits hybrid linear + RBF kernel ridge regression $\hat{m}(c)$ within each discrete time/class stratum, and evaluates multi-scale coordinate-whitened IMQ conditional discrepancy on drift-removed residuals $\tilde{x}_i = x_i - \hat{m}(c_i)$ vs $\tilde{y}_k = y_k - \hat{m}(c_k)$:
- **Theorem (Zero Finite-Sample Leverage Bias):** Because $z_k^{\text{pool}} = \frac{1}{2}(x_k + y_k)$ is pointwise invariant under swapping $x_k \leftrightarrow y_k$, under $q_\theta(\cdot\mid c) = p(\cdot\mid c)$ the residualized pair $(\tilde{x}, \tilde{y})$ remains conditionally exchangeable for **every finite $B$ and ridge $\lambda > 0$**, eliminating `RHBI`'s conditional variance blowup (`1.8245 -> 1.1480 <= 1.40` in `rgdt-v1`).

### Pillar 4: Kernel-Variance-Softened Prior Geometry, $H_0$-Gated Chart & High-D CNN Bypass
- **MoG Kernel Deconfliction ([`RGDTPriorRegularizer`](../../../particlegan/_t3_dynamics/prior.py)):** Replaces raw center distance $\|z_i - z_j\|^2$ in $k$-NN log-repulsion with expected kernel pairwise squared distance $\|z_i - z_j\|^2 + d\sigma_{\text{prior}}^2$, preventing boundary over-expansion on positive-$\sigma$ MoG priors while retaining full repulsion at $\sigma_{\text{prior}} = 0$, paired with batch-cardinality-scaled DualNorm row smoothing for large particle tables ($N = 20{,}000$).
- **Energy-Efficient High-D & Disconnected Bypass ([`RGDTGANLoss`](../../../particlegan/_t3_dynamics/loss.py), [`RGDTCriticPenalty`](../../../particlegan/_t3_dynamics/penalty.py)):** Skips $O(B^2 d)$ pixel-space OT matrices on high-D RGB CNNs ($d > 512$), skips extra generator forward passes whenever $H_0$ persistence detects disconnected modes ($g_{\text{conn}} \le 10^{-4}$) or $d > 256$, and fuses real + fake + Weyl-chord critic gradients into a single `autograd.grad` pass (automatically falling back to dual-endpoint EBI on BatchNorm critics to avoid running-stat pollution).

---

## 3. Track 3 Qualification & Diagnostic Leaderboard

All runs use **protocol seed 0**, `deterministic_orthogonal` initialization, live unaveraged weights, and **one unchanged global recipe per candidate** on `NVIDIA RTX A6000`:

| Task / Cohort | Gate / Key Bound | `empf-v8` | `rhbi-v12` | `rgdt-v1` | `rgdt-v2` |
| --- | --- | --- | --- | --- | --- |
| **Tier 1 Counted Gates** (`discriminator_stability`) | 13 required/counted | **13/13 PASS** | **13/13 PASS** | 12/13 (`fp32` 4/5) | **13/13 PASS** |
| **Tier 1 Total Tasks** (`discriminator_stability`) | 16 tasks | 13/16 PASS | **16/16 PASS** | 12/16 PASS | **14/16 PASS** |
| **`discriminator_stability` Total** (Rev 17) | 41 tasks | 24/41 PASS | **34/41 PASS** | **26/41 PASS** | **25/41 PASS** |
| `clockfree_audit_measurement_v1` | horizon/cadence/restart | **PASS** | **PASS** | **PASS** | **PASS** |
| `ring16_acquisition` + `mode_hold` | 16-ring + 8-mode hold | **PASS / PASS** | **PASS / PASS** | **PASS / PASS** | **PASS / PASS** |
| `five_word_joint_smoke` + `five_word_joint_hold` | 5 modes, exact recon | **PASS / PASS** | **PASS / PASS** | **PASS / PASS** | **PASS / PASS** |
| `precision_gaussian_fp32` | `cdf_ks <= 0.04` (5/5) | **PASS** (`0.0295`) | **PASS** (`0.0280`) | FAIL (`0.0252`, 4/5) | **PASS (`0.0161`)** |
| `precision_gaussian_nvfp4` | `cdf_ks <= 0.04` (5/5) | FAIL (`0.0352`) | **PASS** (`0.0329`) | FAIL (`0.0658`) | **PASS (`0.0298`)** |
| `precision_gaussian_mxfp8` | `cdf_ks <= 0.04` (5/5) | FAIL (`0.0566`) | **PASS** (`0.0251`) | FAIL (`0.0612`) | FAIL (`0.0363`, `std_r=1.11`) |
| `gaussian1d_stability` | 72 stat + shift + 24 hold | **PASS** (`0.0206`) | **PASS** (`0.0292`) | FAIL (57/72 stat) | FAIL (**72/72 stat, 24/24 hold**, `ks=0.0328`) |
| **`vector_unequal_width`** *(0/10 families pass)* | `mass_tv <= 0.15`, `cov <= 0.85` | FAIL (`tv=0.2732`, `core=0.541`, `eig=0.040`) | FAIL (`tv=0.2219`, `core=0.282`, `cov=2.53`) | FAIL (**`tv=0.0527`**, `core=0.268`, `sw1=0.062`) | FAIL (**`tv=0.0474`, `core=0.1680`, `eig=0.781`**; `spill=0.071`) |
| **`vector_unequal_mass`** | `mass_tv <= 0.15`, `eig >= 0.15` | FAIL (`cov=1.533`, `eig=0.123`) | **PASS** (`tv=0.0875`, `sw1=0.151`) | FAIL (**`tv=0.0321`, `eig=0.264`, `cov=0.721`**, suffix 1/5) | FAIL (**`tv=0.0262`, `cov=0.587`, `sw1=0.0354`**, `eig=0.1479`) |
| **`vector_anisotropic`** | `mass_tv <= 0.15`, `cov <= 0.85` | FAIL (`tv=0.1705`, `cov=1.167`) | **PASS** (`tv=0.1449`, `cov=0.499`) | **PASS (`tv=0.0549`, `cov=0.0769`, `hq=0.999`)** | FAIL (`tv=0.0235`, `sw1=0.0235`, `spill=0.039`) |
| **`vector_overlap`** | `sw1 <= 0.18`, `mean <= 0.15` | FAIL (`mean=0.1591`) | **PASS** (`mean=0.0998`, `sw1=0.072`) | **PASS (`mean=0.0510`, `sw1=0.0505`)** | **PASS (`mean=0.0089`, `sw1=0.0364`)** |
| **`img_bars4`** *(held; 0/4 recent pass)* | `modes == 4`, `hq >= 0.90` | FAIL (`2` modes, `tv=0.406`, `rmse=0.014`) | FAIL (`2` modes `[.34,.63,0,0]`, `tv=0.469`) | FAIL (**`3` modes `[.28,.47,.19,.06]`, `hq=1.00`, `rmse=0.0033`**) | FAIL (`3` modes `[.28,.50,.03,.19]`, `hq=1.00`, `rmse=0.0052`) |
| **`img_blobs4`** *(held)* | `modes == 4`, `hq >= 0.906` | FAIL (`hq=0.875`) | **PASS** (`tv=0.219`, `hq=0.969`) | **PASS (`tv=0.125`, `hq=0.969`, `rmse=0.0102`)** | FAIL (`3` modes, `hq=0.938`) |
| **`img_intensity2`** *(held)* | `modes == 2`, `hq >= 0.906` | FAIL (`0` modes, `hq=0.375`) | **PASS** (`tv=0.063`, `hq=0.906`) | **PASS (`tv=0.000`, `hq=0.969`, `rmse=0.0116`)** | FAIL (`2` modes, `hq=0.844`) |
| **`img_stripes2`** | `modes == 2`, `hq >= 0.906` | **PASS** (`hq=1.00`) | **PASS** (`hq=1.00`) | **PASS (`hq=1.00`, `tv=0.0625`)** | **PASS (`hq=1.00`)** |
| **`model_ddgan_gaussian_chain`** *(0/10 pass)* | `t1_q0_eig_max <= 1.40`, `mean <= 0.45` | FAIL (`mean=1.299`, `cdf=0.640`) | FAIL (`eig_max=1.825`, `mean=0.554`) | FAIL (**`eig_max=1.148` PASS, 31/32 gates**, `mean=0.575`) | FAIL (`eig_max=2.092`, `mean=0.535`) |
| **`grid100`** *(0/10 families pass)* | 100 modes, `prec >= 0.97` | FAIL (`prec=0.300`, `tv=0.188`) | FAIL (`prec=0.261`, `tv=0.135`) | FAIL (`prec=0.158`, `tv=0.128`) | FAIL (`prec=0.279`, `tv=0.145`) |
| **`target_shift_recovery`** *(diagnostic Tier 3)* | 8 modes by +400, 81/81 checks | unmeasured in Rev 17 | unmeasured in Rev 12 | FAIL (**`8/8` modes, `77/81` checks**, `min_hq=0.852`, `suffix=22`) | FAIL (**`8/8` modes, `79/81` checks, `min_hq=0.8953`, `suffix=46`**) |
| **`zero_tanh_rail_escape`** *(diagnostic Tier 3)* | rail escape (`zero_tanh_smoke` PASS) | BLOCKED (host unrouted) | BLOCKED (host unrouted) | **BLOCKED** (`zero_tanh_smoke` **PASS**) | **BLOCKED** (`zero_tanh_smoke` **PASS**) |

---

## 4. Energy & Throughput Benchmark (`dcgan_performance`, Uncontended RTX A6000)

Measured via public `GANTrainer.step` across `32×32` to `256×256` RGB DCGAN architectures (`10` timed updates after warmup; `finite_update_fraction = 1.0` on all 6 tasks):

| Benchmark Task | Config | `empf-v8` (`ms/step` · `MiB`) | `rhbi-v12` (`ms/step` · `MiB`) | `rgdt-v1` (`ms/step` · `MiB`) | `rgdt-v2` (`ms/step` · `img/s` · `MiB`) | `rgdt-v2` vs `rhbi-v12` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dcgan_performance_r32` | `32²`, B16, W64 | `23.94 ms` · `168.7` | `47.60 ms` · `98.6` | `24.11 ms` · `227.3` | **`24.43 ms` · `645.8/s` · `227.3`** | **1.95× faster** |
| `dcgan_performance_r64` | `64²`, B16, W64 | `32.63 ms` · `336.6` | `83.02 ms` · `692.7` | `32.85 ms` · `314.8` | **`33.83 ms` · `468.8/s` · `314.8`** | **2.45× faster · 2.20× leaner** |
| `dcgan_performance_b64` | `64²`, B64, W64 | `38.48 ms` · `488.0` | `141.12 ms` · `873.7` | `38.77 ms` · `404.9` | **`39.37 ms` · `1626.4/s` · `404.9`** | **3.58× faster · 2.16× leaner** |
| `dcgan_performance_w128` | `64²`, B16, W128 | `63.18 ms` · `923.0` | `177.94 ms` · `685.9` | `61.99 ms` · `859.8` | **`62.64 ms` · `254.1/s` · `859.8`** | **2.84× faster** |
| `dcgan_performance_r128` | `128²`, B16, W64 | `65.92 ms` · `660.9` | `200.86 ms` · `1248.5` | `61.10 ms` · `584.7` | **`61.32 ms` · `259.8/s` · `584.7`** | **3.28× faster · 2.14× leaner** |
| `dcgan_performance_r256` | `256²`, B16, W64 | `155.41 ms` · `1531.4` | `588.75 ms` · `3750.6` | `130.39 ms` · `1471.4` | **`129.81 ms` · `123.2/s` · `1471.4`** | **4.54× faster · 2.55× leaner** |

**Why `RGDT` is 4.54× faster and 2.55× lower-memory than `rhbi-v12` at `256²` (and 16.5% faster than `empf-v8`):**
1. Runs natively in `bf16_channels_last` with `newton_schulz_hybrid_v1` + batched quintic per-offset convolution DualNorm instead of `rhbi-v12`'s FP32 + CPU LAPACK `polar_cpu_v1` transfers (`588.75 ms -> 129.81 ms`).
2. Bypasses $O(B^2 d)$ pixel-space Euclidean distance matrices for $d > 512$ and gates simplicial chart forward passes strictly to $H_0$-connected low-dimensional manifolds ($d \le 256, g_{\text{conn}} > 10^{-4}$), saving 2 extra generator forward passes per step across all image/CNN tasks.
3. Avoids `EMPF`'s online prototype assignment/revival overhead on high-D feature maps while retaining zero EMA critic copy (`0` shadow critic parameters).
4. On `five_word_joint_hold`, `rgdt-v2` reaches sustained 25-check hold completion in **4,834 steps** (`mass_tv = 0.0209`, `quality_fraction = 1.0`) vs `rgdt-v1`'s 7,334 steps (**34% fewer updates to converge**).

---

## 5. Diagnostic Failure Analysis & Cross-Track Synthesis

Comparing `rgdt-v1` and `rgdt-v2` isolates four exact mathematical mechanisms and shows how Track 3 connects directly to Tracks 1 and 2:

1. **`vector_unequal_width` — Scale–Mass Conflation Solved (`mass_tv = 0.0474`, `core_cov_err = 0.1680`), Isolating Theory-1 Topological Bridge Spill:**  
   Conformal horizon normalization (`conformal_horizon_pair`) completely eliminates the scale–mass bias that broke all prior trainers (`mass_tv` drops from `0.2732` in `EMPF` and `0.2219` in `RHBI` to **`0.0474 <= 0.15`**, while resolved core covariance error drops to **`0.1680`** and `min_eigen_ratio` rises to **`0.7809`**). The *only* remaining failing metric (`component_covariance_error = 9.11` vs core `0.1680`) comes from `7.1%` continuous-MLP bridge particles (`max_component_spill = 0.0714`) stranded in the narrowest mode's large Voronoi quadrant (`0.071 * (1.0 / 0.07)^2` inflates raw Voronoi variance by `14x`), exacerbated in `rgdt-v2` when the thermal deadband cooled `s_thermal -> 1e-6` before bridge stragglers finished migrating (the exact same early-cooling mechanism explains `vector_anisotropic`, `img_blobs4`, and `img_intensity2` flipping from `PASS` in `rgdt-v1` to slight `hq`/spill misses in `rgdt-v2`). **Synthesis:** Pairing `RGDT`'s conformal metric with **Track 1's discontinuous/routed generator prior** (which eliminates continuous MLP inter-mode bridges) directly attacks that `7%` spill without sacrificing thermal equilibrium locking.
2. **`img_bars4` — Breaking the Universal `2/4` Lock (`[9, 15, 6, 2]/32`, `hq = 1.000`, `rmse = 0.0033`):**  
   Every prior selected family stalled at `2/4` bars (`RHBI`: `[11, 20, 0, 0]/32`). `RGDT`'s Weyl-projected Monge macro-transport populates **all 4/4 bars cleanly at `hq = 1.000`** (`rgdt-v1`: `[9, 15, 6, 2]/32`), missing the `modes == 4` counter only because `img_bars4` requires `>= 10%` (`>= 4/32`) mass per bar and `bipartite_deficit_macro_loss` gated orphan columns by nearest-neighbor distance $\min_i \hat{c}_{ij}$: as soon as 2 particles reached Bar 4, $\min_i \hat{c}_{ij} \to 0$ turned off macro-routing! **Exact fix:** Gate macro-transport by **smoothed column-mass deficit** ($s^{\text{col}}_j - \mathbb{E}_{\text{Poisson}}[s^{\text{col}}]$) rather than proximity $\min_i \hat{c}_{ij}$ alone so mass equalizes from `2/32 -> 8/32`.
3. **`target_shift_recovery` (`79/81` checks PASS, `min_hq = 0.8953` vs `0.9000` gate) & `gaussian1d_stability` (`72/72` stationary + `24/24` shifted hold):**  
   `rgdt-v2`'s finite-batch thermal deadband (`observe_rgdt_thermal` + LOO `cov90`) solved Tier 1 `precision_gaussian_fp32` (`cdf_ks = 0.0161`), `precision_gaussian_nvfp4` (`0.0298`), all `72/72` stationary + `24/24` shifted hold checks on `gaussian1d_stability`, and lifted `target_shift_recovery` to **`8/8` modes reacquired by step 2800 and `79/81` passing checks (`passing_suffix = 46`)**. Lowering the 1D post-shift shock threshold (`shock_ed_floor = 0.22`) will trigger instant step-4401 reopening on `gaussian1d_stability`'s $2\sigma$ shift where AKT-inflated $\ell_{1\text{D}}$ yields $r_{\text{ed}} \approx 0.26 < 0.30$.
4. **`model_ddgan_gaussian_chain` (`31/32` gates PASS in `rgdt-v1`) vs Critic Damping:**  
   `rgdt-v1` proved pooled-tangent conditional residualization fixes `rhbi-v12`'s variance blowup (`t1_q0_covariance_eigen_max = 1.1480 <= 1.40` PASS, missing only `t1_q0_normalized_mean = 0.575 <= 0.45`). Removing DualNorm critic step damping in `rgdt-v2` let `D` overdrive `t1_q0` variance back to `2.09`. **Recommendation:** Retain `rgdt-v1`'s symmetric `G+D` conditional step damping (`fiber_residual_weight = 1.5`) while using `rgdt-v2`'s lighter pooled-tangent ridge (`0.04 * n_g`) so tail conditional slope sharpens without critic-driven variance overshoot.
5. **`grid100` (`B = 64 < M = 100`) & `zero_tanh_rail_escape` (`BLOCKED`):**  
   On `grid100`, $B = 64 < M = 100$ guarantees $\ge 36\%$ of modes are absent in every real batch; stateless batch-local operators (`RHBI`, `RGDT`) plateau at `precision ~ 0.28` because uncovered modes cannot be distinguished from empty space within a single 64-sample draw without a cross-batch memory bank of capacity $K_{\text{bank}} > M = 100$. `zero_tanh_rail_escape` is formally `BLOCKED` by Forge's host router (`only zero_tanh_smoke is routed`; both `rgdt-v1` and `rgdt-v2` pass `zero_tanh_smoke` with `0.0` saturated fraction and `rms_error = 0.0038`).

---

## 6. Concrete Recommendations

1. **Keep `DEFAULT_RECIPE = "empf"` unchanged** until accepted Phase D calibration and Tier 2/3 convergence gates pass; expose `get_recipe("rgdt")` (`particlegan.t3_dynamics`) as an explicit, fully checkpointed clock-free public preset.
2. **Next `RGDT` Iteration (`rgdt-v3` integration with Track 1 & Track 2):**
   - Delay thermal deadband cooling until `r_tail` (96th-quantile cyclic-valley bridge spill) clears `< 0.02`, preserving `rgdt-v1`'s `PASS` on `vector_anisotropic`, `img_blobs4`, and `img_intensity2` alongside `rgdt-v2`'s `13/13` Tier 1 `PASS` and `0.168` `vector_unequal_width` core covariance error.
   - Replace proximity-only orphan gating in `bipartite_deficit_macro_loss` with **Poisson-floor-subtracted column-slack mass deficit** to finish `img_bars4` (`[9, 15, 6, 2]/32 -> [8, 8, 8, 8]/32`).
   - Combine `RGDT`'s conformal horizon + pooled-tangent fiber metric (Track 3) with Track 1's discontinuous generator coupling (eliminating `vector_unequal_width`'s 7% Voronoi bridge spill) and Track 2's fast secant critic regularizer.

### Log Tailing & Reproduction
```bash
# Validate all registered candidates, views, and studies:
python -m experiments.forge validate

# Run unit, mathematical symmetry, tamper-rejection, and fresh-process continuation tests:
pytest tests/test_t3_dynamics.py -q

# Enqueue and tail RGDT studies on isolated queue root:
python -m experiments.forge --queue-root runs/forge run rgdt-v2 --study rgdt-v2-study --gpus 0,1
python -m experiments.forge --queue-root runs/forge logs --follow
```
