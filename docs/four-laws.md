# The Four Laws of Stable, Energy-Efficient GAN Convergence

> **Status & Governance Scope:** The theories below synthesize bounded research studies executed through the public `particlegan` API under protocol seed `0`, `deterministic_orthogonal` initialization, live scoring weights, and one unchanged global recipe per candidate on `NVIDIA RTX A6000` GPUs. Every score, timing, VRAM figure, energy measurement, and failure in [Part II](#part-ii-solutions-found-so-far) is grounded verbatim in committed Forge study reports and immutable [`reports/forge/records/readout-*.json`](../reports/forge/records/) receipts. Screening calibration remains provisional (`BLOCKED` under [`configs/forge/calibration/criteria-v1.json`](../configs/forge/calibration/criteria-v1.json)); see [Honest Failure, Calibration & Scope Boundaries](#honest-failure-calibration--scope-boundaries).

ParticleGAN's research program asked why GAN training breaks across disconnected multimodal distributions, high-resolution images, non-stationary shifts, conditional reverse-diffusion chains, `>600×` player capacity imbalances, and deep unnormalized or saturating architectures — and what minimal structural requirements govern stable, energy-efficient convergence without step-count clocks.

The findings decompose into **Four Laws** grouped into **two mathematical halves**:

- **Half A — Measure & Potential Geometry (Laws 1–3):** Representing disconnected target mass, maintaining bounded critic slopes at low computational cost, and identifying local scale, beyond-horizon mass, and conditional fiber geometry.
- **Half B — Function-Space Two-Player Dynamics (Law 4):** Mapping generator and critic parameter updates into bounded, scale-invariant, well-conditioned function-space displacements.

This document intentionally **decouples the laws from their implementations**:

- **[Part I: The Four Laws](#part-i-the-four-laws-solution-agnostic-theories)** states each law as a generic, solution-agnostic theory — the mathematical obstruction, what *any* valid solution must satisfy, and how to falsify a candidate — independent of any specific prior, loss, regularizer, or optimizer.
- **[Part II: Solutions Found So Far](#part-ii-solutions-found-so-far)** catalogs the concrete mechanisms discovered to date (`unified_v11`, `minimal`, `cond-fiber-v1`, `t4-nash-v3`, `law4-energy-v3`/`v10`, and earlier families), Occam's Razor causal ablations, and open gaps where better solutions are still sought.

---

## Part I: The Four Laws (Solution-Agnostic Theories)

Each law describes a distinct, independently isolatable failure mode of adversarial generative modeling. A concrete algorithm is only one candidate realization; simpler, faster, or more general solutions to any law remain open research targets.

### Law 1 — Measure & Prior Topology

**Core Theory (Fundamental Obstruction):**
A continuous neural pushforward $G_\theta: \mathbb{R}^{d_z} \to \mathbb{R}^d$ transforming a connected, fixed-scale unimodal prior onto a disconnected multimodal target $P_Y = \sum_{m=1}^M w_m P_m$ faces an unavoidable topological bottleneck: it must either stretch low-density bridges across empty valleys or create extreme Jacobian cliffs ($\|J_G\| \gg 1$). At the same time, whenever finite minibatches satisfy $B < M$ or mode masses are unequal, naive single-batch transport couplings suffer high Poisson count variance and cross-valley barycentric averaging ("ghost" mass in empty space between modes).

**What Any Valid Solution Must Satisfy:**
1. **Topological Compatibility:** Reconcile the latent prior's support geometry with disconnected, multimodal targets without forcing steep pushforward Jacobians ($\|J_G\| = O(1)$).
2. **Finite-Batch Mass & Valley Integrity:** Preserve discrete mode identities and mass proportions across finite minibatches ($B < M$), separating genuine valleys without splitting contiguous modes or averaging mass across empty space.
3. **Local Pushforward Conditioning:** Keep local pushforward scale and covariance aligned with target components across unequal widths, unequal masses, anisotropy, and partial overlap, without depending on a step-count clock.

**How to Test / Falsify a Candidate:**
- High-multiplicity disconnected lattices (`grid100`, `rotated100`, `staggered100`) where $B$ is comparable to or smaller than $M$.
- Discrete multi-mode visual tasks (`img_bars4`) where barycentric averaging immediately produces ghost states.
- Heterogeneous mixtures (`vector_unequal_width`, `vector_unequal_mass`, `vector_anisotropic`, `vector_overlap`) and single-factor knockouts testing whether mass conservation and mode coalescence are causally load-bearing.

---

### Law 2 — Critic Potential Geometry

**Core Theory (Fundamental Obstruction):**
Adversarial transport relies on a critic potential $D(x)$ whose local spatial slope $\|\nabla_x D(x)\|_2$ stays bounded ($\le \kappa$) under an anchored logit gauge. Without slope control, gradients explode, collapse, or drift; yet enforcing slope regularity through second-order reverse-mode differentiation (`create_graph=True`) multiplies compute and VRAM several-fold and inhibits fused kernels on high-dimensional or convolutional critics. Conversely, naive finite-difference slope surrogates $|D(b) - D(a)|/\|b - a\|_2$ fail for three fundamental geometric reasons:
1. **Symmetric-Flank Chord Cancellation:** Chords spanning equal-height flanks satisfy $D(\mu + r u) - D(\mu - r u) = 0$ even as $\|\nabla_x D\|_2 \to \infty$.
2. **High-Dimensional Projection Attenuation:** A single random unit probe $\hat{u} \in S^{d-1}$ underestimates $\|\nabla_x D\|_2$ by $\Theta(\sqrt{d})$.
3. **Piecewise-Linear Kink Impulses:** Perturbations straddling ReLU/LeakyReLU hyperplanes inject $O(1/\delta)$ parameter-gradient noise.

**What Any Valid Solution Must Satisfy:**
1. **Accurate Local Slope & Gauge Control:** Bound $\|\nabla_x D(x)\|_2 \le \kappa$ and fix the arbitrary additive logit constant across smooth and piecewise-linear critics.
2. **Geometric Faithfulness:** Avoid symmetric-flank cancellation, $\sqrt{d}$ high-dimensional attenuation, and cross-kink gradient impulses.
3. **First-Order / Low-Energy Efficiency:** Achieve slope regularity without paying higher-order graph compute/VRAM overhead, while remaining compatible with low-precision arithmetic and clock-free training.

**How to Test / Falsify a Candidate:**
- Matched latency, VRAM, and hardware energy scaling on high-resolution convolutional critics (`dcgan_performance_r256`).
- High-dimensional structured targets (`five_word_joint_hold`) sensitive to $\sqrt{d}$ attenuation.
- Reduced-precision training (`mxfp8`, `nvfp4`) and direct Pareto ablations replacing the candidate regularizer with a second-order gradient penalty while holding all other components fixed.

---

### Law 3 — Scale, Mass & Conditional Fiber Identifiability

**Core Theory (Fundamental Obstruction):**
Even when **Law 1 is completely removed** ($G = \mathrm{Id}$, zero trainable generator parameters, independent data-space particles) **and Law 2 is strictly satisfied** (a smooth $C^\infty$ critic verified at $\operatorname{med}\|\nabla_x D\|_2 \le 1$), a scalar critic potential $D(x)$ paired with a single global Euclidean transport horizon $\ell_t$ remains mathematically under-determined. It cannot simultaneously identify:
1. **Multi-Scale Local Covariance:** Under large mode-width and mass ratios, a single global Euclidean horizon either over-pulls tight modes or fails to resolve broad ones.
2. **Beyond-Horizon Orphan Modes:** After large distribution shifts or equidistant orthogonal saddle shifts, local gradients vanish or cancel by symmetry, stranding unvisited modes beyond the transport horizon.
3. **Conditional Base-Manifold & Outer-Tail Fiber Geometry:** In stratified or conditional multi-step generation (e.g., reverse-diffusion chains), sparse per-stratum minibatch counts starve outer-tail mean and covariance leverage.

**What Any Valid Solution Must Satisfy:**
1. **Local Scale Invariance:** Measure local support discrepancy invariantly to component width and mass ratio so narrow and broad modes converge simultaneously.
2. **Long-Range Orphan Detection & Rescue:** Detect unvisited modes beyond the local horizon and transport mass across symmetric saddles without destabilizing settled multi-scale mixtures.
3. **Unbiased Conditional Fiber Estimation:** Resolve conditional mean and covariance across sparse strata/fibers without cross-stratum bias, mean-shift contamination, or fragile/prohibitive matrix decompositions.

**How to Test / Falsify a Candidate:**
- Causal isolation probes with $G = \mathrm{Id}$ and verified smooth 1-Lipschitz critics (`t3_iso_scale_mass_8x`, `t3_iso_horizon_saddle_shift`) plus factorial dissociations.
- Non-stationary shift-recovery suites (`t3-shift-recovery-diagnostic-v1`).
- Stratified conditional reverse-diffusion chains (`model_ddgan_gaussian_chain`, `model_ddgan_bimodal_chain`, `model_ae_ddgan_*`).

---

### Law 4 — Function-Space Two-Player Dynamics

**Core Theory (Fundamental Obstruction):**
The GAN minimax game $\min_G \max_D \mathcal{L}(G_\theta, D_\psi)$ operates on pushforward measures and potential functions, whereas parameter-space optimizers step in raw parameter coordinates ($\|\Delta\theta\| \sim \eta$), inducing architecture-dependent function-space shifts $\Delta f(u) \sim \Theta(\alpha L \sqrt{W}\,\eta)$. Across capacity imbalances, deep/saturating networks, and stochastic minibatches, this mismatch breaks down into **five falsifiable sub-laws** (detailed in [`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md#5-the-five-formal-optimizer-laws-of-law-4)):

- **Law 4.1 — Inter-Player Function-Space Impedance:** Under player capacity/gain mismatch $\rho \gg 1$ or hostile initialization, equal parameter steps produce $\Theta(\rho)$ function-space displacement mismatch and finite-time divergence unless realized function-space displacements are bounded.
- **Law 4.2 — Intra-Player Degree-0 Gauge & Rail Escape:** Multiplicative inter-layer Jacobian attenuation across deep networks and $\operatorname{sech}^2$-like activation saturation span many orders of magnitude ($10^{-12}$ to $10^{+2}$), defeating any single player-level learning rate or fixed additive $\varepsilon$ floor. Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish ($\Delta W \to 0$ as $G \to 0$) is required without a step clock.
- **Law 4.3 — Tensor-Rank Spectral Collapse vs. Low-Rank Noise Inflation:** Elementwise, Frobenius, $L_\infty$, and Kronecker-factored preconditioners preserve gradient rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$) and therefore cannot rescue deep skip-free dense layers whose gradient effective rank collapses ($\operatorname{erank}(G) \to 1$). Conversely, applying aggressive spectral whitening to already-diverse convolutional kernels or naturally low-rank equilibrium gradients wastes compute and injects orthogonal agitation.
- **Law 4.4 — Generator–Critic Structural Asymmetry under Bounded Critics:** When Law 2 bounds the critic's spatial slope, the generator carries the nonlinear pushforward geometry across disconnected modes, deep compositions, and saturation rails, whereas the critic primarily needs protection against extreme $D \gg G$ capacity imbalance and equilibrium logit jitter.
- **Law 4.5 — Stochastic Minimax Staleness & Minimum-State Trade-Off:** Once Law 2 damps critic rotational eigenvalues, stochastic minibatch two-player training penalizes stale momentum ($\beta_1 > 0$), high-variance optimistic extrapolation, and heavy coordinate-wise state, favoring low-staleness, minimum-state updates.

**What Any Valid Solution Must Satisfy:**
1. **Bounded Inter-Player Displacement:** Stabilize arbitrary capacity ratios ($\rho > 500\times$), bidirectional LR sweeps, and hostile initializations by regulating realized function-space motion.
2. **Scale-Invariant Layer & Rail Equalization:** Equalize extreme depth attenuation and saturation rails without underflow dead zones, while settling cleanly at equilibrium with no step clock.
3. **Selective Spectral Conditioning:** Restore gradient rank where diagonal conditioning hits the rank-preservation limit (e.g., deep skip-free dense networks) without injecting orthogonal noise or redundant matmuls where layers do not collapse.
4. **Conflict-Free Multi-Objective & Low-Energy Execution:** Handle auxiliary/constrained objectives (e.g., encoder/autoencoder or residual losses) without silent gradient interference, at minimal optimizer state and hardware energy.

**How to Test / Falsify a Candidate:**
- Extreme player-capacity imbalance, hostile-init, and bidirectional LR-asymmetry sweeps (`t4_imbalance_nash_stretch`).
- Pathological deep skip-free MLPs and saturating residual rails (`t4-strange-architectures-v1`: `grid100_mlp16`, `grid100_res16_tanh`, `zero_tanh_rail_escape`).
- Multi-objective constrained tasks (`residual_student`), long-horizon continuations (`14k` tasks), and matched uncontended hardware energy (`J/step` and `Joules-to-PASS`).

---

## Part II: Solutions Found So Far

The mechanisms below are the concrete solutions discovered and benchmarked against Laws 1–4 across Rounds 1–4 and subsequent Law-4 energy studies. None of them is "the law" itself — each represents a point on the quality / simplicity / energy Pareto frontier with explicit strengths and remaining gaps.

### Overview Matrix: Laws vs. Discovered Solutions

| Law | Primary Solutions Found So Far | Lighter / Earlier Alternatives | Known Open Gaps |
| :--- | :--- | :--- | :--- |
| **Law 1** *(Measure & Prior Topology)* | Streaming Laguerre–Pythagorean prototype atlas + LTV coalescence + quota-balanced Laguerre assignment + orthogonal pullback covariance isometry (`t1_prior`, `t1_coupling`, `unified_v11`) | SBNP (`bispf`, `rhbi`), online *e-means* (`empf-v8`), compact `<900`-LOC core (`minimal-v2`) | High host-sync overhead (`113` `cudaStreamSync`/step in pullback on `grid100`); `staggered100` live-suffix margin under default `unified_v11` |
| **Law 2** *(Critic Potential Geometry)* | First-order `fwAD` dual-tangent reconstruction (MLPs) + live-endpoint reuse & separable BF16 micro-secant Born–Infeld with zero-mean gauge (`t2_secant`, `unified_v11`, `minimal`) | Second-order `autograd.grad` penalties (`BCap`, `R1/R2`, `EBI-TCR`, `EMPF`, `RHBI` — `1.48×–55×` slower) | Amortized or cheaper first-order slope estimation on high-res Conv2d (where `d_penalty` is `~15.8%` of step time once optimizer cost is cut) |
| **Law 3** *(Scale–Mass–Fiber Identifiability)* | Scale-invariant $k=4$ density-ratio orphan macro-transport + 1D Weibull batch horizon (`rgdt-v3`, `unified_v11`); cross-batch conditional fiber operator bank (`cond-fiber-v1` with `eigh`/SVD; `unified_v11` zero-`eigh` cyclic Jacobi) | Global scalar Euclidean horizon (`bcap`, `empf`, `rhbi` — `0/2` on `t3_causal_isolation_stretch`) | Outer-tail variance on `model_ddgan_bimodal_chain` (cracked by `cond-fiber-v1` and `law4-energy-v10`, open in default `unified_v11`), `model_ae_ddgan_gaussian_chain`, `interp_saddle_barycentric` |
| **Law 4** *(Function-Space Two-Player Dynamics)* | Dynamic-$10^{-38}$ scale norm + homogeneous `peak_rms` settling + Marchenko–Pastur scaling + `delta_g_max` pushforward trust + 2D Quintic Tikhonov NS + zero-`pinv` `direction_blend` (`t4-arch-v3`, `unified_v11`); asymmetric zero-Conv2d / zero-Critic NS variants (`law4-energy-v3`, `law4-energy-v10`) | Zero-NS equilibrated (`t4-nash-v3`, `minimal-v2`), zero-state zero-NS lower bounds (`law4-sign-mp`, `law4-frob-sgd-mp`), naive Adam/SGD (`0/3` Nash, `0/10` strange-arch) | Consistent live-suffix pass on `grid100_res16_tanh`; unifying Phase-2 asymmetric Law-4 energy gains into a calibrated default |

---

### Solutions to Law 1 (Measure & Prior Topology)

- **Streaming Laguerre–Pythagorean Prototype Quantizer & Law-of-Total-Variance Coalescence** ([`t1-prior-topology.md`](../reports/forge/studies/t1-prior-topology.md), [`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md)): Maintains a clock-free streaming $K$-prototype atlas $(C_k, w_k, v_k)$ that separates modes only across genuine valleys and merges overlapping sub-cells via exact total-variance conservation:
  $$C_{ab} = \frac{w_a C_a + w_b C_b}{w_a + w_b}, \qquad v_{ab} = \frac{w_a v_a + w_b v_b}{w_a + w_b} + \frac{w_a w_b}{(w_a + w_b)^2}\,\frac{\|C_a - C_b\|_2^2}{D}.$$
- **Quota-Balanced Laguerre Monge Assignment & Intra-Mode Halo** ([`t1-deconflicted-coupling.md`](../reports/forge/studies/t1-deconflicted-coupling.md)): Partitions sampled particle rows onto discrete Laguerre cells $k^\star(u) = \arg\min_k (\|G_\theta(\mu_u) - C_k\|_2^2 - \psi_k)$ using Moreau–Yosida-damped zero-mean Kantorovich dual potentials $\psi_k$, eliminating cross-valley barycentric averaging and protecting settled particles inside intra-mode halos when $B_r < M$.
- **Orthogonal Pullback Covariance Isometry:** Matches local pushforward covariance $\Sigma_{\text{push}} = \sigma_z^2 J_G J_G^\top \approx v_k I$ using deterministic orthogonal latent probes without `autograd.grad` or SVD, letting latent support expand smoothly so $\|J_G\| = O(1)$ on 100-mode lattices.

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- `t1-prior-v2` ([`readout-379720b9`](../reports/forge/records/readout-379720b9859815defff0efaa.json), [`readout-c982ddf2`](../reports/forge/records/readout-c982ddf21128b926906438aa.json)): jumped Tier-2 held generalization from `0/7` (`empf-v8`) to **`6/7` PASS** (passing `vector_unequal_width`, `vector_unequal_mass`, `vector_anisotropic`, `vector_overlap`) and `29/41` T1–2 PASS (`+5` over `empf-v8`), `3.22×` faster than `rhbi-v12` on `r256`.
- `t1-coupling-v5` ([`readout-b87c22fb`](../reports/forge/records/readout-b87c22fbdb500d43981e839b.json), [`readout-cedd207f`](../reports/forge/records/readout-cedd207fc183ede3b4e66e38.json)): first recipe to pass **`4/4` discrete image tasks** including **`img_bars4`** (`4/4` modes, `hq = 0.9688`, `tv = 0.1562`, where both `empf-v8` and `rhbi-v12` collapse to `2/4` modes), reaching `31/41` T1–2 PASS.
- Round-4 Occam's Razor Arm 5 (`-chart-laguerre`, [`readout-b27b2689`](../reports/forge/records/readout-b27b26896bd137e2e701ce2d.json)) and Arm 4 (`-coalesce-ltv`, [`readout-04bb3c21`](../reports/forge/records/readout-04bb3c211150d5629c489b5e.json)) showed Laguerre dual mass-balancing (`7/15` PASS without it) and Pythagorean LTV coalescence (`9/15` PASS without it) are the **most irreducible mechanisms among our current solutions**.

---

### Solutions to Law 2 (Critic Potential Geometry)

- **Forward-Mode Tangent Reconstruction (`fwAD.dual_level`, manifolds & MLPs)** ([`t2-efficient-critic.md`](../reports/forge/studies/t2-efficient-critic.md)): Evaluates $(D_\theta(x), \nabla_x D_\theta(x) \cdot u)$ at the **exact primal activation mask** (zero kink impulses, zero `create_graph=True`, single 1st-order `.backward()`): exact canonical basis $I_d$ for $1 \le d \le 4$ (with central-difference fallback for quantized MXFP8/NVFP4 ops lacking `jvp`), chained $O(d)$ Householder frames for $5 \le d \le 8$, and RNG-free Murmur3 Rademacher tangents with local $k$-NN Gaussian Hutchinson pooling ($k=32$) for $d > 8$.
- **Live-Endpoint Reuse + Separable BF16 Micro-Secants (Convolutional Image Critics):** Captures the live 1st-order logits $D(x_{\text{real}}), D(x_{\text{fake}})$ from `d_loss` via a forward hook at zero extra forward cost and evaluates a single BF16 micro-secant probe batch alongside free transport-chord and cyclic-roll secants, with $8 \times 8$ pooled Chamfer scale tracking and a Born–Infeld slope penalty with zero-mean logit gauge (`ebi_beta`).

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- `t2-secant-v2` ([`readout-14cc2a44`](../reports/forge/records/readout-14cc2a44f2d8da942d2873dc.json)): cuts uncontended `dcgan_performance_r256` median latency to **`87.96 ms/step`** (`181.92 img/s`, `1,308.46 MiB` allocated VRAM) — **`1.48×–1.97×` faster than `empf-v8`**, **`3.87×–3.97×` faster than `bispf`**, **`6.69×–6.82×` faster than `rhbi-v12`** (`-72.2%` reserved VRAM), and **`35.4×–55.1×` faster than `bcap`**.
- `t2-secant-v5` ([`readout-a2d41a67`](../reports/forge/records/readout-a2d41a678887a49f9ecc6aa9.json)): clears `11/11` required Tier-1 gates + `clockfree_audit` + `precision_gaussian_mxfp8` + `five_word_joint_hold` with **zero `torch.autograd.grad` calls**.
- Round-4 Occam's Razor Arm 7 (`-fwad-critic`, replacing 1st-order `fwAD`/secants with 2nd-order `autograd.grad` EBI on `unified-v10`, [`readout-4b04c4bf`](../reports/forge/records/readout-4b04c4bf0dc3d7408a82e51e.json), [`readout-b96faf85`](../reports/forge/records/readout-b96faf85499b1dbe9dce102e.json)): demonstrated 1st-order `fwAD`/secant Born–Infeld is **strictly Pareto-dominant** over 2nd-order EBI — `1.48×` faster (`89.66` vs `133.01 ms/step`), `-477.7 MiB` (`-34.8%`) less VRAM (`1372.6` vs `1850.3 MiB`), and higher pass rate (`grid100`, `img_bars4`, and `zero_tanh_rail_escape` all regressed under 2nd-order EBI).

---

### Solutions to Law 3 (Scale, Mass & Conditional Fiber Identifiability)

- **Scale-Invariant Local Horizon & $k=4$ Density-Ratio Orphan Macro-Transport** ([`t3-third-piece-proof.md`](../reports/forge/studies/t3-third-piece-proof.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Compares cross-density directly against local real $k=4$ spacing:
  $$r_{\text{loc}}(y_j) = \frac{d_{k=4}(y_j, X_{\text{fake}})}{d_{k=4}(y_j, Y_{\text{real}} \setminus \{y_j\})}, \qquad g_{\text{orphan}}(y_j) = \operatorname{smoothstep}\!\left(\frac{r_{\text{loc}}(y_j) - 3.52}{1.30}\right),$$
  coupled with exact 1D Weibull batch-horizon scaling (`akt_exp = max(0, 1/d - 0.5)`). Because $r_{\text{loc}}$ is invariant to local mode width $\sigma_m$, tight modes in an $8\times$ scale-mass mixture are never over-pulled, while genuine $42.4\sigma$ equidistant orphan modes trigger full bipartite macro-transport rescue.
- **Swap-Invariant Cross-Batch Conditional Fiber Operator Bank** ([`r3-conditional-ddgan.md`](../reports/forge/studies/r3-conditional-ddgan.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Per discrete stratum $t$, maintains strictly pre-update dual-timescale sufficient statistics on a piecewise-hinge + bounded-bilinear context basis $\phi(c)$ paired with a $384$-entry paired-context FIFO ring. Pooled midpoint residuals $r_X = X - m_{\text{pool}}(c)$, $r_Y = Y - m_{\text{pool}}(c)$ guarantee **exact midpoint cancellation** ($\mathbb{E}[r_X r_X^\top - r_Y r_Y^\top \mid c] = \Sigma_X(c) - \Sigma_Y(c)$ with zero mean-bias outer product $\tfrac{1}{4}\Delta\Delta^\top$). In `unified_v11`, `zero_eigh_symmetric` (branchless closed-form for $d \le 2$, float64 cyclic Jacobi for $d \ge 3$, `< 1.1e-14` agreement with `torch.linalg.eigh`) eliminates `eigh` completely.

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- **Causal Isolation (`t3_causal_isolation_stretch`):** `bcap-pure-v1` `0/2` ([`readout-18e9fbb0`](../reports/forge/records/readout-18e9fbb07a4d8c24ba8e924b.json)), `empf-v8` `0/2` ([`readout-7e0c3159`](../reports/forge/records/readout-7e0c315962234ea1c2c2ba3f.json)), `rhbi-v12` `0/2` ([`readout-8a63e947`](../reports/forge/records/readout-8a63e947313fcb7d2e62a8d2.json)), vs **`rgdt-v3` `2/2` PASS** ([`readout-c30d4a58`](../reports/forge/records/readout-c30d4a5840e8d861f62c1dcb.json)) and **`unified-v11` `2/2` PASS** ([`readout-d1e36bf5`](../reports/forge/records/readout-d1e36bf514f755e46b333e6d.json)), with clean $3 \times 3$ factorial triple dissociation (`readout-efdd1af5`, `readout-e3f9c013`, `readout-8d2fa268`, `readout-b72ef358`).
- **Conditional Reverse-Diffusion Breakthrough:** `cond-fiber-v1` ([`readout-5932f4d3`](../reports/forge/records/readout-5932f4d351cc4e791417835a.json)) delivered the first Forge `PASS` on both `model_ddgan_gaussian_chain` (`38/38` bounds, `9` consecutive terminal passes) and `model_ddgan_bimodal_chain` (`8` consecutive passes), reaching `40/47` T1–2 PASS; `unified-v11` ([`readout-fc5b67a7`](../reports/forge/records/readout-fc5b67a767e57d748f460610.json)) ported the bank to **zero `eigh`** and became the first zero-SVD/zero-`eigh`/zero-`autograd.grad` recipe to pass `model_ddgan_gaussian_chain` (`t1_q0_mean = 0.1790`, `eig_max = 1.3085`, `5/5` suffix).

---

### Solutions to Law 4 (Function-Space Two-Player Dynamics)

- **Underflow-Safe $10^{-38}$ Dynamic Scale Normalization & Homogeneous `peak_rms` Settling** ([`t4-imbalance-nash.md`](../reports/forge/studies/t4-imbalance-nash.md), [`r3-nash-complete.md`](../reports/forge/studies/r3-nash-complete.md)): Normalizes $(m_t, v_{\text{rms},t})$ by $s_t = \max(\max |v_{\text{rms},t-1}|, \max |g_t|, 10^{-38})$ so $10^{-12}$ early-layer and $\operatorname{sech}^2$-attenuated gradients receive the same directional geometry as $10^{+2}$ output gradients, while degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock.
- **Closed-Form Marchenko–Pastur Operator Scaling + Realized Generator Pushforward Trust Band (`delta_g_max`):** Analytically normalizes layer operator gain by $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ in $O(\text{params})$ arithmetic, and measures realized generator output displacement $\delta_G = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ on a detached eval-mode probe to clamp `548×` overpowered generators before they overshoot into mode collapse.
- **Fused $Z = [X, Y]$ Quintic Tikhonov Polar Orthogonalization & Zero-`pinv` Analytic `direction_blend`** ([`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Replaces $O(\min(m,n)^3)$ SVD with 5-step pure-GEMM Quintic Newton–Schulz iterations ($a=3.4445, b=-4.7750, c=2.0315$) batched across same-shape 2D matrices, plus exact closed-form $1\times 1 / 2\times 2$ non-ascent constraint projection (`zero_pinv_project_nonascent`, using standard `.backward()` with zero `autograd.grad` and zero `pinv`).
- **Asymmetric Zero-Conv2d / Zero-Critic NS Energy-Optimal Realizations** ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md)): Applies Laws 4.3–4.5 selectively by dropping redundant 4D Conv2d NS via stride-aware polar-matched Frobenius-MP (`law4-energy-v3`) and dropping 2D Critic NS via exact polar-Frobenius match $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ (`law4-energy-v10`), alongside zero-NS, zero-state research lower bounds (`law4-sign-mp`, `law4-frob-sgd-mp`).

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- **Imbalance-Invariant Nash (`t4_imbalance_nash_stretch`):** `t4-nash-v3` ([`readout-6f996ec2`](../reports/forge/records/readout-6f996ec256a149b2f1620f64.json), [`readout-642d4c55`](../reports/forge/records/readout-642d4c55d093b44ed1e0309e.json)) achieved **`3/3` (`100%`) sustained `5/5` PASS** with **strictly `0` SVD, `0` `eigh`, `0` `pinv`, `0` Newton–Schulz, and `0` `autograd.grad`** at **`39.68 ms/step`** on `256×256` RGB DCGAN (`3.92×` faster than `empf-v8`, `14.84×` faster than `rhbi-v12`), recovering `16/16` modes across a `100×` bidirectional LR-asymmetry sweep (`d_lr_mult` from `10.0` to `0.10`).
- **Pathological 16-Layer & Rail Architectures (`t4-strange-architectures-v1`):** `t4-arch-v3` ([`readout-dc890b70`](../reports/forge/records/readout-dc890b704514d91effb156a6.json), [`readout-4acf96c2`](../reports/forge/records/readout-4acf96c2efad5e17f86d9df2.json)) jumped from `0/10` historical families to **`7/7` (`100%`) PASS** (including `grid100_mlp16` `prec = 0.9814`, `grid100_res16_tanh` `prec = 0.9856, sat = 0.0`, `grid100_res16_scaled_tanh` `prec = 0.9868`, `zero_tanh_rail_escape` `mae = 0.00105`), plus `3/3` fresh-process `14,000`-step native continuations (`grid100_14k`, `rotated100_14k`, `staggered100_14k`, [`readout-ff228439`](../reports/forge/records/readout-ff22843954240927dfd7ecc3.json)).
- **Unified Reference (`unified-v11`):** **`3/3` Nash PASS** ([`readout-2eb07e97`](../reports/forge/records/readout-2eb07e97714a1da4177d5d25.json)), **`6/7` strange-arch PASS** ([`readout-b96d9de4`](../reports/forge/records/readout-b96d9de47bb52cc91fbb4a4f.json)), **`5/5` shift/rail PASS** ([`readout-2029c37a`](../reports/forge/records/readout-2029c37a1359498d3d8f2eb3.json)), and `residual_student` `1.0` (`wrong_pad_rate = 0.0`) via zero-`pinv` `direction_blend`.
- **Asymmetric Energy-Optimal Law-4 Follow-ups ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md), [`law4-energy-champion.md`](../reports/forge/studies/law4-energy-champion.md)):** `law4-energy-v3` ([`readout-d53010c7`](../reports/forge/records/readout-d53010c7356ee6557df70807.json), [`readout-b898e928`](../reports/forge/records/readout-b898e9285957d3feba1bf714.json)) cuts `r256` to **`45.89 ms/step`** / **`12.56 J/step`** (`0` Conv2d NS, `0` critic secant probes); `law4-energy-v10` ([`readout-a0e3e7f6`](../reports/forge/records/readout-a0e3e7f6620e7476c9599d67.json)) adds `0` Critic NS to reach **`42/48`** Tiers 1–2 `discriminator_stability` PASS at **`45.30 ms/step`** and **`11.68 J/step`**; and **`unified-v12`** (`DEFAULT_RECIPE = "unified_v12"`, [`readout-5002641a`](../reports/forge/records/readout-5002641a64ddaa91f1fc3b87.json), [`readout-731e752f`](../reports/forge/records/readout-731e752ff3c4f97e29a84145.json)) adds generator-free 2D critic NS fallback + CountSketch stiff-generator preconditioning + zero-sync pullback fast-path to reach **`46/48` Tiers 1–2 `discriminator_stability` PASS (`29/29` required = `100%`, incl. `2/2` combined AE-DDGAN)**, **`18/18` (`100%`) untouched Tier-3/14k holdouts**, **`16/16` Occam**, **`7/7` standalone `t4-arch`** (`6/7` combined with PR #615: `grid100_res16_tanh` passes `100k` holdout `ctr=0.1224` with a 1-ULP step-`6250` `modes=99` dip), and **`39.22 ms/step` / `11.37 J/step`** on strictly idle RTX A6000 `r256` (**`-45.8%`** J/step vs post-PR-#612+#615 `unified-v11`, **`-52.0%`** vs pre-Phase-3 `unified-v11`).

---

### Unified Realizations & Occam's Razor Synthesis

All four laws are currently realized together in the `0.12.0` default research recipe **`unified_v12`** (`DEFAULT_RECIPE = "unified_v12"`, [`particlegan/unified_v12.py`](../particlegan/unified_v12.py)), retaining explicit **`unified_v11`** ([`particlegan/unified_v11.py`](../particlegan/unified_v11.py)), while the self-contained **`minimal`** preset (`899` LOC, [`particlegan/minimal.py`](../particlegan/minimal.py)) implements a zero-Newton-Schulz fast core.

Round-4 Track 2 ([`r4-occam-ablation-razor.md`](../reports/forge/studies/r4-occam-ablation-razor.md), `18` concluded single-factor knockout readouts on `unified-v10`) and Track 1 ([`r4-minimal-core.md`](../reports/forge/studies/r4-minimal-core.md), `899`-LOC `minimal-v2`) tested which parts of our current solutions are mathematically irreducible versus redundant epicycles:

| Verdict | Mechanism | Exact Knockout Evidence (Seed `0`) |
| :--- | :--- | :--- |
| **Irreducible Core** | **Laguerre dual mass-balancing & $H_0$ chart commitment** | Removing it (Arm 5, [`readout-b27b2689`](../reports/forge/records/readout-b27b26896bd137e2e701ce2d.json)) drops `14/16 -> 7/15` PASS: collapses `grid100` (`prec = 0.2827`), `img_bars4` (`2/4`), `ring16`, `vector_unequal_width`, and both Nash tasks. |
| **Irreducible Core** | **1st-order `fwAD` / multi-probe secant Born–Infeld critic** | Strictly Pareto-dominant over 2nd-order `autograd.grad` EBI (Arm 7, [`readout-4b04c4bf`](../reports/forge/records/readout-4b04c4bf0dc3d7408a82e51e.json), [`readout-b96faf85`](../reports/forge/records/readout-b96faf85499b1dbe9dce102e.json)): `1.48×` faster, `-34.8%` VRAM, and higher pass rate. |
| **Irreducible Core** | **Pythagorean sibling coalescence + LTV variance conservation** | Removing it (Arm 4, [`readout-04bb3c21`](../reports/forge/records/readout-04bb3c211150d5629c489b5e.json)) drops `14/16 -> 9/15` PASS (`grid100` `center_rms_sigma` worsens `0.1379 -> 0.5312`). |
| **Irreducible (1-Regime)** | **Generator pushforward trust ceiling (`delta_g_max`)** | Removing it (Arm 2, [`readout-4e2fd695`](../reports/forge/records/readout-4e2fd695306a5dd6fd222e62.json)) collapses `548×` `nash_overpowered_generator_ring16` to `6/16` modes (`hq = 0.2522`). **Critic logit secant band is redundant** (`overpowered_critic` & `hostile_imbalance` both PASS without it). |
| **Irreducible (1-Regime)** | **2D Quintic Newton–Schulz on deep skip-free MLPs (`L > 5`)** | **Split verdict** (Arm 1 v2, [`readout-690dd40a`](../reports/forge/records/readout-690dd40a4221b390d8680e43.json), [`readout-0219c6e2`](../reports/forge/records/readout-0219c6e2857b1605cbcf3428.json); independently confirmed by `minimal-v2` [`readout-6bac8895`](../reports/forge/records/readout-6bac8895ffa14ef20dcbee75.json)): strictly required ONLY for 16-layer skip-free `grid100_mlp16` (`prec` drops `0.9929 -> 0.0244–0.0714` without it); **100% redundant** on shallow nets, `grid100`, 16-layer residual `res16_scaled_tanh`, `3/3` Nash, and 4D Conv2d DCGANs (dropping Conv2d NS speeds `r256` from `89.66 -> 76.17 ms/step`). |
| **Irreducible (1-Regime)** | **2D (`d_z=2`) pullback covariance isometry & 3-line 1D Weibull AKT** | `d_z=2` isometry is needed for `grid100` covariance (Arm 3, [`readout-44dbbe8a`](../reports/forge/records/readout-44dbbe8ab156d3615c474db4.json)); 3-line 1D AKT is needed for `gaussian1d_smoke` (Arm 6, [`readout-edffe8ce`](../reports/forge/records/readout-edffe8cee489f8b14768e2cb.json)). |
| **Proven Redundant Epicycles** | **4D Conv2d/shallow NS · Critic logit secant band · `d_z=4` Radon–Hurwitz 4-tangent branch · 2D `_conformal_local_horizons` · Case A `w_sat_pt` saturation probe** | Knocking each out preserves target regimes cleanly (e.g., `zero_tanh_rail_escape` still passes at `mae = 0.0026` without `w_sat_pt` because $10^{-38}$ dynamic scale normalization already inverts `sech^2`; `12/12` 2D/4D/deep/Nash/image probes pass bit-for-bit without 2D `_conformal_local_horizons`). |

#### Cross-Champion Summary (Verbatim Committed Forge Receipts, Seed `0`, `NVIDIA RTX A6000`)

| Candidate | Core LOC | `autograd.grad` | SVD/`eigh`/`pinv` | `disc_stab` (T1–2) | `precision` (`fp32/mx8/nv4`) | `ddgan_gauss` | `t4_nash` | `t3_iso` | `t3_shift` | `t4_arch` | `endurance` | `r256` (`ms` · `MiB`) |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unified-v12`** (`DEFAULT_RECIPE`) | — | **0** | **0** | **`46/48`** *(29/29 req)* | **`3/3`** | **`PASS`** *(+2/2 AE-DDGAN)* | **`3/3`** | **`2/2`** | **`5/5`** | **`7/7`** *(6/7 comb.)* | **`18/18` T3** | **`39.76` · `1243.9`** *(`11.37 J/st`)* |
| **`unified-v11`** *(0.11.x explicit)* | `4,713` | **0** | **0** | `41/47` | `3/3` | `PASS` | `3/3` | `2/2` | `5/5` | `6/7` | `3/3` | `89.19` · `1372.6` |
| `unified-v10` | `4,311` | 0 | 0 | `40/46` | `2/3` | `FAIL` | `3/3` | `1/2` | `5/5` | `6/7` | `3/3` | `89.66` · `1372.6` |
| `cond-fiber-v1` *(+`_unified`)* | `4,244` | 0 | `eigh`+SVD | `40/47` | `3/3` | `PASS` | — | — | — | — | — | `87.50` · `1308.4` |
| `t4-nash-v3` | `3,016` | 0 | 0 | `33/46` | — | `FAIL` | `3/3` | — | — | — | — | **`39.68`** · `1372.9` |
| **`minimal-v2`** (`preset="minimal"`) | **`899`** | **0** | **0** *(0 NS)* | `10/17` T1 | `0/3` | — | `2/3` | `1/2` | `2/4` *(u)* | `3/7` *(u)* | — | **`55.50` · `1344.8`** |
| `empf-v8` *(0.10.1 default)* | `1,972` | Yes | Yes | `24/41` | — | — | — | `0/2` | — | — | — | `130.13` · `1435.4` |
| `rhbi-v12` *(0.10.0 default)* | `3,114` | Yes | Yes | `34/41` | — | — | — | `0/2` | — | — | — | `588.75` · `3750.6` |

*(Across all `9` concluded Forge studies in [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md), `unified-v11` passes **`55/62` unique runnable tasks (`88.7%`)**, including **`15/17`** executed Tier-3 `14k` holdout tasks (`grid100_14k` and `rotated100_14k` PASS) and **`3/4`** runnable Tier-4 frontier tasks. In the subsequent Phase-2 Law-4 energy study [`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md), `law4-energy-v10` reached **`42/48`** Tiers 1–2 `discriminator_stability` PASS at **`45.30 ms/step`** / **`11.68 J/step`** on idle `r256`. `—` denotes no committed readout for that candidate/view.)*

---

## Honest Failure, Calibration & Scope Boundaries

Per repository standards, passing software tests or leading screening does not manufacture scientific qualification or hide remaining failures:

1. **Screening Calibration & Default Adoption Remain `BLOCKED`:** Under frozen [`configs/forge/calibration/criteria-v1.json`](../configs/forge/calibration/criteria-v1.json) (audited in [`r3-calibration-holdout.md`](../reports/forge/studies/r3-calibration-holdout.md) and [`r3-calibration-holdout.json`](../reports/forge/calibration/r3-calibration-holdout.json)), screening calibration is `BLOCKED` because no single global recipe passes `16/16` reference tasks, rev-19's 11-task Tier-1 smoke screen exceeds the `<= 900s` / `<= 0.10` cost ratio bounds, and cheap 3-task smoke screens fail the `<= 0.10` false-accept ceiling across partial solvers. Accordingly, `unified_v12` (like `unified_v11` before it) is promoted as the **current default research selection** (`default_adoption: false` in Forge's selection pin), not a calibrated scientific winner; multi-seed confirmatory promotion and untouched real-data image FID confirmation remain open.
2. **Unequal Tuning Budgets Disclosed:** `unified_v11` (`unified-v1`..`v11`) benefited from iterative Round 1–4 development across the screening views, whereas untouched Tier-3 `14k` continuations (`grid100_14k`, `rotated100_14k`) and Tier-4 frontier holdouts provide bounded post-selection checks on seed `0`.
3. **All 7 Remaining Numerical Failures of `unified-v11` (Verbatim from [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)):**
   - `staggered100` (blocking `staggered100_14k`): passes coverage and independent `100k` holdout (`precision = 0.9927`, `center_rms_sigma = 0.1484`), misses live sustained accuracy (`final center_rms_sigma = 0.1649`).
   - `grid100_res16_tanh`: passes `100k` holdout (`precision = 0.9884`, `saturated_fraction = 0.0`), misses intermediate live suffix check.
   - `five_word_joint_hold`: passes `23/25` continuation checks and final endpoint (`5/5` modes, `mass_tv = 0.0346`), misses two mid-window checks (steps `3001` and `3334`).
   - `model_ddgan_bimodal_chain`: reaches `8/24` passing checks (`final_hq = 0.9478`), retains excess outer-tail `t1_q0` variance (`eig_max = 3.1085`) without `cond-fiber-v1`'s exact SVD DualNorm smoothing.
   - `model_ae_ddgan_gaussian_chain` (blocking `model_ae_ddgan_bimodal_chain`): strong marginal & paired reconstruction (`paired_nmse = 0.0530`, `36.91×` shuffled gain), but `t1_q0` remains under-converged (`mean = 0.7536`, `0/24` checks).
   - `interp_img_patch_shift`: hits a known CUDA `float32` mean-reduction edge case in the frozen scorer (`0.99999994 < 1.0`).
   - `interp_saddle_barycentric`: covers `16/16` bins (`barycenter_surface_mae = 0.0741`), short of the stretch surface-accuracy threshold.
4. **Explicitly `BLOCKED` Tier-4 Stretch Hosts (`12` tasks):** `misgan_*`, `pacgan_*`, `alice_spiral_bigan`, `infogan_anisotropic_modes`, `cgan_overlapping_classes`, `cramer_energy_overlap`, `cyclegan_unpaired_constellation`, and `ring16_bigan_continuous` remain `BLOCKED` (distinct from numerical `FAIL`) pending public Forge host adapters.

---

## Further Reading

- [Technique Inventory (Current Canonical Leaderboard)](../reports/forge/technique-inventory.md)
- [API Reference & Recipe Defaults](api.md#recipes-and-defaults)
- [Research Glossary & Ownership Rules](../RESEARCH_GLOSSARY.md)
- [Round-4 Solution-Complete Study (`unified_v11`)](../reports/forge/studies/r4-solution-complete.md)
- [Round-4 Occam's Razor Ablation Study](../reports/forge/studies/r4-occam-ablation-razor.md)
- [Law-4 Optimizer Laws, Causal SGD Anatomy & Energy Study (`law4-energy-v3`/`v10`)](../reports/forge/studies/law4-optimizer-laws.md)
- [Round-4 Minimal `<900`-LOC Core Study (`minimal`)](../reports/forge/studies/r4-minimal-core.md)
- [Round-3 Calibration & Holdout Audit](../reports/forge/studies/r3-calibration-holdout.md)
