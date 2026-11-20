# The Four Laws of Stable, Energy-Efficient GAN Convergence

> **Status & Governance Scope:** The mechanisms and results below summarize bounded research studies executed through the public `particlegan` API under protocol seed `0`, `deterministic_orthogonal` initialization, live scoring weights, and one unchanged global recipe per candidate on `NVIDIA RTX A6000` GPUs. Every score, timing, VRAM figure, and failure is grounded verbatim in committed Forge study reports and immutable [`reports/forge/records/readout-*.json`](../reports/forge/records/) receipts. Screening calibration remains provisional (`BLOCKED` under [`configs/forge/calibration/criteria-v1.json`](../configs/forge/calibration/criteria-v1.json)); see [Honest Failure, Calibration & Scope Boundaries](#6-honest-failure-calibration--scope-boundaries).

ParticleGAN's Round 1–4 research program asked why GAN training breaks across disconnected multimodal distributions, high-resolution images, non-stationary shifts, conditional reverse-diffusion chains, `>600×` player capacity imbalances, and 16-layer unnormalized architectures — and what minimum mathematical structure solves them **without `torch.autograd.grad`, without SVD/`eigh`/`pinv`, and without step-count clocks**.

The empirical answer consists of **Four Laws**, which Round-4's causal Occam's Razor ablation proved collapse cleanly into **two mathematical halves**:

- **Half A — Measure & Potential Geometry (Laws 1–3):** How disconnected target mass, critic Lipschitz slopes, and local scale/conditional fiber gauges are represented.
- **Half B — Function-Space Two-Player Dynamics (Law 4):** How generator and critic parameter updates map into bounded, scale-invariant function-space displacements.

All four laws are united in the `0.11.0` default research recipe **`unified_v11`** (`DEFAULT_RECIPE = "unified_v11"`, [`particlegan/unified_v11.py`](../particlegan/unified_v11.py)), while the self-contained **`minimal`** preset (`899` LOC, [`particlegan/minimal.py`](../particlegan/minimal.py)) implements the zero-Newton-Schulz fast core.

---

## 1. Law 1 — Measure & Prior Topology (Discontinuous Pushforward Resolution)

**Fundamental Obstruction:** A continuous neural pushforward $G_\theta: \mathbb{R}^{d_z} \to \mathbb{R}^d$ transforming a connected Gaussian prior into a disconnected multi-modal law $P_Y = \sum_{m=1}^M w_m P_m$ must either stretch low-density bridges across empty valleys or create steep Jacobian cliffs ($\|J_G\| \gg 10$). Standard prior regularizers tether latent spread to $\operatorname{std}(\mu) \approx 1$, inflating fixed-width Gaussian kernels across separated lattices (`grid100`, `rotated100`, `staggered100`), while single-batch balanced Sinkhorn splits mass across modes (`img_bars4` 0.5-ghost collapse) and suffers $\pm 62\%$ Poisson count noise whenever $B_r < M$.

**First-Principles Resolution:**
1. **Streaming Laguerre–Pythagorean Prototype Quantizer & Law-of-Total-Variance Coalescence** ([`t1-prior-topology.md`](../reports/forge/studies/t1-prior-topology.md), [`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md)): Maintain a clock-free streaming $K$-prototype atlas $(C_k, w_k, v_k)$ that separates modes only across genuine valleys and merges overlapping sub-cells via exact total-variance conservation:
   $$C_{ab} = \frac{w_a C_a + w_b C_b}{w_a + w_b}, \qquad v_{ab} = \frac{w_a v_a + w_b v_b}{w_a + w_b} + \frac{w_a w_b}{(w_a + w_b)^2}\,\frac{\|C_a - C_b\|_2^2}{D}.$$
2. **Quota-Balanced Laguerre Monge Assignment & Intra-Mode Halo** ([`t1-deconflicted-coupling.md`](../reports/forge/studies/t1-deconflicted-coupling.md)): Partition sampled particle rows onto discrete Laguerre cells $k^\star(u) = \arg\min_k (\|G_\theta(\mu_u) - C_k\|_2^2 - \psi_k)$ using Moreau–Yosida-damped zero-mean Kantorovich dual potentials $\psi_k$, eliminating cross-valley barycentric averaging and protecting settled particles inside intra-mode halos when $B_r < M$.
3. **Orthogonal Pullback Covariance Isometry**: Match local pushforward covariance $\Sigma_{\text{push}} = \sigma_z^2 J_G J_G^\top \approx v_{k} I$ using deterministic orthogonal latent probes without `autograd.grad` or SVD, allowing the latent support to expand smoothly so $\|J_G\| = O(1)$ on 100-mode lattices.

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- `t1-prior-v2` ([`readout-379720b9`](../reports/forge/records/readout-379720b9859815defff0efaa.json), [`readout-c982ddf2`](../reports/forge/records/readout-c982ddf21128b926906438aa.json)): jumped Tier-2 held generalization from `0/7` (`empf-v8`) to **`6/7` PASS** (passing `vector_unequal_width`, `vector_unequal_mass`, `vector_anisotropic`, `vector_overlap`) and `29/41` T1–2 PASS (`+5` over `empf-v8`), `3.22×` faster than `rhbi-v12` on `r256`.
- `t1-coupling-v5` ([`readout-b87c22fb`](../reports/forge/records/readout-b87c22fbdb500d43981e839b.json), [`readout-cedd207f`](../reports/forge/records/readout-cedd207fc183ede3b4e66e38.json)): first recipe to pass **`4/4` discrete image tasks** including **`img_bars4`** (`4/4` modes, `hq = 0.9688`, `tv = 0.1562`, where both `empf-v8` and `rhbi-v12` collapse to `2/4` modes), reaching `31/41` T1–2 PASS.
- Round-4 Occam's Razor Arm 5 (`-chart-laguerre`, [`readout-b27b2689`](../reports/forge/records/readout-b27b26896bd137e2e701ce2d.json)) and Arm 4 (`-coalesce-ltv`, [`readout-04bb3c21`](../reports/forge/records/readout-04bb3c211150d5629c489b5e.json)) proved Laguerre dual mass-balancing (`7/15` PASS without it) and Pythagorean LTV coalescence (`9/15` PASS without it) are the **most irreducible mechanisms in the entire codebase**.

---

## 2. Law 2 — First-Order Critic Potential Geometry (Zero-`autograd.grad` Born–Infeld Secants)

**Fundamental Obstruction:** Enforcing bounded discriminator sharpness $\|\nabla_x D(x)\|_2 \le \kappa$ via second-order `torch.autograd.grad(..., create_graph=True)` raises critic cost from $\sim 6 F_D$ to $\sim 15\text{–}21 F_D$ and triggers non-fused higher-order convolution backward kernels (`3,113–4,845 ms/step` in `BCap`, `588.75 ms/step` and `6,320 MiB` reserved VRAM in `RHBI`, `130.13–173.05 ms/step` in `EMPF` at `256×256` RGB). Yet naive finite-difference secants $|D(b) - D(a)|/\|b - a\|_2$ fail for three proved mathematical reasons ([`t2-efficient-critic.md`](../reports/forge/studies/t2-efficient-critic.md)):
1. **Symmetric-Flank Chord Cancellation:** Macro chords across equal-height mode flanks satisfy $S(\mu - r u, \mu + r u) = 0$ even as $\|\nabla_x D\|_2 \to \infty$.
2. **High-Dimensional Projection Attenuation:** A single unit probe $\hat{u} \in S^{d-1}$ underestimates $\|\nabla_x D\|_2$ by $\sqrt{d}$ ($13.04\times$ at $d = 170$ on `five_word_joint_smoke`).
3. **Piecewise-Linear Kink Impulses:** Finite differences crossing ReLU/LeakyReLU hyperplanes inject $O(1/\delta)$ parameter-gradient impulses (`param_grad_cosine` drops to `0.22–0.32`).

**First-Principles Resolution (`t2_secant` / `unified_v11`):**
- **Forward-Mode Tangent Reconstruction (`fwAD.dual_level`, manifolds & MLPs):** Evaluates $(D_\theta(x), \nabla_x D_\theta(x) \cdot u)$ at the **exact primal activation mask** (zero kink impulses, zero `create_graph=True`, single 1st-order `.backward()`): exact canonical basis $I_d$ for $1 \le d \le 4$ (with central-difference fallback for quantized MXFP8/NVFP4 ops lacking `jvp`), chained $O(d)$ Householder frames for $5 \le d \le 8$, and RNG-free Murmur3 Rademacher tangents with local $k$-NN Gaussian Hutchinson pooling ($k=32$) for $d > 8$.
- **Live-Endpoint Reuse + Separable BF16 Micro-Secants (Convolutional Image Critics):** Captures the live 1st-order logits $D(x_{\text{real}}), D(x_{\text{fake}})$ from `d_loss` via a forward hook at zero extra forward cost and evaluates a single BF16 micro-secant probe batch alongside free transport-chord and cyclic-roll secants, with $8 \times 8$ pooled Chamfer scale tracking and zero-mean logit gauge (`ebi_beta`).

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- `t2-secant-v2` ([`readout-14cc2a44`](../reports/forge/records/readout-14cc2a44f2d8da942d2873dc.json)): cuts uncontended `dcgan_performance_r256` median latency to **`87.96 ms/step`** (`181.92 img/s`, `1,308.46 MiB` allocated VRAM) — **`1.48×–1.97×` faster than `empf-v8`**, **`3.87×–3.97×` faster than `bispf`**, **`6.69×–6.82×` faster than `rhbi-v12`** (`-72.2%` reserved VRAM), and **`35.4×–55.1×` faster than `bcap`**.
- `t2-secant-v5` ([`readout-a2d41a67`](../reports/forge/records/readout-a2d41a678887a49f9ecc6aa9.json)): clears `11/11` required Tier-1 gates + `clockfree_audit` + `precision_gaussian_mxfp8` + `five_word_joint_hold` with **zero `torch.autograd.grad` calls**.
- Round-4 Occam's Razor Arm 7 (`-fwad-critic`, replacing 1st-order `fwAD`/secants with 2nd-order `autograd.grad` EBI on `unified-v10`, [`readout-4b04c4bf`](../reports/forge/records/readout-4b04c4bf0dc3d7408a82e51e.json), [`readout-b96faf85`](../reports/forge/records/readout-b96faf85499b1dbe9dce102e.json)): proved 1st-order `fwAD`/secant Born–Infeld is **strictly Pareto-dominant** — `1.48×` faster (`89.66` vs `133.01 ms/step`), `-477.7 MiB` (`-34.8%`) less VRAM (`1372.6` vs `1850.3 MiB`), and higher pass rate (`grid100`, `img_bars4`, and `zero_tanh_rail_escape` all regressed under 2nd-order EBI).

---

## 3. Law 3 — Scale–Mass–Fiber Gauge Identifiability (The 3rd Missing Piece)

**Fundamental Obstruction:** Even when **Law 1 is absent** ($G = \text{nn.Identity}()$, **`0` trainable generator parameters**, independent zero-$\sigma$ data-space particles) **and Law 2 is satisfied** (smooth $C^\infty$ `SiLU` critic verified at $\operatorname{med}\|\nabla_x D\|_2 \le 1.0$), a scalar critic potential $D(x)$ paired with a single scalar Euclidean transport horizon $\ell_t$ **still fails `0/2` across `BCap`, `EMPF`, and `RHBI`** ([`t3-third-piece-proof.md`](../reports/forge/studies/t3-third-piece-proof.md)). Scalar potentials with one Euclidean horizon cannot simultaneously identify:
1. **Multi-scale local covariance** under $8\times$ mode-width / $6\times$ mass ratios (`t3_iso_scale_mass_8x`),
2. **Beyond-horizon orphan modes** after $42.4\sigma$ equidistant orthogonal saddle shifts (`t3_iso_horizon_saddle_shift`), and
3. **Conditional base-manifold drift and outer-tail local covariance** on stratified reverse-diffusion chains (`model_ddgan_gaussian_chain`, `model_ddgan_bimodal_chain`, where $n_t \approx 16$ samples per stratum starve outer `t=1, q_0` tail leverage).

**First-Principles Resolution:**
1. **Scale-Invariant Local Horizon & $k=4$ Density-Ratio Orphan Macro-Transport** ([`t3-third-piece-proof.md`](../reports/forge/studies/t3-third-piece-proof.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Compare cross-density directly against local real $k=4$ spacing:
   $$r_{\text{loc}}(y_j) = \frac{d_{k=4}(y_j, X_{\text{fake}})}{d_{k=4}(y_j, Y_{\text{real}} \setminus \{y_j\})}, \qquad g_{\text{orphan}}(y_j) = \operatorname{smoothstep}\!\left(\frac{r_{\text{loc}}(y_j) - 3.52}{1.30}\right),$$
   coupled with exact 1D Weibull batch-horizon scaling (`akt_exp = max(0, 1/d - 0.5)`). Because $r_{\text{loc}}$ is invariant to local mode width $\sigma_m$, tight modes in an $8\times$ scale-mass mixture are never over-pulled, while genuine $42.4\sigma$ equidistant orphan modes trigger full bipartite macro-transport rescue.
2. **Swap-Invariant Zero-`eigh` Cross-Batch Conditional Fiber Operator Bank** ([`r3-conditional-ddgan.md`](../reports/forge/studies/r3-conditional-ddgan.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Per discrete stratum $t$, maintain strictly pre-update dual-timescale sufficient statistics on a piecewise-hinge + bounded-bilinear context basis $\phi(c)$ paired with a $384$-entry paired-context FIFO ring. Using pooled midpoint residuals $r_X = X - m_{\text{pool}}(c)$, $r_Y = Y - m_{\text{pool}}(c)$ guarantees **exact midpoint cancellation** ($\mathbb{E}[r_X r_X^\top - r_Y r_Y^\top \mid c] = \Sigma_X(c) - \Sigma_Y(c)$ with zero mean-bias outer product $\tfrac{1}{4}\Delta\Delta^\top$), while `zero_eigh_symmetric` (branchless closed-form for $d \le 2$, float64 cyclic Jacobi for $d \ge 3$, `< 1.1e-14` agreement with `torch.linalg.eigh`) eliminates `eigh` completely.

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- **Causal Isolation (`t3_causal_isolation_stretch`):** `bcap-pure-v1` `0/2` ([`readout-18e9fbb0`](../reports/forge/records/readout-18e9fbb07a4d8c24ba8e924b.json)), `empf-v8` `0/2` ([`readout-7e0c3159`](../reports/forge/records/readout-7e0c315962234ea1c2c2ba3f.json)), `rhbi-v12` `0/2` ([`readout-8a63e947`](../reports/forge/records/readout-8a63e947313fcb7d2e62a8d2.json)), vs **`rgdt-v3` `2/2` PASS** ([`readout-c30d4a58`](../reports/forge/records/readout-c30d4a5840e8d861f62c1dcb.json)) and **`unified-v11` `2/2` PASS** ([`readout-d1e36bf5`](../reports/forge/records/readout-d1e36bf514f755e46b333e6d.json)), with clean $3 \times 3$ factorial triple dissociation (`readout-efdd1af5`, `readout-e3f9c013`, `readout-8d2fa268`, `readout-b72ef358`).
- **Conditional Reverse-Diffusion Breakthrough:** `cond-fiber-v1` ([`readout-5932f4d3`](../reports/forge/records/readout-5932f4d351cc4e791417835a.json)) delivered the first-ever Forge `PASS` on both `model_ddgan_gaussian_chain` (`38/38` bounds, `9` consecutive terminal passes) and `model_ddgan_bimodal_chain` (`8` consecutive passes), reaching `40/47` T1–2 PASS; `unified-v11` ([`readout-fc5b67a7`](../reports/forge/records/readout-fc5b67a767e57d748f460610.json)) ported the bank to **zero `eigh`** and became the first zero-SVD/zero-`eigh`/zero-`autograd.grad` recipe to pass `model_ddgan_gaussian_chain` (`t1_q0_mean = 0.1790`, `eig_max = 1.3085`, `5/5` suffix).

---

## 4. Law 4 — Function-Space Two-Player Equilibration & Depth/Rail Trust

**Fundamental Obstruction:** The GAN minimax game $\min_G \max_D \mathcal{L}(G_\theta, D_\psi)$ operates on pushforward measures and potential functions, yet parameter-space optimizers (`Adam`, `SGD`) step in parameter coordinates $\|\Delta\theta\|_\infty = \eta$, inducing function-space shifts $\Delta f(u) \sim \Theta(\alpha L \sqrt{W}\,\eta)$:
1. **Player Function-Space Impedance Mismatch & Hostile-Init Deadlock** ([`t4-imbalance-nash.md`](../reports/forge/studies/t4-imbalance-nash.md), [`r3-nash-complete.md`](../reports/forge/studies/r3-nash-complete.md)): Pairing a `512×4` player (`~793k` params) against a `32×2` player (`~1.3k` params — `>600×` parameter ratio) with compressed/amplified gains and `-30`-logit anti-aligned initial tilts causes **4/4 naive-Adam baselines (`BCap`, `EMPF`, `R1/R2`, `RHBI`) to score `0/3`** on `t4_imbalance_nash_stretch`.
2. **Exponential Depth Skew & `tanh`-Rail Dead-Jacobian Trapping** ([`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md)): In a 16-layer unnormalized skip-free `LeakyReLU(0.2)` MLP (`grid100_mlp16`), forward spread contracts $\sim 156\times$ and early gradients attenuate by $10^2\times\text{–}10^6\times$ into Adam's $\varepsilon = 10^{-8}$ dead zone (`0/10` historical families pass). In an unscaled 16-layer residual `tanh` net (`grid100_res16_tanh`) or `zero_tanh_rail_escape`, unscaled skip variance slams logits into `sech^2` saturation rails (`0/10` historical families pass).

**First-Principles Resolution (`T4Nash` / `T4EquilibratedOptimizer` / `unified_v11`):**
1. **Underflow-Safe $10^{-38}$ Dynamic Scale Normalization & Homogeneous `peak_rms` Settling:** Normalize $(m_t, v_{\text{rms},t})$ by $s_t = \max(\max |v_{\text{rms},t-1}|, \max |g_t|, 10^{-38})$ so $10^{-12}$ early-layer and $\operatorname{sech}^2$-attenuated gradients receive the exact same directional geometry as $10^{+2}$ output gradients, while degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock.
2. **Closed-Form Marchenko–Pastur Operator Scaling + Realized Generator Pushforward Trust Band (`delta_g_max`):** Analytically normalize layer operator gain by $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ in $O(\text{params})$ arithmetic, and measure realized generator output displacement $\delta_G = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ on a detached eval-mode probe to clamp `548×` overpowered generators before they overshoot into mode collapse.
3. **Fused $Z = [X, Y]$ Quintic Tikhonov Polar Orthogonalization (2D Deep Skip-Free MLPs) & Zero-`pinv` Analytic `direction_blend`:** Replace $O(\min(m,n)^3)$ CPU/GPU SVD with 5-step pure-GEMM Quintic Newton–Schulz iterations ($a=3.4445, b=-4.7750, c=2.0315$) batched across same-shape matrices, plus exact closed-form $1\times 1 / 2\times 2$ non-ascent constraint projection (`zero_pinv_project_nonascent`, using standard `.backward()` with zero `autograd.grad` and zero `pinv`).

**Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
- **Imbalance-Invariant Nash (`t4_imbalance_nash_stretch`):** `t4-nash-v3` ([`readout-6f996ec2`](../reports/forge/records/readout-6f996ec256a149b2f1620f64.json), [`readout-642d4c55`](../reports/forge/records/readout-642d4c55d093b44ed1e0309e.json)) achieved **`3/3` (`100%`) sustained `5/5` PASS** with **strictly `0` SVD, `0` `eigh`, `0` `pinv`, `0` Newton–Schulz, and `0` `autograd.grad`** at **`39.68 ms/step`** on `256×256` RGB DCGAN (`3.92×` faster than `empf-v8`, `14.84×` faster than `rhbi-v12`), recovering `16/16` modes across a `100×` bidirectional LR-asymmetry sweep (`d_lr_mult` from `10.0` to `0.10`).
- **Pathological 16-Layer & Rail Architectures (`t4-strange-architectures-v1`):** `t4-arch-v3` ([`readout-dc890b70`](../reports/forge/records/readout-dc890b704514d91effb156a6.json), [`readout-4acf96c2`](../reports/forge/records/readout-4acf96c2efad5e17f86d9df2.json)) jumped from `0/10` historical families to **`7/7` (`100%`) PASS** (including `grid100_mlp16` `prec = 0.9814`, `grid100_res16_tanh` `prec = 0.9856, sat = 0.0`, `grid100_res16_scaled_tanh` `prec = 0.9868`, `zero_tanh_rail_escape` `mae = 0.00105`), plus `3/3` fresh-process `14,000`-step native continuations (`grid100_14k`, `rotated100_14k`, `staggered100_14k`, [`readout-ff228439`](../reports/forge/records/readout-ff22843954240927dfd7ecc3.json)).
- `unified-v11` combines both: **`3/3` Nash PASS** ([`readout-2eb07e97`](../reports/forge/records/readout-2eb07e97714a1da4177d5d25.json)), **`6/7` strange-arch PASS** ([`readout-b96d9de4`](../reports/forge/records/readout-b96d9de47bb52cc91fbb4a4f.json)), **`5/5` shift/rail PASS** ([`readout-2029c37a`](../reports/forge/records/readout-2029c37a1359498d3d8f2eb3.json)), and `residual_student` `1.0` (`wrong_pad_rate = 0.0`) via zero-`pinv` `direction_blend`.

---

## 5. Occam's Razor Synthesis: Two Mathematical Halves vs. Redundant Epicycles

Round-4 Track 2 ([`r4-occam-ablation-razor.md`](../reports/forge/studies/r4-occam-ablation-razor.md), `18` concluded single-factor knockout readouts on `unified-v10`) and Track 1 ([`r4-minimal-core.md`](../reports/forge/studies/r4-minimal-core.md), `899`-LOC `minimal-v2`) tested which parts of Laws 1–4 are mathematically irreducible versus redundant epicycles:

| Verdict | Mechanism | Exact Knockout Evidence (Seed `0`) |
| :--- | :--- | :--- |
| **Irreducible Core** | **Laguerre dual mass-balancing & $H_0$ chart commitment** | Removing it (Arm 5, [`readout-b27b2689`](../reports/forge/records/readout-b27b26896bd137e2e701ce2d.json)) drops `14/16 -> 7/15` PASS: collapses `grid100` (`prec = 0.2827`), `img_bars4` (`2/4`), `ring16`, `vector_unequal_width`, and both Nash tasks. |
| **Irreducible Core** | **1st-order `fwAD` / multi-probe secant Born–Infeld critic** | Strictly Pareto-dominant over 2nd-order `autograd.grad` EBI (Arm 7, [`readout-4b04c4bf`](../reports/forge/records/readout-4b04c4bf0dc3d7408a82e51e.json), [`readout-b96faf85`](../reports/forge/records/readout-b96faf85499b1dbe9dce102e.json)): `1.48×` faster, `-34.8%` VRAM, and higher pass rate. |
| **Irreducible Core** | **Pythagorean sibling coalescence + LTV variance conservation** | Removing it (Arm 4, [`readout-04bb3c21`](../reports/forge/records/readout-04bb3c211150d5629c489b5e.json)) drops `14/16 -> 9/15` PASS (`grid100` `center_rms_sigma` worsens `0.1379 -> 0.5312`). |
| **Irreducible (1-Regime)** | **Generator pushforward trust ceiling (`delta_g_max`)** | Removing it (Arm 2, [`readout-4e2fd695`](../reports/forge/records/readout-4e2fd695306a5dd6fd222e62.json)) collapses `548×` `nash_overpowered_generator_ring16` to `6/16` modes (`hq = 0.2522`). **Critic logit secant band is redundant** (`overpowered_critic` & `hostile_imbalance` both PASS without it). |
| **Irreducible (1-Regime)** | **2D Quintic Newton–Schulz on deep skip-free MLPs (`L > 5`)** | **Split verdict** (Arm 1 v2, [`readout-690dd40a`](../reports/forge/records/readout-690dd40a4221b390d8680e43.json), [`readout-0219c6e2`](../reports/forge/records/readout-0219c6e2857b1605cbcf3428.json); independently confirmed by `minimal-v2` [`readout-6bac8895`](../reports/forge/records/readout-6bac8895ffa14ef20dcbee75.json)): strictly required ONLY for 16-layer skip-free `grid100_mlp16` (`prec` drops `0.9929 -> 0.0244–0.0714` without it); **100% redundant** on shallow nets, `grid100`, 16-layer residual `res16_scaled_tanh`, `3/3` Nash, and 4D Conv2d DCGANs (dropping Conv2d NS speeds `r256` from `89.66 -> 76.17 ms/step`). |
| **Irreducible (1-Regime)** | **2D (`d_z=2`) pullback covariance isometry & 3-line 1D Weibull AKT** | `d_z=2` isometry is needed for `grid100` covariance (Arm 3, [`readout-44dbbe8a`](../reports/forge/records/readout-44dbbe8ab156d3615c474db4.json)); 3-line 1D AKT is needed for `gaussian1d_smoke` (Arm 6, [`readout-edffe8ce`](../reports/forge/records/readout-edffe8cee489f8b14768e2cb.json)). |
| **Proven Redundant Epicycles** | **4D Conv2d/shallow NS · Critic logit secant band · `d_z=4` Radon–Hurwitz 4-tangent branch · 2D `_conformal_local_horizons` · Case A `w_sat_pt` saturation probe** | Knocking each out preserves target regimes cleanly (e.g., `zero_tanh_rail_escape` still passes at `mae = 0.0026` without `w_sat_pt` because $10^{-38}$ dynamic scale normalization already inverts `sech^2`; `12/12` 2D/4D/deep/Nash/image probes pass bit-for-bit without 2D `_conformal_local_horizons`). |

### Cross-Champion Summary (Verbatim Committed Forge Receipts, Seed `0`, `NVIDIA RTX A6000`)

| Candidate | Core LOC | `autograd.grad` | SVD/`eigh`/`pinv` | `disc_stab` (T1–2) | `precision` (`fp32/mx8/nv4`) | `ddgan_gauss` | `t4_nash` | `t3_iso` | `t3_shift` | `t4_arch` | `endurance` | `r256` (`ms` · `MiB`) |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unified-v11`** (`DEFAULT_RECIPE`) | `4,713` | **0** | **0** | **`41/47`** | **`3/3`** | **`PASS`** | **`3/3`** | **`2/2`** | **`5/5`** | **`6/7`** | **`3/3`** | **`89.19` · `1372.6`** |
| `unified-v10` | `4,311` | 0 | 0 | `40/46` | `2/3` | `FAIL` | `3/3` | `1/2` | `5/5` | `6/7` | `3/3` | `89.66` · `1372.6` |
| `cond-fiber-v1` *(+`_unified`)* | `4,244` | 0 | `eigh`+SVD | `40/47` | `3/3` | `PASS` | — | — | — | — | — | `87.50` · `1308.4` |
| `t4-nash-v3` | `3,016` | 0 | 0 | `33/46` | — | `FAIL` | `3/3` | — | — | — | — | **`39.68`** · `1372.9` |
| **`minimal-v2`** (`preset="minimal"`) | **`899`** | **0** | **0** *(0 NS)* | `10/17` T1 | `0/3` | — | `2/3` | `1/2` | `2/4` *(u)* | `3/7` *(u)* | — | **`55.50` · `1344.8`** |
| `empf-v8` *(0.10.1 default)* | `1,972` | Yes | Yes | `24/41` | — | — | — | `0/2` | — | — | — | `130.13` · `1435.4` |
| `rhbi-v12` *(0.10.0 default)* | `3,114` | Yes | Yes | `34/41` | — | — | — | `0/2` | — | — | — | `588.75` · `3750.6` |

*(Across all `9` concluded Forge studies in [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md), `unified-v11` passes **`55/62` unique runnable tasks (`88.7%`)**, including **`15/17`** executed Tier-3 `14k` holdout tasks (`grid100_14k` and `rotated100_14k` PASS) and **`3/4`** runnable Tier-4 frontier tasks. `—` denotes no committed readout for that candidate/view.)*

---

## 6. Honest Failure, Calibration & Scope Boundaries

Per repository standards, passing software tests or leading screening does not manufacture scientific qualification or hide remaining failures:

1. **Screening Calibration & Default Adoption Remain `BLOCKED`:** Under frozen [`configs/forge/calibration/criteria-v1.json`](../configs/forge/calibration/criteria-v1.json) (audited in [`r3-calibration-holdout.md`](../reports/forge/studies/r3-calibration-holdout.md) and [`r3-calibration-holdout.json`](../reports/forge/calibration/r3-calibration-holdout.json)), screening calibration is `BLOCKED` because no single global recipe passes `16/16` reference tasks, rev-19's 11-task Tier-1 smoke screen exceeds the `<= 900s` / `<= 0.10` cost ratio bounds, and cheap 3-task smoke screens fail the `<= 0.10` false-accept ceiling across partial solvers. Accordingly, `unified_v11` is promoted as the **current default research selection** (`default_adoption: false` in Forge's selection pin), not a calibrated scientific winner; multi-seed confirmatory promotion and untouched real-data image FID confirmation remain open.
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
- [Round-4 Minimal `<900`-LOC Core Study (`minimal`)](../reports/forge/studies/r4-minimal-core.md)
- [Round-3 Calibration & Holdout Audit](../reports/forge/studies/r3-calibration-holdout.md)
