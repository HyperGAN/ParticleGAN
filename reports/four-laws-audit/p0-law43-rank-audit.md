# P0 Adversarial Mathematical & Empirical Audit: Law 4.3 (Matrix Rank, Scaling Invariance & Newton–Schulz)

> **Audit Scope & Governance:** Adversarial mathematical and computational audit of Law 4.3 (*Tensor-Rank Spectral Collapse vs. Low-Rank Noise Inflation*) on integration branch `research/four-laws` (audited on branch `research/fl-p0-rank`). All experiments execute through the workspace `particlegan` package under protocol seed `0` and `deterministic_orthogonal_` initialization on `NVIDIA RTX A6000` (`cuda:0`, PyTorch `2.15.0+cu132`) and CPU. Existing Forge receipts and `docs/four-laws.md` are left strictly untouched; exact proposed revisions are provided in [Section 6](#6-stronger-corrected-formulation-of-law-43--exact-proposed-diffs).
>
> **Reproducibility:**
> - Experiment runner: [`scripts/four_laws_audit/p0_rank_experiments.py`](../../scripts/four_laws_audit/p0_rank_experiments.py)
> - Automated proofs & regression controls: [`tests/four_laws_audit/test_p0_law43_rank.py`](../../tests/four_laws_audit/test_p0_law43_rank.py)

---

## Executive Verdict

Law 4.3 as currently stated in [`docs/four-laws.md:87,94`](../../docs/four-laws.md#L87-L94) and [`reports/forge/studies/law4-optimizer-laws.md:137,212-216`](../forge/studies/law4-optimizer-laws.md#L212-L216) **commits two fundamental linear-algebraic errors in the opposite directions**, **misnames stable rank as Roy–Vetterli `erank`**, and **misclassifies Kronecker-factored preconditioners (Shampoo) and ParticleGAN's own optimizer implementations**, **YET its underlying empirical discovery is genuine, reproducible, and sharper once properly formulated**:

1. **FALSIFIED — "Elementwise, Frobenius, $L_\infty$, and Kronecker-factored preconditioners preserve gradient rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$)"**:
   - Arbitrary elementwise (Hadamard) scaling $S \odot G$, 1-step Adam with $\varepsilon > 0$ (closed-form $2\times 2$ counterexample: $\det = -4\delta / [(1+\delta)(4+\delta)(2+\delta)^2] < 0$), multi-step Adam ($\beta_2 > 0$, even with $\varepsilon = 0$ and $\beta_1 = 0$), ParticleGAN's smooth SignSGD (`_smooth_sign_unit`), exact SignSGD on rank-2 inputs ($\operatorname{rank} 2 \to n$), and **even ParticleGAN's own `law4_adafactor_mp`** (which adds $\delta = 0.015 \cdot \text{peak} + 10^{-8}$ *after* forming the rank-1 outer product $r c^\top$) **ALL break exact matrix rank** — none of them has the separable form $D_L G D_R$.
   - Conversely, **invertible separable diagonal scaling $D_L G D_R$** (e.g., Shazeer–Stern Adafactor) and **invertible full Kronecker scaling $L G R$** (e.g., Shampoo, K-FAC) **DO** preserve exact rank ($\operatorname{rank}(L G R) = \operatorname{rank}(G)$), **just as Newton–Schulz does**. Crucially, **1-step instantaneous Kronecker Shampoo** $L G R = (G G^\top + s^2 I)^{-1/4} G (G^\top G + s^2 I)^{-1/4}$ is **algebraically identical to Tikhonov-damped polar orthogonalization** $G (G^\top G + s^2 I)^{-1/2}$ (verified to `7.82e-9` L2 error on rotated $\kappa=10^4$ matrices and lifts stable rank `1.000 -> 15.999`!), directly falsifying the claim that Kronecker preconditioners cannot rescue spectral collapse.
2. **FALSIFIED — "Newton–Schulz restores gradient rank / lifts collapsed gradients back to full rank"**:
   - In exact arithmetic, every odd-polynomial Newton–Schulz map $X_{k+1} = p_k(X_k X_k^\top) X_k$ — **including ParticleGAN's fused Tikhonov $Z = [X, Y]$ variant** (`particlegan/_unified_v11/optimizer.py:103-118`) — factors as $X_k = Q_k(X_0 X_0^\top) X_0$, hence $\operatorname{Col}(X_k) \subseteq \operatorname{Col}(X_0)$, $\operatorname{Row}(X_k) \subseteq \operatorname{Row}(X_0)$, and **$\operatorname{rank}(X_k) \le \operatorname{rank}(X_0)$ identically** with zero cross-coupling leakage from $Y = \beta I_m$.
   - Over 5 steps, ParticleGAN's actual step-varying quintic polynomial has zero-slope gain $\prod_{k=0}^4 a_k \approx 492.75$, acting as a **subspace-preserving spectral band-pass equalizer**: it amplifies existing nonzero singular values $\sigma_i / \|G\|_F \gtrsim 10^{-3}$ toward $1$, cannot lift ultra-tiny modes $\sigma_i / \|G\|_F \le 10^{-6}$ to $O(1)$ in 5 steps (unlike exact SVD polar), and **cannot create new rank**.
3. **CRITICAL NUMERICAL & IMPLEMENTATION DISCOVERIES FROM CODE AUDIT**:
   - **BF16 Rounding-Noise Leakage (`77.8%–86.8%` Orthogonal Noise on Rank-1 Inputs on CUDA):** On CUDA for $m \ge 64$ with `smoothing = 0` (`v11_quintic_2d`, `_unified_v11/optimizer.py:133`, entered whenever `stiff_gate == 1.0` sets `s_base = 0`) and $m \ge 32$ (`t4_quintic_2d`, `_t4_arch/optimizer.py:42`), casting to `bfloat16` introduces $O(10^{-3})$ rounding error that 5 quintic iterations amplify to **$\sigma_2/\sigma_1 = 0.535$ (`77.8%` orthogonal subspace leakage in `v11`) and $\sigma_2/\sigma_1 = 0.829$ (`86.8%` leakage in `t4`)** on exact rank-1 $64\times 64$ inputs (and `62.0%` off-singular-basis noise on `grid100_mlp16` mid-layer `net.16.weight`!). In contrast, FP64 leakage is `1.08e-13`, FP32 is `6.01e-5`, and `v11`'s Tikhonov branch runs in FP32 up to $m \le 256$ (`2.24e-5` leakage).
   - **Non-Tikhonov Frobenius-Rescaling Spectral Norm Blowup ($\sigma_1 \to \sqrt{m}$):** In `_unified_v11/optimizer.py:143-150`, when `smoothing == 0`, `quintic_orthogonalize_2d` rescales `out` to `target_frob = sqrt(min(r, c))`. On an exact rank-1 $128\times 128$ matrix, this inflates the top singular value to **$\sigma_1 = \sqrt{128} = 11.3137$** in FP32 (`6.8855` on CUDA BF16) instead of $\le 1.0$, whereas the Tikhonov branch (`L116`, `joint_frob`) stays bounded at $\sigma_1 = 0.9393$.
   - **Four Documentation-vs-Code Mismatches Caught:**
     1. `docs/four-laws.md:162` states `unified_v11`/`v12` use stationary coefficients `(3.4445, -4.7750, 2.0315)`; actual `_unified_v4/optimizer.py:24-30` (imported by `v11`, `v12`, and `law4_energy`) uses 5 **step-varying** triples `((4.0848,-6.8946,2.9270), ..., (2.8366,-3.0525,1.2012))` (only `_t4_arch/optimizer.py:34-36` and `optim/convolution_quintic.py:13` use stationary `(3.4445,-4.7750,2.0315)`).
     2. `_unified_v11/optimizer.py:70,162` docstrings state `"BF16 for 65..128"` in Tikhonov NS; actual code (`L102,194`) runs **FP32 for $m \le 256$** and BF16 only for $256 < m \le 512$.
     3. `docs/four-laws.md:169` and `law4-optimizer-laws.md:397` describe `unified_v12`'s CountSketch as `"CountSketch spectral preconditioning"`; actual `_unified_v12/optimizer.py:57-74,126-170` is purely a **scalar period-2 sign-flip controller** modifying `_ema_func_gain` (zero effect on matrix rank or singular-value ratios).
     4. `particlegan/_t1_coupling/deconflict.py:9` claims `"pure FP32 coupled Newton-Schulz iterations"`; actual code (`L24`) calls `torch.linalg.eigh`.
4. **EMPIRICAL LAW 4.3 DISCOVERY CONFIRMED & SHARPENED (Why NS Works & Why Adam/SignSGD Fail Despite Breaking Exact Rank):**
   - On actual `grid100_mlp16` (`SimpleMLPGenerator(2, 128, 16, 2)`, `deterministic_orthogonal_(seed=0)`), all 15 hidden $128\times 128$ gradient matrices have **high exact/numerical rank (`rank(1e-12) = 123–128`)** — they are **NOT** exact rank 1 — **yet suffer severe depth-wise spectral anisotropy** (`s2/s1` decays from `0.312` at layer 2 to `1.6e-8` at layer 11+, median stable rank `1.00`, Roy–Vetterli `erank = 1.00`, norm skew `27,170×`), sharply contrasting with shallow `grid100` (`s_rk = 1.16, rv = 3.81`), skip-connected `grid100_res16_tanh` (`s_rk = 1.50, rv = 5.70`), and 4D Conv2d DCGAN per-offset slices (`s_rk = 5.14, rv = 97.98`).
   - On `grid100_mlp16`'s early-mid layer (`net.6.weight`, `s2/s1 = 0.10`, `s_rk = 1.01`), **FP32 5-step NS (`s=0`) and actual per-layer Tikhonov NS (`v11`/`v12`) lift stable rank to `10.08` / `3.84` / `3.02` and allocate `92.4%` / `73.9%` / `66.9%` of update energy directly onto the true subdominant singular modes (`cos_sub16 = 0.939 / 0.678 / 0.618`, `0.0%` off-basis energy)**.
   - By contrast, even though **Adam, Smooth SignSGD, and ParticleGAN Adafactor DO increase exact rank**, their updates on the same spectrally collapsed layer remain trapped at **`s_rk = 1.06–1.17`**, put only **`1.5%–6.1%` of update energy into the true subdominant singular modes (`cos_sub16 = 0.074–0.182`)**, and dump **`45.3%–68.9%` into off-singular-basis coordinate distortion**. **That — spectral-basis-aligned equalization vs. coordinate-aligned Hadamard distortion — is the true mathematical mechanism behind Law 4.3.**

---

## 1. Exhaustive Claim Audit & Actual Optimizer Inventory

### 1.1 Exact Claims Audited Across Documents & Reports

| ID | Source File & Exact Lines | Verbatim Claim (Key Excerpt) | Taxonomy Verdict |
| :--- | :--- | :--- | :--- |
| **C1** | [`docs/four-laws.md:87`](../../docs/four-laws.md#L87), [`law4-optimizer-laws.md:137,214`](../forge/studies/law4-optimizer-laws.md#L214) | *"Elementwise, Frobenius, $L_\infty$, and Kronecker-factored preconditioners preserve gradient rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$)"* | **Falsified claim** (conflates Hadamard/$L_\infty$ with separable $D_L G D_R$; misattributes rank-preserving formula to elementwise & $L_\infty$ maps) |
| **C2** | [`docs/four-laws.md:87`](../../docs/four-laws.md#L87), [`law4-optimizer-laws.md:137,214`](../forge/studies/law4-optimizer-laws.md#L214) | *"...and therefore cannot rescue deep skip-free dense layers whose gradient effective rank collapses ($\operatorname{erank}(G) \to 1$)."* | **Falsified claim** (causal explanation wrong for elementwise/SignSGD; false for full Kronecker/Shampoo, which *does* rescue effective-rank collapse; empirically valid *only* as a failure of Frobenius, diagonal Adafactor, Adam, and SignSGD on `grid100_mlp16` for distinct mechanisms) |
| **C3** | [`docs/four-laws.md:94`](../../docs/four-laws.md#L94), [`law4-optimizer-laws.md:137`](../forge/studies/law4-optimizer-laws.md#L137) | *"Restore gradient rank where diagonal conditioning hits the rank-preservation limit"* / *"Only non-diagonal spectral polar orthogonalization lifts collapsed gradients back to full rank."* | **Falsified claim** (exact-arithmetic polar/NS satisfies $\operatorname{rank}(X_k) \le \operatorname{rank}(G)$ and can never lift exact rank; true mechanism is lifting **effective/stable rank** on already high-exact-rank anisotropic gradients) |
| **C4** | [`law4-optimizer-laws.md:56,213`](../forge/studies/law4-optimizer-laws.md#L213), [`law4_optimizer_profiler.py:757`](../../experiments/law4_optimizer_profiler.py#L757) | Defines *"effective rank $\operatorname{erank}(G_l) = \lVert G_l\rVert_F^2 / \sigma_1(G_l)^2$"* | **Falsified claim** (terminological error: $\lVert G\rVert_F^2/\sigma_1^2$ is **stable rank**, strictly $\le$ Roy–Vetterli entropy `erank` $\exp(H(\sigma/\lVert\sigma\rVert_1))$ and can differ by $>20\times$) |
| **C5** | [`docs/four-laws.md:162`](../../docs/four-laws.md#L162) | *"Replaces $O(\min(m,n)^3)$ SVD with 5-step pure-GEMM Quintic Newton–Schulz iterations ($a=3.4445, b=-4.7750, c=2.0315$)..."* (describing `unified_v11`/`v12`) | **Falsified claim** (actual `_unified_v4`/`v11`/`v12`/`law4_energy` use 5 step-varying coefficients; stationary `(3.4445,-4.7750,2.0315)` is used only in `_t4_arch` and `optim/convolution_quintic.py`) |
| **C6** | [`docs/four-laws.md:169`](../../docs/four-laws.md#L169), [`law4-optimizer-laws.md:397`](../forge/studies/law4-optimizer-laws.md#L397), [`law4-energy-champion.md:124`](../forge/studies/law4-energy-champion.md#L124) | *"CountSketch spectral preconditioning on stiff deep 2D generators"* | **Falsified claim** (`_unified_v12/optimizer.py:57-74,126-170` is a scalar period-2 sign-flip gain controller, not a spectral preconditioner) |
| **C7** | [`docs/four-laws.md:87,185`](../../docs/four-laws.md#L185), [`r4-occam-ablation-razor.md:23,104-106`](../forge/studies/r4-occam-ablation-razor.md#L104-L106), [`r4-minimal-core.md:20`](../forge/studies/r4-minimal-core.md#L20) | 16-layer skip-free MLP (`grid100_mlp16`) suffers severe spectral collapse where diagonal/elementwise conditioning collapses (`prec=0.024–0.255`), while shallow MLPs, 16-layer residual MLPs, and 4D Conv2d kernels do not collapse and do not need NS | **Empirically supported causal mechanism** (verified in Forge knockouts + our direct architectural spectrum & alignment measurements; scoped to seed 0 & tested architectures) |
| **C8** | [`docs/four-laws.md:87`](../../docs/four-laws.md#L87), [`law4-optimizer-laws.md:215,296-298`](../forge/studies/law4-optimizer-laws.md#L296-L298), [`law4-energy-champion.md:65-68`](../forge/studies/law4-energy-champion.md#L65-L68) | Applying 5-step NS to naturally low-rank equilibrium critic gradients inflates orthogonal noise (`3.8×` agitation), whereas 0-NS polar-Frobenius-matched $\sqrt{r}$ updates cut agitation & energy (`-40.8%–64.0%` J/step on Conv2d DCGAN) | **Empirically supported causal mechanism** (and mechanically reinforced by our BF16 rounding-noise discovery when `s -> 0`) |
| **C9** | [`t4-strange-architectures.md:25-26`](../forge/studies/t4-strange-architectures.md#L25-L26) | Skip-free $L=16$ `LeakyReLU(0.2)` orthogonal MLP contracts forward output spread by $0.52^8 \approx 1/187$ and attenuates early-layer gradients by $10^2\times$–$10^6\times$ | **Empirically supported causal mechanism** *(exact only for fixed-input expected squared-norm/RMS contraction $0.52^{L/2}=0.52^8$ under symmetric zero-mean half-space gating; batch spread/variance and `27,170×` inter-layer gradient skew are empirical — see Claim 7)* |
| **C10** | [`particlegan/_unified_v11/optimizer.py:70,162`](../../particlegan/_unified_v11/optimizer.py#L70) vs `L102,194`; [`_t1_coupling/deconflict.py:9`](../../particlegan/_t1_coupling/deconflict.py#L9) vs `L24` | Code comments claiming `"BF16 for 65..128"` in Tikhonov NS and `"pure FP32 coupled Newton-Schulz"` in `deconflict.py` | **Falsified claim** (implementation uses FP32 for $m \le 256$ in Tikhonov NS, BF16 for $m \ge 64$ in non-Tikhonov NS, and `torch.linalg.eigh` in `deconflict.py:24`) |

### 1.2 Verified Ground-Truth Inventory of All ParticleGAN Optimizers

Every optimizer under `particlegan/` (plus companion study branches `research/law4-alt-optimizers` and `research/law4-sgd-anatomy`) was inspected directly in source before assigning mathematical properties:

| Optimizer / Module | Exact NS / Polar Formula & Coeffs | Tikhonov / Smoothing | Momentum / EMA / WD | Dtype & Special Mechanisms | Preserves Exact Rank? (Exact Arith.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`UnifiedV11EquilibratedOptimizer`** (`_unified_v11/optimizer.py`) | 5-step **step-varying** `_QUINTIC_COEFFS` (`_unified_v4/optimizer.py:24-30`), fused `Z=[X,Y]` (`L106-110`); 2D & 4D Conv; critic Schatten-6 decay (`L255-332`) | `s_base = 0.0061*(1-stiff_gate)`, `s_eff = min(s_base, 0.30*peak_rms)` (`L805,994`) | `betas=(0,0.999)` default; activates `b1_eff=0.75*stiff_gate` when `ema_fg<0.18`; `weight_decay=0` enforced (`L465`) | Tikhonov: FP32 `m<=256`, BF16 `257..512` (`L102`); `s=0`: BF16 on CUDA `m>=64` (`L133`); `1e-38` scale; `target_frob/out_frob` rescale (`L150`) | **Yes** (`<= rank(G)`) per step when `b1_eff=0`; **No** across steps when `stiff_gate>0` (`b1_eff>0`) |
| **`UnifiedV12Optimizer`** (`_unified_v12/optimizer.py`, `DEFAULT_RECIPE`) | Same 5-step `_QUINTIC_COEFFS` fused `Z=[X,Y]` on **2D Gen only** (plus 2D Critic if `generator_free`); **0 Conv2d NS**, **0 neural-G Critic NS** (polar-matched Frob-MP `sqrt(r)`) | `s_base = 0.0061*(1-stiff_gate)`, `s_eff = min(s_base, 0.50*peak_rms)` (`L337,762`) | `betas=(0,0.999)` default; `b1_eff=0.75*stiff_gate` when `ema_fg<0.60`; `weight_decay=0` enforced | Prime-67 CountSketch (`L57-74,126-170`) = scalar period-2 flip gain controller; zero-sync foreach Frob-MP fast path; `1e-38` slow path | **Yes** (`<= rank(G)`) per step when `b1_eff=0`; **No** across steps when `stiff_gate>0` |
| **`UnifiedV4EquilibratedOptimizer`** (`_unified_v4/optimizer.py`) | 5-step `_QUINTIC_COEFFS`, unfused `(x, y)` Tikhonov (`L145-150`); `deep_gate` for `L>5` | `s_eff = smoothing*(1-deep_gate)` | `b1_eff = 0.75*deep_gate` (or `0.82` cond); `weight_decay=0` | BF16 on CUDA for **all** `m>=2` (`L142,173`) | **Yes** when `b1_eff=0`; **No** when `b1_eff>0` |
| **`T4EquilibratedOptimizer`** (`_t4_arch/optimizer.py`) | 5-step **stationary** `(3.4445,-4.7750,2.0315)` (`L34-36`), no Tikhonov `Y` | None (`smoothing=0`) | `betas=(0.80,0.97)` default (`b1_eff>=0.75` for `L>=2`) | BF16 on CUDA for `min>=32` (2D) / `>=16` (batched); `1e-38` dynamic moments | **No** (due to `b1_eff>=0.75` momentum); single-step NS itself preserves `<= rank(M)` |
| **`Law4EnergyV1..V11Optimizer`** (`_law4_energy/optimizer.py`) | 0 Conv2d NS; v2 adds `exact_polar_2xn` (`L784-822`); v7-11 drop neural-G Critic NS; v8 adds Renyi-2 `erank_2` gate (`L4162`); v9 uses `_smooth_sign_unit` (`L3575`) on Critic | `0.03*peak_rms` (v1-v6) / `0.50*peak_rms` (v7-v11) | Same as `v11`/`v12`; `weight_decay=0` | `exact_polar_2xn` computes 2×2 Gram in FP64 then casts `inv` to FP32 (`L813-815`) | **Yes** for Frob-MP / NS / `exact_polar_2xn` (`b1=0`); **No** for v9 `_smooth_sign_unit` |
| **`Minimal{Generator,Critic}Optimizer`** (`_minimal/dynamics.py:147-202`) | **0 NS**: elementwise Adam-style `m_hat / (v_hat + 0.015*peak + eps)` + MP scaling | `0.015*peak_rms + eps` additive floor | `betas=(0,0.999)` (or `b1=0.75*deep_g` for `L>5`) | FP32 + `1e-38` dynamic rescaling | **No** (Hadamard division with additive floor & EMA `v_t` breaks exact rank!) |
| **`R4MPNoNSOptimizer`** (`_r4_ablation/optimizer.py:28-395`) | **0 NS**: elementwise `m_hat / (v_hat + 0.015*peak_scaled + 1e-8)` + MP scaling | `0.015*peak_scaled + 1e-8` + scalar Tikhonov | `b1_eff = 0.75*deep_gate` | FP32 + `1e-38` dynamic rescaling | **No** (Hadamard division breaks exact rank!) |
| **`T4Nash` / `NashV3` Adam** (`_t4_nash/optim.py`, `_nash_v3/optim.py`) | **0 NS**: elementwise `m_hat / (v_hat_sqrt + 0.015*peak + eps)` + MP scaling | `0.015*peak + eps` | `betas=(0,0.999)` | FP32 | **No** (Hadamard division breaks exact rank!) |
| **`NormalizedOptimizer` (`dualnorm`)** (`optim/dualnorm.py`, `convolution_quintic.py`, `gram_polar.py`) | Exact SVD `polar_factor` (`tau=max(r,c)*eps*s1`), 32-step cubic NS `_newton_schulz_polar`, FP64 `guarded_gram_polar`, or 5-step stationary `(3.4445,-4.7750,2.0315)` `quintic_direction` | Absolute `smoothing` via `[M; s I]` | `momentum in {0, 0.5, 0.9}`; `weight_decay=0` | `frobenius_direction` (`M/hypot(\|\|M\|\|_F, s)`); `deconflict.py` uses `eigh` (`L24`) for `polar(G P) P` | **Yes** (`<= rank(G)`) for zero-momentum polar / NS / Frobenius / `polar(G P) P` |
| **Study Arms (`research/law4-alt-optimizers`, `research/law4-sgd-anatomy`)** | `law4_adam_equil` (Hadamard Adam+`0.015*peak`), `law4_sign_mp` (`g/hypot(|g|, s)`), `law4_adafactor_mp` (`g/(v_fact + 0.015*peak + 1e-8)`), `law4_frob_sgd_mp` (`g/hypot(||g||_F, s)`) | Per-arm `peak_rms` + scalar Tikhonov | `b1=0` (except `muon_m`, `lion_mp`, `optimistic`) | All `1e-38`-safe FP32 | **Only `law4_frob_sgd_mp` preserves exact rank!** `adam_equil`, `sign_mp`, `lion_mp`, and `law4_adafactor_mp` ALL break exact rank! |

---

## 2. Formal Mathematical Proofs & Explicit Counterexamples

### 2.1 Theorem 1 — Elementwise (Hadamard) Scaling Does NOT Preserve Exact Matrix Rank

**Proposition 1 (Hadamard Rank Inflation):** Let $A, B \in \mathbb{R}^{m \times n}$ and let $A \odot B$ denote their elementwise (Hadamard) product. In general, $\operatorname{rank}(A \odot B) \le \operatorname{rank}(A)\operatorname{rank}(B)$, and strict rank increase $\operatorname{rank}(A \odot B) > \operatorname{rank}(A)$ occurs whenever $\operatorname{rank}(B) > 1$ is non-degenerate on $\operatorname{supp}(A)$.

**Minimal $2\times 2$ Counterexample:**

$$
A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \quad (\operatorname{rank} = 1), \qquad B = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix} \implies A \odot B = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}, \quad \det(A \odot B) = 1 \neq 0 \implies \operatorname{rank}(A \odot B) = 2.
$$

**Corollary 1A (1-Step Adam with $\varepsilon > 0$ Breaks Rank-1):** Let $G = u v^\top$ with $u = v = [1, 2]^\top$ ($\operatorname{rank}(G) = 1$). At step 1 with any additive denominator floor $\delta > 0$ (whether PyTorch Adam's $\varepsilon = 10^{-8}$ or ParticleGAN's $0.015 \cdot \operatorname{RMS} + 10^{-8}$),

$$
U(\delta) = G \oslash (|G| + \delta) = \begin{bmatrix} \frac{1}{1+\delta} & \frac{2}{2+\delta} \\ \frac{2}{2+\delta} & \frac{4}{4+\delta} \end{bmatrix}, \qquad \det U(\delta) = \frac{-4\delta}{(1+\delta)(4+\delta)(2+\delta)^2} < 0 \quad \forall\,\delta > 0.
$$

Hence $\operatorname{rank}(U(\delta)) = 2 > \operatorname{rank}(G) = 1$ for *every* $\delta > 0$. More generally, for $u \in \mathbb{R}^m_{>0}, v \in \mathbb{R}^n_{>0}$ with distinct entries, $U_{ij} = \frac{u_i v_j}{u_i v_j + \delta} = u_i \left(\frac{1}{u_i + \delta/v_j}\right)$ is a diagonally scaled Cauchy matrix, which is **nonsingular (exact full rank $\min(m, n)$)**!

**Corollary 1B (Multi-Step Adam Breaks Rank-1 Even With $\varepsilon = 0$ and $\beta_1 = 0$):** Let $G_1 = u_1 v_1^\top$ and $G_2 = u_2 v_2^\top$ be two independent rank-1 gradients. At step 2 with $\beta_1 = 0, \beta_2 \in (0, 1), \varepsilon = 0$, the second-moment EMA $V_2 \propto \beta_2 (u_1^{\odot 2})(v_1^{\odot 2})^\top + (1-\beta_2)(u_2^{\odot 2})(v_2^{\odot 2})^\top$ has $\operatorname{rank}(V_2) = 2$. Hence $V_2^{\odot -1/2}$ is non-separable ($\operatorname{rank} > 1$) and $U_2 = G_2 \oslash \sqrt{V_2}$ generically has $\operatorname{rank}(U_2) > 1$ (exact full rank in exact arithmetic, `rank(1e-12) = 3` / `rank(1e-6) = 2` on $16\times 16$ at $\beta_2 = 0.999$ in Exp 1) despite $\operatorname{rank}(G_2) = 1$.

**Corollary 1C (Exact SignSGD vs. Smooth SignSGD):**
- On exact rank-1 $G = u v^\top$, **exact** $\operatorname{sign}(u v^\top) = \operatorname{sign}(u)\operatorname{sign}(v)^\top$ preserves $\operatorname{rank} \le 1$.
- On **exact rank-2** $G_{ij} = j - i + \tfrac{1}{2}$ ($i, j \in \{0, \dots, n-1\}$), $G = \mathbf{1}(j+\tfrac{1}{2})^\top - i\mathbf{1}^\top$ has exact rank $2$, yet $\operatorname{sign}(G)$ is upper-triangular $+1$ and strictly lower-triangular $-1$ with $|\det(\operatorname{sign}(G))| = 2^{n-1} \neq 0$ — **exact full rank $n$**!
- **Smooth SignSGD** (`_smooth_sign_unit`, $U_{ij} = G_{ij}/\sqrt{G_{ij}^2 + s^2}$ with $s > 0$) breaks rank even on **rank-1** $G = u v^\top$: for $u = v = [1, 2]^\top, s = 1$, $\det(U) = 4(\tfrac{1}{\sqrt{34}} - \tfrac{1}{5}) \neq 0 \implies \operatorname{rank}(U) = 2$.

**Corollary 1D (Adafactor Separability Split):** Standard Shazeer–Stern Adafactor adds $\varepsilon_1$ *before* row/col averaging, keeping $V_{ij} = r_i c_j / \mathrm{rms}(r)$ rank-1 separable, so $G \oslash V = \mathrm{rms}(r) D_r^{-1} G D_c^{-1}$ **preserves exact rank**. In contrast, ParticleGAN's `law4_adafactor_mp` (`research/law4-alt-optimizers:particlegan/_law4_alt/optimizer.py:1524`) divides by $V_{ij} + \delta$ ($\delta = 0.015 \cdot \text{peak} + 10^{-8}$), turning $1/(r_i c_j + \delta)$ into a full-rank Cauchy matrix and **breaking exact rank** (`1 -> 6` at `tol=1e-12`).

### 2.2 Theorem 2 — Invertible Separable & Kronecker Scaling: Exact Rank vs. Effective Rank

**Theorem 2 (Exact Rank Preservation, Effective Rank Non-Preservation, and Shampoo–Polar Equivalence):**
Let $L \in \mathrm{GL}_m(\mathbb{R})$ and $R \in \mathrm{GL}_n(\mathbb{R})$ be arbitrary invertible matrices. Then:
**1. Exact Rank Invariance:** $\operatorname{rank}(L G R) = \operatorname{rank}(G)$ for all $G \in \mathbb{R}^{m \times n}$ (since $L$ and $R$ are bijective linear maps). This holds for SGD ($L=I, R=I$), scalar/Frobenius-normalized SGD ($c G, c \ne 0$), diagonal separable scaling $D_L G D_R$ (Shazeer–Stern Adafactor), and full Kronecker preconditioning $L G R$ (Shampoo, K-FAC).

**2. Scalar/Frobenius Normalization Preserves the Entire Normalized Spectrum:** For $c > 0$, $\sigma_i(c G) = c \sigma_i(G)$, so exact rank, stable rank, participation ratio, Roy–Vetterli `erank`, and $\kappa(G)$ are all strictly invariant.

**3. Diagonal $D_L G D_R$ Does NOT Preserve Effective Rank — Axis-Aligned vs. Rotated Dichotomy:**
- If $G = \operatorname{diag}(\sigma_1, \dots, \sigma_m)$ is axis-aligned with $\sigma_1 = 1, \sigma_{2..m} = 10^{-4}$ ($r_{\text{stable}} = 1.000$), then $D_L = \operatorname{diag}(\sigma_i^{-1}), D_R = I$ yields $D_L G D_R = I_m$ ($r_{\text{stable}} = m = 16.000$); even Shazeer–Stern Adafactor yields $r_{\text{stable}} = 15.000$ (by squaring the diagonal scale).
- If $G = Q_1 \operatorname{diag}(\sigma) Q_2^\top$ has **dense/rotated singular vectors** (as in deep MLPs), row and column norms of $G$ concentrate around $\|G\|_F/\sqrt{m}$ and $\|G\|_F/\sqrt{n}$, so diagonal $D_L, D_R \approx c I$ cannot equalize rotated singular values (Shazeer–Stern Adafactor keeps $r_{\text{stable}} = 1.000$ on rotated $\kappa = 10^4$).

**4. Full Kronecker 1-Step Shampoo $\equiv$ Exact Tikhonov Polar Orthogonalization:** Let $G = U \Sigma V^\top$ be the full SVD of $G \in \mathbb{R}^{m \times n}$ ($U \in O(m)$, $V \in O(n)$, $\Sigma \in \mathbb{R}^{m \times n}$ with $\Sigma_{ii} = \sigma_i$ for $1 \le i \le \min(m, n)$), and let $L = (G G^\top + s^2 I_m)^{-1/4}$ and $R = (G^\top G + s^2 I_n)^{-1/4}$ ($s > 0$). Then:

$$
L G R = U (\Sigma \Sigma^\top + s^2 I_m)^{-1/4} \Sigma (\Sigma^\top \Sigma + s^2 I_n)^{-1/4} V^\top = U \Sigma_{\text{tikh}} V^\top = G (G^\top G + s^2 I_n)^{-1/2},
$$

where $\Sigma_{\text{tikh}} \in \mathbb{R}^{m \times n}$ has $(\Sigma_{\text{tikh}})_{ii} = \sigma_i / \sqrt{\sigma_i^2 + s^2}$ for $1 \le i \le \min(m, n)$ and $0$ elsewhere.
Therefore, **full Kronecker preconditioning preserves exact rank AND completely rescues effective-rank collapse** (`r_stable: 1.000 -> 15.999` on rotated $\kappa=10^4$, L2 distance to exact Tikhonov polar `7.82e-9` in FP64)!

### 2.3 Theorem 3 — Exact-Arithmetic Subspace & Rank Invariance of Polynomial & Fused Tikhonov `Z = [X, Y]` Newton–Schulz

**Theorem 3 (Subspace & Rank Invariance of Standard and Fused `Z=[X, Y]` Quintic NS):**
Let $G \in \mathbb{R}^{m \times n}$ with $m \le n$ (tall matrices $m > n$ are transposed before iteration and transposed back) have **thin SVD** $G = U \Sigma V^\top$ where $U \in O(m)$, $V \in \mathbb{R}^{n \times m}$ ($V^\top V = I_m$), and $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_m) \in \mathbb{R}^{m \times m}$ with $\sigma_1 \ge \dots \ge \sigma_r > 0 = \sigma_{r+1} = \dots = \sigma_m$ ($\operatorname{rank}(G) = r$). Let $p_k(t) = a_k + b_k t + c_k t^2$ be any sequence of polynomials.

**1. Standard Polynomial NS** ($X_0 = \alpha G$, $X_{k+1} = p_k(X_k X_k^\top) X_k$): By induction (using $V^\top V = I_m$), $X_k = U \operatorname{diag}(\sigma_i^{(k)}) V^\top \in \mathbb{R}^{m \times n}$ where $\sigma_i^{(0)} = \alpha \sigma_i$ and $\sigma_i^{(k+1)} = p_k((\sigma_i^{(k)})^2)\,\sigma_i^{(k)}$. Since $\sigma_i = 0 \implies \sigma_i^{(k)} = 0$ for all $k \ge 0$:

$$
\operatorname{Col}(X_k) \subseteq \operatorname{Col}(G), \qquad \operatorname{Row}(X_k) \subseteq \operatorname{Row}(G), \qquad \operatorname{rank}(X_k) \le \operatorname{rank}(G).
$$

**2. Fused Tikhonov `Z = [X, Y]` NS (`particlegan/_unified_v11/optimizer.py:103-118`):** Initialize $Z_0 = [X_0, Y_0]$ with $X_0 = \alpha G \in \mathbb{R}^{m \times n}$ and $Y_0 = \beta I_m \in \mathbb{R}^{m \times m}$ ($\beta = \alpha s \ge 0$), and iterate $Z_{k+1} = p_k(Z_k Z_k^\top) Z_k = [X_{k+1}, Y_{k+1}]$.

*Proof by induction:* At $k = 0$, $X_0 = U \operatorname{diag}(d_i^{(0)} \alpha \sigma_i) V^\top$ (thin $V \in \mathbb{R}^{n \times m}$) and $Y_0 = U \operatorname{diag}(d_i^{(0)} \beta) U^\top$ with $d_i^{(0)} = 1$. Assume the form holds at step $k$. Using $V^\top V = I_m$ and $U^\top U = I_m$:

$$
Z_k Z_k^\top = X_k X_k^\top + Y_k Y_k^\top = U \operatorname{diag}\!\Big((d_i^{(k)})^2 \big(\alpha^2 \sigma_i^2 + \beta^2\big)\Big) U^\top.
$$

Therefore $p_k(Z_k Z_k^\top) = U \operatorname{diag}\!\big(\gamma_i^{(k)}\big) U^\top$ where $\gamma_i^{(k)} = p_k\!\big((d_i^{(k)})^2(\alpha^2\sigma_i^2 + \beta^2)\big)$. Setting $d_i^{(k+1)} = \gamma_i^{(k)} d_i^{(k)}$ yields

$$
X_{k+1} = U \operatorname{diag}\!\big(d_i^{(k+1)} \alpha \sigma_i\big) V^\top, \qquad Y_{k+1} = U \operatorname{diag}\!\big(d_i^{(k+1)} \beta\big) U^\top,
$$

completing the induction.

*Exact consequences:*
- **Zero cross-coupling leakage:** Every $\sigma_i = 0$ stays identically zero in $X_k$ for all $k$: $\operatorname{Col}(X_k) \subseteq \operatorname{Col}(G)$, $\operatorname{Row}(X_k) \subseteq \operatorname{Row}(G)$, and **$\operatorname{rank}(X_k) \le \operatorname{rank}(G)$**.
- **Exact Tikhonov limit:** Setting $r_i^{(k)} = d_i^{(k)}\sqrt{\alpha^2 \sigma_i^2 + \beta^2}$ gives the scalar recursion $r_i^{(k+1)} = p_k((r_i^{(k)})^2) r_i^{(k)}$. Whenever $r_i^{(k)} \to 1$, $X_k \to U \operatorname{diag}\!\big(\sigma_i / \sqrt{\sigma_i^2 + s^2}\big) V^\top = G(G^\top G + s^2 I_n)^{-1/2}$.
- **Finite 5-step gain ceiling:** For $s = 0$ and small $\sigma_i^{(0)}$, $\sigma_i^{(5)} = \left(\prod_{k=0}^4 a_k\right) \sigma_i^{(0)} + O((\sigma_i^{(0)})^3)$. For ParticleGAN's `_QUINTIC_COEFFS`, $\prod_{k=0}^4 a_k = 492.75$ (stationary `(3.4445,-4.7750,2.0315)`: $a^5 = 484.88$). Hence 5-step NS lifts $\sigma_i / \|G\|_F \in [10^{-3}, 1]$ to $[0.47, 1.02]$, but **cannot lift $\sigma_i / \|G\|_F \le 10^{-6}$ above $5\times 10^{-4}$** in exact arithmetic!

### 2.4 Rigorous Hierarchy of Rank & Spectral Diversity Measures

For any nonzero $X \in \mathbb{R}^{m \times n}$ with singular values $\sigma_1 \ge \dots \ge \sigma_r > 0$, define $p_i = \sigma_i / \sum_{j=1}^r \sigma_j$ and $q_i = \sigma_i^2 / \sum_{j=1}^r \sigma_j^2$:

$$
1 \;\le\; \underbrace{\frac{\|X\|_F^2}{\sigma_1^2}}_{r_{\text{stable}}(X)} \;\le\; \underbrace{\frac{\|X\|_F^4}{\|X X^\top\|_F^2}}_{\operatorname{PR}(X) = \exp(H_2(q))} \;\le\; \exp(H_1(q)) \;\le\; \underbrace{\exp(H_1(p))}_{\operatorname{erank}_{\text{RV}}(X)} \;\le\; \operatorname{rank}(X) \;\le\; \min(m, n).
$$

*Proof:* $\sum \sigma_i^4 \le \sigma_1^2 \sum \sigma_i^2 \implies r_{\text{stable}} \le \operatorname{PR}$; Rényi entropy monotonicity gives $H_2(q) \le H_1(q)$; since $p_1 \ge \dots \ge p_r > 0$ and $\sum_{i=1}^k q_i - \sum_{i=1}^k p_i = (\sum_{j=1}^r p_j^2)^{-1} \sum_{i=1}^k \sum_{j=k+1}^r p_i p_j (p_i - p_j) \ge 0$ for all $k$, $q$ majorizes $p$ ($q \succ p$), so Schur-concavity of Shannon entropy gives $H_1(q) \le H_1(p) \le \ln r$. $\square$

- **Why the naming error in `law4-optimizer-laws.md` matters:** On $\sigma = (1, 10^{-2}, \dots, 10^{-2})$ (100 copies of $10^{-2}$), $r_{\text{stable}} = 1.01$ while Roy–Vetterli $\operatorname{erank}_{\text{RV}} = 20.0$ (and on `grid100_mlp16` `net.2.weight`, $r_{\text{stable}} = 1.14$ while $\operatorname{erank}_{\text{RV}} = 3.66$; on `dcgan_G` `main.3.weight`, per-offset median $r_{\text{stable}} = 3.05\text{–}3.06$ [`3.055`, flattened `3.07`] while $\operatorname{erank}_{\text{RV}} = 141.09$!). Distinct metrics must never share the symbol `erank`.

---

## 3. Computational Verification Results

All numbers below are verbatim outputs of `PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p0_rank_experiments.py --device cuda:0` (seed `0`, `RTX A6000`).

### 3.1 Experiment 1 — Exact Rank Counterexamples & Diagonal vs. Kronecker Dichotomy ($16\times 16$, FP64)

| Transform on Rank-1 $G_1 = u v^\top$ (or Rank-2 $G_2$) | `rank(1e-12)` | `rank(1e-6)` | `stable_rank` | `rv_erank` | $\sigma_2 / \sigma_1$ | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `input_rank1` ($16\times 16$) | **1** | **1** | 1.000 | 1.000 | `1.09e-16` | Exact rank 1 |
| `frob_sgd` ($G_1 / \lVert G_1\rVert_F$) | **1** | **1** | 1.000 | 1.000 | `1.08e-16` | Preserves rank & spectrum |
| `separable_DL_G_DR` ($D_L G_1 D_R$) | **1** | **1** | 1.000 | 1.000 | `8.28e-17` | Preserves exact rank |
| `adafactor_shazeer_stern` ($\varepsilon_1$ inside squares) | **1** | **1** | 1.000 | 1.000 | `9.43e-17` | Separable $\implies$ preserves exact rank |
| **`adafactor_particlegan_law4`** ($+0.015\,\text{peak}+10^{-8}$) | **6** | **3** | 1.000 | 1.035 | `5.51e-03` | **BREAKS exact rank!** |
| `adam_step1_eps0` | **1** | **1** | 1.000 | 1.000 | `1.04e-16` | Rank 1 ($\operatorname{sign}(u)\operatorname{sign}(v)^\top$) |
| **`adam_step1_eps1e-8`** | **2** | **1** | 1.000 | 1.000 | `4.76e-10` | **BREAKS exact rank!** |
| **`adam_step1_peak_rms`** (`0.015*rms+1e-8`) | **6** | **3** | 1.000 | 1.035 | `5.51e-03` | **BREAKS exact rank!** |
| **`adam_step2_eps0_b1_0_on_rank1_g2`** ($\varepsilon=0, \beta_1=0$) | **3** | **2** | 1.000 | 1.003 | `3.19e-04` | **BREAKS exact rank via $V_2$!** |
| `exact_sign_on_rank1` | **1** | **1** | 1.000 | 1.000 | `1.04e-16` | Preserves rank 1 only |
| **`smooth_sign_s0.5_on_rank1`** (`law4_sign_mp`) | **6** | **3** | 1.000 | 1.044 | `7.07e-03` | **BREAKS exact rank!** |
| `input_rank2_for_sign` ($G_{ij} = j - i + 0.5$) | **2** | **2** | 1.805 | 1.997 | `8.97e-01` | Exact rank 2 |
| **`exact_sign_on_rank2`** ($\operatorname{sign}(j-i+0.5)$) | **16** | **16** | 2.459 | 10.642 | `1.00e+00` | **Rank $2 \to 16$ ($\det=2^{15}$)!** |
| `shampoo_1step_s0.1_on_rank1` | **1** | **1** | 1.000 | 1.000 | `6.46e-14` | Preserves exact rank (`1.24e-13` vs polar) |
| `ns_fp64_s0_on_rank1` / `ns_fp64_s0.1_on_rank1` | **1** / **1** | **1** / **1** | 1.000 | 1.000 | `3.5e-14` / `2.6e-14` | Preserves exact rank (zero `Z=[X,Y]` leak!) |

- **Diagonal vs. Kronecker Effective Rank on $\kappa = 10^4$ ($16\times 16$, exact rank 16):**
  - Axis-aligned `raw`: `s_rank = 1.000` $\to$ Shazeer–Stern Adafactor: **`15.000`** (exact $D_L = \operatorname{diag}(\sigma^{-1})$: **`16.000`**).
  - Orthogonally rotated `raw`: `s_rank = 1.000` $\to$ Shazeer–Stern Adafactor: **`1.000`** (`rv = 1.105`) vs. **1-step Kronecker Shampoo (`s=1e-6`): `s_rank = 15.999`, `rv = 16.000`** (`7.82e-9` L2 error vs. exact Tikhonov polar) vs. 5-step FP64 NS (`s=0`): `s_rank = 1.037, rv = 6.262`.

### 3.2 Experiment 2 — Newton–Schulz Across Exact Rank-Deficiency, Near Rank-Deficiency & Precision (`FP64`, `FP32`, `BF16`, `FP16`)

#### (a) Exact Rank-1 ($64 \times 64$) Across Dtypes & Actual ParticleGAN Implementations

| Implementation / Dtype | `rank(1e-12)` | `rank(1e-6)` | `rank(1e-3)` | `stable_rank` | `rv_erank` | $\sigma_1$ | $\sigma_2/\sigma_1$ | `col_leak_rel` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ref_ns_s0_fp64` | **1** | **1** | **1** | 1.000 | 1.000 | 0.978 | `5.41e-14` | `1.08e-13` |
| `ref_ns_tikh_s0.01_fp64` (`Z=[X,Y]`) | **1** | **1** | **1** | 1.000 | 1.000 | 0.984 | `9.78e-15` | `2.20e-14` |
| `ref_ns_s0_fp32` | 64 | 36 | **1** | 1.000 | 1.003 | 0.978 | `3.20e-05` | `6.01e-05` |
| `ref_ns_tikh_s0.01_fp32` (`Z=[X,Y]`) | 64 | 17 | **1** | 1.000 | 1.001 | 0.984 | `5.30e-06` | `1.12e-05` |
| **`ref_ns_s0_bf16`** | 64 | 64 | **55** | **2.531** | **25.747** | 0.978 | **`5.35e-01`** | **`7.78e-01`** |
| `ref_ns_tikh_s0.01_bf16` | 63 | 63 | 45 | 1.067 | 11.383 | 0.985 | `1.29e-01` | `2.51e-01` |
| `ref_ns_s0_fp16` | 64 | 64 | 44 | 1.025 | 8.049 | 0.978 | `7.62e-02` | `1.55e-01` |
| `v11_quintic_2d(s=0)` **CPU** (FP32) | 64 | 37 | **1** | 1.000 | 1.003 | **`8.000`** *(blowup!)* | `2.76e-05` | `5.90e-05` |
| **`v11_quintic_2d(s=0)` CUDA (BF16)** | 64 | 64 | **55** | **2.531** | **25.747** | **`5.028`** *(blowup!)* | **`5.35e-01`** | **`7.78e-01`** |
| `v11_quintic_2d(s=0.0061)` **CUDA** (FP32) | 64 | 26 | **1** | 1.000 | 1.001 | 0.938 | `1.22e-05` | `2.24e-05` |
| **`t4_quintic_2d(s=0)` CUDA (BF16)** | 64 | 64 | **56** | **4.045** | **27.149** | 0.700 | **`8.29e-01`** | **`8.68e-01`** |
| `convolution_quintic(s=0.0061)` CUDA (FP32) | 64 | 27 | **1** | 1.000 | 1.001 | 0.576 | `1.08e-05` | `2.30e-05` |

- **128×128 Rank-1 Norm Blowup Check:** `v11_quintic_2d(s=0)` produces $\sigma_1 = 11.3137 = \sqrt{128}$ on CPU FP32 and $\sigma_1 = 6.8855$ on CUDA BF16 (because `target_frob / out_frob` divides by $\sqrt{s_{\text{rank}}}$), whereas `v11_quintic_2d(s=0.0061)` stays bounded at $\sigma_1 = 0.9393$.

#### (b) Nearly Rank-Deficient ($64 \times 64$, $\sigma_1 = 1, \sigma_{2..64} = 10^{-k}$)

| Ratio $\sigma_2/\sigma_1$ | Input `s_rank` | FP64 NS (`s=0`) `s_rank` | CUDA `v11` NS (`s=0`, BF16) `s_rank` (`rv`) | CUDA `v11` Tikh (`s=0.0061`, FP32) `s_rank` | Exact SVD Polar `s_rank` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `1e-1` | 1.63 | **63.95** | 60.25 (`63.99`) | **63.96** | 64.00 |
| `1e-2` | 1.01 | **63.94** | 62.01 (`64.00`) | **50.92** | 64.00 |
| `1e-3` | 1.00 | **15.33** | 16.43 (`62.24`) | 2.86 | 64.00 |
| `1e-4` | 1.00 | 1.15 | 2.84 (`35.43`) *(noise!)* | 1.02 | 64.00 |
| `1e-6` .. `1e-14` | 1.00 | **1.00** | **2.53 (`25.75`) *(pure BF16 noise!)*** | **1.00** | 64.00 *(for $\ge 10^{-14}$)* |

#### (c) Which Mechanisms CAN vs. CANNOT Introduce Genuinely New Singular Directions?

| Mechanism | Can Increase Exact Rank? | Measured Verification (`p0_rank_experiments.py`) |
| :--- | :---: | :--- |
| Scalar / Frobenius normalization (`law4_frob_sgd_mp`, `nsgda_layer`) | **No** | `rank = 1` preserved identically |
| Invertible separable $D_L G D_R$ (Shazeer–Stern Adafactor) & Kronecker $L G R$ (Shampoo) | **No** | `rank = 1` preserved identically (Shampoo lifts *effective* rank!) |
| Polynomial NS & fused Tikhonov `Z=[X,Y]` NS (exact arithmetic / FP64) | **No** | `rank = 1`, `col_leak = 2.2e-14` (BF16 breaks numerically via rounding!) |
| Dynamic `1e-38` scalar rescaling & `unified_v12` CountSketch flip controller | **No** | Scalar gain only (`flip_cos = -1.0000001` on negated grad) |
| **First-moment momentum ($\beta_1 > 0$)** (e.g. `t4_equil`, `muon_m`, `v11/v12` when `stiff_gate > 0`) | **Yes** | 5 independent rank-1 steps $\to$ exact `rank = 5` (`s_rk = 2.22`, then NS $\to 4.77$) |
| **Minibatch gradient accumulation ($B$ samples)** | **Yes** | $B \in \{1,2,4,8,16,32\} \implies \operatorname{rank} = \min(B, 16)$ |
| **Weight decay ($+\lambda W_t$)** (disabled in `v11`/`v12`, verified via `ValueError`) | **Yes** | Rank-1 grad $+ 10^{-2} W_{\text{orth}} \to$ exact `rank = 16` |
| **Non-separable Hadamard scaling** (Adam $\varepsilon>0$ or multi-step $\beta_2$, Smooth SignSGD, ParticleGAN `law4_adafactor_mp`, Exact SignSGD on rank $\ge 2$) | **Yes** | Increases exact rank via coordinate-aligned distortion (not aligned with subdominant singular subspace!) |

### 3.3 Experiment 3 — Empirical Verification on Real ParticleGAN Architectures (`grid100_mlp16` vs. Shallow vs. ResMLP16 vs. DCGAN Conv2d)

#### (a) Architectural Gradient Spectra (`deterministic_orthogonal_(seed=0)`, $B=256$ vector / $B=16$ image)

| Architecture (`# wide layers`) | Median `rank(1e-12)` | Median `rank(1e-6)` | Median `stable_rank` | Median `PR` | Median `rv_erank` | Median `top1%` | Norm Skew ($\max/\min$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`grid100_mlp16`** (`15` hidden $128\times 128$) | **126.0** *(NOT rank 1!)* | **6.0** *(decays `111 -> 1`)* | **1.00** | **1.00** | **1.00** | **100.0%** | **`27,169.7×`** |
| `grid100_shallow_mlp3` (`2` hidden $128\times 128$) | 128.0 | 109.0 | 1.16 | 1.33 | 3.81 | 93.5% | `1.2×` |
| `grid100_res16_tanh` (`14` residual $128\times 128$) | 128.0 | 120.0 | 1.50 | 2.02 | 5.70 | 70.6% | `1.3×` |
| **`dcgan_r64_conv4d`** (`7` 4D Conv/Deconv, per-offset) | 64–256 | 16–256 | **5.14** *(flat `8.20`)* | — | **97.98** *(flat `207.81`)* | — | — |

- **Depth-wise exponential spectral collapse inside `grid100_mlp16`:** `s2/s1` drops monotonically by $\sim 2\text{–}3\times$ per layer: `net.2` (`0.312`, `s_rk=1.14`, `rv=3.66`) $\to$ `net.6` (`0.100`, `s_rk=1.01`, `rv=2.35`) $\to$ `net.12` (`3.14e-3`, `s_rk=1.00`, `rv=1.05`) $\to$ `net.16` (`1.63e-4`, `s_rk=1.00`, `rv=1.003`) $\to$ `net.20..30` (`~1.6e-8` FP32 floor), while `rank(1e-12)` stays `123–128` across every single layer (and stays `124–126` with `s_rk = 1.00, rv = 1.00–1.02` through trained `unified_v12` steps `1, 10, 50`).

#### (b) Optimizer Update Spectrum & Subdominant Singular-Subspace Alignment on `grid100_mlp16` Early-Mid Layer (`net.6.weight`, `s2/s1 = 0.100`, `s_rk = 1.01`)

| Optimizer Transform | `rank(1e-6)` | `stable_rank` | `rv_erank` | `cos_polar_full` | **`cos_sub16` (True Tail Modes $2..16$)** | **`sub_diag_E` (True Tail Energy)** | **`off_basis_E` (Distortion/Noise)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw_grad` / `frob_mp` | 96 | 1.01 | 2.35 | 0.109 | 0.505 | 1.4% | **0.0%** |
| **`ns_5step_fp32_s0`** | 128 | **10.08** | 27.08 | 0.401 | **0.939** | **92.4%** | **0.0%** |
| `v11_quintic_ns_s0_device` (CUDA BF16) | 128 | 10.96 | 45.79 | 0.383 | 0.895 | 84.3% | 8.6% *(BF16)* |
| **`v11_tikh_ns_seff_0.30peak`** (FP32) | 122 | **3.84** | 11.77 | 0.255 | **0.678** | **73.9%** | **0.0%** |
| **`v12_tikh_ns_seff_0.50peak`** (FP32) | 119 | **3.02** | 9.74 | 0.230 | **0.618** | **66.9%** | **0.0%** |
| **`shampoo_1step_kronecker`** ($L G R$) | 128 | **95.41** | 111.09 | **0.917** | 0.399 | **99.0%** | **0.0%** |
| `exact_polar_svd` | 128 | 128.00 | 128.00 | 1.000 | 0.344 | 99.2% | 0.0% |
| `adafactor_shazeer_stern` ($D_L G D_R$) | 109 | 1.28 | 5.93 | 0.096 | 0.181 | 7.6% | **61.8%** |
| `adafactor_particlegan` ($+0.015\,\text{peak}$) | 123 | 1.11 | 5.88 | 0.106 | 0.182 | 6.1% | **45.3%** |
| `adam_multistep_peak_rms` (`law4_adam_equil`) | 127 | **1.06** | 9.08 | 0.089 | **0.103** | **1.8%** | **45.4%** |
| `sign_smooth_law4` (`law4_sign_mp`) | 110 | **1.17** | 21.99 | 0.065 | **0.074** | **1.5%** | **68.9%** |
| `sign_exact` | 66 | 1.29 | 30.52 | 0.060 | 0.069 | 1.5% | **73.9%** |

*And on deep-mid layer `net.16.weight` (`s2/s1 = 1.63e-4`):* `ns_5step_fp32_s0` has `s_rk = 1.01, off_basis_E = 0.0%` (bounded by $493\times$ 5-step gain), `shampoo_1step_kronecker` cleanly lifts `s_rk = 5.91, sub_diag_E = 83.1%, off_basis_E = 0.0%`, while CUDA BF16 `v11_quintic_ns_s0_device` amplifies BF16 rounding into `s_rk = 2.77, off_basis_E = 62.0%, cos_sub16 = 0.008`!

---

## 4. Eight-Point Structured Analysis of All Substantial Claims

### Claim 1 — "Elementwise, Frobenius, $L_\infty$, and Kronecker-factored preconditioners preserve gradient rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$)"
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims that elementwise (Adam), Frobenius, $L_\infty$ (SignSGD/Lion), and Kronecker-factored (Adafactor/Shampoo) updates all factor as $D_L G D_R$ and therefore preserve exact matrix rank $\operatorname{rank}(U) = \operatorname{rank}(G)$.
2. **Assumptions:** Implicitly assumes every diagonal/elementwise preconditioner acts by separable left/right multiplication and treats exact rank as equivalent to effective rank.
3. **Existing supporting evidence:** Valid *only* for the strict subset of scalar/Frobenius scaling ($c G$) and invertible separable/Kronecker products ($D_L G D_R$, $L G R$).
4. **Potential counterexamples:** Minimal Hadamard $[[1,1],[1,1]] \odot [[1,1],[1,2]]$ (`rank 1 -> 2`); 1-step Adam ($\delta > 0$) with $\det = -4\delta/[(1+\delta)(4+\delta)(2+\delta)^2] < 0$; 2-step Adam ($\varepsilon=0, \beta_1=0$) on rank-1 stream (`rank 1 -> 3+`); Smooth SignSGD on rank-1 (`rank 1 -> 6`); Exact SignSGD on rank-2 $j-i+0.5$ (`rank 2 -> 16`); ParticleGAN's own `law4_adafactor_mp` (`rank 1 -> 6`).
5. **Minimal falsification experiment:** Compute `torch.linalg.matrix_rank` and `det` on `g2 / (g2.abs() + 1e-8)` for $g_2 = [1,2]^\top [1,2]$.
6. **Relevant prior research:** Horn & Johnson (*Matrix Analysis*, §5.1, Schur product rank bound $\operatorname{rank}(A \odot B) \le \operatorname{rank}(A)\operatorname{rank}(B)$); Forster (2002) & Alon et al. (2016) on sign-rank; Shazeer & Stern (2018) *Adafactor*; Gupta et al. (2018) *Shampoo*.
7. **Computational verification:** Verified in `test_hadamard_2x2_counterexample_increases_rank`, `test_adam_elementwise_scaling_breaks_exact_rank_on_rank1`, `test_sign_sgd_and_smooth_sign_rank_behavior`, and `test_adafactor_separable_vs_particlegan_additive_floor` (all PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** (pure algebraic proof + FP64 verification).

### Claim 2 — "Only non-diagonal spectral polar orthogonalization restores gradient rank / lifts collapsed gradients back to full rank"
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims that Newton–Schulz / polar orthogonalization increases exact matrix rank ($\operatorname{rank}(X_k) > \operatorname{rank}(G)$) to restore full rank from collapsed rank-1 gradients, and is the unique mechanism capable of rescuing collapsed spectra.
2. **Assumptions:** Confuses exact rank with stable/effective rank; overlooks full Kronecker preconditioning (Shampoo) and finite-precision BF16 rounding artifacts.
3. **Existing supporting evidence:** Empirically, 5-step Quintic NS rescues `grid100_mlp16` (`prec = 0.992`) where Frobenius-MP (`0.157`), Adam (`0.255`), and Adafactor (`0.174`) fail.
4. **Potential counterexamples:** (a) Theorem 3 proves $\operatorname{Im}(X_k) \subseteq \operatorname{Im}(G)$ and $\operatorname{rank}(X_k) \le \operatorname{rank}(G)$ for *all* odd-polynomial NS and fused `Z=[X, Y]` Tikhonov NS in exact arithmetic; (b) 1-step Kronecker Shampoo $L G R$ equals exact Tikhonov polar and lifts rotated $\kappa=10^4$ stable rank `1.000 -> 15.999` (and `grid100_mlp16` `net.6` to `95.41` with `99.0%` true subdominant energy); (c) on $\sigma_2/\sigma_1 \le 10^{-6}$, 5-step FP64 NS fails to lift stable rank (`1.00`).
5. **Minimal falsification experiment:** Apply 5-step FP64 `ref_poly_ns` (both $s=0$ and fused `Z=[X,Y]` $s=0.01$) to exact rank-1 $64\times 64$ $u v^\top$; measure `rank(1e-12) == 1` and `col_leak = 1.08e-13`.
6. **Relevant prior research:** Higham (2008) *Functions of Matrices* (Ch. 8, polar decomposition & Newton–Schulz iterations); Jordan et al. (2024) *Muon*; Bernstein & Newhouse (2024) *Old Optimizer, New Norm*; Gupta et al. (2018) *Shampoo*.
7. **Computational verification:** Verified in `test_polynomial_ns_and_fused_tikhonov_preserve_subspace_and_rank_fp64`, `test_ns_five_step_spectral_gain_limit_and_tikhonov_cutoff`, and `test_diagonal_vs_kronecker_shampoo_effective_rank_dichotomy` (all PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** for exact arithmetic; open empirical question whether full Kronecker Shampoo matches or beats 5-step NS on end-to-end `grid100_mlp16` training (not yet benchmarked in Forge).

### Claim 3 — "Effective rank $\operatorname{erank}(G_l) = \|G_l\|_F^2 / \sigma_1(G_l)^2$"
- **Classification:** **Falsified claim** *(terminological/mathematical definition error)*
1. **Precise formulation:** Identifies $\|G\|_F^2 / \sigma_1^2$ as "effective rank (`erank`)".
2. **Assumptions:** Implicitly treats all spectral diversity scalars as interchangeable.
3. **Existing supporting evidence:** Used consistently in `experiments/law4_optimizer_profiler.py:757` (while `_law4_energy/optimizer.py:3610,4162` separately defines `erank_2 = ||G||_F^4 / ||G G^T||_F^2`, the Rényi-2 participation ratio).
4. **Potential counterexamples:** $\sigma = (1, 10^{-2}, \dots, 10^{-2}) \in \mathbb{R}^{101}$ gives $r_{\text{stable}} = 1.01$ vs. Roy–Vetterli $\operatorname{erank}_{\text{RV}} = 20.0$; real `dcgan_G` `main.3.weight` gives $r_{\text{stable}} = 3.05$ vs. $\operatorname{erank}_{\text{RV}} = 141.08$.
5. **Minimal falsification experiment:** Evaluate both formulas on `diag(1, 1e-2, ..., 1e-2)`.
6. **Relevant prior research:** Rudelson & Vershynin (2007) / Vershynin (*High-Dimensional Probability*, §7.6: *stable rank* $\|A\|_F^2 / \|A\|_2^2$); Roy & Vetterli (2007, EUSIPCO: *effective rank* $\operatorname{erank}(A) = \exp(H(\sigma/\|\sigma\|_1))$).
7. **Computational verification:** Verified in `test_spectral_metric_inequality_chain_and_stable_vs_rv_gap` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim 4 — "Deep skip-free dense MLPs (`grid100_mlp16`) suffer spectral/effective-rank collapse requiring spectral singular-value equalization, whereas shallow MLPs, 16-layer residual MLPs, and 4D Conv2d layers maintain diverse spectra where NS is redundant"
- **Classification:** **Empirically supported causal mechanism** *(once restated in terms of spectral anisotropy / stable rank and singular-basis-aligned equalization rather than exact rank)*
1. **Precise formulation:** In deep skip-free unnormalized `LeakyReLU(0.2)` MLPs ($L=16$), minibatch gradients retain high exact rank (`123–128`) yet collapse in stable rank ($r_{\text{stable}} \to 1$, $\sigma_2/\sigma_1$ decaying exponentially with depth) with dense/rotated singular vectors. Spectral equalizers that preserve the singular basis $(U, V)$ (5-step NS, polar, 1-step Shampoo) allocate `67%–99%` of update energy cleanly into true subdominant modes (`cos_sub16 = 0.62–0.94`), whereas Frobenius-MP preserves the collapse (`1.4%` tail energy) and coordinate-wise methods (Adam, SignSGD, Adafactor) keep $r_{\text{stable}} \approx 1.06\text{–}1.29$ while injecting `45%–74%` off-singular-basis distortion. By contrast, skip connections (`grid100_res16_tanh`, $r_{\text{stable}} = 1.50, \operatorname{erank}_{\text{RV}} = 5.70$) and spatial 4D Conv2d + BatchNorm (`dcgan_r64`, per-offset $r_{\text{stable}} = 5.14, \operatorname{erank}_{\text{RV}} = 97.98$) avoid rank-1 spectral collapse, making 4D Conv2d NS redundant.
2. **Assumptions:** Holds for unnormalized skip-free `LeakyReLU` MLPs vs. BatchNorm Conv2d DCGANs and identity-skip ResMLPs under protocol seed `0`, `deterministic_orthogonal_` init, and tested Forge tasks.
3. **Existing supporting evidence:** Round-4 Occam Arm 1 (`r4-razor-no-ns-v2`: `grid100_mlp16` drops `0.9929 -> 0.0244`, while `grid100`, `res16_scaled_tanh`, `3/3` Nash, and `4/4` Conv2d images all PASS without NS); `minimal-v2` (`0.0714`); `law4-energy-v3`/`v10`/`unified-v12` (`4/4` Conv2d at `hq=1.0`, `17.4×` faster optimizer, `-40.8%–64.0%` J/step); our Exp 3 direct gradient & alignment measurements.
4. **Potential counterexamples:** (a) Deep **skip-free unnormalized** 4D ConvNets *without* BatchNorm (e.g., 16-layer plain Conv2d without BN) can also exhibit depth-wise spectral collapse — so the dichotomy is **skip/BN spectral diversity vs. deep skip-free contraction**, not intrinsically "2D vs. 4D tensor rank"! (b) On ultra-collapsed deep layers (`net.20..30`, $\sigma_2/\sigma_1 \approx 1.6\times 10^{-8}$), 5-step FP32 NS cannot lift subdominant modes directly; rescue on `grid100_mlp16` operates either via early/mid layers (`net.2..12`) propagating better-conditioned features forward, momentum (`b1_eff = 0.75 * stiff_gate`), and/or BF16 rounding perturbations!
5. **Minimal falsification experiment:** Train a 16-layer skip-free Conv2d network *without* BatchNorm using 0-Conv-NS Frob-MP vs. per-offset NS; and ablate FP32-only vs. BF16 `v11_quintic_2d(s=0)` on `grid100_mlp16` to test whether BF16 rounding noise helps break symmetry on `net.20..30` or is purely harmful!
6. **Relevant prior research:** Daneshmand et al. (2020, NeurIPS) *Batch Normalization Provably Avoids Rank Collapse for Randomly Initialised Deep Networks*; Dong et al. (2021) *Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth*; Saxe et al. (2014) *Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks*; Jordan et al. (2024) *Muon*.
7. **Computational verification:** Verified in `test_real_architectures_exact_vs_effective_rank_and_subdominant_alignment` and Exp 3 tables (PASS).
8. **Confidence & remaining uncertainty:** **High (85%)** for the corrected spectral-anisotropy/basis-alignment mechanism on tested Forge regimes; **unresolved** whether unnormalized deep Conv2d breaks the "4D never needs NS" heuristic, and **unresolved** how much BF16 rounding noise in `v11_quintic_2d(s=0)` vs. FP32 NS affects `grid100_mlp16` convergence.

### Claim 5 — "Aggressive spectral whitening on naturally low-rank equilibrium critic gradients injects orthogonal agitation (`3.8×`), and 0-NS polar-Frobenius-matched $\sqrt{r}$ critic updates eliminate it"
- **Classification:** **Empirically supported causal mechanism** *(reinforced by a newly discovered BF16 numerical mechanism)*
1. **Precise formulation:** When a 2D critic's equilibrium gradient has low effective rank ($r_{\text{stable}} \approx 1.1\text{–}1.4$), applying 5-step Quintic NS amplifies minor stochastic minibatch noise singular values (and in BF16 when `s -> 0`, amplifies $O(10^{-3})$ rounding error up to `77.8%` orthogonal energy), inflating update effective rank (`dwEr = 17–22` vs. `1.0–1.1`) and driving logit jitter. Replacing neural-G critic NS with Frobenius-MP matched to polar Frobenius norm $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ eliminates orthogonal noise amplification.
2. **Assumptions:** Assumes Law 2 bounds critic slope ($\|\nabla_x D\| \le 1$) and a neural generator is present (on generator-free direct particle clouds `t3_iso_*`, `law4-energy-v11` and `unified_v12` had to restore 2D critic NS!).
3. **Existing supporting evidence:** `law4-energy-v10` (`42/48` PASS, cracking `staggered100`, `five_word_joint_hold`, `model_ddgan_bimodal_chain`), `law4-energy-v11` (`44/48`), `unified_v12` (`46/48`), plus our Exp 2 BF16 leakage proof (`col_leak = 0.778` at `s=0` vs. `2.2e-5` in FP32 Tikhonov).
4. **Potential counterexamples:** Generator-free `t3_iso_scale_mass_8x` & `t3_iso_horizon_saddle_shift` (where 0-NS critic failed marginally until `unified_v12` added `_step_generator_free_critic`), or an intentionally deep skip-free 16-layer critic (untested in Forge!).
5. **Minimal falsification experiment:** Evaluate `unified_v12` (0-NS neural-G critic) on a task with a **16-layer skip-free critic** paired with a shallow generator.
6. **Relevant prior research:** Spectral-norm / polar whitening variance inflation under stochastic gradients; Bernstein & Newhouse (2024); Lau et al. (2025) *PolarGrad*.
7. **Computational verification:** Exp 2 directly quantifies the exact small-SV & BF16 noise amplification curve across $\sigma_2/\sigma_1 \in [10^{-14}, 10^{-1}]$ and $s \in \{0, 0.0061\}$.
8. **Confidence & remaining uncertainty:** **High (80%)** on shallow neural critics; **likely falsifiable** if the critic itself is a deep skip-free unnormalized MLP (a hidden assumption currently omitted from Law 4.3/4.4!).

### Claim 6 — Implementation Claims in Docs & Comments (`_QUINTIC_COEFFS`, Tikhonov BF16 Threshold, `s=0` Norm Rescaling, `CountSketch` Preconditioning, `deconflict.py` NS)
- **Classification:** **Falsified claim** *(documentation/comment drift from executed code)*
1. **Precise formulation:** Four explicit claims in `docs/four-laws.md`, `law4-optimizer-laws.md`, and code docstrings misstate the executed numerical formulas.
2. **Assumptions:** None — direct static and dynamic code inspection.
3. **Existing supporting evidence:** None; contradicted by actual executed code paths.
4. **Potential counterexamples:** Direct source inspection + `test_particlegan_ns_coefficients_and_rank1_norm_blowup_audit` and `test_momentum_weight_decay_countsketch_and_optimizer_contracts`:
   - `_unified_v4/optimizer.py:24-30`: 5 step-varying triples, not `(3.4445, -4.7750, 2.0315)`.
   - `_unified_v11/optimizer.py:102,133,150`: Tikhonov is FP32 for $m \le 256$ (not $m \le 64$); non-Tikhonov is BF16 for $m \ge 64$ on CUDA and rescales rank-1 outputs to $\sigma_1 = \sqrt{m}$ (`11.31` on $128\times 128$ FP32!).
   - `_unified_v12/optimizer.py:57-74,126-170`: CountSketch is a scalar period-2 flip-gain controller, not spectral preconditioning.
   - `_t1_coupling/deconflict.py:24`: calls `torch.linalg.eigh`, not coupled Newton–Schulz.
5. **Minimal falsification experiment:** Assert `_QUINTIC_COEFFS[0] != (3.4445, -4.7750, 2.0315)` and `spectral_metrics(v11_quintic_2d(rank1_128, smoothing=0.0))["sigma_1"] == sqrt(128)`.
6. **Relevant prior research:** Cesista / Jordan (2024–2025) non-stationary 5-step quintic Muon/Polar-Express schedules; Charikar et al. (2002) CountSketch.
7. **Computational verification:** Directly asserted in `tests/four_laws_audit/test_p0_law43_rank.py` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim 7 (C9) — "Skip-free $L=16$ `LeakyReLU(0.2)` orthogonal MLP contracts forward output spread by $0.52^8 \approx 1/187$ and attenuates early-layer gradients by $10^2\times$–$10^6\times$" (`t4-strange-architectures.md:25-26`)
- **Classification:** **Empirically supported causal mechanism** *(exact mathematical proof covers only the 1-layer / fixed-input expected squared-norm contraction under symmetric zero-mean gating, not batch spread/variance or inter-layer gradient skew)*
1. **Precise formulation:** Under orthogonal weights ($\|W_\ell\|_2 = 1$), zero biases, and symmetric zero-mean pre-activations (random half-space gating where each coordinate is independently positive with probability $1/2$), each `LeakyReLU(0.2)` layer satisfies $\mathbb{E}[\|D_\ell W_\ell v\|_2^2] = \frac{1 + 0.2^2}{2}\|v\|_2^2 = 0.52\,\|v\|_2^2$, so over $L=16$ layers a fixed input's expected **squared $\ell_2$ norm** contracts by $0.52^{16}$ (RMS norm by $0.52^{L/2} = 0.52^8 \approx 1/187$). Separately, `t4-strange-architectures.md:26` claims this accounts for both batch output spread contraction (`~156×`) and $10^2\times$–$10^6\times$ early-layer gradient attenuation relative to the output layer.
2. **Assumptions:** Orthogonal initialization $\|W_\ell\|_2 = 1$, zero bias, and independent symmetric zero-mean pre-activation signs across all $L=16$ layers.
3. **Existing supporting evidence:** Empirical `~156×` output spread contraction in `problems.md#grid100_mlp16` (`output_sigma = 0.029`); our Exp 3 direct step-0 measurement on `grid100_mlp16` (`SimpleMLPGenerator(2, 128, 16, 2)`, `deterministic_orthogonal_(seed=0)`) confirms monotonic inter-layer gradient norm skew of **`27,169.7×`** (`net.2.weight` $\|G\|_F = 5.89\times 10^{-6}$ vs. `net.30.weight` $\|G\|_F = 1.60\times 10^{-1}$).
4. **Potential counterexamples / deductive gaps:** (a) **Nonzero mean shift breaks batch-variance equivalence:** `LeakyReLU(0.2)` maps symmetric zero-mean inputs to positive-mean outputs ($\mathbb{E}[\phi(z)] = \frac{1-0.2}{\sqrt{2\pi}}\sigma > 0$), so pre-activations after layer 1 are no longer zero-mean symmetric; thus $0.52^{L/2}$ governs fixed-input RMS norm under idealized symmetric gating, **not** batch output variance/spread; (b) **Uniform contraction alone does NOT imply inter-layer gradient skew:** In a uniform-width chain with i.i.d. scalar contraction $c$ per layer, forward activations scale as $\|h_{\ell-1}\| \sim c^{\ell-1}$ and backward sensitivities as $\|\delta_\ell\| \sim c^{L-\ell}$, so hidden-layer weight gradients $\|G_{W_\ell}\| \sim \|\delta_\ell\|\|h_{\ell-1}\| \sim c^{L-1}$ would be depth-uniform! The observed `27,170×` inter-layer gradient skew is therefore an empirical phenomenon driven by width bottlenecks (`2 -> 128 -> 2`), nonzero-mean cone alignment across layers, and gating correlations — not a deductive corollary of $0.52^{L/2}$ alone.
5. **Minimal falsification experiment:** Compare layerwise forward RMS norm, batch std, mean-to-std ratio, and hidden-layer $\|G_{W_\ell}\|_F$ across a uniform-width $128 \to 128$ $L=16$ `LeakyReLU(0.2)` orthogonal MLP (with vs. without mean-centering) versus `SimpleMLPGenerator(2, 128, 16, 2)`.
6. **Relevant prior research:** He et al. (2015) *Delving Deep into Rectifiers* (Kaiming gain $\sqrt{2/(1+\alpha^2)}$); Poole et al. (2016) *Exponential Expressivity in Deep Neural Networks through Transient Chaos*; Schoenholz et al. (2017) *Deep Information Propagation*; Daneshmand et al. (2020) *Batch Normalization Provably Avoids Rank Collapse*.
7. **Computational verification:** Exp 3 verifies the empirical `27,169.7×` gradient norm skew (`net.2` `5.89e-6` to `net.30` `1.60e-1`) and depth-wise `s2/s1` collapse (`0.312 -> 1.6e-8`) on `grid100_mlp16` (`test_real_architectures_exact_vs_effective_rank_and_subdominant_alignment`, PASS).
8. **Confidence & remaining uncertainty:** **100%** for the conditional 1-layer squared-norm identity $0.52$; **High (90%)** for the empirical `grid100_mlp16` forward contraction and `27,170×` gradient skew at seed 0; **falsified as a deductive proof** from $0.52^{L/2}$ alone to batch variance or inter-layer gradient skew.

---

## 5. Engineering & Research Recommendations (Open Risks Caught by P0)

1. **Fix `s=0` Frobenius-Rescaling Norm Blowup (`particlegan/_unified_v11/optimizer.py:150,243`):** Multiplying `out` by `target_frob / out_frob = sqrt(min(r, c)) / out_frob` when `smoothing == 0` violates the polar operator-norm bound $\|U\|_2 \le 1$ by up to $\sqrt{\min(r, c)}$ on rank-deficient matrices (`11.31×` on $128\times 128$). Removing that rescaling (or clamping the multiplier to `<= 1.15`, matching the Tikhonov branch at `L116`) restores $\|U\|_2 \approx 1$. *(Note: per `AGENTS.md`, any change is a declared trainer delta requiring matched Forge re-qualification, not a silent edit!)*
2. **Audit BF16 vs. FP32 in `v11_quintic_2d(s=0)` on `grid100_mlp16`:** Because `s_base = smoothing * (1 - stiff_gate)` hits exact `0.0` when `stiff_gate == 1.0`, `128×128` layers abruptly switch from FP32 Tikhonov NS to BF16 non-Tikhonov NS, injecting `62%–78%` orthogonal rounding noise on nearly rank-1 layers (`net.16..30`). A dedicated FP32-vs-BF16 ablation on `grid100_mlp16` will determine whether BF16 noise is accidentally helping escape `net.20..30` collapse or merely adding jitter.
3. **Test Deep Skip-Free Conv2d (No BN) & Deep Skip-Free Critic:** Law 4.3's current phrasing ("2D dense needs NS, 4D Conv2d & critic don't") relies on BatchNorm/skip-connections in DCGAN and shallow 3-layer critics in Forge. Testing unnormalized deep Conv2d and deep skip-free critics will test the exact scope boundary of the corrected law.

---

## 6. Stronger, Corrected Formulation of Law 4.3 & Exact Proposed Diffs

### 6.1 Corrected Formulation of Law 4.3

> **Law 4.3 — Spectral Anisotropy Equalization vs. Low-Effective-Rank Noise Inflation:**
> Let a weight matrix $W_l \in \mathbb{R}^{m \times n}$ have minibatch gradient $G_l = U \Sigma V^\top$ with exact rank $r = \operatorname{rank}(G_l)$, stable rank $r_{\text{stable}}(G_l) = \|G_l\|_F^2 / \sigma_1(G_l)^2$, and Roy–Vetterli entropy effective rank $\operatorname{erank}_{\text{RV}}(G_l) = \exp(H(\sigma / \|\sigma\|_1))$, satisfying $1 \le r_{\text{stable}} \le \operatorname{PR} \le \operatorname{erank}_{\text{RV}} \le r \le \min(m, n)$.
> 1. **Spectral Equalization Within Existing Subspaces (Not Exact-Rank Creation):** Odd-polynomial Newton–Schulz iterations ($X_{k+1} = p_k(X_k X_k^\top) X_k$, including Tikhonov-augmented $Z = [X, Y]$) and invertible Kronecker preconditioners ($L G_l R$, e.g., Shampoo) **never increase exact matrix rank in exact arithmetic** ($\operatorname{Col}(X_k) \subseteq \operatorname{Col}(G_l)$, $\operatorname{rank}(X_k) \le \operatorname{rank}(G_l)$). Instead, deep skip-free unnormalized dense layers (e.g., `grid100_mlp16`) produce gradients that retain high exact rank across minibatches ($r \approx \min(m, n)$) while suffering severe **effective-rank / spectral collapse** ($r_{\text{stable}} \to 1$, $\sigma_2/\sigma_1 \ll 1$) in rotated singular bases. Basis-aligned spectral equalizers (polar/NS, 1-step Shampoo) rescue these layers by amplifying existing subdominant singular values ($\sigma_k / \|G\|_F \gtrsim (\prod_j a_j)^{-1}$ for finite-step NS) cleanly along $(u_k, v_k)$ with zero off-basis distortion.
> 2. **Why Scalar, Diagonal-Separable, and Coordinate-Wise Optimizers Fail Spectral Collapse (for Distinct Reasons):** Scalar/Frobenius scaling ($c G_l$) leaves the entire normalized spectrum invariant. Diagonal separable scaling ($D_L G_l D_R$, e.g., Shazeer–Stern Adafactor) preserves exact rank and cannot equalize rotated/incoherent singular vectors ($D_L, D_R \approx c I$). Non-separable coordinate-wise preconditioners (Adam with $\varepsilon > 0$ or EMA $v_t$, smooth SignSGD, exact SignSGD on rank $\ge 2$) **do** break exact rank via Hadamard nonlinearity, yet remain dominated by $\operatorname{sign}(u_1)\operatorname{sign}(v_1)^\top$ ($r_{\text{stable}} \approx 1$) and inject coordinate-aligned off-singular-basis distortion rather than lifting true subdominant singular modes.
> 3. **Selective Application & Noise-Inflation Boundary:** On architectures whose per-layer gradients maintain high effective rank without depth-wise contraction — notably BatchNorm-regularized 4D Conv2d/ConvTranspose2d stacks, shallow networks, and skip-connected residual networks — spectral equalization is redundant once Frobenius-MP matches the polar Frobenius norm. Conversely, applying aggressive unregularized spectral equalization (especially low-precision BF16 NS at $s = 0$) to naturally low-effective-rank equilibrium critic gradients inflates minor stochastic and rounding noise into orthogonal agitation.

### 6.2 Exact Proposed Diffs (For Coordinator / P10 Integration — NOT Applied Here)

#### (1) Proposed diff for `docs/four-laws.md` (lines 87, 94, 162, 169)

```diff
--- a/docs/four-laws.md
+++ b/docs/four-laws.md
@@ -84,14 +84,14 @@
 - **Law 4.1 — Inter-Player Function-Space Impedance:** Under player capacity/gain mismatch $\rho \gg 1$ or hostile initialization, equal parameter steps produce $\Theta(\rho)$ function-space displacement mismatch and finite-time divergence unless realized function-space displacements are bounded.
 - **Law 4.2 — Intra-Player Degree-0 Gauge & Rail Escape:** Multiplicative inter-layer Jacobian attenuation across deep networks and $\operatorname{sech}^2$-like activation saturation span many orders of magnitude ($10^{-12}$ to $10^{+2}$), defeating any single player-level learning rate or fixed additive $\varepsilon$ floor. Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish ($\Delta W \to 0$ as $G \to 0$) is required without a step clock.
-- **Law 4.3 — Tensor-Rank Spectral Collapse vs. Low-Rank Noise Inflation:** Elementwise, Frobenius, $L_\infty$, and Kronecker-factored preconditioners preserve gradient rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$) and therefore cannot rescue deep skip-free dense layers whose gradient effective rank collapses ($\operatorname{erank}(G) \to 1$). Conversely, applying aggressive spectral whitening to already-diverse convolutional kernels or naturally low-rank equilibrium gradients wastes compute and injects orthogonal agitation.
+- **Law 4.3 — Spectral Anisotropy Equalization vs. Low-Effective-Rank Noise Inflation:** Deep skip-free unnormalized dense layers produce high-exact-rank minibatch gradients whose *effective* rank collapses ($r_{\text{stable}}(G) = \|G\|_F^2 / \sigma_1(G)^2 \to 1$) in rotated singular bases. Scalar/Frobenius scaling leaves singular-value ratios invariant; diagonal separable scaling ($D_L G D_R$) cannot equalize rotated singular vectors; and coordinate-wise Hadamard methods (Adam, SignSGD) break exact rank yet remain top-1-dominated while injecting off-singular-basis distortion. Basis-aligned spectral equalizers (polynomial Newton–Schulz / polar orthogonalization, or full Kronecker $L G R$ such as Shampoo) preserve exact rank ($\operatorname{rank}(X_k) \le \operatorname{rank}(G)$ in exact arithmetic) while lifting existing subdominant singular modes cleanly along their singular basis. Conversely, applying aggressive spectral whitening (especially unregularized low-precision NS) to already-diverse BatchNorm Conv2d / residual layers or naturally low-effective-rank equilibrium critic gradients wastes compute and amplifies stochastic/rounding orthogonal agitation.
 - **Law 4.4 — Generator–Critic Structural Asymmetry under Bounded Critics:** When Law 2 bounds the critic's spatial slope, the generator carries the nonlinear pushforward geometry across disconnected modes, deep compositions, and saturation rails, whereas the critic primarily needs protection against extreme $D \gg G$ capacity imbalance and equilibrium logit jitter.
 - **Law 4.5 — Stochastic Minimax Staleness & Minimum-State Trade-Off:** Once Law 2 damps critic rotational eigenvalues, stochastic minibatch two-player training penalizes stale momentum ($\beta_1 > 0$), high-variance optimistic extrapolation, and heavy coordinate-wise state, favoring low-staleness, minimum-state updates.
 
 **What Any Valid Solution Must Satisfy:**
 1. **Bounded Inter-Player Displacement:** Stabilize arbitrary capacity ratios ($\rho > 500\times$), bidirectional LR sweeps, and hostile initializations by regulating realized function-space motion.
 2. **Scale-Invariant Layer & Rail Equalization:** Equalize extreme depth attenuation and saturation rails without underflow dead zones, while settling cleanly at equilibrium with no step clock.
-3. **Selective Spectral Conditioning:** Restore gradient rank where diagonal conditioning hits the rank-preservation limit (e.g., deep skip-free dense networks) without injecting orthogonal noise or redundant matmuls where layers do not collapse.
+3. **Selective Spectral Conditioning:** Equalize collapsed singular-value spectra (lifting $r_{\text{stable}}$ along existing subdominant singular directions without creating spurious off-basis distortion) where deep skip-free contraction occurs, while avoiding redundant matmuls and orthogonal noise inflation on spectrally diverse or low-effective-rank equilibrium layers.
 4. **Conflict-Free Multi-Objective & Low-Energy Execution:** Handle auxiliary/constrained objectives (e.g., encoder/autoencoder or residual losses) without silent gradient interference, at minimal optimizer state and hardware energy.
@@ -159,10 +159,10 @@
 - **Underflow-Safe $10^{-38}$ Dynamic Scale Normalization & Homogeneous `peak_rms` Settling** ([`t4-imbalance-nash.md`](../reports/forge/studies/t4-imbalance-nash.md), [`r3-nash-complete.md`](../reports/forge/studies/r3-nash-complete.md)): Normalizes $(m_t, v_{\text{rms},t})$ by $s_t = \max(\max |v_{\text{rms},t-1}|, \max |g_t|, 10^{-38})$ so $10^{-12}$ early-layer and $\operatorname{sech}^2$-attenuated gradients receive the same directional geometry as $10^{+2}$ output gradients, while degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock.
 - **Closed-Form Marchenko–Pastur Operator Scaling + Realized Generator Pushforward Trust Band (`delta_g_max`):** Analytically normalizes layer operator gain by $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ in $O(\text{params})$ arithmetic, and measures realized generator output displacement $\delta_G = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ on a detached eval-mode probe to clamp `548×` overpowered generators before they overshoot into mode collapse.
-- **Fused $Z = [X, Y]$ Quintic Tikhonov Polar Orthogonalization & Zero-`pinv` Analytic `direction_blend`** ([`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Replaces $O(\min(m,n)^3)$ SVD with 5-step pure-GEMM Quintic Newton–Schulz iterations ($a=3.4445, b=-4.7750, c=2.0315$) batched across same-shape 2D matrices, plus exact closed-form $1\times 1 / 2\times 2$ non-ascent constraint projection (`zero_pinv_project_nonascent`, using standard `.backward()` with zero `autograd.grad` and zero `pinv`).
+- **Fused $Z = [X, Y]$ Quintic Tikhonov Polar Orthogonalization & Zero-`pinv` Analytic `direction_blend`** ([`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Replaces $O(\min(m,n)^3)$ SVD with 5-step pure-GEMM Quintic Newton–Schulz iterations (step-varying 5-step schedule in `unified_v4`/`v11`/`v12`; stationary $a=3.4445, b=-4.7750, c=2.0315$ in `t4_arch` and `convolution_quintic`) batched across same-shape 2D matrices, plus exact closed-form $1\times 1 / 2\times 2$ non-ascent constraint projection (`zero_pinv_project_nonascent`, using standard `.backward()` with zero `autograd.grad` and zero `pinv`).
 - **Asymmetric Zero-Conv2d / Zero-Critic NS Energy-Optimal Realizations** ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md)): Applies Laws 4.3–4.5 selectively by dropping redundant 4D Conv2d NS via stride-aware polar-matched Frobenius-MP (`law4-energy-v3`) and dropping 2D Critic NS via exact polar-Frobenius match $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ (`law4-energy-v10`), alongside zero-NS, zero-state research lower bounds (`law4-sign-mp`, `law4-frob-sgd-mp`).
@@ -166,4 +166,4 @@
 - **Imbalance-Invariant Nash (`t4_imbalance_nash_stretch`):** `t4-nash-v3` ([`readout-6f996ec2`](../reports/forge/records/readout-6f996ec256a149b2f1620f64.json), [`readout-642d4c55`](../reports/forge/records/readout-642d4c55d093b44ed1e0309e.json)) achieved **`3/3` (`100%`) sustained `5/5` PASS** with **strictly `0` SVD, `0` `eigh`, `0` `pinv`, `0` Newton–Schulz, and `0` `autograd.grad`** at **`39.68 ms/step`** on `256×256` RGB DCGAN (`3.92×` faster than `empf-v8`, `14.84×` faster than `rhbi-v12`), recovering `16/16` modes across a `100×` bidirectional LR-asymmetry sweep (`d_lr_mult` from `10.0` to `0.10`).
 - **Pathological 16-Layer & Rail Architectures (`t4-strange-architectures-v1`):** `t4-arch-v3` ([`readout-dc890b70`](../reports/forge/records/readout-dc890b704514d91effb156a6.json), [`readout-4acf96c2`](../reports/forge/records/readout-4acf96c2efad5e17f86d9df2.json)) jumped from `0/10` historical families to **`7/7` (`100%`) PASS** (including `grid100_mlp16` `prec = 0.9814`, `grid100_res16_tanh` `prec = 0.9856, sat = 0.0`, `grid100_res16_scaled_tanh` `prec = 0.9868`, `zero_tanh_rail_escape` `mae = 0.00105`), plus `3/3` fresh-process `14,000`-step native continuations (`grid100_14k`, `rotated100_14k`, `staggered100_14k`, [`readout-ff228439`](../reports/forge/records/readout-ff22843954240927dfd7ecc3.json)).
 - **Unified Reference (`unified-v11`):** **`3/3` Nash PASS** ([`readout-2eb07e97`](../reports/forge/records/readout-2eb07e97714a1da4177d5d25.json)), **`6/7` strange-arch PASS** ([`readout-b96d9de4`](../reports/forge/records/readout-b96d9de47bb52cc91fbb4a4f.json)), **`5/5` shift/rail PASS** ([`readout-2029c37a`](../reports/forge/records/readout-2029c37a1359498d3d8f2eb3.json)), and `residual_student` `1.0` (`wrong_pad_rate = 0.0`) via zero-`pinv` `direction_blend`.
-- **Asymmetric Energy-Optimal Law-4 Follow-ups ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md), [`law4-energy-champion.md`](../reports/forge/studies/law4-energy-champion.md)):** `law4-energy-v3` ([`readout-d53010c7`](../reports/forge/records/readout-d53010c7356ee6557df70807.json), [`readout-b898e928`](../reports/forge/records/readout-b898e9285957d3feba1bf714.json)) cuts `r256` to **`45.89 ms/step`** / **`12.56 J/step`** (`0` Conv2d NS, `0` critic secant probes); `law4-energy-v10` ([`readout-a0e3e7f6`](../reports/forge/records/readout-a0e3e7f6620e7476c9599d67.json)) adds `0` Critic NS to reach **`42/48`** Tiers 1–2 `discriminator_stability` PASS at **`45.30 ms/step`** and **`11.68 J/step`**; and **`unified-v12`** (`DEFAULT_RECIPE = "unified_v12"`, [`readout-5002641a`](../reports/forge/records/readout-5002641a64ddaa91f1fc3b87.json), [`readout-731e752f`](../reports/forge/records/readout-731e752ff3c4f97e29a84145.json)) adds generator-free 2D critic NS fallback + CountSketch stiff-generator preconditioning + zero-sync pullback fast-path to reach **`46/48` Tiers 1–2 `discriminator_stability` PASS (`29/29` required = `100%`, incl. `2/2` combined AE-DDGAN)**, **`18/18` (`100%`) untouched Tier-3/14k holdouts**, **`16/16` Occam**, **`7/7` standalone `t4-arch`** (`6/7` combined with PR #615: `grid100_res16_tanh` passes `100k` holdout `ctr=0.1224` with a 1-ULP step-`6250` `modes=99` dip), and **`39.22 ms/step` / `11.37 J/step`** on strictly idle RTX A6000 `r256` (**`-45.8%`** J/step vs post-PR-#612+#615 `unified-v11`, **`-52.0%`** vs pre-Phase-3 `unified-v11`).
+- **Asymmetric Energy-Optimal Law-4 Follow-ups ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md), [`law4-energy-champion.md`](../reports/forge/studies/law4-energy-champion.md)):** `law4-energy-v3` ([`readout-d53010c7`](../reports/forge/records/readout-d53010c7356ee6557df70807.json), [`readout-b898e928`](../reports/forge/records/readout-b898e9285957d3feba1bf714.json)) cuts `r256` to **`45.89 ms/step`** / **`12.56 J/step`** (`0` Conv2d NS, `0` critic secant probes); `law4-energy-v10` ([`readout-a0e3e7f6`](../reports/forge/records/readout-a0e3e7f6620e7476c9599d67.json)) adds `0` Critic NS to reach **`42/48`** Tiers 1–2 `discriminator_stability` PASS at **`45.30 ms/step`** and **`11.68 J/step`**; and **`unified-v12`** (`DEFAULT_RECIPE = "unified_v12"`, [`readout-5002641a`](../reports/forge/records/readout-5002641a64ddaa91f1fc3b87.json), [`readout-731e752f`](../reports/forge/records/readout-731e752ff3c4f97e29a84145.json)) adds generator-free 2D critic NS fallback + CountSketch period-2 flip-bifurcation gain controller + zero-sync pullback fast-path to reach **`46/48` Tiers 1–2 `discriminator_stability` PASS (`29/29` required = `100%`, incl. `2/2` combined AE-DDGAN)**, **`18/18` (`100%`) untouched Tier-3/14k holdouts**, **`16/16` Occam**, **`7/7` standalone `t4-arch`** (`6/7` combined with PR #615: `grid100_res16_tanh` passes `100k` holdout `ctr=0.1224` with a 1-ULP step-`6250` `modes=99` dip), and **`39.22 ms/step` / `11.37 J/step`** on strictly idle RTX A6000 `r256` (**`-45.8%`** J/step vs post-PR-#612+#615 `unified-v11`, **`-52.0%`** vs pre-Phase-3 `unified-v11`).
```

#### (2) Proposed docstring corrections for `particlegan/` code comments (NOT applied here)

```diff
--- a/particlegan/_unified_v11/optimizer.py
+++ b/particlegan/_unified_v11/optimizer.py
@@ -67,7 +67,7 @@
     smoothing: float | torch.Tensor = 0.0,
 ) -> torch.Tensor:
-    """Zero-SVD scale-normalized 5-step Quintic polar (exact FP32 Tikhonov for ``m<=64``, BF16 for ``65..128``)."""
+    """Zero-SVD scale-normalized 5-step Quintic polar (FP32 Tikhonov for ``m<=256``, BF16 for ``257..512``; non-Tikhonov BF16 on CUDA for ``m>=64``)."""
@@ -159,7 +159,7 @@
     smoothing: float | torch.Tensor = 0.0,
 ) -> torch.Tensor:
-    """Batched zero-SVD 5-step Quintic polar (exact FP32 Tikhonov for ``m<=64``, BF16 for ``65..128``)."""
+    """Batched zero-SVD 5-step Quintic polar (FP32 Tikhonov for ``m<=256``, BF16 for ``257..512``; non-Tikhonov BF16 on CUDA for ``m>=64``)."""

--- a/particlegan/_t1_coupling/deconflict.py
+++ b/particlegan/_t1_coupling/deconflict.py
@@ -7,7 +7,7 @@
 Preconditioning in the whitened activation metric ``P = (Sigma_H +
 lambda * mean_eig * I)^{-1/2}`` via ``polar(G_W P) P`` orthogonalizes mode
-subspaces using pure FP32 coupled Newton-Schulz iterations with zero persistent
+subspaces using exact FP32 ``torch.linalg.eigh`` inverse square roots with zero persistent
 state, zero checkpoint overhead, and zero RNG draws.
```
