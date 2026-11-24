# P1 Adversarial Mathematical & Empirical Audit of Law 4.2: Scale Invariance vs. Continuous Settling

- **Track:** P1 — Law 4.2 (*Intra-Player Degree-0 Gauge & Rail Escape*)
- **Branch:** `research/fl-p1-scale` (branched from `origin/research/four-laws`)
- **Reproducible Scripts & Tests:**
  - [`scripts/four_laws_audit/p1_scale_settling.py`](../../scripts/four_laws_audit/p1_scale_settling.py)
  - [`tests/four_laws_audit/test_p1_law42_scale_settling.py`](../../tests/four_laws_audit/test_p1_law42_scale_settling.py)
- **Verification Command:**
  ```bash
  PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p1_scale_settling.py
  PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/pytest tests/four_laws_audit/test_p1_law42_scale_settling.py -v
  ```

---

## Executive Summary

Law 4.2 in [`docs/four-laws.md`](../../docs/four-laws.md#L86) and [`reports/forge/studies/law4-optimizer-laws.md`](../forge/studies/law4-optimizer-laws.md#L206-L210) demands **scale-invariant (degree-0) layerwise conditioning that simultaneously settles smoothly to zero ($\Delta W \to 0$ as $G \to 0$)** without a step-count clock. Our adversarial mathematical and code-level audit establishes five decisive results:

1. **Formal Impossibility of the Memoryless Statement (Theorem 1):** Every nontrivial memoryless update operator $U: \mathbb{R}^d \to \mathbb{R}^d$ satisfying positive 0-homogeneity $U(cg) = U(g)$ ($\forall c > 0$) is **necessarily discontinuous at $0$** ($\liminf_{c \to 0^+} \|U(cg)\| = \|U(g)\| > 0$) and enters an unvanishing $\Omega(\eta)$ limit cycle on smooth strongly convex objectives. Neither [`docs/four-laws.md`](../../docs/four-laws.md) nor any Forge study report currently acknowledges this impossibility.
2. **The Real Discovery Is *Stateful Trajectory* Degree-0 Invariance (Theorem 3) — Bounded by Sharp Information-Theoretic Obstructions (Theorem 4):** Causal stateful trajectory operators $U_t(g_1, \dots, g_t)$ satisfying $U_t(cg_1, \dots, cg_t) = U_t(g_1, \dots, g_t)$ **can** settle continuously whenever $\|F_t(g_{1:t})\| / S_t(g_{1:t}) \to 0$ (e.g., uncapped `peak_rms` ratio $\|g_t\| / \max_{s \le t} \|g_s\|$). However, **every first-order trajectory-degree-0 operator faces fundamental obstructions**:
   - *Step-1 Near-Equilibrium Kick (Obstruction 4.1):* Starting at distance $\delta = 10^{-6}$ from a quadratic minimum forces a $>10^4\times$ step-1 overshoot, because $g_1 = \delta$ is indistinguishable from a $10^{-6}$-rescaled objective far from equilibrium.
   - *Time-Varying Rescaling vs. Convergence Indistinguishability (Obstruction 4.2):* No causal gradient-only operator can be invariant to time-varying rescalings $c_t > 0$ while settling on $g_t \to 0$.
   - *Peak-Memory Trade-Off (Theorems 3a–3c):* Monotonic `peak_rms` ($\gamma = 1$) settles on all $g_t \to 0$ but **irreversibly freezes** after a single transient spike or downward scale shift; exponentially decaying `peak_rms` ($\gamma = 0.9995$) **fails to settle** on every sub-exponential decay ($g_t \sim t^{-p}$ turns around at $t = p/(1-\gamma) = 2000p$ and reverts to $100\%$ uncooled step norm!) while still stalling for $\sim 18{,}400$ steps after a $10^4\times$ spike; sliding-window `peak_rms` ($W = 50$) recovers in $W$ steps but **never settles** on smooth convergence.
3. **Momentum Coherence (`||m|| / ||v||`) Does *Not* Converge to $0$ on Oscillatory/Noisy Equilibria (Theorem 3d — Falsified Claim):** Contrary to [`docs/four-laws.md`](../../docs/four-laws.md#L160) and [`t4-strange-architectures.md`](../forge/studies/t4-strange-architectures.md#L65), fixed-$\beta_1$ coherence converges to strictly positive floors: $\sqrt{(1-\beta_1)/(1+\beta_1)}$ on i.i.d. zero-mean noise (`0.3780` at $\beta_1=0.75$, `0.3333` at $\beta_1=0.80$; verified to `0.002` empirically), $(1-\beta_1)/(1+\beta_1)$ on exact period-2 sign flipping (`0.1111` at $\beta_1=0.80$; verified to `3.89e-8`), and $(1-\beta_1)/\sqrt{1 - 2\beta_1\cos\omega + \beta_1^2}$ on bilinear rotation (near `1.0` for small $\omega$, where momentum phase lag **accelerates divergence**!). Moreover, in default `unified_v11` and `unified_v12` ($\beta_1 = 0$), coherence is **identically `1.0`** on all non-stiff layers (`stiff_gate == 0`).
4. **Six Concrete Code-vs.-Claim Defects Found Directly in `particlegan/`:**
   - **Defect A (`min(s_base, c * peak_rms)` breaks scale invariance above $s_{\text{base}}/c$):** In [`_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py#L337) (`s_base=0.0061, c=0.50`) and [`_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py#L994) (`c=0.30`), `s_eff` saturates at fixed `0.0061` whenever `peak_rms > 0.0122` (`v12`) or `0.0203` (`v11`). Thus actual `UnifiedV12Optimizer` and `UnifiedV11EquilibratedOptimizer` **break scale invariance between $c = 10^{-6}$ and $c = 1$** (`max_diff = 0.326` / `0.238`) and **lose `tanh`-rail settling even at $c = 1$** (`tail_step = 0.158` vs `0.000` at $c = 10^{-6}$)!
   - **Defect B (Catch-22 permanent spike freezing in `UnifiedV12Optimizer`):** Because [`_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py#L688) sets `p_dec = 1.0` on all 1-objective and unconditional tasks, whenever a layer operates in the actual `peak_rms` regime (`c = 1e-6`, below `s_base`), a single $10^4\times$ spike **permanently freezes `UnifiedV12Optimizer`** (`post/pre = 7.77e-4`, `final_err = 3.024` vs `0.016` at $c = 1$, where it only escaped because Defect A broke scale invariance!).
   - **Defect C (Additive `eps = 1e-8` breaks `Minimal` and `T4NashV3` on $10^{-12}$ rails):** Both [`_minimal/dynamics.py`](../../particlegan/_minimal/dynamics.py#L172) (and slow-path line 186 `(0.015 * peak + eps) / sc`!) and [`_nash_v3/optim.py`](../../particlegan/_nash_v3/optim.py#L77) add unscaled `eps = 1e-8` directly to `0.015 * peak_rms`, crushing $10^{-12}$ gradients by $10^{-4}\times$ (`step1_move = 3.17e-6 / 2.47e-6`, deadlocked on the `sech^2(14) = 2.76e-12` rail!) and refuting "Theorem 4A.1 (exact degree-0 homogeneous)" in [`t4-imbalance-nash.md`](../forge/studies/t4-imbalance-nash.md#L38-L40).
   - **Defect D (FP32 `cur_sv_rms` underflow at $10^{-25}$ in `UnifiedV11EquilibratedOptimizer`):** [`_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py#L973-L979) computes raw `gf_safe.norm()`, which squares elements in FP32 and underflows to `0.0` at $c = 10^{-25}$, zeroing `peak_rms` and jumping step norm from `0.0465` to `0.0566` (fixed in `UnifiedV12Optimizer`'s main path at line 326, still present in `_step_generator_free_critic` lines 506–512).
   - **Defect E (1D parameter `clamp_min(eps=1e-8)` crush in `UnifiedV11EquilibratedOptimizer`):** [`_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py#L1106) clamps 1D denominators at `eps = 1e-8`, crushing 1D gradients at $c = 10^{-12}$ by $6{,}600\times$ (`ratio = 1.51e-4`), whereas `UnifiedV12Optimizer` preserves `ratio = 1.0000`.
   - **Defect F (Dead `peak_rms` tracking on prior rows):** Both [`_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py#L878-L895) and [`_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py#L827-L845) compute and checkpoint `st["peak_rms"]` for prior rows, then ignore it completely and plug the constant `s_prior_base = 0.0061` into `hypot(d_norm, s_p_row)`.
5. **No Stabilization of Pure Adversarial Rotation (Regime 5):** On simultaneous bilinear games $\min_x \max_y x^\top y$, radius $r_t$ is strictly increasing ($r_T / r_0 = \sqrt{1 + T\eta_{\text{eff}}^2}$), so $\|g_t\| / P_t \equiv 1$ forever (`peak_rms` is identically inactive!) while momentum coherence accelerates outward spiraling (`r_ratio = 1.76` / `3.83` vs `1.11` for zero-momentum). Equilibrium settling in ParticleGAN relies heavily on **external data-space residual gating (`_step_scale`, `_thermal_scale`) and Law 2's Born–Infeld slope damping**, not on `peak_rms` or coherence alone.

---

## 1. Exhaustive Code & Documentation Audit

### 1.1 Inventory of Documentation Claims

Does [`docs/four-laws.md`](../../docs/four-laws.md) or any Forge study report acknowledge the mathematical incompatibility between memoryless degree-0 scale invariance and continuity at zero? **No.** Across all audited documents, zero passages state the impossibility theorem or distinguish memoryless pointwise homogeneity from stateful trajectory homogeneity.

| Document & Exact Lines | Verbatim / Paraphrased Claim | Acknowledges Memoryless Impossibility? |
| :--- | :--- | :---: |
| [`docs/four-laws.md`](../../docs/four-laws.md#L86) | Law 4.2: *"Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish ($\Delta W \to 0$ as $G \to 0$) is required without a step clock."* | **No** (conflates pointwise degree-0 with continuous settling at 0) |
| [`docs/four-laws.md`](../../docs/four-laws.md#L93) | *"Scale-Invariant Layer & Rail Equalization: Equalize extreme depth attenuation and saturation rails without underflow dead zones, while settling cleanly at equilibrium with no step clock."* | **No** |
| [`docs/four-laws.md`](../../docs/four-laws.md#L115) | *"Dynamic-$10^{-38}$ scale norm + homogeneous `peak_rms` settling"* | **No** |
| [`docs/four-laws.md`](../../docs/four-laws.md#L160) | *"Normalizes $(m_t, v_{\text{rms},t})$ by $s_t = \max(\max |v_{\text{rms},t-1}|, \max |g_t|, 10^{-38})$ ... while degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock."* | **No** (also falsely claims $\text{coherence} \to 0$ and omits `min(s_base, ...)` cap) |
| [`docs/four-laws.md`](../../docs/four-laws.md#L187) | *"`zero_tanh_rail_escape` still passes at `mae = 0.0026` without `w_sat_pt` because $10^{-38}$ dynamic scale normalization already inverts `sech^2`."* | **No** (note `zero_tanh_rail_escape` has $\operatorname{sech}^2 \approx 0.19$, not $10^{-12}$) |
| [`reports/forge/studies/law4-optimizer-laws.md`](../forge/studies/law4-optimizer-laws.md#L206-L210) | Law 4.2 formula: $\Delta W_l = -\eta \gamma_{\text{MP}} \frac{G_l}{\sqrt{\|G_l\|^2 + s_{\text{eff}}^2}}$, $s_{\text{eff}} = \min(s_{\text{base}}, c\,\text{peak\_rms}_l)$ described as *"Degree-0 scale-invariant ... while preserving smooth equilibrium settling ($G_l \to 0 \implies \Delta W_l \to 0$) without a step clock."* | **No** (formula itself breaks degree-0 homogeneity for $c\,\text{peak\_rms}_l > s_{\text{base}}$!) |
| [`reports/forge/studies/t4-imbalance-nash.md`](../forge/studies/t4-imbalance-nash.md#L38-L40) | *"Theorem 4A.1 (Homogeneous Peak-RMS Settling): $u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + 0.015 \sigma_t^{\text{peak}} + \varepsilon}$. Unlike fixed damping $\varepsilon_0 > 0$, $u_t$ is **exact degree-0 homogeneous** under arbitrary constant gradient rescaling $g_t \mapsto c\cdot g_t$ ($c > 0$), yet contracts smoothly as $O(\hat{\sigma}_t / \sigma_t^{\text{peak}}) \to 0$."* | **No** (actually has additive $+\varepsilon = 10^{-8}$, raw $g^2$ underflow, and $\gamma_{\text{peak}}=0.9995$!) |
| [`reports/forge/studies/r3-nash-complete.md`](../forge/studies/r3-nash-complete.md#L21) | Summarizes `peak_rms` from `t4-nash-v2` (and docstring of [`_nash_v3/optim.py`](../../particlegan/_nash_v3/optim.py#L4) claims *"Degree-0 scale-invariant `peak_rms` equilibrium settling"*). | **No** |
| [`reports/forge/studies/t4-strange-architectures.md`](../forge/studies/t4-strange-architectures.md#L46-L65) | Claims $10^{-38}$ dynamic scale norm cancels $s_t$ identically in ratios of $(\tilde{m}_t, \tilde{v}_t)$ and *"at stationary Nash equilibrium (oscillatory minibatch gradients), $\|\tilde{m}_t\|_F \ll \|\tilde{v}_t\|_F$ automatically cools $\text{coherence} \to 0$ without any external step clock."* | **No** (coherence stays at $\ge \frac{1-\beta_1}{1+\beta_1} > 0$, never $\to 0$) |
| [`reports/forge/studies/r4-minimal-core.md`](../forge/studies/r4-minimal-core.md#L33,L61) | Claims *"pure-elementwise $10^{-38}$ MP + `peak_rms`"* and *"falling back seamlessly to dynamic-$10^{-38}$ scale-normalized moments under extreme vanishing/overflowing gradients."* | **No** (omits unscaled `+ eps` bug in `_minimal/dynamics.py:172,186`) |
| [`reports/forge/studies/r4-occam-ablation-razor.md`](../forge/studies/r4-occam-ablation-razor.md#L26,L30,L95,L134) | Attributes `zero_tanh_rail_escape` pass to $10^{-38}$ dynamic scale norm; calls `r4_mp_no_ns` *"$10^{-38}$-safe `peak_rms`"* (which still multiplies by unscaled `s_eff = 0.0061` at `_r4_ablation/optimizer.py:295,318`). | **No** |

### 1.2 Implementation Audit Matrix Across Every Optimizer in `particlegan/`

| Implementation & Lines | Default Recipe Params (`recipes.py`) | Dynamic `1e-38` Norm? | `peak_rms` Formula & Decay `p_dec` | Coherence `\|\|m\|\|/\|\|v\|\|` Active? | Actual Homogeneity Range ($c$ for $\|g_0\| \sim 1$) | Exact Failure / Breakpoint Found in Code |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **`UnifiedV12Optimizer`** *(Default)* [`_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py#L305-L454,L688,L1008-L1084) | `lr=0.014`, `betas=(0, 0.999)`, `eps=1e-8`, `smoothing=0.0061` ([`recipes.py:2816`](../../particlegan/recipes.py#L2816-L2819)) | Yes (slow path `<1e-18` / `>1e18`; fast path `[1e-18, 1e18]`) | `s_eff = min(0.0061, 0.50 * peak_rms)`; **`p_dec = 1.0`** on all 1-obj/uncond tasks (`0.9995` only on 2-obj cond-fiber, line 688) | Only when `ema_fg < 0.60` (`stiff_gate > 0`) or scalar 1D (`clamp(0.35, 1.0)`); **`1.0` otherwise** | `[1e-37, ~0.012]` (below `s_base/0.50`) | **Breaks above `peak_rms = 0.0122`** (`s_base` cap); **permanently freezes** on spikes when `peak_rms < 0.0122` (`p_dec = 1.0`); dead `peak_rms` on prior rows (lines 828–845); `_step_generator_free_critic` line 509 uses raw `gf.norm()` |
| **`UnifiedV11EquilibratedOptimizer`** [`_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py#L878-L1118) | Same (`smoothing=0.0061`, [`recipes.py:2809`](../../particlegan/recipes.py#L2809-L2815)) | Partial (`m/v` factored; `cur_sv_rms` uses raw `gf.norm()`, line 973–979) | `s_eff = min(0.0061, 0.30 * peak_rms)`; `p_dec = 0.9995` if `s_therm >= 0.5` else `1.0` (line 985) | Only when `ema_fg < 0.18` (`stiff_gate > 0`) or scalar 1D (`clamp(0.35, 1.0)`) | `[1e-22, ~0.020]` on 2D; **`[1e-6, ~0.02]` on 1D** | **Breaks above `0.0203`** (`s_base` cap); **`cur_sv_rms` underflows to `0` at `1e-25`**; **1D `clamp_min(eps=1e-8)` crushes `1e-12` by `6600x`** (line 1106); dead `peak_rms` on prior rows (lines 878–940) |
| **`T4EquilibratedOptimizer`** [`_t4_arch/optimizer.py`](../../particlegan/_t4_arch/optimizer.py#L304-L349) | `lr=0.016`, `betas=(0.80, 0.97)`, `smoothing=0.0` ([`recipes.py:2595`](../../particlegan/recipes.py#L2595-L2604)) | Yes (`scale = max(amax(v), g_max, 1e-38)`), unfactored storage in `p.dtype` | **None** (uses coherence + weight-norm `r_trust` instead) | **Always** (`b1_eff = 0.80` for `L >= 2`, `0.20` for `L == 1`) | **`[1e-37, 1e35]`** on FP32 2D+ (truly 24+ decades!) | Does **not** settle to `0` on noisy/oscillatory equilibria (floors at `0.333` i.i.d., `0.111` flip) or slow `1/t` decay (`coh -> 1.0`); unfactored `m/v` overflows/underflows in FP16/BF16 |
| **`MinimalGeneratorOptimizer`** [`_minimal/dynamics.py`](../../particlegan/_minimal/dynamics.py#L163-L188) | `lr=0.0135`, `betas=(0, 0.999)`, `eps=1e-8` ([`recipes.py:2824`](../../particlegan/recipes.py#L2824-L2831)) | Factored `m/v` on slow path (`g < 1e-18`), **undone by `+ eps`** | `denom = vh + (0.015 * peak + eps) / sc` (line 186); `p_dec = 0.9995` | Only when `deep_g > 0` (`n_m > 5`) | `[~1e-6, 1e18]` | **Unscaled `+ eps = 1e-8` crushes `1e-12` gradients by `10^-4` on BOTH fast and slow paths** (lines 172 & 186); deadlocks on `1e-12` `tanh` rail; stalls on spikes (`0.9995`) |
| **`NashV3GeneratorAdam` / `T4NashGeneratorAdam`** [`_nash_v3/optim.py`](../../particlegan/_nash_v3/optim.py#L72-L80), [`_t4_nash/optim.py`](../../particlegan/_t4_nash/optim.py#L118-L125) | `lr=0.0135`, `betas=(0, 0.999)`, `eps=1e-8` ([`recipes.py:2714`](../../particlegan/recipes.py#L2714-L2731)) | **No** (raw `g * g` in FP32, underflows `<1e-22`) | `denom = v_hat_sqrt + 0.015 * peak + eps`; `p_dec = 0.9995` | **No** | `[~1e-6, 1e18]` | **Refutes Theorem 4A.1**: `+ eps = 1e-8` breaks degree-0 below `1e-6`; raw `g^2` underflows to `0` at `1e-22`; `+ 1e-36` floors `cur_rms` at `1e-18` |
| **`R4MPNoNSOptimizer`** [`_r4_ablation/optimizer.py`](../../particlegan/_r4_ablation/optimizer.py#L208-L324) | `smoothing=0.0061`, `settle_rel=0.015` | Yes (dimensionless `1e-8` in `denom_elem`, line 269) | `peak_rms` (`0.9995`, `peak_scaled` clamped `1e6`) **plus** unscaled `s_eff = 0.0061` (lines 295, 318) | Only when `deep_g > 0` or scalar 1D | `[1e-37, ~0.006]` (broken by `s_eff = 0.0061` multiplier) | Unscaled `s_eff = 0.0061` factor multiplies `u_elem`, re-introducing fixed-$\varepsilon$ rail deadlock when `deep_g == 0` |
| **`UnifiedV4EquilibratedOptimizer`** [`_unified_v4/optimizer.py`](../../particlegan/_unified_v4/optimizer.py#L585-L695) | `smoothing=0.0061` ([`recipes.py:2795`](../../particlegan/recipes.py#L2795-L2800)) | Yes (only when `use_moments`), raw `s_eff = 0.0061` | **No `peak_rms`** — raw constant `s_eff = 0.0061 * (1 - deep_gate)` | Only when `deep_g > 0` or scalar 1D | Breaks everywhere `s_eff > 0` | Fixed constant `s_eff = 0.0061` crushes small gradients on shallow nets (`deep_gate = 0`) |
| **`Law4EnergyV1..V11Optimizer`** [`_law4_energy/optimizer.py`](../../particlegan/_law4_energy/optimizer.py#L134-L140,L3721-L3733,L4429) | `smoothing=0.0061` | Yes (slow path) | `v1..v6`: `min(s_base, 0.03 * peak)`, `p_dec=0.9995`; `v7..v11`: `min(s_base, 0.50 * peak)`, `p_dec=1.0` | Only when `ema_fg < 0.60` (`stiff_gate > 0`) | Same `s_base` ceiling as `v12` | Same Defect A (`s_base` cap) and Defect B (`p_dec = 1.0` in `v7..v11`) |
| **`NormalizedOptimizer` (`DualNorm`)** [`optim/dualnorm.py`](../../particlegan/optim/dualnorm.py#L152-L161,L903-L940) | `smoothing=0.001..0.0061` | **No** (hard `norm < eps` skip, fixed `smoothing`) | **None** | **No** | Fixed-$\varepsilon$ crossover at `smoothing` | Pure fixed-$\varepsilon$ Tikhonov polar; deadlocks when $\|G\| \ll \text{smoothing}$ |

---

## 2. Formal Mathematical Theorems & State-Space Characterization

### 2.1 Theorem 1 — Memoryless Impossibility Theorem

> **Theorem 1 (Incompatibility of Memoryless Positive 0-Homogeneity and Continuous Settling at Zero).**
> Let $U: \mathbb{R}^d \to \mathbb{R}^d$ be a memoryless update operator satisfying **positive 0-homogeneity**:
> $$U(c g) = U(g) \qquad \forall c > 0,\;\forall g \in \mathbb{R}^d \setminus \{0\}.$$
> If $\lim_{g \to 0} U(g) = 0$ (in particular, if $U(0) = 0$ and $U$ is continuous at $0$), then $U(g) = 0$ identically for all $g \in \mathbb{R}^d$.

**Proof.** Fix any $g \in \mathbb{R}^d \setminus \{0\}$ and define the sequence $g_n := \frac{1}{n} g$ for $n \in \mathbb{N}$. Then $\lim_{n \to \infty} g_n = 0$. By positive 0-homogeneity with $c = 1/n > 0$, $U(g_n) = U(g)$ for every $n \ge 1$. Taking the limit as $n \to \infty$ yields
$$U(g) = \lim_{n \to \infty} U(g_n) = \lim_{h \to 0} U(h) = 0.$$
Since $g \neq 0$ was arbitrary and $U(0) = 0$, $U \equiv 0$ on $\mathbb{R}^d$. $\square$

> **Corollary 1.1 (Essential Jump Discontinuity & Persistent Limit-Cycle Chatter).**
> Every nontrivial ($U \not\equiv 0$) memoryless positive 0-homogeneous operator is discontinuous at $0$ along every active ray: for any $g \neq 0$ with $U(g) \neq 0$,
> $$\liminf_{c \to 0^+} \|U(cg)\| = \|U(g)\| > 0, \qquad \limsup_{h \to 0} \|U(h)\| \ge \sup_{u \in S^{d-1}} \|U(u)\| > 0.$$
> In particular, for normalized gradient descent $x_{t+1} = x_t - \eta\, U(\nabla L(x_t))$ with $\|U(g)\| = 1$ ($\forall g \neq 0$) on the 1D quadratic $L(x) = \frac{1}{2} x^2$ with initial point $x_0 \notin \eta \mathbb{Z}$, the iterate enters an exact period-2 limit cycle between $r \in (0, \eta)$ and $r - \eta \in (-\eta, 0)$, satisfying $\limsup_{t \to \infty} |x_t| = \max(r, \eta - r) \ge \eta / 2$ forever.

> **Corollary 1.2 (Extension to Matrix / Spectral Operators).**
> The exact same obstruction applies to every memoryless matrix operator $U: \mathbb{R}^{m \times n} \to \mathbb{R}^{m \times n}$ satisfying $U(c G) = U(G)$ for $c > 0$ — including Frobenius normalization $G / \|G\|_F$, elementwise $\operatorname{sign}(G)$, and exact or finite-step scale-normalized Newton–Schulz polar projection $G (G^\top G)^{-1/2}$ ($\|U(cG)\|_F = \sqrt{\operatorname{rank}(G)} \ge 1$ for all $c > 0$).

---

### 2.2 Theorem 2 — Fixed-$\varepsilon$ Crossover Impossibility & Capped-`peak_rms` Ceiling

> **Theorem 2 (No Simultaneous Rail Escape and Settling Under Fixed $\varepsilon > 0$).**
> Consider any fixed-floor regularized operator on $\mathbb{R}^d$,
> $$U_\varepsilon^{\text{add}}(g) = \frac{g}{\|g\| + \varepsilon} \quad \text{or} \quad U_\varepsilon^{\text{Tikh}}(g) = \frac{g}{\sqrt{\|g\|^2 + \varepsilon^2}}, \qquad \varepsilon > 0,$$
> with normalized step gain $\phi(u) := \|U_\varepsilon(g)\| \in [0, 1)$ as a function of $u = \|g\|/\varepsilon$ ($\phi_{\text{add}}(u) = \frac{u}{u+1}$, $\phi_{\text{Tikh}}(u) = \frac{u}{\sqrt{u^2 + 1}}$). Because $\phi$ is strictly increasing on $(0, \infty)$:
> 1. Maintaining $(1-\delta)$-approximate scale invariance ($\phi(g_{\text{rail}}/\varepsilon) \ge 1-\delta$) down to a rail gradient $g_{\text{rail}}$ requires $\varepsilon \le g_{\text{rail}} / \phi^{-1}(1-\delta)$ (e.g., $\delta = 0.1, g_{\text{rail}} = 10^{-12} \implies \varepsilon \le 1.11 \times 10^{-13}$ additive or $4.84 \times 10^{-13}$ Tikhonov).
> 2. Achieving contraction $\phi(g_{\text{eq}}/\varepsilon) \le \tau$ at equilibrium gradient $g_{\text{eq}}$ requires $\varepsilon \ge g_{\text{eq}} / \phi^{-1}(\tau)$ (e.g., $\tau = 0.1, g_{\text{eq}} = 10^{-3} \implies \varepsilon \ge 9 \times 10^{-3}$ additive or $9.95 \times 10^{-3}$ Tikhonov).
> 3. **Impossibility Band:** Whenever $\frac{g_{\text{rail}}}{g_{\text{eq}}} \le \frac{\phi^{-1}(1-\delta)}{\phi^{-1}(\tau)}$ (and in particular whenever $g_{\text{rail}} \le g_{\text{eq}}$ and $\delta + \tau \le 1$), **no fixed $\varepsilon > 0$ can satisfy both requirements**. Specifically, no $\varepsilon > 0$ can preserve $\ge 50\%$ step magnitude down to $10^{-12}$ while providing $\le 50\%$ settling at any gradient norm above $10^{-12}$.

> **Proposition 2.1 (Scale-Invariance Ceiling of ParticleGAN's Capped Floor $s_{\text{eff}} = \min(s_{\text{base}}, \alpha P_t)$).**
> Let $s_{\text{eff}}(g_{1:t}) = \min(s_{\text{base}}, \alpha P_t(g_{1:t}))$ with fixed $s_{\text{base}} > 0$, $\alpha > 0$, and positive 1-homogeneous $P_t$. Under global static rescaling $g_s \mapsto c g_s$ ($\forall s \le t$):
> - For $c \le P_{\text{crit}} / P_t(g_{1:t})$ where $P_{\text{crit}} := s_{\text{base}} / \alpha$ (`0.0122` in `UnifiedV12Optimizer`, `0.0203` in `UnifiedV11EquilibratedOptimizer`), $s_{\text{eff}}$ is exact degree-1 and the update is **trajectory degree-0 invariant**.
> - For $c > P_{\text{crit}} / P_t(g_{1:t})$, $s_{\text{eff}} \equiv s_{\text{base}}$ saturates at a constant, breaking degree-0 invariance and shrinking the effective settling ratio to $s_{\text{base}} / (c \|g_t\|) \to 0$ as $c \to \infty$ (destroying equilibrium settling at large objective scales!).

---

### 2.3 Theorem 3 — Stateful Trajectory Degree-0 Invariance & Sharp Failure Modes

> **Definition (Trajectory Positive 0-Homogeneity).**
> A causal trajectory operator $\mathcal{U} = (U_t)_{t \ge 1}$ with $U_t: (\mathbb{R}^d)^t \to \mathbb{R}^d$ is **trajectory scale-invariant (degree-0)** if for every $t \ge 1$, every history $g_{1:t} = (g_1, \dots, g_t)$, and every static global scalar $c > 0$,
> $$U_t(c g_1, c g_2, \dots, c g_t) = U_t(g_1, g_2, \dots, g_t).$$

> **Theorem 3 (Compatibility of Stateful Trajectory Scale Invariance and Continuous Settling).**
> Let $F_t: (\mathbb{R}^d)^t \to \mathbb{R}^d$ and $S_t: (\mathbb{R}^d)^t \to [0, \infty)$ be any causal **positive 1-homogeneous** functionals ($F_t(c g_{1:t}) = c F_t(g_{1:t})$, $S_t(c g_{1:t}) = c S_t(g_{1:t})$ for all $c > 0$). Then for any $\alpha > 0$,
> $$U_t(g_{1:t}) := \frac{F_t(g_{1:t})}{\sqrt{\|F_t(g_{1:t})\|^2 + \alpha^2 S_t(g_{1:t})^2}}$$
> satisfies:
> 1. **Exact Trajectory Degree-0 Invariance** for all $c > 0$ (in floating-point arithmetic, preserved across the entire normal exponent range down to $10^{-38}$ when $(F_t, S_t)$ are computed in factored mantissa–exponent form).
> 2. **Continuous Settling Along Relative-Decay Trajectories:** Whenever $\lim_{t \to \infty} \frac{\|F_t(g_{1:t})\|}{S_t(g_{1:t})} = 0$ (with $S_t > 0$), $\|U_t(g_{1:t})\| \le \frac{\|F_t(g_{1:t})\|}{\alpha S_t(g_{1:t})} \to 0$.
> 3. **Rail-Escape-Then-Settle Compatibility:** Along a 3-phase trajectory starting on a tiny rail ($\|g_1\| = 10^{-12}$), climbing to $\|g_{t_{\text{mid}}}\| = O(1)$, and decaying to $0$, $S_t$ starts at $10^{-12}$ (giving full step $1/\sqrt{1+\alpha^2}$ on the rail), rises to $O(1)$ during escape, and then contracts $U_t = O(\|g_t\| / S_t) \to 0$ at equilibrium — identically for every static rescaling $c > 0$.

However, **every concrete choice of causal scale functional $S_t$ incurs a provable structural failure mode**:

- **Theorem 3a (Monotonic Peak $P_t = \max_{s \le t} \|g_s\|$, $\gamma = 1$):** Settles on *every* vanishing sequence $g_t \to 0$ (since $P_t \ge \|g_1\| > 0$), **but** suffers **irreversible spike ratcheting**: a single transient outlier $\|g_{t_0}\| = M$ forces $P_t \ge M$ for all $t \ge t_0$, permanently suppressing steps by factor $\le \|g_t\| / (\alpha M)$. On constant-norm zero-mean noise ($\|g_t\| \equiv \sigma$), $\|g_t\| / P_t \equiv 1$ gives **zero settling**; on i.i.d. Gaussian noise, $P_t \sim \sigma \sqrt{d + 2\ln t}$ decays only at the $O(1/\sqrt{\ln t})$ extreme-value rate.
- **Theorem 3b (Exponentially Decaying Peak $P_t^{(\gamma)} = \max(\gamma P_{t-1}^{(\gamma)}, \|g_t\|)$, $\gamma \in (0, 1)$):** Recovers from a spike $M / g_{\text{typ}}$ in $T_{\text{rec}} = \lceil \ln(M / g_{\text{typ}}) / \ln(1/\gamma) \rceil$ steps (`18,417` steps for $10^4\times$ spike at $\gamma = 0.9995$), **but fails to settle on every sub-exponential convergence rate**: for any sequence with $\liminf_{t \to \infty} \|g_{t+1}\| / \|g_t\| > \gamma$ — including **every** algebraic rate $\|g_t\| = c t^{-p}$ ($p > 0$) and slow exponential rate $r^t$ ($r > \gamma$) — the peak decays faster than $\|g_t\|$ once $t > p / (1 - \gamma)$ (`t = 2000` for $p=1, \gamma=0.9995$), causing **non-monotonic anti-settling** (step norm grows while $\|g_t\|$ shrinks!) until $P_t^{(\gamma)} = \|g_t\|$ identically (`t ≈ 19,900`), reverting permanently to $\|U_t\| = 1/\sqrt{1+\alpha^2} > 0$.
- **Theorem 3c (Sliding-Window Peak $P_t^{(W)} = \max_{t-W < s \le t} \|g_s\|$, finite $W$):** Recovers from any spike of arbitrary magnitude $M$ in **strictly $W$ steps**, **but cannot settle to zero on any smooth convergence**: for algebraic $t^{-p}$, $\|g_t\| / P_t^{(W)} = (1 - W/t)^p \to 1$; even for exponential $r^t$, $\|g_t\| / P_t^{(W)} \equiv r^{W-1} > 0$ (strictly positive floor!).
- **Theorem 3d (Exact Asymptotic Floors of Momentum Coherence $C_t = \min(1, \|\hat{m}_t\|_F / \|\hat{v}_t\|_F)$ for Fixed $\beta_1 \in (0, 1)$):**
  1. *Slow monotonic convergence ($\|g_t\| = t^{-p}$ or $r^t$ with $r > \sqrt{\beta_2}$):* $\hat{m}_t \sim g_t$ and $\hat{v}_t \sim |g_t|$, so $C_t \to \mathbf{1.0}$ (**zero settling**!). Coherence only vanishes on deterministic convergence when $\|g_t\|$ decays *faster* than $\sqrt{\beta_2}^t$ — i.e., via $\beta_2$ memory lag acting as a decaying peak reference, NOT directional cancellation!
  2. *I.i.d. zero-mean stationary noise ($g_t \sim (0, \sigma^2 I_d)$):* $\mathbb{E}[\|\hat{m}_t\|^2] \to \frac{1-\beta_1}{1+\beta_1} d\sigma^2$ while $\mathbb{E}[\|\hat{v}_t\|^2] \to d\sigma^2$, so by the law of large numbers (and in RMS ratio for all $d$),
     $$C_t \xrightarrow{t \to \infty} \sqrt{\frac{1 - \beta_1}{1 + \beta_1}} > 0 \qquad (\text{e.g. } 0.3780 \text{ at } \beta_1=0.75,\; 0.3333 \text{ at } \beta_1=0.80).$$
  3. *Deterministic period-2 flipping ($g_t = (-1)^t g_0$):* $\hat{m}_t \to (-1)^t \frac{1-\beta_1}{1+\beta_1} g_0$, so
     $$C_t \xrightarrow{t \to \infty} \frac{1 - \beta_1}{1 + \beta_1} > 0 \qquad (\text{e.g. } 0.1429 \text{ at } \beta_1=0.75,\; 0.1111 \text{ at } \beta_1=0.80).$$
  4. *Pure harmonic rotation ($g_t = (\cos(\omega t), \sin(\omega t))^\top$, bilinear game):* $\hat{m}_t \to \frac{1-\beta_1}{1-\beta_1 e^{-i\omega}} e^{i\omega t}$, so
     $$C_t \xrightarrow{t \to \infty} \frac{1 - \beta_1}{\sqrt{1 - 2\beta_1 \cos\omega + \beta_1^2}} \in \left[\frac{1-\beta_1}{1+\beta_1},\, 1\right],$$
     which approaches $1$ for small game frequency $\omega = O(\eta)$ while introducing destabilizing phase lag $-\arctan\!\left(\frac{\beta_1 \sin\omega}{1 - \beta_1 \cos\omega}\right) \approx -\frac{\beta_1 \omega}{1-\beta_1}$.

---

### 2.4 Theorem 4 — Fundamental Information-Theoretic Obstructions

> **Theorem 4.1 (Step-1 Near-Equilibrium Indistinguishability / Inevitable Initial Kick).**
> Let $\mathcal{U}$ be any nontrivial trajectory degree-0 first-order optimizer using only gradient history $(g_1, \dots, g_t)$ with $\|U_1(e_1)\| = u_0 > 0$. Compare $L_A(x) = \frac{1}{2} x^2$ initialized near equilibrium at $x_0 = \delta > 0$ against $L_B(x) = \frac{\delta}{2} x^2$ initialized far from equilibrium at $x_0 = 1$. Both present the identical step-1 gradient $g_1 = \delta$. By trajectory 0-homogeneity, $U_1(\delta) = U_1(1)$, so on $L_A$ the first update moves $x_0 = \delta$ to $x_1 = \delta - \eta u_0$, amplifying error by $|\eta u_0 / \delta - 1| \to \infty$ as $\delta \to 0^+$.

> **Theorem 4.2 (Impossibility of Time-Varying Scale Invariance with Magnitude-Based Settling).**
> No causal gradient-only operator $\mathcal{U}$ can simultaneously satisfy **time-varying positive 0-homogeneity**,
> $$U_t(c_1 g_1, \dots, c_t g_t) = U_t(g_1, \dots, g_t) \qquad \forall c_1, \dots, c_t > 0,$$
> and continuous settling along monotonic vanishing gradients ($g_t = \lambda^t g_0 \to 0 \implies U_t(g_{1:t}) \to 0$).

**Proof.** Take any converging monotonic sequence $g_t = \lambda^t e_1$ ($\lambda \in (0, 1)$) and choose $c_s = \lambda^{-s} > 0$. Then $(c_1 g_1, \dots, c_t g_t) = (e_1, \dots, e_1)$ is a constant non-converging unit sequence, yet time-varying 0-homogeneity forces $U_t(g_{1:t}) = U_t(e_1, \dots, e_1)$. Thus either $U_t$ fails to settle on $g_t = \lambda^t e_1$ or it freezes on a constant non-zero gradient $e_1$ (a linear objective with no minimum). Equivalently, observing only gradient magnitudes $\|g_1\|, \dots, \|g_t\|$ cannot distinguish a genuine decrease in distance-to-optimum from a time-varying loss rescaling $L_t = c_t L$. $\square$

**What Breaks the Information-Theoretic Obstruction?** Three distinct information channels transcend gradient-magnitude history — and ParticleGAN quietly relies on two of them outside `peak_rms`:
1. **Directional Sign-Flip / Angular Cancellation Channel** (invariant even to *time-varying* $c_t > 0$ when evaluated on unit gradients $\hat{g}_s = g_s / \|g_s\|$, e.g., `UnifiedV12Optimizer._update_period2_flip_feedback` prime-67 CountSketch $\cos(\hat{g}_t, \hat{g}_{t-1}) < -0.45$): detects overshoot/oscillation without magnitude memory, though it cannot detect monotonic $g_t = \lambda^t e_1 \to 0$.
2. **Parameter–Gradient Secant / Curvature Channel** ($\Delta x_{t-1}, \Delta g_t$): For $L(x) = \frac{c}{2}(x-x^*)^2$, the Newton–secant distance estimate $-\frac{g_t \Delta x_{t-1}}{g_t - g_{t-1}} = -(x_{t-1} - x^*)$ is exact degree-0 invariant to static $c$, immune to spike ratcheting, and resolves Obstruction 4.1 on step 2.
3. **External Function-Space / Data-Space Residual Channel** (`_step_scale`, `_thermal_scale` from transport residual $r_{\text{OT}} / s_{\text{data}}$ in `UnifiedV11StepRecord` / `MinimalStepRecord`): directly measures distribution convergence in data coordinates, completely invariant to loss scaling $c$.

---

## 3. Controlled Five-Regime Dynamical Benchmarks

All results below are verbatim from [`scripts/four_laws_audit/p1_scale_settling.py`](../../scripts/four_laws_audit/p1_scale_settling.py) (protocol seed `0`, single-threaded deterministic FP32 CPU execution, verified by `7/7` passing tests in [`tests/four_laws_audit/test_p1_law42_scale_settling.py`](../../tests/four_laws_audit/test_p1_law42_scale_settling.py)).

### 3.1 Regime 1 — Global Objective Rescaling (`c ∈ [1e-12, 1e12]`) & Extreme Underflow Breakpoints

**2D $8 \times 8$ SPD Quadratic (40 steps, `lr = 0.02`) + Extreme Step-1 Underflow Probe + 1D Vector Probe:**

| Optimizer Arm | `diff(1e-12 vs 1e-6)` | `diff(1e-6 vs 1)` | `diff(1e6 vs 1)` | `max_diff` (24 dec) | Step-1 Move @ `1e-12` / `1e-25` / `1e-37` / `1e-39` | 1D Ratio `1e-12/1e-6` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `memless_norm_sgd` | `5.96e-08` | `5.96e-08` | `1.49e-08` | **`5.96e-08`** | `2.00e-2` / `2.00e-2` / `2.00e-2` / `2.00e-2` | — |
| `memless_sign_sgd` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **`0.00e+00`** | — | — |
| `fixed_eps_adam` (`eps=1e-8`) | `6.96e-01` | `9.16e-03` | `1.11e-05` | `6.98e-01` | **`1.40e-5`** / `0.00` / `0.00` / `0.00` | — |
| `fixed_eps_tikh` (`eps=0.0061`) | `3.62e-04` | `3.14e-01` | `2.38e-07` | `3.14e-01` | — | — |
| `pure_peak_mono` ($\gamma=1$) | `5.96e-08` | `5.96e-08` | `5.96e-08` | **`5.96e-08`** | `1.79e-2` / `1.79e-2` / `1.79e-2` / `0.00` | **`1.0000`** |
| `pure_peak_decay` ($\gamma=.9995$) | `5.96e-08` | `2.98e-08` | `1.19e-07` | **`1.19e-07`** | — | — |
| `pure_peak_window` ($W=50$) | `5.96e-08` | `5.96e-08` | `5.96e-08` | **`5.96e-08`** | — | — |
| `pure_coherence` ($\beta_1=.80$) | `5.96e-08` | `2.38e-07` | `2.38e-07` | **`2.38e-07`** | — | — |
| **`pg_unified_v12`** *(Default)* | **`1.19e-07`** | **`3.26e-01`** | `7.79e-03` | **`3.26e-01`** | `4.19e-2` / `4.19e-2` / `4.19e-2` / `0.00` | **`1.0000`** |
| `pg_unified_v12_stiff` (`s_base=0`) | `1.19e-07` | `1.19e-07` | `1.19e-07` | **`2.38e-07`** | — | — |
| **`pg_unified_v11`** | **`1.19e-07`** | **`2.38e-01`** | `7.79e-03` | **`2.38e-01`** | `4.65e-2` / **`5.66e-2`*** / `5.66e-2` / `0.00` | **`1.51e-04`** |
| **`pg_t4_equil`** | `2.98e-07` | `3.58e-07` | `2.38e-07` | **`5.96e-07`** | `8.37e-2` / `8.37e-2` / `8.37e-2` / `8.37e-2` | — |
| **`pg_minimal`** | **`1.76e-01`** | `6.76e-03` | `5.96e-08` | **`1.76e-01`** | **`3.17e-6`** / `0.00` / `0.00` / `0.00` | **`2.87e-03`** |
| **`pg_nash_v3`** | **`1.37e-01`** | `6.23e-03` | `5.96e-08` | **`1.38e-01`** | **`2.47e-6`** / `0.00` / `0.00` / `0.00` | — |

*(\*In `pg_unified_v11`, `peak_after_step1` drops from `2.47e-22` at $c=10^{-22}$ to exact `0.0` at $c=10^{-25}$ because raw `gf_safe.norm()` underflows FP32 squaring, disabling Tikhonov smoothing and jumping the step norm from `0.0465` to unsmoothed polar `0.0566`. In `pg_unified_v12`, `peak_after_step1` stays accurate down to `2.47e-37` at $c=10^{-37}$ and cuts off cleanly at the `1e-38` guard.)*

### 3.2 Regime 2 — Deterministic Convergence, $10^{-12}$ `tanh` Rail Escape, Near-Eq Kick & $1/t$ Decay

**2a (Quadratic, $\kappa=10$, 1000 steps), 2b (`tanh` rail from $W_0 = 14.0 I_4$, $\operatorname{sech}^2(14) = 2.755 \times 10^{-12}$, `lr=0.08`, 1000 steps across $c \in \{10^{-6}, 1, 10^6\}$), 2c (Near-Eq $W_0 = 10^{-6}$ Step-1 Kick):**

| Optimizer Arm | Quad `final_err` / `tail_stp` | Rail Escape Step (`1e-6` / `1` / `1e6`) | Rail Final RMS (`1e-6` / `1` / `1e6`) | Rail Tail Step (`1e-6` / `1` / `1e6`) | Near-Eq Step-1 Kick ($\|W_1\|/\|W_0\|$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `memless_norm_sgd` | `1.30e-2` / `2.00e-2` | `326 / 326 / 326` | `6.1e-6 / 6.1e-6 / 6.1e-6` | **`8.0e-2 / 8.0e-2 / 8.0e-2`** *(chatter)* | **`19,999×`** |
| `fixed_eps_adam` | `2.11e-5` / `3.28e-2` | **`-1 / -1`** / `44` | **`7.00 / 6.99`** / `1.6e-2` | `0.0 / 4.6e-5 / 1.8e-1` | `73,961×` |
| `fixed_eps_tikh` | `9.98e-3` / `2.00e-2` | **`-1 / -1 / -1`** | **`7.00 / 7.00 / 6.98`** | `0.0 / 0.0 / 7.8e-5` | `2.3×` |
| **`pure_peak_mono`** | `4.24e-1` / `7.19e-4` | **`364 / 364 / 364`** | **`0.00 / 0.00 / 0.00`** | **`0.00 / 0.00 / 0.00`** | **`17,888×`** |
| `pure_peak_decay` | `2.64e-1` / `7.46e-4` | `364 / 364 / 364` | `0.00 / 0.00 / 0.00` | `0.00 / 0.00 / 0.00` | `17,888×` |
| `pure_peak_window` | `8.94e-3` / `1.79e-2` | `364 / 364 / 364` | `8.9e-3 / 8.9e-3 / 8.9e-3` | **`7.2e-2 / 7.2e-2 / 7.2e-2`** *(no settle)* | `17,888×` |
| `pure_coherence` | `1.33e-3` / `2.03e-3` | `326 / 326 / 326` | `1.6e-4 / 1.1e-3 / 2.0e-5` | `6.0e-3 / 7.4e-3 / 6.6e-3` *(chatter)* | `19,999×` |
| **`pg_unified_v12`** | `1.91e-2` / `3.79e-2` | `182 / 180 / 170` | **`0.00`** / **`1.98e-2` / `1.50e-2`** | **`0.00`** / **`1.58e-1` / `1.60e-1`** | **`33,695×`** |
| `pg_unified_v12_stiff` | `6.25e-1` / `8.25e-4` | `650 / 650 / 650` | `5.4e-21 / 5.4e-21 / 5.4e-21` | `5.9e-19 / 5.9e-19 / 5.9e-19` | `9,999×` |
| **`pg_unified_v11`** | `1.91e-2` / `3.79e-2` | `170 / 169 / 166` | **`0.00`** / **`1.98e-2` / `1.55e-2`** | **`0.00`** / **`1.58e-1` / `1.60e-1`** | `37,203×` |
| `pg_t4_equil` | `9.28e-3` / `5.15e-3` | `69 / 69 / 69` | `1.5e-3 / 9.3e-4 / 3.9e-4` | `8.4e-3 / 8.7e-3 / 1.1e-2` | `28,283×` |
| **`pg_minimal`** | `6.31e-2` / `4.10e-4` | **`-1 / -1`** / `117` | **`7.00 / 7.00`** / `0.00` | `0.0 / 1.5e-5 / 0.0` | `23,253×` |
| **`pg_nash_v3`** | `1.93e-1` / `8.11e-4` | **`-1 / -1`** / `148` | **`7.00 / 7.00`** / `0.00` | `0.0 / 1.1e-5 / 0.0` | `18,021×` |

**2d — Algebraic $1/t$ Decay (`g_t = 1e-4 / t * W_dir`, Theorem 3b & 3d Verification):**
- `pure_peak_mono` ($\gamma=1.0$): `step_1 = 1.79e-2 → step_2000 = 2.00e-5 → step_5000 = 8.00e-6` (`r5k/1 = 0.0004` — settles monotonically).
- `pg_unified_v12` (`p_dec=1.0`): `step_1 = 3.37e-2 → step_2000 = 4.00e-5 → step_5000 = 1.60e-5` (`r5k/1 = 0.0005` — settles monotonically).
- **`pure_peak_decay` ($\gamma=0.9995$):** `step_1 = 1.79e-2 → step_2000 = 5.43e-5 → step_5000 = 9.75e-5` (**`r5k/2k = 1.7933` — non-monotonic anti-settling!**) and on the 25k-step probe **`step_25000 = 1.7883e-2` (`ratio_25k/1 = 0.9997` — complete reversion to uncooled step!)**.
- **`pg_unified_v11` (`p_dec=0.9995`):** `step_1 = 3.72e-2 → step_2000 = 1.81e-4 → step_5000 = 3.25e-4` (**`r5k/2k = 1.7934` — anti-settling!**).
- `pure_peak_window` ($W=50$): `r5k/1 = 0.9980`; `pure_coherence`: `r5k/1 = 0.9943`; `pg_t4_equil`: `r5k/1 = 3.5154` (zero settling!).

### 3.3 Regime 3 — Noisy Stationary Points (Zero-Mean Noise, 2500 steps, $8 \times 8$)

| Optimizer Arm | Gaussian `tail/early` | Sphere Const-Norm `tail/early` | `100×` Variance-Drop `tail/early` | Gaussian Drift $\|W_T\|_F$ |
| :--- | :---: | :---: | :---: | :---: |
| `memless_norm_sgd` | `1.0000` | `1.0000` | `1.0000` | `0.9618` |
| `pure_peak_mono` | `0.9730` *(slow $\ln^{-1/2} t$)* | **`1.0000`** *(zero settle)* | **`0.0190`** *(tracks drop)* | `0.8024` |
| `pure_peak_decay` | `1.0044` | `1.0000` | `0.0529` | `0.8193` |
| `pure_peak_window` | `1.0038` | `1.0000` | `1.0038` *(forgets in 50 steps)* | `0.8198` |
| `pure_coherence` | `0.9699` | `0.9708` | **`0.9699`** *(blind to var drop)* | `0.9571` |
| `pg_unified_v12` | `0.9711` | **`0.9990`** | **`0.0221`** | `1.9123` |
| `pg_unified_v12_stiff` | `0.9804` | `0.9761` | `0.0385` *($\beta_2=.999$ lag)* | `0.6107` |
| `pg_t4_equil` | `0.9699` | `0.9708` | `0.9699` | `1.6949` |

**Exact Coherence I.I.D. Floor Verification (Theorem 3d):**
- $\beta_1 = 0.75$: Theory $\sqrt{0.25/1.75} = \mathbf{0.3780}$ vs. Empirical **`0.3800`** (`err = 0.0020`).
- $\beta_1 = 0.80$: Theory $\sqrt{0.20/1.80} = \mathbf{0.3333}$ vs. Empirical **`0.3355`** (`err = 0.0022`).
- $\beta_1 = 0.90$: Theory $\sqrt{0.10/1.90} = \mathbf{0.2294}$ vs. Empirical **`0.2311`** (`err = 0.0017`).

### 3.4 Regime 4 — Adversarial Spike Freezing, Distribution Shift + Scale Drop, & Non-Convex Saddle

| Optimizer Arm | Single $10^4\times$ Spike Err (`c=1e-6` / `c=1`) | Post/Pre Step Ratio (`c=1e-6`) | Shift + $1000\times$ Scale-Drop Err | Non-Convex Saddle Escape Step (`(5, 1e-8)`, 1000 steps) |
| :--- | :---: | :---: | :---: | :---: |
| `memless_norm_sgd` | `0.0042` / `0.0042` | `1.000` | `0.0000` | `396` |
| `memless_sign_sgd` | `0.0112` / `0.0112` | `1.000` | `0.0000` | **`51`** |
| `pure_peak_mono` | **`3.300`** / **`3.300`** *(frozen!)* | **`7.57e-04`** | **`3.881`** *(frozen!)* | **`-1`** *(trapped!)* |
| `pure_peak_decay` | **`3.290`** / **`3.290`** *(stalled!)* | **`7.58e-04`** | **`3.787`** *(stalled!)* | **`-1`** *(trapped!)* |
| **`pure_peak_window`** ($W=50$) | **`0.0089`** / **`0.0089`** | `7.57e-04` *(50 steps)* | **`0.0000`** | `531` |
| `pure_coherence` ($\beta_2=0.97$) | `0.0006` / `0.0011` | `0.668` | `0.0000` | `609` |
| **`pg_unified_v12`** *(Default)* | **`3.024`** *(frozen!)* / **`0.0158`*** | **`7.77e-04`** | **`3.767`** *(frozen!)* | **`-1`** *(trapped!)* |
| `pg_unified_v12_stiff` ($\beta_2=.999$) | **`3.321`** / **`3.321`** *(stalled!)* | `0.871` | **`3.790`** *(stalled!)* | `196` |
| **`pg_unified_v11`** | **`2.921`** *(stalled!)* / **`0.0158`*** | **`1.13e-03`** | **`3.335`** *(stalled!)* | **`-1`** *(trapped!)* |
| `pg_t4_equil` ($\beta_2=0.97$) | `0.0007` / `0.0007` | `0.697` | `0.0000` | `24` |
| `pg_minimal` / `pg_nash_v3` | **`2.929 / 3.074`** *(both scales!)* | `2.1e-03` | **`3.454 / 3.617`** | `123 / 155` |

*(\*Catch-22 proof: at `c = 1.0`, `pg_unified_v12` and `pg_unified_v11` recover from the $10^4\times$ spike ONLY because `s_base = 0.0061` caps `s_eff` and breaks scale invariance! Whenever a layer operates below `s_base / alpha` — the exact regime `peak_rms` was built for — a single spike or scale-dropping distribution shift freezes `UnifiedV12Optimizer` irreversibly!)*

### 3.5 Regime 5 — Persistent Adversarial Rotation ($\min_x \max_y x^\top y$, 600 steps, `lr = 0.02`)

| Optimizer Arm | Simultaneous $r_T / r_0$ | Simultaneous Tail Step | Alternating $r_T / r_0$ | Alternating Tail Step |
| :--- | :---: | :---: | :---: | :---: |
| `memless_norm_sgd` | `1.1138` *(exact $\sqrt{1+0.24}=1.1136$)* | `2.00e-02` | `1.0062` | `2.00e-02` |
| `pure_peak_mono` | **`1.0862`** *($\|g_t\|/P_t \equiv 1$!)* | `1.78e-02` | `0.9991` | `1.74e-02` |
| `pure_coherence` ($\beta_1=0.80$) | **`1.7600`** *(momentum divergence!)* | `2.00e-02` | **`1.6913`** | `2.00e-02` |
| `pg_unified_v12` | **`1.5332`** | `3.98e-02` | `1.1505` | `3.98e-02` |
| `pg_unified_v12_stiff` | `1.2141` | `9.92e-03` | `1.1856` | `9.96e-03` |
| `pg_t4_equil` ($\beta_1=0.80$) | **`3.8282`** *(severe spiral blowup!)* | `3.34e-02` | **`3.1392`** | `3.40e-02` |

**Exact Harmonic Rotation Coherence Formula Verification (Theorem 3d, $\beta_1=0.80, \beta_2=0.97$):**
- $\omega = 0.02$ (slow game): Theory `0.99602`, Empirical `0.99602` (`err = 5.67e-7`).
- $\omega = 0.10$: Theory `0.91293`, Empirical `0.91293` (`err = 2.73e-7`).
- $\omega = \pi/2$ (quadrature): Theory `0.15617`, Empirical `0.15617` (`err = 9.69e-8`).
- $\omega = \pi$ (exact period-2 flip): Theory `1/9 = 0.11111`, Empirical `0.11111` (`err = 3.89e-8`).

---

## 4. Structured 10-Category Taxonomy & 8-Point Claim Audit

Following [`reports/four-laws-audit/README.md`](README.md#L25-L38), every substantial claim in Law 4.2 and its implementations is classified and analyzed across all eight required dimensions.

### Summary Classification Table

| ID | Audited Claim | Taxonomy Classification |
| :---: | :--- | :--- |
| **C1** | Memoryless degree-0 scale invariance ($U(cg)=U(g)$) + continuous settling at $0$ ($\Delta W \to 0$ as $G \to 0$) ([`docs/four-laws.md:86`](../../docs/four-laws.md#L86)) | **Falsified claim** *(disproved by Theorem 1 — proven mathematical impossibility)* |
| **C2** | Fixed additive $\varepsilon > 0$ cannot simultaneously escape $10^{-12}$ rails and settle above $10^{-12}$ ([`docs/four-laws.md:86`](../../docs/four-laws.md#L86)) | **Proven mathematical result** *(Theorem 2)* |
| **C3** | Stateful causal trajectory degree-0 invariance ($U_t(cg_{1:t}) = U_t(g_{1:t})$) IS compatible with continuous settling along relative-decay trajectories | **Proven mathematical result** *(Theorem 3)* |
| **C4** | First-order gradient-only scale-invariant optimizers cannot avoid a step-1 near-equilibrium kick or distinguish time-varying scale drops from convergence | **Information-theoretic obstruction** *(Theorems 4.1 & 4.2)* |
| **C5** | ParticleGAN's $s_{\text{eff}} = \min(s_{\text{base}}, c \cdot \text{peak\_rms})$ formula is degree-0 scale-invariant ([`law4-optimizer-laws.md:208`](../forge/studies/law4-optimizer-laws.md#L208), [`docs/four-laws.md:115,160`](../../docs/four-laws.md#L115)) | **Falsified claim** *(breaks above $s_{\text{base}}/c$: `0.0122` in `v12`, `0.0203` in `v11`)* |
| **C6** | "Theorem 4A.1": `T4Nash` / `T4NashV3` / `Minimal` `peak_rms` update $m / (\sqrt{v} + 0.015\,\text{peak\_rms} + \varepsilon)$ is exact degree-0 homogeneous ([`t4-imbalance-nash.md:38-40`](../forge/studies/t4-imbalance-nash.md#L38-L40), [`r4-minimal-core.md:33,61`](../forge/studies/r4-minimal-core.md#L33)) | **Falsified claim** *(unscaled `+ eps = 1e-8` crushes $10^{-12}$ gradients by $10^{-4}\times$)* |
| **C7** | Momentum coherence $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock ([`docs/four-laws.md:160`](../../docs/four-laws.md#L160), [`t4-strange-architectures.md:65`](../forge/studies/t4-strange-architectures.md#L65)) | **Falsified claim** *(strictly positive floors $\sqrt{\frac{1-\beta_1}{1+\beta_1}}$ and $\frac{1-\beta_1}{1+\beta_1}$; inactive in default `v11`/`v12`)* |
| **C8** | $10^{-38}$ dynamic scale-normalized layerwise conditioning is required to equalize deep inter-layer skew ($10^4\times$) and $\operatorname{sech}^2$ saturation rails against single-LR SGD ([`docs/four-laws.md:86`](../../docs/four-laws.md#L86), [`law4-optimizer-laws.md:206-210`](../forge/studies/law4-optimizer-laws.md#L206-L210)) | **Empirically supported causal mechanism** *(within finite dynamic range; **Necessary condition under stated assumptions** for single-LR gradient descent across $10^{12}\times$ attenuation)* |
| **C9** | Monotonic `peak_rms` (`p_dec = 1.0`) vs. decaying `peak_rms` (`p_dec = 0.9995`) resolves equilibrium settling across non-stationary / spiky training without a step clock | **Engineering heuristic** *(fragile trade-off: `p_dec=1` freezes permanently on spikes/shifts; `p_dec<1` anti-settles on $t^{-p}$ decay)* |
| **C10** | Can a purely first-order, clock-free, $O(\text{params})$ stateful rule achieve global trajectory degree-0 invariance, settling on all $g_t \to 0$, *and* spike/shift immunity without external function-space residuals or secant curvature? | **Unresolved question** *(tightly constrained by Theorem 4.2: requires a non-magnitude channel such as directional flips, $(\Delta x, \Delta g)$ secants, or data-space residuals)* |

---

### 8-Point Detailed Analyses

#### Claim C1 — Memoryless Degree-0 Homogeneity + Continuous Settling at Zero (`Falsified claim`)
1. **Formulation:** [`docs/four-laws.md:86`](../../docs/four-laws.md#L86) asserts the requirement for *"Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish ($\Delta W \to 0$ as $G \to 0$)"* without specifying state/history dependence.
2. **Assumptions:** Update is a memoryless map $U: \mathbb{R}^d \to \mathbb{R}^d$ satisfying $U(cg) = U(g)$ ($\forall c > 0, g \neq 0$) and $\lim_{g \to 0} U(g) = 0$.
3. **Supporting evidence:** None — the statement is internally contradictory as a memoryless property; the empirical wins cited in [`law4-optimizer-laws.md`](../forge/studies/law4-optimizer-laws.md) actually come from *stateful* optimizers plus external data-space residual damping (`_step_scale`, `_thermal_scale`).
4. **Counterexamples:** Any nonzero vector $g \neq 0$ with $g_n = g / n \to 0$ has $U(g_n) = U(g) \not\to 0$ (Theorem 1). On $L(x) = \frac{1}{2}x^2$ with $x_0 = 0.013, \eta = 0.02$, normalized SGD oscillates permanently between $+0.013$ and $-0.007$ (`tail_step = 0.0200`).
5. **Minimal falsification experiment:** `test_theorem1_memoryless_discontinuity_and_limit_cycle` in [`test_p1_law42_scale_settling.py`](../../tests/four_laws_audit/test_p1_law42_scale_settling.py#L48-L77).
6. **Prior research:** Classic normalized gradient descent (Nesterov 1984; Hazan et al. 2015; Murray et al. 2019) and SignSGD (Bernstein et al. 2018) explicitly require a decaying step-size schedule $\eta_t \to 0$ precisely because memoryless degree-0 maps are discontinuous at zero.
7. **Computational verification:** Verified in Regime 1 & Regime 2 (`memless_norm_sgd` and `memless_sign_sgd`: `max_diff <= 5.96e-8` over 24 decades, but `rail_tail_step = 8.0e-2` and `4.0e-2` forever).
8. **Confidence & uncertainty:** **100% (deductive mathematical proof).**

#### Claim C2 — Fixed-$\varepsilon$ Crossover Impossibility (`Proven mathematical result`)
1. **Formulation:** No fixed additive floor $\varepsilon > 0$ in $g / (\|g\| + \varepsilon)$ or $g / \sqrt{\|g\|^2 + \varepsilon^2}$ can simultaneously give scale invariance down to $10^{-12}$ and settling above $10^{-12}$ (Theorem 2).
2. **Assumptions:** Strictly increasing radial gain $\phi(\|g\|/\varepsilon)$ with fixed scalar $\varepsilon > 0$.
3. **Supporting evidence:** Exact analytic inversion of $\phi(u)$ (Section 2.2); empirical `fixed_eps_adam` (`eps=1e-8`) and `fixed_eps_tikh` (`eps=0.0061`) completely deadlock on the `sech^2(14) = 2.76e-12` rail (`escaped_step = -1`, `final_rms = 6.99 / 7.00`).
4. **Counterexamples:** None under fixed $\varepsilon > 0$ (changing $\varepsilon$ dynamically with history enters Theorem 3's stateful class).
5. **Minimal falsification experiment:** `test_theorem2_fixed_eps_crossover_impossibility` in [`test_p1_law42_scale_settling.py`](../../tests/four_laws_audit/test_p1_law42_scale_settling.py#L84-L99).
6. **Prior research:** Kingma & Ba (2015, Adam $\varepsilon$ sensitivity); Yuan & Gao (2020, EAdam); Liu et al. (2020, RAdam) document fixed-$\varepsilon$ crossover distorting scale invariance when $|g| \lesssim \varepsilon$.
7. **Computational verification:** Verified analytically and empirically in Regimes 1 & 2b (`fixed_eps_adam` step-1 move collapses to `1.40e-5` at `1e-12` and `0.00` at `1e-18`).
8. **Confidence & uncertainty:** **100% (deductive mathematical proof).**

#### Claim C3 — Stateful Trajectory Degree-0 Compatibility (`Proven mathematical result`)
1. **Formulation:** Causal trajectory operators $U_t(g_{1:t}) = F_t / \sqrt{\|F_t\|^2 + \alpha^2 S_t^2}$ with positive 1-homogeneous $(F_t, S_t)$ satisfy exact static global rescaling invariance $U_t(c g_{1:t}) = U_t(g_{1:t})$ AND settle continuously whenever $\|F_t\|/S_t \to 0$ (Theorem 3).
2. **Assumptions:** Static global positive rescaling $c > 0$; exact arithmetic (or factored mantissa–exponent storage above FP32 normal floor $10^{-38}$).
3. **Supporting evidence:** Exact algebraic proof (Section 2.3); empirical `pure_peak_mono` achieves `max_diff = 5.96e-8` across 24 orders of magnitude (`1e-12` to `1e12`), escapes the $2.76 \times 10^{-12}$ `tanh` rail at step `364` identically across $c \in \{10^{-6}, 1, 10^6\}$, and settles to `final_rms = 0.00e+00`, `tail_step = 0.00e+00`.
4. **Counterexamples (Boundary of Validity):** Fails under *time-varying* rescalings $c_t$, single large spikes (Theorem 3a), sub-exponential decay for $\gamma < 1$ (Theorem 3b), constant-norm stationary noise (Regime 3), and bilinear rotation (Regime 5).
5. **Minimal falsification experiment:** `test_regime1_exact_rescaling_and_actual_particlegan_breakpoints` and `test_regime2_rail_escape_settling_and_obstructions` (`pure_peak_mono`).
6. **Prior research:** Scale-free online learning (Orabona & Pál 2018, *Scale-Free Online Learning*, TCS; Cutkosky & Orabona 2018) and parameter-free/scale-invariant adaptive methods (AdaGrad, Duchi et al. 2011; DoG, Ivgi et al. 2023; D-Adaptation, Defazio & Mishchenko 2023) similarly achieve trajectory-level scale invariance via running gradient norms or maximum gradient norms $\max_{s \le t} \|g_s\|$.
7. **Computational verification:** Verified across Regimes 1 and 2 (`pure_peak_mono` matches to FP32 machine epsilon `5.96e-8` across `[1e-12, 1e12]` and works down to `1e-37`).
8. **Confidence & uncertainty:** **100% (proven theorem; practical boundaries characterized by Theorems 3a–3d and 4).**

#### Claim C4 — Information-Theoretic Obstructions (`Information-theoretic obstruction`)
1. **Formulation:** Every nontrivial first-order gradient-only trajectory-degree-0 optimizer suffers an unavoidable $O(\eta / \delta)$ step-1 kick when initialized at distance $\delta$ from equilibrium (Theorem 4.1), and cannot simultaneously be invariant to time-varying rescalings $c_t > 0$ and settle on monotonic $g_t \to 0$ (Theorem 4.2).
2. **Assumptions:** Causal first-order oracle returning only $(g_1, \dots, g_t)$ with no prior knowledge of objective scale $c$ or function-space residual.
3. **Supporting evidence:** Exact indistinguishability coupling proofs (Section 2.4); in Regime 2c (`W_0 = 1e-6`), **every** trajectory-degree-0 arm amplifies initial error by `9,999×` to `37,203×` on step 1! In Regime 4b (target shift + $1000\times$ scale drop), `pure_peak_mono` and `pg_unified_v12` stall at `err = 3.88 / 3.77`.
4. **Counterexamples (Escaping the Obstruction):** Requires a strictly richer information channel: (a) directional flips $\cos(\hat{g}_t, \hat{g}_{t-1})$, (b) $(\Delta x_{t-1}, \Delta g_t)$ secant curvature, or (c) external data-space residuals (`_step_scale`, `_thermal_scale`).
5. **Minimal falsification experiment:** `test_regime2_rail_escape_settling_and_obstructions` (part d, `near_equilibrium_kick`) and `test_regime4_spike_freezing_shift_and_saddle` (part b, `shift_with_1000x_scale_drop`).
6. **Prior research:** Lower bounds for scale-free/parameter-free online convex optimization (Orabona & Pál 2018; Cutkosky & Boahen 2017; Carmon & Hinder 2022) prove unavoidable initial-guess/Lipschitz-scale penalties when scale is unknown a priori.
7. **Computational verification:** Verified on all 14 arms in Regimes 2c and 4b.
8. **Confidence & uncertainty:** **100% (information-theoretic coupling proof).**

#### Claim C5 — `UnifiedV11` / `UnifiedV12` Capped `min(s_base, c * peak_rms)` Is Degree-0 Scale-Invariant (`Falsified claim`)
1. **Formulation:** [`law4-optimizer-laws.md:208`](../forge/studies/law4-optimizer-laws.md#L208) and [`docs/four-laws.md:115,160`](../../docs/four-laws.md#L115) claim $s_{\text{eff}} = \min(s_{\text{base}}, c\,\text{peak\_rms})$ is degree-0 scale-invariant.
2. **Assumptions:** Evaluated directly on the actual default constructors `UnifiedV12Optimizer(..., smoothing=0.0061)` and `UnifiedV11EquilibratedOptimizer(..., smoothing=0.0061)` from `get_recipe("unified_v12")` and `get_recipe("unified_v11")`.
3. **Supporting evidence:** Holds *only* on the restricted subdomain `peak_rms <= s_base / c` (`<= 0.0122` in `v12`, `<= 0.0203` in `v11`), where `diff(1e-12 vs 1e-6) = 1.19e-7`.
4. **Counterexamples:** At normal $O(1)$ gradients (`c = 1.0`), `peak_rms ≈ 2.47 >> 0.0122`, so `s_eff` saturates at fixed `0.0061`. Trajectory divergence between $c=10^{-6}$ and $c=1$ is **`0.3257`** (`v12`) and **`0.2378`** (`v11`). On Regime 2b (`tanh` rail), `pg_unified_v12` settles to `0.00` at $c=10^{-6}$ but chatters forever (`tail_step = 0.158 / 0.160`) at $c=1$ and $c=10^6$!
5. **Minimal falsification experiment:** `test_regime1_exact_rescaling_and_actual_particlegan_breakpoints` and `test_regime2_rail_escape_settling_and_obstructions` (part b).
6. **Prior research:** Same mechanism as fixed-$\varepsilon$ Tikhonov polar smoothing (Bernstein & Newhouse 2024; ParticleGAN `DualNorm`) above the crossover threshold $s_{\text{base}}/c$.
7. **Computational verification:** Verified directly against `particlegan._unified_v12.UnifiedV12Optimizer` and `particlegan._unified_v11.UnifiedV11EquilibratedOptimizer`. Also exposed Defects D, E, and F (FP32 `gf.norm()` underflow at $10^{-25}$ in `v11`, 1D `clamp_min(1e-8)` in `v11`, and unused prior `peak_rms` in both).
8. **Confidence & uncertainty:** **100% (exact algebra + direct code execution).**

#### Claim C6 — "Theorem 4A.1": `T4Nash` / `T4NashV3` / `Minimal` `peak_rms` Is Exact Degree-0 Homogeneous (`Falsified claim`)
1. **Formulation:** [`t4-imbalance-nash.md:38-40`](../forge/studies/t4-imbalance-nash.md#L38-L40) labels $u_t = \hat{m}_t / (\sqrt{\hat{v}_t} + 0.015 \sigma_t^{\text{peak}} + \varepsilon)$ "Theorem 4A.1" and claims it is *"exact degree-0 homogeneous under arbitrary constant gradient rescaling $g_t \mapsto c\cdot g_t$ ($c > 0$)"*. [`r4-minimal-core.md:33,61`](../forge/studies/r4-minimal-core.md#L33) claims `Minimal` falls back seamlessly to $10^{-38}$ dynamic scale normalization.
2. **Assumptions:** Evaluated directly on `MinimalGeneratorOptimizer` and `NashV3GeneratorAdam` with default `eps = 1e-8`.
3. **Supporting evidence:** Approximately homogeneous *only* for $c \cdot \text{rms}(g) \gg 10^{-8}/0.015 \approx 6.7 \times 10^{-7}$ (`diff(1e6 vs 1) = 5.96e-8`).
4. **Counterexamples:** At $c = 10^{-12}$, `pg_minimal` and `pg_nash_v3` lose trajectory invariance (`diff = 0.176 / 0.138`), their step-1 move collapses $10^4\times$ to `3.17e-6 / 2.47e-6`, and both **completely deadlock** on the $2.76 \times 10^{-12}$ `tanh` rail (`escaped_step = -1`, `final_rms = 7.00`). Even `Minimal`'s slow path ([`_minimal/dynamics.py:186`](../../particlegan/_minimal/dynamics.py#L186)) adds unscaled `+ eps` inside `(0.015 * peak + eps) / sc`, negating its own $10^{-38}$ scaling!
5. **Minimal falsification experiment:** `test_regime1_exact_rescaling_and_actual_particlegan_breakpoints` and `test_regime2_rail_escape_settling_and_obstructions` (part c).
6. **Prior research:** Standard Adam $\varepsilon$-floor breakdown (Kingma & Ba 2015).
7. **Computational verification:** Directly tested on `MinimalGeneratorOptimizer` and `NashV3GeneratorAdam` in Regimes 1 and 2b.
8. **Confidence & uncertainty:** **100% (direct code inspection + deterministic reproduction).**

#### Claim C7 — Momentum Coherence $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ Eliminates Equilibrium Chatter (`Falsified claim`)
1. **Formulation:** [`docs/four-laws.md:160`](../../docs/four-laws.md#L160) and [`t4-strange-architectures.md:65`](../forge/studies/t4-strange-architectures.md#L65) claim momentum-coherence cooling drives $\text{coherence} \to 0$ on oscillatory/noisy stationary equilibria without a step clock.
2. **Assumptions:** Fixed exponential moving average hyperparameters $\beta_1 \in (0, 1)$, $\beta_2 \in (0, 1)$ (specifically $(0.80, 0.97)$ in `T4EquilibratedOptimizer`, $(0.75, 0.999)$ in `UnifiedV11`/`V12` stiff mode, and $(0.0, 0.999)$ in default `UnifiedV11`/`V12`).
3. **Supporting evidence:** Coherence *does* attenuate oscillatory/noisy updates by a constant factor ($\sim 3\times$ on i.i.d. noise, $\sim 9\times$ on period-2 flips) and preserves true 24-decade scale invariance in `T4EquilibratedOptimizer` (`max_diff = 5.96e-7`), and when $\beta_2 = 0.999$ (`pg_unified_v12_stiff`) the slow $\beta_2$ memory lag acts as a decaying peak on rapidly converging trajectories (`rail_rms = 5.38e-21`).
4. **Counterexamples:** Theorem 3d proves strictly positive asymptotic floors: `0.3333` on i.i.d. noise ($\beta_1=0.80$), `0.1111` on period-2 sign flips, `~0.996` on slow bilinear rotation ($\omega = 0.02$, where momentum phase lag causes $3.83\times$ outward spiral blowup in `pg_t4_equil`!), and `1.000` on slow $1/t$ monotonic convergence. Furthermore, in default `UnifiedV11`/`V12`, `coh == 1.0` identically whenever `stiff_gate == 0`.
5. **Minimal falsification experiment:** `test_regime3_noisy_stationary_floors` and `test_regime5_bilinear_rotation_and_coherence_formula`.
6. **Prior research:** EMA SNR / gradient-alignment adaptive step sizing (Balles & Hennig 2018; Zhang et al. 2020; Agarwal et al. 2019) and negative-momentum / bilinear minimax instability (Gidel et al. 2019, ICLR; Mescheder et al. 2018, ICML).
7. **Computational verification:** Theoretical formulas match empirical simulations to `0.002` (i.i.d. noise) and `< 6e-7` (harmonic rotation across 4 frequencies).
8. **Confidence & uncertainty:** **100% (exact closed-form derivation + empirical verification to 6 significant figures).**

#### Claim C8 — Layerwise Scale-Normalized Conditioning for Deep Skew & Rail Escape (`Empirically supported causal mechanism` / `Necessary condition under stated assumptions`)
1. **Formulation:** Under a single player-level learning rate $\eta$, layerwise dynamic-$10^{-38}$ scale normalization is causally required to equalize $10^4\times$ inter-layer gradient skew and escape deep $\operatorname{sech}^2$ saturation rails without underflow dead zones.
2. **Assumptions:** Single global player learning rate $\eta$; unnormalized deep networks or saturated bounded activations ($\operatorname{sech}^2 \ll 10^{-8}$); finite training horizon $T$.
3. **Supporting evidence:** Causal SGD ladder in [`law4-optimizer-laws.md:68-98`](../forge/studies/law4-optimizer-laws.md#L68-L98) (`R2` `law4-sgd-secant` freezes at $a = 8.000$ and scores `4/13`, whereas adding zero-state layerwise Frobenius-MP in `R5` `law4-frob-sgd-mp` escapes immediately and jumps to `9/16`); our Regime 2b confirms `pure_peak_mono`, `pg_unified_v12`, and `pg_t4_equil` escape `sech^2(14) = 2.76e-12` in `69–364` steps while `fixed_eps_adam` and `fixed_eps_tikh` deadlock indefinitely.
4. **Counterexamples (Scope Boundaries):** Does not rescue rank-1 intra-layer spectral collapse (`grid100_mlp16` still fails without Law 4.3 spectral orthogonalization) and does not rescue mixed-saturation batches where a single unsaturated sample dominates shared layer weights ([`t4-strange-architectures.md:29`](../forge/studies/t4-strange-architectures.md#L29)).
5. **Minimal falsification experiment:** Regime 2b (`W_0 = 14.0 I_4` `tanh` rail escape) and `law4-sgd-secant` vs `law4-frob-sgd-mp` (`readout-d5c8d860` vs `readout-12f988a4`).
6. **Prior research:** Layerwise normalized updates: LARS (You et al. 2017), LAMB (You et al. 2020), Modular Duality / Spectral Descent (Bernstein & Newhouse 2024), Muon (Jordan et al. 2024), and RMS-to-RMS $\mu$P scaling (Yang et al. 2022).
7. **Computational verification:** Verified in Regime 2b and Forge causal ladder readouts (note: Forge calibration remains `BLOCKED` and seed `0` screening only).
8. **Confidence & uncertainty:** **High (95%)** for the causal mechanism under single-LR training; **Moderate** regarding exact optimality of $10^{-38}$ vs. alternative dynamic-scaling schemes in low precision (FP16/BF16/FP8/FP4).

#### Claim C9 — `peak_rms` Decay Policy (`p_dec = 1.0` vs. `0.9995`) Resolves Clock-Free Settling Across Non-Stationary Regimes (`Engineering heuristic`)
1. **Formulation:** Maintaining a scalar running peak `peak_rms` with `p_dec = 1.0` (`unified_v12`) or `p_dec = 0.9995` (`unified_v11`, `minimal`, `t4_nash_v3`) provides robust clock-free equilibrium settling.
2. **Assumptions:** Smoothly decaying gradients without large transient spikes or post-settling downward scale shifts; for `p_dec = 0.9995`, convergence rate faster than $0.9995^t$.
3. **Supporting evidence:** Passes Forge's screening tasks (`gaussian1d_stability`, `target_shift_recovery`, `zero_tanh_rail_escape`) in concert with `UnifiedV11StepRecord`'s external `_step_scale` / `_thermal_scale` and the `s_base = 0.0061` cap (which masks spike ratcheting at $O(1)$ gradient scales!).
4. **Counterexamples:** Regime 4a proves `p_dec = 1.0` (`pure_peak_mono` and `pg_unified_v12` at $c=10^{-6}$) **freezes permanently** after a single $10^4\times$ spike (`final_err = 3.30 / 3.02`) and after a $1000\times$ scale-dropping target shift (`shift_err = 3.88 / 3.77`), and stalls on the `(5, 1e-8)` non-convex saddle (`escape_step = -1`). Conversely, Regime 2d proves `p_dec = 0.9995` (`pure_peak_decay` and `pg_unified_v11`) **reverses settling** on $1/t$ decay (`step_5000 / step_2000 = 1.793`, reverting to `ratio = 0.9997` by step 25,000) while still stalling for $>18{,}000$ steps after a $10^4\times$ spike!
5. **Minimal falsification experiment:** `test_regime4_spike_freezing_shift_and_saddle` and `test_regime2_rail_escape_settling_and_obstructions` (part e).
6. **Prior research:** AMSGrad (Reddi et al. 2018) exhibits the exact same monotonic-max ratcheting problem, motivating windowed/decaying second-moment variants (AdaBound, Luo et al. 2019; Adan; AdaBelief, Zhuang et al. 2020).
7. **Computational verification:** Verified across Regimes 2d, 3, 4a, 4b, 4c, and 5.
8. **Confidence & uncertainty:** **High (95%)** — the trade-off between `p_dec = 1`, `p_dec < 1`, and finite window $W$ is mathematically structural, not a tuning artifact.

#### Claim C10 — Clock-Free First-Order Scale-Invariant Settling with Spike/Shift Immunity (`Unresolved question`)
1. **Formulation:** What minimal stateful mechanism achieves *simultaneous* (i) global trajectory degree-0 invariance across $[10^{-35}, 10^{35}]$, (ii) smooth settling on all deterministic and stochastic equilibria (including $t^{-p}$ decay and zero-mean noise), (iii) rapid recovery from transient spikes and downward scale shifts, and (iv) zero destabilization of adversarial minimax rotations?
2. **Assumptions:** Clock-free execution (`clockfree_audit_measurement_v1`), $O(\text{params})$ state/compute.
3. **Supporting evidence:** Partial pieces already exist inside ParticleGAN: prime-67 CountSketch flip-cosine (`UnifiedV12Optimizer._update_period2_flip_feedback`), realized secant probes (`delta_g_max`), and transport-residual gating (`_step_scale`).
4. **Counterexamples:** Theorem 4.2 proves no purely magnitude-based gradient-only rule can satisfy (i)–(iii) jointly.
5. **Minimal falsification experiment:** Any candidate must pass all five regimes of `p1_scale_settling.py` simultaneously (24-decade rescaling, $10^{-12}$ rail + $1/t$ settling, noisy stationary settling, $10^4\times$ spike + $1000\times$ scale-drop shift recovery, and non-divergent bilinear rotation).
6. **Prior research:** Online learning with non-stationary regret / discounted adaptive methods (Cutkosky & Orabona 2018; Jacobsen & Cutkosky 2022), Polyak/secant stepsizes (Loizou et al. 2021), and variance-reduced/extragradient saddle solvers (Korpelevich 1976; Mishchenko et al. 2020).
7. **Computational verification:** Open — none of the 14 tested arms clears all five regimes simultaneously.
8. **Confidence & uncertainty:** **High confidence in the obstruction; open research target for a unified synthesis.**

---

## 5. Rigorous Reformulation of Law 4.2 & Proposed Edits

### 5.1 Mathematically Rigorous Statement of Law 4.2

> **Law 4.2 (Revised) — Stateful Trajectory Scale-Invariance, Rail Escape & Settling Trade-Offs:**
> Multiplicative inter-layer Jacobian attenuation across deep networks and $\operatorname{sech}^2$-like saturation rails span many orders of magnitude ($10^{-12}$ to $10^{+2}$), defeating any single player-level learning rate or fixed additive floor $\varepsilon > 0$ (which cannot simultaneously preserve scale invariance at $g_{\text{rail}}$ and settle at $g_{\text{eq}} \ge g_{\text{rail}}$).
>
> By the **Memoryless Impossibility Theorem**, no nontrivial *memoryless* positive 0-homogeneous update $U(cg) = U(g)$ can be continuous at $0$ ($\Delta W \to 0$ as $G \to 0$); therefore, **temporal state is mathematically necessary** to reconcile scale invariance with smooth equilibrium settling without a step-count clock. Specifically:
> 1. **Trajectory Degree-0 Invariance:** Causal stateful updates $U_t(g_1, \dots, g_t)$ with factored underflow-safe ($10^{-38}$) mantissa–exponent state can satisfy **static trajectory 0-homogeneity** $U_t(c g_1, \dots, c g_t) = U_t(g_1, \dots, g_t)$ ($\forall c > 0$) while contracting continuously whenever a positive 1-homogeneous signal-to-reference ratio $\|F_t(g_{1:t})\| / S_t(g_{1:t}) \to 0$.
> 2. **Structural Trade-Offs & Information-Theoretic Limits:** Using gradient history alone, every nontrivial trajectory-degree-0 operator incurs a step-1 near-equilibrium kick (cannot distinguish small distance from small objective scale at $t=1$) and cannot remain invariant to *time-varying* rescalings $c_t$ while settling on monotonic $g_t \to 0$. Moreover, reference-scale choices trade off memory against adaptability: monotonic peaks ($\gamma=1$) settle on all $g_t \to 0$ but freeze irreversibly after outlier spikes or downward scale shifts; decaying peaks ($\gamma < 1$) revert to uncooled chatter on sub-exponential convergence ($t^{-p}$); fixed-$\beta_1$ momentum coherence ($\|\hat{m}_t\|/\|\hat{v}_t\|$) retains strictly positive noise/flip floors ($\sqrt{(1-\beta_1)/(1+\beta_1)}$, $(1-\beta_1)/(1+\beta_1)$) and destabilizes pure adversarial rotations.
> 3. **Required Separation of Channels:** Genuine equilibrium settling under noise, non-stationary shifts, spikes, and two-player rotation requires combining trajectory-scale-invariant conditioning with a **non-magnitude equilibrium signal** — directional sign-reversal statistics (on unit gradients), parameter–gradient secants $(\Delta\theta, \Delta g)$, or function-space/data-space transport residuals (coupled with Law 2's rotational damping).

### 5.2 Proposed Exact Edits to [`docs/four-laws.md`](../../docs/four-laws.md) (Documented Only — File Untouched per Ownership)

1. **Replace [`docs/four-laws.md`](../../docs/four-laws.md#L86) (Law 4.2 bullet):**
   - *Current:* `Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish (\Delta W -> 0 as G -> 0) is required without a step clock.`
   - *Proposed:* `Exact memoryless degree-0 invariance (U(cG) = U(G)) is mathematically incompatible with continuous settling at zero (every nontrivial memoryless 0-homogeneous map is discontinuous at 0). Reconciling rail escape with clock-free equilibrium settling therefore requires **stateful trajectory-level degree-0 conditioning** (U_t(cG_1,...,cG_t) = U_t(G_1,...,G_t)) with underflow-safe (10^-38) dynamic scaling, paired with a non-magnitude settling signal (directional sign-reversal detection, secant curvature, or data-space transport residual) to avoid peak-ratchet freezing under spikes/scale-drops and rotational instability.`
2. **Replace [`docs/four-laws.md`](../../docs/four-laws.md#L160) (Part II Solutions to Law 4):**
   - *Current:* claims *"degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock."*
   - *Proposed:* Explicitly disclose that (a) `unified_v11`/`unified_v12` cap `s_eff = min(s_base, c * peak_rms)` at `s_base = 0.0061`, restricting trajectory 0-homogeneity to `peak_rms <= s_base / c` (`0.0122` in `v12`, `0.0203` in `v11`), (b) `unified_v12` uses monotonic `p_dec = 1.0` on 1-objective tasks (vulnerable to spike ratcheting below `s_base / 0.50`), (c) momentum coherence is active only when `stiff_gate > 0` (or scalar 1D) and converges to positive floors ($\sqrt{(1-\beta_1)/(1+\beta_1)}$ on i.i.d. noise, $(1-\beta_1)/(1+\beta_1)$ on flips) rather than $0$, and (d) actual equilibrium settling in ParticleGAN relies jointly on `peak_rms`/Tikhonov damping *and* data-space residual cooling (`_step_scale`, `_thermal_scale`).
3. **Update [`docs/four-laws.md`](../../docs/four-laws.md#L115) & [`L187`](../../docs/four-laws.md#L187):** Replace unqualified `"homogeneous peak_rms settling"` with `"capped peak_rms Tikhonov settling (trajectory-homogeneous below s_base/c)"`, and clarify that `zero_tanh_rail_escape` tests $\operatorname{sech}^2 \approx 0.19$ (whereas deep $10^{-12}$ rails are handled by `unified_v12`/`t4_equil` on 2D+ layers, and currently fail on `minimal`/`t4_nash_v3` due to additive `eps = 1e-8`).

### 5.3 Concrete Code Recommendations (Out of P1 File Ownership — For Follow-Up Engineering PRs)

1. **`_minimal/dynamics.py:172,186` and `_nash_v3/optim.py:77`:** Replace unscaled additive `+ eps` with dimensionless relative floor (`+ 1e-8` in `sc`-normalized coordinates, matching `_r4_ablation/optimizer.py:269`) and compute `v` in factored `sc`-normalized form to unlock genuine $10^{-12}$–$10^{-37}$ rail escape.
2. **`_unified_v12/optimizer.py:337,688,1034` and `_unified_v11/optimizer.py:973,994,1106`:**
   - Remove raw `gf_safe.norm()` in `UnifiedV11EquilibratedOptimizer` (lines 973–979) and `UnifiedV12Optimizer._step_generator_free_critic` (lines 506–512), using the underflow-safe `(g_unit_scaled.norm() * safe_scale)` pattern from `_unified_v12/optimizer.py:326`.
   - Remove 1D `clamp_min(eps)` in `_unified_v11/optimizer.py:1106`.
   - Remove dead `st["peak_rms"]` tracking on prior rows (or wire it intentionally into `s_p_row`), and evaluate replacing the `min(s_base, c * peak_rms)` cap + `p_dec = 1.0` ratchet with an uncapped, outlier-robust scale reference (e.g., quantile/Huber-clipped sliding window paired with directional flip-cosine and data-space residual gating) under a formal Forge study.
