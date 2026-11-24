# P4 Adversarial Audit: Generality of Function-Space Optimization Beyond GANs

> **Audit Scope & Governance:** Adversarial mathematical, literature, and empirical audit testing whether ParticleGAN's realized function-space displacement regulation (`delta_g_max` / pushforward trust band, `_ema_func_gain`, Marchenko–Pastur scaling, and `10^-38` dynamic scaling in Law 4) constitutes a genuinely general optimization principle beyond GANs, or a lightweight one-shot step-size controller with well-established literature equivalents and sharp blind spots. Executed on branch `research/fl-p4-funcspace` (branched from `origin/research/four-laws`) using workspace `particlegan` under protocol seed `0` and `deterministic_orthogonal_` initialization (`PyTorch 2.15.0+cu132`, CPU bit-exact reference). Existing Forge receipts and [`docs/four-laws.md`](../../docs/four-laws.md) remain strictly untouched; proposed exact edits are documented in [Section 7](#7-stronger-corrected-formulation--exact-proposed-diffs).
>
> **Reproducibility:**
> - Experiment runner: [`scripts/four_laws_audit/p4_funcspace_experiments.py`](../../scripts/four_laws_audit/p4_funcspace_experiments.py)
> - Automated proofs, controls & falsifications: [`tests/four_laws_audit/test_p4_funcspace_generality.py`](../../tests/four_laws_audit/test_p4_funcspace_generality.py)

---

## Executive Verdict & Taxonomy Classification

**Core Question:** Is realized pushforward trust (`delta_g_max` + `_ema_func_gain`) a genuinely general optimization principle beyond GANs, or an ad-hoc GAN step-size limiter with known literature equivalents?

**Direct Classification (Answering the 4-Option Taxonomy):**
ParticleGAN's realized pushforward trust is **neither** a fundamentally new optimization principle **(4)** **nor** an approximation to full Gauss–Newton / natural-gradient metric geometry **(3)** (it applies a single global scalar multiplier $\alpha \in \mathbb{R}_+$ along a fixed base direction and therefore cannot rotate directions or equalize intra-network layer curvature). It is precisely a **hybrid of (2) an adaptive nonlinear step-size controller and (1) a lightweight, direction-agnostic, $L_2$-output, one-shot-secant approximation to a TRPO-style function-space trust region**, augmented by a closed-loop stiffness EMA (`_ema_func_gain`).

Five sharp conclusions emerge from direct source inspection, formal proofs, literature comparison, and six controlled experiments (Exp 0–5):

1. **What IS Genuinely General & Transferable Beyond GANs (and Distinct from First-Order JVP/Gauss–Newton):**
   - **Realized nonlinear output verification catches $O(\|\Delta\theta\|^2)$, multiplicative-depth, rail-jump, and softmax-flip blowups where first-order linearized JVP ($\|J_\theta f(x)\Delta\theta\|$) is provably blind:** In Exp 2–3, exact first-order JVP underestimates actual realized displacement by **`4,144×`** on a 16-layer unnormalized MLP (`lr=0.2`), **`2,309×`** on a small-weight bilinear saddle (accepting $\alpha_{\text{JVP}} = 1.0$ and suffering a **`10.0×`** trust blowup!), **`96,399×`** on a saturated `tanh` rail jump, and **`5,803×`** on a saturated attention softmax argmax flip (`delta_lin = 6.89e-4` vs. `delta_real = 4.000`).
   - **Realized trust bounds external output gain & hostile LR spikes where parameter-space optimizers (SGD, Adam, LARS, Muon-NS) all break:** In Exp 1 (external gain $\gamma = 1000$, fixed target $y$), raw `SGD` explodes as $\Theta(\gamma^5)$ realized (`6.82 × 10^9`, linear term $\Theta(\gamma^3)$ + bilinear cross-term $\Theta(\gamma^5)$), while `Adam` (`143.8`), `LARS` (`58.07`), and `Muon-NS` (`31.34`) blow up linearly in $\gamma$, whereas `LARS + realized_trust` (`0.0754`) and `Adam + realized_trust` (`0.0900`) keep $\|\Delta f\| \le \delta_{\max}$ ($O(\gamma^0)$ bounded) across all $\gamma$.
   - **Its true domain of comparative advantage is *non-monotone / non-Lyapunov* optimization (minimax GANs, RL policy updates, online non-stationary shifts):** On ordinary supervised minimization (Exp 3 Transformer, Exp 4 spiral classification, hostile regression, pendulum dynamics), whenever a scalar training loss $\mathcal{L}(\theta)$ exists, **classical Armijo backtracking line search matches or beats realized function-space trust** (e.g., pendulum 40-step rollout RMSE: `Armijo 0.0740` vs. `1-step FuncTrust 0.0849` vs. `Adam 0.2802`; Transformer `lr=0.05`: `Armijo 0 spikes` / `GradClip val=0.3293` vs. `Backtrack-FuncTrust 4 spikes, val=0.7320`), and on benign supervised tasks function-space trust stays in its deadband while adding **`+35%–+61%`** wall-clock overhead (`+81%–+134%` for JVP) for zero accuracy gain.

2. **What Is ALREADY KNOWN in the Optimization Literature (No Inflated Novelty):**
   - **Function-space nonlinear trust verification:** Directly predated by **TRPO** (Schulman et al. 2015, exact nonlinear backtracking on realized policy KL divergence) and **function-space regularization / F-KFAC / FSVI** (Benjamin et al. 2019; Sun et al. 2019; Rudner et al. 2022).
   - **Width / operator-norm functional-update scaling:** Directly predated by **muP** (Yang et al. 2021–2022), **Modular Duality / Muon** (Bernstein & Newhouse 2024; Jordan et al. 2024 — ParticleGAN's `unified_v11`/`v12` 2D scaling $\sqrt{\max(1, r/c)}\,\mathrm{NS}(G)$ *is* Modular Duality / Muon with Tikhonov smoothing), and **LARS / LAMB** (You et al. 2017, 2020 — which achieves exact multiplicative layerwise reparameterization invariance where scalar function-space trust does not!).
   - **Dynamic `10^-38` scaling:** Standard IEEE-754 float32 normal-range exponent rescaling (`FLT_MIN = 1.175e-38`, analogous to LAPACK `lassq` / Blue's algorithm) preventing `g^2` underflow for `|g| < 1e-19`.

3. **Critical Implementation Defects & Documentation Mismatches Uncovered in `particlegan/` (Exp 0):**
   - **One-shot linear-secant rescaling WITHOUT post-rescale re-verification violates $\delta_{\max}$ arbitrarily badly on nonlinear maps:** `_unified_v11/optimizer.py:1281-1284`, `_unified_v12/optimizer.py:1211-1217`, `_minimal/dynamics.py:296-299`, and `_t4_nash/optim.py:623-627` apply a single linear ratio $\alpha = \delta_{\max}/\delta(1)$ (or upward amplification $\alpha \le 48$–$96$) and **never re-evaluate $f_{\theta + \alpha\Delta\theta}$**. On saturating `tanh` (`Exp 0`) and 16-layer `tanh` ResNet (`Exp 2B`), one-shot secant **violates $\delta_{\max}$ by `2.65×–6.37×`** (`36/120` trust violations on `mlp16` training!), whereas true iterative backtracking has **`0` violations** across all regimes. Worse, one-shot `delta_g_min` amplification on a superlinear cubic overshoots $\delta_{\max}$ by **`14.17×`**!
   - **`_band_scale` is NOT $C^1$ smooth (falsifying docstrings at `_unified_v4/optimizer.py:48` and `_t4_nash/optim.py:247`):** At $\delta = \delta_{\max} = 0.20$, left/right derivatives jump from `0.00` to `-5.00` ($-1/\delta_{\max}$). Moreover, the amplification threshold is silently shifted below $\delta_{\min}$ to $\delta_{\text{kink}} = \delta_{\min}(1 - 1/\alpha_{\max}) = 0.019583$ (where derivatives jump from `-50.00` to `0.00`), leaving `_band_scale == 1.0` flat at $\delta = \delta_{\min}$!
   - **MP operator-scaling doc mismatch (`docs/four-laws.md:161`):** The $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ denominator exists **only** in elementwise-Adam presets (`t4_nash`, `nash_v3`, `minimal`, `r4_ablation`), **not** in `unified_v11` (which uses $\sqrt{\max(1, r/c)}$ polar aspect) or `unified_v12` (which uses $\sqrt{r}$ on 2D critic Frob-MP and $\sqrt{\text{out}/\text{in}}\sqrt{g/k_{\text{eff}}}\,m_g^{1/4}$ on Conv2d).
   - **`coord_scale` floor (`max(0.05, data_scale / sqrt(out_dim))`) breaks output-scale invariance on tiny per-coordinate scales ($\text{RMS}_{\text{coord}} < 0.05$):** Because `observe_scale` ([`_unified_v11/critic.py:342-343`](../../particlegan/_unified_v11/critic.py#L342-L343)) tracks the full Euclidean L2 RMS $\sqrt{\mathbb{E}[\|x\|_2^2]} = \sqrt{d_{\text{out}}}\,\text{RMS}_{\text{coord}}$ (clamped at `0.05`), standard unit-RMS `64×64×3` and `256×256×3` RGB images have `ema_data_scale = 110.85 / 443.41` and suffer **`1.00×` (zero) floor distortion**. However, tiny per-coordinate scales (`coord_rms = 1e-3`, both 2D vectors and `256×256×3` images) hit the `0.05` floors in `critic.py:343` and `optimizer.py:1260`/`1197`, incurring an exact **`50.0×` distortion** (plus an uninitialized step-0 default `ema_data_scale = 1.0` if `observe_scale` has not run).
   - **Silent bypasses & tiny image probes:** `unified_v11`/`v12` shrink the probe to **`b_p = 2` samples** on `64×64` and `256×256` RGB images (`_unified_v11/critic.py:694`), slice only the batch prefix `[:b_p]`, and disable the probe entirely on simplicial steps, conditional fiber steps (`DDGAN`), and 2-loss `direction_blend` steps; `minimal` disables it for all `out_dim > 32`.

4. **Proven Non-Equivalence to Natural Gradient / LARS (Exp 1 Theorem):**
   - On a functionally equivalent homogeneous reparameterization $W_1' = \alpha W_1, W_2' = \alpha^{-1} W_2$ ($\alpha = 1000$, $10^6\times$ parameter-scale distortion, max initial output error `1.19e-7`), raw `SGD` scales as $\alpha^2$ (`9.32×10^5×`, diverges), `Adam` and `Muon-NS` scale as $\alpha^1$ (`804×` and `545×`, degrading MSE by `989×` and `58,807×`), while **`LARS` is strictly $\alpha^0$-invariant** in BOTH displacement (`5.742e-2`) and layer balance (`0.8514`). By contrast, **undamped minimum-norm Gauss–Newton `-J^+ r` is NOT strictly $\alpha^0$-invariant on homogeneous ReLU networks**, because every $H$-unit ReLU layer has $H$ exact per-neuron scaling symmetries in $\ker(J)$ ($\operatorname{rank}(J) \le P - H = 352 < 384$; verified at `B=128` with `sv_352 = 0.0671` vs. `sv_353 = 9.43e-8`), shifting `W1/W2` (`0.4974 -> 0.4308` at `B=64`, and freezing $W_1'$ completely `0.3554 -> 0.0000` at `B=16`)!
   - Adding scalar realized trust to `SGD`/`Adam`/`Muon-NS` **clamps $\|\Delta f\| \le \delta_{\max}$ and prevents divergence**, **YET leaves layer $W_1'$ frozen** (`W1/W2` contribution ratio collapses from `0.105–0.934` to `0.0–9.68e-7`!), collapsing training onto a frozen-feature linear readout floor (`MSE ~ 0.0049–0.0063`, `18×–23×` worse than `Adam(alpha=1)` = `0.00027`).

5. **Proven Adversarial Probe Blind Spots & Rollout Horizon Limit (Exp 4B & Exp 5):**
   - **Finite-probe i.i.d. underestimation:** Under heavy-tailed localized sensitivity, ParticleGAN's `B = 2` image probe underestimates true test displacement by `>1.5×` on **`90.0%`** of steps (`p95 = 542×`, `max = 1,393×`); even `B = 16` misses `45.5%` (`p95 = 288×`).
   - **1% rare tail mode catastrophe (+ `delta_g_min` weaponization):** A random prefix probe misses a 1% mode with probability **`85.1%`** (`B=16`) / **`98.0%`** (`B=2`). On our explicit localized ReLU failure case, ceiling trust sees `probe = 0.0050 <= 0.10` and lets `rare_disp = 22.71` (`227×` above $\delta_{\max}$) pass untouched — and **ParticleGAN's lower bound `delta_g_min = 0.02` actively amplifies the step by `3.69×`, driving `rare_disp = 83.87` (`839×` above $\delta_{\max}$)!** Stratified probing catches it (`alpha = 0.0167`), yet unweighted RMS still suffers $\sqrt{B_p}$ dilution (`rare_disp = 0.3804 = 3.8× delta_max`).
   - **Exact Probe-Jacobian Nullspace & OOD Blind Spot:** For any network with $P > B_p d_{\text{out}}$ parameters (here $P = 770, B_p d_{\text{out}} = 32 \implies \dim\ker(J_{\text{probe}}) \ge 738$), an explicit nullspace perturbation yields `JVP_ID = 5.09e-9`, `realized_ID = 0.0035`, yet **`realized_OOD = 0.4389` (`127.0×`)**; even a natural Adam step clamped on ID data shifts OOD by **`7.0×`** (`0.6417` vs `0.0915`).
   - **Autoregressive Rollout Limit (Exp 4B):** On Lorenz-63, a tiny 1-step perturbation satisfying `delta_f_1step = 0.0200` amplifies **`21.7×`** (`0.4336`) over an 80-step rollout — proving 1-step function-space trust does not automatically enforce multi-step dynamical stability.

---

## 1. Exact Code Audit of ParticleGAN's Function-Space Mechanisms

Every claim in [`docs/four-laws.md`](../../docs/four-laws.md) and [`reports/forge/studies/law4-optimizer-laws.md`](../forge/studies/law4-optimizer-laws.md) was checked line by line against the actual source code in `particlegan/`:

### 1.1 Ground-Truth Implementation Matrix Across `particlegan/`

| Mechanism | `t4_nash` ([`_t4_nash/optim.py`](../../particlegan/_t4_nash/optim.py)) | `minimal` ([`_minimal/dynamics.py`](../../particlegan/_minimal/dynamics.py)) | `unified_v11` ([`_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py), [`critic.py`](../../particlegan/_unified_v11/critic.py)) | `unified_v12` ([`_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py)) |
| :--- | :--- | :--- | :--- | :--- |
| **Pushforward Trust Rule** | One-shot `_band_scale` (`L239-255`, `L600-629`), **0 post-check**; both G (`alpha_max=96`) & D (`alpha_max=80`, `L384-405`) | One-shot inline band (`L284-299`, `alpha_max=96`), **0 post-check**; G only, **disabled if `ndim!=2` or `d_out>32`** (`L58`) | One-shot `_band_scale` (`L1175-1284`, `alpha_max=48` G / `64` D), **0 post-check**; G + D; bypassed on `is_simp`, `is_cond_fiber`, 2-loss `direction_blend` (`L1183-1185`) | One-shot `_band_scale` (`L1157-1217`, `alpha_max=48`), **0 post-check**; **G only (`0` D probes)**; bypassed on `is_simp`, `is_cond_fiber`, 2-loss `direction_blend` (`L705-713`) |
| **Probe Selection & Budget `b_p`** | Prefix slice `[:k]` (`k<=16`) | Prefix slice `[:k]` (`k=min(16, B)`, `L59`) | Prefix slice `[:b_p]`, `b_p = min(B, max(2, min(16, 8192//out_dim)))` (`critic.py:694`): **`16` for 2D, `2` for images!** | Inherits `_unified_v11/critic.py:694` (`b_p = 16` 2D, **`2` for images!**) |
| **Threshold Formula (`delta_g_max`, `coord_scale`)** | `coord_scale = max(0.05, ds/sqrt(d))` (`L610`); `delta_g_max = (0.008 + 0.105*min(1.5, res))*gamma_g*step*therm` (`L614`) | `coord_scale = max(0.05, ds/sqrt(d))` (`L293`); `delta_g_max = (0.020 + 0.23*min(1.5, res))*g_g*step*mod*wake` (`L294`) | `coord_scale = max(0.05, ds/sqrt(d))` (`L1260`); `delta_g_max = (0.045 + 0.16*min(1.5, push_res))*step_scale` (`L1263`) | Identical to `v11`: `coord_scale` (`L1197`), `delta_g_max` (`L1200`), `delta_g_min` (`L1203`) |
| **Closed-Loop Memory (`_ema_func_gain`)** | None (`last_step_gain` diagnostic only) | None (`last_step_gain` diagnostic only) | Checkpointed `_ema_func_gain` in `[0.05, 1.0]` (`L487,766-768,1289-1308`); drives `pre_gain`, `stiff_gate` (`<0.18`, `L716`), and D coupling `d_couple` (`L771-781`) | Checkpointed `_ema_func_gain` in `[0.05, 1.0]` (`L700-702,1218-1228`); drives `pre_gain`, `stiff_gate` (`<0.60`, `L124`), CountSketch period-2 flip controller (`L126-170`); drops `d_couple` |
| **MP / Operator Scaling Formula** | **True MP bulk edge**: `fan / (sqrt(d_out)+sqrt(d_in))` (`L165-169`), conv `/(sqrt(r)+sqrt(c))sqrt(k_eff)` (`L181-183`) | **Modified MP**: `fan / (0.55*sqrt(do)+sqrt(di))` (`L141`), conv `/(sqrt(r)+sqrt(c))sqrt(ke)` (`L145`) | **NO `sqrt(r)+sqrt(c)`!** Polar aspect `sqrt(max(1, r/c))` (`L1080`), Conv2d `sqrt(out/in)/(kh*kw)` (`L1064`) | **NO `sqrt(r)+sqrt(c)`!** Gen 2D NS `sqrt(max(1, r/c))` (`L191`); Critic 2D Frob `sqrt(r)` (`L319,930`); Conv2d Frob `sqrt(out/in)sqrt(g/k_eff)m_g^0.25` (`L184`) |
| **Dynamic `10^-38` Scaling** | None (standard Adam `g^2` underflows `<1e-19`, `L103`) | Fast path `1e-18..1e18` (`L165`), slow `sc.clamp_min(1e-38)` (`L179-186`) | `g_max > 1e-38` (`L965`); normalized `(m_unit, m_scale, v_rms, v_scale)` (`L1010-1028`) | Fast `_foreach_norm` `1e-18..1e18` (`L1014,1053`), zero-sync slow fallback `1e-38` (`L310,323-325,353-372`) |
| **Genuine Iterative Backtracking?** | **No** (1 probe forward) | **No** (1 probe forward) | **No** (1 probe forward) | **No** (1 probe forward; separate [`StrictProgressOptimizer`](../../particlegan/optim/strict_progress.py#L22-L60) exists in `optim/` with up to 9 Armijo trials for protected losses) |

### 1.2 Exp 0 Direct Verification Findings (Verbatim CPU Seed-0 Outputs)

Running `run_exp0_code_audit(CPU)` directly against the workspace `particlegan` package establishes five concrete audit findings:

1. **`_band_scale` Non-$C^1$ Kinks & Hidden Deadband Shift:**
   - Function ([`_unified_v4/optimizer.py:40-56`](../../particlegan/_unified_v4/optimizer.py#L40-L56), [`_t4_nash/optim.py:239-255`](../../particlegan/_t4_nash/optim.py#L239-L255)):
     $$\alpha(\delta) = \begin{cases} \max(\alpha_{\min}, \delta_{\max}/\delta), & \delta > \delta_{\max}, \\ \min\!\big(\alpha_{\max}, \max(1, \frac{\delta_{\min}}{\delta + \delta_{\min}/\alpha_{\max}})\big), & \delta < \delta_{\min}, \\ 1, & \text{otherwise.} \end{cases}$$
   - **Falsifies docstring *"Smooth $C^1$ impedance-matching gain"* (`_t4_nash/optim.py:247`):**
     - At $\delta = \delta_{\max} = 0.20$: finite-difference left derivative = **`0.0000`**, right derivative = **`-4.999975`** (exact $-1/\delta_{\max} = -5.0$).
     - At $\delta = \delta_{\min} = 0.02$ (`alpha_max = 48`): both left and right derivatives equal **`0.0000`** because $\frac{\delta_{\min}}{\delta_{\min}(1 + 1/48)} = \frac{48}{49} < 1$, clamped to `1.0`! Amplification actually turns on at the shifted kink $\delta_{\text{kink}} = \delta_{\min}(1 - 1/\alpha_{\max}) = 0.0195833$, where the left derivative jumps discontinuously to **`-50.0025`** (exact $-1/\delta_{\min} = -50.0$) vs. right derivative **`0.0000`**.
2. **One-Shot Linear-Secant Trust Violation (Saturating) & Amplification Overshoot (Superlinear):**
   - *Saturating `tanh`* ($f(\theta)=\tanh(\theta), \theta_0=0, \Delta\theta=25, \delta_{\max}=0.20$): $\delta(1) = 1.000 \implies$ `_band_scale` sets $\alpha = 0.200$, yielding **actual post-step displacement $\tanh(5.0) = 0.99991$ (`5.00×` violation of $\delta_{\max}$!)**. Iterative backtracking reaches $\alpha = 0.0078125$ in `8` evals with actual displacement **`0.19287 <= 0.20`**.
   - *Superlinear cubic amplification* ($f(\theta)=0.005\theta^3, \Delta\theta=1, \delta_{\min}=0.05, \delta_{\max}=0.20, \alpha_{\max}=48$): $\delta(1)=0.005 < \delta_{\min} \implies$ `_band_scale` amplifies by $\alpha = 8.276$ without re-checking, producing actual displacement $0.005 \times 8.276^3 =$ **`2.8341` (`14.17×` overshoot above $\delta_{\max}$!)**.
3. **Probe Budget & Real `observe_scale()` + `coord_scale` `0.05` Floor Audit:**
   Because `UnifiedV11StepRecord.observe_scale()` ([`_unified_v11/critic.py:342-343`](../../particlegan/_unified_v11/critic.py#L342-L343)) computes the full Euclidean L2 RMS $\sqrt{\mathbb{E}[\|x\|_2^2]}\,\text{.clamp\_min}(0.05) = \max(0.05, \sqrt{d_{\text{out}}}\,\text{RMS}_{\text{coord}})$, standard unit-RMS RGB images suffer **zero (`1.00×`) floor distortion**, whereas tiny per-coordinate scales ($\text{RMS}_{\text{coord}} < 0.05$) hit the `0.05` clamps in `critic.py:343` and `optimizer.py:1260`/`1197`:

   | Workload Profile (`observe_scale` on workspace `UnifiedV11StepRecord`) | `out_dim` | True `coord_rms` | `b_p` (`critic.py:694`) | Observed `ema_data_scale` | Clamped `coord_scale` | Distortion Factor |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | `vec_2d_rms1` | `2` | `1.0` | **`16`** | `1.4142` | `1.0000` | `1.00×` |
   | `vec_2d_rms1e-3` | `2` | `1e-3` | **`16`** | `0.0500` *(clamped at `critic.py:343`)* | `0.0500` | **`50.00×`** |
   | `img_64x64x3_rms1` | `12,288` | `1.0` | **`2`** | `110.851` | `1.0000` | `1.00×` |
   | `img_256x256x3_rms1` | `196,608` | `1.0` | **`2`** | `443.405` | `1.0000` | `1.00×` |
   | `img_256x256x3_rms1e-3` | `196,608` | `1e-3` | **`2`** | `0.4434` | `0.0500` *(clamped at `optimizer.py:1260`)* | **`50.00×`** |
4. **Dynamic `10^-38` Scaling (`UnifiedV12Optimizer` vs. PyTorch `Adam`):** On a constant FP32 gradient of `1e-24` (where `g^2 = 1e-48` underflows to exact `0.0` below IEEE-754 float32 subnormal range), `UnifiedV12Optimizer` produces a clean `step_norm = 0.02800` whereas `torch.optim.Adam(lr=0.014, eps=1e-8)` outputs **`0.00000`**.
5. **MP-Scaling Formula Inventory (`128×128` matrix):** `t4_nash` elementwise factor = `0.044194` (exact $1/(2\sqrt{128}) = 0.044194$); `minimal` factor = `0.057025` ($1/(1.55\sqrt{128})$); `unified_v11` polar aspect = `1.0000` ($\sqrt{\max(1, r/c)}$); `unified_v12` 2D critic Frob factor = `11.3137` ($\sqrt{128}$). Thus [`docs/four-laws.md:161`](../../docs/four-laws.md#L161) misattributes the $(\sqrt{d_r}+\sqrt{d_c})\sqrt{k_{\text{eff}}}$ formula to `unified_v11`/`v12`.

---

## 2. Formal Mathematical Distinction & Rigorous Literature Positioning

### 2.1 First-Order Linearized JVP vs. Realized Nonlinear Displacement vs. Natural Gradient

Let $f_\theta: \mathcal{X} \to \mathbb{R}^{d_{\text{out}}}$ have parameters $\theta \in \mathbb{R}^P$ and probe batch $X = \{x_1, \dots, x_B\}$. For any candidate step $\Delta\theta \in \mathbb{R}^P$:

1. **First-Order Linearized Function-Space Displacement (Gauss–Newton / JVP Semi-Norm):**
   $$\delta_{\text{lin}}(\Delta\theta; X) := \sqrt{\frac{1}{B d_{\text{out}}} \sum_{i=1}^B \|J_\theta f_\theta(x_i)\,\Delta\theta\|_2^2} = \sqrt{\frac{1}{B d_{\text{out}}} \Delta\theta^\top G_{\text{GN}}(X)\,\Delta\theta}, \quad G_{\text{GN}}(X) := \sum_{i=1}^B J_\theta f_\theta(x_i)^\top J_\theta f_\theta(x_i).$$
   - Evaluates the quadratic pullback metric $G_{\text{GN}}$ along $\Delta\theta$ using one forward-mode JVP (`torch.func.jvp`), capturing **only** first-order local tangent sensitivity.
   - **Invariant to step scaling:** $\delta_{\text{lin}}(t\Delta\theta) = |t|\,\delta_{\text{lin}}(\Delta\theta)$ identically.

2. **Actual Realized Nonlinear Function-Space Displacement:**
   $$\delta_{\text{real}}(\Delta\theta; X) := \sqrt{\frac{1}{B d_{\text{out}}} \sum_{i=1}^B \|f_{\theta + \Delta\theta}(x_i) - f_\theta(x_i)\|_2^2} = \Big\|J_\theta f_\theta\,\Delta\theta + \int_0^1 (1-t)\,\frac{d^2}{dt^2} f_{\theta + t\Delta\theta}\,dt\Big\|_{L_2(\hat\mu_X)}.$$
   - Define the **Nonlinearity Gap Ratio** $\Gamma(\Delta\theta; X) := \delta_{\text{real}}(\Delta\theta; X) / \delta_{\text{lin}}(\Delta\theta; X)$.
   - **Three distinct geometric regimes of $\Gamma$:**
     - **Linear ($\Gamma \approx 1$):** JVP trust, one-shot secant, and backtracking all coincide.
     - **Superlinear / Jump ($\Gamma \gg 1$):** Multiplicative depth composition $(1+\eta)^L$, bilinear saddles $\Delta W_2 \Delta W_1 x$, saturated `tanh` rail escape ($\operatorname{sech}^2 \approx 0$), and softmax argmax flips ($\operatorname{diag}(p)-pp^\top \approx 0$). Here **first-order JVP trust can catastrophically fail to clamp** ($\delta_{\text{lin}} \le \delta_{\max} \ll \delta_{\text{real}}$), whereas realized probes detect the jump immediately!
     - **Concave / Saturating ($\Gamma \ll 1$):** Saturated `tanh` / `softmax` plateaus or ReLU sign-flip cancellation. Here **ParticleGAN's one-shot linear rescaling $\alpha = \delta_{\max} / \delta_{\text{real}}(1)$ provably violates $\delta_{\text{real}}(\alpha\Delta\theta) \le \delta_{\max}$** (because $\delta_{\text{real}}(t)$ is concave in $t$, so $\delta_{\text{real}}(\alpha) > \alpha\,\delta_{\text{real}}(1) = \delta_{\max}$!). Only **iterative realized backtracking** enforces $\delta_{\text{real}}(\alpha\Delta\theta) \le \delta_{\max}$ across *both* $\Gamma \gg 1$ and $\Gamma \ll 1$!

3. **Theorem 1 — Non-Equivalence of Scalar Function-Space Trust to LARS & Natural Gradient Under Homogeneous Reparameterization:**
   Consider a 2-layer positively homogeneous network $f_{W_1, W_2}(x) = W_2 \operatorname{ReLU}(W_1 x)$ ($H$ hidden units, $P$ parameters) and its exact functional reparameterization $W_1' = \alpha W_1, W_2' = \alpha^{-1} W_2$ ($\alpha \ge 1$), so $f_{\theta'}(x) \equiv f_\theta(x)$ for all $x$. Then $\nabla_{W_1'} \mathcal{L} = \alpha^{-1} g_1$ and $\nabla_{W_2'} \mathcal{L} = \alpha g_2$. In the first-order / no-ReLU-sign-flip regime for $\alpha \ge 1$:
   - **Raw SGD:** $\Delta f_{W_1'} = \alpha^{-2} \Delta f_{W_1}$ and $\Delta f_{W_2'} = \alpha^{+2} \Delta f_{W_2} \implies \|\Delta f\| = \Theta(\alpha^2)$ and layer ratio $\|\Delta f_{W_1'}\| / \|\Delta f_{W_2'}\| = \Theta(\alpha^{-4})$.
   - **Coordinate/Spectral-Normalized Base Steps (`Adam` at $\varepsilon \to 0$, `SignSGD`, `Muon-NS`, ParticleGAN `MP/NS`):** $\|\Delta W_1'\| = \|\Delta W_1\|, \|\Delta W_2'\| = \|\Delta W_2\| \implies \Delta f_{W_1'} = \alpha^{-1} \Delta f_{W_1}$ and $\Delta f_{W_2'} = \alpha^{+1} \Delta f_{W_2} \implies \|\Delta f\| = \Theta(\alpha)$ and layer ratio $= \Theta(\alpha^{-2})$.
   - **LARS ($\Delta W_\ell \propto \|W_\ell\|_F g_\ell / \|g_\ell\|_F$):** Satisfies $\Delta W_1' = \alpha \Delta W_1, \Delta W_2' = \alpha^{-1} \Delta W_2$ identically for all $\alpha > 0$, yielding **exact $\alpha^0$ invariance** of BOTH $\|\Delta f\|$ (even including the quadratic cross-term $\Delta W_2' D \Delta W_1' x$ and ReLU sign flips!) AND the layer ratio $\|\Delta f_{W_1'}\|/\|\Delta f_{W_2'}\|$.
   - **Undamped Minimum-Norm Gauss–Newton ($\Delta\theta = -\eta J^\dagger r$):** Under $D = \operatorname{diag}(\alpha I_{P_1}, \alpha^{-1} I_{P_2})$, $J' = J D^{-1}$; the identity $(J D^{-1})^\dagger = D J^\dagger$ holds **only when $J$ has full column rank ($\operatorname{rank}(J) = P$)**. Every 2-layer homogeneous ReLU network with $H$ hidden units has $H$ exact per-neuron scaling symmetries $(w_{1,i}, -w_{2,i}) \in \ker(J)$, so $\operatorname{rank}(J) \le P - H < P$ unconditionally! Consequently, minimum-Euclidean-norm `pinv(J')` minimizes $\alpha^2\|\Delta W_1\|_F^2 + \alpha^{-2}\|\Delta W_2\|_F^2$ in original coordinates: it preserves the linearized update $J'\Delta\theta' = -\eta r$ when $r \in \operatorname{Col}(J)$, **but is NOT strictly $\alpha^0$-invariant in $\Delta\theta$ or $W_1/W_2$**, shifting nullspace mass toward $W_2'$ (and freezing $W_1'$ completely when underdetermined, $n_{\text{out}} < \operatorname{rank}(J_2)$!).
   - **Global Scalar Function-Space Trust ($\theta \leftarrow \theta + c\,\Delta\theta$ with $c \in \mathbb{R}_+$):** Restores $\|\Delta f\| \le \delta_{\max}$ ($O(\alpha^0)$ magnitude!) on SGD/Adam/Muon-NS, **but leaves the intra-network ratio $\|\Delta f_{W_1'}\| / \|\Delta f_{W_2'}\| = \Theta(\alpha^{-4})$ (SGD) or $\Theta(\alpha^{-2})$ (Adam/Muon-NS) strictly unchanged**, freezing $W_1'$!
   - **Dual External-Gain Theorem ($f'(x) = \gamma\,W_2\operatorname{ReLU}(W_1 x)$ for $\gamma \gg 1$, fixed target $y$):** Here residual $r_\gamma = \Theta(\gamma)$ and $J' = \gamma J \implies g' = \Theta(\gamma^2)$, so for raw SGD $\Delta W_1, \Delta W_2 = \Theta(\gamma^2)$, making the linearized shift $\Theta(\gamma^3)$ and the **realized displacement $\Theta(\gamma^5)$** via the bilinear cross-term $\gamma\Delta W_2\operatorname{ReLU}(\Delta W_1 x)$! Meanwhile LARS leaves $\|W_\ell\|$ unchanged so $\|\Delta f_{\text{LARS}}\| = \Theta(\gamma)$ explodes linearly in $\gamma$ (as do Adam and Muon-NS). Composing **`LARS + Realized Function-Space Trust`** achieves **exact $\alpha^0$ invariance to internal homogeneous gauge $\alpha$ AND $O(\gamma^0)$ bounded displacement ($\le \delta_{\max}$) across external output gain $\gamma$**! $\square$

4. **Theorem 2 — Finite-Probe Sampling & Jacobian Nullspace Obstruction:**
   Let $X_{\text{probe}} \subset \mathcal{X}$ have cardinality $B_p$ and let $f_\theta: \mathcal{X} \to \mathbb{R}^{d_{\text{out}}}$ have $P$ parameters.
   - *(a) Minority-Mode Miss Probability:* Any disjoint region/mode $M \subset \mathcal{X}$ of data mass $p \in (0, 1)$ is absent from an i.i.d. probe slice of size $B_p$ with exact probability $(1 - p)^{B_p}$ (`85.15%` for $p=0.01, B_p=16$; `98.01%` for $p=0.01, B_p=2$). Whenever a localized subnetwork (e.g., ReLU/MoE/attention head) has zero support on $X_{\text{probe}}$ and nonzero support on $M$, $\delta_{\text{real}}(\Delta\theta; X_{\text{probe}}) = 0$ even as $\delta_{\text{real}}(\Delta\theta; M) \to \infty$.
   - *(b) Unweighted RMS Dilution:* Even when $1$ rare sample is included in a stratified probe of size $B_p$, unweighted RMS scales that sample's displacement by $1/\sqrt{B_p}$, permitting $\delta_{\text{rare}} \le \sqrt{B_p}\,\delta_{\max}$ (`4×` at $B_p=16$).
   - *(c) Probe-Jacobian Nullspace:* Whenever $P > B_p d_{\text{out}}$, the probe Jacobian $J_{\text{probe}} \in \mathbb{R}^{(B_p d_{\text{out}}) \times P}$ satisfies $\dim\ker(J_{\text{probe}}) \ge P - B_p d_{\text{out}}$. Every tangent step $\Delta\theta \in \ker(J_{\text{probe}})$ has exact $\delta_{\text{lin}}(\Delta\theta; X_{\text{probe}}) = 0$ and $\delta_{\text{real}}(\Delta\theta; X_{\text{probe}}) = O(\|\Delta\theta\|^2)$, while $\delta_{\text{real}}(\Delta\theta; X_{\text{OOD}})$ can be arbitrarily large on shifted inputs $X_{\text{OOD}} \not\subset \operatorname{span}(X_{\text{probe}})$. $\square$

### 2.2 Rigorous Literature Positioning (No Inflated Novelty Claims)

| Prior Framework | Exact Mechanism | Rotates Direction? | Captures Realized Nonlinearity? | Needs Scalar Objective $\mathcal{L}$? | Exact Relationship to ParticleGAN Law 4 (`delta_g_max` + MP/NS) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Natural Gradient** (Amari 1998) | $\Delta\theta = -\eta F(\theta)^{-1} g$ | **Yes** (exact Fisher) | No (1st-order quadratic) | No | Strictly stronger local geometry (exact diffeomorphism invariance); $O(P^3)$ cost vs. ParticleGAN's $O(P)$ scalar rescaling |
| **K-FAC** (Martens & Grosse 2015) | Kronecker Fisher $A_{\ell-1}^{-1} G_\ell G_\ell^{-1}$ + LM damping ratio $\rho$ | **Yes** (block Kronecker) | Loss ratio $\rho$ only | **Yes** (LM $\rho$) | Preconditions direction via curvature + LM loss check; ParticleGAN skips Fisher inversion & checks output RMS instead of loss |
| **TRPO** (Schulman et al. 2015) | CG Natural Gradient + **iterative nonlinear backtracking** enforcing realized $\hat{D}_{\text{KL}}(\pi_{\text{old}}\|\pi_{\theta'}) \le \delta$ | **Yes** (CG FVP) | **Yes** (strictly verified!) | Surrogate only (non-monotone RL) | **Closest direct ancestor!** ParticleGAN's `delta_g_max` is a direction-agnostic, $L_2$-output, *non-backtracking* one-shot analogue of TRPO's KL verification + EMA pre-gain |
| **Functional Reg. / F-KFAC / FSVI / FROMP** (Benjamin et al. 2019; Sun et al. 2019; Pan et al. 2020; Rudner et al. 2022) | Penalizes/constrains $\|f_\theta(X_M) - f_{\theta_0}(X_M)\|$ or KL on context/measurement set $X_M$ | Via penalty/projection | **Yes** (finite $X_M$) | Usually yes | Already identified both finite-set function-space control AND the **measurement-set / OOD blind spot** (Exp 5) |
| **LARS / LAMB** (You et al. 2017, 2020) | Layerwise trust ratio $\eta \|W_\ell\| / \|u_\ell\|$ | Layerwise rescaling | No | No | Strictly stronger for internal homogeneous reparameterization ($\alpha^0$ invariant, Exp 1); fails external gain $\gamma$ & zero-weight init where realized trust succeeds |
| **muP** (Yang et al. 2021–2022) | Asymptotic width-$W$ init & LR scaling so $\|\Delta f\| = \Theta(1)$ | Static per-layer scaling | No (*a priori*) | No | Analytical asymptotic width scaling; blind to finite-width hostile gain ($\gamma=548\times$), depth blowup, and runtime softmax spikes |
| **Modular Duality / Muon** (Bernstein & Newhouse 2024; Jordan et al. 2024) | Per-layer RMS-to-RMS spectral norm $\sqrt{d_{\text{out}}/d_{\text{in}}}\,\mathrm{NS}(G_\ell)$ | **Yes** (per-layer SVD basis) | No (per-layer worst-case bound) | No | ParticleGAN's `unified_v11`/`v12` 2D NS scaling *is* Modular Duality / Muon + Tikhonov smoothing; still blind to multi-layer composition & output gain ($\Theta(\alpha)$ and $\Theta(\gamma)$ in Exp 1) |
| **Armijo / Wolfe & Levenberg–Marquardt** (Armijo 1966; Marquardt 1963; Nocedal & Wright 2006) | Iterative backtracking checking actual loss decrease $\mathcal{L}(\theta+\alpha d) \le \mathcal{L}(\theta) + c_1\alpha g^\top d$ | No (Armijo) / Yes (LM) | **Yes** (objective) | **Yes** (strictly requires Lyapunov $\mathcal{L}$) | Matches or beats function-space trust on supervised minimization (Exp 3, Exp 4); **inapplicable to minimax GAN games**, which is why output-space trust matters in GANs/RL |
| **ParticleGAN Pushforward Trust** (`delta_g_max` + `_ema_func_gain`) | 1-forward probe $\delta_G = \mathrm{RMS}(\Delta G(X_p))/s_{\text{coord}}$, one-shot `_band_scale` + slow EMA pre-gain | **No** (global scalar only) | **Partial** (1-shot only; breaks on concave/amplified steps!) | **No** | Lightweight, cheap ($1$ probe fwd), works without scalar Lyapunov loss; **not** a new principle — an unverified one-shot TRPO-like step-size controller needing iterative backtracking to be sound |

---

## 3. Controlled Beyond-GAN Experiment Results (Verbatim Seed-0 Data)

All numbers below are verbatim outputs of:
`PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p4_funcspace_experiments.py --device cpu` (`41.33s`, `PyTorch 2.15.0+cu132`, protocol seed `0`, `deterministic_orthogonal_` init). Every executed arm is reported verbatim with zero omissions.

### 3.1 Experiment 1 — Functionally Equivalent Reparameterizations ($\alpha \in \{1, 10, 100, 1000\}$) & External Output Gain ($\gamma$)

Two-layer homogeneous bias-free ReLU network $f(x) = W_2 \operatorname{ReLU}(W_1 x)$ ($8 \to 32 \to 4$), reparameterized via $W_1' = \alpha W_1, W_2' = \alpha^{-1} W_2$ (max initial output discrepancy across $10^6\times$ parameter ratio: `0.0`, `1.19e-7`, `1.19e-7`, `1.19e-7`), trained for 100 steps ($\delta_{\max} = 0.10$):

#### Table 3.1A — All 11 Arms Across All 4 Reparameterization Scales ($\alpha \in \{1, 10, 100, 1000\}$)

| Optimizer Arm | $\delta_f^{(1)}(\alpha=1)$ | $\delta_f^{(1)}(\alpha=10)$ | $\delta_f^{(1)}(\alpha=100)$ | $\delta_f^{(1)}(\alpha=1000)$ | $\frac{\Delta f_{W_1}}{\Delta f_{W_2}}(\alpha=1)$ | $\frac{\Delta f_{W_1}}{\Delta f_{W_2}}(\alpha=1000)$ | $\text{MSE}_{100}(\alpha=1)$ | $\text{MSE}_{100}(\alpha=10)$ | $\text{MSE}_{100}(\alpha=100)$ | $\text{MSE}_{100}(\alpha=1000)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sgd` | `8.322e-3` | `7.754e-1` | `7.754e+1` | **`7.754e+3`** ($\propto\alpha^2$) | `0.1051` | `1.70e-13` ($\propto\alpha^{-4}$) | `0.0203` | **`inf`** | **`inf`** | **`inf`** |
| `adam` | `1.283e-1` | `1.029e+0` | `1.031e+1` | **`1.031e+2`** ($\propto\alpha^1$) | `0.4166` | `4.19e-7` ($\propto\alpha^{-2}$) | `0.00027` | `0.00154` | `0.00452` | **`0.2686`** |
| `lars` | `5.742e-2` | `5.742e-2` | `5.742e-2` | **`5.742e-2`** ($\alpha^0$!) | `0.8514` | **`0.8514`** ($\alpha^0$!) | `0.00184` | `0.00184` | `0.00184` | **`0.00184`** |
| `muon_ns` | `3.130e-2` | `1.717e-1` | `1.707e+0` | **`1.707e+1`** ($\propto\alpha^1$) | `0.9339` | `9.34e-7` ($\propto\alpha^{-2}$) | `0.00035` | `0.01205` | `0.1641` | **`20.759`** |
| `exact_gn_undamped` | `3.005e-2` | `3.099e-2` | `3.099e-2` | `3.099e-2` *(rank-def. shift)* | `0.4974` | `0.4308` *(`0.000` @ $B=16$!)* | `3.2e-9` | `2.5e-8` | `2.7e-8` | `2.6e-8` |
| `exact_gn_damped` ($\lambda=10^{-2}$) | `2.646e-2` | `2.981e-2` | `2.994e-2` | `2.994e-2` | `0.2483` | `1.90e-7` *(broken by $\lambda I$)* | `0.00054` | `0.00206` | `0.00230` | `0.00230` |
| `sgd+jvp_trust` | `8.322e-3` | `1.000e-1` | `1.000e-1` | **`1.000e-1`** | `0.1051` | **`0.000e+0`** *(frozen $W_1$)* | `0.0203` | `0.00633` | `0.00633` | `0.00633` |
| `sgd+realized_trust` | `8.322e-3` | `9.000e-2` | `9.000e-2` | **`9.000e-2`** | `0.1051` | **`0.000e+0`** *(frozen $W_1$)* | `0.0203` | `0.00479` | `0.00614` | `0.00492` |
| `adam+realized_trust` | `6.513e-2` | `9.032e-2` | `9.000e-2` | **`9.000e-2`** | `0.4183` | **`5.15e-7`** *(frozen $W_1$)* | `0.00035` | `0.00088` | `0.00272` | `0.00561` |
| `muon_ns+realized_trust` | `3.130e-2` | `8.592e-2` | `9.002e-2` | **`9.000e-2`** | `0.9339` | **`9.68e-7`** *(frozen $W_1$)* | `0.00035` | `0.00479` | `0.00478` | `0.00500` |
| `lars+realized_trust` | `5.742e-2` | `5.742e-2` | `5.742e-2` | **`5.742e-2`** | `0.8514` | **`0.8514`** | `0.00184` | `0.00184` | `0.00184` | **`0.00184`** |

*(Direct rank-deficiency verification in `run_exp1_reparameterization`: on overdetermined $B=128$ [`n_out = 512 > P = 384`], `rank(J, tol=1e-6) = 352 = P - 32` with `sv_352 = 0.0671` vs. `sv_353 = 9.43e-8` from the $H=32$ exact per-neuron scaling symmetries; on underdetermined $B=16$ [`n_out = 64 < rank(J_2) = 128`], `pinv(J)` collapses `W1/W2` from `0.3554` at $\alpha=1$ to **`0.0000`** at $\alpha=1000$, confirming Theorem 1!)*

#### Table 3.1B — Companion External Output-Gain Test ($f'(x) = \gamma\,W_2\operatorname{ReLU}(W_1 x)$, Fixed Target $y$, All 6 Arms)

| Optimizer Arm | $\delta_f(\gamma=1)$ | $\delta_f(\gamma=10)$ | $\delta_f(\gamma=100)$ | $\delta_f(\gamma=1000)$ | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `sgd` | `0.0083` | `5.5343` | `64,744.2` | **`6.818 × 10^9`** ($\propto \gamma^5$ realized!) | Catastrophic $\Theta(\gamma^5)$ blowup (`105,309×` from $\gamma=100\to 1000$) |
| `adam` | `0.1283` | `1.4034` | `14.187` | **`143.80`** ($\propto \gamma$) | Linear blowup |
| `lars` | `0.0574` | `0.5747` | `5.8011` | **`58.068`** ($\propto \gamma$) | **Fails external output gain!** |
| `muon_ns` | `0.0313` | `0.3114` | `3.1344` | **`31.344`** ($\propto \gamma$) | **Fails external output gain!** |
| `lars+realized_trust` | `0.0574` | `0.0929` | `0.0901` | **`0.0754`** ($\alpha=0.00125$) | **Exact $\alpha^0$-invariant + $O(\gamma^0)$ bounded $\le \delta_{\max}$!** |
| `adam+realized_trust` | `0.0651` | `0.0943` | `0.0902` | **`0.0900`** ($\alpha=0.00060$) | Bounded across $\gamma$, freezes $W_1$ under $\alpha$ |

**Mechanistic Takeaway:** Exp 1 cleanly separates **intra-network homogeneous layer gauge $\alpha$** (solved strictly by LARS, *not* by scalar trust and *not* strictly by minimum-norm `pinv(J)` due to ReLU scaling symmetries!) from **external output-gain / LR impedance $\gamma$** (bounded by realized function-space trust, *not* by LARS or Muon-NS alone). Only their composition (`LARS + realized_trust`) handles both!

---

### 3.2 Experiment 2 — Extreme Jacobian Conditioning & Nonlinearity Gap ($\|J\Delta\theta\|$ vs. $\|f(\theta+\Delta\theta) - f(\theta)\|$)

#### Table 3.2A — 16-Layer Skip-Free Unnormalized `LeakyReLU(0.2)` MLP (`delta_max = 0.20`, All 6 LRs)

| Step `lr` | 1st-Order $\delta_{\text{lin}}$ (JVP) | Realized $\delta_{\text{real}}$ | Gap $\Gamma = \frac{\delta_{\text{real}}}{\delta_{\text{lin}}}$ | JVP-Trust Post-$\delta$ | **One-Shot Secant Post-$\delta$** | **Backtrack Post-$\delta$ (`evals`)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1e-4` | `2.043e-1` | `2.027e-1` | `0.992` | `0.1985` | `0.2000` | `0.1016` (`2`) |
| `1e-3` | `2.043e+0` | `1.595e+0` | `0.781` | `0.1985` | **`0.2536`** *(1.27× viol.)* | `0.1147` (`3`) |
| `1e-2` | `2.043e+1` | `6.080e+0` | **`0.298`** *(kink/concave)* | `0.1985` | **`0.6409`** *(**3.20× VIOLATION!**)* | `0.1953` (`3`) |
| `5e-2` | `1.021e+2` | `4.996e+2` | **`4.891`** *(superlinear)* | `0.1985` | `0.0407` *(5× over-damped)* | `0.1795` (`4`) |
| `1e-1` | `2.043e+2` | `1.992e+4` | **`97.53`** | `0.1985` | `0.0204` *(hits $\alpha_{\min}=10^{-4}$)* | `0.1851` (`4`) |
| `2e-1` | `4.086e+2` | `1.693e+6` | **`4,144.2`** | `0.1985` | `0.0407` *(hits $\alpha_{\min}=10^{-4}$)* | `0.1953` (`4`) |

- **120-Step Training on 16-Layer Unnormalized MLP (`lr = 0.05`, `delta_max = 0.20`, All 4 Arms):**
  - `adam`: **`final_mse = inf` (DIVERGED)**, `max_realized_disp = 14,067.0`, `10` violations before blowup.
  - `adam+jvp_trust`: `final_mse = 0.00525`, `max_realized_disp = 0.2165`, `2` mild violations.
  - `adam+oneshot_secant`: `final_mse = 0.00057`, `max_realized_disp = 0.3604` (**`1.80×` violation**), **`36` trust violations**!
  - `adam+backtrack_trust`: `final_mse = 0.00248`, `max_realized_disp = 0.1995`, **`0` trust violations**!

#### Table 3.2B — 16-Layer Saturating `tanh` ResNet (`pre_gain = 5.0`, `delta_max = 0.20`, All 5 LRs) & JVP-Blindness Witnesses

| Regime / `lr` | 1st-Order $\delta_{\text{lin}}$ (JVP) | Realized $\delta_{\text{real}}$ | Gap $\Gamma = \frac{\delta_{\text{real}}}{\delta_{\text{lin}}}$ | JVP-Trust Post-$\delta$ | **One-Shot Secant Post-$\delta$ (Viol.)** | **Backtrack Post-$\delta$ (`evals`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tanh` ResNet16 (`lr=1e-3`) | `10.641` | `0.7019` | `0.0660` | `0.1536` | **`0.5301` (`2.65×`)** | `0.1715` (`5`) |
| `tanh` ResNet16 (`lr=1e-2`) | `106.41` | `1.0703` | `0.0101` | `0.1536` | **`0.7877` (`3.94×`)** | `0.1993` (`6`) |
| `tanh` ResNet16 (`lr=5e-2`) | `532.03` | `1.2625` | `0.0024` | `0.1536` | **`1.0349` (`5.17×`)** | `0.1831` (`7`) |
| `tanh` ResNet16 (`lr=2e-1`) | `2,128.1` | `1.2831` | `0.0006` | `0.1536` | **`1.2333` (`6.17×`)** | `0.1650` (`8`) |
| `tanh` ResNet16 (`lr=1.0`) | `10,640.5` | `1.2102` | `0.0001` | `0.1536` | **`1.2746` (`6.37×`)** | `0.1547` (`9`) |
| **`tanh` Rail Jump** ($b_0=8.0$) | **`5.289e-6`** *(JVP blind!)* | **`0.5099`** | **`96,399×`** | **`0.5099`** *($\alpha=1.0$, `2.55×` viol.!)* | `0.01753` *($\alpha=0.3922$, `11.4×` over-damped)* | **`0.00313`** (`2`) |
| **Bilinear Saddle** ($\|W\|\sim 10^{-3}$) | **`8.686e-4`** *(JVP blind!)* | **`2.0055`** | **`2,308.8×`** | **`2.0055`** *($\alpha=1.0$, **10.0× blowup!**)* | `0.01999` *($\alpha=0.0997$, `10.0×` over-damped)* | **`0.01620`** (`2`) |

**Key Structural Takeaway:** First-order JVP trust fails catastrophically whenever $\delta_{\text{lin}} \le \delta_{\max} \ll \delta_{\text{real}}$ (rail jumps, bilinear/attention saddles); ParticleGAN's one-shot secant fails catastrophically whenever $\Gamma \ll 1$ (`2.65×–6.37×` violations across every single `tanh` ResNet LR!) or when amplifying via `delta_g_min`; **only iterative realized backtracking is sound across all three regimes** ($\Gamma \ll 1$, $\Gamma \approx 1$, $\Gamma \gg 1$)!

---

### 3.3 Experiment 3 — Small Multi-Head Self-Attention Transformer (Associative Recall, Q/K Softmax Nonlinearity)

1. **Exact Softmax Argmax-Flip Witness:** 2-key attention with saturated logits $(12, 0) \to (-2, 14)$ and values $(+2, -2)$:
   - First-order JVP: **`delta_lin = 6.893e-4`** (completely blind because $\operatorname{diag}(p) - p p^\top \approx 6.1 \times 10^{-6}$!).
   - Actual realized jump: **`delta_real = 4.0000`** (**$\Gamma = 5,803.0\times$!**).
2. **2-Layer 4-Head Transformer Nonlinearity Gap Sweep (`qk_gain = 4.0`, `delta_max = 0.25`, All 6 LRs):**

| `lr` | $\delta_{\text{lin}}$ (JVP) | $\delta_{\text{real}}$ | $\Gamma$ | JVP Post-$\delta$ | One-Shot Secant Post-$\delta$ | Backtrack Post-$\delta$ (`evals`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1e-4` | `8.781e-4` | `8.779e-4` | `1.000` | `8.779e-4` | `8.779e-4` | `8.779e-4` (`1`) |
| `1e-3` | `8.781e-3` | `8.767e-3` | `0.998` | `8.767e-3` | `8.767e-3` | `8.767e-3` (`1`) |
| `1e-2` | `8.781e-2` | `8.677e-2` | `0.988` | `8.677e-2` | `8.677e-2` | `8.677e-2` (`1`) |
| `5e-2` | `4.390e-1` | `4.470e-1` | `1.018` | `0.2468` | `0.2423` | `0.2163` (`2`) |
| `1e-1` | `8.781e-1` | `1.043e+0` | `1.188` | `0.2468` | `0.2074` | `0.1865` (`2`) |
| `3e-1` | `2.634e+0` | **`56.561`** | **`21.47`** | `0.2468` | `0.0116` *(21× over-damped)* | `0.1298` (`2`) |

3. **150-Step Associative-Recall Training Sweep (`qk_gain = 2.5`, All 7 Arms × All 3 LRs = 21 Runs Verbatim):**

| Optimizer Arm | `lr=0.005` (`tr` / `va` / `spk` / `maxL` / `ms`) | `lr=0.02` (`tr` / `va` / `spk` / `maxL` / `ms`) | `lr=0.05` (`tr` / `va` / `spk` / `maxL` / `ms`) |
| :--- | :--- | :--- | :--- |
| `adam` | `7.9e-5` / `0.8319` / `0` / `39.7` / `4.81` | `4.7e-5` / `0.7550` / **`11`** / `73.8` / `4.64` | **`inf` / `inf` / `24` / `8.16e8` (DIVERGED)** / `4.82` |
| `adam+grad_clip` | `7.9e-5` / `0.8319` / `0` / `39.7` / `4.85` | `1.8e-5` / `0.7662` / `5` / `74.4` / `4.75` | `7.1e-3` / **`0.3293`** / **`0`** / `86.9` / `4.85` |
| `adam+step_clip` | `4.2e-3` / `0.9481` / `0` / `32.0` / `4.75` | `1.1e-2` / `0.9211` / `0` / `31.6` / `4.59` | `1.1e-2` / `0.9211` / `0` / `31.6` / `4.73` |
| `adam+jvp_trust` | `7.9e-5` / `0.8319` / `0` / `39.7` / `8.71` | `7.5e-4` / `0.6165` / `6` / `45.3` / `8.55` | `3.1e-4` / `0.4927` / `6` / `39.7` / `8.91` |
| `adam+oneshot_secant` | `7.9e-5` / `0.8319` / `0` / `39.7` / `7.18` | `2.1e-4` / **`0.5300`** / `4` / `55.3` / `7.07` | `2.7e-5` / `0.5244` / `2` / `49.8` / `7.39` |
| `adam+backtrack_trust` | `7.9e-5` / `0.8319` / `0` / `39.7` / `6.49` | `2.6e-4` / `0.7342` / `5` / `41.3` / `6.58` | `9.6e-5` / `0.7320` / `4` / `72.8` / `8.33` |
| `adam+armijo` | `2.8e-4` / `0.8493` / `0` / `43.8` / `8.77` | `5.9e-6` / `0.6148` / **`0`** / `51.4` / `8.63` | `2.2e-7` / `0.6218` / **`0`** / `50.4` / `13.87` |

*(Note: `min_attn_entropy` reached `~0.000` [`1e-25` to `1e-7`] across all converged arms on this sharp 1-hot associative lookup task because exact key retrieval requires peaked attention; at `lr=0.05` raw `adam` exploded to `max_logit = 8.16×10^8`.)*

**Honest Interpretation:** Realized function-space trust **does** prevent catastrophic Transformer Q/K logit explosion (`8.16e8 -> 49.8–72.8`) at aggressive `lr=0.05` and outperforms naive parameter-step clipping (`step_clip` stalls at `va=0.9211`). **However**, on this supervised Transformer task, standard **`adam+grad_clip` achieves `0` spikes and the best validation MSE (`0.3293`) at nearly half the runtime (`4.85 ms` vs. `7.39–8.33 ms`)**, and **`adam+armijo` achieves `0` spikes with the lowest training MSE (`2.2e-7`)**. Thus realized function-space trust is *a viable stabilizer*, not a strictly superior replacement for gradient clipping or Armijo backtracking in supervised Transformers.

---

### 3.4 Experiment 4 — Ordinary Supervised Learning & Learned Dynamical System Rollout

#### Table 3.4A — Benign 4-Class Spiral Classification (`lr=0.01`, All 8 Arms) & Hostile `gain=50` Regression (`lr=0.05`, All 7 Arms)

| Optimizer Arm | Benign Spiral `val_acc` | Benign Spiral `val_loss` | Benign `ms/step` | Hostile `gain=50` `final_mse` | Hostile Diverged? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sgd` | `0.4961` | `0.8685` | `0.55` | **`inf`** | **Yes** |
| `adam` | `0.9922` | `0.0335` | `0.74` | `0.08115` | No |
| `lars` | **`0.9961`** | `0.0365` | `0.67` | `0.46066` | No |
| `muon_ns` | `0.9922` | `0.0369` | `1.62` | — *(not in hostile subset)* | — |
| `adam+jvp_trust` | `0.9922` | `0.0335` | `1.73` (`+134%`) | `0.08902` | No |
| `adam+oneshot_secant` | `0.9922` | `0.0335` | `1.19` (`+61%`) | `0.03458` | No |
| `adam+backtrack_trust` | `0.9922` | `0.0335` | `1.05` (`+42%`) | **`0.03025`** (`2.68×` < Adam) | No |
| `adam+armijo` | `0.9922` | `0.0357` | `1.12` (`+51%`) | `0.03433` (`2.36×` < Adam) | No |

#### Table 3.4B — Learned Dynamical System Rollout (Damped Nonlinear Pendulum `gain=10, lr=0.05`, All 5 Arms) & Lorenz-63 Lyapunov Check

| Optimizer Arm (Pendulum 1-Step Training $\to$ 40-Step Open-Loop Rollout) | Val 1-Step MSE | Val 40-Step Rollout RMSE | Val 40-Step Max Err | Diverged Traj. Frac |
| :--- | :---: | :---: | :---: | :---: |
| `adam` | `0.8499` | `0.2802` | `1.6033` | `0.00` |
| `adam+grad_clip` | `0.4033` | `0.5529` *(worse!)* | `1.2214` | `0.00` |
| `adam+1step_func_trust` ($\delta_f \le 0.30$) | **`0.0323`** (`26.3×` < Adam) | **`0.0849`** (`3.30×` < Adam) | **`0.5593`** | `0.00` |
| `adam+rollout20_func_trust` ($\delta_{\text{roll}} \le 0.15$) | `0.0344` | `0.0977` | `0.5923` | `0.00` |
| `adam+armijo_1step` (classical loss backtracking) | **`0.0219`** | **`0.0740`** | `0.5752` | `0.00` |

- **Lorenz-63 Lyapunov Amplification Control:** Applying a controlled output perturbation with exact 1-step realized displacement **`delta_f_1step = 0.0200`** produces an 80-step rollout RMS displacement of **`0.2567`** and terminal step-80 RMS displacement of **`0.4336`** (**`21.68×` amplification!**).
- **Sharp Scope Boundary:** Realized function-space trust rescues hostile-gain dynamics training where raw `Adam` and `GradClip` both fail poorly (`roll40_rmse: 0.280–0.553 -> 0.085`), **yet** (i) when a scalar 1-step loss is available, classic `Armijo` still wins (`0.0740`), and (ii) in chaotic/autoregressive world models, bounding 1-step $\delta_f$ does **not** bound $H$-step rollout divergence without matching the rollout operator norm!

---

### 3.5 Experiment 5 — Adversarial Probe Generalization & Blind-Spot Falsification

#### Table 3.5A — i.i.d. Probe-Size Generalization (`400` Trials vs. `N = 4096` Held-Out Test Set, All 6 Sizes)

| Probe Size $B_p$ | ParticleGAN Regime | Smooth MLP `p95` ($\frac{\delta_{\text{test}}}{\delta_{\text{probe}}}$) | Smooth `max` | Heavy-Tail ReLU `mean` | **Heavy-Tail `p95`** | **Heavy-Tail `max`** | **Underest. $>1.5\times$ Frac** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B = 2`** | **Actual `64×64`/`256×256` RGB (`critic.py:694`)** | `2.12×` | `7.21×` | `272.6×` | **`542.0×`** | **`1,393.4×`** | **`90.0%`** |
| `B = 4` | — | `1.66×` | `2.75×` | `207.4×` | `404.4×` | `685.6×` | `78.5%` |
| `B = 8` | — | `1.39×` | `2.11×` | `154.6×` | `325.4×` | `657.5×` | `63.5%` |
| **`B = 16`** | **Actual 2D vector (`critic.py:694`)** | `1.26×` | `1.87×` | `108.9×` | **`288.5×`** | **`334.3×`** | **`45.5%`** |
| `B = 64` | — | `1.10×` | `1.36×` | `12.4×` | `12.9×` | `246.4×` | `17.8%` |
| `B = 256` | Large stratified probe | `1.05×` | `1.08×` | `1.01×` | `1.25×` | `1.88×` | **`1.0%`** |

#### Table 3.5B — Explicit Adversarial Blind-Spot Constructions (1% Rare Mode + Exact Probe Nullspace + OOD Shift)

| Blind-Spot Scenario | Configuration | Applied $\alpha$ | Measured Probe $\delta_{\text{probe}}$ | **Actual Unprobed / Rare / OOD $\delta$** | **Violation vs. $\delta_{\max}=0.10$** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1% Rare Tail Mode** (`P(miss B=16)=85.1%`, `B=2=98.0%`) | Ceiling-only (`delta_min=0`, maj. probe) | `1.0000` | `0.0050` *(safe!)* | **`22.713`** *(rare cluster destroyed)* | **`227.1×`** |
| **1% Rare Tail Mode** (ParticleGAN `delta_g_min=0.02`!) | **Band with `delta_g_min`** (maj. probe) | **`3.6923`** *(AMPLIFIES!)* | `0.0185` *(safe!)* | **`83.865`** *(weaponized!)* | **`838.7×`** |
| **1% Rare Tail Mode** | Iterative backtrack (maj. probe) | `1.0000` | `0.0050` | **`22.713`** *(backtracking can't fix missing support!)* | **`227.1×`** |
| **1% Rare Tail Mode (Mitigation)** | Stratified probe (`15` maj + `1` rare) | `0.0167` | `0.0900` | **`0.3804`** *(60× safer, yet $\sqrt{16}$ RMS dilution!)* | `3.80×` |
| **Exact Probe Nullspace** ($P=770, B_p d_{\text{out}}=32 \implies \dim\ker \ge 738$) | $\Delta\theta \in \ker(J_{\text{probe}})$ aligned with $J_{\text{OOD}}$ | `1.0000` | `JVP=5.09e-9`, `real=0.0035` | **`0.4389`** (`127.0×` ID probe) | `4.39×` |
| **Natural ID Gradient Step $\to$ OOD Shift** ($\|x\|\approx 1 \to 8$) | ID backtracking Adam step ($\delta_{\text{ID}} \le 0.10$) | `0.3552` | `0.0915` *(safe!)* | **`0.6417`** (`7.01×` ID probe) | **`6.42×`** |

---

## 4. Direct Answers to P4 Research Questions

1. **Is realized pushforward trust a genuinely general principle beyond GANs, or a GAN-specific heuristic?**
   - **Nuanced Verdict:** The *mathematical core* — **bounding realized nonlinear function-space displacement $\|f_{\theta+\alpha\Delta\theta} - f_\theta\|_{L_2(\mu)} \le \delta_{\max}$ via iterative backtracking** — is a **general, domain-agnostic stability mechanism** for **non-monotone / non-Lyapunov optimization** (minimax games, RL policy updates, online shift adaptation, hostile-gain training) that genuinely surpasses both parameter-norm clipping and first-order JVP / Gauss–Newton trust when nonlinearity $\Gamma \gg 1$ (deep compositions, rail jumps, bilinear/attention softmax flips).
   - **However**, it is **neither novel** (TRPO's nonlinear KL verification predates it by a decade; F-KFAC/FSVI formalized finite-probe function-space constraints) **nor universally superior** to classical supervised optimization (where Armijo loss backtracking and gradient clipping match or beat it with less overhead when a true scalar loss exists), **and** ParticleGAN's current realization carries **four GAN-specific or mathematically defective compromises**: (i) one-shot linear rescaling without re-verification (violates $\delta_{\max}$ by `2.65×–6.37×` on saturating maps and `14.17×` under `delta_g_min` amplification), (ii) OT-residual-coupled threshold `delta_g_max(ema_push_res)` and `coord_scale >= 0.05` floor, (iii) `b_p = 2` prefix image probing with `85%–98%` rare-mode blindness, and (iv) inability to fix intra-network layer imbalance (unlike LARS or Natural Gradient).

2. **When does realized function-space trust strictly beat first-order JVP / Gauss–Newton trust?**
   - Precisely when $\delta_{\text{lin}} \le \delta_{\max} \ll \delta_{\text{real}}$ ($\Gamma \gg 1$): (a) saturated `tanh`/sigmoid rails escaping ($\Gamma = 9.64 \times 10^4$), (b) small-weight / bilinear saddles ($\Gamma = 2.31 \times 10^3$), (c) saturated attention softmax argmax flips ($\Gamma = 5.80 \times 10^3$), and (d) deep multiplicative-width/depth blowups ($\Gamma = 4.14 \times 10^3$). Outside these strongly nonlinear regimes ($\Gamma \approx 1$), JVP trust and realized trust behave nearly identically (with realized evaluation still cheaper to implement than JVP on custom ops lacking forward-AD support).

3. **How should a *truly general* beyond-GAN function-space trust controller be formulated?**
   - Replace one-shot `_band_scale` with **1–4 steps of iterative backtracking** verifying actual $\delta_{\text{real}}(\alpha) \le \delta_{\max}$ (and **never** amplifying $\alpha > 1$ without a post-amplification verification forward!).
   - Pair it with **relative-weight-norm / LARS or spectral layer equilibration** so internal reparameterization gauge $\alpha$ AND external output gain $\gamma$ are both invariant (Exp 1).
   - Replace unweighted prefix slicing `[:b_p]` (`b_p = 2` on images!) with **stratified / reservoir probing** and optionally **per-sample $L_\infty$ or quantile ($p_{95}$) displacement** (eliminating $\sqrt{B_p}$ RMS dilution of minority modes), while explicitly acknowledging that **no finite training-support probe can bound unseen OOD regions or long-horizon chaotic rollouts**.
   - Remove the hard `0.05` floor in `coord_scale` (use pure scale-homogeneous `max(1e-12, ema_output_std)` or task-target RMS).

---

## 5. Claim-by-Claim Audit Ledger (10-Category Taxonomy & 8-Point Rubric)

Every substantial claim across `docs/four-laws.md`, `law4-optimizer-laws.md`, `t4-imbalance-nash.md`, `r4-occam-ablation-razor.md`, and `particlegan/` is classified using **one primary label** from the 10-category taxonomy in [`reports/four-laws-audit/README.md`](README.md) and audited across all 8 required dimensions.

### Claim 1 — "Regulating realized function-space displacement into a bounded trust band is a genuinely general optimization principle beyond GANs that replaces parameter-space optimizers"
- **Classification:** **Falsified claim** *(as stated as a novel universal replacement; the narrower valid core is an empirically supported step-size stabilization mechanism specifically for non-monotone / hostile-gain settings, already closely anticipated by TRPO)*
1. **Precise formulation:** Claims that realized pushforward trust (`delta_g_max` + `_ema_func_gain`) is a novel, generally superior function-space optimization principle beyond GANs.
2. **Assumptions:** Implicitly equates scalar output-displacement rescaling with full Riemannian/function-space optimization and assumes supervised, RL, and dynamical-system tasks lack cheaper/stronger scalar Lyapunov backtracking.
3. **Existing supporting evidence:** Strong GAN causal evidence in [`law4-optimizer-laws.md:95,204`](../forge/studies/law4-optimizer-laws.md#L204) (`R1 -> R2` flips both `>500×` Nash tasks from NaN to PASS; Arm 2 knockout collapses `548×` generator `16/16 -> 6/16`) and our Exp 1–4 hostile-gain/LR rescues.
4. **Potential counterexamples:** (a) On benign supervised classification (Exp 4A), realized trust matches Adam identically (`0.9922`) at `+42%–+61%` higher wall-clock cost; (b) on supervised Transformer (`lr=0.05`, Exp 3) and pendulum dynamics (Exp 4B), classical `Armijo` loss backtracking (`0` spikes; `roll40_rmse = 0.0740`) and `GradClip` (`val_mse = 0.3293`, `0` spikes) match or beat realized function-space trust; (c) TRPO (Schulman et al. 2015) already introduced nonlinear function-space (KL) backtracking trust verification.
5. **Minimal falsification experiment:** Compare `adam+backtrack_trust` vs. `adam` and `adam+armijo` on benign 4-class spiral classification and pendulum 1-step dynamics (`run_exp4_supervised_and_dynamics`).
6. **Relevant prior research:** Schulman et al. (2015) *TRPO*; Amari (1998) *Natural Gradient*; Martens & Grosse (2015) *K-FAC*; Benjamin et al. (2019) *Measuring and Regularizing Networks in Function Space*; Armijo (1966); Nocedal & Wright (2006).
7. **Computational verification:** Verified in `test_exp3_transformer_softmax_flip_and_training_stability` and `test_exp4_supervised_scope_and_dynamical_rollout_boundary` (PASS).
8. **Confidence & remaining uncertainty:** **High (95%)** on literature positioning and supervised scope boundary; open empirical question whether combining iterative realized function-space trust with non-stationary continual learning or off-policy RL yields measurable gains over TRPO/PPO/AdamW at large scale.

### Claim 2 — "Scalar realized function-space trust restores reparameterization invariance across functionally equivalent networks (like Natural Gradient / Gauss–Newton)"
- **Classification:** **Falsified claim** *(global scalar trust provably bounds total $\|\Delta f\| \le \delta_{\max}$, yet fails to preserve intra-network layer balance)*
1. **Precise formulation:** Tests whether global scalar rescaling $\Delta\theta \leftarrow c\,\Delta\theta$ keeping $\|\Delta f\| \le \delta_{\max}$ achieves reparameterization invariance under $W_1' = \alpha W_1, W_2' = \alpha^{-1} W_2$ ($\alpha \in \{1, 10, 100, 1000\}$).
2. **Assumptions:** Positively homogeneous 2-layer ReLU network $f(x) = W_2 \operatorname{ReLU}(W_1 x)$ with fixed initial function $f_{\theta'} \equiv f_\theta$.
3. **Existing supporting evidence:** In Exp 1, adding realized trust prevents `SGD` divergence (`inf -> 0.0049`), cuts `Adam` MSE `0.2686 -> 0.0056`, and cuts `Muon-NS` MSE `20.76 -> 0.0050`, bounding $\|\Delta f\| \le 0.090$ across all $\alpha$.
4. **Potential counterexamples:** Theorem 1 proves global scalar multiplication preserves the ratio $\|\Delta f_{W_1'}\| / \|\Delta f_{W_2'}\|$ identically (`0.000` for `sgd+realized_trust`, `5.15e-7` for `adam+realized_trust`, `9.68e-7` for `muon_ns+realized_trust` at $\alpha=1000$), freezing $W_1'$ and trapping all three at the linear-readout floor (`MSE ~ 0.0049–0.0063`, `18×–23×` worse than `Adam(1) = 0.00027` and `2.7×–3.4×` worse than `LARS(1000) = 0.00184`), whereas `LARS` keeps both $\|\Delta f\|$ (`5.742e-2`) and $W_1/W_2$ (`0.8514`) strictly $\alpha^0$-invariant. Note also that minimum-norm undamped `pinv(J)` is itself NOT strictly $\alpha^0$-invariant on homogeneous ReLU nets (`rank(J) <= P - H = 352`, freezing `W1/W2` to `0.000` at `B=16`).
5. **Minimal falsification experiment:** Evaluate step-1 isolated layer displacement ratio $\|\Delta f_{W_1}\| / \|\Delta f_{W_2}\|$ on `HomogeneousTwoLayerNet` at $\alpha = 1000$ (`run_exp1_reparameterization`).
6. **Relevant prior research:** Neyshabur et al. (2015) *Path-SGD: Path-Normalized Optimization in Deep Neural Networks*; You et al. (2017) *LARS*; You et al. (2020) *LAMB*; Amari (1998) *Natural Gradient*; Martens (2020) *New Insights and Perspectives on the Natural Gradient Method*.
7. **Computational verification:** Verified in `test_exp1_reparameterization_scaling_laws_and_non_equivalence_to_natural_gradient` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** (exact algebraic proof + numerical confirmation).

### Claim 3 — "Realized nonlinear function-space probing strictly detects nonlinear explosions ($O(\|\Delta\theta\|^2)$, depth products, rail jumps, attention softmax flips) that first-order linearized JVP / Gauss–Newton trust misses"
- **Classification:** **Proven mathematical result** *(for exact iterative realized verification; supported empirically on MLPs, ResNets, and Attention)*
1. **Precise formulation:** Whenever $\Gamma(\Delta\theta; X) = \delta_{\text{real}} / \delta_{\text{lin}} \gg 1$ with $\delta_{\text{lin}} \le \delta_{\max} < \delta_{\text{real}}$, first-order linearized JVP trust accepts $\alpha = 1.0$ and violates the trust region arbitrarily badly, whereas realized nonlinear evaluation detects and clamps the step.
2. **Assumptions:** Smooth or piecewise-smooth neural map $f_\theta$ evaluated on probe set $X$ (assuming $X$ covers the active region — see Claim 6).
3. **Existing supporting evidence:** Exp 2–3 exact witnesses: 16-layer unnormalized MLP (`Gamma = 4,144×` at `lr=0.2`), small-weight bilinear saddle (`delta_lin = 8.69e-4 <= 0.20`, `jvp_alpha = 1.0`, `delta_real = 2.0055`, `Gamma = 2,309×`, `10.0×` violation), saturated `tanh` rail jump (`delta_lin = 5.29e-6`, `delta_real = 0.5099`, `Gamma = 96,399×`), and saturated attention softmax argmax flip (`delta_lin = 6.89e-4`, `delta_real = 4.0000`, `Gamma = 5,803×`).
4. **Potential counterexamples:** In the small-step linear regime ($\Gamma \approx 1$), JVP and realized trust agree to $O(\|\Delta\theta\|)$ (verified in `test_semantic_controls_jvp_taylor_limit_damage_and_determinism`). Moreover, when `delta_lin > delta_max` on a positively homogeneous ray $t \mapsto t\Delta\theta$, JVP scaling $\alpha = \delta_{\max}/\delta_{\text{lin}}$ can pull the step back into the linear regime (`Exp 2A` single-step sweep), so JVP failure requires either $\delta_{\text{lin}} \le \delta_{\max} \ll \delta_{\text{real}}$ or multi-step curvature mismatch.
5. **Minimal falsification experiment:** Evaluate `_jvp_rms_displacement` vs. `_realized_rms_displacement` on the 2-layer small-weight bilinear saddle and 2-key saturated softmax flip (`run_exp2_nonlinearity_gap`, `run_exp3_transformer_attention`).
6. **Relevant prior research:** Schulman et al. (2015) *TRPO* (motivating nonlinear KL verification over quadratic Fisher approximation); Nocedal & Wright (2006) Ch. 4 (*Trust-Region Methods* — actual vs. predicted model reduction).
7. **Computational verification:** Verified in `test_exp2_nonlinearity_gap_jvp_vs_oneshot_vs_backtracking` and `test_exp3_transformer_softmax_flip_and_training_stability` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** for the mathematical separation; empirically verified on MLPs, ResNets, and Attention.

### Claim 4 — "ParticleGAN's `_band_scale` is a smooth $C^1$ gain keeping the realized function-space step inside $[\delta_{\min}, \delta_{\max}]$" (`_t4_nash/optim.py:247`, `_unified_v4/optimizer.py:48`, `_unified_v11/optimizer.py:1269-1284`, `_unified_v12/optimizer.py:1206-1217`)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims `_band_scale` is $C^1$ smooth and that applying $\theta \leftarrow \theta + \alpha\Delta\theta$ once keeps the resulting realized displacement inside $[\delta_{\min}, \delta_{\max}]$.
2. **Assumptions:** Implicitly assumes $f_{\theta + t\Delta\theta}$ is linear along the ray $t \in [0, \alpha_{\max}]$.
3. **Existing supporting evidence:** Works heuristically on moderately smooth GAN steps when paired with `_ema_func_gain` damping in `unified_v11`/`v12` (passing `3/3` Nash tasks).
4. **Potential counterexamples:** (a) Left/right derivatives of `_band_scale` jump discontinuously at $\delta_{\max}$ (`0.00` vs `-5.00`) and at the shifted amplification kink $\delta_{\min}(1 - 1/\alpha_{\max})$ (`-50.00` vs `0.00`), falsifying $C^1$; (b) on saturating `tanh` (`Exp 0`) and 16-layer `tanh` ResNet (`Exp 2B`), one-shot rescaling violates $\delta_{\max}$ by **`2.65×–6.37×`** (`36/120` violations in `mlp16` training); (c) on superlinear cubic (`Exp 0`), one-shot `delta_g_min` amplification overshoots $\delta_{\max}$ by **`14.17×`**.
5. **Minimal falsification experiment:** Compute finite-difference left/right derivatives of `_band_scale` and compare post-step displacement of `_oneshot_secant_trust` vs. `_backtrack_realized_trust` on $\tanh(25t)$, $0.005t^3$, and `SaturatingTanhResNet` (`run_exp0_code_audit`, `run_exp2_nonlinearity_gap`).
6. **Relevant prior research:** Armijo (1966) backtracking line search; Schulman et al. (2015) *TRPO* iterative backtracking verification; ParticleGAN's own [`StrictProgressOptimizer`](../../particlegan/optim/strict_progress.py#L22-L60).
7. **Computational verification:** Verified in `test_exp0_particlegan_band_scale_non_c1_and_oneshot_failures` and `test_exp2_nonlinearity_gap_jvp_vs_oneshot_vs_backtracking` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim 5 — "Empirical finite-probe pushforward trust (`b_p = 2` on images, `b_p = 16` prefix on vectors) reliably bounds function-space displacement across the data distribution and unseen inputs"
- **Classification:** **Information-theoretic obstruction** *(refutes unconstrained finite-probe generalization whenever support coverage is incomplete or $P > B_p d_{\text{out}}$)*
1. **Precise formulation:** A finite empirical $L_2$ probe of size $B_p$ drawn from the training batch prefix `[:b_p]` (`_unified_v11/critic.py:694`) cannot bound displacement on unprobed rare modes, heavy-tailed localized regions, or out-of-distribution inputs, nor constrain the $(P - B_p d_{\text{out}})$-dimensional probe-Jacobian nullspace.
2. **Assumptions:** Arbitrary neural network with $P > B_p d_{\text{out}}$ parameters or localized/piecewise-linear features.
3. **Existing supporting evidence:** Works empirically on balanced, low-multiplicity unimodal/symmetric GAN benchmarks where `B_p = 16` samples the active modes roughly evenly.
4. **Potential counterexamples:** (a) Under 5%-tail localized sensitivity (`Exp 5A`), `B=2` underestimates true test RMS by `>1.5×` on **`90.0%`** of steps (`p95 = 542×`); `B=16` on **`45.5%`** (`p95 = 288×`); (b) on a 1% rare cluster (`Exp 5B`, missed `85.1%` at `B=16`, `98.0%` at `B=2`), ceiling trust allows **`227×` violation** (`rare = 22.71` while `probe = 0.005`) and **`delta_g_min` amplifies it to `839×` violation (`83.87`)**; (c) exact probe-Jacobian nullspace (`dim >= 738`, `Exp 5C`) gives `JVP_ID = 5.09e-9, real_ID = 0.0035` vs. **`OOD = 0.4389` (`127×`)**, and even a natural ID Adam step produces **`7.0×`** OOD violation (`0.6417` vs `0.0915`).
5. **Minimal falsification experiment:** Run `run_exp5_probe_blindspots` measuring i.i.d. underestimation across `B in {2..256}`, 1% rare-mode failure with/without `delta_g_min`, and exact `ker(J_probe)` OOD shift.
6. **Relevant prior research:** Benjamin et al. (2019); Sun et al. (2019) *Functional Variational Bayesian Neural Networks*; Rudner et al. (2022) *Tractable Function-Space Variational Inference* (measurement-set coverage problem); Pan et al. (2020) *FROMP* (memorable/leverage-point selection).
7. **Computational verification:** Verified in `test_exp5_adversarial_probe_blindspots_and_nullspace_obstruction` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** for the mathematical obstruction (Theorem 2); stratified/leverage-weighted probes with $L_\infty$/quantile aggregation mitigate *in-support* rare modes (`60×` reduction in Exp 5B) while remaining fundamentally unable to guarantee unseen OOD extrapolation.

### Claim 6 — "1-step realized function-space trust stabilizes learned dynamical systems under hostile gain, though it cannot by itself guarantee chaotic long-horizon rollout stability"
- **Classification:** **Empirically supported causal mechanism** *(for 1-step teacher-forced training under hostile gain; falsified if extended to chaotic multi-step rollout sufficiency without matching the rollout operator norm)*
1. **Precise formulation:** Bounding 1-step vector-field displacement $\operatorname{RMS}(\Delta f_\theta(x)) \le \delta_f$ stabilizes 1-step neural dynamics training under hostile output gain (`gain=10, lr=0.05`), but does not bound $H$-step autoregressive divergence when the dynamics carry positive Lyapunov exponents.
2. **Assumptions:** Residual neural ODE / world model $x_{t+1} = x_t + \Delta t\,f_\theta(x_t)$ trained on 1-step pairs.
3. **Existing supporting evidence:** In Exp 4B (damped nonlinear pendulum, hostile `gain=10, lr=0.05`), `adam+1step_func_trust` cuts 1-step MSE **`26.3×`** (`0.8499 -> 0.0323`) and 40-step rollout RMSE **`3.30×`** (`0.2802 -> 0.0849`), beating both raw `adam` and `adam+grad_clip` (`0.5529`).
4. **Potential counterexamples:** (a) Classical `adam+armijo_1step` achieves even better accuracy (`1-step MSE = 0.0219, 40-step RMSE = 0.0740`) because a scalar 1-step loss exists; (b) on chaotic Lorenz-63, a tiny 1-step perturbation satisfying `delta_f_1step = 0.0200` amplifies **`21.68×`** (`0.4336`) over an 80-step rollout due to positive Lyapunov exponent $\lambda_{\max} > 0$.
5. **Minimal falsification experiment:** Train `ResidualDynamicsModel` on damped pendulum and measure 80-step Lorenz-63 terminal displacement under a `0.02` 1-step perturbation (`run_exp4_supervised_and_dynamics`).
6. **Relevant prior research:** Venkatraman et al. (2015) *Improving Multi-Step Prediction of Learned Time Series Models*; Janner et al. (2019) *MBPO* (compounding rollout error); Kidger (2022) *On Neural Differential Equations*.
7. **Computational verification:** Verified in `test_exp4_supervised_scope_and_dynamical_rollout_boundary` (PASS).
8. **Confidence & remaining uncertainty:** **High (90%)**; open research question whether amortized multi-step rollout trust bounds help in **non-stationary model-based RL** (where the policy shifts the state distribution and no fixed scalar loss exists).

### Claim 7 — "Documentation Claims on MP Operator Scaling $(\sqrt{d_r}+\sqrt{d_c})\sqrt{k_{\text{eff}}}$ in `unified_v11`/`v12` and Scale-Invariant `coord_scale`" (`docs/four-laws.md:161`)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims that Law-4 unified solutions normalize layer operator gain by $(\sqrt{d_r}+\sqrt{d_c})\sqrt{k_{\text{eff}}}$ and that `coord_scale = max(0.05, data_scale / sqrt(out_dim))` yields output-scale-invariant pushforward normalization.
2. **Assumptions:** Direct code inspection and numerical verification on workspace `particlegan`.
3. **Existing supporting evidence:** None for `unified_v11`/`v12`; the $(\sqrt{d_r}+\sqrt{d_c})\sqrt{k_{\text{eff}}}$ denominator is present only in elementwise-Adam presets (`_t4_nash/optim.py:168,182`, `_minimal/dynamics.py:141,145`).
4. **Potential counterexamples:** (a) `unified_v11` (`_unified_v11/optimizer.py:1064-1080`) uses Modular-Duality polar aspect $\sqrt{\max(1, r/c)}$ (2D) and $\sqrt{\text{out}/\text{in}}/(k_h k_w)$ (Conv2d); `unified_v12` (`_unified_v12/optimizer.py:180-192,930`) uses $\sqrt{\max(1, r/c)}$ (2D G NS), $\sqrt{r}$ (2D D Frob), and $\sqrt{\text{out}/\text{in}}\sqrt{g/k_{\text{eff}}}m_g^{1/4}$ (Conv2d Frob); (b) although `observe_scale` (`_unified_v11/critic.py:342-343`) tracks full L2 norm $\sqrt{d_{\text{out}}}\,\text{RMS}_{\text{coord}}$ so unit-RMS RGB images incur `1.00×` (zero) distortion, the coupled `0.05` floors in `critic.py:343` and `optimizer.py:1260`/`1197` break output-scale invariance on tiny per-coordinate scales (`coord_rms = 1e-3` incurs exact **`50.0×`** distortion on both 2D vectors and `256×256×3` images!).
5. **Minimal falsification experiment:** Inspect `_unified_v11/optimizer.py:1064-1080`, `_unified_v12/optimizer.py:172-193,930`, and run `run_exp0_code_audit`.
6. **Relevant prior research:** Yang et al. (2021) *muP*; Bernstein & Newhouse (2024) *Modular Duality*; Marchenko & Pastur (1967).
7. **Computational verification:** Verified in `test_exp0_particlegan_band_scale_non_c1_and_oneshot_failures` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim 8 — "Underflow-safe $10^{-38}$ dynamic scale normalization prevents FP32 gradient underflow on extreme attenuation/saturation rails" (`docs/four-laws.md:160`, `_unified_v12/optimizer.py:310,353-372`)
- **Classification:** **Engineering heuristic** *(sound classical numerical-analysis rescaling; not a distinct optimization law)*
1. **Precise formulation:** Normalizing running moments and gradients by their running max-abs scale `sc = max(m_scale, v_scale, g_max).clamp_min(1e-38)` keeps squared terms $g_{\text{scaled}}^2 \in [0, 1]$ across the full IEEE-754 float32 normal range (`>= 1.175e-38`) instead of underflowing at $|g| < 10^{-19}$.
2. **Assumptions:** FP32 arithmetic on gradients in $(10^{-38}, 10^{-19})$ (e.g., deep `tanh` rail saturation $\operatorname{sech}^2(a) \ll 10^{-19}$).
3. **Existing supporting evidence:** In Exp 0 (`run_exp0_code_audit`), on a constant `1e-24` FP32 gradient, `UnifiedV12Optimizer` produces a non-zero step (`step_norm = 0.02800`) whereas `torch.optim.Adam(lr=0.014, eps=1e-8)` underflows to exact `0.0`.
4. **Potential counterexamples:** Subnormal gradients below `1e-38` are explicitly zeroed by `g_max > 1e-38`; in FP64 (`DBL_MIN = 2.2e-308`) or BF16 (`min_normal = 1.18e-38`), the threshold is strictly a floating-point representation engineering parameter, not a mathematical optimization law.
5. **Minimal falsification experiment:** Step `UnifiedV12Optimizer` vs. `torch.optim.Adam` on a `4×4` FP32 tensor with `grad = 1e-24` (`run_exp0_code_audit`).
6. **Relevant prior research:** Blue (1978) *A Portable Fortran Program to Find the Euclidean Norm of a Vector*; LAPACK `SLASSQ` / `hypot` overflow- and underflow-safe norm scaling.
7. **Computational verification:** Verified in `test_exp0_particlegan_band_scale_non_c1_and_oneshot_failures` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

---

## 6. Concrete Engineering Recommendations (Not Applied Here per Ownership Rules)

1. **Upgrade One-Shot `_band_scale` to Bounded Iterative Backtracking (2–4 Max Trials) & Guard Amplification:** Whenever `delta_rel > delta_g_max`, re-evaluate `f(theta + alpha * d)` up to 3–4 times (halving if needed) to eliminate `2.65×–6.37×` saturating-map trust violations. Whenever `delta_g_min` triggers $\alpha > 1$, **mandatory post-amplification probe verification** prevents `14×–839×` superlinear/blind-spot overshoots!
2. **Eliminate `b_p = 2` Image Probe Starvation & `0.05` `coord_scale` Floor:** Replace the coupled `0.05` clamps in `critic.py:343` (`.clamp_min(0.05)`) and `optimizer.py:1260`/`1197` (`max(0.05, data_scale / sqrt(out_dim))`) with scale-homogeneous `max(1e-12, ema_coord_rms)` so tiny-scale datasets ($\text{RMS}_{\text{coord}} < 0.05$) do not suffer `50×` distortion, and use stratified / reservoir probing with `p95` or `max` per-sample aggregation on multi-mode tasks instead of an unweighted 2-sample batch prefix.
3. **Keep Layerwise Gauge Separate from Global Function-Space Trust:** When generalizing Law 4 beyond GANs, pair realized iterative function-space trust with relative-weight-norm (`LARS`/Path-SGD) or spectral (`Muon`) conditioning — recognizing that scalar trust regulates external gain $\gamma$ and nonlinear blowup, *not* internal layer gauge $\alpha$.

---

## 7. Stronger, Corrected Formulation & Exact Proposed Diffs

### 7.1 Corrected Positioning of Law 4.1 & Function-Space Trust

> **Law 4.1 (Sharpened) — Realized Function-Space Trust Control in Non-Monotone Dynamics:**
> In non-monotone or non-Lyapunov optimization (such as minimax GANs or RL policy updates) where classical scalar-loss Armijo backtracking is unavailable, regulating realized nonlinear output displacement $\delta_f(\alpha) = \|f_{\theta+\alpha\Delta\theta} - f_\theta\|_{L_2(\hat\mu_X)} / s_{\text{coord}} \le \delta_{\max}$ prevents $\Theta(\gamma)$ external-gain blowups and $\Gamma \gg 1$ nonlinear jumps (multiplicative depth, rail escapes, bilinear/attention softmax flips) that parameter-norm clipping, LARS, Muon/muP, and first-order JVP/Gauss–Newton trust can miss. However: (i) scalar trust rescaling is **not** equivalent to Natural Gradient or LARS and cannot correct internal layerwise reparameterization imbalance; (ii) sound enforcement across saturating ($\Gamma \ll 1$) and superlinear ($\Gamma \gg 1$) maps requires **iterative realized backtracking** rather than unverified one-shot linear rescaling; (iii) finite empirical probing on $B_p$ samples is fundamentally bounded by support coverage, $\sqrt{B_p}$ RMS dilution, and the $(P - B_p d_{\text{out}})$-dimensional probe-Jacobian nullspace; and (iv) on single-objective supervised minimization, classical Armijo loss backtracking and gradient clipping remain competitive or superior baselines.

### 7.2 Exact Proposed Diffs (For Coordinator / P10 Integration — NOT Applied Here)

#### (1) Proposed diff for `docs/four-laws.md` (lines 85, 160–161)

```diff
--- a/docs/four-laws.md
+++ b/docs/four-laws.md
@@ -82,7 +82,7 @@
 **Core Theory (Fundamental Obstruction):**
 The GAN minimax game $\min_G \max_D \mathcal{L}(G_\theta, D_\psi)$ operates on pushforward measures and potential functions, whereas parameter-space optimizers step in raw parameter coordinates ($\|\Delta\theta\| \sim \eta$), inducing architecture-dependent function-space shifts $\Delta f(u) \sim \Theta(\alpha L \sqrt{W}\,\eta)$. Across capacity imbalances, deep/saturating networks, and stochastic minibatches, this mismatch breaks down into **five falsifiable sub-laws** (detailed in [`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md#5-the-five-formal-optimizer-laws-of-law-4)):
 
-- **Law 4.1 — Inter-Player Function-Space Impedance:** Under player capacity/gain mismatch $\rho \gg 1$ or hostile initialization, equal parameter steps produce $\Theta(\rho)$ function-space displacement mismatch and finite-time divergence unless realized function-space displacements are bounded.
+- **Law 4.1 — Inter-Player Function-Space Impedance:** Under player capacity/gain mismatch $\rho \gg 1$ or hostile initialization, equal parameter steps produce $\Theta(\rho)$ function-space displacement mismatch and finite-time divergence unless realized function-space displacements are bounded via a TRPO-like nonlinear output trust controller (distinguished from first-order JVP/Gauss–Newton by catching $\Gamma \gg 1$ nonlinear jumps, yet distinct from Natural Gradient/LARS because global scalar rescaling cannot fix intra-network layer gauge and remains subject to finite-probe support and nullspace limits).
 - **Law 4.2 — Intra-Player Degree-0 Gauge & Rail Escape:** Multiplicative inter-layer Jacobian attenuation across deep networks and $\operatorname{sech}^2$-like activation saturation span many orders of magnitude ($10^{-12}$ to $10^{+2}$), defeating any single player-level learning rate or fixed additive $\varepsilon$ floor. Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish ($\Delta W \to 0$ as $G \to 0$) is required without a step clock.
@@ -158,8 +158,8 @@
 ### Solutions to Law 4 (Function-Space Two-Player Dynamics)
 
 - **Underflow-Safe $10^{-38}$ Dynamic Scale Normalization & Homogeneous `peak_rms` Settling** ([`t4-imbalance-nash.md`](../reports/forge/studies/t4-imbalance-nash.md), [`r3-nash-complete.md`](../reports/forge/studies/r3-nash-complete.md)): Normalizes $(m_t, v_{\text{rms},t})$ by $s_t = \max(\max |v_{\text{rms},t-1}|, \max |g_t|, 10^{-38})$ so $10^{-12}$ early-layer and $\operatorname{sech}^2$-attenuated gradients receive the same directional geometry as $10^{+2}$ output gradients, while degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock.
-- **Closed-Form Marchenko–Pastur Operator Scaling + Realized Generator Pushforward Trust Band (`delta_g_max`):** Analytically normalizes layer operator gain by $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ in $O(\text{params})$ arithmetic, and measures realized generator output displacement $\delta_G = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ on a detached eval-mode probe to clamp `548×` overpowered generators before they overshoot into mode collapse.
+- **Closed-Form Operator Scaling + One-Shot Realized Generator Pushforward Trust Controller (`delta_g_max` + `_ema_func_gain`):** Normalizes layer operator gain in $O(\text{params})$ arithmetic via Marchenko–Pastur bulk edge $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ on elementwise-Adam presets (`t4_nash`, `minimal`) or Modular-Duality/polar-matched RMS-to-RMS factors ($\sqrt{\max(1, r/c)}$, $\sqrt{r}$, $\sqrt{\text{out}/\text{in}}\sqrt{g/k_{\text{eff}}}\,m_g^{1/4}$) in `unified_v11`/`v12`, and applies a one-shot linear-secant rescaling (`_band_scale`) plus closed-loop stiffness EMA (`_ema_func_gain`) driven by detached eval-mode generator displacement $\delta_G = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ to clamp `548×` overpowered generators (note: strict trust enforcement on saturating or amplified nonlinear steps remains an open upgrade requiring iterative backtracking and stratified probing).
```

#### (2) Proposed docstring corrections for `_band_scale` (NOT applied here)

```diff
--- a/particlegan/_t4_nash/optim.py
+++ b/particlegan/_t4_nash/optim.py
@@ -244,7 +244,7 @@
     alpha_max: float = 12.0,
     alpha_min: float = 1e-4,
 ) -> float:
-    """Smooth C^1 impedance-matching gain keeping realized function-space step inside ``[delta_min, delta_max]``."""
+    """Piecewise-C^0 one-shot linear-secant gain targeting ``[delta_min * (1 - 1/alpha_max), delta_max]`` (unverified on nonlinear rays)."""

--- a/particlegan/_unified_v4/optimizer.py
+++ b/particlegan/_unified_v4/optimizer.py
@@ -45,7 +45,7 @@
     alpha_max: float = 48.0,
     alpha_min: float = 1e-4,
 ) -> float:
-    """Smooth impedance-matching gain keeping realized function-space step inside ``[delta_min, delta_max]``."""
+    """Piecewise-C^0 one-shot linear-secant gain targeting ``[delta_min * (1 - 1/alpha_max), delta_max]`` (unverified on nonlinear rays)."""
```
