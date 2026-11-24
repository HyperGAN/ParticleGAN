# P6 Adversarial Mathematical, Autodiff & Computational Audit: Law 2 (Forward-Mode Critic Regularization, Mixed Derivatives, Micro-Secants & Scaling Regimes)

> **Audit Scope & Governance:** Adversarial mathematical, automatic-differentiation, statistical, and GPU-scaling audit of Law 2 (*Critic Potential Geometry*) on integration branch `research/four-laws` (audited on branch `research/fl-p6-law2`). All experiments execute through the workspace `particlegan` package under protocol seed `0` and `deterministic_orthogonal_` initialization on `NVIDIA RTX A6000` (`cuda:0`, PyTorch `2.15.0+cu132`, hardware NVML mJ counters via `libnvidia-ml.so.1`) and CPU. Existing Forge receipts and `docs/four-laws.md` are left strictly untouched; exact proposed revisions are provided in [Section 6](#6-disciplined-reformulation-of-law-2--exact-proposed-diffs).
>
> **Reproducibility:**
> - Experiment runner: [`scripts/four_laws_audit/p6_law2_autodiff.py`](../../scripts/four_laws_audit/p6_law2_autodiff.py)
> - Automated proofs & semantic controls: [`tests/four_laws_audit/test_p6_law2_autodiff.py`](../../tests/four_laws_audit/test_p6_law2_autodiff.py) (`24/24` PASS, `0` lint additions)

---

## Executive Verdict

Law 2 in [`docs/four-laws.md:40-57,113,133-141,182`](../../docs/four-laws.md#L40-L57) and [`reports/forge/studies/t2-efficient-critic.md:7-62,71-81`](../forge/studies/t2-efficient-critic.md#L7-L62) delivers a **genuine, high-leverage systems engineering contribution** — eliminating reverse-over-reverse (`create_graph=True`) double-backpropagation convolution kernels on high-resolution critics (`3.64×` faster penalty, `-33.1%` peak VRAM, and `-66.4%` GPU energy at `256×256` BF16 `DCGANCritic`) — **yet wraps that engineering win in four serious mathematical conflations, one major documentation-vs-code drift, and two severe numerical/statistical failure modes**:

1. **FALSIFIED — "`fwAD.dual_level()` + `.backward()` is purely first-order / zero higher-order differentiation":**
   - Differentiating any input-gradient penalty $\Phi\big((\nabla_x D_\theta(x)\cdot u)^2\big)$ with respect to parameters $\theta$ computes the **exact mixed second-order derivative** $\nabla_\theta \tau = \left(\frac{\partial^2 D_\theta}{\partial \theta\,\partial x}\right) u$ via **reverse-over-forward (RoF)** automatic differentiation — **NOT** first-order differentiation!
   - On smooth networks (`Tanh`, `SiLU`), we prove analytically and verify in FP64 (`rel_l2 = 1.94e-16–2.55e-16`, `cos = 1.000000000000`) that `fwAD` + `.backward()` literally evaluates the activation second derivative $\sigma''(z)$ (`||dL/db(sigma'')|| = 0.0415–0.1031`; zeroing $\sigma''$ breaks agreement) and matches reverse-over-reverse (RoR) `torch.autograd.grad(..., create_graph=True)` to machine precision (`2.12e-16–8.82e-16`).
   - On piecewise-linear networks (`ReLU`, `LeakyReLU`), even though the pure spatial Hessian vanishes ($\frac{\partial^2 D}{\partial x^2} = 0$ a.e.), the **mixed Hessian does NOT vanish** (`||H_{\theta x} \nabla_x D|| = 21.14–30.93` at active slope `2.0`), and canonical `fwAD` (`d <= 4`) matches RoR identically (`rel_l2 = 3.94e-16–5.62e-16` FP64, `2.40e-7–3.16e-7` FP32).
   - What `fwAD` genuinely eliminates is **reverse-over-reverse (`create_graph=True`) graph retention and double-backward kernels**, not second-order mixed derivatives.

2. **CRITICAL DOCUMENTATION-VS-CODE DRIFT & DEVASTATING MICRO-SECANT FALSIFICATION:**
   - [`docs/four-laws.md:113,136`](../../docs/four-laws.md#L113) claims `unified_v11` (and by inheritance `DEFAULT_RECIPE = "unified_v12"`) uses live-endpoint reuse + separable BF16 micro-secants + transport-chord/cyclic-roll secants on Conv2d critics. **That is false in executed code:** [`particlegan/_unified_v11/penalty.py:404-490,631-695`](../../particlegan/_unified_v11/penalty.py#L404-L490) completely replaced image micro-secants and macro chords with **`_eval_image_fwad_sq`** — a 1-probe interleaved `fwAD.dual_level()` pass (requiring `BatchNorm2d.eval()` and custom [`_bn_dual_bf16_hook`](../../particlegan/_unified_v11/penalty.py#L97-L107) to prevent a PyTorch BF16 autocast dual-tensor crash).
   - Testing the actual BF16 micro-secants that *do* remain in `t2_secant`, `t4_nash_v3`, `unified`/`unified_v4` (`img_delta = 0.02`), and `minimal` (`secant_eps = 1e-5`) on `DCGANCritic` (`32^2`–`256^2`) exposes **catastrophic self-contradiction against Law 2's own obstructions**:
     - At `256×256` BF16, `t2_secant`'s per-pixel step is $0.02/\sqrt{196{,}608} = 4.51\times 10^{-5} \ll \mathrm{ULP}_{\text{BF16}}(0.5) = 3.91\times 10^{-3}$: **`95.2%` of perturbed pixels quantize identically to zero**, **`56%` of samples produce zero logit difference**, **`11,881` LeakyReLU units flip sign**, and the resulting parameter gradient has **`cos = -0.0017` (pure noise)** with **`87,478×` norm explosion** relative to exact directional `fwAD` (even in FP32, `1,734` LeakyReLU flips collapse `cos = -0.0003` and inflate norm `2,617×`).
     - In `minimal` (`eps = 1e-5`), per-pixel step $2.25\times 10^{-8}$ collapses **`100%` of pixels with $|x| \ge 0.5$ even in FP32** (`50.0%` overall) and **`99.99%` in BF16**, giving `100%` dead zero penalty at `r32` BF16 and **`6.74 × 10^{10}×` norm spikes (`cos = +0.0012`)** at `r256` BF16.
     - By contrast, `unified_v11`/`v12`'s actual `_eval_image_fwad_sq` in BF16 incurs **`0` LeakyReLU flips** and achieves **`grad_cos = +0.9869 to +0.9964` (`norm_ratio = 0.976–1.065`)** across `32^2`–`256^2`!

3. **PROVEN OPPOSITE-SIGNED JENSEN'S INEQUALITY BIAS & SEPARABLE RADEMACHER VARIANCE BLOWUP:**
   - Even with exact, kink-free `fwAD` directional derivatives ($\mathbb{E}[\hat{s}] = \|\nabla_x D\|_2^2$ unbiased inside the square root), composing a stochastic $K$-probe estimator $\hat{s}$ with **either** nonlinear slope penalty used in ParticleGAN introduces **large, opposite-signed Jensen bias**:
     - **Strictly concave Born–Infeld $\phi_\beta(s) = (\sqrt{1+\beta s}-1)/\beta$** (`minimal`, [`_minimal/geometry.py:138`](../../particlegan/_minimal/geometry.py#L138)): **strictly negatively biased** in BOTH loss (`-14.8%` to `-21.8%` at $K=1$) AND expected parameter gradient (`-20.4%` to `-23.3%`, asymptotic factor $\sqrt{2/\pi} \approx 0.7979$ for dense probes and $2/\pi \approx 0.6366$ for rank-1 separable probes), PLUS Rademacher **directional rotation** of the expected gradient (`13.30°` by exact $2^4$ enumeration on anisotropic $g \in \mathbb{R}^4$, suppressing minor gradient coordinates by `3.38×`).
     - **Convex-on-cap EBI squared softplus $h_{\beta,\kappa}(s) = [\operatorname{softplus}_\beta(\sqrt{s}-\kappa)]^2$** (`unified_v11`/`v12`, `t2_secant`, `unified`): **strictly positively biased** on the active cap with **severe false triggering below $\kappa = 1$** — at true slope $r = 0.8 < \kappa$, 1-probe Rademacher fires $\sqrt{\hat{s}} > 1$ with probability **`21.1%`**, inflating expected penalty **`1,015×`** (`0.05315` vs. `0.00005`) and expected gradient **`275.8×`** (and `45.9×` penalty / `11.1×` gradient at $r = 1.0$), all while the reported `mean_grad_norm` $\mathbb{E}[\sqrt{\hat{s}}] \approx 0.798 r$ is **`-20.2%` negatively biased**.
   - **Separable Rademacher Variance Theorem:** We derive the exact closed-form variance of 2-way separable probes (`_separable_image_unit_perturbation`, used for `d > 8192` in `unified_v11`/`v12` and all images in `t2_secant`) and prove it scales as $(2 + 6 / r_{\text{eff},2}(G))\|G\|_F^4$ — a **`4×` asymptotic variance blowup** on rank-1 stripe/edge gradients (`7.41` vs. `1.96` on $3\times 32\times 32$), while `minimal`'s 3-way Kronecker probe blows up **`8.3×`** (`15.88` vs. `1.91`) and shares the **exact same spatial pattern across the entire minibatch** (`|cos(u_i, u_j)| = 1.000000`).

4. **COMPUTATIONAL SCALING REGIMES — WHERE `fwAD` WINS VS. WHERE IT LOSES TO EXACT RoR:**
   - **`fwAD` LOSES on ALL small/medium GPU MLPs (`B=256, 4×256`, FP32):** Exact RoR (`1.32–1.59 ms`, constant in $d$) **beats BOTH 1-probe `fwAD` (`2.38–2.78 ms`, `1.66×–2.05×` slower) AND exact-basis `fwAD` (`2.80 ms` at $d=4$ exploding to `164.89 ms` / `4,929 MiB` = `125.4×` slower / `104.6×` VRAM at $d=1024$)** due to dual-mode kernel-launch overhead and $O(d)$ basis scaling!
   - **`fwAD` LOSES on low-resolution Conv2d (`32×32`, `64×64`), `ResNet+GroupNorm` (`64×64`), multi-probe `K >= 4` at `256×256`, Attention (`SDPA`), and `EmulatedLinear`:**
     - Matched-sample 1-probe `fwAD` (`fwAD1b`) is **`1.6×–1.9×` slower than `RoR_eval` at `32^2`–`64^2`**, and **`1.32×–1.78×` slower with `2.09×–2.62×` MORE VRAM** on `ResNet_64_Skip_GroupNorm` (`207` vs. `79 MiB` BF16).
     - At `256×256` BF16, `K=4` probe `fwAD` (`49.63 ms`) is ALREADY **slower than exact full-gradient `RoR_train` (`45.15 ms`) and `1.43×` slower than `RoR_eval` (`34.62 ms`)** — proving `fwAD`'s speed win exists **strictly in the $K \in \{1, 2\}$ high-variance stochastic regime**!
     - **`fwAD` CRASHES** on `DCGANCritic` BF16 without `_bn_dual_bf16_hook`, on `EmulatedLinear` (`mxfp8`/`nvfp4`, where RoR succeeds), on default Flash/MemEff SDPA (where RoR also fails and only micro-secant runs), and **even on explicit `SDPBackend.MATH` SDPA** (PyTorch 2.15 in-place `Exp` error during dual-mode `.backward()`, where RoR succeeds cleanly!).
   - **`fwAD` GENUINELY WINS BIG on compute-bound high-resolution Conv2d (`128×128`–`256×256`, `K=1` interleaved):** At `256×256` BF16 `DCGANCritic`, `fwAD1i` cuts penalty time **`3.64×`** (`12.40` vs. `45.15 ms`), peak VRAM **`-33.1%`** (`576.5` vs. `861.7 MiB`), and hardware GPU energy **`-66.4%`** (`3,799` vs. `11,301 mJ/iter`) while maintaining `0.987` directional gradient cosine — **a major, verified engineering win once accurately scoped**.

---

## 1. Exhaustive Codebase & Claim Audit

### 1.1 Ground-Truth Inventory of Every Critic Penalty in `particlegan/`

Every critic penalty implementation under `particlegan/` (plus non-critic `fwAD` sites in [`_unified_v11/prior.py:983-991`](../../particlegan/_unified_v11/prior.py#L983-L991) and [`_unified_v4/prior.py:311-319`](../../particlegan/_unified_v4/prior.py#L311-L319)) was inspected in source and classified by **true mathematical derivative order**, **autodiff graph mode**, and **outer penalty curvature**:

| Formulation / Class | Source File & Exact Lines | Input Regime | True Mathematical Derivative Order | Autodiff Graph Mode | Outer Slope Penalty $\Phi(s)$ & Curvature in $s$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`UnifiedV11CriticPenalty`** (`unified_v11` & default `unified_v12`!) | [`_unified_v11/penalty.py:218-399,404-490,510-733`](../../particlegan/_unified_v11/penalty.py#L404-L490), [`_unified/penalty.py:67-211,669-725`](../../particlegan/_unified/penalty.py#L67-L211) | `1<=d<=4` (non-quantized) | **Exact 2nd-order mixed** $\frac{\partial^2 D}{\partial \theta \partial x}\nabla_x D$ | **RoF** (`fwAD.dual_level` + `.backward()`, canonical $I_d$) | EBI $[\operatorname{sp}_\beta(\sqrt{s}-\kappa)]^2 + \lambda_{\text{fl}} s$ (**convex on cap**) |
| **`UnifiedV11CriticPenalty`** | [`_unified/penalty.py:696-725`](../../particlegan/_unified/penalty.py#L696-L725), [`_t2_critic/penalty.py:113-143,200-224`](../../particlegan/_t2_critic/penalty.py#L200-L224) | `5<=d<=8` | **2nd-order mixed** (spatial neighbor-mixed Householder $Q=H_2 H_1$) | **RoF** (`fwAD.dual_level` + `.backward()`, 1 tangent/sample) | EBI (**convex on cap**) |
| **`UnifiedV11CriticPenalty`** | [`_unified_v11/penalty.py:54-82,218-338`](../../particlegan/_unified_v11/penalty.py#L54-L82) | `d>8` 2D (e.g. 170D simplex) | **2nd-order mixed** (Murmur3 Rademacher + $k=32$ Gaussian pooling) | **RoF** (`fwAD.dual_level` + `.backward()`, 1 tangent/sample) | EBI (**convex on cap**) |
| **`UnifiedV11CriticPenalty`** | [`_unified_v11/penalty.py:85-107,404-490,631-695`](../../particlegan/_unified_v11/penalty.py#L404-L490) | **All 3D/4D images (`8^2..256^2`)** | **2nd-order mixed** (1-probe interleaved `fwAD`, `BN.eval()` + `_bn_dual_bf16_hook`; dense `d<=8192`, 2-way sep `d>8192`) | **RoF** (`fwAD.dual_level` + `.backward()`); **ZERO micro-secants or chords!** | EBI (**convex on cap**) + $\frac{5\times 10^{-4}}{d} s$ |
| **`UnifiedV11CriticPenalty`** | [`_unified/penalty.py:57-64,94-97,178-183`](../../particlegan/_unified/penalty.py#L57-L64), [`_unified_v11/penalty.py:293-298,479-485`](../../particlegan/_unified_v11/penalty.py#L479-L485) | Quantized (`EmulatedLinear` NVFP4/MXFP8) or `fwAD` error | **Finite-diff approx. to 2nd-order mixed** (central diff $\delta = \max(0.06, 0.04\,\text{scale})$) | **1st-order reverse** on 2 primal calls | EBI (**convex on cap**) |
| **`T2SecantCriticPenalty`** (`t2_secant`, `t4_arch`) & **`NashV3CriticPenalty`** (`t4_nash_v3`) | [`_t2_critic/penalty.py:745-941`](../../particlegan/_t2_critic/penalty.py#L745-L941), [`_nash_v3/penalty.py:414-603`](../../particlegan/_nash_v3/penalty.py#L414-L603) | 2D vectors / simplex | **Exact** (`d<=4`) or **spatial/Hutchinson 2nd-order mixed** (`d>4`); central FD `delta=1e-3` fallback | **RoF** (`fwAD.dual_level` + `.backward()`) | EBI (**convex on cap**) or ReLU hinge (`d>=170`) |
| **`T2SecantCriticPenalty`** & **`NashV3CriticPenalty`** | [`_t2_critic/penalty.py:745-878`](../../particlegan/_t2_critic/penalty.py#L745-L878), [`_nash_v3/penalty.py:414-545`](../../particlegan/_nash_v3/penalty.py#L414-L545) | **Images (`ndim >= 3`)** | **FD approx. to 2nd-order mixed** (live-reuse forward micro-secant `img_delta=0.02`, 2-way sep) + **macro chord/roll lower bounds** | **1st-order reverse** (1 probe fwd + live reuse) | EBI on `amax(micro, s_chord, s_ff, s_rr)` (**convex on cap**) |
| **`UnifiedCriticPenalty`** (`unified`, `unified_v4`, `cond_fiber`) | [`_unified/penalty.py:214-255,454-645`](../../particlegan/_unified/penalty.py#L454-L645), [`_unified_v4/penalty.py:34-40`](../../particlegan/_unified_v4/penalty.py#L34-L40), [`_cond_fiber/penalty.py:10-31`](../../particlegan/_cond_fiber/penalty.py#L10-L31) | Images `d<=1024` / `d>1024` | **FD approx. to 2nd-order mixed**: $K=2$ Gram–Schmidt ortho micro-secants (`0.015`) + 4-NN (`d<=1024`); 1 BF16 sep micro-secant (`0.02`) + chord/roll `amax` (`d>1024`) | **1st-order reverse** (live-endpoint reuse) | EBI (**convex on cap**) |
| **`MinimalCriticPenalty`** (`minimal`) | [`_minimal/geometry.py:31-66,115-143`](../../particlegan/_minimal/geometry.py#L31-L66) | `2D d<=8` vs. `d>8` / images | `d<=8`: **exact** (`I_d`, `d<=4`; 1-Householder `d<=8`) **2nd-order mixed** (`fwAD`); `d>8`/images: **FD approx.** (`eps=1e-5`, 3-way Kronecker for `d>1024`) + chord | `d<=8`: **RoF**; `d>8`/images: **1st-order reverse** (live reuse) | **Strictly concave Born–Infeld** $\frac{\sqrt{1+\beta s}-1}{\beta}$ (`beta=8`) + `0.08 * chord^2` |
| **`R4AutogradCriticPenalty`**, **`T1Coupling`**, **`RHBI`**, **`RGDT`**, **`EMPF`**, **`T4Nash`**, **`BISPF`**, **`K3P`/`KA2`/`BCap`** | [`_r4_ablation/penalty.py:57`](../../particlegan/_r4_ablation/penalty.py#L57), [`_t1_coupling/penalty.py:197,242`](../../particlegan/_t1_coupling/penalty.py#L197), [`_rhbi/penalty.py:200`](../../particlegan/_rhbi/penalty.py#L200), [`_t3_dynamics/penalty.py:224`](../../particlegan/_t3_dynamics/penalty.py#L224), [`_empf/penalty.py:383`](../../particlegan/_empf/penalty.py#L383), [`_t4_nash/penalty.py:350`](../../particlegan/_t4_nash/penalty.py#L350), [`_bispf/penalty.py:252`](../../particlegan/_bispf/penalty.py#L252), [`_bispf/engine.py:163`](../../particlegan/_bispf/engine.py#L163), [`grad_regularizers.py:291,330`](../../particlegan/grad_regularizers.py#L291), [`ka2.py:322`](../../particlegan/ka2.py#L322) | All domains | **Exact full 2nd-order mixed** $\frac{\partial^2 D}{\partial \theta \partial x}\nabla_x D$ | **RoR** (`torch.autograd.grad(..., create_graph=True)` + `.backward()`) | EBI (**convex on cap**) or ReLU hinge (`b_cap`) or linear $s$ (`a_r1r2`) |

*Additional code-level details caught during source audit:*
- **Output-channel reduction discrepancy for multi-logit critics ($K > 1$):** [`_t2_critic/penalty.py:271,289,310,343`](../../particlegan/_t2_critic/penalty.py#L271) and [`_unified_v11/penalty.py:455`](../../particlegan/_unified_v11/penalty.py#L455) average squared JVP over output channels (`.square().mean(dim=-1)`), whereas [`_unified/penalty.py:91,111,133,174`](../../particlegan/_unified/penalty.py#L91) and [`_unified_v11/penalty.py:289,318`](../../particlegan/_unified_v11/penalty.py#L289) sum over output channels (`.square().sum(dim=-1)`), while RoR [`_score_scalar`](../../particlegan/grad_regularizers.py#L15-L24) differentiates the mean logit $\|\nabla_x(\frac{1}{K}\sum_k D_k)\|_2^2$. They coincide identically for scalar critics ($K=1$) and diverge for multi-head critics ($K > 1$).
- **Edge-case index bug in `_prepare_spatial_tangent_plan` (`pool_k = 1`):** [`_t2_critic/penalty.py:240`](../../particlegan/_t2_critic/penalty.py#L240) indexes `knn_d2[:, max(1, k_eff // 2)]`, raising `IndexError` if `pool_k = 1` or `b = 1` (guarded for `b < 2` at `L175`, unguarded for explicit `pool_k = 1`).
- **`EmulatedLinear` has no `jvp`:** [`particlegan/low_precision.py:85-122`](../../particlegan/low_precision.py#L85-L122) defines `forward` and `backward` (with straight-through quantizer proxy `_quantized_proxy`) on `torch.autograd.Function` `_Linear`, omitting `jvp`. Thus `fwAD` always raises `NotImplementedError` and falls back to central finite differences (`fd_delta = 1e-3` in `t2_secant` — too small for NVFP4 bins — vs. `_adaptive_fd_delta >= 0.06` in `unified`/`unified_v11`).
- **`unified_v11`/`v12` changes the BatchNorm operator under `fwAD`:** [`_unified_v11/penalty.py:433-439`](../../particlegan/_unified_v11/penalty.py#L433-L439) switches every `_BatchNorm` layer to `.eval()` during `_eval_image_fwad_sq`, differentiating the frozen running-statistics affine map rather than the training-mode batch-coupled BN operator differentiated by `empf`/`bispf`/`r4_autograd_critic` (explicitly noted in [`experiments/forge/scaling_benchmark.py:127-129`](../../experiments/forge/scaling_benchmark.py#L127-L129)).

### 1.2 Summary Table of Audited Claims (10-Category Taxonomy)

| ID | Source File & Exact Lines | Verbatim Claim (Key Excerpt) | Primary Taxonomy Label |
| :--- | :--- | :--- | :--- |
| **C1** | [`docs/four-laws.md:51,113,135,141,182`](../../docs/four-laws.md#L51), [`t2-efficient-critic.md:7,51,73`](../forge/studies/t2-efficient-critic.md#L7) | *"purely 1st-order Born-Infeld critic formulation"* / *"First-order `fwAD` dual-tangent reconstruction ... single 1st-order `.backward()`"* / Autograd Order: *"1st-order (`0` `autograd.grad`)"* | **Falsified claim** |
| **C2** | [`docs/four-laws.md:113,136`](../../docs/four-laws.md#L113) | Lists `unified_v11` (and default `unified_v12`) under *"Live-Endpoint Reuse + Separable BF16 Micro-Secants (Convolutional Image Critics)"* with *"free transport-chord and cyclic-roll secants"* | **Falsified claim** |
| **C3** | [`docs/four-laws.md:135`](../../docs/four-laws.md#L135), [`t2-efficient-critic.md:51`](../forge/studies/t2-efficient-critic.md#L51) | `fwAD.dual_level` evaluates $(D_\theta(x), \nabla_x D_\theta(x)\cdot u)$ at the exact primal activation mask with zero ReLU/LeakyReLU $O(1/\delta)$ kink impulses | **Proven mathematical result** |
| **C4** | [`docs/four-laws.md:46,136`](../../docs/four-laws.md#L46), [`t2-efficient-critic.md:12,18,43-47,61`](../forge/studies/t2-efficient-critic.md#L18) | Separable BF16 micro-secant (`delta = 0.02`, `t2_secant`/`unified`) & `minimal` micro-secant (`eps = 1e-5`) provide faithful slope regularity compatible with low-precision BF16 while avoiding kink impulses | **Falsified claim** |
| **C5** | [`t2-efficient-critic.md:10-11,41-42`](../forge/studies/t2-efficient-critic.md#L41-L42), [`docs/four-laws.md:44-45`](../../docs/four-laws.md#L44-L45) | Naive macro chords suffer symmetric-flank cancellation ($S = 0$ as $\|\nabla_x D\| \to \infty$) and unscaled single probes suffer $\Theta(\sqrt{d})$ high-D projection attenuation | **Proven mathematical result** |
| **C6** | [`t2-efficient-critic.md:16,55-60`](../forge/studies/t2-efficient-critic.md#L55-L60), [`_t2_critic/penalty.py:15-17`](../../particlegan/_t2_critic/penalty.py#L15-L17) | Spatial-frame Householder reconstruction (`5 <= d <= 8`) has *"exact `0` directional variance"* and $k=32$ local Gaussian Hutchinson pooling (`d > 8`) yields accurate low-bias norm/gradient estimates | **Falsified claim** *(exact only for constant-gradient linear critics; nonlinear critics incur spatial-mixing bias and opposite-signed Jensen bias)* |
| **C7** | [`_t2_critic/penalty.py:145-159`](../../particlegan/_t2_critic/penalty.py#L145-L159), [`_unified_v11/penalty.py:85-94`](../../particlegan/_unified_v11/penalty.py#L85-L94), [`_minimal/geometry.py:54-60`](../../particlegan/_minimal/geometry.py#L54-L60) | Separable Rademacher perturbations (`O(B(CH+W))` 2-way in `t2_secant`/`unified_v11` and 3-way in `minimal`) match dense Rademacher variance $2\|\nabla_x D\|_2^4$ | **Falsified claim** |
| **C8** | [`docs/four-laws.md:43,139-141,182`](../../docs/four-laws.md#L43), [`t2-efficient-critic.md:5,20,31`](../forge/studies/t2-efficient-critic.md#L20) | Eliminating `create_graph=True` RoR cuts high-resolution (`128^2`–`256^2`) Conv2d critic penalty latency, VRAM, and hardware GPU energy dramatically | **Empirically supported causal mechanism** |
| **C9** | [`docs/four-laws.md:43,51,113`](../../docs/four-laws.md#L43), [`t2-efficient-critic.md:29-31`](../forge/studies/t2-efficient-critic.md#L29-L31) | Implicit claim that `fwAD` universally dominates RoR `autograd.grad(create_graph=True)` across architectures, dimensions, probe counts, and precision regimes | **Falsified claim** |
| **C10** | [`docs/four-laws.md:141,182`](../../docs/four-laws.md#L141), [`r4-occam-ablation-razor.md:20,130-131`](../forge/studies/r4-occam-ablation-razor.md#L130-L131) | 1-probe `fwAD`/secant EBI achieves higher Forge pass rate than exact RoR EBI (Arm 7 `-fwad-critic`: `14/16` vs. `11/15` on `r4-occam-ablation-v1`) | **Empirical correlation** *(confounded by Jensen bias, spatial smoothing, BN mode switch, and unequal tuning on `fwAD`)* |

---

## 2. Formal Mathematical Analysis & Proofs

### 2.1 Theorem 1 — Exact Derivative Order of `fwAD.dual_level()` + `.backward()` (Reverse-over-Forward Mixed Second Derivative)

**Theorem 1 (RoF = Exact Mixed Second-Order Derivative):**
Let $f: \mathbb{R}^P \times \mathbb{R}^d \to \mathbb{R}$, $(\theta, x) \mapsto D_\theta(x)$ be $C^2$ in a neighborhood of $(\theta, x)$, let $u \in \mathbb{R}^d$ be independent of $\theta$, and let $\Phi: \mathbb{R}_{\ge 0} \to \mathbb{R}$ be $C^1$.
1. Evaluating $f(\theta, x + \epsilon u)$ in dual arithmetic ($\epsilon^2 = 0$, `fwAD.dual_level()`) returns the primal value $D_\theta(x)$ and the directional spatial derivative
   $$\tau(\theta, x; u) = \left.\frac{d}{d\epsilon}\right|_{\epsilon=0} D_\theta(x + \epsilon u) = \langle \nabla_x D_\theta(x), u \rangle.$$
2. Differentiating $\mathcal{R}(\theta) = \Phi\big(\tau(\theta, x; u)^2\big)$ with respect to $\theta$ via `.backward()` executes **reverse-over-forward (RoF)** automatic differentiation and returns the **exact mixed second-order derivative contraction**:
   $$\nabla_\theta \mathcal{R}(\theta) = 2\,\Phi'\!\big(\tau^2\big)\,\tau(\theta, x; u)\,\nabla_\theta\langle \nabla_x D_\theta(x), u \rangle = 2\,\Phi'\!\big(\tau^2\big)\,\tau(\theta, x; u)\,\underbrace{\left(\frac{\partial^2 D_\theta(x)}{\partial \theta\,\partial x}\right)}_{H_{\theta x} \in \mathbb{R}^{P \times d}} u.$$
3. Summing over any orthonormal basis $\{q_m\}_{m=1}^d \subset \mathbb{R}^d$ yields $S(\theta, x) = \sum_{m=1}^d \tau(\theta, x; q_m)^2 = \|\nabla_x D_\theta(x)\|_2^2$ and
   $$\nabla_\theta \Phi\big(S(\theta, x)\big) = 2\,\Phi'\!\big(\|\nabla_x D_\theta(x)\|_2^2\big)\,\frac{\partial^2 D_\theta(x)}{\partial \theta\,\partial x}\,\nabla_x D_\theta(x),$$
   which is **algebraically identical** to reverse-over-reverse (RoR) differentiation via `torch.autograd.grad(D(x), x, create_graph=True)` followed by `.backward()`.
4. **Explicit 2-Layer Smooth Network ($\sigma''$ Witness):** For $D_\theta(x) = v^\top \sigma(W x + b)$ with $z = W x + b$, $\tau = \sum_h v_h \sigma'(z_h) (W u)_h$, hence:
   $$\frac{\partial \tau}{\partial v_h} = \sigma'(z_h)(W u)_h, \qquad \frac{\partial \tau}{\partial b_h} = v_h\,\mathbf{\sigma''(z_h)}\,(W u)_h, \qquad \frac{\partial \tau}{\partial W_{hj}} = v_h\,\mathbf{\sigma''(z_h)}\,(W u)_h x_j + v_h \sigma'(z_h) u_j.$$
   The bias gradient $\nabla_b \mathcal{R}$ is **purely proportional to $\sigma''(z)$**, proving `fwAD` + `.backward()` explicitly evaluates second-order activation derivatives!
5. **Piecewise-Linear (`ReLU`/`LeakyReLU`) Critics:** Off the measure-zero union of activation hyperplanes $\mathcal{H} = \bigcup_j \{x : z_j(\theta, x) = 0\}$, the activation mask $M(\theta, x)$ is locally constant, so $D_\theta(x) = a(\theta, M)^\top x + c(\theta, M)$. While the pure spatial Hessian vanishes ($\frac{\partial^2 D}{\partial x^2} = 0$ a.e.), the **mixed second-order derivative does NOT vanish**:
   $$\frac{\partial^2 D_\theta(x)}{\partial \theta\,\partial x} = \frac{\partial a(\theta, M)}{\partial \theta} \neq 0,$$
   because $a(\theta, M)$ is a multilinear product of layer weight matrices. Both `fwAD` + `.backward()` (RoF) and `autograd.grad(create_graph=True)` + `.backward()` (RoR) evaluate the exact same primal mask $M(\theta, x)$ with zero distributional kink impulse and agree to machine precision.

### 2.2 Theorem 2 — Finite-Difference Micro-Secants, $O(1/\delta)$ Kink Impulses & BF16/FP32 Quantization Collapse

**Theorem 2 (Micro-Secant Derivative Order, Kink Impulses & Precision Underflow):**
Let $\hat{\tau}_{\text{fwd}}(\theta, x; u, \delta) = \frac{D_\theta(x + \delta u) - D_\theta(x)}{\delta}$ and $\hat{\tau}_{\text{cen}}(\theta, x; u, \delta) = \frac{D_\theta(x + \delta u) - D_\theta(x - \delta u)}{2\delta}$ ($\|u\|_2 = 1$).
1. **True Mathematical Order:** Differentiating $\hat{\tau}_{\text{fwd}}$ or $\hat{\tau}_{\text{cen}}$ once w.r.t. $\theta$ computes a **finite-difference approximation to the mixed second derivative**:
   $$\nabla_\theta \hat{\tau}_{\text{fwd}} = \frac{\nabla_\theta D_\theta(x + \delta u) - \nabla_\theta D_\theta(x)}{\delta} = \frac{\partial^2 D_\theta(x)}{\partial \theta\,\partial x} u + O(\delta) \quad (\text{smooth } D_\theta),$$
   using **genuine first-order reverse-mode AD** on two primal evaluations.
2. **Piecewise-Linear $O(1/\delta)$ Impulse for BOTH Forward and Central Differences:** If the segment $[x, x + \delta u]$ (forward FD) or $[x - \delta u, x + \delta u]$ (central FD) straddles a ReLU/LeakyReLU hyperplane $H_j = \{w_j^\top h + b_j = 0\}$ at signed distance $\eta < \delta$, then the piecewise-affine coefficients jump ($\Delta a_j \neq 0, \Delta c_j \neq 0$ with $\Delta a_j^\top x_H + \Delta c_j = 0$ on $H_j$), so $D_\theta$ is continuous across $H_j$ yet $\nabla_\theta D_\theta$ jumps by $O(1)$ across $H_j$. Dividing by $\delta$ injects an **unbounded $O(1/\delta)$ parameter-gradient impulse** into BOTH $\nabla_\theta \hat{\tau}_{\text{fwd}}$ and $\nabla_\theta \hat{\tau}_{\text{cen}}$!
3. **High-Dimensional Per-Pixel Quantization Collapse:** For a unit Rademacher vector $u \in \{\pm d^{-1/2}\}^d$, each pixel perturbation has magnitude $\Delta x_{\text{px}} = \delta / \sqrt{d}$. In floating-point format with $p$ significand bits (`BF16`: $p=7$, $\mathrm{ULP}(x) = 2^{\lfloor \log_2 |x| \rfloor - 7}$; `FP32`: $p=23$, $\mathrm{ULP}(x) = 2^{\lfloor \log_2 |x| \rfloor - 23}$), whenever $\Delta x_{\text{px}} < \tfrac{1}{2}\mathrm{ULP}(x)$:
   - A pixel rounds back identically to $\mathrm{fl}(x)$ unless it lies within $\Delta x_{\text{px}}$ of a rounding midpoint (survival fraction $\approx 2\Delta x_{\text{px}}/\mathrm{ULP}$), in which case it jumps by a full $\pm \mathrm{ULP} \gg \Delta x_{\text{px}}$!
   - For **`t2_secant`/`unified` (`delta = 0.02`, `256×256×3`, $d = 196{,}608$)**: $\Delta x_{\text{px}} = 4.51 \times 10^{-5}$ vs. $\tfrac{1}{2}\mathrm{ULP}_{\text{BF16}}(0.5) = 1.95 \times 10^{-3}$ (**$43\times$ below half-ULP**). Inside `torch.autocast("cuda", dtype=torch.bfloat16)`, **`95.2%` of pixels vanish** while the `4.8%` near rounding midpoints jump by $87\times$ their nominal perturbation!
   - For **`minimal` (`secant_eps = 1e-5`, `256×256×3`)**: $\Delta x_{\text{px}} = 2.25 \times 10^{-8} < \tfrac{1}{2}\mathrm{ULP}_{\text{FP32}}(0.5) = 2.98 \times 10^{-8}$, so **`100%` of pixels with $|x| \ge 0.5$ vanish identically even in FP32** and **`99.99%` vanish in BF16**!

### 2.3 Theorem 3 — Opposite-Signed Jensen's Inequality Bias for Nonlinear Critic Penalties

**Theorem 3 (Loss, Gradient-Magnitude & Directional Jensen Bias):**
Let $g = \nabla_x D_\theta(x) \in \mathbb{R}^d \setminus \{0\}$, $s = \|g\|_2^2 > 0$, and let $\hat{s} = \frac{d}{K}\sum_{k=1}^K (g^\top u_k)^2$ be an unbiased $K$-probe estimator ($\mathbb{E}[\hat{s}] = s$, $\operatorname{Var}(\hat{s}) > 0$ for non-degenerate $g$ and finite $K$).
1. **Strictly Concave Born–Infeld $\phi_\beta(s) = \frac{\sqrt{1 + \beta s} - 1}{\beta}$ (`minimal`, $\beta > 0$):**
   Since $\phi_\beta''(s) = -\frac{\beta}{4}(1+\beta s)^{-3/2} < 0$, **strict Jensen's inequality** gives strict negative loss bias:
   $$\mathbb{E}\big[\phi_\beta(\hat{s})\big] < \phi_\beta(s).$$
   Moreover, for parameter gradient $\nabla_\theta \phi_\beta(\hat{s}) = \frac{1}{\sqrt{1+\beta \hat{s}}} \frac{d}{K} \sum_k (g^\top u_k) H_{\theta x} u_k$, write $\hat{s} = s T$ with $\mathbb{E}[T] = 1$. In the isotropic / large-$d$ limit ($T \sim \chi_K^2 / K$), the expected parameter gradient satisfies
   $$\mathbb{E}\big[\nabla_\theta \phi_\beta(\hat{s})\big] = \mathbb{E}\!\left[\frac{T}{\sqrt{1 + \beta s T}}\right] H_{\theta x} g \quad \text{vs. exact} \quad \frac{1}{\sqrt{1 + \beta s}} H_{\theta x} g.$$
   Because $\psi(t) := t / \sqrt{1 + \beta s t}$ has $\psi''(t) = -\frac{\beta s (1 + \frac{1}{4}\beta s t)}{(1+\beta s t)^{5/2}} < 0$ for all $t \ge 0$, **$\psi$ is strictly concave** and therefore **BOTH the loss AND the expected parameter gradient are strictly attenuated** for every $s > 0$:
   $$\frac{\|\mathbb{E}[\nabla_\theta \phi_\beta(\hat{s})]\|_2}{\|\nabla_\theta \phi_\beta(s)\|_2} < 1, \qquad \lim_{\beta s \to \infty} \frac{\mathbb{E}[T/\sqrt{1+\beta s T}]}{1/\sqrt{1+\beta s}} = \mathbb{E}[\sqrt{T}] = \begin{cases} \sqrt{2/\pi} \approx 0.7979 & (K=1\text{ dense, }-20.2\%), \\ 2/\pi \approx 0.6366 & (K=1\text{ 2-way sep., rank-1, }-36.3\%). \end{cases}$$
   In addition, for **Rademacher** probes on anisotropic $g$ (unequal $|g_i|$), $\mathbb{E}[\nabla_g \phi_\beta(\hat{s})]$ is **not even collinear with $g$**: $\phi_\beta'((g^\top \epsilon)^2)$ down-weights sign alignments where minor entries reinforce the dominant entry, systematically rotating the expected gradient away from $g$ (verified at **`13.30°`** by exact $2^4$ enumeration on $g = [2.0, 0.4, 0.4, 0.4]$!).
2. **Convex-on-Cap EBI Squared Softplus $h_{\beta,\kappa}(s) = [\operatorname{softplus}_\beta(\sqrt{s} - \kappa)]^2$ (`unified_v11`/`v12`, `t2_secant`, `unified`):**
   Write $h(s) = g(\sqrt{s})$ with $g(r) = \operatorname{sp}_\beta(r-\kappa)^2$. Then $h''(s) = \frac{r g''(r) - g'(r)}{4 r^3}$. In the hinge limit $\beta \to \infty$, $h_{\infty,\kappa}(s) = (\sqrt{s} - \kappa)_+^2$ has $h''(s) = \frac{\kappa}{2} s^{-3/2} > 0$ for $\sqrt{s} > \kappa$ and $0$ for $\sqrt{s} < \kappa$ — **globally convex on $[0, \infty)$**! For finite $\beta \in [6, 12]$ and $\kappa = 1$, $g(r) \approx \beta^{-2} e^{2\beta(r-\kappa)}$ for $r \ll \kappa$, so $r g''(r) - g'(r) \approx 2\beta(2\beta r - 1)g(r)$ crosses zero at $r^\star \approx \frac{1}{2\beta}$ (exact numerical zeros of $h''_{\beta,1}$: $r^\star = 0.04167$ for $\beta = 12$ and $r^\star = 0.08359$ for $\beta = 6$), making $h_{\beta,\kappa}(s)$ strictly convex across the entire active cap and transition region ($r > r^\star \approx \frac{1}{2\beta}$). Consequently, Jensen's inequality reverses sign:
   $$\mathbb{E}\big[h_{\beta,\kappa}(\hat{s})\big] > h_{\beta,\kappa}(s) \qquad (\text{strictly positive over-penalization!}),$$
   causing severe **false cap triggering below $\kappa$** ($\mathbb{P}(\sqrt{\hat{s}} > \kappa) = \mathbb{P}(|Z| > \kappa / r) = 21.1\%$ at $r = 0.8 < \kappa = 1.0$, inflating expected penalty `1,015×` and expected gradient `275.8×`!), even while the logged `mean_grad_norm` $\mathbb{E}[\sqrt{\hat{s}}] < \sqrt{s}$ (`0.798 r`) is **strictly negatively biased**.

### 2.4 Theorem 4 — Exact Closed-Form Variance of Separable Rademacher & Householder Spatial-Frame Geometry

**Theorem 4A (Exact Separable vs. Dense Rademacher Variance):**
Let $G \in \mathbb{R}^{m \times n}$ ($m = CH, n = W$) be a spatial gradient reshaped as a matrix with $\|G\|_F^2 = \|g\|_2^2$.
- For **dense Rademacher** $\epsilon \in \{\pm 1\}^{mn}$, $Z_{\text{dense}} = (\epsilon^\top \operatorname{vec}(G))^2$ satisfies $\mathbb{E}[Z_{\text{dense}}] = \|G\|_F^2$ and
  $$\operatorname{Var}(Z_{\text{dense}}) = 2\big(\|G\|_F^4 - \|G\|_4^4\big).$$
- For **2-way separable Rademacher** $u = (a b^\top) / \sqrt{mn}$ with independent $a \in \{\pm 1\}^m, b \in \{\pm 1\}^n$ ([`_t2_critic/penalty.py:145-159`](../../particlegan/_t2_critic/penalty.py#L145-L159), [`_unified_v11/penalty.py:94`](../../particlegan/_unified_v11/penalty.py#L94)), $Z_{\text{sep2}} = (a^\top G b)^2$ is unbiased ($\mathbb{E}[Z_{\text{sep2}}] = \|G\|_F^2$) with **exact closed-form variance** (conditioning on $b$ and expanding $\mathbb{E}_b[(b^\top G^\top G b)^2]$):
  $$\operatorname{Var}(Z_{\text{sep2}}) = 3\!\left(\|G\|_F^4 + 2\|G^\top G\|_F^2 - 2\sum_{j=1}^n \|G_{:,j}\|_2^4\right) - 2\!\left(3\sum_{i=1}^m \|G_{i,:}\|_2^4 - 2\sum_{i,j} G_{ij}^4\right) - \|G\|_F^4.$$
  Dropping $O(1/m, 1/n)$ fourth-power corrections yields the leading spectral law in terms of the Rényi-2 effective rank $r_{\text{eff},2}(G) = \|G\|_F^4 / \|G^\top G\|_F^2 \in [1, \min(m, n)]$:
  $$\operatorname{Var}(Z_{\text{sep2}}) \approx \left(2 + \frac{6}{r_{\text{eff},2}(G)}\right)\|G\|_F^4.$$
  - For rank-1 spatial gradients ($G = p q^\top$, e.g., horizontal/vertical edges or stripes, $r_{\text{eff},2} = 1$), the formula factors exactly as $\operatorname{Var}(Z_{\text{sep2}}) = (3 - 2\|p\|_4^4/\|p\|_2^4)(3 - 2\|q\|_4^4/\|q\|_2^4)\|G\|_F^4 - \|G\|_F^4 \to \mathbf{8\|G\|_F^4}$ (**`4×` worse** than dense Rademacher!).
  - For `minimal`'s **3-way Kronecker probe** $u_{c,h,w} = d^{-1/2} s_b a_c b_h c_w$ ([`_minimal/geometry.py:56-60`](../../particlegan/_minimal/geometry.py#L56-L60)), a 3-way rank-1 gradient yields $(3 - 2\rho_c)(3 - 2\rho_h)(3 - 2\rho_w) - 1 \to \mathbf{26\|G\|_F^4}$ (`13×` asymptotic; `15.88` vs. `1.91` on $3\times 32\times 32$), AND because axis 0 only multiplies by a batch scalar $s_b \in \{\pm 1\}$, **all $B$ batch samples share the exact same direction up to sign (`|cos(u_i, u_j)| = 1`)**!

**Theorem 4B (Chained 2-Householder Frame Subspace & Spatial-Mixing Bias):**
In [`_t2_critic/penalty.py:133-142`](../../particlegan/_t2_critic/penalty.py#L133-L142), $Q = H_2 H_1 = (I - 2 v_1 v_1^\top)(I - 2 v_0 v_0^\top) \in \mathrm{SO}(d)$ is exact orthonormal ($Q^\top Q = I_d$, $\det Q = +1$), yet every vector $w \in \operatorname{span}(v_0, v_1)^\perp$ satisfies $Q w = w$, so **$Q$ fixes a $(d-2)$-dimensional subspace** (it is a planar rotation, not Haar-random). Furthermore, [`_prepare_spatial_tangent_plan`](../../particlegan/_t2_critic/penalty.py#L200-L224) assigns only **one** basis vector $q_{\text{slot}(i)}$ per sample $x_i$ and reconstructs $\hat{s}_i = \sum_{k=1}^d (\nabla_x D(x_{j_k})\cdot q_k)^2$ from nearest basis-slot neighbors $x_{j_k}$. This is exact ($\hat{s}_i = \|\nabla_x D\|_2^2$) **only** when $\nabla_x D$ is spatially constant (linear critic); on any nonlinear critic ($\nabla^2 D \neq 0$), mixing gradients across $d$ distinct points incurs strictly nonzero spatial-smoothing bias (`rel_l2 = 0.341–0.415` at $d=6, 8$).

### 2.5 Theorem 5 — Transport-Chord & Cyclic-Roll Secant Semantics

For any pair $(a, b)$ with $a \neq b$, the macro secant $S(a, b) = |D(b) - D(a)| / \|b - a\|_2$ equals $\big|\int_0^1 \nabla_x D((1-t)a + tb)\cdot \hat{v}\,dt\big|$ ($\hat{v} = (b-a)/\|b-a\|_2$) — a **lower bound on the global Lipschitz constant** along segment $[a, b]$, NOT the local slope $\|\nabla_x D\|_2$. It fails in three distinct ways:
1. **Symmetric-flank cancellation:** $D(\mu + r u) = D(\mu - r u) \implies S = 0$ identically even as $\|\nabla_x D\|_2 \to \infty$.
2. **High-dimensional unscaled chord attenuation:** In [`_t2_critic/penalty.py:830,846,860`](../../particlegan/_t2_critic/penalty.py#L830) and [`_unified/penalty.py:599,614,628`](../../particlegan/_unified/penalty.py#L599), `s_chord`, `s_ff`, `s_rr` are **not scaled by $\sqrt{d}$**. For linear $D(x) = w^\top x$ and isotropic batch pairs, $\mathbb{E}[S(a, b)] = \sqrt{2/\pi}\,\|w\|_2 / \sqrt{d} = \Theta(d^{-1/2})$ — attenuated **`452.6×`** at `256×256×3` (`0.0044` vs. true `2.000`)!
3. **Oscillatory mean-value averaging:** $D(x) = \sin(\omega x)$ with $\omega = 20\pi$ has Lipschitz constant $62.83$, yet $S(0, 1) = 0$.

---

## 3. Reproducible Empirical Verification (`p6_law2_autodiff.py`)

All numbers below are verbatim from running `PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p6_law2_autodiff.py --device cuda:0` on `NVIDIA RTX A6000` (`PyTorch 2.15.0+cu132`, protocol seed `0`, `deterministic_orthogonal_`).

### 3.1 Experiment 1 — Mixed 2nd-Order Derivative Identity & Codebase Path Audit

#### (a) Closed-Form Analytical Mixed Hessian vs. `fwAD` (RoF) vs. `autograd.grad` (RoR) (FP64, $B=32, d=6, H=24$)

| Activation | Outer $\Phi$ | $\max|\tau_{\text{fwAD}} - \tau_{\text{anal}}|$ | $\|\nabla_b \mathcal{R}\|_2$ *(pure $\sigma''$ term!)* | `fwAD` vs. Analytical `rel_l2` | `fwAD` vs. Analytical `cos` | `fwAD` vs. RoR `rel_l2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tanh` | `bi` | `4.44e-16` | **`0.0737`** | **`2.54e-16`** | **`1.000000000000`** | `2.94e-16` |
| `tanh` | `ebi` | `4.44e-16` | **`0.0864`** | **`2.55e-16`** | **`1.000000000000`** | `2.88e-16` |
| `silu` | `bi` | `3.33e-16` | **`0.1031`** | **`1.94e-16`** | **`1.000000000000`** | `2.12e-16` |
| `silu` | `ebi` | `3.33e-16` | **`0.0415`** | **`2.48e-16`** | **`1.000000000000`** | `8.82e-16` |

#### (b) Full $\|\nabla_x D\|_2^2$ Canonical `fwAD` (`_eval_lowd_forward_grad_sq`) vs. RoR (`d = 4`, normalized to active slope `2.0`)

| Critic Architecture | Dtype | $\|H_{xx}\mathbf{1}\|_2$ *(spatial 2nd deriv.)* | $\|\nabla_\theta \mathcal{R}\|_2$ *(mixed 2nd deriv.)* | `fwAD` vs. RoR `rel_l2` | `fwAD` vs. RoR `cos` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ReLU` MLP (`fourier=0`) | FP64 / FP32 | **`0.000e+00`** | **`21.1354`** / `22.2527` | **`3.94e-16`** / `2.40e-07` | `1.0000000000` |
| `LeakyReLU(0.2)` MLP (`fourier=0`) | FP64 / FP32 | **`0.000e+00`** | **`30.9293`** / `28.2824` | **`5.62e-16`** / `3.16e-07` | `1.0000000000` |
| `SimpleMLPDiscriminator` (`fourier=2`) | FP64 / FP32 | `2.363e+01` / `2.401e+01` | `22.8276` / `19.2036` | `8.32e-16` / `4.33e-07` | `1.0000000000` |
| `Tanh` MLP | FP64 / FP32 | `1.770e+00` / `3.091e+00` | `45.0519` / `45.7985` | `4.83e-16` / `2.30e-07` | `1.0000000000` |
| `SiLU` MLP | FP64 / FP32 | `2.526e+00` / `3.094e+00` | `41.4806` / `38.2553` | `5.19e-16` / `2.95e-07` | `1.0000000000` |

#### (c) Actual Codebase Householder Spatial-Frame (`d=6, 8`) & $k=32$ Hutchinson (`d=32, 170`) Audit vs. Exact RoR

- **Chained Householder $Q = H_2 H_1$ (`_build_orthonormal_frame`):** `orth_err <= 2.38e-7`, `det = +1.000000`, and unit eigenvalues $= d - 2$ (`3/5`, `4/6`, `6/8`) — confirming $Q$ fixes a $(d-2)$-dimensional subspace.
- **Parameter-gradient alignment ($\nabla_\theta \mathcal{R}_{\text{est}}$ vs. exact full RoR $\nabla_\theta \mathcal{R}_{\text{RoR}}$, $B=128$, 4-cluster mixture):**

| Dimension | ParticleGAN Path | Critic | `grad_cos` vs. RoR | `grad_rel_l2` vs. RoR | Unpooled $K=1$ `grad_cos` | Effective Pool $k_{\text{eff}}/32$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `6` | Householder spatial-frame | Linear | **`1.0000`** | **`0.0000`** | — | — |
| `6` | Householder spatial-frame | Nonlinear LeakyReLU MLP | `0.9484` | **`0.3411`** | — | — |
| `8` | Householder spatial-frame | Linear | **`1.0000`** | **`0.0000`** | — | — |
| `8` | Householder spatial-frame | Nonlinear LeakyReLU MLP | `0.9614` | **`0.4154`** | — | — |
| `32` | Murmur3 + $k=32$ Gaussian pool | Nonlinear LeakyReLU MLP | `0.8422` | `1.3418` | `0.7488` | `31.27 / 32` |
| `170` | Murmur3 + $k=32$ Gaussian pool | Nonlinear LeakyReLU MLP | **`0.5248`** | **`1.0482`** | `0.2623` | `31.55 / 32` |

*(Note: $k_{\text{eff}} \approx 31.3\text{–}31.6$ is close to $32$ because pairwise squared distances concentrate tightly around the median bandwidth `knn_d2[:, 16]` in high dimensions; $k=32$ pooling doubles parameter-gradient cosine at $d=170$ from `0.262` to `0.525`, yet still leaves `rel_l2 = 1.05` error.)*

---

### 3.2 Experiment 2 — Exact Jensen Bias, Gradient Attenuation & Separable Variance Blowup

#### (a) Strictly Concave Born–Infeld $\phi_\beta(s) = (\sqrt{1+\beta s}-1)/\beta$ ($d=170$, $40{,}000$ MC probes vs. exact $\chi_1^2$ Gauss–Hermite)

| $\beta$ | True $r$ | Probes $K$ | $\mathbb{E}[\hat{s}]/s$ | Loss Jensen Bias *(exact $\chi_1^2$)* | Grad Magnitude Bias *(exact $\chi_1^2$)* | Loss Rel. Std |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `0.5` | `1` / `4` / `32` | `0.9936` / `0.9996` / `0.9990` | **`-7.86%`** *(`-7.39%`)* / `-2.22%` / `-0.39%` | **`-12.65%`** *(`-12.28%`)* / `-3.86%` / `-0.61%` | `1.170` / `0.643` / `0.233` |
| `1` | `1.0` | `1` / `4` / `32` | `0.9935` / `0.9954` / `0.9990` | **`-14.78%`** *(`-14.41%`)* / `-5.02%` / `-0.73%` | **`-20.39%`** *(`-20.11%`)* / `-7.06%` / `-1.02%` | `0.975` / `0.560` / `0.210` |
| `1` | `2.0` | `1` / `4` / `32` | `1.0054` / `0.9956` / `1.0007` | **`-19.10%`** *(`-19.54%`)* / `-6.71%` / `-0.84%` | **`-22.45%`** *(`-22.83%`)* / `-7.59%` / `-0.95%` | `0.818` / `0.476` / `0.179` |
| `1` | `5.0` | `1` / `4` / `32` | `0.9993` / `1.0001` / `0.9988` | **`-21.42%`** *(`-21.46%`)* / `-6.69%` / `-0.95%` | **`-21.66%`** *(`-21.66%`)* / `-6.42%` / `-0.89%` | `0.697` / `0.403` / `0.148` |
| `8` (`minimal`) | `0.5` | `1` / `4` / `32` | `0.9952` / `1.0062` / `1.0011` | **`-17.56%`** *(`-17.42%`)* / `-5.31%` / `-0.71%` | **`-22.23%`** *(`-22.17%`)* / `-7.03%` / `-0.94%` | `0.884` / `0.521` / `0.194` |
| `8` (`minimal`) | `1.0` | `1` / `4` / `32` | `0.9861` / `1.0024` / `1.0008` | **`-21.50%`** *(`-20.78%`)* / `-6.55%` / `-0.85%` | **`-23.26%`** *(`-22.63%`)* / `-6.86%` / `-0.87%` | `0.756` / `0.444` / `0.165` |
| `8` (`minimal`) | `2.0` | `1` / `4` / `32` | `0.9917` / `0.9977` / `0.9986` | **`-21.79%`** *(`-21.48%`)* / `-6.76%` / `-0.95%` | **`-21.78%`** *(`-21.45%`)* / `-6.43%` / `-0.88%` | `0.684` / `0.395` / `0.144` |
| `8` (`minimal`) | `5.0` | `1` / `4` / `32` | `1.0057` / `1.0019` / `1.0030` | **`-21.04%`** *(`-21.00%`)* / `-6.12%` / `-0.66%` | **`-20.53%`** *(`-20.35%`)* / `-5.84%` / `-0.63%` | `0.644` / `0.361` / `0.133` |

- **Exact $2^4 = 16$ Rademacher Directional Distortion ($d=4$, $g = [2.0, 0.4, 0.4, 0.4]$, $\beta=1$):** Exact $\nabla_g \phi = [0.8544, 0.1709, 0.1709, 0.1709]$ vs. exact Rademacher expectation $\mathbb{E}[\nabla_g \phi(\hat{s})] = [0.8616, 0.0506, 0.0506, 0.0506]$ — a **`13.30°` directional rotation** (`cos = 0.9732`) suppressing minor gradient components by **`3.38×`**!

#### (b) Strictly Convex-on-Cap EBI Squared Softplus Cap $h_{12,1}(s) = [\operatorname{softplus}_{12}(\sqrt{s}-1)]^2$ ($100{,}000$ draws, $\chi_K^2/K$ limit)

| True Slope $r$ | Probes $K$ | Exact $h(s)$ | $\mathbb{E}[h(\hat{s})]$ | **Penalty Inflation Ratio** | $\mathbb{P}(\sqrt{\hat{s}} > 1)$ | **Grad Multiplier Inflation** | Reported Norm $\mathbb{E}[\sqrt{\hat{s}}]/r$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.5`** *(< cap)* | `1` / `4` / `32` | `4.25e-8` | `0.00323` / `0.00004` / `1.3e-7` | **`75,861×`** / `1,015×` / `2.98×` | `4.6%` / `0.3%` / `0.0%` | **`47,731×`** / `1,331×` / `3.54×` | `0.799` / `0.941` / `0.993` |
| **`0.8`** *(< cap)* | `1` / `4` / `32` | `0.00005` | `0.05315` / `0.01009` / `0.00037` | **`1,015.1×`** / `192.7×` / `7.06×` | **`21.1%`** / `18.1%` / `2.2%` | **`275.8×`** / `86.4×` / `6.52×` | **`0.797`** / `0.940` / `0.991` |
| **`1.0`** *(= cap)* | `1` / `4` / `32` | `0.00334` | `0.15313` / `0.05157` / `0.00976` | **`45.90×`** / `15.46×` / `2.93×` | **`31.8%`** / `40.6%` / `46.5%` | **`11.12×`** / `5.61×` / `2.02×` | **`0.800`** / `0.942` / `0.992` |
| `1.2` *(> cap)* | `1` / `4` / `32` | `0.04295` | `0.31533` / `0.14542` / `0.06036` | **`7.34×`** / `3.39×` / `1.41×` | `40.6%` / `59.6%` / `90.1%` | **`2.58×`** / `1.65×` / `1.10×` | `0.801` / `0.940` / `0.992` |
| `2.0` *(> cap)* | `1` / `4` / `32` | `1.00000` | `1.66990` / `1.22429` / `1.03362` | `1.67×` / `1.22×` / `1.03×` | `61.3%` / `91.0%` / `100%` | `1.23×` / `1.06×` / `1.01×` | `0.794` / `0.939` / `0.993` |

- **Local $k$-NN Pooling Curvature-Bias vs. Jensen-Bias Trade-Off (`d = 32`, quadratic $D(x) = \frac{c}{2}\|x\|^2$):** At $c = 0.15$, pooling from $k=1 \to 32$ reduces EBI positive Jensen bias from `+7,594%` to `0%` around $k \approx 16$, then **overshoots to `-69.1%` negative bias** at $k=32$ because spatial averaging blurs high-slope outer-shell points inward; at $c = 0.70$, $k=32$ flips EBI bias from `+17.7%` to `-18.0%` with an irreducible pointwise relative MAE floor of `0.258`.

#### (c) Separable vs. Dense Rademacher Variance Blowup ($3 \times 32 \times 32$, $\|G\|_F = 1$) & Batch Correlation

| Spatial Gradient $G \in \mathbb{R}^{96 \times 32}$ | Rényi-2 $r_{\text{eff},2}(G)$ | Dense Var *(exact th.)* | **2-Way Sep. Var *(exact th.)*** | **3-Way Sep. Var (`minimal`)** |
| :--- | :---: | :---: | :---: | :---: |
| `rank1_2way_stripe` ($p q^\top$) | `1.00` | `1.96` *(`2.00`)* | **`7.41`** *(`7.31`, **`3.78×`**)* | `7.15` |
| `rank1_3way_chw` ($a_c b_h c_w$) | `1.00` | `1.91` *(`2.00`)* | **`7.63`** *(`7.43`, **`3.99×`**)* | **`15.88`** *(**`8.31×`**)* |
| `rank4_lowrank` | `3.26` | `2.00` *(`2.00`)* | `3.43` *(`3.49`, `1.72×`)* | `3.51` |
| `fullrank_isotropic` | `23.89` | `2.09` *(`2.00`)* | `2.02` *(`2.00`, `0.97×`)* | `2.02` |

- **Batch-wide probe collinearity in `minimal` (`_minimal/geometry.py:56-60`):** Across $B=16$ samples on `32×32×3`, `minimal` off-diagonal $|\cos(u_i, u_j)| \in [0.99999996, 0.99999996]$ (all samples share one identical spatial direction up to sign!), whereas `t2_secant`'s `_separable_image_unit_perturbation` has mean off-diagonal $|\cos| = 0.0109$ (max `0.1094`).

---

### 3.3 Experiment 3 — ReLU/LeakyReLU Kink Impulses, Actual `DCGANCritic` BF16 Micro-Secants & Chord Failures

#### (a) Controlled Hyperplane Kink Crossing ($\eta = 10^{-4}$, 2-Layer `LeakyReLU(0.2)`, FP64)

| Method | Step $\delta$ | Crosses Kink? | $\tau$ Rel. Err | $\|\nabla_\theta \hat{\mathcal{R}}\| / \|\nabla_\theta \mathcal{R}_{\text{exact}}\|$ | `grad_cos` vs. Exact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`fwAD` (RoF)** | `0` | No | **`0.000e+00`** | **`1.000`** | **`+1.00000`** |
| Forward FD | `2.0e-2` | Yes | `3.183` | **`45.15×`** | `-0.03243` |
| Central FD | `2.0e-2` | Yes | `1.592` | **`6.13×`** | `-0.08057` |
| Forward FD | `1.0e-3` | Yes | `2.879` | **`776.69×`** | `-0.00170` |
| Central FD | `1.0e-3` | Yes | `1.440` | **`90.85×`** | `-0.00412` |
| Forward FD | `2.0e-4` | Yes | `1.600` | **`1,239.06×`** | `-0.00040` |
| Central FD | `2.0e-4` | Yes | `0.800` | **`206.84×`** | `+0.00089` |
| Forward / Central FD | `5.0e-5` | No ($\delta < \eta$) | `1.3e-11` / `3.0e-12` | `1.000` / `1.000` | `+1.00000` |

#### (b) Actual `DCGANCritic` BF16 Micro-Secant & Kink Audit (`cuda:0`, $B=16$, `BN.eval()`, identical separable direction $u$)

*(Reference: exact FP32 `fwAD` along $u$; RoR along $u$ matches FP32 `fwAD` to `cos = 1.000000` across all 4 resolutions.)*

| Res | Method | Surviving Pixels | LeakyReLU Flips *(count / %)* | Zero Logit Diff Frac | $\hat{\tau}$ Rel. L2 Err | **`grad_cos` vs. Exact** | **Grad Norm Ratio** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`32²`** | **`unified_v11_fwAD_bf16`** | `100.0%` | **`0`** (`0.000%`) | `0.00` | `0.056` | **`+0.9964`** | **`1.013×`** |
| `32²` | `t2_micro_secant_fp32` (`0.02`) | `100.0%` | `92` (`0.020%`) | `0.00` | `0.727` | `+0.0538` | `18.58×` |
| `32²` | `t2_micro_secant_bf16` (`0.02`) | **`24.83%`** | `311` (`0.068%`) | **`0.62`** | `5.334` | **`+0.0064`** | **`254.8×`** |
| `32²` | `minimal_micro_secant_fp32` (`1e-5`) | `100.0%` | `1` (`0.000%`) | `0.00` | `189.7` | **`+0.0160`** | **`37,976×`** |
| `32²` | `minimal_micro_secant_bf16` (`1e-5`) | **`0.05%`** | `3` (`0.001%`) | **`1.00`** | `1.000` | **`0.0000`** | **`0.000×`** *(dead)* |
| **`64²`** | **`unified_v11_fwAD_bf16`** | `100.0%` | **`0`** (`0.000%`) | `0.00` | `0.075` | **`+0.9941`** | **`0.976×`** |
| `64²` | `t2_micro_secant_fp32` (`0.02`) | `100.0%` | `251` (`0.013%`) | `0.00` | `1.633` | `+0.0155` | `102.3×` |
| `64²` | `t2_micro_secant_bf16` (`0.02`) | **`14.82%`** | `1,164` (`0.059%`) | **`0.44`** | `19.63` | **`-0.0020`** | **`2,103×`** |
| `64²` | `minimal_micro_secant_fp32` (`1e-5`) | `100.0%` | `10` (`0.001%`) | `0.00` | `712.0` | **`+0.0009`** | **`1.33 × 10⁷×`** |
| `64²` | `minimal_micro_secant_bf16` (`1e-5`) | **`0.02%`** | `11` (`0.001%`) | **`0.94`** | `9,923` | **`-0.0014`** | **`7.76 × 10⁸×`** |
| **`128²`** | **`unified_v11_fwAD_bf16`** | `100.0%` | **`0`** (`0.000%`) | `0.00` | `0.094` | **`+0.9920`** | **`1.017×`** |
| `128²` | `t2_micro_secant_fp32` (`0.02`) | `100.0%` | `639` (`0.008%`) | `0.00` | `3.131` | `+0.0059` | `305.3×` |
| `128²` | `t2_micro_secant_bf16` (`0.02`) | **`8.49%`** | `3,788` (`0.047%`) | `0.25` | `39.75` | **`-0.0016`** | **`9,315×`** |
| `128²` | `minimal_micro_secant_fp32` (`1e-5`) | `100.0%` | `35` (`0.000%`) | `0.00` | `2,518` | **`+0.0011`** | **`1.24 × 10⁸×`** |
| `128²` | `minimal_micro_secant_bf16` (`1e-5`) | **`0.01%`** | `41` (`0.001%`) | **`0.56`** | `40,996` | **`+0.0021`** | **`9.90 × 10⁹×`** |
| **`256²`** | **`unified_v11_fwAD_bf16`** | `100.0%` | **`0`** (`0.000%`) | `0.00` | `0.150` | **`+0.9869`** | **`1.065×`** |
| `256²` | `t2_micro_secant_fp32` (`0.02`) | `100.0%` | `1,734` (`0.005%`) | `0.00` | `8.671` | **`-0.0003`** | **`2,617×`** |
| `256²` | `t2_micro_secant_bf16` (`0.02`) | **`4.80%`** | **`11,881`** (`0.037%`) | **`0.56`** | `146.6` | **`-0.0017`** | **`87,478×`** |
| `256²` | `minimal_micro_secant_fp32` (`1e-5`) | **`49.96%`** | `103` (`0.000%`) | `0.00` | `8,858` | **`+0.0000`** | **`3.69 × 10⁹×`** |
| `256²` | `minimal_micro_secant_bf16` (`1e-5`) | **`0.01%`** | `163` (`0.001%`) | **`0.81`** | `153,157` | **`+0.0012`** | **`6.74 × 10¹⁰×`** |

*(All `20/20` executed rows reported verbatim; note that `minimal_micro_secant_fp32` already explodes `37,976× -> 1.24 × 10⁸×` at `32²`–`128²` even at `100%` pixel survival because `1–35` LeakyReLU flips are amplified by `1 / 1e-5 = 10⁵`!)*

#### (c) Transport-Chord & Cyclic-Roll Secants
- **Symmetric-flank cancellation** ($D(x) = -\alpha \|x\|_2^2$, pairs $\pm e_1$): `s_chord == 0.0` identically across $\alpha \in \{1, 10, 100\}$ while `fwAD` recovers exact true slopes `{2.0, 20.0, 200.0}`.
- **High-D unscaled chord attenuation** ($D(x) = w^\top x$, $\|w\|_2 = 2.0$): `s_chord` / `s_roll` decay from `1.484 / 1.360` (`d=2`) $\to$ `0.233 / 0.186` (`d=32`) $\to$ `0.105 / 0.111` (`d=170`) $\to$ `0.0217 / 0.0260` (`d=3,072`, `92.3×` attenuation) $\to$ **`0.0044 / 0.0036` (`d=196,608`, `452.6×` attenuation!)**.
- **Oscillatory mean-value cancellation** ($D(x) = \sin(20\pi x)$ on $[0, 1]$): true Lipschitz constant `62.832` vs. macro chord `2.45e-15`.

---

### 3.4 Experiment 4 — RTX A6000 Computational Scaling Regimes (Exact Wins vs. Losses)

#### (a) MLP Dimension Scaling (`NVIDIA RTX A6000`, $B=256$, 4-Layer Width-256 `LeakyReLU`, FP32)

| Dim $d$ | **`RoR_exact`** (`ms` · `MiB`) | **`fwAD_exact_basis`** (`ms` · `MiB` · vs. RoR) | **`fwAD_1probe`** (`ms` · vs. RoR) | `fwAD_4p` (`ms`) | `fwAD_8p` (`ms`) | `micro_secant` (`ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2` | **`1.475` · `13.2`** | `2.915` · `16.9` (**`1.98×`**) | `2.776` (**`1.88×`**) | `2.898` | `2.616` | `1.412` |
| `4` | **`1.437` · `13.3`** | `2.795` · `25.0` (**`1.95×`**) | `2.767` (**`1.93×`**) | `2.868` | `2.766` | `1.697` |
| `8` | **`1.416` · `13.4`** | `2.857` · `41.2` (**`2.02×`**) | `2.639` (**`1.86×`**) | `2.815` | `2.872` | `1.419` |
| `16` | **`1.517` · `13.6`** | `2.872` · `73.9` (**`1.89×`**) | `2.594` (**`1.71×`**) | `2.901` | `2.953` | `1.669` |
| `32` | **`1.592` · `14.1`** | `5.308` · `123.8` (**`3.33×`**) | `2.768` (**`1.74×`**) | `2.871` | `2.900` | `1.528` |
| `64` | **`1.332` · `15.0`** | `10.916` · `218.7` (**`8.19×`**) | `2.560` (**`1.92×`**) | `2.677` | `2.719` | `1.592` |
| `128` | **`1.336` · `16.9`** | `20.478` · `420.5` (**`15.33×`**) | `2.737` (**`2.05×`**) | `2.860` | `2.970` | `1.708` |
| `256` | **`1.537` · `20.9`** | `41.214` · `872.2` (**`26.82×`**) | `2.547` (**`1.66×`**) | `2.677` | `2.687` | `1.576` |
| `1024` | **`1.315` · `47.1`** | `164.887` · **`4,928.9`** (**`125.42×`**) | `2.376` (**`1.81×`**) | `2.502` | `2.739` | `1.511` |

> **Critical MLP Crossover Finding:** On GPU MLPs (`B=256, width=256, FP32`), **exact RoR ALREADY BEATS `fwAD` at $d = 2$** — beating BOTH exact-basis `fwAD` (which scales linearly to **`125.4×` slower / `104.6×` VRAM** at $d=1024$) AND 1-probe `fwAD` (`1.66×–2.05×` slower at every $d \in [2, 1024]$ due to dual-mode kernel-launch overhead) while delivering **zero directional variance and zero Jensen bias**!

#### (b) Conv2d `DCGANCritic` Resolution Scaling (`NVIDIA RTX A6000`, $B=16$, full `(x_real, x_fake)` critic penalty + `.backward()`)

| Res & Precision | `RoR_bn_train` (`ms` · `MiB` · `mJ`) | `RoR_bn_eval` (`ms` · `MiB`) | **`fwAD1_interl`** (`ms` · `MiB` · `mJ` · vs. `RoR_tr`) | `fwAD1_both` (`ms` · vs. `RoR_ev`) | `fwAD4_interl` (`ms`) | `secant_1p` (`ms`) *(broken grad!)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `32²` FP32 | `4.83` · `29.0` · `894` | `3.67` · `28.3` | `3.70` · `21.1` · `383` (`1.30×`) | `7.01` (**`0.52×`**) | `14.08` | `1.25` |
| `32²` BF16 CL | `5.61` · `22.3` · `843` | `4.01` · `21.9` | `3.74` · `19.0` · `339` (`1.50×`) | `6.83` (**`0.59×`**) | `13.54` | `1.30` |
| `64²` FP32 | `7.65` · `106.1` · `1,338` | `6.07` · `102.6` | `5.42` · `70.6` · `617` (`1.41×`) | `9.71` (**`0.63×`**) | `19.65` | `1.57` |
| `64²` BF16 CL | `7.39` · `73.1` · `1,170` | `5.08` · `69.9` | `5.09` · `54.7` · `490` (`1.45×`) | `9.47` (**`0.54×`**) | `19.91` | `1.55` |
| `128²` FP32 | `22.31` · `398.6` · `6,203` | `18.48` · `384.1` | **`8.22`** · `247.6` · **`2,056`** (**`2.71×`**) | `15.54` (`1.19×`) | `31.07` | `2.76` |
| `128²` BF16 CL | `13.43` · `253.6` · `3,103` | `10.40` · `242.1` | **`6.58`** · `174.5` · **`1,508`** (**`2.04×`**) | `12.26` (**`0.85×`**) | `23.96` | `1.99` |
| `256²` FP32 | `79.76` · `1,462.7` · `20,455` | `65.02` · `1,432.1` | **`24.86`** · `812.1` · **`7,721`** (**`3.21×`**) | `49.79` (`1.31×`) | `99.79` | `8.90` |
| **`256²` BF16 CL** | `45.15` · `861.7` · `11,301` | `34.62` · `837.1` | **`12.40`** · **`576.5`** · **`3,799`** (**`3.64×`**) | `24.52` (`1.41×`) | **`49.63`** *(slower!)* | `3.98` |

> **Decomposing the `3.64×` `r256` BF16 Win (`45.15 ms -> 12.40 ms`):** Three separate mechanisms contribute:
> 1. **Switching `BatchNorm2d` to `.eval()`:** `45.15 -> 34.62 ms` (`1.30×` — already available to RoR if BN eval is acceptable!);
> 2. **Genuine RoF (`fwAD`) vs. RoR kernel speedup on matched sample count (`fwAD1_both` vs. `RoR_eval`):** `34.62 -> 24.52 ms` (**`1.41×`**);
> 3. **Halving evaluated samples via 1 interleaved half-real/half-fake batch (`fwAD1_interl` vs. `fwAD1_both`):** `24.52 -> 12.40 ms` (**`1.98×`** — sampling half the batch per step, which doubles per-step estimator variance!).
> Meanwhile, attempting to mitigate 1-probe variance/Jensen bias with `K = 4` probes (`fwAD4_interl = 49.63 ms`) **completely erases the speedup** and becomes slower than exact full-gradient `RoR_train` (`45.15 ms`) and `1.43×` slower than `RoR_eval` (`34.62 ms`)!

#### (c) Architectural Compatibility & Overhead Matrix (`NVIDIA RTX A6000`, PyTorch `2.15.0+cu132`)

| Architecture & Precision | `RoR_exact` (`create_graph=True`) | `fwAD_1probe` (`dual_level` + `.backward()`) | `micro_secant` (`delta=0.02`) |
| :--- | :---: | :---: | :---: |
| `DCGAN_64_BN_train_no_hook` FP32 | **OK** (`3.70 ms`, `71 MiB`) | OK (`5.69 ms`, `77 MiB`) | OK (`2.19 ms`, `67 MiB`) |
| `DCGAN_64_BN_train_no_hook` BF16 | **OK** (`3.44 ms`, `48 MiB`) | **FAIL** (`RuntimeError`: FP32 dual input vs. BF16 Conv weight) | OK (`2.09 ms`, `45 MiB`) |
| `DCGAN_64_BN_eval_no_hook` BF16 | **OK** (`2.50 ms`, `47 MiB`) | **FAIL** (`RuntimeError`: FP32 dual input vs. BF16 Conv weight) | OK (`2.16 ms`, `45 MiB`) |
| `DCGAN_64_BN_eval_with_hook` BF16 | **OK** (`2.69 ms`, `47 MiB`) | OK (`4.77 ms`, `52 MiB`, requires `_bn_dual_bf16_hook`) | OK (`2.01 ms`, `45 MiB`) |
| `ResNet_64_Skip_GroupNorm` FP32 | **OK** (`7.57 ms`, **`109 MiB`**) | OK (`10.02 ms`, **`228 MiB`**, `2.09×` VRAM!) | OK (`3.42 ms`, `106 MiB`) |
| `ResNet_64_Skip_GroupNorm` BF16 | **OK** (`5.95 ms`, **`79 MiB`**) | OK (`10.62 ms`, **`207 MiB`**, **`2.62×` VRAM!**) | OK (`3.74 ms`, `86 MiB`) |
| `Attention_SDPA_default` FP32 / BF16 | **FAIL** (`RuntimeError`: no 2nd deriv. for MemEff/Flash SDPA) | **FAIL** (`NotImplementedError`: no `fwAD` for MemEff/Flash SDPA) | **OK** (`2.00` / `2.42 ms`) |
| `Attention_SDPA_math` FP32 / BF16 | **OK** (`3.58` / `5.09 ms`, `30` / `25 MiB`) | **FAIL** (`RuntimeError`: in-place `Exp` version error in `.backward()`) | **OK** (`2.77` / `3.44 ms`) |
| `EmulatedLinear` (`mxfp8` / `nvfp4`) | **OK** (`3.91` / `7.92 ms`, via STE proxy) | **FAIL** (`NotImplementedError`: missing `jvp` on `_Linear`) | **OK** (`4.53` / `9.10 ms`) |

---

## 4. Eight-Point Structured Analysis of All Substantial Claims

### Claim C1 — "`fwAD.dual_level()` + `.backward()` is purely first-order / eliminates higher-order differentiation"
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims in [`docs/four-laws.md:51,113,135,141,182`](../../docs/four-laws.md#L51) and [`t2-efficient-critic.md:7,51,73`](../forge/studies/t2-efficient-critic.md#L7) classify `fwAD.dual_level()` + `.backward()` as *"purely 1st-order"* / *"Autograd Order: 1st-order"* and equate zero `torch.autograd.grad` calls with zero higher-order differentiation.
2. **Assumptions:** Conflates the **autodiff execution graph mode** (avoiding reverse-over-reverse `create_graph=True` double-backprop) with the **mathematical order of differentiation**.
3. **Existing supporting evidence:** Correctly eliminates `torch.autograd.grad(..., create_graph=True)` calls (`0` calls monkeypatch-verified across Forge tests).
4. **Potential counterexamples:** Theorem 1 proves $\nabla_\theta \Phi\big((\nabla_x D_\theta \cdot u)^2\big) = 2\Phi'\tau \left(\frac{\partial^2 D_\theta}{\partial \theta \partial x}\right) u$. On smooth 2-layer `Tanh`/`SiLU` MLPs, $\nabla_b \mathcal{R}$ is purely proportional to $\sigma''(z)$ (`||dL/db|| = 0.0415–0.1031`), and `fwAD` + `.backward()` matches closed-form analytical mixed Hessians to `1.94e-16–2.55e-16`. On piecewise-linear `ReLU`/`LeakyReLU` MLPs, $H_{xx} = 0$ a.e. yet $H_{\theta x} \nabla_x D \neq 0$ (`21.14–30.93`) and canonical `fwAD` matches RoR to `3.94e-16–5.62e-16`.
5. **Minimal falsification experiment:** Differentiate $\frac{1}{2}(\nabla_x [v^\top \tanh(W x + b)]\cdot u)^2$ w.r.t. $b$ using `fwAD.dual_level()` + `.backward()` and compare to $v \odot \tanh''(W x + b) \odot (W u)$.
6. **Relevant prior research:** Pearlmutter (1994) *Fast Exact Multiplication by the Hessian* (forward-over-reverse / reverse-over-forward $R\{\cdot\}$ operator); Griewank & Walther (2008) *Evaluating Derivatives* (Ch. 13, mixed-mode higher derivatives); Baydin et al. (2018) *Automatic Differentiation in Machine Learning: a Survey*.
7. **Computational verification:** Verified in Exp 1A/1B and `test_fwad_backward_equals_analytical_mixed_hessian_fp64` (including deliberate-damage control dropping $\sigma''$) + `test_canonical_fwad_equals_ror_on_smooth_and_piecewise_linear_critics` (all PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** (exact calculus proof + FP64 machine-precision verification).

### Claim C2 — "`unified_v11` (and default `unified_v12`) uses live-endpoint reuse + separable BF16 micro-secants + transport/cyclic chords on Convolutional Image Critics"
- **Classification:** **Falsified claim** *(documentation-vs-code drift)*
1. **Precise formulation:** [`docs/four-laws.md:113,136`](../../docs/four-laws.md#L113) lists `unified_v11` (and by recipe inheritance [`recipes.py:2816-2819`](../../particlegan/recipes.py#L2816-L2819), `DEFAULT_RECIPE = "unified_v12"`) as executing BF16 micro-secant probes alongside transport-chord and cyclic-roll secants on convolutional image critics.
2. **Assumptions:** Assumes `unified_v11` retained `_unified/penalty.py`'s image secant path unchanged.
3. **Existing supporting evidence:** Accurate *only* for earlier/sibling formulations (`t2_secant`, `t4_nash_v3`, `unified`, `unified_v4`, `cond_fiber`, and `minimal`).
4. **Potential counterexamples:** Direct source inspection of [`particlegan/_unified_v11/penalty.py:404-490,631-695`](../../particlegan/_unified_v11/penalty.py#L404-L490) shows `_UnifiedV11PenaltyKernel.penalty` overrides the entire image branch with `_eval_image_fwad_sq` (1-probe interleaved `fwAD.dual_level()`, `BatchNorm2d.eval()`, `_bn_dual_bf16_hook`, zero micro-secants, zero transport chords, zero cyclic-roll secants; live-endpoint reuse supplies only `d_r_2d, d_f_2d` for the zero-mean logit gauge!).
5. **Minimal falsification experiment:** Inspect `_UnifiedV11PenaltyKernel.penalty` lines 631–695 or trace calls during `UnifiedV11CriticPenalty` on a `4D` image tensor (`fwAD.dual_level` is invoked; `_logit_diff_rms` / `torch.roll` are never called).
6. **Relevant prior research:** Internal lineage evolution (`t2_secant` -> `unified_v4` -> `unified_v11`/`v12`); see also `law4-optimizer-laws.md:306` which accurately profiled `10.1% fwAD` inside `r256` `d_penalty`.
7. **Computational verification:** Verified by direct code inspection and Exp 3B / Exp 4B benchmarking `_eval_image_fwad_sq` + `_bn_dual_bf16_hook` (all PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim C3 — "`fwAD.dual_level()` evaluates directional derivatives at the exact primal activation mask with zero ReLU/LeakyReLU $O(1/\delta)$ kink impulses"
- **Classification:** **Proven mathematical result** *(for continuous piecewise-linear networks off the measure-zero kink set $\mathcal{H}$, and verified numerically in FP64/FP32/BF16)*
1. **Precise formulation:** Propagating dual numbers $(x, u)$ through `ReLU`/`LeakyReLU` gates applies the primal mask indicator $\mathbb{I}[z > 0]$ identically to primal and tangent streams, avoiding cross-hyperplane mismatch between two finite-difference evaluations and eliminating $O(1/\delta)$ parameter-gradient impulses.
2. **Assumptions:** Input $x \notin \mathcal{H}$ (almost everywhere under any continuous measure) and every layer implements `jvp` (fails for custom `autograd.Function` like `EmulatedLinear`, where fallback to central FD reintroduces kinks!).
3. **Existing supporting evidence:** [`t2-efficient-critic.md:51`](../forge/studies/t2-efficient-critic.md#L51) and exact agreement between `fwAD` and RoR.
4. **Potential counterexamples / boundary conditions:** (a) When a custom layer lacks `jvp` (`EmulatedLinear`), ParticleGAN silently falls back to central finite differences (`delta >= 0.06` in `unified_v11`, `1e-3` in `t2_secant`), **reintroducing kink impulses**; (b) on a measure-zero point $x \in \mathcal{H}$ exactly on a ReLU boundary, subdifferential choice (`0` vs. `1`) is convention-dependent.
5. **Minimal falsification experiment:** Run Exp 3A (`eta = 1e-4`, `delta = 2e-4`) and Exp 3B (`DCGANCritic` `32^2`–`256^2` BF16): `fwAD` exhibits `0` LeakyReLU flips (`cos = 0.987–0.996`) vs. `311–11,881` flips for BF16 micro-secant.
6. **Relevant prior research:** Kakade & Lee (2018) *Provably Correct Automatic Sub-differentiation for Qualified Programs*; Bolte & Pauwels (2020) *Conservative Set Valued Fields*.
7. **Computational verification:** Verified in Exp 3A/3B, `test_relu_kink_impulse_fwad_immune_vs_fd_explosion` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** within stated assumptions (native `jvp`-supported ops).

### Claim C4 — "Separable BF16 micro-secants (`t2_secant`/`unified` `delta = 0.02`, `minimal` `eps = 1e-5`) provide faithful slope regularity compatible with BF16 and avoid kink impulses"
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims in [`docs/four-laws.md:50-51,136`](../../docs/four-laws.md#L50-L51) and [`t2-efficient-critic.md:18,61`](../forge/studies/t2-efficient-critic.md#L18) assert that BF16 separable micro-secants enforce accurate local slope control without kink impulses or low-precision degradation on Conv2d critics.
2. **Assumptions:** Implicitly assumes $\delta / \sqrt{d}$ is simultaneously small enough not to cross LeakyReLU hyperplanes AND large enough to survive BF16 pixel rounding and BF16 logit subtraction.
3. **Existing supporting evidence:** `t2-secant-v2` and `minimal-v2` complete finite `dcgan_performance_r256` updates quickly (`87.96` / `55.50 ms/step`), which is a throughput-completion diagnostic, NOT a gradient-faithfulness measurement (`scaling_adapter.py:3-5`).
4. **Potential counterexamples:** Exp 3B on actual `DCGANCritic` (`32^2`–`256^2`):
   - `t2_secant` BF16 (`delta=0.02`): only `24.8% -> 4.80%` of pixels survive BF16 rounding; `311 -> 11,881` LeakyReLU units flip sign; `25%–62%` of samples yield zero logit difference; parameter-gradient cosine drops to **`+0.0064` (`r32`) down to `-0.0017` (`r256`)** with **`255× -> 87,478×` norm explosion**! Even in FP32, `92 -> 1,734` flips collapse cosine to `+0.054 -> -0.0003` (`18.6× -> 2,617×` norm blowup).
   - `minimal` (`eps=1e-5`): `100%` of pixels with $|x| \ge 0.5$ vanish even in FP32 at `r256`; in BF16, `99.95%–99.99%` of pixels vanish, producing `100%` zero gradient at `r32` and **`6.74 × 10^{10}×` spikes (`cos = +0.0012`)** at `r256`.
5. **Minimal falsification experiment:** Evaluate `DCGANCritic(256)` in BF16 autocast along one unit direction `u` via `fwAD` vs. `(D(x + 0.02 u) - D(x)) / 0.02` and compute parameter-gradient cosine (`-0.0017`).
6. **Relevant prior research:** Higham (2002) *Accuracy and Stability of Numerical Algorithms* (catastrophic cancellation in finite differences); Berahas et al. (2022) *A Theoretical and Empirical Comparison of Gradient Approximations in Derivative-Free Optimization*.
7. **Computational verification:** Verified in Exp 3B and `test_micro_secant_quantization_underflow_and_chord_failure_modes` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** for the tested `DCGANCritic` initial/eval regime; open question whether larger per-pixel steps (`delta = Theta(sqrt(d) * u_bf16)`) can trade off BF16 survival against macro curvature & kink flips — though `unified_v11`/`v12` already solved this cleanly by switching images to `_eval_image_fwad_sq`!

### Claim C5 — "Naive macro chord secants fail via symmetric-flank cancellation and $\Theta(\sqrt{d})$ high-D projection attenuation"
- **Classification:** **Proven mathematical result**
1. **Precise formulation:** [`docs/four-laws.md:44-45`](../../docs/four-laws.md#L44-L45) and [`t2-efficient-critic.md:10-11,41-42`](../forge/studies/t2-efficient-critic.md#L41-L42) state that $S(a, b) = |D(b)-D(a)|/\|b-a\|_2$ vanishes on equal-height flanks ($D(\mu+ru)=D(\mu-ru)$) and underestimates $\|\nabla_x D\|_2$ by $\Theta(\sqrt{d})$ along isotropic random directions.
2. **Assumptions:** Radial/even symmetry for flank cancellation; isotropic direction $u \sim \mathcal{U}(S^{d-1})$ or normalized Rademacher for $\mathbb{E}[(g^\top u)^2] = \|g\|_2^2 / d$.
3. **Existing supporting evidence:** Exact algebraic proof + `t2-secant-v1` empirical failure on `gaussian1d_smoke` and `five_word_joint_smoke`.
4. **Potential counterexamples / ironic self-application:** Unscaled `s_chord`, `s_ff`, `s_rr` retained inside `t2_secant`/`unified`'s image `amax` envelope (`_t2_critic/penalty.py:830,846,860`) **themselves suffer the exact `452.6×` unscaled attenuation at `256×256×3`**!
5. **Minimal falsification experiment:** Evaluate $S(-e_1, e_1)$ on $D(x) = -\alpha \|x\|_2^2$ (`0.0` vs. true `2*alpha`) and $S(x_r, x_f)$ on linear $D(x) = w^\top x$ ($\|w\|=2$) across $d \in [2, 196608]$.
6. **Relevant prior research:** Concentration of measure on $S^{d-1}$ (Vershynin, *High-Dimensional Probability*, §3.4); Gulrajani et al. (2017) *Improved Training of Wasserstein GANs* (interpolated gradient penalties); Petzka et al. (2018) *On the Regularization of Wasserstein GANs*.
7. **Computational verification:** Verified in Exp 3C and `test_micro_secant_quantization_underflow_and_chord_failure_modes` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim C6 — "Householder spatial-frame (`5 <= d <= 8`) has zero directional variance, and local $k$-NN Gaussian Hutchinson pooling (`d > 8`) provides low-bias norm/gradient control for nonlinear Born–Infeld/EBI penalties"
- **Classification:** **Falsified claim** *(valid only for constant-gradient linear critics in `5 <= d <= 8`; falsified on nonlinear critics and nonlinear outer penalties)*
1. **Precise formulation:** [`t2-efficient-critic.md:16,55-60`](../forge/studies/t2-efficient-critic.md#L55-L60) and [`_t2_critic/penalty.py:15-17`](../../particlegan/_t2_critic/penalty.py#L15-L17) claim exact zero-variance reconstruction for $d \le 8$ via chained Householder frames across nearest basis-slot neighbors, and unbiased/low-error Hutchinson reconstruction for $d > 8$ via $k=32$ local Gaussian pooling.
2. **Assumptions:** Implicitly treats $\nabla_x D(x_{j_k}) \approx \nabla_x D(x_i)$ across neighbors AND overlooks Jensen's inequality when composing a noisy $\hat{s}$ with strictly concave $\phi_\beta(s)$ or strictly convex-on-cap $h_{\beta,\kappa}(s)$.
3. **Existing supporting evidence:** Exact zero error on linear critics (`cos = 1.0000, rel_l2 = 0.0000` at $d=6, 8$); $k=32$ pooling doubles single-step parameter-gradient cosine on 170D MLPs (`0.262 -> 0.525`) and clears `five_word_joint_smoke`/`hold` in Forge.
4. **Potential counterexamples:**
   - **Householder spatial-mixing bias:** On nonlinear `LeakyReLU` MLPs ($d=6, 8$), borrowing basis slots from neighbors yields `rel_l2 = 0.341` (`d=6`) and `0.415` (`d=8`), and $Q = H_2 H_1$ fixes a $(d-2)$-dim subspace rather than sampling $O(d)$.
   - **Opposite-signed Jensen bias:** Theorem 3 + Exp 2A/2B show `-14.8%..-23.3%` loss/gradient attenuation (+ `13.3°` directional rotation) for concave Born–Infeld (`minimal`) and **`1,015×` penalty / `275.8×` gradient false-triggering** at $r=0.8 < \kappa=1.0$ (`45.9×` / `11.1×` at $r=1.0$) for convex EBI (`unified_v11`/`v12`, `t2_secant`).
   - **Spatial curvature bias of $k=32$ pooling:** On quadratic $D(x) = \frac{c}{2}\|x\|^2$ ($d=32$), $k=32$ pooling flips EBI bias from `+7,594%` to **`-69.1%`** ($c=0.15$) and `+17.7%` to **`-18.0%`** ($c=0.70$).
5. **Minimal falsification experiment:** Compare `_prepare_spatial_tangent_plan` on linear vs. 3-layer LeakyReLU MLP at $d=8$, and evaluate $\mathbb{E}[\phi_\beta(\hat{s})]$ & $\mathbb{E}[h_{12,1}(\hat{s})]$ over Rademacher probes.
6. **Relevant prior research:** Hutchinson (1990) *A Stochastic Estimator of the Trace of the Influence Matrix*; Avron & Toledo (2011) *Randomized Algorithms for Estimating the Trace of an Implicit Symmetric Positive Semi-Definite Matrix*; Jensen's inequality in stochastic composite optimization (Wang et al., 2017, SCGD).
7. **Computational verification:** Verified in Exp 1C, Exp 2A–2E, `test_householder_frame_geometry_and_spatial_mixing_bias`, `test_jensen_inequality_concave_born_infeld_exact_and_mc`, and `test_jensen_inequality_convex_ebi_positive_bias_and_false_triggering` (all PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** on exact bias/variance bounds; open empirical question whether replacing 1-probe EBI with an unbiased/debiasing or $K=2$ orthogonal-probe estimator improves discrete-image tasks (`img_stripes2`, `img_bars4`).

### Claim C7 — "Separable Rademacher image perturbations (`t2_secant`/`unified_v11` 2-way and `minimal` 3-way) preserve isotropic Hutchinson variance $2\|\nabla_x D\|_2^4$"
- **Classification:** **Falsified claim**
1. **Precise formulation:** [`_t2_critic/penalty.py:145-159`](../../particlegan/_t2_critic/penalty.py#L145-L159), [`_unified_v11/penalty.py:85-94`](../../particlegan/_unified_v11/penalty.py#L85-L94), and [`_minimal/geometry.py:56-60`](../../particlegan/_minimal/geometry.py#L56-L60) substitute rank-1 Kronecker-factored sign tensors for dense Rademacher probes while citing the dense variance formula $2\|\nabla_x D\|_2^4$ ([`t2-efficient-critic.md:42,155`](../forge/studies/t2-efficient-critic.md#L155)).
2. **Assumptions:** Implicitly assumes first-moment unbiasedness ($\mathbb{E}[u u^\top] = \frac{1}{d} I_d$) implies matching fourth-moment variance!
3. **Existing supporting evidence:** Correctly preserves first-moment unbiasedness $\mathbb{E}[d(g^\top u)^2] = \|g\|_2^2$ and matches dense variance `~2.0` when $G$ is full-rank isotropic ($r_{\text{eff},2}(G) \gg 1$).
4. **Potential counterexamples:** Theorem 4A proves exact variance scales as $(2 + 6/r_{\text{eff},2}(G))\|G\|_F^4$ for 2-way separable probes (`7.41` vs. `1.96` = **`3.78×` blowup** on rank-1 $3\times 32\times 32$ stripes; asymptotic **`4×`**) and **`15.88` vs. `1.91` (`8.31×`; asymptotic `13×`)** for `minimal`'s 3-way probe — plus `minimal` generates identically collinear probes across all batch rows (`|cos(u_i, u_j)| = 1.000000`)!
5. **Minimal falsification experiment:** Compute variance of $(a^\top G b)^2$ vs. $(\epsilon^\top \operatorname{vec}(G))^2$ on rank-1 $G = p q^\top$ ($\|G\|_F = 1$) and inspect pairwise cosines of `minimal_unit_probes` across batch rows.
6. **Relevant prior research:** Meyer et al. (2021) / Epperly et al. (2024) variance of structured/Khatri–Rao/Kronecker Hutchinson trace estimators; Sun et al. (2018) *Tensor Random Projection*.
7. **Computational verification:** Verified in Exp 2D and `test_separable_rademacher_variance_blowup_and_minimal_probe_collinearity` (including exact $3 \times 4$ enumeration to `< 1e-13`, PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim C8 — "Eliminating RoR `create_graph=True` delivers substantial latency, VRAM, and GPU energy savings on high-resolution (`128²`–`256²`) Conv2d critics"
- **Classification:** **Empirically supported causal mechanism**
1. **Precise formulation:** Replacing RoR `autograd.grad(..., create_graph=True)` on `128×128` and `256×256` `DCGANCritic` with 1-probe interleaved `fwAD` (`_eval_image_fwad_sq`, `BN.eval()` + `_bn_dual_bf16_hook`) substantially cuts critic penalty wall time, peak VRAM, and hardware GPU energy (mJ/step).
2. **Assumptions:** Compute-bound high-resolution Conv2d (`>= 128×128`), $K = 1$ probe, `BatchNorm2d` evaluated in `.eval()` mode, RTX A6000 / cuDNN backend.
3. **Existing supporting evidence:** Controlled Forge Arm 7 (`-fwad-critic`, `readout-4b04c4bf`: `89.66` vs. `133.01 ms/step`, `-477.7 MiB` VRAM); `law4-optimizer-laws.md:306`; our isolated RTX A6000 Exp 4B benchmark showing **`3.64×` faster penalty (`12.40` vs. `45.15 ms`), `-33.1%` peak alloc VRAM (`576.5` vs. `861.7 MiB`), and `-66.4%` NVML energy (`3,799` vs. `11,301 mJ/iter`)** at `256×256` BF16 CL (and `1.41×` pure RoF-vs-RoR speedup even at matched full-batch `BN.eval()`!).
4. **Potential counterexamples / scope boundaries:** Disappears or inverts below `128×128` at matched sample counts (`0.52×–0.63×` at `32²`–`64²`), disappears at `K >= 4` probes even at `256×256` (`49.63` vs. `45.15 ms`), and depends partly (`1.30×` of the `3.64×`) on switching BN to `.eval()` and (`1.98×`) on halving evaluated batch rows via interleaving!
5. **Minimal falsification experiment:** Run Exp 4B (`_bench_conv_res_row(256, "bf16_channels_last", ...)`), comparing `ror_bn_train`, `ror_bn_eval`, `fwad_1probe_both`, `fwad_1probe_interleaved`, and `fwad_4probe_interleaved`.
6. **Relevant prior research:** PyTorch forward-mode AD (`fwAD`) for Jacobian-vector products; non-fused cuDNN double-backward convolution overhead documented in WGAN-GP / R1-GP systems literature.
7. **Computational verification:** Verified in Exp 4B with synchronized CUDA timing, peak memory stats, and hardware NVML mJ counters.
8. **Confidence & remaining uncertainty:** **High (95%)** on RTX A6000 / PyTorch 2.15; hardware- and compiler-dependent (e.g., future `torch.compile` / cuDNN fused double-backward kernels or Hopper/Blackwell GPUs could shift the `128²` crossover resolution).

### Claim C9 — "Forward-mode `fwAD` universally dominates reverse-over-reverse `autograd.grad(create_graph=True)` across dimensions, architectures, and low-precision operators"
- **Classification:** **Falsified claim**
1. **Precise formulation:** Generalization in [`docs/four-laws.md:43,51,113`](../../docs/four-laws.md#L43) implying `fwAD` is uniformly superior to RoR in compute/VRAM and compatible with low-precision and general architectures.
2. **Assumptions:** Assumes 1-probe `fwAD` has negligible dispatch overhead, exact basis is cheap, and PyTorch `fwAD` has universal operator coverage.
3. **Existing supporting evidence:** Valid **only** in the compute-bound high-resolution Conv2d regime ($K=1$, $\ge 128^2$); `fwAD` does not even unlock Flash/MemEff SDPA, where both `fwAD` and `RoR` fail.
4. **Potential counterexamples (all verified in Exp 4A–4C):**
   - **All GPU MLPs ($d \in [2, 1024]$, $B=256$, FP32):** Exact RoR (`1.32–1.59 ms`) beats BOTH `fwAD_1probe` (`2.38–2.78 ms`, `1.66×–2.05×` slower) AND `fwAD_exact_basis` (`1.98×` slower at $d=2$, `125.4×` slower / `104.6×` VRAM at $d=1024$)!
   - **Low-res Conv2d (`32²`, `64²`) at matched sample count:** `fwAD1_both` is `1.6×–1.9×` slower than `RoR_eval`.
   - **Multi-probe `fwAD` (`K=4`) at `256²` BF16:** `49.63 ms` > `RoR_train` (`45.15 ms`) and `RoR_eval` (`34.62 ms`).
   - **`ResNet_64_Skip_GroupNorm` (FP32 & BF16):** `fwAD` is `1.32×–1.78×` slower AND uses **`2.09×–2.62×` MORE VRAM** (`207` vs. `79 MiB` BF16)!
   - **Four PyTorch 2.15 `fwAD` operator crashes where RoR succeeds:** (i) `DCGAN` BF16 without `_bn_dual_bf16_hook`, (ii) `Attention_SDPA_math` FP32 & BF16 (in-place `Exp` error in `.backward()`), (iii) `EmulatedLinear` `mxfp8`, (iv) `EmulatedLinear` `nvfp4`. (And default Flash/MemEff SDPA crashes BOTH `fwAD` and `RoR`!)
5. **Minimal falsification experiment:** Run Exp 4A ($d=4$ and $d=64$ MLP) and Exp 4C (`ResNet_64_Skip_GroupNorm`, `Attention_SDPA_math`, `EmulatedLinear_mxfp8`).
6. **Relevant prior research:** Griewank & Walther (2008) Ch. 3 & 13 (reverse mode is $O(1)$ in input dimension $d$ for scalar-output $f: \mathbb{R}^d \to \mathbb{R}$, whereas exact forward mode is $O(d)$ — the classic Cheap Gradient Principle!).
7. **Computational verification:** Verified in Exp 4A–4C, `test_emulated_linear_fwad_unsupported_while_ror_succeeds`, and `test_cuda_bn_bf16_dual_hook_and_sdpa_backend_boundaries` (all PASS).
8. **Confidence & remaining uncertainty:** **100% certainty** for the $O(d)$ vs. $O(1)$ exact-gradient scaling law and verified PyTorch 2.15 RTX A6000 crossover regimes.

### Claim C10 — "1-probe `fwAD`/secant EBI achieves strictly higher convergence quality than exact RoR EBI (Arm 7 `-fwad-critic`)"
- **Classification:** **Empirical correlation** *(confounded across multiple uncontrolled axes)*
1. **Precise formulation:** [`docs/four-laws.md:141,182`](../../docs/four-laws.md#L141) and [`r4-occam-ablation-razor.md:20,130-131`](../forge/studies/r4-occam-ablation-razor.md#L130-L131) observe that replacing `unified-v10`'s `fwAD`/secant critic with exact RoR EBI (`r4-razor-autograd-critic`, `readout-b96faf85`) regressed `grid100`, `img_bars4`, and `zero_tanh_rail_escape` while passing `t3_iso_scale_mass_8x` (`11/15` vs. `14/16`).
2. **Assumptions:** Interprets the single-seed pass-rate gap on `unified-v10` hyperparameters as intrinsic superiority of `fwAD`/secants over exact $\|\nabla_x D\|_2$.
3. **Existing supporting evidence:** Verbatim Forge readouts `readout-b96faf85` vs. `readout-6da9dee1` (`seed = 0`).
4. **Potential counterexamples / confounds:** Arm 7 did **not** hold the mathematical objective fixed while changing only the AD engine:
   - On `grid100` (`d = 2`) and `zero_tanh_rail_escape` (`d = 1`), **canonical `fwAD` and exact RoR compute the exact same mixed derivative to `3e-7` FP32** (Exp 1B)! Therefore, any regression on `grid100` (`d=2`) or `zero_tanh_rail_escape` (`d=1`) under Arm 7 CANNOT be caused by "first-order vs. second-order differentiation" — it comes from **other code differences** between `_UnifiedPenaltyKernel` and `_T1CouplingPenaltyKernel` (e.g., gauge thermal gating `s_gauge` at [`_unified/penalty.py:762-766`](../../particlegan/_unified/penalty.py#L762-L766) absent in `_T1CouplingPenaltyKernel:263-266`, `observe_scale` calls, or chord-phase ordering)!
   - On `img_bars4` (`8×8`, `d=64`), `unified-v10` used $K=2$ micro-secants + 4-NN smoothing + convex Jensen over-penalization, which alters the effective regularization strength relative to exact RoR with the same `coeff` and `kappa`.
   - Conversely, on `t3_iso_scale_mass_8x`, exact RoR **fixed** `unified-v10`'s failure (`eig = 0.5313 PASS` vs. `0.2155 FAIL`).
5. **Minimal falsification experiment:** Run an apples-to-apples ablation on `grid100` (`d=2`) and `img_bars4` (`d=64`) holding the exact penalty kernel identical (same `s_gauge`, same `observe_scale`, same `BN.eval()`) and swapping *only* `_eval_lowd_forward_grad_sq` vs. `torch.autograd.grad(..., create_graph=True)` (and sweeping `coeff` for `img_bars4`).
6. **Relevant prior research:** Standard experimental design for separating AD implementation effects from objective/hyperparameter shifts.
7. **Computational verification:** Exp 1B proves exact `3.16e-7` FP32 parity between canonical `fwAD` and RoR at `d <= 4`, directly establishing that Arm 7's `d <= 2` pass-rate differences stem from kernel/gauge discrepancies rather than AD order.
8. **Confidence & remaining uncertainty:** **High (90%)** that Arm 7's `d <= 2` differences are kernel-mismatch confounds; **unresolved** whether the implicit stochastic over-penalization of 1-probe `fwAD` on images acts as a beneficial regularizer or merely shifts effective `kappa`/`coeff`.

---

## 5. Computational & Practical Regime Guide (Decision Matrix)

| Regime | Recommended Critic Penalty AD Path | Why (Verified Trade-Off) |
| :--- | :--- | :--- |
| **Low-to-medium GPU MLPs (`d >= 2`, `B <= 1024`, FP32)** | **Exact RoR** (`autograd.grad(create_graph=True)`) | `1.7×–2.0×` faster than even 1-probe `fwAD`, `2×–125×` faster than exact `fwAD`, zero variance, zero Jensen bias. *(Canonical `fwAD` is only justified if a hard constraint forbids `create_graph=True`.)* |
| **High-res Conv2d (`>= 128×128`, BF16/FP32, `BN.eval()`)** | **1-probe interleaved `fwAD`** (`_eval_image_fwad_sq` + `_bn_dual_bf16_hook`, **dense Rademacher** where affordable or 2-way sep. with variance awareness) | **`2.0×–3.6×` faster, `-33%` VRAM, `-66%` GPU energy** vs. `RoR_train`; `0` kink flips (`grad_cos = 0.987–0.996`); strictly dominates BF16 micro-secants (which collapse to `cos ~ 0`). Watch for convex-EBI positive Jensen bias below $\kappa$. |
| **Low-res Conv2d (`32²`–`64²`) & `ResNet + GroupNorm`** | **Exact RoR** (`autograd.grad(create_graph=True)`) | `1.3×–1.9×` faster and up to **`2.6×` LESS VRAM** than `fwAD`, with zero variance/Jensen bias. |
| **Transformer / Attention Critics (`SDPA`)** | **RoR with `SDPBackend.MATH`** (if exactness needed) OR **carefully scaled FP32/central secant** (if Flash/MemEff required; **never** tiny BF16 $\delta=0.02$!) | Default Flash/MemEff SDPA crashes BOTH `fwAD` and `RoR`; even `SDPBackend.MATH` crashes `fwAD` + `.backward()` in PyTorch 2.15 while `RoR` succeeds! |
| **Custom Quantized Ops (`EmulatedLinear` MXFP8/NVFP4)** | **Exact RoR (via STE `backward`)** OR **scale-aware central FD** (`delta >= 0.06`, accepting kink/curvature noise) | Custom `autograd.Function` without `jvp` breaks `fwAD`; RoR works natively through `_quantized_proxy` STE. |
| **NEVER USE** | **BF16 micro-secants (`delta=0.02` in `t2_secant`/`unified`, `eps=1e-5` in `minimal`)** on high-res ReLU/LeakyReLU Conv2d | `75%–99.99%` pixel quantization loss + thousands of LeakyReLU flips -> **`grad_cos ~ 0.000`** and **`255×–6.7×10^{10}×` norm distortion**! |

---

## 6. Disciplined Reformulation of Law 2 & Exact Proposed Diffs

### 6.1 Mathematically Disciplined Reformulation of Law 2

> **Law 2 — Critic Potential Geometry & Mixed-Derivative Regularization Efficiency:**
> Adversarial transport relies on a critic potential $D_\theta(x)$ whose local spatial slope $\|\nabla_x D_\theta(x)\|_2$ stays bounded ($\le \kappa$) under an anchored logit gauge. Differentiating any smooth penalty $\Phi(\|\nabla_x D_\theta(x)\|_2^2)$ with respect to $\theta$ **necessarily computes mixed second-order derivatives** $\frac{\partial^2 D_\theta}{\partial \theta\,\partial x}\,\nabla_x D_\theta$ (nonzero even on piecewise-linear ReLU/LeakyReLU networks where $\frac{\partial^2 D}{\partial x^2} = 0$ a.e.).
> 1. **Geometric Obstructions of Naive Finite-Difference Surrogates:** Replacing exact spatial derivatives by macro chords $|D(b)-D(a)|/\|b-a\|_2$ or finite-difference micro-secants $(D(x+\delta u)-D(x))/\delta$ fails for four distinct reasons: (a) **symmetric-flank cancellation** ($S = 0$ on equal-height flanks as $\|\nabla_x D\| \to \infty$), (b) **$\Theta(\sqrt{d})$ unscaled projection attenuation**, (c) **$O(1/\delta)$ ReLU/LeakyReLU kink impulses** (affecting both forward and central differences whenever a perturbation straddles an activation hyperplane), and (d) **low-precision quantization collapse** whenever per-pixel step $\delta/\sqrt{d} \ll \mathrm{ULP}_{\text{BF16}}(x)$.
> 2. **Reverse-over-Forward (RoF) vs. Reverse-over-Reverse (RoR) Trade-Offs:** Computing directional mixed second derivatives via forward-mode tangents (`fwAD.dual_level` + `.backward()`, RoF) evaluates at the exact primal activation mask (zero kink impulses, zero BF16 pixel underflow) and eliminates reverse-over-reverse (`create_graph=True`) double-backward convolution graphs — yielding large speed, VRAM, and energy wins (`2×–3.6×` faster, `-33%` VRAM, `-66%` GPU energy) in the **compute-bound high-resolution Conv2d regime ($\ge 128^2$, $K = 1$)**. Conversely, by the Cheap Gradient Principle, exact RoR stays $O(1)$ in input dimension $d$ and dominates `fwAD` on small/medium GPU MLPs ($d \in [2, 1024]$), low-res Conv2d ($32^2$–$64^2$), `ResNet+GroupNorm`, multi-probe $K \ge 4$, and operators lacking PyTorch `fwAD` support (math SDPA, custom quantized `autograd.Function`).
> 3. **Nonlinear Jensen Bias & Structured-Probe Variance:** Whenever a stochastic $K$-probe directional estimator $\hat{s}$ ($\mathbb{E}[\hat{s}] = \|\nabla_x D\|_2^2$) is passed through a nonlinear outer penalty $\Phi(\hat{s})$, **Jensen's inequality introduces systematic bias**: strictly concave Born–Infeld $\phi_\beta(s) = (\sqrt{1+\beta s}-1)/\beta$ incurs strict negative loss and parameter-gradient bias (up to $-20\%$ to $-23\%$ at $K=1$, plus Rademacher directional distortion), whereas convex-on-cap squared-softplus EBI $[\operatorname{softplus}_\beta(\sqrt{s}-\kappa)]^2$ incurs strict positive Jensen bias and false cap-triggering below $\kappa$. Separable Kronecker Rademacher probes further inflate variance by up to $4\times$ (2-way) or $13\times$ (3-way) on low-rank spatial gradients relative to dense Rademacher.

### 6.2 Exact Proposed Diffs (For Coordinator / P10 Integration — NOT Applied Here)

#### (1) Proposed diff for `docs/four-laws.md` (lines 40–57, 113, 133–141, 182)

```diff
--- a/docs/four-laws.md
+++ b/docs/four-laws.md
@@ -40,20 +40,21 @@
 ### Law 2 — Critic Potential Geometry
 
 **Core Theory (Fundamental Obstruction):**
-Adversarial transport relies on a critic potential $D(x)$ whose local spatial slope $\|\nabla_x D(x)\|_2$ stays bounded ($\le \kappa$) under an anchored logit gauge. Without slope control, gradients explode, collapse, or drift; yet enforcing slope regularity through second-order reverse-mode differentiation (`create_graph=True`) multiplies compute and VRAM several-fold and inhibits fused kernels on high-dimensional or convolutional critics. Conversely, naive finite-difference slope surrogates $|D(b) - D(a)|/\|b - a\|_2$ fail for three fundamental geometric reasons:
+Adversarial transport relies on a critic potential $D_\theta(x)$ whose local spatial slope $\|\nabla_x D_\theta(x)\|_2$ stays bounded ($\le \kappa$) under an anchored logit gauge. Differentiating any slope penalty $\Phi(\|\nabla_x D_\theta(x)\|_2^2)$ w.r.t. $\theta$ inherently computes **mixed second-order derivatives** $\frac{\partial^2 D_\theta}{\partial \theta\,\partial x}\,\nabla_x D_\theta$. Without slope control, critic gradients explode, collapse, or drift; yet computing mixed derivatives via **reverse-over-reverse (RoR)** double-backpropagation (`create_graph=True`) multiplies compute, VRAM, and GPU energy on high-resolution convolutional critics (`>= 128^2`). Conversely, naive finite-difference slope surrogates fail for four fundamental geometric and numerical reasons:
 1. **Symmetric-Flank Chord Cancellation:** Chords spanning equal-height flanks satisfy $D(\mu + r u) - D(\mu - r u) = 0$ even as $\|\nabla_x D\|_2 \to \infty$.
 2. **High-Dimensional Projection Attenuation:** A single random unit probe $\hat{u} \in S^{d-1}$ underestimates $\|\nabla_x D\|_2$ by $\Theta(\sqrt{d})$.
-3. **Piecewise-Linear Kink Impulses:** Perturbations straddling ReLU/LeakyReLU hyperplanes inject $O(1/\delta)$ parameter-gradient noise.
+3. **Piecewise-Linear Kink Impulses:** Forward or central finite-difference perturbations straddling ReLU/LeakyReLU hyperplanes inject $O(1/\delta)$ parameter-gradient noise.
+4. **Low-Precision Quantization Underflow & Nonlinear Jensen Bias:** In BF16, micro-secant pixel steps $\delta/\sqrt{d} \ll \mathrm{ULP}_{\text{BF16}}(x)$ quantize away (`95%+` pixel loss at `256^2`), while composing any noisy $K$-probe directional estimate $\hat{s}$ with nonlinear penalties incurs opposite-signed **Jensen bias** (negative attenuation for concave $\sqrt{1+\beta s}-1$; positive false-triggering below $\kappa$ for convex-on-cap squared-softplus EBI) and up to $4\times$ variance blowup under separable Rademacher probes on low-rank spatial gradients.
 
 **What Any Valid Solution Must Satisfy:**
 1. **Accurate Local Slope & Gauge Control:** Bound $\|\nabla_x D(x)\|_2 \le \kappa$ and fix the arbitrary additive logit constant across smooth and piecewise-linear critics.
-2. **Geometric Faithfulness:** Avoid symmetric-flank cancellation, $\sqrt{d}$ high-dimensional attenuation, and cross-kink gradient impulses.
-3. **First-Order / Low-Energy Efficiency:** Achieve slope regularity without paying higher-order graph compute/VRAM overhead, while remaining compatible with low-precision arithmetic and clock-free training.
+2. **Geometric & Statistical Faithfulness:** Avoid symmetric-flank cancellation, $\sqrt{d}$ attenuation, cross-kink impulses, and BF16 micro-secant underflow, while bounding directional variance and nonlinear Jensen bias.
+3. **Regime-Matched Mixed-Derivative Efficiency:** Eliminate reverse-over-reverse (`create_graph=True`) graph overhead where it dominates (high-resolution Conv2d via reverse-over-forward `fwAD` tangents), while recognizing the Cheap Gradient Principle crossover where exact RoR dominates (small/medium GPU MLPs, low-res Conv2d, `ResNet+GroupNorm`, multi-probe $K \ge 4$, and ops lacking `fwAD` support such as SDPA and custom quantized `autograd.Function`).
 
 **How to Test / Falsify a Candidate:**
 - Matched latency, VRAM, and hardware energy scaling on high-resolution convolutional critics (`dcgan_performance_r256`).
-- High-dimensional structured targets (`five_word_joint_hold`) sensitive to $\sqrt{d}$ attenuation.
-- Reduced-precision training (`mxfp8`, `nvfp4`) and direct Pareto ablations replacing the candidate regularizer with a second-order gradient penalty while holding all other components fixed.
+- High-dimensional structured targets (`five_word_joint_hold`) sensitive to $\sqrt{d}$ attenuation and directional variance.
+- Reduced-precision training (`mxfp8`, `nvfp4`), direct gradient-cosine audits against exact RoR, and matched kernel-identical Pareto ablations comparing RoF (`fwAD`) vs. RoR (`create_graph=True`).
```

```diff
--- a/docs/four-laws.md
+++ b/docs/four-laws.md
@@ -110,7 +110,7 @@
 | Law | Primary Solutions Found So Far | Lighter / Earlier Alternatives | Known Open Gaps |
 | :--- | :--- | :--- | :--- |
 | **Law 1** *(Measure & Prior Topology)* | Streaming Laguerre–Pythagorean prototype atlas + LTV coalescence + quota-balanced Laguerre assignment + orthogonal pullback covariance isometry (`t1_prior`, `t1_coupling`, `unified_v11`) | SBNP (`bispf`, `rhbi`), online *e-means* (`empf-v8`), compact `<900`-LOC core (`minimal-v2`) | High host-sync overhead (`113` `cudaStreamSync`/step in pullback on `grid100`); `staggered100` live-suffix margin under default `unified_v11` |
-| **Law 2** *(Critic Potential Geometry)* | First-order `fwAD` dual-tangent reconstruction (MLPs) + live-endpoint reuse & separable BF16 micro-secant Born–Infeld with zero-mean gauge (`t2_secant`, `unified_v11`, `minimal`) | Second-order `autograd.grad` penalties (`BCap`, `R1/R2`, `EBI-TCR`, `EMPF`, `RHBI` — `1.48×–55×` slower) | Amortized or cheaper first-order slope estimation on high-res Conv2d (where `d_penalty` is `~15.8%` of step time once optimizer cost is cut) |
+| **Law 2** *(Critic Potential Geometry)* | Reverse-over-forward (RoF) `fwAD` mixed-derivative EBI across MLPs AND images (`_eval_image_fwad_sq` + `_bn_dual_bf16_hook`, zero `create_graph=True`, zero image micro-secants in `unified_v11`/`v12`) with live-endpoint zero-mean gauge | Legacy image BF16 micro-secants (`t2_secant`, `unified_v4`, `minimal` — vulnerable to BF16 pixel underflow & kink flips); RoR `autograd.grad(create_graph=True)` penalties (`BCap`, `R1/R2`, `EMPF`, `RHBI` — faster on small MLPs/`ResNet+GN`, `1.48×–55×` slower on `r256`) | Reducing 1-probe convex-EBI Jensen bias & separable Rademacher variance on high-res Conv2d without losing the `3.6×` `r256` RoF speedup; PyTorch `fwAD` support for attention (`SDPA`) and custom quantized ops |
```

```diff
--- a/docs/four-laws.md
+++ b/docs/four-laws.md
@@ -133,12 +133,12 @@
 ### Solutions to Law 2 (Critic Potential Geometry)
 
-- **Forward-Mode Tangent Reconstruction (`fwAD.dual_level`, manifolds & MLPs)** ([`t2-efficient-critic.md`](../reports/forge/studies/t2-efficient-critic.md)): Evaluates $(D_\theta(x), \nabla_x D_\theta(x) \cdot u)$ at the **exact primal activation mask** (zero kink impulses, zero `create_graph=True`, single 1st-order `.backward()`): exact canonical basis $I_d$ for $1 \le d \le 4$ (with central-difference fallback for quantized MXFP8/NVFP4 ops lacking `jvp`), chained $O(d)$ Householder frames for $5 \le d \le 8$, and RNG-free Murmur3 Rademacher tangents with local $k$-NN Gaussian Hutchinson pooling ($k=32$) for $d > 8$.
-- **Live-Endpoint Reuse + Separable BF16 Micro-Secants (Convolutional Image Critics):** Captures the live 1st-order logits $D(x_{\text{real}}), D(x_{\text{fake}})$ from `d_loss` via a forward hook at zero extra forward cost and evaluates a single BF16 micro-secant probe batch alongside free transport-chord and cyclic-roll secants, with $8 \times 8$ pooled Chamfer scale tracking and a Born–Infeld slope penalty with zero-mean logit gauge (`ebi_beta`).
+- **Reverse-over-Forward Mixed-Derivative Reconstruction (`fwAD.dual_level` + `.backward()`, manifolds & MLPs)** ([`t2-efficient-critic.md`](../reports/forge/studies/t2-efficient-critic.md)): Evaluates $(D_\theta(x), \nabla_x D_\theta(x) \cdot u)$ at the **exact primal activation mask** (zero kink impulses, zero `create_graph=True` RoR graphs; computes exact mixed second derivatives $\frac{\partial^2 D_\theta}{\partial \theta \partial x} u$ via a single `.backward()`): exact canonical basis $I_d$ for $1 \le d \le 4$ (with central-difference fallback for quantized MXFP8/NVFP4 ops lacking `jvp`), chained 2-Householder frames ($Q=H_2 H_1$) with spatial-neighbor reconstruction for $5 \le d \le 8$, and Murmur3 Rademacher tangents with local $k$-NN Gaussian Hutchinson pooling ($k=32$) for $d > 8$.
+- **Convolutional Image Critics — `unified_v11`/`v12` Interleaved `fwAD` vs. Legacy `t2_secant` Micro-Secants:** In `unified_v11` and default `unified_v12` ([`_unified_v11/penalty.py:404-490,631-695`](../particlegan/_unified_v11/penalty.py#L404-L490)), convolutional image critics use **1-probe interleaved `fwAD.dual_level()`** (`_eval_image_fwad_sq`, switching `BatchNorm2d` to `.eval()` with `_bn_dual_bf16_hook`, zero kink flips, `grad_cos = 0.987–0.996`) while reusing live `d_loss` endpoints solely for the zero-mean logit gauge. Earlier `t2_secant` / `unified_v4` / `t4_nash_v3` / `minimal` instead evaluated live-reused BF16 micro-secants (`delta=0.02` or `1e-5`) plus transport/cyclic chords, which P6 showed suffer severe BF16 pixel quantization underflow and LeakyReLU kink impulses (`grad_cos ~ 0`).
 
 **Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
 - `t2-secant-v2` ([`readout-14cc2a44`](../reports/forge/records/readout-14cc2a44f2d8da942d2873dc.json)): cuts uncontended `dcgan_performance_r256` median latency to **`87.96 ms/step`** (`181.92 img/s`, `1,308.46 MiB` allocated VRAM) — **`1.48×–1.97×` faster than `empf-v8`**, **`3.87×–3.97×` faster than `bispf`**, **`6.69×–6.82×` faster than `rhbi-v12`** (`-72.2%` reserved VRAM), and **`35.4×–55.1×` faster than `bcap`**.
 - `t2-secant-v5` ([`readout-a2d41a67`](../reports/forge/records/readout-a2d41a678887a49f9ecc6aa9.json)): clears `11/11` required Tier-1 gates + `clockfree_audit` + `precision_gaussian_mxfp8` + `five_word_joint_hold` with **zero `torch.autograd.grad` calls**.
-- Round-4 Occam's Razor Arm 7 (`-fwad-critic`, replacing 1st-order `fwAD`/secants with 2nd-order `autograd.grad` EBI on `unified-v10`, [`readout-4b04c4bf`](../reports/forge/records/readout-4b04c4bf0dc3d7408a82e51e.json), [`readout-b96faf85`](../reports/forge/records/readout-b96faf85499b1dbe9dce102e.json)): demonstrated 1st-order `fwAD`/secant Born–Infeld is **strictly Pareto-dominant** over 2nd-order EBI — `1.48×` faster (`89.66` vs `133.01 ms/step`), `-477.7 MiB` (`-34.8%`) less VRAM (`1372.6` vs `1850.3 MiB`), and higher pass rate (`grid100`, `img_bars4`, and `zero_tanh_rail_escape` all regressed under 2nd-order EBI).
+- Round-4 Occam's Razor Arm 7 (`-fwad-critic`, replacing RoF `fwAD`/secants with RoR `autograd.grad` EBI on `unified-v10`, [`readout-4b04c4bf`](../reports/forge/records/readout-4b04c4bf0dc3d7408a82e51e.json), [`readout-b96faf85`](../reports/forge/records/readout-b96faf85499b1dbe9dce102e.json)): confirmed `1.48×` faster `r256` step time (`89.66` vs `133.01 ms/step`) and `-477.7 MiB` (`-34.8%`) less VRAM (`1372.6` vs `1850.3 MiB`); pass-rate differences on `d <= 2` (`grid100`, `zero_tanh_rail_escape`) reflect kernel/gauge differences between `_UnifiedPenaltyKernel` and `_T1CouplingPenaltyKernel` rather than derivative order (since canonical `fwAD` and RoR agree to `3e-7` FP32 for `d <= 4`), while exact RoR unlocked `t3_iso_scale_mass_8x`.
```

#### (2) Concrete Engineering Recommendations (Future Work — NOT Applied Here)
1. **Fix `minimal`'s broken `secant_eps = 1e-5` and batch-collinear 3-way probe (`particlegan/_minimal/geometry.py:56-60,124-127`):** Switching `minimal`'s image branch to `_eval_image_fwad_sq` (or fixing `secant_eps` and removing the shared axis-0 batch factor) would eliminate both the dead `r32` BF16 penalty and the `6.7×10^{10}×` FP32/BF16 cancellation spikes.
2. **Use dense Rademacher (`_murmur32_signs`) instead of 2-way separable probes when `d <= 65536` or where spatial gradients are low-rank** to avoid the `4×` rank-1 stripe variance blowup (e.g., `img_stripes2`, `img_bars4`).
3. **Evaluate exact RoR on GPU MLPs & `ResNet+GroupNorm`:** On small/medium MLPs (`d in [2, 1024]`) and `ResNet+GroupNorm`, exact RoR is both faster and variance-/Jensen-bias-free compared to `fwAD`; reserving `fwAD` specifically for compute-bound high-resolution Conv2d (`>= 128²`) aligns the autodiff mode with the Cheap Gradient Principle.
4. **Guard `pool_k == 1` in `_prepare_spatial_tangent_plan` (`particlegan/_t2_critic/penalty.py:240`):** Use `knn_d2[:, min(k_eff - 1, max(1, k_eff // 2))]` (or short-circuit `k_eff == 1`) to avoid `IndexError`.
