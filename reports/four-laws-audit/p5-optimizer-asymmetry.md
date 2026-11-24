# P5 Adversarial Mathematical & Empirical Audit: Law 4.4 (Generator–Critic Optimizer Asymmetry & Adaptive Pathology Conditioning)

> **Audit Scope & Governance:** Adversarial mathematical, code-level, and empirical audit of Law 4.4 (*Generator–Critic Structural Asymmetry under Bounded Critics*) on integration branch `research/four-laws` (audited on branch `research/fl-p5-asymmetry`). All experiments execute through the workspace `particlegan` package under protocol seed `0` and `deterministic_orthogonal_` initialization on `NVIDIA RTX A6000` (`cuda:1`, PyTorch `2.15.0+cu132`) and CPU. Rank, stable-rank, participation-ratio, and Roy–Vetterli `erank` definitions strictly match [P0 (`p0-law43-rank-audit.md`)](p0-law43-rank-audit.md#24-rigorous-hierarchy-of-rank--spectral-diversity-measures). Existing Forge receipts, `docs/four-laws.md`, and `particlegan/` source files are left strictly untouched; exact `git apply --check`-verified proposed diffs are provided in [Section 6](#6-stronger-falsifiable-reformulation-of-law-44--exact-proposed-diffs).
>
> **Reproducibility:**
> - Experiment runner: [`scripts/four_laws_audit/p5_asymmetry_experiments.py`](../../scripts/four_laws_audit/p5_asymmetry_experiments.py) (`PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p5_asymmetry_experiments.py --device cuda:1 --sections 1,2,3`)
> - Automated proofs, code audits & falsification controls: [`tests/four_laws_audit/test_p5_asymmetry.py`](../../tests/four_laws_audit/test_p5_asymmetry.py) (`PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/pytest -v tests/four_laws_audit/test_p5_asymmetry.py`)
> - Lint gate: `PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/check_lint.py` (`0` additions)

---

## Executive Verdict

Law 4.4 as stated in [`docs/four-laws.md:88`](../../docs/four-laws.md#L88), [`reports/forge/studies/law4-optimizer-laws.md:107-115,218-220`](../forge/studies/law4-optimizer-laws.md#L218-L220), and [`reports/forge/studies/law4-energy-champion.md:65-68,90`](../forge/studies/law4-energy-champion.md#L65-L68) — and hardcoded into `UnifiedV12Optimizer` as *"strictly 0 NS on neural-G critics"* — is **FALSIFIED as an unconditional, role-intrinsic law**. Instead, generator–critic optimizer asymmetry is **predominantly pathology-intrinsic, modulated by two genuine (non-universal) role asymmetries and heavily confounded in existing Forge suites by ParticleGAN's direct-OT pullback architecture**:

1. **Direct Adversarial Falsification (Swapping Depth Pathology to $D$ Reverses Law 4.4):**
   - In standard deep-$G$ + shallow-$D$ (`Regime A`: `16L G + 3L D` on `grid100_mlp16`), **`G-only` (`100/100`, `hq = 0.9714`)** and **`Both` (`100/100`, `hq = 0.9697`)** dominate **`Neither` (`34/100`, `hq = 0.1153`)** and **`D-only` (`1/100`, `hq = 0.0151`)** — confirming Law 4.4's empirical observation when depth pathology sits on $G$.
   - Swapping the 16-layer skip-free MLP onto the **critic** in a pure adversarial RpGAN + Law 2 Born–Infeld game (`Regime B1a`: `3L G + 16L D`, `v12`'s $\sqrt{r}$ Frob-MP) **completely reverses the ordering**: **`G-only` FAILS (`7/8` modes, `hq = 0.3701`, valley accuracy `0.520`)** and **`Neither` collapses (`8/8`, `hq = 0.2861`, `||∇D|| = 1.11e-7`)**, whereas **`D-only` (`8/8`, `hq = 0.7947`, `+0.425` over `G-only`)**, **`Adaptive-Pathology` (`8/8`, `hq = 0.8110`)**, and **`Both` (`8/8`, `hq = 0.9055`)** rescue both critic and generator! Even after fixing `v12`'s $\sqrt{r}$ spectral-norm distortion (`Regime B1b`, spectral-norm-matched `G / sigma_1`), **`D-only` (`8/8`, `hq = 0.9482`)** still outperforms **`Neither` (`0.9163`)** and crushes **`G-only` (`7/8`, `0.7285`, `+0.220` gap)**.
   - On a standalone 16-layer skip-free critic with Law 2 BI strictly active (`Regime B` standalone control), `v12`'s 0-NS $\sqrt{r}$ Frob-MP degrades to **`acc_gauss = 0.8750, acc_valley = 0.6396`** (`||∇D|| = 0.074`), whereas **5-step Quintic NS** achieves **`0.9951 / 1.0000`** (`||∇D|| = 0.519`) and **`Adaptive-Pathology`** achieves **`0.9844 / 1.0000`**.
2. **Three Proven Mathematical Mechanisms & One Hidden Benchmark Confounder:**
   - **Theorem 1 (Hypothesis a — Scalar-Potential Linear Rank-1 Bottleneck):** Any $L$-layer linear MLP $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$ has exact gradient rank $\le \min(d_{\text{out}}, B, W)$ at every layer — even with Law 2 gradient penalties active — so scalar critics ($d_{\text{out}} = 1$) pass backprop through a rank-1 output head and rely purely on nonlinear gating across samples for rank $> 1$.
   - **Theorem 2 (Hypotheses a, b, d — Contrastive Mean-Cone Cancellation vs. Deep-Layer Collapse):** On `16L` skip-free `LeakyReLU(0.2)` MLPs, contrastive real-minus-fake subtraction $-(D(x_{\text{real}}) - D(x_{\text{fake}}))$ cancels the shared positive-cone mean mode, slashing inter-layer gradient norm skew **`4,567×`** (`18,820.4× -> 4.1×`, `1.8×` with `fourier=2`) vs. single-sign `G16` (`27,593.1×`), **yet does NOT prevent deep-layer spectral collapse** (`net.24`: $r_{\text{stable}} = 1.000000005$, $\sigma_2/\sigma_1 = 2.85\times 10^{-5}$; `||∇D|| = 4.94e-7`).
   - **Theorem 3 (The $\sqrt{r}$ Spectral-Norm Distortion & Irreducible Frobenius–Spectral Dilemma):** `unified_v12`'s 0-NS critic rule scales Frobenius-MP by $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$, inflating the top singular value by exact factor $\sigma_1(\Delta W^{\text{v12}})/\sigma_1(\Delta W^{\text{Polar}}) = \sqrt{m / r_{\text{stable}}(G_D)}$ (`11.31×` on $128\times 128$ at $r_{\text{stable}} = 1.0$). Conversely, matching spectral norm (`G / sigma_1`) shrinks the Frobenius budget by $\sqrt{r_{\text{stable}}/m}$ (`~0.14×`), causing **Exp 3c (`t3_iso_scale_mass_8x`) to FAIL (`0/5` sustained, `min_eig = 0.3885 < 0.45`)**. Because scalar rescaling locks $\|\Delta W\|_F / \|\Delta W\|_2 \equiv \sqrt{r_{\text{stable}}(G)}$, **no Frobenius-MP scaling can simultaneously match Polar NS in both spectral and Frobenius norm when $r_{\text{stable}} \ll m$**.
   - **Hypothesis (e) — Spatial Force-Field Amplification (`2.9×–7.8×`):** Under equal Frobenius perturbation (`eps = 0.05`), the gradient-orthogonal component of Critic NS perturbs scalar logits `D(x)` by only `3.6%–20.9%` of raw-gradient displacement, **yet perturbs the spatial force field $\nabla_x D(x)$ driving $G$ by `28.1%–61.2%`** (`6.8×` relative amplification in balanced Regime C, `7.2×` in Regime A, `7.8×` in `548×` overpowered-G) — explaining why unnecessary NS on healthy shallow critics agitates $G$ at equilibrium.
   - **Hypothesis (c) — Law 2 One-Sidedness & Orthogonality:** Born–Infeld's upper slope penalty $\operatorname{ReLU}(\|\nabla_x D\| - \kappa)$ is identically zero when a deep skip-free critic contracts to $\|\nabla_x D\| \approx 10^{-7} \ll 1$, so Law 2 **cannot** prevent lower spectral collapse (`Exp 3a`: deep critic + Law 2 ON still fails under `G-only` `hq = 0.432` vs `Both` `0.856`). Conversely, turning Law 2 OFF collapses shallow `3L+3L` from `8/8` modes to `2/8–5/8` (`||∇D|| = 4.76–10.92`) even under full Critic NS — proving Law 2 (upper spatial slope) and Critic NS (lower parameter-space spectral collapse) are **orthogonal, non-interchangeable controls**.
   - **Hidden Confounder in Existing Forge Evidence (`Regime B2`):** Full ParticleGAN (`unified_v12`) includes direct data-space Laguerre–Sinkhorn OT pullback gradients (`UnifiedV11GANLoss` + `UnifiedV11PriorRegularizer`) that bypass $D$, causing ALL 5 arms on `3L G + 16L D` `ring16` to pass identically at **`16/16, hq = 0.9788–0.9858`** (`0.007` spread vs. `0.619` spread in pure adversarial `Regime B1a`), masking deep-critic failure in existing Forge suites (which also placed depth/rail pathologies exclusively on $G$).
3. **Why `unified_v12` Needed a Generator-Free Critic NS Fallback & Three Code/Doc Bugs Caught:**
   - `particlegan/recipes.py:2378-2385` toggles `_generator_free = (total_gen_mats == 0)`, a **syntactically brittle** check where adding a dummy `nn.Linear(2, 2)` turns Critic NS completely off even on a 16-layer skip-free critic!
   - `_step_generator_free_critic` (`particlegan/_unified_v12/optimizer.py:456-639`) silently bundles **three** changes (5-step Critic NS, Tikhonov cap `0.30` vs `0.50 * peak_rms`, hot `peak_rms` decay `0.9995` vs `1.0`). In Exp 3c (`800`-step `t3_iso_scale_mass_8x`), 5-step Critic NS achieves the highest `core_min_eigen_ratio` (`0.5859` vs `0.5353` fallback-OFF, `0.3885` `spec`-matched FAIL, `0.4181` `adaptive` FAIL).
   - **Three concrete code/doc bugs caught:** (i) `docs/four-laws.md:175` and `RESEARCH_GLOSSARY.md:48` link nonexistent `particlegan/unified_v12.py` (actual module: `particlegan/_unified_v12/optimizer.py`); (ii) `particlegan/_unified_v12/optimizer.py:457` docstring claims `"+ Schatten-6"` yet executes **zero** Schatten-6 calls; (iii) `reports/forge/studies/law4-energy-champion.md:90` misquotes `t3_iso_scale_mass_8x`'s `core_min_eigen_ratio` threshold as `0.35` instead of the frozen `0.45` (`configs/forge/tasks/t3_iso_scale_mass_8x.json:99`).
4. **Honest Evaluation of Adaptive Pathology-Triggered Conditioning (`Adaptive-Pathology`, $r_{\text{stable}} < 1.25$):**
   - Being strictly role-agnostic, `Adaptive-Pathology` successfully **rescues BOTH Regime A (`34/100 -> 98/100`, `hq 0.115 -> 0.816`) AND Regime B1a (`hq 0.286 -> 0.811`) AND `620×` `overpowered_critic` (`PASS`, `hq = 0.9067`)**, while saving **`59.4%`** of NS calls on balanced `Regime C1` (`hq = 0.9915`) and **`58.8%`** on `Conv2d img_bars4` (`4/4` PASS).
   - **Key limitation discovered:** Because Theorem 1 gives even *healthy* shallow scalar critics low equilibrium gradient stable rank ($r_{\text{stable}} \in [1.10, 1.35]$), a naive stable-rank-only threshold ($r_{\text{stable}} < 1.25$) **false-triggers on `83.6%` (`3008/3600`) of shallow critic steps in Regime A** and incurs power-iteration overhead (`121.1s` vs `104.1s` for `G-only`), while under-stepping `t3_iso_scale_mass_8x` when paired with `spec` fallback (`0.4181` FAIL). A production pathology gate must therefore condition on **contraction pathology** (severe spectral collapse + inter-layer Jacobian skew / non-equilibrium residual or vanishing $\|\nabla_x D\|$), not $r_{\text{stable}} < 1.25$ alone.

---

## 1. Exhaustive Law 4.4 Claim Audit & Optimizer Code Inventory

### 1.1 Exact Claims Audited Across Documents, Studies & Source Code

Every substantial Law 4.4 claim is classified below using **strictly one** primary label from the 10-category taxonomy ([`reports/four-laws-audit/README.md:27-37`](README.md#L27-L37)); complete 8-point analyses appear in [Section 4](#4-eight-point-structured-analysis-of-all-audited-claims).

| ID | Source File & Exact Lines | Verbatim Claim (Key Excerpt) | Primary Taxonomy Verdict |
| :--- | :--- | :--- | :--- |
| **C1** | [`docs/four-laws.md:88`](../../docs/four-laws.md#L88), [`law4-optimizer-laws.md:114,219`](../forge/studies/law4-optimizer-laws.md#L219) | *"When Law 2 bounds the critic's spatial slope, the generator carries the nonlinear pushforward geometry across disconnected modes, deep compositions, and saturation rails, whereas the critic primarily needs protection against extreme $D \gg G$ capacity imbalance and equilibrium logit jitter."* | **Falsified claim** |
| **C2** | [`law4-optimizer-laws.md:219-220`](../forge/studies/law4-optimizer-laws.md#L219-L220), [`docs/four-laws.md:184`](../../docs/four-laws.md#L184) | *"Removing Law-4 equilibration from $G$ causes severe multi-regime failure even with an equilibrated $D$, whereas removing Law-4 equilibration from $D$ preserves balanced/multimodal/deep-lattice convergence and fails catastrophically only under extreme $D \gg G$ capacity imbalance ($\rho_D \gg 100\times$); under an equilibrated $D$, the critic-side logit secant probe is 100% redundant across `>600×` imbalance."* | **Empirical correlation** |
| **C3** | [`docs/four-laws.md:163`](../../docs/four-laws.md#L163), [`law4-energy-champion.md:65-68`](../forge/studies/law4-energy-champion.md#L65-L68), [`law4-optimizer-laws.md:298`](../forge/studies/law4-optimizer-laws.md#L298) | *"Replacing Critic 5-step Quintic NS with zero-state degree-0 Frobenius-MP scaled by $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ preserves the exact $G:D:\text{prior}$ function-space Frobenius impedance ratio while eliminating 100% of Critic 2D NS matmuls and cutting equilibrium critic agitation by `3.8×`."* | **Empirically supported causal mechanism** |
| **C4** | [`law4-energy-champion.md:90`](../forge/studies/law4-energy-champion.md#L90) | *"Cleanly identifying the sole regime (generator-free direct particle clouds where $D$ is the only neural network) where dropping Critic NS incurs a hair-thin `~0.013` residual cost."* | **Falsified claim** |
| **C5** | [`particlegan/recipes.py:2378-2385`](../../particlegan/recipes.py#L2378-L2385), [`particlegan/_unified_v12/optimizer.py:643,928-938`](../../particlegan/_unified_v12/optimizer.py#L928-L938) | Binary routing `opt_d._generator_free = (total_gen_mats == 0)` applies 5-step Critic NS iff no trainable `ndim >= 2` generator/encoder parameter exists, hardcoding `0` Critic NS and `0` Critic `stiff_gate` for every neural-G critic. | **Engineering heuristic** |
| **C6** | [`docs/four-laws.md:88`](../../docs/four-laws.md#L88), [`law4-optimizer-laws.md:114`](../forge/studies/law4-optimizer-laws.md#L114) | Law 2 Born–Infeld slope regularization already supplies the critic's functional/spectral conditioning so the critic does not require spectral optimization. | **Falsified claim** |
| **C7** | [P5 Hypothesis (a)/(d)/(e)] (this audit, Theorems 1–2 & Exp 1d) | Even on identical architectures, scalar-potential critics exhibit two real (non-universal) role asymmetries vs. vector generators: (i) contrastive real-minus-fake cancellation of the positive-cone mean mode (slashing deep norm skew `4,567×` without removing deep-layer collapse), and (ii) `2.9×–7.8×` higher relative sensitivity of the spatial force field $\nabla_x D(x)$ to orthogonal NS perturbations compared to scalar logits $D(x)$. | **Empirically supported causal mechanism** |
| **C8** | [P5 Adaptive Pathology-Triggered Conditioning] (this audit, Exp 2 & Exp 3) | Per-layer scale-invariant stable-rank gating ($r_{\text{stable}} < 1.25$) is a drop-in role-agnostic replacement for hardcoded Law 4.4 asymmetry across all regimes. | **Falsified claim** |
| **C9** | [`docs/four-laws.md:175`](../../docs/four-laws.md#L175), [`RESEARCH_GLOSSARY.md:48`](../../RESEARCH_GLOSSARY.md#L48), [`_unified_v12/optimizer.py:457`](../../particlegan/_unified_v12/optimizer.py#L457), [`law4-energy-champion.md:90`](../forge/studies/law4-energy-champion.md#L90) | Documentation & docstring claims: `particlegan/unified_v12.py` exists; `_step_generator_free_critic` runs `"+ Schatten-6"`; `t3_iso_scale_mass_8x` `core_min_eigen_ratio` threshold is `0.35`. | **Falsified claim** |

### 1.2 Hardcoded Asymmetry Audit Across ParticleGAN Optimizers

Direct source inspection of [`particlegan/_unified_v11/optimizer.py`](../../particlegan/_unified_v11/optimizer.py), [`particlegan/_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py), [`particlegan/_law4_energy/optimizer.py`](../../particlegan/_law4_energy/optimizer.py), and [`particlegan/recipes.py`](../../particlegan/recipes.py) shows that `unified_v12` encodes **five distinct hardwired G-vs-D asymmetries**, none of which inspects the critic's actual depth, skip structure, or gradient spectrum:

| Mechanism | Generator (`player == "generator"`) | Neural-G Critic (`player == "critic"`, `_generator_free=False`) | Generator-Free Critic (`_generator_free=True`) | Exact Source Citation |
| :--- | :--- | :--- | :--- | :--- |
| **2D Dense Spectral Conditioning (`m >= 2`)** | 5-step fused `Z=[X,Y]` Quintic NS (`_QUINTIC_COEFFS`) | **Strictly `0` NS**; polar-Frobenius-matched Frob-MP (`factor_d = sqrt(r)`) | 5-step fused `Z=[X,Y]` Quintic NS restored | [`_unified_v12/optimizer.py:456,620-627,928-946`](../../particlegan/_unified_v12/optimizer.py#L928-L946) |
| **4D Conv2d Spectral Conditioning** | Strictly `0` NS (stride-aware polar-matched Frob-MP) | Strictly `0` NS (same Frob-MP) | Strictly `0` NS (same Frob-MP) | [`_unified_v12/optimizer.py:582-592,906-916`](../../particlegan/_unified_v12/optimizer.py#L582-L592) |
| **Stiffness Gate & Momentum (`b1_eff = 0.75*stiff_gate`)** | Active when `_ema_func_gain < 0.60` (`s_base -> 0`) | **Hardcoded `0.0`** (`stiff_gate_raw if player == "generator" else 0.0`) | **Hardcoded `0.0`** (`b1_eff = float(b1) = 0.0`) | [`_unified_v12/optimizer.py:484,759`](../../particlegan/_unified_v12/optimizer.py#L759); [`_unified_v11/optimizer.py:802`](../../particlegan/_unified_v11/optimizer.py#L802) |
| **Realized Function-Space Secant Trust Band** | Active (`delta_g_max = 0.045 + 0.16*min(1.5, push_res)`) | **Strictly `0` probes** (`not is_critic_opt`) | **Strictly `0` probes** (`last_probe_consumed = False`) | [`_unified_v12/optimizer.py:633,705-713`](../../particlegan/_unified_v12/optimizer.py#L705-L713) |
| **CountSketch Period-2 Flip Controller** | Active (`not is_critic_opt`, modulates `_ema_func_gain`) | **Disabled** (`not is_critic_opt`) | **Disabled** | [`_unified_v12/optimizer.py:697-698`](../../particlegan/_unified_v12/optimizer.py#L697-L698) |
| **Tikhonov Cap & Hot `peak_rms` Decay** | `min(s_base, 0.50 * peak_new)`; `p_dec = 1.0` (non-cond) | `min(s_base, 0.50 * peak_new)`; `p_dec = 1.0` (non-cond) | **`min(s_base, 0.30 * peak_new)`; `p_dec = 0.9995`** when `s_therm >= 0.5` | [`_unified_v12/optimizer.py:474,524,688,1034`](../../particlegan/_unified_v12/optimizer.py#L474-L527) |

### 1.3 Forensic Dissection of `unified_v12`'s Generator-Free Critic NS Fallback & Doc/Code Bugs

1. **Why `unified_v12` (`law4-energy-v11`) had to add `_step_generator_free_critic`:**
   - When `law4-energy-v10` dropped 2D Critic NS globally, it swept `14/14` neural `G+D` tasks in `r4-occam-ablation-v1` and reached `42/48` on `discriminator_stability` ([`law4-energy-champion.md:78,90-91`](../forge/studies/law4-energy-champion.md#L78-L91)), **yet failed BOTH generator-free ($G = \mathrm{Id}$) isolation probes** — `t3_iso_scale_mass_8x` (`core_min_eigen_ratio = 0.3375 < 0.45`) and `t3_iso_horizon_saddle_shift` (`hq = 0.9062 < 0.92`) — even though both tasks use a $C^\infty$ 2-layer `SmoothSiLUCritic` whose Law 2 slope bound is **strictly verified (`critic_grad_med <= 1.0`)**!
   - This single Forge failure already disproves the strong form of Law 4.4 ("once Law 2 bounds $\|\nabla_x D\| \le 1$, the critic never needs spectral conditioning"). Instead of measuring whether $D$ needed spectral conditioning, `law4-energy-v11` ([`particlegan/_law4_energy/optimizer.py:4403-4635`](../../particlegan/_law4_energy/optimizer.py#L4403-L4635)) and `unified_v12` ([`particlegan/_unified_v12/optimizer.py:456-644`](../../particlegan/_unified_v12/optimizer.py#L456-L644)) added a special-case branch triggered by [`particlegan/recipes.py:2378-2385`](../../particlegan/recipes.py#L2378-L2385):
     ```python
     total_gen_mats += sum(1 for _p in _mod.parameters() if _p.requires_grad and _p.ndim >= 2)
     opt_d._generator_free = total_gen_mats == 0
     ```
2. **Syntactic Brittleness (Verified in `test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs`):**
   - Because `_generator_free` checks ONLY whether `generator` + `encoder` have `0` trainable `ndim >= 2` tensors, pairing a deep skip-free critic with `nn.Identity()` sets `_generator_free = True` (`critic_ns_calls > 0`), whereas replacing `nn.Identity()` with a single trivial `nn.Linear(2, 2)` flips `_generator_free = False` and sets `critic_ns_calls == 0`!
3. **Three Bundled Silent Differences Inside `_step_generator_free_critic`:**
   - Compared to the neural-G critic path (`_unified_v12/optimizer.py:688,930,1034`), `_step_generator_free_critic` silently alters **(i)** 5-step fused `Z=[X,Y]` NS (`L620-627`) vs. `0`-NS $\sqrt{r}$ Frob-MP (`L930,1040`), **(ii)** Tikhonov cap `0.30 * peak_new` (`L524`) vs. `0.50 * peak_new` (`L1034`), and **(iii)** hot `peak_rms` decay `p_dec = 0.9995` (`L474`) vs. monotonic `1.0` (`L688`). Exp 3c isolates these factors quantitatively.
4. **Three Concrete Code & Documentation Bugs Found:**
   - **Bug 1 (Broken module path):** [`docs/four-laws.md:175`](../../docs/four-laws.md#L175) and [`RESEARCH_GLOSSARY.md:48`](../../RESEARCH_GLOSSARY.md#L48) reference nonexistent `particlegan/unified_v12.py`; the actual implementation lives in [`particlegan/_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py) (re-exported via `particlegan/recipes.py`).
   - **Bug 2 (False docstring claim):** [`particlegan/_unified_v12/optimizer.py:457`](../../particlegan/_unified_v12/optimizer.py#L457) claims `"""Zero-sync FP32 device-tensor 5-step 2D Critic NS + Schatten-6 strictly for generator-free tasks."""`, yet `_step_generator_free_critic` executes **zero** Schatten-6 decay calls (`schatten6` does not appear anywhere in the method).
   - **Bug 3 (Misquoted Forge pass threshold):** [`reports/forge/studies/law4-energy-champion.md:90`](../forge/studies/law4-energy-champion.md#L90) writes `t3_iso_scale_mass_8x core_min_eigen_ratio = 0.3375 vs 0.35` (calling it a *"hair-thin `~0.013` residual cost"*), whereas the frozen Forge contract [`configs/forge/tasks/t3_iso_scale_mass_8x.json:99`](../../configs/forge/tasks/t3_iso_scale_mass_8x.json#L99) and [`benchmarks/toy_audit/api_t3_isolation.py:75`](../../benchmarks/toy_audit/api_t3_isolation.py#L75) require `core_min_eigen_ratio >= 0.45` — an actual margin miss of **`0.1125` (`8.7×` larger than reported!)**.

---

## 2. Formal Mathematical Results & Causal Disentanglement of Hypotheses (a)–(f)

### 2.1 Theorem 1 (Hypothesis a — Scalar-Potential Linear Rank-1 Bottleneck)

**Theorem 1:** Let $f_\theta(x) = W_L W_{L-1} \cdots W_1 x \in \mathbb{R}^{d_{\text{out}}}$ be an $L$-layer bias-free linear MLP evaluated on a minibatch $X \in \mathbb{R}^{B \times d_{\text{in}}}$ with hidden width $W$ and arbitrary differentiable scalar loss $\mathcal{L}(f_\theta(X), \nabla_x f_\theta(X))$. Then for **every** layer $\ell \in \{1, \dots, L\}$, the weight gradient $G_{W_\ell} = \nabla_{W_\ell} \mathcal{L}$ satisfies:
$$\operatorname{rank}(G_{W_\ell}) \;\le\; \min(d_{\text{out}}, B, W).$$
In particular, for a scalar critic ($d_{\text{out}} = 1$, $f_\theta(x) = a^\top x$ with $a^\top = w_L W_{L-1}\cdots W_1 \in \mathbb{R}^{1 \times d_{\text{in}}}$), **both** adversarial losses $\mathcal{L}_{\text{adv}}(D(X))$ **and** any input-gradient penalty $\mathcal{L}_{\text{pen}}(\|\nabla_x D(X)\|_2) = \mathcal{L}_{\text{pen}}(\|a\|_2)$ produce **exact rank-1 weight gradients** at every layer:
$$G_{W_\ell} = \big(W_L \cdots W_{\ell+1}\big)^\top \cdot g_a^\top (W_{\ell-1} \cdots W_1)^\top \;\in\; \mathbb{R}^{d_\ell \times d_{\ell-1}}, \qquad \operatorname{rank}(G_{W_\ell}) \le 1,$$
where $g_a = \nabla_a \mathcal{L} \in \mathbb{R}^{d_{\text{in}}}$.

*Proof:* Write $P_\ell = W_L \cdots W_{\ell+1} \in \mathbb{R}^{d_{\text{out}} \times d_\ell}$ and $Q_\ell = W_{\ell-1} \cdots W_1 \in \mathbb{R}^{d_{\ell-1} \times d_{\text{in}}}$. Because $\mathcal{L}$ depends on $W_\ell$ exclusively through the linear map $A = P_\ell W_\ell Q_\ell \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ (both $f_\theta(X) = X A^\top$ and $\nabla_x f_\theta(x) = A$ depend only on $A$), the chain rule yields $G_{W_\ell} = P_\ell^\top (\nabla_A \mathcal{L}) Q_\ell^\top$. Since $P_\ell^\top \in \mathbb{R}^{d_\ell \times d_{\text{out}}}$ has rank at most $\min(d_{\text{out}}, d_\ell)$ and $\nabla_A \mathcal{L}_{\text{adv}} = (\nabla_Y \mathcal{L})^\top X$ has rank at most $\min(d_{\text{out}}, B)$, submultiplicativity of matrix rank gives $\operatorname{rank}(G_{W_\ell}) \le \min(d_{\text{out}}, B, W)$. Setting $d_{\text{out}} = 1$ gives $\operatorname{rank}(G_{W_\ell}) \le 1$ identically, including when Law 2 Born–Infeld slope penalty is added. $\square$

**Consequence:** A scalar critic's adversarial gradient can exceed rank $1$ **only** through sample-dependent nonlinear activation masks $D_\ell(x_i) = \operatorname{diag}(\phi'(a_\ell(x_i)))$ varying across the minibatch. When depth or equilibrium convergence synchronizes activation masks across samples, scalar critics naturally drift toward low effective rank ($r_{\text{stable}} \approx 1.1\text{–}1.3$) — verified in FP64 in Exp 1a and `test_theorem1_scalar_potential_linear_rank1_bottleneck`.

### 2.2 Theorem 2 (Hypotheses a, b, d — Contrastive Mean-Cone Cancellation vs. Deep-Layer Collapse)

**Proposition 2 (Contrastive Cancellation of the Positive-Cone Mean Mode):** Let $h_\ell(x) = \phi(W_\ell h_{\ell-1}(x))$ be an $L$-layer skip-free `LeakyReLU` ($\alpha \in [0, 1)$) MLP initialized orthogonally. Because $\phi$ is positively biased ($\mathbb{E}[\phi(z)] = \frac{1-\alpha}{\sqrt{2\pi}}\sigma > 0$), deeper activations $h_\ell(x)$ align progressively onto a shared positive-cone direction $\mu_\ell \in \mathbb{R}^W$ with residual variation $\tilde{h}_\ell(x) = h_\ell(x) - \mu_\ell$ satisfying $\|\tilde{h}_\ell\| / \|\mu_\ell\| \to 0$ with depth $\ell$ (Daneshmand et al., 2020).
1. Under a **single-sign objective** (such as a generator pushforward loss $\mathcal{L}_G = \frac{1}{B}\sum_i \ell(G(z_i))$ or single-class critic loss $-D(x_{\text{real}}).\text{mean}()$), the top-layer weight gradient $G_{W_\ell} = \frac{1}{B}\sum_i \delta_{\ell,i} h_{\ell-1,i}^\top$ accumulates a coherent rank-1 outer product $\bar{\delta}_\ell \mu_{\ell-1}^\top$ that grows exponentially toward the output relative to attenuated early layers, producing massive inter-layer gradient skew (`27,593×` on `G16`, `18,820×` on single-sign `D16`) and $r_{\text{stable}} \to 1.000$.
2. Under a **contrastive zero-sum critic objective** $\mathcal{L}_D = \frac{1}{B}\sum_i \psi(D(x_i^{\text{real}}) - D(x_i^{\text{fake}}))$ with matched sample weights $\sum_i w_i^{\text{real}} = \sum_i w_i^{\text{fake}}$, the shared mean-cone component cancels at first order:
   $$G_{W_\ell}^{\text{contrast}} = \frac{1}{B}\sum_i w_i\Big[\delta_{\ell}(x_i^{\text{real}}) h_{\ell-1}(x_i^{\text{real}})^\top - \delta_{\ell}(x_i^{\text{fake}}) h_{\ell-1}(x_i^{\text{fake}})^\top\Big] = \bar{\delta}_\ell \big(\mu_{\ell-1}^{\text{real}} - \mu_{\ell-1}^{\text{fake}}\big)^\top + O(\tilde{\delta}\tilde{h}^\top).$$
   At initialization where $\mu_{\ell-1}^{\text{real}} \approx \mu_{\ell-1}^{\text{fake}}$ in deep layers, this slashes inter-layer norm skew by **`4,567×`** (`18,820.4× -> 4.1×`, `1.8×` with `fourier=2`) and lifts mid-layer `med_srk` (`1.000 -> 1.220`). **However, it does NOT cure deep-layer spectral collapse or vanishing input sensitivity:** the deepest layers (`net.24..30`) still collapse to $r_{\text{stable}} = 1.000000005$, $\sigma_2/\sigma_1 = 2.85\times 10^{-5}$, and $\|\nabla_x D\|_2 = 4.94\times 10^{-7}$!

*Verified in Exp 1b and `test_theorem2_contrastive_mean_cone_cancellation_and_deep_critic_collapse`.*

### 2.3 Theorem 3 (The $\sqrt{r}$ Spectral-Norm Distortion & Irreducible Frobenius–Spectral Dilemma)

**Theorem 3:** Let $W_D \in \mathbb{R}^{r \times c}$ with $m = \min(r, c) \ge 2$ have nonzero gradient $G_D = U \Sigma V^\top$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_m \ge 0$ and stable rank $r_{\text{stable}}(G_D) = \|G_D\|_F^2 / \sigma_1^2 \in [1, m]$.
1. **Exact `unified_v12` Aspect-Frobenius Identity:** `UnifiedV12Optimizer` ([`particlegan/_unified_v12/optimizer.py:319,930`](../../particlegan/_unified_v12/optimizer.py#L930)) sets `factor_d = sqrt(max(1, r/c)) * sqrt(min(r, c))`. Checking both cases ($r \ge c \implies \sqrt{r/c}\sqrt{c} = \sqrt{r}$; $r < c \implies 1\cdot\sqrt{r} = \sqrt{r}$) yields the exact identity:
   $$\sqrt{\max(1, r/c)}\,\sqrt{\min(r, c)} \;\equiv\; \sqrt{r} \qquad \forall\, r, c \ge 1.$$
2. **Exact Spectral-Norm Overshoot vs. Polar NS:** For zero-smoothing Polar NS, $\Delta W^{\text{Polar}} = -\eta\,\gamma_{\text{aspect}}\,U V^\top$ has operator norm $\|\Delta W^{\text{Polar}}\|_2 = \eta\,\gamma_{\text{aspect}}$ and Frobenius norm $\eta\,\gamma_{\text{aspect}}\sqrt{m} = \eta\sqrt{r}$ (when $\operatorname{rank}(G_D) = m$). For `unified_v12`'s 0-NS critic update $\Delta W^{\text{v12}} = -\eta\sqrt{r}\,G_D / \|G_D\|_F$, the singular values are $\sigma_k(\Delta W^{\text{v12}}) = \eta\sqrt{r}\,\sigma_k / \|G_D\|_F$, giving the exact spectral-norm distortion ratio:
   $$\frac{\|\Delta W^{\text{v12}}\|_2}{\|\Delta W^{\text{Polar}}\|_2} = \frac{\sqrt{m}\,\sigma_1}{\|G_D\|_F} = \sqrt{\frac{m}{r_{\text{stable}}(G_D)}} \;\in\; \big[1, \sqrt{m}\big], \qquad \frac{\sigma_k(\Delta W^{\text{v12}})}{\sigma_k(\Delta W^{\text{Polar}})} = \sqrt{\frac{m}{r_{\text{stable}}(G_D)}}\,\frac{\sigma_k}{\sigma_1}.$$
   On a $128\times 128$ hidden critic layer with $r_{\text{stable}} = 1.0$, `v12`'s 0-NS rule **inflates the top singular mode by `11.314×`** (`10.328×` at $r_{\text{stable}} = 1.20$) while starving minor modes by $\sigma_k/\sigma_1$.
3. **Irreducible Frobenius–Spectral Matching Dilemma for Scalar Rescaling:** Any scalar-rescaled gradient $\Delta W^{(\alpha)} = -\alpha G_D$ satisfies the invariant ratio
   $$\frac{\|\Delta W^{(\alpha)}\|_F}{\|\Delta W^{(\alpha)}\|_2} \;=\; \sqrt{r_{\text{stable}}(G_D)}, \qquad \text{whereas} \qquad \frac{\|\Delta W^{\text{Polar}}\|_F}{\|\Delta W^{\text{Polar}}\|_2} \;=\; \sqrt{\operatorname{rank}(G_D)}.$$
   Consequently, whenever $r_{\text{stable}}(G_D) \ll m$, **no Frobenius-MP scaling choice $\alpha$ can simultaneously match Polar NS in both operator norm and Frobenius norm**:
   - **Frobenius-matched (`v12` $\sqrt{r}$):** $\|\Delta W\|_F = \eta\sqrt{r}$, but $\|\Delta W\|_2 = \sqrt{m/r_{\text{stable}}}\,\eta\gamma_{\text{aspect}}$ overshoots by up to $\sqrt{m}$ (`11.3×`), destabilizing deep skip-free critics (`Regime B1a` `hq = 0.2861`, standalone `16L D` `acc_valley = 0.6396`).
   - **Spectral-matched (`G / sigma_1`):** $\|\Delta W\|_2 = \eta\gamma_{\text{aspect}}$, stabilizing deep critics (`Regime B1b` `hq = 0.9163`, standalone `0.9814`), but $\|\Delta W\|_F = \sqrt{r_{\text{stable}}/m}\,\eta\sqrt{r}$ shrinks the critic update budget by `~7×`, causing generator-free `t3_iso_scale_mass_8x` to FAIL (`Exp 3c`: `min_eig = 0.3885 < 0.45`).
   - **Only spectral equalization (5-step NS)** decouples the two norms so $\|\Delta W\|_2 \approx \eta\gamma_{\text{aspect}}$ and $\|\Delta W\|_F \approx \eta\sqrt{r}$ hold simultaneously. $\square$

*Verified to machine precision (`< 1e-13`) in Exp 1c and `test_theorem3_sqrt_r_polar_frobenius_spectral_norm_distortion`.*

### 2.4 Hypotheses (c), (e), and (f) — Structural Mechanisms

- **Hypothesis (c) — Law 2 One-Sidedness:** ParticleGAN's Born–Infeld slope penalty penalizes excess spatial slope $\mathcal{R}_{\text{BI}}(D) \propto \mathbb{E}[\sqrt{1 + \max(0, \|\nabla_x D\|_2^2 - \kappa^2)} - 1]$ (one-sided upper cap at $\kappa = 1$). On a deep skip-free critic whose initial Jacobian contracts to $\|\nabla_x D\|_2 \approx 4.94\times 10^{-7} \ll 1$, **the penalty is identically zero (`< 1e-10`, verified in `test_law2_one_sidedness_inactive_on_deep_skip_free_critic`)**. Thus Law 2 protects against **upper** slope explosion ($\|\nabla_x D\| \gg 1$), never against **lower** depth-induced contraction ($\|\nabla_x D\| \to 0$).
- **Hypothesis (e) — Spatial Force-Field Amplification of Orthogonal Critic NS:** Because the generator receives gradients proportional to $\nabla_x D_\psi(G_\theta(z))$ rather than scalar logits $D_\psi(x)$, a parameter update $\Delta\psi$ orthogonal to $\nabla_\psi \mathcal{L}_D$ ($\langle \Delta\psi, \nabla_\psi \mathcal{L}_D\rangle = 0$) cancels to first order in the logit margin $\mathbb{E}[D(x_{\text{real}}) - D(x_{\text{fake}})]$, **yet does NOT cancel under the spatial derivative operator $\nabla_x$**! Exp 1d measures this directly: orthogonal NS perturbations move $\nabla_x D$ **`2.9×–7.8×`** more strongly (relative to raw-gradient displacement) than they move $D(x)$ itself (`50.2%` vs `7.4%` in balanced Regime C; `60.2%` vs `8.4%` in Regime A).
- **Hypothesis (f) — Capacity Ratio Imbalance ($\rho \gg 1$):** Under `620×` $D \gg G$ (`nash_overpowered_critic_ring16`) and `548×` $G \gg D$ (`nash_overpowered_generator_ring16`), the dominant obstruction is **Law 4.1 scale equilibration** (removing equilibration from the overpowered player causes `Equil-G + SGD-D` to hit `NaN` at step `5` on `620× D>>G`, and `SGD-G + Equil-D` to collapse to `7/16, hq = 0.2954` on `548× G>>D`), whereas adding 5-step NS gives a secondary quality boost on `620× D>>G` at `700` steps (`Both` `0.9146 PASS` and `Adaptive` `0.9067 PASS` vs. `Neither` `0.8394` / `G-only` `0.8989`).

---

## 3. Complete Verbatim Empirical Results

All metrics below are copied verbatim from the canonical run of [`scripts/four_laws_audit/p5_asymmetry_experiments.py`](../../scripts/four_laws_audit/p5_asymmetry_experiments.py) on `NVIDIA RTX A6000` (`cuda:1`, seed `0`, PyTorch `2.15.0+cu132`, total wall time `2492.4s`). Every executed arm and row is reported with zero omissions.

### 3.1 Experiment 1 — Analytical & Step-0 Mechanistic Disentanglement

#### Table 1a-1: Linear vs. LeakyReLU Output-Dimension Sweep (`64×64` Hidden Layers, `in_dim=32`, `B=128`, FP64)

| Model Family | Output Dim $d_{\text{out}}$ | `rank(1e-12)` | `rank(1e-6)` | Median $r_{\text{stable}}$ | Median $\operatorname{erank}_{\text{RV}}$ | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `linear` | `1` (scalar critic) | **1.0** | **1.0** | 1.000 | 1.000 | Exact rank-1 bottleneck (Theorem 1) |
| `linear` | `2` (2D generator) | **2.0** | **2.0** | 1.466 | 1.965 | Exact rank $= d_{\text{out}}$ |
| `linear` | `8` | **8.0** | **8.0** | 4.130 | 7.706 | Exact rank $= d_{\text{out}}$ |
| `linear` | `32` | **32.0** | **32.0** | 6.782 | 25.110 | Exact rank $= d_{\text{out}}$ |
| `leaky_relu` | `1` (scalar critic) | 64.0 | 64.0 | **1.254** | 19.470 | Full exact rank via gating, yet low $r_{\text{stable}}$ |
| `leaky_relu` | `2` (2D generator) | 64.0 | 64.0 | 2.327 | 25.208 | Higher $r_{\text{stable}}$ than scalar $d_{\text{out}}=1$ |
| `leaky_relu` | `8` | 64.0 | 64.0 | 2.827 | 28.677 | Higher $r_{\text{stable}}$ |
| `leaky_relu` | `32` | 64.0 | 64.0 | 2.035 | 31.476 | Highest $\operatorname{erank}_{\text{RV}}$ |

#### Table 1a-2: Effect of Adding Law 2 Born–Infeld Tangent Penalty to Scalar Critics (`d_out = 1`, FP64)

| Critic Family | `adv_only` `rank(1e-12)` | `adv_only` $r_{\text{stable}}$ / $\operatorname{erank}_{\text{RV}}$ / $\sigma_2/\sigma_1$ | `+Law2 BI` `rank(1e-12)` | `+Law2 BI` $r_{\text{stable}}$ / $\operatorname{erank}_{\text{RV}}$ / $\sigma_2/\sigma_1$ |
| :--- | :---: | :---: | :---: | :---: |
| `linear_d_out_1` | **1.0** | `1.000` / `1.000` / `8.70e-16` | **1.0** | `1.000` / `1.000` / `1.26e-15` |
| `leaky_relu_d_out_1` | 64.0 | `1.511` / `8.497` / `0.4643` | 64.0 | `1.157` / `6.584` / `0.2776` |

#### Table 1b-1: Contrastive Mean-Cone Cancellation & Architectural Step-0 Spectra (`128`-wide MLPs, FP64 SVD)

| Network & Loss Configuration | Hidden 2D Layers | Median $r_{\text{stable}}$ | Min $r_{\text{stable}}$ | Median $\operatorname{erank}_{\text{RV}}$ | Median $\sigma_2/\sigma_1$ | Inter-Layer Norm Skew | Median $\|\nabla_x D\|_2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `G16_adv` (16L G, single-sign) | 15 | **1.000** | **1.000** | 1.003 | `2.48e-4` | **27,593.1×** | — |
| `D16_f0_single_sign` (16L D, `-D(real)`) | 15 | **1.000** | **1.000** | 1.011 | `6.88e-4` | **18,820.4×** | `4.94e-7` |
| `D16_f0_contrastive` (16L D, `-(D(r)-D(f))`) | 15 | 1.220 | **1.000** | 1.967 | `4.60e-1` | **4.1×** *(4,567× drop!)* | **`4.94e-7`** |
| `D16_f2_contrastive` (16L D, `fourier=2`) | 15 | 1.280 | **1.000** | 2.037 | `4.89e-1` | **1.8×** | **`1.04e-6`** |
| `G3_adv` (3L shallow G) | 2 | 1.170 | 1.056 | 3.598 | `3.47e-1` | 1.2× | — |
| `D3_f2_contrastive` (3L shallow D, `fourier=2`) | 2 | 1.365 | 1.296 | 5.612 | `4.92e-1` | 1.0× | `7.59e-2` |

*Deep-layer collapse inside `D16_f0_contrastive` (Theorem 2):* Even after contrastive cancellation slashes overall skew to `4.1×`, per-layer spectra collapse sharply with depth: `net.2` ($r_{\text{stable}}=1.995, \sigma_2/\sigma_1=0.693$) $\to$ `net.8` ($1.380, 0.554$) $\to$ `net.16` ($1.220, 0.469$) $\to$ **`net.24` ($1.000000005, \sigma_2/\sigma_1 = 2.85\times 10^{-5}$)** $\to$ **`net.30` ($1.00000094, \sigma_2/\sigma_1 = 4.12\times 10^{-4}$)**. (In `D16_f2_contrastive`, `net.24` similarly collapses to $r_{\text{stable}} = 1.000000008, \sigma_2/\sigma_1 = 3.52\times 10^{-5}$.)

#### Table 1b-2: Balanced Conv2d `img_bars4` Step-0 Gradient Spectra

| Module / Parameter | Kind & Shape | $r_{\text{stable}}$ | $\operatorname{erank}_{\text{RV}}$ | Extra Metrics |
| :--- | :--- | :---: | :---: | :--- |
| `img_bars4_G`: `input.weight` | 2D `(48, 8)` | 2.627 | 7.305 | $\sigma_2/\sigma_1 = 0.789$, `rank(1e-12) = 8`, `cond = 3.53` |
| `img_bars4_G`: `first.weight` | 4D per-offset `(12, 12, 4, 4)` | 1.102 | 4.203 | — |
| `img_bars4_G`: `second.weight` | 4D per-offset `(12, 12, 4, 4)` | 1.182 | 4.410 | — |
| `img_bars4_D`: `network.2.weight` | 4D per-offset `(24, 12, 3, 3)` | 1.043 | 2.071 | *(`ImageCritic` has `0` 2D dense layers with `m >= 2`)* |

#### Table 1c: Theorem 3 Numerical Verification — `v12` $\sqrt{r}$ vs. `spec`-Matched vs. Polar ($128\times 128$, FP64)

| Target / Actual $r_{\text{stable}}$ | Theory $\sqrt{128/r_{\text{stable}}}$ | `v12` $\sqrt{r}$ $\sigma_1$ / $\sigma_2$ / $\|\cdot\|_F$ | `spec` Frob $\sigma_1$ / $\sigma_2$ / $\|\cdot\|_F$ | Polar $\sigma_1$ / $\sigma_2$ |
| :---: | :---: | :---: | :---: | :---: |
| `1.00` | `11.314×` | **`11.314`** / `7.30e-15` / `11.314` | `1.000` / `6.58e-16` / `1.000` | `1.000` / `6.58e-16` |
| `1.20` | `10.328×` | **`10.328`** / `0.410` / `11.314` | `1.000` / `0.040` / `1.095` | `1.000` / `1.000` |
| `4.00` | `5.657×` | **`5.657`** / `0.869` / `11.314` | `1.000` / `0.154` / `2.000` | `1.000` / `1.000` |
| `16.00` | `2.828×` | **`2.828`** / `0.972` / `11.314` | `1.000` / `0.344` / `4.000` | `1.000` / `1.000` |
| `64.00` | `1.414×` | **`1.414`** / `0.996` / `11.314` | `1.000` / `0.704` / `8.000` | `1.000` / `1.000` |

#### Table 1d: Hypothesis (e) — Realized Function-Space Sensitivity Under Equal Frobenius Step (`eps_frob = 0.05`)

| Architectural Regime | $\Delta G_{\text{rel}}$ (`raw / ns / orth`) | $\Delta D_{\text{rms}}$ (`raw / ns / orth`) | `D` Logit `orth/raw` | $\Delta(\nabla_x D)_{\text{rms}}$ (`raw / ns / orth`) | `D` Force `orth/raw` | Force / Logit Amplification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Regime_A_16LG_3LD` | `0.0150 / 0.0067 / 0.0003` | `0.0318 / 0.0203 / 0.0027` | `8.4%` | `0.0224 / 0.0167 / 0.0135` | **`60.2%`** | **`7.2×`** |
| `Regime_B_3LG_16LD_f0` | `0.0201 / 0.0095 / 0.0014` | `2.53e-5 / 3.75e-5 / 6.49e-5` | `256.0%` *(vanished)* | `2.78e-7 / 1.82e-7 / 9.28e-8` | `33.3%` *(vanished)* | — *(collapsed)* |
| `Regime_C_3LG_3LD` | `0.0200 / 0.0147 / 0.0040` | `0.0308 / 0.0193 / 0.0023` | `7.4%` | `0.0246 / 0.0175 / 0.0124` | **`50.2%`** | **`6.8×`** |
| `nash_overpowered_critic` | `0.0167 / 0.0057 / 0.0002` | `0.1211 / 0.0306 / 0.0252` | `20.9%` | `0.0727 / 0.0522 / 0.0445` | **`61.2%`** | **`2.9×`** |
| `nash_overpowered_generator` | `0.0568 / 0.0170 / 0.0012` | `0.1028 / 0.0325 / 0.0037` | `3.6%` | `0.0354 / 0.0143 / 0.0099` | **`28.1%`** | **`7.8×`** |

### 3.2 Experiment 2 — Matched 5-Arm Factorial Across All Structural Regimes

*(All arms share identical seed `0`, `deterministic_orthogonal_` weights, and pre-generated real batch sequences; single-run `cuda:1` RTX A6000 wall times.)*

#### Table 2.0: Standalone 16-Layer Skip-Free Critic Control (`fourier=0`, Law 2 BI $\beta=2.0$, `300` steps, `ring8` vs. Gaussian)

| Critic Optimizer Arm | `acc_gauss` | `acc_valley` | Logit Margin | Median $\|\nabla_x D\|_2$ | `ns_D / total_2d_D` | NS Saved | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Frob-MP_v12_sqrt_r` (`v12` exact rule) | **0.8750** | **0.6396** | `+0.379` | `0.074` | `0 / 4,800` | `100.0%` | `22.1` |
| `Frob-MP_spec_matched` (`G / sigma_1`) | 0.9834 | 0.9814 | `+0.313` | `0.281` | `0 / 4,800` | `100.0%` | `31.8` |
| `Frob-MP_aspect` (`aspect * G / ||G||_F`) | 0.9795 | 0.9961 | `+0.318` | `0.249` | `0 / 4,800` | `100.0%` | `22.5` |
| **`5step_Quintic_NS`** | **0.9951** | **1.0000** | `+0.149` | `0.519` | `4,800 / 4,800` | `0.0%` | `25.4` |
| **`Adaptive_Pathology`** ($r_{\text{stable}} < 1.25$) | **0.9844** | **1.0000** | `+0.469` | `0.139` | `3,671 / 4,800` | `23.5%` | `29.0` |

#### Table 2.1: Regime A — Deep 16L Skip-Free G + Shallow 3L D (`grid100_mlp16`, ParticleGAN `unified_v12`, `1,200` steps, `5,000` particles, `bsz=512`)

| Arm | Modes | `hq` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ (Tail $\Delta D$) | `ns_G / total` | `ns_D / total` | NS Saved vs. Both | Step-10 / 600 / 1200 $r_{\text{stable}}(G)$ ; $r_{\text{stable}}(D)$ | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | `34 / 100` | `0.1153` | `0.548` | `0.1167 / 1.580 (1.713)` | `0 / 20,400` | `0 / 3,600` | `100.0%` | `1.03 / 1.35 / 1.61` ; `1.18 / 1.20 / 1.57` | `93.2` |
| **`G-only`** | **`100 / 100`** | **`0.9714`** | `0.238` | `0.0501 / 0.986 (0.430)` | `20,400 / 20,400` | `0 / 3,600` | `15.0%` | `1.00 / 1.08 / 1.01` ; `1.25 / 1.32 / 1.61` | `104.1` |
| `D-only` | `1 / 100` | `0.0151` | `0.908` | `0.1187 / 0.586 (0.728)` | `0 / 20,400` | `3,600 / 3,600` | `85.0%` | `1.00 / 1.00 / 1.05` ; `1.39 / 1.49 / 1.68` | `95.4` |
| **`Both`** | **`100 / 100`** | **`0.9697`** | `0.208` | `0.0509 / 0.335 (0.259)` | `20,400 / 20,400` | `3,600 / 3,600` | `0.0%` | `1.00 / 1.45 / 1.00` ; `1.39 / 1.26 / 1.24` | `98.9` |
| `Adaptive-Pathology` | `98 / 100` | `0.8156` | `0.353` | `0.0814 / 0.664 (0.412)` | `14,282 / 20,400` | `3,008 / 3,600` | `28.0%` | `1.01 / 1.03 / 1.00` ; `1.15 / 1.12 / 1.60` | `121.1` |

#### Table 2.2: Regime B1a — Adversarial Falsification: Shallow 3L G + Deep 16L Skip-Free D (`PureAdv RpGAN + Law 2`, `v12 sqrt(r)` Frob-MP, `450` steps, `ring8`)

| Arm | Modes | `hq` | `mass_tv` | `d_acc_gauss` / `d_acc_valley` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ | `ns_G / total` | `ns_D / total` | NS Saved | Step-10 / 225 / 450 $r_{\text{stable}}(G)$ ; $r_{\text{stable}}(D)$ | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | `8 / 8` | **`0.2861`** | `0.0293` | `0.479 / 0.537` | `1.11e-7` | `0.0542 / 0.157` | `0 / 1,800` | `0 / 7,200` | `100.0%` | `1.02 / 1.11 / 1.13` ; `1.09 / 1.27 / 1.00` | `36.3` |
| `G-only` *(Law 4.4)* | **`7 / 8`** | **`0.3701`** | `0.0334` | `0.513 / 0.520` | `1.70e-3` | `0.0210 / 0.233` | `1,800 / 1,800` | `0 / 7,200` | `80.0%` | `1.15 / 1.15 / 1.08` ; `1.01 / 1.02 / 1.17` | `35.7` |
| **`D-only`** | **`8 / 8`** | **`0.7947`** | `0.0227` | `0.676 / 0.679` | `8.04e-3` | `0.0815 / 0.088` | `0 / 1,800` | `7,200 / 7,200` | `20.0%` | `1.23 / 1.44 / 1.08` ; `1.19 / 1.22 / 1.01` | `36.3` |
| **`Both`** | **`8 / 8`** | **`0.9055`** | `0.0142` | `0.677 / 0.601` | `3.53e-3` | `0.0348 / 0.197` | `1,800 / 1,800` | `7,200 / 7,200` | `0.0%` | `1.14 / 1.02 / 1.37` ; `1.01 / 1.34 / 1.76` | `36.9` |
| **`Adaptive-Pathology`** | **`8 / 8`** | **`0.8110`** | `0.0176` | `0.708 / 0.678` | `5.50e-3` | `0.0630 / 0.214` | `993 / 1,800` | `6,030 / 7,200` | `22.0%` | `1.13 / 1.76 / 1.97` ; `1.22 / 1.01 / 1.02` | `55.1` |

#### Table 2.3: Regime B1b — Theorem 3 Isolation: Same 3L G + 16L D with Spectral-Norm-Matched (`G / sigma_1`) Frob-MP (`450` steps, `ring8`)

| Arm | Modes | `hq` | `mass_tv` | `d_acc_gauss` / `d_acc_valley` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ | `ns_G / total` | `ns_D / total` | NS Saved | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | `8 / 8` | `0.9163` | `0.0217` | `0.752 / 0.669` | `6.31e-3` | `0.0259 / 0.155` | `0 / 1,800` | `0 / 7,200` | `100.0%` | `48.4` |
| `G-only` *(Law 4.4)* | **`7 / 8`** | **`0.7285`** | `0.0330` | `0.609 / 0.585` | `1.30e-2` | `0.0444 / 0.100` | `1,800 / 1,800` | `0 / 7,200` | `80.0%` | `46.3` |
| **`D-only`** | **`8 / 8`** | **`0.9482`** | `0.0217` | `0.647 / 0.565` | `9.81e-3` | `0.0233 / 0.367` | `0 / 1,800` | `7,200 / 7,200` | `20.0%` | `35.6` |
| `Both` | `8 / 8` | `0.9055` | `0.0142` | `0.677 / 0.601` | `3.53e-3` | `0.0348 / 0.197` | `1,800 / 1,800` | `7,200 / 7,200` | `0.0%` | `34.7` |
| `Adaptive-Pathology` | `8 / 8` | `0.7847` | `0.0251` | `0.607 / 0.615` | `1.07e-2` | `0.0249 / 0.224` | `933 / 1,800` | `4,757 / 7,200` | `36.8%` | `39.4` |

*(Comparing B1a vs. B1b isolates two distinct mechanisms: replacing `v12`'s `11.3×`-overshooting $\sqrt{r}$ Frob-MP with spectral-norm-matched `G / sigma_1` lifts `Neither` from `0.2861 -> 0.9163`, yet **`D-only` (`0.9482`) still wins overall and beats `G-only` (`7/8, 0.7285`) by `+0.220`**, confirming that spectral equalization belongs on whichever player is deep — not blindly on $G$!)*

#### Table 2.4: Regime B2 — Direct-OT Masking Confounder (`Full ParticleGAN unified_v12`, `3L G + 16L D`, `ring16`, `350` steps)

| Arm | Modes | `hq` | `mass_tv` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ | `ns_G / total` | `ns_D / total` | NS Saved | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | `16 / 16` | `0.9858` | `0.0984` | `0.118` | `0.0084 / 0.490` | `0 / 1,400` | `0 / 5,600` | `100.0%` | `20.6` |
| `G-only` | `16 / 16` | `0.9834` | `0.1050` | `0.100` | `0.0166 / 0.499` | `1,400 / 1,400` | `0 / 5,600` | `80.0%` | `20.3` |
| `D-only` | `16 / 16` | `0.9802` | `0.1006` | `0.094` | `0.0154 / 0.291` | `0 / 1,400` | `5,600 / 5,600` | `20.0%` | `22.3` |
| `Both` | `16 / 16` | `0.9788` | `0.1038` | `0.167` | `0.0166 / 0.313` | `1,400 / 1,400` | `5,600 / 5,600` | `0.0%` | `21.7` |
| `Adaptive-Pathology` | `16 / 16` | `0.9856` | `0.1060` | `0.266` | `0.0114 / 0.282` | `333 / 1,400` | `5,247 / 5,600` | `20.3%` | `28.9` |

*(Despite using the same 16-layer skip-free critic, ParticleGAN's direct data-space Laguerre–Sinkhorn OT pullback bypasses $D$ and drives all 5 arms to `16/16, hq = 0.979–0.986`, collapsing the `0.619` `hq` gap of pure adversarial B1a down to `0.007` — though note Critic NS in `Both`/`Adaptive` still lifts $\|\nabla_x D\|_2$ to `0.167–0.266` vs. `0.100–0.118` under `0`-NS.)*

#### Table 2.5: Regime B3 — Both Deep (`16L G + 16L D`, Pure Adversarial RpGAN + Law 2, `v12 sqrt(r)`, `400` steps, `ring8`)

| Arm | Modes | `hq` | `mass_tv` | `d_acc_gauss` / `d_acc_valley` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ | `ns_G / total` | `ns_D / total` | NS Saved | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | `0 / 8` | `0.0000` | `0.8750` | `0.572 / 0.536` | `0.053` | `0.3187 / 1.155` | `0 / 6,800` | `0 / 6,400` | `100.0%` | `49.4` |
| `G-only` | `0 / 8` | `0.0000` | `0.6943` | `0.557 / 0.534` | `0.071` | `0.0420 / 1.719` | `6,800 / 6,800` | `0 / 6,400` | `48.5%` | `54.9` |
| `D-only` | `0 / 8` | `0.0000` | `0.8750` | `0.590 / 0.566` | `0.191` | `0.3137 / 1.960` | `0 / 6,800` | `6,400 / 6,400` | `51.5%` | `53.8` |
| `Both` | `0 / 8` | `0.0000` | `0.8750` | `0.580 / 0.518` | `0.193` | `0.0257 / 1.863` | `6,800 / 6,800` | `6,400 / 6,400` | `0.0%` | `55.6` |
| `Adaptive-Pathology` | `1 / 8` | `0.0757` | `0.8750` | `0.548 / 0.493` | `0.177` | `0.0383 / 2.122` | `6,464 / 6,800` | `5,988 / 6,400` | `5.7%` | `68.4` |

*(Faithful numerical FAIL: when BOTH players are 16-layer skip-free MLPs in a zero-momentum pure adversarial game without direct OT, the product of initial contractions — `1/187` output spread on $G$ times `5e-7` spatial slope on $D$ — leaves all 5 arms unable to reach the `radius = 2.5` ring within `400` steps; note `D-only`/`Both`/`Adaptive` still revive $\|\nabla_x D\|_2$ `3×` higher [`0.177–0.193`] than `Neither`/`G-only` [`0.053–0.071`].)*

#### Table 2.6: Regime C1 — Balanced Shallow (`3L G + 3L D`, `ring16`, ParticleGAN `unified_v12`, `350` steps)

| Arm | Modes | `hq` | `mass_tv` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ (Tail $\Delta D$) | `ns_G / total` | `ns_D / total` | NS Saved vs. Both | Step-10 / 175 / 350 $r_{\text{stable}}(G)$ ; $r_{\text{stable}}(D)$ | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | `16 / 16` | `0.9875` | `0.1099` | `0.297` | `0.0115 / 0.278 (0.138)` | `0 / 1,400` | `0 / 1,050` | `100.0%` | `2.28 / 1.34 / 1.89` ; `1.27 / 1.28 / 1.30` | `14.0` |
| `G-only` | `16 / 16` | `0.9775` | `0.1021` | `0.356` | `0.0168 / 0.366 (0.237)` | `1,400 / 1,400` | `0 / 1,050` | `42.9%` | `1.75 / 1.30 / 1.68` ; `1.21 / 1.35 / 1.97` | `13.0` |
| `D-only` | `16 / 16` | `0.9958` | `0.0977` | `0.367` | `0.0159 / 0.131 (0.119)` | `0 / 1,400` | `1,050 / 1,050` | `57.1%` | `1.51 / 1.55 / 1.86` ; `1.37 / 1.29 / 1.39` | `14.7` |
| `Both` | `16 / 16` | `0.9814` | `0.1060` | `0.323` | `0.0162 / 0.145 (0.115)` | `1,400 / 1,400` | `1,050 / 1,050` | `0.0%` | `1.38 / 1.54 / 1.80` ; `1.26 / 1.35 / 1.53` | `15.7` |
| **`Adaptive-Pathology`** | **`16 / 16`** | **`0.9915`** | **`0.0830`** | `0.320` | `0.0085 / 0.174 (0.158)` | `321 / 1,400` | `673 / 1,050` | **`59.4%`** | `1.74 / 1.50 / 1.66` ; `1.27 / 1.20 / 1.32` | `17.2` |

#### Table 2.7: Regime C2 — Balanced Conv2d Multimodal (`img_bars4`, ParticleGAN `unified_v12`, `500` steps)

| Arm | Passed | Modes | `hq` | `mean_rmse` | `dist_tv` | Mean $\Delta G$ / $\Delta D$ | `ns_G / total` | `ns_D / total` | NS Saved vs. Both | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Neither` | **PASS** | `4 / 4` | `1.0000` | `0.0064` | `0.0508` | `0.0045 / 0.162` | `0 / 500` | `0 / 0` | `100.0%` | `23.6` |
| `G-only` | **PASS** | `4 / 4` | `1.0000` | `0.0067` | `0.0742` | `0.0032 / 0.131` | `500 / 500` | `0 / 0` | `0.0%` | `24.7` |
| `D-only` | **PASS** | `4 / 4` | `1.0000` | `0.0064` | `0.0508` | `0.0045 / 0.162` | `0 / 500` | `0 / 0` | `100.0%` | `24.8` |
| `Both` | **PASS** | `4 / 4` | `1.0000` | `0.0067` | `0.0742` | `0.0032 / 0.131` | `500 / 500` | `0 / 0` | `0.0%` | `25.2` |
| `Adaptive-Pathology` | **PASS** | `4 / 4` | `0.9648` | `0.0177` | `0.0684` | `0.0044 / 0.143` | `206 / 500` | `0 / 0` | **`58.8%`** | `25.4` |

*(Note: `ImageCritic` has `0` 2D dense layers with `m >= 2`, so `Neither` and `D-only` are bit-identical, as are `G-only` and `Both`; zero NS is needed anywhere on `img_bars4`.)*

### 3.3 Experiment 3 — Causal Disentanglement of Hypotheses (c), (f), and `t3_iso` Fallback

#### Table 3a: Hypothesis (c) — Law 2 Born–Infeld Penalty ON ($\beta=2.0$) vs. OFF ($\beta=0.0$) Across Shallow & Deep Critics (`250` steps, `ring8`, Pure Adversarial)

| Architecture Cohort | Law 2 | Arm | Modes | `hq` | `mass_tv` | `d_acc_gauss` / `d_acc_valley` | $\|\nabla_x D\|_2$ | Mean $\Delta G$ / $\Delta D$ | `ns_G` / `ns_D` | Wall (`s`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `shallow_3LG_3LD` (`spec`) | **ON** | `Neither` | `8 / 8` | `0.7036` | `0.0632` | `0.912 / 0.994` | `0.054` | `0.0394 / 0.0316` | `0 / 0` | `8.4` |
| `shallow_3LG_3LD` (`spec`) | **ON** | `G-only` | `8 / 8` | `0.7251` | `0.0808` | `0.752 / 0.774` | `0.053` | `0.0610 / 0.0285` | `1,000 / 0` | `8.6` |
| `shallow_3LG_3LD` (`spec`) | **ON** | `D-only` | `8 / 8` | `0.9136` | `0.0493` | `0.792 / 0.709` | `0.028` | `0.0451 / 0.0646` | `0 / 750` | `8.3` |
| `shallow_3LG_3LD` (`spec`) | **ON** | `Both` | `8 / 8` | `0.8940` | `0.0237` | `0.745 / 0.679` | `0.026` | `0.0602 / 0.0440` | `1,000 / 750` | `8.5` |
| `shallow_3LG_3LD` (`spec`) | **OFF** | `Neither` | **`2 / 8`** | `0.8938` | `0.7500` | `0.597 / 0.622` | **`4.757`** | `0.0438 / 0.1618` | `0 / 0` | `6.7` |
| `shallow_3LG_3LD` (`spec`) | **OFF** | `G-only` | **`2 / 8`** | `0.4761` | `0.7446` | `0.740 / 0.775` | **`7.608`** | `0.0638 / 0.1875` | `1,000 / 0` | `6.9` |
| `shallow_3LG_3LD` (`spec`) | **OFF** | `D-only` | **`2 / 8`** | `0.9512` | `0.7500` | `0.664 / 0.748` | **`10.920`** | `0.0479 / 0.5295` | `0 / 750` | `6.7` |
| `shallow_3LG_3LD` (`spec`) | **OFF** | `Both` | **`5 / 8`** | `0.5466` | `0.3523` | `0.906 / 0.981` | **`7.620`** | `0.0678 / 0.5251` | `1,000 / 750` | `6.6` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **ON** | `Neither` | `8 / 8` | **`0.4561`** | `0.0295` | `0.460 / 0.483` | **`2.62e-7`** | `0.0754 / 0.2511` | `0 / 0` | `18.5` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **ON** | `G-only` | `8 / 8` | **`0.4324`** | `0.0129` | `0.511 / 0.488` | **`1.29e-5`** | `0.0279 / 0.3799` | `1,000 / 0` | `19.2` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **ON** | **`D-only`** | **`8 / 8`** | **`0.7578`** | `0.0308` | `0.705 / 0.664` | `0.014` | `0.1118 / 0.1303` | `0 / 4,000` | `21.5` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **ON** | **`Both`** | **`8 / 8`** | **`0.8562`** | `0.0159` | `0.688 / 0.507` | `0.004` | `0.0389 / 0.3278` | `1,000 / 4,000` | `21.2` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **OFF** | `Neither` | `8 / 8` | `0.4016` | `0.0239` | `0.463 / 0.514` | `0.002` | `0.1327 / 1.5317` | `0 / 0` | `12.2` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **OFF** | `G-only` | **`5 / 8`** | `0.1606` | `0.1199` | `0.413 / 0.457` | `0.002` | `0.0534 / 1.3073` | `1,000 / 0` | `13.6` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **OFF** | `D-only` | `8 / 8` | `0.5645` | `0.0732` | `0.964 / 0.983` | **`3.261`** | `0.2532 / 1.4999` | `0 / 4,000` | `16.2` |
| `deep_3LG_16LD` (`v12 sqrt_r`) | **OFF** | `Both` | **`3 / 8`** | `0.5745` | `0.6250` | `0.519 / 0.582` | **`100.055`** | `0.0660 / 45.0112` | `1,000 / 3,984` | `17.0` |

*(Clean causal double proof of orthogonality: removing Law 2 collapses shallow `3L+3L` to `2/8–5/8` modes with $\|\nabla_x D\| = 4.76\text{–}10.92$ even WITH Critic NS; conversely, keeping Law 2 ON does NOT rescue a deep 16L critic under `Neither`/`G-only` (`hq = 0.43–0.46`, $\|\nabla_x D\| \le 1.3\times 10^{-5}$), and only **Law 2 ON + Critic NS** (`D-only`/`Both`) simultaneously prevents upper slope blowup [`100.055` when Law 2 OFF] and lower spectral collapse [`0.758–0.856`].)*

#### Table 3b: Hypothesis (f) — Capacity Ratio Imbalance (`620× D>>G` & `548× G>>D`, Exact Forge Nash Hosts, `700` steps)

| Forge Imbalanced Nash Task | Arm | Finite (Fail Step) | Passed | Modes | `hq` | `mass_tv` | `ns_G` / `ns_D` | NS Saved vs. Both | Wall (`s`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nash_overpowered_critic_ring16`** (`620× D>>G`) | `Neither` | True | FAIL | `16 / 16` | `0.8394` | `0.1284` | `0 / 0` | `100.0%` | `28.2` |
| `nash_overpowered_critic_ring16` (`620× D>>G`) | `G-only` | True | FAIL | `16 / 16` | `0.8989` | `0.1313` | `2,100 / 0` | `57.1%` | `30.0` |
| `nash_overpowered_critic_ring16` (`620× D>>G`) | `D-only` | True | FAIL | `16 / 16` | `0.8823` | `0.1274` | `0 / 2,800` | `42.9%` | `30.8` |
| `nash_overpowered_critic_ring16` (`620× D>>G`) | **`Both`** | True | **PASS** | **`16 / 16`** | **`0.9146`** | `0.1255` | `2,100 / 2,800` | `0.0%` | `34.1` |
| `nash_overpowered_critic_ring16` (`620× D>>G`) | **`Adaptive-Pathology`** | True | **PASS** | **`16 / 16`** | **`0.9067`** | `0.1274` | `1,288 / 2,615` | `20.3%` | `38.4` |
| `nash_overpowered_critic_ring16` (`620× D>>G`) | `Equil-G_SGD-D` | **False (`step 5`)** | **NaN** | `0 / 16` | `0.0000` | `1.0000` | `15 / 0` | — | `0.3` |
| `nash_overpowered_critic_ring16` (`620× D>>G`) | **`SGD-G_Equil-D`** | True | **PASS** | **`16 / 16`** | **`0.9434`** | `0.1284` | `0 / 2,800` | — | `29.1` |
| **`nash_overpowered_generator_ring16`** (`548× G>>D`) | `Neither` | True | **PASS** | `16 / 16` | `0.9624` | `0.1021` | `0 / 0` | `100.0%` | `35.9` |
| `nash_overpowered_generator_ring16` (`548× G>>D`) | **`G-only`** | True | **PASS** | `16 / 16` | **`0.9873`** | `0.1211` | `3,500 / 0` | `28.6%` | `37.5` |
| `nash_overpowered_generator_ring16` (`548× G>>D`) | `D-only` | True | **PASS** | `16 / 16` | `0.9766` | `0.0957` | `0 / 1,400` | `71.4%` | `37.2` |
| `nash_overpowered_generator_ring16` (`548× G>>D`) | `Both` | True | **PASS** | `16 / 16` | `0.9546` | `0.1084` | `3,500 / 1,400` | `0.0%` | `40.3` |
| `nash_overpowered_generator_ring16` (`548× G>>D`) | `Adaptive-Pathology` | True | **PASS** | `16 / 16` | `0.9785` | `0.1113` | `1,848 / 1,039` | `41.1%` | `42.7` |
| `nash_overpowered_generator_ring16` (`548× G>>D`) | `Equil-G_SGD-D` | True | **PASS** | `16 / 16` | `0.9619` | `0.1206` | `3,500 / 0` | — | `33.1` |
| `nash_overpowered_generator_ring16` (`548× G>>D`) | `SGD-G_Equil-D` | True | **FAIL** | **`7 / 16`** | **`0.2954`** | `0.0688` | `0 / 1,400` | — | `24.7` |

*(Notice the exact capacity-ratio symmetry of Law 4.1 equilibration: putting raw SGD on the `620×` overpowered critic explodes at step `5`, and putting raw SGD on the `548×` overpowered generator collapses to `7/16` [`hq = 0.2954`]! Meanwhile, at the `700`-step budget on `620× D>>G`, Critic NS accelerates `hq` past the `0.90` pass bar [`Both 0.9146`, `Adaptive 0.9067`, `SGD-G_Equil-D 0.9434`] while `G-only` [`0.8989`] and `Neither` [`0.8394`] lag behind.)*

#### Table 3c: Forensic Dissection of `unified_v12`'s Generator-Free Critic NS Fallback (`t3_iso_scale_mass_8x`, `G = nn.Identity()`, `800` steps)

| Ablation Arm (Exact Config) | Final Pass | Sust. `5/5` (`600..800`) | `core_min_eigen_ratio` (Worst-5) | `core_cov_err` | `hq` | `mass_tv` | $\|\nabla_x D\|_2$ | `ns_D / total` (Saved) | Step-10/200/400/800 $r_{\text{stable}}(D)$ | Wall (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v12_fallback_ON_exact`** (`ns`, `sqrt_m`, `0.30`, `0.9995`) | **PASS** | **`5 / 5`** | **`0.5859`** (`0.5859`) | `0.2979` | `0.9570` | `0.0276` | `0.0012` | `1,600 / 1,600` (`0.0%`) | `1.17 / 1.23 / 1.59 / 1.12` | `39.6` |
| `v12_fallback_OFF_exact` (`frob`, `sqrt_m`, `0.50`, `1.0`) | PASS | `5 / 5` | `0.5353` (`0.5353`) | `0.2694` | `0.9473` | `0.0334` | `0.0008` | `0 / 1,600` (`100.0%`) | `1.25 / 1.28 / 1.61 / 1.16` | `38.1` |
| `ns_only_ablation_sqrt_r` (`frob`, `sqrt_m`, `0.30`, `0.9995`) | PASS | `5 / 5` | `0.5459` (`0.5459`) | `0.3639` | `0.9609` | `0.0315` | `0.0006` | `0 / 1,600` (`100.0%`) | `1.27 / 1.33 / 1.59 / 1.11` | `38.2` |
| **`spec_matched_0ns_d`** (`frob`, `spec`, `0.30`, `0.9995`) | **FAIL** | **`0 / 5`** | **`0.3885`** (`0.3884`) | `0.4867` | `0.9434` | `0.0373` | `0.0021` | `0 / 1,600` (`100.0%`) | `1.13 / 1.20 / 1.47 / 1.13` | `39.2` |
| **`adaptive_pathology_d`** (`adaptive`, `spec`, `0.30`, `0.9995`, $\tau=1.35$) | **FAIL** | **`0 / 5`** | **`0.4181`** (`0.4181`) | `0.3274` | `0.9473` | `0.0295` | `0.0006` | `1,075 / 1,600` (`32.8%`) | `1.16 / 1.20 / 1.57 / 1.17` | `40.7` |

*(Honest horizon disclosure: at the `800`-step diagnostic horizon, `v12_fallback_OFF_exact` still sits above `0.45` at `0.5353` whereas at the frozen `1,600`-step Forge horizon in `readout-a0e3e7f6` `law4-energy-v10` degrades to `0.3375 < 0.45`. Across all 5 matched arms at `800` steps, **5-step Critic NS** achieves the highest eigenvalue ratio [`0.5859`, `+0.0506` over `fallback_OFF` and `+0.0400` over `ns_only_ablation`], while **`spec`-matched 0-NS (`0.3885`) and `adaptive+spec` (`0.4181`) both FAIL `0/5`** — empirically confirming Theorem 3's Frobenius–Spectral Matching Dilemma!)*

---

## 4. Eight-Point Structured Analysis of All Audited Claims

### Claim C1 — Unconditional Role-Based Law 4.4 (`docs/four-laws.md:88`, `law4-optimizer-laws.md:114,219`)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Claims that whenever Law 2 bounds $\|\nabla_x D\|_2 \le 1$, structural/spectral conditioning is inherently a generator requirement while the critic only needs protection against extreme $D \gg G$ capacity imbalance and equilibrium logit jitter.
2. **Assumptions:** Implicitly assumes standard GAN architectural asymmetry (deep/pathological $G$ paired with shallow, Fourier-encoded, or BatchNorm-regularized $D$) AND ignores generator-free direct-particle critics ($G = \mathrm{Id}$) and pure adversarial games lacking direct data-space OT losses.
3. **Existing supporting evidence:** Forge `r4-occam-ablation-v1` and `discriminator_stability` show `law4-energy-v10` (`0` Critic NS) passes `14/14` neural `G+D` tasks and `42/48` overall, while our Regime A (`16L G + 3L D`) confirms `G-only` (`100/100, hq=0.9714`) beats `Neither` (`34/100`) and `D-only` (`1/100`).
4. **Potential counterexamples:** (a) Swapping the 16-layer skip-free MLP onto $D$ (`Regime B1a` `3L G + 16L D`, Law 2 ON) causes `G-only` to fail (`7/8, hq=0.3701`) while `D-only` (`8/8, hq=0.7947`) and `Both` (`0.9055`) succeed; even with spectral-norm-matched Frob-MP (`Regime B1b`), `D-only` (`0.9482`) beats `G-only` (`7/8, 0.7285`); (b) standalone 16L critic with Law 2 ON (`0.6396` valley acc under `v12` 0-NS vs `1.0000` under 5-step NS); (c) generator-free `t3_iso_scale_mass_8x` & `t3_iso_horizon_saddle_shift` (`G = Id`, Law 2 strictly verified), where `law4-energy-v10` failed (`min_eig = 0.3375 < 0.45`) until `unified_v12` restored 2D Critic NS.
5. **Minimal falsification experiment:** Train a `3L` generator against a `16L` skip-free `LeakyReLU(0.2)` critic (`fourier=0`) under pure RpGAN + Law 2 BI (`beta=2.0`) comparing `G-only` vs. `D-only` (implemented in `Regime B1a/B1b` and `test_deep_critic_v12_sqrt_r_degrades_vs_ns_and_spec_matched`).
6. **Relevant prior research:** Miyato et al. (2018, ICLR) *Spectral Normalization for Generative Adversarial Networks* (critic spectral conditioning); Brock et al. (2019, ICLR) *BigGAN* (orthogonal/spectral conditioning on both G and D); Daneshmand et al. (2020, NeurIPS) *Batch Normalization Provably Avoids Rank Collapse*; Jordan et al. (2024) *Muon*.
7. **Computational verification:** Falsified in Exp 2 (Tables 2.0, 2.2, 2.3), Exp 3a (Table 3a), Exp 3c (Table 3c), and `tests/four_laws_audit/test_p5_asymmetry.py` (`9/9` PASS).
8. **Confidence & remaining uncertainty:** **99% certainty** that the unconditional role-based claim is false; open question whether deep skip-free **Conv2d** critics without BatchNorm show the exact same reversal.

### Claim C2 — Forge Double-Dissociation Proof of Law 4.4 (`law4-optimizer-laws.md:107-115,219-220`)
- **Classification:** **Empirical correlation**
1. **Precise formulation:** Uses `R4` (`Equil-G + SGD-D`: `12/16` Occam PASS, `NaN` on `620× D>>G`) vs. `R3a` (`SGD-G + Equil-D`: `5/13` PASS, `0/5` on hard $G$-geometry probes) to prove an intrinsic generator-vs-critic asymmetry law.
2. **Assumptions:** Assumes the 16-task `r4-occam-ablation-v1` suite is architecturally symmetric between $G$ and $D$, and that $G$'s updates flow purely through $D$.
3. **Existing supporting evidence:** Replicated faithfully on `620×` and `548×` Nash in our Exp 3b (`Equil-G_SGD-D` hits `NaN` at step `5` on `620× D>>G` and passes `0.9619` on `548× G>>D`; `SGD-G_Equil-D` passes `0.9434` on `620× D>>G` and collapses to `7/16, hq=0.2954` on `548× G>>D`).
4. **Potential counterexamples:** Two severe experimental confounders invalidate interpreting `R4 > R3a` as a *role-intrinsic* law: (a) **Task-design asymmetry** — every pathological depth/rail/encoder task in `r4-occam-ablation-v1` (`grid100_mlp16`, `zero_tanh_rail_escape`, `five_word`, `ddgan`) puts the hard geometry exclusively on $G$, never on $D$; (b) **Direct-OT masking confounder (`Regime B2`)** — in ParticleGAN, `UnifiedV11GANLoss` + `UnifiedV11PriorRegularizer` inject direct data-space Laguerre–Sinkhorn OT gradients into $G$ and `prior`, masking deep-critic failure (`16/16, hq = 0.979–0.986` across all 5 arms even on a 16-layer skip-free critic!).
5. **Minimal falsification experiment:** Run the exact same `3L G + 16L D` architecture in `Full ParticleGAN` (`Regime B2`) vs. `Pure Adversarial RpGAN + Law 2` (`Regime B1a`) and observe the `hq` spread jump from `0.007` to `0.619`.
6. **Relevant prior research:** Causal confounding in ablation suites; Genevay et al. (2018) *Sinkhorn Divergences* (non-adversarial OT gradients bypassing the discriminator).
7. **Computational verification:** Verified in Table 2.2 (`Regime B1a`), Table 2.4 (`Regime B2`), and Table 3b (`Exp 3b`).
8. **Confidence & remaining uncertainty:** **95% certainty**; the `R4 vs R3a` numbers themselves are valid receipts, but the causal attribution to an intrinsic player-role law is confounded by task design and direct OT.

### Claim C3 — Zero-NS $\sqrt{r}$ Polar-Frobenius Critic Rule Reduces Agitation & Energy on Shallow Neural-G Critics (`docs/four-laws.md:163`, `law4-energy-champion.md:65-68`, `law4-optimizer-laws.md:298`)
- **Classification:** **Empirically supported causal mechanism**
1. **Precise formulation:** Replacing 5-step Critic NS with Frobenius-MP scaled by $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ eliminates 100% of 2D Critic NS matmuls and removes orthogonal noise inflation on shallow equilibrium critics (where true gradients have $r_{\text{stable}} \approx 1.1\text{–}1.4$), cutting `r256` energy to `11.37–11.68 J/step` and cracking `staggered100`, `five_word_joint_hold`, and `model_ddgan_bimodal_chain`.
2. **Assumptions:** Valid ONLY when the critic is shallow/well-conditioned enough to tolerate Theorem 3's $\sqrt{m / r_{\text{stable}}}$ spectral-norm overshoot AND when a neural generator + direct-OT pullback are present (fails on deep skip-free critics and generator-free `t3_iso` at `1600` steps).
3. **Existing supporting evidence:** `law4-energy-v10` (`42/48` PASS, `11.68 J/st`), `unified_v12` (`46/48` PASS, `18/18` Tier-3, `11.37 J/st`), plus our Exp 1d proving that orthogonal Critic NS perturbations amplify spatial force-field jitter $\Delta(\nabla_x D)$ by `2.9×–7.8×` relative to scalar logit perturbations!
4. **Potential counterexamples:** Theorem 3 proves exact spectral-norm inflation $\sigma_1(\Delta W^{\text{v12}})/\sigma_1(\Delta W^{\text{Polar}}) = \sqrt{m/r_{\text{stable}}(G_D)}$ (`11.31×` at $m=128, r_{\text{stable}}=1$), which degrades standalone 16L critics (`0.6396` vs `1.0000`, Table 2.0) and pure-adversarial `3L G + 16L D` (`0.2861` vs `0.9055`, Table 2.2).
5. **Minimal falsification experiment:** Evaluate `Frob-MP_v12_sqrt_r` vs. `Frob-MP_spec_matched` vs. `5step_Quintic_NS` on `128×128` synthetic matrices (Table 1c) and on a 16-layer skip-free critic (Tables 2.0, 2.2, 2.3).
6. **Relevant prior research:** Bernstein & Newhouse (2024) *Old Optimizer, New Norm*; Lau et al. (2025) *PolarGrad*; Yang et al. (2023) *Spectral Condition for Feature Learning*.
7. **Computational verification:** Verified in Tables 1c, 1d, 2.0, 2.2, 2.3, and `test_theorem3_sqrt_r_polar_frobenius_spectral_norm_distortion` + `test_hypothesis_e_orthogonal_ns_perturbs_critic_force_field_more_than_logits` (PASS).
8. **Confidence & remaining uncertainty:** **95% certainty** for the mechanism on shallow critics and its exact failure boundary on deep critics; multi-seed confirmatory runs on `staggered100` remain open per `AGENTS.md`.

### Claim C4 — Generator-Free Direct Particle Clouds Are the *"Sole Regime"* Where Dropping Critic NS Hurts (`law4-energy-champion.md:90`)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Asserts that `t3_iso_scale_mass_8x` and `t3_iso_horizon_saddle_shift` ($G = \mathrm{Id}$) represent the unique regime where 0-NS critics degrade, and characterizes the miss as a *"hair-thin `~0.013` residual cost"* (`0.3375 vs 0.35`).
2. **Assumptions:** Extrapolates from Forge suites that contain zero deep skip-free critics and zero pure-adversarial (non-OT) benchmarks.
3. **Existing supporting evidence:** Within the existing 48-task `discriminator_stability` suite, only the two `t3_iso_*` probes regressed when dropping Critic NS.
4. **Potential counterexamples:** (a) Deep skip-free critics (`Regime B1a`: `-0.535` `hq` loss vs. `Both`; `Regime B1b`: `-0.220` `hq` loss of `G-only` vs. `D-only`; standalone `16L D`: `-0.360` valley accuracy); (b) `620×` `nash_overpowered_critic_ring16` at `700` steps (`G-only` `0.8989 FAIL` vs. `Both` `0.9146 PASS` and `Adaptive` `0.9067 PASS`, Table 3b); (c) even on `t3_iso_scale_mass_8x` itself, the frozen threshold is `0.45` (not `0.35`), so `0.3375` is a **`0.1125` miss**, not `0.013`.
5. **Minimal falsification experiment:** Run `Regime B1a` (`3L G + 16L D`), `Exp 3b` (`620×` Nash at `700` steps), and inspect `configs/forge/tasks/t3_iso_scale_mass_8x.json:99`.
6. **Relevant prior research:** Benchmark completeness / construct validity in adversarial optimization.
7. **Computational verification:** Verified in Tables 2.0, 2.2, 2.3, 3b, 3c, and `test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim C5 — `unified_v12`'s Binary `_generator_free = (total_gen_mats == 0)` Routing (`recipes.py:2378-2385`, `_unified_v12/optimizer.py:643,928-938`)
- **Classification:** **Engineering heuristic**
1. **Precise formulation:** Uses a static parameter-count predicate (`total_gen_mats == 0`) to dispatch critic updates between 5-step NS (`_step_generator_free_critic`, `0.30*peak`, `p_dec=0.9995`) and 0-NS $\sqrt{r}$ Frob-MP (`0.50*peak`, `p_dec=1.0`).
2. **Assumptions:** Assumes that the presence of ANY trainable 2D tensor in `(generator, encoder)` implies the critic is shallow/well-conditioned and backed by direct OT.
3. **Existing supporting evidence:** Recovers `16/16` Occam and `46/48` `discriminator_stability` on existing Forge suites while keeping `r256` at `39.22 ms/step` / `11.37 J/step`.
4. **Potential counterexamples:** Adding a single trainable `nn.Linear(2, 2)` (even initialized to identity) disables Critic NS across an entire 16-layer skip-free critic (`test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs`). Furthermore, `stiff_gate` is hardcoded `0.0` on critics (`L759`), preventing deep critics from ever receiving momentum/smoothing release.
5. **Minimal falsification experiment:** Construct `rec.make_optimizers(nn.Identity(), d_deep, prior)` vs. `rec.make_optimizers(nn.Linear(2, 2), d_deep, prior)` and check `opt_d._generator_free` and `opt_d.critic_ns_calls`.
6. **Relevant prior research:** Principled vs. syntactic optimizer dispatch; adaptive spectral conditioning.
7. **Computational verification:** Directly asserted in `test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim C6 — Law 2 Born–Infeld Penalty Supplies Spectral/Jacobian Conditioning That Replaces Critic Spectral Optimization (`docs/four-laws.md:88`, `law4-optimizer-laws.md:114`)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Implies that bounding $\|\nabla_x D\|_2 \le 1$ via Law 2 renders parameter-space spectral conditioning (NS) unnecessary on the critic.
2. **Assumptions:** Conflates a one-sided **upper** spatial-slope cap ($\|\nabla_x D\|_2 \le 1$) with a **two-sided** or **lower** spectral condition-number bound on layer Jacobians $J_\ell$.
3. **Existing supporting evidence:** On shallow `3L+3L` critics (`Exp 3a`), Law 2 ON prevents spatial-slope blowup (`||∇D|| = 0.026–0.054` vs `4.76–10.92` OFF) and keeps `8/8` modes covered.
4. **Potential counterexamples:** (a) Theorem 1 & Table 1a-2 show Law 2 preserves exact rank-1 gradients on linear critics and slightly *lowers* $r_{\text{stable}}$ (`1.511 -> 1.157`) on nonlinear critics; (b) `test_law2_one_sidedness_inactive_on_deep_skip_free_critic` proves the Born–Infeld penalty is `< 1e-10` on a collapsed 16L critic (`||∇D|| = 4.94e-7 << 1`); (c) Exp 3a proves Law 2 ON **fails** to rescue deep 16L critics under `G-only` (`hq = 0.4324`), whereas **Law 2 ON + Critic NS** (`Both`: `0.8562`, `D-only`: `0.7578`) is strictly needed; (d) `t3_iso_scale_mass_8x` fails under 0-NS critic despite Law 2 being strictly satisfied (`||∇D|| <= 1.0`).
5. **Minimal falsification experiment:** Run the `2 × 4` Law 2 (ON/OFF) $\times$ NS (`Neither`/`G-only`/`D-only`/`Both`) factorial on shallow `3L+3L` and deep `3L+16L` critics (`Exp 3a`).
6. **Relevant prior research:** Gulrajani et al. (2017) *WGAN-GP* (two-sided vs one-sided gradient penalties); Mescheder et al. (2018) *Which Training Methods for GANs do actually Converge?* ($R_1$/$R_2$); Miyato et al. (2018) *SNGAN*.
7. **Computational verification:** Verified in Table 1a-2, Table 3a, and `test_law2_one_sidedness_inactive_on_deep_skip_free_critic` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

### Claim C7 — Two Genuine (Non-Universal) Role Asymmetries: Contrastive Mean-Cone Cancellation & Spatial Force-Field Amplification (Theorems 1–2 & Exp 1d)
- **Classification:** **Empirically supported causal mechanism**
1. **Precise formulation:** Holding architecture identical, scalar critics differ from vector generators in two concrete mathematical ways: (i) contrastive `real - fake` subtraction cancels the shared positive-cone mean mode (slashing 16L norm skew `18,820× -> 4.1×` without eliminating deep-layer `net.24..30` collapse), and (ii) orthogonal-to-gradient Critic NS perturbations amplify spatial force-field jitter $\Delta(\nabla_x D)$ by `2.9×–7.8×` relative to scalar logit displacement $\Delta D$.
2. **Assumptions:** Holds for relativistic/contrastive scalar critics ($d_{\text{out}} = 1$) with matched real/fake weights and smooth/piecewise-linear spatial derivatives; contrast cancellation assumes real and fake batches share similar initial cone statistics in deep layers.
3. **Existing supporting evidence:** Direct causal measurements in Exp 1a, 1b, and 1d across 5 architectural regimes (`Regime A`, `B`, `C`, `620×`, `548×`), plus Phase-2 equilibrium agitation measurements in `law4-optimizer-laws.md:298` (`3.8×` lower `dDf` without Critic NS).
4. **Potential counterexamples / boundaries:** Neither effect eliminates deep-critic spectral collapse (`net.24` still hits $r_{\text{stable}} = 1.000000005$ in `D16_f0_contrastive`), nor does force-field amplification outweigh the benefit of Critic NS when $D$ is genuinely collapsed (`Regime B1a/B1b`) or generator-free (`Exp 3c`).
5. **Minimal falsification experiment:** Compare single-sign `-D(real)` vs. contrastive `-(D(real)-D(fake))` on identical `16L` critics (`test_theorem2_contrastive_mean_cone_cancellation_and_deep_critic_collapse`), and measure $\Delta(\nabla_x D)/\Delta D$ under gradient-orthogonal NS perturbations (`test_hypothesis_e_orthogonal_ns_perturbs_critic_force_field_more_than_logits`).
6. **Relevant prior research:** Daneshmand et al. (2020) (positive-cone mean alignment in deep MLPs); Jolicoeur-Martineau (2019) *Relativistic GAN*; Mescheder et al. (2018) (Jacobian dynamics of adversarial training).
7. **Computational verification:** Verified in Tables 1a-1, 1b-1, 1d, and unit tests `test_theorem1_*`, `test_theorem2_*`, `test_hypothesis_e_*` (all PASS).
8. **Confidence & remaining uncertainty:** **95% certainty** on MLPs; untested on attention/transformer GAN critics where LayerNorm already removes mean-cone alignment.

### Claim C8 — Naive Stable-Rank Thresholding ($r_{\text{stable}} < 1.25$) as a Drop-In Role-Agnostic Replacement for Law 4.4 (`Adaptive-Pathology`)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Tests whether a single role-agnostic per-layer threshold `r_stable < 1.25` (computed via scale-invariant power iteration) can match or beat `G-only` on Regime A, beat `G-only` on Regime B, save NS on Regime C, and pass `t3_iso_scale_mass_8x`.
2. **Assumptions:** Assumes that low gradient stable rank ($r_{\text{stable}} < 1.25$) uniquely identifies harmful depth-induced spectral collapse.
3. **Existing supporting evidence:** `Adaptive-Pathology` genuinely proves the **feasibility** of role-agnostic conditioning: it rescues Regime A (`34 -> 98/100, hq 0.115 -> 0.816`), rescues Regime B1a (`hq 0.286 -> 0.811`, beating `G-only` by `+0.441`), passes `620×` Nash (`0.9067`, where `G-only` misses at `0.8989`), and saves `59.4%` NS on `C1` (`hq = 0.9915`) and `58.8%` on `C2` (`img_bars4`).
4. **Potential counterexamples:** It FAILS as a drop-in universal optimum for three measured reasons: (a) **Theorem 1 false-triggering** — healthy shallow scalar critics naturally have equilibrium $r_{\text{stable}} \in [1.10, 1.35]$, so `r_stable < 1.25` fires on **`83.6%` (`3,008/3,600`)** of Regime A's shallow critic steps, negating energy savings (`121.1s` vs `104.1s` for `G-only`); (b) **partial Regime A convergence lag** (`98/100, hq = 0.8156` vs `G-only` `100/100, hq = 0.9714` at `1200` steps) when early/mid layers drop in and out of NS; (c) **Theorem 3 `spec`-fallback failure on `t3_iso_scale_mass_8x`** (`0/5` sustained, `min_eig = 0.4181 < 0.45`, Table 3c).
5. **Minimal falsification experiment:** Run `Adaptive-Pathology` across Regimes A, B1a, C1, C2, and Exp 3c and inspect `ns_D / total_2d_D` on Regime A (`3,008/3,600`) and `core_min_eigen_ratio` on Exp 3c (`0.4181`).
6. **Relevant prior research:** Adaptive preconditioning / dynamic Muon switching; `law4-energy-v8` (`_law4_energy/optimizer.py:4162`, Rényi-2 `erank_2 > 1.6` gate, which similarly missed `grid100_mlp16` live suffix in `law4-energy-champion.md:76`).
7. **Computational verification:** Directly measured in Tables 2.0, 2.1, 2.2, 2.3, 2.6, 2.7, 3b, 3c, and `test_adaptive_pathology_detector_scale_invariance_and_accuracy` (PASS).
8. **Confidence & remaining uncertainty:** **95% certainty** that naive `r_stable < 1.25` is insufficient; **unresolved** whether a compound pathology gate (combining severe collapse $r_{\text{stable}} < 1.05$ OR inter-layer skew $> 10\times$ with non-equilibrium residual) achieves strict Pareto dominance over both `G-only` and `Both`.

### Claim C9 — Documentation & Docstring Drift (`particlegan/unified_v12.py`, `+ Schatten-6`, `0.35` Threshold)
- **Classification:** **Falsified claim**
1. **Precise formulation:** Three factual misstatements in repo docs/docstrings: (a) `docs/four-laws.md:175` & `RESEARCH_GLOSSARY.md:48` cite `particlegan/unified_v12.py`; (b) `_unified_v12/optimizer.py:457` claims `_step_generator_free_critic` runs `+ Schatten-6`; (c) `law4-energy-champion.md:90` quotes `t3_iso_scale_mass_8x` threshold as `0.35`.
2. **Assumptions:** None — direct file-system and AST/JSON inspection.
3. **Existing supporting evidence:** None; directly contradicted by repo ground truth.
4. **Potential counterexamples:** (a) `particlegan/unified_v12.py` does not exist (actual file: `particlegan/_unified_v12/optimizer.py`); (b) `schatten6` is absent from `UnifiedV12Optimizer._step_generator_free_critic`; (c) `configs/forge/tasks/t3_iso_scale_mass_8x.json:99` sets `["core_min_eigen_ratio", ">=", 0.45]`.
5. **Minimal falsification experiment:** Run `test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs`.
6. **Relevant prior research:** Software/scientific provenance hygiene (`AGENTS.md`).
7. **Computational verification:** Asserted in `test_code_audit_unified_v12_hardcoded_asymmetry_and_doc_bugs` (PASS).
8. **Confidence & remaining uncertainty:** **100% certainty**.

---

## 5. Broader Synthesis: Role-Intrinsic vs. Pathology-Intrinsic Verdict

### 5.1 Six-Hypothesis Causal Scorecard

| Hypothesis | Mechanism Tested | Causal Verdict | Primary Evidence |
| :--- | :--- | :--- | :--- |
| **(a) Functional Role** ($G: \mathbb{R}^{d_z}\to\mathbb{R}^d$ vs. $D: \mathbb{R}^d\to\mathbb{R}$) | Scalar output rank-1 bottleneck (Thm 1) + contrastive loss structure (Thm 2) | **Real secondary modulator (not sufficient alone)** | Linear $d_{\text{out}}=1$ exact rank `1`; contrastive subtraction cuts 16L skew `4,567×` yet leaves `net.24..30` collapsed (`s2/s1 = 2.85e-5`). |
| **(b) Architecture Depth / Skip / Norm** | Swapping 16L skip-free MLP between $G$ and $D$ | **PRIMARY causal driver** | `16L G + 3L D`: `G-only` (`0.971`) >> `D-only` (`0.015`). `3L G + 16L D` (PureAdv): **`D-only` (`0.795` / `0.948`) >> `G-only` (`0.370` / `0.729`)** — full reversal! |
| **(c) Law 2 Slope Penalty** | Born–Infeld ON ($\beta=2$) vs. OFF ($\beta=0$) $\times$ 4 NS arms | **Orthogonal to Critic NS (does NOT replace it)** | Law 2 ON is `< 1e-10` on collapsed `16L D` (`hq=0.432` under `G-only`); Law 2 OFF blows up `||∇D||` to `4.8–100.1` even WITH Critic NS. |
| **(d) Gradient Spectrum ($r_{\text{stable}}$, $\operatorname{erank}_{\text{RV}}$)** | Per-layer spectrum evolution across steps & architectures | **Causal witness, but ambiguous near equilibrium** | Deep skip-free layers hit $r_{\text{stable}} = 1.000, \sigma_2/\sigma_1 < 10^{-4}$ (harmful), while healthy shallow critics sit at $r_{\text{stable}} \approx 1.1\text{–}1.35$ (benign). |
| **(e) Function-Space Sensitivity ($\Delta G$, $\Delta D$, $\Delta\nabla_x D$)** | Equal-Frobenius perturbations along `raw`, `ns`, and `ns_orth` | **Real secondary role asymmetry explaining shallow Critic NS harm** | Orthogonal Critic NS perturbs $\nabla_x D$ **`2.9×–7.8×`** more strongly (rel. to raw grad) than scalar logits $D(x)$ (`50%–61%` vs `7%–21%`). |
| **(f) Capacity Ratio ($\rho \gg 1$)** | `620× D>>G` and `548× G>>D` exact Forge Nash hosts | **Symmetric for Law 4.1 equilibration; mild Critic NS gain on `620× D>>G`** | Raw SGD on overpowered player fails symmetrically (`step 5` NaN on `620× D>>G`; `7/16` on `548× G>>D`); `Both`/`Adaptive` pass `620×` at `700` steps (`0.915`/`0.907`) where `G-only` lags (`0.899`). |

### 5.2 Honest Scientific Verdict

**Generator–critic spectral conditioning asymmetry is fundamentally *pathology-intrinsic*, NOT *role-intrinsic*:**
- Whichever player — $G$ **or** $D$ — carries an unnormalized deep skip-free composition (or bears sole geometric responsibility as in generator-free $G = \mathrm{Id}$) suffers spectral contraction ($r_{\text{stable}} \to 1$, $\sigma_2/\sigma_1 \to 0$) and requires spectral singular-value equalization (5-step NS) to escape Theorem 3's Frobenius–Spectral Matching Dilemma.
- Standard GANs (and every Forge suite tested to date) made this pathology-driven asymmetry *appear* role-intrinsic for **four compounding reasons**: (1) standard benchmarks placed 16-layer skip-free and rail pathologies exclusively on $G$ while keeping $D$ shallow (`2–3` layers, Fourier features, or BatchNorm Conv2d); (2) ParticleGAN's direct Laguerre–Sinkhorn OT pullback (`Regime B2`) masks critic degradation on multimodal tasks; (3) contrastive `real - fake` losses cancel the positive-cone mean mode (Theorem 2), softening mid-layer critic skew relative to $G$; and (4) on healthy shallow equilibrium critics, orthogonal NS perturbations get amplified `6.8×–7.8×` by the spatial gradient $\nabla_x D$ (Hypothesis e), creating real equilibrium agitation (`3.8×` `dDf`) whenever NS is applied unnecessarily to a well-conditioned critic.

### 5.3 Design Principle for Next-Generation Pathology-Triggered Conditioning

Our `Adaptive-Pathology` arm proved that a role-agnostic detector can simultaneously rescue deep $G$ (`Regime A`), deep $D$ (`Regime B1a`), and `620×` Nash (`Exp 3b`) while cutting `59%` of NS calls on balanced tasks (`Regimes C1/C2`), **and** revealed exactly why naive `r_stable < 1.25` is not yet optimal:
- A principled **pathology-triggered spectral gate** must separate **depth/contraction collapse** from **benign equilibrium low-rank gradients** by gating 5-step NS on a **compound scale-invariant criterion**:
  1. Severe per-layer spectral anisotropy ($r_{\text{stable}}(G_\ell) < \tau_{\text{sev}} \approx 1.05$ or $\sigma_2/\sigma_1 < 0.1$) **coupled with** an explicit non-equilibrium / contraction witness (e.g., inter-layer norm skew $\max_k \|G_k\| / \min_k \|G_k\| > 10$, non-settled transport residual / `ema_func_gain < 0.6`, or critic input-slope under-utilization $\|\nabla_x D\|_2 \ll \kappa$); and
  2. When NS is bypassed on a collapsed layer (or when transitioning dynamically), avoiding `v12`'s unconstrained $\sqrt{m/r_{\text{stable}}}$ spectral-norm overshoot without starving shallow Frobenius budgets (e.g., smooth interpolation between polar-matched Frobenius scale and spectral-norm cap).

---

## 6. Stronger, Falsifiable Reformulation of Law 4.4 & Exact Proposed Diffs

### 6.1 Proposed Reformulation of Law 4.4

> **Law 4.4 — Pathology-Driven (Not Role-Intrinsic) Generator–Critic Conditioning Asymmetry:**
> Generator–critic optimizer asymmetry is **pathology-intrinsic rather than unconditionally role-intrinsic**, modulated by two genuine role effects and by direct-OT auxiliary losses:
> 1. **Pathology Ownership (Symmetry Under Architectural Swaps):** Spectral singular-value equalization (e.g., 5-step Quintic NS) is required on *whichever* network — generator **or** critic — suffers deep skip-free spectral contraction ($r_{\text{stable}} \to 1$, $\sigma_2/\sigma_1 \ll 1$) or cannot resolve the Frobenius–Spectral Matching Dilemma ($\|\Delta W\|_F / \|\Delta W\|_2 = \sqrt{r_{\text{stable}}} \ll \sqrt{m}$ under scalar rescaling, including generator-free $G = \mathrm{Id}$ direct-cloud critics). Law 2's one-sided upper slope bound ($\|\nabla_x D\|_2 \le \kappa$) prevents spatial-slope explosion ($\|\nabla_x D\| \gg \kappa$) and does **not** cure lower depth contraction ($\|\nabla_x D\| \to 0$).
> 2. **Scope of the Zero-NS Shallow-Critic Optimization:** Dropping 2D Critic NS via Frobenius-MP ($\sqrt{r}$) is a **scoped energy-and-stability optimization** valid specifically when (a) a neural generator is present, (b) the critic is shallow, skip-connected, or BatchNorm-regularized so its hidden layers avoid severe depth-wise spectral collapse, and (c) orthogonal Critic NS would otherwise amplify spatial force-field jitter $\Delta(\nabla_x D)$ at equilibrium (`2.9×–7.8×` relative to scalar logits $\Delta D$). When those conditions fail (deep skip-free unnormalized critics or generator-free direct-cloud critics), dropping Critic NS either inflates the top singular mode by $\sqrt{m / r_{\text{stable}}}$ (`v12` $\sqrt{r}$) or starves the Frobenius budget by $\sqrt{r_{\text{stable}}/m}$ (`G / sigma_1`).
> 3. **Capacity-Imbalance Symmetry (Law 4.1):** Under extreme capacity mismatch ($\rho > 500\times$ in either direction), layerwise/function-space equilibration is required symmetrically on the overpowered player (raw SGD explodes on `620× D >> G` and collapses on `548× G >> D`).

### 6.2 Exact Proposed Diffs (Verified via `git apply --check`, NOT Applied Here)

#### (1) Proposed diff for `docs/four-laws.md` (lines 88, 163, 175)

```diff
--- a/docs/four-laws.md
+++ b/docs/four-laws.md
@@ -85,7 +85,7 @@
 - **Law 4.1 — Inter-Player Function-Space Impedance:** Under player capacity/gain mismatch $\rho \gg 1$ or hostile initialization, equal parameter steps produce $\Theta(\rho)$ function-space displacement mismatch and finite-time divergence unless realized function-space displacements are bounded.
 - **Law 4.2 — Intra-Player Degree-0 Gauge & Rail Escape:** Multiplicative inter-layer Jacobian attenuation across deep networks and $\operatorname{sech}^2$-like activation saturation span many orders of magnitude ($10^{-12}$ to $10^{+2}$), defeating any single player-level learning rate or fixed additive $\varepsilon$ floor. Scale-invariant (degree-0) layerwise conditioning that still settles smoothly as gradients vanish ($\Delta W \to 0$ as $G \to 0$) is required without a step clock.
 - **Law 4.3 — Tensor-Rank Spectral Collapse vs. Low-Rank Noise Inflation:** Elementwise, Frobenius, $L_\infty$, and Kronecker-factored preconditioners preserve gradient rank ($\operatorname{rank}(D_L G D_R) = \operatorname{rank}(G)$) and therefore cannot rescue deep skip-free dense layers whose gradient effective rank collapses ($\operatorname{erank}(G) \to 1$). Conversely, applying aggressive spectral whitening to already-diverse convolutional kernels or naturally low-rank equilibrium gradients wastes compute and injects orthogonal agitation.
-- **Law 4.4 — Generator–Critic Structural Asymmetry under Bounded Critics:** When Law 2 bounds the critic's spatial slope, the generator carries the nonlinear pushforward geometry across disconnected modes, deep compositions, and saturation rails, whereas the critic primarily needs protection against extreme $D \gg G$ capacity imbalance and equilibrium logit jitter.
+- **Law 4.4 — Pathology-Driven Generator–Critic Conditioning Asymmetry:** Spectral conditioning asymmetry is pathology-intrinsic rather than unconditionally role-intrinsic: whichever network ($G$ or $D$) suffers deep skip-free spectral collapse or faces the Frobenius–Spectral Matching Dilemma (including generator-free $G = \mathrm{Id}$ critics) requires spectral singular-value equalization, because Law 2's one-sided upper slope cap ($\|\nabla_x D\|_2 \le \kappa$) cannot cure lower depth contraction. Dropping 2D Critic NS is a scoped optimization valid when a neural generator is paired with a shallow, skip-connected, or BatchNorm-regularized critic — where contrastive real-minus-fake cancellation softens mid-layer skew and orthogonal Critic NS would otherwise amplify spatial force-field jitter $\Delta(\nabla_x D)$ at equilibrium (`2.9×–7.8×` relative to logits $\Delta D$). Under extreme capacity mismatch ($\rho > 500\times$), Law 4.1 equilibration is required symmetrically on the overpowered player.
 - **Law 4.5 — Stochastic Minimax Staleness & Minimum-State Trade-Off:** Once Law 2 damps critic rotational eigenvalues, stochastic minibatch two-player training penalizes stale momentum ($\beta_1 > 0$), high-variance optimistic extrapolation, and heavy coordinate-wise state, favoring low-staleness, minimum-state updates.
 
 **What Any Valid Solution Must Satisfy:**
@@ -160,7 +160,7 @@
 - **Underflow-Safe $10^{-38}$ Dynamic Scale Normalization & Homogeneous `peak_rms` Settling** ([`t4-imbalance-nash.md`](../reports/forge/studies/t4-imbalance-nash.md), [`r3-nash-complete.md`](../reports/forge/studies/r3-nash-complete.md)): Normalizes $(m_t, v_{\text{rms},t})$ by $s_t = \max(\max |v_{\text{rms},t-1}|, \max |g_t|, 10^{-38})$ so $10^{-12}$ early-layer and $\operatorname{sech}^2$-attenuated gradients receive the same directional geometry as $10^{+2}$ output gradients, while degree-0 homogeneous `peak_rms` / momentum-coherence cooling $\text{coherence} = \|\tilde{m}_t\|_F / \|\tilde{v}_t\|_F \to 0$ eliminates equilibrium chatter without a step clock.
 - **Closed-Form Marchenko–Pastur Operator Scaling + Realized Generator Pushforward Trust Band (`delta_g_max`):** Analytically normalizes layer operator gain by $(\sqrt{d_r} + \sqrt{d_c})\sqrt{k_{\text{eff}}}$ in $O(\text{params})$ arithmetic, and measures realized generator output displacement $\delta_G = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ on a detached eval-mode probe to clamp `548×` overpowered generators before they overshoot into mode collapse.
 - **Fused $Z = [X, Y]$ Quintic Tikhonov Polar Orthogonalization & Zero-`pinv` Analytic `direction_blend`** ([`t4-strange-architectures.md`](../reports/forge/studies/t4-strange-architectures.md), [`r4-solution-complete.md`](../reports/forge/studies/r4-solution-complete.md)): Replaces $O(\min(m,n)^3)$ SVD with 5-step pure-GEMM Quintic Newton–Schulz iterations ($a=3.4445, b=-4.7750, c=2.0315$) batched across same-shape 2D matrices, plus exact closed-form $1\times 1 / 2\times 2$ non-ascent constraint projection (`zero_pinv_project_nonascent`, using standard `.backward()` with zero `autograd.grad` and zero `pinv`).
-- **Asymmetric Zero-Conv2d / Zero-Critic NS Energy-Optimal Realizations** ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md)): Applies Laws 4.3–4.5 selectively by dropping redundant 4D Conv2d NS via stride-aware polar-matched Frobenius-MP (`law4-energy-v3`) and dropping 2D Critic NS via exact polar-Frobenius match $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ (`law4-energy-v10`), alongside zero-NS, zero-state research lower bounds (`law4-sign-mp`, `law4-frob-sgd-mp`).
+- **Asymmetric Zero-Conv2d / Zero-Critic NS Energy-Optimal Realizations** ([`law4-optimizer-laws.md`](../reports/forge/studies/law4-optimizer-laws.md)): Applies Laws 4.3–4.5 selectively by dropping redundant 4D Conv2d NS via stride-aware polar-matched Frobenius-MP (`law4-energy-v3`) and dropping 2D Critic NS on shallow/well-conditioned neural-G critics via Frobenius-matched $\sqrt{\max(1,r/c)}\sqrt{m} \equiv \sqrt{r}$ (`law4-energy-v10`; restoring 2D Critic NS on generator-free tasks in `unified_v12`, noting that deep skip-free unnormalized critics require spectral conditioning because $\sqrt{r}$ inflates $\sigma_1$ by $\sqrt{m/r_{\text{stable}}}$), alongside zero-NS, zero-state research lower bounds (`law4-sign-mp`, `law4-frob-sgd-mp`).
 
 **Primary Evidence (Protocol Seed `0`, `NVIDIA RTX A6000`):**
 - **Imbalance-Invariant Nash (`t4_imbalance_nash_stretch`):** `t4-nash-v3` ([`readout-6f996ec2`](../reports/forge/records/readout-6f996ec256a149b2f1620f64.json), [`readout-642d4c55`](../reports/forge/records/readout-642d4c55d093b44ed1e0309e.json)) achieved **`3/3` (`100%`) sustained `5/5` PASS** with **strictly `0` SVD, `0` `eigh`, `0` `pinv`, `0` Newton–Schulz, and `0` `autograd.grad`** at **`39.68 ms/step`** on `256×256` RGB DCGAN (`3.92×` faster than `empf-v8`, `14.84×` faster than `rhbi-v12`), recovering `16/16` modes across a `100×` bidirectional LR-asymmetry sweep (`d_lr_mult` from `10.0` to `0.10`).
@@ -172,7 +172,7 @@
 
 ### Unified Realizations & Occam's Razor Synthesis
 
-All four laws are currently realized together in the `0.12.0` default research recipe **`unified_v12`** (`DEFAULT_RECIPE = "unified_v12"`, [`particlegan/unified_v12.py`](../particlegan/unified_v12.py)), retaining explicit **`unified_v11`** ([`particlegan/unified_v11.py`](../particlegan/unified_v11.py)), while the self-contained **`minimal`** preset (`899` LOC, [`particlegan/minimal.py`](../particlegan/minimal.py)) implements a zero-Newton-Schulz fast core.
+All four laws are currently realized together in the `0.12.0` default research recipe **`unified_v12`** (`DEFAULT_RECIPE = "unified_v12"`, [`particlegan/_unified_v12/optimizer.py`](../particlegan/_unified_v12/optimizer.py)), retaining explicit **`unified_v11`** ([`particlegan/unified_v11.py`](../particlegan/unified_v11.py)), while the self-contained **`minimal`** preset (`899` LOC, [`particlegan/minimal.py`](../particlegan/minimal.py)) implements a zero-Newton-Schulz fast core.
 
 Round-4 Track 2 ([`r4-occam-ablation-razor.md`](../reports/forge/studies/r4-occam-ablation-razor.md), `18` concluded single-factor knockout readouts on `unified-v10`) and Track 1 ([`r4-minimal-core.md`](../reports/forge/studies/r4-minimal-core.md), `899`-LOC `minimal-v2`) tested which parts of our current solutions are mathematically irreducible versus redundant epicycles:
 
```

#### (2) Proposed docstring & threshold corrections (NOT applied here)

```diff
--- a/particlegan/_unified_v12/optimizer.py
+++ b/particlegan/_unified_v12/optimizer.py
@@ -454,7 +454,7 @@
 
     @torch.no_grad()
     def _step_generator_free_critic(self, closure: Any = None) -> Any:
-        """Zero-sync FP32 device-tensor 5-step 2D Critic NS + Schatten-6 strictly for generator-free tasks."""
+        """Zero-sync FP32 device-tensor 5-step 2D Critic NS (zero Schatten-6 decay) strictly for generator-free tasks."""
         loss = None
         if closure is not None:
             with torch.enable_grad():
--- a/reports/forge/studies/law4-energy-champion.md
+++ b/reports/forge/studies/law4-energy-champion.md
@@ -87,7 +87,7 @@
      - **`five_word_joint_hold` PASS:** `24/24` fresh-process continuation checks over 4,000 updates
      - **`model_ddgan_bimodal_chain` PASS**
    - **Swept all three 100-mode lattices** (`grid100` `0.0794`, `rotated100` `0.0919`, `staggered100` `0.1310`), **`4/4` Conv2d discrete images** (`img_stripes2` `1.0000`, `img_intensity2` `1.0000`, `img_bars4` `0.9688`, `img_blobs4` `0.9688` at strictly `0` Conv2d NS & `0` Critic NS), **`6/6` vector tasks** (`vector_unequal_width` `0.9993 / cov=0.2582`, `vector_unequal_mass` `0.9990`, `vector_anisotropic` `0.9954`, `vector_two_broad` `0.9944`, `vector_overlap`, `vector_spiral`), **`3/3` `>548×` hostile Nash stretch** (`0.9590 / 0.9900 / 0.9673` at `0` Critic NS & `0` critic secant probes), **`3/3` precision (`fp32/mxfp8/nvfp4`)**, **`2/2` VAEs**, **`2/2` AEs (`model_ae_gan_multimodal` PASS)**, and **`interp_sparse_arc` PASS (`0.8261`)**.
-   - **Exact Law 4.4 Boundary (4 stretch misses):** Only the two generator-free (`G = nn.Identity`) direct-cloud isolation probes (`t3_iso_scale_mass_8x` `core_min_eigen_ratio = 0.3375 vs 0.35`, `t3_iso_horizon_saddle_shift` `pre_shift_hq = 0.9219` PASS / `hq = 0.9062 vs 0.92`) plus `interp_img_patch_shift` and `interp_saddle_barycentric` missed — cleanly identifying the sole regime (generator-free direct particle clouds where $D$ is the only neural network) where dropping Critic NS incurs a hair-thin `~0.013` residual cost.
+   - **Exact Law 4.4 Boundary (4 stretch misses):** Only the two generator-free (`G = nn.Identity`) direct-cloud isolation probes (`t3_iso_scale_mass_8x` `core_min_eigen_ratio = 0.3375 vs 0.45`, `t3_iso_horizon_saddle_shift` `pre_shift_hq = 0.9219` PASS / `hq = 0.9062 vs 0.92`) plus `interp_img_patch_shift` and `interp_saddle_barycentric` missed — identifying a clear failure mode on generator-free direct particle clouds (and deep skip-free unnormalized critics are likewise vulnerable to dropping Critic NS; see `reports/four-laws-audit/p5-optimizer-asymmetry.md`).
 2. **`r4-occam-ablation-v1` (`readout-01a09e00e51374034f41784b.json`): `14 / 16` PASS (`14 / 14` on every neural G+D probe!)** — including `grid100_mlp16` (`0.9924 / 0.0798`), `model_ddgan_gaussian_chain` (`worst_cdf = 0.0747, t1_q0_eig_max = 1.1209`), `vector_unequal_width` (`0.9993 / 0.2582`), and both `>548×` Nash tasks.
 3. **`dcgan_performance` (`readout-9599d9a20cf9a6e935f0d900.json`): `6 / 6` PASS** — `r256`: `56.35 ms/step` (contended), `1243.43 MiB`; `w128`: `28.87 ms`, `424.46 MiB`; `r128`: `32.28 ms`, `455.39 MiB`; `b64`: `30.70 ms`, `319.58 MiB`; `r64`: `28.35 ms`, `163.43 MiB`; `r32`: `23.45 ms`, `66.54 MiB` (strictly `0` Conv2d NS, `0` Critic NS, `0 B/param` non-stiff state).
 
```
