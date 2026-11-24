# P3 Audit: Joint Adversarial Game Dynamics, Minimax Jacobian Theory & Counterexamples to the Four Laws

> **Audit Scope & Reproducibility:** All proofs, linear/spectral calculations, actual ParticleGAN penalty curvature measurements, and analytic/nonlinear counterexamples are reproducibly implemented in [`scripts/four_laws_audit/p3_game_dynamics.py`](../../scripts/four_laws_audit/p3_game_dynamics.py) and verified by [`tests/four_laws_audit/test_p3_game_dynamics.py`](../../tests/four_laws_audit/test_p3_game_dynamics.py). All ParticleGAN source citations reference branch `research/fl-p3-games` off `origin/research/four-laws` (`17b4c198d`). No existing Forge receipts or `docs/four-laws.md` were mutated; exact proposed edits to `docs/four-laws.md` are specified in Section 6.

---

## Executive Summary & Central Scientific Verdict

**Are the Four Laws in [`docs/four-laws.md`](../../docs/four-laws.md) sufficient for GAN convergence?**
**No — explicitly falsified.** Controlling realized function-space displacement ($\Delta f(x) \approx J_\theta f(x)\Delta\theta$, Law 4.1) regulates **step length and non-normal transient amplification**, NOT **rotational stability** of the joint minimax game Jacobian $J_F$. We construct four explicit, clean counterexamples — three analytic (`A1` 2D bounded 1-Lipschitz Dirac-GAN, `A2` 2D codimension-1 manifold-embedded Gaussian under ParticleGAN's default `non_saturating` logistic loss, `A3` 2D full-support Gaussian under ParticleGAN's public `hinge` loss) and one nonlinear 2D MLP minimax game (`B`) executed with ParticleGAN's **actual** `UnifiedV12Optimizer` (2D Quintic Newton–Schulz + `delta_g_max` secant probes + `peak_rms` smoothing + zero momentum) — that **strictly satisfy every operational requirement of Laws 1–4 as stated in `docs/four-laws.md` yet fail to converge**, entering persistent, macroscopic limit cycles.

**The Hidden Mechanistic Discovery (Why ParticleGAN's *Implementations* Converge While Law 2's *Statement* Fails):**
Inspecting the exact critic-penalty code reveals a decisive gap between what `docs/four-laws.md` **states** and what ParticleGAN's **code actually computes**:

1. **Law 2's statement ([`docs/four-laws.md:43–51`](../../docs/four-laws.md#L43-L51))** requires ONLY a spatial slope upper bound $\|\nabla_x D(x)\|_2 \le \kappa$ plus an anchored zero-mean logit gauge $\mathbb{E}[D] = 0$. At Nash equilibrium $D^* \equiv 0$, $\|\nabla_x D^*\|_2 = 0 < \kappa$: any exact slope cap (`b_cap` hinge, architectural 1-Lipschitz `tanh` critic) has **identically zero curvature** $\partial^2 \mathcal{R}/\partial w^2\big|_0 = 0$, and even `unified_v11`/`v12`'s smooth `F.softplus(norm - 1, beta=12)^2` cap alone has curvature $6.46 \times 10^{-7}$ (exponentially suppressed by $e^{-2\beta\kappa}$). Meanwhile, `gamma_gauge = 0.03` damps **only** the constant bias $b$ ($\partial^2\mathcal{R}/\partial b^2 = 0.03$), leaving critic slope eigenvalues purely imaginary ($\pm i\omega$).
2. **ParticleGAN's implementations secretly inject explicit zero-centered R1/R2 quadratic curvature $\mu_D > 0$ into $J_{DD}$** — in two completely different ways undocumented in `docs/four-laws.md`:
   - **`minimal` ([`particlegan/_minimal/geometry.py:138`](../../particlegan/_minimal/geometry.py#L138))** uses the true Born–Infeld formula $\frac{\sqrt{1 + \beta\|\nabla_x D\|^2} - 1}{\beta} = \frac{1}{2}\|\nabla_x D\|^2 + O(\|\nabla_x D\|^4)$, supplying **full unit R1/R2 curvature** $\mu_D = 1.0803$ (including `+0.08` chord) at $\|\nabla_x D\| = 0$.
   - **`unified_v11` / `unified_v12` ([`particlegan/_unified/penalty.py:741–744`](../../particlegan/_unified/penalty.py#L741-L744), [`particlegan/_unified_v11/penalty.py:354–357`](../../particlegan/_unified_v11/penalty.py#L354-L357))** do **not** use $\sqrt{1+\beta g^2}-1$ in the critic penalty; they use `softplus(norm - 1, beta=12)^2` **plus an undocumented explicit R1+R2 quadratic floor** `lambda_floor = 0.005`, yielding $\mu_D = 0.0100$ (which collapses further to $0.0005/d = 2.54 \times 10^{-9}$ on $256^2$ RGB images, [`particlegan/_unified_v11/penalty.py:683,687`](../../particlegan/_unified_v11/penalty.py#L683-L687)).
   - Additionally, on low-D tasks ($d \le 1024$), `UnifiedV11GANLoss` ([`particlegan/_unified_v11/loss.py:859–959`](../../particlegan/_unified_v11/loss.py#L859-L959)) and `MinimalLoss` ([`particlegan/_minimal/geometry.py:320–329`](../../particlegan/_minimal/geometry.py#L320-L329)) add **non-adversarial geometric transport losses** directly to `g_loss` (injecting $J_{GG} \succ 0$), and [`particlegan/training.py:678–696`](../../particlegan/training.py#L678-L696) uses **1:1 alternating D-then-G updates** (exact determinant $\det(M_{\text{alt}}) = 1 - \eta_D \mu_D$, which contracts at $\mu_D = 0.01, \eta_D = 0.021$ where simultaneous Euler GD diverges!).
3. **Additional Law 4.5 Mathematical Finding (OGDA $\lambda = 0.5$ Neutral Boundary & Schur–Cohn Momentum Collapse):**
   - We prove the exact closed-form Schur–Cohn stability ceiling for simultaneous Heavy-Ball momentum on $\lambda = \gamma \pm i\omega$:
     $$\eta_{\max}(\beta_1; \gamma, \omega) = \frac{2\gamma (1 + \beta_1)(1 - \beta_1)^2}{\gamma^2 (1 - \beta_1)^2 + \omega^2 (1 + \beta_1)^2}.$$
     For underdamped adversarial games ($\omega \gg \gamma$), positive momentum $\beta_1 > 0$ shrinks $\eta_{\max}$ by $(1-\beta_1)^2 / (1+\beta_1)$ — **`82.2×` smaller at $\beta_1 = 0.85$ (Muon) and `190×` smaller at $\beta_1 = 0.90$ (Adam)** — proving a deterministic dynamical reason for Law 4.5 independent of minibatch staleness.
   - Simultaneously, we expose a crucial flaw in ParticleGAN's empirical rejection of Optimistic Extragradient ([`law4-optimizer-laws.md:130,135`](../forge/studies/law4-optimizer-laws.md#L130-L135)): `law4-optimistic-equil` used $\lambda = 0.5$ (`1.5 g_t - 0.5 g_{t-1}`), which sits on the **exact $O(\eta^2)$ neutral boundary $1 - 2\lambda = 0$** for purely imaginary eigenvalues ($\rho = 1.000002 > 1$ at $\mu_D = 0$!). Classical OGDA (Daskalakis et al., ICLR 2018) uses $\lambda = 1.0$ (`2 g_t - g_{t-1}`), which **strictly contracts even at $\mu_D = 0$** (`r_tail = 0.00852` vs `0.24573` for $\lambda = 0.5$ on Counterexample A1), albeit at $5.0\times$ i.i.d. noise variance.

**Governance Verdict (No 5th Law Needed):**
The Four Laws as stated are **design principles / partial necessary conditions under unstated assumptions**, NOT sufficient conditions. However, no 5th law is required: every counterexample is absorbed cleanly and coherently by tightening **Law 2** (requiring *Strict Local Equilibrium Dissipation* $\nabla^2_{\psi\psi} \mathcal{R}|_{D=0} \succ 0$ on normal directions, not merely an upper bound $\|\nabla_x D\| \le \kappa$) and **Law 4** (explicitly formulating *Joint Game-Jacobian Spectral Contraction, Update-Order Symmetry/Alternation, and Non-Normal Transient Conditioning*).

---

## 1. Joint Adversarial Game Jacobian Analysis

Consider a smooth two-player differentiable game with joint parameter vector $x = (\theta, \psi) \in \mathbb{R}^{p+q}$ and pseudo-gradient vector field $F(\theta, \psi) = (\nabla_\theta \mathcal{L}_G(\theta, \psi),\, \nabla_\psi \mathcal{L}_D(\theta, \psi))$ (for zero-sum games, $\mathcal{L}_D = -\mathcal{L}_G + \mathcal{R}(D)$). Near a stationary equilibrium $x^* = (\theta^*, \psi^*)$ with $F(x^*) = 0$, the linearized dynamics are governed by the game Jacobian $J_F = \nabla_x F(x^*)$:

$$J_F = \begin{pmatrix} J_{GG} & J_{GD} \\ J_{DG} & J_{DD} \end{pmatrix} = \underbrace{\frac{1}{2}(J_F + J_F^\top)}_{S\text{ (symmetric / dissipative)}} + \underbrace{\frac{1}{2}(J_F - J_F^\top)}_{A\text{ (antisymmetric / Hamiltonian)}}.$$

For zero-sum games with a critic regularizer $\mathcal{R}(D)$ having Hessian $\mu_D I = \nabla^2_{\psi\psi} \mathcal{R}\big|_{x^*} \succeq 0$ and zero generator regularizer ($J_{GG} = 0$), the canonical 2D (or block-$2d$) linearized Jacobian with interaction frequency $\omega > 0$ and player capacity/gain ratio $\rho > 0$ is:

$$J_F(\mu_D, \omega, \rho) = \begin{pmatrix} 0 & -\omega \rho \\ \omega / \rho & \mu_D \end{pmatrix}, \qquad \lambda_\pm(J_F) = \frac{\mu_D \pm \sqrt{\mu_D^2 - 4\omega^2}}{2}.$$

### 1.1 Symmetric vs. Antisymmetric Spectrum & Simultaneous vs. Alternating Updates

1. **Purely Imaginary Spectrum ($\mu_D = 0$):** When $J_{GG} = J_{DD} = 0$, $J_F = A$ is purely skew-symmetric with $\lambda_\pm = \pm i\omega$.
   - **Simultaneous Euler GD ($x_{k+1} = x_k - \eta F(x_k)$, $M_{\text{sim}} = I - \eta J_F$):**
     $$|\lambda_\pm(M_{\text{sim}})| = |1 \mp i \eta \omega| = \sqrt{1 + \eta^2 \omega^2} > 1 \qquad \forall\,\eta > 0.$$
     Simultaneous Euler GD is **unconditionally exponentially divergent** for every positive step size.
   - **Weakly Damped Simultaneous GD ($0 < \mu_D < 2\omega$):** $|1 - \eta \lambda_\pm|^2 = 1 - \eta \mu_D + \eta^2 \omega^2 < 1 \iff 0 < \eta < \mu_D / \omega^2$. At `unified_v11`/`v12`'s actual $\mu_D = 0.01$ (`lambda_floor = 0.005`), simultaneous GD with $\eta = 0.014$ **strictly diverges** ($\rho_{\text{sim}} = 1.000028 > 1$)!
   - **1:1 Alternating D-then-G GD (Exact Architecture of [`particlegan/training.py:678–696`](../../particlegan/training.py#L678-L696)):**
     $$M_{\text{alt}} = \begin{pmatrix} 1 & \eta_G \omega \rho \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ -\eta_D \omega / \rho & 1 - \eta_D \mu_D \end{pmatrix} = \begin{pmatrix} 1 - \eta_G \eta_D \omega^2 & \eta_G \omega \rho (1 - \eta_D \mu_D) \\ -\eta_D \omega / \rho & 1 - \eta_D \mu_D \end{pmatrix}.$$
     **Theorem 1 (Exact Alternating Symplectic Determinant):** $\det(M_{\text{alt}}) = 1 - \eta_D \mu_D$ identically, independent of $\eta_G, \omega, \rho$.
     - When $\mu_D = 0$: $\det(M_{\text{alt}}) = 1$ (area-preserving symplectic integrator). For $0 < \eta_G \eta_D \omega^2 < 4$, both eigenvalues lie **strictly on the unit circle** $|\lambda_\pm| = 1$, trapping every trajectory on a neutrally stable invariant ellipse (never converging).
     - When $\mu_D > 0$ (underdamped complex conjugate regime): $\rho(M_{\text{alt}}) = \sqrt{1 - \eta_D \mu_D} < 1$ for all $0 < \eta_D < 1/\mu_D$ and small $\eta_G$. Thus 1:1 alternation converts ParticleGAN's weak $\mu_D = 0.01$ from simultaneous divergence ($\rho_{\text{sim}} = 1.000042$) into strict contraction ($\rho_{\text{alt}} = 0.999895$).

| Critic Curvature $\mu_D$ | $J_F$ Eigenvalues ($\omega=1$) | $\rho(M_{\text{sim}})$ ($\eta_G{=}0.014, \eta_D{=}0.021$) | $\rho(M_{\text{alt}})$ (1:1 D-then-G) | $\det(M_{\text{alt}}) = 1 - \eta_D \mu_D$ | Regime |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` *(Law 2 cap only)* | $0.0000 \pm 1.0000i$ | **`1.000147` (DIVERGE)** | **`1.000000` (NEUTRAL ORBIT)** | `1.000000` | Unstable / Non-convergent |
| `0.01` *(`v11/v12` `lambda_floor`)* | $0.0050 \pm 1.0000i$ | **`1.000042` (DIVERGE)** | **`0.999895` (CONTRACT)** | `0.999790` | Alternation saves weak floor |
| `0.10` | $0.0500 \pm 0.9987i$ | `0.999097` | `0.998949` | `0.997900` | Underdamped contracting |
| `1.00` *(`minimal` Born–Infeld)* | $0.5000 \pm 0.8660i$ | `0.989593` | `0.989444` | `0.979000` | Strongly damped spiral |
| `2.00` *(`R1+R2`, Jordan block)* | $1.0000 \pm 0.0000i$ | `0.991124` | `0.991230` | `0.958000` | Critically damped (defective) |

### 1.2 Non-Normality & Kreiss Pseudospectral Transient Amplification (Connection to Law 4.1)

Spectral stability $\rho(M) < 1$ guarantees only *asymptotic* decay $\lim_{k\to\infty} \|M^k\| = 0$, NOT bounded finite-time trajectories. Whenever $J_F$ is non-normal ($J_F J_F^\top \ne J_F^\top J_F$) — either from **player capacity/gain mismatch $\rho \gg 1$** or from **defective Jordan coupling** — its $\varepsilon$-pseudospectrum bulges far outside the unit disk, governed by the **Kreiss Matrix Theorem**:
$$\mathcal{K}(M) \le \sup_{k \ge 0} \|M^k\|_2 \le e\, n\, \mathcal{K}(M), \qquad \mathcal{K}(M) = \sup_{|z| > 1} (|z| - 1)\|(zI - M)^{-1}\|_2.$$

From `scripts/four_laws_audit/p3_game_dynamics.py` (fixed contracting eigenvalues $\mu_D = 0.5, \eta = 0.1 \implies \rho(M) = 0.97980 < 1$ identically across all $\rho$):

| System ($M = I - 0.1 J_F$) | $\|J_F J_F^\top - J_F^\top J_F\|_F$ | Eigenvector $\kappa(V)$ | Spectral Radius $\rho(M)$ | Peak Transient $\sup_k \|M^k\|_2$ | Peak Step $k^*$ | Kreiss $\mathcal{K}(M) \ge$ |
| :--- | ---: | ---: | :---: | ---: | :---: | ---: |
| Balanced ($\rho = 1$) | `1.41` | `1.29` | `0.97980` | `1.02×` | `6` | `1.00` |
| Imbalanced ($\rho = 10$) | `141.59` | `10.33` | `0.97980` | `7.64×` | `14` | `4.06` |
| Imbalanced ($\rho = 50$) | `3,535.71` | `51.64` | `0.97980` | `38.14×` | `14` | `20.00` |
| Imbalanced ($\rho = 100$) | `14,142.31` | `103.28` | `0.97980` | `76.29×` | `14` | `39.99` |
| **Imbalanced ($\rho = 500$, Forge Nash scale)** | **`353,553.57`** | **`516.40`** | **`0.97980`** | **`381.42×`** | **`14`** | **`199.90`** |
| Critically Damped Jordan ($\mu_D = 2, \rho = 1$) | `5.66` | $\infty$ | `0.90000` | `1.01×` | `3` | `1.00` |
| Shifted Jordan Block $[[0.2, 50], [0, 0.2]]$ | `3,535.53` | $\infty$ | `0.98000` | **`92.90×`** | `49` | **`62.47`** |

**Geometric Clarification of Law 4.1:** Even when a minimax game is *spectrally stable* ($\rho(M) = 0.9798 < 1$), a $500\times$ capacity ratio inflates finite-time transient amplification to **`381.4×`** within `14` steps — driving nonlinear neural activations into `NaN` or mode collapse before asymptotic contraction can act! **Law 4.1 (inter-player impedance matching $\rho \to 1$ / `delta_g_max`) is rigorously a non-normality / Kreiss-constant controller** ($\kappa(V) \to O(1)$), NOT an eigenvalue-damping mechanism.

### 1.3 Heavy-Ball Momentum ($\beta_1$) & OGDA ($\lambda$): Exact Schur–Cohn Audit of Law 4.5

For simultaneous Heavy-Ball momentum $x_{k+1} = x_k - \eta F(x_k) + \beta_1 (x_k - x_{k-1})$ applied to an eigenmode $\lambda = \gamma + i\omega$ ($\gamma \ge 0$), the characteristic polynomial is $z^2 - (1 + \beta_1 - \eta\lambda) z + \beta_1 = 0$. Applying the exact complex Schur–Cohn criterion ($|a_0| < 1$ and $|a_1 - \bar{a}_1 a_0| < 1 - |a_0|^2$) yields:

**Theorem 2 (Exact Schur–Cohn Momentum Stability Ceiling):** For any $\beta_1 \in (-1, 1)$ and $\gamma > 0$, simultaneous Heavy-Ball momentum is strictly Schur-stable iff $0 < \eta < \eta_{\max}(\beta_1; \gamma, \omega)$, where:
$$\eta_{\max}(\beta_1; \gamma, \omega) = \frac{2\gamma (1 + \beta_1)(1 - \beta_1)^2}{\gamma^2 (1 - \beta_1)^2 + \omega^2 (1 + \beta_1)^2}.$$
- **Pure optimization limit ($\omega = 0$):** Recovers Polyak's (1964) exact ceiling $\eta_{\max} = \frac{2(1+\beta_1)}{\gamma}$ (positive momentum *helps*).
- **Purely imaginary limit ($\gamma = 0, \omega > 0$):** $\eta_{\max} = 0$; small-$\eta$ expansion gives $|z_1|^2 = 1 + \frac{1+\beta_1}{(1-\beta_1)^3}\eta^2\omega^2 + O(\eta^4)$, so $\beta_1 = 0.85$ accelerates instability **`548×`** and $\beta_1 = 0.90$ **`1,900×`**!
- **Underdamped adversarial game ($\omega \gg \gamma$):** $\eta_{\max} \approx \frac{2\gamma}{\omega^2}\frac{(1-\beta_1)^2}{1+\beta_1}$. Positive momentum **collapses** the stable step-size ceiling **`82.2×` at $\beta_1 = 0.85$ (Muon)** and **`190×` at $\beta_1 = 0.90$ (Adam)**, whereas **negative momentum** $\beta_1 = -0.5$ *expands* $\eta_{\max}$ by **`4.5×`** (matching Gidel et al., ICLR 2019)!
- **Alternating GD + Momentum ($4\times 4$ companion spectrum):** At $\mu_D = 0$, alternating GD has $\rho = 1.000000$ at $\beta_1 = 0$, strictly diverges ($\rho > 1$) for **every** $\beta_1 > 0$, and strictly contracts ($\rho < 1$) for $\beta_1 \in (-1, 0)$!

| $\mu_D$ | $\beta_1$ | Exact $\eta_{\max}$ | $\rho_{\text{sim}}(0.98\,\eta_{\max})$ | $\rho_{\text{sim}}(1.02\,\eta_{\max})$ | $\rho_{\text{sim}}(\eta{=}0.014)$ | $\rho_{\text{alt}}(\eta_G{=}0.014, \eta_D{=}0.021)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.00` | `-0.50` | `0.000000` | — | — | `1.000015` | **`0.999956` (CONTRACT)** |
| `0.00` | `0.00` | `0.000000` | — | — | `1.000098` | `1.000000` *(neutral)* |
| `0.00` | `+0.85` | `0.000000` | — | — | `1.026733` | **`1.032375` (EXPLODE)** |
| `0.00` | `+0.90` | `0.000000` | — | — | `1.042556` | **`1.049909` (EXPLODE)** |
| `0.01` | `-0.50` | `0.044991` | `0.999997` | `1.000003` | `0.999968` | `0.999886` |
| `0.01` | `0.00` | `0.010000` | `0.999999` | `1.000001` | `1.000028` | **`0.999895` (CONTRACT)** |
| `0.01` | `+0.85` | **`0.000122`** | `1.000000` | `1.000000` | `1.026480` | **`1.032026` (EXPLODE)** |
| `0.01` | `+0.90` | **`0.000053`** | `1.000000` | `1.000000` | `1.042314` | **`1.049579` (EXPLODE)** |
| `0.20` | `-0.50` | `0.833333` | `0.998524` | `1.001576` | `0.999081` | `0.998557` |
| `0.20` | `0.00` | `0.200000` | `0.999608` | `1.000408` | `0.998697` | `0.997898` |
| `0.20` | `+0.85` | `0.002457` | `0.999969` | `1.000032` | `1.021539` | `1.025155` |
| `0.20` | `+0.90` | `0.001063` | `0.999980` | `1.000021` | `1.037612` | `1.043096` |

**Theorem 3 (OGDA Neutral Boundary at $\lambda = 0.5$):** For simultaneous OGDA $x_{k+1} = x_k - \eta((1+\lambda)F(x_k) - \lambda F(x_{k-1}))$ on $\lambda(J_F) = \pm i\omega$, expanding the dominant root yields $|z_1|^2 = 1 + (1 - 2\lambda)\eta^2\omega^2 + O(\eta^4)$.
- At **$\lambda = 0.5$** (the exact value tested in ParticleGAN's `law4-optimistic-equil`, [`law4-optimizer-laws.md:130,135`](../forge/studies/law4-optimizer-laws.md#L130-L135)), $1 - 2\lambda = 0$: the $O(\eta^2)$ damping vanishes identically, leaving $O(\eta^4) > 0$ expansion ($\rho_{\text{ogda}} = 1.000002$ at $\eta = 0.05, \mu_D = 0$).
- At **$\lambda = 1.0$** (classical Daskalakis et al., ICLR 2018), $1 - 2\lambda = -1$: OGDA **strictly contracts even at $\mu_D = 0$** ($\rho_{\text{ogda}} = 0.998746$), at the cost of inflating i.i.d. minibatch gradient variance by $(1+\lambda)^2 + \lambda^2 = 5.0\times$ (vs $2.5\times$ at $\lambda = 0.5$ and $1.0\times$ at $\lambda = 0$).

### 1.4 Connection to Mescheder et al. (ICML 2018): Dirac-GAN, Data Manifolds & Logistic Curvature

For a linear critic $D_{\psi, b}(x) = \psi^\top x + b$ near Nash equilibrium $(\theta^*, \psi^*, b^*) = (0, 0, 0)$ with target $y \sim P_Y$ ($\mathbb{E}[y] = 0$, $\operatorname{Cov}(y) = \Sigma_Y$) and $x \sim P_G$ ($\Sigma_X = \Sigma_Y$ at equilibrium):
- **Wasserstein (`loss="wasserstein"`) & Hinge (`loss="hinge"`, [`particlegan/gan_loss.py:37–39`](../../particlegan/gan_loss.py#L37-L39)):** On $|D(x)| < 1$, both are linear in $D$ ($f''(0) \equiv 0$), so $\nabla^2_{\psi\psi}\mathcal{L}_D = 0$ **even on full-dimensional distributions ($\Sigma_Y \succ 0$)**!
- **Non-Saturating & Relativistic Logistic (`loss="non_saturating"`, `"relativistic"`, [`particlegan/gan_loss.py:32–35`](../../particlegan/gan_loss.py#L32-L35)):** Since $\operatorname{softplus}''(0) = \frac{1}{4}$, the loss itself induces
  $$\nabla^2_{\psi\psi}\mathcal{L}_D\big|_{x^*} = \frac{1}{4}(\Sigma_Y + \Sigma_X) = \frac{1}{2}\Sigma_Y.$$
  Along **tangent directions** of $P_Y$ ($\operatorname{range}(\Sigma_Y)$), logistic loss supplies intrinsic curvature; along **every normal direction** orthogonal to a lower-dimensional data manifold $\mathcal{M} \subset \mathbb{R}^d$ ($\ker(\Sigma_Y)$, including Dirac-GAN where $\Sigma_Y = 0$ and real image manifolds of codimension $\gg 1$), $\nabla^2_{\psi\psi}\mathcal{L}_D\big|_{\ker(\Sigma_Y)} = 0$ identically (Mescheder et al. 2018, Lemma 2.1 & Theorem 4.1). Only zero-centered critic gradient penalties (R1/R2 or Born–Infeld's quadratic bottom) populate $J_{DD}$ along $\ker(\Sigma_Y)$.

---

## 2. Code-Level Audit of Actual ParticleGAN Law-2 & Law-4 Mechanisms

### 2.1 Law 2: Exact Local Curvature $\mu_D = \partial^2 \mathcal{R}/\partial w^2\big|_{w=0}$ of ParticleGAN Critic Penalties

Measured directly on `particlegan`'s actual penalty classes in `scripts/four_laws_audit/p3_game_dynamics.py` (`audit_particlegan_penalties()`, $D(x) = w^\top x + b$, $d=2$, `coeff=1.0`, `kappa=1.0`):

| Critic Penalty Implementation | Exact Code Location | Formula Summary | $\partial^2 \mathcal{R}/\partial w^2\big|_0$ ($\mu_D$) | $\partial^2 \mathcal{R}/\partial b^2\big|_0$ (Gauge) | $\partial \mathcal{R}/\partial w\big|_{0.5}$ | Provides $J_{DD} \succ 0$? |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| **`GradientPenalty(arm="b_cap")`** | [`grad_regularizers.py:233–235`](../../particlegan/grad_regularizers.py#L233-L235) | $\frac{1}{2}\big(\operatorname{ReLU}(\|g_r\|-1)^2 + \operatorname{ReLU}(\|g_f\|-1)^2\big)$ | **`0.000000e+00`** | `0.000000e+00` | `0.000000e+00` | **NO** |
| **`UnifiedV11CriticPenalty(lambda_floor=0)`** | [`_unified/penalty.py:730–749`](../../particlegan/_unified/penalty.py#L730-L749) | $\frac{1}{2}\big(\operatorname{sp}_{12}(\|g\|-1)^2 + 0.03\bar{D}^2\big)$ | **`6.463026e-07`** | `3.000000e-02` | `1.340168e-06` | **NO** *(flat interior; gauge damps $b$ only)* |
| **`UnifiedV11CriticPenalty` (Default `v11`/`v12`)** | [`_unified/penalty.py:741–744`](../../particlegan/_unified/penalty.py#L741-L744) | $+\, \frac{1}{2}\cdot 0.005\,(\|g_r\|^2 + \|g_f\|^2)$ *(hidden R1+R2 floor!)* | **`1.000065e-02`** | `3.000000e-02` | `5.001350e-03` | **WEAK YES** *(on low-D; `2.54e-9` on `r256`!)* |
| **`MinimalCriticPenalty` (Default `minimal`)** | [`_minimal/geometry.py:138–141`](../../particlegan/_minimal/geometry.py#L138-L141) | $\frac{\sqrt{1 + 8\|g\|^2}-1}{8} + 0.08\,\text{chord} + 0.03(\bar{D}_r^2+\bar{D}_f^2)$ | **`1.080281e+00`** | `1.200000e-01` | `3.288177e-01` | **STRONG YES** *(`108×` stronger than `v11`/`v12`!)* |
| **`GradientPenalty(arm="a_r1r2")`** | [`grad_regularizers.py:231`](../../particlegan/grad_regularizers.py#L231) | $\frac{1}{2}(\|g_r\|^2 + \|g_f\|^2)$ | **`2.000000e+00`** | `0.000000e+00` | `1.000000e+00` | **STRONG YES** |

**Three Critical Code Discrepancies Exposed:**
1. **`docs/four-laws.md` conflates two mathematically distinct penalties under the name "Born–Infeld":** `minimal` implements the actual Born–Infeld energy $\frac{\sqrt{1+\beta g^2}-1}{\beta}$ (which is **strictly quadratic $\frac{1}{2}g^2$ at $g=0$**), whereas `unified_v11`/`v12` implement a **softplus one-sided cap** $\operatorname{softplus}_\beta(g-\kappa)^2$ (whose interior curvature is $6.5 \times 10^{-7}$, made *even flatter* by `settle_beta = 16.0` raising $\beta: 12 \to 28$ at equilibrium!) **bolted onto an undocumented explicit R1+R2 quadratic floor `lambda_floor = 0.005`** (`0` mentions of `lambda_floor` anywhere in `docs/four-laws.md`!).
2. **High-Resolution Image Floor Dilution:** In [`particlegan/_unified_v11/penalty.py:682–687`](../../particlegan/_unified_v11/penalty.py#L682-L687), the image branch replaces `lambda_floor` (`0.005`) with `0.0005 / float(dim)` — on `dcgan_performance_r256` (`dim = 196,608`), the explicit R1+R2 floor collapses to **$2.54 \times 10^{-9}$**, AND the auxiliary `t1c` Sinkhorn transport coupling turns OFF for `dim > 1024` ([`particlegan/_unified_v11/loss.py:859`](../../particlegan/_unified_v11/loss.py#L859)). High-res `unified_v11`/`v12` therefore has practically zero explicit normal-direction rotational damping.
3. **Non-Adversarial $J_{GG} \succ 0$ Assist in Low-D:** For $d \le 1024$, `UnifiedV11GANLoss.g_loss` ([`loss.py:899–958`](../../particlegan/_unified_v11/loss.py#L899-L958)) and `MinimalLoss.g_loss` ([`geometry.py:320–329`](../../particlegan/_minimal/geometry.py#L320-L329)) add direct data-space Sinkhorn/Born–Infeld transport and covariance losses to `g_loss`, injecting a symmetric positive-definite block $J_{GG} \succ 0$ that masks weak critic dissipation on low-D Forge tasks.

### 2.2 Law 4: Exact Audit of `delta_g_max`, `peak_rms`, Coherence Cooling, and Flip Control

Inspecting [`particlegan/_unified_v11/optimizer.py:714–782,983–1053,1173–1315`](../../particlegan/_unified_v11/optimizer.py#L714-L782), [`particlegan/_unified_v12/optimizer.py:122–170,1023–1046,1151–1230`](../../particlegan/_unified_v12/optimizer.py#L122-L170), and [`particlegan/_unified_v4/optimizer.py:40–56`](../../particlegan/_unified_v4/optimizer.py#L40-L56):

1. **`delta_g_max` (`_band_scale`):** Computes a **strictly positive scalar** $\alpha_G \in [10^{-4}, 48.0]$ from realized output displacement $\delta_{\text{rel}} = \operatorname{RMS}(G_{\theta'}(z) - G_\theta(z))/s_{\text{coord}}$ and rescales all generator parameter updates isotropically: $\Delta\theta \mapsto \alpha_G \Delta\theta$.
   - *Analytic Consequence:* Rescaling $F_G \mapsto \alpha_G F_G$ with $\alpha_G > 0$ only rescales effective generator learning rate $\eta_G^{\text{eff}} = \alpha_G \eta_G > 0$. On $J_F = \begin{pmatrix} 0 & -\omega \\ \omega & 0 \end{pmatrix}$, eigenvalues remain purely imaginary ($\pm i \omega \sqrt{\alpha_G \eta_G \eta_D}$ in continuous time; $|1 \pm i\omega\sqrt{\alpha_G \eta_G \eta_D}| > 1$ in simultaneous Euler; $|\lambda| = 1$ in alternating GD). **`delta_g_max` bounds orbit radius / prevents unbounded blowup, converting divergence into a bounded limit cycle, but CANNOT induce rotational contraction.**
2. **Coherence Cooling (`stiff_gate` + `||m||/||v||`) & `UnifiedV12` CountSketch Flip Controller:**
   - `stiff_gate` activates **only** when `_ema_func_gain < 0.18` (`v11`, [`optimizer.py:716`](../../particlegan/_unified_v11/optimizer.py#L716)) or `< 0.60` (`v12`, [`optimizer.py:124`](../../particlegan/_unified_v12/optimizer.py#L124)). On any balanced, well-conditioned counterexample where $\delta_{\text{rel}} \le \delta_{g,\max}$, `_ema_func_gain == 1.0`, so `stiff_gate == 0.0`, `b1_eff == 0.0`, coherence cooling is **identically inactive (`coh = 1.0`)**, and the CountSketch flip controller ([`optimizer.py:134`](../../particlegan/_unified_v12/optimizer.py#L134)) is **bypassed**.
   - Even if forced on, smooth rotation has step-to-step gradient cosine $\cos(\eta \omega) \approx +0.999 > 0$ (NOT period-2 sign flip $\cos < -0.45$), and turning on `b1_eff = 0.75 * stiff_gate > 0` *destabilizes* purely imaginary eigenmodes by Theorem 2!
3. **`peak_rms` Tikhonov Smoothing (`s_eff = min(0.0061, 0.50 * peak_rms)`):**
   - On an undamped limit cycle of radius $R > 0$, gradient norm $\|g_t\| = \Theta(R)$ never decays to $0$, so `peak_rms` remains $\Theta(R)$ and $g / \sqrt{\|g\|^2 + s_{\text{eff}}^2}$ stays $O(1)$ forever. Even if $R \ll s_{\text{eff}}$, the smoothed step merely reduces to linear alternating GD ($\eta_{\text{eff}} = \eta / s_{\text{eff}}$), which at $\mu_D = 0$ still has exact $\det(M_{\text{alt}}) = 1$ and never converges!

---

## 3. Central Adversarial Challenge: Explicit Counterexamples to All Four Operational Laws

Every counterexample below explicitly satisfies all four operational laws as formulated in `docs/four-laws.md`:
- **Law 1:** Unimodal target, exact topological match (zero disconnected valleys, $\|J_G\| = O(1)$, zero ghost mass).
- **Law 2 (as stated):** Smooth $C^\infty$ critic with verified $\|\nabla_x D(x)\|_2 \le 1$ everywhere (provably $< 1$ in A1–A3 via $D_{\psi,b}(x) = \frac{\tanh\|\psi\|}{\|\psi\|}\psi^\top x + b$; empirically verified $\max\|\nabla_x D\| \le 0.983$ in B) AND zero-mean logit gauge penalty $\frac{\gamma}{2}\bar{D}^2$ (`|E[D]| < 0.05`).
- **Law 3:** Single mode, unit scale, zero beyond-horizon orphans, unconditional (zero fiber sparsity).
- **Law 4:** Balanced capacity ($\rho \approx 1$), shallow smooth well-conditioned Jacobian (no rank collapse, no saturation rails), realized pushforward trust band (`delta_g_max`), degree-0 conditioning (`UnifiedV12Optimizer`), zero momentum ($\beta_1 = 0$).

### 3.1 Analytic Counterexamples A1, A2, and A3 (`steps = 3,000`, Seed `0`)

#### A1 — 2D Bounded 1-Lipschitz Wasserstein Dirac-GAN ($y \equiv 0$, $G_\theta(z) = \theta$, $r_0 = 0.40$)

| Optimizer / Update Rule | Critic Interior Penalty | Tail Radius $r_{\text{tail}}$ | Tail Critic $\|\psi\|_{\text{rms}}$ | $\max \|\nabla_x D\|_2$ | `delta_g` Clamps | `ema_fg` / `stiff` | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Simultaneous GD ($\eta=0.05$) | None ($\mu_D = 0$) | **`1.46739`** | `0.8758` | `0.9486` | `0` | — | **DIVERGES to $\tanh$ wall** |
| **Sim GD + Active `delta_g_max=0.004`** | **None ($\mu_D = 0$)** | **`0.31376`** | **`0.6653`** | **`0.8014`** | **`2,849/3,000`** | — | **FAIL: Persistent Limit Cycle** |
| Alternating 1:1 GD | None ($\mu_D = 0$) | **`0.26064`** | `0.3194` | `0.4442` | `0` | — | **FAIL: Symplectic Orbit** |
| Alt GD + Active `delta_g_max=0.002` | None ($\mu_D = 0$) | **`0.18962`** | `0.4510` | `0.5513` | `2,517/3,000` | — | **FAIL: Persistent Limit Cycle** |
| Alt Adam ($\beta_1 = 0.9$) | None ($\mu_D = 0$) | **`2.28921`** | `0.9865` | `0.9922` | `0` | — | **CATASTROPHIC EXPLOSION** |
| Alt Adam ($\beta_1 = 0.0$) | None ($\mu_D = 0$) | **`0.08459`** | `0.1362` | `0.2508` | `0` | — | **FAIL: Persistent Orbit** |
| **Actual `UnifiedV12Optimizer`** | **None ($\mu_D = 0$)** | **`0.20485`** | **`0.3390`** | **`0.5482`** | `0` | `1.000` / `0.000` | **FAIL: Persistent Limit Cycle** |
| **Actual `UnifiedV12Optimizer`** | **`v11` `softplus` (no floor)** | **`0.20252`** | **`0.3373`** | **`0.5474`** | `0` | `1.000` / `0.000` | **FAIL: `softplus` is flat!** |
| **Actual `UnifiedV12Optimizer`** | **Hidden `lambda_floor=0.005`** | **`0.01340`** | `0.0221` | `0.5319` | `0` | `1.000` / `0.000` | **CONVERGES (`15.3×` drop)** |
| **Actual `UnifiedV12Optimizer`** | **True Born–Infeld (`minimal`)** | **`0.00594`** | `0.0098` | `0.2794` | `0` | `1.000` / `0.000` | **CONVERGES (`34.5×` drop)** |
| Alternating 1:1 GD | True Born–Infeld (`minimal`) | **`0.00000`** | `0.0000` | `0.2599` | `0` | — | **EXACT CONVERGENCE** |
| OGDA ($\lambda = 0.5$, Forge value) | None ($\mu_D = 0$) | **`0.24573`** | `0.2677` | `0.3716` | `0` | — | **FAIL ($2\lambda=1$ neutral!)** |
| **OGDA ($\lambda = 1.0$, Daskalakis)** | **None ($\mu_D = 0$)** | **`0.00852`** | `0.0095` | `0.3598` | `0` | — | **CONVERGES AT $\mu_D = 0$!** |

#### A2 — 2D Codimension-1 Manifold Gaussian Under ParticleGAN's Default `non_saturating` Logistic Loss ($y = (y_1, 0), y_1 \sim \mathcal{N}(0, 1)$, $\theta_0 = (0.4, 0.4)$)

| Optimizer | Critic Interior Penalty | Tangent RMS ($\theta_1$, $\text{Var}=1$) | Normal RMS ($\theta_2$, codim-1) | Tail $\|\psi\|_{\text{rms}}$ | $\max \|\nabla_x D\|_2$ | Mechanism Verified |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Sim GD + `delta_g_max=0.002` (`2,770` clamps) | None ($\mu_D = 0$) | **`0.0069`** *(conv.)* | **`0.2334` (FAIL)** | `0.3910` | `0.6290` | Tangent damped by $f''(0)\Sigma_Y$; normal orbits! |
| Alternating 1:1 GD | None ($\mu_D = 0$) | **`0.0035`** *(conv.)* | **`0.1931` (FAIL)** | `0.2526` | `0.4616` | Exact Mescheder Thm 4.1 tangent/normal split |
| **Actual `UnifiedV12Optimizer`** | **None ($\mu_D = 0$)** | `0.0619` | **`0.0682` (FAIL)** | `0.0758` | `0.4770` | Persistent normal orbit (`ema_fg=1.0`, `stiff=0`) |
| **Actual `UnifiedV12Optimizer`** | **`v11` `softplus` (no floor)** | `0.0576` | **`0.0731` (FAIL)** | `0.0796` | `0.4770` | `softplus` cap gives zero normal damping |
| **Actual `UnifiedV12Optimizer`** | **Hidden `lambda_floor=0.005`** | `0.0680` | `0.0445` *(weak)* | `0.0581` | `0.4724` | Weak partial damping from hidden floor |
| **Actual `UnifiedV12Optimizer`** | **True Born–Infeld (`minimal`)** | `0.0550` | **`0.0000` (EXACT)** | `0.0298` | `0.2130` | Complete normal-axis convergence! |

#### A3 — 2D Full-Support Isotropic Gaussian Under ParticleGAN's Public `hinge` Loss ($y \sim \mathcal{N}(0, I_2)$)

Even with **full-dimensional support** ($\Sigma_Y = I_2$, zero manifold degeneracy), `GANLoss("hinge")` has $f''(0) = 0$ inside $|D(x)| < 1$: `UnifiedV12Optimizer` with `pen=none` stalls at `r_tail = 0.24171` (`psi_rms = 0.2886`, `maxLip = 0.4871 < 1`), `unified_lambda_floor` drops only modestly to `0.21049`, while `born_infeld` cuts error by `2.8×` to `0.08600` (`psi_rms = 0.0529`, finite-batch stochastic floor).

### 3.2 Nonlinear Neural Counterexample B (2D Smooth MLPs + Actual `UnifiedV12Optimizer` & Penalties)

Balanced 2D smooth `SiLU` residual MLP $G$ (`162` params) and `SiLU` MLP $D$ (`129` params, $\rho = 1.26$, `steps = 1,000`, tail = last `300` steps, `ns_2d_activations = 2,000`, `probe_consumptions = 1,000`, Seed `0`):

| Actual ParticleGAN Critic Penalty | Loss | $\beta_1$ | Sim? | Tail $|\bar{x}_2|$ | Tail Peak-to-Peak (`ptp`) | Tail $\operatorname{std}(x_2)$ | Critic Normal Slope | $\operatorname{med}/\max\|\nabla_x D\|$ | $|\mathbb{E}[D]|$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bcap_hinge_gauge`** ($\mu_D = 0$) | `wasserstein` | `0.0` | No | **`0.1640`** | **`0.7160` (HUGE ORBIT)** | `0.1221` | **`0.1659`** | `0.249` / **`0.983`** | `0.0467` |
| **`unified_v11_no_floor`** ($\mu_D = 6.5\text{e-}7$) | `wasserstein` | `0.0` | No | **`0.0703`** | **`0.4102` (FAIL)** | `0.0738` | **`0.0494`** | `0.161` / **`0.734`** | `0.0241` |
| **`unified_v11_default`** (`lambda_floor=0.005`) | `wasserstein` | `0.0` | No | `0.0291` | `0.1834` (`3.9×` drop) | `0.0336` | `0.0081` (`20.5×` drop) | `0.021` / `0.292` | `0.0038` |
| **`minimal_born_infeld`** ($\mu_D = 1.08$) | `wasserstein` | `0.0` | No | **`0.0085`** | **`0.0569` (`12.6×` drop)** | `0.0466` | **`0.0005` (`332×` drop)** | `0.002` / `0.041` | `0.0002` |
| `bcap_hinge_gauge` *(momentum test)* | `wasserstein` | **`0.9`** | No | `0.1638` | `0.7296` | **`0.2432`** | **`0.4296`** | `0.829` / **`2.370`** | **`0.5920`** |
| `bcap_hinge_gauge` *(sim test)* | `wasserstein` | `0.0` | **Yes** | `0.0355` | `0.2018` | `0.0410` | `0.1230` | `0.567` / **`1.432`** | `0.1574` |
| `bcap_hinge_gauge` | `non_saturating` | `0.0` | No | `0.0227` | **`0.1417`** | `0.0145` | `0.0173` | `0.017` / `0.074` | `0.0030` |
| `unified_v11_default` | `non_saturating` | `0.0` | No | `0.0452` | **`0.2404`** | `0.0333` | `0.0073` | `0.011` / `0.095` | `0.0012` |
| `minimal_born_infeld` | `non_saturating` | `0.0` | No | **`0.0048`** | **`0.0477`** | `0.0586` | **`0.0006`** | `0.001` / `0.005` | `0.0001` |

**Causal Interpretation of Counterexample B:**
1. On `wasserstein`, every single Law 1–4 criterion holds on `bcap_hinge_gauge` (`lip_max = 0.983 <= 1.0`, `|E[D]| = 0.0467`, `ns_2d = 2000`, `probes = 1000`, `b1 = 0.0`, balanced $\rho = 1.26$), yet the generator normal mean oscillates across **`0.7160` peak-to-peak** forever.
2. Increasing **only** the zero-centered critic curvature $\mu_D$ produces a **strictly monotonic 4-rung collapse** of both oscillation amplitude (`0.7160 -> 0.4102 -> 0.1834 -> 0.0569`, **`12.6×`**) and normal critic slope (`0.1659 -> 0.0494 -> 0.0081 -> 0.0005`, **`332×`**).
3. **Honest Damping–Variance Trade-Off Disclosed:** Under strong Born–Infeld damping (`minimal_born_infeld`, $\mu_D = 1.08$), normal mean oscillation is crushed (`|mean_n| = 0.0048–0.0085`), but the critic's normal slope drops to `0.0005`, leaving slightly higher residual normal spread (`std_n = 0.0466–0.0586`) than `unified_v11_default` (`std_n = 0.0336`) — explaining *why* ParticleGAN's `unified_v11`/`v12` keeps `lambda_floor = 0.005` small and pairs it with direct transport losses (`t1c`) in low-D, while exposing that `lambda_floor = 0.005` alone still leaves `0.18–0.24` peak-to-peak oscillation on pure adversarial nonlinear games without `t1c`.

---

## 4. Formal Claim Audit: 10-Category Taxonomy & 8-Point Rubric

Each distinct claim is assigned **one primary classification** from the 10-category taxonomy (`reports/four-laws-audit/README.md`) and evaluated against the full 8-point rubric.

### Claim P3-1 — Sufficiency of the Four Laws as Currently Stated in `docs/four-laws.md`
- **Classification:** **Falsified claim**
- **1. Precise formulation:** Satisfying all four operational laws as written in `docs/four-laws.md` (unimodal/topologically matched support, bounded critic slope $\|\nabla_x D\|_2 \le \kappa$ with zero-mean logit gauge, identifiable local scale/fiber, and bounded realized function-space displacement + degree-0 conditioning + $\beta_1 = 0$) guarantees convergence to Nash equilibrium.
- **2. Assumptions:** Smooth differentiable two-player minimax game ($F(\theta,\psi) = (\nabla_\theta \mathcal{L}_G, \nabla_\psi \mathcal{L}_D)$), pure adversarial objective ($J_{GG} = 0$ at equilibrium), Law 2 realized strictly as stated ($\|\nabla_x D\| \le \kappa$ inactive on $\|\nabla_x D\| < \kappa$).
- **3. Existing supporting evidence:** High Forge screening pass rates of `unified_v11` (`55/62`) and `unified_v12` (`46/48` T1–2), which implicitly rely on undocumented `lambda_floor = 0.005`, 1:1 alternation, and low-D non-adversarial `t1c` transport.
- **4. Potential counterexamples:** Explicit Counterexamples `A1` (2D bounded 1-Lipschitz Dirac-GAN), `A2` (codimension-1 manifold Gaussian with default `non_saturating` loss), `A3` (full Gaussian with `hinge` loss), and `B` (nonlinear 2D MLP with actual `UnifiedV12Optimizer` + `b_cap` or `unified_v11(lambda_floor=0)`).
- **5. Minimal falsification experiment:** Run 2D Dirac-GAN (`A1`) with $D_\psi(x) = \tanh(\|\psi\|)\frac{\psi^\top}{\|\psi\|} x + b$, `gamma_gauge = 0.03`, and `UnifiedV12Optimizer` for 2,500 steps: `maxLip < 1.0`, `ema_fg = 1.0`, `stiff = 0.0`, yet `r_tail = 0.205` (non-convergent limit cycle).
- **6. Relevant prior research:** Mescheder, Geiger & Nowozin (ICML 2018, *Which Training Methods for GANs do actually Converge?*); Balduzzi et al. (ICML 2018, *The Mechanics of $n$-Player Differentiable Games*); Letcher et al. (JMLR 2019).
- **7. Computational verification:** Verified in `test_counterexample_a1_dirac_2d_all_four_laws_fails_without_dissipation`, `test_counterexample_a2_manifold_nonsat_logistic_normal_limit_cycle`, and `test_counterexample_b_nonlinear_mlp_actual_particlegan_unified_v12`.
- **8. Confidence & remaining uncertainty:** **Very High (Proven counterexample).** Uncertainty remains only on whether high-res `r256` Conv2d GANs ($d = 196{,}608$, where `t1c` is off and `lambda_floor = 2.54e-9`) experience slow normal-direction limit cycles over long horizons (`> 50k` steps).

### Claim P3-2a — Law 2's Statement ("Bound $\|\nabla_x D\|_2 \le \kappa$ + Zero-Mean Gauge") Damps Critic Rotational Eigenvalues (`docs/four-laws.md:49, 89`)
- **Classification:** **Falsified claim**
- **1. Precise formulation:** `docs/four-laws.md:89` asserts *"Once Law 2 damps critic rotational eigenvalues..."* while `docs/four-laws.md:49` defines Law 2 strictly as $\|\nabla_x D\|_2 \le \kappa$ plus an anchored zero-mean logit gauge.
- **2. Assumptions:** Critic has interior Nash equilibrium $D^* \equiv 0$ ($\|\nabla_x D^*\|_2 = 0 < \kappa$).
- **3. Existing supporting evidence:** None for an upper-bound-only cap; existing Forge passes are confounded by `minimal`'s Born–Infeld quadratic bottom and `unified_v11`/`v12`'s undocumented `lambda_floor = 0.005`.
- **4. Potential counterexamples:** Any exact 1-Lipschitz constraint (`b_cap`, spectral norm, `tanh` slope parametrization) or steep softplus cap (`unified_v11` with `lambda_floor = 0`) has $\mu_D \le 6.5 \times 10^{-7}$, failing to damp rotational eigenvalues; zero-mean gauge (`gamma_gauge = 0.03`) damps only the constant bias $b$, not $\nabla_x D$.
- **5. Minimal falsification experiment:** Evaluate local Hessian curvature of `GradientPenalty(arm="b_cap")` and `UnifiedV11CriticPenalty(lambda_floor=0.0)` at $w=0$, then run Counterexample `A1` and `B`.
- **6. Relevant prior research:** Mescheder et al. (ICML 2018, R1/R2 zero-centered gradient penalties vs WGAN-GP 1-centered penalty); Roth et al. (NeurIPS 2017); Nagarajan & Kolter (NeurIPS 2017).
- **7. Computational verification:** Verified in `test_particlegan_law2_penalty_local_curvature_gap` (`b_cap = 0.0`, `v11_no_floor = 6.46e-7`, `d2R/db2 = 0.030`) and Counterexamples `A1`/`A2`/`B`.
- **8. Confidence & remaining uncertainty:** **Very High (Proven mathematically and verified directly on ParticleGAN code).**

### Claim P3-2b — Zero-Centered Quadratic Critic Curvature ($\mu_D = \partial^2 \mathcal{R}/\partial w^2\big|_0 > 0$) Supplies Rotational Dissipation in `minimal` and `unified_v11`/`v12`
- **Classification:** **Proven mathematical result**
- **1. Precise formulation:** Any critic penalty with strictly positive interior curvature $\mu_D = \nabla^2_{\psi\psi}\mathcal{R}\big|_{D=0} \succ 0$ on normal directions shifts the equilibrium game-Jacobian eigenvalues into the open right half-plane ($\operatorname{Re}(\lambda_\pm) = \mu_D / 2 > 0$). In ParticleGAN, this mechanism is supplied by `minimal`'s Born–Infeld bottom ($\frac{\sqrt{1+\beta g^2}-1}{\beta} \approx \frac{1}{2}g^2$, $\mu_D = 1.0803$) and `unified_v11`/`v12`'s undocumented `lambda_floor = 0.005` ($\mu_D = 0.0100$ in low-D).
- **2. Assumptions:** Linearized game around $D^* \equiv 0$ with nonzero normal-direction interaction $\omega > 0$ and compatible update rule (e.g., 1:1 alternation or $\eta < \mu_D/\omega^2$ under simultaneous Euler GD).
- **3. Existing supporting evidence:** Monotonic 4-rung collapse of rotational oscillation across `b_cap` $\to$ `v11_no_floor` $\to$ `v11_default` $\to$ `minimal_born_infeld` in Counterexamples `A1`, `A2`, `A3`, and `B`, plus Forge stability of `minimal` and `unified_v11`/`v12`.
- **4. Potential counterexamples:** Excessive $\mu_D$ slows normal-variance alignment (`std_n = 0.047–0.059` on `minimal_born_infeld` in `B`); conversely, `0.0005 / dim = 2.54e-9` on `r256` makes `v11`/`v12`'s floor negligible in high-D.
- **5. Minimal falsification experiment:** Toggle `lambda_floor` from `0.0` to `0.005` or swap `b_cap` for `MinimalCriticPenalty` on `A1` and `B` holding all other components fixed.
- **6. Relevant prior research:** Mescheder et al. (ICML 2018, Theorems 4.1 & 4.2 on R1/R2 local convergence); Roth et al. (NeurIPS 2017).
- **7. Computational verification:** Verified in `test_particlegan_law2_penalty_local_curvature_gap`, `test_counterexample_a1_dirac_2d_all_four_laws_fails_without_dissipation`, and `test_counterexample_b_nonlinear_mlp_actual_particlegan_unified_v12`.
- **8. Confidence & remaining uncertainty:** **Very High (Proven linear theorem + direct code verification).**

### Claim P3-3a — Law 4.1 (`delta_g_max` Trust Band) Is Sufficient to Stabilize Minimax Dynamics (`law4-optimizer-laws.md:203`)
- **Classification:** **Falsified claim**
- **1. Precise formulation:** [`law4-optimizer-laws.md:203`](../forge/studies/law4-optimizer-laws.md#L203) claims regulating realized function-space displacement into $[\delta_{g,\min}, \delta_{g,\max}]$ is *"necessary and sufficient"* to stabilize imbalanced two-player minimax games.
- **2. Assumptions:** Purely adversarial minimax game with $\mu_D = 0$; isotropic positive scalar rescaling $\Delta\theta \mapsto \alpha_G \Delta\theta$ ($\alpha_G > 0$).
- **3. Existing supporting evidence:** Forge Nash passes (`t4-nash-v3`, `law4-sgd-secant`, `readout-d5c8d860`) all operated *with* `lambda_floor > 0`, 1:1 alternation, and `t1c` transport active, confounding sufficiency.
- **4. Potential counterexamples:** Counterexamples `A1` (`sim_gd_trust`, `2,849/3,000` active clamps $\implies r_{\text{tail}} = 0.314$; `alt_gd`, `2,517` clamps $\implies r_{\text{tail}} = 0.190$) and `A2` (`2,770` clamps $\implies$ normal RMS `0.233`).
- **5. Minimal falsification experiment:** Actively clamp `delta_g_max = 0.004` on 2D bounded Dirac-GAN (`mu_d = 0`): divergence (`1.467`) converts into a persistent bounded limit cycle (`0.314`), NOT convergence.
- **6. Relevant prior research:** Mescheder et al. (ICML 2018); Balduzzi et al. (ICML 2018).
- **7. Computational verification:** Verified in `test_counterexample_a1_dirac_2d_all_four_laws_fails_without_dissipation`.
- **8. Confidence & remaining uncertainty:** **Very High (Proven by isotropic scaling invariance of imaginary spectrum + explicit active-clamping counterexample).**

### Claim P3-3b — Law 4.1 (`delta_g_max` / Impedance Matching) Controls Non-Normal Kreiss Transient Amplification (`docs/four-laws.md:85, 92`)
- **Classification:** **Proven mathematical result**
- **1. Precise formulation:** Under capacity/gain mismatch $\rho \gg 1$, the linearized game Jacobian $J_F(\mu_D, \omega, \rho)$ retains identical contracting eigenvalues yet suffers departure from normality $\Theta(\rho^2)$, eigenvector conditioning $\kappa(V) = \Theta(\rho)$, and finite-time transient amplification $\sup_k \|M^k\|_2 = \Theta(\rho)$ governed by the Kreiss Matrix Theorem. Inter-player impedance matching ($\rho \to 1$) and realized displacement clamping eliminate this transient blowup.
- **2. Assumptions:** Linearized non-normal two-player Jacobian $J_F$ with $\rho(M) < 1$ and $\rho \gg 1$ (or defective Jordan coupling).
- **3. Existing supporting evidence:** `t4-nash-v3` and `law4-sgd-secant` (`readout-d5c8d860`) convert instant `<2s` `NaN` explosions at `548×–620×` capacity mismatch into `5/5` sustained PASS, while removing `delta_g_max` (`readout-4e2fd695`) collapses `548×` `overpowered_generator`.
- **4. Potential counterexamples:** Does not cure asymptotic rotational instability when $\mu_D = 0$ (see Claim P3-3a).
- **5. Minimal falsification experiment:** Sweep $\rho \in \{1, 10, 100, 500\}$ at fixed $\mu_D = 0.5, \eta = 0.1$ ($\rho(M) = 0.97980$ constant) and measure $\sup_k \|M^k\|_2$ and $\mathcal{K}(M)$.
- **6. Relevant prior research:** Trefethen & Embree (*Spectra and Pseudospectra*, 2005 — Kreiss Matrix Theorem); Heusel et al. (NeurIPS 2017 — TTUR).
- **7. Computational verification:** Verified in `test_non_normality_and_kreiss_transient_amplification` (`1.02×` at $\rho=1$ vs `381.42×` at $\rho=500$, plus `92.90×` on shifted Jordan block).
- **8. Confidence & remaining uncertainty:** **Very High (Exact theorem + direct numerical verification).**

### Claim P3-4 — Alternating vs. Simultaneous Updates Under Weak Critic Dissipation (Unstated Prerequisite of `particlegan/training.py`)
- **Classification:** **Proven mathematical result**
- **1. Precise formulation:** On $J_F(\mu_D, \omega)$, 1:1 alternating D-then-G GD satisfies $\det(M_{\text{alt}}) = 1 - \eta_D \mu_D$, giving neutral stability ($\rho = 1$) at $\mu_D = 0$ and strict contraction ($\rho = \sqrt{1 - \eta_D \mu_D} < 1$) for small $\eta_G$ at ANY $\mu_D > 0$, whereas simultaneous Euler GD requires $\eta < \mu_D / \omega^2$.
- **2. Assumptions:** Linearized 2-player game with $\eta_G \eta_D \omega^2 < 4 - 2\eta_D \mu_D$ and $0 < \eta_D \mu_D < 1$.
- **3. Existing supporting evidence:** [`particlegan/training.py:678–696`](../../particlegan/training.py#L678-L696) hardcodes 1:1 alternating D-then-G updates across all Forge studies; `docs/four-laws.md` omits update order completely.
- **4. Potential counterexamples:** Fails to converge when $\mu_D = 0$ ($\det = 1$, perpetual ellipse), when $\eta_G \eta_D \omega^2 > 4$ (flip bifurcation), or when positive momentum $\beta_1 > 0$ is added.
- **5. Minimal falsification experiment:** Compare simultaneous vs alternating GD at $\mu_D = 0.01$ (`v11`/`v12` exact `lambda_floor = 0.005`) and $\eta_G = 0.014, \eta_D = 0.021$: $\rho_{\text{sim}} = 1.000042 > 1$ (diverges) vs $\rho_{\text{alt}} = 0.999895 < 1$ (contracts).
- **6. Relevant prior research:** Bailey, Gidel & Piliouras (COLT 2020, *Finite Regret and Cycles with Fixed Step-Size via Alternating Gradient Descent-Ascent*); Zhang et al. (ICML 2022).
- **7. Computational verification:** Verified to `1e-12` precision in `test_symmetric_antisymmetric_and_sim_vs_alt_exact` and empirically on Counterexample `B` (`sim=True` breaks `lip_max = 1.432` and inflates `|E[D]|` `3.4×` vs `sim=False`).
- **8. Confidence & remaining uncertainty:** **Very High (Exact theorem).**

### Claim P3-5a — Positive Momentum ($\beta_1 > 0$) Destabilizes Underdamped Rotation & OGDA $\lambda = 0.5$ Sits on the Neutral Boundary
- **Classification:** **Proven mathematical result**
- **1. Precise formulation:** (i) By Theorem 2 (Schur–Cohn), simultaneous Heavy-Ball momentum has exact ceiling $\eta_{\max}(\beta_1;\gamma,\omega) = \frac{2\gamma(1+\beta_1)(1-\beta_1)^2}{\gamma^2(1-\beta_1)^2 + \omega^2(1+\beta_1)^2}$, collapsing by $(1-\beta_1)^2/(1+\beta_1)$ (`82.2×` at $\beta_1=0.85$, `190×` at $\beta_1=0.90$) when $\omega \gg \gamma$, and alternating GD with $\beta_1 > 0$ unconditionally diverges at $\mu_D = 0$; (ii) By Theorem 3, simultaneous OGDA satisfies $|z_1|^2 = 1 + (1 - 2\lambda)\eta^2\omega^2 + O(\eta^4)$ at $\mu_D = 0$, so $\lambda = 0.5$ (`law4-optimistic-equil`) sits on the exact $O(\eta^2)$ neutral boundary ($\rho > 1$) whereas classical $\lambda = 1.0$ strictly contracts ($\rho < 1$).
- **2. Assumptions:** Linearized 2-player game with $\lambda(J_F) = \gamma \pm i\omega$, $\beta_1 \in (-1, 1)$, small $\eta$.
- **3. Existing supporting evidence:** Exact match to `law4-optimizer-laws.md` degradation of `Muon` ($\beta_1=0.85$) and `law4-optimistic-equil` ($\lambda=0.5$), plus Counterexamples `A1` (`Adam b1=0.9` explodes to `2.289`) and `B` (`UnifiedV12 b1=0.9` breaks `lip_max = 2.370`).
- **4. Potential counterexamples:** For pure potential optimization ($\omega = 0$), positive momentum *increases* $\eta_{\max} = 2(1+\beta_1)/\gamma$ (Polyak 1964); negative momentum $\beta_1 < 0$ *stabilizes* undamped alternating games ($\rho < 1$).
- **5. Minimal falsification experiment:** Verify companion-matrix spectral radius at $0.98\,\eta_{\max}$ vs $1.02\,\eta_{\max}$ across $\beta_1 \in [-0.5, 0.9]$, and compare `OGDA(lam=0.5)` vs `OGDA(lam=1.0)` on `A1` ($\mu_D = 0$).
- **6. Relevant prior research:** Polyak (1964); Gidel et al. (ICLR 2019, *Negative Momentum for Improved Game Dynamics*); Daskalakis et al. (ICLR 2018); Mokhtari, Ozdaglar & Pattathil (AISTATS 2020).
- **7. Computational verification:** Verified in `test_schur_cohn_momentum_stability_and_alternating_destabilization`, `test_ogda_neutral_boundary_and_noise_tradeoff`, and `test_counterexample_a1_dirac_2d_all_four_laws_fails_without_dissipation`.
- **8. Confidence & remaining uncertainty:** **Very High (Exact closed-form theorems).**

### Claim P3-5b — Law 4.5 Zero-Momentum Dominance Over Optimistic Extragradient (`docs/four-laws.md:89`; `law4-optimizer-laws.md:223–226`)
- **Classification:** **Necessary condition under stated assumptions**
- **1. Precise formulation:** Zero-momentum ($\beta_1 = 0$) outperforms optimistic extrapolation ($\lambda > 0$) **only under the stated joint preconditions** that (a) explicit critic dissipation ($\mu_D > 0$) plus 1:1 alternation already keep rotational modes inside the unit disk, and (b) stochastic minibatch gradient variance $\sigma_g^2 > 0$ dominates so OGDA's $(1+\lambda)^2 + \lambda^2$ variance inflation ($5.0\times$ at $\lambda = 1.0$) harms terminal settling.
- **2. Assumptions:** $\mu_D > 0$ (from Law 2's equilibrium curvature), 1:1 alternating updates, i.i.d. minibatch gradient noise $\sigma_g^2 > 0$.
- **3. Existing supporting evidence:** `law4-optimizer-laws.md` (`law4-energy-v3` `14/16` > `law4-optimistic-equil` `8/14`), though the Forge ablation confounded OGDA by testing $\lambda = 0.5$ rather than $\lambda = 1.0$.
- **4. Potential counterexamples:** When $\mu_D = 0$ (Counterexample `A1`), the ranking **reverses completely**: zero-momentum `UnifiedV12` (`r_tail = 0.205`) and `alt_gd` (`0.261`) fail to converge, while classical `OGDA(lam=1.0)` converges (`0.0085`).
- **5. Minimal falsification experiment:** Sweep $\mu_D \in \{0.0, 0.01, 1.0\}$ and minibatch noise $\sigma_g \in \{0, 0.1\}$ comparing $\beta_1 = 0$, $\beta_1 = -0.3$, and `OGDA(lam=1.0)`.
- **6. Relevant prior research:** Gidel et al. (ICLR 2019); Daskalakis et al. (ICLR 2018); Chavdarova et al. (NeurIPS 2019, *Reducing Noise in GAN Training with Variance Reduced Extragradient*).
- **7. Computational verification:** Verified in `test_ogda_neutral_boundary_and_noise_tradeoff` and `test_counterexample_a1_dirac_2d_all_four_laws_fails_without_dissipation`.
- **8. Confidence & remaining uncertainty:** **High.** Open empirical question: whether low-variance negative momentum ($\beta_1 \in [-0.3, -0.1]$) or variance-reduced extragradient improves high-res `r256` Conv2d training where `lambda_floor = 2.54e-9`.

---

## 5. Necessity vs. Sufficiency Verdict & Why No 5th Law Is Needed

| Question | Rigorous Verdict | Justification |
| :--- | :--- | :--- |
| Are the current Four Laws **sufficient** for GAN convergence? | **No (Falsified)** | Counterexamples `A1`, `A2`, `A3`, and `B` satisfy every operational requirement of Laws 1–4 as written yet fail to converge. |
| Are the current Four Laws **necessary** for GAN convergence? | **Only partially / conditionally** | Each law identifies a real, isolatable obstruction *for a class of problems*, but some sub-laws (e.g., Law 4.5 $\beta_1 = 0$ optimality) reverse when unstated prerequisites ($\mu_D > 0$, 1:1 alternation) are violated. |
| What is their true epistemic status? | **Design principles & partial necessary obstructions** | They become *locally sufficient on well-posed differentiable games* only once Law 2 and Law 4 explicitly include equilibrium dissipation and joint game-Jacobian contraction. |
| Should we invent a **5th Law** for game rotation? | **No** | Rotational stability is determined directly by the symmetric block $J_{DD} \succ 0$ (owned by **Law 2**'s critic regularizer) interacting with the joint discrete-time operator $M(J_F)$ (owned by **Law 4**'s two-player update rule). Tightening Laws 2 and 4 absorbs all counterexamples cleanly without multiplying axioms. |

---

## 6. Exact Proposed Edits to `docs/four-laws.md` (Not Applied — For Coordinator Review)

Below are the exact minimal, mathematically rigorous edits recommended for `docs/four-laws.md`:

1. **Tighten Law 2 — Critic Potential Geometry ([`docs/four-laws.md:43–51`](../../docs/four-laws.md#L43-L51)):**
   - **Current (line 49):** *"1. **Accurate Local Slope & Gauge Control:** Bound $\|\nabla_x D(x)\|_2 \le \kappa$ and fix the arbitrary additive logit constant across smooth and piecewise-linear critics."*
   - **Proposed replacement:**
     > *"1. **Global Slope Cap, Anchored Gauge & Strict Local Equilibrium Dissipation:** Bound $\|\nabla_x D(x)\|_2 \le \kappa$, fix the arbitrary additive logit constant ($\mathbb{E}[D]=0$), **and** maintain strictly positive local gradient-norm curvature at equilibrium ($\nabla^2_{\psi\psi}\mathcal{R}(D)\big|_{D=0} \succ 0$ along directions normal to the data manifold, e.g., via Born–Infeld's $\frac{1}{2}\|\nabla_x D\|_2^2$ quadratic bottom or an explicit R1/R2 floor $\lambda_{\text{floor}} > 0$). An upper-bound-only Lipschitz cap ($\|\nabla_x D\| \le \kappa$ inactive for $\|\nabla_x D\| < \kappa$) leaves the equilibrium game Jacobian purely antisymmetric ($\lambda = \pm i\omega$) and fails to prevent persistent limit cycles (Mescheder et al., ICML 2018)."*
   - **Also clarify Part II ([`docs/four-laws.md:136`](../../docs/four-laws.md#L136)):** Explicitly distinguish `minimal`'s true Born–Infeld energy $\frac{\sqrt{1+\beta\|\nabla_x D\|^2}-1}{\beta}$ ($\mu_D \approx 1.08$) from `unified_v11`/`v12`'s `softplus(norm - 1, beta=12)^2 + lambda_floor * ||grad D||^2` (`lambda_floor = 0.005` on low-D, `0.0005 / dim` on images) and disclose the non-adversarial `t1c` $J_{GG} \succ 0$ transport assist for $d \le 1024$.

2. **Tighten Law 4.1 & Law 4.5 — Function-Space Two-Player Dynamics ([`docs/four-laws.md:85–95`](../../docs/four-laws.md#L85-L95)):**
   - **Law 4.1 (line 85):** Clarify that bounding realized function-space displacement ($\delta_G \le \delta_{g,\max}$) controls **non-normal Kreiss transient amplification** ($\kappa(V) = \Theta(\rho)$) and prevents finite-time divergence, but is **not** sufficient for asymptotic Nash convergence without rotational damping (either Law 2's $\mu_D > 0$ paired with a non-expanding update order such as 1:1 alternation, or a game-contractive operator like $\lambda=1$ OGDA / negative momentum).
   - **Law 4.5 (line 89):** Replace *"Once Law 2 damps critic rotational eigenvalues..."* with:
     > *"**Law 4.5 — Joint Game-Jacobian Contraction, Update Order & Minimum-State Trade-Off:** Under 1:1 alternating updates ($\det M_{\text{alt}} = 1 - \eta_D \mu_D$) paired with nonzero Law-2 equilibrium dissipation ($\mu_D > 0$), positive Heavy-Ball momentum ($\beta_1 > 0$) shrinks the Schur–Cohn stability ceiling on underdamped rotational modes by $(1-\beta_1)^2/(1+\beta_1)$ (`82×` at $\beta_1=0.85$, `190×` at $\beta_1=0.90$), while stochastic minibatch noise penalizes classical optimistic extrapolation ($\lambda = 1.0$, $5\times$ variance; note $\lambda = 0.5$ sits on the undamped $O(\eta^2)$ neutral boundary) and coordinate-wise state, favoring zero-momentum, minimum-state updates. When $\mu_D = 0$, zero-momentum updates do **not** converge and explicit game-contractive updates ($\lambda = 1$ OGDA or negative momentum $\beta_1 < 0$) are required."*

3. **Explicitly Document Update Order & Scope:** Add to the status banner ([`docs/four-laws.md:3`](../../docs/four-laws.md#L3)) that all ParticleGAN recipes execute **1:1 alternating D-then-G updates** (`particlegan/training.py:678–696`), and frame the Four Laws accurately as **empirically grounded necessary obstructions / design principles** (not sufficient theorems unless tightened per above).
