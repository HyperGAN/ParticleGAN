# Relativistic Hellinger-Born-Infeld Transport (RHBI)

## Overview

**Relativistic Hellinger-Born-Infeld Transport (`rhbi`)** replaces unscreened adversarial generator objectives with a **Debiased Slack-Screened Unbalanced Sinkhorn** variational operator (`RHBILoss` / `RHBIGANLoss`) paired with a **Weyl-phased balanced Born-Infeld geodesic chord + state-settling endpoint Lipschitz cap** (`RHBICriticPenalty`), a **Poisson-Weibull Ajtai-Komlós-Tusnády (AKT) empirical OT local horizon** ($\ell_t$), an **impedance-matched false-shock-proof 6-channel Kantorovich-Hellinger thermal equilibrium controller**, and degree-of-freedom-aware SBNP particle prior regularization (`prior_reg = 0.02`, `prior_lr_mult = 2.9`) under strictly constant learning rates (`lr = 0.014`, `lr_floor = 1.0`, `network_lr_floor = 1.0`), zero EMA critic copies, and zero task-shape or step-count gates.

## Target Failure Modes

1. **Poisson Shot-Noise Floor ($B \lesssim M$) & Empty-Valley Bridging:** Linear Born-Infeld horizon screening ($\rho = 3.0$, `rhbi_rapidity_screening = False`) routes unmatched batch singletons and empty-valley midpoint particles ($\hat{c}_{ij} > \rho$) cleanly into the $(B+1)$-th Kantorovich-Hellinger slack reservoir rather than trapping them in 50/50 cross-mode active couplings, while `rhbi_slack_gain = 0.35` amplifies relativistic log-odds ascent in proportion to total slack fraction $s_f$.
2. **1D Empirical OT Horizon Contraction ($d = 1$):** Nearest-neighbor spacings scale as $\big(\frac{-\ln(1-q)}{2 e B_r}\big)^{1/d}$ whereas empirical optimal transport fluctuations scale as $B^{-\min(1/d,\,1/2)}$ (Ajtai-Komlós-Tusnády). Scaling the $q$-th quantile real NN distance by $\big(\frac{2 e B_r}{-\ln(1-q)}\big)^{\max(0,\,1/d - 1/2)}$ (`rhbi_akt_horizon = True`, `rhbi_akt_mode = "weibull"`) centers 1D Gaussian horizons on continuum OT scales (eliminating false 2-sigma tail-slack leakage and unlocking both MXFP8 and NVFP4 1D Gaussian acquisition) while equaling exact identity $1.0$ for all $d \ge 2$ without task-shape branching.
3. **Intra-Mode Barycenter Shrinkage, Discrete Codebook Clone Separation & Wide-Span Connected-Manifold Settling:** Discrete-codebook clone-deduplicated Feydy Sinkhorn self-coupling (`rhbi_self_coupling = "feydy_discrete"`) cancels intra-mode entropic variance contraction and masks off-diagonal zero-distance sampling clones strictly on zero-noise `ParticlePrior` codebooks (without discarding low-bit MXFP8/NVFP4 quantized output bins on continuous `MoG` priors), paired with Pareto-protected Weyl-stratified $k$-NN 1-simplex/2-simplex Born-Infeld manifold regularization (`rhbi_simplex_weight = 0.35`), $H_0$-persistence connected-manifold geodesic chart alignment (`rhbi_simplex_chart_weight = 4.0`), and wide-span macro-simplicial thermal settling (`rhbi_simplex_macro_settle = True`).
4. **Dead-Zone Starvation, Geodesic Chord Conditioning & Thermally Coupled Settling Lipschitz Walls:** Combining real/fake endpoint gradient caps with a deterministic Weyl-phased balanced-Sinkhorn Born-Infeld geodesic chord (`rhbi_chord_weight = 0.25`, `ebi_beta = 12.0`, `rhbi_settle_mode = "thermal"`, `rhbi_settle_beta = 16.0`, `rhbi_gamma_gauge = 0.03`) enforces unit Lipschitz slope across inter-mode valleys while keeping $\beta_{\text{eff}} \equiv 12.0$ throughout pre-equilibrium mode acquisition ($s_t = 1$) and smoothly sharpening softplus slope walls up to $\beta_{\text{eff}} = 28.0$ as thermal cooling ($s_t \to 10^{-6}$) locks equilibrium, without consuming training RNG streams or gating on input tensor shape.
5. **Constant-LR Post-Capture Chatter, Valley-Gated Tail-Precision Spill & Shock-Safe Convex-Unimodal Reacquisition:** Under unit-norm DualNorm updates, uncooled post-capture steps execute finite limit-cycle chatter. A clock-free 6-channel Kantorovich-Hellinger two-sample residual tracker smoothly scales both critic and generator gradients down to $s_{\min} = 10^{-6}$ at equilibrium, boosts convex-unimodal barycenter sensitivity (`rhbi_cvx_mu_gain = 0.58`) with a shock-safe $0.045$ single-batch ceiling while rejecting stationary/quantized single-batch barycenter jitter (`rhbi_thermal_shock_mu = 0.35`), delays premature cooling via a cyclic-valley-gated continuous-prior 96th-quantile tail-precision spill channel (`rhbi_tail_spill_gain = 0.95`) while inter-mode bridges remain occupied, and reopens in 1 step on genuine distribution shifts.
6. **Singleton Continuous-Conditioned Fibres:** On continuous-conditioned critic tuples $D(c, x)$ (`rhbi_conditional_fibre = "slack"`), singleton conditional fibres ($c_i \neq c_j$) combine strictly row-local fibre-centered paired relativistic log-odds with unbiased hollow multi-scale Inverse Multiquadric conditional MMD transport (`rhbi_cond_ot_mode = "imq_cmmd"`, `rhbi_cond_ot_weight = 1.5`) without cross-fibre batch mixing or diagonal paired supervision.

## Formulation Summary

- **Poisson-Weibull AKT Scale-Equivariant Local Horizon ($\ell_t$):** Zero-sync detached EMA ($\beta_\ell = 0.1$) of $\big(\frac{2 e B_r}{-\ln(1-q)}\big)^{\max(0,\,1/d - 1/2)} Q_{q}(d^{\text{NN}}_Y)$ ($q = 0.25$), with scale-equivariant tie resolution on discrete multi-atom codebooks.
- **Overflow-Safe Born-Infeld Ground Cost:** Rationalized $c_{\text{phys}}(x_i, y_j) = \sqrt{\ell_t^2 + \|x_i - y_j\|_2^2} - \ell_t$ and dimensionless screening cost $\hat{c}_{ij} = c_{\text{phys}}(x_i, y_j) / \ell_t$, $C^\infty$ at $x_i = y_j$ and finite up to `FLT_MAX`.
- **Slack-Screened Sinkhorn Operator (`rhbi-v11`):** Deterministic $K = 8$ `logsumexp` iterations on the $(B_f + 1) \times (B_r + 1)$ kernel with linear active energy $\hat{c}_{ij}$, critic-potentiated slack borders $(-\frac{\rho}{2} \pm \frac{\tau}{2}\tilde{d})/\varepsilon$ ($\rho = 3.0, \tau = 2.0, \varepsilon = 0.1$), slack-gain boost $1 + \gamma_{\text{slack}} s_f$ (`rhbi_slack_gain = 0.35`), interior slack reservoir marginal $S = 2$, discrete-codebook clone-deduplicated Feydy self-coupling debiasing (`rhbi_self_coupling = "feydy_discrete"`), Pareto-protected Weyl-stratified simplicial manifold projection (`rhbi_simplex_weight = 0.35`, `rhbi_simplex_chart_weight = 4.0`, `rhbi_simplex_macro_settle = True`), shift-invariant slack-screened relativistic pairing on joint BiGAN hosts, and fibre-centered paired relativistic + hollow multi-scale IMQ conditional MMD updates on continuous-conditioned fibres.
- **Thermally Coupled Weyl-Phased Geodesic Chord Penalty & Zero-Mean Gauge:** `RHBICriticPenalty` combines real/fake endpoint Born-Infeld caps ($\beta_0 = 12.0, \beta_{\text{settle}} = 16.0$ under `rhbi_settle_mode = "thermal"`, $\lambda_{\text{floor}} = 0.005$) with a deterministic golden-ratio Weyl-phased balanced-Sinkhorn geodesic chord (`rhbi_chord_weight = 0.25`) and zero-mean logit gauge (`rhbi_gamma_gauge = 0.03`), skipping batch-doubling concatenation on stateful batch-bound critic view modules.
- **Impedance-Matched 6-Channel Thermal Controller & Complete Continuation:** Both `_RHBIObserverStepRecord` and `RHBIStepRecord` persist schema-3 state (`ell_ema`, `ell_initialized`, `chord_phase`, 6 residual EMAs including `ema_tail`, and `thermal_scale`) with exact bit-for-bit continuation across restarts and tamper validation.
- **Hybrid Exact-Polar / Quintic DualNorm Optimization:** Pairs exact smoothed polar SVD on dense 2D weights via CPU LAPACK (`optimizer_dense_profile = "polar_cpu_v1"`, `optimizer_smoothing = 0.0061`) with the batched quintic spectral filter on wide CUDA convolution slices (`optimizer_convolution_rule = "quintic"`).

## Implementation Sources

- `particlegan/rhbi.py`
- `particlegan/_rhbi/primitive.py`
- `particlegan/_rhbi/loss.py`
- `particlegan/_rhbi/critic.py`
- `particlegan/_rhbi/penalty.py`
- `configs/forge/ideas/rhbi-v1.json`
- `configs/forge/ideas/rhbi-v2.json`
- `configs/forge/ideas/rhbi-v3.json`
- `configs/forge/ideas/rhbi-v4.json`
- `configs/forge/ideas/rhbi-v5.json`
- `configs/forge/ideas/rhbi-v6.json`
- `configs/forge/ideas/rhbi-v7.json`
- `configs/forge/ideas/rhbi-v8.json`
- `configs/forge/ideas/rhbi-v9.json`
- `configs/forge/ideas/rhbi-v10.json`
- `configs/forge/ideas/rhbi-v11.json`
- `configs/forge/studies/rhbi-v1-study.json`
- `configs/forge/studies/rhbi-v2-study.json`
- `configs/forge/studies/rhbi-v3-study.json`
- `configs/forge/studies/rhbi-v4-study.json`
- `configs/forge/studies/rhbi-v5-study.json`
- `configs/forge/studies/rhbi-v6-study.json`
- `configs/forge/studies/rhbi-v7-study.json`
- `configs/forge/studies/rhbi-v8-study.json`
- `configs/forge/studies/rhbi-v9-study.json`
- `configs/forge/studies/rhbi-v10-study.json`
- `configs/forge/studies/rhbi-v11-study.json`
