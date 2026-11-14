<!-- Generated Forge family report -->

# RHBI

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [selectable-loss](../technique-inventory.md#tag-selectable-loss)

## Technique overview

Relativistic Hellinger-Born-Infeld Transport (RHBI) is a clock-free, constant-learning-rate ParticleGAN formulation with no EMA critic copy and no task-shape or step-count gates. It unifies nonlocal Poisson-Weibull Ajtai-Komlos-Tusnady (AKT) scale-equivariant Born-Infeld Optimal Transport, discrete-prior-aware Feydy self-coupling entropic debiasing, slack-gain-amplified local relativistic critic log-odds ascent, multi-scale IMQ conditional MMD fibre geometry, and Weyl-stratified k-NN simplicial manifold projection with H_0-persistence connected-manifold geodesic chart alignment and wide-span macro-simplicial settling inside a single (B+1)x(B+1) slack-screened Sinkhorn variational operator, paired with a Weyl-phased balanced Born-Infeld geodesic chord + thermally coupled settling endpoint Lipschitz cap, an impedance-matched shock-safe convex-unimodal and valley-gated tail-precision 6-channel Kantorovich-Hellinger thermal equilibrium controller, and degree-of-freedom-aware SBNP particle prior regularization.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior with $N$ locations in $\mathbb R^{d_z}$, $y=\{y_j\}_{j=1}^{B_r}$ a real batch, $z\sim P$ a latent batch and $x=\{x_i\}_{i=1}^{B_f}$ with $x_i=G(z_i)$ the generated batch. $\ell_t$ is the exponential moving average ($\beta_\ell=0.1$) of the Poisson-Weibull AKT-scaled detached 25th-quantile real nearest-neighbor distance $(\frac{2eB_r}{-\ln(1-q)})^{\max(0,1/d-1/2)}Q_q(d^{\mathrm{NN}}_Y)$ ($q=0.25$). $c_{\mathrm{phys}}(x_i,y_j)=\sqrt{\ell_t^2+\lVert x_i-y_j\rVert_2^2}-\ell_t$ is the physical Born-Infeld ground cost and $\hat c_{ij}=c_{\mathrm{phys}}(x_i,y_j)/\ell_t$ its dimensionless linear screening form. $\tilde d(u)=C_{\max}\tanh((D(u)-\bar D)/C_{\max})$ is the batch-centered, bounded critic log-odds. $P^{\star(XY)}$ and diagonal-included $P^{\star(XX)}$ are deterministic Sinkhorn couplings on the $(B+1)\times(B+1)$ active-plus-slack polytope ($\rho=3,\tau=2,\varepsilon=0.1,S=2$) with slack-gain boost $1+\gamma_{\mathrm{slack}}s_f$ ($\gamma_{\mathrm{slack}}=0.35$). $s_t\in[10^{-6},1]$ is the 6-channel Kantorovich-Hellinger thermal scale and $\beta_{\mathrm{eff}}=\beta_0+\beta_{\mathrm{settle}}(1-s_t)^2$ ($\beta_0=12,\beta_{\mathrm{settle}}=16$) is the thermally coupled Lipschitz cap sharpness.

**Poisson-Weibull AKT Scale-Equivariant Born-Infeld Cost and Local Horizon**

$$
c_{\mathrm{phys}}(x_i,y_j)=\sqrt{\ell_t^2+\lVert x_i-y_j\rVert_2^2}-\ell_t,\qquad \hat c_{ij}=\frac{c_{\mathrm{phys}}(x_i,y_j)}{\ell_t},\qquad \ell_t=(1-\beta_\ell)\ell_{t-1}+\beta_\ell\,\operatorname{stopgrad}\!\left(\left(\frac{2eB_r}{-\ln(1-q)}\right)^{\max(0,\frac{1}{d}-\frac{1}{2})}Q_{q}(d^{\mathrm{NN}}_Y)+10^{-6}\right)
$$

Evaluated via float64 GEMM and algebraic rationalization so gradients remain smooth at zero distance and finite across the full float32 range, matching exact quadratic W2 across 1D and multidimensional modes (including quantized MXFP8 and NVFP4) while bounding inter-mode pull slopes inside $[-1,+1]$.

**Impedance-Matched Slack-Screened Unbalanced Sinkhorn Objective**

$$
\mathcal L_{\mathrm{RHBI}}(X,Y;\psi)=\lambda_{\mathrm{rhbi}}s_t\left[\sum_{i=1}^{B_f}\sum_{j=1}^{B_r}\operatorname{stopgrad}\!\left(P^{\star(XY)}_{ij}\right)c_{\mathrm{phys}}(x_i,y_j)-\sum_{i=1}^{B_f}\operatorname{stopgrad}\!\left(P^{\star(XY)}_{i,0}(1+\gamma_{\mathrm{slack}}s_f)\right)\frac{\tau}{2}\tilde d(x_i)-\frac{1}{2}\sum_{i,k=1}^{B_f}\operatorname{stopgrad}\!\left(\sqrt{a_i a_k}P^{\star(XX)}_{ik}\right)c_{\mathrm{phys}}(x_i,x_k)\right]
$$

Linear horizon screening ($\rho=3$) routes valley outliers ($\hat c_{ij}>\rho$) to slack-gain-boosted centered critic log-odds ascent ($\tau/2=1,\gamma_{\mathrm{slack}}=0.35$), Feydy diagonal self-coupling cancels intra-mode entropic contraction, and false-shock-proof 6-channel Kantorovich-Hellinger thermal scale $s_t$ locks equilibrium without step clocks.

**Thermally Coupled Settling Weyl-Phased Geodesic Chord and Endpoint Born-Infeld Critic Penalty**

$$
R_{\mathrm{RHBI}}(D)=\frac{\lambda}{2}\sum_{u\in\{y,x\}}\mathbb E_u\!\left[\phi_{\beta_{\mathrm{eff}}}(\lVert\nabla_u D(u)\rVert_2-\kappa)^2+\lambda_{\mathrm{floor}}\lVert\nabla_u D(u)\rVert_2^2\right]+\lambda w_{\mathrm{chord}}\mathbb E_{\hat x}\!\left[\phi_{\beta_{\mathrm{eff}}}(\lVert\nabla_{\hat x}D(\hat x)\rVert_2-\kappa)^2\right]+\frac{\gamma_{\mathrm{gauge}}}{2}\bar D^2
$$

Softly caps critic input slopes at $\kappa=1$ ($\beta_{\mathrm{eff}}=12+16(1-s_t)^2$, identically $12$ while $s_t=1$ during mode acquisition) on real/fake endpoints and deterministic Weyl-phased balanced-Sinkhorn geodesic interpolants ($w_{\mathrm{chord}}=0.25$) while anchoring mean critic gauge ($\gamma_{\mathrm{gauge}}=0.03$) without transport-shape gates or an EMA critic anchor.

## Simplified pseudocode

```text
Fix one global configuration with constant learning rates and DualNorm updates across generator, encoder, critic and row-normalized particle prior.
For each task-owned training update:
  Sample real batch y and latent batch z from the task prior P; form detached fake batch x_det = stop_gradient(G(z)).
  Evaluate thermally scaled non-saturating logistic critic loss plus the Weyl-phased balanced Born-Infeld geodesic chord and thermally coupled settling endpoint input-gradient cap (with zero-mean gauge 0.03 and soft slope floor 0.005) on y and x_det.
  Backpropagate and step D via DualNorm; clear transient critic input-capture ring.
  With D fixed, resample latent z and compute differentiable fake batch x = G(z) together with critic forward captures.
  If continuous conditioning accompanies critic tuples D(c, x), evaluate fibre-centered paired relativistic log-odds softplus(D(c_i, y_i) - D(c_i, x_i)) plus unpaired multi-scale inverse-multiquadric conditional MMD (imq_cmmd, weight 1.5) over joint (c, x) fibres.
  Otherwise update detached Poisson-Weibull AKT local horizon ell_t on device from y, update false-shock-proof 6-channel two-sample Kantorovich-Hellinger thermal scale s_t (with valley-gated continuous-prior 96th-quantile tail-precision channel and 1-step shift reopening, deferring discrete-prior thermal observation to simplicial manifold evaluation when active), and solve K=8 deterministic log-domain Sinkhorn scaling steps for active-plus-slack coupling P*(XY) and discrete-prior-aware Feydy self-coupling P*(XX) (feydy_discrete, masking off-diagonal minibatch particle clones on discrete ParticlePrior hosts while preserving continuous MoG bin collisions).
  Evaluate the impedance-matched Danskin envelope RHBI generator loss combining active Born-Infeld pull, slack-gain-amplified critic ascent and Feydy self-coupling debiasing (or slack-screened relativistic log-odds on joint BiGAN hosts), plus SBNP prior regularization, any task-owned auxiliary losses, and Pareto-protected Weyl-stratified k-NN 1-simplex/2-simplex Born-Infeld manifold regularization (weight 0.35, augmented with H_0-persistence connected-manifold vertex Sinkhorn anchoring, fatness-guarded triangle inradius margins, local inverse-quartic shortest-path geodesic chart stress weight 4.0, and wide-span macro-simplicial thermal settling) on discrete ParticlePrior codebooks.
  Project conflicting protected directions via direction-blend constraint geometry when declared, and step G, P and optional encoder E.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Critics minimize thermally scaled non-saturating logistic loss; unconditional and discrete-class generators minimize the impedance-matched RHBI slack-screened unbalanced Sinkhorn envelope objective combining physical Born-Infeld transport, discrete-prior-aware Feydy self-coupling debiasing (feydy_discrete), slack-gain-boosted centered critic log-odds ascent, and Pareto-protected simplicial manifold projection on discrete particle codebooks; continuous-conditioned fibres minimize fibre-centered paired relativistic log-odds plus multi-scale IMQ conditional MMD; joint BiGAN generator/encoder hosts minimize slack-screened relativistic log-odds across active pairs and slack marginals. |
| Optimizer | Full DualNorm across generator, encoder, critic and row-normalized particle prior (zero momentum, polar smoothing 0.0061, exact CPU-LAPACK smoothed polar factor on dense 2D weights, quintic per-offset convolution rule on wide CUDA kernels, direction-blend constraint geometry) without an EMA critic copy. |
| Learning rates and annealing | Strictly constant base learning rates across all tasks: global LR $0.014$, critic multiplier $1.5$ and prior multiplier $2.9$, with network and prior LR floors fixed at $1.0$ (zero schedule annealing and zero horizon dependence). |
| Parameter-gradient clipping | None on parameter tensors; active transport coordinate gradients are analytically bounded inside $[-1,+1]$ by the Born-Infeld ground cost and critic log-odds are smoothly bounded by $C_{\max}\tanh(\cdot/C_{\max})$. |
| Critic penalties and anchors | Smooth C-infinity Born-Infeld input-gradient cap ($\lambda=1$, $\kappa=1$, base sharpness $\beta_0=12$ settling smoothly up to $\beta_{\mathrm{eff}}=28$ via thermally coupled $\beta_{\mathrm{settle}}(1-s_t)^2$ with $\beta_{\mathrm{settle}}=16$, sub-unit slope floor $\lambda_{\mathrm{floor}}=0.005$, zero-mean logit gauge $\gamma_{\mathrm{gauge}}=0.03$) evaluated on real/fake endpoints plus a deterministic golden-ratio Weyl-phased balanced-Sinkhorn geodesic chord ($w_{\mathrm{chord}}=0.25$), strictly unperturbed during pre-equilibrium mode acquisition ($s_t=1$), with zero EMA critic anchor and automatic non-concatenated evaluation on stateful batch-bound critic view wrappers. |
| Damping and update guards | Clock-free, shape-gate-free 6-channel Kantorovich-Hellinger thermal equilibrium controller scales critic and generator gradients down to $s_{\min}=10^{-6}$ at distribution lock, boosts convex-unimodal barycenter sensitivity (`rhbi_cvx_mu_gain=0.58`) with a shock-safe $0.045$ single-batch ceiling while rejecting stationary/quantized single-batch barycenter jitter via shock ratio $0.35$, adds a cyclic-valley-gated continuous-prior 96th-quantile tail-precision spill channel (`rhbi_tail_spill_gain=0.95`) that delays premature cooling while anisotropic inter-mode bridges remain occupied, defers discrete-prior cooling to joint vertex+barycenter macro-simplicial settling (`rhbi_simplex_macro_settle=true`) on wide-span $H_0$-connected manifolds, and reopens in 1 step on genuine distribution shifts; critic spike guard ratio=$0$, latent damping=$0$; slack reservoir marginal $S=2$ keeps Sinkhorn iterations strictly inside the Birkhoff polytope while degree-of-freedom-aware SBNP regularization ($\lambda_{\mathrm{prior}}=0.02$) stabilizes particle geometry across dimensions. |
| Training and sampling noise | Zero additive critic-input or generator-output training noise ($0.0$). Task-declared prior intrinsic sigma and serving contracts are preserved unchanged. |
| Parameter averaging and serving | Generator EMA decay=$0$ and serve_average=$0$; all evaluations score live unaveraged weights. |

## Configuration differences

- RHBI uses one global trainer configuration across every Tier 1, Tier 2, Tier 3 and Tier 4 task without per-task tuning, step-count gates or input-shape regimes.
- Poisson-Weibull AKT horizon scaling (`rhbi_akt_mode='weibull'`) multiplies the $q$-th quantile real NN distance by $(\frac{2eB_r}{-\ln(1-q)})^{\max(0,1/d-1/2)}$, centering 1D empirical OT horizons on continuum Gaussian scales (unlocking quantized MXFP8 and NVFP4 1D Gaussian acquisition) while remaining exact identity $1.0$ for all $d\ge 2$ without shape branching.
- When discrete class labels accompany critic inputs, cross-class active pairs are masked to minus infinity in log-Sinkhorn space so unmatched class draws route cleanly into the slack reservoir; when continuous conditioning accompanies critic tuples D(c, x), rhbi_conditional_fibre='slack' routes generator updates via fibre-centered paired relativistic log-odds without cross-fibre mixing.
- Both passive observer records and formulation critic records checkpoint schema-3 state (ell_ema, ell_initialized, chord_phase, 6 residual EMAs including ema_tail, and thermal_scale) with zero per-step host-device synchronizations, ensuring exact bit-for-bit continuation across restarts and rejecting tampered or incompatible checkpoints.
- Setting rhbi_weight=0.0 executes an exact zero-strength bypass that leaves horizon buffers and RNG streams untouched.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/rhbi.py](../../../particlegan/rhbi.py)
- [particlegan/_rhbi/primitive.py](../../../particlegan/_rhbi/primitive.py)
- [particlegan/_rhbi/loss.py](../../../particlegan/_rhbi/loss.py)
- [particlegan/_rhbi/critic.py](../../../particlegan/_rhbi/critic.py)
- [particlegan/_rhbi/penalty.py](../../../particlegan/_rhbi/penalty.py)
- [particlegan/optim/dualnorm.py](../../../particlegan/optim/dualnorm.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/training.py](../../../particlegan/training.py)
- [configs/forge/ideas/rhbi-v1.json](../../../configs/forge/ideas/rhbi-v1.json)
- [configs/forge/ideas/rhbi-v2.json](../../../configs/forge/ideas/rhbi-v2.json)
- [configs/forge/ideas/rhbi-v3.json](../../../configs/forge/ideas/rhbi-v3.json)
- [configs/forge/ideas/rhbi-v4.json](../../../configs/forge/ideas/rhbi-v4.json)
- [configs/forge/ideas/rhbi-v5.json](../../../configs/forge/ideas/rhbi-v5.json)
- [configs/forge/ideas/rhbi-v6.json](../../../configs/forge/ideas/rhbi-v6.json)
- [configs/forge/ideas/rhbi-v7.json](../../../configs/forge/ideas/rhbi-v7.json)
- [configs/forge/ideas/rhbi-v8.json](../../../configs/forge/ideas/rhbi-v8.json)
- [configs/forge/ideas/rhbi-v9.json](../../../configs/forge/ideas/rhbi-v9.json)
- [configs/forge/ideas/rhbi-v10.json](../../../configs/forge/ideas/rhbi-v10.json)
- [configs/forge/ideas/rhbi-v11.json](../../../configs/forge/ideas/rhbi-v11.json)
- [configs/forge/studies/rhbi-v1-study.json](../../../configs/forge/studies/rhbi-v1-study.json)
- [configs/forge/studies/rhbi-v2-study.json](../../../configs/forge/studies/rhbi-v2-study.json)
- [configs/forge/studies/rhbi-v3-study.json](../../../configs/forge/studies/rhbi-v3-study.json)
- [configs/forge/studies/rhbi-v4-study.json](../../../configs/forge/studies/rhbi-v4-study.json)
- [configs/forge/studies/rhbi-v5-study.json](../../../configs/forge/studies/rhbi-v5-study.json)
- [configs/forge/studies/rhbi-v6-study.json](../../../configs/forge/studies/rhbi-v6-study.json)
- [configs/forge/studies/rhbi-v7-study.json](../../../configs/forge/studies/rhbi-v7-study.json)
- [configs/forge/studies/rhbi-v8-study.json](../../../configs/forge/studies/rhbi-v8-study.json)
- [configs/forge/studies/rhbi-v9-study.json](../../../configs/forge/studies/rhbi-v9-study.json)
- [configs/forge/studies/rhbi-v10-study.json](../../../configs/forge/studies/rhbi-v10-study.json)
- [configs/forge/studies/rhbi-v11-study.json](../../../configs/forge/studies/rhbi-v11-study.json)
- [reports/forge/rhbi/README.md](../rhbi/README.md)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

## Performance

DCGAN on deterministic mock RGB data measures finite-update speed and memory. These separate performance-source receipts confer no image-quality, scientific qualification or default-promotion credit. Standard BatchNorm is retained in generator and critic; critic input gradients therefore include batch coupling. The shared critic uses a flattened 4×4 Linear scalar head for policy feature extraction; its module layout is fixed across recipes.

Synchronized wall time for one public trainer update, including its critic/generator/prior work; warmup, construction, mock-data generation, evaluation, GIFs and checkpoint writes are excluded.

Every cell is one run; timings are never pooled across devices or contention states. Compare only rows with the same task and matched key (source, runtime/dtype, physical device, initialization and batch streams). Unknown or contended observations are unpaired. Resolution progression changes the task, while each candidate keeps one global recipe. Images/s is B divided by full update time (one D and one G/prior update with separate real batches). Three warmup and ten measured updates give short-window median/p95 measurements; long-horizon policy costs and component phase times remain unknown.

[Compact performance evidence index](../scaling-performance.json) · [Protocol and bounded reproduction](../../../docs/forge-scaling-performance.md)

This registration covers 10 complete recipes. Historical alternatives outside its registered family roster add no missing benchmark cells. A changed selected recipe within a registered family remains unknown until measured.

### NVIDIA RTX A6000 · GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce

Performance source `28267be05193` (commit `5a6af38465597761d0b6a00a6243910565850d35`); runtime `fddddb1b314d`; torch.float32; PyTorch 2.14.1+cu130; 48592.125 MiB device memory.

Task prior: particle_cloud, sigma 0.0, 256 initial locations. The common finite-table performance exception retains each recipe's prior update policy and regularizer. Exact prior and resource specifications remain part of every task key.

| Family / recipe | Task (RGB size, batch, width) | Execution | Median / p95 update ms | Images/s | GPU peak allocated / reserved MiB | Contention | Matched key / evidence |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RHBI · `rhbi-v11` | dcgan_performance_r32: 32², B16, W64 | COMPLETE | 45.56 / 46.41 | 350.09 | 98.55 / 150.00 | clear | `d56240d15558` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/rhbi-dcgan_performance_r32-e30fc2f398e75960.json) |
| RHBI · `rhbi-v11` | dcgan_performance_r64: 64², B16, W64 | COMPLETE | 82.17 / 83.16 | 194.86 | 693.48 / 846.00 | clear | `524c07acf3db` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/rhbi-dcgan_performance_r64-66394ff24c922a42.json) |
| RHBI · `rhbi-v11` | dcgan_performance_w128: 64², B16, W128 | COMPLETE | 179.11 / 180.02 | 89.41 | 685.19 / 1,146.00 | clear | `7ca95dbe0fca` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/rhbi-dcgan_performance_w128-e873a9e4b8a489fc.json) |
| RHBI · `rhbi-v11` | dcgan_performance_b64: 64², B64, W64 | COMPLETE | 140.25 / 142.47 | 455.85 | 876.75 / 1,250.00 | clear | `0c19272217b1` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/rhbi-dcgan_performance_b64-7aa9acc188dbbdf6.json) |
| RHBI · `rhbi-v11` | dcgan_performance_r128: 128², B16, W64 | COMPLETE | 200.00 / 201.83 | 79.90 | 1,254.52 / 1,902.00 | clear | `1b8df8094599` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/rhbi-dcgan_performance_r128-1d5549b4684e7fa6.json) |
| RHBI · `rhbi-v11` | dcgan_performance_r256: 256², B16, W64 | COMPLETE | 588.84 / 590.37 | 27.18 | 3,738.55 / 6,356.00 | clear | `c4765c66fb33` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/rhbi-dcgan_performance_r256-3b75e5cfb89b8f38.json) |

<a name="cohort-cuda-23a94f16ca88"></a>

## Current benchmark

Runtime: **cuda**. Selected configuration: [rhbi-v11](../../../configs/forge/ideas/rhbi-v11.json).

Comparison policy: discriminator_stability revision 9, **tier 1**. Actual executed-policy qualification: **tier 1**, discriminator_stability revision 17 (FAIL). These are separate policy results; no qualification reuse.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `759e40be701e99e2de416754283c99228011a45dc7090ac9b52d8a8a8fda4776`. Candidate revision: `05f71faf03f06323d92bc7def373a5310d32c91bea38bf62b760725e29415f55`. Runtime cohort: `5d64d3269823362b4f406b4d354672ad99f01603d9f6a43dd8f6735c3e7c7e7e`.

[Frozen numerical evidence](../technique-evidence/5addc6ae0c348a468fa76db9ffc295d2753f964b706112dba668979437f0f4ee.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Promote rhbi-v11 (Relativistic Hellinger-Born-Infeld slack-screened transport with Poisson-Weibull 2e empirical OT 1D horizon scaling, mu-scaled intra-mode W2 convex-potential gradient correction rhbi_cvx_mu_gain=0.58, discrete-codebook clone-deduplicated Feydy self-coupling debiasing feydy_discrete, Pareto-protected Weyl-stratified detached-vertex k-NN 1-simplex/2-simplex Born-Infeld manifold projection 0.35 plus H0-connected non-isotropic chart simplex regularization rhbi_simplex_chart_weight=4.0 and wide-span macro-simplicial thermal settling rhbi_simplex_macro_settle=true, hollow multi-scale IMQ conditional MMD fibre geometry 1.5, mu=0.35 false-shock filter, thermally coupled (1 - s_t)^2 Lipschitz settling, Weyl-phased balanced-Sinkhorn Born-Infeld geodesic chord regularization 0.25 at ebi_beta=12.0 with zero-mean critic gauge 0.03, and impedance-matched valley-gated tail-precision 6-channel Kantorovich-Hellinger thermal equilibrium controller rhbi_thermal_tau=0.084, rhbi_tail_spill_gain=0.95) as the canonical RHBI candidate following ordinary discriminator_stability CUDA measurement on NVIDIA RTX A6000 (11/11 required + 2/2 diagnostic Tier 1 PASS = 27/27 Tier 1 view requirements across views, 11/18 required Tier 2 PASS = 56/78 Tier 2 view requirements, 83/115 total counted view requirements plus 28/63 stretch/held view assignments; 32/41 executed discriminator_stability PASS and 31/64 full-suite PASS including held vector_anisotropic across all 6 views, stretch interp_saddle_barycentric, gaussian1d_stability, interp_img_patch_shift, all three 1D precision benchmarks precision_gaussian_fp32, precision_gaussian_mxfp8, and precision_gaussian_nvfp4, held image benchmarks img_blobs4 and img_intensity2 alongside img_stripes2, plus mode_hold and vector_overlap, 0 BLOCKED). Actual executed-policy qualification and comparison-policy projection retain their separate source-bound proofs; no public-default adoption claim.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation) | [3/3](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) | [9/12](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) | [0/1 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-4) | [12/16 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-adaptation) | [4/7](rhbi.md#cohort-cuda-23a94f16ca88-held) |
| [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous) | [4/4](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) | [9/12](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) | [0/6 (6 not run)](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-4) | [13/22 (6 not run)](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous) | [4/7](rhbi.md#cohort-cuda-23a94f16ca88-held) |
| [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability) | [11/11](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | [11/18 (2 not run)](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | [0/1 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | [22/30 (3 not run)](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability) | [8/28 (16 incompatible)](rhbi.md#cohort-cuda-23a94f16ca88-held) |
| [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison) | [3/3](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) | [9/12](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) | [0/1 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-4) | [12/16 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison) | [4/7](rhbi.md#cohort-cuda-23a94f16ca88-held) |
| [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer) | [3/3](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) | [9/12](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) | [0/1 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-4) | [12/16 (1 not run)](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer) | [4/7](rhbi.md#cohort-cuda-23a94f16ca88-held) |
| [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage) | [3/3](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | [9/12](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-4) | [12/15](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage) | [4/7](rhbi.md#cohort-cuda-23a94f16ca88-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [lazy_cadence_coverage](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-2) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-4) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage) |
| [tier1_policy_coverage](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage) | [0/7 (7 not run)](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-2) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-3) | [0/0](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-23a94f16ca88-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_gan_multimodal](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_reconstruction](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_vae_categorical](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_vae_hard](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | PASS | matches recorded run |
| [zero_tanh_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [precision_gaussian_mxfp8](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [precision_gaussian_nvfp4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [model_ae_ddgan_bimodal_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [model_ae_ddgan_gaussian_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [model_ddgan_bimodal_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [model_ddgan_gaussian_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [interp_img_patch_shift](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | PASS | matches recorded run |
| [interp_saddle_barycentric](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | PASS | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | [adaptation](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-3), [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) | BLOCKED | matches recorded run |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [clockfree_continuous](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3), [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3), [formulation_comparison](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3), [host_profile_transfer](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN |

<a name="cohort-cuda-23a94f16ca88-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](rhbi.md#cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cgan_overlapping_classes](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cramer_energy_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cyclegan_unpaired_constellation](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_mlp16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_res16_scaled_tanh](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_res16_tanh](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [infogan_anisotropic_modes](rhbi.md#cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_block_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_mcar_p50_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_mcar_p80_anchored](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [pacgan8_grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [pacgan_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | [discriminator_stability](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc) | [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | Can 12 zero-sigma ParticlePrior locations learn a 270-degree circular arc without MoG so unseen latent k-NN chords stay on the arc without snapping? | Consider counted status only after independent calibration and one global recipe that passes chord interpolation while retaining every counted pass. |
| [precision_gaussian_mxfp8](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [interp_img_patch_shift](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift) | [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | PASS | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | Can 16 zero-sigma ParticlePrior codes learn continuous 8x8 single-patch images without MoG so unseen latent chords shift one bright patch instead of 0.5 ghosts? | Consider counted status only after independent calibration and one global recipe that passes image latent interpolation while retaining every counted pass. |
| [interp_saddle_barycentric](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric) | [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | PASS | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | Can 24 zero-sigma ParticlePrior locations learn a curved 3D saddle sheet without MoG so unseen 2D latent triangle barycenters stay on the curved surface? | Consider counted status only after independent calibration and one global recipe that passes 2D simplex interpolation while retaining every counted pass. |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](rhbi.md#cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](rhbi.md#cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-23a94f16ca88-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality"></a>

## bispf_fastpath_quality

**bispf_fastpath_quality — revision 1**. [View declaration](../../../configs/forge/views/bispf_fastpath_quality.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 5**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **4 / 12 / 6 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1"></a>

## ddgan-conditional-repair-v1

**ddgan-conditional-repair-v1 — revision 2**. [View declaration](../../../configs/forge/views/ddgan-conditional-repair-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 17**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **11 / 18 / 1 / 0**. Held counts (measured and shown, not counted): **3 / 9 / 1 / 15**. Its highest tier, Tier 4, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | required | PASS | matches recorded run |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | required | PASS | matches recorded run |
| [zero_tanh_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke) | required | PASS | matches recorded run |
| [model_ae_gan_multimodal](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) | required | PASS | matches recorded run |
| [model_ae_reconstruction](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction) | required | PASS | matches recorded run |
| [model_vae_categorical](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | required | PASS | matches recorded run |
| [model_vae_hard](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [precision_gaussian_fp32](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_fp32) | diagnostic | PASS | matches recorded run |
| [precision_gaussian_mxfp8](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | PASS | matches recorded run |
| [precision_gaussian_nvfp4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | PASS | matches recorded run |
| [interp_sparse_arc](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | required | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | required | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | PASS | matches recorded run |
| [model_ae_ddgan_gaussian_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) | required | FAIL | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [model_ddgan_gaussian_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) | required | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | matches recorded run |
| [model_ddgan_bimodal_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | matches recorded run |
| [interp_img_patch_shift](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | PASS | matches recorded run |
| [interp_saddle_barycentric](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [zero_tanh_rail_escape](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [grid100_res16_tanh](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_block_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_mcar_p50_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_mcar_p80_anchored](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [pacgan_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [pacgan8_grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [alice_spiral_bigan](rhbi.md#cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [infogan_anisotropic_modes](rhbi.md#cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cgan_overlapping_classes](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cramer_energy_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cyclegan_unpaired_constellation](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [grid100_res16_scaled_tanh](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |
| [img_intensity2_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 6**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |
| [img_intensity2_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage"></a>

## lazy_cadence_coverage

**lazy_cadence_coverage — revision 1**. [View declaration](../../../configs/forge/views/lazy_cadence_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | diagnostic | UNKNOWN | matches recorded run |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | diagnostic | UNKNOWN | matches recorded run |
| [clockfree_audit_measurement_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1"></a>

## misgan-bispf-research-v1

**misgan-bispf-research-v1 — revision 2**. [View declaration](../../../configs/forge/views/misgan-bispf-research-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Scoped research diagnostic; no ordinary qualification or default adoption.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_misgan_four_mode](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-particle_interpolation_stretch"></a>

## particle_interpolation_stretch

**particle_interpolation_stretch — revision 1**. [View declaration](../../../configs/forge/views/particle_interpolation_stretch.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Three zero-sigma particle interpolation stretch toys do not establish whole-suite screen calibration or default adoption.

<a name="cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc) | diagnostic | FAIL | matches recorded run |
| [interp_img_patch_shift](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift) | diagnostic | PASS | matches recorded run |
| [interp_saddle_barycentric](rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1"></a>

## prior-ablation-full-suite-v1

**prior-ablation-full-suite-v1 — revision 1**. [View declaration](../../../configs/forge/views/prior-ablation-full-suite-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [zero_tanh_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke) | diagnostic | PASS | matches recorded run |
| [model_ae_ddgan_gaussian_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | FAIL | matches recorded run |
| [model_ae_gan_multimodal](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) | diagnostic | PASS | matches recorded run |
| [model_ae_reconstruction](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction) | diagnostic | PASS | matches recorded run |
| [model_ddgan_gaussian_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) | diagnostic | FAIL | matches recorded run |
| [model_vae_categorical](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | diagnostic | PASS | matches recorded run |
| [model_vae_hard](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | diagnostic | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [clockfree_audit](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | matches recorded run |
| [model_ddgan_bimodal_chain](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | matches recorded run |
| [ring_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | diagnostic | UNKNOWN | matches recorded run |
| [ring_extension](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | diagnostic | UNKNOWN | matches recorded run |
| [zero_tanh_rail_escape](rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | diagnostic | BLOCKED | matches recorded run |
| [grid100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | diagnostic | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | diagnostic | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | diagnostic | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 4**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 0 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022"></a>

## r1r2-repair-word-retention-20261022

**r1r2-repair-word-retention-20261022 — revision 1**. [View declaration](../../../configs/forge/views/r1r2-repair-word-retention-20261022.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](rhbi.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](rhbi.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](rhbi.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **7 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage"></a>

## vaegan_coverage

**vaegan_coverage — revision 1**. [View declaration](../../../configs/forge/views/vaegan_coverage.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Two toy gates do not establish screen calibration or whole-suite adoption.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_vae_hard](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | diagnostic | PASS | matches recorded run |
| [model_vae_categorical](rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-23a94f16ca88-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.0158929 | <= 0.35 | PASS |
| recon_mse | 0.00539541 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/335d81d733f24d74a3fb2145b1a9d814.json)

Recorded conditions: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise; weights state_selected; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan"></a>

### alice_spiral_bigan

**alice_spiral_bigan: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/alice_spiral_bigan.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.06); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| arc_bins | >= 15 |
| hq | >= 0.85 |
| mass_tv | <= 0.16 |
| recon_rmse | <= 0.14 |
| latent_recon_rmse | <= 0.25 |
| arc_consistency | >= 0.9 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.06); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes"></a>

### cgan_overlapping_classes

**cgan_overlapping_classes: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cgan_overlapping_classes.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| classes_covered | == 4 |
| class_mass_tv | <= 0.08 |
| mean_center_err | <= 0.12 |
| min_cov_eig_ratio | >= 0.6 |
| max_cov_eig_ratio | <= 1.5 |
| overlap_density_ratio | >= 0.55 |


Declared budget: 3000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/d27e03a7c01b4348a5f7b41ca4088949.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: PASS**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| content_kept | 0.894787 | >= 0.75 | PASS |
| leak_ratio | 0.0433731 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0183435 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0629478 | <= 0.2 | PASS |
| same_dir | 0.0501448 | <= 0.25 | PASS |
| u_kept | 0.959978 | >= 0.85 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/6d3896cb8cf44e10a62f3b03c5856fd0.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap"></a>

### cramer_energy_overlap

**cramer_energy_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cramer_energy_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cramer_energy_overlap: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 9 |
| hq | >= 0.88 |
| mass_tv | <= 0.1 |
| energy_distance | <= 0.06 |
| min_cov_eig_ratio | >= 0.35 |


Declared budget: 3000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation"></a>

### cyclegan_unpaired_constellation

**cyclegan_unpaired_constellation: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cyclegan_unpaired_constellation: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.08); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes_y | >= 7 |
| modes_x | >= 7 |
| hq_y | >= 0.85 |
| hq_x | >= 0.85 |
| mass_tv_y | <= 0.15 |
| cycle_rmse | <= 0.12 |
| bijection_ber | <= 0.15 |


Declared budget: 3500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.08); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.12 |
| t1_q0_normalized_mean | <= 0.25 |
| t1_q0_covariance_eigen_min | >= 0.65 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.12 |
| t1_q1_normalized_mean | <= 0.25 |
| t1_q1_covariance_eigen_min | >= 0.65 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.12 |
| t2_q0_normalized_mean | <= 0.25 |
| t2_q0_covariance_eigen_min | >= 0.65 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.12 |
| t2_q1_normalized_mean | <= 0.25 |
| t2_q1_covariance_eigen_min | >= 0.65 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.95 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 0.5 |
| final_local_covariance_min | >= 0.5 |
| final_local_covariance_max | <= 1.5 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 5000 updates; timeout 1200 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../ddgan-conditional-repair/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-ddgan_repair_gaussian_chain_v1"></a>

### ddgan_repair_gaussian_chain_v1

**ddgan_repair_gaussian_chain_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.12 |
| t1_q0_normalized_mean | <= 0.25 |
| t1_q0_covariance_eigen_min | >= 0.65 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.12 |
| t1_q1_normalized_mean | <= 0.25 |
| t1_q1_covariance_eigen_min | >= 0.65 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.12 |
| t2_q0_normalized_mean | <= 0.25 |
| t2_q0_covariance_eigen_min | >= 0.65 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.12 |
| t2_q1_normalized_mean | <= 0.25 |
| t2_q1_covariance_eigen_min | >= 0.65 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 2500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../ddgan-conditional-repair/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../five-word-joint/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. every joint generation/inverse check throughout the fixed continuation window

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 5667 |
| mass_apple | 0.188477 |
| mass_berry | 0.194336 |
| mass_grape | 0.223633 |
| mass_lemon | 0.182617 |
| mass_melon | 0.210938 |
| mass_tv | 0.0345703 |
| minimum_reconstruction_token_probability | 0.999731 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 4.53265e-05 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 5667 |

[Compact metrics and receipt provenance](../technique-receipts/f31945f80fa04a0ea8d800a0562a30e9.json)

Recorded conditions: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

Declared budget: 4000 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Dependencies: five_word_joint_smoke (checkpoint).

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full joint pass plus independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 20001 |
| mass_apple | 0.195312 |
| mass_berry | 0.210938 |
| mass_grape | 0.198242 |
| mass_lemon | 0.208008 |
| mass_melon | 0.1875 |
| mass_tv | 0.0189453 |
| minimum_reconstruction_token_probability | 0.995446 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 0.00102799 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 20001 |

[Compact metrics and receipt provenance](../technique-receipts/ab52ea1a2747477f8dbd024de52f7dfc.json)

Recorded conditions: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../../toy_audit/api_contract/gaussian1d/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0238842 |
| finite_fraction | 1 |
| mean | 2.01246 |
| mean_error_sigma | 0.0249183 |
| sample_count | 4096 |
| std | 0.533345 |
| std_ratio | 1.06669 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/dbfa42d3b2fc43cfbd243dca021d1f07.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. stationary hold, deadline reacquisition and shifted hold

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0291728 |
| finite_fraction | 1 |
| mean | 3.00273 |
| mean_error_sigma | 0.00545639 |
| sample_count | 4096 |
| std | 0.509239 |
| std_ratio | 1.01848 |
| step | 6000 |

[Compact metrics and receipt provenance](../technique-receipts/055554003c644a359fca45e05f567a5f.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 6000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: gaussian1d_smoke (checkpoint).

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-grid100"></a>

### grid100

**grid100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.636746 |
| accuracy_pass | False |
| accuracy_score | 6.16868 |
| all_finite | True |
| center_max_sigma | 1.99342 |
| center_rms_sigma | 1.12524 |
| cov_frob_rms | 0.769394 |
| cov_trace_bias | 0.636746 |
| frozen_pass | False |
| mass_tv | 0.13535 |
| n | 100000 |
| passed | False |
| precision | 0.26119 |
| precision_gap | 0.727701 |
| problem | grid100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.41701 |
| valid_n | 100000 |
| within_radius_n | 26119 |

[Compact metrics and receipt provenance](../technique-receipts/e68fcd3f374e4016ad0045ebb56227ce.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_mlp16"></a>

### grid100_mlp16

**grid100_mlp16: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_mlp16.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_mlp16: native model variants require an explicit profile

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: particle_cloud prior (sigma 0); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh"></a>

### grid100_res16_scaled_tanh

**grid100_res16_scaled_tanh: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_scaled_tanh.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_res16_scaled_tanh: native model variants require an explicit profile

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh"></a>

### grid100_res16_tanh

**grid100_res16_tanh: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_tanh.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_res16_tanh: native model variants require an explicit profile

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-img_bars4"></a>

### img_bars4

**img_bars4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.96875 | >= 0.9 | PASS |
| modes | 2 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/2cb1029649b946c08772af252463e0f5.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_bars4_w64_optimizer_diagnostic"></a>

### img_bars4_w64_optimizer_diagnostic

**img_bars4_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.96875 | >= 0.9 | PASS |
| modes | 4 | >= 4 | PASS |

Recorded terminal passing observations: **5**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/61827fd4efb147f7b02d9bd10e060dcd.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.90625 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **6**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/315941faa28e4006a23c8c5c411b4af2.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/34616a3a80ef488c8c46445b10f38cd1.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_stripes2_w64_optimizer_diagnostic"></a>

### img_stripes2_w64_optimizer_diagnostic

**img_stripes2_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes"></a>

### infogan_anisotropic_modes

**infogan_anisotropic_modes: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/infogan_anisotropic_modes.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.05); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.85 |
| mass_tv | <= 0.12 |
| discrete_nmi | >= 0.85 |
| q_acc | >= 0.9 |
| continuous_r2 | >= 0.65 |

At least 5 consecutive passing terminal observations.

Declared budget: 3500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift"></a>

### interp_img_patch_shift

**interp_img_patch_shift: PASS**. [Current experiment declaration](../../../configs/forge/tasks/interp_img_patch_shift.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed interp_img_patch_shift gates (5/5 terminal checks passed)

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| all_pairs_interp_rmse | 0.0881085 |
| edge_count | 21 |
| endpoint_hq | 1 |
| endpoint_rmse | 0.0355527 |
| finite_fraction | 1 |
| interp_hq | 1 |
| interp_peak | 0.857395 |
| interp_rmse | 0.0360159 |
| midpoint_novelty | 0.384558 |
| output_knn_interp_peak | 0.863847 |
| output_knn_interp_rmse | 0.0373225 |
| quadrant_coverage | 4 |
| step | 800 |

[Compact metrics and receipt provenance](../technique-receipts/8a49fefbc7a9407d9a39e39a78462a69.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| finite_fraction | == 1 |
| endpoint_rmse | <= 0.09 |
| interp_rmse | <= 0.1 |
| interp_peak | >= 0.78 |
| quadrant_coverage | == 4 |
| midpoint_novelty | >= 0.18 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../particle-interpolation-stretch/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric"></a>

### interp_saddle_barycentric

**interp_saddle_barycentric: PASS**. [Current experiment declaration](../../../configs/forge/tasks/interp_saddle_barycentric.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed interp_saddle_barycentric gates (5/5 terminal checks passed)

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| barycenter_hq | 0.835821 |
| barycenter_novelty | 0.543169 |
| barycenter_surface_mae | 0.0667568 |
| edge_midpoint_surface_mae | 0.0574646 |
| endpoint_hq | 0.916667 |
| endpoint_surface_mae | 0.0398555 |
| finite_fraction | 1 |
| output_knn_barycenter_mae | 0.0635101 |
| step | 800 |
| triangle_count | 67 |
| xy_bins_covered | 16 |

[Compact metrics and receipt provenance](../technique-receipts/7c01c632cec44b94acadbdfaf107ab28.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| finite_fraction | == 1 |
| endpoint_surface_mae | <= 0.08 |
| barycenter_surface_mae | <= 0.09 |
| barycenter_hq | >= 0.78 |
| xy_bins_covered | >= 12 |
| barycenter_novelty | >= 0.18 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../particle-interpolation-stretch/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc"></a>

### interp_sparse_arc

**interp_sparse_arc: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/interp_sparse_arc.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. failed interp_sparse_arc thresholds: endpoint_hq >= 0.85, interp_hq >= 0.8, interp_arc_bins >= 22

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [particle_interpolation_stretch / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-particle_interpolation_stretch-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| all_pairs_interp_hq | 0.19697 |
| edge_count | 22 |
| endpoint_arc_bins | 9 |
| endpoint_hq | 0.75 |
| finite_fraction | 1 |
| interp_arc_bins | 25 |
| interp_hq | 0.606061 |
| interp_mean_dist | 0.108211 |
| midpoint_novelty | 0.43106 |
| output_knn_interp_hq | 0.619048 |
| step | 600 |

[Compact metrics and receipt provenance](../technique-receipts/e5009ad4c12243899432453e21aa1e01.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| finite_fraction | == 1 |
| endpoint_hq | >= 0.85 |
| interp_hq | >= 0.8 |
| interp_arc_bins | >= 22 |
| midpoint_novelty | >= 0.2 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True; rng isolation = True.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../particle-interpolation-stretch/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_cos_minus | 1 | >= 0.85 | PASS |
| concept_cos_plus | 0.999902 | >= 0.85 | PASS |
| concept_mag_minus | 0.999984 | <= 1.25 | PASS |
| concept_mag_plus | 1.00047 | <= 1.25 | PASS |
| identity_at_0 | 0.999697 | >= 0.85 | PASS |
| identity_at_mid | 0.999051 | >= 0.85 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f7f9299dfaad440c9ae60fe17f569e81.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer"></a>

### misgan_block_neural_imputer

**misgan_block_neural_imputer: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_neural_imputer.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_block_neural_imputer: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| observed_max_error | <= 1e-06 |
| acc | >= 0.95 |
| acc_lo | >= 0.65 |
| missing_std_ratio | >= 0.7 |
| missing_std_ratio | <= 1.3 |


Declared budget: 7000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery"></a>

### misgan_block_recovery

**misgan_block_recovery: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_recovery.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_block_recovery: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 100 |
| hq | >= 0.9 |
| mass_tv | <= 0.1 |
| off_plane | <= 0.04 |
| mask_tv | <= 0.05 |
| acc | >= 0.96 |
| acc_lo | >= 0.7 |


Declared budget: 7000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery"></a>

### misgan_mcar_p50_recovery

**misgan_mcar_p50_recovery: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 90 |
| hq | >= 0.85 |
| mass_tv | <= 0.12 |
| off_plane | <= 0.05 |
| mask_tv | <= 0.05 |
| acc | >= 0.9 |
| impute_rmse | <= 0.12 |


Declared budget: 4000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored"></a>

### misgan_mcar_p80_anchored

**misgan_mcar_p80_anchored: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_mcar_p80_anchored: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 80 |
| hq | >= 0.8 |
| mass_tv | <= 0.15 |
| off_plane | <= 0.06 |
| imp_acc | >= 0.65 |
| imp_off_plane | <= 0.06 |
| impute_rmse | <= 0.18 |


Declared budget: 4000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-mode_hold"></a>

### mode_hold

**mode_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 8 | >= 8 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8b5c510e94864525a6ab8d6f76169143.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain"></a>

### model_ae_ddgan_bimodal_chain

**model_ae_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.32 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.3 |
| final_covariance_eigen_max | <= 2.6 |
| t1_q0_projection_cdf | <= 0.75 |
| t1_q0_normalized_mean | <= 1.5 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 4.5 |
| t1_q1_projection_cdf | <= 0.75 |
| t1_q1_normalized_mean | <= 1.5 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 4.5 |
| t2_q0_projection_cdf | <= 0.28 |
| t2_q0_normalized_mean | <= 0.7 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.28 |
| t2_q1_normalized_mean | <= 0.7 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.22 |
| t3_q0_normalized_mean | <= 0.5 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.22 |
| t3_q1_normalized_mean | <= 0.5 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.9 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 1.5 |
| final_local_covariance_min | >= 0.3 |
| final_local_covariance_max | <= 2.05 |
| paired_nmse | <= 0.1 |
| paired_worst_axis_nrmse | <= 0.25 |
| paired_shuffled_gain | >= 3 |
| paired_reverse_steps | == 4 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 5000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Dependencies: model_ae_ddgan_gaussian_chain (gate).

[Explanation and existing training artifacts](../api-model-toys/ae-ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain"></a>

### model_ae_ddgan_gaussian_chain

**model_ae_ddgan_gaussian_chain: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| final_covariance_eigen_max | 1.02475 | <= 1.4 | PASS |
| final_covariance_eigen_min | 0.828811 | >= 0.65 | PASS |
| final_normalized_mean | 0.225336 | <= 0.45 | PASS |
| final_projection_cdf | 0.112908 | <= 0.2 | PASS |
| paired_nmse | 0.0601546 | <= 0.1 | PASS |
| paired_reverse_steps | 4 | == 4 | PASS |
| paired_shuffled_gain | 32.3524 | >= 3 | PASS |
| paired_worst_axis_nrmse | 0.262891 | <= 0.25 | FAIL |
| panel_count | 8 | == 8 | PASS |
| samples_per_panel | 2048 | >= 2048 | PASS |
| t1_q0_covariance_eigen_max | 1.30208 | <= 1.8 | PASS |
| t1_q0_covariance_eigen_min | 0.877489 | >= 0.4 | PASS |
| t1_q0_normalized_mean | 0.323484 | <= 0.65 | PASS |
| t1_q0_projection_cdf | 0.160231 | <= 0.26 | PASS |
| t1_q1_covariance_eigen_max | 1.05142 | <= 1.8 | PASS |
| t1_q1_covariance_eigen_min | 0.717037 | >= 0.4 | PASS |
| t1_q1_normalized_mean | 0.297452 | <= 0.65 | PASS |
| t1_q1_projection_cdf | 0.202703 | <= 0.26 | PASS |
| t2_q0_covariance_eigen_max | 0.93057 | <= 1.4 | PASS |
| t2_q0_covariance_eigen_min | 0.790889 | >= 0.6 | PASS |
| t2_q0_normalized_mean | 0.131381 | <= 0.35 | PASS |
| t2_q0_projection_cdf | 0.0660825 | <= 0.2 | PASS |
| t2_q1_covariance_eigen_max | 0.841648 | <= 1.4 | PASS |
| t2_q1_covariance_eigen_min | 0.752506 | >= 0.6 | PASS |
| t2_q1_normalized_mean | 0.0836394 | <= 0.35 | PASS |
| t2_q1_projection_cdf | 0.0686985 | <= 0.2 | PASS |
| t3_q0_covariance_eigen_max | 0.968248 | <= 1.4 | PASS |
| t3_q0_covariance_eigen_min | 0.915376 | >= 0.65 | PASS |
| t3_q0_normalized_mean | 0.044957 | <= 0.25 | PASS |
| t3_q0_projection_cdf | 0.0275369 | <= 0.12 | PASS |
| t3_q1_covariance_eigen_max | 0.986707 | <= 1.4 | PASS |
| t3_q1_covariance_eigen_min | 0.916152 | >= 0.65 | PASS |
| t3_q1_normalized_mean | 0.0266264 | <= 0.25 | PASS |
| t3_q1_projection_cdf | 0.0264778 | <= 0.12 | PASS |
| t4_q0_covariance_eigen_max | 1.03465 | <= 1.4 | PASS |
| t4_q0_covariance_eigen_min | 0.992181 | >= 0.65 | PASS |
| t4_q0_normalized_mean | 0.0258689 | <= 0.25 | PASS |
| t4_q0_projection_cdf | 0.0209575 | <= 0.12 | PASS |
| t4_q1_covariance_eigen_max | 1.01605 | <= 1.4 | PASS |
| t4_q1_covariance_eigen_min | 0.953314 | >= 0.65 | PASS |
| t4_q1_normalized_mean | 0.0246816 | <= 0.25 | PASS |
| t4_q1_projection_cdf | 0.0257535 | <= 0.12 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0049ed3d161d44b088cf84a575caad8c.json)

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.2 |
| final_normalized_mean | <= 0.45 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.26 |
| t1_q0_normalized_mean | <= 0.65 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 1.8 |
| t1_q1_projection_cdf | <= 0.26 |
| t1_q1_normalized_mean | <= 0.65 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 1.8 |
| t2_q0_projection_cdf | <= 0.2 |
| t2_q0_normalized_mean | <= 0.35 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.2 |
| t2_q1_normalized_mean | <= 0.35 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| paired_nmse | <= 0.1 |
| paired_worst_axis_nrmse | <= 0.25 |
| paired_shuffled_gain | >= 3 |
| paired_reverse_steps | == 4 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 2500 updates; timeout 180 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ae-ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal"></a>

### model_ae_gan_multimodal

**model_ae_gan_multimodal: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_gan_multimodal.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_center_rms_sigma | 0.83869 | <= 1 | PASS |
| component_max_eigen_ratio | 1.58937 | <= 4 | PASS |
| component_min_eigen_ratio | 0.520649 | >= 0.1 | PASS |
| mass_tv | 0.0292969 | <= 0.15 | PASS |
| modes | 4 | >= 4 | PASS |
| precision | 0.939453 | >= 0.9 | PASS |
| reconstruction_mode_agreement | 1 | >= 0.98 | PASS |
| reconstruction_mse | 0.00765663 | <= 0.025 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8aa56e9038914714b3c926d066c20412.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.025 |
| reconstruction_mode_agreement | >= 0.98 |
| modes | >= 4 |
| precision | >= 0.9 |
| mass_tv | <= 0.15 |
| component_min_eigen_ratio | >= 0.1 |
| component_max_eigen_ratio | <= 4 |
| component_center_rms_sigma | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction"></a>

### model_ae_reconstruction

**model_ae_reconstruction: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_reconstruction.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| reconstruction_mode_agreement | 1 | >= 0.98 | PASS |
| reconstruction_mse | 0.00650055 | <= 0.025 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/59a42339dd684a93bf53e17657f125d5.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.025 |
| reconstruction_mode_agreement | >= 0.98 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior; exact optimizer updates = True; mechanism exercised = False.

Declared budget: 1200 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain"></a>

### model_ddgan_bimodal_chain

**model_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.32 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.3 |
| final_covariance_eigen_max | <= 2.6 |
| t1_q0_projection_cdf | <= 0.75 |
| t1_q0_normalized_mean | <= 1.5 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 4.5 |
| t1_q1_projection_cdf | <= 0.75 |
| t1_q1_normalized_mean | <= 1.5 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 4.5 |
| t2_q0_projection_cdf | <= 0.28 |
| t2_q0_normalized_mean | <= 0.7 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.28 |
| t2_q1_normalized_mean | <= 0.7 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.22 |
| t3_q0_normalized_mean | <= 0.5 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.22 |
| t3_q1_normalized_mean | <= 0.5 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.9 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 1.5 |
| final_local_covariance_min | >= 0.3 |
| final_local_covariance_max | <= 2.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 5000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Dependencies: model_ddgan_gaussian_chain (gate).

[Explanation and existing training artifacts](../api-model-toys/ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain"></a>

### model_ddgan_gaussian_chain

**model_ddgan_gaussian_chain: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| final_covariance_eigen_max | 1.05237 | <= 1.4 | PASS |
| final_covariance_eigen_min | 0.609396 | >= 0.65 | FAIL |
| final_normalized_mean | 0.185373 | <= 0.25 | PASS |
| final_projection_cdf | 0.112339 | <= 0.12 | PASS |
| panel_count | 8 | == 8 | PASS |
| samples_per_panel | 2048 | >= 2048 | PASS |
| t1_q0_covariance_eigen_max | 0.908127 | <= 1.4 | PASS |
| t1_q0_covariance_eigen_min | 0.513208 | >= 0.4 | PASS |
| t1_q0_normalized_mean | 0.507659 | <= 0.45 | FAIL |
| t1_q0_projection_cdf | 0.243458 | <= 0.26 | PASS |
| t1_q1_covariance_eigen_max | 1.42333 | <= 1.4 | FAIL |
| t1_q1_covariance_eigen_min | 0.437067 | >= 0.4 | PASS |
| t1_q1_normalized_mean | 0.162902 | <= 0.45 | PASS |
| t1_q1_projection_cdf | 0.154585 | <= 0.26 | PASS |
| t2_q0_covariance_eigen_max | 0.9563 | <= 1.4 | PASS |
| t2_q0_covariance_eigen_min | 0.740684 | >= 0.6 | PASS |
| t2_q0_normalized_mean | 0.170599 | <= 0.35 | PASS |
| t2_q0_projection_cdf | 0.0915381 | <= 0.2 | PASS |
| t2_q1_covariance_eigen_max | 1.11043 | <= 1.4 | PASS |
| t2_q1_covariance_eigen_min | 0.78164 | >= 0.6 | PASS |
| t2_q1_normalized_mean | 0.0327818 | <= 0.35 | PASS |
| t2_q1_projection_cdf | 0.036341 | <= 0.2 | PASS |
| t3_q0_covariance_eigen_max | 1.04051 | <= 1.4 | PASS |
| t3_q0_covariance_eigen_min | 0.920883 | >= 0.65 | PASS |
| t3_q0_normalized_mean | 0.0686743 | <= 0.25 | PASS |
| t3_q0_projection_cdf | 0.0496746 | <= 0.12 | PASS |
| t3_q1_covariance_eigen_max | 1.0492 | <= 1.4 | PASS |
| t3_q1_covariance_eigen_min | 1.01696 | >= 0.65 | PASS |
| t3_q1_normalized_mean | 0.0209972 | <= 0.25 | PASS |
| t3_q1_projection_cdf | 0.0274171 | <= 0.12 | PASS |
| t4_q0_covariance_eigen_max | 1.0326 | <= 1.4 | PASS |
| t4_q0_covariance_eigen_min | 1.00293 | >= 0.65 | PASS |
| t4_q0_normalized_mean | 0.00882689 | <= 0.25 | PASS |
| t4_q0_projection_cdf | 0.0161727 | <= 0.12 | PASS |
| t4_q1_covariance_eigen_max | 1.02543 | <= 1.4 | PASS |
| t4_q1_covariance_eigen_min | 0.966362 | >= 0.65 | PASS |
| t4_q1_normalized_mean | 0.011189 | <= 0.25 | PASS |
| t4_q1_projection_cdf | 0.0177262 | <= 0.12 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b360b9257806443a87a252937975abdf.json)

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.26 |
| t1_q0_normalized_mean | <= 0.45 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.26 |
| t1_q1_normalized_mean | <= 0.45 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.2 |
| t2_q0_normalized_mean | <= 0.35 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.2 |
| t2_q1_normalized_mean | <= 0.35 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 2500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_misgan_four_mode"></a>

### model_misgan_four_mode

**model_misgan_four_mode: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_misgan_four_mode.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [misgan-bispf-research-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |
| mass_tv | <= 0.1 |
| off_plane | <= 0.04 |
| mask_tv | <= 0.05 |
| imputer_observed_error | <= 1e-06 |
| imputer_accuracy | >= 0.94 |
| ambiguous_hq | >= 0.9 |
| ambiguous_mass_tv | <= 0.12 |
| ambiguous_query_coverage | >= 0.9 |
| ambiguous_std_ratio | >= 0.7 |
| ambiguous_std_ratio | <= 1.3 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../misgan-bispf/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_vae_categorical"></a>

### model_vae_categorical

**model_vae_categorical: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_categorical.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-vaegan_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| decoder_mean_hq | 0.937256 | >= 0.85 | PASS |
| decoder_mean_mass_tv | 0.0627441 | <= 0.15 | PASS |
| decoder_mean_modes | 2 | >= 2 | PASS |
| elbo_identity_error | 8.88178e-16 | <= 0.0001 | PASS |
| kl_identity_error | 5.31649e-07 | <= 1e-05 | PASS |
| likelihood_hq | 0.911865 | >= 0.85 | PASS |
| likelihood_mass_tv | 0.0881348 | <= 0.15 | PASS |
| likelihood_modes | 2 | >= 2 | PASS |
| posterior_mutual_information | 0.696042 | >= 0.3 | PASS |
| posterior_noise_variance_ratio | 1.0033 | <= 1.2 | PASS |
| reconstruction_mse | 0.0079438 | <= 0.05 | PASS |
| shuffle_reconstruction_ratio | 293.956 | >= 5 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/faea56b7a044460d9f1e9c3384e094d8.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.05 |
| shuffle_reconstruction_ratio | >= 5 |
| posterior_mutual_information | >= 0.3 |
| posterior_noise_variance_ratio | >= 0.8 |
| posterior_noise_variance_ratio | <= 1.2 |
| kl_identity_error | <= 1e-05 |
| elbo_identity_error | <= 0.0001 |
| decoder_mean_hq | >= 0.85 |
| decoder_mean_mass_tv | <= 0.15 |
| decoder_mean_modes | >= 2 |
| likelihood_hq | >= 0.85 |
| likelihood_mass_tv | <= 0.15 |
| likelihood_modes | >= 2 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = encoder, generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/vae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_vae_hard"></a>

### model_vae_hard

**model_vae_hard: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_hard.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-vaegan_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| decoder_mean_hq | 0.999756 | >= 0.85 | PASS |
| decoder_mean_mass_tv | 0.00366211 | <= 0.15 | PASS |
| decoder_mean_modes | 2 | >= 2 | PASS |
| elbo_identity_error | 4.44089e-16 | <= 0.0001 | PASS |
| kl_identity_error | 7.61862e-09 | <= 1e-05 | PASS |
| likelihood_hq | 0.999512 | >= 0.85 | PASS |
| likelihood_mass_tv | 0.00390625 | <= 0.15 | PASS |
| likelihood_modes | 2 | >= 2 | PASS |
| posterior_mutual_information | 1.24379 | >= 0.3 | PASS |
| posterior_noise_variance_ratio | 1.0009 | <= 1.2 | PASS |
| reconstruction_mse | 0.00398122 | <= 0.05 | PASS |
| shuffle_reconstruction_ratio | 544.678 | >= 5 | PASS |

Recorded terminal passing observations: **16**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/340039825a2c4980a8e404e74f86a1e7.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.05 |
| shuffle_reconstruction_ratio | >= 5 |
| posterior_mutual_information | >= 0.3 |
| posterior_noise_variance_ratio | >= 0.8 |
| posterior_noise_variance_ratio | <= 1.2 |
| kl_identity_error | <= 1e-05 |
| elbo_identity_error | <= 0.0001 |
| decoder_mean_hq | >= 0.85 |
| decoder_mean_mass_tv | <= 0.15 |
| decoder_mean_modes | >= 2 |
| likelihood_hq | >= 0.85 |
| likelihood_mass_tv | <= 0.15 |
| likelihood_modes | >= 2 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = encoder, generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/vae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100"></a>

### pacgan8_grid100

**pacgan8_grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/pacgan8_grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.05); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 95 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| min_cov_eig_ratio | >= 0.15 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass"></a>

### pacgan_unequal_mass

**pacgan_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/pacgan_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. pacgan_unequal_mass: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.05); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 5 |
| min_mode_mass | >= 0.01 |
| mass_tv | <= 0.1 |
| hq | >= 0.88 |
| min_cov_eig_ratio | >= 0.2 |


Declared budget: 3000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-precision_gaussian_fp32"></a>

### precision_gaussian_fp32

**precision_gaussian_fp32: PASS**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_fp32.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0279722 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0679366 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.1041 | <= 1.2 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1eff6a7296044bc2ae16331c0aefd3a5.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8"></a>

### precision_gaussian_mxfp8

**precision_gaussian_mxfp8: PASS**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_mxfp8.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0214008 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0105648 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.06879 | <= 1.2 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/57b2437bf63343bb99928405d3ec6531.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4"></a>

### precision_gaussian_nvfp4

**precision_gaussian_nvfp4: PASS**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0328951 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0321007 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.00333 | <= 1.2 | PASS |

Recorded terminal passing observations: **19**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f9516da9cea54f5e8096134c78cbac7b.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-residual_student"></a>

### residual_student

**residual_student: PASS**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.00026976 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/765f38d9560648959ef18d75b8d53b05.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.500346 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.337188 | >= 0.15 | PASS |
| hq | 0.972168 | >= 0.85 | PASS |
| mass_tv | 0.0649414 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **25**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/851c01a4e4224526b6548d3296cd578c.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous"></a>

### ring16_bigan_continuous

**ring16_bigan_continuous: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_bigan_continuous.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring16_bigan_continuous: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 16 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| recon_rmse | <= 0.12 |
| mode_consistency | >= 0.98 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: ring_hold (checkpoint).

<a name="cohort-cuda-23a94f16ca88-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-rotated100"></a>

### rotated100

**rotated100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | unavailable |
| accuracy_pass | False |
| accuracy_score | unavailable |
| all_finite | True |
| center_max_sigma | unavailable |
| center_rms_sigma | unavailable |
| cov_frob_rms | unavailable |
| cov_trace_bias | unavailable |
| frozen_pass | False |
| mass_tv | 0.11251 |
| n | 100000 |
| passed | False |
| precision | 0.20421 |
| precision_gap | 0.784681 |
| problem | rotated100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | unavailable |
| valid_n | 100000 |
| within_radius_n | 20421 |

[Compact metrics and receipt provenance](../technique-receipts/d01b3f427b92448e8325d1b434c8bef3.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: rotated100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-staggered100"></a>

### staggered100

**staggered100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | unavailable |
| accuracy_pass | False |
| accuracy_score | unavailable |
| all_finite | True |
| center_max_sigma | unavailable |
| center_rms_sigma | unavailable |
| cov_frob_rms | unavailable |
| cov_trace_bias | unavailable |
| frozen_pass | False |
| mass_tv | 0.11332 |
| n | 100000 |
| passed | False |
| precision | 0.3086 |
| precision_gap | 0.680291 |
| problem | staggered100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | unavailable |
| valid_n | 100000 |
| within_radius_n | 30860 |

[Compact metrics and receipt provenance](../technique-receipts/fe83cb5c6ff84c0c88b351e8941bff0f.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: staggered100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [adaptation / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-3) · [clockfree_continuous / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-trajectory"></a>

### trajectory

**trajectory: PASS**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000266086 | <= 0.02 | PASS |

Recorded terminal passing observations: **19**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/2cfbfc53633744318d00563479cb7a7d.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.151377 | <= 1 | PASS |
| mean_abs | 0.995421 | >= 0.3 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/5f2cbe6323084935ba2adc9011051129.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_particles_and_critic_gradient; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-unipolar"></a>

### unipolar

**unipolar: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cover | 0.992739 | >= 0.85 | PASS |
| neu_hold | 0.993036 | >= 0.85 | PASS |
| off_caption | 0.00023946 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7bc8448bed0640d38cb5e26d2a9b8d03.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.990558 | >= 0.85 | PASS |
| unused_hold | 0.988799 | >= 0.85 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c6bae942c78041f989a58110a52470a0.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_parameter_measurement; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.498667 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.425563 | >= 0.15 | PASS |
| hq | 0.946777 | >= 0.85 | PASS |
| mass_tv | 0.144857 | <= 0.15 | PASS |
| sw1_normalized | 0.148212 | <= 0.18 | PASS |

Recorded terminal passing observations: **7**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/334ef49e584942a4be71f5df6eee99d8.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.05894 | <= 0.45 | PASS |
| mean_error | 0.0997509 | <= 0.15 | PASS |
| sw1_normalized | 0.0716695 | <= 0.18 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/4bbcdd891b8244e7964317288fc180e8.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.116969 | <= 0.45 | PASS |
| mean_error | 0.0361613 | <= 0.15 | PASS |
| sw1_normalized | 0.0457747 | <= 0.18 | PASS |

Recorded terminal passing observations: **24**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/060b646d6e3b446bb7048c46857ac71d.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.391532 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.621427 | >= 0.15 | PASS |
| hq | 0.97876 | >= 0.85 | PASS |
| mass_tv | 0.0356445 | <= 0.15 | PASS |
| sw1_normalized | 0.0588229 | <= 0.18 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/88c873e57b4a40858c1d389ad8f4be57.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 2.16749 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.209538 | >= 0.15 | PASS |
| hq | 0.973389 | >= 0.85 | PASS |
| mass_tv | 0.0924317 | <= 0.15 | PASS |
| min_mass_ratio | 0.738118 | >= 0.25 | PASS |
| sw1_normalized | 0.149604 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/55da2b3ca6dc4595a982f2a181077167.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 2.5266 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.632873 | >= 0.15 | PASS |
| hq | 0.97583 | >= 0.85 | PASS |
| mass_tv | 0.221924 | <= 0.15 | FAIL |
| sw1_normalized | 0.212217 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e9f3a4c7901c4038aa8082c5cf062305.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](rhbi.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape"></a>

### zero_tanh_rail_escape

**zero_tanh_rail_escape: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_rail_escape.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. zero_tanh_rail_escape: zero_tanh_rail_escape: unsupported tanh host zero_tanh_rail_escape; only zero_tanh_smoke is routed

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 3](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |

Recovery deadline in updates: 400.

Declared budget: 2000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

Dependencies: zero_tanh_smoke (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke"></a>

### zero_tanh_smoke

**zero_tanh_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed the declared final zero-tanh bounds

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](rhbi.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mean_abs_error | 0.00389656 | <= 0.005 | PASS |
| p99_abs_error | 0.0143897 | <= 0.025 | PASS |
| rms_error | 0.00498791 | <= 0.01 | PASS |
| saturated_fraction | 0 | == 0 | PASS |

[Compact metrics and receipt provenance](../technique-receipts/4f480daa7a0d47d68bd076fc9f2fa7aa.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |


Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

## Historical and diagnostic evidence

Separate configurations, API variants and serving laws retain their own scopes and supply no cells above.

- [Other configurations and original evidence bindings](../technique-inventory.json)
- [Compiled experiment memory](../EXPERIMENT_MEMORY.md)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This page is generated alongside the leaderboard. Register new source evidence before refreshing; editing a page cannot change a verdict or earn qualification.

## References

- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895). Entropic Optimal Transport Sinkhorn-Knopp scaling extended to the (B+1)x(B+1) slack-screened unbalanced polytope.
- [Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations](https://arxiv.org/abs/1508.05216). Kantorovich-Hellinger unbalanced mass creation/destruction theory motivating slack-reservoir screening of Poisson batch singletons and empty-valley outliers.
- [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://arxiv.org/abs/1810.08278). Feydy diagonal-included symmetric Sinkhorn self-coupling debiasing adapted to the active-plus-slack unbalanced polytope.
- [On Optimal Matchings](https://doi.org/10.1007/BF02579150). Ajtai-Komlos-Tusnady empirical optimal transport scaling $B^{-\min(1/d,1/2)}$ motivating Poisson-Weibull 1D nearest-neighbor horizon normalization.
