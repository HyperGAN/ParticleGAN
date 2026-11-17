<!-- Generated Forge family report -->

# SMPF

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [selectable-loss](../technique-inventory.md#tag-selectable-loss)

## Technique overview

Scale- & Distortion-Aware Moving Prototype Flow (SMPF) is a clock-free, constant-learning-rate ParticleGAN formulation inspired by s-means (Kocabay, Yuksel & Yalcin, 2026; https://suayptalha.github.io/s-means-blog/) and building on ParticleGAN's e-means-inspired EMPF, with no EMA critic network copy. SMPF replaces EMPF's 1D rank initialization and count-only prototype splitting with a checkpointed s-means sufficient-statistics bank (proto_centroids, proto_usage, proto_sse, feat_scale, fast_sum, fast_count) featuring 4-round deterministic D^1.5 coreset seeding, tiled addmm nearest-prototype assignment, distortion x usage^0.35 anisotropic RMS dead-prototype revival across all dimensions d >= 1, and Empirical-Bayes count-shrunk centroids (tau=0.20), and couples those denoised prototypes directly to generator training via thermally matched physical Born-Infeld Sinkhorn divergence, per-prototype RMS distortion shells, affine-detrended whitened conditional fiber transport, and connected-manifold simplicial chord alignment.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior with $M$ locations in $\mathbb R^{d_z}$, $x$ a real batch of size $B_r$, $z\sim P$ a latent batch and $y=G(z)$ the generated batch of size $B_f$. $(C, u, \text{sse}, v, S^{\text{fast}}, N^{\text{fast}})$ is the online s-means bank of size $K=\min(K_{\max}, B_r)$ with Empirical-Bayes centroids $C_k^{\mathrm{eb}}$, normalized masses $w_k=u_k/\sum_j u_j$, and local RMS quantization radii $\ell_k=\sqrt{\text{sse}_k/u_k}$. $\Pi_{pf}, \Pi_{ff}$ are detached Sinkhorn couplings for physical Born-Infeld costs $c_{\ell_k}(C_k^{\mathrm{eb}}, y_j)=\sqrt{\ell_k^2+\lVert y_j-C_k^{\mathrm{eb}}\rVert_2^2}-\ell_k$.

**Clock-Free Online s-means Distortion-Anisotropic Revival and Empirical-Bayes Centroid Update**

$$
\Delta_s = 0.55\sqrt{\frac{\mathrm{sse}_s}{u_s d}}\,\mathbf r_s\odot\hat{\sigma}_{\mathrm{feat}},\quad C_k^{\mathrm{eb}}=\frac{N_k^{\mathrm{fast}}}{N_k^{\mathrm{fast}}+\tau_{\mathrm{eb}}\bar N^{\mathrm{fast}}}\bar{\mu}_k^{\mathrm{fast}}+\frac{\tau_{\mathrm{eb}}\bar N^{\mathrm{fast}}}{N_k^{\mathrm{fast}}+\tau_{\mathrm{eb}}\bar N^{\mathrm{fast}}}C_k
$$

Splits starved prototypes ($u_k<0.10\bar u$) from active donors scored by $\mathrm{sse}_s\cdot u_s^{0.35}$ along online inter-cluster feature spread $\hat{\sigma}_{\mathrm{feat}}$ at exact local RMS distortion scale without arbitrary $\epsilon$, and shrinks rare-mode fast estimates toward smoothed EMA centroids ($\tau_{\mathrm{eb}}=0.20$).

**Debiased Physical Born-Infeld s-means Prototype Transport and RMS Distortion Shell**

$$
\mathcal L_{\mathrm{SMPF}}(y)=\langle\Pi_{pf}^{\mathrm{det}},c_{\ell}(C^{\mathrm{eb}},y)\rangle-\tfrac{1}{2}\langle\Pi_{ff}^{\mathrm{det}},c_{\bar\ell}(y,y)\rangle+\lambda_{\mathrm{shell}}\sum_k w_k\operatorname{relu}(0.85\ell_k-\hat r_k(y)),\quad c_{\ell_k}(c,y)=\sqrt{\ell_k^2+\lVert y-c\rVert_2^2}-\ell_k
$$

Matches generated samples to Empirical-Bayes prototype masses $w_k$ with prototype-specific Born-Infeld curvature $\ell_k=\sqrt{\mathrm{sse}_k/u_k}$, unit asymptotic gradient slope, entropic self-coupling debiasing, and rare-cluster within-cell dispersion preservation.

## Simplified pseudocode

```text
Choose an adversarial loss, DualNorm/Adam optimizer and constant learning rates.
For each task-owned training update:
  Sample real x and latent z from the task prior P; compute y=G(z).
  On unconditioned continuous vector and compact image domains, seed (4-round D^1.5) or assign x to C via tiled addmm, update (C, u, sse, feat_scale, S_fast, N_fast) with state-settling power-cosine alpha_eff, revive starved prototypes (u_k < 0.10 mean(u)) by splitting top (sse * u^0.35) donors by anisotropic RMS radius 0.55 * sqrt(sse / (u * d)) * r * sigma_feat, and read out Empirical-Bayes centroids C_eb (tau=0.20) and local RMS radii ell_k.
  Compute prototype-chord Elastic Born-Infeld critic gradient penalty + zero-mean logit gauge and update instance-bound thermal scale s_thermal; step D.
  With D fixed, recompute y=G(z) and evaluate SMPFGANLoss: centered adversarial log-odds + thermally coupled debiased physical Born-Infeld s-means prototype Sinkhorn divergence + RMS distortion shell (or affine-detrended whitened IMQ conditional fiber transport on conditioned fibers), plus H_0-connected simplicial chord alignment on zero-sigma ParticlePrior tables and SMPFPriorRegularizer; step G, P and E.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | SMPFGANLoss pairing shift-invariant centered adversarial log-odds with thermally coupled, Kantorovich-Hellinger slack-screened physical Born-Infeld s-means prototype Sinkhorn divergence + Laguerre Monge outlier pull + per-prototype RMS distortion shell on unconditioned continuous/compact-image domains, Nadaraya-Watson detrended coordinate-whitened multi-scale IMQ conditional fiber transport on continuous-conditioned fibers, and standard calibrated GANLoss on discrete simplexes. |
| Optimizer | Supports DualNorm (selected configuration: zero momentum, smoothing 0.001, per-offset quintic convolution, polar_cpu_v1 dense polar profile, batched SVD backend, BF16 channels-last conv training, direction-blend constraint geometry) as well as native Adam and formulation-owned SMPFCriticAdam without an EMA critic network copy. |
| Learning rates and annealing | Strictly constant (`lr_floor=1.0`, `network_lr_floor=1.0`): global LR $0.012$, critic multiplier $1.5$ and DualNorm sampled-row prior multiplier $2.5$ (`prior_reg=0.01`). |
| Parameter-gradient clipping | None. Critic spike guard is disabled; critic and generator transport gradients are bounded smoothly by unit-slope Born-Infeld potentials. |
| Critic penalties and anchors | s-means EB-prototype-chord Elastic Born-Infeld regularizer evaluated on real endpoints, fake endpoints and mass-weighted prototype Sinkhorn chords ($\kappa=1$, base sharpness $\beta_0=6$ tightening quadratically as transport residual settles, harmonic floor $\lambda_{\mathrm{floor}}=0.01$, chord blend $0.08$) plus a settling zero-mean logit gauge ($\gamma_{\mathrm{gauge}}=0.05$). |
| Damping and update guards | Instance-bound shock-filtered transport EMA, state-settling s-means power-cosine prototype turnover, trimmed 1D quantile-$W_2$ floor-subtracted $1/\sqrt{B}$ cooling with 1D DualNorm step-scale damping, prototype-referenced 2D harmonic capture lock, and discrete-simplex post-acquisition Chamfer cooling ($[0.064, 0.084]\to 0.0004$) impedance-matched across critic, generator transport, simplicial chords, and prior regularization. |
| Training and sampling noise | No scheduled additive critic-input or generator-output training noise in the selected benchmark configuration ($0.0$). Task-declared MoG intrinsic sigma is preserved. |
| Parameter averaging and serving | Generator EMA decay=$0$ and serve_average=$0$; all evaluations score live unaveraged weights. |

## Configuration differences

- SMPF translates all four algorithmic pillars of s-means (D^1.5 seeding, tiled L2 assignment, sse_ema x usage^0.35 anisotropic RMS revival across d >= 1, and progressive Empirical-Bayes tau=0.20 shrinkage) into a clock-free, zero-RNG adversarial transport formulation.
- All six s-means sufficient-statistics tensors (proto_centroids, proto_usage, proto_sse, feat_scale, fast_sum, fast_count), low-discrepancy phases, and thermal scalars live exclusively on the critic optimizer's instance-bound step record and serialize bit-for-bit through state_dict() and load_state_dict() with strict corruption rejection.
- Physical Born-Infeld prototype costs sqrt(ell_k^2 + ||y - C_k^eb||^2) - ell_k match each prototype's quadratic basin curvature to its tracked s-means RMS quantization radius ell_k while bounding long-range sample gradients to unit slope across arbitrary coordinate scales.
- One global trainer configuration supplies every displayed result across all tiers and model families without task-specific overrides, clearing 13/13 Tier 1 gates and 29/41 discriminator_stability benchmarks (including vector_unequal_width, vector_unequal_mass, vector_anisotropic, vector_overlap, img_blobs4, img_intensity2, and five_word_joint_hold).

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/smpf.py](../../../particlegan/smpf.py)
- [particlegan/_smpf/prototypes.py](../../../particlegan/_smpf/prototypes.py)
- [particlegan/_smpf/transport.py](../../../particlegan/_smpf/transport.py)
- [particlegan/_smpf/loss.py](../../../particlegan/_smpf/loss.py)
- [particlegan/_smpf/critic.py](../../../particlegan/_smpf/critic.py)
- [particlegan/_smpf/penalty.py](../../../particlegan/_smpf/penalty.py)
- [particlegan/_smpf/prior.py](../../../particlegan/_smpf/prior.py)
- [particlegan/optim/dualnorm.py](../../../particlegan/optim/dualnorm.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [configs/forge/ideas/smpf-v3.json](../../../configs/forge/ideas/smpf-v3.json)
- [configs/forge/studies/smpf-v3-study.json](../../../configs/forge/studies/smpf-v3-study.json)
- [reports/forge/smpf/README.md](../smpf/README.md)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Configuration detail for [EMPF](empf.md).** Its optimizer or settings do not create a separate solution family. This page preserves the original configuration evidence and diagnostics.

## Performance

DCGAN on deterministic mock RGB data measures finite-update speed and memory. These separate performance-source receipts confer no image-quality, scientific qualification or default-promotion credit. Standard BatchNorm is retained in generator and critic; critic input gradients therefore include batch coupling. The shared critic uses a flattened 4×4 Linear scalar head for policy feature extraction; its module layout is fixed across recipes.

Synchronized wall time for one public trainer update, including its critic/generator/prior work; warmup, construction, mock-data generation, evaluation, GIFs and checkpoint writes are excluded.

Every cell is one run; timings are never pooled across devices or contention states. Compare only rows with the same task and matched key (source, runtime/dtype, physical device, initialization and batch streams). Unknown or contended observations are unpaired. Resolution progression changes the task, while each candidate keeps one global recipe. Images/s is B divided by full update time (one D and one G/prior update with separate real batches). Three warmup and ten measured updates give short-window median/p95 measurements; long-horizon policy costs and component phase times remain unknown.

[Compact performance evidence index](../scaling-performance.json) · [Protocol and bounded reproduction](../../../docs/forge-scaling-performance.md)

This registration covers 10 complete recipes. Historical alternatives outside its registered family roster add no missing benchmark cells. A changed selected recipe within a registered family remains unknown until measured.

No current selected-recipe update timings are measured. Missing timing cells remain **unknown**.

<a name="cohort-cuda-daa283c8a640"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [smpf-v3](../../../configs/forge/ideas/smpf-v3.json).

Comparison policy: discriminator_stability revision 9, **tier 1**. Actual executed-policy qualification: **tier 1**, discriminator_stability revision 17 (FAIL). These are separate policy results; no qualification reuse.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `7de3c52187e15029d7122bed110108e089d75c84b177512deb34d28501b05684`. Candidate revision: `ca83937b8fc5443b1eb2cde8426dd8bbf89b3adb54367a726834c2c52252b467`. Runtime cohort: `c286315960a0f4757c22cd425d72c6512e49252cc7425d4b04dcee84324d4a1d`.

[Frozen numerical evidence](../technique-evidence/d2142e759021a9237fef3f4fdaee11acbb51923892e18293b834a0aef5b9536f.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Select smpf-v3 (Scale- & Distortion-Aware Moving Prototype Flow: s-means 4-round D^1.5 coreset seeding, tiled addmm assignment, distortion x usage^0.35 anisotropic RMS revival across d >= 1, Empirical-Bayes tau=0.20 centroid shrinkage, slack-screened physical Born-Infeld prototype Sinkhorn + Laguerre outlier pull + RMS distortion shell, Nadaraya-Watson detrended whitened conditional IMQ-CMMD, H0-connected simplicial chord alignment, unwhitened ParticleEncoding AE reconstruction, and coordinate-whitened SMPF prior regularization) as the canonical SMPF configuration following ordinary discriminator_stability revision-17 CUDA measurement on NVIDIA RTX A6000 (13/13 Tier 1 gates PASS including clockfree_audit_measurement_v1 and precision_gaussian_fp32, plus Tier 2 five_word_joint_hold, mode_hold, vector_unequal_width, vector_unequal_mass, vector_anisotropic, vector_overlap, img_blobs4, and img_intensity2; 29/41 discriminator_stability PASS with 0 BLOCKED). Actual executed-policy qualification and comparison-policy projection retain their separate source-bound proofs; no public-default adoption claim.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation) | [3/3](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1) | [9/12](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) | [0/1 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-4) | [12/16 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-adaptation) | [6/7](smpf.md#cohort-cuda-daa283c8a640-held) |
| [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous) | [4/4](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1) | [9/12](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) | [0/6 (6 not run)](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-4) | [13/22 (6 not run)](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous) | [6/7](smpf.md#cohort-cuda-daa283c8a640-held) |
| [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability) | [11/11](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | [10/18 (2 not run)](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | [0/1 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | [21/30 (3 not run)](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability) | [6/28 (16 incompatible)](smpf.md#cohort-cuda-daa283c8a640-held) |
| [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison) | [3/3](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1) | [9/12](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) | [0/1 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-4) | [12/16 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison) | [6/7](smpf.md#cohort-cuda-daa283c8a640-held) |
| [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer) | [3/3](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1) | [9/12](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) | [0/1 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-4) | [12/16 (1 not run)](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer) | [6/7](smpf.md#cohort-cuda-daa283c8a640-held) |
| [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage) | [3/3](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1) | [9/12](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | [0/0](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-4) | [12/15](smpf.md#cohort-cuda-daa283c8a640-quality_coverage) | [6/7](smpf.md#cohort-cuda-daa283c8a640-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [lazy_cadence_coverage](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage) | [0/0](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) | [0/0](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-2) | [0/0](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-4) | [0/0](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage) |
| [tier1_policy_coverage](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage) | [0/7 (7 not run)](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1) | [0/0](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-2) | [0/0](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-3) | [0/0](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

Certified execution repair: [1fba0229da634f1683b7b461a1667f01](../technique-receipts/1fba0229da634f1683b7b461a1667f01.json) → [27ff411f9b444ec9b74e9ddaae71aa99](../technique-receipts/27ff411f9b444ec9b74e9ddaae71aa99.json). The original outcome and cost remain recorded; displayed task metrics use the certified successor.

Certified execution repair: [cd3d5a0be690417a8b01537af5dbb947](../technique-receipts/cd3d5a0be690417a8b01537af5dbb947.json) → [36d145dd6fd64da0a860f5ce501d0024](../technique-receipts/36d145dd6fd64da0a860f5ce501d0024.json). The original outcome and cost remain recorded; displayed task metrics use the certified successor.

Certified execution repair: [cfd68d5c623a4b88bbc4d8c5259647ba](../technique-receipts/cfd68d5c623a4b88bbc4d8c5259647ba.json) → [cd3d5a0be690417a8b01537af5dbb947](../technique-receipts/cd3d5a0be690417a8b01537af5dbb947.json). The original outcome and cost remain recorded; displayed task metrics use the certified successor.

<a name="cohort-cuda-daa283c8a640-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_gan_multimodal](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_gan_multimodal) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_reconstruction](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_reconstruction) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_vae_categorical](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_categorical) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_vae_hard](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_hard) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_acquisition) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1) | PASS | matches recorded run |
| [zero_tanh_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_smoke) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_sparse_arc) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [precision_gaussian_mxfp8](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_mxfp8) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [precision_gaussian_nvfp4](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_nvfp4) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | INCOMPLETE | matches recorded run |

<a name="cohort-cuda-daa283c8a640-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [model_ae_ddgan_bimodal_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [model_ae_ddgan_gaussian_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | INCOMPLETE | matches recorded run |
| [model_ddgan_bimodal_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [model_ddgan_gaussian_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | FAIL | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [interp_img_patch_shift](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_img_patch_shift) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [interp_saddle_barycentric](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_saddle_barycentric) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_14k) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-3), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100_14k) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100_14k) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-target_shift_recovery) | [adaptation](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-3), [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_rail_escape) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3) | BLOCKED | matches recorded run |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | [clockfree_continuous](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3), [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3), [formulation_comparison](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-3), [host_profile_transfer](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN |

<a name="cohort-cuda-daa283c8a640-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](smpf.md#cohort-cuda-daa283c8a640-experiment-alice_spiral_bigan) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cgan_overlapping_classes](smpf.md#cohort-cuda-daa283c8a640-experiment-cgan_overlapping_classes) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cramer_energy_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-cramer_energy_overlap) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cyclegan_unpaired_constellation](smpf.md#cohort-cuda-daa283c8a640-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_mlp16](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_mlp16) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_res16_scaled_tanh](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_res16_tanh](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_res16_tanh) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [infogan_anisotropic_modes](smpf.md#cohort-cuda-daa283c8a640-experiment-infogan_anisotropic_modes) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_neural_imputer) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_block_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_recovery) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_mcar_p50_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_mcar_p80_anchored](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [pacgan8_grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-pacgan8_grid100) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [pacgan_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-pacgan_unequal_mass) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_bigan_continuous) | [discriminator_stability](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | matches recorded run |

<a name="cohort-cuda-daa283c8a640-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_sparse_arc) | [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | Can 12 zero-sigma ParticlePrior locations learn a 270-degree circular arc without MoG so unseen latent k-NN chords stay on the arc without snapping? | Consider counted status only after independent calibration and one global recipe that passes chord interpolation while retaining every counted pass. |
| [precision_gaussian_mxfp8](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) | INCOMPLETE | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [interp_img_patch_shift](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_img_patch_shift) | [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | Can 16 zero-sigma ParticlePrior codes learn continuous 8x8 single-patch images without MoG so unseen latent chords shift one bright patch instead of 0.5 ghosts? | Consider counted status only after independent calibration and one global recipe that passes image latent interpolation while retaining every counted pass. |
| [interp_saddle_barycentric](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_saddle_barycentric) | [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | Can 24 zero-sigma ParticlePrior locations learn a curved 3D saddle sheet without MoG so unseen 2D latent triangle barycenters stay on the curved surface? | Consider counted status only after independent calibration and one global recipe that passes 2D simplex interpolation while retaining every counted pass. |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2), [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2), [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2), [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) | PASS | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](smpf.md#cohort-cuda-daa283c8a640-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](smpf.md#cohort-cuda-daa283c8a640-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](smpf.md#cohort-cuda-daa283c8a640-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](smpf.md#cohort-cuda-daa283c8a640-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-daa283c8a640-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-daa283c8a640-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-adaptation-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap-tier1-stability-repairs-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-daa283c8a640-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bcap_convolution_images-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bispf_fastpath_quality"></a>

## bispf_fastpath_quality

**bispf_fastpath_quality — revision 1**. [View declaration](../../../configs/forge/views/bispf_fastpath_quality.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-bispf_fastpath_quality-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-bispf_fastpath_quality-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 5**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **4 / 12 / 6 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-daa283c8a640-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-clockfree_continuous-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-dcgan_performance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-dcgan_performance_cpu_smoke-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1"></a>

## ddgan-conditional-repair-v1

**ddgan-conditional-repair-v1 — revision 2**. [View declaration](../../../configs/forge/views/ddgan-conditional-repair-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](smpf.md#cohort-cuda-daa283c8a640-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 17**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **11 / 18 / 1 / 0**. Held counts (measured and shown, not counted): **3 / 9 / 1 / 15**. Its highest tier, Tier 4, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_acquisition) | required | PASS | matches recorded run |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | required | PASS | matches recorded run |
| [zero_tanh_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_smoke) | required | PASS | matches recorded run |
| [model_ae_gan_multimodal](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_gan_multimodal) | required | PASS | matches recorded run |
| [model_ae_reconstruction](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_reconstruction) | required | PASS | matches recorded run |
| [model_vae_categorical](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_categorical) | required | PASS | matches recorded run |
| [model_vae_hard](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_hard) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [precision_gaussian_fp32](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_fp32) | diagnostic | PASS | matches recorded run |
| [precision_gaussian_mxfp8](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | FAIL | matches recorded run |
| [precision_gaussian_nvfp4](smpf.md#cohort-cuda-daa283c8a640-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | INCOMPLETE | matches recorded run |
| [interp_sparse_arc](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_sparse_arc) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | required | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | required | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | required | PASS | matches recorded run |
| [model_ae_ddgan_gaussian_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_gaussian_chain) | required | INCOMPLETE | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | PASS | matches recorded run |
| [model_ddgan_gaussian_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ddgan_gaussian_chain) | required | FAIL | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | required | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | matches recorded run |
| [model_ddgan_bimodal_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | matches recorded run |
| [interp_img_patch_shift](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_img_patch_shift) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | FAIL | matches recorded run |
| [interp_saddle_barycentric](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_saddle_barycentric) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [zero_tanh_rail_escape](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |

<a name="cohort-cuda-daa283c8a640-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [grid100_res16_tanh](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_block_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_mcar_p50_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_mcar_p80_anchored](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [pacgan_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [pacgan8_grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [alice_spiral_bigan](smpf.md#cohort-cuda-daa283c8a640-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [infogan_anisotropic_modes](smpf.md#cohort-cuda-daa283c8a640-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cgan_overlapping_classes](smpf.md#cohort-cuda-daa283c8a640-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cramer_energy_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cyclegan_unpaired_constellation](smpf.md#cohort-cuda-daa283c8a640-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [grid100_res16_scaled_tanh](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |

<a name="cohort-cuda-daa283c8a640-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-daa283c8a640-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | required | FAIL | matches recorded run |
| [img_intensity2_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-daa283c8a640-formulation_comparison-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 6**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-daa283c8a640-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | required | FAIL | matches recorded run |
| [img_intensity2_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-daa283c8a640-host_profile_transfer-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-daa283c8a640-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-k3p_two_pole_horizon-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-lazy_cadence_coverage"></a>

## lazy_cadence_coverage

**lazy_cadence_coverage — revision 1**. [View declaration](../../../configs/forge/views/lazy_cadence_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | diagnostic | UNKNOWN | matches recorded run |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | diagnostic | UNKNOWN | matches recorded run |
| [clockfree_audit_measurement_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-misgan-bispf-research-v1"></a>

## misgan-bispf-research-v1

**misgan-bispf-research-v1 — revision 2**. [View declaration](../../../configs/forge/views/misgan-bispf-research-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Scoped research diagnostic; no ordinary qualification or default adoption.

<a name="cohort-cuda-daa283c8a640-misgan-bispf-research-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_misgan_four_mode](smpf.md#cohort-cuda-daa283c8a640-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-misgan-bispf-research-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-misgan-bispf-research-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-misgan-bispf-research-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-particle_interpolation_stretch"></a>

## particle_interpolation_stretch

**particle_interpolation_stretch — revision 1**. [View declaration](../../../configs/forge/views/particle_interpolation_stretch.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Three zero-sigma particle interpolation stretch toys do not establish whole-suite screen calibration or default adoption.

<a name="cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_sparse_arc) | diagnostic | FAIL | matches recorded run |
| [interp_img_patch_shift](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_img_patch_shift) | diagnostic | FAIL | matches recorded run |
| [interp_saddle_barycentric](smpf.md#cohort-cuda-daa283c8a640-experiment-interp_saddle_barycentric) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1"></a>

## prior-ablation-full-suite-v1

**prior-ablation-full-suite-v1 — revision 1**. [View declaration](../../../configs/forge/views/prior-ablation-full-suite-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [zero_tanh_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_smoke) | diagnostic | PASS | matches recorded run |
| [model_ae_ddgan_gaussian_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | INCOMPLETE | matches recorded run |
| [model_ae_gan_multimodal](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_gan_multimodal) | diagnostic | PASS | matches recorded run |
| [model_ae_reconstruction](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_reconstruction) | diagnostic | PASS | matches recorded run |
| [model_ddgan_gaussian_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ddgan_gaussian_chain) | diagnostic | FAIL | matches recorded run |
| [model_vae_categorical](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_categorical) | diagnostic | PASS | matches recorded run |
| [model_vae_hard](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_hard) | diagnostic | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [clockfree_audit](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | matches recorded run |
| [model_ddgan_bimodal_chain](smpf.md#cohort-cuda-daa283c8a640-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | matches recorded run |
| [ring_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_hold) | diagnostic | UNKNOWN | matches recorded run |
| [ring_extension](smpf.md#cohort-cuda-daa283c8a640-experiment-ring_extension) | diagnostic | UNKNOWN | matches recorded run |
| [zero_tanh_rail_escape](smpf.md#cohort-cuda-daa283c8a640-experiment-zero_tanh_rail_escape) | diagnostic | BLOCKED | matches recorded run |
| [grid100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_recovery) | diagnostic | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](smpf.md#cohort-cuda-daa283c8a640-experiment-misgan_block_neural_imputer) | diagnostic | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_bigan_continuous) | diagnostic | BLOCKED | matches recorded run |

<a name="cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 4**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 0 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-daa283c8a640-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-quality_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022"></a>

## r1r2-repair-word-retention-20261022

**r1r2-repair-word-retention-20261022 — revision 1**. [View declaration](../../../configs/forge/views/r1r2-repair-word-retention-20261022.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

<a name="cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](smpf.md#cohort-cuda-daa283c8a640-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](smpf.md#cohort-cuda-daa283c8a640-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](smpf.md#cohort-cuda-daa283c8a640-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](smpf.md#cohort-cuda-daa283c8a640-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](smpf.md#cohort-cuda-daa283c8a640-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](smpf.md#cohort-cuda-daa283c8a640-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_mass) | diagnostic | PASS | matches recorded run |
| [vector_unequal_width](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_unequal_width) | diagnostic | PASS | matches recorded run |
| [vector_anisotropic](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_anisotropic) | diagnostic | PASS | matches recorded run |
| [vector_overlap](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_overlap) | diagnostic | PASS | matches recorded run |
| [vector_spiral](smpf.md#cohort-cuda-daa283c8a640-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](smpf.md#cohort-cuda-daa283c8a640-experiment-img_blobs4) | diagnostic | PASS | matches recorded run |
| [img_intensity2](smpf.md#cohort-cuda-daa283c8a640-experiment-img_intensity2) | diagnostic | PASS | matches recorded run |
| [grid100](smpf.md#cohort-cuda-daa283c8a640-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](smpf.md#cohort-cuda-daa283c8a640-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](smpf.md#cohort-cuda-daa283c8a640-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-selected-family-coverage-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **7 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](smpf.md#cohort-cuda-daa283c8a640-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-vaegan_coverage"></a>

## vaegan_coverage

**vaegan_coverage — revision 1**. [View declaration](../../../configs/forge/views/vaegan_coverage.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Two toy gates do not establish screen calibration or whole-suite adoption.

<a name="cohort-cuda-daa283c8a640-vaegan_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_vae_hard](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_hard) | diagnostic | PASS | matches recorded run |
| [model_vae_categorical](smpf.md#cohort-cuda-daa283c8a640-experiment-model_vae_categorical) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-daa283c8a640-vaegan_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-vaegan_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-vaegan_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-daa283c8a640-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-daa283c8a640-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00357562 | <= 0.35 | PASS |
| recon_mse | 0.00427908 | <= 0.05 | PASS |

Recorded terminal passing observations: **15**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/09b363eee1d0470d84a9526c8a2da0f4.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-alice_spiral_bigan"></a>

### alice_spiral_bigan

**alice_spiral_bigan: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/alice_spiral_bigan.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-cgan_overlapping_classes"></a>

### cgan_overlapping_classes

**cgan_overlapping_classes: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cgan_overlapping_classes.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/1c11d65b13004ec0b2e6998385dffdbf.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: PASS**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| content_kept | 0.947507 | >= 0.75 | PASS |
| leak_ratio | 0.0106919 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0306171 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0405677 | <= 0.2 | PASS |
| same_dir | 0.0219416 | <= 0.25 | PASS |
| u_kept | 0.946719 | >= 0.85 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/dd21ed8e17ac45d590fe8874c78a629d.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-cramer_energy_overlap"></a>

### cramer_energy_overlap

**cramer_energy_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cramer_energy_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cramer_energy_overlap: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-cyclegan_unpaired_constellation"></a>

### cyclegan_unpaired_constellation

**cyclegan_unpaired_constellation: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cyclegan_unpaired_constellation: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](smpf.md#cohort-cuda-daa283c8a640-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-daa283c8a640-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-ddgan_repair_gaussian_chain_v1"></a>

### ddgan_repair_gaussian_chain_v1

**ddgan_repair_gaussian_chain_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. every joint generation/inverse check throughout the fixed continuation window

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 8167 |
| mass_apple | 0.189453 |
| mass_berry | 0.193359 |
| mass_grape | 0.195312 |
| mass_lemon | 0.199219 |
| mass_melon | 0.222656 |
| mass_tv | 0.0226563 |
| minimum_reconstruction_token_probability | 0.999999 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 1.7484e-07 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 8167 |

[Compact metrics and receipt provenance](../technique-receipts/39540f9277394271b587ba8d5df982a8.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full joint pass plus independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 20001 |
| mass_apple | 0.210938 |
| mass_berry | 0.208008 |
| mass_grape | 0.195312 |
| mass_lemon | 0.198242 |
| mass_melon | 0.1875 |
| mass_tv | 0.0189453 |
| minimum_reconstruction_token_probability | 0.999999 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 2.38419e-07 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 20001 |

[Compact metrics and receipt provenance](../technique-receipts/e37d1e0e8b894e0f85b573112c9fb9a1.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0497063 |
| finite_fraction | 1 |
| mean | 2.06196 |
| mean_error_sigma | 0.123929 |
| sample_count | 4096 |
| std | 0.523406 |
| std_ratio | 1.04681 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/bac63acde9614192be0376c586886491.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. stationary hold, deadline reacquisition and shifted hold

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.042018 |
| finite_fraction | 1 |
| mean | 3.03184 |
| mean_error_sigma | 0.0636888 |
| sample_count | 4096 |
| std | 0.489463 |
| std_ratio | 0.978927 |
| step | 6000 |

[Compact metrics and receipt provenance](../technique-receipts/704b68f87a954414af17e8803a0954c1.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100"></a>

### grid100

**grid100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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
| mass_tv | 0.07197 |
| n | 100000 |
| passed | False |
| precision | 0.22768 |
| precision_gap | 0.761211 |
| problem | grid100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | unavailable |
| valid_n | 100000 |
| within_radius_n | 22768 |

[Compact metrics and receipt provenance](../technique-receipts/dea54fa0cf2d4c349f16841b3cb6e341.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_mlp16"></a>

### grid100_mlp16

**grid100_mlp16: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_mlp16.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_mlp16: native model variants require an explicit profile

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_res16_scaled_tanh"></a>

### grid100_res16_scaled_tanh

**grid100_res16_scaled_tanh: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_scaled_tanh.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_res16_scaled_tanh: native model variants require an explicit profile

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-grid100_res16_tanh"></a>

### grid100_res16_tanh

**grid100_res16_tanh: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_tanh.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_res16_tanh: native model variants require an explicit profile

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-img_bars4"></a>

### img_bars4

**img_bars4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 3 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e1e215c2faed41769f37531906d21e44.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-img_bars4_w64_optimizer_diagnostic"></a>

### img_bars4_w64_optimizer_diagnostic

**img_bars4_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 4 | >= 4 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d844cc4a3ab445a0a2f2bfd88a4b49ac.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **14**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7154eb293bc540fcaa7f1b4677c351c1.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **13**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/9130d351ca674adcbdadb0e77e3dfe10.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-img_stripes2_w64_optimizer_diagnostic"></a>

### img_stripes2_w64_optimizer_diagnostic

**img_stripes2_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-daa283c8a640-experiment-infogan_anisotropic_modes"></a>

### infogan_anisotropic_modes

**infogan_anisotropic_modes: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/infogan_anisotropic_modes.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-interp_img_patch_shift"></a>

### interp_img_patch_shift

**interp_img_patch_shift: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/interp_img_patch_shift.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. failed interp_img_patch_shift thresholds: finite_fraction == 1.0, endpoint_rmse <= 0.09, interp_rmse <= 0.1, interp_peak >= 0.78, quadrant_coverage == 4, midpoint_novelty >= 0.18

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](smpf.md#cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| all_pairs_interp_rmse | 0.054813 |
| edge_count | 24 |
| endpoint_hq | 1 |
| endpoint_rmse | 0.025706 |
| finite_fraction | 1 |
| interp_hq | 1 |
| interp_peak | 0.891488 |
| interp_rmse | 0.0260284 |
| midpoint_novelty | 0.371564 |
| output_knn_interp_peak | 0.894039 |
| output_knn_interp_rmse | 0.0263405 |
| quadrant_coverage | 4 |
| step | 800 |

[Compact metrics and receipt provenance](../technique-receipts/fa154b80ea2445ae9b398313176e92ec.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-interp_saddle_barycentric"></a>

### interp_saddle_barycentric

**interp_saddle_barycentric: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/interp_saddle_barycentric.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. failed interp_saddle_barycentric thresholds: barycenter_hq >= 0.78

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](smpf.md#cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| barycenter_hq | 0.605634 |
| barycenter_novelty | 0.516972 |
| barycenter_surface_mae | 0.0866673 |
| edge_midpoint_surface_mae | 0.0832502 |
| endpoint_hq | 0.708333 |
| endpoint_surface_mae | 0.0767361 |
| finite_fraction | 1 |
| output_knn_barycenter_mae | 0.0829798 |
| step | 800 |
| triangle_count | 71 |
| xy_bins_covered | 15 |

[Compact metrics and receipt provenance](../technique-receipts/bb1d880916a34db9984ae6729c95f6d1.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-interp_sparse_arc"></a>

### interp_sparse_arc

**interp_sparse_arc: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/interp_sparse_arc.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. failed interp_sparse_arc thresholds: interp_hq >= 0.8, endpoint_hq >= 0.85

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [particle_interpolation_stretch / Tier 1](smpf.md#cohort-cuda-daa283c8a640-particle_interpolation_stretch-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| all_pairs_interp_hq | 0.227273 |
| edge_count | 21 |
| endpoint_arc_bins | 12 |
| endpoint_hq | 1 |
| finite_fraction | 1 |
| interp_arc_bins | 27 |
| interp_hq | 0.698413 |
| interp_mean_dist | 0.0847023 |
| midpoint_novelty | 0.360416 |
| output_knn_interp_hq | 0.681818 |
| step | 600 |

[Compact metrics and receipt provenance](../technique-receipts/a9954d84e4d54bbdb85d28351a592e12.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_cos_minus | 0.999998 | >= 0.85 | PASS |
| concept_cos_plus | 0.999932 | >= 0.85 | PASS |
| concept_mag_minus | 1.00499 | <= 1.25 | PASS |
| concept_mag_plus | 0.989027 | <= 1.25 | PASS |
| identity_at_0 | 0.99277 | >= 0.85 | PASS |
| identity_at_mid | 0.978439 | >= 0.85 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/cf1400eef94a4620bf75980d408292bd.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-misgan_block_neural_imputer"></a>

### misgan_block_neural_imputer

**misgan_block_neural_imputer: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_neural_imputer.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_block_neural_imputer: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-misgan_block_recovery"></a>

### misgan_block_recovery

**misgan_block_recovery: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_recovery.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_block_recovery: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-misgan_mcar_p50_recovery"></a>

### misgan_mcar_p50_recovery

**misgan_mcar_p50_recovery: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-misgan_mcar_p80_anchored"></a>

### misgan_mcar_p80_anchored

**misgan_mcar_p80_anchored: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_mcar_p80_anchored: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-mode_hold"></a>

### mode_hold

**mode_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 8 | >= 8 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/fb1e1d7f64f94cfc92c2aad575b4bf21.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_bimodal_chain"></a>

### model_ae_ddgan_bimodal_chain

**model_ae_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-model_ae_ddgan_gaussian_chain"></a>

### model_ae_ddgan_gaussian_chain

**model_ae_ddgan_gaussian_chain: INCOMPLETE**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. task wall-time budget exhausted

Execution: **recorded**.
Attempt: `27ff411f9b444ec9b74e9ddaae71aa99`; raw outcome: timeout.
Recorded execution seconds: 180.351; charged wall seconds: 180.351 (shared execution is charged once to model_ae_ddgan_gaussian_chain).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/27ff411f9b444ec9b74e9ddaae71aa99.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-model_ae_gan_multimodal"></a>

### model_ae_gan_multimodal

**model_ae_gan_multimodal: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_gan_multimodal.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_center_rms_sigma | 0.680848 | <= 1 | PASS |
| component_max_eigen_ratio | 1.07898 | <= 4 | PASS |
| component_min_eigen_ratio | 0.420122 | >= 0.1 | PASS |
| mass_tv | 0.0429688 | <= 0.15 | PASS |
| modes | 4 | >= 4 | PASS |
| precision | 0.986572 | >= 0.9 | PASS |
| reconstruction_mode_agreement | 0.999268 | >= 0.98 | PASS |
| reconstruction_mse | 0.00777564 | <= 0.025 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/cfd4341c9c4248bc8474670015202d4a.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-model_ae_reconstruction"></a>

### model_ae_reconstruction

**model_ae_reconstruction: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_reconstruction.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| reconstruction_mode_agreement | 1 | >= 0.98 | PASS |
| reconstruction_mse | 0.00454776 | <= 0.025 | PASS |

Recorded terminal passing observations: **24**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8b07954b2d654228b1e8b8dfb0663c09.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-model_ddgan_bimodal_chain"></a>

### model_ddgan_bimodal_chain

**model_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-model_ddgan_gaussian_chain"></a>

### model_ddgan_gaussian_chain

**model_ddgan_gaussian_chain: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| final_covariance_eigen_max | 1.27268 | <= 1.4 | PASS |
| final_covariance_eigen_min | 0.998318 | >= 0.65 | PASS |
| final_normalized_mean | 0.0956914 | <= 0.25 | PASS |
| final_projection_cdf | 0.0894697 | <= 0.12 | PASS |
| panel_count | 8 | == 8 | PASS |
| samples_per_panel | 2048 | >= 2048 | PASS |
| t1_q0_covariance_eigen_max | 1.80161 | <= 1.4 | FAIL |
| t1_q0_covariance_eigen_min | 0.951029 | >= 0.4 | PASS |
| t1_q0_normalized_mean | 0.432257 | <= 0.45 | PASS |
| t1_q0_projection_cdf | 0.222414 | <= 0.26 | PASS |
| t1_q1_covariance_eigen_max | 1.61383 | <= 1.4 | FAIL |
| t1_q1_covariance_eigen_min | 1.3151 | >= 0.4 | PASS |
| t1_q1_normalized_mean | 0.220859 | <= 0.45 | PASS |
| t1_q1_projection_cdf | 0.120568 | <= 0.26 | PASS |
| t2_q0_covariance_eigen_max | 1.14307 | <= 1.4 | PASS |
| t2_q0_covariance_eigen_min | 0.733812 | >= 0.6 | PASS |
| t2_q0_normalized_mean | 0.0612251 | <= 0.35 | PASS |
| t2_q0_projection_cdf | 0.0453897 | <= 0.2 | PASS |
| t2_q1_covariance_eigen_max | 0.945998 | <= 1.4 | PASS |
| t2_q1_covariance_eigen_min | 0.656668 | >= 0.6 | PASS |
| t2_q1_normalized_mean | 0.0331442 | <= 0.35 | PASS |
| t2_q1_projection_cdf | 0.0673324 | <= 0.2 | PASS |
| t3_q0_covariance_eigen_max | 0.951879 | <= 1.4 | PASS |
| t3_q0_covariance_eigen_min | 0.94446 | >= 0.65 | PASS |
| t3_q0_normalized_mean | 0.130641 | <= 0.25 | PASS |
| t3_q0_projection_cdf | 0.069181 | <= 0.12 | PASS |
| t3_q1_covariance_eigen_max | 1.00646 | <= 1.4 | PASS |
| t3_q1_covariance_eigen_min | 0.916929 | >= 0.65 | PASS |
| t3_q1_normalized_mean | 0.028788 | <= 0.25 | PASS |
| t3_q1_projection_cdf | 0.0292295 | <= 0.12 | PASS |
| t4_q0_covariance_eigen_max | 1.03459 | <= 1.4 | PASS |
| t4_q0_covariance_eigen_min | 0.993553 | >= 0.65 | PASS |
| t4_q0_normalized_mean | 0.0382824 | <= 0.25 | PASS |
| t4_q0_projection_cdf | 0.0256409 | <= 0.12 | PASS |
| t4_q1_covariance_eigen_max | 1.0274 | <= 1.4 | PASS |
| t4_q1_covariance_eigen_min | 0.954165 | >= 0.65 | PASS |
| t4_q1_normalized_mean | 0.0205782 | <= 0.25 | PASS |
| t4_q1_projection_cdf | 0.0250713 | <= 0.12 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8a6d5e9e035e490eb7698aa94edad90e.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-model_misgan_four_mode"></a>

### model_misgan_four_mode

**model_misgan_four_mode: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_misgan_four_mode.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [misgan-bispf-research-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-misgan-bispf-research-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-model_vae_categorical"></a>

### model_vae_categorical

**model_vae_categorical: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_categorical.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-vaegan_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| decoder_mean_hq | 0.927002 | >= 0.85 | PASS |
| decoder_mean_mass_tv | 0.072998 | <= 0.15 | PASS |
| decoder_mean_modes | 2 | >= 2 | PASS |
| elbo_identity_error | 4.44089e-16 | <= 0.0001 | PASS |
| kl_identity_error | 5.97063e-07 | <= 1e-05 | PASS |
| likelihood_hq | 0.92041 | >= 0.85 | PASS |
| likelihood_mass_tv | 0.0795898 | <= 0.15 | PASS |
| likelihood_modes | 2 | >= 2 | PASS |
| posterior_mutual_information | 0.774307 | >= 0.3 | PASS |
| posterior_noise_variance_ratio | 1.0033 | <= 1.2 | PASS |
| reconstruction_mse | 0.00515926 | <= 0.05 | PASS |
| shuffle_reconstruction_ratio | 435.085 | >= 5 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d622c1e9b56f448d91da8b4b76de01a5.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-model_vae_hard"></a>

### model_vae_hard

**model_vae_hard: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_hard.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-vaegan_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| decoder_mean_hq | 1 | >= 0.85 | PASS |
| decoder_mean_mass_tv | 0.0705566 | <= 0.15 | PASS |
| decoder_mean_modes | 2 | >= 2 | PASS |
| elbo_identity_error | 8.88178e-16 | <= 0.0001 | PASS |
| kl_identity_error | 7.61862e-09 | <= 1e-05 | PASS |
| likelihood_hq | 0.997803 | >= 0.85 | PASS |
| likelihood_mass_tv | 0.0715332 | <= 0.15 | PASS |
| likelihood_modes | 2 | >= 2 | PASS |
| posterior_mutual_information | 2.62864 | >= 0.3 | PASS |
| posterior_noise_variance_ratio | 1.0009 | <= 1.2 | PASS |
| reconstruction_mse | 0.00765197 | <= 0.05 | PASS |
| shuffle_reconstruction_ratio | 295.904 | >= 5 | PASS |

Recorded terminal passing observations: **10**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7f492791f8e84785a90eecd9b1ced563.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-pacgan8_grid100"></a>

### pacgan8_grid100

**pacgan8_grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/pacgan8_grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-pacgan_unequal_mass"></a>

### pacgan_unequal_mass

**pacgan_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/pacgan_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. pacgan_unequal_mass: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4).

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

<a name="cohort-cuda-daa283c8a640-experiment-precision_gaussian_fp32"></a>

### precision_gaussian_fp32

**precision_gaussian_fp32: PASS**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_fp32.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0398802 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0785425 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 0.982959 | <= 1.2 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c430a5d31f5d4d21bb5524bce738ad2f.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-precision_gaussian_mxfp8"></a>

### precision_gaussian_mxfp8

**precision_gaussian_mxfp8: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_mxfp8.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0449666 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.084034 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.05466 | <= 1.2 | PASS |

Recorded terminal passing observations: **1**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c8df068f651248b1ac6d465245148816.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-precision_gaussian_nvfp4"></a>

### precision_gaussian_nvfp4

**precision_gaussian_nvfp4: INCOMPLETE**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. task wall-time budget exhausted

Execution: **recorded**.
Attempt: `36d145dd6fd64da0a860f5ce501d0024`; raw outcome: timeout.
Recorded execution seconds: 300.225; charged wall seconds: 300.225 (shared execution is charged once to precision_gaussian_nvfp4).

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/36d145dd6fd64da0a860f5ce501d0024.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-residual_student"></a>

### residual_student

**residual_student: PASS**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000204121 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/33f3502169da42a7b2b480ca4a2daec9.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.828405 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.850804 | >= 0.15 | PASS |
| hq | 0.90625 | >= 0.85 | PASS |
| mass_tv | 0.0339355 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f2bf79a028ec459eb9aafebf2fbe05fe.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-ring16_bigan_continuous"></a>

### ring16_bigan_continuous

**ring16_bigan_continuous: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_bigan_continuous.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring16_bigan_continuous: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 4](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-rotated100"></a>

### rotated100

**rotated100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.746233 |
| accuracy_pass | False |
| accuracy_score | 5.95425 |
| all_finite | True |
| center_max_sigma | 2.11985 |
| center_rms_sigma | 0.994908 |
| cov_frob_rms | 0.863433 |
| cov_trace_bias | 0.746233 |
| frozen_pass | False |
| mass_tv | 0.05456 |
| n | 100000 |
| passed | False |
| precision | 0.24657 |
| precision_gap | 0.742321 |
| problem | rotated100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.418831 |
| valid_n | 100000 |
| within_radius_n | 24657 |

[Compact metrics and receipt provenance](../technique-receipts/94657345f26b4245a705985cd2df13a1.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-staggered100"></a>

### staggered100

**staggered100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.578074 |
| accuracy_pass | False |
| accuracy_score | 6.06931 |
| all_finite | True |
| center_max_sigma | 2.23484 |
| center_rms_sigma | 1.25153 |
| cov_frob_rms | 0.725346 |
| cov_trace_bias | 0.578074 |
| frozen_pass | False |
| mass_tv | 0.06794 |
| n | 100000 |
| passed | False |
| precision | 0.23688 |
| precision_gap | 0.752011 |
| problem | staggered100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.444261 |
| valid_n | 100000 |
| within_radius_n | 23688 |

[Compact metrics and receipt provenance](../technique-receipts/9a8e478b48e643cfb3de3de8d318e9aa.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [adaptation / Tier 3](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-3) · [clockfree_continuous / Tier 3](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-daa283c8a640-experiment-trajectory"></a>

### trajectory

**trajectory: PASS**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000276611 | <= 0.02 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/129a7a0354a2464180df0181db4f152a.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-daa283c8a640-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.733659 | <= 1 | PASS |
| mean_abs | 0.958313 | >= 0.3 | PASS |

Recorded terminal passing observations: **17**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/3d40648078134fdb849d2febaa3f379b.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](smpf.md#cohort-cuda-daa283c8a640-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](smpf.md#cohort-cuda-daa283c8a640-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-unipolar"></a>

### unipolar

**unipolar: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cover | 0.988441 | >= 0.85 | PASS |
| neu_hold | 0.994636 | >= 0.85 | PASS |
| off_caption | 0.00113044 | <= 0.05 | PASS |

Recorded terminal passing observations: **19**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/95705aed7b02488baa5e20ffc5734a34.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.894352 | >= 0.85 | PASS |
| unused_hold | 0.980466 | >= 0.85 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1d1d8bd8e3d84915ac34ca07e56f2e48.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.127186 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.846477 | >= 0.15 | PASS |
| hq | 0.988525 | >= 0.85 | PASS |
| mass_tv | 0.0392253 | <= 0.15 | PASS |
| sw1_normalized | 0.0470165 | <= 0.18 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c600e03a3fe641c4992bb55c96c555ea.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.0945894 | <= 0.45 | PASS |
| mean_error | 0.0705331 | <= 0.15 | PASS |
| sw1_normalized | 0.0569762 | <= 0.18 | PASS |

Recorded terminal passing observations: **24**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e29ac13e077b4b40ab827f05a88010d2.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.0372064 | <= 0.45 | PASS |
| mean_error | 0.0205208 | <= 0.15 | PASS |
| sw1_normalized | 0.0225948 | <= 0.18 | PASS |

Recorded terminal passing observations: **24**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f8a67e1916114df18cf5c5f405151d04.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.140809 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.830621 | >= 0.15 | PASS |
| hq | 0.989258 | >= 0.85 | PASS |
| mass_tv | 0.0307617 | <= 0.15 | PASS |
| sw1_normalized | 0.0518037 | <= 0.18 | PASS |

Recorded terminal passing observations: **24**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0efb6aa9b90e40ee8202cabfb1bcea09.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.226505 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.358572 | >= 0.15 | PASS |
| hq | 0.994629 | >= 0.85 | PASS |
| mass_tv | 0.0142285 | <= 0.15 | PASS |
| min_mass_ratio | 0.899564 | >= 0.25 | PASS |
| sw1_normalized | 0.0184824 | <= 0.18 | PASS |

Recorded terminal passing observations: **8**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f0a469097df041118dd6ccc073e48a4b.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](smpf.md#cohort-cuda-daa283c8a640-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](smpf.md#cohort-cuda-daa283c8a640-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](smpf.md#cohort-cuda-daa283c8a640-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](smpf.md#cohort-cuda-daa283c8a640-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.121324 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.811149 | >= 0.15 | PASS |
| hq | 0.986816 | >= 0.85 | PASS |
| mass_tv | 0.0532227 | <= 0.15 | PASS |
| sw1_normalized | 0.0628199 | <= 0.18 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f6f08d34e1dd4bab9f4e6d28d1d43dca.json)

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

<a name="cohort-cuda-daa283c8a640-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](smpf.md#cohort-cuda-daa283c8a640-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](smpf.md#cohort-cuda-daa283c8a640-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-daa283c8a640-experiment-zero_tanh_rail_escape"></a>

### zero_tanh_rail_escape

**zero_tanh_rail_escape: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_rail_escape.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. zero_tanh_rail_escape: zero_tanh_rail_escape: unsupported tanh host zero_tanh_rail_escape; only zero_tanh_smoke is routed

Execution: **incompatible; skipped**.

Used by: [discriminator_stability / Tier 3](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-daa283c8a640-experiment-zero_tanh_smoke"></a>

### zero_tanh_smoke

**zero_tanh_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed the declared final zero-tanh bounds

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](smpf.md#cohort-cuda-daa283c8a640-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](smpf.md#cohort-cuda-daa283c8a640-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mean_abs_error | 0.00203167 | <= 0.005 | PASS |
| p99_abs_error | 0.00867768 | <= 0.025 | PASS |
| rms_error | 0.00271688 | <= 0.01 | PASS |
| saturated_fraction | 0 | == 0 | PASS |

[Compact metrics and receipt provenance](../technique-receipts/5054b71607b34d648ad5e3757a828da0.json)

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

- [s-means: outperforming e-means (which outperforms k-means)](https://suayptalha.github.io/s-means-blog/). Multi-round D^1.5 seeding, tiled assignment, distortion x usage^0.35 anisotropic RMS dead-cluster revival, and Empirical-Bayes count-adaptive centroid shrinkage inspiring the SMPF prototype bank and distortion-aware generator transport.
- [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/abs/2611.09501). Online mini-batch sufficient-statistics clustering baseline extended by s-means and EMPF.
- [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://arxiv.org/abs/1810.08278). Entropic self-coupling debiasing adapted here to mass-weighted physical Born-Infeld s-means prototype transport.
