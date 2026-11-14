<!-- Generated Forge family report -->

# BISPF

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [selectable-loss](../technique-inventory.md#tag-selectable-loss)

## Technique overview

Born-Infeld Sinkhorn Particle Flow (BISPF) is a constant-learning-rate ParticleGAN formulation with no EMA critic copy. It couples real and generated samples via scale-free entropic Optimal Transport (Sinkhorn-Knopp), regularizes critic gradients along real endpoints, fake endpoints, and Sinkhorn transport chords using a two-regime Elastic Born-Infeld potential with a zero-mean logit gauge, regularizes the particle prior with a degree-of-freedom-aware Stratified Blue-Noise Particle (SBNP) tether/repulsion law, and applies scale-free Harmonic Mode Capture thermal cooling once all modes are captured.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior with $N$ locations in $\mathbb R^{d_z}$, $x$ a real batch, $z\sim P$ a latent batch and $y=G(z)$ the generated batch. $\Pi$ is the doubly-stochastic Sinkhorn coupling computed from median-normalized squared Euclidean distances $C_{ij}=\lVert x_i-y_j\rVert_2^2/\bar d^2$. $x_{\mathrm{chord}}=(1-u)y+u x_{\pi}$ is a stratified Sinkhorn transport chord. $g_r=\nabla_x D(x)$, $g_f=\nabla_y D(y)$, and $g_c=\nabla_{x_{\mathrm{chord}}} D(x_{\mathrm{chord}})$ are input gradients; $d$ counts critic-input coordinates, $\kappa$ is the cap threshold, $\lambda_{\mathrm{floor}}$ the harmonic floor weight, $\lambda_{\mathrm{cap}}$ the elastic cap weight, and $\gamma_{\mathrm{gauge}}$ the zero-mean logit gauge weight.

**Elastic Born-Infeld Transport-Chord Regularizer and Zero-Mean Gauge**

$$
R_{\mathrm{BISPF}}=\frac{\lambda_{\mathrm{floor}}}{6d}\mathbb E\!\left[\lVert g_r\rVert_2^2+\lVert g_f\rVert_2^2+\lVert g_c\rVert_2^2\right]+\frac{\lambda_{\mathrm{cap}}}{2}\mathbb E\!\left[\phi_\beta(\lVert g_r\rVert_2-\kappa)^2+(1-\alpha_c)\phi_\beta(\lVert g_f\rVert_2-\kappa)^2+\alpha_c\phi_\beta(\lVert g_c\rVert_2-\kappa)^2\right]+\frac{\gamma_{\mathrm{gauge}}}{2}\bar D^2
$$

Here $\phi_\beta(u)=\beta^{-1}\log(1+e^{\beta u})$ is the smooth Born-Infeld elastic wall, $g_c=\nabla_x D((1-u)y+u x_\pi)$ is the critic gradient along Sinkhorn transport chords, and $\bar D=\frac{1}{2}(\mathbb E[D(x)]+\mathbb E[D(y)])$ pins the translational logit gauge without an EMA critic network copy.

**Stratified Blue-Noise Particle (SBNP) Prior Regularizer**

$$
R_{\mathrm{SBNP}}(P)=\lambda_P\left(\mathcal T_{\mathrm{moment}}(P)+w_{\mathrm{repel}}\,\mathbb I(N>2k_{\mathrm{nn}}+d_z)\,\frac{1}{N k_{\mathrm{nn}}}\sum_{i=1}^N\sum_{m=1}^{k_{\mathrm{nn}}}\log\!\left(1+\frac{\bar r_{\mathrm{knn}}^2}{\lVert z_i-z_{\mathcal N_m(i)}\rVert_2^2+\epsilon}\right)\right)
$$

Preserves the initial latent mean and (when $N>2k_{\mathrm{nn}}+d_z$) initial covariance while preventing local particle clumping via scale-free $k$-NN log-repulsion.

## Simplified pseudocode

```text
Choose an adversarial loss, optimizer and constant learning rates.
For each task-owned training update:
  Sample real x and latent z from the task prior P; compute y=G(z).
  Compute scale-free Sinkhorn coupling Pi between x and stop_gradient(y).
  Sample stratified Sinkhorn transport chords x_chord = (1 - u) stop_gradient(y) + u x_pi.
  Evaluate EBI potential on input-gradient norms at x, stop_gradient(y) and x_chord plus zero-mean logit gauge.
  Apply Harmonic Mode Capture thermal cooling when bidirectional 95th-percentile nearest-neighbor transport distance is inside the capture basin.
  Backpropagate critic adversarial loss + BISPF penalty and step D.
  With D fixed, recompute differentiable fake samples y=G(z) and evaluate generator adversarial loss + SBNP prior regularizer.
  Step G, trainable prior P and encoder E when present using constant learning rates.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Selectable non-saturating logistic (selected DualNorm configuration), paired relativistic logistic, Sinkhorn relativistic logistic (SRpGAN), hinge, Wasserstein or least squares. Joint encoder/generator hosts train both critic streams. |
| Optimizer | Supports DualNorm (selected configuration: zero momentum, smoothing 0.001, per-offset convolution) as well as native Adam and formulation-owned BISPF optimizers without an EMA critic network copy. |
| Learning rates and annealing | Strictly constant: selected global LR $0.012$, critic multiplier $1.5$ and prior multiplier $2.5$, with network and prior LR floors fixed at $1.0$ (zero LR annealing). |
| Parameter-gradient clipping | None. Critic spike guard is disabled; gradient bounding acts smoothly through the Elastic Born-Infeld critic potential. |
| Critic penalties and anchors | Two-regime Elastic Born-Infeld Transport-Chord Regularizer (EBI-TCR) evaluated on real, fake and Sinkhorn transport-chord inputs ($\kappa=1$, sharpness $\beta=30$, harmonic floor $\lambda_{\mathrm{floor}}=0.0005$, chord blend $0.02$) plus a zero-mean logit gauge ($\gamma_{\mathrm{gauge}}=0.0005$). No EMA critic anchor. |
| Damping and update guards | Scale-free Harmonic Mode Capture thermal cooling damps post-convergence Poisson mini-batch fluctuations below DualNorm's smoothing floor when the bidirectional 95th-percentile nearest-neighbor distance enters the harmonic capture basin. |
| Training and sampling noise | No scheduled additive critic-input or generator-output training noise in the selected benchmark configuration ($0.0$). Task-declared MoG intrinsic sigma is preserved. |
| Parameter averaging and serving | Generator EMA decay=$0$ and serve_average=$0$; all evaluations score live unaveraged weights. |

## Configuration differences

- BISPF combines the Elastic Born-Infeld Transport-Chord Regularizer (EBI-TCR), zero-mean logit gauge, Stratified Blue-Noise Particle (SBNP) prior regularizer, and Harmonic Mode Capture thermal cooling under constant learning rates.
- The SBNP prior regularizer automatically adapts to prior table capacity: when $N \le 2 k_{\mathrm{nn}} + d_z$ (such as the 5-row discrete word codebook), covariance tethering and $k$-NN repulsion are disabled so discrete codebooks are unperturbed; for continuous particle clouds ($N > 2 k_{\mathrm{nn}} + d_z$), both the initial-moment tether and scale-free $k$-NN log-repulsion are active.
- One global trainer configuration supplies every displayed result across Tier 1, Tier 2 and Tier 3; task-owned architectures, priors, initializations, budgets and gates remain fixed.
- The opt-in throughput recipe combines optimizer_svd_backend="batched" with training_precision="bf16_channels_last". See the [source-bound throughput and quality study](../bispf-throughput/README.md) for workload-specific measurements, unchanged failures and separate current-policy versus endurance-diagnostic results. It does not change the selected recipe or public default.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/bispf.py](../../../particlegan/bispf.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/training.py](../../../particlegan/training.py)
- [configs/forge/ideas/bispf-v1.json](../../../configs/forge/ideas/bispf-v1.json)
- [configs/forge/ideas/bispf-v2.json](../../../configs/forge/ideas/bispf-v2.json)
- [configs/forge/studies/bispf-v1-study.json](../../../configs/forge/studies/bispf-v1-study.json)
- [configs/forge/studies/bispf-v2-study.json](../../../configs/forge/studies/bispf-v2-study.json)
- [reports/forge/bispf/README.md](../bispf/README.md)

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
| BISPF · `bispf-optimized-quintic-v1` | dcgan_performance_r32: 32², B16, W64 | COMPLETE | 29.08 / 30.72 | 550.30 | 90.40 / 124.00 | clear | `d56240d15558` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/bispf-dcgan_performance_r32-79ec08957343240e.json) |
| BISPF · `bispf-optimized-quintic-v1` | dcgan_performance_r64: 64², B16, W64 | COMPLETE | 50.32 / 51.26 | 317.44 | 221.78 / 316.00 | clear | `524c07acf3db` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/bispf-dcgan_performance_r64-ad21eb318743f60e.json) |
| BISPF · `bispf-optimized-quintic-v1` | dcgan_performance_w128: 64², B16, W128 | COMPLETE | 109.42 / 110.77 | 146.12 | 674.18 / 940.00 | clear | `7ca95dbe0fca` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/bispf-dcgan_performance_w128-2cd7dbacbab4d1fd.json) |
| BISPF · `bispf-optimized-quintic-v1` | dcgan_performance_b64: 64², B64, W64 | COMPLETE | 89.36 / 90.55 | 716.18 | 558.22 / 632.00 | clear | `0c19272217b1` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/bispf-dcgan_performance_b64-bb8b8f220f33cfa3.json) |
| BISPF · `bispf-optimized-quintic-v1` | dcgan_performance_r128: 128², B16, W64 | COMPLETE | 121.20 / 122.41 | 131.92 | 632.24 / 842.00 | clear | `1b8df8094599` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/bispf-dcgan_performance_r128-e897cad1c8766512.json) |
| BISPF · `bispf-optimized-quintic-v1` | dcgan_performance_r256: 256², B16, W64 | COMPLETE | 341.59 / 342.25 | 46.87 | 2,119.28 / 2,310.00 | clear | `c4765c66fb33` · [receipt](../scaling-receipts/dcgan-performance-cloud-759e40be701e/bispf-dcgan_performance_r256-d9d3ef615358712e.json) |

<a name="cohort-cuda-1bf9d7d34422"></a>

## Current benchmark

Runtime: **cuda**. Selected configuration: [bispf-optimized-quintic-v1](../../../configs/forge/ideas/bispf-optimized-quintic-v1.json).

Comparison policy: discriminator_stability revision 9, **tier 1**. Actual executed-policy qualification: **tier 3**, discriminator_stability revision 11 (PASS). These are separate policy results; no qualification reuse.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`. Candidate revision: `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`. Runtime cohort: `a5c1a7106c9f87a47b7fba9ef08c8d27f8a8323752eae254adb7d25cca833949`.

[Frozen numerical evidence](../technique-evidence/2c45cc0649ef839b6455d74b353d4331fd789686dda68f104536706fb0970358.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: recorded_source_measurement. Recorded source-bound original measurement retained; stale against current task contracts. No current measurement, qualification reuse or comparable live ranking. Original selection: Preserve every incumbent pass with one globally configured optimized BISPF recipe. Actual executed-policy qualification, older comparison-policy tiers, exercised wide-channel diagnostics and observed performance targets retain their separate source-bound proofs. No public-default adoption or natural-image quality claim.

</details>

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [9/12](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 diagnostic attempt)](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-4) | [12/16 (1 diagnostic attempt)](bispf.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/7](bispf.md#cohort-cuda-1bf9d7d34422-held) |
| [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [4/4](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [9/12](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [1/6 (3 not run, 2 diagnostic attempts)](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-4) | [14/22 (3 not run, 2 diagnostic attempts)](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/7](bispf.md#cohort-cuda-1bf9d7d34422-held) |
| [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [6/11 (4 not run, 1 diagnostic attempt)](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [10/18 (4 not run)](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [1/1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | [17/30 (8 not run, 1 diagnostic attempt)](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/28 (7 not run, 14 diagnostic attempts)](bispf.md#cohort-cuda-1bf9d7d34422-held) |
| [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [9/12](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [1/1](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-4) | [13/16](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/7](bispf.md#cohort-cuda-1bf9d7d34422-held) |
| [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [9/12](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [1/1](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-4) | [13/16](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/7](bispf.md#cohort-cuda-1bf9d7d34422-held) |
| [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [9/12](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-4) | [12/15](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/7](bispf.md#cohort-cuda-1bf9d7d34422-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [bispf_throughput_endurance](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-1) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-2) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-4) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance) |
| [bispf_throughput_small](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-1) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-2) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-4) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small) |
| [lazy_cadence_coverage](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-4) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage) |
| [tier1_policy_coverage](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) | [0/7 (7 not run)](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3) | [0/0](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-1bf9d7d34422-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | changed since run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | PASS | changed since run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_gan_multimodal](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | changed since run |
| [zero_tanh_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | changed since run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | PASS | changed since run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | changed since run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | matches recorded run |
| [model_ae_ddgan_bimodal_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | changed since run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |
| [interp_img_patch_shift](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | PASS | matches recorded run |
| [rotated100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | [adaptation](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3), [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | UNKNOWN | recorded definition unavailable |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [clockfree_continuous](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS |

<a name="cohort-cuda-1bf9d7d34422-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](bispf.md#cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_mlp16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](bispf.md#cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | [discriminator_stability](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | Can 12 zero-sigma ParticlePrior locations learn a 270-degree circular arc without MoG so unseen latent k-NN chords stay on the arc without snapping? | Consider counted status only after independent calibration and one global recipe that passes chord interpolation while retaining every counted pass. |
| [precision_gaussian_mxfp8](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [interp_img_patch_shift](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | Can 16 zero-sigma ParticlePrior codes learn continuous 8x8 single-patch images without MoG so unseen latent chords shift one bright patch instead of 0.5 ghosts? | Consider counted status only after independent calibration and one global recipe that passes image latent interpolation while retaining every counted pass. |
| [interp_saddle_barycentric](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | Can 24 zero-sigma ParticlePrior locations learn a curved 3D saddle sheet without MoG so unseen 2D latent triangle barycenters stay on the curved surface? | Consider counted status only after independent calibration and one global recipe that passes 2D simplex interpolation while retaining every counted pass. |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](bispf.md#cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](bispf.md#cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-1bf9d7d34422-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality"></a>

## bispf_fastpath_quality

**bispf_fastpath_quality — revision 1**. [View declaration](../../../configs/forge/views/bispf_fastpath_quality.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_endurance"></a>

## bispf_throughput_endurance

**bispf_throughput_endurance — revision 2**. [View declaration](../../../configs/forge/views/bispf_throughput_endurance.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | diagnostic | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_small"></a>

## bispf_throughput_small

**bispf_throughput_small — revision 2**. [View declaration](../../../configs/forge/views/bispf_throughput_small.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 5**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **4 / 12 / 6 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | changed since run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | PASS | matches recorded run |
| [grid100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1"></a>

## ddgan-conditional-repair-v1

**ddgan-conditional-repair-v1 — revision 2**. [View declaration](../../../configs/forge/views/ddgan-conditional-repair-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 17**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **11 / 18 / 1 / 0**. Held counts (measured and shown, not counted): **3 / 9 / 1 / 15**. Its highest tier, Tier 4, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | changed since run |
| [ring16_acquisition](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | required | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | required | PASS | changed since run |
| [zero_tanh_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [precision_gaussian_fp32](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_fp32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [interp_sparse_arc](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | required | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | required | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | PASS | matches recorded run |
| [model_ae_ddgan_gaussian_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [model_ddgan_gaussian_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | FAIL | changed since run |
| [model_ae_ddgan_bimodal_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | PASS | matches recorded run |
| [zero_tanh_rail_escape](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [alice_spiral_bigan](bispf.md#cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](bispf.md#cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | FAIL | changed since run |
| [img_intensity2_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 6**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | FAIL | changed since run |
| [img_intensity2_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-lazy_cadence_coverage"></a>

## lazy_cadence_coverage

**lazy_cadence_coverage — revision 1**. [View declaration](../../../configs/forge/views/lazy_cadence_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | diagnostic | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | diagnostic | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1"></a>

## misgan-bispf-research-v1

**misgan-bispf-research-v1 — revision 2**. [View declaration](../../../configs/forge/views/misgan-bispf-research-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Scoped research diagnostic; no ordinary qualification or default adoption.

<a name="cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_misgan_four_mode](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-particle_interpolation_stretch"></a>

## particle_interpolation_stretch

**particle_interpolation_stretch — revision 1**. [View declaration](../../../configs/forge/views/particle_interpolation_stretch.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Three zero-sigma particle interpolation stretch toys do not establish whole-suite screen calibration or default adoption.

<a name="cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1"></a>

## prior-ablation-full-suite-v1

**prior-ablation-full-suite-v1 — revision 1**. [View declaration](../../../configs/forge/views/prior-ablation-full-suite-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [zero_tanh_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [clockfree_audit](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |
| [model_ae_ddgan_bimodal_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | diagnostic | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | diagnostic | PASS | matches recorded run |
| [zero_tanh_rail_escape](bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 4**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 0 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022"></a>

## r1r2-repair-word-retention-20261022

**r1r2-repair-word-retention-20261022 — revision 1**. [View declaration](../../../configs/forge/views/r1r2-repair-word-retention-20261022.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

<a name="cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |

<a name="cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | PASS | changed since run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | PASS | changed since run |
| [trajectory](bispf.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | PASS | changed since run |
| [residual_student](bispf.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | PASS | changed since run |
| [unipolar](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | PASS | changed since run |
| [mode_hold](bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **7 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-vaegan_coverage"></a>

## vaegan_coverage

**vaegan_coverage — revision 1**. [View declaration](../../../configs/forge/views/vaegan_coverage.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Two toy gates do not establish screen calibration or whole-suite adoption.

<a name="cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_vae_hard](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.018445 | <= 0.35 | PASS |
| recon_mse | 0.00366872 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1794aa9a44ec4397b0383f5470bdc915.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan"></a>

### alice_spiral_bigan

**alice_spiral_bigan: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/alice_spiral_bigan.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **FAIL** (new_attempt; qualification unchanged). Failed alice_spiral_bigan bounds: insufficient terminal checks. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes"></a>

### cgan_overlapping_classes

**cgan_overlapping_classes: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cgan_overlapping_classes.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on cgan_overlapping_classes. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **PASS** (new_attempt; qualification unchanged). declared state/horizon/cadence/restart comparisons agree; source audit bound [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/d2be1cfa78f74e1ead5dec8b3acc88af.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: PASS**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| content_kept | 0.962408 | >= 0.75 | PASS |
| leak_ratio | 0.0178634 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0258542 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.025216 | <= 0.2 | PASS |
| same_dir | 0.0130099 | <= 0.25 | PASS |
| u_kept | 0.999095 | >= 0.85 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/adbb1a19ff7a4019a2fd6a083132714d.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap"></a>

### cramer_energy_overlap

**cramer_energy_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cramer_energy_overlap.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on cramer_energy_overlap. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation"></a>

### cyclegan_unpaired_constellation

**cyclegan_unpaired_constellation: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on cyclegan_unpaired_constellation. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_gaussian_chain_v1"></a>

### ddgan_repair_gaussian_chain_v1

**ddgan_repair_gaussian_chain_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. every joint generation/inverse check throughout the fixed continuation window

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 4834 |
| mass_apple | 0.21582 |
| mass_berry | 0.191406 |
| mass_grape | 0.205078 |
| mass_lemon | 0.193359 |
| mass_melon | 0.194336 |
| mass_tv | 0.0208984 |
| minimum_reconstruction_token_probability | 0.951635 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 0.00218637 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 4834 |

[Compact metrics and receipt provenance](../technique-receipts/ebb0483217144ba8b956d045b1458ef4.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. any scheduled full joint pass plus independent same-state confirmation

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 20001 |
| mass_apple | 0.195312 |
| mass_berry | 0.1875 |
| mass_grape | 0.210938 |
| mass_lemon | 0.198242 |
| mass_melon | 0.208008 |
| mass_tv | 0.0189453 |
| minimum_reconstruction_token_probability | 0.999617 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 3.4861e-05 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 20001 |

[Compact metrics and receipt provenance](../technique-receipts/92d31868c53c4b75bc4be574de0de3d7.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.135031 |
| finite_fraction | 1 |
| mean | 1.82751 |
| mean_error_sigma | 0.344978 |
| sample_count | 4096 |
| std | 0.491298 |
| std_ratio | 0.982596 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/f3244ce97e484b038d99ac653ba8a663.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. stationary hold, deadline reacquisition and shifted hold

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0703186 |
| finite_fraction | 1 |
| mean | 3.02737 |
| mean_error_sigma | 0.0547359 |
| sample_count | 4096 |
| std | 0.438255 |
| std_ratio | 0.87651 |
| step | 6000 |

[Compact metrics and receipt provenance](../technique-receipts/6141a1c953cc41189a24685cf8080743.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100"></a>

### grid100

**grid100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.325427 |
| accuracy_pass | False |
| accuracy_score | 6.04411 |
| all_finite | True |
| center_max_sigma | 2.87639 |
| center_rms_sigma | 1.49713 |
| cov_frob_rms | 0.61016 |
| cov_trace_bias | 0.325427 |
| frozen_pass | False |
| mass_tv | 0.17028 |
| n | 100000 |
| passed | False |
| precision | 0.30497 |
| precision_gap | 0.683921 |
| problem | grid100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.42394 |
| valid_n | 100000 |
| within_radius_n | 30497 |

[Compact metrics and receipt provenance](../technique-receipts/e33da1416b6248d1b514ef8c650cd3f8.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16"></a>

### grid100_mlp16

**grid100_mlp16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_mlp16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh"></a>

### grid100_res16_scaled_tanh

**grid100_res16_scaled_tanh: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_scaled_tanh.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Post-optimizer empirical cluster fitting, generator identity resets and particle dispersion are additional task-specific update rules. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on grid100_res16_scaled_tanh. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh"></a>

### grid100_res16_tanh

**grid100_res16_tanh: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_tanh.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-img_bars4"></a>

### img_bars4

**img_bars4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [bispf_throughput_small / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.96875 | >= 0.9 | PASS |
| modes | 2 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/048820009b3b44108cad4bc5fee0854b.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_bars4_w64_optimizer_diagnostic"></a>

### img_bars4_w64_optimizer_diagnostic

**img_bars4_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.8125 | >= 0.9 | FAIL |
| modes | 3 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/9fd0c3e51af84b8895095f51765d84e4.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.59375 | >= 0.9 | FAIL |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/81b4b1e0cd374e9e80a683e5e192b824.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [bispf_throughput_small / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_small-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/10ebffcaed53495b8ddd71207af7ab89.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_stripes2_w64_optimizer_diagnostic"></a>

### img_stripes2_w64_optimizer_diagnostic

**img_stripes2_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes"></a>

### infogan_anisotropic_modes

**infogan_anisotropic_modes: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/infogan_anisotropic_modes.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **FAIL** (new_attempt; qualification unchanged). Failed infogan_anisotropic_modes bounds: hq. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift"></a>

### interp_img_patch_shift

**interp_img_patch_shift: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_img_patch_shift.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric"></a>

### interp_saddle_barycentric

**interp_saddle_barycentric: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_saddle_barycentric.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc"></a>

### interp_sparse_arc

**interp_sparse_arc: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_sparse_arc.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [particle_interpolation_stretch / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_cos_minus | 0.999965 | >= 0.85 | PASS |
| concept_cos_plus | 0.999837 | >= 0.85 | PASS |
| concept_mag_minus | 0.996567 | <= 1.25 | PASS |
| concept_mag_plus | 1.00835 | <= 1.25 | PASS |
| identity_at_0 | 0.993095 | >= 0.85 | PASS |
| identity_at_mid | 0.969 | >= 0.85 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e81eec8db83742b6bd6e1c14e8e95031.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer"></a>

### misgan_block_neural_imputer

**misgan_block_neural_imputer: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_neural_imputer.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Observed-subspace solver overwrites generators/prior tables; neural-imputer observation may use posterior resampling instead of G_i. EMA sampling and complete-state restoration also differ from the declared live contract. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on misgan_block_neural_imputer. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery"></a>

### misgan_block_recovery

**misgan_block_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Observed-subspace solver overwrites generators/prior tables; neural-imputer observation may use posterior resampling instead of G_i. EMA sampling and complete-state restoration also differ from the declared live contract. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on misgan_block_recovery. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery"></a>

### misgan_mcar_p50_recovery

**misgan_mcar_p50_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Observed-subspace solver overwrites generators/prior tables; neural-imputer observation may use posterior resampling instead of G_i. EMA sampling and complete-state restoration also differ from the declared live contract. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on misgan_mcar_p50_recovery. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored"></a>

### misgan_mcar_p80_anchored

**misgan_mcar_p80_anchored: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Observed-subspace solver overwrites generators/prior tables; neural-imputer observation may use posterior resampling instead of G_i. EMA sampling and complete-state restoration also differ from the declared live contract. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on misgan_mcar_p80_anchored. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-mode_hold"></a>

### mode_hold

**mode_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bispf_throughput_endurance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.969238 | >= 0.9 | PASS |
| modes | 8 | >= 8 | PASS |

Recorded terminal passing observations: **10**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e09372c6d8004fd8aa529caff91bb759.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain"></a>

### model_ae_ddgan_bimodal_chain

**model_ae_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain"></a>

### model_ae_ddgan_gaussian_chain

**model_ae_ddgan_gaussian_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal"></a>

### model_ae_gan_multimodal

**model_ae_gan_multimodal: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_gan_multimodal.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction"></a>

### model_ae_reconstruction

**model_ae_reconstruction: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_reconstruction.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain"></a>

### model_ddgan_bimodal_chain

**model_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain"></a>

### model_ddgan_gaussian_chain

**model_ddgan_gaussian_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_misgan_four_mode"></a>

### model_misgan_four_mode

**model_misgan_four_mode: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_misgan_four_mode.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [misgan-bispf-research-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical"></a>

### model_vae_categorical

**model_vae_categorical: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_categorical.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-model_vae_hard"></a>

### model_vae_hard

**model_vae_hard: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_hard.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100"></a>

### pacgan8_grid100

**pacgan8_grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/pacgan8_grid100.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **FAIL** (new_attempt; qualification unchanged). Failed pacgan8_grid100 bounds: hq, min_cov_eig_ratio, modes. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 95 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| min_cov_eig_ratio | >= 0.15 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass"></a>

### pacgan_unequal_mass

**pacgan_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/pacgan_unequal_mass.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **PASS** (new_attempt; qualification unchanged). Sustained 5/5 terminal checks on pacgan_unequal_mass. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_fp32"></a>

### precision_gaussian_fp32

**precision_gaussian_fp32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_fp32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8"></a>

### precision_gaussian_mxfp8

**precision_gaussian_mxfp8: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_mxfp8.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4"></a>

### precision_gaussian_nvfp4

**precision_gaussian_nvfp4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-residual_student"></a>

### residual_student

**residual_student: PASS**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.00023213 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/04219fc2d9d14cfdbb51fb86c841ec85.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.466252 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.380435 | >= 0.15 | PASS |
| hq | 0.951904 | >= 0.85 | PASS |
| mass_tv | 0.092041 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/39d19349c7bd4b75acb5c4e69b53c93a.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous"></a>

### ring16_bigan_continuous

**ring16_bigan_continuous: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring16_bigan_continuous.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

**Scientific scope correction:** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. The original observation remains recorded below as an archived diagnostic; it does not establish faithful execution of this selected recipe/task contract. [Source-bound correction](../evidence-corrections-2026-11-08.json).

Run attempt: **FAIL** (new_attempt; qualification unchanged). Failed ring16_bigan_continuous bounds: hq, recon_rmse. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ring_extension"></a>

### ring_extension

**ring_extension: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed every extension observation without selecting another window

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bispf_throughput_endurance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-1) · [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| confirmation_checks | 200 |
| converged_step | 1782 |
| convergence | first consecutive qualifying window; operational quality criterion |
| extension_checks | 300 |
| extension_passing | 300 |
| first_hold_failure | unavailable |
| frozen_toy_verdicts_unchanged | True |
| hold | the next updates, disjoint from confirmation; no restart after failure |
| hold_budget | 1200 |
| hold_budget_complete | True |
| hold_checks | 1200 |
| longest_settling_streak | 200 |
| min_hq | 0.9 |
| modes | 8 |
| name | first_convergence_then_hold_v1 |
| settling_budget | 4800 |
| settling_checks | 582 |
| settling_failures | 5 |
| start_step | 1200 |
| status | PASS |
| training_feedback | False |

[Compact metrics and receipt provenance](../technique-receipts/b93cb8db7f8146d0970435cf17a55e2d.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-ring_hold"></a>

### ring_hold

**ring_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed first-convergence hold

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bispf_throughput_endurance / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bispf_throughput_endurance-tier-1) · [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| confirmation_checks | 200 |
| converged_step | 1782 |
| convergence | first consecutive qualifying window; operational quality criterion |
| first_hold_failure | unavailable |
| frozen_toy_verdicts_unchanged | True |
| hold | the next updates, disjoint from confirmation; no restart after failure |
| hold_budget | 1200 |
| hold_budget_complete | True |
| hold_checks | 1200 |
| longest_settling_streak | 200 |
| min_hq | 0.9 |
| modes | 8 |
| name | first_convergence_then_hold_v1 |
| settling_budget | 4800 |
| settling_checks | 582 |
| settling_failures | 5 |
| start_step | 1200 |
| status | PASS |
| training_feedback | False |

[Compact metrics and receipt provenance](../technique-receipts/b93cb8db7f8146d0970435cf17a55e2d.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-rotated100"></a>

### rotated100

**rotated100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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
| mass_tv | 0.11132 |
| n | 100000 |
| passed | False |
| precision | 0.22301 |
| precision_gap | 0.765881 |
| problem | rotated100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | unavailable |
| valid_n | 100000 |
| within_radius_n | 22301 |

[Compact metrics and receipt provenance](../technique-receipts/97be11c288dc45bb9a98d1f1165ffbf0.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-staggered100"></a>

### staggered100

**staggered100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.54169 |
| accuracy_pass | False |
| accuracy_score | 5.59569 |
| all_finite | True |
| center_max_sigma | 2.5165 |
| center_rms_sigma | 1.14288 |
| cov_frob_rms | 0.708947 |
| cov_trace_bias | 0.54169 |
| frozen_pass | False |
| mass_tv | 0.12142 |
| n | 100000 |
| passed | False |
| precision | 0.33617 |
| precision_gap | 0.652721 |
| problem | staggered100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.369111 |
| valid_n | 100000 |
| within_radius_n | 33617 |

[Compact metrics and receipt provenance](../technique-receipts/9a8e496fffff47a68976f46b69b1be2d.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **FAIL** (new_attempt; qualification unchanged). recomputed live recovery and matched frozen negative control [Attempt evidence](../attempt-coverage.json).

Used by: [adaptation / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) · [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-trajectory"></a>

### trajectory

**trajectory: PASS**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000219656 | <= 0.02 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1ff520bfed664e8082291bc43dabbe9c.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.876375 | <= 1 | PASS |
| mean_abs | 0.95826 | >= 0.3 | PASS |

Recorded terminal passing observations: **17**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0d2f7899cd3b4751935da9c75ae5717e.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-unipolar"></a>

### unipolar

**unipolar: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cover | 0.989121 | >= 0.85 | PASS |
| neu_hold | 0.995471 | >= 0.85 | PASS |
| off_caption | 0.000135526 | <= 0.05 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d9651820d35d460ab836ec1559294b99.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.972497 | >= 0.85 | PASS |
| unused_hold | 0.989819 | >= 0.85 | PASS |

Recorded terminal passing observations: **15**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/89634c5d4f7f48578d55937f83591389.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.313814 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.32515 | >= 0.15 | PASS |
| hq | 0.995605 | >= 0.85 | PASS |
| mass_tv | 0.175212 | <= 0.15 | FAIL |
| sw1_normalized | 0.213624 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c82e5be061d341f184fbf343515c822c.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.522296 | <= 0.45 | FAIL |
| mean_error | 0.101512 | <= 0.15 | PASS |
| sw1_normalized | 0.149086 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f68126b2dfe44d158054ed18be8960e6.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.0804742 | <= 0.45 | PASS |
| mean_error | 0.0714902 | <= 0.15 | PASS |
| sw1_normalized | 0.0555712 | <= 0.18 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/01c3537f395745d2aa12970ed46509f9.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.306438 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.483331 | >= 0.15 | PASS |
| hq | 0.990967 | >= 0.85 | PASS |
| mass_tv | 0.0588379 | <= 0.15 | PASS |
| sw1_normalized | 0.126826 | <= 0.18 | PASS |

Recorded terminal passing observations: **6**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/cf83acf6ad354b2dbb53ea5042808351.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 1.76764 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0 | >= 0.15 | FAIL |
| hq | 0.977295 | >= 0.85 | PASS |
| mass_tv | 0.0574219 | <= 0.15 | PASS |
| min_mass_ratio | 0 | >= 0.25 | FAIL |
| sw1_normalized | 0.101744 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c66ca8acc4634cf1920c23224cfc6faf.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 9.6869 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.0934703 | >= 0.15 | FAIL |
| hq | 0.981934 | >= 0.85 | PASS |
| mass_tv | 0.277588 | <= 0.15 | FAIL |
| sw1_normalized | 0.287383 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7d0d9f6b12354275b1be7310ca74d34b.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape"></a>

### zero_tanh_rail_escape

**zero_tanh_rail_escape: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_rail_escape.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **FAIL** (new_attempt; qualification unchanged). Failed rail-escape bounds: mean_abs_error <= 0.005, rms_error <= 0.01, p99_abs_error <= 0.025. [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 3](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke"></a>

### zero_tanh_smoke

**zero_tanh_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **FAIL** (current_source_readout; qualification unchanged). failed thresholds: mean_abs_error <= 0.005, rms_error <= 0.01 [Attempt evidence](../records/readout-143d85d4540548b0ecec1604.json).

Used by: [discriminator_stability / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bispf.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895). Entropic Optimal Transport Sinkhorn-Knopp coupling used to construct transport chords and scale-free transport residuals.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base first-order optimizer family supported alongside DualNorm.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Relativistic pairing foundation extended by Sinkhorn Relativistic Pairing (SRpGAN).
