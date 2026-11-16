<!-- Generated Forge family report -->

# BCAP

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [selectable-loss](../technique-inventory.md#tag-selectable-loss)

## Technique overview

BCAP adds a fixed critic penalty whenever the input-gradient norm exceeds a threshold, evaluated separately on real and generated data. This high-level formulation defines the family. Adam, normalized-gradient and DualNorm optimizers, learning rates, momentum and loss settings are configuration choices. The current benchmark uses the explicitly selected winning DualNorm recipe. Historical BCAP-with-K3P formulations and optimizer alternatives retain their complete, separate evidence below.

![Real and generated samples share a critic; a soft penalty discourages excessive input-gradient slopes and adds to the critic loss.](assets/bcap-explainer.png)

Conceptual illustration of the BCAP critic-loss mechanism, not measured training results. Generator/prior updates are described below.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $\eta_0$ is the global base LR, $m_D$ and $m_P$ the role multipliers. A joint host also has an encoder $E$, whose adversarial objective must train both joint critic streams.

**Default paired adversarial loss**

$$
\begin{aligned}\ell_D&=\mathbb E\!\left[\operatorname{softplus}(C(y)-C(x))\right],\\\ell_G&=\mathbb E\!\left[\operatorname{softplus}(C(x)-C(y))\right].\end{aligned}
$$

These scalar losses are minimized. The critic objective is $\ell_D+R$; the generator adds only its declared auxiliary objectives. Critic fakes are detached, and generator fakes and both scores are recomputed after the critic step. $C$ includes any declared fresh input noise. This equation is the canonical paired-loss example; non-saturating, hinge, Wasserstein and least-squares options keep their own loss formulas, and joint hosts use their explicit generator/encoder objective.

**Fixed capped input-gradient penalty**

$$
R_{\mathrm{BCAP}}=\frac{\lambda}{2}\left\{\mathbb E\!\left[(\lVert g_r\rVert_2-\kappa)_+^2\right]+\mathbb E\!\left[(\lVert g_f\rVert_2-\kappa)_+^2\right]\right\}.
$$

The cap is a soft loss penalty on real and fake critic input-gradient norms, without division by input dimension. It does not directly clip critic parameter gradients or weights. The canonical BCAP cards use $\lambda=1$ and $\kappa=1$; source-bound selected recipes may differ.

## Simplified pseudocode

```text
Choose an adversarial loss, optimizer and its configuration.
For each task-owned training update:
  Sample real x and latent z from the task prior P; compute y=G(z).
  Critic objective = adversarial_D(D(x), D(stop_gradient(y))) + fixed BCAP penalty.
  Compute BCAP from real and fake input-gradient norms; the penalty updates D only.
  Backpropagate the critic objective and step D using the configured optimizer.
  With D fixed, recompute differentiable fake samples and scores.
  Backpropagate adversarial_G plus task-owned auxiliary objectives.
  Step G, trainable prior P and encoder E when present using their configured rates.
  Joint encoder/generator hosts reverse labels on both critic streams.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Selectable paired relativistic logistic (canonical/default), non-saturating logistic, hinge, Wasserstein or least squares. The scalar losses are minimized. Relativistic $D$: $\mathbb E[\operatorname{softplus}(C(y)-C(x))]$; $G$ reverses the difference. Joint hosts use the explicit generator/encoder objective. |
| Optimizer | Native PyTorch Adam. Canonical $\beta_1=0$, $\beta_2=0.999$ and $\varepsilon=10^{-8}$; moments are constant. |
| Learning rates and annealing | Constant: canonical global LR $0.00425$, D multiplier $1$ and prior multiplier $2$. The measured round also tested global LR $0.0010625$. Network and prior LR floors are $1$, so the scheduled API path performs no annealing. |
| Parameter-gradient clipping | None. Critic spike guard is disabled; BCAP acts through the loss, not parameter-gradient clipping. |
| Critic penalties and anchors | Fixed real/fake one-sided squared L2 cap. Canonical $\lambda=1$, $\kappa=1$, applied every critic update. No K3P early/late blend or critic-anchor term. |
| Damping and update guards | No A2 latent-row damping, direct-particle response gain, or adaptive critic step intervention. |
| Training and sampling noise | No additive critic-input or generator-output training noise. Sampling noise inherent to a task-declared MoG prior remains part of that prior, not an extra BCAP intervention. |
| Parameter averaging and serving | Canonical generator EMA decay=$0$ and no critic-anchor penalty; the reported round scores clean live weights. |

## Configuration differences

- The fixed capped input-gradient penalty defines BCAP. Optimizer choice, learning rates, momentum, numerical floors and adversarial-loss settings stay within this solution family.
- The selected configuration table reports the executed trainer recipe. Task-owned architecture, initialization, prior, sampling, update budget and auxiliary objectives remain bound to each experiment.
- One whole recorded configuration supplies every displayed result. Passing cells from different recipes or sources are never combined.
- The cap is a soft loss penalty on critic input gradients, not a hard bound on parameter gradients. The L2 norm is not divided by input dimension.
- Corrected joint-host candidates use both generator and encoder adversarial terms. Earlier incompatible receipts preserve their original source and verdicts.
- BCAP with K3P retains its additional K3P training mechanisms and is reported as a separate formulation.
- The opt-in per-offset DualNorm convolution extension completes the four previously unsupported image tasks, but all four sustained gates fail. This separate source-bound diagnostic retains constant learning rates and does not replace the selected configuration or original Tier 2 setup errors. See [the convolution readout](../bcap-convolution/README.md).

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/gan_loss.py](../../../particlegan/gan_loss.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/recipe_schedules.py](../../../particlegan/recipe_schedules.py)
- [configs/forge/ideas/bcap-pure-adam-v2.json](../../../configs/forge/ideas/bcap-pure-adam-v2.json)
- [reports/forge/pure-bcap/README.md](../pure-bcap/README.md)
- [particlegan/optim/dualnorm.py](../../../particlegan/optim/dualnorm.py)
- [docs/dualnorm-convolution.md](../../../docs/dualnorm-convolution.md)
- [reports/forge/bcap-convolution/README.md](../bcap-convolution/README.md)
- [reports/forge/bcap-convolution/readout.json](../bcap-convolution/readout.json)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Configuration detail for [BCAP](bcap-pure.md).** Its optimizer or settings do not create a separate solution family. This page preserves the original configuration evidence and diagnostics.

## Performance

DCGAN on deterministic mock RGB data measures finite-update speed and memory. These separate performance-source receipts confer no image-quality, scientific qualification or default-promotion credit. Standard BatchNorm is retained in generator and critic; critic input gradients therefore include batch coupling. The shared critic uses a flattened 4×4 Linear scalar head for policy feature extraction; its module layout is fixed across recipes.

Synchronized wall time for one public trainer update, including its critic/generator/prior work; warmup, construction, mock-data generation, evaluation, GIFs and checkpoint writes are excluded.

Every cell is one run; timings are never pooled across devices or contention states. Compare only rows with the same task and matched key (source, runtime/dtype, physical device, initialization and batch streams). Unknown or contended observations are unpaired. Resolution progression changes the task, while each candidate keeps one global recipe. Images/s is B divided by full update time (one D and one G/prior update with separate real batches). Three warmup and ten measured updates give short-window median/p95 measurements; long-horizon policy costs and component phase times remain unknown.

[Compact performance evidence index](../scaling-performance.json) · [Protocol and bounded reproduction](../../../docs/forge-scaling-performance.md)

This registration covers 10 complete recipes. Historical alternatives outside its registered family roster add no missing benchmark cells. A changed selected recipe within a registered family remains unknown until measured.

No current selected-recipe update timings are measured. Missing timing cells remain **unknown**.

<a name="cohort-cuda-1bf9d7d34422"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [bcap-pure-adam-v2](../../../configs/forge/ideas/bcap-pure-adam-v2.json).

Recorded qualification: **tier 0**, discriminator_stability revision 9. Other view rows below are navigation over recorded task evidence, not recomputed qualification. Their counts follow the declared view revisions (adaptation 4, clockfree_continuous 5, discriminator_stability 17, formulation_comparison 3, host_profile_transfer 6, quality_coverage 4); the recorded qualification keeps discriminator_stability revision 9 until new ordinary evidence is registered.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `cbb19c5e55e93aff93abd4092bbbe79e10c03a8b9f65a3f5db3b97f0f557d1ec`. Candidate revision: `806ae1707bf9ca246f21fc3a4a11eb8b536dc3002bf5c1febbbb166ad141dfcc`. Runtime cohort: `d5b60bbfb8572dae1be7de3d1c951499ca7cec4e91824b8154dfcd5257d13510`.

[Frozen numerical evidence](../technique-evidence/2880110ed49454a452197b2e42adafed37447bb43d288340299599bc8edc1025.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: archived_base_presentation_only. Exact original declaration-only solution base for navigation; the selected scientific leader remains separately pinned.

</details>

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/3 (3 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-4) | [0/16 (16 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-held) |
| [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/4 (4 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/6 (6 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-4) | [0/22 (22 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-held) |
| [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (11 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/18 (18 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | [0/30 (30 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/28 (28 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-held) |
| [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/3 (3 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-4) | [0/16 (16 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-held) |
| [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/3 (3 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-4) | [0/16 (16 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-held) |
| [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/3 (3 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-4) | [0/15 (15 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [lazy_cadence_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-2) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-4) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage) |
| [tier1_policy_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-1bf9d7d34422-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | UNKNOWN | changed since run |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | UNKNOWN | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | changed since run |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | changed since run |
| [model_ae_gan_multimodal](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | changed since run |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | UNKNOWN | changed since run |
| [zero_tanh_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [model_ae_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | changed since run |
| [rotated100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | [adaptation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | UNKNOWN | recorded definition unavailable |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN |

<a name="cohort-cuda-1bf9d7d34422-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_mlp16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | Can 12 zero-sigma ParticlePrior locations learn a 270-degree circular arc without MoG so unseen latent k-NN chords stay on the arc without snapping? | Consider counted status only after independent calibration and one global recipe that passes chord interpolation while retaining every counted pass. |
| [precision_gaussian_mxfp8](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | Can 16 zero-sigma ParticlePrior codes learn continuous 8x8 single-patch images without MoG so unseen latent chords shift one bright patch instead of 0.5 ghosts? | Consider counted status only after independent calibration and one global recipe that passes image latent interpolation while retaining every counted pass. |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | Can 24 zero-sigma ParticlePrior locations learn a curved 3D saddle sheet without MoG so unseen 2D latent triangle barycenters stay on the curved surface? | Consider counted status only after independent calibration and one global recipe that passes 2D simplex interpolation while retaining every counted pass. |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-1bf9d7d34422-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

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
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | diagnostic | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | diagnostic | UNKNOWN | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

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
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |

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
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |

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
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |

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
| [img_stripes2_w64_optimizer_diagnostic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-4"></a>

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
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | changed since run |
| [grid100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [dcgan_performance_r32](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [dcgan_performance_cpu_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [ddgan_repair_gaussian_chain_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | required | UNKNOWN | changed since run |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | required | UNKNOWN | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | required | UNKNOWN | changed since run |
| [zero_tanh_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | changed since run |
| [precision_gaussian_fp32](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_fp32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | required | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | required | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [model_ae_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [model_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | changed since run |
| [model_ae_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | changed since run |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [alice_spiral_bigan](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | changed since run |
| [img_intensity2_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | changed since run |

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
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | changed since run |
| [img_intensity2_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | changed since run |

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
| [two_pole_800_schedule80_diagnostic_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | diagnostic | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | diagnostic | UNKNOWN | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | diagnostic | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | diagnostic | UNKNOWN | changed since run |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | changed since run |

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
| [model_misgan_four_mode](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | diagnostic | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | diagnostic | UNKNOWN | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [zero_tanh_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_reconstruction) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | changed since run |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |
| [model_ae_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | diagnostic | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | diagnostic | UNKNOWN | changed since run |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | changed since run |

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
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |

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
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

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
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

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
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

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
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

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
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |

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

**ae_gan_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-1).

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

Used by: [ddgan-conditional-repair-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-ddgan-conditional-repair-v1-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**five_word_joint_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**five_word_joint_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**gaussian1d_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**gaussian1d_stability: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

**img_bars4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [bispf_fastpath_quality / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-1).

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

**img_blobs4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

**img_intensity2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**img_stripes2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [bispf_fastpath_quality / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bispf_fastpath_quality-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1).

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

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [particle_interpolation_stretch / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-particle_interpolation_stretch-tier-1).

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

**mid_scale_identity: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

**mode_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [misgan-bispf-research-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-misgan-bispf-research-v1-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-vaegan_coverage-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

**residual_student: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**ring16_acquisition: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

**rotated100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**staggered100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Execution: **not run**.

Used by: [adaptation / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) · [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**two_pole: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**unipolar: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**unused_token_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**vector_anisotropic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_spiral: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_two_broad: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_unequal_width: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

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

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-1bf9d7d34422-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |


Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-0d83d78027c5"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [bcap-pure · 5ea5bdbb2d71](../../../configs/forge/configurations/bcap-pure--5ea5bdbb2d71403dd316e201a51fb2b2b9c1868a8e053156b21b7cecb344d4be.json).

Recorded qualification: **tier 0**, discriminator_stability revision 9. Other view rows below are navigation over recorded task evidence, not recomputed qualification. Their counts follow the declared view revisions (adaptation 4, clockfree_continuous 5, discriminator_stability 17, formulation_comparison 3, host_profile_transfer 6, quality_coverage 4); the recorded qualification keeps discriminator_stability revision 9 until new ordinary evidence is registered.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `eede2a3a5780805489664b7354e8748a4020fa8eb5023aa6cad50c8e3f9d05ac`. Candidate revision: `9139a994e2a88403955e5bc8a0ce95be8c2cd5be8f0b2fad0652350d24b6971c`. Runtime cohort: `38c2e05b80ccb1949cd578d6ba808da0afc90b199e22d39e9817b2e6abdc0d89`.

[Frozen numerical evidence](../technique-evidence/029cb9bf5e422826f1cf0243f5ebfc0bf3e1e342292e592bae90dc00ac3f70c8.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Initial finite Pure BCAP round: required Tier 1 PASS count descending, then configuration hash ascending. One complete recipe; no calibrated default adoption.

</details>

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation) | [3/3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-4) | [3/16 (13 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-held) |
| [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [4/4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0/6 (6 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-4) | [4/22 (18 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-held) |
| [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [3/11 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0/18 (18 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | [3/30 (26 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [0/28 (28 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-held) |
| [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [3/3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-4) | [3/16 (13 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-held) |
| [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [3/3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0/1 (1 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-4) | [3/16 (13 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-held) |
| [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage) | [3/3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0/12 (12 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-4) | [3/15 (12 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [lazy_cadence_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-2) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-4) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage) |
| [tier1_policy_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0/0](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | PASS | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_gan_multimodal) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_reconstruction) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_categorical) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_hard) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | changed since run |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [zero_tanh_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_smoke) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_sparse_arc) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_mxfp8) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_nvfp4) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [model_ae_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_img_patch_shift) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_saddle_barycentric) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | changed since run |
| [rotated100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_rail_escape) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | UNKNOWN | recorded definition unavailable |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN |

<a name="cohort-cuda-0d83d78027c5-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-alice_spiral_bigan) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cgan_overlapping_classes) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cramer_energy_overlap) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_mlp16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_mlp16) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_res16_tanh) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-infogan_anisotropic_modes) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_neural_imputer) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_recovery) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-pacgan8_grid100) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-pacgan_unequal_mass) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_bigan_continuous) | [discriminator_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_sparse_arc) | [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | Can 12 zero-sigma ParticlePrior locations learn a 270-degree circular arc without MoG so unseen latent k-NN chords stay on the arc without snapping? | Consider counted status only after independent calibration and one global recipe that passes chord interpolation while retaining every counted pass. |
| [precision_gaussian_mxfp8](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_img_patch_shift) | [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | Can 16 zero-sigma ParticlePrior codes learn continuous 8x8 single-patch images without MoG so unseen latent chords shift one bright patch instead of 0.5 ghosts? | Consider counted status only after independent calibration and one global recipe that passes image latent interpolation while retaining every counted pass. |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_saddle_barycentric) | [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | Can 24 zero-sigma ParticlePrior locations learn a curved 3D saddle sheet without MoG so unseen 2D latent triangle barycenters stay on the curved surface? | Consider counted status only after independent calibration and one global recipe that passes 2D simplex interpolation while retaining every counted pass. |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | diagnostic | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | diagnostic | FAIL | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | diagnostic | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | diagnostic | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap-tier1-stability-repairs-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bispf_fastpath_quality"></a>

## bispf_fastpath_quality

**bispf_fastpath_quality — revision 1**. [View declaration](../../../configs/forge/views/bispf_fastpath_quality.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-bispf_fastpath_quality-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-bispf_fastpath_quality-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 5**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **4 / 12 / 6 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | changed since run |
| [grid100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-dcgan_performance_cpu_smoke-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1"></a>

## ddgan-conditional-repair-v1

**ddgan-conditional-repair-v1 — revision 2**. [View declaration](../../../configs/forge/views/ddgan-conditional-repair-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 17**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **11 / 18 / 1 / 0**. Held counts (measured and shown, not counted): **3 / 9 / 1 / 15**. Its highest tier, Tier 4, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | required | UNKNOWN | recorded definition unavailable |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | FAIL | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [zero_tanh_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_smoke) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_gan_multimodal) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_reconstruction) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_categorical) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_hard) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | changed since run |
| [precision_gaussian_fp32](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_fp32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_sparse_arc) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [model_ae_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [model_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | changed since run |
| [model_ae_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_img_patch_shift) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_saddle_barycentric) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | changed since run |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [alice_spiral_bigan](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | changed since run |
| [img_intensity2_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 6**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | changed since run |
| [img_intensity2_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-lazy_cadence_coverage"></a>

## lazy_cadence_coverage

**lazy_cadence_coverage — revision 1**. [View declaration](../../../configs/forge/views/lazy_cadence_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | diagnostic | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | diagnostic | FAIL | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | diagnostic | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | diagnostic | UNKNOWN | changed since run |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-misgan-bispf-research-v1"></a>

## misgan-bispf-research-v1

**misgan-bispf-research-v1 — revision 2**. [View declaration](../../../configs/forge/views/misgan-bispf-research-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Scoped research diagnostic; no ordinary qualification or default adoption.

<a name="cohort-cuda-0d83d78027c5-misgan-bispf-research-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_misgan_four_mode](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-misgan-bispf-research-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-misgan-bispf-research-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-misgan-bispf-research-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-particle_interpolation_stretch"></a>

## particle_interpolation_stretch

**particle_interpolation_stretch — revision 1**. [View declaration](../../../configs/forge/views/particle_interpolation_stretch.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Three zero-sigma particle interpolation stretch toys do not establish whole-suite screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_sparse_arc) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_img_patch_shift) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-interp_saddle_barycentric) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1"></a>

## prior-ablation-full-suite-v1

**prior-ablation-full-suite-v1 — revision 1**. [View declaration](../../../configs/forge/views/prior-ablation-full-suite-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | diagnostic | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | diagnostic | FAIL | changed since run |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [zero_tanh_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_gan_multimodal) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_reconstruction) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | changed since run |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |
| [model_ae_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | diagnostic | UNKNOWN | changed since run |
| [ring_extension](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | diagnostic | UNKNOWN | changed since run |
| [zero_tanh_rail_escape](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-zero_tanh_rail_escape) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-misgan_block_neural_imputer) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_bigan_continuous) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 4**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 0 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022"></a>

## r1r2-repair-word-retention-20261022

**r1r2-repair-word-retention-20261022 — revision 1**. [View declaration](../../../configs/forge/views/r1r2-repair-word-retention-20261022.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

<a name="cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-selected-family-coverage-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **7 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-vaegan_coverage"></a>

## vaegan_coverage

**vaegan_coverage — revision 1**. [View declaration](../../../configs/forge/views/vaegan_coverage.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Two toy gates do not establish screen calibration or whole-suite adoption.

<a name="cohort-cuda-0d83d78027c5-vaegan_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_vae_hard](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-vaegan_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-vaegan_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-vaegan_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-0d83d78027c5-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.0237108 | <= 0.35 | PASS |
| recon_mse | 0.00691841 | <= 0.05 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d342f0ff8c4d421aa26360478dbf7680.json)

[Actual-training GIF](../pure-bcap/media/5ea5bdbb2d71/ae_gan_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

<a name="cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-alice_spiral_bigan"></a>

### alice_spiral_bigan

**alice_spiral_bigan: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/alice_spiral_bigan.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-cgan_overlapping_classes"></a>

### cgan_overlapping_classes

**cgan_overlapping_classes: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cgan_overlapping_classes.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/288936eb39174fd3bf07f60e97763a64.json)

[Actual-training GIF](../pure-bcap/media/5ea5bdbb2d71/clockfree_audit_measurement_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-cramer_energy_overlap"></a>

### cramer_energy_overlap

**cramer_energy_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cramer_energy_overlap.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-cyclegan_unpaired_constellation"></a>

### cyclegan_unpaired_constellation

**cyclegan_unpaired_constellation: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-ddgan_repair_gaussian_chain_v1"></a>

### ddgan_repair_gaussian_chain_v1

**ddgan_repair_gaussian_chain_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100"></a>

### grid100

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_mlp16"></a>

### grid100_mlp16

**grid100_mlp16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_mlp16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_res16_scaled_tanh"></a>

### grid100_res16_scaled_tanh

**grid100_res16_scaled_tanh: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_scaled_tanh.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_res16_tanh"></a>

### grid100_res16_tanh

**grid100_res16_tanh: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_tanh.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-img_bars4"></a>

### img_bars4

**img_bars4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_bars4_w64_optimizer_diagnostic"></a>

### img_bars4_w64_optimizer_diagnostic

**img_bars4_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_stripes2_w64_optimizer_diagnostic"></a>

### img_stripes2_w64_optimizer_diagnostic

**img_stripes2_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-infogan_anisotropic_modes"></a>

### infogan_anisotropic_modes

**infogan_anisotropic_modes: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/infogan_anisotropic_modes.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-interp_img_patch_shift"></a>

### interp_img_patch_shift

**interp_img_patch_shift: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_img_patch_shift.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-interp_saddle_barycentric"></a>

### interp_saddle_barycentric

**interp_saddle_barycentric: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_saddle_barycentric.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-interp_sparse_arc"></a>

### interp_sparse_arc

**interp_sparse_arc: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_sparse_arc.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [particle_interpolation_stretch / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-misgan_block_neural_imputer"></a>

### misgan_block_neural_imputer

**misgan_block_neural_imputer: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_neural_imputer.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-misgan_block_recovery"></a>

### misgan_block_recovery

**misgan_block_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p50_recovery"></a>

### misgan_mcar_p50_recovery

**misgan_mcar_p50_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-misgan_mcar_p80_anchored"></a>

### misgan_mcar_p80_anchored

**misgan_mcar_p80_anchored: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-mode_hold"></a>

### mode_hold

**mode_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_bimodal_chain"></a>

### model_ae_ddgan_bimodal_chain

**model_ae_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_ae_ddgan_gaussian_chain"></a>

### model_ae_ddgan_gaussian_chain

**model_ae_ddgan_gaussian_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_ae_gan_multimodal"></a>

### model_ae_gan_multimodal

**model_ae_gan_multimodal: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_gan_multimodal.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_ae_reconstruction"></a>

### model_ae_reconstruction

**model_ae_reconstruction: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_reconstruction.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_ddgan_bimodal_chain"></a>

### model_ddgan_bimodal_chain

**model_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_ddgan_gaussian_chain"></a>

### model_ddgan_gaussian_chain

**model_ddgan_gaussian_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_misgan_four_mode"></a>

### model_misgan_four_mode

**model_misgan_four_mode: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_misgan_four_mode.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [misgan-bispf-research-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-misgan-bispf-research-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_vae_categorical"></a>

### model_vae_categorical

**model_vae_categorical: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_categorical.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-vaegan_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-model_vae_hard"></a>

### model_vae_hard

**model_vae_hard: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_hard.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-vaegan_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-pacgan8_grid100"></a>

### pacgan8_grid100

**pacgan8_grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/pacgan8_grid100.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 95 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| min_cov_eig_ratio | >= 0.15 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-pacgan_unequal_mass"></a>

### pacgan_unequal_mass

**pacgan_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/pacgan_unequal_mass.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4).

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

<a name="cohort-cuda-0d83d78027c5-experiment-precision_gaussian_fp32"></a>

### precision_gaussian_fp32

**precision_gaussian_fp32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_fp32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-precision_gaussian_mxfp8"></a>

### precision_gaussian_mxfp8

**precision_gaussian_mxfp8: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_mxfp8.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-precision_gaussian_nvfp4"></a>

### precision_gaussian_nvfp4

**precision_gaussian_nvfp4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-residual_student"></a>

### residual_student

**residual_student: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 5.36652 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.34812 | >= 0.15 | PASS |
| hq | 0.822754 | >= 0.85 | FAIL |
| mass_tv | 0.0998535 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/aae235aaaaa64ac2b5e36c18f4af3155.json)

[Actual-training GIF](../pure-bcap/media/5ea5bdbb2d71/ring16_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

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

<a name="cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-ring16_bigan_continuous"></a>

### ring16_bigan_continuous

**ring16_bigan_continuous: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring16_bigan_continuous.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-rotated100"></a>

### rotated100

**rotated100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-staggered100"></a>

### staggered100

**staggered100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [adaptation / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.344867 | <= 1 | PASS |
| mean_abs | 0.421174 | >= 0.3 | PASS |

Recorded terminal passing observations: **5**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d295b50685f34011822aa9836946482d.json)

[Actual-training GIF](../pure-bcap/media/5ea5bdbb2d71/two_pole.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-unipolar"></a>

### unipolar

**unipolar: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.983446 | >= 0.85 | PASS |
| unused_hold | 0.988603 | >= 0.85 | PASS |

Recorded terminal passing observations: **15**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/aa60b1572b3f403ca23204b47806a5e1.json)

[Actual-training GIF](../pure-bcap/media/5ea5bdbb2d71/unused_token_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

<a name="cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-zero_tanh_rail_escape"></a>

### zero_tanh_rail_escape

**zero_tanh_rail_escape: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_rail_escape.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 3](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-zero_tanh_smoke"></a>

### zero_tanh_smoke

**zero_tanh_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](bcap-pure-configuration.md#cohort-cuda-0d83d78027c5-prior-ablation-full-suite-v1-tier-1).

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

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base optimizer. Repository-specific guards and damping are separate mechanisms.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic logistic objective; this implementation pairs scores, rather than subtracting batch-average scores.
- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894). Related work on one-sided gradient penalties. The repository BCAP kernel uses real/fake inputs directly and is not a reproduction of this paper or its sampling scheme.
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Non-saturating logistic loss option.
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875). Wasserstein score-loss option. This BCAP recipe does not adopt the paper’s weight clipping or full training algorithm.
- [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076). Least-squares loss option; repository targets are real=1, fake=0 and generator=1.
- [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957). Reference for the hinge adversarial objective option; selecting hinge does not itself enable spectral normalization.
