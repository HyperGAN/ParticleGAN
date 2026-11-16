<!-- Generated Forge family report -->

# GAN v3 release 0.7 (cloud)

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [historical-cohort](../technique-inventory.md#tag-historical-cohort) · [learning-rate-annealing](../technique-inventory.md#tag-learning-rate-annealing)

## Technique overview

This historical entry retains the full v0.7 GAN v3 recipe on its explicit native particle-cloud diagnostic: latent dimension 4, batch 256 and an unstandardized learned ParticlePrior. Fixed real/fake input-gradient caps and latent spread regularization remain. The cloud sampling and named MLP initialization are experiment-owned and cannot be treated as the learned-MoG cohort.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $R_{\mathrm{prior}}$ is the VICReg-inspired latent spread term, $d_z$ the latent dimension, $\Sigma_z$ the unbiased sample covariance and $\epsilon=10^{-4}$ the variance stabilizer. $t$ counts completed updates, $T$ is the declared recipe horizon, $q=t/T$, $\eta^{(0)}$ a base LR and $a(q)$ its multiplier. EMA denotes a parameter exponential moving average.

**Default paired adversarial loss**

$$
\begin{aligned}\ell_D&=\mathbb E\!\left[\operatorname{softplus}(C(y)-C(x))\right],\\\ell_G&=\mathbb E\!\left[\operatorname{softplus}(C(x)-C(y))\right].\end{aligned}
$$

These scalar losses are minimized. The critic objective is $\ell_D+R$; the generator adds only its declared auxiliary objectives. Critic fakes are detached, and generator fakes and both scores are recomputed after the critic step. $C$ includes any declared fresh input noise.

**Fixed capped input-gradient penalty**

$$
R_{\mathrm{BCAP}}=\frac{\lambda}{2}\left\{\mathbb E\!\left[(\lVert g_r\rVert_2-\kappa)_+^2\right]+\mathbb E\!\left[(\lVert g_f\rVert_2-\kappa)_+^2\right]\right\}.
$$

The full release recipe fixes $\lambda=6$ and $\kappa=1.25$, applied every critic update. This is the same soft L2 cap mechanism with stronger and wider settings; K3P handover, critic proximity and optimizer interventions are disabled.

**Latent spread auxiliary objective**

$$
\begin{aligned}R_{\mathrm{prior}}&=\frac{1}{d_z}\sum_{a=1}^{d_z}\left(1-\sqrt{(\Sigma_z)_{aa}+10^{-4}}\right)_+\\&\quad+\frac{1}{d_z}\sum_{a\ne b}(\Sigma_z)_{ab}^2,\\L_G&=\ell_G+0.05R_{\mathrm{prior}}.\end{aligned}
$$

For scalar hosts that expose this auxiliary, use the unbiased sample covariance of the supplied latent batch. It discourages standard deviations below $1$ and off-diagonal covariance, without requiring a Gaussian topology. Batches with fewer than two rows return zero. Behavioral hosts retain their explicitly bound auxiliary objectives.

**Full-horizon learning-rate cosine**

$$
a(q)=\begin{cases}1,&q\le0.6,\\0.05+\dfrac{0.95}{2}\left[1+\cos\!\left(\pi\min\!\left(1,\dfrac{q-0.6}{0.4}\right)\right)\right],&q>0.6,\end{cases}\qquad \eta(t)=\eta^{(0)}a(t/T).
$$

Hold each role at its base rate for $60\%$ of the declared recipe horizon, then decay to a $5\%$ floor and hold. The reusable task-adapted card resolves $T$ explicitly; the historical native MoG/cloud cards retain $T=7000$. An external execution cap does not redefine the schedule.

## Simplified pseudocode

```text
For each training iteration:
  Use the historical 7000-step recipe horizon: hold LR through 60%, then cosine-decay toward 5% of its base value.
  Draw real x and task-prior z; fake = detach(G(z)).
  Compute the paired critic adversarial loss shown above.
  g_r = gradient(D(x),x); g_f = gradient(D(fake),fake), using detached input copies.
  Add the fixed real/fake input-gradient penalty shown above.
  Backpropagate; take the critic Adam step with guard and gradient anchor disabled.
  Freeze D parameters; draw fresh z, regenerate fake and recompute both scores.
  Compute the paired generator adversarial loss from the updated critic.
  On scalar latent-table hosts, add .05 * R_prior to L_G; behavioral hosts retain their original objectives.
  Backpropagate; update G/prior with A2 and direct response gain disabled.
  Maintain the configured .995 generator/prior EMA where the host implements it.
  Score with the frozen task's declared weights, prior and noise law; ordinary gates use clean live outputs.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic, with the fixed real/fake cap regularizer below. Scalar hosts add the declared $0.05$ latent variance/covariance term; behavioral hosts keep their task-owned reconstruction, conditional or direct-particle objectives. |
| Optimizer | K3P-derived Adam wrappers selected by the explicit fixed b_cap arm. $\beta_1=0$, $\beta_2=0.99$, AMSGrad off; guard, anchor, A2 and direct gain are disabled. Observer bookkeeping remains. This recipe is not selected through optimizer_family='adam', despite having its intervention switches off. Direct sample-particle hosts can retain separately declared response moments even when response gain is disabled; inspect the host binding. |
| Learning rates and annealing | Historical native reference LR $0.00425$, critic multiplier $1$ and latent-prior multiplier $2$. All roles follow the $7000$-step full-horizon cosine: $60\%$ hold, then decay toward a $5\%$ floor. The diagnostic's host resource/horizon binding is part of its identity. |
| Parameter-gradient clipping | No critic spike guard or global gradient-norm clipping. The cap penalizes excess critic input-gradient magnitude in the loss; it does not clamp critic parameter gradients or weights. |
| Critic penalties and anchors | Fixed $\lambda=6$ and L2 cap $\kappa=1.25$ on both real and fake samples, applied every step. No RMS normalization, K3P handover or critic-gradient proximity term. |
| Damping and update guards | A2, critic spike guard, critic anchor and direct sample-particle gain are all disabled. This distinguishes the full release recipe from a fixed BCap penalty swapped into an otherwise intact K3P recipe. |
| Training and sampling noise | No critic-input or generated-output training noise. The explicit cloud diagnostic uses ParticlePrior with $\sigma=0$ and no standardization. The recipe's generic prior_kind field is not sufficient to identify the prior: the task's declared particle-cloud exception governs this diagnostic. |
| Parameter averaging and serving | Generator/prior EMA decay $0.995$ is configured where the host implements it; selected ordinary gates use live weights. Behavioral hosts may own no scored EMA or latent table. No critic-gradient anchor average is active. |

## Configuration differences

- Each result retains its executed source, recipe, task prior, initialization, budget and sampling law. These descriptions do not change or requalify recorded measurements.
- Task-owned objectives and active components matter: direct sample particles, learned latent rows and a generator network are different parameter roles. A declared recipe switch does not imply that every host can apply it.
- Historical candidate release07-gan-v3-cloud-v1 is retained at source 2899099048c0a9987eb8720214abfce56d86d92a. Its card fixes z_dim=4, batch_size=256, 20000 rows and a 7000-step reference horizon.
- grid100_release07_cloud_named_v1 explicitly binds a particle cloud and a named z4 MLP host with Xavier-uniform/zero-bias network initialization. This is separate from the generic ordinary-view MoG bindings appearing in historical inventory projections.
- The native cloud diagnostic retains its own clean/live sampling, full update budget and numerical gates. Its failure or any separate historical served/noisy success transfers no learned-MoG or current-family credit.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/release07-gan-v3-cloud-v1.json](../../../configs/forge/ideas/release07-gan-v3-cloud-v1.json)
- [configs/forge/tasks/grid100_release07_cloud_named_v1.json](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json)
- [configs/toy100/release07_public_default_host.json](../../../configs/toy100/release07_public_default_host.json)
- [reports/forge/studies/FORMULATION_COMPARISON_CONTINUATION.md](../studies/FORMULATION_COMPARISON_CONTINUATION.md)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/vicreg_loss.py](../../../particlegan/vicreg_loss.py)
- [https://github.com/255BITS/ParticleGAN/blob/2899099048c0a9987eb8720214abfce56d86d92a/particlegan/recipes.py](https://github.com/255BITS/ParticleGAN/blob/2899099048c0a9987eb8720214abfce56d86d92a/particlegan/recipes.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Historical cohort navigation.** This original recipe/prior row is retained for existing links. [Current GAN v3 solution family](release07-gan-v3.md) uses one whole selected configuration and task-declared priors; these historical cells are not pooled into it.

<a name="cohort-cuda-7f9c23eb0e27"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [release07-gan-v3-cloud-v1](../../../configs/forge/ideas/release07-gan-v3-cloud-v1.json).

Recorded qualification: **tier 0**, discriminator_stability revision 9. Other view rows below are navigation over recorded task evidence, not recomputed qualification. Their counts follow the declared view revisions (adaptation 4, clockfree_continuous 5, discriminator_stability 17, formulation_comparison 3, host_profile_transfer 6, quality_coverage 4); the recorded qualification keeps discriminator_stability revision 9 until new ordinary evidence is registered.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `7306340bac0a4ea67ea7b080513116457b7d72a8cb0d0c1f0727eaae6db38185`. Candidate revision: `95e403581ebf0d72da9e36b0bcc668ec4512daa4e1aafe06013b465b7a8b06da`. Runtime cohort: `6cb4d268b94e96566fca7784d03bfe6e207700c448dba2471de72d51acadbe12`.

[Frozen numerical evidence](../technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Retain the exact recorded incumbent; its outcomes are historical best observed evidence where a registered search exists. Alternatives from another source or runtime are unranked.

</details>

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation) | [0/3 (3 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) | [0/12 (12 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) | [0/1 (1 not run)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-4) | [0/16 (1 not run, 15 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation) | [0/7 (7 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-held) |
| [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous) | [0/4 (1 not run, 3 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) | [0/12 (12 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) | [0/6 (5 not run, 1 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-4) | [0/22 (6 not run, 16 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous) | [0/7 (7 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-held) |
| [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability) | [0/11 (7 not run, 4 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | [0/18 (6 not run, 12 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | [0/1 (1 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | [0/30 (13 not run, 17 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability) | [0/28 (21 not run, 7 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-held) |
| [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison) | [0/3 (3 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) | [0/12 (12 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) | [0/1 (1 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-4) | [0/16 (16 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison) | [0/7 (7 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-held) |
| [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer) | [0/3 (3 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) | [0/12 (12 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) | [0/1 (1 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-4) | [0/16 (16 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer) | [0/7 (7 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-held) |
| [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage) | [0/3 (3 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | [0/12 (12 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-4) | [0/15 (15 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage) | [0/7 (7 incompatible)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [lazy_cadence_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-2) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-4) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage) |
| [tier1_policy_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage) | [0/7 (7 not run)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-2) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-3) | [0/0](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-7f9c23eb0e27-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | BLOCKED | changed since run |
| [clockfree_audit_measurement_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) | UNKNOWN | recorded definition unavailable |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_gan_multimodal) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_reconstruction) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_categorical) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_hard) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | BLOCKED | changed since run |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | BLOCKED | changed since run |
| [zero_tanh_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_smoke) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_sparse_arc) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_mxfp8) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_nvfp4) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [model_ae_ddgan_bimodal_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [interp_img_patch_shift](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_img_patch_shift) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_saddle_barycentric) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | BLOCKED | changed since run |
| [rotated100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | [adaptation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3), [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_rail_escape) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) | UNKNOWN | recorded definition unavailable |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [clockfree_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3), [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3), [formulation_comparison](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3), [host_profile_transfer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | BLOCKED |

<a name="cohort-cuda-7f9c23eb0e27-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-alice_spiral_bigan) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cgan_overlapping_classes) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cramer_energy_overlap) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_mlp16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_mlp16) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_tanh) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-infogan_anisotropic_modes) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_neural_imputer) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_recovery) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-pacgan8_grid100) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-pacgan_unequal_mass) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_bigan_continuous) | [discriminator_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_sparse_arc) | [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | Can 12 zero-sigma ParticlePrior locations learn a 270-degree circular arc without MoG so unseen latent k-NN chords stay on the arc without snapping? | Consider counted status only after independent calibration and one global recipe that passes chord interpolation while retaining every counted pass. |
| [precision_gaussian_mxfp8](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [interp_img_patch_shift](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_img_patch_shift) | [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | Can 16 zero-sigma ParticlePrior codes learn continuous 8x8 single-patch images without MoG so unseen latent chords shift one bright patch instead of 0.5 ghosts? | Consider counted status only after independent calibration and one global recipe that passes image latent interpolation while retaining every counted pass. |
| [interp_saddle_barycentric](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_saddle_barycentric) | [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | Can 24 zero-sigma ParticlePrior locations learn a curved 3D saddle sheet without MoG so unseen 2D latent triangle barycenters stay on the curved surface? | Consider counted status only after independent calibration and one global recipe that passes 2D simplex interpolation while retaining every counted pass. |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-7f9c23eb0e27-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | diagnostic | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | diagnostic | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | diagnostic | BLOCKED | changed since run |
| [ring16_acquisition](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | diagnostic | BLOCKED | changed since run |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-repairs-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality"></a>

## bispf_fastpath_quality

**bispf_fastpath_quality — revision 1**. [View declaration](../../../configs/forge/views/bispf_fastpath_quality.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 5**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **4 / 12 / 6 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | changed since run |
| [clockfree_audit_measurement_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | BLOCKED | changed since run |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | changed since run |
| [grid100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-dcgan_performance_cpu_smoke-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1"></a>

## ddgan-conditional-repair-v1

**ddgan-conditional-repair-v1 — revision 2**. [View declaration](../../../configs/forge/views/ddgan-conditional-repair-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 17**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **11 / 18 / 1 / 0**. Held counts (measured and shown, not counted): **3 / 9 / 1 / 15**. Its highest tier, Tier 4, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | required | UNKNOWN | recorded definition unavailable |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | changed since run |
| [ring16_acquisition](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | required | BLOCKED | changed since run |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [zero_tanh_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_smoke) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_gan_multimodal) | required | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_reconstruction) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_categorical) | required | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_hard) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_fp32](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_fp32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_mxfp8](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [precision_gaussian_nvfp4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | UNKNOWN | recorded definition unavailable |
| [interp_sparse_arc](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_sparse_arc) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_sparse_arc) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | changed since run |
| [model_ae_ddgan_gaussian_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | BLOCKED | changed since run |
| [model_ddgan_gaussian_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_gaussian_chain) | required | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | changed since run |
| [model_ae_ddgan_bimodal_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_img_patch_shift) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_img_patch_shift) | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_saddle_barycentric) | [stretch: frontier stretch benchmark beyond the required suite](../particle-interpolation-stretch/README.md#interp_saddle_barycentric) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | BLOCKED | changed since run |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | changed since run |
| [zero_tanh_rail_escape](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_tanh](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p50_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [misgan_mcar_p80_anchored](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [pacgan8_grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [alice_spiral_bigan](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [infogan_anisotropic_modes](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cgan_overlapping_classes](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cramer_energy_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [cyclegan_unpaired_constellation](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |
| [grid100_res16_scaled_tanh](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | changed since run |
| [img_intensity2_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | BLOCKED | changed since run |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 6**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | changed since run |
| [img_intensity2_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | BLOCKED | changed since run |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage"></a>

## lazy_cadence_coverage

**lazy_cadence_coverage — revision 1**. [View declaration](../../../configs/forge/views/lazy_cadence_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | diagnostic | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | diagnostic | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | diagnostic | BLOCKED | changed since run |
| [ring16_acquisition](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | diagnostic | BLOCKED | changed since run |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | diagnostic | BLOCKED | changed since run |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | diagnostic | BLOCKED | changed since run |
| [clockfree_audit_measurement_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-misgan-bispf-research-v1"></a>

## misgan-bispf-research-v1

**misgan-bispf-research-v1 — revision 2**. [View declaration](../../../configs/forge/views/misgan-bispf-research-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Scoped research diagnostic; no ordinary qualification or default adoption.

<a name="cohort-cuda-7f9c23eb0e27-misgan-bispf-research-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_misgan_four_mode](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-misgan-bispf-research-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-misgan-bispf-research-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-misgan-bispf-research-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch"></a>

## particle_interpolation_stretch

**particle_interpolation_stretch — revision 1**. [View declaration](../../../configs/forge/views/particle_interpolation_stretch.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Three zero-sigma particle interpolation stretch toys do not establish whole-suite screen calibration or default adoption.

<a name="cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [interp_sparse_arc](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_sparse_arc) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_img_patch_shift](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_img_patch_shift) | diagnostic | UNKNOWN | recorded definition unavailable |
| [interp_saddle_barycentric](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-interp_saddle_barycentric) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1"></a>

## prior-ablation-full-suite-v1

**prior-ablation-full-suite-v1 — revision 1**. [View declaration](../../../configs/forge/views/prior-ablation-full-suite-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | diagnostic | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | diagnostic | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | diagnostic | BLOCKED | changed since run |
| [ring16_acquisition](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | diagnostic | BLOCKED | changed since run |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [zero_tanh_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_ddgan_gaussian_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_gan_multimodal](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_gan_multimodal) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ae_reconstruction](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_reconstruction) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_gaussian_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_gaussian_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_hard](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [clockfree_audit](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |
| [model_ae_ddgan_bimodal_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_ddgan_bimodal_chain](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | diagnostic | BLOCKED | changed since run |
| [ring_extension](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | diagnostic | BLOCKED | changed since run |
| [zero_tanh_rail_escape](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_rail_escape) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_neural_imputer](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-misgan_block_neural_imputer) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ring16_bigan_continuous](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_bigan_continuous) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 4**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 0 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022"></a>

## r1r2-repair-word-retention-20261022

**r1r2-repair-word-retention-20261022 — revision 1**. [View declaration](../../../configs/forge/views/r1r2-repair-word-retention-20261022.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

<a name="cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | diagnostic | BLOCKED | changed since run |
| [residual_student](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | diagnostic | BLOCKED | changed since run |
| [unipolar](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | diagnostic | BLOCKED | changed since run |
| [cover_leftover](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | diagnostic | BLOCKED | changed since run |
| [mid_scale_identity](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | diagnostic | BLOCKED | changed since run |
| [mode_hold](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | diagnostic | BLOCKED | changed since run |
| [vector_two_broad](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_mass](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | diagnostic | BLOCKED | changed since run |
| [vector_unequal_width](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | diagnostic | BLOCKED | changed since run |
| [vector_anisotropic](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | diagnostic | BLOCKED | changed since run |
| [vector_overlap](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | diagnostic | BLOCKED | changed since run |
| [vector_spiral](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | diagnostic | BLOCKED | changed since run |
| [img_stripes2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | diagnostic | BLOCKED | changed since run |
| [img_bars4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | diagnostic | BLOCKED | changed since run |
| [img_blobs4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | diagnostic | BLOCKED | changed since run |
| [img_intensity2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | diagnostic | BLOCKED | changed since run |
| [grid100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | diagnostic | BLOCKED | changed since run |
| [rotated100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | diagnostic | BLOCKED | changed since run |
| [staggered100](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | diagnostic | BLOCKED | changed since run |

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-selected-family-coverage-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **7 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-vaegan_coverage"></a>

## vaegan_coverage

**vaegan_coverage — revision 1**. [View declaration](../../../configs/forge/views/vaegan_coverage.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Two toy gates do not establish screen calibration or whole-suite adoption.

<a name="cohort-cuda-7f9c23eb0e27-vaegan_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_vae_hard](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_hard) | diagnostic | UNKNOWN | recorded definition unavailable |
| [model_vae_categorical](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-experiment-model_vae_categorical) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-7f9c23eb0e27-vaegan_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-vaegan_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-vaegan_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. ae_gan_hold: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'batch_size' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'conditioning' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'model' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'num_classes' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'num_particles' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'total_steps' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'z_dim' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'batch_size' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'conditioning' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'model' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'num_classes' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'num_particles' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'total_steps' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; ae_gan_hold: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-alice_spiral_bigan"></a>

### alice_spiral_bigan

**alice_spiral_bigan: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/alice_spiral_bigan.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-cgan_overlapping_classes"></a>

### cgan_overlapping_classes

**cgan_overlapping_classes: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cgan_overlapping_classes.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. cover_leftover: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'batch_size' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'conditioning' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'model' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'num_classes' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'num_particles' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'total_steps' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'z_dim' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'batch_size' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'conditioning' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'model' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'num_classes' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'num_particles' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'total_steps' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; cover_leftover: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-cramer_energy_overlap"></a>

### cramer_energy_overlap

**cramer_energy_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cramer_energy_overlap.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-cyclegan_unpaired_constellation"></a>

### cyclegan_unpaired_constellation

**cyclegan_unpaired_constellation: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-7f9c23eb0e27-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ddgan_repair_gaussian_chain_v1"></a>

### ddgan_repair_gaussian_chain_v1

**ddgan_repair_gaussian_chain_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-ddgan-conditional-repair-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100"></a>

### grid100

**grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_mlp16"></a>

### grid100_mlp16

**grid100_mlp16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_mlp16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_scaled_tanh"></a>

### grid100_res16_scaled_tanh

**grid100_res16_scaled_tanh: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_scaled_tanh.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_res16_tanh"></a>

### grid100_res16_tanh

**grid100_res16_tanh: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_tanh.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_bars4"></a>

### img_bars4

**img_bars4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_bars4_w64_optimizer_diagnostic"></a>

### img_bars4_w64_optimizer_diagnostic

**img_bars4_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_w64_optimizer_diagnostic"></a>

### img_stripes2_w64_optimizer_diagnostic

**img_stripes2_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-infogan_anisotropic_modes"></a>

### infogan_anisotropic_modes

**infogan_anisotropic_modes: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/infogan_anisotropic_modes.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-interp_img_patch_shift"></a>

### interp_img_patch_shift

**interp_img_patch_shift: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_img_patch_shift.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-interp_saddle_barycentric"></a>

### interp_saddle_barycentric

**interp_saddle_barycentric: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_saddle_barycentric.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [particle_interpolation_stretch / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-interp_sparse_arc"></a>

### interp_sparse_arc

**interp_sparse_arc: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/interp_sparse_arc.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [particle_interpolation_stretch / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-particle_interpolation_stretch-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. mid_scale_identity: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'batch_size' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'conditioning' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'model' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'num_classes' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'num_particles' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'total_steps' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'z_dim' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'batch_size' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'conditioning' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'model' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'num_classes' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'num_particles' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'total_steps' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; mid_scale_identity: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-misgan_block_neural_imputer"></a>

### misgan_block_neural_imputer

**misgan_block_neural_imputer: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_neural_imputer.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-misgan_block_recovery"></a>

### misgan_block_recovery

**misgan_block_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p50_recovery"></a>

### misgan_mcar_p50_recovery

**misgan_mcar_p50_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-misgan_mcar_p80_anchored"></a>

### misgan_mcar_p80_anchored

**misgan_mcar_p80_anchored: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-mode_hold"></a>

### mode_hold

**mode_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_bimodal_chain"></a>

### model_ae_ddgan_bimodal_chain

**model_ae_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_ae_ddgan_gaussian_chain"></a>

### model_ae_ddgan_gaussian_chain

**model_ae_ddgan_gaussian_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_ae_gan_multimodal"></a>

### model_ae_gan_multimodal

**model_ae_gan_multimodal: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_gan_multimodal.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_ae_reconstruction"></a>

### model_ae_reconstruction

**model_ae_reconstruction: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_reconstruction.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_bimodal_chain"></a>

### model_ddgan_bimodal_chain

**model_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_ddgan_gaussian_chain"></a>

### model_ddgan_gaussian_chain

**model_ddgan_gaussian_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_misgan_four_mode"></a>

### model_misgan_four_mode

**model_misgan_four_mode: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_misgan_four_mode.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [misgan-bispf-research-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-misgan-bispf-research-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_vae_categorical"></a>

### model_vae_categorical

**model_vae_categorical: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_categorical.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-vaegan_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-model_vae_hard"></a>

### model_vae_hard

**model_vae_hard: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_hard.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-vaegan_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-pacgan8_grid100"></a>

### pacgan8_grid100

**pacgan8_grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/pacgan8_grid100.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 95 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| min_cov_eig_ratio | >= 0.15 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-pacgan_unequal_mass"></a>

### pacgan_unequal_mass

**pacgan_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/pacgan_unequal_mass.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_fp32"></a>

### precision_gaussian_fp32

**precision_gaussian_fp32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_fp32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_mxfp8"></a>

### precision_gaussian_mxfp8

**precision_gaussian_mxfp8: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_mxfp8.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-precision_gaussian_nvfp4"></a>

### precision_gaussian_nvfp4

**precision_gaussian_nvfp4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-residual_student"></a>

### residual_student

**residual_student: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. residual_student: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; residual_student: recipe override 'batch_size' is owned by the frozen host; revise its task specification; residual_student: recipe override 'conditioning' is owned by the frozen host; revise its task specification; residual_student: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; residual_student: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; residual_student: recipe override 'model' is owned by the frozen host; revise its task specification; residual_student: recipe override 'num_classes' is owned by the frozen host; revise its task specification; residual_student: recipe override 'num_particles' is owned by the frozen host; revise its task specification; residual_student: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; residual_student: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; residual_student: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; residual_student: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; residual_student: recipe override 'total_steps' is owned by the frozen host; revise its task specification; residual_student: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; residual_student: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; residual_student: recipe override 'z_dim' is owned by the frozen host; revise its task specification; residual_student: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; residual_student: recipe override 'batch_size' is owned by the frozen host; revise its task specification; residual_student: recipe override 'conditioning' is owned by the frozen host; revise its task specification; residual_student: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; residual_student: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; residual_student: recipe override 'model' is owned by the frozen host; revise its task specification; residual_student: recipe override 'num_classes' is owned by the frozen host; revise its task specification; residual_student: recipe override 'num_particles' is owned by the frozen host; revise its task specification; residual_student: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; residual_student: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; residual_student: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; residual_student: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; residual_student: recipe override 'total_steps' is owned by the frozen host; revise its task specification; residual_student: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; residual_student: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; residual_student: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring16_bigan_continuous"></a>

### ring16_bigan_continuous

**ring16_bigan_continuous: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring16_bigan_continuous.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 4](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring_extension"></a>

### ring_extension

**ring_extension: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring_hold"></a>

### ring_hold

**ring_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-rotated100"></a>

### rotated100

**rotated100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-staggered100"></a>

### staggered100

**staggered100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource z_dim; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource z_dim; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [adaptation / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3) · [clockfree_continuous / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-7f9c23eb0e27-experiment-trajectory"></a>

### trajectory

**trajectory: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. trajectory: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; trajectory: recipe override 'batch_size' is owned by the frozen host; revise its task specification; trajectory: recipe override 'conditioning' is owned by the frozen host; revise its task specification; trajectory: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; trajectory: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; trajectory: recipe override 'model' is owned by the frozen host; revise its task specification; trajectory: recipe override 'num_classes' is owned by the frozen host; revise its task specification; trajectory: recipe override 'num_particles' is owned by the frozen host; revise its task specification; trajectory: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; trajectory: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; trajectory: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; trajectory: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; trajectory: recipe override 'total_steps' is owned by the frozen host; revise its task specification; trajectory: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; trajectory: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; trajectory: recipe override 'z_dim' is owned by the frozen host; revise its task specification; trajectory: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; trajectory: recipe override 'batch_size' is owned by the frozen host; revise its task specification; trajectory: recipe override 'conditioning' is owned by the frozen host; revise its task specification; trajectory: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; trajectory: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; trajectory: recipe override 'model' is owned by the frozen host; revise its task specification; trajectory: recipe override 'num_classes' is owned by the frozen host; revise its task specification; trajectory: recipe override 'num_particles' is owned by the frozen host; revise its task specification; trajectory: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; trajectory: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; trajectory: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; trajectory: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; trajectory: recipe override 'total_steps' is owned by the frozen host; revise its task specification; trajectory: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; trajectory: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; trajectory: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole"></a>

### two_pole

**two_pole: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. two_pole: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; two_pole: recipe override 'batch_size' is owned by the frozen host; revise its task specification; two_pole: recipe override 'conditioning' is owned by the frozen host; revise its task specification; two_pole: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; two_pole: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; two_pole: recipe override 'model' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_classes' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_particles' is owned by the frozen host; revise its task specification; two_pole: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; two_pole: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; two_pole: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; two_pole: recipe override 'total_steps' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'z_dim' is owned by the frozen host; revise its task specification; two_pole: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; two_pole: recipe override 'batch_size' is owned by the frozen host; revise its task specification; two_pole: recipe override 'conditioning' is owned by the frozen host; revise its task specification; two_pole: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; two_pole: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; two_pole: recipe override 'model' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_classes' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_particles' is owned by the frozen host; revise its task specification; two_pole: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; two_pole: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; two_pole: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; two_pole: recipe override 'total_steps' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unipolar"></a>

### unipolar

**unipolar: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. unipolar: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; unipolar: recipe override 'batch_size' is owned by the frozen host; revise its task specification; unipolar: recipe override 'conditioning' is owned by the frozen host; revise its task specification; unipolar: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; unipolar: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; unipolar: recipe override 'model' is owned by the frozen host; revise its task specification; unipolar: recipe override 'num_classes' is owned by the frozen host; revise its task specification; unipolar: recipe override 'num_particles' is owned by the frozen host; revise its task specification; unipolar: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; unipolar: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; unipolar: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; unipolar: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; unipolar: recipe override 'total_steps' is owned by the frozen host; revise its task specification; unipolar: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; unipolar: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; unipolar: recipe override 'z_dim' is owned by the frozen host; revise its task specification; unipolar: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; unipolar: recipe override 'batch_size' is owned by the frozen host; revise its task specification; unipolar: recipe override 'conditioning' is owned by the frozen host; revise its task specification; unipolar: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; unipolar: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; unipolar: recipe override 'model' is owned by the frozen host; revise its task specification; unipolar: recipe override 'num_classes' is owned by the frozen host; revise its task specification; unipolar: recipe override 'num_particles' is owned by the frozen host; revise its task specification; unipolar: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; unipolar: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; unipolar: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; unipolar: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; unipolar: recipe override 'total_steps' is owned by the frozen host; revise its task specification; unipolar: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; unipolar: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; unipolar: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. unused_token_hold: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'batch_size' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'conditioning' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'model' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'num_classes' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'num_particles' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'total_steps' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'z_dim' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'batch_size' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'conditioning' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'model' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'num_classes' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'num_particles' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'total_steps' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; unused_token_hold: recipe override 'z_dim' is owned by the frozen host; revise its task specification

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task; candidate overrides frozen host resource batch_size; declare a different task; candidate overrides frozen host resource num_particles; declare a different task; candidate overrides frozen host resource total_steps; declare a different task

Execution: **incompatible; skipped**.

Used by: [adaptation / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_rail_escape"></a>

### zero_tanh_rail_escape

**zero_tanh_rail_escape: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_rail_escape.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 3](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-zero_tanh_smoke"></a>

### zero_tanh_smoke

**zero_tanh_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](release07-gan-v3-cloud.md#cohort-cuda-7f9c23eb0e27-prior-ablation-full-suite-v1-tier-1).

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

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base Adam optimizer.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Related paired relativistic adversarial objective.
- [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). Inspiration for latent variance/covariance regularization; no full VICReg objective is claimed.
- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894). Related work on one-sided input-gradient penalties. The repository evaluates caps directly on real/fake inputs; it does not reproduce the paper's full training or sampling algorithm.
