# BCAP Tier 2 failures and prior regularization investigation

The selected BCAP direction-blend recipe passes **9/21 Tier 2 tasks and fails 12**.
Its failures separate into temporal retention, mass allocation, density shape and
image fidelity. **A global latent variance collapse does not explain the saved
100-Gaussian endpoints:** both prior coordinates already exceed VICReg's unit
standard-deviation floor. Prior regularization remains a testable recipe change,
but that floor alone does not target the largest native failures.

This report is based on `origin/develop` commit
`0337bbbc2300fd351e77e9a222b92401137fd421` and the selected recipe's existing
[full ordinary comparison](../bcap-default-baseline/README.md). It adds **zero
training updates and zero random draws**. The 21-task measurement appendix includes
every numerical/boolean endpoint field, every declared numerical bound, temporal
requirements, and additional native coverage measurements from the same saved
holdout samples. Diagnostic fields explicitly have no qualification target.
The [current technique leaderboard](../technique-inventory.md) remains the single
ranking; no recipe selection or qualification changes here.

## Failures at a glance

The following rows follow task order, not performance rank. Scores are endpoints
unless a temporal requirement is named. Full per-task tables below retain the
passing measurements alongside failures.

| Task | Recorded result | Failed scores and targets, or passing control |
| --- | --- | --- |
| [gaussian1d_stability](#gaussian1d_stability-fail) | **FAIL** | Stationary checks **2/72**, shifted hold **0/24**, deadline reacquisition **FAIL/PASS**; KS **.320623/≤.05**, mean error **.255238/≤.2**, std ratio **.662330/[.8,1.2]** |
| [five_word_joint_hold](#five_word_joint_hold-pass) | PASS | All **25/25** primary checks and confirmations pass; generation and inverse mapping retained |
| [trajectory](#trajectory-pass) | PASS | Identity MSE **.000258480/≤.02**, terminal suffix **19/≥5** |
| [residual_student](#residual_student-pass) | PASS | Identity MSE **.000249089/≤.02**, success **1/≥1**, wrong-pad rate **0/≤0**, suffix **21/≥5** |
| [unipolar](#unipolar-pass) | PASS | All three endpoint bounds pass; suffix **18/≥5** |
| [cover_leftover](#cover_leftover-pass) | PASS | All six endpoint bounds pass; suffix **22/≥5** |
| [mid_scale_identity](#mid_scale_identity-pass) | PASS | All six endpoint bounds pass; suffix **20/≥5** |
| [mode_hold](#mode_hold-fail) | **FAIL** | Terminal suffix **3/≥5**, despite endpoint modes **8/≥8** and quality **.989258/≥.9** |
| [vector_two_broad](#vector_two_broad-pass) | PASS | All five endpoint bounds pass; suffix **22/≥5** |
| [vector_unequal_mass](#vector_unequal_mass-fail) | **FAIL** | Full component covariance error **3.69165/≤.85**, minimum eigen ratio **.00907291/≥.15**, minimum mass ratio **.207520/≥.25**; suffix **0/≥5** |
| [vector_unequal_width](#vector_unequal_width-fail) | **FAIL** | Full component covariance error **6.28756/≤.85**, mass TV **.291504/≤.15**, normalized SW1 **.295865/≤.18**; suffix **0/≥5** |
| [vector_anisotropic](#vector_anisotropic-fail) | **FAIL** | Mass TV **.195964/≤.15**, normalized SW1 **.197952/≤.18**; component shape bounds pass, suffix **0/≥5** |
| [vector_overlap](#vector_overlap-fail) | **FAIL** | Mean error **.388766/≤.15**, normalized SW1 **.255769/≤.18**; suffix **0/≥5** |
| [vector_spiral](#vector_spiral-pass) | PASS | All three endpoint bounds pass; suffix **23/≥5** |
| [img_stripes2](#img_stripes2-pass) | PASS | Quality **1/≥.9**, modes **2/≥2**, suffix **12/≥5** |
| [img_bars4](#img_bars4-fail) | **FAIL** | Genuine modes **2/≥4**; quality **.968750/≥.9** passes; suffix **0/≥5** |
| [img_blobs4](#img_blobs4-fail) | **FAIL** | Genuine modes **2/≥4**, quality **.75/≥.9**; suffix **0/≥5** |
| [img_intensity2](#img_intensity2-fail) | **FAIL** | Quality **.6875/≥.9**; modes **2/≥2** pass; suffix **0/≥5** |
| [grid100](#grid100-fail) | **FAIL** | Live endpoint genuine modes **9/100**; holdout precision **.24072/≥.97**, mass TV **.14718/≤.06**, center RMS **1.52031/≤.2**, covariance trace bias magnitude **.371385/≤.1**, radial KS **.490883/≤.04**; terminal accuracy **0/5** |
| [rotated100](#rotated100-fail) | **FAIL** | Live endpoint genuine modes **13/100**; holdout precision **.25552/≥.97**, mass TV **.14325/≤.06**; center/covariance/radial accuracy unavailable; terminal accuracy **0/5** |
| [staggered100](#staggered100-fail) | **FAIL** | Live endpoint genuine modes **20/100**; holdout precision **.30168/≥.97**, mass TV **.12277/≤.06**; center/covariance/radial accuracy unavailable; terminal accuracy **0/5** |

Image TV and mean RMSE are diagnostics, not additional frozen gate requirements.
Mixture-vector overall mean/covariance summaries and four-sigma core summaries
also cannot replace their task's full component gates. Overlap uses observable
distribution metrics; nearest-component labels are not true mixture membership.

## Which recipe these scores describe

The measured source is `d378734f40b09ce223a389e8f54a9783ec6a0c75`, scientific
digest `6a225fcdd6922cdad37c9c947e163fb6f164f6ad4390293a3d3091b8f741ce44`.
The selected global recipe uses non-saturating BCAP, coefficient/cap/interval
**1/1/1**, zero-momentum DualNorm, smoothing **.001**, per-offset convolution,
constant G/E step **.012**, D step **.018**, prior step **.030**, and
`constraint_geometry_mode=direction_blend`. Prior regularization, transport,
additive input/output noise and EMA are disabled. The recipe learns prior locations;
tasks own initial distributions, capacity, fixed widths and sampling laws.
The later recipe-ownership refactor in develop does not constitute another trained
cohort. See [the named baseline decision](../bcap-default-baseline/DEFAULT_SELECTION.md).

Both comparison arms use protocol seed **0**, public deterministic initialization,
the same per-task architecture, data/batches, initial prior, sampling, update budget,
evaluation cadence and RTX A6000 runtime. Constructor/data/training/evaluation
streams are isolated and checkpointed. Fixed fixtures remain explicit cohorts.
Frozen scalar/vector hosts retain their original same-real-tensor D/G law;
this investigation does not substitute the newer batch protocol.
The sole compared trainer delta is `none → direction_blend`.
All twelve failed tasks have **identical final metric packets and evaluator
summaries** in the two arms, and the [saved counter proof](../bcap-default-baseline/saved-counter-summary.json)
records no direction-blend activation on them. This supports investigating
distribution dynamics rather than attributing these failures to the new blend.
It does not isolate which G/D/prior force caused a failure.

## Why 100 Gaussians remains a useful target

Each native task has **20,000** uniform learned MoG locations in two latent
dimensions, fixed kernel sigma **.025**, a 128-wide three-hidden-layer generator,
and **7,000** training updates. An even allocation would provide **200 latent
centers per target mode**, unlike the small rare-component vector problem.
This is ample nominal center count; it is not a constructive proof that this
exact architecture and fixed-width prior can satisfy every native shape gate.
The numerical target sampler controls already pass the accuracy audit:

| Existing 100000-draw target control | Grid | Rotated | Staggered | Target |
| --- | ---: | ---: | ---: | --- |
| Precision | .98914 | .98914 | .98914 | ≥.97; analytic target .988891 |
| Mass TV | .01226 | .01226 | .01226 | ≤.06 |
| Center RMS in target sigmas | .0417376 | .0417374 | .0417374 | ≤.2 |
| Absolute covariance trace bias | .000279445 | .000279440 | .000279439 | ≤.1 |
| Radial KS | .00151412 | .00151327 | .00151475 | ≤.04 |

These are already-recorded target draws, not new seed experiments or trained
positive references. They show the scorer accepts the desired distribution.
They do not prove the trainer can learn it. On the trained holdouts, **69.8–75.9%**
of outputs miss the three-sigma quality radius. Grid's center and radial errors
are many times their limits; rotated/staggered shape fields are `null` because
at least one component has fewer than two in-radius samples. `null` is missing
evidence, not zero error. Nearest-cell occupancy is insufficient: the original
and current saved holdouts occupy all **100/100** nearest cells while
genuine-quality coverage remains poor. The appendix recomputes the current
occupancy; the earlier [saved winner analysis](../bcap-tier2-search/FAILURE_ANALYSIS.md)
retains its separate evidence identity.

The older [README 100-Gaussian demonstration](../../readme-100gaussians/README.md)
does not supply a positive for these gates: its 20k-update, annealed, finite-cloud
EMA cohort ends at **.9327/≥.97** quality, with a different mode-count criterion.
Its visual progress supports investigating the question, while its original
budget, serving and initialization identity remain separate.

Coverage also constrains each mode's mass and local spread. The appendix shows
all its bounds, including zero minimum genuine mass/eigen/radial values and
excessive maximum spread. Endpoint coverage uses the original float32 kernel;
accuracy uses float64 residuals. Small quality-radius boundary differences
between their precision fields are preserved. Live endpoint draws, independent
holdouts and target controls remain separate. The receipt's legacy `ema` slots
are not extra credit: EMA decay was zero and all reported gates here are live.

## What the prior measurements say about VICReg

The repository implements a **VICReg-inspired variance hinge plus covariance
penalty** on latent center rows. It has no two-view invariance term. The original
[VICReg paper](https://arxiv.org/abs/2105.04906) motivates variance floors and
decorrelation for avoiding representation collapse; it does not establish a
GAN density or mixture-allocation guarantee. In this implementation,

```text
std_j = sqrt(unbiased_variance(center_coordinate_j) + 1e-4)
variance_loss = mean(max(0, 1 - std_j))
covariance_loss = sum(off_diagonal_covariance^2) / latent_dimension
prior_objective = prior_reg * (variance_loss + covariance_loss)
```

The [saved prior table measurements](#saved-prior-measurements) inspect copied,
hash-verified endpoint centers in float64. For grid, rotated and staggered,
minimum coordinate std is **1.09135, 1.04453 and 1.07113**, respectively: the
unit-floor hinge is **zero**. Scalar retention also has zero hinge despite its
contracted and distorted output law. Thus latent spread and output density are
different quantities. The network can transform a well-spread prior into an
incorrect or narrow output law. Covariance penalties can still move those priors;
their unweighted native full-table losses are small, and neither term directly
controls output centers, mode masses, within-mode covariance or spill.
The largest full-table native row gradients are only **8.89e-7–2.95e-6**
before weighting, compared with optimizer smoothing **.001**. Word hold's
corresponding diagnostic is **.190005**, and image bars' is **2.10516**.
One global coefficient therefore faces very different force scales. These are
copied-center probes, not the combined adversarial update or native sampled-subset
gradient; they motivate measuring the actual applied force before choosing a weight.

The image priors likewise have zero variance hinge, while their covariance terms
can be large. Several passing conditional controls have a positive hinge. A global
regularizer can therefore perturb successful learned codes as well as failures.
Native training evaluates VICReg on unique generator-sampled rows when capacity
exceeds 1,024; smaller standard trainer tables use the whole table. These full-table
probes **do not measure the historical minibatch penalty**. Sampled subsets can
activate a floor even when the whole table does not. The other public component
hosts retain their own declared row-selection law.

The existing [row update contract](../../../particlegan/optim/dualnorm.py) moves
only actually sampled prior rows, even if a whole-table regularizer produces
dense gradients. Its smoothed row response is proportional to
`g / sqrt(||g||² + .001²)` at step **.030**. A small coefficient need not imply
small motion once the combined row gradient exceeds smoothing. This makes
actual regularizer/adversarial gradient balance and finite output travel more
informative than the nominal loss coefficient alone. The probes report
unweighted gradients, not a prediction of the combined trained update.

## Recommendations for a bounded follow-up

1. **Test prior dependence before assuming global latent collapse.** A separately
   declared `prior_update=learned → frozen` diagnostic from the identical initial
   distribution would keep G/D learning and isolate dependence on moving inputs.
   It needs its own smoke producer for Gaussian continuation. It is a recipe
   intervention, not a smaller-capacity task or a new initialization. Preserve
   original failures and do not import historical frozen or affine fixtures.
2. **If pursuing VICReg, test one global recipe change.** Keep learned locations
   and all task conditions fixed; change only `prior_reg=0 → one declared positive
   coefficient`, retaining `prior_regularizer=vicreg`, `target_std=1`, `eps=1e-4`.
   Choose and freeze the coefficient from saved gradient/motion diagnostics before
   admission, rather than proposing a blind grid. Measure floor activation,
   covariance loss, sampled-row gradient balance, prior displacement and actual
   output travel over acquisition, regression and target shift. Falsify the
   collapse account if latent statistics improve while native precision/shape
   and complete retention gates remain poor. Watch all existing passes, especially
   conditional codes, word inverse mapping and ring acquisition.
3. **Prioritize output motion and density together for native repair.** Earlier
   [physical-theory rounds](../bcap-physics/README.md),
   [round two](../bcap-physics/round2/README.md),
   [round three](../bcap-physics/round3/README.md), and
   [round four](../bcap-physics/round4/README.md) already reject simple versions of
   several proposals. Finite native acceptance raised precision to **.65938/≥.97**
   without solving the task. Local transport repaired unequal mass in a scoped
   cohort. [Round five](../bcap-physics/round5/README.md) repaired unequal width
   with balanced assignment but lost rare density and worsened native quality;
   prior-only transport routing also failed. Do not pool these specialist PASSes
   into the selected recipe's 9/21 result or rerun rejected packages unchanged.
4. **Use complete gates as stopping rules.** Mode hold is close only in terminal
   duration; Gaussian retention is an all-check acquisition/retention/adaptation
   requirement. Good endpoint moments or an improved average error do not repair
   them. A new ready Forge study must name its control, global delta, predictions,
   finite campaign budget and stopping rules. Complete all runnable Tier 1 jobs;
   require all **6/6** before ordinary Tier 2. Tier 1 reserves **2,220 seconds**;
   through Tier 2 reserves **42,720 seconds per recipe**, or **43,020** with the
   optional audit. Any selected deeper work after smoke failure requires the
   separately registered calibration lane and grants no ordinary qualification.

No candidate or paid campaign is admitted by this report. Provisional calibration,
the original gates and Tier 3 prerequisite veto remain in force. A later
public-API toy comparison must provide its numerical metric and actual-training
GIF under the [toy test contract](../../toy_audit/api_contract/README.md).

## Evidence and reproduction

[The numerical receipt](measurements.json) binds the selected attempt IDs, recipe
revision, original task/evaluation/sampling contracts, full-precision scores,
targets, source hashes and saved artifact hashes. The reproduction script checks
the input publication against the pinned develop commit,
each frozen request against its Git-pinned task and original raw result, verifies
saved checkpoint/native summary/holdout hashes, checks all twelve unchanged
failure packets against the matched control, and verifies that consumed global
Torch RNG state is unchanged. This is a display and analysis projection, not a
qualification input or a rerun of the independent grader.

The original bulk artifacts remain under
`/mnt/ml7tb/ParticleGAN-forge/bcap-default-adoption-20261010/queue`.
Missing or changed originals block saved-state analysis; they do not justify
another training run. Existing [actual-training media](../bcap-default-baseline/media/index.json)
illustrate all tasks. The original paired comparison's **56 attempts** and
**4,187.683587 charged seconds** remain its costs; this report adds no training
charge. Analysis stdout stays in ignored local storage and is easy to tail:

```sh
mkdir -p runs/software/bcap-tier2-investigation
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/bcap-tier2-investigation/investigate.py --check \
  > runs/software/bcap-tier2-investigation/analysis.log 2>&1
tail -F runs/software/bcap-tier2-investigation/analysis.log
```

Omit `--check` to regenerate this report's measurement sections and compact receipt
from the same original evidence. It launches no trainer, draws no samples and
does not regenerate a leaderboard. The current bound implementations of the
[VICReg-inspired loss](../../../particlegan/vicreg_loss.py) and
[standard trainer](../../../particlegan/training.py) explain the proposed mechanism;
their source is distinct from trained qualification evidence.

<!-- BEGIN GENERATED MEASUREMENTS -->

## Measurements for all 21 Tier 2 tasks

Scores use six significant digits; [the numerical receipt](measurements.json) keeps full precision. PASS/FAIL in a metric row checks that row only. The task result retains its complete recorded gate. UNAVAILABLE fails a required numerical check. DIAGNOSTIC means no declared gate; it grants no PASS. Each task links to its existing actual-training GIF; images do not determine these grades.

### gaussian1d_stability FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-gaussian1d_stability.gif) · Prior `mog`, sigma `0.1` · Declared updates `6000` · allowance `600 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `stationary passing checks` | 2 | == 72 | **FAIL** |
| `deadline reacquisition` | false | == true | **FAIL** |
| `shifted hold passing checks` | 0 | == 24 | **FAIL** |
| `frozen-copy passing checks` | 0 | Diagnostic control out of 48; no qualification gate | DIAGNOSTIC |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sample_count` | 4096 | >= 4096 | PASS |
| `finite_fraction` | 1 | == 1 | PASS |
| `mean_error_sigma` | 0.255238 | <= 0.2 | **FAIL** |
| `std_ratio` | 0.66233 | >= 0.8 and <= 1.2 | **FAIL** |
| `cdf_ks` | 0.320623 | <= 0.05 | **FAIL** |
| `mean` | 3.12762 | No declared gate | DIAGNOSTIC |
| `std` | 0.331165 | No declared gate | DIAGNOSTIC |
| `step` | 6000 | No declared gate | DIAGNOSTIC |

### five_word_joint_hold PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-five_word_joint_hold.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `4000` · allowance `300 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `passing primary checks` | 25 | == 25 | PASS |
| `confirmed checks` | 25 | == 25 | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sample_count` | 1024 | >= 1024 | PASS |
| `quality_fraction` | 1 | >= 0.95 | PASS |
| `modes` | 5 | == 5 | PASS |
| `mass_tv` | 0.0208984 | <= 0.1 | PASS |
| `reconstruction_exact` | 1 | == 1 | PASS |
| `minimum_reconstruction_token_probability` | 0.951635 | >= 0.9 | PASS |
| `completed_steps` | 4834 | No declared gate | DIAGNOSTIC |
| `mass_apple` | 0.21582 | No declared gate | DIAGNOSTIC |
| `mass_berry` | 0.191406 | No declared gate | DIAGNOSTIC |
| `mass_grape` | 0.205078 | No declared gate | DIAGNOSTIC |
| `mass_lemon` | 0.193359 | No declared gate | DIAGNOSTIC |
| `mass_melon` | 0.194336 | No declared gate | DIAGNOSTIC |
| `output_noise_added` | 0 | No declared gate | DIAGNOSTIC |
| `policy_latent_perturbation` | 0 | No declared gate | DIAGNOSTIC |
| `reconstruction_nll` | 0.00218637 | No declared gate | DIAGNOSTIC |
| `served_averaged` | 0 | No declared gate | DIAGNOSTIC |
| `step` | 4834 | No declared gate | DIAGNOSTIC |

### trajectory PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-trajectory.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `400` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 19 | >= 5 | PASS |
| `passing scheduled observations` | 19 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `identity_mse` | 0.00025848 | <= 0.02 | PASS |

**Additional saved host diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `critic_gradient_max` | 1.04098 | No declared gate | DIAGNOSTIC |
| `critic_gradient_median` | 0.955124 | No declared gate | DIAGNOSTIC |
| `own_nearest_fraction` | 1 | No declared gate | DIAGNOSTIC |
| `paired_target_mse` | 0.00025848 | No declared gate | DIAGNOSTIC |
| `particle_mean_square` | 0.751163 | No declared gate | DIAGNOSTIC |
| `particle_std_mean` | 0.902331 | No declared gate | DIAGNOSTIC |
| `set_cover` | 0.00413568 | No declared gate | DIAGNOSTIC |

### residual_student PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-residual_student.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `400` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 21 | >= 5 | PASS |
| `passing scheduled observations` | 21 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `identity_mse` | 0.000249089 | <= 0.02 | PASS |
| `success_rate` | 1 | >= 1 | PASS |
| `wrong_pad_rate` | 0 | <= 0 | PASS |

**Additional saved host diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `both_land_rows` | 12 | No declared gate | DIAGNOSTIC |
| `cover_weight` | 1.5 | No declared gate | DIAGNOSTIC |
| `endpoint_l2` | 0.0328873 | No declared gate | DIAGNOSTIC |
| `fm_weight` | 0 | No declared gate | DIAGNOSTIC |
| `land_tol` | 0.25 | No declared gate | DIAGNOSTIC |
| `n_particles` | 12 | No declared gate | DIAGNOSTIC |
| `paired_target_mse` | 0.000249089 | No declared gate | DIAGNOSTIC |
| `particle_l2` | 0.02 | No declared gate | DIAGNOSTIC |
| `pass` | true | No declared gate | DIAGNOSTIC |
| `pass_identity_mse` | 0.02 | No declared gate | DIAGNOSTIC |
| `reg_coeff` | 1 | No declared gate | DIAGNOSTIC |
| `reg_kappa` | 1 | No declared gate | DIAGNOSTIC |
| `residual_weight` | 1 | No declared gate | DIAGNOSTIC |
| `seed` | 0 | No declared gate | DIAGNOSTIC |
| `slow_impact_max` | 0.1 | No declared gate | DIAGNOSTIC |
| `steps` | 400 | No declared gate | DIAGNOSTIC |
| `success_min` | 1 | No declared gate | DIAGNOSTIC |
| `vicreg_weight` | 0.05 | No declared gate | DIAGNOSTIC |

### unipolar PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-unipolar.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `400` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 18 | >= 5 | PASS |
| `passing scheduled observations` | 20 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `cover` | 0.9938 | >= 0.85 | PASS |
| `off_caption` | 6.61748e-05 | <= 0.05 | PASS |
| `neu_hold` | 0.989787 | >= 0.85 | PASS |

**Additional saved host diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `cos_plus` | 0.999908 | No declared gate | DIAGNOSTIC |
| `cover_weight` | 0 | No declared gate | DIAGNOSTIC |
| `even_norm` | 0.496544 | No declared gate | DIAGNOSTIC |
| `fm_weight` | 0 | No declared gate | DIAGNOSTIC |
| `half_norm_ratio` | 0.500457 | No declared gate | DIAGNOSTIC |
| `hit` | true | No declared gate | DIAGNOSTIC |
| `mag_ratio` | 0.993892 | No declared gate | DIAGNOSTIC |
| `odd_norm` | 0.496544 | No declared gate | DIAGNOSTIC |
| `origin_norm` | 0.0102133 | No declared gate | DIAGNOSTIC |
| `polarity` | 1 | No declared gate | DIAGNOSTIC |
| `reg_coeff` | 1 | No declared gate | DIAGNOSTIC |
| `reg_is_gradient_penalty` | false | No declared gate | DIAGNOSTIC |
| `reg_kappa` | 1 | No declared gate | DIAGNOSTIC |
| `reg_lazy` | 1 | No declared gate | DIAGNOSTIC |
| `seed` | 0 | No declared gate | DIAGNOSTIC |
| `steps` | 400 | No declared gate | DIAGNOSTIC |

### cover_leftover PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-cover_leftover.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `800` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 22 | >= 5 | PASS |
| `passing scheduled observations` | 22 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `u_kept` | 1.00052 | >= 0.85 | PASS |
| `content_kept` | 1.00015 | >= 0.75 | PASS |
| `leak_ratio` | 0.00777234 | <= 0.2 | PASS |
| `pole_rel_err_plus` | 0.00590539 | <= 0.2 | PASS |
| `pole_rel_err_minus` | 0.0095229 | <= 0.2 | PASS |
| `same_dir` | 0.00651053 | <= 0.25 | PASS |

**Additional saved host diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `cover_weight` | 1.5 | No declared gate | DIAGNOSTIC |
| `covered` | true | No declared gate | DIAGNOSTIC |
| `even_norm` | 0.00088297 | No declared gate | DIAGNOSTIC |
| `fm_weight` | 0 | No declared gate | DIAGNOSTIC |
| `kappa` | 1 | No declared gate | DIAGNOSTIC |
| `leak_frac` | -0.999999 | No declared gate | DIAGNOSTIC |
| `n_particles` | 12 | No declared gate | DIAGNOSTIC |
| `odd_norm` | 1.1131 | No declared gate | DIAGNOSTIC |
| `on_content` | 0.535927 | No declared gate | DIAGNOSTIC |
| `on_e` | -0.000122681 | No declared gate | DIAGNOSTIC |
| `on_u` | 0.975464 | No declared gate | DIAGNOSTIC |
| `particle_l2` | 0.02 | No declared gate | DIAGNOSTIC |
| `particle_rms` | 0.198193 | No declared gate | DIAGNOSTIC |
| `pass` | true | No declared gate | DIAGNOSTIC |
| `reg_coeff` | 1 | No declared gate | DIAGNOSTIC |
| `seed` | 0 | No declared gate | DIAGNOSTIC |
| `steps` | 800 | No declared gate | DIAGNOSTIC |

### mid_scale_identity PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-mid_scale_identity.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `800` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 20 | >= 5 | PASS |
| `passing scheduled observations` | 21 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `concept_cos_plus` | 0.999944 | >= 0.85 | PASS |
| `concept_cos_minus` | 0.999962 | >= 0.85 | PASS |
| `concept_mag_plus` | 1.00204 | >= 0.75 and <= 1.25 | PASS |
| `concept_mag_minus` | 1.00013 | >= 0.75 and <= 1.25 | PASS |
| `identity_at_0` | 0.984797 | >= 0.85 | PASS |
| `identity_at_mid` | 0.992997 | >= 0.85 | PASS |

**Additional saved host diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `cover_weight` | 1.5 | No declared gate | DIAGNOSTIC |
| `fm_weight` | 0 | No declared gate | DIAGNOSTIC |
| `leak_frac` | -0.99985 | No declared gate | DIAGNOSTIC |
| `pass` | true | No declared gate | DIAGNOSTIC |
| `reg_calls` | 3200 | No declared gate | DIAGNOSTIC |
| `reg_coeff` | 1 | No declared gate | DIAGNOSTIC |
| `reg_is_gradient_penalty` | false | No declared gate | DIAGNOSTIC |
| `reg_kappa` | 1 | No declared gate | DIAGNOSTIC |
| `reg_lazy` | 1 | No declared gate | DIAGNOSTIC |
| `same_dir` | 0.00864692 | No declared gate | DIAGNOSTIC |
| `seed` | 0 | No declared gate | DIAGNOSTIC |
| `steps` | 800 | No declared gate | DIAGNOSTIC |

### mode_hold FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-mode_hold.gif) · Prior `mog`, sigma `0.025` · Declared updates `1200` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 3 | >= 5 | **FAIL** |
| `passing scheduled observations` | 10 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `modes` | 8 | >= 8 | PASS |
| `hq` | 0.989258 | >= 0.9 | PASS |
| `cover` | 1 | No declared gate | DIAGNOSTIC |
| `effective_modes` | 7.15797 | No declared gate | DIAGNOSTIC |
| `n_modes` | 8 | No declared gate | DIAGNOSTIC |
| `step` | 1200 | No declared gate | DIAGNOSTIC |

### vector_two_broad PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-vector_two_broad.gif) · Prior `mog`, sigma `0.025` · Declared updates `1200` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 22 | >= 5 | PASS |
| `passing scheduled observations` | 22 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sw1_normalized` | 0.13459 | <= 0.18 | PASS |
| `mass_tv` | 0.0629883 | <= 0.15 | PASS |
| `hq` | 0.98877 | >= 0.85 | PASS |
| `component_covariance_error` | 0.385581 | <= 0.85 | PASS |
| `component_min_eigen_ratio` | 0.400066 | >= 0.15 | PASS |
| `component_core_covariance_error` | 0.385581 | No declared gate | DIAGNOSTIC |
| `component_core_min_eigen_ratio` | 0.400066 | No declared gate | DIAGNOSTIC |
| `covariance_error` | 0.111471 | No declared gate | DIAGNOSTIC |
| `max_component_spill` | 0.0184358 | No declared gate | DIAGNOSTIC |
| `mean_error` | 0.191464 | No declared gate | DIAGNOSTIC |
| `min_mass_ratio` | 0.874023 | No declared gate | DIAGNOSTIC |
| `resolved_core_covariance_error` | 0.385581 | No declared gate | DIAGNOSTIC |
| `resolved_core_min_eigen_ratio` | 0.400066 | No declared gate | DIAGNOSTIC |
| `resolved_max_component_spill` | 0.0184358 | No declared gate | DIAGNOSTIC |
| `sample_count` | 4096 | No declared gate | DIAGNOSTIC |
| `step` | 1200 | No declared gate | DIAGNOSTIC |
| `target_scale` | 1.05556 | No declared gate | DIAGNOSTIC |

### vector_unequal_mass FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-vector_unequal_mass.gif) · Prior `mog`, sigma `0.025` · Declared updates `1200` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 0 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sw1_normalized` | 0.154638 | <= 0.18 | PASS |
| `mass_tv` | 0.0706543 | <= 0.15 | PASS |
| `hq` | 0.95459 | >= 0.85 | PASS |
| `component_covariance_error` | 3.69165 | <= 0.85 | **FAIL** |
| `component_min_eigen_ratio` | 0.00907291 | >= 0.15 | **FAIL** |
| `min_mass_ratio` | 0.20752 | >= 0.25 | **FAIL** |
| `component_core_covariance_error` | 0.530393 | No declared gate | DIAGNOSTIC |
| `component_core_min_eigen_ratio` | 0.00907291 | No declared gate | DIAGNOSTIC |
| `covariance_error` | 0.0803272 | No declared gate | DIAGNOSTIC |
| `max_component_spill` | 0.227273 | No declared gate | DIAGNOSTIC |
| `mean_error` | 0.211882 | No declared gate | DIAGNOSTIC |
| `resolved_core_covariance_error` | 0.387724 | No declared gate | DIAGNOSTIC |
| `resolved_core_min_eigen_ratio` | 0.380592 | No declared gate | DIAGNOSTIC |
| `resolved_max_component_spill` | 0.227273 | No declared gate | DIAGNOSTIC |
| `sample_count` | 4096 | No declared gate | DIAGNOSTIC |
| `step` | 1200 | No declared gate | DIAGNOSTIC |
| `target_scale` | 1.75933 | No declared gate | DIAGNOSTIC |

### vector_unequal_width FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-vector_unequal_width.gif) · Prior `mog`, sigma `0.025` · Declared updates `1200` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 0 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sw1_normalized` | 0.295865 | <= 0.18 | **FAIL** |
| `mass_tv` | 0.291504 | <= 0.15 | **FAIL** |
| `hq` | 0.97876 | >= 0.85 | PASS |
| `component_covariance_error` | 6.28756 | <= 0.85 | **FAIL** |
| `component_min_eigen_ratio` | 0.307472 | >= 0.15 | PASS |
| `component_core_covariance_error` | 0.444178 | No declared gate | DIAGNOSTIC |
| `component_core_min_eigen_ratio` | 0.307472 | No declared gate | DIAGNOSTIC |
| `covariance_error` | 0.25809 | No declared gate | DIAGNOSTIC |
| `max_component_spill` | 0.0862944 | No declared gate | DIAGNOSTIC |
| `mean_error` | 0.409949 | No declared gate | DIAGNOSTIC |
| `min_mass_ratio` | 0.192383 | No declared gate | DIAGNOSTIC |
| `resolved_core_covariance_error` | 0.444178 | No declared gate | DIAGNOSTIC |
| `resolved_core_min_eigen_ratio` | 0.307472 | No declared gate | DIAGNOSTIC |
| `resolved_max_component_spill` | 0.0862944 | No declared gate | DIAGNOSTIC |
| `sample_count` | 4096 | No declared gate | DIAGNOSTIC |
| `step` | 1200 | No declared gate | DIAGNOSTIC |
| `target_scale` | 2.1344 | No declared gate | DIAGNOSTIC |

### vector_anisotropic FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-vector_anisotropic.gif) · Prior `mog`, sigma `0.025` · Declared updates `1200` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 0 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sw1_normalized` | 0.197952 | <= 0.18 | **FAIL** |
| `mass_tv` | 0.195964 | <= 0.15 | **FAIL** |
| `hq` | 0.97998 | >= 0.85 | PASS |
| `component_covariance_error` | 0.449625 | <= 0.85 | PASS |
| `component_min_eigen_ratio` | 0.204013 | >= 0.15 | PASS |
| `component_core_covariance_error` | 0.344346 | No declared gate | DIAGNOSTIC |
| `component_core_min_eigen_ratio` | 0.204013 | No declared gate | DIAGNOSTIC |
| `covariance_error` | 0.338893 | No declared gate | DIAGNOSTIC |
| `max_component_spill` | 0.0858896 | No declared gate | DIAGNOSTIC |
| `mean_error` | 0.236065 | No declared gate | DIAGNOSTIC |
| `min_mass_ratio` | 0.477539 | No declared gate | DIAGNOSTIC |
| `resolved_core_covariance_error` | 0.344346 | No declared gate | DIAGNOSTIC |
| `resolved_core_min_eigen_ratio` | 0.204013 | No declared gate | DIAGNOSTIC |
| `resolved_max_component_spill` | 0.0858896 | No declared gate | DIAGNOSTIC |
| `sample_count` | 4096 | No declared gate | DIAGNOSTIC |
| `step` | 1200 | No declared gate | DIAGNOSTIC |
| `target_scale` | 2.03229 | No declared gate | DIAGNOSTIC |

### vector_overlap FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-vector_overlap.gif) · Prior `mog`, sigma `0.025` · Declared updates `1200` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 5 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sw1_normalized` | 0.255769 | <= 0.18 | **FAIL** |
| `mean_error` | 0.388766 | <= 0.15 | **FAIL** |
| `covariance_error` | 0.43849 | <= 0.45 | PASS |
| `sample_count` | 4096 | No declared gate | DIAGNOSTIC |
| `step` | 1200 | No declared gate | DIAGNOSTIC |
| `target_scale` | 0.850144 | No declared gate | DIAGNOSTIC |

### vector_spiral PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-vector_spiral.gif) · Prior `mog`, sigma `0.025` · Declared updates `1600` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 23 | >= 5 | PASS |
| `passing scheduled observations` | 23 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `sw1_normalized` | 0.0587562 | <= 0.18 | PASS |
| `mean_error` | 0.0824902 | <= 0.15 | PASS |
| `covariance_error` | 0.0718014 | <= 0.45 | PASS |
| `sample_count` | 4096 | No declared gate | DIAGNOSTIC |
| `step` | 1600 | No declared gate | DIAGNOSTIC |
| `target_scale` | 1.22418 | No declared gate | DIAGNOSTIC |

### img_stripes2 PASS

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-img_stripes2.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `600` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 12 | >= 5 | PASS |
| `passing scheduled observations` | 12 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `modes` | 2 | >= 2 | PASS |
| `hq` | 1 | >= 0.9 | PASS |
| `distribution_tv` | 0.125 | No declared gate | DIAGNOSTIC |
| `mean_rmse` | 5.96898e-05 | No declared gate | DIAGNOSTIC |
| `step` | 600 | No declared gate | DIAGNOSTIC |

### img_bars4 FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-img_bars4.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `600` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 0 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `modes` | 2 | >= 4 | **FAIL** |
| `hq` | 0.96875 | >= 0.9 | PASS |
| `distribution_tv` | 0.40625 | No declared gate | DIAGNOSTIC |
| `mean_rmse` | 0.0139485 | No declared gate | DIAGNOSTIC |
| `step` | 600 | No declared gate | DIAGNOSTIC |

### img_blobs4 FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-img_blobs4.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `600` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 0 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `modes` | 2 | >= 4 | **FAIL** |
| `hq` | 0.75 | >= 0.9 | **FAIL** |
| `distribution_tv` | 0.1875 | No declared gate | DIAGNOSTIC |
| `mean_rmse` | 0.0562021 | No declared gate | DIAGNOSTIC |
| `step` | 600 | No declared gate | DIAGNOSTIC |

### img_intensity2 FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-img_intensity2.gif) · Prior `particle_cloud`, sigma `0.0` · Declared updates `600` · allowance `1800 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `terminal passing suffix` | 0 | >= 5 | **FAIL** |
| `passing scheduled observations` | 0 | Diagnostic count out of 24; suffix is the gate | DIAGNOSTIC |
| `complete declared execution` | true | == true | PASS |

**Recorded endpoint**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `modes` | 2 | >= 2 | PASS |
| `hq` | 0.6875 | >= 0.9 | **FAIL** |
| `distribution_tv` | 0.03125 | No declared gate | DIAGNOSTIC |
| `mean_rmse` | 0.0601416 | No declared gate | DIAGNOSTIC |
| `step` | 600 | No declared gate | DIAGNOSTIC |

### grid100 FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-grid100.gif) · Prior `mog`, sigma `0.025` · Declared updates `7000` · allowance `3600 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `coverage decision` | false | == true | **FAIL** |
| `terminal live accuracy passing checks` | 0 | == 5 | **FAIL** |

**Live endpoint coverage, 20000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 20000 | >= 20000 | PASS |
| `valid_n` | 20000 | == 20000 | PASS |
| `modes` | 9 | == 100 | **FAIL** |
| `precision` | 0.24435 | >= 0.97 | **FAIL** |
| `min_hq_mode_mass` | 0 | >= 0.005 | **FAIL** |
| `mass_tv` | 0.148 | <= 0.1 | **FAIL** |
| `max_mode_mass` | 0.0228 | <= 0.02 | **FAIL** |
| `min_cov_eig_ratio` | 0 | >= 0.4 | **FAIL** |
| `max_cov_eig_ratio` | 4.14073 | <= 1.7 | **FAIL** |
| `min_radial_median_ratio` | 0 | >= 0.65 | **FAIL** |
| `max_radial_median_ratio` | 2.50679 | <= 1.4 | **FAIL** |
| `min_hq_count` | 0 | >= 100 | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `hq` | 0.24435 | No declared gate | DIAGNOSTIC |

**Live endpoint accuracy, 20000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 20000 | >= 20000 | PASS |
| `valid_n` | 20000 | == 20000 | PASS |
| `all_finite` | true | == true | PASS |
| `precision` | 0.2444 | >= 0.97 | **FAIL** |
| `mass_tv` | 0.148 | <= 0.06 | **FAIL** |
| `center_rms_sigma` | unavailable | <= 0.2 | **UNAVAILABLE** |
| `abs_cov_trace_bias` | unavailable | <= 0.1 | **UNAVAILABLE** |
| `radial_ks` | unavailable | <= 0.04 | **UNAVAILABLE** |
| `accuracy_pass` | false | == true | **FAIL** |
| `frozen_pass` | false | == true | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `accuracy_score` | unavailable | No declared gate | **UNAVAILABLE** |
| `center_max_sigma` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_frob_rms` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_trace_bias` | unavailable | No declared gate | **UNAVAILABLE** |
| `precision_gap` | 0.744491 | No declared gate | DIAGNOSTIC |
| `within_radius_n` | 4888 | No declared gate | DIAGNOSTIC |

**Independent live holdout accuracy, 100000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 100000 | >= 20000 | PASS |
| `valid_n` | 100000 | == 100000 | PASS |
| `all_finite` | true | == true | PASS |
| `precision` | 0.24072 | >= 0.97 | **FAIL** |
| `mass_tv` | 0.14718 | <= 0.06 | **FAIL** |
| `center_rms_sigma` | 1.52031 | <= 0.2 | **FAIL** |
| `abs_cov_trace_bias` | 0.371385 | <= 0.1 | **FAIL** |
| `radial_ks` | 0.490883 | <= 0.04 | **FAIL** |
| `accuracy_pass` | false | == true | **FAIL** |
| `frozen_pass` | false | == true | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `accuracy_score` | 6.51012 | No declared gate | DIAGNOSTIC |
| `center_max_sigma` | 2.49686 | No declared gate | DIAGNOSTIC |
| `cov_frob_rms` | 0.662866 | No declared gate | DIAGNOSTIC |
| `cov_trace_bias` | 0.371385 | No declared gate | DIAGNOSTIC |
| `precision_gap` | 0.748171 | No declared gate | DIAGNOSTIC |
| `within_radius_n` | 24072 | No declared gate | DIAGNOSTIC |

**Independent holdout coverage recomputed on those same saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 100000 | >= 20000 | PASS |
| `valid_n` | 100000 | == 100000 | PASS |
| `modes` | 8 | == 100 | **FAIL** |
| `precision` | 0.24072 | >= 0.97 | **FAIL** |
| `min_hq_mode_mass` | 2e-05 | >= 0.005 | **FAIL** |
| `mass_tv` | 0.14718 | <= 0.1 | **FAIL** |
| `max_mode_mass` | 0.02201 | <= 0.02 | **FAIL** |
| `min_cov_eig_ratio` | 0 | >= 0.4 | **FAIL** |
| `max_cov_eig_ratio` | 2.62235 | <= 1.7 | **FAIL** |
| `min_radial_median_ratio` | 1.5768 | >= 0.65 | PASS |
| `max_radial_median_ratio` | 2.29117 | <= 1.4 | **FAIL** |
| `min_hq_count` | 2 | >= 500 | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `hq` | 0.24072 | No declared gate | DIAGNOSTIC |

**Saved holdout placement diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `nearest cells occupied` | 100 | Diagnostic reference 100; genuine modes use the coverage gate | DIAGNOSTIC |
| `median nearest-center distance in target sigmas` | 4.45666 | No declared gate | DIAGNOSTIC |

### rotated100 FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-rotated100.gif) · Prior `mog`, sigma `0.025` · Declared updates `7000` · allowance `3600 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `coverage decision` | false | == true | **FAIL** |
| `terminal live accuracy passing checks` | 0 | == 5 | **FAIL** |

**Live endpoint coverage, 20000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 20000 | >= 20000 | PASS |
| `valid_n` | 20000 | == 20000 | PASS |
| `modes` | 13 | == 100 | **FAIL** |
| `precision` | 0.2478 | >= 0.97 | **FAIL** |
| `min_hq_mode_mass` | 0 | >= 0.005 | **FAIL** |
| `mass_tv` | 0.1397 | <= 0.1 | **FAIL** |
| `max_mode_mass` | 0.02485 | <= 0.02 | **FAIL** |
| `min_cov_eig_ratio` | 0 | >= 0.4 | **FAIL** |
| `max_cov_eig_ratio` | 2.97754 | <= 1.7 | **FAIL** |
| `min_radial_median_ratio` | 0 | >= 0.65 | **FAIL** |
| `max_radial_median_ratio` | 2.18165 | <= 1.4 | **FAIL** |
| `min_hq_count` | 0 | >= 100 | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `hq` | 0.2478 | No declared gate | DIAGNOSTIC |

**Live endpoint accuracy, 20000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 20000 | >= 20000 | PASS |
| `valid_n` | 20000 | == 20000 | PASS |
| `all_finite` | true | == true | PASS |
| `precision` | 0.24785 | >= 0.97 | **FAIL** |
| `mass_tv` | 0.1397 | <= 0.06 | **FAIL** |
| `center_rms_sigma` | unavailable | <= 0.2 | **UNAVAILABLE** |
| `abs_cov_trace_bias` | unavailable | <= 0.1 | **UNAVAILABLE** |
| `radial_ks` | unavailable | <= 0.04 | **UNAVAILABLE** |
| `accuracy_pass` | false | == true | **FAIL** |
| `frozen_pass` | false | == true | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `accuracy_score` | unavailable | No declared gate | **UNAVAILABLE** |
| `center_max_sigma` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_frob_rms` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_trace_bias` | unavailable | No declared gate | **UNAVAILABLE** |
| `precision_gap` | 0.741041 | No declared gate | DIAGNOSTIC |
| `within_radius_n` | 4957 | No declared gate | DIAGNOSTIC |

**Independent live holdout accuracy, 100000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 100000 | >= 20000 | PASS |
| `valid_n` | 100000 | == 100000 | PASS |
| `all_finite` | true | == true | PASS |
| `precision` | 0.25552 | >= 0.97 | **FAIL** |
| `mass_tv` | 0.14325 | <= 0.06 | **FAIL** |
| `center_rms_sigma` | unavailable | <= 0.2 | **UNAVAILABLE** |
| `abs_cov_trace_bias` | unavailable | <= 0.1 | **UNAVAILABLE** |
| `radial_ks` | unavailable | <= 0.04 | **UNAVAILABLE** |
| `accuracy_pass` | false | == true | **FAIL** |
| `frozen_pass` | false | == true | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `accuracy_score` | unavailable | No declared gate | **UNAVAILABLE** |
| `center_max_sigma` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_frob_rms` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_trace_bias` | unavailable | No declared gate | **UNAVAILABLE** |
| `precision_gap` | 0.733371 | No declared gate | DIAGNOSTIC |
| `within_radius_n` | 25552 | No declared gate | DIAGNOSTIC |

**Independent holdout coverage recomputed on those same saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 100000 | >= 20000 | PASS |
| `valid_n` | 100000 | == 100000 | PASS |
| `modes` | 14 | == 100 | **FAIL** |
| `precision` | 0.25552 | >= 0.97 | **FAIL** |
| `min_hq_mode_mass` | 1e-05 | >= 0.005 | **FAIL** |
| `mass_tv` | 0.14325 | <= 0.1 | **FAIL** |
| `max_mode_mass` | 0.0252 | <= 0.02 | **FAIL** |
| `min_cov_eig_ratio` | 0 | >= 0.4 | **FAIL** |
| `max_cov_eig_ratio` | 2.58847 | <= 1.7 | **FAIL** |
| `min_radial_median_ratio` | 1.37206 | >= 0.65 | PASS |
| `max_radial_median_ratio` | 2.488 | <= 1.4 | **FAIL** |
| `min_hq_count` | 1 | >= 500 | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `hq` | 0.25552 | No declared gate | DIAGNOSTIC |

**Saved holdout placement diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `nearest cells occupied` | 100 | Diagnostic reference 100; genuine modes use the coverage gate | DIAGNOSTIC |
| `median nearest-center distance in target sigmas` | 4.73238 | No declared gate | DIAGNOSTIC |

### staggered100 FAIL

[Actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-default-baseline/media/candidate-staggered100.gif) · Prior `mog`, sigma `0.025` · Declared updates `7000` · allowance `3600 s`.

**Required temporal gate**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `coverage decision` | false | == true | **FAIL** |
| `terminal live accuracy passing checks` | 0 | == 5 | **FAIL** |

**Live endpoint coverage, 20000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 20000 | >= 20000 | PASS |
| `valid_n` | 20000 | == 20000 | PASS |
| `modes` | 20 | == 100 | **FAIL** |
| `precision` | 0.29765 | >= 0.97 | **FAIL** |
| `min_hq_mode_mass` | 0 | >= 0.005 | **FAIL** |
| `mass_tv` | 0.12455 | <= 0.1 | **FAIL** |
| `max_mode_mass` | 0.0225 | <= 0.02 | **FAIL** |
| `min_cov_eig_ratio` | 0 | >= 0.4 | **FAIL** |
| `max_cov_eig_ratio` | 3.07548 | <= 1.7 | **FAIL** |
| `min_radial_median_ratio` | 0 | >= 0.65 | **FAIL** |
| `max_radial_median_ratio` | 2.35846 | <= 1.4 | **FAIL** |
| `min_hq_count` | 0 | >= 100 | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `hq` | 0.29765 | No declared gate | DIAGNOSTIC |

**Live endpoint accuracy, 20000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 20000 | >= 20000 | PASS |
| `valid_n` | 20000 | == 20000 | PASS |
| `all_finite` | true | == true | PASS |
| `precision` | 0.29765 | >= 0.97 | **FAIL** |
| `mass_tv` | 0.12455 | <= 0.06 | **FAIL** |
| `center_rms_sigma` | unavailable | <= 0.2 | **UNAVAILABLE** |
| `abs_cov_trace_bias` | unavailable | <= 0.1 | **UNAVAILABLE** |
| `radial_ks` | unavailable | <= 0.04 | **UNAVAILABLE** |
| `accuracy_pass` | false | == true | **FAIL** |
| `frozen_pass` | false | == true | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `accuracy_score` | unavailable | No declared gate | **UNAVAILABLE** |
| `center_max_sigma` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_frob_rms` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_trace_bias` | unavailable | No declared gate | **UNAVAILABLE** |
| `precision_gap` | 0.691241 | No declared gate | DIAGNOSTIC |
| `within_radius_n` | 5953 | No declared gate | DIAGNOSTIC |

**Independent live holdout accuracy, 100000 saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 100000 | >= 20000 | PASS |
| `valid_n` | 100000 | == 100000 | PASS |
| `all_finite` | true | == true | PASS |
| `precision` | 0.30168 | >= 0.97 | **FAIL** |
| `mass_tv` | 0.12277 | <= 0.06 | **FAIL** |
| `center_rms_sigma` | unavailable | <= 0.2 | **UNAVAILABLE** |
| `abs_cov_trace_bias` | unavailable | <= 0.1 | **UNAVAILABLE** |
| `radial_ks` | unavailable | <= 0.04 | **UNAVAILABLE** |
| `accuracy_pass` | false | == true | **FAIL** |
| `frozen_pass` | false | == true | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `accuracy_score` | unavailable | No declared gate | **UNAVAILABLE** |
| `center_max_sigma` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_frob_rms` | unavailable | No declared gate | **UNAVAILABLE** |
| `cov_trace_bias` | unavailable | No declared gate | **UNAVAILABLE** |
| `precision_gap` | 0.687211 | No declared gate | DIAGNOSTIC |
| `within_radius_n` | 30168 | No declared gate | DIAGNOSTIC |

**Independent holdout coverage recomputed on those same saved draws**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `n` | 100000 | >= 20000 | PASS |
| `valid_n` | 100000 | == 100000 | PASS |
| `modes` | 19 | == 100 | **FAIL** |
| `precision` | 0.30168 | >= 0.97 | **FAIL** |
| `min_hq_mode_mass` | 0 | >= 0.005 | **FAIL** |
| `mass_tv` | 0.12277 | <= 0.1 | **FAIL** |
| `max_mode_mass` | 0.02277 | <= 0.02 | **FAIL** |
| `min_cov_eig_ratio` | 0 | >= 0.4 | **FAIL** |
| `max_cov_eig_ratio` | 2.25366 | <= 1.7 | **FAIL** |
| `min_radial_median_ratio` | 0 | >= 0.65 | **FAIL** |
| `max_radial_median_ratio` | 2.26078 | <= 1.4 | **FAIL** |
| `min_hq_count` | 0 | >= 500 | **FAIL** |
| `passed` | false | == true | **FAIL** |
| `hq` | 0.30168 | No declared gate | DIAGNOSTIC |

**Saved holdout placement diagnostics**

| Measurement | Score | Target | Result |
| --- | ---: | --- | --- |
| `nearest cells occupied` | 100 | Diagnostic reference 100; genuine modes use the coverage gate | DIAGNOSTIC |
| `median nearest-center distance in target sigmas` | 4.1277 | No declared gate | DIAGNOSTIC |

## Saved prior measurements

Full-table diagnostics use the existing `target_std=1`, `eps=1e-4` regularizer on copied endpoint centers. They do not change the trained recipe, whose coefficient is zero. Standard deviations include epsilon, matching the implementation. The numerical receipt also retains each dimension, table shape and unweighted gradient norms.

| Task | Center std min / max, target ≥1 | Variance hinge, diagnostic ideal 0 | Covariance loss, diagnostic ideal 0 | Maximum row gradient norm, no declared gate |
| --- | ---: | ---: | ---: | ---: |
| `gaussian1d_stability` (prior) | 1.12099 / 1.21515 | 0 | 0.037461 | 0.00469488 |
| `five_word_joint_hold` (prior) | 0.576753 / 0.836369 | 0.293439 | 0.039326 | 0.190005 |
| `trajectory` (prior0) | 0.838617 / 1.0001 | 0.0976387 | 0.00140368 | 0.0529975 |
| `residual_student` (prior0) | 0.997038 / 0.998379 | 0.00213768 | 0.0002563 | 0.0667876 |
| `cover_leftover` (prior0) | 0.0347319 / 0.475732 | 0.801581 | 0.00695256 | 0.0881665 |
| `cover_leftover` (prior1) | 0.0194051 / 0.0271894 | 0.976746 | 8.78793e-08 | 0.0544356 |
| `mode_hold` (prior) | 1.11939 / 1.34982 | 0 | 0.367153 | 0.219211 |
| `vector_two_broad` (prior) | 1.00363 / 1.09846 | 0 | 0.0382247 | 0.00294362 |
| `vector_unequal_mass` (prior) | 1.02571 / 1.22665 | 0 | 0.00983391 | 0.0018067 |
| `vector_unequal_width` (prior) | 1.02738 / 1.14094 | 0 | 0.0337837 | 0.00297295 |
| `vector_anisotropic` (prior) | 0.960356 / 1.23523 | 0.0139284 | 0.0133319 | 0.00380455 |
| `vector_overlap` (prior) | 1.01584 / 1.04155 | 0 | 0.0191547 | 0.00282889 |
| `vector_spiral` (prior) | 1.02805 / 1.09705 | 0 | 0.0208496 | 0.00309863 |
| `img_stripes2` (prior) | 1.1539 / 3.8847 | 0 | 12.2469 | 0.757156 |
| `img_bars4` (prior) | 1.39001 / 2.95263 | 0 | 44.3065 | 2.10516 |
| `img_blobs4` (prior) | 1.06976 / 2.4138 | 0 | 16.3263 | 0.779939 |
| `img_intensity2` (prior) | 1.06335 / 1.57557 | 0 | 1.59169 | 0.197483 |
| `grid100` (prior) | 1.09135 / 1.09182 | 0 | 1.64442e-05 | 1.53509e-06 |
| `rotated100` (prior) | 1.04453 / 1.07602 | 0 | 7.05559e-05 | 2.95092e-06 |
| `staggered100` (prior) | 1.07113 / 1.075 | 0 | 5.99182e-06 | 8.89079e-07 |

<!-- END GENERATED MEASUREMENTS -->
