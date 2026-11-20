# R1/R2 failure investigation

R1/R2's measured blocker is within-mode density quality. Finding all 16 ring
modes does not satisfy its covariance, sample-quality and sustained-spread
gates. An independent public-trainer oracle finds no error in the inspected
loss signs, half-gamma penalty, detach paths, native Adam updates or schedules.
There is no demonstrated recipe that repairs the unchanged full ordinary task
suite. This investigation preserves that result.

The subsequent [bounded repair comparison](../r1r2-repair-20261022/README.md)
trains the stronger reference, R1-only and explicit joint reconstruction
controls. Its new source cohorts and results are separate from this audit.

The [current family leaderboard](../technique-inventory.md) remains the only
generated ranking. Its selected R1/R2 row records five passed tasks, two failed
tasks and 42 not measured under its published projection. Recorded source
measurements, newer task contracts and separate diagnostic attempts retain
their own identities. This report adds no qualification or replacement recipe.

The work uses three independent reviews:

- [Protocol and prior repairs](protocol-audit.md): exact task/recipe ownership,
  duration controls, acquisition versus retention, and negative search history.
- [Implementation and update oracle](implementation-audit.md): the public
  trainer agrees with literal equations and native Adam within absolute
  `2e-12` over seven small CPU software updates; 110 focused tests pass.
- [Saved endpoint forces](saved-force-diagnosis.json): archived checkpoints,
  critic gradients and parameter-space projections; no optimizer is advanced.

## What the existing experiments establish

These are diagnostic evidence comparisons, not a new ranking. Values keep
their original source, recipe and sampling contracts.

| Recorded Ring16 cohort | Covariance error <= .85 | Minimum eigen ratio >= .15 | HQ >= .85 | Original result |
| --- | ---: | ---: | ---: | --- |
| Selected scheduled recipe; 1,600 updates | 2.066388 | .223238 | .844238 | FAIL; 0/96 full passes |
| Stronger unselected constant NS/AMSGrad control | 1.556546 | .094075 | .888672 | FAIL; covariance and spread |
| Energy-U with learned prior | 6.371506 | 1.027582 | .747559 | FAIL; covariance and HQ |
| Energy-U with frozen prior | 6.259248 | .748672 | .552246 | FAIL; covariance and HQ |
| Selected family, separate stretched-schedule 6,400-update diagnostic | .423491 | .157181 | .977539 | PASS; exactly 5 terminal checks |

Sources: [selected-recipe publication](../recipe-prior-refactor/publication/results.json),
[concluded repairs](../r1r2-concluded-archive/README.md), and
[schedule diagnostic](../ring16-schedule-20261011/results.json).
The last row changes the schedule and execution budget; it cannot fill the
1,600-update task. Its earlier 1,600/3,200/4,800 cuts all fail. Its final spread
margin is only .007181, so it is a scoped, fragile result.

The task deliberately binds 1,600 permitted updates to the original schedule
horizon of 400. Beta2 and penalty strength settle after 80 updates, and LR
reaches its floor after 400. This explains why the remaining 1,200 updates
use the floor; it is a declared protocol choice. The longer matched schedule
contrast supports schedule exhaustion as a contributor at a longer budget,
without isolating LR from beta2 and gamma or solving the short task.

Density failure also remains after removing schedules. All 34 completed repair
revisions in the concluded archive fail Ring16. Momentum, AMSGrad, rate roles,
spectral updates, weaker penalties, non-saturating loss, transport, energy and
prior freezing already have negative evidence. Energy restores spread while
worsening covariance and HQ; freezing prior locations does not remove that
failure. A single spread statistic cannot stand in for distribution fidelity.

Retention is a separate problem. The selected word recipe first acquires a
confirmed solution at update 2,501, then loses it. Its own 4,000-update hold
ends with one quality word and inverse NLL 11.006416. Acquisition remains PASS;
the separate hold remains FAIL.

## What the saved-state analysis adds

The strongest non-saturating control has substantial critic input gradients
(median norm .1645) and a non-saturating loss multiplier near .5. Simple
adversarial gradient starvation does not explain this endpoint. Its radial
variance is already too narrow while tangential variance is too wide. Through
the saved generator Jacobian and frozen AMSGrad denominator, the current
adversarial direction contracts radial variance further. This sign survives
the two declared Gaussian-quadrature resolutions (drifts -.02956 and -.02856).
The original covariance-error directional derivative worsens at both
resolutions (+.00139 and +.00267).

These are local directional derivatives with fixed component assignments,
frozen critic weights and frozen optimizer geometry. They do not simulate a
future update, account for critic refitting, or establish a passing trajectory.
The squared covariance-error proxy is kept separate from the original
relative-Frobenius covariance metric. The selected scheduled cohort's small
aggregate covariance-direction estimate is sensitive to quadrature resolution;
do not extrapolate that sign.

The [MMD projection check](mmd-projection-diagnosis.json) addresses a missing
prerequisite of the deferred proposal. On one explicitly separate, deterministic
2,048-point panel of the saved control, the archived fixed-bandwidth MMD force
improves the output-space covariance-error derivative (-.46987). After
projection through the generator/prior and frozen AMSGrad denominator, the
derivative instead worsens (+.16685). Adding it to the adversarial force changes
the same-panel projected derivative from -.00658 to +.16027. The sigma-point
panel is an integration approximation; even its adversarial-only covariance
sign differs from the finer quadrature. This is evidence against treating the
earlier output-only force result as sufficient motivation for training, not
a trained rejection of MMD. A second 4,096-point corner panel also reverses
the MMD covariance direction: -.27584 before projection, +.17606 after it.

The [word-state diagnostic](word-saturation-diagnosis.json) enumerates the five
saved prior atoms and five declared words at the acquired and failed states.
At update 6,501, generator logits span roughly -375 to 405 and 458 of 840
float32 probabilities underflow to zero. Four of five atoms attenuate the
adversarial probability-to-logit gradient by a norm ratio below `1e-6`, versus
one at acquisition. That four-atom attenuation also occurs in float64, so
float32 underflow alone does not explain it. The generator-mediated prior
gradient norm is .000431 versus .01975 from the direct joint-latent path.
The generator's total parameter-gradient norm remains nonzero and exceeds its
acquisition value; this is a categorical output bottleneck, not evidence that
every network gradient vanished or proof of what first caused the collapse.

The frozen word fixture optimizes joint adversarial G/E loss with zero prior
regularization. Inverse reconstruction is measured by its grader, without an
explicit reconstruction training term. A logit-space reconstruction objective
would be a new declared recipe intervention; changing the grader's clamp
cannot supply that missing training force. This identifies a focused retention
hypothesis without solving Ring16 or selecting a new global recipe.

## Recommendation and limits

Retain the measured control and negative findings. The 6,400-update schedule
is an existing scoped repair for a different protocol; it is not a replacement
for the original Ring16 requirement. Do not repeat the concluded optimizer,
energy or frozen-prior revisions, combine per-task successes, or interpret
unmeasured cells as scientific failures.

The deferred output-MMD proposal has no trained results. Its saved output-force
motivation does not survive this finite-panel generator/optimizer check. Keep
it deferred. A new repair needs a supported mechanism, one global recipe, a
fresh matched control and its own finite campaign budget. This investigation
opens no paid campaign and changes no production update law.

The original [actual-training GIFs](../r1r2-concluded-archive/README.md) and
[schedule GIF](../ring16-schedule-20261011/r1r2.gif) retain their numerical
results and source identities. No new scientific toy is trained here. The
seven-update oracle is software verification and grants no quality credit.

## Reproduce

Run from the repository root. Saved-state analysis requires the original
local archives; missing originals are an explicit blocker, not a reason to
retrain. The full per-mode output and logs belong outside Git.

```sh
python -m pytest tests/test_r1r2_update_oracle.py tests/test_fixed_critic_penalties.py tests/test_recipe_schedules.py tests/test_forge_taskrecipes.py tests/test_adversarial_losses.py -q
python reports/forge/r1r2-failure-diagnosis-20261022/saved_force_probe.py \
  > /mnt/ml7tb/ParticleGAN-forge/r1r2-failure-diagnosis-20261022/reproduction.json \
  2> /mnt/ml7tb/ParticleGAN-forge/r1r2-failure-diagnosis-20261022/reproduction.log
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-failure-diagnosis-20261022/*.log
```

For the MMD panel check, run `mmd_projection_probe.py` in the same directory;
`--panel corners` selects the second declared integration panel. For word
attenuation, run `word_saturation_probe.py`. Redirect their stdout to that
local artifact directory as above. `summarize_probes.py` verifies the retained
raw core/MMD receipts and appends their resolution comparisons to the two
compact diagnosis JSONs. The commands and exact input hashes in those JSONs
retain their own source and integration identities.
