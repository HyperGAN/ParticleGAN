# R1/R2 repair comparison

**Runner defects are fixed; no tested global density repair qualifies.** The
stronger NS/AMSGrad reference passes 10/13 current required Tier 1 tasks.
Removing R2 reduces that to 9/13. Adding explicit joint reconstruction improves
word confidence, but retains those four scientific failures. All three pass
their separate own-checkpoint word hold, including all 25 checks.

Keep R2 and preserve the recorded family selection. The new reconstruction
objective is an opt-in research control, not a qualified replacement recipe.
The [current technique inventory](../technique-inventory.md) remains the only
generated leaderboard. This is one completed, source-bound comparison; no
historical qualification was rewritten.

## Final comparison

| Metric / original gate | NS/AMSGrad reference | R1 only | R1 + joint reconstruction |
| --- | ---: | ---: | ---: |
| Required Tier 1 | **10/13 PASS** | **9/13 PASS** | **9/13 PASS** |
| Separate clock audit | PASS | PASS | PASS |
| Ring component covariance error, <= .85 | 1.556546 FAIL | 3.000381 FAIL | 3.000381 FAIL |
| Ring minimum eigenvalue ratio, >= .15 | .094075 FAIL | .252591 PASS | .252591 PASS |
| Ring mass TV, <= .15 | .049805 PASS | .159180 FAIL | .159180 FAIL |
| DDGAN final projection CDF, <= .12 | .320936 FAIL | .552668 FAIL | .552668 FAIL |
| AE-DDGAN final projection CDF, <= .12 | .330916 FAIL | .267938 FAIL | .267938 FAIL |
| Word first confirmed acquisition update | 834 | 1,667 | 834 |
| Word acquisition endpoint minimum inverse token probability, >= .9 | .999085 | .998599 | .999993 |
| Own-checkpoint word hold | **25/25 PASS** | **25/25 PASS** | **25/25 PASS** |
| Word hold endpoint minimum inverse token probability, >= .9 | .997923 | .997469 | .999969 |
| Worst word hold inverse token probability across primary and independent checks, >= .9 | .994125 | .910063 | .999809 |

The table shows individual endpoint bounds, alongside complete task verdicts.
Passing an endpoint bound does not override the full sustained evaluator.
All word endpoints also retain five generated modes, quality 1 and exact inverse
reconstruction. Hold restores each recipe's earliest confirmed state and trains
4,000 more updates: reference/composed 834→4,834, R1-only 1,667→5,667. Those
different continuation windows are required by the original task; the hold
comparison is not a causal comparison over identical held minibatches.

| Current required task | Reference | R1 only | Composed |
| --- | --- | --- | --- |
| gaussian1d_smoke | PASS | PASS | PASS |
| two_pole | PASS | PASS | PASS |
| unused_token_hold | PASS | PASS | PASS |
| ae_gan_hold | PASS | PASS | PASS |
| ring16_acquisition | FAIL | FAIL | FAIL |
| five_word_joint_smoke | PASS | PASS | PASS |
| zero_tanh_smoke | PASS | PASS | PASS |
| model_ae_ddgan_gaussian_chain | FAIL | FAIL | FAIL |
| model_ae_gan_multimodal | PASS | FAIL | FAIL |
| model_ae_reconstruction | PASS | PASS | PASS |
| model_ddgan_gaussian_chain | FAIL | FAIL | FAIL |
| model_vae_categorical | PASS | PASS | PASS |
| model_vae_hard | PASS | PASS | PASS |

All 12 nonword required trajectories are numerically identical between R1-only
and the composed recipe, including saved model/optimizer/named-stream parts and
observation hashes. The reconstruction coefficient is explicitly inapplicable
there. Separate recipe identities and physical costs remain recorded.

## What the failures establish

R1-only improves Ring's narrowest local spread but nearly doubles covariance
error and fails mass balance. It also collapses multimodal AE-GAN generation
(TV .701416, minimum spread .000854). DDGAN final normalized mean error rises
from .433671 to 2.671887, with covariance eigenvalues changing from
[.231828, 1.441154] to [.099352, 7.088243]. Removing R2 is rejected globally.

AE-DDGAN reconstruction itself passes: reference paired NMSE .009060 and
shuffled gain 219.28; R1-only/composed .014512 and 138.02. Its generation and
conditional density still fail. The [saved DDGAN analysis](../r1r2-local-repair-20261022/ddgan-diagnosis.md)
finds adverse mean-correction fields on seven of eight contexts, but does not
establish temporal critic lag or justify dropping D momentum. High-noise
transition passes are weak evidence for a correct clean predictor.

The [saved Ring prerequisites](../r1r2-local-repair-20261022/README.md) reject
ordinary projected MMD, several local forces and a cheap block-metric proposal.
Accurate output-tangent solves improve some population directions but need
109–134 iterations on original 128-example batches, retain a bad tangent sign,
and lack an affordable global implementation. None was advanced to training.
These are frozen-state analysis cohorts, not new qualification results.

## Implementation and evidence

The public API now supports fixed real/fake penalty weights and an explicit
joint reconstruction objective. Categorical reconstruction uses raw-logit
cross entropy, retaining corrective gradients for confidently wrong words.
Defaults preserve legacy behavior and serialized identities. The three global
recipes, exact deltas, unchanged task contracts and finite budgets are in
[PLAN.md](PLAN.md). The reference reproduces the archived stronger recipe;
it is not a single-factor comparison with the older scheduled incumbent.

Four operational repairs make the current suite reviewable:

- Snapshot the AE provider's exact task-definition dependencies.
- Let native Adam use separate G/E role groups for public model policy binding;
  parameter updates and per-parameter optimizer state remain bit-exact.
- Preserve the provider's original error receipt when execution fails before
  the audit attaches.
- Route the declared zero-tanh final scorer, retaining its four original bounds
  and adding no retention requirement.

[Software validation](software-validation.json) records separate overlapping
suites of 177, 32 and 96 passing tests, plus the successful
[eight-provider frozen-source smoke check](infra-v2-snapshot-smoke.json).
Two independent source reviews found no actionable defect. The original
[update oracle](../r1r2-failure-diagnosis-20261022/implementation-audit.md)
found no loss/sign/half-gamma/native-Adam error. No new seed experiment was run.

[Final metrics and provenance](publication/results.json) preserve all receipts,
certificate/source hashes, full recipe/prior/initialization/budget/sampling
bindings, comparison proofs and [actual-training GIFs](publication/media/).
[Strict final validation](final-validation.json) verifies all 61 GIFs; the five
original model infrastructure failures have explicit media gaps.
All three holds verify their own acquired checkpoint bytes, state, key and
revision. Direct consumed-batch hashes match across candidates on all seven
tasks retaining them; constructor and named-stream proofs retain their explicit
coverage gaps. Ambient process RNG snapshots are distinct from consumed named
streams and are not silently equated across independent processes.

The repaired source is `b33e375de0a1e9cf754f4d0fa24e0fd216db4fec86b1b549ccb5ea2bb0e4f175`,
frozen at commit `df196bf43dc50c166a6e77491fa5fc7a75eae323`. It has 42 ordinary
and six diagnostic physical attempts. The original interrupted source retains
18 attempts, including five model execution failures and the original BLOCKED
zero-tanh grade. All **66 physical attempts cost 2,201.552423 paid seconds**,
charged once against the 24,000-second campaign cap. No scientific retries or
default promotion occurred. Ordinary Tier 2 remains blocked by scientific
failures; word hold supplies only its declared diagnostic claim.

Diagnostic and ordinary acquisition keys differ by evidence-use policy, so
each diagnostic uses a separately paid acquisition receipt; none was silently
substituted. [Admission correction](infra-v2-admission-preview.json),
[original interruption](v1-infrastructure-interruption.json),
[future source bindings](infrastructure-source-binding.json) and
[archived media aliases](media-alias-archive.json) preserve the exact identities.
Bulk logs, traces, checkpoints and state dumps remain outside Git.
The six concluded Forge records retain certified final summaries; their
[byte-exact full readouts](readout-archive.json) are archived locally with hashes
and blob identities. [compact_readouts.py](compact_readouts.py) reproduces that
projection without changing any scientific verdict.

## Recommendation and reproduction

Reject R2 removal as a global repair. Keep reconstruction as an explicit,
tested word inverse control; it improves margin but is unnecessary for the
reference's measured hold gates. Close this finite three-role campaign rather
than promoting a recipe with density regressions. A further global candidate
needs a demonstrated density-force or conditional-field prerequisite and a
feasible whole-update cost before consuming another ordinary comparison.

Use the six frozen studies and commands in [PLAN.md](PLAN.md). Saved-state
drivers and original evidence identities are linked above. Execution artifacts
live at `/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/`; the queue is drained.
The concise local log remains easy to inspect:

```sh
tail -n 60 /mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/logs/progress.log
```
