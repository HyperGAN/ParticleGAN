# R1/R2 protocol and prior-repair audit

Read-only audit of declared settings and published numerical evidence. Zero
training updates, sampling draws, queue claims, regrading or qualification reuse.
The [technique inventory](../technique-inventory.md) remains the only generated
leaderboard for this goal. No replacement recipe is selected here.

The selected recipe is
[`recipe-prior-refactor-r1r2-v1`](../../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json),
candidate revision `594012d6f37dcf9bb8d8c2e9c84085cfcd896d5d44843496901fb5d0019c617a`.
Its recorded source digest is
`a5eafc73e4c4a995cc388b9be00bea5aec289fd35863099e369008e7def0ab14`,
executed commit `a0493ca3c213d559523eadc76e244ec9f9c1f81f`.
The current family report labels its recorded measurement stale against newer
contracts; this audit supplies no current-source qualification.

The current stability view is revision **14**; recorded qualification remains
bound to revision **9**. The inventory's 49-task scientific-verdict projection
shows **5 PASS, 2 FAIL and 42 not measured** for this selected recipe. These
counts are different from repeated per-view requirements and attempted
diagnostics. In particular, 77 diagnostic attempts are not 77 scientific FAILs.
The separate [Tier 2 diagnostic cohort](../tier-coverage-20261014/README.md)
records **9/21 PASS** under its own declaration; its failures remain visible but
cannot be pooled into ordinary prerequisite-qualified cells.

The recipe uses native Adam, base LR .0085 with a .1 cosine floor, beta1 zero,
beta2 .9 to .99, paired R1/R2 coefficient 1 to .1, learned prior locations,
zero prior-regularizer weight and zero additive training noise/EMA. Beta2 and
penalty strength reach their endpoints after 20% of the task-bound horizon.

## Horizon ownership and the actual failure

The current [Ring16 task](../../../configs/forge/tasks/ring16_acquisition.json)
declares **1,600 execution updates and original schedule horizon 400**. The
[public binding](../../../experiments/forge/api.py) deliberately sets the resolved
recipe's `total_steps` from `original_schedule_horizon` when present and rejects
candidate resource overrides. It does not silently choose the execution cap.
Consequently beta2/gamma settle by update 80 and LR reaches its floor by 400,
leaving 1,200 permitted updates at that floor. This distinction is declared
protocol, not a demonstrated adapter defect.

Changing the task's horizon 400 to 1,600 would create a new task/protocol identity.
A reusable trainer schedule policy would instead need its own explicit recipe
delta and host support. Neither may overwrite old failures. The existing
6,400-horizon result below is not evidence for an unmeasured horizon-1,600 recipe.

The frozen task uses 16 equally weighted 2D Gaussians, radius 3 and sigma .1;
256 initial MoG locations, latent dimension 4, batch 128, a width-64/two-layer
host with Fourier setting 2; seed 0 and public deterministic initialization.
It scores 96 clean/live panels of 4,096 public-prior samples and requires five
terminal full passes. The original endpoint finds all 16 modes and passes mass
TV and minimum spread, but **HQ .844238 < .85 and component covariance error
2.066388 > .85**. None of its 96 panels passes the full gate. Occupancy alone
therefore does not diagnose this failure; local distribution quality does.

## Existing duration and schedule evidence

These are frozen archived-source Ring-only diagnostics. Their stored outcomes
are copied below without recomputation. They have no ordinary qualification
credit. The schedule contrast matches actual initial tensors/prior, consumed
target batches and the serialization boundary with its saved duration control.

| Schedule horizon | Execution cut | Stored full verdict | Terminal passes | Covariance error (<= .85) | Min eigen ratio (>= .15) | HQ (>= .85) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 400 | 1,600 | FAIL | 0 | 2.066388 | .223238 | .844238 |
| 400 | 3,200 | FAIL | 0 | 1.260007 | .117109 | .894287 |
| 400 | 4,800 | FAIL | 0 | .815628 | .135986 | .934814 |
| 400 | 6,400 | FAIL | 3 | .525637 | .221104 | .947754 |
| 6,400 | 1,600 | FAIL | 0 | 1.816648 | .548108 | .780762 |
| 6,400 | 3,200 | FAIL | 0 | 5.860979 | .701077 | .229248 |
| 6,400 | 4,800 | FAIL | 0 | .868858 | .276721 | .960205 |
| 6,400 | 6,400 | PASS | 5 | .423491 | .157181 | .977539 |

The 400-horizon endpoint at 6,400 passes every scalar endpoint bound but fails
the sustained rule: only three terminal passes. The published duration readout
records a minimum-spread failure at update 6,350 (.13328 < .15). The stretched
schedule passes with exactly five terminal checks, and only .007181 minimum
spread margin. Its 3,200 cut also fails mode coverage (5 of 16). Its 4,800 cut
fails covariance. These are measured failures rather than unknown cells.

This supports schedule exhaustion as one contributor at the longer budget,
without establishing LR as the sole cause: beta2 and gamma change together.
Every reported 1,600 cut still fails. Neither a longer budget nor a stretched
schedule establishes a repair within the unchanged ordinary task allowance.

Sources: [duration results](../ring16-duration-20261011/results.json),
[schedule results](../ring16-schedule-20261011/results.json) and
[schedule protocol](../ring16-schedule-20261011/protocol.json).

## Do not repeat the concluded search

The [PR #401 archive](../r1r2-concluded-archive/README.md) preserves 34 complete
measured revisions; all fail Ring16. Already tested: constant policies/rates,
critic/prior rate changes, Adam momentum roles, AMSGrad, direction/denominator
changes, spectral updates, local/global sliced transport, non-saturating loss,
weaker gamma, raw signed energy and freezing learned prior locations. The
strongest complete recipes retain five of six ordinary passes plus their own
clock audit. Their endpoints still fail density; per-task winners cannot be
combined into one qualified recipe.

The scheduled baseline's clock failure is expected under its explicit scheduled
claim. It is separate from the ordinary Ring16 blocker. An ordinary acquisition
PASS on Gaussian or words also does not certify its endpoint or later hold.

The [selected-recipe publication](../recipe-prior-refactor/publication/results.json)
makes the word distinction concrete. Ordinary smoke attempt
`8432785e8a6645378778b83f1f457819` has one independently confirmed full pass at
update **2,501**, so its acquisition grade is PASS. Its update-20,001 endpoint
has four words, mass TV .217188, failed exact inverse reconstruction and minimum
reconstruction token probability zero; this does not erase the acquired PASS.
The separate hold attempt `d765aa2da2d1434b9234f7114309c60e` restores the successful
2,501 state, then adds 4,000 updates. It FAILs its original every-check hold
contract, ending with one quality word, quality fraction .383789, mass TV .8 and
inverse NLL 11.006416. Thus retention is a measured failure. Categorical softmax
saturation is supported as a local recovery obstruction by the subsequent
[saved-state diagnostic](word-saturation-diagnosis.json): four of five enumerated
prior atoms attenuate the critic probability force by more than a million-fold
through softmax, in float32 and float64, while direct joint-latent force remains.
The complete generator gradient is nonzero; two endpoints do not identify the
initial trigger or prove a causal repair. Original acquisition and hold grades
remain unchanged.

Constant non-saturating AMSGrad m110 is a stronger **unselected** historical
control: five of six ordinary acquisition/behavioral passes plus its own clock
PASS, with all 24 word acquisition checks confirmed. It still fails Ring16
(covariance 1.556546 > .85 and minimum spread .094075 < .15), and its word
endpoint does not supply an unexecuted later hold result. It is not an automatic
family replacement or a way to combine task-specific successes.

The old campaign is concluded at 238 unique paid attempts / 7,168.248061 seconds
under its original 8,000-second cap. Do not reopen or enlarge it. Output-MMD is
untrained deferred preparation, supported only by limited saved output-force
evidence; native least squares was deferred. Neither is a measured solution.
A new user-authorized campaign would still need a distinct supported hypothesis,
fresh frozen source/control bindings, one global recipe and its own finite cap.

## Pinned inputs

SHA-256 identities at this audit; archived identities use original commit
`2cf7b0a4c796e29ace6cca12d2e553ad67211b5c`.

| Input | SHA-256 |
| --- | --- |
| Selected recipe JSON | `f93f64d4872fc8965ecde70d006f640a3d3c0e2a00aa44a03f4c1ec5c6283b67` |
| Current Ring16 task JSON | `83ae3f6e1ff2bc2230882db72afbb0a2e075e271147382e69ee236b4a8df35f0` |
| Public binding api.py | `c3f00eddbc8466027b4180f66d167467efb36c287688fb2b328aff74f21260cb` |
| Family selection JSON | `35871846445af79feec0775aa398b3c1ad037a6aa1723cccbf56fe73bd85b7ec` |
| Duration results JSON | `c46c30cd31f15e258673a0f3b99135ce4c6b8411bb5e4d0186d7a0f7a266554f` |
| Schedule results JSON | `c34c0580d36e2598b8621105b37302a178bc77d27f5bab56601b438c19301dd0` |
| Archived baseline results JSON | `29ef8926634f453735238ed6385feb42c3a01b271944a4d8b04039a561ab1835` |
| Archived next-mechanism README | `ea867d5d14aa7d20f7cb9f1356ae0c8ccd39a5bd3b6edff9e758a71d1ad451e0` |
