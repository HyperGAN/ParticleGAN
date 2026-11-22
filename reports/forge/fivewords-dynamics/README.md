# Five-word saved-state dynamics diagnosis

**Movement of the encoder/prior correspondence can directly break a passing
word solution. Local adversarial rotation is also present, but this diagnosis
does not identify it as the sole cause of the historical hold failures.**

The clearest intervention holds G, D and all prior rows fixed. At the passing
smoothed truncated/threaded endpoint, one ordinary E-only normalized update
reduces the minimum correct reconstruction-token probability from **.999492 to
.038852**, while generation is unchanged. That failure requires neither a
changing opponent nor minibatch noise. The joint adversarial objective can still
have a nonzero E gradient at a state that passes the reconstruction test.
The unchanged joint GAN objective **improves**, .694092→.693229, during that
actual E-only move. The immediate failure therefore cannot be explained solely
as a loss-increasing optimizer overshoot: the local critic surrogate favors a
code movement that harms the inverse goal. Hard normalization governs its pace.

## Frozen scope and comparison

This user-requested diagnostic starts from develop
`83b099d1e4330dda953d5fce4f68ce00f75fa9a6`. It consumes eight saved **update 20,001
endpoints**, with their original model, optimizer, recipe and RNG packets, from
[word PR #343](https://github.com/255BITS/ParticleGAN/pull/343) and
[smoothing PR #344](https://github.com/255BITS/ParticleGAN/pull/344).
Intermediate passing observations have output tensors and model hashes, but no
recoverable model states. No historical training is rerun to reconstruct them.
The [protocol](protocol.json) and [reproduction source](diagnose.py) were committed
before model-update probes at `31e239097`. The imported numerical runtime is the
exact smoothing execution commit `34db5abbf164229ee449d2862bba53e043da7f82`.

Every cohort keeps its own matrix truncation, autograd threading and smoothing
settings. All retain the selected BCAP/DualNorm recipe: constant rates G/E .012,
D .018 and prior .03, momentum 0, prior regularization 0; five learned 2D cloud rows,
sigma 0; G 2→64→128→168, E 168→128→64→2 and D 170→256→128→1.
The original batch 256 training and 24 observations retain their original verdicts.

The diagnosis instead enumerates **all 25 real-word/prior-row Cartesian pairs**
with equal mass. Pairing each word with the same-numbered prior row would be
incorrect for nonlinear RpGAN: the original IDs are independent. Role probes
apply the actual public optimizer, D first where enabled, for eight bounded
updates from a common endpoint; masks are all roles, G, E, prior, D, and
G/E/prior with D frozen. This is **T(E[gradient])**, not the expectation of
stochastic normalized updates **E[T(gradient)]**. No sampling/noise draws occur.

The exact five-atom generation law is scored through the retained public clean
served model. Equal repeats of each atom form the scorer's deterministic 1,025-row
input; they are **not 1,025 independent draws and grant no ordinary 1,024-sample
credit**. Paired inversion still uses the five correctly matched words. The
[compact readout](readout.json) records this separate scope and original metrics.

| Saved endpoint | Original five-check hold | Exact endpoint goal | G/E/prior with D frozen: passing probes /8 | All roles: passing probes /8 | Raw projected skew / symmetric |
|---|---|---|---:|---:|---:|
| Unsmoothed, truncated / serial | FAIL | PASS | 7 | 8 | Not identified: zero G step |
| Unsmoothed, full / serial | FAIL | FAIL | 0 | 0 | .108 |
| Unsmoothed, truncated / threaded | FAIL | PASS | 5 | 8 | 8.191 |
| Unsmoothed, full / threaded | PASS | PASS | 5 | 8 | 10.481 |
| Smoothed 1e-4, truncated / serial | FAIL | FAIL | 0 | 3 | .298 |
| Smoothed 1e-4, full / serial | FAIL | FAIL | 0 | 0 | .0248 |
| Smoothed 1e-4, truncated / threaded | FAIL | PASS | 0 | 6 | Not identified: zero G step |
| Smoothed 1e-4, full / threaded | FAIL | FAIL | 0 | 0 | .823 |

This is the sole generated comparison for this goal, not a new technique ranking.
All four passing endpoints lose the full generation/inverse goal within eight
G/E/prior updates when D is frozen. The full learner preserves or recovers each
of those endpoints over this short probe. The smoothed serial current-default
failure also recovers in the full probe at step 6. **Eight deterministic updates
are not evidence of sustained historical retention or a qualified solution.**

## What the interventions identify

**Changing representation is directly demonstrated.** With D frozen,
unsmoothed truncated/threaded E-only updates first fail at probe 6; by probe 8
minimum correct token probability is 1.55e-15, while generated support remains
perfect. Smoothed truncated/threaded E-only updates first fail at probe 1; encoder
outputs move by up to .104 in latent space, and by probe 8 the minimum correct
probability is 4.46e-37. G, D and the prior were fixed. Conversely, unsmoothed
full/threaded prior-only updates first fail at probe 7: generation quality falls
to .6 by probe 8 while inverse reconstruction remains exact with token probability 1.
Coverage and inverse correspondence can fail independently.

A retrospective reduction of saved latent coordinates computes exact optimal
one-to-one encoder/prior matching across all 120 permutations on CUDA. RMS
matching distance is divided by minimum prior-row separation, avoiding any
assumed word↔row label or common latent scale. The four passing endpoints start
with normalized distances **.326, .470, .376 and .330**, respectively; passing
reconstruction does not imply E(word) equals a prior row. After eight frozen-D
G/E/prior probes they become **1.081, 1.677, 2.425 and 1.145**. Full learner probes
instead finish at **.180, .200, .299 and .230**. This supports continuing joint
alignment pressure and shows that a frozen critic can give stale guidance.
The matching statistic is descriptive; it is not a task bound or proof that
geometric nearest rows uniquely determine nonlinear reconstruction.

**The local surrogate/output-goal tension is measured directly.** A separately
frozen [zero-update supplement](supplement-protocol.json) evaluates all eight
endpoints and the retained first E-only codes, holding original G/D/prior fixed.
It interpolates latent codes at t = 0/.1/.25/.5/.75/1 and evaluates exact Cartesian
losses and inversion. Every actual t = 1 E-only endpoint reduces the joint loss.
In the passing smoothed truncated/threaded case, loss decreases monotonically
while correct-token probability becomes .992511 at t = .25, **.899206 at t = .5**,
.375204 at t = .75 and .038852 at t = 1. A smaller move can preserve the output gate
locally, but reducing the surrogate continues to move toward a bad inverse.
Only t = 0/1 correspond to actual encoder parameter states; the intermediate
latent line is decoder/objective geometry, not an interpolated parameter path.
The [supplement readout](supplement-readout.json) records all 48 points and exact
t = 0/1 reconstruction matches, with no optimizer updates or sampling draws.
This establishes local tension at the probed state, not incompatibility of the
objectives everywhere or a chronic full-game cause.

**Normalization fixes the pace despite unequal gradient magnitudes.** At all
four passing endpoints G's raw gradient after the original D-first move is only
7.53e-18, 7.27e-14, 1.27e-12 and 3.67e-13, while E remains roughly .0032–.0331 and
prior .0011–.0059. Actual E parameter-step norms are .0465, .0465, .1684 and .0462;
prior aggregate step norms are .0671, .0671, .0671 and .0670. Thus a saturated,
nearly stationary G is accompanied by continuing E/prior motion. These E/prior
signals are nonzero; this is not a finding that their updates contain only
numerical noise. The matrix epsilon skip also matters: the current unsmoothed
and smoothed truncated/threaded endpoints have exactly zero G parameter motion,
so the optimizer does not normalize every arbitrarily tiny gradient into a full
matrix step.

The failed unsmoothed full/serial endpoint illustrates a different amplification:
G raw gradient norm 1.25e-8 becomes an actual parameter step norm .1555, about
1.25e7 times the raw norm. The smoothed full/serial G step is .000828 at raw norm
5.42e-6. These are separate saved states, not a same-state causal smoothing
comparison; cross-state changes cannot be attributed solely to smoothing.
The coefficient 1e-4 largely damps the very weak G signal while E/prior continue
to take sizable steps. The role probes make that unequal motion consequential,
but do not establish whether reducing E/prior movement alone would repair the
full original learner. A passing output gate is not a joint-game equilibrium.

**Local rotation is measured, not inferred from an oscillating metric.** The raw
field is F_i=∇θ_i L_i, with every role minimizing its own loss; D includes the
original BCAP and G/E/prior use the unchanged joint adversarial objective. For
four disjoint unit parameter directions from the first actual float32 public
step, the probe estimates B_ij=e_iᵀ(∂F_i/∂θ_j)e_j. Central finite differences
use CUDA float64 at h = 1e-3 and 3e-4. This is a four-dimensional projection of the raw
field, **not the full Jacobian or the Jacobian of the normalized update map**.
S=(B+Bᵀ)/2 and A=(B−Bᵀ)/2 separate local symmetric and antisymmetric response.

All six nonzero-direction cases agree between scales to relative Frobenius
error below 1.86e-6, against the declared .05 tolerance. Unsmoothed
truncated/threaded and full/threaded passing endpoints have ||A||/||S|| = 8.19 and
10.48, with complex eigenpairs .00252±.02695i and .000420±.006325i. Those directions
contain substantial local adversarial circulation. The failed endpoints have
ratios below 1; symmetric curvature dominates this particular projection.
Their off-diagonal couplings still contain antisymmetric response, so this does
not prove rotation is absent in other directions. Two zero-G-step cases have no
four-direction basis and remain explicit nulls. Extremely small nonzero G
steps in other passing states also limit interpretation of that basis direction.

This resembles the game geometry motivating
[optimism](https://arxiv.org/pdf/1711.00141) and
[extragradient](https://arxiv.org/pdf/1802.10551), but neither a local complex
pair nor frozen-D damage establishes which intervention would fix stochastic
20k-update retention. There is no single-player optimum assumption here.

## Recommendation and limits

Prioritize **magnitude-sensitive E/prior motion and joint code alignment** in the
next bounded investigation. Record paired reconstruction, generation confidence
and permutation-independent code matching throughout hold. To separate step
overshoot across other parameter directions from harmful surrogate guidance,
retain both the joint objective and inverse confidence in the next intervention.
The supplement already identifies that tension at the smoothed passing endpoint.
Stabilization should preserve correspondence while the players jointly align;
a smaller E/prior step can slow movement without making its direction safe.
Keep one fixed global recipe and constant learning rates.
A direct paired reconstruction constraint or anchor is one justified structural
option: the current joint-host training loss has no direct reconstruction term.
That option changes the training objective and would require explicitly declared
mechanics and comparison controls. Merely slowing E/prior movement does not
resolve local surrogate/goal tension. Local rotation keeps extragradient relevant;
it is not a complete diagnosis of the frozen-D failures. No next training comparison is launched by this report.

The evidence identifies sufficient causes in explicit endpoint interventions,
not the entire historical collapse mechanism. Endpoints differ across trajectories;
there are no saved model states around the intermediate collapse. Exact finite
support eliminates minibatch variation but changes the normalized stochastic
update law. Role freezing changes the game. Float64 finite differences measure
local raw-loss geometry, whereas public optimizer probes use original float32.
The four-role projection does not survey all network directions. No default,
threshold, task qualification, search outcome or Tier2 credit changes here.

## Fidelity, cost and reproduction

The original endpoint packet has completed_steps 20,001 while its recipe schedule
horizon is 20,000. The pinned public UpdatePolicy loader rejects that count before
any probe. A documented saved-state workaround temporarily supplies count 20,000
to the loader, immediately reinstates 20,001, and checks the **entire policy packet
digest**, including models, optimizer state/critic record, aliases and every
policy stream, plus Forge's named streams and the data stream. Original packet
bytes are unchanged. No policy begin_step or schedule evaluation occurs. This
is an endpoint loader limitation, not a tensor conversion or a resumed-training
claim. The separate acquisition/hold implementation repairs this horizon/budget
check in the shared API.

A first preflight also rejected the post-report smoothing checkout because
optimizer docstring bytes differed from executed 34db5abb; a detached checkout
of the exact execution commit passed the source checks. Both corrections occur
before model-update probes; all eight diagnostic endpoints execute once. Raw
preflight logs, source receipts and checkpoint inputs are retained.

The run consumes **384 diagnostic logical updates, zero historical-training
updates, zero sampling draws and 13.560 measured seconds**, below the declared
900-second cap, on physical GPU1 (RTX A6000). The separately frozen supplement
adds 48 zero-update forward points in .806 seconds, below its 120-second cap. Every model, gradient, optimizer,
finite-difference and code-matching calculation runs on CUDA; the inherited
NumPy word scorer processes frozen outputs as in the original contract. The
bilinear/skew and quadratic/symmetric CUDA controls pass with absolute finite
difference error below 1e-10; all eight restore, expected-pair-loss and RNG checks
pass. The supplemental t = 0/1 reconstruction endpoints match all retained
probabilities exactly. No broad training tests are rerun for this report.

The [historical goal GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/fivewords-dynamics/media/archived-smoothed-truncated-serial.gif) illustrates
actual scored training from PR #344; [its receipt](media/receipt.json) retains that
source identity. It is not a new probe animation. Bulk inputs, per-probe saved
coordinates, logs and source packets remain outside Git in the local archive
pinned by [archive.json](archive.json); every member is reopened and byte-verified.
The [publisher](publish.py) reproduces compact metrics without model updates.
Original training source commits/digests and task declarations are retained
separately from the shared frozen diagnostic runtime.

To reproduce the closed diagnostic, restore PRs #343/#344's original input archives
and use an empty output directory. Checkout 34db5abb as a separate runtime root:

```sh
git worktree add --detach /tmp/ParticleGAN-dynamics-runtime \
  34db5abbf164229ee449d2862bba53e043da7f82
CUDA_VISIBLE_DEVICES=1 /home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/fivewords-dynamics/diagnose.py \
  --runtime-root /tmp/ParticleGAN-dynamics-runtime \
  --unsmoothed-root /path/to/bcap-word-regression \
  --smoothed-root /path/to/smooth-polar-factorial-v1 \
  --output runs/forge/fivewords-dynamics-reproduction/diagnosis \
  > runs/forge/fivewords-dynamics-reproduction/diagnosis.log 2>&1
tail -F runs/forge/fivewords-dynamics-reproduction/diagnosis.log
```

The supplementary readout can be reproduced from that retained raw readout,
with no optimizer updates:

```sh
CUDA_VISIBLE_DEVICES=1 /home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/fivewords-dynamics/supplement.py \
  --runtime-root /tmp/ParticleGAN-dynamics-runtime \
  --raw runs/forge/fivewords-dynamics-reproduction/diagnosis \
  --unsmoothed-root /path/to/bcap-word-regression \
  --smoothed-root /path/to/smooth-polar-factorial-v1 \
  --output runs/forge/fivewords-dynamics-reproduction/latent-line-readout.json
```

For compact reduction and an archive of retained outputs, run the publisher:

```sh
CUDA_VISIBLE_DEVICES=1 /home/martyn/dev/ParticleGAN/.venv/bin/python \
  reports/forge/fivewords-dynamics/publish.py \
  --runtime-root /tmp/ParticleGAN-dynamics-runtime \
  --raw runs/forge/fivewords-dynamics-reproduction/diagnosis \
  --unsmoothed-root /path/to/bcap-word-regression \
  --smoothed-root /path/to/smooth-polar-factorial-v1 \
  --archive artifacts/fivewords-dynamics-reproduction.tar.gz \
  --gif /path/to/original-smoothed-truncated-serial.gif
```

These commands are reproduction guidance, not an automatic continuation of the
concluded study. The exact original inputs, complete recipe/RNG packets, source
bindings and endpoint null cases remain in the published receipts.
