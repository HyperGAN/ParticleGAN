# Transition GAN with an observation encoder

The no-argument example and `configs/transition/default.yaml` now select
`encoder_shared_state`, the winning recipe, with a fresh default output path
`results/transition/default`. Named historical configs preserve their original
settings; completed results and their pinned metadata remain unchanged.

This page documents the original toy encoder. The
[Lunar Lander lessons below](#lessons-from-the-lunar-lander-follow-up) describe
the later state-only control architecture and its distinct training objectives.

This experiment asks whether learning to encode real transitions also helps the
original joint sampler. It adds an inference path to the existing MoG/bcap
example, using the repository's deterministic `particle_ae` routing.

```text
Original sampling (same noisy z and observed context):
G1 -> st
G2 -> at
G3 -> st+1

Real input:
E(st, at) -> z -> G1 -> reconstructed st
              -> G2 -> reconstructed at
              -> G3 -> predicted st+1

Synthetic composition:
G1 -> st ─┐
G2 -> at ─┴-> E(st, at) -> z_hat -> G3 -> st+1
```

Each G remains an independent MLP. E also receives the observed geometry, time
and preference class; it never receives the target next state. E chooses one of
1,024 standardized MoG centers and adds a bounded offset:
`z_hat = selected_center + sigma * 3 * tanh(offset / 3)`.
Routing uses temperature .25, summed squared distances and the repository's
straight-through gradient. E's query is layer-normalized, its offset head starts
at zero, and there is no KL penalty or latent-ID reconstruction target.

## The two arms

Both start from the settings of `concat_class8_marginals`: three Gs of width128,
class-input gain8, geometry/time gain1, scalar class-concatenated critics.

- `encoder_separate`: `D_joint(st, at, st+1)`, `D_state(st)`, `D_action(at)`,
  `D_next(st+1)`.
- `encoder_shared_state`: the same encoder/objective, with one `D_state` serving
  both state roles. `D_joint` and `D_action` remain separate.

The shared state critic receives actual time: t for st and t+dt for st+1. It
uses a common state scaler derived from an equally weighted mixture of the two
training-state populations. The conversion first undoes the original per-block
normalization; the common variance includes between-population mean differences.
The joint/action critics and benchmark scaler retain their original coordinates.
This arm therefore tests weight sharing together with the normalization and time
alignment needed to make that sharing meaningful.

## Exact objectives and gradient paths

Let `A(x)` denote the generator's usual adversarial loss:
`joint(x) + mean(state(x), action(x), next_state(x))`.
Let x be the original generated triple and y the composed triple that retains
x's state/action but gets its next state by encoding those inputs and running G3.

```text
L_G,E,prior = (A(x) + A(y)) / 2
            + MSE(G(E(real_st, real_at)), real_triple)
            + MSE(G1/G2(E(fake_st, fake_at)), stopgrad(fake_st, fake_at))
            + raw_particle_spread
```

Both MSE weights are1. Real MSE averages all six normalized coordinates, so it
weights the three two-coordinate roles equally. Synthetic MSE averages the four
state/action coordinates. Real prediction supervision is new relative to the
adversarial-only baseline. The G update reuses its existing real batch; it makes
no additional real-data draws.

The synthetic input remains differentiable: feedback can flow through G3, E,
and the original G1/G2. Only the synthetic reconstruction target is detached.
The selected particle centers also receive decoder gradients; routing uses the
existing straight-through surrogate. The prior spread term is applied once to
the full raw center table. No objective uses the analytic `st + at` identity.

For D, the first half of a batch uses original triples and the second half uses
composed triples, generated without gradients. The real batch, D batch size,
number of updates and per-role bcap penalties stay fixed. Each role contributes
its full Rp logistic plus bcap loss. Shared state weights receive both state-role
losses; generator marginal weighting remains an average over three roles.
The logged `g_terms` describe original-path adversarial terms; `g_loss` is the
mean over the original and composed paths.

## Recipe and comparison limits

The recorded comparison below used the original MoG study settings. Current
callers use explicit MoG fields with the shared winning defaults; the historical
settings and results here are not a claim about that new configuration. Both arms use seed24002, 1,024 particles,
z_dim32, fixed sigma_rel.025 (calibrated sigma approximately.13112), batch256,
28k updates, Rp logistic, bcap cap1/coefficient1/every update, the existing
optimizer groups, raw-center spread, cosine schedule and EMA. E joins G's Adam
parameter group and gets its own EMA; the prior retains its faster optimizer
group. E initialization has a separate seed, identical between these two arms.
No seed-only repeats.

The real draw budget remains14,336,000 training observations plus32,768 for
normalization. These are fresh analytic samples, not a finite scarce dataset.
The encoder adds capacity and computation. Its exact parameter count is reported
separately from G and D; leaderboard encoder entries form an explicitly marked
paired-supervision cohort. Historical generator-capacity and protocol checks
remain in force. GPU wall times are descriptive, not controlled throughput tests.

Action is displacement in this toy, so `st+1 = st + at` is already a perfect
analytic predictor. Learning this through E/G3 is a representation/coordination
experiment, not evidence of learning unknown physical dynamics. This deterministic
encoder also does not model multiple possible outcomes for an identical input.

## Evaluation and running

The original prior-generated benchmark is unchanged and still determines the
leaderboard rank. The same saved references additionally evaluate:

- Real-input next-state Euclidean error, its p95, and state/action reconstruction.
- Synthetic-composition joint SW1, coverage and transition residual.
- Real and synthetic encoder component usage and entropy-effective component
  count, including train/test, interpolation/extrapolation and per-class paired
  errors. Usage counts summarize routing, not latent semantic correctness.

`test_inference.npz` and `train_inference.npz` save the outputs and routing IDs;
`inference.json` and `summary.json` hold the metrics. `final.pt` contains EMA G,
E and prior, the scaler, configuration, and final (non-EMA) D. Restore constructor
settings from the checkpoint configuration. The encoder is used at inference for
the composed/conditional paths only; original prior sampling needs no E.

```bash
.venv/bin/python -u examples/transition_gan.py --config configs/transition/encoder_separate.yaml
.venv/bin/python -u examples/transition_gan.py --config configs/transition/encoder_shared_state.yaml
tail -F results/transition/live.log
```

Completed output folders cannot be reused. For a short smoke check, supply
`--steps 2 --device cpu --out-dir <fresh-directory>`. Full results and interpretation
are recorded in [the leaderboard](../reports/transition/leaderboard/README.md)
and [readout](../reports/transition/leaderboard/READOUT.md).


## Completed round

The shared-state arm leads the original generation benchmark at joint SW1
0.08565, versus 0.09538 with separate critics and 0.10027 for the previous
adversarial-only leader. Coverage favors separate critics (27.7% versus 24.8%).
Real-input next-state errors are 0.01665 and 0.01794 for shared/separate critics;
the synthetic paths have residuals 0.01471 and 0.01322. The full
[readout](../reports/transition/leaderboard/READOUT.md) explains the tradeoffs,
narrow encoder routing, action-response diagnostic and recommended next benchmark.

Reusable verification and frozen action-response diagnostic:

```bash
.venv/bin/python experiments/verify_transition.py results/transition/encoder/encoder_shared_state
.venv/bin/python experiments/audit_transition_actions.py results/transition/encoder/encoder_shared_state --out /tmp/action-audit.json
.venv/bin/python -m unittest tests.test_transition tests.test_trajectory tests.test_transition_encoder
```

The action probe holds state/context fixed and perturbs physical actions. These
inputs leave the training route manifold; the probe is separate from the pinned
prior-generation benchmark. Metrics and leaderboards drove this round's analysis.

## Lessons from the Lunar Lander follow-up

**Choose encoder inputs around the inference task.** The toy's `E(st, at)` is
appropriate for predicting a successor after choosing an action. Choosing the
action itself needs a different input contract. The Lander experiments used
three distinct stages:

| Stage | Control encoder | What learned |
| --- | --- | --- |
| Initial prototype | Reused E_pair with `st, at-1, terrain` | No control adaptation; paired training had used current actions |
| Original imitation fine-tune, 50/50 | Separate `E_control(st, at-1, terrain)` | Expert action MSE updated E_control/G2 from pretrained weights |
| Later state-only probes, also 50/50 | `E(st, terrain)` | Scratch action training of E/G2/prior; detached G1/G3 probes |

For the original successful fine-tune, E_control started as a copy of E_pair.
G1/G3, E_pair, prior, and D stayed frozen. Previous actions were expert commands
during training and the learner's own commands during rollout; the first command
was `[-1, 0]`. Thus the successful fine-tune **did retain at-1**, with an encoder
trained for that input, rather than simply reusing the mismatched prototype.
The [original control guide](gym-control.md) records this experiment. Its joint
GAN counterpart already combined action MSE with joint and marginal critics,
but started from the earlier world-model checkpoint, not the 50/50 imitation
checkpoint. Its control encoder received action MSE; the GAN paths used the
prior and paired encoder.

We subsequently trained a single state-only encoder from scratch, with terrain
context, for the later control loop and first scratch GAN comparison:

```text
st -> E -> z -> G1 -> reconstructed st
            -> G2 -> chosen at
            -> G3 -> predicted st+1

chosen at -> actual simulator -> observed st+1 -> E -> ...
```

All three Gs remain independent networks. Sharing z lets their losses shape a
common representation, but does not enforce that G2's action causes G3's output
in the simulator. This G3 has no alternative-action input. Action-conditioned
prediction and intervention tests would be needed to establish counterfactual
dynamics; good reconstruction or a successful landing does not establish that.

**Partial observations can train the joint discriminator directly.** The
sparse-action Lander GAN uses complete triples from five action-labeled episodes and
state/successor pairs from all 47 expert episodes. Real and fake receive identical
fixed action masks, with the mask and terrain supplied as D context. Masking also
happens inside D so bcap cannot use hidden coordinates. Hidden actions are absent
from training arrays and action normalization uses only available labels. The
mask is not learned, and there is no missingness generator. This brings the
experiment closer to the original masked-observation idea without reproducing
the full MisGAN architecture. Observed successors still carry action information;
masking alone does not establish that the missing joint distribution is identifiable.

**The later GAN objective differs from the toy's synthetic cycle.** Its two fake
paths are `prior -> z -> G1/G2/G3` and `E(real st) -> z -> G1/G2/G3`. Both receive
adversarial losses on complete and action-masked views. All Gs/E/prior train from
scratch for all 2,500 updates, with paired action MSE, state/successor continuous
MSE and contact BCE, plus prior regularization. There is no synthetic composition
loss or imitation-only stage. The marginal arm adds D_action and one shared
D_state with a current/successor role flag, using common state normalization.
That role flag replaces the toy's physical-time context. See the
[Lander GAN guide](gym-gan-control.md) and
[frozen objectives](gym-gan-control-plan.md) for exact weights and gradient paths.

**Better auxiliary predictions did not reliably help the policy.** Before
restoring GAN training, the matched
[sparse non-GAN experiment](../reports/gym/lunar_lander_sparse_action/READOUT.md)
compared detached G1/G3 probes with letting their losses also update E/prior.
The latter improved expert action MSE by 25% and successor MSE by 8.8×, yet
landed 34/50 versus 44/50 for detached probes. Detached means only G1/G3's inputs
were detached; the action loss still trained E/G2/prior. These are historical
non-GAN findings, excluded from the current GAN-only leaderboard.

The [GAN comparison](../reports/gym/lunar_lander_gan_control/READOUT.md) produced
a similar tradeoff: joint plus marginal critics improved several expert-data
prediction and sample-distribution metrics, yet landed 16/50 against 34/50 for
joint alone. On the same learner-state traces, its G3 predictions were worse.
Even on expert data, copying the current continuous state as the next-state
prediction (persistence) scored MSE 0.022230, versus 0.051738 for joint G3 and
0.047770 for marginal G3; learned contact predictions did beat persistence.

These results motivate evaluating each claim separately: prior-generated sample
quality, prediction against persistence, and paired closed-loop landing outcomes.
Use fixed learner-state traces to compare models on identical inputs; errors on
each controller's own visited states mix model quality with state distribution.
The non-GAN and GAN rounds used different test worlds and protocols, so their
landing counts are not a matched GAN-versus-MSE result. The next proposed GAN
test weakens marginal generator pressure to 0.1 while keeping GAN updates active
throughout. Whether that preserves sample quality and improves control remains open.

The subsequent [previous-action scratch GAN](gym-previous-gan.md) retained
`E(st, at-1, terrain)` and action MSE, now sending joint/marginal GAN feedback
directly through that single encoder and all three Gs. With all 47 episodes
labeled, it landed 7/50 on fresh worlds; the original imitation fine-tune landed
50/50 on the same worlds. On identical inputs from the new GAN's own traces,
its lateral-engine agreement with the heuristic was 33.7%, versus 86.9% for the
imitation model. This recipe did not recover imitation performance; the experiment
does not isolate previous-action feedback, pretraining, or loss competition as
the cause. Full [results and diagnostics](../reports/gym/lunar_lander_previous_gan/READOUT.md)
preserve that distinction.


## Visual demo and sharing

[Open the standalone toy viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/demo/index.html) to compare
real transitions, original sampling, synthetic composition and real-input next-state
prediction. Select either encoder, both classes, all train/test geometries and
five saved times. Detail views use matched axes and show consistency errors as
red connectors. Time slices contain independent transitions, not rollouts.

The page includes the architecture diagram and a copy-ready description with
measured results and limitations. The [ZIP bundle](../reports/transition/demo/transition_demo.zip)
contains that offline HTML plus PNG/PDF toy plots and PNG/SVG/PDF architecture.
The static toy plot discloses its interpolation scene/class; the interactive view
also exposes the extrapolation failures. No new training was required.
