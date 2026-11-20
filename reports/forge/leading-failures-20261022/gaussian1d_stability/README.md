# BISPF Gaussian stability failure investigation

The selected `bispf-optimized-quintic-v1` fails to stay settled both before and
after the target moves. This is broader than slow post-shift transport: its mean
already reaches the allowed shifted-target interval at update 4,084, but it never
maintains the required distribution. Every failed scheduled observation fails
the CDF gate, including many observations with acceptable mean and width.

This investigation analyzes retained public-API training evidence with **zero
new updates and zero new sampling draws**. It preserves the original FAIL and
the [single current leaderboard](../../technique-inventory.md). Its 18 PASS / 11
FAIL / 20 unmeasured selected full-suite cells are a different denominator from
the optimized recipe's completed original 30-task study (19 PASS / 11 FAIL).

## Exact question and evidence

Continue the recipe's own passing 1,000-update Gaussian smoke checkpoint without
resetting optimizer histories or streams. The target is initially `N(2, 0.5²)`;
at update 4,001 it becomes `N(3, 0.5²)`. Require all 72 stationary observations,
the last five observations before the update-5,000 recovery deadline, and all 24
subsequent shifted-hold observations to pass. A matched frozen pre-shift model
is a negative control. The ordinary gate is stronger than five endpoint passes.

The [task declaration](../../../../configs/forge/tasks/gaussian1d_stability.json)
matches the executed declaration. Each observation uses 4,096 public-prior
samples, live weights, clean output (MoG kernel noise retained), finite fraction
1, mean error at most 0.2 target standard deviations, width ratio in [0.8, 1.2],
and empirical CDF KS at most 0.05 against the analytic target CDF.

| Binding | Recorded identity |
| --- | --- |
| Candidate revision | `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def` |
| Executed source | `d9d72d53761c898bce20102e1995fc3d899c1447` |
| Source digest | `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213` |
| Attempt | [`6141a1c953cc41189a24685cf8080743`](../../technique-receipts/6141a1c953cc41189a24685cf8080743.json) |
| Initial conditions | Seed 0, public deterministic orthogonal initialization, 256 learned MoG locations, sigma 0.1, fixed width and uniform weights |
| Host | Two-hidden-layer, width-32 scalar MLP G/D; critic Fourier-2 input; latent dimension 2; batch 128 |
| Recipe | Non-saturating loss; BISPF penalty; zero-momentum DualNorm, smoothing 0.001; LR 0.012, D multiplier 1.5, prior multiplier 2.5; learned prior with SBNP weight 0.01 |
| Budget and runtime | Own 1,000-update prefix plus 5,000 continuation updates, 600-second allowance; RTX A6000 / Torch 2.14.0 CUDA; 89.604 recorded worker seconds for this attempt |

The original readout is retained at
`/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`,
SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`.
That matches the original archive identity in the
[quality publication](../../bispf-optimized/quality.json), despite its older
`/home/martyn/dev/ParticleGAN/...` location. The audit hashes the original before
reading its 120 active and 48 frozen observations; compact summaries alone are
not substituted for those histories. [analysis.json](analysis.json) retains the
bindings, source file hashes, aggregate findings and original artifact hashes.

## What the saved numbers establish

| Phase | Passing / required checks | CDF failures | Mean failures | Width failures | Longest passing run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stationary, updates 1,042–4,000 | 6 / 72 | 66 | 35 | 6 | 2 |
| Recovery window, 4,042–5,000 | 2 / 24; last-five gate FAIL | 22 | 8 | 3 | 1 |
| Shifted hold, 5,042–6,000 | 2 / 24 | 22 | 12 | 7 | 1 |
| Frozen negative control, 4,042–6,000 | 0 / 48 | 48 | 48 | 0 | 0 |

Failures overlap within a check; counts must not be added as independent trials.
The recovery window requires its final five checks, not all 24. Only update
4,959 passes among those final five; update 5,000 fails CDF KS at 0.09705.

The stationary mean error reaches 0.830 sigma and CDF KS reaches 0.373.
Shifted-hold width ratio ranges from 0.659 to 2.876, while the frozen model's
width ratio stays between 0.863 and 0.900. The active model learns the shifted
location, but repeatedly leaves the permitted distribution. The frozen model
remains near the old location and fails all shifted checks, so simply stopping
all learning cannot solve the whole question.

At update 6,000, mean error 0.05474 sigma and width ratio 0.87651 pass, but CDF
KS **0.07032 > 0.05** fails. Mean and width do not characterize the full law.
This does not identify a particular skew, tail or modality defect without the
saved sample tensors. The older problem-list figures, 5/72 and 0/24 for
`bispf-v2`, must not replace this selected recipe's 6/72 and 2/24.

All state remained finite; all G/D/prior counters reach 6,000; 121 evaluation RNG
audits report zero unintended deviations. Continuity records an exact restore of
the recipe's own passing smoke state, no history reset, and matched frozen draws.
The FAIL is numerical, rather than a timeout, missing prerequisite or RNG error.

## Mechanisms checked and limits

Source inspection uses the **executed commit**, because current `bispf.py` and
`dualnorm.py` have since changed. The frozen
[recipe factory](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/recipes.py)
selects `GANLoss("non_saturating")`; `SinkhornRelativisticLoss` is selected only
for the separate `loss="relativistic"` setting. Sinkhorn remains active inside
the BISPF penalty/cooler, not as this recipe's adversarial pairing loss.

The frozen
[BISPF penalty and hook](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py)
enable continuous scalar transport chords and state-dependent thermal gradient
scaling on this host. Its special late two-dimensional density rule is
inapplicable to scalar outputs. The
[DualNorm update](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/optim/dualnorm.py)
uses smoothed polar updates for dense weights and normalized directions for
biases/prior rows. The nominal recipe rates remain high even after a smoke PASS;
thermal state and smoothing affect the actual move. Continued motion is a
plausible mechanism to inspect, **not an established causal attribution**.

The quintic optimization is restricted to eligible convolution groups; this
dense scalar host does not execute it. The ordinary quality publication records
exact equality to the older incumbent's endpoint metrics. Blaming the quintic
convolution delta for the Gaussian failure is unsupported.

The archive readout retains hashes for initial, pre-shift, frozen and final
states and saved samples, but those tensor files were not found at their
declared paths or in the inspected mounted study tree. Thus this PR cannot
measure the actual thermal history, per-update gradient direction or cause of
the oscillations. Missing state is an artifact-retrieval task, not permission
to repeat unchanged training. The existing
[actual-training GIF](../../bispf-optimized/quality-media/gaussian1d_stability.gif)
and its [receipt](../../bispf-optimized/quality-media/gaussian1d_stability.json)
retain this exact attempt and original observed-sample hash; metrics above
determine the findings.

## Next investigation and checks

First retrieve the exact state and sample files listed in `analysis.json` and
verify their digests. With **zero optimizer updates**, inspect the saved
1,000/4,000/6,000 states and 120 observed sample sets for distribution shape,
thermal scale and adversarial versus SBNP gradient contributions. Keep the
restored training streams unchanged. A successful readout must identify a
specific measured update mechanism before proposing a new structural recipe;
the present evidence supports failure to settle but cannot choose that recipe.

The [earlier research controls](../../test-dispositions-20261015/README.md#gaussian1d_stability)
include Gaussian passers that lose ring16 or mode hold, and an idle-rule idea
that still failed Gaussian stability and regressed acquisition. These are
separate recipes and source cohorts. Any later repair must declare one global
trainer delta, including its prior policy and regularizer, and preserve the
unchanged Gaussian, ring16 and mode-hold gates before broader qualification.
No new run or altered gate is declared here.

Reproduce the dependency-free audit from the repository root:

```sh
python -u reports/forge/leading-failures-20261022/gaussian1d_stability/audit.py \
  --readout /mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json \
  --check
```

`audit=PASS` means the retained analysis matches, while the original scientific
gate remains FAIL. Run stdout may be redirected to an ignored `runs/` log and
tailed with `tail -F`; no bulk execution history is committed.
