# Independent four-mode MisGAN and archived coupled-state results

> **Current execution is v2 and has no learned-quality measurement.** The code now gives data, mask and imputer critics independent public optimizer, penalty, loss and thermal histories. The shared generator/prior policy is explicitly anchored to the data critic. The original summed-critic backward and generator/prior update order remain; unsupported multi-game mechanisms refuse execution. Complete v2 checkpoints restore in a fresh object and reject v1 or malformed auxiliary state before mutation.
>
> **All numerical results below belong to archived coupled-state v1.** V1 shared one critic control history across the three games, so its measured PASS is not evidence for the corrected independent-game execution. Original grades, metrics, GIFs and source identities are unchanged. No campaign was rerun and v2 inherits no PASS. [Exact v1 task](archive/task-coupled-v1.json), [v1 source](https://github.com/255BITS/ParticleGAN/tree/2025ca2875517cd3cdf8a91bba1ecfeebf3ba7ca), and [v2 integration receipt](../pr-integration-20261109/pr-462.json) retain the boundary.

The archived v1 transport-and-settling repair **passed** its scoped task:
1,200 updates, 24 post-update scoring observations and all five terminal checks
pass every unchanged bound. Endpoint generator HQ is 0.9890, mass TV 0.0940,
mask TV 0.0106, imputation accuracy 0.9727, ambiguous-query HQ 0.9373 and branch
TV 0.0781; branch coverage is 1.0 and observed-coordinate error is zero. The
[current leaderboard](LEADERBOARD.md) retains all three earlier failures and
links all four actual-training GIFs. Their sharper endpoints did not satisfy
the required mass and terminal-window gates.

This is an explicit four-mode research diagnostic using the selected
`bispf-optimized-quintic-v1` recipe and declared deltas. It grants no ordinary
qualification or PASS on the unchanged historical 100-mode recovery/imputer
tasks. Its affine rank-two generator is a scoped architectural assumption, not
a general nonparametric identifiability claim. The gates measure support, mass,
manifold recovery and conditional branch behavior; they do not certify
within-mode covariance or full noise-density calibration.

The legacy public MisGAN fixtures used plain Adam and a scalar recipe loss,
omitted the Born-Infeld penalty, never optimized the prior or applied SBNP,
and omitted consumed RNG streams from checkpoints. Their published failures
retain their original source and execution identities. The new host binds
public recipe optimizers, three Born-Infeld games, learned prior/SBNP,
direction-blend backward and UpdatePolicy. Its initialization, data, training
noise and functional evaluation streams are isolated; complete checkpoints
include consumed training/constructor states, model modes and gradients.

The [frozen proposal](proposal.json) declares the target, architecture, bounds
and finite budget. Four noisy corners are lifted to eight sensors. The original
block-mask patterns and 40/30/20/10% probabilities remain. A single sensor sees
only the sum of two coordinates: central queries retain two equally probable
branches. The complete training matrix is discarded after forming observed
entries and masks. Training never receives complete real labels, analytic
posterior probabilities or evaluation examples.

The original comparison requested the selected BISPF baseline configuration, an observed-only
masked mixture/mask likelihood auxiliary, and that same auxiliary plus
own-generator synthetic component classification. The neural conditional logits
learn from component IDs sampled by their own generator. At serving, the imputer
receives only observed entries and masks, samples its neural categorical
probabilities and decodes independent MoG noise. It never computes component
distances or performs ppost resampling. Training and serving use the same MoG
kernel; the adversarial categorical gradient uses a declared straight-through
Gumbel estimator.

The [MisGAN paper](https://arxiv.org/abs/1902.09599) and
[official imputation code](https://github.com/steveli/misgan/blob/master/src/misgan_impute.py)
motivate independent data, mask and imputer games. These new architectures and
auxiliaries are declared extensions. The
[first-round diagnosis](round1-diagnosis.json) records that narrow likelihood
sharpens support without correcting uniform component counts. It does not
establish synthetic classification as superior to the other arms.

The bounded second round replaces narrow masked-mixture likelihood with the
public deterministic projected-quantile transport loss, applied separately
within actual observed-mask cohorts. It reuses existing fake and observed rows,
with no new data, labels, RNG draws, sampling reweighting or clean target access.
This follows the distribution-matching idea in
[Deshpande et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.html),
using this repository's finite deterministic frame. It retains mask likelihood
weight 1 and synthetic neural classification weight 5. A global cosine schedule
runs from update 0 to 1200 with network/prior floor 0.05; initial rates, all BISPF
core mechanisms and the task remain fixed. This is a composite repair, so its
outcome cannot isolate transport from settling.

All arms use seed 0, public deterministic initialization, the same architecture,
initial MoG64 in R2 with fixed width .025, observed batch sequence, sampling,
1200 updates and 24 scoring observations. Five terminal checks must pass every
bound together: mode/mass/manifold recovery, mask law, observed identity,
overall imputation accuracy and ambiguous branch quality/mass/coverage/spread.
The [compact results](results.json) retain each final and terminal bound,
recipe/source identities, model and batch hashes, guards, costs and artifact
receipts. The [analytic conditional oracle and destructive scorer controls](scorer-controls.json) reject averaging, noise ignore,
wrong branches/mass, complete-generator collapse and mask collapse.

Validation: 128 focused API, scorer, ownership, checkpoint/replay, RNG, BISPF,
prior and Forge diagnostic tests pass. Another 66 compatibility checks compare
inactive training, state, continuation and receipts with an independently
archived develop tree and check the maintained component packet contract.
CI caught a maintained inventory count (194 versus the newly registered 195),
stale publication summaries and three newly added zero-valued Recipe fields
leaking into old component receipts. The fixes preserve the old packet fields,
omit only the new inactive defaults and refresh display-only publications.
Frozen historical catalogs and scientific verdicts retain their original identity.
The source migration preserves exact prior task blobs, code hashes and complete
revision ancestry; no scientific experiment was rerun for these software fixes.

Reproduce the first comparison from executed source `2054e69cf` in a fresh
worktree and queue; the three `*-reviewed` shared-queue subscriptions were
cancelled before training and must not be replayed as the executed study:

```sh
python -m experiments.forge --queue-root /tmp/misgan-r1-queue enqueue bispf-optimized-quintic-v1 --study misgan-bispf-baseline-r1-snapshot
python -m experiments.forge --queue-root /tmp/misgan-r1-queue enqueue bispf-misgan-observed-v1 --study misgan-bispf-observed-r1-snapshot
python -m experiments.forge --queue-root /tmp/misgan-r1-queue enqueue bispf-misgan-amortized-v1 --study misgan-bispf-amortized-r1-snapshot
python -m experiments.forge --queue-root /tmp/misgan-r1-queue drain --gpus 0 --workers-per-gpu 3 --allow-sharing --campaign misgan-bispf-snapshot-repair-r1
```

Reproduce the final repair from executed source `21efa29ae` in a fresh worktree:

```sh
python -m experiments.forge --queue-root /tmp/misgan-r2-queue enqueue bispf-misgan-transport-settle-v1 --study misgan-bispf-transport-settle-r2
python -m experiments.forge --queue-root /tmp/misgan-r2-queue drain --gpus 0 --allow-sharing --campaign misgan-bispf-transport-settle-r2
```

The complete local archive is
`/nvme/dev/particlegan/artifacts/misgan-bispf-20261019`; tail `progress.log`.
Raw stdout, metric streams, JUnit/CI logs, observations and complete checkpoints
stay outside Git. Six cancelled zero-spend subscriptions and three pre-update
snapshot errors remain in the [infrastructure repair receipt](infrastructure-repair.json).
The errors' charges are included once. The
[source-only fingerprint migration](source-fingerprint-migration.json) preserves
original task Git blobs and code SHA identities without changing scientific
conditions or archived results. GPU 0 is shared; elapsed costs are not speed
benchmarks.

The archived v1 recommendation was `bispf-misgan-transport-settle-v1` for this one
explicit task. Its minimum terminal ambiguous HQ remains above 0.93 and its
largest terminal mass TV is 0.0940, leaving only 0.0060 margin to that bound.
No robustness or seed claim follows. The useful next comparison is an explicit
larger-target variant using the same observed-only transport and neural
amortization principles; original 100-mode qualification remains outstanding.
Do not select separate recipes for recovery and imputation or credit this result
to either transport or settling alone.

Total charged elapsed execution is 2,307.489 seconds, including the three
pre-update infrastructure errors exactly once. The first-round paid ledger was
1,860.101 seconds; the final full reservation was 1,800 seconds within the
predeclared 7,200-second goal allowance. All scientific runs completed without
cancellation or shortened training.
