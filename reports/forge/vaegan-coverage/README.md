# VAEGAN prior coverage with BISPF

One global `bispf-vaegan-settle-v1` recipe passes both declared hard and categorical
VAEGAN toys at all five terminal checks. The matched inventory baseline and
transport-only repair both fail the sustained gates. Six full-budget Forge runs
retain four failures and two passes; no third round was needed. These are scoped
two-mode results, with no whole-suite qualification or public-default change.

<!-- current-comparison -->

| Global recipe | Hard VAEGAN | Categorical VAEGAN | Task passes |
| --- | --- | --- | ---: |
| `bispf-vaegan-settle-v1` | [PASS](media/bispf-vaegan-settle-v1-model_vae_hard.gif) (5/5 terminal; suffix 6) | [PASS](media/bispf-vaegan-settle-v1-model_vae_categorical.gif) (5/5 terminal; suffix 9) | 2/2 |
| `bispf-optimized-quintic-v1` | [FAIL](media/bispf-optimized-quintic-v1-model_vae_hard.gif) (2/5 terminal; suffix 1) | [FAIL](media/bispf-optimized-quintic-v1-model_vae_categorical.gif) (3/5 terminal; suffix 2) | 0/2 |
| `bispf-vaegan-transport-v1` | [FAIL](media/bispf-vaegan-transport-v1-model_vae_hard.gif) (0/5 terminal; suffix 0) | [FAIL](media/bispf-vaegan-transport-v1-model_vae_categorical.gif) (0/5 terminal; suffix 0) | 0/2 |

All measured arms complete 1,200 updates and24 checks; the last five determine
PASS. This table is the single current comparison for `vaegan_coverage`.
Every revision and failure remains in [compact results](results.json).
GPU0 was shared under load; recorded costs supply no speed ranking.

<!-- /current-comparison -->

Every run completes 1,200 updates, 24 fixed numerical checks and 4,096 held-out
inputs per check. All six share the exact scientific source manifest, initial
encoder/decoder/critic/prior hashes and consumed target-batch digest. Origin
commits `c4beb25d6` and `86ee9d9d8` freeze the two rounds; their scientific source
files are byte-identical. [Compact evidence](results.json) retains exact recipe,
source, task, sampling, initialization, runtime, independent grader and artifact
bindings. [Terminal minima and maxima](terminal-summary.json) cover every metric.
The table above is the sole current generated comparison for this research goal.

| Passing task | Maximum reconstruction MSE | Minimum decoder-mean HQ | Maximum decoder-mean TV | Minimum likelihood HQ | Maximum likelihood TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard VAEGAN | .0095024 | .985840 | .014160 | .971191 | .028809 |
| Categorical VAEGAN | .0049299 | .976074 | .023926 | .954834 | .045166 |

These are worst values over updates 1,000, 1,050, 1,100, 1,150 and 1,200. Bounds
are reconstruction MSE ≤.05, each HQ ≥.85 and each TV ≤.15; both independent
sampling laws retain both modes. Shuffling codes increases error by at least
236.28×/452.79× and posterior mutual information stays above .85089/.69348 nats
(hard/categorical). Local latent-noise second-moment ratios are 1.00090/1.00330,
within [.8, 1.2]. Maximum KL identity errors are 7.62e-9/7.03e-7; maximum Gaussian
ELBO identity errors are 4.45e-16/1.34e-15, below the original 1e-5/1e-4 bounds.
The hard model retains genuine constant KL `log(16)`; categorical KL remains
variable. Final reconstruction MSE is .0047111/.0046466, and final actual Gaussian
likelihood HQ is .986084/.970947. Passing suffixes are six/nine checks. The hard
run still has a reconstruction failure at update 900, preserved in the raw
states and actual-training GIF; the original requirement concerns the last five.

The question is whether a variational encoder can retain useful stochastic,
input-dependent reconstruction while independently sampling both target modes.
Tasks retain balanced anchors (-1.5, 0)/(1.5, 0), data sigma .05, a uniform
16-by-2 MoG with fixed local sigma .025, width-32 encoder/decoder/critic networks,
Gaussian observation sigma .03, two training/eight evaluation posterior draws,
seed 0 and the public deterministic initializer. Decoder means and actual
Gaussian likelihood draws have separate gates. Architecture, data law, initial
prior/capacity, sampling, batch streams, update allowance and cadence are fixed
across trainer candidates. Constructor, target, prior identity/noise, posterior,
training noise, evaluation and penalty streams are isolated and checkpointed.

The [archived investigation](../vae-failure-investigation/README.md) linked every
hard endpoint quality miss to rows with low posterior use, while reconstruction
and input information passed. The passing categorical control showed the same
association with fewer misses, so low usage alone never established causation.
The archived CPU hard FAIL and categorical PASS remain under source
`bee9f06ed21695e518c1cb702e12a385ab9389e8` and their original contracts. This
matched CUDA baseline fails both sustained gates despite passing endpoints.
We have not isolated the reason for the difference between those cohorts.

[Larsen et al.](https://proceedings.mlr.press/v48/larsen16.pdf) combine variational
inference and adversarial decoder training; their D0 ablation uses pixel-space
reconstruction. Our Gaussian vector observation model is a scoped variant with
discrete MoG inference. [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558)
motivates examining aggregate posterior/prior mismatch. The projected data-space
repair is our inference from that diagnosis, rather than a reproduction of WAE.

Round 1 activates existing public `Recipe.kinetic_transport_loss`, weight 1 with
32 deterministic projections, on independent-prior generator batches. It matches
empirical projected quantiles, normalized by detached target variance, with
gradients to the generator and sampled prior rows. It adds no labels, target
anchors, samples, RNG streams or persistent controller. The matched baseline
passes only 2/5 hard and 3/5 categorical terminal checks. Transport passes 0/5
for both; its final likelihood HQ is .81909/.68579. Good reconstruction does not
ensure useful independent generation.

The [round-1 terminal diagnosis](round1-diagnosis.json) reads all 20 saved states
with zero updates, using a separately declared CPU row-stratified panel. Baseline
misses are entirely concentrated on rows below 1% posterior use at every terminal
state. Transport also harms higher-usage rows and leaves more low-usage rows:
13–14 versus 11–12 for hard, 8–9 versus seven for categorical. Thus this auxiliary
alone failed to repair the coverage problem. Detailed row arrays stay outside Git.

Round 2 keeps transport and changes exactly three recipe fields:
`lr_anneal_start: .6 → 0`, `lr_floor: 1 → .05`, and
`network_lr_floor: 1 → .05`. This activates the existing cosine schedule from
update zero over the fixed 1,200-update task horizon, explicitly giving up the
baseline's constant-rate claim. Adversarial loss, genuine scaled Gaussian ELBO,
Born-Infeld critic penalty, normalized optimizer, thermal hook, learned prior and
SBNP regularization at .01 remain unchanged. Saved baseline update-1,000 rates
are G/E .012, prior .03 and D .018. Lowering these together tests whether finite
normalized steps keep recoverable rows fluctuating. The cosine form is related
to [SGDR](https://arxiv.org/abs/1608.03983); our single decay has no restarts and
makes no general convergence claim.

The [round-2 saved-state panel](round2-diagnosis.json) still finds 12–13 low-usage
hard rows and nine categorical rows. Equal-row quality remains approximately
.987–.993/.977–.986 across terminal states. The repair settles decoded quality
without balancing posterior row usage. This distinguishes successful generation
from a claim that every prior row becomes well-used by inference.

Recommend this complete recipe for these two declared toys. The next useful
separate causal question is the same settling schedule with transport disabled;
this study does not establish that transport is necessary. Broader mode counts,
within-mode density calibration, indefinite constant-rate stability and the
remaining technique-inventory tasks are unmeasured. No seed repeat, continuation,
outcome-selected checkpoint or default promotion follows.

The [frozen protocol](protocol.json) capped round 1 at four runs/1,200 reserved
seconds and each of at most two substantive repairs at two runs/600 seconds.
Only six runs/1,800 reserved seconds were used from the 2,400-second ceiling.
Recorded initialization/training/evaluation totals 1,301.847 seconds; complete
worker time is 1,367.423 seconds. GPU 0 was shared under load, so these costs
supply no speed comparison. Every task retains a 300-second cap. The
[research diagnostic view](../../../configs/forge/views/vaegan_coverage.json)
confers no ordinary qualification. Source-bound Forge readouts preserve each
exact revision and its failed or passing numerical gates.

Validation: 23 focused public-API, scorer, checkpoint replay, RNG-isolation and
Forge tests pass. [Fourteen oracle/destructive control verdicts](scorer-controls.json)
match expectations: both oracles pass; shuffled reconstructions, missing latent
noise, input-independent posteriors, collapsed prior/likelihood outputs and
missing KL fail their relevant bounds. The optional recorder saves all 25 actual
observations, including the initial state, with complete model/optimizer/policy/
constructor/named-stream state. Recording is read-only and exact continuation is
tested. [Source migration](source-migration.json) retains exact original task
blobs and hashes; no archived result is rewritten or retrained. Export copies
saved GIFs and independent frozen-source grader receipts, adding no training,
sampling or new rendering. Its reporting guard excludes candidate-derived field
ownership from fixed task comparisons while retaining the full binding hash.

The full CI shard found one stale generated-report guard: adding this diagnostic
view had not refreshed the technique inventory’s display metadata. The
[display-only repair](ci-report-refresh.json) refreshes the existing inventory
and family navigation. The exact failing test passes; all nine selected
scientific rows, their original task contracts and all six VAEGAN results stay
byte-exact. No training, sampling or qualification regrading was added.

Reproduce scientific runs from a frozen checkout using the three declared
studies, with an unused external queue root. Planning is read-only:

```sh
python -m experiments.forge --queue-root /outside-git/vaegan-queue \
  plan bispf-vaegan-settle-v1 --study vaegan-coverage-settle-v1
python -m experiments.forge --queue-root /outside-git/vaegan-queue \
  enqueue bispf-vaegan-settle-v1 --study vaegan-coverage-settle-v1
python -m experiments.forge --queue-root /outside-git/vaegan-queue \
  drain --gpus 0 --workers-per-gpu 2 --allow-sharing --campaign vaegan-coverage-round2-v1
```

For report-only recovery from saved artifacts, without training or sampling:

```sh
python reports/forge/vaegan-coverage/publish.py --queue-root /path/to/saved/queue
python reports/forge/vaegan-coverage/analyze_states.py \
  --campaign /path/to/saved/queue/vaegan-coverage-round2-v1 \
  --output /outside-git/new-row-diagnosis.json
```

Bulk logs, complete states and row arrays remain at
`/nvme/dev/particlegan/artifacts/vaegan-bispf-20261019/`.
Tail the combined stream with
`tail -F /nvme/dev/particlegan/artifacts/vaegan-bispf-20261019/stdout.log`.
Each attempt also has its own `queue/vaegan-coverage-round*-v1/*/run.log`.
