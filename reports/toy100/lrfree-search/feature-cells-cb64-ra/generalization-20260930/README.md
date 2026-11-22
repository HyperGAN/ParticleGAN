# ParticleGAN Atlas: qualification and convergence

<!-- superseded-banner: generated -->
> **Superseded — historical record.** Current status: [reports/forge/technique-inventory.md](../../../../forge/technique-inventory.md).
> Results below are preserved as recorded; statements about "current" selections, defaults,
> leaderboards, queues or GPUs describe the time of writing.

Atlas adds local population checks and guarded reopening to
[PR155’s E22](https://github.com/255BITS/ParticleGAN/pull/155) at `cabe2084`.
It is implemented in [PR223](https://github.com/255BITS/ParticleGAN/pull/223).
Use [`atlas.json`](../../../../../configs/100gaussians/atlas.json) with your
own models, initialization, population, latent width, batch size, and data
stream. The [illustrated guide](../../../../../docs/atlas.md) explains E22,
the changes, their advantages, and their limits.

**Atlas passes all 19 original quality gates and the Toy gate.** On the original
MNIST fixture it improves the recorded embedding metrics over current PR155
E22; MNIST has no separate numerical acceptance threshold. The latest merged
source passes the full CUDA-visible suite and strict checkpoint replays.

## Formulation and use

E22 already learns the latent particle table alongside the generator, adapts
rates using optimizer history, balances population mass in critic features,
checks support, and reopens after optimizer shocks. Output noise, critic
regularization, averaged serving, and caller-owned training APIs are shared.

Atlas adds temporary regions fitted from recent real critic features. Its
eligible feature-cell path checks regional counts and conditional raw-output
averages, with separate count, placement, and real-anchor support actions.
Placement repair uses checked row reallocation/cloning within groups. Birth
proposals create latent codes tested through G; they do not copy real outputs.
Capability selection requires enough particles and at most eight raw output
coordinates. Other shapes and custom/routed models retain their existing paths.

The optional settled guard qualifies optimizer reopening with a prior network
contraction witness and rebases known critic-objective transitions. After an
actual reopen, mean correction can use occupied frozen groups; missing groups
have zero direction and weight, without redistributing their mass. CPU
optimizer isolation, explicit CPU planning, alias-preserving checkpoint
transfer, and atomic shape validation repair the diagnosed execution failures.

See the [implementation guide](../../../../../docs/feature-cells.md) for
selection, the empirical one-quarter generator/noise base-rate calibration,
action budgets, serving coherence, and the existing `E22Policy` API.
`ra13-settled.json` remains a byte-identical historical configuration alias.

## Current PR155 E22 comparison

The baseline is freshly executed **current PR155 E22**, including optimizer
reopening and anchor release. Both recipes use the original Toy/MNIST models,
initialization, data streams, fixed seed, scorers, and 2,000-update budget.
Atlas retains its original fresh execution labels and checked source bridges.

| Original final metric | PR155 E22 | Atlas |
| --- | ---: | ---: |
| Toy noisy-sample precision ↑ | 71.55% | **96.53%** |
| Toy covered modes | 25/25 | 25/25 |
| Toy mass total variation ↓ | 0.28455 | **0.05211** |
| Toy original quality gate | FAIL | **PASS** |
| MNIST active embedding Fréchet distance ↓ | 1.88097 | **0.54449** |
| MNIST embedding precision ↑ | 76.76% | **86.91%** |
| MNIST embedding recall ↑ | 71.92% | **84.72%** |
| MNIST confident class coverage | 10/10 | 10/10 |
| Static MNIST reopen events | 1, recorded after 202 completed updates | **0** |

Toy precision increases 24.99 percentage points. MNIST precision increases
10.16 points and recall 12.79 points. MNIST uses kNN for both recipes; this
does not demonstrate image-scale feature cells. The recipe comparison includes
all selected controls and calibration, rather than isolating one cause.
[Comparison report](mnist/pr155-e22-current/REPORT.md),
[pinned metrics and inputs](mnist/pr155-e22-current/comparison-closure/COMPARISON.json).

## Watch learning and distribution-shift recovery

![Atlas and current PR155 E22 learning the rotating 100-Gaussian distribution](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/visualization-pr223-convergence/render/final-media-1/atlas-vs-e22-convergence.gif)

[Full-resolution MP4](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/visualization-pr223-convergence/render/final-media-1/atlas-vs-e22-convergence.mp4)
· [Poster](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/visualization-pr223-convergence/render/final-media-1/poster.png)
· [Capture validation](visualization-pr223-convergence/closure-v1/DATA-RECEIPT.json)

Fresh `rotated100` runs start from the same particles and models. The target turns 30°
after updates 500 and 1,000. The animation uses **153 actual observations per
method**: 4,096 generated points every ten updates, plus two target-jump frames
that hold the model fixed. The local zoom reveals individual Gaussian fits;
the synchronized curves show the drops and recovery. Sample clouds
are not interpolated.

The curves use separate 4,096-point diagnostics. Filled checkpoint markers
and endpoint scores use the unchanged original **20,000-point acceptance
draws**, shown below. The common 90% visual guide is illustrative; original
shift gates require at least 95 modes and quality at least 90% of each
method’s own update-500 baseline.
Quality is the fraction of noisy generated points within 0.09 of a target
center (three Gaussian standard deviations).

| Original 20k draw | PR155 E22 quality / modes | Atlas quality / modes |
| --- | --- | --- |
| 500, initial fit | 88.10% / 100 | **96.09% / 100** |
| 1,000, after first shift | 83.99% / 100 | **96.09% / 100** |
| 1,500, after second shift | **94.73% / 100** | 92.59% / 98 |

**Both methods pass both original shift gates.** Atlas improves the initial
and first-shift endpoints by 7.99 and 12.10 percentage points. E22 finishes
2.14 points higher and covers two more modes. The fresh paired capture closes
302 observation-preservation checks and reproduces Atlas’s retained original
checkpoint state, excluding only observational elapsed-time metadata.

On the 4k diagnostic, Atlas first reaches the shared 90% guide at update 220
initially and 290 updates after the first shift; E22 does not reach it within
either 500-update phase. Both first reach it 330 updates after the second
shift. These are recorded crossing times, not a speed or scaling-law claim.

## Original quality qualification

| Original task group | Atlas |
| --- | --- |
| Portability | 13/13 PASS |
| Static native Gaussian grids | 3/3 PASS |
| Moving native Gaussian grids | 3/3 PASS, both turns |
| Toy25 learned model | PASS, 25/25 modes |
| Ring-shift diagnostic, 4,600 updates | PASS; final HQ 99.51%, 8/8 modes |

PR155 E22 already reports the 13/13 portability, 3/3 static native, and 3/3
moving passes. Atlas preserves those pass counts. Historical development
failures are retained as provenance, not used as the current E22 baseline.
The [sealed qualification](release-prep/final-v4-attempt1/QUALIFICATION.md)
and [machine-readable receipt](release-prep/final-v4-attempt1/QUALIFICATION.json)
keep their original execution labels and comparator identities. The earlier
learned comparator there disabled reopening; the fresh current-E22 comparison
above supplements that immutable record.

## Latest source verification

| Check | Actual result |
| --- | --- |
| Full suite with CUDA visible | 1,502 passed, 12 skipped, 18 subtests passed |
| Required CUDA routing/readback and portability cases | All four executed and passed |
| Toy/MNIST checkpoint continuation | 40 update calls PASS: 10 per task and load path |
| CPU contracts | 104 PASS, CUDA uninitialized |
| CPU versus CUDA default-device diagnostic | 40 update calls PASS |
| Upstream CPU CLI smoke | PASS |

The full suite ran in 403.52 seconds. Its 12 skips comprise missing optional
Gymnasium/torch-fidelity dependencies, opt-in CIFAR/native research gates,
and two assertions requiring CUDA to be absent.
[Suite receipt](validation-ra17/FULL-TESTS.json),
[test log](https://github.com/255BITS/ParticleGAN/blob/bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f/reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/validation-ra17/full-pytest.log),
[JUnit](https://github.com/255BITS/ParticleGAN/blob/bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f/reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/validation-ra17/full-pytest-junit.xml),
[replay closure](mnist/ra17-replay/CLOSED.json).

The `run_full_suite.py` receipts above ran before CUDA tests became opt-in
(2026-10-25). Today's `pyproject.toml` deselects `cuda`-marked tests, so a
rerun must add `-m "cuda or not cuda"` to the `pytest.main` arguments to
reproduce the same selection ([docs/testing.md](../../../../../docs/testing.md)).

Replays compare losses, semantic training state, and RNG streams after every
update, and primary sample bytes at the endpoints, for native and CPU-mapped
checkpoints restored onto CUDA.
Only the original observational birth-evaluation duration is excluded from
state comparisons; the native branch also matches the pinned original control.

## How earlier fresh runs apply to the latest source

The qualification preserves **15 fresh RA14 quality executions and four fresh
RA15 executions**, rather than relabeling them as RA17 training runs. Toy and
MNIST retain their fresh 2,000-update RA13 execution labels. The new dense
moving comparison freshly executes both current packages.

- RA14 repairs checkpoint transfer only. RA15's occupied-group recovery is
  inactive on retained zero-fire or kNN runs. All three moving tasks and the
  ring-shift task were rerun in full because they exercise recovery.
- RA16's explicit CPU factories preserve arithmetic under the original CPU
  default; shape validation accepts the original valid checkpoints. Source
  comparisons, reaction/replay contracts, and actual CUDA default-device
  checks verify this bridge.
- RA17 integrates current PR155. Lazy diagnostics and batched routing preserve
  checked valid arithmetic. The noise-floor change is inactive throughout all
  21 retained quality/learned fixtures: lifetime table-stationarity counts are
  at most two, proving table scale stays at least 0.25, above the 1/64 floor
  boundary. This is a lifetime source proof, rather than an endpoint inference.
  The latest full suite and CUDA checkpoint replays run the merged source.

See the [RA16 bridge](portability/ra16-portability/REPORT.md),
[RA17 bridge](portability/ra17-current-pr155/REPORT.md), and
[noise-floor applicability proof](diagnostics/upstream-noise-floor-applicability/REPORT.md).
The tested package SHA is
`500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61`;
the shared configuration SHA is
`a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`.

## Recommendation and scope

Use Atlas for the recorded tasks and as an explicit starting
point with caller-owned models. Feature-cell selection is qualified for these
low-dimensional tasks; MNIST's kNN result does not establish image-scale
feature-cell behavior. Extra checks cost computation: Toy took longer with
Atlas, while MNIST took less time in the recorded separate runs. Larger models
still need their own memory, throughput, and quality measurements. Fixed-seed
results do not establish scaling laws or robustness over seeds.

The archive retains failed candidates, preparation errors, the RA13 checkpoint
alias failure, and the RA14 rotated-grid failure. Sealed receipts keep their
original statuses. Raw checkpoints, datasets, and sample clouds stay local,
with paths and hashes in the manifests. The published media exports actual
observations; the explanatory infographics use schematic dots. Earlier sparse
visualizations remain intact.
