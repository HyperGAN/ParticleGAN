# BCAP convolution support and four-image results

**All four previously unsupported image tasks now complete training. All four
still fail their sustained numerical gates.** The per-offset DualNorm extension
for Conv2d and ConvTranspose2d was merged into develop in
[PR 352](https://github.com/255BITS/ParticleGAN/pull/352) before execution.
[Final metrics, audits and archive identity](readout.json) preserve the four
source-bound results. The original [Tier 2 readout](../bcap-tier2/README.md)
retains its **6 PASS, 11 FAIL and 4 INCOMPLETE** under its original source.

## What happened

Each task runs its unchanged 600 updates and 24 checks, every 25 updates.
Quality must be at least `0.9`, all target modes must have sufficient quality
mass, and **the last five consecutive checks must pass**. An endpoint pass or
an earlier passing stretch does not satisfy that sustained gate.

| Task | Final quality | Final qualifying modes | Passing checks | Terminal passing checks / required | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| [stripes2](receipts/img_stripes2.json) | 1.00000 | 2/2 | 12/24 | 1/5 | FAIL |
| [bars4](receipts/img_bars4.json) | 0.65625 | 2/4 | 0/24 | 0/5 | FAIL |
| [blobs4](receipts/img_blobs4.json) | 0.96875 | 3/4 | 0/24 | 0/5 | FAIL |
| [intensity2](receipts/img_intensity2.json) | 0.96875 | 2/2 | 4/24 | 4/5 | FAIL |

**Stripes acquires the target, then loses it.** Every check from update 275
through 525 passes. Quality then falls from `1` at 525 to `0.03125` at 550
and `0.375` at 575, recovering to `1` at 600. Final mean RMSE is `0.001838`,
but the final passing suffix contains only one check. This is a retention
failure after successful acquisition.

**Bars misses quality and coverage.** Final mean RMSE is `0.067936`, but only
`21/32` centers qualify overall, below the `0.9` quality gate. Quality mass
fractions are `[0.28125, 0.34375, 0, 0.03125]`; all four modes require at least
`0.125` each. One component is absent and another is sparse. Distribution TV
is `0.46875`. No scheduled check passes both gates.

**Blobs produces accurate images with one weak mode.** Final quality is
`31/32` and mean RMSE is `0.008317`. Quality mass fractions are
`[0.28125, 0.25, 0.09375, 0.34375]`; the third component has only three
qualifying centers against the four required by `min_mode_fraction=0.125`.
Four centers are assigned to that component, so it is underrepresented in
quality mass, rather than absent. No check covers all four qualifying modes.

**Intensity reaches the target too late for confirmation.** Updates 525,
550, 575 and 600 pass. Final quality is `31/32`, both modes qualify, mean
RMSE is `0.040409`, and assigned mass is balanced. Four terminal checks are
one short of the required five within the declared budget. This does not
establish that an extra check would pass; the run is not extended or regraded.

The study's auxiliary prediction, stripes endpoint quality `>=0.9`, is
observed. Its endpoint prediction is narrower than the task's sustained gate
and does not turn any FAIL into a PASS.

## Implementation and fixed comparison

The only trainer delta from the selected BCAP configuration is
`Recipe.optimizer_convolution="per_offset"`. The
[kernel update](../../../docs/dualnorm-convolution.md) applies a channel-matrix
polar factor separately to each group and spatial offset, scaled by
`sqrt(out_channels/in_channels)/(kernel_height*kernel_width)`. Transposed
convolution uses its actual input/output storage layout. Dense, vector and
sampled-prior updates retain their existing rules. The layer update bound
does not imply convergence of adversarial training.

The single global recipe retains G/E `.012`, D `.018`, sampled-prior `.03`,
smoothing `1e-5` and zero momentum. **Learning rates remain constant; there is
no annealing.** Publication checks the declared schedule at all 601 clock
positions and verifies initial and endpoint optimizer-group rates and kernel
metadata from each saved checkpoint. These are declaration/endpoint audits,
not an invented per-update LR trace.

The [candidate](../../../configs/forge/ideas/bcap-dualnorm-convolution-v1.json),
[ready study](../../../configs/forge/studies/bcap-convolution-images-v1.json),
[diagnostic view](../../../configs/forge/views/bcap_convolution_images.json)
and [frozen plan](plan.json) bind all four unchanged tasks. Architecture,
target law, batch sequence, prior, clean/live enumeration, budgets, cadence
and gates retain their contracts. Seed is `0`, with the public deterministic
initializer and checkpointed named RNG streams. The task-owned prior is a
32-center learnable particle cloud with `sigma=0`, explicitly retained for
clean enumeration; the candidate's general MoG default is not used here.
Every task completes 600
generator, discriminator and prior updates, with finite checks and zero
unintended RNG deviations.

This capability-repair diagnostic retains the original Tier 2 task definitions.
Diagnostic Tier 1 scheduler placement only permits execution of the selected
subset; it does not promote tasks or qualify this new source. The original
six Tier 1 passes, eleven other Tier 2 failures and four setup errors keep
their original source identities. No Tier 1 run repeats and no Tier 3 run
follows. The [current technique inventory](../technique-inventory.md) remains
the sole goal leaderboard; this new source does not fill its ordinary cells.

## Evidence and cost

Four attempts perform **2,400 host updates** and charge **100.266944 worker
seconds**, against a declared 7,200-second cap. Both A6000 GPUs run one worker
each. All reservations are released; there are zero scientific retries, seed
repeats or tuning runs.

All **96 already-scored sample sets** reproduce their numerical metrics offline,
with exact discrete quantities and floating-point tolerance for CPU/GPU RMSE.
The actual-training GIFs show all 32 clean outputs and target templates:
[stripes](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-convolution/media/img_stripes2.gif), [bars](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-convolution/media/img_bars4.gif),
[blobs](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-convolution/media/img_blobs4.gif), [intensity](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-convolution/media/img_intensity2.gif).
[Renderer receipts](media/index.json) bind saved observations and GIF hashes.
Rendering prohibits learned-model construction, forward calls, training and
sampling; it adds zero optimizer updates or sampling draws. Images are not
used to select or grade results.

The ignored local archive `artifacts/bcap-convolution-images-v1.tar.gz`
contains 1,276 byte-verified original files: logs, metrics, checkpoints,
saved outputs, original envelopes, queue state and frozen source. Its SHA-256
and member digest are recorded in [readout.json](readout.json). Git contains
compact receipts, metrics, media, provenance and reproduction sources.

Implementation validation completed [527 distinct passing checks](validation.json)
and an independent 12-case numerical layer-bound review, including grouped,
rectangular, expanding/narrowing, strided and transposed convolutions. Those
checks establish implementation behavior; the trained image gates remain FAIL.

## Recommendation

Retain the opt-in convolution implementation: it fixes the demonstrated
construction failure, and all four tasks now train. These results do not
support a passing global BCAP recipe or public-default adoption.

A next bounded comparison can test a smaller **constant global base rate**,
preserving role multipliers and smoothing, against this exact configuration
on one shared source. That hypothesis targets stripes' observed oscillation;
intensity's late acquisition could become slower, so endpoint quality alone
cannot select it. Bars and blobs also require mode coverage. Requalify the
shared-source Tier 1 recipe before ordinary Tier 2 qualification. No further
experiment is launched by this report.

## Reproduction and logs

Executed develop commit: `515e2de01f0198d2e58899869a698a2af0f03c72`.
Scientific source digest:
`2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`.
Candidate revision:
`dd0702302681db384f1f1a23e10bfd29a40bae4ccbfea5e01e872d8a4d1a9061`.

Execution commands, from that checkout and recorded runtime:

```sh
python reports/forge/bcap-convolution/run.py --stage plan
python reports/forge/bcap-convolution/run.py --stage enqueue
python reports/forge/bcap-convolution/run.py --stage drain --gpus 0,1
tail -F runs/bcap-convolution/drain.log
tail -F runs/forge/events.jsonl
```

The completed archive needs no further training. The publication-only fixes to
`publish.py` audit the public recipe defaults and saved checkpoint schema;
they change no execution source or scientific result. Offline publication uses
that reviewed publisher and the archived evidence at its recorded paths:

```sh
python reports/forge/bcap-convolution/publish.py
python reports/forge/bcap-convolution/link_readout.py
python -m experiments.forge compile --summaries-only
```

Archive creation is immutable; restore evidence into an isolated checkout with
an unused archive destination to reproduce publication.
