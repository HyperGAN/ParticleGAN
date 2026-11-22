# One hidden layer acquires the Gaussian, then loses it

**The shallow architecture passes Tier 1 smoke and fails Tier 2 stability.** Two
independent same-state observations confirm the full Gaussian law at updates 584
and 792. The old five-terminal acquisition gate still fails. This supports the
requested separation between basic acquisition and continuous learning; it does
not establish convergence or a stability repair.

This is the depth-only arm alongside the separate Fourier-removal investigation.
It builds on [PR322](https://github.com/255BITS/ParticleGAN/pull/322), which adds the
shared public-API smoke/stability runner. The original depth 2 ordinary Gaussian
architecture remains the main-view default: its separate unchanged baseline
already passes confirmed smoke. Depth 1 remains an explicit diagnostic variant.

| Declared question | Result | Evidence |
| --- | --- | --- |
| Tier 1: any full-law passing state by 1,000, independently confirmed | **PASS** | 584 and 792; 2/24 primary checks |
| Original five terminal acquisition checks | **FAIL** | Terminal suffix 0 |
| Tier 2: retain quality through 4,000 | **FAIL** | 4/72 stationary holdchecks; longest streak 1 |
| Tier 2: mean 2→3 reacquisition by 5,000 | **FAIL** | 0/24 reacquisition checks |
| Tier 2: retain shifted law through 6,000 | **FAIL** | 0/24 shifted holdchecks |
| Own 4,000-state frozen shift control | **FAIL** | 0/48 checks; zero updates |

This diagnostic table is not a second qualification leaderboard. The repository's
single generated [technique inventory](../technique-inventory.md) owns the goal
comparison. [Compact results](results.json) and [saved-observation summary](saved-observation-summary.json)
retain the full declared grades and overlapping failure counts.

## What changed

Only both networks' hidden depth changes 2→1. The generator is Linear 2→32,
LeakyReLU(.2), Linear 32→1 (**129 parameters**). The critic retains x, sin(πx),
sin(2πx), cos(πx), cos(2πx), then Linear 5→32, LeakyReLU(.2), Linear 32→1
(**225 parameters**). The original depth 2 networks have 1,185/1,281 parameters.

Target N(2,.5²), z 2, width 32, learned 256-location uniform MoG sigma .1,
standardize False, initial prior scale 1, batch 128 and seed 0 stay fixed. Public
named deterministic orthogonal initialization remains the policy. Changed
network shapes necessarily produce different initial network tensors; the exact
prior tensor and all training stream states match the original depth 2 baseline.
The original ordinary sigma .025 acquisition evidence remains its separate,
unchanged historical cohort.

The whole BCAP recipe remains alternating, relativistic logistic loss, BCAP
coefficient/cap 1, dualnorm networks and sampled-row normalized prior updates.
Rates remain G .012, D .012×1.5 and prior .012×2.5 throughout; both schedule floors 1.
There is no annealing, EMA, output noise, prior regularizer, magnitude cap,
extrapolation or target-informed fixture.

Every full-law observation uses 4,096 finite clean live public samples, mean
error ≤ .2 target sigma, std ratio [.8,1.2], and analytic CDF KS ≤ .05. The first 1,000
updates complete all 24 scheduled primary/independent confirmation pairs. Training
continues to 4,000 with every 72 later stationary check required for Tier 2. At 4,000
the mean shifts 2→3; five terminal full passes by 5,000 and all 24 remaining checks
through 6,000 are required. Evaluation never stops training or changes its RNG.

## What the trajectory shows

At 584 the primary/confirmation KS is .03361/.03823 and std ratio .88745/.88189;
at 792 KS is .04069/.04145 and std ratio .98661/.98964. These independently confirmed
hits answer the smoke question. By 1,000 KS is .16577 and std ratio 1.29377, so
acquisition does not persist even within the short budget.

All 68 failed stationary holdchecks fail KS; 30 also fail mean, 12 width below .8,
and 12 width above 1.2. The counts overlap. Mean error ranges .00221–1.38085 target
sigma and width ratio .54837–2.79517. All 48 shifted checks fail KS; the final
6,000-update KS is .12542 and width ratio .74054. Simplifying depth alone leaves
large changes in distribution quality while the learner continues.

The [original controls](controls.json) remain source-bound and retain their
original five-terminal FAIL and 1/72 stationary hold result. Their trajectory and
this arm are not a statistical depth comparison: one deterministic initializer
policy necessarily produces different shapes/tensors, and no robustness or seed
study was conducted. The useful conclusion is acquisition capacity with either
architecture, followed by failure to settle under the unchanged trainer. It does
not isolate whether the generator, critic or moving prior causes the drift.

## Execution, repair and verification

The [original protocol](protocol.json) froze at commit
`326f0e0b82d014dfcd728cd8f5719b2450fc627a` before training. After 1,000 updates the
shared host's strict parent artifact check found its receipt added after the
payload manifest had been captured. No stability update had executed.

The original raw prefix, receipt and all payload bytes remain untouched. The
[explicit migration](migrate_prefix.py) copies the three certified payloads into
a separate complete evaluator tree; its working receipt changes only the artifact
location. Strict verification still rejects extra/missing/changed certified files.
The [continuation amendment](continuation-protocol.json), frozen at
`6f0375011a50e70e505ccd556767e211b82a6246`, binds those exact prefix identities and
the software artifact/compatibility repair. It executes only the remaining 5,000
updates with the same recipe, task, model/optimizer state and RNG streams.

Actual cost is **6,000 new updates**, 768,000 real examples and 45.754935 measured adapter
seconds (7.909566 prefix + 37.845369 continuation), within the 720-second reservation.
There are zero repeated prefix updates, scientific retries or follow-on tuning.
Adapter timing includes sampling/scoring and some I/O; it is not a speed ranking.

[Six exact CUDA restores](restore-proof.json), [219 saved sample-set metric
recomputations](verification.json), both grades, both 9-frame actual-training GIFs,
and every finite/mechanism/RNG/actual-optimizer-count guard pass. Forge validate
passes. The [data/stream audit](data-and-stream-proof.json) matches original real
batches and training/primary-evaluation streams exactly at 4,000 and 6,000, including
the entire 4,000-update stationary digest and 2,000-update shift digest. Constructor
streams differ only as required by the new shapes; confirmation uses its separately
checkpointed stream. All neural training, sampling and context restores use CUDA.
Original CPU target generation, saved-sample scoring and media rendering remain
explicit exceptions preserving the matched law.

[Smoke GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-shallow/smoke.gif) · [Stability GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-shallow/stability.gif) · [Numerical curves](metrics.svg)
· [Provenance and archive identity](provenance.json)

Raw logs/checkpoints remain local in `runs/api/gaussian-shallow-v1/` and
`runs/api/gaussian-shallow-v1-continued/`. The deterministic artifact archive is
`artifacts/gaussian-shallow-v1.tar.gz`, SHA256
`319cbec56850ed05711a983d336ee204834c933ec34d6168fd1e5098e8e36837`.

```sh
tail -f runs/api/gaussian-shallow.run.log
tail -f runs/api/gaussian-shallow.continuation.log

# Read-only verification of the completed CUDA contexts and saved numerical draws.
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python -m benchmarks.toy_audit.gaussian_architecture_publish \
  --raw runs/api/gaussian-shallow-v1-continued --output reports/forge/gaussian-shallow \
  --verify-state --device cuda:1
```

Reproduction sources are the two frozen task cards, whole candidate card, public
shared runner, protocol/amendment and byte-exact migration/resume wrappers, all
retained at their frozen commits and inside the local archive. The original
wrapper deliberately refuses changed numerical source; the continuation wrapper
binds the exact archived prefix instead of starting a fresh smoke run.

Retain this as a positive smoke diagnostic. Keep depth 2 as the ordinary default,
move stability to Tier 2, and stop this exact depth arm. The authorized parent
integration will rerun the required inventory and newly eligible Tier 2 families
under the revised view; these standalone diagnostic receipts supply no automatic
whole-view qualification.
