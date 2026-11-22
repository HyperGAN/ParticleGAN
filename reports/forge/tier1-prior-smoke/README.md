# Tier 1 ring16 and scalar Gaussian: prior diagnosis

**No prior configuration repairs both original tests.** All 18 GPU runs fail
the full sustained gates. A wider MoG helps scalar acquisition, while the ring
smoke projection favors the existing small table and narrow kernel.

This user-requested task-only diagnostic tests particle count, cloud versus MoG,
and MoG width on the two failing acquisition questions. It is based on develop
`7183d65d`. It grants no ordinary Forge qualification, changes no public default,
and preserves every archived result and the provisional calibration status.
The later [user-requested duration follow-up](../tier1-prior-duration/README.md)
finds a full ring pass at 1,600 updates and scalar regression at 4,000; it retains
this original-budget study unchanged.
The existing [current technique leaderboard](../technique-inventory.md) remains
the only leaderboard for `discriminator_stability`.

## Frozen comparison

[Protocol](protocol.json) declares nine prior configurations: 256, 1,024 and
4,096 learned rows, crossed with public `ParticlePrior` (sigma zero) and
`MoGParticlePrior` at fixed sigma .025 or .1. Locations use the public deterministic
R2 initializer, with the current Forge init_std=1 and uniform mixture weights.
Standardization is false. A cloud is an explicitly different finite-atom cohort.

Every configuration uses the exact [selected BCAP-pure recipe](../dualnorm-pacing-v2/DEFAULT_SELECTION.md):
dualnorm, G step .012, D/G 1.5, prior/G 2.5, momentum zero, relativistic loss,
fixed real/fake BCAP coefficient/cap 1, no spread regularization or additive
training noise, and constant schedules. **Trainer delta: none.** Task-owned
particle count and prior law are the only experimental changes. Network
architectures, latent dimension, target/data law, batch size 128, named seed-0
streams, training budgets and evaluation cadence remain fixed.

Scalar training uses 1,000 updates and at most 120 seconds per attempt; ring16
uses 400 updates and at most 300 seconds. The finite round contains 18 cells,
12,600 updates and 3,780 reserved seconds, with no retries or continuations.
Training and model/prior sampling require CUDA and refuse CPU fallback. Target
batch construction and the frozen scientific scorers retain their explicit CPU
reference law, then batches move to the GPU. Changing those reference draws
would confound the comparison. CPU work here is data generation, grading and
media rendering, not neural training.

Clean live public sampling supplies 4,096 samples at all 24 original checkpoints;
at least five consecutive terminal checks must pass. Initial illustration draws
preserve the evaluation stream. Constructor, target, training-noise and evaluation
streams are isolated and checkpointed. Publication checks identical data-sequence
and initial network hashes across all arms, identical initial locations within
each particle count, and a common trainer recipe after removing the two declared
task-owned resource/prior fields.

## Smoke versus distribution quality

Original full gates stay unchanged. A separately frozen **proposed smoke
projection** tests scalar location and width, or ring coverage, precision and
mass balance, at the same five terminal observations. It does not certify scalar
CDF shape or per-component covariance. Its bounds are existing acquisition
bounds, with quality requirements omitted rather than thresholds fitted to runs.

Independent controls verify both questions before training. Full gates reject
same-moment scalar uniform/two-atom laws and sixteen exact center atoms. The
reduced smoke question intentionally accepts these shape impostors. Both reject
point collapse to one location, a shifted scalar law, doubled scalar width and
an incorrect ring radius. This is explicit scope reduction, not evidence that
the original full-quality failures were false. Oracle controls validate scorers;
they do not calibrate a qualification profile.

## Reproduction and logs

Use the frozen scientific commit recorded in the completed receipt. From its
repository root, in the project Python environment, choose a fresh ignored path:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -m cuda -q tests/test_tier1_prior_smoke.py
mkdir -p runs/api
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -u -m benchmarks.toy_audit.tier1_prior_smoke run \
  --device cuda:0 --output runs/api/tier1-prior-smoke-v1 \
  > runs/api/tier1-prior-smoke-v1.runner.log 2>&1
tail -F runs/api/tier1-prior-smoke-v1.runner.log
# Each start event names its per-task log, with live numerical observations.
# Render with Python 3.12 / Torch 2.14 / Matplotlib 3.11 in the final PR checkout.
RENDER_PYTHON=/home/martyn/dev/ParticleGAN/.venv/bin/python
$RENDER_PYTHON -m benchmarks.toy_audit.tier1_prior_smoke publish \
  --raw runs/api/tier1-prior-smoke-v1 \
  --output reports/forge/tier1-prior-smoke
python reports/forge/tier1-prior-smoke/analyze.py \
  --raw runs/api/tier1-prior-smoke-v1 --published reports/forge/tier1-prior-smoke
```

Raw stdout, curves, tensors and complete checkpoints remain local. Publication
verifies hashes and renders nine actual saved observations per GIF, with the
target law alongside outputs; it adds no updates or model sampling draws.
Numerical verdicts use all 24 observations, not the nine displayed frames.

## Completed readout

**Completed: 18/18 cells, zero execution errors, zero retries, 0/18 original
full-gate PASS and 3/18 smoke-projection PASS.** Every run completed its full
budget and all 24 observations on one NVIDIA RTX A6000. Scientific execution
source is `ec9be602b2b8d587b8c8c3f8bc10e98c96580dc1`.
The measured training/evaluation loops total **134.613 seconds**; this excludes
process startup, construction, final serialization and rendering. It is cost
evidence, not a speed ranking. The round stayed within its declared allowances.

[Final receipts and terminal metrics](results.json), [analysis](analysis.json),
[scorer controls](controls.json), and [baseline identity check](baseline-comparison.json)
retain source, recipe, RNG, data-sequence and artifact hashes. All initial networks
and training batches match across arms; initial prior locations match across
cloud/MoG widths within each count. Both baseline endpoints exactly match every
shared numerical endpoint in the original selected BCAP receipts. This is a
current-source matched control, not new qualification for the archived source.

The following parameter grids report endpoint metrics and sustained verdicts;
they are not another ranked leaderboard. A suffix of at least 5 is required.

### Scalar Gaussian: location/scale versus CDF

| Particles | Prior / sigma | Mean error / sigma | Std ratio | CDF KS | Full suffix | Smoke suffix / verdict | Actual training |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 256 | cloud / 0 | 0.05706 | 1.20216 | 0.09153 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/cloud-n256-gaussian1d_acquisition.gif) |
| 256 | MoG / 0.025 | 0.12788 | 1.05499 | 0.11428 | 0 | 1 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog025-n256-gaussian1d_acquisition.gif) |
| 256 | MoG / 0.1 | 0.01304 | 0.95739 | 0.04297 | 1 | 5 / PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n256-gaussian1d_acquisition.gif) |
| 1,024 | cloud / 0 | 0.36259 | 0.71934 | 0.22083 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/cloud-n1024-gaussian1d_acquisition.gif) |
| 1,024 | MoG / 0.025 | 0.40263 | 0.79160 | 0.21693 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog025-n1024-gaussian1d_acquisition.gif) |
| 1,024 | MoG / 0.1 | 0.14235 | 0.99118 | 0.11241 | 0 | 1 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n1024-gaussian1d_acquisition.gif) |
| 4,096 | cloud / 0 | 0.68015 | 0.82643 | 0.32810 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/cloud-n4096-gaussian1d_acquisition.gif) |
| 4,096 | MoG / 0.025 | 0.28464 | 1.38785 | 0.19220 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog025-n4096-gaussian1d_acquisition.gif) |
| 4,096 | MoG / 0.1 | 0.32669 | 1.22568 | 0.12875 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n4096-gaussian1d_acquisition.gif) |

The 256-row MoG at sigma .1 is the only scalar smoke-projection PASS. Its
five terminal CDF errors are **.06993, .06581, .04941, .10015, .04297**.
The final endpoint passes the original CDF bound, but the full terminal suffix
is only one. All five location/width checks pass. The original sigma-.025
baseline passes just two of the last five location/width checks, so dropping
CDF shape alone would still leave that scalar smoke test failing.

Changing to a cloud does not fix the scalar target, and increasing count does
not improve the sustained gate. The wider kernel changes both training latent
noise and the public served distribution; this does not isolate an evaluation-only
smoothing effect or establish sigma .1 as a global default.

### Ring16: all modes versus local distribution fidelity

| Particles | Prior / sigma | Modes | Mass TV | HQ | Full covariance error | Full suffix | Smoke suffix / verdict | Actual training |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 256 | cloud / 0 | 16 | 0.05981 | 0.93042 | 11.02202 | 0 | 9 / PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/cloud-n256-ring16_acquisition.gif) |
| 256 | MoG / 0.025 | 16 | 0.09302 | 0.94385 | 9.61552 | 0 | 8 / PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog025-n256-ring16_acquisition.gif) |
| 256 | MoG / 0.1 | 16 | 0.07666 | 0.79907 | 6.67240 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n256-ring16_acquisition.gif) |
| 1,024 | cloud / 0 | 16 | 0.07031 | 0.81860 | 7.47256 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/cloud-n1024-ring16_acquisition.gif) |
| 1,024 | MoG / 0.025 | 16 | 0.06909 | 0.80347 | 8.68836 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog025-n1024-ring16_acquisition.gif) |
| 1,024 | MoG / 0.1 | 16 | 0.05566 | 0.56128 | 8.08002 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n1024-ring16_acquisition.gif) |
| 4,096 | cloud / 0 | 16 | 0.04150 | 0.59009 | 12.46360 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/cloud-n4096-ring16_acquisition.gif) |
| 4,096 | MoG / 0.025 | 16 | 0.04370 | 0.57910 | 9.50799 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog025-n4096-ring16_acquisition.gif) |
| 4,096 | MoG / 0.1 | 16 | 0.03711 | 0.40063 | 11.35422 | 0 | 0 / FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n4096-ring16_acquisition.gif) |

All nine endpoints acquire all sixteen modes and pass mass balance. Only the
256-row cloud and sigma-.025 MoG sustain the complete smoke projection. Their
HQ values are .93042 and .94385; the wider 256-row MoG falls to .79907, below
the unchanged .85 precision bound. Larger tables worsen precision in this budget.

The existing MoG endpoint has **230/4,096 samples outside three sigma** and
**7/16 components above the full covariance-error bound**. Its core covariance
error is .48050 versus full assigned-component error 9.61552. These are different
statistics: excluding tails would change the quality question. Clean sampling
already removes additive output noise, so that noise cannot explain these tails.
The public MoG kernel remains part of the declared target comparison law.

### Why more particles did not solve this budget

At 256 rows the ring has sixteen declared locations per target component.
The [shared vector scorer](../../../benchmarks/transfer_suite/vector_tasks.py)
has a 32-location resolution floor derived from a historical finite-atom oracle.
The ring task explicitly disables finite-atom exemptions and gates full assigned
covariance anyway. That historical floor supports examining resource adequacy;
it does not calibrate MoG kernels or prove this ring gate impossible.

Increasing count also reduces how often each row receives an update. With
batch 128 and 400 G updates, a uniformly sampled row appears in an expected
**157.6 minibatches at 256 rows, 47.0 at 1,024, and 12.3 at 4,096**.
The selected dualnorm optimizer moves sampled prior rows by approximately .03
per supported update and leaves unsampled rows untouched. Thus increasing
capacity at fixed batch/budget also reduces per-row adaptation opportunities.
This is a plausible contributor, not a measured causal separation: count also
changes the initial finite table and sampling law. It does not authorize changing
the matched training budget or scaling rates after looking at these results.

A 256-atom scalar law is not inherently incapable of KS .05: Gaussian midpoint
quantile atoms have population CDF error at most 1/(2*256), about .00195.
The current scalar failures therefore cannot be dismissed as an inevitable
discreteness floor. Location/width oscillations and non-Gaussian shape remain
real quality and stability failures under the current trainer/budget.

## Recommendations

1. **Keep 256 particles for these fixed-budget smoke questions.** The tested
   larger tables add no sustained pass and regress acquisition. Keep learned
   locations and uniform masses; this study does not test frozen priors, latent
   dimension changes, learnable mixture weights or prior initialization scale.
2. **For a revised scalar location/scale smoke task, use MoG sigma .1.** It
   passes the frozen reduced question at all five terminal checks. Keep the
   exact Gaussian-CDF question as a separate quality requirement; the .1 run
   still fails it.
3. **For a revised ring mode-acquisition smoke task, retain MoG sigma .025.**
   It already passes coverage/precision/balance for eight terminal checks.
   `ParticlePrior` is a runnable alternative, but it does not repair full local
   shape and offers lower endpoint precision here. There is no supported reason
   to replace the shared default with a cloud on this evidence.
4. **Register distinct smoke questions and retain the original full questions
   as quality tasks.** The concrete smoke bounds are frozen in `protocol.json`:
   scalar finite/count/location/width, and ring count/modes/mass/HQ. Suggested
   IDs are `gaussian1d_smoke_location_scale_v1` and `ring16_smoke_modes_v1`.
   A future view revision must retain full CDF/covariance gates and original
   evidence identities, then undergo bounded calibration. These diagnostic
   projections are not an implemented ordinary 6/6 Tier 1 qualification.
5. **Stop this prior grid.** No tested prior passes both original tasks, and
   no single tested prior passes both proposed smoke projections. Task-owned
   prior widths can differ while using one global trainer configuration.
   A subsequent question could separate fixed-budget acquisition from per-row
   exposure, with its budget and controls declared before spending. No new
   round, longer training or seed study follows automatically.

The existing leaderboard continues to record the selected BCAP configuration
at **4/6 required Tier 1 passes** under its original contracts. This report
adds diagnosis and concrete proposed smoke scopes, with all failures retained.

## Verification and retained artifacts

[Verification receipt](verification.json): four CUDA software checks pass: oracle/destructive scorer behavior, actual
prior bindings and matching initialized networks for each target, and rejection
of a changed frozen task before spending. All eighteen training receipts certify
the declared G/D/prior update counts and evaluation-stream separation. Publication
checks source/data/initialization/recipe matching, verifies raw hashes, and
exports eighteen nine-frame actual-training GIFs without new model draws.
All eighteen saved final checkpoints were loaded onto CUDA and verified finite;
prior row counts, completed updates, named-stream envelopes and GIF hashes/frame
counts match the recorded protocol.

Scientific training used Python 3.14.7 and Torch 2.14.0.
Rendering used the separate Python 3.12 / Torch 2.14 / Matplotlib 3.11 environment.
The final PR adds the read-only analysis helper; training remains bound to
the scientific commit above. Raw stdout, full curves, tensors and checkpoints
are local under `runs/api/tier1-prior-smoke-v1/` and its ignored archive.
Compact [artifact provenance](artifact-provenance.json) records their exact hashes.
