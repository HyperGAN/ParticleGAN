# R1/R2 saved-state repair prerequisites

Output-tangent geometry is a supported research hypothesis, **not a trained
repair**. It prevents the earlier fixed-MMD force reversal on the saved NS
control. Exact original 128-example batches weaken the population result: radial
spread improves, but tangent variance increases slightly. A complete, affordable
global recipe remains unestablished. No trainer was advanced, no gate was changed,
and no independent random stream or seed experiment was added.

The [current technique inventory](../technique-inventory.md) remains the only
generated leaderboard. These derivative comparisons are analysis cohorts, not
new task results. The [source-bound summary](repair-prerequisites.json) pins raw
artifacts, drivers, negative findings and numerical checks; the
[preceding diagnosis](../r1r2-failure-diagnosis-20261022/README.md) binds the
original failure and the 34 concluded global repairs.

## One geometry hypothesis survives the population proxy

Let `J` be the Jacobian of generated outputs with respect to G and learned prior
locations, and `P` the saved positive AMS diagonal inverse. The desired output
velocity `v` is negative batch size times the output gradient of native NS plus
the previously declared fixed seven-width MMD, at coefficient 1. Solve

```text
(J P Jᵀ + λ I) u = lr v
Δθ = P Jᵀ u
λ = .01 * trace(J P Jᵀ) / number_of_output_coordinates
```

This changes G/prior update geometry and adds the declared MMD. D, positive
R1/R2, prior learning policy, initial distribution/capacity and all task contracts
would remain explicit. The probe uses the fixed archived widths
(.0625,.125,.25,.5,1,2,4), lr 0.0085 and a relative residual requirement 1e-4.
No coefficient or damping search was performed. CG with 32 iterations and PCG with 256 iterations did not meet the
numerical requirement and remain local rejected solver receipts. Extending the
same equation's numerical allowance to 1024 iterations gives accepted solutions.

The large-panel target expectation integrates the original 16-Gaussian target
law. Fake outputs use 2048 axial or 4096 corner cubature points from all 256 saved
prior rows. These integration rules match initial Gaussian first/second moments;
they are not new training or evaluation sampling laws.

| Original-covariance-error directional derivative | Axial panel | Corner panel |
| --- | ---: | ---: |
| NS+MMD, ordinary saved AMS projection | +.160265 | +.171174 |
| NS+MMD, output-tangent geometry | -.012520 | -.012324 |

With tangent geometry, center squared error decreases, underwide radial variance
increases and overwide tangent variance decreases on both panels. CG takes 555
and 794 iterations, respectively. True relative residuals are 9.60e-5/9.63e-5,
and independently checked analytic kernel diagonals differ by less than 4e-16.
The [population receipt](gn-force-diagnosis.json) retains its limits. CPU float64
wall time is 4.47/18.82 seconds, distinct from paid training charges.

## Original 128-example draws give a mixed result

[Historical replay](replay_training_batches.py) reproduces all 1,600 real target
batches and both 3,200-draw prior-index/noise streams. Both original target-role
digests and all three terminal generator states match exactly. It uses isolated
clones of the receipt-bound streams, retains the original protocol seed 0, and
leaves ambient Python/NumPy/CPU/CUDA RNG states unchanged. It reconstructs draw
identities, not intermediate trained models.

The proxy evaluates original G-phase draws 1585–1600 with the one frozen endpoint
G/prior/D. Each force uses its actual 128 real examples and 128 original latent
row/noise identities. MMD now uses empirical targets, with no task centers or
labels in its force. The table reports the arithmetic mean of 16 independently
computed hypothetical parameter directions on the two fixed scoring panels.
No resulting direction is applied to a trained state.

| Mean directional derivative, axial / corner panel | NS-only tangent | NS+MMD tangent |
| --- | ---: | ---: |
| Original covariance error | -.000171 / -.000173 | -.000092 / -.000105 |
| Radial variance, initially too small | -.000022 / -.000008 | +.000755 / +.000774 |
| Tangent variance, initially too large | -.000266 / -.000260 | +.000161 / +.000145 |
| Center squared error | -.001517 / -.001450 | -.006964 / -.006874 |

All 32 objective/batch solves pass the 1e-4 true-residual requirement, requiring
109–134 iterations. Independent diagonal checks and an empirical-MMD derivative
check also pass. The [batch receipt](gn-batch-diagnosis.json) retains the original
stream identities and individual solve results. Repeating the scoring panel is
integration sensitivity of the same 16 directions, not an independent trial.

GN improves covariance and avoids the strong MMD projection regression. MMD
changes the radial direction favorably, while its small tangent expansion and
weak covariance improvement remain contrary evidence. NS-only GN does not solve
the radial compression. Neither result establishes a full density repair.

The generic solver's separate float32 CUDA prototype also lacks a demonstrated
cheap, accurate all-host implementation. These accurate CPU float64 results do
not transfer its numerical or runtime acceptance. A truncated public policy is
a distinct candidate, and its word task must retain the original 900-second
allowance rather than shortening or omitting it.

## Rejected data-only auxiliary prerequisites

Three fixed auxiliary probes also retain their negative findings. Their recipe
functions receive only real/fake coordinates; target means appear only in the
separate cubature fixture and shape diagnostic.

| Auxiliary at weight 1 | Saved projected result |
| --- | --- |
| Existing relative local kernel-feature means | Covariance improves, but center squared-error derivative becomes strongly positive. |
| Mass-normalized conditional first/full-covariance moments | Radial variance contracts sharply; centering changes sign across panels. No runtime helper is adopted. |
| Local target-minus-fake KDE scores | Output-space covariance improves, but ordinary G/AMS projection reverses it and expands tangent variance. |

These are saved prerequisites, not trained FAILs. Full per-mode diagnostics and
intermediate numerical receipts remain outside Git at
`/mnt/ml7tb/ParticleGAN-forge/r1r2-local-repair-20261022`.

## Fixed block approximation is rejected

The cheaper Euclidean primal block-Kronecker approximation retains the full
scalar NS+empirical-MMD gradient, exact sampled-prior row blocks and fixed .01
trace-normalized damping. On the same 16 original batches, covariance error,
radial variance and centering improve, but tangent variance increases on both
panels (+.001278 / +.001108). This violates its declared prerequisite. The
[readout](block-metric-prerequisite.md), [receipt](block-metric-diagnosis.json)
and [source](block_metric_probe.py) retain the failed sign, dense numerical
oracles, source/stream bindings and local cost. No tuning or training followed.
This approximation uses a different metric from the earlier output solve.

## DDGAN remains a conditional and final failure

The original native NS control also fails the full DDGAN Gaussian-chain
protocol: final CDF .32094 and step-1 CDF .52395/.29191, with zero joint passing
observations. High-noise reverse kernels mask missing clean-predictor variance.
A separate saved-field probe finds mean correction opposed on seven of eight
fixed contexts at both quadrature orders. Stored critic momentum improves all
eight alignments locally; fresh R1+R2 and R1-only critic directions worsen the
remaining step-1 context. These controls do not establish a momentum or R2
repair. The [DDGAN readout](ddgan-diagnosis.md), [compact receipt](ddgan-diagnosis.json)
and [source](ddgan_critic_field_probe.py) preserve the exact original cohort and
negative controls. No new training, samples or qualification results were added.

## Reproduce

Run from the repository root with PyTorch 2.14.0+cu130. CUDA 0 is required only to
replay the original CUDA generator algorithms; the force/solve probes use one
CPU thread and float64. All trained checkpoints and source snapshots remain
under their original archive identities.

```sh
REPAIR_ARTIFACTS=/mnt/ml7tb/ParticleGAN-forge/r1r2-local-repair-20261022
PROBES=reports/forge/r1r2-local-repair-20261022
for panel in axes corners; do
  python "$PROBES/gn_force_probe.py" --max-iterations 1024 --panel "$panel" \
    > "$REPAIR_ARTIFACTS/gn-force-pcg1024-$panel.stdout.json" \
    2> "$REPAIR_ARTIFACTS/gn-force-pcg1024-$panel.run.log"
  python "$PROBES/gn_batch_probe.py" --panel "$panel" \
    > "$REPAIR_ARTIFACTS/gn-batch-$panel.stdout.json" \
    2> "$REPAIR_ARTIFACTS/gn-batch-$panel.run.log"
  for mechanism in local conditional score; do
    python "$PROBES/local_repair_probe.py" --panel "$panel" --mechanism "$mechanism" \
      > "$REPAIR_ARTIFACTS/final-$mechanism-$panel.stdout.json" \
      2> "$REPAIR_ARTIFACTS/final-$mechanism-$panel.run.log"
  done
done
python "$PROBES/summarize_repair.py" > "$REPAIR_ARTIFACTS/summary-validation.log" 2>&1
tail -F "$REPAIR_ARTIFACTS/gn-batch-axes.run.log"
```

For a read-only receipt/source check using retained raw outputs, run only the
summary command. It adds no gradients or RNG replay. Preserve all historical
qualification results; resolve accurate bounded runtime and all-host contracts
before admitting this geometry hypothesis to paid training.
