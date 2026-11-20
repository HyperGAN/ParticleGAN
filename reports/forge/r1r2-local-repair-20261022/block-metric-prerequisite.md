# Fixed block-Kronecker prerequisite: rejected

This saved-state approximation fails the prespecified tangent-variance sign on
both fixed scoring panels. Stop this exact mechanism without tuning or training.
No task verdict, recipe default or leaderboard changes.

The [reproduction source](block_metric_probe.py) and
[compact receipt](block-metric-diagnosis.json) bind the original NS/AMS control
checkpoint, immutable publication, actual recipe/prior/source and historical
streams. The same original G-phase batches 1585–1600 (128 rows each) are replayed
at one frozen endpoint. Each hypothetical direction is computed independently;
the scoring direction is their arithmetic mean. No parameter, optimizer, stream
or trained state is advanced. Ambient Python/NumPy/CPU/CUDA RNG states match.

For each generator affine layer, including its bias, define
`A = mean(a_aug a_augᵀ)` and `B = mean(J_preᵀ J_pre)`, where `J_pre` maps that
layer's preactivation to generated output. Solve the row-major parameter block
`(B ⊗ A + λ I) Δ = -.0085 ∇θ L`, using exact Kronecker eigenvalue denominators.
Each sampled prior row uses its exact accumulated `mean(J_zᵀ J_z)` block; unseen
rows stay exactly zero. Fixed damping is `.01 * trace(block) / block_dimension`
for each block. No damping or coefficient search was performed.

`L` is the full scalar non-saturating generator objective plus the declared
unbiased empirical seven-width MMD at coefficient 1. Both fake arguments are
differentiated. Targets are the original replayed real rows; target means enter
only the scoring panels. This Euclidean primal block approximation discards
generator/prior cross blocks and within-layer activation/Jacobian correlations.
It replaces the saved AMS direction and has a different normalization from the
earlier dual-output GN probe. Neither probe qualifies the other mechanism.

| Mean directional derivative | 2048 axial points | 4096 corner points | Required sign |
| --- | ---: | ---: | --- |
| Original covariance error | -.00008044 | -.00020894 | negative |
| Radial variance | +.00405728 | +.00404283 | positive |
| Tangent variance | +.00127815 | +.00110796 | nonpositive: **fails** |
| Center squared error | -.07232220 | -.07126958 | nonpositive |
| Squared covariance error, additional contrary evidence | +.00325960 | +.00095935 | not the admission metric |

The original covariance error is the mean per-mode Frobenius norm, rather than
the squared proxy. Their different signs are retained. Both panels integrate
the same mean parameter direction; they are sensitivity checks, not independent
trials or trained gate observations.

The deterministic Cartesian linear oracle matches the explicitly materialized
block metric to `1.12e-16` and its dense damped solve to `1.78e-15`; the
rank-deficient oracle error is `2.96e-15`. Saved-batch full-parameter objective
finite differences agree within `1.73e-12`; prior-block autograd error is below
`1.74e-18`. Maximum actual affine-block solve residual is `2.69e-14`.

CPU float64, one thread: full objective gradient plus block construction/solve
averages 17.77 ms per batch (maximum 29.54 ms); total diagnostic wall time is
2.20 seconds, including binding, replay and scoring. This is a local cost
measurement, not GPU or word-task feasibility. Raw stdout and tail-friendly logs
remain outside Git under
`/mnt/ml7tb/ParticleGAN-forge/r1r2-local-repair-20261022/block-metric/`.

Reproduce from the repository root, with the original CUDA stream archive
available, using `python reports/forge/r1r2-local-repair-20261022/block_metric_probe.py
--output /tmp/block-metric-diagnosis.json`. No paid experiment is authorized by
this rejected prerequisite.
