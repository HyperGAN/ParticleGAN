# One capped schedule: two accurate 100-mode cases, one failure

This experimental configuration passes **2/3** strict 100-mode cases. It is
not a common 22-case success. All three use the same config, initialization,
and optimizer schedule; only the problem sampler changes.

The trainable affine generator starts at identity. Its learned particles
start uniformly in the generic square [-5, 5]², without target centers,
labels, or target samples. The existing cosine schedule uses
`min(total_steps, 2400)` for every optimizer group, then holds its floor
through update 7,000. G, D, and the particle prior continue training.

| Case | Strict gate | Final live modes | Holdout HQ | Holdout mass TV | Center RMS / σ | Covariance bias | Radial KS |
|---|---|---:|---:|---:|---:|---:|---:|
| grid100 | FAIL | 37/100 | .40672 | .21453 | undefined | undefined | undefined |
| rotated100 | PASS | 100/100 | .97648 | .03751 | .13162 | −.02904 | .00960 |
| staggered100 | PASS | 100/100 | .98409 | .03978 | .11411 | −.02539 | .00933 |

Rotated100 and staggered100 pass all five terminal 20,000-draw checks and
independent 100,000-draw holdouts. Grid100 has all 100 modes at update 1,000,
then becomes unstable by 1,250, before decay starts at 1,440. That failure
rules out promoting this schedule as a shared configuration.

The [combined GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy100/accuracy-horizon/toy100-progress.gif) renders saved live samples for all three,
including the failing grid. Optimizer traces, exact probe source, shared
config, snapshots, terminal samples, and holdouts are included. The existing
19 hosts have budgets no longer than 1,600, so this cap alone would leave
their schedules unchanged; it cannot repair existing noise/core failures.
This prototype has an explicit extra architecture and schedule declaration
and is not production shared-gate evidence.

```bash
python -m benchmarks.toy100 accuracy --output reports/toy100/accuracy-horizon
# Expected: FAIL, 2/3. To reproduce one case in a new directory:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u reports/toy100/accuracy_horizon_probe.py \
  --config reports/toy100/accuracy-failure/rotated100/config.json \
  --prior square --scale 5 --problem grid100 --horizon 2400 \
  --output /tmp/new-affine-horizon-run
```
