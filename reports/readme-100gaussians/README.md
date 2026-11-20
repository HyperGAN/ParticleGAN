# README 100-Gaussian visualization

## Current default

The root `100gaussians.gif` was generated from `get_recipe()` in **0.11.0**
(`DEFAULT_RECIPE = "unified_v11"`) on the explicit Forge `grid100` task-prior
cohort (`MoGParticlePrior(num_particles=20000, z_dim=2, sigma=0.025, standardize=False)`),
with zero technique-recipe overrides. The
[current protocol](CURRENT_PROTOCOL.md) declares the fixed task, seed-0 public
deterministic initialization (`generator=0, critic=1, prior=2`), separate RNG
streams, 7,000-update default budget and mode-acquisition gate before execution.
The archived 0.9.0 `ParticlePrior(20000)` runs and historical K3P runs below
are separate cohorts and retain their original identities and receipts.

| Current illustration | Prior cohort | Updates | Covered modes | Within 3σ | Mode TV ↓ | Sliced W1 ↓ | Beyond 10σ ↓ | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.11.0 `unified_v11`, live/clean, seed 0 | `MoGParticlePrior(20000, σ=0.025, std=False)` | 7,000 | 100/100 | 98.520% | 0.04555 | 0.08353 | 0.400% | **PASS** |

The model covers all 100 modes and passes the declared ≥95% within-3σ
mode-acquisition gate at all five terminal scoring checks:

- step `5,833`: `100/100` modes, `hq = 99.085%`, `mode_tv = 0.04390` (`PASS`)
- step `6,125`: `100/100` modes, `hq = 98.870%`, `mode_tv = 0.04560` (`PASS`)
- step `6,417`: `100/100` modes, `hq = 99.390%`, `mode_tv = 0.04720` (`PASS`)
- step `6,708`: `100/100` modes, `hq = 98.635%`, `mode_tv = 0.04675` (`PASS`)
- step `7,000`: `100/100` modes, `hq = 98.520%`, `mode_tv = 0.04555` (`PASS`)

For reference, the independent 20,000-draw oracle control scores `100/100` modes,
`hq = 98.755%`, and `mode_tv = 0.02915`, while the collapsed zero control scores
`0/100` modes and `hq = 0.0%`. At step 7,000, the per-mode core width ratio is
`0.957` (`per_mode_core_std = 0.02872` vs target `σ = 0.0300`, target ratio `1`),
and the per-mode covariance eigenvalue ratio spans `[0.812, 1.769]` across all
100 audited modes. Training took `260.49` seconds (`263.47` seconds including
recording overhead) on an NVIDIA RTX A6000.

The [current receipt](summary-current.json) records the exact resolved recipe,
explicit `MoGParticlePrior` task cohort, initializer, controls, all terminal
checks, source hashes, rendering receipt (`4,389,753` bytes, `960×640`, `251`
frames at 25 fps with 2.5 s final hold), and artifact archive identities.
Restoring the saved `final.pt` public-API checkpoint reproduces both the 4,096
plotted rows and the 20,000 evaluation draws bit for bit; independent metric
recomputation matches `summary-current.json`, and `render.py` rejects altered
observation hashes and forged status verdicts.

This is one fixed-seed visual illustration, not Forge qualification (see
[reports/forge/technique-inventory.md](../forge/technique-inventory.md) for full
Forge status, where screening calibration remains provisional/`BLOCKED` and
remaining failures stay disclosed). Because this 0.11.0 run uses the Forge
`grid100` `MoGParticlePrior(20000, σ=0.025, standardize=False)` task prior while
the archived 0.9.0 runs used zero-sigma `ParticlePrior(20000)`, they are
explicitly separate cohorts rather than a controlled prior-held-fixed comparison.

Reproduce this animation from the current 0.11.0 checkout:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u reports/readme-100gaussians/generate.py \
  --steps 7000 --prior-kind mog --prior-sigma 0.025 --no-standardize \
  --device cuda:0 --output-dir /path/to/new-run > /path/to/training.log 2>&1
tail -f /path/to/training.log
```

Use `--device cpu`, `cuda:0`, or `cuda:1` as appropriate. The output directory
must be new. Playback requests 250 training frames plus the untrained state
(251 frames total), 25 fps, a smooth quadratic time warp and a 2.5-second final
hold. The plot follows 4,096 fixed particle row IDs; trails connect saved
observations. Targets are true 3σ circles. The right panel reports independent
20,000-draw metrics and their observed history; fixed scoring checks determine
the terminal verdict, independently of media density. The command writes results
and exits 1 if the gate fails.

Re-style the same saved observations without running training again:

```bash
python reports/readme-100gaussians/render.py /path/to/new-run
```

`observations.npz`, complete `initial.pt`/`final.pt` checkpoints, raw
`metrics.jsonl`, the frozen `contract.json`, `summary.json`, previews and GIF
stay in that local output directory. The renderer verifies the observations'
hash, schedules and verdict against the receipt. Publish only the final GIF and
compact summary/render receipt. `--record-only` separates recording from
rendering; `--steps 4 --frames 4` is a software smoke, not a passing scientific
run. No seed sweep or best-checkpoint selection is used.

### Archived 0.9.0 default (ParticlePrior cohort)

The prior root `100gaussians.gif` was generated from `get_recipe()` in **0.9.0**
(`bispf_optimized`, native SVD, float32) on the zero-sigma `ParticlePrior(20000)`
particle-cloud cohort and is preserved byte for byte in Git history:

- 20,000-update continuation publication commit:
  [`1bc01a7bf`](https://github.com/255BITS/ParticleGAN/commit/1bc01a7bf4b151c01073c35f59a1d894d0917571)
  — archived [GIF](https://github.com/255BITS/ParticleGAN/blob/1bc01a7bf4b151c01073c35f59a1d894d0917571/100gaussians.gif)
  (Git blob `4f97aee33969d391182338dd41a90e3c896eb11a`) and
  [receipt](https://github.com/255BITS/ParticleGAN/blob/1bc01a7bf4b151c01073c35f59a1d894d0917571/reports/readme-100gaussians/summary-current.json)
  (Git blob `f6f31cc543f7960c6c79d95d3e2a7e7db4eecb5b`).
- Initial 7,000-update publication commit:
  [`d266a1233`](https://github.com/255BITS/ParticleGAN/commit/d266a1233304fba1c7328412b258725d5f7b3593)
  — archived [GIF](https://github.com/255BITS/ParticleGAN/blob/d266a1233304fba1c7328412b258725d5f7b3593/100gaussians.gif)
  and [receipt](https://github.com/255BITS/ParticleGAN/blob/d266a1233304fba1c7328412b258725d5f7b3593/reports/readme-100gaussians/summary-current.json).

| Archived 0.9.0 illustration | Prior cohort | Updates | Covered modes | Within 3σ | Mode TV ↓ | Sliced W1 ↓ | Beyond 10σ ↓ | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.9.0 optimized BISPF, clean/live, seed 0 | `ParticlePrior(20000)` | 20,000 | 100/100 | 42.115% | 0.11660 | 0.15783 | 1.30% | FAIL |
| 0.9.0 optimized BISPF, clean/live, seed 0 | `ParticlePrior(20000)` | 7,000 | 100/100 | 36.645% | 0.12055 | 0.16725 | 3.39% | FAIL |

The 0.9.0 model covered all modes, but missed the declared 95% precision goal at
all five terminal scoring checks (final 20k per-mode core width ratio `1.780`,
total training `754.2` seconds on an RTX A6000 across source commits
[`dd2cfd8b0`](https://github.com/255BITS/ParticleGAN/commit/dd2cfd8b0) and
[`c2a3813aa`](https://github.com/255BITS/ParticleGAN/commit/c2a3813aa4e1ca0f1183542989532aeaf060bc4e)).
The 0.9.1 README additionally illustrated its faster default with
[actual candidate image training](../forge/bispf-throughput/README.md).

## Historical runs

The following results describe archived GAN v3 and K3P runs. Their prior
publication and root GIF are preserved at
[commit 2b2108c6f](https://github.com/255BITS/ParticleGAN/tree/2b2108c6ff0270f72f599ab5ab4c9b1bef6c3007/reports/readme-100gaussians).
The historical commands below require each run's original source bindings and
resolved recipe, recorded in its summary. The old
[root GIF](https://github.com/255BITS/ParticleGAN/blob/2b2108c6ff0270f72f599ab5ab4c9b1bef6c3007/100gaussians.gif)
is preserved exactly as Git blob `a29394a6db151ad9ae25f044d871e818e22c123b`.

An earlier root `100gaussians.gif` showed one run of
`get_recipe("gan", total_steps=20000)` through the explicitly imported
`GANTrainer`, using the existing `examples/100gaussians.py` generator and
Fourier-feature discriminator. Only the training budget changes; all other
current GAN defaults and seed 1234 are retained. There is no seed comparison. Target:
100 equally weighted Gaussian modes on a 10×10 grid, each with σ = 0.03.

This run starts from the same initialization as the prior 7,000-update run;
it is not a continuation of that run's final checkpoint. Increasing the
budget also moves the delayed cosine anneal from step 4,200 to step 12,000.
Both schedules hold the initial learning rates for 60% of their budgets,
then decay to a 5% floor. The comparison therefore changes both duration
and absolute anneal timing; the library's default budget remains 7,000.

The animation displays 4,096 fixed target samples and outputs from the same
4,096 particle indices at each snapshot. Numeric metrics use 20,000 random
prior draws from the EMA generator and EMA prior, resetting a separate
evaluation RNG at each checkpoint for a consistent comparison. A covered mode has at least
10 generated samples within Euclidean distance 3σ of its center.

There are 200 frames, captured every 100 updates from step 100 through
20,000. Playback is 25 frames per second with a one-second final hold.
Metrics are evaluated at step 100 and every 500 updates; intermediate
frames explicitly label the last metric evaluation step. A shared palette
keeps colors consistent across the animation.

## Budget comparison

| Configuration | Updates | HQ modes | Within 3σ | Mode TV ↓ | Sliced W1 ↓ | Beyond 10σ ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k budget, current GIF, EMA | 20,000 | 100/100 | 93.27% | 0.1644 | 0.1457 | 2.010% |
| 7k default budget, previous GIF, EMA | 7,000 | 99/100 | 75.08% | 0.1692 | 0.1301 | 4.385% |

The 20k run took 167.7 seconds of training on an NVIDIA RTX A6000, versus
59.9 seconds for 7k. In the 20k run, HQ reached 95.99% at step 19,000,
dropped to 91.25% at 19,500, and ended at 93.27%. The animation includes
every 100-update snapshot and ends at the prescribed budget rather than
selecting the best checkpoint. The longer schedule improved final HQ,
mode coverage, and tail mass; sliced W1 was slightly worse.

For the prior 7k run, HQ rose from 2.69% at step 5,000 to 56.01% at 6,500
and 75.08% at 7,000. Both runs sharpen late in their respective annealing
schedules. Their matching early metrics precede the schedule divergence
at step 4,200.

This is an illustration and a quality-control observation, not a multi-run
estimate or evidence of complete Gaussian calibration. The learned prior has 20,000 zero-noise particles,
so generated support is finite. The 20,000 evaluation draws contain 12,662
distinct output points in each run. The 20k per-mode core width ratio is
0.769 (target 1), compared with 1.116 at 7k; its sharper modes are narrower
than the target. Tail mass, unequal mode weights, and full covariance
distortion remain material. These EMA results do
not establish live-model performance or contradict the transfer suite's
separate metrics and discriminator choices.

Recommendation: use the explicit 20k budget to reproduce this visualization.
It improves mode matching for this example, but the mixed distribution
metrics do not justify promoting it to a new universal default. Assess
budget or discriminator changes across the intended research tasks before
changing shared defaults. No seed sweep or additional tuning was performed.

## Reproduce

Install the repository's experiment dependencies. Generate both live and
EMA previews from one 7,000-update training run:

```bash
python -u reports/readme-100gaussians/generate.py \
  --steps 7000 --views live ema --frames 280 --frame-power 1 --fps 25 --hold-seconds 1 \
  --output-dir artifacts/100gaussians-7k > /tmp/readme-image-7k.log 2>&1
tail -f /tmp/readme-image-7k.log
```

These are the default training/rendering options; the default output directory
is `artifacts/100gaussians`. Select `--views live` or `--views ema` for one GIF.
The seed stays 1234. Supported frame rates are 1, 2, 4, 5, 10, 20, 25, 50,
and 100 fps, which GIF timing can represent exactly. Each GIF holds its final
frame for `--hold-seconds` (default 2.5).

To reproduce the 20k EMA configuration shown in the README:

```bash
python -u reports/readme-100gaussians/generate.py \
  --steps 20000 --views ema --frames 200 --frame-power 1 --fps 25 --hold-seconds 1 \
  --output-dir artifacts/100gaussians-20k > /tmp/readme-image-20k.log 2>&1
tail -f /tmp/readme-image-20k.log
```

Outputs stay in the selected directory: `100gaussians-live.gif` and/or
`100gaussians-ema.gif`, per-view PNG frames and `final_samples.npz`, shared
`final-models.pt`, `metrics.jsonl`, and `summary.json`. Both views use the same
fixed particle indices and evaluation draws, with separate metrics and
explicit labels. The summary records all CLI options, recipe settings,
timing, source hashes, and environment. Choose a fresh output directory for
each invocation; existing results are not overwritten. `artifacts/` is
ignored by Git. The command does not replace the root GIF or historical
report summaries; preview the output before publishing it.

The historical 20k run's log is `/tmp/particlegan-develop-qc/readme-image-20k.log`.
[summary.json](summary.json) records all resolved recipe fields, checkpoint
metrics, source hashes, checkout commit, and environment. Source hashes cover
the working tree at the start of this run, including the renderer changes;
the checkout commit alone does not describe those changes. The earlier run's
complete summary and provenance are preserved in [summary-7000.json](summary-7000.json).

The previous GIF was generated in [commit 8664db9](https://github.com/255BITS/ParticleGAN/commit/8664db993f718e7ba290becc352224a46d619313)
using the older 0.1.2 recipe: 7,000 updates, seed 1, cap coefficient 1.0,
and generator learning rate 0.0006. Its final frame explicitly displays
`step 7000/7000`, `modes 100/100`, and `hq 0.992`; the commit also records
the same budget and EMA readout. It did not use a longer training budget.
Different hyperparameters and seeds make this an uncontrolled comparison,
and the old GIF has no equivalently audited shape-metric protocol here.

## Historical README hero GIF

The historical root `100gaussians.gif` showed the then-default formulation:
`GANTrainer(get_recipe("gan"), G, D, seed=1234)` (K3P, 7,000 updates) with the
small MLP generator and Fourier-feature discriminator from `examples/100gaussians.py`.
No recipe, optimizer, penalty or schedule field is overridden; only visualization
settings are chosen here. The animation shows live weights. Full provenance and
metrics are in [summary-hero.json](summary-hero.json).

All source hashes in that receipt match core checkout
[`d1b06bcd6`](https://github.com/255BITS/ParticleGAN/commit/d1b06bcd604fee1fb74f2951975a1c16d50df6d9)
with `generate.py` from
[`bae4365a2`](https://github.com/255BITS/ParticleGAN/commit/bae4365a2).
Use those recorded bytes to reproduce the historical hero; running the current
command resolves today's selected default.

Frames follow a smooth time warp, step = 7000·(i/250)², so early training
(where the motion is) is dense and the cadence never jumps: 250 frames at
25 fps (10 s) plus a 2.5 s hold on the final frame. The displayed 4,096 points
are the same particle indices every frame, drawn as noise-free generator
outputs; the overlay metrics use 20,000 fixed evaluation draws per frame.

Final live metrics: 100/100 modes, 98.9% within 3σ, mode TV 0.162, sliced W1
0.155, 1.0% beyond 10σ. The per-mode core width ratio is 0.19 (target 1): the
noise-free generator places modes as tight points, narrower than σ = 0.03
(the recipe's 0.029 output noise, not shown, supplies most of the width).

```bash
python -u reports/readme-100gaussians/generate.py --views live ema \
  --frames 250 --frame-power 2 --fps 25 --hold-seconds 2.5 \
  --output-dir artifacts/100gaussians-hero > /tmp/readme-hero.log 2>&1
tail -f /tmp/readme-hero.log
cp artifacts/100gaussians-hero/100gaussians-live.gif 100gaussians.gif
```

Those are the script defaults. The earlier toy100-config animation
(`render_hero.py`, `hero-config.json`) is available in the repository history.
