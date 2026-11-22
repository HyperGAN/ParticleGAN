# MoG / DDGAN small-generator capacity screen

All four runs completed and passed the runner's config, source, and summary
certification. Training finished in 4.2 minutes on two GPUs, one job per GPU.
Branch: `feat/mog-denoising`. No seed experiments were run.

The clearest result is that MoG supplies useful local output variation with this
small generator. The screen does not establish greater eventual capacity for
DDGAN, or an overall MoG quality win.

## Matched design

- One shared seed (24002), 14,000 updates, batch 256, constant learning rates.
- Generator width 32 and depth 3; discriminator width 128 and depth 3.
- Class-only UCD in every arm; Gaussian reverse noise in DDGAN.
- Both priors use `MoGParticlePrior`, 400 components, standardized reads,
  prior LR multiplier 100 and prior Adam betas (0.5, 0.999).
- Only model type and sigma_rel (0 or .025) differ, apart from output paths.
- Generator parameters: 2,466 GAN / 2,658 DDGAN. Prior: 1,600 in every arm.
  Discriminator parameters: 34,948 GAN / 35,716 DDGAN due to conditioning inputs.

The model comparison also changes the objective, inputs, and repeated sampling
computation. It cannot isolate architecture capacity by itself. The optimizer
was held fixed, not separately tuned for each model.

## Results and interpretation

| Model | Conditional SW1 ↓ | Joint HQ ↑ | Modes ↑ | Core width (ideal 1) |
|---|---:|---:|---:|---:|
| GAN + atoms | .5784 | 96.40% | 96 | .559 |
| GAN + MoG | .5984 | 94.21% | 99 | 1.120 |
| DDGAN + atoms | .1884 | 5.17% | 28 | 9.317 |
| DDGAN + MoG | .1817 | 5.24% | 24 | 9.456 |

The [SW1 leaderboard](TABLE.md) puts DDGAN first on broad distribution distance,
but its roughly 57% far-tail rate, 44–45% class accuracy and very poor widths
preclude calling it the quality winner. The checkerboard class target can look
close under projected distance while placing mass between the true modes.
The real-versus-real conditional SW1 reference is .0702.

For GAN, MoG improves width, coverage, class fidelity, and tails; atoms have better
SW1 and HQ. Neither dominates. The atom model produces 1,600 unique outputs across
20,000 draws (400 particles times four classes); MoG produces 20,000.

The [frozen-checkpoint intervention](NOISE_PROBE.md) is particularly informative:
turning noise off in GAN+MoG changes width from 1.120 to .453. Adding noise to the
already-trained GAN+atoms changes width from .559 to 1.219 **without training**;
SW1 stays almost unchanged (.5784 to .5771). Thus much of the width benefit can
be obtained through sampling variation alone. This supports a support/variation
explanation, not a demonstrated need for more learned parameters or training.
That intervention also reduces HQ (96.40% to 92.24%), so it is not a free win.

DDGAN's two curves remain close. Its small MoG SW1 advantage at equal updates
reverses at the common training-time budget, and both remain poor locally. HQ
and class balance are still improving. This is inconclusive about eventual
capacity, not evidence that DDGAN cannot exploit the extra computation.

## Recommended next experiments

1. Cheapest practical follow-up: choose sampling sigma on held-out draws using
   the frozen GAN+atoms checkpoint, checking width, tails, HQ, and balance
   together. No retraining is needed to investigate local calibration.
2. For the capacity hypothesis: run just the GAN atom/MoG pair with G width 128,
   keeping the current 14k budget, discriminator and all other settings fixed.
   Test whether the small-G MoG width advantage shrinks with larger G.
3. Defer a larger DDGAN factorial. If its eventual capacity remains the priority,
   a longer matched atom/MoG pair is needed; these 14k curves do not answer it.
   Preserve the constant LR schedule and compare equal-time curves as well as
   equal-update endpoints.

## Reproduction and artifacts

```bash
.venv/bin/python -u experiments/run_grid.py \
  --config_manifest configs/denoising/mog_capacity/manifest.json \
  --trainer experiments/train_denoising.py --gpus 0,1 --workers_per_gpu 1
.venv/bin/python experiments/analyze_mog_capacity.py
.venv/bin/python experiments/probe_mog_capacity.py
tail -F results/denoising/mog_capacity/*/log.txt
```

[Learning curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/mog_capacity/curves.png) · [Samples](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/mog_capacity/samples.png) ·
[Leaderboard CSV](leaderboard.csv) · [Full comparison](comparison.json) ·
[Frozen noise intervention](noise_probe.json).

Raw configs, logs, source snapshots, EMA checkpoints, metrics, final samples,
and completion certificates are under `results/denoising/mog_capacity/<run>/`.
Reviewable copies of each run's metrics, summary/config, and completion
certificate are committed under `runs/<run>/` alongside this report.
The focused compatibility/config test suite passed all 48 tests before launch.
The analyzer verified that only the two intended factors vary and that all
four runs share identical training-source provenance. Frozen baseline evaluation
reproduced each original final conditional SW1 within 1e-5.
