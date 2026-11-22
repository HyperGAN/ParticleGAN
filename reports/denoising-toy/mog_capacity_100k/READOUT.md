# 100k updates: DDGAN recovers coverage and width, but still trails on HQ

All four runs completed successfully in **28.9 minutes** on two GPUs. No seed
sweep was run. Work is on `feat/mog-denoising`.

DDGAN recovers from roughly 5% joint HQ at 14k to 78–80% at 100k, covers all
100 modes, and reaches near-target core widths. The one-shot models retain
93–95% HQ but lose modes and worsen their distribution fit. There is no single
winner across all quality measures.

## Final leaderboard

Sorted by conditional sliced W1, lower is better. All endpoints use 20,000 draws.
Core width should approach 1; HQ alone can reward concentration on fewer modes.

| Model | Conditional SW1 ↓ | Joint HQ ↑ | Modes ↑ | Core width | Conditional TV ↓ | Far tails ↓ |
|---|---:|---:|---:|---:|---:|---:|
| DDGAN + atoms | .1172 | 77.86% | 100 | .989 | .1016 | 8.25% |
| DDGAN + MoG | .1190 | 79.57% | 100 | .889 | .1072 | 7.95% |
| GAN + atoms | .7595 | 93.46% | 77 | .854 | .4212 | 2.99% |
| GAN + MoG | .9025 | 95.19% | 77 | .701 | .4969 | 1.70% |

DDGAN class accuracy is about 91%, versus 98% for GAN. Its real-versus-real
conditional SW1 reference is .0702. The remaining DDGAN tails and class errors
are material, even though the mode centers and core widths look much better.

## What longer training changed

- DDGAN+atoms: HQ **5.17% → 77.86%**, modes **28 → 100**, width **9.317 → .989**.
- DDGAN+MoG: HQ **5.24% → 79.57%**, modes **24 → 100**, width **9.456 → .889**.
- GAN+atoms: modes **96 → 77**, SW1 **.5784 → .7595**.
- GAN+MoG: modes **99 → 77**, SW1 **.5984 → .9025**.

The [curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/mog_capacity_100k/curves.png) show different trajectories: DDGAN improves coverage,
class fidelity and width slowly, while one-shot HQ stays high even as mode mass
becomes less balanced. Thus the early 14k comparison was misleading about
DDGAN's eventual usefulness. It still has not caught the one-shot models on HQ.

MoG's effect is modest and metric-dependent. Over 81k–100k checkpoints, DDGAN+MoG
averages 76.80% HQ, .1200 SW1 and 1.014 width, versus 74.03%, .1363 and 1.203 for
atoms. MoG also leads DDGAN SW1 at the common training-time budget. The single
final checkpoint reverses the SW1 ordering. These are trajectory summaries,
not independent trials or uncertainty estimates.

One-shot final widths are noisy: the late mean width is .495 for atoms and .670
for MoG, less favorable than their final 20k-draw values. MoG's earlier coverage
advantage does not survive to 100k. Both one-shot models deteriorate on balance
under this constant-LR regime, so that deterioration cannot establish a hard
representational limit.

DDGAN is still improving late: mean HQ rises from 71.6% to 76.5% for atoms and
75.2% to 78.4% for MoG between 81k–90k and 91k–100k. A 100k endpoint therefore
does not establish an asymptote either.

## Frozen-noise check

The [no-retraining intervention](NOISE_PROBE.md) still changes one-shot local
width without repairing global mode balance. Adding noise to the trained atom
GAN changes width .854 → 1.263, while SW1 barely changes (.7595 → .7585).
Removing noise from the trained MoG GAN changes width .701 → .451.

DDGAN barely changes in HQ or width when component noise is toggled at inference.
The MoG-trained DDGAN still reaches 79.75% HQ with component noise disabled.
DDGAN already has continuous observation/reverse noise, so its MoG training
advantage is not simply evidence that extra inference noise is necessary.
Noise toggling changes RNG consumption; individual sampled paths are not paired,
and small SW1 differences should not be treated as a definitive intervention win.

## Recommendations

1. **Capacity question:** compare the two one-shot models with G width 128 at the
   same 100k budget, holding D and all other settings fixed. Check whether the
   coverage/balance gap closes. These experiments alone do not separate capacity
   from the changed DDGAN objective, conditioning and repeated computation.
2. **Training stability question:** test learning-rate decay for the one-shot
   pair at the same budget. Their constant-LR late collapse makes optimizer
   behavior a plausible competing explanation; this would be an optimization
   control, not proof about capacity.
3. **If DDGAN HQ is the priority:** a longer matched DDGAN pair is justified by
   its still-rising HQ, but there is no evidence here for a particular budget
   guaranteeing that it reaches the one-shot HQ level.

## Controls, verification and reproduction

Only `steps` (14,000 → 100,000) and `out_dir` changed from the screen. G width 32,
D width 128, depth 3, 400 components, one shared seed (24002), standardization,
constant learning rates, optimizer settings, class-only UCD and Gaussian reverse
noise stayed fixed. The atom control remains `MoGParticlePrior(sigma_rel=0)`.

Training restarted from the same initialization because the earlier checkpoints
contain EMA inference weights without optimizer state. All 56 original
checkpoint records match exactly through 14k after excluding timing fields;
training source provenance also matches. All four completion certificates were
verified. Frozen baseline inference reproduced the final SW1 values within 1e-5.

```bash
.venv/bin/python -u experiments/run_grid.py \
  --config_manifest configs/denoising/mog_capacity_100k/manifest.json \
  --trainer experiments/train_denoising.py --gpus 0,1 --workers_per_gpu 1
.venv/bin/python experiments/analyze_mog_capacity.py \
  --manifest configs/denoising/mog_capacity_100k/manifest.json \
  --baseline-manifest configs/denoising/mog_capacity/manifest.json \
  --out reports/denoising-toy/mog_capacity_100k
.venv/bin/python experiments/probe_mog_capacity.py \
  --root results/denoising/mog_capacity_100k \
  --out reports/denoising-toy/mog_capacity_100k
tail -F results/denoising/mog_capacity_100k/progress.log
```

[Full tables](TABLE.md) · [Curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/mog_capacity_100k/curves.png) · [Samples](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/mog_capacity_100k/samples.png) ·
[Leaderboard CSV](leaderboard.csv) · [Milestones](milestones.csv) ·
[Comparison data](comparison.json) · [Prefix audit](prefix_audit.json).

Raw logs, checkpoints, source snapshots, samples and completion certificates:
`results/denoising/mog_capacity_100k/<run>/`. No training jobs remain active.
Reviewable copies of each run's metrics, summary/config, and completion
certificate are committed under `runs/<run>/` alongside this report.
