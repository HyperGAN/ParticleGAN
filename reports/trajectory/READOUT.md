# First trajectory round: fast transfer, incomplete variation and extrapolation

All **12 GPU runs completed successfully**: six 1k viability scouts and six
10k comparisons, with one worker per GPU. Total training across both GPUs:
10.72 GPU minutes. Each 10k model trained in 91–100 seconds. Both GPUs are
free, nothing is queued, and no CIFAR or Gaussian-toy defaults were changed.

Open [the interactive recipe gallery](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/round1/index.html), the
[baseline viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/round1/learned/viewer.html), or the
[animated futures](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/round1/learned/futures.gif).

## Results at 10k updates

Lower conditional SW1 is better. Validity is the declared joint geometric
criterion, not just collision avoidance. Interpolation uses six held-out
contexts; extrapolation uses two contexts with a larger, higher obstacle and
higher starting point. Test SW1 mixes all eight contexts equally.

| Recipe | Test SW1 ↓ | Route TV ↓ | Interpolation valid | Extrapolation valid | Test collisions | Within-route variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDGAN + UCD + learned prior + Gaussian noise | **.0466** | .0466 | 99.4% | 21.4% | 6.3% | .35 | 100.2 |
| DDGAN + concat + learned prior | .0483 | .0390 | **99.8%** | 14.8% | 6.0% | .33 | 99.3 |
| DDGAN + UCD + Gaussian prior | .0498 | .0615 | 99.7% | 21.3% | 7.1% | .34 | **90.8** |
| DDGAN + UCD + learned prior + learned noise table | .0506 | .0625 | 97.9% | 37.4% | **2.8%** | .26 | 99.7 |
| One-shot GAN + UCD + learned prior | .0520 | **.0271** | 98.8% | 0.0% | 11.4% | .29 | 93.1 |
| DDGAN + UCD + learned prior + fixed noise table | .0637 | .1198 | 96.4% | **49.7%** | 3.1% | .45 | 99.6 |

Real-vs-real test floor: SW1 .0070, route TV .0156, validity 100%, collisions 0%.
Variance ratio is the median across interpolation context/route pairs of
generated-to-real mean coefficient variance, restricted to valid generated
paths. Target is 1. It is a calibration diagnostic, not an overall quality score.
See [machine-exported table](round1/TABLE.md), [leaderboard JSON](round1/leaderboard.json),
and each run's full context metrics in `round1/<recipe>/summary.json`.

## Answers to the four questions

1. **Does DDGAN beat a typical GAN?** Mixed. The particle DDGAN improves pooled
   held-out SW1 and extrapolation validity over our matched one-shot particle
   GAN. But one-shot wins training-context SW1 (.0192 versus .0283) and route
   probabilities. Its 0% extrapolation validity means every sample fails at
   least one joint threshold, not that every path collides. One-shot's D lacks
   noisy-future inputs/time heads and is smaller (169,730 versus 204,040 params).
   This is a useful baseline comparison, not isolation of diffusion alone.
2. **Does UCD help?** No clear overall advantage. It slightly improves pooled
   test SW1 and extrapolation validity relative to concat, but concat improves
   route calibration and training-context SW1 (.0244 versus .0283).
3. **Do latent particles help?** Small observed gains in test SW1 and route TV
   versus Gaussian latent draws, at about 10% extra training cost. Validity and
   within-route spread are nearly tied. This tests the full learned-prior
   recipe, including particle optimization/VICReg. It does not establish a
   broadly useful particle advantage.
4. **Does learned particle step noise help?** It reduces test collisions and
   improves extrapolation validity, while worsening SW1, route calibration,
   interpolation validity and within-route spread. Fixed noise particles also
   improve extrapolation, more strongly, but have the worst route calibration
   and SW1. There is no basis to promote either noise table from this round.

No seed-only repetitions or significance claims. All six use the same seed,
sample exposure, constant learning rates and shared source fingerprint. Each
10k optimizer phase sees 1.28M examples; total real draws per run are 2.56M.
Different computation/random-draw paths and CUDA arithmetic permit different
trajectories. A 1k ranking is not predictive: one-shot initially has test SW1
.0287 and 87.1% validity, then finishes worse on both at 10k. The endpoint alone
does not diagnose why; no intermediate checkpoints/metrics were selected.

## What the visualizations show

All six recover both routes in all training contexts, but none reproduce the
target's full continuous variation. Typical within-route variance is only
26–45% of target in interpolation contexts, even with validity near 100%.
The three interpolation geometries usually look reasonable; the extrapolation
case exposes misplaced arcs, invalid boundaries and obstacle intersections.
The plotted gray reference paths help separate plausible-looking output from
the specified target distribution.

The baseline's [particle probe](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/round1/learned/particle_probe.png) supports the
intuition that one particle can participate in multiple outputs. Holding ID 1
at every reverse step and rerolling all Gaussian randomness yields 6 upper and
18 lower routes, all 24 valid. ID 2 gives 19 upper / 5 lower; ID 3 gives 17 upper / 7 lower
(23 valid). ID 0 gives 24 upper routes. All four probes contain 24 distinct floating
outputs each. These are four illustrative IDs under one context, not a census
of particle specialization. Terminal and reverse-step randomness both vary;
the probe does not isolate reverse-step noise alone. Ordinary sampling draws
fresh particle IDs at each reverse step. [Numeric audit](particle_probe_audit.json).

## Recommendation

Keep the current particle DDGAN/UCD/Gaussian-noise recipe as the experimental
default; it has the best observed pooled test SW1, without claiming the other
metrics agree. This round validates a useful non-image testbed, not a solved
trajectory problem or a general advantage for particles.

The next focused round should target **within-route calibration and context
generalization**. A broader geometry distribution would test whether the
extrapolation failure mainly reflects training coverage; a D with temporal
features or explicit multiscale sequence inputs could test whether D misses
small-scale path variation. Keep the adversarial/DDGAN formulation and constant
LR. These are proposed experiments, not queued jobs. Avoid another long run
solely to improve route counts: those already look good while dispersion does
not. The present task has two routes, three continuous coefficients, and a
simple straight observed past; real motion remains a meaningful next step.

## Implementation and checks

- New `lib/trajectory.py`, `lib/trajectory_visuals.py`, trainer, analyzer, full
  YAMLs and seven unit tests. Existing loss/schedule/prior implementations are
  imported without modification. Only `.gitignore` changes an existing file.
- Shared DDGAN posterior, Rp logistic, joint time/class UCD with no c/t input
  to the D backbone, exact lazy-4 bcap, unique-particle VICReg and EMA. No
  supervised reconstruction/collision loss, projection or endpoint clamping.
- Seven tests pass: analytic real validity/route probability, between-frame
  collision, metric detection of collapse/invalid paths, candidate/particle
  gradients, joint UCD label exclusion, final posterior, fixed-randomness
  sampling, alternative arms and learned step-noise gradients.
- All 12 completion certificates checked against current trainer/lib source;
  no startup warnings, tracebacks or runner failures. Every run used CUDA;
  GPU 0 and GPU1 both used. Parameter counts: G 102,402; joint-UCD D 204,040;
  concat D 203,777; one-shot D 169,730; learned prior 640,000. The learned step
  noise table adds 131,072 parameters; fixed/Gaussian tables have no learned
  noise parameters.
- All six exported viewers passed a Node DOM/canvas-stub control/playback
  check (not a full browser rendering test). The 64-frame GIF decodes; route
  panels and the baseline particle probe were visually inspected.
- Raw samples, EMA checkpoints and exact source archives remain under ignored
  `results/trajectory/`. Reports preserve summaries, configs, metrics, hashes,
  certificates and visualizations. Full reproduction commands are in
  [README.md](README.md). Tail log: `results/trajectory/live.log`.

[All 12 runs](all_runs.json) · [Default config](../../configs/trajectory/default.yaml)
