# Conditional trajectories with Particle DDGAN

**Hybrid discriminator round complete:** [results and interpretation](hybrid/READOUT.md),
[interactive comparison](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/hybrid/confirm_10k/index.html),
[next-session motion handoff](hybrid/MOTION_HANDOFF.md).

**Geometry × discriminator round complete:** [results and interpretation](diversity/READOUT.md),
[interactive comparison](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/diversity/confirm_10k/index.html).

**First round complete:** [results and interpretation](READOUT.md),
[interactive gallery](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/trajectory/round1/index.html), [leaderboard](round1/TABLE.md).

Feature branch: `experiment/trajectory-ddgan`. CIFAR and Gaussian-toy defaults
are unchanged. Entry point: `experiments/train_trajectory.py`; no arguments
loads `configs/trajectory/default.yaml`. All settings are YAML-controlled.

## Task

Predict a 64-position, two-coordinate future from an eight-position observed
approach, obstacle height/radius, and a discrete route-preference label.
Preference 0 requests 80% upper / 20% lower paths; preference 1 requests
30% upper / 70% lower paths. Actual route and path coefficients are hidden.
Each route contains three independent uniform continuous coefficients for
longitudinal progression, clearance, and lateral asymmetry. There is no finite
training dataset: independent analytic samples are drawn each optimizer phase.

Four training geometries × two preferences give eight training contexts.
Four unseen geometries × two preferences give eight evaluation contexts:
the first three are interpolation probes, the fourth is an extrapolation probe
(higher starting point, obstacle height and radius). Context-specific results
are retained; the pooled test score mixes these two types of generalization.
An observed past is included explicitly; this first task only varies its height,
so it does not establish general understanding of arbitrary past motion.

## Formulation and architecture

Reuse `DiffusionSchedule`, `DrawSource`, `GANLoss`, `GradRegularizer`, and
`VICRegLikeLoss` from the existing implementation. G predicts the clean whole
future; the shared DDGAN posterior forms the candidate at the preceding
diffusion timestep. Four reverse steps, alpha_bar `[1,.9,.5,.05,.0001]`, fresh
latent draw each reverse step, Gaussian terminal and forward noise. Reverse
step noise is Gaussian by default; the last step has zero posterior variance.

G is a small temporal convolutional U-Net with full-resolution skip connections,
particle/context/time modulation, and a sequence-position channel. D is a
three-hidden-layer MLP over the candidate, noisy future, and observed context.
Joint UCD has eight heads, `(t-1)*2+c`; class and diffusion timestep are excluded
from the D backbone, while the observed continuous context is supplied to it.
UCD cross entropy is applied to real and fake logits only in the D phase.
G receives the selected-head adversarial loss, not an extra classification loss.

Rp logistic; bcap kappa=1, coefficient=1, exact input gradients, every fourth D
update ×4; selected unique learned particles receive existing VICReg. Separate
real batches for D and G; Adam beta1=0/beta2=.999, G LR=.0006, D multiplier1.5,
prior multiplier10, constant LR, EMA .995, fused Adam. No supervised path loss,
collision loss, oracle projection, endpoint clamping, or alternate sampler.

## Metrics and visualization

Evaluation is endpoint-only, 512 generated and independent real samples per
context. Real-vs-real reference floors use the identical metrics. Report:

- Conditional sliced Wasserstein-1 on the complete flattened trajectory.
- Upper-route probability error (TV for this binary choice), including invalid
  outputs; this cannot alone demonstrate good paths.
- Continuous segment/circle collision rate, including between sampled frames.
- Distance to the known bounded path family, maximum start/end position error.
- Validity: no collision, boundary error <.1, support RMS distance <.05. These
  are declared diagnostic tolerances, not learned thresholds or training losses.
- Coverage: valid mass in a route must exceed 5% of its target mass.
- Within-route coefficient variance relative to real samples, computed only
  for valid generated paths with >=10 samples. Missing routes yield null, not
  fabricated zero variance. Full trajectory SW1 and residual distance accompany
  this projection-based diagnostic so it cannot hide off-family errors.

PNG route panels and probability bars, `futures.gif`, and self-contained
`viewer.html` (scene/preference/physical-time controls) are saved per run.
Particle probes fix particle ID 0,1,2,3 at all diffusion steps and reroll noise;
a second probe fixes terminal/step randomness and varies particle sequences.
These are interventions, not claims about a single persistent particle in
ordinary sampling. The arbitrary four IDs are illustrative, not an exhaustive
particle-coverage audit.

## First round

Learned latent prior vs fresh Gaussian latent prior; matched architecture,
initial G/D weights, batch128, optimizer settings and sample exposure. The
learned arm includes its usual particle optimizer/VICReg, so this compares
complete prior recipes. No seed-only repeats. First 1k viability scouts, then
10k comparisons if execution is sound. A 1k quality ranking is not a conclusion.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/follow_grid.py \
  --root results/trajectory/scout_1k --log results/trajectory/live.log -- \
  --configs 'configs/trajectory/scout_1k/*.yaml' --gpus 0,1 --workers_per_gpu 1 \
  --trainer experiments/train_trajectory.py
tail -F results/trajectory/live.log
```

For the longer pair replace `scout_1k` with `confirm_10k` in both paths. Logs
report training losses/progress every250 updates; metrics are inspected only
after both jobs complete. Saved configs, environment, exact source archives,
hashes, raw sample arrays, EMA checkpoints and grid completion certificates
make runs reviewable. Checkpoints are inference artifacts, not full optimizer
resume checkpoints. Training throughput counts samples in one optimizer phase;
total real draws are twice updates×batch. No FID or pretrained image network.

Additional controlled comparisons, now tested in the first round: `d_mode: concat`,
`model: gan` (one-shot), and `noise: learned`/`fixed` (reverse-step tables only,
unconstrained except their GAN gradients). Configs live in `ablations_1k` and
`ablations_10k`; none are established overall wins.

Export a matched-budget comparison:

```sh
.venv/bin/python experiments/analyze_trajectory.py \
  --roots results/trajectory/confirm_10k results/trajectory/ablations_10k \
  --out reports/trajectory/round1
```
