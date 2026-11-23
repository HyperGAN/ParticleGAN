# CIFAR training speed and toy shape audits

**Complete: 40 runs, no failures.** The promoted CIFAR recipe uses exact bcap
at four times its weight every fourth D update, cached frozen conditioning
features, fused Adam, and no unused regularizer scalar synchronization.
It reaches **31.555 final FID in 9.22 training minutes** at 10k updates.
The shared toy defaults remain unchanged. FD stays experimental.

The scope is the existing ParticleGAN DDGAN with joint UCD and bcap. We are
measuring implementation optimizations and finite differences for that same
local gradient-norm cap. No alternative cap, interpolation objective, diffusion
schedule, architecture capacity, or latent/step-noise formulation was added.

## CIFAR 10k promotion results

All three use batch 64, frozen-condition feature caching, fused Adam, NCHW,
and suppressed unused regularizer scalar statistics. Final FID uses 50k samples.

| Bcap implementation | Steady samples/s | Training minutes | Final FID ↓ |
|---|---:|---:|---:|
| Exact, every update | 644.6 | 16.59 | 32.821 |
| Exact, every fourth update ×4 | 1161.8 | 9.22 | **31.555** |
| FD, every fourth update ×4 | 1250.0 | 8.56 | 37.003 |

Exact lazy-4 saves 44% of training time against the matched cached/fused control.
It also halves the older 20.05-minute 10k baseline (FID 31.741). The small FID
improvement is not evidence of a quality advantage; the useful result is retained
quality at substantially lower cost. FD saves only another 40 seconds and loses
quality. It is not a default candidate at the tested epsilon and frequency.

See the [promotion leaderboard](promotions/TABLE.md) and
[winning full config](../../../configs/cifar_ddgan/speed_promote_unet/lazy4_10k.yaml).

## Short scout and toy findings

- Cached frozen noisy-input features improve U-Net steady throughput from
  513 to 627 samples/second (22%). Cached features are reused only within the
  current optimizer batch. Candidate feature gradients remain intact.
- Channels-last is slower for U-Net. Fused Adam adds only about 1% on top of
  caching for CIFAR, but helps the smaller toy networks more.
- U-Net exact lazy-4 reaches 1,164 samples/second; FD/lazy-4 reaches 1,255.
  Their 1k FIDs are 68.46 and 64.20, versus 69.07 for cached/fused exact bcap.
  These are early screens, not final quality measurements.
- Batch 128 and 256 improve throughput less than lazy bcap and worsen early
  FID at matched sample exposure. We keep batch 64 for CIFAR promotions.
- NCSN++ has a striking but not yet established numerical sensitivity. The
  full cache/channels-last/fused bundle gets 1k FID 59.33, versus 306.67 for
  the original path, with effectively unchanged speed. Cache-only gets
  82.94, fused-only 195.83, channels-last-only 331.49, and cache+fused without
  channels-last 162.14. These single-seed configuration comparisons do not
  establish a reliable numerical fix. The full bundle failed its 10k promotion: see below.
- The toy backport already catches a problem that 1k CIFAR FID missed:
  FD/lazy-4 retains all 100 one-shot modes at 10k, but only 85.87% of samples
  lie within 3 standard deviations of a center, versus 99.01% for exact bcap
  and 98.84% for exact lazy-4. Its tails and within-mode shapes are worse.
  In a radius-10-sigma core audit, 65/100 mode cores have under 10% of target
  variance along their narrowest axis, versus zero for the other variants.
- The 10k denoising toy has 63–64 well-covered conditional modes with exact,
  fused, or exact lazy-4, versus 32 with FD/lazy-4. Controls are still immature
  at this budget. At 56k, all four cover 100 modes; see below.

## Native 56k denoising confirmation

| Variant | Samples/s | Joint HQ | Modes | Mode TV ↓ |
|---|---:|---:|---:|---:|
| Exact | 30,725 | 92.35% | 100 | 0.035 |
| Exact + fused | 33,185 | 91.98% | 100 | 0.043 |
| Exact lazy-4 + fused | 47,495 | 91.46% | 100 | 0.034 |
| FD lazy-4 + fused | 45,732 | 90.17% | 100 | 0.046 |

The FD denoising run recovers all modes with enough training. It does not show
that toy's one-shot core-collapse pattern at this budget: zero of its 100
cores have minimum eigenvariance below 0.1× target, compared with three exact
and one exact lazy-4 cores. However, FD has broader cores/tails overall,
lower joint HQ and worse mass balance, and is slightly slower than exact
lazy-4 on the MLP. The one-shot FD failure is not a universal collapse claim.
Exact lazy-4 transfers substantially better across the tested domains.
Fused-only also preserves all 100 modes, with 91.98% joint HQ and an 8% speedup.
Exact lazy-4 adds a 43% speedup over that fused-only control.

## NCSN++ follow-up did not validate

The full cache/channels-last/fused bundle finishes 10k at **161.587 final
50k-sample FID**, taking 58.48 training minutes (68.46 minutes including
maintenance). Its 5k-sample diagnostic FIDs swing from 143.88 at 1k to 72.74
at 2k, 285.72 at 5k, 78.63 at 8k and 164.36 at 10k. The original 1k bundle's
59.33 was not reproduced in this promotion. This is not a seed experiment:
the training budget and shared implementation version differ; CUDA numerical
nondeterminism is also permitted. The result does not isolate the cause, but
it rules out promoting this bundle as a demonstrated stability fix.

## Recommendation and defaults

Use the [CIFAR default config](../../../configs/cifar_ddgan/default.yaml):
`reg_method: autograd`, `reg_every: 4`, `cache_condition: true`,
`fused_adam: true`, `reg_sync_stats: false`, `channels_last: false`.
Batch size stays 64. DDGAN, joint UCD, particles, Gaussian step noise, learning
rates and model capacity stay as before. For a pixel-only discriminator,
explicitly set `cache_condition: false` because it has no pretrained feature cache.

Do not promote FD at the tested epsilon/frequency. Its small extra CIFAR speed
gain costs quality, and it is not faster than exact lazy-4 on the toy MLPs.
Keep the toy no-argument defaults unchanged; full exact/fused/lazy/FD configs
are provided for review and reproduction. The one-shot lazy recipe is much
better than FD here, but even its within-mode shape is not identical to exact.

A future 50k CIFAR quality run with the promoted recipe is the useful next
validation, estimated at roughly 46 training minutes from this 10k run,
plus evaluation/I/O. It has **not** been run; the prior every-step 50k result
was FID 26.680. No further experiments are queued. No alternate bcap objective
or broad epsilon search is proposed in this session.

## Reports and configurations

- [CIFAR scout leaderboard](TABLE.md) and [quality/speed plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/cifar-ddgan/speed/scout_quality_speed.png).
- [Toy 1k screen](toys_1k/TABLE.md), [toy 10k screen](toys_10k/TABLE.md),
  [native 56k confirmations](toys_56k/TABLE.md),
  and [one-shot shape comparison](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/cifar-ddgan/speed/one_shot_10k_shape.png).
- `configs/cifar_ddgan/speed_*` contain full, reproducible CIFAR YAMLs.
- `configs/speed_100gaussians_*` and `configs/speed_denoising_*` contain toy YAMLs.
- `runs/` contains exported configurations, metrics, certificates and figures.
  Checkpoints, source archives, raw samples and profiler traces remain in
  ignored `results/` directories.
- Tail all work: `tail -F results/cifar_ddgan/speed.live.log`.

## What FD means here

At each real and fake candidate separately, with class, timestep and noisy
conditioning input held fixed:

```
v = normalize(gradient_x D(x))     # detached
s = [D(x + h*v) - D(x - h*v)] / (2*h)
penalty = coefficient/2 * (mean(relu(s_real - 1)^2)
                         + mean(relu(s_fake - 1)^2))
```

This uses a first input backward to choose the direction, then ordinary
parameter backprop through two perturbed evaluations. It avoids double
backward but adds forwards. It is not a secant between real and fake samples.
Bcap limits excessive local slope; it permits flat slopes and does not impose
constraints throughout the gaps between modes.

Lazy-4 applies the same penalty every fourth D update at four times its weight,
inside the existing optimizer step. We do not add a separate optimizer phase
or change Adam's learning rates/betas. NVIDIA's [StyleGAN2-ADA implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py)
uses separate regularizer phases and therefore also rescales optimizer settings;
that is a different update schedule.

CIFAR h=0.05 is an L2 image displacement. The toy uses
h=0.05*sqrt(2/3072)=0.0012757759 to match displacement RMS per input component.
This choice is explicit in YAML; it is not claimed to be universally optimal.
On a trained CIFAR D, h=0.05 estimates the gradient norm reasonably closely
but gives quite different parameter gradients (cosine about 0.37 in our small
probe). Nonsmooth networks and finite precision make this a real approximation.

## Shared implementation and validation

`lib/grad_regularizers.py` owns both the exact and finite-difference algorithms,
lazy weighting, and optional suppression of unused scalar statistics. CIFAR
and `train_denoising.py` call it. The YAML runner `train_100gaussians.py` calls
the actual `examples/100gaussians.py` training loop; it does not duplicate it.
Fused Adam is optional in all three entry points. The toy MLPs have no frozen
image encoder or convolution layout, so the image-specific cache/layout
optimizations have no direct toy counterpart.

Tests cover analytic FD norm and parameter derivatives, exact/lazy scaling,
shared FD agreement with the initial CIFAR implementation, cached logits,
candidate gradients and bcap parameter gradients, existing prior controls,
runner provenance, and four CUDA resume cases. Final validation: 53 tests plus
13 subtests, and four CUDA resume tests with the promoted cache/fused/lazy
settings, all passed. Pixel-D replay explicitly disables the unavailable cache.
On eight real samples from a saved discriminator, a zero-threshold cap audit
(to activate every sample) finds cached versus uncached parameter-gradient
relative error about 2e-4 with TF32 and 3e-7 with full FP32. This supports the
cache implementation but is not a proof of identical training trajectories.

## Measurement limits

Samples/second means batch size times updates/second per optimizer. CIFAR and
the Rp toy recipe draw separate real batches for D and G, so total real draws
are twice this count. Larger-batch scouts preserve sample exposure and adjust
EMA decay in sample units, but still have fewer optimizer updates.

Steady CIFAR throughput excludes the first 12,800 samples per optimizer,
including the brief profiler window in the two baseline runs. Baseline total
training minutes include profiler overhead; use steady throughput for speed
ratios. Evaluation/checkpoint I/O are excluded from training time. GPU1 also
drives the desktop; device assignments are recorded in the leaderboard.

CIFAR scouts use 5k generated samples for FID; promotion finals use 50k with
the existing cached CIFAR-train reference and TF-compatible Inception protocol.
Comparisons use the same seed, not seed sweeps. Small FID differences do not
establish equivalence or superiority. Mode count on the toy requires enough
near-center, correctly conditioned samples; undertrained broad distributions
can have poor mode count without assigning all mass to a few cells. Shape,
mass balance and quality metrics must be read together.
