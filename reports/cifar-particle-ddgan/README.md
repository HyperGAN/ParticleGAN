# Direct particle AE-GAN versus DDGAN at 10k updates

The direct models win this CIFAR-10 comparison. Direct GAN reaches FID50k
**19.48**, and direct particle AE-GAN **20.05**. Adding the particle encoder improves
DDGAN from **49.47 to 43.23**, but neither DDGAN variant catches the direct
models within this budget. All four same-count FID5k curves improve through
10k, without the previous direct scout's late regression.

See the [leaderboard and ablations](LEADERBOARD.md), [exact protocol](PROTOCOL.md),
[machine-readable results](leaderboard.json), and [learning curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/cifar-particle-ddgan/learning_curves.png).

| Model | Final FID50k ↓ | Training minutes | What it provides |
|---|---:|---:|---|
| Direct GAN | **19.483** | 7.91 | Prior-based generation |
| Direct particle AE-GAN | **20.054** | 8.77 | Generation plus an encoder and reconstruction |
| DDGAN + particle AE | **43.233** | 10.65 | Four-step generation plus an encoder |
| DDGAN | **49.475** | 9.10 | Four-step generation |

These are final EMA checkpoints after **10,000 training updates**, evaluated on
**50,000 generated images**. All use the same unconditional CIFAR task, fixed-sigma
MoG prior (1,024 particles, 64 dimensions), update budget, and learning rates.
The encoder objective is the only intended change within each architecture pair;
initialization and all training random streams match. Cross-architecture comparisons
have different networks and sampling costs: direct G has 645,123 parameters and
one call per generated image; DDGAN G has 1,035,491 parameters and four calls.
No training seed sweeps were performed.

## What worked

The DDGAN encoder contributes real information. At the noisiest timestep, its
clean-prediction test MSE is **0.0972**. Shuffling encoded codes while holding the
noisy input fixed raises that to **0.3982** (4.10 times); replacing only the selected
particle, keeping the offset, raises it to **0.1514**. At low noise the noisy-image
path supplies most of the information, so code shuffling has a smaller effect.
These measurements are one-step denoising, not latent-only or full reverse-chain
reconstruction.

Direct particle AE-GAN has test reconstruction MSE **0.0790**, versus **0.3049** with
zero offsets and **0.1287** with shuffled particles. Both particle identity and
offset matter, with offsets carrying much of the information. The encoder adds
about **10.9% training time** over direct GAN. For DDGAN, it adds **16.9%** and
reduces FID by **12.6%**. Direct GAN's 0.57-point FID advantage over direct particle AE-GAN
is small; this single matched experiment does not establish a robust small-margin
ranking across retraining or other budgets.

## Variation without inspecting images

Frozen checkpoints were tested on 512 held-out inputs with eight draws each.
For DDGAN, X_t and t remain fixed, and metrics measure the predicted clean image
before reverse-transition noise. Matching noisy-input hashes were verified across
the two DDGAN checkpoints. No audit images were saved or inspected.

```
E(X) -> z_X
Direct: G(z_X + 0.5 * sigma * noise)
DDGAN:  G(z_X + 0.5 * sigma * noise, fixed_X_t, t)
```

At half-sigma, both encoders produce eight distinct quantized outputs for every
input. Direct particle AE-GAN has pair pixel RMSE **6.96** on the 0–255 scale,
reconstruction MSE **+1.93%**, and **99.78%** nearest-own-reconstruction retrieval.
DDGAN + particle AE's changes are smaller: RMSE **0.86–4.06** across noise levels,
MSE increases **0.08–0.48%**, and retrieval is **100%**. Its feature pair distance
is 1.35–5.01% of the distance between unrelated deterministic reconstructions;
direct particle AE-GAN is 12.31%. These are modest local variations, not proof of semantic
diversity, preserved identity, or calibrated posterior samples. The encoders are
deterministic; stochasticity is injected externally. There is no KL or ELBO.

## Limits and what to try next

Halving all learning rates was the extra stability experiment. Compared with the
[previous direct scout](../cifar-particle-ae/README.md), final particle AE-GAN FID improves
from 33.27 to 20.05, and direct GAN from 81.41 to 19.48. The previous late decline
is absent here. Reconstruction gets worse (0.0634 to 0.0790), so the lower rate
is not an across-the-board improvement. Intermediate numbered checkpoints now
remain available for every arm.

The prior still is not matched to encoded codes. DDGAN + particle AE has about 433 effective
particles and 50.7% of offset coordinates near saturation; direct particle AE-GAN has 295
and 31.1%. Uniform particle/Gaussian-offset sampling differs from encoder usage.
The historical class-conditioned DDGAN score of about 31.56 at 10k used labels,
a different discrete prior, learning rates and auxiliary loss, so it is context,
not the control for this encoder experiment.

**Recommendation:** keep direct particle AE-GAN at these lower rates as the cheap baseline
when encoding/reconstruction matters; use direct GAN for generation alone.
DDGAN + particle AE helps its matched baseline, but these results do not justify
scaling it up yet.

**Next experiment, after compaction:** return to the toy problem to formulate and
test a genuinely variational model against deterministic particle AE-GAN. The
variational objective and experiment protocol have not been chosen; no new run
has been started.

## Cost and verification

The four full jobs took **26.8 minutes wall time across two A6000s**, **52.11 summed
GPU-process minutes including evaluation**, and **36.42 training minutes**. Pilots
added 1.02 process minutes; all three variation audits added about 2.06. Peak CUDA
allocated memory was 5.34 GiB including FID. Process minutes include CPU/I/O time
inside GPU-assigned jobs, not an accelerator utilization measurement.

28 tests and 13 subtests passed. Four 200-step pilots and all four full runs
completed without failure. The analyzer verifies source/config completion
certificates, complete budgets, initialization and RNG matching, fixed sigma,
frozen critic features, saved reconstruction arrays, all 16 checkpoint hashes,
and variation metrics against per-input arrays. Checkpoints and full source
archives remain under `runs/cifar_particle_ddgan/`; portable metrics and configs
are copied into this report. Main-worktree tracked source was not modified.

```
tail -F runs/cifar_particle_ddgan/scout.live.log
tail -F runs/cifar_particle_ddgan/variation_*.log
```
