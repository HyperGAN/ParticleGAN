# DCGAN speed and scaling readout

BISPF-v2 completes the declared 32–256-pixel RGB DCGAN workloads with finite
updates. Memory fits comfortably on the measured RTX A6000; update latency is
the larger practical concern. This measures execution on mock data and supplies
no image-quality or default-promotion evidence. The default is unchanged.

The [current Performance overview](../technique-inventory.md#performance) is the
single current comparison. Its [compact evidence index](../scaling-performance.json)
links every final metric and exact execution receipt. The
[protocol](../../../docs/forge-scaling-performance.md) describes the reusable
Forge commands, timing scope and comparison rules.

All **48/48** main cells completed and independently passed the numerical
execution checks: eight selected global recipes across six tasks. No cell was
retried. The [compact audit](audit.json) checks shared initialization and batch
sequences per task, one global recipe per candidate, common runtime/source/GPU,
full budgets, finite state and actual generator/discriminator parameter changes.

## What BISPF does

At batch 16 and base width 64, median complete-update times are **282.29 ms at
32×32, 609.52 ms at 64×64, 2,080.19 ms at 128×128, and 3,143.55 ms at 256×256**.
Peak allocated memory rises from 90.40 MiB to 2,119.28 MiB (2.07 GiB).

At 64×64, batch 64 takes 680.49 ms per update and 558.22 MiB, compared with
609.52 ms and 221.78 MiB at batch 16. The wider base-128/channel-cap-1024 host
takes 1,526.29 ms and 674.18 MiB. Batch, width and resolution therefore need
separate measurements; pixel count alone does not predict the complete update.
Larger resolutions also add DCGAN stages under the declared channel cap.

The selected BCAP DualNorm recipe has nearby observations: 278.99 ms at 32×32,
602.88 ms at 64×64, 2,054.91 ms at 128×128 and 3,091.40 ms at 256×256. These
whole-recipe observations do not isolate an individual BISPF mechanism. Both
recipes use exact per-offset DualNorm SVD updates. Component profiling is still
needed before attributing the latency to the optimizer, transport calculation
or higher-order critic gradients.

Recommendation: retain an explicit large-image performance decision before
promoting BISPF. Its measured memory footprint supports larger-image pilots,
while the update-time budget needs attention. Investigate the shared normalized
optimizer cost first, and consider larger batches where memory permits. Any
optimizer replacement is a new global recipe comparison and needs its own
scientific evidence; these timing results cannot establish preserved quality.

## Fixed workload and interpretation

The six tasks use seed 0, independent Uniform[-1,1] mock RGB pixels, latent
dimension 128, 256 common particle-cloud locations, three warmup updates and ten
measured updates. The finite-cloud exception permits all selected controllers
to run. Recipes retain their prior update policy, regularizer, optimizer and
other training mechanisms. The public deterministic initializer and separate
named data, constructor, training and evaluation streams are used throughout.

The prior comparison uses the state recorded by the public initializer. BISPF
then adds its recipe-owned moment-tether buffers, so its complete post-setup
prior-state hash differs without changing the initial locations. Receipts retain
both proofs; the original post-setup identity is not rewritten.

Generator and critic retain BatchNorm. A shared flattened 4×4 Linear critic head
allows the existing Atlas/E22 feature API; it has the same forward function
class and parameter count as the corresponding scalar convolution, with an
explicitly different dense-head optimizer layout. This is a fixed DCGAN-family
performance host, rather than a natural-image training-quality claim.

All main measurements use physical GPU
`GPU-ed080e41-3193-3755-6756-f3d46c433331`, FP32, deterministic execution and one
Forge worker. NVIDIA's real utility reports a driver/library mismatch, while the
PATH wrapper supplies static device listings. Contention telemetry is therefore
**unknown**. The overview preserves separate unpaired observations and makes no
certified speed ranking. PyTorch allocator peaks and CUDA-synchronized update
times are actual measurements. CPU portability is covered by separate small
API/adapter tests, not by substituting CPU observations into these cells.

The measured region includes a complete public `GANTrainer.step`, including
recipe controllers, discriminator, generator and prior work. Setup, mock-data
generation, hashing, evaluation, GIFs and checkpoint writes are outside it.
Ten measured updates give a short empirical p95, not sustained-policy behavior.

## Execution and provenance

The immutable campaign is `dcgan-performance-cloud-v2`, executed from source
commit `58ade47791367abd46c68ede8d5e7d9fea3dbafb`. Its declared reservation was
9,600 seconds. The 48 core benchmark wall times sum to 414.79 seconds, while
Forge's completed-event worker wall times sum to 752.50 seconds; neither is the
sum of synchronized update times. Campaign progress spans 01:16:55–01:46:02 UTC
on 2026-10-17, including scheduling, collection and other control-plane work.
Exact cost scopes, receipt hashes and timestamps are retained in the audit.

The preliminary `dcgan-performance-cloud-v1` queued five requests but ran no
attempts. They were cancelled before execution after hardware-identity validation
was corrected; its frozen registration remains historical provenance. It supplies
no timing observations. Main v2 evidence was collected once and published without
training again or regrading scientific qualification evidence.

## Software validation

All 63 new benchmark/registration/export/publication tests pass, alongside 293
related integration checks and all 112 family/tier report tests. The generated
display refresh preserves scientific rows, selection pins, recorded qualification
sections and all 48 performance receipt hashes. Forge declaration validation,
generated performance checks and compiled-memory freshness pass.

Baseline comparisons reproduce 25 existing BCAP compatibility failures and eight
existing inventory-fixture failures unchanged on `develop`. The broader CPU sweep
was bounded at 1,200 seconds and terminated before completion; it supplies no
full-suite pass verdict. The [validation receipt](validation.json) retains exact
commands, scoped results and hashes of local verification logs. Repository CI
currently runs on `master`, so this `develop` PR has no CI jobs.

## Actual-training illustrations and artifacts

Metrics determine the result. These small BISPF GIFs illustrate the actual
initial, warmup and final training states for each numerical execution test:

- [32×32](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/scaling-media/dcgan-performance-cloud-v2/dcgan_performance_r32.gif)
- [64×64](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/scaling-media/dcgan-performance-cloud-v2/dcgan_performance_r64.gif)
- [128×128](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/scaling-media/dcgan-performance-cloud-v2/dcgan_performance_r128.gif)
- [256×256](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/scaling-media/dcgan-performance-cloud-v2/dcgan_performance_r256.gif)
- [Batch 64](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/scaling-media/dcgan-performance-cloud-v2/dcgan_performance_b64.gif)
- [Base width 128](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/scaling-media/dcgan-performance-cloud-v2/dcgan_performance_w128.gif)

Every run also retains its own GIF, full checkpoint, named RNG states, timing
trace and stdout in the local artifact archive. Compact receipt provenance
records their original paths and hashes. Bulk artifacts are not committed.

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/dcgan-performance-cloud-v2/progress.jsonl
python -m experiments.forge scaling plan dcgan-performance-cloud-v2
python -m experiments.forge scaling publish dcgan-performance-cloud-v2
python -m experiments.forge.scaling_publication --check
```
