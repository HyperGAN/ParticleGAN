# Optimized BISPF investigation

The [current technique inventory](../technique-inventory.md) remains the single
comparison board. This readout records component diagnosis and bounded optimizer
iterations. The [protocol](../../../docs/bispf-optimization.md) freezes comparison
rules and performance targets. The surviving quintic recipe passes the declared
latency targets and preserves all 19 incumbent numerical passes. The
[selection proof](selection-proof.json) binds ordinary quality, the exercised
wide-channel path and matched performance to the same frozen source.

## Measured bottleneck

Unwrapped public `GANTrainer.step` medians on the existing DCGAN mock workloads:

| Workload | Native BISPF | R1/R2 | Native BISPF SVD share of instrumented update |
| --- | ---: | ---: | ---: |
| 64×64, batch 16 | 605.73 ms | 25.66 ms | 87.5%, 145 SVD calls |
| 256×256, batch 16 | 3,067.20 ms | 252.25 ms | 88.5%, 209 SVD calls |

The SVD share comes from a separate synchronized component replay. It does not
replace plain complete-update latency. Both replays consume identical initial
models, public prior initialization, batches and RNG streams and produce
identical final learned parameter hashes. Generator optimizer timings include
its prior groups; only prior sampling and regularization have separate spans.
Sinkhorn coupling and prior regularization each take a few milliseconds in
these workloads. Neither explains the original latency gap.

All observations use the same physical RTX A6000. Contention telemetry is
unavailable, so ratios describe observed runs and confer no certified speed
ranking. The [compact profiling receipt](profiling.json) records exact source
maps, declarations, runtime, hardware, component measurements, replay checks,
artifact identities and cost scopes. The original baseline combines core
`b5d2372f2` with the separately hashed added profiling harness; it is explicitly
a mixed source snapshot. Subsequent profiles bind complete committed trees.

## Retained optimizer iterations

The initial coupled Newton–Schulz solver formed a regularized channel Gram
matrix, then checked the inverse-root residual against that original matrix.
Although random matrix microbenchmarks were fast, real training gradients
triggered native SVD fallback for about 80 of 144 convolution slices per update
at 64 pixels. Plain medians were 515.81 ms at 64 pixels and 2,962.35 ms at 256
pixels. These results rejected that implementation as a latency solution;
neither its quality study nor its full performance campaign was launched.

The second solver orthogonalizes the augmented matrix `[M; lambda I]`, keeping
the absolute smoothing scale without squaring the original condition number
before iteration. It checks both augmented orthogonality and the relation of
the result to the original gradient. This removes all observed late-window
convolution fallbacks at 64 pixels, with a plain median of 102.02 ms. Its
24-iteration limit still rejects large critic kernels at 256 pixels, giving
1,513.20 ms. This iteration also has no full quality or performance campaign execution.

The third solver adds optimizer-local CUDA graphs, checks from eight through
32 iterations, and retains the same fail-closed numerical guards. At source
`4d70a8039e1e5d969d01c664dec0c2b4ea47f612`, plain medians improve to
71.74 ms at 64 pixels but remain 1,426.41 ms at 256 pixels. Peak allocated memory
rises from about 222 to 349 MiB at 64 pixels, and from 2,119 to 2,344 MiB at 256.
The first update pays graph-capture overhead. The large-image target still fails.

Its ordinary quality study was started before the 256-pixel profile finished,
then canceled when that profile failed the target. Preserve the completed
Gaussian smoke PASS (attempt `697e75ed0af449a7b3beb42dfcbd3763`, 1,000
updates, 23.45 seconds worker wall time) and the canceled two-pole attempt
`a6c0a67123eb4403b9dc10b5af0a9436`. Neither constitutes full qualification.
The other ordinary tasks and all six performance-matrix arms were unexecuted.

A separately bounded [guard capture](guard-capture.json) reproduces the original
final state and captures every failed batch rather than one representative per
shape. Eager, cold-cache and warm-cache replay are bitwise equal: the graph
cache is not the cause. Large early critic gradients make the fixed absolute
smoothing scale too small for reliable FP32 inverse-root recovery. Raising the
iteration count can increase relation drift after a slice briefly converges.
The original guards correctly reject those slices. The [kernel audit](KERNEL_AUDIT.md)
retains the actual-input solver alternatives, prototype identities and negative
results; its microbenchmarks provide no training-quality credit.

Every successful approximate branch omits native SVD's hard numerical-rank
cutoff. This is an explicit numerical trainer change. On one captured 512×512
critic gradient, the converged unmasked result differs from the masked native
direction by about 1.16% in relative Frobenius norm. Small residuals alone do
not establish equality to the masked update or preservation of image quality.

## Alternative optimizer result

The cheap Frobenius rule removes the bottleneck but fails both direct training
diagnostics. At the original rates and 600-update budget it collapses stripes
to one mode and retains only two of four bars modes. Its HQ-only automatic study
prediction says `prediction_observed`, but the actual numerical gate is FAIL.
The immutable prediction is retained and the recipe is stopped; no full quality
or performance-matrix execution follows.

The surviving `bispf-optimized-quintic-v1` changes one global field,
`optimizer_convolution_rule="quintic"`. It applies five fixed FP32 steps with
coefficients `(3.4445, -4.775, 2.0315)` to the augmented matrix `[M; lambda I]`,
then divides the top block by `1.21`. The filter has an operator bound in exact
arithmetic, retains absolute smoothing and the original channel/kernel factor,
and introduces no momentum or learning-rate retuning. It is an explicitly
different optimizer inspired by Muon's polynomial, with no SVD equivalence or
inverse-root convergence claim. [Kernel details and scaling](KERNEL_AUDIT.md)
explain the remaining forward/backward and activation-memory costs.

At frozen source `d9d72d53761c898bce20102e1995fc3d899c1447`, all 18 final
performance arms complete with the same initialization, batches, runtime and
physical GPU. At 64 pixels, native 608.99 ms becomes 51.42 ms (11.84x), versus
R1/R2 24.72 ms. At 256 pixels, native 3,115.15 ms becomes 338.70 ms (9.20x),
versus R1/R2 252.05 ms. The declared observed targets pass: at least 8x native
at both sizes and at most 2x R1/R2 at 256. Contention remains unknown, so these
are observed ratios rather than a certified paired ranking. Exact final matrices
are archived for the [controls](../scaling-history/bispf-matrix-controls-v1-d0053fc17b21f7a7.json)
and [quintic recipe](../scaling-history/bispf-matrix-quintic-v1-7c73a339487b7bfb.json).

![Observed DCGAN update latency versus image resolution: optimized BISPF approaches R1/R2 while native SVD costs dominate.](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bispf-optimized/resolution-scaling.png)

The [vector plot](resolution-scaling.svg) and [renderer](reproduce_scaling.py)
use the four resolution tasks only, holding batch size and width fixed. They
read the exact archived manifests and validated original initialization proofs.

Across all six tasks the native speedup ranges from 7.61x to 17.43x; it is not
uniform. Relative overhead versus R1/R2 decreases from 2.94x at 32 pixels to
2.08x at 64, 1.68x at 128 and 1.34x at 256. Increasing batch size to 64 at
64 pixels gives 88.16 ms versus R1/R2 60.83 ms, while doubling base channels
to 128 gives 107.13 ms versus 54.31 ms. Wider matrices still increase the
fixed polynomial cost; larger images increasingly spend time in the shared
network computation. Peak allocated memory matches native BISPF on every
workload, including 2,119 MiB at 256 pixels, without persistent graph caches.

The [six-arm direct training readout](wide-quality.json) passes all shared
initialization, public-prior, task, initial RNG and final consumed-data-stream
checks. Native BISPF passes stripes and fails the sustained bars gate; quintic
passes both. Each run uses the same 600 updates and 24 observations. Quintic's
stripes first pass is at update 325 versus native 275; both retain the required
five-check suffix. Quintic bars retain nine passing terminal checks versus
native two, with endpoint HQ 1.0 versus 0.90625. This improves that sustained
gate, not every metric: endpoint bars distribution TV is 0.1875 versus native
0.125. Actual-training GIFs and exact receipts are linked in the readout and
add no training updates or sampling draws. Total charged worker time is
215.20 seconds of the declared 1,800-second campaign.

## Quality scope

The [completed ordinary readout](quality.json) contains all 30 task verdicts:
19 PASS and the same 11 historical FAILs, with no regression. Its 29 original
execution groups consume 3,155.05 worker seconds of the declared 46,620-second
cap, with no retries or canceled tasks. The ring protocol confirms convergence
at update 1,782, then passes all 1,200 disjoint hold checks and 300 extension
checks, ending at update 3,282 within its 7,500-update maximum.

The five-word smoke completes its entire 20,001-update allowance, with 23 of
24 scheduled checks passing. One late failure at update 15,835 recovers; it
remains explicit in the numerical readout. The declared smoke gate accepts a
confirmed scheduled state after budget completion. Its separate hold resumes
the earliest confirmed passing checkpoint at update 834 and passes all checks
over its own 4,000-update continuation. Endpoint success alone supplies no hold
credit. Saved-training GIFs illustrate these numerical outcomes without adding
updates or sampling draws.

The actually executed revision-11 policy qualifies Tier 3: all 17 counted gates
and both positive diagnostics pass. The inventory retains its revision-9
comparison policy so other techniques keep their original denominators. Under
that policy, BISPF remains Tier 1 / FAIL, with tier counts 6/6, 10/21 and 2/2,
plus the passing clock diagnostic. The separate executed-policy proof is retained;
none of the 11 held failures is relabeled as a success or omitted. This is a
recipe selection with preserved evidence, not a policy migration or public
default adoption.

CPU kernels, tiny convolution kernels, dense weights, vector updates and sampled
prior-row updates retain native behavior. All current image-quality hosts have
at most 24 channels, so a complete tie on the existing suite would preserve
known quality results while exercising no accelerated convolution there.

The separate width-64 stripes and bars diagnostics widen only the original
architecture and declare their shorter wall allowance explicitly. Within each
new task, native and optimized BISPF share architecture, data law, initial
prior, public initializer, seen batches, sampling, 600 updates, 24 observations
and the original numerical quality gates. They exercise the accelerated path
and retain actual-training GIFs. This portability scope provides no ordinary
qualification and does not establish quality on large natural images.

The inventory now selects the quintic recipe for BISPF: it preserves known
passes, improves the exercised sustained bars gate and meets every declared
latency target. Native polar behavior remains the API default, and the overall
public trainer default remains BCAP. Larger natural-image quality remains
unmeasured.

The [publication integrity receipt](publication-integrity.json) verifies that all
eight other scientific rows, seven other selection pins and 42 other performance
cells remain unchanged. The previous selection is archived byte for byte.
[Software checks](software-checks.json) retain the optimizer, checkpoint,
selection, source-reconstruction and final generated-page results, including
the earlier publication failure and its verified resolution.
