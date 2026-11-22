<a name="precision-stretch"></a>

# Blackwell-format low-precision training stretch tests

Two optional Tier 1 tests ask whether each family's unchanged global recipe can
learn `N(2, 0.5²)` from scratch with **NVFP4 (4-bit)** or **MXFP8 (8-bit)**
linear training. `precision_gaussian_fp32` is their matched diagnostic control.
They belong to `discriminator_stability` revision 15, never gate required
qualification, and appear in the existing technique inventory's stretch table.
The new dated full-suite list includes the nine existing GAN formulation
stretch tests and both precision tests, preserving all 49 older tests.
The [view archive receipt](view-provenance.json) pins the initial proposal and
the exact updated `develop` parent under their original Git/blob identities.

## Frozen question and numerical gates

All three tasks use the same 32-wide two-hidden-layer MLPs, scalar output,
Fourier-2 critic, 256-location two-dimensional MoG (sigma 0.025), batch 128,
public deterministic initializer, protocol seed 0 and 1,000 updates. Only the
linear execution precision changes. Target batches, public prior sampling,
losses, optimizer settings, prior learning/freezing and regularization remain
recipe-owned and identical across these conditions for a given family. No
per-family or per-precision tuning, extra updates or seed variants are selected.

There are 24 scheduled observations of 4,096 clean/live samples. The final five
must all have finite samples, normalized mean error <= 0.2, standard-deviation
ratio in [0.8, 1.2], and exact Gaussian CDF KS <= 0.05. The analytic scorer is
the existing scalar acquisition scorer, with its target and destructive controls;
histogram bins affect GIF presentation only. Each task reserves 300 seconds,
including evaluation and its actual-training GIF. The three tasks add at most
900 seconds per candidate to a through-Tier-1 plan. Existing campaigns retain
their original caps; inspect the expanded plan before admission.
The read-only CPU BISPF plan reserves 5,340 seconds through Tier 1 and 70,620
seconds through Tier 4. The historical 46,620-second per-candidate inventory
campaign does not cover that full Tier 4 scope; a full-suite submission needs
an explicitly adequate campaign. No full-budget precision run was launched here.

## Meaning of simulation

This is a numerical software reference of the linear GEMM operands. It uses
Blackwell formats, rather than uniform integer rounding or weight-only QAT:

| Condition | Values and scales | Backward |
| --- | --- | --- |
| NVFP4 | E2M1; E4M3 scale per 16 values plus FP32 tensor scale; weights use 16×16 shared-scale tiles | Quantized DGRAD and WGRAD, stochastic gradient rounding; shared fixed randomized 16×16 Hadamard transform for both WGRAD operands |
| MXFP8 | E4M3; power-of-two E8M0 scale per 32 values, rounded up to avoid overflow | Quantized DGRAD and WGRAD; transpose operands quantized independently from the original high-precision tensor |
| FP32 control | Original unmodified Linear execution | Original autograd |

NVFP4 deterministic casts round to nearest, ties to even. GEMM operands are
first rounded to BF16; products accumulate in FP32 and outputs round to BF16.
Alignment is zero padding, without adding trainable capacity. All G/D Linear
layers, including input and output heads, participate. Bias addition uses a
BF16 bias operand; biases are stored/updated in FP32. Nonlinearities, Fourier
features, losses, prior sampling/updates/regularization and optimizer arithmetic
remain FP32. This is mixed-precision training, not an all-tensors-4-bit claim.
The prior is not frozen or silently quantized. Evaluation keeps the quantized
generator active and removes only training output noise.

The native-format rules follow NVIDIA's [NVFP4 reference](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
and [MXFP8 reference](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)
(Transformer Engine documentation 2.20.2, consulted 2026-11-06). NVIDIA's
[training paper](https://arxiv.org/abs/2509.25149) describes the corresponding
NVFP4 training mechanisms. These references motivate numerical rules, not a GAN
quality result. The local RTX A6000s cannot provide native FP4/FP8 parity data.

**Native parity remains unverified.** FP32 reference dot products cannot promise
the native kernel's accumulation order, rounding RNG sequence, fused operations,
performance or storage footprint. Quantizer derivatives use an explicit
straight-through reference so family gradient penalties can take second
derivatives. That higher-order path is a software research cohort, not a claim
that Transformer Engine supplies the same higher-order autograd. A native
comparison must verify quantization fixtures, FPROP/DGRAD/WGRAD, and the full
penalty path before transferring conclusions. Do not label these results native
Blackwell training or claim speed/memory savings.

## Execution, evidence and next decisions

The [software validation receipt](software/validation.json) records three genuine
public-trainer updates in each precision condition, matching initial master
parameters and target batches, finite gradients/state and isolated RNGs. The
receipt retains its original source commit and hashes. Actual
training: [FP32](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/precision-stretch/software/fp32.gif), [8-bit](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/precision-stretch/software/mxfp8.gif),
[4-bit](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/precision-stretch/software/nvfp4.gif). These are explicitly **INCOMPLETE** prefixes of the
1,000-update tests; they establish execution/media/checkpoint behavior only.
The [selected-family preflight](preflight.json) finds seven ready recipes and an
E22 policy-serving contract blocker. All full-budget precision cells remain
unmeasured. No quality leaderboard pass is inferred from these software checks.

The shared `transfer_vector` adapter calls public `GANTrainer` and
`particlegan.emulate_linear_training_`. No family-specific training loop or
recipe override is introduced. A family whose actual recipe is unsupported
retains an explicit preflight blocker. Tier 1's complete-current-tier execution
policy lets runnable stretch peers finish even if a required peer fails. A
failure on either precision task never blocks required qualification.

Every receipt records the complete precision contract, executed quantized
forward/backward counts, initial master-parameter hashes, target-batch digest,
named RNGs and a complete provenance checkpoint. NVFP4 rounding generators are
private to each Linear and checkpointed in its extra state; the shared fixed
Hadamard matrix is a model buffer. Evaluation consumes no rounding draws.
Actual-training GIFs reuse the exact scored samples; no extra observations or
training RNG changes are introduced by rendering.

```sh
python -m experiments.forge validate
python -m experiments.forge plan bispf-optimized-quintic-v1 \
  --view discriminator_stability --through-tier 1 --device cpu
python -m experiments.forge queue
python -m experiments.forge logs --follow
```

Registration alone supplies no pass credit. Keep per-attempt stdout, JSONL,
observations and checkpoints outside Git. Publish compact metrics, provenance
and actual-training GIFs; refresh the existing technique inventory instead of
creating another leaderboard for this goal. Interpret results against the
matched FP32 control: a failure shared by FP32 does not isolate low precision.
If FP32 passes and MXFP8 fails, inspect operand errors and gradient penalties
before attempting a 4-bit repair. If MXFP8 passes but NVFP4 fails, inspect scale
underflow, gradient rounding and the small output heads before changing a
global recipe. Native kernel validation remains a separate next step.
