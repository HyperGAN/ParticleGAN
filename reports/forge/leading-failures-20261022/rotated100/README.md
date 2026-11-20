# BISPF investigation: rotated100

Why does the selected BISPF recipe fail the 25-degree rotated 100-mode task, and do missing shape metrics hide an evaluator rotation issue?

**Original verdict: FAIL, unchanged.** The failure is already decisive before any shape comparison: clean holdout precision is 0.22301 against ≥0.97 and mass TV is 0.11132 against ≤0.06. Every final 20k check and the 100k holdout has undefined conditional shape fields. The frozen evaluator returns those fields as null when at least one of the 100 modes receives fewer than two in-radius samples. Null means insufficient conditional evidence; it never means zero error or a passing shape gate.

## Exact evidence and gates

This concerns the selected global recipe **`bispf-optimized-quintic-v1`**, revision
`8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`, attempt `97be11c288dc45bb9a98d1f1165ffbf0`,
at source `d9d72d53761c898bce20102e1995fc3d899c1447` / digest `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`.
The executed revision-11 policy holds this test; the inventory keeps the older
revision-9 full-suite comparison failure visible. Holding is not a PASS.
The [ordinary quality publication](../../bispf-optimized/quality.json) owns the
reported numerical summary. Older numbers in the problem list and BISPF README
belong to earlier source-bound cohorts and are not this selected run.

Target: The same 100 equal-weight isotropic Gaussians as grid100, with all centers rotated 25° and common standard deviation 0.03. G and D are three-hidden-layer width-128 LeakyReLU MLPs;
D adds two axis-aligned Fourier frequencies. The task fixes 20,000 initial learned
MoG centers, latent dimension 2, kernel σ=0.025, uniform masses, no standardization,
batch 2,048, protocol seed 0 and the public deterministic orthogonal initializer.
The recipe learns prior locations with SBNP regularization weight 0.01; it uses
non-saturating GAN loss, DualNorm, base LR 0.012, D multiplier 1.5 and prior multiplier
2.5. Input/output noise and EMA decay are zero. Architecture, prior width/capacity,
public sampling, 7,000-update budget and cadence are not changed by this investigation.

Clean live sampling must pass coverage (including all 100 modes, precision ≥0.97,
mass TV ≤0.1, per-mode mass/eigenvalue/radius limits), the four stricter accuracy
bounds below, all five final 20k checks at updates 6000/6250/6500/6750/7000, and a
separate 100k holdout. [The audit receipt](audit-result.json) retains the complete
frozen gate declaration. The actual run performs 7,000 D/G/prior updates each;
35 evaluation RNG audits report zero unintended deviations.

| Independent holdout metric | Observed | Gate | Result |
| --- | ---: | ---: | --- |
| Precision (within 0.09 of a center) | 0.22301 | ≥0.97 | FAIL |
| Mass TV | 0.11132 | ≤0.06 | FAIL |
| Center RMS / target σ | undefined | ≤0.2 | undefined → FAIL |
| Absolute conditional trace bias | undefined | ≤0.1 | undefined → FAIL |
| Radial KS | undefined | ≤0.04 | undefined → FAIL |
| Finite samples | 100,000/100,000 | all | PASS |

The terminal passing count is **0/5**. Terminal precision ranges from
0.2047 to 0.23085; conditional shape is
undefined in 5/5 checks. Undefined conditional metrics
are produced by `hq_counts < 2` for any component in the
[frozen accuracy source](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/benchmarks/toy100/accuracy.py).
Do not infer an exact missing-mode count from that aggregate condition.
The same run's true-target holdout passes: precision 0.98914,
mass TV 0.01226, center RMS 0.041737435σ,
trace bias 0.0002794405, radial KS 0.0015132736.

## What the source does and does not explain

At least one component remains poorly populated inside the quality radius even with 100k draws; the receipt does not identify which component or prove zero raw occupancy. The true-target oracle passes the rotated evaluator. The critic appends axis-aligned sin/cos features at π and 2π, so its representation is not structurally rotation equivariant. That is a competing explanation to investigate, not a causal conclusion from comparing two separately trained tasks. The unrotated task also fails.

The [frozen host adapter](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/experiments/forge/adapters.py)
and [optimizer dispatch](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/optim/dualnorm.py)
exclude the quintic optimization as an exercised mechanism here: both host
networks contain only Linear/LeakyReLU/Sequential modules. The quintic delta dispatches
only to eligible CUDA convolution groups (>32 channels); there are **zero** such
groups. Dense and sampled-prior update laws retain native behavior. The frozen
`Recipe.make_loss` selects ordinary non-saturating loss for this receipt; the
presence of a Sinkhorn-relativistic class elsewhere in the API does not make it
the executed generator objective. Sinkhorn remains part of the critic penalty.

The [frozen penalty](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py)
applies on all 7,000 updates. Its dense-multimode thermal branch
is eligible from update 1000 for batch 2,048, but the advertised late harmonic
capture lock requires real batch ≤512 and cannot run on this task. Eligibility
is a source fact; the actual residual/thermal scale, local critic field and relative
adversarial/SBNP gradients cannot be recovered from aggregate endpoint metrics.
No claim that cooling, SBNP, the critic representation or the smooth generator
caused the observed trajectory is established here.

The [earlier disposition audit](../../test-dispositions-20261015/README.md#grid100)
records completed failures, longer-budget BISPF plateaus and historically passing
noisy/fixed-particle cohorts. Those budgets and serving/prior contracts remain
separate; neither longer training nor added evaluation noise is an authorized repair
of this selected clean 7k cell. Its [source-bound original training GIF](../../bispf-optimized/quality-media/rotated100.gif)
and [media receipt](../../bispf-optimized/quality-media/rotated100.json) illustrate the
actual 34 saved observations, with zero new updates or sampling draws. Numerical
metrics above determine the result. The [current inventory](../../technique-inventory.md)
remains the single leaderboard; no second board or qualification change is introduced.

## Bounded next diagnostic and current blocker

Test scorer geometry before a rotation-specific trainer change. Recover the exact final holdout cloud, inverse-rotate its coordinates by 25° and evaluate against grid100 with the frozen evaluator; compare with the original cloud scored as rotated100. Repeat on the saved target control and on a deliberately displaced target control. Declare absolute differences ≤1e-4 for precision, mass TV and every defined accuracy metric, and identical defined/null patterns; a larger difference falsifies scorer rotational consistency and warrants source inspection. If consistency holds, it rejects this scorer explanation only; critic-feature causation remains untested. Cap this at one saved 100k cloud and its saved target control, 0 updates and 120 CPU seconds. Never substitute the transformed cohort for the original grade.

The byte-exact full readout is available at
`/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`,
SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`; it reproduces the published endpoint,
terminal summary, initialization and stream receipts. The referenced original
`rotated100/training-state.pt` and `holdout_samples.npz` are absent at their recorded
paths and were not found in the mounted `ParticleGAN-forge` tree. Their exact
expected hashes, original paths and final named-stream hashes are retained in the
audit receipt. The mounted full readout is not a tensor/checkpoint archive. Recover
and hash-verify the original artifact tree before any saved-state or sample-cloud
probe; absence blocks that analysis and does not authorize replay. No new training,
seed study, continuation, trainer delta, qualification or promotion was performed.

## Reproduce the zero-training audit

From this branch's repository root:

```sh
python reports/forge/leading-failures-20261022/rotated100/audit.py \
  --archive /mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json
```

[audit.py](audit.py) checks the selected row and exact archive digest, extracts source
from the pinned Git commit, executes only its scalar accuracy predicate on existing
measurements, checks dense-only dispatch/thermal eligibility, verifies the training
GIF and compares the compact [audit result](audit-result.json). It imports no trainer,
loads no tensor pickle and performs zero optimizer updates or sampling draws.
The audit's PASS means evidence consistency; the scientific test remains FAIL.
Raw execution output belongs in local `runs/forge/leading-failures-20261022/rotated100/audit.log`
for easy `tail -f`; it is not committed. Missing exact Git objects or archive bytes
fail explicitly. No new toy test or synthetic training footage is introduced.
