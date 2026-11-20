# BISPF investigation: staggered100

Does the selected BISPF staggered-grid failure reflect a global geometric distortion or mode-local mass and shape errors?

**Original verdict: FAIL, unchanged.** The failure is numerical throughout the final window: all five final checks fail, with precision only 0.3285–0.348. The clean independent holdout fails precision, mass, centering, covariance-trace bias and radial shape while remaining finite. Conditional shape is measurable on all five terminal checks, so absent conditional estimates cannot explain this failure.

## Exact evidence and gates

This concerns the selected global recipe **`bispf-optimized-quintic-v1`**, revision
`8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`, attempt `9a8e496fffff47a68976f46b69b1be2d`,
at source `d9d72d53761c898bce20102e1995fc3d899c1447` / digest `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`.
The executed revision-11 policy holds this test; the inventory keeps the older
revision-9 full-suite comparison failure visible. Holding is not a PASS.
The [ordinary quality publication](../../bispf-optimized/quality.json) owns the
reported numerical summary. Older numbers in the problem list and BISPF README
belong to earlier source-bound cohorts and are not this selected run.

Target: 100 equal-weight isotropic Gaussians of standard deviation 0.03. Adjacent grid rows shift by −0.25/+0.25 in the second coordinate, and first-coordinate spacing contracts by a factor of 0.85. G and D are three-hidden-layer width-128 LeakyReLU MLPs;
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
| Precision (within 0.09 of a center) | 0.33617 | ≥0.97 | FAIL |
| Mass TV | 0.12142 | ≤0.06 | FAIL |
| Center RMS / target σ | 1.1428836 | ≤0.2 | FAIL |
| Absolute conditional trace bias | 0.54168964 | ≤0.1 | FAIL |
| Radial KS | 0.369111 | ≤0.04 | FAIL |
| Finite samples | 100,000/100,000 | all | PASS |

The terminal passing count is **0/5**. Terminal precision ranges from
0.3285 to 0.348; conditional shape is
undefined in 0/5 checks. Undefined conditional metrics
are produced by `hq_counts < 2` for any component in the
[frozen accuracy source](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/benchmarks/toy100/accuracy.py).
Do not infer an exact missing-mode count from that aggregate condition.
The same run's true-target holdout passes: precision 0.98914,
mass TV 0.01226, center RMS 0.041737404σ,
trace bias 0.00027943895, radial KS 0.0015147474.

## What the source does and does not explain

The signed conditional covariance-trace bias is +0.54169, meaning the mean in-radius covariance trace is about 1.542 times its truncated-target reference. This is not an untruncated Gaussian width estimate, and the aggregate does not show that every individual mode is broad. Center RMS is 1.14288σ and radial KS is 0.36911. Fixing populations alone cannot satisfy those independent accuracy gates. Per-mode covariances, raw tail structure and a learned global warp remain unidentified without saved samples.

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
of this selected clean 7k cell. Its [source-bound original training GIF](../../bispf-optimized/quality-media/staggered100.gif)
and [media receipt](../../bispf-optimized/quality-media/staggered100.json) illustrate the
actual 34 saved observations, with zero new updates or sampling draws. Numerical
metrics above determine the result. The [current inventory](../../technique-inventory.md)
remains the single leaderboard; no second board or qualification change is introduced.

## Bounded next diagnostic and current blocker

Test the narrow claim that a single global affine warp explains the geometry error. Recover the exact 100k saved holdout cloud; deterministically split it in half, fit an evaluator-only affine map from first-half points to their frozen nearest target centers, then evaluate the transformed second half against the unchanged bounds. Keep identity, saved-target and deliberately sheared/translated-target controls. Reject the global-warp explanation if held-out precision remains <0.97 or any accuracy bound fails, and retain the raw original score beside the counterfactual. This uses evaluator-only labels and is never a proposed labeled trainer. Cap it at one saved 100k cloud and its saved target control, 0 updates and 120 CPU seconds; it earns no qualification.

The byte-exact full readout is available at
`/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`,
SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`; it reproduces the published endpoint,
terminal summary, initialization and stream receipts. The referenced original
`staggered100/training-state.pt` and `holdout_samples.npz` are absent at their recorded
paths and were not found in the mounted `ParticleGAN-forge` tree. Their exact
expected hashes, original paths and final named-stream hashes are retained in the
audit receipt. The mounted full readout is not a tensor/checkpoint archive. Recover
and hash-verify the original artifact tree before any saved-state or sample-cloud
probe; absence blocks that analysis and does not authorize replay. No new training,
seed study, continuation, trainer delta, qualification or promotion was performed.

## Reproduce the zero-training audit

From this branch's repository root:

```sh
python reports/forge/leading-failures-20261022/staggered100/audit.py \
  --archive /mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json
```

[audit.py](audit.py) checks the selected row and exact archive digest, extracts source
from the pinned Git commit, executes only its scalar accuracy predicate on existing
measurements, checks dense-only dispatch/thermal eligibility, verifies the training
GIF and compares the compact [audit result](audit-result.json). It imports no trainer,
loads no tensor pickle and performs zero optimizer updates or sampling draws.
The audit's PASS means evidence consistency; the scientific test remains FAIL.
Raw execution output belongs in local `runs/forge/leading-failures-20261022/staggered100/audit.log`
for easy `tail -f`; it is not committed. Missing exact Git objects or archive bytes
fail explicitly. No new toy test or synthetic training footage is introduced.
