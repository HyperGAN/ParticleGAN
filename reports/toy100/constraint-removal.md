# Removing constraints from the shared 22-toy recipe

**A simpler shared recipe passes all 22 problems.** Starting from the previous
winner, jointly changing `reg_kappa: 1.176 → 1`, `reg_coeff: 6 → 1`, and
`prior_reg: .05 → 0` preserves every strict gate in a fresh production-runner
replay. The particle regularizer is removed. The discriminator penalty now has
a unit threshold and coefficient; those remain fixed modeling choices.

The [exact config](../../configs/toy100/constraints_simple_regularization.json),
[winning recipe and complete evidence](simpler22/README.md),
[22-case leaderboard](simpler22/compatibility.md), and
[new convergence GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy100/simpler22/toy100/toy100-progress.gif) are retained.
Config SHA-256: `4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`.
Training used frozen source `43d800d`. The older
[`shared_candidate.json`](../../configs/toy100/shared_candidate.json) and
[previous 22/22 result](shared22/README.md) remain historical controls.

**Constant learning rates remain unresolved.** None of the bounded Adam,
Optimistic Adam, AMSGrad, genuine ExtraAdam, or persistent-noise comparisons
passed the cheap strict screens. There is no continual-learning winner or
claim that eliminating these remaining constraints is impossible.

## Constraint audit

Every attempted case kept its fixed training seed, full host budget, live-weight
requirement, and original thresholds. One recipe must pass all 22; individual
passes cannot be combined across configurations. Native model/resource policies
are shared across all three 100-mode geometries; older hosts retain their frozen
architectures and resources.

| Constraint | Removal or simplification attempted | Result under the complete requirement |
| --- | --- | --- |
| Tuned κ=1.176, coefficient 6, prior penalty .05 | Joint unit cap/coefficient and zero prior penalty | **22/22 PASS**; selected default. Individual changes did not establish this interaction. |
| Remembering the preset and stricter accuracy switch | Both run commands default to the new winner; native runs require accuracy by default | Removed from the normal command. `--coverage-only` is an explicit diagnostic opt-out. |
| LR .00425 and particle/network ratio 2 | Rounded rates, equal player rates, ratio changes, and joint searches | No replacement passed all 22. LR .0025/ratio2 reached unequal-mass before failing the cheap screen. |
| Adam β=(0,.999) | Alternative moments, ordinary/optimistic/extra-gradient updates and AMSGrad comparisons | No replacement passed all 22. |
| Decaying LR | Constant actual optimizer-group rates; 96 Adam configurations, 79 optimistic/Adam/AMSGrad rows, 24 ExtraAdam plus 8 controls, 8 persistent-noise rows | Every valid row failed a strict cheap screen. No native or continuation promotion. |
| Separate .01 network and .05 particle floors | One common .05 floor, retaining the 1600 cap | Older19 and rotated/staggered passed; grid's last center RMS was .2829σ, above .20σ, despite 100 modes. **21/22 FAIL** across those separate recorded checks. |
| Same floor using .01 | Set the prior floor to .01 too | Failed cheap bars: four terminal passes where five are required. |
| Network horizon cap 1600 | One full-budget cosine and common .05 floor | Older19 and rotated passed; grid ended with 15 modes, staggered with 8. **20/22 FAIL** across the recorded checks. |
| Output σ=.029, close to target σ=.03 | Fixed/learned widths, no output noise, alternate RNG, data-only KDE bandwidths | No replacement passed all 22. The only complete ten-host reference in the 62-row global-RNG comparison retained .029. |
| Input smoothing .5 and 10% decay; output warmup 20% | Remove input noise, remove/change warmup, keep input noise persistent with constant LR | No replacement passed all 22. Persistent-noise rows failed sustained mode-hold. |
| Manually chosen particle box [−5,5]² | Normal prior; empirical min/max; mean ±√3 population SD from an unlabeled batch | No complete pass. Empirical min/max with the simpler core passed older19 and two native shapes but grid had 79 modes. Moment initialization failed grid with 98 modes and inaccurate shapes. |
| Identity affine generator | Random affine initialization and public MLP generators with normal priors | Failed native grid checks; no broader promotion. |
| Fourier3, width128/depth3, 20k particles, batch2048 | Individual and joint capacity/resource reductions | Some grid passes, no passing replacement across all three native geometries. Width64 passed 2/3; the joint smaller model passed only grid. |
| Explicit gradient penalty | R1/R2, alternative penalties, no penalty, and gradient-normalized discriminator scores | No replacement passed the cheap screens. Gradient normalization failed trajectory for both regularizer cores. |
| Fixed CPU dispatch, seeds, budgets and thresholds | Preserve the benchmark contract | Unchanged. Results establish fixed-profile regression behavior, not seed, platform, scale or arbitrary-data robustness. |

The winning recipe still uses ordinary Adam; G/D LR .00425, particles .0085;
cosine network horizon `min(budget,1600)` with floor .01 and prior full-budget
floor .05, both starting at 60%; input noise .5 decaying over the first 10%;
output noise .029 warming over the first 20%; and the original native model
and resources. [The recipe table](simpler22/README.md) lists all consequential
settings. Choosing the preset avoids entering these numbers manually; it does
not make their sensitivity disappear.

## Finite search ledger

The [machine-readable root ledger](constraint-removal.json) binds declarations,
config hashes, completed strict case metrics, skipped stages, and the six
native schedule promotions. Counts below are declared rows within each epoch,
including controls; they are not a count of unique configurations across all
research lanes.

| Root epoch | Valid rows | Strict outcome |
| --- | ---: | --- |
| One-change ablations | 24 | Baseline passed older19; 23 alternatives failed |
| Corrected Adam-moment ablations | 4 | All failed |
| Alternative penalty formulations | 64 | All failed |
| Rounded cap/coefficient/prior interaction | 72 | One older19 survivor, subsequently **22/22**; 71 failed |
| Simpler-core LR, moments, noise and schedule interactions | 30 | Two older19 survivors; both failed native promotion; 28 stopped cheaply |
| Common .01 floor | 1 | Failed bars after four preceding host passes |
| **Total** | **195** | **4 older19 passes including the incumbent control; 191 screen failures** |

There were additionally **68 setup-invalid attempts**, excluded from those
195 valid rows: four moment rows in the first epoch and an entire initial
64-row formulation epoch used mixed integer/float Adam beta values. The
corrected matrices were redeclared and rerun. Public Recipe construction now
normalizes numeric beta pairs, with a regression test. Harness errors are
recorded as INVALID, never numerical FAIL or PASS. Earlier constant-LR and
KDE harness-invalid epochs are described separately in their lane reports.

Independent lanes retained their own complete declarations and numerical
reports:

| Lane | Valid comparison scope | Result / report |
| --- | --- | --- |
| Constant-rate Adam | 96 rows, 128 attempted full-budget host episodes | 13 passed trajectory, 10 residual, 9 stripes, 0 mode-hold; [ledger](constant-lr-wave1.md) |
| Optimistic/Adam/AMSGrad constant-rate methods | 79 full-budget mode-hold rows | 0 sustained passes; [ledger](constant-game-screen.md) |
| Genuine ExtraAdam | 24 configurations +8 simultaneous-Adam controls | 0 sustained mode-hold passes; [ledger](extra-adam-screen.md) |
| Persistent instance noise at constant LR | 8 full-budget mode-hold rows | 0 sustained passes; [ledger](persistent-noise-screen.md) |
| Noise bandwidth | 56 isolated-stream +62 global-stream/Fourier rows; 4 data-only algorithm/core trials | No qualifying replacement; [ledger](bandwidth-constraints.md) |
| Gradient normalization | 2 regularizer cores | Both failed trajectory; [formulation and Jacobian audit](gradient-normalization-scratch.md) |
| Initialization/capacity | 19 first-epoch grid rows, 8 remaining-geometry promotions, plus empirical/moment interactions | No qualifying replacement; [ledger](constraints-geometry.md) |

All successful native promotions and the final winner were evaluated using
five terminal 20,000-draw checks and an independent 100,000-draw holdout.
One empirical-box main-run attempt was disqualified after its completed grid
failure; its incomplete common run is not presented as a completed 22-case
gate. The separate geometry ledger supplies the full empirical-box interaction
results. Raw failed-search archives are retained in ignored local `artifacts/`
directories of the lane worktrees; linked JSON/Markdown ledgers and configs
are checked in. The complete winner's raw evidence is checked in.

## Fail quickly on complete cheap tests

The reusable [constraint screen](../../benchmarks/toy100/constraint_screen.py)
predeclares the matrix, source and config hashes, then parallelizes candidates.
It runs trajectory, residual-student, stripes, mode-hold, bars, overlap, blobs,
intensity, unequal mass and unequal width in that order. Each attempted case
runs its frozen full budget; the first strict failure skips later hosts. A
survivor gets a fresh full19 replay before expensive native promotion. After
the first 96 constant-rate rows identified mode-hold as a strong disqualifier,
the game-optimizer and persistent-noise lanes started there. This changes
screening order, never a gate threshold or training budget.

```bash
python -u -m benchmarks.toy100.constraint_screen prepare \
  --base configs/toy100/constraints_simple_regularization.json \
  --variants reports/toy100/constraint-simple-interactions-v1.json \
  --output artifacts/toy-constraints/new-screen
python -u -m benchmarks.toy100.constraint_screen run \
  --output artifacts/toy-constraints/new-screen --workers 6
tail -F artifacts/toy-constraints/new-screen/logs/simple_one_floor.log
```

Preparation binds the current source; changing source afterwards invalidates
that declaration. These are screening results, not the common gate. Reproduce
the final result with `python -u -m benchmarks.toy_suite run --output NEW_DIR`;
inspect one native problem with `python -u -m benchmarks.toy100 run --problem
grid100 --output NEW_DIR`. Use the recorded CPU runtime profile in the
[run guide](../../docs/toy100.md).

## Research interpretation and remaining work

[Optimism](https://arxiv.org/abs/1711.00141) motivates correcting rotating game
updates. [Gidel et al.](https://arxiv.org/abs/1802.10551) motivates extrapolation;
the tested ExtraAdam follows Algorithm 4, including moment updates at both
half-steps and final parameters based on the saved original parameters. It
uses twice the gradient evaluations per outer update, which the report counts
explicitly. Neither result is a convergence guarantee for these nonlinear,
stochastic toy games.

[Mescheder et al.](https://arxiv.org/abs/1801.04406) motivates persistent instance
noise through a local convergence analysis. Here, noise .1 or .5 with constant
Adam group rates failed sustained mode-hold. [Gradient normalization](https://arxiv.org/html/2109.02235v2)
motivates normalizing scores by input-gradient magnitude and score magnitude.
Its piecewise-linear assumptions do not cover this Fourier/Softplus critic;
both direct comparisons failed. Batch-dependent critics additionally require
the per-example Jacobian diagonal; the cheap gradient of summed scores is not
that quantity. The report records this implementation limitation.

A constant nominal Adam learning rate still permits changing effective
per-coordinate updates through its moment estimates. A credible continuous
learning claim would require a passing shared constant-rate recipe, prolonged
continuation without resetting optimizer/noise burn-in, and a distribution
change test. None reached that promotion boundary, so those tests were skipped.
The deployable result of this search is the simpler scheduled recipe. Further
work should target sustained mode-hold first and preserve the independent
accuracy checks before claiming that the remaining constraints are removed.
