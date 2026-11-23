# Historical findings log (2026-08-21 to 2026-10-06)

<!-- superseded-banner: generated -->
> **Superseded — historical record.** Current status: [reports/forge/technique-inventory.md](reports/forge/technique-inventory.md).
> This log is no longer updated; its last entry is dated 2026-10-06. Results are preserved as recorded.

Contents, in order:

1. [Gradient-penalty centering in RpGAN](#gradient-penalty-centering-in-rpgan-round-3-provenance-bake-off-corrected-audit)
   (rounds 1–3, August 2026; the only study here with five seeds and bootstrap intervals).
2. Fixed-sigma MoG prior: [Stage 0 gate](#draft-fixed-sigma-mog-prior--stage-0-gate-2026-09-16) and
   follow-ups (2026-09-16).
3. BCAP-pure: [optimizer-only Tier 1 screen](#bcap-pure--optimizer-only-tier-1-screen-2026-10-05) (2026-10-05) and
   [dualnorm pacing follow-up](#bcap-pure--dualnorm-pacing-follow-up-2026-10-06) (2026-10-06).

## Gradient-penalty centering in RpGAN (round 3: provenance, bake-off, corrected audit)

**Benchmark:** `examples/100gaussians.py` recipe (RpGAN, ParticlePrior, Fourier-2 D, EMA eval,
Adam β1=0, delayed cosine anneal), 7 000 steps, 5 seeds per cell. **420 runs, 0 failures**;
training is deterministic (48/48 audit reruns bit-identical). Metrics: exact W1/W2 via POT
(n=4096; sampling floor W1 ≈ 0.14), `hq` (mass within 3σ_true = 0.09 of a center), per-mode
width measured two ways after round 3 — raw mean-centered σ ratio (tail-inflated; kept for
context) and **median-centered core σ ratio** (the honest number; see
`results/metric_recon.md`), plus tail mass beyond 10σ_true. Stability: Arnoldi spectral radius
of the alternating-GD update map (comparative only). Tables: `results/TABLE.md`,
`results/bootstrap.md`, `results/LEADERBOARD.md`, `results/provenance.md`.

### The organizing result: compatibility vs. curvature

- **Curvature (whether you converge) is center-agnostic.** Unregularized: radius 1.055 ± 0.052,
  |Im λ| ≈ 0.037, 47/100 modes. Any sample-point penalty, any centering: radius ≤ 1.002,
  |Im λ| ≤ 0.0015, zero collapses in all penalized logistic runs. (Mescheder Lemma 3.3 made
  empirical; new: zero-centering damps rotation ~4× harder at matched coeff — 2.8e-4 vs 1.25e-3.)
- **Compatibility (where you can converge) is loss-family-relative.** Under logistic RpGAN the
  optimal critic flattens at match; center-at-1 forbids that and pays a residual W1 floor
  (~2.5×) — the Dirac-GAN incompatibility in its soft, finite-training form. Under the
  **Wasserstein objective the same penalties at the same coefficients are a statistical wash on
  W1** (all CI95 include 0) and the capped arms take the fidelity cells. Centering must match
  the loss family's optimal critic: margin device for f-div, transport device for IPM.

### Round 3 verdicts

**Provenance of the cap's stability — trajectory regularizer, not endpoint regularizer.**
The blocking question (hinge slack at a flat critic ⇒ where does B's radius ≤ 1.002 come
from?) resolves to *inherited from the state, earned by the trajectory*: at B's endpoints
≤ 0.4% of samples clear the cap and masking the penalty out moves the dominant modulus by
−0.0004 (adding a zero-centered probe penalty there: also nothing), while the identical masked
measurement on `f_none`'s own checkpoints reads 1.068 — the state differs, not the
measurement. The hinge *was* engaged for 56–75% of training (q90 ‖∇D‖ above the cap until
step ~5 200–5 900) and went slack only when the LR anneal collapsed the critic to near-flat.
**Deployment consequence:** the cap supplies no standing damping at convergence. Safe where
the equilibrium is self-stable; in a game that keeps injecting rotation (e.g. recurrent
world-model training), standing curvature (R1/R2) is the defensible choice.

**Corrected variance audit — retraction and a third strike against strict eikonal.** Round 2's
"no variance collapse anywhere (σ ratios 2.1–5.4)" was an artifact of the mean-centered second
moment: the far tail inflates the moment *and* drags the per-mode mean. Median-centered core
widths: champion `b_cap` **0.87–0.92** (≈ true width; 0.8% tail mass beyond 10σ), `a_r1r2@0.1`
0.91–0.96, `a_r1r2@1.0` **2.16** (the blur is real; hq 0.60 is honest), and
`c_eikonal@0.1` **core 0.55 ± 0.23 — variance-compressed** — hidden under a raw ratio of 4.80
by 3.15% of mass stranded whole grid cells away (99th-pct nearest-center distance 2.01 vs
0.47 for b_cap): the expected signature of a critic pinned to unit slope between modes. The
leaderboard's compression guard now reads the core ratio; core width predicts measured hq to
within 0.05, so the two metrics no longer conflict.

**Wasserstein + cap bake-off — won on the stated metrics, then demoted by the corrected
audit.** On W1/hq/tails the caps look dominant: every cap cell matches A on W1 (0.159–0.172,
CIs include 0), hq 0.94–0.975 vs 0.87–0.92, bar-pass 5/5 in all six cap cells vs 0/5 for
A@1.0, and a broad coefficient plateau (0.3/1/3). But the core-ratio audit shows **every
bar-passing WGAN cell — cap AND R1/R2@0.1 — is variance-compressed** (cores 0.38–0.73;
raw ratios of 1.5–2.2 hid it), while the only core-honest WGAN cells, A@1.0 (core 1.02–1.14),
never pass the bar. Under IPM on this benchmark, sharpness is purchased substantially with
core compression. What survives: `wgan_a_r1r2@1.0` is the best *calibration* cell in the
study (W1 0.154, core 1.02, recall 1.000, tail 2.1% — soft on hq, which is the conservative
direction), and unpenalized WGAN detonates (collapse 0.8, radius ~1e9), so a penalty — any
penalty — is load-bearing there.

**Curriculum (B→A hard switch) — prediction failed, coherently.** The switch keeps B's
bar-crossing speed (3 860) but the endpoint re-equilibrates to the destination arm's frontier
position within the remaining 2 800 steps (→A@0.1: W1 0.359/hq 0.929 ≈ B's own endpoint;
→A@1.0: hq collapses to 0.65). No two-knob escape: consistent with provenance — the
trajectory bequeaths the *basin*; the W1/hq operating point is set by the penalty active at
the end. Together with round 2's κ-anneal null: neither soft nor hard schedules cross the
frontier; the frontier is a property of the *final* penalty regime.

### Standing verdicts (rounds 1–2, updated wording)

- **H1:** falsified for f-div on transport metrics (CI-backed); the sharpness half survives
  the *corrected* audit for the cap family only (core ≈ 0.9), not for strict eikonal
  (core-compressed). The W1-vs-concentration dissociation stands.
- **H2:** confirmed (0.045 vs 1.04 med ‖∇D‖ mid-training; late sag as the data term wins).
- **H3:** inverted into the decomposition above — zero-centering is not needed for damping,
  but is doubly right under f-div: compatible center *and* strongest rotation damper per unit
  coefficient, *and* (round 3) the only family with standing curvature at the endpoint.
  R3GAN chose correctly; these results spell out why.
- **H4:** partial — caps fastest to the sharp bar at base LR (CI95 [−900, −380] steps);
  2× LR improves endpoints, not bar speed. OAdam: clean null (rotation is already gone;
  residual instability is symmetric-part, which optimism cannot fix). Lazy ≡ every-step at
  matched time-average.

### Leaderboard & promotion (see `results/LEADERBOARD.md`)

Logistic promotion track CHAMPION: **`b_cap` L2, coeff 1.0, 2×LR** (W1 0.290, hq 0.986,
core σ 0.866 — clears the corrected guard; runner-up L1 variants 0.82–0.85 also honest;
the L∞ variant, d_asym, e_interp, and all measurable c_eikonal cells are now flagged
variance-compressed). The logistic cap cells and the curriculum→A@0.1 cell (core 0.911,
W1 0.283, hq 0.938) are the only sharp-and-core-honest configurations in the study.
**Bake-off outcome:** the WGAN+cap recommendation is withdrawn — its dominance was partly
core compression. If the objective may change and the deployment metric is *calibration*,
`wasserstein + a_r1r2@1.0` is the best honest cell (W1 0.154, core 1.02); if the metric is
*sharpness*, stay logistic and promote the champion. Provenance caveat travels with the cap
either way.

### Caveats

Symmetric R1+R2 only (R1-alone diverges per R3GAN's ablations); γ is a frontier dial, not an
escape. ~40 CI cells → multiplicity; headline claims rest on the largest, replicated,
cross-validated effects. Spectral radii comparative (GD map, no optimizer state; FD smoothing
bias identical across arms). One 2-D toy, small MLPs. Deployment validation was planned on a
target domain in a separate world-model project outside this repository (its
`docs/eikonal-branch-notes.md` is not part of ParticleGAN), with this suite retained as a CI
canary rather than a benchmark.

### Reproduce

Check out archive commit
[`40e637e82`](https://github.com/HyperGAN/ParticleGAN/tree/40e637e82cdedb51624e6a69e3dd6926c62a1924)
(round 3, 2026-08-21; an ancestor of `develop`), where every file below exists.
[`lib/grad_regularizers.py`](https://github.com/HyperGAN/ParticleGAN/blob/40e637e82cdedb51624e6a69e3dd6926c62a1924/lib/grad_regularizers.py),
[`experiments/train_arm.py`](https://github.com/HyperGAN/ParticleGAN/blob/40e637e82cdedb51624e6a69e3dd6926c62a1924/experiments/train_arm.py),
[`experiments/gen_configs.py`](https://github.com/HyperGAN/ParticleGAN/blob/40e637e82cdedb51624e6a69e3dd6926c62a1924/experiments/gen_configs.py)
and [`experiments/provenance.py`](https://github.com/HyperGAN/ParticleGAN/blob/40e637e82cdedb51624e6a69e3dd6926c62a1924/experiments/provenance.py)
were removed from the tree by #165 (`e0cdb036b`, 2026-09-25); the other files still exist but
have changed since. The current package keeps only the fixed `a_r1r2` and `b_cap` arms, as
`GradientPenalty(arm=...)` in [`particlegan/grad_regularizers.py`](particlegan/grad_regularizers.py)
(v0.7 L2 kernels); the other arms are not in the package. Files used by the study:

`lib/grad_regularizers.py` (arms, norms, anneal; FD-gradchecked), `lib/oadam.py` (vendored —
do not substitute third-party copies), `lib/game_jacobian.py`, `lib/toy_metrics.py` (incl.
core-ratio estimator), `experiments/train_arm.py` (per-run `ckpt.pt` + `final_samples.npy`),
`experiments/gen_configs.py --stage {main,lr_sens,lazy,audit,anneal,wgan,dualnorm,oadam,
bakeoff,curriculum}`, `experiments/run_grid.py`, `experiments/analyze.py`,
`experiments/leaderboard.py`, `experiments/provenance.py`, `experiments/metric_recon.py`.
Raw runs in `results/runs*/` (gitignored, on disk).

## Draft: fixed-sigma MoG prior — Stage 0 gate (2026-09-16)

Stage 0 only, three seeds (1–3), 7k steps, 200k final EMA samples per run and
matched real references. Full [report and deviations](results/mog/STAGE0.md),
[results CSV](results/mog/results.csv), and [traces plot](results/mog/stage0_traces.png).
No small-N sweep has run; predictions 1–7 are **inconclusive** pending their
specified comparisons.

C0 regression passes **3/3**: all 70 original log entries per seed and every final
EMA generator tensor plus the particle table are bit-identical to pre-change
`af1843a`. The measured code is `5db6d3a`; the new prior uses sigma=0 and no
standardization for this control.

| Reference | Pass rate | HQ / real | Core width / real | HQ-only KL |
|---|---:|---:|---:|---:|
| C0: 20k learned atoms | 0/3 | 0.99978 ± 0.00107 | 0.84343 ± 0.04312 | 0.03478 ± 0.00195 |
| C1: fresh Gaussian | 0/3 | 0.06069 ± 0.00310 | 10.18916 ± 0.04109 | 0.39484 ± 0.02837 |

C0 covers all 100 modes and crosses the old coverage bar at step 5500 in every
seed, but fails the stricter width and balance criteria. Evaluating each of its
20k atoms exactly once gives KL **0.03684 / 0.03361 / 0.03303**: the imbalance
is in the learned distribution, not eval sampling noise. Fair-share-normalized
mode shares range from **0.465–0.526** at the minimum to **1.650–1.798** at the maximum.
Historical `hist_kl` includes bridge samples; applying that estimator directly
to the reproduced pre-change 20k-sample outputs gives **0.03498 / 0.03221 / 0.03240**.
These are current-recipe reproductions, not recovered historical study scores.

The simulated allocation-null KL means are **0.56891 / 0.28368 / 0.13208** for
N=100/200/400 (1,000 draws each); mean empty-mode counts are 36.557/13.526/1.829.
The historical core-width estimator is reused exactly (median radius around
per-mode coordinate medians, no HQ truncation), with same-size real normalization.

Recommendation: **go to the Stage 1 optimizer pilot**, retaining the stricter
pass criterion; do not equate matching C0's HQ with passing. Its balance and width
leave room for a useful result. This is a recommendation only: later stages are
paused for the owner's decision. The six Stage 0 runs took 3.4 minutes on two
A6000s; C0 averaged 70.2 seconds/run including the new metric suite.

## Fixed-sigma MoG prior — Stage 1 optimizer pilot (2026-09-16)

The owner authorized Stage 1 and revised the criterion to **original or better**.
Before pilot results, we froze the observed three-seed C0 envelope: modes=100,
HQ/real ≥ **0.99883228**, width/real **0.79091586–1.20908414**, and HQ-only
KL ≤ **0.03750414**. All C0 references pass this rule. The old design criterion
remains recorded separately; Stage 0 history is unchanged.

**No match at nominal r=1/8 within 7k steps:** 36 unique runs, **0/36 passes**.
All 36 fail HQ, width, and balance individually; this is not a marginal threshold
failure. We completed the 30-run LR sweep and six new momentum runs, reusing the
six identical beta=0 comparisons. Tests: 35 passed plus 13 subtests. All 36 runs
are certified complete; all 2,520 metric-trace rows and frozen configs validated.
Training sources: `33b3144`.

Selected by pass rate, then HQ, then width distance from real:

| N | Particle LR multiplier | Particle β1 | Passes | HQ/real | Width/real | KL |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 10× (initial LR 0.06) | 0 | 0/3 | 0.29137 | 4.400 | 0.37679 |
| 400 | 10× (initial LR 0.06) | 0.5 | 0/3 | 0.38016 | 3.373 | 0.15626 |

Higher LR improves HQ but does not approach C0. The momentum preference is small:
N=100 beta=0.5 has HQ/real 0.28825; N=400 beta=0 has 0.37668. The selected N=400
momentum setting has worse balance than beta=0 (0.15626 vs 0.13829); it wins only
through the preregistered HQ tie-break after both pass rates are zero. Neither is
an established optimum; both LR winners are at the tested upper boundary.

**The Gaussian centers fit much better than their noisy neighborhoods.** In the
selected N=400 cell, **99.4%** of component centers map within a data mode's 3σ
radius, but **37.6%** of noisy samples do. HQ-conditioned purity is effectively
1.0, with no empty majority allocations. N=100 has 91.0% HQ component centers,
28.8% HQ noisy samples, purity 0.9803, and 13.3 empty majority allocations. These
center-only measurements are explicit diagnostics; all primary evaluation keeps
noise on. Excessive output spread, plus N=100 allocation failures, explains the
poor quality better than an evaluation-only pass-rule issue.

**Clumping complicates the proposed noise/separation ratio.** Selected N=400
r_eff averages **4.63**, despite nominal r=0.125; 91.2% of evaluable components'
nearest neighbors have the same majority output mode. This is not the nominal
r=2 Gaussian-collapse control. All three selected runs at each N also exceed the
2× raw-scale drift threshold. The report retains those flags and separately
records distances to neighbors with different majority modes; the prescribed
r_eff and ranking are unchanged.

Recommendation: before the full Stage 2 grid, test N=400 with the selected
optimizer at **r=0, 1/32, and 1/16**. The atoms control isolates the small-table
limitation; smaller noise tests whether width can recover. r=1/32 would be an
explicit addition to the original design. **No follow-up runs launched.** The
seven original predictions remain inconclusive for their full stated comparisons;
the pilot's supporting and contrary observations are enumerated in the report.

[Full report](results/mog/STAGE1.md) · [per-run results](results/mog/stage1_results.csv)
· [leaderboard](results/mog/stage1_leaderboard.csv)
· [optimizer plots](results/mog/stage1_optimizer.png)
· [runbook](results/mog/STAGE1_RUNBOOK.md).
The timed first run took 68 seconds; the remaining LR batch took 10.2 minutes
and the momentum round 2.5 minutes with two workers per A6000.

## Fixed-sigma MoG — smaller noise and longer training (2026-09-16)

The authorized follow-up completed 15 runs across five settings, retaining the
selected N=400 optimizer and all three original seeds. Lower fixed noise and a
longer training budget improve MoG substantially, but do not yet match the frozen
C0 envelope. The adaptive r=1/32 and r=1/40 points extend the original grid.

| Setting | Steps | HQ/real | Width/real | KL | Pass |
|---|---:|---:|---:|---:|---:|
| C0, 20k atoms | 7k | 0.99978 | 0.84343 | 0.03478 | 3/3 |
| 400 atoms, selected optimizer | 7k | 0.99171 | 0.393 | 0.02754 | 0/3 |
| 400 MoG, r=1/32 | 7k | 0.96220 | 1.154 | 0.04966 | 0/3 |
| 400 MoG, r=1/32 | 14k | 0.97851 | 1.066 | 0.03591 | 0/3 |
| 400 MoG, r=1/40 | 14k | 0.99641 | 0.926 | 0.03588 | 0/3 |

Numbers are three-seed means. At r=1/40, all three runs meet coverage, width and
balance; only HQ fails. This uses **50× fewer components and 2× the training
steps** of C0, with width closer to the real data. It is not an equal-budget win.
The 14k runs restart from the same initialization with their cosine schedule
scaled to the longer budget. Their first 42 original log entries match the
corresponding 7k r=1/32 runs exactly for all three seeds.

**Overlap needs a destination-aware interpretation.** At r=1/8, the independent
latent posterior audit estimates component-identity ambiguity at 52.3%, but
ambiguity between components grouped by their learned output-mode labels at only
0.020%, versus 62.4% generated non-HQ mass. Same-mode clouds may overlap without
hurting mode identity. Lower-noise settings also have negligible sampled
destination ambiguity despite non-HQ tails. This suggests neighborhood shaping
is the larger remaining problem in these trained models; it does not establish
a causal mechanism or prove Gaussian supports are disjoint. The quantity
`bridge=1-hq` includes excessively wide within-mode tails.

Next: investigate component count and training budget while retaining matched
zero-noise controls. [Full follow-up report](results/mog/NOISE_CHECK.md),
[results](results/mog/noise_check_results.csv), and
[overlap diagnostics](results/mog/noise_check_overlap.csv).

## Fixed-sigma MoG — component count and 28k budget (2026-09-16)

The owner requested larger tables and more training, with no seed-only experiments.
Completed **21 new configurations at seed 1**: a 13-setting count/noise/optimizer
screen, four additional shipped-optimizer controls, and four 28k runs. Reused
existing seed-1 references; no new seed repetitions. These are exploratory
configuration results, not pass-rate estimates. All use the unchanged shared
trainer, 200k final EMA samples with matched reals, and the frozen C0 envelope.

**MoG now passes at 400 components with sufficient training.** It also passes at
20k components with the shipped optimizer. Selected comparisons:

| Prior | N | r | Steps | HQ/real | Width/real | KL | Envelope |
|---|---:|---:|---:|---:|---:|---:|---|
| Original C0, seed 1 | 20,000 | 0 | 7k | 1.001274 | 0.7909 | 0.03750 | pass |
| MoG, unstandardized | 20,000 | 1/40 | 7k | 1.000066 | 0.8258 | 0.03570 | pass |
| C0, longer | 20,000 | 0 | 28k | 0.999110 | 0.9438 | 0.02655 | pass |
| MoG, unstandardized | 20,000 | 1/16 | 28k | 1.000046 | 0.9543 | 0.02657 | pass |
| MoG, unstandardized | 20,000 | 1/40 | 28k | 0.999778 | 0.9567 | 0.02742 | pass |
| MoG, standardized | 400 | 1/40 | 28k | 0.999535 | 0.9264 | 0.02888 | pass |

Every row covers 100 modes. Ratios above 1 can occur because a model is narrower
than the real distribution and/or because of evaluation sampling variation.
Historical C0 means over three seeds are HQ/real 0.99978, width/real 0.84343,
KL 0.03478. Only the new 20k r=1/16, 28k MoG clears all three C0-mean thresholds.
It still does not dominate the matching seed-1 original C0 on HQ, and costs four
times the updates. No positive-noise run dominates original seed-1 C0 on all
coverage/HQ/width-error/KL dimensions.

The strongest compact result has **50× fewer components and 4× the training
steps**, with better width and balance than original C0. Its raw-table scale grows
4.11×, a gauge/optimizer drift flag; standardized reads keep the read-space scale
controlled. There is not yet a matched 400-atom 28k control, so this establishes a
small MoG achieving the envelope, not that noise is necessary at that budget.

At matched 28k budget, 20k r=1/16 MoG slightly improves HQ and width over atoms,
while KL is effectively tied (0.026566 versus 0.026554). Thus much of the gain over
original 7k C0 comes from longer training, not necessarily MoG. The unstandardized
20k priors do not exhibit the hypothesized inflation escape here: raw scale grows
about 11–13%, and effective sigma/spacing increases. This low-noise finding does
not substitute for the original high-noise C3 control.

Increasing count is not a monotonic win. The fast optimizer selected at N=400
(particle LR 0.06, beta1=0.5) transfers poorly to 6,400 and 20k. With shipped
particle LR 0.006 and beta1=0, standardized 1,600/6,400 MoGs recover width to
0.918/0.893 at 7k, but KL remains 0.0542/0.0466. Standardized 20k r=1/40 also
misses balance (0.0520); its unstandardized counterpart passes. Cross-N changes
also alter calibrated absolute sigma, and N>1,024 uses sampled-row VICReg.

The 28k schedules are fresh runs with annealing starting at 16,800, not resumed
checkpoints. All original log entries before each shorter parent's annealing
onset match exactly (42 per 20k comparison, 84 for N=400). All **2,310** new trace
rows, configs, certificates and summary digests validated. The baseline CSV and
training implementation remain unchanged. Timings: 77s initial run, 4.1-minute
screen remainder, 1.3-minute optimizer follow-up, 5.2-minute longer batch.

Recommendation: retain both priors. The compact MoG is now a viable measured
tradeoff. Next compare a matched 400-atom 28k run, then optimize the compact MoG's
schedule to reduce its training cost. For a large-table quality baseline, keep
20k r=1/16 MoG and 20k atoms at matched budget. No new seed repetitions are needed
to answer those configuration questions.

[Full report and all failures](results/mog/COMPONENT_SCALE.md) ·
[per-run metrics](results/mog/component_scale_results.csv) ·
[leaderboard](results/mog/component_scale_leaderboard.csv) ·
[component-count plot](results/mog/component_scale_count.png) ·
[training curves](results/mog/component_scale_training.png).

## BCAP-pure — optimizer-only Tier 1 screen (2026-10-05)

The owner narrowed the proposed dual-norm study to the current **BCAP-pure
Tier 1** first. This screen uses protocol seed 0, the public deterministic
initializer and one global trainer configuration across all tasks. It changes
only optimizer rules, their step sizes and declared network momentum. Losses,
BCAP coefficient/cap, task-owned auxiliary terms, architecture, data/prior laws,
sampling, schedule shape, update budgets and grading remain unchanged. Existing
failing tasks are retained; their investigation belongs to another PR.

The control is the current public BCAP recipe: Adam beta=(0,.999), base rate
.00425, D multiplier 1, prior multiplier 2, constant schedule and clean/live
evaluation. The pasted .0006/EMA/delayed-cosine native recipe is a different
cohort. This run supplies no EMA/native100/core-sigma or scale-transfer result.
The two-pole host's declared zero-coordinate/stored-weight fixture remains an
explicit separate initialization cohort; it is not a substitute for the learned
MoG hosts.

The finite grid contains **41 configurations** across Adam and the seven new
options: `sgda`, `nsgda_global`, `nsgda_layer`, `ada_nsgda`, `dualnorm`,
`dualnorm_D_only`, and `particle_rownorm_only`. SGDA spans four decades; normalized
rates use their own units. Full dualnorm tests momentum 0/.5/.9. Hybrid arms
retain baseline Adam rates for unchanged players. The first stage fixes D/G=1.5
and prior step=.03 for all-player normalized arms; it does not execute the full
ratio/prior Cartesian grid. The tensor magnitude graft uses D/G=1 and prior/G=2
to match Adam's nominal per-player rates. Every configuration runs the six
required tasks plus the separately graded clock diagnostic, with a 103,320-second
campaign ceiling. No seed repetitions, automatic edge extensions or later tiers
follow from this declaration.

Use the [current family leaderboard](reports/forge/technique-inventory.md) for
the single ranked goal table. [All final configurations and metrics](reports/forge/dualnorm-tier1/analysis.json),
[search results](reports/forge/dualnorm-tier1/results.json) and the
[study/reproduction guide](reports/forge/dualnorm-tier1/README.md) preserve task
statuses and exact whole-recipe selections. The original search selection
maximizes required Tier 1 PASS count, then breaks ties by configuration hash.
Those nine search outcomes remain unchanged. The owner subsequently requested
a new optimizer as the BCAP starting point even on a tie. The current dualnorm
measurement therefore uses the .01 zero-momentum recipe: among the 3/6 ties,
it retains ring coverage, improves ring HQ and passes words. This retrospective
preference is explicit in the [starter receipt](reports/forge/dualnorm-tier1/starter-selection.json);
it is not an independent confirmation or a newly qualified default.

**No new optimizer beats the 3/6 Adam control on required Tier 1 PASS count.**
Global nSGDA, tensor nSGDA, full dualnorm with zero momentum, and prior-only row
normalization each reach 3/6. Plain SGDA, the Adam-magnitude graft and D-only
dualnorm each top out at 2/6. No configuration passes Gaussian or ring
acquisition. The word task can pass at lower Adam/normalized rates, but those
recipes lose other required passes. For example, global nSGDA at .01 passes
unused-token hold, AE hold and words while failing two-pole; at .03/.1 it passes
the three hold tasks and fails words. These are different whole recipes, not a
combined four-pass candidate. The current .00425 Adam control reproduces its
three hold passes and three acquisition failures.

The 287 final cells comprise 246 required measurements and 41 separate clock
diagnostics. Required outcomes are 70 PASS, 171 FAIL and five numerical
INCOMPLETE; all 41 clock diagnostics pass. There are 288 paid attempts because
one interrupted attempt has a linked repair. All 282 eligible saved-training
GIFs pass certificate/input verification; the five numerical errors and original
interrupted attempt have no eligible complete observation stream. Recorded
attempt wall time totals 17,119.27 seconds, including the interrupted original;
this is cost evidence, not a speed comparison.

**The aggregate tie hides substantial quality tradeoffs.** Full dualnorm at
mu=0, eta=.01 reaches 16/16 ring modes with HQ=.93018 versus the control's
.82275, and passes word acquisition. Ring still fails its full covariance/
component and sustained gate. Two-pole has 14 passing observations and a terminal
passing suffix of four, below the required five, so this recipe remains 3/6.
Its ring core minimum eigenvalue ratio is .17293 versus Adam's .24163, so the
higher HQ also comes with a weaker core-width metric.
At mu=.5, eta=.01, ring HQ rises further to .94336 at 16 modes, but the whole
recipe reaches only 2/6. D-only eta=.03 reaches ring HQ=.87231 at 16 modes yet
also remains 2/6. These are endpoint improvements, not new gate passes.

Conversely, hash-selected tensor nSGDA at eta=.1 drops ring coverage to five
modes/HQ=.21729, and prior-only eta=.01 drops it to four modes/HQ=.16260, despite
their 3/6 scores. The original hash-selected full dualnorm eta=.03 generates word samples with
quality fraction 1 but only three modes and fails reconstruction; quality fraction
alone does not establish coverage. No tested ring configuration improves HQ
over the control while losing ring modes, but several tied recipes lose both.
R1/R2 was not run, so no cross-regularizer claim is available.

The diagnostic plots show the exact .00425 Adam control and the top three
non-Adam arms under the frozen count/hash rule, with the owner's tied dualnorm
starting recipe substituted: tensor nSGDA .1, zero-momentum dualnorm .01 and
prior-only .01. The original plot candidates remain recorded in the analysis. See
[Gaussian](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/dualnorm-tier1/diagnostics/gaussian1d_acquisition.png),
[ring](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/dualnorm-tier1/diagnostics/ring16_acquisition.png) and
[joint words](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/dualnorm-tier1/diagnostics/five_word_joint_acquisition.png).
Each plots relative update speed, G/E and D parameter norms, and the log spectral
product. In joint words, the dualnorm starter's spectral-log total variation is
4.580 versus Adam's 9.115 over the same 24 checkpoints, but it is higher than
Adam's on Gaussian and ring; smoothing is not universal. One dualnorm G matrix
grows 19.3x over those word checkpoints. Adam's D output matrix also
grows 13.6x, and some biases grow further. These are finite-budget growth flags,
not optimizer-specific proof of divergence. Explicit sampled-row traces for
dualnorm and prior-only normalization certify zero unsampled raw-row drift at
the recorded steps; native Adam reports gradient support instead.

The hash-selected global/tensor nSGDA, magnitude-graft and D-only rates lie on
their upper grid edges; prior-only's selection lies on its lower edge. These
are unresolved edges under a coarse PASS/hash objective, not calibrated optimal
rates. Swept Adam selects .016 by hash and also reaches 3/6; it does not improve
the current .00425 control's pass count.

Publication also preserves nine older trainer measurements under their original
joint-word evaluator source binding. Their existing selection metadata called
them current measurements even though the current v4 evaluator fingerprint
differs. They are now historical incumbents, with numerical results and every
recipe/source/runtime/RNG/task identity retained; current-contract validation is
not weakened. The [migration receipt](reports/forge/dualnorm-tier1/selection-migration.json)
records the original selection-card commit/blob and changed display metadata.
The historical Adam BCAP-pure pin is retained as a control; the current
dualnorm pin now selects the requested experimental starting point. The new pure-Adam declaration
joins BCAP-pure only for current presentation; its frozen family identity and
the existing canonical declaration remain intact.

Recommendation: **start the next BCAP-pure work from full dualnorm**, with
etaG=.01, D/G=1.5, etaPrior=.03 and network momentum 0. Use the same BCAP/loss
and task settings; these normalized step sizes have their own units. This is
the owner's requested experimental starting recipe, with Adam retained as the
control and the existing acquisition failures still investigated in the separate PR. Any ratio/prior-rate or edge extension needs its own finite
declaration and an explicit whole-recipe objective; this screen does not trigger
automatic expansion or width/depth training. The strongest acquisition follow-up
is zero-momentum dualnorm near .01. One possible **unexecuted** refinement is
etaG={.012,.016,.022}, D/G=1.5 and etaPrior=.03 with unchanged Tier 1 contracts;
three configurations would require a separately declared 7,560-second ceiling.
Original controls may be reused only under exact source/runtime/protocol keys.

Original predictions retain their requested native-benchmark/five-seed scope.
The repository forbids seed-only repetitions, and this authorized first stage
does not execute that scope. Their scores are therefore:

| Prediction | Score in the original scope | Seed-0 Tier 1 observation |
|---|---|---|
| P1: global/layer nSGDA match Adam | Unscored | Both reach Adam's 3/6 PASS count; no equivalence or seed-noise estimate. |
| P2: plain SGDA loses modes or needs a narrow window | Unscored | Best SGDA reaches 2/6; its largest rate produces five numerical incompletes. The coarse grid does not estimate a usable window. |
| P3: D-only dualnorm improves quality with smoother spectral proxy | Unscored | D-only dualnorm tops out at 2/6 versus Adam's 3/6; no native/core-sigma or convergence claim follows from its spectral traces. |
| P4: momentum .5 is best | Unscored | Zero momentum reaches 3/6; .5 and .9 each reach 2/6, opposing that preference in this screen. |
| P5: prior row normalization matches Adam | Unscored | All three rates tie 3/6, but all have worse ring HQ (.163/.600/.663 versus .823). A gate-count tie is not quality equivalence; sampled-row and native-Adam gradient-support diagnostics remain distinct. |
| P6: dualnorm rates transfer across width better than Adam | Unscored | Width/depth transfer was not run; no optimal-rate-vs-width plot is supplied. |

**Magnitude or direction?** Global and tensor-normalized SGD recover the
control's aggregate PASS count using raw gradient directions, so per-coordinate
adaptivity is not necessary to recover this particular count. However, the
tensor Adam-magnitude graft reaches only 2/6 in its own coarse sweep and does not
recover all of Adam's behavior. Normalization is useful here; this screen does
not establish that Adam's value is exclusively magnitude or exclusively
direction. Dualnorm supplies a common matrix/vector/prior update rule and reaches
3/6 at zero momentum, but rate transfer across width/depth remains untested.
There is no evidence yet for a single calibrated optimizer form that scales.

The executed source remains
`15eb7cb0911905e401bdfcd7e264945a7ea64d97`, with original recipe, runtime,
initialization, stream and result identities retained. One environment-interrupted
attempt has an explicitly linked execution-repair retry, with the original
receipt and cost preserved. Numerical failures are terminal and remain in the
denominator. Later API/observer and large-matrix safeguards are software fixes
for future runs; these numerical receipts do not qualify the corrected PR source.
The frozen observer can mislabel generator-phase inputs as the next critic's
real/fake inputs, so **all input-gradient curves are excluded**. Actual update,
weight and spectral-product traces remain usable; observer invariance tests
check model, optimizer and RNG states.

Matching audits use emitted complete-component state hashes where available,
named stream starts and frozen task contracts. Missing hashes and early-error
audits are explicit; no direct consumed-batch digest was recorded. Weight growth
is a finite-budget observation, not proof of unbounded growth. The critic proxy
is the product of matrix spectral norms, excluding Fourier/input maps and
nonlinearities; polar updates do not enforce a critic Lipschitz constraint.

[Actual-training GIF index](reports/forge/dualnorm-tier1/media.json) renders
saved certified observations without extra model calls or training. The
[artifact inventory](reports/forge/dualnorm-tier1/artifact-inventory.json) binds
raw logs, traces, states, original failures and recovery history outside Git.
[Software verification](reports/forge/dualnorm-tier1/software-verification.json)
records segmented test scopes and retained output hashes; overlapping counts
are not a single final-HEAD suite total. The screening profile remains
provisional, so this study supplies measurements rather than default adoption.

## BCAP-pure — dualnorm pacing follow-up (2026-10-06)

**The new experimental starting recipe passes 4/6 required Tier 1 tasks, up
from 3/6 for the matched starter.** Use full dualnorm with G/E step .012,
D/G=1.5 (D step .018), sampled-prior step .03 and network momentum 0:

```python
get_recipe("bcap", optimizer_family="dualnorm", lr=.012,
           d_lr_mult=1.5, prior_lr_mult=2.5, optimizer_momentum=0.)
```

Only optimizer settings changed. BCAP-pure loss, cap/coefficient, task
auxiliaries, initialization, architecture, data/prior laws, sampling, schedule,
update budgets and grading retain their existing contracts. This is one
global recipe at protocol seed 0, with no task-specific winner mixing. The
public BCAP default was still Adam when this readout was published; the later
[owner default decision](reports/forge/dualnorm-pacing-v2/DEFAULT_SELECTION.md)
makes this winning recipe the public `bcap` default while preserving the study's
unqualified status. The [single current leaderboard](reports/forge/technique-inventory.md)
uses this new experimental measurement; the [completed readout](reports/forge/dualnorm-pacing-v2/README.md),
[exact results](reports/forge/dualnorm-pacing-v2/results.json),
[verified analysis](reports/forge/dualnorm-pacing-v2/analysis.json) and
[selection receipt](reports/forge/dualnorm-pacing-v2/measurement-selection.json)
preserve the complete recipe and remaining failures. The original 41-recipe
screen above retains its original conclusions and source.

All 25 configurations completed their seven actual attempts: six required
tasks plus a separate clock diagnostic. There were 52 required PASS and 98
FAIL cells, 25 diagnostic passes, no numerical errors or retries. Both GPUs
were used. The finite search finished in approximately three hours for
19,655.90 worker seconds; its 12-hour and 63,000-worker-second ceilings were
not targets to exhaust. Completion verified all worker/child/lease exits,
zero reservations and delivery of the assistant callback.

Independent D/prior pacing at G=.01 did not improve the best complete count
beyond 3/6. The predeclared intermediate-rate stage found .012, which adds
two-pole: its terminal passing suffix grows from four to 17, above the required
five. Unused-token hold, AE hold and joint words retain their passes. Positive
network momenta .5/.9 at that winning pace reach at most 2/6. No larger search
or positive-momentum rescue follows automatically.

**The two remaining failures concern distribution shape.** Gaussian mean
error .12788 sigma and std ratio 1.05499 meet their bounds, but CDF KS .11428
exceeds .05. Its width improves while KS worsens relative to the current
.01 control. Ring retains all 16 modes and improves HQ .93018 to .94385,
but full nearest-assigned component covariance error 9.61552 exceeds .85.
Its mass TV .09302 and full minimum eigenvalue ratio .29107 pass. Core-only
covariance error .48050 and overall covariance error .09988 cannot replace
the full-component gate: all assigned tail samples count. Better HQ is not
evidence that the complete ring test passes.

The scientific execution is commit
`a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be`, digest
`f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226`.
Historical `15eb7cb0` receipts are not reused as this current-source control.
The scalar/ring hosts' critic-phase input-gradient probes are usable. The word
fixture leaves its critic in training mode during generator forwards, which
can still contaminate the observer's input labels; those word curves are
excluded. Actual update, weight, spectral and sampled-row traces remain usable.
Matrix spectral products exclude Fourier/input maps and nonlinearities and do
not bound the complete critic. [Actual-training GIFs](reports/forge/dualnorm-pacing-v2/media-index.json)
render saved observations without new training. [New archive provenance](reports/forge/dualnorm-pacing-v2/artifact-inventory.json)
preserves the raw evidence separately from the original archive.

**Magnitude or direction?** This follow-up improves performance by changing
pace while keeping dualnorm's direction rule fixed. Combined with the earlier
normalized-SGD ties, it supports continued normalization/pacing work, but
does not isolate the reason Adam works. No matched Adam/graft comparison ran
in this new source cohort. Dualnorm has one reusable update form; optimal-rate
transfer across width/depth remains untested. P1-P6 retain their original
native/five-seed scope and remain unscored. Before another paid study, analyze
the saved Gaussian CDF and ring assigned tails alongside the update diagnostics;
the remaining errors are not explained by mode count or two moments alone.
