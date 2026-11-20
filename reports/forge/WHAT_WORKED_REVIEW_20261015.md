# What actually worked: Forge leaderboard work, 2026-10-02 to 2026-10-15

<!-- superseded-banner: generated -->
> Historical review with follow-ups through 2026-10-16. Counts and recommendations below describe those dates.
> See the [current technique inventory](technique-inventory.md) for current status.
> Concluded negative studies retain their evidence in the immutable readouts linked below; their trainer patches
> are not adopted. The longer-budget adoption is tracked separately in [PR #440](https://github.com/255BITS/ParticleGAN/pull/440).

*Follow-up experiments: round 1 in §6, round 2 (2026-10-16) in §7, decisions needed in §7.6.*

A read-only review of the work since the Forge leaderboard was centralised (#232, 2026-10-02): about 190 PRs (#219 to #423) and the local experiment directories under `/ml2/hypergan`. Five independent reviewers each covered one area: the BCAP selected recipe, BCAP research recipes, Atlas/E22, the other families plus test and harness changes, and a test-by-test pass matrix built from `attempt-coverage.json`. The key claims were then re-checked against the PRs and the code. No experiments were run for this review.

**Summary.** A handful of trainer changes produced real, matched gains. A large share of the Tier 1 movement came from redefining tests rather than from trainer changes. Most research directions after 10-09 repaired one test and broke another. Every result is a single seed (seed 0), and several margins are thin.

How to read the counts:

- "71/152" counts *view requirements*. A test required by six views counts six times. BCAP's distinct-test record is 6/6 core Tier 1 plus 9/21 Tier 2.
- "Matched" means the same tests, task-owned budget, seed and source. Many leaderboard rows compare different source digests and runtimes. `technique-inventory.json` itself says "No ranking across incompatible source or runtime cohorts."

## 1. What actually worked, ranked

### 1.1 DualNorm with constant rates (BCAP's foundation)

DualNorm is an optimizer that rescales each update to a fixed size. In the 96-configuration search (#355), DualNorm was the only optimizer family with a 6/6 Tier 1 survivor: 4 of 64 DualNorm configurations, against 0 of 20 Adam configurations (best 4/6) and at most 3/6 for every other family (`bcap-tier2-search/README.md`).

Two small additions were required:

| Addition | What it did |
| --- | --- |
| Numerical-rank truncation (#331/#332, now the default) | Ring16 passes in an ordinary uninterrupted run: acquired at update 684, then 47 terminal passes. |
| Tiny smoothing, 1e-5 to 1e-3 (#345/#350) | Restored Gaussian smoke after serial autograd (#339) broke it. The margin is thin: confirmed KS 0.0448 against a 0.05 limit. |

Settings that mattered:

- Momentum hurts. DualNorm momentum of 0.5 or 0.9 gives 2/6 (#303).
- All four survivors used the same rates: G 0.012, D 0.018, prior 0.03 (#310 made these the default; that change alone took Tier 1 from 3/6 to 4/6 via two-pole).
- Loss choice is second-order. Non-saturating against relativistic swaps word hold for mode hold.
- Constant rates are why BCAP and BISPF pass the clock-free audit. Every family with learning-rate annealing (K3P, KA2, R1/R2, GAN v3) fails it.

### 1.2 Direction blend (#361, #367, #377, #383)

This is the cleanest gain of the period. Tier 2 went from 7/21 to 9/21 and no test was lost.

| Test | Before | After |
| --- | --- | --- |
| trajectory, identity error | 0.2399 | 0.000258 |
| residual_student | 0.0610 | 0.000249 |

On tasks where direction blend never fires, the trained tensors are byte-identical to the baseline.

### 1.3 Prior moment tether with a small-prior exemption (the real source of BISPF's lead)

BISPF (#415) scores 77/152 against BCAP's 71/152. That gap is a single test, `mode_hold` (BCAP 3 of the 5 required terminal passes, BISPF 12), counted once in each of six views.

- BCAP with a moment tether at 0.01 (#390) already turned `mode_hold` FAIL into PASS (streak 3 → 7). It also broke word hold (25/25 → 23/25).
- BISPF uses the same strength (`prior_reg=0.01`, `prior_regularizer=sbnp`). Its SBNP regularizer returns zero when the prior has at most `2*k_nn + d` rows (`particlegan/bispf.py:172-176`), which exempts the 5-row word table.
- The likely cause of the gain is therefore the tether plus the exemption, not the Born-Infeld chord regularizer, the Sinkhorn chords or harmonic capture. **No ablation has been run.**
  - *Update 2026-10-16: the ablation has now been run and this guess did not hold up (§6.1). On seed 0, the tether alone does not reproduce BISPF's mode_hold pass.*
  - *Update 2026-10-16, round 2: the harmonic/thermal capture factor is not the link either (§7.5). On BCAP + tether, adding BISPF's small Sinkhorn chord term alone gives the pass, but other single parts also do, which looks like trajectory sensitivity rather than one mechanism. On current source, BISPF still passes mode_hold and BCAP still fails it, by two checks, not one (§7.1).*
- #415's headline omits that rotated100 precision drops (0.290 → 0.266); grid100 (0.258 → 0.370) and staggered100 (0.300 → 0.342) improve.
- #393 explains the mode_hold-versus-word trade-off. The fixed tether's force relative to the adversarial force grows from 0.011 to 0.075 to 0.341 as the word adversarial signal shrinks. The proposed relative-force limit has not been run.

### 1.4 Longer horizons: many failures are early stops

- **K3P, KA2, R1/R2.** All three pass the full ring16 gate at 6,400 updates with their full schedule (`EXPERIMENT_MEMORY.md`, `ring16-schedule-20261011`). Terminal passes: K3P 66, KA2 47, R1/R2 exactly 5. Their matched 400-horizon baselines all fail. The board stops them at 1,600 updates.
- **Atlas.** Two-pole fails at 80 updates and passes at 320 (#313). Ring16 fails at 400 and passes at 1,600 (#313). The Gaussian still fails at 2,000 (#312).

Annealing plus a longer horizon conflicts with the clock-free requirement, so this is a policy decision as much as a technique question (§5).

### 1.5 Atlas local population control (#223): wins only on its own suite

| Measure (against PR155 E22) | E22 | Atlas |
| --- | --- | --- |
| Toy precision | 71.6% | 96.5% |
| Toy mass TV | 0.285 | 0.052 |
| MNIST embedding Fréchet distance | 1.88 | 0.54 |
| MNIST recall | 71.9% | 84.7% |

- Rotating 100-Gaussian task: Atlas reaches 96.1% early quality where E22 has 88.1%, but E22 wins at the endpoint.
- A faithful replay reproduces 19/19 gates (#266).
- On Forge tests, Atlas and E22 get identical verdicts. The leaderboard's 0/152 is mostly a blanket constructor refusal (`experiments/forge/api.py` `task_policy_blockers`). When the refused tasks are run anyway, they mostly FAIL.

## 2. Changes that moved the counts but are not technique gains

| Change | Effect | Note |
| --- | --- | --- |
| #316: ring16 rebuilt from BCAP's passing run (256 particles, MoG sigma 0.1, 1,600 updates, 96 checks; defaults to the exact BCAP DualNorm config) | Before (per the #289 Tier 1 run, not re-verified here): K3P and KA2 passed ring and BCAP failed. After: BCAP passes; K3P, KA2, R1/R2 and GAN v3 fail. | Swaps the family ranking. This one test now keeps K3P (11/21 Tier 2), KA2 and R1/R2 out of the counted totals. |
| #322: Gaussian split into Tier 1 smoke (any 1 of 24 states plus a confirming draw) and Tier 2 stability | Gaussian 0/9 → 19 PASS / 3 FAIL | Easier Tier 1. Stability has 0 passes in every selected family. |
| #347: word test split into acquisition and hold | Words 2/9 → all five runnable families pass | Easier Tier 1 |
| #339: serial autograd | BCAP Gaussian PASS → FAIL, causally isolated (#342) | Passes depend on floating-point accumulation order. |
| #263 / #280: ring, words and Gaussian moved into required Tier 1 | Lowered scores | Stricter |
| Per-view counting (#413/#419) | One test moves a total by up to 6 | Inflates gaps (BISPF +6 is one test) |
| 900 s word-smoke wall-clock limit on shared GPUs | Many TIMEOUT/INCOMPLETE results | Confidence LR's "+2" over its baseline is exactly these timeouts. |
| Stale-label display bug | BCAP is as stale as BISPF in `technique-inventory.json`, but the main table doesn't show it | The labeler looks up `bcap-dualnorm`; the row links to `bcap-pure`. |

Harness repairs that turned skips into measured FAILs, without adding a pass: #274, #275, #292, #341, #352, #414, #422.

## 3. Clearly did not work

| Direction | PRs / evidence | Result |
| --- | --- | --- |
| Five prior regularizers: proximal spring, Coulomb, SIGReg, moment tether, initial-distribution MMD | #387 to #392 | No net gain. Ring16 covariance is the gate most of them break. |
| Per-step speed limits: hydraulic, Armijo, role balancing, idle generator, bounded secant | #360, #370, #371, #381, #397 to #399 | Raise native precision (0.24 → 0.5-0.66) and Gaussian stationary holds (up to 71/72). Break covariance, ring16 or mode_hold. |
| Lookahead methods: network or raw-gradient optimism, extragradient, evidence tempering | #359, #362, #364, #378 | Consistently regress acquisition and mode_hold |
| Transport and density forces: local-v2, Sinkhorn, confidence transport, tail-moment, prior-only, anisotropic kNN | #358, #369, #376, #379, #380, #382 | Fix vector density tests; break two_pole, unused_token and ring16 |
| 2026-10-15 combination screens (direction + transport, ± whole-G or adversarial-channel history; 4 repair arms) | `combination-training-20261015/TRAINING-DIAGNOSIS.md` | All fail. Only the direction-only baseline passes both two_pole and ring16. |
| Confidence, signed and positive-evidence learning rates | #405 | Test-for-test identical to the matched baseline, apart from timeout artifacts |
| G-history attenuation | PR399 rows | Only family to pass `gaussian1d_stability` (72/72), but loses Tier 1 gates, net -2 |
| About 40 Atlas repairs, 10-08 to 10-11 (clean 100-Gaussian, images, ring) | `atlas-*-task*` dirs | All rejected. The clean 100-Gaussian failure is a training-dynamics problem; capacity is sufficient. |
| 100-mode grids (grid100, rotated100, staggered100) | 84 attempts | 0 passes. Best precision about 0.55 against 0.97; the true target scores 0.989. |
| 2026-10-14/15 backfill | `attempt-coverage.json` | 853 attempts, about 29.5 GPU-hours, 0 selected totals changed. All 130 new passes are on research rows. |

Recurring pattern: **endpoint metrics often pass while the sustained (terminal-suffix) gate fails.** Read partial wins from the gate, not the endpoint.

## 4. Each family owns a capability; no single recipe bridges them

| Capability | Who has it | What it costs |
| --- | --- | --- |
| ring16 acquisition, clock-free audit | DualNorm + constant rates (BCAP, BISPF) | — |
| mode_hold | Moment tether (BISPF); finite critic cap | Finite cap loses Gaussian smoke, stripes and two_broad. |
| Gaussian retention, vector_anisotropic, vector_overlap | Annealing (K3P/KA2); G-history | Clock-free audit, ring16 within 1,600 updates |
| vector_unequal_mass, img_blobs4, img_intensity2 | Kinetic transport and cell-moment recipes | ring16, two_pole, mode_hold |
| img_bars4, img_intensity2 | R1/R2, GAN v3 | img_stripes2 (0 of 21 recipes pass both image tests), ring16 |
| vector_unequal_width | Balanced transport (#375) only | Rare density, native precision |

Coverage of the 29 required tests:

| Combination | Tests passed |
| --- | --- |
| Best single recipe (BISPF) | 18/29 |
| Union of selected recipes | 23/29 |
| Union of all recipes | 25-26/29 |

The binding conflict: **no recipe that passes ring16 passes any held test that someone else passes.**

## 5. Where to dig next

*Updated 2026-10-16 after round 2 (§7). Round 1 (§6) ran the five original leads; round 2 ran five more. Every result is seed 0 only.*

What is now answered:

1. **BISPF ablation: guess refuted (§6.1), and the harmonic/thermal link refuted too (§7.5).** BISPF's mode_hold lead reproduces on current source (§7.1). On BCAP + tether, BISPF's chord term gives the pass, but so does another single part; read it as sensitivity, not a mechanism.
2. **Relative-force tether: refuted at rho=0.03 (§6.2).** The #393 word-hold question is still untested.
3. **Horizon and clock for K3P/KA2: mostly answered (§6.3, §7.2, §7.4).** At a 6,400 cap, K3P and KA2 fail only because their own `network_lr_horizon_cap=1600` floors the learning rate; without it they pass. A fixed internal counter is just the step clock relabelled. K3P at constant rate passes the 6,400 cap and the unchanged audit.
4. **cell-quota-min-change: refuted as a Tier 1 re-entry (§6.4).**
5. **Wall-clock hits: partly answered (§7.3).** Signed Confidence LR and the BCAP baseline reach 6/6 Tier 1 on rerun (with a runtime-mix caveat); Positive-evidence LR and two kinetic recipes are still NOT MEASURED; Confidence LR's lead over the baseline shrinks from +2 to +1, the remaining point being an unmeasured hold.

Next steps, in priority order:

1. **Make the policy decisions in §7.6** (ring16 cap, clock audit, wall-clock reporting). Nothing below changes a counted score until they are made.
2. **Make GPU measurements trustworthy.** Pause jobs outside the lease during reruns (or run at a quiet time), fix the starving `gpu-lease.sh`, and land a coordinator opt-out for the post-completion compile. Contention cost about 75 minutes of waiting per lead and left five cells unresolved.
3. **Finish the remaining wall-clock cells** (about 1.3 GPU-hours): Positive-evidence LR with the fixed contention-aware limit, the two kinetic word cells, the BCAP baseline's word hold, and the C / Z + chord word smoke from §7.5. Rerun on the declared runtime, or record the runtime mix.
4. **If the 6,400 cap is adopted,** measure K3P-const on the rest of Tier 1 and Tier 2 as a dual-pass candidate, and decide whether K3P/KA2 drop `network_lr_horizon_cap`.
5. **Add seeds only where a verdict hangs on a thin margin:** BCAP mode_hold (misses by 2), BCAP unused_token_hold (exactly 5), R1/R2 ring16 @ 6,400 (exactly 5), and the Z + chord mode_hold pass.
6. **Lower priority.** Signed Confidence LR's checkpoint-resume error; export the relative cap's in-training scale before trying another rho; the DualNorm plus cell objective arm on mode_hold and gaussian1d_stability.

## 6. Follow-up experiments (2026-10-15/16)

One workflow ran each §5 lead in its own worktree, with a decision rule written down before training. An independent verifier then rechecked each lead against the attempt receipts (`graded-result.json`, `terminal.json`, queue events). Verdicts below are **as corrected by the verifier**. The hygiene lead used only CPU and saved files and had no verifier pass.

Shared conditions: seed 0, one attempt per arm, an RTX A6000, deterministic CUDA, and one frozen source digest per lead. Every verdict comes from the sustained Forge gate. "Suffix" or "streak" is the number of consecutive passing checks at the end of the run, and most gates need at least 5. All results are single-seed: they show what happened on this seed, not how robust it is. No change alters a default, gate or budget, and nothing was merged.

Total GPU time across the four trainer leads: 1,041.6 s (about 17 minutes).

### 6.1 BISPF ablation: REFUTED (holds up)

**Question.** Does BISPF's mode_hold pass over BCAP come from the SBNP prior moment tether alone (`prior_reg=0.01`, off for small tables), rather than from its critic-side parts (Born-Infeld chord penalty, Sinkhorn pairing, harmonic capture)?

**Arms.** C = BCAP control; T = BCAP plus the tether only; B = full BISPF; B0 = BISPF with the tether off. B0 is exploratory: it was declared after the main verdict was known, and the readout says so.

| Arm | mode_hold | Checks passing | Suffix | Notes |
| --- | --- | --- | --- | --- |
| C BCAP control | FAIL | 10/24 | 3 | one dip to hq 0.869 at step 1050; the final check alone would pass |
| T BCAP + tether | FAIL | 0/24 | 0 | stuck at 7 of 8 modes from step 650 |
| B full BISPF | PASS | 13/24 | 12 | 8 modes from step 350 |
| B0 BISPF, tether off | FAIL | 0/24 | 0 | stuck at 7 of 8 modes from step 400 |

**Verdict.** REFUTED under the preregistered rule (committed in dccb6839 before the first attempt): on this seed the tether alone does not reproduce BISPF's pass. Stage 2 was correctly skipped. Verifier corrections:

- The reading that BISPF needs the tether *and* its critic-side parts together rests on one seed per arm, and partly on the post-hoc B0 arm. It is a lead, not a finding. T and B0 both stalling at 7 modes could also mean mode_hold is fragile at seed 0 under any change.
- T is a stronger tether than BISPF's own: BISPF scales the tether down with a harmonic thermal factor when d<=4, and T cannot have that factor.
- The readout overstates B0's hq at steps 400-650. It was 0.81-0.90, not at least 0.90.
- The control fails by a single check.

**GPU time:** 215.5 s. **Links:** branch `task/dig-bispf-ablation-20261015`, draft [PR #426](https://github.com/255BITS/ParticleGAN/pull/426), readout `reports/forge/dig-20261015/bispf-ablation.md` on that branch (local: `/ml2/hypergan/dig-20261015/bispf-ablation/READOUT.md`).

### 6.2 Relative-force tether: REFUTED (holds up)

**Question.** If the 0.01 moment tether's gradient on the latent table is capped at rho=0.03 times the other (mostly adversarial) gradient, does BCAP keep the fixed tether's mode_hold repair, avoid its word-hold regression and keep ring16? The new opt-in field is `prior_tether_relative_cap`. Its default of 0 keeps the old fixed tether exactly. It uses only gradient norms, so it is portable.

**Arms.** C = BCAP leader control; F = fixed 0.01 tether (reproduces #390); R = 0.01 tether capped at rho=0.03.

| Arm | mode_hold | ring16_acquisition |
| --- | --- | --- |
| C control | FAIL, suffix 3, 10/24 | PASS, suffix 26, 26/96, covariance error 0.497 |
| F fixed tether | PASS, suffix 7, 10/24 | PASS, suffix 9, 10/96, covariance error 0.726 |
| R capped tether | FAIL, suffix 0, 0/24, never reaches 8 modes; ends at 6 modes, hq 0.863 | FAIL, suffix 0, 0/96, covariance error 1.343 (limit 0.85) |

Word smoke, word hold and the rest of Tier 1 were not run: rule 1 stops the run after stage 1.

**Verdict.** REFUTED for rho=0.03 on seed 0. R is worse than both C and F, so it is not a middle ground between them. As in #390, these gates react non-monotonically to small changes in prior force. The #393 word-hold question is **still untested**. Verifier corrections:

- How often the cap actually engaged is unmeasured. Its in-training scale is recorded in memory but not exported to any receipt. R leaving F's trajectory around step 250 is unexpected if the cap rarely binds, and no check rules out an implementation artifact. The verifier's code read found no bug, and the 11 unit tests pass.
- The harness story in the readout is partly wrong. The `deepcopy` import was already in the frozen commit 1f89a8241. The logged first-drain crash was the `_attempts` TypeError, not a NameError. The arms still share source digest 120d9cb4.
- R also drops to 6 modes at steps 400 and 1000, not only at the end.

**GPU time:** 262.6 s. **Links:** branch `task/dig-relative-tether-20261015`, draft [PR #429](https://github.com/255BITS/ParticleGAN/pull/429), readout `reports/forge/dig-20261015/relative-tether.md` on that branch (local: `/ml2/hypergan/dig-20261015/relative-tether/READOUT.md`).

### 6.3 Horizon and clock: progress-driven schedule for K3P: REFUTED (holds up)

**Question.** Can a schedule that does not count steps give K3P the benefit annealing has on ring16, passing ring16 within the 1,600-update cap as well as the clock-free audit? The new opt-in `Recipe.progress_schedule` replaces step/horizon with an internal progress value p in K3P's learning-rate decay, input-noise anneal and output-noise warmup. p rises by at most 0.001 per update, and only while the critic's paired order accuracy (an exponential moving average) is below 0.75. p never decreases and is saved with the trainer.

**Arms.** A = shipped K3P step schedule; B = constant rates, no noise anneal; C = progress schedule; D = ungated internal counter (diagnostic). B, C and D also start the critic spike guard at update 0 (A keeps 200), so A against the rest is not a pure schedule comparison.

| Arm | ring16 (0/96 passing for all) | Final covariance error (min) | Clock-free audit |
| --- | --- | --- | --- |
| A step | FAIL; also fails hq (0.81) at the end | 2.950 (2.873) | FAIL: step_label and horizon change the state |
| B constant | FAIL; 39/96 checks pass every gate except covariance | 1.337 (1.285) | PASS, 4/4 comparisons agree |
| C progress | FAIL; 30/96 pass every gate except covariance | 1.787 (1.734) | FAIL by declaration only; 4/4 measured comparisons agree |
| D ungated | FAIL; 47/96 pass every gate except covariance | 1.895 (1.851) | FAIL by declaration only; 4/4 measured comparisons agree |

**Verdict.** REFUTED; Tier 2 skipped by the stop rule. The progress gate behaved as designed: it held p back while the critic was winning (p 0.098 at update 200), then advanced it. The learning rate was close to annealed by update 1,550 (multiplier 0.059 against a floor of 0.01), not fully annealed as the readout says. Annealing inside the cap did not help: the arm that never anneals came closest. **For K3P on ring16, the 1,600-update horizon is the binding constraint, not the clock rule.** C and D fail the audit only because its declaration table lists K3P's step-schedule fields. Changing that table is a gate change and was not made. The readout's 6,400-update policy figures were not verified. Verifier corrections: "covariance fails every time" does not fully describe arm A, which also fails hq. Every drain crashed afterwards with the existing `deepcopy` NameError, but the graded results were already written.

**GPU time:** 175 s. **Links:** branch `task/dig-horizon-clock-20261015`, draft [PR #425](https://github.com/255BITS/ParticleGAN/pull/425), readout `reports/forge/dig-20261015/horizon-clock.md` on that branch (local: `/ml2/hypergan/dig-20261015/horizon-clock/READOUT.md`).

### 6.4 cell-quota-min-change re-entry probe: REFUTED (holds up)

**Question.** cell-quota-minimum-change-forward-kl-g-v1 is the only recipe seen passing both gaussian1d_stability and mode_hold. Does it also pass ring16, which would make it a cheap Tier 1 re-entry? The recipe existed only in the unmerged Astra workspace. It was ported with additive, default-off changes and was confirmed faithful: gaussian1d_smoke first confirms at step 292 with 18/24, identical to the Astra record.

| Arm | ring16 | mode_hold | gaussian1d_stability | gaussian1d_smoke |
| --- | --- | --- | --- | --- |
| cell-quota candidate | FAIL, 0/96; 16 modes, hq 0.716, covariance error 3.13 (flat plateau) | PASS, streak 21 | PASS, 72/72 stationary, 24/24 shifted | PASS, 18/24 |
| BCAP DualNorm control | PASS, streak 26 from step 1184 | FAIL, streak 3 | FAIL, 2/72, 0/24 | PASS, 3/24 |
| Candidate's NSGDA base, no cell objective | FAIL, 15 modes, hq 0.458, covariance error 54.2 | — | — | — |
| DualNorm control + cell objective (Addendum A) | FAIL, 0/96; best hq 0.902, best covariance error 0.960 | not run | not run | — |

two_pole and five_word_joint_smoke are **BLOCKED**. The cell objective needs the public GANTrainer path, so planning refuses those tasks. The candidate can therefore never qualify for Tier 1.

**Verdict.** REFUTED. This pair is the review's ring16-against-held-tests conflict (§4) reproduced as a clean matched pair. The switch away from DualNorm causes most of the ring16 deficit, and the cell objective by itself also works against ring16. Verifier corrections:

- The control's ring16 pass is thin and late, on a single seed (covariance error 0.46 to 3.8 in the second half), so "decisive" oversells the separation.
- The times written inside the preregistration addenda are wrong (21:20 and 21:22, actually 21:16 and 21:18). The order was still correct.
- After step 400, mass_tv ranges 0.014-0.038.
- Stage C's clusters are not "over-wide" (min eigen ratio about 0.79).
- The blocker checks the `public_components` execution path, not sigma directly.

**GPU time:** 388.5 s. **Links:** branch `task/dig-cell-quota-probe-20261015`, draft [PR #427](https://github.com/255BITS/ParticleGAN/pull/427), readout `reports/forge/dig-20261015/cell-quota-probe.md` on that branch (local: `/ml2/hypergan/dig-20261015/cell-quota-probe/READOUT.md`).

### 6.5 Hygiene: DONE (CPU only, not independently verified)

| Item | Result |
| --- | --- |
| (a) Stale-label bug | Confirmed and fixed. The labeller checked only `family["id"]`, and its anchor could never match a row that carries a configuration link. A row is now labelled when its own id or its displayed cohort's `configuration_family` is stale. Regenerating with `--refresh-publication` changed only the BCAP label and two provenance hashes, and no score moved. Related test failures are 124 both before and after (the same set), plus 1 new passing unit test. |
| (b) TIMEOUT/INCOMPLETE | 20 of 22 are wall-clock hits, and no FAIL is. Five word-smoke runs had already confirmed a pass (step 834) before the 900 s kill. Counting them moves signed Confidence LR, Positive-evidence LR and the matched BCAP baseline to 6/6 Tier 1. Confidence LR's "+2" is exactly word smoke plus the blocked five_word_joint_hold. Of 13 Tier 3/ring16 hits, 5 are determined FAIL and 8 never had a passing check, so none flips. Timed-out runs ran at 2.8-13.9 updates/s against 24-28 normally, which points to GPU contention. |
| (c) Gaussian smoke margin | Confirmed: KS 0.044836, confirmation 0.033797, with 1/24 pairs passing. The thin margin is general: 16% of 180 Gaussian-smoke passes rest on 1 observation and 28% on 2 or fewer, including GAN v3 (1/24), BISPF (2/24) and BCAP (3/24). |

Caveats: the would-PASS calls are inferred from the earliest-confirmation rule, not graded, so a late guard failure is still possible. The timing comparison mixes recipe cost with contention. Recommended but not applied: report wall-clock hits as NOT MEASURED, use a contention-aware cap, and re-run under the GPU lease before counting.

*Round 2 update (§7.3): under the GPU lease, signed Confidence LR and the BCAP baseline did pass word smoke, but Positive-evidence LR timed out again, so two recipes reached 6/6, not three.*

**GPU time:** 0 s. **Links:** branch `task/dig-hygiene-20261015`, draft [PR #428](https://github.com/255BITS/ParticleGAN/pull/428) (stacked on #424), readout `reports/forge/dig-20261015/hygiene.md` on that branch (local: `/ml2/hypergan/dig-20261015/hygiene/READOUT.md`).

### 6.6 Notes common to all leads

- Every follow-up branch is based on this review's commit, so PRs #425 to #429 also show this document in their diffs.
- `tests/test_forge_dualnorm_smoothing_search.py::test_other_families_do_not_claim_an_active_smoothing_axis` failed in the relative-tether checkout. It touches none of that branch's code and was not re-run on clean develop.

## 7. Round 2 follow-ups (2026-10-16)

A second workflow ran five leads drawn from §5 and §6, each in its own worktree from develop + #430 (the `compile_memory` crash fix), each with a decision rule committed before training. An independent verifier then rechecked every lead against the receipts (`graded-result.json`, `result.json`, request files, queue events) and reran the unit tests on CPU. It did not rerun GPU jobs. All five leads held up. Verdicts below are **as corrected by the verifier**.

Shared conditions as in §6: seed 0, one attempt per arm, RTX A6000 cards, deterministic CUDA, sustained Forge gates. Every task was marked `research_diagnostic`, so these receipts do not fill ordinary leaderboard cells, and no published score was regenerated. **All results are single-seed.** Several deciding margins are one or two checks.

Terms used below:

- **Suffix / terminal streak**: consecutive passing checks at the end of a run. Most gates need at least 5.
- **Cap**: the number of training updates a task allows. ring16's current cap is 1,600.
- **NOT MEASURED (wall-clock)**: a run killed by its wall-clock limit before the gate could decide. It is neither PASS nor FAIL.
- **GPU lease**: a lock that lets only one of this workflow's jobs use a GPU at a time. It does not stop jobs outside the workflow from sharing the card.
- **Clock-free audit**: the Tier 1 test that restarts a recipe from saved state and checks that its behaviour does not depend on the global step count or declared run length (the "horizon").
- **Internal counter**: a counter the recipe keeps in its own saved state, for example "updates done so far".

Total GPU time for round 2: 13,326 s (3.7 GPU-hours), of which the wall-clock reruns used 6,366 s.

### 7.1 BISPF and BCAP refresh on current source: SUPPORTED (holds up)

**Question.** On current source (develop + #430), do all recorded verdicts of BISPF (`bispf-v1`) and the selected BCAP recipe (`recipe-prior-refactor-bcap-pure-v1`) on the 17 still-counted Tier 1 and Tier 2 tests reproduce? In particular, does BISPF still pass mode_hold while BCAP fails it?

**Arms.** BISPF and BCAP, one frozen source digest (6681974e), Tier 2 drained only after Tier 1 passed.

| Arm | Tier 1 (7 tests + word hold) | Tier 2 (9 tests) | mode_hold | Notes |
| --- | --- | --- | --- | --- |
| BISPF | all PASS | all PASS | PASS, 13/24, suffix 12 | ring16 9/96 suffix 9 |
| BCAP | all PASS | 8 PASS, mode_hold FAIL | FAIL, 10/24, suffix 3 (needs 5) | endpoint looks fine (8 modes, hq 0.989); a late quality dip fails the sustained gate; unused_token_hold passes with exactly 5 |

**Verdict.** SUPPORTED: 34 of 34 cells reproduce, and every sustained margin matches the older receipts exactly, so the newer runtime changed nothing on these tests. BCAP's mode_hold curve equals the 2026-10-15 ablation control to the last digit. Implied counted scores stay at BISPF 77 and BCAP 71 (not regenerated). Verifier corrections:

- BCAP's Tier 2 request was *enqueued* (not drained) before its Tier 1 finished, a small departure from the rule as written. Enqueuing trains nothing, so no verdict is affected.
- "Word smoke confirmed after 20,001 updates" is loose: the first confirmed steps were 2,501 (BISPF) and 834 (BCAP); 20,001 is the completed run length.
- The preregistration called BCAP's mode_hold miss "one check short". A suffix of 3 against 5 is two short. (§1.3 and §6.1 said "one check"; that is corrected here.)
- Both Tier 1 coordinators were stopped after every attempt was graded, so queue lifecycle and compiled memory for those two requests were never finalised.

**GPU time:** 2,732.6 s. **Links:** branch `task/dig2-bispf-refresh-20261016`, draft [PR #431](https://github.com/255BITS/ParticleGAN/pull/431), readout `reports/forge/dig2-20261016/bispf-refresh.md` on that branch (local: `/ml2/hypergan/dig2-20261016/bispf-refresh/READOUT.md`).

### 7.2 ring16 cap 1,600 → 6,400: REFUTED as stated, but it found the real blocker (holds up)

**Question.** If ring16's cap rises to 6,400 updates, with each family's step schedule spread over the full cap and the gate unchanged, do K3P, KA2 and R1/R2 pass while BCAP and BISPF still pass with no late collapse? The new cap is a separate task, `ring16_acquisition_6400` (384 checks, same cadence), which pins the original task's sha256. The original task and its receipts are untouched.

| Recipe | ring16 @ 6,400 | Terminal streak | ring16 @ 1,600 control |
| --- | --- | --- | --- |
| BCAP | PASS (robust) | 314 | PASS |
| BISPF | PASS (robust), one covariance dip at 3,584-3,884 | 151 | PASS |
| K3P selected | FAIL, 0/384; local spread decays after 1,600 | 0 | FAIL |
| KA2 selected | FAIL, 0/384 | 0 | FAIL |
| R1/R2 selected | PASS (thin, fragile); fails on a 96-check reading | 5 (the minimum) | FAIL |
| GAN v3 release 0.7 | PASS (robust) | 43 | FAIL |
| D1: K3P with `network_lr_horizon_cap=None` (diagnostic) | PASS | 66 | — |
| D1: KA2 with `network_lr_horizon_cap=None` (diagnostic) | PASS | 47 | — |

**Verdict.** REFUTED under the rule: it needed at least 2 of K3P, KA2 and R1/R2, and only R1/R2 passed. The longer cap is safe for BCAP and BISPF. The preregistered diagnostic D1 found why K3P and KA2 fail: their own `network_lr_horizon_cap=1600` drops the generator/critic learning rate to its 0.01 floor at update 1,600. With only that field removed, both pass and exactly reproduce the 2026-10-11 6,400-update study. So the earlier "full schedule" passes needed the anneal to span the whole run. D1 is diagnostic and not counted.

Recount (CPU only, reproduces the published 71/77/20/20/20/20 first): with the new task, R1/R2 and GAN v3 go to 21/152; K3P and KA2 stay at 20. All four are still blocked from Tier 2 by the clock-free audit. With diagnostic Tier 2 credit, R1/R2 and GAN v3 would reach 75/152. A proposed revision 10 of the `discriminator_stability` view is in the PR but not activated. Verifier notes: the learning-rate-floor mechanism is inferred from the card config plus D1, not from a logged learning rate. D1 reused the main runs as its controls (same digest), which is valid.

**GPU time:** 953 s. **Links:** branch `task/dig2-ring16-cap-20261016`, draft [PR #433](https://github.com/255BITS/ParticleGAN/pull/433), readout `reports/forge/dig2-20261016/ring16-cap.md` on that branch (local: `/ml2/hypergan/dig2-20261016/ring16-cap/READOUT.md`).

### 7.3 Wall-clock reruns: SUPPORTED, with corrections (holds up mostly)

**Question.** Are most 900 s word-smoke TIMEOUT/INCOMPLETE results (and one ring16 timeout) wall-clock hits caused by shared-GPU contention rather than recipe failures? Do solo reruns under the unchanged limit give real verdicts?

| Recipe | Test | Before | Rerun | Key numbers |
| --- | --- | --- | --- | --- |
| BCAP baseline (`bcap-default-baseline-direction-v1`) | five_word_joint_smoke | INCOMPLETE | **PASS** | 795.6 s, 23/24 confirmed |
| Signed Confidence LR | five_word_joint_smoke | TIMEOUT | **PASS** | 789.3 s, 23/24; earlier run was 1.8-2.0x slower per update |
| Confidence LR (control) | five_word_joint_smoke | PASS | PASS | 819.4 s; paired smoke delta vs baseline = 0 |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | ring16_acquisition | TIMEOUT | **FAIL** | 0/96 in 33 s (earlier 300 s, about 10x contention) |
| Positive-evidence LR | five_word_joint_smoke | TIMEOUT | NOT MEASURED | 900 s at step 16,668/20,001, GPU shared with non-lease jobs |
| kinetic_transport_local_v2 | five_word_joint_smoke | TIMEOUT | NOT MEASURED | 900 s at 17,501; would need about 1,020 s |
| kinetic-local-dualnorm-g-history-v1 | five_word_joint_smoke | TIMEOUT | NOT MEASURED | 900 s at 12,501; would need about 1,370 s |
| BCAP baseline (A1) | five_word_joint_hold | — | NOT MEASURED | 300 s at step 4,501/4,834, all 23 checks so far confirmed |
| Signed Confidence LR (A1) | five_word_joint_hold | — | ERROR | "word continuation changed named RNG bindings" when resuming its own smoke checkpoint |
| Positive-evidence LR + contention-aware limit (A2) | five_word_joint_smoke | — | ERROR | supervisor extended to 1,350 s, but the adapter's own deadline fired at 901 s; fixed and unit-tested, not rerun on GPU |

**Verdict.** SUPPORTED for the three cells that became real verdicts. Signed Confidence LR and the BCAP baseline go from 5/6 to 6/6 Tier 1; the ring16 row loses its timeout marker and stays 0/6. Positive-evidence LR did **not** reach 6/6, so §6.5's "three recipes would reach 6/6" was one too many. New opt-in code (`experiments/forge/wallclock.py`): a NOT MEASURED outcome kept out of fail totals and paired deltas, a contention-aware limit, and a rerun overlay that changes a cell only on a real PASS/FAIL. Verifier corrections:

- The reruns ran on Python 3.12 / torch 2.13, while the requests and original cells declare Python 3.11 / torch 2.11. The baseline's new 6/6 therefore mixes one torch 2.13 PASS with five torch 2.11 cells. The readout did not disclose this.
- The baseline and Confidence LR reruns used fill-state requests, not "the same request bytes". The science fingerprints (revision, execution/evaluation hashes, seed, RNG bindings) are identical.
- Confidence LR's "+2" did not fully disappear. The smoke point was a timeout artifact; the remaining +1 (15/29 against 14/29) is the baseline's hold, which is still NOT MEASURED.
- The "slow at the declared limit" label on the two kinetic cells rests on contended reruns, against the rule's solo-throughput condition.
- The lease is not solo: Atlas/E22 fill workers, llama-server and a desktop game shared the GPUs in several cells. The shared `gpu-lease.sh` also starves waiters (holders re-take within 2-3 s, waiters poll every 20 s); lanes waited 60-75 minutes.

**GPU time:** 6,365.6 s. **Links:** branch `task/dig2-wallclock-reruns-20261016`, draft [PR #434](https://github.com/255BITS/ParticleGAN/pull/434), readout `reports/forge/dig2-20261016/wallclock-reruns.md` on that branch (local: `/ml2/hypergan/dig2-20261016/wallclock-reruns/READOUT.md`).

### 7.4 Measured clock-free audit (proposed v2): SUPPORTED by the letter, empty in substance (holds up)

**Question.** Can K3P or KA2 annealing pass both ring16 at a 6,400 cap and the clock-free audit, if the audit decides from measured parity (a proposed `measured_parity_v2` rule) rather than from v1's declaration table? Under v2, any measured difference fails; a declared step-schedule entry is overridden only when a power witness shows the probe could have caught it; fixed internal counters are overridden only under `internal_counter_policy='measured'`.

**Arms.** K3P and KA2 × {step = shipped schedule; const = constant rates; count6400 = anneal driven by an internal counter at rate 1/6400; gated = anneal advanced only while the critic is winning}, on a scoped research copy of the 6,400-update ring16 task.

| Arm | ring16 @ 6,400 (streak) | v1 audit | v2 measured | v2 strict |
| --- | --- | --- | --- | --- |
| K3P step | PASS (104) | FAIL | FAIL | FAIL |
| K3P const | PASS (47) | **PASS** | PASS | PASS |
| K3P count6400 | PASS (104), bit-identical to step | FAIL | PASS | FAIL |
| K3P gated | FAIL (0), covariance 1.99 | FAIL | PASS | PASS |
| KA2 step | PASS (54) | FAIL | FAIL | FAIL |
| KA2 const | FAIL (3), destabilises near 4,800 | FAIL | PASS | FAIL |
| KA2 count6400 | PASS (54), bit-identical to step | FAIL | PASS | FAIL |
| KA2 gated | FAIL (0), covariance 1.92 | FAIL (restart) | FAIL (restart) | FAIL |

**Verdict.** Technically SUPPORTED under the rule (count6400 passes ring16 and v2 with measured counters, for both families), but **the support adds nothing**: count6400 is bit-identical to the step schedule in all 384 observations. A fixed internal counter is the horizon clock written as a recipe constant, so v2-measured would simply accept a relabelled clock. Under strict counters no annealing arm passes both; the horizon-free critic-gated anneal fails ring16. The useful finding is **K3P at constant rate passes ring16 @ 6,400 and the unchanged v1 audit**, so if the 6,400 cap is adopted, K3P-const is a dual-pass candidate with no audit change. KA2-const does not hold. Verifier corrections: the guard-counter witness is hardcoded to true rather than inspecting state; the step controls fail v2 on both step_label and horizon; KA2 gated fails v1 on restart, not on a declaration; 704 s is the sum of graded seconds, not measured lease time. Also, `FORGE_SKIP_COMPLETION_COMPILE`, used by other leads, exists only on their branches (#429, #431), not in develop.

**GPU time:** 704 s. **Links:** branch `task/dig2-clock-audit-measured-20261016`, draft [PR #432](https://github.com/255BITS/ParticleGAN/pull/432), readout `reports/forge/dig2-20261016/clock-audit-measured.md` on that branch (local: `/ml2/hypergan/dig2-20261016/clock-audit-measured/READOUT.md`).

### 7.5 BISPF harmonic/thermal link: REFUTED (holds up)

**Question.** §5 lead 4: does BISPF's mode_hold pass come from its harmonic/thermal capture factor, which scales the SBNP tether? What is the smallest set of BISPF parts that gives mode_hold on BCAP + tether?

**Arms.** Knock-outs from full BISPF (B), and knock-ins onto the reduced arm Z (BISPF with all six parts off, which reproduces BCAP + tether bit for bit).

| Arm | mode_hold | Checks passing | Suffix | Notes |
| --- | --- | --- | --- | --- |
| C BCAP control | FAIL | 10/24 | 3 | |
| T BCAP + tether | FAIL | 0/24 | 0 | 7 modes |
| B full BISPF | PASS | 13/24 | 12 | |
| H0 no capture | PASS | 13/24 | 12 | identical to B through step 1,050 |
| S0 no tether thermal scaling | PASS | 13/24 | 12 | identical to B through step 1,050 |
| K0 no chord | PASS | 5/24 | 5 | thin |
| P0 hinge cap (softplus off) | FAIL | 7/24 | 4 | |
| F0 no floor/gauge | PASS | 10/24 | 10 | |
| Z reduced (= T bitwise) | FAIL | 0/24 | 0 | |
| Z + capture | FAIL | 0/24 | 0 | identical to Z; capture never fires |
| Z + softplus | FAIL | 0/24 | 0 | |
| Z + chord | PASS | 12/24 | 11 | minimal set |
| Z + floor/gauge | PASS | 10/24 | 6 | |

Stage 2, C vs Z + chord: gaussian1d_smoke, two_pole and ring16 pass in both (ring16 suffix 26 vs 31). five_word_joint_smoke hit the 900 s limit at step 7,501 in **both** arms on busy GPUs, so word hold is unresolved.

**Verdict.** REFUTED: capture first fires at critic call 1,075, after BISPF already holds 8 modes, and the passing window is set before it can act. By the rule the minimal set is BISPF's 2%-weight Sinkhorn chord term on BCAP + tether. The attribution is fragile: chord alone and floor/gauge alone each suffice on BCAP + tether, softplus alone does not, yet inside BISPF only softplus is necessary. That looks like trajectory sensitivity to small critic-penalty changes once the tether is on, not a single mechanism. New opt-in field `Recipe.bispf_ablation` (default "none"). Verifier corrections: the claimed 1,105-call bit-identity test against the old `bispf.py` is not in the committed tests (the committed test checks only step 1 on the new code); default behaviour is still confirmed because B and T reproduce the round-1 #426 runs byte for byte. Stage 1 and stage 2 used different frozen digests, each matched internally.

**GPU time:** 2,570.9 s. **Links:** branch `task/dig2-harmonic-link-20261016`, draft [PR #435](https://github.com/255BITS/ParticleGAN/pull/435), readout `reports/forge/dig2-20261016/harmonic-link.md` on that branch (local: `/ml2/hypergan/dig2-20261016/harmonic-link/READOUT.md`).

### 7.6 Decisions for Mikkel

1. **ring16 cap value.** Adopt the 6,400-update cap as revision 10 of `discriminator_stability` (proposal in PR #433, not active)? Effect today: BCAP and BISPF unchanged; R1/R2 and GAN v3 +1/152 each; Tier 1 time per candidate rises by about 900 s. Sub-choices: (a) accept R1/R2's 5-check pass, or require a longer sustained window (for example 20 of 384), which R1/R2 would fail and everyone else would still pass; (b) whether K3P and KA2 should drop `network_lr_horizon_cap=1600`, which is a new selected revision needing its long-horizon Tier 2 tasks re-measured.
2. **Clock-audit revision.** Choose among (a) v2 with measured counters, which in effect abolishes the clock rule for any recipe that writes its run length as a constant (not recommended); (b) v2 strict, under which no annealing family passes both; (c) keep v1, under which K3P-const already dual-passes if the 6,400 cap is adopted. The lead recommends (b) or (c); this review agrees, and leans (c) because it needs no gate change.
3. **Wall-clock reporting.** (a) Adopt NOT MEASURED (wall-clock) as an outcome and the contention-aware limit (suggested: 3x ceiling with a per-task solo reference) by default, or keep them opt-in; (b) for truly solo reruns, pause GPU jobs outside the lease (Atlas/E22 fill, llama-server, games), or accept the lease as workflow-only; (c) replace `gpu-lease.sh` with a first-come-first-served ticket queue; (d) approve about 1.3 more GPU-hours to check the fixed contention-aware limit on GPU and rerun the two kinetic word cells; (e) decide whether runtime-mixed cells (torch 2.13 rerun beside torch 2.11 cells) may count toward 6/6.
4. **Publication.** Refresh the published leaderboard from these diagnostic receipts (needs ordinary-view runs or a publication refresh)? Scores would not change for BISPF/BCAP; signed Confidence LR and the BCAP baseline would show 6/6 Tier 1.
5. **Harness.** Land a coordinator opt-out for the post-completion compile (`FORGE_SKIP_COMPLETION_COMPILE` exists only on #429/#431; it held GPU leases idle for over 20 minutes in three leads), or fix the compile cost.
6. **Smaller.** Rerun word smoke/hold for C and Z + chord on a quieter GPU? Keep `Recipe.bispf_ablation` in develop or on PR #435 only? Signed Confidence LR's snapshot refuses to resume its own smoke checkpoint (RNG bindings); its owner should look.

### 7.7 Notes common to round 2

- Round 2 branches start from develop + #430, not from this document's branch, so their diffs do not include this file.
- Pre-existing test failures were seen in several checkouts (`test_forge_current_technique_inventory.py` missing a fixture; 110-118 failures in the board test files; `test_forge_views.py::test_frozen_thresholds...` after #425). Each was confirmed or claimed identical on the base commit; none touches changed code.
- Three leads stopped a drain whose post-completion compile held the GPU lease idle, after grading was complete, and reverted the leaderboard files the partial compile rewrote.

## Sources

- PR bodies #219 to #423 (`gh pr view`).
- `reports/forge/`:
  - leaderboard data: `technique-inventory.{md,json}`, `attempt-coverage.json`, `test-performance.json`
  - findings: `families/*.md`, `EXPERIMENT_MEMORY.md`, `OPTIMIZER_TYPES.md`
  - BCAP studies: `bcap-tier2-search/`, `bcap-tier2-investigation/`, `bispf/README.md`
- Local studies: `/ml2/hypergan/combination-review-20261014/`, `/ml2/hypergan/combination-training-20261015/`, `/ml2/hypergan/test-dispositions-20261015/RECONCILED.md`, `/ml2/hypergan/atlas-*-task*/`.

Not verified for this review:

- No raw receipts were recomputed.
- Leaderboard-row gains and losses are protocol-matched, not causally matched.
- The PR399 history rows' Tier 1 gaps are largely constructor refusals.
- The K3P "about 81/152 if its Tier 2 counted" figure is an estimate.


## Preserved follow-up evidence (integration, 2026-11-08)

- [PR #425: Try a learning schedule driven by training progress rather than step labels.](https://github.com/255BITS/ParticleGAN/blob/1399d3c6c255a8ea72dc392b37b8c3b9d9c7187e/reports/forge/dig-20261015/horizon-clock.md)
- [PR #426: Test whether the prior tether alone explains BISPF mode retention.](https://github.com/255BITS/ParticleGAN/blob/1ded1c70b4a4c945f1f1352afd42281955c373fb/reports/forge/dig-20261015/bispf-ablation.md)
- [PR #427: Test a cell-based objective that trades mode retention against ring density.](https://github.com/255BITS/ParticleGAN/blob/7595f0ae219e7dba09270abc6d75ff49d8d8008c/reports/forge/dig-20261015/cell-quota-probe.md)
- [PR #429: Test whether limiting the prior tether improves the BCAP trade-off.](https://github.com/255BITS/ParticleGAN/blob/474ee2a9729f3a2120a270b60a0e948466880807/reports/forge/dig-20261015/relative-tether.md)
- [PR #431: Check that the BISPF and BCAP baseline results reproduce on a common source.](https://github.com/255BITS/ParticleGAN/blob/62711fe540c44e83da8a6ed3e7a5c64fe9c9bc93/reports/forge/dig2-20261016/bispf-refresh.md)
- [PR #432: Expose a loophole in a proposed clock-free training audit.](https://github.com/255BITS/ParticleGAN/blob/92e4eeff00f4f166e043db6fc9386a0cebc29d0a/reports/forge/dig2-20261016/clock-audit-measured.md)
- [PR #433: Test whether a longer ring-training budget is enough.](https://github.com/255BITS/ParticleGAN/blob/c8b732f87e24e61414dacbaaf42493ffb44a9f03/reports/forge/dig2-20261016/ring16-cap.md)
- [PR #434: Check whether apparent training failures are actually runtime-limit hits.](https://github.com/255BITS/ParticleGAN/blob/ea51ecf0b4f9eb2dc256b68c973b1ac23d57cc32/reports/forge/dig2-20261016/wallclock-reruns.md)
- [PR #435: Find which critic changes help BISPF keep learned modes.](https://github.com/255BITS/ParticleGAN/blob/febb161a15c34901fe3962c6b5068bde4c8934e8/reports/forge/dig2-20261016/harmonic-link.md)
