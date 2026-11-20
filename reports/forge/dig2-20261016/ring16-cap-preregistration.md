# Preregistration: ring16 cap change (dig2 lead `ring16-cap`), 2026-10-16

Written and committed before any training for this lead.

## Question

The user has authorised changing the ring16 cap. If ring16 runs for 6,400 updates instead of 1,600, with each recipe's own step schedule declared over the full 6,400 updates, which selected recipes pass the unchanged ring16 gate? In particular, do BCAP and BISPF still pass over the longer run?

## Cap choice (made before running)

**6,400 updates.** This is the only cap with direct evidence. `ring16-schedule-20261011` shows K3P, KA2 and R1/R2 passing the unchanged gate at 6,400 updates with full-budget schedules. They failed at its 1,600, 3,200 and 4,800 cuts. Those cuts are not equivalent to running with a shorter cap, because the schedule depends on the declared horizon. Even so, no shorter cap has any passing evidence. A longer cap, for example 8,000 updates, might widen R1/R2's thin margin (5 terminal passes, last failure at update 6,317, endpoint min-eigen ratio 0.157 against 0.15). But no evidence supports it, and choosing a cap after seeing R1/R2's margin would be tuning the test to one recipe. Any other cap is left to the user as a decision.

## New task revision

`configs/forge/tasks/ring16_acquisition_6400.json` is a new identity. `ring16_acquisition` and all its receipts are unchanged. The changed fields are steps (1,600 → 6,400), host steps, `original_schedule_horizon` (400 → 6,400), observations (96 → 384) and timeout (300 → 1,200 s). The cadence stays at one check every 16.67 updates, the same as the original task and the 6,400 study. The prior, host, initializer, target, sampling law, scorer, thresholds and the 5-check terminal rule are unchanged. Every family's recipe is unchanged. That includes K3P and KA2's own `network_lr_horizon_cap=1600`, so their G/D learning rate anneals over min(6,400, 1,600) updates while the prior and noise schedules span 6,400.

## Arms

These are the six selected recipe cards, unchanged, on seed 0 with one frozen source commit and runtime (python312 runtime, CUDA RTX A6000): BCAP (`recipe-prior-refactor-bcap-pure-v1`), BISPF (`bispf-v1`), K3P, KA2, R1/R2 and GAN v3 release 0.7 (`recipe-prior-refactor-*-v1`). Each runs two tasks:

- `ring16_acquisition_6400`, the treatment.
- `ring16_acquisition` at 1,600 updates, the matched control on the same source and runtime.

## Gate and definitions

- **Pass (primary, the task's own gate).** At least 5 consecutive *terminal* observations, out of 384, meet every threshold: modes ≥ 16, mass_tv ≤ 0.15, hq ≥ 0.85, component covariance error ≤ 0.85 and min eigen ratio ≥ 0.15. The final live metrics must also pass.
- **Terminal streak.** The number of consecutive passing observations ending at the last one.
- **Robust vs thin.** A pass is **robust** if the terminal streak is ≥ 20 observations, which is about 333 updates and equals 5 checks of a 96-check grid over 6,400 updates. It is **thin** if the streak is 5-19. Every 4th observation of the 384 is exactly the 96-check grid ceil(i·6400/96), so I also report the pass/fail that a 96-check gate would give, from the same run.
- **Late collapse.** A run has late collapse if it reaches a 5-pass streak and later loses it by the end (terminal streak < 5). I report the longest streak, the first sustained step, the last failing step and which gate failed last.

## Decision rule

1. **Cap change is non-destructive** if BCAP and BISPF both PASS `ring16_acquisition_6400`. If either fails, the cap change is **harmful** for that family. I then recommend against adopting 6,400 as a replacement. A view revision would not be included in the PR, and any alternative (for example "pass at either cap") goes to the user.
2. **Cap change helps** if at least 2 of K3P, KA2 and R1/R2 PASS `ring16_acquisition_6400`. GAN v3 is reported but not counted in this rule, because it has no prior 6,400 evidence.
3. **Overall verdict.** *Supported* if 1 and 2 both hold. *Refuted* if 2 fails, meaning the 6,400 study does not reproduce on today's source. *Mixed* if 2 holds but 1 fails.
4. **R1/R2 special attention.** If R1/R2 passes with a terminal streak < 20 (thin), or fails, I report it as fragile. I do not change the cap to rescue it.
5. **Controls.** I expect the 1,600-update controls to repeat the published pattern: BCAP and BISPF pass, the other four fail. If any control differs, I report the discrepancy and keep the treatment verdicts. Any comparison uses only the same-source control.
6. **Conditional diagnostic (not counted).** This applies only if K3P or KA2 fails at 6,400 and its G/D learning rate has been at its 0.01 floor from update 1,600 (from its own `network_lr_horizon_cap`). In that case I may run one labelled diagnostic of that family with `network_lr_horizon_cap=None`, under its own card. It is not a selected-recipe result and does not enter the recount.
7. **Stop rules.** One seed, no seed sweeps, no reruns of a scientific failure. Infrastructure failures may be retried with the reason recorded. The GPU budget is 2.5 GPU-hours. The expected cost is about 15 minutes.

## Recount simulation (CPU only, after the runs)

From `reports/forge/technique-inventory.json` (view revision 9) and `reports/forge/attempt-coverage.json`, I recount each selected family with its measured `ring16_acquisition_6400` verdict in place of `ring16_acquisition`. Two scenarios:

- **(a) Strict.** The clock-free audit stays a Tier 1 requirement in `clockfree_continuous`.
- **(b) Diagnostic credit.** The family's existing same-recipe Tier 2/3 diagnostic attempts count as if their Tier 1 were complete apart from the clock-free audit.

Published scores are not regenerated.

## Addendum, 2026-10-16: triggering conditional diagnostic D1 (written before any D1 training)

The main 12 runs are complete. K3P and KA2 FAIL `ring16_acquisition_6400` with 0 of 384 passing observations. In both, the last failing gate is the min eigen ratio; the covariance error is below 0.85 at the end. Their G/D learning rate reaches its 0.01 floor at update 1,600, because of `network_lr_horizon_cap=1600`. That meets the rule 6 trigger. D1 runs these arms on one new frozen source:

- `k3p-r16cap-net-horizon-full-v1` and `ka2-r16cap-net-horizon-full-v1`: the selected card with `network_lr_horizon_cap=None`, every other field unchanged.
- The unchanged selected K3P and KA2 cards, re-enqueued as same-source matched controls.

Each arm runs only `ring16_acquisition_6400`, with the same gate. D1 is labelled diagnostic, never selected-recipe evidence, and it does not enter the recount or change the main verdict. If a D1 arm passes, the finding is that the recipe's fixed 1,600-update network-anneal cap is what blocks the 6,400 pass. Whether to change any recipe is then a decision for the user. If the D1 arms fail, the cap is not the cause, and the most likely remaining difference from `ring16-schedule-20261011` is the recipe-owned prior refactor. That is not tested here.
