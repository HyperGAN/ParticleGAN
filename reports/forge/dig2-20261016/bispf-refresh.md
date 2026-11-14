# BISPF refresh: do BISPF's lead and its other verdicts reproduce on the current source?

<!-- superseded-banner: generated -->
> Historical 2026-10-16 reproducibility study. “Current source” and counted totals below
> refer to that date, not today. See the [current technique inventory](../technique-inventory.md).
> [Original study, views, campaign and coordinator code](https://github.com/255BITS/ParticleGAN/tree/62711fe540c44e83da8a6ed3e7a5c64fe9c9bc93)
> remain pinned for reproduction. This integration carries the report only; it does not register old runnable studies,
> bypass current publication, rerun training or change qualification.

Status: DONE. Preregistered verdict: **SUPPORTED** (34 of 34 cells reproduced). The rule was committed
(ec0711d1) and pushed before any training; results are below the rule.

## Why

BISPF (`bispf-v1`) leads the Forge board (77/152 today, 77/81 after the zero-pass audit). Its record is
stale: it was measured on source digest `fafd6615...` with a different Python/numpy runtime, and several of
its task contracts have changed since ("Test definition changed since this run" on the board). The selected
BCAP recipe (`recipe-prior-refactor-bcap-pure-v1`, configuration family `bcap-dualnorm`, BISPF's parent) is
equally stale (source `a5eafc73...`; its stale label was fixed in #428). BISPF's only extra pass over BCAP is
`mode_hold`.

Words used here:

- **Counted test**: a test that still counts toward the leaderboard after the zero-pass audit
  (`/ml2/hypergan/test-dispositions-20261015/RECONCILED.md`). Held tests (four vector tests, three image
  tests, grid100/rotated100/staggered100, gaussian1d_stability, the Tier 3 tests) are skipped.
- **Verdict**: the Forge grader's PASS/FAIL, rebuilt from the run's curve. Sustained gates need at least 5
  passing checks in a row at the end of the run ("terminal suffix"); the word and Gaussian smoke tests use an
  earliest-confirmed gate (a passing check confirmed by an independent redraw, with all updates completed).
  Endpoint values are never used for verdicts.
- **Tier rule** (runner policy `complete_current_tier`): every runnable job of a tier finishes even if one
  fails; any non-PASS in the tier stops spending on the next tier.
- **Frozen source**: the source digest that `enqueue` freezes; both recipes run on the same one.

## Hypothesis

H: on the current source (develop + the compile_memory crash fix, #430), seed 0, CUDA RTX A6000, every
recorded verdict of BISPF and of the selected BCAP recipe on the 17 counted Tier 1/Tier 2 tests reproduces,
in particular BISPF `mode_hold` PASS vs BCAP `mode_hold` FAIL (BISPF's lead).

Recorded verdicts (from `reports/forge/technique-inventory.json`, selected cohorts):

| Tier | Test | BISPF recorded | BCAP recorded |
| --- | --- | --- | --- |
| 1 | clockfree_audit_measurement_v1 | PASS | PASS |
| 1 | two_pole | PASS | PASS |
| 1 | unused_token_hold | PASS | PASS |
| 1 | ae_gan_hold | PASS | PASS |
| 1 | gaussian1d_smoke | PASS | PASS |
| 1 | ring16_acquisition | PASS | PASS |
| 1 | five_word_joint_smoke | PASS | PASS |
| 2 | five_word_joint_hold | PASS | PASS |
| 2 | unipolar | PASS | PASS |
| 2 | trajectory | PASS | PASS |
| 2 | residual_student | PASS | PASS |
| 2 | cover_leftover | PASS | PASS |
| 2 | mid_scale_identity | PASS | PASS |
| 2 | vector_spiral | PASS | PASS |
| 2 | vector_two_broad | PASS | PASS |
| 2 | img_stripes2 | PASS | PASS |
| 2 | mode_hold | PASS | **FAIL** |

## Design

- Two candidates, unchanged: `bispf-v1` and `recipe-prior-refactor-bcap-pure-v1`. No code, default, task,
  gate or budget changes; only opt-in research-diagnostic views, studies and one campaign are added.
- Two research-diagnostic views (all tasks `diagnostic`, so these runs never fill ordinary cells and the
  leaderboard is not regenerated):
  - `dig2-bispf-refresh-tier1-v1`: the 7 counted Tier 1 tests, cheapest first, plus
    `five_word_joint_hold`. The word hold is a Tier 2 test that continues the `five_word_joint_smoke`
    checkpoint, so it has to run in the same request as the smoke test. Its verdict is admissible only if the
    recipe passes all 7 Tier 1 tests; otherwise it is reported as NOT ADMISSIBLE.
  - `dig2-bispf-refresh-tier2-v1`: the other 9 counted Tier 2 tests, cheapest first.
- Ordinary path: `forge enqueue <candidate> --study <study>`, then `forge drain --gpus $LEASED_GPU
  --workers-per-gpu 1 --request <id>` under the workflow GPU lease, one request per lease.
- Both recipes' Tier 1 requests are enqueued from the same commit, so they share one frozen source digest;
  Tier 2 requests are enqueued from the same commit as well. If the frozen digests differ between the two
  recipes for the same tier, the comparison is flagged as unmatched.

## Preregistered decision rule

1. Tier rule: a recipe's Tier 2 request is enqueued only if all 7 counted Tier 1 tests PASS on this
   source. Inside a tier, all jobs run to completion.
2. Per cell (recipe x test): REPRODUCED if the current verdict equals the recorded verdict; FLIPPED if it is
   the opposite PASS/FAIL; UNDECIDED if INCOMPLETE, TIMEOUT, INVALID, BLOCKED or NOT_RUN (reported with the
   exact error; no reruns, no other seeds).
3. BISPF's lead is REPRODUCED iff BISPF `mode_hold` = PASS and BCAP `mode_hold` = FAIL on this source. It is
   NOT REPRODUCED if either cell flips (both pass, or both fail, or reversed).
4. Overall verdict on H:
   - SUPPORTED: all 34 cells REPRODUCED.
   - REFUTED: any decided cell FLIPPED (the readout names each flipped cell and says separately whether the
     lead itself reproduced).
   - INCONCLUSIVE: no cell flipped but at least one cell is UNDECIDED.
5. Budget: at most 2 GPU-hours of attempt wall time in total. If the running total would exceed it, remaining
   requests are not launched and their cells are NOT_RUN (UNDECIDED).
6. Margins are reported for every sustained gate (passing checks, terminal suffix vs required), because a
   one-check margin (like BCAP's earlier `mode_hold` miss by one check) means a small perturbation can move
   the verdict. Margins do not change verdicts.

Prior context (motivation only, not evidence for this rule): the 2026-10-15 BISPF ablation reran `mode_hold`
for both recipes on frozen digest `5ea30969...` and got BISPF PASS (12 terminal passes) and BCAP FAIL (3
terminal passes, one check short, a single hq dip to 0.869).

## Short answer

Yes. On the current source, BISPF's lead reproduces (BISPF `mode_hold` PASS with 12 terminal passing
checks; BCAP `mode_hold` FAIL with 3, two short of the 5 needed), and **every one of the 34 verdicts**
(17 counted tests x 2 recipes) equals its recorded verdict. More than that: every sustained-gate margin
(passing checks, terminal suffix, first confirmed step) is identical to the stale record. The older
source `fafd6615`/`a5eafc73` and the older Python/numpy runtime did not change any result on these tests.
Implied counted score (not published): BISPF 77/81, BCAP 71/81, unchanged.

## Results

All four requests froze source digest `6681974eea685623b777f77aeaba98b16836b16ee68a845f57a386d49f0f4900`
(branch commit ec0711d1 = develop + #430 + this dig's config files), seed 0, CUDA RTX A6000,
`--workers-per-gpu 1`, each through the workflow GPU lease. Requests: BISPF Tier 1 `b5dd73c453e8db18b958ba37`,
BCAP Tier 1 `f4993b400cc19a7c0960193f`, BISPF Tier 2 `f90c6b37991f79f94453e28b`, BCAP Tier 2
`15e9ac401d976e5024fd1962`. Both recipes passed all 7 counted Tier 1 tests, so both Tier 2 requests ran and
`five_word_joint_hold` is admissible for both.

"Checks" = passing checks / total checks; "suffix" = passing checks in a row at the end (needs 5). "Same as
record" compares checks, suffix and verdict with the stale record in `reports/forge/technique-inventory.json`.
The word, Gaussian and clockfree tests have no sustained suffix (earliest-confirmed or parity gates).

| Tier | Test | BISPF verdict | BISPF checks / suffix | BCAP verdict | BCAP checks / suffix | Same as record | BISPF s | BCAP s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 1 | clockfree_audit_measurement_v1 | PASS | parity 4/4 | PASS | parity 4/4 | yes / yes | 7.5 | 7.3 |
| 1 | two_pole | PASS | 17/24, 17 | PASS | 17/24, 17 | yes / yes | 7.6 | 6.9 |
| 1 | unused_token_hold | PASS | 16/24, 12 | PASS | 14/24, **5** | yes / yes | 9.4 | 8.4 |
| 1 | ae_gan_hold | PASS | 22/24, 22 | PASS | 22/24, 22 | yes / yes | 11.6 | 9.8 |
| 1 | gaussian1d_smoke | PASS | confirmed | PASS | confirmed | yes / yes | 24.0 | 18.4 |
| 1 | ring16_acquisition | PASS | 9/96, 9 | PASS | 26/96, 26 | yes / yes | 40.8 | 33.0 |
| 1 | five_word_joint_smoke | PASS | confirmed, 20,001 updates | PASS | confirmed, 20,001 updates | yes / yes | 825.9 | 795.5 |
| 2 | five_word_joint_hold | PASS | every check in window | PASS | every check in window | yes / yes | 202.2 | 175.8 |
| 2 | unipolar | PASS | 20/24, 18 | PASS | 20/24, 18 | yes / yes | 15.7 | 13.4 |
| 2 | trajectory | PASS | 20/24, 20 | PASS | 19/24, 19 | yes / yes | 16.7 | 14.5 |
| 2 | residual_student | PASS | 20/24, 20 | PASS | 20/24, 20 | yes / yes | 16.9 | 15.1 |
| 2 | cover_leftover | PASS | 22/24, 22 | PASS | 22/24, 22 | yes / yes | 30.5 | 26.3 |
| 2 | mid_scale_identity | PASS | 21/24, 20 | PASS | 21/24, 20 | yes / yes | 38.4 | 32.3 |
| 2 | vector_spiral | PASS | 23/24, 23 | PASS | 23/24, 23 | yes / yes | 37.6 | 46.8 |
| 2 | vector_two_broad | PASS | 21/24, 14 | PASS | 22/24, 22 | yes / yes | 30.3 | 45.8 |
| 2 | img_stripes2 | PASS | 12/24, 12 | PASS | 12/24, 12 | yes / yes | 29.2 | 39.6 |
| 2 | **mode_hold** | **PASS** | 13/24, 12 | **FAIL** | 10/24, **3** | yes / yes | 46.4 | 52.9 |

`mode_hold` detail: a check passes when `modes >= 8` (all 8 target modes have high-quality samples) and
`hq >= 0.9` (fraction of samples near some mode). BCAP's endpoint (step 1200: modes 8, hq 0.989) would pass;
its sustained gate does not, because of an hq dip below 0.9 late in the run. This is the same curve, to the
last digit (hq 0.9892578125, effective modes 7.15797), as the 2026-10-15 ablation's control on digest
`5ea30969`, so the result is deterministic across these source revisions.

Rule outcome: precondition-free rule; 34/34 cells REPRODUCED, 0 FLIPPED, 0 UNDECIDED -> **SUPPORTED**.
BISPF's lead is REPRODUCED (rule 3).

GPU seconds (attempt wall time): BISPF 1,390.7 s, BCAP 1,341.9 s, total **2,732.6 s = 0.76 GPU-hours** of
the 2-hour allowance. 70% of it is the two five-word tests.

## Margins worth knowing

- BCAP `unused_token_hold` passes with a suffix of exactly 5 (the minimum). One more failing check at the end
  would flip it. BISPF has 12 there.
- BCAP `mode_hold` misses by 2 checks of suffix (3 vs 5). BISPF passes with 12. The lead is real on this
  source, but the BCAP side is near the line.
- BISPF `vector_two_broad` (suffix 14) and `ring16_acquisition` (suffix 9 of 96) have the thinnest BISPF
  margins; both are comfortably above 5.

## Exact commands

Python: `/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python` (as `$PY`),
`OMP_NUM_THREADS=1`, run from the worktree root. `LEASE=.../scratchpad/dig/gpu-lease.sh`;
`R=/ml2/hypergan/dig2-20261016/bispf-refresh`.

```sh
$PY -m experiments.forge validate
$PY -m experiments.forge plan bispf-v1 --study dig2-bispf-refresh-bispf-tier1-v1   # no preflight blockers
# enqueue (no training); each request in its own queue root so the two recipes drain in parallel
$PY -m experiments.forge --queue-root $R/queue        enqueue bispf-v1 --study dig2-bispf-refresh-bispf-tier1-v1
$PY -m experiments.forge --queue-root $R/queue-bcap   enqueue recipe-prior-refactor-bcap-pure-v1 --study dig2-bispf-refresh-bcap-tier1-v1
$PY -m experiments.forge --queue-root $R/queue-t2     enqueue bispf-v1 --study dig2-bispf-refresh-bispf-tier2-v1
$PY -m experiments.forge --queue-root $R/queue-bcap-t2 enqueue recipe-prior-refactor-bcap-pure-v1 --study dig2-bispf-refresh-bcap-tier2-v1
# drain one request per lease (Tier 2 drained only after that recipe's Tier 1 all passed)
$LEASE drain.sh <queue-root> <request-id>
#   = FORGE_SKIP_COMPLETION_COMPILE=1 $PY -m experiments.forge --queue-root <root> drain \
#       --gpus $LEASED_GPU --workers-per-gpu 1 --campaign dig2-bispf-refresh-v1 --request <request-id>
tail -F $R/<queue-root>/events.jsonl    # live per-check metrics
```

Evidence (graded-result.json, result.json, run.log with live per-check metrics, request receipts):
`$R/{queue,queue-bcap,queue-t2,queue-bcap-t2}/dig2-bispf-refresh-v1/<attempt>/`. Drain logs: `$R/drain-*.log`.

## Caveats

- Seed 0 only, one attempt per cell, by design. Identical margins to the stale record show the runs are
  deterministic on this hardware; they say nothing about other seeds.
- `five_word_joint_hold` was run inside the Tier 1 request (it continues the smoke checkpoint). Both recipes
  passed all of Tier 1, so the preregistered admissibility condition holds.
- The two Tier 2 requests were enqueued (not drained) before BCAP's Tier 1 had finished, so all four share one
  frozen digest. BCAP's Tier 2 was drained only after its Tier 1 passed, as the tier rule requires.
- Post-run compile: the two Tier 1 drains ran the coordinator's full post-completion compile, which took more
  than 10 CPU-minutes per drain while holding a GPU lease and rewrote two tracked leaderboards
  (`reports/forge/leaderboards/{adaptation,clockfree_continuous}.md`). I stopped each coordinator after all
  its attempts were recorded and graded (exit 143) and reverted the leaderboard rewrites; nothing was
  published. For the Tier 2 drains I added the same opt-in `FORGE_SKIP_COMPLETION_COMPILE=1` used by the
  relative-tether dig (commit 0de254e0; coordinator only, default unchanged; workers run from the frozen
  snapshot, so no frozen request is affected). The ordinary boards and memory were not recompiled by these
  runs.
- These are research-diagnostic views (all tasks `diagnostic`), so the results do not fill ordinary
  leaderboard cells. To refresh the published board, the same candidates would need ordinary-view runs (or a
  publication refresh that admits these receipts); not done here, as instructed.
- At the original study commit, two research-diagnostic views, four studies and one campaign were added.
  These declarations and the coordinator shortcut remain at the immutable source linked above.
  This documentation-only integration changes no candidate, task, gate, default or budget.
