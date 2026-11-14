# Dig 2: ring16 cap at 6,400 updates (lead `ring16-cap`), 2026-10-16

**Verdict under the preregistered rule: refuted.** Raising the ring16 cap from 1,600 to 6,400 updates is safe for the two families that already pass: BCAP and BISPF still pass, with long sustained runs and no late collapse. It does not admit the selected K3P and KA2 recipes. Both fail on local spread, with 0 of 384 passing checks. R1/R2 passes with the bare minimum of 5 terminal checks. GAN v3 release 0.7 passes for the first time. The rule needed 2 of K3P, KA2 and R1/R2 to pass, and only R1/R2 did.

**The cause is identified (diagnostic D1).** K3P and KA2 carry their own `network_lr_horizon_cap=1600`. Under it, their generator and critic learning rate reaches its 0.01 floor at update 1,600 and stays there for the remaining 4,800 updates. With that cap removed and nothing else changed, both pass. They reproduce the 2026-10-11 6,400-update study exactly: K3P 66 terminal passes with its last failure at update 5,300, KA2 47 with its last failure at 5,617. R1/R2 reproduces that study exactly with the selected recipe: 5 passes, last failure at 6,317. So the earlier "full schedule" passes required the network anneal to span the whole run. The cap change alone does not deliver them for K3P and KA2. A recipe change is needed too, which is a decision for the user.

## Terms

- **ring16.** Sixteen equal 2-D Gaussian clusters on a circle of radius 3, each with sigma 0.1, learned from scratch.
- **Gate (unchanged).** At every check, all of these must hold: modes ≥ 16, mass_tv ≤ 0.15, hq ≥ 0.85, component covariance error ≤ 0.85 and component min eigen ratio ≥ 0.15. The *min eigen ratio* is the smallest ratio, over clusters, of the generated cluster's narrowest spread to its target spread. A low value means some cluster has collapsed into a thin streak.
- **Check cadence.** The original task checks 96 times in 1,600 updates. The new task checks 384 times in 6,400 updates. Both check once every 16.67 updates.
- **Pass.** The last 5 checks in a row pass, and so do the final live metrics.
- **Terminal streak.** The number of consecutive passing checks that end at the final check.
- **Robust / thin.** I set these before running. A pass is *robust* if its terminal streak is at least 20 checks, about 333 updates. That equals 5 checks of a 96-check grid over 6,400 updates. A pass is *thin* if the streak is 5-19.
- **grid96.** The verdict that a 96-check grid would give, read from the same run. Every 4th of the 384 checkpoints is exactly that grid.
- **Late collapse.** A run that reaches a 5-check streak and has lost it by the end.
- **Schedule horizon.** The step count a recipe's step schedules (cosine learning rate, noise anneals) are declared over. The old task gives 400. The new task gives 6,400, the full cap. K3P and KA2 then cap their G/D learning-rate anneal at min(horizon, `network_lr_horizon_cap` = 1600).
- **Clock-free audit.** The Tier 1 measurement that the trainer state does not depend on the step label, the horizon, the evaluation cadence or a restart. Every annealed family fails it. This lead does not change it.

## Hypothesis and preregistered rule

The rule was committed before any training: `24a014d27`, `reports/forge/dig2-20261016/ring16-cap-preregistration.md`. The D1 addendum was committed before D1 ran, in `817699cbb`.

1. The cap change is non-destructive if BCAP and BISPF both pass at 6,400. **Result: holds.**
2. The cap change helps if at least 2 of K3P, KA2 and R1/R2 (selected recipes) pass at 6,400. **Result: fails.** Only R1/R2 passes.
3. Supported if 1 and 2 both hold; refuted if 2 fails. **Result: refuted.**
4. R1/R2 is reported as fragile if its streak is below 20 checks. **Result: fragile**, with a streak of 5.
5. The 1,600-update controls should repeat the published pattern. **Result: they do.** BCAP and BISPF pass; K3P, KA2, R1/R2 and GAN v3 fail.
6. Conditional diagnostic D1 was triggered: K3P and KA2 failed, and their G/D learning rate sat at its floor from update 1,600. **Result:** both pass with `network_lr_horizon_cap=None`. D1 is not counted.

**Cap choice: 6,400 updates.** It is the only cap with direct passing evidence. No shorter cap has any: the earlier study's 1,600, 3,200 and 4,800 cuts all failed, and #425 found no K3P schedule that passes within 1,600. A longer cap, for example 8,000, might widen R1/R2's margin. I did not test that, because picking the cap from R1/R2's result would tune the test to one recipe. It is left as a decision.

## Implementation

- **`configs/forge/tasks/ring16_acquisition_6400.json`** is a new task identity. It records `task_revision.revision_of = ring16_acquisition` and that task's sha256. The changed fields are steps and host steps (1,600 → 6,400), `original_schedule_horizon` (400 → 6,400), observations (96 → 384) and timeout (300 → 1,200 s). The prior, host, initializer, target, sampling law, scorer, thresholds and the 5-check suffix rule are byte-identical. `ring16_acquisition` and every receipt graded under it are untouched. `tests/test_ring16_cap_revision.py` pins all of this.
- **`configs/forge/view-proposals/discriminator_stability-v10-ring16-6400.json`** is a proposed revision 10 of the live `discriminator_stability` view, which swaps the required Tier 1 ring16 task. **It is not activated.** No loader reads that directory, and published scores were not regenerated. I trial-activated it, ran `validate` and the board tests, then restored revision 9:
  - `validate` passes.
  - Only 2 tests start failing, both of which pin the revision-9 task list or budget: `test_forge_views.py::test_current_inventory_has_complete_quality_and_distinct_claim_views` and `test_default_plan_and_report_include_acquisition_smoke_without_new_views`.
  - The same 7 test files already show 110 failures on the unmodified base, with or without this lead's files.
- **Diagnostic scope.** The diagnostic views are `dig2-ring16-cap-diagnostic-v1` (both ring16 tasks) and `dig2-ring16-cap-d1-diagnostic-v1`. There is one study per selected family plus four D1 studies. The D1 cards are `k3p-r16cap-net-horizon-full-v1` and `ka2-r16cap-net-horizon-full-v1`: the selected card with `network_lr_horizon_cap: null` and nothing else changed.

## Exact commands

```sh
cd <worktree on task/dig2-ring16-cap-20261016>
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python; export OMP_NUM_THREADS=1
Q=/ml2/hypergan/dig2-20261016/ring16-cap/queue
PYTHONPATH=. $PY reports/forge/dig2-20261016/ring16-cap-make.py        # task revision, view, 6 studies
$PY -m experiments.forge --root . validate
for C in recipe-prior-refactor-bcap-pure-v1 bispf-v1 recipe-prior-refactor-k3p-v1 recipe-prior-refactor-ka2-v1 \
         recipe-prior-refactor-r1r2-v1 recipe-prior-refactor-release07-gan-v3-v1; do
  $PY -m experiments.forge --root . --queue-root $Q enqueue $C --study dig2-ring16-cap-$C-study; done
# drain through the workflow GPU lease (one worker); see caveat on the compile hook
gpu-lease.sh bash -c '$PY -m experiments.forge --root . --queue-root $Q drain --gpus $LEASED_GPU --workers-per-gpu 1 --campaign dig2-ring16-cap-v1'
#   (resumed with reports/forge/dig2-20261016/ring16-cap-drain-nocompile.py: same Queue/drain call with on_completion=None)
PYTHONPATH=. $PY reports/forge/dig2-20261016/ring16-cap-make-d1.py     # D1 cards, view, 4 studies
for C in k3p-r16cap-net-horizon-full-v1 recipe-prior-refactor-k3p-v1 ka2-r16cap-net-horizon-full-v1 recipe-prior-refactor-ka2-v1; do
  $PY -m experiments.forge --root . --queue-root $Q enqueue $C --study dig2-ring16-cap-d1-$C-study; done
gpu-lease.sh ... drain_nocompile.py $Q $LEASED_GPU dig2-ring16-cap-d1-v1
python3 reports/forge/dig2-20261016/ring16-cap-analyze.py $Q dig2-ring16-cap-v1 results-main.json
python3 reports/forge/dig2-20261016/ring16-cap-analyze.py $Q dig2-ring16-cap-d1-v1 results-d1.json
python3 reports/forge/dig2-20261016/ring16-cap-recount.py reports/forge/dig2-20261016/ring16-cap-results.json
python3 reports/forge/dig2-20261016/ring16-cap-make-view-proposal.py
```

All of the following are tail-able:

- `/ml2/hypergan/dig2-20261016/ring16-cap/queue/events.jsonl`
- `.../queue/<campaign>/<attempt>/run.log`, one JSON observation per check
- `/ml2/hypergan/dig2-20261016/ring16-cap/logs/`

## Results

All runs use seed 0 and one frozen source snapshot, `6681974e…` (commit `24a014d27`; the D1 declarations in `817699cbb` did not change the snapshot). Hardware is an RTX A6000 with the python312 runtime.

### Per-test table

Final metrics are covariance error / min eigen ratio / hq / mass_tv. "Min eig, last 25%" is the lowest min eigen ratio over the final quarter of the checks.

| Recipe | Task | Verdict | Passing checks | Terminal streak | Longest streak | First 5-streak at | Last failure (gate) | Final cov / eig / hq / tv | Min eig, last 25% | grid96 (streak) | Train s |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: |
| BCAP | 6,400 | **PASS, robust** | 314/384 | 314 | 314 | 1,250 | 1,167 cov | 0.488 / 0.282 / 0.985 / 0.038 | 0.168 | PASS (79) | 108.4 |
| BCAP | 1,600 control | PASS | 26/96 | 26 | 26 | 1,250 | 1,167 cov | 0.497 / 0.242 / 0.961 / 0.065 | 0.164 | n/a | 30.4 |
| BISPF | 6,400 | **PASS, robust** | 287/384 | 151 | 151 | 1,534 | 3,884 cov | 0.409 / 0.179 / 0.983 / 0.070 | 0.169 | PASS (38) | 233.2 |
| BISPF | 1,600 control | PASS | 9/96 | 9 | 9 | 1,534 | 1,450 cov | 0.616 / 0.277 / 0.936 / 0.092 | 0.233 | n/a | 59.0 |
| K3P (selected) | 6,400 | **FAIL** | 0/384 | 0 | 0 | — | 6,400 min-eig | 0.421 / 0.077 / 0.985 / 0.051 | 0.059 | FAIL (0) | 83.6 |
| K3P (selected) | 1,600 control | FAIL | 0/96 | 0 | 0 | — | 1,600 hq, cov | 2.704 / 0.442 / 0.779 / 0.063 | 0.328 | n/a | 22.5 |
| KA2 (selected) | 6,400 | **FAIL** | 0/384 | 0 | 0 | — | 6,400 min-eig | 0.666 / 0.084 / 0.964 / 0.048 | 0.072 | FAIL (0) | 95.8 |
| KA2 (selected) | 1,600 control | FAIL | 0/96 | 0 | 0 | — | 1,600 hq, cov | 2.870 / 0.486 / 0.764 / 0.059 | 0.357 | n/a | 20.4 |
| R1/R2 | 6,400 | **PASS, thin** | 69/384 | **5** | 14 | 3,834 | 6,317 min-eig | 0.423 / 0.157 / 0.978 / 0.050 | 0.119 | **FAIL (4)** | 58.0 |
| R1/R2 | 1,600 control | FAIL | 0/96 | 0 | 0 | — | 1,600 hq, cov | 2.066 / 0.223 / 0.844 / 0.052 | 0.154 | n/a | 14.6 |
| GAN v3 rel. 0.7 | 6,400 | **PASS, robust** | 59/384 | 43 | 43 | 5,534 | 5,684 min-eig | 0.562 / 0.442 / 0.957 / 0.039 | 0.035 | PASS (15) | 62.1 |
| GAN v3 rel. 0.7 | 1,600 control | FAIL | 0/96 | 0 | 0 | — | 1,600 hq, cov | 3.483 / 0.204 / 0.813 / 0.046 | 0.129 | n/a | 16.3 |
| D1: K3P, net anneal over 6,400 | 6,400 | PASS (diagnostic) | 86/384 | 66 | 66 | 5,134 | 5,300 cov | 0.567 / 0.413 / 0.963 / 0.041 | 0.252 | PASS (21) | 69.9 |
| D1: KA2, net anneal over 6,400 | 6,400 | PASS (diagnostic) | 66/384 | 47 | 47 | 3,184 | 5,617 min-eig | 0.563 / 0.213 / 0.975 / 0.055 | 0.047 | PASS (13) | 79.1 |

The D1 matched controls are the unchanged K3P and KA2 cards, re-enqueued in the D1 campaign. Forge reused the same-source 6,400 attempts above (`49b20539`, `ce7d5dbd`), because the source snapshot and compatibility key are identical.

### Sustained-window reading

- **BCAP shows no late collapse.** Its first 1,600 updates are the same trajectory as the 1,600-update control, with the same last failure at update 1,167. That is expected, because constant rates do not read the horizon. From there it passes every one of the next 314 checks, through update 6,400. Its lowest min eigen ratio after update 3,200 is 0.168. That is the closest it comes to failing, a 0.018 margin over the 0.15 limit.
- **BISPF dips once, then passes for the rest of the run.** It passes from update 1,534 to 3,567. It then fails covariance on 10 checks between updates 3,584 and 3,884, with values up to 1.10 against 0.85. After that it passes the remaining 151 checks. The longer run exposes a mid-run covariance excursion that the 1,600-update task cannot see. Its spread margin at the end is thin: min eigen ratio 0.169-0.18 against 0.15.
- **The selected K3P and KA2 lose spread steadily.** K3P's covariance error falls below 0.85 by about update 4,000, and hq keeps rising to 0.985. Meanwhile its min eigen ratio drifts down: 0.225 at 1,600, 0.138 at 2,400, 0.077 at 6,400. The G/D learning rate has been at its 0.01 floor since update 1,600. With the anneal spread over 6,400 updates (D1), the min eigen ratio ends at 0.41 (K3P) and 0.21 (KA2).
- **R1/R2 is as fragile as in the 6,400 study.** Its run is identical to that study, with the same last failure at update 6,317. It fails min eigen ratio at 0.119 to 0.13 within the final 20 checks, and it fails 92 checks after update 3,800, including a brief mode loss around 4,417-4,484. A 96-check grid would fail it. Its pass depends on where the last checks happen to fall.
- **GAN v3 recovers late.** Its min eigen ratio collapses between updates 5,017 and 5,684, reaching a low of 0.035. It then recovers to 0.44 and passes the last 43 checks.

### Recount simulation (CPU only; nothing was published)

`reports/forge/dig2-20261016/ring16-cap-recount.py` reads only the committed `attempt-coverage.json` and the current view files. Its `published` scenario reproduces the board exactly: BCAP 71/152, BISPF 77/152, K3P/KA2/R1R2/GAN v3 20/152. A count is passes over view requirements; a test counts once in each of the six ordinary views that require it.

| Family | Published | Strict (ring16 at 6,400; clock audit still required) | Diagnostic credit (Tier 2 diagnostics count when only the clock audit is missing) | Distinct Tier 2 tests in those diagnostics |
| --- | ---: | ---: | ---: | --- |
| BCAP | 71/152 | 71/152 | 71/152 | 9/21 (already counted) |
| BISPF | 77/152 | 77/152 | 77/152 | 10/21 (already counted) |
| GAN v3 release 0.7 | 20/152 | 21/152 | **75/152** | 9/21 (incl. img_bars4, img_intensity2) |
| R1/R2 | 20/152 | 21/152 | **75/152** | 9/21 (incl. img_bars4, img_blobs4, img_intensity2, vector_anisotropic) |
| K3P (selected) | 20/152 | 20/152 | 20/152 | 11/21, not reachable: ring16 still fails |
| KA2 (selected) | 20/152 | 20/152 | 20/152 | 10/21, not reachable: ring16 still fails |

- **Who completes Tier 1 apart from the clock-free audit:** GAN v3 release 0.7 and R1/R2, at 21 of 22 Tier 1 requirements. BCAP and BISPF were already complete at 22/22. The selected K3P and KA2 do not.
- **If R1/R2's 5-check pass were graded on a 96-check grid,** it would fail, and R1/R2 would stay at 20/152.
- **D1 recipes (not selected):** if their existing Tier 2 diagnostics still held, K3P would reach 82/152 and KA2 76/152 under diagnostic credit. That is an upper bound only. Removing `network_lr_horizon_cap` changes every task whose schedule horizon exceeds 1,600 updates, for example native 7k and five-word 20k. Those Tier 2 results would have to be re-measured.
- Under the strict rule, no annealed family's Tier 2 counts. The clock-free audit still blocks all four.

### GPU seconds

- **Training:** 804 s across the 12 main runs (adapter seconds) plus 149 s for D1, so 953 s, about 0.26 GPU-hours. Queue-charged seconds were 1,043.
- **Lease wall time:** about 38 minutes. The first 21 minutes held GPU 0 mostly idle while the coordinator ran the post-completion `compile_memory` (see caveats).
- **Budget:** 2.5 GPU-hours. Well under.

## Conclusion

1. **6,400 is a safe cap for the families that already pass.** BCAP passes robustly. BISPF passes robustly, after one mid-run covariance dip. Neither collapses late.
2. **The cap alone admits only R1/R2 (thin) and GAN v3 (robust) among the annealed families.** The selected K3P and KA2 fail. Their own `network_lr_horizon_cap=1600` holds the G/D learning rate at its 0.01 floor for the last 4,800 updates, and local spread decays.
3. **The 2026-10-11 6,400-update study is reproduced exactly** by R1/R2 as selected and by K3P/KA2 with the network anneal over the full cap (D1). This identifies what "full schedule" meant in that study.
4. **The clock-free audit remains the binding Tier 1 gate** for every annealed family. Even with ring16 solved, Tier 2 counts only if the user relaxes it.

## Caveats

- One seed, as the rules require. R1/R2's pass rests on 5 checks. BISPF's and BCAP's spread margins near the end are about 0.02 above the 0.15 limit.
- BCAP ran on GPU 0 (first drain). Everything else ran on GPU 1 (second drain) or GPU 0 (D1). All are RTX A6000s on one frozen snapshot and runtime. The BCAP control and treatment ran on the same device and agree with each other step for step.
- **Harness.** The CLI `drain` calls `compile_memory(root)` after each completed request. On this checkout (after the #430 crash fix), that took over 20 minutes per request and held the GPU lease idle. It also partly rewrote three `reports/forge/leaderboards/*.md` files in the worktree; I reverted them, so nothing was committed. I stopped that drain (SIGTERM) after BCAP's two attempts were recorded and resumed with `reports/forge/dig2-20261016/ring16-cap-drain-nocompile.py`. That is the identical `Queue(...)` and `drain(...)` call with `on_completion=None`. Execution, grading and receipts are unchanged. **Fixing the slow compile hook is a separate follow-up.**
- The 384-check task keeps the original cadence, so its 5-check terminal window spans about 83 updates. That is the same absolute window as before, but a smaller fraction of a 4× longer run. The grid96 column shows the 96-check alternative.
- The recount uses `attempt-coverage.json` statuses. Some of them are graded under task contracts marked CHANGED since they ran; the board keeps those earlier verdicts, and so does this recount. The GAN v3 and R1/R2 Tier 2 statuses come from diagnostic attempts recorded on an earlier source digest than this lead's runs.
- The proposed view revision 10 is not active. Activating it is a policy change for the user. It also needs the two pinned tests updated and a new campaign budget: Tier 1 reservation rises by 900 s per candidate.
