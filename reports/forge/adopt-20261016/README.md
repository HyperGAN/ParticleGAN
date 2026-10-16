# Adopt 2026-10-16: ring16 at 6,400 updates, K3P/KA2 without the LR horizon cap, longer wall-clock limits

The user decided this after dig2 round 2. This page records what changed and the measurements that are published
here. It replaces drafts #433 (ring16 cap) and #434 (wall-clock reruns).

Every measurement here ran on frozen source `1e02f185` (snapshot digest `862b24b1`), seed 0, through the normal
Forge path: plan, enqueue, then `drain --no-publish` under a GPU lease, one job per GPU. They are **source-bound
diagnostic rows**. They do not change any selected recipe's recorded qualification or family pin (see Caveats).

## What changed

1. **The counted ring16 is `ring16_acquisition_6400`.** It runs 6,400 updates with each recipe's schedule
   declared over the full 6,400, and 384 checks at the same 16.67-update cadence. The gate and the 5-check terminal
   sustained rule are the same as the 1,600-update `ring16_acquisition`. That old task and every receipt graded
   under it keep their identity. In `discriminator_stability` this is **revision 12**: develop's revision 11
   (test dispositions: 11 held Tier 2 tests, and ring_hold removed from the count as a phase of ring_extension)
   with only the ring16 question swapped. Revision 11 is archived at
   `configs/forge/view-history/discriminator_stability-v11.json`. The measurements below ran under the unmerged
   adopt-branch draft numbered 10, which is revision 9 with the same swap and no holds. That draft is kept as
   provenance in [`measured-view-draft10.json`](measured-view-draft10.json). It is not a view-history revision,
   because develop skipped the number 10 to avoid an older draft.
2. **K3P and KA2 drop `network_lr_horizon_cap`.** The new cards are `recipe-prior-refactor-k3p-v2` and
   `recipe-prior-refactor-ka2-v2`. Each is identical to v1 except `network_lr_horizon_cap: null`, and the v1 cards
   are unchanged. `configs/forge/selections/family-revisions-20261016.json` declares them for `forge inventory`.
   Without the cap, the G/D cosine anneal spans each task's own horizon, for example 20,000 updates on word smoke.
3. **Wall-clock limits end tests on their update budget** (`experiments/forge/wallclock.py`). A wall-clock hit is
   reported as **NOT MEASURED (wall-clock)**, never FAIL. The raised operational limits are: word smoke 1,800 s
   (declared 900 s), word hold 600 s (300 s), the toy100 tests 7,200 s (3,600 s), ring16 1,200 s, and 1,600-update
   ring16 600 s. On top of that, the contention-aware extension can extend a run once, up to 3x, by the measured
   slowdown. Declared `timeout_seconds` and update budgets are unchanged.

Also in this PR:
- develop is merged in, with the view resolved as revision 12 (above).
- The dispositions check now accepts a later view revision whose fingerprinted archive chain reaches the
  disposition revision.
- The ring16 problem definition now names the 6,400-update task.
- These results are rendered as leaderboard research rows (`experiments/forge/research_tiers.py: adopt_rows`).

## Before and after, per family

"Before" is develop at `6a91a4991`. "After" is this PR, regenerated with `--refresh-publication`.

**Selected recipes (recorded cohorts, unchanged evidence).** None of the recorded cohorts was ever measured on
the 6,400-update ring16. Swapping that question therefore turns each one's recorded 1,600-update ring16 verdict
into *not run*. No verdict was regraded.

| Family | Full suite, 33 tests: before (pass / fail / other) | after | Qualification subset, 81 view requirements: before | after |
| --- | --- | --- | --- | --- |
| BCAP | 16 / 12 / 5 not measured | 15 / 12 / 6 not measured | 71/81 (4 blocked) | 70/81 (1 not run, 4 blocked) |
| BISPF | 17 / 11 / 5 | 16 / 11 / 6 | 77/81 (4 diagnostic attempts) | 76/81 (1 not run, 4 diagnostic attempts) |
| GAN v3 | 5 / 2 / 26 | 5 / 1 / 27 | 20/81 (4 blocked, 55 diagnostic attempts) | 20/81 (1 not run, 4 blocked, 55 diagnostic attempts) |
| K3P | 5 / 2 / 26 | 5 / 1 / 27 | 20/81 (same) | 20/81 (1 not run, same) |
| KA2 | 5 / 2 / 26 | 5 / 1 / 27 | 20/81 (same) | 20/81 (1 not run, same) |
| R1/R2 | 5 / 2 / 26 | 5 / 1 / 27 | 20/81 (same) | 20/81 (1 not run, same) |
| Atlas, E22 | 0 / 0 / 29 blocked, 4 not measured | 0 / 0 / 28 blocked, 5 not measured | 0/81 | 0/81 (1 not run) |

**New adopt rows (this PR's measurements).** Distinct tests use the measured draft-10 assignment (6/21/2 = 29
tests, the same scale as the other research rows). The revision-12 columns count only what the current view counts:
17 tests in `discriminator_stability`, and 81 view requirements over the six ordinary views, with shared tests
counted in every view that requires them.

| Family / recipe | Before: closest research row | Distinct tests, 29 | Tier 1 / 2 / 3 | Rev 12 `discriminator_stability`, 17 | Rev 12 six views, 81 requirements |
| --- | --- | --- | --- | --- | --- |
| K3P `k3p-v2` (no cap) | v1 Tier 2 diagnostics 11/21 (older source); Tier 1 failed ring16 at 1,600 | **16/29** (11 fail, 2 blocked) | 6/6 · 10/21 · 0/2 blocked | 14/17 (2 fail, 1 blocked) | 69/81 (8 fail, 4 blocked) |
| KA2 `ka2-v2` (no cap) | v1 Tier 2 diagnostics 10/21; Tier 1 failed ring16 at 1,600 | **15/29** (12 fail, 2 blocked) | 6/6 · 9/21 · 0/2 blocked | 14/17 (2 fail, 1 blocked) | 69/81 (8 fail, 4 blocked) |
| R1/R2 (unchanged) | Tier 2 diagnostics 9/21; Tier 1 failed ring16 at 1,600 | **16/29** (11 fail, 2 blocked) | 6/6 · 10/21 · 0/2 blocked | 12/17 (4 fail, 1 blocked) | 57/81 (20 fail, 4 blocked) |
| GAN v3 (unchanged) | Tier 2 diagnostics 9/21; Tier 1 failed ring16 at 1,600 | **15/29** (12 fail, 2 blocked) | 6/6 · 9/21 · 0/2 blocked | 14/17 (2 fail, 1 blocked) | 69/81 (8 fail, 4 blocked) |
| BCAP (unchanged, partial) | recorded cohort above | **7/29** (1 fail, 21 not run) | 6/6 · 1/21 · 0/2 not run | 7/17 (10 not run) | 23/81 (58 not run) |
| BISPF (unchanged, partial) | recorded cohort above | **7/29** (1 fail, 21 not run) | 6/6 · 1/21 · 0/2 not run | 7/17 (10 not run) | 23/81 (58 not run) |

All six recipes now pass Tier 1, including ring16 at 6,400 updates. K3P, KA2, R1/R2 and GAN v3 get past Tier 1 in a
tier-gated request for the first time. Every one of them fails `mode_hold` and `five_word_joint_hold`, so the
`complete_current_tier` policy releases no Tier 3 work.

## Wall-clock outcomes

Every measured cell ended on its declared update budget. No cell is NOT MEASURED, no extension was used, and
`wallclock.decision` is null in every record. The slowest runs relative to their limit were word smoke for BCAP and
BISPF: 799 s and 814 s against the 900 s declared limit and the 1,800 s operational limit. For the other families,
word smoke took 119-184 s, toy100 71-102 s against 7,200 s, and ring16 6,400 took 61-145 s against 1,200 s.

## Caveats

- **Single seed (0).** No seed sweep, by rule. R1/R2's ring16 pass is thin: spread ratio 0.157 against 0.15,
  stable only from update 6,334 (exactly the 5 required final checks), with 69/384 checks passing.
- **Weak smoke passes.** `gaussian1d_smoke` and `five_word_joint_smoke` use their declared "any confirmed pass"
  gates, which are not sustained. Word smoke passed early and then collapsed by update 20,001 for K3P v2
  (3/24 checks), KA2 v2 (4/24), R1/R2 (1/24) and GAN v3 (1/24). GAN v3's gaussian smoke is 1/24. Word hold fails
  for all four. Without the cap, the anneal spans 20,000 updates on word smoke, and this collapse fits that, but one
  seed does not isolate the cause. KA2 v1's word-hold PASS was a diagnostic run on older source `a5eafc73`, so it is
  context and not a measured regression.
- **`gaussian1d_stability`** (held in revision 12) fails for all six. K3P/KA2 keep 71/72 stationary checks but fail
  all 48 frozen checks and re-acquisition. The readout's "107/121" figure counted live observations, not grader
  checks.
- **Co-tenancy.** nvidia-smi was sampled every 15 s during each drain. GPU 0 shared about 260-300 MiB with
  desktop processes (nano-work-server, the desktop, an occasional small python3). GPU 1 had a resident llama-server
  of about 17.2 GB for every run on it (KA2, GAN v3, BISPF). No result depends on a wall-clock limit, so co-tenancy
  changed no verdict. Lease-held GPU time: K3P 1,009 s, KA2 1,011 s, R1/R2 + GAN v3 1,667 s, BCAP + BISPF 2,533 s.
- **Partial BCAP/BISPF.** Their campaigns were paused after `five_word_joint_hold`; the other 21 counted tests were
  not run on this source. Neither recipe had a wall-clock hit under the old limits, so they were measured only for
  the ring16 change. Archived results from other sources are **not** carried over as `1e02f185` results.
- **Excluded, not published:** Tier 3 (`ring_hold`, `ring_extension`) appears only as *blocked (tier veto)* for the
  four full families and *not run* for BCAP/BISPF, never as FAIL. `clockfree_audit_measurement_v1` is recorded as
  a diagnostic: it fails for K3P/KA2/R1/R2/GAN v3 and passes for BCAP/BISPF. It is kept out of the 29-test rows,
  although revision 12's clock-free view counts it, as the six-view column shows. No unverified cell was imported:
  all 130 records (27 or 8 counted cells per family, plus the audit) were checked by an independent verifier
  against the queue receipts and the source digest.
- **Not registered as qualification.** The family pins in `family-current-v1.json` and the K3P/KA2 declared revisions
  (`status: awaiting_measurement`) are unchanged. Registering ordinary evidence (`--advance-policy --source-commit`)
  needs every ordinary view measured on one source; these runs cover only `discriminator_stability`, and only up to
  the tier veto.

## Reproduce

```sh
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python
OMP_NUM_THREADS=1 $PY reports/forge/adopt-20261016/collect.py --state /ml2/hypergan/adopt-20261016
OMP_NUM_THREADS=1 $PY reports/forge/regenerate_technique_inventory.py --refresh-publication
```

`collect.py` imports only the verified cell set. It refuses a queue whose source, candidate, seed, measured view,
graded outcome or wall-clock decision differs. Raw receipts stay in the state directories
(`/ml2/hypergan/adopt-20261016/<group>/queue/adopt-20261016-<group>-v1/<attempt>/`); their sha256 are in
[`results.json`](results.json). Readouts are in `/ml2/hypergan/adopt-20261016/{k3p,ka2,r1r2-ganv3,bcap-bispf}/READOUT.md`.

<a name="measured-cells"></a>

## Measured cells

Graded results from the queue's independent grader, on source `1e02f185`, seed 0. ⚠ marks a cell with a
verifier note in `results.json`. "Spread" is `component_min_eigen_ratio` (gate ≥ 0.15). "Suffix" is the terminal
run of passing checks. Toy100 precision is the 100,000-sample holdout value.

| Test (measured tier) | Rev 12 role | K3P v2 | KA2 v2 | R1/R2 | GAN v3 | BCAP | BISPF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gaussian1d_smoke` (T1) | counted | **PASS** · 7/24 checks pass, first confirmed 167 | **PASS** · 7/24 checks pass, first confirmed 167 | **PASS** · 6/24 checks pass, first confirmed 292 | **PASS** · 1/24 checks pass, first confirmed 709 | **PASS** · 3/24 checks pass, first confirmed 375 | **PASS** · 2/24 checks pass, first confirmed 667 |
| `two_pole` (T1) | counted | **PASS** · mean_abs 0.448, grad_med 0.187 | **PASS** · mean_abs 0.441, grad_med 0.141 | **PASS** · mean_abs 0.381, grad_med 0.693 | **PASS** · mean_abs 0.455, grad_med 0.488 | **PASS** · mean_abs 0.959, grad_med 0.952 | **PASS** · mean_abs 0.958, grad_med 0.914 |
| `unused_token_hold` (T1) | counted | **PASS** · unused_hold 1, concept_move 0.997 | **PASS** · unused_hold 1, concept_move 0.996 | **PASS** · unused_hold 1, concept_move 1 | **PASS** · unused_hold 0.99, concept_move 0.938 | **PASS** · unused_hold 0.987, concept_move 0.959 | **PASS** · unused_hold 0.986, concept_move 0.928 |
| `ae_gan_hold` (T1) | counted | **PASS** · recon_mse 0.00462, hold 0.00191 | **PASS** · recon_mse 0.00369, hold 0.00312 | **PASS** · recon_mse 0.00285, hold 0.00164 | **PASS** · recon_mse 0.00382, hold 0.000345 | **PASS** · recon_mse 0.00435, hold 0.0158 | **PASS** · recon_mse 0.00881, hold 0.035 |
| `ring16_acquisition_6400` (T1) | counted | **PASS** · modes 16, spread 0.413, suffix 66/384 | **PASS** · modes 16, spread 0.213, suffix 47/384 | **PASS** · modes 16, spread 0.157, suffix 5/384 ⚠ | **PASS** · modes 16, spread 0.442, suffix 43/384 | **PASS** · modes 16, spread 0.282, suffix 314/384 | **PASS** · modes 16, spread 0.179, suffix 151/384 |
| `five_word_joint_smoke` (T1) | counted | **PASS** · 3/24 checks pass, first confirmed 834, endpoint **fails** | **PASS** · 4/24 checks pass, first confirmed 834, endpoint **fails** | **PASS** · 1/24 checks pass, first confirmed 2501, endpoint **fails** | **PASS** · 1/24 checks pass, first confirmed 834, endpoint **fails** | **PASS** · 23/24 checks pass, first confirmed 834, endpoint pass | **PASS** · 21/24 checks pass, first confirmed 2501, endpoint pass |
| `gaussian1d_stability` (T2) | held | **FAIL** · stationary 71/72, frozen 0/48, shift hold 22/24, re-acq FAIL | **FAIL** · stationary 71/72, frozen 0/48, shift hold 16/24, re-acq FAIL | **FAIL** · stationary 19/72, frozen 0/48, shift hold 5/24, re-acq FAIL | **FAIL** · stationary 11/72, frozen 0/48, shift hold 3/24, re-acq FAIL | **FAIL** · stationary 2/72, frozen 0/48, shift hold 0/24, re-acq FAIL | **FAIL** · stationary 5/72, frozen 0/48, shift hold 0/24, re-acq FAIL |
| `five_word_joint_hold` (T2) | counted | **FAIL** · 12/25 checks pass, first confirmed 834, endpoint **fails** | **FAIL** · 18/25 checks pass, first confirmed 834, endpoint **fails** ⚠ | **FAIL** · 1/25 checks pass, first confirmed 2501, endpoint **fails** | **FAIL** · 3/25 checks pass, first confirmed 834, endpoint **fails** | **PASS** · 25/25 checks pass, first confirmed 834, endpoint pass | **PASS** · 25/25 checks pass, first confirmed 2501, endpoint pass |
| `trajectory` (T2) | counted | **PASS** · identity_mse 0.000814 | **PASS** · identity_mse 0.000813 | **FAIL** · identity_mse 0.26 | **PASS** · identity_mse 1.59e-05 | not run | not run |
| `residual_student` (T2) | counted | **PASS** · identity_mse 0.000822, success_rate 1 | **PASS** · identity_mse 0.000822, success_rate 1 | **FAIL** · identity_mse 0.0443, success_rate 0.583 | **PASS** · identity_mse 1.2e-05, success_rate 1 | not run | not run |
| `unipolar` (T2) | counted | **PASS** · cover 0.996, off_caption 0.00109 | **PASS** · cover 0.997, off_caption 2.62e-06 | **PASS** · cover 1, off_caption 6.13e-12 | **PASS** · cover 0.966, off_caption 0.000264 | not run | not run |
| `cover_leftover` (T2) | counted | **PASS** · u_kept 0.998, content_kept 1 | **PASS** · u_kept 0.997, content_kept 1 | **PASS** · u_kept 0.999, content_kept 1 | **PASS** · u_kept 1, content_kept 1.01 | not run | not run |
| `mid_scale_identity` (T2) | counted | **PASS** · concept_cos_plus 1, concept_cos_minus 1 | **PASS** · concept_cos_plus 1, concept_cos_minus 1 | **PASS** · concept_cos_plus 1, concept_cos_minus 1 | **PASS** · concept_cos_plus 1, concept_cos_minus 1 | not run | not run |
| `mode_hold` (T2) | counted | **FAIL** · modes 5 | **FAIL** · modes 6 | **FAIL** · modes 5 | **FAIL** · modes 7 | not run | not run |
| `vector_two_broad` (T2) | counted | **PASS** · sw1_normalized 0.0222, mass_tv 0.00659 | **PASS** · sw1_normalized 0.0392, mass_tv 0.0239 | **PASS** · sw1_normalized 0.0299, mass_tv 0.0146 | **PASS** · sw1_normalized 0.0865, mass_tv 0.0107 | not run | not run |
| `vector_unequal_mass` (T2) | held | **FAIL** · component_covariance_error 1.2, component_min_eigen_ratio 0.0985 | **FAIL** · component_covariance_error 1.65, component_min_eigen_ratio 0.0911 | **FAIL** · component_covariance_error 6.35, component_min_eigen_ratio 0.108 | **FAIL** · component_covariance_error 1.75, component_min_eigen_ratio 0.0526 | not run | not run |
| `vector_unequal_width` (T2) | held | **FAIL** · component_covariance_error 0.966 | **FAIL** · component_covariance_error 0.993 | **FAIL** · component_covariance_error 12.1 | **FAIL** · component_min_eigen_ratio 0.094 | not run | not run |
| `vector_anisotropic` (T2) | held | **PASS** · sw1_normalized 0.103, mass_tv 0.105 | **PASS** · sw1_normalized 0.107, mass_tv 0.105 | **PASS** · sw1_normalized 0.0845, mass_tv 0.0533 | **FAIL** · component_covariance_error 0.938 | not run | not run |
| `vector_overlap` (T2) | held | **PASS** · sw1_normalized 0.0558, mean_error 0.0483 | **FAIL** · sw1_normalized 0.0902, mean_error 0.116 | **FAIL** · mean_error 0.155 | **FAIL** · sw1_normalized 0.201, mean_error 0.286 | not run | not run |
| `vector_spiral` (T2) | counted | **PASS** · sw1_normalized 0.0306, mean_error 0.0335 | **PASS** · sw1_normalized 0.0317, mean_error 0.0217 | **PASS** · sw1_normalized 0.037, mean_error 0.0375 | **PASS** · sw1_normalized 0.0267, mean_error 0.02 | not run | not run |
| `img_stripes2` (T2) | counted | **PASS** · modes 2, hq 1 | **PASS** · modes 2, hq 1 | **PASS** · modes 2, hq 0.969 | **PASS** · modes 2, hq 1 | not run | not run |
| `img_bars4` (T2) | held | **FAIL** · modes 3 | **FAIL** · modes 3 | **PASS** · modes 4, hq 0.938 | **FAIL** · modes 3 | not run | not run |
| `img_blobs4` (T2) | held | **FAIL** · modes 0, hq 0 | **FAIL** · modes 0, hq 0 | **PASS** · modes 4, hq 1 | **FAIL** · modes 3, hq 0.875 | not run | not run |
| `img_intensity2` (T2) | held | **FAIL** · modes 0, hq 0 | **FAIL** · modes 0, hq 0 | **PASS** · modes 2, hq 0.969 | **PASS** · modes 2, hq 1 | not run | not run |
| `grid100` (T2) | held | **FAIL** · precision 0.732, mass_tv 0.18 | **FAIL** · precision 0.735, mass_tv 0.147 | **FAIL** · precision 0.602, mass_tv 0.149 | **FAIL** · precision 0.388, mass_tv 0.349 | not run | not run |
| `rotated100` (T2) | held | **FAIL** · precision 0.643, mass_tv 0.0884 | **FAIL** · precision 0.656, mass_tv 0.108 | **FAIL** · precision 0.639, mass_tv 0.075 | **FAIL** · precision 0.598, mass_tv 0.126 | not run | not run |
| `staggered100` (T2) | held | **FAIL** · precision 0.652, mass_tv 0.136 | **FAIL** · precision 0.647, mass_tv 0.163 | **FAIL** · precision 0.538, mass_tv 0.145 | **FAIL** · precision 0.528, mass_tv 0.206 | not run | not run |
| `ring_hold` (T3) | not counted | blocked (tier veto) | blocked (tier veto) | blocked (tier veto) | blocked (tier veto) | not run | not run |
| `ring_extension` (T3) | counted | blocked (tier veto) | blocked (tier veto) | blocked (tier veto) | blocked (tier veto) | not run | not run |
| `clockfree_audit_measurement_v1` (T1, diagnostic) | not counted | **FAIL** · step_label changed the update or common-prefix state | **FAIL** · step_label changed the update or common-prefix state | **FAIL** · step_label changed the update or common-prefix state | **FAIL** · step_label changed the update or common-prefix state | **PASS** · declared state/horizon/cadence/restart comparisons agree; source audit bound | **PASS** · declared state/horizon/cadence/restart comparisons agree; source audit bound |
