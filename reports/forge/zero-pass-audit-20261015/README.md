# Forge zero pass test review

<!-- superseded-banner: generated -->
> Historical audit of 2026-10-15. Recommendations and counts below belong to the pinned snapshot.
> Later dispositions are in [the test disposition report](../test-dispositions-20261015/README.md);
> current status is in the [technique inventory](../technique-inventory.md).
> The extraction now reads the frozen snapshot directly, so `--check` remains reproducible on later checkouts.

Keep the useful coverage and density tests. Change how composite tests, incompatible formulations and checkpoint dependencies are presented. Consolidate `ring_extension` into the endurance result instead of counting the same uninterrupted run as another independent test.

At develop snapshot [`17df086e92e4`](https://github.com/255BITS/ParticleGAN/tree/17df086e92e4ae700a7676993238094191d24dee), **18 of the 35 distinct required tests have no PASS in the eight selected family cohorts**. For these 18, the recommendations are **11 keep, 6 modify and 1 remove as a separate counted test**. All 18 are reviewed individually below. The seven separate policy questions and 17 other variants visible in the family index are also covered individually.

The selected board's zeros do not establish that nobody has ever passed. Nine of the 18 already have PASS records in the 837-pair backfill, under their own diagnostic or research contracts. Of the nine without an indexed backfill PASS, four are blocked continuations. One of those, BISPF's ring extension, already has passing saved observations: this review recomputed **300/300 passing extension checks with the existing evaluator**, adding no training or sampling. Its original index entry remains BLOCKED pending artifact certification and publication.

Sources: [selected leaderboard](../technique-inventory.md), [full indexed data](../technique-inventory.json), [837 exact attempt records](../attempt-coverage.json), [earlier deeper diagnostics](../actual-missing-diagnostics-20261014/README.md), and the [reproducible extraction](evidence.json). Counts below are fixed to the snapshot, including its one RUNNING attempt.

## Scope and meaning of a zero

The six ordinary views contain 152 required-view instances but only 35 unique tasks. A shared task appears repeatedly in those totals. This review selects each family's `current_cohort_anchor`, checks the actual task verdicts, and deduplicates task IDs across views. It does not select a different successful recipe for each task or transfer diagnostic passes into ordinary qualification.

Three distinctions explain much of the apparent universal failure:

- **Qualification versus measurement.** Later-tier diagnostics can pass while the selected qualification remains blocked by earlier failures, source changes or a different runtime. BCAP's strict clock audit is an example.
- **Failure versus inability to run.** Atlas/E22's ordinary learned-MoG hosts are incompatible with their particle birth/death formulation. Conditional, encoder and joint-word hosts also lack complete policy ownership contracts. These are contract or capability gaps, rather than measured failures of all those learning problems.
- **A checkpoint dependency versus another independent trial.** A continuation must consume the same recipe's certified passing parent state. A failed or unbound parent prevents execution. Repeated blocked children provide no extra evidence that the child numerical gate is impossible.

Recorded Torch 2.14 selected results, Torch 2.13 diagnostics, selected-policy serving and historical noisy/native results keep their original identities. The following counts are **backfill record counts for the named task**, not pooled success probabilities or a ranking of interchangeable trials.

## Decisions for all 18 required tests

“Keep” retains the question and numerical gate. “Modify” identifies a specific reporting, calibration or execution-contract change; any scientific change requires a new version. “Remove” below means remove a separate denominator entry while preserving its measurements and requirements within the parent test. No implementation or historical verdict was changed by this report.

| Test | Backfill outcomes | Why the selected board has zero and what the evidence shows | Decision |
| --- | --- | --- | --- |
| [`clockfree_audit`](../../../configs/forge/tasks/clockfree_audit.json) | 2 PASS, 4 FAIL, 2 BLOCKED | BCAP passes all four state comparisons in a separate Torch 2.13 cohort; BISPF also passes. Four scheduled recipes actually fail the step-label probe. The selected cohort has no compatible recorded PASS. | **Keep.** It tests a real clock-free claim. Show its diagnostic PASS beside the selected-cohort status and apply it to recipes claiming clock independence. |
| [`gaussian1d_stability`](../../../configs/forge/tasks/gaussian1d_stability.json) | 3 PASS, 16 FAIL, 7 BLOCKED, 1 ERROR | The selected BCAP and BISPF states lose scalar distribution quality during continuation/shift. Their final KS values are 0.320623 and 0.224280 against 0.05; standard-deviation ratios are 0.662330 and 0.634153 against [0.8, 1.2]. Three kinetic research recipes have recorded full passes under their own contracts. | **Modify.** Expose stationary retention, deadline reacquisition and shifted hold as named phase results beneath one complete gate. Keep the original conjunction and thresholds. |
| [`vector_unequal_mass`](../../../configs/forge/tasks/vector_unequal_mass.json) | 3 PASS, 20 FAIL, 4 BLOCKED | BCAP fails rare-mode shape and occupancy: minimum eigen ratio 0.009073 < 0.15 and minimum mass ratio 0.207520 < 0.25. BISPF fixes aggregate mass error (TV 0.023369) but its minimum eigen ratio is still 0.004335. Research passes exist. | **Modify.** Calibrate rare-component shape measurements against this task's actual nonzero-width MoG sampling law and expose per-component support. Preserve occupancy, quality and distribution gates; do not copy the finite-atom exemption without that calibration. |
| [`vector_unequal_width`](../../../configs/forge/tasks/vector_unequal_width.json) | 23 FAIL, 4 BLOCKED | This remains numerically unsolved in the backfill. BCAP fails full covariance, mass and transport distance. BISPF passes shape at the endpoint but still has mass TV 0.250244 > 0.15 and normalized SW1 0.266550 > 0.18. Both have zero fully passing observations out of 24. | **Keep.** Unequal scales expose a real mass-allocation failure. Adding core/spill diagnostics is useful; changing only covariance grading would not fix BISPF's failed mass and distance gates. |
| [`vector_anisotropic`](../../../configs/forge/tasks/vector_anisotropic.json) | 12 PASS, 11 FAIL, 4 BLOCKED | BCAP's endpoint shape passes, but mass TV 0.195964 and SW1 0.197952 exceed 0.15 and 0.18. BISPF also fails these bounds. K3P, KA2, R1/R2 and several research recipes have diagnostic passes. | **Keep.** This distinguishes directional shape from correct mixture mass. Show the existing successful diagnostic controls. |
| [`vector_overlap`](../../../configs/forge/tasks/vector_overlap.json) | 11 PASS, 12 FAIL, 4 BLOCKED | BCAP passes the endpoint covariance bound but fails mean error 0.388766 > 0.15 and SW1 0.255769 > 0.18. It has five passing observations during training and zero terminal suffix passes. BISPF also drifts out of the gate. K3P and research passes exist. | **Keep.** The scorer correctly uses observable distribution statistics rather than unidentifiable component labels. A temporary earlier pass should not replace terminal quality. |
| [`img_bars4`](../../../configs/forge/tasks/img_bars4.json) | 2 PASS, 19 FAIL, 5 BLOCKED, 1 ERROR | BCAP and BISPF each finish with only 2 of 4 required quality modes, even though HQ is 0.96875 and 1.0. R1/R2 and released GAN v3 pass in the diagnostic panel. | **Keep.** This catches missing modes that good-looking individual images and high HQ conceal. Preserve the 600-update budget and five-check suffix. |
| [`img_blobs4`](../../../configs/forge/tasks/img_blobs4.json) | 6 PASS, 15 FAIL, 5 BLOCKED, 1 ERROR | BCAP ends with 2 modes and HQ 0.75; BISPF has 3 modes and HQ 0.875. The bounds are 4 modes and HQ ≥ 0.9. R1/R2 and five research recipes pass. | **Keep.** The failure is incomplete coverage and quality, with successful controls already available. |
| [`img_intensity2`](../../../configs/forge/tasks/img_intensity2.json) | 3 PASS, 18 FAIL, 5 BLOCKED, 1 ERROR | BCAP has 2 modes but HQ 0.6875; BISPF has 1 mode and HQ 0.5. Released GAN v3, R1/R2 and a kinetic research recipe pass. Separate Atlas/E22 diagnostics reach a good endpoint but only a four-check terminal suffix, short of five. | **Keep.** Intensity fidelity and sustained quality are useful requirements. Preserve the suffix; display the near miss without turning it into a pass. |
| [`grid100`](../../../configs/forge/tasks/grid100.json) | 19 FAIL, 5 BLOCKED, 3 TIMEOUT | Selected BCAP's 100k holdout precision is 0.24072 and BISPF's is 0.37015, far below 0.97. Both also fail centering, covariance, radial shape and mass criteria. The recorded true-target oracle passes. | **Keep.** This is a substantial density-learning failure. Report the individual gate margins; 100 occupied modes alone do not establish accurate density. |
| [`rotated100`](../../../configs/forge/tasks/rotated100.json) | 17 FAIL, 5 BLOCKED, 5 TIMEOUT | Selected holdout precision is 0.25552 for BCAP and 0.26601 for BISPF. BISPF's center RMS is 1.363758σ against 0.2σ. Some BCAP shape values are unavailable because the accuracy scorer cannot resolve them; precision and mass already fail. | **Keep.** Rotation provides a separate geometry check. Retain unavailable values explicitly and distinguish timed-out executions from completed failures. |
| [`staggered100`](../../../configs/forge/tasks/staggered100.json) | 17 FAIL, 5 BLOCKED, 4 TIMEOUT, 1 RUNNING | Selected holdout precision is 0.30168 for BCAP and 0.34184 for BISPF. BISPF also has center RMS 1.067593σ and covariance trace bias 0.571332, against 0.2σ and 0.1. One research attempt was still running in this snapshot. | **Keep.** Staggering probes geometry beyond a square grid. No conclusion should be assigned to the running attempt. |
| [`grid100_14k`](../../../configs/forge/tasks/grid100_14k.json) | 9 BLOCKED | The 7k grid parent fails or lacks the certified passing artifact; the child also requires a strict clock audit. The observed zero comes from refused starts, not 14k numerical results. | **Modify.** Make this an explicitly eligible endurance continuation of a passing grid state. Display parent failure separately. A “can another 7k updates rescue a failed grid?” question needs a different diagnostic declaration. |
| [`rotated100_14k`](../../../configs/forge/tasks/rotated100_14k.json) | 9 BLOCKED | Same dependency chain for the rotated 7k parent. No complete child trajectory exists in this backfill. | **Modify.** Retain the exact own-state and clock requirements, label it conditional, and separate a failed-parent duration study from the retention claim. |
| [`staggered100_14k`](../../../configs/forge/tasks/staggered100_14k.json) | 9 BLOCKED | Same dependency chain for the staggered 7k parent. Repeated refusals do not measure long-horizon quality. | **Modify.** Apply the same eligible-continuation presentation and preserve the source, schedule horizon and named RNG state. |
| [`ring_hold`](../../../configs/forge/tasks/ring_hold.json) | 1 PASS, 21 FAIL, 5 BLOCKED | The original selected cohorts have no qualifying Tier 3 measurement. In diagnostics, BCAP acquires 200 consecutive good checks, then loses hold at update 3,760; four older recipes never acquire the window by update 6,000. BISPF now passes 1,200 uninterrupted hold checks. | **Keep.** Acquiring quality once and maintaining it are different capabilities. Show the BISPF diagnostic pass and the acquisition/hold phase causes separately. |
| [`ring_extension`](../../../configs/forge/tasks/ring_extension.json) | 27 BLOCKED | Most parents fail. BISPF's parent passes, but the backfill requires top-level checkpoint/manifest keys that its grouped endurance receipt does not supply. That same run already includes the 300 extension observations, all passing on recheck. | **Remove as a separate counted test.** Fold the required 300-check extension into the single endurance result, with its own visible subphase. Repair evidence binding and certify the existing observations before changing any displayed verdict. |
| [`target_shift_recovery`](../../../configs/forge/tasks/target_shift_recovery.json) | 6 FAIL, 2 BLOCKED | The gate requires every check from update 2,800 through 3,600 to pass, plus a matched frozen negative control. BCAP passes 66/81 but keeps failing intermittently. BISPF also passes 66/81 and becomes continuously good only at 3,020, after the 2,800 deadline. | **Modify.** Name and display deadline acquisition, post-deadline hold and frozen-control rejection separately. Keep the original complete gate; the current label hides whether a recipe is late or unstable. |

The selected BCAP numbers above come from its actual selected runtime receipts, rather than a different BCAP baseline quoted elsewhere. Full endpoint values, exact receipt paths, declarations and backfill witness identities are retained in [evidence.json](evidence.json).

## Why the hard numerical failures are useful

### Density and mass are failing independently

The narrow and broad components in `vector_unequal_width` have target standard deviations 0.07, 0.12, 0.20 and 0.30. The common architecture must learn both their proportions and scales. BCAP's full covariance error is 6.287559 while its core-only diagnostic is 0.444178, showing that tails or misplaced samples matter strongly. But BISPF's acceptable full covariance error, 0.492137, coexists with badly wrong mass and SW1. A covariance-only relaxation would therefore leave a material failure.

In `vector_unequal_mass`, a 2% target component receives about 5.12 of the 256 equally weighted prior locations at proportional occupancy. The upstream finite-atom oracle documents unreliable shape estimates at such small support, and its newer scorer excludes under-resolved components from some shape aggregates. **Forge deliberately retains full-component gates because its MoG has nonzero width.** The finite-atom oracle is a reason to calibrate the exact MoG measurement, not proof that the current gate is invalid. Preserve the observed rare-mode collapse and distinguish sampling uncertainty from a trained model's missing spread. [Scorer and support rules](../../../benchmarks/transfer_suite/vector_tasks.py); [finite-atom oracle](../../transfer_suite/silu_rare_collapse/README.md#oracle-how-32-was-chosen).

For the three 100-mode tasks, the recorded true-target oracle has precision 0.98914, center RMS about 0.04174σ and radial KS about 0.00151, within the numerical limits. This supports the scorer's ability to accept the target law; it does not prove that every declared model can learn it. Selected trained precision around 0.24–0.37 is a large shortfall, not a boundary rounding issue. Historical Atlas native passes use another prior/serving contract and cannot fill these clean-MoG cells. [Original accuracy and coverage gates](../../../configs/forge/tasks/grid100.json); [preserved Atlas cohorts](../atlas-compatible-coverage-20261013/historical/README.md).

### A complete temporal claim is stricter than a good endpoint

The Gaussian stability question combines 72 stationary checks, shifted reacquisition by update 5,000 and 24 subsequent hold checks. The ring adaptation question instead demands all 81 post-deadline checks pass. These are different protocols. Both expose real instability, but a single FAIL hides which capability failed. Phase reporting would make further work more specific without retrospectively weakening the verdict.

Keep the image tests' five-check terminal suffix too. Atlas/E22 Intensity diagnostics end at HQ 1.0 with both modes, yet only four terminal observations pass. That is a near miss under a declared sustained criterion. The other recipes' successful image diagnostics show that the original test is achievable. [Image and adaptation observations](../actual-missing-diagnostics-20261014/results.json).

### The ring extension has an evidence plumbing gap

For BISPF attempt `bf5d13ff3aeb4cba98d07373963839d8`, the saved curve is contiguous from update 1,201 through 2,900. The first qualifying 200-check window ends at 1,400; the required 1,200-check hold ends at 2,600. The following updates 2,601–2,900 contain **300/300 passing checks**, with minimum HQ **0.9921875**. The saved continuity record says uninterrupted, zero resumes.

The existing [`_ring` evaluator](../../../experiments/forge/views.py) returns PASS on those saved observations for the extension declaration. The original raw/result hashes match the public backfill entry, and the saved provenance checkpoint file's hash and size were verified. This is a numerical diagnostic recheck, not a new certified qualification. [Compact recheck receipt](ring-extension-recheck.json).

The concrete mismatch is in [`prerequisite_bindings`](../../../experiments/forge/attempt_coverage.py): it requires `raw.evidence.checkpoint` and `raw.evidence.artifact_manifest`. The grouped ring receipt instead carries dense observations and a nested `provenance_checkpoint`. The adapter already trains through the extension in the producer run. Certifying the grouped evidence is preferable to training it again or restarting at a fictitious parent state. Even if the separate denominator entry is retained, this binding should be fixed.

## Decisions for the seven zero pass policy questions

These are separate selected-policy particle-cloud questions, each with the suffix `_tier1_policy_selected_cloud_v1`. They retain their own scope and cannot be pooled with the ordinary clean/live MoG tests. The table names every question, including those refused before training. [Original seven-question results](../atlas-compatible-coverage-20261013/README.md); [later adapter investigation](../actual-missing-diagnostics-20261014/README.md); [latest three Atlas attempts](../ATTEMPT_COVERAGE.md#remaining-attempts).

| Exact task | Why no pass is recorded | Decision |
| --- | --- | --- |
| `gaussian1d_acquisition_tier1_policy_selected_cloud_v1` | Atlas/E22 complete 1,000 updates but fail export because the strict JSON state digest rejects a policy diagnostic NaN. The saved Gaussian endpoint has KS 0.029021, mean error 0.030552σ and width ratio 1.023067. These endpoint values do not establish the complete acquisition gate. The saved state passes its existing finite-policy health check. | **Modify.** Give legitimate nonfinite diagnostic sentinels a canonical checkpoint representation while continuing to reject nonfinite learned tensors. Validate full-curve and same-state confirmation evidence before certifying a pass. |
| `ring16_acquisition_tier1_policy_selected_cloud_v1` | The same export error prevents a complete receipt. There is also a numerical failure: original Atlas covariance error 1.812179 > 0.85; later E22 diagnostics report 1.139007 > 0.85. | **Modify.** Repair the same serialization contract, retaining the covariance gate and the separate original numerical negatives. Export success alone would not repair Ring16. |
| `clockfree_audit_tier1_policy_selected_cloud_v1` | Actual state comparisons fail: the original Atlas cohort matches horizon and evaluation cadence, but differs on step label and restart. E22 also fails an actual audit. | **Keep.** Exact restart and label independence are meaningful claims. A failure is informative for a policy with clock-sensitive behavior. |
| `two_pole_tier1_policy_selected_cloud_v1` | Initially rejected by inherited optional transport metadata even when transport was disabled. After a narrow admission fix, Atlas/E22 complete 80 updates but lack recipe-owned prior-policy proof. Their measured mean absolute coordinate is 0.002586 versus the required 0.3. | **Modify.** Correct admission and emit the actual prior-policy proof. Preserve the 80-update question and its observed numerical shortfall; do not report this solely as an unrun test. |
| `unused_token_hold_tier1_policy_selected_cloud_v1` | The explicit component host lacks a complete policy-control and parameter-ownership contract. No compatible numerical run establishes a failure. | **Modify.** Specify which rows may move, which must remain fixed, and which selected state is measured. Keep the useful unused-token retention control; require that runnable contract before spending. |
| `ae_gan_hold_tier1_policy_selected_cloud_v1` | The encoder/component host lacks complete policy ownership and transport support. Historical AE repairs belong to different auxiliary-prior/serving cohorts. | **Modify.** Define encoder, generator and prior ownership plus reconstruction/hold sampling for the policy host. Retain historical evidence and the original hold/reconstruction purposes. |
| `five_word_joint_acquisition_tier1_policy_selected_cloud_v1` | The five-row joint BiGAN fixture conflicts with policy population/row semantics and lacks a separate minimum-population contract. It was refused before training. | **Modify.** Declare a policy-compatible joint host with explicit row ownership and minimum population, preserving the word acquisition and reconstruction controls. Changed capacity is a new task version, not a repaired old score. |

The ordinary Atlas/E22 “incompatible” cells should link to these concrete causes and separately scoped results. They should not be converted to ordinary PASS or FAIL merely to fill the main board.

## Decisions for 17 other indexed variants

These task IDs appear in family navigation but are not required in the six main-view totals. Their selected cohorts have no PASS; many have no selected execution at all. They are included to avoid treating an empty variant cell as an unexplained universal failure. The exact definitions and empty backfill coverage are in [evidence.json](evidence.json).

| Indexed task | Explanation | Decision |
| --- | --- | --- |
| `grid100_affine_paired_laws_v1` | Separately declared affine/paired-law host. Archived full-horizon studies fail centering or shape despite some coverage gains. It is not the ordinary grid task. | **Keep** as a scoped geometry control; show its historical receipts and its unmeasured selected cells separately. |
| `grid100_affine_square_named_v1` | Named affine fixture with explicit initialization/RNG identity; no selected execution in this index. | **Keep** as a reproducibility control, outside aggregate qualification until explicitly selected. |
| `grid100_release07_cloud_named_v1` | Faithful released particle-cloud law, distinct from learned MoG. Archived 7k studies fail; no ordinary MoG credit follows. | **Keep** as a formulation comparison with the cloud label visible. |
| `rotated100_affine_square_named_v1` | Named rotated affine variant; no selected execution. | **Keep** as a scoped rotation control; an empty cell is coverage, not a numerical failure. |
| `staggered100_affine_square_named_v1` | Named staggered affine variant; no selected execution. | **Keep** as a scoped geometry control with the same explicit evidence status. |
| `img_bars4_residual16` | Changes the image architecture to the residual variant; no selected execution. | **Keep** as a distinct architecture transfer control; do not fill it with transpose-convolution receipts. |
| `img_blobs4_residual16` | Same architecture distinction for four blobs; no selected execution. | **Keep** as a distinct coverage/quality control. |
| `img_intensity2_residual16` | Same distinction for intensity fidelity; no selected execution. | **Keep** as a distinct architecture control with its own quality tolerance. |
| `img_stripes2_residual16` | Same distinction for stripes; the ordinary Stripes PASS does not measure it. | **Keep** as a simple positive-control candidate for the residual architecture. |
| `two_pole_800_schedule80_diagnostic_v1` | Extends training to 800 updates while retaining the original 80-step schedule horizon. | **Keep** as an explicitly scoped persistence diagnostic; it isolates continuation after the old schedule. |
| `two_pole_800_schedule800_diagnostic_v1` | Both training duration and schedule horizon become 800, so it asks a different question from the preceding variant. | **Keep** as the paired schedule-duration control, without counting either as an independent ordinary two-pole qualification. |
| `vector_anisotropic_published` | Published-fixture replay has its own host definition and no selected execution. | **Keep** for historical replay; require its exact prior, serving and source identity. |
| `vector_overlap_published` | Published overlapping-density replay; no selected execution. | **Keep** with observable-distribution grading and explicit replay scope. |
| `vector_spiral_published` | Published spiral replay; ordinary spiral passes cannot substitute for an unmeasured binding. | **Keep** for replay and distribution transfer. |
| `vector_two_broad_published` | Published broad-mixture replay; no selected execution. | **Keep** as a broad-component positive control under its own binding. |
| `vector_unequal_mass_published` | Published rare-mixture replay; no selected execution. The support/calibration issue must be assessed for its actual sampling law. | **Modify** its presentation to state the exact law and calibration status; preserve original grades and do not silently import a newer finite-atom exemption. |
| `vector_unequal_width_published` | Published unequal-width replay; no selected execution. Historical adapted Atlas width passes are another cohort. | **Keep** for replay; link those historical successes without filling this cell. |

## Recommended implementation order

1. **Repair the evidence view first.** Add per-task links from selected zeros to compatible diagnostics and historical context, retain source/runtime labels, and expose executed FAIL, prerequisite BLOCKED, incompatible, ERROR and RUNNING separately. Do not sum them into a single “nobody passes” count.
2. **Certify the existing BISPF endurance evidence.** Reconcile the grouped producer with the checkpoint-binding contract. Preserve the current blocked receipt and publish the saved-data recheck under its own identity. Consolidate hold and extension into one future counted task with visible subphases.
3. **Fix policy receipt production.** Address canonical serialization and missing prior-policy proof before additional training. Keep finite learned-state guards and the measured two-pole/Ring16 negatives.
4. **Expose temporal phases and dependency eligibility.** Gaussian and target-shift reports should explain whether failure is stationary drift, late recovery or post-recovery loss. Native 14k starts should clearly require passing own-state parents; a rescue study must be declared separately.
5. **Calibrate the rare-mode measurement before changing its science.** Use an oracle matching the declared continuous MoG and evaluation cadence, plus collapse and missing-mass controls. Keep the full existing gate until a prospective revision is justified. Other numerical failures remain useful research targets.

These recommendations do not launch another training study or select task-specific winners. The read-only analysis establishes which predicate failed and which evidence is missing; it does not establish the optimizer-level cause of each learned failure.

## Reproduction and verification

[`collect_evidence.py`](collect_evidence.py) derives all 18 required zero-pass IDs, all seven policy IDs and all 17 other indexed IDs directly from the snapshot. It preserves the eight selected cohort identities, task definitions, input hashes, exact backfill witnesses and blockers. The extraction checks byte-for-byte with:

```sh
python reports/forge/zero-pass-audit-20261015/collect_evidence.py --check
```

Recheck the existing ring observations with the project Python runtime and the retained local attempt directory:

```sh
python reports/forge/zero-pass-audit-20261015/collect_evidence.py --check \
  --ring-run /ml2/hypergan/attempt-fill-20261015/attempts/bf5d13ff3aeb4cba98d07373963839d8
```

The second command requires the saved artifacts; it verifies their recorded hashes and calls the existing numerical evaluator. The compact [recheck receipt](ring-extension-recheck.json) remains inspectable without those bulk files. Zero optimizer updates and zero sampling draws were added. Task declarations, canonical leaderboards, prior recipes and original receipts remain unchanged.
