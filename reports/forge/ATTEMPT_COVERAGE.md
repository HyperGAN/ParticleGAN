# Selected-family and non-seed research run-attempt coverage

Every record identifies an attempt to run an unchanged selected or archived research recipe and task. PASS/FAIL diagnostics remain separate from ordinary qualification. BLOCKED means an actual start was refused by its constructor or required checkpoint; it is not a numerical failure.

| Family or exact recipe | Attempted / distinct tests | Pending | Outcome counts |
| --- | ---: | ---: | --- |
| atlas | 60/60 | 0 | 29 FAIL, 31 PASS |
| bcap-pure | 60/60 | 0 | 34 FAIL, 26 PASS |
| bispf | 60/60 | 0 | 30 FAIL, 30 PASS |
| e22 | 60/60 | 0 | 34 FAIL, 26 PASS |
| empf | 62/62 | 0 | 4 BLOCKED, 15 ERROR, 16 FAIL, 27 PASS |
| k3p | 60/60 | 0 | 30 FAIL, 30 PASS |
| ka2 | 60/60 | 0 | 32 FAIL, 28 PASS |
| atlas selected-cloud after investigation | 3/3 | 0 | 2 ERROR, 1 FAIL |
| r1r2 | 60/60 | 0 | 33 FAIL, 27 PASS |
| release07-gan-v3 | 60/60 | 0 | 31 FAIL, 29 PASS |
| E22 checkpoint-export repair (same recipe) | 2/2 | 0 | 2 FAIL |
| ATLAS checkpoint-export repair (same recipe) | 2/2 | 0 | 2 FAIL |
| BISPF continuation attempts before prior-metadata fix | 3/3 | 0 | 1 BLOCKED, 2 ERROR |
| BISPF rebuilt checkpoints + matched-prior 14k diagnostics | 6/6 | 0 | 6 FAIL |
| signed-confidence-lr-gaussian-v1 | 29/29 | 0 | 2 BLOCKED, 13 FAIL, 13 PASS, 1 TIMEOUT |
| G-only coherent mean (original seed0) | 29/29 | 0 | 27 BLOCKED, 1 FAIL, 1 PASS |
| kinetic-local-dualnorm-g-history-u-v1 | 29/29 | 0 | 11 BLOCKED, 11 FAIL, 7 PASS |
| kinetic_transport_local_v2 | 29/29 | 0 | 2 BLOCKED, 13 FAIL, 13 PASS, 1 TIMEOUT |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | 29/29 | 0 | 11 BLOCKED, 8 FAIL, 8 PASS, 2 TIMEOUT |
| cell-quota-dualnorm-cross-v1 | 29/29 | 0 | 11 BLOCKED, 4 ERROR, 6 FAIL, 5 PASS, 3 TIMEOUT |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | 29/29 | 0 | 11 BLOCKED, 9 FAIL, 7 PASS, 2 TIMEOUT |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | 29/29 | 0 | 12 BLOCKED, 12 FAIL, 5 PASS |
| Fresh matched protected NS/transport control | 29/29 | 0 | 27 BLOCKED, 1 FAIL, 1 PASS |
| generator-idle-ema-bcap-direction-v1 | 29/29 | 0 | 3 BLOCKED, 12 FAIL, 2 INCOMPLETE, 12 PASS |
| kinetic-local-dualnorm-g-history-reference8-v1 | 29/29 | 0 | 12 BLOCKED, 8 FAIL, 6 PASS, 3 TIMEOUT |
| generator-confidence-lr-bcap-direction-v1 | 29/29 | 0 | 1 BLOCKED, 13 FAIL, 15 PASS |
| bcap-default-baseline-direction-v1 | 29/29 | 0 | 2 BLOCKED, 1 ERROR, 12 FAIL, 1 INCOMPLETE, 13 PASS |
| positive-evidence-lr-gaussian-v1 | 29/29 | 0 | 2 BLOCKED, 13 FAIL, 13 PASS, 1 TIMEOUT |
| confidence-lr-critic-orientation-gaussian-v1 | 29/29 | 0 | 20 BLOCKED, 7 FAIL, 2 PASS |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | 29/29 | 0 | 12 BLOCKED, 6 FAIL, 7 PASS, 4 TIMEOUT |
| confidence-lr-finite-cap-gaussian-v1 | 29/29 | 0 | 3 BLOCKED, 14 FAIL, 11 PASS, 1 TIMEOUT |
| kinetic-local-dualnorm-g-history-v1 | 29/29 | 0 | 2 BLOCKED, 15 FAIL, 11 PASS, 1 TIMEOUT |
| kinetic-local-nsgda-cross-v1 | 29/29 | 0 | 2 BLOCKED, 19 FAIL, 8 PASS |

[Exact per-task attempts](attempt-coverage.json) · [Main technique leaderboard](technique-inventory.md)

The runner reuses matched source/recipe/task receipts, resumes pending work, and never replays a completed attempt automatically. It uses at most four workers, two per GPU. Each child retains the task's original timeout. Raw logs and checkpoints are stored outside Git in the configured state directory.

```sh
python -m experiments.forge.attempt_coverage plan --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --include-research
python -m experiments.forge.attempt_coverage run --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --include-research --refresh
python -m experiments.forge.attempt_coverage status --state-root RUN_DIRECTORY
```

For a durable user service, adapt [the systemd template](../../ops/forge-attempt-coverage.service.example), then run `systemctl --user enable --now particlegan-attempt-fill.service`. Follow its starts and completions with `journalctl --user -u particlegan-attempt-fill.service -f` or `tail -f RUN_DIRECTORY/events.jsonl`. Each event names the family, task and attempt directory ID.

`--watch` rescans the local selected rows every 30 seconds after draining. `--refresh` regenerates the local main leaderboard during work and when the queue drains. An optional `--completion-hook ops/publish_attempt_coverage.py` publishes the completed batch through a generated-report-only commit and a checked GitHub PR on the dedicated branch. For multiple selected source cohorts, `RUN_DIRECTORY/source-roots.json` maps family IDs to their frozen checkouts; every source digest and candidate revision is checked. A changed source or recipe requires its matching frozen checkout; a mismatched source stops preparation. Interrupted attempts are retained for review, not replayed. The research intake includes the 19 seed-zero variants already shown in the leaderboard; historical seed-only repetitions are excluded. Original pinned requests supply each recipe and snapshot, and original source task declarations supply the missing questions. A mismatch with a frozen evaluator source blocks the attempt. Completed new same-source measurements fill previously empty research cells; archived verdicts and selected-family qualification never change. Checkpoint/data dependencies still require passing certified artifacts; gate-only failures are bypassed solely for this authorized diagnostic lane. No new qualification is granted.

<a name="remaining-attempts"></a>

## Follow-up attempts of remaining cells

These cover the three previously empty Atlas policy cells, three current BISPF continuation declarations, and one unchanged retry after a GPU-memory failure. Earlier receipts remain intact. The 324 missing pairs in historical seed-only studies are excluded from this backfill.

| Family / recipe | Test | Outcome | Updates / saved observations | Cause |
| --- | --- | --- | --- | --- |
| atlas selected-cloud after investigation | clockfree_audit | FAIL | 24 / — | step_label changed the update or common-prefix state |
| atlas selected-cloud after investigation | gaussian1d_acquisition | ERROR | 1000 / 24 | ValueError: Out of range float values are not JSON compliant: nan |
| atlas selected-cloud after investigation | ring16_acquisition | ERROR | 1600 / 96 | ValueError: Out of range float values are not JSON compliant: nan |
| kinetic-local-dualnorm-g-history-reference8-v1 | staggered100 | TIMEOUT | — / — | original task wall-time limit exhausted |

Export errors retain their saved numerical observations and actual-training GIFs in the local attempt directories. Those observations do not replace the recorded gate. The original BISPF attempts were blocked by failed parent gates; the checkpoint recovery section below rebuilds their states and runs explicitly nonqualifying continuations past those gates. Original timeouts and incompatible formulations are not changed to obtain a score.

<a name="checkpoint-recovery"></a>

## Checkpoint export repair and BISPF rebuilds

The checkpoint hasher now preserves diagnostic NaN/Inf values while retaining every finite-state hash. Atlas/E22 rerun their unchanged Gaussian and Ring16 questions with the export fix. BISPF rebuilds each original 7k producer and continues its exact own state to 14k. A failed parent quality gate does not stop these user-authorized diagnostics. Recipes, seeds, initialization, prior laws, numerical gates and original budgets remain fixed; original failures stay recorded and these repaired-source diagnostics grant no qualification.

| Family | Recovery run | Test | Outcome | Completed updates | Cause |
| --- | --- | --- | --- | ---: | --- |
| atlas | ATLAS checkpoint-export repair (same recipe) | gaussian1d_acquisition_tier1_policy_selected_cloud_v1 | FAIL | 1000 | recomputed complete live curve and terminal suffix |
| atlas | ATLAS checkpoint-export repair (same recipe) | ring16_acquisition_tier1_policy_selected_cloud_v1 | FAIL | 1600 | recomputed complete live curve and terminal suffix |
| bispf | BISPF rebuilt checkpoints + matched-prior 14k diagnostics | grid100 | FAIL | 7000 | coverage, sustained live accuracy, or independent holdout failed |
| bispf | BISPF rebuilt checkpoints + matched-prior 14k diagnostics | rotated100 | FAIL | 7000 | coverage, sustained live accuracy, or independent holdout failed |
| bispf | BISPF rebuilt checkpoints + matched-prior 14k diagnostics | staggered100 | FAIL | 7000 | coverage, sustained live accuracy, or independent holdout failed |
| bispf | BISPF rebuilt checkpoints + matched-prior 14k diagnostics | grid100_14k | FAIL | 14000 | coverage, sustained live accuracy, or independent holdout failed |
| bispf | BISPF continuation attempts before prior-metadata fix | grid100_14k | ERROR | — | ValueError: invalid context checkpoint schema |
| bispf | BISPF rebuilt checkpoints + matched-prior 14k diagnostics | rotated100_14k | FAIL | 14000 | coverage, sustained live accuracy, or independent holdout failed |
| bispf | BISPF continuation attempts before prior-metadata fix | rotated100_14k | ERROR | — | ValueError: invalid context checkpoint schema |
| bispf | BISPF rebuilt checkpoints + matched-prior 14k diagnostics | staggered100_14k | FAIL | 14000 | coverage, sustained live accuracy, or independent holdout failed |
| bispf | BISPF continuation attempts before prior-metadata fix | staggered100_14k | BLOCKED | 0 | checkpoint parent staggered100: unavailable |
| e22 | E22 checkpoint-export repair (same recipe) | gaussian1d_acquisition_tier1_policy_selected_cloud_v1 | FAIL | 1000 | recomputed complete live curve and terminal suffix |
| e22 | E22 checkpoint-export repair (same recipe) | ring16_acquisition_tier1_policy_selected_cloud_v1 | FAIL | 1600 | recomputed complete live curve and terminal suffix |

Continuation checks still require a complete own checkpoint, matching recipe and RNG identity, unchanged 7k scheduling horizon, exact prefix restoration and the full 14k update count. Ordinary qualification rejects the explicitly marked diagnostic continuation receipts.

## Results recovered from existing uninterrupted runs

These task results reuse their producer's already measured observations. No new training or sampling was performed. Original refused-start receipts remain linked in each certificate; ordinary qualification is unchanged.

| Family | Test | Result | Passing extension checks | Evidence |
| --- | --- | --- | ---: | --- |
| empf | ring_extension | PASS | 300/300 | [Verified saved run](grouped-evidence/fb39a9510f02d01ecb7e3b5c08dfaa6d47ec34737e1c7ceecc9af3b52224490d.json) |

## Attempted blockers and incomplete runs

These are recorded start attempts, not missing runs. A source or formulation change is not applied to make them pass.

| Recipe | Task | Outcome | Reason |
| --- | --- | --- | --- |
| BISPF continuation attempts before prior-metadata fix | grid100_14k | ERROR | ValueError: invalid context checkpoint schema |
| BISPF continuation attempts before prior-metadata fix | rotated100_14k | ERROR | ValueError: invalid context checkpoint schema |
| BISPF continuation attempts before prior-metadata fix | staggered100_14k | BLOCKED | checkpoint parent staggered100: unavailable |
| Fresh matched protected NS/transport control | ae_gan_hold | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | cover_leftover | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| Fresh matched protected NS/transport control | five_word_joint_smoke | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | grid100 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | img_bars4 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | img_blobs4 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | img_intensity2 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | img_stripes2 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | mid_scale_identity | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | mode_hold | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | residual_student | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | ring16_acquisition | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | ring_extension | BLOCKED | checkpoint parent ring_hold: BLOCKED |
| Fresh matched protected NS/transport control | ring_hold | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | rotated100 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | staggered100 | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | trajectory | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | two_pole | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | unipolar | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | unused_token_hold | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | vector_anisotropic | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | vector_overlap | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | vector_spiral | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | vector_two_broad | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | vector_unequal_mass | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| Fresh matched protected NS/transport control | vector_unequal_width | BLOCKED | augmented transport-primary protection has no certified hook for this component/non-Gaussian host |
| G-only coherent mean (original seed0) | ae_gan_hold | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | cover_leftover | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| G-only coherent mean (original seed0) | five_word_joint_smoke | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | grid100 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | img_bars4 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | img_blobs4 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | img_intensity2 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | img_stripes2 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | mid_scale_identity | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | mode_hold | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | residual_student | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | ring16_acquisition | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | ring_extension | BLOCKED | checkpoint parent ring_hold: BLOCKED |
| G-only coherent mean (original seed0) | ring_hold | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | rotated100 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | staggered100 | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | trajectory | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | two_pole | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | unipolar | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | unused_token_hold | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | vector_anisotropic | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | vector_overlap | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | vector_spiral | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | vector_two_broad | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | vector_unequal_mass | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| G-only coherent mean (original seed0) | vector_unequal_width | BLOCKED | G-gradient response is declared only for the original ordinary Gaussian host |
| atlas selected-cloud after investigation | gaussian1d_acquisition_tier1_policy_selected_cloud_v1 | ERROR | ValueError: Out of range float values are not JSON compliant: nan |
| atlas selected-cloud after investigation | ring16_acquisition_tier1_policy_selected_cloud_v1 | ERROR | ValueError: Out of range float values are not JSON compliant: nan |
| bcap-default-baseline-direction-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: INCOMPLETE |
| bcap-default-baseline-direction-v1 | gaussian1d_stability | ERROR | ValueError: named-stream checkpoint identity mismatch |
| bcap-default-baseline-direction-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| cell-quota-dualnorm-cross-v1 | ae_gan_hold | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | cover_leftover | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| cell-quota-dualnorm-cross-v1 | five_word_joint_smoke | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | grid100 | TIMEOUT | original task wall-time limit exhausted |
| cell-quota-dualnorm-cross-v1 | img_bars4 | ERROR | ValueError: Positive-width unstandardized MoG required |
| cell-quota-dualnorm-cross-v1 | img_blobs4 | ERROR | ValueError: Positive-width unstandardized MoG required |
| cell-quota-dualnorm-cross-v1 | img_intensity2 | ERROR | ValueError: Positive-width unstandardized MoG required |
| cell-quota-dualnorm-cross-v1 | img_stripes2 | ERROR | ValueError: Positive-width unstandardized MoG required |
| cell-quota-dualnorm-cross-v1 | mid_scale_identity | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | residual_student | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| cell-quota-dualnorm-cross-v1 | rotated100 | TIMEOUT | original task wall-time limit exhausted |
| cell-quota-dualnorm-cross-v1 | staggered100 | TIMEOUT | original task wall-time limit exhausted |
| cell-quota-dualnorm-cross-v1 | trajectory | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | two_pole | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | unipolar | BLOCKED | Cell objective requires the public GANTrainer path |
| cell-quota-dualnorm-cross-v1 | unused_token_hold | BLOCKED | Cell objective requires the public GANTrainer path |
| confidence-lr-critic-orientation-gaussian-v1 | ae_gan_hold | BLOCKED | ae_gan_hold: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | cover_leftover | BLOCKED | cover_leftover: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| confidence-lr-critic-orientation-gaussian-v1 | five_word_joint_smoke | BLOCKED | five_word_joint_smoke: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | grid100 | BLOCKED | grid100: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | img_bars4 | BLOCKED | img_bars4: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | img_blobs4 | BLOCKED | img_blobs4: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | img_intensity2 | BLOCKED | img_intensity2: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | img_stripes2 | BLOCKED | img_stripes2: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | mid_scale_identity | BLOCKED | mid_scale_identity: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | mode_hold | BLOCKED | mode_hold: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | residual_student | BLOCKED | residual_student: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: BLOCKED |
| confidence-lr-critic-orientation-gaussian-v1 | ring_hold | BLOCKED | ring_hold: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | rotated100 | BLOCKED | rotated100: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | staggered100 | BLOCKED | staggered100: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | trajectory | BLOCKED | trajectory: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | two_pole | BLOCKED | two_pole: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | unipolar | BLOCKED | unipolar: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-critic-orientation-gaussian-v1 | unused_token_hold | BLOCKED | unused_token_hold: critic orientation has only a public scalar GANTrainer consumer |
| confidence-lr-finite-cap-gaussian-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: TIMEOUT |
| confidence-lr-finite-cap-gaussian-v1 | five_word_joint_smoke | TIMEOUT | original task wall-time limit exhausted |
| confidence-lr-finite-cap-gaussian-v1 | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: FAIL |
| confidence-lr-finite-cap-gaussian-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| empf | alice_spiral_bigan | ERROR | ValueError: unsupported vector target declaration |
| empf | cgan_overlapping_classes | ERROR | ValueError: unsupported vector target declaration |
| empf | cramer_energy_overlap | ERROR | ValueError: unsupported vector target declaration |
| empf | cyclegan_unpaired_constellation | ERROR | ValueError: unsupported vector target declaration |
| empf | grid100_14k | BLOCKED | checkpoint parent grid100: FAIL |
| empf | grid100_mlp16 | ERROR | ValueError: native model variants require an explicit profile |
| empf | grid100_res16_scaled_tanh | ERROR | ValueError: native model variants require an explicit profile |
| empf | grid100_res16_tanh | ERROR | ValueError: native model variants require an explicit profile |
| empf | infogan_anisotropic_modes | ERROR | ValueError: unsupported vector target declaration |
| empf | misgan_block_neural_imputer | ERROR | ValueError: unsupported vector target declaration |
| empf | misgan_block_recovery | ERROR | ValueError: unsupported vector target declaration |
| empf | misgan_mcar_p50_recovery | ERROR | ValueError: unsupported vector target declaration |
| empf | misgan_mcar_p80_anchored | ERROR | ValueError: unsupported vector target declaration |
| empf | pacgan8_grid100 | ERROR | ValueError: unsupported vector target declaration |
| empf | pacgan_unequal_mass | ERROR | ValueError: unsupported vector target declaration |
| empf | ring16_bigan_continuous | ERROR | ValueError: unsupported vector target declaration |
| empf | rotated100_14k | BLOCKED | checkpoint parent rotated100: FAIL |
| empf | staggered100_14k | BLOCKED | checkpoint parent staggered100: FAIL |
| empf | zero_tanh_rail_escape | BLOCKED | zero_tanh_rail_escape: unsupported tanh host zero_tanh_rail_escape; only zero_tanh_smoke is routed |
| generator-confidence-lr-bcap-direction-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| generator-idle-ema-bcap-direction-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: INCOMPLETE |
| generator-idle-ema-bcap-direction-v1 | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: INCOMPLETE |
| generator-idle-ema-bcap-direction-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | ae_gan_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | cover_leftover | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | five_word_joint_smoke | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | mid_scale_identity | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | residual_student | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | rotated100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | staggered100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | trajectory | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | two_pole | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | unipolar | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | unused_token_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | ae_gan_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | cover_leftover | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | five_word_joint_smoke | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | mid_scale_identity | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | residual_student | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | rotated100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | staggered100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | trajectory | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | two_pole | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | unipolar | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | unused_token_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | ae_gan_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | cover_leftover | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | five_word_joint_smoke | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: FAIL |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | mid_scale_identity | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | residual_student | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | trajectory | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | two_pole | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | unipolar | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | unused_token_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | ae_gan_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | cover_leftover | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | five_word_joint_smoke | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: FAIL |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | grid100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | mid_scale_identity | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | residual_student | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | ring16_acquisition | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | rotated100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | staggered100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | trajectory | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | two_pole | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | unipolar | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | unused_token_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | ae_gan_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | cover_leftover | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| kinetic-local-dualnorm-g-history-reference8-v1 | five_word_joint_smoke | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: FAIL |
| kinetic-local-dualnorm-g-history-reference8-v1 | grid100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-v1 | mid_scale_identity | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | residual_student | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-reference8-v1 | rotated100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-v1 | staggered100 | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-reference8-v1 | trajectory | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | two_pole | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | unipolar | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-reference8-v1 | unused_token_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | ae_gan_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | cover_leftover | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| kinetic-local-dualnorm-g-history-u-v1 | five_word_joint_smoke | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | mid_scale_identity | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | residual_student | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-dualnorm-g-history-u-v1 | trajectory | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | two_pole | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | unipolar | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-u-v1 | unused_token_hold | BLOCKED | Original task source binding unavailable; do not port the formulation: particlegan/kinetic_transport.py |
| kinetic-local-dualnorm-g-history-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: TIMEOUT |
| kinetic-local-dualnorm-g-history-v1 | five_word_joint_smoke | TIMEOUT | original task wall-time limit exhausted |
| kinetic-local-dualnorm-g-history-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic-local-nsgda-cross-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: FAIL |
| kinetic-local-nsgda-cross-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| kinetic_transport_local_v2 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: TIMEOUT |
| kinetic_transport_local_v2 | five_word_joint_smoke | TIMEOUT | original task wall-time limit exhausted |
| kinetic_transport_local_v2 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| positive-evidence-lr-gaussian-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: TIMEOUT |
| positive-evidence-lr-gaussian-v1 | five_word_joint_smoke | TIMEOUT | original task wall-time limit exhausted |
| positive-evidence-lr-gaussian-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| signed-confidence-lr-gaussian-v1 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: TIMEOUT |
| signed-confidence-lr-gaussian-v1 | five_word_joint_smoke | TIMEOUT | original task wall-time limit exhausted |
| signed-confidence-lr-gaussian-v1 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
